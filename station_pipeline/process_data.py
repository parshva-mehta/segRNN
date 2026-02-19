from __future__ import annotations

import argparse
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    CONTINUOUS_FEATURES,
    DEFAULT_N_STEPS_IN,
    DEFAULT_N_STEPS_OUT,
    DEFAULT_TEST_SPLIT,
    PLACEHOLDERS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


def _latest_raw_csv() -> Path:
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"No raw CSV files found in {RAW_DATA_DIR}")
    return csv_files[0]


def _preprocess_dataframe(df: pd.DataFrame, normalize: bool) -> tuple[pd.DataFrame, StandardScaler | None]:
    expected_cols = {"station", "valid", *CONTINUOUS_FEATURES}
    missing_cols = sorted(expected_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Input CSV missing required columns: {missing_cols}")

    df = df.copy()
    df["valid"] = pd.to_datetime(df["valid"], errors="coerce")
    df = df.dropna(subset=["valid"]).sort_values(["station", "valid"]).reset_index(drop=True)

    df[CONTINUOUS_FEATURES] = df[CONTINUOUS_FEATURES].replace(PLACEHOLDERS, np.nan).astype(str)
    for col in CONTINUOUS_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[CONTINUOUS_FEATURES] = df.groupby("station")[CONTINUOUS_FEATURES].transform(
        lambda group: group.interpolate(method="linear")
    )
    df[CONTINUOUS_FEATURES] = df.groupby("station")[CONTINUOUS_FEATURES].transform(
        lambda group: group.ffill().bfill()
    )

    df = df.dropna(subset=CONTINUOUS_FEATURES).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows remain after preprocessing and NaN handling.")

    scaler = None
    if normalize:
        scaler = StandardScaler()
        df[CONTINUOUS_FEATURES] = scaler.fit_transform(df[CONTINUOUS_FEATURES])

    return df, scaler


def _create_sequences(
    df: pd.DataFrame, n_steps_in: int, n_steps_out: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    rows: list[dict[str, str]] = []

    for station, station_df in df.groupby("station"):
        station_df = station_df.reset_index(drop=True)
        values = station_df[CONTINUOUS_FEATURES].to_numpy(dtype=np.float32)
        valid_times = station_df["valid"].to_numpy()

        max_start = len(station_df) - n_steps_in - n_steps_out + 1
        if max_start <= 0:
            continue

        for i in range(max_start):
            x_window = values[i : i + n_steps_in, :]
            y_window = values[i + n_steps_in : i + n_steps_in + n_steps_out, :]
            target_idx = i + n_steps_in + n_steps_out - 1

            x_list.append(x_window)
            y_list.append(y_window)
            rows.append(
                {
                    "station": str(station),
                    "target_valid": pd.Timestamp(valid_times[target_idx]).isoformat(),
                }
            )

    if not x_list:
        raise ValueError("No sequences were created. Try a wider date range for the selected station.")

    x = np.stack(x_list).astype(np.float32)
    y = np.stack(y_list).astype(np.float32)
    if n_steps_out == 1:
        y = np.squeeze(y, axis=1)

    meta_df = pd.DataFrame(rows)
    return x, y, meta_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process downloaded weather CSV into model-ready sequences."
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Raw CSV path. Defaults to most recent file in data/raw.",
    )
    parser.add_argument("--n-steps-in", type=int, default=DEFAULT_N_STEPS_IN)
    parser.add_argument("--n-steps-out", type=int, default=DEFAULT_N_STEPS_OUT)
    parser.add_argument("--test-split", type=float, default=DEFAULT_TEST_SPLIT)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable StandardScaler normalization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_csv) if args.input_csv else _latest_raw_csv()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    if not (0.0 < args.test_split < 1.0):
        raise ValueError("--test-split must be between 0 and 1.")

    raw_df = pd.read_csv(input_path, low_memory=False)
    processed_df, scaler = _preprocess_dataframe(raw_df, normalize=not args.no_normalize)
    x, y, sequence_meta = _create_sequences(processed_df, args.n_steps_in, args.n_steps_out)

    split_idx = int(len(x) * (1.0 - args.test_split))
    if split_idx <= 0 or split_idx >= len(x):
        raise ValueError("Invalid split; adjust --test-split or provide more data.")

    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    train_meta = sequence_meta.iloc[:split_idx].reset_index(drop=True)
    test_meta = sequence_meta.iloc[split_idx:].reset_index(drop=True)

    stem = input_path.stem
    processed_csv_path = PROCESSED_DATA_DIR / f"{stem}_processed.csv"
    processed_df.to_csv(processed_csv_path, index=False)

    np.save(PROCESSED_DATA_DIR / "X_train.npy", x_train)
    np.save(PROCESSED_DATA_DIR / "X_test.npy", x_test)
    np.save(PROCESSED_DATA_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DATA_DIR / "y_test.npy", y_test)
    train_meta.to_csv(PROCESSED_DATA_DIR / "train_targets.csv", index=False)
    test_meta.to_csv(PROCESSED_DATA_DIR / "test_targets.csv", index=False)

    with (PROCESSED_DATA_DIR / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(CONTINUOUS_FEATURES, f, indent=2)

    if scaler is not None:
        with (PROCESSED_DATA_DIR / "scaler.pkl").open("wb") as f:
            pickle.dump(scaler, f)

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input_csv": str(input_path),
        "processed_csv": str(processed_csv_path),
        "normalize": not args.no_normalize,
        "n_steps_in": args.n_steps_in,
        "n_steps_out": args.n_steps_out,
        "test_split": args.test_split,
        "num_sequences": int(len(x)),
        "train_sequences": int(len(x_train)),
        "test_sequences": int(len(x_test)),
    }
    with (PROCESSED_DATA_DIR / "processing_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Input CSV: {input_path}")
    print(f"Processed CSV: {processed_csv_path}")
    print(f"Sequences created: {len(x)}")
    print(f"Train/Test split: {len(x_train)}/{len(x_test)}")
    print(f"Processed outputs directory: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()

