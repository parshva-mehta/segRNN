from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import ARTIFACTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize station pipeline metrics from predictions and metrics.json."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Directory containing metrics.json and predictions.csv",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="Optional override path for metrics JSON",
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=None,
        help="Optional override path for predictions CSV",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path for the combined summary report",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for per-feature metrics",
    )
    return parser.parse_args()


def _load_base_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {metrics_path}")
    return payload


def _feature_columns(df: pd.DataFrame) -> list[str]:
    actual_cols = [c for c in df.columns if c.startswith("actual_")]
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    pred_features = {c.replace("pred_", "", 1) for c in pred_cols}
    return sorted(c.replace("actual_", "", 1) for c in actual_cols if c.replace("actual_", "", 1) in pred_features)


def _per_feature_metrics(predictions: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for feature in _feature_columns(predictions):
        y_true = predictions[f"actual_{feature}"].to_numpy(dtype=float)
        y_pred = predictions[f"pred_{feature}"].to_numpy(dtype=float)
        err = y_pred - y_true
        rows.append(
            {
                "feature": feature,
                "mse": float(np.mean(err**2)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "bias": float(np.mean(err)),
            }
        )
    return sorted(rows, key=lambda r: r["mae"])


def _directional_accuracy(predictions: pd.DataFrame) -> float:
    features = _feature_columns(predictions)
    if not features:
        return float("nan")
    actual = predictions[[f"actual_{f}" for f in features]].to_numpy(dtype=float)
    pred = predictions[[f"pred_{f}" for f in features]].to_numpy(dtype=float)
    return float(((pred > 0) == (actual > 0)).mean())


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    metrics_path = args.metrics_file or (artifacts_dir / "metrics.json")
    predictions_path = args.predictions_file or (artifacts_dir / "predictions.csv")
    output_json = args.output_json or (artifacts_dir / "metrics_report.json")
    output_csv = args.output_csv or (artifacts_dir / "per_feature_metrics.csv")

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {predictions_path}. Run train.py first."
        )

    base_metrics = _load_base_metrics(metrics_path)
    predictions = pd.read_csv(predictions_path)
    if predictions.empty:
        raise ValueError(f"Predictions file is empty: {predictions_path}")

    feature_rows = _per_feature_metrics(predictions)
    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(output_csv, index=False)

    report = {
        "num_prediction_rows": int(len(predictions)),
        "num_features": int(len(feature_rows)),
        "overall_directional_accuracy_from_predictions": _directional_accuracy(predictions),
        "global_mse_from_predictions": float(
            np.mean(
                (
                    predictions[[c for c in predictions.columns if c.startswith("pred_")]].to_numpy(dtype=float)
                    - predictions[[c for c in predictions.columns if c.startswith("actual_")]].to_numpy(dtype=float)
                )
                ** 2
            )
        ),
        "global_mae_from_predictions": float(
            np.mean(
                np.abs(
                    predictions[[c for c in predictions.columns if c.startswith("pred_")]].to_numpy(dtype=float)
                    - predictions[[c for c in predictions.columns if c.startswith("actual_")]].to_numpy(dtype=float)
                )
            )
        ),
        "base_metrics_json": base_metrics,
        "best_3_features_by_mae": feature_rows[:3],
        "worst_3_features_by_mae": feature_rows[-3:],
    }

    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Loaded predictions: {predictions_path}")
    print(f"Loaded base metrics: {metrics_path if metrics_path.exists() else 'not found'}")
    print(f"Saved per-feature metrics: {output_csv}")
    print(f"Saved combined report: {output_json}")
    print(
        "Global metrics -> "
        f"directional_accuracy={report['overall_directional_accuracy_from_predictions']:.4f}, "
        f"mse={report['global_mse_from_predictions']:.4f}, "
        f"mae={report['global_mae_from_predictions']:.4f}"
    )


if __name__ == "__main__":
    main()

