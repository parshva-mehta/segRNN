import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def normalize_tag(tag: str) -> str:
    t = str(tag).strip().lower()
    for suffix in ("_epoch", "_step"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
    return t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/validation metrics from latest station pipeline training run."
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("station_pipeline/logs/segrnn_experiment"),
        help="Root directory containing version_* run folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("station_pipeline/artifacts/training_validation_metrics.png"),
        help="Output image path for generated plots",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional specific run directory. If omitted, latest run is used.",
    )
    return parser.parse_args()


def has_metrics_source(run_dir: Path) -> bool:
    if (run_dir / "metrics.csv").exists():
        return True
    if list(run_dir.glob("events.out.tfevents.*")):
        return True
    return False


def find_latest_run_dir(log_root: Path) -> Path:
    if not log_root.exists():
        raise FileNotFoundError(f"Log root not found: {log_root}")

    version_dirs = [p for p in log_root.glob("version_*") if p.is_dir()]
    if not version_dirs:
        raise ValueError(
            f"No version_* directories found in: {log_root}. "
            "Run training first to generate logs."
        )

    candidates = [p for p in version_dirs if has_metrics_source(p)]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    newest = max(version_dirs, key=lambda p: p.stat().st_mtime)
    raise ValueError(
        f"Found run directories but no metric files in {log_root}. "
        f"Latest run is {newest}, which contains no metrics.csv or event file."
    )


def load_scalars_from_metrics_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame(columns=["tag", "step", "value", "epoch"])

    if "step" not in df.columns:
        df["step"] = range(len(df))

    metric_cols = [c for c in df.columns if c not in {"epoch", "step"}]
    rows: list[dict] = []
    for _, r in df.iterrows():
        for metric in metric_cols:
            value = pd.to_numeric(r.get(metric), errors="coerce")
            if pd.notna(value):
                rows.append(
                    {
                        "tag": str(metric),
                        "step": float(r.get("step", 0)),
                        "value": float(value),
                        "epoch": pd.to_numeric(r.get("epoch"), errors="coerce"),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["tag", "step", "value", "epoch"])
    out = out.sort_values("step").reset_index(drop=True)
    return out


def load_scalars_from_event_file(event_file: Path) -> pd.DataFrame:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise ImportError(
            "tensorboard is required to read event files. "
            "Install it with: pip install tensorboard"
        ) from exc

    event_acc = EventAccumulator(str(event_file))
    event_acc.Reload()

    tags = event_acc.Tags().get("scalars", [])
    rows: list[dict] = []
    for tag in tags:
        for event in event_acc.Scalars(tag):
            rows.append(
                {
                    "tag": tag,
                    "step": float(event.step),
                    "value": float(event.value),
                    "epoch": None,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["tag", "step", "value", "epoch"])
    out = out.sort_values("step").reset_index(drop=True)
    return out


def load_scalars_from_event_files(event_files: list[Path]) -> pd.DataFrame:
    frames = []
    for event_file in sorted(event_files, key=lambda p: p.stat().st_mtime):
        try:
            frame = load_scalars_from_event_file(event_file)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            print(f"Warning: failed to read {event_file.name}: {exc}")

    if not frames:
        return pd.DataFrame(columns=["tag", "step", "value", "epoch"])

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("step").reset_index(drop=True)
    return out


def load_scalars_from_run_dir(run_dir: Path) -> pd.DataFrame:
    metrics_csv = run_dir / "metrics.csv"
    if metrics_csv.exists():
        return load_scalars_from_metrics_csv(metrics_csv)

    event_files = [p for p in run_dir.glob("events.out.tfevents.*")]
    if event_files:
        return load_scalars_from_event_files(event_files)

    raise ValueError(
        f"No metrics source found in {run_dir}. "
        "Expected metrics.csv or events.out.tfevents.*"
    )


def build_epoch_mapping(df: pd.DataFrame) -> pd.DataFrame | None:
    if "epoch" in df.columns:
        explicit_epoch = df[["step", "epoch"]].dropna().copy()
        if not explicit_epoch.empty:
            explicit_epoch["epoch"] = explicit_epoch["epoch"].round().astype(int)
            return (
                explicit_epoch.sort_values("step")
                .drop_duplicates(subset=["step"], keep="last")
                .reset_index(drop=True)
            )

    epoch_rows = df[df["tag"].str.lower() == "epoch"][["step", "value"]].copy()
    if epoch_rows.empty:
        return None
    epoch_rows = epoch_rows.rename(columns={"value": "epoch"})
    epoch_rows["epoch"] = epoch_rows["epoch"].round().astype(int)
    return epoch_rows.sort_values("step").drop_duplicates(subset=["step"], keep="last")


def get_metric_series(
    df: pd.DataFrame, tag_aliases: list[str], epoch_map: pd.DataFrame | None
) -> pd.Series | None:
    wanted = {normalize_tag(t) for t in tag_aliases}
    working_df = df.copy()
    working_df["normalized_tag"] = working_df["tag"].map(normalize_tag)
    metric_df = working_df[working_df["normalized_tag"].isin(wanted)][["step", "value"]].copy()

    if metric_df.empty:
        return None

    metric_df = metric_df.sort_values("step")
    if epoch_map is not None and not epoch_map.empty:
        metric_df = pd.merge_asof(
            metric_df,
            epoch_map.sort_values("step"),
            on="step",
            direction="nearest",
        )
        metric_df = metric_df.dropna(subset=["epoch"])
        if metric_df.empty:
            return None
        metric_df["epoch"] = metric_df["epoch"].astype(int)
        return metric_df.groupby("epoch")["value"].mean().sort_index()

    return metric_df.groupby("step")["value"].mean().sort_index()


def plot_metric_triplet(
    ax: plt.Axes,
    df: pd.DataFrame,
    epoch_map: pd.DataFrame | None,
    title: str,
    train_aliases: list[str],
    val_aliases: list[str],
    test_aliases: list[str],
) -> None:
    train_series = get_metric_series(df, train_aliases, epoch_map)
    val_series = get_metric_series(df, val_aliases, epoch_map)
    test_series = get_metric_series(df, test_aliases, epoch_map)

    if train_series is None and val_series is None and test_series is None:
        ax.text(
            0.5,
            0.5,
            f"No data found for {title.lower()}",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        return

    if train_series is not None:
        ax.plot(train_series.index, train_series.values, marker="o", label="Train")
    if val_series is not None:
        ax.plot(val_series.index, val_series.values, marker="s", label="Validation")
    if test_series is not None:
        ax.plot(test_series.index, test_series.values, marker="^", label="Test")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir is not None else find_latest_run_dir(args.log_root)
    df = load_scalars_from_run_dir(run_dir)
    epoch_map = build_epoch_mapping(df)

    output_parent = args.output.parent
    output_parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    plot_metric_triplet(
        axes[0],
        df,
        epoch_map,
        "Loss",
        train_aliases=["train_mse_loss", "train_loss", "loss/train", "train/loss"],
        val_aliases=["val_loss", "validation_loss", "loss/val", "val/loss"],
        test_aliases=["test_loss", "loss/test", "test/loss"],
    )
    plot_metric_triplet(
        axes[1],
        df,
        epoch_map,
        "Accuracy",
        train_aliases=["train_accuracy", "train_acc", "accuracy/train", "train/accuracy"],
        val_aliases=["val_accuracy", "val_acc", "accuracy/val", "val/accuracy"],
        test_aliases=["test_accuracy", "test_acc", "accuracy/test", "test/accuracy"],
    )
    plot_metric_triplet(
        axes[2],
        df,
        epoch_map,
        "F1 Score",
        train_aliases=["train_f1", "f1/train", "train/f1"],
        val_aliases=["val_f1", "f1/val", "val/f1"],
        test_aliases=["test_f1", "f1/test", "test/f1"],
    )
    plot_metric_triplet(
        axes[3],
        df,
        epoch_map,
        "Recall",
        train_aliases=["train_recall", "recall/train", "train/recall"],
        val_aliases=["val_recall", "recall/val", "val/recall"],
        test_aliases=["test_recall", "recall/test", "test/recall"],
    )

    fig.suptitle("Training and Validation Metrics per Epoch", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(f"Using run directory: {run_dir}")
    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()

