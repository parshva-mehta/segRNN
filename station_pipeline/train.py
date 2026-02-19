from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from config import ARTIFACTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, PROCESSED_DATA_DIR
from model import SegRNNModel


class WeatherDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _directional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pred_bin = (y_pred > 0).astype(np.int32)
    true_bin = (y_true > 0).astype(np.int32)
    accuracy = float((pred_bin == true_bin).mean())

    tp = float(((pred_bin == 1) & (true_bin == 1)).sum())
    fp = float(((pred_bin == 1) & (true_bin == 0)).sum())
    fn = float(((pred_bin == 0) & (true_bin == 1)).sum())
    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "test_accuracy": accuracy,
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
    }


def _build_predictions_df(
    y_true: np.ndarray, y_pred: np.ndarray, feature_cols: list[str], metadata_path: Path
) -> pd.DataFrame:
    pred_df = pd.DataFrame()
    if metadata_path.exists():
        meta = pd.read_csv(metadata_path)
        pred_df["station"] = meta.get("station")
        pred_df["target_valid"] = meta.get("target_valid")

    for i, feature in enumerate(feature_cols):
        pred_df[f"actual_{feature}"] = y_true[:, i]
        pred_df[f"pred_{feature}"] = y_pred[:, i]
    return pred_df


def _load_feature_columns(processed_dir: Path) -> list[str]:
    feature_path = processed_dir / "feature_columns.json"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_path}")
    return json.loads(feature_path.read_text(encoding="utf-8"))


def _split_train_val(x_train: np.ndarray, y_train: np.ndarray, val_fraction: float):
    split_idx = int(len(x_train) * (1.0 - val_fraction))
    split_idx = max(1, min(split_idx, len(x_train) - 1))
    return (
        x_train[:split_idx],
        y_train[:split_idx],
        x_train[split_idx:],
        y_train[split_idx:],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SegRNN on processed weather sequences.")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    parser.add_argument("--logs-dir", default=str(LOGS_DIR))
    parser.add_argument("--checkpoints-dir", default=str(CHECKPOINTS_DIR))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--segment-length", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    artifacts_dir = Path(args.artifacts_dir)
    logs_dir = Path(args.logs_dir)
    checkpoints_dir = Path(args.checkpoints_dir)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    x_train = np.load(processed_dir / "X_train.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    x_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    feature_cols = _load_feature_columns(processed_dir)

    if y_train.ndim != 2 or y_test.ndim != 2:
        raise ValueError("This trainer currently expects n_steps_out=1 (2D targets).")
    if x_train.ndim != 3:
        raise ValueError("Input sequences must be 3D arrays: (N, seq_len, num_features).")
    if len(x_train) < 2:
        raise ValueError("Need at least 2 training samples for train/validation split.")

    x_tr, y_tr, x_val, y_val = _split_train_val(x_train, y_train, args.val_fraction)

    train_loader = DataLoader(WeatherDataset(x_tr, y_tr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(WeatherDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(WeatherDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = SegRNNModel(
        input_size=x_train.shape[2],
        hidden_size=args.hidden_size,
        output_size=y_train.shape[1],
        segment_length=args.segment_length,
        learning_rate=args.lr,
    )

    logger = TensorBoardLogger(save_dir=str(logs_dir), name="segrnn_experiment")
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="segrnn-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb],
    )
    trainer.fit(model, train_loader, val_loader)

    test_results = trainer.test(ckpt_path="best", dataloaders=test_loader)
    best_checkpoint = checkpoint_cb.best_model_path
    if not best_checkpoint:
        raise RuntimeError("Training finished without producing a best checkpoint.")

    best_model = SegRNNModel.load_from_checkpoint(best_checkpoint)
    best_model.eval()
    best_model.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)

    test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        predictions = best_model(test_tensor).detach().cpu().numpy()

    predictions_df = _build_predictions_df(
        y_true=y_test,
        y_pred=predictions,
        feature_cols=feature_cols,
        metadata_path=processed_dir / "test_targets.csv",
    )
    predictions_path = artifacts_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    metrics = _directional_metrics(y_test, predictions)
    metrics["test_mse"] = float(np.mean((predictions - y_test) ** 2))
    metrics["test_mae"] = float(np.mean(np.abs(predictions - y_test)))
    metrics["best_checkpoint"] = best_checkpoint
    metrics["created_at_utc"] = datetime.now(UTC).isoformat()
    if test_results:
        metrics["lightning_test"] = test_results[0]

    metrics_path = artifacts_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Predictions saved: {predictions_path}")
    print(f"Metrics saved: {metrics_path}")
    print(
        "Final test metrics -> "
        f"accuracy={metrics['test_accuracy']:.4f}, "
        f"mse={metrics['test_mse']:.4f}, "
        f"mae={metrics['test_mae']:.4f}"
    )


if __name__ == "__main__":
    main()

