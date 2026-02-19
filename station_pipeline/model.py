from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn


class SegRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, segment_length: int):
        super().__init__()
        self.segment_length = segment_length
        self.gru_segment = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru_aggregate = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        num_segments = seq_len // self.segment_length
        x = x[:, : num_segments * self.segment_length, :]
        x = x.view(batch_size * num_segments, self.segment_length, -1)

        _, h_n_segment = self.gru_segment(x)
        h_n_segment = h_n_segment.squeeze(0).view(batch_size, num_segments, -1)

        _, h_n_aggregate = self.gru_aggregate(h_n_segment)
        h_n_aggregate = h_n_aggregate.squeeze(0)
        return self.fc(h_n_aggregate)


class SegRNNModel(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        segment_length: int,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SegRNN(input_size, hidden_size, output_size, segment_length)
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _classification_metrics(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_binary = (outputs > 0).int()
        target_binary = (targets > 0).int()

        accuracy = (pred_binary == target_binary).float().mean()
        tp = ((pred_binary == 1) & (target_binary == 1)).sum().float()
        fp = ((pred_binary == 1) & (target_binary == 0)).sum().float()
        fn = ((pred_binary == 0) & (target_binary == 1)).sum().float()
        eps = 1e-8

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return accuracy, precision, recall, f1

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        inputs, targets = batch
        outputs = self(inputs)
        mse_loss = self.mse(outputs, targets)
        mae_loss = self.mae(outputs, targets)
        accuracy, precision, recall, f1 = self._classification_metrics(outputs, targets)
        return {
            "mse_loss": mse_loss,
            "mae_loss": mae_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        metrics = self._shared_step(batch)
        self.log("train_mse_loss", metrics["mse_loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_mae_loss", metrics["mae_loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_accuracy", metrics["accuracy"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_precision", metrics["precision"], on_step=False, on_epoch=True)
        self.log("train_recall", metrics["recall"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_f1", metrics["f1"], prog_bar=True, on_step=False, on_epoch=True)
        return metrics["mse_loss"]

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        metrics = self._shared_step(batch)
        self.log("val_loss", metrics["mse_loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae_loss", metrics["mae_loss"], on_step=False, on_epoch=True)
        self.log("val_accuracy", metrics["accuracy"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_precision", metrics["precision"], on_step=False, on_epoch=True)
        self.log("val_recall", metrics["recall"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", metrics["f1"], prog_bar=True, on_step=False, on_epoch=True)
        return metrics["mse_loss"]

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        metrics = self._shared_step(batch)
        self.log("test_loss", metrics["mse_loss"], on_step=False, on_epoch=True)
        self.log("test_mae_loss", metrics["mae_loss"], on_step=False, on_epoch=True)
        self.log("test_accuracy", metrics["accuracy"], on_step=False, on_epoch=True)
        self.log("test_precision", metrics["precision"], on_step=False, on_epoch=True)
        self.log("test_recall", metrics["recall"], on_step=False, on_epoch=True)
        self.log("test_f1", metrics["f1"], on_step=False, on_epoch=True)
        return metrics["mse_loss"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

