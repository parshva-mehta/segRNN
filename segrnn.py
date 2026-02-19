import torch
import torch.nn as nn
import pytorch_lightning as pl

class SegRNN(nn.Module):
   def __init__(self, input_size, hidden_size, output_size, segment_length):
      super(SegRNN, self).__init__()
      self.segment_length = segment_length
      self.gru_segment = nn.GRU(input_size, hidden_size, batch_first=True)
      self.gru_aggregate = nn.GRU(hidden_size, hidden_size, batch_first=True)
      self.fc = nn.Linear(hidden_size, output_size)

   def forward(self, x):
      batch_size, seq_len, _ = x.size()
      num_segments = seq_len // self.segment_length
      x = x[:, :num_segments * self.segment_length, :]
      x = x.view(batch_size * num_segments, self.segment_length, -1)
      
      # Process each segment independently
      _, h_n_segment = self.gru_segment(x)
      h_n_segment = h_n_segment.squeeze(0).view(batch_size, num_segments, -1)
      
      # Aggregate segment embeddings
      _, h_n_aggregate = self.gru_aggregate(h_n_segment)
      h_n_aggregate = h_n_aggregate.squeeze(0)
      
      # Final output
      out = self.fc(h_n_aggregate)
      return out
   
class SegRNNModel(pl.LightningModule):
   def __init__(self, input_size, hidden_size, output_size, segment_length, learning_rate=0.001):
      super(SegRNNModel, self).__init__()
      self.model = SegRNN(input_size, hidden_size, output_size, segment_length)
      self.criterion = nn.MSELoss()
      self.criterion2 = nn.L1Loss()
      self.learning_rate = learning_rate

   def forward(self, x):
      return self.model(x)

   def _classification_metrics(self, outputs, targets):
      # Targets are standardized continuous values; threshold at 0 to track
      # directional correctness (above/below dataset mean) as classification metrics.
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

   def training_step(self, batch, batch_idx):
      inputs, targets = batch
      outputs = self(inputs)
      loss = self.criterion(outputs, targets)
      loss2 = self.criterion2(outputs, targets)
      accuracy, precision, recall, f1 = self._classification_metrics(outputs, targets)

      self.log("train_mse_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
      self.log("train_mae_loss", loss2, prog_bar=True, on_step=False, on_epoch=True)
      self.log("train_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)
      self.log("train_precision", precision, on_step=False, on_epoch=True)
      self.log("train_recall", recall, prog_bar=True, on_step=False, on_epoch=True)
      self.log("train_f1", f1, prog_bar=True, on_step=False, on_epoch=True)
      return loss

   def validation_step(self, batch, batch_idx):
      inputs, targets = batch
      outputs = self(inputs)
      loss = self.criterion(outputs, targets)
      loss2 = self.criterion2(outputs, targets)
      accuracy, precision, recall, f1 = self._classification_metrics(outputs, targets)

      self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
      self.log("val_mae_loss", loss2, on_step=False, on_epoch=True)
      self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)
      self.log("val_precision", precision, on_step=False, on_epoch=True)
      self.log("val_recall", recall, prog_bar=True, on_step=False, on_epoch=True)
      self.log("val_f1", f1, prog_bar=True, on_step=False, on_epoch=True)
      return loss

   def test_step(self, batch, batch_idx):
      inputs, targets = batch
      outputs = self(inputs)
      loss = self.criterion(outputs, targets)
      loss2 = self.criterion2(outputs, targets)
      accuracy, precision, recall, f1 = self._classification_metrics(outputs, targets)

      self.log("test_loss", loss, on_step=False, on_epoch=True)
      self.log("test_mae_loss", loss2, on_step=False, on_epoch=True)
      self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)
      self.log("test_precision", precision, on_step=False, on_epoch=True)
      self.log("test_recall", recall, on_step=False, on_epoch=True)
      self.log("test_f1", f1, on_step=False, on_epoch=True)
      return loss

   def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
      return optimizer