import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error

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
      self.learning_rate = learning_rate

   def forward(self, x):
      return self.model(x)

   def training_step(self, batch, batch_idx):
      inputs, targets = batch
      outputs = self(inputs)
      loss = self.criterion(outputs, targets)
      self.log("train_loss", loss, prog_bar=True)
      return loss

   def validation_step(self, batch, batch_idx):
      inputs, targets = batch
      outputs = self(inputs)
      loss = self.criterion(outputs, targets)
      self.log("val_loss", loss, prog_bar=True)
      return loss

   def test_step(self, batch, batch_idx):
      inputs, targets = batch
      outputs = self(inputs)
      loss = self.criterion(outputs, targets)
      self.log("test_loss", loss)
      return loss

   def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
      return optimizer