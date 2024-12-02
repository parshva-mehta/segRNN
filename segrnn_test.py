# %%
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from segrnn import SegRNNModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

# %%
# Load the dataset
# Replace 'your_dataset.csv' with your actual data file
df = pd.read_csv('JRB.csv', low_memory=False)
# Step 1: Ensure 'valid' column is datetime and sort the DataFrame
df['valid'] = pd.to_datetime(df['valid'])
df = df.sort_values(by=['station', 'valid']).reset_index(drop=True)

# %%
df.head()

# %%
# Step 2: Replace placeholders with np.nan in continuous columns
continuous_cols = ['tmpf', 'dwpf', 'relh', 'feel', 'drct', 'sknt', 'gust',
                   'peak_wind_gust', 'peak_wind_drct', 'alti', 'mslp', 'vsby',
                   'p01i', 'ice_accretion_1hr', 'ice_accretion_3hr', 'ice_accretion_6hr',
                   'skyl1', 'skyl2', 'skyl3', 'skyl4', 'snowdepth', 'peak_wind_time']

# List of placeholders to replace
placeholders = ['M', 'T', '', 'NaN', 'NULL', 'None']

# Replace placeholders with np.nan
df[continuous_cols] = df[continuous_cols].replace(placeholders, np.nan).astype(str)

# Convert continuous columns to numeric, coercing errors to np.nan
for col in continuous_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# %%
print("Missing values in continuous columns before processing:")
print(df[continuous_cols].isnull().sum())

# %%
# Step 3: Handle missing values in continuous variables


# Identify columns to drop due to high NaN count
nan_threshold = df.shape[0] / 2                     # Remove columns with more than 50% missing values
print(f"nan thresh is {nan_threshold}")
bad_columns = [col for col in df.columns if df[col].isnull().sum() >= nan_threshold]
print(f"bad columns are {bad_columns}")

# Add less relevant and irrelevant features to the removal list
irrelevant_features = [
    'ice_accretion_1hr', 'ice_accretion_3hr', 'ice_accretion_6hr',  # Fully NaN
    'skyl1', 'skyl2', 'skyl3', 'skyl4',  # Less relevant (Sky level altitudes)
    'skyc1', 'skyc2', 'skyc3', 'skyc4',  # Less relevant (Sky coverage)
    'wxcodes',  # Categorical, redundant with precipitation/visibility
    'metar'  # Text format, unusable directly
]

# Combine both lists and ensure no duplicates
columns_to_remove = list(set(bad_columns + irrelevant_features))
df.drop(columns=columns_to_remove, inplace=True)

# Update continuous columns to exclude removed columns
continuous_cols = list(set(continuous_cols) - set(columns_to_remove))
print(f"Remaining continuous columns: {continuous_cols}")


# Apply linear interpolation within each station group using transform
df[continuous_cols] = df.groupby('station')[continuous_cols].transform(
    lambda group: group.interpolate(method='linear')
)

# Handle any remaining missing values with forward and backward fill using transform
df[continuous_cols] = df.groupby('station')[continuous_cols].transform(
    lambda group: group.ffill().bfill()
)


# Verify missing values are filled
print("Missing values in continuous columns after processing:")
print(df[continuous_cols].isnull().sum())

# %%
# Step 5: Feature scaling
# List of all features (excluding 'valid' and 'metar')
feature_cols = continuous_cols #+ categorical_cols
print(df[feature_cols].shape)

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the features
df[feature_cols] = scaler.fit_transform(df[feature_cols])


# %%
# Step 6: Prepare sequences for LSTM input
# Assuming we are predicting 'tmpf' (temperature) as the target variable
# and using previous 24 time steps/8 hours (n_steps_in) to predict the next time step/20 minutes from now (n_steps_out)
# create sliding window sequences X: (114640, 24, 10), y: (114640, 10)

n_steps_in = 24  # Number of past time steps
n_steps_out = 1  # Number of future time steps to predict

# We'll create sequences for each station separately
def create_sequences(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:(i + n_steps_in), :])
        y.append(data[(i + n_steps_in):(i + n_steps_in + n_steps_out), :])
    return np.array(X), np.array(y)

# Prepare data for each station
X_list = []
y_list = []
stations = df['station'].unique()

for station in stations:
    station_data = df[df['station'] == station]
    station_data = station_data.reset_index(drop=True)
    data_values = station_data[feature_cols].values
    # target_col_index = feature_cols.index('tmpf')  # Index of target variable in features

    X_station, y_station = create_sequences(data_values, n_steps_in, n_steps_out)
    X_list.append(X_station)
    y_list.append(y_station)


# Concatenate data from all stations
X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)


if n_steps_out == 1:
    y = y.squeeze(1)  # Shape becomes (num_samples, num_features) = (114640, 10) for JRB


print(X.shape)
print(y.shape)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# %%
# Step 7: Split the data into training and testing sets
# Since it's time-series data, we'll use the first 80% for training and the rest for testing
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Now the data is ready for training the LSTM model

# Define a PyTorch Dataset
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# %%

# Create DataLoaders
batch_size = 32
train_dataset = WeatherDataset(X_train, y_train)
test_dataset = WeatherDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Hyperparameters for SegRNN
input_size = X.shape[2]  # Number of features
hidden_size = 512  # Based on the SEGRNN paper
output_size = X.shape[2]  # Predict all features
segment_length = 8  # Based on the SEGRNN paper
learning_rate = 0.001

# Initialize SegRNNModel
model = SegRNNModel(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    segment_length=segment_length,
    learning_rate=learning_rate
)

# Logger
logger = TensorBoardLogger("logs", name="segrnn_experiment")

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    filename="segrnn-{epoch:02d}-{val_loss:.4f}",
    save_top_k=1,
    monitor="val_loss",
    mode="min"
)

# Trainer with logging and checkpointing
trainer = Trainer(
    max_epochs=4,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=logger,
    callbacks=[checkpoint_callback]
)

# Train the model
trainer.fit(model, train_loader)

# Optional: Evaluate on the test set
trainer.test(model, test_loader)
