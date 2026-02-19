# Station Pipeline

This folder is a standalone weather-station workflow that is isolated from the rest
of the repository. It supports:

1. Fetching data from Iowa Environmental Mesonet (IEM) based on user city/state.
2. Processing the downloaded data with the same core logic used in
   `data_processing.ipynb`.
3. Training SegRNN and exporting predictions + accuracy metrics.

## Folder Layout

- `fetch_data.py` - city/state -> nearest station lookup + raw CSV download
- `process_data.py` - notebook-equivalent preprocessing + sequence creation
- `train.py` - SegRNN training/evaluation + prediction/metrics export
- `model.py` - local SegRNN + Lightning module copy
- `station_lookup.py` - location geocoding and nearest-station resolution
- `data/raw/` - downloaded station CSV files
- `data/processed/` - processed arrays and metadata for training
- `artifacts/` - predictions and metrics outputs
- `logs/`, `checkpoints/` - training artifacts

## Setup

```bash
cd /Users/parshvamehta/segRNN-GNN/segRNN/station_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1: Fetch Data

```bash
python fetch_data.py --city Rochester --state NY --days-back 365
```

Optional parameters:
- `--network` to override default network (`<STATE>_ASOS`)
- `--start YYYY-MM-DD --end YYYY-MM-DD` for explicit range

Outputs:
- `data/raw/<station>_<start>_<end>.csv`
- `artifacts/fetch_metadata.json`

## Step 2: Process Data

```bash
python process_data.py
```

Optional:
- `--input-csv data/raw/<file>.csv`
- `--n-steps-in 24 --n-steps-out 1`
- `--test-split 0.2`
- `--no-normalize`

Outputs in `data/processed/`:
- `*_processed.csv`
- `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`
- `train_targets.csv`, `test_targets.csv`
- `feature_columns.json`, `processing_metadata.json`, `scaler.pkl` (if normalized)

## Step 3: Train and Evaluate

```bash
python train.py --epochs 100 --batch-size 32 --hidden-size 512 --segment-length 8 --lr 0.001
```

Outputs:
- `checkpoints/segrnn-*.ckpt`
- `logs/segrnn_experiment/`
- `artifacts/predictions.csv`
- `artifacts/metrics.json`

All outputs stay within this `station_pipeline/` folder.

