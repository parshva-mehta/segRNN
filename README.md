# segRNN

This repository contains experiments for weather time-series forecasting with a
SegRNN model (plus supporting notebooks/scripts).

## Data Source

Raw weather data is from the Iowa Environmental Mesonet:
https://mesonet.agron.iastate.edu/

## Quick Start (Run the code)

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3) Prepare input CSV

The training script `segrnn_test.py` currently reads `JRB.csv` from the
repository root. The dataset already exists under `csvs/JRB.csv`, so copy it:

```bash
cp csvs/JRB.csv JRB.csv
```

### 4) Run training

```bash
python segrnn_test.py
```

This trains a SegRNN model using PyTorch Lightning and writes:

- logs to `logs/segrnn_experiment/`
- checkpoints to `checkpoints/`

## Run the GNN

The GNN workflow is implemented in `gnn.ipynb` (not a standalone `.py` script).

### 1) Install dependencies

If you have not already:

```bash
pip install -r requirements.txt
```

### 2) Make sure data files exist

The notebook expects these in `csvs/`:

- `csvs/_nylocations.csv`
- station files like `csvs/JRB.csv`, `csvs/ROC.csv`, etc.

Most of these files are already included in this repo.

### 3) Launch and run the notebook

```bash
jupyter notebook gnn.ipynb
```

Then run all cells from top to bottom. The notebook will:

- preprocess station CSVs into `csvs/*_processed.csv`
- build graph/time-series features
- train and evaluate the model

### 4) Output artifacts

Training outputs are written to:

- `logs/segrnn_experiment/`
- `checkpoints/`

## Notes

- `segrnn.py` contains the model (`SegRNN`) and Lightning module
  (`SegRNNModel`).
- `segrnn_test.py` is the main runnable training script.
- `gnn.ipynb` is the main runnable GNN pipeline.
- `convert.py` can export TensorBoard scalar events to CSV after training.