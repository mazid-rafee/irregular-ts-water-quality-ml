# Deep Learning on Water Quality Data as Irregular Time Series

Research code for forecasting irregularly sampled water quality time series from USGS and DbHydro. The pipeline builds sequences with irregular time gaps, trains sequence models, and reports multiple error metrics.

## Highlights
- Supports USGS and DbHydro station/analyte sets.
- Preprocessing handles irregular timestamps, scaling, and outlier removal.
- Baselines: LSTM, BiLSTM, LayerNorm variants, Neural ODE.
- Hybrid models: LSTM with attention, nODE-BiLSTM, nODE-LSTM.
- Metrics: sMAPE, RMSE, MAPE, MSE.

## Repository structure
- `src/` training and evaluation entry points plus utilities.
- `models/` model definitions for baselines and proposed variants.
- `data_analysis/` scripts for regularity analysis and dataset exploration.
- `playground/` exploratory plotting and decomposition scripts.
- `results/` outputs from training runs (CSV-like text).

## Data
The training scripts expect tab-delimited files under `data/` with at least the following columns:
- `Station Identifier`, `Date`, `Time`, and analyte columns (e.g. `Dissolved Oxygen (mg/L)`)

Expected filenames:
- `data/USGS PhysChem Top 5 - Time Series Research.txt`
- `data/DbHydro PhysChem Top 100 - Time Series Research.txt`

These files are not included in this repository. Place them at the paths above or adjust the paths in `src/main.py`.

## Setup
Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch pandas numpy scikit-learn pytz torchdiffeq
```

If you want to use `fast-kan/`, follow its `README.md` for installation.

## Training and evaluation
Run training for one or more models and a dataset:
```bash
python src/main.py --dataset USGS --models LSTM BiLSTM nODEBiLSTM
```

Results are appended to:
- `results/USGS_model_results.txt`
- `results/DbHydro_model_results.txt`

Each row includes: `Station, Main Analyte, Associated Analytes, Model, sMAPE, RMSE, MAPE, MSE`.

## Notes
- Sequence length and batch size are configured in `src/main.py` via `prepare_data_loaders(...)`.
- GPU is used when available. `src/utils/model_trainer_evaluator.py` sets `CUDA_VISIBLE_DEVICES=1` by default.
- If you add new analytes or stations, update `src/stations_and_analytes.py`.
