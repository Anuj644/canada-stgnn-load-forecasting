# Spatio-Temporal Graph Neural Network for Day-Ahead Electricity Demand Forecasting in Canada

End-to-end electricity demand forecasting system for Canadian regions using the Statistics Canada High-Frequency Electricity Data (HFED) dataset, spatio-temporal graph neural networks, and baseline models.

---

## Project Overview

This system forecasts electricity demand for all 24 hours of the following day across up to 8 consistently available HFED regions. A forecast is issued at **23:00 local time** using only information available at that moment.

**Key research question:** Does an STGNN with a regional graph outperform strong time-only models for full day-ahead, per-region, and peak-hour forecasting?

The core comparison is between the STGNN with different graph configurations (geographic, correlation, interconnection) versus the **identity-graph control** that shares no information across regions. This isolates the value of spatial modelling.

---

## Regions

| Code | Region | Demand Variable |
|---|---|---|
| AB | Alberta | INTERNAL_LOAD |
| BC | British Columbia | LOAD |
| NB | New Brunswick | DEMAND |
| NL | Newfoundland & Labrador | DEMAND |
| NS | Nova Scotia | LOAD |
| ON | Ontario | ONTARIO_DEMAND |
| PE | Prince Edward Island | ON_ISL_LOAD |
| QC | Quebec | DEMAND |
| YK | Yukon (optional) | TOTAL (proxy) |

Saskatchewan (SK) is excluded: it only publishes daily-resolution data, which is insufficient for hourly forecasting.

---

## Project Structure

```
canada-stgnn-load-forecasting/
├── hfed_dataset_downloader.py   # download raw HFED data from Statistics Canada
├── train.ipynb                  # training, EDA, and evaluation workflow
├── train.py                     # original command-line workflow
├── data/                        # raw downloads, holiday cache, processed data
├── models/                      # saved checkpoints by model family
├── results/                     # forecasts, metrics, and high-demand flags
├── config/
│   └── config.yaml              # all paths, hyperparameters, feature settings
├── src/
│   ├── data/
│   │   ├── hfed_loader.py       # load & normalize raw HFED CSVs
│   │   ├── resampler.py         # sub-hourly → hourly aggregation
│   │   ├── holiday_client.py    # Nager.Date API + local cache
│   │   ├── feature_engineering.py  # calendar, lag, rolling features
│   │   └── pipeline.py          # end-to-end preprocessing pipeline
│   ├── models/
│   │   ├── naive.py             # seasonal naive baselines (day, week)
│   │   ├── xgboost_model.py     # XGBoost multi-region
│   │   ├── lstm_model.py        # LSTM multi-horizon
│   │   ├── gru_model.py         # GRU multi-horizon
│   │   ├── stgnn/
│   │   │   ├── graph_builder.py # 4 graph configs: identity, geo, corr, ic
│   │   │   ├── stgcn.py         # STGCN architecture (Yu et al. 2018)
│   │   │   ├── trainer.py       # original training loop and ablation runner
│   │   │   └── residual.py      # residual STGCN used in the notebook
│   │   └── pbc/                 # probabilistic bias correction
│   │       ├── demand.py        # demand CDF calibration used in the notebook
│   │       ├── debias_plus.py   # Debias++-inspired correction
│   │       └── persistence_plus.py  # Persistence++-inspired correction
│   ├── evaluation/
│   │   ├── metrics.py           # MAE, RMSE, MAPE, CRPS, pinball
│   │   └── evaluator.py         # full evaluation pipeline
│   ├── high_demand/
│   │   └── detector.py          # threshold-based high-demand flagging
│   └── api/
│       └── app.py               # FastAPI backend
├── dashboard/
│   └── app.py                   # Streamlit interactive dashboard
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py    # resampling tests
│   ├── test_features.py         # calendar, lag, rolling, quality flags
│   ├── test_leakage.py          # data leakage prevention tests
│   ├── test_holidays.py         # holiday client + feature tests
│   ├── test_graph_builder.py    # all 4 graph configurations
│   ├── test_forecast_dimensions.py  # output shape + metrics tests
│   └── test_high_demand.py      # detection + threshold tests
└── docs/
    └── data_audit.md            # Phase 1 HFED data audit report
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For PyTorch (LSTM, GRU, STGNN), install the appropriate wheel for your platform:
```bash
# CPU (all platforms)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA 12.4, on supported NVIDIA hardware)
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### 2. Download HFED data

Download the eight provinces for the configured study period:
```bash
python hfed_dataset_downloader.py --start 2019-01-01 --end 2024-12-31 --provinces AB BC NB NL NS ON PE QC
```

The downloader skips already-completed regions. Use `--overwrite` to re-fetch.

### 3. Run the training notebook

```bash
python -m jupyterlab train.ipynb
```

Select the project’s Python kernel and run all cells. The notebook prepares missing processed data and baseline forecasts, then fits the residual STGCN, local XGBoost, and PBC models. To regenerate artifacts from the existing residual checkpoints, set `RETRAIN_NEW_MODELS = False`. To explicitly rebuild processed data or refit the original baselines, enable `REBUILD_DATA` or `RETRAIN_BASELINES`.

The notebook writes checkpoints to `models/stgnn/`, `models/xgboost/`, and `models/pbc/`. It publishes forecasts, the matched `model_comparison.csv`, per-model metrics, and high-demand flags in `results/` for the dashboard and API. Rerunning refreshes these result files.

### 4. Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

### 5. Launch the API backend

```bash
python -m uvicorn src.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

---

## Development Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Repository inspection & HFED data audit | ✅ Complete |
| 2 | Data cleaning, feature engineering, holiday integration | ✅ Complete |
| 3 | Seasonal naive & XGBoost baselines | ✅ Complete |
| 4 | LSTM & GRU baselines | ✅ Complete |
| 5 | STGNN + graph ablation | ✅ Complete |
| 6 | Evaluation, comparison, peak-demand analysis | ✅ Complete |
| 7 | High-demand detection | ✅ Complete |
| 8 | Interactive dashboard + FastAPI backend | ✅ Complete |
| 9 | Probabilistic bias correction | Implemented in `train.ipynb` |

---

## Forecasting Problem

- **Issue time:** 23:00 local time on day D
- **Target window:** Hours 0–23 of day D+1 (midnight through 11 PM)
- **Output shape:** 24 hours × N regions
- **Features used:** Demand lags, rolling statistics, calendar features, holiday flags, generation mix (wind, solar, hydro where available)
- **Future weather:** Not used. HFED contains no weather observations. Any experiment using future generation mix as input is documented as an ideal-information experiment.

---

## Graph Ablation

Four graph configurations are compared with identical model architecture:

| Graph | Description |
|---|---|
| **identity** | No cross-region edges — the most important experimental control |
| **geographic** | Edges between regions within 2,000 km of each other |
| **correlation** | Edges where training-period Pearson correlation ≥ 0.6 |
| **interconnection** | Only documented electricity tie-lines |

The key comparison is **STGNN (regional graph) vs STGNN (identity)** — same architecture, same features, only the graph differs. This isolates the value of spatial modelling.

---

## Evaluation

All models are evaluated on:
- **Overall** MAE, RMSE, MAPE across all regions and hours
- **Per region** (8 regions separately)
- **Per forecast hour** (hours 0–23)
- **Peak demand** (90th percentile threshold, computed from training data only)
- **Holiday periods**

MAPE excludes observations with |actual| < 10 MW to avoid division near zero.

---

## Data Audit Summary

See [`docs/data_audit.md`](docs/data_audit.md) for the complete Phase 1 audit.

Key findings:
- **10 HFED regions** available; **8 included** in the hourly model (SK excluded: daily resolution only)
- All data in UTC with local-time offset provided per row
- Sub-hourly regions (NB, NL, PE, QC, YK) are resampled to hourly using mean aggregation with ≥50% coverage requirement
- **No weather data** in HFED; generation mix variables (wind output, solar output, hydro output) used as weather-correlated proxies
- OBS_STATUS = 'A' (normal) in all sampled data

---

## Probabilistic Bias Correction (Phase 9)

The notebook adapts Mouatadid et al. (2026) to electricity demand. Debias++ estimates calendar-dependent changes to the residual CDF. Persistence++ adds recent errors from observations available by forecast issuance. The notebook projects each CDF with isotonic regression, averages the two, and evaluates CRPS, pinball loss, and prediction interval coverage. The dashboard reads the intervals from `results/probabilistic_predictions.parquet`.

---

## Reproducibility

- Baseline seeds are in `config/config.yaml`. The notebook runs the residual graph experiments with seeds 42 and 123.
- Scalers, graph statistics, and peak thresholds are fitted on training data only
- Splits are strictly chronological (no shuffling)
- Model checkpoints are saved to `models/`
- Prediction DataFrames are saved to `results/` as Parquet

---

## Important Notes

- **No hardcoded results.** All metrics displayed in the dashboard are computed from actual trained models.
- **No invented data.** Region coordinates, interconnections, and holiday information come from documented sources only.
- **Decision support only.** High-demand flags are for planner awareness, not grid control or dispatch.
- The `data/` directory is git-ignored. Run the downloader to populate it.
