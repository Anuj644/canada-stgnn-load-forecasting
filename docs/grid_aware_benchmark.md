# Brazilian benchmark against Grid-Aware

This benchmark retrains our existing residual STGCN modules on the base paper's
Brazilian state dataset. It also trains an independent implementation of the
paper's frozen multiscale GATv2 plus LSTM. It is an initial controlled comparison,
not an exact reproduction of the paper's hyperparameter search or a published win.

Sources: [author implementation](https://github.com/ugoorji12/Grid-Aware_STGNN_for_Multi-Horizon_Load_Forecasting),
[updated dataset](https://zenodo.org/records/18595751),
[original static attributes](https://zenodo.org/records/17175783).

## Run

From the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-benchmark.txt
.venv/Scripts/python.exe scripts/download_grid_aware.py
.venv/Scripts/python.exe scripts/benchmark_grid_aware.py --audit
.venv/Scripts/python.exe scripts/benchmark_grid_aware.py --output results/grid_aware_pilot_seed42 --epochs 25 --batch-times 64 --patience 6
.venv/Scripts/python.exe scripts/summarize_grid_aware.py --results results/grid_aware_pilot_seed42
```

The old `dynamic_data.csv` contains only 2019–2020 and cannot supply the paper's
training years. The updated file contains 2017–2020. Its timestamps use
`DD/MM/YYYY HH:MM`; parsing them month-first silently corrupts chronology.
Checksums pin exact files. All input files stay in ignored `data/`.

## Protocol and interpretation

- Train: 2017–2018. Validation: January–June 2019. Test: July–December 2019.
  The extra 2020 data are not used in fitting, selection, or reported 2019 metrics.
- Issue each hour using 24 hours of history. Predict the three endpoints at
  +1, +6, and +24 hours; this is different from our Canadian daily issuance task.
- Keep every history and target inside its split. Skip the first seven days of
  training so all residual seasonal features exist. All models use identical keys.
- Fit input/target scalers only on training data. Supply only historical weather
  to the paper implementation. The residual model uses demand histories,
  daily lags and deterministic target calendars, matching its existing design.
- Cache the frozen graph embedding for all states in evaluation mode and save it
  in the checkpoint. The published snippets have inconsistent indexing, missing
  cache serialization, missing helpers, and executable Markdown fences; they
  cannot be run unchanged. We retain the downloaded files without executing them.
- Use published LSTM dimensions, but no Optuna trials. Batch 64 timestamps across
  27 states, rather than the paper's batch of 32 samples. The pilot is one seed
  with a 25-epoch cap and validation-based early stopping. These differences mean
  its scores must be labeled reimplementation/pilot, not exact replication.
- The residual adapter calls the existing `ResidualSTGCN` temporal encoder,
  region embedding, seasonal anchor and prediction head. It vectorizes shared
  histories and trains just the three endpoints. It does not add the proposed
  dynamic gate. Identity and physical-grid variants isolate neighbor access.
- Use pooled per-state MW MAE/RMSE and the author's MAPE formula without our
  Canadian 10 MW exclusion. Do not compare these to national aggregate errors.
- Curated upstream preprocessing is inherited, not independently audited for
  original publication-time availability or imputation provenance.

Each output directory contains a manifest with hashes and deviations, model
checkpoints, keyed Parquet predictions, and summary metrics by horizon. Existing
summary results are protected against overwriting. Published Table 3 values are
reference metadata, clearly separate from calculated metrics.

Before claiming superiority, reproduce both strong graph backbones (including
the published GRU), use multiple seeds and comparable validation tuning budgets,
and quantify paired error differences over blocks of days. The current adapter
uses historical neighbor loads, a richer information set than the frozen static
graph embedding; include temporal controls and explicit feature-set ablations.

## Completed first pilot

Seed 42, 25 epochs per model, 117,963 identical observations at each lead time.
The three causal indexing/model parity tests passed. Static node attributes in
the original release were independently checked against the updated release and
match for all 27 states. All downloaded dataset MD5 hashes match Zenodo metadata.

| Model | 24h MAE, MW | RMSE, MW | MAPE |
|---|---:|---:|---:|
| Published GAT-GRU, Table 3 reference | 63.70 | 146.30 | 4.54% |
| Published GAT-LSTM, Table 3 reference | 64.44 | 151.10 | 3.93% |
| Local frozen-graph LSTM pilot | 96.39 | 189.49 | 7.94% |
| Existing residual model, identity | 52.82 | 135.60 | 2.07% |
| Existing residual model, physical grid | 52.92 | 136.08 | 2.07% |
| Previous week | 107.13 | 286.56 | 3.91% |
| Previous day | 147.98 | 407.71 | 5.67% |

These results establish a promising transfer experiment, not a confirmed published
win. The local base implementation has not reproduced published accuracy. Models
use different feature sets, losses and scaling, and the pilot does not repeat the
paper's optimization budget. They do share targets, split periods, and metrics.
Do not present the numerical comparison to Table 3 as an exact replication.

At +1h the residual identity MAE is 32.98 MW (published LSTM: 31.79); at +6h it
is 38.93 MW (published LSTM: 48.95). Improvement is not uniform over all horizons.
At +24h the physical graph minus identity MAE difference is +0.097 MW, with a
seven-day moving-block bootstrap 95% interval of [-0.105, +0.299] MW. This single
seed does not establish incremental spatial value.

Calculated metrics and keyed forecasts are in `results/grid_aware_pilot_seed42/`.
`comparison.csv` adds fixed naive baselines; `paired_graph_comparison.json`
contains the block-bootstrap estimates. Keep this run fixed and use only earlier
validation windows for subsequent tuning. The extra 2020 data remain unused.
