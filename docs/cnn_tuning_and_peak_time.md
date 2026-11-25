# CNN tuning and peak-time reporting

## Peak time
`src/evaluation/peak_time.py` and `scripts/report_peak_time.py` score the daily peak hour of every saved forecast.
Only fully observed 24-hour days are used, and all models are scored on the same days. Timing error is the forecast
peak hour minus the actual peak hour (local time). Also reported: exact hits, within 1 and 2 hours, peak-size error
(MW), and regret (how far real demand at the forecast's peak hour falls below the true peak, as % of the true peak).
Outputs: `results/hfed_extension_eccc_v4/peak_time_*.csv`. Shown in Model results, Peak timing tab.

## Lowering the selected CNN's MAE
Plan: `results/hfed_cnn_tuning_v1/plan.json` (written before any run). Candidates were scored on validation only,
on rolling folds (Jan-Jun 2022, 2023, 2024) with seeds 42, 123 and 456. A candidate is accepted if it lowers mean
validation MAE and does not raise macro provincial MAPE.

| Candidate | Validation MAE (MW) | vs baseline |
|---|---|---|
| baseline (frozen design) | 280.3 | 0.0% |
| weather channels | 267.0 | -4.8% |
| weather + weighted loss 0.5 | 267.7 | -4.5% |
| wider network | 274.8 | -2.0% |
| weighted loss 0.5 | 280.7 | +0.1% |
| weighted loss 1.0 | 283.8 | +1.2% |

Full table: `results/hfed_cnn_tuning_v1/search_summary.csv`. Only `weather` was fitted on the frozen split
(3 seeds) and evaluated once: test2024 MAE 185.4 MW (frozen CNN 188.4), test2025 MAE 237.7 MW (frozen CNN 244.3).

Caveats: 2025 was inspected earlier for the frozen run, so its gain is post-hoc. Feature scalers use data before
2024, which mildly leaks into the 2022 and 2023 folds. Frozen v4 artifacts are unchanged.

Reproduce:
```
python scripts/tune_residual_cnn.py search
python scripts/tune_residual_cnn.py aggregate
python scripts/tune_residual_cnn.py final --candidate weather
python scripts/report_peak_time.py --runs results/hfed_extension_eccc_v4 results/hfed_temporal_controls_v1 results/hfed_cnn_tuning_v1
```
