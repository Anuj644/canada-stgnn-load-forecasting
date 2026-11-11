# HFED Data Audit
*Phase 1 — Repository Inspection and Data Audit*

## 1. Repository State (Pre-Development)

| Item | Finding |
|---|---|
| Existing code | `hfed_dataset_downloader.py` — robust SDMX/CSV downloader |
| Existing data | None committed (data/ is .gitignored) |
| Dependencies | Only `requests` and `tqdm` installed |
| Architecture | No data pipeline, models, or dashboard yet |

---

## 2. HFED API and Dataset Overview

**Source:** Statistics Canada High-Frequency Electricity Data (HFED)  
**Base URL:** `https://api.statcan.gc.ca/hfed-dehf/sdmx/rest`  
**Format:** SDMX 2.1 / CSV (`dimensionAtObservation=AllDimensions`)  
**Data version:** All flows at v1.0

### 2.1 Available Regions (10 total)

| Code | Full Name | SDMX Flow |
|---|---|---|
| AB | Alberta | DF_HFED_AB |
| BC | British Columbia | DF_HFED_BC |
| NB | New Brunswick | DF_HFED_NB |
| NL | Newfoundland and Labrador | DF_HFED_NL |
| NS | Nova Scotia | DF_HFED_NS |
| ON | Ontario | DF_HFED_ON |
| PE | Prince Edward Island | DF_HFED_PE |
| QC | Quebec | DF_HFED_QC |
| SK | Saskatchewan | DF_HFED_SK |
| YK | Yukon | DF_HFED_YK |

---

## 3. CSV Schema (All Regions, Identical)

Every downloaded file shares these 15 columns:

| Column | Type | Description |
|---|---|---|
| `DATAFLOW` | string | e.g. `CCEI:DF_HFED_ON(1.0)` |
| `FREQ` | string | `H` (hourly) or `N` (non-standard / sub-hourly) |
| `REF_AREA` | string | ISO subdivision code e.g. `CA_ON` |
| `COUNTERPART_AREA` | string | Specific sub-area or `_Z` for region total |
| `ENERGY_FLOWS` | string | Variable code e.g. `ONTARIO_DEMAND` |
| `TIME_PERIOD` | string | UTC datetime ISO 8601 `YYYY-MM-DDTHH:MM:SS` |
| `OBS_VALUE` | float-string | Measurement value |
| `DATETIME_LOCAL` | string | Local datetime (naive ISO 8601) |
| `DAYLIGHT_OFFSET` | float-string | DST offset component (hours) |
| `DATETIME_LOCAL_OFFSET` | float-string | UTC offset (hours) — **inconsistent formatting: `4`, `4.0`, `5`, `5.0`** |
| `UNIT_MEASURE` | string | `MW` or `CAD` (ON only for price) |
| `UNIT_MULT` | int-string | Power-of-10 multiplier; always `0` observed |
| `MEASURE_TYPE` | string | Always `ENERGY` observed |
| `OBS_STATUS` | string | Always `A` (normal) in sampled data |
| `CONF_STATUS` | string | Always `F` (free) in sampled data |

**Key finding:** `TIME_PERIOD` is in UTC. `DATETIME_LOCAL` provides the correct local time for each region. Both must be retained.

---

## 4. Per-Region Variable Inventory

### 4.1 Demand Variable Mapping

| Region | Primary Demand Variable | Secondary / Notes |
|---|---|---|
| AB | `INTERNAL_LOAD` | Also has `TOTAL_NET_GENERATION` |
| BC | `LOAD` | Only 2 variables total |
| NB | `DEMAND` | Also `LOAD`; sub-hourly (non-integer seconds) |
| NL | `DEMAND` | Sub-hourly (5-min intervals observed) |
| NS | `LOAD` | Also `DEMAND_PEAK` (daily, not hourly) |
| ON | `ONTARIO_DEMAND` | Also `MARKET_DEMAND`; richest dataset |
| PE | `ON_ISL_LOAD` | 15-min intervals |
| QC | `DEMAND` | 15-min intervals |
| SK | `SYSTEM_DEMAND` | Daily-resolution only (1 obs/day for each var) |
| YK | `TOTAL` | 10-min intervals; no explicit demand variable |

### 4.2 Temporal Resolution by Region

| Region | Observed Frequency | Native granularity | Usable for hourly model? |
|---|---|---|---|
| AB | H + N | Hourly | ✅ Yes |
| BC | H + N | Hourly | ✅ Yes |
| NB | N only | Sub-hourly (~5 min, irregular) | ⚠️ Needs resampling |
| NL | N only | Sub-hourly (~5 min) | ⚠️ Needs resampling |
| NS | H + N | Hourly | ✅ Yes |
| ON | H + N | Hourly | ✅ Yes |
| PE | N only | 15-min | ⚠️ Needs resampling |
| QC | H + N | 15-min to hourly | ⚠️ Needs resampling |
| SK | N only | **Daily — only 1 obs per day** | ❌ Insufficient for hourly model |
| YK | N only | 10-min, irregular | ⚠️ Needs resampling |

### 4.3 Variable Richness by Region

| Region | Total Variables | Has weather-proxy vars? |
|---|---|---|
| AB | 11 | WIND, SOLAR, HYDRO, GAS, COAL |
| BC | 2 | NSI only |
| NB | 5 | RM_10, RM_30 (reserve margins) |
| NL | 1 | Demand only |
| NS | 6 | WIND, IMPORT, EXPORT |
| ON | 34 | Richest: generation mix, prices, sector demand |
| PE | 7 | WIND, IMPORT_CABLES |
| QC | 12 | WIND, SOLAR, HYDRO, THERMAL, sector breakdown |
| SK | 9 | WIND, SOLAR, HYDRO, COAL, GAS (but daily only) |
| YK | 3 | HYDRO, THERMAL |

---

## 5. Data Quality Findings

### 5.1 Issues Identified

| Issue | Affected Regions | Severity | Mitigation |
|---|---|---|---|
| `DATETIME_LOCAL_OFFSET` inconsistent format (`4` vs `4.0`) | All | Low | Normalize to float on load |
| Sub-hourly resolution requires resampling to hourly | NB, NL, PE, QC, YK | Medium | Mean/sum aggregation with coverage check |
| SK only has 1 observation per day | SK | **Critical** | **Exclude SK from hourly model** |
| YK has no explicit "demand" variable; `TOTAL` = generation total | YK | Medium | Use `TOTAL` as proxy; document |
| NB `DEMAND` has irregular sub-second timestamps | NB | Medium | Floor to minute, then resample |
| `COUNTERPART_AREA` has 742+ sub-areas for ON (generation units) | ON | Medium | Filter to `_Z` (region total) for demand |
| No weather observations in HFED | All | Important | HFED contains only electricity variables; external weather data required or use generation mix as proxy |
| DST transitions: `DAYLIGHT_OFFSET` is always 0 in sampled data | All | Low | Monitor; handle spring-forward/fall-back gaps/duplicates |

### 5.2 Weather Variables

**HFED does NOT contain weather observations** (temperature, precipitation, wind speed, etc.). The only weather-proxy signals available are:
- Generation mix (hydro, wind, solar output) — available for some regions
- These are endogenous to the power system, not exogenous weather inputs

**Decision:** Use generation mix variables as features where available. No external weather API is in scope (the project specification does not mandate one beyond HFED). This is documented clearly in the feature schema.

---

## 6. Regions Selected for the Hourly Forecasting Model

Based on the audit, **8 regions** have sufficient hourly (or resampleable sub-hourly) coverage:

| Region | Demand Variable | Resample needed | Include |
|---|---|---|---|
| AB | `INTERNAL_LOAD` | No (hourly) | ✅ |
| BC | `LOAD` | No (hourly) | ✅ |
| NB | `DEMAND` | Yes (sub-hourly → hourly) | ✅ |
| NL | `DEMAND` | Yes (sub-hourly → hourly) | ✅ |
| NS | `LOAD` | No (hourly) | ✅ |
| ON | `ONTARIO_DEMAND` | No (hourly) | ✅ |
| PE | `ON_ISL_LOAD` | Yes (15-min → hourly) | ✅ |
| QC | `DEMAND` | Yes (15-min → hourly) | ✅ |
| **SK** | `SYSTEM_DEMAND` | Daily only | ❌ Excluded |
| **YK** | `TOTAL` (proxy) | Yes (10-min → hourly) | ⚠️ Optional |

**Primary cohort:** AB, BC, NB, NL, NS, ON, PE, QC (8 regions)  
**YK:** Included as optional — documented as generation total proxy, not true demand.

---

## 7. Nager.Date Holiday API

**Version:** Community API v4  
**Base URL:** `https://nagerholidays.com/api/v4`  
**Example:** `GET /Holidays/CA/{year}` returns JSON array with `date`, `localName`, `name`, `countryCode`, `fixed`, `global`, `counties` fields.  
**Subdivision mapping to HFED regions:**

| HFED Region | Nager.Date subdivision |
|---|---|
| AB | `CA-AB` |
| BC | `CA-BC` |
| NB | `CA-NB` |
| NL | `CA-NL` |
| NS | `CA-NS` |
| ON | `CA-ON` |
| PE | `CA-PE` |
| QC | `CA-QC` |
| SK | `CA-SK` |
| YK | `CA-YT` |

National (`global: true`) holidays apply to all regions. Provincial holidays (`counties` list) apply to matching subdivisions only.

---

## 8. Proposed Project Architecture

```
canada-stgnn-load-forecasting/
├── hfed_dataset_downloader.py      # existing
├── requirements.txt                # expand
├── config/
│   └── config.yaml                 # paths, hyperparams, feature config
├── src/
│   ├── data/
│   │   ├── hfed_loader.py          # load & normalize raw HFED CSVs
│   │   ├── resampler.py            # sub-hourly → hourly aggregation
│   │   ├── holiday_client.py       # Nager.Date API + local cache
│   │   ├── feature_engineering.py  # calendar, lag, rolling features
│   │   └── pipeline.py             # end-to-end preprocessing pipeline
│   ├── models/
│   │   ├── naive.py                # seasonal naive baselines
│   │   ├── xgboost_model.py        # XGBoost multi-region
│   │   ├── lstm_model.py           # LSTM multi-horizon
│   │   ├── gru_model.py            # GRU multi-horizon
│   │   ├── stgnn/
│   │   │   ├── graph_builder.py    # geographic, correlation, interconnection, identity
│   │   │   ├── stgcn.py            # STGCN architecture
│   │   │   └── trainer.py          # training loop
│   │   └── pbc/                    # Phase 8: probabilistic bias correction
│   │       ├── debias_plus.py
│   │       └── persistence_plus.py
│   ├── evaluation/
│   │   ├── metrics.py              # MAE, RMSE, MAPE, CRPS
│   │   └── evaluator.py            # full evaluation pipeline
│   ├── high_demand/
│   │   └── detector.py             # threshold-based flagging
│   └── api/
│       └── app.py                  # FastAPI backend
├── dashboard/                      # Streamlit or Dash frontend
│   └── app.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_leakage.py
│   └── test_holidays.py
├── notebooks/
│   └── 01_data_exploration.ipynb
└── docs/
    └── data_audit.md               # this file
```

---

## 9. Key Design Decisions

1. **UTC anchor + local calendar:** All timestamps stored in UTC (`TIME_PERIOD`). Local-time calendar features derived from `DATETIME_LOCAL` or by applying the documented UTC offset.
2. **Demand variable is unambiguous per region** — no averaging or blending of multiple demand signals.
3. **SK excluded from the main forecasting cohort** due to daily-only resolution.
4. **No weather API**: HFED provides no weather. Generation mix (wind output, solar output, hydro output) is used as a weather-correlated proxy where available. This is documented as a limitation.
5. **Forecast cutoff at 23:00 local time** — all features computed from data available at that time only.
6. **Scalers / graph statistics fitted on training data only** — applied to val/test without refitting.
7. **Chronological splits**: 70% train / 15% val / 15% test, no shuffling.
