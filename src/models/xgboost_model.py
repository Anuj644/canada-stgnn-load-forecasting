"""XGBoost day-ahead electricity demand forecaster — per-region models.

Architecture (v2):
- One XGBoost model per (region, forecast_hour) pair.
  8 regions × 24 hours = 192 models.
- Each model is trained only on its own region's data, eliminating
  cross-region scale confusion for small regions (NB, NL, NS, PE, YK).
- Features: demand lags, rolling stats, calendar features, generation mix.
  Region dummy is omitted (each model is region-specific already).

Data leakage prevention:
- Lag and rolling features are computed with safe offsets in feature_engineering.
- The input row is always the 23:00 observation of the issue day.
- Scalers / encoders fitted on training data only (via feature_engineering pipeline).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("xgboost not installed.  XGBoostForecaster unavailable.")

FORECAST_HOURS = list(range(24))


# ---------------------------------------------------------------------------
# Feature matrix builder (no region dummies — per-region models)
# ---------------------------------------------------------------------------

_DROP_COLS = {
    "time_utc", "time_local", "DATETIME_LOCAL_OFFSET",
    "demand_flag_missing", "demand_flag_negative", "demand_flag_extreme",
    "holiday_name", "year", "region",   # region excluded: each model is region-specific
}


def _build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Extract a numeric feature matrix, dropping metadata and string columns."""
    df = df.copy()
    drop = [c for c in df.columns if c in _DROP_COLS]
    df = df.drop(columns=drop, errors="ignore")

    # Also drop demand_mw itself (target is handled separately)
    df = df.drop(columns=["demand_mw"], errors="ignore")

    # Boolean → int
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    # Keep only numeric
    df = df.select_dtypes(include=[np.number])

    # Replace inf/NaN with column median
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())

    return df.values.astype(np.float32), list(df.columns)


# ---------------------------------------------------------------------------
# Day-ahead sample builder
# ---------------------------------------------------------------------------

def build_day_ahead_samples(
    df: pd.DataFrame,
    horizon: int = 24,
    time_col: str = "time_utc",
    demand_col: str = "demand_mw",
    region_col: str = "region",
) -> pd.DataFrame:
    """Build (issue_row_features, target_demand) pairs per region.

    For each forecast issue time (23:00 local on day D):
        X features: all columns from the 23:00 row of day D
        y: demand at hour h of day D+1, for h in 0..23

    Returns a DataFrame with all feature columns plus:
        target_demand, target_hour, forecast_date, region
    """
    if "time_local" in df.columns:
        local_hour = pd.to_datetime(df["time_local"]).dt.hour
        local_date = pd.to_datetime(df["time_local"]).dt.date
    else:
        local_hour = pd.to_datetime(df[time_col]).dt.hour
        local_date = pd.to_datetime(df[time_col]).dt.date

    df = df.copy()
    df["_local_hour"] = local_hour.values
    df["_local_date"] = local_date.values

    all_samples = []
    for region in df[region_col].unique():
        rdf = df[df[region_col] == region].sort_values(time_col).reset_index(drop=True)

        # Build a (date, hour) → demand lookup
        target_lookup = {}
        for _, row in rdf.iterrows():
            d = row["_local_date"]
            h = int(row["_local_hour"])
            v = row[demand_col]
            target_lookup[(d, h)] = v

        # Issue rows: local hour == 23
        issue_rows = rdf[rdf["_local_hour"] == 23]

        for _, issue_row in issue_rows.iterrows():
            issue_date = issue_row["_local_date"]
            next_date = (pd.Timestamp(issue_date) + pd.Timedelta(days=1)).date()

            for target_hour in range(horizon):
                target_demand = target_lookup.get((next_date, target_hour), np.nan)
                sample = issue_row.to_dict()
                sample["target_demand"] = target_demand
                sample["target_hour"] = target_hour
                sample["forecast_date"] = next_date
                all_samples.append(sample)

    return pd.DataFrame(all_samples)


# ---------------------------------------------------------------------------
# XGBoost Forecaster — per region, per hour
# ---------------------------------------------------------------------------

class XGBoostForecaster:
    """One XGBoost model per (region, target_hour) — 8 regions × 24 hours."""

    name = "xgboost"

    def __init__(self, config: dict):
        if not XGB_AVAILABLE:
            raise ImportError("Install xgboost:  pip install xgboost")
        self.config = config
        xgb_cfg = config.get("xgboost", {})
        self.n_estimators = xgb_cfg.get("n_estimators", 500)
        self.max_depth = xgb_cfg.get("max_depth", 6)
        self.learning_rate = xgb_cfg.get("learning_rate", 0.05)
        self.subsample = xgb_cfg.get("subsample", 0.8)
        self.colsample_bytree = xgb_cfg.get("colsample_bytree", 0.8)
        self.random_state = xgb_cfg.get("random_state", 42)
        self.early_stopping_rounds = xgb_cfg.get("early_stopping_rounds", 50)

        # models[(region, hour)] = fitted XGBRegressor
        self.models: dict[tuple[str, int], "xgb.XGBRegressor"] = {}
        self.feature_names: list[str] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_dir: Optional[str | Path] = None,
    ) -> "XGBoostForecaster":
        """Fit one model per (region, hour).

        Parameters
        ----------
        train_df, val_df : Feature-engineered DataFrames with demand_mw.
        model_dir : If given, save models here.
        """
        logger.info("Building day-ahead samples for XGBoost training...")
        train_samples = build_day_ahead_samples(train_df)
        val_samples   = build_day_ahead_samples(val_df)

        regions = sorted(train_samples["region"].unique())
        logger.info("XGBoost: fitting %d regions × 24 hours = %d models",
                    len(regions), len(regions) * 24)

        # Get feature names from first valid sample
        drop_extra = ["target_demand", "target_hour", "forecast_date",
                      "_local_hour", "_local_date"]
        sample_feat = train_samples.drop(columns=drop_extra, errors="ignore").head(5)
        _, self.feature_names = _build_feature_matrix(sample_feat)

        for region in regions:
            tr_r = train_samples[train_samples["region"] == region]
            va_r = val_samples[val_samples["region"] == region]

            for target_hour in FORECAST_HOURS:
                tr_h = tr_r[tr_r["target_hour"] == target_hour].dropna(subset=["target_demand"])
                va_h = va_r[va_r["target_hour"] == target_hour].dropna(subset=["target_demand"])

                if len(tr_h) < 5:
                    logger.warning("Region %s hour %02d: only %d training samples; skipping.",
                                   region, target_hour, len(tr_h))
                    continue

                tr_feat = tr_h.drop(columns=drop_extra, errors="ignore")
                X_tr, _ = _build_feature_matrix(tr_feat)
                y_tr    = tr_h["target_demand"].values

                eval_set = None
                if len(va_h) >= 5:
                    va_feat = va_h.drop(columns=drop_extra, errors="ignore")
                    X_va, _ = _build_feature_matrix(va_feat)
                    y_va    = va_h["target_demand"].values
                    eval_set = [(X_va, y_va)]

                model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    random_state=self.random_state,
                    tree_method="hist",
                    verbosity=0,
                    early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
                )
                model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
                self.models[(region, target_hour)] = model

            logger.info("XGBoost: fitted region %s (%d hour models).",
                        region, sum(1 for k in self.models if k[0] == region))

        if model_dir:
            self.save(model_dir)
        logger.info("XGBoost fitting complete: %d models total.", len(self.models))
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Generate 24-hour predictions for all regions in test_df.

        Returns DataFrame: region, forecast_date, target_hour, predicted, actual
        """
        if not self.models:
            raise RuntimeError("Call fit() before predict().")

        test_samples = build_day_ahead_samples(test_df)
        drop_extra = ["target_demand", "target_hour", "forecast_date",
                      "_local_hour", "_local_date"]
        rows = []

        for region in sorted(test_samples["region"].unique()):
            tr_r = test_samples[test_samples["region"] == region]
            for target_hour in FORECAST_HOURS:
                key = (region, target_hour)
                if key not in self.models:
                    continue
                subset = tr_r[tr_r["target_hour"] == target_hour]
                if subset.empty:
                    continue
                feat = subset.drop(columns=drop_extra, errors="ignore")
                X, _ = _build_feature_matrix(feat)
                preds = self.models[key].predict(X)

                for i, (_, srow) in enumerate(subset.iterrows()):
                    rows.append({
                        "region": region,
                        "forecast_date": srow["forecast_date"],
                        "target_hour": target_hour,
                        "predicted": float(preds[i]),
                        "actual": float(srow["target_demand"]) if not pd.isna(srow["target_demand"]) else np.nan,
                    })

        return pd.DataFrame(rows).sort_values(
            ["region", "forecast_date", "target_hour"]
        ).reset_index(drop=True)

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir) / self.name
        model_dir.mkdir(parents=True, exist_ok=True)
        for (region, hour), model in self.models.items():
            model.save_model(str(model_dir / f"{region}_hour_{hour:02d}.ubj"))
        meta = {"feature_names": self.feature_names,
                "regions": sorted({r for r, _ in self.models})}
        (model_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info("XGBoost models saved to %s (%d files).", model_dir, len(self.models))

    def load(self, model_dir: str | Path) -> "XGBoostForecaster":
        model_dir = Path(model_dir) / self.name
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        self.feature_names = meta["feature_names"]
        self.models = {}
        for path in sorted(model_dir.glob("*_hour_*.ubj")):
            # filename: {REGION}_hour_{HH}.ubj
            stem = path.stem  # e.g. "NB_hour_00"
            parts = stem.split("_hour_")
            if len(parts) != 2:
                continue
            region = parts[0]
            hour = int(parts[1])
            m = xgb.XGBRegressor()
            m.load_model(str(path))
            self.models[(region, hour)] = m
        logger.info("Loaded %d XGBoost models from %s.", len(self.models), model_dir)
        return self
