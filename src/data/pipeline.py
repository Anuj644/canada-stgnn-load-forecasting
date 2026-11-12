"""End-to-end data preprocessing pipeline.

Orchestrates:
1. Load raw HFED data for all configured regions.
2. Resample sub-hourly regions to hourly UTC.
3. Fetch and cache Canadian holidays from Nager.Date.
4. Build features (calendar, holiday, lags, rolling).
5. Combine all regions into a single multi-region DataFrame.
6. Apply chronological train/val/test splits.
7. Fit scalers on training data only; transform val/test.
8. Save the processed dataset to Parquet.

Usage:
    python -m src.data.pipeline

Or import and call run_pipeline(config).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.data.hfed_loader import load_all_regions, audit_regions
from src.data.resampler import resample_all_regions
from src.data.holiday_client import get_holiday_df_for_data_range
from src.data.feature_engineering import build_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def apply_splits(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    time_col: str = "time_utc",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame chronologically into train / val / test.

    Parameters
    ----------
    df : Multi-region hourly DataFrame, sorted by time_utc.
    train_end : Last date of training period (inclusive), 'YYYY-MM-DD'.
    val_end : Last date of validation period (inclusive), 'YYYY-MM-DD'.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    train_end_ts = pd.Timestamp(train_end) + pd.Timedelta(hours=23)
    val_end_ts = pd.Timestamp(val_end) + pd.Timedelta(hours=23)

    train = df[df[time_col] <= train_end_ts].copy()
    val = df[(df[time_col] > train_end_ts) & (df[time_col] <= val_end_ts)].copy()
    test = df[df[time_col] > val_end_ts].copy()

    logger.info(
        "Split: train=%d rows (%s to %s), val=%d rows (%s to %s), test=%d rows (%s to %s)",
        len(train), train[time_col].min(), train[time_col].max(),
        len(val), val[time_col].min() if not val.empty else "N/A", val[time_col].max() if not val.empty else "N/A",
        len(test), test[time_col].min() if not test.empty else "N/A", test[time_col].max() if not test.empty else "N/A",
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Scaler (fitted on training data only)
# ---------------------------------------------------------------------------

class RegionalStandardScaler:
    """Per-region, per-feature standard scaler.

    Fitted on training data; applied consistently to val and test.
    Stores means and stds so they can be saved and reused.
    """

    def __init__(self):
        self.stats: dict[str, dict[str, tuple[float, float]]] = {}
        # stats[region][feature] = (mean, std)

    def fit(self, df: pd.DataFrame, feature_cols: list[str], region_col: str = "region") -> "RegionalStandardScaler":
        for region in df[region_col].unique():
            self.stats[region] = {}
            subset = df[df[region_col] == region]
            for col in feature_cols:
                if col in subset.columns:
                    mean = float(subset[col].mean())
                    std = float(subset[col].std())
                    if std == 0 or np.isnan(std):
                        std = 1.0
                    self.stats[region][col] = (mean, std)
        return self

    def transform(self, df: pd.DataFrame, feature_cols: list[str], region_col: str = "region") -> pd.DataFrame:
        df = df.copy()
        # Cast all scale_cols to float64 upfront so integer columns accept float values
        for col in feature_cols:
            if col in df.columns and not pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype(np.float64)
        for region in df[region_col].unique():
            mask = df[region_col] == region
            if region not in self.stats:
                logger.warning("Region %s not seen during scaler fitting; skipping.", region)
                continue
            for col in feature_cols:
                if col in df.columns and col in self.stats[region]:
                    mean, std = self.stats[region][col]
                    df.loc[mask, col] = (df.loc[mask, col] - mean) / std
        return df

    def inverse_transform_demand(
        self, values: np.ndarray, region: str, demand_col: str = "demand_mw"
    ) -> np.ndarray:
        if region not in self.stats or demand_col not in self.stats[region]:
            return values
        mean, std = self.stats[region][demand_col]
        return values * std + mean

    def to_dict(self) -> dict:
        return self.stats

    @classmethod
    def from_dict(cls, d: dict) -> "RegionalStandardScaler":
        scaler = cls()
        scaler.stats = d
        return scaler


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    config: dict,
    raw_data_dir: Optional[str | Path] = None,
    force_holiday_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RegionalStandardScaler]:
    """Run the full preprocessing pipeline.

    Parameters
    ----------
    config : Loaded config dict.
    raw_data_dir : Override for raw data directory.
    force_holiday_refresh : Re-fetch holidays from API even if cached.

    Returns
    -------
    (train_df, val_df, test_df, scaler)

    Each DataFrame has all features attached.
    Demand is NOT scaled in the returned DataFrames (to keep units
    interpretable); a separate `scaled` copy can be produced via
    scaler.transform() if needed by the model training code.
    """
    processed_dir = Path(config["paths"]["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load raw HFED data ---
    logger.info("=== Step 1: Loading raw HFED data ===")
    region_dfs = load_all_regions(config, raw_data_dir=raw_data_dir)
    if not region_dfs:
        raise RuntimeError(
            "No region data loaded.  Run the downloader first:\n"
            "  python hfed_dataset_downloader.py"
        )

    audit = audit_regions(region_dfs)
    logger.info("Region audit:\n%s", audit.to_string())

    # --- 2. Resample to hourly ---
    logger.info("=== Step 2: Resampling to hourly UTC ===")
    min_coverage = config["preprocessing"].get("min_hourly_coverage", 0.5)
    region_dfs = resample_all_regions(region_dfs, min_coverage=min_coverage)

    # --- 3. Determine dataset year range for holiday fetching ---
    all_years = set()
    for df in region_dfs.values():
        years = pd.to_datetime(df["time_utc"]).dt.year.unique()
        all_years.update(years.tolist())
    start_year = min(all_years)
    end_year = max(all_years)

    # --- 4. Fetch holidays ---
    logger.info("=== Step 3: Fetching Canadian holidays %d–%d ===", start_year, end_year)
    holiday_df = get_holiday_df_for_data_range(
        start_year=start_year,
        end_year=end_year,
        cache_path=config["holidays"]["cache_file"],
        hfed_regions=list(region_dfs.keys()),
        force_refresh=force_holiday_refresh,
    )
    logger.info("Holiday entries: %d", len(holiday_df))

    # --- 5. Feature engineering ---
    logger.info("=== Step 4: Building features ===")
    lag_hours = config["features"].get("demand_lags", [1, 2, 3, 6, 12, 24, 48, 168])
    rolling_windows = config["features"].get("rolling_windows", [24, 168])
    demand_min = config["preprocessing"].get("demand_min_valid", 0.0)
    demand_max_mult = config["preprocessing"].get("demand_max_multiplier", 5.0)

    featured_dfs = []
    for region, df in region_dfs.items():
        logger.info("Building features for region %s ...", region)
        df_feat = build_features(
            df,
            holiday_df=holiday_df,
            lag_hours=lag_hours,
            rolling_windows=rolling_windows,
            demand_min=demand_min,
            demand_max_multiplier=demand_max_mult,
        )
        featured_dfs.append(df_feat)

    combined = pd.concat(featured_dfs, ignore_index=True)
    combined = combined.sort_values(["time_utc", "region"]).reset_index(drop=True)

    # --- 6. Save un-scaled processed dataset ---
    out_path = processed_dir / "hfed_processed.parquet"
    combined.to_parquet(out_path, index=False)
    logger.info("Saved processed dataset to %s (%d rows).", out_path, len(combined))

    # --- 7. Chronological splits ---
    logger.info("=== Step 5: Applying chronological splits ===")
    train_end = config["splits"]["train_end"]
    val_end = config["splits"]["val_end"]
    train, val, test = apply_splits(combined, train_end=train_end, val_end=val_end)

    # --- 8. Fit scaler on training data ---
    logger.info("=== Step 6: Fitting scaler on training data ===")
    # Identify numeric feature columns (excluding flags, identifiers, raw timestamps)
    skip_cols = {
        "time_utc", "time_local", "region", "DATETIME_LOCAL_OFFSET",
        "demand_flag_missing", "demand_flag_negative", "demand_flag_extreme",
        "is_holiday", "is_weekend", "is_day_before_holiday", "is_day_after_holiday",
        "holiday_name", "year",
    }
    feature_cols = [
        c for c in combined.columns
        if c not in skip_cols
        and pd.api.types.is_numeric_dtype(combined[c])
    ]

    scaler = RegionalStandardScaler()
    scaler.fit(train, feature_cols=feature_cols)

    # Save split DataFrames
    train.to_parquet(processed_dir / "train.parquet", index=False)
    val.to_parquet(processed_dir / "val.parquet", index=False)
    test.to_parquet(processed_dir / "test.parquet", index=False)

    # Save scaler statistics
    import json
    scaler_path = processed_dir / "scaler_stats.json"
    scaler_path.write_text(json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")
    logger.info("Saved scaler stats to %s.", scaler_path)

    logger.info(
        "Pipeline complete.  Regions: %s  |  Train: %d  |  Val: %d  |  Test: %d",
        list(region_dfs.keys()),
        len(train),
        len(val),
        len(test),
    )

    return train, val, test, scaler


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


if __name__ == "__main__":
    _setup_logging()
    cfg = load_config()
    run_pipeline(cfg)
