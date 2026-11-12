"""Load and normalize raw HFED CSV files into tidy DataFrames.

Responsibilities:
- Read all.csv (or per-variable CSV) for each downloaded region.
- Standardize column names and types.
- Normalize DATETIME_LOCAL_OFFSET to consistent float.
- Select the correct demand variable per region (from config).
- Select available generation-mix variables.
- Return one DataFrame per region (long → wide by variable).

No resampling, imputation, or feature engineering is done here.
Those are delegated to resampler.py and feature_engineering.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Low-level CSV reader
# ---------------------------------------------------------------------------

def _read_hfed_csv(csv_path: Path) -> pd.DataFrame:
    """Read a single HFED CSV, normalising types and column names."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)

    # Normalise DATETIME_LOCAL_OFFSET: '4', '4.0', '5', '5.0' → float
    if "DATETIME_LOCAL_OFFSET" in df.columns:
        df["DATETIME_LOCAL_OFFSET"] = pd.to_numeric(
            df["DATETIME_LOCAL_OFFSET"], errors="coerce"
        )
    if "DAYLIGHT_OFFSET" in df.columns:
        df["DAYLIGHT_OFFSET"] = pd.to_numeric(df["DAYLIGHT_OFFSET"], errors="coerce")

    # Parse UTC timestamp — handles plain ISO 8601, Z-suffix, and sub-second precision.
    # Try the fast ISO8601 path first; fall back to format='mixed'+utc=True only if
    # Z-suffix or timezone-aware strings are present (NB, NS sub-minute records).
    raw_times = df["TIME_PERIOD"]
    has_z = raw_times.str.contains("Z", na=False).any()
    if has_z:
        df["TIME_PERIOD"] = pd.to_datetime(
            raw_times, format="mixed", utc=True, errors="coerce"
        ).dt.tz_localize(None)
    else:
        df["TIME_PERIOD"] = pd.to_datetime(raw_times, format="ISO8601", errors="coerce")

    # Parse local timestamp — fast ISO8601 path for most regions, mixed for Z-suffix ones
    if "DATETIME_LOCAL" in df.columns:
        raw_local = df["DATETIME_LOCAL"]
        has_z_local = raw_local.str.contains("Z", na=False).any()
        if has_z_local:
            df["DATETIME_LOCAL"] = pd.to_datetime(
                raw_local, format="mixed", utc=True, errors="coerce"
            ).dt.tz_localize(None)
        else:
            df["DATETIME_LOCAL"] = pd.to_datetime(raw_local, format="ISO8601", errors="coerce")

    # Parse observation value
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")

    # Drop rows where the UTC timestamp could not be parsed
    n_before = len(df)
    df = df.dropna(subset=["TIME_PERIOD"])
    if len(df) < n_before:
        logger.warning(
            "%s: dropped %d rows with unparseable TIME_PERIOD",
            csv_path.name,
            n_before - len(df),
        )

    return df


# ---------------------------------------------------------------------------
# Region loader
# ---------------------------------------------------------------------------

def load_region(
    region_code: str,
    raw_data_dir: str | Path,
    demand_variable: str,
    generation_variables: list[str],
    counterpart_area_filter: str = "_Z",
) -> pd.DataFrame | None:
    """Load demand + generation variables for one HFED region.

    Returns a DataFrame with columns:
        time_utc, time_local, region, demand_mw,
        [generation variable columns in MW]

    or None if the data directory / file is not found.

    Parameters
    ----------
    region_code:
        Two-letter HFED region code, e.g. "ON".
    raw_data_dir:
        Root of downloaded data, e.g. "data/hfed/all_history".
    demand_variable:
        ENERGY_FLOWS code for the primary demand signal.
    generation_variables:
        List of ENERGY_FLOWS codes to include as features.
    counterpart_area_filter:
        Restrict to this COUNTERPART_AREA value.  "_Z" means region total.
        For ON, most relevant variables use "_Z".  Pass None to skip filter.
    """
    raw_data_dir = Path(raw_data_dir)
    region_dir = raw_data_dir / region_code
    all_csv = region_dir / "all.csv"
    if not all_csv.exists():
        logger.warning("Region %s: all.csv not found at %s", region_code, all_csv)
        return None

    logger.info("Loading region %s from %s", region_code, all_csv)
    df_raw = _read_hfed_csv(all_csv)

    # Apply counterpart-area filter
    if counterpart_area_filter is not None and "COUNTERPART_AREA" in df_raw.columns:
        df_raw = df_raw[df_raw["COUNTERPART_AREA"] == counterpart_area_filter]

    # Collect the variables we care about
    variables_wanted = [demand_variable] + list(generation_variables)
    df_raw = df_raw[df_raw["ENERGY_FLOWS"].isin(variables_wanted)].copy()

    if df_raw.empty:
        logger.warning(
            "Region %s: no rows for variables %s", region_code, variables_wanted
        )
        return None

    # Pivot to wide: one column per ENERGY_FLOWS variable
    df_wide = df_raw.pivot_table(
        index="TIME_PERIOD",
        columns="ENERGY_FLOWS",
        values="OBS_VALUE",
        aggfunc="mean",    # average if duplicates exist
    )
    df_wide.columns.name = None
    df_wide = df_wide.reset_index()

    # Bring in the first available DATETIME_LOCAL for each TIME_PERIOD
    df_local = (
        df_raw[["TIME_PERIOD", "DATETIME_LOCAL", "DATETIME_LOCAL_OFFSET"]]
        .drop_duplicates("TIME_PERIOD")
        .sort_values("TIME_PERIOD")
    )
    df_wide = df_wide.merge(df_local, on="TIME_PERIOD", how="left")

    # Rename for clarity
    df_wide = df_wide.rename(
        columns={
            "TIME_PERIOD": "time_utc",
            "DATETIME_LOCAL": "time_local",
        }
    )

    # Rename demand variable to standardised 'demand_mw'
    if demand_variable in df_wide.columns:
        df_wide = df_wide.rename(columns={demand_variable: "demand_mw"})
    else:
        logger.warning(
            "Region %s: demand variable %s not in pivoted columns %s",
            region_code,
            demand_variable,
            list(df_wide.columns),
        )
        df_wide["demand_mw"] = float("nan")

    # Add region identifier
    df_wide["region"] = region_code

    # Sort chronologically
    df_wide = df_wide.sort_values("time_utc").reset_index(drop=True)

    logger.info(
        "Region %s: %d rows, range %s to %s",
        region_code,
        len(df_wide),
        df_wide["time_utc"].min(),
        df_wide["time_utc"].max(),
    )
    return df_wide


# ---------------------------------------------------------------------------
# Multi-region loader
# ---------------------------------------------------------------------------

def load_all_regions(
    config: dict,
    raw_data_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load all configured regions.  Returns {region_code: DataFrame}."""
    if raw_data_dir is None:
        raw_data_dir = config["paths"]["raw_data_dir"]
    raw_data_dir = Path(raw_data_dir)

    regions = config["download"]["regions"]
    demand_vars = config["download"]["demand_variables"]
    gen_vars = config["download"]["generation_variables"]

    result = {}
    for region in regions:
        df = load_region(
            region_code=region,
            raw_data_dir=raw_data_dir,
            demand_variable=demand_vars[region],
            generation_variables=gen_vars.get(region, []),
        )
        if df is not None:
            result[region] = df
        else:
            logger.warning("Region %s could not be loaded; skipping.", region)

    return result


# ---------------------------------------------------------------------------
# Quick audit helper
# ---------------------------------------------------------------------------

def audit_regions(region_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a summary DataFrame describing each loaded region."""
    rows = []
    for region, df in region_dfs.items():
        demand = df["demand_mw"] if "demand_mw" in df.columns else pd.Series(dtype=float)
        rows.append(
            {
                "region": region,
                "n_rows": len(df),
                "start_utc": df["time_utc"].min(),
                "end_utc": df["time_utc"].max(),
                "demand_null_pct": round(demand.isna().mean() * 100, 2),
                "demand_min": round(demand.min(), 1) if not demand.empty else None,
                "demand_max": round(demand.max(), 1) if not demand.empty else None,
                "demand_mean": round(demand.mean(), 1) if not demand.empty else None,
                "columns": sorted(df.columns.tolist()),
            }
        )
    return pd.DataFrame(rows)
