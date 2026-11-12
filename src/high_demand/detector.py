"""High-demand detection component.

Identifies forecast hours where predicted electricity demand exceeds
a configurable historical threshold per region.

Thresholds must be pre-computed from training data via
evaluator.compute_peak_thresholds() and passed in here.

Output columns per forecast row:
  - region
  - forecast_date
  - target_hour
  - predicted_demand_mw
  - threshold_mw
  - is_high_demand            (bool: predicted >= threshold)
  - actual_demand_mw          (if available)
  - forecast_error_mw         (actual - predicted, if actual available)
  - prob_exceed_threshold     (float, if probabilistic preds available)

Terminology note: "high demand" means predicted demand exceeds the
regional historical threshold.  This is a decision-support flag for
planners — it does NOT imply grid failure, outage, or capacity violation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Demand classification
# ---------------------------------------------------------------------------

def classify_demand_level(
    predicted: float,
    threshold: float,
) -> str:
    """Return 'High demand' or 'Normal demand' based on threshold comparison."""
    if np.isnan(predicted) or np.isnan(threshold):
        return "Unknown"
    return "High demand" if predicted >= threshold else "Normal demand"


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

def flag_high_demand(
    forecast_df: pd.DataFrame,
    peak_thresholds: dict[str, float],
    predicted_col: str = "predicted",
    actual_col: str = "actual",
    region_col: str = "region",
    hour_col: str = "target_hour",
    date_col: str = "forecast_date",
    prob_col: str | None = None,
) -> pd.DataFrame:
    """Annotate a forecast DataFrame with high-demand flags.

    Parameters
    ----------
    forecast_df : Prediction DataFrame with at minimum
                  [region, forecast_date, target_hour, predicted].
    peak_thresholds : Per-region thresholds from compute_peak_thresholds().
    predicted_col : Predicted demand column name.
    actual_col : Actual demand column name (optional; use None if unavailable).
    region_col : Region column name.
    hour_col : Forecast hour column name.
    date_col : Forecast date column name.
    prob_col : Column with probability of exceeding threshold (optional).

    Returns
    -------
    DataFrame with high-demand flags and classifications.
    """
    df = forecast_df.copy()

    # Map thresholds to each row
    df["threshold_mw"] = df[region_col].map(peak_thresholds)

    # High-demand flag
    df["is_high_demand"] = (df[predicted_col] >= df["threshold_mw"]).fillna(False)

    # Classification string
    df["demand_level"] = df.apply(
        lambda row: classify_demand_level(row[predicted_col], row["threshold_mw"]),
        axis=1,
    )

    # Rename for clarity
    df = df.rename(columns={predicted_col: "predicted_demand_mw"})

    # Forecast error (if actuals available)
    if actual_col in df.columns:
        df = df.rename(columns={actual_col: "actual_demand_mw"})
        df["forecast_error_mw"] = df["actual_demand_mw"] - df["predicted_demand_mw"]
    else:
        df["actual_demand_mw"] = float("nan")
        df["forecast_error_mw"] = float("nan")

    # Probability column (if probabilistic predictions available)
    if prob_col and prob_col in df.columns:
        df = df.rename(columns={prob_col: "prob_exceed_threshold"})
    else:
        df["prob_exceed_threshold"] = float("nan")

    # Summary log
    n_high = df["is_high_demand"].sum()
    n_total = len(df)
    logger.info(
        "High-demand detection: %d / %d forecast hours flagged (%.1f%%)",
        n_high, n_total, 100 * n_high / max(n_total, 1),
    )

    # Per-region breakdown
    for region in sorted(df[region_col].unique()):
        rdf = df[df[region_col] == region]
        n_high_r = rdf["is_high_demand"].sum()
        thr = peak_thresholds.get(region, float("nan"))
        logger.info(
            "  %s: %d high-demand hours (threshold=%.1f MW)",
            region, n_high_r, thr,
        )

    return df


# ---------------------------------------------------------------------------
# Summary statistics for the dashboard
# ---------------------------------------------------------------------------

def high_demand_summary(
    flagged_df: pd.DataFrame,
    region_col: str = "region",
    date_col: str = "forecast_date",
) -> pd.DataFrame:
    """Aggregate high-demand statistics by region.

    Returns DataFrame with columns:
        region, n_forecast_hours, n_high_demand_hours,
        high_demand_pct, peak_predicted_mw, threshold_mw,
        mean_error_mw (when actuals are available)
    """
    rows = []
    for region in sorted(flagged_df[region_col].unique()):
        rdf = flagged_df[flagged_df[region_col] == region]
        n_total = len(rdf)
        n_high = rdf["is_high_demand"].sum()
        rows.append(
            {
                "region": region,
                "n_forecast_hours": n_total,
                "n_high_demand_hours": int(n_high),
                "high_demand_pct": round(100 * n_high / max(n_total, 1), 1),
                "peak_predicted_mw": round(rdf["predicted_demand_mw"].max(), 1),
                "threshold_mw": round(rdf["threshold_mw"].iloc[0], 1) if "threshold_mw" in rdf.columns else float("nan"),
                "mean_error_mw": round(rdf["forecast_error_mw"].mean(), 1) if "forecast_error_mw" in rdf.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows)
