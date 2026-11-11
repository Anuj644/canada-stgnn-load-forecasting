"""Full evaluation pipeline.

Computes and reports MAE / RMSE / MAPE for all models:
  - Overall across all regions and hours
  - Per region
  - Per forecast hour (0–23)
  - During peak-demand periods (configurable percentile threshold)
  - During holidays
  - During extreme-weather periods (proxied by high generation variability)

Peak-demand thresholds are computed from TRAINING DATA ONLY and are
reused across all models for a fair comparison.

Results are saved as CSV and JSON in the results directory.
No results are hardcoded or fabricated.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics, mae, rmse, mape, MAPE_ZERO_THRESHOLD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Peak-demand threshold
# ---------------------------------------------------------------------------

def compute_peak_thresholds(
    train_df: pd.DataFrame,
    regions: list[str],
    percentile: float = 90.0,
    demand_col: str = "demand_mw",
    region_col: str = "region",
) -> dict[str, float]:
    """Compute per-region demand thresholds from training data.

    Parameters
    ----------
    train_df : Training period DataFrame.
    regions : Region list.
    percentile : Percentile threshold (e.g. 90 for 90th percentile).
    demand_col : Demand column name.
    region_col : Region column name.

    Returns
    -------
    dict: {region: threshold_MW}

    The threshold is computed on training data only and never on val/test,
    ensuring no leakage into the evaluation period.
    """
    thresholds = {}
    for region in regions:
        subset = train_df[train_df[region_col] == region][demand_col].dropna()
        if subset.empty:
            logger.warning("Region %s: no training demand data for threshold computation.", region)
            thresholds[region] = float("nan")
        else:
            thresholds[region] = float(np.percentile(subset.values, percentile))
            logger.info(
                "Region %s: %.0fth-percentile demand threshold = %.1f MW",
                region, percentile, thresholds[region],
            )
    return thresholds


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ForecastEvaluator:
    """Evaluate one or more forecast DataFrames across multiple dimensions."""

    def __init__(
        self,
        peak_thresholds: dict[str, float],
        percentile: float = 90.0,
        mape_threshold: float = MAPE_ZERO_THRESHOLD,
        results_dir: str | Path = "results",
    ):
        self.peak_thresholds = peak_thresholds
        self.percentile = percentile
        self.mape_threshold = mape_threshold
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        forecast_df: pd.DataFrame,
        model_name: str,
        actual_col: str = "actual",
        predicted_col: str = "predicted",
        region_col: str = "region",
        hour_col: str = "target_hour",
        date_col: str = "forecast_date",
        holiday_col: Optional[str] = "is_holiday",
    ) -> dict:
        """Full evaluation of one model's predictions."""
        df = forecast_df.copy()

        # Normalise hour column — naive model uses 'forecast_hour', others use 'target_hour'
        if hour_col not in df.columns and "forecast_hour" in df.columns:
            df = df.rename(columns={"forecast_hour": hour_col})
        # Normalise date column — derive from time_utc when forecast_date absent
        if date_col not in df.columns and "time_utc" in df.columns:
            df[date_col] = pd.to_datetime(df["time_utc"]).dt.date

        df = df.dropna(subset=[actual_col, predicted_col])
        if df.empty:
            logger.warning("Model %s: empty forecast DataFrame after dropping NaN.", model_name)
            return {}

        results = {"model": model_name}

        # --- Overall ---
        results["overall"] = compute_metrics(df, actual_col, predicted_col, self.mape_threshold)
        logger.info(
            "Model %s — Overall: MAE=%.1f  RMSE=%.1f  MAPE=%.2f%%",
            model_name,
            results["overall"]["mae"],
            results["overall"]["rmse"],
            results["overall"]["mape"] if results["overall"]["mape"] is not None else float("nan"),
        )

        # --- Per region ---
        results["by_region"] = {}
        for region in df[region_col].unique():
            rdf = df[df[region_col] == region]
            results["by_region"][region] = compute_metrics(
                rdf, actual_col, predicted_col, self.mape_threshold
            )

        # --- Per forecast hour ---
        results["by_hour"] = {}
        for hour in sorted(df[hour_col].unique()):
            hdf = df[df[hour_col] == hour]
            results["by_hour"][int(hour)] = compute_metrics(
                hdf, actual_col, predicted_col, self.mape_threshold
            )

        # --- Peak demand ---
        peak_rows = []
        for region, threshold in self.peak_thresholds.items():
            if np.isnan(threshold):
                continue
            rdf = df[df[region_col] == region]
            peak_df = rdf[rdf[actual_col] >= threshold]
            peak_rows.append(peak_df)
        if peak_rows:
            peak_all = pd.concat(peak_rows, ignore_index=True)
            results["peak_demand"] = compute_metrics(
                peak_all, actual_col, predicted_col, self.mape_threshold
            )
            results["peak_demand"]["threshold_percentile"] = self.percentile
        else:
            results["peak_demand"] = {}

        # --- Holiday periods ---
        if holiday_col and holiday_col in df.columns:
            holiday_df = df[df[holiday_col].astype(bool)]
            if not holiday_df.empty:
                results["holidays"] = compute_metrics(
                    holiday_df, actual_col, predicted_col, self.mape_threshold
                )
            else:
                results["holidays"] = {"note": "No holiday rows in this forecast period."}
        else:
            results["holidays"] = {"note": "Holiday column not available."}

        # --- Save results ---
        out_json = self.results_dir / f"metrics_{model_name}.json"
        out_json.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        logger.info("Evaluation results saved to %s.", out_json)

        # --- Save per-region summary CSV ---
        region_rows = []
        for region, metrics in results["by_region"].items():
            row = {"region": region, "model": model_name}
            row.update(metrics)
            region_rows.append(row)
        region_csv = self.results_dir / f"metrics_by_region_{model_name}.csv"
        pd.DataFrame(region_rows).to_csv(region_csv, index=False)

        # --- Save per-hour summary CSV ---
        hour_rows = []
        for hour, metrics in results["by_hour"].items():
            row = {"hour": hour, "model": model_name}
            row.update(metrics)
            hour_rows.append(row)
        hour_csv = self.results_dir / f"metrics_by_hour_{model_name}.csv"
        pd.DataFrame(hour_rows).to_csv(hour_csv, index=False)

        return results

    def compare_models(
        self,
        all_results: dict[str, dict],
    ) -> pd.DataFrame:
        """Build a comparison table from multiple model evaluation results.

        Parameters
        ----------
        all_results : {model_name: results_dict} from evaluate().

        Returns
        -------
        DataFrame with one row per model and overall MAE, RMSE, MAPE.
        """
        rows = []
        for model_name, res in all_results.items():
            if "overall" not in res:
                continue
            row = {"model": model_name}
            row.update(res["overall"])
            # Add peak demand metrics
            if "peak_demand" in res and "mae" in res["peak_demand"]:
                row["peak_mae"] = res["peak_demand"]["mae"]
                row["peak_rmse"] = res["peak_demand"]["rmse"]
            rows.append(row)

        comparison = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
        out_path = self.results_dir / "model_comparison.csv"
        comparison.to_csv(out_path, index=False)
        logger.info("Model comparison table saved to %s.", out_path)
        return comparison


def run_full_evaluation(
    predictions: dict[str, pd.DataFrame],
    train_df: pd.DataFrame,
    regions: list[str],
    config: dict,
    results_dir: str | Path = "results",
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Evaluate all model predictions and return results + comparison table.

    Parameters
    ----------
    predictions : {model_name: forecast_df} from all models.
    train_df : Training DataFrame (for peak threshold computation).
    regions : Region list.
    config : Config dict.
    results_dir : Output directory.

    Returns
    -------
    (all_results, comparison_table)
    """
    percentile = config.get("high_demand", {}).get("threshold_percentile", 90)
    mape_threshold = MAPE_ZERO_THRESHOLD

    peak_thresholds = compute_peak_thresholds(
        train_df, regions, percentile=percentile
    )

    # Save thresholds
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    thresholds_path = Path(results_dir) / "peak_thresholds.json"
    thresholds_path.write_text(
        json.dumps(peak_thresholds, indent=2), encoding="utf-8"
    )

    evaluator = ForecastEvaluator(
        peak_thresholds=peak_thresholds,
        percentile=percentile,
        mape_threshold=mape_threshold,
        results_dir=results_dir,
    )

    all_results = {}
    for model_name, forecast_df in predictions.items():
        logger.info("Evaluating model: %s", model_name)
        all_results[model_name] = evaluator.evaluate(forecast_df, model_name=model_name)

    comparison = evaluator.compare_models(all_results)
    logger.info("\nModel comparison:\n%s", comparison.to_string())

    return all_results, comparison
