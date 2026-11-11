"""Persistence++ — recent-observation-informed demand correction.

Inspired by Mouatadid et al. (2026).  Adapted for electricity demand.

Persistence++ exploits the autocorrelation in demand forecast errors:
if the model has been biased in the same direction over recent hours,
that pattern is used to adjust the current forecast.

Steps:
  1. Build a recent-error window from the last W forecast-observation pairs.
  2. Compute a weighted mean recent bias.
  3. Blend this with the climatological (Debias++) correction using α.
  4. Produce a corrected probability distribution.
  5. Apply isotonic regression for valid CDF.

This is an independent adaptation.  The original paper targets multi-week
weather subseasonal forecasts.  Here it targets hourly demand.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


class PersistencePlus:
    """Recent-error informed demand correction.

    Parameters
    ----------
    window_hours:
        Number of recent forecast hours to use for persistence estimation.
    alpha:
        Blending weight: 0 = pure Debias++, 1 = pure Persistence++.
        Default 0.5 (equal blend).
    n_quantiles:
        Number of quantile levels for output distribution.
    """

    def __init__(
        self,
        window_hours: int = 48,
        alpha: float = 0.5,
        n_quantiles: int = 9,
    ):
        self.window_hours = window_hours
        self.alpha = alpha
        self.n_quantiles = n_quantiles
        self.quantile_levels = np.linspace(0.1, 0.9, n_quantiles)
        # error_history[region] = deque of (predicted, actual) tuples
        self.error_history: dict[str, deque] = {}
        # Climatological std per region/hour from calibration
        self.clim_std: dict[str, dict[int, float]] = {}

    def fit(
        self,
        calib_df: pd.DataFrame,
        predicted_col: str = "predicted",
        actual_col: str = "actual",
        region_col: str = "region",
        hour_col: str = "target_hour",
    ) -> "PersistencePlus":
        """Estimate climatological spread from calibration data."""
        for region in calib_df[region_col].unique():
            self.clim_std[region] = {}
            rdf = calib_df[calib_df[region_col] == region]
            for hour in range(24):
                hdf = rdf[rdf[hour_col] == hour]
                residuals = (hdf[actual_col] - hdf[predicted_col]).dropna()
                self.clim_std[region][hour] = float(
                    max(residuals.std(), 1.0) if len(residuals) > 1 else 1.0
                )
            # Seed error history with calibration residuals (chronological order)
            self.error_history[region] = deque(
                zip(
                    calib_df[calib_df[region_col] == region][predicted_col].values,
                    calib_df[calib_df[region_col] == region][actual_col].values,
                ),
                maxlen=self.window_hours,
            )
        logger.info("PersistencePlus fitted on %d regions.", len(self.clim_std))
        return self

    def _persistence_bias(self, region: str) -> float:
        """Compute recent weighted mean error for a region."""
        history = self.error_history.get(region, deque())
        if not history:
            return 0.0
        preds, actuals = zip(*history)
        errors = np.array(actuals, dtype=float) - np.array(preds, dtype=float)
        # Exponential decay weights (most recent = highest weight)
        weights = np.exp(np.linspace(-2, 0, len(errors)))
        weights /= weights.sum()
        return float(np.dot(weights, errors))

    def correct(
        self,
        forecast_df: pd.DataFrame,
        debias_corrected_col: str = "predicted_corrected",
        region_col: str = "region",
        hour_col: str = "target_hour",
        actual_col: Optional[str] = "actual",
    ) -> pd.DataFrame:
        """Apply persistence correction and blend with Debias++ correction.

        Returns DataFrame with additional columns:
            predicted_pbc     — final blended corrected point estimate
            q{10,...,90}_pbc  — blended quantile predictions
        """
        df = forecast_df.copy()
        q_cols_in = [f"q{int(q * 100):02d}" for q in self.quantile_levels]
        q_cols_out = [f"q{int(q * 100):02d}_pbc" for q in self.quantile_levels]
        df["predicted_pbc"] = df[debias_corrected_col].copy()
        for col in q_cols_out:
            df[col] = float("nan")

        for region in df[region_col].unique():
            persist_bias = self._persistence_bias(region)
            mask = df[region_col] == region

            for hour in range(24):
                hmask = mask & (df[hour_col] == hour)
                if hmask.sum() == 0:
                    continue
                std = self.clim_std.get(region, {}).get(hour, 1.0)
                debias_pt = df.loc[hmask, debias_corrected_col].values
                blended = (1 - self.alpha) * debias_pt + self.alpha * (debias_pt + persist_bias)
                df.loc[hmask, "predicted_pbc"] = blended

                for q, col_in, col_out in zip(self.quantile_levels, q_cols_in, q_cols_out):
                    debias_q = df.loc[hmask, col_in].values if col_in in df.columns else debias_pt
                    persist_q = debias_pt + persist_bias + (q - 0.5) * 2 * std
                    blended_q = (1 - self.alpha) * debias_q + self.alpha * persist_q
                    df.loc[hmask, col_out] = blended_q

            # Update error history with any available actuals
            if actual_col and actual_col in df.columns:
                region_df = df[mask].sort_values(hour_col)
                for _, row in region_df.iterrows():
                    pred = row.get("predicted_pbc", np.nan)
                    actual = row.get(actual_col, np.nan)
                    if not (np.isnan(pred) or np.isnan(actual)):
                        self.error_history.setdefault(region, deque(maxlen=self.window_hours)).append(
                            (pred, actual)
                        )

        # Isotonic regression for monotone quantiles
        if SKLEARN_OK:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            for region in df[region_col].unique():
                for hour in range(24):
                    mask = (df[region_col] == region) & (df[hour_col] == hour)
                    for row_idx in df[mask].index:
                        vals = df.loc[row_idx, q_cols_out].values.astype(float)
                        if not np.any(np.isnan(vals)):
                            df.loc[row_idx, q_cols_out] = iso.fit_transform(
                                self.quantile_levels, vals
                            )

        return df


# ---------------------------------------------------------------------------
# Combined PBC pipeline
# ---------------------------------------------------------------------------

def run_pbc_pipeline(
    calib_df: pd.DataFrame,
    test_df_with_debias: pd.DataFrame,
    region_col: str = "region",
    hour_col: str = "target_hour",
) -> pd.DataFrame:
    """Run full Debias++ → Persistence++ correction pipeline.

    Parameters
    ----------
    calib_df : Validation-period predictions with actual demand.
    test_df_with_debias : Test predictions already processed by DebiasPlus.correct().

    Returns
    -------
    DataFrame with both Debias++ and Persistence++ corrected columns.
    """
    pp = PersistencePlus()
    pp.fit(calib_df, region_col=region_col, hour_col=hour_col)
    result = pp.correct(test_df_with_debias, region_col=region_col, hour_col=hour_col)
    return result
