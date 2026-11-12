"""Debias++ — systematic bias correction for electricity demand forecasts.

Inspired by the probabilistic bias correction (PBC) approach from:
  Mouatadid et al. (2026) "Advancing subseasonal forecasting with machine learning"
  Nature, https://www.nature.com/articles/s41586-026-11126-x

IMPORTANT DISCLAIMER:
  The original paper applies PBC to weather subseasonal forecasts.
  This module is an independent adaptation for electricity demand forecasting.
  It does NOT reproduce or replicate the original paper's experiments.
  It investigates whether the correction philosophy transfers to demand.

Debias++ corrects systematic errors in a point-forecast distribution
by learning the residual distribution from historical forecast-observation
pairs and applying that correction to new forecasts.

Steps:
  1. Collect historical (forecast, actual) pairs from the validation period.
  2. Estimate the conditional error distribution per region and hour bin.
  3. For a new forecast, shift the distribution by the estimated bias.
  4. Apply isotonic regression to enforce a valid CDF.

All fitting uses only training/validation data.  Test data is never used
to estimate bias.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


class DebiasPlus:
    """Demand-specific Debias++ correction.

    Estimates per-region, per-hour systematic bias from a held-out
    calibration set (typically the validation split) and corrects
    point forecasts accordingly.
    """

    def __init__(self, n_quantiles: int = 9):
        """
        Parameters
        ----------
        n_quantiles:
            Number of quantile levels for the corrected output distribution.
            Default 9 → [0.1, 0.2, ..., 0.9].
        """
        self.n_quantiles = n_quantiles
        self.quantile_levels = np.linspace(0.1, 0.9, n_quantiles)
        # bias_stats[region][hour] = {mean_bias, std_bias, n}
        self.bias_stats: dict[str, dict[int, dict]] = {}
        self._fitted = False

    def fit(
        self,
        calib_df: pd.DataFrame,
        predicted_col: str = "predicted",
        actual_col: str = "actual",
        region_col: str = "region",
        hour_col: str = "target_hour",
    ) -> DebiasPlus:
        """Estimate systematic bias from calibration (validation) data.

        Parameters
        ----------
        calib_df : DataFrame with forecast-observation pairs from the
                   validation period.  MUST NOT include test data.
        """
        for region in calib_df[region_col].unique():
            self.bias_stats[region] = {}
            rdf = calib_df[calib_df[region_col] == region]
            for hour in range(24):
                hdf = rdf[rdf[hour_col] == hour]
                residuals = (hdf[actual_col] - hdf[predicted_col]).dropna()
                if len(residuals) == 0:
                    self.bias_stats[region][hour] = {"mean_bias": 0.0, "std_bias": 1.0, "n": 0}
                else:
                    self.bias_stats[region][hour] = {
                        "mean_bias": float(residuals.mean()),
                        "std_bias": float(max(residuals.std(), 1.0)),
                        "n": len(residuals),
                    }
        self._fitted = True
        logger.info(
            "DebiasPlus fitted on %d regions, 24 hours each.",
            len(self.bias_stats),
        )
        return self

    def correct(
        self,
        forecast_df: pd.DataFrame,
        predicted_col: str = "predicted",
        region_col: str = "region",
        hour_col: str = "target_hour",
    ) -> pd.DataFrame:
        """Apply bias correction and generate a quantile prediction envelope.

        Returns the input DataFrame with additional columns:
            predicted_corrected   — bias-corrected point estimate
            q{10,20,...,90}       — quantile predictions
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before correct().")

        df = forecast_df.copy()
        df["predicted_corrected"] = df[predicted_col].copy()

        for q in self.quantile_levels:
            df[f"q{int(q * 100):02d}"] = float("nan")

        for region in df[region_col].unique():
            if region not in self.bias_stats:
                continue
            mask = df[region_col] == region
            for hour in range(24):
                hmask = mask & (df[hour_col] == hour)
                if hmask.sum() == 0:
                    continue
                stats = self.bias_stats[region][hour]
                mean_bias = stats["mean_bias"]
                std_bias = stats["std_bias"]

                point = df.loc[hmask, predicted_col].values
                corrected = point + mean_bias
                df.loc[hmask, "predicted_corrected"] = corrected

                # Gaussian quantile envelope around corrected forecast
                from scipy.stats import norm as _norm
                for q in self.quantile_levels:
                    qval = corrected + _norm.ppf(q) * std_bias
                    df.loc[hmask, f"q{int(q * 100):02d}"] = qval

        # Apply isotonic regression to quantile columns per region×hour to
        # enforce non-crossing (valid CDF ordering)
        if SKLEARN_OK:
            df = self._enforce_monotone_quantiles(df, region_col, hour_col)

        return df

    def _enforce_monotone_quantiles(
        self,
        df: pd.DataFrame,
        region_col: str,
        hour_col: str,
    ) -> pd.DataFrame:
        """Use isotonic regression to fix quantile crossing."""
        q_cols = [f"q{int(q * 100):02d}" for q in self.quantile_levels]
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        for region in df[region_col].unique():
            for hour in range(24):
                mask = (df[region_col] == region) & (df[hour_col] == hour)
                if mask.sum() == 0:
                    continue
                for i in range(len(df[mask])):
                    row_idx = df[mask].index[i]
                    quantile_vals = df.loc[row_idx, q_cols].values.astype(float)
                    if not np.any(np.isnan(quantile_vals)):
                        fixed = iso.fit_transform(
                            self.quantile_levels, quantile_vals
                        )
                        df.loc[row_idx, q_cols] = fixed
        return df
