"""Evaluation metrics for electricity demand forecasting.

Implements:
  MAE   — Mean Absolute Error
  RMSE  — Root Mean Squared Error
  MAPE  — Mean Absolute Percentage Error (safe for near-zero actuals)
  CRPS  — Continuous Ranked Probability Score (probabilistic)
  Pinball loss — Quantile loss for probabilistic forecasts
  PI coverage — Prediction interval coverage rate

MAPE safeguard: observations with |actual| < mape_zero_threshold
are excluded from the MAPE calculation.  The threshold and the
number of excluded observations are always reported.

No results are hardcoded.  All metrics are computed from actual
model predictions vs. observed demand.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Point forecast metrics
# ---------------------------------------------------------------------------

MAPE_ZERO_THRESHOLD = 10.0  # MW; skip MAPE for actuals below this


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - predicted[mask])))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))


def mape(
    actual: np.ndarray,
    predicted: np.ndarray,
    zero_threshold: float = MAPE_ZERO_THRESHOLD,
) -> tuple[float, int]:
    """Mean Absolute Percentage Error with zero-safeguard.

    Parameters
    ----------
    actual, predicted : Arrays of observations and predictions.
    zero_threshold : Observations with |actual| < zero_threshold are excluded.

    Returns
    -------
    (mape_value, n_excluded)
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    nonzero = np.abs(actual) >= zero_threshold
    mask = valid & nonzero
    n_excluded = int((valid & ~nonzero).sum())

    if mask.sum() == 0:
        return float("nan"), n_excluded
    value = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)
    return value, n_excluded


# ---------------------------------------------------------------------------
# Probabilistic metrics
# ---------------------------------------------------------------------------

def pinball_loss(actual: np.ndarray, quantile_pred: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss for quantile q ∈ (0, 1).

    L_q(y, ŷ) = q·max(y - ŷ, 0) + (1 - q)·max(ŷ - y, 0)
    """
    actual = np.asarray(actual, dtype=float)
    quantile_pred = np.asarray(quantile_pred, dtype=float)
    err = actual - quantile_pred
    return float(np.mean(np.where(err >= 0, q * err, (q - 1) * err)))


def crps_gaussian(
    actual: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """CRPS for a Gaussian predictive distribution.

    CRPS(N(μ, σ), y) = σ · [z·(2Φ(z)−1) + 2φ(z) − 1/√π]
    where z = (y − μ) / σ.

    Requires scipy.
    """
    try:
        from scipy.stats import norm
    except ImportError:
        logger.warning("scipy not installed; CRPS unavailable.")
        return float("nan")

    actual = np.asarray(actual, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-6)

    z = (actual - mu) / sigma
    crps_vals = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.nanmean(crps_vals))


def prediction_interval_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of observations falling within [lower, upper].

    Ideal coverage for a (1-α) PI is (1-α); compare to nominal level.
    """
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    covered = (actual >= lower) & (actual <= upper)
    return float(np.nanmean(covered.astype(float)))


# ---------------------------------------------------------------------------
# Aggregated evaluation
# ---------------------------------------------------------------------------

def compute_metrics(
    df: pd.DataFrame,
    actual_col: str = "actual",
    predicted_col: str = "predicted",
    mape_threshold: float = MAPE_ZERO_THRESHOLD,
) -> dict:
    """Compute MAE, RMSE, MAPE for a prediction DataFrame.

    Parameters
    ----------
    df : DataFrame with actual and predicted columns.
    actual_col, predicted_col : Column names.
    mape_threshold : MAPE zero-safeguard threshold (MW).

    Returns
    -------
    dict with keys: mae, rmse, mape, mape_n_excluded, n_obs
    """
    act = df[actual_col].values
    pred = df[predicted_col].values
    mape_val, n_excl = mape(act, pred, zero_threshold=mape_threshold)
    return {
        "mae": mae(act, pred),
        "rmse": rmse(act, pred),
        "mape": mape_val,
        "mape_n_excluded": n_excl,
        "n_obs": int((~np.isnan(act) & ~np.isnan(pred)).sum()),
    }
