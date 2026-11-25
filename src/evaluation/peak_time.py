"""Daily peak timing: when does demand peak, and does the forecast pick the same hour?

A forecast day is one province and one issue time. Only days where all 24 target hours
were observed are scored, so the actual peak hour is well defined. Hours are local
clock hours of the forecast day (0-23); horizon h in the saved predictions is hour h-1.

Timing error = predicted peak hour - actual peak hour, in hours. Positive means the
forecast places the peak later than it occurred. The day is a fixed 0-23 window, so the
error is not wrapped around midnight.

Regret is how much real demand at the forecast's chosen hour falls short of the actual
daily peak. It keeps a forecast from being judged harshly on days with two nearly equal
peaks (for example a morning and an evening peak), where the argmax hour can flip on a
tiny difference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOURS = 24
KEYS = ["state", "issued_at"]


def daily_peaks(forecast, keys=KEYS, horizon="horizon", actual="actual", predicted="predicted"):
    """One row per fully observed forecast day with actual and predicted peak hour and size."""
    frame = forecast[[*keys, horizon, actual, predicted]].dropna()
    frame = frame.sort_values([*keys, horizon])
    counts = frame.groupby(keys)[horizon].transform("size")
    frame = frame.loc[counts.eq(HOURS)]
    columns = [*keys, "actual_peak_hour", "predicted_peak_hour", "actual_peak_mw",
               "predicted_peak_mw", "demand_at_predicted_peak_mw"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    days = frame[keys].drop_duplicates().reset_index(drop=True)
    shape = (len(days), HOURS)
    truth = frame[actual].to_numpy(float).reshape(shape)
    forecasts = frame[predicted].to_numpy(float).reshape(shape)
    rows = np.arange(len(days))
    actual_hour = truth.argmax(axis=1)
    predicted_hour = forecasts.argmax(axis=1)
    return days.assign(
        actual_peak_hour=actual_hour,
        predicted_peak_hour=predicted_hour,
        actual_peak_mw=truth[rows, actual_hour],
        predicted_peak_mw=forecasts[rows, predicted_hour],
        demand_at_predicted_peak_mw=truth[rows, predicted_hour],
    )


def peak_time_metrics(days):
    """Aggregate peak-timing and peak-size metrics over the days from ``daily_peaks``."""
    if days.empty:
        return {"n_days": 0}
    shift = days["predicted_peak_hour"] - days["actual_peak_hour"]
    size_error = days["predicted_peak_mw"] - days["actual_peak_mw"]
    regret = days["actual_peak_mw"] - days["demand_at_predicted_peak_mw"]
    return {
        "n_days": int(len(days)),
        "peak_hour_mae": float(shift.abs().mean()),
        "peak_hour_median_abs": float(shift.abs().median()),
        "peak_hour_bias": float(shift.mean()),
        "peak_hour_exact": float(shift.eq(0).mean()),
        "peak_hour_within_1h": float(shift.abs().le(1).mean()),
        "peak_hour_within_2h": float(shift.abs().le(2).mean()),
        "peak_size_mae_mw": float(size_error.abs().mean()),
        "peak_size_bias_mw": float(size_error.mean()),
        "peak_size_mape": float((size_error.abs() / days["actual_peak_mw"].abs()).mean() * 100),
        "peak_regret_mw": float(regret.mean()),
        "peak_regret_pct": float((regret / days["actual_peak_mw"].abs()).mean() * 100),
    }


def peak_hour_counts(days, column):
    """Number of days whose peak falls in each local hour, as a length-24 Series."""
    return days[column].value_counts().reindex(range(HOURS), fill_value=0).sort_index()
