"""Peak-time metrics: peak hour, timing error and peak size."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.peak_time import daily_peaks, peak_hour_counts, peak_time_metrics


def day(state, issued, actual, predicted):
    return pd.DataFrame({"state": state, "issued_at": pd.Timestamp(issued), "horizon": np.arange(1, 25),
                         "actual": np.asarray(actual, float), "predicted": np.asarray(predicted, float)})


def profile(peak_hour, height=1000.0, base=500.0):
    values = np.full(24, base)
    values[peak_hour] = height
    return values


def test_perfect_forecast_has_zero_timing_error():
    frame = pd.concat([day("ON", "2025-01-01", profile(18), profile(18)),
                       day("ON", "2025-01-02", profile(8), profile(8))])
    metrics = peak_time_metrics(daily_peaks(frame))
    assert metrics["n_days"] == 2
    assert metrics["peak_hour_mae"] == 0 and metrics["peak_hour_exact"] == 1
    assert metrics["peak_size_mae_mw"] == 0 and metrics["peak_regret_mw"] == 0


def test_late_forecast_gives_positive_signed_error():
    days = daily_peaks(day("QC", "2025-01-01", profile(17), profile(19)))
    assert days.actual_peak_hour.iloc[0] == 17 and days.predicted_peak_hour.iloc[0] == 19
    metrics = peak_time_metrics(days)
    assert metrics["peak_hour_bias"] == 2 and metrics["peak_hour_mae"] == 2
    assert metrics["peak_hour_exact"] == 0 and metrics["peak_hour_within_1h"] == 0
    assert metrics["peak_hour_within_2h"] == 1


def test_hit_rates_use_absolute_error():
    frame = pd.concat([day("ON", "2025-01-01", profile(10), profile(11)),
                       day("ON", "2025-01-02", profile(10), profile(9)),
                       day("ON", "2025-01-03", profile(10), profile(13)),
                       day("ON", "2025-01-04", profile(10), profile(10))])
    metrics = peak_time_metrics(daily_peaks(frame))
    assert metrics["peak_hour_exact"] == 0.25
    assert metrics["peak_hour_within_1h"] == 0.75
    assert metrics["peak_hour_mae"] == pytest.approx((1 + 1 + 3 + 0) / 4)
    assert metrics["peak_hour_bias"] == pytest.approx((1 - 1 + 3 + 0) / 4)


def test_incomplete_days_are_not_scored():
    full = day("ON", "2025-01-01", profile(18), profile(18))
    short = day("ON", "2025-01-02", profile(18), profile(18)).iloc[:23]
    assert len(daily_peaks(pd.concat([full, short]))) == 1
    gap = day("ON", "2025-01-03", profile(18), profile(18))
    gap.loc[gap.index[5], "actual"] = np.nan
    assert daily_peaks(gap).empty


def test_provinces_with_the_same_issue_time_are_separate_days():
    frame = pd.concat([day("ON", "2025-01-01", profile(18), profile(18)),
                       day("QC", "2025-01-01", profile(8), profile(9))])
    days = daily_peaks(frame).set_index("state")
    assert days.loc["ON", "predicted_peak_hour"] == 18 and days.loc["QC", "predicted_peak_hour"] == 9


def test_regret_is_small_when_two_peaks_are_nearly_equal():
    actual = np.full(24, 500.0)
    actual[8], actual[18] = 1000.0, 999.0
    forecast = np.full(24, 500.0)
    forecast[8], forecast[18] = 990.0, 1000.0
    metrics = peak_time_metrics(daily_peaks(day("ON", "2025-01-01", actual, forecast)))
    assert metrics["peak_hour_mae"] == 10
    assert metrics["peak_regret_mw"] == pytest.approx(1.0)
    assert metrics["peak_regret_pct"] == pytest.approx(0.1)


def test_peak_size_error_compares_daily_maxima():
    metrics = peak_time_metrics(daily_peaks(day("ON", "2025-01-01", profile(18, 1000), profile(18, 900))))
    assert metrics["peak_size_mae_mw"] == 100 and metrics["peak_size_bias_mw"] == -100
    assert metrics["peak_size_mape"] == pytest.approx(10)


def test_ties_resolve_to_the_first_hour_and_counts_cover_all_hours():
    flat = np.full(24, 700.0)
    days = daily_peaks(day("ON", "2025-01-01", flat, flat))
    assert days.actual_peak_hour.iloc[0] == 0
    counts = peak_hour_counts(days, "actual_peak_hour")
    assert len(counts) == 24 and counts.sum() == 1 and counts.loc[0] == 1


def test_empty_input_returns_empty_metrics():
    empty = daily_peaks(pd.DataFrame({"state": [], "issued_at": [], "horizon": [], "actual": [], "predicted": []}))
    assert empty.empty and peak_time_metrics(empty) == {"n_days": 0}
