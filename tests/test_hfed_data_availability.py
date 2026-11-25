"""Local-hour and meteorological alignment checks independent of model training."""
import numpy as np
import pandas as pd

from src.data.day_ahead_benchmark import local_day_grid, known_lags, hourly_native, align_weather, purged_splits


def test_issue_uses_completed_intervals_and_masks_unfinished_daily_lag():
    for region in ["ON", "NL"]:
        issue, history, targets = local_day_grid("2024-02-02", region)
        assert len(targets) == 24
        assert history[-1] + pd.Timedelta(hours=1) == issue
        index = pd.date_range(history[0] - pd.Timedelta(days=8), targets[-1], freq="h")
        _, known = known_lags(targets, issue, index)
        assert known[0, 0]
        assert not known[-1, 0]
        assert known[-1, 1]
        assert targets[0] == issue + pd.Timedelta(hours=1)


def test_newfoundland_resampling_preserves_half_hour_utc_alignment():
    readings = pd.DataFrame({"time_utc": pd.to_datetime(["2024-02-02 03:30", "2024-02-02 03:50", "2024-02-02 04:10"]),
                             "demand_mw": [100.0, 110.0, 120.0]})
    result = hourly_native(readings, "NL")
    assert result.index[0] == pd.Timestamp("2024-02-02 03:30")
    assert result.iloc[0] == 110.0


def test_dst_days_are_explicitly_non_24_hour_targets():
    assert len(local_day_grid("2024-03-10", "ON")[2]) == 23
    assert len(local_day_grid("2024-11-03", "ON")[2]) == 25


def test_weather_alignment_never_interpolates_a_future_observation():
    observed = pd.DataFrame({"temperature_c": [10.0, np.nan, 30.0]},
                            index=pd.date_range("2024-01-01 20:00", periods=3, freq="h"))
    targets = pd.to_datetime(["2024-01-01 20:30", "2024-01-01 21:00", "2024-01-01 21:30"])
    aligned = align_weather(observed, targets)
    assert aligned.iloc[0, 0] == 10.0
    assert np.isnan(aligned.iloc[1, 0])
    assert np.isnan(aligned.iloc[2, 0])


def test_split_targets_mature_before_the_earliest_next_period_issue():
    records = []
    for region in ["BC", "ON", "NL"]:
        for date in ["2023-12-30", "2023-12-31", "2024-01-01", "2024-06-29", "2024-06-30", "2024-07-01", "2025-01-01"]:
            issue, _, targets = local_day_grid(date, region)
            records.append({"region": region, "forecast_date": pd.Timestamp(date), "issued_at": issue,
                            "target_start": targets[0]})
    frame = pd.DataFrame(records)
    splits, purges = purged_splits(frame)
    assert purges["purged_train_boundary"] == 3
    assert purges["purged_validation_boundary"] == 3
    assert (frame.iloc[splits["train"]].target_start + pd.Timedelta(hours=24)).max() <= frame.iloc[splits["validation"]].issued_at.min()
    assert (frame.iloc[splits["validation"]].target_start + pd.Timedelta(hours=24)).max() <= frame.iloc[splits["test2024"]].issued_at.min()
