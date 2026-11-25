"""Retrospective target-day temperature strata, never forecasting features."""
import numpy as np
import pandas as pd


def classify_target_days(records, train_ids, weather, states):
    labels = records[["issued_at", "region"]].rename(columns={"region": "state"}).copy()
    labels["target_day_temperature_c"] = np.nan
    thresholds = {}
    for state in states:
        ids = np.flatnonzero(records.region.eq(state))
        times = (records.iloc[ids].target_start.to_numpy()[:, None] +
                 np.arange(24)[None] * np.timedelta64(1, "h"))
        observed = weather.loc[weather.region.eq(state)].set_index("time_utc").temperature_c.sort_index()
        values = observed.reindex(times.ravel(), method="ffill", tolerance=pd.Timedelta(minutes=30)).to_numpy().reshape(-1, 24)
        counts = np.isfinite(values).sum(axis=1)
        means = np.divide(np.nansum(values, axis=1), counts,
                          out=np.full(len(ids), np.nan), where=counts >= 12)
        labels.loc[ids, "target_day_temperature_c"] = means
        training = labels.iloc[train_ids]
        subset = training.loc[training.state.eq(state), "target_day_temperature_c"].dropna()
        if subset.empty:
            raise ValueError(f"No training target-day temperature for {state}")
        low, high = np.quantile(subset, [.05, .95])
        thresholds[state] = {"cold_5pct_c": float(low), "hot_95pct_c": float(high)}
    labels["regime"] = "weather_unavailable"
    for state, limits in thresholds.items():
        valid = labels.state.eq(state) & labels.target_day_temperature_c.notna()
        labels.loc[valid, "regime"] = "typical"
        labels.loc[valid & labels.target_day_temperature_c.le(limits["cold_5pct_c"]), "regime"] = "cold_target_day"
        labels.loc[valid & labels.target_day_temperature_c.ge(limits["hot_95pct_c"]), "regime"] = "hot_target_day"
    return labels, thresholds
