"""Causal local-day HFED examples for the cross-dataset Grid-Aware extension.

Hourly load is conservatively treated as an interval [t,t+1h). At 23:00 local,
the newest input interval starts at 22:00. Newfoundland retains its :30 UTC
alignment by rebuilding its hourly demand from the cached native observations.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.data.hfed_loader import _read_hfed_csv, load_region
from src.data.holiday_client import build_holiday_dataframe
from src.models.stgnn.graph_builder import REGION_COORDS, _haversine_km
from src.data.regional_timezones import TIMEZONES

REGIONS = ["AB", "BC", "NB", "NL", "NS", "ON", "PE", "QC"]
WEATHER = ["temperature_c", "relative_humidity", "wind_speed"]

ARRAYS = ["dynamic", "covariates", "anchor", "actual", "nodes", "node_x", "edge_index", "edge_features",
          "center", "scale", "feature_center", "feature_scale"]


def prepared_data(root, cache):
    fingerprint = hashlib.sha256(Path(__file__).read_bytes() +
                                 Path(__file__).with_name("regional_timezones.py").read_bytes()).hexdigest()
    if (cache / "metadata.json").exists():
        meta = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        if meta["builder_sha256"] != fingerprint:
            raise ValueError("Builder changed: select a new cache directory")
        with np.load(cache / "examples.npz", allow_pickle=False) as archive:
            data = {key: archive[key] for key in ARRAYS}
            data["splits"] = {key: archive[f"split_{key}"] for key in meta["split_names"]}
        data.update({k: v for k, v in meta.items() if k not in ["builder_sha256", "split_names"]})
        data["records"] = pd.read_parquet(cache / "records.parquet")
        return data
    data = load_hfed_examples(root)
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "examples.npz.partial").open("wb") as handle:
        np.savez_compressed(handle, **{key: data[key] for key in ARRAYS},
                            **{f"split_{key}": value for key, value in data["splits"].items()})
    (cache / "examples.npz.partial").replace(cache / "examples.npz")
    data["records"].to_parquet(cache / "records.parquet", index=False)
    meta = {key: data[key] for key in ["features", "states", "horizons", "target_center", "target_scale", "exclusions"]}
    meta.update({"builder_sha256": fingerprint, "split_names": list(data["splits"])})
    sources = [root / "data/processed/hfed_processed.parquet", root / "data/weather/eccc/historical_weather.parquet",
               root / "data/holidays/canada_holidays.json", root / "data/hfed/2019-01-01_to_2024-12-31/NL/all.csv",
               *sorted((root / "data/external/hfed_holdout_2025").glob("*.csv"))]
    meta["source_sha256"] = {}
    for path in sources:
        with path.open("rb") as handle:
            meta["source_sha256"][str(path.relative_to(root))] = hashlib.file_digest(handle, "sha256").hexdigest()
    (cache / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return data


def hourly_native(readings, region):
    """Mean native observations in the region's hour-aligned UTC intervals."""
    offset = pd.Timedelta(minutes=30 if region == "NL" else 0)
    bucket = (readings.time_utc - offset).dt.floor("h") + offset
    return readings.groupby(bucket).demand_mw.mean().rename("demand_mw")


def local_day_grid(date, region):
    start = pd.Timestamp(date).tz_localize(TIMEZONES[region])
    stop = (pd.Timestamp(date) + pd.Timedelta(days=1)).tz_localize(TIMEZONES[region])
    targets = pd.date_range(start, stop, freq="h", inclusive="left").tz_convert("UTC").tz_localize(None)
    issue = (start - pd.Timedelta(hours=1)).tz_convert("UTC").tz_localize(None)
    history = pd.date_range(end=issue - pd.Timedelta(hours=1), periods=24, freq="h")
    return issue, history, targets


def known_lags(targets, issue, index):
    times = targets.to_numpy()[:, None] - np.arange(1, 8)[None] * np.timedelta64(24, "h")
    positions = index.get_indexer(times.ravel()).reshape(len(targets), 7)
    known = (positions >= 0) & (times + np.timedelta64(1, "h") <= issue.to_datetime64())
    return positions, known


def align_weather(observations, index):
    """Use only an exact or <=30-minute earlier observation; never a future value."""
    return observations.reindex(index, method="ffill", tolerance=pd.Timedelta(minutes=30))


def purged_splits(records):
    """Every fitted/selected target must mature before the next period's first issue."""
    dates = pd.DatetimeIndex(records.forecast_date)
    ends = pd.DatetimeIndex(records.target_start) + pd.Timedelta(hours=24)
    masks = {"train": dates < "2024-01-01",
             "validation": (dates >= "2024-01-01") & (dates < "2024-07-01"),
             "test2024": (dates >= "2024-07-01") & (dates < "2025-01-01"),
             "test2025": dates >= "2025-01-01"}
    original = {key: int(mask.sum()) for key, mask in masks.items()}
    masks["train"] &= ends <= records.loc[masks["validation"], "issued_at"].min()
    masks["validation"] &= ends <= records.loc[masks["test2024"], "issued_at"].min()
    return ({key: np.flatnonzero(mask) for key, mask in masks.items()},
            {f"purged_{key}_boundary": original[key] - int(masks[key].sum()) for key in ["train", "validation"]})


def calendar_values(index, region, holiday_dates):
    local = index.tz_localize("UTC").tz_convert(TIMEZONES[region])
    return np.column_stack([
        np.sin(local.hour * np.pi / 12), np.cos(local.hour * np.pi / 12),
        np.sin(local.dayofyear * 2 * np.pi / 366), np.cos(local.dayofyear * 2 * np.pi / 366),
        np.eye(7)[local.dayofweek], np.isin(local.date, list(holiday_dates)),
    ]).astype("float32")


def load_hfed_examples(root: Path):
    processed = pd.read_parquet(root / "data/processed/hfed_processed.parquet",
                               columns=["region", "time_utc", "demand_mw"])
    weather = pd.read_parquet(root / "data/weather/eccc/historical_weather.parquet")
    holiday_data = json.loads((root / "data/holidays/canada_holidays.json").read_text(encoding="utf-8"))
    holidays = build_holiday_dataframe({int(k): v for k, v in holiday_data.items()}, hfed_regions=REGIONS)
    if 2025 not in {int(k) for k in holiday_data}:
        raise ValueError("Cache 2025 holidays before constructing the fresh holdout")
    region_frames, holiday_sets = {}, {}
    first_validation_issue = min(local_day_grid("2024-01-01", region)[0] for region in REGIONS)
    for region in REGIONS:
        if region == "NL":
            native = load_region(region, root / "data/hfed/2019-01-01_to_2024-12-31", "DEMAND", [])
            old = hourly_native(native, region)
        else:
            old = processed.loc[processed.region.eq(region)].set_index("time_utc").demand_mw
        fresh_path = root / f"data/external/hfed_holdout_2025/{region}.csv"
        if not fresh_path.exists():
            raise ValueError(f"Fresh demand missing for {region}; do not silently drop a province")
        fresh = _read_hfed_csv(fresh_path).rename(columns={"TIME_PERIOD": "time_utc", "OBS_VALUE": "demand_mw"})
        if "UNIT_MEASURE" in fresh and not fresh.UNIT_MEASURE.eq("MW").all():
            raise ValueError(f"Unexpected demand units for {region}")
        new = hourly_native(fresh, region)
        load = pd.concat([old, new]).groupby(level=0).first().sort_index()
        load = load.where(load >= 0)
        index = pd.date_range(load.index.min(), load.index.max(), freq="h")
        frame = load.reindex(index).to_frame()
        region_weather = weather.loc[weather.region.eq(region)].set_index("time_utc")[WEATHER].sort_index()
        # Preserve native station time offsets and leave missing observations masked.
        frame[WEATHER] = align_weather(region_weather, index)
        holiday_sets[region] = set(holidays.loc[holidays.region.eq(region), "date"])
        cal = calendar_values(index, region, holiday_sets[region])
        obs = frame[["demand_mw", *WEATHER]].notna().to_numpy().astype("float32")
        for j, name in enumerate(["demand_mw", *WEATHER]):
            frame[f"{name}_observed"] = obs[:, j]
        for j in range(12):
            frame[f"calendar_{j}"] = cal[:, j]
        frame["is_train"] = ((index.tz_localize("UTC").tz_convert(TIMEZONES[region]).year < 2024)
                             & (index + pd.Timedelta(hours=1) <= first_validation_issue))
        region_frames[region] = frame
    feature_names = ["demand_mw", *WEATHER, *[f"{c}_observed" for c in ["demand_mw", *WEATHER]],
                     *[f"calendar_{j}" for j in range(12)]]
    train_values = np.concatenate([f.loc[f.is_train, feature_names].to_numpy() for f in region_frames.values()])
    scaler = RobustScaler().fit(train_values)
    target_scaler = RobustScaler().fit(train_values[:, :1])
    stats = {region: (float(f.loc[f.is_train, "demand_mw"].mean()),
                      float(f.loc[f.is_train, "demand_mw"].std())) for region, f in region_frames.items()}
    sequences, covariates, anchors, actuals, nodes, records = [], [], [], [], [], []
    exclusions = {"dst_day": 0, "insufficient_history": 0, "no_actual": 0}
    for node, region in enumerate(REGIONS):
        frame = region_frames[region]
        index = frame.index
        demand = frame.demand_mw.to_numpy()
        normalized = (demand - stats[region][0]) / stats[region][1]
        dynamic = np.nan_to_num(scaler.transform(frame[feature_names].to_numpy())).astype("float32")
        for date in pd.date_range("2019-01-08", "2025-12-31", freq="D"):
            issue, history, targets = local_day_grid(date, region)
            if len(targets) != 24:
                exclusions["dst_day"] += 1
                continue
            hi, ti = index.get_indexer(history), index.get_indexer(targets)
            if np.any(hi < 0) or np.isfinite(demand[hi]).mean() < 0.5:
                exclusions["insufficient_history"] += 1
                continue
            actual = np.where(ti >= 0, demand[np.maximum(ti, 0)], np.nan)
            if not np.isfinite(actual).any():
                exclusions["no_actual"] += 1
                continue
            lp, known = known_lags(targets, issue, index)
            known &= np.isfinite(normalized[np.maximum(lp, 0)])
            lags = np.where(known, normalized[np.maximum(lp, 0)], 0).astype("float32")
            anchor = np.take_along_axis(lags, known.argmax(axis=1)[:, None], axis=1)[:, 0]
            cal = calendar_values(targets, region, holiday_sets[region])
            sequences.append(dynamic[hi])
            covariates.append(np.column_stack([lags, known, cal]).astype("float32"))
            anchors.append(anchor)
            actuals.append(actual.astype("float32"))
            nodes.append(node)
            records.append({"region": region, "forecast_date": date, "issued_at": issue,
                            "target_start": targets[0]})
    records = pd.DataFrame(records)
    splits, purges = purged_splits(records)
    exclusions.update(purges)
    node_x = RobustScaler().fit_transform(np.array([REGION_COORDS[r] for r in REGIONS]))
    edges, attrs = [], []
    for i, a in enumerate(REGIONS):
        for j, b in enumerate(REGIONS):
            if i != j:
                distance = _haversine_km(*REGION_COORDS[a], *REGION_COORDS[b])
                if distance <= 2000:
                    edges.append([i, j])
                    attrs.append([distance])
    return {"dynamic": np.asarray(sequences), "covariates": np.asarray(covariates),
            "anchor": np.asarray(anchors), "actual": np.asarray(actuals), "nodes": np.asarray(nodes),
            "records": records, "splits": splits, "features": feature_names,
            "states": REGIONS, "horizons": list(range(1, 25)), "node_x": node_x,
            "edge_index": np.array(edges).T, "edge_features": RobustScaler().fit_transform(attrs),
            "center": np.array([stats[r][0] for r in REGIONS]),
            "scale": np.array([stats[r][1] for r in REGIONS]),
            "feature_center": scaler.center_, "feature_scale": scaler.scale_,
            "target_center": float(target_scaler.center_[0]), "target_scale": float(target_scaler.scale_[0]),
            "exclusions": exclusions}
