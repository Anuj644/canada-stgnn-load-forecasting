"""Serve the verified HFED checkpoints with the benchmark's causal features.

Inference is separate from fitting. Forecast-day observations are comparison
labels only; missing target hours never prevent producing a full 24-hour curve.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.data.day_ahead_benchmark import (
    REGIONS, WEATHER, align_weather, calendar_values, hourly_native,
    known_lags, local_day_grid, prepared_data,
)
from src.data.holiday_client import build_holiday_dataframe, fetch_holidays
from src.data.regional_timezones import TIMEZONES

MAIN_RUN = "hfed_extension_eccc_v4"
CACHE = "data/processed/hfed_extension_eccc_v2"


@dataclass(frozen=True)
class ForecastSpec:
    label: str
    model: str
    folder: str = MAIN_RUN
    features: str = "demand_weather_calendar"


SPECS = {
    "cnn": ForecastSpec("Selected residual CNN", "residual_identity"),
    "cnn_weather": ForecastSpec("Residual CNN + weather channels", "residual_weather", "hfed_cnn_tuning_v1"),
    "graph": ForecastSpec("Seasonal GRU · geographic graph", "seasonal_grid_aware"),
    "calendar": ForecastSpec("Seasonal GRU · demand + weather + calendar", "seasonal_identity"),
    "weather": ForecastSpec("Seasonal GRU · demand + weather", "seasonal_identity", "hfed_ablation_weather_v4", "demand_weather"),
    "demand": ForecastSpec("Seasonal GRU · demand only", "seasonal_identity", "hfed_ablation_demand_v4", "demand"),
    "lstm": ForecastSpec("Grid-Aware LSTM · HFED adaptation", "grid_aware_lstm"),
    "gru": ForecastSpec("Grid-Aware GRU · HFED adaptation", "grid_aware_gru"),
    "xgboost": ForecastSpec("XGBoost", "xgboost"),
    "daily": ForecastSpec("Available daily naive", "naive_day_available"),
    "weekly": ForecastSpec("Weekly naive", "naive_week"),
    "temporal_lstm": ForecastSpec("Temporal-only LSTM", "temporal_lstm", "hfed_temporal_controls_v1"),
    "temporal_gru": ForecastSpec("Temporal-only GRU", "temporal_gru", "hfed_temporal_controls_v1"),
}


def feature_mask(dynamic, covariates, names, feature_set):
    dynamic, covariates = dynamic.copy(), covariates.copy()
    if feature_set == "demand":
        for i, name in enumerate(names):
            if any(name.startswith(prefix) for prefix in WEATHER):
                dynamic[..., i] = 0
    if feature_set != "demand_weather_calendar":
        for i, name in enumerate(names):
            if name.startswith("calendar_"):
                dynamic[..., i] = 0
        covariates[..., 14:] = 0
    return dynamic, covariates


def build_request(data, region, forecast_date, demand, weather, holiday_dates):
    """Construct one sample using frozen training scalers and no future inputs."""
    issue, history, targets = local_day_grid(forecast_date, region)
    if len(targets) != 24:
        raise ValueError("This model supports 24-hour days. Select a date outside the daylight-saving transition.")
    start = targets[0] - pd.Timedelta(days=8)
    index = pd.date_range(start, targets[-1], freq="h")
    demand = demand.groupby(level=0).mean().sort_index().where(lambda x: x >= 0)
    frame = demand.reindex(index).rename("demand_mw").to_frame()
    frame[WEATHER] = align_weather(weather[WEATHER].sort_index(), index)
    observed = frame[["demand_mw", *WEATHER]].notna().to_numpy(dtype="float32")
    for j, name in enumerate(["demand_mw", *WEATHER]):
        frame[f"{name}_observed"] = observed[:, j]
    calendar = calendar_values(index, region, holiday_dates)
    for j in range(12):
        frame[f"calendar_{j}"] = calendar[:, j]
    hi = index.get_indexer(history)
    history_count = int(frame.demand_mw.iloc[hi].notna().sum())
    if history_count < 12:
        raise ValueError(f"Only {history_count}/24 input hours have demand readings; at least 12 are required. Try another date or refresh the API.")
    node = data["states"].index(region)
    normalized = (frame.demand_mw.to_numpy() - data["center"][node]) / data["scale"][node]
    positions, known = known_lags(targets, issue, index)
    known &= np.isfinite(normalized[np.maximum(positions, 0)])
    lags = np.where(known, normalized[np.maximum(positions, 0)], 0).astype("float32")
    anchor = np.take_along_axis(lags, known.argmax(1)[:, None], axis=1)[:, 0]
    raw = frame[data["features"]].to_numpy()
    dynamic = np.nan_to_num((raw - data["feature_center"]) / data["feature_scale"]).astype("float32")
    sample = {"dynamic": dynamic[hi][None],
              "covariates": np.column_stack([lags, known, calendar_values(targets, region, holiday_dates)])[None].astype("float32"),
              "anchor": anchor[None], "nodes": np.array([node]),
              "actual": frame.demand_mw.reindex(targets).to_numpy(dtype="float32")[None]}
    records = data.get("records")
    matching = (records.index[records.region.eq(region) & records.forecast_date.eq(pd.Timestamp(forecast_date))]
                if records is not None else [])
    sample["_record_index"] = np.array([int(matching[0]) if len(matching) == 1 else -1])
    details = frame.loc[history, ["demand_mw", *WEATHER]].reset_index(names="time_utc")
    return sample, issue, targets, details


class BenchmarkForecaster:
    def __init__(self, root, device=None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.root = Path(root)
        self.data = prepared_data(self.root, self.root / CACHE)
        self.models = {}
        self.plans = {}
        self.thresholds = json.loads((self.root / "results" / MAIN_RUN / "training_peak_thresholds.json").read_text())
        torch.set_num_threads(2)

    def plan(self, spec):
        if spec.folder not in self.plans:
            self.plans[spec.folder] = json.loads((self.root / "results" / spec.folder / "plan.json").read_text())
        return self.plans[spec.folder]

    def _model(self, spec, seed):
        torch = self.torch
        from src.models.stgnn.grid_aware import GridAware, SeasonalGridAware, TemporalResidual, TemporalOnly
        key = spec.folder, spec.model, seed
        if key not in self.models:
            saved = torch.load(self.root / "results" / spec.folder / f"{spec.model}_{seed}_best.pt", map_location=self.device, weights_only=False)
            if saved["features"] != self.data["features"] or saved["states"] != self.data["states"]:
                raise ValueError("Checkpoint feature/region order differs from the verified data cache")
            for name, value in saved["scalers"].items():
                np.testing.assert_array_equal(np.asarray(self.data[name]), np.asarray(value))
            if spec.model.startswith("temporal_"):
                model = TemporalOnly(self.data, backbone=spec.model.removeprefix("temporal_"))
            elif spec.model.startswith("grid_aware"):
                model = GridAware(self.data, backbone=spec.model.removeprefix("grid_aware_"))
            elif spec.model == "residual_identity":
                model = TemporalResidual(self.data)
            elif spec.model == "residual_weather":
                from src.models.cnn_tuning import ResidualCNN
                model = ResidualCNN(self.data, weather=True)
            else:
                graph = self.data if spec.model != "seasonal_identity" else dict(self.data,
                    edge_index=np.empty((2, 0), dtype=np.int64), edge_features=np.empty((0, self.data["edge_features"].shape[1])))
                model = SeasonalGridAware(graph)
            model.to(self.device)
            model.load_state_dict(saved["state_dict"])
            self.models[key] = model.eval()
        return self.models[key]

    def predict(self, sample, keys, seeds):
        torch = self.torch
        rows = []
        node = int(sample["nodes"][0])
        # cuDNN can select numerically different RNN kernels for batch=1.
        # Repeat independent copies (no batch-normalization/cross-sample layers)
        # to use the same inference shape as the saved evaluation. Keep only
        # the first result. CPU fallback uses one sample and may differ slightly.
        batch_size = 256 if self.device == "cuda" else 1
        if self.device == "cuda":
            record_index = int(sample.get("_record_index", [-1])[0])
            for split, ids in self.data["splits"].items():
                positions = np.flatnonzero(ids == record_index)
                if len(positions):
                    size = 256 if split.startswith("test") else 32
                    batch_size = min(size, len(ids) - int(positions[0]) // size * size)
                    break
        for key in keys:
            spec = SPECS[key]
            allowed = self.plan(spec)["seeds"]
            if not set(seeds).issubset(allowed):
                raise ValueError("Unknown seed for the frozen experiment")
            for seed in seeds:
                dynamic, cov = feature_mask(sample["dynamic"], sample["covariates"], self.data["features"], spec.features)
                if spec.model in ["xgboost", "naive_day_available", "naive_week"]:
                    from types import SimpleNamespace
                    from src.models.hfed_tabular import predict_control
                    small = dict(self.data, **sample)
                    args = SimpleNamespace(output=self.root / "results" / spec.folder, device=self.device, feature_set=spec.features)
                    values = predict_control(spec.model, seed, small, np.array([0]), args)[0]
                else:
                    model = self._model(spec, seed)
                    with torch.inference_mode():
                        seq = torch.from_numpy(dynamic).to(self.device).repeat(batch_size, 1, 1)
                        nodes = torch.from_numpy(sample["nodes"]).to(self.device).repeat(batch_size)
                        if spec.model.startswith(("grid_aware", "temporal_")):
                            result = model.forward_samples(seq, nodes)
                            values = (result * self.data["target_scale"] + self.data["target_center"]).cpu().numpy()[0]
                        else:
                            result = model.forward_samples(seq, nodes,
                                torch.from_numpy(cov).to(self.device).repeat(batch_size, 1, 1),
                                torch.from_numpy(sample["anchor"]).to(self.device).repeat(batch_size, 1))
                            values = (result * float(self.data["scale"][node]) + float(self.data["center"][node])).cpu().numpy()[0]
                rows.extend({"key": key, "model": spec.label, "seed": seed, "hour": h,
                             "predicted": float(value), "actual": float(sample["actual"][0, h])}
                            for h, value in enumerate(values))
        return pd.DataFrame(rows)

    def cached_sample(self, region, day):
        records = self.data["records"]
        selected = records.index[records.region.eq(region) & records.forecast_date.eq(pd.Timestamp(day))]
        if len(selected) != 1:
            return None
        i = int(selected[0])
        sample = {key: self.data[key][i:i+1].copy() for key in ["dynamic", "covariates", "anchor", "nodes", "actual"]}
        sample["_record_index"] = np.array([i])
        return sample


@lru_cache(maxsize=8)
def archive_demand(root, region):
    root = Path(root)
    from src.data.hfed_loader import _read_hfed_csv, load_region
    if region == "NL":
        old = hourly_native(load_region(region, root / "data/hfed/2019-01-01_to_2024-12-31", "DEMAND", []), region)
    else:
        frame = pd.read_parquet(root / "data/processed/hfed_processed.parquet", columns=["region", "time_utc", "demand_mw"])
        old = frame.loc[frame.region.eq(region)].set_index("time_utc").demand_mw
    fresh = _read_hfed_csv(root / f"data/external/hfed_holdout_2025/{region}.csv").rename(columns={"TIME_PERIOD": "time_utc", "OBS_VALUE": "demand_mw"})
    return pd.concat([old, hourly_native(fresh, region)]).groupby(level=0).first().sort_index()


def request_observations(root, region, day, use_api=False, uploaded=None):
    """Return native-hour demand and past weather, keeping targets separate."""
    root = Path(root)
    issue, _, targets = local_day_grid(day, region)
    start, stop = targets[0] - pd.Timedelta(days=8), targets[-1] + pd.Timedelta(hours=1)
    warnings = []
    if uploaded is not None:
        demand = hourly_native(uploaded.loc[uploaded.region.eq(region)], region)
        source = "Uploaded HFED observations"
    elif use_api:
        from src.data.hfed_api import _get_csv, series_url
        from src.data.hfed_loader import load_config
        from src.data.upload_forecast import read_observations
        config = load_config(root / "config/config.yaml")
        content = _get_csv(series_url(region, [config["download"]["demand_variables"][region]]), start, stop, timeout=45)
        if content is None:
            raise ValueError("HFED has no published observations for this range")
        demand = hourly_native(read_observations([BytesIO(content)], config), region)
        source = "Fresh HFED API observations"
    else:
        demand = archive_demand(str(root), region)
        source = "Frozen HFED archive"
    demand = demand.loc[demand.index.to_series().between(start, stop)].copy()
    weather = pd.read_parquet(root / "data/weather/eccc/historical_weather.parquet")
    weather = weather.loc[weather.region.eq(region)].set_index("time_utc")[WEATHER]
    if use_api and pd.Timestamp(day).year > 2025:
        provenance = json.loads((root / "data/weather/eccc/provenance.json").read_text())
        station = next(item["station"] for item in provenance["selections"] if item["region"] == region)
        try:
            response = requests.get("https://api.weather.gc.ca/collections/climate-hourly/items", params={
                "CLIMATE_IDENTIFIER": station["CLIMATE_IDENTIFIER"], "datetime": f"{start.isoformat()}Z/{issue.isoformat()}Z",
                "limit": 1000, "properties": "UTC_DATE,TEMP,RELATIVE_HUMIDITY,WIND_SPEED", "f": "csv"}, timeout=(10, 45))
            response.raise_for_status()
            fresh = pd.read_csv(BytesIO(response.content)).rename(columns={"UTC_DATE": "time_utc", "TEMP": "temperature_c", "RELATIVE_HUMIDITY": "relative_humidity", "WIND_SPEED": "wind_speed"})
            fresh["time_utc"] = pd.to_datetime(fresh.time_utc, utc=True).dt.tz_localize(None)
            weather = pd.concat([weather, fresh.set_index("time_utc")[WEATHER]]).groupby(level=0).last().sort_index()
        except (requests.RequestException, ValueError, KeyError):
            warnings.append("Recent station weather is unavailable; weather inputs are masked. The demand-only CNN can still forecast.")
    holidays_path = root / "data/holidays/canada_holidays.json"
    holidays = json.loads(holidays_path.read_text())
    year = pd.Timestamp(day).year
    serving_path = root / "data/holidays/dashboard_holidays.json"
    if serving_path.exists():
        holidays.update(json.loads(serving_path.read_text()))
    missing = [y for y in range(start.year, year+1) if not holidays.get(str(y))]
    if missing:
        fetched = fetch_holidays(missing, cache_path=serving_path,
                                 force_refresh=any(str(y) in holidays for y in missing))
        if any(not fetched.get(y) for y in missing):
            raise ValueError("The holiday calendar could not be retrieved; retry before generating this forecast")
        holidays.update({str(y): value for y, value in fetched.items()})
    holiday_frame = build_holiday_dataframe({int(k): v for k, v in holidays.items()}, hfed_regions=REGIONS)
    dates = set(holiday_frame.loc[holiday_frame.region.eq(region), "date"])
    return demand, weather, dates, source, warnings
