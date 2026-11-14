"""Inference for uploaded HFED demand, with forecast-day actuals kept separate."""

import json

import numpy as np
import pandas as pd

from src.data.feature_engineering import build_features
from src.data.hfed_loader import load_config
from src.data.holiday_client import build_holiday_dataframe
from src.data.pipeline import RegionalStandardScaler
from src.models.naive import NaiveDay, NaiveWeek

TIME_ZONES = {
    "AB": "America/Edmonton", "BC": "America/Vancouver", "NB": "America/Moncton",
    "NL": "America/St_Johns", "NS": "America/Halifax", "ON": "America/Toronto",
    "PE": "America/Halifax", "QC": "America/Toronto",
}

HFED_COLUMNS = {
    "Reference area": "REF_AREA", "Energy flow": "ENERGY_FLOWS",
    "Time period (UTC)": "TIME_PERIOD", "Observation value": "OBS_VALUE",
    "Unit measure": "UNIT_MEASURE", "Counterpart area": "COUNTERPART_AREA",
}


def read_observations(files, config):
    frames = []
    demand_codes = config["download"]["demand_variables"]
    generation_codes = config["download"]["generation_variables"]
    for file in files:
        raw = pd.read_csv(file, encoding="utf-8-sig", keep_default_na=False)
        raw.columns = raw.columns.str.strip()
        raw = raw.rename(columns=HFED_COLUMNS)
        if not {"REF_AREA", "ENERGY_FLOWS", "TIME_PERIOD", "OBS_VALUE"} <= set(raw):
            raise ValueError("Each CSV needs Reference area, Energy flow, Time period (UTC), and Observation value.")
        raw["region"] = raw["REF_AREA"].str.strip().str.removeprefix("CA_")
        if "COUNTERPART_AREA" in raw:
            raw = raw[raw["COUNTERPART_AREA"].str.strip().isin(["", "_Z", "N/A"])]
        if "UNIT_MEASURE" in raw:
            raw = raw[raw["UNIT_MEASURE"].str.strip().eq("MW")]
        for region, subset in raw.groupby("region"):
            if region not in TIME_ZONES:
                continue
            flows = [demand_codes[region], *generation_codes.get(region, [])]
            selected = subset[subset["ENERGY_FLOWS"].isin(flows)].copy()
            selected["feature"] = selected["ENERGY_FLOWS"].replace({demand_codes[region]: "demand_mw"})
            selected["time_utc"] = pd.to_datetime(
                selected["TIME_PERIOD"], utc=True, format="ISO8601", errors="coerce",
            ).dt.tz_localize(None)
            selected["value"] = pd.to_numeric(
                selected["OBS_VALUE"].astype(str).str.replace(",", "", regex=False), errors="coerce",
            )
            supplied = selected["OBS_VALUE"].astype(str).str.strip().ne("")
            if selected["time_utc"].isna().any() or not np.isfinite(selected.loc[supplied, "value"]).all():
                raise ValueError(f"{region} has an invalid UTC timestamp or MW reading.")
            if selected.loc[selected["feature"].eq("demand_mw"), "value"].lt(0).any():
                raise ValueError(f"{region} has a negative demand reading.")
            frames.append(selected.loc[supplied, ["region", "time_utc", "feature", "value"]])
    if not frames or not any(frame["feature"].eq("demand_mw").any() for frame in frames):
        raise ValueError("No supported provincial demand readings were found.")
    long = pd.concat(frames, ignore_index=True)
    return long.groupby(["region", "time_utc", "feature"])["value"].mean().unstack().reset_index()


def local_time(timestamps, region):
    return timestamps.dt.tz_localize("UTC").dt.tz_convert(TIME_ZONES[region]).dt.tz_localize(None)


def hourly_observations(data, region):
    subset = data[data["region"].eq(region)].drop(columns="region")
    # Keep the local top-of-hour grid, including Newfoundland's UTC half-hours.
    offset = (local_time(subset["time_utc"], region).dt.minute - subset["time_utc"].dt.minute) % 60
    shift = pd.to_timedelta(offset, unit="min")
    subset["time_utc"] = (subset["time_utc"] + shift).dt.floor("h") - shift
    return subset.groupby("time_utc").mean().sort_index()


def target_hours(region, forecast_date):
    start = pd.Timestamp(forecast_date)
    zone = TIME_ZONES[region]
    hours = pd.date_range(start.tz_localize(zone), (start + pd.Timedelta(days=1)).tz_localize(zone),
                          freq="h", inclusive="left")
    if len(hours) != 24:
        raise ValueError("Select a 24-hour forecast date. The saved models use 24 output slots.")
    return hours.tz_convert("UTC").tz_localize(None).rename("time_utc")


def interpolate_short_demand_gaps(demand):
    """Fill complete interior runs of at most two missing hourly readings."""
    missing = demand.isna()
    runs = missing.ne(missing.shift()).cumsum()
    eligible = missing & missing.groupby(runs).transform("size").le(2)
    interpolated = demand.interpolate(method="time", limit_area="inside")
    return demand.mask(eligible, interpolated)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _holidays(root):
    path = root / "data" / "holidays" / "canada_holidays.json"
    return build_holiday_dataframe(_json(path) if path.exists() else {}, list(TIME_ZONES))


def has_cached_holidays(root, year):
    path = root / "data" / "holidays" / "canada_holidays.json"
    return path.is_file() and bool(_json(path).get(str(year)))


def _recurrent(features, region, model_dir, config, scaler):
    import torch

    from src.models.gru_model import GRUForecaster
    from src.models.lstm_model import LSTMForecaster

    metadata = _json(model_dir / "lstm" / "meta.json")
    names = metadata["feature_cols"]
    frame = features.reindex(columns=["region", *names])
    frame = scaler.transform(frame, list(scaler.stats[region]))
    for name in metadata["region_dummy_cols"]:
        frame[name] = float(name == f"region_{region}")
    for cls in (LSTMForecaster, GRUForecaster):
        model = cls(config).load(model_dir)
        inputs = torch.as_tensor(frame.drop(columns="region").tail(model.lookback_hours)
                                 .fillna(0).to_numpy("float32")[None], device=model.device)
        with torch.inference_mode():
            values = model.model(inputs).cpu().numpy()[0]
        yield model.name, scaler.inverse_transform_demand(values, region)


def _xgboost(features, region, model_dir):
    from xgboost import XGBRegressor

    path = model_dir / "xgboost"
    names = _json(path / "meta.json")["feature_names"]
    # These checkpoints output centered MW. Restore each hourly center
    # recovered from matching checkpoint and archived forecast outputs.
    offsets = _json(path / "prediction_offsets.json")[region]
    row = features.iloc[[-1]].reindex(columns=names).to_numpy("float32")
    values = []
    for hour, offset in enumerate(offsets):
        model = XGBRegressor(n_jobs=1)
        model.load_model(path / f"{region}_hour_{hour:02d}.ubj")
        values.append(float(model.predict(row)[0]) + offset)
    return values


def _graph_forecasts(history, region, forecast_date, targets, holidays, root, *, full_graph):
    import torch
    from joblib import load
    from xgboost import XGBRegressor

    from src.models.stgnn.residual import ResidualSTGCN, build_examples, predict_graph

    model_dir = root / "models"
    graph_dir = model_dir / "stgnn"
    meta = _json(graph_dir / "residual_metadata.json")
    regions = meta["regions"]
    stats = pd.DataFrame.from_dict(meta["scaling"], orient="index")
    utc_grid = pd.date_range(targets[0].floor("h") - pd.Timedelta(days=14),
                             targets[-1].ceil("h"), freq="h", name="time_utc")
    parts = []
    for code in regions:
        # The graph was trained on top-of-hour UTC bins.
        rows = history.loc[history["region"].eq(code)]
        utc_series = rows["demand_mw"].groupby(rows["time_utc"].dt.floor("h")).mean()
        part = utc_series.reindex(utc_grid).reset_index()
        part["region"] = code
        part["time_local"] = local_time(part["time_utc"], code)
        holiday_dates = pd.to_datetime(holidays.loc[holidays["region"].eq(code), "date"])
        part["is_holiday"] = part["time_local"].dt.normalize().isin(holiday_dates).astype(int)
        parts.append(part)
    dates = {pd.Timestamp(forecast_date) - pd.Timedelta(days=i) for i in range(4)}
    frame, tensors, _ = build_examples(pd.concat(parts, ignore_index=True), None, regions,
                                       stats=stats, forecast_dates=dates)
    nodes = tensors[3].numpy()
    mask = nodes == regions.index(region)
    if not mask.any():
        raise ValueError("The uploaded history does not cover the STGNN forecast window.")
    selected_graph = meta["selected_graph"]
    day = frame.loc[np.repeat(mask, 24), "forecast_date"].eq(pd.Timestamp(forecast_date))
    if day.sum() != 24:
        raise ValueError("The residual models could not build all 24 target hours.")

    hist, calendar, anchors = (tensor.numpy()[mask] for tensor in tensors[:3])
    own = hist[:, :, regions.index(region), :].reshape(mask.sum(), -1)
    inputs = np.concatenate([calendar, np.repeat(own[:, None, :], 24, axis=1)], axis=-1)
    local = XGBRegressor(n_jobs=1)
    local.load_model(model_dir / "xgboost" / f"local_residual_{region}.json")
    residual = local.predict(inputs.reshape(-1, inputs.shape[-1])).reshape(-1, 24)
    local_mw = ((anchors + residual).ravel() * stats.loc[region, "std"]
                + stats.loc[region, "mean"]).clip(min=0)
    predictions = {"local_residual_xgboost": local_mw[day.to_numpy()]}
    graphs = {}
    variants = ("identity", "geographic", "interconnection", "correlation") if full_graph else ("identity",)
    for variant in variants:
        name = f"residual_stgcn_{variant}"
        members = []
        for seed in (42, 123):
            state = torch.load(graph_dir / f"{name}_{seed}.pt", map_location="cpu", weights_only=True)
            model = ResidualSTGCN(state["adjacency"], tensors[1].shape[-1])
            model.load_state_dict(state)
            members.append(predict_graph(model, frame, tensors, stats, mask))
        graph = members[0]
        graph["predicted"] = np.mean([member["predicted"].to_numpy() for member in members], axis=0)
        graph_day = graph["forecast_date"].eq(pd.Timestamp(forecast_date))
        predictions[name] = graph.loc[graph_day, "predicted"].to_numpy()
        graphs[name] = graph

    if not full_graph:
        return predictions, None
    graph = graphs[selected_graph]
    hybrid = graph.copy()
    weight = meta["hybrid_graph_weights"][region]
    hybrid["predicted"] = weight * graph["predicted"] + (1 - weight) * local_mw
    day = hybrid["forecast_date"].eq(pd.Timestamp(forecast_date))
    predictions["stgcn_local_hybrid"] = hybrid.loc[day, "predicted"].to_numpy()

    pbc_dir = model_dir / "pbc"
    base = _json(pbc_dir / "meta.json")["base_model"]
    calibrated = hybrid if base == "stgcn_local_hybrid" else graphs[base].copy()
    # Only earlier observations from forecast history can feed the recent-error branch.
    calibrated.loc[day, "actual"] = np.nan
    calibrator = load(pbc_dir / "model.joblib")
    calibrated = calibrated.loc[calibrated["issued_at"].ge(calibrator.calibration["issued_at"].max())]
    if calibrated.empty:
        return predictions, None
    pbc = calibrator.predict(calibrated, _json(root / "results" / "peak_thresholds.json"))
    return predictions, pbc.loc[pbc["forecast_date"].eq(pd.Timestamp(forecast_date))]


def run_uploaded_forecast(history, region, forecast_date, root):
    config = load_config(root / "config" / "config.yaml")
    targets = target_hours(region, forecast_date)
    observed = hourly_observations(history, region)
    past = pd.date_range(end=targets[0] - pd.Timedelta(hours=1), periods=336, freq="h", name="time_utc")
    observed = observed.reindex(past)
    original_demand = observed["demand_mw"]
    observed["demand_mw"] = interpolate_short_demand_gaps(original_demand)
    estimated = observed.loc[
        original_demand.isna() & observed["demand_mw"].notna(), "demand_mw"
    ]
    missing = past[observed["demand_mw"].isna()]
    if len(missing):
        preview = missing[:8].tz_localize("UTC")
        local = preview.tz_convert(TIME_ZONES[region])
        timestamps = ", ".join(
            f"{utc:%Y-%m-%d %H:%M} UTC ({wall:%Y-%m-%d %H:%M %Z})"
            for utc, wall in zip(preview, local)
        )
        if len(missing) > len(preview):
            timestamps += f", and {len(missing) - len(preview)} more"
        raise ValueError(
            f"{region} is missing {len(missing)} of {len(past)} hourly demand readings "
            f"before {forecast_date}. Missing readings: {timestamps}."
        )

    holidays = _holidays(root)
    hourly = observed.reset_index()
    hourly["region"] = region
    hourly["time_local"] = local_time(hourly["time_utc"], region)
    features = build_features(hourly, holidays)
    model_dir = root / "models"
    scaler = RegionalStandardScaler.from_dict(_json(model_dir / "lstm" / "scaler_stats.json"))

    result = pd.DataFrame(index=targets)
    timeline = observed[["demand_mw"]].reindex(past.append(targets)).reset_index()
    timeline["region"] = region
    for model in (NaiveDay(), NaiveWeek()):
        result[model.name] = model.predict(timeline).set_index("time_utc")["predicted"]
    result["xgboost"] = _xgboost(features, region, model_dir)
    for name, values in _recurrent(features, region, model_dir, config, scaler):
        result[name] = values
    cutoff = targets[0] - pd.Timedelta(hours=1)
    history = history[history["time_utc"].lt(targets[0])]
    recent = history.loc[history["time_utc"].gt(cutoff - pd.Timedelta(hours=48))
                         & history["demand_mw"].notna(), "region"]
    missing = sorted(set(TIME_ZONES) - set(recent))
    points, pbc = _graph_forecasts(
        history, region, forecast_date, targets, holidays, root, full_graph=not missing,
    )
    for name, values in points.items():
        result[name] = values

    graph_note = None
    if missing:
        graph_note = ("Upload recent demand for " + ", ".join(missing)
                      + " to show the geographic, interconnection, and correlation graphs, the hybrid, and PBC.")
    elif pbc is not None and len(pbc) == 24:
        result[["pbc", "lower_80", "upper_80"]] = pbc[["predicted", "lower_80", "upper_80"]].to_numpy()
    else:
        graph_note = "PBC is available for dates after its calibration period."
    if not estimated.empty:
        values = ", ".join(f"{hour:%Y-%m-%d %H:%M} UTC ({mw:,.1f} MW)"
                           for hour, mw in estimated.items())
        note = (
            f"Linearly interpolated {region} demand for the naïve, XGBoost, LSTM, "
            f"and GRU forecasts at {values}. Residual models use their existing missing-data masks."
        )
        graph_note = f"{note} {graph_note}" if graph_note else note
    return result.clip(lower=0), graph_note
