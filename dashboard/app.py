"""Streamlit dashboard for verified HFED day-ahead demand forecasts.

Active pages serve date-based forecasts from corrected checkpoints, benchmark
evidence, fixed-architecture ablations, and the proposal's planning diagnostics.
The earlier prototype functions below are retained for reference; main() routes
only to dashboard.benchmark_pages and its verified artifacts.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
    DEPS_OK = True
except ImportError as exc:
    DEPS_OK = False
    _MISSING = str(exc)


RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
HOLIDAYS_PATH = ROOT / "data" / "holidays" / "holidays_df.parquet"
LIVE_META_PATH = MODELS_DIR / "stgnn_live" / "metadata.json"
LIVE_METRICS_PATH = RESULTS_DIR / "live_model_metrics.json"

REGION_NAMES = {
    "AB": "Alberta",
    "BC": "British Columbia",
    "NB": "New Brunswick",
    "NL": "Newfoundland and Labrador",
    "NS": "Nova Scotia",
    "ON": "Ontario",
    "PE": "Prince Edward Island",
    "QC": "Quebec",
}
TIME_ZONES = {
    "AB": "America/Edmonton", "BC": "America/Vancouver", "NB": "America/Moncton",
    "NL": "America/St_Johns", "NS": "America/Halifax", "ON": "America/Toronto",
    "PE": "America/Halifax", "QC": "America/Toronto",
}
MODEL_LABELS = {
    "residual_stgcn_geographic": "Residual STGCN · Geographic",
    "residual_stgcn_identity": "Residual STGCN · Identity",
    "residual_stgcn_interconnection": "Residual STGCN · Interconnection",
    "residual_stgcn_correlation": "Residual STGCN · Correlation",
    "stgcn_local_hybrid": "STGCN + local XGBoost hybrid",
    "pbc": "PBC (calibrated hybrid)",
    "gru": "GRU",
    "lstm": "LSTM",
    "local_residual_xgboost": "Local residual XGBoost",
    "xgboost": "XGBoost",
    "naive_day": "Naive · previous day",
    "naive_week": "Naive · previous week",
}

FORECAST = "#0d9488"
ACTUAL = "#f59e0b"
HISTORY = "#94a3b8"
THRESHOLD = "#ef4444"
PALETTE = ["#0d9488", "#6366f1", "#f43f5e", "#0ea5e9", "#a855f7", "#84cc16", "#f97316"]

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px;}
h1 {font-weight: 700; letter-spacing: -0.02em;}
.page-subtitle {opacity: 0.72; font-size: 1.02rem; margin: -0.4rem 0 1.4rem 0;}
.hero {
    background: linear-gradient(120deg, #0f2a3d 0%, #14506b 55%, #0d9488 100%);
    border-radius: 18px;
    padding: 1.6rem 2rem 1.5rem 2rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 24px rgba(15, 42, 61, 0.18);
}
.hero-kicker {color: #99f6e4; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.08em;
              text-transform: uppercase; margin-bottom: 0.45rem;}
.hero-title {color: #ffffff; font-size: 1.75rem; font-weight: 750; line-height: 1.25;
             letter-spacing: -0.01em; margin-bottom: 0.5rem;}
.hero-text {color: #d7e7ef; font-size: 0.98rem; margin: 0;}
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dbe5ec;
    border-left: 5px solid #0d9488;
    border-radius: 14px;
    padding: 0.85rem 1.1rem;
    box-shadow: 0 2px 8px rgba(15, 42, 61, 0.06);
}
[data-testid="stMetricLabel"] p {font-size: 0.82rem; color: #4a6276;}
[data-testid="stMetricValue"] {font-size: 1.55rem; color: #0f2a3d;}
[data-testid="stMetricValue"] > div {white-space: normal !important; overflow: visible !important;
                                     text-overflow: clip !important; line-height: 1.25;}
.kpi-row {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
           gap: 1rem; margin-bottom: 1.2rem;}
.kpi {background: #ffffff; border: 1px solid #dbe5ec; border-left: 5px solid #0d9488;
      border-radius: 14px; padding: 0.85rem 1.1rem; box-shadow: 0 2px 8px rgba(15, 42, 61, 0.06);}
.kpi-label {font-size: 0.82rem; color: #4a6276; margin-bottom: 0.25rem;}
.kpi-value {font-size: 1.35rem; font-weight: 650; color: #0f2a3d; line-height: 1.25;
            overflow-wrap: anywhere;}
.kpi-sub {font-size: 0.78rem; color: #0d9488; margin-top: 0.3rem;}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    background: #ffffff;
    border: 1px solid #dbe5ec;
    border-radius: 14px;
    padding: 0.4rem;
}
.stTabs [data-baseweb="tab-list"] {gap: 0.4rem;}
.stTabs [aria-selected="true"] {color: #0d9488;}
[data-testid="stSidebarNav"] {padding-top: 0.6rem;}
.sidebar-brand {font-weight: 700; font-size: 1.05rem; line-height: 1.3;}
.sidebar-note {opacity: 0.65; font-size: 0.82rem; line-height: 1.4;}
</style>
"""


def header(title, subtitle):
    st.title(title)
    st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def style_figure(fig, height=420, y_title="Demand (MW)"):
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def label(model):
    return MODEL_LABELS.get(model, model)


def region_label(code):
    return f"{REGION_NAMES.get(code, code)} ({code})"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_comparison():
    path = RESULTS_DIR / "model_comparison.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_predictions(model):
    path = RESULTS_DIR / f"predictions_{model}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    frame["forecast_date"] = pd.to_datetime(frame["forecast_date"]).dt.date
    return frame


@st.cache_data(show_spinner=False)
def load_region_metrics(models):
    frames = []
    for model in models:
        path = RESULTS_DIR / f"metrics_by_region_{model}.csv"
        if path.exists():
            frames.append(pd.read_csv(path).assign(model=model))
    return pd.concat(frames, ignore_index=True) if frames else None


@st.cache_data(show_spinner=False)
def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


@st.cache_data(show_spinner=False)
def load_high_demand():
    path = RESULTS_DIR / "high_demand_flags.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_holidays():
    return pd.read_parquet(HOLIDAYS_PATH) if HOLIDAYS_PATH.exists() else None


@st.cache_data(ttl=900, max_entries=24, show_spinner=False)
def live_forecast(last_day):
    """Fetch API demand around ``last_day`` and forecast every province for the next day."""
    from src.data.hfed_api import fetch_demand_history
    from src.data.hfed_loader import load_config
    from src.data.holiday_client import get_holiday_df_for_data_range
    from src.models.stgnn.live import HISTORY_DAYS, forecast_next_day

    config = load_config(ROOT / "config" / "config.yaml")
    regions = sorted(config["download"]["regions"])
    start = pd.Timestamp(last_day) - pd.Timedelta(days=HISTORY_DAYS)
    end = pd.Timestamp(last_day) + pd.Timedelta(days=3)
    demand, unavailable = fetch_demand_history(regions, start, end, config, timeout=30)
    if demand.empty:
        return demand, pd.DataFrame(), unavailable
    year = pd.Timestamp(last_day).year
    holidays = get_holiday_df_for_data_range(
        year - 1, year + 1, hfed_regions=regions, cache_path=ROOT / config["holidays"]["cache_file"],
    )
    try:
        forecast = forecast_next_day(demand, holidays, last_day, ROOT)
    except ValueError as exc:
        if "No complete forecast-day grid" not in str(exc):
            raise
        forecast = pd.DataFrame()
    return demand, forecast, unavailable


def local_times(utc, region):
    return pd.DatetimeIndex(utc).tz_localize("UTC").tz_convert(TIME_ZONES[region]).tz_localize(None)


def day_length(day, region):
    zone = TIME_ZONES[region]
    start = pd.Timestamp(day).tz_localize(zone)
    return int(((start + pd.Timedelta(days=1)).tz_localize(None).tz_localize(zone) - start)
               / pd.Timedelta(hours=1))


# ---------------------------------------------------------------------------
# Page: Live forecast
# ---------------------------------------------------------------------------

HERO = """
<div class="hero">
  <div class="hero-kicker">Day-ahead forecasting · 8 provinces · Statistics Canada HFED</div>
  <div class="hero-title">Spatio-Temporal Graph Neural Network for Day-Ahead
  Electricity Demand Forecasting in Canada</div>
  <p class="hero-text">Forecasts all 24 hours of the next day for every province, issued at
  23:00 local time from the latest HFED API readings.</p>
</div>
"""


def page_live_forecast():
    st.markdown(HERO, unsafe_allow_html=True)
    header("Live day-ahead forecast",
           "Pick the last day with observed demand. The residual STGCN, retrained on HFED API data, "
           "pulls the latest readings and forecasts the next 24 hours for every province.")
    meta = load_json(LIVE_META_PATH)
    if meta is None:
        st.warning("The live model has not been trained yet. Run `python train_live.py` from the "
                   "repository root, then reload this page.")
        return

    regions = meta["regions"]
    today = pd.Timestamp.now(tz="America/Toronto").date()
    with st.form("live_form", border=True):
        controls = st.columns([1.3, 1, 0.8], vertical_alignment="bottom")
        chosen_region = controls[0].selectbox("Province", regions, index=None, format_func=region_label,
                                              placeholder="Choose a province")
        chosen_day = controls[1].date_input(
            "Last observed day", value=None, min_value=date(2019, 1, 15), max_value=today,
            help="Demand up to 23:00 local time on this day is the model input. "
                 "The forecast covers the following day.",
        )
        submitted = controls[2].form_submit_button("Forecast next day", type="primary", width="stretch")
    if submitted:
        if chosen_region is None or chosen_day is None:
            st.warning("Select a province and a last observed day, then click Forecast next day.")
            return
        live_forecast.clear(chosen_day)
        st.session_state["live_request"] = (chosen_region, chosen_day)
    if "live_request" not in st.session_state:
        st.info("Choose a province and the last day with observed demand, then click "
                "**Forecast next day**. The model forecasts the 24 hours of the following day.")
        return
    region, last_day = st.session_state["live_request"]
    target_day = last_day + pd.Timedelta(days=1)
    st.markdown(f"#### {REGION_NAMES[region]} · forecast for {target_day:%A, %B %d, %Y}")

    if day_length(target_day, region) != 24:
        st.warning(f"{target_day:%B %d, %Y} has {day_length(target_day, region)} hours in "
                   f"{REGION_NAMES[region]} because of a daylight-saving change. The model forecasts "
                   "24-hour days, so pick another day.")
        return

    try:
        with st.spinner("Downloading recent demand from the HFED API and running the model..."):
            demand, forecast, unavailable = live_forecast(last_day)
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Could not generate the live forecast. {exc}")
        return

    rows = forecast if forecast.empty else forecast.loc[forecast["region"].eq(region)].sort_values("target_hour")
    if rows.empty:
        reason = unavailable.get(region)
        if reason is None:
            reason = f"The model needs demand readings in the 48 hours ending {last_day:%b %d} at 23:00 local time."
        st.warning(f"No forecast for {REGION_NAMES[region]}. {reason}")
        if not forecast.empty:
            _all_provinces(forecast, meta)
        return

    history = demand.loc[demand["region"].eq(region)].copy()
    history["local"] = local_times(history["time_utc"], region)
    issued = pd.Timestamp(last_day) + pd.Timedelta(hours=23)
    latest = history.loc[history["local"].le(issued), "local"].max()
    if pd.notna(latest) and latest < issued:
        st.info(f"The latest {REGION_NAMES[region]} reading before the forecast is "
                f"{latest:%b %d %H:%M} local time. Later hours are treated as missing.")

    threshold = meta["high_demand_threshold_mw"].get(region)
    predicted = rows["predicted"].to_numpy()
    actual = rows["actual"].to_numpy(dtype=float)
    observed = np.isfinite(actual)
    peak_hour = int(np.argmax(predicted))

    cards = st.columns(4)
    cards[0].metric("Peak forecast", f"{predicted.max():,.0f} MW", f"at {peak_hour:02d}:00", delta_color="off")
    cards[1].metric("Daily energy", f"{predicted.sum() / 1000:,.1f} GWh")
    high = int((predicted >= threshold).sum()) if threshold else 0
    cards[2].metric("High-demand hours", f"{high} of 24",
                    f"threshold {threshold:,.0f} MW" if threshold else None, delta_color="off")
    if observed.any():
        error = np.abs(predicted[observed] - actual[observed])
        mape = float(np.mean(error / actual[observed]) * 100)
        cards[3].metric("Accuracy so far", f"{mape:.2f}% MAPE",
                        f"MAE {error.mean():,.0f} MW · {observed.sum()}/24 h", delta_color="off")
    else:
        cards[3].metric("Accuracy", "Pending", "actuals not published yet", delta_color="off")

    fig = go.Figure()
    shown = history.loc[history["local"].between(pd.Timestamp(last_day) - pd.Timedelta(days=2), issued)]
    fig.add_scatter(x=shown["local"], y=shown["demand_mw"], name="Observed demand",
                    line={"color": HISTORY, "width": 1.8})
    target_local = local_times(rows["target_at"], region)
    fig.add_scatter(x=target_local, y=predicted, name="STGCN forecast", mode="lines+markers",
                    line={"color": FORECAST, "width": 3}, marker={"size": 5})
    if observed.any():
        fig.add_scatter(x=target_local, y=actual, name="Actual demand", mode="lines+markers",
                        line={"color": ACTUAL, "width": 2, "dash": "dot"}, marker={"size": 4})
    if threshold:
        fig.add_hline(y=threshold, line={"color": THRESHOLD, "dash": "dash", "width": 1.2})
        fig.add_annotation(x=target_local[-1], y=threshold, text="High-demand threshold",
                           showarrow=False, yshift=10, xanchor="right", font={"color": THRESHOLD, "size": 11})
    start = pd.Timestamp(target_day)
    fig.add_vrect(x0=start, x1=start + pd.Timedelta(hours=23), fillcolor=FORECAST, opacity=0.06, line_width=0)
    fig.add_annotation(x=start, y=1, yref="paper", text="Forecast day", showarrow=False,
                       xanchor="left", yanchor="top", xshift=6, font={"color": FORECAST, "size": 12})
    style_figure(fig, height=440)
    fig.update_xaxes(title=f"Local time ({REGION_NAMES[region]})")
    st.plotly_chart(fig, width="stretch")

    tab_hourly, tab_all, tab_model = st.tabs(["Hourly forecast", "All provinces", "About the model"])
    with tab_hourly:
        daily = go.Figure()
        hours = rows["target_hour"]
        daily.add_scatter(x=hours, y=predicted, name="STGCN forecast", mode="lines+markers",
                          line={"color": FORECAST, "width": 2.4}, marker={"size": 5})
        if observed.any():
            daily.add_scatter(x=hours, y=actual, name="Actual demand", mode="lines+markers",
                              line={"color": ACTUAL, "width": 2.4, "dash": "dot"}, marker={"size": 5})
        if threshold:
            daily.add_hline(y=threshold, line={"color": THRESHOLD, "dash": "dash", "width": 1.2})
        style_figure(daily, height=440)
        daily.update_xaxes(title="Local hour", dtick=2)
        st.plotly_chart(daily, width="stretch", key="live_forecast_day")

        table = pd.DataFrame({
            "Hour": [f"{hour:02d}:00" for hour in rows["target_hour"]],
            "Forecast (MW)": predicted.round(0),
            "Actual (MW)": np.round(actual, 0),
            "Error (MW)": np.round(predicted - actual, 0),
            "High demand": predicted >= threshold if threshold else False,
        })
        st.dataframe(table, hide_index=True, width="stretch", height=320)
        st.caption("Error = forecast minus actual. Hours after the latest API reading have no actual yet.")
    with tab_all:
        _all_provinces(forecast, meta)
    with tab_model:
        _about_live_model(meta)


def _all_provinces(forecast, meta):
    records = []
    for code, rows in forecast.groupby("region"):
        rows = rows.sort_values("target_hour")
        predicted = rows["predicted"].to_numpy()
        actual = rows["actual"].to_numpy(dtype=float)
        observed = np.isfinite(actual)
        threshold = meta["high_demand_threshold_mw"].get(code)
        records.append({
            "Province": REGION_NAMES.get(code, code),
            "Peak forecast (MW)": round(float(predicted.max())),
            "Peak hour": f"{int(np.argmax(predicted)):02d}:00",
            "Daily energy (GWh)": round(float(predicted.sum()) / 1000, 1),
            "High-demand hours": int((predicted >= threshold).sum()) if threshold else 0,
            "MAPE so far": (f"{np.mean(np.abs(predicted[observed] - actual[observed]) / actual[observed]) * 100:.2f}%"
                            if observed.any() else "pending"),
        })
    st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")
    missing = sorted(set(meta["regions"]) - set(forecast["region"]))
    if missing:
        st.caption("Forecast unavailable for "
                   + ", ".join(REGION_NAMES.get(code, code) for code in missing) + ".")


def _about_live_model(meta):
    metrics = load_json(LIVE_METRICS_PATH)
    st.markdown(
        f"Residual STGCN, **{meta['selected_graph']}** graph, average of {len(meta['seeds'])} seeds. "
        f"Trained on HFED demand from {meta['data_start'][:10]} to {meta['data_end'][:10]} "
        "(processed archive through 2024, HFED API after that). The graph was chosen on July–August 2026 "
        "forecasts; September 2026 onward was held out."
    )
    if metrics:
        table = pd.DataFrame(metrics["overall"]).T
        table = table.rename(columns={"mae": "MAE (MW)", "rmse": "RMSE (MW)", "mape": "MAPE (%)",
                                      "n_hours": "Hours"})
        st.markdown(f"**Held-out test, {metrics['test_window']}**")
        st.dataframe(table.round(2), width="stretch")
    st.caption("Forecasts use only readings available by 23:00 local time on the last observed day. "
               "Readings on the forecast day are shown only for comparison.")


# ---------------------------------------------------------------------------
# Page: Forecast from CSV (unchanged upload flow)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=16)
def load_uploaded_forecast(history, region, forecast_date):
    from src.data.upload_forecast import run_uploaded_forecast

    return run_uploaded_forecast(history, region, forecast_date, ROOT)


def page_csv_forecast():
    from src.data.hfed_loader import load_config
    from src.data.upload_forecast import (
        has_cached_holidays,
        hourly_observations,
        local_time,
        read_observations,
    )
    from src.evaluation.metrics import mae, rmse

    header("Forecast from CSV",
           "Upload two weeks of hourly HFED demand to forecast the next day with every model "
           "and compare against ground truth.")
    actual_file = None
    ground_truth = None

    try:
        config = load_config(ROOT / "config" / "config.yaml")
        st.caption("Upload two weeks of hourly HFED demand. Include all eight provinces to show all 12 forecasts.")
        files = st.file_uploader("Forecast history", type="csv", accept_multiple_files=True)
        actual_file = st.file_uploader("Ground truth for the forecast date (optional)", type="csv")
        if not files:
            return
        history = read_observations(files, config)
        regions = sorted(history.loc[history["demand_mw"].notna(), "region"].unique())
        region = st.selectbox("Province", regions, format_func=REGION_NAMES.get, key="csv_province")
        observations = hourly_observations(history, region)["demand_mw"].dropna()
        latest = local_time(pd.Series(observations.index[-1:]), region).iloc[0]
        forecast_date = st.date_input(
            "Forecast date (province local time)", (latest + pd.Timedelta(days=1)).date(),
            key=f"upload_date_{region}_{latest.date()}",
        )
        with st.spinner("Generating next-day forecasts…"):
            forecasts, graph_note = load_uploaded_forecast(history, region, forecast_date)
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        st.error(f"Could not generate the forecast. {exc}")
        return

    actual = None
    if actual_file is not None or ground_truth is not None:
        try:
            if actual_file is not None:
                ground_truth = read_observations([actual_file], config)
            actual = hourly_observations(ground_truth, region)["demand_mw"].reindex(forecasts.index)
            count = int(actual.notna().sum())
            if count < 24:
                st.warning(f"Ground truth covers {count} of 24 forecast hours for {REGION_NAMES[region]}.")
        except (KeyError, ValueError) as exc:
            st.error(f"Could not read ground truth. {exc}")

    if graph_note:
        st.info(graph_note)
    if not has_cached_holidays(ROOT, forecast_date.year):
        st.caption("Holiday features default to zero where the local holiday calendar has no coverage.")
    generation = history.columns.difference(["region", "time_utc", "demand_mw"])
    provincial_history = history.loc[history["region"].eq(region), generation]
    if not provincial_history.notna().any().any():
        st.caption("Generation readings were unavailable for this province. XGBoost uses missing feature values. LSTM and GRU use their zero-fill convention.")

    labels = {
        "naive_day": "Previous-day naïve",
        "naive_week": "Weekly naïve",
        "xgboost": "XGBoost",
        "lstm": "LSTM",
        "gru": "GRU",
        "local_residual_xgboost": "Local residual XGBoost",
        "residual_stgcn_identity": "Residual STGCN · Identity",
        "residual_stgcn_geographic": "Residual STGCN · Geographic",
        "residual_stgcn_interconnection": "Residual STGCN · Interconnection",
        "residual_stgcn_correlation": "Residual STGCN · Correlation",
        "stgcn_local_hybrid": "STGCN and local XGBoost hybrid",
        "pbc": "PBC · Hybrid forecast with 80% prediction interval",
    }
    unavailable_models = [label for name, label in labels.items() if name not in forecasts]
    if unavailable_models:
        st.caption(f"{len(labels) - len(unavailable_models)} of {len(labels)} forecasts available for this selection.")
        with st.expander("Unavailable forecasts"):
            st.write("\n".join(f"- {label}" for label in unavailable_models))
    if actual is not None:
        model_names = [name for name in labels if name in forecasts]
        comparable = forecasts[model_names].assign(actual=actual).dropna()
        if not comparable.empty:
            errors = comparable[model_names].sub(comparable["actual"], axis=0).abs().mean()
            best = errors.min()
            winners = errors.index[np.isclose(errors, best, rtol=0, atol=1e-9)]
            names = ", ".join(labels[name] for name in winners)
            heading = "Best forecast" if len(winners) == 1 else "Best forecasts (tie)"
            st.success(f"{heading} — {names}")
            st.caption(
                f"Lowest mean absolute error: {best:,.2f} MW across "
                f"{len(comparable)} of {len(forecasts)} hours shared by "
                f"all {len(model_names)} displayed forecasts. "
                "MAE measures the average size of the miss in either direction."
            )
            runners_up = errors.drop(index=winners)
            if not runners_up.empty:
                next_best = runners_up.idxmin()
                st.caption(f"Next lowest MAE: {labels[next_best]} at {errors[next_best]:,.2f} MW.")
    hours = list(range(24))
    for name, label in labels.items():
        if name not in forecasts:
            continue
        fig = go.Figure()
        if name == "pbc":
            fig.add_scatter(x=hours, y=forecasts["lower_80"], line={"width": 0},
                            showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=hours, y=forecasts["upper_80"], line={"width": 0},
                            fill="tonexty", fillcolor="rgba(31, 119, 180, 0.18)",
                            name="80% prediction interval")
        fig.add_scatter(x=hours, y=forecasts[name], name="Forecast", mode="lines")
        if actual is not None and actual.notna().any():
            fig.add_scatter(x=hours, y=actual, name="Ground truth", mode="lines")
        fig.update_layout(title=label, xaxis_title="Local hour", yaxis_title="Demand (MW)")
        st.plotly_chart(fig, width="stretch", key=f"uploaded_forecast_{name}")
        if actual is not None:
            matched = pd.DataFrame({"Forecast (MW)": forecasts[name], "Actual (MW)": actual}).dropna()
            if not matched.empty:
                predicted = matched["Forecast (MW)"].to_numpy()
                measured = matched["Actual (MW)"].to_numpy()
                st.caption(
                    f"MAE {mae(measured, predicted):,.1f} MW · "
                    f"RMSE {rmse(measured, predicted):,.1f} MW · "
                    f"{len(matched)} of {len(forecasts)} hours matched"
                )
                with st.expander(f"Hourly forecast errors · {label}"):
                    matched["Error (MW)"] = predicted - measured
                    matched.insert(0, "Local hour", forecasts.index.get_indexer(matched.index))
                    st.caption("Error = forecast − actual. Positive values mean the forecast was too high.")
                    st.dataframe(matched.round(1), hide_index=True)


# ---------------------------------------------------------------------------
# Page: Model results
# ---------------------------------------------------------------------------

def page_model_results():
    header("Model results",
           "Held-out accuracy on the same 26,221 province-hours in July–December 2024, "
           "forecast at 23:00 local time for the next day.")
    comparison = load_comparison()
    if comparison is None:
        st.warning("No evaluation results yet. Run the training notebook first.")
        return
    comparison = comparison.sort_values("mae").reset_index(drop=True)
    comparison["Model"] = comparison["model"].map(label)
    best = comparison.iloc[0]
    gru = comparison.loc[comparison["model"].eq("gru")]

    gain = (1 - best["mae"] / gru["mae"].iloc[0]) * 100 if not gru.empty else None
    cards = [
        ("Best model", label(best["model"]), "lowest MAE on the test set"),
        ("Best MAE", f"{best['mae']:,.1f} MW", f"{best['mape']:.2f}% MAPE"),
        ("vs GRU baseline", f"{gain:.1f}% lower MAE" if gain is not None else "n/a",
         "strongest non-graph model"),
        ("Models compared", f"{len(comparison)}", "same province-hours"),
    ]
    html = "".join(f'<div class="kpi"><div class="kpi-label">{name}</div>'
                   f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>'
                   for name, value, sub in cards)
    st.markdown(f'<div class="kpi-row">{html}</div>', unsafe_allow_html=True)

    graph_family = comparison["model"].str.contains("stgcn|pbc")
    fig = px.bar(comparison.iloc[::-1], x="mae", y="Model", orientation="h", text="mae",
                 color=graph_family.iloc[::-1].map({True: "Graph models", False: "Baselines"}),
                 color_discrete_map={"Graph models": FORECAST, "Baselines": HISTORY})
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    style_figure(fig, height=460, y_title="")
    fig.update_layout(xaxis_title="MAE (MW), lower is better", legend_title_text="", hovermode="closest")
    st.plotly_chart(fig, width="stretch")

    table = comparison[["Model", "mae", "rmse", "mape", "peak_mae"]].rename(columns={
        "mae": "MAE (MW)", "rmse": "RMSE (MW)", "mape": "MAPE (%)", "peak_mae": "Peak-hour MAE (MW)"})
    st.dataframe(table.round(2), hide_index=True, width="stretch")

    st.subheader("Accuracy by province")
    regional = load_region_metrics(tuple(comparison["model"]))
    if regional is not None:
        pivot = regional.pivot_table(index="model", columns="region", values="mape")
        pivot = pivot.reindex(comparison["model"]).rename(index=label)
        pivot.columns = [REGION_NAMES.get(code, code) for code in pivot.columns]
        heat = px.imshow(pivot.round(2), text_auto=".2f", aspect="auto", color_continuous_scale="Teal",
                         labels={"color": "MAPE (%)"})
        heat.update_layout(height=470, margin={"l": 10, "r": 10, "t": 10, "b": 10})
        heat.update_xaxes(side="top", title="")
        heat.update_yaxes(title="")
        st.plotly_chart(heat, width="stretch")
        st.caption("MAPE (%) per province. Small provinces have low MW errors but similar percentage errors.")

    flags = load_high_demand()
    if flags is not None:
        from src.high_demand.detector import high_demand_summary

        st.subheader("High-demand hours")
        summary = high_demand_summary(flags)
        summary["Province"] = summary["region"].map(REGION_NAMES)
        left, right = st.columns([1.2, 1])
        bars = px.bar(summary, x="Province", y="n_high_demand_hours", color_discrete_sequence=[THRESHOLD])
        style_figure(bars, height=320, y_title="Flagged hours")
        bars.update_layout(xaxis_title="", hovermode="closest")
        left.plotly_chart(bars, width="stretch")
        right.dataframe(summary[["Province", "n_high_demand_hours", "high_demand_pct", "threshold_mw"]].rename(
            columns={"n_high_demand_hours": "Flagged hours", "high_demand_pct": "Share (%)",
                     "threshold_mw": "Threshold (MW)"}), hide_index=True, width="stretch")
        st.caption("Hours where the PBC forecast exceeds the province's 90th-percentile training demand. "
                   "A planning flag only, not a grid-capacity or outage prediction.")


# ---------------------------------------------------------------------------
# Page: Forecast explorer
# ---------------------------------------------------------------------------

def page_forecast_explorer():
    header("Forecast explorer",
           "Browse saved day-ahead forecasts from the July–December 2024 test period and compare models "
           "against observed demand.")
    comparison = load_comparison()
    if comparison is None:
        st.warning("No saved forecasts yet. Run the training notebook first.")
        return
    models = comparison.sort_values("mae")["model"].tolist()
    reference = load_predictions(models[0])
    regions = sorted(reference["region"].unique())
    dates = sorted(reference["forecast_date"].unique())

    controls = st.columns([1.2, 1, 2])
    region = controls[0].selectbox("Province", regions, index=regions.index("ON"), format_func=region_label)
    day = controls[1].date_input("Forecast date", dates[-1], min_value=dates[0], max_value=dates[-1])
    defaults = [name for name in (models[0], "gru", "naive_day") if name in models]
    chosen = controls[2].multiselect("Models", models, default=defaults, format_func=label)
    show_band = st.toggle("Show PBC 80% prediction interval", value=True)

    fig = go.Figure()
    actual = None
    scores = []
    if show_band:
        pbc = load_predictions("pbc")
        band = pbc.loc[pbc["region"].eq(region) & pbc["forecast_date"].eq(day)].sort_values("target_hour")
        if not band.empty and {"lower_80", "upper_80"} <= set(band):
            fig.add_scatter(x=band["target_hour"], y=band["lower_80"], line={"width": 0},
                            showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=band["target_hour"], y=band["upper_80"], line={"width": 0}, fill="tonexty",
                            fillcolor="rgba(13,148,136,0.14)", name="PBC 80% interval")
    for index, model in enumerate(chosen):
        frame = load_predictions(model)
        rows = frame.loc[frame["region"].eq(region) & frame["forecast_date"].eq(day)].sort_values("target_hour")
        if rows.empty:
            continue
        if actual is None and rows["actual"].notna().any():
            actual = rows[["target_hour", "actual"]]
        fig.add_scatter(x=rows["target_hour"], y=rows["predicted"], name=label(model), mode="lines",
                        line={"color": PALETTE[index % len(PALETTE)], "width": 2.4 if index == 0 else 1.8})
        matched = rows.dropna(subset=["actual"])
        if not matched.empty:
            error = (matched["predicted"] - matched["actual"]).abs()
            scores.append({"Model": label(model), "MAE (MW)": round(error.mean(), 1),
                           "MAPE (%)": round((error / matched["actual"]).mean() * 100, 2)})
    if actual is not None:
        fig.add_scatter(x=actual["target_hour"], y=actual["actual"], name="Actual demand", mode="lines+markers",
                        line={"color": ACTUAL, "width": 2.4, "dash": "dot"}, marker={"size": 5})
    thresholds = load_json(RESULTS_DIR / "peak_thresholds.json") or {}
    if thresholds.get(region):
        fig.add_hline(y=thresholds[region], line={"color": THRESHOLD, "dash": "dash", "width": 1.2})
    style_figure(fig, height=440)
    fig.update_xaxes(title="Local hour", dtick=2)
    st.plotly_chart(fig, width="stretch")

    holidays = load_holidays()
    if holidays is not None:
        names = holidays.loc[holidays["region"].eq(region)
                             & pd.to_datetime(holidays["date"]).dt.date.eq(day), "holiday_name"]
        if not names.empty:
            st.info(f"{day:%B %d, %Y} is a holiday in {REGION_NAMES[region]}: {', '.join(names)}.")
    if scores:
        st.dataframe(pd.DataFrame(scores).sort_values("MAE (MW)"), hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main():
    if not DEPS_OK:
        raise SystemExit(f"Missing dependency: {_MISSING}. Run: pip install streamlit plotly")
    st.set_page_config(page_title="Canada Electricity Demand Forecast", page_icon=":material/bolt:",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    from dashboard.benchmark_pages import forecast_page, results_page, ablation_page, methods_page
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">Canada day-ahead<br>electricity demand</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="sidebar-note">Verified HFED forecasts, model comparisons and '
                    'weather/calendar/graph ablations for eight provinces.</div>', unsafe_allow_html=True)
    navigation = st.navigation([
        st.Page(forecast_page, title="HFED forecasts", icon=":material/bolt:", default=True),
        st.Page(results_page, title="Model results", icon=":material/leaderboard:"),
        st.Page(ablation_page, title="Ablation study", icon=":material/query_stats:"),
        st.Page(methods_page, title="Proposal and methods", icon=":material/description:"),
    ])
    navigation.run()


if __name__ == "__main__":
    main()
