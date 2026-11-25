"""Dashboard views backed exclusively by the corrected benchmark artifacts."""
from __future__ import annotations

from datetime import timedelta
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.benchmark_forecast import (
    BenchmarkForecaster, MAIN_RUN, REGIONS, SPECS, TIMEZONES,
    build_request, local_day_grid, request_observations,
)

ROOT = Path(__file__).resolve().parents[1]
HFED = ROOT / "results" / MAIN_RUN
BRAZIL = ROOT / "results/grid_aware_confirm_v3"
CONTROLS = ROOT / "results/hfed_temporal_controls_v1"
REGION_NAMES = {"AB": "Alberta", "BC": "British Columbia", "NB": "New Brunswick", "NL": "Newfoundland and Labrador",
                "NS": "Nova Scotia", "ON": "Ontario", "PE": "Prince Edward Island", "QC": "Quebec"}
LABELS = {spec.model: spec.label for key, spec in SPECS.items() if key not in ["demand", "weather"]}
PERIODS = {"test2025": "2025 · independent holdout", "test2024": "Jul–Dec 2024 · previously inspected"}
COLORS = ["#0d9488", "#6366f1", "#f97316", "#0284c7", "#a855f7", "#64748b", "#e11d48", "#65a30d"]


@st.cache_data(show_spinner=False)
def csv(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def document(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hfed_csv(filename):
    primary = csv(HFED / filename)
    additional = CONTROLS / filename
    if additional.exists():
        if document(CONTROLS / "plan.json")["cache_sha256"] != document(HFED / "plan.json")["cache_sha256"]:
            raise ValueError("Supplementary controls use a different HFED snapshot")
        return pd.concat([primary, csv(additional)], ignore_index=True)
    return primary


@st.cache_resource(show_spinner=False)
def forecaster():
    return BenchmarkForecaster(ROOT)


def figure(fig, height=420, y="Demand (MW)"):
    fig.update_layout(height=height, margin=dict(l=15, r=15, t=55, b=25),
        legend=dict(orientation="h", y=1.02, yanchor="bottom"), hovermode="x unified", yaxis_title=y)
    return fig


def metric_table(frame):
    out = frame.copy()
    out["Model"] = out.model.map(LABELS).fillna(out.model)
    for key, title in [("mae", "MAE (MW)"), ("rmse", "RMSE (MW)"), ("mape", "MAPE (%)")]:
        out[title] = [f"{mean:,.2f} ± {sd:.2f}" for mean, sd in zip(out[f"{key}_mean"], out[f"{key}_std"].fillna(0))]
    return out[["Model", "MAE (MW)", "RMSE (MW)", "MAPE (%)"]]


def daily_scores(rows):
    observed = rows.loc[np.isfinite(rows.actual)].copy()
    if observed.empty:
        return None
    error = observed.predicted - observed.actual
    valid = observed.actual.abs().ge(10)
    return {"MAE (MW)": error.abs().mean(), "RMSE (MW)": np.sqrt(np.square(error).mean()),
            "MAPE (%)": (error[valid].abs() / observed.actual[valid].abs()).mean() * 100,
            "Observed hours": len(observed)}


@st.cache_data(ttl=900, max_entries=32, show_spinner=False)
def run_forecast(region, day, source, keys, seeds, uploads):
    uploaded = None
    if uploads:
        from src.data.hfed_loader import load_config
        from src.data.upload_forecast import read_observations
        uploaded = read_observations([BytesIO(value) for value in uploads], load_config(ROOT / "config/config.yaml"))
    use_api = source == "Refresh from HFED API" or (source == "Automatic" and pd.Timestamp(day).year > 2025)
    demand, weather, holidays, origin, notices = request_observations(ROOT, region, day, use_api, uploaded)
    engine = forecaster()
    sample, issue, targets, history = build_request(engine.data, region, day, demand, weather, holidays)
    predictions = engine.predict(sample, keys, seeds)
    return predictions, history, issue, targets, origin, notices, pd.Timestamp(day).date() in holidays


def forecast_page():
    st.title("HFED day-ahead forecasts")
    st.write("Choose the Canadian province and day to forecast. Each forecast is issued at **23:00 local time on the previous day** and covers all 24 next-day hours.")
    source = st.selectbox("Observation source", ["Automatic", "Frozen archive", "Refresh from HFED API", "Upload HFED CSV"],
        help="Automatic uses the verified archive through 2025 and the HFED API for later dates. CSV upload is optional; choose it only to use your own native HFED observations.")
    st.caption("No CSV upload is needed. Automatic loads HFED observations for the selected date; you can optionally select Upload HFED CSV to provide your own history.")
    with st.form("verified_forecast_form"):
        a, b = st.columns(2)
        region = a.selectbox("Province", REGIONS, index=REGIONS.index("ON"), format_func=lambda value: REGION_NAMES[value])
        now = pd.Timestamp.now(tz=TIMEZONES[region])
        maximum = now.date() + timedelta(days=int(now.hour >= 23))
        day = b.date_input("Forecast date (local)", value=pd.Timestamp("2025-01-15").date(), min_value=pd.Timestamp("2024-01-01").date(), max_value=maximum)
        uploads = st.file_uploader("HFED history CSVs", type="csv", accept_multiple_files=True,
            help="Include at least eight days of native HFED demand for the selected province.") if source == "Upload HFED CSV" else []
        mode = st.radio("Compare", ["Models and baselines", "Feature ablations"], horizontal=True)
        # The same picker supports models and the four fixed-architecture ablations.
        chosen = st.multiselect("Forecast curves", list(SPECS), default=["cnn", "lstm", "xgboost", "daily"], format_func=lambda key: SPECS[key].label,
            help="For ablations, the four seasonal GRU stages are selected automatically. The selected CNN is included as a reference.")
        seed_view = st.selectbox("Forecast member", ["Seed 42", "Seed 123", "Seed 456", "Mean of three seeds"],
            help="Benchmark tables average scores over three independently trained models. A mean forecast is an interactive ensemble, not the benchmark's three-seed mean score.")
        submitted = st.form_submit_button("Generate HFED forecast", type="primary")
    if submitted:
        if source == "Upload HFED CSV" and not uploads:
            st.session_state.pop("verified_request", None)
            st.warning("Upload native HFED observations before generating this forecast.")
            return
        issue, _, targets = local_day_grid(day, region)
        if len(targets) != 24:
            st.session_state.pop("verified_request", None)
            st.warning("This date has 23 or 25 hours because of daylight saving. Select a 24-hour day.")
            return
        if issue > pd.Timestamp.now(tz="UTC").tz_localize(None):
            st.session_state.pop("verified_request", None)
            st.warning("The previous-day 23:00 issue time has not arrived yet. Select an earlier forecast date.")
            return
        keys = ("cnn", "demand", "weather", "calendar", "graph") if mode == "Feature ablations" else tuple(chosen)
        if not keys:
            st.session_state.pop("verified_request", None)
            st.warning("Select at least one forecast curve.")
            return
        seeds = (42, 123, 456) if seed_view.startswith("Mean") else (int(seed_view.split()[-1]),)
        try:
            if source == "Refresh from HFED API":
                run_forecast.clear()
            with st.spinner("Loading causal HFED history and running the verified checkpoints…"):
                result = run_forecast(region, day, source, keys, seeds,
                                      tuple(file.getvalue() for file in uploads) if source == "Upload HFED CSV" else ())
            st.session_state["verified_request"] = (region, day, seed_view, result)
        except (ValueError, OSError, RuntimeError, AssertionError) as exc:
            st.session_state.pop("verified_request", None)
            st.error(f"Forecast unavailable: {exc}")
            return
    if "verified_request" not in st.session_state:
        st.info("Select a date and click Generate HFED forecast. Try Ontario on 15 January 2025 to inspect a held-out example with actuals.")
        return
    region, day, seed_view, result = st.session_state["verified_request"]
    raw, history, issue, targets, origin, notices, holiday = result
    st.subheader(f"{REGION_NAMES[region]} · {day:%A, %d %B %Y}")
    local_issue = issue.tz_localize("UTC").tz_convert(TIMEZONES[region])
    st.caption(f"Issued {local_issue:%d %b %Y %H:%M %Z} · {origin} · {seed_view}. No forecast-day demand or realized future weather is used as input.")
    if day < pd.Timestamp("2024-07-01").date():
        st.info("This date belongs to the validation period used for checkpoint selection, not the held-out benchmark.")
    elif day.year >= 2026:
        st.info("This is inference on a later date. The reported 2025 benchmark scores describe the held-out archive, not this day's guaranteed accuracy.")
    for notice in notices:
        st.info(notice)
    if holiday:
        st.info("This forecast date is a public holiday in the regional calendar.")
    threshold = forecaster().thresholds[region]
    shown = raw.groupby(["key", "model", "hour"], as_index=False).agg(predicted=("predicted", "mean"), actual=("actual", "first"))
    primary_key = "cnn" if "cnn" in shown.key.values else shown.key.iloc[0]
    primary = shown.loc[shown.key.eq(primary_key)].sort_values("hour")
    peak = primary.loc[primary.predicted.idxmax()]
    a, b, c, d = st.columns(4)
    a.metric("Peak forecast", f"{peak.predicted:,.0f} MW", f"{int(peak.hour):02d}:00 local", delta_color="off")
    b.metric("Daily forecast energy", f"{primary.predicted.sum()/1000:,.2f} GWh")
    c.metric("High-demand hours", f"{primary.predicted.ge(threshold).sum()} / 24")
    scores = daily_scores(primary)
    d.metric("Day MAE", f"{scores['MAE (MW)']:,.2f} MW" if scores else "Pending actuals",
             f"{int(scores['Observed hours'])}/24 observed" if scores else "No target labels published", delta_color="off")
    fig = go.Figure()
    actual = shown.drop_duplicates("hour").sort_values("hour")
    for index, (_, rows) in enumerate(shown.groupby("model", sort=False)):
        fig.add_scatter(x=rows.hour, y=rows.predicted, name=rows.model.iloc[0], mode="lines+markers",
                        line=dict(color=COLORS[index % len(COLORS)], width=2.5), marker=dict(size=4))
    fig.add_scatter(x=actual.hour, y=actual.actual, name="Actual demand", mode="lines+markers",
                    line=dict(color="#f59e0b", width=3, dash="dot"))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444", annotation_text="Training 90th percentile")
    fig.update_xaxes(title="Forecast-day local hour", dtick=2, range=[0, 23])
    figure(fig, 500)
    fig.update_layout(margin=dict(l=15, r=15, t=20, b=120),
                      legend=dict(orientation="h", y=-.16, yanchor="top", font=dict(size=11)))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Flags indicate unusually high demand relative to the province's training history. They are planning indicators, not capacity limits, outage predictions or dispatch instructions.")
    forecast_tab, error_tab, weather_tab, about_tab = st.tabs(["Hourly predictions", "Errors and flags", "Weather and input history", "Model and data"])
    with forecast_tab:
        table = shown.pivot(index="hour", columns="model", values="predicted")
        table["Actual demand"] = actual.set_index("hour").actual
        st.dataframe(table.round(2), use_container_width=True)
        export = shown.assign(region=region, forecast_date=str(day), issued_at_utc=str(issue),
                              target_at_utc=shown.hour.map(dict(enumerate(targets))), forecast_member=seed_view)
        st.download_button("Download hourly predictions", export.to_csv(index=False), file_name=f"hfed_{region}_{day}.csv", mime="text/csv")
    with error_tab:
        records = []
        for model, rows in shown.groupby("model", sort=False):
            score = daily_scores(rows)
            if score:
                records.append({"Model": model, **score})
        if records:
            st.dataframe(pd.DataFrame(records).round(2), hide_index=True, use_container_width=True)
        else:
            st.info("Actual demand is not yet available. The 24-hour forecasts are still shown; errors remain pending.")
        flagged = primary.assign(error_mw=primary.predicted-primary.actual,
                                 high_demand=primary.predicted.ge(threshold))
        st.dataframe(flagged[["hour", "predicted", "actual", "error_mw", "high_demand"]].round(2), hide_index=True, use_container_width=True)
        st.caption("MAPE excludes |actual| < 10 MW. No missing actuals are imputed. Displayed daily errors are for the chosen seed or ensemble, not the aggregate benchmark score.")
    with weather_tab:
        history = history.copy()
        history["Local time"] = history.time_utc.dt.tz_localize("UTC").dt.tz_convert(TIMEZONES[region]).dt.tz_localize(None)
        count = int(history.temperature_c.notna().sum())
        st.write(f"Known input weather: **{count}/24 temperature readings**. Latest input interval ends at the 23:00 issue time; station observations have a one-hour reporting buffer.")
        chart = px.line(history, x="Local time", y="temperature_c", markers=True, labels={"temperature_c": "Temperature (°C)"})
        st.plotly_chart(figure(chart, 260, "Temperature (°C)"), use_container_width=True)
        st.dataframe(history.drop(columns="time_utc").round(2), hide_index=True, use_container_width=True)
        st.caption("Weather is a representative station, not province-wide coverage. Missing inputs are masked. The selected CNN uses demand and calendars; weather affects the GRU variants.")
    with about_tab:
        st.write("The selected residual CNN learns corrections to causal seasonal demand lags, with province-specific scaling and known calendars. The graph extension retains the frozen graph/GRU design and adds a seasonal residual head.")
        st.write("Ablation curves use separately trained checkpoints with fixed architecture: demand → weather → calendar → geographic graph. The CNN is an additional reference, not part of that feature ladder.")
        st.caption("Training: before 2024; validation: Jan–Jun 2024, with boundary purging. Archived demand is assumed available at interval end; exact historical publication delays and revisions are not reproduced.")
        st.caption(f"Inference device: {forecaster().device}. CPU fallback can differ slightly from GPU benchmark predictions because of floating-point implementation differences.")


def results_page():
    st.title("Model results and paper comparison")
    st.write("Evidence from the corrected, frozen experiments. Forecasts and reported scores are computed from saved model outputs.")
    hfed_tab, paper_tab = st.tabs(["HFED · Canadian results", "Brazil · benchmark evidence only"])
    with hfed_tab:
        period = st.selectbox("HFED evaluation period", list(PERIODS), format_func=PERIODS.get)
        table = hfed_csv("metrics_mean_std.csv")
        selected = table.loc[table.period.eq(period)].sort_values("mae_mean")
        cnn = selected.loc[selected.model.eq("residual_identity")].iloc[0]
        reference = selected.loc[selected.model.str.startswith("grid_aware")].sort_values("mae_mean").iloc[0]
        xgb = selected.loc[selected.model.eq("xgboost")].iloc[0]
        coverage = csv(HFED / "regional_coverage.csv").query("period == @period")
        a, b, c, d = st.columns(4)
        a.metric("Selected CNN MAE", f"{cnn.mae_mean:,.2f} MW", f"± {cnn.mae_std:.2f} across seeds", delta_color="off")
        b.metric("vs best Grid-Aware", f"{100*(1-cnn.mae_mean/reference.mae_mean):.1f}% lower MAE")
        c.metric("vs XGBoost", f"{100*(1-cnn.mae_mean/xgb.mae_mean):.1f}% lower MAE")
        d.metric("Matched observed hours", f"{int(coverage.n_obs.sum()):,}")
        bars = selected.assign(Model=selected.model.map(LABELS), Family=np.where(selected.model.eq("residual_identity"), "Selected CNN", "Comparison"))
        chart = px.bar(bars.iloc[::-1], x="mae_mean", y="Model", error_x="mae_std", orientation="h", color="Family",
                       color_discrete_map={"Selected CNN": "#0d9488", "Comparison": "#94a3b8"}, labels={"mae_mean": "MAE (MW)"})
        st.plotly_chart(figure(chart, 430, ""), use_container_width=True)
        st.dataframe(metric_table(selected), hide_index=True, use_container_width=True)
        st.caption("Mean ± sample standard deviation across seeds 42, 123 and 456. These are independent-run scores, not an ensemble forecast. Temporal-only LSTM/GRU are supplementary proposal controls added after the primary test was inspected, using fixed published RNN settings and validation-only selection. They are descriptive comparisons, not new blinded confirmations.")
        detail_tabs = st.tabs(["Paired evidence", "By province and coverage", "Peak demand", "Weather conditions", "Downloads and provenance"])
        with detail_tabs[0]:
            paired = [r for r in document(HFED / "paired_comparisons.json") if r["period"] == period and r["candidate"] == "residual_identity"]
            if (CONTROLS / "paired_with_selected.json").exists():
                paired += [r for r in document(CONTROLS / "paired_with_selected.json") if r["period"] == period]
            rows = [{"Reference": LABELS.get(r["reference"], r["reference"]), "MAE reduction (%)": r["improvement_percent"],
                     "CNN minus reference (MW)": r["paired_mae_difference"], "95% lower": r["week_block_95pct_ci"][0],
                     "95% upper": r["week_block_95pct_ci"][1], "Wins every seed": r["lower_mae_each_seed"]} for r in paired]
            st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, use_container_width=True)
            st.caption("Negative differences favor the CNN. Seven-day block intervals resample paired dates after averaging differences across the three seeds. They describe test-date variation, not forecast prediction intervals.")
        with detail_tabs[1]:
            regional = hfed_csv("regional_metrics_by_seed.csv").query("period == @period")
            measure = st.selectbox("Regional metric", ["mae", "rmse", "mape", "peak_mae"])
            pivot = regional.pivot_table(index="model", columns="state", values=measure).rename(index=LABELS)
            heat = px.imshow(pivot.round(2), text_auto=".2f", aspect="auto", color_continuous_scale="Teal", labels={"color": measure.upper()})
            st.plotly_chart(figure(heat, 420, ""), use_container_width=True)
            st.dataframe(coverage, hide_index=True, use_container_width=True)
            st.caption("Coverage is uneven, especially Nova Scotia. Missing target labels are excluded equally across methods; this is an available-observation benchmark, not a complete year for every province.")
        with detail_tabs[2]:
            peak = hfed_csv("peak_metrics_by_seed.csv").query("period == @period")
            summary = peak.groupby("model")[["peak_mae", "peak_rmse", "peak_precision", "peak_recall"]].mean().rename(index=LABELS)
            st.dataframe(summary.round(3), use_container_width=True)
            st.write("Peak observations exceed each province's 90th percentile of training demand. The same fixed thresholds drive the date-based forecast flags.")
            st.dataframe(pd.Series(document(HFED / "training_peak_thresholds.json"), name="Threshold (MW)").round(2), use_container_width=True)
        with detail_tabs[3]:
            weather = csv(HFED / "weather_regime_metrics_by_seed.csv").query("period == @period")
            show = weather.loc[weather.model.isin(["residual_identity", "grid_aware_lstm", "xgboost"])].groupby(["model", "weather_regime"])[["mae", "rmse"]].mean().reset_index()
            show["model"] = show.model.map(LABELS)
            st.dataframe(show.round(2), hide_index=True, use_container_width=True)
            st.caption("Cold/hot at issue uses provincial training-only 5th/95th percentiles of observed 24-hour input temperature, with at least 12 observations. These are not labels for realized target-day extremes.")
            st.markdown("**Realized forecast-day temperature extremes · retrospective diagnostic**")
            realized = HFED / "realized_weather_metrics_by_seed.csv"
            if realized.exists():
                target = hfed_csv("realized_weather_metrics_by_seed.csv").query("period == @period")
                summary = target.groupby(["model", "regime"])[["mae", "rmse", "mape", "n_obs", "forecast_days"]].mean().reset_index()
                summary["model"] = summary.model.map(LABELS)
                st.dataframe(summary.round(2), hide_index=True, use_container_width=True)
                st.caption("Observed mean temperature over the 24 target hours, with at least 12 readings. Cold/hot thresholds are provincial training-only 5th/95th percentiles. This post-result diagnostic evaluates realized extreme-temperature days; this future weather never enters forecasts or checkpoint selection.")
                with st.expander("Target-day diagnostic thresholds and scope"):
                    st.json(document(HFED / "realized_weather_thresholds.json"))
        with detail_tabs[4]:
            st.download_button("Download HFED scores", table.to_csv(index=False), file_name="hfed_verified_metrics.csv")
            st.download_button("Download paired evidence", json.dumps(document(HFED / "paired_comparisons.json"), indent=2), file_name="hfed_paired_evidence.json")
            with st.expander("Frozen experiment plan and source provenance"):
                st.json(document(HFED / "plan.json"))
    with paper_tab:
        st.subheader("Grid-Aware paper: 1h, 6h and 24h")
        st.write("Brazil is used only as benchmark evidence here. The interactive prediction tool serves HFED forecasts.")
        horizon = st.select_slider("Brazil forecast horizon", options=[1, 6, 24], value=24, format_func=lambda h: f"+{h} hours")
        period = st.selectbox("Brazil evaluation period", ["test2019", "test2020"], format_func=lambda p: "Jul–Dec 2019" if p == "test2019" else "2020 additional holdout")
        table = csv(BRAZIL / "metrics_mean_std.csv")
        selected = table.loc[table.period.eq(period) & table.horizon.eq(horizon)].sort_values("mae_mean")
        st.markdown("**Controlled comparison: identical held-out keys**")
        st.dataframe(metric_table(selected), hide_index=True, use_container_width=True)
        ours = selected.loc[selected.model.eq("residual_identity")].iloc[0]
        ref = selected.loc[selected.model.str.startswith("grid_aware")].sort_values("mae_mean").iloc[0]
        st.metric("CNN MAE reduction vs best matched reference", f"{100*(1-ours.mae_mean/ref.mae_mean):.1f}%")
        if horizon == 1:
            st.info("At +1h, the CNN lowers MAE/MAPE but has higher RMSE than the GRU. The results do not support superiority on every metric.")
        with st.expander("Original published scores · Table 3", expanded=True):
            published = document(ROOT / "dashboard/published_grid_aware.json")
            frame = pd.DataFrame(published["scores"])
            st.dataframe(frame.loc[frame.horizon.eq(horizon)].drop(columns="horizon"), hide_index=True, use_container_width=True)
            published_rows = frame.loc[frame.horizon.eq(horizon)].sort_values("MAE (MW)")
            comparator = st.selectbox("Published comparator", published_rows.model.tolist())
            reported = published_rows.loc[published_rows.model.eq(comparator)].iloc[0]
            if period == "test2019":
                columns = st.columns(3)
                for column, metric, title in zip(columns, ["mae", "rmse", "mape"], ["MAE (MW)", "RMSE (MW)", "MAPE (%)"]):
                    value = ours[f"{metric}_mean"]
                    change = 100 * (value/reported[title]-1)
                    column.metric(f"Our CNN {title}", f"{value:.2f}", f"{change:+.1f}% vs reported", delta_color="inverse")
            st.markdown(f"Source: [Grid-Aware paper, Table 3]({published['source']}).")
            st.caption("Published values are historical reported scores, not reruns. The public data release and incomplete author code differ from the original pipeline. Lower numbers are a numerical comparison; the matched reimplementations above are the primary evidence. The residual methods also use seven daily lags, regional scaling and known target calendars.")
        st.caption("2019 was inspected during the pilot; 2020 was inspected before the scheduler execution correction. Neither period is newly blinded in the corrected rerun. CNN selection used validation; HFED 2025 remained unused for forecast evaluation until all HFED fits completed.")
        st.download_button("Download all Brazil horizon scores", table.to_csv(index=False), file_name="brazil_verified_metrics.csv")


def ablation_page():
    st.title("Ablation study")
    st.write("Does weather help? Do calendars help? Does the graph add value after both? These comparisons keep the seasonal GRU architecture, sample keys, seed set and training rules fixed.")
    period = st.selectbox("Ablation evaluation period", list(PERIODS), format_func=PERIODS.get)
    rows = csv(HFED / "ablation_metrics_by_seed.csv").query("period == @period")
    order = ["demand", "demand_weather", "demand_weather_calendar", "demand_weather_calendar_graph"]
    names = {"demand": "Demand + seasonal lags", "demand_weather": "+ Past weather", "demand_weather_calendar": "+ Known calendars", "demand_weather_calendar_graph": "+ Geographic graph"}
    grouped = rows.groupby("features")[["mae", "rmse", "mape"]].agg(["mean", "std"]).reindex(order)
    grouped.columns = [f"{a}_{b}" for a,b in grouped.columns]
    grouped = grouped.reset_index()
    grouped["Stage"] = grouped.features.map(names)
    chart = px.bar(grouped, x="Stage", y="mae_mean", error_y="mae_std", color="Stage", color_discrete_sequence=COLORS,
                   labels={"mae_mean": "MAE (MW)"})
    st.plotly_chart(figure(chart, 370, "MAE (MW)"), use_container_width=True)
    st.dataframe(grouped[["Stage", "mae_mean", "mae_std", "rmse_mean", "mape_mean"]].round(2), hide_index=True, use_container_width=True)
    paired = [r for r in document(HFED / "ablation_paired_comparisons.json") if r["period"] == period]
    evidence = [{"Addition": names[r["after"]], "MAE reduction (%)": r["improvement_percent"], "Difference (MW)": r["paired_mae_difference"],
                 "95% lower": r["week_block_95pct_ci"][0], "95% upper": r["week_block_95pct_ci"][1], "Wins every seed": r["lower_mae_each_seed"]} for r in paired]
    st.dataframe(pd.DataFrame(evidence).round(2), hide_index=True, use_container_width=True)
    if period == "test2025":
        st.info("On 2025, weather and calendars improve this GRU. Geographic edges add no reliable gain: their paired interval includes zero. The residual CNN has the lowest overall mean MAE and does not consume weather.")
    else:
        st.info("These 2024 results come from a previously inspected period. The paired table describes each incremental feature effect; use the separate 2025 evaluation for the independent primary holdout.")
    st.caption("Intervals are the predeclared descriptive, unadjusted ablation comparisons. The forecast page's Feature ablations mode runs these same separately trained checkpoints for a chosen HFED date.")
    st.download_button("Download ablation scores", rows.to_csv(index=False), file_name=f"hfed_ablations_{period}.csv")


def methods_page():
    st.title("Proposal, method and data")
    st.subheader("Forecasting method")
    st.write("The selected model anchors each target hour to available seasonal demand and learns a residual correction. Province-specific scaling and known target calendars are included. The graph-preserving alternative adds this residual head to the frozen graph/GRU backbone. Each architecture is retrained separately in Brazil and Canada.")
    st.subheader("Proposal coverage")
    st.dataframe(pd.DataFrame([
        ("23:00 local → next-day 24 hours", "HFED date-based forecasts; interval-end availability; DST transition days excluded"),
        ("Baseline comparison", "Daily/weekly naive, XGBoost, temporal-only LSTM/GRU and adapted Grid-Aware references"),
        ("Demand → weather → calendar → graph", "Fixed-architecture, three-seed ablations and selected-day curves"),
        ("Overall and regional errors", "MAE, RMSE, MAPE; regional heatmap and actual coverage"),
        ("High demand and planning flags", "Training-only provincial 90th percentiles; hourly flags and peak metrics"),
        ("Weather and extreme-temperature days", "Official station history, conditions at issue and realized target-day cold/hot diagnostics; training-only thresholds"),
        ("Different graph constructions", "Current controlled run: geographic vs identity. Interconnection/correlation are earlier exploratory work, not part of this matched result"),
        ("Chronological evaluation", "Fit before 2024, Jan–Jun 2024 validation, Jul–Dec 2024 and unused 2025 tests; boundary purging"),
    ], columns=["Proposal item", "Implemented evidence / scope"]), hide_index=True, use_container_width=True)
    st.subheader("What the results establish")
    st.write("The seasonal graph extension and residual CNN improve the matched Grid-Aware references. The CNN has the lowest overall mean MAE in the evaluated model set. The HFED ablation does not establish extra value from geographic edges after weather and calendar inputs; that negative spatial result is part of the proposal's research question.")
    st.subheader("Data and interpretation")
    st.markdown("- Eight provinces; actual observed HFED demand, official ECCC station weather and Canadian holidays.\n- One station per province is a proxy, not spatially aggregated weather.\n- Missing targets are excluded, never imputed; missing inputs are masked.\n- Newfoundland retains its half-hour UTC alignment.\n- Archive revisions and exact publication delays are not reconstructed; demand is assumed available at interval end.\n- Capacity data are unavailable for HFED, so geographic edges are not claimed to be real electrical connections.\n- High-demand flags support planning; the app does not simulate power flow or predict outages.")
    stations = document(ROOT / "data/weather/eccc/provenance.json")
    st.dataframe(pd.DataFrame([{"Province": r["region"], "Station": r["station"]["STATION_NAME"], "Distance to reference point (km)": r["distance_km"]} for r in stations["selections"]]).round(2), hide_index=True, use_container_width=True)
    with st.expander("Reproducibility and evaluation protocol"):
        protocol = (ROOT / "docs/cross_dataset_validation.md").read_text(encoding="utf-8")
        protocol = "\n".join(line for line in protocol.splitlines() if not line.startswith("!["))
        st.markdown(protocol)
        st.image(str(ROOT / "reports/cross_dataset_validation.png"), use_container_width=True)
