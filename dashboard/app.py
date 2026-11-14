"""Interactive Streamlit dashboard for STGNN electricity demand forecasting.

Sections:
  A. Overview
  B. Regional Forecast (24-hour chart with actual, predicted, threshold)
  C. Regional Comparison
  D. Weather & Calendar
  E. Model Comparison (MAE / RMSE / MAPE table)
  F. High-Demand Analysis
  G. Forecast Uncertainty (if probabilistic predictions available)
  H. Forecast from Uploaded Data

Historical sections display saved predictions. The upload page generates
forecasts from user-supplied observations.

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

# Add project root to path so src.* imports work
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


# ---------------------------------------------------------------------------
# Config & data helpers
# ---------------------------------------------------------------------------

RESULTS_DIR = ROOT / "results"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
HOLIDAYS_PATH = ROOT / "data" / "holidays" / "holidays_df.parquet"

REGION_NAMES = {
    "AB": "Alberta",
    "BC": "British Columbia",
    "NB": "New Brunswick",
    "NL": "Newfoundland & Labrador",
    "NS": "Nova Scotia",
    "ON": "Ontario",
    "PE": "Prince Edward Island",
    "QC": "Quebec",
    "SK": "Saskatchewan",
    "YK": "Yukon",
}


@st.cache_data(show_spinner=False)
def load_predictions() -> dict[str, pd.DataFrame]:
    preds = {}
    for path in sorted(RESULTS_DIR.glob("predictions_*.parquet")):
        name = path.stem.replace("predictions_", "")
        preds[name] = pd.read_parquet(path)
    for path in sorted((MODELS_DIR / "stgnn").glob("predictions_*.parquet") if (MODELS_DIR / "stgnn").exists() else []):
        name = "stgnn_" + path.stem.replace("predictions_", "")
        preds[name] = pd.read_parquet(path)
    return preds


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame | None:
    path = RESULTS_DIR / "model_comparison.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_peak_thresholds() -> dict:
    path = RESULTS_DIR / "peak_thresholds.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_high_demand() -> pd.DataFrame | None:
    path = RESULTS_DIR / "high_demand_flags.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_holidays() -> pd.DataFrame | None:
    return pd.read_parquet(HOLIDAYS_PATH) if HOLIDAYS_PATH.exists() else None


def _not_ready(msg: str = "No data available yet.  Run the training pipeline first."):
    st.warning(msg)
    st.stop()


@st.cache_data(show_spinner=False, max_entries=16)
def load_uploaded_forecast(history, region, forecast_date):
    from src.data.upload_forecast import run_uploaded_forecast

    return run_uploaded_forecast(history, region, forecast_date, ROOT)


@st.cache_data(ttl=900, max_entries=16, show_spinner=False)
def load_hfed_forecast_data(region, forecast_date, config):
    from src.data.hfed_api import fetch_forecast_observations

    return fetch_forecast_observations(region, forecast_date, config)


def render_uploaded_forecast():
    from src.data.hfed_loader import load_config
    from src.data.upload_forecast import (
        has_cached_holidays,
        TIME_ZONES,
        hourly_observations,
        local_time,
        read_observations,
    )
    from src.evaluation.metrics import mae, rmse

    st.title("Forecast and Compare")
    source = st.selectbox("Data source", ["Statistics Canada HFED API", "CSV upload"])
    actual_file = None
    ground_truth = None

    try:
        config = load_config(ROOT / "config" / "config.yaml")
        if source == "CSV upload":
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
        else:
            regions = config["download"]["regions"]
            region = st.selectbox("Province", regions, index=regions.index("ON"),
                                  format_func=REGION_NAMES.get, key="api_province")
            today = pd.Timestamp.now(tz=TIME_ZONES[region]).date()
            forecast_date = st.date_input("Forecast date (province local time)", today, key="api_date")
            with st.spinner("Downloading forecast history and actual demand from HFED…"):
                history, ground_truth, unavailable = load_hfed_forecast_data(region, forecast_date, config)
            if unavailable:
                st.info("HFED data unavailable for " + ", ".join(unavailable) + " in this request.")
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
        st.plotly_chart(fig, use_container_width=True, key=f"uploaded_forecast_{name}")
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
# Main app
# ---------------------------------------------------------------------------

def main():
    if not DEPS_OK:
        st.error(f"Missing dependency: {_MISSING}\nRun: pip install streamlit plotly")
        st.stop()

    st.set_page_config(
        page_title="Canada STGNN — Electricity Demand Forecasting",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Go to",
        [
            "A. Overview",
            "B. Regional Forecast",
            "C. Regional Comparison",
            "D. Weather & Calendar",
            "E. Model Comparison",
            "F. High-Demand Analysis",
            "G. Forecast Uncertainty",
            "H. Forecast from Uploaded Data",
        ],
    )

    if section == "H. Forecast from Uploaded Data":
        render_uploaded_forecast()
        return

    preds = load_predictions()
    comparison = load_comparison()
    thresholds = load_peak_thresholds()
    high_demand_df = load_high_demand()
    holidays_df = load_holidays()

    # Determine available models and best model
    if comparison is not None and "model" in comparison.columns:
        model_names = comparison["model"].tolist()
        best_model = model_names[0]
    elif preds:
        model_names = list(preds.keys())
        best_model = model_names[0]
    else:
        model_names = []
        best_model = None

    # Determine available dates and regions from best model
    available_dates: list[date] = []
    available_regions: list[str] = []
    if best_model and best_model in preds:
        df0 = preds[best_model]
        df0["forecast_date"] = pd.to_datetime(df0["forecast_date"]).dt.date
        available_dates = sorted(df0["forecast_date"].unique())
        available_regions = sorted(df0["region"].unique())

    # -----------------------------------------------------------------------
    # A. Overview
    # -----------------------------------------------------------------------
    if section == "A. Overview":
        st.title("Canada STGNN — Day-Ahead Electricity Demand Forecasting")
        st.markdown(
            "Spatio-Temporal Graph Neural Network forecast for all consistently available "
            "HFED regions.  Forecasts are issued at 23:00 local time for the following 24 hours."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Available Regions", len(available_regions) if available_regions else "—")
        col2.metric("Available Models", len(model_names) if model_names else "—")
        col3.metric("Forecast Days", len(available_dates) if available_dates else "—")

        if high_demand_df is not None:
            n_high = int(high_demand_df["is_high_demand"].sum()) if "is_high_demand" in high_demand_df.columns else 0
            col4.metric("Total High-Demand Hours", n_high)

        if not preds:
            _not_ready()

        st.subheader("Regions Included")
        region_table = pd.DataFrame(
            [
                {
                    "Code": r,
                    "Name": REGION_NAMES.get(r, r),
                    "Threshold (MW)": f"{thresholds.get(r, 'N/A'):.0f}" if isinstance(thresholds.get(r), float) else "N/A",
                }
                for r in available_regions
            ]
        )
        st.dataframe(region_table, use_container_width=True)

        if comparison is not None:
            st.subheader("Model Performance Summary")
            disp = comparison[["model", "mae", "rmse", "mape"]].copy()
            disp.columns = ["Model", "MAE (MW)", "RMSE (MW)", "MAPE (%)"]
            disp = disp.round(2)
            st.dataframe(disp, use_container_width=True)

    # -----------------------------------------------------------------------
    # B. Regional Forecast
    # -----------------------------------------------------------------------
    elif section == "B. Regional Forecast":
        st.title("B. Regional Forecast — 24-Hour Day-Ahead")

        if not preds:
            _not_ready()

        col1, col2, col3 = st.columns(3)
        with col1:
            selected_region = st.selectbox(
                "Region", available_regions,
                format_func=lambda r: f"{r} — {REGION_NAMES.get(r, r)}"
            )
        with col2:
            selected_date = st.selectbox("Forecast Date", available_dates[::-1])
        with col3:
            selected_model = st.selectbox("Model", model_names, index=0)

        if selected_model not in preds:
            st.warning(f"Model '{selected_model}' predictions not loaded.")
            st.stop()

        df = preds[selected_model].copy()
        df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
        row_df = df[
            (df["region"] == selected_region)
            & (df["forecast_date"] == selected_date)
        ].sort_values("target_hour")

        if row_df.empty:
            st.warning("No forecast available for this region / date / model combination.")
        else:
            hours = row_df["target_hour"].values
            pred = row_df["predicted"].values
            actual = row_df["actual"].values if "actual" in row_df.columns else None
            thr = thresholds.get(selected_region)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hours, y=pred, mode="lines+markers",
                name="Predicted", line={"color": "#1f77b4", "width": 2},
            ))
            if actual is not None and not np.all(np.isnan(actual.astype(float))):
                fig.add_trace(go.Scatter(
                    x=hours, y=actual, mode="lines+markers",
                    name="Actual", line={"color": "#2ca02c", "width": 2, "dash": "dot"},
                ))
            if thr:
                fig.add_hline(
                    y=thr, line_dash="dash", line_color="red",
                    annotation_text=f"High-demand threshold ({thr:.0f} MW)",
                    annotation_position="top right",
                )
            # Shade high-demand hours
            if thr:
                for h, p in zip(hours, pred):
                    if p >= thr:
                        fig.add_vrect(
                            x0=h - 0.5, x1=h + 0.5,
                            fillcolor="red", opacity=0.08, line_width=0,
                        )

            fig.update_layout(
                title=f"{REGION_NAMES.get(selected_region, selected_region)} — "
                      f"24-Hour Forecast for {selected_date}",
                xaxis_title="Hour of day (local time)",
                yaxis_title="Electricity Demand (MW)",
                hovermode="x unified",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            tbl = row_df[["target_hour", "predicted"]].copy()
            if "actual" in row_df.columns:
                tbl["actual"] = row_df["actual"].values
                tbl["error_mw"] = tbl["actual"] - tbl["predicted"]
            if thr:
                tbl["is_high_demand"] = tbl["predicted"] >= thr
            tbl = tbl.round(1)
            tbl.columns = [c.replace("_", " ").title() for c in tbl.columns]
            st.dataframe(tbl, use_container_width=True)

    # -----------------------------------------------------------------------
    # C. Regional Comparison
    # -----------------------------------------------------------------------
    elif section == "C. Regional Comparison":
        st.title("C. Regional Demand Comparison")

        if not preds:
            _not_ready()

        selected_model = st.selectbox("Model", model_names)
        selected_date = st.selectbox("Forecast Date", available_dates[::-1])
        selected_regions = st.multiselect(
            "Regions to compare", available_regions, default=available_regions[:4]
        )

        if not selected_regions:
            st.info("Select at least one region.")
        else:
            df = preds[selected_model].copy()
            df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
            subset = df[
                (df["forecast_date"] == selected_date)
                & (df["region"].isin(selected_regions))
            ]

            fig = go.Figure()
            for region in selected_regions:
                rdf = subset[subset["region"] == region].sort_values("target_hour")
                if rdf.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=rdf["target_hour"].values,
                    y=rdf["predicted"].values,
                    mode="lines+markers",
                    name=f"{region} — {REGION_NAMES.get(region, region)}",
                ))

            fig.update_layout(
                title=f"Regional Demand Comparison — {selected_date}",
                xaxis_title="Hour (local)",
                yaxis_title="Predicted Demand (MW)",
                hovermode="x unified",
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # D. Weather & Calendar
    # -----------------------------------------------------------------------
    elif section == "D. Weather & Calendar":
        st.title("D. Calendar Features & Holidays")

        selected_region = st.selectbox(
            "Region", available_regions,
            format_func=lambda r: f"{r} — {REGION_NAMES.get(r, r)}"
        )
        selected_year = st.selectbox("Year", list(range(2019, 2027))[::-1])

        if holidays_df is not None:
            hdf = holidays_df[
                (holidays_df["region"] == selected_region)
                & (pd.to_datetime(holidays_df["date"]).dt.year == selected_year)
            ].copy()
            hdf["date"] = pd.to_datetime(hdf["date"])
            hdf = hdf.sort_values("date")

            st.subheader(f"Public Holidays — {selected_region} ({selected_year})")
            if hdf.empty:
                st.info("No holidays found for this region/year.")
            else:
                disp = hdf[["date", "holiday_name", "is_national"]].copy()
                disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
                disp.columns = ["Date", "Holiday Name", "National"]
                st.dataframe(disp, use_container_width=True)
        else:
            st.info("Holiday data not available.  Run the data pipeline first.")

        # Calendar feature explanation
        st.subheader("Calendar Features Used in Models")
        st.markdown(
            """
| Feature | Description |
|---|---|
| `hour` | Hour of day (0–23), local time |
| `day_of_week` | 0=Monday, 6=Sunday |
| `is_weekend` | 1 if Saturday or Sunday |
| `month` | 1–12 |
| `day_of_year` | 1–365/366 |
| `week_of_year` | ISO week number |
| `is_holiday` | 1 if national/provincial holiday |
| `is_day_before_holiday` | 1 if next day is a holiday |
| `is_day_after_holiday` | 1 if previous day was a holiday |
| Cyclical encodings | sin/cos transforms for hour, day-of-week, day-of-year, month |
"""
        )

    # -----------------------------------------------------------------------
    # E. Model Comparison
    # -----------------------------------------------------------------------
    elif section == "E. Model Comparison":
        st.title("E. Model Performance Comparison")

        if comparison is None:
            _not_ready("Evaluation results not available.  Run the evaluation pipeline first.")

        st.subheader("Overall Performance (Test Set)")
        disp = comparison.copy()
        # Rename for display
        rename_map = {
            "mae": "MAE (MW)",
            "rmse": "RMSE (MW)",
            "mape": "MAPE (%)",
            "peak_mae": "Peak MAE (MW)",
            "peak_rmse": "Peak RMSE (MW)",
        }
        disp = disp.rename(columns=rename_map)
        st.dataframe(disp.round(2), use_container_width=True)

        # Bar chart
        if "MAE (MW)" in disp.columns:
            fig = px.bar(
                disp,
                x="model",
                y="MAE (MW)",
                title="MAE by Model",
                color="model",
                labels={"model": "Model", "MAE (MW)": "MAE (MW)"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Per-region breakdown
        st.subheader("MAE by Region and Model")
        region_rows = []
        for model_name in model_names:
            path = RESULTS_DIR / f"metrics_by_region_{model_name}.csv"
            if path.exists():
                rdf = pd.read_csv(path)
                rdf["model"] = model_name
                region_rows.append(rdf)
        if region_rows:
            region_comp = pd.concat(region_rows, ignore_index=True)
            pivot = region_comp.pivot_table(
                index="region", columns="model", values="mae", aggfunc="mean"
            ).round(1)
            st.dataframe(pivot, use_container_width=True)

            fig2 = px.bar(
                region_comp,
                x="region",
                y="mae",
                color="model",
                barmode="group",
                title="MAE by Region and Model",
                labels={"mae": "MAE (MW)", "region": "Region"},
            )
            st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------------------------------------------
    # F. High-Demand Analysis
    # -----------------------------------------------------------------------
    elif section == "F. High-Demand Analysis":
        st.title("F. High-Demand Analysis")

        if high_demand_df is None:
            _not_ready("High-demand flags not available.  Run the high-demand detection step first.")

        from src.high_demand.detector import high_demand_summary
        summary = high_demand_summary(high_demand_df)

        st.subheader("High-Demand Summary by Region")
        disp = summary.rename(columns={
            "region": "Region",
            "n_forecast_hours": "Total Forecast Hours",
            "n_high_demand_hours": "High-Demand Hours",
            "high_demand_pct": "High-Demand %",
            "peak_predicted_mw": "Peak Predicted (MW)",
            "threshold_mw": "Threshold (MW)",
            "mean_error_mw": "Mean Error (MW)",
        })
        st.dataframe(disp.round(1), use_container_width=True)

        # Bar chart
        fig = px.bar(
            summary,
            x="region",
            y="high_demand_pct",
            title="High-Demand Frequency by Region (%)",
            labels={"region": "Region", "high_demand_pct": "High-Demand Hours (%)"},
            color="high_demand_pct",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Region drill-down
        st.subheader("Region Drill-Down")
        sel_region = st.selectbox(
            "Select Region",
            sorted(high_demand_df["region"].unique()),
            format_func=lambda r: f"{r} — {REGION_NAMES.get(r, r)}",
        )
        rdf = high_demand_df[high_demand_df["region"] == sel_region].copy()
        rdf["forecast_date"] = pd.to_datetime(rdf["forecast_date"]).dt.date

        # Scatter: predicted vs threshold, coloured by is_high_demand
        if "predicted_demand_mw" in rdf.columns:
            fig2 = px.scatter(
                rdf,
                x="target_hour",
                y="predicted_demand_mw",
                color="is_high_demand",
                color_discrete_map={True: "crimson", False: "steelblue"},
                title=f"{REGION_NAMES.get(sel_region, sel_region)} — Predicted Demand by Hour",
                labels={
                    "target_hour": "Forecast Hour",
                    "predicted_demand_mw": "Predicted (MW)",
                    "is_high_demand": "High Demand",
                },
                opacity=0.5,
            )
            thr = thresholds.get(sel_region)
            if thr:
                fig2.add_hline(
                    y=thr, line_dash="dash", line_color="red",
                    annotation_text=f"Threshold: {thr:.0f} MW",
                )
            st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------------------------------------------
    # G. Forecast Uncertainty
    # -----------------------------------------------------------------------
    elif section == "G. Forecast Uncertainty":
        st.title("G. Forecast Uncertainty (Probabilistic Extension)")
        st.info(
            "This section shows prediction intervals and exceedance probabilities "
            "when the probabilistic bias-correction module (Phase 8) has been trained."
        )

        prob_path = RESULTS_DIR / "probabilistic_predictions.parquet"
        if not prob_path.exists():
            st.warning(
                "Probabilistic predictions not found.\n\n"
                "Complete the core forecasting pipeline (Phases 1–7) first, then run "
                "the probabilistic bias correction module (Phase 8) to generate "
                "prediction intervals."
            )
        else:
            prob_df = pd.read_parquet(prob_path)
            st.success("Probabilistic predictions loaded.")

            sel_region = st.selectbox(
                "Region",
                sorted(prob_df["region"].unique()),
                format_func=lambda r: f"{r} — {REGION_NAMES.get(r, r)}",
            )
            prob_df["forecast_date"] = pd.to_datetime(prob_df["forecast_date"]).dt.date
            sel_date = st.selectbox("Forecast Date", sorted(prob_df["forecast_date"].unique())[::-1])

            rdf = prob_df[
                (prob_df["region"] == sel_region)
                & (prob_df["forecast_date"] == sel_date)
            ].sort_values("target_hour")

            if rdf.empty:
                st.warning("No probabilistic predictions for this selection.")
            else:
                fig = go.Figure()
                if "lower_80" in rdf.columns and "upper_80" in rdf.columns:
                    fig.add_trace(go.Scatter(
                        x=pd.concat([rdf["target_hour"], rdf["target_hour"][::-1]]),
                        y=pd.concat([rdf["upper_80"], rdf["lower_80"][::-1]]),
                        fill="toself", fillcolor="rgba(31,119,180,0.15)",
                        line={"color": "rgba(255,255,255,0)"},
                        name="80% PI",
                    ))
                fig.add_trace(go.Scatter(
                    x=rdf["target_hour"], y=rdf["predicted"],
                    mode="lines", name="Predicted", line={"color": "#1f77b4", "width": 2},
                ))
                if "actual" in rdf.columns:
                    fig.add_trace(go.Scatter(
                        x=rdf["target_hour"], y=rdf["actual"],
                        mode="lines+markers", name="Actual",
                        line={"color": "#2ca02c", "dash": "dot"},
                    ))
                thr = thresholds.get(sel_region)
                if thr:
                    fig.add_hline(y=thr, line_dash="dash", line_color="red",
                                  annotation_text="High-demand threshold")
                fig.update_layout(
                    title=f"Probabilistic Forecast — {sel_region} — {sel_date}",
                    xaxis_title="Hour (local)",
                    yaxis_title="Demand (MW)",
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
