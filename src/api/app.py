"""FastAPI backend for the STGNN electricity demand forecasting dashboard.

Endpoints:
  GET /                         Health check
  GET /regions                  List available HFED regions
  GET /forecast/{region}/{date} 24-hour forecast for a region and date
  GET /models                   Available model names
  GET /comparison               Model performance comparison table
  GET /high-demand/{region}     High-demand flags for a region
  GET /high-demand/summary      Summary across all regions
  GET /holiday/{region}/{year}  Holiday calendar for a region/year
  GET /metrics/{model}          Detailed metrics for one model

All data served by this API is derived from actual model predictions
and actual HFED observations.  No hardcoded illustrative results.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("fastapi not installed.  API server unavailable.")


# ---------------------------------------------------------------------------
# Data loader (lazy, cached after first load)
# ---------------------------------------------------------------------------

class DataStore:
    """Lazy-loading store for processed data and results."""

    def __init__(self, results_dir: str = "results", processed_dir: str = "data/processed"):
        self.results_dir = Path(results_dir)
        self.processed_dir = Path(processed_dir)
        self._predictions: Optional[dict[str, pd.DataFrame]] = None
        self._comparison: Optional[pd.DataFrame] = None
        self._peak_thresholds: Optional[dict] = None
        self._high_demand: Optional[pd.DataFrame] = None
        self._holidays: Optional[pd.DataFrame] = None

    def _parquet_or_none(self, path: Path) -> Optional[pd.DataFrame]:
        if path.exists():
            return pd.read_parquet(path)
        return None

    def get_predictions(self) -> dict[str, pd.DataFrame]:
        if self._predictions is None:
            self._predictions = {}
            for path in self.results_dir.glob("predictions_*.parquet"):
                model_name = path.stem.replace("predictions_", "")
                self._predictions[model_name] = pd.read_parquet(path)
            # Also check models/stgnn for STGNN predictions
            stgnn_dir = Path("models/stgnn")
            for path in stgnn_dir.glob("predictions_*.parquet") if stgnn_dir.exists() else []:
                model_name = "stgnn_" + path.stem.replace("predictions_", "")
                self._predictions[model_name] = pd.read_parquet(path)
        return self._predictions

    def get_comparison(self) -> Optional[pd.DataFrame]:
        if self._comparison is None:
            path = self.results_dir / "model_comparison.csv"
            if path.exists():
                self._comparison = pd.read_csv(path)
        return self._comparison

    def get_peak_thresholds(self) -> dict:
        if self._peak_thresholds is None:
            path = self.results_dir / "peak_thresholds.json"
            if path.exists():
                self._peak_thresholds = json.loads(path.read_text(encoding="utf-8"))
            else:
                self._peak_thresholds = {}
        return self._peak_thresholds

    def get_high_demand(self) -> Optional[pd.DataFrame]:
        if self._high_demand is None:
            path = self.results_dir / "high_demand_flags.parquet"
            if path.exists():
                self._high_demand = pd.read_parquet(path)
        return self._high_demand

    def get_holidays(self) -> Optional[pd.DataFrame]:
        if self._holidays is None:
            path = Path("data/holidays/holidays_df.parquet")
            if path.exists():
                self._holidays = pd.read_parquet(path)
        return self._holidays


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    results_dir: str = "results",
    processed_dir: str = "data/processed",
) -> "FastAPI":
    if not FASTAPI_AVAILABLE:
        raise ImportError("Install fastapi and uvicorn:  pip install fastapi uvicorn")

    app = FastAPI(
        title="Canada STGNN — Day-Ahead Electricity Demand Forecast API",
        description="Serves actual model predictions for the STGNN forecasting system.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    store = DataStore(results_dir=results_dir, processed_dir=processed_dir)

    # Load region metadata from config
    try:
        with open("config/config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        REGIONS = cfg["download"]["regions"]
    except Exception:
        REGIONS = ["AB", "BC", "NB", "NL", "NS", "ON", "PE", "QC", "YK"]

    REGION_NAMES = {
        "AB": "Alberta",
        "BC": "British Columbia",
        "NB": "New Brunswick",
        "NL": "Newfoundland and Labrador",
        "NS": "Nova Scotia",
        "ON": "Ontario",
        "PE": "Prince Edward Island",
        "QC": "Quebec",
        "SK": "Saskatchewan",
        "YK": "Yukon",
    }

    # -----------------------------------------------------------------------
    # Endpoints
    # -----------------------------------------------------------------------

    @app.get("/")
    def health_check():
        return {"status": "ok", "service": "canada-stgnn-load-forecasting"}

    @app.get("/regions")
    def list_regions():
        return [
            {"code": r, "name": REGION_NAMES.get(r, r)}
            for r in REGIONS
        ]

    @app.get("/models")
    def list_models():
        preds = store.get_predictions()
        comparison = store.get_comparison()
        model_list = list(preds.keys())
        if comparison is not None and "model" in comparison.columns:
            model_list = comparison["model"].tolist()
        return {"models": model_list}

    @app.get("/forecast/{region}/{forecast_date}")
    def get_forecast(
        region: str,
        forecast_date: str,
        model: str = Query(default=None, description="Model name; defaults to best available"),
    ):
        """Return the 24-hour forecast for a region on a given date.

        forecast_date format: YYYY-MM-DD
        """
        region = region.upper()
        if region not in REGIONS:
            raise HTTPException(status_code=404, detail=f"Region '{region}' not found.")

        try:
            target_date = date.fromisoformat(forecast_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

        preds = store.get_predictions()
        if not preds:
            raise HTTPException(
                status_code=503,
                detail="No model predictions available yet.  Run training first.",
            )

        # Select model
        if model is None:
            comparison = store.get_comparison()
            if comparison is not None and "model" in comparison.columns:
                model = comparison.iloc[0]["model"]
            else:
                model = list(preds.keys())[0]

        if model not in preds:
            raise HTTPException(status_code=404, detail=f"Model '{model}' not found.")

        df = preds[model]
        df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
        mask = (df["region"] == region) & (df["forecast_date"] == target_date)
        subset = df[mask].sort_values("target_hour")

        if subset.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No forecast for region={region} date={forecast_date} model={model}.",
            )

        thresholds = store.get_peak_thresholds()
        threshold = thresholds.get(region, None)

        hours = []
        for _, row in subset.iterrows():
            pred = float(row["predicted"]) if not pd.isna(row.get("predicted")) else None
            actual = float(row["actual"]) if not pd.isna(row.get("actual", float("nan"))) else None
            hours.append(
                {
                    "hour": int(row["target_hour"]),
                    "predicted_mw": pred,
                    "actual_mw": actual,
                    "threshold_mw": round(threshold, 1) if threshold else None,
                    "is_high_demand": pred >= threshold if (pred is not None and threshold is not None) else None,
                }
            )

        return {
            "region": region,
            "region_name": REGION_NAMES.get(region, region),
            "forecast_date": forecast_date,
            "model": model,
            "hours": hours,
        }

    @app.get("/comparison")
    def model_comparison():
        comparison = store.get_comparison()
        if comparison is None:
            raise HTTPException(
                status_code=503,
                detail="Comparison table not available.  Run evaluation first.",
            )
        return comparison.replace({float("nan"): None}).to_dict(orient="records")

    @app.get("/metrics/{model_name}")
    def get_metrics(model_name: str):
        path = Path(results_dir) / f"metrics_{model_name}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Metrics for '{model_name}' not found.")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/high-demand/summary")
    def high_demand_summary():
        hd = store.get_high_demand()
        if hd is None:
            raise HTTPException(
                status_code=503,
                detail="High-demand flags not available.  Run detection step first.",
            )
        from src.high_demand.detector import high_demand_summary as _summary
        summary = _summary(hd)
        return summary.replace({float("nan"): None}).to_dict(orient="records")

    @app.get("/high-demand/{region}")
    def high_demand_region(
        region: str,
        forecast_date: Optional[str] = Query(default=None),
    ):
        region = region.upper()
        hd = store.get_high_demand()
        if hd is None:
            raise HTTPException(
                status_code=503,
                detail="High-demand flags not available.  Run detection step first.",
            )
        df = hd[hd["region"] == region]
        if forecast_date:
            try:
                target_date = date.fromisoformat(forecast_date)
                df["forecast_date"] = pd.to_datetime(df["forecast_date"]).dt.date
                df = df[df["forecast_date"] == target_date]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format.")

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No high-demand data for region={region}.",
            )

        return df.replace({float("nan"): None}).to_dict(orient="records")

    @app.get("/holiday/{region}/{year}")
    def get_holidays(region: str, year: int):
        holidays = store.get_holidays()
        if holidays is None:
            raise HTTPException(
                status_code=503,
                detail="Holiday data not available.  Run the pipeline first.",
            )
        region = region.upper()
        df = holidays[
            (holidays["region"] == region)
            & (pd.to_datetime(holidays["date"]).dt.year == year)
        ]
        return df.replace({float("nan"): None}).to_dict(orient="records")

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
