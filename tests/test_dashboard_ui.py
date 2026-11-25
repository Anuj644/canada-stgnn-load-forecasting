"""Exercise the real dashboard pages and date/ablation interaction."""
from datetime import date
import pytest
from streamlit.testing.v1 import AppTest
from pathlib import Path


def page(name):
    app = AppTest.from_string(f"from dashboard.benchmark_pages import {name}; {name}()")
    app.run(timeout=120)
    assert not app.exception
    return app


@pytest.mark.parametrize("name", ["results_page", "ablation_page", "methods_page"])
def test_evidence_pages_render_from_real_artifacts(name):
    app = page(name)
    assert len(app.dataframe) >= 2
    assert not app.error


def choose(app, province="ON", day=date(2025, 1, 15)):
    next(w for w in app.selectbox if w.label == "Province").set_value(province)
    app.date_input[0].set_value(day)
    app.multiselect[0].set_value(["cnn", "lstm", "xgboost", "daily"])
    next(w for w in app.selectbox if w.label == "Forecast member").set_value("Seed 42")


def test_forecast_page_starts_empty_and_requires_province_and_date():
    app = page("forecast_page")
    assert next(w for w in app.selectbox if w.label == "Province").value is None
    assert app.date_input[0].value is None
    assert app.multiselect[0].value == []
    assert next(w for w in app.selectbox if w.label == "Forecast member").value is None
    assert not app.metric
    app.button[0].click().run(timeout=120)
    assert any("Select a province, a forecast date" in item.value for item in app.warning)
    assert not app.metric


def test_date_forecast_and_ablation_curves_render():
    app = page("forecast_page")
    assert not app.get("file_uploader")
    source = next(widget for widget in app.selectbox if widget.label == "Observation source")
    assert "Upload HFED CSV" not in source.options
    app.toggle[0].set_value(True).run(timeout=120)
    assert len(app.get("file_uploader")) == 1
    assert not any(widget.label == "Observation source" for widget in app.selectbox)
    choose(app)
    app.button[0].click().run(timeout=120)
    assert any("Upload native HFED" in item.value for item in app.warning)
    app.toggle[0].set_value(False).run(timeout=120)
    assert not app.get("file_uploader")
    choose(app)
    app.radio[0].set_value("Feature ablations")
    app.button[0].click().run(timeout=120)
    assert not app.exception
    assert not app.error
    assert app.dataframe[0].value.shape == (24, 6)
    assert "Actual demand" in app.dataframe[0].value.columns
    assert len(app.metric) == 4
    choose(app, "ON", date(2025, 3, 9))
    app.button[0].click().run(timeout=120)
    assert not app.exception
    assert any("daylight saving" in item.value for item in app.warning)


def test_main_navigation_loads_updated_forecast_page():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard/app.py")).run(timeout=120)
    assert not app.exception
    assert app.title[0].value == "HFED day-ahead forecasts"
