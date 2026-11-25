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


def test_date_forecast_and_ablation_curves_render():
    app = page("forecast_page")
    app.radio[0].set_value("Feature ablations")
    app.button[0].click().run(timeout=120)
    assert not app.exception
    assert not app.error
    assert app.dataframe[0].value.shape == (24, 6)
    assert "Actual demand" in app.dataframe[0].value.columns
    assert len(app.metric) == 4
    app.date_input[0].set_value(date(2025, 3, 9))
    app.button[0].click().run(timeout=120)
    assert not app.exception
    assert any("daylight saving" in item.value for item in app.warning)


def test_main_navigation_loads_updated_forecast_page():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "dashboard/app.py")).run(timeout=120)
    assert not app.exception
    assert app.title[0].value == "HFED day-ahead forecasts"
