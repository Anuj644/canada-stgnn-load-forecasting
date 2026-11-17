import pandas as pd
import requests

from src.data import hfed_api


def test_csv_download_retries_service_unavailable(monkeypatch):
    statuses = iter([503, 200])

    def get(*args, **kwargs):
        response = requests.Response()
        response.status_code = next(statuses)
        response._content = b"TIME_PERIOD,OBS_VALUE\n"
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "get", get)
    monkeypatch.setattr(hfed_api, "sleep", lambda _: None)
    result = hfed_api._get_csv("https://example.com", "2026-11-08", "2026-11-15", 30)

    assert result == b"TIME_PERIOD,OBS_VALUE\n"


def test_demand_history_preserves_available_data_and_reports_failures(monkeypatch):
    ontario = pd.DataFrame({
        "region": ["ON"],
        "time_utc": [pd.Timestamp("2026-11-09T22:00:00")],
        "demand_mw": [16873.0],
    })

    def fetch(code, *_):
        if code == "ON":
            return ontario
        if code == "QC":
            raise requests.Timeout()
        if code == "AB":
            raise ValueError("Unexpected CSV schema.")
        return None

    monkeypatch.setattr(hfed_api, "fetch_demand", fetch)
    demand, unavailable = hfed_api.fetch_demand_history(
        iter(["ON", "QC", "AB", "BC"]), "2026-11-08", "2026-11-15", {},
    )

    pd.testing.assert_frame_equal(demand, ontario)
    assert unavailable == {
        "QC": "The HFED request timed out.",
        "AB": "The HFED response could not be read. Unexpected CSV schema.",
        "BC": "HFED returned no demand readings for the requested dates.",
    }
