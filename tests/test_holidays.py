"""Tests for the holiday client and holiday-region mapping."""
import pandas as pd

from src.data.holiday_client import (
    ALL_HFED_REGIONS,
    SUBDIVISION_TO_REGION,
    build_holiday_dataframe,
)

# ---------------------------------------------------------------------------
# Fixtures: mock API response data (no live API call needed)
# ---------------------------------------------------------------------------

MOCK_HOLIDAYS_2024 = [
    # National (global) holiday
    {
        "date": "2024-01-01",
        "localName": "New Year's Day",
        "name": "New Year's Day",
        "countryCode": "CA",
        "fixed": True,
        "global": True,
        "counties": None,
    },
    # Ontario-specific
    {
        "date": "2024-02-19",
        "localName": "Family Day",
        "name": "Family Day",
        "countryCode": "CA",
        "fixed": False,
        "global": False,
        "counties": ["CA-ON", "CA-AB", "CA-BC", "CA-SK"],
    },
    # Quebec-specific
    {
        "date": "2024-06-24",
        "localName": "Saint-Jean-Baptiste Day",
        "name": "Saint-Jean-Baptiste Day",
        "countryCode": "CA",
        "fixed": True,
        "global": False,
        "counties": ["CA-QC"],
    },
]


# ---------------------------------------------------------------------------
# Subdivision mapping
# ---------------------------------------------------------------------------

class TestSubdivisionMapping:
    def test_all_hfed_regions_mapped(self):
        """Every HFED region must have a Nager.Date subdivision code."""
        mapped_regions = set(SUBDIVISION_TO_REGION.values())
        for region in ALL_HFED_REGIONS:
            assert region in mapped_regions, f"HFED region {region} has no subdivision mapping"

    def test_on_maps_to_ca_on(self):
        inverse = {v: k for k, v in SUBDIVISION_TO_REGION.items()}
        assert inverse.get("ON") == "CA-ON"

    def test_yk_maps_to_ca_yt(self):
        inverse = {v: k for k, v in SUBDIVISION_TO_REGION.items()}
        assert inverse.get("YK") == "CA-YT"


# ---------------------------------------------------------------------------
# build_holiday_dataframe
# ---------------------------------------------------------------------------

class TestBuildHolidayDataframe:
    def setup_method(self):
        self.mock_data = {2024: MOCK_HOLIDAYS_2024}

    def test_national_holiday_applies_to_all_regions(self):
        df = build_holiday_dataframe(self.mock_data)
        new_years = df[df["holiday_name"] == "New Year's Day"]
        hfed_regions = sorted(ALL_HFED_REGIONS)
        for region in hfed_regions:
            assert region in new_years["region"].values, \
                f"New Year's Day missing for region {region}"

    def test_provincial_holiday_only_for_correct_regions(self):
        df = build_holiday_dataframe(self.mock_data)
        # Family Day should apply to ON, AB, BC but NOT QC
        family_day = df[df["holiday_name"] == "Family Day"]
        assert "ON" in family_day["region"].values
        assert "AB" in family_day["region"].values
        assert "QC" not in family_day["region"].values

    def test_qc_holiday_only_for_qc(self):
        df = build_holiday_dataframe(self.mock_data)
        sjb = df[df["holiday_name"] == "Saint-Jean-Baptiste Day"]
        assert list(sjb["region"].values) == ["QC"]

    def test_is_holiday_column_is_true(self):
        df = build_holiday_dataframe(self.mock_data)
        assert df["is_holiday"].all()

    def test_no_duplicate_region_dates(self):
        df = build_holiday_dataframe(self.mock_data)
        dupes = df.duplicated(subset=["date", "region"])
        assert dupes.sum() == 0, f"Duplicate (date, region) rows found:\n{df[dupes]}"

    def test_date_column_type(self):
        df = build_holiday_dataframe(self.mock_data)
        # Should be convertible to datetime
        dates = pd.to_datetime(df["date"])
        assert not dates.isna().any()

    def test_empty_holiday_data_returns_empty_df(self):
        df = build_holiday_dataframe({2024: []})
        assert len(df) == 0
        assert "date" in df.columns
        assert "region" in df.columns

    def test_national_flag_set_correctly(self):
        df = build_holiday_dataframe(self.mock_data)
        new_years = df[df["holiday_name"] == "New Year's Day"].iloc[0]
        assert new_years["is_national"] is True or new_years["is_national"] == True

        family_day = df[
            (df["holiday_name"] == "Family Day") & (df["region"] == "ON")
        ].iloc[0]
        assert family_day["is_national"] is False or family_day["is_national"] == False


# ---------------------------------------------------------------------------
# Holiday feature integration
# ---------------------------------------------------------------------------

class TestHolidayFeatureIntegration:
    def test_add_holiday_features_smoke(self):
        from src.data.feature_engineering import add_holiday_features
        times = pd.date_range("2024-01-01", periods=24, freq="h")
        df = pd.DataFrame({
            "time_utc": times,
            "time_local": times,
            "region": "ON",
            "demand_mw": 15000.0,
        })
        holiday_data = {2024: MOCK_HOLIDAYS_2024}
        hol_df = build_holiday_dataframe(holiday_data, hfed_regions=["ON"])
        result = add_holiday_features(df, hol_df)
        assert "is_holiday" in result.columns
        assert "holiday_name" in result.columns
        assert "is_day_before_holiday" in result.columns
        assert "is_day_after_holiday" in result.columns

    def test_new_years_day_flagged(self):
        from src.data.feature_engineering import add_holiday_features
        # 2024-01-01 hours should be holidays
        times = pd.date_range("2024-01-01T00:00", periods=24, freq="h")
        df = pd.DataFrame({
            "time_utc": times,
            "time_local": times,
            "region": "ON",
            "demand_mw": 15000.0,
        })
        holiday_data = {2024: MOCK_HOLIDAYS_2024}
        hol_df = build_holiday_dataframe(holiday_data, hfed_regions=["ON"])
        result = add_holiday_features(df, hol_df)
        assert result["is_holiday"].all(), "All hours on 2024-01-01 should be holidays"

    def test_day_before_holiday_flag(self):
        from src.data.feature_engineering import add_holiday_features
        # 2023-12-31 should be day-before New Year's
        times = pd.date_range("2023-12-31T00:00", periods=24, freq="h")
        df = pd.DataFrame({
            "time_utc": times,
            "time_local": times,
            "region": "ON",
            "demand_mw": 15000.0,
        })
        holiday_data = {2023: [], 2024: MOCK_HOLIDAYS_2024}
        hol_df = build_holiday_dataframe(holiday_data, hfed_regions=["ON"])
        result = add_holiday_features(df, hol_df)
        assert result["is_day_before_holiday"].all(), \
            "2023-12-31 should be flagged as day-before New Year's"
