"""Seasonal naive baseline forecasters.

Two variants:
  - NaiveDay:  For each target hour h, predict the value at hour h yesterday.
  - NaiveWeek: For each target hour h, predict the value at hour h last week.

Both produce 24-hour day-ahead forecasts for all regions.
No fitting is required; they read directly from historical data.

Usage
-----
    from src.models.naive import NaiveDay, NaiveWeek

    # train is a multi-region hourly DataFrame with columns
    # ['time_utc', 'region', 'demand_mw', ...]
    model = NaiveDay()
    forecasts = model.predict(test_df, regions=['ON', 'QC'])
    # forecasts: DataFrame with columns [time_utc, region, forecast_hour, predicted, actual]
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)

# Hours in one day and one week
HOURS_PER_DAY = 24
HOURS_PER_WEEK = 168


class BaseNaive(ABC):
    """Abstract base class for seasonal naive forecasters."""

    name: str = "naive"

    def predict(
        self,
        df: pd.DataFrame,
        regions: list[str] | None = None,
        time_col: str = "time_utc",
        demand_col: str = "demand_mw",
        region_col: str = "region",
    ) -> pd.DataFrame:
        """Generate 24-hour day-ahead forecasts for all available forecast days.

        A forecast is issued at 23:00 on day D, targeting hours 0–23 of day D+1.
        For a row at hour H on day D+1, the lag is:
            NaiveDay:  H + 1 hours (23:00 yesterday to H today)
            NaiveWeek: H + 1 + 167 = H + 168 hours

        Parameters
        ----------
        df : Multi-region hourly DataFrame (must cover the full lag period).
        regions : Regions to forecast.  None → all in df.
        time_col : UTC datetime column.
        demand_col : Demand column.
        region_col : Region identifier column.

        Returns
        -------
        DataFrame with columns:
            time_utc, region, forecast_hour, predicted, actual
        """
        df = df.copy().sort_values([region_col, time_col])

        if regions is None:
            regions = df[region_col].unique().tolist()

        results = []
        for region in regions:
            region_df = df[df[region_col] == region].copy()
            region_df = region_df.set_index(time_col)
            demand = region_df[demand_col]

            # Identify forecast issue times: 23:00 each day that has sufficient history
            # We need lag_hours of look-back from the first target hour
            lag = self._lag_hours()
            shifted = demand.shift(lag)

            # Keep only valid (non-NaN) predictions
            shifted = shifted.dropna()

            # Build output: each row in shifted is a target timestamp with a prediction
            result_df = pd.DataFrame(
                {
                    "time_utc": shifted.index,
                    "region": region,
                    "target_hour": shifted.index.hour,
                    "forecast_hour": shifted.index.hour,
                    "predicted": shifted.values,
                    "actual": demand.loc[shifted.index].values,
                }
            )
            results.append(result_df)

        if not results:
            return pd.DataFrame(
                columns=["time_utc", "region", "forecast_hour", "predicted", "actual"]
            )

        return pd.concat(results, ignore_index=True)

    @abstractmethod
    def _lag_hours(self) -> int:
        """Return the lag in hours for this naive strategy."""
        ...


class NaiveDay(BaseNaive):
    """Predict each hour using the same hour from the previous day (24h lag)."""

    name = "naive_day"

    def _lag_hours(self) -> int:
        return HOURS_PER_DAY


class NaiveWeek(BaseNaive):
    """Predict each hour using the same hour from the previous week (168h lag)."""

    name = "naive_week"

    def _lag_hours(self) -> int:
        return HOURS_PER_WEEK


# ---------------------------------------------------------------------------
# Convenience: predict and return in the standard forecast output format
# ---------------------------------------------------------------------------

def run_naive_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    regions: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run both naive baselines on the test set.

    Returns a dict: {model_name: forecast_df}
    """
    # Combine train and test for look-back; test contains the target
    all_df = pd.concat([train_df, test_df], ignore_index=True).sort_values("time_utc")

    results = {}
    for model_cls in [NaiveDay, NaiveWeek]:
        model = model_cls()
        # Predict only on the test period by using the combined df
        # and filtering predictions to the test period
        preds = model.predict(all_df, regions=regions)
        test_start = test_df["time_utc"].min()
        test_end = test_df["time_utc"].max()
        preds = preds[
            (preds["time_utc"] >= test_start) & (preds["time_utc"] <= test_end)
        ].copy()
        results[model.name] = preds
        logger.info(
            "Naive %s: %d predictions over %d regions",
            model.name,
            len(preds),
            preds["region"].nunique(),
        )

    return results
