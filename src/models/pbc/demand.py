"""Demand adaptation of the paper's parallel CDF corrections and projection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import isotonic_regression
from sklearn.linear_model import Ridge

PROBABILITIES = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])


def _recent_errors(forecasts):
    """Attach only forecast errors whose UTC observation interval has ended."""
    output = forecasts.copy()
    output["recent_error"] = 0.0
    output["recent_count"] = 0.0
    for _, region in output.groupby("region", sort=False):
        observed = region.dropna(subset=["actual", "predicted"]).sort_values(
            "target_at"
        )
        available = observed["target_at"] + pd.Timedelta(hours=1)
        errors = pd.Series(
            (observed["actual"] - observed["predicted"]).to_numpy(),
            index=pd.DatetimeIndex(available),
        )
        rolling = errors.rolling("48h")
        history = pd.DataFrame(
            {
                "available": available.to_numpy(),
                "recent_error": rolling.mean().to_numpy(),
                "recent_count": rolling.count().to_numpy(),
            }
        )
        issued = region[["issued_at"]].reset_index(names="row").sort_values("issued_at")
        aligned = pd.merge_asof(
            issued,
            history,
            left_on="issued_at",
            right_on="available",
            tolerance=pd.Timedelta(hours=48),
        ).set_index("row")
        cols = ["recent_error", "recent_count"]
        output.loc[aligned.index, cols] = aligned[cols].fillna(0)
    return output


def _cdf_features(frame, scale):
    angle = frame["target_hour"].to_numpy() * np.pi / 12
    weekday = frame["forecast_date"].dt.dayofweek.to_numpy()
    seasonal = np.column_stack(
        [
            np.sin(angle),
            np.cos(angle),
            np.sin(2 * angle),
            np.cos(2 * angle),
            weekday >= 5,
        ]
    )
    persistence = np.column_stack(
        [
            seasonal,
            np.clip(frame["recent_error"].to_numpy() / scale, -5, 5),
            frame["recent_count"].to_numpy() / 48,
        ]
    )
    return seasonal, persistence


def _project(values):
    return np.stack([isotonic_regression(row).x for row in values]).clip(0, 1)


class DemandPBC:
    """Fit regional CDF regressions on a held-out calibration period.

    Thresholds are standardized residual offsets around each incoming point
    forecast. Debias++ uses calendar features. Persistence++ adds recent
    observed errors available at issuance. Average the two projected CDFs.
    """

    def fit(self, calibration):
        calibration = _recent_errors(calibration.reset_index(drop=True))
        self.calibration = calibration.drop(columns=["recent_error", "recent_count"])
        self.regions = {}
        for region, frame in calibration.dropna(subset=["actual"]).groupby("region"):
            residual = (frame["actual"] - frame["predicted"]).to_numpy()
            scale = max(np.median(np.abs(residual)), 1.0)
            standardized = residual / scale
            edge = max(8, float(np.quantile(np.abs(standardized), 0.99)) * 1.25)
            grid = np.linspace(-edge, edge, 161)
            labels = (standardized[:, None] <= grid).astype(float)
            baseline = labels.mean(axis=0)
            seasonal, persistence = _cdf_features(frame, scale)
            debias = Ridge(alpha=30).fit(seasonal, labels - baseline)
            recent = Ridge(alpha=60).fit(persistence, labels - baseline)
            self.regions[region] = (scale, grid, baseline, debias, recent)
        return self

    def predict(self, forecasts, thresholds):
        if forecasts["issued_at"].min() < self.calibration["issued_at"].max():
            raise ValueError("Forecast issuance must follow the calibration period.")
        combined = pd.concat([self.calibration, forecasts], ignore_index=True)
        output = _recent_errors(combined).iloc[len(self.calibration) :].copy()
        for region, frame in output.groupby("region", sort=False):
            scale, grid, baseline, debias, recent = self.regions[region]
            seasonal, persistence = _cdf_features(frame, scale)
            debias_cdf = _project(baseline + debias.predict(seasonal))
            recent_cdf = _project(baseline + recent.predict(persistence))
            cdf = (debias_cdf + recent_cdf) / 2
            cdf[:, 0], cdf[:, -1] = 0, 1
            point = frame["predicted"].to_numpy()
            quantiles = point[:, None] + scale * np.stack(
                [np.interp(PROBABILITIES, row, grid) for row in cdf]
            )
            output.loc[frame.index, "base_predicted"] = point
            for column, values in zip(
                (f"q{q * 100:02.0f}" for q in PROBABILITIES), quantiles.T
            ):
                output.loc[frame.index, column] = np.maximum(values, 0)
            offsets = (thresholds[region] - point) / scale
            output.loc[frame.index, "prob_exceed_threshold"] = [
                1 - np.interp(offset, grid, row) for offset, row in zip(offsets, cdf)
            ]
            observed = (frame["actual"].to_numpy() - point) / scale
            indicators = observed[:, None] <= grid
            for name, distribution in [("pbc", cdf), ("empirical", baseline)]:
                # Integrate on the CDF support and include observations outside it.
                tails = np.maximum(grid[0] - observed, 0) + np.maximum(
                    observed - grid[-1], 0
                )
                scores = scale * (
                    trapezoid((distribution - indicators) ** 2, grid, axis=1) + tails
                )
                output.loc[frame.index, f"crps_{name}"] = scores
        output["predicted"] = output["q50"]
        output["lower_80"] = output["q10"]
        output["upper_80"] = output["q90"]
        return output.reset_index(drop=True)
