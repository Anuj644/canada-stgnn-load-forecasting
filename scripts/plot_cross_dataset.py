"""Render the completed matched comparisons without changing method selection."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "grid_aware_lstm": "Grid-Aware LSTM",
    "grid_aware_gru": "Grid-Aware GRU",
    "seasonal_identity": "Seasonal GRU (identity)",
    "seasonal_grid_aware": "Seasonal GRU (graph)",
    "residual_identity": "Residual CNN (identity)",
    "xgboost": "XGBoost",
    "naive_day_available": "Available daily naive",
    "naive_week": "Weekly naive",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brazil", type=Path, default=ROOT / "results/grid_aware_confirm_v3")
    parser.add_argument("--hfed", type=Path, default=ROOT / "results/hfed_extension_eccc_v4")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/cross_dataset_validation.png")
    args = parser.parse_args()
    selected = json.loads((args.brazil / "method_selection.json").read_text(encoding="utf-8"))["selected_method"]
    brazil = pd.read_csv(args.brazil / "metrics_mean_std.csv")
    brazil = brazil.loc[brazil.horizon.eq(24)]
    hfed = pd.read_csv(args.hfed / "metrics_mean_std.csv")
    panels = [
        (brazil, "test2019", "Brazil: Jul-Dec 2019, +24h endpoint"),
        (brazil, "test2020", "Brazil: additional 2020 holdout, +24h endpoint"),
        (hfed, "test2024", "HFED: Jul-Dec 2024, previously inspected period"),
        (hfed, "test2025", "HFED: fresh 2025, complete next-day forecasts"),
    ]
    # Accept only the actual declared period spelling; fail instead of empty panels.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    for ax, (table, period, title) in zip(axes.ravel(), panels):
        subset = table.loc[table.period.eq(period)].sort_values("mae_mean")
        if subset.empty or subset.model.duplicated().any():
            raise ValueError(f"Missing or duplicate metrics for {period}")
        colors = ["#d97706" if model == selected else "#2563eb" if model.startswith("grid_aware")
                  else "#94a3b8" for model in subset.model]
        positions = np.arange(len(subset))
        ax.barh(positions, subset.mae_mean, xerr=subset.mae_std.fillna(0),
                color=colors, capsize=3, error_kw={"elinewidth": 1})
        ax.set_yticks(positions, [LABELS.get(model, model) for model in subset.model], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("MAE (MW); lower is better")
        ax.set_title(title, loc="left", fontsize=11, pad=12)
        ax.xaxis.grid(True, alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, float((subset.mae_mean + subset.mae_std.fillna(0)).max()) * 1.18)
        for position, value, deviation in zip(positions, subset.mae_mean, subset.mae_std.fillna(0)):
            ax.annotate(f"{value:.1f}", (value + deviation, position), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=8)
    fig.suptitle("Matched load forecasting comparisons\n"
                 "Models retrained for each dataset; references are controlled reimplementations\n"
                 "Orange: method selected on Brazilian validation; blue: Grid-Aware references\n"
                 "Bars: three-seed means; whiskers: +/- one sample standard deviation",
                 fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(json.dumps({"output": str(args.output)}))


if __name__ == "__main__":
    main()
