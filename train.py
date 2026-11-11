"""End-to-end training script.

Runs all phases in order:
  Phase 1: Data pipeline (load, resample, features, holidays)
  Phase 2: Seasonal naive baselines
  Phase 3: XGBoost
  Phase 4: LSTM
  Phase 5: GRU
  Phase 6: STGNN (all graph configurations)
  Phase 7: Evaluation and comparison
  Phase 8: High-demand detection
  Phase 9: Optional PBC probabilistic extension

Usage:
    python train.py                      # full pipeline
    python train.py --skip-dl-models     # skip LSTM/GRU/STGNN (fast run)
    python train.py --only-evaluate      # re-evaluate saved predictions
    python train.py --config custom.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_predictions(preds_df: pd.DataFrame, name: str, results_dir: Path) -> None:
    out = results_dir / f"predictions_{name}.parquet"
    preds_df.to_parquet(out, index=False)
    logger.info("Saved predictions for %s -> %s", name, out)


def inverse_transform_preds(
    preds_df: pd.DataFrame,
    scaler,
    demand_col_pred: str = "predicted",
    demand_col_actual: str = "actual",
    region_col: str = "region",
) -> pd.DataFrame:
    """Convert z-scored predictions back to MW using the fitted scaler.
    Used after deep learning models (LSTM, GRU, STGNN) trained on scaled data.
    """
    df = preds_df.copy()
    # Ensure target columns are float so inverse-transformed values can be assigned
    for col in [demand_col_pred, demand_col_actual]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    for region in df[region_col].unique():
        mask = df[region_col] == region
        if demand_col_pred in df.columns:
            df.loc[mask, demand_col_pred] = scaler.inverse_transform_demand(
                df.loc[mask, demand_col_pred].values.astype(float), region
            )
        if demand_col_actual in df.columns:
            df.loc[mask, demand_col_actual] = scaler.inverse_transform_demand(
                df.loc[mask, demand_col_actual].values.astype(float), region
            )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Canada STGNN training pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--skip-dl-models", action="store_true",
        help="Skip LSTM, GRU, and STGNN (useful for fast testing)",
    )
    parser.add_argument(
        "--only-evaluate", action="store_true",
        help="Skip training; re-run evaluation on saved predictions",
    )
    parser.add_argument(
        "--phases", nargs="+", type=int, default=None,
        help="Run only specific phases (1-9)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = Path(cfg["paths"]["results_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    run_phases = set(args.phases) if args.phases else set(range(1, 10))

    # -----------------------------------------------------------------------
    # Phase 1: Data pipeline
    # -----------------------------------------------------------------------
    if 1 in run_phases and not args.only_evaluate:
        logger.info("=" * 60)
        logger.info("Phase 1: Data pipeline")
        logger.info("=" * 60)

        # Check data is downloaded
        raw_dir = Path(cfg["paths"]["raw_data_dir"])
        if not raw_dir.exists() or not any(raw_dir.iterdir()):
            logger.error(
                "Raw HFED data not found at %s.\n"
                "Run the downloader first:\n"
                "  python hfed_dataset_downloader.py",
                raw_dir,
            )
            sys.exit(1)

        from src.data.pipeline import run_pipeline
        train_df, val_df, test_df, scaler = run_pipeline(cfg)

    else:
        logger.info("Loading processed data from parquet...")
        train_df = pd.read_parquet(processed_dir / "train.parquet")
        val_df = pd.read_parquet(processed_dir / "val.parquet")
        test_df = pd.read_parquet(processed_dir / "test.parquet")

        import json as _json
        from src.data.pipeline import RegionalStandardScaler
        scaler_stats = _json.loads(
            (processed_dir / "scaler_stats.json").read_text(encoding="utf-8")
        )
        scaler = RegionalStandardScaler.from_dict(scaler_stats)

    regions = sorted(train_df["region"].unique().tolist())
    logger.info("Regions: %s", regions)
    logger.info(
        "Data: train=%d  val=%d  test=%d rows",
        len(train_df), len(val_df), len(test_df),
    )

    # Build scaled copies for deep learning models (LSTM, GRU, STGNN).
    # XGBoost and naive baselines use raw (unscaled) demand — they handle
    # scale internally or are insensitive to it.
    # Scaler is fitted on train only; applied consistently to val and test.
    skip_scale = {
        "time_utc", "time_local", "region", "DATETIME_LOCAL_OFFSET",
        "demand_flag_missing", "demand_flag_negative", "demand_flag_extreme",
        "is_holiday", "is_weekend", "is_day_before_holiday", "is_day_after_holiday",
        "holiday_name", "year",
    }
    scale_cols = [
        c for c in train_df.columns
        if c not in skip_scale and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    train_scaled = scaler.transform(train_df, feature_cols=scale_cols)
    val_scaled   = scaler.transform(val_df,   feature_cols=scale_cols)
    test_scaled  = scaler.transform(test_df,  feature_cols=scale_cols)

    all_predictions: dict[str, pd.DataFrame] = {}

    # -----------------------------------------------------------------------
    # Phase 2: Seasonal naive baselines
    # -----------------------------------------------------------------------
    if 2 in run_phases and not args.only_evaluate:
        logger.info("=" * 60)
        logger.info("Phase 2: Seasonal naive baselines")
        logger.info("=" * 60)
        from src.models.naive import run_naive_baselines
        naive_preds = run_naive_baselines(train_df, test_df, regions=regions)
        for name, preds in naive_preds.items():
            all_predictions[name] = preds
            save_predictions(preds, name, results_dir)

    # -----------------------------------------------------------------------
    # Phase 3: XGBoost
    # -----------------------------------------------------------------------
    if 3 in run_phases and not args.only_evaluate:
        logger.info("=" * 60)
        logger.info("Phase 3: XGBoost")
        logger.info("=" * 60)
        try:
            from src.models.xgboost_model import XGBoostForecaster
            xgb_model = XGBoostForecaster(cfg)
            xgb_model.fit(train_df, val_df, model_dir=model_dir)
            xgb_preds = xgb_model.predict(test_df)
            all_predictions["xgboost"] = xgb_preds
            save_predictions(xgb_preds, "xgboost", results_dir)
        except ImportError as exc:
            logger.warning("XGBoost skipped: %s", exc)

    # -----------------------------------------------------------------------
    # Phase 4: LSTM
    # -----------------------------------------------------------------------
    if 4 in run_phases and not args.only_evaluate and not args.skip_dl_models:
        logger.info("=" * 60)
        logger.info("Phase 4: LSTM")
        logger.info("=" * 60)
        try:
            from src.models.lstm_model import LSTMForecaster
            lstm = LSTMForecaster(cfg)
            lstm.fit(train_scaled, val_scaled, model_dir=model_dir)
            # Filter test to training regions only (YK may appear in test but not train)
            train_regions = sorted(train_scaled["region"].unique().tolist())
            test_for_lstm = test_scaled[test_scaled["region"].isin(train_regions)]
            lstm_preds = lstm.predict(test_for_lstm)
            lstm_preds = inverse_transform_preds(lstm_preds, scaler)
            all_predictions["lstm"] = lstm_preds
            save_predictions(lstm_preds, "lstm", results_dir)
        except ImportError as exc:
            logger.warning("LSTM skipped: %s", exc)

    # -----------------------------------------------------------------------
    # Phase 5: GRU
    # -----------------------------------------------------------------------
    if 5 in run_phases and not args.only_evaluate and not args.skip_dl_models:
        logger.info("=" * 60)
        logger.info("Phase 5: GRU")
        logger.info("=" * 60)
        try:
            from src.models.gru_model import GRUForecaster
            gru = GRUForecaster(cfg)
            gru.fit(train_scaled, val_scaled, model_dir=model_dir)
            train_regions = sorted(train_scaled["region"].unique().tolist())
            test_for_gru = test_scaled[test_scaled["region"].isin(train_regions)]
            gru_preds = gru.predict(test_for_gru)
            gru_preds = inverse_transform_preds(gru_preds, scaler)
            all_predictions["gru"] = gru_preds
            save_predictions(gru_preds, "gru", results_dir)
        except ImportError as exc:
            logger.warning("GRU skipped: %s", exc)

    # -----------------------------------------------------------------------
    # Phase 6: STGNN
    # -----------------------------------------------------------------------
    if 6 in run_phases and not args.only_evaluate and not args.skip_dl_models:
        logger.info("=" * 60)
        logger.info("Phase 6: STGNN (graph ablation)")
        logger.info("=" * 60)
        try:
            from src.models.stgnn.trainer import STGCNTrainer
            stgnn_trainer = STGCNTrainer(cfg)
            stgnn_preds = stgnn_trainer.run(
                train_scaled, val_scaled,
                test_scaled[test_scaled["region"].isin(regions)],
                regions=regions,
                model_dir=model_dir / "stgnn",
            )
            for graph_type, preds in stgnn_preds.items():
                name = f"stgnn_{graph_type}"
                preds = inverse_transform_preds(preds, scaler)
                all_predictions[name] = preds
                save_predictions(preds, name, results_dir)
        except ImportError as exc:
            logger.warning("STGNN skipped: %s", exc)

    # Load any saved predictions we didn't just compute
    if args.only_evaluate or not all_predictions:
        logger.info("Loading saved predictions from %s ...", results_dir)
        for path in sorted(results_dir.glob("predictions_*.parquet")):
            name = path.stem.replace("predictions_", "")
            if name not in all_predictions:
                all_predictions[name] = pd.read_parquet(path)
                logger.info("  Loaded %s (%d rows)", name, len(all_predictions[name]))

    # -----------------------------------------------------------------------
    # Phase 7: Evaluation
    # -----------------------------------------------------------------------
    if 7 in run_phases:
        logger.info("=" * 60)
        logger.info("Phase 7: Evaluation and comparison")
        logger.info("=" * 60)
        if all_predictions:
            from src.evaluation.evaluator import run_full_evaluation
            all_results, comparison = run_full_evaluation(
                all_predictions,
                train_df=train_df,
                regions=regions,
                config=cfg,
                results_dir=results_dir,
            )
            logger.info("\nFinal comparison:\n%s", comparison.to_string())
        else:
            logger.warning("No predictions available for evaluation.")

    # -----------------------------------------------------------------------
    # Phase 8: High-demand detection
    # -----------------------------------------------------------------------
    if 8 in run_phases and all_predictions:
        logger.info("=" * 60)
        logger.info("Phase 8: High-demand detection")
        logger.info("=" * 60)
        from src.evaluation.evaluator import compute_peak_thresholds
        from src.high_demand.detector import flag_high_demand

        thresholds = compute_peak_thresholds(
            train_df, regions,
            percentile=cfg["high_demand"]["threshold_percentile"],
        )

        # Use best available model (identity STGNN or XGBoost or naive)
        best_model_name = next(
            (n for n in ["stgnn_identity", "stgnn_geographic", "xgboost", "naive_week"]
             if n in all_predictions),
            list(all_predictions.keys())[0] if all_predictions else None,
        )

        if best_model_name:
            hd_df = flag_high_demand(all_predictions[best_model_name], thresholds)
            hd_path = results_dir / "high_demand_flags.parquet"
            hd_df.to_parquet(hd_path, index=False)
            logger.info("High-demand flags saved to %s.", hd_path)

    # -----------------------------------------------------------------------
    # Phase 9: Probabilistic bias correction (optional)
    # -----------------------------------------------------------------------
    if 9 in run_phases and all_predictions:
        logger.info("=" * 60)
        logger.info("Phase 9: Probabilistic bias correction (optional extension)")
        logger.info("=" * 60)
        # Requires STGNN identity predictions on both val and test
        stgnn_identity = all_predictions.get("stgnn_identity")
        if stgnn_identity is None:
            logger.info("STGNN identity predictions not found; skipping PBC phase.")
        else:
            try:
                from src.models.pbc.debias_plus import DebiasPlus
                from src.models.pbc.persistence_plus import run_pbc_pipeline

                # Calibration = val predictions (needs to be generated)
                val_preds_path = results_dir / "predictions_stgnn_identity_val.parquet"
                if not val_preds_path.exists():
                    logger.info(
                        "Validation predictions for PBC not found at %s; skipping PBC.",
                        val_preds_path,
                    )
                else:
                    calib_df = pd.read_parquet(val_preds_path)
                    debias = DebiasPlus()
                    debias.fit(calib_df)
                    test_debias = debias.correct(stgnn_identity)
                    test_pbc = run_pbc_pipeline(calib_df, test_debias)
                    pbc_path = results_dir / "probabilistic_predictions.parquet"
                    test_pbc.to_parquet(pbc_path, index=False)
                    logger.info("PBC predictions saved to %s.", pbc_path)
            except Exception as exc:
                logger.warning("PBC phase failed: %s", exc)

    logger.info("=" * 60)
    logger.info("Training pipeline complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
