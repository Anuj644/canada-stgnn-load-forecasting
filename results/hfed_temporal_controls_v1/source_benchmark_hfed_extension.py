"""Retrain the Brazilian method on HFED local-day examples, preserving old results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.day_ahead_benchmark import prepared_data
from src.models.stgnn.grid_aware import GridAware, SeasonalGridAware, TemporalResidual, TemporalOnly
from src.models.checkpoint_io import save_checkpoint, save_json
from src.evaluation.experiment_plan import validate_frozen_plan
from src.models.hfed_tabular import NAMES as CONTROLS, fit_control, predict_control
from src.models.stgnn.cuda_training import CapturedSequenceStep, TensorPreservingPlateau

MODELS = ["grid_aware_lstm", "grid_aware_gru", "seasonal_identity", "seasonal_grid_aware", "residual_identity", *CONTROLS]
SUPPLEMENTARY = ["temporal_lstm", "temporal_gru"]


def is_reference(name):
    return name.startswith(("grid_aware", "temporal_"))


def inputs(data, device, feature_set):
    tensors = {key: torch.tensor(data[key], device=device, dtype=torch.long if key == "nodes" else torch.float32)
               for key in ["dynamic", "covariates", "anchor", "actual", "nodes"]}
    # Ablations remove information while keeping sample keys and architecture fixed.
    if feature_set == "demand":
        off = [data["features"].index(name) for name in data["features"]
               if name in ["temperature_c", "relative_humidity", "wind_speed"]
               or name in ["temperature_c_observed", "relative_humidity_observed", "wind_speed_observed"]]
        tensors["dynamic"][:, :, off] = 0
    if feature_set != "demand_weather_calendar":
        off = [i for i, name in enumerate(data["features"]) if name.startswith("calendar_")]
        tensors["dynamic"][:, :, off] = 0
        tensors["covariates"][:, :, 14:] = 0
    return tensors


def create_model(name, data, device):
    if is_reference(name):
        model = (TemporalOnly(data, backbone=name.removeprefix("temporal_")) if name.startswith("temporal_") else
                 GridAware(data, backbone=name.removeprefix("grid_aware_")))
        center, scale = data["target_center"], data["target_scale"]
    else:
        graph = data if name != "seasonal_identity" else dict(data, edge_index=np.empty((2, 0), dtype=np.int64),
                    edge_features=np.empty((0, data["edge_features"].shape[1])))
        model = TemporalResidual(data) if name == "residual_identity" else SeasonalGridAware(graph)
        center = torch.tensor(data["center"], device=device, dtype=torch.float32)
        scale = torch.tensor(data["scale"], device=device, dtype=torch.float32)
    return model.to(device), center, scale


def forward_batch(model, tensors, idx, center, scale, reference, capture=False):
    nodes = tensors["nodes"][idx]
    seq, actual = tensors["dynamic"][idx], tensors["actual"][idx]
    valid = actual.isfinite()
    c, s = (center, scale) if reference else (center[nodes, None], scale[nodes, None])
    target = (actual.nan_to_num() - c) / s
    pred = (model.forward_samples(seq, nodes) if reference else
            model.forward_samples(seq, nodes, tensors["covariates"][idx], tensors["anchor"][idx]))
    if capture:
        elementwise = (nn.functional.mse_loss(pred, target, reduction="none") if reference else
                       nn.functional.smooth_l1_loss(pred, target, beta=0.25, reduction="none"))
        loss = elementwise.masked_fill(~valid, 0).sum() / valid.sum()
    else:
        loss = (nn.functional.mse_loss(pred[valid], target[valid]) if reference else
                nn.functional.smooth_l1_loss(pred[valid], target[valid], beta=0.25))
    return pred * s + c, loss, valid


def fit_one(name, seed, data, tensors, args):
    if name in CONTROLS:
        if args.device == "cuda":
            torch.cuda.empty_cache()
        return fit_control(name, seed, data, args)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, center, scale = create_model(name, data, args.device)
    reference = is_reference(name)
    use_capture = getattr(args, "cuda_graphs", False) and args.device == "cuda"
    lr, wd = ((1.19e-4, 3.21e-6) if name.endswith("lstm") else (2.11e-4, 3.62e-5)) if name != "residual_identity" else (1e-3, 0.01)
    optimizer = (torch.optim.Adam if reference else torch.optim.AdamW)(model.parameters(),
        lr=torch.tensor(lr, device=args.device) if use_capture else lr, weight_decay=wd,
        capturable=use_capture, foreach=False if use_capture else None)
    scheduler = TensorPreservingPlateau(optimizer, factor=0.1, patience=5)
    key, best, best_epoch, history = f"{name}_{seed}", float("inf"), -1, []
    latest, first_epoch = args.output / f"{key}_latest.pt", 0
    if latest.exists() and args.resume:
        saved = torch.load(latest, map_location=args.device, weights_only=False)
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, first_epoch = saved["history"], saved["epoch"] + 1
        best, best_epoch = saved["best"], saved["best_epoch"]
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["rng_cpu"].cpu())
        if args.device == "cuda":
            torch.cuda.set_rng_state(saved["rng_cuda"].cpu())
    elif (args.output / f"{key}_best.pt").exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {key}")
    captured = None
    if use_capture and first_epoch - 1 - best_epoch < args.patience:
        model.train()
        def loss_builder(ids, mask):
            # Only full groups are captured; partial batches use the eager path.
            return forward_batch(model, tensors, ids, center, scale, reference, capture=True)[1]
        captured = CapturedSequenceStep(model, optimizer, loss_builder, args.batch_size,
            args.capture_steps, args.device, initial_ids=torch.tensor(
                data["splits"]["train"][:args.batch_size * args.capture_steps], device=args.device))
    started = time.time()
    for epoch in range(first_epoch, args.epochs):
        if first_epoch and first_epoch - 1 - best_epoch >= args.patience:
            break
        losses, maes = [], []
        for split, training in [("train", True), ("validation", False)]:
            model.train(training)
            order = data["splits"][split].copy()
            if training:
                np.random.shuffle(order)
            loss_sum, absolute, count = 0.0, 0.0, 0
            if training and captured is not None:
                order_gpu = torch.tensor(order, device=args.device)
                step_losses, steps = 0.0, 0
                for offset in range(0, len(order), captured.size):
                    group = order_gpu[offset:offset + captured.size]
                    if len(group) == captured.size:
                        step_losses += float(captured(group)) * args.capture_steps
                        steps += args.capture_steps
                    else:
                        for rest in range(0, len(group), args.batch_size):
                            idx = group[rest:rest + args.batch_size]
                            _, loss, _ = forward_batch(model, tensors, idx, center, scale, reference)
                            optimizer.zero_grad(set_to_none=False)
                            loss.backward()
                            optimizer.step()
                            step_losses += float(loss.detach())
                            steps += 1
                losses.append(step_losses / steps)
                maes.append(float("nan"))  # Training MAE is not used or reported.
                continue
            with torch.set_grad_enabled(training):
                for offset in range(0, len(order), args.batch_size):
                    idx = torch.tensor(order[offset:offset + args.batch_size], device=args.device)
                    pred, loss, valid = forward_batch(model, tensors, idx, center, scale, reference)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    n = int(valid.sum())
                    loss_sum += float(loss.detach()) * n
                    absolute += float((pred.detach() - tensors["actual"][idx])[valid].abs().sum())
                    count += n
            losses.append(loss_sum / count)
            maes.append(absolute / count)
        scheduler.step(losses[1])
        row = {"model": name, "seed": seed, "epoch": epoch + 1, "train_loss": losses[0],
               "validation_loss": losses[1], "validation_mae": maes[1], "seconds": time.time() - started}
        history.append(row)
        print(json.dumps(row), flush=True)
        if losses[1] < best:
            best, best_epoch = losses[1], epoch
            save_checkpoint({"state_dict": model.state_dict(), "model": name, "seed": seed, "best_epoch": epoch + 1,
                        "validation_mae": maes[1], "validation_loss": best,
                        "features": data["features"], "states": data["states"],
                        "scalers": {k: np.asarray(data[k]).tolist() for k in ["center", "scale", "feature_center",
                                      "feature_scale", "target_center", "target_scale"]}}, args.output / f"{key}_best.pt")
        save_json(history, args.output / f"{key}_history.json")
        save_checkpoint({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "history": history, "epoch": epoch,
                        "best": best, "best_epoch": best_epoch, "numpy_rng": np.random.get_state(),
                        "rng_cpu": torch.get_rng_state(),
                        "rng_cuda": torch.cuda.get_rng_state() if args.device == "cuda" else None}, latest)
        if epoch - best_epoch >= args.patience:
            break
    return {"model": name, "seed": seed, "best_epoch": best_epoch + 1, "epochs": len(history), "validation_loss": best}


def evaluate_one(name, seed, data, tensors, args):
    torch.manual_seed(seed)
    if name not in CONTROLS:
        model, center, scale = create_model(name, data, args.device)
        saved = torch.load(args.output / f"{name}_{seed}_best.pt", map_location=args.device, weights_only=False)
        if saved["features"] != data["features"] or saved["states"] != data["states"]:
            raise ValueError("Evaluation feature/state ordering changed")
        for scaler_name, values in saved["scalers"].items():
            np.testing.assert_array_equal(np.asarray(data[scaler_name]), np.asarray(values),
                                          err_msg=f"Evaluation scaler changed: {scaler_name}")
        model.load_state_dict(saved["state_dict"])
        model.eval()
    summaries = []
    for period in ["test2024", "test2025"]:
        path = args.output / period / f"predictions_{name}_{seed}.parquet"
        if path.exists():
            raise FileExistsError(path)
        frames = []
        with torch.inference_mode():
            for offset in range(0, len(data["splits"][period]), 256):
                ids = data["splits"][period][offset:offset + 256]
                idx = torch.tensor(ids, device=args.device)
                if name in CONTROLS:
                    predictions = predict_control(name, seed, data, ids, args)
                else:
                    pred, _, _ = forward_batch(model, tensors, idx, center, scale, is_reference(name))
                    predictions = pred.cpu().numpy()
                records = data["records"].iloc[ids]
                actual = data["actual"][ids]
                frames.append(pd.DataFrame({"issued_at": np.repeat(records.issued_at.to_numpy(), 24),
                    "state": np.repeat(records.region.to_numpy(), 24), "horizon": np.tile(np.arange(1, 25), len(ids)),
                    "target_at": np.repeat(records.target_start.to_numpy(), 24) + np.tile(np.arange(24), len(ids)) * np.timedelta64(1, "h"),
                    "actual": actual.ravel(), "predicted": predictions.ravel()}))
        forecast = pd.concat(frames, ignore_index=True).dropna(subset=["actual"])
        path.parent.mkdir(exist_ok=True)
        forecast.to_parquet(path, index=False)
        actual, pred = forecast.actual.to_numpy().astype(float), forecast.predicted.to_numpy().astype(float)
        error, nonzero = pred - actual, np.abs(actual) >= 10
        summaries.append({"model": name, "seed": seed, "period": period,
            "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
            "mape": float((np.abs(error[nonzero] / actual[nonzero])).mean() * 100),
            "mape_n_excluded": int((~nonzero).sum()), "n_obs": len(actual)})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "fit", "evaluate"])
    parser.add_argument("--cache", type=Path, default=ROOT / "data/processed/hfed_extension_eccc_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/hfed_extension_eccc_v4")
    parser.add_argument("--models", nargs="+", choices=MODELS + SUPPLEMENTARY, default=MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--feature-set", choices=["demand", "demand_weather", "demand_weather_calendar"], default="demand_weather_calendar")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capture-steps", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = prepared_data(ROOT, args.cache)
    print(json.dumps({"examples": {k: len(v) for k, v in data["splits"].items()}, "exclusions": data["exclusions"]}), flush=True)
    if args.stage == "prepare":
        return
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "plan.json"
    if args.stage == "fit":
        if plan_path.exists() and not args.resume:
            raise FileExistsError("Choose a new output directory")
        plan = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
        plan.update({"train": "forecast dates before 2024-01-01", "validation": "Jan-Jun 2024",
                     "boundary_purging": data["exclusions"],
                     "tests": {"test2024": "Jul-Dec 2024 (previously inspected period)", "test2025": "fresh 2025 holdout"},
                     "weather": "Official ECCC hourly station observations; one-hour reporting buffer assumed, archived revisions not reproduced.",
                     "issue": "23:00 local; latest input interval ends at issue time",
                     "graph": "Geographic <=2000km, edge attribute distance; no invented grid capacities",
                     "information_difference": "Seasonal/residual methods add seven daily lags and target calendars to the 24h history used by the Grid-Aware references.",
                     "residual_cnn_inputs": "Past demand and observation mask, seasonal load lags and known target calendars; no weather inputs.",
                     "training_loss_reporting": "Mean optimization-step loss for CUDA replay; valid-observation weighted loss for eager execution. Validation always uses valid-observation weighting.",
                     "scheduler": "TensorPreservingPlateau; factor 0.1, patience 5; captured LR storage updated in place",
                     "environment": {"torch": torch.__version__, "python": sys.version,
                                     "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None},
                     "cache_sha256": hashlib.sha256((args.cache / "examples.npz").read_bytes()).hexdigest()})
        selection = ROOT / "results/grid_aware_confirm_v3/method_selection.json"
        if selection.exists():
            plan["brazil_validation_selection"] = json.loads(selection.read_text(encoding="utf-8"))
        if any(model in SUPPLEMENTARY for model in args.models):
            plan["supplementary_scope"] = "Own-history temporal controls added for proposal/dashboard after primary 2025 results were inspected. Fixed published RNN dimensions/rates; validation-only checkpoint selection; no test tuning. Descriptive comparison, not another blinded confirmation."
        if plan_path.exists():
            validate_frozen_plan(json.loads(plan_path.read_text(encoding="utf-8")), plan)
        else:
            plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            for source in [Path(__file__), ROOT / "src/data/day_ahead_benchmark.py",
                           ROOT / "src/data/regional_timezones.py", ROOT / "src/models/stgnn/grid_aware.py",
                           ROOT / "src/models/stgnn/residual.py", ROOT / "src/models/hfed_tabular.py",
                           ROOT / "src/models/stgnn/cuda_training.py", ROOT / "src/models/checkpoint_io.py"]:
                shutil.copyfile(source, args.output / f"source_{source.name}")
        summaries = []
        tensors = inputs(data, args.device, args.feature_set)
        for seed in args.seeds:
            for name in args.models:
                summaries.append(fit_one(name, seed, data, tensors, args))
                save_json(summaries, args.output / "fit_summary.json")
    else:
        if (args.output / "evaluation.json").exists():
            raise FileExistsError("Evaluation already exists")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if args.models != plan["models"] or args.seeds != plan["seeds"] or args.feature_set != plan["feature_set"]:
            raise ValueError("Evaluation must match the frozen training plan")
        if hashlib.sha256((args.cache / "examples.npz").read_bytes()).hexdigest() != plan["cache_sha256"]:
            raise ValueError("Prepared data changed after training")
        fits = json.loads((args.output / "fit_summary.json").read_text(encoding="utf-8"))
        if {(r["model"], r["seed"]) for r in fits} != {(m, s) for m in args.models for s in args.seeds}:
            raise ValueError("Training matrix incomplete")
        tensors = inputs(data, args.device, args.feature_set)
        summaries = []
        for seed in args.seeds:
            for name in args.models:
                summaries.extend(evaluate_one(name, seed, data, tensors, args))
        (args.output / "evaluation.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
