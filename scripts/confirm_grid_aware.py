"""Train with validation-only selection, then evaluate a frozen repeated-seed plan.

fit never uses test labels for selection. evaluate loads frozen best checkpoints and does not
update weights, hyperparameters, or blending weights. Original pilot is preserved.
"""
from __future__ import annotations

import argparse
import gc
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
from scripts.benchmark_grid_aware import (
    GridAware, HORIZONS, ResidualAdapter, SPLITS, device_data, issue_indices,
    load_data, make_batch, score,
)
from src.models.stgnn.grid_aware import SeasonalGridAware
from src.models.stgnn.cuda_training import CapturedStep, CapturedSequenceStep, TensorPreservingPlateau
from src.models.checkpoint_io import save_checkpoint, save_json
from src.evaluation.experiment_plan import validate_frozen_plan

MODELS = ["grid_aware_lstm", "grid_aware_gru", "seasonal_identity", "seasonal_grid_aware",
          "residual_identity", "residual_grid"]


def sample_batch(arrays, issues, nodes):
    history = issues[:, None] - torch.arange(23, -1, -1, device=issues.device)
    targets = issues[:, None] + torch.as_tensor(HORIZONS, device=issues.device)[None]
    return (arrays["dynamic"][history, nodes[:, None]],
            arrays["demand"][targets, nodes[:, None]])


def sample_seasonal_covariates(arrays, issues, nodes):
    targets = issues[:, None] + torch.as_tensor(HORIZONS, device=issues.device)[None]
    lag_idx = targets[:, :, None] - 24 * torch.arange(1, 8, device=issues.device)[None, None]
    lags = arrays["normalized"][lag_idx, nodes[:, None, None]]
    calendar = arrays["calendar"][targets]
    holiday = arrays["holiday"][targets, nodes[:, None]][..., None]
    return torch.cat([lags, torch.ones_like(lags), calendar, holiday], -1), lags[..., 0]


def model_and_scalers(name, data, device):
    if name.startswith("grid_aware"):
        model = GridAware(data, backbone=name.removeprefix("grid_aware_"))
        return model.to(device), data["target_center"], data["target_scale"]
    if name.startswith("seasonal"):
        inputs = data
        if name == "seasonal_identity":
            inputs = dict(data, edge_index=np.empty((2, 0), dtype=np.int64),
                          edge_features=np.empty((0, data["edge_features"].shape[1])))
        model = SeasonalGridAware(inputs).to(device)
    else:
        graph = data["adjacency"] if name == "residual_grid" else np.eye(len(data["states"]))
        model = ResidualAdapter(graph).to(device)
    center = torch.tensor(data["center"], device=device, dtype=torch.float32)[None, :, None]
    scale = torch.tensor(data["scale"], device=device, dtype=torch.float32)[None, :, None]
    return model, center, scale


def validation(model, name, data, arrays, center, scale, device):
    model.eval()
    loss_sum = torch.zeros((), device=device)
    squared, absolute, count = torch.zeros(3, device=device), torch.zeros(3, device=device), 0
    with torch.inference_mode():
        for offset in range(0, len(data["splits"]["validation"]), 64):
            idx = data["splits"]["validation"][offset:offset + 64]
            batch = make_batch(arrays, idx)
            target = (batch["actual"] - center) / scale
            pred = model(batch)
            loss = (nn.functional.mse_loss(pred, target, reduction="sum") if name.startswith("grid_aware")
                    else nn.functional.smooth_l1_loss(pred, target, beta=0.25, reduction="sum"))
            error = pred * scale + center - batch["actual"]
            absolute += error.abs().sum((0, 1))
            squared += error.square().sum((0, 1))
            count += error.shape[0] * error.shape[1]
            loss_sum += loss
    return (float(loss_sum.item() / (count * 3)), (absolute / count).cpu().tolist(),
            (squared / count).sqrt().cpu().tolist())


def train(name, seed, data, arrays, args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, center, scale = model_and_scalers(name, data, args.device)
    reference = name.startswith("grid_aware")
    recurrent = reference or name.startswith("seasonal")
    lr, wd = ((1.19e-4, 3.21e-6) if name.endswith("lstm") else (2.11e-4, 3.62e-5)) if recurrent else (1e-3, 0.01)
    optimizer_class = torch.optim.Adam if reference else torch.optim.AdamW
    use_capture = args.cuda_graphs and recurrent and args.device == "cuda"
    optimizer = optimizer_class(model.parameters(), lr=torch.tensor(lr, device=args.device) if use_capture else lr,
                                weight_decay=wd, capturable=use_capture, foreach=False)
    scheduler = TensorPreservingPlateau(optimizer, factor=0.1, patience=5)
    key = f"{name}_{seed}"
    latest, best_path = args.output / f"{key}_latest.pt", args.output / f"{key}_best.pt"
    history, first_epoch, best, best_epoch = [], 0, float("inf"), -1
    if latest.exists():
        if not args.resume:
            raise FileExistsError(f"Existing {latest}; use --resume to continue the same frozen plan")
        saved = torch.load(latest, map_location=args.device, weights_only=False)
        best_saved = torch.load(best_path, map_location="cpu", weights_only=False)
        if best_saved["features"] != data["features"] or best_saved["states"] != data["states"]:
            raise ValueError("Resume feature/state ordering changed")
        for scaler_name, values in best_saved["scalers"].items():
            np.testing.assert_array_equal(np.asarray(data[scaler_name]), np.asarray(values),
                                          err_msg=f"Resume scaler changed: {scaler_name}")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, first_epoch = saved["history"], saved["epoch"] + 1
        best, best_epoch = saved["best"], saved["best_epoch"]
        torch.set_rng_state(saved["rng_cpu"].cpu())
        if args.device == "cuda":
            torch.cuda.set_rng_state(saved["rng_cuda"].cpu())
    train_issues = torch.as_tensor(data["splits"]["train"], device=args.device)
    states = len(data["states"])
    captured = None
    if use_capture:
        model.train()
        history_offsets = torch.arange(23, -1, -1, device=args.device)
        horizons = torch.tensor(HORIZONS, device=args.device)
        day_offsets = 24 * torch.arange(1, 8, device=args.device)

        def loss_builder(chosen, mask):
            issues, nodes = train_issues[chosen // states], chosen % states
            targets = issues[:, None] + horizons[None]
            seq = arrays["dynamic"][issues[:, None] - history_offsets[None], nodes[:, None]]
            actual = arrays["demand"][targets, nodes[:, None]]
            if reference:
                target = (actual - center) / scale
                pred = model.forward_samples(seq, nodes)
                loss_values = (pred - target).square()
            else:
                target = (actual - center[0, nodes, 0][:, None]) / scale[0, nodes, 0][:, None]
                lag_idx = targets[:, :, None] - day_offsets[None, None]
                lags = arrays["normalized"][lag_idx, nodes[:, None, None]]
                calendar = arrays["calendar"][targets]
                holiday = arrays["holiday"][targets, nodes[:, None]][..., None]
                cov = torch.cat([lags, torch.ones_like(lags), calendar, holiday], -1)
                pred = model.forward_samples(seq, nodes, cov, lags[..., 0])
                loss_values = nn.functional.smooth_l1_loss(pred, target, beta=0.25, reduction="none")
            return (loss_values * mask[:, None]).sum() / (mask.sum() * 3)

        captured = (CapturedSequenceStep(model, optimizer, loss_builder, args.sample_batch,
                                        args.capture_steps, args.device) if args.capture_steps > 1 else
                    CapturedStep(model, optimizer, loss_builder, args.sample_batch, args.device))
    started = time.time()
    for epoch in range(first_epoch, args.epochs):
        if first_epoch and first_epoch - 1 - best_epoch >= args.patience:
            break
        model.train()
        loss_sum = torch.zeros((), device=args.device)
        count = 0
        if recurrent:
            order = torch.randperm(len(train_issues) * states, device=args.device)
            step_size = args.sample_batch * (args.capture_steps if captured is not None else 1)
        else:
            order = torch.randperm(len(train_issues), device=args.device)
            step_size = args.residual_batch_times
        epoch_started = time.time()
        for offset in range(0, len(order), step_size):
            chosen = order[offset:offset + step_size]
            if isinstance(captured, CapturedSequenceStep) and len(chosen) != captured.size:
                for remainder in range(0, len(chosen), args.sample_batch):
                    ids = chosen[remainder:remainder + args.sample_batch]
                    optimizer.zero_grad(set_to_none=False)
                    loss = loss_builder(ids, torch.ones(len(ids), device=args.device))
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.detach() * (len(ids) * 3)
                    count += len(ids) * 3
                continue
            if captured is not None:
                loss = captured(chosen)
                count_elements = len(chosen) * 3
            elif recurrent:
                issues, nodes = train_issues[chosen // states], chosen % states
                seq, actual = sample_batch(arrays, issues, nodes)
                if reference:
                    target = (actual - center) / scale
                    pred = model.forward_samples(seq, nodes)
                    loss = nn.functional.mse_loss(pred, target)
                else:
                    target = (actual - center[0, nodes, 0][:, None]) / scale[0, nodes, 0][:, None]
                    covariates, anchor = sample_seasonal_covariates(arrays, issues, nodes)
                    pred = model.forward_samples(seq, nodes, covariates, anchor)
                    loss = nn.functional.smooth_l1_loss(pred, target, beta=0.25)
            else:
                batch = make_batch(arrays, train_issues[chosen])
                target = (batch["actual"] - center) / scale
                pred = model(batch)
                loss = nn.functional.smooth_l1_loss(pred, target, beta=0.25)
            if captured is None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                count_elements = target.numel()
            loss_sum += loss.detach() * count_elements
            count += count_elements
            if offset and offset // step_size % 4000 == 0:
                print(json.dumps({"model": name, "seed": seed, "epoch": epoch + 1,
                                  "progress": round(offset / len(order), 3)}), flush=True)
        val_loss, val_mae, val_rmse = validation(model, name, data, arrays, center, scale, args.device)
        scheduler.step(val_loss)
        row = {"epoch": epoch + 1, "train_loss": float(loss_sum.item() / count),
               "validation_loss": val_loss, "validation_mae_1_6_24": val_mae,
               "validation_rmse_1_6_24": val_rmse,
               "lr": float(optimizer.param_groups[0]["lr"]), "epoch_seconds": time.time() - epoch_started}
        history.append(row)
        improved = val_loss < best
        if improved:
            best, best_epoch = val_loss, epoch
            save_checkpoint({"state_dict": model.state_dict(), "model": name, "seed": seed,
                        "best_epoch": best_epoch + 1, "validation_loss": best,
                        "validation_mae_1_6_24": val_mae,
                        "scalers": {k: np.asarray(data[k]).tolist() for k in
                         ["center", "scale", "feature_center", "feature_scale", "target_center", "target_scale"]},
                        "states": data["states"], "features": data["features"]}, best_path)
        save_checkpoint({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "history": history, "epoch": epoch,
                    "best": best, "best_epoch": best_epoch, "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state() if args.device == "cuda" else None}, latest)
        save_json(history, args.output / f"{key}_history.json")
        print(json.dumps({"model": name, "seed": seed, **row, "best_epoch": best_epoch + 1}), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    return {"model": name, "seed": seed, "epochs": len(history), "best_epoch": best_epoch + 1,
            "best_validation_loss": best, "session_seconds": time.time() - started}


def evaluate(name, seed, data, arrays, args):
    torch.manual_seed(seed)
    model, center, scale = model_and_scalers(name, data, args.device)
    key = f"{name}_{seed}"
    saved = torch.load(args.output / f"{key}_best.pt", map_location=args.device, weights_only=False)
    if saved["features"] != data["features"] or saved["states"] != data["states"]:
        raise ValueError("Evaluation feature/state ordering changed")
    for scaler_name, values in saved["scalers"].items():
        np.testing.assert_array_equal(np.asarray(data[scaler_name]), np.asarray(values),
                                      err_msg=f"Evaluation scaler changed: {scaler_name}")
    model.load_state_dict(saved["state_dict"])
    model.eval()
    rows = []
    periods = {"test2019": data["splits"]["test"]}
    if args.include_2020:
        periods["test2020"] = issue_indices(data["index"], "2020-01-01", "2021-01-01")
    for period, indices in periods.items():
        path = args.output / period / f"predictions_{key}.parquet"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        frames = []
        with torch.inference_mode():
            for offset in range(0, len(indices), 64):
                issues = indices[offset:offset + 64]
                batch = make_batch(arrays, issues)
                pred = (model(batch) * scale + center).cpu().numpy()
                actual = batch["actual"].cpu().numpy()
                for h, horizon in enumerate(HORIZONS):
                    frames.append(pd.DataFrame({
                        "issued_at": np.repeat(data["index"][issues].to_numpy(), len(data["states"])),
                        "state": np.tile(data["states"], len(issues)), "horizon": int(horizon),
                        "actual": actual[:, :, h].ravel(), "predicted": pred[:, :, h].ravel()}))
        frame = pd.concat(frames, ignore_index=True)
        path.parent.mkdir(exist_ok=True)
        frame.to_parquet(path, index=False)
        rows.append({"period": period, "model": name, "seed": seed,
                     "best_epoch": saved["best_epoch"],
                     "metrics": {str(h): score(g.actual.to_numpy(), g.predicted.to_numpy())
                                 for h, g in frame.groupby("horizon")}})
    return rows


def load_run_data(args):
    print(json.dumps({"stage": "loading_dataset", "operation": args.stage}), flush=True)
    data = load_data(args.data)
    arrays = device_data(data, args.device)
    for key in ["dynamic", "history", "normalized", "calendar", "holiday", "demand"]:
        del data[key]
    gc.collect()
    print(json.dumps({"stage": "dataset_ready", "samples": {k: len(v) for k, v in data["splits"].items()}}), flush=True)
    return data, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "evaluate"])
    parser.add_argument("--data", type=Path, default=ROOT / "data/external/grid_aware")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--sample-batch", type=int, default=32)
    parser.add_argument("--residual-batch-times", type=int, default=32)
    parser.add_argument("--include-2020", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capture-steps", type=int, default=16,
                        help="Sequential optimizer steps per GPU replay; does not change optimization batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "plan.json"
    if args.stage == "fit":
        plan = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ["resume", "stage"]}
        plan.update({"selection": "training objective on Jan-Jun 2019 only; no test evaluation during fit",
                     "scheduler": "TensorPreservingPlateau; factor 0.1, patience 5; captured LR storage updated in place",
                     "data_md5": "f5003d072297deea746f63ff59ed4e47", "splits": SPLITS,
                     "information_difference": "Residual uses seven days of direct lags and known target calendar; paper baseline uses 24h history.",
                     "environment": {"torch": torch.__version__, "python": sys.version,
                                     "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None}})
        imported_path = args.output / "imported_fits.json"
        if imported_path.exists():
            plan["imported_unaffected_eager_fits"] = json.loads(imported_path.read_text(encoding="utf-8"))
        if plan_path.exists():
            prior = json.loads(plan_path.read_text(encoding="utf-8"))
            validate_frozen_plan(prior, plan)
        else:
            plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            for source in [Path(__file__), ROOT / "scripts/benchmark_grid_aware.py", ROOT / "src/models/stgnn/residual.py", ROOT / "src/models/stgnn/grid_aware.py", ROOT / "src/models/stgnn/cuda_training.py"]:
                shutil.copyfile(source, args.output / f"source_{source.name}")
        data, arrays = load_run_data(args)
        results = []
        for seed in args.seeds:
            for name in args.models:
                results.append(train(name, seed, data, arrays, args))
                save_json(results, args.output / "fit_summary.json")
    else:
        if (args.output / "evaluation.json").exists():
            raise FileExistsError("Evaluation already exists")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if args.models != plan["models"] or args.seeds != plan["seeds"]:
            raise ValueError("Evaluate every model and seed in the frozen plan")
        if args.include_2020 != plan["include_2020"]:
            raise ValueError("Evaluate exactly the predeclared test periods")
        fits = json.loads((args.output / "fit_summary.json").read_text(encoding="utf-8"))
        if {(r["model"], r["seed"]) for r in fits} != {(m, s) for m in args.models for s in args.seeds}:
            raise ValueError("Training matrix is incomplete; test evaluation is not permitted yet")
        data, arrays = load_run_data(args)
        results = []
        for seed in args.seeds:
            for name in args.models:
                results.extend(evaluate(name, seed, data, arrays, args))
        (args.output / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
