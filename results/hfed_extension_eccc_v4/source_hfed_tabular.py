"""Matched HFED naive and XGBoost controls using the same causal examples."""
import json
from functools import lru_cache

import numpy as np

NAMES = ["xgboost", "naive_day_available", "naive_week"]


def tabular_features(data, ids, feature_set):
    dynamic, cov = data["dynamic"][ids].copy(), data["covariates"][ids].copy()
    if feature_set == "demand":
        for i, name in enumerate(data["features"]):
            if any(name.startswith(prefix) for prefix in ["temperature_c", "relative_humidity", "wind_speed"]):
                dynamic[:, :, i] = 0
    if feature_set != "demand_weather_calendar":
        for i, name in enumerate(data["features"]):
            if name.startswith("calendar_"):
                dynamic[:, :, i] = 0
        cov[:, :, 14:] = 0
    base = np.concatenate([dynamic[:, :, 0], dynamic[:, -1], dynamic[:, :, :4].mean(1),
                           dynamic[:, :, :4].std(1)], axis=1)
    nodes = np.eye(len(data["states"]), dtype="float32")[data["nodes"][ids]]
    return np.concatenate([np.repeat(base, 24, axis=0), cov.reshape(-1, 26),
                           np.repeat(nodes, 24, axis=0)], axis=1).astype("float32")


def fit_control(name, seed, data, args):
    from src.models.checkpoint_io import atomic_bytes, save_json
    meta_path = args.output / f"{name}_{seed}_metadata.json"
    if meta_path.exists():
        if not args.resume:
            raise FileExistsError(meta_path)
        if name == "xgboost" and not (args.output / f"{name}_{seed}_best.json").exists():
            raise FileNotFoundError("XGBoost metadata exists but its model is missing")
        return json.loads(meta_path.read_text(encoding="utf-8"))
    result = {"model": name, "seed": seed, "epochs": 0, "best_epoch": 0, "validation_loss": None}
    if name == "xgboost":
        import xgboost as xgb
        ids, val_ids = data["splits"]["train"], data["splits"]["validation"]
        x, v = tabular_features(data, ids, args.feature_set), tabular_features(data, val_ids, args.feature_set)
        y, target = data["actual"][ids].ravel(), data["actual"][val_ids].ravel()
        valid, val_valid = np.isfinite(y), np.isfinite(target)
        model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=4,
            tree_method="hist", device=args.device, objective="reg:squarederror",
            eval_metric="mae", early_stopping_rounds=50)
        model.fit(x[valid], y[valid], eval_set=[(v[val_valid], target[val_valid])], verbose=False)
        atomic_bytes(args.output / f"{name}_{seed}_best.json", model.get_booster().save_raw(raw_format="json"))
        history = model.evals_result()["validation_0"]["mae"]
        result.update(epochs=len(history), best_epoch=int(model.best_iteration) + 1,
                      validation_loss=float(model.best_score), n_features=x.shape[1])
        save_json(history, args.output / f"{name}_{seed}_history.json")
    save_json(result, meta_path)
    print(json.dumps(result), flush=True)
    return result


@lru_cache(maxsize=4)
def _loaded_booster(path, device):
    import xgboost as xgb
    model = xgb.Booster(params={"device": device})
    model.load_model(bytearray(path.read_bytes()))
    return model


def predict_control(name, seed, data, ids, args):
    if name == "xgboost":
        import xgboost as xgb
        model = _loaded_booster(args.output / f"{name}_{seed}_best.json", args.device)
        meta = json.loads((args.output / f"{name}_{seed}_metadata.json").read_text(encoding="utf-8"))
        x = tabular_features(data, ids, args.feature_set)
        return model.predict(xgb.DMatrix(x), iteration_range=(0, meta["best_epoch"])).reshape(-1, 24)
    normalized = data["anchor"][ids]
    if name == "naive_week":
        cov = data["covariates"][ids]
        normalized = np.where(cov[:, :, 13] > 0, cov[:, :, 6], normalized)
    nodes = data["nodes"][ids]
    return normalized * data["scale"][nodes, None] + data["center"][nodes, None]
