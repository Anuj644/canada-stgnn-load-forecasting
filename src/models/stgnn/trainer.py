"""STGCN training loop and graph-ablation experiment runner.

Responsibilities:
- Build (B, T, N, F) tensor batches from the multi-region hourly DataFrame.
- Train the STGCN with early stopping on validation loss.
- Save/load checkpoints.
- Run the full ablation: train one model per graph configuration.
- Support multiple random seeds for uncertainty estimation.

Data leakage prevention:
- Input sequences always end at the forecast issue hour (23:00 local).
- Targets are hours 0–23 of the next calendar day.
- The adjacency matrix for the correlation graph is built from training
  data only before training starts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed.  STGCN trainer unavailable.")

from src.models.stgnn.stgcn import STGCN
from src.models.stgnn.graph_builder import build_all_graphs, describe_graph

FORECAST_HOURS = 24


# ---------------------------------------------------------------------------
# Graph-aware sequence dataset
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class STGCNDataset(Dataset):
        """Builds (X: T×N×F, y: N×H) samples for the STGCN.

        For each forecast issue day:
            X: lookback_hours × N_regions × n_features
            y: N_regions × 24  (demand at each hour of the next day)
        """

        def __init__(
            self,
            df: pd.DataFrame,
            regions: list[str],
            lookback_hours: int = 168,
            feature_cols: Optional[list[str]] = None,
            demand_col: str = "demand_mw",
            time_col: str = "time_utc",
            region_col: str = "region",
        ):
            self.regions = regions
            self.n_nodes = len(regions)
            self.lookback_hours = lookback_hours
            self.demand_col = demand_col

            if feature_cols is None:
                skip = {time_col, "time_local", region_col, "DATETIME_LOCAL_OFFSET",
                        "holiday_name", "year"}
                sample_region = df[region_col].iloc[0]
                feature_cols = [
                    c for c in df.columns
                    if c not in skip and pd.api.types.is_numeric_dtype(df[c])
                ]
            self.feature_cols = feature_cols
            self.n_features = len(feature_cols)

            self.X_list: list[np.ndarray] = []
            self.y_list: list[np.ndarray] = []
            self.meta: list[dict] = []

            self._build(df, time_col, region_col)

        def _build(self, df: pd.DataFrame, time_col: str, region_col: str) -> None:
            # Pivot: index=time_utc, columns=region, values=feature_cols
            # Build a 3-D array: (T_total, N, F)
            pivot_dict = {}
            for feat in self.feature_cols:
                wide = df.pivot_table(
                    index=time_col,
                    columns=region_col,
                    values=feat,
                    aggfunc="mean",
                )
                # Align columns to our region ordering
                wide = wide.reindex(columns=self.regions)
                pivot_dict[feat] = wide

            # Build a complete regular hourly common time index to align all regions.
            # Different regions may have different start dates, so pivot tables can
            # have different row counts. Reindex to a shared hourly grid fills gaps with NaN.
            raw_index = pivot_dict[self.feature_cols[0]].index
            common_times = pd.date_range(
                start=raw_index.min(), end=raw_index.max(), freq="h"
            )
            for feat in self.feature_cols:
                pivot_dict[feat] = pivot_dict[feat].reindex(common_times).fillna(0.0)
            n_times = len(common_times)

            # Stack into (T, N, F)
            feat_array = np.stack(
                [pivot_dict[f].values for f in self.feature_cols], axis=2
            ).astype(np.float32)
            # Replace NaN with 0
            feat_array = np.nan_to_num(feat_array, nan=0.0)

            # Demand array for target construction: (T, N)
            demand_wide = df.pivot_table(
                index=time_col, columns=region_col, values=self.demand_col, aggfunc="mean"
            ).reindex(columns=self.regions).reindex(common_times).fillna(0.0)
            demand_array = demand_wide.values.astype(np.float32)

            # Local hour: take from first available region
            first_region = self.regions[0]
            region_df = df[df[region_col] == first_region].sort_values(time_col)
            if "time_local" in region_df.columns:
                local_dt = pd.to_datetime(region_df["time_local"])
            else:
                local_dt = pd.to_datetime(region_df[time_col])

            time_to_idx = {t: i for i, t in enumerate(common_times)}

            # Build a local-time index aligned to common_times
            aligned_local = region_df.set_index(time_col).reindex(common_times)
            if "time_local" in aligned_local.columns:
                local_hours = pd.to_datetime(aligned_local["time_local"]).dt.hour.fillna(-1).astype(int).values
                local_dates = pd.to_datetime(aligned_local["time_local"]).dt.date.values
            else:
                utc_times = pd.DatetimeIndex(common_times)
                local_hours = utc_times.hour.values
                local_dates = utc_times.date

            # Find issue indices (local hour == 23)
            issue_indices = np.where(local_hours == 23)[0]

            for idx in issue_indices:
                start = idx - self.lookback_hours + 1
                if start < 0:
                    continue

                X_seq = feat_array[start: idx + 1]  # (lookback, N, F)
                if X_seq.shape[0] != self.lookback_hours:
                    continue

                # Find next-day target
                issue_date = local_dates[idx]
                if issue_date is None or (isinstance(issue_date, float) and np.isnan(issue_date)):
                    continue
                try:
                    next_date = (pd.Timestamp(issue_date) + pd.Timedelta(days=1)).date()
                except Exception:
                    continue

                # Find indices for all 24 hours of next_date in local time
                next_day_mask = (local_dates == next_date)
                next_day_hours_arr = local_hours[next_day_mask]
                next_day_indices = np.where(next_day_mask)[0]

                if len(next_day_indices) < FORECAST_HOURS:
                    continue

                # Sort by local hour and take exactly 24
                sort_order = np.argsort(next_day_hours_arr[:FORECAST_HOURS])
                next_day_indices_sorted = next_day_indices[:FORECAST_HOURS][sort_order]

                y = demand_array[next_day_indices_sorted]  # (24, N)
                y = y.T  # (N, 24)

                self.X_list.append(X_seq)
                self.y_list.append(y)
                self.meta.append({
                    "issue_date": issue_date,
                    "forecast_date": next_date,
                })

            logger.info(
                "STGCNDataset: %d samples, lookback=%d, nodes=%d, features=%d",
                len(self.X_list), self.lookback_hours, self.n_nodes, self.n_features,
            )

        def __len__(self) -> int:
            return len(self.X_list)

        def __getitem__(self, idx: int):
            X = torch.tensor(self.X_list[idx], dtype=torch.float32)
            y = torch.tensor(self.y_list[idx], dtype=torch.float32)
            return X, y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_stgcn(
    model: "STGCN",
    train_dataset: "STGCNDataset",
    val_dataset: "STGCNDataset",
    config: dict,
    device: "torch.device",
    checkpoint_path: Optional[Path] = None,
) -> dict:
    """Train a STGCN model with early stopping.

    Returns a dict with training history.
    """
    stgnn_cfg = config.get("stgnn", {})
    batch_size = stgnn_cfg.get("batch_size", 32)
    epochs = stgnn_cfg.get("epochs", 150)
    lr = stgnn_cfg.get("learning_rate", 0.001)
    patience = stgnn_cfg.get("patience", 20)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=patience // 3, factor=0.5, min_lr=1e-5
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) \
        if len(val_dataset) > 0 else None

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)          # (B, N, H)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))
        history["train_loss"].append(avg_train)

        if val_loader:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_v, y_v in val_loader:
                    X_v, y_v = X_v.to(device), y_v.to(device)
                    val_losses.append(criterion(model(X_v), y_v).item())
            avg_val = float(np.mean(val_losses))
            history["val_loss"].append(avg_val)
            scheduler.step(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                if checkpoint_path:
                    torch.save(best_state, checkpoint_path)
            else:
                patience_counter += 1

            if epoch % 10 == 0 or patience_counter == 0:
                logger.info(
                    "STGCN Epoch %03d/%d  train=%.4f  val=%.4f  patience=%d",
                    epoch + 1, epochs, avg_train, avg_val, patience_counter,
                )

            if patience_counter >= patience:
                logger.info("STGCN: early stopping at epoch %d.", epoch + 1)
                break
        else:
            history["val_loss"].append(None)
            if epoch % 10 == 0:
                logger.info("STGCN Epoch %03d/%d  train=%.4f", epoch + 1, epochs, avg_train)

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_stgcn(
    model: "STGCN",
    dataset: "STGCNDataset",
    batch_size: int = 32,
    device: Optional["torch.device"] = None,
) -> pd.DataFrame:
    """Run inference on a dataset and return a tidy prediction DataFrame.

    Returns columns: forecast_date, region, target_hour, predicted, actual
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            pred = model(X_b).cpu().numpy()   # (B, N, H)
            all_preds.append(pred)
            all_actuals.append(y_b.numpy())

    preds = np.concatenate(all_preds, axis=0)     # (n_samples, N, H)
    actuals = np.concatenate(all_actuals, axis=0)  # (n_samples, N, H)

    rows = []
    for sample_idx, meta in enumerate(dataset.meta):
        if sample_idx >= len(preds):
            break
        for node_idx, region in enumerate(dataset.regions):
            for h in range(FORECAST_HOURS):
                rows.append({
                    "forecast_date": meta["forecast_date"],
                    "region": region,
                    "target_hour": h,
                    "predicted": float(preds[sample_idx, node_idx, h]),
                    "actual": float(actuals[sample_idx, node_idx, h]),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full ablation runner
# ---------------------------------------------------------------------------

class STGCNTrainer:
    """Orchestrates the full STGNN training + graph ablation experiments."""

    name = "stgnn"

    def __init__(self, config: dict):
        if not TORCH_AVAILABLE:
            raise ImportError("Install torch:  pip install torch")
        self.config = config
        stgnn_cfg = config.get("stgnn", {})
        self.lookback_hours = stgnn_cfg.get("lookback_hours", 168)
        self.graph_types = stgnn_cfg.get("graph_types",
                                          ["identity", "geographic", "correlation", "interconnection"])
        self.random_seeds = stgnn_cfg.get("random_seeds", [42])
        self.num_blocks = stgnn_cfg.get("num_st_blocks", 3)
        self.channels = stgnn_cfg.get("channels", [64, 64, 64])
        self.kernel_size = stgnn_cfg.get("kernel_size", 3)
        self.dropout = stgnn_cfg.get("dropout", 0.1)
        self.horizon = stgnn_cfg.get("horizon", 24)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results: dict[str, dict] = {}

    def run(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        regions: list[str],
        model_dir: Optional[str | Path] = None,
    ) -> dict[str, pd.DataFrame]:
        """Train and evaluate all graph configurations.

        Returns {graph_type: prediction_DataFrame}
        """
        model_dir = Path(model_dir) if model_dir else Path("models/stgnn")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Build graphs (correlation uses training data only)
        stgnn_cfg = self.config.get("stgnn", {})
        graphs = build_all_graphs(
            regions=regions,
            train_df=train_df,
            max_distance_km=stgnn_cfg.get("geographic_max_distance_km", 2000),
            correlation_threshold=stgnn_cfg.get("correlation_threshold", 0.6),
        )

        # Log graph descriptions
        for gname, A in graphs.items():
            logger.info("\n%s", describe_graph(A, regions, name=gname))

        # Save graph adjacency matrices
        import json as _json
        for gname, A in graphs.items():
            np.save(model_dir / f"adj_{gname}.npy", A)

        # Build datasets (shared across graph types)
        logger.info("Building STGCN training dataset...")
        train_ds = STGCNDataset(train_df, regions=regions, lookback_hours=self.lookback_hours)
        val_ds = STGCNDataset(val_df, regions=regions, lookback_hours=self.lookback_hours)
        test_ds = STGCNDataset(test_df, regions=regions, lookback_hours=self.lookback_hours)

        if len(train_ds) == 0:
            logger.error("STGCN: no training samples. Check data coverage.")
            return {}

        n_nodes = len(regions)
        n_features = train_ds.n_features

        all_predictions = {}

        for graph_type in self.graph_types:
            if graph_type not in graphs:
                logger.warning("Graph type '%s' not built; skipping.", graph_type)
                continue

            A = graphs[graph_type]
            logger.info("=" * 60)
            logger.info("Training STGCN with '%s' graph (%d seeds)", graph_type, len(self.random_seeds))

            seed_preds = []
            for seed in self.random_seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)

                model = STGCN(
                    n_nodes=n_nodes,
                    in_features=n_features,
                    horizon=self.horizon,
                    num_blocks=self.num_blocks,
                    channels=self.channels,
                    kernel_size=self.kernel_size,
                    dropout=self.dropout,
                    adj=A,
                )

                ckpt_path = model_dir / f"stgcn_{graph_type}_seed{seed}.pt"
                history = train_stgcn(
                    model, train_ds, val_ds,
                    config=self.config,
                    device=self.device,
                    checkpoint_path=ckpt_path,
                )
                self.results[f"{graph_type}_seed{seed}"] = history

                # Save history
                hist_path = model_dir / f"history_{graph_type}_seed{seed}.json"
                hist_path.write_text(
                    _json.dumps(history, indent=2), encoding="utf-8"
                )

                preds = predict_stgcn(model, test_ds, device=self.device)
                preds["seed"] = seed
                preds["graph_type"] = graph_type
                seed_preds.append(preds)

            if seed_preds:
                combined_preds = pd.concat(seed_preds, ignore_index=True)
                # Ensemble: mean over seeds
                ensemble = (
                    combined_preds.groupby(["forecast_date", "region", "target_hour"])
                    .agg(predicted=("predicted", "mean"), actual=("actual", "first"))
                    .reset_index()
                )
                ensemble["graph_type"] = graph_type
                all_predictions[graph_type] = ensemble

                # Save predictions
                preds_path = model_dir / f"predictions_{graph_type}.parquet"
                ensemble.to_parquet(preds_path, index=False)
                logger.info(
                    "STGCN '%s' predictions saved: %d rows", graph_type, len(ensemble)
                )

        return all_predictions
