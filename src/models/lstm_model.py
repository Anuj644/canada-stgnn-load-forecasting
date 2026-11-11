"""LSTM multi-region day-ahead electricity demand forecaster.

Architecture:
- Encoder LSTM reads a lookback window of hourly observations.
- A linear decoder head maps the final hidden state to 24 target outputs.
- One model trained on all regions; region is included via a one-hot
  embedding concatenated to the input at each time step.
- Loss: MSE on the 24 target hours.

Data leakage:
- Input sequences end at the 23:00 observation of the issue day.
- Targets are hours 0–23 of the following day.
- No future information is used in the input.
"""

from __future__ import annotations

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
    logger.warning("PyTorch not installed.  LSTMForecaster unavailable.")


FORECAST_HOURS = 24


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DemandSequenceDataset:
    """Builds (X_sequence, y_24h) pairs for sequence models.

    For each forecast issue time (23:00 local on day D):
        X: lookback_hours × n_features array ending at 23:00 of day D
        y: 24-element vector of demand for hours 0–23 of day D+1
    """

    def __init__(
        self,
        df: pd.DataFrame,
        lookback_hours: int = 168,
        feature_cols: Optional[list[str]] = None,
        demand_col: str = "demand_mw",
        time_col: str = "time_utc",
        region_col: str = "region",
        all_regions: Optional[list[str]] = None,  # fixed region list for consistent dummies
    ):
        if feature_cols is None:
            skip = {time_col, "time_local", region_col, "DATETIME_LOCAL_OFFSET",
                    "holiday_name", "year"}
            feature_cols = [
                c for c in df.columns
                if c not in skip and pd.api.types.is_numeric_dtype(df[c])
            ]
        self.feature_cols = feature_cols
        self.lookback_hours = lookback_hours
        self.demand_col = demand_col
        self.all_regions = all_regions  # stored for reuse in predict

        self.X_list: list[np.ndarray] = []
        self.y_list: list[np.ndarray] = []
        self.meta: list[dict] = []
        self.region_dummy_cols: list[str] = []  # set during _build

        self._build(df, time_col, region_col)

    def _build(self, df: pd.DataFrame, time_col: str, region_col: str) -> None:
        # One-hot encode region using fixed region list if provided (for train/test alignment)
        if self.all_regions is not None:
            # Build dummy scaffold from the fixed list
            scaffold = pd.DataFrame(
                {f"region_{r}": pd.Series(dtype=np.float32) for r in sorted(self.all_regions)}
            )
            dummy_cols = list(scaffold.columns)
        else:
            region_dummies = pd.get_dummies(df[region_col], prefix="region")
            dummy_cols = list(region_dummies.columns)

        self.region_dummy_cols = dummy_cols

        for region in df[region_col].unique():
            rdf = df[df[region_col] == region].copy().sort_values(time_col)
            rdf = rdf.reset_index(drop=True)

            if "time_local" in rdf.columns:
                local_dt = pd.to_datetime(rdf["time_local"])
            else:
                local_dt = pd.to_datetime(rdf[time_col])

            rdf["_local_hour"] = local_dt.dt.hour
            rdf["_local_date"] = local_dt.dt.date

            # Build region dummy row for this region, aligned to dummy_cols
            rdummies = pd.get_dummies(rdf[region_col], prefix="region")
            rdummies = rdummies.reindex(columns=dummy_cols, fill_value=0)
            rdf = pd.concat([rdf.reset_index(drop=True), rdummies.reset_index(drop=True)], axis=1)

            # Feature matrix for this region
            feat_cols_full = self.feature_cols + dummy_cols
            feat_matrix = rdf[feat_cols_full].copy()
            feat_matrix = feat_matrix.fillna(0)
            feat_np = feat_matrix.values.astype(np.float32)

            # Demand series for target construction
            demand_arr = rdf[self.demand_col].fillna(0).values.astype(np.float32)

            # Issue rows: local hour == 23
            issue_indices = rdf.index[rdf["_local_hour"] == 23].tolist()

            for idx in issue_indices:
                # Lookback window
                start_idx = idx - self.lookback_hours + 1
                if start_idx < 0:
                    continue  # not enough history

                X_seq = feat_np[start_idx : idx + 1]  # (lookback_hours, n_features)
                assert X_seq.shape[0] == self.lookback_hours

                # Target: demand for hours 0–23 of the next day
                # Find next-day date
                issue_date = rdf.loc[idx, "_local_date"]
                next_date = (pd.Timestamp(issue_date) + pd.Timedelta(days=1)).date()
                next_day_mask = (rdf["_local_date"] == next_date)

                if next_day_mask.sum() < FORECAST_HOURS:
                    continue  # incomplete target day

                next_day_rows = rdf[next_day_mask].sort_values("_local_hour")
                if len(next_day_rows) < FORECAST_HOURS:
                    continue

                y = demand_arr[next_day_rows.index[:FORECAST_HOURS]]

                self.X_list.append(X_seq)
                self.y_list.append(y)
                self.meta.append(
                    {
                        "region": region,
                        "forecast_date": next_date,
                        "issue_idx": idx,
                    }
                )

        logger.info(
            "DemandSequenceDataset: %d samples built from %d regions.",
            len(self.X_list),
            df[region_col].nunique(),
        )

    def __len__(self) -> int:
        return len(self.X_list)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.X_list[idx], self.y_list[idx]


if TORCH_AVAILABLE:

    class _TorchDataset(Dataset):
        def __init__(self, ds: DemandSequenceDataset):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            X, y = self.ds[idx]
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    # ---------------------------------------------------------------------------
    # LSTM model
    # ---------------------------------------------------------------------------

    class LSTMModel(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.1,
            horizon: int = 24,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_size, horizon)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (batch, seq_len, input_size)
            _, (h_n, _) = self.lstm(x)
            # h_n: (num_layers, batch, hidden_size) — take last layer
            last_hidden = h_n[-1]  # (batch, hidden_size)
            return self.fc(last_hidden)  # (batch, horizon)


# ---------------------------------------------------------------------------
# Forecaster wrapper
# ---------------------------------------------------------------------------

class LSTMForecaster:
    name = "lstm"

    def __init__(self, config: dict):
        if not TORCH_AVAILABLE:
            raise ImportError("Install torch:  pip install torch")
        self.config = config
        cfg = config.get("lstm", {})
        self.hidden_size = cfg.get("hidden_size", 128)
        self.num_layers = cfg.get("num_layers", 2)
        self.dropout = cfg.get("dropout", 0.1)
        self.lookback_hours = cfg.get("lookback_hours", 168)
        self.batch_size = cfg.get("batch_size", 64)
        self.epochs = cfg.get("epochs", 100)
        self.lr = cfg.get("learning_rate", 0.001)
        self.patience = cfg.get("patience", 15)
        self.seed = cfg.get("random_seed", 42)

        self.model: Optional[LSTMModel] = None
        self.train_dataset: Optional[DemandSequenceDataset] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_dir: Optional[str | Path] = None,
    ) -> "LSTMForecaster":
        self._set_seed()
        logger.info("Building LSTM training dataset (lookback=%dh)...", self.lookback_hours)

        # Fix all_regions to training regions so val/test dummies align
        all_regions = sorted(train_df["region"].unique().tolist())

        self.train_dataset = DemandSequenceDataset(
            train_df, lookback_hours=self.lookback_hours, all_regions=all_regions
        )
        val_dataset = DemandSequenceDataset(
            val_df, lookback_hours=self.lookback_hours, all_regions=all_regions
        )
        self._all_regions = all_regions
        self._feature_cols = self.train_dataset.feature_cols
        self._region_dummy_cols = self.train_dataset.region_dummy_cols

        if len(self.train_dataset) == 0:
            logger.error("LSTM: no training samples built.")
            return self

        input_size = self.train_dataset.X_list[0].shape[1]
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon=FORECAST_HOURS,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.patience // 3, factor=0.5
        )

        train_loader = DataLoader(
            _TorchDataset(self.train_dataset),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            _TorchDataset(val_dataset),
            batch_size=self.batch_size,
            shuffle=False,
        ) if len(val_dataset) > 0 else None

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            avg_train = np.mean(train_losses)

            if val_loader:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for X_v, y_v in val_loader:
                        X_v, y_v = X_v.to(self.device), y_v.to(self.device)
                        p = self.model(X_v)
                        val_losses.append(criterion(p, y_v).item())
                avg_val = np.mean(val_losses)
                scheduler.step(avg_val)

                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if epoch % 10 == 0 or patience_counter == 0:
                    logger.info(
                        "LSTM Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                        epoch + 1, self.epochs, avg_train, avg_val,
                    )

                if patience_counter >= self.patience:
                    logger.info("LSTM: early stopping at epoch %d.", epoch + 1)
                    break
            else:
                if epoch % 10 == 0:
                    logger.info("LSTM Epoch %d/%d  train_loss=%.4f", epoch + 1, self.epochs, avg_train)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("LSTM: restored best model (val_loss=%.4f).", best_val_loss)

        if model_dir:
            self.save(model_dir)

        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")

        # Pass the training regions so test dummies align with trained model
        all_regions = getattr(self, "_all_regions", None)
        ds = DemandSequenceDataset(
            test_df, lookback_hours=self.lookback_hours, all_regions=all_regions
        )
        if len(ds) == 0:
            logger.warning("LSTM: no test samples (need %d hours lookback).", self.lookback_hours)
            return pd.DataFrame(
                columns=["region", "forecast_date", "target_hour", "predicted", "actual"]
            )

        loader = DataLoader(_TorchDataset(ds), batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                pred = self.model(X_batch).cpu().numpy()
                all_preds.append(pred)

        preds_flat = np.concatenate(all_preds, axis=0)  # (n_samples, 24)

        rows = []
        for i, meta in enumerate(ds.meta):
            for h in range(FORECAST_HOURS):
                rows.append(
                    {
                        "region": meta["region"],
                        "forecast_date": meta["forecast_date"],
                        "target_hour": h,
                        "predicted": float(preds_flat[i, h]),
                        "actual": float(ds.y_list[i][h]) if i < len(ds.y_list) else np.nan,
                    }
                )
        return pd.DataFrame(rows)

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir) / self.name
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_dir / "model.pt")
        import json as _json
        meta = {
            "all_regions": getattr(self, "_all_regions", None),
            "feature_cols": getattr(self, "_feature_cols", []),
            "region_dummy_cols": getattr(self, "_region_dummy_cols", []),
            "input_size": self.model.lstm.input_size if self.model else None,
        }
        (model_dir / "meta.json").write_text(_json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("LSTM model saved to %s.", model_dir)

    def load(self, model_dir: str | Path, input_size: int = None) -> "LSTMForecaster":
        model_dir = Path(model_dir) / self.name
        import json as _json
        meta_path = model_dir / "meta.json"
        if meta_path.exists():
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            self._all_regions = meta.get("all_regions")
            self._feature_cols = meta.get("feature_cols", [])
            self._region_dummy_cols = meta.get("region_dummy_cols", [])
            if input_size is None:
                input_size = meta.get("input_size", input_size)
        if input_size is None:
            raise ValueError("input_size required when loading LSTM without meta.json")
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon=FORECAST_HOURS,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_dir / "model.pt", map_location=self.device))
        self.model.eval()
        logger.info("LSTM model loaded from %s.", model_dir)
        return self
