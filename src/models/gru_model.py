"""GRU multi-region day-ahead electricity demand forecaster.

Identical interface to LSTMForecaster but uses a GRU encoder instead.
Shares the same DemandSequenceDataset and training loop from lstm_model.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.lstm_model import DemandSequenceDataset, FORECAST_HOURS

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class GRUModel(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            dropout: float = 0.1,
            horizon: int = 24,
        ):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_size, horizon)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            _, h_n = self.gru(x)
            last_hidden = h_n[-1]
            return self.fc(last_hidden)

    class _TorchDataset(Dataset):
        def __init__(self, ds: DemandSequenceDataset):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            X, y = self.ds[idx]
            return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class GRUForecaster:
    name = "gru"

    def __init__(self, config: dict):
        if not TORCH_AVAILABLE:
            raise ImportError("Install torch:  pip install torch")
        self.config = config
        cfg = config.get("gru", {})
        self.hidden_size = cfg.get("hidden_size", 128)
        self.num_layers = cfg.get("num_layers", 2)
        self.dropout = cfg.get("dropout", 0.1)
        self.lookback_hours = cfg.get("lookback_hours", 168)
        self.batch_size = cfg.get("batch_size", 64)
        self.epochs = cfg.get("epochs", 100)
        self.lr = cfg.get("learning_rate", 0.001)
        self.patience = cfg.get("patience", 15)
        self.seed = cfg.get("random_seed", 42)

        self.model: Optional[GRUModel] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_dir: Optional[str | Path] = None,
    ) -> "GRUForecaster":
        self._set_seed()
        logger.info("Building GRU training dataset (lookback=%dh)...", self.lookback_hours)

        all_regions = sorted(train_df["region"].unique().tolist())
        self._all_regions = all_regions

        train_dataset = DemandSequenceDataset(
            train_df, lookback_hours=self.lookback_hours, all_regions=all_regions
        )
        val_dataset = DemandSequenceDataset(
            val_df, lookback_hours=self.lookback_hours, all_regions=all_regions
        )

        if len(train_dataset) == 0:
            logger.error("GRU: no training samples built.")
            return self

        input_size = train_dataset.X_list[0].shape[1]
        self.model = GRUModel(
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

        train_loader = DataLoader(_TorchDataset(train_dataset), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(_TorchDataset(val_dataset), batch_size=self.batch_size, shuffle=False) \
            if len(val_dataset) > 0 else None

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X_b), y_b)
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
                        val_losses.append(criterion(self.model(X_v), y_v).item())
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
                        "GRU Epoch %d/%d  train=%.4f  val=%.4f",
                        epoch + 1, self.epochs, avg_train, avg_val,
                    )

                if patience_counter >= self.patience:
                    logger.info("GRU: early stopping at epoch %d.", epoch + 1)
                    break
            else:
                if epoch % 10 == 0:
                    logger.info("GRU Epoch %d/%d  train=%.4f", epoch + 1, self.epochs, avg_train)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if model_dir:
            self.save(model_dir)
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")

        all_regions = getattr(self, "_all_regions", None)
        ds = DemandSequenceDataset(
            test_df, lookback_hours=self.lookback_hours, all_regions=all_regions
        )
        if len(ds) == 0:
            return pd.DataFrame(columns=["region", "forecast_date", "target_hour", "predicted", "actual"])

        loader = DataLoader(_TorchDataset(ds), batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for X_b, _ in loader:
                all_preds.append(self.model(X_b.to(self.device)).cpu().numpy())

        preds_flat = np.concatenate(all_preds, axis=0)
        rows = []
        for i, meta in enumerate(ds.meta):
            for h in range(FORECAST_HOURS):
                rows.append({
                    "region": meta["region"],
                    "forecast_date": meta["forecast_date"],
                    "target_hour": h,
                    "predicted": float(preds_flat[i, h]),
                    "actual": float(ds.y_list[i][h]),
                })
        return pd.DataFrame(rows)

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir) / self.name
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_dir / "model.pt")
        import json as _json
        meta = {
            "all_regions": getattr(self, "_all_regions", None),
            "input_size": self.model.gru.input_size if self.model else None,
        }
        (model_dir / "meta.json").write_text(_json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("GRU model saved to %s.", model_dir)

    def load(self, model_dir: str | Path, input_size: int = None) -> "GRUForecaster":
        model_dir = Path(model_dir) / self.name
        import json as _json
        meta_path = model_dir / "meta.json"
        if meta_path.exists():
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            self._all_regions = meta.get("all_regions")
            if input_size is None:
                input_size = meta.get("input_size", input_size)
        if input_size is None:
            raise ValueError("input_size required when loading GRU without meta.json")
        self.model = GRUModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon=FORECAST_HOURS,
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_dir / "model.pt", map_location=self.device))
        self.model.eval()
        return self
