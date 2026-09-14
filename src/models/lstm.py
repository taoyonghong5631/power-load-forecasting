# -*- coding: utf-8 -*-
"""LSTM 递归多步预测：支持时间序验证集 + 早停，以及可选的外生变量通道。"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..features import build_exog, exog_columns
from . import BaseForecaster


class _WindowDataset(Dataset):
    """把 (载荷, 外生) 序列切成 (batch, look_back, n_features) 的监督窗口。"""

    def __init__(self, features: np.ndarray, target: np.ndarray, look_back: int,
                 index: np.ndarray):
        self.features = features
        self.target = target
        self.look_back = look_back
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        end = self.index[i]
        x = self.features[end - self.look_back:end]
        y = self.target[end]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMForecaster(BaseForecaster):
    needs_refit_per_origin = False

    def __init__(self, cfg: Config, groups: Iterable[str] = ("base",),
                 name: str = "LSTM"):
        self.cfg = cfg
        self.name = name
        # 对 LSTM 来说 "base" 就是序列本身的滞后；外生组只有 calendar / temperature
        self.groups = tuple(g for g in groups if g != "base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[_LSTMNet] = None
        self.history: List[float] = []
        self.exog_cols: List[str] = []

    # ---------------- 内部工具 ----------------
    def _exog_frame(self, df: pd.DataFrame) -> np.ndarray:
        if not self.exog_cols:
            return np.empty((len(df), 0), dtype=float)
        exog = build_exog(df, self.groups, include_annual=self.cfg.include_annual)
        return exog[self.exog_cols].to_numpy(dtype=float)

    def _encode(self, load: np.ndarray, exog: np.ndarray) -> np.ndarray:
        """把 (载荷, 外生) 拼成模型输入矩阵 (n, 1+n_exog)。"""
        scaled_load = self.scaler_level.transform(load.reshape(-1, 1))
        if exog.shape[1] == 0:
            return scaled_load
        return np.concatenate([scaled_load, self.scaler_x.transform(exog)], axis=1)

    # ---------------- 训练 ----------------
    def fit(self, train_df: pd.DataFrame, verbose: bool = True) -> "LSTMForecaster":
        from sklearn.preprocessing import StandardScaler

        lcfg = self.cfg.lstm
        torch.manual_seed(self.cfg.task.seed)
        np.random.seed(self.cfg.task.seed)

        self.exog_cols = exog_columns(self.groups,
                                      include_annual=self.cfg.include_annual)
        load = train_df["load"].to_numpy(dtype=float)
        exog = self._exog_frame(train_df)
        self.target_mode = self.cfg.lstm.target_mode

        # 输入通道永远用"水平值"的标准化；预测目标可以是水平值或增量
        self.scaler_level = StandardScaler().fit(load.reshape(-1, 1))
        self.scaler_x = StandardScaler().fit(exog) if exog.shape[1] else None
        features = self._encode(load, exog)

        if self.target_mode == "delta":
            raw = np.diff(load, prepend=np.nan)
            ok = ~np.isnan(raw)
            self.scaler_y = StandardScaler().fit(raw[ok].reshape(-1, 1))
            target = np.full(len(load), np.nan)
            target[ok] = self.scaler_y.transform(raw[ok].reshape(-1, 1)).ravel()
        else:
            self.scaler_y = self.scaler_level
            target = self.scaler_y.transform(load.reshape(-1, 1)).ravel()

        look_back = self.cfg.task.look_back
        idx = np.arange(look_back, len(target))
        # 验证集必须按时间顺序切（不能随机打乱），否则相邻窗口互相泄漏
        n_val = max(24, int(len(idx) * lcfg.val_ratio))
        train_idx, val_idx = idx[:-n_val], idx[-n_val:]

        train_loader = DataLoader(
            _WindowDataset(features, target, look_back, train_idx),
            batch_size=lcfg.batch_size, shuffle=True)
        val_loader = DataLoader(
            _WindowDataset(features, target, look_back, val_idx),
            batch_size=lcfg.batch_size, shuffle=False)

        input_size = features.shape[1]
        self.model = _LSTMNet(input_size, lcfg.hidden_size, lcfg.num_layers,
                              lcfg.dropout).to(self.device)
        optim = torch.optim.Adam(self.model.parameters(), lr=lcfg.learning_rate)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, factor=0.5)
        criterion = nn.MSELoss()

        best_val, best_state, bad_epochs = float("inf"), None, 0
        self.history = []
        for epoch in range(lcfg.epochs):
            self.model.train()
            total = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optim.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optim.step()
                total += loss.item() * len(xb)
            train_loss = total / len(train_loader.dataset)

            self.model.eval()
            val_total = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    val_total += criterion(self.model(xb), yb).item() * len(xb)
            val_loss = val_total / len(val_loader.dataset)
            self.history.append({"epoch": epoch + 1, "train": train_loss, "val": val_loss})
            sched.step(val_loss)

            if val_loss < best_val - 1e-6:
                best_val, bad_epochs = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad_epochs += 1

            if verbose and (epoch + 1) % 5 == 0:
                print("  [%s] epoch %3d/%d  train %.5f  val %.5f"
                      % (self.name, epoch + 1, lcfg.epochs, train_loss, val_loss),
                      flush=True)
            if bad_epochs >= lcfg.patience:
                if verbose:
                    print("  [%s] 验证集 %d 轮无改善，提前停止于 epoch %d"
                          % (self.name, lcfg.patience, epoch + 1))
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    # ---------------- 递归多步预测 ----------------
    def predict(self, df: pd.DataFrame, origin: int, horizon: int) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("请先调用 fit()")
        look_back = self.cfg.task.look_back
        start = origin + 1 - look_back
        if start < 0:
            raise ValueError("历史长度不足 look_back=%d" % look_back)

        past_idx = df.index[start:origin + 1]
        past_load = df["load"].iloc[start:origin + 1].to_numpy(dtype=float)
        past_exog = self._exog_frame(df.iloc[start:origin + 1])

        future_idx = df.index[origin + 1:origin + 1 + horizon]
        future = pd.DataFrame(index=future_idx)
        if "temp" in df.columns:
            future["temp"] = df["temp"].reindex(future_idx).to_numpy()
        future_exog = self._exog_frame(future) if self.exog_cols else np.empty((horizon, 0))

        # 逐行特征：历史行 (载荷已知, 外生已知)，未来行 (载荷待填, 外生已知)
        past_scaled = self.scaler_level.transform(past_load.reshape(-1, 1)).ravel()
        exog_past = (self.scaler_x.transform(past_exog) if self.exog_cols
                     else np.empty((len(past_load), 0)))
        exog_fut = (self.scaler_x.transform(future_exog) if self.exog_cols
                    else np.empty((horizon, 0)))
        rows = [np.concatenate([[v], e]) for v, e in zip(past_scaled, exog_past)]
        rows += [np.concatenate([[np.nan], e]) for e in exog_fut]

        preds: List[float] = []
        cur_level = float(past_load[-1])
        self.model.eval()
        with torch.no_grad():
            for step in range(horizon):
                # 第 step 步的输入窗口 [step, step+look_back) 右端正好是 t-1，
                # 该行及以前的载荷都已填好，窗口内不会出现 NaN
                x = np.array(rows[step:step + look_back], dtype=float)
                xb = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
                pred_scaled = float(self.model(xb).cpu().item())
                if self.target_mode == "delta":
                    delta = float(self.scaler_y.inverse_transform([[pred_scaled]])[0, 0])
                    cur_level = cur_level + delta
                else:
                    cur_level = float(self.scaler_y.inverse_transform([[pred_scaled]])[0, 0])
                preds.append(cur_level)
                if step + look_back < len(rows):
                    rows[step + look_back][0] = float(
                        self.scaler_level.transform([[cur_level]])[0, 0])

        return np.asarray(preds, dtype=float)
