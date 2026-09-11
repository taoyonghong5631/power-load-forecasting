# -*- coding: utf-8 -*-
"""统一的滚动起点（rolling-origin）评估协议。

所有模型都在**完全相同的预测窗口**上打分：先在训练段末端确定一批起点，
每个起点向后预测 horizon 小时，再把这些窗口的指标平均。这样 LSTM / XGBoost /
ARIMA 的对比表才是可比的。
"""
from __future__ import annotations

import time
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from . import metrics as M
from .config import Config


def train_end_index(n_total: int, train_ratio: float) -> int:
    """训练段最后一个样本的下标（含）。"""
    return int(n_total * train_ratio) - 1


def make_origins(n_total: int, train_end: int, horizon: int,
                 stride: int = 24, n_origins: int = 30) -> List[int]:
    """在测试段上均匀取一批预测起点。

    起点 o 表示"已知到 o 时刻"，要预测 o+1 ... o+horizon。
    """
    last_valid = n_total - 1 - horizon          # 保证预测窗口不越界
    candidates = list(range(train_end, last_valid + 1, stride))
    if not candidates:
        raise ValueError("测试段太短，无法构造预测起点")
    if len(candidates) <= n_origins:
        return candidates
    pos = np.linspace(0, len(candidates) - 1, n_origins)
    return sorted({candidates[int(round(p))] for p in pos})


def evaluate_model(model, df: pd.DataFrame, origins: Sequence[int],
                   horizon: int, verbose: bool = True) -> Dict[str, object]:
    """跑完所有起点，返回指标与逐窗口预测结果。"""
    load = df["load"].to_numpy(dtype=float)
    per_origin: List[Dict[str, float]] = []
    preds = np.full((len(origins), horizon), np.nan)
    truths = np.full((len(origins), horizon), np.nan)
    stamps: List[pd.DatetimeIndex] = []

    t_pred = 0.0
    for i, origin in enumerate(origins):
        t0 = time.perf_counter()
        # 注意：ARIMA 的拟合发生在 predict 内部，所以这里的耗时包含了重拟合时间
        pred = model.predict(df, origin, horizon)
        t_pred += time.perf_counter() - t0

        true = load[origin + 1:origin + 1 + horizon]
        preds[i, :len(pred)] = pred
        truths[i, :len(true)] = true
        stamps.append(df.index[origin + 1:origin + 1 + horizon])
        per_origin.append(M.forecast_metrics(true, pred))
        if verbose:
            print("    起点 %s  MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%"
                  % (df.index[origin], per_origin[-1]["MAE"],
                     per_origin[-1]["RMSE"], per_origin[-1]["MAPE"]))

    agg = M.aggregate(per_origin)
    # 池化指标：把所有窗口的点拼起来算一次，反映整体误差水平
    pooled = M.forecast_metrics(truths.ravel(), preds.ravel())
    agg.update({"pooled_" + k: v for k, v in pooled.items()})
    agg["n_origins"] = len(origins)
    agg["predict_seconds"] = t_pred
    return {
        "name": getattr(model, "name", "model"),
        "metrics": agg,
        "per_origin": per_origin,
        "preds": preds,
        "truths": truths,
        "timestamps": stamps,
        "origins": list(origins),
    }


def results_to_table(results: Dict[str, Dict[str, object]],
                     baseline: str | None = None) -> pd.DataFrame:
    """把多个模型的结果整理成对比表。"""
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        rows.append({
            "模型": name,
            "MAE": m.get("MAE"),
            "RMSE": m.get("RMSE"),
            "MAPE(%)": m.get("MAPE"),
            "sMAPE(%)": m.get("sMAPE"),
            "尖峰MAE": m.get("PeakMAE"),
            "R2": m.get("R2"),
            "窗口数": m.get("n_origins"),
            "耗时(s)": m.get("predict_seconds"),
        })
    table = pd.DataFrame(rows).set_index("模型")
    if baseline and baseline in table.index:
        base = table.loc[baseline]
        for col in ["MAE", "RMSE", "MAPE(%)", "sMAPE(%)", "尖峰MAE"]:
            table[col + "↓%"] = (base[col] - table[col]) / abs(base[col]) * 100.0
    return table


def ablation_table(results: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    """特征消融表：按插入顺序计算每一级的相对提升。"""
    table = results_to_table(results)
    table["MAE提升"] = np.nan
    names = list(results.keys())
    for prev, cur in zip(names[:-1], names[1:]):
        base = results[prev]["metrics"]["MAE"]
        new = results[cur]["metrics"]["MAE"]
        table.loc[cur, "MAE提升"] = M.improvement(base, new)
    table.loc[names[0], "MAE提升"] = 0.0
    return table
