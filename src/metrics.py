# -*- coding: utf-8 -*-
"""评估指标。所有指标都做了除零保护，负荷数据里 0 值很常见。"""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

EPS = 1e-8


def _flat(x: Iterable[float]) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _flat(y_true), _flat(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _flat(y_true), _flat(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, floor_ratio: float = 0.1) -> float:
    """平均绝对百分比误差（%）。

    负荷数据里接近 0 的点很常见（本项目 MT_001 的测试段有 23% 的点 < 1 kW，
    最低 0.32 kW），直接除以真值会让单点误差贡献几百个百分点，把整体 MAPE 拉爆。
    这里给分母加一个"平均负荷的 10%"下限（业界常见的做法），并把 WAPE 作为
    更稳健的百分比指标一起报告。
    """
    y_true, y_pred = _flat(y_true), _flat(y_pred)
    floor = floor_ratio * (np.mean(np.abs(y_true)) + EPS)
    denom = np.maximum(np.abs(y_true), floor)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def wape(y_true, y_pred) -> float:
    """加权绝对百分比误差 = Σ|误差| / Σ|真值|，对近零值不敏感，负荷预测常用。"""
    y_true, y_pred = _flat(y_true), _flat(y_pred)
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + EPS) * 100)


def smape(y_true, y_pred) -> float:
    """对称 MAPE（%），真值很小时比 MAPE 稳定。"""
    y_true, y_pred = _flat(y_true), _flat(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(denom, EPS)) * 100)


def r2(y_true, y_pred) -> float:
    y_true, y_pred = _flat(y_true), _flat(y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / (ss_tot + EPS))


def peak_mae(y_true, y_pred, top_frac: float = 0.15) -> float:
    """尖峰时段 MAE：只统计真值最高的 top_frac 比例的点（电力调度最关心）。"""
    y_true, y_pred = _flat(y_true), _flat(y_pred)
    k = max(1, int(round(len(y_true) * top_frac)))
    idx = np.argsort(y_true)[-k:]
    return float(np.mean(np.abs(y_true[idx] - y_pred[idx])))


def forecast_metrics(y_true, y_pred) -> Dict[str, float]:
    """单窗口的全部指标。"""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "PeakMAE": peak_mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }


METRIC_ORDER = ["MAE", "RMSE", "WAPE", "MAPE", "sMAPE", "PeakMAE", "R2"]


def aggregate(per_origin: List[Dict[str, float]]) -> Dict[str, float]:
    """把多个起点的指标聚合成均值（标准差单独放在 *_std 键里）。"""
    out: Dict[str, float] = {}
    if not per_origin:
        return out
    for key in METRIC_ORDER:
        vals = np.array([m[key] for m in per_origin if key in m], dtype=float)
        if vals.size:
            out[key] = float(vals.mean())
            out[key + "_std"] = float(vals.std())
    return out


def improvement(baseline: float, new: float) -> float:
    """相对提升百分比（正数表示误差下降）。"""
    if abs(baseline) < EPS:
        return 0.0
    return (baseline - new) / abs(baseline) * 100.0
