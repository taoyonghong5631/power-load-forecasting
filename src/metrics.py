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


def mape(y_true, y_pred) -> float:
    """平均绝对百分比误差（%），对接近 0 的真值做截断保护。"""
    y_true, y_pred = _flat(y_true), _flat(y_pred)
    denom = np.maximum(np.abs(y_true), 1e-3 * (np.mean(np.abs(y_true)) + EPS))
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


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
        "MAPE": mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "PeakMAE": peak_mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }


METRIC_ORDER = ["MAE", "RMSE", "MAPE", "sMAPE", "PeakMAE", "R2"]


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
