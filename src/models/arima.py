# -*- coding: utf-8 -*-
"""ARIMA/SARIMA 基线：季节性差分捕捉日周期，每个预测起点用末尾窗口重新拟合。

为什么每个起点都要重拟合：ARIMA 是纯统计模型，参数强依赖最近的均值/方差水平，
固定参数跑完整段测试期会明显不公平（也失去它"能在线滚动更新"的优势）。
因此这里采用业界标准的 walk-forward 用法，并在 README 里写明与树模型的差异。
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from . import BaseForecaster


class ARIMAForecaster(BaseForecaster):
    needs_refit_per_origin = True

    def __init__(self, cfg: Config, name: str = "ARIMA",
                 use_temperature: Optional[bool] = None):
        self.cfg = cfg
        self.name = name
        self.acfg = cfg.arima
        self.use_temperature = (self.acfg.use_temperature_exog
                                if use_temperature is None else use_temperature)
        self._cache: Dict[int, object] = {}
        self.aic: List[float] = []
        self.n_fallback = 0

    # ---------------- 拟合 ----------------
    def fit(self, train_df: pd.DataFrame, verbose: bool = True) -> "ARIMAForecaster":
        # ARIMA 在 predict 里按起点重拟合，这里只保存训练段末端信息
        self.train_end = train_df.index[-1]
        self.train_tail = train_df.iloc[-self.acfg.train_window:].copy()
        if verbose:
            print("  [%s] walk-forward 模式：order=%s seasonal=%s window=%d"
                  % (self.name, self.acfg.order, self.acfg.seasonal_order,
                     self.acfg.train_window))
        return self

    def _window(self, df: pd.DataFrame, origin: int) -> pd.DataFrame:
        lo = max(0, origin + 1 - self.acfg.train_window)
        return df.iloc[lo:origin + 1]

    def _fit_at(self, df: pd.DataFrame, origin: int):
        if origin in self._cache:
            return self._cache[origin]
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        win = self._window(df, origin)
        y = win["load"].to_numpy(dtype=float)
        exog = win[["temp"]].to_numpy(dtype=float) if self.use_temperature else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y, exog=exog,
                order=self.acfg.order,
                seasonal_order=self.acfg.seasonal_order,
                enforce_stationarity=self.acfg.enforce_stationarity,
                enforce_invertibility=self.acfg.enforce_invertibility,
                trend="n",
            )
            res = model.fit(disp=False, maxiter=50)
        self._cache[origin] = res
        self.aic.append(float(res.aic))
        # SARIMAX 的 results 对象非常大（每个都带着完整数据和状态空间矩阵），
        # 只保留最近一个就够了，否则 30 个起点会把内存和磁盘都撑爆
        for old in [k for k in self._cache if k != origin]:
            self._cache.pop(old, None)
        return res

    def __getstate__(self) -> dict:
        """序列化时丢掉拟合结果缓存。

        踩过的坑：把 30 个 SARIMAX results 一起 joblib.dump 会生成 7 GB 的模型文件
        （本项目真的把磁盘写满过一次）。拟合结果本来也不需要持久化——
        换一个起点重新拟合即可。
        """
        state = self.__dict__.copy()
        state["_cache"] = {}
        state["train_tail"] = None
        return state

    def predict(self, df: pd.DataFrame, origin: int, horizon: int) -> np.ndarray:
        try:
            res = self._fit_at(df, origin)
        except Exception as exc:
            # 个别窗口不收敛时退回"季节性朴素预测"（用上一周期同时刻的值），
            # 保证对比表不会因为一个窗口失败而整列缺失
            self.n_fallback += 1
            print("    [%s] 起点 %s 拟合失败(%s)，退回季节朴素预测"
                  % (self.name, df.index[origin], type(exc).__name__))
            return self._seasonal_naive(df, origin, horizon)
        exog_future = None
        if self.use_temperature:
            future_idx = df.index[origin + 1:origin + 1 + horizon]
            exog_future = df.loc[future_idx, ["temp"]].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = res.forecast(steps=horizon, exog=exog_future)
        out = np.asarray(fc, dtype=float).ravel()
        if not np.isfinite(out).all():
            self.n_fallback += 1
            return self._seasonal_naive(df, origin, horizon)
        return out

    def _seasonal_naive(self, df: pd.DataFrame, origin: int, horizon: int) -> np.ndarray:
        """季节性朴素：直接用 24 小时前的值（ARIMA 失败时的安全兜底）。"""
        load = df["load"].to_numpy(dtype=float)
        period = 24
        preds = []
        for step in range(horizon):
            idx = origin + 1 + step - period
            preds.append(load[idx] if idx >= 0 else load[origin])
        return np.asarray(preds, dtype=float)


def select_order(train: pd.Series, cfg: Config, max_p: int = 2, max_q: int = 2,
                 verbose: bool = True) -> Tuple[tuple, tuple]:
    """小网格 AIC 选阶（在末尾一小段上做，避免太慢）。"""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = train.iloc[-min(len(train), 1000):].to_numpy(dtype=float)
    best = (None, None, np.inf)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for P, Q in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = SARIMAX(y, order=(p, 1, q), seasonal_order=(P, 1, Q, 24),
                                      trend="n", enforce_stationarity=False,
                                      enforce_invertibility=False).fit(disp=False, maxiter=50)
                    if res.aic < best[2]:
                        best = ((p, 1, q), (P, 1, Q, 24), float(res.aic))
                except Exception:
                    continue
    if verbose:
        print("[arima] AIC 选阶结果: order=%s seasonal=%s (AIC=%.1f)" % best[:3])
    return best[0], best[1]
