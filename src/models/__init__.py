# -*- coding: utf-8 -*-
"""预测模型：统一接口 ``fit(train_df)`` + ``predict(df, origin, horizon)``。

三个模型都遵守同一个评估契约：
* ``origin`` 是"最后一个已知观测点"在 df 中的位置（整数下标）；
* ``predict`` 返回 ``origin+1 ... origin+horizon`` 这 horizon 个点的预测值；
* 单位与 df['load'] 一致（kW），不需要调用方再反标准化。

这样滚动起点评估时可以拿同一套 index 喂给所有模型，指标才可比。
"""
from __future__ import annotations

from typing import Iterable, Tuple

from ..config import Config


class BaseForecaster:
    """所有模型的公共接口。"""

    name = "model"
    needs_refit_per_origin = False   # ARIMA 会置 True（每个起点重新拟合）

    def fit(self, train_df, **kwargs) -> "BaseForecaster":
        raise NotImplementedError

    def predict(self, df, origin: int, horizon: int):
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


def build_model(name: str, cfg: Config, groups: Iterable[str],
                label: str | None = None) -> BaseForecaster:
    """按名字构造模型（延迟导入，避免只用 XGB 时还去加载 torch）。"""
    key = name.lower()
    groups = tuple(groups)
    if key in ("lstm", "rnn"):
        from .lstm import LSTMForecaster
        return LSTMForecaster(cfg, groups=groups, name=label or "LSTM")
    if key in ("xgb", "xgboost", "xgb_direct", "xgb_recursive"):
        from .xgb import XGBForecaster
        strat = {"xgb_direct": "direct", "xgb_recursive": "recursive"}.get(key)
        default_label = {"xgb_direct": "XGBoost (直接多步)",
                         "xgb_recursive": "XGBoost (递归)"}.get(key, "XGBoost")
        return XGBForecaster(cfg, groups=groups, name=label or default_label,
                             strategy=strat)
    if key in ("arima", "sarima", "sarimax"):
        from .arima import ARIMAForecaster
        return ARIMAForecaster(cfg, name=label or "ARIMA")
    if key in ("naive", "persistence"):
        from .naive import NaiveForecaster
        return NaiveForecaster(cfg, season=1, name=label or "持久性 (t-1)")
    if key in ("seasonal_naive", "snaive", "naive24"):
        from .naive import NaiveForecaster
        return NaiveForecaster(cfg, season=24, name=label or "季节朴素 (t-24)")
    raise KeyError("未知模型: %s（可选 naive / seasonal_naive / lstm / xgb / arima）" % name)
