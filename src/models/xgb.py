# -*- coding: utf-8 -*-
"""XGBoost 对比模型：滞后特征 + 日期特征 + 温度特征，递归多步预测。"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ..config import Config
from ..features import build_supervised, next_step_row
from . import BaseForecaster


class XGBForecaster(BaseForecaster):
    needs_refit_per_origin = False

    def __init__(self, cfg: Config, groups: Iterable[str] = ("base",),
                 name: str = "XGBoost"):
        self.cfg = cfg
        self.name = name
        self.groups = tuple(groups)
        self.model: Optional[XGBRegressor] = None
        self.cols: List[str] = []

    def fit(self, train_df: pd.DataFrame, verbose: bool = True) -> "XGBForecaster":
        xcfg = self.cfg.xgb
        X, y, cols = build_supervised(train_df, self.groups,
                                       include_annual=self.cfg.include_annual)
        self.cols = cols
        self.model = XGBRegressor(
            n_estimators=xcfg.n_estimators,
            max_depth=xcfg.max_depth,
            learning_rate=xcfg.learning_rate,
            subsample=xcfg.subsample,
            colsample_bytree=xcfg.colsample_bytree,
            reg_lambda=xcfg.reg_lambda,
            min_child_weight=xcfg.min_child_weight,
            tree_method=xcfg.tree_method,
            n_jobs=xcfg.n_jobs,
            random_state=self.cfg.task.seed,
            objective="reg:squarederror",
        )
        self.model.fit(X.values, y.values, verbose=False)
        if verbose:
            print("  [%s] 训练完成：%d 个样本 × %d 个特征"
                  % (self.name, len(X), len(cols)))
        return self

    def predict(self, df: pd.DataFrame, origin: int, horizon: int) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("请先调用 fit()")
        ctx_cols = ["load"] + (["temp"] if "temp" in df.columns else [])
        ctx = df.iloc[:origin + 1][ctx_cols].copy()

        preds: List[float] = []
        for step in range(horizon):
            ts = df.index[origin + 1 + step]
            temp = float(df["temp"].loc[ts]) if "temp" in df.columns else None
            row = next_step_row(ctx, self.groups, ts, temp=temp,
                                include_annual=self.cfg.include_annual)
            value = float(self.model.predict(row[self.cols].to_numpy(dtype=float))[0])
            preds.append(value)

            new_row = pd.DataFrame({"load": [value]}, index=pd.DatetimeIndex([ts]))
            if "temp" in ctx.columns:
                new_row["temp"] = np.nan if temp is None else temp
            ctx = pd.concat([ctx, new_row])
        return np.asarray(preds, dtype=float)

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("请先调用 fit()")
        imp = pd.DataFrame({
            "feature": self.cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return imp.head(top_n).reset_index(drop=True)
