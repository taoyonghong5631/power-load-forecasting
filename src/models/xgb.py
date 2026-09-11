# -*- coding: utf-8 -*-
"""XGBoost 对比模型：滞后/日期/温度特征，支持递归多步与直接多步两种策略。"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ..config import Config
from ..features import (build_exog, build_features, build_supervised, exog_columns,
                        feature_names, next_step_row)
from . import BaseForecaster


class XGBForecaster(BaseForecaster):
    needs_refit_per_origin = False

    def __init__(self, cfg: Config, groups: Iterable[str] = ("base",),
                 name: str = "XGBoost", strategy: Optional[str] = None):
        self.cfg = cfg
        self.name = name
        self.groups = tuple(groups)
        self.model: Optional[XGBRegressor] = None
        self.models: Dict[int, XGBRegressor] = {}
        self.cols: List[str] = []
        self.cols_t: List[str] = []
        self.fwd_cols: List[str] = []
        self.target_mode = cfg.xgb.target_mode
        self.strategy = strategy or cfg.xgb.strategy

    # ---------------- 构造底层模型 ----------------
    def _new_model(self) -> XGBRegressor:
        xcfg = self.cfg.xgb
        return XGBRegressor(
            n_estimators=xcfg.max_rounds if xcfg.use_early_stopping else xcfg.n_estimators,
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
            early_stopping_rounds=(xcfg.early_stopping_rounds
                                   if xcfg.use_early_stopping else None),
        )

    def _fit_one(self, X: np.ndarray, y: np.ndarray) -> XGBRegressor:
        """按时间顺序切验证集做早停。

        增量目标的信噪比很低（大部分时刻 Δy=0），不给早停的话 600 棵树会把噪声
        一起学进去，预测反而比"什么都不做"更差。
        """
        n_val = max(48, int(len(X) * self.cfg.xgb.val_ratio))
        mdl = self._new_model()
        if not self.cfg.xgb.use_early_stopping:
            mdl.fit(X, y, verbose=False)
            return mdl
        mdl.fit(X[:-n_val], y[:-n_val],
                eval_set=[(X[-n_val:], y[-n_val:])], verbose=False)
        return mdl

    def fit(self, train_df: pd.DataFrame, verbose: bool = True) -> "XGBForecaster":
        self.target_mode = self.cfg.xgb.target_mode
        self.strategy = self.strategy or self.cfg.xgb.strategy
        if self.strategy == "direct":
            return self._fit_direct(train_df, verbose)
        return self._fit_recursive(train_df, verbose)

    def _fit_recursive(self, train_df: pd.DataFrame,
                       verbose: bool = True) -> "XGBForecaster":
        X, y, cols = build_supervised(train_df, self.groups,
                                      include_annual=self.cfg.include_annual,
                                      target_mode=self.target_mode)
        self.cols = cols
        self.model = self._fit_one(X.values, y.values)
        best_iter = getattr(self.model, "best_iteration", None)
        if verbose:
            print("  [%s] 递归模式训练完成：%d 个样本 × %d 个特征，目标=%s，实际树数=%s"
                  % (self.name, len(X), len(cols),
                     "增量 Δy" if self.target_mode == "delta" else "绝对值",
                     best_iter + 1 if best_iter is not None else "?"))
        return self

    def _fit_direct(self, train_df: pd.DataFrame,
                    verbose: bool = True) -> "XGBForecaster":
        """直接多步：第 h 步预测 y_{t+h} - y_t，特征 = 起点特征 + 目标时刻的日历/温度。

        y_{t+h} - y_t 的基准永远是**起点真实值**，所以 24 步之间互不影响，
        不会像递归那样把预测误差一路滚成 24 倍。
        """
        horizon = self.cfg.task.horizon
        feats = build_features(train_df, self.groups)
        self.cols_t = feature_names(self.groups, include_annual=self.cfg.include_annual)
        # 目标时刻可提前知道的外生量：日历 + 温度（预报值）
        want_fwd = [g for g in ("calendar", "temperature") if g in self.groups]
        self.fwd_cols = exog_columns(want_fwd, include_annual=self.cfg.include_annual) \
            if want_fwd else []
        self.fwd_cols = [c for c in self.fwd_cols if c in feats.columns]

        base = feats[self.cols_t].copy()
        base["__load__"] = feats["load"]
        self.models = {}
        for h in range(1, horizon + 1):
            block = base.copy()
            for c in self.fwd_cols:
                block["tgt_" + c] = feats[c].shift(-h)
            block["__y__"] = feats["load"].shift(-h) - feats["load"]
            data = block.dropna()
            # 注意排除辅助列 __load__，否则特征数与预测时会不一致
            cols = self.cols_t + ["tgt_" + c for c in self.fwd_cols]
            mdl = self._fit_one(data[cols].values, data["__y__"].values)
            self.models[h] = mdl
        self.cols = self.cols_t + ["tgt_" + c for c in self.fwd_cols]
        if verbose:
            print("  [%s] 直接多步训练完成：%d 个 horizon 模型 × %d 个特征"
                  % (self.name, len(self.models), len(self.cols)))
        return self

    def predict(self, df: pd.DataFrame, origin: int, horizon: int) -> np.ndarray:
        if self.model is None:
            if not self.models:
                raise RuntimeError("请先调用 fit()")
            return self._predict_direct(df, origin, horizon)
        return self._predict_recursive(df, origin, horizon)

    def _predict_direct(self, df: pd.DataFrame, origin: int,
                        horizon: int) -> np.ndarray:
        from ..features import LAG_STEPS, ROLL_WINDOWS
        warmup = max(list(LAG_STEPS) + list(ROLL_WINDOWS)) + 5
        ctx = df.iloc[:origin + 1].tail(warmup)
        row = build_features(ctx, self.groups).iloc[-1]

        future_idx = df.index[origin + 1:origin + 1 + horizon]
        future = pd.DataFrame(index=future_idx)
        if "temp" in df.columns:
            future["temp"] = df["temp"].reindex(future_idx).to_numpy()
        fut_feats = (build_exog(future, [g for g in ("calendar", "temperature")
                                         if g in self.groups],
                                include_annual=self.cfg.include_annual)
                     if self.fwd_cols else None)

        anchor = float(df["load"].iloc[origin])       # 全部锚定在起点真实值
        preds = []
        for h in range(1, horizon + 1):
            mdl = self.models.get(h)
            if mdl is None:
                preds.append(anchor)
                continue
            parts = [np.asarray(row[self.cols_t].to_numpy(dtype=float), dtype=float)]
            if self.fwd_cols:
                parts.append(fut_feats[self.fwd_cols].iloc[h - 1].to_numpy(dtype=float))
            x = np.concatenate(parts).reshape(1, -1)
            preds.append(anchor + float(mdl.predict(x)[0]))
        return np.asarray(preds, dtype=float)

    def _predict_recursive(self, df: pd.DataFrame, origin: int,
                           horizon: int) -> np.ndarray:
        ctx_cols = ["load"] + (["temp"] if "temp" in df.columns else [])
        ctx = df.iloc[:origin + 1][ctx_cols].copy()

        preds: List[float] = []
        for step in range(horizon):
            ts = df.index[origin + 1 + step]
            temp = float(df["temp"].loc[ts]) if "temp" in df.columns else None
            row = next_step_row(ctx, self.groups, ts, temp=temp,
                                include_annual=self.cfg.include_annual)
            raw = float(self.model.predict(row[self.cols].to_numpy(dtype=float))[0])
            if self.target_mode == "delta":
                # 模型只学增量，基准是上一时刻的值（真实值或上一步预测值）
                value = float(ctx["load"].iloc[-1]) + raw
            else:
                value = raw
            preds.append(value)

            new_row = pd.DataFrame({"load": [value]}, index=pd.DatetimeIndex([ts]))
            if "temp" in ctx.columns:
                new_row["temp"] = np.nan if temp is None else temp
            ctx = pd.concat([ctx, new_row])
        return np.asarray(preds, dtype=float)

    def feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self.models:
            # 直接多步：24 个模型的平均重要性
            imp_values = np.mean([m.feature_importances_ for m in self.models.values()],
                                 axis=0)
        elif self.model is not None:
            imp_values = self.model.feature_importances_
        else:
            raise RuntimeError("请先调用 fit()")
        imp = pd.DataFrame({
            "feature": self.cols,
            "importance": imp_values,
        }).sort_values("importance", ascending=False)
        return imp.head(top_n).reset_index(drop=True)
