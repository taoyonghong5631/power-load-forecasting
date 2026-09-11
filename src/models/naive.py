# -*- coding: utf-8 -*-
"""朴素基线：任何预测模型都必须先打赢它，否则结论没有意义。

* ``season=1``  持久性（persistence）：直接用上一小时的值
* ``season=24`` 季节性朴素：直接用昨天同一时刻的值（日周期基准，M4/M5 竞赛的标准基准之一）

本项目用的 MT_001 表计长时间停在同一个读数上（连续十几小时 14.9 kW 这种），
持久性基线因此非常强——把它放进对比表才看得出模型到底有没有价值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from . import BaseForecaster


class NaiveForecaster(BaseForecaster):
    needs_refit_per_origin = False

    def __init__(self, cfg: Config, season: int = 1, name: str | None = None):
        self.cfg = cfg
        self.season = int(season)
        self.name = name or ("持久性 (t-1)" if season == 1 else "季节朴素 (t-24)")

    def fit(self, train_df: pd.DataFrame, verbose: bool = True) -> "NaiveForecaster":
        if verbose:
            print("  [%s] 无需训练（直接用 %d 小时前的真实值）" % (self.name, self.season))
        return self

    def predict(self, df: pd.DataFrame, origin: int, horizon: int) -> np.ndarray:
        """多步朴素预测：第 h 步用 y[origin + h - season]，超出已知范围时按周期回退。

        注意不能写成 ``iloc[origin+1-season : origin+1-season+horizon]``——当
        horizon > season 时那会取到 origin 之后的真实值（未来信息泄漏）。
        对 season=1 来说正确结果是"整段都用 origin 时刻的值"（持久性）。
        """
        if origin + 1 - self.season < 0:
            raise ValueError("历史长度不足 %d 小时" % self.season)
        load = df["load"].to_numpy(dtype=float)
        preds = []
        for h in range(1, horizon + 1):
            j = origin + h - self.season
            while j > origin:                 # 需要用到"未来"就回退一个周期
                j -= self.season
            preds.append(load[j])
        return np.asarray(preds, dtype=float)
