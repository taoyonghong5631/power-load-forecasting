# -*- coding: utf-8 -*-
"""全局配置：一个 dataclass 管住所有实验参数，方便命令行覆盖与复现。"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
SHOT_DIR = os.path.join(RESULTS_DIR, "screenshots")

for _d in (DATA_DIR, CACHE_DIR, RESULTS_DIR, FIG_DIR, SHOT_DIR):
    os.makedirs(_d, exist_ok=True)


@dataclass
class DataConfig:
    txt_path: str = os.path.join(DATA_DIR, "LD2011_2014.txt")
    zip_path: str = os.path.join(DATA_DIR, "LD2011_2014.txt.zip")
    client_col: int = 1                  # 0 是时间列，1 即 MT_001
    freq: str = "h"                      # 原始 15 分钟，重采样到小时
    zero_as_missing: bool = True         # 原始数据里 0 多为读数缺失
    interp_limit: int = 24               # 连续缺失超过 24 小时则不插值
    # 预处理阶段的极端值处理方式（可关）。默认用 Isolation Forest
    outlier_method: str = "iforest"      # 'none' | '3sigma' | 'iforest'
    outlier_contamination: float = 0.005


@dataclass
class TaskConfig:
    look_back: int = 24                  # 输入窗口：过去 24 小时
    horizon: int = 24                    # 预测未来 24 小时
    train_ratio: float = 0.8
    n_eval_origins: int = 30             # 滚动起点个数
    eval_stride: int = 24                # 起点间隔（小时）
    seed: int = 42


@dataclass
class LSTMConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 8                    # 早停
    val_ratio: float = 0.15


@dataclass
class XGBConfig:
    n_estimators: int = 600
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    tree_method: str = "hist"
    n_jobs: int = 6


@dataclass
class ARIMAConfig:
    # 固定阶数避免昂贵的 auto_arima 搜索；需要时可 --auto-arima 触发小网格 AIC 选阶
    order: tuple = (1, 1, 1)
    seasonal_order: tuple = (1, 1, 1, 24)
    train_window: int = 1500             # 每个起点用末尾 1500 小时重新拟合
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    use_temperature_exog: bool = False


@dataclass
class AnomalyConfig:
    contamination: float = 0.01
    sigma_k: float = 3.0
    n_estimators: int = 200
    max_features: float = 1.0
    rolling_window: int = 24


@dataclass
class TemperatureConfig:
    """数据集本身没有温度列，用 Open-Meteo 历史再分析温度做外生变量。"""
    source: str = "open-meteo"           # 'open-meteo' | 'synthetic' | 'none'
    latitude: float = 38.72              # 里斯本（数据来自葡萄牙 EDP 电网）
    longitude: float = -9.14
    timeout: int = 60
    cache_path: str = os.path.join(CACHE_DIR, "temperature_lisbon.csv")


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    xgb: XGBConfig = field(default_factory=XGBConfig)
    arima: ARIMAConfig = field(default_factory=ARIMAConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    use_calendar: bool = True
    use_temperature: bool = True
    # 年度特征（month / month_sin / month_cos）需要训练集覆盖至少一整年才有意义，
    # 否则会退化成"记住训练期那几个月"，在测试期反而拖后腿。
    # run_pipeline 会按训练跨度自动设置；也可用 --annual on/off 强制。
    include_annual: bool = True
    device: str = "cpu"

    # ---------- 工具方法 ----------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def fingerprint(self) -> str:
        """关键参数指纹，用于缓存命名。"""
        keys = [self.data.client_col, self.data.freq, self.task.look_back,
                self.task.horizon, self.task.train_ratio, self.use_temperature]
        raw = "_".join(str(k) for k in keys)
        return raw.replace("-", "m").replace(".", "").replace(" ", "")

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    def apply_quick(self) -> "Config":
        """快速模式：参数缩到能在几分钟内跑完，用于冒烟测试。"""
        self.task.n_eval_origins = 6
        self.task.eval_stride = 96
        self.lstm.epochs = 6
        self.xgb.n_estimators = 200
        self.arima.train_window = 720
        self.arima.seasonal_order = (1, 1, 0, 24)
        self.anomaly.n_estimators = 100
        return self


def get_config(**overrides) -> Config:
    """构造配置，支持 ``get_config(seed=0)`` 或嵌套覆盖 ``task__horizon=48``。"""
    cfg = Config()
    for key, val in overrides.items():
        if "__" in key:
            group, _, attr = key.partition("__")
            setattr(getattr(cfg, group), attr, val)
        elif hasattr(cfg, key):
            setattr(cfg, key, val)
        else:
            raise KeyError("未知配置项: %s" % key)
    return cfg


LAG_STEPS = (1, 2, 3, 6, 12, 24, 48, 72, 168)
ROLL_WINDOWS = (24, 168)

FEATURE_GROUPS = {
    "base": ["lag_%d" % k for k in LAG_STEPS] + [
        "roll_mean_24", "roll_std_24", "roll_mean_168", "diff_1", "diff_24"],
    "calendar": ["hour", "dow", "month", "is_weekend",
                 "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                 "month_sin", "month_cos", "is_holiday"],
    "temperature": ["temp", "temp_lag_24", "temp_roll_mean_24", "hdd_18", "cdd_22"],
}

# 消融实验：逐级加特征
ABLATION_LEVELS = {
    "Lags only": ("base",),
    "+ Calendar": ("base", "calendar"),
    "+ Calendar + Temperature": ("base", "calendar", "temperature"),
}
