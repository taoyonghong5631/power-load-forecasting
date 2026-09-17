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

for _d in (DATA_DIR, CACHE_DIR, RESULTS_DIR, FIG_DIR):
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
    # 'delta'：预测 y_t - y_{t-1}（推荐）。这类表计长时间停在同一个值上，
    # 直接预测水平会被 MSE 拉向均值、把平台和跳变都抹平；预测增量等于让模型
    # 从"持久性"出发只学修正量，ARIMA 的 d=1 也是同一个道理。
    target_mode: str = "delta"           # 'level' | 'delta'


@dataclass
class XGBConfig:
    n_estimators: int = 600              # 不用早停时的树数量
    max_rounds: int = 2000               # 用早停时的最大树数量（实际由验证集决定）
    early_stopping_rounds: int = 30
    val_ratio: float = 0.1               # 从训练段末尾切出的时间序验证集比例
    use_early_stopping: bool = True
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    min_child_weight: float = 1.0
    tree_method: str = "hist"
    n_jobs: int = 6
    target_mode: str = "delta"           # 'level' | 'delta'
    # 'direct'：每个预测步单独训练一个模型，预测"相对起点真实值的增量"，
    #   24 步全部锚定在起点上，不存在递归误差累积（日 Ahead 负荷预测的标准做法）。
    # 'recursive'：只训练一步模型，然后把自己的输出喂回去走完 24 步。
    # 本项目实测（MT_001，30 个滚动起点）：recursive 池化 MAE 1.669，direct 1.604，
    # 两者都列在对比表里；该表计跳变不可预测，哪种策略都无法显著超过持久性基线。
    strategy: str = "recursive"          # 'direct' | 'recursive'


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
class LLMConfig:
    """大模型配置。默认 DeepSeek：国内直连、OpenAI 兼容协议、价格极低。

    换服务商只改这三行即可（都提供 OpenAI 兼容接口）：
        智谱 GLM   base_url=https://open.bigmodel.cn/api/paas/v4   model=glm-4-flash
        通义千问   base_url=https://dashscope.aliyuncs.com/compatible-mode/v1  model=qwen-turbo
        Kimi       base_url=https://api.moonshot.cn/v1             model=moonshot-v1-8k
    """
    provider: str = "deepseek"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: int = 90
    max_tool_rounds: int = 5          # Agent 最多调用几轮工具
    env_path: str = os.path.join(ROOT, ".env")


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    xgb: XGBConfig = field(default_factory=XGBConfig)
    arima: ARIMAConfig = field(default_factory=ARIMAConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
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
