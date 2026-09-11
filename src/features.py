# -*- coding: utf-8 -*-
"""特征工程：滞后/滚动特征 + 日期特征 + 温度特征。

关键约束：训练特征和递归预测时的特征必须用**同一个函数**生成，否则会有
训练/推理偏置（train-serving skew）。所以这里所有特征都写成"给定一个
含 load 列的 DataFrame，返回加好特征的 DataFrame"，递归预测时把预测值
追加回 load 列再调用同一个函数即可。
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Config, FEATURE_GROUPS, LAG_STEPS, ROLL_WINDOWS

ANNUAL_FEATURES = ("month", "month_sin", "month_cos")


def holiday_dates(years: Sequence[int], country: str = "PT") -> set:
    """节假日日期集合；没装 holidays 包就返回空集合（特征恒为 0）。"""
    try:
        import holidays as _holidays
    except ImportError:
        return set()
    try:
        return set(_holidays.country_holidays(country, years=list(years)).keys())
    except Exception:
        return set()


def add_calendar_features(df: pd.DataFrame, country: str = "PT") -> pd.DataFrame:
    """小时、星期、月份、周末、节假日 + 周期项的 sin/cos 编码。"""
    out = df.copy()
    idx = out.index
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    month = idx.month.to_numpy()

    out["hour"] = hour
    out["dow"] = dow
    out["month"] = month
    out["is_weekend"] = (dow >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    days = idx.normalize()
    hols = holiday_dates(sorted(set(idx.year)), country)
    out["is_holiday"] = np.array([1 if d.date() in hols else 0 for d in days],
                                 dtype=int) if hols else 0
    return out


def add_lag_features(df: pd.DataFrame, target: str = "load",
                     lags: Sequence[int] = LAG_STEPS,
                     windows: Sequence[int] = ROLL_WINDOWS) -> pd.DataFrame:
    """滞后值、滚动均值/标准差、一阶与日差分。

    全部基于 ``shift(1)`` 起步，保证预测 t 时刻时只用 t 之前的信息。
    """
    out = df.copy()
    base = out[target]
    for k in lags:
        out["lag_%d" % k] = base.shift(k)
    past = base.shift(1)
    for w in windows:
        out["roll_mean_%d" % w] = past.rolling(w, min_periods=max(2, w // 4)).mean()
        out["roll_std_%d" % w] = past.rolling(w, min_periods=max(2, w // 4)).std()
    out["diff_1"] = base.shift(1) - base.shift(2)
    out["diff_24"] = base.shift(1) - base.shift(25)
    return out


def add_temperature_features(df: pd.DataFrame, col: str = "temp") -> pd.DataFrame:
    """温度本身 + 昨日同时刻温度 + 24h 均温 + 采暖/制冷度（18℃ / 22℃ 基准）。"""
    out = df.copy()
    if col not in out.columns:
        return out
    t = out[col]
    out["temp"] = t
    out["temp_lag_24"] = t.shift(24)
    out["temp_roll_mean_24"] = t.rolling(24, min_periods=6).mean()
    out["hdd_18"] = np.maximum(0.0, 18.0 - t)
    out["cdd_22"] = np.maximum(0.0, t - 22.0)
    return out


def feature_names(groups: Iterable[str], include_annual: bool = True) -> List[str]:
    """按给定特征组，展平成列名列表。"""
    names: List[str] = []
    for g in groups:
        if g not in FEATURE_GROUPS:
            raise KeyError("未知特征组: %s（可选: %s）" % (g, list(FEATURE_GROUPS)))
        names.extend(FEATURE_GROUPS[g])
    if not include_annual:
        names = [n for n in names if n not in ANNUAL_FEATURES]
    return names


def exog_columns(groups: Iterable[str], include_annual: bool = True) -> List[str]:
    """LSTM 等序列模型的外生输入列：日期特征 + 原始温度（不含滞后项）。

    不用滞后温度是为了让"未来 24 小时的输入"能直接从日历和天气预报拿到，
    推理时不需要预测外生变量。
    """
    cols: List[str] = []
    for g in groups:
        if g == "calendar":
            cols.extend(FEATURE_GROUPS["calendar"])
        elif g == "temperature":
            cols.append("temp")
        elif g == "base":
            continue
        else:
            raise KeyError("序列模型不支持的特征组: %s" % g)
    if not include_annual:
        cols = [c for c in cols if c not in ANNUAL_FEATURES]
    return cols


def build_exog(df: pd.DataFrame, groups: Iterable[str],
               country: str = "PT", include_annual: bool = True) -> pd.DataFrame:
    """生成序列模型的外生输入矩阵（索引与 df 对齐）。"""
    cols = exog_columns(groups, include_annual=include_annual)
    out = pd.DataFrame(index=df.index)
    if any(c in cols for c in FEATURE_GROUPS["calendar"]):
        out = add_calendar_features(out, country=country)
    if "temp" in cols:
        if "temp" not in df.columns:
            raise KeyError("需要温度特征但数据里没有 temp 列")
        out["temp"] = df["temp"]
    out = out[cols]
    if "is_holiday" in out.columns and not out["is_holiday"].astype(bool).any():
        out["is_holiday"] = out["is_holiday"].astype(float)
    return out


def build_features(df: pd.DataFrame, groups: Iterable[str],
                   country: str = "PT") -> pd.DataFrame:
    """生成指定特征组的数据框（索引与原 df 对齐，不 dropna）。"""
    groups = tuple(groups)
    out = df.copy()
    if "base" in groups:
        out = add_lag_features(out, target="load")
    if "calendar" in groups:
        out = add_calendar_features(out, country=country)
    if "temperature" in groups:
        out = add_temperature_features(out, col="temp")
    return out


def build_supervised(df: pd.DataFrame, groups: Iterable[str],
                     target: str = "load", country: str = "PT",
                     include_annual: bool = True
                     ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """返回 (X, y, 特征列名)，去掉含 NaN 的行（滞后特征的开头部分）。"""
    cols = feature_names(groups, include_annual=include_annual)
    feats = build_features(df, groups, country=country)
    missing = [c for c in cols if c not in feats.columns]
    if missing:
        raise KeyError("特征列缺失: %s" % missing)
    data = pd.concat([feats[cols], feats[target].rename("__y__")], axis=1).dropna()
    return data[cols], data["__y__"], cols


def next_step_row(context: pd.DataFrame, groups: Iterable[str],
                  timestamp: pd.Timestamp, temp: float | None = None,
                  country: str = "PT", include_annual: bool = True) -> pd.DataFrame:
    """递归预测用的单步特征行。

    ``context`` 是"已知历史（含已预测值）"的数据框，最后一行时间 <= timestamp-1h；
    本函数把 timestamp 追加进去后取最后一行特征。未来时刻的温度/日历都是已知量。
    """
    warmup = max(list(LAG_STEPS) + list(ROLL_WINDOWS)) + 5
    context = context.tail(warmup)
    row = pd.DataFrame(index=pd.DatetimeIndex([timestamp]))
    if "temp" in context.columns:
        row["temp"] = np.nan if temp is None else float(temp)
    row["load"] = np.nan
    extended = pd.concat([context, row])
    feats = build_features(extended, groups, country=country)
    return feats.iloc[[-1]]
