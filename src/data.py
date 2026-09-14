# -*- coding: utf-8 -*-
"""数据层：下载/加载原始 UCI 数据、重采样清洗、外接温度、以及界面上传文件的解析。

数据说明
--------
UCI ElectricityLoadDiagrams20112014：葡萄牙 370 个客户 2011-01-01 ~ 2014-12-31
的 15 分钟用电量（单位 kW，即 15 分钟平均功率）。原始文件用 ``;`` 分隔、
小数点用 ``,``，体积约 400+ MB。

需要注意的数据特性
------------------
* 很多客户（含 MT_001）在 2011 年很长一段时间读数恒为 0，这是**采集缺失**
  而不是真实零负荷，所以要先把 0 当缺失再插值。
* 文件里的极端值用全局均值替换会抹平日周期，这里改成"标记 + 局部插值"。
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests

from .anomaly import build_anomaly_features, global_3sigma_mask
from .config import CACHE_DIR, ROOT, Config


# --------------------------------------------------------------------------- #
# 原始数据
# --------------------------------------------------------------------------- #
def ensure_raw_data(cfg: Config) -> str:
    """确保 data/LD2011_2014.txt 存在，不存在就调用 scripts/fetch_data.py 下载。"""
    txt = cfg.data.txt_path
    if os.path.exists(txt) and os.path.getsize(txt) > 1024:
        return txt
    script = os.path.join(ROOT, "scripts", "fetch_data.py")
    print("[data] 本地缺失原始数据，开始下载（zip 约 261 MB，服务器较慢，请耐心）...")
    ret = subprocess.call([sys.executable, script])
    if ret != 0 or not os.path.exists(txt):
        raise RuntimeError(
            "原始数据不可用。请手动下载 LD2011_2014.txt.zip 放到 data/ 目录，"
            "或运行 python scripts/fetch_data.py")
    return txt


def load_raw(cfg: Config, nrows: Optional[int] = None) -> pd.DataFrame:
    """读取 15 分钟原始序列（只取时间列 + 目标客户列，避免读 700MB 全表）。"""
    txt = ensure_raw_data(cfg)
    client_col = cfg.data.client_col
    print("[data] 读取 %s（第 %d 列客户）..." % (os.path.basename(txt), client_col))
    df = pd.read_csv(
        txt, sep=";", decimal=",", index_col=0, parse_dates=True,
        usecols=[0, client_col], nrows=nrows, low_memory=False,
    )
    df.columns = ["load"]
    df = df.astype(float)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def preprocess(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """重采样到小时 + 0 值当缺失 + 插值 + 极端值处理。"""
    dcfg = cfg.data
    out = df.resample(cfg.data.freq).mean()
    out = out.astype(float)

    # 补全小时网格：中间出现缺口会让 lag_1 不再是"上一小时"，破坏滞后特征语义
    if len(out) > 1:
        full_idx = pd.date_range(out.index.min(), out.index.max(), freq=dcfg.freq)
        missing_slots = len(full_idx) - len(out)
        out = out.reindex(full_idx)
        if missing_slots:
            print("[data] 重采样网格补全了 %d 个缺口小时" % missing_slots)

    if dcfg.zero_as_missing:
        out.loc[out["load"] <= 0, "load"] = np.nan

    # 缺失比例过高时给出提示（例如客户从年中才开始计量）
    miss = float(out["load"].isna().mean())
    if miss > 0.2:
        print("[data] 警告：重采样后缺失比例 %.1f%%，该客户可能很晚才开始计量。" % (miss * 100))

    out["load"] = out["load"].interpolate(method="time", limit=dcfg.interp_limit,
                                          limit_area="inside")
    # 只在"首个到最后一个有效观测之间"做前向/后向填充，避免在序列两端
    # 用相邻值伪造出一段不存在的读数
    first_valid = out["load"].first_valid_index()
    last_valid = out["load"].last_valid_index()
    if first_valid is not None:
        out = out.loc[first_valid:last_valid]
    out["load"] = out["load"].ffill(limit=dcfg.interp_limit).bfill(limit=dcfg.interp_limit)
    n_before = len(out)
    out = out.dropna(subset=["load"])
    if len(out) < n_before:
        print("[data] 丢弃 %d 个无法修复的小时（多为客户尚未开始计量的时段）"
              % (n_before - len(out)))
    try:
        out.index.freq = pd.infer_freq(out.index)
    except (ValueError, TypeError):
        print("[data] 警告：时间索引存在不规则缺口，滞后特征可能跨过缺口")

    if dcfg.outlier_method == "3sigma":
        mask = global_3sigma_mask(out["load"].to_numpy(), k=3.0)
        out.loc[mask, "load"] = np.nan
        out["load"] = out["load"].interpolate(method="time", limit_area="inside")
        out["load"] = out["load"].fillna(out["load"].median())
    elif dcfg.outlier_method == "iforest":
        from sklearn.ensemble import IsolationForest
        feats = build_anomaly_features(out, window=24).dropna()
        if len(feats) > 100:
            model = IsolationForest(
                n_estimators=cfg.anomaly.n_estimators,
                contamination=dcfg.outlier_contamination,
                random_state=cfg.task.seed, n_jobs=1,
            )
            pred = model.fit_predict(feats.values)
            bad = feats.index[pred == -1]
            out.loc[bad, "load"] = np.nan
            out["load"] = out["load"].interpolate(method="time", limit_area="inside")
            out["load"] = out["load"].fillna(out["load"].median())
            print("[data] Isolation Forest 标记并修复了 %d 个小时级极端点" % len(bad))
    return out


def hourly_series(cfg: Config, use_cache: bool = True) -> pd.DataFrame:
    """小时级序列（带缓存，避免每次都去解析 700MB 文本）。"""
    min_hours = 24 * 30
    cache = os.path.join(CACHE_DIR, "hourly_client%d_%s.csv"
                         % (cfg.data.client_col, cfg.data.freq))
    if use_cache and os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if len(df) < min_hours:
            print("[data] 缓存文件过小（%d 小时），忽略并重新生成" % len(df))
            df = None
        else:
            try:
                df.index.freq = pd.infer_freq(df.index)
            except (ValueError, TypeError):
                pass
            print("[data] 命中缓存 %s（%d 小时）" % (os.path.basename(cache), len(df)))
            return df
    raw = load_raw(cfg)
    df = preprocess(raw, cfg)
    if len(df) < min_hours:
        raise ValueError(
            "客户列 %d 清洗后只剩 %d 小时有效数据（需要至少 %d 小时）。"
            "该客户很可能从 2012 年才开始计量，请换一个 --client-col，"
            "例如 300（该客户在 2011 年就有读数）。"
            % (cfg.data.client_col, len(df), min_hours))
    if use_cache:
        df.to_csv(cache)
        print("[data] 已写缓存 %s" % cache)
    return df


# --------------------------------------------------------------------------- #
# 温度（外生变量）
# --------------------------------------------------------------------------- #
def synthetic_temperature(index: pd.DatetimeIndex, seed: int = 42) -> pd.Series:
    """离线兜底：用季节 + 日周期 + 噪声合成一条"像样"的温度曲线。"""
    rng = np.random.default_rng(seed)
    doy = index.dayofyear.to_numpy()
    hour = index.hour.to_numpy()
    seasonal = 16.0 + 8.0 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    diurnal = 4.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    noise = rng.normal(0, 1.5, len(index))
    return pd.Series(seasonal + diurnal + noise, index=index, name="temp")


def _fetch_openmeteo(start: pd.Timestamp, end: pd.Timestamp, cfg: Config,
                     chunk_days: int = 330) -> pd.Series:
    """从 Open-Meteo 历史再分析接口取小时温度（免费、无需 key）。"""
    tcfg = cfg.temperature
    url = "https://archive-api.open-meteo.com/v1/archive"
    times, temps = [], []
    cur = start
    while cur <= end:
        stop = min(cur + pd.Timedelta(days=chunk_days), end)
        params = {
            "latitude": tcfg.latitude, "longitude": tcfg.longitude,
            "start_date": cur.strftime("%Y-%m-%d"), "end_date": stop.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m", "timezone": "Europe/Lisbon",
        }
        resp = requests.get(url, params=params, timeout=tcfg.timeout)
        resp.raise_for_status()
        payload = resp.json()["hourly"]
        times.extend(payload["time"])
        temps.extend(payload["temperature_2m"])
        print("[data] 温度 %s ~ %s 已获取" % (params["start_date"], params["end_date"]))
        cur = stop + pd.Timedelta(days=1)
    idx = pd.to_datetime(pd.Series(times))
    s = pd.Series(np.asarray(temps, dtype=float), index=idx, name="temp")
    return s[~s.index.duplicated(keep="first")].sort_index()


def load_temperature(index: pd.DatetimeIndex, cfg: Config) -> pd.Series:
    """获取与 index 对齐的小时温度；失败自动退回合成温度。"""
    tcfg = cfg.temperature
    if tcfg.source == "none":
        return pd.Series(np.nan, index=index, name="temp")

    if tcfg.source == "synthetic":
        return synthetic_temperature(index, cfg.task.seed)

    series = None
    cache = tcfg.cache_path
    if os.path.exists(cache):
        try:
            cached = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
            cached.index = pd.to_datetime(cached.index)
        except Exception:
            cached = None
        if (cached is not None and len(cached) > 100
                and cached.index.min() <= index.min()
                and cached.index.max() >= index.max()):
            series = cached
            print("[data] 命中温度缓存 %s" % os.path.basename(cache))
    if series is None:
        try:
            series = _fetch_openmeteo(index.min(), index.max(), cfg)
            if len(series):
                series.to_frame("temp").to_csv(cache)
        except Exception as exc:
            warnings.warn("温度获取失败(%s)，改用合成温度兜底" % type(exc).__name__)
            return synthetic_temperature(index, cfg.task.seed)

    aligned = series.reindex(series.index.union(index)).interpolate(
        method="time", limit_direction="both").reindex(index)
    aligned.name = "temp"
    return aligned


def get_dataset(cfg: Config, use_temperature: Optional[bool] = None,
                use_cache: bool = True) -> pd.DataFrame:
    """返回建模用的数据框：index=时间, columns=['load', ('temp')]。"""
    df = hourly_series(cfg, use_cache=use_cache)
    use_temp = cfg.use_temperature if use_temperature is None else use_temperature
    if use_temp:
        df = df.copy()
        df["temp"] = load_temperature(df.index, cfg)
    return df


# --------------------------------------------------------------------------- #
# 界面上传文件的解析
# --------------------------------------------------------------------------- #
def parse_uploaded_file(source, name: str = "") -> pd.DataFrame:
    """把用户上传的 CSV/TXT 解析成小时级 'load' 序列。

    支持三种常见格式：
    1. 单文件两列：``时间, 负荷``（自动嗅探分隔符与小数点）
    2. UCI 原始格式：``;`` 分隔 + ``,`` 小数点 + 首个 MT_xxx 客户列
    3. 带表头的第一列时间戳
    """
    raw = source.read() if hasattr(source, "read") else open(source, "rb").read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    head = raw[:4096].decode("utf-8", "ignore")
    sep = ";" if head.count(";") > head.count(",") else ","
    decimal = "," if sep == ";" else "."

    df = pd.read_csv(io.BytesIO(raw), sep=sep, decimal=decimal, index_col=0,
                     parse_dates=True, low_memory=False)
    df = df.select_dtypes(include=[np.number]).astype(float)
    if df.shape[1] == 0:
        raise ValueError("没有解析出数值列，请检查文件格式")
    df = df.iloc[:, [0]]
    df.columns = ["load"]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    hourly = df.resample("h").mean().interpolate(method="time")
    hourly = hourly.dropna()
    if len(hourly) < 240:
        raise ValueError("有效数据不足 240 小时（10 天），无法建模")
    return hourly


# --------------------------------------------------------------------------- #
# 离线兜底：合成数据
# --------------------------------------------------------------------------- #
def synthetic_dataset(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """无网络/无数据时的演示数据集：日周期 + 周周期 + 年周期 + 节假日效应 + 噪声。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-01-01", periods=n_days * 24, freq="h")
    t = np.arange(len(idx))
    daily = 60.0 * np.sin(2 * np.pi * (idx.hour.to_numpy() - 8) / 24.0)
    weekly = 25.0 * np.sin(2 * np.pi * idx.dayofweek.to_numpy() / 7.0)
    yearly = 40.0 * np.sin(2 * np.pi * (idx.dayofyear.to_numpy() - 110) / 365.25)
    weekend = np.where(idx.dayofweek.to_numpy() >= 5, -30.0, 0.0)
    temp = synthetic_temperature(idx, seed).to_numpy()
    trend = 0.002 * t
    noise = rng.normal(0, 8.0, len(idx))
    load = 300.0 + daily + weekly + yearly + weekend + 3.0 * temp + trend + noise
    return pd.DataFrame({"load": np.maximum(load, 20.0), "temp": temp}, index=idx)
