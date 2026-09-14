# -*- coding: utf-8 -*-
"""异常检测：3σ 基线 vs Isolation Forest，并给出可量化的对比实验。

最朴素的"全局 3σ + 把异常值替换成全局均值"在负荷这种强周期性数据上有两处硬伤：

1. 全局均值和标准差被早晚高峰撑大，阈值过宽，漏检白天的尖峰与夜间骤降；
2. 用均值替换异常点会把整条曲线拉平，破坏日周期形态，下游预测会学坏。

这里的做法：
    * 3σ 基线用**滚动** 3σ（以 24 小时窗口的局部中位数与标准差为准），能跟上日周期；
    * 主检测器换成 Isolation Forest（多特征：负荷、相对滚动中位数的偏离、
      一阶差分、滚动波动率、小时/星期的周期编码）；
    * 检测到的异常**标记出来**而不是粗暴置为均值，交给下游决定是否剔除。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .config import AnomalyConfig, Config


def _as_bool(mask) -> np.ndarray:
    return np.asarray(mask, dtype=bool).ravel()


# --------------------------------------------------------------------------- #
# 基线：3σ
# --------------------------------------------------------------------------- #
def rolling_3sigma_mask(values, k: float = 3.0, window: int = 24,
                        min_periods: Optional[int] = None) -> np.ndarray:
    """滚动 3σ：以窗口内的局部均值/标准差为基准，能跟上日周期。

    ``window=None`` 时退化为全局 3σ（作为最朴素的对照）。
    """
    s = pd.Series(np.asarray(values, dtype=float).ravel())
    if window is None:
        mu, sd = s.mean(), s.std()
        return _as_bool((s - mu).abs() > k * sd)
    min_periods = min_periods or max(4, window // 2)
    mu = s.rolling(window, min_periods=min_periods, center=True).median()
    sd = s.rolling(window, min_periods=min_periods, center=True).std()
    sd = sd.replace(0.0, np.nan)
    return _as_bool((s - mu).abs() > k * sd.fillna(sd.max()))


def sigma_scores(values, window: Optional[int] = 24) -> np.ndarray:
    """3σ 的连续异常分数（|偏差| / 局部标准差），用于计算与阈值无关的 AUC。"""
    s = pd.Series(np.asarray(values, dtype=float).ravel())
    if window is None:
        mu, sd = s.mean(), s.std()
        return np.abs((s - mu) / (sd if sd else 1.0)).to_numpy()
    min_periods = max(4, window // 2)
    mu = s.rolling(window, min_periods=min_periods, center=True).median()
    sd = s.rolling(window, min_periods=min_periods, center=True).std()
    sd = sd.replace(0.0, np.nan)
    score = np.abs((s - mu) / sd)
    return score.fillna(0.0).to_numpy()


def global_3sigma_mask(values, k: float = 3.0) -> np.ndarray:
    """全局 3σ（保留做对比基线）。"""
    return rolling_3sigma_mask(values, k=k, window=None)


# --------------------------------------------------------------------------- #
# Isolation Forest
# --------------------------------------------------------------------------- #
def build_anomaly_features(df: pd.DataFrame, load_col: str = "load",
                           window: int = 24) -> pd.DataFrame:
    """异常检测用的特征矩阵（只依赖当前点及历史，可用于在线检测）。"""
    s = df[load_col].astype(float)
    roll_med = s.rolling(window, min_periods=max(3, window // 4)).median()
    roll_std = s.rolling(window, min_periods=max(3, window // 4)).std()
    roll_mean = s.rolling(window, min_periods=max(3, window // 4)).mean()

    feats = pd.DataFrame(index=df.index)
    feats["load"] = s
    feats["dev_median"] = s - roll_med
    feats["dev_ratio"] = s / roll_med.replace(0.0, np.nan)
    feats["diff_1"] = s.diff(1)
    feats["roll_std"] = roll_std
    feats["z_local"] = (s - roll_mean) / roll_std.replace(0.0, np.nan)

    idx = df.index
    feats["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24.0)
    feats["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24.0)
    feats["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
    feats["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7.0)

    return feats.replace([np.inf, -np.inf], np.nan)


def fit_isolation_forest(df: pd.DataFrame, cfg: Config,
                         load_col: str = "load") -> Tuple[IsolationForest, list]:
    """在（应该是干净的）训练段上拟合 IF，返回模型与特征列名。"""
    acfg: AnomalyConfig = cfg.anomaly
    feats = build_anomaly_features(df, load_col=load_col,
                                   window=acfg.rolling_window).dropna()
    model = IsolationForest(
        n_estimators=acfg.n_estimators,
        contamination=acfg.contamination,
        max_features=acfg.max_features,
        random_state=cfg.task.seed,
        n_jobs=1,
    )
    model.fit(feats.values)
    return model, list(feats.columns)


def iforest_predict(model: IsolationForest, df: pd.DataFrame,
                    feature_cols: list, load_col: str = "load",
                    window: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (异常掩码, 异常分数)。分数越低越异常。"""
    feats = build_anomaly_features(df, load_col=load_col, window=window)
    valid = feats[feature_cols].notna().all(axis=1)
    mask = np.zeros(len(df), dtype=bool)
    scores = np.full(len(df), np.nan)
    if valid.any():
        x = feats.loc[valid, feature_cols].values
        pred = model.predict(x)
        scores[valid.to_numpy()] = model.decision_function(x)
        mask[valid.to_numpy()] = pred == -1
    return mask, scores


# --------------------------------------------------------------------------- #
# 可控对比实验：注入已知异常，算 precision / recall / F1
# --------------------------------------------------------------------------- #
ANOMALY_KINDS = {1: "尖峰(全局极端)", 2: "骤降(全局极端)", 3: "平台(连续)",
                 4: "上下文(局部异常)"}


def inject_anomalies(values, n_spikes: int = 15, n_drops: int = 15,
                     n_flats: int = 8, n_context: int = 15, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """在真实序列上注入四类已知异常，返回 (被污染序列, 真值掩码, 类型码, 事件表)。

    * 尖峰 1：*2.5~4.0 倍，全局极端
    * 骤降 2：*0.15~0.40 倍，全局极端
    * 平台 3：连续 3~6 小时复制前一个值
    * 上下文 4：把低负荷时段的负荷抬到"白天量级"，**仍在全局正常范围内**，
      但相对当时的小时/局部模式明显异常 —— 这正是全局 3σ 的盲区。
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(values, dtype=float).copy()
    n = len(y)
    truth = np.zeros(n, dtype=bool)
    kinds = np.zeros(n, dtype=int)
    events: list = []
    lo, hi = 48, n - 48

    order = rng.permutation(np.arange(lo, hi))
    cursor = 0

    q25 = float(np.quantile(y, 0.25))

    def _next_free(span: int = 1, predicate=None):
        nonlocal cursor
        while cursor < len(order):
            p = int(order[cursor])
            cursor += 1
            if truth[max(0, p - 1):p + span + 1].any():
                continue
            if predicate is not None and not predicate(p):
                continue
            return p
        return None

    for _ in range(n_spikes):
        p = _next_free()
        if p is None:
            break
        y[p] *= rng.uniform(2.5, 4.0)
        truth[p] = True
        kinds[p] = 1
        events.append((1, np.array([p])))

    for _ in range(n_drops):
        p = _next_free()
        if p is None:
            break
        y[p] *= rng.uniform(0.15, 0.40)
        truth[p] = True
        kinds[p] = 2
        events.append((2, np.array([p])))

    for _ in range(n_context):
        # 只在"低负荷时段"注入，这样抬升后的值仍在全局正常范围内
        p = _next_free(predicate=lambda idx: y[idx] <= q25)
        if p is None:
            break
        level = float(np.quantile(y, 0.90)) * rng.uniform(0.85, 1.00)
        y[p] = max(y[p], level)
        truth[p] = True
        kinds[p] = 4
        events.append((4, np.array([p])))

    for _ in range(n_flats):
        length = int(rng.integers(3, 7))
        p = _next_free(span=length)
        if p is None:
            break
        end = min(p + length, hi)
        y[p:end] = y[p - 1]
        truth[p:end] = True
        kinds[p:end] = 3
        events.append((3, np.arange(p, end)))

    return y, truth, kinds, events


def detection_metrics(true_mask, pred_mask, tolerance: int = 0) -> Dict[str, float]:
    """点对点 precision/recall/F1；tolerance>0 时用 ±tolerance 邻域匹配事件。"""
    true_mask, pred_mask = _as_bool(true_mask), _as_bool(pred_mask)
    if tolerance <= 0:
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
    else:
        tp = fp = 0
        hit = np.zeros(len(true_mask), dtype=bool)
        for i in np.flatnonzero(pred_mask):
            seg = slice(max(0, i - tolerance), min(len(true_mask), i + tolerance + 1))
            if true_mask[seg].any():
                tp += 1
                hit[seg] |= pred_mask[seg] & true_mask[seg]
            else:
                fp += 1
        fn = int(np.sum(true_mask & ~hit))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn,
            "Precision": prec, "Recall": rec, "F1": f1}


def top_rate_mask(scores, rate: float) -> np.ndarray:
    """把连续分数转成"报警率固定为 rate"的掩码（匹配工作点用）。"""
    s = np.asarray(scores, dtype=float).ravel()
    k = max(1, int(round(len(s) * rate)))
    idx = np.argsort(-s)[:k]
    mask = np.zeros(len(s), dtype=bool)
    mask[idx] = True
    return mask


def threshold_free_metrics(scores, truth) -> Dict[str, float]:
    """与阈值无关的排序质量：ROC-AUC 与 PR-AUC（不平衡数据看 PR-AUC）。"""
    from sklearn.metrics import average_precision_score, roc_auc_score

    s = np.nan_to_num(np.asarray(scores, dtype=float).ravel(), nan=0.0)
    t = _as_bool(truth).astype(int)
    out = {}
    try:
        out["ROC_AUC"] = float(roc_auc_score(t, s))
    except ValueError:
        out["ROC_AUC"] = float("nan")
    try:
        out["PR_AUC"] = float(average_precision_score(t, s))
    except ValueError:
        out["PR_AUC"] = float("nan")
    return out


def detector_scores(clean_df: pd.DataFrame, target_df: pd.DataFrame, cfg: Config
                    ) -> Dict[str, np.ndarray]:
    """三种检测器在 target_df 上的连续异常分数（越大越可疑）。

    Isolation Forest 只在**训练段（干净数据）**上拟合，再去目标段打分，避免泄漏。
    """
    acfg = cfg.anomaly
    values = target_df["load"].to_numpy(dtype=float)
    model, cols = fit_isolation_forest(clean_df, cfg)
    _, if_score = iforest_predict(model, target_df, cols, window=acfg.rolling_window)
    # decision_function 越小越异常，取负号让"越大越可疑"
    if_score = -np.nan_to_num(if_score, nan=0.0)
    return {
        "3-sigma (global)": sigma_scores(values, window=None),
        "3-sigma (rolling 24h)": sigma_scores(values, window=acfg.rolling_window),
        "Isolation Forest": if_score,
    }


def benchmark_detectors(clean_df: pd.DataFrame, test_df: pd.DataFrame,
                        cfg: Config, n_days: int = 30,
                        alert_rate: float = 0.05) -> Dict[str, object]:
    """在测试段注入已知异常，对三种检测器做**同工作点**的公平比较。

    直接比 P/R/F1 是不公平的：3σ 的阈值由 k=3 决定，IF 的阈值由 contamination
    决定，两者报警率天差地别。所以这里额外给出：
    * ROC-AUC / PR-AUC：与阈值无关的排序质量；
    * 固定报警率（默认 5%）下的 P/R/F1：把两者拉到同一工作点再比。
    另外保留各自的"默认阈值"结果作为参考。
    """
    acfg = cfg.anomaly
    values = test_df["load"].to_numpy(dtype=float)
    scale = max(1.0, len(values) / (24 * n_days))
    corrupted, truth, kinds, events = inject_anomalies(
        values,
        n_spikes=int(round(15 * scale)),
        n_drops=int(round(15 * scale)),
        n_flats=int(round(8 * scale)),
        n_context=int(round(15 * scale)),
        seed=cfg.task.seed,
    )
    corrupted_df = test_df.copy()
    corrupted_df["load"] = corrupted

    scores = detector_scores(clean_df, corrupted_df, cfg)
    default_masks = {
        "3-sigma (global)": sigma_scores(corrupted, window=None) > acfg.sigma_k,
        "3-sigma (rolling 24h)": sigma_scores(corrupted, window=acfg.rolling_window) > acfg.sigma_k,
    }
    model, cols = fit_isolation_forest(clean_df, cfg)
    m_if, _ = iforest_predict(model, corrupted_df, cols, window=acfg.rolling_window)
    default_masks["Isolation Forest"] = m_if

    rows: Dict[str, Dict[str, float]] = {}
    for name, score in scores.items():
        row = threshold_free_metrics(score, truth)
        matched = detection_metrics(truth, top_rate_mask(score, alert_rate))
        row.update({"P@%d%%" % int(alert_rate * 100): matched["Precision"],
                    "R@%d%%" % int(alert_rate * 100): matched["Recall"],
                    "F1@%d%%" % int(alert_rate * 100): matched["F1"]})
        base = detection_metrics(truth, default_masks[name])
        row.update({"默认阈值_Precision": base["Precision"],
                    "默认阈值_Recall": base["Recall"],
                    "默认阈值_F1": base["F1"],
                    "默认阈值_报警数": float(np.sum(default_masks[name]))})
        rows[name] = row

    # 分类型召回：点级 + 事件级（事件级 = 该异常事件至少有一个点被检出）
    by_kind: Dict[str, Dict[str, float]] = {}
    by_kind_event: Dict[str, Dict[str, float]] = {}
    for name, mask in default_masks.items():
        mask = np.asarray(mask)
        by_kind[name], by_kind_event[name] = {}, {}
        for code, label in ANOMALY_KINDS.items():
            sel = kinds == code
            if sel.any():
                by_kind[name][label] = float(np.mean(mask[sel]))
            evs = [pos for c, pos in events if c == code]
            if evs:
                by_kind_event[name][label] = float(
                    np.mean([bool(mask[pos].any()) for pos in evs]))

    return {"table": rows, "scores": scores, "truth": truth, "corrupted": corrupted,
            "kinds": kinds, "alert_rate": alert_rate, "default_masks": default_masks,
            "by_kind": by_kind, "by_kind_event": by_kind_event, "events": events}
