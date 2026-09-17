# -*- coding: utf-8 -*-
"""上下文层：把 results/ 与 data/cache 里的产物统一加载、压缩成"给模型看的数据包"。

这一层是三个功能的共同地基：
    * 日报（方向一）  —— build_daily_brief() 生成一份紧凑 JSON
    * 问答（方向二）  —— DataHub 提供查询方法，由 tools.py 包装成工具
    * 归因（方向三）  —— 同样复用 DataHub + ToolBox

**所有数值都在这里算好**，模型只拿到结果，不接触原始序列、不做算术。
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import CACHE_DIR, RESULTS_DIR, Config


# --------------------------------------------------------------------------- #
# 数据中枢
# --------------------------------------------------------------------------- #
class DataHub:
    """加载一次，反复查询。所有方法都返回「已经算好的数字」，可直接喂给模型。"""

    def __init__(self, cfg: Config, results_dir: str = RESULTS_DIR):
        self.cfg = cfg
        self.results_dir = results_dir
        self.series = self._load_series()
        self.summary = self._load_json("metrics_summary.json")
        self.run_config = self._load_json("run_config.json")
        self.model_table = self._load_csv("model_comparison.csv")
        self.ablation = self._load_csv("ablation_features.csv")
        self.anomaly_bench = self._load_csv("anomaly_benchmark.csv")
        self.anomaly_recall = self._load_csv("anomaly_recall_by_kind.csv")
        # 特征重要性是两列（feature, importance）没有索引列，不能按 index_col=0 读
        self.feature_importance = self._load_csv("feature_importance.csv", index_col=None)
        self.anomaly_points = self._load_csv("anomaly_points.csv")
        self.predictions = self._load_predictions()

    # ---------------- 载入 ----------------
    def _load_series(self) -> Optional[pd.DataFrame]:
        name = "hourly_client%d_%s.csv" % (self.cfg.data.client_col, self.cfg.data.freq)
        path = os.path.join(CACHE_DIR, name)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.sort_index()
        # 小时级缓存里只有负荷；温度是另一份缓存，这里拼回来，
        # 否则日报里的"温度分析"永远是空的
        if "temp" not in df.columns:
            tpath = self.cfg.temperature.cache_path
            if os.path.exists(tpath):
                try:
                    t = pd.read_csv(tpath, index_col=0, parse_dates=True).iloc[:, 0]
                    t = t[~t.index.duplicated(keep="first")].sort_index()
                    aligned = t.reindex(t.index.union(df.index)).interpolate(
                        method="time", limit_direction="both").reindex(df.index)
                    df["temp"] = aligned
                except Exception:
                    pass
        return df

    def _load_csv(self, name: str, index_col=0) -> Optional[pd.DataFrame]:
        path = os.path.join(self.results_dir, name)
        if not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path, index_col=index_col)
        except Exception:
            return None

    def _load_json(self, name: str) -> dict:
        path = os.path.join(self.results_dir, name)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _load_predictions(self) -> Dict[str, dict]:
        out = {}
        for path in sorted(glob.glob(os.path.join(self.results_dir, "pred_*.npz"))):
            key = os.path.basename(path)[5:-4]      # 去掉 pred_ 前缀和 .npz 后缀
            try:
                d = np.load(path, allow_pickle=True)
                out[key] = {
                    "preds": d["preds"], "truths": d["truths"],
                    "origins": d["origins"] if "origins" in d else None,
                    "stamps": pd.to_datetime(d["stamps"]) if "stamps" in d else None,
                }
            except Exception:
                continue
        return out

    # ---------------- 基础信息 ----------------
    @property
    def ready(self) -> bool:
        return self.series is not None and len(self.series) > 0

    def describe(self) -> dict:
        """数据字典：告诉模型手上有哪些数据、字段是什么。"""
        d = {
            "客户": "MT_%03d" % self.cfg.data.client_col,
            "小时级样本数": int(len(self.series)) if self.ready else 0,
            "时间范围": ([str(self.series.index.min()), str(self.series.index.max())]
                     if self.ready else []),
            "包含温度": bool(self.ready and "temp" in self.series.columns),
            "已有产物": [k for k, v in {
                "模型对比": self.model_table, "特征消融": self.ablation,
                "异常检测基准": self.anomaly_bench, "特征重要性": self.feature_importance,
                "异常点清单": self.anomaly_points,
            }.items() if v is not None],
            "可用预测模型": list(self.predictions.keys()),
        }
        return d

    def latest_forecast(self, model: Optional[str] = None) -> Optional[dict]:
        """最近一次滚动预测窗口（24 小时）：预测值、实际值、误差。"""
        if not self.predictions:
            return None
        if model is None:
            model = self.best_model_name()
        if model not in self.predictions:
            model = list(self.predictions)[0]
        p = self.predictions[model]
        preds, truths, stamps = p["preds"], p["truths"], p["stamps"]
        if preds is None or len(preds) == 0:
            return None
        k = len(preds) - 1                      # 最后一个窗口 = 最近的一次预测
        times = (pd.date_range(stamps[k], periods=preds.shape[1], freq="h")
                 if stamps is not None else None)
        err = np.abs(truths[k] - preds[k])
        return {
            "模型": model,
            "起点": str(stamps[k]) if stamps is not None else "?",
            "时间": [str(t) for t in times] if times is not None else [],
            "预测值": [round(float(v), 2) for v in preds[k]],
            "实际值": [round(float(v), 2) for v in truths[k]],
            "平均绝对误差": round(float(np.nanmean(err)), 3),
        }

    def best_model_name(self) -> Optional[str]:
        """按 MAE 挑最好的「学习型」模型（朴素基线不作为日报主模型）。"""
        if self.model_table is not None and "MAE" in self.model_table.columns:
            learned = [i for i in self.model_table.index
                       if not any(t in i for t in ("持久性", "季节朴素"))]
            if learned:
                return min(learned, key=lambda k: self.model_table.loc[k, "MAE"])
        return list(self.predictions)[0] if self.predictions else None

    # ---------------- 查询（供工具调用） ----------------
    def slice_series(self, start=None, end=None) -> pd.DataFrame:
        if not self.ready:
            return pd.DataFrame()
        df = self.series
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            # 只给日期时，包含当天整天
            end_ts = pd.Timestamp(end)
            if end_ts == end_ts.normalize():
                end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
            df = df[df.index <= end_ts]
        return df

    def anomalies(self) -> pd.DataFrame:
        if self.anomaly_points is None:
            return pd.DataFrame()
        return self.anomaly_points


# --------------------------------------------------------------------------- #
# 导出 LLM 需要的两个产物（run_pipeline 会调用）
# --------------------------------------------------------------------------- #
def export_feature_importance(importance: pd.DataFrame, results_dir: str = RESULTS_DIR) -> str:
    """把特征重要性数值落盘（原来只有 PNG 图，模型读不了图）。"""
    path = os.path.join(results_dir, "feature_importance.csv")
    importance.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def export_anomaly_points(test_df: pd.DataFrame, masks: Dict[str, np.ndarray],
                          score: Optional[np.ndarray] = None,
                          results_dir: str = RESULTS_DIR) -> str:
    """把异常点整理成表：时刻、负荷、各检测器是否报警、异常分。"""
    out = pd.DataFrame({"load": test_df["load"].values}, index=test_df.index)
    for name, mask in masks.items():
        out[name] = np.asarray(mask, dtype=bool)
    if score is not None:
        out["IF异常分"] = np.asarray(score, dtype=float)
    out["报警数"] = out[[c for c in masks]].sum(axis=1)
    flagged = out[out["报警数"] > 0].copy()
    flagged.index.name = "时间"
    flagged = flagged.sort_values("报警数", ascending=False)
    path = os.path.join(results_dir, "anomaly_points.csv")
    flagged.to_csv(path, encoding="utf-8-sig", float_format="%.4f")
    return path


# --------------------------------------------------------------------------- #
# 日报数据包
# --------------------------------------------------------------------------- #
def _peak_valley(s: pd.Series) -> dict:
    if len(s) == 0:
        return {}
    return {
        "峰值时刻": str(s.idxmax()), "峰值": round(float(s.max()), 2),
        "谷值时刻": str(s.idxmin()), "谷值": round(float(s.min()), 2),
        "均值": round(float(s.mean()), 2),
        "标准差": round(float(s.std()), 2),
    }


def temperature_load_relation(hub: DataHub, days: int = 60) -> dict:
    """温度与负荷的关系：相关系数 + 分温度段的平均负荷（比单看斜率更稳）。"""
    if not hub.ready or "temp" not in hub.series.columns:
        return {}
    df = hub.series.tail(24 * days).dropna()
    if len(df) < 48:
        return {}
    corr = float(df["load"].corr(df["temp"]))
    bins = [(-50, 10), (10, 18), (18, 25), (25, 99)]
    seg = []
    for lo, hi in bins:
        sub = df[(df["temp"] >= lo) & (df["temp"] < hi)]
        if len(sub) >= 12:
            seg.append({"温度区间": "%d~%d℃" % (lo, hi),
                        "平均负荷": round(float(sub["load"].mean()), 2),
                        "样本小时": int(len(sub))})
    return {"统计区间": "%s ~ %s（近 %d 天）" % (df.index.min(), df.index.max(), days),
            "相关系数": round(corr, 3), "分温度段平均负荷": seg}


def build_daily_brief(hub: DataHub, hours: int = 24) -> dict:
    """组装日报用的紧凑数据包（每个数字都是算好的）。"""
    brief: dict = {"数据概况": hub.describe()}
    if not hub.ready:
        return brief

    recent = hub.series.tail(hours)
    prev = hub.series.tail(hours * 2).head(hours)
    load_brief = _peak_valley(recent["load"])
    if len(prev) == len(recent) and prev["load"].mean():
        delta = (recent["load"].mean() - prev["load"].mean()) / prev["load"].mean() * 100
        load_brief["与前24小时相比"] = "%+.1f%%" % delta
    brief["最近24小时实际负荷"] = load_brief

    if "temp" in hub.series.columns:
        t = recent["temp"]
        brief["最近24小时温度"] = {
            "均值": round(float(t.mean()), 1), "最高": round(float(t.max()), 1),
            "最低": round(float(t.min()), 1),
            "最高时刻": str(t.idxmax()), "最低时刻": str(t.idxmin()),
        }
        brief["温度与负荷关系"] = temperature_load_relation(hub)

    fc = hub.latest_forecast()
    if fc:
        brief["最近一次24小时预测"] = {
            "模型": fc["模型"], "预测起点": fc["起点"],
            "平均绝对误差_kW": fc["平均绝对误差"],
            "预测峰值": round(max(fc["预测值"]), 2) if fc["预测值"] else None,
            "实际峰值": round(max(fc["实际值"]), 2) if fc["实际值"] else None,
        }

    if hub.model_table is not None:
        cols = [c for c in ["MAE", "RMSE", "WAPE(%)", "R2"] if c in hub.model_table.columns]
        brief["模型表现"] = {
            str(k): {c: round(float(v), 3) for c, v in row[cols].items()}
            for k, row in hub.model_table.iterrows()
        }
        best = hub.best_model_name()
        brief["推荐模型"] = best

    if hub.feature_importance is not None and len(hub.feature_importance):
        top = hub.feature_importance.head(8)
        cols = list(top.columns)
        name_col = "feature" if "feature" in cols else cols[0]
        val_col = "importance" if "importance" in cols else cols[-1]
        brief["特征重要性Top8"] = [[str(r[name_col]), round(float(r[val_col]), 4)]
                                   for _, r in top.iterrows()]

    if hub.anomaly_points is not None and len(hub.anomaly_points):
        ap = hub.anomaly_points
        brief["检测到的异常点"] = {
            "总数": int(len(ap)),
            "按检测器": {c: int(ap[c].sum()) for c in ap.columns
                       if c not in ("load", "IF异常分", "报警数")},
            "最严重的8个": [
                {"时间": str(i), "负荷_kW": round(float(r["load"]), 2),
                 "报警检测器数": int(r["报警数"])}
                for i, r in ap.head(8).iterrows()
            ],
        }
    if hub.anomaly_bench is not None:
        brief["异常检测器评测"] = {
            str(k): {c: round(float(v), 3) for c, v in row.items() if pd.notna(v)}
            for k, row in hub.anomaly_bench.iterrows()
        }
    return brief
