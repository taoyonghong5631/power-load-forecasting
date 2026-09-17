# -*- coding: utf-8 -*-
"""工具层：把项目里的查询能力包装成「大模型可以调用的函数」。

方向二（问答）和方向三（Agent 归因）共用这一层。模型只负责决定"调哪个工具、
传什么参数"，真正的数字全部由下面的 Python 代码算出来。

每个工具的返回值都是 JSON 可序列化的字典；画图类工具会额外把 figure 存进
``ToolBox.figures``，由界面取出来展示。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import Config
from .context import DataHub, temperature_load_relation

# --------------------------------------------------------------------------- #
# 给模型看的工具说明书（OpenAI function calling 格式）
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS: List[dict] = [
    {"type": "function", "function": {
        "name": "describe_dataset",
        "description": "查看当前有哪些数据可用：客户编号、时间范围、包含哪些结果产物。"
                       "不确定数据范围时先调这个。",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "query_load",
        "description": "查询某个时间段的负荷与温度统计（均值/峰值/谷值）。用于回答"
                       "\"某天或某周负荷怎么样\"这类问题。",
        "parameters": {"type": "object", "properties": {
            "start": {"type": "string", "description": "开始日期 YYYY-MM-DD，可省略"},
            "end": {"type": "string", "description": "结束日期 YYYY-MM-DD，可省略"},
            "agg": {"type": "string", "enum": ["hourly", "daily"],
                    "description": "hourly=逐小时明细（最多 72 条），daily=按天汇总"},
        }},
    }},
    {"type": "function", "function": {
        "name": "find_anomalies",
        "description": "查询异常负荷点排行（负荷骤降、尖峰、与同期明显不符的时刻）。",
        "parameters": {"type": "object", "properties": {
            "top_k": {"type": "integer", "description": "返回最严重的几个，默认 10"},
            "start": {"type": "string", "description": "只看这个日期之后，可省略"},
            "end": {"type": "string", "description": "只看这个日期之前，可省略"},
        }},
    }},
    {"type": "function", "function": {
        "name": "compare_models",
        "description": "返回各预测模型的误差指标（MAE/RMSE/WAPE/R²），用于回答"
                       "\"哪个模型最好\"或\"为什么某个时段偏差大\"。",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "get_feature_importance",
        "description": "返回模型最依赖的特征排行，用于回答\"什么因素影响负荷\"。",
        "parameters": {"type": "object", "properties": {
            "top_k": {"type": "integer", "description": "返回前几个，默认 10"},
        }},
    }},
    {"type": "function", "function": {
        "name": "get_temperature_relation",
        "description": "返回温度与负荷的相关系数，以及分温度段的平均负荷。用于回答"
                       "\"温度和负荷什么关系\"、\"升温 10 度负荷变化多少\"。",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "统计最近多少天，默认 60"},
        }},
    }},
    {"type": "function", "function": {
        "name": "plot_forecast",
        "description": "画出某个日期附近的「预测 vs 实际」对比图，并返回该窗口的误差。",
        "parameters": {"type": "object", "properties": {
            "date": {"type": "string", "description": "日期 YYYY-MM-DD"},
            "model": {"type": "string", "description": "模型名，可省略（默认用最优模型）"},
        }, "required": ["date"]},
    }},
    {"type": "function", "function": {
        "name": "plot_anomalies",
        "description": "画出最近一段时间的负荷曲线并标出异常点。",
        "parameters": {"type": "object", "properties": {
            "days": {"type": "integer", "description": "最近多少天，默认 14"},
        }},
    }},
]


def tool_names() -> List[str]:
    return [t["function"]["name"] for t in TOOL_SCHEMAS]


# --------------------------------------------------------------------------- #
# 工具箱
# --------------------------------------------------------------------------- #
class ToolBox:
    def __init__(self, hub: DataHub, cfg: Config):
        self.hub = hub
        self.cfg = cfg
        self.figures: Dict[str, object] = {}      # 名称 -> plotly Figure
        self.trace: List[dict] = []               # 调用记录，界面上可以展示

    def _log(self, name: str, args: dict, summary: str) -> None:
        self.trace.append({"工具": name, "参数": args, "结果摘要": summary})

    # ---------------- 各工具 ----------------
    def describe_dataset(self) -> dict:
        d = self.hub.describe()
        self._log("describe_dataset", {}, "样本 %d 小时" % d.get("小时级样本数", 0))
        return d

    def query_load(self, start: Optional[str] = None, end: Optional[str] = None,
                   agg: str = "daily") -> dict:
        df = self.hub.slice_series(start, end)
        if len(df) == 0:
            rng = self.hub.describe().get("时间范围") or ["?", "?"]
            return {"错误": "该时间段没有数据，可用范围是 %s ~ %s" % (rng[0], rng[1])}
        if agg == "hourly":
            sub = df.tail(72)
            rows = [{"时间": str(i), "负荷": round(float(r["load"]), 2),
                     **({"温度": round(float(r["temp"]), 1)} if "temp" in df.columns else {})}
                    for i, r in sub.iterrows()]
            out = {"聚合方式": "逐小时（最多 72 条）", "条数": len(rows), "明细": rows}
        else:
            g = df.resample("D")["load"].agg(["mean", "max", "min"])
            rows = [{"日期": str(i.date()), "均值": round(float(r["mean"]), 2),
                     "峰值": round(float(r["max"]), 2), "谷值": round(float(r["min"]), 2)}
                    for i, r in g.iterrows()]
            out = {"聚合方式": "按天", "天数": len(rows), "明细": rows}
        load = df["load"]
        out["区间统计"] = {
            "时间范围": [str(df.index.min()), str(df.index.max())],
            "均值": round(float(load.mean()), 2),
            "最高": "%.2f（%s）" % (load.max(), load.idxmax()),
            "最低": "%.2f（%s）" % (load.min(), load.idxmin()),
        }
        if "temp" in df.columns:
            out["温度"] = {"均值": round(float(df["temp"].mean()), 1),
                          "最高": round(float(df["temp"].max()), 1),
                          "最低": round(float(df["temp"].min()), 1)}
        self._log("query_load", {"start": start, "end": end, "agg": agg},
                  "%s ~ %s，均值 %.2f" % (out["区间统计"]["时间范围"][0],
                                       out["区间统计"]["时间范围"][1], load.mean()))
        return out

    def find_anomalies(self, top_k: int = 10, start: Optional[str] = None,
                       end: Optional[str] = None) -> dict:
        ap = self.hub.anomalies()
        if ap is None or len(ap) == 0:
            return {"说明": "还没有异常点清单，请先运行 python run_pipeline.py 生成 "
                            "results/anomaly_points.csv"}
        ap = ap.copy()
        ap.index = pd.to_datetime(ap.index)
        if start:
            ap = ap[ap.index >= pd.Timestamp(start)]
        if end:
            ap = ap[ap.index <= pd.Timestamp(end) + pd.Timedelta(days=1)]
        if len(ap) == 0:
            return {"说明": "该时间段内没有检测到异常点"}
        cols = [c for c in ap.columns if c not in ("load", "IF异常分", "报警数")]
        top = ap.sort_values(["报警数"] + cols, ascending=False).head(int(top_k))
        items = []
        for i, r in top.iterrows():
            item = {"时间": str(i), "负荷_kW": round(float(r["load"]), 2),
                    "报警的检测器": [c for c in cols if bool(r[c])]}
            if "IF异常分" in r and pd.notna(r["IF异常分"]):
                item["IF异常分"] = round(float(r["IF异常分"]), 4)
            items.append(item)
        out = {"异常点总数": int(len(ap)),
               "区间": [str(ap.index.min()), str(ap.index.max())],
               "最严重的%d个" % len(items): items}
        self._log("find_anomalies", {"top_k": top_k, "start": start, "end": end},
                  "共 %d 个异常点" % len(ap))
        return out

    def compare_models(self) -> dict:
        if self.hub.model_table is None:
            return {"说明": "还没有模型对比结果，请先运行 python run_pipeline.py"}
        cols = [c for c in ["MAE", "RMSE", "WAPE(%)", "sMAPE(%)", "尖峰MAE", "R2"]
                if c in self.hub.model_table.columns]
        table = self.hub.model_table[cols]
        out = {
            "指标说明": "MAE/RMSE 单位 kW（越小越好）；WAPE/sMAPE 单位 %；R² 越接近 1 越好",
            "各模型": {str(k): {c: round(float(v), 3) for c, v in row.items()}
                     for k, row in table.iterrows()},
            "最优学习型模型": self.hub.best_model_name(),
        }
        if "MAE" in table.columns:
            out["MAE最低的模型"] = str(table["MAE"].idxmin())
        self._log("compare_models", {}, "最优: %s" % out.get("MAE最低的模型"))
        return out

    def get_feature_importance(self, top_k: int = 10) -> dict:
        fi = self.hub.feature_importance
        if fi is None or len(fi) == 0:
            return {"说明": "还没有特征重要性文件（results/feature_importance.csv），"
                            "请先运行 python run_pipeline.py"}
        top = fi.head(int(top_k))
        cols = list(top.columns)
        name_col = "feature" if "feature" in cols else cols[0]
        val_col = "importance" if "importance" in cols else cols[-1]
        out = {"特征重要性Top": [[str(r[name_col]), round(float(r[val_col]), 4)]
                              for _, r in top.iterrows()],
               "说明": "数值是 XGBoost 中该特征参与分裂带来的增益占比，越大越重要"}
        self._log("get_feature_importance", {"top_k": top_k}, "共 %d 个特征" % len(fi))
        return out

    def get_temperature_relation(self, days: int = 60) -> dict:
        out = temperature_load_relation(self.hub, days=int(days))
        if not out:
            return {"说明": "数据里没有温度列，无法分析"}
        self._log("get_temperature_relation", {"days": days},
                  "相关系数 %.3f" % out.get("相关系数", 0))
        return out

    def plot_forecast(self, date: str, model: Optional[str] = None) -> dict:
        """找出离指定日期最近的那个预测窗口，画「预测 vs 实际」。"""
        from .. import plots
        if not self.hub.predictions:
            return {"说明": "没有预测结果文件（results/pred_*.npz）"}
        model = model or self.hub.best_model_name()
        if model not in self.hub.predictions:
            model = list(self.hub.predictions)[0]
        p = self.hub.predictions[model]
        stamps, preds, truths = p["stamps"], p["preds"], p["truths"]
        if stamps is None:
            return {"说明": "预测文件里没有时间戳"}
        target = pd.Timestamp(date)
        k = int(np.argmin(np.abs((pd.to_datetime(stamps) - target).days)))
        start = pd.Timestamp(stamps[k])
        times = pd.date_range(start, periods=preds.shape[1], freq="h")
        fig = plots.forecast_figure(times, truths[k], preds[k],
                                    title="%s 预测 vs 实际（起点 %s）" % (model, start))
        name = "forecast_%s" % start.strftime("%Y%m%d")
        self.figures[name] = fig
        mae = float(np.nanmean(np.abs(truths[k] - preds[k])))
        out = {"模型": model, "预测起点": str(start), "平均绝对误差_kW": round(mae, 3),
               "实际峰值_kW": round(float(np.nanmax(truths[k])), 2),
               "预测峰值_kW": round(float(np.nanmax(preds[k])), 2), "图表": name}
        self._log("plot_forecast", {"date": date, "model": model},
                  "起点 %s，MAE %.2f" % (start, mae))
        return out

    def plot_anomalies(self, days: int = 14) -> dict:
        from .. import plots
        if not self.hub.ready:
            return {"说明": "没有负荷数据"}
        seg = self.hub.series.tail(int(days) * 24)
        ap = self.hub.anomalies()
        masks = {}
        if ap is not None and len(ap):
            ap = ap.copy()
            ap.index = pd.to_datetime(ap.index)
            for c in ap.columns:
                if c in ("load", "IF异常分", "报警数"):
                    continue
                sel = ap.index.intersection(seg.index)
                m = pd.Series(False, index=seg.index)
                m.loc[sel] = ap.loc[sel, c].astype(bool).values
                if m.any():
                    masks[c] = m.values
        fig = plots.anomaly_figure(seg["load"], masks,
                                   title="最近 %d 天负荷与异常点" % days)
        name = "anomalies_last%d" % days
        self.figures[name] = fig
        self._log("plot_anomalies", {"days": days},
                  "命中 %d 个异常点" % sum(int(m.sum()) for m in masks.values()))
        return {"区间": [str(seg.index.min()), str(seg.index.max())],
                "异常点数": {k: int(v.sum()) for k, v in masks.items()},
                "图表": name}

    # ---------------- 调用分发 ----------------
    def call(self, name: str, arguments: dict) -> dict:
        """按名字调用工具（Agent 循环使用）。"""
        fn = getattr(self, name, None)
        if fn is None or name not in tool_names():
            return {"错误": "没有这个工具: %s" % name}
        try:
            return fn(**(arguments or {}))
        except TypeError as exc:
            return {"错误": "参数不对: %s" % exc}
        except Exception as exc:
            return {"错误": "%s: %s" % (type(exc).__name__, exc)}
