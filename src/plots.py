# -*- coding: utf-8 -*-
"""所有交互式图表（Plotly）。Streamlit 界面直接复用这里的函数。"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .config import FIG_DIR

TEMPLATE = "plotly_white"
COLOR_TRUE = "#1f2d5a"
COLOR_PRED = "#e4572e"
COLOR_BAND = "rgba(228,87,46,0.18)"
MODEL_COLORS = ["#e4572e", "#17bebb", "#2e86ab", "#f5a623", "#7b2cbf", "#4c956c"]


def _finish(fig: go.Figure, title: str, height: int = 460) -> go.Figure:
    # 标题左对齐、图例右上，避免长标题和图例挤在一起
    fig.update_layout(
        template=TEMPLATE,
        title=dict(text=title, x=0.0, xanchor="left", y=0.97, yanchor="top"),
        height=height,
        margin=dict(l=60, r=30, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1.0, xanchor="right"),
    )
    return fig


def save(fig: go.Figure, name: str, png: bool = True, html: bool = True) -> Dict[str, str]:
    """保存图表到 results/figures/。PNG 用 kaleido，失败不影响主流程。"""
    paths = {}
    if html:
        path = os.path.join(FIG_DIR, name + ".html")
        fig.write_html(path, include_plotlyjs="cdn")
        paths["html"] = path
    if png:
        try:
            path = os.path.join(FIG_DIR, name + ".png")
            fig.write_image(path, width=1280, height=max(420, int(fig.layout.height or 460)))
            paths["png"] = path
        except Exception as exc:  # 没有 kaleido/chrome 时降级
            print("[plots] PNG 导出失败(%s)，仅保留 HTML" % type(exc).__name__)
    return paths


# --------------------------------------------------------------------------- #
# 预测结果
# --------------------------------------------------------------------------- #
def forecast_figure(index, y_true, y_pred, title: str = "24 小时负荷预测",
                    band: Optional[np.ndarray] = None, history: Optional[pd.Series] = None
                    ) -> go.Figure:
    """单窗口预测曲线 + 95% 区间 + 可选历史段。"""
    fig = go.Figure()
    if history is not None and len(history):
        fig.add_trace(go.Scatter(x=history.index, y=history.values, name="历史负荷",
                                 line=dict(color="#9aa5b1", width=1.6)))
    fig.add_trace(go.Scatter(x=index, y=y_true, name="真实值",
                             line=dict(color=COLOR_TRUE, width=2.6)))
    fig.add_trace(go.Scatter(x=index, y=y_pred, name="预测值",
                             line=dict(color=COLOR_PRED, width=2.6, dash="dot")))
    if band is not None and len(band):
        upper, lower = y_pred + band, y_pred - band
        fig.add_trace(go.Scatter(x=index, y=upper, name="95% 置信区间",
                                 line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=index, y=lower, name="95% 置信区间",
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor=COLOR_BAND))
    fig.update_yaxes(title_text="负荷 (kW)")
    return _finish(fig, title)


def overview_figure(df: pd.DataFrame, train_end: int, pred_index, y_true, y_pred,
                    title: str = "训练/测试划分与预测片段") -> go.Figure:
    """全序列概览：训练段、测试段与某个预测窗口的位置。"""
    fig = go.Figure()
    train = df.iloc[:train_end + 1]
    test = df.iloc[train_end + 1:]
    fig.add_trace(go.Scatter(x=train.index, y=train["load"], name="训练段",
                             line=dict(color="#9aa5b1", width=1)))
    fig.add_trace(go.Scatter(x=test.index, y=test["load"], name="测试段",
                             line=dict(color=COLOR_TRUE, width=1)))
    fig.add_trace(go.Scatter(x=pred_index, y=y_true, name="真实值（预测窗口）",
                             line=dict(color="#17bebb", width=2.6)))
    fig.add_trace(go.Scatter(x=pred_index, y=y_pred, name="预测值",
                             line=dict(color=COLOR_PRED, width=2.6, dash="dot")))
    # 注意：不能用 add_vline(annotation_text=...)，plotly 会去对 Timestamp 求和，
    # 与新版 pandas 不兼容；改用 shape + annotation 手工画。
    split = df.index[train_end]
    fig.add_shape(type="line", x0=split, x1=split, y0=0, y1=1, yref="paper",
                  line=dict(color="#e4572e", dash="dash", width=1.5))
    fig.add_annotation(x=split, y=1.02, yref="paper", showarrow=False,
                       text="训练/测试分界", font=dict(color="#e4572e"))
    fig.update_yaxes(title_text="负荷 (kW)")
    return _finish(fig, title, height=520)


def error_profile_figure(truths: np.ndarray, preds: np.ndarray,
                         stamps: Sequence[pd.DatetimeIndex],
                         title: str = "误差结构分析") -> go.Figure:
    """残差分布 + 误差随时间 + 按小时的平均误差。"""
    err = (truths - preds).ravel()
    err = err[~np.isnan(err)]
    err_all = (truths - preds)
    mask = ~np.isnan(err_all)
    # stamps 已经是每个窗口的 24 个时间戳，展平后与 err_all.ravel() 一一对应
    hours_flat = np.concatenate([s.hour.to_numpy() for s in stamps])[mask.ravel()]
    err_flat = err_all[mask]

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("残差分布", "残差自相关(前 48 阶)",
                                        "按小时的平均绝对误差", "预测 vs 真实散点"),
                        specs=[[{}, {}], [{}, {}]])
    fig.add_trace(go.Histogram(x=err, nbinsx=40, name="残差", marker_color="#2e86ab"),
                  row=1, col=1)

    acf_vals = [1.0]
    e = err - err.mean()
    denom = np.dot(e, e) + 1e-12
    for lag in range(1, 49):
        acf_vals.append(float(np.dot(e[:-lag], e[lag:]) / denom))
    fig.add_trace(go.Bar(x=list(range(49)), y=acf_vals, name="ACF",
                         marker_color="#17bebb"), row=1, col=2)

    hour_err = pd.Series(np.abs(err_flat)).groupby(hours_flat).mean()
    fig.add_trace(go.Bar(x=hour_err.index, y=hour_err.values, name="MAE",
                         marker_color="#e4572e"), row=2, col=1)

    fig.add_trace(go.Scatter(x=preds.ravel(), y=truths.ravel(), mode="markers",
                             name="样本", marker=dict(size=4, color="#1f2d5a", opacity=0.4)),
                  row=2, col=2)
    lo = float(np.nanmin([np.nanmin(truths), np.nanmin(preds)]))
    hi = float(np.nanmax([np.nanmax(truths), np.nanmax(preds)]))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], name="y=x",
                             line=dict(color="#e4572e", dash="dash")), row=2, col=2)
    fig.update_xaxes(title_text="小时", row=2, col=1)
    fig.update_yaxes(title_text="MAE (kW)", row=2, col=1)
    fig.update_xaxes(title_text="预测值", row=2, col=2)
    fig.update_yaxes(title_text="真实值", row=2, col=2)
    fig = _finish(fig, title, height=800)
    # 图例放到底部，避免压住第一行子图标题
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.08, x=0),
                      margin=dict(l=60, r=30, t=70, b=90))
    return fig


def metric_comparison_figure(table: pd.DataFrame,
                             title: str = "模型误差对比") -> go.Figure:
    """多指标分组柱状图（越小越好）。"""
    metrics = [c for c in ["MAE", "RMSE", "MAPE(%)", "sMAPE(%)", "尖峰MAE"] if c in table.columns]
    fig = go.Figure()
    for i, m in enumerate(metrics):
        fig.add_trace(go.Bar(x=table.index, y=table[m], name=m,
                             marker_color=MODEL_COLORS[i % len(MODEL_COLORS)],
                             text=np.round(table[m].to_numpy(), 2),
                             textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="误差（越小越好）")
    return _finish(fig, title, height=480)


def per_origin_figure(results: Dict[str, Dict[str, object]],
                      metric: str = "MAE") -> go.Figure:
    """每个预测起点的误差折线，看模型稳定性。"""
    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        vals = [m[metric] for m in res["per_origin"]]
        fig.add_trace(go.Scatter(x=list(range(1, len(vals) + 1)), y=vals, name=name,
                                 mode="lines+markers",
                                 line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)])))
    fig.update_xaxes(title_text="预测起点序号")
    fig.update_yaxes(title_text="%s (kW)" % metric)
    return _finish(fig, "各预测起点的 %s 对比" % metric)


def horizon_error_figure(truths: np.ndarray, preds: np.ndarray,
                         title: str = "误差随预测步长的增长") -> go.Figure:
    """第 1 步到第 24 步的 MAE —— 看递归预测的误差累积。"""
    mae_by_step = np.nanmean(np.abs(truths - preds), axis=0)
    fig = go.Figure(go.Bar(x=list(range(1, len(mae_by_step) + 1)), y=mae_by_step,
                           marker_color="#2e86ab",
                           text=np.round(mae_by_step, 1), textposition="outside"))
    fig.update_xaxes(title_text="预测步长（小时）")
    fig.update_yaxes(title_text="MAE (kW)")
    return _finish(fig, title)


# --------------------------------------------------------------------------- #
# 异常检测
# --------------------------------------------------------------------------- #
def anomaly_figure(series: pd.Series, masks: Dict[str, np.ndarray],
                   title: str = "异常检测结果对比") -> go.Figure:
    """同一条负荷曲线，不同检测器标出的异常点。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="负荷",
                             line=dict(color="#9aa5b1", width=1.2)))
    symbols = ["circle", "x", "triangle-up", "square"]
    for i, (name, mask) in enumerate(masks.items()):
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=series.index[mask], y=series.values[mask], name=name, mode="markers",
            marker=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)], size=8,
                        symbol=symbols[i % len(symbols)],
                        line=dict(width=1, color="white"))))
    fig.update_yaxes(title_text="负荷 (kW)")
    return _finish(fig, title)


def detection_benchmark_figure(bench: Dict[str, Dict[str, float]],
                               metrics: Sequence[str] = ("PR_AUC", "ROC_AUC", "F1@5%"),
                               title: str = "异常检测器在注入异常上的表现") -> go.Figure:
    """阈值无关指标 + 同工作点 F1 的对比。"""
    names = list(bench.keys())
    fig = go.Figure()
    for i, metric in enumerate(metrics):
        vals = [bench[n].get(metric, float("nan")) for n in names]
        fig.add_trace(go.Bar(
            x=names, y=vals, name=metric,
            marker_color=MODEL_COLORS[i % len(MODEL_COLORS)],
            text=[("" if v != v else round(v, 3)) for v in vals],
            textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="分数（越高越好）", range=[0, 1.2])
    return _finish(fig, title)


def feature_importance_figure(imp: pd.DataFrame,
                              title: str = "XGBoost 特征重要性") -> go.Figure:
    imp = imp.iloc[::-1]
    colors = ["#e4572e" if ("temp" in f or "hdd" in f or "cdd" in f)
              else ("#17bebb" if f in ("hour", "dow", "month", "is_weekend", "is_holiday")
                    or f.endswith(("_sin", "_cos")) else "#2e86ab")
              for f in imp["feature"]]
    fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h",
                           marker_color=colors))
    fig.update_xaxes(title_text="importance")
    return _finish(fig, title, height=max(420, 26 * len(imp) + 160))


# --------------------------------------------------------------------------- #
# 其它
# --------------------------------------------------------------------------- #
def training_curve_figure(history: Sequence[Dict[str, float]],
                          title: str = "LSTM 训练曲线") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[h["epoch"] for h in history], y=[h["train"] for h in history],
                             name="训练损失", line=dict(color="#2e86ab")))
    fig.add_trace(go.Scatter(x=[h["epoch"] for h in history], y=[h["val"] for h in history],
                             name="验证损失", line=dict(color="#e4572e")))
    fig.update_xaxes(title_text="epoch")
    fig.update_yaxes(title_text="MSE（标准化后）")
    return _finish(fig, title)


def temperature_scatter_figure(df: pd.DataFrame,
                               title: str = "负荷-温度关系") -> go.Figure:
    """负荷对温度呈"V 型"（制冷+采暖），这是加温度特征的理论依据。"""
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df["temp"], y=df["load"], mode="markers", name="样本",
        marker=dict(size=3, color=df.index.hour, colorscale="Turbo", opacity=0.5,
                    colorbar=dict(title="小时"))))
    fig.update_xaxes(title_text="温度 (℃)")
    fig.update_yaxes(title_text="负荷 (kW)")
    return _finish(fig, title)


def series_overview_figure(df: pd.DataFrame, train_ratio: float | None = None,
                           title: str = "负荷序列概览") -> go.Figure:
    """纯数据概览：负荷曲线（+温度副轴），可选标出训练/测试分界。"""
    fig = make_subplots(specs=[[{"secondary_y": "temp" in df.columns}]])
    fig.add_trace(go.Scatter(x=df.index, y=df["load"], name="负荷",
                             line=dict(color="#1f2d5a", width=1.2)), secondary_y=False)
    if "temp" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["temp"], name="温度",
                                 line=dict(color="#e4572e", width=1.1)), secondary_y=True)
        fig.update_yaxes(title_text="温度 (℃)", secondary_y=True)
    if train_ratio:
        split = df.index[min(len(df) - 1, int(len(df) * train_ratio))]
        fig.add_shape(type="line", x0=split, x1=split, y0=0, y1=1, yref="paper",
                      line=dict(color="#e4572e", dash="dash", width=1.5))
        fig.add_annotation(x=split, y=1.03, yref="paper", showarrow=False,
                           text="训练/测试分界", font=dict(color="#e4572e"))
    fig.update_yaxes(title_text="负荷 (kW)", secondary_y=False)
    fig = _finish(fig, title, height=460)
    return fig
