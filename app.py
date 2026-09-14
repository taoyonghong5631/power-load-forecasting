# -*- coding: utf-8 -*-
"""Streamlit 交互界面：上传数据 → 训练/预测 → 看 Plotly 图 → 异常检测。

运行:
    streamlit run app.py
"""
from __future__ import annotations

import io
import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src import plots
from src.anomaly import (benchmark_detectors, build_anomaly_features,
                         fit_isolation_forest, iforest_predict,
                         rolling_3sigma_mask, sigma_scores)
from src.config import RESULTS_DIR, get_config
from src.data import get_dataset, parse_uploaded_file, synthetic_dataset
from src.evaluate import evaluate_model, make_origins, results_to_table, train_end_index
from src.models import build_model
from src.registry import load_model

st.set_page_config(page_title="电力负荷短期预测与异常检测", page_icon="⚡", layout="wide")


# --------------------------------------------------------------------------- #
# 缓存的数据加载
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_builtin(client_col: int, use_temperature: bool):
    cfg = get_config(data__client_col=client_col)
    cfg.use_temperature = use_temperature
    try:
        df = get_dataset(cfg, use_temperature=use_temperature)
        return df, None
    except Exception as exc:                     # 没数据/没网络时自动兜底
        return synthetic_dataset(400, seed=cfg.task.seed), "%s: %s" % (type(exc).__name__, exc)


@st.cache_data(show_spinner=False)
def load_uploaded(payload: bytes, name: str) -> pd.DataFrame:
    return parse_uploaded_file(io.BytesIO(payload), name)


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    span = (df.index.max() - df.index.min()).total_seconds() / 3600
    return pd.DataFrame({
        "值": ["%d" % len(df),
               str(df.index.min()), str(df.index.max()),
               "%d 小时（%.1f 天）" % (span, span / 24),
               "%.2f" % df["load"].mean(), "%.2f" % df["load"].std(),
               "%.2f" % df["load"].min(), "%.2f" % df["load"].max(),
               "有" if "temp" in df.columns else "无"],
    }, index=["样本数", "开始时间", "结束时间", "跨度", "平均负荷 (kW)",
              "负荷标准差", "最小负荷", "最大负荷", "温度列"])


# --------------------------------------------------------------------------- #
# 侧边栏
# --------------------------------------------------------------------------- #
st.sidebar.title("⚡ 负荷预测控制台")

with st.sidebar:
    st.subheader("1. 数据源")
    # 本地还没有原始数据时默认用合成数据，避免一进界面就静默触发 40 分钟下载
    has_builtin = os.path.exists(get_config().data.txt_path)
    source = st.radio("选择数据", ["内置数据集 (UCI)", "上传文件", "合成演示数据"],
                      index=0 if has_builtin else 2, label_visibility="collapsed")
    if source == "内置数据集 (UCI)" and not has_builtin:
        st.warning("本地没有 `data/LD2011_2014.txt`，点击运行会先自动下载"
                   "约 261 MB 的 zip（解压后 711 MB，UCI 服务器很慢，可能要 1 小时以上）。"
                   "想立刻体验请选『合成演示数据』。")
    uploaded = None
    if source == "上传文件":
        uploaded = st.file_uploader("上传 CSV/TXT（时间列 + 负荷列）",
                                    type=["csv", "txt"])
        st.caption("支持 UCI 原始格式（; 分隔、, 小数点）和自己的两列 CSV；"
                   "少于 10 天的数据会自动放宽特征。")
    client_col = st.number_input("客户编号（1 = MT_001）", 1, 370, 1, step=1,
                                 disabled=source != "内置数据集 (UCI)")

    st.subheader("2. 任务设置")
    look_back = st.slider("输入窗口 look_back（小时）", 12, 168, 24, step=12)
    horizon = st.slider("预测步长（小时）", 6, 72, 24, step=6)
    train_ratio = st.slider("训练集比例", 0.5, 0.95, 0.8, step=0.05)

    st.subheader("3. 特征与模型")
    use_temperature = st.checkbox("加入温度特征", True)
    use_calendar = st.checkbox("加入日期特征", True)
    model_names = st.multiselect(
        "参与对比的模型",
        ["持久性 (t-1)", "季节朴素 (t-24)", "LSTM", "XGBoost", "ARIMA"],
        default=["持久性 (t-1)", "XGBoost", "ARIMA"],
        help="朴素基线必须一起看：本项目 MT_001 长时间停在同一个读数上，持久性基线很强。")

    st.subheader("4. 实验规模")
    quick = st.checkbox("快速模式（少轮数、少起点）", True)
    n_origins = st.slider("滚动预测起点数", 1, 30, 6, disabled=not quick)
    epochs = st.slider("LSTM 训练轮数", 3, 60, 8, step=1, disabled=not quick)
    use_cached = st.checkbox("复用 results/models 里的已训练模型", False,
                             help="只建议在『内置数据集』且参数未改动时勾选，"
                                  "否则会用到与当前数据不匹配的旧模型。")

    st.subheader("5. 异常检测")
    sigma_k = st.slider("3σ 的 k", 1.5, 5.0, 3.0, step=0.5)
    contamination = st.slider("Isolation Forest 预期异常比例", 0.001, 0.05, 0.01,
                              step=0.001, format="%.3f")

    run = st.button("🚀 开始训练与预测", type="primary", width="stretch")


# --------------------------------------------------------------------------- #
# 主区域
# --------------------------------------------------------------------------- #
st.title("电力负荷短期预测与异常检测")
st.caption("朴素基线 / LSTM / XGBoost(递归+直接) / ARIMA 滚动起点对比 · "
           "日期与温度特征消融 · 3σ vs Isolation Forest")

cfg = get_config(data__client_col=int(client_col))
cfg.use_temperature = use_temperature
cfg.use_calendar = use_calendar
cfg.task.look_back = int(look_back)
cfg.task.horizon = int(horizon)
cfg.task.train_ratio = float(train_ratio)
cfg.anomaly.sigma_k = float(sigma_k)
cfg.anomaly.contamination = float(contamination)
if quick:
    cfg.task.n_eval_origins = int(n_origins)
    cfg.task.eval_stride = max(24, int(horizon))
    cfg.lstm.epochs = int(epochs)
    cfg.xgb.n_estimators = 300
    cfg.arima.train_window = 1200

df, fallback_msg = None, None
try:
    if source == "上传文件":
        if uploaded is not None:
            df = load_uploaded(uploaded.getvalue(), uploaded.name)
            if use_temperature and "temp" not in df.columns:
                from src.data import load_temperature
                df = df.copy()
                df["temp"] = load_temperature(df.index, cfg)
        else:
            st.info("👈 先在左侧上传一个 CSV/TXT 文件。没有文件？可以先用『合成演示数据』或『内置数据集』。")
    elif source == "合成演示数据":
        df = synthetic_dataset(400, seed=cfg.task.seed)
    else:
        with st.spinner("加载内置数据集（首次会读取几百 MB 原始文件，需要一会）..."):
            df, fallback_msg = load_builtin(int(client_col), use_temperature)
except Exception as exc:
    st.error("数据加载失败：%s: %s" % (type(exc).__name__, exc))
    df = None

if fallback_msg:
    st.warning("内置数据不可用（%s），已自动切换为合成演示数据。" % fallback_msg)

if df is not None and len(df) > 48:
    _te = train_end_index(len(df), cfg.task.train_ratio)
    _span_days = (df.index[_te] - df.index[0]).days
    cfg.include_annual = _span_days >= 365      # 训练不足一年时关掉 month 特征

if df is not None and len(df) > 0:
    tab_data, tab_forecast, tab_anomaly, tab_about = st.tabs(
        ["📊 数据概览", "📈 预测对比", "🚨 异常检测", "ℹ️ 说明"])

    # ------------------------------ 数据概览 ------------------------------ #
    with tab_data:
        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(plots.series_overview_figure(
                df, train_ratio=cfg.task.train_ratio, title="全序列概览"),
                width="stretch")
        with right:
            st.dataframe(dataset_summary(df), width="stretch")
            if "temp" in df.columns:
                st.plotly_chart(plots.temperature_scatter_figure(
                    df.sample(min(4000, len(df)), random_state=cfg.task.seed)),
                    width="stretch")
        st.line_chart(df["load"].iloc[-24 * 14:], height=220)

    # ------------------------------ 预测对比 ------------------------------ #
    with tab_forecast:
        if run:
            n_total = len(df)
            train_end = train_end_index(n_total, cfg.task.train_ratio)
            train_df, test_df = df.iloc[:train_end + 1], df.iloc[train_end + 1:]
            if len(test_df) < horizon + 1:
                st.error("测试段太短，请调小训练集比例或预测步长。")
            elif not model_names:
                st.error("请至少选择一个模型。")
            else:
                origins = make_origins(n_total, train_end, cfg.task.horizon,
                                       cfg.task.eval_stride, cfg.task.n_eval_origins)
                groups = tuple(["base"] + (["calendar"] if use_calendar else [])
                               + (["temperature"] if use_temperature and "temp" in df.columns
                                  else []))
                results, trained = {}, {}
                progress = st.progress(0.0, text="准备训练...")
                for i, name in enumerate(model_names):
                    progress.progress(i / len(model_names), text="训练 %s ..." % name)
                    kind = {"LSTM": "lstm", "XGBoost": "xgb", "ARIMA": "arima",
                            "持久性 (t-1)": "naive",
                            "季节朴素 (t-24)": "seasonal_naive"}[name]
                    g = ("base",) if kind == "arima" else groups
                    model = load_model("%s_%s" % (name, cfg.fingerprint)) if use_cached else None
                    if model is None:
                        model = build_model(kind, cfg, g, label=name).fit(train_df,
                                                                         verbose=False)
                    with st.spinner("评估 %s 的滚动起点预测..." % name):
                        results[name] = evaluate_model(model, df, origins,
                                                       cfg.task.horizon, verbose=False)
                    trained[name] = model
                progress.progress(1.0, text="完成")

                # 训练完成后按钮消失，用 session_state 保存结果
                st.session_state["results"] = results
                st.session_state["trained"] = trained
                st.session_state["origins"] = origins
                st.session_state["df_key"] = (len(df), str(df.index[-1]), horizon)

        results = st.session_state.get("results")
        current_key = (len(df), str(df.index[-1]), horizon)
        if results and st.session_state.get("df_key") != current_key:
            st.warning("当前展示的是**上一次运行**的结果（数据或预测步长已改动）。"
                       "请重新点击『开始训练与预测』。")
        if not results:
            st.info("👈 在左侧选择好参数后，点击『开始训练与预测』。"
                    "（也可以先运行 `python run_pipeline.py --quick` 生成完整报告）")
        else:
            table = results_to_table(results,
                                     baseline="XGBoost" if "XGBoost" in results else None)
            st.subheader("滚动起点对比表")
            st.dataframe(table.style.format("{:.3f}"), width="stretch")
            st.download_button("下载对比表 CSV",
                               table.to_csv().encode("utf-8-sig"),
                               file_name="model_comparison.csv", mime="text/csv")

            best_name = min(results, key=lambda k: results[k]["metrics"]["MAE"])
            best = results[best_name]
            c1, c2, c3 = st.columns(3)
            c1.metric("最优模型", best_name)
            c2.metric("MAE (kW)", "%.2f" % best["metrics"]["MAE"])
            c3.metric("MAPE (%)", "%.2f" % best["metrics"]["MAPE"])

            st.plotly_chart(plots.metric_comparison_figure(table),
                            width="stretch")

            pick = int(np.argsort([m["MAE"] for m in best["per_origin"]])
                       [len(best["per_origin"]) // 2])
            ts = best["timestamps"][pick]
            origin = best["origins"][pick]
            horizon = best["truths"].shape[1]      # 用实际窗口长度，避免和滑块不一致
            hist = df["load"].iloc[max(0, origin + 1 - cfg.task.look_back):origin + 1]
            band = np.full(len(ts), float(np.nanstd(best["truths"] - best["preds"])))
            st.plotly_chart(plots.forecast_figure(
                ts, best["truths"][pick], best["preds"][pick],
                title="%s：起点 %s 向后 %d 小时" % (best_name, df.index[origin], horizon),
                band=band, history=hist), width="stretch")

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plots.horizon_error_figure(best["truths"], best["preds"]),
                                width="stretch")
            with col2:
                st.plotly_chart(plots.per_origin_figure(results, "MAE"),
                                width="stretch")
            st.plotly_chart(plots.error_profile_figure(best["truths"], best["preds"],
                                                       best["timestamps"]),
                            width="stretch")

            xgb = trained.get("XGBoost")
            if xgb is not None and hasattr(xgb, "feature_importance"):
                st.plotly_chart(plots.feature_importance_figure(xgb.feature_importance(22)),
                                width="stretch")

    # ------------------------------ 异常检测 ------------------------------ #
    with tab_anomaly:
        if df is None:
            st.info("请先选择数据源。")
        else:
            n_total = len(df)
            train_end = train_end_index(n_total, cfg.task.train_ratio)
            train_df, test_df = df.iloc[:train_end + 1], df.iloc[train_end + 1:]
            values = test_df["load"].to_numpy(dtype=float)
            masks = {
                "3σ (全局)": sigma_scores(values, None) > cfg.anomaly.sigma_k,
                "3σ (滚动 24h)": rolling_3sigma_mask(values, k=cfg.anomaly.sigma_k,
                                                     window=cfg.anomaly.rolling_window),
            }
            # IF 训练开销不小，用 session_state 记住结果，避免每次拖动控件都重训
            if_key = (cfg.fingerprint, len(df), str(test_df.index[0]),
                      cfg.anomaly.contamination, cfg.anomaly.n_estimators)
            if st.session_state.get("if_key") != if_key:
                with st.spinner("训练 Isolation Forest ..."):
                    iforest, cols = fit_isolation_forest(train_df, cfg)
                    mask_if, score_if = iforest_predict(
                        iforest, test_df, cols, window=cfg.anomaly.rolling_window)
                st.session_state["if_key"] = if_key
                st.session_state["if_result"] = (mask_if, score_if)
            else:
                mask_if, score_if = st.session_state["if_result"]
            masks["Isolation Forest"] = mask_if

            c1, c2, c3 = st.columns(3)
            c1.metric("3σ(全局) 检出", int(masks["3σ (全局)"].sum()))
            c2.metric("3σ(滚动) 检出", int(masks["3σ (滚动 24h)"].sum()))
            c3.metric("Isolation Forest 检出", int(mask_if.sum()))

            n_show = min(len(test_df), 24 * 30)
            st.plotly_chart(plots.anomaly_figure(
                test_df["load"].iloc[:n_show],
                {k: v[:n_show] for k, v in masks.items()},
                title="异常检测结果（测试段前 30 天）"), width="stretch")

            a, b = masks["3σ (滚动 24h)"], mask_if
            union = int(np.sum(a | b))
            st.write("滚动 3σ 与 Isolation Forest 重合 %d 个点，Jaccard = %.3f"
                     % (int(np.sum(a & b)), (np.sum(a & b) / union) if union else 0.0))

            if st.button("跑注入异常基准测试（衡量各检测器 F1 / 召回）"):
                with st.spinner("注入已知异常并评估..."):
                    bench = benchmark_detectors(train_df, test_df, cfg, alert_rate=0.05)
                bench_df = pd.DataFrame(bench["table"]).T
                st.dataframe(bench_df.style.format("{:.3f}"), width="stretch")
                st.plotly_chart(plots.detection_benchmark_figure(bench["table"]),
                                width="stretch")
                kind_df = pd.DataFrame(bench["by_kind_event"]).T
                st.write("事件级召回率（默认阈值）")
                st.dataframe(kind_df.style.format("{:.3f}"), width="stretch")

            detail = pd.DataFrame({
                "负荷": test_df["load"], "IF异常分": score_if, "IF标记": mask_if,
                "3σ标记": masks["3σ (滚动 24h)"],
            })
            st.download_button("下载异常检测明细 CSV",
                               detail.to_csv().encode("utf-8-sig"),
                               file_name="anomaly_detail.csv", mime="text/csv")
            st.dataframe(detail[mask_if].head(50), width="stretch")

    # ------------------------------ 说明 ------------------------------ #
    with tab_about:
        st.markdown("""
### 这个界面做了什么

* **预测**：持久性 / 季节朴素两条朴素基线，加上 LSTM（递归多步）、XGBoost（递归多步与直接多步
  两种策略）、ARIMA/SARIMA（每个起点用末尾窗口重新拟合），在同一批滚动起点上预测同样的窗口，
  指标可直接比较。**朴素基线一定要一起看**：本项目 MT_001 长时间停在同一个读数上，
  持久性基线很强，模型打不过它是要如实报告的。
* **特征工程**：日期特征（小时/星期/月份/节假日 + sin/cos 周期编码，训练跨度不足一年会自动
  关掉 month 类年度项）与温度特征（当前温度、昨日同时刻温度、24h 均温、采暖度 HDD18、制冷度 CDD22）。
* **预测目标**：XGBoost / LSTM 默认预测增量 Δy（从"持久性"出发只学修正量），
  XGBoost 另用训练段末尾 10% 的时间序验证集做早停。
* **异常检测**：全局 3σ → 滚动 3σ → Isolation Forest（9 维特征，只在训练段拟合）。
* **指标**：以池化 WAPE / MAE / RMSE 为主，MAPE 只作参考（这台表计测试段有 23% 的点 < 1 kW，
  百分比误差会被近零点放大）。

### 上传文件格式

```
timestamp,load
2012-01-01 00:00:00,312.5
2012-01-01 01:00:00,298.1
```

也支持 UCI 原始格式（`;` 分隔、`,` 小数点），会自动取第二个客户列。
""")
        st.write("结果目录：`%s`" % RESULTS_DIR)
        if os.path.exists(os.path.join(RESULTS_DIR, "metrics_summary.json")):
            st.success("检测到 `results/metrics_summary.json`：已经跑过完整实验，"
                       "`results/` 里有全部对比表、图（PNG/HTML）和训练好的模型。")
