# -*- coding: utf-8 -*-
"""一键跑完整实验：对比模型 + 特征工程消融 + 异常检测升级，产出所有表格和图。

用法示例
--------
    python run_pipeline.py                 # 完整实验（默认 MT_001，4 年小时级数据）
    python run_pipeline.py --quick         # 冒烟测试，几分钟跑完
    python run_pipeline.py --synthetic     # 无数据/无网络时用合成数据演示
    python run_pipeline.py --models xgb,arima --no-temperature

产物
----
    results/model_comparison.csv     三种模型的滚动起点对比表
    results/ablation_features.csv    日期特征/温度特征的消融表
    results/anomaly_benchmark.csv    3σ vs Isolation Forest 的检测指标
    results/figures/*.png|html       所有图表（PNG 供 README，HTML 可交互）
    results/models/*.joblib          训练好的模型（Streamlit 直接复用）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Windows 控制台默认 GBK，中文/希腊字母会报 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src import metrics as M
from src import plots
from src.anomaly import (benchmark_detectors, global_3sigma_mask,
                         iforest_predict, fit_isolation_forest,
                         rolling_3sigma_mask)
from src.config import RESULTS_DIR, get_config
from src.data import get_dataset, synthetic_dataset
from src.evaluate import evaluate_model, make_origins, results_to_table, train_end_index
from src.models import build_model
from src.registry import save_model


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78, flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="电力负荷预测完整实验流水线")
    ap.add_argument("--quick", action="store_true", help="快速模式（小样本、少轮数）")
    ap.add_argument("--synthetic", action="store_true", help="用合成数据离线演示")
    ap.add_argument("--client-col", type=int, default=1, help="使用第几列客户（默认 MT_001）")
    ap.add_argument("--data-file", default=None,
                    help="自定义原始 txt 路径（默认 data/LD2011_2014.txt）")
    ap.add_argument("--models",
                    default="naive,seasonal_naive,lstm,xgb_recursive,xgb_direct,arima",
                    help="要跑的模型，逗号分隔（naive=持久性基线，seasonal_naive=昨天同时刻）")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖 LSTM 训练轮数")
    ap.add_argument("--origins", type=int, default=None, help="滚动起点个数")
    ap.add_argument("--no-temperature", action="store_true", help="不使用温度外生变量")
    ap.add_argument("--no-calendar", action="store_true", help="不使用日期特征")
    ap.add_argument("--annual", default="auto", choices=["auto", "on", "off"],
                    help="是否使用 month 类年度特征：auto=训练跨度≥1年时启用")
    ap.add_argument("--auto-arima", action="store_true", help="用 AIC 小网格为 ARIMA 选阶")
    ap.add_argument("--no-cache", action="store_true", help="忽略小时级数据缓存")
    return ap.parse_args()


def load_data(cfg, args):
    if args.synthetic:
        df = synthetic_dataset(n_days=400, seed=cfg.task.seed)
        print("[data] 使用合成数据：%d 小时（%.0f 天）" % (len(df), len(df) / 24))
        return df
    df = get_dataset(cfg, use_temperature=not args.no_temperature,
                     use_cache=not args.no_cache)
    print("[data] 实际数据：%d 小时，%s ~ %s"
          % (len(df), df.index.min(), df.index.max()))
    return df


def main() -> int:
    args = parse_args()
    t_start = time.time()

    cfg = get_config(data__client_col=args.client_col)
    if args.data_file:
        cfg.data.txt_path = os.path.abspath(args.data_file)
    cfg.use_temperature = not args.no_temperature
    cfg.use_calendar = not args.no_calendar
    if args.quick:
        cfg.apply_quick()
    if args.epochs is not None:
        cfg.lstm.epochs = args.epochs
    if args.origins is not None:
        cfg.task.n_eval_origins = args.origins

    df = load_data(cfg, args)
    n_total = len(df)
    train_end = train_end_index(n_total, cfg.task.train_ratio)
    train_df = df.iloc[:train_end + 1]
    test_df = df.iloc[train_end + 1:]
    origins = make_origins(n_total, train_end, cfg.task.horizon,
                           cfg.task.eval_stride, cfg.task.n_eval_origins)
    print("[task] 训练 %d 小时 / 测试 %d 小时；滚动起点 %d 个，每个预测 %d 步"
          % (len(train_df), len(test_df), len(origins), cfg.task.horizon))

    span_days = (train_df.index[-1] - train_df.index[0]).days
    if args.annual == "auto":
        cfg.include_annual = span_days >= 365
        print("[task] 训练跨度 %.0f 天 -> 年度特征(month) %s"
              % (span_days, "启用" if cfg.include_annual else "禁用（不足一年）"))
    else:
        cfg.include_annual = (args.annual == "on")

    feature_groups = tuple(
        ["base"]
        + (["calendar"] if cfg.use_calendar else [])
        + (["temperature"] if cfg.use_temperature and "temp" in df.columns else [])
    )
    budget_hours = len(df) / 24.0
    if budget_hours < 10:      # 不足 10 天时，168h 滞后/温度滞后特征全是 NaN
        feature_groups = tuple(g for g in feature_groups if g != "temperature")
        print("[task] 数据较短（%.0f 天），自动精简特征组" % budget_hours)

    requested = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    results = {}
    trained = {}
    importance = None

    # ------------------------------------------------------------------ #
    # 1) 对比模型
    # ------------------------------------------------------------------ #
    banner("1/3 对比模型：LSTM vs XGBoost vs ARIMA（滚动起点 %d 个 × %d 步）"
           % (len(origins), cfg.task.horizon))

    model_specs = {
        "naive": ("naive", ("base",), "持久性 (t-1)"),
        "seasonal_naive": ("seasonal_naive", ("base",), "季节朴素 (t-24)"),
        "lstm": ("lstm", feature_groups, "LSTM"),
        "xgb": ("xgb", feature_groups, "XGBoost"),
        "xgb_recursive": ("xgb_recursive", feature_groups, "XGBoost (递归多步)"),
        "xgb_direct": ("xgb_direct", feature_groups, "XGBoost (直接多步)"),
        "arima": ("arima", ("base",), "ARIMA"),
    }
    for key in requested:
        if key not in model_specs:
            print("[skip] 未知模型 %s" % key)
            continue
        kind, groups, label = model_specs[key]
        t0 = time.time()
        print("\n-- 训练 %s ..." % label)
        model = build_model(kind, cfg, groups, label=label)
        if kind == "arima" and args.auto_arima:
            from src.models.arima import select_order
            order, seasonal = select_order(train_df["load"], cfg)
            cfg.arima.order, cfg.arima.seasonal_order = order, seasonal
        model.fit(train_df)
        train_seconds = time.time() - t0
        try:
            res = evaluate_model(model, df, origins, cfg.task.horizon)
        except Exception as exc:
            print("[error] %s 评估失败：%s: %s" % (label, type(exc).__name__, exc))
            continue
        res["metrics"]["train_seconds"] = train_seconds
        results[label] = res
        trained[label] = model
        # 落盘预测结果，供 scripts/make_gif.py、scripts/make_screenshots.py 复用
        np.savez_compressed(
            os.path.join(RESULTS_DIR, "pred_%s.npz" % label.replace(" ", "_")),
            preds=res["preds"], truths=res["truths"], origins=np.asarray(res["origins"]),
            stamps=np.array([str(ts[0]) for ts in res["timestamps"]]),
            index=np.array([str(t) for t in df.index]),
            load=df["load"].to_numpy(dtype=float),
        )
        if hasattr(model, "feature_importance"):
            importance = model.feature_importance(top_n=22)
        save_model(model, "%s_%s" % (label, cfg.fingerprint))
        print("-- %s 完成：MAE=%.3f  RMSE=%.3f  WAPE=%.2f%%  (训练 %.1fs)"
              % (label, res["metrics"]["MAE"], res["metrics"]["RMSE"],
                 res["metrics"]["WAPE"], train_seconds))

    if not results:
        print("[fatal] 没有任何模型跑成功")
        return 1

    table = results_to_table(results, baseline="LSTM" if "LSTM" in results else None)
    table.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"),
                 encoding="utf-8-sig", float_format="%.4f")
    banner("对比表（滚动起点平均）")
    print(table.round(3).to_string())

    # 图：对比柱状图 / 各起点稳定性 / 误差随步长
    plots.save(plots.metric_comparison_figure(table), "01_model_comparison")
    plots.save(plots.per_origin_figure(results, "MAE"), "02_per_origin_mae")
    best_name = min(results, key=lambda k: results[k]["metrics"]["MAE"])
    best = results[best_name]
    # 出图时用"最好的学习型模型"：朴素基线是一条平线，不适合当示例图，
    # 但表里仍然如实保留它的排名
    learned = {k: v for k, v in results.items()
               if not any(t in k for t in ("持久性", "季节朴素"))}
    fig_name = min(learned, key=lambda k: learned[k]["metrics"]["MAE"]) if learned else best_name
    fig_res = results[fig_name]
    print("[report] 表格最优: %s；示例图用最好的学习型模型: %s" % (best_name, fig_name))
    plots.save(plots.horizon_error_figure(fig_res["truths"], fig_res["preds"]),
               "03_horizon_error")
    plots.save(plots.error_profile_figure(fig_res["truths"], fig_res["preds"],
                                          fig_res["timestamps"]),
               "04_error_profile")

    # 单窗口示例图（挑误差中位数的那一个窗口，避免只展示最好看的一天）
    per_mae = np.array([m["MAE"] for m in fig_res["per_origin"]])
    pick = int(np.argsort(per_mae)[len(per_mae) // 2])
    origin = fig_res["origins"][pick]
    ts = fig_res["timestamps"][pick]
    hist = df["load"].iloc[origin + 1 - cfg.task.look_back:origin + 1]
    band = np.full(len(ts), float(np.nanstd(fig_res["truths"] - fig_res["preds"])))
    plots.save(plots.forecast_figure(
        ts, fig_res["truths"][pick], fig_res["preds"][pick],
        title="%s 单窗口预测（起点 %s）" % (fig_name, df.index[origin]),
        band=band, history=hist), "05_sample_forecast")
    plots.save(plots.overview_figure(df, train_end, ts, fig_res["truths"][pick],
                                     fig_res["preds"][pick]), "06_overview")
    if trained.get("LSTM") is not None and getattr(trained["LSTM"], "history", None):
        plots.save(plots.training_curve_figure(trained["LSTM"].history), "07_lstm_training")

    # ------------------------------------------------------------------ #
    # 2) 特征工程消融
    # ------------------------------------------------------------------ #
    banner("2/3 特征工程消融：滞后 → +日期 → +温度（XGBoost）")
    ablation_cfg = cfg
    levels = [("Lags only", ("base",)), ("+ Calendar", ("base", "calendar"))]
    if cfg.use_temperature and "temp" in df.columns:
        levels.append(("+ Calendar + Temperature", ("base", "calendar", "temperature")))
    elif cfg.use_temperature:
        print("[warn] 数据中没有温度列，跳过温度消融")
    else:
        print("[task] --no-temperature 已指定，仅对比是否加入日期特征")

    ablation = {}
    for label, groups in levels:
        t0 = time.time()
        print("\n-- %s ..." % label)
        mdl = build_model("xgb", ablation_cfg, groups, label=label)
        mdl.fit(train_df)
        res = evaluate_model(mdl, df, origins, cfg.task.horizon, verbose=False)
        res["metrics"]["train_seconds"] = time.time() - t0
        ablation[label] = res
        print("-- %s：MAE=%.3f  RMSE=%.3f  WAPE=%.2f%%"
              % (label, res["metrics"]["MAE"], res["metrics"]["RMSE"],
                 res["metrics"]["WAPE"]))
        if label.startswith("+ Calendar + Temperature"):
            importance = mdl.feature_importance(top_n=22)

    ab_table = results_to_table(ablation)
    ab_table["MAE相对提升(%)"] = [
        M.improvement(ablation[levels[max(0, i - 1)][0]]["metrics"]["MAE"],
                      ablation[levels[i][0]]["metrics"]["MAE"])
        if i > 0 else 0.0
        for i in range(len(levels))
    ]
    ab_table["累计提升(%)"] = [
        M.improvement(ablation[levels[0][0]]["metrics"]["MAE"],
                      ablation[levels[i][0]]["metrics"]["MAE"])
        for i in range(len(levels))
    ]
    ab_table.to_csv(os.path.join(RESULTS_DIR, "ablation_features.csv"),
                    encoding="utf-8-sig", float_format="%.4f")
    banner("特征消融表")
    print(ab_table.round(3).to_string())
    plots.save(plots.metric_comparison_figure(ab_table, "特征工程消融对比"),
               "08_feature_ablation")
    if importance is not None:
        plots.save(plots.feature_importance_figure(importance), "09_feature_importance")
    if "temp" in df.columns:
        sample = df.sample(min(4000, len(df)), random_state=cfg.task.seed)
        plots.save(plots.temperature_scatter_figure(sample), "10_temperature_scatter")

    # ------------------------------------------------------------------ #
    # 3) 异常检测升级
    # ------------------------------------------------------------------ #
    banner("3/3 异常检测：全局 3σ / 滚动 3σ / Isolation Forest")
    bm = cfg.anomaly
    values = test_df["load"].to_numpy(dtype=float)
    masks = {
        "3-sigma (global)": global_3sigma_mask(values, k=bm.sigma_k),
        "3-sigma (rolling 24h)": rolling_3sigma_mask(values, k=bm.sigma_k,
                                                     window=bm.rolling_window),
    }
    iforest, cols = fit_isolation_forest(train_df, cfg)
    mask_if, _ = iforest_predict(iforest, test_df, cols, window=bm.rolling_window)
    masks["Isolation Forest"] = mask_if

    counts = pd.DataFrame({
        "检出异常点数": {k: int(np.sum(v)) for k, v in masks.items()},
        "占比(%)": {k: float(np.mean(v) * 100) for k, v in masks.items()},
    })
    print(counts.round(3).to_string())

    # 两个检测器的重合度
    a = masks["3-sigma (rolling 24h)"]
    b = masks["Isolation Forest"]
    inter = int(np.sum(a & b))
    union = int(np.sum(a | b))
    print("\n滚动 3σ 与 Isolation Forest 的重合：%d 个点，Jaccard=%.3f"
          % (inter, inter / union if union else 0.0))

    bench = benchmark_detectors(train_df, test_df, cfg, alert_rate=0.05)
    bench_df = pd.DataFrame(bench["table"]).T
    bench_df.index.name = "检测器"
    bench_df.to_csv(os.path.join(RESULTS_DIR, "anomaly_benchmark.csv"),
                    encoding="utf-8-sig", float_format="%.4f")
    banner("注入已知异常后的检测表现（PR-AUC/ROC-AUC 与阈值无关；@5%% 为同报警率）")
    print(bench_df.round(3).to_string())
    kind_df = pd.DataFrame(bench["by_kind_event"]).T
    kind_df.index.name = "检测器"
    print("\n事件级召回率（默认阈值；\"平台\"这类连续异常只要检出任一点即算命中）：")
    print(kind_df.round(3).to_string())
    kind_df.to_csv(os.path.join(RESULTS_DIR, "anomaly_recall_by_kind.csv"),
                   encoding="utf-8-sig", float_format="%.4f")
    plots.save(plots.detection_benchmark_figure(bench["table"]), "11_anomaly_benchmark")
    plots.save(plots.detection_benchmark_figure(
        bench["by_kind"], metrics=list(next(iter(bench["by_kind"].values())).keys()),
        title="分类型召回率（默认阈值）"), "13_anomaly_recall_by_kind")
    n_show = min(len(test_df), 24 * 30)
    plots.save(plots.anomaly_figure(
        test_df["load"].iloc[:n_show],
        {k: v[:n_show] for k, v in masks.items()},
        title="真实测试段上的异常检测结果（前 30 天）"),
        "12_anomaly_timeline")

    # ------------------------------------------------------------------ #
    # 汇总
    # ------------------------------------------------------------------ #
    summary = {
        "data": {
            "client_col": cfg.data.client_col,
            "hours": int(n_total),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "has_temperature": "temp" in df.columns,
            "source": "synthetic" if args.synthetic else os.path.basename(cfg.data.txt_path),
        },
        "task": {"look_back": cfg.task.look_back, "horizon": cfg.task.horizon,
                 "train_ratio": cfg.task.train_ratio, "origins": len(origins)},
        "models": {name: {k: v for k, v in res["metrics"].items()}
                   for name, res in results.items()},
        "ablation": {name: {k: v for k, v in res["metrics"].items()}
                     for name, res in ablation.items()},
        "anomaly": {"counts": counts.to_dict()["检出异常点数"],
                    "benchmark": bench["table"],
                    "alert_rate": bench["alert_rate"]},
        "elapsed_seconds": time.time() - t_start,
    }
    with open(os.path.join(RESULTS_DIR, "metrics_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=float)
    cfg.save(os.path.join(RESULTS_DIR, "run_config.json"))

    banner("全部完成，用时 %.1f 分钟" % ((time.time() - t_start) / 60))
    print("结果目录: %s" % RESULTS_DIR)
    print("下一步: streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
