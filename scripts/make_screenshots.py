# -*- coding: utf-8 -*-
"""把 CSV 结果渲染成 PNG 表格图，并把关键图整理到 results/screenshots/。

用途：README 里直接引用这些图片。前置：先跑 ``python run_pipeline.py``。

用法:
    python scripts/make_screenshots.py
"""
from __future__ import annotations

import glob
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
SHOTS = os.path.join(RESULTS, "screenshots")
os.makedirs(SHOTS, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def render_table(csv_name: str, out_name: str, title: str, fmt: str = "{:.3f}") -> bool:
    path = os.path.join(RESULTS, csv_name)
    if not os.path.exists(path):
        print("跳过 %s（不存在）" % csv_name)
        return False
    df = pd.read_csv(path, index_col=0)
    # 只保留有意义的列，避免表格太宽
    keep = [c for c in df.columns if not c.endswith("_std")]
    df = df[keep].astype(float).round(3)
    fig, ax = plt.subplots(figsize=(1.35 * len(keep) + 3.2, 0.55 * len(df) + 2.0),
                           dpi=110)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    table = ax.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dbe3")
        if r == 0:
            cell.set_facecolor("#1f2d5a")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == -1:
            cell.set_facecolor("#eef2f7")
    fig.tight_layout()
    out = os.path.join(SHOTS, out_name)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("已生成 %s" % out)
    return True


KEY_FIGURES = [
    ("01_model_comparison.png", "01_model_comparison.png"),
    ("05_sample_forecast.png", "02_sample_forecast.png"),
    ("06_overview.png", "03_train_test_split.png"),
    ("04_error_profile.png", "04_error_profile.png"),
    ("08_feature_ablation.png", "05_feature_ablation.png"),
    ("09_feature_importance.png", "06_feature_importance.png"),
    ("10_temperature_scatter.png", "07_temperature_scatter.png"),
    ("11_anomaly_benchmark.png", "08_anomaly_benchmark.png"),
    ("13_anomaly_recall_by_kind.png", "09_anomaly_recall_by_kind.png"),
    ("12_anomaly_timeline.png", "10_anomaly_timeline.png"),
]


def main() -> int:
    render_table("model_comparison.csv", "00_model_comparison_table.png",
                 "多模型滚动起点对比")
    render_table("ablation_features.csv", "00b_ablation_table.png", "特征工程消融")
    render_table("anomaly_benchmark.csv", "00c_anomaly_table.png", "异常检测器对比")

    copied = 0
    for src, dst in KEY_FIGURES:
        p = os.path.join(FIGS, src)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(SHOTS, dst))
            copied += 1
    print("复制了 %d 张关键图到 %s" % (copied, SHOTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
