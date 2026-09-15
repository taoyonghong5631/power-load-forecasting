# -*- coding: utf-8 -*-
"""把 results/*.csv 渲染成 PNG 表格图，输出到 results/figures/（README 直接引用）。

前置：先跑 ``python run_pipeline.py`` 生成 CSV。

用法:
    python scripts/make_tables.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 要渲染的表格：CSV 文件名 -> 输出 PNG 名 -> 标题
TABLES = [
    ("model_comparison.csv", "00_model_comparison_table.png", "多模型滚动起点对比"),
    ("ablation_features.csv", "00b_ablation_table.png", "特征工程消融"),
    ("anomaly_benchmark.csv", "00c_anomaly_table.png", "异常检测器对比"),
]

# 表格太宽会看不清，只保留关键列
TABLE_COLUMNS = {
    "model_comparison.csv": ["MAE", "RMSE", "WAPE(%)", "sMAPE(%)", "尖峰MAE", "R2", "耗时(s)"],
}


def render_table(csv_name: str, out_name: str, title: str) -> bool:
    path = os.path.join(RESULTS, csv_name)
    if not os.path.exists(path):
        print("跳过 %s（不存在，请先运行 run_pipeline.py）" % csv_name)
        return False

    df = pd.read_csv(path, index_col=0)
    wanted = TABLE_COLUMNS.get(csv_name)
    keep = ([c for c in wanted if c in df.columns] if wanted
            else [c for c in df.columns if not c.endswith("_std")])
    df = df[keep].astype(float).round(3)

    fig, ax = plt.subplots(figsize=(1.35 * len(keep) + 3.2, 0.55 * len(df) + 2.0), dpi=110)
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
    out = os.path.join(FIGS, out_name)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("已生成 %s" % out)
    return True


def main() -> int:
    done = sum(render_table(*t) for t in TABLES)
    print("\n共生成 %d/%d 张表格图，输出目录：%s" % (done, len(TABLES), FIGS))
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
