# -*- coding: utf-8 -*-
"""把 results/*.csv 里的实验结果自动写回 README.md 的结果区块。

README 里用一对标记划定区域，本脚本只替换标记之间的内容：

    <!-- BEGIN:RESULTS -->
    ...
    <!-- END:RESULTS -->

用法:
    python scripts/update_readme_results.py
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
README = os.path.join(ROOT, "README.md")
BEGIN = "<!-- BEGIN:RESULTS -->"
END = "<!-- END:RESULTS -->"


def md_table(df: pd.DataFrame, float_fmt: str = "%.3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "| " + " | ".join(["---"] * (len(cols) + 1)) + " |"]
    for name, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(float_fmt % v)
            elif isinstance(v, int):
                cells.append(str(v))
            else:
                cells.append(str(v))
        lines.append("| **%s** | %s |" % (name, " | ".join(cells)))
    return "\n".join(lines)


def build_block() -> str:
    parts = []
    summary_path = os.path.join(RESULTS, "metrics_summary.json")
    meta = None
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        d = meta["data"]
        parts.append("> 数据：`%s` 第 %d 列客户，%d 小时（%s ~ %s）；"
                     "滚动起点 %d 个 × 每窗口 %d 步。\n"
                     % (d["source"], d["client_col"], d["hours"],
                        str(d["start"])[:16], str(d["end"])[:16],
                        meta["task"]["origins"], meta["task"]["horizon"]))

    cmp_path = os.path.join(RESULTS, "model_comparison.csv")
    if os.path.exists(cmp_path):
        df = pd.read_csv(cmp_path, index_col=0)
        keep = [c for c in ["MAE", "RMSE", "MAPE(%)", "sMAPE(%)", "尖峰MAE", "R2",
                            "窗口数", "耗时(s)"] if c in df.columns]
        df = df[keep].sort_values("MAE")
        parts.append("### 1. 多模型对比（滚动起点平均）\n\n" + md_table(df) + "\n")

    ab_path = os.path.join(RESULTS, "ablation_features.csv")
    if os.path.exists(ab_path):
        df = pd.read_csv(ab_path, index_col=0)
        keep = [c for c in ["MAE", "RMSE", "MAPE(%)", "尖峰MAE", "R2",
                            "MAE相对提升(%)", "累计提升(%)"] if c in df.columns]
        parts.append("### 2. 特征工程消融（XGBoost）\n\n"
                     + md_table(df[keep], float_fmt="%.2f") + "\n")

    bench_path = os.path.join(RESULTS, "anomaly_benchmark.csv")
    if os.path.exists(bench_path):
        df = pd.read_csv(bench_path, index_col=0)
        keep = [c for c in ["ROC_AUC", "PR_AUC", "F1@5%",
                            "默认阈值_Precision", "默认阈值_Recall", "默认阈值_F1"]
                if c in df.columns]
        parts.append("### 3. 异常检测器对比（注入已知异常）\n\n"
                     + md_table(df[keep]) + "\n")
        kind_path = os.path.join(RESULTS, "anomaly_recall_by_kind.csv")
        if os.path.exists(kind_path):
            kdf = pd.read_csv(kind_path, index_col=0)
            parts.append("事件级召回率（默认阈值）：\n\n"
                         + md_table(kdf, float_fmt="%.2f") + "\n")

    if meta:
        parts.append("完整配置见 `results/run_config.json`，原始指标见 "
                     "`results/metrics_summary.json`。\n")
    return "\n".join(parts)


def main() -> int:
    if not os.path.exists(README):
        print("README.md 不存在")
        return 1
    with open(README, encoding="utf-8") as fh:
        text = fh.read()
    if BEGIN not in text or END not in text:
        print("README.md 里没有找到结果标记，跳过")
        return 1
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new = head + BEGIN + "\n\n" + build_block() + "\n" + END + tail
    with open(README, "w", encoding="utf-8") as fh:
        fh.write(new)
    print("README 结果区块已更新")
    return 0


if __name__ == "__main__":
    sys.exit(main())
