# -*- coding: utf-8 -*-
"""用已经跑出来的预测结果生成滚动预测 GIF（给 README 用）。

前置：先跑 ``python run_pipeline.py``，会在 results/ 下生成 pred_*.npz。

用法:
    python scripts/make_gif.py --model "XGBoost_(递归多步)" --days 10
    python scripts/make_gif.py --model ARIMA --days 10
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(ROOT, "results", "screenshots")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DARK = "#1f2d5a"
RED = "#e4572e"
GREY = "#9aa5b1"


def load(model: str):
    """按模型名找预测文件：先精确匹配，再退回子串匹配。"""
    files = glob.glob(os.path.join(RESULTS, "pred_*.npz"))
    want = model.lower().replace(" ", "_")
    exact = [p for p in files if os.path.basename(p)[5:-4].lower() == want]
    cand = exact or [p for p in files if want in os.path.basename(p).lower()]
    if not cand:
        avail = sorted(os.path.basename(p)[5:-4] for p in files)
        raise SystemExit("没找到 %s 的预测结果。可选的模型名：%s\n"
                         "请先运行 python run_pipeline.py"
                         % (model, "、".join(avail) if avail else "（results/ 下没有任何 pred_*.npz）"))
    if len(cand) > 1:
        print("提示：%s 匹配到多个结果，使用 %s" % (model, os.path.basename(cand[0])))
    d = np.load(cand[0], allow_pickle=True)
    return d, os.path.basename(cand[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="XGBoost_(递归多步)",
                    help="results/pred_<模型名>.npz 里的模型名，如 XGBoost_(递归多步) / ARIMA")
    ap.add_argument("--days", type=int, default=10, help="GIF 覆盖的预测窗口数")
    ap.add_argument("--fps", type=float, default=2.5)
    ap.add_argument("--out", default="forecast_roll.gif")
    args = ap.parse_args()

    d, src = load(args.model)
    preds, truths = d["preds"], d["truths"]
    stamps = pd.to_datetime(d["stamps"])
    full_index = pd.to_datetime(d["index"])
    full_load = d["load"].astype(float)
    n = min(args.days, len(preds))
    step = max(1, len(preds) // n)
    sel = list(range(0, len(preds), step))[:n]

    # 完整曲线需要一个包含这些窗口的连续区间
    lo = max(0, int(np.searchsorted(full_index, stamps[sel[0]])) - 72)
    hi = min(len(full_index), int(np.searchsorted(full_index, stamps[sel[-1]])) + 24 * 2)
    ctx_index = full_index[lo:hi]
    ctx_load = full_load[lo:hi]

    frames = []
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6), dpi=100,
                             gridspec_kw={"height_ratios": [1, 1.25]})
    for k in sel:
        ax1, ax2 = axes
        ax1.clear()
        ax2.clear()
        ax1.plot(ctx_index, ctx_load, color=GREY, lw=1.4, label="实际负荷")
        ts = stamps[k]
        win = pd.date_range(ts, periods=len(truths[k]), freq="h")
        err = np.abs(truths[k] - preds[k]).mean()
        ax1.axvspan(win[0], win[-1], color=RED, alpha=0.12)
        ax1.plot(win, truths[k], color=DARK, lw=2.0)
        ax1.plot(win, preds[k], color=RED, lw=2.0, ls="--")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.set_title("%s 滚动 24 小时预测（第 %d/%d 个窗口，起点 %s）"
                      % (args.model, k + 1, len(preds), ts), fontsize=11)
        ax1.set_ylabel("负荷 (kW)")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))

        ax2.plot(win, truths[k], color=DARK, lw=2.4, label="真实值")
        ax2.plot(win, preds[k], color=RED, lw=2.4, ls="--", label="预测值")
        ax2.fill_between(win, preds[k], truths[k], color=RED, alpha=0.13)
        ax2.set_title("该窗口 MAE = %.2f kW" % err, fontsize=11)
        ax2.set_ylabel("负荷 (kW)")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        for ax in (ax1, ax2):
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())))
    plt.close(fig)

    out = os.path.join(OUT_DIR, args.out)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    size = os.path.getsize(out) / 1e6
    print("已生成 %s（%d 帧，%.2f MB，来源 %s）" % (out, len(frames), size, src))
    return 0


if __name__ == "__main__":
    sys.exit(main())
