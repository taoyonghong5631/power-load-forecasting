# -*- coding: utf-8 -*-
"""命令行生成 AI 分析日报（不需要打开界面）。

用法:
    python scripts/ai_report.py                    # 生成并打印
    python scripts/ai_report.py --hours 48         # 用最近 48 小时
    python scripts/ai_report.py --out results/daily_report.md
    python scripts/ai_report.py --template         # 不调用大模型，只出本地模板版

没有配置 API 密钥时自动降级为本地模板日报，不会报错。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import ROOT, get_config
from src.llm.client import has_api_key
from src.llm.context import DataHub
from src.llm.report import generate_report, template_report


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 AI 负荷分析日报")
    ap.add_argument("--hours", type=int, default=24, help="分析最近多少小时，默认 24")
    ap.add_argument("--out", default=None, help="保存到文件，默认只打印")
    ap.add_argument("--template", action="store_true", help="强制使用本地模板（不调用大模型）")
    args = ap.parse_args()

    cfg = get_config()
    hub = DataHub(cfg)
    if not hub.ready:
        print("没有找到实验结果，请先运行：python run_pipeline.py")
        return 1

    if args.template:
        from src.llm.context import build_daily_brief
        text, source = template_report(build_daily_brief(hub, hours=args.hours)), "template"
    else:
        if not has_api_key(cfg):
            print("提示：未检测到 API 密钥，改用本地模板生成。\n")
        result = generate_report(hub, cfg, hours=args.hours)
        text, source = result["text"], result["source"]
        if result.get("error"):
            print("提示：大模型调用失败（%s），已降级为本地模板。\n" % result["error"])

    print(text)
    if args.out:
        path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print("\n[已保存] %s（来源：%s）" % (path, source))
    return 0


if __name__ == "__main__":
    sys.exit(main())
