# -*- coding: utf-8 -*-
"""方向一：AI 智能日报。

把 DataHub 算好的数字打包成 JSON，交给大模型组织成调度员看得懂的自然语言日报。
模型只负责"表达"和"推测可能原因"，**不做任何计算**——这样日报里每个数字都能在
results/ 里找到出处。

没有配置密钥、或调用失败时，自动降级为 ``template_report`` 生成的本地模板日报，
保证界面永远有内容。
"""
from __future__ import annotations

import json
from typing import Iterator, Optional

from ..config import Config
from .client import LLMUnavailable, chat, chat_stream
from .context import DataHub, build_daily_brief


SYSTEM_PROMPT = """你是电网调度中心的负荷分析专家，负责把预测模型的输出翻译成调度员能直接用的分析日报。

写作要求：
1. **只能使用给定数据里的数字，绝对不要自己计算、推算或编造任何数值**；
2. 某项数据缺失时直接写"该项数据缺失"，不要猜测或用常识补全；
3. 面向一线调度员，语言简练，先给结论再给依据，不要堆砌术语；
4. 不要逐条复述原始数字，要给出判断（例如"负荷处于近期高位"而不是"均值是 5.24"）；
5. 异常原因只能基于给出的温度、负荷形态做"可能"的推测，用词要留有余地，
   不要断言是窃电或设备故障；
6. 输出 Markdown，不要用代码块包裹全文。
"""

USER_TEMPLATE = """请根据下面的数据写一份分析日报，包含三个部分：

## 一、整体负荷趋势
（最近 24 小时的负荷水平、峰谷出现时间、与前一日相比的变化；温度条件如何）

## 二、异常情况分析
（有没有异常点？集中在什么时段？结合当时的温度，推测可能的原因。没有异常就明确说没有）

## 三、给调度员的建议
（2~4 条可执行的建议，例如关注哪个时段、是否需要核查计量设备、
 哪个模型当前更可信、负荷高峰时段的应对）

<数据>
{payload}
</数据>
"""


def build_messages(brief: dict) -> list:
    payload = json.dumps(brief, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(payload=payload)},
    ]


def template_report(brief: dict) -> str:
    """不用大模型也能生成的日报（降级方案，纯本地计算）。"""
    L = brief.get("最近24小时实际负荷", {})
    meta = brief.get("数据概况", {})
    lines = ["# 负荷分析日报（本地模板生成）", ""]
    lines.append("> 客户 %s ｜ 数据区间 %s ~ %s ｜ 共 %s 小时"
                 % (meta.get("客户", "?"),
                    (meta.get("时间范围") or ["?", "?"])[0],
                    (meta.get("时间范围") or ["?", "?"])[1],
                    meta.get("小时级样本数", "?")))
    lines += ["", "## 一、整体负荷趋势", ""]
    if L:
        lines.append("- 最近 24 小时平均负荷 **%.2f kW**，峰值 **%.2f kW**（%s），"
                     "谷值 **%.2f kW**（%s）"
                     % (L.get("均值", 0), L.get("峰值", 0), L.get("峰值时刻", "?"),
                        L.get("谷值", 0), L.get("谷值时刻", "?")))
        if "与前24小时相比" in L:
            lines.append("- 与前 24 小时相比：**%s**" % L["与前24小时相比"])
    else:
        lines.append("- 最近 24 小时数据缺失")

    T = brief.get("最近24小时温度")
    if T:
        lines.append("- 气温 %.1f~%.1f℃，均值 %.1f℃"
                     % (T.get("最低", 0), T.get("最高", 0), T.get("均值", 0)))

    rel = brief.get("温度与负荷关系", {})
    if rel.get("分温度段平均负荷"):
        seg = "；".join("%s 平均 %.2f kW" % (s["温度区间"], s["平均负荷"])
                       for s in rel["分温度段平均负荷"])
        lines.append("- 温度与负荷的分段关系：%s（相关系数 %.3f）"
                     % (seg, rel.get("相关系数", 0)))

    fc = brief.get("最近一次24小时预测")
    if fc:
        lines.append("- 最近一次 %s 的 24 小时预测，平均绝对误差 %.2f kW"
                     % (fc.get("模型", "?"), fc.get("平均绝对误差_kW", 0)))

    lines += ["", "## 二、异常情况分析", ""]
    an = brief.get("检测到的异常点")
    if an and an.get("总数"):
        by = "、".join("%s %d 个" % (k, v) for k, v in an.get("按检测器", {}).items())
        lines.append("- 共检出 **%d** 个异常点（%s）" % (an["总数"], by))
        for it in an.get("最严重的8个", [])[:5]:
            n_alert = it.get("报警检测器数", it.get("报警次数", 0))
            lines.append("  - %s：负荷 %.2f kW，%d 个检测器同时报警"
                         % (it["时间"], it["负荷_kW"], n_alert))
        lines.append("- 可能原因需要结合现场核查；温度异常与计量突变是常见诱因。")
    else:
        lines.append("- 最近数据中未检出显著异常点。")

    lines += ["", "## 三、给调度员的建议", ""]
    mt = brief.get("模型表现") or {}
    rec = brief.get("推荐模型")
    if rec and rec in mt:
        m = mt[rec]
        lines.append("1. 当前表现最好的学习型模型是 **%s**（MAE %.2f kW、WAPE %.1f%%），"
                     "日常预测建议以它为主。" % (rec, m.get("MAE", 0), m.get("WAPE(%)", 0)))
    else:
        lines.append("1. 请先运行 `python run_pipeline.py` 生成完整的模型对比结果。")
    if L and L.get("峰值时刻"):
        lines.append("2. 关注 **%s** 前后的负荷高峰，提前确认备用容量。"
                     % str(L["峰值时刻"])[11:16])
    if an and an.get("总数"):
        lines.append("3. 对上述异常点对应的计量点做一次数据核查，"
                     "重点看是否为采集中断而不是真实负荷突变。")
    lines.append("4. 建议开启「智能问答」，可以直接追问某个时段的具体情况。")
    lines += ["", "---", "", "*本报告由本地模板生成（未调用大模型）。"
              "在 .env 中配置 API 密钥后可获得由大模型撰写的版本。*"]
    return "\n".join(lines)


def generate_report(hub: DataHub, cfg: Config, hours: int = 24,
                    extra: Optional[str] = None) -> dict:
    """生成日报（非流式）。返回 {text, source, context, error?}。"""
    brief = build_daily_brief(hub, hours=hours)
    messages = build_messages(brief)
    if extra:
        messages.append({"role": "user", "content": extra})
    try:
        msg, usage = chat(cfg, messages)
        return {"text": msg.content, "source": "llm", "context": brief,
                "usage": getattr(usage, "total_tokens", None)}
    except LLMUnavailable as exc:
        return {"text": template_report(brief), "source": "template",
                "context": brief, "error": str(exc)}


def stream_report(hub: DataHub, cfg: Config,
                  hours: int = 24) -> Iterator[str]:
    """流式生成日报，逐段 yield 文本；失败时自动降级成本地模板。"""
    brief = build_daily_brief(hub, hours=hours)
    try:
        got = False
        for piece in chat_stream(cfg, build_messages(brief)):
            got = True
            yield piece
        if not got:
            yield template_report(brief)
    except LLMUnavailable as exc:
        yield "> ⚠️ 大模型调用失败（%s），以下是本地模板生成的日报。\n\n" % exc
        yield template_report(brief)
