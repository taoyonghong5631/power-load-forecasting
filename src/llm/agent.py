# -*- coding: utf-8 -*-
"""方向二 / 方向三：把工具箱接上大模型，做成能查数据的问答助手与自动归因 Agent。

* ``run_agent``       —— 自然语言提问 → 模型决定调哪些工具 → 用真实结果回答（方向二）
* ``attribute_anomaly`` —— 针对一个异常点，自动跑一轮归因分析（方向三）

循环逻辑很简单：模型要求调工具 → 我们执行 → 把结果塞回对话 → 模型继续，
直到它不再要求调工具为止。每一轮调用都记录在 ``ToolBox.trace`` 里，
界面上可以展示"它查了什么"，便于核对答案是否可信。
"""
from __future__ import annotations

import json
from typing import List, Optional

from ..config import Config
from .client import LLMUnavailable, chat
from .context import DataHub
from .tools import TOOL_SCHEMAS, ToolBox


SYSTEM_PROMPT = """你是电网负荷分析助手，面向调度员回答关于负荷预测与异常检测的问题。

工作方式：
1. 先判断问题需要哪些数据，再调用提供的工具去取——**不要凭常识或猜测回答**；
2. 拿到的工具结果里如果有错误或说明数据缺失，要如实告诉用户，不要编造；
3. 回答里出现的每个数字都必须来自工具返回值，**不要自己算或改写数值**；
4. 回答用中文，简明扼要；能画图的问题（例如"看一下某天的预测"）就调用画图工具；
5. 如果问题超出数据范围（例如问别的客户），直接说明当前只有 {client} 的数据。

当前可用数据的概况：
{overview}
"""


def _system_prompt(hub: DataHub) -> str:
    d = hub.describe()
    overview = json.dumps(d, ensure_ascii=False, indent=2)
    return SYSTEM_PROMPT.format(client=d.get("客户", "?"), overview=overview)


def _tool_call_payload(tool_calls) -> List[dict]:
    """把 SDK 的 tool_calls 转成可以直接塞回 messages 的字典。"""
    out = []
    for tc in tool_calls:
        out.append({
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name,
                         "arguments": tc.function.arguments or "{}"},
        })
    return out


def run_agent(question: str, toolbox: ToolBox, cfg: Config,
              history: Optional[List[dict]] = None) -> dict:
    """跑一轮工具调用循环，返回 {text, figures, trace, error?}。"""
    hub = toolbox.hub
    messages = [{"role": "system", "content": _system_prompt(hub)}]
    for h in (history or [])[-6:]:            # 只带最近几轮，控制 token
        if h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": question})

    n_fig_before = len(toolbox.figures)
    try:
        for _ in range(max(1, cfg.llm.max_tool_rounds)):
            msg, _usage = chat(cfg, messages, tools=TOOL_SCHEMAS)
            calls = getattr(msg, "tool_calls", None)
            if not calls:
                return {"text": msg.content or "（模型没有返回内容）",
                        "figures": list(toolbox.figures)[n_fig_before:],
                        "trace": toolbox.trace}
            messages.append({"role": "assistant", "content": msg.content,
                             "tool_calls": _tool_call_payload(calls)})
            for tc in calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = toolbox.call(tc.function.name, args)
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })
        # 工具调太多轮还没收敛：让模型直接给结论
        messages.append({"role": "user",
                         "content": "请立刻基于已获得的数据给出最终回答，不要再调用工具。"})
        msg, _usage = chat(cfg, messages)
        return {"text": msg.content or "（模型没有返回内容）",
                "figures": list(toolbox.figures)[n_fig_before:],
                "trace": toolbox.trace}
    except LLMUnavailable as exc:
        return {"text": "⚠️ 暂时无法调用大模型：%s\n\n"
                        "你仍然可以在「预测对比」「异常检测」页面查看图表，"
                        "或点击「生成分析日报」使用本地模板版本。" % exc,
                "figures": [], "trace": toolbox.trace, "error": str(exc)}


def attribute_anomaly(toolbox: ToolBox, cfg: Config,
                      top_k: int = 5) -> dict:
    """方向三：针对最严重的若干异常点，自动跑一轮归因分析。

    提示词强制模型按固定顺序取证（异常点 → 同期温度/负荷 → 温度关系 → 模型误差），
    再给结论，避免它跳过数据直接下判断。
    """
    question = (
        "请对数据里最严重的 %d 个异常点做自动归因分析，按下面的顺序取证再下结论：\n"
        "1. 先调用 find_anomalies 取出最严重的 %d 个异常点（时间、负荷、被哪些检测器报警）；\n"
        "2. 对每个异常点，用 query_load 取当天前后的负荷与温度，看是不是温度骤变或采集问题；\n"
        "3. 调用 get_temperature_relation 判断该时段负荷是否偏离温度应有的水平；\n"
        "4. 调用 compare_models 看当时模型的误差是否偏大。\n\n"
        "最后输出一张表格：异常时间 | 负荷(kW) | 可能原因（按可能性排序）| 建议的核查动作。\n"
        "注意：原因只能写「可能」，并说明依据来自哪个数字；数据不足就写「需现场核查」。"
        % (top_k, top_k))
    result = run_agent(question, toolbox, cfg)
    result["kind"] = "anomaly_attribution"
    return result
