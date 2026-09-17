# -*- coding: utf-8 -*-
"""LLM 能力层：日报生成、数据问答、异常归因 Agent。

设计原则（重要）
----------------
**所有数字都由本项目的 Python 代码算好，再交给模型组织语言。**
模型不接触原始数据、不做算术，只做"归因"和"表达"。这样可以从根子上避免
大模型编造数字——日报里的每一个数值都能在 results/ 里找到出处。
"""
from __future__ import annotations

__all__ = ["client", "context", "tools", "report", "agent"]
