# -*- coding: utf-8 -*-
"""把大模型输出的 Markdown 处理成适合在 Streamlit 里显示的形式。

踩过的两个坑
------------
1. **波浪号被当成删除线**：GFM（Streamlit 前端用的就是它）支持单波浪号删除线，
   而电力数据里到处都是区间写法 —— "10~18℃"、"-50~10℃"、"6.7~10.8 kW"。
   两个单波浪号会被配成一对，中间一整段文字被划掉，用户看到的就是
   "文字中间一条横线、波浪号还不见了"。
2. **表格渲染**：不同版本的 Streamlit 对 Markdown 表格支持不一致，直接把原文交给
   ``st.markdown`` 有风险（可能原样显示 ``|---|---|``）。这里统一把表格抽出来，
   交给调用方用原生表格组件渲染。
"""
from __future__ import annotations

import re
from typing import List, Tuple, Union

import pandas as pd

_TABLE_SEP = re.compile(r"^\s*\|?[\s:\-|]+\|[\s:\-|]*$")


def protect_tildes(text: str) -> str:
    """把正文里的半角波浪号换成全角，避免被解析成删除线。

    反引号包裹的行内代码保持原样（那里不参与 Markdown 解析）。
    """
    out: List[str] = []
    in_code = False
    for ch in text:
        if ch == "`":
            in_code = not in_code
            out.append(ch)
        elif ch == "~" and not in_code:
            out.append("～")
        else:
            out.append(ch)
    return "".join(out)


def parse_md_table(lines: List[str]) -> pd.DataFrame:
    """把 Markdown 表格的若干行转成 DataFrame（列数不齐时补齐或截断）。"""
    def split_row(line: str) -> List[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    header = split_row(lines[0])
    width = max(1, len(header))
    rows = [split_row(l) for l in lines[2:]]
    norm = [(r + [""] * width)[:width] for r in rows]
    return pd.DataFrame(norm, columns=header)


def split_markdown_tables(text: str) -> List[Tuple[str, Union[str, pd.DataFrame]]]:
    """把文本拆成 [("md", 文本) | ("table", DataFrame)] 序列。"""
    lines = (text or "").split("\n")
    blocks: List[Tuple[str, Union[str, pd.DataFrame]]] = []
    buf: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_row = line.strip().startswith("|") and line.strip().endswith("|")
        is_table = (is_row and i + 1 < len(lines) and "|" in lines[i + 1]
                    and _TABLE_SEP.match(lines[i + 1]))
        if is_table:
            chunk = [line, lines[i + 1]]
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                chunk.append(lines[j])
                j += 1
            if buf:
                blocks.append(("md", "\n".join(buf)))
                buf = []
            blocks.append(("table", parse_md_table(chunk)))
            i = j
            continue
        buf.append(line)
        i += 1
    if buf:
        blocks.append(("md", "\n".join(buf)))
    return blocks
