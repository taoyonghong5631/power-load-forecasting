# -*- coding: utf-8 -*-
"""LLM 能力层测试。

默认只跑离线部分（不花钱、不需要网络）：
    python tests/test_llm_smoke.py

加上 --live 会真实调用一次 DeepSeek 接口（约 0.01 元）：
    python tests/test_llm_smoke.py --live
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config
from src.llm.client import has_api_key
from src.llm.context import DataHub, build_daily_brief, temperature_load_relation
from src.llm.report import build_messages, template_report
from src.llm.tools import TOOL_SCHEMAS, ToolBox, tool_names


def _hub():
    return DataHub(get_config())


def test_hub_loads():
    hub = _hub()
    assert hub.ready, "没有加载到小时级数据（先跑 run_pipeline.py）"
    d = hub.describe()
    assert d["小时级样本数"] > 1000
    assert "temp" in hub.series.columns, "温度没有拼回数据里"


def test_daily_brief_structure():
    brief = build_daily_brief(_hub(), hours=24)
    for key in ("数据概况", "最近24小时实际负荷", "最近24小时温度", "温度与负荷关系",
                "模型表现", "特征重要性Top8"):
        assert key in brief, "日报数据包缺少 %s（部分产物可能需要先跑 run_pipeline.py）" % key
    assert brief["最近24小时实际负荷"]["峰值"] >= brief["最近24小时实际负荷"]["谷值"]


def test_template_report_is_markdown():
    brief = build_daily_brief(_hub(), hours=24)
    text = template_report(brief)
    assert text.startswith("#") and "## 一、整体负荷趋势" in text
    assert "## 二、异常情况分析" in text and "## 三、给调度员的建议" in text


def test_prompt_contains_numbers_only_from_data():
    brief = build_daily_brief(_hub(), hours=24)
    msgs = build_messages(brief)
    assert len(msgs) == 2 and msgs[0]["role"] == "system"
    body = msgs[1]["content"]
    assert "不要自己计算" in msgs[0]["content"]
    payload = body.split("<数据>")[1].split("</数据>")[0]
    parsed = json.loads(payload)
    assert parsed["数据概况"]["客户"].startswith("MT_")


def test_all_tools_return_dict():
    hub = _hub()
    box = ToolBox(hub, get_config())
    # 每个工具都必须能独立跑通并返回可序列化的字典
    calls = [
        ("describe_dataset", {}),
        ("query_load", {"agg": "daily"}),
        ("query_load", {"start": "2014-09-01", "end": "2014-09-05", "agg": "hourly"}),
        ("find_anomalies", {"top_k": 3}),
        ("compare_models", {}),
        ("get_feature_importance", {"top_k": 5}),
        ("get_temperature_relation", {"days": 30}),
        ("plot_forecast", {"date": "2014-09-15"}),
        ("plot_forecast", {}),                      # 不传日期 = 画最近一次
        ("forecast_error_profile", {"top_k": 3}),
        ("plot_anomalies", {"days": 7}),
        ("不存在的工具", {}),
    ]
    for name, args in calls:
        out = box.call(name, args)
        assert isinstance(out, dict), "%s 没有返回字典" % name
        json.dumps(out, ensure_ascii=False, default=str)   # 必须可序列化
    assert len(box.figures) >= 1, "画图工具没有产出 figure"
    assert tool_names()[0] == "describe_dataset"
    assert "forecast_error_profile" in tool_names()
    assert all(t["type"] == "function" for t in TOOL_SCHEMAS)


def test_offline_agent_message_without_key():
    """没有密钥时，Agent 也要给出可读的提示而不是崩溃。"""
    cfg = get_config()
    if has_api_key(cfg):
        print("      （检测到密钥，跳过无密钥降级测试）")
        return
    from src.llm.agent import run_agent
    box = ToolBox(_hub(), cfg)
    out = run_agent("最近负荷怎么样？", box, cfg)
    assert "无法调用大模型" in out["text"]


def test_live_llm():
    """真实调用一次 DeepSeek：日报 + 一次工具调用问答。"""
    from src.llm.agent import run_agent
    from src.llm.report import generate_report
    cfg = get_config()
    if not has_api_key(cfg):
        print("      （没有密钥，跳过联网测试）")
        return

    rep = generate_report(_hub(), cfg, hours=24)
    assert rep["source"] == "llm", "日报没有走大模型：%s" % rep.get("error")
    assert len(rep["text"]) > 200, "日报太短"
    print("      日报长度 %d 字，消耗 %s tokens" % (len(rep["text"]), rep.get("usage")))

    box = ToolBox(_hub(), cfg)
    ans = run_agent("最近一个月最异常的时段是什么时候？", box, cfg)
    assert ans.get("trace"), "Agent 没有调用任何工具"
    assert len(ans["text"]) > 30
    print("      问答调用了 %d 个工具，回答 %d 字"
          % (len(ans["trace"]), len(ans["text"])))


def main() -> int:
    live = "--live" in sys.argv
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v) and (live or k != "test_live_llm")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("PASS  %s" % fn.__name__)
        except AssertionError as exc:
            failed += 1
            print("FAIL  %s: %s" % (fn.__name__, exc))
        except Exception as exc:
            failed += 1
            print("ERROR %s: %s: %s" % (fn.__name__, type(exc).__name__, exc))
    print("\n%d/%d 通过" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
