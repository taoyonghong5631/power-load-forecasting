# -*- coding: utf-8 -*-
"""Streamlit 界面的冒烟测试（用官方 AppTest，无需浏览器）。

重点回归一类很容易犯的错：**变量只在 `if 按钮被点击` 分支里赋值，
但重跑时（按钮没被点击）另一处又去用它**——本项目真的踩过：
训练完之后点任何按钮都会 NameError: name 'trained' is not defined。

运行:
    python tests/test_app_smoke.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _fresh(timeout: int = 1800) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    return at


def test_app_renders_without_error():
    at = _fresh()
    assert list(at.exception) == [], "首屏就报错了：%s" % list(at.exception)
    assert len(at.tabs) == 5, "标签页数量不对：%s" % [t.label for t in at.tabs]


def test_ai_entries_exist():
    at = _fresh()
    labels = [b.label for b in at.button]
    assert any("生成分析日报" in x for x in labels), labels
    assert any("自动归因" in x for x in labels), labels
    assert any("最异常" in x for x in labels), "缺少快捷提问按钮：%s" % labels


def test_rerun_after_training_does_not_crash():
    """回归：训练完成后，任何一次重跑（点按钮 / 刷新）都不能崩。"""
    at = _fresh()
    # 只跑 XGBoost，快一些
    at.multiselect[0].set_value(["XGBoost"])
    # 侧边栏的「开始训练与预测」按钮
    train_btn = [b for b in at.button if "开始训练" in b.label][0]
    train_btn.click()
    at.run()
    assert list(at.exception) == [], "训练阶段就报错：%s" % list(at.exception)
    assert at.session_state.filtered_state.get("results"), "训练结果没有存进 session_state"
    # 关键：再跑一次（此时 run=False，等价于点任意按钮触发的重跑）
    at.run()
    assert list(at.exception) == [], (
        "训练后的重跑报错了——通常是某个变量只在 if run 分支里赋值，"
        "却在外面被使用：%s" % list(at.exception))


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
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
