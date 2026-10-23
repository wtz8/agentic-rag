"""CLI 入口的终端编码护栏（M31）。

中文 Windows 控制台默认 cp936，`✓` / `✗` 编码不了，`print` 会抛 UnicodeEncodeError。
崩的是**最后一行结果输出**，前面的摄取/检索都白跑，所以这条要在 CLI 入口兜住。
"""

from __future__ import annotations

import io
import sys

from agentic_rag.cli import _relax_console_encoding, build_parser, main


def test_gbk_console_does_not_crash_on_checkmark(monkeypatch):
    """cp936 流上打印 ✗ 必须不抛（护栏把不可编码字符替换掉）。"""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="gbk")
    monkeypatch.setattr(sys, "stdout", stream)

    _relax_console_encoding()
    print("✓ 已通过 / ✗ 未通过")  # 不加护栏这里就是 UnicodeEncodeError
    stream.flush()

    assert raw.getvalue()  # 确实写出了内容（替换字符），而不是静默失败


def test_guard_tolerates_streams_without_reconfigure(monkeypatch):
    """被替换成 StringIO / 管道时没有 reconfigure，护栏只能跳过，不能抛。"""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    _relax_console_encoding()  # 不抛即通过


def test_parser_exposes_all_documented_commands():
    """README/docs 承诺的 6 个子命令必须都在（04 §11）。"""
    parser = build_parser()
    actions = [
        a for a in parser._subparsers._group_actions  # noqa: SLF001 - 测试专用反射
    ]
    names = set(actions[0].choices)
    assert {"ingest", "ask", "probe", "eval", "inspect", "serve"} <= names


def test_unknown_command_exits_nonzero():
    try:
        main(["definitely-not-a-command"])
    except SystemExit as e:  # argparse 的用法错误
        assert e.code != 0
    else:  # pragma: no cover - 不应该走到
        raise AssertionError("未知子命令应当以非零码退出")
