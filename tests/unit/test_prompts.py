"""提示词模板契约（07）：加载、关键词、渲染、schema_version。"""

from __future__ import annotations

import re

import pytest

from agentic_rag.prompts.loader import _parse, all_prompts, get_prompt, prompt_schema_versions

# 与 loader 一致的占位符识别：{{ }} 是转义，其余 {xxx} 是待注入变量
_PLACEHOLDER = re.compile(r"\{\{|\}\}|\{([^{}]*)\}")

EXPECTED = {
    "route",
    "rewrite",
    "decompose",
    "hyde",
    "answer",
    "claim_extract",
    "claim_verify",
    "judge_faithfulness",
    "judge_relevance",
    "judge_citation",
}


def test_all_templates_present():
    names = {p.name for p in all_prompts()}
    assert EXPECTED <= names, f"缺少模板: {EXPECTED - names}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_template_parses(name):
    p = get_prompt(name)
    assert p.schema_version >= 1
    assert p.user_template.strip()
    assert p.model in ("light", "main")


@pytest.mark.parametrize(
    "name,keyword",
    [
        ("route", "intent"),
        ("rewrite", "queries"),
        ("decompose", "queries"),
        ("hyde", "queries"),
        ("claim_extract", "claims"),
        ("claim_verify", "unsupported"),
        ("judge_faithfulness", "supported"),
        ("judge_relevance", "score"),
        ("judge_citation", "correct"),
    ],
)
def test_mock_keyword_present(name, keyword):
    """05 §3：MockLLM 靠英文 schema 关键词匹配，模板改掉关键词就会静默跑歪。"""
    p = get_prompt(name)
    assert keyword in p.user_template


def test_json_mode_flags():
    for name in ("route", "rewrite", "decompose", "hyde", "claim_extract", "claim_verify",
                 "judge_faithfulness", "judge_relevance", "judge_citation"):
        assert get_prompt(name).json_mode is True, name
    assert get_prompt("answer").json_mode is False


def test_render_injects_variables():
    p = get_prompt("route")
    msgs = p.messages(query="它的上限是多少？", history="-", intents="simple | multi_hop")
    assert msgs[0]["role"] == "system"
    assert "它的上限是多少？" in msgs[1]["content"]
    assert "{query}" not in msgs[1]["content"]


def test_answer_template_has_citation_contract():
    p = get_prompt("answer")
    rendered = p.render(query="Q", context="[1] xxx", max_citations=5, language="中文")
    assert "[1]" in rendered
    # max_citations 由配置注入，不得写死在模板正文里
    assert "{max_citations}" in p.system
    assert str(5) in p.system.format(max_citations=5, language="中文")


def test_system_section_is_rendered_too():
    """M30：只渲染 <user> 会让 answer/decompose 的 system 段带着花括号发给模型。

    这条用例走 `messages()`（生产路径），不手工 `format`——手工 format 永远能通过，
    正是它掩盖了"system 段从未被渲染"。
    """
    p = get_prompt("answer")
    msgs = p.messages(query="Q", context="[1] xxx", max_citations=5, language="中文")
    system = msgs[0]["content"]
    assert "{max_citations}" not in system and "{language}" not in system
    assert "5" in system and "中文" in system

    d = get_prompt("decompose")
    d_system = d.messages(query="Q", history="-", max_sub_queries=4)[0]["content"]
    assert "{max_sub_queries}" not in d_system
    assert "4" in d_system


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_no_placeholder_left_after_render(name):
    """任何模板渲染后都不该残留 `{变量名}`——两段一起查。

    只查"具体变量名"，不查任意花括号：渲染后的 JSON 示例本身就是字面花括号。
    """
    p = get_prompt(name)
    names = set()
    for section in (p.system, p.user_template):
        for m in _PLACEHOLDER.finditer(section):
            if m.group(1):
                names.add(m.group(1))
    kwargs = {n: "X" for n in names}
    for msg in p.messages(**kwargs):
        for var in names:
            token = "{" + var + "}"
            assert token not in msg["content"], f"{name} 的 {msg['role']} 段残留 {token}"


def test_unescaped_literal_brace_is_rejected(tmp_path):
    """字面 JSON 花括号必须转义，否则 str.format 会把它当成变量。"""
    bad = tmp_path / "bad.md"
    bad.write_text(
        '<system>\n输出 {"ok": true}\n</system>\n\n<user>\n{q}\n</user>\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="未转义的字面花括号"):
        _parse(bad)

    good = tmp_path / "good.md"
    good.write_text(
        '<system>\n输出 {{"ok": true}}\n</system>\n\n<user>\n{q}\n</user>\n',
        encoding="utf-8",
    )
    p = _parse(good)
    assert '{"ok": true}' in p.system_text(q="x")


def test_schema_versions_stable_set():
    v = prompt_schema_versions()
    assert len(v) == len(all_prompts())
    assert all(isinstance(x, int) for x in v.values())


def test_no_business_constants_in_templates():
    """模板里不得写死阈值 / 模型名 / top_k（05 §0.1 第 5 条）。"""
    for p in all_prompts():
        body = p.system + p.user_template
        for bad in ("qwen-plus", "qwen-flash", "0.35", "1024", "top_k=50"):
            assert bad not in body, f"{p.name} 里出现了业务常量 {bad}"


def test_unknown_template_raises():
    with pytest.raises(KeyError):
        get_prompt("not_exist")
