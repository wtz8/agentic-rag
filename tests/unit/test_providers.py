"""Provider：extract_json 宽容解析 + Mock 的确定性与契约（07 §2）。"""

from __future__ import annotations

import json

import pytest

from agentic_rag.config import get_config
from agentic_rag.providers.base import (
    ProviderResponseError,
    extract_json,
    first_json_object,
)
from agentic_rag.providers.embedding import MockEmbedding, get_embedding
from agentic_rag.providers.llm import MockLLM, get_llm
from agentic_rag.providers.rerank import MockRerank, get_rerank


def test_extract_json_bare():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    raw = '说明一下：\n```json\n{"intent": "simple"}\n```\n以上。'
    assert extract_json(raw) == {"intent": "simple"}


def test_extract_json_surrounded_by_prose():
    raw = '好的，结果如下：\n{"claims": ["a", "b"]}\n希望有帮助。'
    assert extract_json(raw) == {"claims": ["a", "b"]}


def test_extract_json_list():
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_extract_json_invalid_raises():
    with pytest.raises(ProviderResponseError):
        extract_json("这不是 JSON")
    with pytest.raises(ProviderResponseError):
        extract_json("")


def test_first_json_object_type_check():
    assert first_json_object('{"a": 1}')["a"] == 1
    with pytest.raises(ProviderResponseError):
        first_json_object("[1]")


# ── Mock 的确定性（可复现的前提）──────────────────────────────────────


def test_mock_embedding_deterministic():
    e = MockEmbedding(dim=64)
    a = e.embed(["华东区带宽采购单价", "华北区带宽采购单价"])
    b = e.embed(["华东区带宽采购单价", "华北区带宽采购单价"])
    assert a == b
    assert len(a[0]) == 64


def test_mock_embedding_literal_overlap_scores_higher():
    e = MockEmbedding(dim=128)
    q, near, far = "华东区带宽采购单价", "华东区带宽采购单价为 3.2 万", "设备固件升级步骤说明"
    vq, vn, vf = e.embed([q, near, far])

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))

    assert cos(vq, vn) > cos(vq, vf), "字面重叠更多必须更相似，这是 Mock 被选中的理由"


def test_mock_llm_keyword_contract():
    llm = MockLLM()

    def ask(user: str) -> dict:
        return json.loads(llm.chat([{"role": "user", "content": user}]))

    route = ask('待分类问题：它的上限是多少？\n\n合法取值：simple\n\n只输出 JSON：{"intent": ..., "confidence": ..., "reason": ...}')
    assert route["intent"] == "simple" and 0 <= route["confidence"] <= 1

    rw = ask('原始问题：它的上限是多少？\n\n只输出 JSON：{"queries": [...], "strategy": "rewrite"}')
    assert rw["queries"] == ["它的上限是多少？"]

    claims = ask("答案正文：单卡上限是 5 万元。年费为 200 元。\n\n只输出 JSON：{\"claims\": [...]}")
    assert claims["claims"] == ["单卡上限是 5 万元。", "年费为 200 元。"]

    ver = ask('命题列表：\n- a\n\n只输出 JSON：{"unsupported": [...], "faithfulness": 0.9}')
    assert ver["unsupported"] == []

    assert ask('只输出 JSON：{"score": 4}')["score"] == 5
    assert ask('只输出 JSON：{"supported": [], "unsupported": []}')["supported"] == []
    assert ask('只输出 JSON：{"correct": [], "wrong": []}')["wrong"] == []


def test_mock_llm_fallback_is_deterministic_text():
    llm = MockLLM()
    a = llm.chat([{"role": "user", "content": "随便问点什么"}])
    b = llm.chat([{"role": "user", "content": "随便问点什么"}])
    assert a == b
    assert not a.startswith("{")


def test_mock_llm_stream_has_citation_marker():
    llm = MockLLM()
    text = "".join(llm.stream([{"role": "user", "content": "问题：X"}]))
    assert "[1]" in text


def test_mock_rerank_ranking():
    r = MockRerank()
    out = r.rerank(
        "华东区带宽采购单价",
        ["华东区带宽采购单价为 3.2 万元", "完全不相关的固件说明"],
        top_n=2,
    )
    assert out[0][0] == 0
    assert out[0][1] > out[1][1]
    assert all(0.0 <= s <= 1.0 for _, s in out)


def test_mock_rerank_empty():
    assert MockRerank().rerank("q", [], 5) == []


# ── get_* 按有无 Key 切换 ────────────────────────────────────────────


def test_factories_return_mock_without_key(cfg):
    assert isinstance(get_llm(cfg, "main"), MockLLM)
    assert isinstance(get_embedding(cfg), MockEmbedding)
    assert isinstance(get_rerank(cfg), MockRerank)


def test_factories_return_dashscope_with_key():
    c = get_config(env={"DASHSCOPE_API_KEY": "sk-test"})
    from agentic_rag.providers.embedding import DashScopeEmbedding
    from agentic_rag.providers.llm import DashScopeLLM
    from agentic_rag.providers.rerank import DashScopeRerank

    assert isinstance(get_llm(c, "main"), DashScopeLLM)
    assert isinstance(get_embedding(c), DashScopeEmbedding)
    assert isinstance(get_rerank(c), DashScopeRerank)


# ── embedding 分批：条数与总长度双约束（真实 API 暴露）──────────────────


def test_plan_batches_respects_count_and_char_budget():
    from agentic_rag.providers.embedding import _BATCH, _MAX_INPUT_CHARS, _plan_batches

    # 全是长块：条数上限还没到，长度上限先到 → 必须提前切
    longs = ["a" * 20000 for _ in range(5)]
    batches = _plan_batches(longs)
    assert len(batches) == 5, "每条 20000 字符，两条就超 30000 预算"
    assert all(sum(len(t) for t in b) <= _MAX_INPUT_CHARS for b in batches)

    # 全是短块：按条数切
    shorts = ["短"] * 25
    batches = _plan_batches(shorts)
    assert [len(b) for b in batches] == [10, 10, 5]
    assert all(sum(len(t) for t in b) <= _MAX_INPUT_CHARS for b in batches)

    assert _BATCH == 10  # 08 §3.2 实测值，别顺手改


def test_plan_batches_isolates_oversized_single_text():
    """单条就超限的，单独成批且不截断——失败时能精确定位到那一条。"""
    from agentic_rag.providers.embedding import _MAX_INPUT_CHARS, _plan_batches

    texts = ["短", "b" * (_MAX_INPUT_CHARS + 5000), "短"]
    batches = _plan_batches(texts)
    assert ["短"] in batches
    assert batches.count(["b" * (_MAX_INPUT_CHARS + 5000)]) == 1
    assert all(len(b) == 1 or sum(len(t) for t in b) <= _MAX_INPUT_CHARS for b in batches)
