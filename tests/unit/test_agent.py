"""Agent 编排层：条件边与 8 个节点（04 §6 / H4 / L5 / L9 / L10 / K1）。

节点测试的重点不是"能跑通"，而是**每条分支都走对**：
条件边只读 state，节点负责自增与事件，两者混起来就会出现死代码或事件重复。
"""

from __future__ import annotations

import json

import pytest

from agentic_rag.agent.conditions import (
    _reaches_floor,
    dispatch_by_intent,
    is_sufficient,
    need_regenerate,
    should_regenerate,
)
from agentic_rag.agent.context import GraphDeps, NodeContext
from agentic_rag.agent.graph import build_graph, initial_state
from agentic_rag.agent.nodes import (
    degrade_answer_node,
    direct_answer_node,
    generate_node,
    rerank_node,
    retrieve_node,
    rewrite_node,
    route_node,
    verify_node,
)
from agentic_rag.index.store import IndexStore
from agentic_rag.observability.trace import Tracer
from agentic_rag.providers.base import ProviderError
from agentic_rag.providers.embedding import MockEmbedding
from agentic_rag.retrieval.retriever import HybridRetriever
from agentic_rag.schema import Candidate, Chunk, ScoredChunk


# ── 测试替身 ──────────────────────────────────────────────────────────


class FakeLLM:
    """按子字符串匹配返回 JSON；`raise_on` 里的关键词触发异常。"""

    def __init__(self, payloads: dict[str, dict] | None = None, raise_on: tuple[str, ...] = ()):
        self.payloads = payloads or {}
        self.raise_on = raise_on
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake-main"

    def chat(self, messages, temperature=0.0, max_tokens=2048, json_mode=False):
        self.calls += 1
        user = messages[-1]["content"]
        for kw in self.raise_on:
            if kw in user:
                raise ProviderError(f"{kw} 调用失败")
        for kw, payload in self.payloads.items():
            if kw in user:
                return json.dumps(payload, ensure_ascii=False)
        return "（兜底）确定性答案。[1]"

    def stream(self, messages, temperature=0.0, max_tokens=2048):
        for ch in "流式答案[1]":
            yield ch


class FakeRerank:
    def __init__(self, order=None):
        self.order = order

    def rerank(self, query: str, documents: list[str], top_n: int):
        if self.order is not None:
            return self.order[:top_n]
        return [(i, 1.0 - 0.01 * i) for i in range(len(documents))][:top_n]


def _mk(cid: str, text: str, parent_id: str | None = None, is_parent: bool = False) -> Chunk:
    c = Chunk(
        chunk_id=cid,
        doc_id=cid.split(":")[0],
        parent_id=parent_id,
        is_parent=is_parent,
        text=text,
        raw_text=text,
        title_path=["标题"],
        source="t.md",
    )
    c.tokens = max(1, len(text) // 2)
    return c


CHUNKS = [
    _mk("d0:c0000", "华东区 2024 年带宽采购单价为 3.2 万元/Gbps。", parent_id="d0:p000"),
    _mk("d0:c0001", "华东区 2024 年带宽采购总量为 800Gbps。", parent_id="d0:p000"),
    _mk("d1:c0000", "华北区 2024 年带宽采购单价为 2.8 万元/Gbps。", parent_id="d1:p000"),
    _mk("d0:p000", "父块：华东区带宽采购情况汇总。", is_parent=True),
    _mk("d1:p000", "父块：华北区带宽采购情况汇总。", is_parent=True),
]


@pytest.fixture
def store(cfg, tmp_path):
    cfg.index.path = str(tmp_path / "idx_agent")
    cfg.index.backend = "memory"
    st = IndexStore(cfg)
    st.build(CHUNKS)
    return st


@pytest.fixture
def deps(cfg, store):
    return GraphDeps(
        llm_light=FakeLLM(),
        llm_main=FakeLLM(),
        retriever=HybridRetriever(cfg, store, MockEmbedding(dim=64)),
        reranker=FakeRerank(),
        store=store,
        embedder=MockEmbedding(dim=64),
    )


@pytest.fixture
def ctx_factory(cfg, deps, tmp_path):
    cfg.observability.trace_dir = str(tmp_path / "traces")

    def _make(emit=None, stream=False, query_id="q1"):
        return NodeContext(
            cfg=cfg,
            deps=deps,
            tracer=Tracer(query_id=query_id, cfg=cfg),
            query_id=query_id,
            emit=emit,
            stream=stream,
        ).bind()

    return _make


def _cand(cid: str, vec: float | None = 0.9, fused: float = 0.03) -> Candidate:
    c = Candidate(chunk_id=cid, fused_score=fused)
    c.vector_score = vec
    return c


# ── 条件边 ────────────────────────────────────────────────────────────


def test_dispatch_by_intent_valid():
    cfg_of = {"configurable": {"ctx": None}}
    assert dispatch_by_intent({"intent": "multi_hop"}, cfg_of) == "multi_hop"


def test_dispatch_by_intent_falls_back_to_simple_not_chitchat():
    """误判成 chitchat 会把问题挡在门外，误判成 simple 只是多花一次检索。"""
    cfg_of = {"configurable": {"ctx": None}}
    assert dispatch_by_intent({}, cfg_of) == "simple"
    assert dispatch_by_intent({"intent": "不存在的意图"}, cfg_of) == "simple"
    assert dispatch_by_intent({"intent": None}, cfg_of) == "simple"


def test_reaches_floor_rejects_none_score():
    class R:
        sufficiency_vector_floor = 0.35

    assert _reaches_floor(_cand("a", vec=0.9), R()) is True
    assert _reaches_floor(_cand("a", vec=0.10), R()) is False
    assert _reaches_floor(_cand("a", vec=None), R()) is False  # 单路失败不能当成 0.0 达标


@pytest.mark.parametrize(
    "state,expected",
    [
        ({"sufficient": True}, "sufficient"),
        ({"sufficient": False, "reflect_strong": 0}, "degrade"),
        ({"sufficient": False, "reflect_strong": 1, "intent": "simple"}, "sufficient"),
        ({"sufficient": False, "reflect_strong": 1, "intent": "multi_hop", "retrieve_iterations": 3}, "sufficient"),
        ({"sufficient": False, "reflect_strong": 1, "intent": "multi_hop", "retrieve_iterations": 1}, "retry"),
    ],
)
def test_is_sufficient_branches(state, expected, cfg):
    ctx = NodeContext(cfg=cfg, deps=None, tracer=Tracer("x", cfg), query_id="x")
    assert is_sufficient(state, {"configurable": {"ctx": ctx}}) == expected


def test_is_sufficient_allows_reflect_for_simple_when_enabled(cfg):
    cfg.agent.allow_reflect_for_simple = True
    ctx = NodeContext(cfg=cfg, deps=None, tracer=Tracer("x", cfg), query_id="x")
    state = {"sufficient": False, "reflect_strong": 1, "intent": "simple", "retrieve_iterations": 1}
    assert is_sufficient(state, {"configurable": {"ctx": ctx}}) == "retry"


def test_need_regenerate_defaults_to_low_confidence(cfg):
    """重生成默认关闭，未达标答案如实标注而不是自动重来。"""
    cfg.generation.faithfulness_threshold = 0.8
    cfg.generation.regenerate_on_unsupported = False
    ctx = NodeContext(cfg=cfg, deps=None, tracer=Tracer("x", cfg), query_id="x")
    cfg_of = {"configurable": {"ctx": ctx}}
    assert need_regenerate({"faithfulness": 1.0}, cfg_of) == "accept"
    assert need_regenerate({"faithfulness": 0.5, "generate_iterations": 1}, cfg_of) == "low_confidence"


def test_need_regenerate_respects_budget(cfg):
    cfg.generation.faithfulness_threshold = 0.8
    cfg.generation.regenerate_on_unsupported = True
    ctx = NodeContext(cfg=cfg, deps=None, tracer=Tracer("x", cfg), query_id="x")
    cfg_of = {"configurable": {"ctx": ctx}}
    assert need_regenerate({"faithfulness": 0.5, "generate_iterations": 1}, cfg_of) == "regenerate"
    assert need_regenerate({"faithfulness": 0.5, "generate_iterations": 2}, cfg_of) == "low_confidence"


def test_should_regenerate_is_the_single_source_of_truth(cfg):
    """L10：条件边与 regenerate 事件共用它，不可能不一致。"""
    cfg.generation.regenerate_on_unsupported = True
    cfg.agent.max_generate_iterations = 2
    assert should_regenerate({"faithfulness": 0.5, "generate_iterations": 0}, cfg) is True
    assert should_regenerate({"faithfulness": 0.5, "generate_iterations": 2}, cfg) is False
    assert should_regenerate({"faithfulness": 0.9, "generate_iterations": 0}, cfg) is False
    assert should_regenerate({}, None) is False


# ── route_node ────────────────────────────────────────────────────────


def test_route_node_returns_valid_intent_and_route_confidence(ctx_factory, deps):
    deps.llm_light = FakeLLM({"intent": {"intent": "multi_hop", "confidence": 0.87, "reason": "跨两个指标"}})
    out = route_node({"query": "华东和华北哪个贵", "history": []}, {"configurable": {"ctx": ctx_factory()}})
    assert out["intent"] == "multi_hop"
    assert out["route_confidence"] == 0.87
    assert "confidence" not in out, "路由置信度绝不能写进答案级 confidence（H17）"


def test_route_node_rejects_intent_out_of_whitelist(ctx_factory, deps):
    deps.llm_light = FakeLLM({"intent": {"intent": "weather", "confidence": 0.9}})
    out = route_node({"query": "今天天气", "history": []}, {"configurable": {"ctx": ctx_factory()}})
    assert out["intent"] == "simple"
    assert "route_invalid_intent" in out["route_reason"]


def test_route_node_falls_back_on_llm_failure(ctx_factory, deps):
    deps.llm_light = FakeLLM(raise_on=("intent",))
    out = route_node({"query": "华东区单价", "history": []}, {"configurable": {"ctx": ctx_factory()}})
    assert out["intent"] == "simple"
    assert "route_failed_fallback" in out["route_reason"]


# ── rewrite_node ──────────────────────────────────────────────────────


def test_rewrite_node_uses_retrieve_iterations(ctx_factory, deps):
    deps.llm_light = FakeLLM({"queries": {"queries": ["华东区 带宽 采购 单价"]}})
    out = rewrite_node(
        {"query": "那个价格", "intent": "simple", "retrieve_iterations": 1},
        {"configurable": {"ctx": ctx_factory()}},
    )
    assert out["sub_queries"] == ["华东区 带宽 采购 单价"]
    assert out["transform_strategy"] == "rewrite"


def test_rewrite_node_first_round_skips_rewrite_for_simple(ctx_factory):
    out = rewrite_node(
        {"query": "华东区带宽采购单价", "intent": "simple", "retrieve_iterations": 0},
        {"configurable": {"ctx": ctx_factory()}},
    )
    assert out["sub_queries"] == ["华东区带宽采购单价"]
    assert out["transform_strategy"] == "none"


def test_rewrite_node_survives_llm_failure(ctx_factory, deps):
    deps.llm_light = FakeLLM(raise_on=("queries",))
    out = rewrite_node(
        {"query": "那个价格", "intent": "multi_hop", "retrieve_iterations": 0},
        {"configurable": {"ctx": ctx_factory()}},
    )
    assert out["sub_queries"] == ["那个价格"]


# ── retrieve_node ─────────────────────────────────────────────────────


def test_retrieve_node_increments_counter_and_sets_sufficiency(ctx_factory, cfg):
    """H4：自增写在执行动作的节点里；首轮返回 1 而不是 0。"""
    out = retrieve_node(
        {"query": "华东区带宽采购单价", "intent": "simple", "retrieve_iterations": 0},
        {"configurable": {"ctx": ctx_factory()}},
    )
    assert out["retrieve_iterations"] == 1
    assert out["sufficiency_basis"] == "candidate_count"  # Mock 向量的余弦没有跨语料量纲
    assert isinstance(out["reflect_strong"], int)
    assert out["reflect_reason"] in ("corpus_missing", "query_expression")
    assert out["candidates"]


def test_retrieve_node_uses_vector_floor_only_when_scores_are_calibrated(ctx_factory, cfg):
    """K1：判据只在余弦分有绝对量纲时成立；否则降级为候选数并写进 trace。"""
    cfg.dashscope.api_key = "sk-test"  # provider_mode → dashscope
    out = retrieve_node(
        {"query": "华东区带宽采购单价", "intent": "simple", "retrieve_iterations": 0},
        {"configurable": {"ctx": ctx_factory()}},
    )
    assert out["sufficiency_basis"] == "vector_score"
    assert out["reflect_strong"] <= len(out["candidates"])
    assert out["sufficient"] == (out["reflect_strong"] >= cfg.retrieval.min_candidates)


def test_retrieve_node_counts_only_candidates_reaching_floor(ctx_factory, cfg, deps, monkeypatch):
    """K1：达标判据必须有区分度——分数低的候选不能算数，否则反思循环永不触发。"""
    cfg.dashscope.api_key = "sk-test"
    cfg.retrieval.sufficiency_vector_floor = 0.35
    cfg.retrieval.min_candidates = 2
    cands = [_cand("a", vec=0.90), _cand("b", vec=0.10), _cand("c", vec=0.40)]
    monkeypatch.setattr(deps.retriever, "retrieve", lambda *a, **k: cands)

    out = retrieve_node(
        {"query": "华东区带宽采购单价", "intent": "simple", "retrieve_iterations": 0},
        {"configurable": {"ctx": ctx_factory()}},
    )
    assert out["sufficiency_basis"] == "vector_score"
    assert out["reflect_strong"] == 2, "0.90 与 0.40 达标，0.10 不达标"
    assert out["sufficient"] is True
    assert out["reflect_reason"] == "query_expression"


def test_retrieve_node_zero_strong_is_corpus_missing(ctx_factory, cfg, deps, monkeypatch):
    """L8：一条都没达标才是 corpus_missing，而不是"不够 min_candidates"。"""
    cfg.dashscope.api_key = "sk-test"
    cfg.retrieval.sufficiency_vector_floor = 0.9
    monkeypatch.setattr(deps.retriever, "retrieve", lambda *a, **k: [_cand("a", vec=0.1)])
    out = retrieve_node(
        {"query": "华东区带宽采购单价", "intent": "simple", "retrieve_iterations": 0},
        {"configurable": {"ctx": ctx_factory()}},
    )
    assert out["reflect_strong"] == 0
    assert out["reflect_reason"] == "corpus_missing"
    assert out["sufficient"] is False


def test_retrieve_node_degrades_when_vector_route_yields_no_score(ctx_factory, cfg, deps, monkeypatch):
    """向量路不可用（索引里没有向量 / 调用失败）时，BM25 的命中必须还能作答。

    否则一次 embedding 故障就会让系统对"BM25 明明命中的语料"整体拒答。
    """
    cfg.dashscope.api_key = "sk-test"
    monkeypatch.setattr(deps.retriever, "retrieve", lambda *a, **k: [_cand("a", vec=None), _cand("b", vec=None)])
    out = retrieve_node(
        {"query": "华东区带宽采购单价", "intent": "simple", "retrieve_iterations": 0},
        {"configurable": {"ctx": ctx_factory()}},
    )
    assert out["sufficiency_basis_reason"] == "vector_route_unavailable"
    assert out["sufficiency_basis"] == "candidate_count"
    assert out["reflect_strong"] == 2


def test_retrieve_node_degrades_basis_when_vector_disabled(ctx_factory, cfg):
    """K1：关掉向量路就没有余弦分可用，必须换判据并写进 trace，而不是让达标数恒 0。"""
    cfg.retrieval.use_vector = False
    out = retrieve_node(
        {"query": "华东区带宽采购单价", "intent": "simple", "retrieve_iterations": 0},
        {"configurable": {"ctx": ctx_factory()}},
    )
    assert out["sufficiency_basis"] == "candidate_count"
    assert out["reflect_strong"] == len(out["candidates"])


def test_retrieve_node_merges_across_rounds(ctx_factory):
    """L11：第二轮不许把首轮的命中挤掉。"""
    prev = [_cand("d0:c0000", vec=0.9)]
    out = retrieve_node(
        {
            "query": "华北区带宽采购单价",
            "intent": "simple",
            "retrieve_iterations": 1,
            "candidates": prev,
        },
        {"configurable": {"ctx": ctx_factory()}},
    )
    assert "d0:c0000" in {c.chunk_id for c in out["candidates"]}
    assert out["retrieve_iterations"] == 2


def test_retrieve_node_emits_reflect_and_fuse_once(ctx_factory):
    """H13：reflect 由节点推，fuse 只进 trace 不进 SSE，各一次。"""
    events: list[tuple[str, dict]] = []
    ctx = ctx_factory(emit=lambda t, d: events.append((t, d)))
    retrieve_node(
        {"query": "华东区带宽采购单价", "intent": "simple", "retrieve_iterations": 0},
        {"configurable": {"ctx": ctx}},
    )
    types = [t for t, _ in events]
    assert types.count("reflect") == 1
    assert "fuse" not in types  # fuse 只落 trace，不占 SSE 通道
    steps = [e.step for e in ctx.tracer.events]
    assert "retrieve" in steps and "fuse" in steps


def test_retrieve_node_hyde_keeps_original_query_for_bm25(ctx_factory, store, monkeypatch):
    """J7：HyDE 的假设文档只走向量路，BM25 路仍用原 query 做字面匹配。"""
    seen: list[str] = []
    orig = store.bm25.search
    monkeypatch.setattr(store.bm25, "search", lambda q, k: (seen.append(q), orig(q, k))[1])
    retrieve_node(
        {
            "query": "华东区单价",
            "sub_queries": ["华东区 2024 年带宽采购单价为 3.2 万元/Gbps，含运维服务。"],
            "intent": "simple",
            "retrieve_iterations": 2,
            "transform_strategy": "hyde",
        },
        {"configurable": {"ctx": ctx_factory()}},
    )
    assert seen == ["华东区单价"]


# ── rerank_node ───────────────────────────────────────────────────────


def test_rerank_node_collapses_same_parent(ctx_factory, cfg):
    """H18：同一父块只回填一次，去重后顺位取下一个不同父块的子块。"""
    cfg.generation.max_context_chunks = 5
    ctx = ctx_factory()
    out = rerank_node(
        {
            "query": "带宽采购单价",
            "candidates": [_cand("d0:c0000"), _cand("d0:c0001"), _cand("d1:c0000")],
        },
        {"configurable": {"ctx": ctx}},
    )
    parents = [sc.chunk.chunk_id for sc in out["context"]]
    assert parents == ["d0:p000", "d1:p000"]  # d0:c0001 与 d0:c0000 同父，被折叠
    ev = next(e for e in ctx.tracer.events if e.step == "rerank")
    assert ev.payload["dedup_collapsed"] == 1


def test_rerank_node_empty_candidates(ctx_factory):
    assert rerank_node({"query": "x", "candidates": []}, {"configurable": {"ctx": ctx_factory()}}) == {"context": []}


def test_rerank_node_falls_back_to_rrf_order(ctx_factory, cfg, deps):
    """rerank 失败不阻断：用 RRF 排序而非整条链路失败，且失败要写进 trace。"""

    class Boom:
        def rerank(self, *a, **k):
            raise RuntimeError("rerank 服务不可用")

    deps.reranker = Boom()
    ctx = ctx_factory()
    out = rerank_node(
        {"query": "带宽采购单价", "candidates": [_cand("d0:c0000"), _cand("d1:c0000")]},
        {"configurable": {"ctx": ctx}},
    )
    assert out["context"]
    ev = next(e for e in ctx.tracer.events if e.step == "rerank")
    assert ev.payload["rerank_failed"] is True
    assert "RuntimeError" in ev.payload["rerank_error"]


def test_rerank_node_respects_max_context_chunks(ctx_factory, cfg):
    cfg.generation.max_context_chunks = 1
    out = rerank_node(
        {"query": "带宽采购单价", "candidates": [_cand("d0:c0000"), _cand("d1:c0000")]},
        {"configurable": {"ctx": ctx_factory()}},
    )
    assert len(out["context"]) == 1


# ── generate_node ─────────────────────────────────────────────────────


def test_generate_node_increments_generate_iterations(ctx_factory):
    """H4：生成侧自增同样写在 generate_node 内，否则 1 < 1 恒假、重生成永不执行。"""
    out = generate_node(
        {"query": "华东区单价", "context": [], "generate_iterations": 0},
        {"configurable": {"ctx": ctx_factory()}},
    )
    assert out["generate_iterations"] == 1
    assert isinstance(out["draft"], str)


def test_generate_node_streams_tokens_and_parses_citations(ctx_factory):
    events: list[tuple[str, dict]] = []
    ctx = ctx_factory(emit=lambda t, d: events.append((t, d)), stream=True)
    sc = ScoredChunk(chunk=_mk("d0:c0000", "华东区单价 3.2 万元。"), score=0.9, rank=1)
    out = generate_node(
        {"query": "华东区单价", "context": [sc], "generate_iterations": 0},
        {"configurable": {"ctx": ctx}},
    )
    assert [t for t, _ in events].count("token") > 0
    assert out["citations"], "流式路径同样要在结束后解析引用"


def test_generate_node_non_stream_pushes_citation_events(ctx_factory):
    events: list[tuple[str, dict]] = []
    ctx = ctx_factory(emit=lambda t, d: events.append((t, d)))
    sc = ScoredChunk(chunk=_mk("d0:c0000", "华东区单价 3.2 万元。"), score=0.9, rank=1)
    generate_node(
        {"query": "华东区单价", "context": [sc], "generate_iterations": 0},
        {"configurable": {"ctx": ctx}},
    )
    assert any(t == "citation" for t, _ in events)


# ── verify_node ───────────────────────────────────────────────────────


def _verify(ctx, state, cfg, deps, **payloads):
    deps.llm_light = FakeLLM(payloads)
    return verify_node(state, {"configurable": {"ctx": ctx}})


def test_verify_node_computes_faithfulness_by_formula(ctx_factory, cfg, deps):
    """faithfulness 由代码算，不采信模型自报数字。"""
    out = _verify(
        ctx_factory(),
        {"draft": "华东区单价 3.2 万元。华北区单价 2.8 万元。", "context": [], "reflect_strong": 5},
        cfg,
        deps,
        claims={"claims": ["华东区单价 3.2 万元", "华北区单价 2.8 万元", "西南区单价 9.9 万元"]},
        unsupported={"unsupported": [2]},
    )
    assert out["faithfulness"] == round(1 - 1 / 3, 6)
    assert out["unsupported"] == [2]


def test_verify_node_confidence_high(ctx_factory, cfg, deps):
    out = _verify(
        ctx_factory(),
        {"draft": "A。", "context": [], "reflect_strong": 5},
        cfg,
        deps,
        claims={"claims": ["A"]},
        unsupported={"unsupported": []},
    )
    assert out["confidence"] == "high"


def test_verify_node_confidence_medium_when_evidence_thin(ctx_factory, cfg, deps):
    """L5：有证据但不足 min_candidates，照常生成并标 medium，不拒答。"""
    cfg.retrieval.min_candidates = 3
    out = _verify(
        ctx_factory(),
        {"draft": "A。", "context": [], "reflect_strong": 1},
        cfg,
        deps,
        claims={"claims": ["A"]},
        unsupported={"unsupported": []},
    )
    assert out["confidence"] == "medium"
    assert out["weak_evidence"] is True


def test_verify_node_confidence_medium_when_unsupported(ctx_factory, cfg, deps):
    out = _verify(
        ctx_factory(),
        {"draft": "A。B。", "context": [], "reflect_strong": 5},
        cfg,
        deps,
        claims={"claims": ["A", "B"]},
        unsupported={"unsupported": [1]},
    )
    assert out["confidence"] == "medium"


def test_verify_node_confidence_medium_when_verify_failed(ctx_factory, cfg, deps):
    deps.llm_light = FakeLLM(raise_on=("claims",))
    out = verify_node({"draft": "A。", "context": [], "reflect_strong": 5}, {"configurable": {"ctx": ctx_factory()}})
    assert out["verify_failed"] is True
    assert out["confidence"] == "medium"
    assert out["faithfulness"] == 1.0  # 没拆出命题不能当成"全部无支撑"


def test_verify_node_confidence_low_when_degraded(ctx_factory, cfg, deps):
    out = _verify(
        ctx_factory(),
        {"draft": "未找到", "context": [], "reflect_strong": 0, "degraded": True},
        cfg,
        deps,
        claims={"claims": ["A"]},
        unsupported={"unsupported": []},
    )
    assert out["confidence"] == "low"


def test_verify_node_pushes_regenerate_only_when_judged(ctx_factory, cfg, deps):
    """L10：regenerate 事件由 verify 推，且与条件边共用判据。"""
    cfg.generation.regenerate_on_unsupported = True
    cfg.generation.faithfulness_threshold = 0.8
    events: list[tuple[str, dict]] = []
    ctx = ctx_factory(emit=lambda t, d: events.append((t, d)))
    _verify(
        ctx,
        {"draft": "A。B。", "context": [], "reflect_strong": 5, "generate_iterations": 1},
        cfg,
        deps,
        claims={"claims": ["A", "B"]},
        unsupported={"unsupported": [1]},
    )
    assert [t for t, _ in events] == ["verify", "regenerate"]  # 先出结论，再声明动作
    assert should_regenerate({"faithfulness": 0.5, "generate_iterations": 1}, cfg) is True


def test_verify_node_no_duplicate_verify_event(ctx_factory, cfg, deps):
    """verify 事件由 tracer.span 退出时发出，节点不得再 push 一次。"""
    events: list[tuple[str, dict]] = []
    ctx = ctx_factory(emit=lambda t, d: events.append((t, d)))
    _verify(ctx, {"draft": "A。", "context": [], "reflect_strong": 5}, cfg, deps, claims={"claims": ["A"]})
    assert [t for t, _ in events].count("verify") == 1


# ── 终止应答 ──────────────────────────────────────────────────────────


def test_direct_answer_is_preset_template(ctx_factory, deps):
    before = deps.llm_main.calls
    out = direct_answer_node({"intent": "chitchat", "query": "你好"}, {"configurable": {"ctx": ctx_factory()}})
    assert out["confidence"] == "high"
    assert out["citations"] == []
    assert deps.llm_main.calls == before, "chitchat 不调生成模型"


def test_degrade_answer_corpus_missing(ctx_factory):
    out = degrade_answer_node(
        {"reflect_reason": "corpus_missing", "candidates": [], "retrieve_iterations": 1},
        {"configurable": {"ctx": ctx_factory()}},
    )
    assert out["degraded"] is True
    assert out["confidence"] == "low"
    assert "未在资料中找到" in out["answer"]


def test_degrade_answer_lists_near_miss_candidates(ctx_factory):
    out = degrade_answer_node(
        {
            "reflect_reason": "corpus_missing",
            "candidates": [_cand("d0:c0000")],
            "retrieve_iterations": 1,
        },
        {"configurable": {"ctx": ctx_factory()}},
    )
    assert "字面相近" in out["answer"]


def test_degrade_answer_query_expression_reason(ctx_factory):
    out = degrade_answer_node(
        {"reflect_reason": "query_expression", "candidates": [], "retrieve_iterations": 2},
        {"configurable": {"ctx": ctx_factory()}},
    )
    assert "不足" in out["answer"]


# ── 图结构 ────────────────────────────────────────────────────────────


def test_graph_compiles_with_eight_nodes(cfg, deps):
    g = build_graph(cfg, deps)
    assert set(g.get_graph().nodes) >= {
        "route",
        "rewrite",
        "retrieve",
        "rerank",
        "generate",
        "verify",
        "direct_answer",
        "degrade_answer",
    }


def test_initial_state_counters_start_at_zero():
    st = initial_state("问题")
    assert st["retrieve_iterations"] == 0
    assert st["generate_iterations"] == 0
    assert st["history"] == []


def test_nodes_take_config_argument(cfg, deps):
    """H21：节点必须是 (state, config) 签名，否则 configurable 注入静默失效。"""
    import inspect

    for node in (route_node, rewrite_node, retrieve_node, rerank_node, generate_node, verify_node):
        params = list(inspect.signature(node).parameters)
        assert params == ["state", "config"], f"{node.__name__} 签名异常：{params}"


def test_rerank_node_pool_depth_is_rerank_top_k(ctx_factory, cfg):
    """`rerank_top_k` 必须真的生效：它是回填与顺位补齐的池子深度。

    早先代码直接遍历全部 ordered，只靠 max_context_chunks 截断，
    单独调 rerank_top_k 没有任何效果（死配置）。
    """
    cfg.retrieval.rerank_top_k = 1
    ctx = ctx_factory()
    out = rerank_node(
        {
            "query": "带宽采购单价",
            "candidates": [_cand("d0:c0000"), _cand("d0:c0001"), _cand("d1:c0000")],
        },
        {"configurable": {"ctx": ctx}},
    )
    ev = next(e for e in ctx.tracer.events if e.step == "rerank")
    assert ev.payload["requested"] == 3
    assert ev.payload["pool_depth"] == 1
    assert len(out["context"]) == 1  # 池子只有 1 条，去重补齐无从谈起
