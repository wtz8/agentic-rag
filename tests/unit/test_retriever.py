"""混合检索（04 §4）：两路并行、单路失败降级、过滤下推的位置差异。"""

from __future__ import annotations

import pytest

from agentic_rag.index.store import IndexStore
from agentic_rag.providers.base import ProviderError
from agentic_rag.providers.embedding import MockEmbedding
from agentic_rag.retrieval.retriever import HybridRetriever, _attach_route_scores
from agentic_rag.schema import Candidate, Chunk


def _mk(cid: str, text: str, block_type: str = "paragraph", parent_id: str | None = None) -> Chunk:
    c = Chunk(
        chunk_id=cid,
        doc_id=cid.split(":")[0],
        parent_id=parent_id,
        text=text,
        raw_text=text,
        title_path=["标题"],
        block_type=block_type,
        source="t.md",
    )
    c.tokens = max(1, len(text) // 2)
    return c


CHUNKS = [
    _mk("d0:c0000", "华东区 2024 年带宽采购单价为 3.2 万元/Gbps。"),
    _mk("d1:c0000", "华北区 2024 年带宽采购单价为 2.8 万元/Gbps。"),
    _mk("d2:c0000", "设备 XR-2400 支持双电源冗余，最大吞吐 400Gbps。"),
    _mk("d3:c0000", "| 区域 | 单价 |\n| --- | --- |\n| 华东 | 3.2 |", block_type="table"),
]


@pytest.fixture
def store(cfg, tmp_path):
    cfg.index.path = str(tmp_path / "idx_ret")
    cfg.index.backend = "memory"
    st = IndexStore(cfg)
    st.build(CHUNKS)
    return st


@pytest.fixture
def retriever(cfg, store):
    return HybridRetriever(cfg, store, MockEmbedding(dim=64))


# ── 基本召回 ──────────────────────────────────────────────────────────


def test_retrieve_returns_candidates_from_both_routes(retriever):
    diag: dict = {}
    out = retriever.retrieve(["华东区带宽采购单价"], diag=diag)
    assert out, "混合两路都应有命中"
    ids = {c.chunk_id for c in out}
    assert "d0:c0000" in ids
    assert set(diag["routes"]) == {"vector", "bm25"}
    assert diag["vector_hits"] == 1 and diag["bm25_hits"] == 1


def test_retrieve_attaches_route_scores(retriever):
    out = retriever.retrieve(["华东区带宽采购单价"])
    with_vec = [c for c in out if c.vector_score is not None]
    with_bm = [c for c in out if c.bm25_score is not None]
    assert with_vec and with_bm
    assert all(c.vector_rank and c.bm25_rank for c in with_vec if c.bm25_score is not None)


def test_retrieve_empty_queries_returns_empty(retriever):
    assert retriever.retrieve([]) == []
    assert retriever.retrieve(["", "  "]) == []


def test_retrieve_respects_fusion_top_k(cfg, retriever):
    cfg.retrieval.fusion_top_k = 2
    assert len(retriever.retrieve(["带宽"])) <= 2


# ── 单路开关与降级 ────────────────────────────────────────────────────


def test_use_vector_false_leaves_vector_score_none(cfg, store):
    cfg.retrieval.use_vector = False
    r = HybridRetriever(cfg, store, MockEmbedding(dim=64))
    out = r.retrieve(["华东区带宽采购单价"])
    assert out
    assert all(c.vector_score is None for c in out), "关掉向量路就不能拿 None 当 0 参与比较（L19）"
    assert any(c.bm25_score is not None for c in out)


def test_use_bm25_false_leaves_bm25_score_none(cfg, store):
    cfg.retrieval.use_bm25 = False
    r = HybridRetriever(cfg, store, MockEmbedding(dim=64))
    out = r.retrieve(["华东区带宽采购单价"])
    assert out
    assert all(c.bm25_score is None for c in out)


def test_vector_route_failure_does_not_block_bm25(cfg, store, monkeypatch):
    """单路失败不阻断：向量库挂了仍返回 BM25 结果（04 §4）。"""
    diag: dict = {}

    class BoomEmbedder:
        dim = 64

        def embed(self, texts):
            raise ProviderError("向量服务不可用")

    r = HybridRetriever(cfg, store, BoomEmbedder())
    out = r.retrieve(["华东区带宽采购单价"], diag=diag)
    assert out
    assert diag["errors"].get("vector")
    assert diag["routes"] == ["bm25"]
    assert diag["vector_hits"] == 0


def test_bm25_route_failure_does_not_block_vector(cfg, store, monkeypatch):
    diag: dict = {}
    monkeypatch.setattr(store, "bm25_available", True)
    monkeypatch.setattr(store.bm25, "search", lambda q, k: (_ for _ in ()).throw(RuntimeError("倒排损坏")))
    r = HybridRetriever(cfg, store, MockEmbedding(dim=64))
    out = r.retrieve(["华东区带宽采购单价"], diag=diag)
    assert out
    assert diag["errors"].get("bm25")
    assert diag["routes"] == ["vector"]


def test_bm25_unavailable_is_not_an_error(cfg, store, monkeypatch):
    monkeypatch.setattr(store, "bm25_available", False)
    r = HybridRetriever(cfg, store, MockEmbedding(dim=64))
    diag: dict = {}
    r.retrieve(["带宽"], diag=diag)
    assert "bm25" not in diag.get("errors", {})


# ── 过滤下推与表格加权 ────────────────────────────────────────────────


def test_filter_tables_only_keeps_only_tables(cfg, retriever):
    """table_numeric 且开启过滤时，非表格块必须被后置筛掉（BM25 路没有前置过滤）。"""
    cfg.retrieval.filter_tables_only = True
    diag: dict = {}
    out = retriever.retrieve(["单价"], intent="table_numeric", diag=diag)
    assert out
    assert all(c.chunk_id == "d3:c0000" for c in out)
    assert diag["filter_pushdown"] is True


def test_filter_not_applied_to_other_intents(cfg, retriever):
    cfg.retrieval.filter_tables_only = True
    diag: dict = {}
    retriever.retrieve(["单价"], intent="simple", diag=diag)
    assert diag["filter_pushdown"] is False


def test_table_boost_raises_table_score(cfg, retriever):
    cfg.retrieval.table_boost = 2.0
    plain = {c.chunk_id: c.fused_score for c in retriever.retrieve(["单价"], intent="simple")}
    boosted = {c.chunk_id: c.fused_score for c in retriever.retrieve(["单价"], intent="table_numeric")}
    if "d3:c0000" in plain and "d3:c0000" in boosted:
        assert boosted["d3:c0000"] > plain["d3:c0000"]


# ── 多子查询并发 ──────────────────────────────────────────────────────


def test_multiple_queries_are_all_searched(retriever):
    diag: dict = {}
    out = retriever.retrieve(["华东区带宽单价", "设备 XR-2400 吞吐"], diag=diag)
    ids = {c.chunk_id for c in out}
    assert "d0:c0000" in ids and "d2:c0000" in ids


def test_bm25_query_override_applies_only_to_bm25_route(cfg, store):
    """HyDE：向量路用假设文档，BM25 路用原 query（J7）。"""
    seen: dict = {}

    orig = store.bm25.search

    def spy(q, k):
        seen["bm25_query"] = q
        return orig(q, k)

    store.bm25.search = spy
    r = HybridRetriever(cfg, store, MockEmbedding(dim=64))
    r.retrieve(["假设文档：华东区带宽采购单价为 3.2 万元"], bm25_query="华东区单价")
    assert seen["bm25_query"] == "华东区单价"


def test_attach_route_scores_keeps_best_rank():
    cs = [Candidate(chunk_id="a"), Candidate(chunk_id="b")]
    pairs = [
        ([("a", 0.5), ("b", 0.4)], [("b", 8.0)]),
        ([("a", 0.9)], []),
    ]
    _attach_route_scores(cs, pairs)
    by = {c.chunk_id: c for c in cs}
    assert by["a"].vector_score == 0.9 and by["a"].vector_rank == 1  # 取更高分对应的排名
    assert by["b"].vector_score == 0.4 and by["b"].vector_rank == 2
    assert by["b"].bm25_score == 8.0
    assert sorted(by["b"].hit_routes) == ["bm25", "vector"]
    assert by["a"].hit_routes == ["vector"]


def test_retrieve_deduplicates_across_routes(retriever):
    out = retriever.retrieve(["华东区带宽采购单价"])
    ids = [c.chunk_id for c in out]
    assert len(ids) == len(set(ids))
    assert out == sorted(out, key=lambda c: -c.fused_score)


# ── 并发子查询的 diag 合并（回归：早先并发分支传 {} 导致 durations/errors 全丢）──


class _BoomEmbedder:
    """只在特定查询上让向量路失败——用来验证并发下错误不会被互相覆盖。"""

    def __init__(self, fail_on: str):
        self.fail_on = fail_on
        self.dim = 64

    def embed(self, texts):
        out = []
        for t in texts:
            if t == self.fail_on:
                raise ProviderError("vector route down")
            out.append(MockEmbedding(dim=self.dim).embed([t])[0])
        return out


def test_multi_query_merges_diag_durations(cfg, store):
    r = HybridRetriever(cfg, store, MockEmbedding(dim=64))
    diag: dict = {}
    r.retrieve(["带宽采购单价", "设备 XR-2400", "华北区"], diag=diag)
    # 三条子查询 → 三条 duration；早先并发分支把 diag 丢进 {}，这里是 []
    assert len(diag.get("durations", [])) == 3, diag
    assert diag["sub_query_count"] == 3


def test_multi_query_errors_are_not_overwritten(cfg, store):
    r = HybridRetriever(cfg, store, _BoomEmbedder(fail_on="设备 XR-2400"))
    diag: dict = {}
    r.retrieve(["带宽采购单价", "设备 XR-2400"], diag=diag)
    # 只有第 2 条子查询的向量路失败；key 必须带下标，否则同名路由互相覆盖
    assert list(diag["errors"]) == ["vector@1"], diag
    assert diag["vector_hits"] == 1 and diag["bm25_hits"] == 2
