"""查询变换（04 §2 / L2）。

策略升级只看 `retrieve_iterations`，不看上一轮用了什么——
这是 hyde 能被真正触发的必要条件，单测必须把 0/1/2 三级都走一遍。
"""

from __future__ import annotations

import json

from agentic_rag.config import Config
from agentic_rag.providers.base import ProviderError
from agentic_rag.retrieval.query_transform import TransformResult, _pick_strategy, transform_query


class FakeLLM:
    """按响应表返回 JSON；`raise_on` 里的模板名用来模拟该步失败。"""

    def __init__(self, responses: dict[str, dict] | None = None, raise_on: set[str] | None = None):
        self.responses = responses or {}
        self.raise_on = raise_on or set()
        self.seen: list[str] = []

    @property
    def model_name(self) -> str:
        return "fake"

    def chat(self, messages, temperature=0.0, max_tokens=2048, json_mode=False):
        user = messages[-1]["content"]
        name = next((k for k in self.responses if k in user), "")
        self.seen.append(name)
        if name in self.raise_on:
            raise ProviderError(f"{name} 失败")
        return json.dumps(self.responses.get(name, {}), ensure_ascii=False)

    def stream(self, messages, temperature=0.0, max_tokens=2048):
        yield ""


# ── 策略选择（L2）─────────────────────────────────────────────────────


def test_pick_strategy_iteration_zero_follows_intent(cfg: Config):
    assert _pick_strategy("simple", 0, cfg) == "none"  # skip_rewrite_for_simple 默认开
    assert _pick_strategy("multi_hop", 0, cfg) == "decompose"
    assert _pick_strategy("table_numeric", 0, cfg) == "rewrite"


def test_pick_strategy_iteration_zero_simple_without_skip(cfg: Config):
    cfg.agent.skip_rewrite_for_simple = False
    assert _pick_strategy("simple", 0, cfg) == "rewrite"


def test_pick_strategy_iteration_one_is_rewrite_regardless_of_intent(cfg: Config):
    """第一次重试一律换措辞——成本最低的另一种假设。"""
    for intent in ("simple", "multi_hop", "table_numeric"):
        assert _pick_strategy(intent, 1, cfg) == "rewrite"


def test_pick_strategy_iteration_two_escalates_to_hyde(cfg: Config):
    """第二次重试换范式，而不是再改一遍措辞。"""
    for intent in ("simple", "multi_hop", "table_numeric"):
        assert _pick_strategy(intent, 2, cfg) == "hyde"


def test_hyde_is_reachable_within_retrieval_budget(cfg: Config):
    """L2 的关键后果：max_retrieve_iterations=3 时 hyde 必须至少轮到一次。"""
    strategies = [_pick_strategy("multi_hop", i, cfg) for i in range(cfg.agent.max_retrieve_iterations)]
    assert strategies == ["decompose", "rewrite", "hyde"]


# ── 各策略的产出与 bm25_query 语义 ────────────────────────────────────


def test_strategy_none_keeps_query_and_feeds_bm25(cfg: Config):
    r = transform_query("华东区单价", "simple", FakeLLM(), iteration=0, cfg=cfg)
    assert r.strategy == "none"
    assert r.queries == ["华东区单价"]
    assert r.bm25_query == "华东区单价"


def test_rewrite_returns_single_query(cfg: Config):
    llm = FakeLLM({"queries": {"queries": ["华东区 2024 年 带宽 采购 单价"]}})
    r = transform_query("那个价格是多少", "simple", llm, iteration=1, cfg=cfg)
    assert r.strategy == "rewrite"
    assert r.queries == ["华东区 2024 年 带宽 采购 单价"]


def test_hyde_uses_hypothetical_doc_for_vector_and_original_for_bm25(cfg: Config):
    """J7：假设文档只走向量路，字面匹配仍用原 query。"""
    para = "华东区 2024 年带宽采购单价为 3.2 万元/Gbps，该价格包含运维服务费。" * 3
    llm = FakeLLM({"queries": {"queries": [para]}})
    r = transform_query("华东区单价", "simple", llm, iteration=2, cfg=cfg)
    assert r.strategy == "hyde"
    assert r.queries == [para]
    assert r.bm25_query == "华东区单价"


def test_hyde_falls_back_to_rewrite_when_paragraph_unusable(cfg: Config):
    """HyDE 产出太短/等于原问题时退回 rewrite，而不是拿无效假设文档去检索。"""
    llm = FakeLLM({"queries": {"queries": ["太短"]}})
    r = transform_query("华东区单价", "simple", llm, iteration=2, cfg=cfg)
    assert r.strategy == "rewrite"
    assert r.bm25_query is None


def test_decompose_returns_multiple_queries_and_no_bm25_query(cfg: Config):
    """decompose 时每条子查询各自走 BM25，不能拿原 query 覆盖（04 §4）。"""
    llm = FakeLLM({"queries": {"queries": ["A 指标", "B 指标"]}})
    r = transform_query("A 和 B 哪个高", "multi_hop", llm, iteration=0, cfg=cfg)
    assert r.strategy == "decompose"
    assert r.queries == ["A 指标", "B 指标"]
    assert r.bm25_query is None


def test_decompose_degrades_to_single_hop_when_only_one_subquery(cfg: Config):
    llm = FakeLLM({"queries": {"queries": ["只有一个"]}})
    r = transform_query("A 和 B 哪个高", "multi_hop", llm, iteration=0, cfg=cfg)
    assert r.strategy == "rewrite"  # 2 ≤ n ≤ max_sub_queries 不成立
    assert r.queries == ["只有一个"]


def test_decompose_truncates_to_max_sub_queries(cfg: Config):
    cfg.agent.max_sub_queries = 2
    llm = FakeLLM({"queries": {"queries": ["a", "b", "c", "d"]}})
    r = transform_query("多跳问题", "multi_hop", llm, iteration=0, cfg=cfg)
    assert len(r.queries) == 2


# ── 失败回落 ──────────────────────────────────────────────────────────


def test_llm_failure_falls_back_to_original_query(cfg: Config):
    """改写环节绝不能成为链路的单点故障。"""
    llm = FakeLLM({"queries": {"queries": ["改写结果"]}}, raise_on={"queries"})
    r = transform_query("原问题", "simple", llm, iteration=1, cfg=cfg)
    assert r.queries == ["原问题"]
    assert r.strategy == "rewrite"


def test_malformed_json_falls_back_to_original_query(cfg: Config):
    class BadLLM:
        model_name = "bad"

        def chat(self, messages, temperature=0.0, max_tokens=2048, json_mode=False):
            return "不是 JSON"

        def stream(self, *a, **k):
            yield ""

    r = transform_query("原问题", "simple", BadLLM(), iteration=1, cfg=cfg)
    assert r.queries == ["原问题"]


def test_history_is_passed_to_transform(cfg: Config):
    llm = FakeLLM({"queries": {"queries": ["改写"]}})
    transform_query("它多少钱", "simple", llm, history=[{"role": "user", "content": "华东区单价"}], iteration=1, cfg=cfg)
    assert llm.seen == ["queries"]


def test_transform_result_defaults():
    r = TransformResult(["q"], "none")
    assert r.bm25_query is None


def test_rewrite_has_no_bm25_override(cfg: Config):
    """rewrite 的改写结果要同时喂两路，`bm25_query` 必须是 None。

    填原 query 会让两处口径分叉：消费方一旦开始认这个字段，
    改写就突然变成"只走向量路"，BM25 那边还在用没改写的原问题。
    """
    llm = FakeLLM({"queries": {"queries": ["华东区 2024 年 带宽 采购 单价"]}})
    r = transform_query("那个价格是多少", "simple", llm, iteration=1, cfg=cfg)
    assert r.strategy == "rewrite"
    assert r.bm25_query is None
