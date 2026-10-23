"""在线端到端：真实图 → 真实缓存 → 真实 trace（04 §6 / §10）。

与单测的分工：单测保证每个节点走对分支，这里保证**它们连起来还是那个系统**——
迭代次数、终止出口、事件序列、缓存键、trace 落盘。
"""

from __future__ import annotations

import pytest

from agentic_rag.cache.result_cache import ResultCache
from agentic_rag.observability.trace import load_trace
from agentic_rag.pipeline import EVENT_TYPES, Event
from helpers import ScriptedLLM


def _types(events) -> list[str]:
    return [e.type for e in events]


# ── 主链路 ────────────────────────────────────────────────────────────


def test_simple_query_runs_main_path_once(icfg, store_with_corpus, pipeline_factory):
    p = pipeline_factory(store_with_corpus)
    r = p.run("华东区带宽采购单价是多少")
    assert r.answer
    assert r.retrieve_iterations == 1, "证据充分时只检索一轮（H4：首轮结束即为 1 而非 0）"
    assert r.generate_iterations == 1
    assert r.confidence in ("high", "medium", "low")
    assert r.trace_id
    assert not r.degraded


def test_main_path_event_sequence(icfg, store_with_corpus, pipeline_factory):
    p = pipeline_factory(store_with_corpus)
    events: list[Event] = []
    r = p._execute("华东区带宽采购单价", None, emit=lambda t, d: events.append(Event(t, d)), stream=False)
    seq = _types(events)
    # SSE 侧：04 §10.2 的协议里没有 `generate`，生成由 token 表示
    for step in ("route", "rewrite", "retrieve", "reflect", "rerank", "verify"):
        assert step in seq, f"缺少事件 {step}：{seq}"
    assert seq.index("route") < seq.index("retrieve") < seq.index("verify")
    assert all(t in EVENT_TYPES for t in seq), f"协议外事件类型：{set(seq) - set(EVENT_TYPES)}"

    # trace 侧：generate 是完整的一步（首字延迟与迭代序号在这里）
    row = load_trace(r.trace_id, icfg)
    for step in ("route", "rewrite", "retrieve", "fuse", "rerank", "generate", "verify"):
        assert step in row["steps"], f"trace 缺少 {step}：{row['steps']}"
    assert r.confidence in ("high", "medium", "low")


def test_trace_is_written_before_done(icfg, store_with_corpus, pipeline_factory):
    """done 必须在 trace 落盘之后推，否则用户立刻点进 /api/trace/{id} 会查不到。"""
    p = pipeline_factory(store_with_corpus)
    r = p.run("华东区带宽采购单价")
    row = load_trace(r.trace_id, icfg)
    assert row is not None
    assert "fuse" in row["steps"], "融合是独立一步，必须可观测"
    assert "retrieve" in row["steps"] and "verify" in row["steps"]


def test_stream_yields_tokens_and_done(icfg, store_with_corpus, pipeline_factory):
    p = pipeline_factory(store_with_corpus, llm_main=ScriptedLLM())
    events = list(p.run_stream("华东区带宽采购单价"))
    seq = _types(events)
    assert "token" in seq
    assert seq[-1] == "done"
    done = events[-1].data
    assert done["trace_id"]
    assert done["confidence"] in ("high", "medium", "low")
    assert "".join(e.data["text"] for e in events if e.type == "token")


def test_stream_event_is_serializable_as_sse(icfg, store_with_corpus, pipeline_factory):
    p = pipeline_factory(store_with_corpus, llm_main=ScriptedLLM())
    for ev in p.run_stream("华东区带宽采购单价"):
        s = ev.to_sse()
        assert s.startswith(f"event: {ev.type}\ndata: ")
        assert s.endswith("\n\n")


def test_events_have_single_sender(icfg, store_with_corpus, pipeline_factory):
    """H13：同一事件不得有两个发送方（trace 与 SSE 各推一份 = 前端重复渲染）。"""
    p = pipeline_factory(store_with_corpus)
    events: list[Event] = []
    p._execute("华东区带宽采购单价", None, emit=lambda t, d: events.append(Event(t, d)), stream=False)
    seq = _types(events)
    for t in ("route", "retrieve", "rerank", "verify"):
        assert seq.count(t) == 1, f"{t} 出现 {seq.count(t)} 次"
    assert seq.count("reflect") == 1


# ── 反思循环与终止（L2 / L5 / K1）─────────────────────────────────────


def test_reflection_runs_three_rounds_then_answers_with_weak_evidence(icfg, tiny_store, pipeline_factory):
    """证据够不上 min_candidates 但也不是零：跑满预算后照常作答并标 medium（L5）。"""
    icfg.retrieval.min_candidates = 3  # 索引里只有 2 个 chunk → 永远够不上
    icfg.agent.allow_reflect_for_simple = True
    p = pipeline_factory(tiny_store)
    r = p.run("带宽采购单价")
    assert r.retrieve_iterations == icfg.agent.max_retrieve_iterations == 3
    assert not r.degraded, "有证据就不能拒答"
    assert r.weak_evidence is True
    assert r.confidence == "medium"


def test_reflection_escalates_to_hyde(icfg, tiny_store, pipeline_factory):
    """L2：策略升级只看迭代序号，第三轮必须轮到 hyde，否则它永远不可达。"""
    icfg.retrieval.min_candidates = 3
    icfg.agent.allow_reflect_for_simple = True
    p = pipeline_factory(tiny_store, llm_light=ScriptedLLM())
    r = p.run("带宽采购单价")
    row = load_trace(r.trace_id, icfg)
    strategies = [e["payload"].get("strategy") for e in row["events"] if e["step"] == "rewrite"]
    assert strategies[-1] == "hyde", f"三级升级未生效：{strategies}"


def test_degrade_when_corpus_has_nothing_relevant(icfg, tiny_store, pipeline_factory):
    """L5 / L8：一条候选都没有才拒答，且与迭代次数无关。"""
    icfg.retrieval.use_vector = False  # 只留 BM25，查询词在语料里根本不存在
    p = pipeline_factory(tiny_store)
    r = p.run("量子计算原型机")
    assert r.degraded is True
    assert r.retrieve_iterations == 1, "零证据应立刻拒答，不必跑满预算"
    assert r.confidence == "low"
    assert "未在资料中找到" in r.answer


def test_sufficiency_basis_is_recorded_and_falls_back_without_vector(icfg, tiny_store, pipeline_factory):
    """K1：判据只在余弦分有绝对量纲时成立；否则降级为候选数，并写进 trace。"""
    p = pipeline_factory(tiny_store)
    r = p.run("带宽采购单价")
    row = load_trace(r.trace_id, icfg)
    reflect = [e["payload"] for e in row["events"] if e["step"] == "reflect"]
    assert reflect
    # Mock 向量的余弦被块长稀释，没有跨语料可比量纲 → 与关闭向量路同样处理
    assert reflect[0]["sufficiency_basis"] == "candidate_count"
    assert reflect[0]["sufficiency_floor"] is None

    icfg.retrieval.use_vector = False
    p2 = pipeline_factory(tiny_store)
    r2 = p2.run("带宽采购单价")
    row2 = load_trace(r2.trace_id, icfg)
    reflect2 = [e["payload"] for e in row2["events"] if e["step"] == "reflect"]
    assert reflect2[0]["sufficiency_basis"] == "candidate_count"
    assert not r2.degraded


def test_graph_exception_becomes_structured_error(icfg, tiny_store, pipeline_factory, monkeypatch):
    """04 §13：未捕获异常要变成"结构化错误 + trace_id"，而不是把 traceback 甩给用户。"""
    p = pipeline_factory(tiny_store)
    monkeypatch.setattr(p.graph, "invoke", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("模拟崩溃")))
    r = p.run("带宽采购单价")
    assert r.error
    assert r.confidence == "low"
    assert r.trace_id in r.answer, "answer 里必须带 trace_id，UI 才能查"
    assert load_trace(r.trace_id, icfg) is not None, "失败也要留 trace，否则事后无从排查"


def test_failed_result_is_not_cached(icfg, tiny_store, pipeline_factory, monkeypatch):
    """一次临时故障不能被缓存 TTL 固化成一整周的错误答案。"""
    p = pipeline_factory(tiny_store)
    monkeypatch.setattr(p.graph, "invoke", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("模拟崩溃")))
    p.run("带宽采购单价")
    monkeypatch.undo()
    assert p.run("带宽采购单价").cache_hit is False


# ── 缓存键（L3 / J6）──────────────────────────────────────────────────


def test_second_identical_query_hits_cache(icfg, tiny_store, pipeline_factory):
    p = pipeline_factory(tiny_store)
    first = p.run("带宽采购单价")
    second = p.run("带宽采购单价")
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.answer == first.answer


def test_cache_key_includes_history(icfg, tiny_store, pipeline_factory):
    """L3：漏了历史指纹，同一句'它多少钱'会跨会话串味。"""
    p = pipeline_factory(tiny_store)
    p.run("它多少钱")
    hist = [{"role": "user", "content": "华东区单价"}, {"role": "assistant", "content": "3.2 万元"}]
    after = p.run("它多少钱", history=hist)
    assert after.cache_hit is False


def test_cache_key_includes_retrieval_signature(icfg, tiny_store, pipeline_factory):
    """切了消融开关后缓存必须失效，否则切换开关后 UI 会立刻给出自相矛盾的结果。"""
    p = pipeline_factory(tiny_store)
    p.run("带宽采购单价")
    icfg.retrieval.use_rerank = False
    p2 = pipeline_factory(tiny_store)
    assert p2.run("带宽采购单价").cache_hit is False


def test_cache_key_includes_generation_signature(icfg, tiny_store, pipeline_factory):
    """J6：换终答模型后，缓存仍返回旧风格答案是最难发现的一类故障。"""
    p = pipeline_factory(tiny_store)
    p.run("带宽采购单价")
    icfg.dashscope.llm_main = "qwen-max"
    p2 = pipeline_factory(tiny_store)
    assert p2.run("带宽采购单价").cache_hit is False


def test_cache_key_includes_index_version(icfg, tiny_store, pipeline_factory):
    p = pipeline_factory(tiny_store)
    p.run("带宽采购单价")
    icfg.dashscope.embedding_dim = 512
    p2 = pipeline_factory(tiny_store)
    assert p2.run("带宽采购单价").cache_hit is False


def test_result_cache_make_key_is_stable_and_distinct(icfg):
    c = ResultCache(icfg)
    assert c.make_key("带宽采购单价") == c.make_key("带宽采购单价")
    assert c.make_key("带宽采购单价") != c.make_key("研发投入")
    assert len(c.make_key("x")) == 20


# ── 端到端契约 ────────────────────────────────────────────────────────


def test_run_result_roundtrips_through_cache(icfg, tiny_store, pipeline_factory):
    """缓存里没有 context 实体，只有 chunk_id——回放必须能复原。"""
    p = pipeline_factory(tiny_store)
    first = p.run("带宽采购单价")
    blob = first.to_cache()
    back = type(first).from_cache(blob, tiny_store)
    assert back.answer == first.answer
    assert back.confidence == first.confidence
    assert back.cache_hit is True
    assert [c.chunk.chunk_id for c in back.context] == [c.chunk.chunk_id for c in first.context]
    # 分数与命中路由也必须跟着走：不落盘的话重建出来全是 None/空，
    # K1 的达标计数与 UI 的路由标记会拿到"检索到了但分是 0"这种假象
    assert [c.score for c in back.context] == [c.score for c in first.context]
    assert [c.hit_routes for c in back.context] == [c.hit_routes for c in first.context]
    assert blob["_cached_at"], "cache_hit 事件的 cached_at 取这个字段，不写就永远是空串"


def test_from_cache_reads_legacy_context_ids(icfg, tiny_store, pipeline_factory):
    """旧格式的缓存（只有 id 列表）仍要能回放，分数未知时是 None 而不是 0.0。"""
    p = pipeline_factory(tiny_store)
    first = p.run("带宽采购单价")
    legacy = first.to_cache()
    legacy.pop("context", None)
    legacy["context_ids"] = [c.chunk.chunk_id for c in first.context]
    back = type(first).from_cache(legacy, tiny_store)
    assert [c.chunk.chunk_id for c in back.context] == [c.chunk.chunk_id for c in first.context]
    assert all(c.score is None for c in back.context)


def test_chitchat_never_retrieves(icfg, tiny_store, pipeline_factory):
    llm = ScriptedLLM(intent="chitchat")
    p = pipeline_factory(tiny_store, llm_light=llm)
    r = p.run("你好呀")
    assert r.intent == "chitchat"
    assert r.retrieve_iterations == 0
    assert r.citations == []
    assert r.confidence == "high"


def test_table_numeric_keeps_only_table_blocks(icfg, store_with_corpus, pipeline_factory):
    icfg.retrieval.filter_tables_only = True
    llm = ScriptedLLM(intent="table_numeric")
    p = pipeline_factory(store_with_corpus, llm_light=llm)
    r = p.run("各区域单价分别是多少")
    row = load_trace(r.trace_id, icfg)
    retrieve = next(e["payload"] for e in row["events"] if e["step"] == "retrieve")
    assert retrieve["filter_pushdown"] is True
    # 断言召回层而非 context：context 里放的是回填后的父块，父块本身不是 table
    fused = retrieve["fused"]
    assert fused, "语料里有表格，table_numeric 不该零召回"
    assert all(store_with_corpus.get_chunk(cid).block_type == "table" for cid in fused), (
        "BM25 路没有前置过滤，只能后置筛——漏过非表格块说明过滤没生效"
    )


@pytest.mark.parametrize("intent", ["simple", "multi_hop", "table_numeric"])
def test_all_retrieval_intents_produce_an_answer(icfg, store_with_corpus, pipeline_factory, intent):
    llm = ScriptedLLM(intent=intent)
    p = pipeline_factory(store_with_corpus, llm_light=llm)
    r = p.run("华东区带宽采购单价")
    assert r.answer
    assert r.intent == intent
    assert r.retrieve_iterations >= 1


# ── M24：trace 按步骤归因 token ───────────────────────────────────


def test_trace_attributes_tokens_to_the_step_that_spent_them(
    icfg, store_with_corpus, pipeline_factory
):
    """端到端跑完后，trace 要能回答「这条查询花了多少、花在哪一步」。

    回归点（M24）：`TraceEvent.tokens_in/out` 曾经是恒 0 的死字段——`event()`
    接受这两个参数，但全库没有任何调用方传值。事故后想从 trace 复盘
    "钱花在哪条查询上"时，拿到的就是一列 0。
    """
    p = pipeline_factory(store_with_corpus)
    r = p.run("华东区带宽采购的单价与合同期限分别是多少")
    row = load_trace(r.trace_id, icfg)
    assert row is not None

    per_step = {e["step"]: e["tokens_in"] + e["tokens_out"] for e in row["events"]}
    assert any(v > 0 for v in per_step.values()), "跑完一次查询，不该所有步骤都是 0"
    assert per_step.get("route", 0) > 0, "路由调了 LLM，这一步必须归因到用量"
    # 汇总 = 各步骤之和：既不能重复计，也不能漏计
    assert sum(per_step.values()) == row["tokens_in"] + row["tokens_out"]
