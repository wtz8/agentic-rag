"""trace：自动计时、崩溃不丢、汇总字段齐全（07 §3）。"""

from __future__ import annotations

import json
import time

from agentic_rag.observability.trace import Tracer, load_trace


def test_span_auto_timing(tmp_path, cfg):
    cfg.observability.trace_dir = str(tmp_path / "traces")
    t = Tracer("q1", cfg=cfg)
    with t.span("retrieve") as payload:
        time.sleep(0.01)
        payload["vector_hits"] = 50
    assert len(t.events) == 1
    assert t.events[0].step == "retrieve"
    assert t.events[0].duration_ms >= 10
    assert t.events[0].payload["vector_hits"] == 50


def test_span_records_even_on_exception(tmp_path, cfg):
    cfg.observability.trace_dir = str(tmp_path / "traces")
    t = Tracer("q2", cfg=cfg)
    try:
        with t.span("generate") as p:
            p["partial"] = True
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert len(t.events) == 1 and t.events[0].payload["partial"] is True


def test_finalize_and_write(tmp_path, cfg):
    cfg.observability.trace_dir = str(tmp_path / "traces")
    t = Tracer("q3", cfg=cfg)
    with t.span("route"):
        pass
    t.events[-1].tokens_in = 100
    t.events[-1].tokens_out = 20
    meta = t.finalize("答案")
    assert meta["query_id"] == "q3"
    assert meta["tokens_in"] == 100 and meta["tokens_out"] == 20
    assert meta["answer_len"] == 2 and meta["steps"] == ["route"]

    path = t.write()
    assert path is not None and path.exists()
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["events"][0]["step"] == "route"


def test_write_appends_and_keeps_history(tmp_path, cfg):
    cfg.observability.trace_dir = str(tmp_path / "traces")
    t1 = Tracer("a", cfg=cfg)
    t1.event("route", {"i": 1})
    t1.finalize("x")
    t1.write()
    t2 = Tracer("b", cfg=cfg)
    t2.event("route", {"i": 2})
    t2.finalize("y")
    t2.write()
    lines = (tmp_path / "traces").glob("*.jsonl")
    rows = []
    for p in lines:
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert [r["query_id"] for r in rows] == ["a", "b"]


def test_load_trace_by_query_id(tmp_path, cfg):
    cfg.observability.trace_dir = str(tmp_path / "traces")
    t = Tracer("findme", cfg=cfg)
    t.event("route", {"intent": "simple"})
    t.finalize("ok")
    t.write()
    assert load_trace("findme", cfg)["query_id"] == "findme"
    assert load_trace("nope", cfg) is None


def test_disabled_tracer_writes_nothing(tmp_path, cfg):
    cfg.observability.trace_dir = str(tmp_path / "traces")
    t = Tracer("off", cfg=cfg, enabled=False)
    t.event("route", {})
    assert t.events == []
    assert t.write() is None


def test_keep_top_k_default_matches_fusion_top_k(cfg):
    t = Tracer("k", cfg=cfg)
    assert t.keep_top_k == cfg.retrieval.fusion_top_k


def test_on_event_callback_receives_payload(cfg):
    seen = []
    t = Tracer("cb", cfg=cfg, enabled=False, on_event=lambda s, p: seen.append((s, p)))
    t.event("reflect", {"sufficient": False})
    assert seen == [("reflect", {"sufficient": False})]
    t.event("reflect", {"sufficient": True}, emit=False)
    assert len(seen) == 1


# ── M24：按步骤归因用量 ────────────────────────────────────────────


class _UsageProbe:
    """只提供 `usage_total` 的最小替身——`record_usage` 对 provider 的唯一契约。"""

    def __init__(self) -> None:
        self.usage_total: dict[str, int] = {}


def test_usage_is_attributed_to_the_step_that_spent_it(cfg):
    """M24 回归点：trace 的 tokens_in/out 曾经恒为 0。

    `event()` 接受这两个参数，但全库没有任何调用方传值；provider 侧累计的
    `usage_total` 又只在评测报告层被消费。后果是事故后想从 trace 回答
    "钱花在哪条查询上"时，拿到的是一列 0。
    """
    from agentic_rag.providers.base import record_usage

    p = _UsageProbe()
    t = Tracer("m24", cfg=cfg)
    t.watch(p)

    with t.span("route") as payload:
        record_usage(p, {"tokens_in": 100, "tokens_out": 10})
        payload["intent"] = "simple"
    with t.span("retrieve"):
        record_usage(p, {"tokens_in": 900, "tokens_out": 0})

    assert (t.events[0].tokens_in, t.events[0].tokens_out) == (100, 10)
    assert (t.events[1].tokens_in, t.events[1].tokens_out) == (900, 0), (
        "每一步只算自己花掉的那部分，不能把累计值重复计进去"
    )


def test_explicit_tokens_override_auto_attribution(cfg):
    """显式传值优先于自动归因——调用方比 Tracer 更清楚该记在谁头上。"""
    from agentic_rag.providers.base import record_usage

    p = _UsageProbe()
    t = Tracer("m24b", cfg=cfg)
    t.watch(p)
    record_usage(p, {"tokens_in": 500, "tokens_out": 50})

    ev = t.event("generate", {}, tokens_in=7, tokens_out=3)
    assert (ev.tokens_in, ev.tokens_out) == (7, 3)


def test_unwatched_tracer_yields_zero_without_error(cfg):
    """没登记 provider 时归因退化为 0——缺依赖不能连埋点一起崩掉。"""
    t = Tracer("m24c", cfg=cfg)
    ev = t.event("route", {})
    assert (ev.tokens_in, ev.tokens_out) == (0, 0)


def test_finalize_sums_per_step_usage(cfg):
    """汇总口径：各步骤增量之和 = 本次查询总用量。"""
    from agentic_rag.providers.base import record_usage

    p = _UsageProbe()
    t = Tracer("m24d", cfg=cfg)
    t.watch(p)
    with t.span("route"):
        record_usage(p, {"tokens_in": 100, "tokens_out": 10})
    with t.span("generate"):
        record_usage(p, {"tokens_in": 200, "tokens_out": 30})

    meta = t.finalize("答案")
    assert (meta["tokens_in"], meta["tokens_out"]) == (300, 40)
