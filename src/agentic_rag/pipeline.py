"""在线链路唯一入口（04 §9.7）：server 与 eval 共用同一条执行路径。

这不是为了整洁：如果评测跑的是一条自己组装的链路、UI 跑的是另一条，
那么"评测数字"和"演示效果"之间就没有任何保证它们描述的是同一个系统。
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from .agent.context import GraphDeps, NodeContext
from .agent.graph import build_graph, initial_state
from .cache.result_cache import ResultCache
from .config import Config
from .index.store import IndexStore
from .observability.trace import Tracer
from .schema import Citation, ScoredChunk

EVENT_TYPES = (
    "cache_hit",
    "route",
    "rewrite",
    "retrieve",
    "reflect",
    "rerank",
    "token",
    "citation",
    "verify",
    "regenerate",
    "degrade",
    "done",
    "error",
)


@dataclass
class Event:
    type: str
    data: dict

    def to_sse(self) -> str:
        return f"event: {self.type}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


@dataclass
class RunResult:
    query_id: str
    query: str
    intent: str
    answer: str
    confidence: str  # high | medium | low（取值见 04 §6.7 的 L9 定义）
    citations: list[Citation]
    context: list[ScoredChunk]
    claims: list[str]
    unsupported: list[int]
    faithfulness: float
    retrieve_iterations: int
    generate_iterations: int
    weak_evidence: bool = False
    degraded: bool = False
    cache_hit: bool = False
    error: str = ""  # 非空 = 本次查询失败（04 §13：结构化错误 + trace_id）
    trace_id: str = ""
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_cache(self) -> dict:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "intent": self.intent,
            "answer": self.answer,
            "confidence": self.confidence,
            "citations": [c.__dict__ for c in self.citations],
            # `hit_routes` 与 rerank 分数也必须落盘：命中缓存时 context 是从 store 反查重建的，
            # 不存就拿不到，下游（UI 的路由标记、K1 的达标计数、评测的 top_rerank）
            # 会拿到 score=0.0 与空路由——看起来"检索到了但全是 0 分"。
            "context": [
                {
                    "id": c.chunk.chunk_id,
                    "score": c.score,
                    "hit_routes": list(c.hit_routes),
                }
                for c in self.context
            ],
            # 缓存命中事件的 `cached_at` 取这个字段；不写就永远是空串
            "_cached_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "claims": self.claims,
            "unsupported": self.unsupported,
            "faithfulness": self.faithfulness,
            "retrieve_iterations": self.retrieve_iterations,
            "generate_iterations": self.generate_iterations,
            "weak_evidence": self.weak_evidence,
            "degraded": self.degraded,
            "error": self.error,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_cache(cls, d: dict, store: IndexStore | None = None) -> "RunResult":
        # 兼容两种落盘格式：`["id", ...]`（旧）与 `[{"id","score","hit_routes"}, ...]`（新）。
        # 旧缓存没有分数与路由信息，重建时只能留空——显式置 None 而不是 0.0：
        # 0.0 会被当成"检索到了但分数是 0"，None 才是"不知道"。
        raw_context = d.get("context", d.get("context_ids", []))
        ids = [x.get("id") if isinstance(x, dict) else x for x in raw_context]
        chunks = {}
        if store is not None:
            chunks = {c.chunk_id: c for c in store.get_chunks([i for i in ids if i])}
        context = []
        for item in raw_context:
            if isinstance(item, dict):
                cid = item.get("id")
                c = chunks.get(cid)
                if c is None:
                    continue
                sc = ScoredChunk(
                    chunk=c,
                    score=item.get("score"),
                    rank=len(context) + 1,
                )
                sc.hit_routes = list(item.get("hit_routes") or [])
                context.append(sc)
            else:
                c = chunks.get(item)
                if c is not None:
                    context.append(ScoredChunk(chunk=c, score=None, rank=len(context) + 1))
        citations = [Citation(**c) for c in d.get("citations", [])]
        return cls(
            query_id=d.get("query_id", ""),
            query=d.get("query", ""),
            intent=d.get("intent", ""),
            answer=d.get("answer", ""),
            confidence=d.get("confidence", "medium"),
            citations=citations,
            context=context,
            claims=d.get("claims", []),
            unsupported=d.get("unsupported", []),
            faithfulness=float(d.get("faithfulness", 1.0)),
            retrieve_iterations=int(d.get("retrieve_iterations", 0)),
            generate_iterations=int(d.get("generate_iterations", 0)),
            weak_evidence=bool(d.get("weak_evidence", False)),
            degraded=bool(d.get("degraded", False)),
            error=str(d.get("error", "")),
            cache_hit=True,
        )


_SENTINEL = object()


class Pipeline:
    def __init__(self, cfg: Config, deps: GraphDeps, store: IndexStore):
        self.cfg = cfg
        self.deps = deps
        self.store = store
        self.graph = build_graph(cfg, deps)
        self.cache = ResultCache(cfg)

    # ── 同步执行 ──────────────────────────────────────────

    def run(
        self,
        query: str,
        history: list[dict] | None = None,
        on_event: "callable | None" = None,
    ) -> RunResult:
        """同步执行。`on_event` 是只读事件旁路（评测用：召回明细不在 RunResult 里，
        而在 retrieve 事件中）——不改执行路径，只把已经在流动的事件抄送一份。"""
        cached = self.cache.get(query, history)
        if cached is not None:
            result = RunResult.from_cache(cached, self.store)
            tracer = Tracer(result.query_id or "cache", cfg=self.cfg, enabled=self.cfg.observability.enabled)
            tracer.meta["cache_hit"] = True
            tracer.meta["retrieval_signature"] = self.cfg.retrieval_signature
            tracer.finalize(result.answer)
            tracer.write()  # 命中也要写 trace，否则缓存命中率这个指标无从统计
            return result

        events: list[Event] = []
        emit = (lambda t, d: events.append(Event(t, d))) if on_event is None else (
            lambda t, d: (events.append(Event(t, d)), on_event(t, d))
        )
        result = self._execute(query, history, emit=emit, stream=False)
        if not result.error:
            self.cache.put(query, history, result.to_cache())
        return result

    # ── 流式执行 ──────────────────────────────────────────

    def run_stream(self, query: str, history: list[dict] | None = None) -> Iterator[Event]:
        """真实流式：图在后台线程跑，token 到达即 yield（04 §10.4 的队列桥接）。"""
        cached = self.cache.get(query, history)
        if cached is not None:
            result = RunResult.from_cache(cached, self.store)
            yield Event(
                "cache_hit",
                {"cached_at": cached.get("_cached_at", ""), "source_query_id": result.query_id},
            )
            yield Event("token", {"text": result.answer})  # 完整答案一条推完，不伪装打字机
            for c in result.citations:
                yield Event(
                    "citation",
                    {
                        "index": c.index,
                        "source": c.source,
                        "title_path": c.title_path,
                        "page": c.page,
                        "quote": c.quote,
                    },
                )
            yield Event(
                "done",
                {
                    "confidence": result.confidence,
                    "trace_id": result.query_id,
                    "weak_evidence": result.weak_evidence,
                    "cache_hit": True,
                },
            )
            return

        q: queue.Queue = queue.Queue()

        def worker() -> None:
            try:
                result = self._execute(
                    query, history, emit=lambda t, d: q.put(Event(t, d)), stream=True
                )
                if result.error:
                    q.put(Event("error", {"message": result.error, "trace_id": result.trace_id}))
                else:
                    self.cache.put(query, history, result.to_cache())
                q.put(Event("done", self._done_payload(result)))
            except Exception as e:  # noqa: BLE001 - 未捕获异常也要给 UI 一个出口
                q.put(Event("error", {"message": f"{type(e).__name__}: {e}"}))
            finally:
                q.put(_SENTINEL)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while True:
            ev = q.get()
            if ev is _SENTINEL:
                break
            yield ev
        t.join(timeout=self.cfg.dashscope.timeout_s)

    # ── 内部 ──────────────────────────────────────────────

    def _execute(self, query: str, history, emit, stream: bool) -> RunResult:
        query_id = uuid.uuid4().hex[:12]
        tracer = Tracer(query_id, cfg=self.cfg, enabled=self.cfg.observability.enabled)
        # M24：登记 provider，让每个步骤的埋点自动带上「这一步」的 token 增量。
        # 少了这一行，trace 里的 tokens_in/out 恒为 0，无法按查询归因成本。
        tracer.watch(
            self.deps.llm_light,
            self.deps.llm_main,
            self.deps.reranker,
            self.deps.embedder,
        )
        ctx = NodeContext(
            cfg=self.cfg, deps=self.deps, tracer=tracer, query_id=query_id, emit=emit, stream=stream
        ).bind()
        try:
            final = self.graph.invoke(
                initial_state(query, history), config={"configurable": {"ctx": ctx}}
            )
            result = self._to_result(query_id, query, final, tracer)
        except Exception as e:  # noqa: BLE001 - 04 §13：未捕获异常也要给 UI 一个出口
            result = RunResult(
                query_id=query_id,
                query=query,
                intent="",
                answer=f"本次查询失败（{type(e).__name__}）：{e}。trace_id={query_id}",
                confidence="low",
                citations=[],
                context=[],
                claims=[],
                unsupported=[],
                faithfulness=0.0,
                retrieve_iterations=0,
                generate_iterations=0,
                error=f"{type(e).__name__}: {e}",
                trace_id=query_id,
            )
        # done 必须在 trace 落盘之后推送，否则用户立刻点进 /api/trace/{id} 会查不到
        tracer.meta["error"] = result.error
        tracer.finalize(result.answer)
        tracer.write()
        # 失败结果不进缓存：否则一次临时故障会被缓存 TTL 固化成一整周的错误答案
        return result

    def _to_result(self, query_id: str, query: str, final: dict, tracer: Tracer) -> RunResult:
        latency = {e.step: round(e.duration_ms, 2) for e in tracer.events}
        latency["total"] = round(sum(e.duration_ms for e in tracer.events), 2)
        return RunResult(
            query_id=query_id,
            query=query,
            intent=final.get("intent", ""),
            answer=final.get("answer") or final.get("draft") or "",
            confidence=final.get("confidence", "medium"),
            citations=list(final.get("citations") or []),
            context=list(final.get("context") or []),
            claims=list(final.get("claims") or []),
            unsupported=list(final.get("unsupported") or []),
            faithfulness=float(final.get("faithfulness", 1.0)),
            retrieve_iterations=int(final.get("retrieve_iterations", 0)),
            generate_iterations=int(final.get("generate_iterations", 0)),
            weak_evidence=bool(final.get("weak_evidence", False)),
            degraded=bool(final.get("degraded", False)),
            trace_id=query_id,
            latency_ms=latency,
        )

    def _done_payload(self, r: RunResult) -> dict:
        return {
            "confidence": r.confidence,
            "trace_id": r.trace_id,
            "weak_evidence": r.weak_evidence,
            "degraded": r.degraded,
            "intent": r.intent,
            "faithfulness": round(r.faithfulness, 4),
            "retrieve_iterations": r.retrieve_iterations,
            "generate_iterations": r.generate_iterations,
            "cache_hit": False,
        }


__all__ = ["Pipeline", "RunResult", "Event", "EVENT_TYPES"]
