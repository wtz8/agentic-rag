"""结构化 trace（03 §3）。

一次采集、两处消费：UI 的决策时间线 + 评测的数据源（延迟分布 / token / 迭代次数）。

两条硬约束：
- **埋点宽度 `trace_keep_top_k` 必须 ≥ 评测要上报的最大 K**（默认 50 = `fusion_top_k`）。
  trace 事后无法补采——写 trace 那几行代码就决定了报告日后能回答哪些问题。
- **追加写入**：进程崩溃时已采集的部分不丢。
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from ..config import Config, get_config, project_path
from ..schema import TraceEvent

TRACE_STEPS = ("route", "rewrite", "retrieve", "fuse", "rerank", "reflect", "generate", "verify")


@dataclass
class Tracer:
    query_id: str
    cfg: Config | None = None
    enabled: bool = True
    on_event: Callable[[str, dict], None] | None = None  # 推 SSE 事件（单发送方：产生它的节点）
    events: list[TraceEvent] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)
    # M24：按步骤归因用量用的登记簿。provider 侧的 `record_usage` 已经在累加
    # `usage_total`，这里只是把累计量拆到「这一步花了多少」的粒度上。
    _watched: list[Any] = field(default_factory=list, repr=False)
    _usage_baseline: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.cfg is None:
            self.cfg = get_config()

    # ── 用量归因（M24） ────────────────────────────────────

    def watch(self, *providers: Any) -> "Tracer":
        """登记要归因用量的 provider，之后每次 `event()` 自动算增量。

        为什么需要显式登记：一个步骤可能同时用到多个 provider（例如 retrieve
        走 embedder + 向量库，verify 走 llm_light），而 Tracer 本身不持有依赖，
        拿不到 provider 实例——由装配方（`Pipeline._execute`）登记一次即可。
        """
        for p in providers:
            if p is not None:
                self._watched.append(p)
        self._usage_baseline = self._usage_snapshot()
        return self

    def _usage_snapshot(self) -> dict[str, int]:
        snap: dict[str, int] = {}
        for p in self._watched:
            for k, v in (getattr(p, "usage_total", None) or {}).items():
                snap[k] = snap.get(k, 0) + int(v)
        return snap

    def _usage_delta(self) -> tuple[int, int]:
        """自上次事件以来的用量增量，归类成 (in, out)。"""
        now = self._usage_snapshot()
        tin = tout = 0
        for k, v in now.items():
            d = max(0, v - self._usage_baseline.get(k, 0))
            if k.endswith("tokens_in"):
                tin += d
            elif k.endswith("tokens_out"):
                tout += d
        self._usage_baseline = now
        return tin, tout

    @property
    def keep_top_k(self) -> int:
        assert self.cfg is not None
        return self.cfg.observability.trace_keep_top_k

    @property
    def path(self) -> Path:
        assert self.cfg is not None
        day = datetime.now().strftime("%Y-%m-%d")
        return project_path(self.cfg.observability.trace_dir) / f"{day}.jsonl"

    # ── 采集 ──────────────────────────────────────────────

    def event(
        self,
        step: str,
        payload: dict | None = None,
        *,
        duration_ms: float = 0.0,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        emit: bool = True,
    ) -> TraceEvent:
        payload = payload or {}
        # M24：默认值用 None 而不是 0，好区分「调用方没传」和「传了 0」。
        # 没传就按登记簿的累计差值自动归因——少了它，trace 里两列恒为 0，
        # 事后无法按查询归因成本。
        if tokens_in is None or tokens_out is None:
            auto_in, auto_out = self._usage_delta()
            tokens_in = auto_in if tokens_in is None else tokens_in
            tokens_out = auto_out if tokens_out is None else tokens_out
        ev = TraceEvent(
            step=step,
            ts=time.time(),
            duration_ms=duration_ms,
            payload=payload,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        if self.enabled:
            self.events.append(ev)
        if emit and self.on_event is not None:
            self.on_event(step, dict(payload))
        return ev

    @contextmanager
    def span(self, step: str, emit: bool = True) -> Iterator[dict]:
        """自动计时的埋点：节点往 payload 里填，退出时自动记录并推事件。"""
        payload: dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            yield payload
        finally:
            self.event(step, payload, duration_ms=(time.perf_counter() - t0) * 1000.0, emit=emit)

    # ── 落盘 ──────────────────────────────────────────────

    def finalize(self, answer: str = "") -> dict:
        """汇总本次查询（落盘前调用）。"""
        total = sum(e.duration_ms for e in self.events)
        self.meta.update(
            {
                "query_id": self.query_id,
                "finished_at": time.time(),
                "total_ms": round(total, 3),
                "tokens_in": sum(e.tokens_in for e in self.events),
                "tokens_out": sum(e.tokens_out for e in self.events),
                "answer_len": len(answer or ""),
                "steps": [e.step for e in self.events],
            }
        )
        return dict(self.meta)

    def write(self) -> Path | None:
        if not self.enabled:
            return None
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(self.meta)
        row["events"] = [e.to_dict() for e in self.events]
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path


def load_trace(query_id: str, cfg: Config | None = None) -> dict | None:
    """按 query_id 回查一次调用的 trace（`GET /api/trace/{id}`）。"""
    cfg = cfg or get_config()
    trace_dir = project_path(cfg.observability.trace_dir)
    if not trace_dir.exists():
        return None
    for path in sorted(trace_dir.glob("*.jsonl"), reverse=True):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("query_id") == query_id:
                    return row
    return None


__all__ = ["Tracer", "TraceEvent", "TRACE_STEPS", "load_trace"]
