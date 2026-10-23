"""答案生成节点（04 §6.6）：分级供给（qwen-plus）+ 逐句引用标注。

流式中断时返回已生成部分，不抛异常。**在本节点末尾 `generate_iterations += 1`**（H4）。
"""

from __future__ import annotations

import time

from ...generation.citation import parse_citations
from ...generation.generator import Generator
from ..context import ctx_of


def generate_node(state, config) -> dict:
    ctx = ctx_of(config)
    cfg = ctx.cfg
    query = state.get("query", "")
    context = list(state.get("context") or [])
    gen = Generator(cfg, ctx.deps.llm_main)

    draft = ""
    citations = []
    first_token_ms = None
    # 04 §10.2 的 SSE 协议里没有 `generate` 事件——生成这一步在 UI 上由第一个 token 表示，
    # 再推一个同名事件等于让前端收到协议外的类型。这里只落 trace（延迟 / 首字 / 迭代序号），
    # SSE 侧保持 token + citation 两类的干净语义。
    with ctx.tracer.span("generate", emit=False) as payload:
        t0 = time.perf_counter()
        if ctx.stream and ctx.emit is not None:
            for delta in gen.stream(query, context):
                if not delta:
                    continue
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - t0) * 1000.0
                draft += delta
                ctx.push("token", {"text": delta})
        else:
            draft, citations = gen.generate(query, context)
            first_token_ms = (time.perf_counter() - t0) * 1000.0

        if ctx.stream and ctx.emit is not None:
            draft, citations = parse_citations(draft, context)
        for c in citations:
            ctx.push(
                "citation",
                {
                    "index": c.index,
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "title_path": c.title_path,
                    "page": c.page,
                    "quote": c.quote,
                    "score": None if c.score is None else round(c.score, 6),
                },
            )
        payload.update(
            {
                "chars": len(draft),
                "citations": len(citations),
                "first_token_ms": None if first_token_ms is None else round(first_token_ms, 2),
                "iteration": int(state.get("generate_iterations", 0)) + 1,
                "stream": bool(ctx.stream and ctx.emit is not None),
            }
        )

    return {
        "draft": draft,
        "citations": citations,
        "generate_iterations": int(state.get("generate_iterations", 0)) + 1,
    }


__all__ = ["generate_node"]
