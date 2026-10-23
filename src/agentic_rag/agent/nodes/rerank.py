"""精排节点（04 §6.5）：Top-50 → Top-5 子块 → 父子块回填 + 去重。

- **请求 `top_n = fusion_top_k`（50），本地只取前 `rerank_top_k`（5）条子块**（J9）：
  多排序不花钱（08 §3.2：候选 10→50 延迟仅 +21%），但去重后的"顺位补齐"需要池子
- **同一父块只保留一次**（保留其中 rerank 分数最高的子块的得分），
  去重后不足 `max_context_chunks` 时按 rerank 排名顺位取下一个**父块不同**的子块补齐
- `parent_id is None` 时用它自身（structural 策略），这让 B2 消融的两组走同一段回填代码
"""

from __future__ import annotations

from ...schema import ScoredChunk
from ..context import ctx_of
from ...utils import truncate_by_tokens


def rerank_node(state, config) -> dict:
    ctx = ctx_of(config)
    cfg = ctx.cfg
    r = cfg.retrieval
    query = state.get("query", "")
    candidates = list(state.get("candidates") or [])
    if not candidates:
        return {"context": []}

    pool = candidates[: r.fusion_top_k]
    chunks = {c.chunk_id: c for c in ctx.deps.store.get_chunks([p.chunk_id for p in pool])}

    with ctx.tracer.span("rerank") as payload:
        ordered: list[tuple[str, float]] = []  # (chunk_id, score)
        rerank_failed = False
        if r.use_rerank:
            texts = [
                truncate_by_tokens(chunks[p.chunk_id].text if p.chunk_id in chunks else "", r.rerank_max_tokens)
                for p in pool
            ]
            try:
                results = ctx.deps.reranker.rerank(query, texts, top_n=len(pool))
                for idx, score in results:
                    if 0 <= idx < len(pool):
                        ordered.append((pool[idx].chunk_id, float(score)))
            except Exception as e:  # noqa: BLE001 - rerank 失败不阻断，用 RRF 排序
                rerank_failed = True
                payload["rerank_error"] = f"{type(e).__name__}: {e}"
        if not ordered:
            ordered = [(p.chunk_id, float(p.fused_score)) for p in pool]

        before_rank = {p.chunk_id: i + 1 for i, p in enumerate(pool)}
        for c in candidates:
            c.rerank_score = None
        score_map = {cid: s for cid, s in ordered}
        for c in candidates:
            if c.chunk_id in score_map:
                c.rerank_score = score_map[c.chunk_id]

        # ── 父子块回填 + 去重（H18）─────────────────────────
        context: list[ScoredChunk] = []
        seen_parents: set[str] = set()
        dedup_collapsed = 0
        # 回填与顺位补齐的池子深度 = `rerank_top_k`（50 → 5）。
        # 早先这里直接遍历全部 ordered（靠 max_context_chunks 截断），
        # `rerank_top_k` 因此成了死配置——单独调它没有任何效果，
        # 而它的本意正是"去重后从多深的池子里顺位补齐"。
        pool_depth = min(len(ordered), r.rerank_top_k)
        for cid, score in ordered[:pool_depth]:
            if len(context) >= cfg.generation.max_context_chunks:
                break
            child = chunks.get(cid)
            if child is None:
                continue
            # B3 消融开关：关掉回填时上下文直接用子块原文（检索单元粒度），
            # 父块去重仍然生效——否则变体之间连"去重"这个变量都不一样，B3 就不是单变量实验了
            if r.parent_backfill:
                parent = ctx.deps.store.parent_of(child.chunk_id) or child
            else:
                parent = child
            if parent.chunk_id in seen_parents:
                dedup_collapsed += 1
                continue
            seen_parents.add(parent.chunk_id)
            context.append(
                ScoredChunk(
                    chunk=parent,
                    score=float(score),
                    rank=len(context) + 1,
                    hit_routes=list(_routes_of(cid, candidates)),
                )
            )

        changes = []
        for i, (cid, score) in enumerate(ordered[: cfg.observability.trace_keep_top_k], start=1):
            before = before_rank.get(cid)
            if before and before != i:
                changes.append({"from": before, "to": i, "chunk_id": cid})

        payload.update(
            {
                "requested": len(pool),
                "returned": len(ordered),
                "pool_depth": pool_depth,  # 真正参与回填与顺位补齐的条数（= rerank_top_k 生效后的值）
                "top": [
                    {"chunk_id": cid, "score": round(s, 6)}
                    for cid, s in ordered[: cfg.observability.trace_keep_top_k]
                ],
                "dedup_collapsed": dedup_collapsed,
                "context_blocks": len(context),
                "changes": changes,
                "rerank_failed": rerank_failed,
                "use_rerank": r.use_rerank,
            }
        )

    return {"context": context}


def _routes_of(cid: str, candidates) -> list[str]:
    for c in candidates:
        if c.chunk_id == cid:
            return c.hit_routes
    return []


__all__ = ["rerank_node"]
