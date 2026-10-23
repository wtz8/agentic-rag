"""混合检索节点（04 §6.3）。

三件事都发生在这里，因为**判据与事件只允许有一个发送方**（H13 / L10）：
1. 调 `HybridRetriever` 召回，并把各路命中与耗时记进 trace
2. **在本节点内完成充分性预判**（`reflect_strong` / `sufficient` / `reflect_reason` /
   `sufficiency_basis`），并推 `reflect` 事件——条件函数只做路由
3. **在本节点末尾 `retrieve_iterations += 1`**（H4），语义为"已完成的检索轮数"
"""

from __future__ import annotations

from ...config import Config
from ...schema import Candidate
from ..conditions import _reaches_floor
from ..context import ctx_of
from ...retrieval.fusion import merge_candidates


def retrieve_node(state, config) -> dict:
    ctx = ctx_of(config)
    cfg: Config = ctx.cfg
    r = cfg.retrieval
    query = state.get("query", "")
    intent = state.get("intent", "simple")
    queries = state.get("sub_queries") or [query]
    # HyDE 的假设文档只走向量路，BM25 路用原 query（I3 / J7）
    bm25_query = query if state.get("transform_strategy") == "hyde" else None

    diag: dict = {}
    with ctx.tracer.span("retrieve") as payload:
        cur = ctx.deps.retriever.retrieve(queries, intent=intent, bm25_query=bm25_query, diag=diag)
        prev: list[Candidate] = list(state.get("candidates") or [])
        merged_from_prev = 0
        if prev:
            # 跨轮 RRF 合并，不覆盖：第二轮更差也不会丢掉首轮的好候选（L11）
            cur, merged_from_prev = merge_candidates(prev, cur, k=r.rrf_k, top_k=r.fusion_top_k)
        candidates = cur[: r.fusion_top_k]
        # "实际参与合并的轮数"：本轮 +（有上一轮候选则 +1）。
        # 不能拿 diag["durations"] 的长度当轮数——那是子查询条数（decompose 下是 3、5…），
        # 与"轮"是两个量纲，混用会让 A4（跨轮合并）的归因彻底错位。
        merged_rounds = 2 if prev else 1

        # ── 充分性预判（纯规则，不调 LLM）──────────────────
        # K1：判据必须有**绝对量纲**才成立。`sufficiency_vector_floor` 是对
        # text-embedding-v4 的余弦分布标的初值，换成别的分数分布就不再是同一个阈值：
        # - use_vector=False：根本没有余弦分
        # - Mock 向量：字符 3-gram 词袋的哈希投影，余弦被块长稀释（同一查询在短文本上
        #   0.41、在真实分块上只有 0.10），没有跨语料可比量纲
        # 两种情况都降级为候选数判据，并把口径写进 trace——否则 Mock 模式下
        # 达标数恒为 0，每次查询都拒答，链路连"生成"这一步都到不了。
        has_vector_scores = any(c.vector_score is not None for c in candidates)
        if r.use_vector and cfg.provider_mode == "dashscope" and has_vector_scores:
            sufficiency_basis = "vector_score"
            basis_reason = "calibrated"
            strong = [c for c in candidates if _reaches_floor(c, r)]
        else:
            # 三种降级原因都写进 trace：判据被换掉这件事本身必须可见
            sufficiency_basis = "candidate_count"
            basis_reason = (
                "vector_route_unavailable"  # 向量路没给出分数（索引无向量或调用失败）
                if r.use_vector and not has_vector_scores
                else "use_vector_off" if not r.use_vector else "mock_embedding"
            )
            strong = list(candidates)
        reflect_strong = len(strong)
        sufficient = reflect_strong >= r.min_candidates
        # L8：降级路径下 strong == candidates，corpus_missing 只在真的一条都没有时成立
        reflect_reason = "corpus_missing" if reflect_strong == 0 else "query_expression"

        payload.update(
            {
                "queries": queries,
                "intent": intent,
                "vector_hits": diag.get("vector_hits", 0),
                "bm25_hits": diag.get("bm25_hits", 0),
                "routes": diag.get("routes", []),
                "filter_pushdown": diag.get("filter_pushdown", False),
                "durations": diag.get("durations", []),
                "errors": diag.get("errors", {}),
                "sub_query_count": int(diag.get("sub_query_count", len(queries) or 1)),
                "fused": [c.chunk_id for c in candidates[: cfg.observability.trace_keep_top_k]],
                "scores": [round(c.fused_score, 6) for c in candidates[: cfg.observability.trace_keep_top_k]],
                "retrieve_rounds": merged_rounds,
                "merged_from_prev": merged_from_prev,
            }
        )

    # 融合是独立一步，单独计时与记录（03 §3 的 step 枚举含 fuse）
    ctx.tracer.event(
        "fuse",
        {
            "strategy": r.fusion_strategy,
            "rrf_k": r.rrf_k,
            "top": [c.chunk_id for c in candidates[: cfg.observability.trace_keep_top_k]],
            "scores": [round(c.fused_score, 6) for c in candidates[: cfg.observability.trace_keep_top_k]],
            "merged_from_prev": merged_from_prev,
        },
        emit=False,
    )

    # reflect 事件的唯一发送方（协议里列了它，条件函数在物理上推不了事件）
    ctx.tracer.event(
        "reflect",
        {
            "iteration": int(state.get("retrieve_iterations", 0)) + 1,
            "sufficient": sufficient,
            "reason": reflect_reason,
            "strong": reflect_strong,
            "min_candidates": r.min_candidates,
            "sufficiency_basis": sufficiency_basis,
            "sufficiency_basis_reason": basis_reason,
            "sufficiency_floor": r.sufficiency_vector_floor if sufficiency_basis == "vector_score" else None,
            "retrieve_rounds": merged_rounds,
            "merged_from_prev": merged_from_prev,
            "action": "accept" if sufficient else ("degrade" if reflect_strong == 0 else "retry"),
        },
    )
    return {
        "candidates": candidates,
        "reflect_strong": reflect_strong,
        "sufficient": sufficient,
        "reflect_reason": reflect_reason,
        "sufficiency_basis": sufficiency_basis,
        "sufficiency_basis_reason": basis_reason,
        "retrieve_iterations": int(state.get("retrieve_iterations", 0)) + 1,
    }


__all__ = ["retrieve_node"]
