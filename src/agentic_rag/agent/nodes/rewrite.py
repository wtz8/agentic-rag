"""查询改写节点（04 §6.2）：指代消解 / 同义扩展 / 多跳分解 / HyDE。

策略升级只看 `retrieve_iterations`（0 → 意图策略，1 → rewrite，≥2 → hyde），
与"上一轮用了什么"无关（L2）。

失败一律回落原 query，绝不让改写环节成为链路的单点故障。
"""

from __future__ import annotations

from ...retrieval.query_transform import transform_query
from ..context import ctx_of


def rewrite_node(state, config) -> dict:
    ctx = ctx_of(config)
    cfg = ctx.cfg
    llm = ctx.deps.llm_light
    query = state.get("query", "")
    intent = state.get("intent", "simple")
    # 已完成的检索轮数 = 本轮改写所处的迭代序号（0 = 首轮）
    iteration = int(state.get("retrieve_iterations", 0) or 0)

    with ctx.tracer.span("rewrite") as payload:
        result = transform_query(
            query=query,
            intent=intent,
            llm=llm,
            history=state.get("history"),
            iteration=iteration,
            cfg=cfg,
        )
        payload.update(
            {
                "original": query,
                "sub_queries": result.queries,
                "strategy": result.strategy,
                "iteration": iteration,
                "bm25_query": result.bm25_query,
            }
        )
    return {"sub_queries": result.queries, "transform_strategy": result.strategy}


__all__ = ["rewrite_node"]
