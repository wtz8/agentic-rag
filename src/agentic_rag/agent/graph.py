"""LangGraph 状态机（04 §7）。

节点注册的是**函数本身，不用 partial**（H21）：
`inspect.signature(partial(f, cfg=...))` 会把已绑定参数从签名里移除，
LangGraph 认为节点不接受 config → `configurable` 注入静默失效 → trace 一直为空。

图里两条循环各有独立计数器：
- `retrieve → rewrite → retrieve`：`retrieve_iterations` / `max_retrieve_iterations`
- `verify → generate → verify`：`generate_iterations` / `max_generate_iterations`
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .conditions import dispatch_by_intent, is_sufficient, need_regenerate
from .context import GraphDeps, NodeContext, ctx_of  # noqa: F401 - NodeContext 在此统一导出
from .nodes import (
    degrade_answer_node,
    direct_answer_node,
    generate_node,
    rerank_node,
    retrieve_node,
    rewrite_node,
    route_node,
    verify_node,
)
from .state import AgentState


def build_graph(cfg, deps: GraphDeps):
    g = StateGraph(AgentState)

    # 主链路 6 个
    g.add_node("route", route_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("generate", generate_node)
    g.add_node("verify", verify_node)
    # 终止应答 2 个
    g.add_node("direct_answer", direct_answer_node)
    g.add_node("degrade_answer", degrade_answer_node)

    g.set_entry_point("route")

    # 自适应路由在代码里的落点：意图决定整条路径
    g.add_conditional_edges(
        "route",
        dispatch_by_intent,
        {
            "chitchat": "direct_answer",
            "simple": "rewrite",
            "multi_hop": "rewrite",
            "table_numeric": "rewrite",
        },
    )

    g.add_edge("rewrite", "retrieve")

    # 反思循环：成本控制的核心
    g.add_conditional_edges(
        "retrieve",
        is_sufficient,
        {
            "sufficient": "rerank",
            "retry": "rewrite",  # retrieve_iterations += 1 在 retrieve_node 内完成
            "degrade": "degrade_answer",
        },
    )

    g.add_edge("rerank", "generate")
    g.add_edge("generate", "verify")

    g.add_conditional_edges(
        "verify",
        need_regenerate,
        {
            "accept": END,
            "regenerate": "generate",  # generate_iterations += 1 在 generate_node 内完成
            "low_confidence": END,
        },
    )

    g.add_edge("direct_answer", END)
    g.add_edge("degrade_answer", END)

    return g.compile()


def initial_state(query: str, history: list[dict] | None = None) -> dict:
    return {
        "query": query,
        "history": list(history or []),
        "retrieve_iterations": 0,
        "generate_iterations": 0,
    }


__all__ = ["build_graph", "initial_state", "GraphDeps", "NodeContext"]
