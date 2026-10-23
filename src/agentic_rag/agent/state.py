"""Agent 状态（04 §5）。

`Tracer` 不在 state 里，走 `config={"configurable": {"ctx": ctx}}` 注入——
state 保持纯数据才能 checkpoint / 回放（01 把"原生 checkpoint"列为选 LangGraph 的理由，
而 Tracer 持有文件句柄，不可序列化，两者互斥）。

两个迭代计数器语义统一为"该动作已执行几次"，自增写在执行动作的节点里（H4）。
"""

from __future__ import annotations

from typing import TypedDict

from ..schema import Candidate, Citation, ScoredChunk


class AgentState(TypedDict, total=False):
    query: str
    history: list[dict]
    intent: str  # simple | multi_hop | table_numeric | chitchat
    route_reason: str
    route_confidence: float  # 路由置信度（H17），与答案级 confidence 严格分离
    sub_queries: list[str]
    transform_strategy: str  # none | rewrite | decompose | hyde
    candidates: list[Candidate]
    sufficient: bool  # 由 retrieve_node 判定写入，条件函数只读
    reflect_strong: int  # 达标候选数，由 retrieve_node 写入（K1 / L5）
    sufficiency_basis: str  # vector_score | candidate_count
    sufficiency_basis_reason: str  # calibrated | vector_route_unavailable | use_vector_off | mock_embedding
    reflect_reason: str  # query_expression | corpus_missing
    context: list[ScoredChunk]
    draft: str
    citations: list[Citation]
    retrieve_iterations: int
    generate_iterations: int
    confidence: str  # high | medium | low（答案级，唯一）
    answer: str
    claims: list[str]
    unsupported: list[int]
    faithfulness: float
    degraded: bool  # 走了 degrade_answer
    verify_failed: bool  # verify 调用失败，无判定结果
    weak_evidence: bool  # 达标证据 ∈ [1, min_candidates)


__all__ = ["AgentState"]
