"""三个条件边函数（04 §6.4 / §6.10 / §6.11）+ 共用的重生成判据。

三条统一约束（违反任何一条都属于破坏性变更）：
1. 只读 state，绝不修改 state（自增一律在节点里）
2. 不调 LLM、不做 IO、不推 SSE 事件——判定所需的中间产物由上游节点写入 state（H13）
3. 必须给出完整的返回字面量集合，且与 `add_conditional_edges` 的映射键完全对应
"""

from __future__ import annotations

from typing import Literal

from ..config import Config
from ..schema import Candidate
from .state import AgentState


def _reaches_floor(c: Candidate, r) -> bool:
    """单路失败时 `vector_score` 为 None，此时不能算达标，也不能当成 0.0 参与比较（L19）。"""
    return c.vector_score is not None and c.vector_score >= r.sufficiency_vector_floor


def dispatch_by_intent(
    state: AgentState, config=None
) -> Literal["chitchat", "simple", "multi_hop", "table_numeric"]:
    """回落选 `simple` 而非 `chitchat`：

    路由失败的代价不对称——误判成 chitchat 会把一个本该检索的问题直接挡在门外，
    误判成 simple 只是多花一次检索。
    """
    ctx = _ctx(config)
    intents = ctx.cfg.agent.intents if ctx else ("simple", "multi_hop", "table_numeric", "chitchat")
    intent = state.get("intent")
    if not intent or intent not in intents:
        return "simple"
    return intent  # type: ignore[return-value]


def is_sufficient(state: AgentState, config=None) -> Literal["sufficient", "retry", "degrade"]:
    """循环边的终止条件（L5 / K1）。

    判定顺序（短路）：
      1. sufficient is True                                  → sufficient
      2. reflect_strong == 0                                 → degrade（一条达标都没有才拒答）
      3. 不允许反思（simple 且未开 allow_reflect_for_simple）→ sufficient
      4. retrieve_iterations >= max_retrieve_iterations      → sufficient
      5. 否则                                                → retry
    """
    ctx = _ctx(config)
    cfg = ctx.cfg if ctx else None
    if state.get("sufficient"):
        return "sufficient"
    if state.get("reflect_strong", 0) == 0:
        return "degrade"
    if cfg is not None:
        intent = state.get("intent")
        if intent == "simple" and not cfg.agent.allow_reflect_for_simple:
            return "sufficient"
        if state.get("retrieve_iterations", 0) >= cfg.agent.max_retrieve_iterations:
            return "sufficient"
    return "retry"


def need_regenerate(state: AgentState, config=None) -> Literal["accept", "regenerate", "low_confidence"]:
    """判定顺序见 04 §6.11。默认路径是 `low_confidence`（重生成默认关闭）。"""
    ctx = _ctx(config)
    cfg = ctx.cfg if ctx else None
    if state.get("faithfulness", 1.0) >= (cfg.generation.faithfulness_threshold if cfg else 0.8):
        return "accept"
    if should_regenerate(state, cfg):
        return "regenerate"
    return "low_confidence"


def should_regenerate(state: AgentState, cfg: Config | None) -> bool:
    """**判据只实现一份**（L10）：条件函数与 `regenerate` 事件共用它，不可能不一致。"""
    if cfg is None:
        return False
    g = cfg.generation
    return (
        state.get("faithfulness", 1.0) < g.faithfulness_threshold
        and g.regenerate_on_unsupported
        and state.get("generate_iterations", 0) < cfg.agent.max_generate_iterations
    )


def _ctx(config):
    if isinstance(config, dict):
        return config.get("configurable", {}).get("ctx")
    return getattr(config, "ctx", None)


__all__ = ["dispatch_by_intent", "is_sufficient", "need_regenerate", "should_regenerate", "_reaches_floor"]
