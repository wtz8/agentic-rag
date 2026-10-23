"""意图路由节点（04 §6.1）：决定走哪条路径与检索参数。

- 输出 intent 必须落在 `AgentConfig.intents` 内，否则回落 `simple`
- **路由置信度写 `route_confidence`（float），绝不写进 `confidence`**——后者是答案级的（H17）
"""

from __future__ import annotations

from ...prompts.loader import get_prompt
from ...providers.base import ProviderError, as_float, first_json_object
from ..context import ctx_of


def route_node(state, config) -> dict:
    ctx = ctx_of(config)
    cfg = ctx.cfg
    llm = ctx.deps.llm_light
    query = state.get("query", "")
    history = (state.get("history") or [])[-cfg.agent.history_max_turns * 2 :]

    intent, confidence, reason = "simple", 0.0, "route_failed_fallback"
    with ctx.tracer.span("route") as payload:
        try:
            p = get_prompt("route")
            raw = llm.chat(
                p.messages(
                    query=query,
                    history="\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in history) or "（无）",
                    intents=" | ".join(cfg.agent.intents),
                ),
                temperature=p.temperature,
                json_mode=True,
            )
            data = first_json_object(raw)
            got = str(data.get("intent", "")).strip()
            if got in cfg.agent.intents:
                intent = got
                confidence = as_float(data.get("confidence"), 0.0)
                reason = str(data.get("reason", ""))[:200]
            else:
                reason = f"route_invalid_intent:{got or 'empty'}"
        except ProviderError as e:
            reason = f"route_failed_fallback:{type(e).__name__}"
        except Exception as e:  # noqa: BLE001
            reason = f"route_failed_fallback:{type(e).__name__}"

        payload.update(
            {
                "intent": intent,
                "confidence": confidence,
                "reason": reason,
                "skip_rewrite": intent == "simple" and cfg.agent.skip_rewrite_for_simple,
            }
        )
    return {"intent": intent, "route_reason": reason, "route_confidence": confidence}


__all__ = ["route_node"]
