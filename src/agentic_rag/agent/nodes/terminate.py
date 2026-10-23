"""两个终止应答节点（04 §6.8 / §6.9）。

两者都不调生成模型：前者是"不该检索"，后者是"检索了但一条相关的都没有"。
"""

from __future__ import annotations

from ..context import ctx_of

_DIRECT_TEMPLATE = (
    "这是企业文档问答助手，只能基于已入库的资料回答问题。\n"
    "您的问题看起来不属于资料范围，我没有启动检索。\n"
    "如果要查资料，请告诉我具体的文档主题、条款、指标或时间范围。"
)


def direct_answer_node(state, config) -> dict:
    """chitchat / 越界问题的出口。**不检索、不调生成模型**。

    输出必须是预置模板而非 LLM 生成：既省一次调用，也避免模型在"不知道该说什么"时编造能力范围。
    """
    ctx = ctx_of(config)
    answer = _DIRECT_TEMPLATE
    # token 事件的发送方在本路径下是本节点（chitchat 不会走到 generate_node，两个发送方互斥）
    if ctx.emit is not None:
        ctx.push("token", {"text": answer})
    ctx.tracer.event(
        "generate",
        {"intent": state.get("intent"), "route_reason": state.get("route_reason"), "direct": True},
        emit=False,
    )
    return {"answer": answer, "confidence": "high", "citations": [], "claims": [], "unsupported": [], "faithfulness": 1.0}


def degrade_answer_node(state, config) -> dict:
    """**唯一入口条件：`reflect_strong == 0`**（L5）——一条达标候选都没有才拒答。

    注意不是"迭代耗尽"：预算耗尽但手里有证据时走的是生成而不是本节点。
    拒答是比"少证据地回答"更重的判断，它应当要求更强的证据。
    """
    ctx = ctx_of(config)
    cfg = ctx.cfg
    reason = state.get("reflect_reason", "corpus_missing")
    candidates = list(state.get("candidates") or [])

    if reason == "corpus_missing":
        answer = "未在资料中找到与该问题相关的信息，我不能凭推测回答。建议换个说法或确认资料是否已入库。"
    else:
        answer = "资料中与这个问题相关的内容不足，为避免给出没有依据的回答，我暂不作答。"

    near: list[str] = []
    if candidates:
        # 有候选但全部未达标：给出近似内容比直接说"没有"更有用，也比装作知道更诚实
        chunks = ctx.deps.store.get_chunks([c.chunk_id for c in candidates[: cfg.retrieval.rerank_top_k]])
        near = [
            f"- {c.source}｜{' > '.join(c.title_path) or '（无标题）'}：{c.raw_text[:100]}"
            for c in chunks[: cfg.retrieval.rerank_top_k]
        ]
        if near:
            answer = (
                "未在资料中找到能直接回答该问题的内容。以下是字面相近的片段，供你判断是否相关：\n"
                + "\n".join(near)
            )

    ctx.push(
        "degrade",
        {
            "reason": reason,
            "candidates": len(candidates),
            "retrieve_iterations": int(state.get("retrieve_iterations", 0)),
            "sufficiency_basis": state.get("sufficiency_basis", ""),
            "reflect_strong": int(state.get("reflect_strong", 0)),
        },
    )
    ctx.tracer.event(
        "verify",
        {
            "degraded": True,
            "reason": reason,
            "candidates": len(candidates),
            "retrieve_iterations": int(state.get("retrieve_iterations", 0)),
        },
        emit=False,
    )
    return {
        "answer": answer,
        "confidence": "low",
        "degraded": True,
        "citations": [],
        "claims": [],
        "unsupported": [],
        "faithfulness": 0.0,
    }


__all__ = ["direct_answer_node", "degrade_answer_node"]
