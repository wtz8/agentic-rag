"""引用校验节点（04 §6.7）：命题级忠实度。

**本节点是 2 次 LLM 调用**（claim_extract + claim_verify），不是 1 次（H7）。
`faithfulness = 1 - len(unsupported)/len(claims)` 由**代码按公式算**，不采信模型自报的数字。

`confidence` 由本节点统一写入，其他任何节点都不得写它（L9）。
"""

from __future__ import annotations

from ...config import Config
from ...generation.citation import strip_marks
from ...prompts.loader import get_prompt
from ...providers.base import ProviderError, first_json_object
from ..context import ctx_of
from ..conditions import should_regenerate


def verify_node(state, config) -> dict:
    ctx = ctx_of(config)
    cfg: Config = ctx.cfg
    llm = ctx.deps.llm_light
    draft = state.get("draft", "") or ""
    context = list(state.get("context") or [])
    body = strip_marks(draft)

    claims: list[str] = []
    unsupported: list[int] = []
    verify_failed = False

    with ctx.tracer.span("verify") as payload:
        # ① 命题拆分
        try:
            p = get_prompt("claim_extract")
            data = first_json_object(
                llm.chat(p.messages(answer=body), temperature=p.temperature, json_mode=True)
            )
            raw_claims = data.get("claims")
            if isinstance(raw_claims, list):
                claims = [str(c).strip() for c in raw_claims if str(c).strip()]
        except ProviderError:
            verify_failed = True
        except Exception:  # noqa: BLE001
            verify_failed = True

        # ② 逐条支持判定
        if claims and not verify_failed:
            ctx_block = "\n\n".join(
                f"[{i}] {sc.chunk.raw_text}" for i, sc in enumerate(context, start=1)
            )
            try:
                p = get_prompt("claim_verify")
                data = first_json_object(
                    llm.chat(
                        p.messages(claims="\n".join(claims), context=ctx_block),
                        temperature=p.temperature,
                        json_mode=True,
                    )
                )
                raw = data.get("unsupported")
                if isinstance(raw, list):
                    unsupported = sorted(
                        {int(i) for i in raw if isinstance(i, (int, float)) and 0 <= int(i) < len(claims)}
                    )
            except ProviderError:
                verify_failed = True
            except Exception:  # noqa: BLE001
                verify_failed = True

        if not claims:
            # 没有可校验的命题：不把"没拆出来"当成"全部无支撑"
            faithfulness = 1.0
        else:
            faithfulness = round(1 - len(unsupported) / len(claims), 6)

        confidence = _decide_confidence(
            state=state,
            cfg=cfg,
            faithfulness=faithfulness,
            verify_failed=verify_failed,
            degraded=bool(state.get("degraded")),
        )
        weak = 0 < int(state.get("reflect_strong", 0)) < cfg.retrieval.min_candidates

        out = {
            "claims": claims,
            "unsupported": unsupported,
            "faithfulness": faithfulness,
            "confidence": confidence,
            "answer": draft,
            "verify_failed": verify_failed,
            "weak_evidence": weak,
        }

        # verify 事件由 tracer.span 退出时发出（唯一发送方），此处不再另行 push，
        # 否则同一事件会出现两次（trace + SSE 各一份）
        payload.update(
            {
                "claims": len(claims),
                "unsupported": unsupported,
                "faithfulness": faithfulness,
                "confidence": confidence,
                "verify_failed": verify_failed,
            }
        )

    # regenerate 事件的唯一发送方，且与 need_regenerate 共用同一份判据（L10）。
    # 放在 span 之后推，UI 时间线上才是"先出校验结论、再声明重生成"。
    if should_regenerate({**state, **out}, cfg):
        ctx.push(
            "regenerate",
            {
                "iteration": int(state.get("generate_iterations", 0)),
                "reason": f"{len(unsupported)} 条命题未找到出处，忠实度 {faithfulness:.2f}",
            },
        )
    return out


def _decide_confidence(
    state: dict, cfg: Config, faithfulness: float, verify_failed: bool, degraded: bool
) -> str:
    """04 §6.7 的四条分支，覆盖全部终止出口，`done` 里的 confidence 因此总有确定取值。"""
    if degraded:
        return "low"  # 一条达标证据都没有，系统选择不编
    if verify_failed:
        return "medium"  # 无法给出判定，如实标注"未校验"
    if faithfulness < cfg.generation.faithfulness_threshold:
        return "medium"  # 答案成立，但有命题未找到出处
    if int(state.get("reflect_strong", 0)) < cfg.retrieval.min_candidates:
        return "medium"  # 答案成立，但支撑偏薄（L5 的证据偏少路径）
    return "high"


__all__ = ["verify_node", "_decide_confidence"]
