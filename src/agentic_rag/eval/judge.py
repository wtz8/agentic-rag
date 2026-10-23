"""LLM-as-judge（04 §9.2 / 05 §2）。

三条接口签名与设计文档保持一致；三条共同的不变量：

1. **裁判失败不得静默兜底**。返回 None + warnings，绝不填 0（拉低结论）
   也绝不填 1（虚高结论）——两者都会让消融的差值不可信。
2. **Mock 模式下 judge 不产生质量结论**（04 §9.5 的硬规矩在代码层的落点）：
   MockLLM 返回全空判定，本模块把"全空"判为"未作答"而非"全部支持"，
   于是 faithfulness / citation_accuracy 在 Mock 报告里自动变成"不可用"，
   而不是一组漂亮的假数字。
3. **faithfulness 的下标语义是代码拆分的命题**（05 §2，schema v2）：
   拆分用 `utils.split_sentences`，渲染与解析同一条规则，下标不可能错位。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..providers.base import LLMProtocol, AccountArrearage, first_json_object, is_arrearage_error
from ..prompts.loader import get_prompt
from ..schema import Citation
from ..utils import split_sentences, strip_citation_marks

__all__ = [
    "JudgeVerdict",
    "judge_faithfulness",
    "judge_relevance",
    "judge_citation_accuracy",
    "judge_all",
]


@dataclass
class JudgeVerdict:
    """一次完整判定的聚合结果（runner 逐条收集后聚合进 VariantResult）。"""

    faithfulness: float | None = None
    unsupported_claims: list[str] = field(default_factory=list)
    relevance: float | None = None  # 归一化到 [0,1]：(score-1)/4
    relevance_raw: int | None = None  # 裁判原始 1–5 分，便于人工复核
    citation_accuracy: float | None = None
    citation_judged: int = 0  # 参与引用判定的条数（分母，None 时为 0）
    warnings: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)


def _call(llm: LLMProtocol, prompt_name: str, **kwargs) -> dict:
    p = get_prompt(prompt_name)
    # first_json_object 内部已做宽容解析（extract_json），不能再套一层
    return first_json_object(llm.chat(p.messages(**kwargs), json_mode=p.json_mode))


def _fmt_claims(claims: list[str]) -> str:
    return "\n".join(f"{i}. {c}" for i, c in enumerate(claims))


def _fmt_context(context: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context))


def _fmt_citations(citations: list[Citation]) -> str:
    lines = []
    for i, c in enumerate(citations):
        loc = f"P{c.page}" if c.page else "无页码"
        lines.append(
            f"{i}. 出处 {c.source}｜{' > '.join(c.title_path) or '无标题'}｜{loc}\n   引文：{c.quote}"
        )
    return "\n".join(lines)


def _norm_indices(raw, n: int) -> set[int]:
    out: set[int] = set()
    if not isinstance(raw, list):
        return out
    for v in raw:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n:
            out.add(i)
    return out


def judge_faithfulness(
    answer: str, context: list[str], llm: LLMProtocol
) -> tuple[float | None, list[str]]:
    """忠实度：答案命题被上下文支持的比例。

    返回 (支持率, 无支撑命题列表)。支持率为 None 表示裁判未产生可用判定
    （claims 为空 / 裁判返回全空 / 调用失败）——None 与 0.0 含义不同：
    0.0 是"裁判判了，全都不支持"，None 是"这次没判成"。
    """
    claims = split_sentences(strip_citation_marks(answer))
    if not claims:
        return None, []
    if not context:
        return None, []
    try:
        d = _call(
            llm,
            "judge_faithfulness",
            answer=strip_citation_marks(answer),
            claims=_fmt_claims(claims),
            context=_fmt_context(context),
        )
    except Exception as e:  # noqa: BLE001 - 裁判失败不让整轮评测崩掉，但要可见
        if is_arrearage_error(e):  # 欠费是终态，吞掉它会让整轮 judge 静默变 None
            raise AccountArrearage(f"judge_faithfulness 阶段账号欠费: {e}") from e
        return None, []
    supported = _norm_indices(d.get("supported"), len(claims))
    unsupported = _norm_indices(d.get("unsupported"), len(claims))
    if not supported and not unsupported:
        # 全空 = 未作答（Mock 的满分结构走的就是这条路），不能当成"全部支持"
        return None, []
    judged = supported | unsupported
    if not judged:
        return None, []
    bad = [claims[i] for i in sorted(unsupported & judged)]
    return round(1.0 - len(judged - supported) / len(judged), 4), bad


def judge_relevance(question: str, answer: str, llm: LLMProtocol) -> float | None:
    """相关性：答案是否回应了问题（1–5 归一化到 [0,1]）。只判回应，不判对错。"""
    try:
        d = _call(llm, "judge_relevance", question=question, answer=answer)
        score = int(d.get("score"))
    except (ValueError, TypeError):
        return None
    except Exception as e:  # noqa: BLE001
        if is_arrearage_error(e):
            raise AccountArrearage(f"judge_relevance 阶段账号欠费: {e}") from e
        return None
    if not 1 <= score <= 5:
        return None
    return round((score - 1) / 4.0, 4)


def judge_citation_accuracy(
    citations: list[Citation], answer: str, llm: LLMProtocol, context: list[str] | None = None
) -> float | None:
    """引用准确率：被引用的出处是否真的支持它标注的那句话。

    与答案正确性分开统计——答案对但引错出处，在企业场景同样不可用。
    无引用时返回 None（无引用谈不上准确率，不能拿 1.0 稀释）。
    """
    if not citations:
        return None
    try:
        d = _call(
            llm,
            "judge_citation",
            answer=answer,
            citations=_fmt_citations(citations),
            context=_fmt_context(context or [c.quote for c in citations]),
        )
    except Exception as e:  # noqa: BLE001
        if is_arrearage_error(e):
            raise AccountArrearage(f"judge_citation 阶段账号欠费: {e}") from e
        return None
    correct = _norm_indices(d.get("correct"), len(citations))
    wrong = _norm_indices(d.get("wrong"), len(citations))
    if not correct and not wrong:
        return None  # 全空 = 未作答，不是"全部正确"
    judged = correct | wrong
    if not judged:
        return None
    return round(len(correct & judged) / len(judged), 4)


def judge_all(
    question: str,
    answer: str,
    context: list[str],
    citations: list[Citation],
    llm: LLMProtocol,
) -> JudgeVerdict:
    """一次性跑三个裁判。单条查询的完整判定，供 runner 逐条调用。"""
    v = JudgeVerdict()
    v.claims = split_sentences(strip_citation_marks(answer))
    if answer.strip():
        faith, bad = judge_faithfulness(answer, context, llm)
        v.faithfulness = faith
        v.unsupported_claims = bad
        if faith is None:
            v.warnings.append("judge_faithfulness_unavailable")
        v.relevance_raw = None
        rel = judge_relevance(question, answer, llm)
        v.relevance = rel
        if rel is None:
            v.warnings.append("judge_relevance_unavailable")
    else:
        v.warnings.append("empty_answer")
    if citations:
        acc = judge_citation_accuracy(citations, answer, llm, context)
        v.citation_accuracy = acc
        v.citation_judged = 0 if acc is None else len(citations)
        if acc is None:
            v.warnings.append("judge_citation_unavailable")
    return v
