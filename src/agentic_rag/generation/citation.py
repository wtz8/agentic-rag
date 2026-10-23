"""引用溯源（02 §5.3 / 04 §8.1）。

企业用户不信没有出处的答案。三条规则：
- 上下文按 `[1] [2] …` 编号注入，模型直接输出编号（不输出 32 字符的 chunk_id——
  复制长 ID 的错误率显著高于写一个数字，H20）
- 无法映射的编号**直接丢弃**，不把裸 `[n]` 暴露给用户
- 引用准确率与答案准确率是两个独立维度（引对答案但引错出处同样不可用）
"""

from __future__ import annotations

import re

from ..schema import Citation, ScoredChunk

_MARK = re.compile(r"\[(\d+)\]")
_SENT = re.compile(r"(?<=[。！？；!?;])")


def build_context_block(context: list[ScoredChunk]) -> str:
    """把上下文组装为带 `[n]` 编号的文本（编号从 1 连续）。"""
    parts = []
    for i, sc in enumerate(context, start=1):
        parts.append(f"[{i}] {sc.chunk.raw_text}")
    return "\n\n".join(parts)


def strip_marks(text: str) -> str:
    """去掉角标后的正文（命题拆分用）。"""
    return _MARK.sub("", text or "").strip()


def parse_citations(draft: str, context: list[ScoredChunk]) -> tuple[str, list[Citation]]:
    """把 draft 中的 `[n]` 映射为 Citation 列表，并归一化编号。

    返回 (归一化后的答案, citations)。越界编号与无法映射的编号直接丢弃。
    """
    if not draft or not context:
        return (draft or ""), []

    order: list[int] = []  # 首次出现顺序 → 新编号
    mapping: dict[int, int] = {}
    for m in _MARK.finditer(draft):
        try:
            old = int(m.group(1))
        except ValueError:
            continue
        if old < 1 or old > len(context):
            continue  # 越界：丢弃，不保留裸 [n]
        if old not in mapping:
            mapping[old] = len(order) + 1
            order.append(old)

    def _sub(m: re.Match) -> str:
        try:
            old = int(m.group(1))
        except ValueError:
            return ""
        if old not in mapping:
            return ""
        return f"[{mapping[old]}]"

    clean = _MARK.sub(_sub, draft)
    clean = re.sub(r"\s{2,}", " ", clean).strip()

    citations: list[Citation] = []
    for old in order:
        sc = context[old - 1]
        citations.append(
            Citation(
                index=mapping[old],
                chunk_id=sc.chunk.chunk_id,
                source=sc.chunk.source,
                title_path=list(sc.chunk.title_path),
                page=sc.chunk.page,
                quote=_quote(sc.chunk.raw_text),
                # 不做 float(sc.score)：score 是 `float | None`，None 表示"拿不到分数"。
                # 在这里兜成 0.0 会把"不知道"写死成"相似度为零"，
                # 与 ScoredChunk 的约定、以及 K1 达标计数的口径分叉。
                score=sc.score,
            )
        )
    return clean, citations


def _quote(text: str, limit: int = 120) -> str:
    t = (text or "").strip().replace("\n", " ")
    return t[:limit] + ("…" if len(t) > limit else "")


def extract_claims(answer: str) -> list[str]:
    """按句号把答案拆成原子命题（在线链路用 LLM 拆分，这里是规则兜底）。"""
    parts = [p.strip() for p in _SENT.split(strip_marks(answer or ""))]
    return [p for p in parts if p]


__all__ = ["parse_citations", "build_context_block", "extract_claims", "strip_marks"]
