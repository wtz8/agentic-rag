"""切分公共逻辑（03 §6）：上下文增强块头、文本切分、表格四档规则、策略工厂。"""

from __future__ import annotations

import re
from typing import Protocol

from ...config import IngestionConfig
from ...schema import Chunk, ParsedDocument
from ...utils import count_tokens
from ..quality import split_table_header

_SENT = re.compile(r"(?<=[。！？；!?;])")
_PARA = re.compile(r"\n\s*\n")


def with_header(chunk: Chunk) -> str:
    """`[文档名 > 一级标题 > 二级标题] + 原文`。

    零成本，解决 chunk 脱离文档后的指代丢失（"它的上限是多少？" —— "它"指谁？）。
    """
    path = " > ".join([chunk.source, *chunk.title_path])
    return f"[{path}]\n{chunk.raw_text}"


def split_text(text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    """按段落 → 句子 → 字符的次序切分，尽量保持语义边界。"""
    text = (text or "").strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]
    pieces: list[str] = []
    for para in _PARA.split(text):
        para = para.strip()
        if not para:
            continue
        if count_tokens(para) <= max_tokens:
            pieces.append(para)
            continue
        buf = ""
        for sent in _SENT.split(para):
            if not sent:
                continue
            if buf and count_tokens(buf + sent) > max_tokens:
                pieces.append(buf.strip())
                buf = _tail(buf, overlap_tokens)
            buf += sent
        if buf.strip():
            pieces.append(buf.strip())
    # 字符级兜底：无句读的文本（整段源码、base64、长 URL 串）在句子级切不开，
    # 会原样漏出一个巨块。56k 字符的块超过 embedding 模型的输入上限 →
    # 该块永远拿不到向量（且会连累同批次的块一起 400）。docstring 承诺的第三级
    # 以前只是写了没实现。
    out: list[str] = []
    for p in pieces:
        if count_tokens(p) > max_tokens:
            out.extend(_hard_split(p, max_tokens, overlap_tokens))
        else:
            out.append(p)
    return [p for p in out if p.strip()]


def _hard_split(text: str, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    """按 token 预算硬切（二分找切点），保留 overlap 后缀。

    只在段落级与句子级都切不开时才触发——它破坏语义边界，但总好过产出一个
    超限到无法 embedding 的块。
    """
    out: list[str] = []
    start, n = 0, len(text)
    while start < n:
        lo, hi = start + 1, n  # 找满足 token 预算的最长 end
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count_tokens(text[start:mid]) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        end = lo
        piece = text[start:end]
        out.append(piece.strip())
        if end >= n:
            break
        nxt = end
        if overlap_tokens > 0:
            tail = _tail(piece, overlap_tokens)
            if tail:
                nxt = end - len(tail)
        if nxt <= start:  # 兜底防死循环：overlap 吃掉了整个窗口
            nxt = end
        start = nxt
    return [p for p in out if p]


def _tail(text: str, overlap_tokens: int) -> str:
    """取末尾 overlap 个 token 作为下一片的重叠前缀。"""
    if overlap_tokens <= 0 or not text:
        return ""
    chars = len(text)
    lo, hi = 0, chars
    while lo < hi:  # 二分找最短后缀
        mid = (lo + hi) // 2
        if count_tokens(text[mid:]) <= overlap_tokens:
            hi = mid
        else:
            lo = mid + 1
    return text[lo:]


def pack_pieces(pieces: list[str], max_tokens: int) -> list[str]:
    """贪心合并相邻小片，避免产出大量碎片 chunk（单片超限则独立成块）。"""
    out: list[str] = []
    buf = ""
    for p in pieces:
        if not buf:
            buf = p
            continue
        if count_tokens(buf) + count_tokens(p) <= max_tokens:
            buf = buf + "\n" + p
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return [o for o in out if o.strip()]


def split_table_text(text: str, cfg: IngestionConfig) -> list[tuple[str, bool]]:
    """表格四档规则（03 §6.4）：**原子性优先于子块粒度**。

    | tokens | 处置 | is_atomic |
    | --- | --- | --- |
    | ≤ child_chunk_tokens | 整表一块 | True |
    | ≤ table_split_threshold_tokens | 一个超长子块（豁免子块上限） | True |
    | > table_split_threshold_tokens | 按行组切，每组重复表头 | True |
    | 单行 > atomic_max_tokens | 放弃原子性，按字符切 | False |
    """
    t = count_tokens(text or "")
    if t <= cfg.child_chunk_tokens:
        return [(text, True)]
    if t <= cfg.table_split_threshold_tokens:
        return [(text, True)]
    header, rows = split_table_header(text)
    if not rows:
        return [(text, True)]
    # 单行就超过硬上限 → 放弃原子性
    if any(count_tokens(r) > cfg.atomic_max_tokens for r in rows):
        pieces = split_text(text, cfg.table_split_threshold_tokens)
        return [(p, False) for p in pieces]
    groups: list[list[str]] = [[]]
    cur = count_tokens(header)
    for r in rows:
        rt = count_tokens(r)
        if cur + rt > cfg.table_split_threshold_tokens and groups[-1]:
            groups.append([])
            cur = count_tokens(header)
        groups[-1].append(r)
        cur += rt
    out = []
    for g in groups:
        if not g:
            continue
        body = (header + "\n" + "\n".join(g)).strip()
        out.append((body, True))
    return out or [(text, True)]


class BaseChunker(Protocol):
    name: str

    def chunk(self, doc: ParsedDocument) -> list[Chunk]: ...


class _Unit:
    """切分前的中间单元：一段不可分割或可切的原文。"""

    __slots__ = ("text", "is_atomic", "block_type", "page", "title_path", "tokens")

    def __init__(self, text: str, is_atomic: bool, block_type: str, page, title_path: list[str]):
        self.text = text
        self.is_atomic = is_atomic
        self.block_type = block_type
        self.page = page
        self.title_path = list(title_path)
        self.tokens = count_tokens(text)


def doc_units(doc: ParsedDocument, cfg: IngestionConfig) -> list[_Unit]:
    """把 blocks 摊平成切分单元：表格走四档规则，其余按原文。"""
    units: list[_Unit] = []
    for b in doc.blocks:
        if b.type == "table":
            for text, atomic in split_table_text(b.text, cfg):
                units.append(_Unit(text, atomic, "table", b.page, b.path))
            continue
        if b.type == "heading":
            # 标题自身不单独成块：它已进 title_path，避免重复入库
            continue
        if not (b.text or "").strip():
            continue
        units.append(_Unit(b.text.strip(), False, b.type, b.page, b.path))
    return units


def section_key(u: _Unit) -> str:
    return "\x1f".join(u.title_path)


def make_chunk(
    doc_id: str,
    chunk_id: str,
    parent_id: str | None,
    is_parent: bool,
    raw_text: str,
    title_path: list[str],
    block_type: str,
    is_atomic: bool,
    page,
    source: str,
) -> Chunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        parent_id=parent_id,
        is_parent=is_parent,
        raw_text=raw_text,
        title_path=list(title_path),
        block_type=block_type,
        is_atomic=is_atomic,
        page=page,
        source=source,
        tokens=count_tokens(raw_text),
    )
    chunk.text = with_header(chunk)
    return chunk


def get_chunker(cfg: IngestionConfig):
    from .parent_child import ParentChildChunker
    from .structural import StructuralChunker

    if cfg.chunk_strategy == "structural":
        return StructuralChunker(cfg)
    return ParentChildChunker(cfg)


__all__ = [
    "with_header",
    "split_text",
    "pack_pieces",
    "split_table_text",
    "doc_units",
    "make_chunk",
    "get_chunker",
    "BaseChunker",
    "section_key",
]
