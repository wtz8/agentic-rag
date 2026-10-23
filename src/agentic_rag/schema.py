"""全局数据结构（03 §0）。

这些结构贯穿所有模块。实现阶段不得擅自变更字段语义：
- `Chunk.text` 是入库文本（含上下文增强块头），`Chunk.raw_text` 是展示用原文；
- `chunk_id` 的分类型段规则见 03 §0.1，父子块不共用 seq 空间。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "Block",
    "ParsedDocument",
    "Chunk",
    "QualityReport",
    "Candidate",
    "ScoredChunk",
    "Citation",
    "TraceEvent",
    "doc_id_of",
]


def doc_id_of(content: bytes) -> str:
    """`sha1(文件内容)[:16]`。

    不含文件路径：含路径会让同一份文档换个目录就产生新 doc_id，破坏幂等；
    不含路径时内容相同的副本自然合并，这正是想要的语义（02 §3.6）。
    """
    return hashlib.sha1(content).hexdigest()[:16]


@dataclass
class Block:
    """解析器输出的统一中间表示单元（防腐层的核心）。"""

    type: str  # heading | paragraph | table | list | image_caption | code
    text: str  # 表格为 Markdown 表格串
    level: int = 0  # 标题层级；非标题块为 0
    path: list[str] = field(default_factory=list)  # 到本块的标题路径
    page: int | None = None
    is_atomic: bool = False  # True 表示切分时不可再分（表格）
    meta: dict = field(default_factory=dict)


@dataclass
class QualityReport:
    """解析质量三道关卡的单文档结论（03 §4）。"""

    parser: str = ""
    page_count: int = 0
    is_scanned: bool = False
    char_count: int = 0  # 解析结果字符数
    raw_char_count: int = 0  # raw 基线字符数（0 表示无基线）
    digit_count_raw: int = 0
    digit_count_parsed: int = 0
    table_count_detected: int = 0
    table_count_parsed: int = 0
    table_integrity: float = 1.0  # 表格 round-trip 一致率 0-1
    heading_jumps: int = 0
    digit_loss: float | None = None  # None = 不适用
    text_coverage: float | None = None
    table_recall: float | None = None
    not_applicable: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    verdict: str = "ok"  # ok | warned | rejected

    def as_dict(self) -> dict[str, Any]:
        return {
            "parser": self.parser,
            "page_count": self.page_count,
            "is_scanned": self.is_scanned,
            "char_count": self.char_count,
            "raw_char_count": self.raw_char_count,
            "digit_count_parsed": self.digit_count_parsed,
            "digit_count_raw": self.digit_count_raw,
            "table_count_detected": self.table_count_detected,
            "table_count_parsed": self.table_count_parsed,
            "table_integrity": round(self.table_integrity, 4),
            "heading_jumps": self.heading_jumps,
            "digit_loss": None if self.digit_loss is None else round(self.digit_loss, 4),
            "text_coverage": None if self.text_coverage is None else round(self.text_coverage, 4),
            "table_recall": None if self.table_recall is None else round(self.table_recall, 4),
            "not_applicable": list(self.not_applicable),
            "issues": list(self.issues),
            "verdict": self.verdict,
        }


@dataclass
class ParsedDocument:
    doc_id: str
    source: str  # 文件名
    source_path: str
    blocks: list[Block]
    parser: str  # 实际使用的解析器名
    page_count: int
    quality: QualityReport = field(default_factory=QualityReport)


@dataclass
class Chunk:
    """检索单元或父块（03 §0 / §0.1）。"""

    chunk_id: str  # {doc_id}:c{seq:04d} 检索单元 | {doc_id}:p{seq:03d} 父块
    doc_id: str
    parent_id: str | None  # 子块指向所属父块；父块与 structural 块为 None
    is_parent: bool = False  # True = 仅供回填，不进向量与 BM25
    text: str = ""  # 入库文本 = 块头 + 原文
    raw_text: str = ""  # 不含块头的原文
    title_path: list[str] = field(default_factory=list)
    block_type: str = "paragraph"  # paragraph | table | heading | list
    is_atomic: bool = False
    page: int | None = None
    source: str = ""
    tokens: int = 0

    def to_jsonl(self) -> dict[str, Any]:
        """`data/index/chunks.jsonl` 的物化视图（只存检索需要的字段）。"""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "parent_id": self.parent_id,
            "is_parent": self.is_parent,
            "text": self.text,
            "raw_text": self.raw_text,
            "title_path": list(self.title_path),
            "block_type": self.block_type,
            "is_atomic": self.is_atomic,
            "page": self.page,
            "source": self.source,
            "tokens": self.tokens,
        }

    @classmethod
    def from_jsonl(cls, d: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            parent_id=d.get("parent_id"),
            is_parent=bool(d.get("is_parent", False)),
            text=d.get("text", ""),
            raw_text=d.get("raw_text", ""),
            title_path=list(d.get("title_path", [])),
            block_type=d.get("block_type", "paragraph"),
            is_atomic=bool(d.get("is_atomic", False)),
            page=d.get("page"),
            source=d.get("source", ""),
            tokens=int(d.get("tokens", 0)),
        )


@dataclass
class Candidate:
    """融合前的一路召回结果（04 §1）。

    `vector_score` / `bm25_score` 用 `float | None = None` 而非 `0.0`：
    `None` 只表示"这一路没产生分数"（`use_vector=False`，或该路调用失败）。
    """

    chunk_id: str
    vector_score: float | None = None
    vector_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    hit_routes: list[str] = field(default_factory=list)


@dataclass
class ScoredChunk:
    chunk: Chunk
    # None = "这个分数拿不到"（缓存命中重建时旧格式没存分数）。
    # 不能填 0.0 顶替：0.0 会被 K1 的达标计数与 UI 当成"检索到了但分是 0"。
    score: float | None
    rank: int
    hit_routes: list[str] = field(default_factory=list)


@dataclass
class Citation:
    index: int  # 正文中的角标序号
    chunk_id: str
    source: str
    title_path: list[str]
    page: int | None
    quote: str
    # 与 ScoredChunk.score 同一套语义：None = 拿不到分数（缓存回放的旧格式没存），
    # 不是 0.0。这里若填 0.0，UI 与评测会把"不知道"显示成"相似度为零"。
    score: float | None


@dataclass
class TraceEvent:
    """03 §3。step 枚举含 `fuse`。"""

    step: str  # route | rewrite | retrieve | fuse | rerank | reflect | generate | verify
    ts: float
    duration_ms: float
    payload: dict = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "ts": self.ts,
            "duration_ms": round(self.duration_ms, 3),
            "payload": self.payload,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }


def chunk_sort_key(items: Iterable[Chunk]) -> list[Chunk]:
    """按 chunk_id 稳定排序，保证落盘顺序与语料内容无关（可复现）。"""
    return sorted(items, key=lambda c: c.chunk_id)
