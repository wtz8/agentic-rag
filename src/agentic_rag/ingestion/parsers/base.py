"""解析器公共契约（03 §5）。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from ...schema import Block, ParsedDocument, QualityReport


class ParserError(Exception):
    """解析失败，交给路由器走降级链。"""


class BaseParser(Protocol):
    name: str

    def parse(self, path: Path) -> ParsedDocument: ...


def make_doc(
    path: Path,
    blocks: list[Block],
    parser: str,
    page_count: int = 0,
    quality: QualityReport | None = None,
) -> ParsedDocument:
    """统一构造 ParsedDocument：doc_id 由**纯内容**哈希生成（不含路径）。"""
    data = Path(path).read_bytes()
    return ParsedDocument(
        doc_id=hashlib.sha1(data).hexdigest()[:16],
        source=Path(path).name,
        source_path=str(path),
        blocks=blocks,
        parser=parser,
        page_count=page_count,
        quality=quality or QualityReport(),
    )
