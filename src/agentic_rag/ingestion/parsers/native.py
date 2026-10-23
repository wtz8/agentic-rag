"""原生文本解析器：Markdown / TXT（03 §5）。"""

from __future__ import annotations

from pathlib import Path

from ...schema import ParsedDocument
from ..quality import markdown_to_blocks
from .base import ParserError, make_doc


class MarkdownParser:
    name = "native"

    def parse(self, path: Path) -> ParsedDocument:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            raise ParserError(f"Markdown 为空: {path}")
        return make_doc(path, markdown_to_blocks(text), self.name)


class TextParser:
    name = "native"

    def parse(self, path: Path) -> ParsedDocument:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            raise ParserError(f"文本为空: {path}")
        blocks = markdown_to_blocks(text)  # 无标题时退化为整段 paragraph
        return make_doc(path, blocks, self.name)


__all__ = ["MarkdownParser", "TextParser"]
