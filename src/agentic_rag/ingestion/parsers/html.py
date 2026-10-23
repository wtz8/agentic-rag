"""HTML 解析器（03 §5）。"""

from __future__ import annotations

from pathlib import Path

from ...schema import ParsedDocument
from ..quality import markdown_to_blocks
from .base import ParserError, make_doc

try:  # 与 office 共用同一条 markitdown 通路
    from markitdown import MarkItDown as _MarkItDown
except Exception:  # noqa: BLE001
    _MarkItDown = None  # type: ignore


class HtmlParser:
    name = "markitdown"

    def __init__(self) -> None:
        self.via_fallback = False

    def parse(self, path: Path) -> ParsedDocument:
        md = ""
        if _MarkItDown is not None:
            try:
                md = _MarkItDown().convert(str(path)).text_content or ""
            except Exception:  # noqa: BLE001
                md = ""
        if not md.strip():
            md = _bs4_md(Path(path))
            self.via_fallback = True
        if not md.strip():
            raise ParserError(f"HTML 解析结果为空: {path}")
        doc = make_doc(path, markdown_to_blocks(md), self.name)
        if self.via_fallback:
            doc.quality.issues.append("parser_fallback_native")
        return doc


def _bs4_md(path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(Path(path).read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    out: list[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table", "pre"]):
        name = el.name
        if name.startswith("h") and name[1:].isdigit():
            out.append("#" * int(name[1:]) + " " + el.get_text(strip=True))
        elif name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [c.get_text(strip=True) for c in tr.find_all(["th", "td"])]
                if cells:
                    rows.append(cells)
            if rows:
                width = max(len(r) for r in rows)
                lines = []
                for i, r in enumerate(rows):
                    cells = list(r) + [""] * (width - len(r))
                    lines.append("| " + " | ".join(cells) + " |")
                    if i == 0:
                        lines.append("| " + " | ".join(["---"] * width) + " |")
                out.append("\n".join(lines))
        elif name == "pre":
            out.append("```\n" + el.get_text() + "\n```")
        else:
            text = el.get_text(strip=True)
            if text:
                out.append(text)
    return "\n\n".join(out)


__all__ = ["HtmlParser"]
