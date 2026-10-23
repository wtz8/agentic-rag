"""解析器路由与降级链（03 §5）。

```
DoclingParser      失败/未安装 → PdfParser
PdfParser          失败        → PdfPlainTextParser（仅抽文本，丢结构，记 warned）
PdfPlainTextParser 失败        → 拒绝入库
```

**降级必须写进 `QualityReport.issues`**——不记录的降级等于静默降质。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..schema import ParsedDocument
from .parsers import (
    DoclingParser,
    DocxParser,
    HtmlParser,
    MarkdownParser,
    PdfParser,
    PdfPlainTextParser,
    PptxParser,
    TextParser,
    XlsxParser,
)
from .quality import ProbeResult

SUPPORTED_EXTS = {
    ".pdf",
    ".docx",
    ".doc",
    ".xlsx",
    ".xls",
    ".pptx",
    ".ppt",
    ".html",
    ".htm",
    ".md",
    ".markdown",
    ".txt",
}


class UnsupportedFormat(Exception):
    """未知扩展名：拒绝入库，而不是静默当成 txt。"""


def route(path: Path, probe: ProbeResult, cfg: Config) -> object:
    """按扩展名与探测结果选解析器（返回 03 §5 清单中的具体类实例）。"""
    ext = Path(path).suffix.lower()
    if ext in (".docx", ".doc"):
        return DocxParser()
    if ext in (".xlsx", ".xls"):
        return XlsxParser()
    if ext in (".pptx", ".ppt"):
        return PptxParser()
    if ext in (".html", ".htm"):
        return HtmlParser()
    if ext in (".md", ".markdown"):
        return MarkdownParser()
    if ext == ".txt":
        return TextParser()
    if ext == ".pdf":
        if cfg.ingestion.use_docling and probe.table_density > cfg.ingestion.docling_table_density_threshold:
            return DoclingParser()
        if probe.is_scanned:
            return DoclingParser()  # 扫描件走 OCR；未安装则由降级链兜底
        return PdfParser(cfg)  # 传入 cfg：页码恢复要靠后端数页数（N9）
    raise UnsupportedFormat(f"不支持的格式: {ext}")


@dataclass
class ParseOutcome:
    doc: ParsedDocument | None
    parser_used: str
    degraded: bool
    error: str = ""


def parse_with_fallback(path: Path, probe: ProbeResult, cfg: Config) -> ParseOutcome:
    """按降级链解析；失败即降级，降级写进 quality.issues。"""
    path = Path(path)
    try:
        primary = route(path, probe, cfg)
    except UnsupportedFormat as e:
        return ParseOutcome(None, "", False, str(e))

    chain: list[object] = [primary]
    if type(primary) is DoclingParser:
        chain.append(PdfParser(cfg))
    if type(primary) in (DoclingParser, PdfParser):
        chain.append(PdfPlainTextParser(cfg))

    last_err = ""
    for i, parser in enumerate(chain):
        try:
            doc = parser.parse(path)  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001 - 任一解析器失败都走下一级
            last_err = f"{type(parser).__name__}: {e}"
            continue
        if i > 0:
            issue = f"parser_degraded:{type(parser).__name__}"
            doc.quality.issues.append(issue)
            return ParseOutcome(doc, doc.parser, True, "")
        return ParseOutcome(doc, doc.parser, False, "")
    return ParseOutcome(None, "", True, last_err or "全部解析器失败")


__all__ = ["route", "parse_with_fallback", "UnsupportedFormat", "SUPPORTED_EXTS", "ParseOutcome"]
