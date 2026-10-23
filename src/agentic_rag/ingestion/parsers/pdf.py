"""PDF 解析器（03 §5）。

- `PdfParser`：默认快速通道，`pymupdf4llm` 转 Markdown（保留标题与表格结构）
- `PdfPlainTextParser`：降级用，只抽纯文本，丢结构，必须记 warned
- `DoclingParser`：heavy backend，表格密集或扫描件（含 OCR）时启用，未安装则降级

**页码从哪来（N9）**：早期实现假设 `pymupdf4llm` 会在页与页之间插入 `-----` 分隔线，
再按分隔线数页。实测该分隔线**从未存在**（1.28.2 与 2.0 都不插入），于是多页 PDF 的
全部块都被标成第 1 页——而本地因为 `pymupdf4llm` 装成了空壳包、走降级路径，测试反而"绿"。
现在改用 `to_markdown(page_chunks=True)` 的逐页结果，页码取 `metadata.page_number`；
拿不到页边界时**宁可降级到保页码的纯文本路径**，也不能把 3 页标成 1 页。
"""

from __future__ import annotations

import re
from pathlib import Path

from ...config import Config
from ...schema import Block, ParsedDocument
from ..pdf_backend import get_pdf_backend
from ..quality import markdown_to_blocks
from .base import ParserError, make_doc

_PAGE_SEP = re.compile(r"^\s*-{4,}\s*$")


def _split_pages(md: str) -> list[tuple[int, str]]:
    """按 `-----` 分页的**遗留**路径：仅当 pymupdf4llm 太老、不支持 page_chunks 时使用。"""
    out: list[tuple[int, str]] = []
    cur_page = 1
    buf: list[str] = []
    for line in (md or "").splitlines():
        if _PAGE_SEP.match(line):
            if buf:
                out.append((cur_page, "\n".join(buf)))
                buf = []
            cur_page += 1
            continue
        buf.append(line)
    if buf:
        out.append((cur_page, "\n".join(buf)))
    return out


class PdfParser:
    name = "pymupdf4llm"

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg

    # ── 页码恢复 ──────────────────────────────────────────────────
    def _page_markdowns(self, path: Path, module) -> list[tuple[int, str]]:
        """返回 [(页码, markdown)]。优先逐页 API，其次分隔线，最后宁可降级。"""
        try:
            chunks = module.to_markdown(str(path), page_chunks=True)
        except TypeError:
            chunks = None  # 老版本不认这个参数
        if isinstance(chunks, list) and chunks:
            pages: list[tuple[int, str]] = []
            for i, chunk in enumerate(chunks, start=1):
                if not isinstance(chunk, dict):
                    continue
                meta = chunk.get("metadata") or {}
                page_no = meta.get("page_number") or i
                pages.append((int(page_no), chunk.get("text") or ""))
            if pages:
                return pages

        # 遗留路径：plain markdown + 分隔线
        md = module.to_markdown(str(path))
        pages = _split_pages(md)
        if self._needs_more_pages(path) and len(pages) == 1:
            raise ParserError(
                "markdown 未恢复页边界（该 PDF 多页但只解析出 1 页），"
                "降级到保页码的纯文本路径"
            )
        return pages

    def _backend_page_count(self, path: Path) -> int:
        try:
            cfg = self.cfg
            if cfg is None:
                from ...config import get_config

                cfg = get_config()
            return int(get_pdf_backend(cfg).page_count(path))
        except Exception:  # noqa: BLE001 - 数不清页数时不阻断解析
            return 0

    def _needs_more_pages(self, path: Path) -> bool:
        return self._backend_page_count(path) > 1

    def parse(self, path: Path) -> ParsedDocument:
        try:
            import pymupdf4llm
        except Exception as e:  # noqa: BLE001
            raise ParserError(f"pymupdf4llm 不可用: {e}") from e
        try:
            pages = self._page_markdowns(path, pymupdf4llm)
        except ParserError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ParserError(f"PDF 解析失败: {e}") from e
        if not any((text or "").strip() for _, text in pages):
            raise ParserError("PDF 解析结果为空")
        blocks: list[Block] = []
        for page_no, chunk in pages:
            for b in markdown_to_blocks(chunk):
                b.page = page_no
                blocks.append(b)
        if not blocks:
            raise ParserError("PDF 解析结果为空")
        return make_doc(path, blocks, self.name, page_count=max(1, len(pages)))


class PdfPlainTextParser:
    """仅 PDF 降级用，不参与首次路由（J8）。"""

    name = "pypdf_text"

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg

    def parse(self, path: Path) -> ParsedDocument:
        from ...config import get_config

        cfg = self.cfg or get_config()
        try:
            texts = get_pdf_backend(cfg).page_texts(Path(path))
        except Exception as e:  # noqa: BLE001
            raise ParserError(f"纯文本抽取失败: {e}") from e
        blocks = [
            Block(type="paragraph", text=t.strip(), page=i + 1)
            for i, t in enumerate(texts)
            if t.strip()
        ]
        if not blocks:
            raise ParserError("纯文本抽取结果为空")
        return make_doc(path, blocks, self.name, page_count=len(texts))


class DoclingParser:
    """可选 heavy backend：表格密集 PDF 与扫描件（含 OCR）走它。"""

    name = "docling"

    def parse(self, path: Path) -> ParsedDocument:
        try:
            from docling.document_converter import DocumentConverter  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise ParserError(f"docling 未安装: {e}") from e
        try:
            result = DocumentConverter().convert(str(path))
            md = result.document.export_to_markdown()
        except Exception as e:  # noqa: BLE001
            raise ParserError(f"docling 解析失败: {e}") from e
        if not (md or "").strip():
            raise ParserError("docling 解析结果为空")
        return make_doc(path, markdown_to_blocks(md), self.name)


__all__ = ["PdfParser", "PdfPlainTextParser", "DoclingParser"]
