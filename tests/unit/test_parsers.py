"""解析器路由与降级链（07 §3）。"""

from __future__ import annotations

import pytest

from agentic_rag.ingestion.parsers import (
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
from agentic_rag.ingestion.quality import ProbeResult
from agentic_rag.ingestion.router import UnsupportedFormat, parse_with_fallback, route


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a.pdf", PdfParser),
        ("a.docx", DocxParser),
        ("a.doc", DocxParser),
        ("a.xlsx", XlsxParser),
        ("a.xls", XlsxParser),
        ("a.pptx", PptxParser),
        ("a.ppt", PptxParser),
        ("a.html", HtmlParser),
        ("a.htm", HtmlParser),
        ("a.md", MarkdownParser),
        ("a.markdown", MarkdownParser),
        ("a.txt", TextParser),
    ],
)
def test_route_covers_every_extension(cfg, tmp_path, name, expected):
    p = tmp_path / name
    p.write_text("x", encoding="utf-8")
    assert isinstance(route(p, ProbeResult(), cfg), expected)


def test_route_scanned_pdf_goes_docling(cfg, tmp_path):
    p = tmp_path / "s.pdf"
    p.write_bytes(b"%PDF-1.4")
    probe = ProbeResult(is_scanned=True)
    assert isinstance(route(p, probe, cfg), DoclingParser)


def test_route_table_dense_pdf_goes_docling_when_enabled(cfg, tmp_path):
    p = tmp_path / "t.pdf"
    p.write_bytes(b"%PDF-1.4")
    cfg.ingestion.use_docling = True
    probe = ProbeResult(table_density=cfg.ingestion.docling_table_density_threshold + 0.1)
    assert isinstance(route(p, probe, cfg), DoclingParser)
    cfg.ingestion.use_docling = False
    assert isinstance(route(p, probe, cfg), PdfParser)


def test_unsupported_extension_raises(cfg, tmp_path):
    p = tmp_path / "a.xyz"
    p.write_text("x")
    with pytest.raises(UnsupportedFormat):
        route(p, ProbeResult(), cfg)
    out = parse_with_fallback(p, ProbeResult(), cfg)
    assert out.doc is None and "不支持的格式" in out.error


@pytest.mark.parametrize(
    "kind,expect_block_types",
    [
        ("md", {"heading", "table", "paragraph"}),
        ("txt", {"paragraph"}),
        ("html", {"heading", "table", "list"}),
        ("docx", {"heading", "table", "paragraph"}),
        ("xlsx", {"heading", "table"}),
        ("pptx", {"heading", "paragraph"}),
        ("pdf", {"paragraph"}),
    ],
)
def test_parsers_produce_blocks(cfg, make_file, kind, expect_block_types):
    p = make_file(kind)
    out = parse_with_fallback(p, ProbeResult(suggested_parser="x"), cfg)
    assert out.doc is not None, f"{kind} 解析失败: {out.error}"
    types = {b.type for b in out.doc.blocks}
    assert expect_block_types <= types, f"{kind} 缺块类型: {expect_block_types - types}"
    assert out.doc.doc_id and len(out.doc.doc_id) == 16
    assert all(b.text.strip() for b in out.doc.blocks)


def test_pdf_parser_keeps_page_numbers(cfg, make_file):
    """多页 PDF 的块必须带正确页码，且必须由**主解析器**给出（不是降级蒙对的）。

    N9：早期实现靠 markdown 里的 `-----` 分隔线数页，而该分隔线从未存在过，
    于是 3 页全标成第 1 页；本地因为 pymupdf4llm 装成空壳、走了保页码的降级路径，
    这条用例反而"绿"。所以这里同时断言 `degraded is False`——降级通过不算通过。
    """
    p = make_file("pdf", pages=3)
    out = parse_with_fallback(p, ProbeResult(suggested_parser="pymupdf4llm"), cfg)
    pages = {b.page for b in out.doc.blocks if b.page}
    assert pages == {1, 2, 3}
    assert out.degraded is False, f"页码正确但走了降级路径: {out.doc.quality.issues}"
    assert out.parser_used == "pymupdf4llm"


def test_pdf_parser_degrades_when_page_boundaries_are_lost(cfg, make_file, monkeypatch):
    """拿不到页边界时必须降级（并记 issue），绝不把多页标成第 1 页。

    构造一个"老版 pymupdf4llm"：不认 `page_chunks` 参数，且 plain markdown 里没有分隔线。
    """
    import sys
    import types

    from agentic_rag.ingestion.parsers.pdf import PdfParser
    from agentic_rag.ingestion.parsers.base import ParserError

    fake = types.ModuleType("pymupdf4llm")

    def to_markdown(path, page_chunks=False, **kw):  # noqa: ARG001
        if page_chunks:
            raise TypeError("unexpected keyword argument 'page_chunks'")
        return "第一页正文\n\n第二页正文\n\n第三页正文"  # 无 ----- 分隔线

    fake.to_markdown = to_markdown  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pymupdf4llm", fake)

    p = make_file("pdf", pages=3)
    with pytest.raises(ParserError, match="页边界"):
        PdfParser(cfg).parse(p)

    out = parse_with_fallback(p, ProbeResult(suggested_parser="pymupdf4llm"), cfg)
    assert out.degraded is True
    assert "parser_degraded:PdfPlainTextParser" in out.doc.quality.issues
    assert {b.page for b in out.doc.blocks if b.page} == {1, 2, 3}


def test_office_fallback_is_recorded(cfg, make_file, monkeypatch):
    """降级必须写进 issues——不记录的降级等于静默降质。"""
    import agentic_rag.ingestion.parsers.office as office

    monkeypatch.setattr(office, "_markitdown_md", lambda path: (_ for _ in ()).throw(RuntimeError("boom")))
    p = make_file("docx")
    out = parse_with_fallback(p, ProbeResult(suggested_parser="markitdown"), cfg)
    assert out.doc is not None
    assert "parser_fallback_native" in out.doc.quality.issues


def test_docling_missing_degrades_to_pdf(cfg, make_file):
    """docling 未安装时降级到 PdfParser，再降 PdfPlainTextParser。"""
    p = make_file("pdf")
    probe = ProbeResult(is_scanned=True)
    out = parse_with_fallback(p, probe, cfg)
    assert out.doc is not None
    assert out.doc.parser in ("pymupdf4llm", "pypdf_text")
    assert out.degraded is True
    assert any("parser_degraded" in i for i in out.doc.quality.issues)


def test_scanned_pdf_without_ocr_is_rejected(cfg, make_file):
    """扫描件 + 无 OCR → 全部解析器空手而归 → 拒绝入库（不是产出空文档）。"""
    p = make_file("scanned_pdf")
    out = parse_with_fallback(p, ProbeResult(is_scanned=True), cfg)
    # docling 未安装时：PdfParser 空 → PdfPlainTextParser 空 → 拒绝
    if out.doc is None:
        assert out.error
    else:
        assert out.doc.quality.issues or not out.doc.blocks


def test_pdf_plain_text_parser(cfg, make_file):
    p = make_file("pdf", pages=2)
    doc = PdfPlainTextParser(cfg).parse(p)
    assert doc.parser == "pypdf_text"
    assert len(doc.blocks) == 2
    assert doc.page_count == 2
