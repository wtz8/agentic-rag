"""解析质量三道关卡（07 §3）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.ingestion.quality import (
    build_report,
    check_conservation,
    count_tables_detected,
    markdown_to_blocks,
    probe_before_parse,
    raw_baseline,
    validate_structure,
)
from agentic_rag.schema import Block, ParsedDocument, QualityReport


def _doc(blocks, source_path="x.pdf", parser="pymupdf4llm") -> ParsedDocument:
    return ParsedDocument(
        doc_id="d" * 16,
        source=Path(source_path).name,
        source_path=source_path,
        blocks=blocks,
        parser=parser,
        page_count=1,
    )


# ── raw 基线（03 §4.3.0）─────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["md", "txt", "html", "docx", "xlsx", "pptx", "pdf"])
def test_raw_baseline_non_empty_for_every_format(cfg, make_file, kind):
    p = make_file(kind)
    raw = raw_baseline(p, cfg)
    assert raw.strip(), f"{kind} 取不到 raw 基线，守恒检查会整项失效"


def test_raw_baseline_empty_for_scanned_pdf(cfg, make_file):
    """扫描件抽不出文本是**预期行为**：此时守恒检查必须声明不适用，而不是算出假分数。"""
    p = make_file("scanned_pdf")
    assert raw_baseline(p, cfg).strip() == ""


def test_conservation_not_applicable_when_raw_empty(cfg):
    doc = _doc([Block(type="paragraph", text="解析出来的正文 12345")])
    issues = check_conservation(doc, "", cfg.quality)
    assert doc.quality.digit_loss is None
    assert doc.quality.text_coverage is None
    assert "digit_loss" in doc.quality.not_applicable
    assert "text_coverage" in doc.quality.not_applicable
    assert issues == []  # 不适用不得变成"通过"也不得变成"失败"


def test_conservation_not_applicable_below_minimum(cfg):
    """raw 数字量 / 字符量低于下限时同样不适用。"""
    doc = _doc([Block(type="paragraph", text="只有一点点字")])
    raw = "同" * (cfg.quality.raw_char_min_for_check - 1)
    check_conservation(doc, raw, cfg.quality)
    assert doc.quality.digit_loss is None
    assert doc.quality.text_coverage is None


def test_conservation_computes_when_applicable(cfg):
    text = "解析结果保留了数字 12345 与 6789012345"
    doc = _doc([Block(type="paragraph", text=text)])
    raw = text + "原文多出来的一段文字，用于让覆盖率小于 1" * 30  # 超过 raw_char_min_for_check
    check_conservation(doc, raw, cfg.quality)
    assert doc.quality.digit_loss is not None
    assert doc.quality.text_coverage is not None
    assert 0 < doc.quality.text_coverage < 1
    assert "digit_loss" not in doc.quality.not_applicable
    assert "table_recall" in doc.quality.not_applicable  # detected=0 → 不适用


def test_conservation_appends_warn_issues(cfg):
    """warn 级检查项由 check_conservation 记入 issues，_verdict 据此判 warned。"""
    q = cfg.quality
    # raw 25 个数字、解析保留 21 个 → loss ≈ 0.16，落在 warn 与 reject 之间
    doc = _doc([Block(type="paragraph", text="数字 123456789012345678901")])
    raw = "数字 1234567890123456789012345"
    issues = check_conservation(doc, raw, q)
    assert "digit_loss_warn" in issues
    assert _verdict_of(doc.quality, cfg, ocr=True) == "warned"


def test_conservation_not_applicable_for_markdown(cfg):
    """md / txt 的 raw 与解析结果同源，守恒检查自动降级为不适用。"""
    doc = _doc([Block(type="paragraph", text="# 标题\n\n正文 12345")], source_path="a.md")
    raw = "# 标题\n\n正文 12345"
    check_conservation(doc, raw, cfg.quality)
    assert doc.quality.digit_loss is None and doc.quality.text_coverage is None
    assert "digit_loss" in doc.quality.not_applicable


# ── 结构校验 ─────────────────────────────────────────────────────────


def test_validate_structure_ok(cfg):
    doc = _doc(
        [
            Block(type="heading", level=1, text="一", path=["一"]),
            Block(type="paragraph", text="正文", path=["一"]),
            Block(type="heading", level=2, text="1.1", path=["一", "1.1"]),
            Block(type="table", text="| a | b |\n| --- | --- |\n| 1 | 2 |", path=["一", "1.1"]),
        ]
    )
    issues = validate_structure(doc, cfg.quality)
    assert issues == []
    assert doc.quality.heading_jumps == 0
    assert doc.quality.table_integrity == 1.0


def test_validate_structure_detects_table_mismatch(cfg):
    doc = _doc([Block(type="table", text="| a | b |\n| --- | --- |\n| 1 |")])
    issues = validate_structure(doc, cfg.quality)
    assert "table_row_col_mismatch" in issues
    assert doc.quality.table_integrity < 1.0


def test_validate_structure_detects_heading_jump(cfg):
    blocks = []
    for i in range(cfg.quality.heading_jump_warn + 2):
        blocks.append(Block(type="heading", level=1, text=f"{i}"))
        blocks.append(Block(type="heading", level=5, text=f"{i}.1"))  # 1 → 5 每次都算跳变
    doc = _doc(blocks)
    issues = validate_structure(doc, cfg.quality)
    assert "heading_level_jump" in issues
    assert doc.quality.heading_jumps > cfg.quality.heading_jump_warn


def test_validate_structure_detects_orphan_caption(cfg):
    doc = _doc([Block(type="image_caption", text="![图1](x.png)")])
    assert "orphan_image_caption" in validate_structure(doc, cfg.quality)


# ── 处置判定（03 §4 的 _verdict 四条）────────────────────────────────


def _verdict_of(report: QualityReport, cfg, ocr: bool) -> str:
    doc = _doc([Block(type="paragraph", text="x")])
    doc.quality = report
    return build_report(doc, "", cfg.quality, ocr).verdict


def test_scanned_without_ocr_rejected(cfg):
    r = QualityReport(is_scanned=True, char_count=5000)
    assert _verdict_of(r, cfg, ocr=False) == "rejected"
    assert "scanned_without_ocr" in r.issues


def test_scanned_with_ocr_but_too_small_rejected(cfg):
    r = QualityReport(is_scanned=True, char_count=cfg.quality.ocr_min_chars - 1)
    assert _verdict_of(r, cfg, ocr=True) == "rejected"
    assert "ocr_output_too_small" in r.issues


def test_scanned_with_ocr_enough_passes(cfg):
    r = QualityReport(is_scanned=True, char_count=cfg.quality.ocr_min_chars + 100)
    assert _verdict_of(r, cfg, ocr=True) == "ok"


def test_digit_loss_thresholds(cfg):
    q = cfg.quality
    mid = (q.digit_loss_warn + q.digit_loss_reject) / 2
    r = QualityReport(digit_loss=mid, issues=["digit_loss_warn"])
    assert _verdict_of(r, cfg, ocr=True) == "warned"
    r2 = QualityReport(digit_loss=q.digit_loss_reject + 0.01)
    assert _verdict_of(r2, cfg, ocr=True) == "rejected"
    r3 = QualityReport(digit_loss=q.digit_loss_warn - 0.01)
    assert _verdict_of(r3, cfg, ocr=True) == "ok"


def test_text_coverage_thresholds(cfg):
    q = cfg.quality
    r = QualityReport(text_coverage=(q.text_coverage_warn + q.text_coverage_reject) / 2, issues=["text_coverage_warn"])
    assert _verdict_of(r, cfg, ocr=True) == "warned"
    r2 = QualityReport(text_coverage=q.text_coverage_reject - 0.01)
    assert _verdict_of(r2, cfg, ocr=True) == "rejected"
    r3 = QualityReport(text_coverage=q.text_coverage_warn + 0.01)
    assert _verdict_of(r3, cfg, ocr=True) == "ok"


def test_none_metrics_never_reject(cfg):
    """不适用的检查项不得参与判定（守恒检查的 None 语义）。"""
    r = QualityReport(digit_loss=None, text_coverage=None)
    assert _verdict_of(r, cfg, ocr=True) == "ok"


def test_table_recall_warn_only(cfg):
    """表格还原度只到 warn，没有 reject 档（03 §4 的阈值表）。"""
    r = QualityReport(table_recall=cfg.quality.table_recall_warn - 0.01, issues=["table_recall_warn"])
    assert _verdict_of(r, cfg, ocr=True) == "warned"


# ── 关卡 1 探测 ──────────────────────────────────────────────────────


def test_probe_detects_scanned_pdf(cfg, make_file):
    p = make_file("scanned_pdf")
    probe = probe_before_parse(p, cfg)
    assert probe.is_scanned is True
    assert probe.chars_per_page < cfg.quality.scanned_chars_per_page


def test_probe_normal_pdf_not_scanned(cfg, make_file):
    p = make_file("pdf")
    probe = probe_before_parse(p, cfg)
    assert probe.is_scanned is False
    assert probe.ok is True
    assert probe.chars_per_page >= cfg.quality.scanned_chars_per_page


def test_probe_non_pdf_ok(cfg, make_file):
    probe = probe_before_parse(make_file("docx"), cfg)
    assert probe.ok and not probe.is_scanned
    assert probe.suggested_parser == "markitdown"


def test_probe_corrupted_file(cfg, tmp_path):
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"not a pdf at all")
    probe = probe_before_parse(p, cfg)
    assert probe.ok is False


def test_count_tables_detected(cfg, make_file):
    assert count_tables_detected(make_file("docx"), cfg) >= 1
    assert count_tables_detected(make_file("xlsx"), cfg) >= 1
    assert count_tables_detected(make_file("html"), cfg) >= 1


# ── Markdown → blocks ────────────────────────────────────────────────


def test_markdown_to_blocks_structure():
    md = "# 一\n\n段落一\n\n## 1.1\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n- 要点\n\n```\ncode\n```\n\n![图](x.png)\n"
    blocks = markdown_to_blocks(md)
    types = [b.type for b in blocks]
    assert "heading" in types and "table" in types and "list" in types
    assert "code" in types and "image_caption" in types
    table = [b for b in blocks if b.type == "table"][0]
    assert table.is_atomic is True
    assert table.path == ["一", "1.1"]
    assert [b for b in blocks if b.type == "heading"][1].level == 2


def test_rejection_records_its_own_reason(cfg):
    """守恒项拒答必须把拒因写进 issues。

    `pipeline` 用 issues[-1] 当拒答理由报给运维；不写的话报出去的是最后一个
    无关的轻微项（标题跳级、表格行列对不上…），真正的拒因反而消失。
    """
    doc = _doc([Block(type="paragraph", text="几乎没有数字的一段解析结果")])
    raw = "12345678901234567890 " * 60  # 数字几乎全丢 → 超 reject 阈值
    check_conservation(doc, raw, cfg.quality)
    assert doc.quality.digit_loss > cfg.quality.digit_loss_reject
    assert _verdict_of(doc.quality, cfg, ocr=True) == "rejected"
    assert any(i.startswith("digit_loss_reject:") for i in doc.quality.issues)
    assert doc.quality.issues[-1].startswith("digit_loss_reject:")
