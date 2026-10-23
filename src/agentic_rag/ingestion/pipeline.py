"""摄取阶段 A · 解析（03 §8）。

单篇：raw 基线 → 关卡1探测 → 路由/降级解析 → 关卡2结构校验 → 关卡3守恒检查 → 处置判定。
并发由调用方（CLI / `load_documents`）用线程池驱动——本阶段**无共享状态**，天然可并发。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config, project_path
from ..schema import Chunk, ParsedDocument, QualityReport
from .chunking import get_chunker, validate_chunks
from .quality import (
    build_report,
    check_conservation,
    count_tables_detected,
    probe_before_parse,
    raw_baseline,
    validate_structure,
)
from .router import parse_with_fallback

REJECTED_DIR = "data/rejected"


def ocr_available() -> bool:
    """扫描件的 OCR 通道是否可用（当前只有 docling 提供）。"""
    try:
        import docling  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


@dataclass
class DocResult:
    path: Path
    doc: ParsedDocument | None = None
    chunks: list[Chunk] = field(default_factory=list)
    report: QualityReport = field(default_factory=QualityReport)
    rejected: bool = False
    reason: str = ""
    issues: list[str] = field(default_factory=list)


def process_file(path: Path, cfg: Config, move_rejected: bool = True) -> DocResult:
    """单篇文档的完整摄取前处理（解析 + 质量三关）。"""
    path = Path(path)
    res = DocResult(path=path)

    raw = raw_baseline(path, cfg)
    probe = probe_before_parse(path, cfg)

    if not probe.ok:
        res.rejected = True
        res.reason = "encrypted" if probe.encrypted else "corrupted_or_empty"
        res.report = QualityReport(verdict="rejected", issues=[res.reason])
        _maybe_quarantine(path, cfg, move_rejected)
        return res

    if probe.suggested_parser == "":
        res.rejected = True
        res.reason = "unsupported_format"
        res.report = QualityReport(verdict="rejected", issues=["unsupported_format"])
        return res

    if probe.is_scanned and not ocr_available():
        # 关卡 1 的处置：扫描件 → OCR 通道。OCR 不可用就是不可摄取，
        # 不能让它混进"所有解析器都失败"——那样运维从报告里看不出该装 OCR 还是该修文件，
        # 而 build_report 里的 scanned_without_ocr 判定也会变成永远走不到的死代码。
        res.rejected = True
        res.reason = "scanned_without_ocr"
        res.report = QualityReport(verdict="rejected", issues=["scanned_without_ocr"], is_scanned=True)
        _maybe_quarantine(path, cfg, move_rejected)
        return res

    outcome = parse_with_fallback(path, probe, cfg)
    if outcome.doc is None:
        res.rejected = True
        res.reason = outcome.error or "all_parsers_failed"
        res.report = QualityReport(verdict="rejected", issues=["all_parsers_failed"])
        _maybe_quarantine(path, cfg, move_rejected)
        return res

    doc = outcome.doc
    doc.quality.table_count_detected = count_tables_detected(path, cfg)
    validate_structure(doc, cfg.quality)
    check_conservation(doc, raw, cfg.quality)
    build_report(doc, raw, cfg.quality, ocr_available(), probe)
    res.doc = doc
    res.report = doc.quality

    if doc.quality.verdict == "rejected":
        res.rejected = True
        res.reason = doc.quality.issues[-1] if doc.quality.issues else "quality_rejected"
        _maybe_quarantine(path, cfg, move_rejected)
        return res

    chunker = get_chunker(cfg.ingestion)
    res.chunks = chunker.chunk(doc)
    res.issues = validate_chunks(res.chunks, cfg.ingestion)
    return res


def _maybe_quarantine(path: Path, cfg: Config, move: bool) -> None:
    if not move or cfg.quality.on_reject != "quarantine":
        return
    try:
        dst_dir = project_path(REJECTED_DIR)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / Path(path).name
        if dst.exists():
            dst = dst_dir / f"{Path(path).stem}_{path.stat().st_size}{Path(path).suffix}"
        shutil.move(str(path), str(dst))
    except Exception:  # noqa: BLE001 - 移不动也要让摄取继续，报告里看得见
        pass


def iter_corpus(input_dir: Path, cfg: Config) -> list[Path]:
    """列出待摄取文件（按扩展名过滤，跳过已入库的辅助文件）。"""
    from .router import SUPPORTED_EXTS

    base = Path(input_dir)
    if base.is_file():
        return [base] if base.suffix.lower() in SUPPORTED_EXTS else []
    files = [p for p in sorted(base.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return [f for f in files if f.name not in ("MANIFEST.md",)]


def load_documents(
    input_dir: Path,
    cfg: Config,
    workers: int | None = None,
    move_rejected: bool = True,
) -> tuple[list[Chunk], list[DocResult]]:
    """阶段 A：并发解析 + 切分，返回（全部 chunk 平铺列表, 每篇结果）。"""
    from concurrent.futures import ThreadPoolExecutor

    files = iter_corpus(input_dir, cfg)
    n = max(1, workers or cfg.ingestion.workers)
    results: list[DocResult] = []
    if files:
        with ThreadPoolExecutor(max_workers=min(n, len(files))) as ex:
            results = list(ex.map(lambda p: process_file(p, cfg, move_rejected), files))

    all_chunks: list[Chunk] = []
    for r in results:
        all_chunks.extend(r.chunks)
    return all_chunks, results


__all__ = ["DocResult", "process_file", "load_documents", "iter_corpus", "ocr_available"]
