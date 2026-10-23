"""切分后完整性检查（03 §6.5）。

返回 issue 列表，**不抛异常中断整个摄取批次**——单文档问题不应拖垮全量构建。
"""

from __future__ import annotations

from ...config import IngestionConfig
from ...schema import Chunk


def validate_chunks(chunks: list[Chunk], cfg: IngestionConfig) -> list[str]:
    issues: list[str] = []
    if not chunks:
        return ["empty_chunks"]

    by_id = {c.chunk_id: c for c in chunks}
    parents = {c.chunk_id for c in chunks if c.is_parent}
    docs = {c.doc_id for c in chunks}

    for c in chunks:
        if not c.chunk_id or not c.doc_id:
            issues.append(f"{c.chunk_id}: missing_id")
        if c.is_parent:
            if c.parent_id is not None:
                issues.append(f"{c.chunk_id}: parent_must_have_no_parent_id")
            if not (1 <= c.tokens <= cfg.parent_chunk_tokens):
                issues.append(f"{c.chunk_id}: parent_tokens_out_of_range({c.tokens})")
        else:
            if cfg.chunk_strategy == "parent_child":
                if not c.parent_id:
                    issues.append(f"{c.chunk_id}: child_missing_parent_id")
                elif c.parent_id not in parents:
                    issues.append(f"{c.chunk_id}: parent_id_not_found({c.parent_id})")
                elif by_id.get(c.parent_id) and by_id[c.parent_id].doc_id != c.doc_id:
                    issues.append(f"{c.chunk_id}: parent_id_cross_doc")
            if c.is_atomic:
                if not (1 <= c.tokens <= cfg.atomic_max_tokens):
                    issues.append(f"{c.chunk_id}: atomic_tokens_out_of_range({c.tokens})")
            else:
                if not (1 <= c.tokens <= cfg.child_chunk_tokens):
                    issues.append(f"{c.chunk_id}: chunk_tokens_out_of_range({c.tokens})")
        if not c.title_path and not c.source:
            issues.append(f"{c.chunk_id}: no_title_path_and_no_source")
        if not c.raw_text.strip():
            issues.append(f"{c.chunk_id}: empty_text")
        if not c.text.startswith("[") or "\n" not in c.text:
            issues.append(f"{c.chunk_id}: missing_context_header")

    # 表格原子块的行完整性：每行的单元格数 == 表头列数
    for c in chunks:
        if c.block_type == "table" and c.is_atomic:
            rows = [l for l in c.raw_text.splitlines() if l.strip().startswith("|")]
            rows = [l for l in rows if not set(l.strip().replace("|", "").strip()) <= set("-: ")]
            if not rows:
                issues.append(f"{c.chunk_id}: table_no_rows")
                continue
            ncols = len(rows[0].strip().strip("|").split("|"))
            if any(len(r.strip().strip("|").split("|")) != ncols for r in rows):
                issues.append(f"{c.chunk_id}: table_row_col_mismatch")

    # parent_child 策略下不允许出现"有父块标记但文档内没有父块"的孤儿
    if cfg.chunk_strategy == "parent_child" and not parents and docs:
        issues.append("no_parent_chunks_under_parent_child_strategy")

    return issues


__all__ = ["validate_chunks"]
