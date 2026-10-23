"""切分：表格四档规则、父子映射、完整性检查（07 §3）。"""

from __future__ import annotations

import pytest

from agentic_rag.ingestion.chunking import get_chunker, validate_chunks, with_header
from agentic_rag.ingestion.chunking.base import pack_pieces, split_table_text, split_text
from agentic_rag.schema import Block, Chunk, ParsedDocument
from agentic_rag.utils import count_tokens


def _table(rows: int, cell: str = "数据") -> str:
    head = "| 列一 | 列二 | 列三 |\n| --- | --- | --- |"
    body = "\n".join(f"| {cell}{i} | {cell}{i} | {cell}{i} |" for i in range(rows))
    return head + "\n" + body


def _table_with_tokens(target: int) -> str:
    """生成 token 数达到 target 的表格（用于验证四档边界）。"""
    rows = 1
    while True:
        t = _table(rows)
        if count_tokens(t) >= target:
            return t
        rows += 1
        if rows > 5000:
            return t


def _doc(blocks) -> ParsedDocument:
    return ParsedDocument(
        doc_id="a" * 16,
        source="t.pdf",
        source_path="t.pdf",
        blocks=blocks,
        parser="test",
        page_count=1,
    )


# ── 上下文增强块头 ────────────────────────────────────────────────────


def test_with_header_format():
    c = Chunk(chunk_id="x:c0000", doc_id="x", parent_id=None, raw_text="正文", title_path=["一", "1.1"], source="a.pdf")
    assert with_header(c) == "[a.pdf > 一 > 1.1]\n正文"


def test_with_header_empty_path():
    c = Chunk(chunk_id="x:c0000", doc_id="x", parent_id=None, raw_text="正文", source="a.pdf")
    assert with_header(c) == "[a.pdf]\n正文"


# ── 文本切分 ──────────────────────────────────────────────────────────


def test_split_text_respects_limit(cfg):
    text = "。".join(f"第{i}句话内容" for i in range(200))
    pieces = split_text(text, cfg.ingestion.child_chunk_tokens)
    assert len(pieces) > 1
    assert all(count_tokens(p) <= cfg.ingestion.child_chunk_tokens for p in pieces)


def test_split_text_overlap(cfg):
    text = "。".join(f"第{i}句话内容" for i in range(200))
    a = split_text(text, cfg.ingestion.child_chunk_tokens, cfg.ingestion.child_overlap_tokens)
    b = split_text(text, cfg.ingestion.child_chunk_tokens, 0)
    assert sum(len(x) for x in a) > sum(len(x) for x in b), "有重叠时总文本量必须更大"
    joined_ok = all(count_tokens(p) <= cfg.ingestion.child_chunk_tokens for p in a)
    assert joined_ok


def test_pack_pieces_merges_small():
    out = pack_pieces(["短一", "短二", "短三"], 256)
    assert out == ["短一\n短二\n短三"]


# ── 表格四档规则（03 §6.4）────────────────────────────────────────────


def test_table_tier1_small(cfg):
    t = _table_with_tokens(cfg.ingestion.child_chunk_tokens - 20)
    out = split_table_text(t, cfg.ingestion)
    assert len(out) == 1 and out[0][1] is True


def test_table_tier2_oversized_but_atomic(cfg):
    """256–1024 之间的表格：原子性优先，豁免子块上限。"""
    t = _table_with_tokens(cfg.ingestion.child_chunk_tokens + 50)
    assert count_tokens(t) <= cfg.ingestion.table_split_threshold_tokens
    out = split_table_text(t, cfg.ingestion)
    assert len(out) == 1 and out[0][1] is True


def test_table_tier3_row_groups_with_header(cfg):
    t = _table_with_tokens(cfg.ingestion.table_split_threshold_tokens * 2 + 200)
    out = split_table_text(t, cfg.ingestion)
    assert len(out) > 1
    assert all(atomic for _, atomic in out)
    head_line = t.splitlines()[0]
    for text, _ in out:
        assert text.splitlines()[0] == head_line, "每组必须重复表头，否则列语义丢失"
        assert count_tokens(text) <= cfg.ingestion.table_split_threshold_tokens + 50


def test_table_tier4_gives_up_atomicity(cfg):
    """单行就超过 atomic_max_tokens：放弃原子性，按字符切并记 issue。"""
    cell = "很宽的列内容"
    repeat = 3
    while count_tokens("| " + " | ".join([cell * repeat] * 3) + " |") <= cfg.ingestion.atomic_max_tokens:
        repeat *= 2
    wide_row = "| " + " | ".join([cell * repeat] * 3) + " |"
    t = "| 列一 | 列二 | 列三 |\n| --- | --- | --- |\n" + wide_row + "\n" + wide_row
    out = split_table_text(t, cfg.ingestion)
    assert out and all(not atomic for _, atomic in out)


# ── 父子切分 ──────────────────────────────────────────────────────────


def test_parent_child_structure(cfg):
    blocks = [
        Block(type="heading", level=1, text="第一章", path=["第一章"]),
        Block(type="paragraph", text="段落内容。" * 200, path=["第一章"]),
        Block(type="heading", level=2, text="1.1", path=["第一章", "1.1"]),
        Block(type="paragraph", text="另一段内容。" * 200, path=["第一章", "1.1"]),
        Block(type="table", text=_table(4), path=["第一章", "1.1"], is_atomic=True),
    ]
    chunks = get_chunker(cfg.ingestion).chunk(_doc(blocks))
    parents = [c for c in chunks if c.is_parent]
    children = [c for c in chunks if not c.is_parent]
    assert parents and children
    assert all(c.parent_id for c in children), "每个子块都必须有 parent_id"
    parent_ids = {p.chunk_id for p in parents}
    assert all(c.parent_id in parent_ids for c in children)
    assert all(p.parent_id is None for p in parents)
    # ID 规则：p{seq:03d} / c{seq:04d}，父子不共用 seq 空间
    assert all(c.chunk_id.startswith("a" * 16 + ":p") for c in parents)
    assert all(c.chunk_id.startswith("a" * 16 + ":c") for c in children)
    # 平铺列表顺序：父块在其子块之前
    first_child_idx = min(i for i, c in enumerate(chunks) if not c.is_parent)
    assert any(c.is_parent for c in chunks[:first_child_idx])


def test_child_tokens_within_limit(cfg):
    blocks = [Block(type="heading", level=1, text="章", path=["章"])] + [
        Block(type="paragraph", text=f"第{i}段内容。" * 30, path=["章"]) for i in range(10)
    ]
    chunks = get_chunker(cfg.ingestion).chunk(_doc(blocks))
    for c in chunks:
        if c.is_parent:
            assert c.tokens <= cfg.ingestion.parent_chunk_tokens + 200
        elif not c.is_atomic:
            assert c.tokens <= cfg.ingestion.child_chunk_tokens + 50


def test_table_becomes_atomic_child(cfg):
    doc = _doc([Block(type="table", text=_table(6), path=["章"], is_atomic=True, page=3)])
    chunks = get_chunker(cfg.ingestion).chunk(doc)
    tables = [c for c in chunks if c.block_type == "table"]
    assert tables and all(t.is_atomic for t in tables)
    assert tables[0].page == 3


def test_structural_has_no_parents(cfg):
    cfg.ingestion.chunk_strategy = "structural"
    blocks = [
        Block(type="heading", level=1, text="章", path=["章"]),
        Block(type="paragraph", text="内容。" * 300, path=["章"]),
    ]
    chunks = get_chunker(cfg.ingestion).chunk(_doc(blocks))
    assert chunks and all(not c.is_parent and c.parent_id is None for c in chunks)
    assert validate_chunks(chunks, cfg.ingestion) == []


def test_chunker_factory(cfg):
    from agentic_rag.ingestion.chunking import ParentChildChunker, StructuralChunker

    cfg.ingestion.chunk_strategy = "parent_child"
    assert isinstance(get_chunker(cfg.ingestion), ParentChildChunker)
    cfg.ingestion.chunk_strategy = "structural"
    assert isinstance(get_chunker(cfg.ingestion), StructuralChunker)


# ── 完整性检查 ────────────────────────────────────────────────────────


def _child(cid, parent_id=None, tokens=10, raw_text="正文", doc_id="a" * 16, **kw) -> Chunk:
    c = Chunk(
        chunk_id=cid,
        doc_id=doc_id,
        parent_id=parent_id,
        raw_text=raw_text,
        title_path=["章"],
        source="t.pdf",
        tokens=tokens,
        **kw,
    )
    c.text = with_header(c)
    return c


def test_validate_clean(cfg):
    parent = _child("a" * 16 + ":p000", tokens=500, is_parent=True)
    child = _child("a" * 16 + ":c0000", parent_id=parent.chunk_id, tokens=100)
    assert validate_chunks([parent, child], cfg.ingestion) == []


def test_validate_catches_missing_parent(cfg):
    assert any("child_missing_parent_id" in i for i in validate_chunks([_child("a" * 16 + ":c0000")], cfg.ingestion))


def test_validate_catches_dangling_parent(cfg):
    issues = validate_chunks([_child("a" * 16 + ":c0000", parent_id="a" * 16 + ":p999")], cfg.ingestion)
    assert any("parent_id_not_found" in i for i in issues)


def test_validate_catches_cross_doc_parent(cfg):
    parent = _child("b" * 16 + ":p000", tokens=500, is_parent=True, doc_id="b" * 16)
    child = _child("a" * 16 + ":c0000", parent_id=parent.chunk_id, doc_id="a" * 16)
    issues = validate_chunks([parent, child], cfg.ingestion)
    assert any("parent_id_cross_doc" in i for i in issues)


def test_validate_catches_parent_with_parent_id(cfg):
    bad = _child("a" * 16 + ":p000", parent_id="a" * 16 + ":p001", is_parent=True, tokens=10)
    assert any("parent_must_have_no_parent_id" in i for i in validate_chunks([bad], cfg.ingestion))


def test_validate_catches_token_range(cfg):
    big = _child("a" * 16 + ":c0000", parent_id=None, tokens=cfg.ingestion.child_chunk_tokens + 100)
    assert any("chunk_tokens_out_of_range" in i for i in validate_chunks([big], cfg.ingestion))
    atomic_ok = _child("a" * 16 + ":c0001", tokens=cfg.ingestion.child_chunk_tokens + 100, is_atomic=True)
    assert not any("chunk_tokens_out_of_range" in i for i in validate_chunks([atomic_ok], cfg.ingestion))


def test_validate_catches_table_row_mismatch(cfg):
    bad = _child(
        "a" * 16 + ":c0000",
        block_type="table",
        is_atomic=True,
        raw_text="| a | b |\n| --- | --- |\n| 1 |",
    )
    bad.text = with_header(bad)
    assert any("table_row_col_mismatch" in i for i in validate_chunks([bad], cfg.ingestion))


def test_validate_catches_missing_header(cfg):
    c = _child("a" * 16 + ":c0000")
    c.text = "没有块头的裸文本"
    assert any("missing_context_header" in i for i in validate_chunks([c], cfg.ingestion))


def test_split_text_hard_splits_text_without_sentence_boundaries(cfg):
    """无句读文本必须被切开。

    真实故障：语料里有整段 api.py 源码（56k 字符、没有句号），段落级和句子级
    都切不动，原样漏出一个巨块 → 超过 embedding 模型 33000 字符的输入上限 →
    该块拿不到向量，还会连累同批次一起 400。docstring 承诺的"字符级"以前没实现。
    """
    code = "x = 1\n" * 4000  # 无任何句读与空行
    pieces = split_text(code, cfg.ingestion.child_chunk_tokens)
    assert len(pieces) > 1, "无句读文本被整个漏出"
    assert all(count_tokens(p) <= cfg.ingestion.child_chunk_tokens for p in pieces)


def test_hard_split_never_exceeds_budget_and_covers_text(cfg):
    from agentic_rag.ingestion.chunking.base import _hard_split

    text = "abcdefghij" * 500
    out = _hard_split(text, cfg.ingestion.child_chunk_tokens, 0)
    assert len(out) > 1
    assert all(count_tokens(p) <= cfg.ingestion.child_chunk_tokens for p in out)
    assert "".join(out) == text, "硬切不得丢内容"


def test_hard_split_with_overlap_terminates(cfg):
    """overlap 吃掉整个窗口时会死循环，必须有兜底。"""
    from agentic_rag.ingestion.chunking.base import _hard_split

    out = _hard_split("y" * 3000, cfg.ingestion.child_chunk_tokens, cfg.ingestion.child_overlap_tokens)
    assert len(out) > 1
    assert all(count_tokens(p) <= cfg.ingestion.child_chunk_tokens for p in out)
