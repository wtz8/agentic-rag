"""离线端到端：真实文件 → 解析三关 → 切分 → 索引 → 可检索（04 / 06 §6）。

这里不断言解析器的内部实现，只断言**结算凭证**：
进了多少篇、拒了多少篇、探针问题能否命中。数字对不上就说明链路断了。
"""

from __future__ import annotations

import shutil

import pytest

from agentic_rag.index.store import IndexStore, IndexVersionMismatch
from agentic_rag.ingestion import pipeline as pipe
from agentic_rag.ingestion.pipeline import load_documents, process_file
from agentic_rag.providers.embedding import MockEmbedding
from agentic_rag.retrieval.retriever import HybridRetriever
from agentic_rag.schema import Chunk


def test_ingest_corpus_produces_indexable_chunks(icfg, corpus_dir):
    chunks, results = load_documents(corpus_dir, icfg, move_rejected=False)
    assert chunks, "7 种格式里至少要有一批产出 chunk"
    ok = [r for r in results if not r.rejected]
    assert ok, f"全部被拒：{[r.reason for r in results]}"
    assert all(r.chunks for r in ok), "未被拒的文档必须有 chunk"
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunks_carry_provenance_and_typed_ids(icfg, corpus_dir):
    chunks, _ = load_documents(corpus_dir, icfg, move_rejected=False)
    for c in chunks:
        assert c.chunk_id and c.doc_id
        assert c.source, "溯源三要素缺一：来源"
        assert c.text and c.raw_text
    units = [c for c in chunks if not c.is_parent]
    parents = [c for c in chunks if c.is_parent]
    assert all(":c" in c.chunk_id for c in units)  # 分类型段（03 §0.1）
    assert all(":p" in c.chunk_id for c in parents)


def test_scanned_pdf_is_rejected_not_silently_indexed(icfg, make_file):
    """宁可拒绝入库，也不能带病入库。"""
    p = make_file("scanned_pdf", name="scanned.pdf")
    res = process_file(p, icfg, move_rejected=False)
    assert res.rejected is True
    assert "scanned" in res.reason or "scanned_without_ocr" in res.report.issues
    assert res.chunks == [], "扫描件不得产出 chunk"
    assert p.exists(), "move_rejected=False 时不该移动源文件"


def test_rejected_file_is_quarantined_when_enabled(icfg, make_file, tmp_path, monkeypatch):
    """on_reject=quarantine 时坏文件要挪走，否则每次摄取都会重复处理它。"""
    monkeypatch.setattr(pipe, "REJECTED_DIR", str(tmp_path / "rejected"))
    p = make_file("scanned_pdf", name="q.pdf")
    res = process_file(p, icfg, move_rejected=True)
    assert res.rejected is True
    assert not p.exists()
    assert (tmp_path / "rejected" / "q.pdf").exists()


def test_index_build_then_retrieval_hits(icfg, store_with_corpus):
    st = store_with_corpus
    stat = st.stats()
    assert stat["chunks"] > 0
    assert stat["bm25_available"] is True
    assert stat["index_version"] == icfg.index_version

    assert st.bm25.search("带宽", 5), "语料里明确有'带宽'，BM25 必须命中"

    r = HybridRetriever(icfg, st, MockEmbedding(dim=icfg.dashscope.embedding_dim))
    cands = r.retrieve(["华东区带宽采购单价"], intent="table_numeric")
    assert cands
    assert any("华东" in (st.get_chunk(c.chunk_id).raw_text or "") for c in cands)


def test_incremental_upsert_is_visible_to_bm25(icfg, store_with_corpus):
    """增量入库后必须能立刻被检索到——BM25 是否全量重建的可观测证据。"""
    st = store_with_corpus
    assert st.bm25.search("量子计算", 3) == []
    st.upsert_documents(
        [
            Chunk(
                chunk_id="zz:c0000",
                doc_id="zz",
                parent_id=None,
                text="量子计算原型机 2026 年完成整机联调。",
                raw_text="量子计算原型机 2026 年完成整机联调。",
                title_path=["前沿"],
                source="new.md",
                tokens=12,
            )
        ]
    )
    assert st.bm25.search("量子计算", 3), "增量后 BM25 未重建 → 新文档永远检索不到"


def test_cli_incremental_skips_already_indexed_docs(icfg, corpus_dir, monkeypatch, capsys):
    """`ingest --incremental` 必须有 CLI 入口 —— `upsert_documents` 不能只有测试在调。

    这条同时钉住两个失效模式：
    1. 重复增量 → 什么都不能加（物化视图是追加写的，不去重会同一个块索引两次）
    2. 真有新增 → 必须进索引且 BM25 立刻能检索到（增量后不重建 BM25，新文档永远查不到）
    """
    from agentic_rag import cli
    from agentic_rag.index.store import IndexStore

    monkeypatch.setattr(cli, "get_config", lambda: icfg)
    icfg.ingestion.run_probe_after_build = False
    base = ["ingest", "--input", str(corpus_dir), "--keep-rejected"]

    assert cli.main(base) == 0
    capsys.readouterr()
    before = len(IndexStore(icfg).all_chunks())
    assert before > 0

    assert cli.main([*base, "--incremental"]) == 0
    out = capsys.readouterr().out
    assert "没有新增文档" in out
    assert len(IndexStore(icfg).all_chunks()) == before, "重复增量不得追加任何块"

    (corpus_dir / "new.md").write_text(
        "# 新增章节\n\n量子计算原型机 2026 年完成整机联调。\n", encoding="utf-8"
    )
    assert cli.main([*base, "--incremental"]) == 0
    out = capsys.readouterr().out
    assert "1 篇新增" in out

    st = IndexStore(icfg)
    assert len(st.all_chunks()) > before
    assert st.bm25.search("量子计算", 3), "增量后 BM25 未全量重建 → 新文档永远检索不到"


def test_cli_incremental_falls_back_to_full_build(icfg, corpus_dir, monkeypatch, capsys):
    """索引不存在时，增量没有合并对象 —— 必须退回全量，而不是报错退出。"""
    from agentic_rag import cli
    from agentic_rag.index.store import IndexStore

    monkeypatch.setattr(cli, "get_config", lambda: icfg)
    icfg.ingestion.run_probe_after_build = False

    rc = cli.main(["ingest", "--input", str(corpus_dir), "--keep-rejected", "--incremental"])
    err = capsys.readouterr().err
    assert rc == 0, "没有索引时用 --incremental 应当退回全量构建而不是失败"
    assert "退回全量构建" in err
    assert len(IndexStore(icfg).all_chunks()) > 0


def test_version_mismatch_blocks_load(icfg, store_with_corpus):
    """换 embedding 维度而不重建，会表现为'能跑但效果差'——必须直接拒绝。"""
    cfg2 = icfg.clone()
    cfg2.dashscope.embedding_dim = 512
    with pytest.raises(IndexVersionMismatch):
        IndexStore(cfg2).load()


def test_doc_id_is_content_hash_not_path(icfg, tmp_path, make_file):
    """doc_id = sha1(内容)[:16]：同一内容换路径仍幂等。"""
    p = make_file("txt", name="dup.txt")
    r1 = process_file(p, icfg, move_rejected=False)
    moved = tmp_path / "moved.txt"
    shutil.copy(p, moved)
    r2 = process_file(moved, icfg, move_rejected=False)
    assert r1.doc.doc_id == r2.doc.doc_id
    assert len(r1.doc.doc_id) == 16
