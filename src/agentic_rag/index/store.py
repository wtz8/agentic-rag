"""索引门面（03 §7.3）：全量构建 / 增量 upsert / 加载与版本校验 / 反查。

两条硬规则：
- **版本校验是硬性的**：换 embedding 或切分参数而不重建索引，会表现为"系统还能跑但效果变差"，
  这是最危险的一类故障，所以不匹配时直接抛错。
- **父块单独存储**：不进向量索引（否则索引体积翻倍），命中子块后按 `parent_id` 反查。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config, project_path
from ..providers.embedding import get_embedding
from ..schema import Chunk
from ..utils import read_jsonl, sha256_file, write_jsonl
from .bm25 import BM25Index, load_user_dict
from .vector_store import (
    IndexVersionMismatch,
    chunk_payload,
    get_vector_store,
    payload_to_chunk,
)


@dataclass
class BuildReport:
    """摄取链路对外唯一的结算凭证（CLI 打印、探针写入、集成测试断言）。

    字段必须是**可断言的具体数值**；`probe_hit_rate` 用派生属性而不是落盘字段，
    避免两处数据不一致。
    """

    docs_ok: int = 0
    docs_rejected: int = 0
    chunks_indexed: int = 0  # 检索单元数（is_parent=False）
    parents_written: int = 0  # 父块数（is_parent=True）
    warnings: list[str] = field(default_factory=list)
    probe_total: int = 0
    probe_hits: int = 0
    probe_failures: list[tuple[str, list[str]]] = field(default_factory=list)
    index_version: str = ""
    verified: bool = True
    partial_failure: bool = False
    elapsed_s: float = 0.0

    @property
    def probe_hit_rate(self) -> float:
        return self.probe_hits / self.probe_total if self.probe_total else 1.0


class IndexStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = project_path(cfg.index.path)
        self.vector_store = get_vector_store(cfg)
        self.bm25 = BM25Index(k1=cfg.index.bm25_k1, b=cfg.index.bm25_b)
        self.bm25_available = False
        # 加载期发现的非致命不一致（签名缺失、词典漂移…）。必须留痕：
        # 这类问题不会抛异常，只能靠"有人看到"才不会静默错下去。
        self.warnings: list[str] = []
        self._chunks: dict[str, Chunk] = {}
        self._parents: dict[str, Chunk] = {}
        self._loaded = False

    # ── 路径 ──────────────────────────────────────────────

    @property
    def chunks_path(self) -> Path:
        return self.dir / "chunks.jsonl"

    @property
    def parents_path(self) -> Path:
        return self.dir / "parents.jsonl"

    @property
    def bm25_path(self) -> Path:
        return self.dir / "bm25.pkl"

    # ── 构建 ──────────────────────────────────────────────

    def build(self, chunks: list[Chunk]) -> BuildReport:
        """全量构建。**入参是已切分的 chunk 平铺列表**（H5）。"""
        t0 = time.perf_counter()
        report = BuildReport(index_version=self.cfg.index_version)
        self.dir.mkdir(parents=True, exist_ok=True)

        parents = [c for c in chunks if c.is_parent]
        units = [c for c in chunks if not c.is_parent]

        # 物化视图：BM25 全量重建的数据源 / InMemory 降级时的 payload 来源 / 损坏时可重建的基线
        write_jsonl(self.chunks_path, [c.to_jsonl() for c in units], mode="w")
        write_jsonl(self.parents_path, [c.to_jsonl() for c in parents], mode="w")

        words = load_user_dict(self.cfg.index.user_dict_path)
        if words is None:
            report.warnings.append("user_dict_missing")  # H12：降级但不静默

        # 向量：批量 embedding（并发由 provider 内部做）→ 一次 upsert
        if units:
            try:
                embedder = get_embedding(self.cfg, workers=self.cfg.ingestion.workers)
                self.vector_store.create(embedder.dim, self.cfg.index_version)
                vectors = embedder.embed([c.text for c in units])
                self.vector_store.upsert(
                    [c.chunk_id for c in units], vectors, [chunk_payload(c) for c in units]
                )
            except Exception as e:  # noqa: BLE001 - 单批失败要标记，不能假装成功
                report.warnings.append(f"vector_upsert_failed:{e}")
                report.partial_failure = True

        # BM25：全量重建，全程只此一次
        try:
            self.bm25.build(
                units, user_dict=words, user_dict_path=self.cfg.index.user_dict_path
            )
            self.bm25.save(self.bm25_path)
            self.bm25_available = True
        except Exception as e:  # noqa: BLE001
            report.warnings.append(f"bm25_build_failed:{e}")
            self.bm25_available = False

        self._persist_if_memory()
        self.vector_store.save_meta(
            {
                "version": self.cfg.index_version,
                "dim": self.cfg.dashscope.embedding_dim,
                "embedding_model": self.cfg.dashscope.embedding_model,
                # 向量的实际来源。Mock 与真实 API 用同一个模型名、同一个维度，
                # 因此 index_version 完全相同——不单独记这一项，切回真实模式就会
                # 静默加载一批假向量：校验全部通过、检索全是噪声、一声不吭。
                "provider_mode": self.cfg.provider_mode,
                "chunk_strategy": self.cfg.ingestion.chunk_strategy,
                "built_at": time.time(),
                "chunks": len(units),
                "parents": len(parents),
                "retrieval_signature": self.cfg.retrieval_signature,
                "bm25_signature": self.cfg.bm25_signature,
                # 向量路失败过的索引必须自带标记。否则它和完整索引长得一模一样：
                # load() 校验通过、系统长期静默地只跑 BM25，而没人知道少了一路召回。
                "degraded": bool(report.partial_failure),
            }
        )

        self._chunks = {c.chunk_id: c for c in units}
        self._parents = {c.chunk_id: c for c in parents}
        self._loaded = True

        report.chunks_indexed = len(units)
        report.parents_written = len(parents)
        report.elapsed_s = round(time.perf_counter() - t0, 3)
        return report

    def upsert_documents(self, chunks: list[Chunk]) -> BuildReport:
        """增量：向量 upsert（增量）→ 追加物化视图 → BM25 **全量**重建。

        返回 `BuildReport`（与 `build()` 同类型），CLI 才能复用同一套结算打印。

        这里**不跑探针**：探针回答的是"整条链路通不通"，而增量只新增了少量文档，
        把早已通过的老语料再验一遍没有增量信息，却要付一遍全量探针的钱。
        版本校验由 `load()` 承担——增量写入前必须先确认它要和谁合并。
        """
        t0 = time.perf_counter()
        report = BuildReport(index_version=self.cfg.index_version)
        if not chunks:
            report.elapsed_s = 0.0
            return report

        self.load()
        parents = [c for c in chunks if c.is_parent]
        units = [c for c in chunks if not c.is_parent]
        # 按 chunk_id 去重：物化视图是 `mode="a"` 追加的，重复摄取同一文档会
        # 直接堆重复行（BM25 按全量视图重建，于是同一个块被索引两次、
        # IDF 与 docs 计数全错），而 meta 里的 chunks/parents 还是旧数字。
        units = [c for c in units if c.chunk_id not in self._chunks]
        parents = [c for c in parents if c.chunk_id not in self._parents]
        if not units and not parents:
            report.elapsed_s = round(time.perf_counter() - t0, 3)
            return report  # 全是重复，本次什么都没加
        write_jsonl(self.chunks_path, [c.to_jsonl() for c in units], mode="a")
        write_jsonl(self.parents_path, [c.to_jsonl() for c in parents], mode="a")
        if units:
            try:
                embedder = get_embedding(self.cfg, workers=self.cfg.ingestion.workers)
                vectors = embedder.embed([c.text for c in units])
                self.vector_store.upsert(
                    [c.chunk_id for c in units], vectors, [chunk_payload(c) for c in units]
                )
            except Exception as e:  # noqa: BLE001 - 单批失败要标记，不能假装成功
                report.warnings.append(f"vector_upsert_failed:{e}")
                report.partial_failure = True
        # 先并入内存视图，再重建 BM25 —— 否则重建用的还是旧的全量视图（缺本次增量）
        self._chunks.update({c.chunk_id: c for c in units})
        self._parents.update({c.chunk_id: c for c in parents})
        words = load_user_dict(self.cfg.index.user_dict_path)
        if words is None:
            report.warnings.append("user_dict_missing")  # H12：降级但不静默
        try:
            self.bm25.build(  # 全量视图，不是本次增量
                self.all_chunks(), user_dict=words, user_dict_path=self.cfg.index.user_dict_path
            )
            self.bm25.save(self.bm25_path)
            self.bm25_available = True
        except Exception as e:  # noqa: BLE001
            report.warnings.append(f"bm25_build_failed:{e}")
            self.bm25_available = False
        self._save_meta_counts(len(self._chunks), len(self._parents))
        self._persist_if_memory()

        report.chunks_indexed = len(units)
        report.parents_written = len(parents)
        report.elapsed_s = round(time.perf_counter() - t0, 3)
        return report

    def indexed_doc_ids(self) -> set[str]:
        """已入库文档的 `doc_id` 集合（`ingest --incremental` 判定"谁是新增"的依据）。

        读物化视图而不是 meta：meta 里只有计数，而 `chunks.jsonl` 每行都带 `doc_id`，
        且它与磁盘上的索引同源——meta 的计数一旦和视图不同步，靠 meta 判
        "是否已入库"会把新文档误判成旧的而整批跳过。
        """
        self.ensure_loaded()
        return {c.doc_id for c in self._chunks.values()} | {
            c.doc_id for c in self._parents.values()
        }

    def _save_meta_counts(self, chunks: int, parents: int) -> None:
        """回写计数：`stats()` 与物化视图必须对得上。

        增量写入不更新 meta 的话，`stats()["chunks"]` 会长期停在构建时的数字，
        而磁盘上的 chunks.jsonl 一直在长——两者不一致且没有任何报错。
        """
        meta = self.vector_store.load_meta() or {}
        meta.update(
            {
                "version": self.cfg.index_version,
                "chunks": chunks,
                "parents": parents,
                "retrieval_signature": self.cfg.retrieval_signature,
                "bm25_signature": self.cfg.bm25_signature,
            }
        )
        self.vector_store.save_meta(meta)

    # ── 加载 ──────────────────────────────────────────────

    def load(self) -> None:
        if self._loaded:
            return
        meta = self.vector_store.load_meta()
        if not meta:
            raise IndexVersionMismatch("索引不存在，请先构建：python -m agentic_rag.cli ingest")
        actual = meta.get("version")
        if actual != self.cfg.index_version:
            raise IndexVersionMismatch(
                f"索引版本不匹配（索引 {actual} vs 配置 {self.cfg.index_version}），"
                "请重建：python -m agentic_rag.cli ingest --rebuild"
            )
        # 只比 version 是不够的：version 由 embedding 模型 + 维度 + 切分参数算出，
        # 不含 BM25 打分参数 / 用户词典 / 融合与反思判据。改了这些却不重建，
        # 加载的是旧 BM25 与旧分词，检索结果错而系统一声不吭——正是 03 §7.3 要防的
        # 那类故障。meta 里没有该字段时按"旧索引"处理，只警告不拦（向前兼容）。
        self._check_signature(meta)
        self._check_provider_mode(meta)
        if meta.get("degraded"):
            self.warnings.append(
                "index_degraded：该索引构建时向量路失败（只有 BM25 生效）。"
                "检索会长期少一路召回，请重建：python -m agentic_rag.cli ingest --rebuild"
            )
        self.vector_store.create(int(meta.get("dim", self.cfg.dashscope.embedding_dim)), actual)

        self._chunks = {d["chunk_id"]: Chunk.from_jsonl(d) for d in read_jsonl(self.chunks_path)}
        self._parents = {d["chunk_id"]: Chunk.from_jsonl(d) for d in read_jsonl(self.parents_path)}
        if self.bm25_path.exists():
            try:
                self.bm25 = BM25Index.load(self.bm25_path)
                self.bm25_available = True
                self._check_user_dict_sha()
            except IndexVersionMismatch:
                raise
            except Exception:  # noqa: BLE001
                self.bm25_available = False
        else:
            self.bm25_available = False
        self._loaded = True
        self._persist_if_memory()

    def _check_signature(self, meta: dict) -> None:
        """加载时只校验**索引期**参数（`bm25_signature`）。

        校验整体 `retrieval_signature` 是错的：它含 `use_bm25` / `use_vector` /
        `fusion_strategy` / `min_candidates` 等查询期开关，改这些不需要重建索引，
        而消融变体改的恰恰就是它们——按整体签名拦会直接把变体挡在门外。
        """
        want = self.cfg.bm25_signature
        got = meta.get("bm25_signature")
        if not got:
            self.warnings.append(
                "index_meta_without_bm25_signature：该索引由旧版本构建，"
                "无法校验 BM25 打分参数与用户词典是否与当前配置一致。"
            )
            return
        if got != want:
            raise IndexVersionMismatch(
                f"索引与配置的 BM25 参数/用户词典不一致（索引 {got} vs 配置 {want}）："
                "这两项直接决定已落盘的 BM25 与分词结果，改了必须重建。"
                "请重建：python -m agentic_rag.cli ingest --rebuild"
            )

    def _check_provider_mode(self, meta: dict) -> None:
        """索引里的向量是哪个供给方算的，必须与当前模式一致。

        Mock 与真实 API 共用同一个 `embedding_model` 名与维度 → `index_version`
        一模一样 → 光比 version 拦不住。这是"两个模式都能跑、但只有一个是对的"
        那类故障：mock 建的索引拿真实查询去搜，余弦分照样算得出来，只是毫无意义。
        """
        want = self.cfg.provider_mode
        got = meta.get("provider_mode")
        if not got:
            self.warnings.append(
                "index_meta_without_provider_mode：该索引未记录向量供给方，"
                f"无法确认它是不是当前模式（{want}）下算出来的。"
            )
            return
        if got != want:
            raise IndexVersionMismatch(
                f"索引与配置的向量供给方不一致（索引由 {got} 模式构建，当前 {want}）："
                "两者的向量不在同一个语义空间，混用得到的余弦分没有意义。"
                "请重建：python -m agentic_rag.cli ingest --rebuild"
            )

    def _check_user_dict_sha(self) -> None:
        """词典指纹不一致 = 当前 BM25 是用另一份词典建的，分词口径已经变了。"""
        if self.bm25 is None or not self.bm25_available:
            return
        want = (
            sha256_file(project_path(self.cfg.index.user_dict_path))[:16]
            if self.cfg.index.user_dict_path
            and project_path(self.cfg.index.user_dict_path).exists()
            else "none"
        )
        got = getattr(self.bm25, "user_dict_sha", "none") or "none"
        if got != want:
            self.warnings.append(
                f"user_dict_mismatch：BM25 以词典 {got} 构建，当前配置为 {want}。"
                "重建索引后才会生效。"
            )

    def _persist_if_memory(self) -> None:
        persist = getattr(self.vector_store, "persist", None)
        if callable(persist):
            persist()

    # ── 反查 ──────────────────────────────────────────────

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def get_chunks(self, ids: list[str]) -> list[Chunk]:
        self.ensure_loaded()
        out: list[Chunk] = []
        pending = []
        for i in ids:
            if i in self._chunks:
                out.append(self._chunks[i])
            elif i in self._parents:
                out.append(self._parents[i])
            else:
                pending.append(i)
        if pending:
            payloads = getattr(self.vector_store, "get_payloads", None)
            if callable(payloads):
                for i, p in payloads(pending).items():
                    out.append(payload_to_chunk(p))
        return out

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        got = self.get_chunks([chunk_id])
        return got[0] if got else None

    def all_chunks(self) -> list[Chunk]:
        """全量检索单元视图：BM25 重建的数据源（H5）。"""
        self.ensure_loaded()
        if self._chunks:
            return [self._chunks[k] for k in sorted(self._chunks)]
        return [Chunk.from_jsonl(d) for d in read_jsonl(self.chunks_path)]

    def parent_of(self, child_id: str) -> Chunk | None:
        self.ensure_loaded()
        child = self._chunks.get(child_id)
        if child is None:
            got = self.get_chunks([child_id])
            child = got[0] if got else None
        if child is None or not child.parent_id:
            return None
        return self._parents.get(child.parent_id)

    def stats(self) -> dict:
        self.ensure_loaded()
        meta = self.vector_store.load_meta()
        sources = sorted({c.source for c in self._chunks.values()})
        return {
            "index_version": meta.get("version", ""),
            "config_version": self.cfg.index_version,
            "backend": self.cfg.index.backend,
            "chunks": len(self._chunks),
            "parents": len(self._parents),
            "vectors": self.vector_store.count(),
            "documents": len(sources),
            "sources": sources,  # 左栏文档列表与 `inspect` 都要用
            "bm25_available": self.bm25_available,
            "bm25": self.bm25.stats(),
            "chunk_strategy": meta.get("chunk_strategy", self.cfg.ingestion.chunk_strategy),
            "retrieval_signature": meta.get("retrieval_signature", ""),
            "config_retrieval_signature": self.cfg.retrieval_signature,
            "bm25_signature": meta.get("bm25_signature", ""),
            "config_bm25_signature": self.cfg.bm25_signature,
            "degraded": bool(meta.get("degraded", False)),
            "warnings": list(self.warnings),
        }


__all__ = ["IndexStore", "BuildReport", "IndexVersionMismatch"]
