"""混合检索（04 §4）：向量 ‖ BM25 并行召回 → 融合 → 候选。

- **两路必须并行**：串行会让召回耗时翻倍
- **单路失败不阻断**：向量库不可用仍返回 BM25 结果，反之亦然，并标记降级
- **过滤下推**：`filter_tables_only` 在向量路下推到 ANN 之前，BM25 路只能后置筛——
  两条路的过滤位置天然不同，差异写进 trace
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from ..config import Config
from ..index.store import IndexStore
from ..providers.base import EmbeddingProtocol
from ..schema import Candidate
from .fusion import fuse


class HybridRetriever:
    def __init__(self, cfg: Config, store: IndexStore, embedder: EmbeddingProtocol):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder

    # ── 单路 ──────────────────────────────────────────────

    def _vector_route(self, query: str, top_k: int, flt: dict | None = None) -> list[tuple[str, float]]:
        if not self.cfg.retrieval.use_vector:
            return []
        vec = self.embedder.embed([query])[0]
        return self.store.vector_store.search(vec, top_k, flt=flt)

    def _bm25_route(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self.cfg.retrieval.use_bm25 or not self.store.bm25_available:
            return []
        return self.store.bm25.search(query, top_k)

    # ── 单条查询的两路并行 ────────────────────────────────

    def _one_query(
        self,
        query: str,
        bm25_query: str | None,
        intent: str,
        diag: dict,
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        r = self.cfg.retrieval
        flt = {"block_type": "table"} if (r.filter_tables_only and intent == "table_numeric") else None
        b_query = bm25_query or query

        results: dict[str, list[tuple[str, float]]] = {"vector": [], "bm25": []}
        errors: dict[str, str] = {}

        def _run_vec():
            try:
                results["vector"] = self._vector_route(query, r.vector_top_k, flt)
            except Exception as e:  # noqa: BLE001
                errors["vector"] = str(e)

        def _run_bm25():
            try:
                results["bm25"] = self._bm25_route(b_query, r.bm25_top_k)
            except Exception as e:  # noqa: BLE001
                errors["bm25"] = str(e)

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as ex:
            list(ex.map(lambda f: f(), [_run_vec, _run_bm25]))
        diag.setdefault("durations", []).append(round((time.perf_counter() - t0) * 1000, 2))
        for k, v in errors.items():
            diag.setdefault("errors", {})[k] = v
        return results["vector"], results["bm25"]

    # ── 对外：多条查询 ────────────────────────────────────

    def retrieve(
        self,
        queries: list[str],
        intent: str = "simple",
        bm25_query: str | None = None,
        diag: dict | None = None,
    ) -> list[Candidate]:
        """外层并发（query × ）内层并发（两路）→ 融合 → Top-K 候选。"""
        diag = diag if diag is not None else {}
        r = self.cfg.retrieval
        queries = [q for q in (queries or []) if q and q.strip()]
        if not queries:
            return []

        pairs: list[tuple[list[tuple[str, float]], list[tuple[str, float]]]] = []
        if len(queries) == 1:
            pairs.append(self._one_query(queries[0], bm25_query, intent, diag))
        else:
            # 子查询之间必须并发，否则多跳延迟线性叠加。
            # 每条子查询用自己的 diag：并发下共享同一个 dict 会互相覆盖
            # （`errors`/`durations` 是 append/setdefault 语义）。跑完按序合并回来——
            # 早先这里传的是 `{}`，结果 durations 与 errors 全部被丢弃，
            # 多子查询（decompose/hyde）的向量路整体失败在 trace 里完全不可见。
            sub_diags: list[dict] = [{} for _ in queries]
            with ThreadPoolExecutor(max_workers=min(len(queries), 8)) as ex:
                pairs = list(
                    ex.map(
                        lambda iq: self._one_query(iq[1], bm25_query, intent, sub_diags[iq[0]]),
                        enumerate(queries),
                    )
                )
            _merge_diags(diag, sub_diags)

        per_query: list[list[tuple[str, float]]] = []
        for vec_hits, bm_hits in pairs:
            if vec_hits:
                per_query.append(vec_hits)
            if bm_hits:
                per_query.append(bm_hits)

        diag["vector_hits"] = sum(1 for v, _ in pairs if v)
        diag["bm25_hits"] = sum(1 for _, b in pairs if b)
        diag["routes"] = [
            name
            for name, ok in (("vector", any(v for v, _ in pairs)), ("bm25", any(b for _, b in pairs)))
            if ok
        ]
        diag["filter_pushdown"] = bool(r.filter_tables_only and intent == "table_numeric")
        diag["sub_query_count"] = len(queries)

        if not per_query:
            return []

        fused = fuse(per_query, r)[: r.fusion_top_k]
        by_id = {cid: score for cid, score in fused}

        # 表格加权：只对 table_numeric 生效
        boost = r.table_boost if intent == "table_numeric" else 1.0
        tables_only = r.filter_tables_only and intent == "table_numeric"

        indexed = {c.chunk_id: c for c in self.store.get_chunks(list(by_id))}
        candidates: list[Candidate] = []
        for cid, score in by_id.items():
            chunk = indexed.get(cid)
            if chunk is None:
                continue
            if tables_only and chunk.block_type != "table":
                continue  # BM25 路没有前置过滤，只能后置筛
            if boost != 1.0 and chunk.block_type == "table":
                score = score * boost
            candidates.append(Candidate(chunk_id=cid, fused_score=score))

        # 各路原始分数与排名（K1 的达标计数与 trace 都要用）
        _attach_route_scores(candidates, pairs)
        candidates.sort(key=lambda c: -c.fused_score)
        return candidates[: r.fusion_top_k]


def _attach_route_scores(candidates: list[Candidate], pairs) -> None:
    vmap: dict[str, tuple[float, int]] = {}
    bmap: dict[str, tuple[float, int]] = {}
    # 取优口径与跨轮合并一致：同路多次命中取**最高分**（L19 / K1）。
    # 若改成"取最好排名"，两处口径会分叉：跨轮取高分、跨子查询取好名次，
    # 同一 chunk 在两种场景下得到不同的 vector_score，达标计数因此不可复现。
    for vec_hits, bm_hits in pairs:
        for rank, (cid, score) in enumerate(vec_hits, start=1):
            prev = vmap.get(cid)
            if prev is None or float(score) > prev[0]:
                vmap[cid] = (float(score), rank)
        for rank, (cid, score) in enumerate(bm_hits, start=1):
            prev = bmap.get(cid)
            if prev is None or float(score) > prev[0]:
                bmap[cid] = (float(score), rank)
    for c in candidates:
        if c.chunk_id in vmap:
            c.vector_score, c.vector_rank = vmap[c.chunk_id]
            if "vector" not in c.hit_routes:
                c.hit_routes.append("vector")
        if c.chunk_id in bmap:
            c.bm25_score, c.bm25_rank = bmap[c.chunk_id]
            if "bm25" not in c.hit_routes:
                c.hit_routes.append("bm25")


def _merge_diags(dst: dict, subs: list[dict]) -> None:
    """把并发子查询的 diag 合并回外层（并发下不能共享同一个 dict）。

    - `durations`：逐条累加，顺序对应 `queries`，便于定位是哪条子查询慢
    - `errors`：按 `route@index` 记 key——并发下同名路由的错误会互相覆盖，
      丢掉"3 条子查询里只有 1 条向量路失败"这类信息（而它正是多跳归因要的）
    - `durations` 缺失时补 None，保证长度仍等于子查询条数
    """
    durations: list = []
    for i, d in enumerate(subs):
        ds = d.get("durations") or []
        durations.extend(ds if ds else [None])
        for k, v in (d.get("errors") or {}).items():
            dst.setdefault("errors", {})[f"{k}@{i}"] = v
    if durations:
        dst.setdefault("durations", []).extend(durations)


__all__ = ["HybridRetriever"]
