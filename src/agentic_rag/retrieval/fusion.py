"""融合（04 §3）：RRF 与加权融合。纯函数，便于单测。

**为什么默认 RRF**：向量相似度 ∈ [0,1]，BM25 分数无上界且随语料长度分布漂移。
直接加权相加等于让量纲大的一方支配排序；归一化又引入对异常值的敏感性。
RRF 只依赖排名，天然免疫量纲问题。

`weighted_score_fusion` 保留为对照实现，用于消融实验量化"RRF 比加权融合好多少"。
"""

from __future__ import annotations

from typing import Sequence

from ..schema import Candidate


def _rank(candidates: Sequence[Candidate]) -> list[str]:
    """按融合分数排序后的 chunk_id 列表（跨轮合并与单轮两路合并共用）。"""
    return [c.chunk_id for c in sorted(candidates, key=lambda c: -c.fused_score)]


def merge_candidates(
    prev: Sequence[Candidate],
    cur: Sequence[Candidate],
    k: int = 60,
    top_k: int | None = None,
) -> tuple[list[Candidate], int]:
    """跨轮 RRF 合并（L11）：第二轮只能往里加东西、把排名往上推，不会挤掉首轮的命中。

    同一 `chunk_id` 取两轮中**较高**的 `vector_score`——更乐观地估计其相关性，
    避免第二轮分数低就把首轮的高分候选判成不达标。

    `merged_from_prev` 的口径是**上一轮候选中进入最终 Top-K 的条数**（故 `top_k`
    必须传入）。不能写成"合并结果里有多少条来自上一轮"——RRF 的输入就是
    `[prev, cur]` 两个排名，合并结果必然包含 prev 的全部 id，那个计数恒等于
    `len(prev)`，A4 消融拿它量化不出任何东西。

    返回 (合并后的候选, 上一轮留进 Top-K 的条数)。
    """
    if not prev:
        return list(cur), 0
    prev_ids = {c.chunk_id for c in prev}
    by_id: dict[str, Candidate] = {}
    for c in cur:
        by_id[c.chunk_id] = c
    for c in prev:
        ex = by_id.get(c.chunk_id)
        if ex is None:
            by_id[c.chunk_id] = c
            continue
        if c.vector_score is not None and (ex.vector_score is None or c.vector_score > ex.vector_score):
            ex.vector_score, ex.vector_rank = c.vector_score, c.vector_rank
        if c.bm25_score is not None and (ex.bm25_score is None or c.bm25_score > ex.bm25_score):
            ex.bm25_score, ex.bm25_rank = c.bm25_score, c.bm25_rank
        for route in c.hit_routes:
            if route not in ex.hit_routes:
                ex.hit_routes.append(route)

    fused = reciprocal_rank_fusion([_rank(prev), _rank(cur)], k=k)
    out: list[Candidate] = []
    for cid, score in fused:
        c = by_id.get(cid)
        if c is None:
            continue
        c.fused_score = score
        out.append(c)
    # top_k 为 None 时按"全部保留"计（等价于旧口径，仅测试兼容）
    window = fused[:top_k] if top_k else fused
    merged_from_prev = sum(1 for cid, _ in window if cid in prev_ids)
    return out, merged_from_prev


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """`score(d) = Σ_i 1 / (k + rank_i(d))`。"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def weighted_score_fusion(
    rankings: Sequence[Sequence[tuple[str, float]]],
    weights: Sequence[float],
) -> list[tuple[str, float]]:
    """加权融合：先对每路分数做 min-max 归一化，再按权重相加。

    归一化是必要的（否则量纲大的一路支配排序），也是它的代价——
    对异常值敏感，这正是 RRF 默认优于它的原因。
    """
    if not rankings:
        return []
    weights = list(weights) or [1.0] * len(rankings)
    if len(weights) < len(rankings):
        weights += [weights[-1]] * (len(rankings) - len(weights))
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]  # 权重归一化

    scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        if not ranking:
            continue
        vals = [s for _, s in ranking]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        # 单元素路由（或整路分数全等）时 `hi == lo`，分子恒为 0 → 这一路的贡献被
        # 静默清零：明明是唯一命中的一路，融合结果里却看不见它。此时取 1.0
        # （它就是本路排第一的那个），而不是 0。
        flat = hi == lo
        for chunk_id, s in ranking:
            norm = 1.0 if flat else (s - lo) / span
            scores[chunk_id] = scores.get(chunk_id, 0.0) + w * norm
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def _as_ids(rankings) -> list[list[str]]:
    out: list[list[str]] = []
    for r in rankings:
        if r and isinstance(r[0], (tuple, list)):
            out.append([x[0] for x in r])
        else:
            out.append(list(r))
    return out


def fuse(rankings, cfg) -> list[tuple[str, float]]:
    """按 `cfg.fusion_strategy` 选实现。"""
    if getattr(cfg, "fusion_strategy", "rrf") == "weighted":
        as_scores = []
        for r in rankings:
            if r and isinstance(r[0], (tuple, list)):
                as_scores.append([(x[0], float(x[1])) for x in r])
            else:
                as_scores.append([(x, float(len(r) - i)) for i, x in enumerate(r)])
        return weighted_score_fusion(as_scores, list(cfg.weighted_weights))
    return reciprocal_rank_fusion(_as_ids(rankings), k=getattr(cfg, "rrf_k", 60))


__all__ = ["reciprocal_rank_fusion", "weighted_score_fusion", "fuse"]
