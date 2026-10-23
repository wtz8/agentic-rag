"""融合层：RRF 手算对照、跨轮合并、加权融合的归一化代价（04 §3 / L11）。"""

from __future__ import annotations

from agentic_rag.retrieval.fusion import (
    fuse,
    merge_candidates,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from agentic_rag.schema import Candidate


def _c(cid: str, fused: float = 0.0, **kw) -> Candidate:
    return Candidate(chunk_id=cid, fused_score=fused, **kw)


# ── RRF ──────────────────────────────────────────────────────────────


def test_rrf_hand_computed():
    """手算对抗：score(d) = Σ 1/(k+rank)，k=60。"""
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert dict(fused)["a"] == 1 / 61
    assert dict(fused)["b"] == 1 / 62 + 1 / 61
    assert dict(fused)["c"] == 1 / 62
    assert [cid for cid, _ in fused] == ["b", "a", "c"]


def test_rrf_crosses_routes_over_single_route_top1():
    """两路都命中的第 2 名 > 只在单路命中的第 1 名——RRF 的核心行为。"""
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert fused[0][0] == "b"  # b 在两路各排 2；a 只在第一路排 1
    assert fused[0][1] > fused[1][1]


def test_rrf_single_ranking_preserves_order():
    assert [cid for cid, _ in reciprocal_rank_fusion([["x", "y", "z"]])] == ["x", "y", "z"]


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_duplicate_id_in_one_route_accumulates():
    """同一路里重复出现只应累加一次排名贡献的实现是错的：重复意味着更强的证据。"""
    fused = dict(reciprocal_rank_fusion([["a", "a"]], k=60))
    assert fused["a"] == 1 / 61 + 1 / 62


def test_rrf_ties_break_by_id_for_determinism():
    """同分时排序必须稳定，否则消融实验的两次运行不可比。"""
    a = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    b = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    assert [cid for cid, _ in a] == [cid for cid, _ in b] == ["a", "b"]


# ── 加权融合 ──────────────────────────────────────────────────────────


def test_weighted_fusion_normalizes_weights():
    r1 = [("a", 10.0), ("b", 0.0)]
    r2 = [("a", 0.0), ("b", 10.0)]
    out = dict(weighted_score_fusion([r1, r2], [3.0, 1.0]))
    # 各路 min-max 归一化后：路1 a=1,b=0；路2 a=0,b=1；权重归一化为 0.75/0.25
    assert out["a"] == 0.75
    assert out["b"] == 0.25


def test_weighted_fusion_broadcasts_missing_weights():
    out = dict(weighted_score_fusion([[("a", 1.0), ("b", 0.0)], [("a", 5.0), ("b", 1.0)]], [1.0]))
    assert set(out) == {"a", "b"}


def test_weighted_fusion_is_sensitive_to_outlier():
    """这是默认选 RRF 的理由：min-max 让 100 与 1 的差距被压成 0/1 两极。"""
    out = dict(weighted_score_fusion([[("a", 100.0), ("b", 1.0), ("c", 0.5)]], [1.0]))
    assert out["a"] == 1.0 and out["c"] == 0.0
    # b 与 a 的绝对差是 c 与 b 的 50 倍，归一化后只差 0.0049
    assert out["b"] < 0.01


def test_weighted_fusion_empty():
    assert weighted_score_fusion([], []) == []


def test_fuse_dispatches_by_strategy():
    class C:
        fusion_strategy = "weighted"
        weighted_weights = (0.5, 0.5)
        rrf_k = 60

    out = fuse([[("a", 3.0), ("b", 1.0)]], C())
    assert out and out[0][0] == "a"  # 单路时加权融合退化为保序

    class R:
        fusion_strategy = "rrf"
        weighted_weights = (0.5, 0.5)
        rrf_k = 60

    assert [cid for cid, _ in fuse([["b", "a"]], R())] == ["b", "a"]


def test_fuse_accepts_id_only_rankings():
    class R:
        fusion_strategy = "rrf"
        rrf_k = 60
        weighted_weights = (1.0, 1.0)

    assert [cid for cid, _ in fuse([["a", "b"]], R())] == ["a", "b"]


# ── 跨轮合并（L11）────────────────────────────────────────────────────


def test_merge_without_prev_returns_cur():
    cur = [_c("a", 0.5), _c("b", 0.4)]
    out, merged = merge_candidates([], cur)
    assert [c.chunk_id for c in out] == ["a", "b"]
    assert merged == 0


def test_merge_keeps_prev_only_hits():
    """第二轮更差也不许把首轮的命中挤掉（单调不退化）。"""
    prev = [_c("p1", 0.9), _c("p2", 0.5)]
    cur = [_c("c1", 0.8)]
    out, merged = merge_candidates(prev, cur, k=60)
    ids = {c.chunk_id for c in out}
    assert {"p1", "p2", "c1"} <= ids
    assert merged == 2  # p1 / p2 来自上一轮


def test_merge_takes_higher_vector_score():
    """同一 chunk 取较高分：不能因为第二轮分数低就把首轮的高分候选判成不达标（K1）。"""
    prev = [_c("a", 0.9, vector_score=0.80)]
    cur = [_c("a", 0.2, vector_score=0.40)]
    out, _ = merge_candidates(prev, cur, k=60)
    assert out[0].chunk_id == "a"
    assert out[0].vector_score == 0.80


def test_merge_takes_higher_bm25_score():
    prev = [_c("a", 0.9, bm25_score=1.0)]
    cur = [_c("a", 0.2, bm25_score=5.0)]
    out, _ = merge_candidates(prev, cur, k=60)
    assert out[0].bm25_score == 5.0


def test_merge_unions_hit_routes_across_rounds():
    prev = [_c("a", 0.9)]
    prev[0].hit_routes.append("vector")
    cur = [_c("a", 0.2)]
    cur[0].hit_routes.append("bm25")
    out, _ = merge_candidates(prev, cur, k=60)
    assert sorted(out[0].hit_routes) == ["bm25", "vector"]


def test_merge_rewrites_fused_score_by_cross_round_rrf():
    prev = [_c("p", 0.9)]
    cur = [_c("p", 0.1), _c("c", 0.2)]  # 本轮 c 排第 1（按 fused_score 降序）
    out, _ = merge_candidates(prev, cur, k=60)
    scores = {c.chunk_id: c.fused_score for c in out}
    assert scores["p"] == 1 / 61 + 1 / 62  # 首轮第 1 + 本轮第 2
    assert scores["c"] == 1 / 61  # 只在本轮排第 1
    assert scores["p"] > scores["c"]  # 跨轮一致 > 单轮靠前


def test_weighted_route_with_single_hit_is_not_zeroed():
    """min-max 归一化下 `hi == lo` 时该路整体被清零 —— 单结果路由的隐性丢失。

    某一路只召回 1 条时（BM25 只命中一个词是很常见的），`span = (hi-lo) or 1.0`
    取到 1.0，而分子 `(s - lo)` 恒为 0 → 这一路的贡献是 0，等于把它参与的证据
    从融合里删掉了。B1 变体（加权 vs RRF）因此跑不出这部分的差异。
    修复后单结果按归一化上界 1.0 计入——它本就是该路的第一名。
    """
    rankings = [[("a", 0.83)], [("b", 0.90), ("a", 0.50), ("c", 0.10)]]
    out = dict(weighted_score_fusion(rankings, [0.5, 0.5]))
    # a = 0.5*1.0 + 0.5*0.5 = 0.75｜b = 0.5*1.0 = 0.5｜c = 0.5*0 = 0
    assert out["a"] > out["b"] > out["c"], "两路都命中的 a 必须排在第一路独占的 b 之前"
    assert out["a"] == 0.75


def test_merged_from_prev_counts_only_top_k_window():
    """口径必须是"上一轮进 Top-K 的条数"。

    写成"合并结果里有多少条来自上一轮"会恒等于 len(prev)——RRF 的输入就是
    [prev, cur] 两个排名，结果必然含 prev 全部 id。A4 消融拿常量量化不出任何东西。
    """
    prev = [_c(f"p{i}", 1.0 - i * 0.01) for i in range(6)]
    cur = [_c(f"c{i}", 0.99 - i * 0.01) for i in range(6)]
    _, full = merge_candidates(prev, cur, k=60)
    _, win3 = merge_candidates(prev, cur, k=60, top_k=3)
    _, win12 = merge_candidates(prev, cur, k=60, top_k=12)
    assert full == 6  # 未给 top_k 时按全量计（向后兼容）
    assert win12 == 6
    assert 0 < win3 < 6, "窗口收窄后上一轮不可能全部留下"
