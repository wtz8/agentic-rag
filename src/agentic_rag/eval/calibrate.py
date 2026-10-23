"""`sufficiency_vector_floor` 标定（K1 / 06 §7.2 P3 项）。

不标定 = 检索侧反思能力整体失效：阈值定高了每次查询都跑满 3 轮检索（费钱），
定低了永远一次通过（反思循环变成死代码）——两种失效都**不可见**。

方法：用 mini 集的锚点当"相关判定的真值"——命中的 golden 家族的候选为正样本、
其余为负样本，在候选的余弦分数分布上找 Youden's J（TPR−FPR）最大的切点。
不依赖人工逐条判定：锚点本来就是人工审过的。

硬前提：**只对 dashscope 模式标定**。Mock 向量是字符 3-gram 词袋的哈希投影，
余弦被块长稀释（同一查询在短文本上 0.41、在真实分块上 0.10），没有跨语料
量纲——拿它标出来的阈值换到真实模式就是一把量错尺寸的尺子。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from ..index.store import IndexStore
from ..providers.embedding import get_embedding
from .runner import EvalCase, family_map, resolve_anchors

__all__ = ["ScorePoint", "FloorCalibration", "collect_score_pairs", "suggest_floor", "calibrate_floor"]


@dataclass
class ScorePoint:
    vector_score: float
    relevant: bool
    question: str = ""
    chunk_id: str = ""


@dataclass
class FloorCalibration:
    """标定结果。`as_dict()` 直接进 EvalReport.calibration。"""

    pairs: int = 0
    relevant: int = 0
    irrelevant: int = 0
    floor: float | None = None  # Youden's J 最大化的切点
    tpr: float | None = None
    fpr: float | None = None
    auc: float | None = None
    relevant_p25: float | None = None  # 正样本分数的 25 分位（保守阈值参考）
    irrelevant_median: float | None = None
    basis: str = "golden_anchor_family"
    warning: str = ""

    def as_dict(self) -> dict:
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }


def collect_score_pairs(
    cases: list[EvalCase], cfg: Config, store: IndexStore, k: int = 50
) -> list[ScorePoint]:
    """对每条有锚点的用例做一次向量检索，收集 (score, 是否命中 golden 家族)。"""
    store.ensure_loaded()
    family = family_map(store)
    embedder = get_embedding(cfg, workers=1)
    points: list[ScorePoint] = []
    for case in cases:
        golden, _failed = resolve_anchors(case, store)
        if not golden:
            continue  # unanswerable / 锚点解析失败：没有真值，不进标定集
        g = set(golden)
        vec = embedder.embed([case.question])[0]
        for chunk_id, score in store.vector_store.search(vec, k):
            fam = family.get(chunk_id, chunk_id)
            points.append(
                ScorePoint(
                    vector_score=float(score),
                    relevant=fam in g,
                    question=case.question,
                    chunk_id=chunk_id,
                )
            )
    return points


def suggest_floor(points: list[ScorePoint]) -> FloorCalibration:
    """在分数分布上找 Youden's J 最大的切点。"""
    pos = sorted(p.vector_score for p in points if p.relevant)
    neg = sorted(p.vector_score for p in points if not p.relevant)
    cal = FloorCalibration(pairs=len(points), relevant=len(pos), irrelevant=len(neg))
    if not pos or not neg:
        cal.warning = (
            "正样本或负样本为空，无法标定——锚点全部命中或全部未命中 Top-K，"
            "此时任何阈值都无区分度（这正是 K1 判据的第二个必要条件）。"
        )
        return cal
    # 候选切点：全部唯一分数。J = TPR − FPR，扫一遍 O(n²) 在几千个点内可接受
    candidates = sorted({p.vector_score for p in points})
    best_j, best_cut, best_tpr, best_fpr = -1.0, None, None, None
    for cut in candidates:
        tp = sum(1 for s in pos if s >= cut)
        fp = sum(1 for s in neg if s >= cut)
        tpr, fpr = tp / len(pos), fp / len(neg)
        j = tpr - fpr
        if j > best_j:
            best_j, best_cut, best_tpr, best_fpr = j, cut, tpr, fpr
    cal.floor = round(best_cut, 4) if best_cut is not None else None
    cal.tpr = round(best_tpr, 4) if best_tpr is not None else None
    cal.fpr = round(best_fpr, 4) if best_fpr is not None else None
    cal.relevant_p25 = _quantile(pos, 0.25)
    cal.irrelevant_median = _quantile(neg, 0.50)
    cal.auc = round(_auc(pos, neg), 4)
    return cal


def calibrate_floor(cases: list[EvalCase], cfg: Config, store: IndexStore) -> FloorCalibration:
    """入口：mock 模式直接拒绝标定（见模块 docstring），dashscope 走完整流程。"""
    if cfg.provider_mode != "dashscope":
        return FloorCalibration(
            warning=(
                f"provider_mode={cfg.provider_mode}：Mock 向量没有跨语料量纲，"
                "不标定（04 §9.5 / K1）。无真实余弦分时 K1 判据降级为候选数判据，"
                "原因写进 trace。"
            )
        )
    return suggest_floor(collect_score_pairs(cases, cfg, store))


def _quantile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return round(sorted_vals[lo] * (1 - (pos - lo)) + sorted_vals[hi] * (pos - lo), 4)


def _auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney U / (|pos|·|neg|)：正样本分数高于负样本的概率。"""
    import bisect

    wins = 0
    for s in pos:
        # 严格大于的负样本数 + 相等的一半（并列记 0.5）
        lt = bisect.bisect_left(neg, s)
        eq = bisect.bisect_right(neg, s) - lt
        wins += lt + eq * 0.5
    return wins / (len(pos) * len(neg))
