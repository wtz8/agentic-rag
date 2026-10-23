"""检索指标与显著性判据（04 §9.1 / §9.4）。

两个性质完全不同的东西放在同一个文件里，因为它们回答的是同一个问题的两半：
"差了多少"和"这个差值能不能信"。

- 上半：recall@K / MRR / NDCG@K / precision@K —— 纯函数，不依赖 LLM，毫秒级。
- 下半：I5 的显著性判据 —— 25 条样本上 1–2 条之差就是 4–8 个百分点，
  没有判据的话 A0–A4 的排序可能只是采样噪声。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "hit",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "PairedComparison",
    "compare_paired",
    "sign_test_p_value",
    "RepeatStats",
    "repeat_stats",
    "intervals_overlap",
    "VERDICT_DIFFERENT",
    "VERDICT_WEAK",
    "VERDICT_NOISE",
]


# ─────────────────────────── 检索指标 ───────────────────────────


def _top(retrieved: list[str], k: int | None) -> list[str]:
    if k is None:
        return list(retrieved)
    if k <= 0:
        return []
    return list(retrieved[:k])


def hit(retrieved: list[str], golden: list[str], k: int | None = None) -> int:
    """命中率：golden 是否出现在 Top-K 内（0/1）。

    与 recall@K 的区别：recall 看"召回了 golden 的几分之几"，hit 只看"有没有"。
    对 single_hop 这类只有一条 golden 的用例两者相同；对 multi_hop 不一样，
    而"至少找到一处"恰恰是决定答案能不能生成出来的那个量，所以单独留一个。
    """
    if not golden:
        return 0
    top = set(_top(retrieved, k))
    return int(bool(top & set(golden)))


def recall_at_k(retrieved: list[str], golden: list[str], k: int | None = None) -> float:
    """召回率：golden 中被召回到 Top-K 的比例。

    `golden` 为空时返回 0.0 而不是 1.0——`unanswerable` 用例的 golden 本来就是空的，
    给它 1.0 会把"拒答"伪装成"全召回"。这类用例由 runner 单独用 `abstain_rate` 统计。
    """
    if not golden:
        return 0.0
    top = set(_top(retrieved, k))
    return len(top & set(golden)) / len(set(golden))


def precision_at_k(retrieved: list[str], golden: list[str], k: int | None = None) -> float:
    """精确率：Top-K 里有多少是 golden。K 为 None 时按实际返回长度算。"""
    top = _top(retrieved, k)
    if not top:
        return 0.0
    return len(set(top) & set(golden)) / len(top)


def mrr(retrieved: list[str], golden: list[str]) -> float:
    """首个相关结果排名的倒数。全不命中为 0.0。"""
    g = set(golden)
    if not g:
        return 0.0
    for i, cid in enumerate(retrieved, start=1):
        if cid in g:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], golden: list[str], k: int | None = None) -> float:
    """NDCG@K：二值相关性（命中即 1），DCG 用 1/log2(i+1) 折扣。

    二值相关是本项目的实际情形——标注只说"答案在这块里"，不给分级相关性。
    强行编一个分级只会引入无法解释的自由度。
    """
    g = set(golden)
    if not g:
        return 0.0
    top = _top(retrieved, k)
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(top, start=1) if cid in g)
    ideal_n = min(len(g), len(top))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


# ─────────────────────────── 显著性判据（I5） ───────────────────────────

VERDICT_DIFFERENT = "different"
VERDICT_WEAK = "weak"
VERDICT_NOISE = "noise"

# 04 §9.4 定案的阈值
NET_WIN_DIFFERENT = 0.20
NET_WIN_NOISE = 0.10
TIE_EPSILON = 0.02  # 差值绝对值小于它记为 tie
P_THRESHOLD = 0.05


def sign_test_p_value(n_consistent: int, n_total: int) -> float:
    """双侧精确符号检验。

    `n_total` 含 tie（tie 既不算一致也不算相反）。这一点与教科书常见的
    "只在非 tie 样本上做检验"不同，但 04 §9.4 的临界值表（N=20→15、N=25→18）
    就是按含 tie 的 N 算出来的，实现必须与文档同源，否则报告里的判据与
    文档写的不是同一把尺子。
    """
    if n_total <= 0:
        return 1.0
    k = max(n_consistent, n_total - n_consistent)
    tail = sum(math.comb(n_total, i) for i in range(k, n_total + 1))
    return min(1.0, 2.0 * tail / (2**n_total))


@dataclass
class PairedComparison:
    """逐条配对比分的结果（04 §9.4 第一级，必报）。"""

    metric: str
    n: int
    win: int = 0
    loss: int = 0
    tie: int = 0
    net_win_rate: float = 0.0
    p_value: float = 1.0
    verdict: str = VERDICT_NOISE
    mean_delta: float = 0.0  # 均值差，只作参考：判据不看它

    @property
    def significant(self) -> bool:
        return self.verdict == VERDICT_DIFFERENT

    def as_text(self) -> str:
        """报告里"A 比 B 好"必须同时给出 net_win_rate 与 N——只报总分的结论不得进摘要。"""
        return (
            f"{self.metric}: net_win_rate={self.net_win_rate:+.3f} "
            f"(win/loss/tie={self.win}/{self.loss}/{self.tie}, N={self.n}, "
            f"p={self.p_value:.4f}) → {self.verdict}"
        )


def compare_paired(
    metric: str,
    scores_x: list[float],
    scores_y: list[float],
    *,
    tie_epsilon: float = TIE_EPSILON,
) -> PairedComparison:
    """逐条配对比分：X 相对 Y 是赢是输。

    用配对而不是比总分，是因为 25 条样本上总分会被 1–2 条极端值支配，
    而"在多少条题上赢"这个量既能做符号检验，又能直接写进报告让人看懂。
    """
    n = min(len(scores_x), len(scores_y))
    if n == 0:
        return PairedComparison(metric=metric, n=0, verdict=VERDICT_NOISE)
    win = loss = tie = 0
    delta_sum = 0.0
    for i in range(n):
        d = float(scores_x[i]) - float(scores_y[i])
        delta_sum += d
        if abs(d) < tie_epsilon:
            tie += 1
        elif d > 0:
            win += 1
        else:
            loss += 1
    net = (win - loss) / n
    p = sign_test_p_value(max(win, loss), n)
    if abs(net) >= NET_WIN_DIFFERENT and p < P_THRESHOLD:
        verdict = VERDICT_DIFFERENT
    elif abs(net) < NET_WIN_NOISE:
        verdict = VERDICT_NOISE
    else:
        verdict = VERDICT_WEAK
    return PairedComparison(
        metric=metric,
        n=n,
        win=win,
        loss=loss,
        tie=tie,
        net_win_rate=net,
        p_value=p,
        verdict=verdict,
        mean_delta=delta_sum / n,
    )


# ─────────────────────────── 重复跑方差（第二级） ───────────────────────────


@dataclass
class RepeatStats:
    """同配置重复跑的均值与离散度（04 §9.4 第二级，用于中间地带）。"""

    metric: str
    runs: list[float] = field(default_factory=list)
    mean: float = 0.0
    std: float = 0.0  # 样本标准差（n-1）；n=1 时为 0.0

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def lo(self) -> float:
        return self.mean - 2 * self.std

    @property
    def hi(self) -> float:
        return self.mean + 2 * self.std

    def as_text(self) -> str:
        return f"{self.metric}: {self.mean:.4f} ± {self.std:.4f} (n={self.n})"


def repeat_stats(metric: str, runs: list[float]) -> RepeatStats:
    vals = [float(v) for v in runs]
    n = len(vals)
    if n == 0:
        return RepeatStats(metric=metric)
    mean = sum(vals) / n
    if n == 1:
        return RepeatStats(metric=metric, runs=vals, mean=mean, std=0.0)
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return RepeatStats(metric=metric, runs=vals, mean=mean, std=math.sqrt(var))


def intervals_overlap(a: RepeatStats, b: RepeatStats) -> bool:
    """两组 `均值 ± 2σ` 区间是否重叠——重叠即判定不可区分。"""
    if a.n == 0 or b.n == 0:
        return True
    return not (a.hi < b.lo or b.hi < a.lo)


def judge_precision_sigma(sigma: float) -> float:
    """仪器精度门槛：任何小于 `2·σ_judge` 的组间差值直接判为噪声（04 §9.4 前置动作）。

    "先测仪器精度，再测样品"——不知道尺子精度就去量东西，量出来的数字没有意义。
    """
    return 2.0 * sigma
