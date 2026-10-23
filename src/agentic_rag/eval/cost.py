"""调用成本估算（离线预估用，不是计费依据）。

为什么要这个模块：一次真实 API 全量摄取约 578 万 token，是百炼免费额度（100 万）
的 5.8 倍——这个量级决定了它必须"先估再跑"。在 `--dry-run` 之前，
没有任何办法在动手前知道这一趟要花多少，只能跑完再看账单。
成本不可预知是工程缺陷，不是运气问题。

这里的数字全部是**估算**：token 量按字符数折算，费用按下方价目表换算。
真实消耗以 provider 回传的 `usage` 为准（`providers/base.record_usage` 的
`usage_total`），两者对不上的时候以实测为准并回来修折算系数。

价目表来源：阿里云百炼模型定价页，2026-10-22 查得。**价格会变**，改模型或
换地域时必须重新核对，别把它当常数。
"""

from __future__ import annotations

from dataclasses import dataclass

# ── 价目表（元 / 千 token）──────────────────────────────────────────────
# 只列已查实的。未查实的一律留 None 并在报告里显式写"未计入"，
# 不给一个看起来精确的错数字——那比不估更糟。

PRICE_EMBED_PER_1K = 0.0005  # text-embedding-v4，中国内地
PRICE_EMBED_BATCH_PER_1K = 0.00025  # 同一模型的 Batch（异步）接口，**半价**
PRICE_RERANK_PER_1K = 0.0006  # qwen3-rerank
# LLM 单价随型号/渠道变化较大，未查实 → 不估费用，只报 token 量。
PRICE_LLM_PER_1K: float | None = None

# 免费额度：百炼开通后 90 天内 100 万 token（text-embedding-v4）。
# 一次全量摄取是它的 5.8 倍——这个比例决定了摄取必须"先估量级再决定怎么跑"。
FREE_QUOTA_EMBED_TOKENS = 1_000_000

# 中文折算系数：Qwen tokenizer 下 1 个汉字约 0.6–1 token，保守取上界。
# 取 1.0（即"1 字符 ≈ 1 token"）会高估英文、低估中文；这里按 0.75 估，
# 并用区间展示，不假装是精确值。
CHARS_PER_TOKEN_LOW = 1.0  # 最贵（1 字符 1 token）
CHARS_PER_TOKEN_HIGH = 2.0  # 最便宜（2 字符 1 token）


@dataclass(frozen=True)
class CostEstimate:
    """一段区间的估算结果。区间不是偷懒——折算系数本身就是不确定的。"""

    label: str
    tokens_low: int
    tokens_high: int
    yuan_low: float
    yuan_high: float
    notes: list[str]

    @property
    def tokens(self) -> tuple[int, int]:
        return self.tokens_low, self.tokens_high

    def render(self) -> str:
        a, b = self.tokens_low, self.tokens_high
        if self.yuan_low == 0.0 and self.yuan_high == 0.0:
            money = "费用未计入（单价未查实）"
        else:
            money = f"≈ ¥{self.yuan_low:.2f} – ¥{self.yuan_high:.2f}"
        head = f"{self.label}: token {a:,} – {b:,}  {money}"
        return "\n".join([head, *(f"    · {n}" for n in self.notes)])


def _token_range(chars: int) -> tuple[int, int]:
    """字符数 → token 区间（下限=最省，上限=最贵）。"""
    return int(chars / CHARS_PER_TOKEN_HIGH), int(chars / CHARS_PER_TOKEN_LOW)


def estimate_ingest(
    units_chars: int,
    n_units: int,
    *,
    batch: bool = False,
) -> CostEstimate:
    """全量摄取（给所有检索单元算向量）的成本。

    `units_chars` 只算**检索单元**（`is_parent=False`）。父块不进向量路，
    所以父块的文本量不该计入——父块字符量是检索单元的 0.81 倍，把它算进来
    会让估算虚高近一倍。
    """
    lo, hi = _token_range(units_chars)
    price = PRICE_EMBED_BATCH_PER_1K if batch else PRICE_EMBED_PER_1K
    notes: list[str] = [
        f"{n_units:,} 个检索单元，共 {units_chars:,} 字符",
        f"单价 ¥{price}/千 token{'（Batch 接口，离线摄取适用）' if batch else ''}",
    ]
    if not batch:
        notes.append(
            f"改用 Batch 接口可省一半：≈ ¥{lo * PRICE_EMBED_BATCH_PER_1K / 1000:.2f} "
            f"– ¥{hi * PRICE_EMBED_BATCH_PER_1K / 1000:.2f}（摄取是非实时的，完全适用）"
        )
    if lo > FREE_QUOTA_EMBED_TOKENS:
        notes.append(
            f"⚠ 超过免费额度（{FREE_QUOTA_EMBED_TOKENS:,} token）约 "
            f"{lo / FREE_QUOTA_EMBED_TOKENS:.1f} 倍——建议改用 Batch 接口或分批摄取"
        )
    return CostEstimate(
        label="摄取（embedding）",
        tokens_low=lo,
        tokens_high=hi,
        yuan_low=lo * price / 1000,
        yuan_high=hi * price / 1000,
        notes=notes,
    )


def estimate_eval(
    n_cases: int,
    n_variants: int,
    *,
    rerank_top_k: int,
    avg_chunk_chars: int = 345,
    llm_calls_per_query: int = 5,
) -> CostEstimate:
    """评测的成本。

    查询侧的 embedding 只有一条问题文本，可忽略；rerank 只送 `rerank_top_k`
    条（默认 5），不是 `fusion_top_k`（50）——这两者常被混为一谈，
    混了会把 rerank 成本估高 10 倍。

    真正的大头是 LLM：单次查询 5 次调用（route + rerank + generate + verify×2）。
    但 LLM 单价未查实，这里**只报 token 量不报钱**。
    """
    queries = n_cases * n_variants
    rerank_chars = queries * rerank_top_k * avg_chunk_chars
    lo, hi = _token_range(rerank_chars)
    notes = [
        f"{n_cases} 条 × {n_variants} 个变体 = {queries} 次查询",
        f"rerank 只送 top-{rerank_top_k}（不是 fusion_top_k）",
        f"LLM 约 {queries * llm_calls_per_query:,} 次调用，token 量级 "
        f"{queries * llm_calls_per_query * 1500:,}",
    ]
    if PRICE_LLM_PER_1K is None:
        notes.append("⚠ LLM 单价未查实 → 费用**未计入**，本估算只覆盖 rerank")
    return CostEstimate(
        label="评测（rerank）",
        tokens_low=lo,
        tokens_high=hi,
        yuan_low=lo * PRICE_RERANK_PER_1K / 1000,
        yuan_high=hi * PRICE_RERANK_PER_1K / 1000,
        notes=notes,
    )


def render_report(estimates: list[CostEstimate], *, title: str) -> str:
    lines = [title, ""]
    lines += [e.render() for e in estimates]
    lines += [
        "",
        "注：token 按字符数折算（1 字符 ≈ 0.5–1 token），费用按 2026-10-22 百炼价目表换算。",
        "真实消耗以 provider 回传的 usage 为准；两者偏差大时回来修折算系数。",
    ]
    return "\n".join(lines)


__all__ = [
    "CostEstimate",
    "estimate_eval",
    "estimate_ingest",
    "render_report",
]
