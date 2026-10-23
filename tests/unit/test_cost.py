"""成本估算的口径测试。

估算错了比不估更危险：它会让人以为"只花 3 块钱"然后放心地跑两次。
这里守的是两个最容易错的口径。
"""

from agentic_rag.eval.cost import (
    FREE_QUOTA_EMBED_TOKENS,
    PRICE_EMBED_BATCH_PER_1K,
    PRICE_EMBED_PER_1K,
    estimate_eval,
    estimate_ingest,
)


def test_ingest_estimate_is_based_on_units_not_parents():
    """只算检索单元。父块不进向量路，算进来会让估算虚高近一倍。

    真实比例：父块字符量是检索单元的 0.81 倍。这条不是洁癖，
    是"要不要重建索引"这个决定的输入。
    """
    units_chars = 1_000_000
    e = estimate_ingest(units_chars, n_units=3000)
    # 1M 字符 → 50 万–100 万 token
    assert 400_000 < e.tokens_low <= e.tokens_high <= 1_100_000
    assert e.yuan_low == e.tokens_low * PRICE_EMBED_PER_1K / 1000


def test_batch_halves_the_price():
    """离线摄取是非实时的，Batch 接口半价——建议必须真的更便宜。"""
    a = estimate_ingest(1_000_000, 3000)
    b = estimate_ingest(1_000_000, 3000, batch=True)
    assert b.yuan_low == a.yuan_low * (
        PRICE_EMBED_BATCH_PER_1K / PRICE_EMBED_PER_1K
    ), "Batch 应当是标准价的一半"


def test_estimate_warns_when_exceeding_free_quota():
    """一次全量摄取是免费额度的数倍——这就是余额快速见底的原因，必须说出口。"""
    small = estimate_ingest(100_000, 300)
    big = estimate_ingest(11_558_196, 33_477)  # 本项目实测规模
    assert not any("免费额度" in n for n in small.notes)
    assert any("免费额度" in n for n in big.notes)
    assert big.tokens_low > FREE_QUOTA_EMBED_TOKENS


def test_eval_rerank_uses_rerank_top_k_not_fusion_top_k():
    """rerank 只送 `rerank_top_k` 条（默认 5），不是 `fusion_top_k`（50）。

    混了会把 rerank 成本估高一个数量级——然后据此做出"别用 rerank"的错误决定。
    """
    a = estimate_eval(78, 5, rerank_top_k=5)
    b = estimate_eval(78, 5, rerank_top_k=50)
    assert b.tokens_low == a.tokens_low * 10


def test_eval_estimate_says_llm_cost_is_missing():
    """LLM 单价未查实就别报一个假的钱数，只报 token 量。"""
    e = estimate_eval(78, 5, rerank_top_k=5)
    assert any("未计入" in n for n in e.notes), "不说明缺口会让人以为 Total 就是全部花费"
    assert e.label == "评测（rerank）"  # 标签本身就得限定范围


def test_render_mentions_it_is_an_estimate():
    from agentic_rag.eval.cost import render_report

    out = render_report([estimate_ingest(1_000_000, 3000)], title="T")
    assert "折算" in out and "usage" in out
