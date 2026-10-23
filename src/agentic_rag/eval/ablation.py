"""消融实验变体表（04 §9.4）。

A 组是**累加链**（A0 ⊂ A1 ⊂ … ⊂ A4），回答"每个组件的边际收益"；
B 组是**单变量开关**，回答"这一个设计决策换掉会怎样"。

两组的跑法都必须用同一份 mini 集（06 §7.2），结论才可比。
变体补丁写的是**该变体相对默认配置树的取值**，不是"相对上一个变体的增量"——
累加链的语义由补丁内容的包含关系体现（A0 ⊂ A1 ⊂ … ⊂ A4）。
⚠️ 因此**A4 不等于基线**：基线 `allow_reflect_for_simple=False` 且
`skip_rewrite_for_simple=True`，而 A4 把两个都打开了。
"""

from __future__ import annotations

from .runner import VariantSpec

__all__ = ["VARIANTS", "GROUP_A", "GROUP_B", "get_variant", "get_variants"]


VARIANTS: dict[str, VariantSpec] = {
    # ── 基线：当前配置树原样（A4 语义上等于它，但 retrieval_signature 相同与否取决于配置）──
    "base": VariantSpec(name="base", patch={}, note="基线：当前配置树原样执行"),
    # ── A 组：累加链 ──────────────────────────────────────
    # 注意：累加链必须**从最小配置逐个打开**组件，而配置树默认
    # use_bm25/use_rerank 全开——所以每个补丁都必须显式写出前面组件的
    # 关闭状态（M25：曾经 A0 只关了 BM25 却留着 rerank、A2 的 use_rerank=True
    # 是对默认值的空操作，导致"A2 = A1 + Rerank"这条边际收益根本没被测过）。
    "A0": VariantSpec(
        name="A0",
        patch={"retrieval": {"use_bm25": False, "use_rerank": False}},
        note="仅向量检索（无 BM25、无 rerank）",
    ),
    "A1": VariantSpec(
        name="A1",
        patch={"retrieval": {"use_bm25": True, "use_rerank": False}},
        note="A0 + BM25 混合（RRF）",
    ),
    "A2": VariantSpec(
        name="A2",
        patch={"retrieval": {"use_bm25": True, "use_rerank": True}},
        note="A1 + Rerank（此时检索配置 = 基线）",
    ),
    "A3": VariantSpec(
        name="A3",
        patch={
            "retrieval": {"use_bm25": True, "use_rerank": True},
            "agent": {"skip_rewrite_for_simple": False},
        },
        note="A2 + 查询改写（simple 也改写）",
    ),
    "A4": VariantSpec(
        name="A4",
        patch={
            "retrieval": {"use_bm25": True, "use_rerank": True},
            "agent": {
                "skip_rewrite_for_simple": False,
                "allow_reflect_for_simple": True,
                "max_retrieve_iterations": 3,
            },
        },
        # 早先的注记写"等于基线语义"是错的：基线 `allow_reflect_for_simple=False`，
        # 而 A4 把它打开了，两者并不相等。A4 的对照是 A3（单变量 = 反思循环）。
        note="A3 + 反思循环（simple 也进反思；与基线不等价，基线的 simple 不反思）",
    ),
    # ── B 组：单变量开关（对照 = 基线 / A4） ────────────────
    "B1": VariantSpec(
        name="B1",
        patch={"retrieval": {"fusion_strategy": "weighted"}},
        note="融合策略：加权 vs 基线 RRF",
    ),
    "B2": VariantSpec(
        name="B2",
        patch={"ingestion": {"chunk_strategy": "structural"}},
        note="切分：structural vs 基线 parent_child（需重建索引，runner 自动处理）",
    ),
    "B3": VariantSpec(
        name="B3",
        patch={"retrieval": {"parent_backfill": False}},
        note="父块回填关闭：上下文直接用子块",
    ),
    "B4": VariantSpec(
        name="B4",
        patch={"agent": {"allow_reflect_for_simple": True}},
        note="simple 也进检索侧反思循环（基线为关）",
    ),
    "B5": VariantSpec(
        name="B5",
        patch={"generation": {"regenerate_on_unsupported": True}},
        note="生成侧重生成开启（基线为关）",
    ),
}

GROUP_A = ["A0", "A1", "A2", "A3", "A4"]
GROUP_B = ["B1", "B2", "B3", "B4", "B5"]


def get_variant(name: str) -> VariantSpec:
    if name not in VARIANTS:
        raise KeyError(f"未定义的消融变体: {name}（可选：{', '.join(VARIANTS)}）")
    return VARIANTS[name]


def get_variants(names: list[str] | None = None) -> list[VariantSpec]:
    """None → 全部（A 组 + B 组，不含 base——base 只是跑通链路的冒烟档）。
    B2 会触发索引重建，跑之前想清楚成本。"""
    if names is None:
        names = GROUP_A + GROUP_B
    return [get_variant(n) for n in names]
