"""K1 充分性阈值的离线参考标定（docs/04 §6.4）。

为什么要这个脚本
----------------
`sufficiency_vector_floor = 0.35` 是设计阶段拍的初值，从来没在真实语料上验证过。
正式的 K1 标定需要「query 向量 vs chunk 向量」的相似度分布，而 query 向量必须调用
embedding API —— 账号不可用时这条路走不通。

于是退一步：**用索引里已有的真实向量度量这个 embedding 模型在本语料上的相似度分离度**。
文本对的两侧都是真实 text-embedding-v4 输出（1024d，已归一化），所以相似度的**量纲**
是真实的；变的是配对方式。

口径限制（引用结论时必须一并带上）
----------------------------------
- 正样本是 **chunk↔chunk**（同父块内的兄弟块），**不是 query↔chunk**。
  查询通常比正文块短、措辞也更不像原文，因此查询侧的真实相似度一般会**更低**。
- 所以本脚本的输出是「**这个模型在本语料上，相关与不相关的分界大概落在哪**」的量纲参考，
  **不能替代**正式标定，也不能直接写进 config 当实测值用。
- 兄弟块之间若存在切分重叠，正样本相似度会被高估；脚本因此同时报告无重叠子集的结果。

用法
----
    python scripts/calibrate_k1.py                 # 默认采样 4000 点 / 每组 2000 对
    python scripts/calibrate_k1.py --points 8000 --pairs 5000
    python scripts/calibrate_k1.py --index data/index/qdrant
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

import numpy as np
from qdrant_client import QdrantClient

COLLECTION = "agentic_rag"
DEFAULT_INDEX = "data/index/qdrant"


def _quantiles(vals: np.ndarray, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> str:
    return "  ".join(f"p{int(q * 100)}={np.quantile(vals, q):.3f}" for q in qs)


def _best_threshold(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float, float, float]:
    """在候选阈值里挑 Youden's J = TPR − FPR 最大的点。

    返回 (阈值, J, TPR, FPR)。TPR/FPR 的定义相对"正样本 = 语义相关"。
    """
    grid = np.unique(np.round(np.concatenate([pos, neg]), 3))
    if grid.size > 2000:  # 网格过密时抽稀，避免 O(n²)
        grid = np.quantile(grid, np.linspace(0, 1, 2000))
    best = (float(grid[0]), -1.0, 0.0, 0.0)
    for t in grid:
        tpr = float((pos >= t).mean())
        fpr = float((neg >= t).mean())
        j = tpr - fpr
        if j > best[1]:
            best = (float(t), j, tpr, fpr)
    return best


def _pairs_same_parent(groups: dict[str, list[int]], n: int, rng: random.Random) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    buckets = [idx for idx in groups.values() if len(idx) >= 2]
    rng.shuffle(buckets)
    for idx in buckets:
        if len(pairs) >= n:
            break
        a, b = rng.sample(idx, 2)
        pairs.append((a, b))
    return pairs


def _pairs_cross_doc(labels: list[str], n: int, rng: random.Random) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    guard = 0
    while len(pairs) < n and guard < n * 50:
        guard += 1
        i = rng.randrange(len(labels))
        j = rng.randrange(len(labels))
        if labels[i] != labels[j]:
            pairs.append((i, j))
    return pairs


def _overlap_ratio(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def main() -> int:
    p = argparse.ArgumentParser(description="K1 阈值离线参考标定")
    p.add_argument("--index", default=DEFAULT_INDEX, help="Qdrant local 数据目录")
    p.add_argument("--points", type=int, default=4000, help="采样点数")
    p.add_argument("--pairs", type=int, default=2000, help="每组样本对数")
    p.add_argument("--seed", type=int, default=20261022)
    args = p.parse_args()

    rng = random.Random(args.seed)
    client = QdrantClient(path=args.index)
    info = client.get_collection(COLLECTION)
    print(f"collection: {COLLECTION}")
    print(f"  points   = {info.points_count}")
    print(f"  dim      = {info.config.params.vectors.size}")
    print(f"  distance = {info.config.params.vectors.distance}")
    print()

    pts, _ = client.scroll(COLLECTION, limit=args.points, with_vectors=True, with_payload=True)
    if len(pts) < 50:
        print("采样点太少，无法标定")
        return 1

    vecs = np.array([p.vector for p in pts], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1)
    print(f"sampled   = {len(pts)}  |  norm mean = {norms.mean():.4f}")
    # 余弦 = 点积；若服务端未完全归一化则先归一，保证口径是"余弦"
    vecs = vecs / np.clip(norms, 1e-12, None)[:, None]

    payloads = [p.payload or {} for p in pts]
    doc_ids = [str(pl.get("doc_id", "")) for pl in payloads]
    parent_ids = [str(pl.get("parent_id", "")) for pl in payloads]
    texts = [str(pl.get("text", "")) for pl in payloads]

    groups: dict[str, list[int]] = defaultdict(list)
    for i, pid in enumerate(parent_ids):
        if pid:
            groups[pid].append(i)

    pos_pairs = _pairs_same_parent(groups, args.pairs, rng)
    neg_pairs = _pairs_cross_doc(doc_ids, args.pairs, rng)

    if not pos_pairs or not neg_pairs:
        print("正/负样本对为空，无法标定")
        return 1

    pi = np.array([a for a, _ in pos_pairs])
    pj = np.array([b for _, b in pos_pairs])
    ni = np.array([a for a, _ in neg_pairs])
    nj = np.array([b for _, b in neg_pairs])

    pos = np.einsum("ij,ij->i", vecs[pi], vecs[pj])
    neg = np.einsum("ij,ij->i", vecs[ni], vecs[nj])

    print()
    print("=== 余弦相似度分布（本语料，真实 text-embedding-v4 向量）===")
    print(f"相关样本（同父块兄弟块） n={len(pos)}  |  {_quantiles(pos)}")
    print(f"无关样本（跨文档随机对） n={len(neg)}  |  {_quantiles(neg)}")
    print()

    t, j, tpr, fpr = _best_threshold(pos, neg)
    print(f"Youden's J 最大点：threshold = {t:.3f}  (J = {j:.3f}, TPR = {tpr:.3f}, FPR = {fpr:.3f})")

    # 重叠敏感性：兄弟块若带切分重叠，会把正样本相似度抬高，阈值随之偏高
    ratios = np.array([_overlap_ratio(texts[a], texts[b]) for a, b in pos_pairs])
    clean = ratios <= 0.5
    if clean.sum() >= 100:
        pos_c = pos[clean]
        t2, j2, tpr2, fpr2 = _best_threshold(pos_c, neg)
        print(f"剔除高重叠对后（ratio≤0.5, n={int(clean.sum())}）：threshold = {t2:.3f}  "
              f"(J = {j2:.3f}, TPR = {tpr2:.3f}, FPR = {fpr2:.3f})")
        print()
        print(f"参考区间：{min(t, t2):.3f} – {max(t, t2):.3f}")

    print()
    print("提示：这是 chunk↔chunk 的量纲参考，不是 query↔chunk 的正式标定值。")
    print("      正式标定请用真实 API 跑 eval --calibrate。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
