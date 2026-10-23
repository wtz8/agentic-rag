"""变体配对比较：在**两变体都成功**的交集上比较，而不是拿总分横比。

用途
----
全量消融跑中断时（例如 A0–A2 跑满 78 条、A3 只跑完 46 条），
直接把两变体的总分并列比较是错的：缺失是**按用例顺序整段中断**，不是随机缺失，
两变体的题目难度构成不同，横比读出来的差值里混着题目难度，不是组件增益。

本脚本的做法：只在"两变体都成功评分"的那些题上逐条配对求差。
同一道题在配对内两侧都出现，难度被抵消，剩下的才是两变体本身的差异。

产出
----
- 每个指标的配对差 Δ、95% 置信区间（成对 bootstrap）、净胜率与符号检验 p 值
- 外推值 = 基线全量实测 + Δ，用于和 A0–A2 的完整分母放到同一量尺上
- `subset_bias` = 基线子集值 − 基线全量值，用于检验这个子集是否偏难
  （它不修正 Δ，只回答"子集能不能代表全量"）

用法
----
    PYTHONPATH=src python scripts/pairwise_variant_compare.py \
        --from "2026-10-22 15:58" --to "2026-10-22 16:33" \
        --baseline A2 --target A3

离线：只读 trace 与已建索引，不发起任何 API 调用。
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rag.config import get_config  # noqa: E402
from agentic_rag.eval import runner  # noqa: E402
from agentic_rag.eval.metrics import compare_paired, hit, ndcg_at_k, recall_at_k  # noqa: E402
from agentic_rag.index.store import IndexStore  # noqa: E402

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20261022


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", default="data/traces/2026-10-22.jsonl")
    p.add_argument("--from", dest="t_from", required=True, help="窗口开始（如 '2026-10-22 15:58'）")
    p.add_argument("--to", dest="t_to", required=True, help="窗口结束（如 '2026-10-22 16:33'）")
    p.add_argument("--variants", default="A0,A1,A2,A3,A4")
    p.add_argument("--baseline", default="A2")
    p.add_argument("--target", default="A3")
    p.add_argument("--qa-set", default="eval/qa_set.jsonl")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--out-dir", default="eval/results")
    return p.parse_args()


def window(ts_text: str) -> float:
    return datetime.datetime.strptime(ts_text, "%Y-%m-%d %H:%M").timestamp()


def retrieved_of(rec: dict) -> list[str]:
    """与 runner 同口径：最后一次 retrieve 事件的 fused 列表。"""
    out: list[str] = []
    for e in rec.get("events", []):
        if e.get("step") == "retrieve" and isinstance(e.get("payload", {}).get("fused"), list):
            out = [str(x) for x in e["payload"]["fused"]]
    return out


def load_window(path: str, t0: float, t1: float) -> list[dict]:
    recs = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if t0 <= d.get("finished_at", 0) <= t1:
            recs.append(d)
    recs.sort(key=lambda d: d["finished_at"])
    return recs


def score_all(
    block: list[dict],
    golden_by_case: list[list[str]],
    family: dict[str, str],
    k: int,
) -> dict[str, list[float | None]]:
    """逐条算指标；该条失败或无 golden 时记 None（不填 0，避免污染分母）。"""
    out: dict[str, list[float | None]] = {
        f"recall@{k}": [],
        "hit_rate": [],
        f"ndcg@{k}": [],
    }
    for i, rec in enumerate(block):
        g = golden_by_case[i] if i < len(golden_by_case) else []
        g = sorted({family.get(x, x) for x in g})
        if not g or rec.get("error"):
            for key in out:
                out[key].append(None)
            continue
        rc = [family.get(c, c) for c in retrieved_of(rec)]
        out[f"recall@{k}"].append(recall_at_k(rc, g, k))
        out["hit_rate"].append(float(hit(rc, g, k)))
        out[f"ndcg@{k}"].append(ndcg_at_k(rc, g, k))
    return out


def bootstrap_ci(diffs: list[float], n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """成对 bootstrap：重抽**差值序列**本身，保持配对结构。"""
    if len(diffs) < 2:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = 0.0
        for _ in range(len(diffs)):
            s += diffs[rng.randrange(len(diffs))]
        means.append(s / len(diffs))
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[min(int(0.975 * n), n - 1)]
    return (round(lo, 4), round(hi, 4))


def main() -> int:
    args = parse_args()
    cases = runner.load_cases(args.qa_set)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    n = len(cases)
    k = args.top_k

    cfg = get_config()
    store = IndexStore(cfg)
    store.load()
    family = runner.family_map(store)
    golden_by_case = [runner.resolve_anchors(c, store)[0] for c in cases]

    recs = load_window(args.trace, window(args.t_from), window(args.t_to))
    blocks = {v: recs[i * n : (i + 1) * n] for i, v in enumerate(variants)}
    if args.baseline not in blocks or args.target not in blocks:
        print(f"变体名不在 {variants} 内：{args.baseline} / {args.target}")
        return 2

    base = score_all(blocks[args.baseline], golden_by_case, family, k)
    tgt = score_all(blocks[args.target], golden_by_case, family, k)

    metrics = [f"recall@{k}", "hit_rate", f"ndcg@{k}"]
    rows = []
    for m in metrics:
        b, t = base[m], tgt[m]
        pairs = [(bi, ti) for bi, ti in zip(b, t) if bi is not None and ti is not None]
        if not pairs:
            rows.append({"metric": m, "n_pairs": 0, "note": "无交集，无法比较"})
            continue
        bs = [p[0] for p in pairs]
        ts = [p[1] for p in pairs]
        diffs = [ti - bi for bi, ti in pairs]
        cmp_ = compare_paired(m, ts, bs)
        lo, hi = bootstrap_ci(diffs)
        # 基线全量实测（该指标上所有可评分条目）
        b_all = [x for x in b if x is not None]
        base_full = sum(b_all) / len(b_all) if b_all else 0.0
        rows.append(
            {
                "metric": m,
                "n_pairs": len(pairs),
                "baseline_subset": round(sum(bs) / len(bs), 4),
                "target_subset": round(sum(ts) / len(ts), 4),
                "delta": cmp_.mean_delta,
                "ci95": [lo, hi],
                "net_win_rate": cmp_.net_win_rate,
                "p_value": cmp_.p_value,
                "verdict": cmp_.verdict,
                "baseline_full": round(base_full, 4),
                "subset_bias": round(sum(bs) / len(bs) - base_full, 4),
                "target_full_extrapolated": round(base_full + cmp_.mean_delta, 4),
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pairwise_{args.baseline}_{args.target}"
    (out_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "baseline": args.baseline,
                "target": args.target,
                "k": k,
                "window": [args.t_from, args.t_to],
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    md = [
        f"# 配对比较：{args.target} vs {args.baseline}（中断场景下无偏估计）",
        "",
        f"窗口 {args.t_from} → {args.t_to}　变体顺序 {','.join(variants)}　top_k={k}",
        "",
        "| 指标 | 配对数 | 基线子集 | 目标子集 | Δ | 95%CI | 净胜率 | p | 判定 | 外推全量 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        if r.get("n_pairs", 0) == 0:
            md.append(f"| {r['metric']} | 0 | — | — | — | — | — | — | 无交集 | — |")
            continue
        md.append(
            f"| {r['metric']} | {r['n_pairs']} | {r['baseline_subset']:.3f} | "
            f"{r['target_subset']:.3f} | **{r['delta']:+.3f}** | "
            f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['net_win_rate']:+.2f} | "
            f"{r['p_value']:.3f} | {r['verdict']} | ≈ {r['target_full_extrapolated']:.3f} |"
        )
    md += [
        "",
        "**读法**",
        "",
        f"- Δ 是**目标减基线**（{args.target} − {args.baseline}），负号表示目标更差。",
        "- Δ 只在两变体都成功的交集上算，题目难度在配对内抵消。",
        "- 外推全量 = 基线全量实测 + Δ，把目标拉回到与 A0–A2 相同的分母量尺。",
        "- `subset_bias` = 基线子集值 − 基线全量值，只用于判断子集是否偏难，**不用它修正 Δ**。",
        "",
        f"> 口径纪律：本表是**从 trace 离线重算**的结果，不是新跑的实测；"
        f"外推值依赖\"配对差在全量上保持不变\"这一假设，须复跑验证。",
    ]
    (out_dir / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"配对 {args.baseline} / {args.target}：{len([r for r in rows if r.get('n_pairs', 0)])} 个指标")
    for r in rows:
        if r.get("n_pairs", 0):
            print(
                f"  {r['metric']}: n={r['n_pairs']} Δ={r['delta']:+.4f} "
                f"CI=[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] "
                f"净胜率={r['net_win_rate']:+.2f} → 外推 {r['target_full_extrapolated']:.3f}"
            )
    print(f"产物：{out_dir / stem}.md / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
