"""按意图/类型拆解召回失败，回答"召回低在哪"而不是只看一个总数。

用法：
    PYTHONPATH=src python scripts/diagnose_recall.py \
        --trace data/traces/2026-10-22.jsonl \
        --from "2026-10-22 15:58" --to "2026-10-22 16:33" \
        --variants A0,A1,A2,A3,A4

输出：每个变体按 kind / intent 分组的 hit@5、recall@5、recall@20，
以及 recall@20 < 1 的"真·检索失败"用例清单（含问题与来源文档）。
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rag.config import get_config  # noqa: E402
from agentic_rag.eval import runner  # noqa: E402
from agentic_rag.eval.metrics import hit, recall_at_k  # noqa: E402
from agentic_rag.index.store import IndexStore  # noqa: E402


def window(ts_text: str) -> float:
    return datetime.datetime.strptime(ts_text, "%Y-%m-%d %H:%M").timestamp()


def question_of(rec: dict) -> str | None:
    for e in rec.get("events", []):
        pl = e.get("payload", {})
        if "original" in pl:
            return pl["original"]
    return None


def retrieved_of(rec: dict) -> list[str]:
    out: list[str] = []
    for e in rec.get("events", []):
        if e.get("step") == "retrieve":
            out = e.get("payload", {}).get("fused") or []
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="data/traces/2026-10-22.jsonl")
    ap.add_argument("--qa-set", default="eval/qa_set.jsonl")
    ap.add_argument("--from", dest="t_from", required=True)
    ap.add_argument("--to", dest="t_to", required=True)
    ap.add_argument("--variants", default="A0,A1")
    ap.add_argument("--top", type=int, default=12, help="列出多少条失败用例")
    args = ap.parse_args()

    cases = runner.load_cases(args.qa_set)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    n = len(cases)

    cfg = get_config()
    store = IndexStore(cfg)
    store.load()
    family = runner.family_map(store)

    golden_by_case = []
    for case in cases:
        golden, _ = runner.resolve_anchors(case, store)
        golden_by_case.append(golden)

    recs = []
    for line in open(args.trace, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        t = d.get("finished_at") or 0
        if window(args.t_from) <= t <= window(args.t_to):
            recs.append(d)
    recs.sort(key=lambda d: d["finished_at"])

    for bi, name in enumerate(variants):
        block = recs[bi * n : (bi + 1) * n]
        if len(block) < n:
            print(f"\n### {name}: 只有 {len(block)}/{n} 条，跳过")
            continue
        by_kind: dict[str, list[dict]] = defaultdict(list)
        by_intent: dict[str, list[dict]] = defaultdict(list)
        misses: list[tuple[float, int, str, str]] = []

        for i, rec in enumerate(block):
            case = cases[i]
            q = question_of(rec)
            if q is not None and q != case.question:
                continue
            if rec.get("error") or not golden_by_case[i]:
                continue
            rc = [family.get(c, c) for c in retrieved_of(rec)]
            gc = [family.get(c, c) for c in golden_by_case[i]]
            row = {
                "hit5": hit(rc, gc, 5),
                "r5": recall_at_k(rc, gc, 5),
                "r20": recall_at_k(rc, gc, 20),
            }
            by_kind[case.kind].append(row)
            by_intent[case.intent].append(row)
            if row["r20"] < 1.0:
                src = ",".join(sorted({a.source for a in case.golden_anchors}))[:40]
                misses.append((row["r20"], i, case.question[:44], src))

        def line(name_: str, rows: list[dict]) -> str:
            m = len(rows)
            return (
                f"  {name_:<18} n={m:<3} hit@5={sum(r['hit5'] for r in rows)/m:.2f}  "
                f"recall@5={sum(r['r5'] for r in rows)/m:.3f}  "
                f"recall@20={sum(r['r20'] for r in rows)/m:.3f}"
            )

        print(f"\n### {name}（{len(block)} 条，按类型拆解）")
        for k in sorted(by_kind, key=lambda x: -len(by_kind[x])):
            print(line(k, by_kind[k]))
        print("  — 按意图 —")
        for k in sorted(by_intent, key=lambda x: -len(by_intent[x])):
            print(line(k, by_intent[k]))

        misses.sort()
        print(f"  — 真·检索失败（recall@20 < 1）共 {len(misses)} 条，最差 {args.top} 条 —")
        for r20, i, q, src in misses[: args.top]:
            print(f"    [{i:>2}] r20={r20:.2f} | {q} | ← {src}")
        print("  失败来源文档分布:", Counter(m[3] for m in misses).most_common(5))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
