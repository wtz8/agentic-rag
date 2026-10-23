"""从查询级 trace 重建评测的检索侧指标（产物追溯通道）。

设计意图：聚合报告是**结果**，查询级 trace 是**证据**。两者独立落盘，
于是任一侧需要复核时——指标口径存疑、要换算法重算、要按查询归因成本——
都能从 `data/traces/*.jsonl` 精确重建，不必重跑付费调用。

本脚本就是这条通道：retrieve 事件带每条查询的最终融合候选列表，
据此重算检索侧指标（`eval/results/recovered_*.md` 由它产出）。

重建口径与 `eval/runner.py` 完全一致：
- 候选 = 最后一次 retrieve 事件的 `fused` 列表（跨轮合并后的最终候选集）；
- 命中判定按父块家族规范化（`family_map` 后比对）；
- unanswerable 不进 Recall 分母；锚点解析失败的用例同样排除并显式列出。

变体归属：评测按 `--variants A0,A1,A2,A3,A4` 顺序串行执行、每变体 78 条、
用例顺序固定 → 时间排序后第 b 块的第 i 条 trace 对应第 i 个用例。
归属用问题文本交叉校验（rewrite.original == case.question）。

用法：
    PYTHONPATH=src python scripts/recover_eval_from_traces.py \
        --trace data/traces/2026-10-22.jsonl --from "2026-10-22 15:58" \
        --to "2026-10-22 16:33" --variants A0,A1,A2,A3,A4

只读不写索引；输出 JSON + Markdown 到 eval/results/。
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rag.config import get_config  # noqa: E402
from agentic_rag.eval import runner  # noqa: E402
from agentic_rag.eval.metrics import hit, mrr, ndcg_at_k, recall_at_k  # noqa: E402
from agentic_rag.index.store import IndexStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", default="data/traces/2026-10-22.jsonl")
    p.add_argument("--from", dest="t_from", required=True, help="窗口开始（如 '2026-10-22 15:58'）")
    p.add_argument("--to", dest="t_to", required=True, help="窗口结束（如 '2026-10-22 16:33'）")
    p.add_argument("--variants", default="A0,A1,A2,A3,A4")
    p.add_argument("--qa-set", default="eval/qa_set.jsonl")
    p.add_argument("--out-dir", default="eval/results")
    return p.parse_args()


def window(ts_text: str) -> float:
    return datetime.datetime.strptime(ts_text, "%Y-%m-%d %H:%M").timestamp()


def question_of(rec: dict) -> str | None:
    """从 trace 里取原始问题（rewrite 事件的 original 字段）。"""
    for e in rec.get("events", []):
        if e.get("step") == "rewrite":
            return e.get("payload", {}).get("original")
        pl = e.get("payload", {})
        if isinstance(pl, dict) and pl.get("original"):
            return pl["original"]
    return None


def retrieved_of(rec: dict) -> list[str]:
    """与 runner 同口径：最后一次 retrieve 事件的 fused 列表。"""
    out: list[str] = []
    for e in rec.get("events", []):
        if e.get("step") == "retrieve" and isinstance(e.get("payload", {}).get("fused"), list):
            out = [str(x) for x in e["payload"]["fused"]]
    return out


def verify_of(rec: dict) -> dict:
    for e in rec.get("events", []):
        if e.get("step") == "verify":
            return e.get("payload", {})
    return {}


def mean(vals: list) -> float | None:
    xs = [float(v) for v in vals if v is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def main() -> int:
    args = parse_args()
    cases = runner.load_cases(args.qa_set)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    n = len(cases)

    cfg = get_config()
    store = IndexStore(cfg)
    store.load()
    family = runner.family_map(store)

    # 锚点解析一次（全量消融五个变体共用基线索引，A 组不改索引期参数）
    golden_by_case: list[list[str]] = []
    unresolved = 0
    for case in cases:
        golden, failed = runner.resolve_anchors(case, store)
        golden_by_case.append(golden)
        unresolved += len(failed)

    # 载入 trace 窗口
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

    if len(recs) != n * len(variants):
        print(
            f"⚠ 窗口内 trace 数 {len(recs)} ≠ 用例 {n} × 变体 {len(variants)}，"
            "请检查 --from/--to 窗口；仍按前 N 块尽力恢复。",
            file=sys.stderr,
        )

    report_variants: list[dict] = []
    mismatches: Counter = Counter()
    for bi, name in enumerate(variants):
        block = recs[bi * n : (bi + 1) * n]
        if len(block) < n:
            print(f"⚠ {name} 只有 {len(block)}/{n} 条 trace，跳过", file=sys.stderr)
            continue
        rows = []
        n_err = 0
        for i, rec in enumerate(block):
            case = cases[i]
            q = question_of(rec)
            if q is not None and q != case.question:
                mismatches[name] += 1
            if rec.get("error"):
                n_err += 1
                continue
            retrieved = [family.get(cid, cid) for cid in retrieved_of(rec)]
            golden = [family.get(cid, cid) for cid in golden_by_case[i]]
            row = {
                "idx": i,
                "kind": case.kind,
                "error": bool(rec.get("error")),
                "latency_ms": rec.get("total_ms"),
                "answer_len": rec.get("answer_len", 0),
                "verify": verify_of(rec),
            }
            if golden:  # unanswerable / 锚点失败的不进检索分母（与 runner 一致）
                rc, gc = retrieved, golden
                row.update(
                    hit5=hit(rc, gc, 5),
                    hit20=hit(rc, gc, 20),
                    recall5=recall_at_k(rc, gc, 5),
                    recall20=recall_at_k(rc, gc, 20),
                    mrr_v=mrr(rc, gc),
                    ndcg5=ndcg_at_k(rc, gc, 5),
                )
            rows.append(row)

        ok = [r for r in rows if not r["error"]]
        scored = [r for r in ok if "recall5" in r]
        report_variants.append(
            {
                "variant": name,
                "traces": n,
                "errors": n_err,
                "scored_cases": len(scored),
                "metrics": {
                    "recall@5": mean([r["recall5"] for r in scored]),
                    "recall@20": mean([r["recall20"] for r in scored]),
                    "mrr": mean([r["mrr_v"] for r in scored]),
                    "ndcg@5": mean([r["ndcg5"] for r in scored]),
                    "hit_rate": mean([r["hit5"] for r in scored]),
                },
                # 系统内部校验分（verify 节点），非 LLM judge，口径不同只作参考
                "pipeline_faithfulness": mean(
                    [r["verify"].get("faithfulness") for r in ok if r["verify"]]
                ),
                "latency_p50_ms": mean(
                    sorted(r["latency_ms"] or 0 for r in ok)[len(ok) // 2 : len(ok) // 2 + 1]
                ),
            }
        )

    out = {
        "recovered_from": {
            "trace": args.trace,
            "window": [args.t_from, args.t_to],
            "note": (
                "由查询级 trace 重建的检索侧指标；"
                "口径与 runner 一致（最后 retrieve 事件的 fused、父块家族规范化、"
                "unanswerable 不进分母）。judge 的 faithfulness/relevance/citation "
                "与答案文本不可恢复；verify 字段是系统内部校验分，不是 judge。"
            ),
        },
        "qa_set": args.qa_set,
        "case_count": n,
        "unresolved_anchors": unresolved,
        "question_mismatches": dict(mismatches),
        "variants": report_variants,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "recovered_20261022_full_ablation.json"
    raw.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# 消融评测恢复报告（trace 重建，检索侧）", ""]
    md.append(f"- 来源：`{args.trace}` 窗口 {args.t_from} ~ {args.t_to}")
    md.append(f"- 用例：{n} 条 × {len(variants)} 变体；未解析锚点 {unresolved} 个用例不计入分母")
    md.append("- ⚠ 生成侧 judge 指标（faithfulness/relevance/citation）**不可恢复**，本表只有检索侧")
    if mismatches:
        md.append(f"- ⚠ 问题匹配失败 {sum(mismatches.values())} 条（按位置归属，不影响指标）")
    md.append("")
    md.append("| 变体 | 完好/总数 | 错误 | recall@5 | recall@20 | mrr | ndcg@5 | hit_rate | verify忠实度 | p50(ms) |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for v in report_variants:
        m = v["metrics"]
        md.append(
            f"| {v['variant']} | {v['traces'] - v['errors']}/{v['traces']} | {v['errors']} "
            f"| {m['recall@5']} | {m['recall@20']} | {m['mrr']} | {m['ndcg@5']} "
            f"| {m['hit_rate']} | {v['pipeline_faithfulness']} | {v['latency_p50_ms']} |"
        )
    summary = out_dir / "recovered_20261022_full_ablation.md"
    summary.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"原始结果: {raw}")
    print(f"摘要报告: {summary}")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
