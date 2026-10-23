"""命令行入口（04 §11）。

`inspect` 是刻意保留的：质量模块的价值必须能被看到——
一份扫描件的报告里写着"数字守恒：不适用"，恰恰说明这套检查是真的在工作，而不是在硬算。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bootstrap import build_pipeline, build_store
from .config import ConfigError, get_config
from .index.store import IndexStore, IndexVersionMismatch
from .ingestion.pipeline import load_documents, process_file


# ── ingest ────────────────────────────────────────────────────────────


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = get_config()
    cfg.ingestion.workers = args.workers

    # `--dry-run` 也走解析（解析是本地的、不花钱），因为只有解析完才知道
    # 实际有多少检索单元——按文件数估会和真实值差出好几倍。
    # 但解析会移动拒收文件，dry-run 下必须关掉这个副作用。
    store = IndexStore(cfg)
    chunks, results = load_documents(
        Path(args.input), cfg, move_rejected=not args.keep_rejected and not args.dry_run
    )
    ok = [r for r in results if not r.rejected]
    bad = [r for r in results if r.rejected]

    if not chunks:
        print(f"没有可入库的 chunk（共 {len(results)} 个文件），索引未重建。", file=sys.stderr)
        return 1

    if args.dry_run:
        from .eval.cost import estimate_ingest, render_report

        units = [c for c in chunks if not c.is_parent]
        print(
            render_report(
                [estimate_ingest(sum(len(c.text) for c in units), len(units))],
                title="成本预估（--dry-run，未调用任何付费 API）",
            )
        )
        print(f"\n文件：{len(ok)} 篇入库 / {len(bad)} 篇拒收")
        if args.incremental:
            print("注：增量模式下只对**新增文档**付费；上面的估算是全量口径（上界）。")
        return 0

    if args.incremental:
        try:
            store.load()  # 先确认"增量要并到谁身上"，失败则没有任何增量可言
        except IndexVersionMismatch as e:
            print(f"索引不可用，--incremental 退回全量构建（{e}）", file=sys.stderr)
            args.incremental = False
        else:
            return _incremental(cfg, store, ok, bad, args)

    report = store.build(chunks)
    # 03 §10：构建完成且 run_probe_after_build=True 时自动跑探针。
    # 探针失败不阻断入库（告警而非门禁），但索引标记 unverified 且失败用例必须看得见。
    # 只跑期望来源在本次构建语料里的探针——探针集是语料绑定的，
    # 对语料子集构建时其他条目不适用，跑了只会把报警变成噪声。
    probe_result = None
    if cfg.ingestion.run_probe_after_build:
        from .eval.probe import filter_cases_for_sources, load_probe_set, run as run_probe

        probe_cases = filter_cases_for_sources(load_probe_set(), store.stats()["sources"])
        if probe_cases:
            probe_result = run_probe(cfg, k=5, store=store, cases=probe_cases)
            report.probe_total = probe_result.total
            report.probe_hits = probe_result.hit
            report.probe_failures = [(c.question, got) for c, got in probe_result.failures]
            if probe_result.total and probe_result.hit_rate < 0.9:
                report.verified = False
    _print_ingest(report, ok, bad, store)
    if probe_result is not None and report.probe_failures:
        print("探针未命中（期望来源未进 Top-5）：")
        for q, got in report.probe_failures:
            print(f"  × {q} → 实际来源: {', '.join(got) or '（空）'}")
    if probe_result is not None and probe_result.content_failures:
        print("内容未命中（期望片段不在期望来源的 Top-5 块内，属排名问题非连通性问题）：")
        for c, got in probe_result.content_failures:
            print(f"  △ {c.question}（期望含「{c.expect_answer_contains}」）")
    return 0 if report.verified else 2


def _incremental(cfg, store: IndexStore, ok: list, bad: list, args) -> int:
    """`--incremental`：只把**本次新增**的文档并入既有索引。

    两条必须说清的限制：
    - **BM25 仍是全量重建**（见 `index/bm25.py` 模块 docstring：IDF 依赖全语料的
      N 与 df）。省下的只是向量侧的 embedding 调用，不是全部摄取成本——
      把它宣传成"增量很便宜"会误导成本预期。
    - **新增判定按 `doc_id`（文件内容 sha1），不按文件名**。同一份文档改名或移到
      别的子目录不会被当成新文档重复入库，这正是 `doc_id_of` 去掉路径的原因。
    """
    known = store.indexed_doc_ids()
    fresh = [r for r in ok if r.doc is not None and r.doc.doc_id not in known]
    skipped = len(ok) - len(fresh)
    if not fresh:
        print(f"没有新增文档（{skipped} 篇已在索引中），索引未变动。")
        for r in bad:
            print(f"  - 拒收 {r.path.name}: {r.reason}")
        return 0

    delta = [c for r in fresh for c in r.chunks]
    print(f"增量：{len(fresh)} 篇新增 / {skipped} 篇已入库 / {len(bad)} 篇拒收")
    report = store.upsert_documents(delta)
    _print_ingest(report, fresh, bad, store)
    return 0 if not report.partial_failure else 2


def _print_ingest(report, ok, bad, store) -> None:
    print(f"入库文档    : {len(ok)} 篇")
    print(f"拒绝入库    : {len(bad)} 篇")
    for r in bad:
        print(f"  - {r.path.name}: {r.reason}")
    print(f"检索单元    : {report.chunks_indexed}")
    print(f"父块        : {report.parents_written}")
    print(f"索引版本    : {report.index_version}")
    print(f"耗时        : {report.elapsed_s}s")
    for w in report.warnings:
        print(f"  ! {w}")
    if report.partial_failure:
        print("  ! 部分失败：请查看上面的 warning，索引可能不完整")
    st = store.stats()
    print(f"后端        : {st['backend']}（BM25 {'可用' if st['bm25_available'] else '不可用'}）")


# ── ask ───────────────────────────────────────────────────────────────


def cmd_ask(args: argparse.Namespace) -> int:
    cfg = get_config()
    pipeline = build_pipeline(cfg)
    r = pipeline.run(args.query)
    print(f"[{r.confidence}] {r.answer}")
    print()
    for c in r.citations:
        loc = f"P{c.page}" if c.page else "（无页码）"
        print(f"  [{c.index}] {c.source}｜{' > '.join(c.title_path) or '（无标题）'}｜{loc}")
        print(f"      {c.quote}")
    print()
    print(
        f"intent={r.intent} 检索轮次={r.retrieve_iterations} 生成次数={r.generate_iterations} "
        f"忠实度={r.faithfulness:.2f} 弱证据={r.weak_evidence} 拒答={r.degraded} "
        f"缓存命中={r.cache_hit}"
    )
    if args.show_trace:
        print(f"trace_id={r.trace_id}")
        print(json.dumps(r.latency_ms, ensure_ascii=False, indent=2))
    return 0


# ── probe ─────────────────────────────────────────────────────────────


def cmd_probe(args: argparse.Namespace) -> int:
    """探针：重建索引后跑一遍，回答"链路通不通"，不回答"效果好不好"（03 §10）。"""
    from .eval.probe import load_probe_set, run

    cfg = get_config()
    cases = load_probe_set(args.set)
    if not cases:
        print(f"探针集 {args.set} 为空或不存在，退回最小临时探针（06 §7.3）。", file=sys.stderr)
    result = run(cfg, k=args.k, cases=cases or None)
    print(result.as_text())
    if result.failures:
        return 1
    return 0


# ── inspect ───────────────────────────────────────────────────────────


def cmd_inspect(args: argparse.Namespace) -> int:
    cfg = get_config()
    target = Path(args.doc)
    if target.exists():
        res = process_file(target, cfg, move_rejected=False)
        _print_doc_result(res)
        return 0

    # 不是路径就当 doc_id：从物化视图里找
    store = build_store(cfg)
    hits = [c for c in store.all_chunks() if c.doc_id == args.doc]
    if not hits:
        print(f"找不到文档：{args.doc}", file=sys.stderr)
        return 1
    print(f"doc_id={args.doc}  source={hits[0].source}  chunk 数={len(hits)}")
    for c in hits:
        print(f"  {c.chunk_id}  {c.block_type}  {' > '.join(c.title_path)}")
        print(f"    {c.raw_text[:120]}")
    return 0


def _print_doc_result(res) -> None:
    print(f"文件        : {res.path}")
    if res.rejected:
        print(f"处置        : 拒绝入库（{res.reason}）")
        print(f"质量报告    : {json.dumps(res.report.as_dict(), ensure_ascii=False)}")
        return
    doc = res.doc
    print(f"doc_id      : {doc.doc_id}")
    print(f"解析器      : {doc.parser}")
    print(f"页数/字符数 : {doc.quality.page_count} / {doc.quality.char_count}")
    print(f"判定        : {doc.quality.verdict}")
    print(f"问题        : {doc.quality.issues or '（无）'}")
    print(f"不适用项    : {doc.quality.not_applicable or '（无）'}")
    print(f"chunk 数    : {len(res.chunks)}")
    for issue in res.issues:
        print(f"  ! {issue}")


# ── serve ─────────────────────────────────────────────────────────────


def cmd_serve(args: argparse.Namespace) -> int:
    from .server.app import serve

    cfg = get_config()
    serve(cfg, host=args.host, port=args.port)
    return 0


# ── eval ──────────────────────────────────────────────────────────────


def cmd_eval(args: argparse.Namespace) -> int:
    """评测：跑 mini/完整评测集 + 消融变体，产出 EvalReport（04 §9）。

    Mock 模式下报告会带 §9.5 声明：只可引用工程性指标，检索质量结论不可引用。
    """
    from .eval.ablation import get_variants
    from .eval.calibrate import calibrate_floor
    from .eval.report import write_report
    from .eval.runner import load_cases, run_suite
    from .providers.llm import get_llm

    cfg = get_config()
    if args.profile:
        cfg.profile = args.profile
    if cfg.profile == "eval":
        # eval 档的两条硬校验（锁模型版本、关缓存）在 get_config 里只对 env 生效，
        # --profile 显式覆盖后必须重跑校验，否则档位与校验脱节
        from .config import validate_config

        cfg.cache.enabled = False
        validate_config(cfg)

    cases = load_cases(args.qa_set)
    if not cases:
        print(f"评测集为空：{args.qa_set}", file=sys.stderr)
        return 1
    total_cases = len(cases)
    if args.limit:
        # 前 N 条，不抽样——固定子集才能在不同变体之间逐条对比。
        # 抽样会让"这条为什么分数变了"无从查起。
        cases = cases[: args.limit]

    names = [v.strip() for v in args.variants.split(",") if v.strip()] if args.variants else None
    variants = get_variants(names)

    scope = f"{len(cases)}/{total_cases} 条" if args.limit else f"{total_cases} 条"
    print(f"评测集: {args.qa_set}（{scope}）  变体: {', '.join(v.name for v in variants)}")
    print(f"provider_mode: {cfg.provider_mode}")

    if args.dry_run:
        from .eval.cost import estimate_eval, render_report

        print(
            render_report(
                [
                    estimate_eval(
                        len(cases), len(variants), rerank_top_k=cfg.retrieval.rerank_top_k
                    )
                ],
                title="成本预估（--dry-run，未调用任何付费 API）",
            )
        )
        return 0

    judge_llm = get_llm(cfg, "light") if not args.no_judge else None

    report = run_suite(
        cases,
        cfg,
        variants,
        corpus_dir=args.corpus,
        judge_llm=judge_llm,
        repeats=args.repeats,
        progress=lambda s: print(s, flush=True),
        checkpoint=lambda r: write_report(r, args.out),  # 每变体落盘（M23）
    )

    # 先落盘再标定：标定是可选增强，它崩溃不该吞掉整轮结果
    # （M23：2026-10-22 全量消融 390 条跑完后死在标定上，报告从未写盘）。
    raw, summary = write_report(report, args.out)
    if args.calibrate:
        from .bootstrap import build_store

        try:
            store = build_store(cfg)
            report.calibration = calibrate_floor(cases, cfg, store).as_dict()
        except Exception as e:  # noqa: BLE001 - 标定失败只降级，不丢主结果
            note = f"K1 标定未完成（{type(e).__name__}: {e}），检索/生成指标不受影响。"
            report.notes = f"{report.notes} {note}".strip()
            print(f"⚠ {note}", file=sys.stderr)
        else:
            raw, summary = write_report(report, args.out)  # 带标定值重写同一文件
    print()
    print(f"原始结果: {raw}")
    print(f"摘要报告: {summary}")
    if report.unresolved_anchors:
        print(f"⚠ 未解析锚点 {len(report.unresolved_anchors)} 条（详见报告）", file=sys.stderr)
    if report.mock_disclosure:
        print("⚠ Mock 模式：检索与生成质量结论不得对外引用（04 §9.5）", file=sys.stderr)
    return 0


# ── 参数 ──────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentic_rag", description="企业异构文档 Agentic RAG")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("ingest", help="构建索引")
    g.add_argument("--input", default="data/raw")
    g.add_argument("--rebuild", action="store_true", help="（保留位：build 本身即全量重建）")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--keep-rejected", action="store_true", help="不把拒收文件移到隔离区")
    g.add_argument(
        "--incremental",
        action="store_true",
        help="只摄取本次新增的文档，并入既有索引（BM25 仍全量重建，见 index/bm25.py）。"
        "索引不存在或版本不匹配时自动退回全量构建",
    )
    g.add_argument(
        "--dry-run",
        action="store_true",
        help="只估算这一趟要花多少 token / 多少钱，不调用任何付费 API",
    )
    g.set_defaults(func=cmd_ingest)

    a = sub.add_parser("ask", help="单次问答")
    a.add_argument("query")
    a.add_argument("--show-trace", action="store_true")
    a.set_defaults(func=cmd_ask)

    pr = sub.add_parser("probe", help="探针校验链路连通性")
    pr.add_argument("--k", type=int, default=5)
    pr.add_argument("--set", default="eval/probe_set.jsonl", help="探针集文件")
    pr.set_defaults(func=cmd_probe)

    e = sub.add_parser("eval", help="跑评测集与消融变体，产出报告")
    e.add_argument("--qa-set", default="eval/qa_set.jsonl", help="评测集（JSONL）")
    e.add_argument("--variants", default=None, help="逗号分隔的变体名（A0,A1,…,B5）；缺省为全部")
    e.add_argument("--no-judge", action="store_true", help="跳过生成侧 LLM-as-judge")
    e.add_argument("--calibrate", action="store_true", help="顺带标定 sufficiency_vector_floor（K1）")
    e.add_argument("--corpus", default="data/raw", help="语料目录（B2 重建索引用）")
    e.add_argument("--out", default="eval/results", help="报告输出目录")
    e.add_argument("--profile", default=None, help="覆盖 AR_PROFILE（eval 档锁模型版本并关缓存）")
    e.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只跑前 N 条（固定子集，便于逐条对比）。真实 API 下先用小样本"
        "确认链路，再放全量——配合 --dry-run 事前核预算",
    )
    e.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="同配置重复跑 N 次以估 σ_judge（I5 第二级，04 §9.4）。"
        "默认 1 —— 重复跑是 N 倍成本，不该默认发生",
    )
    e.add_argument(
        "--dry-run",
        action="store_true",
        help="只估算这一趟要花多少 token / 多少钱，不调用任何付费 API",
    )
    e.set_defaults(func=cmd_eval)

    i = sub.add_parser("inspect", help="查看解析结果与质量报告")
    i.add_argument("--doc", required=True, help="文件路径或 doc_id")
    i.set_defaults(func=cmd_inspect)

    s = sub.add_parser("serve", help="启动 Web UI")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.set_defaults(func=cmd_serve)

    return p


def _relax_console_encoding() -> None:
    """中文 Windows 控制台默认 cp936，输出里的 `✓` / `✗`（U+2713/U+2717）编码不了。

    后果比"显示成问号"严重：`print` 直接抛 UnicodeEncodeError，整个命令以 traceback 收场，
    而且崩在**最后一行结果输出**上——前面的工作全都白跑。这里只放宽错误处理（不换编码），
    任何字形都不再能让 CLI 崩溃；文件与 HTTP 输出不受影响。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # 非 TextIOWrapper（如被重定向为管道）
            pass


def main(argv: list[str] | None = None) -> int:
    _relax_console_encoding()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except IndexVersionMismatch as e:
        # 两类启动故障里最常见的一种，且它有明确的下一步动作（重建索引）——
        # 让 traceback 打给用户等于把"请先 ingest"这句话藏起来
        print(f"索引不可用：{e}", file=sys.stderr)
        return 2
    except ConfigError as e:
        print(f"配置错误：{e}", file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
