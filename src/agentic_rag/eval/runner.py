"""评测 runner（04 §9.3 / 06 §5.2）。

对文档签名的一处必要偏离：文档写 `run_suite(cases, pipeline, cfg, variants)`，
但 Pipeline 在构造时就绑定了 Config，而消融的每个变体需要不同的 Config
（B2 甚至需要重建索引）。绑死的 pipeline 无法表达"同一语料、不同配置"，
因此改为 `run_suite(cases, cfg, variants)`，内部按变体构造 Pipeline——
仍然全部经 `Pipeline.run` 执行，H11（server 与 eval 共用同一条执行路径）不变。

四个容易做错的地方，这里显式处理：

1. **缓存必须在 runner 里强制关掉**（而不是信任配置）。消融改的就是配置，
   而 `ResultCache` 的键含 retrieval_signature——不关的话 A 组跑完 B 组
   全部命中"同签名旧答案"，只有第一条是真的。
2. **命中判定按"父块家族"规范化**。parent_child 下检索单元是子块、
   rerank 回填后上下文是父块；把两侧 chunk_id 都映射到 `parent_id`
   （无父块则映射到自身）再算指标，否则 Recall 会系统性偏低且不报错。
3. **unanswerable 用例不进 Recall 分母**。它的 golden 本来就是空的，
   混进去等于奖励拒答行为（04 §5.1：答出来反而是失败）。
4. **变体索引不落 `data/index`**。B2 换切分策略需要重建索引，落盘到
   `data/index_eval/{variant}`，不污染基线索引；语料文件不会被移动。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import Config
from ..index.store import IndexStore
from ..pipeline import Pipeline
from ..providers.base import LLMProtocol, AccountArrearage, is_arrearage_error
from ..utils import normalize_text, read_jsonl
from .judge import JudgeVerdict, judge_all
from .metrics import compare_paired, hit, mrr, ndcg_at_k, recall_at_k, repeat_stats
from .report import EvalReport, VariantResult, make_report_skeleton, p50, p95

__all__ = [
    "GoldenAnchor",
    "EvalCase",
    "CaseRecord",
    "VariantSpec",
    "load_cases",
    "resolve_anchors",
    "family_map",
    "patch_config",
    "run_suite",
]


# ─────────────────────────── 数据结构 ───────────────────────────


@dataclass
class GoldenAnchor:
    """文档级锚点，不绑 chunk_id（H14 / 06 §5.2）。

    绑 chunk_id 的标注在 B2 消融（切分策略对比）一跑就全废——chunk_id 是
    切分参数的函数。`title_path + quote` 抗切分变化，且人类可读、可复核。
    """

    source: str  # 文件名（可带子目录，匹配只看文件名部分）
    title_path: list[str] = field(default_factory=list)
    quote: str | None = None  # 用于字符级包含校验
    block_type: str | None = None  # paragraph | table

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenAnchor":
        return cls(
            source=str(d.get("source", "")),
            title_path=list(d.get("title_path") or []),
            quote=d.get("quote"),
            block_type=d.get("block_type"),
        )


@dataclass
class EvalCase:
    question: str
    golden_anchors: list[GoldenAnchor] = field(default_factory=list)
    golden_answer: str | None = None
    intent: str = "simple"
    kind: str = "single_hop"  # single_hop | multi_hop | table | unanswerable

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "golden_anchors": [a.to_dict() for a in self.golden_anchors],
            "golden_answer": self.golden_answer,
            "intent": self.intent,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalCase":
        return cls(
            question=str(d.get("question", "")),
            golden_anchors=[GoldenAnchor.from_dict(a) for a in d.get("golden_anchors") or []],
            golden_answer=d.get("golden_answer"),
            intent=str(d.get("intent", "simple")),
            kind=str(d.get("kind", "single_hop")),
        )


@dataclass
class VariantSpec:
    """一个消融变体：名字 + 对配置树的补丁。"""

    name: str
    patch: dict[str, dict] = field(default_factory=dict)
    note: str = ""


@dataclass
class CaseRecord:
    """单条用例在单个变体下的执行记录（配对比分的原料）。"""

    question: str
    kind: str
    retrieved_canon: list[str] = field(default_factory=list)  # 家族规范化后的融合 Top-K
    golden_canon: list[str] = field(default_factory=list)
    hit5: int = 0
    hit20: int = 0
    recall5: float = 0.0
    recall20: float = 0.0
    mrr_v: float = 0.0
    ndcg5: float = 0.0
    verdict: JudgeVerdict | None = None
    abstained: bool = False
    error: bool = False
    # unanswerable 机器校验的原料（04 §6.3 分工表：unanswerable 判据 = rerank 分数 < floor，
    # 消费方是评测不是在线链路——在线链路的拒答走 corpus_missing 粗筛层）
    top_rerank: float | None = None
    latency_total: float = 0.0
    latency_generate: float = 0.0


# ─────────────────────────── 数据集读写 ───────────────────────────


def load_cases(path: str | Path) -> list[EvalCase]:
    return [EvalCase.from_dict(d) for d in read_jsonl(path)]


# ─────────────────────────── 锚点解析 ───────────────────────────


def _quote_in(quote: str, text: str) -> bool:
    return normalize_text(quote) in normalize_text(text)


def family_map(store: IndexStore) -> dict[str, str]:
    """chunk_id → 家族 id（子块映射到 parent_id，其余映射到自身）。

    这是命中判定的规范化基准：golden 与 retrieved 双方都用它映射后再比较，
    父子两个粒度就永远不会互相 miss。
    """
    store.ensure_loaded()
    return {c.chunk_id: (c.parent_id or c.chunk_id) for c in store.all_chunks()}


def resolve_anchors(case: EvalCase, store: IndexStore) -> tuple[list[str], list[GoldenAnchor]]:
    """三级回退解析锚点（06 §5.2）：source → title_path → quote。

    返回 (golden 家族 id 列表, 解析失败的锚点列表)。
    失败锚点必须由 runner 上报，不得静默跳过——否则 Recall 会被虚高分母稀释。
    """
    store.ensure_loaded()
    units = store.all_chunks()
    family = family_map(store)

    golden: set[str] = set()
    failed: list[GoldenAnchor] = []
    for anchor in case.golden_anchors:
        ids = _match_anchor(anchor, units)
        if not ids:
            failed.append(anchor)
            continue
        golden.update(family.get(cid, cid) for cid in ids)
    return sorted(golden), failed


def _match_anchor(anchor: GoldenAnchor, units: list) -> set[str]:
    """单个锚点 → chunk_id 集合（保留原始粒度，家族化由调用方统一做）。"""
    # 1. source 匹配（锚点允许写纯文件名或带子目录的相对路径）
    name = anchor.source.replace("\\", "/").split("/")[-1]
    pool = [c for c in units if c.source.replace("\\", "/").split("/")[-1] == name]
    if not pool:
        return set()

    # 2. title_path 候选池，从最具体到最宽松：精确 → 前缀 → 全文档。
    #    关键：**回退由 quote 是否还能命中驱动**，不是"精确层存在就锁死"。
    #    早先的写法先按精确层收敛、再套 quote，于是出现了这种故障：
    #    章节标题本身也是一个块（如 `['Agent 和 Function Call']`），它不含锚点的
    #    quote，而真正含 quote 的子节块在 `['Agent 和 Function Call','Agent Factory']`，
    #    结果精确层"命中但无 quote"直接判失败——三级回退被写成了二级就死。
    #    标注是人工写的，层级深浅不一致是常态，这种锚点会静默失效。
    if anchor.title_path:
        tp = list(anchor.title_path)
        pools = [
            [c for c in pool if list(c.title_path) == tp],
            [c for c in pool if list(c.title_path)[: len(tp)] == tp],
            pool,
        ]
    else:
        pools = [pool]

    # 3. quote 字符级包含（raw_text 与 text 都找：块头会改变文本位置）。
    #    给了 quote 却没有任何 chunk 包含它 → 解析失败（06 §5.2 第 3 级）：
    #    退回"该文档全部 chunk"会把锚点稀释成文档级，Recall 虚高且不报错。
    for cand in pools:
        if not cand:
            continue
        if anchor.quote:
            cand = [
                c
                for c in cand
                if _quote_in(anchor.quote, c.raw_text) or _quote_in(anchor.quote, c.text)
            ]
            if not cand:
                continue  # 这一层没有块含该 quote → 回退到更宽松的一层
        if anchor.block_type:
            bt = [c for c in cand if c.block_type == anchor.block_type]
            if bt:
                cand = bt
        return {c.chunk_id for c in cand}
    return set()


# ─────────────────────────── 配置补丁 ───────────────────────────


def patch_config(cfg: Config, patch: dict[str, dict]) -> Config:
    """把 `{"retrieval": {"use_bm25": False}}` 应用到配置树，返回新 Config。"""
    new = cfg.clone()
    for section, kw in (patch or {}).items():
        if section == "profile":
            new.profile = str(kw)
            continue
        obj = getattr(new, section, None)
        if obj is None or not dataclasses.is_dataclass(obj):
            raise ValueError(f"未知的配置段: {section}")
        setattr(new, section, dataclasses.replace(obj, **kw))
    # 评测硬规矩：变体执行期缓存一律关闭（见模块 docstring 第 1 条）
    new.cache = dataclasses.replace(new.cache, enabled=False)
    return new


# ─────────────────────────── 主流程 ───────────────────────────


def run_suite(
    cases: list[EvalCase],
    cfg: Config,
    variants: list[VariantSpec] | None = None,
    *,
    corpus_dir: str | Path = "data/raw",
    judge_llm: LLMProtocol | None = None,
    compare_pairs: list[tuple[str, str]] | None = None,
    repeats: int = 1,
    progress: Callable[[str], None] | None = None,
    checkpoint: Callable[[EvalReport], None] | None = None,
) -> EvalReport:
    """跑完整评测矩阵，返回带齐存证字段的 EvalReport。

    - `checkpoint`：每完成一个变体就回调一次（CLI 用它做增量落盘）。
      真实 API 下一轮消融是几十分钟、真金白银的产物，**不能等到全部
      变体跑完才第一次落盘**——后续变体或可选步骤（K1 标定）崩溃时，
      已完成变体的结果必须已经在盘上（M23：2026-10-22 教训）。
    - 账号欠费（`AccountArrearage`）时立即中止剩余变体：欠费后继续跑
      只会产出降级垃圾指标（检索还能出数但答案全是 fallback），
      截断的变体**不聚合**（用例按固定顺序截断是有偏子集），
      原因写进报告 notes。

    - `variants` 为 None 时只跑基线（名字 base）。
    - 变体的 index_version 与基线不同（如 B2 换切分策略）时自动重建索引
      到 `data/index_eval/`；同版本变体共享一次构建。
    - `judge_llm` 为 None 时跳过生成侧判定（只出检索指标与工程指标）。
    - `repeats` > 1 时同配置重复跑 N 次，用于估 `σ_judge`（I5 第二级，04 §9.4）。
      **默认 1**：重复跑是 N 倍成本，不该默认发生。逐条配对比分（`comparisons`）
      始终取第一轮，两组逐条记录必须同轮才可比。
    """
    from ..bootstrap import build_pipeline
    from ..ingestion.pipeline import load_documents

    if variants is None:
        variants = [VariantSpec(name="base")]
    log = progress or (lambda s: None)
    corpus = Path(corpus_dir)

    report = make_report_skeleton(cfg)
    report.case_count = len(cases)
    report.repeats = max(1, repeats)
    if cfg.provider_mode == "mock":
        report.mock_disclosure = (
            "本报告产生于 Mock 模式：向量路≈字面重合、rerank≈字面重合度，"
            "A1（混合检索）的互补增益被严重低估、A2（rerank）的增益几乎全是噪声。"
            "只可引用工程性指标（延迟、连通性、token 计数、引用链路完整度、探针通过率），"
            "检索与生成质量的绝对值与增益一律不得对外引用（04 §9.5）。"
        )

    per_variant: dict[str, list[CaseRecord]] = {}
    unresolved_total: list[tuple[str, GoldenAnchor]] = []
    stores: dict[str, IndexStore] = {}
    first_pipeline: Pipeline | None = None
    arrearage_note: str | None = None

    for spec in variants:
        vc = patch_config(cfg, spec.patch)
        if vc.index_version != cfg.index_version:
            vc = dataclasses.replace(
                vc, index=dataclasses.replace(vc.index, path=f"data/index_eval/{spec.name}")
            )
        # index_version 只由模型/切分参数决定，改 path 不影响它

        cached_store = stores.get(vc.index_version)
        if cached_store is not None:
            store = cached_store
        elif vc.index_version != cfg.index_version:
            log(f"[{spec.name}] 索引版本变化（{vc.index_version}），重建索引…")
            store = IndexStore(vc)
            chunks, _results = load_documents(corpus, vc, move_rejected=False)
            if not chunks:
                raise RuntimeError(f"变体 {spec.name} 重建索引失败：语料解析后无可入库 chunk")
            store.build(chunks)
            stores[vc.index_version] = store
        else:
            store = IndexStore(vc)
            store.load()  # 版本不匹配会抛 IndexVersionMismatch，带明确的重建指引
            stores[vc.index_version] = store

        pipeline = build_pipeline(vc, store)
        if first_pipeline is None:
            first_pipeline = pipeline  # models_actual 的采集点：跑过之后 last_model 才是响应体真值

        # 锚点逐变体解析：B2 换切分后 chunk 集合不同，锚点必须对各自的索引解析
        resolved: list[tuple[EvalCase, list[str]]] = []
        for case in cases:
            golden, failed = resolve_anchors(case, store)
            for a in failed:
                unresolved_total.append((case.question, a))
            resolved.append((case, golden))

        log(f"[{spec.name}] 开始执行 {len(cases)} 条用例…")
        runs: list[VariantResult] = []
        aborted = False
        for r_i in range(max(1, repeats)):
            if repeats > 1:
                log(f"[{spec.name}] 第 {r_i + 1}/{repeats} 轮（I5：估 σ_judge）…")
            try:
                records = _run_variant(spec.name, resolved, pipeline, store, judge_llm, log)
            except AccountArrearage as e:
                # 欠费是终态：该变体是从固定顺序截断的有偏子集，不聚合；
                # 剩余变体也不再执行。已完成的变体已通过 checkpoint 落盘。
                arrearage_note = f"{spec.name}: {e}"
                log(f"⚠ 账号欠费，中止评测（{e}）")
                aborted = True
                break
            if r_i == 0:
                per_variant[spec.name] = records  # 逐条配对只认第一轮
            runs.append(_aggregate(spec.name, vc, records, pipeline))
        if aborted:
            break
        report.variants.append(runs[0])
        report.repeat_stats.extend(_repeat_stats_of(spec.name, runs))
        if checkpoint is not None:
            checkpoint(report)  # 每个变体完成即落盘（M23）

    report.unresolved_anchors = list(unresolved_total)
    report.models_actual = _collect_models(cfg, first_pipeline)
    report.comparisons = _pairwise(per_variant, variants, compare_pairs)

    notes: list[str] = []
    if judge_llm is None:
        notes.append("未提供 judge_llm：生成侧指标（faithfulness/relevance/citation）未采集。")
    else:
        judge_missing = sum(
            1
            for recs in per_variant.values()
            for rec in recs
            if rec.verdict is not None and rec.verdict.faithfulness is None
        )
        if judge_missing:
            notes.append(f"{judge_missing} 条用例的 judge 未产生可用判定（计为缺失，不填 0 也不填 1）。")
    if unresolved_total:
        notes.append(f"未解析锚点 {len(unresolved_total)} 条，已显式列出；Recall 分母不含它们。")
    if arrearage_note:
        notes.append(
            f"评测因账号欠费中止（{arrearage_note}）。被截断的变体未计入结果——"
            "固定顺序的截断子集有偏，聚合它等于伪造指标；其后的变体未执行。"
        )
    report.notes = " ".join(notes)
    return report


def _run_variant(
    name: str,
    resolved: list[tuple[EvalCase, list[str]]],
    pipeline: Pipeline,
    store: IndexStore,
    judge_llm: LLMProtocol | None,
    log: Callable[[str], None],
) -> list[CaseRecord]:
    family = family_map(store)
    records: list[CaseRecord] = []
    for i, (case, golden) in enumerate(resolved):
        events: list[tuple[str, dict]] = []
        r = pipeline.run(case.question, on_event=lambda t, d: events.append((t, d)))
        # 欠费是终态：继续跑只会产出降级垃圾指标，立即上抛让 run_suite
        # 中止并保住已完成变体（M23）。当前未完成的变体整体作废。
        if r.error and is_arrearage_error(r.error):
            raise AccountArrearage(
                f"第 {i + 1}/{len(resolved)} 条查询被拒（Arrearage），已中止该变体"
            )
        # 取最后一次 retrieve 事件的融合列表：跨轮合并发生在融合处，
        # 最后一轮的列表就是最终候选集（早轮的好候选已被合并且不会丢，L11）
        retrieved: list[str] = []
        for t, d in events:
            if t == "retrieve" and isinstance(d.get("fused"), list):
                retrieved = [str(x) for x in d["fused"]]

        rec = CaseRecord(
            question=case.question,
            kind=case.kind,
            retrieved_canon=[family.get(cid, cid) for cid in retrieved],
            golden_canon=golden,
        )
        rec.error = bool(r.error)
        if not r.error:
            if golden:
                rc, gc = rec.retrieved_canon, rec.golden_canon
                rec.hit5 = hit(rc, gc, 5)
                rec.hit20 = hit(rc, gc, 20)
                rec.recall5 = recall_at_k(rc, gc, 5)
                rec.recall20 = recall_at_k(rc, gc, 20)
                rec.mrr_v = mrr(rc, gc)
                rec.ndcg5 = ndcg_at_k(rc, gc, 5)
            rec.abstained = bool(r.degraded or r.confidence == "low")
            scores = [sc.score for sc in r.context if sc.score is not None]
            rec.top_rerank = max(scores) if scores else None
            if judge_llm is not None:
                rec.verdict = judge_all(
                    case.question,
                    r.answer,
                    [sc.chunk.text for sc in r.context],
                    list(r.citations),
                    judge_llm,
                )
            rec.latency_total = float(r.latency_ms.get("total", 0.0))
            rec.latency_generate = float(r.latency_ms.get("generate", 0.0))
        if (i + 1) % 5 == 0 or i + 1 == len(resolved):
            log(f"[{name}] {i + 1}/{len(resolved)}")
        records.append(rec)
    return records


def _mean(vals: list) -> float | None:
    xs = [float(v) for v in vals if v is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _repeat_stats_of(name: str, runs: list[VariantResult]) -> list[dict]:
    """I5 第二级：同配置重复跑 → 每个指标的 `均值 ± 2σ`（04 §9.4）。

    **只跑一轮时返回空**——没重复跑就不许声称做过显著性检验。
    这是 M21 的由来：`repeat_stats()` / `intervals_overlap()` / `judge_precision_sigma()`
    三个函数此前在 `src/` 里零调用方，I5 第二级只存在于文档里，报告照出、结论照写。

    注意缓存已在 `patch_config` 里强制关闭，否则重复跑会拿到一模一样的答案、
    σ=0，看起来"judge 极其稳定"，实际是缓存回放——这是比没测更糟的假结论。
    """
    if len(runs) < 2:
        return []
    out: list[dict] = []
    for section in ("metrics", "generation"):
        keys = sorted({k for r in runs for k in getattr(r, section, {}) or {}})
        for k in keys:
            vals = [v for v in ((getattr(r, section) or {}).get(k) for r in runs) if v is not None]
            if len(vals) < 2:
                continue
            st = repeat_stats(f"{name}.{k}", vals)
            out.append(
                {
                    "variant": name,
                    "section": section,
                    "metric": k,
                    "mean": st.mean,
                    "std": st.std,
                    "n": st.n,
                    "runs": list(st.runs),
                    "lo": st.lo,
                    "hi": st.hi,
                }
            )
    return out


def _aggregate(name: str, vc: Config, records: list[CaseRecord], pipeline: Pipeline) -> VariantResult:
    scored = [r for r in records if r.golden_canon]  # unanswerable 不进检索指标分母
    unans = [r for r in records if r.kind == "unanswerable"]
    answered = [r for r in records if not r.error and not r.abstained]

    def _opt(d: dict, key: str, val) -> None:
        if val is not None:
            d[key] = val

    metrics: dict[str, float] = {}
    _opt(metrics, "recall@5", _mean([r.recall5 for r in scored]))
    _opt(metrics, "recall@20", _mean([r.recall20 for r in scored]))
    _opt(metrics, "hit_rate", _mean([r.hit5 for r in scored]))
    _opt(metrics, "hit@20", _mean([r.hit20 for r in scored]))
    _opt(metrics, "mrr", _mean([r.mrr_v for r in scored]))
    _opt(metrics, "ndcg@5", _mean([r.ndcg5 for r in scored]))

    generation: dict[str, float] = {}
    judged = [r for r in answered if r.verdict is not None]
    _opt(generation, "faithfulness", _mean([r.verdict.faithfulness for r in judged]))
    _opt(generation, "relevance", _mean([r.verdict.relevance for r in judged]))
    cited = [r for r in judged if r.verdict.citation_accuracy is not None]
    _opt(generation, "citation_accuracy", _mean([r.verdict.citation_accuracy for r in cited]))
    if unans:
        _opt(generation, "abstain_rate", _mean([1.0 if r.abstained else 0.0 for r in unans]))
        _opt(generation, "false_answer_rate", _mean([0.0 if r.abstained else 1.0 for r in unans]))
        # unanswerable 机器校验（04 §6.3）：精排后仍不相关的判据。报均值供与
        # unanswerable_rerank_floor 对比——Mock 的字面重合度无量纲，只看真实模式的数字
        _opt(generation, "unanswerable_top_rerank", _mean([r.top_rerank for r in unans]))

    latency = {
        "p50_total": p50([r.latency_total for r in records if r.latency_total]),
        "p95_total": p95([r.latency_total for r in records if r.latency_total]),
        "p50_generate": p50([r.latency_generate for r in records if r.latency_generate]),
        "p95_generate": p95([r.latency_generate for r in records if r.latency_generate]),
    }

    by_kind: dict[str, dict[str, float]] = {}
    for kind in sorted({r.kind for r in records}):
        sub = [r for r in records if r.kind == kind]
        entry: dict[str, float] = {}
        if any(r.golden_canon for r in sub):
            _opt(entry, "recall@5", _mean([r.recall5 for r in sub if r.golden_canon]))
            _opt(entry, "hit_rate", _mean([r.hit5 for r in sub if r.golden_canon]))
        if kind == "unanswerable":
            _opt(entry, "abstain_rate", _mean([1.0 if r.abstained else 0.0 for r in sub]))
        _opt(entry, "error_rate", _mean([1.0 if r.error else 0.0 for r in sub]))
        by_kind[kind] = entry

    return VariantResult(
        variant=name,
        retrieval_signature=vc.retrieval_signature,
        metrics=metrics,
        generation=generation,
        latency=latency,
        tokens=_usage_of(pipeline),
        by_kind=by_kind,
        errors=sum(1 for r in records if r.error),
    )


def _usage_of(pipeline: Pipeline) -> dict[str, int]:
    """从 provider 的累计计数器汇总 token 用量（评测期内的净增量）。"""
    deps = pipeline.deps
    out: dict[str, int] = {}
    for attr in ("llm_light", "llm_main", "embedder", "reranker"):
        p = getattr(deps, attr, None)
        total = getattr(p, "usage_total", None) or {}
        for k, v in total.items():
            out[f"{attr}.{k}"] = int(v)
    out["tokens_total"] = sum(out.values())
    return out


def _collect_models(cfg: Config, pipeline: Pipeline | None) -> dict[str, str]:
    """`models_actual`：响应体返回的实际模型名（08 §1.1：请求值可能是会漂移的别名）。

    从 suite 里用过的 provider 上读 `last_model`——它只在真调过 API 后才被
    响应体更新，所以这里拿到的就是"这轮评测实际跑在什么模型上"。
    """
    fallback = {
        "llm_main": cfg.dashscope.llm_main,
        "llm_light": cfg.dashscope.llm_light,
        "embedding": cfg.dashscope.embedding_model,
        "rerank": cfg.dashscope.rerank_model,
    }
    if pipeline is None:
        return fallback
    deps = pipeline.deps
    out = dict(fallback)
    for key, attr in (
        ("llm_main", "llm_main"),
        ("llm_light", "llm_light"),
        ("embedding", "embedder"),
        ("rerank", "reranker"),
    ):
        p = getattr(deps, attr, None)
        for a in ("last_model", "model_name", "model"):
            v = getattr(p, a, None)
            if v:
                out[key] = str(v)
                break
    return out


def _pairwise(
    per_variant: dict[str, list[CaseRecord]],
    variants: list[VariantSpec],
    compare_pairs: list[tuple[str, str]] | None,
) -> list:
    """I5 第一级：逐条配对比分。默认比相邻变体（A0 vs A1、A1 vs A2 …）。

    recall@K 按文档取该条是否命中（0/1）；faithfulness 取该条分数，
    任一端裁判未判（None）的条目不参与——把 None 当 0 会把"没判成"
    混进"判了且不支持"，两者的含义完全不同。
    """
    names = [v.name for v in variants]
    pairs = compare_pairs if compare_pairs is not None else [
        (names[i], names[i + 1]) for i in range(len(names) - 1)
    ]
    out = []
    for x, y in pairs:
        if x not in per_variant or y not in per_variant:
            continue
        rx = {r.question: r for r in per_variant[x]}
        ry = {r.question: r for r in per_variant[y]}
        common = [q for q in rx if q in ry and rx[q].golden_canon]
        out.append(
            compare_paired(
                f"{x}_vs_{y}/hit@5",
                [float(rx[q].hit5) for q in common],
                [float(ry[q].hit5) for q in common],
            )
        )
        fq = [
            q
            for q in common
            if rx[q].verdict
            and ry[q].verdict]
        xs, ys = [], []
        for q in fq:
            fx, fy = rx[q].verdict.faithfulness, ry[q].verdict.faithfulness
            if fx is not None and fy is not None:
                xs.append(fx)
                ys.append(fy)
        if xs:
            out.append(compare_paired(f"{x}_vs_{y}/faithfulness", xs, ys))
    return out
