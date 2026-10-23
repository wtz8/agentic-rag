"""评测层（04 §9 / 03 §10）：指标、显著性判据、judge、锚点解析、报告。"""

from __future__ import annotations

import json

import pytest

from agentic_rag.eval import ablation, calibrate, judge, metrics, probe, report, runner
from agentic_rag.index.store import IndexStore
from agentic_rag.schema import Chunk, Citation


# ─────────────────────────── metrics ───────────────────────────


class TestMetrics:
    def test_recall_at_k(self):
        retrieved = ["a", "b", "c", "d", "e", "f"]
        assert metrics.recall_at_k(retrieved, ["a", "x"], 5) == 0.5
        assert metrics.recall_at_k(retrieved, ["a"], 5) == 1.0
        assert metrics.recall_at_k(retrieved, ["x"], 5) == 0.0

    def test_recall_empty_golden_is_zero_not_one(self):
        # unanswerable 用例的 golden 为空：返回 1.0 会把"拒答"伪装成"全召回"
        assert metrics.recall_at_k(["a"], [], 5) == 0.0

    def test_hit(self):
        assert metrics.hit(["a", "b"], ["b"], 5) == 1
        assert metrics.hit(["a", "b"], ["c"], 5) == 0
        assert metrics.hit(["a", "b"], ["b"], 1) == 0  # K 截断

    def test_precision_at_k(self):
        assert metrics.precision_at_k(["a", "b", "c"], ["a"], 2) == 0.5

    def test_mrr(self):
        assert metrics.mrr(["x", "a", "b"], ["a"]) == 0.5
        assert metrics.mrr(["x", "y"], ["a"]) == 0.0

    def test_ndcg_ideal_ordering_is_one(self):
        assert metrics.ndcg_at_k(["a", "b"], ["a", "b"], 5) == 1.0
        # 二值相关性下，命中集合相同只是顺序不同，NDCG 相同——这是正确行为；
        # 顺序敏感体现在 K 截断上：第一名命中 vs 第二名才命中
        assert metrics.ndcg_at_k(["b", "a"], ["a"], 1) == 0.0
        assert metrics.ndcg_at_k(["a", "b"], ["a"], 1) == 1.0
        assert metrics.ndcg_at_k(["x"], ["a"], 5) == 0.0


class TestSignificance:
    """I5 定案的符号检验临界值（04 §9.4）：N=20→15、N=25→18。实现必须与文档同源。"""

    @pytest.mark.parametrize("n_consistent,n_total,significant", [
        (15, 20, True),
        (14, 20, False),
        (18, 25, True),
        (17, 25, False),
        (20, 20, True),
    ])
    def test_sign_test_critical_values(self, n_consistent, n_total, significant):
        p = metrics.sign_test_p_value(n_consistent, n_total)
        assert (p < 0.05) is significant, f"p={p}"

    def test_compare_paired_clear_win(self):
        xs = [1.0] * 18 + [0.0, 0.0]
        ys = [0.0] * 20
        c = metrics.compare_paired("m", xs, ys)
        assert c.win == 18 and c.loss == 0
        assert c.net_win_rate == 0.9
        assert c.verdict == metrics.VERDICT_DIFFERENT

    def test_compare_paired_ties_dont_count(self):
        xs = [0.5] * 20  # 与 y 全部相等 → 全 tie
        c = metrics.compare_paired("m", xs, [0.5] * 20)
        assert c.tie == 20 and c.verdict == metrics.VERDICT_NOISE

    def test_compare_paired_noise_band(self):
        # net_win_rate = 2/20 = 0.1 → 恰在噪声带内（< 0.10 的边界语义：等于不算噪声带上限）
        xs = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        c = metrics.compare_paired("m", xs, [0] * 20)
        assert abs(c.net_win_rate) == 0.1

    def test_repeat_stats_and_overlap(self):
        a = metrics.repeat_stats("m", [0.5, 0.5, 0.5])
        b = metrics.repeat_stats("m", [0.9, 0.9, 0.9])
        assert a.std == 0.0
        assert not metrics.intervals_overlap(a, b)
        c = metrics.repeat_stats("m", [0.48, 0.52])  # σ=0.028，2σ 区间覆盖 0.5
        assert metrics.intervals_overlap(a, c)


# ─────────────────────────── judge ───────────────────────────


class ScriptedJudge:
    """按 judge 模板里的 schema 关键词返回固定 JSON（05 §3）。"""

    def __init__(self, supported=None, unsupported=None, score=None, correct=None, wrong=None):
        self.supported = supported or []
        self.unsupported = unsupported or []
        self.score = score
        self.correct = correct or []
        self.wrong = wrong or []

    @property
    def model_name(self):
        return "scripted-judge"

    def chat(self, messages, temperature=0.0, max_tokens=2048, json_mode=False):
        user = messages[-1]["content"].lower()
        if "supported" in user:
            return json.dumps({"supported": self.supported, "unsupported": self.unsupported})
        if "correct" in user:
            return json.dumps({"correct": self.correct, "wrong": self.wrong})
        if "score" in user:
            return json.dumps({"score": self.score if self.score is not None else 5})
        return json.dumps({})

    def stream(self, messages, temperature=0.0, max_tokens=2048):
        yield ""


ANSWER = "华东区单价为 3.2 万元/Gbps[1]。华北区单价为 2.8 万元/Gbps[2]。"
CONTEXT = ["华东区 2024 年带宽采购单价为 3.2 万元/Gbps。", "无关文本。"]


class TestJudge:
    def test_faithfulness_full_support(self):
        llm = ScriptedJudge(supported=[0, 1])
        f, bad = judge.judge_faithfulness(ANSWER, CONTEXT, llm)
        assert f == 1.0 and bad == []

    def test_faithfulness_partial(self):
        llm = ScriptedJudge(supported=[0], unsupported=[1])
        f, bad = judge.judge_faithfulness(ANSWER, CONTEXT, llm)
        assert f == 0.5
        assert len(bad) == 1 and "华北" in bad[0]

    def test_faithfulness_all_empty_is_none_not_one(self):
        # Mock 的满分结构（全空）必须判为"未作答"，不能当成"全部支持"
        llm = ScriptedJudge()
        f, _ = judge.judge_faithfulness(ANSWER, CONTEXT, llm)
        assert f is None

    def test_faithfulness_empty_answer(self):
        f, bad = judge.judge_faithfulness("", CONTEXT, ScriptedJudge(supported=[0]))
        assert f is None and bad == []

    def test_relevance_normalization(self):
        assert judge.judge_relevance("q", "a", ScriptedJudge(score=5)) == 1.0
        assert judge.judge_relevance("q", "a", ScriptedJudge(score=3)) == 0.5
        assert judge.judge_relevance("q", "a", ScriptedJudge(score=1)) == 0.0

    def test_relevance_out_of_range(self):
        assert judge.judge_relevance("q", "a", ScriptedJudge(score=9)) is None

    def test_citation_accuracy(self):
        cits = [
            Citation(index=1, chunk_id="a", source="s", title_path=[], page=1, quote="q", score=1.0),
            Citation(index=2, chunk_id="b", source="s", title_path=[], page=2, quote="q", score=1.0),
            Citation(index=3, chunk_id="c", source="s", title_path=[], page=3, quote="q", score=1.0),
        ]
        acc = judge.judge_citation_accuracy(cits, ANSWER, ScriptedJudge(correct=[0, 1], wrong=[2]))
        assert acc == pytest.approx(2 / 3, abs=1e-3)

    def test_citation_no_citations_is_none(self):
        assert judge.judge_citation_accuracy([], "a", ScriptedJudge(correct=[0])) is None

    def test_judge_all_aggregates(self):
        llm = ScriptedJudge(supported=[0, 1], score=4, correct=[0], wrong=[])
        cits = [Citation(index=1, chunk_id="a", source="s", title_path=[], page=1, quote="q", score=1.0)]
        v = judge.judge_all("q", ANSWER, CONTEXT, cits, llm)
        assert v.faithfulness == 1.0
        assert v.relevance == 0.75
        assert v.citation_accuracy == 1.0


# ─────────────────────────── 锚点解析与配置补丁 ───────────────────────────


def _chunk(cid, source, title_path, text, block_type="paragraph", parent_id=None):
    return Chunk(
        chunk_id=cid,
        doc_id=cid.split(":")[0],
        parent_id=parent_id,
        text=text,
        raw_text=text,
        title_path=list(title_path),
        block_type=block_type,
        source=source,
    )


ANCHOR_CHUNKS = [
    _chunk("d1:c0000", "report.pdf", ["第三章", "3.2 明细"], "华东区 2024 年带宽采购单价为 3.2 万元/Gbps。", parent_id="d1:p000"),
    _chunk("d1:c0001", "report.pdf", ["第三章", "3.3 其他"], "设备 XR-2400 支持双电源冗余。", parent_id="d1:p001"),
    _chunk("d1:c0002", "report.pdf", ["附录"], "| 区域 | 单价 |\n| --- |\n| 华东 | 3.2 |", block_type="table", parent_id="d1:p002"),
    _chunk("d2:c0000", "spec.html", [], "错误码 E0107 表示光模块收光功率低于阈值。"),
]


@pytest.fixture
def anchor_store(cfg, tmp_path):
    cfg.index.path = str(tmp_path / "idx_anchor")
    cfg.index.backend = "memory"
    st = IndexStore(cfg)
    # 父块也写入（family 化需要 parent 存在于物化视图）
    parents = [
        _chunk("d1:p000", "report.pdf", ["第三章", "3.2 明细"], "…父块全文…"),
        _chunk("d1:p001", "report.pdf", ["第三章", "3.3 其他"], "…父块全文…"),
        _chunk("d1:p002", "report.pdf", ["附录"], "…父块全文…"),
    ]
    allc = ANCHOR_CHUNKS + parents
    for c in parents:
        c.is_parent = True
    st.build(allc)
    return st


class TestResolveAnchors:
    def test_title_path_exact(self, anchor_store):
        case = runner.EvalCase(
            question="q",
            golden_anchors=[runner.GoldenAnchor(source="report.pdf", title_path=["第三章", "3.2 明细"])],
        )
        golden, failed = runner.resolve_anchors(case, anchor_store)
        assert not failed
        assert golden == ["d1:p000"]  # 家族化：子块 → 父块 id

    def test_title_path_prefix_fallback(self, anchor_store):
        case = runner.EvalCase(
            question="q",
            golden_anchors=[runner.GoldenAnchor(source="report.pdf", title_path=["第三章"])],
        )
        golden, failed = runner.resolve_anchors(case, anchor_store)
        assert not failed
        assert set(golden) == {"d1:p000", "d1:p001"}

    def test_quote_fallback(self, anchor_store):
        # title_path 没有精确命中时，quote 做字符级包含校验收窄到目标 chunk
        case = runner.EvalCase(
            question="q",
            golden_anchors=[runner.GoldenAnchor(source="report.pdf", quote="E0107 了吗")],
        )
        golden, failed = runner.resolve_anchors(case, anchor_store)
        # report.pdf 里没有 E0107 → 解析失败（不是静默跳过）
        assert golden == [] and len(failed) == 1
        case2 = runner.EvalCase(
            question="q",
            golden_anchors=[runner.GoldenAnchor(source="spec.html", quote="E0107")],
        )
        golden2, failed2 = runner.resolve_anchors(case2, anchor_store)
        assert not failed2 and golden2 == ["d2:c0000"]

    def test_source_with_dir_prefix(self, anchor_store):
        case = runner.EvalCase(
            question="q",
            golden_anchors=[runner.GoldenAnchor(source="pdf/report.pdf")],
        )
        golden, failed = runner.resolve_anchors(case, anchor_store)
        assert not failed and golden

    def test_block_type_filter(self, anchor_store):
        case = runner.EvalCase(
            question="q",
            golden_anchors=[runner.GoldenAnchor(source="report.pdf", block_type="table")],
        )
        golden, failed = runner.resolve_anchors(case, anchor_store)
        assert golden == ["d1:p002"]

    def test_unresolved_reported_not_skipped(self, anchor_store):
        case = runner.EvalCase(
            question="q",
            golden_anchors=[
                runner.GoldenAnchor(source="不存在.pdf"),
                runner.GoldenAnchor(source="report.pdf", title_path=["第三章", "3.2 明细"]),
            ],
        )
        golden, failed = runner.resolve_anchors(case, anchor_store)
        assert len(failed) == 1 and failed[0].source == "不存在.pdf"
        assert golden == ["d1:p000"]


class TestPatchConfig:
    def test_applies_patch_and_disables_cache(self, cfg):
        vc = runner.patch_config(cfg, {"retrieval": {"use_bm25": False}})
        assert vc.retrieval.use_bm25 is False
        assert cfg.retrieval.use_bm25 is True  # 原 cfg 不被改
        assert vc.cache.enabled is False  # 评测硬规矩

    def test_unknown_section_raises(self, cfg):
        with pytest.raises(ValueError):
            runner.patch_config(cfg, {"no_such_section": {}})


# ─────────────────────────── ablation ───────────────────────────


class TestAblation:
    def test_variant_table(self):
        assert ablation.get_variant("A0").patch["retrieval"]["use_bm25"] is False
        assert ablation.get_variant("B2").patch["ingestion"]["chunk_strategy"] == "structural"
        assert ablation.get_variant("B3").patch["retrieval"]["parent_backfill"] is False

    def test_a_group_is_a_strict_cumulative_chain(self, cfg):
        """A 组累加链的硬约束（M25）：每个 A_k 在 A_{k-1} 基础上恰好多开一个
        组件，且相邻变体 retrieval_signature 必须可区分——曾经 A2 的
        use_rerank=True 是对默认值的空操作，"A2 = A1 + Rerank"从未被真正测过。"""
        names = ["A0", "A1", "A2", "A3", "A4"]
        sigs = {}
        for k, name in enumerate(names):
            vc = runner.patch_config(cfg, ablation.get_variant(name).patch)
            sigs[name] = vc.retrieval_signature
            if k:
                assert sigs[name] != sigs[names[k - 1]], f"{name} 与 {names[k-1]} 不可区分"
        r = [
            runner.patch_config(cfg, ablation.get_variant(n).patch).retrieval for n in names[:3]
        ]
        assert (r[0].use_vector, r[0].use_bm25, r[0].use_rerank) == (True, False, False)
        assert (r[1].use_vector, r[1].use_bm25, r[1].use_rerank) == (True, True, False)
        assert (r[2].use_vector, r[2].use_bm25, r[2].use_rerank) == (True, True, True)

    def test_b2_changes_index_version(self, cfg):
        vc = runner.patch_config(cfg, ablation.get_variant("B2").patch)
        assert vc.index_version != cfg.index_version

    def test_unknown_variant(self):
        with pytest.raises(KeyError):
            ablation.get_variant("Z9")


# ─────────────────────────── report ───────────────────────────


class TestReport:
    def test_write_report(self, cfg, tmp_path):
        rep = report.make_report_skeleton(cfg)
        rep.variants.append(
            report.VariantResult(
                variant="A0",
                retrieval_signature="abc",
                metrics={"recall@5": 0.5},
                generation={"faithfulness": 0.9},
            )
        )
        rep.unresolved_anchors = [("q", runner.GoldenAnchor(source="x.pdf"))]
        raw, summary = report.write_report(rep, tmp_path / "results")
        assert raw.exists() and summary.exists()
        text = summary.read_text(encoding="utf-8")
        # 四条存证字段必须出现在摘要里
        for key in ("provider_mode", "retrieval_signature", "user_dict_sha256", "models_actual"):
            assert key in text
        assert "未解析锚点" in text
        data = json.loads(raw.read_text(encoding="utf-8"))
        assert data["case_count"] == 0
        assert data["unresolved_anchors"][0]["anchor"]["source"] == "x.pdf"

    def test_quantiles(self):
        assert report.p50([1, 2, 3, 4]) == 2.5
        assert report.p95([]) == 0.0


# ─────────────────────────── probe ───────────────────────────


class TestProbe:
    def test_run_hits(self, cfg, anchor_store):
        cases = [
            probe.ProbeCase(
                question="错误码 E0107",
                expect_source="spec.html",
                expect_answer_contains="E0107",
            )
        ]
        result = probe.run(cfg, k=5, store=anchor_store, cases=cases)
        assert result.total == 1 and result.hit == 1
        probe.assert_healthy(result)

    def test_run_miss_reports_failure(self, cfg, anchor_store):
        # 问题命中 report.pdf 的内容，但期望来源是 spec.html → 未命中。
        # k=1：只有 4 个 chunk，k 太大时 Top-K 永远包含所有来源，测不出 miss
        cases = [probe.ProbeCase(question="带宽采购单价是多少", expect_source="spec.html")]
        result = probe.run(cfg, k=1, store=anchor_store, cases=cases)
        assert result.hit == 0
        assert result.failures
        with pytest.raises(AssertionError):
            probe.assert_healthy(result, min_hit_rate=0.99)

    def test_title_path_mismatch_is_miss(self, cfg, anchor_store):
        cases = [
            probe.ProbeCase(
                question="带宽采购",
                expect_source="report.pdf",
                expect_title_path=["第十章"],
            )
        ]
        result = probe.run(cfg, k=5, store=anchor_store, cases=cases)
        assert result.hit == 0


# ─────────────────────────── calibrate ───────────────────────────


class TestCalibrate:
    def test_suggest_floor_separates_distributions(self):
        pts = [calibrate.ScorePoint(0.6 + i * 0.01, True) for i in range(10)]
        pts += [calibrate.ScorePoint(0.1 + i * 0.01, False) for i in range(10)]
        cal = calibrate.suggest_floor(pts)
        assert cal.floor is not None
        assert 0.19 <= cal.floor <= 0.61
        assert cal.tpr == 1.0 and cal.fpr == 0.0
        assert cal.auc == 1.0

    def test_no_discrimination_warns(self):
        cal = calibrate.suggest_floor([calibrate.ScorePoint(0.5, True)] * 5)
        assert cal.floor is None
        assert "无法标定" in cal.warning

    def test_mock_mode_refuses(self, cfg, anchor_store):
        # cfg 无 Key → provider_mode == mock：必须拒绝标定而不是给一个假阈值
        cal = calibrate.calibrate_floor([], cfg, anchor_store)
        assert cal.floor is None
        assert "不标定" in cal.warning


class TestMatchAnchorFallback:
    """回归：三级回退必须"由 quote 是否还能命中"驱动，不能"精确层存在就锁死"。

    真实故障：`opensource_langchain-chatchat.md` 里 `['Agent 和 Function Call']`
    本身也是一个块（章节导语，不含锚点 quote），而含 quote 的块在
    `['Agent 和 Function Call', 'Agent Factory']`。早先先按精确层收敛再套 quote，
    于是"精确层命中但无 quote"直接判失败——标注写得比实际层级浅一点就静默失效。
    """

    @pytest.fixture
    def shadow_store(self, cfg, tmp_path):
        cfg.index.path = str(tmp_path / "idx_shadow")
        cfg.index.backend = "memory"
        st = IndexStore(cfg)
        st.build(
            [
                _chunk("s1:c0000", "guide.md", ["Ch"], "章节导语，不含目标词。"),
                _chunk("s1:c0001", "guide.md", ["Ch", "Sub"], "小节正文：Agent Factory 含目标词。"),
            ]
        )
        return st

    def test_exact_level_without_quote_falls_back_to_prefix(self, shadow_store):
        store = shadow_store
        units = store.all_chunks()
        a = runner.GoldenAnchor(source="guide.md", title_path=["Ch"], quote="Agent Factory")
        assert runner._match_anchor(a, units) == {"s1:c0001"}

    def test_exact_level_with_quote_still_wins(self, shadow_store):
        """能精确命中就别放宽——回退不是无脑取最宽松的一层。"""
        units = shadow_store.all_chunks()
        a = runner.GoldenAnchor(source="guide.md", title_path=["Ch", "Sub"], quote="Agent Factory")
        assert runner._match_anchor(a, units) == {"s1:c0001"}

    def test_quote_missing_everywhere_is_a_failure_not_a_doc_fallback(self, shadow_store):
        """quote 在所有层级都没有 → 解析失败，不许退回整个文档（06 §5.2 第 3 级）。"""
        units = shadow_store.all_chunks()
        a = runner.GoldenAnchor(source="guide.md", title_path=["Ch"], quote="根本不存在的词")
        assert runner._match_anchor(a, units) == set()


class TestI5SecondLevel:
    """I5 第二级：同配置重复跑估 σ_judge（M21）。

    M21 的教训是"函数和单测都齐了，但没人调用它"——I5 第二级只存在于文档里。
    所以这里的测试**必须跑 run_suite 的入口**，而不是只测 `_repeat_stats_of`。
    """

    def test_single_run_produces_no_repeat_stats(self):
        """只跑一轮就不得声称做过显著性检验——空列表，不是假的 σ=0。"""
        out = runner._repeat_stats_of("A0", [_vr({"recall@20": 0.7})])
        assert out == []

    def test_repeated_runs_report_spread(self):
        a = _vr({"recall@20": 0.70}, {"faithfulness": 0.90})
        b = _vr({"recall@20": 0.74}, {"faithfulness": 0.80})
        c = _vr({"recall@20": 0.66}, {"faithfulness": 0.85})
        out = runner._repeat_stats_of("A0", [a, b, c])
        by = {o["metric"]: o for o in out}
        assert by["recall@20"]["n"] == 3
        assert by["recall@20"]["mean"] == pytest.approx(0.70)
        assert by["recall@20"]["std"] > 0, "有波动就必须报出来，σ=0 才可疑"
        assert by["recall@20"]["lo"] < by["recall@20"]["hi"]
        # generation 段也要覆盖：judge 的波动才是 I5 关心的对象
        assert by["faithfulness"]["section"] == "generation"

    def test_missing_metric_in_one_run_is_skipped_not_zeroed(self):
        """某一轮某个指标缺失 → 跳过，不填 0（填 0 会把 σ 撑大、均值拉低）。"""
        out = runner._repeat_stats_of("A0", [_vr({"recall@20": 0.7}), _vr({})])
        assert out == []

    def test_run_suite_defaults_to_single_run(self):
        """默认 repeats=1：重复跑是 N 倍成本，不该默认发生。"""
        import inspect

        sig = inspect.signature(runner.run_suite)
        assert sig.parameters["repeats"].default == 1


def _vr(metrics: dict, generation: dict | None = None) -> object:
    from agentic_rag.eval.report import VariantResult

    return VariantResult(variant="A0", retrieval_signature="x", metrics=metrics, generation=generation or {})


# ───────────────────── M23：欠费判定与 judge 上抛 ─────────────────────


class TestArrearage:
    def test_markers(self):
        from agentic_rag.providers.base import is_arrearage_error

        assert is_arrearage_error('400: {"type":"Arrearage"}')
        assert is_arrearage_error("Access denied, account not in good standing")
        assert is_arrearage_error(ValueError("see overdue-payment for details"))
        assert not is_arrearage_error("ProviderTimeoutError: timeout")
        assert not is_arrearage_error("")
        assert not is_arrearage_error(None)

    def test_judge_reraises_arrearage_not_swallows(self):
        """欠费时 judge 必须上抛而不是返回 None——吞掉它会让整轮
        judge 静默变"未作答"，报告里只剩一句无害的 warning。"""
        from agentic_rag.eval.judge import judge_faithfulness, judge_relevance
        from agentic_rag.providers.base import AccountArrearage, ProviderResponseError

        class BadLLM:
            def chat(self, *a, **k):
                raise ProviderResponseError('请求错误 400: {"type":"Arrearage"}')

        with pytest.raises(AccountArrearage):
            judge_faithfulness("这是答案。第二句。", ["上下文一", "上下文二"], BadLLM())
        with pytest.raises(AccountArrearage):
            judge_relevance("问题", "答案", BadLLM())

    def test_judge_swallows_ordinary_errors(self):
        """普通错误仍按原语义吞掉（裁判失败不让整轮评测崩）。"""
        from agentic_rag.eval.judge import judge_faithfulness
        from agentic_rag.providers.base import ProviderResponseError

        class TimeoutLLM:
            def chat(self, *a, **k):
                raise ProviderResponseError("请求错误 504: timeout")

        faith, bad = judge_faithfulness("这是答案。第二句。", ["上下文一", "上下文二"], TimeoutLLM())
        assert faith is None and bad == []
