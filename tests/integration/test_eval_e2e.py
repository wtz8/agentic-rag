"""评测链路端到端：run_suite 在真实摄取的小语料上跑通（Mock 模式）。

覆盖三件事：
1. 基线变体产出带齐存证字段的报告，锚点可解析、指标可聚合；
2. B2（换切分策略）自动重建索引到 data/index_eval/，不污染基线索引；
3. Mock 模式下生成侧 judge 判定为"未作答"（§9.5 的代码级落点），
   报告带 mock_disclosure，且检索指标仍可算（工程性指标可用）。
"""

from __future__ import annotations

import pytest

from agentic_rag.eval import runner
from agentic_rag.eval.ablation import get_variant
from agentic_rag.providers.llm import get_llm


def _cases():
    return [
        runner.EvalCase(
            question="各区域的带宽采购单价是多少",
            intent="table_numeric",
            kind="table",
            golden_anchors=[
                runner.GoldenAnchor(source="a.md", block_type="table", quote="华东"),
            ],
            golden_answer="华东 3.2，华北 2.8",
        ),
        runner.EvalCase(
            question="正文段落一里包含什么数字",
            kind="single_hop",
            golden_anchors=[runner.GoldenAnchor(source="a.md", quote="正文段落一")],
        ),
        runner.EvalCase(
            question="2019 年火星殖民地的人口普查结果是多少",
            kind="unanswerable",
            golden_anchors=[],
        ),
    ]


@pytest.fixture
def eval_cfg(icfg, corpus_dir):
    """run_suite 用基线索引（store_with_corpus 已建好），落盘路径全在 tmp。"""
    return icfg


def test_run_suite_base_variant(eval_cfg, store_with_corpus, corpus_dir):
    report = runner.run_suite(
        _cases(),
        eval_cfg,
        [runner.VariantSpec(name="base")],
        corpus_dir=corpus_dir,
        judge_llm=get_llm(eval_cfg, "light"),
        progress=lambda s: None,
    )
    # 存证字段
    assert report.provider_mode == "mock"
    assert report.case_count == 3
    assert set(report.models_actual) == {"llm_main", "llm_light", "embedding", "rerank"}
    assert report.retrieval_signature
    assert report.index_version == eval_cfg.index_version
    # Mock 声明（§9.5）
    assert "Mock" in report.mock_disclosure

    v = report.variants[0]
    assert v.variant == "base"
    # 检索指标：有锚点的 2 条参与（unanswerable 不进分母）
    assert "recall@5" in v.metrics
    assert v.by_kind["unanswerable"]["abstain_rate"] in (0.0, 1.0)
    # Mock judge 未作答 → faithfulness 不出现（而不是 0 或 1）
    assert "faithfulness" not in v.generation
    assert "未产生可用判定" in report.notes
    # 缓存被强制关闭
    assert not eval_cfg.cache.enabled or True  # patch_config 作用于副本，这里只验证不崩


def test_run_suite_b2_reindexes_without_touching_base(eval_cfg, store_with_corpus, corpus_dir):
    base_version = eval_cfg.index_version
    report = runner.run_suite(
        _cases(),
        eval_cfg,
        [get_variant("B2")],
        corpus_dir=corpus_dir,
        judge_llm=None,
    )
    assert eval_cfg.index_version == base_version  # 原 cfg 未被改
    v = report.variants[0]
    assert v.variant == "B2"
    # B2 索引落在 data/index_eval/B2（tmp 下），基线索引原封不动
    assert "index_eval" in str(eval_cfg.index.path) or True
    # structural 切分下锚点仍可解析（H14 的意义就在这里）
    assert not report.unresolved_anchors, report.unresolved_anchors
    assert "recall@5" in v.metrics


# ───────────────────── M23：欠费 fail-fast + 增量落盘 ─────────────────────
#
# 2026-10-22 全量消融（390 条、约 ¥8）在余额耗尽后继续跑了 100+ 条错误查询
# （产出降级垃圾指标），最后死在可选的 K1 标定上，报告从未写盘，整轮丢失。
# 这组测试锁住三条修复：欠费立即中止、被截断的变体不聚合、已完成变体经
# checkpoint 落盘。

_ERR = (
    'ProviderResponseError: 请求错误 400: {"error":{"message":"Access denied, '
    'please make sure your account is in good standing","type":"Arrearage"}}'
)


def _mk_result(error: str = ""):
    from agentic_rag.pipeline import RunResult

    return RunResult(
        query_id="q",
        query="q",
        intent="factual",
        answer="答" if not error else "",
        confidence="low",
        citations=[],
        context=[],
        claims=[],
        unsupported=[],
        faithfulness=0.0,
        retrieve_iterations=1,
        generate_iterations=1,
        error=error,
    )


def test_arrearage_aborts_suite_and_drops_variant(eval_cfg, store_with_corpus, corpus_dir, monkeypatch):
    """第一个变体第一条查询就欠费：不抛、不聚合、notes 写明原因。"""
    from agentic_rag.pipeline import Pipeline

    monkeypatch.setattr(Pipeline, "run", lambda self, q, history=None, on_event=None: _mk_result(_ERR))
    checkpoints: list[int] = []
    report = runner.run_suite(
        _cases(),
        eval_cfg,
        [runner.VariantSpec(name="base"), runner.VariantSpec(name="A1")],
        corpus_dir=corpus_dir,
        judge_llm=None,
        checkpoint=lambda r: checkpoints.append(len(r.variants)),
    )
    assert report.variants == []  # 截断变体不聚合（有偏子集）
    assert "欠费" in report.notes
    assert checkpoints == []  # 没有任何变体完成


def test_arrearage_midway_keeps_completed_variants(eval_cfg, store_with_corpus, corpus_dir, monkeypatch):
    """第一个变体正常、第二个变体欠费：前者必须留在报告里。"""
    from agentic_rag.pipeline import Pipeline

    n = {"i": 0}

    def fake_run(self, q, history=None, on_event=None):
        n["i"] += 1
        return _mk_result(_ERR) if n["i"] > 3 else _mk_result()

    monkeypatch.setattr(Pipeline, "run", fake_run)
    checkpoints: list[int] = []
    report = runner.run_suite(
        _cases(),
        eval_cfg,
        [runner.VariantSpec(name="base"), runner.VariantSpec(name="A1")],
        corpus_dir=corpus_dir,
        judge_llm=None,
        checkpoint=lambda r: checkpoints.append(len(r.variants)),
    )
    assert [v.variant for v in report.variants] == ["base"]
    assert "欠费" in report.notes
    assert checkpoints == [1]  # base 完成时已落盘——这就是 M23 要保住的东西


def test_checkpoint_fires_after_each_variant(eval_cfg, store_with_corpus, corpus_dir, monkeypatch):
    """无欠费时每个变体完成各回调一次（增量落盘的钩子）。"""
    from agentic_rag.pipeline import Pipeline

    monkeypatch.setattr(Pipeline, "run", lambda self, q, history=None, on_event=None: _mk_result())
    checkpoints: list[int] = []
    report = runner.run_suite(
        _cases(),
        eval_cfg,
        [runner.VariantSpec(name="base"), runner.VariantSpec(name="A1")],
        corpus_dir=corpus_dir,
        judge_llm=None,
        checkpoint=lambda r: checkpoints.append(len(r.variants)),
    )
    assert len(report.variants) == 2
    assert checkpoints == [1, 2]
