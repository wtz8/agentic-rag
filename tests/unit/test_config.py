"""配置树：版本哈希敏感性、指纹敏感性、启动校验（07 §2）。"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agentic_rag.config import (
    Config,
    ConfigError,
    compute_index_version,
    generation_signature,
    get_config,
    history_fingerprint,
    retrieval_signature,
    validate_config,
)
from agentic_rag.utils import count_tokens

DOCS = Path(__file__).resolve().parents[2] / "docs"


def _mutate(base: Config, **path_value) -> Config:
    """返回一个改过某一项的配置副本。path_value 形如 retrieval__use_rerank=False。"""
    import copy

    cfg = copy.deepcopy(base)
    for key, value in path_value.items():
        section, field = key.split("__", 1)
        setattr(getattr(cfg, section), field, value)
    return cfg


# ── index_version 对每一个哈希因子的敏感性（H19）──────────────────────


@pytest.mark.parametrize(
    "change",
    [
        {"dashscope__embedding_model": "text-embedding-v3"},
        {"dashscope__embedding_dim": 768},
        {"ingestion__chunk_strategy": "structural"},
        {"ingestion__parent_chunk_tokens": 2048},
        {"ingestion__child_chunk_tokens": 512},
        {"ingestion__child_overlap_tokens": 64},
        {"ingestion__atomic_max_tokens": 4096},
        {"ingestion__table_split_threshold_tokens": 512},
    ],
)
def test_index_version_sensitive_to_every_factor(cfg, change):
    base = compute_index_version(cfg)
    changed = compute_index_version(_mutate(cfg, **change))
    assert base != changed, f"改了 {change} 但版本号没变，会导致换配置不重建索引"


def test_index_version_stable(cfg):
    assert compute_index_version(cfg) == cfg.index_version == compute_index_version(cfg)


# ── retrieval_signature 对每个消融开关的敏感性 ─────────────────────────


@pytest.mark.parametrize(
    "change",
    [
        {"retrieval__use_vector": False},
        {"retrieval__use_bm25": False},
        {"retrieval__use_rerank": False},
        {"retrieval__fusion_strategy": "weighted"},
        {"retrieval__rrf_k": 10},
        {"retrieval__vector_top_k": 20},
        {"retrieval__bm25_top_k": 20},
        {"retrieval__fusion_top_k": 40},
        {"retrieval__min_candidates": 5},
        {"retrieval__sufficiency_vector_floor": 0.5},
        {"retrieval__table_boost": 1.5},
        {"retrieval__filter_tables_only": True},
        {"index__bm25_k1": 1.2},
        {"index__bm25_b": 0.3},
        # M15：精排池深与编排旋钮同样改变最终答案，漏了它们 UI 现场切开关会命中旧答案
        {"retrieval__rerank_top_k": 8},
        {"agent__allow_reflect_for_simple": True},
        {"agent__skip_rewrite_for_simple": False},
        {"agent__max_retrieve_iterations": 2},
        {"agent__max_sub_queries": 6},
    ],
)
def test_retrieval_signature_sensitive_to_every_switch(cfg, change):
    base = retrieval_signature(cfg)
    changed = retrieval_signature(_mutate(cfg, **change))
    assert base != changed, f"改了 {change} 但指纹没变，缓存会继续返回旧配置的答案"


def test_ablation_variants_are_distinguishable(cfg):
    """每个变体都得有自己能区别于基线的指纹——否则它和基线跑的是同一份缓存。"""
    from agentic_rag.eval.ablation import VARIANTS
    from agentic_rag.eval.runner import patch_config

    for name, spec in VARIANTS.items():
        if name == "base":
            continue
        vc = patch_config(cfg, spec.patch)
        same = (
            vc.retrieval_signature == cfg.retrieval_signature
            and vc.generation_signature == cfg.generation_signature
        )
        # A2 检索配置 = 基线（累加链终点），同指纹是设计使然；
        # A1（关 rerank）必须与基线可区分（M25 修复后）。
        if name == "A2":
            assert same, f"{name} 检索配置应等于基线，指纹应当相同"
        else:
            assert not same, f"{name} 改了配置却与基线同指纹，切面演示会返回旧答案"


def test_retrieval_signature_includes_user_dict(cfg, tmp_path):
    """词典是文件不是配置字段：改一个字节指纹必须变（L6）。"""
    d = tmp_path / "user_dict.txt"
    d.write_text("单卡上限\n", encoding="utf-8")
    cfg.index.user_dict_path = str(d)
    s1 = retrieval_signature(cfg)
    d.write_text("单卡上限\n年费\n", encoding="utf-8")
    s2 = retrieval_signature(cfg)
    assert s1 != s2
    cfg.index.user_dict_path = str(tmp_path / "missing.txt")
    assert retrieval_signature(cfg) != s1


@pytest.mark.parametrize(
    "change",
    [
        {"dashscope__llm_main": "qwen-max"},
        {"dashscope__llm_light": "qwen-turbo"},
        {"generation__temperature": 0.9},
        {"generation__max_tokens": 1024},
        {"generation__lang": "en"},
        {"generation__max_citations": 3},
        {"generation__max_context_chunks": 3},
        {"generation__faithfulness_threshold": 0.5},
        {"generation__regenerate_on_unsupported": True},
    ],
)
def test_generation_signature_sensitive(cfg, change):
    base = generation_signature(cfg)
    assert generation_signature(_mutate(cfg, **change)) != base


def test_generation_signature_changes_with_prompt_version(cfg, monkeypatch):
    """prompt 用 schema_version 而非正文哈希：结构变了必须失效缓存。"""
    import agentic_rag.config as cfg_mod
    from agentic_rag.prompts.loader import Prompt

    base = generation_signature(cfg)
    orig = cfg_mod.all_prompts()

    def patched():
        out = []
        for p in orig:
            out.append(
                p if p.name != "route"
                else Prompt(
                    name=p.name,
                    schema_version=p.schema_version + 1,
                    model=p.model,
                    temperature=p.temperature,
                    json_mode=p.json_mode,
                    system=p.system,
                    user_template=p.user_template,
                )
            )
        return out

    monkeypatch.setattr(cfg_mod, "all_prompts", patched)
    assert generation_signature(cfg) != base


# ── history 指纹（L3）───────────────────────────────────────────────


def test_history_fingerprint_truncate_and_stability():
    h = [{"role": "user", "content": f"m{i}"} for i in range(10)]
    fp_full = history_fingerprint(h, max_turns=3)
    assert fp_full == history_fingerprint(h, max_turns=3)
    assert fp_full != history_fingerprint(h[:6], max_turns=3)
    assert history_fingerprint(None, 3) == "-"
    # 时间戳之类的噪声不进键
    noisy = [dict(m, ts=i) for i, m in enumerate(h)]
    assert history_fingerprint(noisy, 3) == fp_full


# ── 启动校验逐条有对应的"应当抛错"用例 ───────────────────────────────


@pytest.mark.parametrize(
    "change",
    [
        {"ingestion__child_chunk_tokens": 2048},  # child >= parent
        {"ingestion__parent_chunk_tokens": 4096},  # parent > atomic_max
        {"ingestion__table_split_threshold_tokens": 4096},
        {"ingestion__child_overlap_tokens": 0},
        {"ingestion__child_overlap_tokens": 512},
        {"retrieval__rerank_top_k": 100},  # > fusion_top_k
        {"retrieval__fusion_top_k": 200},  # > 两路之和
        {"retrieval__fusion_top_k": 10},  # < max(单路)
        {"observability__trace_keep_top_k": 10},  # < fusion_top_k
        {"dashscope__embedding_dim": 32},
        {"quality__digit_loss_warn": 0.9},  # warn > reject
        {"quality__digit_loss_reject": 1.5},  # 越界
        {"quality__text_coverage_reject": 0.9},  # reject > warn
        {"quality__raw_char_min_for_check": 0},
        {"quality__ocr_min_chars": 0},
        {"quality__on_reject": "delete"},
        {"retrieval__unanswerable_rerank_floor": 0.0},
        {"retrieval__sufficiency_vector_floor": -1.0},
        {"retrieval__min_candidates": 0},
        {"generation__max_context_chunks": 8},  # > rerank_top_k
        {"agent__max_retrieve_iterations": 0},
        {"agent__max_retrieve_iterations": 4},
        {"agent__max_generate_iterations": 0},
        {"agent__max_sub_queries": 0},
        {"agent__max_sub_queries": 9},
        {"agent__history_max_turns": -1},
        {"generation__lang": "jp"},
        {"generation__max_citations": 0},
        {"generation__max_citations": 9},  # > max_context_chunks
        {"generation__faithfulness_threshold": 1.5},
        {"ingestion__chunk_strategy": "late_chunking"},
        {"retrieval__fusion_strategy": "sum"},
        {"index__backend": "lancedb"},
        {"ingestion__pdf_backend": "pdfminer"},
        {"server__max_concurrent_queries": 0},
        {"profile": "prod"},
    ],
)
def test_validation_rejects(cfg, change):
    if "profile" in change:
        bad = _mutate(cfg)
        bad.profile = change["profile"]
    else:
        bad = _mutate(cfg, **change)
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_validation_rejects_both_routes_off(cfg):
    bad = _mutate(cfg, retrieval__use_vector=False)
    bad.retrieval.use_bm25 = False
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_validation_requires_url_for_qdrant_server(cfg):
    bad = _mutate(cfg, index__backend="qdrant_server")
    bad.index.url = None
    with pytest.raises(ConfigError):
        validate_config(bad)
    bad.index.url = "http://127.0.0.1:6333"
    validate_config(bad)


def test_eval_profile_requires_pinned_models_and_no_cache():
    c = get_config(env={"DASHSCOPE_API_KEY": "", "AR_PROFILE": "eval"}, validate=False)
    c.cache.enabled = False
    c.dashscope.llm_main = "qwen3.7-plus"
    c.dashscope.llm_light = "qwen3.8-flash"
    validate_config(c)  # 通过

    c2 = get_config(env={"DASHSCOPE_API_KEY": "", "AR_PROFILE": "eval"}, validate=False)
    c2.cache.enabled = False
    with pytest.raises(ConfigError):  # 别名未锁版本
        validate_config(c2)

    c3 = get_config(env={"DASHSCOPE_API_KEY": "", "AR_PROFILE": "eval"}, validate=False)
    c3.dashscope.llm_main = "qwen3.7-plus"
    c3.dashscope.llm_light = "qwen3.8-flash"
    with pytest.raises(ConfigError):  # 缓存未关
        validate_config(c3)


def test_default_config_valid(cfg):
    validate_config(cfg)
    assert cfg.provider_mode == "mock"


def test_get_config_reads_api_key(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    c = get_config()
    assert c.dashscope.api_key == "sk-test"
    assert c.provider_mode == "dashscope"


# ── 配置树字段完整性对账（H1）────────────────────────────────────────


_PLACEHOLDER = re.compile(r"^(?:[XxYy]+|xxx|<.*>)$")


def _cfg_refs_in_docs() -> set[tuple[str, ...]]:
    refs: set[tuple[str, ...]] = set()
    pattern = re.compile(r"\bcfg\.([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)")
    for name in ("03-模块设计-离线链路.md", "04-模块设计-在线链路.md"):
        text = (DOCS / name).read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            parts = tuple(m.group(1).split("."))
            if any(_PLACEHOLDER.match(p) for p in parts):
                continue  # cfg.<X> 这类示意写法
            refs.add(parts)
    return refs


def _resolve(root: Config, path: tuple[str, ...]) -> bool:
    """文档里的 `cfg` 随模块签名不同而指向根配置或某个子配置，两种都要认。"""
    roots: list[object] = [root]
    for name in (
        "dashscope",
        "quality",
        "ingestion",
        "index",
        "retrieval",
        "agent",
        "generation",
        "cache",
        "server",
        "observability",
    ):
        roots.append(getattr(root, name))
    for obj in roots:
        cur: object = obj
        ok = True
        for part in path:
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok:
            return True
    return False


def test_config_tree_covers_every_doc_reference():
    """扫描 04/05 正文里所有 `cfg.<X>` 引用，断言每个都能在配置树上取到。

    这条测试存在的唯一理由：它抓到过 `ocr_min_chars` 与 `raw_char_min_for_check`
    两个凭空引用的字段。
    """
    cfg = Config()
    missing = [".".join(p) for p in sorted(_cfg_refs_in_docs()) if not _resolve(cfg, p)]
    assert not missing, f"文档引用了配置树上不存在的字段: {sorted(set(missing))}"


def test_no_magic_numbers_in_quality_module():
    """质量模块不得硬编码阈值（03 §4）：只断言"文件里没有浮点阈值字面量"这一条。"""
    src = Path(__file__).resolve().parents[2] / "src" / "agentic_rag" / "ingestion" / "quality.py"
    if not src.exists():
        pytest.skip("quality.py 尚未实现")
    text = src.read_text(encoding="utf-8")
    for bad in ("0.10", "0.30", "0.60", "0.40", "0.80"):
        assert bad not in text, f"quality.py 里出现了硬编码阈值 {bad}"


def test_count_tokens_deterministic():
    assert count_tokens("") == 0
    assert count_tokens("华东区带宽采购单价") == count_tokens("华东区带宽采购单价")
    assert count_tokens("a" * 100) > count_tokens("a" * 10)
