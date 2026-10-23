"""引用解析 + 生成（02 §5.3 / 04 §8）。

编号协议的关键风险：模型输出越界编号（幻觉）时，必须**丢弃**而不是把裸 `[9]` 留给用户。
"""

from __future__ import annotations

from agentic_rag.generation.citation import (
    build_context_block,
    extract_claims,
    parse_citations,
    strip_marks,
)
from agentic_rag.generation.generator import Generator
from agentic_rag.schema import Chunk, ScoredChunk


def _sc(i: int, text: str, source: str | None = None) -> ScoredChunk:
    ch = Chunk(
        chunk_id=f"doc{i}:c{i:04d}",
        doc_id=f"doc{i}",
        parent_id=None,
        text=text,
        raw_text=text,
        title_path=[f"标题{i}"],
        source=source or f"t{i}.md",
        page=i,
    )
    return ScoredChunk(chunk=ch, score=0.9 - 0.1 * i, rank=i + 1)


CTX = [_sc(0, "华东区单价 3.2 万元。"), _sc(1, "华北区单价 2.8 万元。"), _sc(2, "研发投入 1.2 亿元。")]


# ── 上下文组装 ────────────────────────────────────────────────────────


def test_build_context_block_numbers_from_one():
    block = build_context_block(CTX)
    assert block.startswith("[1] 华东区单价 3.2 万元。")
    assert "[2] 华北区单价 2.8 万元。" in block
    assert "[3] 研发投入 1.2 亿元。" in block


def test_build_context_block_empty():
    assert build_context_block([]) == ""


# ── 编号归一化 ────────────────────────────────────────────────────────


def test_parse_citations_renumbers_by_first_appearance():
    clean, cites = parse_citations("A[2] B[1] C[2]", CTX)
    assert clean == "A[1] B[2] C[1]"
    assert [c.index for c in cites] == [1, 2]
    assert cites[0].chunk_id == CTX[1].chunk.chunk_id  # 旧 [2] → 新 [1]
    assert cites[1].chunk_id == CTX[0].chunk.chunk_id


def test_parse_citations_dedups_repeated_marks():
    clean, cites = parse_citations("X[1] Y[1] Z[1]", CTX)
    assert clean == "X[1] Y[1] Z[1]"
    assert len(cites) == 1


def test_parse_citations_drops_out_of_range():
    """越界编号直接丢弃，绝不把裸 [9] 暴露给用户（H20）。"""
    clean, cites = parse_citations("A[9] B[1]", CTX)
    assert "9" not in clean
    assert clean == "A B[1]"
    assert [c.index for c in cites] == [1]


def test_parse_citations_keeps_unknown_score_as_none():
    """拿不到分数时必须留 None，不能兜成 0.0。

    `ScoredChunk.score` 是 `float | None`：None 表示"这一路没产生产分数"
    （缓存回放的旧格式没存），0.0 表示"检索到了但分是 0"。
    早先这里写 `float(sc.score)`——None 会直接抛 TypeError，
    而按另一种修法兜成 0.0，则会把手 silently 变成一条"相似度为零"的假证据。
    """
    ctx = [ScoredChunk(chunk=CTX[0].chunk, score=None, rank=1)]
    _, cites = parse_citations("华东区单价 3.2 万元[1]。", ctx)
    assert len(cites) == 1
    assert cites[0].score is None


def test_parse_citations_passes_through_real_score():
    """有分数时必须原样带上（正常路径不能被上面的修复带偏）。"""
    ctx = [ScoredChunk(chunk=CTX[0].chunk, score=0.42, rank=1)]
    _, cites = parse_citations("华东区单价 3.2 万元[1]。", ctx)
    assert cites[0].score == 0.42


def test_parse_citations_drops_zero_and_negative():
    clean, cites = parse_citations("A[0] B[1]", CTX)
    assert clean == "A B[1]"
    assert len(cites) == 1


def test_parse_citations_without_marks_returns_empty():
    clean, cites = parse_citations("没有引用标记的答案。", CTX)
    assert clean == "没有引用标记的答案。"
    assert cites == []


def test_parse_citations_empty_context_is_passthrough():
    clean, cites = parse_citations("A[1]", [])
    assert clean == "A[1]"
    assert cites == []


def test_parse_citations_carries_provenance():
    _, cites = parse_citations("A[3]", CTX)
    c = cites[0]
    assert c.index == 1
    assert c.source == "t2.md"
    assert c.title_path == ["标题2"]
    assert c.page == 2
    assert c.quote.startswith("研发投入")


def test_parse_citations_collapses_whitespace_left_by_dropped_marks():
    clean, _ = parse_citations("A [9] B", CTX)
    assert clean == "A B"


# ── 正文处理 ──────────────────────────────────────────────────────────


def test_strip_marks_removes_all_brackets():
    assert strip_marks("A[1] B[2]。") == "A B。"


def test_extract_claims_splits_by_sentence():
    claims = extract_claims("第一句。[1]第二句！第三句；")
    assert claims == ["第一句。", "第二句！", "第三句；"]


def test_extract_claims_empty():
    assert extract_claims("") == []
    assert extract_claims(None) == []


# ── Generator ─────────────────────────────────────────────────────────


class _LLM:
    def __init__(self, text: str = "答案正文[1]补充[2]"):
        self.text = text
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake"

    def chat(self, messages, temperature=0.0, max_tokens=2048, json_mode=False):
        self.calls += 1
        return self.text

    def stream(self, messages, temperature=0.0, max_tokens=2048):
        for ch in self.text:
            yield ch


def test_generator_parses_and_limits_citations(cfg):
    cfg.generation.max_citations = 1
    llm = _LLM("A[1] B[2]")
    gen = Generator(cfg, llm)
    text, cites = gen.generate("问题", CTX)
    assert text == "A[1] B[2]"
    assert len(cites) == 1  # max_citations 截断
    assert llm.calls == 1


def test_generator_stream_yields_deltas(cfg):
    gen = Generator(cfg, _LLM("流式答案"))
    assert "".join(gen.stream("问题", CTX)) == "流式答案"


def test_generator_stream_interruption_returns_partial(cfg, monkeypatch):
    """流式中断返回已生成部分，不抛异常（04 §6.6）。"""

    class BoomLLM:
        model_name = "boom"

        def chat(self, *a, **k):
            raise RuntimeError("x")

        def stream(self, *a, **k):
            yield "已生成"
            raise RuntimeError("连接中断")

    gen = Generator(cfg, BoomLLM())
    assert "".join(gen.stream("问题", CTX)) == "已生成"


def test_generator_injects_citation_protocol_into_prompt(cfg):
    """提示词必须告诉模型用编号而不是 chunk_id（H20）。"""
    seen = {}

    class SpyLLM:
        model_name = "spy"

        def chat(self, messages, temperature=0.0, max_tokens=2048, json_mode=False):
            seen["user"] = messages[-1]["content"]
            return "A[1]"

        def stream(self, *a, **k):
            yield ""

    Generator(cfg, SpyLLM()).generate("问题", CTX)
    assert "[1]" in seen["user"]
    assert "chunk_id" not in seen["user"].lower()
