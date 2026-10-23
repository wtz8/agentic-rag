"""探针校验（03 §10）：答案位置已知，只回答"链路通不通"。

与评测集的本质区别：探针判"期望来源是否进 Top-K"，是**链路连通性回归**，
每次重建索引后都要跑；评测集判检索与生成质量，改配置后跑。
前者保**没坏**，后者保**变好了**——两者并存，不可互相替代。

探针集必须包含表格数值类，因为"表格被切碎 → 检索到表头拿不到数值"是
本项目的头号失效模式，探针是唯一能在每次重建索引后自动发现它的手段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..index.store import IndexStore
from ..utils import read_jsonl

__all__ = ["ProbeCase", "ProbeRunResult", "run", "assert_healthy", "load_probe_set"]


@dataclass
class ProbeCase:
    question: str
    expect_source: str  # 期望命中的文件名
    expect_title_path: list[str] = field(default_factory=list)  # 可选：更严的校验
    expect_block_type: str | None = None  # paragraph | table，单独校验表格链路
    expect_answer_contains: str | None = None  # 期望答案中包含的字面片段（数值型尤其有效）
    # 同一事实在多格式语料中重复存在（同一开源文档的 md/docx/pdf/xlsx 版本），
    # 检索命中其中任何一个都算链路通。给出时取代 expect_source 的单一匹配。
    expect_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "question": self.question,
            "expect_source": self.expect_source,
            "expect_title_path": list(self.expect_title_path),
            "expect_block_type": self.expect_block_type,
            "expect_answer_contains": self.expect_answer_contains,
        }
        if self.expect_sources:
            d["expect_sources"] = list(self.expect_sources)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProbeCase":
        return cls(
            question=str(d.get("question", "")),
            expect_source=str(d.get("expect_source", "")),
            expect_title_path=list(d.get("expect_title_path") or []),
            expect_block_type=d.get("expect_block_type"),
            expect_answer_contains=d.get("expect_answer_contains"),
            expect_sources=list(d.get("expect_sources") or []),
        )


# P2 最小临时探针（06 §7.3）：正式探针集（eval/probe_set.jsonl）就位前的空档填补。
# 每条只标"来源 + 一句话事实"，只验证链路通不通。
FALLBACK_PROBES = (
    ProbeCase(
        question="华东区带宽采购单价",
        expect_source="a.md",
        expect_answer_contains="3.2",
    ),
)


@dataclass
class ProbeRunResult:
    """整批探针的结论。命名按 L12：与 quality.ProbeResult（单篇解析前探测）区分。

    两个命中数刻意分开（03 §10：判据是"期望来源是否进 Top-K"）：
    - `hit`：来源级命中——链路连通性的判据，`assert_healthy` 只看它；
    - `content_hit`：内容级命中（`expect_answer_contains` 的字面片段出现在来自期望来源
      的 Top-K 块里）——"表格没被切碎、数值没丢"的信号。来源对了但内容块没进
      Top-K 是**排名问题**（评测集的职责），不是**连通性问题**（探针的职责），
      两者混在一个数里会让探针报警报错对象。
    """

    total: int = 0
    hit: int = 0
    content_hit: int = 0
    failures: list[tuple[ProbeCase, list[str]]] = field(default_factory=list)  # (用例, 实际命中的 Top-K 来源)
    content_failures: list[tuple[ProbeCase, list[str]]] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.hit / self.total if self.total else 1.0

    def as_text(self) -> str:
        lines = [
            f"探针命中率: {self.hit}/{self.total}（{self.hit_rate:.0%}）",
            f"内容命中:   {self.content_hit}/{self.total}"
            "（期望片段出现在期望来源的 Top-K 块内；未命中=排名问题，属评测集职责）",
        ]
        for case, got in self.failures:
            # 用 GBK 能编码的 × 而不是 U+2717：中文 Windows 控制台下后者会让 print 直接抛异常
            lines.append(f"  × 来源未命中 {case.question}")
            lines.append(f"    期望 {case.expect_source}｜{' > '.join(case.expect_title_path) or '（任意节）'}")
            lines.append(f"    实际 Top-K 来源: {', '.join(got) or '（空）'}")
        for case, got in self.content_failures:
            frag = case.expect_answer_contains or ""
            lines.append(f"  △ 内容未命中 {case.question}（期望含「{frag}」）")
            lines.append(f"    Top-K 来源: {', '.join(got) or '（空）'}")
        return "\n".join(lines)


def load_probe_set(path: str | Path = "eval/probe_set.jsonl") -> list[ProbeCase]:
    """读正式探针集；不存在时返回空列表（调用方决定是否用临时探针补位）。

    相对路径按项目根解析（与 data/ 一致的口径），CLI 在任何工作目录下都指向同一份。
    """
    from ..config import project_path

    p = project_path(str(path))
    return [ProbeCase.from_dict(d) for d in read_jsonl(p)]


def filter_cases_for_sources(cases: list[ProbeCase], sources: list[str]) -> list[ProbeCase]:
    """只保留期望来源在本次构建的语料里的探针。

    探针集是语料绑定的：对语料子集重建索引（增量、测试语料）时，
    其他文档的探针条目不适用——跑了必然全错，把"探针报警"变成噪声，
    而探针的价值恰恰在于报警时必须当真。
    """
    have = {s.replace("\\", "/").split("/")[-1] for s in sources}
    out = []
    for c in cases:
        wanted = {w.replace("\\", "/").split("/")[-1] for w in (c.expect_sources or [c.expect_source]) if w}
        if wanted & have:
            out.append(c)
    return out


def _matches_source(case: ProbeCase, chunk) -> bool:
    """来源级判定：source（或 expect_sources 任一）+ 可选 title_path / block_type。"""
    wanted = case.expect_sources or [case.expect_source]
    names = {w.replace("\\", "/").split("/")[-1] for w in wanted if w}
    if chunk.source.replace("\\", "/").split("/")[-1] not in names:
        return False
    if case.expect_title_path:
        tp = list(case.expect_title_path)
        if list(chunk.title_path)[: len(tp)] != tp:
            return False
    if case.expect_block_type and chunk.block_type != case.expect_block_type:
        return False
    return True


def _content_ok(case: ProbeCase, chunks) -> bool:
    """内容级判定：期望片段出现在**来自期望来源**的 Top-K 块里。

    只在期望来源的块里找，不在全部 Top-K 里找——否则"别的文档恰好含相同数字"
    会把内容命中虚高（多格式语料里同一事实的重复副本由 expect_sources 表达）。
    """
    if not case.expect_answer_contains:
        return True
    frag = case.expect_answer_contains
    return any(frag in (c.raw_text or "") + (c.text or "") for c in chunks if _matches_source(case, c))


def run(
    cfg: Config,
    k: int = 5,
    *,
    store: IndexStore | None = None,
    cases: list[ProbeCase] | None = None,
) -> ProbeRunResult:
    """跑探针：经混合检索（不是只走 BM25），判期望来源是否进 Top-K。

    探针走的是与在线链路相同的 `HybridRetriever`——探针的意义是端到端连通性，
    单测某一路（比如只查 BM25）会把向量路的故障漏掉。
    """
    from ..bootstrap import build_store
    from ..providers.embedding import get_embedding
    from ..retrieval.retriever import HybridRetriever

    store = store or build_store(cfg)
    if cases is None:
        cases = load_probe_set() or list(FALLBACK_PROBES)
    if not cases:
        return ProbeRunResult()

    retriever = HybridRetriever(cfg, store, get_embedding(cfg, workers=cfg.ingestion.workers))
    result = ProbeRunResult(total=len(cases))
    for case in cases:
        diag: dict = {}
        candidates = retriever.retrieve([case.question], intent="simple", diag=diag)
        top = candidates[:k]
        chunks = store.get_chunks([c.chunk_id for c in top])
        sources_hit = any(_matches_source(case, c) for c in chunks)
        if sources_hit:
            result.hit += 1
        else:
            result.failures.append((case, [c.source for c in chunks]))
        if not _content_ok(case, chunks):
            result.content_failures.append((case, [c.source for c in chunks]))
        else:
            result.content_hit += 1
    return result


def assert_healthy(result: ProbeRunResult, min_hit_rate: float = 0.9) -> None:
    """命中率低于阈值时抛 AssertionError（03 §10：告警而非门禁，但必须看得见）。"""
    if result.total and result.hit_rate < min_hit_rate:
        raise AssertionError(
            f"探针命中率 {result.hit_rate:.0%} 低于阈值 {min_hit_rate:.0%}：\n{result.as_text()}"
        )
