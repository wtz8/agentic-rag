"""查询变换（04 §2）：改写 / 分解 / HyDE。

**升级规则只看 `iteration`，不看上一轮用了什么策略**（L2）。
原规则写成"≥1 且上一轮已是 rewrite"，走一遍就发现 hyde 永远不可达：
multi_hop 首轮命中 decompose，iteration=1 时该条件为假 → 落到 rewrite → 预算耗尽 → hyde 一次都轮不到。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config, get_config
from ..prompts.loader import get_prompt
from ..providers.base import LLMProtocol, ProviderError, first_json_object


@dataclass
class TransformResult:
    queries: list[str]
    strategy: str  # none | rewrite | decompose | hyde
    # BM25 路专用 query（J7）：hyde 时 = 原 query（假设文档不该做字面匹配）；
    # decompose 时为 None —— 让每条子查询各自走 BM25（04 §4）。
    bm25_query: str | None = None


def _pick_strategy(intent: str, iteration: int, cfg: Config) -> str:
    if iteration == 0:
        if intent == "simple" and cfg.agent.skip_rewrite_for_simple:
            return "none"
        if intent == "multi_hop":
            return "decompose"
        if intent == "table_numeric":
            return "rewrite"
        return "rewrite"
    if iteration == 1:
        return "rewrite"  # 第一次重试：换措辞（成本最低的另一种假设）
    return "hyde"  # 第二次重试：换范式，而不是再改一遍措辞


def _history_text(history: list[dict] | None, max_turns: int) -> str:
    if not history:
        return "（无）"
    turns = history[-max_turns * 2 :]
    return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in turns)


def _llm_json(llm: LLMProtocol, name: str, **kwargs) -> dict | None:
    p = get_prompt(name)
    try:
        raw = llm.chat(p.messages(**kwargs), temperature=p.temperature, json_mode=True)
        return first_json_object(raw)
    except ProviderError:
        return None
    except Exception:  # noqa: BLE001 - 宽容解析失败同样走回落
        return None


def transform_query(
    query: str,
    intent: str,
    llm: LLMProtocol,
    history: list[dict] | None = None,
    iteration: int = 0,
    cfg: Config | None = None,
) -> TransformResult:
    """把用户原始问法转化为更适合检索的查询集合。

    **失败一律回落原 query，绝不让改写环节成为链路的单点故障。**
    """
    cfg = cfg or get_config()
    strategy = _pick_strategy(intent, iteration, cfg)
    hist = _history_text(history, cfg.agent.history_max_turns)

    if strategy == "none":
        return TransformResult([query], "none", query)

    if strategy == "decompose":
        data = _llm_json(llm, "decompose", query=query, history=hist, max_sub_queries=cfg.agent.max_sub_queries)
        queries = _clean_queries(data, query, max_n=cfg.agent.max_sub_queries)
        if not (2 <= len(queries) <= cfg.agent.max_sub_queries):
            # 退回单跳，但保留模型给出的那一条查询——它仍是改写后的表达，
            # 丢掉它会让"改写失败"和"判定为单跳"两种结果混为一谈
            return TransformResult(queries, "rewrite", None)
        return TransformResult(queries, "decompose", None)

    if strategy == "hyde":
        data = _llm_json(llm, "hyde", query=query, history=hist)
        queries = _clean_queries(data, query, max_n=1)
        para = queries[0]
        if para and 100 <= len(para) <= 800 and para != query:
            return TransformResult([para], "hyde", query)
        # HyDE 失败 → 退回 rewrite 的结果；rewrite 也失败 → 原 query
        data = _llm_json(llm, "rewrite", query=query, history=hist, intent=intent, iteration=iteration)
        queries = _clean_queries(data, query, max_n=1)
        return TransformResult(queries, "rewrite", None)

    # rewrite
    data = _llm_json(llm, "rewrite", query=query, history=hist, intent=intent, iteration=iteration)
    queries = _clean_queries(data, query, max_n=1)
    # `bm25_query` 只在 hyde 下有意义（假设文档只喂向量路，BM25 仍用原 query）。
    # rewrite 的改写结果要同时喂两路，所以这里是 None 而不是原 query——
    # 早先填了原 query，而消费方只认 strategy=="hyde"，字段成了死字段，
    # 两处口径已经分叉：改消费方就会突然变成"改写只走向量路"。
    return TransformResult(queries, "rewrite", None)


def _clean_queries(data: dict | None, fallback: str, max_n: int) -> list[str]:
    if not data:
        return [fallback]
    raw = data.get("queries")
    if not isinstance(raw, list) or not raw:
        return [fallback]
    out = [str(q).strip() for q in raw if str(q).strip()]
    return (out or [fallback])[:max_n]


__all__ = ["TransformResult", "transform_query", "_pick_strategy"]
