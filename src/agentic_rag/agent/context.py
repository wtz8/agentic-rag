"""节点上下文（04 §7 的 `GraphDeps` / `NodeContext`）。

依赖走 LangGraph 官方的依赖注入通道 `config={"configurable": {"ctx": ctx}}`，
**不使用 `partial`**——`inspect.signature` 会把 partial 已绑定的参数从签名里移除，
LangGraph 因此认为节点不接受 config，trace 注入静默失效（H21）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..config import Config
from ..index.store import IndexStore
from ..observability.trace import Tracer
from ..providers.base import EmbeddingProtocol, LLMProtocol, RerankProtocol
from ..retrieval.retriever import HybridRetriever


@dataclass
class GraphDeps:
    llm_light: LLMProtocol  # 路由 / 改写 / 校验
    llm_main: LLMProtocol  # 终答生成
    retriever: HybridRetriever
    reranker: RerankProtocol
    store: IndexStore
    embedder: EmbeddingProtocol | None = None


@dataclass
class NodeContext:
    cfg: Config
    deps: GraphDeps
    tracer: Tracer
    query_id: str
    emit: Callable[[str, dict], None] | None = None
    stream: bool = False  # 是否边生成边推 token（SSE 场景）

    def push(self, event_type: str, data: dict) -> None:
        if self.emit is not None:
            self.emit(event_type, data)

    def bind(self) -> "NodeContext":
        """把 tracer 的事件回调接到 SSE 发射器上：一次采集、两处消费。"""
        if self.emit is not None:
            self.tracer.on_event = self.emit
        return self


def ctx_of(config) -> NodeContext:
    if config is None:
        raise RuntimeError("节点缺少 config：依赖走 configurable['ctx'] 注入")
    if isinstance(config, dict):
        ctx = config.get("configurable", {}).get("ctx")
    else:
        ctx = getattr(config, "ctx", None)
    if ctx is None:
        raise RuntimeError("configurable['ctx'] 缺失：节点依赖未注入（H21）")
    return ctx


__all__ = ["GraphDeps", "NodeContext", "ctx_of"]
