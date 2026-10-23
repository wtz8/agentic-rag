"""装配入口：把配置、索引、模型供给组装成一条可执行的在线链路。

server / eval / cli 共用它——评测跑的链路与 UI 跑的链路必须是同一条（04 §9.7）。
"""

from __future__ import annotations

from .agent.context import GraphDeps
from .config import Config
from .index.store import IndexStore
from .pipeline import Pipeline
from .providers.embedding import get_embedding
from .providers.llm import get_llm
from .providers.rerank import get_rerank
from .retrieval.retriever import HybridRetriever


def build_deps(cfg: Config, store: IndexStore) -> GraphDeps:
    store.load()
    embedder = get_embedding(cfg, workers=cfg.ingestion.workers)
    return GraphDeps(
        llm_light=get_llm(cfg, "light"),
        llm_main=get_llm(cfg, "main"),
        retriever=HybridRetriever(cfg, store, embedder),
        reranker=get_rerank(cfg),
        store=store,
        embedder=embedder,
    )


def build_pipeline(cfg: Config, store: IndexStore | None = None) -> Pipeline:
    store = store or IndexStore(cfg)
    deps = build_deps(cfg, store)
    return Pipeline(cfg, deps, store)


def build_store(cfg: Config) -> IndexStore:
    store = IndexStore(cfg)
    store.load()
    return store


__all__ = ["build_deps", "build_pipeline", "build_store"]
