"""FastAPI 服务（04 §10）。

三条不能妥协的约束：
1. **SSE endpoint 必须 `async def` + 线程池跑同步图**：图内部是 httpx 同步调用 + jieba + 线程池，
   直接在 `async def` 里调会阻塞 event loop，01 §9 宣称的并发能力就成了空话。
2. **信号量限流**：线程池大小有限，超出不排队会在池里堆积，
   表现为"第 5 个请求卡死"而不是"排队等待"。宁可让前端转圈，也不要静默挂起。
3. **跨线程投递只能用 `call_soon_threadsafe`**：在子线程里 `await q.put()` 会抛错。
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..bootstrap import build_pipeline, build_store
from ..config import Config, get_config
from ..observability.trace import load_trace
from ..pipeline import Pipeline

STATIC_DIR = Path(__file__).parent / "static"
_SENTINEL = object()


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list)


class ConfigPatch(BaseModel):
    """现场演示消融用的开关白名单（04 §10.3 左栏）。

    只放开**检索侧**的只读于查询时的开关：它们不需要重建索引，
    且变化会自动进 `retrieval_signature` → 结果缓存键随之失效，不会出现"改了开关还是旧答案"。
    切分与 embedding 相关的参数不放开——那需要重建索引，动态改只会制造不一致的假象。
    """

    use_vector: bool | None = None
    use_bm25: bool | None = None
    use_rerank: bool | None = None
    filter_tables_only: bool | None = None
    min_candidates: int | None = Field(default=None, ge=1, le=50)
    sufficiency_vector_floor: float | None = Field(default=None, ge=0.0, le=1.0)


def create_app(cfg: Config | None = None, pipeline: Pipeline | None = None) -> FastAPI:
    cfg = cfg or get_config()
    app = FastAPI(title="Agentic RAG", version="0.1.0")
    store = build_store(cfg)
    pipe = pipeline or build_pipeline(cfg, store)
    pool = ThreadPoolExecutor(max_workers=cfg.server.max_concurrent_queries)
    sem = asyncio.Semaphore(cfg.server.max_concurrent_queries)

    app.state.cfg = cfg
    app.state.store = store
    app.state.pipeline = pipe

    # ── 问答（SSE）───────────────────────────────────────

    @app.post("/api/ask")
    async def ask(req: AskRequest) -> StreamingResponse:
        return StreamingResponse(
            _stream(pipe, req, pool, sem),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲，否则流式变成一次性返回
            },
        )

    # ── 轻量接口：不占线程池 ─────────────────────────────

    @app.get("/api/stats")
    async def stats() -> dict:
        s = store.stats()
        s["provider_mode"] = cfg.provider_mode
        s["index_version_matches"] = s["index_version"] == s["config_version"]
        return s

    @app.get("/api/config")
    async def get_config_api() -> dict:
        r = cfg.retrieval
        return {
            "use_vector": r.use_vector,
            "use_bm25": r.use_bm25,
            "use_rerank": r.use_rerank,
            "filter_tables_only": r.filter_tables_only,
            "min_candidates": r.min_candidates,
            "sufficiency_vector_floor": r.sufficiency_vector_floor,
            "retrieval_signature": cfg.retrieval_signature,
            "provider_mode": cfg.provider_mode,
        }

    @app.post("/api/config")
    async def patch_config(patch: ConfigPatch) -> dict:
        from ..config import validate_config

        r = cfg.retrieval
        before = (
            r.use_vector,
            r.use_bm25,
            r.use_rerank,
            r.filter_tables_only,
            r.min_candidates,
            r.sufficiency_vector_floor,
        )
        for field_name, value in patch.model_dump(exclude_none=True).items():
            setattr(r, field_name, value)
        try:
            validate_config(cfg)
        except Exception as e:  # noqa: BLE001 - 演示面板的非法值要回退，不能留下半个改过的配置
            (
                r.use_vector,
                r.use_bm25,
                r.use_rerank,
                r.filter_tables_only,
                r.min_candidates,
                r.sufficiency_vector_floor,
            ) = before
            raise HTTPException(status_code=400, detail=str(e))
        return await get_config_api()

    @app.get("/api/trace/{query_id}")
    async def trace(query_id: str) -> dict:
        row = load_trace(query_id, cfg)
        if row is None:
            raise HTTPException(status_code=404, detail=f"trace 不存在：{query_id}")
        return row

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "provider_mode": cfg.provider_mode}

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(str(STATIC_DIR / "index.html"))

    return app


async def _stream(
    pipe: Pipeline, req: AskRequest, pool: ThreadPoolExecutor, sem: asyncio.Semaphore
) -> AsyncIterator[str]:
    """线程池跑同步图 → 队列桥接 → 逐条 SSE。"""
    async with sem:
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run() -> None:
            try:
                for ev in pipe.run_stream(req.query, req.history):
                    loop.call_soon_threadsafe(q.put_nowait, ev)
            except Exception as e:  # noqa: BLE001 - 未捕获异常也要给 UI 一个出口
                from ..pipeline import Event

                loop.call_soon_threadsafe(
                    q.put_nowait, Event("error", {"message": f"{type(e).__name__}: {e}"})
                )
            finally:
                loop.call_soon_threadsafe(q.put_nowait, _SENTINEL)

        task = loop.run_in_executor(pool, _run)
        while True:
            ev = await q.get()
            if ev is _SENTINEL:
                break
            yield ev.to_sse()
        await task


def serve(cfg: Config | None = None, host: str | None = None, port: int | None = None) -> None:
    """`cli serve` 的入口（uvicorn 在运行时才导入，避免 CLI 冷启动变慢）。"""
    import uvicorn

    cfg = cfg or get_config()
    uvicorn.run(
        create_app(cfg),
        host=host or cfg.server.host,
        port=int(port or cfg.server.port),
        log_level="info",
    )


__all__ = ["create_app", "serve", "AskRequest", "STATIC_DIR"]
