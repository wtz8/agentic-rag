"""Rerank 供给（03 §2.4）。

端点走**原生**端点（兼容端点不提供 rerank）：
`{native_base_url}/services/rerank/text-rerank/text-rerank`

返回结构（2026-10-21 实测）：`output.results[]`，不是 `results[]`——漏掉 `output`
这一层会在精排节点抛 KeyError，而精排是图的必经节点。
"""

from __future__ import annotations

import time

import httpx

from ..config import Config, DashScopeConfig
from ..utils import truncate_by_tokens
from .base import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    RerankProtocol,
    record_usage,
    server_error,
)


class DashScopeRerank:
    def __init__(self, cfg: DashScopeConfig, max_tokens: int = 4000):
        self.cfg = cfg
        self.max_tokens = max_tokens
        self.last_model = cfg.rerank_model
        self.last_usage: dict[str, int] = {}
        self._client = httpx.Client(timeout=cfg.timeout_s, trust_env=True)

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        docs = [truncate_by_tokens(d or "", self.max_tokens) for d in documents]
        top_n = max(1, min(top_n, len(docs)))
        url = f"{self.cfg.native_base_url.rstrip('/')}/services/rerank/text-rerank/text-rerank"
        payload = {
            "model": self.cfg.rerank_model,
            "input": {"query": query, "documents": docs},
            "parameters": {"top_n": top_n},
        }
        last: Exception | None = None
        data: dict = {}
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.cfg.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if resp.status_code in (401, 403):
                    raise ProviderAuthError(f"鉴权失败 {resp.status_code}")
                if resp.status_code == 429:
                    raise ProviderRateLimitError("限流 429")
                if resp.status_code >= 500:
                    raise server_error(f"服务端错误 {resp.status_code}")
                if resp.status_code >= 400:
                    raise ProviderResponseError(f"请求错误 {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                self.last_model = data.get("model", self.cfg.rerank_model)
                usage = data.get("usage") or {}
                record_usage(self, {"tokens_in": int(usage.get("total_tokens", 0) or 0)})
                results = data["output"]["results"]
                return [(int(r["index"]), float(r["relevance_score"])) for r in results]
            except (KeyError, TypeError) as e:
                raise ProviderResponseError(f"rerank 响应结构异常: {str(data)[:200]}") from e
            except httpx.TimeoutException as e:
                last = ProviderTimeoutError(str(e))
            except httpx.HTTPError as e:
                last = ProviderError(str(e))
            except ProviderError as e:
                if not e.retryable:
                    raise
                last = e
            except Exception as e:  # noqa: BLE001
                last = ProviderError(str(e))
            if attempt < self.cfg.max_retries:
                time.sleep(self.cfg.retry_backoff[min(attempt, len(self.cfg.retry_backoff) - 1)])
        raise last or ProviderError("unknown error")


class MockRerank:
    """按字面重合度打分，与 MockEmbedding 保持一致的伪语义（确定性，无随机）。"""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.last_model = "mock-rerank"
        self.last_usage: dict[str, int] = {}
        self.usage_total: dict[str, int] = {}

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        # M24：与 embed 同理，Mock 侧也要累计用量，见 embedding.py 的说明。
        record_usage(
            self, {"tokens_in": (len(query or "") + sum(len(d or "") for d in documents)) // 4}
        )
        scored = [(i, _overlap(query, d)) for i, d in enumerate(documents)]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[: max(1, min(top_n, len(scored)))]


def _overlap(query: str, doc: str) -> float:
    q = set(_grams(query or ""))
    d = set(_grams(doc or ""))
    if not q or not d:
        return 0.0
    return round(len(q & d) / len(q), 6)


def _grams(s: str, n: int = 2) -> list[str]:
    s = "".join(ch for ch in s if not ch.isspace())
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def get_rerank(cfg: Config) -> RerankProtocol:
    ds = cfg.dashscope
    max_tokens = cfg.retrieval.rerank_max_tokens
    if (ds.api_key or "").strip():
        return DashScopeRerank(ds, max_tokens=max_tokens)
    return MockRerank(max_tokens=max_tokens)


__all__ = ["DashScopeRerank", "MockRerank", "get_rerank", "RerankProtocol"]
