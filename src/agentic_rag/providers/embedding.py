"""Embedding 供给（03 §2.3）。

DashScope 走 OpenAI 兼容端点 `/embeddings`，**必须显式传 `dimensions`**——
不传则取服务端默认值，一旦与向量库 collection 定义不一致，报错会发生在写入阶段而非启动阶段。
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

from ..config import Config, DashScopeConfig
from .base import (
    EmbeddingProtocol,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    record_usage,
    server_error,
)

_BATCH = 10  # 实测：10 条批量摊薄到 54ms/条，单条 261ms（08 §3.2）
# 单次请求的**输入总长度**上限（text-embedding-v4 实测报
# "Range of input length should be [1, 33000]"）。只按条数分批是不够的：
# 块长差异极大（中位 316 字符、最长 5 万+），10 条长块凑一起照样超限，
# 于是整个 batch 400、对应块全部没有向量 → 索引静默降级为半残。
# 留 10% 余量：长度是按字符估的，而服务端按 token 计。
_MAX_INPUT_CHARS = 30000


class DashScopeEmbedding:
    def __init__(self, cfg: DashScopeConfig, dim: int, workers: int = 4):
        self.cfg = cfg
        self._dim = dim
        self.workers = max(1, workers)
        self.last_model = cfg.embedding_model
        self.last_usage: dict[str, int] = {}
        self._client = httpx.Client(timeout=cfg.timeout_s, trust_env=True)

    @property
    def dim(self) -> int:
        return self._dim

    def _post(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.cfg.base_url.rstrip('/')}/embeddings"
        payload = {"model": self.cfg.embedding_model, "input": texts, "dimensions": self._dim}
        last: Exception | None = None
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
                    raise ProviderRateLimitError(f"限流 429")
                if resp.status_code >= 500:
                    raise server_error(f"服务端错误 {resp.status_code}")
                if resp.status_code >= 400:
                    raise ProviderResponseError(f"请求错误 {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                self.last_model = data.get("model", self.cfg.embedding_model)
                usage = data.get("usage") or {}
                record_usage(self, {"tokens_in": int(usage.get("total_tokens", 0) or 0)})
                rows = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
                return [r["embedding"] for r in rows]
            except httpx.TimeoutException as e:
                last = ProviderTimeoutError(str(e))
            except httpx.HTTPError as e:
                last = ProviderError(str(e))
            except ProviderError as e:
                # 已分好类的异常不再重新包裹：否则 429/超时被降级成基类
                # ProviderError（retryable=False），调用方按"永久失败"处理，
                # 而它明明只是被限流了。不可重试的（鉴权/响应解析）直接抛出。
                if not e.retryable:
                    raise
                last = e
            except Exception as e:  # noqa: BLE001
                last = ProviderError(str(e))
            if attempt < self.cfg.max_retries:
                time.sleep(self.cfg.retry_backoff[min(attempt, len(self.cfg.retry_backoff) - 1)])
        raise last or ProviderError("unknown error")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        batches = _plan_batches(texts)
        if len(batches) == 1:
            return self._post(batches[0])
        with ThreadPoolExecutor(max_workers=min(self.workers, len(batches))) as ex:
            out: list[list[float]] = []
            for part in ex.map(self._post, batches):
                out.extend(part)
            return out


def _plan_batches(texts: list[str]) -> list[list[str]]:
    """同时受两个上限约束：条数 `_BATCH` 与单请求输入总长度 `_MAX_INPUT_CHARS`。

    单条本身就超限的，宁可让它单独失败并**报出 chunk 位置**，也不静默截断——
    截断后的向量不代表那段文本，等于给检索埋了一颗不知道在哪的雷。
    """
    batches: list[list[str]] = []
    cur: list[str] = []
    cur_chars = 0
    for t in texts:
        n = len(t or "")
        if n > _MAX_INPUT_CHARS:
            if cur:
                batches.append(cur)
                cur, cur_chars = [], 0
            batches.append([t])  # 单独发：失败时能精确定位到这一条
            continue
        if cur and (len(cur) >= _BATCH or cur_chars + n > _MAX_INPUT_CHARS):
            batches.append(cur)
            cur, cur_chars = [], 0
        cur.append(t)
        cur_chars += n
    if cur:
        batches.append(cur)
    return batches


class MockEmbedding:
    """字符 3-gram 哈希投影（03 §2.3）。

    不用随机向量：随机向量会让检索结果每次不同，评测不可复现。
    字面重叠多的文本相似度更高——这条性质是它被选中的理由，由测试锁住。
    """

    def __init__(self, dim: int = 1024):
        self._dim = dim
        self.last_model = "mock-embedding"
        self.last_usage: dict[str, int] = {}
        self.usage_total: dict[str, int] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        # M24：Mock 也必须走 record_usage。否则 trace 的 token 归因在没有
        # Key 的环境下恒为 0，CI 永远验证不了这条链路——而"某字段在什么
        # 条件下是常量"正是本项目要抓的失效模式。用字符数/4 估算 token，
        # 保持确定性，不引入随机。
        record_usage(self, {"tokens_in": sum(len(t or "") for t in texts) // 4})
        return [_pseudo_embed(t, self._dim) for t in texts]


def _pseudo_embed(text: str, dim: int) -> list[float]:
    vec = np.zeros(dim, dtype=np.float64)
    s = text or ""
    if len(s) < 3:
        s = s + "\x00" * (3 - len(s))
    for i in range(len(s) - 2):
        h = int(hashlib.md5(s[i : i + 3].encode("utf-8")).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def get_embedding(cfg: Config, workers: int = 4) -> EmbeddingProtocol:
    ds = cfg.dashscope
    if (ds.api_key or "").strip():
        return DashScopeEmbedding(ds, ds.embedding_dim, workers=workers)
    return MockEmbedding(dim=ds.embedding_dim)


__all__ = ["DashScopeEmbedding", "MockEmbedding", "get_embedding", "EmbeddingProtocol"]
