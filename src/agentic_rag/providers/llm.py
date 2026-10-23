"""LLM 供给：DashScope（OpenAI 兼容端点）与 Mock 双实现（03 §2.2）。

不用任何厂商 SDK，只用 httpx 直连 REST。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Iterator

import httpx

from ..config import Config, DashScopeConfig
from ..utils import split_sentences as _split_sentences
from .base import (
    LLMProtocol,
    record_usage,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    server_error,
)

# 命题拆分时，把 prompt 里的指令段与字段标签剥掉，剩下的当作正文
_LABEL = re.compile(r"^(?:答案正文（已去掉角标）|答案正文|上下文|命题列表|引用列表|待分类问题|原始问题|问题)\s*[:：]?\s*")
_INSTRUCTION = re.compile(r"只输出 JSON[\s\S]*$")


class DashScopeLLM:
    """OpenAI 兼容端点 `/chat/completions`。"""

    def __init__(self, cfg: DashScopeConfig, model: str):
        self.cfg = cfg
        self.model = model
        self.timeout = cfg.timeout_s
        # 实际生效的模型名（响应体返回，可能与请求值不同——别名会静默漂移）
        self.last_model: str = model
        self.last_usage: dict[str, int] = {}
        self._client = httpx.Client(timeout=self.timeout, trust_env=True)

    @property
    def model_name(self) -> str:
        return self.last_model or self.model

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, payload: dict) -> dict:
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._client.post(url, headers=self._headers(), json=payload)
                if resp.status_code == 401 or resp.status_code == 403:
                    raise ProviderAuthError(f"鉴权失败 {resp.status_code}: {resp.text[:200]}")
                if resp.status_code == 429:
                    raise ProviderRateLimitError(f"限流 429: {resp.text[:200]}")
                if resp.status_code >= 500:
                    raise server_error(f"服务端错误 {resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 400:
                    raise ProviderResponseError(f"请求错误 {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                self.last_model = data.get("model", self.model)
                usage = data.get("usage") or {}
                record_usage(self, {
                    "tokens_in": int(usage.get("prompt_tokens", 0) or 0),
                    "tokens_out": int(usage.get("completion_tokens", 0) or 0),
                })
                return data
            except httpx.TimeoutException as e:
                last = ProviderTimeoutError(str(e))
            except httpx.HTTPError as e:
                last = ProviderError(str(e))
            except (ProviderAuthError, ProviderResponseError) as e:
                raise
            except ProviderRateLimitError as e:
                last = e
            except Exception as e:  # noqa: BLE001
                last = ProviderError(str(e))
            if attempt < self.cfg.max_retries:
                time.sleep(self.cfg.retry_backoff[min(attempt, len(self.cfg.retry_backoff) - 1)])
        raise last or ProviderError("unknown error")

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post(payload)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise ProviderResponseError(f"响应结构异常: {str(data)[:200]}") from e

    def stream(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        with self._client.stream("POST", url, headers=self._headers(), json=payload) as resp:
            if resp.status_code == 401 or resp.status_code == 403:
                raise ProviderAuthError(f"鉴权失败 {resp.status_code}")
            if resp.status_code == 429:
                raise ProviderRateLimitError(f"限流 429: {resp.text[:200]}")
            if resp.status_code >= 400:
                raise ProviderResponseError(f"请求错误 {resp.status_code}: {resp.text[:200]}")
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if chunk.get("model"):
                    self.last_model = chunk["model"]
                if chunk.get("usage"):
                    u = chunk["usage"]
                    record_usage(self, {
                        "tokens_in": int(u.get("prompt_tokens", 0) or 0),
                        "tokens_out": int(u.get("completion_tokens", 0) or 0),
                    })
                try:
                    delta = chunk["choices"][0]["delta"].get("content")
                except (KeyError, IndexError):
                    delta = None
                if delta:
                    yield delta


def _body_of(text: str) -> str:
    """剥掉 prompt 里的指令段与字段标签，剩下正文（用于按句号拆命题）。"""
    s = _INSTRUCTION.sub("", text or "")
    out = []
    for line in s.splitlines():
        line = _LABEL.sub("", line.strip())
        if line:
            out.append(line)
    return "\n".join(out).strip()


class MockLLM:
    """无 API Key 时的确定性替身（03 §2.2 / 05 §3）。

    只认**英文 schema 关键词**，不认中文措辞——中文提示词正文会频繁微调，
    schema 字段名稳定。返回必须是合法 schema，且不得含随机内容。
    """

    def __init__(self, model: str = "mock"):
        self.model = model
        self.last_usage: dict[str, int] = {"tokens_in": 0, "tokens_out": 0}
        self.usage_total: dict[str, int] = {}

    @property
    def model_name(self) -> str:
        return self.model

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        user = _last_user(messages)
        payload = self._mock_payload(user)
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        # M24：Mock 侧累计用量（字符数/4 估算，确定性），理由见 embedding.py。
        record_usage(
            self,
            {
                "tokens_in": sum(len(str(m.get("content", ""))) for m in (messages or [])) // 4,
                "tokens_out": len(text) // 4,
            },
        )
        return text

    def stream(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        user = _last_user(messages)
        text = (
            "（Mock 模式）这是确定性占位回答，用于验证链路连通性，不代表真实检索效果。"
            "真实效果需要配置 DASHSCOPE_API_KEY 后重跑。[1]"
        )
        if user:
            text = f"（Mock 模式）已收到问题：{user[:40]}。这是确定性占位回答，用于验证链路连通性。[1]"
        # M24：流式同样要累计用量，否则 SSE 场景（UI 主路径）的 trace 全是 0。
        record_usage(
            self,
            {
                "tokens_in": sum(len(str(m.get("content", ""))) for m in (messages or [])) // 4,
                "tokens_out": len(text) // 4,
            },
        )
        for ch in text:
            yield ch

    def _mock_payload(self, user: str) -> dict | str:
        low = (user or "").lower()
        # 顺序即消歧：judge_faithfulness 同时含 supported / unsupported，
        # claim_verify 只含 unsupported，claim_extract 只含 claims
        if "supported" in low:
            return {"supported": [], "unsupported": []}
        if "unsupported" in low:
            return {"unsupported": [], "faithfulness": 1.0}
        if "claims" in low:
            body = _body_of(user)
            claims = _split_sentences(body)[:8]
            return {"claims": claims or ["（空）"]}
        if "intent" in low:
            return {"intent": "simple", "confidence": 0.9, "reason": "mock"}
        if "queries" in low:
            q = _original_query(user)
            return {"queries": [q], "strategy": "rewrite"}
        if "correct" in low:
            return {"correct": [], "wrong": []}
        if "score" in low:
            return {"score": 5}
        return "（Mock 模式）确定性兜底回答。"


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def _original_query(user: str) -> str:
    """从 prompt 里取出原始问题那一行（用于 Mock 的 queries 返回）。"""
    for line in (user or "").splitlines():
        s = line.strip()
        if s.startswith("原始问题：") or s.startswith("待分类问题："):
            return s.split("：", 1)[1].strip()
    return (user or "").strip()[:50]


def get_llm(cfg: Config, kind: str = "main") -> LLMProtocol:
    kind = kind if kind in ("main", "light") else "main"
    ds = cfg.dashscope
    model = ds.llm_main if kind == "main" else ds.llm_light
    if (ds.api_key or "").strip():
        return DashScopeLLM(ds, model)
    return MockLLM(model=f"mock-{model}")


__all__ = ["DashScopeLLM", "MockLLM", "get_llm", "LLMProtocol"]
