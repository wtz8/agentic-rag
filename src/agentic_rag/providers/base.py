"""模型供给抽象（03 §2.1）。

业务代码只依赖这里的 Protocol，不依赖任何厂商 SDK 与具体实现。
异常细分的目的是让调用方能区分"可重试的临时故障"与"必须降级的永久故障"。
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator, Protocol, runtime_checkable


class ProviderError(Exception):
    """模型供给异常基类。默认不可重试。"""

    retryable = False


class ProviderAuthError(ProviderError):
    """鉴权失败，不可重试。"""


class ProviderRateLimitError(ProviderError):
    """限流，可重试。"""

    retryable = True


class ProviderTimeoutError(ProviderError):
    """超时，可重试。"""

    retryable = True


class ProviderResponseError(ProviderError):
    """响应解析失败（含模型输出非法 JSON），不可重试。"""


def server_error(msg: str) -> ProviderError:
    """5xx：服务端临时故障，**可重试**。

    不能直接 `raise ProviderError(...)` —— 基类的 `retryable` 是 False，
    重试循环里若按这个标记决定要不要继续，服务端抖一下就会被当成永久失败。
    """
    e = ProviderError(msg)
    e.retryable = True
    return e


@runtime_checkable
class LLMProtocol(Protocol):
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str: ...

    def stream(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Iterator[str]: ...

    @property
    def model_name(self) -> str: ...


@runtime_checkable
class EmbeddingProtocol(Protocol):
    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class RerankProtocol(Protocol):
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]: ...


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(raw: str) -> dict | list:
    """宽容解析模型输出的 JSON。

    支持：裸 JSON / ```json 代码块包裹 / 前后有解释文字。
    解析失败抛 `ProviderResponseError`，由调用方按该节点的失败回落处理——
    宽容解析不是为了吞掉错误，而是把"模型多说了两句话"和"模型真的没给 JSON"区分开。
    """
    if raw is None:
        raise ProviderResponseError("empty response")
    text = raw.strip()

    candidates: list[str] = [text]
    m = _FENCE.search(text)
    if m:
        candidates.insert(0, m.group(1).strip())
    # 前后有解释文字：截取第一个 { 或 [ 到最后一个配对符号
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i >= 0 and j > i:
            candidates.append(text[i : j + 1])

    last_err: Exception | None = None
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception as e:  # noqa: BLE001 - 逐个尝试，最后统一抛
            last_err = e
    raise ProviderResponseError(f"无法解析为 JSON: {text[:200]}") from last_err


def first_json_object(raw: str) -> dict:
    d = extract_json(raw)
    if not isinstance(d, dict):
        raise ProviderResponseError(f"期望 JSON 对象，实际为 {type(d).__name__}")
    return d


def as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def record_usage(obj: Any, usage: dict[str, int]) -> None:
    """记录一次调用的用量：更新 `last_usage` 并累加到 `usage_total`。

    `last_usage` 只反映最后一次调用（排查单次异常用）；
    `usage_total` 是进程内累计（评测报告的 token 成本列用）。
    此前只有 last_usage，评测层要算"这轮消融花了多少 token"时拿不到数据。
    """
    obj.last_usage = {k: int(v) for k, v in (usage or {}).items()}
    if not hasattr(obj, "usage_total") or obj.usage_total is None:
        obj.usage_total = {}
    for k, v in obj.last_usage.items():
        obj.usage_total[k] = obj.usage_total.get(k, 0) + v


# ── 账号欠费（Arrearage）的显式建模 ─────────────────────────
#
# 欠费与普通 provider 错误有本质区别：它是**终态**——重试不会好、继续跑
# 只会烧时间并产出降级垃圾指标。2026-10-22 全量消融（390 条、约 ¥8）在
# 余额耗尽后仍继续跑完 100+ 条错误查询，最后报告因标定崩溃整轮丢失。
# 上层（eval runner）依赖这两个符号做 fail-fast 中止并保住已完成的变体。

_ARREARAGE_MARKERS = ("Arrearage", "good standing", "overdue-payment")


class AccountArrearage(RuntimeError):
    """账号欠费：应立即中止付费调用并落盘已有结果，而不是继续或重试。"""


def is_arrearage_error(err: Any) -> bool:
    """判断异常对象或错误串是否为账号欠费（按 DashScope 错误体特征匹配）。"""
    s = f"{type(err).__name__}: {err}" if isinstance(err, BaseException) else str(err or "")
    return any(m in s for m in _ARREARAGE_MARKERS)
