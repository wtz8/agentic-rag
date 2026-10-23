"""答案生成（04 §8.2）：分级供给（qwen-plus 写终答）+ 引用标注。"""

from __future__ import annotations

from typing import Iterator

from ..config import Config
from ..prompts.loader import get_prompt
from ..providers.base import LLMProtocol
from ..schema import Citation, ScoredChunk
from .citation import build_context_block, parse_citations


class Generator:
    def __init__(self, cfg: Config, llm: LLMProtocol):
        self.cfg = cfg
        self.llm = llm

    def _messages(self, query: str, context: list[ScoredChunk]):
        p = get_prompt("answer")
        lang = "中文" if self.cfg.generation.lang == "zh" else "English"
        return p.messages(
            query=query,
            context=build_context_block(context),
            max_citations=self.cfg.generation.max_citations,
            language=lang,
        )

    def generate(self, query: str, context: list[ScoredChunk]) -> tuple[str, list[Citation]]:
        p = get_prompt("answer")
        raw = self.llm.chat(
            self._messages(query, context),
            temperature=self.cfg.generation.temperature,
            max_tokens=self.cfg.generation.max_tokens,
        )
        text, citations = parse_citations(raw, context)
        return text, self._limit(citations)

    def stream(self, query: str, context: list[ScoredChunk]) -> Iterator[str]:
        """流式生成。**中断时返回已生成部分**，不抛异常。"""
        p = get_prompt("answer")
        try:
            for delta in self.llm.stream(
                self._messages(query, context),
                temperature=self.cfg.generation.temperature,
                max_tokens=self.cfg.generation.max_tokens,
            ):
                yield delta
        except Exception:  # noqa: BLE001 - 流式中断不是致命错误
            return

    def _limit(self, citations: list[Citation]) -> list[Citation]:
        n = self.cfg.generation.max_citations
        return citations[:n]


__all__ = ["Generator"]
