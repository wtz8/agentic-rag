"""跨模块的小工具：token 计数、文件与哈希辅助。

token 计数必须有确定性回退：tiktoken 首次使用要联网拉取 BPE 词表，
离线环境下不能让摄取链路因此崩掉，所以回退到固定的启发式估算。
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

_CJK = re.compile(
    r"[　-〿぀-ヿ㐀-䶿一-鿿豈-﫿＀-￯]"
)


def _is_cjk(ch: str) -> bool:
    return bool(_CJK.match(ch))


def _heuristic_tokens(text: str) -> int:
    """回退估算：CJK 每字约 1 token，拉丁词约 1.3 token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    latin = len(re.findall(r"[A-Za-z0-9]+", text))
    other = len(re.findall(r"[^\sA-Za-z0-9]", text)) - cjk
    return max(1, int(round(cjk + latin * 1.3 + max(0, other) * 0.5)))


try:  # pragma: no cover - 依赖网络环境
    import tiktoken as _tiktoken

    _ENC = _tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - 离线环境下的既定回退
    _ENC = None


def count_tokens(text: str) -> int:
    """统计文本的 token 数（有 tiktoken 用 tiktoken，否则启发式）。"""
    if not text:
        return 0
    if _ENC is not None:
        try:
            return len(_ENC.encode(text))
        except Exception:  # pragma: no cover
            pass
    return _heuristic_tokens(text)


def truncate_by_tokens(text: str, max_tokens: int) -> str:
    """按 token 截断（rerank 单条上限 4000 时用）。"""
    if max_tokens <= 0 or not text:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    # 二分找最长前缀，避免逐 token 编码的开销
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return "none"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, rows: Iterable[dict], mode: str = "w") -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(s: str) -> str:
    """NFKC 归一化 + 去零宽字符，用于缓存键与字面匹配的稳定性。"""
    s = unicodedata.normalize("NFKC", s or "")
    return s.replace("​", "").replace("﻿", "")


def digit_count(s: str) -> int:
    """数字字符数（守恒检查用的是"数字字符"，不是数值个数）。"""
    return sum(1 for ch in s or "" if ch.isdigit())


_SENT_SPLIT = re.compile(r"(?<=[。！？；\n])")


def split_sentences(text: str, limit: int = 0) -> list[str]:
    """按句末标点拆分，用于命题级判定。

    放在 utils 而不是某个模块内部，是因为**评测裁判与在线校验必须拆出同一份下标**：
    judge 模板输出的是命题下标（05 §2），若两处拆分规则不同，下标就对不上，
    表现为"忠实度里的无支撑命题张冠李戴"——不报错，只是结论错位。
    """
    parts = [p.strip() for p in _SENT_SPLIT.split(text or "")]
    out = [p for p in parts if p]
    return out[:limit] if limit > 0 else out


def strip_citation_marks(text: str) -> str:
    """去掉正文里的 `[n]` 角标——命题拆分不应把角标算进句子内容。"""
    return re.sub(r"\[\d+\]", "", text or "").strip()
