"""确定性函数缓存（03 §1 / 02 §5.4）：embedding 与 rerank 共用。

两者同构——都是"确定性函数的入参哈希 → 结果"，只有键的构造不同。
合成一个模块比开两个文件更不容易漏掉某一级的失效逻辑。

键必须包含**模型名**（换模型而缓存未失效会返回上一种模型的向量/排序）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import Config, project_path
from ..utils import sha1_str


class ComputeCache:
    def __init__(self, cfg: Config, level: str):
        self.cfg = cfg
        self.level = level
        self.dir = project_path(cfg.cache.dir) / "compute" / level

    @property
    def enabled(self) -> bool:
        return self.cfg.cache.enabled and self.level in self.cfg.cache.enabled_levels

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str):
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        if time.time() - float(row.get("ts", 0)) > self.cfg.cache.ttl_s:
            try:
                p.unlink()
            except Exception:  # noqa: BLE001
                pass
            return None
        return row.get("value")

    def set(self, key: str, value) -> None:
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            self._path(key).write_text(
                json.dumps({"ts": time.time(), "value": value}, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001 - 缓存写不进去不该让主链路失败
            pass
        self._enforce_limit()

    def _enforce_limit(self) -> None:
        try:
            files = list(self.dir.glob("*.json"))
            if len(files) <= self.cfg.cache.max_entries:
                return
            files.sort(key=lambda p: p.stat().st_mtime)
            for p in files[: len(files) - self.cfg.cache.max_entries]:
                p.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    # ── 键构造 ────────────────────────────────────────────

    @staticmethod
    def embedding_key(text: str, model: str, dim: int) -> str:
        return sha1_str(f"emb|{model}|{dim}|{text}")[:20]

    @staticmethod
    def rerank_key(query: str, doc_ids: list[str], model: str, top_n: int) -> str:
        return sha1_str(f"rr|{model}|{top_n}|{query}|{'|'.join(doc_ids)}")[:20]


__all__ = ["ComputeCache"]
