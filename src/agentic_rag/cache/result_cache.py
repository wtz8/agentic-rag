"""查询结果缓存（02 §5.4 / L3 / J6）。

键 = hash(规范化 query + **history 指纹** + index_version + retrieval_signature + generation_signature)

三条缺一不可的因子：
- `history` 指纹：同一句"它的上限是多少"在不同会话里本就该有不同答案，
  漏了它会跨会话串味，而这类故障只在多轮里出现，单轮回归永远测不到（L3）
- `retrieval_signature`：切了消融开关后缓存会继续返回上一个配置的答案（UI 切换开关后会立刻看到自相矛盾的结果）
- `generation_signature`：换终答模型或改提示词后，缓存仍返回旧风格与旧引用格式的答案（J6）
"""

from __future__ import annotations

import json
import time

from ..config import Config, history_fingerprint, project_path
from ..utils import normalize_text, sha1_str


class ResultCache:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = project_path(cfg.cache.dir) / "result"

    @property
    def enabled(self) -> bool:
        return self.cfg.cache.enabled and "result" in self.cfg.cache.enabled_levels

    def make_key(self, query: str, history: list[dict] | None = None) -> str:
        parts = [normalize_text(query), history_fingerprint(history, self.cfg.agent.history_max_turns)]
        parts.append(self.cfg.index_version)
        # provider_mode 必须进键：index_version 只由"模型名 + 维度 + 切分参数"算出，
        # Mock 与真实 API 用的是同一个模型名，不带上它，配好 Key 后第一次查询
        # 会命中 Mock 时期留下的答案——表现为"换了 Key，答案还是假的"。
        parts.append(self.cfg.provider_mode)
        if self.cfg.cache.include_config_signature:
            parts.append(self.cfg.retrieval_signature)
            parts.append(self.cfg.generation_signature)
        return sha1_str("|".join(parts))[:20]

    def get(self, query: str, history: list[dict] | None = None) -> dict | None:
        if not self.enabled:
            return None
        p = self.dir / f"{self.make_key(query, history)}.json"
        if not p.exists():
            return None
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        if time.time() - float(row.get("ts", 0)) > self.cfg.cache.ttl_s:
            return None
        return row.get("value")

    def put(self, query: str, history: list[dict] | None, value: dict) -> None:
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            (self.dir / f"{self.make_key(query, history)}.json").write_text(
                json.dumps({"ts": time.time(), "value": value}, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001
            pass


__all__ = ["ResultCache"]
