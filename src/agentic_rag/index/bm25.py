"""自研 BM25（03 §7.2）。

为什么自研：中文分词与打分口径必须可控——领域词典注入、表格块加权、逐路消融，
三件事都要求拿到每一路的**原始分数**并控制过滤发生的层。
（边界见 01 §0"自研边界"：规模到百万级应换 Elasticsearch / OpenSearch，而不是优化自研实现。）

**BM25 不支持增量**：IDF 依赖全语料的 N 与 df，因此任何一次写入都会触发全量重建。
重建的数据源必须是 `data/index/chunks.jsonl`（全量物化视图），而不是本次增量的入参（H5）。
"""

from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Iterable

from ..config import project_path
from ..schema import Chunk
from ..utils import sha256_file

_PUNCT = re.compile(r"^[\s\W_]+$", re.UNICODE)

_DICT_LOADED = False


def load_user_dict(path: str) -> list[str] | None:
    """加载 jieba 用户词典。**文件缺失不抛错**，返回 None 由调用方记 warning（H12）。

    路径一律走 `project_path` 相对**项目根**解析：早先这里用 `Path(path).exists()`，
    是相对**当前工作目录**的——换个目录跑 CLI，词典就静默不加载，而
    `retrieval_signature` 里那份 `user_dict_sha256` 仍按"有词典"算，
    于是缓存键与真实分词口径脱钩（同一语料两种结果，且缓存不会失效）。
    """
    global _DICT_LOADED
    if not path:
        return None
    p = project_path(path)
    if not p.exists():
        return None
    try:
        import jieba

        jieba.load_userdict(str(p))
        _DICT_LOADED = True
        words = []
        for line in p.read_text(encoding="utf-8").splitlines():
            w = line.strip()
            if w and not w.startswith("#"):
                words.append(w.split(" ")[0])
        return words
    except Exception:  # noqa: BLE001
        return None


def tokenize(text: str) -> list[str]:
    """jieba 精确模式；词典由 `load_user_dict` 预加载。"""
    try:
        import jieba
    except ImportError:  # pragma: no cover - jieba 是硬依赖
        return [t for t in re.findall(r"\w+", text or "") if t]
    tokens = jieba.lcut(text or "")
    out = []
    for t in tokens:
        t = t.strip().lower()
        if not t or _PUNCT.match(t):
            continue
        out.append(t)
    return out


class BM25Index:
    """倒排表 + IDF + 长度归一化。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunk_ids: list[str] = []
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.df: dict[str, int] = {}
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}
        self.user_dict_sha: str = "none"

    # ── 构建 ──────────────────────────────────────────────

    def build(
        self,
        chunks: list[Chunk],
        user_dict: list[str] | None = None,
        user_dict_path: str | None = None,
    ) -> None:
        """构建倒排表，并落定词典指纹。

        `user_dict` 是"这次构建实际加载到 jieba 里的词"，同时兼作适用性判据——
        没加载成功（`None`）时指纹必须是 `"none"`，与每一次实际的打分口径对齐。

        `user_dict_path` 是词典文件（相对项目根解析）。指纹必须由**文件路径**算出，
        不能由 `user_dict` 这个 `list[str]` 算出：早先这里判的是
        `isinstance(user_dict, str)`，而调用方传的是列表 → 分支恒为 False，
        这行成了死代码，真正的值反而是 `store.py` 在 `build()` 之后另行补写的
        （`build_from_chunks()` 这条路径没人补，`user_dict_sha` 永远是 `"none"`）。
        指纹的写入方必须只有一个，否则"哪一处漏了"是看不见的。
        """
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.doc_lens = []
        self.df = {}
        self.postings = {}
        for idx, c in enumerate(chunks):
            tokens = tokenize(c.text)
            self.doc_lens.append(len(tokens))
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for t, n in tf.items():
                self.postings.setdefault(t, []).append((idx, n))
                self.df[t] = self.df.get(t, 0) + 1
        n = len(self.chunk_ids)
        self.avgdl = (sum(self.doc_lens) / n) if n else 0.0
        # IDF: log(1 + (N - df + 0.5) / (df + 0.5))
        self.idf = {
            t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in self.df.items()
        }
        if user_dict and user_dict_path:
            p = project_path(user_dict_path)
            if p.exists():
                self.user_dict_sha = sha256_file(p)[:16]

    # ── 检索 ──────────────────────────────────────────────

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self.chunk_ids:
            return []
        tokens = tokenize(query)
        scores: dict[int, float] = {}
        for t in tokens:
            postings = self.postings.get(t)
            if not postings:
                continue
            idf = self.idf.get(t, 0.0)
            for idx, tf in postings:
                dl = self.doc_lens[idx] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[idx] = scores.get(idx, 0.0) + idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [(self.chunk_ids[i], round(s, 6)) for i, s in ranked[:top_k]]

    # ── 持久化 ────────────────────────────────────────────

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(
            pickle.dumps(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "chunk_ids": self.chunk_ids,
                    "doc_lens": self.doc_lens,
                    "avgdl": self.avgdl,
                    "df": self.df,
                    "postings": self.postings,
                    "idf": self.idf,
                    "user_dict_sha": self.user_dict_sha,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        data = pickle.loads(Path(path).read_bytes())
        idx = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        idx.chunk_ids = data.get("chunk_ids", [])
        idx.doc_lens = data.get("doc_lens", [])
        idx.avgdl = data.get("avgdl", 0.0)
        idx.df = data.get("df", {})
        idx.postings = data.get("postings", {})
        idx.idf = data.get("idf", {})
        idx.user_dict_sha = data.get("user_dict_sha", "none")
        return idx

    def stats(self) -> dict:
        return {
            "chunks": len(self.chunk_ids),
            "terms": len(self.df),
            "avgdl": round(self.avgdl, 3),
            "k1": self.k1,
            "b": self.b,
            "user_dict_sha": self.user_dict_sha,
        }


def build_from_chunks(chunks: Iterable[Chunk], cfg_index) -> BM25Index:
    """便捷构造：词典从 `IndexConfig.user_dict_path` 加载（缺失时降级，不抛错）。"""
    idx = BM25Index(k1=cfg_index.bm25_k1, b=cfg_index.bm25_b)
    words = load_user_dict(cfg_index.user_dict_path)
    idx.build(list(chunks), user_dict=words, user_dict_path=cfg_index.user_dict_path)
    return idx


__all__ = ["BM25Index", "tokenize", "load_user_dict", "build_from_chunks"]
