"""向量库（03 §7.1）：Protocol + Qdrant(local/server) + InMemory。

三条硬约束：
1. **local 与 server 不分叉**：`get_vector_store()` 只按 `backend` 决定连接方式，其余代码路径完全一致。
2. **距离度量写死 `Cosine`，不做成配置项**：K1 的充分性判据直接消费 `search()` 返回的分数，
   留成配置项等于埋一个"改了度量、阈值静默失效"的坑。`create()` 时断言已有集合的度量与预期一致。
3. **meta 不由向量库托管**：Qdrant 没有 collection 级 metadata API，`index_version` / `dim`
   照旧写 `data/index/meta.json`。**因此 Qdrant local 的数据必须放在子目录
   `data/index/qdrant/`**——Qdrant local 也会在它的 path 下写自己的 `meta.json`
   （内容是 `{"collections": ...}`），与我们的同名文件冲突：后写的一方会覆盖先写的，
   表现为 Qdrant 启动时 KeyError: 'collections' → 静默降级 InMemory，
   而这个故障从外面看只是"向量检索结果不对"。
"""

from __future__ import annotations

import json
import math
import pickle
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..config import Config, project_path
from ..schema import Chunk

COSINE = "Cosine"
# Qdrant 的 point id 只接受 UUID / 无符号整数，chunk_id 是字符串：
# 用 UUID5 从 chunk_id 稳定派生，chunk_id 本身存在 payload 里，检索后从 payload 取回。
_ID_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


def to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


class IndexVersionMismatch(Exception):
    """索引版本与配置不匹配：拒绝加载并要求重建。"""


class IndexLocked(Exception):
    """local 模式数据目录被其他进程占用。"""


@runtime_checkable
class VectorStore(Protocol):
    def create(self, dim: int, version: str) -> None: ...

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None: ...

    def search(self, vector: list[float], top_k: int, flt: dict | None = None) -> list[tuple[str, float]]: ...

    def delete_by_doc(self, doc_id: str) -> None: ...

    def count(self) -> int: ...

    def load_meta(self) -> dict: ...

    def save_meta(self, meta: dict) -> None: ...


def chunk_payload(c: Chunk) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "parent_id": c.parent_id,
        "block_type": c.block_type,
        "is_atomic": c.is_atomic,
        "source": c.source,
        "title_path": list(c.title_path),
        "page": c.page,
        "tokens": c.tokens,
        "text": c.text,
        "raw_text": c.raw_text,
    }


def payload_to_chunk(p: dict) -> Chunk:
    return Chunk(
        chunk_id=p["chunk_id"],
        doc_id=p["doc_id"],
        parent_id=p.get("parent_id"),
        is_parent=False,
        text=p.get("text", ""),
        raw_text=p.get("raw_text", ""),
        title_path=list(p.get("title_path", [])),
        block_type=p.get("block_type", "paragraph"),
        is_atomic=bool(p.get("is_atomic", False)),
        page=p.get("page"),
        source=p.get("source", ""),
        tokens=int(p.get("tokens", 0)),
    )


class QdrantStore:
    """local（`path`）与 server（`url`）共用一套实现，只有连接参数不同。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.collection = cfg.index.collection
        self.dim = cfg.dashscope.embedding_dim
        self._base = project_path(cfg.index.path)
        self._client = self._make_client()

    def _make_client(self):
        from qdrant_client import QdrantClient

        if self.cfg.index.backend == "qdrant_server":
            url = self.cfg.index.url
            if not url:
                raise ValueError("backend=qdrant_server 时 url 必填")
            return QdrantClient(url=url, timeout=self.cfg.dashscope.timeout_s)
        # local 模式独占目录。**只能靠 Qdrant 自己的报错判断占用，不能看 `.lock` 文件在不在**：
        # Qdrant 关闭客户端后并不删除 `.lock`，重开也完全正常——按"文件存在"判断，
        # 结果是第二次运行 CLI 就报"被其他进程打开"，而这个进程根本不存在。
        # 目录用子目录 `qdrant/`：Qdrant local 会往 path 下写它自己的 meta.json，
        # 与 IndexStore.save_meta 的 meta.json 是两个不同的文件（见模块 docstring 第 3 条）。
        qdir = self._base / "qdrant"
        qdir.mkdir(parents=True, exist_ok=True)
        try:
            return QdrantClient(path=str(qdir))
        except RuntimeError as e:
            if "already accessed" in str(e):
                raise IndexLocked(
                    f"索引目录已被其他进程打开: {self._base}。"
                    f"local 模式独占：请先停止服务再重建索引，或改用 qdrant_server 后端。原始错误：{e}"
                )
            raise

    # ── 生命周期 ──────────────────────────────────────────

    def create(self, dim: int, version: str) -> None:
        from qdrant_client import models

        self.dim = dim
        exists = self._client.collection_exists(self.collection)
        if exists:
            info = self._client.get_collection(self.collection)
            cfg_distance = getattr(getattr(info.config, "params", None), "vectors", None)
            distance = getattr(cfg_distance, "distance", None) if cfg_distance else None
            distance = str(distance) if distance is not None else COSINE
            if distance.upper() != COSINE.upper():
                raise ValueError(
                    f"已有集合的距离度量为 {distance}，预期 {COSINE}："
                    "改度量会让 sufficiency_vector_floor 静默失效，请重建索引"
                )
            return
        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        from qdrant_client import models

        if not ids:
            return
        self._client.upsert(
            collection_name=self.collection,
            points=models.Batch(ids=[to_point_id(i) for i in ids], vectors=vectors, payloads=payloads),
        )

    def search(self, vector: list[float], top_k: int, flt: dict | None = None) -> list[tuple[str, float]]:
        flt_model = self._to_filter(flt)
        resp = self._client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            query_filter=flt_model,
        )
        out = []
        for p in getattr(resp, "points", []):
            payload = p.payload or {}
            out.append((str(payload.get("chunk_id", p.id)), float(p.score)))
        return out

    def _to_filter(self, flt: dict | None):
        if not flt:
            return None
        from qdrant_client import models

        conds = []
        for k, v in flt.items():
            conds.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
        return models.Filter(must=conds)

    def get_payloads(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}
        points = self._client.retrieve(
            collection_name=self.collection, ids=[to_point_id(i) for i in ids], with_payload=True
        )
        out = {}
        for p in points:
            payload = p.payload or {}
            out[str(payload.get("chunk_id", p.id))] = payload
        return out

    def delete_by_doc(self, doc_id: str) -> None:
        from qdrant_client import models

        self._client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                )
            ),
        )

    def count(self) -> int:
        try:
            return int(self._client.count(collection_name=self.collection, exact=True).count)
        except Exception:  # noqa: BLE001
            return 0

    # ── meta ────────────────────────────────────────────

    @property
    def _meta_path(self) -> Path:
        return self._base / "meta.json"

    def load_meta(self) -> dict:
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text(encoding="utf-8"))
        return {}

    def save_meta(self, meta: dict) -> None:
        self._base.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


class InMemoryStore:
    """qdrant-client 不可用时的降级实现：brute-force 余弦，无持久化（除 meta 与 payload）。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.collection = cfg.index.collection
        self.dim = cfg.dashscope.embedding_dim
        self._base = project_path(cfg.index.path)
        self._vectors: dict[str, list[float]] = {}
        self._payloads: dict[str, dict] = {}
        self._path = self._base / "memory_store.pkl"

    def create(self, dim: int, version: str) -> None:
        self.dim = dim
        if self._path.exists():
            try:
                data = pickle.loads(self._path.read_bytes())
                self._vectors = data.get("vectors", {})
                self._payloads = data.get("payloads", {})
            except Exception:  # noqa: BLE001
                self._vectors, self._payloads = {}, {}

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        for i, v, p in zip(ids, vectors, payloads):
            self._vectors[i] = v
            self._payloads[i] = p

    def search(self, vector: list[float], top_k: int, flt: dict | None = None) -> list[tuple[str, float]]:
        scored = []
        for cid, vec in self._vectors.items():
            payload = self._payloads.get(cid, {})
            if flt and any(payload.get(k) != v for k, v in flt.items()):
                continue
            scored.append((cid, _cosine(vector, vec)))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]

    def get_payloads(self, ids: list[str]) -> dict[str, dict]:
        return {i: self._payloads[i] for i in ids if i in self._payloads}

    def delete_by_doc(self, doc_id: str) -> None:
        for cid, p in list(self._payloads.items()):
            if p.get("doc_id") == doc_id:
                self._payloads.pop(cid, None)
                self._vectors.pop(cid, None)

    def count(self) -> int:
        return len(self._vectors)

    def persist(self) -> None:
        self._base.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(pickle.dumps({"vectors": self._vectors, "payloads": self._payloads}))

    @property
    def _meta_path(self) -> Path:
        return self._base / "meta.json"

    def load_meta(self) -> dict:
        if self._meta_path.exists():
            return json.loads(self._meta_path.read_text(encoding="utf-8"))
        return {}

    def save_meta(self, meta: dict) -> None:
        self._base.mkdir(parents=True, exist_ok=True)
        self._meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return s / (na * nb)


def get_vector_store(cfg: Config) -> VectorStore:
    if cfg.index.backend == "memory":
        return InMemoryStore(cfg)
    try:
        return QdrantStore(cfg)
    except IndexLocked:
        # 目录被占用是**可操作的错误**，必须让用户看见，而不是悄悄换成内存库
        raise
    except Exception as e:  # noqa: BLE001 - qdrant 装不上时降级，保证任何环境可演示
        import warnings

        # 注意文案曾经是"不具备持久化"，但 InMemoryStore 其实会落 memory_store.pkl——
        # 文案错会让排查方向错。真实语义：单机可用，但无并发访问、无过滤下推等
        # 向量库能力，规模上去后必须换真后端。
        warnings.warn(f"Qdrant 不可用（{e}），降级为 InMemoryStore（单机 pickle 持久化，无并发访问）")
        return InMemoryStore(cfg)


__all__ = [
    "VectorStore",
    "QdrantStore",
    "InMemoryStore",
    "get_vector_store",
    "chunk_payload",
    "payload_to_chunk",
    "IndexVersionMismatch",
    "IndexLocked",
]
