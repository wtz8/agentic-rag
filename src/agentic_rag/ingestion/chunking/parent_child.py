"""父子切分（默认策略，03 §6.3）。

```
文档 → 按标题树切父块（≤1024 token），{doc_id}:p{seq:03d}，is_parent=True
     → 父块内按段落切子块（≤256 token，overlap 48），{doc_id}:c{seq:04d}
     → 子块进向量与 BM25；命中子块后回填父块给生成
     → chunk() 返回 [父块, *子块] 平铺列表，调用方按 is_parent 分流
```

Small-to-Big：子块向量语义集中召回准，父块保证生成时上下文完整。
子块 seq 是**文档级自增**而非父块内自增，这样 B2 消融切换策略时 ID 不撞车。
"""

from __future__ import annotations

from ...config import IngestionConfig
from ...schema import Chunk, ParsedDocument
from .base import doc_units, make_chunk, pack_pieces, section_key, split_text


class ParentChildChunker:
    name = "parent_child"

    def __init__(self, cfg: IngestionConfig):
        self.cfg = cfg

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        cfg = self.cfg
        units = doc_units(doc, cfg)
        out: list[Chunk] = []
        p_seq = 0
        c_seq = 0

        # 按标题路径分组，组内按父块上限滚动切分
        groups: list[list] = []
        current: list = []
        cur_key = None
        for u in units:
            key = section_key(u)
            if key != cur_key:
                if current:
                    groups.append(current)
                current, cur_key = [], key
            current.append(u)
        if current:
            groups.append(current)

        parents: list[tuple[str, list, list[str], int | None, str]] = []  # (pid, units, path, page, block_type)
        for group in groups:
            buf: list = []
            budget = 0
            path = group[0].title_path
            for u in group:
                # 原子块超限：单独成为一个父块（原子性优先于父块上限）
                if u.tokens > cfg.parent_chunk_tokens:
                    if buf:
                        parents.append((f"{doc.doc_id}:p{p_seq:03d}", buf, path, buf[0].page, "paragraph"))
                        p_seq += 1
                        buf, budget = [], 0
                    parents.append((f"{doc.doc_id}:p{p_seq:03d}", [u], u.title_path, u.page, u.block_type))
                    p_seq += 1
                    continue
                if budget + u.tokens > cfg.parent_chunk_tokens and buf:
                    parents.append((f"{doc.doc_id}:p{p_seq:03d}", buf, path, buf[0].page, "paragraph"))
                    p_seq += 1
                    buf, budget = [], 0
                buf.append(u)
                budget += u.tokens
            if buf:
                parents.append((f"{doc.doc_id}:p{p_seq:03d}", buf, path, buf[0].page, "paragraph"))
                p_seq += 1

        for pid, group, path, page, block_type in parents:
            raw = "\n\n".join(u.text for u in group)
            is_atomic = len(group) == 1 and group[0].is_atomic
            parent = make_chunk(
                doc.doc_id, pid, None, True, raw, path,
                group[0].block_type if is_atomic else "paragraph",
                is_atomic, page, doc.source,
            )
            out.append(parent)

            # 子块：原子单元整体成块，可切单元按 child 上限切并贪心合并
            child_pieces: list[tuple[str, bool, str, object]] = []
            pending: list[str] = []
            for u in group:
                if u.is_atomic:
                    if pending:
                        for p in pack_pieces(split_text("\n\n".join(pending), cfg.child_chunk_tokens, cfg.child_overlap_tokens), cfg.child_chunk_tokens):
                            child_pieces.append((p, False, u.block_type, u.page))
                        pending = []
                    child_pieces.append((u.text, True, u.block_type, u.page))
                else:
                    pending.append(u.text)
            if pending:
                for p in pack_pieces(split_text("\n\n".join(pending), cfg.child_chunk_tokens, cfg.child_overlap_tokens), cfg.child_chunk_tokens):
                    child_pieces.append((p, False, block_type, page))

            for text, atomic, btype, pg in child_pieces:
                out.append(
                    make_chunk(
                        doc.doc_id, f"{doc.doc_id}:c{c_seq:04d}", pid, False, text,
                        path, btype if atomic else "paragraph", atomic, pg, doc.source,
                    )
                )
                c_seq += 1
        return out


__all__ = ["ParentChildChunker"]
