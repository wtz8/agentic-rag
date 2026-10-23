"""结构感知切分（单层基线，用于 B2 消融对照，03 §6.2）。

- 按标题树的**最深层标题**切分子树
- 子树内若超 token 上限，按段落边界再切，切出的每段都保留完整标题路径
- 表格块整体不切（四档规则见 `base.split_table_text`）

产出的块 `is_parent=False` / `parent_id=None`：没有父块可回填，
rerank 回填时用它自身（03 §0.1），这让 B2 的两组走同一段回填代码。
"""

from __future__ import annotations

from ...config import IngestionConfig
from ...schema import Chunk, ParsedDocument
from .base import doc_units, make_chunk, pack_pieces, section_key, split_text


class StructuralChunker:
    name = "structural"

    def __init__(self, cfg: IngestionConfig):
        self.cfg = cfg

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        cfg = self.cfg
        units = doc_units(doc, cfg)
        out: list[Chunk] = []
        c_seq = 0

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

        for group in groups:
            path = group[0].title_path
            pieces: list[tuple[str, bool, str, object]] = []
            pending: list[str] = []
            for u in group:
                if u.is_atomic:
                    if pending:
                        for p in pack_pieces(
                            split_text("\n\n".join(pending), cfg.child_chunk_tokens),
                            cfg.child_chunk_tokens,
                        ):
                            pieces.append((p, False, "paragraph", u.page))
                        pending = []
                    pieces.append((u.text, True, u.block_type, u.page))
                else:
                    pending.append(u.text)
            if pending:
                for p in pack_pieces(
                    split_text("\n\n".join(pending), cfg.child_chunk_tokens), cfg.child_chunk_tokens
                ):
                    pieces.append((p, False, "paragraph", group[0].page))

            for text, atomic, btype, page in pieces:
                out.append(
                    make_chunk(
                        doc.doc_id, f"{doc.doc_id}:c{c_seq:04d}", None, False, text,
                        path, btype, atomic, page, doc.source,
                    )
                )
                c_seq += 1
        return out


__all__ = ["StructuralChunker"]
