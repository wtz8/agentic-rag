#!/usr/bin/env python3
"""语料登记：写 MANIFEST.md 表格 + manifest_detail.json 明细。

被 `fetch_corpus.py`（下载）、`make_scanned.py`（自制扫描件）、
`make_office.py`（生成 DOCX/PPTX）共用，保证各渠道写出的表格格式一致。

为什么分成两个文件
------------------
MANIFEST.md 是**给人看的**：七列，一眼能看出这批语料的构成与预期难点。
manifest_detail.json 是**给机器用的**：sha256、完整 URL、获取日期。
把 sha256 塞进表格会让列宽失控到没法读，但不记又无法校验"拿到的是不是同一批文件"。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "MANIFEST.md"
DETAIL = RAW / "manifest_detail.json"

HEADER = (
    "# 语料清单 · MANIFEST\n\n"
    "> 本文件是语料的**指纹清单**，随版本库发布；语料文件本身不进仓库\n"
    "> （体积 + 再分发许可，见 `docs/06-语料方案.md` §6.2）。\n"
    "> 复现：`python scripts/fetch_corpus.py --channel all`，再按 manifest_detail.json 校验 sha256。\n\n"
    "| 文件 | 格式 | 来源 | 许可/公开性质 | 页数/大小 | 预期难点 | 预期 verdict |\n"
    "| --- | --- | --- | --- | --- | --- | --- |\n"
)


@dataclass
class Doc:
    """一篇已登记的语料。字段与 MANIFEST 表头一一对应。"""

    relpath: str
    fmt: str
    source: str
    license_: str
    extent: str = ""
    difficulty: str = ""
    verdict: str = "ok"
    url: str = ""
    sha256: str = ""
    fetched: str = ""

    def row(self) -> str:
        return (
            f"| `{self.relpath}` | {self.fmt} | {self.source} | {self.license_} "
            f"| {self.extent} | {self.difficulty} | {self.verdict} |"
        )


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(docs: list[Doc]) -> None:
    """登记语料，**幂等**：按 relpath 覆盖而非追加。

    刻意不做简单的追加。脚本会被反复重跑（调参数、补抓、修 bug），
    追加会让 MANIFEST 出现同一篇的多行，且只有最后一行是最新的——
    一个"看起来有 72 篇、实际重复登记了 9 篇"的清单，比没有清单更糟。

    做法：以 manifest_detail.json 为唯一数据源，按 relpath 去重后全量重建表格。
    """
    RAW.mkdir(parents=True, exist_ok=True)
    old = json.loads(DETAIL.read_text(encoding="utf-8")) if DETAIL.exists() else []
    by_path = {r["relpath"]: r for r in old}
    for d in docs:
        by_path[d.relpath] = asdict(d)

    records = sorted(by_path.values(), key=lambda r: r["relpath"])
    DETAIL.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    # 表格由明细全量重建，两者不可能不一致
    lines = [HEADER.rstrip("\n")] + [Doc(**r).row() for r in records]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def today() -> str:
    return time.strftime("%Y-%m-%d")
