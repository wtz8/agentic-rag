"""PDF 后端工厂（03 §4 的 J10）。

探测（`probe_before_parse`）、raw 基线（`raw_baseline`）与 `PdfParser` 必须共用同一个后端，
否则 `PDF_BACKEND=pypdf` 这条 AGPL 规避只生效一半——主解析路径改了、配角没改。

已知代价（写在 03 §4）：`pypdf` 对线条对象的统计能力弱于 `pymupdf`，
因此 `table_density` 在 pypdf 后端下可能恒为 0 → 表格密集走 docling 的分支不可达。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..config import Config


class PdfBackend(Protocol):
    name: str

    def page_count(self, path: Path) -> int: ...

    def page_texts(self, path: Path) -> list[str]:
        """逐页纯文本流（raw 基线用；words 级更贴近底层全量，见 03 §4.3.0.1）。"""

    def is_encrypted(self, path: Path) -> bool: ...

    def table_density(self, path: Path) -> float:
        """表格线条对象占比，0 表示探测不到（pypdf 后端的已知代价）。"""

    def count_tables(self, path: Path) -> int: ...


class PyMuPdfBackend:
    name = "pymupdf"

    def _doc(self, path: Path):
        import fitz  # pymupdf

        return fitz.open(str(path))

    def page_count(self, path: Path) -> int:
        with self._doc(path) as d:
            return d.page_count

    def page_texts(self, path: Path) -> list[str]:
        with self._doc(path) as d:
            out = []
            for page in d:
                words = page.get_text("words")
                out.append(" ".join(w[4] for w in words) if words else page.get_text("text"))
            return out

    def is_encrypted(self, path: Path) -> bool:
        with self._doc(path) as d:
            return bool(d.is_encrypted)

    def table_density(self, path: Path) -> float:
        with self._doc(path) as d:
            total_lines = 0
            pages = 0
            for page in d:
                pages += 1
                try:
                    total_lines += len(page.get_drawings())
                except Exception:  # noqa: BLE001
                    pass
                try:
                    total_lines += len(page.find_tables().tables)
                except Exception:  # noqa: BLE001
                    pass
            if pages == 0:
                return 0.0
            return round(min(1.0, total_lines / (pages * 20.0)), 4)

    def count_tables(self, path: Path) -> int:
        with self._doc(path) as d:
            n = 0
            for page in d:
                try:
                    n += len(page.find_tables().tables)
                except Exception:  # noqa: BLE001
                    pass
            return n


class PypdfBackend:
    name = "pypdf"

    def _reader(self, path: Path):
        from pypdf import PdfReader

        return PdfReader(str(path))

    def page_count(self, path: Path) -> int:
        try:
            return len(self._reader(path).pages)
        except Exception:  # noqa: BLE001
            return 0

    def page_texts(self, path: Path) -> list[str]:
        try:
            reader = self._reader(path)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:  # noqa: BLE001
                    return []
            return [(p.extract_text() or "") for p in reader.pages]
        except Exception:  # noqa: BLE001
            return []

    def is_encrypted(self, path: Path) -> bool:
        try:
            return bool(self._reader(path).is_encrypted)
        except Exception:  # noqa: BLE001
            return False

    def table_density(self, path: Path) -> float:
        # pypdf 不暴露绘图对象，表格密度探测能力弱于 pymupdf：恒为 0 是已知代价
        return 0.0

    def count_tables(self, path: Path) -> int:
        return 0


def get_pdf_backend(cfg: Config) -> PdfBackend:
    name = getattr(cfg.ingestion, "pdf_backend", "pymupdf")
    if name == "pypdf":
        return PypdfBackend()
    return PyMuPdfBackend()


__all__ = ["PdfBackend", "PyMuPdfBackend", "PypdfBackend", "get_pdf_backend"]
