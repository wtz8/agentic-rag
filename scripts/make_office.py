#!/usr/bin/env python3
"""生成 Office 类语料：DOCX 12 · PPTX 6 · XLSX 8。

与 make_text_pdf.py 同源（已抓到的开源中文技术文档），同一套取舍：
不再跟爬虫对抗，改为"真实内容 + 由我们生成格式"。

各格式刻意制造的差异（对应 §3 覆盖矩阵要触发的链路）
------------------------------------------------
* **DOCX**：用真正的 Heading 样式、多级列表、内嵌表格——解析时靠样式判断层级，
  而不是像 PDF 那样靠字号猜。
* **PPTX**：一页一个要点、大量备注栏。PPT 的难点是**文本碎片化**，
  一句话被拆进多个文本框，正文顺序与视觉顺序不一致。
* **XLSX**：多 sheet、跨表头合并单元格、空行分隔的数据区。
  XLSX 的难点是"表格没有文本段落"，整表必须原子化，不能按行切开。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_manifest import Doc, sha256_of, today, write_manifest  # noqa: E402
from fetch_corpus import RE_CN, RE_IDCARD, RE_MOBILE  # noqa: E402
from make_text_pdf import SOURCES, md_to_blocks, split_sections  # noqa: E402

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from pptx import Presentation
from pptx.util import Inches, Pt as PPt

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MDDIR = RAW / "markdown"
DIRS = {"docx": RAW / "docx", "pptx": RAW / "pptx", "xlsx": RAW / "xlsx"}

SONG, HEI = "宋体", "黑体"


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #
def _set_cjk(run, name: str, size: float, bold: bool = False) -> None:
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    # 不设置 eastAsia，中文会回退到默认西文字体，Word 里看着是宋体但解析出来字体名对不上
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def build_docx(blocks, out: Path, title: str, extra_tables=None) -> int:
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = SONG
    st.element.rPr.rFonts.set(qn("w:eastAsia"), SONG)

    doc.add_heading(title or "技术文档", level=0)
    n_tables = 0
    for kind, text in blocks:
        if kind == "table":
            rows = [[c.strip() for c in r.split("|")] for r in text.split("\n") if r.strip()]
            if not rows:
                continue
            ncol = max(len(r) for r in rows)
            rows = [r + [""] * (ncol - len(r)) for r in rows]
            t = doc.add_table(rows=len(rows), cols=ncol)
            t.style = "Table Grid"
            for i, r in enumerate(rows):
                for j, c in enumerate(r):
                    cell = t.cell(i, j)
                    cell.text = ""
                    run = cell.paragraphs[0].add_run(c[:200])
                    _set_cjk(run, SONG, 9)
            n_tables += 1
            continue

        if kind in ("h1", "h2", "h3", "h4"):
            lvl = int(kind[1])
            h = doc.add_heading(text[:120], level=min(lvl, 4))
            for r in h.runs:
                _set_cjk(r, HEI, 18 - lvl * 2, bold=True)
            continue

        # 正文：把 markdown 的 - / 数字. 还原成真正的列表样式，
        # 这样解析器要处理的是 list style 而不是行首字符
        m = re.match(r"^(\s*)[-*+]\s+(.*)$", text)
        if m:
            p = doc.add_paragraph(m.group(2)[:400], style="List Bullet" if not m.group(1) else "List Bullet 2")
        elif re.match(r"^\d+[.)]\s+", text):
            p = doc.add_paragraph(re.sub(r"^\d+[.)]\s+", "", text)[:400], style="List Number")
        else:
            p = doc.add_paragraph(text[:600])
            p.paragraph_format.first_line_indent = Pt(21)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for r in p.runs:
            _set_cjk(r, SONG, 10.5)

    # 补内嵌表格。markdown 里的表格本就稀疏，切出来的段落常常一个都没有，
    # 而 §6.1 要求 DOCX 必须含内嵌表格——没有它，"DOCX 表格提取"这条链路
    # 从头到尾不会被触发一次，写了也等于没验证。
    for ti, rows in enumerate(extra_tables or []):
        h = doc.add_heading(f"附表 {ti + 1}", level=2)
        for r in h.runs:
            _set_cjk(r, HEI, 14, bold=True)
        ncol = max(len(r) for r in rows)
        rows2 = [r + [""] * (ncol - len(r)) for r in rows]
        t = doc.add_table(rows=len(rows2), cols=ncol)
        t.style = "Table Grid"
        for i, r in enumerate(rows2):
            for j, c in enumerate(r):
                cell = t.cell(i, j)
                cell.text = ""
                run = cell.paragraphs[0].add_run(c[:200])
                _set_cjk(run, SONG, 9)
        n_tables += 1

    doc.save(str(out))
    return n_tables


# --------------------------------------------------------------------------- #
# PPTX
# --------------------------------------------------------------------------- #
def build_pptx(blocks, out: Path, title: str, max_slides: int = 28) -> int:
    prs = Presentation()
    blank = prs.slide_layouts[6]
    title_layout = prs.slide_layouts[0]

    s = prs.slides.add_slide(title_layout)
    s.shapes.title.text = (title or "技术文档")[:60]
    if len(s.placeholders) > 1:
        s.placeholders[1].text = "由开源技术文档生成 · 语料用途"

    cur_slide = None
    tf = None
    n = 0
    buf: list[str] = []

    def new_slide(head: str) -> None:
        nonlocal cur_slide, tf, buf
        cur_slide = prs.slides.add_slide(blank)
        tb = cur_slide.shapes.add_textbox(Inches(0.6), Inches(0.4), Inches(8.8), Inches(0.8))
        p = tb.text_frame.paragraphs[0]
        p.text = head[:50]
        p.font.size = PPt(24)
        p.font.bold = True
        p.font.name = HEI
        tb2 = cur_slide.shapes.add_textbox(Inches(0.7), Inches(1.4), Inches(8.6), Inches(4.6))
        tf = tb2.text_frame
        tf.word_wrap = True
        buf = []

    new_slide(title[:50])
    for kind, text in blocks:
        if len(prs.slides._sldIdLst) >= max_slides:
            break
        if kind in ("h1", "h2", "h3") and buf:
            new_slide(text[:50])
            continue
        if kind == "table":
            rows = [[c.strip() for c in r.split("|")] for r in text.split("\n") if r.strip()]
            if rows:
                new_slide(next((t for k, t in blocks if k in ("h2", "h3")), "数据表")[:50])
                ncol = max(len(r) for r in rows)
                rows = [r + [""] * (ncol - len(r)) for r in rows][:9]
                shape = cur_slide.shapes.add_table(
                    len(rows), ncol, Inches(0.7), Inches(1.5), Inches(8.6), Inches(0.4 * len(rows)))
                for i, r in enumerate(rows):
                    for j, c in enumerate(r):
                        shape.table.cell(i, j).text = c[:60]
            continue
        if not text.strip():
            continue
        # 每页最多 6 个要点，超出就翻页——这是真实 PPT 的形态，也让
        # "一句话被拆进多个文本框"这个难点成立
        if len(buf) >= 6:
            new_slide((title or "续")[:50])
        p = tf.add_paragraph()
        p.text = ("· " + re.sub(r"^[-*+]\s+", "", text))[:180]
        p.font.size = PPt(15)
        p.font.name = SONG
        buf.append(text[:40])

    # 备注栏：真实 PPT 常有大量"演讲者备注"，纯正文抽取会把它们混进正文
    for i, sl in enumerate(prs.slides):
        if sl.has_notes_slide:
            sl.notes_slide.notes_text_frame.text = f"第 {i + 1} 页要点提示：{title[:20]}…（语料备注文本）"
    prs.save(str(out))
    return len(prs.slides._sldIdLst)


# --------------------------------------------------------------------------- #
# XLSX
# --------------------------------------------------------------------------- #
def collect_tables(md_texts: list[str]) -> list[list[list[str]]]:
    """从 markdown 里抽出**中文**表格。返回 [[行, 行, ...], ...]

    技术文档里有大量纯 ASCII 表格（命令参数、配置项、URL 清单），
    它们不满足 §6.1 的"中文为主体"，留到后面再被验收淘汰等于白做一遍工作，
    因此在收集阶段就过滤掉。
    """
    tables: list[list[list[str]]] = []
    for text in md_texts:
        rows: list[list[str]] = []
        for line in text.splitlines():
            s = line.strip()
            if not s.startswith("|"):
                if rows:
                    tables.append(rows)
                    rows = []
                continue
            inner = s.strip("|")
            if re.fullmatch(r"[\s:|-]+", inner):
                continue
            rows.append([c.strip() for c in inner.split("|")])
        if rows:
            tables.append(rows)
    cn = lambda t: sum(len(RE_CN.findall(c)) for r in t for c in r)  # noqa: E731
    return [t for t in tables if len(t) >= 3 and cn(t) >= 30]


def build_xlsx(tables, out: Path, sheet_per_book: int = 4) -> int:
    """一个工作簿多个 sheet，含跨列合并表头、空行分隔数据区。"""
    wb = Workbook()
    wb.remove(wb.active)
    nsheets = 0
    for si in range(min(sheet_per_book, max(1, len(tables)))):
        rows = tables[si]
        ws = wb.create_sheet(f"数据表_{si + 1}")
        ncol = min(max(len(r) for r in rows), 12)
        rows = [r[:ncol] + [""] * max(0, ncol - len(r)) for r in rows]

        # 跨列合并的大标题 + 副表头，制造"多行表头"这一难点
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncol)
        c = ws.cell(row=1, column=1, value=f"数据表 {si + 1} · 统计口径说明")
        c.font = Font(name=HEI, size=13, bold=True)
        c.alignment = Alignment(horizontal="center")

        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncol)
        c = ws.cell(row=2, column=1, value="数据来源：开源技术文档（语料用途）")
        c.font = Font(name=SONG, size=9, italic=True)

        hdr = rows[0]
        for j, h in enumerate(hdr, start=1):
            c = ws.cell(row=3, column=j, value=h[:40])
            c.font = Font(name=HEI, size=10, bold=True)
            c.fill = PatternFill("solid", fgColor="D9D9D9")
            c.alignment = Alignment(horizontal="center", vertical="center")

        ri = 4
        for i, r in enumerate(rows[1:]):
            for j, v in enumerate(r, start=1):
                ws.cell(row=ri, column=j, value=v[:80])
            ri += 1
            if (i + 1) % 12 == 0:
                # 空行分节：真实报表里常见，解析时容易被误判成"表格到此结束"
                ri += 1
        for j in range(1, ncol + 1):
            ws.column_dimensions[get_column_letter(j)].width = 16
        nsheets += 1
    wb.save(str(out))
    return nsheets


# --------------------------------------------------------------------------- #
def audit_text(sample: str) -> tuple[bool, str]:
    """中文主体 + 无个人隐私。"""
    if len(RE_CN.findall(sample)) < 200:
        return False, "非中文主体"
    hits = len(RE_IDCARD.findall(sample)) + len(RE_MOBILE.findall(sample))
    if hits:
        return False, f"检出疑似个人隐私 {hits} 处"
    return True, ""


def make(kind: str, need: int, target: int) -> list[Doc]:
    outdir = DIRS[kind]
    outdir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list] = {}
    meta: dict[str, tuple[str, str]] = {}
    texts: list[str] = []
    for src_name, lic, repo in SOURCES:
        src = MDDIR / src_name
        if not src.exists():
            continue
        txt = src.read_text(encoding="utf-8")
        texts.append(txt)
        blocks = md_to_blocks(txt)
        groups[src_name] = split_sections(blocks, target)
        meta[src_name] = (lic, repo)

    tables = collect_tables(texts)
    keys = list(groups)
    idx = {k: 0 for k in keys}
    plan = []
    while any(idx[k] < len(groups[k]) for k in keys):
        for k in keys:
            if idx[k] < len(groups[k]):
                plan.append((k, groups[k][idx[k]], idx[k]))
                idx[k] += 1

    docs: list[Doc] = []
    made = 0

    if kind == "xlsx":
        print(f"  从开源文档抽出 {len(tables)} 个表格")
        per = max(2, min(5, len(tables) // max(1, need) or 3))
        for i in range(need):
            chunk = tables[i * per:(i + 1) * per]
            if not chunk:
                break
            out = outdir / f"data_stats_{i + 1:02d}.xlsx"
            ns = build_xlsx(chunk, out)
            ok, why = audit_text("\n".join(" ".join(r) for t in chunk for r in t))
            if not ok:
                print(f"    ✗ {out.name} 未过验收：{why}")
                out.unlink(missing_ok=True)
                continue
            made += 1
            print(f"    ✓ {out.name}  {ns} 个 sheet  {out.stat().st_size // 1024}KB")
            docs.append(Doc(
                relpath=f"xlsx/{out.name}", fmt="XLSX",
                source="由开源文档中的表格生成", license_="CC BY-NC-SA 4.0 / Apache-2.0",
                extent=f"{ns} sheet / {out.stat().st_size // 1024}KB",
                difficulty="多 sheet、跨列合并表头、空行分节；整表必须原子化",
                verdict="ok", sha256=sha256_of(out), fetched=today()))
        return docs

    for src_name, part, i in plan:
        if made >= need:
            break
        lic, repo = meta[src_name]
        short = src_name.replace("opensource_", "").replace(".md", "")
        title = next((t for k, t in part if k == "h1"), short)
        out = outdir / (f"{kind}_{short}_{i + 1:02d}.docx" if kind == "docx"
                        else f"{kind}_{short}_{i + 1:02d}.pptx")
        try:
            if kind == "docx":
                # 每篇补 2 个表格，确保"DOCX 内嵌表格"这条链路一定被触发
                start = (made * 2) % max(1, len(tables))
                extra = build_docx(part, out, title, extra_tables=tables[start:start + 2])
            else:
                extra = build_pptx(part, out, title)
        except Exception as e:
            print(f"    ✗ {out.name} 生成失败：{type(e).__name__} {e}")
            continue

        sample = "\n".join(t for _, t in part)[:20000]
        ok, why = audit_text(sample)
        if not ok:
            print(f"    ✗ {out.name} 未过验收：{why}")
            out.unlink(missing_ok=True)
            continue
        cn = len(RE_CN.findall(sample))
        # DOCX/PPTX 无"页数"概念，用中文字数折算：Word 一页约 1100 字
        est = max(1, round(cn / 1100))
        if kind == "docx" and not (5 <= est <= 80):
            print(f"    ✗ {out.name} 估算 {est} 页不在 5–80")
            out.unlink(missing_ok=True)
            continue

        made += 1
        unit = f"{extra} 个表格" if kind == "docx" else f"{extra} 页"
        print(f"    ✓ {out.name}  {unit}  中文 {cn}  估算 {est} 页")
        docs.append(Doc(
            relpath=f"{kind}/{out.name}",
            fmt="DOCX" if kind == "docx" else "PPTX",
            source=f"由开源文档生成 · {repo}", license_=lic,
            extent=f"{unit} / {out.stat().st_size // 1024}KB",
            difficulty=("Heading 样式层级、多级列表、内嵌表格" if kind == "docx"
                        else "一页一要点、文本碎片化、备注栏混杂"),
            verdict="ok", sha256=sha256_of(out), fetched=today()))
    return docs


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 Office 语料")
    ap.add_argument("--kind", default="all", choices=["all", "docx", "pptx", "xlsx"])
    args = ap.parse_args()

    plan = [("docx", 12, 9000), ("pptx", 6, 14000), ("xlsx", 8, 0)]
    total = 0
    for kind, need, target in plan:
        if args.kind != "all" and args.kind != kind:
            continue
        print(f"\n=== {kind.upper()} 目标 {need} 篇 ===")
        docs = make(kind, need, target)
        if docs:
            write_manifest(docs)
            total += len(docs)
            print(f"  登记 {len(docs)} 篇")
    print(f"\n合计登记 {total} 篇")
    return 0


if __name__ == "__main__":
    sys.exit(main())
