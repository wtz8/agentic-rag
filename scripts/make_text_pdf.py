#!/usr/bin/env python3
"""生成 PDF · 文本版语料（20 篇），内容取自已抓到的开源中文技术文档。

为什么是"生成"而不是再去找现成的 PDF
-------------------------------------
上游渠道实测下来都不通：GitHub Contents API 按 IP 限流（第一条就 403）；
Chrome/Edge 无头打印 exit=0 却不产出文件；gov.cn 政策栏目是 JS 渲染，
两层 BFS 只挖到 7 个正文页；csrc 的"规则"链接点进去是新闻发布稿而非规则全文。

继续跟反爬和前端渲染较劲，是在把时间花在**与本项目无关**的问题上——
本项目要验证的是解析与检索，不是爬虫对抗。因此改为：内容用已抓到的
真实中文技术文档（CC BY-NC-SA / Apache-2.0），格式用 reportlab 生成**带文本层**
的真 PDF。这样得到的是真实内容 + 真实格式，只是载体由我们生成。

刻意不做的事：不用 PIL 渲染（那是位图，等于又造一批扫描件）；
不伪造"来源"——MANIFEST 里逐篇写明内容源与许可，生成方式也写清楚。

验收：跑 06 §6.1 四选四硬标准。篇幅标准按文档类型分档（见 `--min-pages`），
长文档（教程/手册）要求 20–250 页。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_manifest import Doc, sha256_of, today, write_manifest  # noqa: E402
from fetch_corpus import check_pdf  # noqa: E402

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MDDIR = RAW / "markdown"
OUTDIR = RAW / "pdf_text"

F_SONG = "C:/Windows/Fonts/simsun.ttc"
F_HEI = "C:/Windows/Fonts/simhei.ttf"

# 内容源：只用已抓到的开源文档，许可随之
SOURCES = [
    ("opensource_self-llm.md", "CC BY-NC-SA 4.0", "datawhalechina/self-llm"),
    ("opensource_langchain-chatchat.md", "Apache-2.0", "chatchat-space/Langchain-Chatchat"),
    ("opensource_happy-llm.md", "CC BY-NC-SA 4.0", "datawhalechina/happy-llm"),
    ("opensource_pumpkin-book.md", "CC BY-NC-SA 4.0", "datawhalechina/pumpkin-book"),
]


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("Song", F_SONG))
    pdfmetrics.registerFont(TTFont("Hei", F_HEI))


def styles():
    """标题用黑体、正文用宋体——与中文出版习惯一致，也让解析器能靠字号区分层级。"""
    base = dict(fontName="Song", fontSize=10.5, leading=16, alignment=TA_LEFT,
                spaceAfter=4, wordWrap="CJK")
    hei = lambda size, before, after: ParagraphStyle(  # noqa: E731
        f"h{size}", fontName="Hei", fontSize=size, leading=size * 1.5,
        spaceBefore=before, spaceAfter=after, wordWrap="CJK")
    return {
        "h1": hei(17, 14, 10), "h2": hei(14, 12, 8),
        "h3": hei(12, 9, 6), "h4": hei(11, 7, 5),
        "body": ParagraphStyle("body", **base),
        "mono": ParagraphStyle("mono", fontName="Song", fontSize=8.8, leading=13,
                               wordWrap="CJK", textColor=colors.HexColor("#222222")),
    }


def md_to_blocks(text: str) -> list[tuple[str, str]]:
    """markdown → [(类型, 文本)]。类型：h1–h4 / body / table。"""
    out: list[tuple[str, str]] = []
    in_fence = False
    buf: list[str] = []

    def flush_table() -> None:
        if buf:
            out.append(("table", "\n".join(buf)))
            buf.clear()

    for raw in text.splitlines():
        s = raw.rstrip()
        if s.lstrip().startswith("```"):
            in_fence = not in_fence
            flush_table()
            continue
        if s.startswith("<!--") or s.startswith("<") or s.startswith("!["):
            continue
        if in_fence:
            flush_table()
            out.append(("body", s if s.strip() else " "))
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            flush_table()
            out.append((f"h{len(m.group(1))}", m.group(2).strip()))
            continue
        if s.lstrip().startswith("|"):
            row = s.strip().strip("|")
            if re.fullmatch(r"[\s:|-]+", row):      # |---|---| 分隔行
                continue
            buf.append(row)
            continue
        flush_table()
        if s.strip():
            out.append(("body", s.strip()))
    flush_table()
    return out


def split_sections(blocks: list[tuple[str, str]], target: int) -> list[list[tuple[str, str]]]:
    """按二级标题边界切成若干篇，尽量让每篇体量接近且结构完整。

    不按固定行数切：那样会把一个章节从中间劈开，标题层级校验（关卡 2）
    会误报"层级不连续"，而这是切分的问题不是文档的问题。
    """
    parts: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    size = 0
    for b in blocks:
        cur.append(b)
        size += len(b[1])
        if b[0] == "h2" and size >= target:
            parts.append(cur)
            cur, size = [], 0
    if size > target * 0.35:
        parts.append(cur)
    return parts


def build_pdf(blocks: list[tuple[str, str]], out: Path, title: str) -> None:
    st = styles()
    def esc(t: str) -> str:
        # reportlab 的 Paragraph 按 XML 解析，未转义的 & < > 会抛异常。
        # 用 ord() 拼数字实体，避免源码里出现会被二次解码的实体字面量。
        return "".join("&#%d;" % ord(c) if c in "&<>" else c for c in t)

    flow: list = []
    for kind, text in blocks:
        if kind == "table":
            rows = [r.split("|") for r in text.split("\n") if r.strip()]
            rows = [[c.strip() for c in r] for r in rows]
            if rows:
                t = Table(rows, colWidths=[A4[0] / max(1, len(rows[0])) - 12 * mm] * len(rows[0]),
                          repeatRows=1)
                t.setStyle(TableStyle([
                    ("FONTNAME", (0, 0), (-1, -1), "Song"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]))
                flow += [Paragraph("表：", st["h4"]), t, Spacer(1, 6)]
            continue
        style = st.get(kind, st["body"])
        flow.append(Paragraph(esc(text), style))

    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=22 * mm, rightMargin=22 * mm, topMargin=20 * mm, bottomMargin=20 * mm,
        title=title, author="agentic-rag corpus",
    )
    doc.build(flow)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成 PDF·文本版语料")
    ap.add_argument("--need", type=int, default=20)
    ap.add_argument("--target-chars", type=int, default=42000,
                    help="每篇目标字符数（决定页数，默认约 25–35 页）")
    ap.add_argument("--min-pages", type=int, default=20)
    ap.add_argument("--max-pages", type=int, default=250)
    args = ap.parse_args()

    register_fonts()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 阶段 1：先把每个源都切好
    groups: dict[str, list] = {}
    meta: dict[str, tuple[str, str]] = {}
    for src_name, lic, repo in SOURCES:
        src = MDDIR / src_name
        if not src.exists():
            print(f"  · 缺少 {src_name}，跳过")
            continue
        blocks = md_to_blocks(src.read_text(encoding="utf-8"))
        parts = split_sections(blocks, args.target_chars)
        groups[src_name] = parts
        meta[src_name] = (lic, repo)
        print(f"  [{src_name}] {len(blocks)} 段 → {len(parts)} 篇")

    # 阶段 2：按源交错取。
    # 若按源顺序取，self-llm 一个仓库就能填满 20 篇配额，语料会严重同质化——
    # 同一批文档的行文、术语、结构高度相似，检索难度反而被低估。
    plan: list[tuple[str, list, int]] = []
    keys = list(groups)
    idx = {k: 0 for k in keys}
    while any(idx[k] < len(groups[k]) for k in keys):
        for k in keys:
            if idx[k] < len(groups[k]):
                plan.append((k, groups[k][idx[k]], idx[k]))
                idx[k] += 1
    print(f"  交错排序后共 {len(plan)} 篇待生成\n")

    docs: list[Doc] = []
    made = 0
    for src_name, part, i in plan:
        if made >= args.need:
            break
        lic, repo = meta[src_name]
        title = next((t for k, t in part if k == "h1"), src_name.replace(".md", ""))
        fname = f"text_{src_name.replace('opensource_', '').replace('.md', '')}_{i + 1:02d}.pdf"
        out = OUTDIR / fname
        try:
            build_pdf(part, out, title)
        except Exception as e:
            print(f"    ✗ {fname} 生成失败：{type(e).__name__} {e}")
            continue

        ok, info = check_pdf(out)
        if not ok:
            print(f"    ✗ {fname} 未过验收：{info.get('reject') or info.get('error')}")
            out.unlink(missing_ok=True)
            continue
        n = info["pages"]
        if not (args.min_pages <= n <= args.max_pages):
            print(f"    ✗ {fname} 篇幅 {n} 页不在 {args.min_pages}–{args.max_pages}")
            out.unlink(missing_ok=True)
            continue

        made += 1
        size_kb = out.stat().st_size / 1024
        print(f"    ✓ {fname}  {n}页 {size_kb:.0f}KB  中文 {info['cn_chars_sampled']}  数值行 {info['numeric_rows']}")
        docs.append(Doc(
            relpath=f"pdf_text/{fname}",
            fmt="PDF · 文本版",
            source=f"由开源文档生成 · {repo}",
            license_=lic,
            extent=f"{n}页 / {size_kb:.0f}KB",
            difficulty="标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配",
            verdict="ok",
            url=f"内容源：{src_name}",
            sha256=sha256_of(out),
            fetched=today(),
        ))

    if docs:
        write_manifest(docs)
        print(f"\n登记 {len(docs)} 篇 → data/raw/MANIFEST.md")
    print(f"本轮共生成 {made} 篇（目标 {args.need}）")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
