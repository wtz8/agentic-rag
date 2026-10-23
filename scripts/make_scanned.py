#!/usr/bin/env python3
"""按 docs/06-语料方案.md §6.3 自制 3 份扫描件（无文本层 PDF）。

设计原则
--------
**三份内容完全相同，只变难度。** 这不是偷懒——只有单变量对比，才能干净地回答
"难度梯度对 OCR 的影响"。若三份内容也不同，某份 OCR 失败时分不清是内容难还是版面难。

内容源用开源许可文档（CC BY-NC-SA 4.0 / Apache-2.0），不用受版权保护的材料。

制作方式为**程序合成**而非"打印—扫描"，理由见 §6.3 的初衷：自制的价值在于可控。
程序合成比物理扫描更可控——难度参数可复现、可写进脚本，任何人跑一次得到同一批文件。
代价是缺少真实纸张纹理与扫描仪 CCD 噪声，这一点在 MANIFEST 里如实标注。

验收（必须实测，不凭"我觉得是扫描件"）
--------------------------------------
用 pypdf 提取每页文本，每页字符数必须 < `scanned_chars_per_page`（默认 50），
即关卡 1 确实会把它判为扫描件。某份提取出了文本 = 制作失败，必须重做。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_manifest import Doc, sha256_of, today, write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUTDIR = RAW / "pdf_scanned"

# 内容源许可：三份扫描件渲染的是同一份开源文档，许可随之
SOURCE_LICENSE = "内容源 CC BY-NC-SA 4.0 · datawhalechina/happy-llm"

# 预期 verdict：
#   前两份应能 OCR 入库；第三份刻意压到阈值边缘——它的用途是验证
#   on_reject 拒答分支确实可达（06 §3：一个从不触发的拒绝逻辑等于没有）。
VERDICT = {
    "干净": "ok",
    "倾斜+阴影": "ok",
    "低对比度+噪点": "ok/rejected 均可（阈值边缘，验证 on_reject 可达）",
}

# ---------------------------------------------------------------- 字体与版面
F_CJK = "C:/Windows/Fonts/msyh.ttc"        # 微软雅黑
F_CJK_B = "C:/Windows/Fonts/msyhbd.ttc"    # 雅黑粗体（标题）
F_MONO = "C:/Windows/Fonts/simsun.ttc"     # 宋体：中文等宽，适合表格对齐

DPI = 300
A4 = (int(8.27 * DPI), int(11.69 * DPI))   # 2480 x 3507
MARGIN = int(0.75 * DPI)                   # 0.75 英寸页边距
PT = DPI / 72.0                            # 1pt 对应的像素数

SIZE = {"h1": 20, "h2": 15, "h3": 13, "h4": 12, "body": 10.5, "mono": 9.5}
GAP_AFTER = {"h1": 26, "h2": 20, "h3": 14, "h4": 10, "body": 4, "mono": 0}

_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(path: str, pt: float) -> ImageFont.FreeTypeFont:
    key = (path, round(pt * PT))
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, int(round(pt * PT)))
    return _font_cache[key]


# ---------------------------------------------------------------- 内容选取
def pick_content(md_path: Path, target_chars: int = 13000) -> list[tuple[str, str]]:
    """从 markdown 取一段内容，返回 [(类型, 文本)]。

    类型决定字号与字体：h1–h4 / body / mono（表格与代码块）。
    保留表格与代码块，是因为"扫描件里有表格"才能验证 OCR 之后的表格还原链路。
    """
    raw_lines = md_path.read_text(encoding="utf-8").splitlines()
    flat: list[tuple[str, str]] = []
    in_fence = False
    for s in (l.rstrip() for l in raw_lines):
        if s.lstrip().startswith("```"):
            in_fence = not in_fence
            flat.append(("mono", s.strip()))
            continue
        if s.startswith("<!--") or s.startswith("<") or s.startswith("!["):
            continue
        if in_fence:
            flat.append(("mono", s if s.strip() else " "))
            continue
        if s.startswith("#### "):
            flat.append(("h4", s[5:].strip()))
        elif s.startswith("### "):
            flat.append(("h3", s[4:].strip()))
        elif s.startswith("## "):
            flat.append(("h2", s[3:].strip()))
        elif s.startswith("# "):
            flat.append(("h1", s[2:].strip()))
        elif s.lstrip().startswith("|"):
            flat.append(("mono", s.strip()))
        else:
            flat.append(("body", s))

    # 从第一个二级标题起取，跳过封面与目录
    start = next((i for i, (k, _) in enumerate(flat) if k == "h2"), 0)
    out: list[tuple[str, str]] = []
    total = 0
    for k, t in flat[start:]:
        out.append((k, t))
        total += len(t)
        if total > target_chars:
            break

    # 扫描件里没有表格，就测不到"OCR → 表格还原"这条链路。
    # 正文前段常常没有表格（实测 happy-llm 前 13000 字表格行数为 0），
    # 因此不足时从全文里捞表格块补到末尾，并给它一个标题以保持层级连续。
    have = sum(1 for k, t in out if k == "mono" and t.startswith("|"))
    if have < 12:
        i, added = 0, 0
        while i < len(flat) and have + added < 12:
            if flat[i][0] == "mono" and flat[i][1].startswith("|"):
                j = i
                while j < len(flat) and flat[j][0] == "mono" and flat[j][1].startswith("|"):
                    j += 1
                if j - i >= 3:
                    out.append(("h2", "附录：数据表"))
                    out.extend(flat[i:j])
                    added += j - i
                i = j
            else:
                i += 1
    return out


# ---------------------------------------------------------------- 排版与渲染
def wrap(text: str, f: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """按像素宽度折行。中文逐字断行；西文优先在空格处断，避免 OCR 把单词切碎。"""
    if not text:
        return [""]
    lines: list[str] = []
    cur = ""
    for ch in text:
        if f.getlength(cur + ch) <= max_w or not cur:
            cur += ch
            continue
        if ch.isascii() and ch.isalpha() and " " in cur:
            i = cur.rfind(" ")
            head, cur = cur[:i], cur[i + 1:] + ch
            lines.append(head)
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def render_pages(content: list[tuple[str, str]], max_pages: int = 14) -> list[Image.Image]:
    pages: list[Image.Image] = []
    img = Image.new("L", A4, 255)
    draw = ImageDraw.Draw(img)
    y = MARGIN
    max_w = A4[0] - 2 * MARGIN

    def newpage() -> None:
        nonlocal img, draw, y
        pages.append(img)
        img = Image.new("L", A4, 255)
        draw = ImageDraw.Draw(img)
        y = MARGIN

    for kind, text in content:
        pt = SIZE[kind]
        f = font(F_CJK_B if kind in ("h1", "h2", "h3", "h4") else
                 (F_MONO if kind == "mono" else F_CJK), pt)
        line_h = int(pt * PT * 1.55)
        for wl in wrap(text, f, max_w):
            if y + line_h > A4[1] - MARGIN:
                newpage()
                if len(pages) >= max_pages:
                    return pages
            draw.text((MARGIN, y), wl, font=f, fill=0)
            y += line_h
        y += GAP_AFTER[kind]
    pages.append(img)
    return pages[:max_pages]


# ---------------------------------------------------------------- 三档难度
def degrade_clean(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """干净：正放、无噪声，作为 OCR 能力上限的基线。"""
    return img


def degrade_skew_shadow(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """中等：整体倾斜 + 装订侧渐变阴影。

    倾斜角度取 1.5–2.5 度——真实扫描仪的进纸偏差量级。刻意不用大角度：
    大角度属于"扫描失败"而非"扫描质量差"，会掩盖版面矫正本身有没有生效。
    """
    angle = float(rng.uniform(1.5, 2.5))
    a = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
    arr = np.asarray(a).astype(np.float32)
    h, w = arr.shape
    grad = np.linspace(0.70, 1.0, w, dtype=np.float32)          # 左暗右亮
    grad = grad[None, :] * (0.97 + 0.03 * np.sin(np.linspace(0, 3.0, h)))[:, None]
    return Image.fromarray(np.clip(arr * grad, 0, 255).astype(np.uint8), mode="L")


def degrade_low_contrast(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """困难：低对比度 + 噪点 + 轻微失焦，逼近 OCR 可用下限。

    动态范围压到约 42%（白底 232、黑字约 125），叠加高斯噪点与模糊。
    这一份的用途是验证 `on_reject` 拒答分支**确实会触发**——
    一个从不触发的拒绝逻辑等于没有（06 §3 同一条原则）。

    `rng` 由调用方持有并跨页推进：若在函数内现场创建生成器，每页都会拿到
    同一段噪声序列，页面之间的噪点完全雷同，一眼就是合成的。
    """
    arr = np.asarray(img).astype(np.float32)
    arr = 232.0 - (255.0 - arr) * 0.42                      # 压缩动态范围
    arr = arr + rng.normal(0, 13, arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    return out.filter(ImageFilter.GaussianBlur(0.7))


SPECS = [
    ("scanned_clean.pdf", "干净", degrade_clean, 95),
    ("scanned_skew_shadow.pdf", "倾斜+阴影", degrade_skew_shadow, 88),
    ("scanned_lowcontrast.pdf", "低对比度+噪点", degrade_low_contrast, 70),
]


# ---------------------------------------------------------------- 验收
def verify(path: Path, threshold: int = 50) -> tuple[list[int], bool]:
    """实测每页可提取字符数。全部 < threshold 才算合格。"""
    reader = PdfReader(str(path))
    counts = [len((p.extract_text() or "").strip()) for p in reader.pages]
    return counts, all(c < threshold for c in counts)


def main() -> int:
    ap = argparse.ArgumentParser(description="自制扫描件语料")
    ap.add_argument("--source", default="data/raw/markdown/opensource_happy-llm.md")
    ap.add_argument("--chars", type=int, default=13000, help="取用内容字符数")
    ap.add_argument("--max-pages", type=int, default=14)
    args = ap.parse_args()

    src = ROOT / args.source
    if not src.exists():
        print(f"内容源不存在：{src}\n请先跑 scripts/fetch_corpus.py --channel opensource")
        return 2

    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261021)
    content = pick_content(src, args.chars)
    n_tbl = sum(1 for k, t in content if k == "mono" and t.startswith("|"))
    n_code = sum(1 for k, t in content if k == "mono" and not t.startswith("|"))
    n_head = sum(1 for k, _ in content if k in ("h1", "h2", "h3", "h4"))
    print(f"内容源：{src.name}  {len(content)} 段  标题 {n_head} / 表格行 {n_tbl} / 代码行 {n_code}")

    pages = render_pages(content, args.max_pages)
    print(f"排版完成：{len(pages)} 页 @ {DPI}dpi\n")

    docs: list[Doc] = []
    rows = []
    all_ok = True
    for name, desc, fn, quality in SPECS:
        imgs = [fn(p, rng) for p in pages]
        out = OUTDIR / name
        imgs[0].save(out, "PDF", save_all=True, append_images=imgs[1:],
                     resolution=float(DPI), quality=quality)
        counts, ok = verify(out)
        all_ok &= ok
        size_kb = out.stat().st_size / 1024
        mx = max(counts) if counts else 0
        flag = "✓" if ok else "✗"
        print(f"{flag} {name:32s} {desc:12s} {len(imgs)}页 {size_kb:7.0f}KB  "
              f"每页最大可提取字符数 = {mx}")
        rows.append((name, desc, len(imgs), size_kb, mx, ok))
        docs.append(Doc(
            relpath=f"pdf_scanned/{name}",
            fmt="PDF · 扫描件",
            source="自制（程序合成，非物理扫描）",
            license_=SOURCE_LICENSE,
            extent=f"{len(imgs)}页 / {size_kb:.0f}KB",
            difficulty=f"{desc}；每页最大可提取字符 {mx}",
            verdict=VERDICT[desc] if ok else "rejected（OCR 不可用）",
            sha256=sha256_of(out),
            fetched=today(),
        ))

    print()
    if not all_ok:
        print("验收失败：存在可提取文本的页，说明该份制作不合格，必须重做")
        return 1
    print("验收通过：三份均无文本层（每页 < 50 字符），关卡 1 会判定为扫描件")

    write_manifest(docs)
    print(f"登记 {len(docs)} 篇 → data/raw/MANIFEST.md")

    print("\n| 文件 | 难度 | 页数 | 大小 | 每页最大字符数 | 验收 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for n, d, p, s, m, ok in rows:
        print(f"| `{n}` | {d} | {p} | {s:.0f}KB | {m} | {'通过' if ok else '失败'} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
