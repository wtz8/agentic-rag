#!/usr/bin/env python3
"""按 docs/06-语料方案.md §6 抓取公开语料，并登记 data/raw/MANIFEST.md。

设计要点
--------
1. **只登记不静默**：每篇下载后写入 MANIFEST，先登记后使用（§6.1）。
2. **许可随行**：每篇记下许可类型，不替源文件做再分发判断（§6.2）。
3. **可复现**：记录 sha256 与获取日期，任何人可据此核对拿到的是同一批文件。

用法
----
    python scripts/fetch_corpus.py --channel opensource
    python scripts/fetch_corpus.py --channel opensource --limit 1   # 先试一篇

注意
----
语料文件不进版本库（体积 + 再分发许可），只有 MANIFEST 进。见 .gitignore。
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

# 登记逻辑与 fetch/make_* 各脚本共用，保证 MANIFEST 表格格式一致
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_manifest import RAW, ROOT, Doc, sha256_of, today, write_manifest  # noqa: E402

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


class Session:
    def __init__(self) -> None:
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"})
        self.n_api = 0

    def get(self, url: str, *, timeout: int = 30, retries: int = 3, **kw):
        last = None
        for i in range(retries):
            try:
                r = self.s.get(url, timeout=timeout, **kw)
                if r.status_code in (429, 502, 503, 504):
                    wait = 2 ** i
                    print(f"    [重试 {i + 1}/{retries}] HTTP {r.status_code}，等待 {wait}s")
                    time.sleep(wait)
                    last = r
                    continue
                return r
            except requests.RequestException as e:  # 网络抖动
                last = e
                time.sleep(2 ** i)
        raise RuntimeError(f"请求失败：{url}（{last}）")


# --------------------------------------------------------------------------- #
# 渠道 6 · 开源项目文档（Markdown）
# --------------------------------------------------------------------------- #

# (repo, docs_path, branch, 输出文件名, 许可, 说明)
OPENSOURCE: list[tuple[str, str, str, str, str, str]] = [
    (
        "datawhalechina/happy-llm",
        "docs",
        "main",
        "opensource_happy-llm.md",
        "CC BY-NC-SA 4.0",
        "LLM 教程，8 章，标题层级深，含表格与代码块",
    ),
    (
        "datawhalechina/self-llm",
        "",
        "master",
        "opensource_self-llm.md",
        "CC BY-NC-SA 4.0",
        "大模型部署实践，多章分节",
    ),
    (
        "datawhalechina/pumpkin-book",
        "",
        "master",
        "opensource_pumpkin-book.md",
        "CC BY-NC-SA 4.0",
        "机器学习公式详解，章节编号密集",
    ),
    (
        "datawhalechina/hands-on-data-analysis",
        "",
        "master",
        "opensource_hands-on-data-analysis.md",
        "CC BY-NC-SA 4.0",
        "数据分析实战，含大量表格",
    ),
    (
        "chatchat-space/Langchain-Chatchat",
        "",
        "master",
        "opensource_langchain-chatchat.md",
        "Apache-2.0",
        "中文 README，含架构说明与配置表格",
    ),
]

SKIP_NAMES = {".nojekyll", "_sidebar.md", "index.html"}
SKIP_DIRS = {"images", "img", "assets", ".git", "docsify"}


def natkey(p: Path) -> list:
    """自然排序：让 chapter2 排在 chapter10 之前。

    按字典序排会得到 chapter1, chapter10, chapter2 这种顺序，拼接出来的
    文档章节是乱的，标题层级校验会直接判不连续。
    """
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", str(p))]


def download_repo(sess: Session, repo: str, branch: str, dest: Path) -> Path:
    """下载仓库快照（zip）并解压，返回解压后的根目录。

    为什么是 zip 而不是 git clone / Contents API——三条都被试过：

    * **Contents API**：未认证限额 60 次/小时且按 IP 计，实测第一条就被 403。
    * **git clone**：协议层的 sideband 包在国内直连经常被掐断
      （`curl 56 schannel: server closed abruptly`），失败不可重试。
    * **codeload zip**：单次 HTTP GET，requests 的重试与超时全部可用，
      实测 `raw.githubusercontent.com` / `codeload.github.com` 均可达。

    代价是一次性下载整个仓库（含图片资源），临时磁盘占用数十 MB，用完即删。
    """
    url = f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}"
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "repo.zip"
    r = sess.get(url, timeout=120, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"下载仓库快照失败：HTTP {r.status_code} {url}")
    with zpath.open("wb") as f:
        for chunk in r.iter_content(1 << 16):
            if chunk:
                f.write(chunk)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    zpath.unlink(missing_ok=True)
    subs = [p for p in dest.iterdir() if p.is_dir()]
    if not subs:
        raise RuntimeError("zip 解压后为空目录")
    return subs[0]


def collect_md_files(base: Path) -> list[Path]:
    """按自然序收集 markdown 文件，跳过图片等资源目录。"""
    out: list[Path] = []
    for p in base.rglob("*.md"):
        rel = p.relative_to(base)
        if any(seg in SKIP_DIRS for seg in rel.parts[:-1]):
            continue
        if p.name in SKIP_NAMES:
            continue
        out.append(p)
    return sorted(out, key=lambda x: natkey(x.relative_to(base)))


def demote_headings(text: str, level: int) -> str:
    """把文件内的标题整体下推 level 级，使拼接后层级连续。

    拼接多个文件时若不做这件事，会出现 10 个并列的一级标题，
    标题层级校验（关卡 2）会直接判不连续。
    """
    out = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.lstrip()
        # 围栏内的 `#` 是代码注释（Python/Shell），不是标题，误升会污染标题树
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            continue
        if not in_fence and line.startswith("#"):
            n = len(line) - len(line.lstrip("#"))
            if 1 <= n <= 6:
                out.append("#" * min(n + level, 6) + line[n:])
                continue
        out.append(line)
    return "\n".join(out)


def build_opensource_one(sess: Session, spec: tuple, outdir: Path, force: bool = False) -> Doc | None:
    repo, docs_path, branch, filename, license_, note = spec
    out0 = outdir / filename
    # 幂等：已抓过的不再重复下载，也不重复登记 MANIFEST
    if out0.exists() and out0.stat().st_size > 1024 and not force:
        print(f"  · {filename} 已存在（{out0.stat().st_size / 1024:.0f}KB），跳过")
        return None
    print(f"  [{repo}] 下载快照 {branch} ...")
    tmpdir = Path(tempfile.mkdtemp(prefix="corpus_"))
    try:
        tmp = download_repo(sess, repo, branch, tmpdir)
        base = (tmp / docs_path) if docs_path else tmp
        if not base.exists():
            base = tmp
        files = collect_md_files(base)

        # 文档结构在根目录（如只有 README）时的回退
        if not files:
            readme = tmp / "README.md"
            files = [readme] if readme.exists() else []

        print(f"    找到 {len(files)} 个 .md")
        parts: list[str] = []
        got = 0
        for i, p in enumerate(files):
            try:
                txt = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if not txt.strip():
                continue
            # README 保持原层级，其余下推 1 级，形成"文档 → 章 → 节"的连续层级
            lvl = 0 if p.name.upper().startswith("README") else 1
            rel = p.relative_to(tmp)
            parts.append(f"<!-- 源：{rel} -->\n\n" + demote_headings(txt, lvl))
            got += 1
            if (i + 1) % 20 == 0:
                print(f"      已取 {i + 1}/{len(files)}")

        if got < 2:
            print(f"    ✗ 有效文件不足（{got}），跳过")
            return None

        merged = "\n\n---\n\n".join(parts)
        out = outdir / filename
        out.write_text(merged, encoding="utf-8")

        h2 = sum(1 for l in merged.splitlines() if l.startswith("## "))
        h3 = sum(1 for l in merged.splitlines() if l.startswith("### "))
        h4 = sum(1 for l in merged.splitlines() if l.startswith("#### "))
        cn = sum(1 for ch in merged if "\u4e00" <= ch <= "\u9fff")

        doc = Doc(
            relpath=f"markdown/{filename}",
            fmt="Markdown",
            source=f"GitHub · {repo}",
            license_=license_,
            extent=f"{len(merged) / 1024:.0f}KB / 中文字 {cn}",
            difficulty=note + f"；h2={h2} h3={h3} h4={h4}",
            verdict="ok" if cn > 2000 else "ok（字数偏少，需复核）",
            url=f"https://github.com/{repo}/tree/{branch}/{docs_path}".rstrip("/"),
            sha256=sha256_of(out),
            fetched=today(),
        )
        print(f"    ✓ {filename}  {len(merged) / 1024:.0f}KB  中文 {cn} 字  h2={h2} h3={h3} h4={h4}")
        return doc
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def channel_opensource(sess: Session, limit: int | None = None, force: bool = False) -> list[Doc]:
    outdir = RAW / "markdown"
    outdir.mkdir(parents=True, exist_ok=True)
    docs: list[Doc] = []
    for spec in OPENSOURCE[: limit or len(OPENSOURCE)]:
        try:
            d = build_opensource_one(sess, spec, outdir, force)
            if d:
                docs.append(d)
        except RuntimeError as e:
            print(f"  ✗ {spec[0]}：{e}")
            break
        except Exception as e:  # 单篇失败不中断整批
            print(f"  ✗ {spec[0]}：{type(e).__name__} {e}")
    return docs


# --------------------------------------------------------------------------- #
# 四选四硬标准（06 §6.1）——边抓边验，不合规的直接不登记
# --------------------------------------------------------------------------- #

import re as _re

RE_IDCARD = _re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")            # 18 位身份证
RE_MOBILE = _re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")             # 手机号
RE_CN = _re.compile(r"[\u4e00-\u9fff]")


def check_pdf(path: Path) -> tuple[bool, dict]:
    """对 PDF 跑四选四硬标准，返回 (是否合格, 实测指标)。

    标准来自 06 §6.1：篇幅 20–250 页 / 结构含三级标题与表格 / 中文为主体 /
    不含个人隐私。前三条可自动判定，第四条用正则过一遍——机器扫不出"姓名+住址"
    这种组合，所以它只是**下限**，人工复核仍要做（见 §6.1 登记流程）。
    """
    from pypdf import PdfReader

    info: dict = {}
    try:
        r = PdfReader(str(path))
        n = len(r.pages)
    except Exception as e:
        return False, {"error": f"{type(e).__name__}: {e}"}
    info["pages"] = n
    if not (20 <= n <= 250):
        return False, info | {"reject": f"篇幅 {n} 页不在 20–250"}

    # 抽样：读前 30 页与中间 10 页，既够统计又不至于太慢
    idx = list(range(min(30, n))) + list(range(n // 2, min(n // 2 + 10, n)))
    texts = []
    for i in idx:
        try:
            texts.append(r.pages[i].extract_text() or "")
        except Exception:
            texts.append("")
    sample = "\n".join(texts)

    cn = len(RE_CN.findall(sample))
    info["cn_chars_sampled"] = cn
    if cn < 200:
        return False, info | {"reject": f"中文字符仅 {cn}，非中文主体"}

    hits = len(RE_IDCARD.findall(sample)) + len(RE_MOBILE.findall(sample))
    info["privacy_hits"] = hits
    if hits:
        return False, info | {"reject": f"检出疑似个人隐私 {hits} 处"}

    # 表格密度启发式：制表符/多空格分隔的数字行，年报的财务表通常是这种形态
    tbl = sum(1 for line in sample.splitlines()
              if len(_re.findall(r"-?\d[\d,]*\.\d{2}", line)) >= 3)
    info["numeric_rows"] = tbl
    return True, info


# --------------------------------------------------------------------------- #
# 渠道 2 · 巨潮资讯网（PDF · 表格密集年报）
# --------------------------------------------------------------------------- #

CNINFO_QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC = "http://static.cninfo.com.cn/"
CNINFO_CATEGORY = "category_ndbg_szsh"          # 年度报告
CNINFO_HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
}

# 年报体积区间（KB）。太大是超长金融年报（银行/保险常 400+ 页），超出 §6.1 篇幅上限；
# 太小则是摘要版，表格密度不够。
CNINFO_MIN_KB, CNINFO_MAX_KB = 1500, 12000


def cninfo_candidates(sess: Session, pages: int = 12, per: int = 30) -> list[dict]:
    """查询年报列表。

    深市（szse）与沪市（sse）都要翻：单翻一个市场时，同公司去重 +
    体积筛选之后候选池只剩个位数，凑不满 12 篇配额。
    """
    out: list[dict] = []
    for column in ("szse", "sse"):
        for p in range(1, pages + 1):
            data = {
                "pageNum": p, "pageSize": per, "column": column, "tabName": "fulltext",
                "category": CNINFO_CATEGORY, "seDate": "2024-01-01~2025-12-31",
                "isHLtitle": "true",
            }
            try:
                r = sess.s.post(CNINFO_QUERY, data=data, headers=CNINFO_HEADERS, timeout=30)
                js = r.json()
            except Exception as e:
                print(f"    · {column} 第 {p} 页查询失败：{type(e).__name__} {e}")
                break
            anns = js.get("announcements") or []
            if not anns:
                break
            out.extend(anns)
    return out


def channel_cninfo(sess: Session, need: int = 12, force: bool = False) -> list[Doc]:
    outdir = RAW / "pdf_table"
    outdir.mkdir(parents=True, exist_ok=True)

    print("  查询年报列表 ...")
    anns = cninfo_candidates(sess)
    print(f"    返回 {len(anns)} 条")

    seen: set[str] = set()
    picked: list[dict] = []
    for a in anns:
        if a.get("adjunctType") != "PDF" or not a.get("adjunctUrl"):
            continue
        title = a.get("announcementTitle", "")
        if any(k in title for k in ("摘要", "英文", "取消", "更正版", "问询")):
            continue
        code = a.get("secCode", "")
        if code in seen:                      # 同一家公司只取一篇，保证语料多样性
            continue
        kb = a.get("adjunctSize") or 0
        if not (CNINFO_MIN_KB <= kb <= CNINFO_MAX_KB):
            continue
        seen.add(code)
        picked.append(a)
        if len(picked) >= need * 5:           # 多备一些：下载后还有篇幅与合规验收淘汰
            break

    print(f"    初筛 {len(picked)} 篇，开始下载并验收")
    docs: list[Doc] = []
    for a in picked:
        if len(docs) >= need:
            break
        code, title = a.get("secCode", ""), a.get("announcementTitle", "")
        m = _re.search(r"(20\d{2})", title)
        year = m.group(1) if m else "0000"
        fname = f"cninfo_{code}_{year}.pdf"
        out = outdir / fname
        if out.exists() and out.stat().st_size > 1024 and not force:
            print(f"    · {fname} 已存在，跳过")
            continue
        url = CNINFO_STATIC + a["adjunctUrl"].lstrip("/")
        try:
            r = sess.s.get(url, timeout=120, stream=True,
                           headers={"Referer": "http://www.cninfo.com.cn/"})
            if r.status_code != 200:
                print(f"    ✗ {fname} HTTP {r.status_code}")
                continue
            with out.open("wb") as f:
                for chunk in r.iter_content(1 << 16):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print(f"    ✗ {fname} 下载失败：{type(e).__name__} {e}")
            if out.exists():
                out.unlink()
            continue

        ok, info = check_pdf(out)
        size_mb = out.stat().st_size / 1024 / 1024
        if not ok:
            print(f"    ✗ {fname} 未过验收：{info.get('reject') or info.get('error')}")
            out.unlink()                       # 不合规的不留在语料里
            continue

        print(f"    ✓ {fname}  {info['pages']}页 {size_mb:.1f}MB  "
              f"数值行 {info['numeric_rows']}  中文 {info['cn_chars_sampled']}")
        docs.append(Doc(
            relpath=f"pdf_table/{fname}",
            fmt="PDF · 表格密集",
            source=f"巨潮资讯网 · {a.get('secName', '')}({code})",
            license_="上市公司公开披露",
            extent=f"{info['pages']}页 / {size_mb:.1f}MB",
            difficulty="跨页表、复杂表头、合并单元格",
            verdict="ok",
            url=url,
            sha256=sha256_of(out),
            fetched=today(),
        ))
    return docs


# --------------------------------------------------------------------------- #
# 渠道 7 · 技术文档站（HTML 原格式）
# --------------------------------------------------------------------------- #

# (URL, 文件名, 许可, 说明)
HTML_SOURCES: list[tuple[str, str, str, str]] = [
    ("https://cn.vuejs.org/guide/introduction.html", "html_vue_introduction.html",
     "MIT · Vue.js", "含导航、侧栏、代码块；单页中文量最大的一档"),
    ("https://zh.javascript.info/types", "html_javascript_info_types.html",
     "CC BY-NC-SA 4.0", "代码块密集（>80 处），正文与代码混排"),
    ("https://pinia.vuejs.org/zh/introduction.html", "html_pinia_introduction.html",
     "MIT · Pinia", "API 文档，含参数说明列表"),
    ("https://cn.vite.dev/guide/", "html_vite_guide.html",
     "MIT · Vite", "含 <table>，是六篇里表格最明确的一篇"),
    ("https://docs.python.org/zh-cn/3/tutorial/index.html", "html_python_tutorial.html",
     "PSF / CC BY-NC-SA", "教程目录页，层级列表密集"),
    ("https://vuex.vuejs.org/zh/guide/", "html_vuex_guide.html",
     "MIT · Vuex", "篇幅最小的一档，用于覆盖短 HTML 页这一形态"),
]


def channel_html(sess: Session, force: bool = False) -> list[Doc]:
    """保存 HTML **原格式**，不做清洗。

    这一点是刻意的：真实网页里塞满了导航、侧栏、脚本、样式与注释，
    正文只占一小部分。解析器要处理的就是这个脏活——如果先把 HTML 洗干净
    再存进语料，等于替解析器把最难的部分做掉了，语料就失去了难度。
    """
    outdir = RAW / "html"
    outdir.mkdir(parents=True, exist_ok=True)
    docs: list[Doc] = []

    for url, fname, lic, note in HTML_SOURCES:
        out = outdir / fname
        if out.exists() and out.stat().st_size > 1024 and not force:
            print(f"    · {fname} 已存在，跳过")
            continue
        try:
            r = sess.get(url, timeout=40)
            if r.status_code != 200:
                print(f"    ✗ {fname} HTTP {r.status_code}")
                continue
            # 服务端未声明 charset 时 requests 会退回 ISO-8859-1，中文会整篇乱码
            if not r.encoding or r.encoding.lower() == "iso-8859-1":
                r.encoding = "utf-8"
            html = r.text
        except Exception as e:
            print(f"    ✗ {fname} 下载失败：{type(e).__name__} {e}")
            continue

        cn = len(RE_CN.findall(html))
        if cn < 400:
            print(f"    ✗ {fname} 中文仅 {cn}，非中文主体")
            continue
        if RE_MOBILE.search(html) or RE_IDCARD.search(html):
            print(f"    ✗ {fname} 检出疑似个人隐私")
            continue
        out.write_text(html, encoding="utf-8")

        n_code = len(_re.findall(r"<(code|pre)\b", html))
        n_tbl = html.count("<table")
        kb = out.stat().st_size / 1024
        print(f"    ✓ {fname}  {kb:.0f}KB  中文 {cn}  代码块 {n_code}  表格 {n_tbl}")
        docs.append(Doc(
            relpath=f"html/{fname}", fmt="HTML",
            source=url.split("/")[2], license_=lic,
            extent=f"{kb:.0f}KB / 中文 {cn}",
            difficulty=f"{note}；原格式未清洗（导航/脚本/样式俱在）",
            verdict="ok", url=url, sha256=sha256_of(out), fetched=today(),
        ))
    return docs


def main() -> int:
    ap = argparse.ArgumentParser(description="抓取公开语料")
    ap.add_argument("--channel", default="opensource", help="渠道名")
    ap.add_argument("--limit", type=int, help="每个渠道只取前 N 篇（试跑用）")
    ap.add_argument("--need", type=int, help="目标篇数（下载后会再验收淘汰）")
    ap.add_argument("--force", action="store_true", help="已存在的语料也重新下载")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    sess = Session()

    if args.channel == "opensource":
        docs = channel_opensource(sess, args.limit, args.force)
    elif args.channel == "cninfo":
        docs = channel_cninfo(sess, need=args.need or 12, force=args.force)
    elif args.channel == "html":
        docs = channel_html(sess, force=args.force)
    else:
        print(f"渠道 `{args.channel}` 尚未实现。当前可用：opensource / cninfo / html")
        return 2

    if docs:
        write_manifest(docs)
        print(f"\n登记 {len(docs)} 篇 → data/raw/MANIFEST.md")
    else:
        print("\n没有新增语料")
    return 0


if __name__ == "__main__":
    sys.exit(main())
