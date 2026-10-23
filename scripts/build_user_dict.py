#!/usr/bin/env python3
"""构建 jieba 用户词典（data/user_dict.txt）。

为什么需要这个脚本
-------------------
01 §6 把"注入领域术语"列为自研 BM25 的三条工程理由之一，03 §7.2 把
``user_dict_path`` 写进配置，07 §6 要求"用户词典固定并入库"——但此前
**没有任何一处说过这个词典怎么产生，仓库里也没有它**。
自研 BM25 有一半的说服力押在"中文分词可控"上，而可控的前提恰恰是这份词典。

做法：从已登记语料里抽 TF-IDF Top-N，减去停用词与通用词，输出 jieba
userdict 格式（``词 频 词性``）。

产物为什么要进仓库
-------------------
07 §6 把它列为可复现性约定：词典变化会改变分词与 BM25 召回集。
它只有几 KB，进仓库才能让"同一份语料 + 同一份词典 = 同一批召回"成立。
语料文件本身不进仓库（体积 + 再分发许可，见 06 §6.2），词典则是纯术语表、
不含受许可约束的正文内容，所以可以进。

⚠ 必须人工过一遍再入库（06 §6.6 约束 1）
---------------------------------------
TF-IDF 会抽出"参见""合计""本页"这类噪声词，塞进用户词典反而会扭曲分词。
脚本 ``--report`` 输出候选清单供人工勾选，约 15 分钟，不能省。
过筛结果落到 ``data/user_dict.reviewed.txt``（可选，见 --reviewed）。

用法：
    python scripts/build_user_dict.py                    # 输出候选清单
    python scripts/build_user_dict.py --reviewed         # 用已过筛清单写产物
    python scripts/build_user_dict.py --topk 300 --out data/user_dict.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest_detail.json"
DEFAULT_OUT = ROOT / "data" / "user_dict.txt"
# 注意：过筛清单放在 data/ 而不是 data/raw/ —— 后者在 .gitignore 里被整体排除，
# 放在那里会导致"产物进仓库"（06 §6.6）这条约定落空。
REVIEWED = ROOT / "data" / "user_dict.reviewed.txt"
CANDIDATES = ROOT / "data" / "user_dict.candidates.txt"

# 允许参与抽取的词性（jieba 词性标注集）
# n / ns / nt / nz：名词、地名、机构名、其他专名
# eng：英文术语（Transformer、LoRA 等）
# l / i / j：习语、成语、简称
DEFAULT_ALLOW_POS = ("n", "ns", "nt", "nz", "eng", "l", "i", "j")

# 停用词与"通用词"：这些词 TF-IDF 分数可能不低，但注入词典只会帮倒忙。
#
# 判据只有一条：**jieba 默认词典已经能切对的词，不进用户词典。**
# 用户词典的价值在于"通用分词器会切错的那部分"——产品名、部门名、缩写、
# 指标名、编号。把"模型""函数""参数"这种通用词塞进来，除了让复合词被
# 过度合并之外没有任何收益。
STOPWORDS = {
    # ① 版式 / 导航噪声：来自 HTML 与 PDF 页眉页脚，不是术语
    "参见", "合计", "本页", "上一页", "下一页", "目录", "附录", "注释", "说明",
    "如下", "以上", "以下", "其中", "因此", "但是", "如果", "可以", "需要",
    "进行", "通过", "由于", "对于", "关于", "以及", "并且", "或者",
    "这里", "那里", "其他", "另外", "例如", "比如", "这样", "那样",
    # ② 指代表述：高频但不承载领域语义
    "我们", "你们", "他们", "这个", "那个", "这些", "那些", "什么", "怎么",
    "如何", "为什么", "是否", "没有", "已经", "还是", "一个", "一种",
    # ③ 通用技术词：默认词典已覆盖，注入无收益
    "使用", "实现", "支持", "提供", "包含", "表示", "定义", "描述", "配置",
    "安装", "运行", "创建", "获取", "设置", "返回", "生成", "处理",
    "模型", "函数", "参数", "文档", "文本", "数据", "内容", "方法",
    "问题", "结果", "示例", "代码", "文件", "系统", "用户", "时间",
    "类型", "版本", "信息", "项目", "目录", "环境", "步骤", "方式",
    "情况", "部分", "注意", "建议", "相关", "不同", "同时", "之后",
}

# 单个字符不入词典（除非是明确术语，此处一律不要）
MIN_LEN = 2

# 抽取后按 TF-IDF 排序，只保留前 N 个
DEFAULT_TOPK = 300


# ---------------------------------------------------------------- 文本抽取

def _text_from_plain(path: Path) -> str:
    """Markdown / 纯文本：直接读。"""
    return path.read_text(encoding="utf-8", errors="ignore")


def _text_from_html(path: Path) -> str:
    """HTML：去掉 script / style / nav / footer 后取正文。"""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _text_from_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.append(" ".join(c.text for c in row.cells))
    return "\n".join(parts)


def _text_from_pptx(path: Path) -> str:
    from pptx import Presentation

    prs = Presentation(str(path))
    parts: list[str] = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    parts.append(" ".join(c.text for c in row.cells))
    return "\n".join(parts)


def _text_from_xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    parts: list[str] = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            parts.append(" ".join(str(v) for v in row if v is not None))
    wb.close()
    return "\n".join(parts)


def _text_from_pdf(path: Path) -> str:
    """PDF：只取文本层。扫描件无文本层，返回空串是正确的（不是错误）。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


_EXTRACTORS = {
    ".md": _text_from_plain,
    ".txt": _text_from_plain,
    ".html": _text_from_html,
    ".htm": _text_from_html,
    ".docx": _text_from_docx,
    ".pptx": _text_from_pptx,
    ".xlsx": _text_from_xlsx,
    ".pdf": _text_from_pdf,
}


def extract_text(entry: dict) -> tuple[str, str | None]:
    """返回 (正文, 失败原因)。失败不抛异常——单篇抽不出来不该中断整批。"""
    rel = entry.get("relpath", "")
    path = RAW / rel
    if not path.exists():
        return "", "missing"
    fn = _EXTRACTORS.get(path.suffix.lower())
    if fn is None:
        return "", f"unsupported:{path.suffix}"
    try:
        return fn(path), None
    except Exception as exc:  # noqa: BLE001 - 构建脚本要的是"继续跑完"而不是"第一炮就崩"
        return "", f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- 候选抽取

def load_whitelist() -> list[dict]:
    """按 MANIFEST 白名单取语料；verdict != ok 的一律不参与。

    理由：被判 rejected 的文件本就不该影响索引口径，让它们参与词典抽取
    等于让"没入库的文档"反过来决定"入库文档的切词方式"。
    """
    if not MANIFEST.exists():
        raise SystemExit(
            f"[build_user_dict] 缺少 {MANIFEST}；请先跑 scripts/fetch_corpus.py 与 scripts/corpus_manifest.py"
        )
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return [e for e in data if e.get("verdict") == "ok"]


def clean(text: str) -> str:
    """丢掉代码块、行内代码、URL 与表格分隔线——它们会产生大量伪术语。"""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`\n]+`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", " ", text, flags=re.M)
    return text


def _is_code_or_latex(word: str) -> bool:
    """过滤纯 ASCII 小写串里的代码标识符与 LaTeX 命令。

    保留两类纯 ASCII 词：
      - 全大写缩写/编号：LLM / RAG / BM25 / API / GPU / SQL / PDF / HTML / TF-IDF
      - 驼峰或含大写的专名：Qwen / LoRA / ChatGLM / LangChain / HuggingFace
    其余纯 ASCII 词（`import` / `pip` / `torch` / `boldsymbol` / `frac` …）
    一律视为代码或排版噪声——它们来自代码块而非领域术语。
    """
    if re.search(r"[\u4e00-\u9fff]", word):
        return False                                  # 含汉字，走 STOPWORDS 判断
    if re.fullmatch(r"[A-Z0-9][A-Z0-9\-]{1,}", word):
        return False                                  # 全大写缩写 / 编号
    return not re.search(r"[A-Z]", word)              # 无大写 → 代码/LaTeX 噪声


def candidates(corpus: str, topk: int, allow_pos: tuple[str, ...]) -> list[tuple[str, float]]:
    """TF-IDF 抽取 + 两级过滤。返回 [(词, 权重)] 按权重降序。"""
    import jieba.analyse

    tags = jieba.analyse.extract_tags(
        clean(corpus), topK=topk * 3, withWeight=True, allowPOS=allow_pos
    )

    out: list[tuple[str, float]] = []
    for word, weight in tags:
        word = word.strip()
        if len(word) < MIN_LEN:
            continue
        if word in STOPWORDS:
            continue
        if word.isdigit():
            continue
        if not re.search(r"[\u4e00-\u9fff]|[a-zA-Z]{2,}", word):
            continue
        if _is_code_or_latex(word):
            continue
        out.append((word, weight))
        if len(out) >= topk:
            break
    return out


# ---------------------------------------------------------------- 产物写出

def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_dict(path: Path, words: list[tuple[str, float]], source: str) -> None:
    """写成 jieba userdict 格式：``词 频 词性``。

    频次取 TF-IDF 权重 ×1000 后向上取到 10 的倍数，落在 jieba 建议的
    "足够大以覆盖默认词典"区间；词性统一标 ``nz``（其他专名）。
    """
    lines = [
        "# jieba 用户词典 —— 由 scripts/build_user_dict.py 生成，请勿手改",
        f"# 来源: {source}",
        f"# 词数: {len(words)}",
        "# 格式: 词 词频 词性",
        "#",
        "# ⚠ 本文件由 TF-IDF 自动抽取后经人工过筛生成。",
        "#   重新生成前请先跑一次 --reviewed 流程（见 06 §6.6 约束 1）。",
        "",
    ]
    for word, weight in words:
        freq = max(10, int(round(weight * 1000 / 10.0)) * 10)
        lines.append(f"{word} {freq} nz")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="构建 jieba 用户词典")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK, help=f"保留词数（默认 {DEFAULT_TOPK}）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出路径")
    ap.add_argument(
        "--reviewed",
        action="store_true",
        help=f"从 {REVIEWED.name} 读取人工过筛后的词表写产物（无该文件则回退到自动抽取）",
    )
    ap.add_argument("--report", action="store_true", help="只输出候选清单，不写产物（供人工过筛）")
    args = ap.parse_args()

    entries = load_whitelist()
    print(f"[build_user_dict] 白名单语料 {len(entries)} 篇")

    skipped: Counter[str] = Counter()
    chunks: list[str] = []
    for entry in entries:
        text, err = extract_text(entry)
        if err:
            skipped[err] += 1
            continue
        if text.strip():
            chunks.append(text)
    if not chunks:
        raise SystemExit("[build_user_dict] 没有抽到任何正文，检查 data/raw/ 是否已构建")
    corpus = "\n".join(chunks)
    print(f"[build_user_dict] 成功抽取 {len(chunks)} 篇 / 正文 {len(corpus)} 字符")
    for reason, n in skipped.most_common():
        print(f"[build_user_dict] 跳过 {n} 篇（{reason}）")

    auto = candidates(corpus, args.topk, DEFAULT_ALLOW_POS)

    if args.reviewed and REVIEWED.exists():
        # 过筛清单决定"留哪些"，权重仍取语料实测值（不在候选里的补录词取 1.0）
        weights = dict(auto)
        words = []
        for line in REVIEWED.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.append((line, weights.get(line, 1.0)))
        source = f"人工过筛 · {REVIEWED.name}（权重取自语料 TF-IDF）"
        print(f"[build_user_dict] 使用过筛词表 {len(words)} 条（自动候选 {len(auto)} 条）")
    else:
        if args.reviewed:
            print(f"[build_user_dict] 未找到 {REVIEWED.name}，回退到自动抽取")
        words, source = auto, "TF-IDF 自动抽取（scripts/build_user_dict.py）"

    if args.report:
        CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
        CANDIDATES.write_text(
            "\n".join(word for word, _ in words) + "\n", encoding="utf-8"
        )
        print(
            f"\n候选清单已写出 {CANDIDATES}\n"
            f"人工过筛流程：删掉噪声词 → 另存为 {REVIEWED.name} → 重跑 --reviewed"
        )
        return 0

    write_dict(Path(args.out), words, source)
    print(f"[build_user_dict] 已写出 {args.out}（{len(words)} 词, sha256={sha256_of(Path(args.out))}）")
    print("[build_user_dict] 提示：词典变更会改变 BM25 召回，索引需重建，且 EvalReport 会记录新指纹")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
