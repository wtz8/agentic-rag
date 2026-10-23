"""文档契约对账（N 组催生）：把"文档与实现必须同源"从人工纪律变成 CI 里的断言。

四类对账，每一类都对应一次真实事故：

1. **配置树 ↔ `config.py`**：N7 漏了两个真实字段（`ingestion.pdf_backend` /
   `retrieval.parent_backfill`）。已有的 `test_config.py` 只查"正文里的 `cfg.X` 引用
   能不能取到属性"，方向是文档→代码；这里补反方向：配置树代码块里写的每个字段，
   代码里必须真有；代码里有的，配置树也必须写到。
2. **提示词 ↔ `docs/07`**：新增模板却忘了写契约，运行时只会表现为"这个模板没有失败回落说明"。
3. **依赖 ↔ `pyproject.toml`**：N6 的 `pydantic` / `pymupdf` 靠传递依赖恰好装上，
   换版本或裁依赖就断——这类缺失在本机永远看不见。
4. **Markdown 链接 ↔ 文件系统**：删文档、改标题最容易留下死链。

这些用例的价值不在"跑绿"，而在**文档一改就会红**：红了就说明有一边忘了同步。
"""

from __future__ import annotations

import ast
import dataclasses
import re
import tomllib
from pathlib import Path

import pytest

from agentic_rag import config as cfgmod

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

# 顶层模块名 → pyproject 里声明的发行名（不一致的才需要列）
_IMPORT_ALIAS = {
    "bs4": "beautifulsoup4",
    "docx": "python-docx",
    "fitz": "pymupdf",
    "pptx": "python-pptx",
}
# 可选依赖：不在 dependencies 里，但必须在某个 extra 里
_OPTIONAL_IMPORTS = {"docling"}


def _normalize(dist: str) -> str:
    return re.split(r"[<>=!~\[ ;]", dist.strip())[0].lower().replace("_", "-")


# ── 1. 配置树 ↔ config.py ─────────────────────────────────────────────


def _python_blocks(md: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", md.read_text(encoding="utf-8"), re.S)


def _documented_config_fields() -> dict[str, set[str]]:
    """从 04 的配置树代码块里取出 `class Xxx:` 的字段名。

    配置树分散在多个代码块里（顶层容器 `Config` 单独一块），且夹着仅示意签名的片段，
    所以逐块解析、解析失败的跳过——只认真正是 Python 的那些块。
    """
    found: dict[str, set[str]] = {}
    for block in _python_blocks(DOCS / "03-模块设计-离线链路.md"):
        if "@dataclass" not in block:
            continue
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue  # 示意用的片段（只有签名没有函数体）
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            fields = {
                st.target.id
                for st in node.body
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)
            }
            if fields:
                found.setdefault(node.name, set()).update(fields)
    return found


def _code_config_fields() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name in dir(cfgmod):
        obj = getattr(cfgmod, name)
        if (
            dataclasses.is_dataclass(obj)
            and isinstance(obj, type)
            and obj.__module__ == cfgmod.__name__
        ):
            out[name] = {f.name for f in dataclasses.fields(obj)}
    return out


def _expected_config_classes() -> set[str]:
    """顶层容器 + 它聚合的全部子配置类（数据契约类如 Chunk 不在此列）。"""
    cfg = cfgmod.get_config(env={"DASHSCOPE_API_KEY": ""})
    names = {"Config"}
    for f in dataclasses.fields(cfgmod.Config):
        value = getattr(cfg, f.name)
        if dataclasses.is_dataclass(value):
            names.add(type(value).__name__)
    return names


def test_config_tree_matches_config_classes():
    """docs/03 §1 的配置树必须与 config.py 的 dataclass 字段完全一致（双向）。

    04 的代码块里还夹着 `Chunk` / `ParsedDocument` 这类**数据契约**，它们不是配置类，
    所以只在"配置类"的范围内比对；但文档里任何以 `Config` 结尾却对不上代码的类名都要报错。
    """
    doc = _documented_config_fields()
    code = _code_config_fields()
    expected = _expected_config_classes()

    assert code, "没有反射到任何配置类"
    typo = sorted(name for name in doc if name.endswith("Config") and name not in code)
    assert not typo, f"配置树里写了、代码里没有这些配置类（可能是笔误）: {typo}"

    problems: list[str] = []
    for name in sorted(expected):
        doc_fields = doc.get(name)
        assert doc_fields is not None, f"配置类 {name} 没有出现在 03 §1 的配置树里"
        only_code = sorted(code[name] - doc_fields)
        only_doc = sorted(doc_fields - code[name])
        if only_code:
            problems.append(f"{name} 代码有、配置树没写: {only_code}")
        if only_doc:
            problems.append(f"{name} 配置树写了、代码没有: {only_doc}")
    assert not problems, "配置树与代码不一致：\n  " + "\n  ".join(problems)


# ── 2. 提示词 ↔ docs/07 ───────────────────────────────────────────────


def test_every_prompt_template_is_documented():
    """每个 prompts/*.md 都必须在 07 里有契约（名称出现即可，表格式与分节式都算）。"""
    from agentic_rag.prompts.loader import all_prompts

    doc = (DOCS / "05-提示词契约.md").read_text(encoding="utf-8")
    missing = [p.name for p in all_prompts() if p.name not in doc]
    assert not missing, f"07 未记录这些模板的契约: {missing}"


def test_prompt_placeholders_are_documented():
    """模板用到的每个 `{var}` 都应在 07 里出现，否则"输入变量"栏就是过期的。"""
    from agentic_rag.prompts.loader import all_prompts

    doc = (DOCS / "05-提示词契约.md").read_text(encoding="utf-8")
    unknown: list[str] = []
    for p in all_prompts():
        for section in (p.system, p.user_template):
            for m in re.finditer(r"\{([a-z_][a-z0-9_]*)\}", section):
                var = m.group(1)
                if var not in doc:
                    unknown.append(f"{p.name}:{var}")
    assert not unknown, f"这些变量没在 07 里出现: {sorted(set(unknown))}"


# ── 3. 依赖 ↔ pyproject.toml ──────────────────────────────────────────


def _src_third_party_modules() -> set[str]:
    import sys

    stdlib = set(sys.stdlib_module_names)
    mods: set[str] = set()
    for path in (ROOT / "src" / "agentic_rag").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mods.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
    return {m for m in mods if m not in stdlib and m != "agentic_rag"}


def _declared_dists() -> dict[str, set[str]]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    declared = {_normalize(s) for s in project["dependencies"]}
    optional: set[str] = set()
    for specs in project.get("optional-dependencies", {}).values():
        optional |= {_normalize(s) for s in specs}
    return {"required": declared, "optional": optional}


def test_all_imported_packages_are_declared():
    """`src/` 里 import 的第三方包必须显式声明——不能靠别的包"恰好"带进来。"""
    declared = _declared_dists()
    required = declared["required"]
    optional = declared["optional"]

    undeclared: list[str] = []
    for mod in sorted(_src_third_party_modules()):
        dist = _IMPORT_ALIAS.get(mod, mod.replace("_", "-"))
        if mod in _OPTIONAL_IMPORTS:
            assert dist in optional, f"{mod} 是可选依赖，但没有任何 extra 声明 {dist}"
            continue
        if dist not in required:
            undeclared.append(f"{mod} -> {dist}")
    assert not undeclared, f"这些导入没有在 pyproject 的 dependencies 里声明: {undeclared}"


# ── 4. Markdown 链接 ──────────────────────────────────────────────────


def _markdown_files() -> list[Path]:
    return [ROOT / "README.md", *sorted(DOCS.glob("*.md"))]


def test_markdown_links_resolve():
    """仓库内相对链接必须指向真实文件（相对链接按所在文件解析）。"""
    broken: list[str] = []
    for md in _markdown_files():
        for raw in re.findall(r"\]\(([^)\s]+)\)", md.read_text(encoding="utf-8")):
            if raw.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = raw.split("#")[0]
            if not target:
                continue
            resolved = (md.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{md.relative_to(ROOT)} -> {raw}")
    assert not broken, "Markdown 里有死链:\n  " + "\n  ".join(broken)


@pytest.mark.parametrize("doc", sorted(p.name for p in DOCS.glob("*.md")))
def test_docs_are_reachable_from_readme(doc: str):
    """每份设计文档都要能从 README 的文档索引点到——否则等于没写。"""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"docs/{doc}" in readme, f"README 的文档索引里没有 {doc}"


# ── 5. CI 工作流本身 ──────────────────────────────────────────────────


def test_ci_workflow_is_valid_yaml():
    """`ci.yml` 必须是合法 YAML。

    这条不是形式主义：它曾经整份失效——`run: python -c "` 后面跟多行代码，
    其中 `try:` 以冒号结尾，YAML 扫描器把整份文件判为非法，
    **GitHub 会直接拒绝运行该工作流**；而本地 `pytest` 全绿，没有任何东西提示
    "CI 从来没跑过"。现在把护栏逻辑挪进 `scripts/check_config_guard.py`，
    YAML 只留 `run: |` 块，并由本用例守住这点。
    """
    import yaml

    data = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    jobs = data["jobs"]
    assert jobs, "工作流里没有任何 job"
    steps = [s.get("name") for s in jobs["test"]["steps"] if s.get("name")]
    for required in ("Install", "Test", "Config guard", "Package build"):
        assert required in steps, f"CI 缺少步骤 {required}；现有步骤: {steps}"


def test_ci_runs_without_any_secret():
    """CI 不配 secret：测试步骤必须显式传空 key，否则 Mock 模式不会被走到。"""
    import yaml

    data = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    test_step = next(s for s in data["jobs"]["test"]["steps"] if s.get("name") == "Test")
    assert test_step.get("env", {}).get("DASHSCOPE_API_KEY") == "", (
        "Test 步骤没有显式传空 DASHSCOPE_API_KEY——CI 里若有 secret，测试就不再是无 Key 验证"
    )
