"""提示词模板加载器（05 §0）。

模板是独立资产：一个模板一个 `.md` 文件，文件名即模板名。
头部 front-matter 用 HTML 注释写 `schema_version` / `model` / `temperature` / `json_mode`，
正文用 `<system>` / `<user>` 两段包裹。

**两段都会渲染**（M30）：`{var}` 在 `<system>` 与 `<user>` 里都生效，由调用方一并注入。
早先只渲染 `<user>`，于是 `answer` 的"最多引用 {max_citations} 个来源"、
`decompose` 的"最多 {max_sub_queries} 条"原样带着花括号发给了模型——
渲染只在用户段发生这件事本身不报错，模型照常回答，所以只能靠这条契约兜住。
字面花括号（如 JSON 示例）必须写成 `{{...}}`，否则加载即报错（`_check_placeholders`）。

设计约束：
- 输出 schema 变更即破坏性变更 → 递增 `schema_version`；
- 结构化输出一律 JSON，`json_mode=True` 走 `response_format={"type": "json_object"}`；
- 模板正文不得写死业务常量（阈值 / 模型名 / top_k），一律由输入变量注入；
- 模板里必须保留 05 §3 的英文 schema 关键词，MockLLM 靠它匹配返回结构。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent

_FRONT = re.compile(r"<!--\s*(schema_version|model|temperature|json_mode)\s*:\s*([^\s>]+)\s*-->")
_SECTION = re.compile(r"<(system|user)>(.*?)</\1>", re.S)
# 转义花括号 {{ }} 先匹配掉，剩下的 {xxx} 必须是合法占位符
_BRACE = re.compile(r"\{\{|\}\}|\{([^{}]*)\}")
_PLACEHOLDER = re.compile(r"[a-z_][a-z0-9_]*(\.[a-z0-9_]+)*")


def _check_placeholders(name: str, section: str, where: str) -> None:
    """模板正文会被 `str.format` 渲染：未转义的字面花括号必须在加载期就暴露。"""
    for m in _BRACE.finditer(section):
        token = m.group(1)
        if token is None:  # {{ 或 }}，合法转义
            continue
        if not _PLACEHOLDER.fullmatch(token):
            raise ValueError(
                f"模板 {name} 的 <{where}> 段有未转义的字面花括号 {m.group(0)!r}；"
                f"模板会被 str.format 渲染，字面量必须写成双花括号（JSON 示例写成 "
                f'{{{{"key": "value"}}}}）'
            )


@dataclass(frozen=True)
class Prompt:
    name: str
    schema_version: int
    model: str  # light | main
    temperature: float
    json_mode: bool
    system: str  # 原始 system 模板（可能含 {var}），渲染见 system_text()
    user_template: str

    def render(self, **kwargs) -> str:
        """渲染 `<user>` 段。"""
        return self.user_template.format(**kwargs)

    def system_text(self, **kwargs) -> str:
        """渲染 `<system>` 段（M30：两段共用同一批注入变量）。"""
        return self.system.format(**kwargs)

    def messages(self, **kwargs) -> list[dict]:
        return [
            {"role": "system", "content": self.system_text(**kwargs)},
            {"role": "user", "content": self.render(**kwargs)},
        ]


def _parse(path: Path) -> Prompt:
    raw = path.read_text(encoding="utf-8")
    meta = {k: v for k, v in _FRONT.findall(raw)}
    sections = {k: v.strip() for k, v in _SECTION.findall(raw)}
    if "user" not in sections:
        raise ValueError(f"模板 {path.name} 缺少 <user> 段")
    name = path.stem
    system = sections.get("system", "")
    _check_placeholders(name, system, "system")
    _check_placeholders(name, sections["user"], "user")
    return Prompt(
        name=name,
        schema_version=int(meta.get("schema_version", 1)),
        model=meta.get("model", "light"),
        temperature=float(meta.get("temperature", 0.0)),
        json_mode=meta.get("json_mode", "false").lower() in ("true", "1", "yes"),
        system=system,
        user_template=sections["user"],
    )


_CACHE: dict[str, Prompt] | None = None


def _load_all() -> dict[str, Prompt]:
    out: dict[str, Prompt] = {}
    for path in sorted(_PROMPTS_DIR.glob("*.md")):
        p = _parse(path)
        out[p.name] = p
    return out


def get_prompt(name: str) -> Prompt:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_all()
    if name not in _CACHE:
        raise KeyError(f"未定义的提示词模板: {name}")
    return _CACHE[name]


def all_prompts() -> list[Prompt]:
    """按模板名排序返回全部模板（顺序稳定，`generation_signature` 依赖它）。"""
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_all()
    return [_CACHE[k] for k in sorted(_CACHE)]


def prompt_schema_versions() -> dict[str, int]:
    return {p.name: p.schema_version for p in all_prompts()}


def reload_prompts() -> None:
    """测试用：强制重新加载（磁盘上的模板被改过之后）。"""
    global _CACHE
    _CACHE = None
