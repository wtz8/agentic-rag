"""EvalReport 与报告落盘（04 §9.7 / §9.6）。

四条存证字段（`provider_mode` / `retrieval_signature` / `models_actual` /
`user_dict_sha256`）合并起来才能回答"这份数字是在什么条件下跑出来的"。
没有它们，报告只是数字；有了它们，报告才是可复核的实验记录。

对文档契约的三处扩展（都是消融结论成立的前提，不是锦上添花）：
- `by_kind`：unanswerable 的 abstain_rate 必须与其他 kind 分开统计，混在一起会
  把"拒答"平均进 Recall，得出"拒答让检索变差"这种荒谬结论。
- `comparisons`：I5 定案"每条 A 比 B 好，必须同时给出 net_win_rate 与 N"，
  这个义务落在数据结构上而不是报告写作上，否则漏写无法被检查。
- `calibration`：K1 的 `sufficiency_vector_floor` 标定方法与结果必须进报告
  （06 §7.2：不标定则检索侧反思能力整体失效，而失效是不可见的）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import Config, user_dict_sha256
from ..prompts.loader import all_prompts
from .metrics import PairedComparison

__all__ = ["VariantResult", "EvalReport", "write_report", "p50", "p95"]


def p50(vals: list[float]) -> float:
    return _quantile(vals, 0.50)


def p95(vals: list[float]) -> float:
    return _quantile(vals, 0.95)


def _quantile(vals: list[float], q: float) -> float:
    """线性插值分位数。空列表返回 0.0（报告列必须可渲染，不能是 NaN）。"""
    if not vals:
        return 0.0
    s = sorted(float(v) for v in vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return round(s[lo] * (1 - frac) + s[hi] * frac, 3)


@dataclass
class VariantResult:
    variant: str  # A0 / A1 / ... / B5
    retrieval_signature: str
    metrics: dict[str, float] = field(default_factory=dict)  # recall@5 / recall@20 / mrr / ndcg@5 / hit_rate
    generation: dict[str, float] = field(default_factory=dict)  # faithfulness / relevance / citation_accuracy / abstain_rate
    latency: dict[str, float] = field(default_factory=dict)  # p50 / p95（total 与 generate）
    tokens: dict[str, int] = field(default_factory=dict)
    by_kind: dict[str, dict[str, float]] = field(default_factory=dict)  # 扩展：分 kind 的指标
    errors: int = 0  # 执行失败（RunResult.error 非空）的用例数

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalReport:
    created_at: str
    provider_mode: str  # mock | dashscope        —— 必带
    retrieval_signature: str  # 整体指纹          —— 必带
    models_actual: dict[str, str]  # 响应体里的实际模型名 —— 必带
    user_dict_sha256: str  # 分词词典指纹          —— 必带
    prompt_schema_versions: dict[str, int]  # 05 §0.2
    index_version: str
    variants: list[VariantResult] = field(default_factory=list)
    unresolved_anchors: list[tuple[str, object]] = field(default_factory=list)  # (question, 锚点)，H14
    case_count: int = 0
    notes: str = ""

    # ── 扩展字段 ──
    comparisons: list[PairedComparison] = field(default_factory=list)  # I5：逐条配对比分
    calibration: dict | None = None  # K1：sufficiency_vector_floor 标定记录
    mock_disclosure: str = ""  # §9.5：Mock 下的声明，必须写进报告顶部

    # ── I5 第二级：仪器精度（04 §9.4"先测仪器精度，再测样品"）──
    # 同配置重复 N 次，用 ±2σ 给出组间差值的可分辨下限。
    # `repeats=1` 时此项为空——**没有重复跑就不得声称做过显著性检验**。
    repeats: int = 1
    repeat_stats: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["unresolved_anchors"] = [
            {"question": q, "anchor": _anchor_dict(a)} for q, a in self.unresolved_anchors
        ]
        return d

    def summary_text(self) -> str:
        return render_summary(self)


def _anchor_dict(a) -> dict:
    return getattr(a, "__dict__", None) or {"repr": repr(a)}


# ─────────────────────────── 落盘 ───────────────────────────


def write_report(report: EvalReport, out_dir: str | Path) -> tuple[Path, Path]:
    """`eval/results/{timestamp}.json` + `summary.md`（04 §9.6）。返回两个路径。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / f"{report.created_at.replace(':', '').replace('-', '')}.json"
    raw.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = out / "summary.md"
    summary.write_text(render_summary(report), encoding="utf-8")
    return raw, summary


def render_summary(r: EvalReport) -> str:
    lines: list[str] = []
    lines.append("# 评测报告")
    lines.append("")
    lines.append(f"- 生成时间：{r.created_at}")
    lines.append(f"- provider_mode：**{r.provider_mode}**")
    lines.append(f"- retrieval_signature：`{r.retrieval_signature}`")
    lines.append(f"- index_version：`{r.index_version}`")
    lines.append(f"- user_dict_sha256：`{r.user_dict_sha256}`")
    lines.append(f"- models_actual：{json.dumps(r.models_actual, ensure_ascii=False)}")
    lines.append(
        f"- prompt_schema_versions：{json.dumps(r.prompt_schema_versions, ensure_ascii=False)}"
    )
    lines.append(f"- 用例数：{r.case_count}")
    if r.unresolved_anchors:
        lines.append(f"- ⚠ 未解析锚点：{len(r.unresolved_anchors)} 条（Recall 分母不含它们）")
    # unanswerable 机器校验的对比提示（04 §6.3 分工表）
    for v in r.variants:
        top = v.generation.get("unanswerable_top_rerank")
        if top is not None:
            lines.append(
                f"- unanswerable 精排分均值（{v.variant}）：{top:.4f}——与 "
                "`unanswerable_rerank_floor`（默认 0.35）对比，持续高于阈值说明 floor 过松或精排失效"
            )
    lines.append("")

    if r.mock_disclosure:
        lines.append("> ⚠ " + r.mock_disclosure)
        lines.append("")

    lines.append("## 变体指标")
    lines.append("")
    header = (
        "| 变体 | signature | recall@5 | recall@20 | mrr | ndcg@5 | hit_rate "
        "| faithfulness | relevance | citation | abstain | p50(ms) | p95(ms) | tokens | 失败 |"
    )
    lines.append(header)
    lines.append("|" + " --- |" * 15)
    for v in r.variants:
        m, g, lat, tok = v.metrics, v.generation, v.latency, v.tokens
        lines.append(
            f"| {v.variant} | {v.retrieval_signature} "
            f"| {_f(m.get('recall@5'))} | {_f(m.get('recall@20'))} | {_f(m.get('mrr'))} "
            f"| {_f(m.get('ndcg@5'))} | {_f(m.get('hit_rate'))} "
            f"| {_f(g.get('faithfulness'))} | {_f(g.get('relevance'))} "
            f"| {_f(g.get('citation_accuracy'))} | {_f(g.get('abstain_rate'))} "
            f"| {_f(lat.get('p50_total'), 0)} | {_f(lat.get('p95_total'), 0)} "
            f"| {tok.get('tokens_total', 0)} | {v.errors} |"
        )
    lines.append("")

    if r.comparisons:
        lines.append("## 配对比分（I5 第一级判据）")
        lines.append("")
        for c in r.comparisons:
            lines.append(f"- {c.as_text()}")
        lines.append("")

    if r.calibration:
        lines.append("## sufficiency_vector_floor 标定（K1）")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(r.calibration, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    if r.unresolved_anchors:
        lines.append("## 未解析锚点（必须显式列出，不得静默跳过）")
        lines.append("")
        for q, a in r.unresolved_anchors:
            lines.append(f"- {q} → {json.dumps(_anchor_dict(a), ensure_ascii=False)}")
        lines.append("")

    if r.notes:
        lines.append("## 备注")
        lines.append("")
        lines.append(r.notes)
    return "\n".join(lines) + "\n"


def _f(v, nd=3):
    if v is None:
        return "—"
    return f"{float(v):.{nd}f}"


def make_report_skeleton(cfg: Config) -> EvalReport:
    """按当前配置构造带齐存证字段的报告骨架。

    存证字段的采集规则：
    - `models_actual` 取响应体返回的模型名（`provider.model_name`），
      而非配置树的请求值——两者可能不同，且不同之处恰恰是关键信息（08 §1.1）。
    - `prompt_schema_versions` 取自模板 front-matter（05 §0.2）。
    """
    from datetime import datetime

    return EvalReport(
        created_at=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        provider_mode=cfg.provider_mode,
        retrieval_signature=cfg.retrieval_signature,
        models_actual={},
        user_dict_sha256=user_dict_sha256(cfg.index.user_dict_path),
        prompt_schema_versions={p.name: p.schema_version for p in all_prompts()},
        index_version=cfg.index_version,
    )
