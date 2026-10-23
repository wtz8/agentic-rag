"""测试替身与构造辅助（单测与集成测试共用）。"""

from __future__ import annotations

import json


class ScriptedLLM:
    """按提示词里的**结构标记**返回固定 JSON，让每条分支可被确定性地走到。

    识别依据是模板骨架标记（"第几轮改写" / "子查询条数上限" / "答案正文"），
    不是中文措辞——措辞会频繁微调，骨架不会。
    """

    def __init__(self, intent: str = "simple", claims=None, unsupported=None, answer: str | None = None):
        self.intent = intent
        self.claims = claims if claims is not None else ["事实一"]
        self.unsupported = unsupported if unsupported is not None else []
        self.answer = answer or "华东区单价为 3.2 万元/Gbps[1]。"
        self.seen: list[str] = []

    @property
    def model_name(self) -> str:
        return "scripted"

    def chat(self, messages, temperature=0.0, max_tokens=2048, json_mode=False):
        user = messages[-1]["content"]
        self.seen.append(user[:20])
        if "待分类问题" in user:
            return json.dumps({"intent": self.intent, "confidence": 0.9, "reason": "scripted"})
        if "子查询条数上限" in user:
            q = _orig(user)
            return json.dumps({"queries": [f"{q} 指标一", f"{q} 指标二"], "strategy": "decompose"})
        if "第几轮改写" in user:
            return json.dumps({"queries": [_orig(user)], "strategy": "rewrite"})
        if "原始问题" in user:  # hyde：无"第几轮改写"标记
            q = _orig(user)
            # HyDE 段落必须在 100–800 字符之间，否则会被判为无效产物并退回 rewrite
            para = (
                f"根据文档记载，{q}的相关口径如下：该指标按年度统计，单位为万元，"
                "数值以最终审计口径为准，分部数据单独列示，历史年度不做追溯调整。" * 2
            )
            return json.dumps({"queries": [para], "strategy": "hyde"})
        if "答案正文" in user:
            return json.dumps({"claims": self.claims})
        if "命题列表" in user:
            return json.dumps({"unsupported": self.unsupported, "faithfulness": 1.0})
        return self.answer

    def stream(self, messages, temperature=0.0, max_tokens=2048):
        for ch in self.answer:
            yield ch


def _orig(user: str) -> str:
    for line in user.splitlines():
        s = line.strip()
        if s.startswith("原始问题：") or s.startswith("待分类问题："):
            return s.split("：", 1)[1].strip()
    return user.strip()[:40]


__all__ = ["ScriptedLLM"]
