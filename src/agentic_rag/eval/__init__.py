"""评测层（04 §9 / 03 §10 / 06 §5）。

三件事分得很清楚，共用代码但不可互相替代：

- **探针**（`probe.py`）：答案位置已知，只判"期望来源是否进 Top-K"。
  每次重建索引后跑，保的是**没坏**。
- **评测集**（`runner.py`）：判检索质量与生成质量，改配置后跑，保的是**变好了**。
- **消融**（`ablation.py`）：改配置对比变体，回答"哪个组件真的有用"。

一条硬规矩（04 §9.5）：`EvalReport` 必须携带 `provider_mode` 与 `retrieval_signature`。
Mock 模式下三路检索高度同构，检索质量的绝对值与增益**不可引用**——
没有这两项的报告不得对外引用。
"""

from __future__ import annotations

__all__ = [
    "metrics",
    "judge",
    "runner",
    "report",
    "probe",
    "ablation",
    "calibrate",
]
