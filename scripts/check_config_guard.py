"""CI 配置护栏（.github/workflows/ci.yml 的一个独立步骤）。

为什么单独一个脚本而不是写进 YAML：pyproject 里的 `run: python -c "..."` 多行写法
在 YAML 里只有在**不加缩进块**时才勉强成立，一旦出现 `try:` 这类以冒号结尾的行，
整个工作流文件就是非法 YAML——GitHub 会直接拒绝运行，而本地 `pytest` 全绿，
**没有任何地方会告诉你 CI 从来没跑过**（2026-10-23 实际踩到）。

这里做两件事：
1. 配置树的启动校验是硬性的，单独跑一次，让它在配置被改坏时以**独立步骤**失败，
   而不是埋在某个用例里；
2. 打印实际生效的 provider_mode / profile / index_version，方便对照报告。
"""

from __future__ import annotations

import sys

from agentic_rag.config import ConfigError, get_config, validate_config


def main() -> int:
    cfg = get_config()
    print("provider_mode =", cfg.provider_mode)
    print("profile =", cfg.profile, "| index_version =", cfg.index_version)

    # 故意破坏一条不变量：两路召回都关掉，候选恒空，属于误配置而不是实验。
    cfg.retrieval.use_vector = False
    cfg.retrieval.use_bm25 = False
    try:
        validate_config(cfg)
    except ConfigError as exc:
        print("expected:", str(exc).splitlines()[0])
    else:
        print("配置校验失效：不变量被破坏却没有报错", file=sys.stderr)
        return 1

    print("config guard ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
