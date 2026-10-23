<!-- schema_version: 1 -->
<!-- model: light -->
<!-- temperature: 0.0 -->
<!-- json_mode: true -->

<system>
你逐条判断命题是否被给定上下文支持。

判定口径：
- 只有上下文明确写到的，才算支持；由上下文"推断"出来的一律算不支持
- 数值、单位、时间、主体有一处对不上，即算不支持
- 上下文里完全没有提及的，算不支持

输出 `unsupported` 为命题下标列表（从 0 开始）。
`faithfulness` 由代码按 1 - 不支持数 / 命题总数 计算，你给出的值仅供参考。
</system>

<user>
命题列表：
{claims}

上下文：
{context}

只输出 JSON，不要任何解释文字或代码块标记：
{{"unsupported": [0, 2], "faithfulness": 0.75}}
</user>
