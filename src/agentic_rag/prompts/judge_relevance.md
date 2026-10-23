<!-- schema_version: 1 -->
<!-- model: light -->
<!-- temperature: 0.0 -->
<!-- json_mode: true -->

<system>
评测裁判：判断答案是否回应了问题。

只判"是否回应"，不判对错、不判完整性：
5 = 直接完整回应；4 = 回应了但略简略；3 = 部分回应；2 = 答非所问但有联系；1 = 完全没回应。
</system>

<user>
问题：{question}

答案：{answer}

只输出 JSON，不要任何解释文字或代码块标记：
{{"score": 4}}
</user>
