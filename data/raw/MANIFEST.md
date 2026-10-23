# 语料清单 · MANIFEST

> 本文件是语料的**指纹清单**，随版本库发布；语料文件本身不进仓库
> （体积 + 再分发许可，见 `docs/08-语料方案.md` §6.2）。
> 复现：`python scripts/fetch_corpus.py --channel all`，再按 manifest_detail.json 校验 sha256。

| 文件 | 格式 | 来源 | 许可/公开性质 | 页数/大小 | 预期难点 | 预期 verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `docx/docx_happy-llm_01.docx` | DOCX | 由开源文档生成 · datawhalechina/happy-llm | CC BY-NC-SA 4.0 | 2 个表格 / 48KB | Heading 样式层级、多级列表、内嵌表格 | ok |
| `docx/docx_happy-llm_02.docx` | DOCX | 由开源文档生成 · datawhalechina/happy-llm | CC BY-NC-SA 4.0 | 2 个表格 / 58KB | Heading 样式层级、多级列表、内嵌表格 | ok |
| `docx/docx_happy-llm_03.docx` | DOCX | 由开源文档生成 · datawhalechina/happy-llm | CC BY-NC-SA 4.0 | 2 个表格 / 63KB | Heading 样式层级、多级列表、内嵌表格 | ok |
| `docx/docx_happy-llm_04.docx` | DOCX | 由开源文档生成 · datawhalechina/happy-llm | CC BY-NC-SA 4.0 | 2 个表格 / 56KB | Heading 样式层级、多级列表、内嵌表格 | ok |
| `docx/docx_happy-llm_06.docx` | DOCX | 由开源文档生成 · datawhalechina/happy-llm | CC BY-NC-SA 4.0 | 2 个表格 / 66KB | Heading 样式层级、多级列表、内嵌表格 | ok |
| `docx/docx_happy-llm_07.docx` | DOCX | 由开源文档生成 · datawhalechina/happy-llm | CC BY-NC-SA 4.0 | 2 个表格 / 55KB | Heading 样式层级、多级列表、内嵌表格 | ok |
| `docx/docx_happy-llm_09.docx` | DOCX | 由开源文档生成 · datawhalechina/happy-llm | CC BY-NC-SA 4.0 | 6 个表格 / 46KB | Heading 样式层级、多级列表、内嵌表格 | ok |
| `docx/docx_langchain-chatchat_02.docx` | DOCX | 由开源文档生成 · chatchat-space/Langchain-Chatchat | Apache-2.0 | 2 个表格 / 419KB | Heading 样式层级、多级列表、内嵌表格 | ok |
| `docx/docx_pumpkin-book_02.docx` | DOCX | 由开源文档生成 · datawhalechina/pumpkin-book | CC BY-NC-SA 4.0 | 2 个表格 / 58KB | Heading 样式层级、多级列表、内嵌表格 | ok |
| `docx/docx_pumpkin-book_04.docx` | DOCX | 由开源文档生成 · datawhalechina/pumpkin-book | CC BY-NC-SA 4.0 | 2 个表格 / 48KB | Heading 样式层级、多级列表、内嵌表格 | ok |
| `docx/docx_pumpkin-book_14.docx` | DOCX | 由开源文档生成 · datawhalechina/pumpkin-book | CC BY-NC-SA 4.0 | 2 个表格 / 53KB | Heading 样式层级、多级列表、内嵌表格 | ok |
| `docx/docx_self-llm_09.docx` | DOCX | 由开源文档生成 · datawhalechina/self-llm | CC BY-NC-SA 4.0 | 3 个表格 / 54KB | Heading 样式层级、多级列表、内嵌表格 | ok |
| `html/html_javascript_info_types.html` | HTML | zh.javascript.info | CC BY-NC-SA 4.0 | 49KB / 中文 2814 | 代码块密集（>80 处），正文与代码混排；原格式未清洗（导航/脚本/样式俱在） | ok |
| `html/html_pinia_introduction.html` | HTML | pinia.vuejs.org | MIT · Pinia | 101KB / 中文 2057 | API 文档，含参数说明列表；原格式未清洗（导航/脚本/样式俱在） | ok |
| `html/html_python_tutorial.html` | HTML | docs.python.org | PSF / CC BY-NC-SA | 37KB / 中文 1940 | 教程目录页，层级列表密集；原格式未清洗（导航/脚本/样式俱在） | ok |
| `html/html_vite_guide.html` | HTML | cn.vite.dev | MIT · Vite | 87KB / 中文 1955 | 含 <table>，是六篇里表格最明确的一篇；原格式未清洗（导航/脚本/样式俱在） | ok |
| `html/html_vue_introduction.html` | HTML | cn.vuejs.org | MIT · Vue.js | 114KB / 中文 3461 | 含导航、侧栏、代码块；单页中文量最大的一档；原格式未清洗（导航/脚本/样式俱在） | ok |
| `html/html_vuex_guide.html` | HTML | vuex.vuejs.org | MIT · Vuex | 25KB / 中文 688 | 篇幅最小的一档，用于覆盖短 HTML 页这一形态；原格式未清洗（导航/脚本/样式俱在） | ok |
| `markdown/opensource_hands-on-data-analysis.md` | Markdown | GitHub · datawhalechina/hands-on-data-analysis | CC BY-NC-SA 4.0 | 30KB / 中文字 4143 | 数据分析实战，含大量表格；h2=18 h3=9 h4=16 | ok |
| `markdown/opensource_happy-llm.md` | Markdown | GitHub · datawhalechina/happy-llm | CC BY-NC-SA 4.0 | 347KB / 中文字 90573 | LLM 教程，8 章，标题层级深，含表格与代码块；h2=42 h3=49 h4=118 | ok |
| `markdown/opensource_langchain-chatchat.md` | Markdown | GitHub · chatchat-space/Langchain-Chatchat | Apache-2.0 | 772KB / 中文字 318616 | 中文 README，含架构说明与配置表格；h2=34 h3=346 h4=424 | ok |
| `markdown/opensource_pumpkin-book.md` | Markdown | GitHub · datawhalechina/pumpkin-book | CC BY-NC-SA 4.0 | 527KB / 中文字 83505 | 机器学习公式详解，章节编号密集；h2=38 h3=123 h4=353 | ok |
| `markdown/opensource_self-llm.md` | Markdown | GitHub · datawhalechina/self-llm | CC BY-NC-SA 4.0 | 2404KB / 中文字 405493 | 大模型部署实践，多章分节；h2=435 h3=1686 h4=646 | ok |
| `pdf_scanned/scanned_clean.pdf` | PDF · 扫描件 | 自制（程序合成，非物理扫描） | 内容源 CC BY-NC-SA 4.0 · datawhalechina/happy-llm | 12页 / 14640KB | 干净；每页最大可提取字符 0 | ok |
| `pdf_scanned/scanned_lowcontrast.pdf` | PDF · 扫描件 | 自制（程序合成，非物理扫描） | 内容源 CC BY-NC-SA 4.0 · datawhalechina/happy-llm | 12页 / 12773KB | 低对比度+噪点；每页最大可提取字符 0 | ok/rejected 均可（阈值边缘，验证 on_reject 可达） |
| `pdf_scanned/scanned_skew_shadow.pdf` | PDF · 扫描件 | 自制（程序合成，非物理扫描） | 内容源 CC BY-NC-SA 4.0 · datawhalechina/happy-llm | 12页 / 10446KB | 倾斜+阴影；每页最大可提取字符 0 | ok |
| `pdf_table/cninfo_300128_2021.pdf` | PDF · 表格密集 | 巨潮资讯网 · 锦富技术(300128) | 上市公司公开披露 | 223页 / 6.4MB | 跨页表、复杂表头、合并单元格 | ok |
| `pdf_table/cninfo_300152_2024.pdf` | PDF · 表格密集 | 巨潮资讯网 · ST新动力(300152) | 上市公司公开披露 | 203页 / 4.3MB | 跨页表、复杂表头、合并单元格 | ok |
| `pdf_table/cninfo_300338_2024.pdf` | PDF · 表格密集 | 巨潮资讯网 · ST开元(300338) | 上市公司公开披露 | 240页 / 5.4MB | 跨页表、复杂表头、合并单元格 | ok |
| `pdf_table/cninfo_300460_2022.pdf` | PDF · 表格密集 | 巨潮资讯网 · 惠伦晶体(300460) | 上市公司公开披露 | 218页 / 4.6MB | 跨页表、复杂表头、合并单元格 | ok |
| `pdf_table/cninfo_300500_2024.pdf` | PDF · 表格密集 | 巨潮资讯网 · 启迪设计(300500) | 上市公司公开披露 | 193页 / 1.8MB | 跨页表、复杂表头、合并单元格 | ok |
| `pdf_table/cninfo_300551_2024.pdf` | PDF · 表格密集 | 巨潮资讯网 · 古鳌科技(300551) | 上市公司公开披露 | 188页 / 4.6MB | 跨页表、复杂表头、合并单元格 | ok |
| `pdf_table/cninfo_603106_2024.pdf` | PDF · 表格密集 | 巨潮资讯网 · 恒银科技(603106) | 上市公司公开披露 | 223页 / 1.9MB | 跨页表、复杂表头、合并单元格 | ok |
| `pdf_table/cninfo_603366_2024.pdf` | PDF · 表格密集 | 巨潮资讯网 · 日出东方(603366) | 上市公司公开披露 | 242页 / 3.6MB | 跨页表、复杂表头、合并单元格 | ok |
| `pdf_table/cninfo_603388_2024.pdf` | PDF · 表格密集 | 巨潮资讯网 · *ST元成(603388) | 上市公司公开披露 | 246页 / 4.2MB | 跨页表、复杂表头、合并单元格 | ok |
| `pdf_table/cninfo_603598_2024.pdf` | PDF · 表格密集 | 巨潮资讯网 · 引力传媒(603598) | 上市公司公开披露 | 204页 / 1.8MB | 跨页表、复杂表头、合并单元格 | ok |
| `pdf_table/cninfo_688577_2024.pdf` | PDF · 表格密集 | 巨潮资讯网 · 浙海德曼(688577) | 上市公司公开披露 | 232页 / 2.0MB | 跨页表、复杂表头、合并单元格 | ok |
| `pdf_table/cninfo_920670_2024.pdf` | PDF · 表格密集 | 巨潮资讯网 · 数字人(920670) | 上市公司公开披露 | 182页 / 2.2MB | 跨页表、复杂表头、合并单元格 | ok |
| `pdf_text/text_happy-llm_01.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/happy-llm | CC BY-NC-SA 4.0 | 31页 / 296KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_happy-llm_02.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/happy-llm | CC BY-NC-SA 4.0 | 31页 / 337KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_happy-llm_03.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/happy-llm | CC BY-NC-SA 4.0 | 54页 / 271KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_happy-llm_04.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/happy-llm | CC BY-NC-SA 4.0 | 51页 / 343KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_happy-llm_05.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/happy-llm | CC BY-NC-SA 4.0 | 82页 / 360KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_langchain-chatchat_01.pdf` | PDF · 文本版 | 由开源文档生成 · chatchat-space/Langchain-Chatchat | Apache-2.0 | 99页 / 449KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_langchain-chatchat_03.pdf` | PDF · 文本版 | 由开源文档生成 · chatchat-space/Langchain-Chatchat | Apache-2.0 | 21页 / 170KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_pumpkin-book_01.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/pumpkin-book | CC BY-NC-SA 4.0 | 33页 / 259KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_pumpkin-book_02.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/pumpkin-book | CC BY-NC-SA 4.0 | 27页 / 184KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_pumpkin-book_03.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/pumpkin-book | CC BY-NC-SA 4.0 | 50页 / 273KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_pumpkin-book_04.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/pumpkin-book | CC BY-NC-SA 4.0 | 63页 / 296KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_pumpkin-book_05.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/pumpkin-book | CC BY-NC-SA 4.0 | 60页 / 274KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_pumpkin-book_06.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/pumpkin-book | CC BY-NC-SA 4.0 | 41页 / 268KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_self-llm_01.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/self-llm | CC BY-NC-SA 4.0 | 34页 / 322KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_self-llm_02.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/self-llm | CC BY-NC-SA 4.0 | 38页 / 294KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_self-llm_03.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/self-llm | CC BY-NC-SA 4.0 | 38页 / 289KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_self-llm_04.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/self-llm | CC BY-NC-SA 4.0 | 37页 / 252KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_self-llm_05.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/self-llm | CC BY-NC-SA 4.0 | 48页 / 275KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_self-llm_06.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/self-llm | CC BY-NC-SA 4.0 | 186页 / 684KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pdf_text/text_self-llm_07.pdf` | PDF · 文本版 | 由开源文档生成 · datawhalechina/self-llm | CC BY-NC-SA 4.0 | 63页 / 374KB | 标题层级深、章节编号（1.1/1.1.1）密集，验证 BM25 精确匹配 | ok |
| `pptx/pptx_happy-llm_01.pptx` | PPTX | 由开源文档生成 · datawhalechina/happy-llm | CC BY-NC-SA 4.0 | 28 页 / 72KB | 一页一要点、文本碎片化、备注栏混杂 | ok |
| `pptx/pptx_langchain-chatchat_01.pptx` | PPTX | 由开源文档生成 · chatchat-space/Langchain-Chatchat | Apache-2.0 | 28 页 / 61KB | 一页一要点、文本碎片化、备注栏混杂 | ok |
| `pptx/pptx_langchain-chatchat_02.pptx` | PPTX | 由开源文档生成 · chatchat-space/Langchain-Chatchat | Apache-2.0 | 28 页 / 64KB | 一页一要点、文本碎片化、备注栏混杂 | ok |
| `pptx/pptx_pumpkin-book_01.pptx` | PPTX | 由开源文档生成 · datawhalechina/pumpkin-book | CC BY-NC-SA 4.0 | 28 页 / 66KB | 一页一要点、文本碎片化、备注栏混杂 | ok |
| `pptx/pptx_self-llm_01.pptx` | PPTX | 由开源文档生成 · datawhalechina/self-llm | CC BY-NC-SA 4.0 | 28 页 / 59KB | 一页一要点、文本碎片化、备注栏混杂 | ok |
| `pptx/pptx_self-llm_02.pptx` | PPTX | 由开源文档生成 · datawhalechina/self-llm | CC BY-NC-SA 4.0 | 28 页 / 63KB | 一页一要点、文本碎片化、备注栏混杂 | ok |
| `xlsx/data_stats_01.xlsx` | XLSX | 由开源文档中的表格生成 | CC BY-NC-SA 4.0 / Apache-2.0 | 4 sheet / 8KB | 多 sheet、跨列合并表头、空行分节；整表必须原子化 | ok |
| `xlsx/data_stats_02.xlsx` | XLSX | 由开源文档中的表格生成 | CC BY-NC-SA 4.0 / Apache-2.0 | 4 sheet / 10KB | 多 sheet、跨列合并表头、空行分节；整表必须原子化 | ok |
| `xlsx/data_stats_03.xlsx` | XLSX | 由开源文档中的表格生成 | CC BY-NC-SA 4.0 / Apache-2.0 | 4 sheet / 9KB | 多 sheet、跨列合并表头、空行分节；整表必须原子化 | ok |
| `xlsx/data_stats_04.xlsx` | XLSX | 由开源文档中的表格生成 | CC BY-NC-SA 4.0 / Apache-2.0 | 4 sheet / 9KB | 多 sheet、跨列合并表头、空行分节；整表必须原子化 | ok |
| `xlsx/data_stats_05.xlsx` | XLSX | 由开源文档中的表格生成 | CC BY-NC-SA 4.0 / Apache-2.0 | 4 sheet / 8KB | 多 sheet、跨列合并表头、空行分节；整表必须原子化 | ok |
| `xlsx/data_stats_06.xlsx` | XLSX | 由开源文档中的表格生成 | CC BY-NC-SA 4.0 / Apache-2.0 | 4 sheet / 8KB | 多 sheet、跨列合并表头、空行分节；整表必须原子化 | ok |
| `xlsx/data_stats_07.xlsx` | XLSX | 由开源文档中的表格生成 | CC BY-NC-SA 4.0 / Apache-2.0 | 4 sheet / 8KB | 多 sheet、跨列合并表头、空行分节；整表必须原子化 | ok |
| `xlsx/data_stats_08.xlsx` | XLSX | 由开源文档中的表格生成 | CC BY-NC-SA 4.0 / Apache-2.0 | 4 sheet / 8KB | 多 sheet、跨列合并表头、空行分节；整表必须原子化 | ok |
