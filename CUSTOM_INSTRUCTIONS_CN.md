## Local Read MCP 文档处理策略

当需要处理文档文件时,请严格遵循以下规则:

#### 1. 工具选择优先级

**使用MCP工具处理二进制文件**:
- PDF文件 -> `read_pdf` 工具(自动修复LaTeX公式,支持数学符号)
- Word文档(.docx/.doc) -> `read_word` 工具(保留格式,转换为markdown)
- Excel表格(.xlsx/.xls) -> `read_excel` 工具(转换为markdown表格,保留颜色格式)
- PowerPoint(.pptx/.ppt) -> `read_powerpoint` 工具(提取幻灯片内容)
- HTML文件 -> `read_html` 工具(自动清理脚本和样式)
- ZIP压缩包 -> `read_zip` 工具(自动解压并处理内部文件)
- JSON文件 -> `read_json` 工具(格式化输出)
- CSV文件 -> `read_csv` 工具(转换为markdown表格)
- YAML文件 -> `read_yaml` 工具(解析YAML结构)

**避免使用Read工具**:
- **[错误]** 不要用Read工具直接读取二进制文件(PDF、Word、Excel、PPT等)
- **[错误]** Read工具会返回乱码或无法解析的内容
- **[正确]** Read工具只适用于纯文本文件(.txt、.md、.py、.sh、.log等)

#### 2. 大文件处理策略

对于大文件(预计超过10,000字符),使用分步策略:

**第一步 - 预览模式**:
```python
# 先快速预览前100行,判断文件性质
read_pdf(
    file_path="large_file.pdf",
    preview_only=True,
    preview_lines=100
)
```

**第二步 - 获取元数据**:
```python
# 了解文件大小和结构
read_pdf(
    file_path="large_file.pdf",
    extract_metadata=True,
    return_format="json"
)
```

**第三步 - 分页处理**:
```python
# 逐页读取,避免超时
read_pdf(
    file_path="large_file.pdf",
    page=1,              # 页码从1开始
    page_size=10000,     # 每页10000字符
    return_format="json"
)
```

**第四步 - 精确提取**(可选):
```python
# 使用offset和limit精确提取特定位置的内容
read_pdf(
    file_path="large_file.pdf",
    offset=5000,         # 从第5000字符开始
    limit=3000,          # 读取3000字符
    return_format="json"
)
```

#### 3. 结构化提取功能

当需要深度分析文档时,启用结构化提取:

**提取章节结构**:
```python
read_pdf(
    file_path="document.pdf",
    extract_sections=True,    # 提取所有标题和章节
    return_format="json"      # 获取sections数组
)
```

**提取表格信息**(Excel专用):
```python
read_excel(
    file_path="data.xlsx",
    extract_tables=True,      # 提取每个工作表的表格信息
    return_format="json"
)
```

**提取文件元数据**:
```python
read_pdf(
    file_path="document.pdf",
    extract_metadata=True,    # 获取文件大小、路径、时间戳等
    return_format="json"
)
```

#### 4. 学术论文特殊处理

处理包含数学公式的PDF时,`read_pdf`会自动修复LaTeX符号:
- CID占位符 `(cid:16)` -> 左尖括号
- LaTeX命令 `\alpha` -> 希腊字母alpha
- 数学符号 `\sum` -> 求和符号
- 数学符号 `\int` -> 积分符号

这对科研文献、技术文档的准确理解至关重要.

#### 5. 会话管理

处理同一文件的多次请求时,复用session_id提升性能:

```python
# 第一次请求,获取session_id
result1 = read_pdf(
    file_path="document.pdf",
    page=1,
    return_format="json"
)
session_id = result1["session_id"]

# 后续请求复用
result2 = read_pdf(
    file_path="document.pdf",
    page=2,
    session_id=session_id,    # 复用会话
    return_format="json"
)
```

#### 6. 返回格式选择

**text格式**(默认):
- 简单任务使用
- 只返回文本内容和标题
- 兼容性最好

**json格式**(推荐用于复杂任务):
- 返回完整的结构化数据
- 包含元数据、章节、表格、分页信息
- 便于程序化处理和深度分析

```python
# 简单任务
result = read_pdf(file_path="doc.pdf")  # 默认text格式

# 复杂分析
result = read_pdf(
    file_path="doc.pdf",
    extract_sections=True,
    extract_metadata=True,
    return_format="json"  # 获取完整结构
)
```

#### 7. 性能优化检查清单

在处理文档前,请自检:

- 是否使用了正确的MCP工具?(不要用Read读取二进制文件)
- 大文件是否先预览再处理?(preview_only=True)
- 是否需要结构化提取?(extract_sections/tables/metadata)
- 复杂任务是否使用JSON格式?(return_format="json")
- 是否考虑了分页处理?(page参数或offset/limit)
- 连续请求是否复用了session_id?

#### 8. 常见错误避免

**[错误示例]**:
```python
# 不要这样做!
Read("/path/to/document.pdf")  # 会得到乱码
```

**[正确示例]**:
```python
# 应该这样做
read_pdf("/path/to/document.pdf")
```

**常见错误对比**:

<error>
<case>一次性读取100MB的PDF全文</case>
<solution>先预览 -> 检查大小 -> 分页处理</solution>
</error>

<error>
<case>每次都重新读取整个文件</case>
<solution>使用session_id和分页增量读取</solution>
</error>

<error>
<case>只获取纯文本,丢失章节和表格信息</case>
<solution>启用extract_*参数,使用return_format="json"</solution>
</error>

---

**遵循这些规则,可确保高效、准确地处理各类文档,充分发挥Local Read MCP的全部能力.**
