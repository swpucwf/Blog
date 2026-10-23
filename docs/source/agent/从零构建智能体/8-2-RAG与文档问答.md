# RAG 与文档问答

记忆解决「发生过什么」，剩下的问题是「文档里写了什么」。模型没读过那份 PDF，也不该为了读它而重训一次。RAG 的做法很朴素：把文档切好、存成向量，提问时先捞出最相关的几段，塞进上下文再让模型回答。

## 1. 三个词与两个阶段

检索增强生成可以拆成三件事：**检索**是从知识库查出相关内容，**增强**是把结果拼进提示词，**生成**是模型基于这些上下文给出答案。整个流程分成两段：

- **数据准备**：数据提取 → 文本分割 → 向量化 → 建索引。
- **应用**：接收提问 → 检索 → 注入 Prompt → 生成答案。

<img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/helloagents-rag-pipeline.png" width="85%" alt="RAG 系统的核心工作原理" />

第一段是离线的、可批量的，第二段是在线的、有延迟预算的。两段的优化手段完全不同：前者拼的是分块质量与索引结构，后者拼的是查询改写与召回策略。很多 RAG 效果不佳，根源是只调后者，却没意识到分块已经把上限锁死了。

## 2. 三代演进

| 阶段 | 检索方式 | 生成方式 | 主要问题 |
| --- | --- | --- | --- |
| 朴素 RAG（2020–2021） | TF-IDF / BM25 关键词匹配 | 检索结果原样拼接 | 只认字面，换个说法就漏 |
| 高级 RAG（2022–2023） | 稠密嵌入的语义检索 | 查询重写、分块优化、重排 | 组件增多，参数需要调 |
| 模块化 RAG（2023–） | 混合检索、多查询扩展、HyDE | 思维链、自我反思与修正 | 复杂度与成本上升 |

这条演进线其实在回答同一个问题：**查询和文档在语义空间里对不上**。关键词匹配解决不了同义改写，稠密向量缓解了一部分，但疑问句与陈述句的分布差异依然存在——于是有了后面要讲的 MQE 与 HyDE。

## 3. 五层七步

工程上把 RAG 拆成五层：用户层（`RAGTool` 统一接口）、应用层（问答、检索、管理）、处理层（解析、分块、向量化）、存储层（向量库、文档存储）、基础层（嵌入模型、LLM、数据库）。分层的意义在于每层可独立替换——把嵌入模型从百炼换成 `all-MiniLM-L6-v2`，上层代码一行不用改。

```python
class RAGTool(Tool):
    def __init__(self, knowledge_base_path="./knowledge_base", qdrant_url=None,
                 qdrant_api_key=None, collection_name="rag_knowledge_base",
                 rag_namespace="default"):
        self._pipelines: Dict[str, Dict[str, Any]] = {}
        self.llm = HelloAgentsLLM()
        self._pipelines[rag_namespace] = create_rag_pipeline(...)
```

`rag_namespace` 是个容易被忽视但很实用的设计：同一个向量库里按命名空间隔离不同用户或不同知识库，检索时带上过滤条件，避免把别人的文档召回进来。

## 4. 文档统一：先转成 Markdown

分块质量取决于结构信息，而 PDF、Word、Excel 的结构各不相同。解法是**先把一切转成 Markdown**，再统一处理。这里用微软开源的 MarkItDown：

```python
def _convert_to_markdown(path: str) -> str:
    ext = (os.path.splitext(path)[1] or "").lower()
    if ext == ".pdf":
        return _enhanced_pdf_processing(path)          # PDF 单独走增强处理
    md_instance = _get_markitdown_instance()
    if md_instance is None:
        return _fallback_text_reader(path)
    try:
        result = md_instance.convert(path)
        markdown_text = getattr(result, "text_content", None)
        return markdown_text if isinstance(markdown_text, str) else ""
    except Exception:
        return _fallback_text_reader(path)
```

支持面很宽：PDF/Word/Excel/PPT 走解析，图片走 OCR，音频走转录，纯文本与代码直接读。两条兜底是有意留的——PDF 需要专门处理（版式复杂、常有扫描件），而 MarkItDown 缺失或转换失败时不能让整条流水线挂掉。

统一到 Markdown 之后，结构信息就保住了：标题层级代表章节关系，段落边界代表语义边界。这正是分块能「切得准」的前提。

## 5. 分块：先按结构，再按 Token

分块分两步。第一步按标题层次切段落，用一个栈维护「当前在哪一节下面」：

```python
for ln in lines:
    if ln.strip().startswith("#"):
        flush_buf(char_pos)                       # 遇到标题先收尾上一段
        level = len(ln) - len(ln.lstrip("#"))
        title = ln.lstrip("#").strip()
        if level <= len(heading_stack):
            heading_stack = heading_stack[:level - 1]
        heading_stack.append(title)
        continue
    ...
    paragraphs.append({"content": content,
                       "heading_path": " > ".join(heading_stack) or None,
                       "start": ..., "end": ...})
```

每个段落都带上 `heading_path`（如 `第三章 > 3.2 注意力机制`），这一步的价值在检索阶段才显现：返回片段时同时给出它所属的章节路径，模型才知道这段在讲什么位置的内容，用户也能判断可信度。

第二步按 Token 数打包，并保留重叠：

```python
while i < len(paragraphs):
    p = paragraphs[i]
    p_tokens = _approx_token_len(p["content"]) or 1
    if cur_tokens + p_tokens <= chunk_tokens or not cur:
        cur.append(p); cur_tokens += p_tokens; i += 1
    else:
        chunks.append({...})                       # 收一个块
        if overlap_tokens > 0:                     # 从尾部回退若干段落做重叠
            ...
```

以段落为单位而非按字符硬切，能避免把一句话劈成两半。重叠的作用是保住跨块的连续性：一个概念的解释可能横跨两个块，没有重叠时两边的信息都不完整，检索命中哪一块都答不周全。

Token 估算专门处理了中英文混排：

```python
def _approx_token_len(text: str) -> int:
    cjk = sum(1 for ch in text if _is_cjk(ch))     # 一个汉字按一个 token
    non_cjk_tokens = len([t for t in text.split() if t])
    return cjk + non_cjk_tokens
```

这个估算不精确，但对「控制块大小」这个目的是够用的。用真实 tokenizer 当然更准，代价是引入分词器依赖与计算开销——在分块这种粗粒度场景里不值得。

## 6. 嵌入与索引

嵌入模型决定检索上限。统一接口提供三级方案：百炼 API（质量最好）、本地 `all-MiniLM-L6-v2`（离线可用）、TF-IDF（无依赖兜底）。索引时批量编码并校验维度：

```python
embedder = get_text_embedder()
dimension = get_dimension(384)

for i in range(0, len(processed_texts), batch_size):
    part = processed_texts[i:i + batch_size]
    part_vecs = embedder.encode(part)
    for v in part_vecs:
        v_norm = [float(x) for x in (v.tolist() if hasattr(v, "tolist") else v)]
        if len(v_norm) != dimension:                # 维度不一致就补齐或截断
            v_norm = (v_norm + [0.0] * dimension)[:dimension]
        vecs.append(v_norm)
```

维度检查看着啰嗦，实际必写：换嵌入模型时维度会变（384、768、1024 都常见），而向量库的集合维度是建库时定死的。没有这层校验，报错会出现在 Qdrant 那一侧，且信息模糊。

编码前还有一步预处理——把 Markdown 语法符号（`#`、`|`、代码围栏）淡化掉。这些符号对语义没有贡献，却会稀释向量。

## 7. 高级检索：改写查询，而不是改文档

**MQE（多查询扩展）** 的洞察是：同一个问题有多种问法，不同问法会命中不同文档。让 LLM 先生成几个语义等价的查询，再并行检索、合并结果。

```python
def _prompt_mqe(query: str, n: int) -> List[str]:
    prompt = [
        {"role": "system", "content": "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。"},
        {"role": "user", "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。"},
    ]
    text = llm.invoke(prompt)
    outs = [ln.strip("- \t") for ln in (text or "").splitlines() if ln.strip()]
    return outs[:n] or [query]                      # 失败就退回原查询
```

注意最后一行：扩展失败时返回原查询，而不是空列表。检索链路上的每个增强步骤都该有这种「退化即原样」的兜底。

**HyDE（假设文档嵌入）** 处理的是另一类鸿沟：问题与文档在语义空间里分布不同——一个是疑问句，一个是陈述句。解法是让模型先写一段「可能的答案」，再用这段答案去检索。

```python
def _prompt_hyde(query: str) -> Optional[str]:
    prompt = [
        {"role": "system", "content": "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。"},
        {"role": "user", "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"},
    ]
    return llm.invoke(prompt)
```

关键在于**假设答案不必正确**——它提供的是术语分布与表述风格。即便内容有偏差，只要它用了文档里会出现的词，就能把检索导向正确的区域。

两者合进一个框架，流程是「扩展 → 检索 → 合并」：

```python
expansions = [query]
if enable_mqe:  expansions.extend(_prompt_mqe(query, mqe_expansions))
if enable_hyde: expansions.append(_prompt_hyde(query))

pool = max(top_k * candidate_pool_multiplier, 20)   # 候选池放大 4 倍
per = max(1, pool // max(1, len(expansions)))

agg = {}
for q in expansions:
    for h in store.search_similar(embed_query(q), limit=per, where=where):
        mid = h["metadata"]["memory_id"]
        if mid not in agg or h["score"] > agg[mid]["score"]:
            agg[mid] = h                                # 去重，保留最高分

return sorted(agg.values(), key=lambda x: x["score"], reverse=True)[:top_k]
```

`candidate_pool_multiplier` 是这套机制的成本开关：候选池放大 4 倍，意味着检索更充分，也意味着更高的向量查询开销。去重时按 `memory_id` 合并并保留最高分，是对的——同一段文档被三个扩展查询命中，说明它更相关，不该被平均掉。

三种策略的选择可以简化成一条：一般查询开 MQE；专业领域（术语多、表述差异大）同时开 MQE 与 HyDE；延迟敏感就只留基础检索。

## 8. 拼成文档问答助手

把两个工具组合成助手，核心只有 `ask` 方法——它的价值在于**两个系统各司其职并互相喂数据**：

```python
def ask(self, question: str, use_advanced_search: bool = True) -> str:
    if not self.current_document:
        return "⚠️ 请先加载文档！"

    self.memory_tool.execute("add", content=f"提问: {question}",
                             memory_type="working", importance=0.6, session_id=self.session_id)

    answer = self.rag_tool.execute("ask", question=question, limit=5,
                                   enable_advanced_search=use_advanced_search,
                                   enable_mqe=use_advanced_search,
                                   enable_hyde=use_advanced_search)

    self.memory_tool.execute("add", content=f"关于'{question}'的学习",
                             memory_type="episodic", importance=0.7,
                             event_type="qa_interaction", session_id=self.session_id)
    self.stats["questions_asked"] += 1
    return answer
```

三次调用对应三种职责：问题进工作记忆（当前会话的短期上下文），RAG 负责取知识与生成，事件进情景记忆（长期可回顾的学习轨迹）。工作记忆重要性 0.6、情景记忆 0.7 的差异也不是随手写的——后者是要长期留存的经历，阈值高一点才能扛过后续的整合与遗忘。

<img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/helloagents-doc-assistant-flow.png" width="85%" alt="智能问答助手的执行流程" />

完整应用的流程是五步：加载文档（分块入向量库 + 记录事件）→ 提问（检索 + 记忆）→ 记忆管理（增删查整合）→ 智能路由（按问题类型决定走 RAG 还是走记忆）→ 生成学习报告（汇总统计）。前两步是主线，后三步是把两个系统真正用起来的部分。

## 9. 什么时候值得上 RAG

先划清边界。RAG 适合的情况：知识有明确出处、会更新、体量大到塞不进上下文、需要给出引用。不适合的情况：知识是稳定常识（模型的参数里就有）、任务是推理而非查证（检索帮不上逻辑推演）、以及对延迟极敏感的高频调用。

几个成本最容易被低估的地方：

- **分块决定上限**。块太小丢上下文，太大稀释语义，检索到一段八百字里只有两句相关的文字，模型照样答不准。标题路径、重叠、块大小这三个参数要按文档类型调，不能一套走天下。
- **top_k 与阈值要一起定**。返回 5 条低分片段比返回 1 条高分片段更糟——噪声会淹没信号。`min_score` 这类阈值不是可选项。
- **检索失败要能看出来**。RAG 最常见的坏结果是「检索没命中，模型凭记忆编」，而输出看上去毫无异常。所以助手类应用通常要把命中的片段与来源一起返回，让人能核对。
- **需要单独的评估**。答案对不对、引用准不准、该召回的有没有召回，是三个不同的指标，混在一起看只会得出「感觉还行」的结论。评估方法留到[12-智能体性能评估](/source/agent/从零构建智能体/12-智能体性能评估)。

记忆与检索填上之后，框架已经能记住事、查到知识。但真正决定智能体表现的那一项还没碰——每次调用到底往上下文里放什么。下一篇进入上下文工程——[9-1-上下文工程与ContextBuilder](/source/agent/从零构建智能体/9-1-上下文工程与ContextBuilder)。
