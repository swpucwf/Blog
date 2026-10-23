# RAG 框架选型

## 1. 主流框架

| 框架 | 特点 |
|---|---|
| RAGFlow | 企业级知识处理引擎，深度文档解析与表格识别最强 |
| LlamaIndex | 数据框架出身，RAG 核心工具，Agent 集成好 |
| LangChain | RAG 基础组件最丰富，生态最大 |
| Haystack | deepset 出品，模块化管线，向量库与 LLM 可自由切换 |
| FastGPT | 轻量易用，小时级搭出知识库问答 |

## 2. 横向对比

| 框架 | 文档解析 | 企业级 | 部署 | 学习曲线 | 适用规模 |
|---|---|---|---|---|---|
| RAGFlow | 极强（深度解析、表格识别） | 是 | 自托管 / 云 | 中 | 大 |
| LlamaIndex | 强 | 是 | 自托管 | 中 | 大 |
| LangChain | 强（组件丰富） | 是 | 自托管 | 中 | 大 |
| Haystack | 强（模块化） | 是 | 自托管 | 中 | 中到大 |
| FastGPT | 中 | 否 | 自托管 | 低 | 中小 |

差距最大的一列是文档解析。PDF 里的表格、多栏排版、扫描件，是 RAG 效果的第一道分水岭——RAGFlow 在这块投入最重，LlamaIndex 与 LangChain 靠插件与 Unstructured 补齐，FastGPT 则定位轻量。

## 3. LlamaIndex 最小示例

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

documents = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(documents)
engine = index.as_query_engine(similarity_top_k=3)
print(engine.query("文档的核心内容是什么？"))
```

两个参数最常调：`similarity_top_k` 控制召回数量，`chunk_size` 控制切分粒度（默认 1024，中文建议 256～512）。切分与嵌入的原理细节，见[RAG 技术全栈](/source/agent/RAG技术全栈/README)的数据与嵌入章节。

## 4. 选型路径

- 快速验证：FastGPT，小时级上线；
- 文档密集（PDF、表格、扫描件）：RAGFlow；
- 需要灵活编排与 Agent 集成：LlamaIndex、LangChain；
- 要模块化切换向量库与模型：Haystack；
- 企业私有化：RAGFlow、Dify（低代码、可私有部署）。

判断顺序建议是「文档形态 → 团队栈 → 规模」，而不是「谁的 star 多」。文档解析能力跟不上时，后面所有检索优化都是在错误的分块上做文章。

## 5. 四条演进方向

### 5.1 GraphRAG

在向量检索之上叠加知识图谱层，先做实体关系抽取构建语义网络，再用图遍历与向量混合检索。

```text
文档 → 实体/关系抽取 → 知识图谱 → 图遍历 + 向量检索 → 生成
```

代表项目：Microsoft GraphRAG（支持全局摘要与局部检索两种模式）、Neo4j + LLM（适合已有图数据库的企业）、LightRAG（轻量、成本更低）。

适合的场景有共同特征——**信息之间存在显式关系**：法律判例的法条引用、药物的靶点与疾病、企业制度的上下级引用、论文的引用网络。这些场景下向量检索容易漏掉关联先例，沿图遍历能补回来。

### 5.2 Agentic RAG

用 Agent 替代固定流水线：自主判断该不该查、去哪查、查完够不够。

| 维度 | 传统 RAG | Agentic RAG |
|---|---|---|
| 检索次数 | 一次 | 多轮，直到满足 |
| 检索源 | 固定向量库 | 向量库 + Web + API + 数据库 |
| 决策主体 | 预设管线 | Agent 自主决策 |
| 失败处理 | 静默返回不完整结果 | 重写查询、换源、自我纠错 |

落地三条建议：先用传统 RAG 跑通，发现「一次检索不够」再加循环；检索步数设上限（3～5 步），避免无限消耗 token；每次检索的 query、来源、结果都要可追溯。

### 5.3 多模态 RAG

支持图像、表格、音视频检索。代表项目：ColPali（视觉语言模型驱动的 PDF 检索，无需 OCR）、LlamaIndex 多模态索引、RAGFlow 的深度文档解析。

### 5.4 上下文工程

从「丢一段检索结果给 LLM」升级为系统性编排：去噪、排序、压缩、多文档融合，确保每个 token 都有信息量。这一层与前三条不冲突，是它们共同的底座。

## 6. 一个务实判断

框架选型对最终效果的影响，远小于数据处理与评测。同一套 RAGFlow，解析干净的和解析糟糕的，效果差距可能是数量级；而换框架带来的差异通常在 10%～20% 区间。所以顺序应该是：**先把文档解析与分块做对，再考虑框架替换**。

## 写在最后

框架决定流程怎么组织，向量库决定检索跑在哪。下一篇[向量知识库选型](10-向量知识库选型)从架构路线讲到索引类型与混合检索。
