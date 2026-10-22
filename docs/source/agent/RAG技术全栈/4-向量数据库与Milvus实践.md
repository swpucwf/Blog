# 向量数据库与 Milvus 实践

## 1. 为什么需要专用数据库

当向量从几百个涨到百万、亿级，「从海量向量里找最相似的几个」就成了核心问题。暴力遍历的代价随规模线性上升，传统数据库的 B-Tree 索引对高维距离计算无能为力。

| 维度 | 向量数据库 | 传统关系型数据库 |
|---|---|---|
| 核心数据 | 高维向量 | 结构化数据 |
| 查询方式 | 相似性搜索（ANN） | 精确匹配 |
| 索引机制 | HNSW、IVF、LSH、PQ | B-Tree、Hash |
| 数据规模 | 千亿级向量 | 千万到亿级行 |
| 一致性 | 通常最终一致 | 强一致（ACID） |

两者不是替代关系，而是互补：关系库存业务元数据，向量库存语义表示，查询时先向量召回、再按元数据过滤或校验。

## 2. 四类近似最近邻技术

- **基于树**：如 Annoy 的随机投影树，把搜索复杂度压到对数级；
- **基于哈希**：如 LSH，用哈希函数把相似向量映射进同一个桶；
- **基于图**：如 HNSW，构建多层邻近图，从粗到细导航；
- **基于量化**：如 Faiss 的 IVF 与 PQ，先用聚类分桶，再压缩桶内向量。

四类的取舍集中在三个指标上：召回率、查询延迟、内存占用。没有全赢的方案，只有匹配数据规模与延迟预算的方案。

## 3. 主流产品怎么选

| 产品 | 定位 | 适合场景 |
|---|---|---|
| Chroma | 轻量、本地优先、零配置 | 原型、教学、小规模 |
| FAISS | 算法库，索引落本地文件 | 快速原型、中小规模 |
| Milvus | 开源分布式，支持 GPU 与多种索引 | 大规模、高性能、自建 |
| Qdrant | Rust 实现，二进制量化 | 性能敏感、高并发 |
| Weaviate | GraphQL 接口、内置 AI 模块 | 多模态、快速集成 |
| Pinecone | 全托管 Serverless | 企业生产、免运维 |

经验规则：百万级以下从 Chroma 或 FAISS 起步，几行代码就能跑通；超过百万级，或需要高并发、实时更新、复杂元数据过滤时，再考虑 Milvus、Weaviate 或托管服务。

## 4. FAISS：把索引当文件

FAISS 本质是算法库而非数据库服务，索引直接保存成 `.faiss` 文件加 `.pkl` 映射，轻量且高效。

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("./faiss_index_store")

loaded = FAISS.load_local(
    "./faiss_index_store", embeddings,
    allow_dangerous_deserialization=True,
)
results = loaded.similarity_search("FAISS 是做什么的？", k=1)
```

`from_documents` 内部是四层调用：`from_documents` 抽取文本与元数据 → `from_texts` 批量向量化 → `__from` 建立空索引框架（默认 L2）→ `__add` 写入向量、文档与 ID 映射。理解这四层，排查「召回为空」时能快速判断是向量化出错还是映射丢失。

## 5. Milvus 单机部署

```bash
# 下载官方单机编排文件
wget https://github.com/milvus-io/milvus/releases/download/v2.5.14/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker compose up -d
docker ps      # 应看到 milvus-standalone / minio / etcd 三个容器
```

三个容器各有职责：`etcd` 存元数据，`MinIO` 做对象存储，`milvus-standalone` 提供服务，默认端口 `19530`。常用管理命令：`docker compose down` 停止并保留数据卷，`docker compose down -v` 连同数据一起清除。

## 6. 四个核心概念

用一个图书馆比喻最省事：

| 概念 | 类比 | 作用 |
|---|---|---|
| Collection | 图书馆 | 数据顶层容器，相当于一张表 |
| Partition | 分区（小说区、科技区） | 物理隔离，查询可只扫指定分区 |
| Schema | 图书卡片规则 | 定义主键、向量、标量字段 |
| Alias | 推荐书单 | 指向某个 Collection，支持原子切换 |

Schema 是最需要提前设计的部分。一个图文混合的典型设计：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/allinrag_4_1.webp" width="80%" alt="多模态混合向量 Schema 设计" />
  <p>图 1：一篇文章的 ID、文本元数据、密集向量与稀疏向量</p>
</div>

一个 Collection 可以有多个向量字段（文本密集、图像密集、文本稀疏），这正是原生混合检索的基础。分区上限 1024 个，合理使用能把扫描量降下来。

Alias 的价值在更新场景：建好 `collection_v2` 并导入索引完成后，把指向 `collection_v1` 的别名原子切到新集合，应用层完全无感。这是生产环境做索引重建的标准做法。

## 7. 索引选型

| 索引 | 原理 | 召回 | 内存 | 场景 |
|---|---|---|---|---|
| FLAT | 暴力精确计算 | 100% | 高 | 百万级以内、要求精确 |
| IVF_FLAT | 聚类分桶后桶内精确搜 | 高 | 中 | 内存充足、追求吞吐 |
| IVF_SQ8 / IVF_PQ | 桶内向量量化压缩 | 中 | 低 | 内存受限 |
| HNSW | 多层邻近图导航 | 很高 | 高 | 低延迟、可全量入内存 |
| DiskANN | SSD 上的图索引 | 高 | 低 | 数据量巨大、无法全量入内存 |

参数上要理解两个旋钮：IVF 的 `nlist`（分桶数）与查询时的 `nprobe`（扫几个桶），HNSW 的 `M`（每点连边数）与 `ef`（搜索宽度）。调大通常换召回、牺牲延迟与内存。

## 8. 从建集合到检索

```python
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType

client = MilvusClient("http://localhost:19530")
dim = len(encoder.encode_image(image_list[0]))

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),
]
client.create_collection(
    collection_name="multimodal_demo",
    schema=CollectionSchema(fields, description="多模态图文检索"),
)
```

写入与检索：

```python
data = [{"vector": encoder.encode_image(p), "image_path": p} for p in image_list]
client.insert(collection_name="multimodal_demo", data=data)

client.create_index(
    collection_name="multimodal_demo",
    field_name="vector",
    index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}},
)
client.load_collection("multimodal_demo")

hits = client.search(
    collection_name="multimodal_demo",
    data=[encoder.encode_query(image_path="query.png", text="similar image")],
    anns_field="vector",
    limit=5,
    output_fields=["image_path"],
)
```

三个必须对齐的点：`dim` 必须与模型输出维度一致；`metric_type` 必须与索引、查询保持一致；插入后要 `load_collection` 才能检索。

## 9. 增强检索的四种形态

基础 ANN 之外，Milvus 提供四类能力，覆盖大多数业务需求：

- **过滤检索**：先按标量条件筛实体，再在子集内做 ANN。电商「只看 500 元以下且有库存」就是这一类；
- **范围检索**：返回距离落在阈值区间内的全部结果，适合人脸验证、异常检测；
- **多向量混合检索**：同一次请求里检索多个向量字段，用 `RRFRanker` 或 `WeightedRanker` 融合结果。RAG 里常见的「密集 + 稀疏」双路召回就是它；
- **分组检索**：按字段去重，避免 Top-10 全来自同一本文档，提升结果多样性。

## 10. 实践中的坑

- **维度写死**：换嵌入模型必须重建集合，维度不同无法兼容；
- **度规不一致**：建索引用 COSINE、查询用 L2，结果会莫名其妙地差；
- **忘记 load**：数据插入成功但检索为空，先检查集合是否已加载；
- **分区滥用**：分区用于隔离扫描范围，不是分类标签，按业务合理划分即可；
- **只看召回不看延迟**：HNSW 参数调大召回会上升，但延迟与内存同样上升，必须一起测。

## 写在最后

集合建好、索引可用，下一步是让检索「更准」。下一篇[索引优化](5-索引优化)讨论分片、量化、参数调优与容量规划，回答「同样的数据，怎么让它查得更快更准」。
