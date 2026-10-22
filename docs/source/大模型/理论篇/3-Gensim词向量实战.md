# Gensim词向量实战

## 1. Gensim 简介与核心概念

**Gensim（Generate Similar）** 是专用于处理原始非结构化纯文本的 Python 库，内置 Word2Vec、TF-IDF、LSA、LDA 等多种词向量和主题模型算法，`pip install gensim` 即装。

使用 Gensim 绕不开五个概念：

- **语料库**：Gensim 处理的主要对象，可简单理解为训练数据集。分词后的文档表示为 `list[list[str]]`；用于 TF-IDF、LDA 等模型的标准 BoW 语料库是稀疏向量可迭代对象，每篇文档形如 `[(token_id, frequency), ...]`。
- **词典**：把词语（token）映射到唯一整数 ID 的**词汇表**。用词袋模型前必须先根据整个语料库构建。
- **向量**：文档最终的数学表示。词袋下 `["我","爱","我"]` → `[(0, 2), (1, 1)]`。
- **稀疏向量**：为省内存的高效表示法——One-Hot/词袋这类维度巨大且绝大多数为 0 的向量，只记录**非零项的索引和值**，`[2,1,0,0,...,0]` 存成 `[(0,2),(1,1)]`。
- **模型**：一个**向量转换**算法。如 `TfidfModel` 把词频 BoW 向量转成 TF-IDF 权重向量。

内置算法与前面学的理论一一对应：`models.TfidfModel`（TF-IDF）、`models.LsiModel`（LSA）、`models.LdaModel`（LDA）、`models.Nmf`（NMF）、`models.Word2Vec`、`models.FastText`、`models.Doc2Vec`（段落向量化）。

## 2. 标准工作流：语料 → 词典 → BoW

把原始文本转成 TF-IDF 或主题模型向量遵循三步流程：

1. **准备语料**：分词后整理成 `list[list[str]]`，每个子列表是一篇文档；
2. **创建词典**：遍历所有文档，把每个唯一词元映射到从 0 起的整数 ID；
3. **词袋化**：用词典把每篇文档转成稀疏 BoW 向量 `[(token_id, frequency), ...]`。

最终的 **BoW 语料库**就是训练 TF-IDF、LDA 等模型的标准输入。

> 注意适用范围：三步流程适用于 TF-IDF、LSA、LDA、NMF 等**基于 BoW** 的模型；**不适用于** Word2Vec/FastText/Doc2Vec 等神经网络词向量模型——后者直接以分词后的句子序列 `list[list[str]]` 为输入，无需词袋化。

```python
import jieba
from gensim import corpora

raw_headlines = [
    "央行降息，刺激股市反弹",
    "球队赢得总决赛冠军，球员表现出色"
]
tokenized_headlines = [jieba.lcut(doc) for doc in raw_headlines]

dictionary = corpora.Dictionary(tokenized_headlines)   # {'刺激': 0, '反弹': 1, '央行': 2, ...}
corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized_headlines]
# [[(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)], [(5,1),(6,1),(7,1),(8,1),(9,1),(10,1),(11,1)]]
```

## 3. TF-IDF 与关键词权重

以 6 条含「财经/体育」两个明显主题的新闻标题为语料，训练 TF-IDF 并转换向量：

```python
from gensim import corpora, models

dictionary = corpora.Dictionary(tokenized_headlines)
corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized_headlines]

tfidf_model = models.TfidfModel(corpus_bow)      # 训练
corpus_tfidf = tfidf_model[corpus_bow]            # 语料库转 TF-IDF 向量

new_headline_bow = dictionary.doc2bow(jieba.lcut("股市大涨，牛市来了"))
new_headline_tfidf = tfidf_model[new_headline_bow]  # 对新文本应用模型
```

第一篇标题的输出（按权重降序）：`刺激/反弹/央行/股市/降息` 各约 `0.441`，标点「，」仅 `0.170`；新标题只得到 `[(3, 0.933), (5, 0.361)]` 两项。三个观察：

- BoW 向量是整数词频，TF-IDF 向量是浮点权重——多篇文档共现的「，」权重被压低，特定财经标题里的「股市」「降息」权重更高，正是 TF-IDF 理念的实际效果；
- **词典外的新词被忽略**：新标题「大涨」「牛市」都是 OOV，向量只剩「股市」和「，」；
- 标点进了词典并占有权重，若不希望其干扰相似度，应在构建词典前先移除标点/停用词。

得到的 TF-IDF 向量可用于计算文档相似度或作为机器学习模型的输入特征。

## 4. LDA 与文档主题挖掘

主题模型能从大量文档中自动发现隐藏的、无监督的主题，输入同样是词典和 BoW 语料库：

```python
lda_model = models.LdaModel(corpus=corpus_bow, id2word=dictionary,
                            num_topics=2, random_state=100)
lda_model.print_topics()   # 查看每个主题的关键词及概率
```

小语料上训练出的两个主题（截断展示）：主题 0 聚集「公布/名单/足球/集训/国家队/A股」，主题 1 聚集「篮球/刷新/历史/记录/得分/巨星」。对新标题「巨星詹姆斯获得常规赛MVP」推断出的主题分布为 `[(0, 0.27), (1, 0.73)]`——「巨星」把它拉向体育主题。

LDA 的两种用法由此明确：每篇文档可表示为一个**主题概率分布**（如 90% 体育、10% 财经），每个主题可展开为**核心词列表**。一个边界情况：若新文本在词典中几乎无重叠词（`doc2bow` 结果为空），推断出的主题分布会接近均匀（2 个主题时约 0.5/0.5）——无语料证据时模型不偏袒任何主题。

## 5. Word2Vec 实战

### 5.1 训练与核心参数

Word2Vec 的输入直接是**分词后的句子列表**，目标不是加权或找主题，而是根据上下文学习每个词内在的稠密语义向量。训练结束后网络本身通常被丢弃——**最终目标是那张词向量查询表**，存在 `model.wv`（`KeyedVectors` 实例）里，后续所有应用都围绕它展开。

```python
from gensim.models import Word2Vec

# headlines 为 16 条财经+体育新闻标题，扩充自第 3 节语料
model = Word2Vec(tokenized_headlines, vector_size=50, window=3, min_count=1, sg=1)
```

示例用到的参数：

| 参数 | 含义 | 取值要点 |
|---|---|---|
| `sentences` | 语料库 | 必须 `list[list[str]]` |
| `vector_size` | 词向量维度 | 越高语义越丰富但计算量越大，通常 50-300 |
| `window` | 上下文窗口大小 | 预测一个词时考虑其前后几个词 |
| `min_count` | 最小词频过滤 | 低于阈值的词直接忽略：过滤错别字/罕见词噪音、显著减小模型体积 |
| `sg` | 训练算法 | `0`=CBOW（上下文→中心词）；`1`=Skip-gram（中心词→上下文） |

尚未使用但重要的优化与采样参数：

- `hs`：优化策略，`0`=负采样（Negative Sampling），`1`=Hierarchical Softmax；
- `negative`：`hs=0` 时每个正样本配多少负样本，通常 5-20；
- `sample`：**高频词二次重采样阈值**，随机跳过「的」「是」这类高频词，减少它们对训练的过度影响并提速。

### 5.2 使用词向量

所有操作围绕 `model.wv` 展开：

```python
model.wv.most_similar('股市')        # 寻找最相似的词
model.wv.similarity('球队', '球员')  # 余弦相似度
model.wv['市场'].shape               # 取词向量 → (50,)
```

小语料上的实测输出：与「股市」最相似的是「名单」(0.37)、「央行」(0.36)、「联赛」(0.24)……「球队」和「球员」的相似度甚至是 **-0.1552**（负值）。这个结果不必惊讶——16 条标题的信息量太少，向量远未收敛，数值低且不稳定，仅作 API 演示；但它仍能体现模型学到了主题内的关联（「央行」「证监会」「理财产品」等财经词确实排进了「股市」的近邻）。

> **补记**：把「理想案例」和「实测翻车」放在一起看反而更有价值——Word2Vec 的语义质量完全取决于语料规模与上下文多样性，这也是为什么下一节实战（文本分类）不会直接用这么小的向量，以及为什么大模型时代宁可用字/子词粒度也不用手工小语料训练。

### 5.3 模型持久化

实际项目通常把训练好的词向量存盘。若后续不再增量训练，推荐**只保存 `KeyedVectors`（`model.wv`）**：完整模型对象带着哈夫曼树、梯度累积量等仅训练期需要的中间状态，剔除冗余既省存储又降内存、加快加载，保障线上服务稳定。

```python
from gensim.models import KeyedVectors

model.wv.save("news_vectors.kv")            # 保存词向量
loaded_wv = KeyedVectors.load("news_vectors.kv")  # 加载后同 API 可用
```

## 写在最后

词向量让「词」有了语义，但句子是词的序列，要建模序列就得处理「顺序」——下一篇[循环神经网络](4-循环神经网络)记录 RNN 的「时间展开」与 LSTM/GRU 如何对抗梯度消失。
