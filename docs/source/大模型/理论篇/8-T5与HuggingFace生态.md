# T5与HuggingFace生态

## 1. Encoder-Decoder 路线的继承者：T5

BERT 基于 Encoder 擅长理解，GPT 基于 Decoder 擅长生成，那么 Transformer 最原始的 **Encoder-Decoder** 结构去哪了？Google 的 **T5（Text-to-Text Transfer Transformer）** 回归了这一经典架构——它不去调和「理解 vs 生成」，而是在**输入输出形式**上找通解：**把所有 NLP 任务都视为「文本到文本」的转换**。

<div align="center">
  <img src="{{IMG:5_3_1.png}}" width="90%" alt="Text-to-Text 框架示意图" />
  <p>图 1：Text-to-Text 框架</p>
</div>

### 1.1 万物皆文本

BERT 模式下不同任务要设计不同结构：分类加全连接头、序列标注接 CRF、相似度用双塔或拼接回归层。T5 打破了这种模式——**无论什么任务，输入是文本，输出也是文本**：

- **翻译**：`翻译成英文: 黑神话悟空真好玩` → `Black Myth: Wukong is really fun.`
- **情感分类**：`情感分析: 黑神话悟空真好玩` → `正面`
- **摘要**：`摘要: 黑神话悟空是一款以中国神话为背景的动作角色扮演游戏...` → `黑神话悟空是国产 3A 动作游戏。`
- **回归打分**：`计算相似度 句子1: 黑神话悟空真好玩 句子2: 这猴子游戏真不错` → `4.5`（直接生成数字文本）

### 1.2 任务前缀：提示词的先驱

为让同一个模型区分任务，T5 引入 **Task Prefix**：翻译加 `翻译成英文:`、摘要加 `摘要:`。这正是大模型时代**提示词的雏形**。虽然同期的 GPT-2 也展示过类似续写能力，但 T5 是较早系统化地把「**用自然语言指令显式定义任务**」应用于大规模预训练与多任务设置的模型——它证明可以通过改变输入文本（Instruction）来「编程」模型，而不只是改结构。

> **T5 Prefix 与 GPT Prompt 的出发点是不同的**：Prefix 主要用于**有监督微调**——模型训练时见过这些前缀，它们是多任务学习的标记，告诉模型当前该调动哪部分参数；Prompt 主要用于**零/少样本推理**——预训练时可能没见过特定 Prompt，模型靠泛化能力理解意图、无需更新参数。

多任务训练还须解决数据失衡：翻译、摘要、分类的数据量差异巨大，简单混合会让大任务淹没小任务。T5 采用**带上限的比例混合**——对大数据集设采样上限、适当提升小数据集采样概率，确保各种能力「雨露均沾」。

### 1.3 Span Corruption：为 Encoder-Decoder 定制的预训练目标

不同于 BERT 的单字掩码或 GPT 的单向预测，T5 设计了**片段破坏与重构（Span Corruption）**：**破坏**——随机选若干**连续片段（Span）**替换为哨兵符 `<extra_id_0>`、`<extra_id_1>`；**重构**——要求 Decoder 生成被遮盖的片段。

<div align="center">
  <img src="{{IMG:5_3_2.png}}" width="90%" alt="Span Corruption 预训练目标示意图" />
  <p>图 2：Span Corruption 预训练目标</p>
</div>

两个关键设计：

1. **Mask 比例与长度**：大量实验定下遮盖 **15%** token、平均片段长 **3** 的最佳配置——比 BERT 只遮单个 token 更有挑战性，迫使模型理解更长的上下文依赖；
2. **哨兵符唯一性**：`<extra_id_i>` 不与普通词共享，Decoder 能精确定位正在恢复哪一段。

例：原文「黑神话悟空是一款以中国神话为背景的动作角色扮演游戏。」——Encoder 输入 `黑神话悟空是一款<extra_id_0>的动作<extra_id_1>游戏。`，Decoder 目标 `<extra_id_0>以中国神话为背景<extra_id_1>角色扮演<extra_id_2>`，末尾 `<extra_id_2>` 兼作结束符。这个任务兼顾**理解（Encoder）与生成（Decoder）**，配合超大规模清洗语料 **C4**，模型学到极丰富的语言知识。

### 1.4 架构与三项技术改进

宏观结构几乎与原始 Transformer 一致：Encoder 理解输入（如 BERT）、Decoder 自回归生成（如 GPT），双向理解 + 生成能力完美契合 Text-to-Text。微观上 T5 引入的多项「黑科技」后来成了 PaLM、LLaMA 等现代大模型的标配：

- **相对位置编码（分桶）**：绝对位置编码给每个位置固定向量，T5 认为注意力应关注词间**相对距离**。基于**分桶（Bucketing）**：近距离（<8）精确区分，远距离经**对数映射**归入同桶；编码不加入 Input Embedding，而是作为 **Bias 直接加在 Attention Score（Q·K）矩阵**上；且**参数在所有层之间共享**，大幅省参。
- **简化 LayerNorm**：去掉加性偏置（Additive Bias），仅对激活值缩放，减参数减开销。FFN 激活原商用标准 ReLU，**T5 v1.1** 才引入 GEGLU 门控激活。
- **SentencePiece 分词**：BERT 的 WordPiece 需先按空格预分词，对中文等无空格语言不友好；SentencePiece 直接在**原始文本**上训练，空格视为特殊字符 `_`，天然支持多语言混合训练——符合「大一统」哲学。

## 2. T5 实战与分桶源码

### 2.1 Text-to-Text 演示

`T5Tokenizer` + `T5ForConditionalGeneration` 加载 `t5-small`，输入必须带任务前缀：

```python
input_text_1 = "translate English to German: The house is wonderful."
input_text_2 = "stsb sentence1: The rhino grazed on the grass. sentence2: A rhino is grazing in a field."
outputs = model.generate(**tokenizer([input_text_1, input_text_2], return_tensors="pt", padding=True))
# 输出 1: Das Haus ist wunderbar.
# 输出 2: 4.0
```

同一模型靠 `translate...`/`stsb...` 前缀灵活切换；第二个例子不是像 BERT 那样输出回归数值，而是直接生成字符串 `"4.0"`——万物皆文本。

> **换前缀测翻译的意外发现**：把 `translate English to German` 改成 `translate English to Chinese`，模型**依然输出德语**——原版 T5 训练时只有英→德/法/罗马尼亚语等少数语种翻译任务，没有英→中指令；词表基于英文相关语料几乎不含中文字符。模型只是机械匹配到 `translate English to` 模式，按训练分布生成最常见的目标语种之一。处理中文任务要用支持 101 种语言的 **mT5**。

### 2.2 相对位置编码的源码逻辑

`transformers` 中 T5 的位置编码不在 Embedding 而在 Attention 层——`T5Attention.compute_bias` 封装了「位置索引 → Bias 矩阵」全链条（训练前向不一定直接调它，但最适合切入理解原理）：

```python
context_position = torch.arange(query_length)[:, None]    # Query 位置列
memory_position = torch.arange(key_length)[None, :]       # Key 位置行
relative_position = memory_position - context_position    # (q_len, k_len) 相对距离矩阵
relative_position_bucket = self._relative_position_bucket(relative_position, ...)
values = self.relative_attention_bias(relative_position_bucket)  # 查可学习 Embedding 得 bias
```

三点设计意图：

1. **解耦位置与内容**：BERT 把位置加进 Input Embedding，位置和内容第一层就混合；T5 在每层注意力上直接加位置偏置，让位置信息**更直接地作用于注意力权重**；
2. **参数效率**：为每个距离学一个 Bias 参数量爆炸，分桶把无限距离映射到有限桶（如 32 个）；
3. **对数映射「近密远疏」**：人类语言对近距离语法依赖（主谓搭配）敏感需精确建模，远距离语义依赖知道「大概很远」就够。

分桶函数 `_relative_position_bucket` 的实测映射（32 桶、max_distance=128）：

```text
真实距离: [-10, -5, -1, 0, 1, 5, 10, 50, 100]
映射桶号: [  8,  5,  1, 0, 17, 21, 24, 29, 31]
```

双向注意力把桶一分为二：负向（左边）在 0-15、正向从 16 起，`-1→1`、`+1→17`，「左/右」被区分；近距离每个距离独立成桶，远距离 50 与 100 相差一倍、桶号却只差 2——对数压缩的效果。性能与效率在此平衡。

## 3. Hugging Face：AI 时代的基础设施

BERT/GPT/T5 的实战都离不开 **Transformers** 库。Hugging Face 已从一家 NLP 初创公司演变为覆盖「数据获取 → 模型开发 → 训练微调 → 评估部署」全生命周期的生态：

**协作平台层**——**HF Hub 是 AI 领域的 GitHub**（支持 Git 版本控制）：**Models** 托管数十万预训练模型（NLP/CV/Audio 多模态，可下载可在线体验）；**Datasets** 托管公开数据集（预览 + 高效流式传输）；**Spaces** 用 Gradio/Streamlit/Docker 快速搭建 Web 演示。

<div align="center">
  <img src="{{IMG:5_4_1.png}}" width="90%" alt="Hugging Face 官网" />
  <p>图 3：Hugging Face 官网</p>
</div>

**核心库层**——**Transformers**（生态引擎：统一 API 下载、加载、使用预训练模型）、**Tokenizers**（文本与模型的桥梁：底层 **Rust** 实现，Fast Tokenizers 速度极致）、**Datasets**（数据加速器：标准化加载接口 + 内存映射）。

**辅助库层**——**Accelerate**（简化分布式：同一套代码跑 CPU/单卡/多卡/TPU，免手写分布式逻辑）、**Evaluate**（标准化评估：Accuracy/F1/BLEU/ROUGE 数十种指标）、**Diffusers**（Stable Diffusion 等生成式 AI）、**PEFT**（LoRA 等参数高效微调）。

## 4. Transformers 核心用法

### 4.1 Pipeline：开箱即用

**Pipeline** 把「预处理 → 推理 → 后处理」封装成黑盒，两种用法：

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")          # 只指定任务，自动下载该任务默认模型
classifier("I love Hugging Face!")

vision_classifier = pipeline(model="apple/mobilevit-small")   # 跨模态：图像分类
vision_classifier("https://.../coco_sample.png")
```

「自动导航」无需指定模型只指定任务；默认模型通常为英文，中文任务显式指定中文模型。Transformers 早已突破 NLP 边界，CV、Audio 皆可同一 API 调用。

### 4.2 AutoClass 与三段式流程

**`AutoClass`**（`AutoTokenizer`/`AutoModel`/`AutoConfig`）是库设计精髓：**按 Checkpoint 名称自动推断并加载正确的模型架构**——`AutoModel.from_pretrained("bert-base-chinese")` 会自动识别 BERT 架构并实例化 `BertModel`。`from_pretrained` 支持 Hub 在线下载或本地目录，`save_pretrained` 保存权重、配置和词表。

核心处理流程三个阶段，以 `bert-base-chinese` + `AutoModelForSequenceClassification` 走一遍「文本 → Tensor → Logits → 概率」：

1. **Tokenizer**：`tokenizer(text, return_tensors="pt")` → `input_ids` 如 `[101, 100, 100, 6375, 100, 1359, 2533, 5042, 1296, 102]`——`101`=`[CLS]`、`102`=`[SEP]`、`6375`=「让」；英文 `Hugging/Face/NLP` 因中文词表未收录全变 `100`(`[UNK]`)。Rust Fast Tokenizer 比纯 Python 快几个数量级；
2. **Model**：`outputs = model(**inputs)` → `logits` 如 `[[0.2436, 0.1108]]`——**未归一化**得分，`[1,2]` 对应二分类两类原始分（演示前向流程；实际须微调后两个维度才有具体类别语义）；
3. **Post-processing**：`softmax(logits, dim=-1)` → `[[0.5306, 0.4694]]`，转成和为 1 的概率分布。

## 5. Datasets：数据流水线

清洗逻辑仍要按业务定制，但 `datasets` 解决了加载、内存管理、并行的底层工程问题：

- **内存映射**：底层 **Apache Arrow** 格式，不把整个数据集读进 RAM 即可处理——TB 级数据也能在普通笔记本上跑；还支持**流式模式**，按需迭代无需全量下载。

```python
from datasets import load_dataset
dataset = load_dataset("rotten_tomatoes")
# DatasetDict: train 8530 / validation 1066 / test 1066, features: ['text','label']
```

- **并行预处理**：`map` 把预处理函数应用到每个样本；`batched=True` 一次处理一批吃满 Tokenizer 多线程，`num_proc=4` 多进程显著加速大数据集：

```python
tokenized_datasets = dataset.map(lambda ex: tokenizer(ex["text"], padding="max_length", truncation=True),
                                 batched=True, num_proc=4)
```

## 6. Trainer 与 Evaluate：训练三步走

手写 PyTorch 循环当然可以，但 `Trainer` 集成了混合精度、梯度累积、分布式支持等工程化特性（底层自动调用 Accelerate）。标准流程：

1. **准备组件**：Model + 分词后的 Dataset——`AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)`；打印结构可见末尾 `classifier: Linear(768, 2)` 与 `num_labels` 对应，说明已正确初始化为二分类任务（DistilBERT 本体为 6 层 TransformerBlock，另加 `pre_classifier`）；
2. **配置参数**：`TrainingArguments(output_dir="test_trainer", eval_strategy="epoch", num_train_epochs=1)`；
3. **启动训练**：`Trainer(model=..., args=..., train_dataset=..., eval_dataset=...).train()`。

**Evaluate** 与 Trainer 无缝集成——定义 `compute_metrics` 注入即可，训练中每个评估节点自动计算验证集指标：

```python
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

## 7. 内容小结

- **BERT（理解）**：Transformer Encoder + MLM（及 NSP）学双向上下文表示，分类/序列标注等 NLU 任务的经典方案；
- **GPT（生成）**：Transformer Decoder + 自回归预测下一词，NLG 之外经 GPT-3 系统性展示提示工程与上下文学习的潜力；
- **T5（统一）**：回归 Encoder-Decoder，Text-to-Text 万物皆文本——翻译、分类、回归统一成文本生成，一个模型解多种任务；
- **Hugging Face（工程）**：Transformers/Tokenizers/Datasets/Evaluate 四大件，以极低代码量复现和应用现代 NLP 模型。

> **补记**：理论篇到此形成一条完整闭环——积木（Transformer）→ 三条路线（BERT/GPT/T5）→ 工具链（HF）。三条路线的分野可以压缩成三个问题：预训练任务「看得到什么」（双向掩码/单向续写/片段重构）、「输出是什么」（表示/词/文本到文本）、以及「怎么编程它」（加结构头/加 Prompt/加前缀）。下一章起进入工程深水区：亲手从零搭一个 Llama2。

## 练习

- 把此前基于 Word2Vec/LSTM 的文本分类模型迁移为 BERT 微调，对比模型构建、训练流程与最终性能（可参考 chapter7《微调 BERT 模型进行文本分类》）。

## 参考文献

- Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5)
- Xue, L. et al. (2020). *mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer*
- Hugging Face. *Transformers / Tokenizers / Datasets / Evaluate Documentation*

## 写在最后

调用别人训好的模型终归隔了一层——下一篇[手搓Llama2大模型](9-手搓Llama2大模型)从零搭建类 LLaMA 结构，把这些架构知识落成自己的代码。
