# 预训练模型BERT与GPT

## 1. 同一片架构土壤，两条路线

Transformer 催生的预训练模型往往只取架构的一部分：**GPT 采用解码器**，任务是预测下一个词元，天然适合生成；**BERT（Bidirectional Encoder Representations from Transformers）完全基于编码器**，核心优势是**双向性**——自注意力同时关注上下文，在理解深层语义上取得突破。这一设计也决定了 BERT 是强大的**语言理解**模型，但不改造结构就**不适用于生成**：它预测时能「看到」整个输入序列，而生成模型必须严格遵守「只能看到过去」。

## 2. BERT：动态词向量与预训练-微调范式

Word2Vec 给每个词一个固定向量，解决不了一词多义——「破防」在「我出了一件**破防**装备」和「NLP 算法给我学**破防**了」中含义完全不同，向量却相同。BERT 的设计初衷正是生成**动态的、与上下文相关的词向量**；它不只是词向量工具，更是**预训练语言模型**，工作范式分两阶段：

<div align="center">
  <img src="{{IMG:5_1_1.png}}" alt="BERT 的预训练与微调流程" />
  <p>图 1：BERT 的预训练与微调流程</p>
</div>

1. **预训练**：在庞大通用语料（维基百科、书籍）上以特定无监督任务训练深度网络——目标不是完成某个具体任务，而是学习语言本身的规律（语法结构、语义关系、上下文依赖），得到参数训练好的**预训练模型**；
2. **微调**：面对下游任务（分类、NER）不再从头训练，而是把预训练模型作**基础结构**、其参数作**初始值**，在其上加一个小的任务输出层，在自己的任务数据上训练（整个模型或仅顶部输出层）。因为模型已具备强大语言理解力，微调通常很快、所需数据也少。

「预训练 + 微调」是**迁移学习**的一种实现：把海量数据学到的通用语言知识迁移到数据有限的特定任务。

> **Bi-LSTM 也能做动态词向量，为什么还需要 BERT？** Bi-LSTM 的「双向」本质是两个独立单向 RNN 的**浅层拼接**——正向 RNN 不知道未来信息、反向 RNN 不知道过去信息。BERT 基于自注意力实现**真正的「深度双向」**：每一层计算任何一个词的表示时，都能**同时与序列中所有其他词直接交互**。全局视野让 BERT 能捕捉比 RNN 更复杂、更长距离的依赖，并行计算也远比串行高效。可以说 BERT 实现了 RNN 想达到但未能完美实现的目标。

## 3. BERT 架构详解

### 3.1 模型规模

与 Transformer 原论文 N=6 不同，BERT 提供多种规模适配不同算力与性能需求：

| 模型 | 层数 L | 隐藏层大小 H | 注意力头数 A | 总参数量 |
|---|---|---|---|---|
| `BERT-Base` | 12 | 768 | 12 | ~1.1 亿 |
| `BERT-Large` | 24 | 1024 | 16 | ~3.4 亿 |

加深加宽的网络让 BERT 拥有比原始 Transformer 更强的特征提取与表示能力。

### 3.2 输入表示：三种嵌入相加

$$ Input_{embedding} = Token_{embedding} + Position_{embedding} + Segment_{embedding} $$

<div align="center">
  <img src="{{IMG:5_1_2.svg}}" alt="BERT 输入表示" />
  <p>图 2：BERT 输入表示</p>
</div>

- **词元嵌入**：每个词元自身的向量。BERT 用 **WordPiece** 分词，把不常见词拆成更小子词单元（"studying" → "study" + "##ing"），有效处理未登录词。`bert-base-chinese` 的词表**以单字为主、含少量常用词**，处理中文时效果接近按字分词，但准确说法是**子词切分**。
- **片段嵌入**：为句子对任务（如判断两句是否连续）引入，区分输入中的不同句子——句子 A 的所有词元加同一个「A 嵌入」，句子 B 加另一个「B 嵌入」。
- **位置嵌入**：自注意力不含位置信息，必须额外注入。与原始 Transformer 的固定正余弦不同，BERT 用**可学习的位置嵌入**：建 `[max_position_embeddings, hidden_size]` 的嵌入表，预训练中自己学每个位置的最佳向量。**BERT 的最大长度限制正来自这张表**——`BERT-Base` 通常是 `[512, 768]`，故最多处理 512 个词元；这不是 Transformer 架构的限制，而是 BERT 预训练时的参数设定。

### 3.3 特殊词元

| 特殊 Token | 全称 | 说明 |
|---|---|---|
| `[CLS]` | Classification | 加在**每个输入序列开头**，其最终输出向量被视为整个序列的**聚合表示**，用于文本分类等句子级任务 |
| `[SEP]` | Separator | **分隔句子**：单句加句末，句子对加在两句之间及序列末尾 |
| `[MASK]` | Mask | 仅**预训练阶段**使用，遮盖词元，是 MLM 任务核心 |
| `[PAD]` | Padding | 补齐等长，计算注意力时被 `Attention Mask` 屏蔽 |
| `[UNK]` | Unknown | 词表中不存在的词/字符被替换为它 |

## 4. 两个预训练任务

<div align="center">
  <img src="{{IMG:5_1_3.svg}}" alt="BERT 预训练任务图示" />
  <p>图 3：BERT 预训练任务 MLM 与 NSP</p>
</div>

### 4.1 MLM：完形填空

传统语言模型（如 GPT）单向，只能根据前文预测下一个词。想利用双向上下文、简单地把左右都作输入预测当前词，会导致多层网络中模型「间接看到自己」，预测任务毫无意义。MLM 的思路：**随机遮盖一部分词元，训练模型根据上下文预测被遮盖的词**——像做完形填空，迫使模型学习词元间深层语义与句法。执行策略：

1. **随机选择** 15% 词元作预测目标；
2. 对被选中词元用 **80/10/10 替换**（以 `My son is a good stu` 的 `son` 为例）：
   - **80%** 替换为 `[MASK]`（`My [MASK] is a good stu`）——最核心操作，强制利用上下文预测「挖空」词；
   - **10%** 替换为随机词元（`My apple is a good stu`）——引入噪声：模型不仅要理解上下文还要识别纠正错误词，增强鲁棒性；也促使模型学习每个输入词元的分布式表示，而非仅仅「依赖」`[MASK]` 触发预测；
   - **10%** 保持不变（`My son is a good stu`）——让模型对真实词元也预测自己，更好学习每个真实词元的上下文表示，减轻预训练（有 `[MASK]`）与微调（无 `[MASK]`）的**数据不匹配**。

MLM 由此促使 BERT 学到**动态的、与上下文深度融合的词向量**。

> **MLM 的局限与 WWM**：随机 Mask 单个字/子词可能割裂完整词语的内部语义——「菩提老祖」可能只遮「菩」或「提」，无法在词层面学习。后续 **WWM（Whole Word Masking）**：一词 parts 之一下被选中则整词一起 Mask，中英文任务上均有提升。

### 4.2 NSP：下一句预测

目标是理解句间逻辑关系。模型接收句子对 (A, B)，判断 B 是否为 A 在原文中的下一句：50% 正例、50% 随机负例，本质二分类。预测通过 `[CLS]` 的输出完成——其最终隐藏状态送入二分类器判 `IsNext`/`NotNext`，从而训练 `[CLS]` 学习**句子级聚合特征**。

> **NSP 的有效性讨论**：RoBERTa、ALBERT 等后续研究质疑 NSP——更大规模预训练下移除它或换句子顺序预测等任务效果更好。但 BERT 原论文的消融实验证明：在当时的训练设置下，移除 NSP 会让 QNLI、MNLI 等任务性能明显下降——它确实帮助原始 BERT 学到了句子级关系，特定场景仍有价值。

## 5. 微调下游任务与实践技巧

- **文本分类**：利用 `[CLS]` 聚合表示——格式化输入（加 `[CLS]`/`[SEP]`）→ 提取 `[CLS]` 最终输出向量 → 上加全连接分类器（输出维度=类别数）→ 任务数据训练，同时以**较小学习率**微调 BERT 参数。
- **词元分类**（NER/分词/词性标注）：提取**所有词元**的最终输出序列（`(batch, seq_len, hidden)`）→ 上加全连接层（时间维共享权重）把每个 `hidden_size` 向量映射到类别维度 → 端到端训练微调。
- **其他**：问答可把问题与段落作两个句子输入，预测答案在段落中的起止位置。

两点直接关系能否正常运行的细节：

1. **最大长度**：标准模型（如 `bert-base-chinese`）通常 512 token，其中含特殊 Token，单句实际有效长度最多 **510**（句子对则含 3 个特殊 Token 总长不超 512）；
2. **特殊 Token 必须加**：开头 `[CLS]`、结尾 `[SEP]`——Tokenizer 通常自动处理，但手动构建输入或分析数据时务必牢记。

一个进阶技巧：通常用最后一层输出作特征，但 BERT 不同层侧重不同——**底层偏词法语法表层信息，高层偏语义语境深层信息**，NER 等任务把最后几层（如四层）拼接或相加，有时优于单用最后一层。

## 6. BERT 代码实战：加载与特征提取

`pip install transformers` 后，几行代码即可完成「加载 → 编码 → 推理 → 提特征」全流程；国内下载模型可设镜像 `os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'`：

```python
import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-chinese"
texts = ["我来自中国", "我喜欢自然语言处理"]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state            # (batch, seq_len, 768)
sentence_features = last_hidden_state[:, 0, :]           # [CLS] 句子级特征
first_sentence_tokens = last_hidden_state[0, 1:6, :]     # 第一句逐字词元特征
```

关键解析：

- **Tokenizer 输出是一个字典**：`input_ids` 里自动加了 `101`(`[CLS]`)、`102`(`[SEP]`)，短句用 `0`(`[PAD]`) 补齐；`token_type_ids` 就是**片段嵌入**的体现；`attention_mask` 告诉模型哪些是真词元（1）哪些是填充（0）。
- **Attention Mask 如何工作**：mask 中的 `0` 被转成极大负数（如 -10000）加到注意力得分上，Softmax 后这些位置权重趋 0，计算中被忽略。
- **`torch.no_grad()`**：推理禁用梯度计算，省内存提速。
- **特征提取**：`last_hidden_state` 含每个词元最高层向量；句子级取 `[:, 0, :]`；`pooler_output`（若存在）是其再过一层全连接+Tanh（原为 NSP 准备）——用哪个以验证集效果为准，平均池化、拼接最后几层也常见。「我来自中国」被转为 `['[CLS]','我','来','自','中','国','[SEP]']`，切片 `[0, 1:6, :]` 恰好取出五个汉字。

### 6.1 模型结构分解

`print(model)` 输出的树状结构与理论一一对应：

```text
BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(21128, 768, padding_idx=0)   # 词元嵌入：21128=中文词表大小
    (position_embeddings): Embedding(512, 768)                # 可学习位置嵌入：512 即最大长度由来
    (token_type_embeddings): Embedding(2, 768)                # 片段嵌入：区分句子 A/B
    (LayerNorm) (dropout)                                     # 三嵌入相加后归一化+Dropout
  )
  (encoder): BertEncoder(
    (layer): ModuleList((0-11): 12 x BertLayer(
      (attention): ...query/key/value Linear(768,768)...       # W^Q W^K W^V + 输出 dense=W^O
      (intermediate): Linear(768, 3072) + GELU                 # FFN 升维
      (output): Linear(3072, 768) + LayerNorm                  # FFN 降维
    ))
  )
  (pooler): BertPooler(Linear(768,768) + Tanh)                 # 处理 [CLS] 最终输出，预训练供 NSP
)
```

`vocab.txt` 正好 21128 行可自查验证（直接浏览器打开可能导致卡死，别问笔者怎么知道的）。

## 7. GPT：因果语言模型与从微调到提示

GPT（Generative Pre-trained Transformer）完全基于 **Transformer 解码器**，重心在「生成」：根据给定上文以自回归方式预测下一个最可能词元，在写作、对话、续写上无可替代。从 GPT-1 到 ChatGPT 背后的 GPT-3.5/4，OpenAI 沿解码器路线不断刷新认知。

### 7.1 因果语言模型（CLM）

GPT 的成功根植于简洁强大的预训练任务——**预测下一个词**：给定 $x_1...x_t$，最大化下一词元概率，在海量文本上反复「续写」：

$$ P(x_1,...,x_T) = \prod_{t=1}^{T} P(x_t | x_1,...,x_{t-1}) $$

看似简单，却迫使模型学习语法结构、事实知识、上下文逻辑——只有真正「理解」前文才能准确预测后文。预测第 $t$ 词只能看前 $t-1$ 词的单向自回归特性，是 GPT 与 BERT 最根本的区别。

### 7.2 从 GPT-1 到 GPT-3：使用范式的三级跳

1. **GPT-1：有监督微调**。先无监督预训练学通用语言表示，再在任务标注数据上有监督微调。关键创新是**任务特定的输入变换**：不改架构，用特殊分隔符（Start/Delimiter/Extract）把结构化输入转成有序序列——如文本蕴含把前提与假设用 `$` 拼接输入。
2. **GPT-2：零样本学习**。观点：**语言模型即多任务学习者**。规模到 1.5B、数据换成过滤后的 WebText（Reddit 高赞链接），模型展现无需微调的零样本能力——直接输入指令 `Translate English to French: cheese =>` 就能续写 `fromage`，说明预训练中已隐式学会各种任务。架构上将 LayerNorm 移到子层输入端（Pre-activation），上下文 512→1024。
3. **GPT-3：上下文学习（ICL）**。1750 亿参数，定位**少样本学习者**。ICL：推理阶段**不更新任何权重**，仅凭提示中的少量示例完成任务——只给任务描述是 **Zero-shot**、一个示例 **One-shot**、多个示例 **Few-shot**（常显著提升、可达微调效果）。标志转变：不再为每个任务训练专用模型，而是设计 **Prompt** 激发通用大模型能力。

几乎所有 NLP 任务由此统一成「问答/续写」：文本分类旧法是「句子 → `[CLS]` → 分类头 → 标签」，新法是输入 `请判断以下评论的情感类别（正面/负面）："这家餐厅的菜品味道惊艳，服务也很周到。"` 让模型续写「正面」；文本相似度同理，提示两句让模型续写「相似」。

## 8. GPT 架构解析

### 8.1 解码器堆叠：去掉交叉注意力

GPT 本质是 Transformer 解码器模块的堆叠，标准层含两个关键子层：**掩码多头自注意力**（单向自回归的关键——掩码确保任一位置只能关注左侧）与**位置前馈网络**（与编码器完全相同）。原始 Transformer 解码器中连接编码器的**交叉注意力层被移除**——GPT 根本没有编码器。

<div align="center">
  <img src="{{IMG:5_2_1.png}}" width="90%" alt="GPT-1 模型架构与输入变换" />
  <p>图 4：GPT-1 架构（左）与针对不同任务的输入变换（右）</p>
</div>

GPT-1 的架构与微调策略：左侧模型主体微调时除任务分类输出外，还保留语言模型预测作辅助目标——$L = L_{task} + \lambda L_{lm}$，多任务学习既助泛化又加速收敛；右侧是输入变换适配：分类任务加 `Start`/`Extract` 标记；蕴含任务用 `Delim` 拼接；**相似度任务**因 GPT 单向且对句序敏感，把两句话按两种顺序（Text1-Text2 与 Text2-Text1）分别输入、特征相加；多选题则将上下文与每个选项分别拼接独立计算得分。

### 8.2 关键架构演进

- **特殊 Token**：GPT 系列主要用 `<|endoftext|>`——文本结束符、预训练文档分隔符，微调/Few-shot 时也用于分隔示例；
- **归一化位置**：GPT-1 用 Post-Norm（同原 Transformer 与 BERT）；GPT-2/3 转 **Pre-Norm**（`x + SubLayer(LayerNorm(x))`），**缓解**深层梯度消失/爆炸，让 96 层的 GPT-3 训练成为可能；
- **输入表示**：词元嵌入用**字节级 BPE**；位置嵌入始终**可学习**；上下文窗口逐代扩大 512→1024→2048；
- **无片段嵌入**：输入总是一个连续文本流，不区分句子片段，没有 BERT 的 Segment Embeddings。

GPT-3 家族各规模配置（层数/维度增长带来参数量指数级膨胀）：

| Model | 参数 | 层数 | d_model | 头数 | d_head | Batch | LR |
|---|---|---|---|---|---|---|---|
| Small | 125M | 12 | 768 | 12 | 64 | 0.5M | 6.0e-4 |
| Medium | 350M | 24 | 1024 | 16 | 64 | 0.5M | 3.0e-4 |
| Large | 760M | 24 | 1536 | 16 | 96 | 0.5M | 2.5e-4 |
| XL | 1.3B | 24 | 2048 | 24 | 128 | 1M | 2.0e-4 |
| 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 1M | 1.6e-4 |
| 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2M | 1.2e-4 |
| 13B | 13.0B | 40 | 5140 | 40 | 128 | 2M | 1.0e-4 |
| 175B | 175.0B | 96 | 12288 | 96 | 128 | 3.2M | 0.6e-4 |

### 8.3 BERT vs GPT 对照总表

| 特性 | BERT (Encoder) | GPT (Decoder) |
|---|---|---|
| 核心结构 | Transformer 编码器 | Transformer 解码器 |
| 注意力 | 标准自注意力（双向，看全文） | 掩码自注意力（单向，只看左） |
| 预训练任务 | MLM + NSP | 因果语言模型 / 预测下一个词 |
| 输入表示 | 词元 + 位置 + **片段** | 词元 + 位置 |
| 最大长度 | 通常 512 | 1024 (GPT-2) / 2048 (GPT-3) |
| 归一化 | Post-Norm | Pre-Norm (GPT-2/3) |
| 应用模式 | **微调**：加分类头、更新权重 | **提示**：设计 Prompt、不更新权重 |
| 适用场景 | 语言理解 NLU：分类、NER、抽取式问答 | 语言生成 NLG：写作、对话、生成式任务 |

## 9. GPT 代码实战：自回归生成与分词器

### 9.1 手动逐 token 生成（英中对照）

用 `AutoModelForCausalLM` 加载 gpt2——比 `AutoModel` 多一个 `lm_head` 层，把隐藏状态映射回词表预测下一词：

```python
outputs = model(generated_ids)                  # 完整当前序列入模
next_token_logits = outputs.logits[:, -1, :]    # 取最后一个位置=对“下一个未知词”的预测
next_token_id = torch.argmax(next_token_logits, dim=-1)   # 词表维取最大：贪心搜索
generated_ids = torch.cat([generated_ids, next_token_id], dim=1)  # 拼接后再次入模=自回归核心
```

并列跑英文 `"I like eating fried"` 与中文「我喜欢吃炸」得到两个结论。**分词效率**：4 个英文词编码成 4 个 token `['I','Ġlike','Ġeating','Ġfried']`（`Ġ` 前缀表示词开始）；同样 4 个汉字被编成多得多的字节级 token——中文编码效率显著下降。**生成质量**：英文流畅续写 `I like eating fried chicken...`；中文输出一片乱码——模型完全不懂这些字节组合的含义，只能按英文统计规律预测下一字节，续写出符合英文规律却不符合中文编码规则的字节序列。解法是换中文预训练模型（如 `uer/gpt2-chinese-cluecorpussmall`）。

### 9.2 GPT-2 结构分解

```text
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)     # 词元嵌入，50257=GPT-2 词表
    (wpe): Embedding(1024, 768)      # 可学习位置嵌入，决定最大长度 1024；无片段嵌入
    (h): 12 x GPT2Block(
      (ln_1) (attn: c_attn Conv1D(2304,768) + c_proj)   # c_attn 一次映射 768->2304=Q/K/V 拼接
      (ln_2) (mlp: c_fc 768->3072, act, c_proj 3072->768)
    )
    (ln_f): LayerNorm
  )
  (lm_head): Linear(768, 50257, bias=False)   # 每个位置的输出映射到词表
)
```

`Conv1D` 在这里被巧妙地用作全连接层。注意 `lm_head` 与 BERT `pooler` 的对比：pooler 只接收 `[CLS]` 输出做**句子级**二分类；`lm_head` 接收**每个位置**的输出向量映射到 50257 维词表，做**词元级**「预测下一个词」——它是连接模型特征与词元预测的桥梁（手动贪心循环里 `logits[:, -1, :]` 正是它的输出）。

### 9.3 BPE 分词器原理

GPT 的**字节级 BPE**从根本上杜绝 `[UNK]`：基础词汇表含全部单字节（0-255），任何文本都能无损表示为字节序列；训练时贪心、迭代地合并最高频相邻字节对形成更长子词入表（`t`+`h`→`th`→`the`）；编码时先转字节序列，再用表中最长子词贪心替换。生僻字「龘」BERT 分词器可能直接判 `[UNK]`，GPT-2 会拆成 UTF-8 字节 token。代价是**对非核心语言效率低**：`gpt2` 词表为英文构建、不含中文词，一个汉字拆 2-3 个字节 token——英文提示 10 个 token 的内容中文要 19 个，既拉长序列又增加语义学习难度、消耗上下文预算。

### 9.4 pipeline：一行搞定

理解原理之外，实际用 `transformers` 的高阶封装即可：

```python
from transformers import pipeline
generator = pipeline("text-generation", model="gpt2", device=device)
generator("I like eating fried", max_new_tokens=5, num_return_sequences=1)
```

分词、推理、解码全部自动处理，结果与手动循环一致——底层执行相似逻辑，只是接口极度简化。

> **补记**：把 BERT 与 GPT 的差异再压一层——两者真正的分野只在「预测任务看得到什么」：MLM 允许看双向，代价是预训练与微调的输入分布不一致（80/10/10 就是在补这个洞）、且不生成；CLM 只看左侧，训练即推理、天然可续写，但双向理解要靠架构外的手段。后续「BERT 还是 GPT」的路线之争，本质是这个可见性约束之争；而 GPT-2/GPT-3 逐级展现的「零样本/上下文学习」说明：当模型足够大，理解可以被生成吸收，路线天平最终倒向了解码器。

## 练习

- 总结 BERT 和 Transformer（Encoder）的区别。

## 参考文献

- Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805
- Wu, Y. et al. (2016). *Google's Neural Machine Translation System*（WordPiece）
- Cui, Y. et al. (2019). *Pre-training with Whole Word Masking for Chinese BERT*
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
- Lan, Z. et al. (2019). *ALBERT: A Lite BERT for Self-supervised Learning*
- Radford, A. et al. (2018). *Improving Language Understanding by Generative Pre-Training*（GPT-1）
- Radford, A. et al. (2019). *Language Models are Unsupervised Multitask Learners*（GPT-2）
- Brown, T. et al. (2020). *Language Models are Few-Shot Learners*（GPT-3）
- Sennrich, R. et al. (2015). *Neural Machine Translation of Rare Words with Subword Units*（BPE）
- Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*

## 写在最后

两条路线的源头是「预训练语料从哪来、任务怎么自举」——下一篇[T5与HuggingFace生态](8-T5与HuggingFace生态)记录把一切任务统一成文本到文本的 T5，以及撑起整个预训练生态的 Hugging Face 工具箱。
