# NER项目全流程

## 1. 命名实体识别概要

文本分类是对文本的**宏观**理解，NER 则走向**微观**：从非结构化文本中精准定位并分类出业务关心的实体（人名、地名、机构、疾病、药物……）。它是信息抽取、知识图谱构建、智能问答等应用的**关键环节**。

常见实体类型：人名 PER、地名 LOC、组织机构 ORG、产品 PROD、作品 WORK、时间 TIME、其他 MISC（含义依数据集而定）。**一个词算不算实体、属于哪类，完全由业务场景决定**——「苹果」在通用场景是水果，在数码讨论里可能是品牌/机构。不同数据集标签集差异也大（CoNLL 2003 仅 PER/ORG/LOC/MISC；OntoNotes 5 则有 PERSON、GPE、ORG、FAC、PRODUCT、EVENT、WORK_OF_ART 等更细类别），实际项目应先明确标签集合。

**价值**：抓住「谁、在哪、做了什么」。知识图谱构建（抽实体及关系第一步）、信息抽取（无结构转结构化）、搜索优化（识别 query 中的实体给结构化结果）、问答/对话（抽关键实体作回答依据）。医疗场景可从病历文献中抽「II型糖尿病」（疾病）、「多饮多食」（症状）、「血糖检测」（检查）、「二甲双胍口服」（治疗），进而构建医疗知识图谱。

两个典型流程：**智能搜索**——Query 理解分两步（意图识别 → 按意图有针对性抽实体，如属性查询抽主体+属性、汇率转换抽金额+源/目标币种、单位换算抽数值+源/目标单位）→ 信息检索 → 排序呈现；**智能客服/辅助诊断**——电商里「快递到哪了/查订单/发货了吗」都归到「物流信息」意图实体；诊断系统里 CV 处理影像、NLP 用 NER 抽症状与病史再配合关系抽取形成 `[症状 A,B] => 建议检查[X]` → `[X 结果] => 结论[Y]` → `[症状,结论] => 高概率诊断[Z]` 的推理链。

### 1.1 数据标注

数据质量决定模型上限。标注方式按成本/效率权衡：

- **人工标注**：质量高但贵、慢；Excel 或专门前端平台皆可；
- **LLM 辅助标注**：先让大模型预标注再人工校对，人从「从零创造」变为「审核修正」；LLM 输出不稳定，仍需人工保底；
- **半监督/迭代式**：人工标一小批训「学生」模型 → 预测大量未标注数据 → 人工修正（比从零标快得多）→ 修正数据加入训练集再训更强学生，循环迭代。

### 1.2 实现方法谱系

**字典与规则匹配**：维护实体词典做字符串匹配。简单、快，封闭领域准确率可能很高；但泛化差（认不出词典外新词）、规则维护成本极高。

**序列标注**（当前常见做法）：把 NER 变成给每个 token 打标签的问题。

- **Token 级标签预测（BMES/BIO）**：`B/M/E/S` 或 `B/I/O` 体系。例：「西安的大雁塔门票多少钱」→ BMES 标注 `B-LOC E-LOC O B-LOC M-LOC E-LOC B-ATTR E-ATTR O O O`。模型结构 `Token Embedding → 序列模型 → Token 分类`；分类层可用 Softmax（各类独立）或 **CRF**（额外学标签转移概率、全局最优解码——对 RNN/CNN 编码器有效，但在 BERT 等强预训练编码器上收益常变小，是否用以实验为准）。**缺点：解决不了嵌套实体**——「他就读于北京大学」中「北京」(LOC) 与「北京大学」(ORG) 是包含关系，单标签无法同时表达。
- **指针网络与片段网络**（解决嵌套）：**片段网络**枚举所有连续片段再逐个分类（「美式咖啡」产生 10 个候选片段），理论上可解嵌套，但候选数随长度平方增长、正负样本严重失衡；**指针网络**为每个 token 训多个二分类器判断「是否某类实体开头/结尾」，再按类型配对生成候选（`星`+`克`→星巴克，`美`+`啡`→美式咖啡，天然支持嵌套）；两者结合即**先指针定位边界、配对成候选、再片段分类**，既解嵌套又降低计算量。
- **基于分词的分类**：先分词再对词分类。效果高度依赖分词质量，「大雁塔」错分成「大雁+塔」就全盘皆输。

**生成式模型**：把 NER 统一成 Text-to-Text 任务，用 Prompt 让模型直接生成结构化实体。三种形态——标准 Encoder-Decoder（Decoder 生成 `地点实体:西安[SEP]大雁塔[SEP];属性实体:门票`）、带「实体清单」提示约束输出空间、续写式 Decoder-Only（输入输出拼成一条串，**只在「|」之后计算损失**，推理时只给前半段让模型续写）。优点是框架统一、Few/Zero-shot 省标注、天然支持嵌套等复杂结构；缺点是输出格式不稳定需后处理、有幻觉（可能生成原文没有的实体）。

### 1.3 项目实现思路

沿用「数据处理 → 模型构建 → 训练评估持久化」流程：

- **数据处理**：分词/分 token → Token-ID 映射（词表）→ 标签-ID 映射；数据增强可做**实体替换**（同类型词典随机替换，如北京→深圳，教模型看语境而非背词）、**引入噪声**（同音字替换模拟输入错误）、**随机遮盖**（非实体 token 换 `<UNK>`，逼模型依赖上下文）、**拼音替换**（模拟弱规范输入）。强约束领域（如医疗）做实体替换须保证不破坏实体间真实语义关系（疾病-症状-药物搭配约束）。
- **模型构建**：经典组合 `Embedding + 动态编码器（BERT/Bi-LSTM 等）+ Token 分类层（Softmax/CRF）`；输入 `[N, T]` 的 token ID，输出 `[N, T, num_classes]`。迁移学习四种微调策略：冻结参数、同等处理、**差分学习率**（迁移参数小学习率、新增参数正常学习率）、分层/部分冻结。
- **训练评估**：Token 级指标用于监控，**实体级 P/R/F1 是核心**；标签序列→实体片段的解码按 BMES/BIO 规则扫描（B 开片段、M/I 续、E 结束、S 单字、O 断）；持久化按实体级 F1 决定是否保存。

## 2. 数据处理：CMeEE-V2 实战

目标是产出：模型输入 `token_ids [B, T]`、标签 `label_ids [B, T]`，以及两张映射表（字符→ID、标签→ID）。

### 2.1 数据格式

CMeEE-V2（中文医学实体抽取）是 JSON 数组，每条形如：

```json
{"text": "（2）室上性心动过速可用常规抗心律失常药物控制，年龄小于5岁。",
 "entities": [{"start_idx": 3, "end_idx": 9, "type": "dis", "entity": "室上性心动过速"},
              {"start_idx": 14, "end_idx": 20, "type": "dru", "entity": "抗心律失常药物"}]}
```

**索引包含性**是坑：本数据 `start_idx` 与 `end_idx` 均为**闭区间**，实体应为 `text[start:end+1]`，长度 `end-start+1`。`entities` 的 `entity` 字段可用于校验。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_8_2_1.png" alt="PyCharm 调试器观察数据结构" />
  <p>图 1：用调试器观察数据结构（断点→Debug→Step Over→展开变量）</p>
</div>

> 先断点看清 `all_data` 的结构（list → dict → `text`/`entities` 两键）再写遍历，是通用的调试方法：哪里不懂就在哪里下断点、Step Over 后展开变量看。

### 2.2 构建 BMES 标签映射

从 train+dev 两个文件收集全部实体类型（本例 9 类：`dru/dep/dis/bod/mic/equ/sym/pro/ite`），**排序**保证每次生成一致，然后按 `O=0`、每类依次 `B-/M-/E-/S-` 递增赋 ID：

```python
tag_to_id = {'O': 0}
for entity_type in sorted(list(all_entity_types)):
    for prefix in ['B', 'M', 'E', 'S']:
        tag_to_id[f"{prefix}-{entity_type}"] = len(tag_to_id)
# 9 类 → 1 + 9*4 = 37 个标签，保存为 categories.json
```

### 2.3 构建字符词汇表

```python
def create_char_vocab(data_files, output_file, min_freq=1):
    char_counts = Counter()
    for path in data_files:
        for data in load_json(path):
            char_counts.update(list(normalize_text(data['text'])))   # 先规范化
    frequent = sorted([c for c, n in char_counts.items() if n >= min_freq])
    save_json(["<PAD>", "<UNK>"] + frequent, output_file)
```

**文本规范化**是必要一步：数据里全角 `，`、`（` 与半角 `,`、`(` 语义相同却算两个 token，会无谓膨胀词表。用 `str.maketrans` 把所有全角字符转半角：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_8_2_2.png" alt="全角/半角字符混用" />
  <p>图 2：字符频率统计中的全角/半角混用</p>
</div>

```python
def normalize_text(text):
    full_width = "０１２３…（全角字符表）"
    half_width = r"0123...（对应半角表）"
    return text.translate(str.maketrans(full_width, half_width))
```

> 注意：规范化后**位置索引不变**（一对一替换），所以实体 `start/end` 仍可直接用于切片，不必重新对齐。

### 2.4 Dataset 与 DataLoader

`NerDataset.__getitem__` 把单条样本变成 `(token_ids, label_ids)`：先规范化+切字，再默认整条打 `O`，然后遍历实体按闭区间写 BMES：

```python
tags = ['O'] * len(tokens)
for e in record.get('entities', []):
    t, s, eidx = e['type'], e['start_idx'], e['end_idx']
    if eidx >= len(tokens): continue
    if s == eidx:
        tags[s] = f'S-{t}'
    else:
        tags[s] = f'B-{t}'; tags[eidx] = f'E-{t}'
        for i in range(s + 1, eidx): tags[i] = f'M-{t}'
label_ids = [self.tag_to_id.get(tag, self.tag_to_id['O']) for tag in tags]
```

`collate_batch` 动态填充并生成 attention_mask；**标签的填充值是 `-100`**：

```python
padded_token_ids = pad_sequence(..., batch_first=True, padding_value=vocab.pad_id)
padded_label_ids = pad_sequence(..., batch_first=True, padding_value=-100)
attention_mask   = (padded_token_ids != vocab.pad_id).long()
```

> **为什么是 -100**：PyTorch 的 `CrossEntropyLoss` 有 `ignore_index` 参数，默认值恰为 `-100`；标签为该值的位置自动不参与损失与梯度。用 0 则会把「填充」当合法类别 `O` 学。

## 3. 模型构建：从单向 GRU 到 BiGRU

NER 模型三件套：**Token Embedding**（静态词向量）→ **动态特征提取**（Bi-GRU 或 BERT，融合上下文）→ **分类决策**（Linear 映射到 `num_tags`）。本质是逐 token 的分类模型。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_8_3_1.png" alt="数据加载器输出示例" />
  <p>图 3：Batch 内的 token_ids / attention_mask / label_ids（-100 标记忽略位）</p>
</div>

### 3.1 单向 GRU 基线：`nn.ModuleList` 堆叠 + 残差

先看一个设计选择：容器用 `nn.ModuleList` 还是 `nn.Sequential`？

- **Sequential**：像自动化流水线，数据按序流过每层，适合简单线性堆叠，**无法在层间插入自定义逻辑**；
- **ModuleList**：只是「存模块的列表」，不自动执行，你在 `forward` 里手写循环，于是可以**在层与层之间加残差连接**。

```python
class GRUNerNetWork(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_tags, num_gru_layers=1):
        self.embedding = nn.Embedding(vocab_size, hidden_size)   # embed_dim == hidden_size，便于残差相加
        self.gru_layers = nn.ModuleList([
            nn.GRU(hidden_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False)
            for _ in range(num_gru_layers)])
        self.classifier = nn.Linear(hidden_size, num_tags)
    def forward(self, token_ids, attention_mask=None):
        current = self.embedding(token_ids)
        for gru in self.gru_layers:
            out, _ = gru(current)
            current = out + current        # 残差
        return self.classifier(current)
```

**损失与形状**：`CrossEntropyLoss(ignore_index=-100, reduction='none')` 要求类别维在前，需 `logits.permute(0, 2, 1)`；实测 loss 形状 `[2, 8]`，`-100` 位置损失为 0——`ignore_index` 生效。

> **一个「侥幸」**：单向 GRU 的 `forward` 没用 `attention_mask` 也正常，原因是单向从左到右计算，末尾 Padding **不影响**前面真实 token 的特征；填充位产生的 logits 又被 `-100` 在损失里屏蔽。但**这个侥幸在双向模型里不复存在**——反向路径会从序列末尾开始，Padding 会作为初始状态污染真实 token 的表示。

### 3.2 双向 GRU：必须 pack

双向不能简单设 `bidirectional=True` 了事，得让 RNN「知道」每个序列的真实长度。PyTorch 的工具体系是：

- `pack_padded_sequence(input, lengths, batch_first, enforce_sorted=False)`：把填充批次压成紧凑的 `PackedSequence`（填充位被暂时移除），`enforce_sorted=False` 让函数内部自动排序；
- `pad_packed_sequence(packed, batch_first=True, total_length=token_ids.shape[1])`：计算完再解包还原成规整张量，`total_length` 保证长度与原始输入一致。

```python
class BiGRUNerNetWork(nn.Module):
    def __init__(...):
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru_layers = nn.ModuleList([nn.GRU(hidden_size, hidden_size, num_layers=1,
                                                batch_first=True, bidirectional=True) ...])
        self.fc = nn.Linear(hidden_size * 2, hidden_size)   # 融合双向特征，output 维度 2H → H
        self.classifier = nn.Linear(hidden_size, num_tags)

    def forward(self, token_ids, attention_mask):
        lengths = attention_mask.sum(dim=1).cpu()                    # 真实长度
        packed = rnn.pack_padded_sequence(self.embedding(token_ids), lengths,
                                          batch_first=True, enforce_sorted=False)
        for gru in self.gru_layers:
            packed_out, _ = gru(packed)
            out, _ = rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=token_ids.shape[1])
            features = self.fc(out)                                   # 双向融合回 H
            inp, _ = rnn.pad_packed_sequence(packed, batch_first=True, total_length=token_ids.shape[1])
            current = features + inp                                  # 残差
            packed = rnn.pack_padded_sequence(current, lengths, batch_first=True, enforce_sorted=False)
        final, _ = rnn.pad_packed_sequence(packed, batch_first=True, total_length=token_ids.shape[1])
        return self.classifier(final)
```

要点：pack 后**每层都要解包才能做残差**，做完再重新 pack 给下一层；`total_length` 保证解包后长度与原始输入对齐。这样才算健壮处理变长序列的双向模型。

## 4. 组件化训练框架

设计思路是**组件式组装 + 部门化分工**：`Trainer` 只负责训练/评估循环，不关心模型与数据怎么来；模型、优化器、DataLoader、分词器、评估函数全在外部创建再「注入」。这样换模型（BiGRU→BERT）不用改 `Trainer`。

### 4.1 用 dataclass 管配置

```python
@dataclass
class NerConfig:
    data_dir: str = "data"; train_file: str = "CMeEE-V2_train.json"; dev_file: str = "CMeEE-V2_dev.json"
    vocab_file: str = "vocabulary.json"; tags_file: str = "categories.json"; output_dir: str = "output"
    batch_size: int = 32; epochs: int = 20; learning_rate: float = 1e-3
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    hidden_size: int = 256; num_gru_layers: int = 2
```

`@dataclass` 自动生成 `__init__` 与友好的 `__repr__`，`print(config)` 即可看清全部参数。

### 4.2 Trainer 的拆分粒度

`fit` 调度 → `_train_one_epoch` → `_train_step`（前向/损失/反向）→ `_evaluate` → `_evaluation_step` → `_save_checkpoint`。`_train_step` 核心：

```python
batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
logits = self.model(token_ids=batch['token_ids'], attention_mask=batch['attention_mask'])
loss = self.loss_fn(logits.permute(0, 2, 1), batch['label_ids'])   # [B,C,L] 对 [B,L]
self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
```

`_evaluate` 在 `torch.no_grad()` 下累计 loss，并收集 `all_logits / all_labels / all_attention_mask` 交给 `eval_metric_fn` 算实体级指标。

### 4.3 抽象基类统一接口

`BaseNerNetwork(nn.Module, ABC)` 用 `@abstractmethod` 强制 `forward(token_ids, attention_mask)` 签名，保证未来任何新模型都能无缝接入 `Trainer`、接口一目了然；分词器同理有 `BaseTokenizer`（`text_to_tokens` / `tokens_to_ids` / `encode` / `get_pad_id`）与 `CharTokenizer` 实现。

### 4.4 实体级评估

Token 准确率会骗人——绝大多数 token 是 `O`，全预测 `O` 也能很高分，但一个实体都没抽出来。所以核心指标是**实体级 P/R/F1**：解码成实体集合后算 TP/FP/FN。

```python
def _trans_entity2tuple(label_ids, id2tag):
    """严格 BMES 解码：仅遇 E-/S- 落盘，B-/O 不闭合未完成片段"""
    entities, cur = [], None
    for i, lid in enumerate(label_ids):
        tag = id2tag.get(lid.item(), 'O')
        if tag.startswith('B-'):
            cur = (tag[2:], i, i + 1)
        elif tag.startswith('M-'):
            cur = (cur[0], cur[1], i + 1) if cur and cur[0] == tag[2:] else None
        elif tag.startswith('E-'):
            if cur and cur[0] == tag[2:]:
                entities.append((cur[0], cur[1], i + 1))
            cur = None
        elif tag.startswith('S-'):
            entities.append((tag[2:], i, i + 1)); cur = None
        else:
            cur = None
    return set(entities)
```

两条批量评估的坑与解法：

1. **处理填充**：用 `attention_mask` 过滤掉填充位（`row = labels[b][mask.bool()]`），只对有效片段解码；
2. **追踪样本来源**：同一批次的多条样本可能在相同位置有同类型实体，直接存 `set` 会把它们误判为同一实体。解法是给每个实体加上**样本序号前缀**：`(sample_idx, type, start, end)` 四元组唯一标识。

```python
true_entities.update({(sample_idx,) + e for e in te})
```

P/R/F1 计算含零保护：`precision = TP/(TP+FP)`、`recall = TP/(TP+FN)`、`f1 = 2PR/(P+R)`。

## 5. 推理：从标签序列到结构化实体

### 5.1 解码（状态机式）

模型输出的标签 ID 序列（如 `[0, 9, 10, 11, 0, ...]`）不能直接交付，要用**严格 BMES 解码**转成 `[{"text": "高血压", "type": "dis", "start": 3, "end": 6}]`：

1. `B-`：若已有未闭合实体则**放弃**，新建记录类型/起始；
2. `M-`：类型一致才续接文本，不一致则重置（非法序列）；
3. `E-`：类型匹配则补上末字、记 `end=i+1`、落盘，**必须**重置；
4. `S-`：先弃未闭合，再直接落盘单字实体；
5. `O`：弃未闭合，重置。

**严格模式**保证输出规范（只有 B 没有 E 的片段直接丢弃）；若业务更看重召回，可换**宽松策略**（如 `B-M-O` 保留 `B-M` 作实体），取决于精确率/召回率的取舍。

### 5.2 NerPredictor 封装

一个合格推理器应满足：只给模型目录就能自动加载全部资源、对外只暴露 `predict(text)`、与训练代码解耦。

`__init__` 四步：加载 `config.json` → 按配置路径加载词汇表与标签映射（构建 `id2tag`）→ 初始化 `CharTokenizer` → 按配置重建 `BiGRUNerNetWork` 并 `load_state_dict(torch.load(path, map_location=device, weights_only=True)['model_state_dict'])`、`.to(device)`、`.eval()`（关掉 Dropout，保证确定性）。

> **一个安全细节**：加载别人产出的权重文件时，建议 `torch.load(..., weights_only=True)`（PyTorch 2.6+ 默认即此模式），避免反序列化可执行对象；只加载自己刚训练出的可信文件才考虑其它选项。

> **开发插曲：训练与推理配置必须同步**。写推理器时容易撞上 `FileNotFoundError: output/config.json`——训练脚本没保存配置，推理无从重建模型。修法是回 `05_train.py` 加一步：训练开始前 `save_json(asdict(config), output/config.json)`。

`predict` 三步：预处理（文本→ids→加 batch 维→生成全 1 的 mask→搬设备）→ `torch.no_grad()` 前向得 logits → 后处理（`argmax` 得标签 ID → `id2tag` 映射 → `_extract_entities` 解码）。

```bash
python 06_predict.py --model_dir "output" --text "患者自述发热、咳嗽，伴有轻微头痛。"
# Entities: [{"text":"发","type":"sym",...}, {"text":"咳","type":"sym",...}]  ← 未调优基线，结果不完美属正常
```

## 6. 自定义损失：对抗数据不均衡

`O` 占绝大多数，标准交叉熵会被多数派主导。两种上游策略：

**加权交叉熵**：实体标签权重 10、非实体 1——弄错实体 token 的惩罚是弄错 `O` 的 10 倍，逼模型关注实体。

**硬负样本挖掘**：海量非实体中大部分是模型轻松答对的「简单样本」，价值低；只取损失最大（Top-K）的那部分参与计算，等价于从多数派里筛出「疑难样本」重点学。

```python
class NerLoss(nn.Module):
    def __init__(self, loss_type='cross_entropy', entity_weight=10.0, hard_negative_ratio=0.5, ignore_index=-100):
        self.base_loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
    def _weighted_cross_entropy(self, logits, labels):
        loss_per_token = self.base_loss_fn(logits, labels)           # [B, L]
        em, nm = (labels > 0).float(), (labels == 0).float()          # 实体 / 非实体掩码
        entity_loss = (loss_per_token * em).sum() / (em.sum() + 1e-8)
        non_loss    = (loss_per_token * nm).sum() / (nm.sum() + 1e-8)
        return self.entity_weight * entity_loss + non_loss, entity_loss.detach(), non_loss.detach()
    def _hard_negative_mining(self, logits, labels):
        loss_per_token = self.base_loss_fn(logits, labels)
        em = (labels > 0).float(); nm = (labels == 0).float()
        entity_loss = (loss_per_token * em).sum() / (em.sum() + 1e-8)
        non_flat = (loss_per_token * nm).view(-1)                     # 展平才能全局 topk
        k = min(int(em.sum().item() * self.hard_negative_ratio), int(nm.sum().item()))
        if k == 0:
            hard = torch.tensor(0.0, device=logits.device)
        else:
            hard = torch.mean(torch.topk(non_flat, k=k)[0])
        return self.entity_weight * entity_loss + hard, entity_loss.detach(), hard.detach()
```

两个实现细节不可省：`topk` 的 `k` 不得超过非实体 token 总数（`min` 保护）；二维损失必须先 `view(-1)` 展平，才是「所有非实体样本里损失最大的 k 个」。

损失返回三元组 `(总损, 实体损, 非实体损)`，`Trainer` 相应升级：**只用主损 `loss[0]` 反向传播**、训练日志分项累计、评估 loss 用主损、保存格式改为 `{'model_state_dict': ...}` 以便推理端加载。

> **验证 loss 不降反升是正常的**：评估也用这个**加权**损失，它反映的是「训练目标」而非标准评估指标——实体权重 10 让少数实体错误就能大幅拉动 loss，硬负挖掘又让模型不断面对更难的负样本。所以用高级损失时**别再盯验证 loss**，盯实体级 F1——这也是 `Trainer` 改以 F1 保存最佳模型的原因。

## 7. 训练工作流增强

**TensorBoard 可视化**：封装 `TensorBoardLogger`（`SummaryWriter` 的初始化/记录/关闭），`log_metrics` 按元组（三项 loss）或字典（验证指标）分头写标量；配合 `seed_everything(seed)`（`manual_seed` + `cudnn.deterministic=True/benchmark=False`）保证可复现。

**提前停止**：独立 `EarlyStopping` 回调类，`patience` 轮不提升即 `early_stop=True`；`mode='max'/'min'` 统一成「越大越好」再比较，`delta` 给容忍范围。`Trainer` 每轮用 `current_metric = eval_metrics.get('f1', -loss)` 判断是否触发。

**断点续训**：配置加 `resume_checkpoint`；`Trainer.__init__` 检测到路径就调 `_resume_checkpoint`（恢复模型权重、优化器状态、起始轮 `start_epoch`）；`fit` 循环从 `self.start_epoch` 起；主脚本先用 `os.path.exists` 校验路径，不存在则置 `None` 从零开始，避免直接报错。

## 8. 小结：这套脚手架的价值

- **数据**：CMeEE-V2 解析 → BMES 标签映射（37 类）→ 规范化字符词表 → `-100` 填充的 DataLoader；
- **模型**：Bi-GRU 三件套（Embedding → 双向编码 + pack/pad → Linear）；`ModuleList` 换 `Sequential` 是为了残差；
- **框架**：dataclass 配置 + 基类接口 + 组件注入的 Trainer + 实体级 P/R/F1；
- **推理**：严格 BMES 解码 + 配置同步的 NerPredictor；
- **优化**：加权交叉熵/硬负挖掘、TensorBoard、早停、断点续训。

这套框架本身比基线模型的分数更重要——它把「换模型、换损失、换指标」都变成替换一个组件，后续接 BERT-NER 只需实现一个新的 `BaseNerNetwork` 子类。

> **补记**：这一篇真正值得记住的是四个「反直觉点」：① 标签填充用 `-100` 而不是 0；② 单向 RNN 可以侥幸忽略 mask，双向则必须 pack；③ 评估必须用实体级 F1，Token 准确率会被 `O` 淹没；④ 用加权/硬负损失后验证 loss 失去意义。这四点几乎覆盖了 NER 工程里最容易踩的坑。

## 写在最后

「文本任务实战」两篇到此收尾：分类验证了预训练-微调的价值，NER 搭起了一套模块化序列标注脚手架。下一系列「微调与量化」进入模型训练侧——从 PEFT 与 LoRA 开始，看大模型如何用少量参数完成私有数据适配。
