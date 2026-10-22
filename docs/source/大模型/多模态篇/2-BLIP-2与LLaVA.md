# BLIP-2与LLaVA

## 1. 从对齐到生成

CLIP 通过对比学习实现了图文在特征空间的对齐，但只完成了「理解」的第一步。作为**双塔判别式**模型，CLIP 擅长「给定候选文本算相似度」的检索/分类判断，而不是**自由形式的文本生成**——无法直接对「画里有什么？」给出自然语言答案。要突破这一局限，需要赋予模型「**开口说话**」的生成能力。

DeepMind 的 **Flamingo** 等先驱尝试在冻结的视觉编码器与语言模型之间**插入跨注意力等新模块**，并在海量交错图文数据上训练新模块，效果惊艳但数据与训练成本并非普通实验室可承受。

换思路：手头已有「**视觉地基**」（ViT）与「**语言大脑**」（LLM），能不能只训练一个轻量「适配器」把它们连起来？Salesforce 的 **BLIP-2** 正是这一思路的代表：采用 **Bootstrapping** 策略，利用冻结的图像编码器与 LLM，以极低计算成本获得强多模态能力——论文报告在 VQAv2 零样本 test-dev 上相对 Flamingo-80B 高 **8.7 个百分点**，同时只需 **1/54 的可训练参数**。四两拨千斤的关键是轻量 **Q-Former**：作为**信息瓶颈**，从视觉特征中「萃取」最关键的信息并「翻译」给 LLM。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_20_1_1.png" width="90%" alt="BLIP-2 总体架构" />
  <p>图 1：BLIP-2 总体架构</p>
</div>

整个模型像搭积木：左侧冻结的图像编码器负责「看」，右侧冻结的 LLM 负责「说」，主要训练中间的轻量 Q-Former（以及对接 LLM 的小映射层）做高效「传译」。

### 1.1 Q-Former 与两阶段预训练

**Q-Former（Querying Transformer）** 是初始化自 **BERT-base** 的轻量 Transformer（仅 **188M 参数**），核心目标是连接冻结的视觉与语言模型。内部含两个**共享 Self-Attention 层**的子模块：

- **左侧路径（Image Transformer）**：接收一组固定数量的**可学习查询向量**（论文设定 **32 个**），通过每层的 **Cross-Attention** 与冻结图像编码器输出交互，从海量视觉特征中萃取精华；
- **右侧路径（Text Transformer）**：作为文本编码器/解码器处理文本输入。

双塔共享权重的设计，让 Queries 既能通过 Cross-Attention 学视觉特征，又能通过共享的 Self-Attention 与文本特征交互。32 个 Query 远小于原始图像特征的空间尺寸，强制高强度信息压缩——这就是「**信息瓶颈**」，确保传给 LLM 的都是经筛选、与文本最相关的有效信息。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_20_1_2.png" width="90%" alt="Q-Former 架构" />
  <p>图 2：Q-Former 详细架构</p>
</div>

**第一阶段：视觉-语言表征学习**（图像编码器冻结）。为让可学习查询向量提取既含视觉信息又与文本对齐的特征，Q-Former 设计三种预训练目标，用特定 **Attention Mask** 在同一架构中同时优化：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_20_1_3.png" width="70%" alt="第一阶段预训练目标" />
  <p>图 3：第一阶段预训练目标（a）ITM（b）ITG（c）ITC</p>
</div>

- **图文匹配（ITM）**：**Bi-directional Mask**，Query 与 Text 互相完全可见，学细粒度图文对齐；
- **图文生成（ITG）**：**Multimodal Causal Mask**，Query 可相互注意但看不见 Text，Text 可看见所有 Query 与之前的 Text，引导基于图像的文本生成；
- **图文对比学习（ITC）**：**Uni-modal Mask**，Query 与 Text 互不可见，专注对齐整体视觉与语言表征。

三种策略结合，确保 Query Embeddings 包含最精华的视觉语义。

> **Mask 图解**：每个方阵是注意力掩码，横纵坐标对应 Query 与 Text 的 Token，分四象限——左上 Q-Q、右下 T-T、右上/左下 Q-T 交叉；**空白=可见**（unmasked），**深色=不可见**（Masked）。ITC 中 Query 与 Text 互不可见，右上/左下即深色。

**第二阶段：视觉-语言生成学习**（LLM 也冻结）。引入全连接层把 Query Embeddings（$Z$）线性映射到 LLM 文本 Embedding 维度，作为「软视觉提示」拼在文本 Embedding 之前：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_20_1_4.png" width="90%" alt="BLIP-2 第二阶段预训练" />
  <p>图 4：第二阶段预训练——对接 Decoder-based 或 Encoder-Decoder-based LLM</p>
</div>

- 对接 **Decoder-based LLM**（如 OPT）：Q-Former 输出作前缀，单向注意力让它看见视觉 Prompt 并据此生成后续文本；
- 对接 **Encoder-Decoder LLM**（如 Flan-T5）：Q-Former 输出与文本前缀拼接后输入 Encoder，Decoder 根据跨模态表示生成后缀文本。

通过「软提示」机制，BLIP-2 用极小代价（主要训 Q-Former 与全连接映射层）把视觉感知「嫁接」到大语言模型上。

## 2. LLaVA 与视觉指令微调

### 2.1 为什么还需要指令微调

BLIP-2 虽已展示通过提示词做零样本指令式图像到文本生成，但其预训练目标核心仍围绕「模态对齐」与「图像条件生成」。在实际「助手式」交互场景中，模型**可以生成**，但对复杂指令的稳定遵循、多轮对话格式、更贴近人类偏好的回答风格，并没有被系统性对齐与强化（尤其缺少专门的视觉指令对话数据做端到端微调）。所以 BLIP-2 更像「能看懂、也能说」的通用接口原型，距离像 ChatGPT 一样可对话、可推理、强指令遵循的视觉助手仍有差距。

**LLaVA（Large Language and Vision Assistant）** 由此引入**视觉指令微调**，把多模态模型从「看图说话」工具升级为通用「智能视觉助手」。正如 NLP 从 GPT-3 到 ChatGPT 的进化离不开指令微调，多模态模型也需要高质量**视觉指令数据**来学习遵循人类意图——LLaVA 不仅给出简单架构，更重要的是给出低成本构建这些数据的方法。

### 2.2 数据构建：GPT-4 当老师

LLaVA 团队（2023.4）发现当时虽缺图像-指令对数据，但有丰富的**图像-文本对**（如 COCO）。于是用 GPT-4 作「老师」，以**上下文学习**方式把图像的**符号化表示**（多视角图像描述 + 现成的目标边界框/类别标注）与少量人工设计的种子样例喂给 GPT-4，生成总计 **158K** 条高质量指令数据：

- **58K 对话**：模拟人与助手的日常交互；
- **23K 详细描述**：要求对图片各方面做详尽刻画；
- **77K 复杂推理**：提出需基于图片内容深层逻辑推理的问题，并要求给出较清晰的推理过程。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_20_1_5.png" width="90%" alt="LLaVA 数据构建示例" />
  <p>图 5：LLaVA 数据构建示例</p>
</div>

注意 GPT-4 **并没有通过视觉编码器「看」原始图片**：它根据「Captions」（如「一群人站在黑色车旁」「SUV 在地下车库」）与「Boxes」（如 person 坐标、suitcase 坐标）在「脑海」里脑补出完整场景，再生成问答、细致描述，甚至推理出人们正面临「如何把大量行李装进车里」的挑战。**利用大语言模型的常识推理能力生成视觉指令数据**，正是 LLaVA 的核心创新之一。

### 2.3 架构与两阶段训练

LLaVA 架构非常简洁：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_20_1_6.png" width="80%" alt="LLaVA 模型架构" />
  <p>图 6：LLaVA 模型架构</p>
</div>

输入图片 $X_v$ 经 **Vision Encoder**（预训练 CLIP ViT-L/14）提取视觉特征 $Z_v$——实验比较了最后一层与倒数一层特征，在 ScienceQA 设置下用**倒数一层**带来约 **0.96 个百分点**提升；随后 **Projection Layer**（一个简单线性层 $W$）把视觉特征映射为 LLM 能理解的 Embedding $H_v$；开源 LLM **Vicuna**（基于 LLaMA 微调）同时处理视觉 Embedding 与文本指令 Embedding $H_q$，生成回复 $X_a$。

**两阶段训练**：

1. **特征对齐预训练**：CC3M 过滤至 **595K** 图像-文本对；**冻结 Vision Encoder 与 LLM，仅训投影层**——目标是训练与 LLM 兼容的「视觉 Tokenizer」，让图像特征对齐到 LLM 语义空间；
2. **视觉指令微调**：用 158K 高质量指令数据（对话/描述/推理），**冻结 Vision Encoder**，更新**投影层 + LLM 权重**——这一阶段让模型真正学会作为多模态助手与人交互。

**实验结果**：日常聊天展现出色多模态能力；ScienceQA（多模态科学问答）报告 **92.53%** 来自一种「集成」设置——当 LLaVA 与**文本版** GPT-4 结合、由 GPT-4 充当裁判在两者答案不一致时做最终仲裁时可达该准确率。

> **补记**：两种范式的对比值得记住——BLIP-2 是「**少训**」路线：冻结两端、只训 188M 的 Q-Former，用信息瓶颈换取极低训练成本，但生成能力受预训练目标限制；LLaVA 是「**换数据**」路线：架构极简（一个线性投影层），把成本从「结构设计」转移到「指令数据构建」（GPT-4 蒸馏 158K 条），换来助手式的对话与推理能力。这也预示了一个趋势：多模态模型的竞争，正在从「怎么连」转向「数据怎么造」。

### 2.4 投影层到底长什么样

跨模态对齐在代码层面比名字朴素得多：LLaVA 原版只用一个线性层，后续改进版换成两层 MLP，参数量在百万级，相对 7B 级 LLM 几乎可以忽略：

```python
import torch.nn as nn

# 视觉特征 1024 维（如 CLIP ViT-L/14 输出），LLM 词嵌入 4096 维
projector_linear = nn.Linear(1024, 4096)          # LLaVA 1.0 的做法

projector_mlp = nn.Sequential(                    # LLaVA 1.5 起的做法
    nn.Linear(1024, 4096),
    nn.GELU(),
    nn.Linear(4096, 4096),
)

# vision_features: (batch, num_tokens, 1024) -> llm_input: (batch, num_tokens, 4096)
llm_input = projector_mlp(vision_features)
```

两层的差别不在表达力上限，而在训练稳定性与收敛速度。这也解释了为什么「对齐」阶段可以只训投影层：它承担的是坐标系换算，不承担语义理解——语义理解已经由冻结的视觉编码器与 LLM 各自完成。

## 写在最后

BLIP-2 与 LLaVA 仍属于「连接」范式——视觉编码器与 LLM 各自独立、靠适配器拼接。下一篇[原生统一架构与简化Omni](3-原生统一架构与简化Omni)看 Flamingo、Chameleon、GPT-4o 如何走向单一网络，并用 Chameleon 的「一切皆 Token」思路亲手搭一个简化版 Omni。
