# PEFT综述与LoRA详解

## 1. 大模型时代的微调困境

「预训练-微调」范式自 BERT 起大获成功，但模型从数亿级跃升到千亿级后，**全量微调（Full Fine-Tuning）** 撞上四道墙：

- **训练成本高昂**：千亿参数微调需数百 GB 显存与大量时间；
- **存储压力巨大**：每个下游任务存一份完整模型副本；
- **灾难性遗忘**：针对特定任务微调会「忘记」预训练学到的通用知识；
- **训练不稳定**：网络又宽又深，对学习率极敏感，易梯度消失/爆炸。

### 1.1 提示范式的兴起与局限

GPT-3 论文带来无需训练的 In-Context Learning：不改参数，仅靠输入中的示例（Prompt）引导模型完成任务，催生了提示工程。这种人工设计的离散文本指令称**硬提示（Hard Prompt）**，但有三个局限：找最优提示依赖大量试错（玄学）；离散文本表达能力有上限；跨模型/跨语言迁移效果差。

### 1.2 参数高效微调（PEFT）

> **核心思想**：**冻结**预训练模型 99% 以上的参数，仅调整极小一部分（通常 <1%），或增加一些额外的「小参数」，以极低成本让模型适应下游任务。

思想借鉴 CV 迁移学习（冻结 ResNet 卷积层、只微调全连接层），应用到 Transformer 后发展出多条路线。

## 2. PEFT 技术脉络

### 2.1 Adapter Tuning（2019，Google）

在 Transformer 每个块中**插入**小型适配器：插到各子层（注意力、FFN）内部，与主干形成残差连接，训练时只更新 Adapter 参数。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_11_1_1.png" width="60%" alt="Adapter Tuning 结构" />
  <p>图 1：Adapter Tuning 结构</p>
</div>

Adapter 自身是「瓶颈式」结构：**降维 FC → 非线性激活 → 升维 FC**，外加贯穿模块的残差连接保证信息流稳定。用极少参数模拟任务知识，参数效率高、训练稳定、性能接近全量微调；代价是各层插入额外模块带来**激活内存与算力开销**，千亿级资源受限场景下工程挑战大。

### 2.2 Prefix Tuning（2021，Stanford）

与 Adapter 在模型内部「动手术」不同，Prefix Tuning 选择**在模型外部做文章**——给模型带一张「小抄」：

- **前缀激活值**：`PREFIX` 部分的激活 $h_i$（$i \in P_{idx}$）从专门的可训练矩阵 $P_\theta$ 中提取，这就是微调对象；
- **模型计算的激活值**：原始输入 $x$ 与输出 $y$ 的激活由**冻结**的 Transformer 正常算出。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_11_1_2.png" width="80%" alt="Prefix Tuning 注解示例" />
  <p>图 2：Prefix Tuning 注解示例（上：自回归；下：编码器-解码器）</p>
</div>

不但在输入层加前缀，还在 Transformer **每一层**加可学习 Prefix，并通过小 MLP 生成这些参数。优点是参数极省、显存友好（不更新原权重，无需优化器状态；但需为各层前缀 K/V 预留显存）、通用性强（GPT-2、T5/BART 均有效）。缺点是直接优化 Prefix 比微调 Adapter 更难、训练不稳、对超参与初始化敏感；且前缀长度通常计入注意力配额，会挤占有效上下文窗口。

### 2.3 Prompt Tuning（2021，Google）

Prefix Tuning 的**简化版**（软提示）——只在输入 **Embedding 层**加可学习的虚拟 Token（**Soft Prompt**），不干预任何中间层。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_11_1_3.png" width="80%" alt="Model Tuning 与 Prompt Tuning 对比" />
  <p>图 3：Model Tuning 与 Prompt Tuning 对比</p>
</div>

左图全量微调是「一个任务一个模型」：每个任务（Task A/B/C）各自微调一份 110 亿参数模型副本，存储与部署开销巨大。右图提示微调把任务知识**外置**到轻量 Prompt：为情感分析、问答、摘要分别学习 `Soft Prompt A/B/C`——它们不是人类可读文本，而是一组可经反向传播优化的连续向量（可用「情感/正面/负面」等词向量初始化，最终被调成「最优虚拟指令」）。不同任务的（数据+各自 Soft Prompt）可组成**混合批次**送入**同一个冻结模型**训练，只更新各 Prompt。结果：训练对象是万级参数的任务提示文件，大模型被所有任务共享。

论文最重要的发现是**规模缩放效应（The Power of Scale）**：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_11_1_4.png" width="60%" alt="Prompt Tuning 性能与模型规模的关系" />
  <p>图 4：Prompt Tuning 性能与模型规模的关系</p>
</div>

模型规模小时（如 1 亿），Prompt Tuning 远不如传统微调；**规模超过 100 亿后追平甚至超越全量微调**。说明模型「足够大」时，仅学一个微型 Soft Prompt 就能涌现任务适应能力——但也暴露其**强依赖模型规模**、中小模型效果差的局限。

### 2.4 P-Tuning v2：通用化的答案

P-Tuning 系列（清华团队）为解决软提示的稳定性与规模依赖而提出。

**P-Tuning v1** 的目标是解决**离散提示的不稳定性**：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_11_1_5.png" width="80%" alt="离散提示搜索与 P-Tuning 对比" />
  <p>图 5：离散提示搜索与 P-Tuning 对比</p>
</div>

（a）离散提示搜索在离散文本空间找最优词组合，只能依赖离散奖励信号，优化困难、易得次优解。（b）P-Tuning 改为在连续空间优化：定义一组可学习的**伪提示（Pseudo Prompts）** $[P_0],...,[P_m]$，送入小型 `Prompt Encoder`（如 LSTM）编码依赖关系，输出任务相关向量 $h_0,...,h_m$ 作为大模型输入。Prompt Encoder 与伪提示都可端到端反向传播——「找最优提示」从人工试错变成**梯度下降可解**的数学问题。

以首都预测为例：传统离散模板 `"The capital of Britain is [MASK]"` 换成 `"Britain's capital city is [MASK]"` 输出就可能剧变；P-Tuning 则把输入变成 `[h_0,...,h_i], "capital", "Britain", [h_{i+1},...,h_m], [MASK]`——$h$ 不是词表里的词，而是**可训练向量**，训练中在连续空间不断调整，最终收敛为人类读不懂、但对模型最优的提示特征。但 v1 仍对模型规模敏感，且复杂 NLU 任务（尤其序列标注）表现不佳。

**P-Tuning v2** 博采 Prefix Tuning 之长，解决 v1 两大痛点：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_11_1_6.png" width="80%" alt="P-Tuning v1 与 P-Tuning v2 的结构对比" />
  <p>图 6：P-Tuning v1（a）与 P-Tuning v2（b）结构对比</p>
</div>

1. **深层提示（Deep Prompting）**：v1 的提示向量**仅插在输入层**（Shallow Prompting），信息要穿透所有层才能影响深层行为，这是中小模型效果差的根因。v2 把提示**复制并独立注入每一层**（Layer 1/2... Prompts），每层 Transformer 块都能直接接收可学习提示——相当于层层「开小门」，小模型也能显著受益。
2. **回归传统分类头（No Verbalizer）**：v1 为利用预训练目标，强行把所有任务包装成完形填空（如情感分类必须预测 "good"/"bad" 再映射标签），序列标注、抽取式阅读理解这类逐 token 细粒度任务几乎无法设计 Verbalizer。v2 抛弃 Verbalizer，最后一层直接接线性分类头，像传统微调一样输出标签。

所以 P-Tuning v2 = **Prefix Tuning 的多层结构 + 传统微调的输出头 + Prompt Tuning 的轻量化**，仅微调 0.1%~3% 参数，同时找回全量微调在复杂任务上的通用性与小模型上的稳定性。

## 3. LoRA：直击权重矩阵

前面的方法都在尽量不「打扰」原始权重，靠影响**激活值**适配任务。**LoRA** 另辟蹊径——直击**权重矩阵**，其出发假设是：

> **大语言模型是过参数化的（Over-parametrized），针对特定任务微调时，权重更新矩阵 $\Delta W$ 具有很低的「内在秩」（Intrinsic Rank）。**

全量微调要为每个 $W$ 存一个同样大的 $\Delta W$；LoRA 用两个小矩阵之积近似它：

$$ \Delta W = B \cdot A, \qquad W_0 \in \mathbb{R}^{d \times k},\ B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d,k) $$

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_11_2_1.svg" width="50%" alt="LoRA 结构" />
  <p>图 7：LoRA 结构</p>
</div>

计算走两条路：**主路** $xW_0$（$W_0$ 冻结），**旁路** $xAB$（$A$ 先压到秩 $r$，$B$ 再解压回 $d$），最终：

$$ h = W_0 x + (B A) x $$

训练只更新 $A$、$B$，可训练参数从 $d \times k$ 降到 $d \times r + r \times k$；$r$ 取 8/16/64，参数量仅为全量的千分之一甚至万分之一。

> **初始化与缩放**：$A$ 用高斯分布随机初始化、**$B$ 初始化为全零**——保证训练开始时旁路输出为零，从原始预训练状态出发，训练初期稳定。前向公式还带缩放因子 $s$：$h = W_0 x + s(BA)x$，$s = \alpha/r$（$\alpha$ 可调），让调整 $r$ 时不必重调学习率。

### 3.1 核心优势

- **参数与存储效率**：每个任务只存极小的 $A$、$B$，checkpoint 体积可缩至**万分之一**（350GB → 35MB）；训练不用为冻结参数算梯度/存优化器状态，可省高达 **2/3 GPU 显存**、提速约 **25%**；
- **零额外推理延迟**（对比 Adapter 最大卖点）：训练完把旁路**合并**回原权重 $W' = W_0 + sBA$，网络结构与原模型完全一致，无额外计算。代价是想在同一 batch 混合服务多个任务的 LoRA 权重不再直接；
- **效果媲美全量微调且不占输入长度**：直接改权重，比作用于激活的方法更深入；不添加 virtual token，长文本任务更有优势；
- **可组合**：设计正交，可与 Prefix-Tuning 等结合。

### 3.2 关键实践：改哪些矩阵、秩取多大

把研究范围限定在**自注意力**权重（$W_q, W_k, W_v, W_o$），冻结其余。表 11-1 的结论：固定可训练参数预算下，**多种类型权重组合**（尤其 $W_q$+$W_v$）比把预算全堆到单一权重的更大秩更好——**仅在注意力模块应用 LoRA 并冻结其余部分**是高效策略。

表 11-2 的结论：**小小的秩就够了**（4、8 甚至 1），盲目增大 $r$ 增加参数甚至降性能；$W_q$+$W_v$ 组合在 $r=1/2$ 时已有竞争力，甚至超过 $r=64$。这反证了权重更新确实是低秩的。

**LoRA 究竟如何生效？** 论文分析发现 $\Delta W$ 不是复制 $W_0$ 中最重要的特征，恰恰相反——它学到的是预训练中**未被充分强调、却对下游任务关键**的「隐藏特征」并大幅放大。它不重复模型已擅长的事，而是**查缺补漏**。

## 4. AdaLoRA：让秩自适应

LoRA 的固定设置仍留两个问题：秩 $r$ 是固定超参、无法训练中自适应；改哪些矩阵靠手工选（原论文只看注意力，后续研究证明 FFN 同样重要）。实验已揭示重要性差异——同等预算下微调 FFN（$W_{f1}, W_{f2}$）收益显著高于注意力权重，微调高层（10-12 层）也比底层（1-3 层）更有效：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_11_2_2.png" width="70%" alt="AdaLoRA 动机" />
  <p>图 8：不同模块与层级的微调性能对比</p>
</div>

手动为每个矩阵与层级找最优秩，超参空间几乎不可行。**AdaLoRA** 改为**按权重重要性动态分配参数预算**，三项关键创新：

**（1）基于 SVD 的参数化**：不用 $BA$，改用 $P \Lambda Q$ 模拟奇异值分解——$P$、$Q$ 决定更新「方向」，对角阵 $\Lambda$ 的奇异值 $\lambda_i$ 决定各方向更新「幅度」。训练时 $P,\Lambda,Q$ 均为可训练参数，只是**形式上模拟** SVD，不做昂贵分解；并加入**正交正则项**保证 $P$、$Q$ 的正交性。

**（2）重要性评分与动态预算**：把每个奇异值与对应左右向量组成**三元组** $\mathcal{G}_{k,i} = \{P_{k,*i}, \lambda_{k,i}, Q_{k,i*}\}$，为其算重要性 $S_{k,i}$。单参数重要性 $s(w)$ = 平滑后的**敏感度** $\bar I(w)$ × 相关项，其中：

$$ I(w) = |w \cdot \nabla_w \mathcal{L}|, \qquad \bar I^{(t)}(w) = \beta_1 \bar I^{(t-1)}(w) + (1-\beta_1) I^{(t)}(w) $$

敏感度直观看是「把该参数置零，损失变化多大」；因 mini-batch 梯度随机，用 EMA 平滑；同时用 $\bar U^{(t)}(w) = \beta_2 \bar U^{(t-1)} + (1-\beta_2)|I^{(t)} - \bar I^{(t)}|$ 度量**不确定性**。既高敏又稳定的参数更重要。三元组分数由内部参数重要性聚合，排序后按预算**裁剪**得分最低的三元组（奇异值置 0），实现动态分配。

**（3）全局预算调度器**：目标函数 $\mathcal{L} = \mathcal{C} + \gamma\sum R(P_k,Q_k)$（原损失 + 正交正则）；预算 $b(t)$（第 $t$ 步保留的奇异值总数）分段控制：**热身**（从略高初始预算出发，充分探索）→ **裁剪**（按三次方曲线逐步降到目标值）→ **微调**（固定预算收敛）。「先探索、后收敛」让模型充分发现哪些权重真正重要。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_11_2_3.png" width="90%" alt="AdaLoRA 最终秩分配结果示意图" />
  <p>图 9：AdaLoRA 最终秩分配结果示意</p>
</div>

结果与实验观察吻合：模型自动给 FFN（$W_{f1}, W_{f2}$）与高层（6-12 层）分配了更高的秩。消融还证明：**即使不做动态预算，仅把参数化从 $BA$ 换成 $P\Lambda Q$ 就有性能提升**，说明 SVD 结构本身更优。

> **两个联系**：与 Adapter 都是「高维→低维→高维」瓶颈，但 Adapter 作用于**激活值**、串行、增延迟；LoRA/AdaLoRA 作用于**权重**、并行、可合并。与理论篇的主题模型 SVD 同源——那边分解「词-文档」矩阵发现语义主题（数据层面低秩），这里分解「权重更新矩阵」找关键参数变化方向（模型层面低秩）。

## 5. QLoRA：把基座压到 4-bit

LoRA/AdaLoRA 还有个共同前提：冻结的模型仍以 FP16/BF16 加载，百千亿参数本身就是巨大显存开销。**QLoRA** 把基座模型**量化为 4-bit**，训练时梯度穿过冻结的 4-bit 模型反传到 16-bit 适配器、只更新适配器；配合**分页优化器**在显存不足时把优化器状态换出到 CPU。成果：65B 模型微调压到一块 **48GB** GPU；Guanaco 系列在 Vicuna 基准达 ChatGPT 99.3%，单卡 24 小时。三板斧：

**（1）4-bit NormalFloat（NF4）**：神经网络权重是**零中心正态分布**（大量集中在 0 附近、少量离群值），均匀量化不适配这种非均匀分布。传统 8-bit 均匀量化靠 `absmax` 缩放：

$$ X^{Int8} = \mathrm{round}\left( \frac{127}{\mathrm{absmax}(X^{FP32})} X^{FP32} \right) = \mathrm{round}(c^{FP32} \cdot X) $$

对离群值极敏感。NF4 基于**分位数量化**——让每个量化「桶」包含相同数量的分布值，数据密集处量化点密、尾部稀疏。构建：取标准正态 $N(0,1)$ → 算出 $2^4 = 16$ 个把 CDF 等分为 16 段的**分位数**作可表示值 → 对每个权重块先按 absmax 归一化到 $[-1,1]$（缩放因子即该块量化常数，32-bit）→ 每个值映射到最近的 NF4 分位点。更精确地：

$$ q_i = \frac{1}{2}\left( Q_X\!\left(\frac{i}{2^k+1}\right) + Q_X\!\left(\frac{i+1}{2^k+1}\right) \right) $$

$Q_X(\cdot)$ 是标准正态分位数函数，即第 $i$ 个量化点取相邻两个等概率分位点的中点。4 bit 便实现对正态权重的高精度近似，信息论上对正态分布最优。

**（2）双量化（DQ）**：每个权重块要存一个 32-bit 量化常数，block=64 时每参数多 0.5 bit 开销。DQ 思想是**对量化常数本身再量化一次**——用 8-bit 浮点做二级量化，把额外开销从 0.5 bit 降到约 **0.127 bit**。

**（3）分页优化器**：梯度和优化器状态（Adam 的动量/方差）在长序列下会产生瞬时显存峰值，易 OOM。借鉴操作系统虚拟内存，利用 **NVIDIA Unified Memory**，显存不足时按需把优化器状态分页到 CPU、需要时再载回，大幅提升训练稳定性。

**工作流程（存算分离）**：

1. **加载与量化（存）**：16-bit 模型权重 → 量化为 4-bit NF4 + 双量化，冻结在 GPU，显存占用极低；
2. **前向（算）**：插入 16-bit（BF16）LoRA 适配器；计算时所用基座权重**动态反量化回 BF16**，用完立即丢弃释放显存；
3. **反向与更新**：梯度只从冻结 4-bit 模型反传到 16-bit 适配器并只更新适配器；显存峰值时由分页优化器兜底。

$$ Y^{BF16} = X^{BF16} \cdot \mathrm{doubleDequant}(c_1^{FP32}, c_2^{k\text{-}bit}, W^{NF4}) + X^{BF16} L_1^{BF16} L_2^{BF16} $$

前半是反量化后的主路、后半是全程 16-bit 的 LoRA 旁路。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_11_2_4.png" width="80%" alt="QLoRA 与其他方法对比" />
  <p>图 10：全量微调、LoRA 与 QLoRA 的机制对比</p>
</div>

> **补记**：PEFT 的三条技术路线可以按「作用对象」一句话记：Adapter 改**结构**（插模块）、Prompt/Prefix/P-Tuning 改**激活**（加软提示）、LoRA/AdaLoRA/QLoRA 改**权重**（低秩旁路 + 量化基座）。演进逻辑也清晰：Adapter 增延迟 → LoRA 用并行旁路且可合并；LoRA 秩固定 → AdaLoRA 重要性自适应；LoRA 基座仍是 16-bit → QLoRA 把基座压到 4-bit 并解决量化常数与显存峰值。理解这条「作用对象 + 补齐短板」的链条，后面看任何 PEFT 新方法都能快速定位它在解决什么。

## 写在最后

理论就位，接下来是动手——下一篇[LoRA实战与Qwen2.5微调](2-LoRA实战与Qwen2.5微调)用 peft 库把 LoRA 跑起来，再到 Qwen2.5 私有数据上完成一次完整的 QLoRA 微调。
