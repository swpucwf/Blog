# MoE架构解析

## 1. 稠密模型的天花板

Llama2、GPT-3 这类是**稠密模型（Dense Model）**：每个输入 token，模型从第一层到最后一层的**所有参数**都参与计算。向万亿参数迈进时，全量计算的算力成本变得不可承受。**混合专家模型（MoE）** 用「**稀疏激活**」机制，兼得大规模参数的知识容量与极低的推理成本——Mistral 8x7B 等证明 MoE 在开源大模型同样有巨大潜力。

## 2. MoE 的来龙去脉

### 2.1 自适应局部专家混合（1991）

MoE 思想最早追溯到 Jordan 与 Hinton 的《Adaptive Mixture of Local Experts》——它不只提出分治架构，更从理论层面解决了神经网络多任务学习的根本难题。

**干扰效应与分治**：单体网络同时学多个迥异子任务（既学做菜又学修车）会出现**强干扰效应**——所有权重参与所有任务，为任务 A 调参会破坏任务 B 已学到的特征，导致学得慢、泛化差。解法是**分治**：

- **专家网络**：多个独立子网，每个只专注输入空间的一个局部区域/一类子任务；
- **门控网络**：协调者，接收与专家同样的输入 $x$，输出一组**混合比例** $p_i$（选每个专家的概率），像软性随机开关决定当前输入由哪位专家主导。

**损失函数设计**是关键贡献——要鼓励专家**竞争**而非合作。若简单线性相加 $\hat{y}=\sum p_i E_i$ 再在输出上算误差：

$$ Loss_{coop} = \left\| \mathbf{y} - \sum_{i} p_i \mathbf{E}_i \right\|^2 $$

合作模式下每个专家都去弥补别人的残差，每个案例牵动所有专家权重，干扰没解决。作者把系统视为随机生成器，改用**负对数似然**：

$$ Loss_{comp} = - \log \sum_{i} p_i e^{-\frac{1}{2} \| \mathbf{y} - \mathbf{E}_i \|^2} $$

某专家 $E_i$ 输出越接近 $y$（误差项小），指数项越接近 1（误差大的趋 0）；因 $\sum p_i=1$，为最大化求和（最小化负对数 Loss），门控会显著抬高「表现好」专家的 $p_i$、压低其他——即**赢家通吃**，实现**权重解耦**：反向传播时只有胜出的专家和门控显著更新，其他几乎不动，有效缓解干扰，「让专业的人干专业的事」。

> **MoE vs 集成学习**：结构相似但本质不同。集成学习（如随机森林）假设基模型独立/互补，预测时**所有模型都参与**、投票或加权平均；MoE 强调**动态条件计算**——根据输入本身动态划分任务空间，不同输入激活不同子网络路径。

### 2.2 深度 MoE（2013）

Sutskever 等的《Learning Factored Representations in a Deep Mixture of Experts》把 MoE 与深度学习开创性结合，提出 **Deep Mixture of Experts（DMoE）**：MoE 不再是孤立浅层模型，而是**可插拔的层**——在网络不同层各插入 MoE 模块，每层有独立门控与专家集。

- **层级化门控**：输入 $x$ 经第一层门控 $g^1$ 路由到专家 $f^1_i$，其输出 $z^1$ 再作第二层门控 $g^2$ 输入路由到 $f^2_j$；
- **组合路径指数增长**：第一层 $N$ 个专家、第二层 $M$ 个，潜在组合路径达 $N\times M$，每个样本按特性动态选一条最适合的路径。

在 Jittered MNIST 上观察到自动**特征解耦**：第一层专家按数字**位置**分工（Where Experts），第二层按**类别**分工（What Experts）——深度 MoE 能把复杂任务分解成多个正交子问题，为后来 MoE 在 Transformer 中的应用奠基。

### 2.3 稀疏门控 MoE（2017）

Google Brain（Hinton、Jeff Dean 等）的《Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer》把 MoE 带进百亿参数时代。

**条件计算**回答两难困境：参数越多准确率越高，但成本随之飙升——能否**不增计算量而大幅增加参数量**？答案是**稀疏门控 MoE 层**：每个样本处理时只激活网络一小部分。引入 1370 亿参数的 MoE 层（数千专家），在语言建模与机器翻译上显著超 SOTA，计算效率仅微损。架构上把 MoE 层嵌入堆叠 LSTM 之间，门控算稀疏权重、只激活少数专家，其余闲置，输出加权求和传给下一层——对每个 token 动态选不同专家组合。

<div align="center">
  <img src="{{IMG:6_2_1.png}}" width="70%" alt="LSTM + MoE 架构图" />
  <p>图 1：稀疏门控 MoE 层架构</p>
</div>

落地须解决两挑战：

- **如何稀疏选择——带噪声门控**：Softmax 门控给所有专家非零权重（都得算，省不了算力）。加可训练高斯噪声，只保留权重最大的 $k$ 个（如 $k=4$），其余强制置 $-\infty$（概率 0）。效果：计算预算基本不变时，专家数从 4 增到 4096，参数量剧增而困惑度显著下降——条件计算能在不加推理成本下用海量参数提升性能。

<div align="center">
  <img src="{{IMG:6_2_2.png}}" width="50%" alt="模型容量与效果对比图" />
  <p>图 2：模型容量与测试集困惑度关系</p>
</div>

- **如何防专家崩塌——负载均衡**：稀疏选择易引发**马太效应**——初期某些专家因随机初始化「运气好」被选中、得到梯度更新变「更聪明」、更常被选中；落选专家得不到训练持续「愚钝」最终「饿死」，模型退化成只有少数活跃专家的稠密模型。解法是在总损失加**辅助损失**（Importance Loss + Load Loss），不服务准确率、专罚「分配不均」，强制门控「雨露均沾」。

## 3. 大模型时代的 MoE

Transformer 时代，Google 用 GShard、Switch Transformer、GLaM 一系列工作确立现代大规模 MoE 范式。

### 3.1 GShard：迈向六千亿参数

2020 年 GShard 把 Transformer 扩到 **600B** 参数——它不只是模型，更是一套支持超大稀疏模型的并行训练框架，把**数据并行**与**模型并行**优雅结合。

**MoE 层构建**标准化：不全盘替换，而用**隔层替换**——Self-Attention 保持（参数小且计算关键），FFN 替换为 MoE 层（如 1、3、5 层用 MoE、2、4、6 留标准 FFN）。输出是门控选出的专家加权和：

$$ \mathbf{y} = \sum_{i=1}^{N} p_i(\mathbf{x}) \cdot \mathbf{E}_i(\mathbf{x}), \qquad \mathbf{E}_i(\mathbf{x}) = \mathbf{W}_{out} \cdot \text{ReLU}(\mathbf{W}_{in} \cdot \mathbf{x}) $$

$E_i$ 内部就是标准双层全连接。GShard 提出 **Top-2 Gating**：发现每 token 只路由给 2 个专家就够——第一专家取最高权重保主任务，第二专家按权重概率随机取或取次高、引入辅助处理与随机性助负载均衡。Top-2 后来成了标准配置（包括 Mistral）。

**分布式并行**：单卡装不下 600B。非 MoE 层（Attention）**复制**到所有设备做数据并行；MoE 层**分片**（如 2048 专家分 2048 TPU 核）；token 要路由到不在本设备的专家时，用 **All-to-All** 通信发过去算完传回。

<div align="center">
  <img src="{{IMG:6_2_3.png}}" width="80%" alt="GShard MoE Transformer Encoder 架构图" />
  <p>图 3：GShard 架构与「复制+分片」并行策略</p>
</div>

由此实现**亚线性**成本增长：参数量增 16 倍（37.5B→600B），训练算力成本仅增不到 4 倍。

### 3.2 Switch Transformer：万亿参数与 Top-1 路由

2021 年 Switch Transformer 简化路由、改进训练，把参数推到 **1.6 万亿**，同时实现 4 倍于 T5-XXL 的训练速度。

**Switch Layer** 用稀疏 FFN 替换稠密 FFN，核心是 **Top-1 Routing**——每 token 只发给**唯一**一个专家（区别于传统 Top-k）。三大优势：路由决策更简单、每 token 只发一个目的地降通信、每专家处理的 token 数（Expert Capacity）至少减半。看似激进，实验证明不降质量还大幅提速。

<div align="center">
  <img src="{{IMG:6_2_4.png}}" width="80%" alt="Switch Transformer Encoder 架构图" />
  <p>图 4：Switch Transformer 架构</p>
</div>

**专家容量与 Token Dropping**：硬件要静态 Tensor 形状，须预设每专家最大处理 token 数——

$$ \text{Capacity} = \frac{\text{TotalTokens}}{\text{NumExperts}} \times \text{CapacityFactor} $$

Capacity Factor 通常 >1.0，给专家额外缓冲槽位应对分配不均；超容量时触发 **Token Dropping**——多余 token 不经专家处理、直接走残差传到下一层（保证静态形状但可能损失信息，容量设置很关键）。

<div align="center">
  <img src="{{IMG:6_2_5.png}}" width="90%" alt="Token Routing Dynamics" />
  <p>图 5：Token 路由动态与专家容量</p>
</div>

辅助损失鼓励 token 均匀分布、减少丢弃：

$$ Loss_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i $$

$f_i$ 是**实际**分给专家 $i$ 的比例、$P_i$ 是**预期**路由给它的概率和——让「实际去的」与「计划去的」点积最小，只有全均匀分布时才最小，逼 Router 不偏科。

**训练/微调稳定性四招**：**Router z-loss** 罚过大 logit 减数值溢出；**选择性精度**只在局部路由计算用 float32、其余 bfloat16，稳定又不增通信成本；**更小初始化方差**（高斯标准差缩到 1/10）提升初期稳定；**专家正则化**——微调时专家层内高 Dropout（0.4）、非专家层低（0.1）防过拟合。

### 3.3 GLaM：高能效通用语言模型

2021 年 GLaM 用与 GPT-3 相同的 **Decoder-only** 架构（适合 Few/Zero-shot），把参数推到 **1.2 万亿**（GPT-3 的 7 倍），却因稀疏 MoE 有更高训推效率。隔层替换 FFN 为 MoE 层，每 MoE 层 64 专家取 **Top-2**；总参数 1.2T 但每 token 仅激活 **96.6B（约 8%）**，推理 FLOPs 约为 GPT-3 的一半。团队还发现**高质量数据**很关键——用质量分类器严格过滤网页数据训练的模型优于未过滤海量数据。

<div align="center">
  <img src="{{IMG:6_2_6.png}}" width="40%" alt="GLaM 模型架构图" />
  <p>图 6：GLaM 架构</p>
</div>

<div align="center">
  <img src="{{IMG:6_2_7.png}}" width="100%" alt="GLaM 与 GPT-3 性能与成本对比" />
  <p>图 7：GLaM 与 GPT-3 性能与成本对比</p>
</div>

GLaM 用数据证明稀疏模型的「更大更强更环保」：

| 对比维度 | 指标 | GPT-3 (175B) | GLaM (1.2T) | 变化 |
|---|---|---|---|---|
| 成本 | 推理 FLOPs/token | 350 G | 180 G | **-48.6%** |
| 成本 | 训练能耗 | 1287 MWh | 456 MWh | **-64.6%** |
| 平均准确率 | Zero-shot | 56.9 | 62.7 | **+10.2%** |
| 平均准确率 | One-shot | 61.6 | 65.5 | **+6.3%** |
| 平均准确率 | Few-shot | 65.2 | 68.1 | **+4.4%** |

这打破了「性能提升必须靠堆更多算力」的固有认知。

## 4. MoE 架构的创新与实践

开源社区让 MoE 不再是巨头专属，Mistral 8x7B 与 DeepSeek-R1 分别在中等/超大尺度证明开源 MoE 实力。

### 4.1 Mistral 8x7B：如何以小博大

**Mixtral** 证明合理设计的稀疏模型不必万亿参数也能越级：

- **架构参数**：总参数约 **47B**，每 token 仅激活 **13B**——推理有 13B 的速度、47B 的知识容量。但注意：算量小，可所有专家参数都要加载进内存，**显存占用仍是 47B 级别**；
- **路由机制**：每层 8 专家、标准 **Top-2**——每 token 分给 8 个中 2 个，两个专家输出加权求和作层输出；

<div align="center">
  <img src="{{IMG:6_2_8.png}}" width="90%" alt="Mixture of Experts Layer" />
  <p>图 8：Mistral 8x7B 的 Top-2 路由</p>
</div>

- **性能**：GSM8K、MMLU、HumanEval 等基准上以 13B 活跃参数超越稠密的 Llama 2 70B 与 GPT-3.5，数学/代码尤著；支持 **32k** 上下文，Passkey Retrieval 达 100% 召回；

<div align="center">
  <img src="{{IMG:6_2_9.png}}" width="90%" alt="Mistral 8x7B 性能对比" />
  <p>图 9：Mistral 8x7B 与 Llama 2 各基准对比</p>
</div>

- **路由分析（反直觉发现）**：专家**没有按「学科领域」（生物/数学/哲学）分工**。统计不同领域数据（arXiv/PubMed/Wikipedia）在各层的专家分配比例，同一专家在不同数据集上颜色深浅极接近——**无论输入哪个领域，Router 选各专家的概率分布几乎一样**。专家更多按**语法**和 **Token 结构**（缩进、介词）分工，而非人类定义的知识领域。

<div align="center">
  <img src="{{IMG:6_2_10.png}}" width="90%" alt="Mistral 8x7B 路由专家分布" />
  <p>图 10：各领域数据的专家路由分布，显示无领域偏差</p>
</div>

### 4.2 DeepSeekMoE 与 DeepSeek-R1

**DeepSeek-R1**（基座 DeepSeek-V3）把开源 MoE 推到与顶尖闭源模型（OpenAI o1）比肩。DeepSeek 提出 **DeepSeekMoE** 解决传统 Top-k 路由的「知识冗余」和「专业化不足」。两大策略（区别于 Mistral 的粗粒度专家）：

- **细粒度专家分割**：把一个大专家拆成多个小专家（专家 1 拆为 1、2），保持总量不变则激活数 $K$ 倍增（2→4）——组合可能性指数增加，更灵活拼「知识碎片」应对复杂输入，专家专业化更高；
- **共享专家隔离**：指定某专家为**共享专家**，不经 Router 选择、独立通路直收输入、对所有 token 总激活；Router 只从剩余路由专家选 $K=3$ 补充。共享专家捕获通用跨任务知识（语法），路由专家专注特定领域——「通用+专用」分离减少重复学通用知识的冗余，提升参数效率：

$$ \mathbf{y} = \underbrace{\sum_{i \in \mathcal{S}} \mathbf{E}_i(\mathbf{x})}_{\text{Shared}} + \underbrace{\sum_{j \in \text{TopK}(\mathcal{R})} p_j(\mathbf{x}) \cdot \mathbf{E}_j(\mathbf{x})}_{\text{Routed}} $$

$\mathcal{S}$ 共享专家集合（总激活）、$\mathcal{R}$ 路由专家集合（选择性激活）——双路径是它区别于传统 MoE（前面 $\mathbf{y}=\sum p_i E_i$）的关键。

<div align="center">
  <img src="{{IMG:6_2_11.png}}" width="80%" alt="DeepSeekMoE 架构图" />
  <p>图 11：DeepSeekMoE 演进（传统 Top-2→细粒度→细粒度+共享专家）</p>
</div>

DeepSeek-R1 更靠**大规模强化学习**具备强推理：AIME 2024 Pass@1 达 79.8%（略高 OpenAI-o1-1217），MATH-500 达 97.3%（与 o1 持平），Codeforces Elo 2029（超 96.3% 人类参赛者）。

<div align="center">
  <img src="{{IMG:6_2_12.png}}" width="80%" alt="DeepSeek-R1 性能对比" />
  <p>图 12：DeepSeek-R1 在数学/代码/知识基准的表现</p>
</div>

## 5. MoE 代码实战：给 Llama2 换上 MoE 层

只需两处改动、不到 50 行：`ffn.py` 加含门控与多专家的 `MoE` 类，`transformer.py` 用它替换 `FeedForward`；Attention、RoPE、Norm 全不动。MoE 层在 Transformer Block 的 FFN 位置引入 Router 与 Experts，输入经 RMSNorm 后进 MoE——Router 算权重选 Top-k，选中专家并行处理再加权求和。

<div align="center">
  <img src="{{IMG:6_2_13.svg}}" width="40%" alt="Llama2 + MoE 架构图" />
  <p>图 13：Llama2 + MoE 架构</p>
</div>

```python
class MoE(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ffn_dim_multiplier, num_experts=8, top_k=2):
        self.gate = nn.Linear(dim, num_experts, bias=False)        # 门控：决定 token 去哪个专家
        self.experts = nn.ModuleList([FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)
                                      for _ in range(num_experts)])  # 8 个独立 FFN 专家
    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        gate_logits = self.gate(x_flat)                            # ① 门控打分
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)  # ② Top-k 路由
        weights = F.softmax(weights, dim=-1)                       #    归一化 k 个权重
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            batch_idx, k_idx = torch.where(indices == i)           # ③ 找选中专家 i 的 token
            if len(batch_idx) == 0: continue
            expert_out = expert(x_flat[batch_idx])                 #    分发计算
            output.index_add_(0, batch_idx, expert_out * weights[batch_idx, k_idx].unsqueeze(-1))  # ④ 加权聚合
        return output.view(B, T, D)
```

逻辑清晰（虽循环不如 CUDA Kernel 高效）：**Gate** 算每 token 对 8 专家的打分；**Top-k** 选最高 2 个并 softmax 归一化权重；**Dispatch** 遍历专家、`torch.where` 找出被分配的 token 索引进对应 SwiGLU FFN；**Combine** 专家输出乘门控权重、`index_add_` 累加回对应位置。每 token 最终输出=它激活的 2 个专家输出的加权和。

`transformer.py` 里把 `self.feed_forward` 换成 `MoE(num_experts=8, top_k=2)`——正好 Mistral 8x7B 的经典配置。`main.py` 完全不改即可跑通（对调用者接口无变化，MoE 复杂性全封装在层内），输出 `logits shape: (2, 16, 1000)`。**可插拔**地集成进现有 Transformer，正是 MoE 架构优雅之处。

> **补记**：MoE 一部史可归为一条「怎么把参数做大又不把算力拖垮」的演进——1991 用负对数似然让专家竞争解耦，2013 把 MoE 做成可插拔层、组合路径指数增长，2017 加噪 Top-k 实现稀疏门控 + 辅助损失防崩塌，GShard 定型「隔层替换 FFN + Top-2 + 复制/分片并行」，Switch 反向简化到 Top-1 提速，GLaM/Mistral 把范式推到万亿与开源，DeepSeek 再抠细粒度 + 共享专家。全部围绕稀疏率、负载均衡、专家专业化三个旋钮，理解了这三者，看任何新 MoE 模型都能快速定位它的改动落在哪。

## 参考文献

- Jacobs, Jordan, Nowlan, Hinton (1991). *Adaptive Mixtures of Local Experts*
- Eigen, Ranzato, Sutskever (2013). *Learning Factored Representations in a Deep Mixture of Experts*
- Shazeer et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer*
- Lepikhin et al. (2020). *GShard: Scaling Giant Models with Conditional Computation*
- Fedus, Zoph, Shazeer (2021). *Switch Transformers*
- Du et al. (2021). *GLaM: Efficient Scaling of Language Models with MoE*
- Jiang et al. (2024). *Mixtral of Experts*
- DeepSeek-AI (2024/2025). *DeepSeek-V3 / DeepSeek-R1 / DeepSeekMoE*

## 写在最后

MoE 是训练侧的扩容思路，而模型训好后「怎么吐字」同样关键——下一篇[大模型生成策略](11-大模型生成策略)记录从贪心到束搜索、采样与各类解码技巧，把输出质量握在解码这一环。
