# DeepSpeed框架介绍

## 1. DeepSpeed 全家桶

前面用 QLoRA、PEFT、RLHF 体验了**单机单卡**（甚至消费级显卡）完成微调的可能。但目标若从「微调 7B」升级为**从零预训练、全量微调乃至训练万亿参数**，只靠量化与 LoRA 就不够了——这时需要微软开源的 **DeepSpeed**：PyTorch 分布式训练与推理优化库，目标是让开发者在相同硬件下**用更少显存训练更大模型、训练更快、扩展更稳**。它与 Hugging Face 的分工可概括为：HF 偏「模型与数据」生态（transformers、datasets），DeepSpeed 偏「系统与算力」生态（显存优化、并行策略、通信调度）。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_13_2_1.png" width="100%" alt="DeepSpeed 官方文档 Getting Started 页面" />
  <p>图 1：DeepSpeed 官方文档 Getting Started 页面</p>
</div>

主要子模块：

- **DeepSpeed Training**：本篇主角，含 ZeRO、Offload、Infinity、3D 并行等训练技术；
- **DeepSpeed Inference / Compression**：推理加速与压缩（量化、剪枝、KV Cache 优化）；
- **DeepSpeed-MoE**：面向 MoE 大模型的高效路由与通信；
- **DeepSpeed-Chat / RLHF**：大规模 RLHF、对话模型训练；
- **DeepSpeed for Science**：把系统能力扩展到科学计算。

日常最常见两种用法：单机多卡/多机多卡做预训练或全量微调时重点用 **Training + ZeRO**；部署高吞吐推理服务时结合 **Inference 与量化/压缩**。

### 1.1 要解决的三大挑战

1. **显存不够**：单卡 80GB 也放不下万亿模型训练所需状态（权重、梯度、优化器状态、激活值）；
2. **加速比不理想**：单纯堆 GPU，数据并行很快撞上通信瓶颈，难获近线性加速；
3. **硬件异构**：集群里 GPU 显存、CPU 内存、NVMe SSD 甚至远程存储并存，如何让它们一起干活而非让 GPU 空等。

围绕这三点，DeepSpeed 以 **ZeRO（Zero Redundancy Optimizer）** 为核心（ZeRO、ZeRO-Offload、ZeRO-Infinity），再叠加数据/模型/流水线并行构成完整 3D 并行框架。

## 2. 训练显存账本与 ZeRO

推理阶段只需加载权重做前向；**训练相对推理多出来的那部分**才是关键——全量预训练/微调要为每个参数额外维护梯度与优化器状态。以 7B 模型 + Adam + 混合精度为例：

- **模型权重**：FP16/BF16，$7\times10^9 \times 2\text{B} \approx 14$ GB；
- **梯度**：反向传播同样 16-bit，$\approx 14$ GB；
- **优化器状态（Adam）**：参数提升为 FP32 计算更新，每参数需 FP32 master weights + 一阶动量 $m$ + 二阶动量 $v$，各 4 Bytes，共 **12 Bytes**，总计 $\approx 84$ GB；
- **合计**：$14+14+84 = 112$ GB，加上激活、临时张量与显存碎片，峰值常到 **120GB+**。

抽象成「每参数字节数」：

$$ \underbrace{2}_{\text{参数}} + \underbrace{2}_{\text{梯度}} + \underbrace{12}_{\text{优化器}} = 16\ \text{Bytes/Param} $$

即经典数据并行 + Adam 下，**每个可训练参数约需 16 Bytes**（不含激活）。DeepSpeed 要动的就是这里——**把 16 Bytes 中的冗余复制部分打散分摊到多卡**，每卡就不必承担完整开销，固定 GPU 数下即可训练更大模型。

### 2.1 冗余从哪来：数据并行的「人人一份」

Data Parallel 用 $N$ 张 GPU，**每张都存一份完整的模型状态副本**，各自处理不同数据子集，算完通过 `all-reduce` 汇总梯度统一更新。于是整套状态在每卡复制一份、共 $N$ 份——这就是「冗余」。**ZeRO** 的名字正是针对它：把这些冗余状态在 GPU 间**分片（shard）**，从「人人持有一整份」变成「大家一起拼一份」。

> **All-Reduce 可拆成两步**：**Reduce-Scatter**（把不同部分归约到不同 GPU，每卡只握一部分完整结果）+ **All-Gather**（广播互换，每卡拼出完整副本）。

### 2.2 ZeRO 三个 Stage 在分谁

经典数据并行下单卡开销 $\text{Mem}_{DP} \propto OS + G + P$（优化器状态、梯度、参数）。ZeRO 三个 Stage 逐步把这三类状态打散：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_13_2_2.png" width="80%" alt="ZeRO 显存优化对比图" />
  <p>图 2：ZeRO 各阶段显存占用对比</p>
</div>

- **ZeRO-1（仅分片优化器状态）**：$\text{Mem} \propto P + G + OS/DP$。参数与梯度仍每卡完整，只有 Adam 动量等被切分。优化器状态占总显存约 75%，仅这一步就收益巨大，因此实践中 ZeRO-1 常被视为「最低配置」默认开启。
- **ZeRO-2（OS + G 分片）**：$\text{Mem} \propto P + G/DP + OS/DP$，视并行度可省 8~16 倍显存。
- **ZeRO-3（OS + G + P 全分片）**：$\text{Mem} \propto (P + G + OS)/DP$，理论上单卡显存可压到 $1/DP$，是训练万亿参数的关键。

代入 16 Bytes 分解：

$$ \text{Bytes/Param} \approx \begin{cases} 2+2+12 & \text{Baseline}\\ 2+2+\frac{12}{DP} & \text{ZeRO-1}\\ 2+\frac{2}{DP}+\frac{12}{DP} & \text{ZeRO-2}\\ \frac{2}{DP}+\frac{2}{DP}+\frac{12}{DP} & \text{ZeRO-3} \end{cases} $$

`DP=64` 时 ZeRO-3 可把每参数显存从 16 Bytes 压到约 **0.25 Bytes**（不含激活与碎片）——这就是论文能在 64 张 V100（16GB）上训练万亿模型的底层逻辑。

论文实测（表 13-2 摘选，单位 GB）：1T 模型不用 ZeRO 需单卡 **16TB**；ZeRO-3 + 1024 张 GPU，单卡仅 **15.6 GB**——现有 16/32GB 显卡集群即可训练万亿模型。7.5B 模型在 64 卡上，ZeRO-1/2/3 分别为 31.4 / 16.6 / **1.9** GB。

### 2.3 通信开销与「线性加速比」的平衡

分片有显存红利，也有通信代价（论文以参数量 $\Psi$ 分析）：

- **ZeRO-1 / ZeRO-2**：通信量 **$2\Psi$**，与标准数据并行同阶——等价于把 `all-reduce` 换成 `reduce-scatter + all-gather` 的组合，量级不变、形态变化；
- **ZeRO-3**：通信量 **$3\Psi$**，约标准 DP 的 **1.5 倍**——前向/反向需临时 all-gather 当前层参数、再叠加梯度 reduce-scatter 与参数同步；换来的是参数级全分片、单卡显存随 GPU 数近似按 $1/DP$ 下降。

ZeRO 的工程本质就是在**显存、带宽、计算时间**三者间找平衡点。省下的显存常允许用**更大的 Batch Size**，显著提升算术强度，部分场景甚至能观察到**超线性加速**。

### 2.4 ZeRO-R：别让碎片偷走显存

即使参数/梯度/优化器都分片了，显存峰值仍可能被其他因素「偷走」：激活缓存、临时张量、显存碎片、不均匀 layer 配置导致的部分 GPU 负载不均。**ZeRO-R** 从三维度补充：

- **分区激活检查点（$P_a$）**：结合 Activation Checkpointing，在模型并行中删除激活复制冗余，只留分片检查点、需要时 all-gather 重建，激活显存随 MP 并行度线性下降；
- **恒定大小缓冲区**：把大量临时张量重定向到可复用的大 buffer，避免频繁小块分配/释放造成碎片；
- **显存碎片整理**：更智能地分配可用区域，长时间训练中保持有效比例。

这些看似「工程小技巧」，往往决定你是**刚好能跑起来**还是**训练到一半 OOM**。

### 2.5 ZeRO-Offload：把 CPU 内存拉进来

现实中 GPU 显存常很有限（16~24GB），CPU 内存反而「富裕」。**ZeRO-Offload** 把最占显存的 **FP32 master weights 与优化器状态**（及部分配置下的梯度缓冲）卸载到 CPU 内存，由 CPU 负责更新；GPU 专注前向/反向计算：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_13_2_3.png" width="90%" alt="ZeRO-Offload 数据流示意图" />
  <p>图 3：ZeRO-Offload 数据流与计算划分</p>
</div>

GPU 侧是计算密集的 FWD-BWD Super Node，CPU 侧是显存密集的 Update Super Node；一次 step 在 CPU↔GPU 间交换的数据量级通常拆解为：梯度约 `2M`（FP16）+ 回传参数约 `2M`（FP16）≈ `4M`，而 FP32 master 与优化器状态常驻 CPU。

**单卡下的一次 step**：

1. **GPU 前反向**：保留 FP16/BF16 参数副本算前向与 loss、反向得梯度，再按配置把梯度（或分片）经 PCIe/NVLink 拷到 CPU，释放显存并备好更新；
2. **CPU 更新**：利用刚拷来的梯度更新 CPU 上的 FP32 master 与 Adam 状态，得到新参数。为防 CPU 优化器成瓶颈，采用高度优化的 CPU Adam（SIMD + 多线程）；并用**单步延迟更新（DPU）**让 CPU 的参数更新与 GPU 下一步计算重叠执行；
3. **参数回传**：把更新后的 FP16/BF16 计算副本同步回 GPU 进入下一步（实际系统分层/分 bucket 同步并尽量与计算重叠）。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_13_2_4.png" width="90%" alt="单步延迟更新机制" />
  <p>图 4：单步延迟更新机制（One-Step Delayed Parameter Update）</p>
</div>

图 4 用「计算流 vs 交换流」展示 overlap：Step i 梯度产生后 CPU 开始更新，GPU 可更早进入 Step i+1。GPU 可能用旧一步参数计算，但实验证明几乎不影响收敛。

效果：单卡 V100（32GB）可训练规模提升到 **10B 量级（论文示例最高约 13B）**，并保持较高吞吐。多卡时不会让每卡各自与 CPU 大规模同步（易撞 PCIe/主机内存带宽瓶颈），而是结合 **ZeRO-2 梯度分片**：先 Reduce-Scatter 把梯度切 $N$ 份 → 每卡只 Offload 自己那份 → CPU 并行更新对应分片 → 回传后 All-Gather 同步。「先分片、再 offload」使 CPU↔GPU 交换开销不随卡数线性膨胀。

### 2.6 ZeRO-Infinity：把 NVMe 也纳入内存池

ZeRO-Offload 把 CPU 内存纳入训练资源，**ZeRO-Infinity** 更进一步把 **NVMe SSD 甚至远程存储**纳入统一「内存池」，构建三层异构体系：

- **GPU HBM**：Fast 层，容量最小、带宽最高（几百 GB/s），只放当前参与计算的参数 shard 与激活；
- **CPU DRAM**：Medium 层，容量较大、带宽较低，放近期访问的参数、优化器状态与激活片段；
- **NVMe/SSD**：Slow 层，容量最大、延迟最差，放远期访问或备用的状态。

Infinity Offload Engine 自动管理三层流动，实现**全量 Offload**——只要 NVMe 够大，理论上可存任意规模状态。

> **NVMe** 是专为 SSD 设计的高速协议，直接走 PCIe，相比为机械硬盘设计的 SATA 吞吐极高、延迟极低。

NVMe 最大的挑战是带宽（PCIe 3.0/4.0 远慢于 HBM）。**Bandwidth-Centric Partitioning** 用 **All-Gather** 让所有 GPU **同时**从 CPU/NVMe 拉不同分片，有效加载带宽随 GPU 数**线性增加**——64 卡集群可利用 64 条 PCIe 聚合带宽掩盖 NVMe 慢的问题。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_13_2_5.png" width="90%" alt="ZeRO-Infinity 带宽为中心的切分示意图" />
  <p>图 5：ZeRO-Infinity 带宽为中心的切分</p>
</div>

**突破单层限制**：万亿模型的**单个层**也可能超出单卡显存。**Memory-Centric Tiling** 把巨大 Operator（如超大矩阵乘）拆成一系列小 Tiles，每次只加载一个参与计算、算完释放。这类 tiling 能降低对张量并行的刚性依赖，但不等于完全替代模型并行，工程上仍需按算子形态、带宽与吞吐综合选型。

得益于这些技术，ZeRO-Infinity 实现了**自动化模型切分**（初始化阶段即切分与 Offload，防初始化 OOM），扩展性惊人：单台 DGX-2（16 张 V100）即可微调 **1 万亿**参数；512 张 GPU 可训练 **32 万亿**参数——同硬件下比最先进的 3D Parallelism（0.64T）高 **50 倍**。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_13_2_6.png" width="90%" alt="ZeRO-Infinity 扩展性对比图" />
  <p>图 6：ZeRO-Infinity 与 3D 并行在不同 GPU 规模下支持的最大参数量</p>
</div>

## 3. 并行策略与生态

### 3.1 四类并行

ZeRO 解决「模型状态怎么分片」，大规模集群高效训练还需与其他并行配合：

1. **数据并行（DP）**：模型复制到多卡、各处理不同数据子集。实现简单、易扩到几十上百卡，但每卡需完整模型副本（大模型显存开销巨大）——这正是 ZeRO 要打破的限制；
2. **模型并行（MP）**：把计算与参数拆到不同 GPU，工程上更常见**张量并行（TP）**——单层内把大矩阵按列/行切到多卡，多卡协作完成同一算子。优点是细粒度拆分单层算子，缺点是通信频繁、对跨节点带宽敏感（all-gather/reduce-scatter）。**注意区分**：ZeRO-3 虽切分参数，但本质仍是**数据并行**的优化（计算时不拆算子）；TP 是**算子内部**的并行；
3. **流水线并行（PP）**：按层段分配 GPU（48 层 → GPU0 管 1-12、GPU1 管 13-24…）。朴素按层切分不配 micro-batch 会让后续 GPU 长时间空等；PP 用 **Micro-batch** 把大 batch 切小块交错处理填满时间片，并常配 Activation Checkpointing 降激活显存。难点是设计 micro-batch 大小与流水线深度以减少「气泡」；
4. **3D 并行**：实际超大规模训练常同时用 DP × TP × PP。DeepSpeed 与 Megatron-LM 联合在 BLOOM 等模型上实践：**DP 解决「更多数据」，TP/PP 解决「更大模型」，ZeRO + Offload/Infinity 解决「显存不够 & 内存异构」**。

### 3.2 生态其他模块

| 模块 | 核心功能 | 应用价值 |
|---|---|---|
| **DeepSpeed-Inference / Compression** | Kernel 优化（算子融合、kernel injection）、KV Cache 优化、量化/剪枝集成（GPTQ、AWQ、LLM.int8） | 与量化互补——量化「把模型变小」，它「让小/中模型在多卡/多节点跑得更快」，显著提升吞吐 |
| **DeepSpeed-Chat 与 RLHF** | 全流程封装 SFT / RM / RLHF（PPO 等）流水线，内置 ZeRO 与 3D 并行 | 降低大规模对齐训练门槛，解决显存不足、并行复杂、混合调度困难 |
| **DeepSpeed for Science** | 优化稀疏算子、复杂网格等非 Transformer 结构，迁移 LLM 系统经验 | 用有限 GPU 训练超大规模科学模型（分子模拟、气候建模） |

> **补记**：DeepSpeed 的整套设计可以用「账本 → 分片 → 借资源」三步记——先算清每参数 16 Bytes 的账（2+2+12），再用 ZeRO-1/2/3 把优化器状态、梯度、参数依次分片（16 → 2+2+12/DP → … → 16/DP），显存还不够就把 CPU 内存（Offload）乃至 NVMe（Infinity）借进来，最后用 3D 并行解决「更大模型 + 更多数据」。看到任何分布式训练配置（如 HF 的 `deepspeed_stage3.json`），都能对上这三个旋钮。

## 写在最后

「微调量化篇」六篇到此收官：从 PEFT 理论到 LoRA 实战、从 RLHF 对齐到 DPO 落地、从量化压缩到分布式训练。下一系列「模型部署工程」转向把这些训练成果真正上线——从 FastAPI 服务化开始。
