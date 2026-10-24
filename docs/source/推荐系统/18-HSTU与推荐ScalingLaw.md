# HSTU 与推荐 Scaling Law

## 1. 推荐为什么一直没能 scale

ResNet 把深度堆到上千层、Transformer 堆到万亿参数，性能都遵循可预测的幂律。推荐系统却是个例外：特征越做越多、模型越堆越大，指标很快见顶。

三个根因：

- **特征瓶颈**：依赖手工统计特征（点击率、平均时长）压缩历史，模型容量上去了，输入信息没上去；
- **架构碎片化**：FM、DCN、DIN、MMoE 异构拼装，单独扩某个模块只能局部改善；
- **item-level 范式**：每个样本只对应一个 \((u,i,a)\) 三元组，每次前向只产出一个监督信号；计算成本还随候选数线性增长。

Meta 的解法是把推荐重新表述为**行为序列的生成问题**。

## 2. 从物品序列到行为序列

传统序列推荐只建模物品：\([\Phi_0,\Phi_1,\dots]\)。但推荐里最有价值的信息是**用户反馈**——点击、点赞、观看时长。

于是把内容和行为交替排列：

$$
[\Phi_0, a_0, \Phi_1, a_1, \dots, \Phi_{n_c-1}, a_{n_c-1}]
$$

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/funrec_dlrms_vs_grs_training_v2.png" width="56%" alt="DLRM 与 GR 对比" />
  <p>图 1：DLRM 与生成式推荐在特征与训练方式上的差别</p>
</div>

按链式法则展开联合分布：

$$
p(\Phi_0)\cdot p(a_0|\Phi_0)\cdot p(\Phi_1|\Phi_0,a_0)\cdots
$$

这一步同时定义了两个任务：**排序**对应 \(p(a_i\mid \Phi_0,a_0,\dots,\Phi_i)\)——看到候选后才预测行为，是 target-aware 的；**召回**对应 \(p(\Phi_i\mid \Phi_0,a_0,\dots,a_{i-1})\)——给定历史预测下一个物品。

一套序列、一个目标，召回与排序统一了。

## 3. 统一异构特征

这是整个方案里最容易被忽略、但最脏的活。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/funrec_dlrms_vs_grs.png" width="36%" alt="架构对比" />
  <p>图 2：DLRM 的多模块拼装 vs HSTU 的单塔序列建模</p>
</div>

**类别特征**（用户、物品、创作者、话题、城市）按时间轴对齐：

1. 以交互历史为主时间线；
2. 慢变特征做**连续同值压缩**——`[张三,张三,张三,李四,李四,王五]` → `[张三,李四,王五]`；
3. 压缩后按时间戳合并进主时间线。

**数值特征**（点击率、平均时长、统计计数）不序列化，因为变化太频繁、序列会爆炸。关键的观察是：**这些数值特征本质上是对类别特征的聚合统计**——「科技话题点击率」就是对「科技类物品点击行为」的统计，而这些基础信号已经在序列里了。

所以假设是：序列足够长、模型足够强，聚合统计可以自己学出来。形式化地：

$$
\mathcal{F}_{GR} = \mathrm{Seq}(\text{sparse features}),\qquad \lim_{n\to\infty}\mathcal{F}_{GR}\approx\mathcal{F}_{DLRM}
$$

代价必须说清楚：消融显示，把 DLRM 也限制成「仅类别特征」后性能明显下降。**低算力场景下手写数值特征仍然值钱**，GR 是「用算力换特征工程」。

## 4. 训练效率：一次前向，多个监督

一条含 \(n_c\) 个交互的序列，长度 \(n=2n_c\)。因果掩码下，一次前向就能同时算出 \(n_c\) 个位置的预测：位置 0 预测 \(a_0\)，位置 2 预测 \(a_1\)……

- DLRM：\(M\) 个样本 → \(M\) 次前向；
- GR：\(M\) 个样本 → \(M/n_c\) 次前向。

**训练效率提升 \(n_c\) 倍**。平均 500 次交互，就是 500 倍——同样的预算可以训大一到两个数量级的模型。这是能 scale 的第一块基石。

## 5. HSTU 的三个架构改动

直接套标准 Transformer 不合适，HSTU 改了三处。

### 5.1 点向聚合替代 softmax

$$
A(X)V(X) = \varphi_2\left(Q(X)K(X)^T + \mathrm{rab}_{p,t}\right)\odot V(X)
$$

$$
\mathrm{Output} = \mathrm{LayerNorm}(A(X)V(X))\odot U(X)
$$

softmax 抹掉了**绝对强度**：用户 A 在科技 10 次、娱乐 1 次，用户 B 科技 100 次、娱乐 10 次，归一化后分布可能几乎一样，但 B 的兴趣强度显然更高。

SiLU 不做全局归一化，每个位置权重独立、总和可以大于 1，模型因此能表达「兴趣很强烈」。排序阶段要预测观看时长、点赞这类深度行为，强度信息不可省。

### 5.2 相对时间偏置

推荐序列的时间是连续且不均匀的：两次交互可能隔几秒，也可能隔几个月。HSTU 的偏置同时看位置差、时间差与 token 类型：

$$
\mathrm{bias}_{i,j} = f(p_i-p_j,\ t_i-t_j,\ \mathrm{type}_i,\ \mathrm{type}_j)
$$

类型区分内容 token 与行为 token——内容对内容的注意力、行为对内容的注意力本就该不同。这让模型能学到「最近更重要」「浏览比点赞衰减更快」这类规律。

### 5.3 简化 FFN

标准 Transformer 的 FFN 中间层是隐层的 4 倍，占了大头参数和激活内存。HSTU 用逐元素门控替代：

$$
\mathrm{Block}(X) = \mathrm{LayerNorm}\left(X + \mathrm{Gate}(X)\odot \mathrm{Attention}(X)\right)
$$

每层激活内存从 33 倍隐层降到 14 倍。工业训练常用数万到数十万的 batch，激活内存是硬瓶颈，这个改动直接决定了能不能训深。

## 6. 两个工程杠杆

**Stochastic Length**：自注意力是 \(O(n^2)\)，超长序列训不动。但用户行为在多时间尺度上有冗余——长期偏好、中期演化、短期场景需求。

于是对长度 \(n\) 的序列以概率 \(p=1-N_\alpha/n^2\) 采样长度 \(N_{\alpha/2}\) 的子序列，否则用全长。\(\alpha=2\) 时计算上界被压到 \(O(N_c^2)\)；取 1.6～1.7 时仍保留一定概率看完整历史，避免遗忘长期依赖。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/funrec_exp_stochastic_length.png" width="56%" alt="Stochastic Length" />
  <p>图 3：不同 α 下的效果，大范围内几乎无损</p>
</div>

双收益：序列稀疏度可到 80% 以上，训练快数倍；随机截断还起了正则作用，泛化反而略好。

**M-FALCON**：排序要给上千个候选打分，逐个拼到历史后面推理，延迟不可接受。三层优化：

1. **Batched Inference**：多个候选拼一次前向，用掩码禁止候选间互相可见；
2. **Microbatching**：候选太多时切成 \(\lceil m/b_m\rceil\) 个微批，让 \((n+b_m)^2\approx 4n^2\)；
3. **KV Caching**：用户历史部分的 K/V 在所有微批中相同，首个微批算完直接复用，后续只算候选增量。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/funrec_m-falcon_inference.png" width="56%" alt="M-FALCON" />
  <p>图 4：批推理 + 微批 + KV 缓存的组合</p>
</div>

组合效果：批推理数十倍，微批与 KV 缓存再 2～4 倍，总计可达数百倍。这是万亿参数能上线的工程基础。

## 7. Scaling Law 的实证

实验维度：序列 512→8192、隐层 256→1024、深度到 24 层。训练算力按 365 天归一化，便于与 GPT-3、LLaMA-2 对比。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/funrec_exp_scaling_pflops.png" width="72%" alt="Scaling 实验" />
  <p>图 5：排序 NE 与召回 HR@100 的幂律关系</p>
</div>

拟合结果与语言模型同形：

$$
\mathrm{HR@100} = 0.15 + 0.0195\ln C,\qquad
\mathrm{NE} = 0.549 - 0.0053\ln C
$$

换算成直观结论：**算力每增 10 倍，HR@100 提升约 4.5 个百分点，NE 下降约 1.2 个百分点**，且在三个数量级范围内稳定成立。

最大配置（8192 序列、1024 维、24 层）参数达 **1.5 万亿**，并已部署到 Meta 多个场景。线上排序核心指标提升达双位数百分比。

## 8. 四层原因，缺一不可

- **范式**：item-level → user-level，解除算力与候选数的线性绑定；
- **架构**：点向聚合、时间偏置、门控 FFN 的针对性设计；
- **工程**：Stochastic Length、M-FALCON、激活内存优化；
- **特征**：统一到序列，端到端学交互，提升参数效率。

这也留下真问题：完全生成式训练是否必需？哪些设计是本质的？多任务、多场景能否沿用同一套？后续的 GenRank、MTGR、RankMixer 正是在回答这些。

## 写在最后

HSTU 证明了推荐可以 scale，也把「生成式」从概念推到了工业部署。下一篇看两条更轻的路径：GenRank 用「预测动作」替代「预测物品」，MTGR 用「用户样本聚合」在保留 DLRM 特征的同时拿到生成式架构的扩展性。
