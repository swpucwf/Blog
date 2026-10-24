# RankMixer 与 OneTrans：把推荐模型做成 GPU 的第一类公民

## 1. MFU 差十倍，问题不在参数

传统 DLRM 在 GPU 上的 MFU 只有 4～5%，而 LLM 能到 40～60%。这个差距意味着：即使加参数，新增算力大多浪费在低效访存上。

时间都去哪了？一个典型 DLRM 的分解：

| 环节 | 占比 | 性质 |
|---|---|---|
| Embedding lookup | ~40% | memory-bound |
| 小 kernel 操作 | ~35% | MFU < 10% |
| Kernel launch 开销 | ~20% | 纯开销 |
| 有效 GEMM | ~5% | 唯一吃到 Tensor Core 的部分 |

根因是这套架构继承自 CPU 时代：核心操作访存大于计算、计算图由多个手工模块串起来、小规模向量运算用不上 Tensor Core。

RankMixer 的原则很直接：**从硬件特性反推架构**——所有核心操作都应是矩阵乘法、计算图尽量简洁、同时保住推荐需要的表达能力。

## 2. RankMixer：不要 attention，只要 mixing

自注意力复杂度 \(O(T^2D)\)，瓶颈在 token 两两相似度矩阵。但推荐需要的其实是**信息混合**，不是基于相似度的动态加权——「年轻用户在一线城市更喜欢科技品」这种高阶交互，本质是把多个 token 的信息融成新表示，不必显式算 pair 相似度。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/funrec_rankmixer_arch.png" width="56%" alt="RankMixer 架构" />
  <p>图 1：Token Mixing + Per-Token FFN 堆叠</p>
</div>

**Token Mixing** 两步：

1. 多头切分：把每个 token 切成 H 个头；
2. 按头重排：把各 token 的第 h 个头拼成一个长向量。

数据组织从「按 token」变成「按 head」，每个 head 内部自然混入了所有 token 的信息。实现上取 \(H=T\)，混完后 token 数不变，残差连接照旧。

复杂度从 \(O(T^2D+TD^2)\) 降到 \(O(TD^2)\)，且没有 softmax 的全局归约，少了 reduction kernel。

**不显式算相似度，怎么保证学到交互？** 靠堆叠。单层是 feature-level mixing，第 1 层输出已含所有 token 的一阶信息，第 2 层再混就是二阶……L 层可建模 L 阶多项式交互，实际 6～12 层足够。

## 3. Per-Token FFN：参数隔离

标准 FFN 所有 token 共享权重，前提是 token 在同一语义空间。推荐里完全不同：用户 ID 是隐式偏好、类目是粗分类、点击率长尾、时间戳有周期——强行共享会互相干扰。

于是每个 token 一套 FFN：

$$
v_t = W^{t,2}_{pffn}\cdot \mathrm{GELU}(W^{t,1}_{pffn}s_t + b^{t,1}_{pffn}) + b^{t,2}_{pffn}
$$

参数量变为 \(2TkD^2\)（共享版是 \(2kD^2\)），但 **FLOPs 不变**——共享 FFN 本来也要对 T 个 token 各算一次。增加的是「专用」参数，学习效率更高。

与 MMoE 的区别值得记清：MMoE 的多个 expert 看同一个输入、由门控加权组合；Per-Token FFN 是每个 token 独立输入、独立变换，没有门控。前者是「共享输入 + 动态选专家」，后者是「参数按 token 隔离」。

跨 token 交互交给 Token Mixing，深度建模交给 Per-Token FFN，两者交替堆叠。

## 4. Sparse MoE：扩参不增延迟

扩到十亿、百亿参数时，直接加宽加深会让推理延迟同比上涨。MoE 的思路是每个样本只激活少数专家。

但推荐做 MoE 有三个坑：

- 高维稀疏特征的组合空间是指数级，样本缺乏清晰 cluster，门控难学稳；
- 数据极不均衡，头部用户贡献一半样本，早期路由偏差会让少数专家过载、其余闲置；
- 训练负载均衡不代表推理均衡，某些专家可能成为延迟瓶颈。

RankMixer 用两招应对：

**ReLU Routing**：不用 Top-k + softmax，而是 \(\mathrm{ReLU}(h(s_i))\)，每个 token 可激活不同数量的专家，高信息量 token 自动激活更多。加 L1 正则 \(\mathcal{L}_{reg}=\sum_i\sum_j G_{i,j}\) 控制平均激活数。

**DTSI-MoE**：训练用 dense router 激活全部专家保证训充分，推理用 sparse router 只激活少数；两个 router 同时训练，但只有推理 router 受稀疏正则约束。既训得充分，又跑得便宜。

负载均衡靠 \(\mathcal{L}_{reg}\) 软约束：某专家总激活量过大时，梯度会压低它的门控概率。相比 capacity 硬限制，软约束不会把样本硬塞给次优专家。

**效果**：MFU 从 4～5% 提到 **45%**。Token Mixing 与 PFFN 占约 85% 计算时间，全是可吃满 Tensor Core 的大 GEMM。

## 5. OneTrans：把序列与特征交互合成一个 Transformer

RankMixer 解决了单模块效率，但推荐架构还有一层碎片化：主流的 encode-then-interaction 先用 DIN/LONGER 把序列压成定长向量，再和静态特征拼起来过交互模块。

两个后果：序列被压缩，静态特征没法在编码阶段起作用；两个模块独立执行，享受不到 KV Cache、FlashAttention 这些 LLM 系统优化，也难以形成统一的 scaling law。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/funrec_onetrans-compare.png" width="56%" alt="架构对比" />
  <p>图 2：encode-then-interaction vs 单一 backbone</p>
</div>

OneTrans 的做法是用**一个 Transformer backbone** 同时做序列建模与特征交互。

**统一 Tokenization**：

- S-tokens（序列）：每种行为序列先用行为专属 MLP 对齐到统一维度，再合并。有时间戳就按时间交错并加行为类型标识（timestamp-aware，实验更好）；没有时间戳就按意图强度排序并用 `[SEP]` 分隔；
- NS-tokens（非序列）：所有特征 concat 后过一个 MLP，再切成 \(L_{NS}\) 个 token（Auto-Split），避免人工分组的随意性。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/funrec_onetrans-arch.png" width="56%" alt="OneTrans 架构" />
  <p>图 3：统一 token 序列 + 混合参数化 + 金字塔堆叠</p>
</div>

## 6. 混合参数化：该共享的共享

统一序列后立刻遇到异构性问题：S-tokens 都在物品空间，同质性强；NS-tokens 分别来自人口统计、价格、文本，语义空间完全不同。全共享会参数冲突，全独立则参数爆炸。

于是：

$$
W^\Psi_i = \begin{cases}W^\Psi_S & i\le L_S\\ W^\Psi_{NS,i} & i>L_S\end{cases}
$$

S-tokens 共享一套 Q/K/V 与 FFN，每个 NS-token 独立。参数量 \(O(L_{NS}d^2+d^2)\)，而 RankMixer 的 Per-Token FFN 是 \(O(Td^2)\)。由于 \(L_{NS}\ll L_S\)，这个共享不是妥协——行为序列同质，共享反而学得更高效。

**混合因果注意力**的三种信息流很值得琢磨：

1. **S 侧因果**：每个 S-token 只看之前的 S-token。有时间戳时是时间因果；按意图排序时，因果让「下单」的信息流向「点击」，实现强信号过滤弱信号；
2. **NS 侧全局**：NS-token 可看全部 S-tokens 与之前的 NS-tokens。于是「物品类目」token 能自动 attend 到历史上所有同类目点击，等价于自己学出「用户对该类目的历史偏好」——这正是前面提到的交叉特征；
3. **支持金字塔**：方向性让信息自然向尾部聚集。

配套用 Pre-norm + RMSNorm：S/NS 数值尺度差异大，Post-norm 容易让 attention logits 失衡。

## 7. Pyramid Stack：逐层丢 token

因果注意力有一个副产品：越靠后的 token 汇聚的信息越多，NS-tokens 在末尾会积累整个序列。

于是每层只让尾部 \(L'\) 个 token 产生 query，而 K/V 仍来自全部 L 个 token：

$$
q_i = W^Q_i x_i,\quad i\in\mathcal{Q}=\{L-L'+1,\dots,L\}
$$

层间让 \(L'\) 递减（如 1190 → 595 → 297 → … → 12），形成金字塔。

两个收益：

- **渐进式蒸馏**：浅层学局部模式（相邻物品相似），深层在压缩后的 token 上学全局模式（长期兴趣漂移），最终信息全汇入 NS-tokens；
- **算力下降**：attention 从 \(O(L^2d)\) 降到 \(O(LL'd)\)，FFN 从 \(O(Ld^2)\) 降到 \(O(L'd^2)\)。

为什么能丢？因为 LLM 要输出每个位置的预测，必须保留全长；推荐只要最终排序分，中间 token 可以逐层收掉——**因果方向保证了尾部已经看过前面**。

## 8. Cross-Request KV Caching：真正的端到端红利

一次请求返回几百个候选，它们的**用户侧特征完全相同**，只有物品侧不同。传统架构里序列模块可复用，但交互模块必须逐候选重算。

统一 Transformer 天然支持两阶段：

- **Stage I（每请求一次）**：算完所有 S-tokens 每层的 K/V 并缓存；
- **Stage II（每候选一次）**：只算该候选的 NS-tokens，用缓存做 cross-attention。

复杂度从 \(O(C\cdot L_S)\) 降到 \(O(L_S+C\cdot L_{NS})\)，近似与候选数无关。

再进一步，用户行为序列是 append-only 的，跨请求可复用 KV cache：首次请求算全长并缓存，后续只算新增的 \(\Delta L\)（通常个位数）。信息流刷新场景下，这个收益非常直接。

这也解释了为什么统一架构重要：两个独立模块的中间表示无法跨候选复用，因为输入和参数都不同；只有把两类特征放进同一个 attention 矩阵，S-tokens 的 KV 才可能被所有候选共享。

此外还继承了 FlashAttention-2、BF16 混合精度、激活重计算等 LLM 系统优化。

## 9. 两篇的分工与边界

| 维度 | RankMixer | OneTrans |
|---|---|---|
| 解决的问题 | 单模块 GPU 效率（MFU 4%→45%） | 模块间碎片化（序列 + 交互统一） |
| 核心手段 | Token Mixing、Per-Token FFN、Sparse MoE | 统一 tokenization、混合参数化、金字塔、KV 缓存 |
| 参数策略 | 每 token 独立 FFN | S 共享、NS 独立 |
| 最大红利 | 算得满 | 跨候选 / 跨请求复用 |

两者是叠加关系：RankMixer 让模型算得快，OneTrans 让 pipeline 少重复。组合起来才是「推荐模型成为 GPU 第一类公民」的完整路径。

## 写在最后

ch6 到这里收尾：HSTU 证明能 scale，GenRank 证明可以更省，MTGR 说明不必丢特征，RankMixer 解决硬件效率，OneTrans 打通模块壁垒。下一篇进入端到端生成，先看 OneRec 如何用单一模型替代整条推荐流水线。
