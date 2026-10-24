# OneRec：端到端生成式推荐

## 1. 把「挑选」换成「生成」

级联架构的三个结构性问题是老问题：算力耗在通信与存储、各阶段目标冲突、与 LLM 的技术栈脱节。OneRec 的答案是：**不再从候选集里挑，而是直接生成推荐序列**。

从 V1 到 V2 的演进，正好展示了一个端到端系统从能跑通到能规模化的两次关键修正。

## 2. V1：语义 ID 与四通路编码

### 2.1 物品怎么被「说出」

数十亿视频不可能用原子 ID 生成。V1 给每个视频 3 个语义 token，词表 \(N_t\)，编码空间 \(N_t^{L_t}\)，远大于实际物品数，保证覆盖的同时引入更多参数。

生成分两步：

**协同感知的多模态表示**。标题、标签、ASR、OCR、封面、采样帧经 VLM（miniCPM-V-8B）编码为 1280 个 token 向量，QFormer 压成 4 个查询向量。但纯内容特征学不到协同信号，于是加入**物品对对比学习**：

$$
\mathcal{L}_{I2I} = -\frac{1}{|\mathcal{B}|}\sum_{(i,j)\in\mathcal{B}}\log\frac{\exp(\mathrm{sim}(M_i,M_j)/\tau)}{\sum_{(i',j')\in\mathcal{B}}\exp(\mathrm{sim}(M_i,M_{j'})/\tau)}
$$

再配一个标题生成辅助任务防止表示退化。

**RQ-Kmeans 量化**。在残差上做 K-means，逐层得到 3 级 ID。选 K-means 而不是 RQ-VAE，理由和前一篇一样：解耦、易维护、新物品只需检索。

### 2.2 编码器：四条时间尺度

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/funrec_onerec_encoder_decoder.png" width="56%" alt="OneRec-V1 结构" />
  <p>图 1：四通路编码 + 解码器生成语义 ID</p>
</div>

| 通路 | 长度 | 内容 |
|---|---|---|
| 用户静态 | 1 | ID、年龄、性别等画像 |
| 短期行为 | 20 | 最近交互，含作者、标签、时长、互动标签 |
| 正反馈行为 | 256 | 高参与度交互 |
| 超长期历史 | 128 | 10 万条历史 → 分层 K-means 压到 2000 → QFormer 128 查询 |

超长期这条通路是 V1 的创新：用户可能有十万条记录，直接算会爆。先用分层聚类（簇数取 \(\lfloor\sqrt[3]{|D|}\rfloor\)）压到 2000 个代表物品，再用 128 个可学习查询做交叉注意力，得到紧凑表示。

四路拼接后过 \(L_{enc}\) 层 Transformer，编码器输出维度 \(1+L_s+L_p+128\)。

解码器是标准的「因果自注意力 + 交叉注意力 + MoE FFN」，损失就是语义 ID 的下一 token 预测。

## 3. V1 的三层奖励

预训练只是在模仿过去系统的曝光分布，上限被传统推荐锁死。要突破就得引入奖励。

**P-Score（偏好分）**。传统做法是人工权重把 CTR、LTR、VTR 加权成一个分，既不精准也不个性化。V1 用 SIM 架构为每个目标建独立塔，各塔用自身标签算 BCE 作辅助任务，隐状态拼接后过 MLP 输出 P-Score。不同目标权重 \(w^{xtr}\) 可调，并且能针对具体用户动态调整。

**格式奖励**。编码空间远大于物品集，推理可能生成映射不到真实物品的非法序列。加入 RL 后这个问题会恶化，原因是**挤压效应**：对负优势样本做优化时，概率质量被压向当前最优输出，部分合法 token 的概率被压到与非法 token 相近，模型分不清了。

解法是随机抽部分生成样本做合法性强化：合法样本优势记 +1，非法样本直接丢弃——不惩罚，避免挤压。

**工业对齐**。级联架构里，生态、冷启动、商业化只能挂在某一阶段打补丁，容易陷入恶性循环。端到端后，这些目标可以直接进奖励。以病毒内容为例，占比超过最优值 \(f\) 时对 P-Score 降权：

$$
r_i' = \begin{cases} r_i & o_i\notin I_{viral}\\ \alpha r_i & o_i\in I_{viral}\end{cases},\quad \alpha\in(0,1)
$$

实验报告病毒内容曝光下降 9.59%，核心指标稳定。

**ECPO**。在 GRPO 基础上做早期裁剪：负优势样本的策略比率原本可以任意大，容易梯度爆炸；ECPO 用 \(\delta\) 给负优势样本的比率设上界，既稳定又保留负样本的作用。

## 4. V1 暴露的两个瓶颈

- **算力分配失衡**：Encoder-Decoder 里绝大部分算力花在上下文编码，真正产生梯度的目标 token 解码占比极低；
- **RL 的采样效率与 reward hacking**：奖励模型是代理指标，容易被钻空子。

V2 分别从架构与算法两侧修。

## 5. V2：Lazy Decoder-Only

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/funrec_onerec_lazy_decoder_only.png" width="56%" alt="Lazy Decoder-Only" />
  <p>图 2：上下文只算一次 KV，解码器只处理目标 token</p>
</div>

核心哲学一句话：**把算力集中到对损失有贡献的目标 token 上**。

**Context Processor** 把异构用户特征统一转成解码器要的 KV 对：

$$
d_{context} = S_{kv}\cdot L_{kv}\cdot G_{kv}\cdot d_{head}
$$

沿特征维切成 \(L_{kv}\) 组，每组 RMSNorm 后生成对应的 K/V。关键在于：**这些 KV 在整个前向中对同一上下文不变**，可以被所有解码器层共享。实验显示即使极致共享（\(L_{kv}=1, S_{kv}=1\)）性能也没明显下降。

**Lazy Decoder Block** 不把上下文当序列，而当作静态条件，只通过交叉注意力访问。训练时输入只有 3 个 token：`[BOS, s¹, s²]`。

每层三步：

1. **Lazy Cross-Attention**：不做 KV 投影，直接用预算好的 KV；第 \(l\) 层取第 \(\lfloor l\cdot L_{kv}/N_{layer}\rfloor\) 组；用 GQA 让多个查询头共享 KV 头；
2. **Causal Self-Attention**：在目标物品的几个语义 token 之间自回归；
3. **FFN / MoE**。

「Lazy」的另一层含义是：**只在目标 token 位置算损失**，不再对整条序列每个位置都算下一 token 预测。

**效率对比**：

| 架构 | 参数量 | 计算量 (GFLOPs) | 收敛损失 |
|---|---|---|---|
| Encoder-Decoder (1:1) | 1B | 296.36 | 3.28 |
| Lazy Decoder-Only | 1B | 18.89 | 3.27 |

计算量降 **94%**，训练资源省 **90%**，损失几乎不变。这是端到端方案能规模化的前提。

## 6. Scaling 与用户反馈 RL

**Scaling**：从 0.1B 扩到 8B，损失呈幂律：

$$
\hat{L}(N) = E + \frac{A}{N^\alpha},\quad E=3.13,\ A=3660,\ \alpha=0.489
$$

| 规模 | 参数量 | 收敛损失 |
|---|---|---|
| Dense | 0.1B | 3.57 |
| Dense | 1B | 3.27 |
| Dense | 4B | 3.20 |
| Dense | 8B | 3.19 |
| MoE | 4B（激活 0.5B） | 3.22 |

MoE 那行值得注意：4B 总参、只激活 0.5B，损失 3.22 已优于 2B 密集模型的 3.23，算力却只相当于 0.5B。算力受限时的性价比路径。

**用真实反馈替代奖励模型**。短视频里播放时长是最密集的反馈，且与 App 停留、7 日留存高度相关。但原始时长有偏：长视频天然累积更久。V2 做时长感知的奖励塑形：

1. 按对数分桶 \(\mathcal{F}(d)=\lfloor\log_\beta(d+\epsilon)\rfloor\) 处理长尾；
2. 算目标时长在同桶用户历史中的百分位 \(q_i\)；
3. 前 25% 记 \(+1\)，明确负反馈记 \(-1\)，其余过滤为 0。

**GBPO**。V2 发现传统裁剪对策略比率为 1 的样本失效——它们被当作「稳定」而不裁剪，但负样本仍会炸梯度。看梯度：

$$
\frac{\partial \mathcal{J}^i}{\partial\theta} = -A_i\cdot\frac{1}{\pi_\theta}\frac{\partial\pi_\theta}{\partial\theta}
$$

\(\pi_\theta\) 越小梯度越大。正样本概率小意味着提升空间大，大梯度合理；负样本概率小意味着抑制空间有限，大梯度就会过拟合甚至崩。

GBPO 用 BCE 的稳定梯度去界定 RL 梯度，并把参照策略改写为：

$$
\pi'_{old} = \begin{cases}\max(\pi_{old},\mathrm{sg}(\pi_\theta)) & A_i\ge0\\ \max(\pi_{old},1-\mathrm{sg}(\pi_\theta)) & A_i<0\end{cases}
$$

两个好处：保留所有样本的梯度（鼓励探索），同时让梯度有界（训练稳定）。

## 7. V1 到 V2 的三条经验

- **先跑通再谈效率**：V1 证明了端到端可行，V2 才解决 90% 的算力浪费；
- **奖励要贴近真实反馈**：奖励模型是代理，能被 hack；直接用时长分桶百分位更稳；
- **端到端不等于堆大模型**：Lazy 结构 + MoE 让 8B 模型的计算量落在可接受范围。

## 写在最后

OneRec 解决的是「推荐」这一条链路。下一篇看搜索：OneSearch 面对的是查询与文档的语义匹配、以及更严的延迟约束，它的端到端生成要走另一条路。
