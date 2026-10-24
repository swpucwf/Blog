# 端到端广告：EGA 与 GPR

## 1. 广告比推荐多出来的三件事

推荐只要「推得准」，广告还要「卖得出去、收得回来、广告主愿意继续投」。三条硬约束：

**激励相容（IC）与个体理性（IR）**。广告主是博弈方，会试探规则。IC 要求「如实出价最优」，IR 要求支付不超过竞价：

$$
u_i(v_i;v_i,b_{-i})\ge u_i(v_i;b_i,b_{-i}),\qquad p_i\le b_i
$$

**POI 与创意要联合决策**。同一家餐厅，对美食爱好者给菜品特写、对价格敏感用户给满减券、对家庭用户给环境图。选谁 + 用哪张，必须一起定。

**分配与支付要解耦**。若直接把竞价当生成概率权重并按其支付，就是「赢家诅咒」——广告主会倾向压价。传统 GSP 靠「按下一位支付」解决，但生成式里分配是 Softmax 概率，没有明确的「第二名」。

## 2. EGA：把竞价塞进生成

### 2.1 双模态语义 ID 与概率分解

POI 与创意各用一套 RQ-VAE：3 层、每层码本 1024，各得 3 个 token。用户历史是 (POI, 创意) 对，并带 `ad/organic` 类型标记。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/funrec_ega_pretrain.png" width="56%" alt="EGA 预训练" />
  <p>图 1：双解码器分别生成 POI 与创意</p>
</div>

如果简单把 6 个 token 拼成一条序列自回归，会出现「餐厅 A 的 POI + 健身房 B 的创意」这种错配。EGA 做概率分解：

$$
P(a^{poi},a^{img}\mid S) = P(a^{poi}\mid S)\cdot P(a^{img}\mid a^{poi},S)
$$

先定内容主体，再定呈现方式。实现上是两个解码器：创意解码器的输入包含已生成的 POI token，因此天然知道「该配什么素材」。MTP 模块每步同时监督两个解码器，共享底层表示、加快收敛。

### 2.2 排列感知奖励：广告不是独立的

推荐里常假设 pCTR 独立，广告里不成立，存在三类位置外部性：

- **位置效应**：位置 1 的 CTR 远高于位置 5；
- **相邻效应**：3、4 位都是餐饮广告，用户只看一个；
- **对比效应**：高质量广告后紧跟低质量广告，后者 CTR 被压低。

所以 \(pCTR_i = f(u,i,\mathcal{Y}_{-i},pos_i)\)，必须建模序列依赖。EGA 的奖励模型用 Self-Attention 让每个广告「看到」其他广告，再挂三塔（POI-CTR、创意-CTR、CVR）。相邻同类广告的注意力分散，pCTR 自然被抑制。

### 2.3 Token 级竞价 + POI 级支付

**分配侧**用 token 级竞价引导生成。一个 token 可能对应多个广告，取最大值聚合：

$$
b(a_i^j) = \max(b_1,\dots,b_{N_i})
$$

用 max 而非 avg，是因为高竞价信号不该被低竞价稀释。调整后的概率：

$$
z(a_i^j) = \frac{w(a_i^j)e^{a_i^j}}{\sum_k w(a^{j,k})e^{a^{j,k}}},\qquad w = [b(a_i^j)]^\alpha+\beta
$$

两个超参都有业务含义：\(\alpha\) 控制竞价影响（0 退化为纯兴趣，趋于无穷变成纯竞价排序），\(\beta\) 控制广告与有机内容比例。

**支付侧**单独用一个网络。输入三部分：

1. 生成的 POI 序列表示；
2. **自排除竞价矩阵** \(\mathcal{B}^-=\{b_{-1},\dots,b_{-K}\}\)——按机制设计理论，IC 要求「支付只依赖他人竞价与自己是否被分配」；
3. 期望价值 \(Z\cdot\Theta\)（分配概率 × 排列感知 pCTR）。

输出支付率 \(\hat{p}=\sigma(\mathrm{MLP}(\cdot))\)，支付 \(p_i=\hat{p}_i b_i\)。Sigmoid 天然保证 \(p_i\le b_i\)，即 IR。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/funrec_ega_training.png" width="56%" alt="EGA 训练" />
  <p>图 2：预训练 + 竞价后训练两阶段</p>
</div>

**IC 怎么近似保证**：用 ex-post regret 度量「谎报最多能多赚多少」：

$$
\widehat{\mathrm{rgt}}_i = \frac{1}{N_v}\sum_j \max\left(0,\ u_i(v_i;b'_j,b_{-i})-u_i(v_i;v_i,b_{-i})\right)
$$

然后解约束优化：最大化平台收益 \(\mathrm{Rev}=\sum p_i\hat{r}_i\)，同时把 regret 压到 0。用 Lagrangian 交替更新——固定 \(\lambda\) 训网络，再按 \(\lambda_i\leftarrow\lambda_i+\rho\cdot\widehat{\mathrm{rgt}}_i\) 抬高惩罚。某个广告主越容易靠谎报获利，对应的乘子越大。

### 2.4 后训练三件事

奖励模型（多任务 BCE，训完冻结）→ 非自回归策略梯度（奖励定义为「去掉这个 POI 后平台收益下降多少」，即边际贡献）→ 支付网络（Lagrangian）。三者在后训练阶段交替迭代。

与 OneRec 的差别很清楚：多了竞价项与 IC 约束，奖励模型还必须排列感知；与 GSP 的差别是：不再假设独立、也不再写死支付公式。

## 3. EGA 的三个瓶颈

- **跨场景异构**：信息流要视觉冲击、搜索要高相关、短视频要融入风格，单一 RQ-VAE + Encoder-Decoder 难统一；
- **超长序列**：数万条交互，\(O(L^2)\) 撑不住，采样截断又丢信息，且大量误触噪声；
- **推理无效候选**：Beam Search 会生成预算耗尽、定向不符、地域不符的广告，只能后处理过滤，白增延迟。

## 4. GPR：预训练优先，约束前移

微信生态把这三个问题放大：视频号/朋友圈/公众号/小程序四类场景、数万条行为、100ms 延迟要求。

### 4.1 四类 Token 统一异构序列

| Token | 内容 |
|---|---|
| U | 人口统计、消费能力、兴趣标签 |
| O | 有机内容（短视频 SID、文章表示、朋友圈多模态） |
| E | 环境：时间、地理、设备、当前场景 |
| I | 交互过的广告（POI + 创意 SID） |

四类拼成一条时间线，跨场景行为就连起来了，模型也只需维护一个。

**RQ-Kmeans+** 解决码本坍塌：先用 RQ-Kmeans 初始化（保证每个码字都有样本），再转 RQ-VAE 端到端微调，编码器侧加残差 \(z=\mathrm{Enc}(e)+\alpha e\) 稳定早期训练。码本利用率 65% → 92%，重构误差降 15%。

### 4.2 三层异构解码器

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/funrec_gpr_architecture.png" width="60%" alt="GPR 架构" />
  <p>图 3：理解 → 推理 → 生成 → 评估</p>
</div>

**HSD（意图理解层）** 三个设计：

- **Hybrid Attention Mask**：U/O/E 是 Prompt，内部双向可见；I 是 Target，内部因果；Target 可看全部 Prompt。比一刀切的 causal mask 更合理——用户属性、内容、环境本就该互相解释；
- **Token-Aware Normalization**：四类 token 分布差异大（离散、连续、周期、竞价相关），各自独立 LayerNorm + FFN；
- **MoR（Mixture-of-Recursions）**：同一层递归调用多次，不增参数地加深「思考」。

可选再注入 LLM 生成意图描述作为额外 O-Token，给高价值用户加一层语义。

**PTD（推理生成层）** 走「Thinking → Refining → Generation」：

1. 4 个 Thinking Token 用交叉注意力从意图里筛关键信息（推餐饮就只看美食行为）；
2. Refining 借扩散思路：加高斯噪声再用条件去噪网络重建，强迫模型「二次确认」，生成质量 +2～3%，长序列复杂用户收益更明显；
3. 自回归生成 3 层 SID。

**HTE（价值评估层）** 的特别之处是**每层 token 都出价值估计**，而不是等整条序列生成完：

$$
\text{final\_value} = w_1 pCTR + w_2 pCVR + w_3 eCPM
$$

它有两个用途：Beam Search 剪枝、RL 里的 critic。

### 4.3 价值引导的 Trie Beam Search

约束前移到解码：先按用户画像与投放约束（预算、定向、地域）过滤出有效广告池，把它们的 SID 建成 Trie；生成第 \(l\) 层时只从当前节点的子节点采样，搜索空间从 \(W^3\) 收缩到 \(|\mathcal{X}_{valid}|\)。

束宽也动态化：

$$
B_{next}(a^l)=\max\left(B_{min},\ B_{base}\exp\left(\frac{v^l-\bar{v}}{\tau}\right)\right)
$$

高价值分支探得更深，低价值分支提前收。效果：延迟 150ms → 80ms（−47%），有效候选占比 40% → 95%，Top-1 准确率 +3.2%。

### 4.4 三阶段训练与 HEPO

1. **MTP 预训练**：微信全量行为日志、数十亿用户，最大 8B 参数，不引入竞价；
2. **价值感知微调**：冻结 HSD/PTD，只训 HTE 多任务塔；
3. **HEPO 层次化策略优化**：token 级优势 \(A^l_{token}=v^l-\bar{v}^l\) 与 item 级优势 \(A_{item}=R_{item}-\frac{1}{N}\sum R\) 一起用，按折扣 \(\gamma\) 聚合：

$$
\mathcal{L}_{HEPO} = -\mathbb{E}\left[\sum_l A^l_{token}\log\pi_\theta(a^l) + \beta A_{item}\log\pi_\theta(\text{item})\right]
$$

token 级信号密集、方差小；item 级信号贴近收益。两者结合既快又稳。

线上结果：级联 5 个模型合并为 1 个，延迟 200ms+ → 80ms；GMV 与 CTCVR 均有提升（原文以占位符给出）。

## 5. EGA 与 GPR 的取舍

| 维度 | EGA | GPR |
|---|---|---|
| 竞价 | 深度嵌入生成（token 级竞价 + 支付网络） | 不嵌入生成，价值评估后置 |
| 场景 | 单场景（本地生活） | 多场景统一（微信生态） |
| 序列 | 标准 Transformer | 三层解耦 + Trie 剪枝 |
| 优势 | 机制设计严谨、IC 有理论保证 | 工程效率高、可跨场景迁移 |
| 代价 | 跨场景与长序列吃力 | 架构复杂（代码量约 2 倍）、可解释性弱 |

三条权衡值得记住：复杂度换通用性；预训练成本换零样本迁移（新广告位少量微调即可）；端到端黑盒换取效率，但排障要靠中间层监督（Thinking Token 可视化、HTE 分层价值）。

## 6. 落地判断

- **先确认是否需要 IC**：只有竞价场景才需要支付网络，普通推荐/搜索不需要；
- **约束尽量前移**：预算、定向、地域做成 Trie 比后处理过滤便宜得多；
- **奖励模型必须排列感知**：忽略位置外部性，离线指标会系统性高估；
- **支付网络要可审计**：regret 是近似约束，上线后仍需监控广告主出价分布是否漂移；
- **多场景优先统一 token 体系**：四类 token 的收益主要来自跨场景行为联通。

## 写在最后

推荐、搜索、广告三条链路的端到端方案到齐。下一篇换个角度：不再问「怎么生成得更准」，而问「模型能不能先想再答」——推理型推荐如何把 CoT 与对齐引入推荐。
