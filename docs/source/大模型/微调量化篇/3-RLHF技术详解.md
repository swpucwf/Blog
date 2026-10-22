# RLHF技术详解

## 1. 从「模仿」到「偏好」：对齐的必要性

SFT 教会模型**模仿**高质量范例，却教不会它理解**人类偏好**。面对「为我的项目写一个既专业又有创意的介绍」这类开放指令，SFT 模型会暴露两个问题：**泛化不足**（只熟数据集里出现过的指令模式）、**对齐不足**（事实正确但风格/语气/安全性/有用性不符合期望）——它只知道「正确答案」，不知道「更好的答案」。

**基于人类反馈的强化学习（RLHF）** 的核心目标就是让模型超越模仿、真正内化人类价值观，输出更符合期望。SFT 与 RLHF 的关键差异：

| 维度 | SFT | RLHF |
|---|---|---|
| 核心目标 | 模仿正确答案（指令遵循） | 对齐人类偏好（有用/无害/诚实） |
| 数据需求 | 高质量 `(指令, 回答)` 对 | 提示词 + 偏好排序/评分 |
| 学习方式 | 填鸭式（拟合分布） | 探索式（试错与反馈） |
| 优化信号 | Token 级交叉熵 | 整句生成质量的标量奖励 |
| 泛化能力 | 局限训练分布 | 可泛化到未见复杂指令 |

InstructGPT 的实践给出了强证据：经 RLHF 对齐后，**13 亿参数**的模型在人类评估中胜过 **1750 亿参数**的原始 GPT-3。让模型「懂」人类偏好，比单纯堆参数更能提升实用性。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_1_1.png" width="60%" alt="InstructGPT 模型与 GPT-3 的人类偏好对比" />
  <p>图 1：InstructGPT 与 GPT-3 的人类偏好对比</p>
</div>

## 2. 通往对齐模型的三大阶段

（1）**基础模型预训练**：万亿级无标注文本上自回归训练，通过预测下一个词习得广泛知识与基础生成能力。此时未做指令/偏好对齐；这一阶段门槛极高（算力 + 超大规模收敛稳定性）。

（2）**有监督指令微调**：用少量（千到几万条）高质量「指令-回答」对做 SFT，教模型理解并执行指令（可结合 QLoRA）。目标是在指令上下文中条件生成，得到可控的「指令模型」。SFT 数据集按目标分两类：

- **任务型**：如 `WizardLM Evol-Instruct 70k`（AI 生成）、`Dolly-15k`（专家编写），单轮「指令-回答」为主，教模型「做事」；
- **对话型**：代表是 **OpenAssistant (OASST)**——1.3 万+ 真实志愿者众包、话题与语言风格多样；核心是**多轮对话树**（完整记录上下文、分支、追问）；还带人工质量评分与多种标签，既能做 SFT 也是训练奖励模型、做 RLHF 的宝贵材料。

一句话：任务型数据集教「智商」，对话型数据集给「情商」。

（3）**基于人类反馈的强化学习**：SFT 模型仍是模仿学习，受标注质量所限「只知其一不知其二」，易幻觉或不安全。RLHF 引入人类价值观作指导信号——构建**奖励模型**模拟人类判别标准，用强化学习让模型在生成中「试错」、探索更符合 **3H 原则（Helpful / Honest / Harmless）** 的回复路径。

## 3. 形式化：文本生成即 MDP

把文本生成建模为**片段式（Episodic）马尔可夫决策过程**，是连接 NLP 与 RL 的桥梁（一个 episode = 从接收提示到生成完整输出）：

- **状态** $s_t$：已生成的上下文 $s_t = (x, y_1, ..., y_{t-1})$；
- **动作** $a_t$：当前步生成的下一个 token $y_t$；
- **策略** $\pi_\theta$：即大模型，$\pi(a_t|s_t)$ 是下一 token 的概率分布；
- **奖励** $R$：整段回答生成完毕后由奖励模型给标量 $R(x,y)$，中间步骤通常为 0。

目标即找最优策略使期望奖励最大：

$$ \max_\pi \mathbb{E}_{x \sim \mathcal{D},\ y \sim \pi(\cdot|x)} [R(x, y)] $$

理解了这层形式化，就明白 PPO 等算法为何能靠「采样试错 + 奖励反馈」调整参数，让生成轨迹更符合人类偏好。

## 4. RLHF 三步法

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_1_2.png" width="70%" alt="RLHF 经典三步法示意图" />
  <p>图 2：RLHF 经典三步法</p>
</div>

### 4.1 训练奖励模型（RM）

RM 是「裁判」：输入 `(prompt, response)`，输出标量分数表示人类偏好程度。InstructGPT 的 RM 数据集约 33k 训练提示（衍生出数量级更大的成对比较样本）。

**收集偏好数据**：准备指令（真实用户请求或标注员设计）→ 让 SFT 模型生成多个回答（InstructGPT 为 4-9 个）→ 人工**排序**（从最好到最差）。排序比打绝对分对人类更友好、更一致。得到偏好数据集 $\mathcal{D} = \{(x, y_w, y_l)\}$，$y_w$ 是更受偏好的 winner、$y_l$ 是 loser。

**训练 RM**：

- 把一条含 K 个回答的排序转换成 $\binom{K}{2}$ 个成对比较（A>B、A>C、B>C…）；
- RM 通常与待优化语言模型结构相似（可小得多），任务不是生成，而是**输出标量分数**；
- 基于 **Bradley-Terry 模型**——人类偏好概率用两回答潜在奖励差经 Sigmoid 建模：

$$ P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l)) $$

负对数似然损失：

$$ \text{loss}(\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log \sigma(r_\theta(x,y_w) - r_\theta(x,y_l))\right] $$

训练后 RM 逼近人类偏好函数，作为自动化评估器为策略优化提供稳定标量奖励。

### 4.2 策略优化：PPO / DPO / GRPO

传统 RL 高方差、不稳定，主流策略优化算法有三类。三者的流程侧重不同：经典 RLHF 依赖奖励模型 + PPO 在线优化；DPO 直接基于偏好数据离线优化；RLVR 用确定性验证器（如代码执行器）结合 GRPO 训练。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_1_3.png" width="90%" alt="不同强化学习对齐范式的对比" />
  <p>图 3：RLHF / DPO / RLVR 三种对齐范式对比</p>
</div>

**（1）PPO 与「对齐税」**：PPO 在最大化奖励的同时用约束项限制新旧策略差异，避免单步更新过大导致崩溃。最常用的 **Clipped Surrogate Objective**：

$$ L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat A_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat A_t\right)\right] $$

其中 $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$ 是新旧策略概率比，$\hat A_t$ 是优势估计，$\epsilon$ 限制更新范围（常取 0.2）。裁剪构建了悲观下界，防止策略更新过激。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_1_4.png" width="60%" alt="PPO 的裁剪代理目标函数" />
  <p>图 4：PPO 的裁剪代理目标函数</p>
</div>

- 当 $\hat A_t>0$ 时，目标随 $r_t$ 增加而增加，但增长被限制在 $1+\epsilon$；
- 当 $\hat A_t<0$ 时，目标随 $r_t$ 增加而减小，但减小幅度被限制在 $1-\epsilon$。

单纯用 PPO 优化奖励，会让模型在某些传统 NLP 任务（如 SQuAD）上性能下降，称为**对齐税（Alignment Tax）**。InstructGPT 的 **PPO-ptx** 变体在优化奖励时混入部分预训练数据梯度：

$$ \text{objective} = \mathbb{E}_{D_{\pi^{RL}_\phi}}\left[r_\theta(x,y) - \beta \log\frac{\pi^{RL}_\phi(y|x)}{\pi^{SFT}(y|x)}\right] + \gamma \mathbb{E}_{D_{pretrain}}\left[\sum_t \log \pi^{RL}_\phi(x_t|x_{<t})\right] $$

第一项是 PPO 主目标（奖励 - 每 token KL 约束，抑制偏离参考模型 $\pi^{SFT}$ 的灾难性漂移），第二项是在预训练语料上的最大似然学习（保留通用能力、缓解对齐税），$\beta$ 控 KL 强度、$\gamma$ 控 ptx 权重。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_1_5.png" width="60%" alt="PPO-ptx 缓解对齐税" />
  <p>图 5：PPO-ptx 缓解对齐税</p>
</div>

图中 PPO-ptx（红）在 SQuADv2、DROP 等任务上显著优于纯 PPO（橙）。

**（2）DPO：绕过奖励模型**。PPO 流程复杂（要训奖励模型、价值模型，还要在线采样），成本高、不稳定。DPO 证明可以直接从偏好数据 $\mathcal{D}=\{(x,y_w,y_l)\}$ 推出最优策略——把问题转化为分类任务，**不显式训 RM、不在线采样**：

$$ \mathcal{L}_{DPO}(\pi_\theta;\pi_{ref}) = -\mathbb{E}_{\mathcal{D}}\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right] $$

$\pi_{ref}$ 是参考模型（通常是 SFT 模型）。梯度更新有一直观解释：给**被隐式奖励模型错误排序的样本**（认为 $y_l$ 比 $y_w$ 好）更大权重，集中修正判断失误。隐式奖励定义为：

$$ \hat r_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} $$

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_1_6.png" width="60%" alt="DPO 与 PPO 的 Reward-KL 效率前沿对比" />
  <p>图 6：DPO 与 PPO 的 Reward-KL 效率前沿对比</p>
</div>

在 IMDb 情感生成任务的 Reward-KL 效率前沿上，DPO（黄点）在各 KL 值都取得最高期望奖励，优于 PPO 等基线。凭借简单稳定高效，DPO 及其变种正迅速成为 RLHF 新范式。

**（3）GRPO：为推理任务去 Critic 化**。DPO 主要用于偏好对齐；数学、代码等**带可验证奖励（RLVR）** 的场景更需要在线探索来寻找正确路径。RLVR 用确定性验证器（如代码测试用例 $v(x,y)\in\{0,1\}$）取代主观奖励模型——强反馈消除 Reward Hacking 中「欺骗人类」的可能，并驱动模型通过自我博弈涌现思维链。但 PPO 要维护与策略同规模的价值网络（Critic），千亿模型显存开销巨大。GRPO（DeepSeekMath, 2024）的关键创新是**去 Critic 化**，用组内统计量替代基线估计：

- **组采样**：对同一提示 $x$ 用旧策略采样一组输出 $\{y_1,...,y_G\}$；
- **组内优势**：用奖励模型/验证器打分，直接用组内奖励均值作动态基线，标准化得优势：

$$ A_i = \frac{r_i - \mathrm{mean}(\{r_1,...,r_G\})}{\mathrm{std}(\{r_1,...,r_G\})} $$

- **KL 正则**：与 PPO 在奖励里扣 KL 不同，GRPO 把 $D_{KL}(\pi_\theta\|\pi_{ref})$ 直接加进损失作正则，保持优势估计的纯粹性。

由此 GRPO 既保留 PPO 的在线探索能力（适合推理），又享受接近 DPO 的低资源消耗，成为 DeepSeek-R1 等推理模型背后的关键技术。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_1_7.png" width="90%" alt="基于强化学习训练的推理模型" />
  <p>图 7：基于强化学习训练的推理模型时间线</p>
</div>

**算法选择指南**：

- **PPO**：需要成熟 RL 框架、或要对模型行为做**极细粒度控制**（复杂指令遵循、多轮对话策略）时首选；显式 Critic 能处理更复杂奖励，但计算成本高；
- **DPO**：**样本稀缺**或**追求高训练效率**时的利器，离线偏好数据直接优化，适合快速迭代/资源有限（对话风格微调、摘要）；上限可能不如 RL 方法；
- **GRPO**：**奖励常在序列末尾给出**（数学答案、代码运行结果）的场景最佳，组采样 + 组内优势、去 Critic 省显存，适合需要在线探索涌现 CoT 的复杂推理任务。

## 5. 实际效果与「不完美」的真相

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_1_8.png" width="60%" alt="InstructGPT 在 TruthfulQA 上的真实性表现" />
  <p>图 8：InstructGPT 在 TruthfulQA 上的真实性表现</p>
</div>

以 InstructGPT 为例，RLHF 效果显著但复杂：

- **提升真实性**：TruthfulQA 等基准上生成真实且信息丰富答案的频率约提升一倍，封闭问答中「捏造事实」比例减半（注意：原论文报告 1.3B 的 PPO-ptx 在 TruthfulQA 上略低于同尺寸 GPT-3）；
- **降低有害性**：被明确要求「尊重地」回答时，有毒输出比 GPT-3 减少约 25%；
- **偏见仍未改善**：Winogender 等社会偏见数据集上无明显提升；
- **仍会犯错**：会盲从错误前提（如回答「为什么要饭后吃袜子」），或在简单问题上过度含糊其辞。

## 6. 实践挑战与前沿

RLHF 的挑战（综述系统梳理）：

- **奖励过拟合（Reward Hacking）**：模型学会钻奖励模型漏洞——靠冗长、避而不答、「听起来不错」的无意义内容骗高分，需在数据与训练中仔细约束；
- **评估困境**：如何准确评估「与人类价值观对齐」是开放问题；基准难覆盖偏好广度，LLM-as-a-Judge 也有偏见与提示敏感性问题；
- **多模态对齐**：如何把 RLHF 有效扩展到视频/音频，解决视觉幻觉、时序理解等新问题；
- **文化与价值观对齐**：主流偏好数据来自单一文化背景，模型可能不理解多样化价值观、甚至放大偏见；
- **效率与成本**：PPO 类流程昂贵，轻量对齐（如推理时对齐）与更高效算法是关键。

前沿方向：

1. **RLAIF**：用更强 AI 产生偏好信号，补充或替代人类标注，降成本、缓偏见；
2. **迭代式后训练**：如 LLaMA 3，把「采样 → 偏好标注 → DPO/PPO 训练」做成循环，上一轮最优模型作下一轮数据生成器，持续自我进化；
3. **从对话到推理**：o1、DeepSeek-R1 用 RLVR + 长 CoT，让后训练不止对齐偏好，更通过强化学习探索复杂推理路径、涌现「慢思考」——RLHF 正从「对齐工具」演变为激发深层智能的手段。

> **补记**：这一篇的骨架可以压成一张「信号从哪来」的对照表——SFT 信号来自标准答案（交叉熵）；PPO 信号来自**学出来的**奖励模型（还要 Critic 估基线）；DPO 把奖励模型**解析地消掉**，信号直接来自「偏好对 + 参考模型」；GRPO 把 Critic 也去掉，用**组内相对排名**当基线。四条路线的取舍本质都是「奖励信号的可信度 vs 训练成本」，理解这一点，再看任何新对齐算法都能定位它把哪一项换掉了。

## 写在最后

理论之外，对齐如何落到一台消费级显卡上跑通？下一篇[LLaMA-Factory DPO实战](4-LLaMA-Factory-DPO实战)用 LLaMA-Board 零代码完成一次 Qwen2.5 的 DPO 训练，并对比训练前后的行为风格变化。
