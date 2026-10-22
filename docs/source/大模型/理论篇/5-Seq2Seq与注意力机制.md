# Seq2Seq与注意力机制

## 1. Seq2Seq：为「不对齐多对多」而生

RNN/LSTM 擅长三类任务：**多对一**（整句压缩成特征向量做分类）、**多对多严格对齐**（输入输出等长、逐词标注，如词性标注/NER）、**一对多**（固定输入生成序列，如图文描述）。但机器翻译属于更棘手的**多对多不对齐**任务——输入输出长度通常不等、元素间没有一对一映射，模型不能「边读边写」，必须先完整接收输入、形成全局理解，再重新组织输出：「我爱人工智能」（3 词）→ "I love artificial intelligence"（4 词）。

2014 年提出的 **Seq2Seq（Sequence-to-Sequence）** 架构用通用的**编码器-解码器（Encoder-Decoder）** 框架解决了序列转换问题，随后在机器翻译、文本摘要、对话系统等领域大获成功。

### 1.1 与自编码器的关系

Seq2Seq 借鉴自编码器但做了关键**泛化**：自编码器的编码器把输入压缩成低维**潜在表示**、解码器将其**重构**回原始输入（训练目标是输出≈输入，用于降维/特征学习/去噪）；Seq2Seq 不再要求输出与输入相同，而是生成一个与输入语义相关的**全新目标序列**——目标从「重构」变成了「转换」。

### 1.2 整体架构

Seq2Seq 的核心思想模拟人类翻译：先完整读完并理解源句、形成综合语义表示，再基于它逐词生成译文。

<div align="center">
  <img src="{{IMG:4_1_1.svg}}" width="80%" alt="Seq2Seq详细工作流程" />
  <p>图 1：基于 LSTM 的 Seq2Seq 工作流程（来源：base-llm 教程）</p>
</div>

**编码器**负责「阅读和理解」：RNN/LSTM 逐词读入 $x_1...x_T$，每步按 $h_t = f(h_{t-1}, x_t)$（LSTM 为 $(h_t,c_t)=\text{LSTM}((h_{t-1},c_{t-1}),x_t)$）更新状态，读完后的最终状态即**上下文向量 $C$**（LSTM 通常 $C=(h_T,c_T)$；也可由全部 $\{h_1..h_T\}$ 拼接过线性层或平均池化得到）。编码器能看整个输入序列，所以可用**双向 RNN** 获得更丰富的表示。

**解码器**负责「组织语言并生成」：先由 $C$ 初始化状态（LSTM 即 $(h'_0,c'_0)=(h_T,c_T)$），然后**自回归生成**——第一步以 `<SOS>` 起步生成 $y_1$，第二步把上一步生成的 $y_1$ 作输入生成 $y_2$，$(h'_t,c'_t)=\text{LSTM}((h'_{t-1},c'_{t-1}),y_{t-1})$，直到生成 `<EOS>` 或达最大长度。每步 $h'_t$ 过全连接+Softmax 得词表概率分布，取最高者输出。因要满足因果性（预测当前词不能看未来），解码器用**单向 RNN**。

> **视角很重要**：解码器本身就是一个**条件语言模型**。普通语言模型凭空「预测下一个词」（如输入法补全）；Seq2Seq 解码器同样预测下一个词，但**以 $C$ 为初始条件**。所以 **Seq2Seq 的本质 = 编码器（理解和压缩信息）+ 条件语言模型（特定条件下生成信息）**。把编码器去掉、只留解码器，模型不再接收外部 $C$、仅基于自身前缀（提示词）预测下一个词——这就是 **decoder-only 语言模型**，与 GPT 等现代大模型的训练范式一脉相承：海量文本上以「预测下一个词」预训练，使用时通过提示词条件化生成。

### 1.3 实现细节

- **Embedding 是否共享**：源/目标语言词汇表彼此独立（未联合词表）时各用一层；词汇大量重叠（如摘要）或合并大词表时可共享——省参数还可能帮模型学到跨语言词元联系。
- **上下文向量 $C$ 的两种注入策略**：
  1. **作解码器初始状态**（经典）：概念直观，但全部生成步骤只源于这「唯一一次」输入，长序列下初始信息随多步传递被「稀释」；
  2. **作每个时间步的额外输入**：解码器保持零初始状态，每步把词嵌入与 $C$ 拼接或逐元素相加（相加要求同维、需 broadcast）后送入 RNN——每步都「提醒」全局上下文，但注入的始终是**同一个静态向量**，无法解决更深层次的「对齐」问题。
  两种都无法根本解决信息瓶颈——而这正是注意力机制诞生的原因。
- **损失**：每步 $h'_t$ 经 Softmax 得分布 $p_t$，交叉熵 $Loss_t = -\log p_t(y_t)$，序列总损失 $\sum_t Loss_t$。PyTorch 需把 `(N,L,C)` 转置适配 `CrossEntropyLoss`，并用 `ignore_index` 忽略 `<PAD>` 位置。
- **特殊词元**：`<PAD>` 填充对齐、`<SOS>`/`<GO>` 启动生成、`<EOS>` 终止、`<UNK>` 兜底未登录词。
- **错位训练**：目标序列 `W,X,Y,Z` 的解码器输入构造为 `<SOS>,W,X,Y`、目标为 `W,X,Y,Z,<EOS>`，保证每步学「正确历史 → 下一个正确词」的映射。

## 2. 训练与推理：教师强制与自回归

**教师强制（Teacher Forcing）**：若训练时就按推理的自回归模式（上一步预测作下一步输入），初期预测不准会造成错误不断累积、收敛缓慢，且逐步依赖难以并行。教师强制让解码器第 $t$ 步直接吃**真实标签** $y_{t-1}$ 而非预测值，每步都有正确历史信息，收敛稳定快速。（注意：RNN 解码器时间维仍串行；到 Transformer 等无递归结构中，教师强制配合掩码才能实现时间步并行。）

**自回归推理**：没有正确答案可喂，模型「自己教自己」——$C$ + `<SOS>` 生成 $y_1$，$y_1$ 生成 $y_2$……直到 `<EOS>`。朴素实现每生成一步都把**已生成的完整序列**重喂，$h'_1,h'_2$ 被反复重算，序列越长浪费越大。高效实现：**缓存并传递状态**——每步只输入上一个词元 $y'_{t-1}$ 和上一步状态 $h'_{t-1}$，单步计算后新状态覆盖旧状态。

每步取最高概率词的**贪心搜索**简单高效，但局部最优可能葬送全局（一步错步步错，如下棋只看眼前）。两条改进思路：**提升模型能力**（更深网络、更大数据，让每步预测本身更准）；**改进解码策略**（**束搜索 Beam Search**：每步保留多个候选序列，最后选整体概率最高者）。

## 3. PyTorch 实现骨架

编码器取最后状态作上下文、解码器单步前向、包装模块按概率决定教师强制——关键代码结构：

```python
class Encoder(nn.Module):            # nn.LSTM(bidirectional=False)
    def forward(self, x):
        embedded = self.embedding(x)                 # (B, T, H)
        _, (hidden, cell) = self.rnn(embedded)       # 中间步输出不用，取最后状态作 C
        return hidden, cell

class Decoder(nn.Module):             # 单步设计，为高效推理而生
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)                           # (B,) -> (B,1) 补伪序列维
        outputs, (hidden, cell) = self.rnn(self.embedding(x), (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))    # (B, vocab_size) logits
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        hidden, cell = self.encoder(src)
        input = trg[:, 0]                            # <SOS>
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)
        return outputs                               # (B, trg_len, vocab)
```

推理侧 `greedy_decode` 体现状态缓存：编码器只调一次，循环内 `trg_tensor` 仅含上一个词元，解码一步、状态覆盖一步，遇 `<EOS>` 提前停。变体 `DecoderAlt` 演示第二种 $C$ 注入：`input_size=hidden_size*2`，每步把上下文与词嵌入 `torch.cat` 后送入 LSTM（注意传入 RNN 的 `(hidden,cell)` 是解码器自身上一步状态，上下文已从输入端注入）。

## 4. 应用广度与信息瓶颈

Encoder-Decoder 框架本质是「一种数据形态 → 另一种数据形态」的通用范式，应用远超文本到文本：语音识别（Audio-to-Text，编码器提语音特征）、图像描述（Image-to-Text，CNN 编码图片「看图说话」）、TTS（Text-to-Speech）、问答（文章+问题编码后解码答案），甚至分类任务也能「生成化」——构造 Prompt 引导模型直接生成类别名，统一任务范式。**换掉编码器/解码器的实现，框架就能跨模态复用。**

但标准 Seq2Seq 有明显缺陷——**信息瓶颈（Information Bottleneck）**。它与长距离依赖概念相似但层面不同：

- **长距离依赖**是 RNN **内部**问题：信息在单一序列处理中因梯度累乘难从头传到尾（LSTM 用门控+细胞状态缓解）；
- **信息瓶颈**是 Encoder-Decoder **架构层面**问题：规定了两边唯一的沟通桥梁是**固定长度**向量 $C$，无论输入多长都要全部压进去——好比让人把长文所有细节总结成一句话、再仅凭这句话复述原文，必然大量丢失。编码器自身的长距离依赖还会加剧瓶颈：即便用了 LSTM，长句的最终 $C$ 仍难承载全部语义细节。

对联任务最能暴露问题：上联「两个黄鹂鸣翠柳」，期望生成「一行」时参考「两个」、「白鹭」时参考「黄鹂」。但标准架构下：

1. **信息稀释**：「两个」的信息经多步 RNN 传递后在 $C$ 中已非常微弱；
2. **信息无差别**：解码每个词依赖的都是**同一个**概要向量 $C$，没有机制「特别关注」当前位置对应的输入部分。

即便把 $C$ 作每步额外输入也无济于事——注入的仍是同一个静态向量，模型还是学不会**有选择、有侧重**地利用输入。解决这个瓶颈与对齐问题的方案，就是让解码器每步生成词元时都能「回头看」输入序列、动态计算权重分布的**注意力机制**——它极大提升长序列性能，并直接催生了 Transformer。

## 5. 注意力机制：从「一言以蔽之」到「择其要者而观之」

人类做阅读理解或翻译时从不把整句信息平均记在脑海里，而是回答特定问题时自然聚焦相关部分。注意力机制（Bahdanau et al., 2014）模拟这种认知：**解码每个词元时不再依赖固定 $C$，而是「回头看」完整输入序列，按当前解码需求为输入各部分分配权重，加权求和生成动态的、专属当前时间步的上下文向量**。生成「一行」时把最大权重给「两个」的状态，生成「白鹭」时给「黄鹂」。

### 5.1 推导：为什么是动态加权

从固定对齐策略出发逐步逼近。最直觉的想法是建立固定对齐——$C_1=h_1, C_2=h_2, ...$（生成第 $t$ 个词只用第 $t$ 个编码状态）。它只在对联这类等长、词序对应的「特例」可行：不等长的中译英里一对一映射立刻失效，且假设对齐关系固定不变，处理不了一对多/多对一。太理想化、缺乏通用性。

退一步：把**所有**输入信息都利用起来，但分配不同「重要性」——**加权求和**：

$$ C_t = \sum_{j=1}^{T_x} \alpha_{tj} h_j $$

$\alpha_{tj}$ 即解码第 $t$ 词时分配给输入第 $j$ 词的**注意力权重**；固定对齐只是它的特例（$\alpha_{11}=1$ 其余为 0）。新问题：$\alpha_{tj}$ 从何而来？它必须**动态**——解码器正要生成与「黄鹂」对应的词时 $\alpha_{t,\text{黄鹂}}$ 就该最大。于是需要一个额外模块：拿当前解码状态（上一时刻隐藏状态 $h'_{t-1}$）与每个编码器状态 $h_j$ **比较**，据比较结果生成权重组 $(\alpha_{t1},...,\alpha_{t,T_x})$。让模型**自行学习**怎么算这组权重，正是注意力机制的关键。

### 5.2 注意力计算三部曲

带注意力的模型结构与标准 Seq2Seq 类似，区别在解码器：编码器要提供**所有时间步**的状态序列 $(h_1..h_{T_x})$（不再只给最后一个）。解码第 $t$ 步以上一时刻状态 $h'_{t-1}$ 为起点：

1. **计算相似度（打分）**：$e_{tj} = \text{score}(h'_{t-1}, h_j)$，分数越高关联越强；方式有点积、也可引入可学习网络层；
2. **Softmax 归一化成权重**：$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_i \exp(e_{ti})}$，得到总和为 1 的概率分布；
3. **加权求和生成上下文**：$C_t = \sum_j \alpha_{tj} h_j$——每步专属、比固定 $C$ 更具针对性。

随后把 $C_t$ 与当前输入词元 $y_{t-1}$ 的词嵌入**拼接**成信息更丰富的向量，连同 $h'_{t-1}$ 送入解码 RNN 得 $h'_t$ 并预测 $y_t$。

<div align="center">
  <img src="{{IMG:4_2_1.svg}}" width="90%" alt="Attention 工作流程" />
  <p>图 2：Attention 工作流程（来源：base-llm 教程）</p>
</div>

高效打分函数是**缩放点积**——直接算查询向量与键向量的点积再除以 $\sqrt{d_k}$：

$$ \text{score}(h'_{t-1}, h_j) = \frac{{h'_{t-1}}^T \cdot h_j}{\sqrt{d_k}} $$

点积衡量相似度；**除以 $\sqrt{d_k}$ 是关键的缩放**：$d_k$ 大时点积方差也大，过大的值会把 Softmax 推向梯度极小区域（概率极端趋近 0 或 1）造成梯度消失，缩放后训练更稳定。

注意力机制的价值不止修补瓶颈：**动态聚焦源序列全部信息，长序列性能大幅提升**；**权重矩阵 $\alpha$ 可解释**——可视化即可观察生成每个输出词时模型在「看」输入的哪些位置，天然实现「词对齐」分析。

## 6. QKV 范式：把注意力抽象成信息检索

**查询-键-值（Query-Key-Value）** 范式把上述过程类比为一次信息检索：

- **Query（查询）**：当前需求或意图——Seq2Seq 中是解码器状态 $h'_{t-1}$，在问「根据我现在的情况，最需要输入序列的哪部分？」
- **Key（键）**：输入各信息片段的「标签/索引」，用于与查询匹配——编码器各状态 $h_j$；
- **Value（值）**：与键对应的实际内容——基础注意力中 K 与 V 通常相同，都来自编码器状态。

> 无论形式怎么变，注意力本质都是：**用 Q 与一组 K 算相关性权重，再用权重对 V 加权求和**。不同任务只需定义场景中的 Q/K/V——Seq2Seq 里 Q 是解码器状态、K/V 是编码器状态序列；后续的自注意力中 Q/K/V 全部来自同一序列。为增强表达力，还可以在计算前对原始 Q/K/V 各过一个独立全连接层做**线性变换**，让模型在不同「子空间」中匹配和聚合信息。

统一表达即 **缩放点积注意力（Scaled Dot-Product Attention）**，Transformer 的核心组件：

$$ \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

四步：$QK^T$ 算原始分数 → 除 $\sqrt{d_k}$ 缩放 → softmax 转权重 → 乘 $V$ 加权求和输出。

## 7. PyTorch 实现：两种注意力模块

改造要点：编码器返回**所有时间步**输出（作 K/V）；新增独立 `Attention` 模块；解码器每步调用它并把上下文融入输入。注意 RNN 解码本身按时间步串行，Attention 并未改变这一点——逐步解码便于每步显式算注意力并插入教师强制。

支持注意力的编码器通常用双向 + 降维：

```python
self.rnn = nn.LSTM(..., bidirectional=True)      # outputs: (B, src_len, H*2)
self.fc  = nn.Linear(hidden_size * 2, hidden_size)
outputs = torch.tanh(self.fc(outputs))           # 融回 (B, src_len, H)，作 K/V
return outputs, hidden, cell
```

**版本一：无参数注意力**（对应最基础的数学思想）——Query 取解码器最后一层状态，与所有 Key 做缩放点积：

```python
class AttentionSimple(nn.Module):
    def __init__(self, hidden_size):
        self.register_buffer("scale_factor", torch.sqrt(torch.FloatTensor([hidden_size])))  # non-learnable
    def forward(self, hidden, encoder_outputs):
        query = hidden[-1].unsqueeze(1)                        # (B, 1, H)
        energy = torch.bmm(query, encoder_outputs.transpose(1, 2)) / self.scale_factor
        return torch.softmax(energy, dim=2).squeeze(1)         # (B, src_len)
```

**版本二：带参数注意力**——小网络自主学习怎么判断相关性。Q 来自解码器、K/V 来自编码器，两个不同序列互动对齐，故称**交叉注意力（Cross-Attention）**（与 Q/K/V 同源的自注意力相对）：

```python
class AttentionParams(nn.Module):
    def __init__(self, hidden_size):
        self.attn = nn.Linear(hidden_size * 2, hidden_size)   # 学习 Q-K 拼接后的复杂对齐
        self.v = nn.Parameter(torch.rand(hidden_size))        # 把多维能量压成单一分数
    def forward(self, hidden, encoder_outputs):
        hidden_rep = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden_rep, encoder_outputs), dim=2)))
        attention = torch.sum(self.v * energy, dim=2)
        return torch.softmax(attention, dim=1)
```

通用解码器把两者都装得下——`input_size=hidden_size*2`，每步「算权重 → `bmm` 加权求和得 context → 与词嵌入拼接 → 送 RNN → fc 预测」：

```python
a = self.attention(hidden, encoder_outputs).unsqueeze(1)   # (B, 1, src_len)
context = torch.bmm(a, encoder_outputs)                    # 权重 × Values
rnn_input = torch.cat((embedded, context), dim=2)
outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
```

包装模块里有个双向状态适配的细节：双向编码器 `hidden` 形状 `(num_layers*2, B, H)` 而单向解码器要 `(num_layers, B, H)`，`hidden.view(num_layers, 2, batch, -1).sum(dim=1)` 在方向维度上求和巧妙合并；循环中每步把 `encoder_outputs` 完整传给解码器——这是注意力能「回顾」整个输入序列的关键。

## 8. 注意力机制的类型

早期受硬件算力限制，衍生出几组类型：

- **Soft Attention**（前文一直讨论的）：为**所有**位置算 0~1 连续权重再加权求和。端到端可微、标准梯度下降可训；缺点是每步都要与全部输入状态算相似度，超长序列开销大。
- **Hard Attention**（Xu et al., 2015）：每步只**选择最相关的一个**位置（非 0 即 1）。计算量大减；但选择过程离散**不可微**，常规反向传播失效，需借助强化学习等技巧。
- **Global vs Local Attention**（Luong et al., 2015，按计算范围区分）：Global 与 Soft 基本一致，考虑所有编码器状态；**Local** 是 Soft 与 Hard 之间的折中——先由小网络仅依赖当前解码状态预测对齐位置 $p_t = T_x \cdot \text{sigmoid}(W_p h'_t + b_p)$（避免与所有状态比较），再以 $p_t$ 为中心取 $2D+1$ 窗口，只在窗口内做 Soft Attention。

> 硬件与内核优化推动了全局注意力在常规长度任务中的普及，但 $O(N^2)$ 成本在长序列、低延迟或资源受限场景仍是瓶颈——局部/稀疏/窗口化/混合注意力至今在这些场景常用。

> **补记**：本篇两个架构的因果链值得再压一遍——Seq2Seq 把「转换」拆成理解（编码为 $C$）+生成（条件语言模型），但 $C$ 是定长瓶颈；固定对齐太脆 → 加权求和太泛 → 需要「会动态分配的权重」→ 注意力；注意力再抽象一步就是 QKV 检索范式，下一步把 Q/K/V 全部换成序列自己，就是 Transformer 的自注意力。

## 练习（源教程无，略）

## 参考（源教程引注）

- Sutskever, I. et al. (2014). *Sequence to sequence learning with neural networks*. NeurIPS
- Cho, K. et al. (2014). *Learning phrase representations using RNN encoder-decoder*. EMNLP
- Bengio, S. et al. (2015). *Scheduled Sampling for Sequence Prediction with RNNs*. NeurIPS
- Bahdanau, D. et al. (2014). *Neural machine translation by jointly learning to align and translate*. arXiv:1409.0473
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS
- Xu, K. et al. (2015). *Show, attend and tell*. arXiv:1502.03044
- Luong, M. T. et al. (2015). *Effective approaches to attention-based NMT*. arXiv:1508.04025

## 写在最后

QKV 范式把 Q/K/V 的来源换成序列自身，就得到自注意力——下一篇[深入解析Transformer](6-深入解析Transformer)拆开那个「完全抛弃循环与卷积」的新网络架构。
