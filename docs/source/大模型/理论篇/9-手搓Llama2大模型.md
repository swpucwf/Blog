# 手搓Llama2大模型

## 1. 从「调用模型」到「实现模型」

前面学过注意力机制、Transformer，以及 Encoder 衍生的 BERT、Decoder 衍生的 GPT。这一篇亲手实现一个（曾经的）前沿大模型 Llama2（Meta AI 开源），**不依赖 `transformers` 的高度封装，从零落地每个组件**——先梳理关键思想与设计取舍，再写成代码。

## 2. Llama2 架构总览

Llama2 遵循 GPT 开创的 **Decoder-Only** 架构，完全由 Transformer 解码器层堆叠，天然适合自回归生成。它的数据流：输入嵌入 → N×Transformer Block → 最终归一化 + 线性输出得 logits。与原始 Transformer 解码器相比，Llama2 及同类模型做了四项组件升级：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_6_1_1.svg" width="40%" alt="Llama2 架构图" />
  <p>图 1：Llama2 架构示意</p>
</div>

- **预归一化（Pre-Norm）+ RMSNorm**：输入进入注意力层和前馈网络**之前**先过一次 `RMS Norm`——正如 GPT-2/3 转向 Pre-Norm 解决了深层训练难题，预归一化被视为提升大模型训练稳定性的关键；归一化统一用 RMSNorm；
- **Grouped-Query Attention（GQA）**：Llama2-70B 采用；小模型可视为 `n_kv_heads == n_heads` 的 MHA 特例；
- **SwiGLU**：前馈网络的激活变体；
- **RoPE 旋转位置编码**：位置信息不在输入端与词嵌入相加，而是在注意力层内部**动态施加于 Q、K 向量**之上。

每个子层（注意力、前馈）输出都经残差连接 `+` 与输入相加，完整无损保留原始信息流；所有 Block 之上再最后一次 RMSNorm 加线性层出 logits。

## 3. 组件一：RMSNorm

标准 LayerNorm 要减均值、除标准差，计算偏复杂。**RMSNorm** 简化归一化：一方面**移除均值中心化**，只用输入的均方根缩放；另一方面**保留可学习增益** $\gamma$ 恢复表达能力：

$$ y = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} \cdot \gamma $$

输入 `[batch_size, seq_len, dim]`，每个词元在 `dim` 维独立归一化，输出同形。

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))    # 公式中的 gamma
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # x / RMS(x)
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight
```

`torch.rsqrt` 一次算出「平方均值的倒数平方根」，`eps` 防除零保数值稳定，先转 `float()` 计算再 `type_as` 回原精度。

## 4. 组件二：RoPE 旋转位置编码

传统位置编码（固定或可学习）都是**绝对位置编码**——给每个位置分配独立向量、用**加法**注入词嵌入。Llama2 用 **RoPE**，是**相对位置编码**，策略根本不同：**位置信息不再是「加」到词嵌入，而是在算注意力时通过复数「乘法」「旋转」Q 和 K 向量**。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_6_1_2.png" width="80%" alt="RoPE 旋转位置编码示意图" />
  <p>图 2：RoPE 工作原理</p>
</div>

**数学原理**：把向量每对维度 $(x_1,x_2)$ 视为复数 $x_1+ix_2 = re^{i\theta}$，两复数相乘模相乘、幅角相加。RoPE 的 `freqs_cis` 是模为 1 的复数 $e^{im\theta}$（$m$ 为位置），与 Q/K 相乘后只改变方向、不改变长度——位置越靠后旋转角度越大，把位置编码进向量的**方向**。

**三大优势**：

1. **相对位置编码**：位置 $m$、$n$ 的 Q/K 旋转后点积只与相对距离 $m-n$ 有关、与绝对位置无关——注意力具**平移不变性**（相距 2 的词元关系无论出现在序列何处算法一致）；
2. **长度外推**：依赖相对位置，对超出训练长度的序列仍能较好处理位置关系；
3. **计算高效**：复数乘法实现、不改模长，避免额外归一化。

实现分两步。`precompute_freqs_cis`（初始化算一次，输出 `[end, dim/2]` 复数张量）：

```python
def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))  # 1/theta^(2i/dim)
    t = torch.arange(end, device=freqs.device)                     # 位置序列 0..end-1
    freqs = torch.outer(t, freqs).float()                          # 位置×频率 = 相位
    return torch.polar(torch.ones_like(freqs), freqs)              # e^{i·相位}，模为1
```

`apply_rotary_emb`（把旋转施加到 Q、K）：

```python
def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # head_dim 实数 -> head_dim/2 复数
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * reshape_for_broadcast(freqs_cis, xq_)).flatten(3)  # 复数乘=旋转
    xk_out = torch.view_as_real(xk_ * reshape_for_broadcast(freqs_cis, xk_)).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xq)
```

`xq_ * freqs_cis` 正是旋转的实现——复数域相乘即幅角相加，`freqs_cis` 模为 1 故只旋转不缩放。Q、K 各自生成广播视图（`reshape_for_broadcast`）是为兼容 GQA 的头数差异（`n_heads` vs `n_kv_heads`）。`theta=10000.0` 是控制频率范围的「基底」标准值；工程上预计算长度常取 `max_seq_len * 2` 作缓冲，避免推理处理更长序列时重算。

## 5. 组件三：分组查询注意力 GQA

**标准 MHA** 给每个 Query 头配独立 K、V 头，K/V 投影矩阵尺寸与推理 KV 缓存大小都正比于 `n_heads`，模型一大开销就显著。**GQA** 的核心优化：**多个 Query 头共享一组 K、V 头**。

- MHA：`n_heads == n_kv_heads`（每 Q 头有自己 K/V）；
- GQA：`n_heads > n_kv_heads`（每组 Q 共享一组 K/V）；
- MQA：`n_kv_heads == 1`（所有 Q 共享唯一 K/V），是 GQA 特例。

GQA 在保持 MHA 大部分性能的同时显著减少 K/V 计算与显存：**显存**上 KV 缓存从正比 `n_heads` 降为正比 `n_kv_heads`（约 `n_kv_heads/n_heads`，70B 模型可省数十 GB）；**加速**上 K/V 投影及后续矩阵乘法计算量同步下降。例：`n_heads=32`、`n_kv_heads=8`，K/V 相关参数、计算、缓存都缩到四分之一。

关键实现：算注意力分数前用 `repeat_kv` 把 K、V 头复制 `n_rep = n_heads/n_kv_heads` 次与 Q 头对齐。`repeat_kv` 与 RoPE 函数同放 `rope.py`——把注意力的「无状态张量算子」（`apply_rotary_emb`、`precompute_freqs_cis`、`repeat_kv`）集中复用、解耦 `attention.py`：

```python
def repeat_kv(x, n_rep):
    b, s, n_kv, hd = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(b, s, n_kv, n_rep, hd)
              .reshape(b, s, n_kv * n_rep, hd))     # 复制到与 Q 头数对齐

class GroupedQueryAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads=None, ...):
        self.n_rep = n_heads // n_kv_heads          # Q 头与 KV 头的重复比
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)     # 输出对应 n_heads
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)  # 输出对应 n_kv_heads
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
    def forward(self, x, start_pos, freqs_cis, mask):
        xq = self.wq(x).view(b, s, self.n_local_heads,    self.head_dim)
        xk = self.wk(x).view(b, s, self.n_local_kv_heads, self.head_dim)
        xv = self.wv(x).view(b, s, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)      # 先旋转注位置
        # ... KV Cache ...
        keys, values = repeat_kv(keys, self.n_rep), repeat_kv(values, self.n_rep)   # 关键对齐
        scores = (xq.transpose(1,2) @ keys.transpose(1,2).transpose(2,3)) / ...
```

`wq/wk/wv` 输出维度不同，直接体现 GQA 设计；测试用 `n_heads=8, n_kv_heads=2` 跑通并验证输出形状与输入一致。

## 6. 组件四：SwiGLU 前馈网络

标准 FFN 是两个线性层加 ReLU。Llama2 用变体 **SwiGLU**——引入**门控机制**，用三个线性变换（`W`、`V`、`W2`）：`xW` 先过 Swish（`swish(x)=x·sigmoid(x)`），`xV` 作「门」与前结果逐元素相乘，`W2` 输出：

$$ \text{SwiGLU}(x, W, V, W_2) = (\text{swish}(xW) \otimes xV)W_2 $$

门控结构让网络动态控制信息流，被认为是它优于标准 ReLU FFN 的主因。中间 `hidden_dim` 通常大于 `dim`，Llama2 按公式计算并用 `multiple_of` 对齐以提高硬件效率：

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of, ...):
        hidden_dim = int(2 * hidden_dim / 3)                              # SwiGLU 惯例缩到 2/3
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)  # 对齐
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # W
        self.w2 = nn.Linear(hidden_dim, dim,  bias=False)  # W2
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # V（门）
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))    # silu=Swish，* 即门控
```

`F.silu` 即 PyTorch 内置 Swish，`* self.w3(x)` 实现门控。

## 7. 组装：TransformerBlock 与 LlamaTransformer

**TransformerBlock** 是基本单元，预归一化 + 残差一目了然——先 `RMSNorm` 再进子层、再加残差：

```python
class TransformerBlock(nn.Module):
    def __init__(self, layer_id, ...):
        self.attention = GroupedQueryAttention(...)
        self.feed_forward = FeedForward(...)
        self.attention_norm = RMSNorm(...)
        self.ffn_norm = RMSNorm(...)
    def forward(self, x, start_pos, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)  # Pre-Norm + 残差
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
```

**LlamaTransformer** 顶层模型堆叠 Block 并处理输入输出：`tok_embeddings` 转向量、`nn.ModuleList` 堆 N 层、末尾 `norm` + `output`，`register_buffer` 预缓存 `freqs_cis`。

```python
class LlamaTransformer(nn.Module):
    def forward(self, tokens, start_pos):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]      # ① 按位置切旋转矩阵
        mask = None
        if seqlen > 1:                                                   # ② 因果掩码
            mask = torch.triu(torch.full((seqlen, seqlen), float("-inf")), diagonal=1)
            mask = torch.hstack([torch.zeros(seqlen, start_pos, ...), mask])  # 配合 KV 缓存偏移
        for layer in self.layers:                                        # ③ 逐层
            h = layer(h, start_pos, freqs_cis, mask)
        return self.output(self.norm(h)).float()                          # ④ 最终 norm + 线性出 logits
```

forward 三步值得记：`freqs_cis` 按 `start_pos` 与 `seq_len` 切片取用；`mask` 用 `torch.triu` 造上三角 `-inf` 保证每位置只关注自己和之前——这是**因果语言模型**的关键，`torch.hstack` 补 `start_pos` 列零配合 **KV 缓存**（推理时 `start_pos>0` 让当前 Q 能关注缓存里全部历史 K）；最后逐层过 Block 归一化出 logits。

## 8. 整体验证

用小尺寸参数（`dim=256, n_layers=2, n_heads=8, n_kv_heads=2`）实例化，喂随机 `tokens (2,16)` 前向，输出 `logits shape: (2, 16, 1000)`——正好 `[batch_size, seq_len, vocab_size]`，模型能正确处理输入返回预期 logits。

> **补记**：Llama2 相对 GPT-2 的四项改动（RMSNorm/Pre-Norm、RoPE、GQA、SwiGLU）没有一个在动「Transformer 骨架」，全在骨架的接缝处省算力、稳训练——把每个组件对回理论篇 6 讲的那张「如果只用注意力还缺什么」清单，会发现它们是同一批问题的现代答卷：位置无关 → RoPE（且升级为相对 + 外推），深层难训 → RMSNorm + Pre-Norm，推理显存墙 → GQA + KV 缓存，非线性表达 → SwiGLU。读懂这四点，基本就读懂了开源大模型架构的公约数。

## 练习

- 依 `code/C6/llama2` 目录，逐文件实现 `norm.py`、`rope.py`、`attention.py`、`ffn.py`、`transformer.py` 并以形状测试驱动验证。

## 参考文献

- Touvron, H. et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. arXiv:2307.09288
- Zhang, B. & Sennrich, R. (2019). *Root Mean Square Layer Normalization*. NeurIPS
- Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864
- Ainslie, J. et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models*. arXiv:2305.13245
- Shazeer, N. (2020). *GLU Variants Improve Transformer*. arXiv:2002.05202

## 写在最后

搭完稠密的 Llama2，下一篇[MoE架构解析](10-MoE架构解析)看另一种扩容思路——不把所有参数塞进一个前馈网络，而用稀疏专家按需激活。
