# 原生统一架构与简化Omni

## 1. 迈向统一的探索

BLIP-2 与 LLaVA 是「**连接**」范式：用轻量适配模块连接冻结的视觉编码器与 LLM，虽极大降低训练成本，但底层模态分离与输入端生硬拼接，限制了实时音视频交互场景的表现。为突破这些局限，学术界与工业界正加速向**原生统一**架构演进。

### 1.1 Flamingo：过渡形态

DeepMind 2022 年的 **Flamingo** 虽早于 BLIP-2，却在大一统进程中占据独特生态位：它不追求「利用现有组件高效连接」，而探索**大规模图文交错序列（Interleaved Image-Text）** 的学习极限，事实上是从「简单连接」向「原生统一」进化的关键**过渡形态**——在保留视觉模型与 LLM 各自预训练知识的同时，实现**极少样本**下对**任意交错**图文序列的处理。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_20_2_1.png" width="90%" alt="Flamingo 架构" />
  <p>图 1：Flamingo 架构</p>
</div>

图像先经**左侧视觉路径**编码与重采样（Vision Encoder + Perceiver Resampler）生成固定长度视觉特征，再注入**右侧语言路径**，通过插在 LLM 层间的门控交叉注意力接口（Gated XATTN-DENSE）参与文本生成。三个关键组件：

**（1）视觉感知与特征重采样**：用冻结的 **NFNet-F6** 提取图像/视频特征。由于视觉输入的分辨率与帧数可变，特征图大小和数量也随之变化；**Perceiver Resampler** 预定义一组固定数量的**可学习潜在查询向量（Latent Queries）**，通过 Cross-Attention 与视觉特征交互，输出固定数量的视觉 Token（论文为 **64 个**）。这既显著降低视觉-文本 Cross-Attention 的开销，也通过「压缩视觉 token 数量」缓解 Transformer 的 $O(N^2)$ 压力（严格说不是「消除」$O(N^2)$，而是不让 $N$ 被视觉 token 拉得过大）。对视频输入，展平前会为每帧特征加入可学习**时间嵌入**；作者明确**未显式加入空间网格位置编码**（空间信息更多由 CNN 特征隐式携带）。Key/Value 可由**时空视觉特征与 learned latent vectors 拼接**而成，让 Latent Queries 查询时也「参考」自身当前状态，保持特征提取连贯性。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_20_2_2.png" width="80%" alt="Perceiver Resampler 机制" />
  <p>图 2：Perceiver Resampler 机制</p>
</div>

**（2）视觉注入与门控**：在冻结的 **Chinchilla** LLM 层间插入 **GATED XATTN-DENSE**——以语言特征为 Query、Perceiver 输出的视觉特征为 Key/Value，让语言模型主动从视觉序列取信息。为维持冻结 LLM 原有语言能力，Cross-Attention 与随后 FFW 层配置 **tanh 门控**，参数初始化为 0：训练初期模型表现为纯语言模型，避免视觉噪声冲击导致的不稳定；随训练进行门控值逐渐增大，视觉信息以残差方式「平滑」融入语言特征流。这既防灾难性遗忘，又让模型通过交错的 Self-Attention 与 Cross-Attention 实现深度多模态对齐。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_20_2_3.png" width="90%" alt="Gated Cross-Attention" />
  <p>图 3：Gated Cross-Attention</p>
</div>

**（3）多图支持与分段掩码**：为支持任意数量图像/视频，**Masked Cross-Attention** 采用特殊**分段掩码**——每个文本 Token **仅允许关注其直接前导的那一张图像**，而非所有历史图像。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_20_2_4.png" width="90%" alt="多模态掩码注意力" />
  <p>图 4：多模态掩码注意力</p>
</div>

这限制了单次交叉注意力的复杂度、防止无关图像干扰；跨图像长程依赖交给底层 **LM Self-Attention**，以文本 Token 为中介间接融合。基于该机制，Flamingo 能高效处理多达 **32 对图文**的 Few-Shot Prompt，展现强上下文学习能力。

**局限**：强少样本能力很大程度上归功于数据——DeepMind 构建 **M3W（Multi-Modal Massive Web）**，从 **4300 万个网页**提取图文交错序列，模拟人类浏览网页的体验；还混合 ALIGN、LTIP 与视频文本对 VTP 并用加权损失联合训练。但它在纯图像分类上不如 CLIP 这类对比模型（生成式目标所致）；继承 LLM 缺点会**幻觉**；最关键的是仍保留「冻结视觉编码器」的连接范式特征——视觉与语言在底层特征空间无法真正融合，OCR、细粒度识别等细腻感知能力易在层层传递中丢失。

### 1.2 走向纯原生

**Chameleon** 与 **GPT-4o** 开启了**纯粹的原生**时代：模型不再是「拼凑」出来的，而是从一开始就把所有模态视为地位平等的「语言」，在同一大模型中**混合模态端到端预训练**——就像人类婴儿并非先学会「看」再学会「说」，而是在成长中同时通过视觉、听觉与语言感知世界。架构上**彻底摒弃独立模态编码器**，把像素、波形与文本一视同仁地映射为 Token，直接送入同一个「大脑」。

**Chameleon 与统一词表**（Meta AI，早期融合 Early-Fusion 的代表）：理念激进——**把一切都视为 Token**。三个层面创新：

1. **统一词表与离散化**：用 Gafni 图像分词器把 $512\times512$ 图像量化为 **1024 个离散 Image Tokens**，与文本 Token 共同组成大小为 **65,536** 的统一词表（含 8192 个图像 Codebook Token）。在模型眼中，像素与文字都是同一字典里的「单词」；
2. **混合序列与端到端架构**：图像 Token 与文本 Token 按逻辑顺序拼成**混合模态序列**，直接输入**统一 Transformer**；理解与生成都转化为**自回归下一 Token 预测**，不再依赖外挂视觉编码器或复杂跨模态对齐模块。在**边界符（Start/End Image）** 辅助下，图文真正实现端到端全原生理解与生成；
3. **攻克训练稳定性**：「从头」训练混合模态大模型面临巨大优化稳定性挑战（模态竞争导致 Logit 漂移）。Chameleon 引入 **QK-Norm** 与特殊层归一化布局。实验揭示本质：**图像生成任务是导致不稳定的根源**（禁用图像生成时 Loss 曲线平滑）；单纯加 Dropout 不能解决发散，**层归一化重排（Norm Reordering）** 才是稳定训练的关键。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_20_2_5.png" width="90%" alt="Chameleon 混合模态序列" />
  <p>图 5：Chameleon 混合模态序列</p>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_20_2_6.png" width="90%" alt="Chameleon 训练稳定性实验" />
  <p>图 6：Chameleon 训练稳定性实验</p>
</div>

Chameleon 7B/34B 在架构参数上与 LLaMA-2 总体一致（Context、GQA），为适配混合模态训练在优化上调整明显：引入 **Z-loss** 与 **QK-Norm**，训练数据量提升到 **4.4T Token**（约为 LLaMA-2 的两倍）。这种原生统一实现了真正的全模态理解与生成，但代价昂贵——需约 **10T token** 混合数据大规模预训练，且对超参数极度敏感。

**GPT-4o 与全模态原生**：在图文统一基础上进一步打破音频边界。此前语音模式是三个独立模型的级联（ASR 转文本 → LLM 处理 → TTS 转语音）；GPT-4o 跨文本、视觉与音频**端到端训练单一新模型**，所有输入输出由同一神经网络处理——因此能直接感知语气、多个说话者、背景噪音，并输出带情感的语音，语音对话响应延迟最短约 **232ms**。

## 2. 工业实践：端到端紧耦合 Omni

Chameleon 与 GPT-4o 定义了「纯粹原生」的最终形态，但从头训练成本极高。工业界与开源社区更常见也更容易规模化落地的路线是**端到端紧耦合（End-to-End Tightly-Coupled）** 的 Omni 系统（如 **Qwen3-Omni**、**MiniCPM-o 4.5**）。它们严格来说**不属于**单一网络原生架构（仍保留专门的音频或视觉编码器），但打破「冻结外挂」的连接范式，通过**全参数解冻的端到端联合训练**、**深度隐藏态耦合**与**高速流式架构**，在体验上无限逼近 GPT-4o 的实时语音交互。

### 2.1 Qwen3-Omni：解耦架构与流式生成

**（1）通用听觉底座**：不依赖 Whisper 等现成冻结 ASR，而用从头训练、约 **6 亿参数**的 **AuT（Audio Transformer）** 音频编码器——Transformer 架构（**32 层 Encoder + 8 层 Decoder**），在 **2000 万小时**监督音频数据上预训练，既能处理语音，也能理解环境音与音乐。特征提取时用 **3 层下采样卷积**把输入音频（10ms 帧移的 FBank 特征）在时间维压缩 8 倍，采样率降至 **12.5Hz**（每 80ms 一个 Token）；32 层 Encoder 中引入**分块窗口注意力（Block-wise Window Attention）**，支持动态窗口，使实时流式输入时能高效 Prefill 而无需等整段音频。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_20_2_7.png" width="90%" alt="Audio Transformer (AuT) 架构概览" />
  <p>图 7：Audio Transformer（AuT）架构概览</p>
</div>

**（2）Thinker-Talker 双模型与极致流式**：为达成 **234ms** 端到端延迟，采用 **Thinker（思考者）+ Talker（表达者）** 双模型协作，且均升级为 **MoE** 应对高并发——Thinker 是强大**多模态 MoE**，负责「脑力」：理解来自 AuT 的音频流、视觉编码器的视频流与文本输入，深度推理并生成文本回复；Talker 是专门的**流式语音生成 MoE**，关键设计是**不再直接消费 Thinker 的高层文本表征/文本 Token**，而接收其输出的**多模态高维表征**并共享对话历史。解耦让 Talker 能专注对齐语音韵律、情感与语速，不受语言模型逐词生成节奏制约。

生成流程体现了「极速流式」：Thinker 先生成高层语义表征 → Talker **自回归**预测第 0 层（主）Codebook → 轻量 **MTP 模块**快速预测同帧其余残差 Codebook → 所有层级 Codebook 送入**流式 Codec 解码器**逐帧合成波形。只要第一个 codec token 产生，声音就能立即「流」出来，无需等完整句子甚至完整词。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_20_2_8.png" width="90%" alt="Qwen3-Omni 架构概览" />
  <p>图 8：Qwen3-Omni 架构概览</p>
</div>

**（3）三阶段训练**：**编码器对齐**（冻结 LLM，只训 AuT 与视觉编码器的适配器，对齐到语言空间）→ **通用预训练**（全参数解冻，2 万亿 Token 多模态数据混合训练）→ **长上下文扩展**（序列扩至 32k，增强长视频/音频理解）。这套联合优化与流式调度证明：通过精巧系统设计，模型能在复用现成模块的同时打破延迟瓶颈。

### 2.2 全双工与端侧：MiniCPM-o 4.5

除了模型体量与理解能力，**系统级实时流式交互（全双工，Full-Duplex）** 与**端侧部署**是另一个重要方向。以往语音助手多为「半双工」（你听我说、我说你听）或简单打断，**MiniCPM-o 4.5** 实现真正的「边听边想边说」：接收实时视频与音频输入流的同时，不阻塞地并行输出文本与语音流。底层把多模态数据流（并行输入/输出）在毫秒级时间轴精密对齐同步，模型可在任意时刻主动决定是否发声，实现自然的「主动插话」与「响应打断」。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_20_2_9.png" width="90%" alt="MiniCPM-o 4.5 全双工流式对齐机制" />
  <p>图 9：MiniCPM-o 4.5 全双工流式对齐机制</p>
</div>

机制上，模型在 LLM 输出序列中插入 `[silent]` **占位 Token** 维持流式节拍，并把生成的 Token 与隐藏态 $h$ 一并送入语音 Token 解码器，产生 **25Hz speech tokens**，由此在统一时间轴上对齐输入与输出。它整合视觉（SigLIP2）、听觉（Whisper-medium）、语音合成（CosyVoice2）与 LLM（Qwen3-8B），端到端深度融合后整体约 **9B** 参数；配合量化（Int4 仅需 **11GB** 显存）与专属推理框架，可在个人电脑乃至手机流畅运行全双工音视频对话——在有限算力下逼近 GPT-4o 级实时体验。

## 3. 从 0 训练简化版 Omni

把前面所学融会贯通：在单张消费级显卡上从 0 实现一个简版 Omni 模型，目标是同时接收**文本**与**图像**两路输入，通过投影与注入把视觉特征与文本序列结合，做端到端联合计算与生成。

### 3.1 数据与分词器

参考 Qwen3-Omni 的分阶段策略，采取「**先炼纯文本基座，再做多模态对齐**」：

- **纯文本**：借用开源项目 MiniMind 的高质量精简数据——`pretrain_hq.jsonl`（语言预训练）与 `sft_mini_512.jsonl`（指令微调）；
- **图文**：**Flickr8k** 配中文描述 **Flickr8k-CN**，在中文语境下交互。

工程上用 `uv init` 建 `seeker-omni` 项目，目录分 `dataprep/download`（下载）与 `dataprep/prepare`（清洗、分词器训练），中间产物包括：去乱码/去低质后的预训练语料、统一成 ChatML（system/user/assistant）的指令数据、以及把图片路径与中文描述绑定的图文 JSONL：

```json
{"id": "flickr8k-...", "system": "你是一个只用中文回答的助手。", "prompt": "请描述这张图片。", "answer": "在草地上赛跑的两只黑狗。", "image": "data/raw/flickr8k/Flickr8k_Dataset/....jpg"}
```

**分词器**要先于模型训练。关键设计是**硬性绑定特殊控制符**——`<|endoftext|>`、`<|im_start|>`、`<|im_end|>` 以及界定视觉特征插入位置的 `<img_bos>`、`<img>`、`<img_eos>`。只有构建字典时强制这批特殊标记占住词表最前位置（0、1、2…），后续端到端训练的切片与视觉特征注入才不会错位。训练采用与 GPT 相同的**字节级 BPE（BBPE）**，`vocab_size=6400`：

```python
tok = Tokenizer(BPE(unk_token=unk, byte_fallback=True))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
trainer = BpeTrainer(vocab_size=6400, special_tokens=special_tokens,
                     initial_alphabet=ByteLevel.alphabet())
tok.train([str(input_path)], trainer=trainer)
for i, t in enumerate(special_tokens):          # 断言特殊 token ID 稳定
    assert tok.token_to_id(t) == i
```

> **分词器怎么「训练」**：BBPE 不涉及神经网络梯度，而是纯粹的词频统计与合并——先把语料打碎成 256 种单字节，再反复统计最高频的相邻字节对、粘合成新词条，循环直到词表达到上限。常见词被切成完整一块，生僻字拆成两三个底层字节。

**特征序列化**：若在 DataLoader 里实时分词海量变长字符串，CPU/磁盘 I/O 会成为 GPU 瓶颈。预处理阶段提前把文本转成等长数字 ID、把图片过视觉编码器抽成特征，用 Numpy **`memmap`** 写成连续二进制矩阵（`input_ids.bin`、`labels.bin`、`attention_mask.bin`、`image_feats.bin`），训练时零拷贝直读。

### 3.2 三阶段训练流水线

在有限算力下，稳妥做法是让模型**先掌握语言 → 再学指令 → 最后注入视觉**：

1. **纯文本无监督预训练**：二进制语料 + 因果语言建模，让模型「学会说话」；
2. **纯文本指令微调**：ChatML 格式问答数据 + `-100` 掩码屏蔽用户问题，蜕变为能听懂指令的助手；
3. **端到端多模态对齐**：引入图文描述记录，训练中同时更新视觉投影层甚至解冻部分 LLM 骨干。与早期「梯度断裂」的拼接模型不同，这种**端到端联合反向传播**正是「紧耦合」的精髓——拉近离散文本空间与连续视觉特征的距离。

（RLHF 偏好对齐与 MoE 等进阶优化在此略去，聚焦多模态核心回路。）

### 3.3 架构：三组件 + 视觉注入

整体架构三个核心组件：**文本处理**（词嵌入 + 多模态特殊占位符）、**视觉特征投影**（线性层 + 门控把视觉特征对齐到 LLM 维度并注入序列）、**统一 LLM**（标准 Transformer Decoder，含 GQA、RoPE、SwiGLU、RMSNorm、与嵌入共享权重的 LM Head，接收融合特征端到端自回归推理）。

**视觉特征压缩（PerceiverResampler）**：不把 ViT 写进 LLM 前向重训，而接收训练好的 SigLIP 提取的特征；但其特征序列很长（如 729 Token），直接送入会拖慢推理、吃显存。用少量可学习潜在查询向量做交叉注意力，把长序列压缩到固定较短长度（如 64 Token）：

```python
class PerceiverResampler(nn.Module):
    def forward(self, x):                    # x: [B, T_img, D]
        latents = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        for layer in self.layers:
            attn_out, _ = layer["attn"](layer["ln_q"](latents),
                                        layer["ln_kv"](x), layer["ln_kv"](x))
            latents = latents + attn_out
            latents = latents + layer["ff"](layer["ln_ff"](latents))
        return latents                        # [B, num_latents, D]
```

**投影与门控注入**：把压缩特征线性投影到 LLM 隐藏维度，并乘 `tanh(gate)`——门控初始为 0，使模型初期等同于纯文本模型，联合训练更稳定；然后按 `input_ids == <img>` 的位置把视觉 Token 加到对应嵌入上：

```python
img_tokens = img_proj(image_feats) * torch.tanh(img_gate)
mask = input_ids == img_token_id
pos = mask.nonzero(as_tuple=False)
x[pos[:, 0], pos[:, 1]] += img_tokens.reshape(-1, H)
```

这里有一个**解耦设计**：`projector.py` 不直接调用 `resampler.py`，压缩在上游预处理/Pipeline 完成，压缩后的张量以 `image_feats` 参数干净传入主干——主干代码读起来仿佛在处理纯文本。

### 3.4 训练循环的几个关键设计

- **MemmapDataset**：`np.memmap` 以 `mode="c"` 只读映射，`__getitem__` 用 `torch.from_numpy` 零拷贝搬到 Tensor；
- **`apply_stage_freeze`（参数冰冻术）**：多模态训练中「什么时候冻结什么」比「怎么写 Loss」更重要。若把预训练聪明的 LLM 底座与随机初始化的视觉投影层一起放开，激烈梯度会在第一个 Epoch 就摧毁文本能力（灾难性遗忘）。所以按阶段冻结——如多模态对齐初期只解冻投影层；
- **多模态教师强制扰动**：在监督对的答案开头强制注入 UNK 掩码，断掉语言模型的「续写本能」，逼迫它观察视觉特征；
- **混合精度 + 梯度累积 + 梯度裁剪**：`autocast` 前向、`GradScaler` 防 FP16 下溢、`accum` 凑齐后真更新、`clip_grad_norm_` 防坏数据导致梯度爆炸；
- **精细化权重衰减**：不对视觉门控矩阵、词表与归一化层施加 L2 正则；
- **余弦学习率调度**：含 warmup。

> **补记**：多模态篇三篇构成一条清晰的演进线——**连接**（BLIP-2 训 188M Q-Former、LLaVA 训投影层+LLM）→ **过渡**（Flamingo 门控交叉注意力 + 交错数据）→ **原生**（Chameleon 统一词表、GPT-4o 单一网络）→ **紧耦合落地**（Qwen3-Omni 全参解冻 + 流式、MiniCPM-o 全双工端侧）。而亲手搭简化 Omni 的经历揭示了一个工程要点：**架构可以简单（投影层 + 门控注入），真正的难点在「阶段化冻结 + 数据序列化 + 流式对齐」这些训练工程细节**。

## 写在最后

「大模型」专栏的六个子系列在这里收束：从理论篇的架构根基，到实战、微调量化、部署、安全，再到多模态前沿，覆盖了从 NLP 基础到 Omni 系统的完整链路。若要把多模态继续往工程侧推，[多模态大模型实战](/source/大模型/多模态大模型实战/README)承接后续链路：数据配方、评测验收、推理部署、Demo 与多模态 Agent。
