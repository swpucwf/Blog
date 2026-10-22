# LoRA实战与Qwen2.5微调

## 1. 用 peft 库把理论跑起来

Hugging Face 的 `peft` 库把 LoRA、Prefix Tuning、Adapter 等技术抽象成统一简洁的接口，以最小代码改动把 PEFT 无缝应用到 Hub 上的大模型。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_11_3_1.png" width="90%" alt="Hugging Face PEFT 库官方文档首页" />
  <p>图 1：Hugging Face PEFT 库官方文档首页</p>
</div>

**设计定位**：`peft` 不取代 `transformers`，而是作为其**插件/增强模块**。用游戏打个比方——基础预训练模型是「天命人」（本身已有金箍棒），`peft` 是「七十二变」法术库（LoRA、Prefix、Prompt Tuning），`PeftConfig` 是「法术搭配方案」（启用哪种神通、参数怎么配），`get_peft_model` 则是「临阵变身」：接收基础模型和方案，把低秩矩阵「加持」上去，得到面向特定任务的 `PeftModel`。基础模型大部分权重冻结，只训练和切换轻量插件。

### 1.1 核心组件

**`PeftConfig`**（声明式配置基类）两个通用参数：`peft_type` 指定插件类型（如 `PeftType.LORA`）；`task_type` 指定下游任务（`CAUSAL_LM` 用于自回归、`SEQ_2_SEQ_LM` 用于 T5 类），库据此正确适配模型头部。

`LoraConfig` 的专属超参直接源于 LoRA 论文：

- `r`：秩，决定 A、B 的中间维度，控制新增参数量与适应能力（建议从 8/16 起试）；
- `lora_alpha`：缩放因子（最终更新按 `alpha/r` 缩放），常见设为 `r` 的 2 倍；
- `target_modules`：LoRA 作用在哪些模块（如 `["q_proj","v_proj"]`）；
- `lora_dropout`：LoRA 层 Dropout 防过拟合；
- `bias`：`'none'`（全冻结，最省）、`'all'`（全训）、`'lora_only'`（只训 LoRA 自身 bias）。

**`get_peft_model`** 做四件事：解析配置 → 按 `target_modules` 找到目标层 → 把 `nn.Linear` 替换/封装为注入 LoRA 的模块（内部保留冻结原权重 + 可训练低秩分支） → 返回接口与基础模型完全兼容的 `PeftModel`，可直接交给 `Trainer`。调试神器 `print_trainable_parameters()` 打印可训练参数占比。

### 1.2 LoRA 微调流程（Pythia-2.8B 实战）

**（1）加载量化模型与分词器**：用 `bitsandbytes` 在加载时做 8-bit 量化（旧参数 `load_in_8bit=True` 已弃用，改为 `BitsAndBytesConfig` 传给 `quantization_config`），`device_map="auto"` 让 accelerate 自动分配层：

```python
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config,
                                             dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token      # Pythia 默认无 pad_token
```

打印模型可见 `Linear8bitLt(...)` 替换了 `nn.Linear`，说明量化生效。Pythia 的注意力权重合并在 `query_key_value` 一层——这直接决定 `target_modules` 怎么写。

**（2）k-bit 训练预处理**：`prepare_model_for_kbit_training(model)`（PEFT ≥0.10，替代旧的 `prepare_model_for_int8_training`）做四件事：把 LayerNorm 等需高精度计算的层上调精度；启用**梯度检查点**（用重算换显存，降低训练峰值）；预处理输出嵌入层以兼容 LoRA；为需要的输入启用梯度，保证 k-bit 反向传播兼容。

**（3）定义 LoRA 配置并注入**：

```python
config = LoraConfig(r=16, lora_alpha=32,
                    target_modules=["query_key_value", "dense"],
                    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# trainable params: 7,864,320 || all params: 2,783,073,280 || trainable%: 0.2826
```

可训练参数仅 **0.28%**。`target_modules` 必须先打印模型结构、以实际层名为准——Pythia/GPT-NeoX 的注意力合并为 `query_key_value`，FFN 常见 `dense`、`dense_h_to_4h`、`dense_4h_to_h`；原封不动套到 Qwen2.5 上必然失败。

**（4）数据处理**：用 `Abirate/english_quotes` 名言数据集，定义 `tokenize(batch)` 只编码 `quote` 字段，`dataset.map(..., batched=True)` 批量应用，得到 `input_ids` 与 `attention_mask`。

**（5）Trainer 训练**：`per_device_train_batch_size × gradient_accumulation_steps` 共同决定**有效批量**（显存不够时用梯度累积模拟大批量）；`warmup_steps` 预热、`max_steps` 快速原型 vs `num_train_epochs` 正式训练、`fp16` 混合精度。关键：**直接把 `PeftModel` 交给 `Trainer`**，它会自动只训练 LoRA 参数。细节：`peft_model.config.use_cache = False`（训练时关 KV 缓存，与梯度检查点配合更稳）；数据整理器 `DataCollatorForLanguageModeling(tokenizer, mlm=False)`。

**（6）保存与推理**：`save_pretrained()` 只存增量适配器（`adapter_model.safetensors` + `adapter_config.json`，几十 MB）；`merge_and_unload()` 把旁路合并回原权重，得到结构与原模型一致的标准模型、推理零额外延迟（注意：8/4-bit 加载的基座合并后通常转为 FP16/FP32，要 k-bit 部署需重新量化）。

推理三个要点：显式传 `attention_mask`；`do_sample=True` 才能让 `temperature/top_p/top_k` 生效；推理前 `use_cache=True`。示例补全：

```text
'Be yourself; everyone else is taken.” - Oscar Wilde "I have found that people will forget what you said...'
```

仅 200 步微调，模型就学到了名言的风格与内容，印证 PEFT 的高效。

> **输出非确定性**：主要来源是解码阶段的**采样策略**（`do_sample=True` 时按概率随机抽样），其次是 CUDA 算子与浮点精度。本地结果与文档不完全一致属正常。

## 2. Qwen2.5 要点：先读懂模型，再定微调策略

本次以 `Qwen2.5-7B-Instruct` 为例做 QLoRA 微调。**微调的正确姿势由模型架构决定**——上一节 Pythia 的 `target_modules` 套到 Qwen 上就会失败。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_11_4_1.png" width="90%" alt="Qwen系列模型性能随预训练数据量增长" />
  <p>图 2：Qwen 系列模型性能随预训练数据量增长</p>
</div>

### 2.1 预训练与架构

Qwen2.5 家族覆盖 0.5B~72B，并衍生 Math、Coder 等专业模型。其能力源于 **18 万亿 Token** 高质量数据：通用知识之外，重点加强了**代码**与**数学**。知道模型的「学前教育」背景，才能判断它是否适合你的微调目标。

架构是标准 **Decoder-only**，关键组件与 Llama 类似：**GQA**（降 KV 缓存、提长序列效率）、**SwiGLU**（实现为 SiLU 门控 FFN）、**RoPE**、**RMSNorm**（子层前置归一化）、**QKV 偏置**（注意力投影带 bias，增强表达）。

### 2.2 分词器与对话模板

- **分词器**：字节级 BPE（BBPE），词表 151,643，天然处理多语言与未登录词；
- **对话模板**：Qwen2.5 内置官方模板，用 `tokenizer.apply_chat_template` 应用。典型多轮对话被格式化为：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好，请介绍一下你自己。<|im_end|>
<|im_start|>assistant
你好，我是通义千问...<|im_end|>
```

`<|im_start|>`/`<|im_end|>` 界定角色与内容边界，确保模型理解「谁在说话、指令是什么、从哪开始答」。**格式不符则模型无法正确理解指令**——这是数据预处理的核心。

### 2.3 四个问题定策略

1. **架构天性**：Decoder-only + 因果语言建模 → `task_type="CAUSAL_LM"`，数据必须构造成 `[提示] → [期望输出]` 的因果序列；
2. **沟通语言**：数据严格转成官方对话模板，最可靠的方式是 `apply_chat_template`，而非手工拼接字符串；
3. **注入部位**：`target_modules` 优先指向注意力（`q_proj, k_proj, v_proj, o_proj`）与 FFN（`gate_proj, up_proj, down_proj`）；
4. **先天优势**：微调是在「通才」基础上深造，若目标与知识/代码/数学相关，Qwen2.5 是绝佳起点，事半功倍。

## 3. 构建私有数据集：教师模型蒸馏

### 3.1 微调前先评估

用 4-bit 加载 `Qwen2.5-7B-Instruct`：

```python
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="cuda:0")
```

打印可见 `q_proj/k_proj/v_proj/o_proj`、`gate_proj/up_proj/down_proj` 均变为 `Linear4bit`，`SiLUActivation`、`Qwen2RMSNorm`、`Qwen2RotaryEmbedding` 就位——这些层名就是后续 `target_modules` 的依据。

定义 `chat()`：把 system+user 消息用 `apply_chat_template(..., add_generation_prompt=True)` 格式化 → `generate(max_new_tokens=256)` → 解码时跳过 prompt 部分。

> **Base 与 Instruct 的区别**：`Qwen2.5-7B`（Base）擅长续写、不直接对话；`-Instruct` 是在 Base 上用「指令-回答」数据再微调的对话模型，训练时接收的是带特殊格式的对话结构。这就是为什么必须用 `apply_chat_template`。

实测两个《黑神话：悟空》问题（「怎么成为天命人」「如何获得并合成出云棍」），模型回答**细节缺失且偏差**（甚至声称游戏未发布），说明确有必要微调。

### 3.2 用教师模型造数据

思路是「教师模型 + 私有文档 → 指令数据集」，把强模型的能力**蒸馏**到待微调的小模型上。

**配置**：用 235B 级模型（如 `Qwen/Qwen3-235B-A22B-Instruct-2507`）作教师，OpenAI 兼容客户端调 SiliconFlow API。选大教师是「金钱换时间」——单次生成质量高、少返工；预算有限可换小模型但需打磨 Prompt。演示中 API Key 不应硬编码。

**切分源文档**：读取百科整理的 `blackwukong.md`，用正则 `^(#{2,3})\s+(.+)$` 按二/三级标题切块、丢弃 <100 字的片段，得 14 个知识片段。

**生成基础问答对**：`SYS_PROMPT` 要求「严格输出 JSON：`{"instruction": 用户问题, "output": 权威完整答案}`，output 仅依据原文不臆测，禁止额外说明」；逐片段调用（`temperature=0.2, max_tokens=600, response_format={"type":"json_object"}`），解析后写入 `wukong_base_*.jsonl`——14 片段得 14 条。

**数据增强造多样问法**：对每条基础问题让教师生成 14 种等价问法（Prompt 要求严格输出 `{"paraphrases":[...]}`），去重后与同一答案配对写入最终数据集。一篇文档即加工出一百多条指令微调数据。

## 4. QLoRA 微调与评估

### 4.1 加载与量化

```python
train_set = load_dataset("json", data_files=train_jsonl, split="train")
tokenizer = AutoTokenizer.from_pretrained(checkpoint_id, trust_remote_code=True)
# pad_token='<|endoftext|>'(151643)，eos_token='<|im_end|>'(151645)，自带给 padding 所需设置
```

四参数量化配置是 QLoRA 的核心：

```python
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # 正态浮点 4-bit
    bnb_4bit_use_double_quant=True,     # 双量化
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(checkpoint_id, trust_remote_code=True,
                                                  quantization_config=bnb_cfg, device_map="cuda:0")
base_model = prepare_model_for_kbit_training(base_model)
```

> **LoRA 与 QLoRA 的区分**：LoRA 本体是冻结基础权重、只训低秩适配器，不限定基座精度（上节是 8-bit 基座 + FP16 训练）；QLoRA 是在 LoRA 框架下的一套具体工程方案——基座 4-bit（常用 NF4）+ 双量化 + 分页优化器（PagedAdamW）+ 仅以 16-bit（推荐 BF16）训练适配器。

### 4.2 target_modules 的省事写法

`peft` 内置各模型推荐映射表，直接取 `qwen2` 的推荐值，避免手工检查：

```python
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["qwen2"])
peft_model = get_peft_model(base_model, lora_cfg)
```

### 4.3 只对回答算损失

数据格式化必须走官方模板，并**把问题部分的 labels 置 -100**：

```python
prompt_ids = tokenizer.apply_chat_template(msgs_no_assist, ...)          # 只含 system+user
full_ids   = tokenizer.apply_chat_template(msgs_no_assist + [{"role":"assistant","content":ans}], ...)
cut = len(prompt_ids)
labels = [-100] * cut + full_ids[cut:]        # 只对答案部分计算损失
return {"input_ids": full_ids, "labels": labels}
```

### 4.4 训练超参

```python
args = TrainingArguments(
    output_dir=run_dir,
    per_device_train_batch_size=1, gradient_accumulation_steps=8,   # 有效批量 = 8
    learning_rate=1e-3, num_train_epochs=4,
    lr_scheduler_type="linear", warmup_ratio=0.03,
    logging_steps=1, optim="adamw_torch", bf16=True,
)
```

**bf16 优先于 fp16**：BF16 动态范围接近 FP32，能避免 FP16 常见的梯度下溢，训练更稳；Ampere 及以上 GPU 都支持。训练后 `save_pretrained(run_dir)` 只存适配器。

### 4.5 效果对比

同样两问，微调后：

```text
Q: 我该怎么成为天命人？
A: 想要成为天命之人，首先你需要收集“部分灵根”，然后到特定地点（如石卵处）激活……
Q: 如何获得并合成出云棍？
A: 在“黄风岭-挟魂崖”找到6个佛目珠后，前往“挟魂崖-枕石坪”BOSS石先锋所在区域……
```

细节准确、贴合游戏设定——私有知识被成功「吸收」。提醒：本例侧重流程演示，未系统控制过拟合，数据少、步数/学习率不当会出现「背题记忆」。

## 5. 何时该微调：提示工程 → RAG → 微调

技术选型应遵循**成本效益原则**，从低到高依次尝试：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_11_4_2.svg" width="70%" alt="技术选型路径" />
  <p>图 3：技术选型路径图</p>
</div>

两个维度理解：**横轴「LLM 优化」**（对模型本身的修改程度）——提示工程与 RAG 完全不改权重，微调直接改参数；**纵轴「上下文优化」**（对输入信息的增强程度）——提示工程只优化提问，RAG 引入外部知识库大幅丰富上下文。

选择路径由此清晰：

1. **提示词工程**：成本最低，适合任务简单、模型已有相关知识的场景；
2. **RAG**：模型缺特定/实时知识时，外挂知识库「开卷考试」；
3. **微调**：前两者只能「引导」和「提供知识」，无法改变模型的「行为」与「技能」。当目标变成**如何做**而非**知道什么**时——严格遵循某种输出格式（如特定 JSON 结构）、模仿特定对话风格、深度适配专业行话、把复杂指令蒸馏进权重优化 API 调用——微调才是最终且合适的选择。

> **补记**：Qwen 案例最值得记的是「顺序」——先评估（确认必要性）→ 再读模型（架构/分词器/模板决定配置）→ 再定策略（四个问题）→ 造数据（教师蒸馏+增强）→ 微调（NF4/双量化/BF16 + 只算答案损失）→ 再评估（同题对比）。以及最后那张选型图给出的判断准则：**微调改的是「怎么做」，不是「知道什么」**。这条准则能挡掉大量「其实上 RAG 就够」的错误微调需求。

## 写在最后

LoRA 一族解决了「怎么省着训」，但大模型对齐的另一半——让输出符合人类偏好——靠的是 RLHF 与 DPO。下一篇[RLHF技术详解](3-RLHF技术详解)从 InstructGPT 的三步法讲起，拆开奖励模型与 PPO 的机制。
