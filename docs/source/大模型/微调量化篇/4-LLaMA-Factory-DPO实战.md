# LLaMA-Factory DPO实战

## 1. LLaMA-Factory：零代码微调平台

DPO 省去了独立奖励模型与复杂在线采样，直接在偏好数据上优化策略。这一篇用开源微调框架 **LLaMA-Factory**，配合轻量指令模型 **Qwen2.5-0.5B-Instruct**，在单张消费级显卡（甚至 CPU）上跑通 DPO 对齐。

LLaMA-Factory 的特点：

- **广泛的模型支持**：LLaMA、Qwen、Baichuan、ChatGLM、Mistral、Yi 等主流开源模型；
- **全流程覆盖**：预训练 → SFT → RLHF（奖励模型、PPO、DPO、KTO、ORPO 等）；
- **高效微调技术**：内置 LoRA、QLoRA、DoRA、GaLore 等 PEFT 方法，大幅降显存；
- **易用界面**：**LLaMA-Board** Web UI，可视化配置参数、监控训练、对话测试；
- **硬件友好**：支持 DeepSpeed、FlashAttention 与 4/8-bit 量化训练，消费级显卡也能跑。

## 2. 环境搭建

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
conda activate peft
pip install -e .[metrics]
llamafactory-cli version
# | Welcome to LLaMA Factory, version 0.9.4.dev0 |
```

启动 Web UI：

```bash
llamafactory-cli webui
# Visit http://127.0.0.1:7860
```

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_2_1.png" width="80%" alt="LLaMA-Factory Web UI" />
  <p>图 1：LLaMA-Factory Web UI 界面</p>
</div>

## 3. DPO 训练准备

RLHF（DPO）通常建立在模型已有指令遵循能力的基础上。这里直接用已做过指令微调的 `Qwen2.5-0.5B-Instruct`，跳过 SFT 直接进入 DPO。

### 3.1 配置模型

在 WebUI 下拉框选择 **`Qwen2.5-0.5B-Instruct`**（聊天/指令版本），设置下载源（默认 Hugging Face，网络问题可换 ModelScope 或 OpenMind）。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_2_2.png" width="80%" alt="模型选择与配置" />
  <p>图 2：模型选择与配置</p>
</div>

### 3.2 配置微调参数

- **微调方法**：保持默认 `lora`；
- **量化配置**：显存小可将 `Quantization bit` 设为 8 或 4 启用量化训练（QLoRA），显存充足或求更好效果则保持不量化；`Quantization method` 默认 `bnb`；
- **Chat template**：Base 模型可选 `default`/`alpaca`/`vicuna`；**Instruct/Chat 模型务必选对应模板**（Qwen 系列选 `qwen`），且**训练与推理必须用同一模板**；
- **RoPE scaling**：用于扩展上下文长度；**Booster** 保持 `auto`，自动选择 FlashAttention 等加速后端。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_2_3.png" width="80%" alt="微调方法与量化配置" />
  <p>图 3：LoRA 微调与 qwen 对话模板配置</p>
</div>

### 3.3 选用偏好数据集

左上角 Stage 下拉选 `DPO`；数据集选 LLaMA-Factory 自带的 **`dpo_zh_demo`**——含 chosen/rejected 偏好对比对；点 **Preview dataset** 预览确认格式无误。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_2_4.png" width="80%" alt="数据集选择与预览" />
  <p>图 4：DPO 数据集选择与预览</p>
</div>

## 4. 启动训练

### 4.1 训练参数

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_2_5.png" width="80%" alt="训练参数配置" />
  <p>图 5：训练参数配置</p>
</div>

关键参数：

| 参数 | 取值 | 说明 |
|---|---|---|
| Learning rate | `1e-4` | 学习率 |
| Epochs | `3.0` | 训练轮数 |
| Compute type | `fp16` | 兼容性好；显卡适配可选 bf16 |
| Cutoff length | `2048` | 截断长度 |
| Batch size | `1` | 视显存而定 |
| Gradient accumulation | `16` | 建议 Batch × 累积 ≈ 16~32 |
| LoRA rank | `8` | 秩，决定微调参数量（默认 8） |
| LoRA alpha | `16` | 缩放系数，惯例为 rank 的 2 倍 |

### 4.2 启动

设置输出目录 → **Preview command** 预览命令 → **Start**。训练完成后适配器保存在 `saves/Qwen2.5-0.5B-Instruct/lora/Qwen2.5-0.5B-dpo-demo`，Train 面板可看 Loss 曲线：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_2_6.png" width="80%" alt="预览命令并启动训练" />
  <p>图 6：预览命令并启动训练</p>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_2_7.png" width="80%" alt="训练 Loss 曲线" />
  <p>图 7：训练 Loss 曲线</p>
</div>

## 5. 效果测试：训练前后的行为风格

没有测试集，跳过 Evaluate&nbsp;&&nbsp;Predict 面板，直接用 LLaMA-Board 的 `Chat` 测试。

### 5.1 先测原始模型

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_2_8.png" width="80%" alt="加载原始模型" />
  <p>图 8：加载原始 Qwen2.5-0.5B-Instruct</p>
</div>

**测试配置**：System prompt 要求「尽量忠实地完成任务，执行时逐步思考并合理解释步骤」；Input 为一段关于盐与脂肪促进过量进食的长文阅读理解题（四选一）。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_2_9.png" width="80%" alt="原始模型对话测试" />
  <p>图 9：原始模型对话测试</p>
</div>

原始模型回答**简短且武断**：完全忽略「逐步思考、解释步骤」的指令，仅凭文中提到「午餐」的表面关联直接猜测。这种「急于给答案」的风格在未对齐模型中很常见。

### 5.2 再测 DPO 模型

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_2_10.png" width="80%" alt="加载 DPO Adapter" />
  <p>图 10：加载 DPO 训练后的 Adapter</p>
</div>

在 Chat 面板把 `Checkpoint path` 指向刚训好的 Adapter 目录，重新 **Load model**，用同样的问题测试：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_12_2_11.png" width="80%" alt="DPO 模型对话测试" />
  <p>图 11：DPO 模型对话测试</p>
</div>

前后对比清晰：

- **原始模型**：简短直接、缺解释、倾向猜测，「急于完成任务」但缺严谨性；
- **DPO 模型**：更**严谨**——给出选项的同时忠实执行「解释步骤」的要求，还明确指出原文信息的局限（「缺乏关于这些人群饮食习惯的直接证据」），展示了推理过程。

这才是 RLHF 的价值：不只是让模型「做对题」，更让它学会人类偏好的**交互风格**。通过 `dpo_zh_demo` 训练，模型初步对齐了「诚实且严谨」的偏好风格。

> **工程提示**：可视化界面便于上手，但实际项目往往更倾向命令行（`llamafactory-cli train ...`），以获得对训练过程更细致的控制与可复现性。

> **补记**：这次实战把「对齐」拆成了两个可观测的层面——**能力**（答对题）与**风格**（怎么答）。SFT 主要解决前者，DPO 这类偏好优化改的是后者：同样的知识，回答从「一句话猜」变成「分步推理＋声明局限」。这也解释了为什么对齐数据（chosen/rejected）的设计比算法更重要：你喂什么样的偏好，模型就长成什么样的「性格」。

## 写在最后

微调与对齐之后，模型动辄几十上百 GB，如何部署到有限显存里？下一篇[模型量化实战](5-模型量化实战)从量化原理讲到 int8/int4 实践，把大模型真正塞进显存。
