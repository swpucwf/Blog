# 动手跑通第一个 VLM

## 1. 两种最小可运行方式

多模态推理的入口通常只有两条路，其余都是它们的组合：

1. **本地加载模型**：用 `Transformers` 直接把视觉语言模型拉进显存，自己控制 processor 与生成参数；
2. **调用服务**：用 OpenAI 兼容接口把图片编码成 data URL 提交给服务端，业务层不碰权重。

两条路的价值不同：前者能看清「图像怎么进模型」，后者能看清「服务化长什么样」。先跑通前者，再切后者，排错时就不容易混淆是模型问题还是协议问题。

## 2. 方式一：Transformers 本地推理

依赖大致是：`torch`、`transformers >= 4.45`（含 Qwen2.5-VL 支持）、`qwen-vl-utils`（`process_vision_info`）、`python-dotenv`。

配置通过环境变量传入，避免把模型名与路径写死在脚本里：

```powershell
$env:MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
$env:IMAGE_PATH="sample_ui.png"
$env:QUESTION="请详细描述这张图，并指出其中最重要的视觉元素。"
python transformers_chat.py
```

Linux / macOS 换成 `export` 即可。核心推理代码只有几十行：

```python
import os
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")
QUESTION = os.getenv("QUESTION", "请详细描述这张图。")
IMAGE_PATH = os.getenv("IMAGE_PATH", "sample_ui.png")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_PATH},
            {"type": "text", "text": QUESTION},
        ],
    }
]

prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[prompt], images=image_inputs, videos=video_inputs,
    padding=True, return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=256)
trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
print(processor.batch_decode(trimmed, skip_special_tokens=True)[0])
```

### 2.1 这段代码在做什么

四个步骤对应链路里的四段职责：

1. **加载模型与 processor**：模型负责生成，processor 负责把图文消息编码成张量。
2. **构造图文消息**：`content` 是列表，图片与文本并列，顺序即模型看到的顺序。
3. **套 chat template**：`apply_chat_template` 把消息列表转成模型训练时见过的对话格式，模板套错会直接导致回答空白或复读。
4. **处理视觉输入并生成**：`process_vision_info` 把图片解析成视觉张量，`generate` 后要把输入 prompt 部分裁掉，只保留新增输出。

值得重点看的三个点：图像消息在代码里如何表示、模板如何拼接图文、生成结果如何裁剪。这三点决定了后面换成任何模型时该改哪几行。

## 3. 方式二：OpenAI 兼容接口

服务端已经加载模型时，应用层只需两件事：把图片转成 data URL，用 `chat.completions` 提交图文消息。

```python
import base64, mimetypes, os
from pathlib import Path
from openai import OpenAI

def to_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
)
response = client.chat.completions.create(
    model=os.getenv("MODEL_ID", "your-vlm"),
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": os.getenv("QUESTION", "这张图传递了什么信息？")},
            {"type": "image_url", "image_url": {"url": to_data_url("sample_ui.png")}},
        ],
    }],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

运行方式同样走环境变量：

```powershell
$env:OPENAI_BASE_URL="你的接口地址"
$env:OPENAI_API_KEY="你的密钥"
$env:MODEL_ID="你的模型名"
$env:IMAGE_PATH="sample_ui.png"
$env:QUESTION="这张图传递了什么信息？"
python openai_compatible_client.py
```

注意 data URL 必须带 MIME 前缀（`data:image/png;base64,...`），只传本地路径或裸 base64 都会在服务端报格式错误。

## 4. 为什么值得保留接口层

如果后续要做 Web Demo、API 服务、Agent 工作流或多模型路由，OpenAI 兼容接口比在业务层直接加载模型灵活得多：换模型只改 `base_url` 与 `model` 两个值，业务代码不动。代价是多一层协议约定，需要服务端正确支持多模态 content schema。

## 5. 常见报错定位

| 现象 | 优先怀疑 |
|---|---|
| 回答空白、极短、复读模板 | `apply_chat_template`、消息结构、processor 是否吃到图 |
| HTTP 400/422，服务端报缺图或格式错 | `image_url` / `data:` URL、字段名、多模态 schema |
| 显存溢出、进程被 kill | 分辨率、`max_new_tokens`、模型体量与量化 |

按顺序排查更省时间：先看模型与模板，再看协议，最后看资源。

**模型能加载但回答奇怪**，检查模板是否匹配模型、图像是否真的传给 processor、消息里的 `type` 字段是否符合要求。

**接口通但说图像格式不对**，检查是否把本地路径当 URL 发了、data URL 是否缺 MIME 前缀、服务端是否支持多模态消息。

**显存不够**，可以换更小的模型、做量化、降低输入分辨率、减少 `max_new_tokens`——按对效果影响从小到大依次尝试。

## 6. 跑通后的三个小实验

跑通脚本只是起点，更值钱的是立刻把它变成实验工具。

- **同一张图问不同问题**：观察模型是否围绕问题聚焦不同证据。描述类问题看整体，细节类问题看局部，两者差异最能暴露模型的注意力分配。
- **换成截图或文档**：感受自然图像能力与 OCR/文档能力的落差。同一模型在 COCO 风格图上表现不错，在密集表格上可能直接崩。
- **问图中不存在的东西**：「这张图里有没有红色汽车」这类问题专门测幻觉，回答「有」就是典型的存在性幻觉。

观察时记录四件事：回答是否基于图像证据、是否出现模板化表述、对截图小字与图表是否明显变差、同一问题多次请求是否稳定。

## 7. 从「会跑命令」到「会做实验」

把单次提问扩成实验任务，只需固定一个变量、变化另一个：

1. 固定 5 张图片，比较同一 prompt 下的稳定性；
2. 固定 1 张图片，比较不同 prompt 对结果的影响；
3. 固定任务，比较本地推理与服务接口的回答差异。

再进一步，就是写一个小循环：遍历一个目录下的图片，把 `image_path / question / answer` 追加写入 JSONL。这份文件既是实验记录，也是后续评测集的雏形——字段稍作调整就能喂给评测脚本。

## 写在最后

这一篇的重点不是记住某个模型的启动命令，而是把「图像 → processor → 模板 → 生成」这条链路变具体。摸清 `messages` 与 chat template 之后，下一篇[图像问答 Demo](5-图像问答Demo)会把同一套调用嵌进 Gradio 界面，请求体与这里保持一致，界面里才不会出现「脚本对、界面错」。
