# 图像问答 Demo

## 1. 为什么先做 Demo 再谈复杂系统

一个简单 Demo 的价值常被低估。它能同时验证四件事：

1. 模型接口是否稳定；
2. 图像输入链路是否正确；
3. 用户问题与输出格式是否合理；
4. 这项能力是否真的值得继续做成产品。

如果连一个小 Demo 都经常答错、答慢、答偏，直接上复杂系统只会把问题放大。Demo 同时也是评测集的最小试验场：界面里复现的 bug，可以立刻回到评测集里对照同一条样本。

## 2. 最小结构

一个可用的图文问答 Demo 只需要四段：

- 前端：`Gradio`，负责上传图片、输入问题、展示答案；
- 后端：OpenAI 兼容接口，业务层不加载模型；
- 转换：图片转 data URL，符合标准多模态消息格式；
- 状态：把「未上传 / 完成 / 失败」和耗时反馈到界面。

这四段覆盖了大量多模态应用的最小主链路，剩下的都是在这条主链路上加分支。

## 3. 环境与启动

依赖装好后，用环境变量注入配置，避免把接口地址与模型名写进代码：

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=EMPTY
MODEL_ID=your-vlm
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7860
DEFAULT_MODE=自由问答
```

启动后打开本地页面即可。页眉会显示当前 `OPENAI_BASE_URL` 与 `MODEL_ID`，终端也会打印同一行——改配置后需要重启进程，否则页面显示的仍是旧值。

可以直接试的提问：

- 请描述这张图的主要内容
- 这张截图里报了什么错误
- 请提取图片中的关键信息
- 请用三点总结这张图的核心内容

## 4. 四种基础任务模式

同一套请求体，用不同的提示模板就能覆盖四类常见场景：

| 模式 | 模板要点 |
|---|---|
| 自由问答 | 直接把用户问题作为 prompt |
| 截图报错分析 | 定位关键问题、解释可能原因、给出排查建议 |
| 关键信息提取 | 按条目列出图中关键信息 |
| 商品图卖点总结 | 依据外观与使用场景总结三条卖点 |

把模板集中成一张字典，既方便新增场景，也方便把模板文本纳入版本管理：

```python
TASK_PROMPTS = {
    "自由问答": "{question}",
    "截图报错分析": "请阅读这张截图，定位最关键的问题，解释可能原因，并给出下一步排查建议。\n用户补充问题：{question}",
    "关键信息提取": "请从图片中提取关键信息，并尽量按条目列出。\n用户补充问题：{question}",
    "商品图卖点总结": "请根据图片中的商品外观和使用场景，总结 3 条简洁卖点。\n用户补充问题：{question}",
}

def build_prompt(task_mode: str, question: str) -> str:
    template = TASK_PROMPTS.get(task_mode, "{question}")
    return template.format(question=question.strip() or "请分析这张图片。")
```

## 5. 代码骨架

后端的核心是一个函数：校验输入、转 data URL、拼 prompt、发请求、回填耗时与状态。

```python
import base64, mimetypes, os, time
from pathlib import Path
import gradio as gr
from openai import OpenAI

def to_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"

def ask_vlm(image_path: str, question: str, task_mode: str) -> tuple[str, str]:
    if not image_path:
        return "请先上传一张图片。", "状态：未执行"
    start = time.perf_counter()
    try:
        client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
                        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))
        resp = client.chat.completions.create(
            model=os.getenv("MODEL_ID", "your-vlm"),
            messages=[{"role": "user", "content": [
                {"type": "text", "text": build_prompt(task_mode, question)},
                {"type": "image_url", "image_url": {"url": to_data_url(image_path)}},
            ]}],
            max_tokens=512,
        )
        elapsed = time.perf_counter() - start
        return resp.choices[0].message.content, f"状态：已完成，模式={task_mode}，耗时={elapsed:.2f}s"
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return f"请求失败：{exc}", f"状态：失败，模式={task_mode}，耗时={elapsed:.2f}s"

with gr.Blocks(title="图像问答 Demo") as demo:
    gr.Markdown("# 图像问答 Demo")
    image_input = gr.Image(type="filepath", label="上传图片")
    task_mode = gr.Dropdown(choices=list(TASK_PROMPTS), value="自由问答", label="任务模式")
    question_input = gr.Textbox(label="问题", lines=4)
    submit_btn = gr.Button("开始分析")
    answer_output = gr.Textbox(label="模型回答", lines=12)
    status_output = gr.Textbox(label="运行状态", lines=1)
    submit_btn.click(ask_vlm, [image_input, question_input, task_mode],
                     [answer_output, status_output])

if __name__ == "__main__":
    demo.launch(server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
                server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")))
```

关键点在于请求体与命令行脚本完全一致：同样的 `messages` 结构、同样的 data URL 编码。界面只负责收集输入和展示输出，不引入第二套请求习惯——否则「脚本里对、界面里错」的问题会反复出现。

## 6. 从最小版走向产品原型

确认最小版能跑通后，按价值排序加能力：

1. **输出模式切换**：自由回答、JSON 输出、关键信息抽取、摘要。
2. **多图输入**：允许上传多张图做比较、归纳、前后对照。
3. **历史记录**：保留上下文，做多轮视觉对话。
4. **任务模板**：预设「截图报错分析」「商品图卖点提取」「文档字段抽取」等场景。

结构化输出最值得早做：业务需要字段时，不要让模型自由发挥，而是给 JSON 模板 + 字段校验 + 规则回退，把不确定性收敛在可检查的边界内。

## 7. 比特效更重要的是可观测性

Demo 迭代时优先记录：

- 用户原始问题；
- 图片文件名或摘要；
- 模型输出；
- 请求耗时；
- 失败样例。

有了这五项，才能判断下一步该优化 prompt、换模型、改接口，还是调图像预处理。界面上至少保留：用户可见的状态提示、基础异常信息、默认任务模式、输出格式约束。

## 8. 一个可演示 Demo 的四个标准

如果这个仓库要继续对外复用，每个 Demo 都值得满足：

1. 一条命令能启动；
2. 有清楚的环境变量说明；
3. 能用默认问题直接演示；
4. 支持继续接别的模型。

把配置外置、把模型名参数化，Demo 才不会绑死在某一台机器或某一个服务上。

## 9. 三个可直接落地的场景扩展

**截图报错分析**：模板里固化「错误定位 → 原因解释 → 下一步操作建议」三段，最容易做出即时可感知的效果。

**文档字段抽取**：输出从自由文本改为 JSON，字段固定为标题、日期、金额、关键结论；OCR 接入、版面分析与规则校验的完整做法，见[OCR 与文档理解](7-OCR与文档理解)。

**商品图卖点生成**：先识别商品元素，再按电商文案格式生成卖点，重点在风格一致性与幻觉控制。

## 写在最后

多模态项目最合适的起步方式往往不是先训练，而是先做一个可交互 Demo。界面稳定之后，接数据、接 Agent、接服务编排都会顺很多。下一篇[从单模态 Agent 到多模态 Agent](6-从单模态Agent到多模态Agent)把这里验证过的接口，升级成可拆步骤、可降级的工作流。
