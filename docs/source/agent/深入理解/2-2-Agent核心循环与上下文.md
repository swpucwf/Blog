# Agent 核心循环与上下文

## 1. 一次带工具的请求

用户问：「温哥华现在几点、天气怎么样？」模型凭自身知识答不了，需要调用外部工具。下面看 Agent 框架与模型之间的每一步交互。

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/aiagent_fig2-3.svg)

<p>图 1：两次模型 API 调用的完整交互序列</p>

图中的两次调用都指**调用模型 API**，不是先后调用两个工具。这个例子里，`get_current_time` 的时区与 `get_weather` 的城市、单位都可以直接确定；天气服务会自行返回最新天气，不依赖时间工具的输出，因此框架可以并行执行它们。如果后一个工具的参数必须来自前一个结果，模型就需要在后续一轮再发起调用，两个工具只能串行。

### 1.1 第一次调用：声明工具并发出请求

```javascript
{
  "model": "Qwen3-0.6B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed."},
    {"role": "user", "content": "What's the current time and weather in Vancouver?"}
  ],
  "tools": [
    {"type": "function", "function": {
      "name": "get_current_time",
      "description": "Get the current date and time in a specific timezone",
      "parameters": {"type": "object", "properties": {
        "timezone": {"type": "string", "description": "Timezone name, e.g. America/Vancouver"}
      }}
    }},
    {"type": "function", "function": {
      "name": "get_weather",
      "description": "Get the current weather for a specific city",
      "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      }}
    }}
  ]
}
```

这份 `tools` 清单是开发者事先注册好的静态元数据——名称、描述、参数 schema 都与用户这次问了什么无关。无论问天气还是订机票，发出去的都是同一份清单；示例只列两个工具是为了让请求体短一些，真实 Agent 往往一次挂上几十个工具。

**不是 Agent 先把用户输入拆成「查时间」和「查天气」再生成工具描述**——拆解发生在模型一侧，就是下面响应里的 `tool_calls`。

### 1.2 模型返回工具调用请求

```javascript
{
  "choices": [{"message": {
    "role": "assistant",
    "content": null,
    "tool_calls": [
      {"id": "call_abc123", "type": "function", "function": {
        "name": "get_current_time",
        "arguments": "{\"timezone\": \"America/Vancouver\"}"
      }},
      {"id": "call_def456", "type": "function", "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"Vancouver\", \"unit\": \"celsius\"}"
      }}
    ]
  }}]
}
```

模型没有直接回答，而是返回两个工具调用请求：它判断「当前时间」和「天气」需要工具，且两者没有依赖，可以并行。**模型只发出请求，真正执行工具的是 Agent 框架**——模型负责决策，框架负责执行。

### 1.3 执行工具并第二次调用

框架执行工具后，把**完整对话历史 + 工具结果**一起送回：

```javascript
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "What's the current time and weather in Vancouver?"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_abc123", "function": {"name": "get_current_time", "arguments": "..."}},
      {"id": "call_def456", "function": {"name": "get_weather", "arguments": "..."}}
    ]},
    {"role": "tool", "tool_call_id": "call_abc123",
     "content": "{\"datetime\": \"2025-09-13T05:18:47\", \"day_of_week\": \"Saturday\"}"},
    {"role": "tool", "tool_call_id": "call_def456",
     "content": "{\"temperature\": 13.2, \"unit\": \"celsius\", \"conditions\": \"clear\", \"humidity\": 93}"}
  ],
  "tools": ["...same as above..."]
}
```

三个关键细节：

1. **第二次请求包含第一次的全部历史**——模型不会「记住」上一次对话，框架必须每次把完整历史送回；
2. **第一次的 assistant 消息被原样放回**，让模型看到自己之前做了什么决策；
3. **tool 消息通过 `tool_call_id` 与调用关联**，模型据此知道哪个结果对应哪个请求。

模型随后返回文本回复、不再请求工具，循环结束。这个「请求 → 工具调用 → 执行 → 送回结果 → 再请求」的循环，就是 ReAct 在 API 层的实现。用户若追问「那东京呢？」，框架把追问追加到历史末尾，再发起调用，循环继续。

## 2. 用代码实现核心循环

```python
from openai import OpenAI

client = OpenAI()

tools = [
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Get the current date and time in a specific timezone",
        "parameters": {"type": "object", "properties": {
            "timezone": {"type": "string", "description": "Timezone name"}
        }},
    }},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a specific city",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        }},
    }},
]

def execute_tool(name, arguments):
    if name == "get_current_time":
        return '{"datetime": "2025-09-13T05:18:47", "day_of_week": "Saturday"}'
    if name == "get_weather":
        return '{"temperature": 13.2, "unit": "celsius", "conditions": "clear"}'

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the current time and weather in Vancouver?"},
]

while True:
    response = client.chat.completions.create(model="Qwen3-0.6B", messages=messages, tools=tools)
    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    if not assistant_message.tool_calls:
        print(assistant_message.content)
        break

    for tool_call in assistant_message.tool_calls:
        result = execute_tool(tool_call.function.name, tool_call.function.arguments)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })
```

核心逻辑只有一个循环和一个判断：**有 `tool_calls` 就执行并继续，没有就输出并退出**。生产代码需要 `max_iterations` 上限，否则 Agent 可能反复调用同一个工具。

跟踪 `messages` 的变化：

```text
初始：system, user
第 1 次调用后：+ assistant(tool_calls), + tool(时间), + tool(天气)
第 2 次调用后：+ assistant(最终回复)，循环结束
```

**Agent 框架的核心工作就是管理这个 messages 列表**——在合适时机追加消息，再把整个列表送给模型。后续所有上下文技术，本质上都是在优化这个列表的内容与结构。

## 3. 从 API 看上下文的构成

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/aiagent_fig2-4.svg)

<p>图 2：每次调用模型时的上下文构成</p>

上半部分（System Prompt + Tool Definitions）在整个对话中不变，下半部分（对话历史，也就是轨迹）随交互增长。这正是「静态前缀 + 轨迹」在 API 层的具体样子，也是后续 KV Cache 优化与压缩的基础——理解它，才能理解为什么「前面不能动、后面可以压缩」。

把布局决策写成伪代码：

```python
stable_prefix = system_message
stable_tools = core_tool_schemas
trajectory = load_message_history(session)
status_message = make_status_message(derive_current_state(trajectory))

if estimated_tokens(stable_prefix, trajectory, status_message) > budget:
    trajectory = compress_old_evidence(
        trajectory,
        preserve=[decisions, constraints, failures, citations],
    )

request.messages = [stable_prefix] + trajectory + [status_message]
request.tools = stable_tools
response = call_model(request)
```

系统提示词与核心工具定义尽量稳定；旧工具输出只在接近预算时成批压缩；当前状态放在轨迹尾部，让模型不必从长历史中重新推导。

## 4. 小模型也能跑工具调用

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/aiagent_fig2-5.svg)

<p>图 3：本地 LLM 工具调用架构</p>

一个常见误解是「工具调用必须大模型」。本地部署 0.6B 小模型的实验表明：在合理的提示词与系统架构下，小模型也能可靠地完成工具调用。

值得观察的细节：

- **内部思考**：支持思维链的模型会先在 `<think>` 标签内分析意图、评估工具、规划顺序，再输出工具调用；
- **输出顺序**：思考 → 文本回复 → 工具调用。流式实现可以在 `<think>` 出现时切换到「思考中」，参数完整生成并通过校验后立即执行，不必等后续调用；
- **并行调用**：模型发现两个子问题无依赖，会在一次输出中同时生成两个请求，框架可并行执行；
- **终止判断**：工具结果送回后，模型判断信息是否足够；够了就输出最终回复，不够就继续请求；
- **缓存影响**：保持系统提示词不变，第二次首 token 明显更快；改动开头几个字符，前缀缓存失效，需要重新计算。

实验的结论不是「小模型可以替代大模型」，而是：**模型大小重要，但不是唯一决定因素**。提示词设计、工具 Schema、上下文结构，都会显著影响工具调用的可靠性。

## 5. 补记

这段循环里最容易被忽略的是「谁负责什么」：模型只负责决策，框架负责执行、校验、追加结果。很多「模型不会用工具」的问题，实际是框架把工具结果放错了角色、丢了 `tool_call_id`，或者没有把历史完整送回。

另一个工程点是并行与串行的判断。模型在同一轮请求多个工具，说明它认为这些调用互不依赖；但如果后一个参数其实依赖前一个结果，模型就会给出错误参数。框架能做的是按依赖关系执行，并在结果校验失败时把错误作为观察送回，让模型下一轮修正。

## 写在最后

下一篇[KV Cache 友好的上下文设计](2-3-KVCache友好的上下文设计)解释一个看似无害的改动为何能让首 token 延迟涨一个量级：前缀、Chat Template 与缓存之间到底发生了什么。
