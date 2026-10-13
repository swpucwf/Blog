## 1. Agent 如何调用大模型：理解 API 的上下文结构
### 1.1 消息的四种角色
API核心载体为`messages`消息数组，数组内每条消息通过`role`角色标识区分来源与含义：

1.  **system（系统提示词）**：由开发者预先编写，用于设定Agent身份、行为准则、业务约束，拥有最高指令优先级；一次对话一般仅一条，放置在消息列表最前端。
2.  **user（用户消息）**：终端用户的原始提问，是Agent需要响应的目标请求。
3.  **assistant（助手消息）**：大模型输出的返回内容，既可以是普通文本答案，也可以是工具调用请求。多轮交互中，该消息会放回消息列表，让模型感知自身过往的决策。
4.  **tool（工具结果消息）**：Agent框架执行完工具之后，把返回结果封装为此类消息回传给模型；依靠`tool_call_id`，和模型之前下发的工具调用请求一一绑定，让模型识别结果归属。

> 补充：`tools`是请求顶层独立字段，不属于messages消息列表，用来向模型声明可调用工具的名称、功能描述、入参JSON结构。

### 1.2 单轮对话：最简单的API调用

属于无工具调用的基础交互模式，请求消息列表仅包含`system` + `user`两条消息，模型直接返回`assistant`文本答复。 **核心关键点：大模型API是无状态接口。模型本身不会记忆任何上一轮对话内容，每一次API调用，都必须把当前轮次需要的全部信息完整送入messages列表。** 附带JS请求/响应JSON示例，直观展示基础请求体结构。
![](https://i-blog.csdnimg.cn/direct/402d30dcb1504579966bab4440e5640a.png)


### 1.3 带工具调用的多轮交互：Agent的核心循环
![](https://i-blog.csdnimg.cn/direct/77a9ebf5e391414c9066c105f57011e4.png)



这是真实Agent业务场景，也是**ReAct思考-行动-观察循环在API层面的落地实现**。 场景：用户问题需要外部实时数据，模型自身知识库无法作答，需要调用外部工具。

1.  第一轮API请求：传入system、user消息 + tools工具定义；模型分析问题，判断需要调用外部工具，返回携带`tool_calls`的assistant消息，不输出最终文本。
2.  权责划分：**模型只负责决策：选哪些工具、填写什么入参；Agent框架负责执行：发起外部接口、运行代码获取真实数据**。多个无依赖的工具可一次性请求，并行执行，提升效率。
3.  框架拿到工具返回数据，封装成多条tool消息，追加到原有的完整消息列表尾部，携带全部历史记录发起新一轮API请求。
4.  模型读取历史对话+工具返回结果进行判断：
    -   信息充足：直接输出assistant文本，作为最终答案，循环终止；
    -   信息不足：再次返回tool\_calls，继续调用工具，重复循环。 附带完整JS请求/响应样例，展示每一轮messages消息列表的追加变化，重点体现三条细节：每次请求携带完整历史、assistant工具调用消息保留、tool消息依靠tool\_call\_id关联。

### 1.4 用代码实现Agent的核心循环【代码逐行详细讲解】

> 目标：用Python复刻上一节ReAct工具调用循环，代码实现了完整Agent运行逻辑，下面对代码分模块逐段拆解。

```
# 1. 导入SDK包，创建模型客户端
from openai import OpenAI
client = OpenAI()
```

-   `from openai import OpenAI`：导入OpenAI官方SDK，封装了Chat Completions API网络请求，不用手动拼接HTTP请求、解析JSON。
-   `client = OpenAI()`：实例化客户端对象，后续所有调用模型的请求，都通过这个`client`发起；实际项目需要配置API密钥、服务地址。

```
# 2. tools工具定义数组，传给大模型
tools = [
    {
"type": "function",
"function": {
"name": "get_current_time",
"description": "Get the current date and time in a specific timezone",
"parameters": {
"type": "object",
"properties": {
"timezone": {"type": "string", "description": "Timezone name, e.g. America/Vancouver"}
                },
            },
        },
    },
    {
"type": "function",
"function": {
"name": "get_weather",
"description": "Get the current weather for a specific city",
"parameters": {
"type": "object",
"properties": {
"city": {"type": "string", "description": "City name"},
"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
            },
        },
    },
]
```

-   `tools`数组对应API顶层tools参数，**给大模型看的工具说明书**。
-   `name`：工具函数名称，模型决定调用工具时，会返回这个名字。
-   `description`：工具功能描述，让大模型理解这个工具能干什么，什么时候应该调用。
-   `parameters`：定义工具需要的入参、参数类型、参数说明；模型会严格按照这个结构生成参数JSON。

```
# 3. 工具执行函数：真正运行工具逻辑
def execute_tool(name, arguments):
if name == "get_current_time":
return '{"datetime": "2025-09-13T05:18:47", "day_of_week": "Saturday"}'
elif name == "get_weather":
return '{"temperature": 13.2, "unit": "celsius", "conditions": "clear", "humidity": 93}'
```

-   `execute_tool(name, arguments)`：**由Agent框架执行，大模型不会运行这段代码**。
-   入参`name`：模型选择的工具名；`arguments`：模型生成的JSON格式参数。
-   当前是模拟桩函数，直接返回写死的测试结果；生产环境这里需要解析arguments，调用真实第三方API、数据库查询等。
-   返回结果必须是字符串JSON，后续会包装成`tool`角色消息送回模型。

```
# 4. 初始化消息列表messages
messages = [
    {"role": "system", "content": "You are a helpful assistant. Use tools to get real-time information when needed."},
    {"role": "user", "content": "What's the current time and weather in Vancouver?"},
]
```

-   `messages`就是API里的消息列表，**整个Agent运行过程中，这个列表会不断追加消息，持续保存完整对话历史**。
-   初始仅包含system系统指令、user用户提问，对应对话起点。

```
# 5. Agent核心while循环，ReAct循环主体
while True:
    # 调用大模型接口，传入当前全部消息和工具定义
    response = client.chat.completions.create(
        model="Qwen3-0.6B", messages=messages, tools=tools
)
    # 取出模型返回的assistant消息
    assistant_message = response.choices[0].message
    # 将模型本次返回消息追加进消息列表，永久保存本次决策
    messages.append(assistant_message)

    # 分支1：模型没有请求工具调用，输出答案，结束循环
if not assistant_message.tool_calls:
print(assistant_message.content)
break

    # 分支2：模型返回tool_calls，遍历执行每一个工具调用
for tool_call in assistant_message.tool_calls:
        # 调用工具执行函数，拿到工具返回结果
        result = execute_tool(tool_call.function.name, tool_call.function.arguments)
        # 把工具执行结果封装为tool角色消息，追加进messages
        messages.append({
"role": "tool",
"tool_call_id": tool_call.id,
"content": result,
        })
# 循环回到开头，携带更新后的完整messages再次请求大模型
```

逐行解析主循环：

1.  `while True`：开启无限循环，代表Agent可以反复思考、调用工具；⚠️生产代码**必须增加最大迭代次数max\_iterations**，防止死循环（模型反复调用相同工具无法终止）。
2.  `client.chat.completions.create()`：发起API请求，把当前完整`messages`历史和`tools`工具定义传给大模型。
3.  `assistant_message = response.choices[0].message`：读取模型返回的消息对象，里面可能包含文本content，或者tool\_calls工具调用请求。
4.  `messages.append(assistant_message)`：**关键操作**，把模型刚刚生成的决策存入消息列表，下一轮请求模型时，模型能看见自己上一轮选择调用了哪些工具。
5.  `if not assistant_message.tool_calls` 判断分支：
    -   没有tool\_calls：模型已经收集到足够信息，生成最终文本答案，打印内容，break跳出循环，Agent任务结束。
    -   存在tool\_calls：模型要求调用外部工具，进入工具执行逻辑。
6.  `for tool_call in assistant_message.tool_calls`：遍历模型请求的全部工具调用，支持一次性多个并行工具调用。
7.  `execute_tool(...)`：执行对应工具，获取结果。
8.  构造`role:tool`消息，并且带上`tool_call_id`，用来和模型刚才的工具调用请求一一关联，追加到messages列表末尾。
9.  一轮工具执行完毕，代码回到`while`循环最上方，使用更新、变长后的`messages`再次请求大模型，开启新一轮思考。

> messages列表三阶段变化跟踪

1.  初始状态：仅system、user两条消息
2.  第一轮模型调用后：追加assistant（带tool\_calls）+ 多条tool结果消息
3.  第二轮模型调用：传入全部历史，模型生成最终assistant文本消息，追加到列表，循环结束

**核心结论：Agent框架最核心的工作就是维护、持续更新messages消息列表；本章后续所有上下文工程技术，本质上都是对这个消息列表的内容、长度、结构做优化。**

### 1.5 从API视角看上下文的构成
Agent送入大模型的全部上下文拆分为两大组成部分：
1.  **静态前缀**：System Prompt + Tools工具定义，在一次完整对话生命周期内基本固定不变。
2.  **动态轨迹（对话历史）**：user、assistant、tool消息，随每一轮交互不断追加、持续变长。 该“静态前缀+动态轨迹”的分层结构，是后续KV Cache推理加速、上下文压缩等优化技术的底层理论基础。

![](https://i-blog.csdnimg.cn/direct/1c5e48d2ff214fffa3af2849a8b0176d.png)



