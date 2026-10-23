# ReAct 范式

在 ReAct 之前，智能体的两条路都不完整：纯思考（思维链）能推理却闭门造车，遇到事实问题就编；纯行动（直接输出动作）能干活却没规划，错了也不知道回头。ReAct 的洞察很朴素——思考与行动互为条件，把两者交替起来跑。

<img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/4-1.png" width="90%" alt="ReAct 的思考—行动—观察循环" />

## 1. 一个循环，三种输出

ReAct 用提示词约束模型每一步只能产出固定轨迹：

- **Thought**：内心独白，分析现状、拆解任务、规划下一步、反思上一步结果。
- **Action**：具体动作，通常是调用工具，如 `Search[华为最新手机]`。
- **Observation**：工具返回的结果，由外部程序注入，不是模型生成的。

形式化地看，第 t 步的策略（即 LLM \(\pi\)）基于问题与历史轨迹生成思考与行动，环境中的工具再返回观察：

\[(th_t,a_t)=\pi\big(q,(a_1,o_1),\ldots,(a_{t-1},o_{t-1})\big),\quad o_t=T(a_t)\]

关键在于 `Observation` 提供的是**外部事实**，而不是模型的推测。推理因此有了依据，行动也因此有了方向。它最适合三类任务：需要外部知识的（实时信息、专业资料）、需要精确计算的（交给计算器而不是让模型心算）、需要与 API 交互的（查库、下单）。

## 2. 工具：名称、描述、执行逻辑

工具是智能体的手脚，一个合格的工具定义只有三部分：**名称**（唯一标识，供 `Action` 调用）、**描述**（自然语言说明用途）、**执行逻辑**（真正干活的函数）。其中描述最容易被忽视，却最关键——LLM 就是靠这段描述判断「什么时候该用哪个工具」。

描述要写清使用边界，而不是功能清单：

```python
search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
```

工具实现还有两个工程约定。一是**返回值要"压过"**：搜索结果里优先取答案框与知识图谱，拿不到再退回前三条摘要，因为原始 JSON 塞进上下文既费 Token 又冲淡信号。

```python
def search(query: str) -> str:
    results = SerpApiClient(params).get_dict()
    if "answer_box" in results and "answer" in results["answer_box"]:
        return results["answer_box"]["answer"]                    # 最精确
    if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
        return results["knowledge_graph"]["description"]
    snippets = [f"[{i+1}] {r.get('title','')}\n{r.get('snippet','')}"
                for i, r in enumerate(results.get("organic_results", [])[:3])]
    return "\n\n".join(snippets) if snippets else f"没有找到关于 '{query}' 的信息。"
```

二是**错误也要返回字符串**，不抛异常。工具失败在智能体里是常态，把 `错误: 未配置 SERPAPI_API_KEY` 直接当成观察回灌，模型才有机会改参数或换策略。

## 3. ToolExecutor：让工具可插拔

工具一多就需要统一注册与调度。三个方法足够：

```python
class ToolExecutor:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        self.tools[name] = {"description": description, "func": func}

    def getTool(self, name: str) -> callable:
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        return "\n".join(f"- {n}: {i['description']}" for n, i in self.tools.items())
```

`getAvailableTools()` 的输出会被直接拼进提示词，等于把工具箱「翻译」成模型能读的说明书。工具描述写得好不好，在这一步就能看出差别。

## 4. 提示词：把格式钉死

ReAct 的稳定性几乎全部取决于提示词模板。它只需要四块内容：

```text
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{tool_name}[{tool_input}]`: 调用一个可用工具。
- `Finish[最终答案]`: 当你认为已经获得最终答案时。

现在，请开始解决以下问题:
Question: {question}
History: {history}
```

- **角色定义**：限定身份，避免模型自由发挥。
- **工具清单**：`{tools}` 注入上一步生成的描述。
- **格式规约**：强制结构化，让代码能用正则解析。
- **动态上下文**：`{question}` 与 `{history}` 注入原始问题与累积轨迹。

`History` 而非「完整对话」是个有用的取舍：只回灌 Action 与 Observation，省掉重复的 Thought，上下文增长慢一半，且不影响决策。

## 5. 循环：格式化 → 调用 → 解析 → 执行 → 回灌

`ReActAgent` 的主体就是一个带上限的 while 循环：

```python
class ReActAgent:
    def __init__(self, llm_client, tool_executor, max_steps: int = 5):
        self.llm_client, self.tool_executor = llm_client, tool_executor
        self.max_steps, self.history = max_steps, []

    def run(self, question: str):
        for step in range(self.max_steps):
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=self.tool_executor.getAvailableTools(),
                question=question,
                history="\n".join(self.history),
            )
            text = self.llm_client.think([{"role": "user", "content": prompt}])

            thought, action = self._parse_output(text)      # Thought / Action 分离
            if not action:
                break
            if action.startswith("Finish"):
                return re.match(r"Finish\[(.*)\]", action, re.DOTALL).group(1)

            name, arg = self._parse_action(action)          # Search[xxx] -> (Search, xxx)
            fn = self.tool_executor.getTool(name)
            observation = fn(arg) if fn else f"错误:未找到名为 '{name}' 的工具。"

            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
        return None
```

两个解析函数都是正则，但细节决定鲁棒性：

```python
def _parse_output(self, text):
    thought = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
    action  = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
    return (thought.group(1).strip() if thought else None,
            action.group(1).strip() if action else None)

def _parse_action(self, action_text):
    m = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
    return (m.group(1), m.group(2)) if m else (None, None)
```

`max_steps` 不是可选项而是安全阀：模型可能在两个工具间来回打转，没有上限就会一直烧 Token。

## 6. 一次真实运行

问「华为最新的手机是哪一款？它的主要卖点是什么？」，两轮结束：

- **第 1 步**：Thought 判断知识库不足 → `Search[华为最新手机型号及主要卖点]` → 返回三条官网与推荐文摘要。
- **第 2 步**：Thought 从摘要里识别出 Mate 70 与 Pura 80 Pro+ 两条候选，判断已足够回答 → `Finish[最终答案]`。

这条轨迹把 ReAct 的价值展示得很清楚：模型先承认「不知道」，再借助工具获得事实，最后组织答案。整个过程可读、可审计，出问题也容易定位到具体某一步。

## 7. 优缺点与调试

**三个优点**：高可解释性（Thought 链就是决策日志）、动态规划与纠错（上一步搜偏了，下一步能换词）、工具协同（LLM 负责运筹，工具负责事实与计算）。

**四个局限**：

- **强依赖模型能力**：推理弱或格式遵循差的模型，会在 Thought 环节规划错，或在 Action 环节输出不规范，流程直接断。
- **串行开销**：每步一次 LLM 调用，复杂任务的总耗时与费用随步数线性增长。
- **提示词脆弱**：模板里改动几个字都可能改变行为，换模型更是要重调。
- **局部最优**：走一步看一步，缺少全局计划，容易在眼前信息上打转。

**调试五步**，按成本从低到高：打印每次调用前**完整的格式化提示词**；解析失败时打印**模型原始输出**，先分清是模型不听话还是正则写错；检查 `tool_input` 与 `observation` 的格式是否匹配；加入一两个完整的 Thought-Action-Observation 示例做 few-shot；最后才是换模型或把 `temperature` 调到 0。

ReAct 的短板正好是下一种范式的起点：先把计划整体列出来，再逐步执行——[4-2-PlanAndSolve与Reflection](/source/agent/从零构建智能体/4-2-PlanAndSolve与Reflection)。
