# Plan-and-Solve 与 Reflection

ReAct 的两个短板在这里被分别针对：走一步看一步、缺全局计划，交给 Plan-and-Solve；一稿定稿、缺自我纠错，交给 Reflection。前者把「思考与行动交替」改成「先谋后动」，后者在任一范式外面再套一层「事后复盘」。

## 1. 先谋后动：两个阶段

Plan-and-Solve Prompting 由 Lei Wang 在 2023 年提出，动机是思维链在多步复杂问题上容易「偏离轨道」。与 ReAct 把思考和行动揉进每一步不同，它把流程切成两段：

- **规划阶段**：接收完整问题后的第一件事不是作答，也不是调工具，而是把问题分解成清晰、分步的行动计划。这个计划本身就是一次 LLM 调用的产物。
- **执行阶段**：严格按计划的步骤逐一执行，每一步是一次独立调用，或对上一步结果的加工，直到计划走完、答案落地。

<strong>形式化</strong>：规划模型根据原始问题产出含 \(n\) 步的计划，执行模型在第 \(i\) 步同时依赖原始问题、完整计划与全部前序结果，最终答案取最后一步：

\[P=\pi_{\text{plan}}(q),\qquad s_i=\pi_{\text{solve}}\big(q,P,(s_1,\dots,s_{i-1})\big)\]

<img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/helloagents-plan-and-solve.png" width="90%" alt="Plan-and-Solve 的两阶段工作流" />

一个便于记的比喻：ReAct 像侦探，跟着现场痕迹（Observation）随时改调查方向；Plan-and-Solve 像建筑师，动工前先出蓝图，再照着图纸施工。因此它更适合结构性强的任务——多步数学应用题、需要整合多个信息源的报告、先定结构再实现的代码生成。

## 2. 规划阶段：可解析比正确更重要

规划提示词要同时解决三件事：角色（收窄语气）、任务（讲清是分解而非作答）、格式（让代码能稳定解析）。

````python
PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""
````

强制输出 Python 列表，是为了让解析比解析自然语言可靠得多。解析用 `ast.literal_eval` 而非 `eval`：前者只认字面量，不会执行任意代码。

```python
class Planner:
    def plan(self, question: str) -> list[str]:
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        text = self.llm_client.think(messages=[{"role": "user", "content": prompt}]) or ""
        try:
            plan_str = text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError):
            return []          # 解析失败就返回空，不带半截计划往下走
```

失败即返回空列表，由主流程据此终止——「宁可停下，也不要带着错误计划乱走」是这个范式的基本纪律。

## 3. 执行阶段：状态管理才是主体

执行器的提示词和规划器不同：它不分解问题，只解决当前这一步，因此四块上下文缺一不可。

```python
EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""
```

四块信息各自的作用：原始问题守住最终目标，完整计划告诉模型自己正处在链条的哪一环，历史结果是当前步骤的直接输入，当前步骤则把注意力钉死在一步上。代码层面，执行器真正承担的是**状态管理**——按顺序累积 history，并把它回灌给下一步：

```python
class Executor:
    def execute(self, question: str, plan: list[str]) -> str:
        history = ""
        for i, step in enumerate(plan):
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question, plan=plan,
                history=history or "无", current_step=step,
            )
            text = self.llm_client.think(messages=[{"role": "user", "content": prompt}]) or ""
            history += f"步骤 {i+1}: {step}\n结果: {text}\n\n"
        return text            # 最后一步的结果就是最终答案
```

「仅输出该步骤的最终答案」这条约束不是客套：模型多写的解释性文字会作为 history 进入下一步，既费 Token，又可能被下一步误当成数据。计划的步骤描述只是意图，history 才是事实，两者不一致时以 history 为准。

## 4. 组合与一次完整运行

规划器与执行器各自独立，智能体只做协调，本身不含业务逻辑：

```python
class PlanAndSolveAgent:
    def __init__(self, llm_client):
        self.planner = Planner(llm_client)
        self.executor = Executor(llm_client)

    def run(self, question: str):
        plan = self.planner.plan(question)
        if not plan:
            print("无法生成有效的行动计划。")
            return None
        return self.executor.execute(question, plan)
```

拿水果店问题跑一遍：周一卖出 15 个苹果，周二是周一的两倍，周三比周二少 5 个，问三天总和。规划器给出四步（周一 15 → 周二 30 → 周三 25 → 合计 70），执行器逐步算出 `15 / 30 / 25 / 70`。

从日志能看出两件事：计划被强制成 Python 列表，解析几乎没有歧义；步骤 2 用到了 15、步骤 3 用到了 30，全靠 history 传递，而不是指望模型「自己记得」。这正是把隐式推理拆成显式状态的好处——每一步的输入都是确定的，出错时能定位到具体某一步。

## 5. Reflection：执行 → 反思 → 优化

前两种范式的共同点是一稿定稿：交出结果，流程就结束。Reflection 引入一个事后（post-hoc）的自我校正循环，由三个角色轮转：

1. **执行**：先用熟悉的方法（ReAct 或 Plan-and-Solve）产出初稿。
2. **反思**：另一个 LLM 实例扮演评审员，按维度审视初稿——事实性错误、逻辑漏洞、效率问题、遗漏信息——产出结构化反馈。
3. **优化**：把初稿与反馈一起作为新上下文，据此修订出「修订稿」。

<img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/helloagents-reflection.png" width="70%" alt="Reflection 的执行—反思—优化迭代循环" />

形式化地，反思模型针对第 \(i\) 版输出给出反馈，优化模型再结合任务、上一版输出与反馈产出新版，循环到反思不再发现新问题、或达到迭代上限：

\[F_i=\pi_{\text{reflect}}(\text{Task},O_i),\qquad O_{i+1}=\pi_{\text{refine}}(\text{Task},O_i,F_i)\]

与 ReAct 的关键差别在**纠错信号的来源**：ReAct 依赖外部工具返回的 Observation，能纠正「参数填错、查错了东西」这类表层问题；Reflection 依赖内部评审，因此能修正更高层次的策略与逻辑错误。顺带，整条「执行—反思—优化」轨迹天然是一份短期记忆——不仅留下最终答案，还留下「从有缺陷的初稿怎么走到终版」的过程。

## 6. 记忆模块：迭代的前提

迭代的前提是记住之前试过什么、被指出过什么。当上下文很长时，把全部信息塞给评审员既冗余又稀释重点，所以需要一个短小的记忆模块，只存两类记录：

```python
class Memory:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        # record_type 取 'execution'（代码）或 'reflection'（反馈）
        self.records.append({"type": record_type, "content": content})

    def get_trajectory(self) -> str:
        parts = []
        for r in self.records:
            if r["type"] == "execution":
                parts.append(f"--- 上一轮尝试 (代码) ---\n{r['content']}")
            elif r["type"] == "reflection":
                parts.append(f"--- 评审员反馈 ---\n{r['content']}")
        return "\n\n".join(parts)

    def get_last_execution(self):
        for r in reversed(self.records):
            if r["type"] == "execution":
                return r["content"]
        return None
```

三个方法各司其职：`add_record` 按序落盘，`get_trajectory` 把轨迹序列化成可直接插入提示词的文本，`get_last_execution` 倒序取最新初稿。用列表而非对话历史，好处是记录可被重新裁剪——真到了上下文吃紧的时候，可以只取最近两轮。

## 7. 三套提示词与终止条件

Reflection 需要三个角色协同，不是一份提示词改三遍。

初始执行只要结果，约束在格式与规范上：

```text
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
要求: {task}
请直接输出代码，不要包含任何额外的解释。
```

反思提示词是机制的灵魂，关键在把评审员的注意力锁死在一个维度上：

```text
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下Python代码，并专注于找出其在算法效率上的主要瓶颈。

# 原始任务: {task}
# 待审查的代码: {code}

请分析该代码的时间复杂度，并思考是否存在一种算法上更优的解决方案来显著提升性能。
如果存在，请清晰地指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法替代试除法）。
如果代码在算法层面已经达到最优，才能回答"无需改进"。
```

「专注于算法效率」这个限定很重要：只笼统要求「挑毛病」，模型会给出命名、注释、异常处理之类的边角建议，迭代容易变成来回打转；锁定维度之后，每轮反馈才指向真正能提升的改动。

优化提示词把三样东西拼在一起——原始任务、上一版代码、评审反馈，并要求它按反馈重写：

```text
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。
# 原始任务: {task}
# 你上一轮尝试的代码: {last_code_attempt}
评审员的反馈：{feedback}
请根据评审员的反馈，生成一个优化后的新版本代码。
```

主循环短得几乎不像一个「智能体」：

```python
class ReflectionAgent:
    def __init__(self, llm_client, max_iterations: int = 3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, task: str):
        code = self._ask(INITIAL_PROMPT_TEMPLATE.format(task=task))
        self.memory.add_record("execution", code)

        for _ in range(self.max_iterations):
            last = self.memory.get_last_execution()
            feedback = self._ask(REFLECT_PROMPT_TEMPLATE.format(task=task, code=last))
            self.memory.add_record("reflection", feedback)

            if "无需改进" in feedback:          # 收敛信号
                break

            code = self._ask(REFINE_PROMPT_TEMPLATE.format(
                task=task, last_code_attempt=last, feedback=feedback))
            self.memory.add_record("execution", code)

        return self.memory.get_last_execution()
```

终止条件用字符串匹配「无需改进」，是刻意留给提示词的接口：提示词里写清「只有算法层面已最优才能这样回答」，代码只认这四个字。它确实脆弱（模型换个说法就漏判），但在受控提示词下够用；要更稳，就让模型输出结构化判定字段，而不是自然语言。

## 8. 素数案例：两次跃迁

任务：写一个 Python 函数，找出 1 到 n 之间所有素数。选它做案例的理由很实在——初版大概率是低效的试除法，反思点清晰（复杂度），优化方向也明确（换算法）。

- **初稿**：逐个试除，时间复杂度 \(O(n\sqrt{n})\)。
- **第 1 轮反思**：指出每个数都要试除是主要瓶颈，建议埃拉托斯特尼筛法，\(O(n\log\log n)\)。
- **优化后**：按筛法重写，从 \(p^2\) 起按步长 \(p\) 划掉合数。
- **第 2 轮反思**：确认已是筛法，顺带提到分段筛、奇数筛等更进一步方案，但判断「在一般情况下无需改进」——触发终止。

```python
def find_primes(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    p = 2
    while p * p <= n:
        if is_prime[p]:
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
        p += 1

    return [num for num in range(2, n + 1) if is_prime[num]]
```

这次运行说明 Reflection 的价值不止修 bug，而是把「功能正确」推到「性能合理」。但它有个隐含前提：评审员的判断力必须高于初稿。若评审模型本身说不清复杂度，迭代就可能朝错误方向走——所以实践中常用更强的模型担任评审、或用另一种视角的提示词（例如「挑边界情况」与「挑复杂度」分开评）。

## 9. 成本收益与范式边界

三项成本都很直接：

- **调用开销**：每轮至少两次额外调用（反思 + 优化），多轮下来成本与算力成倍增加。
- **延迟**：反思与优化是串行的，必须等上一轮反馈完成，总耗时显著拉长。
- **提示工程复杂度**：执行、反思、优化三套提示词要分别设计调试，收益高度依赖提示词质量。

两项收益：**质量跃迁**（从合格到优秀，从粗糙到严谨）与**鲁棒性增强**（内部纠错能发现逻辑漏洞、事实错误、边界遗漏）。

因此它是一种典型的「以成本换质量」策略，适合对准确性与可靠性要求极高、而实时性宽松的场景：关键业务代码、技术报告、科研中的复杂推演、需要深度分析的决策支持。反之，若要求快速响应、或「大致正确」就够，轻量的 ReAct 或 Plan-and-Solve 更划算。

三种范式的差别可以压成一张表：

| 范式 | 决策方式 | 纠错来源 | 适合 |
| --- | --- | --- | --- |
| ReAct | 走一步看一步，边想边做 | 外部工具返回的 Observation | 需要外部事实、实时信息、API 交互 |
| Plan-and-Solve | 先出完整计划，再逐步执行 | 计划自身的逻辑顺序 | 结构性强、可清晰分解的多步任务 |
| Reflection | 执行后自评再修订，可叠加在前两者之上 | 内部评审员 | 质量优先、可接受延迟与多轮成本 |

三者不是替代关系。ReAct 的每一步可以交给 Plan-and-Solve 拆解，Reflection 也可以裹在任何一个范式外面——实际系统里更常见的是组合，而不是单选。

下一步把这些范式做成框架里的可复用组件：抽象基类、注册表与工具系统——[7-1-框架架构与接口](/source/agent/从零构建智能体/7-1-框架架构与接口)。
