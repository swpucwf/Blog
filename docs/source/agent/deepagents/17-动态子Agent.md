# 动态子 Agent：用代码编排多个 Agent

## 1. 从单次委派到动态编排

仓库里有 24 个路由文件。主 Agent 可以把第一个交给审查 Agent，等结果回来再决定是否看第二个；也可以先列出全部文件，用 JavaScript 一次组织 24 次审查，再把可疑发现交给另一个 Agent 复核。两种方式都用了子 Agent，差别在于**谁控制拆分、并发和下一阶段**。

| 任务形状 | 优先方式 | 原因 |
|---|---|---|
| 把一个明确问题交给一个专家 | 普通 `task` Tool | 路径短，容易观察与逐次审批 |
| 对一批独立对象执行同类分析 | Dynamic Subagents | JS 可完整覆盖输入并并行扇出 |
| 输入混杂，需先分类再分发 | Dynamic Subagents | 分类结果直接决定下一批 `subagentType` |
| 同一发现需要独立复核 | Dynamic Subagents | 第一阶段输出可在代码中进入第二阶段 |
| 范围未知，需反复搜索直到无新结果 | Dynamic Subagents | 循环与停止条件由代码明确表达 |
| 子任务要后台持续运行、稍后查询 | Async Subagents | 生命周期目标不同，不应混为并行扇出 |

**Dynamic Subagents 不是新的子 Agent 类型**，它改变的是编排位置：普通委派由主模型逐次调用 `task` Tool；动态编排由一次 `eval` 中的 JavaScript 多次调用 `task()`。

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_52-comparison-direct-vs-dynamic-subagents.png)

<p>图 1：普通 task 与动态编排</p>

### 1.1 为什么代码编排更稳定

审查 24 个文件时，模型逐次委派意味着每次返回都可能改变后续决策；上下文变长后还可能提前总结，留下未处理的文件。把列表放进 JavaScript 后，可以明确表达：

1. 每个输入都进入一次审查；
2. 独立任务按批次并行；
3. 第一阶段发现统一去重筛选；
4. 只有高风险发现进入复核；
5. 最终只把已确认结果交还主模型。

代码通过检查后，覆盖规则是确定的，但**不保证每个子任务成功，也不保证判断正确**——每次 `task()` 都启动完整的 Agent 推理循环，结果仍受模型、Prompt、工具与输入影响。

## 2. 运行链路与 `task()` 契约

同时配置子 Agent 与 Interpreter 中间件后，JavaScript 环境出现全局函数 `task()`：

1. 主 LLM 读用户请求，决定是否调用 `eval`，并为当前任务编写 JavaScript；
2. JS 在 QuickJS 中运行，遇到 `task()` 时启动指定子 Agent；
3. 每个子 Agent 运行自己的完整推理循环，可用分配给它的工具；
4. JS 收集、筛选、合并结果，只把需要的信息交回主 LLM。

所以 `task()` 不是开发者手写的固定工作流函数，而是 Interpreter 注入到 JS 环境的全局函数；调用它的代码通常由主 LLM 现场生成。这对模型要求更高：既要稳定调用工具，也要写出可运行的异步 JS（`await`、数组方法、循环、分支、Schema），还要选对真实存在的 `subagentType`。模型能力不足时容易语法错误、角色名漂移、遗漏输入或循环失控。

提示词里的 `workflow` 只是提醒主 LLM 优先考虑代码编排，**不是 API 参数**；真正提供能力的是 `subagents` 配置与 `CodeInterpreterMiddleware`。

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_53-framework-quickjs-task-orchestration.png)

<p>图 2：QuickJS 中的 task() 编排</p>

### 2.1 `task()` 只有三个字段

| 字段 | 必填 | 作用 |
|---|---|---|
| `description` | 是 | 发给子 Agent 的任务说明，应含目标、范围、约束、输出要求 |
| `subagentType` | 是 | 选择哪个已配置子 Agent，值必须匹配 `subagents` 的 `name` |
| `responseSchema` | 否 | 用 JSON Schema 约束返回值；设置后得到的就是 JS 对象 |

```typescript
const review = await task({
  description: "读取 src/auth/login.ts，检查认证绕过；引用行号。",
  subagentType: "reviewer",
  responseSchema: {
    type: "object",
    properties: {
      issues: {
        type: "array",
        items: {
          type: "object",
          properties: {
            line: { type: "number" },
            severity: { type: "string" },
            evidence: { type: "string" },
          },
          required: ["line", "severity", "evidence"],
        },
      },
    },
    required: ["issues"],
  },
});

const highRisk = review.issues.filter((item) => item.severity === "high");
```

设置 `responseSchema` 后 `review` 已是对象，不需要 `JSON.parse`。一次 `task()` 返回本次运行结果，不是可继续对话的会话；下一次调度仍要提供自包含的 `description`。

实际生成的代码可能用 `for...of`、`Promise.all`、辅助函数或不同变量名。判断是否正确，看**输入是否完整覆盖、角色是否选对、停止条件是否生效、结果是否满足 Schema**，不要要求逐字一致。

## 3. 准备角色，再触发 workflow

开发者预先定义主 LLM 可以选哪些子 Agent、每个角色的边界；用户在请求中写 `workflow`，说明按什么顺序处理哪些输入；主 LLM 据此现场生成 JS。

### 3.1 配置角色

`name` 是 `subagentType` 的可用值，`description` 帮主 LLM 选角色，`system_prompt` 约束角色怎么完成任务：

```python
subagents = [
    {"name": "classifier", "description": "将工单归为 bug、feature、question 或 unknown",
     "system_prompt": "只负责分类；信息不足时返回 unknown，不处理工单。"},
    {"name": "bug-fixer", "description": "调查缺陷工单并给出复现步骤",
     "system_prompt": "核对现象与上下文，返回复现步骤和影响。"},
    {"name": "feature-analyst", "description": "评估功能请求的可行性和成本",
     "system_prompt": "说明用户价值、实现条件和主要取舍。"},
    {"name": "support-agent", "description": "根据已有材料回答使用问题",
     "system_prompt": "只根据可用材料回答；缺少依据时明确说明。"},
    {"name": "reviewer", "description": "审查代码并给出文件、行号和证据",
     "system_prompt": "只报告有代码证据的候选问题；稳定 ID 使用'文件:行号:问题类型'。"},
    {"name": "verifier", "description": "独立复核候选问题并优先识别误报",
     "system_prompt": "重新读取代码，寻找反证后再确认或反驳。"},
    {"name": "analyzer", "description": "在给定范围内分轮查找死代码",
     "system_prompt": "用文件和符号组成稳定 ID，不要重复已发现项。"},
]

agent = create_deep_agent(
    model=model,
    subagents=subagents,
    middleware=[CodeInterpreterMiddleware(mode="turn")],
)
```

### 3.2 流程提示骨架

一个可检查的请求通常包含四部分：

1. 用 `workflow` 明确提示采用工作流式编排；
2. 给出明确的输入范围与可用角色；
3. 说明先做什么、后做什么、哪些可并行；
4. 写清去重、数量上限、停止条件与最终返回内容。

```text
运行一个 workflow，处理以下范围：[明确的输入范围]。
先让 [角色 A] 完成 [第一步]，再让 [角色 B] 根据第一步结果完成 [第二步]。
每个输入都要 [覆盖、去重或限制数量]；遇到 [异常或停止条件] 时 [处理方式]。
最后返回 [结果字段和未处理信息]。
```

骨架描述业务流程，不规定底层函数与写法；主 LLM 可能选择并发、循环、分支或结构化返回，能否生成正确代码仍取决于模型能力。

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_54-framework-three-orchestration-patterns.png)

<p>图 3：角色准备与三种编排模式</p>

### 3.3 示例一：分类后分流

提示词：

```text
运行一个 workflow，处理下面三条工单：
- T-101：点击登录后返回 500。
- T-102：希望订单页可以导出 CSV。
- T-103：如何重置密码？

先让 classifier 把每条归为 bug、feature、question 或 unknown，再分别交给
bug-fixer、feature-analyst 或 support-agent。
每条工单只能进入一个处理分支，不得遗漏或重复。无法确定时保留 unknown，不要猜。
最后按工单 ID 返回类别、处理结果和未处理原因。
```

可能的 JS：

```typescript
const tickets = [
  { id: "T-101", text: "点击登录后返回 500" },
  { id: "T-102", text: "订单页导出 CSV" },
  { id: "T-103", text: "如何重置密码" },
];
const categorySchema = {
  type: "object",
  properties: { category: { type: "string", enum: ["bug", "feature", "question", "unknown"] } },
  required: ["category"],
};
const handler = { bug: "bug-fixer", feature: "feature-analyst", question: "support-agent" };

const results = await Promise.all(tickets.map(async (ticket) => {
  const { category } = await task({
    description: `只对工单分类：${ticket.id}\n${ticket.text}`,
    subagentType: "classifier",
    responseSchema: categorySchema,
  });
  if (!handler[category]) {
    return { id: ticket.id, category, unhandledReason: "分类为 unknown" };
  }
  const result = await task({
    description: `处理工单 ${ticket.id}：${ticket.text}`,
    subagentType: handler[category],
  });
  return { id: ticket.id, category, result };
}));
results;
```

检查三点：类别受 Schema 限制、unknown 没被强行路由、三个工单 ID 都出现在结果里。

### 3.4 示例二：并行审查后复核

```text
运行一个 workflow，审查下面三个文件：
- src/routes/login.ts
- src/routes/session.ts
- src/routes/reset-password.ts

让 reviewer 分别检查每个文件的认证问题，候选结果必须包含稳定 ID、文件、行号和证据。
按稳定 ID 去重，再让 verifier 独立确认或反驳，最多复核 20 条。
最后只返回已确认问题，并说明被反驳和未复核的数量。
```

```typescript
const files = ["src/routes/login.ts", "src/routes/session.ts", "src/routes/reset-password.ts"];

const reviews = await Promise.all(files.map((file) => task({
  description: `审查 ${file} 的认证问题；每条候选返回稳定 ID、文件、行号和证据，稳定 ID 使用"文件:行号:问题类型"。`,
  subagentType: "reviewer",
  responseSchema: findingsSchema,
})));

const findings = reviews.flatMap((review) => review.findings);
const unique = [...new Map(findings.map((item) => [item.id, item])).values()];
const candidates = unique.slice(0, 20);

const verdicts = await Promise.all(candidates.map((finding) => task({
  description: `复核 ${finding.file}:${finding.line}，确认或反驳：${finding.evidence}`,
  subagentType: "verifier",
  responseSchema: verdictSchema,
})));

const confirmed = candidates.filter((_, index) => verdicts[index].confirmed);
({ confirmed, rejected: candidates.length - confirmed.length,
   unreviewed: unique.length - candidates.length });
```

验收：三个文件是否全部进入 reviewer、每条候选是否都有独立结论、超出 20 条的部分是否计入未复核。

### 3.5 示例三：分轮搜索直到停止

```text
运行一个 workflow，在 src/legacy/ 范围内分轮查找死代码。
每轮都让 analyzer 继续查找，并把已发现项的稳定 ID 传给它，避免重复。
如果某一轮没有新增项就停止；无论如何最多运行 5 轮，每轮最多保留 20 项。
最后返回已发现并保留的去重结果、实际轮数和停止原因。
```

```typescript
const seen = new Set();
const found = [];
let rounds = 0;
let stopReason = "达到 5 轮上限";

for (let round = 0; round < 5; round += 1) {
  rounds = round + 1;
  const { items } = await task({
    description: `在 src/legacy/ 中继续查找死代码，每轮最多返回 20 项。已发现：${[...seen].join(", ") || "无"}`,
    subagentType: "analyzer",
    responseSchema: itemsSchema,
  });
  const fresh = items.filter((item) => !seen.has(item.id)).slice(0, 20);
  if (fresh.length === 0) {
    stopReason = "本轮没有新增项";
    break;
  }
  for (const item of fresh) { seen.add(item.id); found.push(item); }
}
({ rounds, stopReason, found });
```

「没有新结果」是业务停止条件，「最多 5 轮」是资源上限，两者都应能从生成的代码里直接看到。

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_55-flow-dynamic-subagents-guardrails.png)

<p>图 4：检查生成代码的四个维度</p>

三个示例的 JS 都只是可能结果：变量名、循环和并发写法可以不同，但**输入覆盖、角色名、结果结构、停止条件**必须与提示词一致。要在运行前看到代码，可以先要求只输出草案，或对 `eval` 本身设置审批。

## 4. 回到 Interpreter：同一组参数继续生效

Dynamic Subagents 没有另起执行环境，主 LLM 生成的 JS 仍由 `CodeInterpreterMiddleware` 提供的 `eval` 执行，因此内存、超时、输出、状态、PTC 与子 Agent 开关都继续约束编排：

```python
middleware = CodeInterpreterMiddleware(
    mode="turn",
    subagents=True,
    memory_limit=64 * 1024 * 1024,
    timeout=30.0,
    capture_console=True,
    max_result_chars=8000,
    ptc=["glob"],
    max_ptc_calls=64,
    max_snapshot_bytes=64 * 1024 * 1024,
)
```

| 参数 | 对动态编排的影响 |
|---|---|
| `memory_limit` | 文件列表、候选结果与汇总变量都占 QuickJS 堆 |
| `timeout` | 调度、等待与合并都发生在一次 eval 内，默认 5 秒通常不够 |
| `max_result_chars` | 汇总过长会直接截断，影响主模型后续判断 |
| `ptc` / `max_ptc_calls` | 只约束 `tools.*` 调用，**不限制 `task()` 数量** |
| `subagents` | 设为 False 后只能走普通 `task` Tool |
| `mode` | 只决定 JS 状态保留范围，不改变子 Agent 会话语义 |

最容易混淆的是 `max_ptc_calls`：它限制 `tools.glob(...)` 这类 PTC 调用，不限制 `task()`。当前配置表没有 `max_task_calls`，动态调度数量要靠输入批次、循环上限、Prompt 约束与业务预算控制。

`responseSchema` 也不是中间件参数，而是每次 `task()` 的调用契约——Schema 只保证返回形状可组合，不保证结论正确；汇总时仍要保留来源文件、行号、证据、复核状态与稳定 ID，方便去重与回查。

## 5. 本篇验收

- [ ] 能判断普通 `task` Tool、Dynamic Subagents、Async Subagents 的适用边界；
- [ ] 知道 `task()` 是 Interpreter 注入的全局函数，代码由主 LLM 生成；
- [ ] 记得三个字段与 `responseSchema` 的作用；
- [ ] 能用 name/description/system_prompt 准备角色；
- [ ] 会写含输入范围、顺序、去重、上限、停止条件的 workflow 提示；
- [ ] 能检查生成代码的输入覆盖、角色名、结果结构与停止条件；
- [ ] 知道 `max_ptc_calls` 不限制 `task()`，调度上限要靠提示与业务预算；
- [ ] 记住代码编排保证覆盖规则，不保证子任务成功或判断正确。

> **补记**：动态子 Agent 的收益来自「**把覆盖与顺序从模型决策变成代码约束**」，代价是对主模型提出更高要求——它必须同时会写异步 JS 和选对角色。工程上最值得检查的不是代码写法，而是三件事：每个输入是否都有归属、角色名是否真实存在、停止条件是否真的能触发。`responseSchema` 只能约束形状，结论是否正确仍要靠复核阶段或评分量规。

## 写在最后

下一篇[决策模型 Harness](18-决策模型Harness)把「用哪个模型、这个动作能不能执行」变成程序可消费的判断：Model Router 与 Auto Mode。
