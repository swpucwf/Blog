# IDE 编程工具

## 1. 先分清三种形态

AI 编程工具早已不等于「编辑器插件」。入口从编辑器换成了 agent，于是分化出三种形态，按交出的控制权从少到多排列：

| 形态 | 中心 | 用法 | 代表 |
|---|---|---|---|
| AI IDE | 编辑器 | 逐步操纵，agent 在会话里加速 | Cursor、TraeCode、Antigravity |
| Agent CLI | 终端 | 委派任务，它自己读码、改码、跑测试并自验 | Claude Code、Codex CLI、OpenCode |
| ADE | agent | 甩出目标，调度多个 agent 并行推进 | ZCode、JetBrains Air、Warp |

**ADE（Agentic Development Environment）** 是 2026 年的新品类词，与 AI IDE 最本质的区别是**没有传统编辑器**：文件树、终端、Git 面板、浏览器预览都降级成 agent 调用的工具，主界面变成一个任务和它的执行过程。

注意这个词有两个含义：Letta 最早用它指「开发和调试 agent 本身的平台」，现在开发工具圈普遍指「用 agent 造软件的工作台」。

## 2. AI IDE 与编辑器

| 工具 | 特点 | 最新情况 |
|---|---|---|
| Cursor | 基于 VSCode，Tab 补全、Composer 多文件编辑、Agent 模式 | 归属 Anysphere，持续强化 Agent 与 MCP 集成 |
| Devin Desktop（原 Windsurf） | 流畅的 AI 协作体验，本地 Agent 用 Rust 重写 | 2026-06 由 Windsurf 更名，原域名已 301 跳转 |
| Antigravity | Google 基于 VS Code 的 Agent-first IDE | 个人档免费，可选 Gemini 3 系列、Claude Sonnet/Opus 4.6、gpt-oss |
| TraeCode（原 TRAE IDE） | IDE 模式 + SOLO 模式（AI 主导规划与执行） | 基础版免费，中文理解好；办公能力已拆为独立产品 TraeWork |
| GitHub Copilot | 插件形态的内联补全与 Chat | 适合已有 VS Code / JetBrains 工作流 |
| Visual Studio 2026 | .NET / C++ 主力 IDE | 内置 Copilot Free，含 Profiler Agent 与 Debugger Agent |
| CodeBuddy | 混元 + DeepSeek 双引擎 | Plan / Coding / Deploy 三 Agent 协同 |

## 3. ADE

共同特征：没有传统编辑器、以任务为单位、原生支持多 agent 并行。

| 工具 | 特点 |
|---|---|
| ZCode（智谱） | Goal 驱动全流程自主，支持远程触发与 BYOK；曾因代码库数据上传致歉并承诺开源，企业采用建议等开源后再评估 |
| JetBrains Air | 多 agent 编排 + 团队协作 + 治理与成本控制；坚持多供应商，可接 Claude Agent、Codex、Copilot、OpenCode |
| Warp | 从终端起家的 ADE，配云端编排平台 Oz；已开源核心产品，支持多种 CLI agent 并排运行 |
| Emdash | 开源桌面端多 agent 编排，用 git worktree 给每个 agent 隔离工作目录 |

一个关键技术前提：**多 agent 并行必须用 git worktree 隔离**。多个 agent 同时改同一份代码会互相覆盖；给每个 agent 一个独立目录、共享同一份 `.git`，跑完再用 diff 视图挑最优结果合并。这也是 ADE 这个品类成立的技术基础。

同一思路用到办公场景，就是 AI 原生工作台：TraeWork、WorkBuddy、Kimi Work，产物从代码换成了文档、表格与自动化流程。

## 4. Web 编程工具

| 场景 | 推荐 | 理由 |
|---|---|---|
| 快速出原型 | Bolt.new | 零配置，浏览器直接出全栈应用 |
| UI / 组件生成 | V0 | 文本描述生成 React 组件，设计感强 |
| 在线学习与轻量开发 | Replit | 编码 + 运行 + 部署一体化 |
| 非技术人员做产品 | Bolt.new + Vercel | 自然语言出应用，一键上线 |

Web 工具适合验证想法；项目变大后建议导出代码，转到 Cursor 或 Claude Code 继续迭代。

## 5. 选型对照

| 工具 | 形态 | 免费 | Agent 能力 | 中文体验 | 适用场景 |
|---|---|---|---|---|---|
| Cursor | AI IDE | 付费 | Composer + Agent | 良 | 专业开发、大型项目 |
| Devin Desktop | AI IDE | 付费 | Devin Local | 良 | 流畅协作 |
| Antigravity | AI IDE | 个人档免费 | Agent-first，2.0 并行多 agent | 良 | Google 生态、零成本起步 |
| TraeCode | AI IDE | 基础版免费 | IDE + SOLO | 优 | 中文优先、零成本入门 |
| Copilot | 插件 | 付费 | Agent 模式 | 良 | 已有工作流不换 IDE |
| ZCode | ADE | 免费下载 | Goal 驱动 | 优 | 放弃编辑器、整流程托管 |
| JetBrains Air | ADE | 自带订阅 | 多 agent 编排 + 治理 | 中 | JetBrains 用户、团队协作 |
| Warp | ADE | 免费下载 | 云端编排、多 harness 并行 | 良 | 终端党、多 agent 混用 |

## 6. 怎么选

- 主力开发：Cursor 或 Devin Desktop，成熟、Agent 能力强；
- 零成本入门：TraeCode、Antigravity、CodeBuddy，都有免费档；
- 不想换 IDE：Copilot 或 Visual Studio 2026；
- 快速原型：Bolt.new、V0；全栈部署用 Replit；
- 中文优先 + 国产模型：TraeCode、CodeBuddy；
- 只想给目标：ZCode、Warp；
- 团队要多 agent 并行 + 治理：JetBrains Air、Warp Oz；
- 不写代码只要文档与自动化：TraeWork、WorkBuddy、Kimi Work。

一条判断路径：已有工作流就加插件；愿意换 IDE 就选 Cursor；预算敏感先用免费档；只做验证用 Web 工具；目标不是写代码就换工作台。项目变大后，Web 工具导代码到 IDE，单 agent 不够就上 ADE 做并行。

## 7. 一个趋势判断

三种形态会长期并存，而不是互相取代：

- **AI IDE** 适合需要精细控制的重构与调试，人仍然在环；
- **Agent CLI** 适合可委派的中等任务，验证靠测试而非肉眼；
- **ADE** 适合可拆分的大目标，前提是任务能并行、验收标准清晰。

选错的典型表现是「用 ADE 做需要逐步确认的敏感改动」，或「用 IDE 手工推进本该委派的批量任务」。

## 写在最后

工具解决「在哪写」，框架解决「agent 怎么搭」。下一篇[Agent 框架选型](8-Agent框架选型)从五种设计模式讲到主流框架与决策树。
