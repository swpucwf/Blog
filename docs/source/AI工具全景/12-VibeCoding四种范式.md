# VibeCoding 四种范式

## 1. 四种开发者原型

按「技术知识 × AI 依赖度」两轴划分，2026 年的开发者分成四类：

| 原型 | 技术知识 | AI 依赖 | 特征 |
|---|---|---|---|
| 传统工程师 | 高 | 低 | 全面理解代码库，谨慎使用 AI，聚焦系统完整性 |
| 增强工程师 | 高 | 高 | 深度依赖 AI 但保持技术判断，评估并优化 AI 方案 |
| 新兴开发者 | 低 | 低 | 传统方式学基础，群体在缩小 |
| 领域创造者 | 低 | 高 | 专注问题描述而非实现细节，快速原型 |

趋势很清楚：**增强工程师会成为主导形态**——AI 承担产出，人承担判断。领域创造者则会分化，一部分留在产品侧，一部分补齐技术成为增强工程师。

## 2. 四种编程范式

| 范式 | 核心思想 | 前置工作 | 适用场景 | 优势 |
|---|---|---|---|---|
| Vibe Coding | 灵感驱动，边做边改 | 很少 | 创意探索、原型验证 | 最快出结果 |
| Spec Coding | 先定义，再开发 | 完整规格 | 需求明确、模块交付 | 输出可控 |
| Glue Coding | 复用整合，拼装交付 | 梳理已有模块 | MVP、应用整合 | 交付最快 |
| Harness Engineering | 质检门禁，返工迭代 | 定义质检标准 | 生产环境、大型项目 | 质量可控 |

四种不是替代关系，而是同一项目不同阶段的取法。

**Vibe**：在 Cursor Agent 里说「做一个极简 Markdown 笔记应用，支持实时预览、标签、localStorage」，五分钟就能跑起来，不满意继续对话加暗色主题。关键是不纠结代码质量，先验证想法。

**Spec**：需求明确时先写规格——输入、校验规则、输出格式、异常码，再让 AI 按规格实现。规格越清晰，输出越可预测，返工越少。细分做法见[规范驱动开发](/source/vibecoding/规范驱动开发/README)。

**Glue**：已有 RAG 服务、MCP Server、前端组件时，让 AI 写「胶水」把它们串起来。AI 最擅长的就是连接代码，你提供组件，它负责拼装。

**Harness**：上线前定义质检门禁——单测覆盖率、lint 通过、输入校验、无硬编码密钥、错误有友好提示，让 AI 循环执行直到全部通过。

## 3. 何时切换

| 信号 | 切换方向 | 原因 |
|---|---|---|
| 生成代码越来越难维护 | Vibe → Spec | 代码量超出上下文理解能力 |
| 同一功能多次对话仍不满意 | Vibe → Spec | 需求明确但缺规格约束 |
| 要集成 3 个以上已有模块 | 任意 → Glue | 拼装比重写高效 |
| 项目要上线或多人协作 | 任意 → Harness | 生产必须有质检门禁 |
| 同一 bug 修复超过两次 | 任意 → Harness | 缺测试保护导致回归 |
| 需求一句话说不清 | Glue/Harness → Spec | 先理清再动手 |

核心节奏：**早期用 Vibe，中期切 Spec，上线前加 Harness**。同一项目里不同模块可以混用——探索性功能用 Vibe，核心模块用 Spec 与 Harness。

## 4. 范式与工具映射

| 范式 | CLI | IDE | Skill / 工作流 |
|---|---|---|---|
| Vibe | Claude Code、OpenClaw | Cursor Agent、Bolt.new | 前端设计、创意生成 |
| Spec | Claude Code（Plan 模式） | Cursor、Devin Desktop | skill-creator、doc-coauthoring |
| Glue | OpenClaw（Skills 驱动） | TraeCode、Replit | mcp-builder、API 对接 |
| Harness | Claude Code（TDD 模式） | Cursor + Copilot、ADE | webapp-testing、CodeReview |

Harness 阶段往往要同时跑多个 agent——一个写、一个测、一个审，这正是 ADE 的用武之地。

## 5. 范式之上的四层工程谱系

四种范式回答「这个项目该用哪种协作方式」。往上还有一条层叠谱系，每层建立在下一层之上：

| 层 | 名称 | 解决什么问题 |
|---|---|---|
| L1 | Prompt Engineering | 单次指令怎么写 |
| L2 | Context Engineering | 喂给模型哪些上下文、工具与记忆 |
| L3 | Harness Engineering | 给 agent 搭工具链、权限与验证门禁 |
| L4 | Loop Engineering | 设计「提示 agent 的系统」，而非自己提示 |

一个容易混的术语：L3 的 Harness Engineering 是**工程实践**，而「Harness」本身是**承载 agent 的运行时**——工具、权限、记忆、验证的总和。Claude Code、Codex、DSH、Pi 都属于 Harness，Harness Engineering 则是设计它的一套方法论。

### 5.1 Loop Engineering

一句话：不再手动给 agent 写 prompt，而是设计一套会自动给 agent 写 prompt 的系统。

一个完整的 loop 由五个组件构成：

| 组件 | 作用 | 例子 |
|---|---|---|
| Trigger | 什么启动循环 | 定时、PR 打开、测试失败、人的一次性指令 |
| Goal | 可验证的终态 | 「所有测试通过」，而不是「优化一下」 |
| Actions | agent 能用的工具 | 读写文件、跑命令、调 MCP、派生子 agent |
| Verification | 怎么知道该停 | 测试退出码、reviewer 复核、CI 通过 |
| Memory | 跨轮次保留什么 | 会话续接、项目说明文件、外部记忆库 |

常见落地形态是挂成 CI 定时任务：每日 triage、PR 陪伴、CI 清扫、依赖升级、changelog 草拟。

风险必须提前想清楚：**无监督的 loop 会无监督地犯错**。token 成本随子 agent 与长跑循环迅速膨胀；验证责任仍在开发者身上；设计时就要写入停止规则——预算上限、失败次数上限、哪些操作必须人工裁决。

放权节奏建议 **L1 只报告 → L2 可修改但待审 → L3 无人值守**，逐级建立信心后再放手。

## 6. 实操建议

- 从 Vibe 入门，先用 Cursor Agent 或 Bolt.new 感受效率；
- 项目变大切 Spec，用规格文档约束输出；
- 上线前必须上 Harness，测试门禁与 CodeReview 不能省；
- 同一项目混合使用，别拘泥于单一范式；
- 想让 agent 自己跑，先设计 loop（trigger、可验证目标、验证方式、停止规则），再逐级放权。

## 写在最后

范式决定协作方式，资源决定信息来源。下一篇[资源导航](13-资源导航)整理值得长期关注的导航站、榜单与社区。
