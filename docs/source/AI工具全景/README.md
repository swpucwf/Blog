# AI 工具全景

> 内容整理自开源教程 [ai-wiki](https://github.com/datawhalechina/ai-wiki)，作为参考笔记。

AI 开发的横向选型地图：16 篇覆盖模型与数据层、框架与协议层、工具与平台层、范式与方法论层，最后从「人与商业」收束。与站内各专题的区别是——这里回答「选哪个」，专题回答「怎么做」。

## 全景结构

```text
┌────────────────── 范式与方法论层 ──────────────────┐
│  12 VibeCoding 四范式        14 Prompt Engineering  │
├────────────────── 工具与平台层 ────────────────────┤
│  4 CLI     7 IDE     5 Skills    1 OpenClaw   2 Coding Plan │
├────────────────── 框架与协议层 ────────────────────┤
│  8 Agent 框架      9 RAG 框架      6 MCP            │
├────────────────── 模型与数据层 ────────────────────┤
│  3 模型 API        11 Embedding    10 向量库        │
└────────────────────────────────────────────────────┘
        13 资源导航 · 15 实战项目 · 16 角色与商业
```

**依赖关系**：模型与数据层是基础；框架与协议层依赖它（Agent 调模型、RAG 依赖嵌入与向量库、MCP 是工具协议）；工具与平台层依赖框架；范式层是方法论；第 16 篇横切所有层，把「人」与 Token 成本贯穿回去。

## 三条学习路径

**AI 编程入门**：3 模型 API → 4 CLI → 7 IDE → 5 Skills → 12 范式 → 2 Coding Plan，目标是能用 AI 工具独立完成一个小项目。

**RAG 应用开发**：3 → 11 Embedding → 10 向量库 → 9 RAG 框架 → 6 MCP，目标是搭出端到端的知识库问答。

**Agent 工程师**：6 MCP → 8 Agent 框架 → 9 RAG → 5 Skills → 14 Prompt，目标是设计可调用工具的多步 Agent。

## 目录

- [1-OpenClaw产品生态](/source/AI工具全景/1-OpenClaw产品生态)
- [2-CodingPlan与成本](/source/AI工具全景/2-CodingPlan与成本)
- [3-模型API选型](/source/AI工具全景/3-模型API选型)
- [4-CLI工具盘点](/source/AI工具全景/4-CLI工具盘点)
- [5-Skills技能生态](/source/AI工具全景/5-Skills技能生态)
- [6-MCP协议与生态](/source/AI工具全景/6-MCP协议与生态)
- [7-IDE编程工具](/source/AI工具全景/7-IDE编程工具)
- [8-Agent框架选型](/source/AI工具全景/8-Agent框架选型)
- [9-RAG框架选型](/source/AI工具全景/9-RAG框架选型)
- [10-向量知识库选型](/source/AI工具全景/10-向量知识库选型)
- [11-Embedding模型选型](/source/AI工具全景/11-Embedding模型选型)
- [12-VibeCoding四种范式](/source/AI工具全景/12-VibeCoding四种范式)
- [13-资源导航](/source/AI工具全景/13-资源导航)
- [14-PromptEngineering](/source/AI工具全景/14-PromptEngineering)
- [15-端到端实战项目](/source/AI工具全景/15-端到端实战项目)
- [16-角色与商业](/source/AI工具全景/16-角色与商业)

## 时效性说明

模型版本、订阅价格、GitHub 星标、榜单排名都属于快速变化的数据，文中数值只反映整理时的状态。做决策前请以官方页面与最新榜单为准，建议每季度复核一次。

## 与站内其他专题的关系

- 想深入 Agent 原理与上下文工程：[Agent 深入理解](/source/agent/深入理解/README)；
- 想系统学 RAG 全链路：[RAG 技术全栈](/source/agent/RAG技术全栈/README)；
- 想练 AI 编程方法论：[VibeCoding 专栏](/source/vibecoding/README)；
- 想深入底层性能：[GPU 算子优化](/source/GPU算子优化/README)。
