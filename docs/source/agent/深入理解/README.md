# Agent 深入理解

> 内容整理自开源书稿 [bojieli/ai-agent-book](https://github.com/bojieli/ai-agent-book)，作为参考笔记。

围绕 Agent 的三要素与 Harness 工程展开：从现代 Agent 的组成、ReAct 循环与上下文工程，到用户记忆、工具、Coding Agent、交互扩展、评估、后训练、持续进化与多 Agent 协作。

## 目录

第一章 · AI Agent 入门

- [1-1 现代 Agent：LLM + 上下文 + 工具](/source/agent/深入理解/1-1-现代Agent三要素)
- [1-2 ReAct 循环与轨迹](/source/agent/深入理解/1-2-ReAct循环与轨迹)
- [1-3 Harness 工程](/source/agent/深入理解/1-3-Harness工程)
- [1-4 编排模式与护栏](/source/agent/深入理解/1-4-编排模式与护栏)
- [1-5 Agent 工程决策十问](/source/agent/深入理解/1-5-Agent工程决策十问)

第二章 · 上下文工程

- [2-1 上下文与消息结构](/source/agent/深入理解/2-1-上下文与消息结构)
- [2-2 Agent 核心循环与上下文](/source/agent/深入理解/2-2-Agent核心循环与上下文)
- [2-3 KV Cache 友好的上下文设计](/source/agent/深入理解/2-3-KVCache友好的上下文设计)
- [2-4 提示工程：优化系统提示词](/source/agent/深入理解/2-4-提示工程-优化系统提示词)
- [2-5 动态提示词与 Skills](/source/agent/深入理解/2-5-动态提示词与Skills)
- [2-6 Agent 状态栏](/source/agent/深入理解/2-6-Agent状态栏)
- [2-7 Agent 上下文压缩策略](/source/agent/深入理解/2-7-Agent上下文压缩策略)

<!-- 后续章节按主题续写后登记：
第三章 · 用户记忆和知识库：3-1 用户记忆系统 …
第四章 · 工具：4-1 工具分类与设计原则 …
第五章 · Coding Agent 与通用 Agent：5-1 Coding 是基础能力 …
第六章 · 交互：6-1 异步与事件驱动 …
第七章 · Agent 评估：7-1 评估任务与指标 …
第八章 · 模型后训练：8-1 四阶段全景 …
第九章 · 持续进化：9-1 学习信号 …
第十章 · 多 Agent 协作：10-1 分类框架 …
-->
