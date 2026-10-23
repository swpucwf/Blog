# Skills 技能生态

## 1. Skill 是什么

Skills 是为 AI 预定义的可复用能力包：把常用操作流程、专业知识、行为规范封装成技能，让 AI 在特定场景下自动激活。可以把它想成一份配方——AI 照着它执行，但配方之外的判断仍由模型完成。

## 2. 一个 Skill 的四类文件

| 文件 / 目录 | 角色 | 说明 |
|---|---|---|
| `SKILL.md` | 配方 | 技能核心定义：功能、触发条件、执行流程；每个 Skill 必须有 |
| `references/` | 参考书 | 背景知识、API 文档、规范说明，AI 按需查阅但不直接执行 |
| `scripts/` | 工具 | 可执行脚本，处理格式转换、数据提取等确定性任务 |
| `assets/` | 材料 | 模板、图片、字体等静态资源 |

以官方 PDF 技能为例，结构是 `SKILL.md` + `reference.md` + `forms.md` + `scripts/`：定义在配方里，规范在引用里，真正跑起来的转换逻辑在脚本里。这种划分让「模型判断」和「确定性执行」各归其位——脚本不会幻觉，模型不擅长精确格式转换。

要自己造一个，可以从官方的 `skill-creator` 起步，或照这个结构手写。

## 3. 生态与市场

| 平台 | 规模 | 特点 |
|---|---|---|
| Anthropic 官方 Skills | 约 17.8 万 star，19 个官方技能 | 质量最高，与 Claude Code 深度绑定 |
| OpenClaw Skills Registry | 约 5.3 万 star，5400+ 技能 | 社区驱动，覆盖最广 |
| awesome-claude-skills | 约 1.5 万 star | 精选列表 |
| Cursor Skills | IDE 内建 | 与 Agent、MCP 集成 |

规模差异说明一件事：官方技能解决「通用高频场景」，社区技能解决「长尾需求」。选型顺序通常是先官方、再社区、最后自建。

## 4. 官方 19 个技能的分类

按用途可以分成五组：

**开发编码**：`claude-api`（Claude API 应用开发）、`mcp-builder`（构建 MCP Server）、`frontend-design`（生产级前端）、`web-artifacts-builder`（React + Tailwind + shadcn/ui 复杂 Web 应用）、`webapp-testing`（Playwright 端到端测试）。

**文档办公**：`pdf`、`docx`、`xlsx`、`pptx` 覆盖 Office 三件套与 PDF，`doc-coauthoring` 负责文档协作撰写，`canvas-design` 负责画布设计。

**设计创意**：`brand-guidelines`（品牌视觉规范）、`theme-factory`（主题与风格）、`algorithmic-art`（p5.js 算法艺术）、`slack-gif-creator`（GIF 动图）。

**团队协作**：`internal-comms`（内部通讯文案）。

**技能管理与质检**：`skill-creator`（创建与评估技能）、`academy-guide`（推荐官方课程）、`discernment-nudge`（回答后追加 2～3 个追问，帮用户核查事实与遗漏）。

其中 `discernment-nudge` 的设计值得单独注意：它不改变回答内容，只在末尾补充质疑式追问。这类「元技能」对降低误用风险很有效，也提示了 Skill 的另一种用法——不是增加能力，而是加一道检查。

## 5. 选型建议

- 刚接触 Skill：先从官方 19 个开始，质量与文档都有保障；
- 需要特定功能：到 OpenClaw 注册表按分类搜，5400+ 覆盖面最广；
- 有固定工作流：用 `skill-creator` 自己封装，把团队规范写进 `references/`；
- 在 IDE 内使用：Cursor、Devin Desktop、TraeCode 各有内建技能生态，可直接激活。

## 6. 自建时容易忽略的点

- **触发条件要写窄**。触发描述太宽，技能会在不相关场景被激活，反而干扰回答；
- **确定性逻辑放脚本**。文件转换、正则清洗、格式校验交给 `scripts/`，不要让模型「照着描述做」；
- **规范放 references**。团队编码规范、API 版本差异这类长文本放引用目录，AI 需要时才读，不占用默认上下文；
- **一个技能只做一件事**。把「写文档 + 做 PPT + 发邮件」塞进一个技能，触发条件和执行流程都会变得含糊。

## 7. 与其他机制的关系

Skill 与 MCP 常被混为一谈，区别在层次：**Skill 是「怎么做」的知识封装，MCP 是「怎么连」的协议标准**。一个技能可以调用多个 MCP 工具，一个 MCP Server 也可以被多个技能复用。

下一节就从协议侧看 MCP 的设计与生态。

## 写在最后

技能解决「专业能力从哪来」，协议解决「能力怎么接到模型上」。下一篇[MCP 协议与生态](6-MCP协议与生态)从 2026 年的无状态化改版讲到常用服务器清单。
