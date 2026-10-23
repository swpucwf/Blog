# CLI 工具盘点

## 1. 主流 AI CLI

CLI 工具是终端里的 AI 编程助手，支持自然语言驱动开发全流程。当前主流选择：

| 工具 | 厂商 | 特点 |
|---|---|---|
| Claude Code | Anthropic | 终端级编码助手，社区偏好度最高，GitHub 星标约 14.8 万 |
| Codex CLI | OpenAI | 默认模型 `gpt-6-astra`，可选 Sol / Luna；默认挂低推理档防额度超支 |
| OpenCode | 开源 | GitHub 星标约 21 万，多模型自由 |
| DSH | DeepSeek | 开源 Agent 运行时，「一切皆插件」，但仍是开发者预览版，官方明示不可用于生产 |
| Pi | Earendil Works | 极简底座：只保留读、写、改文件与跑命令四个工具 |
| Antigravity CLI（`agy`） | Google | Gemini CLI 的继任者，个人档 $0，可选 Gemini、Claude、gpt-oss |
| Qoder CLI | Qoder-AI | 支持 CLI 与 IDE 双模式，自主编程与 CodeReview |
| CodeBuddy | 腾讯 | 命令行内自然语言驱动开发 |

## 2. 模型自由度是分水岭

能不能换模型，直接决定会不会被单一厂商绑定，也决定能否用国内 Coding Plan 降低成本：

| 工具 | 默认模型 | 可切换范围 | 自由度 |
|---|---|---|---|
| Claude Code | Claude（Opus 5.5） | 原生仅 Anthropic；可通过 Base URL 指向第三方端点 | 中 |
| Codex CLI | `gpt-6-astra` | Sol（复杂 agentic）、Luna（高频轻量） | 低（仅 OpenAI） |
| Antigravity CLI | Gemini 3.x | Gemini 3.8/3.7/3.6、Claude Sonnet/Opus 4.6、gpt-oss-120b | 中（不支持 BYOK） |
| OpenCode | 内置免费模型 | Go 订阅打包 30 款，含 GLM、Kimi、DeepSeek、Qwen、MiniMax | 极高 |
| DSH | 可切换 | 近 40 家提供方，模型 / 工具 / 循环 / UI 均可替换 | 极高 |
| Pi | 可切换 | 20+ 供应商，支持单会话内跨供应商接力 | 极高 |
| Qoder CLI | 多模型 | 多模型支持 | 中 |
| CodeBuddy | 混元系 | 混元 + DeepSeek | 低 |

换模型有两条路：工具内置的模型选择器（如 `codex --model gpt-6-sol`），或通过 Base URL + API Key 指向第三方端点。后者是国内用户把 Claude Code、Cursor 接到 Coding Plan 上的通用做法。

## 3. 选型对比

| 工具 | 开源 | 模型灵活度 | MCP | 学习门槛 | 适用场景 |
|---|---|---|---|---|---|
| Claude Code | 否 | Claude 系列 | 深度集成 | 低 | 深度编码、大型项目重构 |
| Codex CLI | 是 | OpenAI 三档 | 支持 | 低 | OpenAI 生态 |
| OpenCode | 是 | 多模型自由 | 支持 | 中 | 开源优先、自定义工作流 |
| Antigravity CLI | 否 | Gemini + Claude + gpt-oss | 支持 | 低 | Google 生态、多模态 |
| Qoder CLI | 否 | 多模型 | 支持 | 低 | 自主编程、CodeReview |
| CodeBuddy | 否 | 混元 + DeepSeek | 支持 | 低 | 腾讯生态全流程 |
| DSH | 是 | 近 40 家 | 支持 | 中 | 插件化改造 agent 运行时 |
| Pi | 是 | 20+ 供应商 | 支持 | 中 | 极简底座、跨模型接力 |

## 4. 选型建议

- **追求编码体验**：Claude Code，Agent 能力强，适合大型项目；
- **开源 + 多模型**：OpenCode，社区最活跃；
- **OpenAI 深度用户**：Codex CLI，按任务在 Astra / Sol / Luna 之间切；
- **国内生态**：Qoder CLI、CodeBuddy，中文优化好；
- **Google 全家桶**：Antigravity CLI，个人档免费且可选 Claude；
- **想拼装自己的 agent 底座**：DSH（插件化，但预览版）、Pi（四工具极简底座）。

## 5. Claude Code 五分钟上手

```bash
# 安装
npm install -g @anthropic-ai/claude-code

# 启动（首次需登录）
claude
```

进入交互后可以直接用自然语言下指令：

```text
> 帮我创建一个 Python FastAPI 项目，包含健康检查接口
> 读取 main.py 并解释它的功能
> 给这个项目加一个用户登录接口
> 运行测试并修复所有失败的用例
> 帮我做一次 CodeReview
```

国内用户可以通过 Coding Plan 使用阿里云或腾讯云的端点接入，降低调用成本；换端点的做法就是前面提到的 Base URL + API Key。

## 6. 辅助工具

多工具并存时，切换成本会成为摩擦点。`cc-switch` 这类桌面工具可以一键在 Claude Code、Codex、OpenCode、OpenClaw、Antigravity CLI 之间切换，省去手改配置。

## 7. 一个判断

CLI 工具的差异正在收敛：模型能力由 API 决定，工具层的竞争集中在三件事——**工具调用的深度**（能否稳定读写文件、跑命令、调 MCP）、**上下文管理**（长任务能否不丢线索）、**模型自由度**（会不会被绑定）。选工具时按这三条排序，比看 star 数更有效。

## 写在最后

CLI 解决「在终端里写代码」，Skills 解决「把常用能力打包复用」。下一篇[Skills 技能生态](5-Skills技能生态)看技能市场里都有什么、怎么自己造一个。
