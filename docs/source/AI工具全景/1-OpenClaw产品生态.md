# OpenClaw 产品生态

## 1. OpenClaw 是什么

「龙虾」是开源 AI 智能体 OpenClaw（原名 Clawdbot / Moltbot）的昵称，源于其红色机械龙虾图标。它能理解自然语言指令，自主调用电脑工具完成任务，形成从意图到执行的闭环。仓库 `openclaw/openclaw` 为 TypeScript 实现、MIT 协议，六个多月即达到 39 万 star。

核心能力可以归成六类：

| 能力 | 具体内容 |
|---|---|
| 模型自由 | 对接 Claude、GPT、Gemini、DeepSeek、Kimi、Qwen、GLM，也支持 Ollama 本地模型 |
| 系统操控 | 读写文件、运行代码（可沙盒）、模拟键鼠、操控浏览器 |
| 多 IM 接入 | WhatsApp、Telegram、Discord、Slack、Signal、iMessage、微信、飞书、QQ |
| 持久记忆 | 记住偏好与上下文，跨会话保持个性化 |
| 技能系统 | ClawHub 技能市场（已拆为 Skills + Plugins 双结构），社区另维护数千个 Skills |
| 多智能体 | 同时运行多个实例协同；支持 cron 定时、后台常驻、语音通话 |

## 2. 版本与生态动态

项目节奏非常快，几个值得记住的节点：

- 2026-09 发布 v2026.9.5 正式版，另设 extended-stable 渠道面向求稳的生产用户；
- 2026-08 发布 2.0（v2026.8.1），官方口径为 987 位贡献者、近 1.7 万个 PR；
- 2026-07 转入 OpenClaw Foundation（美国 501(c)(3) 非营利组织），协议保持 MIT；
- 2026-02 创始人加入 OpenAI，项目仍以开源形式推进；
- 从创建到 37 万 star 只用了约半年。

对一个工具而言，这种节奏意味着两件事：功能更新快，但**接口与最佳实践也在变**，生产使用前应锁定版本，并优先选择 extended-stable 渠道。

## 3. 国内产品矩阵

2026 年 3 月起，国内厂商密集推出 Claw 类产品，从最初的 6 款扩展到 12 款以上。纳入对比的产品都有厂商官方入口：

| 产品 | 厂商 | 端形态 | 模型 | 定位 |
|---|---|---|---|---|
| OpenClaw | 开源社区 | 本地 / 云端 | 任意模型 | 开源旗舰，生态最全 |
| AutoClaw（澳龙） | 智谱 AI | 本地（Win / macOS 一键安装） | 完全开放 | 国内首个一键安装本地版，预置 50+ Skills |
| KimiClaw | 月之暗面 | 云端（另有桌面 / 安卓托管） | Kimi 系 | 一键云部署，40GB 云存储 |
| MaxClaw | MiniMax | 云端 | MiniMax 系 | 10 秒部署，语音与音乐能力强 |
| ArkClaw | 火山引擎 | 云端 SaaS | 豆包系 + 第三方 | 提供企业版，席位制 |
| QClaw | 腾讯 | 本地 PC | 混元系 | 支持多 Agent 并行与连接器 |
| 小艺Claw | 华为 | 手机 / 平板 / PC | 盘古、DeepSeek、MiniMax 可切 | HarmonyOS 全机型开放，500+ Skills |
| JVSClaw | 阿里云（无影） | 云端（Web / iOS） | 多模型 | 运行在 ClawSpace 沙箱 |
| DuMate | 百度智能云 | PC + 移动 | 文心系 | 已发布 15 个行业套件 |
| Xiaomi miclaw | 小米 | 手机（含手表版） | MiMo | 邀请制封测，走 OTA |
| LobsterAI | 网易有道 | PC 桌面（可 NAS 部署） | 模型自由 | 开源桌面办公 Agent |
| AstronClaw | 科大讯飞 | 云端 SaaS | 星火 + 多模型可切 | 支持企微、钉钉、飞书 |

另有几款值得单独留意：HiClaw 是阿里云 Higress 团队的开源团队版（Manager–Worker 架构）；纳米Work 由 360 安全龙虾升级而来；UniClaw 来自中国联通；Mavis 是 MiniMax Agent 升级更名后的多智能体平台，与 MaxClaw 并存。

## 4. 引用前必须核对的易错点

社区文章里的厂商标注错误相当集中，几条高频：

1. **AutoClaw 属于智谱 AI**，不是字节跳动，中文名「澳龙」；
2. **UniClaw 属于中国联通**，不是腾讯；
3. **JVSClaw、HiClaw、CoPaw 同属阿里云但分属不同团队**（无影 / Higress / 通义）；
4. **MaxClaw ≠ Mavis**：前者是云端托管服务，后者是多智能体平台；
5. **AutoClaw 的模型完全开放**，可接任意模型的 Coding Plan 或 API，并非只能跑 GLM；
6. **KimiClaw 的官方入口是 `kimi.com/bot`**，第三方「kimi-claw」站点多为仿冒。

## 5. 按端形态选型

| 端形态 | 代表产品 | 适合谁 |
|---|---|---|
| 本地 PC | OpenClaw、AutoClaw、QClaw、LobsterAI | 数据不出本机、要操控本地文件与软件 |
| 云端 | KimiClaw、MaxClaw、ArkClaw、JVSClaw、AstronClaw | 零部署开箱即用、要 7×24 常驻 |
| 手机 / 平板 | 小艺Claw、Xiaomi miclaw | 随时下派任务，受机型限制较多 |
| 跨端 | DuMate、小艺Claw | 桌面与移动都要覆盖 |

能力对照上，差距主要在四处：是否开源、模型自由度、Skills 市场、本地部署。开源只有 OpenClaw 与 LobsterAI；模型自由度只有 OpenClaw 与 AutoClaw 真正开放；Skills 市场规模以 ClawHub 与 AutoClaw 预置为主；本地部署则集中在 PC 类产品。

## 6. 选型建议

- 追求最大自由度与生态规模：OpenClaw；
- 想低门槛本地跑起来：AutoClaw（一键安装、预置技能、模型开放）；
- 数据必须留在本机且要开源可控：LobsterAI；
- 不想装东西：KimiClaw、MaxClaw、JVSClaw；
- 手机上下派任务：小艺Claw（机型覆盖广）、Xiaomi miclaw（邀请制）；
- 企业采购要席位管理：ArkClaw 企业版、HiClaw；
- 需要高性能或桌面体验：Rust 重写版 claw-code、GUI 版 ClawX。

## 7. 周边生态

社区围绕 OpenClaw 已经长出一圈工具：claw-code（Rust 高性能重写，曾刷新最快破 10 万 star 的纪录）、NVIDIA NemoClaw（安全沙箱）、ClawX（桌面 GUI）、AutoResearchClaw（科研自动化）、ClawWork（自动化工作流）、ClawTeam（多 Agent 集群）、clawpanel（多引擎管理面板）、汉化版（CLI 与 Dashboard 全中文）。

这一层生态的价值在于补齐短板：官方版本偏「能力全」，周边工具分别解决性能、安全、易用性与行业化。

## 写在最后

Claw 类产品的竞争点已经从「能不能自主执行」转向「模型自由度、技能生态与部署形态」。下一篇[Coding Plan 与成本](2-CodingPlan与成本)看订阅套餐的定价结构与选型逻辑——这直接决定日常用 AI 编程的月度成本。
