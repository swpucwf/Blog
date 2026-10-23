# MCP 协议与生态

## 1. MCP 解决什么问题

MCP（Model Context Protocol）是 Anthropic 提出的开放协议，为 LLM 应用与外部数据源、工具之间提供标准化集成方式，常被比作「AI 应用的 USB-C 接口」。

没有它时，每接一个工具都要写一套适配；有了它，工具按协议暴露能力，任何支持 MCP 的客户端都能直接调用。

## 2. 协议演进：从有状态到无状态

传输与状态模型经历了两轮变化，排错和选型前值得先理清这条线：

| 阶段 | 传输方式 | 状态模型 |
|---|---|---|
| 早期 | stdio（本地）/ HTTP+SSE（远程） | Server 维护会话状态 |
| 2025 年 | Streamable HTTP 取代 HTTP+SSE | 仍依赖会话 |
| 2026-07-28 规范 | Streamable HTTP 继续演进 | 协议核心转为无状态 |

2026-07-28 是 MCP 发布以来最大的一次修订，核心是去状态化：

- **移除握手与会话**：删掉 `initialize` / `initialized` 握手与 `Mcp-Session-Id`，Server 不再维护连接级状态，可直接水平扩展、按请求无服务化部署；
- **请求头路由**：新增 `Mcp-Method` / `Mcp-Name`，网关和负载均衡无需解析 body 就能按方法、工具名路由；
- **能力发现与缓存**：新增 `server/discover`，列表类结果可缓存，减少重复往返；
- **多轮请求**：支持单次交互内多次往返确认；
- **扩展框架**：引入可选扩展机制（首批为 Tasks、MCP Apps、EMA）；
- **安全与治理**：OAuth 2.1 加固，并确立正式弃用政策，被弃用能力保留 12 个月过渡窗口。

同时被弃用的是 `Roots`、`Sampling`、`Logging` 三个早期能力，以及旧的 HTTP+SSE 传输。

迁移建议很直接：新项目直接按新规范实现；如果 Server 依赖会话状态，把状态改成显式参数或外部存储。

## 3. 常用 MCP Server

| Server | 功能领域 |
|---|---|
| Chrome DevTools | 调试、性能分析，与自己写前端时的排查流程最贴 |
| Context7 | 实时获取最新库文档与代码示例，编码场景必备 |
| Playwright | 微软官方，精确控制网页、自动化测试与抓取 |
| GitHub | Issues、PR、仓库管理，官方维护 |
| Serena | 语义级代码检索与编辑，给 Agent 装上 IDE 级理解力 |
| Firecrawl / Jina AI | 网页抓取转 Markdown、语义与跨模态搜索 |
| Notion / 飞书 | 知识库、IM、日历打通 |
| Figma | 把设计稿的布局与样式信息提供给 Agent |
| n8n | 工作流自动化，数百个集成，原生支持 MCP |
| FastMCP | 快速构建 MCP Server 的 Python 框架 |

更全的清单可以看社区维护的 `awesome-mcp-servers`。

## 4. 按场景选

| 场景 | 推荐 Server |
|---|---|
| 编码 | Context7 + GitHub + Playwright |
| 设计协作 | Figma |
| 团队知识 | Notion、飞书 |
| Web 自动化 | Playwright、Chrome DevTools |
| 数据抓取 | Firecrawl、Jina AI |
| 自建服务 | FastMCP |

一个经验判断：**编码场景的收益最直接**。Context7 解决「模型不知道库的最新 API」，GitHub 解决「能读仓库能提 PR」，Playwright 解决「改完能自己验证」。这三件凑齐，Agent 才算形成闭环。

## 5. 客户端支持

主流客户端基本都已支持：Claude Desktop、Claude Code、Cursor、Devin Desktop（原 Windsurf）、TraeCode、Cherry Studio。选客户端时看两点——能不能配项目级 MCP、能不能看到工具调用日志。

## 6. 五分钟写一个 MCP Server

```bash
pip install fastmcp
```

```python
from fastmcp import FastMCP

mcp = FastMCP("my-first-server")

@mcp.tool()
def hello(name: str) -> str:
    """向用户打招呼"""
    return f"你好，{name}！这是你的第一个 MCP 工具。"

if __name__ == "__main__":
    mcp.run()
```

运行 `fastmcp run server.py` 即可。在 Claude Code 里接入，项目级配置写在根目录 `.mcp.json`：

```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["server.py"]
    }
  }
}
```

也可以用 `claude mcp add my-server -- python server.py`，默认写入用户级配置，加 `-s project` 写入项目级。

两个容易踩的点：MCP 配置**不在** `.claude/settings.json` 里（那个文件放 env、hooks、permissions）；工具的描述文本直接决定模型会不会选它、参数填得对不对，写得含糊就会误调用。

## 7. 设计层面的三条经验

- **工具粒度按「一个动作」切分**。一个工具里塞多个分支，模型很难选对参数；
- **返回结构要稳定**。字段名变化会让上层提示词失效；
- **把权限控制放在工具侧，而不是提示词里**。只读与写入分成两组工具，比在提示词里反复强调「不要删文件」可靠得多。

## 写在最后

协议解决「能力怎么接」，下一个问题是「写代码的界面长什么样」。下一篇[IDE 编程工具](7-IDE编程工具)对比 AI 原生 IDE 与 Web 编程平台。
