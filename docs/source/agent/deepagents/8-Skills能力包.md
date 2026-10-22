# Skills：可复用的 Agent 能力包

## 1. Skills 是什么

工具（Tools）是原子操作——搜索一次、读一个文件、调一次 API。但有些能力需要的不是一次操作，而是**多步骤工作流 + 领域知识 + 模板资源**的组合：按团队规范做代码审查、查阅最新文档再回答、生成公司格式的报告。

一个 Skill 就是一个目录，核心是 `SKILL.md`，加上可选的脚本、参考文档和模板资源。它遵循开放的 **Agent Skills 规范**，不是 Deep Agents 私有概念。截至 2026 年，30+ 主流工具已采用：编程 Agent（Claude Code、Codex、Gemini CLI、Cursor、VS Code）、Agent 框架（Deep Agents、Goose、Roo Code、Amp、Letta）、企业平台（GitHub、Databricks、Snowflake、Spring AI）等。

> 类比：Skills 之于 AI Agent，就像 npm 包之于 Node.js——标准化的能力分发与复用机制。写一个 Skill，可以在所有兼容产品里用。

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_19-infographic-skills-ecosystem.png)

<p>图 1：Agent Skills 生态</p>

### 1.1 目录结构

```text
skills/
└── langgraph-docs/
    ├── SKILL.md              # 必需：元数据 + 指令
    ├── scripts/              # 可选：可执行脚本
    ├── references/           # 可选：详细文档
    └── assets/               # 可选：模板、数据、schema
```

| 目录 | 用途 | 加载时机 |
|---|---|---|
| `SKILL.md` | 元数据 + 核心指令 | 元数据启动时进入提示词；正文按需读取 |
| `scripts/` | 可执行脚本（Python/Bash/JS） | 按指令需要时执行 |
| `references/` | API 模式、风格指南等 | 需要深入信息时读取 |
| `assets/` | 模板、数据文件、schema | 需要时读取 |

## 2. SKILL.md 的结构

由 **YAML frontmatter** 与 **Markdown body** 组成。

### 2.1 Frontmatter

| 字段 | 必填 | 说明 |
|---|---|---|
| `name` | 是 | 小写字母、数字、连字符，1-64 字符，**必须与父目录名一致** |
| `description` | 是 | 功能与适用场景，最大 1024 字符 |
| `license` | 否 | 许可证名称 |
| `compatibility` | 否 | 环境要求（网络、特定 CLI），最大 500 字符 |
| `metadata` | 否 | 任意键值对（author、version、entrypoint） |
| `allowed-tools` | 否 | 空格分隔的预授权工具列表 |

### 2.2 Body

Frontmatter 之后是详细指令，是 Agent 激活 Skill 后实际执行的「剧本」：

```markdown
---
name: langgraph-docs
description: Use this skill for requests related to LangGraph in order to fetch relevant documentation to provide accurate, up-to-date guidance.
---

# langgraph-docs

## Instructions

### 1. Fetch the documentation index
Use the fetch_url tool to read: https://docs.langchain.com/llms.txt

### 2. Select relevant documentation
Identify 2-4 most relevant URLs. Prioritize how-to guides, core concepts, tutorials, reference.

### 3. Fetch and synthesize
Read selected URLs, then answer the user's question.
```

### 2.3 description 是最重要的字段

**Agent 选择 Skill 的唯一依据就是 `description`**——它不会提前读正文。写得具体、带明确触发条件：

```yaml
description: 当用户询问 LangGraph 的节点定义、状态管理、中断机制或部署配置时使用此技能。
```

反例：「帮助用户解决编程问题」「处理各种任务」。模糊描述会导致两种问题：**漏召回**（该用没匹配到）与**误召回**（不该用被触发）。

## 3. Progressive Disclosure：三级加载

Skills 最关键的决策是**渐进式披露**——不是一次把所有内容塞给 Agent，而是分层加载。

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_20-framework-skill-structure.png)

<p>图 2：元数据 → 核心指令 → 辅助资源</p>

| 层级 | 内容 | 加载时机 | 处理者 |
|---|---|---|---|
| Level 1 Metadata | name + description + 文件路径 | 启动时 | `SkillsMiddleware` |
| Level 2 Instructions | SKILL.md 正文 | Agent 判断相关后 `read_file` | LLM 决策 |
| Level 3 Resources | scripts/references/assets | 指令引用到某文件时 | LLM 自行决策 |

流程：

1. **启动**：`SkillsMiddleware` 扫描配置目录，从各 `SKILL.md` frontmatter 提取 name/description；系统提示词包含这些元数据与路径，正文不进上下文；
2. **读取**：用户请求后，Agent 按 description 判断是否需要，再调用 `read_file` 读 `SKILL.md`；
3. **执行**：按正文指令工作；引用到 `references/`、`assets/` 时，Agent 自行决定是否读取。

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_21-flowchart-progressive-disclosure.png)

<p>图 3：匹配与加载流程</p>

三个好处：**省 token**（20 个 Skills 只加载 20 条描述，而非 20 份完整指令）、**精准匹配**（无关指令不干扰判断）、**无限扩展**（Skill 从 5 个到 50 个，启动开销只是几十条描述）。

## 4. 三种加载方式

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_22-arch-skills-backends.png)

<p>图 4：Skills 的三种存储后端</p>

### 4.1 FilesystemBackend：直接读磁盘

```python
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

backend = FilesystemBackend(root_dir="./my-project", virtual_mode=True)

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    backend=backend,
    skills=["/skills/"],
)
```

要点：`skills` 是**路径列表**，每个路径指向「包含 Skill 子目录的父目录」——`/skills/` 对应 `/skills/myskill/SKILL.md`；直接写 `/skills/myskill.md` 不符合目录扫描约定。路径用正斜杠、相对 Backend 根目录；本地磁盘后端建议显式 `virtual_mode=True`；多路径同名 Skill **后者覆盖前者**（last wins）。

### 4.2 StateBackend：通过 state 注入

适合无磁盘环境（serverless）或动态注入：

```python
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from langgraph.checkpoint.memory import MemorySaver

backend = StateBackend()
skills_files = {
    "/skills/langgraph-docs/SKILL.md": create_file_data(skill_content),
}

agent = create_deep_agent(
    model=...,
    backend=backend,
    skills=["/skills/"],
    checkpointer=MemorySaver(),
)

result = agent.invoke(
    {"messages": [...], "files": skills_files},   # 通过 files 注入
    config={"configurable": {"thread_id": "12345"}},
)
```

要点：虚拟路径必须以 `/` 开头；内容必须用 `create_file_data()` 格式化（原始字符串会报错）；每次 `invoke` 都要传 `files`。

### 4.3 StoreBackend：跨线程持久化

文件只写一次，所有线程可访问：

```python
from deepagents.backends import StoreBackend
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
backend = StoreBackend(namespace=lambda _rt: ("filesystem",))

store.put(
    namespace=("filesystem",),
    key="/skills/langgraph-docs/SKILL.md",
    value=create_file_data(skill_content),
)

agent = create_deep_agent(model=..., backend=backend, store=store, skills=["/skills/"])
```

`namespace` 是函数，接收 runtime config 返回命名空间元组。适合生产环境多用户、多线程共享 Skills。

### 4.4 多源与分层覆盖

```python
agent = create_deep_agent(
    model=...,
    skills=[
        "/skills/shared/",    # 团队共享
        "/skills/project/",   # 项目专属，优先级更高
    ],
)
```

同名时后面的覆盖前面的。典型分层：组织级（公司规范、安全审查）、团队级（工作流、Review 标准）、项目级（部署流程、测试策略）。

### 4.5 运行时动态加载

`skills` 是普通 Python 列表，可按用户角色、租户、请求类型动态构造：

```python
SKILLS_BY_ROLE = {
    "engineering": ["/skills/code-review/", "/skills/testing/"],
    "data": ["/skills/sql-analysis/", "/skills/visualization/"],
}

def create_agent_for_user(role: str):
    return create_deep_agent(model=..., skills=SKILLS_BY_ROLE.get(role, []))
```

好处：只加载当前任务相关的 Skills，既省 token 又减少误匹配。

## 5. Skills 与子 Agent

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_24-arch-skills-subagent.png)

<p>图 5：子 Agent 的 Skills 继承规则</p>

- **general-purpose 子 Agent**：自动继承主 Agent 的所有 Skills；
- **自定义子 Agent**：不继承，必须在定义时用 `skills` 字段显式指定；
- **状态完全隔离**：每个 Agent 有独立 Skill 状态空间，一个 Agent 修改 Skill 文件不影响其他 Agent。

```python
agent = create_deep_agent(
    model=...,
    skills=["/skills/main/"],        # 主 Agent 与 GP 子 Agent 使用
    subagents=[{
        "name": "researcher",
        "description": "Research assistant with specialized skills",
        "system_prompt": "You are a researcher.",
        "tools": [web_search],
        "skills": ["/skills/research/", "/skills/web-search/"],  # 子 Agent 专属
    }],
)
```

职责更清晰：研究型子 Agent 不需要接触主 Agent 的编排类 Skill；状态隔离也避免并发竞态。

## 6. 权限控制

生产环境的 Skill 权限有三个维度：**可见性**（能否发现与读取）、**写入权限**（能否修改文件）、**审批流程**（写入是否需要人确认）。

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_23-arch-skills-permissions.png)

<p>图 6：Skill 权限控制（共享与个人分层）</p>

### 6.1 只读 Skills（企业知识库）

目标：允许读 `/skills/**`，拒绝写入。

```python
agent = create_deep_agent(
    model=...,
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/skills/": StoreBackend(namespace=org_skill_namespace)},
    ),
    skills=["/skills/"],
    permissions=[
        FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="deny"),
    ],
    store=store,
)
```

`deny` 会拦截 `write_file`、`edit_file` 与 v0.7 默认的 `delete`；Skill 库只能由管理员直接操作 Store 更新。

> **路径要区分**：`CompositeBackend` 收到 `/skills/test-skill/SKILL.md` 后先剥离挂载前缀 `/skills/`，再把 `/test-skill/SKILL.md` 交给内部 Store。因此通过 Store API 预填时 key 写 `/test-skill/SKILL.md`；误写成 `/skills/...` 会出现 `/skills/skills/...`。只有直接把 StoreBackend 当根 Backend、不经路由时，key 才保留完整路径。

### 6.2 写入需审批

```python
agent = create_deep_agent(
    model=...,
    skills=["/skills/personal/"],
    permissions=[
        FilesystemPermission(operations=["write"], paths=["/skills/**"], mode="interrupt"),
    ],
    checkpointer=MemorySaver(),   # interrupt 必需
)
```

Agent 尝试写入时暂停，把修改呈现给审批者；通过后恢复，拒绝则回滚。需要 `deepagents>=0.6.8`；LangGraph Studio 会自动渲染审批 UI，API 调用需轮询状态并提交结果。

### 6.3 共享 + 个人分层

最常见部署是两层：团队共享只读库 + 用户个人可写空间。

```python
agent = create_deep_agent(
    model=...,
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/skills/shared/": StoreBackend(namespace=shared_skill_namespace),
            "/skills/personal/": StoreBackend(namespace=personal_skill_namespace),
        },
    ),
    skills=["/skills/shared/", "/skills/personal/"],
    permissions=[
        FilesystemPermission(operations=["write"], paths=["/skills/shared/**"], mode="deny"),
    ],
    store=store,
)
```

效果：shared 写入被 deny，只有管理员更新；personal 无写入限制；两处都在 `skills` 列表，启动时发现并注入元数据。同名时 last-wins——personal 排在后面，个人版本覆盖共享版本，允许个性化而不影响他人。

> `StoreBackend` 不会自动创建存储。本地跑必须接实际 Store（`InMemoryStore` 仅演示，进程退出即丢）；LangSmith Deployment 托管部署时平台自动提供 Store，应省略 `store=`。

## 7. 用 Skills 执行代码

![配图](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/deepagents_25-arch-skills-code-execution.png)

<p>图 7：两种代码执行模式</p>

### 7.1 沙箱脚本

Skill 可在 `scripts/` 放可执行脚本。Agent 可从任何后端**读取**脚本，但要**执行**需要支持沙箱执行的后端（如 DaytonaSandbox）。

```text
skills/arxiv-search/
├── SKILL.md
└── scripts/search.py
```

`SKILL.md` 指令：运行 `scripts/search.py` 并传入用户查询 → 解析结果 → 展示标题、作者、摘要与链接。

执行机制：沙箱在隔离容器运行脚本，Agent 取 stdout 作为结果。若 Skill 文件在沙箱外（如 StateBackend），需自定义中间件在 `before_agent` 上传脚本、`after_agent` 下载输出。

### 7.2 解释器技能

把 Skill 中的代码模块暴露给 Agent 的代码解释器，Agent 直接 `import` 经过测试的函数，不必每次重新生成。

前提：`deepagents[quickjs]`、`langchain-quickjs>=0.2.0`、Python 3.11+。解释器运行在 QuickJS 内存环境，适合确定性 JS/TS helper；要执行 Shell、装包、访问完整文件系统，应使用沙箱。

三步配置：frontmatter 的 `metadata.entrypoint` 指定入口（JS/TS）→ `CodeInterpreterMiddleware` 使用同一后端 → Agent 通过 `await import("@/skills/<name>")` 导入。

```yaml
---
name: order-helpers
description: Helper functions for normalizing and grouping order records.
metadata:
  entrypoint: scripts/index.ts
---
```

```typescript
// skills/order-helpers/scripts/index.ts
export function groupByStatus(orders: Order[]) {
  return orders.reduce((acc, order) => {
    (acc[order.status] ??= []).push(order);
    return acc;
  }, {});
}
```

```python
agent = create_deep_agent(
    model=...,
    backend=backend,
    skills=["/skills/"],
    middleware=[CodeInterpreterMiddleware(skills_backend=backend)],
)
```

优势是**确定性**（测试过的函数不会因 LLM 重写而偏差）与**省 token**（只写一行 import，而不是几十行实现）。

## 8. Skills vs Memory vs Tools

|  | Skills | Memory（AGENTS.md） | Tools |
|---|---|---|---|
| 用途 | 按需加载的领域能力 | 启动时加载的持久上下文 | 可调用的原子操作 |
| 时机 | 判断相关时才读 | 启动时 | 每轮可用 |
| 格式 | 目录中的 `SKILL.md` | `AGENTS.md` | 绑定到 Agent 的函数 |
| 优先级 | last wins | 用户级 + 项目级合并 | 创建时定义 |
| 场景 | 任务专属、可能很大的指令集 | 始终相关的全局规范 | 需要执行操作 |

决策规则：**所有对话都需要的** → Memory；**特定任务才需要的专业指令** → Skills；**需要执行的原子操作** → Tools。

Skills 与 Memory 是连续光谱：Agent 可在工作中更新自己的 Skills，因此 Skills 也能充当**渐进式披露的记忆**。

## 9. 编写高效 Skills

1. **Frontmatter 精简**：description 具体、带触发条件，避免泛泛而谈；
2. **正文控制在 5000 tokens / 500 行内**：超出部分拆到 `references/`，在正文中引用；
3. **为 Agent 写结构**：步骤化流程（1-2-3）、决策标准（if X then A）、输入/输出示例、边界情况处理；
4. **管控数量**：少量清晰胜于大量模糊；Skill 太多容易误匹配，定期合并重叠、删除过时、拆分过大。

## 10. 本篇验收

- [ ] 能说清 Skills 与 Tools 的差别（工作流 + 知识 + 资源）；
- [ ] 记得 `SKILL.md` 的必填字段与「name 必须等于目录名」；
- [ ] 理解三级渐进式加载，以及省 token/精准匹配/可扩展三个理由；
- [ ] 能分别用 Filesystem / State / Store 三种方式加载 Skills；
- [ ] 知道多源同名 last-wins 与组织/团队/项目分层；
- [ ] 记得 general-purpose 继承、自定义子 Agent 不继承、状态隔离；
- [ ] 能配 deny / interrupt 权限与共享+个人分层；
- [ ] 知道沙箱脚本与解释器技能的前置条件；
- [ ] 能按「全局/任务/操作」区分 Memory、Skills、Tools。

> **补记**：Skills 真正解决的问题是「**按需加载的专业能力**」——元数据常驻、正文按需、资源再按需。写好一个 Skill 的关键几乎都在 description 与正文结构：description 决定能否被选中，步骤化指令决定被选中后能否稳定执行。权限上最容易错的是 CompositeBackend 的路径前缀（Store key 用路由相对路径），以及共享库的写保护必须显式 deny。

## 写在最后

下一篇[长期记忆](9-长期记忆)讲跨对话记忆：Checkpointer 的短期状态、Store 的跨线程记忆，以及 `memory=` 加载的 AGENTS.md 到底解决什么问题。
