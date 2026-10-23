# A2A 与 ANP 协议

MCP 让智能体拿到了别人的工具，但工具之间的编排仍然留在代码里：谁先跑、结果交给谁、失败了找谁重试，全在外部写死。想让智能体自己去协商任务，就需要另一类协议——面向智能体之间而不是智能体与工具之间。这一层目前有两条路线：A2A 管点对点协作，ANP 管大规模网络里的发现与路由。

## 1. 星型编排的三个问题

多智能体协作最直观的做法是中心协调者：研究员、撰写员、编辑都听一个调度器的。小规模下没问题，放大之后有三个结构性缺陷：

- **单点故障**：协调器挂了，整条链路停摆。
- **性能瓶颈**：所有消息都过中心节点，并发上限就是中心节点的吞吐。
- **扩展困难**：加一个角色要改中心逻辑，角色越多，调度器越像一棵长歪的分支树。

A2A（Agent-to-Agent）换的是拓扑：网状、点对点，智能体直接对话。它不设中心节点，每个参与者既发请求也接请求。

<img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/helloagents-a2a-design.png" width="85%" alt="A2A 设计思想：智能体之间点对点通信，每个节点既是提供者也是消费者" />

## 2. Task 与 Artifact：A2A 的两个新抽象

MCP 的交互单元是「一次工具调用」，输入参数、输出结果，调用完就结束。协作不是这样——它需要跨越多次消息、可能被拒绝、可能被转交、可能中途要补充信息。所以 A2A 引入了两个概念：

- **Task（任务）**：协作的最小单元，包含目标、状态、参与方。它是有生命周期的对象，不是一次性请求。
- **Artifact（工件）**：任务的产出物。撰写员交出的稿件、编辑给出的评审意见，都是 artifact，可以分版本、可以互相引用。

Task 的状态覆盖了协作中真实会发生的事：创建、协商、转交（代理）、执行中、完成、失败。有了显式状态，进度跟踪和异常处理才有落点——「对方说做不了」和「对方没收到」是两回事，前者应该触发重新分派。

<img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/helloagents-a2a-task-lifecycle.png" width="85%" alt="A2A 任务生命周期：创建、协商、转交、执行、完成或失败" />

一次 A2A 请求走四步：**代理发现**（先弄清对方有哪些能力）→ **身份验证**（确认对面是谁）→ **发送消息**（普通请求）→ **发送消息流**（需要持续进度时用流式）。后两步是同一件事的两种形态，区别只在要不要中途回包。

和 MCP 的差别也在这里：MCP 关心「这个服务器有哪些函数」，A2A 关心「这个智能体能不能接下这个任务、它现在有没有空、它交出来的东西算不算完成」。

## 3. 把能力声明成技能

A2A 的服务端写法与 MCP 服务器很像，但注册的不是函数工具，而是**技能**（skill）——一个技能代表一类可以承接的任务：

```python
from hello_agents.protocols import A2AServer

researcher = A2AServer(
    name="researcher",
    description="负责搜索和分析资料的Agent",
    version="1.0.0",
)

@researcher.skill("research")
def handle_research(text: str) -> str:
    """处理研究请求"""
    topic = text.removeprefix("research ").strip()
    return str({"topic": topic, "findings": f"关于{topic}的研究结果..."})

researcher.run(host="localhost", port=5000)
```

`capabilities` 与技能名是给外部看的「能力目录」。发现方据此判断该不该把任务发过来——这部分信息等价于 MCP 里的 `list_tools()`，只是粒度更粗：工具是「能做什么动作」，技能是「能接什么活」。

客户端只有一句：

```python
from hello_agents.protocols import A2AClient

client = A2AClient("http://localhost:5000")
response = client.execute_skill("research", "research AI在医疗领域的应用")
print(response.get("result"))
```

注意 `execute_skill` 的第二个参数是一整段自然语言，而不是结构化的参数字典。这是 A2A 与 MCP 在接口风格上的分野：MCP 面向函数，参数必须严格匹配 schema；A2A 面向对话，交给对方一段话，怎么理解是对方的事。

## 4. 串起一条协作链

单跳调用只是开始，A2A 真正省事的地方是把「编排」变成「调用」：

```python
research = researcher_client.execute_skill("research", f"research {topic}").get("result", "")
article  = writer_client.execute_skill("write", f"write {research}").get("result", "")
final    = editor_client.execute_skill("edit", f"edit {article}").get("result", "")
```

三个服务各跑在一个端口上，互相不知道对方存在，链路由客户端串起来。这里仍然有编排者，但它是**流程的一部分**而不是**基础设施的一部分**——想换掉编辑环节，改一句调用就行，不需要动其他智能体的注册信息。

更进一步是协商。A2A 允许对方案还价：

```python
# 提案方
proposal = {"task": task, "deadline": deadline}

# 承接方评估后可以拒绝并给出反提案
{"accepted": False, "message": "时间太紧", "counter_proposal": {"deadline": 7}}
```

这在 MCP 里是没有对应物的：工具不会跟你讨价还价，它要么执行要么报错。协商能力来自「对等」这个前提——双方都是智能体，都有判断力。

## 5. A2ATool：远端智能体就是一个工具

对上层智能体来说，「调用另一个智能体」和「调用一个工具」在使用上没有区别。`A2ATool` 正是这层转换：

```python
from hello_agents.tools import A2ATool

coordinator = SimpleAgent(name="协调者", llm=llm)
coordinator.add_tool(A2ATool(
    name="researcher",
    description="研究员Agent，可以搜索和分析资料",
    agent_url="http://localhost:5000",
))

coordinator.run("请让研究员帮我研究AI在教育领域的应用")
```

包装之后，协调者的工具表里多了一个 `researcher`，模型按描述决定何时调用。典型的客服系统就是这么搭的：接待员是唯一面向用户的智能体，技术专家和销售顾问各自是一个 `A2AServer`，通过两个 `A2ATool` 挂到接待员身上，由模型按问题类型分派。

这个结构有个值得注意的性质：**中心节点回来了，但只在需要的时候回来**。接待员确实是中心，但它只负责分诊；专家之间若要直接协作，不必绕回接待员。相比把全部逻辑压在一个调度器里，这种「局部集中、整体对等」更接近真实组织的工作方式。

## 6. ANP：网络变大之后的新问题

A2A 假设你知道对方在哪——`agent_url` 是写死的。十个智能体可以这么干，一千个不行。ANP（Agent Network Protocol）处理的就是这一层：

- **服务发现**：新任务到达时，怎么找到能处理它的智能体？
- **智能路由**：多个智能体都能做，按负载、成本、能力选哪个？
- **动态扩展**：新节点加入后，怎么被其他成员发现？

ANP 的做法是公开一个发现服务，靠爬取各智能体暴露的标准端点（`.well-known/agent-descriptions`）建立索引，查询时按语义或功能描述匹配。身份验证走 DID：请求方用私钥对含自身 DID 的请求签名，接收方解析 DID 拿到公钥验签，双方不需要预先交换凭证。

三步流程可以概括为：**发现与匹配 → 基于 DID 的身份验证 → 标准化的服务执行**。

<img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/helloagents-anp-flow.png" width="85%" alt="ANP 整体流程：服务发现、DID 身份验证、标准化服务执行" />

这里需要说清楚定位：ANP 目前是社区维护的概念性框架，生态还没成型。它的价值不在「现在就能上生产」，而在于把「开放网络里的智能体如何互认」这个问题提前摆出来——DID 做信任根、标准描述做发现，这套思路值得记住。

## 7. 注册、发现、按负载选

ANP 的实现把「注册」和「发现」拆成两个动作，中间靠元数据做决策：

```python
from hello_agents.protocols import ANPDiscovery, register_service, discover_service

discovery = ANPDiscovery()

register_service(
    discovery=discovery,
    service_id="nlp_agent_1",
    service_name="NLP处理专家A",
    service_type="nlp",
    capabilities=["text_analysis", "sentiment_analysis", "ner"],
    endpoint="http://localhost:8001",
    metadata={"load": 0.3, "price": 0.01, "version": "1.0.0"},
)

nlp_services = discover_service(discovery, service_type="nlp")
best = min(nlp_services, key=lambda s: s.metadata.get("load", 1.0))
```

`metadata` 是这套设计的重点。服务发现只解决「有哪些」，选哪个要靠元数据：负载、价格、版本、GPU 有无。这些都是运行期会变的量，写不进静态配置，只能随注册信息一起上报。

有了发现能力，调度就能交给模型：

```python
scheduler = SimpleAgent(name="任务调度器", llm=llm, system_prompt="""...选择节点时考虑：负载、CPU核心数、内存、GPU等因素。""")
scheduler.add_tool(ANPTool(name="service_discovery", description="服务发现工具，可以查找和选择计算节点", discovery=discovery))

scheduler.run("为以下任务选择最合适的计算节点：训练一个大型深度学习模型，需要GPU支持")
```

也可以不用模型，直接用规则做负载均衡——`min(servers, key=lambda s: s.metadata["load"])` 就是最小连接数策略。两种做法没有高下之分：规则可预测、成本低；模型能处理「需要 GPU 且内存大于 32G」这类复合条件，但每次决策都要付一次推理。

`ANPNetwork` 再往外一层，把注册好的服务连成图，提供节点数与连接关系的统计。它解决的是「拓扑可见性」：出问题时能看出是某个节点挂了，还是某条链路断了。

## 8. 三协议的分层与选型

三种协议在 HelloAgents 里被压成一致的三层，这是整个设计最省心的地方：

<img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/helloagents-protocol-layers.png" width="85%" alt="HelloAgents 通信协议三层设计：协议实现层、工具封装层、智能体集成层" />

- **协议实现层**：MCP 基于 FastMCP，A2A 基于 `a2a-sdk`，ANP 是自研的轻量实现。
- **工具封装层**：`MCPTool` / `A2ATool` / `ANPTool` 都继承 `BaseTool`，对外都是 `run()`。
- **智能体集成层**：智能体只管把协议工具加进工具表，不感知底层是哪种协议。

对上层来说，三种协议的区别只剩下工具描述。这也是选型的判断依据：

| 需求 | 选 | 理由 |
| --- | --- | --- |
| 访问文件、数据库、外部 API | MCP | 生态最成熟，服务器现成可用 |
| 多个智能体分工完成一个任务 | A2A | 有 Task/Artifact 与协商语义，能表达「做不完」 |
| 大规模网络里动态找服务 | ANP | 有发现、路由与身份机制，但仍是概念实现 |

三条边界提醒：

- **MCP 与 A2A 不是二选一**。一个智能体可以同时挂 `MCPTool` 和 `A2ATool`，用工具干活、用 A2A 找同伴。
- **A2A 的实现仍以示例为主**。`a2a-sdk` 需要单独安装，很多能力停留在 sample code 级别，接入前先确认版本与维护状态。
- **协议不解决治理**。谁有权调用、调用记录留多久、失败如何补偿，都在协议之外，需要自己设计。

协议层看完，还有一个更实际的问题：这些架构到底有没有变好？下一个主题换到评估——如何用可复现的指标判断一个智能体是不是真的更强：[12-智能体性能评估](/source/agent/从零构建智能体/12-智能体性能评估)。
