# 上下文工程与 ContextBuilder

记忆提供了「可以取的信息」，RAG 提供了「可以查的知识」，但每次调用真正送给模型的，是临时拼装出来的一段文本。拼什么、拼多少、按什么顺序拼，直接决定输出质量。这件事有个比提示工程更大的名字：上下文工程。

## 1. 从提示工程到上下文工程

提示工程关心的是「话怎么说」——系统提示怎么写、结构怎么组织、示例放几个。上下文工程关心的是「推理时那组 token 的最优集合」：除了提示本身，还包括工具定义、检索结果、消息历史、外部文件，凡进入上下文窗口的都算。

<img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/helloagents-prompt-vs-context.webp" width="85%" alt="提示工程与上下文工程的差别" />

两者的分界来自工作形态的变化。单轮分类或文案改写，提示写好就定了；而循环运行的智能体每轮都在产生新信息，这些信息必须被周期性地挑选、压缩、丢弃。于是问题从「怎么写提示」变成「从不断膨胀的候选信息里，挑哪些进窗口」——这正是工程化的抓手。

## 2. 上下文腐蚀与注意力预算

模型并不同等地利用长上下文。针堆找针一类基准显示：**窗口里的 token 越多，模型准确回忆其中信息的能力反而下降**，这个现象叫上下文腐蚀（context rot）。原因在架构里：Transformer 让每个 token 与所有 token 建立关联，理论上 \(n^2\) 级别的关系，长度一长，注意力被「拉薄」；而训练数据里短序列远多于长序列，模型对长程依赖的经验本就稀少。

位置编码插值一类技术能让模型在推理时适配更长序列，代价是对位置的精确理解变差。综合结果是**性能梯度而不是悬崖**：长上下文依然可用，但检索与长程推理的精度会下滑。

工程结论很直接：**上下文是有限资源，且边际收益递减**。每多放一个 token 就消耗一点注意力预算，所以要按「用尽可能少、但信号密度高」的标准来组织。

## 3. 有效上下文的三个组件

- **系统提示**：信息层级要「刚刚好」。两个极端都坏——把复杂 if-else 硬编码进提示，脆弱且难维护；只给宏观目标，又缺少期望输出的具体信号。做法是分区组织（背景、指令、工具指引、输出描述），用 Markdown 或 XML 分隔，先在最简版本上试跑，按失败模式增补。追求「最小必要信息集」，注意最小不等于最短。
- **工具**：工具定义是智能体与信息空间的契约。要职责单一、彼此低重叠、入参无歧义、对错误鲁棒。常见失败是「臃肿工具集」——功能边界模糊，选哪个工具本身就含混；一句经验判断很实用：**如果人类工程师都说不准该用哪个，别指望模型选对**。
- **示例**：始终推荐给示例，但不必罗列所有边界条件。挑一组多样且典型的，直接画出期望行为。

<img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/helloagents-calibrating-prompt.webp" width="85%" alt="系统提示的校准" />

## 4. 检索的两种形态

「智能体 = 在循环中自主调用工具的 LLM」。既然有循环，检索就不必都在推理前一次性完成，于是分成两条路：

**预检索（embedding 检索）**：提问时先查一遍，把 top-k 片段塞进上下文。优点是简单、快、可缓存；缺点是把「当时以为相关」的东西一次押上，命中率取决于查询改写。

**及时检索（just-in-time）**：不预先加载数据，只维护轻量引用——文件路径、URL、数据库查询串——运行时再用工具取。像人查资料：不背全书，靠目录、书签与搜索按需取用。引用的元数据本身也在传递信号：`tests/test_utils.py` 与 `src/core/test_utils.py` 的语义暗示完全不同，时间戳暗示时效，目录层级暗示归属。

JIT 还有个副产品叫**渐进式披露**：每一步交互产生的新上下文会指导下一步决策——文件大小暗示复杂度，命名暗示用途，时间戳暗示相关性。智能体按层构建理解，工作记忆只留当前必要子集，其余靠笔记持久化。

代价是运行时探索比预检索慢，而且需要「有主见」的设计：没有引导，智能体会误用工具、追死胡同、错过关键信息，白白烧掉上下文。所以实践中常见**混合策略**：前置放少量高价值上下文保证速度（比如项目约定、README），再给出 `glob`、`grep` 这类原语让它自己找。

## 5. 长时程任务的三种手段

指望把窗口无限加大解决不了上下文污染。长程任务（大型代码迁移、跨小时的系统研究）要靠工程手段：

- **压缩整合（Compaction）**：接近上限时做高保真总结，用摘要重启一个新窗口。摘要保留架构决策、未解决缺陷、实现细节，丢掉重复的工具输出。调参顺序是先保召回、再提精度；最安全的轻量做法是只清理「深历史里的工具调用与结果」。
- **结构化笔记（Structured note-taking）**：以固定频率把关键信息写到上下文之外的持久存储，后续按需拉回。维护 TODO、进展笔记、结论与阻塞项索引，跨几十次工具调用与多轮上下文重置仍能保持连贯。第九章后半的 NoteTool 就是这个思路的落地。
- **子代理架构（Sub-agent）**：主代理做高层规划与综合，多个子代理在干净的窗口里各自深挖，只回传凝练摘要（常见 1000–2000 tokens）。庞杂的搜索上下文留在子代理内部，主代理只看结论——这是关注点分离在上下文层面的用法。

三者的适用面可以按任务形态分：要长对话接力用压缩，有阶段性里程碑用笔记，需要并行探索用子代理。

## 6. ContextBuilder 的四个目标

把这些原则做成可复用组件，就是 `ContextBuilder`。它要解决四件事：

- **统一入口**：把获取、选择、结构化、压缩抽象成一条流水线，避免每个 Agent 各写一遍拼上下文的模板代码。
- **稳定形态**：输出固定骨架，便于调试、A/B 与评估。骨架分六区：`[Role & Policies]`、`[Task]`、`[State]`、`[Evidence]`、`[Context]`、`[Output]`。
- **预算守护**：在 token 预算内尽量留住高价值信息，超限时兜底压缩。
- **最小规则**：不引入来源、优先级等一堆分类维度，只用「相关性 + 新近性」打分。规则越少越不容易互相打架。

## 7. 两个数据结构

候选信息统一成 `ContextPacket`，内容、时间、token 数、相关性四项：

```python
@dataclass
class ContextPacket:
    content: str
    timestamp: datetime
    token_count: int
    relevance_score: float = 0.5
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.metadata = self.metadata or {}
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))
```

配置集中到 `ContextConfig`，并在构造时校验约束：

```python
@dataclass
class ContextConfig:
    max_tokens: int = 3000
    reserve_ratio: float = 0.2          # 给系统指令预留
    min_relevance: float = 0.1
    enable_compression: bool = True
    recency_weight: float = 0.3
    relevance_weight: float = 0.7

    def __post_init__(self):
        assert abs(self.recency_weight + self.relevance_weight - 1.0) < 1e-6
```

`reserve_ratio` 是个容易被忽略但必要的参数：系统指令属于「必须进」的内容，如果不预先预留空间，它在贪心填充里可能被后来的高相似度片段挤掉——那不是优化，那是破坏契约。两个权重之和必须为 1 的断言，也让「分数可比」这件事有了保证。

## 8. GSSC 流水线

**Gather**：从五个来源汇集候选——系统指令（相关性固定 1.0，不参与评分）、记忆检索（top 10，重要性 ≥ 0.3）、RAG 检索（top 5，分数 ≥ 0.3）、最近 5 条对话、自定义包。每个外部源都包在 try-except 里：

```python
if self.rag_tool:
    try:
        rag_results = self.rag_tool.run({"action": "search", "query": user_query,
                                         "limit": 5, "min_score": 0.3})
        packets.extend(self._parse_rag_results(rag_results, user_query))
    except Exception as e:
        print(f"[WARNING] RAG 检索失败: {e}")
```

单源失败不能拖垮整轮构建——上下文构建是「尽可能好」的服务，不是「必须全有」的事务。这一点决定了它和业务接口在容错策略上的差别。

**Select**：先扣掉系统指令占用的 token，再对剩余包打分并贪心填充：

```python
system_tokens = sum(p.token_count for p in system_packets)
remaining_tokens = available_tokens - system_tokens

for packet in other_packets:
    if packet.relevance_score == 0.5:                       # 仍是默认值，说明没算过
        packet.relevance_score = self._calculate_relevance(packet.content, user_query)
    recency = self._calculate_recency(packet.timestamp)
    combined = (self.config.relevance_weight * packet.relevance_score +
                self.config.recency_weight * recency)
    if packet.relevance_score >= self.config.min_relevance:
        scored_packets.append((combined, packet))

scored_packets.sort(key=lambda x: x[0], reverse=True)
selected, current = system_packets.copy(), system_tokens
for score, packet in scored_packets:
    if current + packet.token_count <= available_tokens:
        selected.append(packet); current += packet.token_count
    else:
        break
```

相关性与新近性各自怎么算也很朴素：相关性用 Jaccard 词重叠，新近性用指数衰减（24 小时内保持高分，最低 0.1 保底）。这两处都标注了「生产环境可替换」——关键在于流水线的形状，而不在打分函数有多花哨。

**Structure**：按类型分组，套进固定骨架：

```python
if system_instructions:
    sections.append("[Role & Policies]\n" + "\n".join(system_instructions))
sections.append(f"[Task]\n{user_query}")
if evidence:
    sections.append("[Evidence]\n" + "\n---\n".join(evidence))
if context:
    sections.append("[Context]\n" + "\n".join(context))
sections.append("[Output]\n请基于以上信息,提供准确、有据的回答。")
return "\n\n".join(sections)
```

分区最实际的价值在**可调试**：回答错了，能一眼看出是 `[Evidence]` 里塞了不相关片段，还是 `[Context]` 根本没内容。散装拼接的上下文做不到这种定位。

**Compress**：按分区依次填充，装不下的部分截断并留标记：

```python
for section in sections:
    section_tokens = self._count_tokens(section)
    if current_total + section_tokens <= max_tokens:
        compressed_sections.append(section); current_total += section_tokens
    else:
        remaining = max_tokens - current_total
        if remaining > 50:                                  # 太少就别截了，没有意义
            compressed_sections.append(self._truncate_text(section, remaining) +
                                       "\n[... 内容已压缩 ...]")
        break
```

保留结构、截断内容，是压缩阶段的原则：宁可某一区只有半段，也不要整区消失，因为分区消失会让模型误判「没有这类信息」，而截断至少留了提示。

## 9. 取舍与已知局限

这套实现有意做了几处让步，用之前该知道：

- **相关性是词法而非语义**。Jaccard 词重叠对同义改写不敏感，`如何提升推理速度` 与 `加速推理的方法` 在它眼里几乎无关。要更好就换成向量相似度，代价是每轮都要嵌入计算。
- **压缩是截断而非摘要**。截断便宜、确定，但可能把结论切掉。要质量就用 LLM 摘要，代价是每轮多一次调用与不确定的延迟。
- **Token 是估算**。中文按字符、英文按单词 ×1.3，落在预算附近时误差可能几百 tokens。真的顶到模型上限时，这一点误差就够触发报错。
- **只保留最近 5 条历史**。短会话够用，长会话会丢早期约定——这正是下一篇要解决的问题。

到这里，上下文的「一次性拼装」有了方法。但长程任务里还有另一半问题：跨几十轮之后，怎么让智能体记得住已经做过什么、还剩什么没做。下一篇进入结构化笔记与文件系统访问——[9-2-结构化笔记与长程实战](/source/agent/从零构建智能体/9-2-结构化笔记与长程实战)。
