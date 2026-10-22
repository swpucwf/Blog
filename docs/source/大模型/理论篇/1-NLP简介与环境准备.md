# NLP 简介与环境准备

## 1. 自然语言处理是什么

**自然语言处理（Natural Language Processing, NLP）** 是人工智能领域的重要组成部分，它赋予计算机**理解、解释、生成人类语言**的能力，并基于这些能力**对文本数据进行决策**。NLP 旨在弥合人类交流的模糊性、情境性和复杂性与计算机精确、形式化的指令系统之间的鸿沟——比如机器需要理解「我今天很蓝」里的「蓝」并非颜色而是情绪。

工程实践中，NLP 通常概括为两类能力，共同构成机器与人类语言交互的完整闭环：

- **自然语言理解（NLU）**：负责「读懂」，输入语言、输出结构化信息。目标是从非结构化文本中提取意义，如识别意图、提取实体、分析情感。例：用户输入「帮我订一张明天去上海的机票」，NLU 需解析出「订票」意图以及「上海」「明天」等关键信息。
- **自然语言生成（NLG）**：负责「说出」，输入结构化信息、输出语言。将内部数据与决策转化为人类可读文本。例：天气 APP 根据 `{地点: "北京", 温度: "25℃", 天气: "晴"}` 生成「北京今日晴，气温 25℃，祝您有愉快的一天」。

现代大语言模型常以统一的「预测下一个 token」为训练目标，在同一模型内同时覆盖理解与生成等多类任务。

NLP 的技术体系可以看作层层递进的金字塔，处理过程由浅入深分为四个层次：

| 层次 | 做什么 | 典型例子 |
|---|---|---|
| 词法分析 | 处理基础单元——词：**分词**与**词性标注** | 「南京市长江大桥」应切分为 `南京市 / 长江大桥` |
| 句法分析 | 分析句子语法结构，形成「语法树」 | 分析「我爱北京天安门」的主谓宾结构 |
| 语义分析 | 理解真实含义，解决歧义：**词义消歧**、**关系抽取** | 确定「苹果」指水果还是公司；从「马云创立了阿里巴巴」提取 `创始人(马云, 阿里巴巴)` |
| 语用分析 | 在特定语境下理解语言意图，最具挑战的层次 | 理解「房间里真冷」可能是请求关窗而非陈述事实 |

## 2. 发展历程：从规则到智能

<p align="center">
  <img src="{{IMG:1_1_1.gif}}" width="80%" alt="NLP的发展历程" />
  <br />
  <em>图 1：NLP 的发展历程</em>
</p>

NLP 的发展经历了从符号主义到连接主义、从依赖专家规则到拥抱海量数据的深刻变革：

（1）**萌芽期（1950s）**：1950 年图灵发表论文《计算机器与智能》，提出的「**图灵测试**」成为衡量机器智能的终极愿景。1954 年 IBM 与 Georgetown University 做了俄英机器翻译公开演示（约 250 个词条、6 条规则、翻译 60 余句），证明了机器处理语言的可能性，同时也显著抬高了外界预期——当时新闻稿甚至出现「三到五年即可达到实用」的乐观判断，但语言的复杂性被严重低估。

（2）**规则时代（1960s-1980s）**：语言学家主导，**用逻辑规则描述语言**，乔姆斯基的形式语言理论影响深远。代表作：1966 年 `ELIZA` 通过关键词匹配和句式重组模拟心理治疗师；1970 年 `SHRDLU` 能在虚拟积木世界中理解并执行「把红色积木放到蓝色积木上面」这类指令。瓶颈在于语言的复杂性与歧义性远超预期，规则难以穷尽，系统无法处理规则之外的任何情况。

（3）**统计时代（1990s-2000s）**：范式重大转变，统计学派主张**让计算机从数据中学习规律**而非专家灌输规则。核心思想是将语言现象的合理性（句子通顺度、翻译准确度）转化为基于真实文本概率的数学计算，N-gram、隐马尔可夫模型（HMM）、条件随机场（CRF）成为主流。标志性成果是 2006 年上线的 **Google 翻译**，基于统计机器翻译实现了远超规则系统的翻译质量。

（4）**深度学习时代（2010s-至今）**：神经网络的复兴带来革命性突破，关键里程碑串成一条技术演进线：

- **2013 Word2Vec**：将词语表示为稠密向量，在向量空间中捕捉**分布式语义规律**。经典示例 $vector(\text{国王}) - vector(\text{男人}) + vector(\text{女人}) \approx vector(\text{女王})$——词向量在统计意义上学到了类比结构，但并不等同于人类层面的「理解」。
- **2014 注意力机制**：在神经机器翻译中被系统化提出，生成当前词时对输入序列不同位置分配不同权重，缓解信息压缩带来的性能瓶颈。
- **2017 Transformer**：将**自注意力**作为核心计算单元，摆脱 RNN 的顺序计算限制，每个 token 可同时利用全局上下文，并显著提升并行训练效率，成为后续大模型的基础架构。
- **2018 BERT**：通过掩码语言模型等预训练任务学习上下文语义表示，推动**预训练-微调**范式普及。
- **2020 GPT-3**：大规模参数与数据带来更强的**少样本/零样本**泛化能力，标志**大语言模型（LLM）**时代到来。
- **2022 ChatGPT**：通过**指令微调**与**人类反馈强化学习（RLHF）**等对齐方法，让大模型以符合人类使用习惯的对话方式呈现，彻底引爆全球 AI 浪潮。

> **补记**：这条时间线值得记住的不是模型名字，而是每次跃迁的动因——规则穷不完 → 交给数据；定长向量压不下长句 → 交给注意力；顺序计算没法并行 → 交给自注意力；单任务标注太贵 → 交给自监督预训练；alignment 靠人写回复太累 → 交给 RLHF。后面各篇展开的技术都落在这条因果链上。

## 3. NLP 的主要任务

| 任务 | 是什么 | 有什么用 | 举例 |
|---|---|---|---|
| **文本分类** | 给文本自动分配预定义标签 | 信息组织与过滤，最广泛的入门任务 | 情感分析、垃圾邮件过滤、新闻分类 |
| **命名实体识别（NER）** | 找出并分类人名、地名、组织、时间等实体 | 非结构化文本转结构化信息，信息抽取关键一步 | 「马云」「1999年」「杭州」「阿里巴巴」 |
| **关系抽取** | 在实体基础上判断实体间语义关系 | 构建知识图谱 | 创始人（马云, 阿里巴巴） |
| **机器翻译** | 自动将一种自然语言翻译成另一种 | 消除语言隔阂 | Attention is all you need → 注意力机制就是你所需要的一切 |
| **文本摘要** | 长文本压缩为保留核心信息的简短摘要 | 快速获取要点 | 新闻摘要、会议纪要 |
| **问答系统** | 针对问题给出精准简洁的答案 | 智能客服/搜索的核心能力 | 「珠穆朗玛峰多高？→ 8848.86米」 |
| **文本生成** | 根据输入自动生成文本 | 内容创作、报告自动化 | AI 写作、代码生成 |
| **对话系统** | 模拟多轮对话，理解上下文并恰当回应 | 连续对话、记忆上下文 | 智能助理、情感陪伴、客服 |

## 4. NLP 面临的主要挑战

（1）**语言、知识与推理的挑战**：**语言歧义性**是首要经典难题——既有「朝阳」因读音不同指代迥异的**词法歧义**，也有「咬死了猎人的狗」因句法结构不同而产生相反解读的**结构歧义**。机器还缺乏人类与生俱来的**常识与世界知识**，在需要隐性背景知识的场景容易「露怯」；深层**推理能力**（复杂逻辑、因果分析）仍显吃力；反讽、幽默、成语、网络梗等高度依赖**语境与文化**的语言现象最难捕捉弦外之音。

（2）**技术、数据与伦理的挑战**：生成式机制带来了新的副作用——**模型幻觉**以极度自信的语气编造事实，「一本正经地胡说八道」严重影响新闻、法律、科研等严谨场景的可靠性。数据层面存在双重困境：绝大多数语言属于**低资源语言**难以享受技术红利；训练数据中潜藏的性别歧视、种族偏见等**数据偏差**会被模型内化甚至放大。此外模型规模指数级增长推高**计算成本**（算力集群、能耗碳排放构成经济门槛）；「黑箱」性质带来**可解释性与安全性**风险——医疗诊断、金融风控等高风险领域难以接受无法解释的决策，同时还需防止模型被恶意利用生成暴力、色情或虚假信息。

## 5. 环境准备

后续实战均基于 Python 生态，环境主线是 **Anaconda + conda 虚拟环境 + PyTorch**。

（1）**为什么选 Anaconda**：面向数据科学的 Python 发行版，一个安装包同时包含解释器、常用科学计算库（NumPy/Pandas/Matplotlib 等）与环境管理工具 conda，支持为不同项目创建彼此隔离的虚拟环境，Windows/macOS/Linux 命令行体验基本一致。已熟悉 `pyenv`、`venv` 或 `uv` 的也可以继续沿用。

（2）**发行版选择**：官网提供 **Anaconda Distribution**（预装 Python、Conda 与 1500+ 数据科学包，体积约 1 GB，开箱即用）和 **Miniconda**（仅含 Python 与 Conda，轻量但需手动装库）。首次接触数据科学环境建议前者；macOS / Linux 推荐 Miniconda 脚本安装：

```bash
curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/miniconda.sh
```

Windows 图形化安装的两个要点：安装路径务必全英文无空格（不建议默认 C 盘）；勾选 `Register Anaconda3 as default Python`，并在系统环境变量 `Path` 中补上 `<安装目录>`、`<安装目录>\Scripts`、`<安装目录>\Library\bin` 三条路径，最后 `conda --version` 验证。

<div align="center">
  <img src="{{IMG:1_2_1.png}}" width="80%" alt="Anaconda 官网下载页" />
  <p>图 2：Anaconda 官网下载页</p>
</div>

<div align="center">
  <img src="{{IMG:1_2_2.png}}" width="80%" alt="Anaconda 安装向导" />
  <p>图 3：Anaconda 安装向导</p>
</div>

<div align="center">
  <img src="{{IMG:1_2_3.png}}" width="80%" alt="Anaconda 高级安装选项" />
  <p>图 4：Anaconda 高级安装选项</p>
</div>

<div align="center">
  <img src="{{IMG:1_2_4.png}}" width="80%" alt="Windows 系统环境变量配置" />
  <p>图 5：Windows 系统环境变量配置</p>
</div>

<div align="center">
  <img src="{{IMG:1_2_5.png}}" width="80%" alt="编辑环境变量" />
  <p>图 6：编辑环境变量</p>
</div>

<div align="center">
  <img src="{{IMG:1_2_6.png}}" width="80%" alt="添加 Path 路径" />
  <p>图 7：添加 Path 路径</p>
</div>

（3）**配置国内镜像（可选但推荐）**：conda 官方源在国内可能较慢，可切清华镜像：

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2/
conda config --set show_channel_urls yes
```

遇到软件包冲突或安装失败时，可根据提示临时切回官方源重试。

（4）**创建专用虚拟环境并安装依赖**：为项目单独建环境避免互相污染：

```bash
conda create -n base-llm python=3.10   # 项目推荐 Python 3.10
conda activate base-llm                # 激活后前缀出现 (base-llm)
pip install numpy pandas matplotlib scikit-learn jupyter
```

PyTorch 按设备二选一：纯 CPU 直接安装 CPU 版：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

有 GPU 则先用 `nvidia-smi` 查看右上角 `CUDA Version: 12.x` 确认支持的最高版本，再到 [PyTorch 官网](https://pytorch.org/get-started/locally/)选择与驱动匹配的指令安装。退出环境用 `conda deactivate`。

> **补记**：安装各步均为向导式普通操作，上方截图按「官网下载→安装向导→环境变量配置」的顺序展示，照文字要点走即可。

## 参考文献

- Manning, C. (2024). *CS224N: Natural Language Processing with Deep Learning*. Stanford University
- Turing, A. M. (1950). *Computing Machinery and Intelligence*. Mind, 59(236), 433-460
- Mikolov, T. et al. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv:1301.3781
- Bahdanau, D. et al. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*. arXiv:1409.0473
- Vaswani, A. et al. (2017). *Attention Is All You Need*. arXiv:1706.03762
- Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805
- Brown, T. et al. (2020). *Language Models are Few-Shot Learners*. arXiv:2005.14165

## 写在最后

概念和时间线之外，NLP 让机器「懂词」的第一步是把词变成数字——下一篇[文本表示与词向量](2-文本表示与词向量)从分词讲起，记录词向量与 Word2Vec 如何把语言的意义装进向量空间。
