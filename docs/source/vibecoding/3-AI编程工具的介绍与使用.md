## 1. 如何搭配 AI 主动学习

💡 最重要的学习原则：有问题，先问 AI

从现在开始，请把 AI 当成你的**第一求助对象**。无论是概念没听懂、工具不会装、命令不会写、程序报错，还是不知道下一步该怎么做，**先问 AI，再自己动手尝试**。

凡是你原本会问老师、同学或群友的问题，都应该优先问 AI。它响应更快，可以随时追问，也能结合你的报错信息和项目上下文立即给出下一步建议。不要等到卡住很久才想起 AI，也不要只在“让 AI 写代码”时才使用它。

**真正的 AI 原生学习，不只是用 AI 完成任务，而是让 AI 参与学习的全过程：遇到问题就问，得到答案就试，出现新问题再继续问。**

### 1.1  AI 原生学习闭环

AI 主动学习不是“把任务丢给 AI，自己等结果”，而是让 AI 帮你缩短等待答案的时间，把更多精力用在**动手、验证和理解**上。下面这个闭环:

![image-20261021005404795](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/image-20261021005404795.png)

### 1.2 常见问题，可以直接这样问

| 你遇到的情况            | 可以直接对 AI 说                                             |
| :---------------------- | :----------------------------------------------------------- |
| **概念看不懂**          | “我没有编程基础，请用生活中的类比解释什么是 API，再给我一个最小示例。” |
| **环境不会安装**        | “我使用 macOS，想运行这个项目。请先检查需要哪些环境，再一次只带我完成一步，并解释这一步的作用。” |
| **命令运行报错**        | “这是我执行的命令和完整报错。请先判断最可能的原因，再给我最小修改方案；如果信息不够，请告诉我还要提供什么。” |
| **不知道下一步做什么**  | “我的目标是做一个贪吃蛇，目前已经能显示游戏画面。请告诉我接下来最小的一步，以及完成标准。” |
| **AI 写了代码但看不懂** | “先不要继续改代码。请按模块解释刚才的修改，每一部分解决了什么问题？” |
| **有多个方案不会选**    | “请根据我是初学者、希望快速做出可运行版本这两个条件，对比方案 A 和 B，并明确推荐一个。” |
| **结果和预期不一样**    | “我希望页面是这样的：……，现在实际是这样的：……。请比较差异，先只修复最明显的一处。” |

一个通用的提问模板

**我的目标是……；我目前做到……；我尝试了……；实际结果或完整报错是……；请先解释原因，再告诉我下一步最小的操作。**

信息越具体，AI 越容易给出真正适合你的回答。第一次回答没解决问题也很正常，把新结果继续发给它，这本身就是 AI 原生学习的一部分。





## 1. 写代码需要什么环境和工具

### 1.1 思维转变：遇到问题，先问 AI

在开始介绍各种环境和工具之前，首先提示你需要**转变你的思维习惯**。

在传统的编程学习中，如果你要安装 Python、配置 Conda、或者解决 npm 安装失败的问题，你通常会打开搜索引擎，找到一篇教程，然后按照步骤一步步操作。如果中间报错了，你可能需要再搜报错信息，反复尝试。

错！❌

在 AI 时代，特别是在使用 AI IDE 时，请记住一个核心原则：**任何操作，都可以先问一遍 AI，甚至让它直接帮你做。**

- **不知道怎么安装环境？** 直接在侧边栏问 AI：“我想写 Python，帮我检查一下有没有安装 Python，如果没有请帮我安装。”
- **网络卡住了？** 如果安装依赖包时一直转圈或报错，直接把错误丢给 AI：“下载失败了，是不是网络问题？能不能帮我换个国内的镜像源？”
- **命令记不住？** 不需要死记硬背 Git 命令或 Conda 命令，直接告诉 AI：“帮我创建一个新的虚拟环境，名字叫 demo。”

### 1.2 为什么需要环境和工具

从"试着写几行代码"到"做可长期维护的项目"，对环境和工具的要求完全不同。

理论上用系统自带的记事本也能写代码，但问题很快会出现：

- **代码全是黑色文字**，关键字、字符串、注释混在一起，很难一眼看出结构
- **没有智能提示**，每个单词都要完整手敲，拼错一个字母就要反复检查
- **文件多了就乱套**，十几个文件来回切换，经常找不到要改的那一行在哪
- **出错只能猜**，程序崩了不知道哪里出问题，只能一行行打印日志试错

因此，你需要一款 IDE（集成开发环境）。它会把代码用不同颜色显示、输入时自动提示、文件按项目整理、还能一步步追踪错误，让开发更高效、更少出错。

## 2. 什么是 IDE，为什么需要 IDE

预习提示

如果你还不熟悉 IDE 是什么、各个界面元素有什么作用，建议先阅读 [IDE 简介](https://datawhalechina.github.io/easy-vibe/zh-cn/appendix/2-development-tools/ide-basics.html) 进行预习，了解 IDE 的基本概念和常见功能。

在早期编程时代，我们只需要简单文本编辑器和语言处理器即可。但随着项目复杂度增加，开发者迫切需要一种能高效管理文件、支持语法高亮和调试的工具，于是集成开发环境（IDE）应运而生。

你可以把 IDE 理解成专门用来“编辑、管理、运行和调试”代码的程序。早期的 IDE 外观非常“原始”，几乎完全通过键盘操作。

终端界面（Terminal） 图片来源：https://en.wikipedia.org/wiki/File:Emacs-screenshot.png

知名且功能成熟的“内置 IDE”如 `Vim`，常用于服务器远程操作。

![img](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/image3.DVAC6UMM.webp)

为了更高效，我们需要支持鼠标操作的现代 IDE，通常包含：

- **源代码编辑器**：语法高亮、自动补全。
- **构建与运行工具**：内置编译器/解释器。
- **调试器**：断点调试、变量查看。

现代 IDE 往往还内置 Git 等工具。最流行的是微软的 **[Visual Studio Code (VS Code)](https://code.visualstudio.com/)**，它轻量且可扩展。虽然也有 JetBrains 全家桶等专业 IDE，但 VS Code 对初学者最友好。

VS Code 的核心理念是“一切皆插件”。它通过插件机制支持各种语言，安装 Python 插件就是 Python IDE，安装 C++ 插件就是 C++ IDE。不装插件时，它只是个高级文本编辑器。

![img](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/image5.DeXifyNb.webp)

总之，IDE 是一套帮助开发者高效写代码和运行程序的工具集。

更具体的详细内容解释，请查看[附录中的 虚拟 IDE 可视化 IDE 原理部分](https://datawhalechina.github.io/easy-vibe/zh-cn/appendix/2-development-tools/ide-basics.html)。

## 3. AI IDE 和普通 IDE 有什么不同

普通 IDE（比如原版 VS Code）本质上是一套“工具箱”：
可以打开项目、写代码、运行和调试，也能装插件，但前提是你需要自己知道要做什么、怎么做：

- 报错时，自己读提示、自己查哪一行有问题；
- 想加新页面或新接口，自己找对应文件、自己写代码；
- 想配置环境或打包，自己查文档、按步骤操作。

但在 AI IDE 里，你可以直接使用大语言模型帮助你进行编码和修改文件：

- 直接说“做一个登录页”，它先生成基础代码结构；
- 把报错信息和相关代码丢给它，让它先分析原因并给出修改建议；
- 在你确认后，让它自动新建文件、批量改代码，处理跨文件的体力活。

例如，你可以选中一段代码，让它“重构一下”或“加注释”；也可以在侧栏里问“这个项目是怎么设计的？”，通过 `@文件名` 或 `@整个项目` 指定参考范围，用一句话自动完成新建文件、写代码和运行的繁琐操作。

在最新版 VS Code 中，已经内置了一个大语言模型助手。你可以直接针对整个代码仓库、某个文件，甚至某个函数与模型对话。你也可以像之前在 Web 端使用自动写代码工具一样，将需求以提示词的形式发给内置的编码 Agent，让它自动帮你实现所需功能、创建文件、修改代码、配置环境等。

可以下载安装 VS Code，在点击右上角的侧边栏入口，打开 AI 功能区域，体验这些能力。

不过，VS Code 并不是 AI 能力最强的 IDE。对于需要大量 AI 辅助编码的场景，我们往往希望使用“更聪明、效率更高”的工具——好的 AI IDE 能显著节省写代码和改 Bug 的时间。下面我们会介绍几款目前比较流行的 AI IDE，你可以根据个人喜好选择任意一款 AI IDE 使用。

由于 VS Code 是开源的（任何人都可以下载源码并自行编译），目前市面上绝大多数 AI IDE 都是在 VS Code 基础上二次开发而来。所以你不必担心要“学习很多种 IDE”——**只要你熟悉了 VS Code 的基本用法**，迁移到这些 AI IDE 并不需要重新学习。

一般而言，对于不同 AI IDE 之间的差异，主要集中在四个方面：价格；可使用的模型种类（部分高级模型在某些地区可能受限）；Agent 的能力（在协助写代码时的智能程度和执行能力）；以及运行速度与性能。你可以根据实际测试效果进行选用，适合自己的才是最好的。

> 典型的 AI IDE 一般具备以下核心能力：
>
> - 智能代码生成与补全：在传统 IDE 中，我们通常是输入几个字符来补全变量名或函数名；在现代 AI IDE 中，你可以写几行伪代码或者简单说明需求，让 IDE 自动补全完整的逻辑，甚至根据指令直接生成一大段甚至整块代码。
> - 代码理解与问答：IDE 能够理解并回答关于某段代码、某个文件，甚至整个工程目录结构的问题。
> - 代码重构与优化：IDE 可以根据你的意图，重写或优化指定代码片段的实现逻辑。
> - 自动生成测试：IDE 可以自动生成针对不同函数和模块的测试代码，方便你进行有针对性的测试。
> - Agent 式任务执行：智能 Agent 可以自动生成、打包、安装、运行和修改代码，在很多任务上可以部分替代初级软件工程师的工作。

![image-20261021011054248](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/image-20261021011054248.png)

### 4.1 准备工作：安装并了解 Trae

#### 4.1.1 什么是 Trae

Trae 的全称可以理解为 “The Real AI Engineer”，是一款由字节跳动开发的自适应 AI 集成开发环境（IDE）。它是在流行的 VS Code 基础之上构建的，这意味着，如果你之前已经习惯了 VS Code，那么在使用 Trae 时，无论是界面布局还是基础操作都会感到非常熟悉、舒适。

Trae 的核心目标是成为开发者的“智能编程伙伴”。通过深度集成 AI 能力，它可以自动处理大量重复性工作，为你提供更直观、更高效的开发体验。它并不仅仅是一个“代码补全工具”，而是希望贯穿整个开发工作流，从创建项目、编写代码、调试、测试到部署都提供帮助。

#### 4.1.2 安装 Trae

Trae 分为国际版和中国版。国际版需要能够访问海外网络，但可以使用 GPT-5 等最新的海外模型；中国版则主要支持国内最新的大模型，例如 GLM、Qwen、Kimi 等。

国际版下载地址：https://www.trae.ai/ 中国版下载地址：https://www.trae.cn/

##### Trae 定价与使用方式

💡 版本选择提示（零基础推荐 CN 版）

- **零基础入门强烈推荐下载中国版（CN 版，trae.cn）**，目前使用体验更好，且基础功能免费，无需海外网络
- 如果你需要使用 GPT-5 等海外模型，且网络条件允许，可以选择国际版
- 如果已有第三方模型的 API Key，接入第三方模型可以灵活控制成本

> 💡 **目前推荐使用 OpenRouter 免费模型进行测试**
>
> 截至教程编写时间（2026-02-12），目前仍可免费试用 StepFun 的模型。具体可以参考下面 4.2 章节部分的模型接入方式，接入 `stepfun/step-3.5-flash:free`。

关于 Trae 的费用和使用方式，有以下几个选项可供选择：

- **国内版 CN 版（强烈推荐）**：基础使用免费，目前整体使用效果优于国际版，非常适合零基础入门。由于用户较多可能偶尔需要排队等待。
- **国际版**：订阅价格大约为每月 3 美元左右，可以访问 GPT-5 等海外模型，但需要能够访问海外网络。
- **第三方模型接入**：如果你已经有国内大模型的 Token API（如 DeepSeek、通义千问、Kimi 等），可以通过 Trae 的第三方模型配置功能将这些 API 接入使用。各大云服务厂商（如阿里云、腾讯云、百度云等）通常提供 Coding Plan 订阅计划，购买后可以以更优惠的价格使用其大模型 API。这样你可以自由选择自己喜欢的模型，同时控制使用成本。

建议初学者从国内 CN 版免费版开始体验（下载地址：https://www.trae.cn/ ），目前 CN 版的使用效果更好且完全免费。如果遇到排队问题或需要更稳定的服务，可以考虑接入第三方模型并购买对应云厂商的 Coding Plan 计划。

#### 4.1.3 Trae 界面简介

从界面形态上看，Trae 与我们日常使用的 VS Code 高度相似：同样是左侧资源管理器、中间编辑区、右侧扩展面板的经典三栏布局。

![img](https://datawhalechina.github.io/easy-vibe/assets/image17.B7plH7fr.webp)

右侧的侧边栏就是 Copilot 交互窗口，也可以理解为 Agent 窗口。如果你暂时看不到它，可以点击 Trae 右上角的侧边栏图标将其打开。

![img](https://datawhalechina.github.io/easy-vibe/assets/image18.C3-unbGE.webp)

打开侧边栏之后，你会看到一个 `Builder` 选项，这就是 Agent 模式。简单理解，它相当于 z.ai 的“本地版”，可以帮你操作本机环境，安装运行环境、打开网页等。

![img](https://datawhalechina.github.io/easy-vibe/assets/image19.BkBudVcv.webp)

点击 “Builder” 后，你会看到 “Chat” 模式和 “Builder with MCP” 模式：

- **Chat 模式**：主要用于和当前文件夹里的代码对话，或者当作普通聊天模型来使用。（你可以通过左上角的 “File” 菜单打开一个文件夹，在这个文件夹中进行编辑操作。在这种情况下，Builder 创建或修改的文件都只会发生在这个文件夹内部。）
- **Builder with MCP 模式**：为 Agent 提供了更多可用工具（例如把语言模型和其他软件联通起来、查询天气等）。你可以简单理解为：MCP 能让语言模型更方便地调用各种外部工具。

![img](https://datawhalechina.github.io/easy-vibe/assets/image20.DylAzBb1.png)

在下面的区域，你还会看到模型选择选项，点击即可修改当前使用的大模型。在中国版中，你可以选择使用 Kimi k2 或 GLM 等国内模型；如果你使用的是国际版 Trae，还可以选择 ChatGPT 或 Claude 等海外模型。不过，由于国内大模型发展非常快，Kimi、Qwen、GLM 等在很多任务上的实际体验已经接近 Claude 3.5 或 3.7，对日常开发来说已经完全够用，这里不强制要求使用国际版或者国内版进行操作。

**需要注意的是，这里不推荐使用 Auto 模式（自动选择模型），如果是国际版，我们推荐使用 Gemini 或者 GPT 模型， 如果是国内版，我们推荐你尝试 Kimi k2 或 Minimax、GLM 等国内模型，** 不同模型有不同的使用场景，没有教条式的一定谁比谁好在哪，你可以在不同任务遇到困难无法解决时换一个模型，通过多次测试得到属于自己的最佳实验结果。

![img](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/image21.DZCpe4lD.webp)

以上就是对 Trae 的一个简单介绍。接下来，我们可以回顾一下之前在 z.ai 中做过的操作，并尝试在 Trae 中做同样的事情。

### 4.2 第一步：新建空文件夹并用 AI IDE 打开

在正式动手之前，我们首先需要准备一个干净的项目工作目录。 以本小节示例为例，可以在本地新建一个名为 snake-game-react 的空文件夹。

随后，打开已安装好的 AI IDE，在启动界面选择打开文件夹或Open Folder，将该空文件夹作为项目根目录导入；也可以直接将文件夹拖入 IDE 窗口完成打开。此时，左侧资源管理器中不会出现任何代码文件，表示我们正从一个完全空白的项目状态开始。

<details class="details custom-block" style="box-sizing: border-box; --vp-code-copy-copied-text-content: &quot;已复制&quot;; border: 1px solid rgba(0, 0, 0, 0); border-radius: 8px; padding: 16px 16px 8px; line-height: 24px; font-size: 14px; color: rgb(60, 60, 67); margin: 16px 0px; background-color: rgba(142, 150, 170, 0.14); font-family: -apple-system, BlinkMacSystemFont, &quot;SF Pro Text&quot;, &quot;Segoe UI&quot;, Roboto, Helvetica, Arial, sans-serif, &quot;Apple Color Emoji&quot;, &quot;Segoe UI Emoji&quot;, &quot;Segoe UI Symbol&quot;; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; white-space: normal; text-decoration-thickness: initial; text-decoration-style: initial; text-decoration-color: initial;"><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;">本节将介绍如何接入云服务厂商的 API 或 Coding Plan，以获得更稳定、更频繁的模型调用。结尾部分会给出 Trae 中接入的截图。</p><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;"><strong style="box-sizing: border-box; font-weight: 600; --vp-code-copy-copied-text-content: &quot;已复制&quot;;">什么是 Coding Plan</strong></p><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;">Coding Plan 是各大云服务厂商推出的订阅计划，购买后你可以在一定时期内<strong style="box-sizing: border-box; font-weight: 600; --vp-code-copy-copied-text-content: &quot;已复制&quot;;">无限制或高频次地使用</strong>该厂商的大模型 API。相比于按 Token 计费的方式，Coding Plan 更像是"包月套餐"——你付一笔固定的费用，就能放心大胆地一直用，不用担心每次调用都要计费。</p><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;"><strong style="box-sizing: border-box; font-weight: 600; --vp-code-copy-copied-text-content: &quot;已复制&quot;;">为什么需要购买 Coding Plan</strong></p><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;">你可能会问：既然可以直接使用 API 调用大模型，为什么还要购买 Coding Plan 呢？主要有以下原因：<strong style="box-sizing: border-box; font-weight: 600; --vp-code-copy-copied-text-content: &quot;已复制&quot;;">可以一直用</strong>：Coding Plan 最核心的优势就是你可以随时、频繁地调用大模型，不用担心用多了费用爆炸，也不需要频繁看计费表</p><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;"><strong style="box-sizing: border-box; font-weight: 600; --vp-code-copy-copied-text-content: &quot;已复制&quot;;">推荐的国内云服务 Coding Plan</strong></p><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;">以下是国内主流云服务厂商提供的 Coding Plan 推荐选项：</p><ul style="box-sizing: border-box; list-style: disc; margin: 16px 0px; padding: 0px 0px 0px 1.25rem; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 1.65 !important;"><li style="box-sizing: border-box; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; margin: 4px 0px;">智谱 AI（BigModel Plan）：<a href="https://bigmodel.cn/glm-coding" target="_blank" rel="noreferrer" style="box-sizing: border-box; touch-action: manipulation; color: rgb(0, 113, 227); text-decoration: none !important; --vp-code-copy-copied-text-content: &quot;已复制&quot;; font-weight: 500; text-underline-offset: 2px; transition: border-bottom-color 0.2s ease-in-out; border-bottom: 1px solid transparent;">https://bigmodel.cn/glm-coding</a></li><li style="box-sizing: border-box; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; margin: 8px 0px 4px;">火山引擎（字节云 AI Plan）：<a href="https://www.volcengine.com/activity/codingplan" target="_blank" rel="noreferrer" style="box-sizing: border-box; touch-action: manipulation; color: rgb(0, 113, 227); text-decoration: none !important; --vp-code-copy-copied-text-content: &quot;已复制&quot;; font-weight: 500; text-underline-offset: 2px; transition: border-bottom-color 0.2s ease-in-out; border-bottom: 1px solid transparent;">https://www.volcengine.com/activity/codingplan</a></li></ul><blockquote style="box-sizing: border-box; margin: 16px 0px !important; --vp-code-copy-copied-text-content: &quot;已复制&quot;; border-left: 2px solid rgb(226, 226, 227); padding: 0px 0px 0px 16px !important; transition: border-color 0.5s; color: rgb(103, 103, 108); font-size: 0.9em !important; line-height: 1.65 !important;"><p style="box-sizing: border-box; margin: 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px; font-size: 14px; transition: color 0.5s; color: inherit;">💡 <strong style="box-sizing: border-box; font-weight: 600; --vp-code-copy-copied-text-content: &quot;已复制&quot;;">也可以直接接入大模型 API</strong> 除了 Coding Plan，你也可以直接通过 Add Model 接入各大模型的 API。你可以参考下文接入 OpenRouter StepFun 免费 API 的方式，将 API 接入 Trae 进行使用。经测试可满足基本的编程需求。 如果需要充值，建议先充值 10 元感受一下能用多久，比如 DeepSeek 等性价比较高的模型</p></blockquote><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;"><strong style="box-sizing: border-box; font-weight: 600; --vp-code-copy-copied-text-content: &quot;已复制&quot;;">如何接入 Coding Plan</strong></p><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;">接入 Coding Plan 的步骤非常简单，只需几分钟即可完成：</p><ol style="box-sizing: border-box; list-style: decimal; margin: 16px 0px; padding: 0px 0px 0px 1.25rem; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 1.65 !important;"><li style="box-sizing: border-box; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; margin: 4px 0px;">访问你选择的云服务厂商官网（如智谱 AI：<a href="https://bigmodel.cn/glm-coding" target="_blank" rel="noreferrer" style="box-sizing: border-box; touch-action: manipulation; color: rgb(0, 113, 227); text-decoration: none !important; --vp-code-copy-copied-text-content: &quot;已复制&quot;; font-weight: 500; text-underline-offset: 2px; transition: border-bottom-color 0.2s ease-in-out; border-bottom: 1px solid transparent;">https://bigmodel.cn/glm-coding</a> ，火山引擎：<a href="https://www.volcengine.com/activity/codingplan%EF%BC%89" target="_blank" rel="noreferrer" style="box-sizing: border-box; touch-action: manipulation; color: rgb(0, 113, 227); text-decoration: none !important; --vp-code-copy-copied-text-content: &quot;已复制&quot;; font-weight: 500; text-underline-offset: 2px; transition: border-bottom-color 0.2s ease-in-out; border-bottom: 1px solid transparent;">https://www.volcengine.com/activity/codingplan）</a></li><li style="box-sizing: border-box; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; margin: 8px 0px 4px;">注册账号并登录</li><li style="box-sizing: border-box; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; margin: 8px 0px 4px;">找到 "定价" 或 "Coding Plan" 页面</li><li style="box-sizing: border-box; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; margin: 8px 0px 4px;">选择适合你的套餐并完成支付</li><li style="box-sizing: border-box; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; margin: 8px 0px 4px;">支付成功后，你会获得一个 API Key 或 Plan ID</li></ol><div class="tip custom-block" style="box-sizing: border-box; --vp-code-copy-copied-text-content: &quot;已复制&quot;; border: 1px solid rgba(0, 0, 0, 0); border-radius: 8px; padding: 16px 16px 8px; line-height: 24px; font-size: 14px; color: rgb(60, 60, 67); background-color: rgba(0, 113, 227, 0.14); margin: 16px 0px;"><p class="custom-block-title" style="box-sizing: border-box; margin: 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; font-weight: 600; line-height: 24px;">🎯 自定义模型推荐</p><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;">在 Trae 中接入自定义模型时，我们<strong style="box-sizing: border-box; font-weight: 600; --vp-code-copy-copied-text-content: &quot;已复制&quot;;">默认推荐使用 OpenRouter 方案</strong>。OpenRouter 提供了统一的 API 接口，可以方便地接入多种大语言模型。</p><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;"><strong style="box-sizing: border-box; font-weight: 600; --vp-code-copy-copied-text-content: &quot;已复制&quot;;">截至 2026 年 2 月 12 日，你还可以使用 StepFun 的免费 API：</strong></p><ul style="box-sizing: border-box; list-style: disc; margin: 16px 0px; padding: 0px 0px 0px 1.25rem; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 1.65 !important;"><li style="box-sizing: border-box; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; margin: 4px 0px;"><strong style="box-sizing: border-box; font-weight: 600; --vp-code-copy-copied-text-content: &quot;已复制&quot;;"><code style="box-sizing: border-box; font-family: ui-monospace, Menlo, Monaco, Consolas, &quot;Liberation Mono&quot;, &quot;Courier New&quot;, monospace; --vp-code-copy-copied-text-content: &quot;已复制&quot;; font-size: 12.25px; color: rgb(0, 113, 227); border-radius: 4px; padding: 3px 6px; background-color: rgba(142, 150, 170, 0.14); transition: color 0.25s, background-color 0.5s;">stepfun/step-3.5-flash:free</code></strong>：StepFun（阶跃星辰）提供的免费模型，同样支持在 Trae 中直接接入使用。</li></ul><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;"><strong style="box-sizing: border-box; font-weight: 600; --vp-code-copy-copied-text-content: &quot;已复制&quot;;">其它免费模型：</strong></p><ul style="box-sizing: border-box; list-style: disc; margin: 16px 0px; padding: 0px 0px 0px 1.25rem; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 1.65 !important;"><li style="box-sizing: border-box; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; margin: 4px 0px;"><strong style="box-sizing: border-box; font-weight: 600; --vp-code-copy-copied-text-content: &quot;已复制&quot;;"><code style="box-sizing: border-box; font-family: ui-monospace, Menlo, Monaco, Consolas, &quot;Liberation Mono&quot;, &quot;Courier New&quot;, monospace; --vp-code-copy-copied-text-content: &quot;已复制&quot;; font-size: 12.25px; color: rgb(0, 113, 227); border-radius: 4px; padding: 3px 6px; background-color: rgba(142, 150, 170, 0.14); transition: color 0.25s, background-color 0.5s;">openrouter/free</code></strong>：这是一个默认使用免费 LLM API 的模型选项，可以直接在 Trae 的 Custom Model 接入中使用（直接写进模型 ID 即可），无需付费即可体验 AI 编程功能。</li></ul><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;">这些免费选项非常适合初学者体验，在正式投入生产环境前，可以先通过这些免费方案熟悉 AI IDE 的工作流程。</p><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;"><strong style="box-sizing: border-box; font-weight: 600; --vp-code-copy-copied-text-content: &quot;已复制&quot;;">可选：接入大模型调用 API（以 DeepSeek 为例）</strong></p><ol style="box-sizing: border-box; list-style: decimal; margin: 16px 0px; padding: 0px 0px 0px 1.25rem; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 1.65 !important;"><li style="box-sizing: border-box; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; margin: 4px 0px;">访问 DeepSeek 平台：<a href="https://platform.deepseek.com/usage" target="_blank" rel="noreferrer" style="box-sizing: border-box; touch-action: manipulation; color: rgb(0, 113, 227); text-decoration: none !important; --vp-code-copy-copied-text-content: &quot;已复制&quot;; font-weight: 500; text-underline-offset: 2px; transition: border-bottom-color 0.2s ease-in-out; border-bottom: 1px solid transparent;">https://platform.deepseek.com/usage</a></li><li style="box-sizing: border-box; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; margin: 8px 0px 4px;">注册账号并登录</li><li style="box-sizing: border-box; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; margin: 8px 0px 4px;">在充值页面购买 10 元的 Token 包</li><li style="box-sizing: border-box; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; margin: 8px 0px 4px;">充值成功后，在 API Keys 页面创建并复制 API Key</li><li style="box-sizing: border-box; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; margin: 8px 0px 4px;">在 Trae 中点击 <strong style="box-sizing: border-box; font-weight: 600; --vp-code-copy-copied-text-content: &quot;已复制&quot;;">"Add Model"</strong>，找到 DeepSeek，选择对应模型，输入 API Key 即可使用</li></ol><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;">通过下列界面，你可以成功添加（注意看选择模型的选项后【一定要滑动到最底部】，下面有一个“自定义模型“，点击后才可以输入模型 ID，此时可以输入上述推荐的模型 ID 如 <code style="box-sizing: border-box; font-family: ui-monospace, Menlo, Monaco, Consolas, &quot;Liberation Mono&quot;, &quot;Courier New&quot;, monospace; --vp-code-copy-copied-text-content: &quot;已复制&quot;; font-size: 12.25px; color: rgb(0, 113, 227); border-radius: 4px; padding: 3px 6px; background-color: rgba(142, 150, 170, 0.14); transition: color 0.25s, background-color 0.5s;">stepfun/step-3.5-flash:free</code> 直接写入即可，同时点击下方的获取 Key 前往官网获得对应的 API Key 写入即可正常使用。）</p><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;"><img src="https://datawhalechina.github.io/easy-vibe/assets/index-2026-02-12-14-14-51.Cl544t7e.webp" alt="" decoding="async" loading="lazy" style="box-sizing: border-box; display: block; max-width: 100%; height: auto; --vp-code-copy-copied-text-content: &quot;已复制&quot;; margin-left: auto; margin-right: auto;"></p><p style="box-sizing: border-box; margin: 8px 0px; overflow-wrap: break-word; --vp-code-copy-copied-text-content: &quot;已复制&quot;; line-height: 24px;"><img src="https://datawhalechina.github.io/easy-vibe/assets/index-2026-02-12-14-15-29.D8Y30RMi.webp" alt="" decoding="async" loading="lazy" style="box-sizing: border-box; display: block; max-width: 100%; height: auto; --vp-code-copy-copied-text-content: &quot;已复制&quot;; margin-left: auto; margin-right: auto;"></p></div></details>

### 4.3 第二步：在侧边栏聊天，让 AI 用 React 设计贪吃蛇游戏

接下来，打开 AI 聊天侧边栏：一般是按 `Ctrl+L` 或点击右侧聊天图标。然后在聊天里输入一个足够清晰的提示：

> 请你用 React 架构实现贪吃蛇游戏，包含键盘控制、吃到食物变长加分、撞墙或撞到自己时显示“游戏结束”并支持重新开始。实现后帮我启动这个项目。如果遇到没安装的程序环境就自动安装没安装的环境。

在这个过程中，你需要意识到 AI 不只是聊天模型，它能够帮助你操作本机环境：创建文件、安装依赖、执行启动命令等。你可以直接用自然语言描述想要达成的目标，由 AI 来决定具体执行哪些命令、如何组织代码。

如果执行过程中遇到问题，AI 会在对话里展示报错和处理方案，你可以继续通过对话让它调整，而不必自己记住所有命令细节。

⚠️ 需要注意

例如下图所示，**有时候 AI Agent 会在执行的过程中暂停，这是因为它需要等待你输入一些信息进行交互**，比如输入创建的名字，或者回车确认指令执行。或者点击指令进行执行。一般情况我们直接回车即可，如果你不确定这步需要做什么，你可以截图当前界面询问大模型应该如何操作。

如图所示，这里我们需要点击 Run 进行确认：![img](https://datawhalechina.github.io/easy-vibe/assets/index-2026-01-09-10-52-55.BU8zRN_0.webp)

如图所示，这里我们只需要输入 y 即可确认：![img](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/index-2026-01-09-10-53-24.CGQ04B3C.webp)

![img](https://datawhalechina.github.io/easy-vibe/assets/index-2026-01-09-10-26-33.D_-gv_Ao.webp)

如图所示，这里我们正在创建模板，但不知道如何操作，我们可以截图该部分对大模型进行询问：

![img](https://datawhalechina.github.io/easy-vibe/assets/index-2026-01-09-10-29-12.DSgWk1xx.webp)

AI Agent 在执行过程中暂停的还有一部分原因是因为此时启动了一个“服务”，我们的贪吃蛇本身属于一种“服务”，如果你看到如下命令的网址，则表示 Agent 帮我们执行了一个本地的电脑服务，我们可以访问对应的网址访问我们的贪吃蛇，由于服务需要持续启动，这里会陷入暂停。我们只需要点击 `Skip` 按钮即可。

![img](https://datawhalechina.github.io/easy-vibe/assets/index-2026-01-09-10-30-51.DHutQwZi.webp)

在这个过程中，如果你遇到一些术语和看不懂的内容，不用担心，你可以查阅附录中的“计算机术语解释”部分，或者直接向 AI 咨询，或者及时提问！

如果你在过程中遇到不符预期的现象，例如贪吃蛇撞墙后不会结束游戏，贪吃蛇点击开始后不会移动，这时你只需要把现象描述给侧边栏 Agent 即可。如果遇到报错问题，记得截图或者复制错误到侧边栏 Agent，如果多次仍然不能解决问题，请你尝试更换模型操作。

稍作片刻，我们即可得到类似 z.ai 一样的结果：

![img](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/index-2026-01-09-10-33-37.AwSORceg.webp)

我们可以点击右下角的打勾进行确定代码的变更，也可以点击 `Cancel` 按钮取消变更。或者点击 2 files need review 的地方展开查看变动后的代码。

这里还值得注意的是，由于修改代码并不一定正确，我们还需要知道所有的 IDE 的 Agent 都支持代码回退，例如，假设我这里不小心做了个错误的修改操作，或者这次操作的结果让你感到不满意，在修改结束后我们可以返回输入框的部分，点击 Revert 按钮将操作回退到修改前的状态，你可以修改输入的文字进行再一次操作：

![img](https://datawhalechina.github.io/easy-vibe/assets/index-2026-01-09-10-42-53.B1wktg3P.webp)

### 4.4 第三步（可选）：向 AI 追问代码实现细节

当贪吃蛇游戏已经可以正常运行时，如果你对前端或 React 还不熟，可以继续在同一个聊天窗口里，请 AI 用尽量口语化的方式帮你导览代码。你不需要切换工具，也不必刻意去翻文档，只要围绕当前项目持续发问即可。

一个比较实用的做法是，让 AI 先整体讲一遍“游戏是怎么动起来的”，再拆到具体细节。比如你可以直接提问：

> “请从上到下讲一遍，这个贪吃蛇游戏每一步是怎么动起来的？尽量少用专业术语。”

![img](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/index-2026-01-09-10-44-36.CDa8fIUM.webp)

然后再顺着它的回答继续追问关键点，例如：

> “蛇在屏幕上的每一节身体，是用什么数据结构来记的？能打个比方吗？”
> “你是怎么控制‘隔一段时间动一下’的？这在代码里是哪一段？”
> “蛇吃到食物时，你做了哪几步操作？在哪一段逻辑里判断吃到了？”
> “撞墙和撞到自己，分别是在哪些代码里判断出来的？”

如果你看到某个文件（比如 `SnakeGame.tsx`）但完全不知道它在干什么，也可以直接请 AI 分块说明：

> “请把 `SnakeGame.tsx` 按功能分几块讲一下：每一块大概负责什么，用通俗一点的说法。”

在这一轮对话中，你可以把不懂的词一律当成追问入口，比如：

> “你刚才说的‘状态’具体指什么？能用一个生活中的例子解释吗？”
> “你说的‘定时器’在这里主要是干嘛的？如果把它去掉，会发生什么？”

通过这种方式，你的目标不是一下子记住所有概念，而是先搞清三件事：这款游戏里有哪些核心数据（蛇、食物、分数、游戏状态等），这些数据在什么时机会发生变化（移动、吃到食物、游戏结束等），以及每一种变化对应的是哪一小段代码。只要这三点理顺了，你就基本可以看懂这份代码的主干逻辑。

### 4.5 第四步：让 AI 把画面变好看一点

这里先提醒一件对小白很重要的事情：不要只对 AI 说一句“我要把这个画面变好看”。这种说法对人类设计师都太模糊，更别说对模型了——“好看”是什么风格、哪些地方需要调整、是排版问题还是配色问题，AI 都无法从你这一句里读出来。为了让 AI 真正做出接近你心里预期的效果，你需要学会把“我想要好看”这种模糊目标拆成一串具体、可执行的小要求。

比如，很多人一开始会这样说：

> “我要把这个画面变好看一点。”

例如，你可以先给出一组整体需求：

> “请帮我把游戏界面整体美化一下：
>
> - 游戏区域居中显示，不要贴在左上角；
> - 换成较浅的背景色，让蛇和食物更醒目；
> - 把分数放大，放在明显的位置；
> - 以蓝色为主色调，美化一下整体配色和按钮。”

如果你希望在“游戏结束”时有更清晰的反馈，可以进一步补充：

> “当游戏结束时，请在画面中央显示‘游戏结束’，下面有一个‘重新开始’按钮，可以重置游戏。”

AI 会根据你的描述，直接修改 React 组件和样式。保存后刷新浏览器，你就能看到新的界面。如果效果和你想象的还有差距，可以继续做小步调整，例如：

> “分数再大一点，颜色更醒目一些。”
> “游戏区域再紧凑一点，四周预留一点留白。”
> “重新开始按钮改成蓝色圆角风格，放在提示下方居中。”

在这个阶段，如果某次修改导致报错，也不需要自己硬查。直接把错误信息复制到聊天窗口，或者配合一段简要描述，比如“这是我刚才美化界面后出现的错误”，让 AI 在当前项目上下文里帮你定位和修复。这样你就可以在“不断对话、不断刷新”的循环中，把一个能跑的 Demo 逐步打磨成界面清晰、交互顺畅的小型成品。

### 4.6 （可选）参考 z.ai 架构修改贪吃蛇结果

对于 vibe coding 小白来说，最难的事情反而是不知道什么才算是“最佳实践“，不知道怎么样的架构才是最适合的；因为不知道计算机基础，所以没办法很好的引导 AI，解决这个难题的方法是”直接参考“；还记得我们之前说过的 z.ai 中可以查看代码吗？其实对应 README（项目中用于介绍功能和技术架构的部分）中已经给出了一个最佳架构参考：

![img](https://datawhalechina.github.io/easy-vibe/assets/index-2026-01-09-10-49-33.CFzmtk-B.webp)

我们想要让本地的结果尽量符合 z.ai 的结果，我们可以复制这个 README 的全部内容，粘贴到 Trae 的侧边栏中，让他根据 README 的架构，修改本地的代码。

![img](https://datawhalechina.github.io/easy-vibe/assets/index-2026-01-09-10-50-31._n_odzWJ.webp)

最后我们能得到与 z.ai 高度相似的页面设计风格：

![img](https://datawhalechina.github.io/easy-vibe/assets/index-2026-01-09-11-00-57.YlH8lu_g.webp)

环境认知

理解 IDE 与 AI IDE

本地实战

Trae 打造贪吃蛇

工具详解

熟悉 IDE 界面

沟通技巧

高效与 AI 对话

## 5. 界面上每个按钮是干什么的

在上述操作中，我们已经快速跑通了最小程序生成闭环，但我们仍然对 IDE 不能说得上熟悉。为了彻底熟悉这个之后与我们长期相处的工具。我们会在这一节中对 IDE 的每个细节环节进行深入解释，首先从界面开始，不同 AI IDE 的界面略有差异，但大部分都延续了 [VS Code 的布局](https://code.visualstudio.com/docs/getstarted/getting-started)。

![img](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/image32.CCunjUkl.webp)

其中每个部分的具体作用为：

- **Title Bar（标题栏）**：显示文件名和窗口控制按钮。
- **Activity Bar（活动栏）**：切换文件、搜索等功能视图。
- **Side Bar（侧边栏）**：展示文件列表等具体内容。
- **Editor Groups（编辑区）**：编写代码的核心区域。
- **Breadcrumbs（路径导航）**：显示文件路径，支持跳转。
- **Minimap（代码缩略图）**：快速预览和定位代码。
- **Panel（底部面板）**：包含终端和输出窗口。
- **Status Bar（状态栏）**：显示当前环境状态。

更具体的详细内容解释，请查看[附录中的 虚拟 IDE 可视化 IDE 原理部分](https://datawhalechina.github.io/easy-vibe/zh-cn/appendix/2-development-tools/ide-basics.html)。

环境认知

理解 IDE 与 AI IDE

本地实战

Trae 打造贪吃蛇

工具详解

熟悉 IDE 界面

沟通技巧

高效与 AI 对话

## 6. 怎么跟 AI 说话才有效

随着 AI 能力越来越强，我们已经可以把很多“让程序员写代码”的工作交给 AI 来完成。
但是，在实际使用中你会发现：同样是用同一个 AI，有的人几句话就能要到能跑起来的小项目，有的人聊了半天，结果却完全不是自己想要的，其差别往往不在于“谁更聪明”，而在于——你跟 AI 说话的方式，是不是足够具体、足够有步骤。
本节我们从几个常见场景出发，介绍一些适合完全小白的提问方式，帮助你更稳定地让 AI 给出可用的结果。

### 6.1 说清楚你的需求：从“模糊想法”到“具体说明”

很多人第一次用 AI 时，习惯只说一句非常笼统的话，比如：

> “帮我做个网页。”
> “帮我写个小程序。”

在这种情况下，AI 只能自己“脑补”你想要什么，于是它会随便给你一个看上去挺完整的东西，但往往和你真正想做的差很多。
要让 AI 更听得懂你的意思，需要把“脑子里的想法”拆开，用几句话一步步说清楚。

可以从这几个方面来补充：

1. **告诉它，你拿这个东西来干嘛**
   比如，不要只说“个人网站”，而是说：
   - “我想做一个只包含一页内容的个人简介网页，用来发给招聘的人看。”
2. **告诉它，大概需要哪几块内容**
   不用说专业词，只要描述你希望页面上出现什么，比如：
   - “页面要有三个部分：最上面是名字和一句自我介绍，中间列出几条工作经历，最下面放邮箱和微信号。”
3. **告诉它，你的水平和限制**
   让 AI 按照小白能接受的方式来做，比如：
   - “我完全不会写代码，请只用最简单的写法，让我可以直接复制到一个文件里，在浏览器里打开。”
4. **告诉它，你希望怎么拿到结果**
   例如：
   - “请给我一份可以直接保存为 `index.html` 并在浏览器里打开的完整代码。”

综合起来，可以让你对 AI 这样说：

> “我完全不会写代码，想做一个只包含一页内容的个人简介网页，用来发给招聘的人看。
> 页面需要三个部分：上面一行是名字和一句自我介绍，中间是几条工作经历，下面是邮箱和微信号。

当你把这些信息说清楚之后，AI 就能更接近你真正的需求，而不是随便给你一个“看起来很厉害但用不上的东西”。

### 6.2 用对节奏：先“能跑起来”，再一点点变复杂

对完全小白来说，最常见的坑是：一上来就想做一个“非常完整”“功能很多”的东西。
比如：

> “帮我做一个像淘宝那样的网站。”
> “帮我做一个可以注册、登录、下单的系统。”

结果往往是：AI 给你一大团代码，你复制之后不是打不开，就是到处报错；你也看不懂哪里出了问题，最后只能放弃。

更适合的做法，是**主动控制节奏**，让 AI 跟着你一步一步来，而不是一次性把所有东西都砸给你。可以按下面这个顺序提要求：

1. **第一步：先要一个“最小的例子”**
   只检查一件事：能不能在浏览器里看到东西。
   例如：

   > “请先给我一个最简单的示例，只要在浏览器里能看到一行‘这是我的主页’就行。
   > 再一步步告诉我：文件名该叫什么，应该怎么保存，怎么打开。”

2. **第二步：在这个基础上，慢慢把内容加完整**
   当你确认“确实能看到那一行字”之后，再说：

   > “在刚才的基础上，帮我增加一个‘工作经历’区域，把完整代码重新发给我。不要只发改动的部分。”

3. **第三步：结构差不多之后，再考虑好不好看**
   例如：

   > “现在页面已经能正常显示内容了。接下来请帮我稍微美化一下：整体居中，标题大一点，用一个比较舒服的字体。请给出更新后的完整代码。”

每加一步，你都先运行一次，确认真的有变化，再让 AI 往下加。这样就算哪一步出问题，你也可以很快回到“上一版还正常”的状态，而不用完全推倒重来。

### 6.3 善用截图和复制：不会说就“把画面扔给 AI”

很多完全小白遇到的难点，不在于“不会改代码”，而是在于**不知道怎么把问题说出来**。
比如：

- 浏览器里突然弹出一大堆英文报错，你完全看不懂。
- 网页的排版和你想的不一样，但你也不知道该用什么词来形容。

在这些情况下，不需要硬挤专业术语，最简单的方式就是——**把你看到的东西原样丢给 AI**。

可以这样做：

1. **复制报错文字**
   当你看到一串红色错误消息时，可以直接复制出来，然后说：

   > “这是我运行后出现的完整错误信息。我看不懂这些英文，请先用普通人能听懂的话解释一下，这大概是什么意思。
   > 然后告诉我，我现在最简单应该怎么改。”

2. **给 AI 看截图** 如果你觉得"这个页面看着就是不对"，但又不会描述，可以：

   - 截一张当前页面的图；

   - 把你正在用的那份代码，一整段复制给 AI；

   - 然后说明：

     > "这是现在页面的样子，这是我现在的完整代码。 我原本希望它是三列排版，现在变成一列了。请你帮我看一下原因，并给我一份改好后的完整代码。"

   💡 关于截图功能的补充说明

   需要注意的是，**并非所有 AI 模型都支持"看图片"**。这涉及到两个不同的概念：

   - **纯文本大模型（LLM）**：只能处理文字输入，无法识别图片内容。如果你给它发截图，它要么拒绝处理，要么无法正确理解图片中的信息。
   - **多模态模型**：能够同时处理文字、图片等多种类型的输入，可以"看懂"你发的截图，并根据图片内容给出建议。

   **常见模型的能力参考**（以 Trae 中可选的模型为例）：

   | 模型                | 是否支持图片输入 |
   | :------------------ | :--------------- |
   | Doubao-Seed 系列    | ✅ 支持           |
   | GLM-4.7 / 4.6       | ❌ 不支持         |
   | MiniMax-M2.7 / M2.5 | ❌ 不支持         |
   | DeepSeek-V3.1       | ❌ 不支持         |
   | Kimi-K2.5           | ✅ 支持           |
   | Kimi-K2-0905        | ❌ 不支持         |
   | Qwen-3-Coder        | ❌ 不支持         |
   | Gemini 系列         | ✅ 支持           |
   | GPT 系列            | ✅ 支持           |

   **使用建议**：如果你想通过截图让 AI 帮你排查界面问题，请先确认你使用的模型支持图像输入。如果不支持，你可以改用文字描述问题，或者将错误信息复制粘贴给 AI。

3. **遇到喜欢的网页，想做个类似的**
   不需要说“这个布局叫什么”，直接：

   - 截图或复制那页的主要标题、段落；

   - 再说：

     > “我想做一个结构和这个差不多的页面，不需要一模一样。
     > 请帮我用简单一点的代码，搭一个类似的框架出来，然后我再自己把文字换成我的。”

简单来说：你负责“把看到的东西搬给 AI”，再用最朴素的话说“我希望它变成什么样”；剩下的“翻译成代码、解释名词、找问题”，交给 AI 来做。

### 6.4 当 AI 生成的代码不工作时：一套通用应对方法

在实际练习中，你一定会遇到这种情况：
AI 很认真地给了你一段代码，你也老老实实地复制进去了，但结果要么是浏览器一片空白，要么完全不是它说的那个效果。
这并不代表你“学不会”，也不代表 AI 完全错了，而是你们之间还缺少几轮“来回确认”。

当代码“不工作”时，可以按下面这套固定流程来跟 AI 说：

1. **先把“你做了什么 + 现在什么样”说清楚**
   避免只说“打不开”“不行”。可以这样描述：

   > 打开之后，页面是完全空白的，没有显示你说的那句欢迎文字。 我打开了 xxxx 页面，其中没有刚才我说的部分啊，这不能用

2. **把你现在的完整代码发给 AI**
   很多时候问题出在：复制少了一行、或者上一次和这一次的内容混在一起了。
   你可以说：

   > “下面是我现在这个文件里的全部代码。
   > 请你对比一下有没有哪里少了、写错了，或者顺序不对。
   > 请直接给我一份修正后的完整代码，不要只发一小段。”

3. **如果有错误提示，一并给出**
   比如浏览器右上角弹出的错误，或者底部的一些红字。你可以：

   - 把错误文字复制出来；

   - 或者截一张图；

   - 然后说：

     > “这是我看到的错误提示。我完全看不懂，请先用简单的方式说明这大概是什么问题，再告诉我现在最需要改哪几行。”

4. **要求对方用“小白模式”一步一步讲**
   你可以直接把自己的情况说清楚，让它别省略中间步骤：

   > “我完全不会写代码，请你一步一步告诉我：
   > 第 1 步要改哪一行，
   > 第 2 步要怎么保存，
   > 第 3 步要怎么重新打开或者刷新页面。
   > 每一步都请用完整的句子写出来。”

5. **最后，请它帮你做“应该看到什么”的对照**
   例如：

   > 请先说一下，按照你改好的代码，正常情况下我打开网页应该看到什么内容。

只要你按照这套流程来和 AI 交互，大部分“代码不工作”的情况，都可以在几轮来回中解决掉。
同时，你也会逐渐熟悉常见的问题类型，下次再遇到类似情况就能直接解决。

## 7. 小结与下一步

这一章里，你完成了一次从“能在网页里玩一个 AI 生成的贪吃蛇”，到“能在本地用 AI IDE 自己搭出一个小游戏”的升级。你大致搞清了三件事：写代码为什么离不开一个像 VS Code 这样的 IDE；在这个基础上，再加上 AI（Trae、Cursor 等）之后，IDE 不再只是工具箱，而是多了一个能听懂自然语言、帮你新建文件、装环境、改代码的“实习工程师”；以及 IDE 界面上每一块区域（左侧文件、底部终端、中间编辑区、右侧 AI 面板）分别管什么，用起来就不再一头雾水。

更重要的是，你已经实际跑通了一次完整流程：在本地新建空文件夹 → 用 AI IDE 打开 → 在侧边栏对话里描述需求 → 让 AI 生成项目并启动开发服务器 → 出现问题时，把“现象 + 全部代码 + 报错截图”一起丢给 AI，请它用“小白模式”一步步修。这个过程中，你也练习了如何写更有效的提示词：说清目标、内容结构和自己的水平，控制好节奏，从“先能跑起来”到“再变好看、变好玩”。