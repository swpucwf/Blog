# UI 与 UX：从好看到好用

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_mlkzwvbv-cfdf61ece2ce1cbf.jpg

<p>图 1：界面（UI）与交互（UX）</p>

## 1. 先懂三个概念：CSS、组件、布局

看着 AI 生成的网页，觉得太丑，尤其是千篇一律的紫色渐变；试图让它把按钮变漂亮点，结果代码里多了一堆 `<style>` 标签，改了这里坏了那里。要解决这个问题，先得懂三个概念：

- **CSS（层叠样式表）**：HTML 是网页的骨架（素颜），CSS 就是化妆品——决定按钮是圆角还是直角、背景是渐变还是纯色；
- **组件（Component）**：现代网页不是画出来的，而是搭出来的。导航栏、按钮、输入框都是预先做好的乐高积木，不用每次手写「带圆角、有阴影、悬停变色的红色按钮」，直接用 `<Button />`；
- **布局（Layout）**：元素怎么摆放。朋友圈内容从上到下排列是 Flexbox；小红书照片像棋盘填满屏幕是 Grid。只需要说「把三个按钮横向排列」「把页面分成左右两列」，具体属性交给 AI。

**Tailwind 与 shadcn/ui** 是这套玩法的地基。传统 CSS 像写作文：给每个元素起名（`login-box`），再在单独文件里描述长宽颜色，很容易撞名——你叫 `card`，AI 也叫 `card`，样式打架页面就乱。Tailwind 允许直接在标签里写 `class="p-4 bg-red-500 rounded-lg"`：**AI 最擅长做填空题**，让它堆砌描述性单词，比管理复杂文件引用准确得多。

shadcn/ui 是基于 Tailwind 的高质量组件库，但和 Ant Design 这类传统组件库完全不同：它不是安装在 `node_modules` 里的黑盒，而是通过命令行**把组件源码复制进项目**。这意味着拥有完全控制权——AI 不仅能使用积木，还能改积木的内部构造。配合 Lucide React 作为指定图标库，可以避免 AI 瞎编图标名。

## 2. UI 只是皮囊，UX 才是灵魂

单纯好看还不够。让 AI 扮演**交互设计师**而不是代码生成器：不再直接下令「画一个页面」，而是描述场景，让它完善交互逻辑。几个经典 UX 避坑点：

| 场景 | 错误示范 | AI 指令 |
|---|---|---|
| 按钮反馈 | 点击「支付」毫无反应，用户狂点导致扣款两次 | 给提交按钮加 Loading 状态：点击后变灰、显示转圈、不可再次点击，直到请求结束 |
| 破坏性操作 | 点「删除」图标，数据瞬间消失，手滑误删 | 设计 AlertDialog，要求点「确认删除」红色按钮，甚至输入 DELETE 才能执行 |
| 空状态 | 新用户进来没有任何数据，页面空白像出 Bug | 列表无数据时展示插画组件，并放一个「创建第一个项目」的引导按钮 |

**提示词技巧**：创建组件时明确技术栈与来源——「创建一个登录表单，包含邮箱、密码、登录按钮，使用 shadcn/ui 组件和 Tailwind CSS」；优化布局时说清响应式——「移动端单列，桌面端双列」。新手常犯的三个错误：一次性让 AI 生成复杂页面（应分步）、忽略响应式（应提前说明）、过度设计（应从简单开始迭代）。UI 开发本质是堆样式细节，而 AI 在细节上容易出错，所以提示词清晰、分步迭代是关键。

**响应式设计**：同一套代码根据屏幕尺寸自动调整布局——手机上单栏、电脑上双栏；移动端隐藏次要信息、桌面端完整展示。这些细节 AI 会处理，只需要在提示词里说清「要适配移动端」。

**性能意识**：高清大图背景有好几兆，移动网络下用户等十几秒就会关掉。选图要控制大小，非用大图就让 AI 压缩。但不必现在深究优化——部署到云平台后会自动压缩图片、加速访问；等有了真实用户、反馈慢再优化，**过早优化是浪费时间**。

**从 UI 到 UX 的产品思维**：AI 擅长写 CSS、堆样式，但它不知道用户是谁、在什么场景下使用。这些「为什么」只有人能回答：

- **你负责理解用户**：是医生还是学生？地铁上用还是办公室用？赶时间还是慢慢浏览？
- **你负责定义场景**：「这个按钮是用户填完长表单后点击的，他此时很紧张，怕数据丢失」；
- **AI 负责实现细节**：根据场景加 Loading、本地保存草稿、明确提示「提交成功」。

**好设计不是技术炫技，而是理解用户。**

## 3. AI 设计工具：先出原型，再集成

这些工具的定位是**生成初始 UI**，Claude Code 擅长**集成和修改**。推荐流程：设计工具出原型 → 复制代码到项目 → Claude Code 做业务逻辑与细节调整。

### 3.1 Google AI Studio

Google 的 AI 应用开发平台，用 Gemini 生成完整 Web 应用（含 UI）。可以理解成在线版 Claude Code：打开浏览器就能用，不需要本地环境。适合快速搭 AI 应用原型、需要集成 Gemini、想一站式从设计到部署、以及免费用户。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222173132952.png

<p>图 2：Google AI Studio 入口</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222173412942.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222173528903.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222173540818.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222173829118.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222173907920.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222173934821.png

<p>图 3-8：Google AI Studio 的创建与生成界面</p>

### 3.2 Google Stitch

从文字描述或草图生成完整 UI 设计稿。适合从零设计页面布局、需要**设计稿而不是代码**（设计稿是能看不能跑的图，代码是能直接运行的程序）、快速验证视觉想法，免费。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222173221042.png

<p>图 9：Google Stitch 入口</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260227000703458.png

<p>图 10：Stitch 生成的设计稿</p>

### 3.3 v0.app

Vercel 出品，用自然语言直接生成 React + Tailwind + shadcn/ui 代码。适合快速生成单个组件或页面原型、需要 shadcn 风格、想要可直接复制进项目的代码。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222173330046.png

<p>图 11：v0.app 首页</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222174541135.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222174730883.jpg

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222174800532.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222174901391.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222174926727.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222174958816.png

<p>图 12-17：v0 的生成结果与迭代过程</p>

### 3.4 Figma

专业设计师标配，也加入了 AI 辅助。它不直接生成代码，但能做出精确设计稿。如果对视觉效果有明确想法但用文字描述不清，Figma 就是画板——先画出来，再让 AI 照着写代码。适合像素级设计稿、团队协作、设计系统与组件规范。

与 Claude Code 配合时，可以通过 MCP 连接 Figma，直接说「根据 Figma 设计稿实现这个页面」，AI 会读取颜色、间距、字体并生成匹配的代码。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222180858852.png

<p>图 18：Figma 工作区</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222181216119.png

<p>图 19：通过 MCP 连接 Figma</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222180953413.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222181002982.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222181024025.jpg

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222181106163.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222180950077.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222181145149.png

<p>图 20-25：Figma 中的设计稿与协作界面</p>

### 3.5 Lovable

不只做 UI，能生成包含前后端的完整应用，适合快速验证产品想法；相比 Google AI Studio 更侧重生成可部署的成品，而不只是原型。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222183416399.png

<p>图 26：Lovable 首页</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222183506857.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222183546410.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222183557356.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222183609472.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222183617930.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222183631655.png

<p>图 27-32：Lovable 的应用生成与预览</p>

**工具选择**：不确定用哪个就直接问 AI——「我想做一个电商产品详情页，有图片轮播、价格展示、加入购物车按钮，用什么工具最快？」它会根据需求推荐工具与工作流。工具只是手段，**产品设计思维才是核心竞争力**。

## 4. 组件库：现成的乐高积木

选对组件库能省掉 80% 的 UI 工作。

### 4.1 shadcn/ui（首选）

核心理念：不是 npm 包，而是把组件源码复制进项目，完全控制。普通组件库装完代码藏在 `node_modules` 深处，AI 看不到也改不了；shadcn 把代码放进项目文件夹，AI 可以像改自己写的代码一样修改。优势：代码可读可改、无障碍支持好（视障用户用屏幕朗读器也能操作）、生态丰富。

告诉 AI「项目使用 shadcn/ui」，它就会优先使用这些组件。生态扩展里 **Magic UI** 提供动画与特效组件，**Aceternity** 提供高级 UI 特效。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222210620318.png

<p>图 33：shadcn/ui 官网</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222210651904.png

<p>图 34：shadcn 组件源码直接进入项目</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222210713180.png

<p>图 35：在 AI 对话中指定 shadcn/ui</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222210923284.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222210905692.png

<p>图 36-37：Magic UI 与 Aceternity 的特效组件</p>

### 4.2 其他主流组件库

| 组件库 | 特点 | 适用 |
|---|---|---|
| Ant Design | 阿里出品，60+ 组件，中文文档最完善 | 后台管理、企业内部工具、数据密集型 |
| Element Plus | 饿了么出品，Vue 3 生态最主流 | Vue 后台默认选择 |
| TDesign | 腾讯出品，同时支持 React/Vue 2/3/小程序，含 Figma 资源 | 多端、设计体系 |
| Arco Design | 字节出品，60+ 组件经大规模验证，内置图标与风格配置 | 数据密集型、主题定制 |
| Material Design 3 | Google 官方设计规范（设计语言本身，非组件库） | 跨平台、Dynamic Color |
| MUI | 基于 Material Design 的 React 组件库，全球使用最广 | 国际化、主题定制 |
| Radix UI | 无样式底层组件库，只管行为与无障碍，shadcn 的底层 | 完全定制 |
| HeroUI（原 NextUI） | 基于 Tailwind，开箱即用、内置动画 | 现代感、快速开发 |

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222211429267.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222212101447.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222212117578.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222212134560.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222222521284.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222211543203.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222211603616.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222212148692.png

<p>图 38-45：八种主流组件库的界面风格</p>

用组件库的关键，是在提示词里**明确说清用的是哪个**：「用 shadcn/ui 的 Table 组件展示用户列表，支持排序和分页」「用 Ant Design 做一个订单管理表格，支持筛选和导出」。

## 5. 动画与交互库

恰到好处的动画能让页面从「能用」变成「好用」，但过度动画会让用户头晕。**好动画应该是用户几乎察觉不到的**——它只是让交互更流畅，而不是抢戏。

**Motion（原 Framer Motion）** 是 React 生态最流行的动画库。只需要描述「动画结束时元素长什么样」，它自己算中间过程：

| 想要的效果 | 告诉 AI |
|---|---|
| 元素淡入 | 用 Motion 给卡片加淡入动画 |
| 列表逐个出现 | 用 Motion 的 `staggerChildren` 让列表项依次淡入 |
| 拖拽排序 | 用 Motion 的 Reorder 组件 |
| 页面切换 | 用 AnimatePresence 做过渡 |
| 滚动触发 | 用 `whileInView` 在元素进入视口时触发 |

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222213320488.jpg

<p>图 46：Motion 动画示例</p>

**GSAP** 是 Web 动画领域的老牌王者，性能极强。支持时间线（Timeline，像剪辑视频的时间轴，编排动画先后与间隔）、插件丰富（ScrollTrigger、Draggable、MorphSVG），不限于 React。适合营销落地页的炫酷滚动、复杂 SVG 路径动画、精确时间线序列、性能要求极高的场景。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222213406770.png

<p>图 47：GSAP 官网</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222213514935.jpg

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222213459756.png

<p>图 48-49：GSAP 滚动与时间线效果</p>

**Three.js** 是网页 3D 的主流选择，配合 `@react-three/fiber` 可以用 React 组件方式写 3D 场景。适合 3D 产品展示、粒子背景、交互式 3D、3D 数据可视化。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222213553727.jpg

<p>图 50：Three.js 3D 场景</p>

**不是所有动画都要引入大型库**：

| 效果 | 方案 | 是否需要库 |
|---|---|---|
| 按钮悬停变色 | CSS `transition` | 不需要 |
| 元素淡入淡出 | CSS `@keyframes` | 不需要 |
| 简单滑入滑出 | Tailwind `animate-*` | 不需要 |
| 列表交错动画 | Motion | 需要 |
| 复杂滚动动画 | GSAP ScrollTrigger | 需要 |
| 3D 效果 | Three.js | 需要 |

提示词示例：「给这个按钮加悬停效果，用纯 CSS transition，不要引入额外的库」。

**动画的度**：时长 200-500ms（太快看不清、太慢觉得卡）；用户主动触发的动画（点击、悬停）可以明显一点；自动播放的（进入视口）要含蓄；移动端减少动画，省电省流量。

## 6. UI 风格与灵感

说「要一个现代风格的登录页」，AI 理解的「现代」和你想的可能完全不一样。跟 AI 沟通设计，光说「好看」没用，得说「怎么个好看法」。

**灵感网站**：

| 网站 | 定位 | 怎么用 |
|---|---|---|
| Awwwards | 顶级网页设计奖 | 浏览获奖作品，截图发给 AI「照这个风格来」 |
| Dribbble | 设计师社区 | 按页面类型搜索（dashboard、landing page） |
| Mobbin | 真实 App/网站截图库 | 搜索具体功能（onboarding、pricing）看成熟产品怎么做 |
| Land-book | 落地页灵感 | 做产品首页或营销页时参考 |

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222213812614.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222213829480.png

<p>图 51-52：Awwwards 获奖作品</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222213914501.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222213927522.png

<p>图 53-54：Dribbble 作品流</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222213947631.png

<p>图 55：Mobbin 的真实应用截图</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260222214021688.png

<p>图 56：Land-book 的落地页集合</p>

**风格关键词**：用专业术语比说「好看」有效 100 倍。在 Dribbble 搜 `glassmorphism` 能精准找到毛玻璃风格，搜 `beautiful design` 只会得到一堆不相关结果；与 AI 沟通同理，每种风格都有自己的标准名字。

**向 AI 描述风格的公式**：

```text
风格关键词 + 配色方案 + 参考网站/截图 + 具体场景
```

**配色工具**：Realtime Colors（实时预览配色在网页上的效果）、Coolors（随机生成和谐配色）、Happy Hues（完整配色方案 + 预览）、Tailwind 调色板（内置颜色系统）。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260226235525847.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260226235543860.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260226235612718.png

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260226235631335.png

<p>图 57-60：配色工具的界面</p>

配色捷径：不想自己选，就告诉 AI「用 Tailwind 的 slate 色系做暗色模式」「用 blue 色系做主色调」——Tailwind 的颜色系统已经调好明暗层次。

## 7. 让页面更「高级」的效果

有些页面一看就很贵，但说不清为什么——它们用了标准化的高级效果：**过渡效果**（元素优雅滑入而非「啪」地蹦出）、**按钮特效**（点击最多的元素，有质感就提升一档）、**视觉特效**（纯视觉点缀）、**滚动效果**（Apple 官网那种图片与文字不同速移动）、**交互效果**（拖、滑、靠近，让页面从「看」变成「玩」）。

使用原则：

1. **先做功能，再加效果**——效果是锦上添花，不是雪中送炭；
2. **一个页面最多 2-3 个高级效果**，全都要 = 花里胡哨；
3. **移动端谨慎**，复杂动画可能卡顿；
4. **性能优先**，让页面变慢就果断去掉；
5. **有意义的动画**——传达信息或引导注意力，不为炫技。

## 8. 让 AI 记住设计规范

用 shadcn 做几个页面后，会出现新问题：第一次说「用 shadcn 组件」，AI 照做；第二次得再说一遍；而且它有时会「忘记」——上个页面蓝色主题，这个页面突然紫色；说好按钮圆角 8px，这次给 12px。本质是 **AI 没有记忆**，每次对话都从零开始。

### 8.1 设计系统的困境

没有明确规范时，只能凭感觉调：这个按钮圆一点、那个卡片方一点；这里蓝色、那里绿色。单看都还行，放一起很乱，像好几个人做的。**设计系统**就是一套「长什么样」的标准：颜色（主色、成功、错误）、间距、圆角、阴影、字体。大厂设计团队会花几个月打磨并强制遵守；一个人没有设计团队怎么办？

### 8.2 设计规范其实已经有了

打开项目里的 `tailwind.config.js`——**这就是设计规范**。Tailwind 配置本身就是一套设计系统，把它翻译成人话告诉 AI「只能用这些颜色、这些间距」，它就不会乱来。

### 8.3 第一步：让 AI 扫描生成规范

不用手动整理，让 AI 并行扫描：`tailwind.config.js`（颜色、间距、圆角、阴影等 tokens）、`src/components/ui/`（已安装组件清单）、现有页面代码（实际使用的样式模式），整理成 `DESIGN_SYSTEM.md`：

```text
请并行扫描 tailwind.config.js 和 src/components/ui 目录，分析项目的设计规范，
生成一份 DESIGN_SYSTEM.md 文档。文档要包含：颜色系统、间距系统、圆角、阴影、
字体规范、可用组件列表。
```

生成的内容包括颜色系统（主色、辅助色、状态色、文本色）、间距系统（内边距、组件间距、页面边距）、圆角与阴影、可用组件清单（Button、Card、Form、Input、Select、Dialog、Table 等）。

### 8.4 第二步：创建项目 Skill

为什么放 Skill 而不是直接写进 CLAUDE.md？因为 Skill 有**渐进式加载**：

- Skill 的描述（name + description）永远在 context 里，告诉 AI「这个项目有设计规范」；
- 详细内容只在需要时加载，不会一直占用 context。

若把整个 `DESIGN_SYSTEM.md` 写进 CLAUDE.md，每次对话都会占用大量 context，哪怕只是在改后端 API。

结构：

```text
.claude/skills/my-project-design/
├── SKILL.md
└── references/
    └── DESIGN_SYSTEM.md
```

```yaml
---
name: my-project-design
description: 本项目的设计规范和组件库约束。当需要创建或修改 UI 组件时使用。
---

# 项目设计规范

本项目使用 shadcn/ui + Tailwind CSS。

## 何时阅读详细规范

创建或修改 UI 组件时，请阅读 references/DESIGN_SYSTEM.md，了解：
- 可用的颜色、间距、圆角等 design tokens
- 已安装的 shadcn/ui 组件清单
- 自定义业务组件（如 DataCard）
- B 端场景的特殊约束（表单分组、表格操作确认等）
```

SKILL.md 只有几行，告诉 AI「需要时去读 references」——写后端时不会被设计规范占用 context，写前端时会自动加载。

### 8.5 第三步：自动触发与定期更新

创建后不需要每次说「请遵循设计规范」：AI 看到 description 里的「当需要创建或修改 UI 组件时使用」，判断「创建用户列表页面」属于该场景，自动加载 Skill、读取 `DESIGN_SYSTEM.md`、按规范生成代码。

项目变大后要定期更新：添加新组件后让 AI 重新扫描 `src/components/`；改了 Tailwind 配置后重新提取 tokens。几秒钟完成，AI 会对比新旧版本，只更新变化部分，保留手动添加的说明与约束。

### 8.6 通用 Skill + 项目 Skill

| 类型 | 解决什么 | 例子 |
|---|---|---|
| 项目 Skill | 做成什么样（项目特定规范、自定义组件、业务约束） | `my-project-design` |
| 通用 Skill | 怎么做（shadcn 使用指南、Tailwind 最佳实践、响应式、无障碍） | `frontend-design` |

两者可同时使用。安装通用 Skill：

```bash
npx skills add https://github.com/anthropics/skills --skill frontend-design
```

其他常用 UI/UX Skills：`web-design-guidelines`（设计规范审查）、`ui-ux-pro-max`（专业设计指南）、`building-native-ui`（原生移动端 UI）、`audit-website`（网站 UI/UX 审计）。

### 8.7 B 端与团队协作

B 端追求「好用」，C 端追求「好看」：C 端可以有炫酷动画，B 端要的是快速完成任务；C 端可以简化功能，B 端常常一页塞几十个字段。B 端规范要额外写明：

- **表单**：字段分组（Card 包裹）、组标题清晰、必填标红星、全部 Zod 校验、提交按钮有 loading；
- **表格**：排序、分页、筛选；操作列二次确认（尤其删除）；空状态提示；
- **权限**：所有页面都要考虑权限控制，隐藏无权访问的功能。

团队协作时，把 `.claude/skills/my-project-design/` 提交到 Git，其他人拉取即可用——所有人的 AI 自动加载同一份规范，新人不需要花时间学设计规范。**规范不再口口相传，而是封装进 Skill，AI 自动触发。**

## 9. 本篇验收

- [ ] 能说清 CSS、组件、布局三者的分工；
- [ ] 知道 shadcn/ui 与传统组件库的本质差异；
- [ ] 能在合适场景选择 AI 设计工具（原型 vs 代码 vs 设计稿）；
- [ ] 能根据项目类型选组件库，并在提示词里明确指定；
- [ ] 掌握 Motion / GSAP / Three.js 与纯 CSS 动画的取舍；
- [ ] 能用「风格关键词 + 配色 + 参考 + 场景」描述界面；
- [ ] 理解 UI 与 UX 的分界：好看是皮囊，好用才是灵魂；
- [ ] 能用 `DESIGN_SYSTEM.md` + 项目 Skill 约束 AI 的设计输出。

> **补记**：这一章的落点不是「学会设计」，而是**把设计意图变成可约束的规则**。三条最实用的主线：① 用组件库与 Tailwind 把样式问题变成「填空题」；② 用「风格关键词 + 配色 + 参考 + 场景」替代「好看一点」；③ 用 `DESIGN_SYSTEM.md` + 项目 Skill 把一致性固化下来——这正好呼应第二章「你定规则，AI 执行」。

## 写在最后

下一篇[数据持久化与数据库](6-数据持久化与数据库)让应用「记住」数据：从本地存储到数据库的演进、表结构与增删改查，以及数据库设计的基本方法。
