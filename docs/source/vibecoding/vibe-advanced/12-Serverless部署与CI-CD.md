# Serverless 部署与 CI/CD：让网站永久在线

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_mll089oi-6566edafa4114ef0.jpg

<p>图 1：无服务器部署与 CI/CD 自动化</p>

## 1. 从临时链接到永久在线

代码推到 GitHub、和朋友用分支与 PR 协作了几次，一切看起来顺利——直到朋友说：「你上次用 Tunnel 给我看的链接打不开了，能不能弄个一直能用的？」

第十章的内网穿透链接，关掉终端就失效。需要一个**永久在线**的地址：任何人、任何时候都能打开。这就是「部署」要解决的问题。

**部署的本质**：让云端的电脑替你跑本地跑过的那些命令。你在本地 `pnpm install` 装依赖、`pnpm build` 构建，平台做完全一样的事——只不过它 24 小时不关机，而且有公网地址。

## 2. 零成本起步方案

一套零成本方案 = 一个**部署平台** + 一个**数据库托管**。

**部署平台**负责运行前后端代码，提供 Serverless 环境。Serverless 不是「没有服务器」，而是**你不需要管理服务器**——只管写代码，扩容与维护交给平台。

| 平台 | 国内访问 | 特点 |
|---|---|---|
| 腾讯云 EdgeOne Pages | 快 | 基于腾讯全球边缘网络，连 GitHub 自动部署，对 Next.js 支持好 |
| 阿里云 ESA 函数和 Pages | 快 | 一站式全栈平台，深度集成 Git 工作流 |
| Vercel | 较慢 | Next.js 官方平台，体验极佳，适合海外用户 |
| Cloudflare Pages | 较快 | 全球 CDN，免费额度慷慨，支持边缘函数 |
| GitHub Pages | 一般 | 最简单的静态托管，适合纯前端 |

**数据库托管**推荐 Neon：专注 PostgreSQL 的无服务器云服务，按需扩缩容，标准连接串 + Drizzle ORM，不被厂商捆绑——哪天想换服务商，只改 `DATABASE_URL` 就行，这叫**保持架构独立性**。需要认证、存储、实时订阅等后端能力时可以考虑 Supabase，但尽量用 Drizzle 直连数据库，而不是大量使用其独有 SDK，才能保持迁移灵活性。

## 3. 部署前的三个确认

不是走流程，每一项都有实际原因：

1. **代码已经推到 GitHub**：平台从 GitHub 拉代码，只在本地它拿不到；
2. **本地 `pnpm build` 能成功**：云端做的事和本地一模一样，本地构建不过，云端一定不过；
3. **记录好 `.env` 里的环境变量**：`.env` 被 `.gitignore` 排除，平台拉到的代码里没有数据库密码、认证密钥，需要在平台上手动配置。

## 4. 部署到 EdgeOne Pages

### 4.1 注册并连接 GitHub

打开 [EdgeOne Pages 控制台](https://console.cloud.tencent.com/edgeone/pages)，微信扫码或腾讯云账号登录（首次需实名认证）。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260227003947136.png

<p>图 2：EdgeOne Pages 控制台</p>

点击 **导入 Git** 仓库 → **连接 GitHub**。首次连接会跳到 GitHub 授权页：可以选「授权所有仓库」或「Only select repositories」。选后者、只授权当前项目，这是**最小权限原则**——只给平台需要的权限，以后要部署其他项目再回来加。

授权后选择项目，点下一步。也可以直接上传，平台通常会自动识别项目架构。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260227005154181.png

<p>图 3：选择要部署的仓库</p>

### 4.2 配置构建设置

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260227005228223.png

<p>图 4：构建设置页面</p>

- **框架预设**选 Next.js：平台自动填好构建命令与输出目录；
- **根目录**保持 `./`：代码不在仓库根目录（如 `apps/web/`）才需要改；
- **构建命令** `pnpm build`（或 `npm run build`）：和本地完全一致；
- **输出目录** `.next`：Next.js 构建产物位置；
- **Node 版本**：EdgeOne 通常默认 Node 22，多数项目可直接通过；**只有日志出现 Node 版本不兼容报错时**，再改成和本地一致。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260227005254177.png

<p>图 5：Node 版本设置</p>

### 4.3 配置环境变量

`.env` 没上传 GitHub，所以要在平台手动填。EdgeOne 支持**批量导入**：直接把 `.env` 内容粘贴进去，平台自动解析成键值对。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260227005320209.png

<p>图 6：批量导入环境变量</p>

```bash
DATABASE_URL="postgresql://..."        # 缺了应用连不上数据库
BETTER_AUTH_SECRET="xxx"               # 缺了登录直接 500
BETTER_AUTH_URL="https://你的域名"      # 认证回调地址，先留空，拿到链接后再改
```

不确定需要哪些变量，就打开本地 `.env`，里面有的这里都要有。

> **环境变量修改后需要重新部署**才会生效——只改变量不重新构建，运行中的版本仍用旧值。

### 4.4 选择加速区域（与备案）

国内平台特有的一步，与 ICP 备案有关：

- 选包含「中国大陆」的加速区域 → 需要 ICP 备案（个人可申请，审核约一到两周）；
- 选「全球（不含中国大陆）」→ 不需要备案，部署后立刻可用。虽然名字这么叫，绑定自定义域名后国内通常也能正常访问。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260227005154181.png

<p>图 7：加速区域选择</p>

### 4.5 构建失败怎么排查

正常情况首次部署会直接成功。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302003046402.png

<p>图 8：部署成功记录</p>

遇到红色 **Failed**，先看构建日志——**答案通常在最后几行**。常见报错：

| 报错关键词 | 原因 | 解决 |
|---|---|---|
| `engine "node" is incompatible` | Node 版本不匹配 | 调整 Node 版本后重新部署 |
| `Cannot find module` | 依赖没装上 | 检查 `package.json` 与锁文件是否提交 |
| `DATABASE_URL is not defined` | 环境变量没配 | 在环境变量页面添加 |
| `Build failed` 无具体信息 | 构建命令错误 | 确认与本地一致 |
| `ENOENT: no such file` | 路径错误 | 检查根目录配置 |

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260227005254177.png

<p>图 9：调整 Node 版本后重新部署</p>

看不懂的日志直接复制给 AI，通常足够定位问题。

### 4.6 部署成功

构建成功后平台生成预览链接（如 `xxx.edgeone.dev`）。点开看到「个人豆瓣」和本地一模一样，但这次在公网上——发给朋友，秒开。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302003255608.png

<p>图 10：部署成功后的访问链接</p>

> 选「全球（不含中国大陆）」时，默认域名可能有 3 小时访问限制；**绑定自定义域名后就没有这个限制**（域名配置见下一篇）。

连接 GitHub 后，每次 push 到主分支都会自动重新部署——这就是 CI/CD 的雏形，后面展开。

## 5. 类 Vercel 平台与选择

12.1 学到的概念——连 GitHub、配构建、设环境变量——在任何平台都一样，区别只在界面、免费额度与网络覆盖。

### 5.1 平台速览

| 平台 | 国内访问 | 免费额度 | 特色 | 适合 |
|---|---|---|---|---|
| Vercel | 较慢（需自定义域名） | 100 GB 带宽/月 | Next.js 官方平台，体验最好 | 海外用户、个人项目 |
| Cloudflare Pages | 较快 | 无限带宽 | 全球 CDN、Workers 边缘计算 | 静态优先、可部署 Next.js |
| 阿里云 ESA | 快 | 有免费额度 | 阿里云生态、国内快 | 国内用户 |
| Netlify | 较慢 | 100 GB 带宽/月 | 表单、身份认证内置 | 静态站点、JAMstack |
| Railway | 一般 | $5 额度/月 | 支持数据库与后端服务 | 全栈应用 |

### 5.2 Vercel

Next.js 的创造者与官方推荐平台，部署体验最丝滑。

**网页部署**：打开 [vercel.com](https://vercel.com)，用 GitHub 登录 → **Add New → Project** 选择仓库。Vercel 自动识别 Next.js，大部分配置不用改。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302004336073.png

<p>图 11：Vercel 控制台</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302004314090.png

<p>图 12：导入 Git 仓库</p>

只需两件事：添加环境变量、点击 Deploy。成功后得到 `xxx.vercel.app` 链接。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302004247091.png

<p>图 13：部署配置与 Deploy</p>

**命令行部署**：

```bash
npm i -g vercel
vercel login
vercel deploy       # 首次会询问项目配置
vercel --prod       # 部署到生产环境
```

> `*.vercel.app` 国内访问不稳定，解决方案是绑定自定义域名，再用 Cloudflare 做 DNS 代理加速。

Vercel 的独特能力：

- **预览部署**：每个 PR 自动生成独立预览链接，发给朋友在合并前看效果，与 PR 工作流完美配合；
- **分析面板**：通常在项目接入 Analytics / Speed Insights 后才有完整数据，按控制台与官方文档操作；
- **Edge Functions**：把代码复制到全球 CDN 节点，动态逻辑就近执行，不用所有请求都飞到同一台服务器。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302005118317.png

<p>图 14：PR 的预览部署链接</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302005153216.png

<p>图 15：Vercel 分析面板</p>

### 5.3 Cloudflare Pages

最大卖点是**无限带宽**：免费套餐不限流量，哪怕文章突然爆火也不产生额外费用。

快速体验：访问 `https://workers.new/templates/next-starter-template`，它会自动在你的 GitHub 建仓库、连 Cloudflare Pages、完成首次部署，两分钟看到运行在边缘网络上的 Next.js 应用。

部署已有项目：登录 Cloudflare Dashboard → **Workers & Pages → Create** → 连接 GitHub 选仓库 → 框架预设 Next.js → 添加环境变量 → 部署。步骤与 EdgeOne、Vercel 几乎一模一样。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302005022505.png

<p>图 16：Cloudflare Workers & Pages</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302005442304.png

<p>图 17：连接 Git 仓库</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302005452914.png

<p>图 18：构建设置</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302012941453.png

<p>图 19：部署成功</p>

Cloudflare 还有一整套边缘生态：**Workers**（边缘函数）、**D1**（轻量数据库，现有 Neon 已够用）、**R2**（兼容 S3 的对象存储）。

### 5.4 阿里云 ESA Pages

阿里云用户的国内选择：登录 ESA Pages 控制台 → 创建 Pages 项目并连接 GitHub → 配置构建与环境变量 → 选择加速区域（国内需备案）→ 部署。与 EdgeOne 定位类似，选哪个主要看更熟悉哪家生态。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302013222049.png

<p>图 20：阿里云 ESA Pages</p>

### 5.5 所有平台的核心流程一样

```text
连接 GitHub → 选择仓库 → 配置构建设置 → 添加环境变量 → 部署
```

它们解决同一个问题：把 GitHub 上的代码变成可访问的网站。底层逻辑一样，只是界面、额度、网络覆盖不同。在 EdgeOne 上完整做过一次，换任何平台都能很快上手。

**怎么选**：

- 用户主要在国内 → EdgeOne Pages 或阿里云 ESA；
- 用户主要在海外 → Vercel（Next.js 最佳）或 Cloudflare Pages；
- 用户遍布全球 → Cloudflare Pages（无限带宽 + 全球 CDN）；
- 不确定 → 随便选一个先跑起来，选错也不用改代码，换平台重做一次即可，迁移成本几乎为零。

## 6. CI/CD 与自动化

### 6.1 你已经在用 CI/CD 了

部署成功的第二天，小明修了个 bug 推送到 GitHub，几分钟后打开网站——bug 已经修好了。「我没点部署啊？」

CI/CD 用大白话说就是两件事：**CI（持续集成）**——每次推送自动运行检查与构建；**CD（持续部署）**——检查通过后自动发布上线。你只负责写代码和推送，中间全自动。

连接 GitHub 时，平台注册了一个 **webhook**：每次 `git push`，GitHub 主动通知平台「有新代码了」。像淘宝下单后物流主动发短信，不需要每隔五分钟刷物流页。平台收到通知后自动拉代码、装依赖、构建、发布——这就是「推送即部署」。

### 6.2 平台内置的 CI/CD

EdgeOne Pages、Vercel、Cloudflare Pages 这类平台已内置 CI/CD，连接 GitHub 后自动响应：

- **推送到 main** → 自动部署到生产环境；
- **创建 Pull Request** → 自动构建并生成预览链接。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302013311084.png

<p>图 21：PR 上的自动构建结果</p>

预览链接是 CI/CD 最直观的产物。

### 6.3 什么时候需要自定义流水线

平台内置的「推送即部署」对个人项目够用。项目变复杂后，可能想在部署前多做几件事：自动跑测试、检查代码风格、检查类型错误。朋友一句「加个自动测试吧，免得合并有 bug 的代码」——这正是第九章测试的延续：写了测试但每次手动跑容易忘，放到推送时自动跑就不会遗漏。

**GitHub Actions** 是 GitHub 自带的 CI/CD 工具，用 `.github/workflows/` 下的 YAML 配置，可以理解成一份「菜谱」：

- **触发条件（`on`）**：推送到 main、创建 PR 时运行；
- **运行环境（`runs-on`）**：通常是 `ubuntu-latest`，每次启动一台干净临时虚拟机，用完销毁；
- **执行步骤（`steps`）**：拉代码 → 装依赖 → lint → build → test。

```yaml
name: CI
on:
  pull_request:
  push:
    branches: [main]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
      - run: pnpm install --frozen-lockfile
      - run: pnpm lint
      - run: pnpm build
      - run: pnpm test
```

某一步失败（比如测试没过），流水线停下，PR 页面出现红色 ✗——代码有问题，不应合并。**个人项目大部分不需要 GitHub Actions**；有了协作者、测试覆盖上来后再加。告诉 AI「帮我配置 GitHub Actions，在每次 PR 时自动运行 lint、build 和 test」即可。

### 6.4 分支与部署环境

- **main** 对应生产环境：合并即上线；
- **feature/*** 对应预览环境：每个 PR 自动生成预览链接，只有参与者能看到。

个人项目两层就够了；团队需要 `develop` 等更多环境时，平台都支持配置。

> **渐进式自动化**：不要一开始追求完美流水线。先用平台内置的「推送即部署」，等项目复杂、有协作者、有测试后，再逐步加 GitHub Actions。自动化是为了省事，不是为了折腾。

## 7. 运维基础与成本优化

### 7.1 上线不是终点

上线一周一切正常——直到被问：「你知道有多少人在用吗？出了问题你怎么知道？」**上线不是终点，是运维的起点。**

个人开发者的运维核心就三件事：

1. 知道应用是否正常（有人访问吗？有报错吗？）；
2. 出问题能快速定位（哪里出问题？什么时候开始？）；
3. 变更时降低风险（新功能先小范围测试）。

不需要 24 小时盯屏幕，平台已经做了大部分工作，只需要知道去哪里看。

### 7.2 看懂数据与日志

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302021249583.png

<p>图 22：部署平台的数据分析面板</p>

面板位置：EdgeOne 控制台 → 项目 → 数据分析；Vercel Dashboard → 项目 → Analytics；Cloudflare Dashboard → Workers & Pages → 分析。

某天朋友说「网站打不开，一直转圈」。打开部署日志，看到：

```text
Error: Connection timed out - database not responding
```

原因清楚：数据库连接超时——Neon 免费版数据库因长时间不活跃被暂停，重新激活后恢复。

**日志排查套路**：先找红色 ERROR → 看报错时间点前后的上下文 → 看不懂就把关键信息复制给 AI。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302021359067.png

<p>图 23：部署与运行日志</p>

### 7.3 平台已经帮你做的优化

- **自动扩缩容**：根据实际访问量自动调整资源，不用提前猜「需要几台服务器」；
- **全球 CDN**：静态资源分发到全球节点，就近返回；
- **图片优化**：自动压缩、转 WebP/AVIF，2MB PNG 用户可能只下载 200KB；
- **HTTPS**：自动申请与续期证书。

传统服务器上需要自己配 Nginx、Let's Encrypt、CDN——现在平台全包了。

### 7.4 可以关注的优化

用户反馈「慢」时让 AI 分析性能，常见方向：

- **图片太大**：用 `next/image`，按设备尺寸返回合适大小；
- **接口太慢**：检查数据库缺索引或 N+1 查询；
- **包体积大**：检查是否引入不必要依赖，让 AI 分析 bundle；
- **无缓存**：为不常变化的内容加浏览器与 CDN 缓存。

### 7.5 成本：免费额度够用吗

免费额度是用来体验的，不是长期跑生产的。但**不要在阶段一担心阶段三的问题**：

| 阶段 | 用户量 | 方案 | 月成本 |
|---|---|---|---|
| MVP | < 1,000 | Serverless 免费套餐 | ¥0 |
| 增长期 | 1,000 - 10,000 | 升级付费套餐 | 约 ¥50-200 |
| 规模化 | > 10,000 | 评估托管与自建 | 按业务判断 |

**免费额度陷阱**：

- **Vercel** 免费版单个请求最多跑 10 秒，超时返回 504；调用外部 AI、查大量数据的接口可能不够，可优化性能或升级 Pro（60 秒）；
- **Neon** 免费版计算时长有限，长时间不活跃会暂停，下次访问冷启动慢 1-2 秒；
- **Supabase** 免费版只能建 2 个项目，7 天不活跃会暂停。

**AI 模型成本**可能是最大开销：在 API 平台设置每日/每月消费上限，防止代码 bug（如死循环调用）导致天价账单。还可以做**智能路由**：简单翻译、摘要用便宜的国产模型，复杂推理才用高价模型——某应用用 70% GLM + 30% Claude，API 成本降低 80%。关键是匹配任务复杂度与模型能力，不要用大炮打蚊子。

### 7.6 轻量级灰度发布

个人项目不需要复杂灰度系统，**预览链接 → 测试 → 合并**就是最简单的灰度：

1. 在功能分支开发；
2. 创建 PR，平台自动生成预览链接；
3. 自己和朋友在预览链接测试；
4. 确认没问题，合并到 main，自动部署生产。

新功能上线前至少有人测试过，这就够了。

## 8. 本篇验收

- [ ] 能解释部署的本质与 Serverless 的含义；
- [ ] 会做部署前的三个确认（GitHub、本地构建、环境变量）；
- [ ] 能在 EdgeOne Pages 完成一次部署并处理构建失败；
- [ ] 理解加速区域与 ICP 备案的关系；
- [ ] 能说出主流平台的差异与选择依据；
- [ ] 理解 webhook 与「推送即部署」；
- [ ] 知道何时该加 GitHub Actions，并能读懂一份基础 workflow；
- [ ] 会看分析面板与日志排查问题；
- [ ] 能按阶段做成本决策，避开免费额度陷阱。

> **补记**：这一章的主线是「本地命令搬到云端」——本地 `install → build → start` 对应平台的构建与发布，区别只是 24 小时在线和公网地址。三个最容易踩的坑：**本地构建必须先通过**、**环境变量要在平台单独配置**、**Node 版本要与本地一致**。部署完成后，CI/CD 把「发布」变成 push 的副产品，而运维与成本意识决定它能稳定跑多久。

## 写在最后

下一篇[域名与 DNS](13-域名与DNS)给站点绑上自己的域名：域名注册、DNS 解析、备案与 HTTPS 接入，让地址更专业、更稳定。
