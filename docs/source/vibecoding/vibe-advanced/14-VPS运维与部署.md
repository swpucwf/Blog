# VPS 运维与部署：拿回完全控制权

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_mll08oqx-9a52a9d3a1e83c5d.jpg

<p>图 1：云服务器运维与项目部署</p>

## 1. 为什么要有自己的服务器

网站跑在 Serverless 平台上，域名绑好、HTTPS 绿锁，一切顺利。但心里总有个念头：平台倒了怎么办？想装点平台不支持的东西呢？**想要完全的控制权，就得有自己的服务器。**

在云厂商新用户活动页挑一台香港轻量服务器——免备案、即买即用。下单后拿到三样东西：公网 IP、用户名（`root`）、密码。终端输入 `ssh root@1.2.3.4`，确认指纹、输入密码，黑底白字的 `root@xxx:~#` 出现——像拿到一间空房子的钥匙，接下来要自己装修。

先别急着装东西，**先锁好门窗**。裸奔的服务器在互联网上平均几分钟就会被扫描到：公网 IP 是公开的，上线几分钟就有自动化脚本在挨个试密码。

## 2. VPS 选购

VPS（Virtual Private Server）就是云上租的一台电脑：24 小时开机、有公网 IP，可以远程登录装任何东西。和你自己的电脑唯一区别是——它没有屏幕，只能通过终端操作。

**主流厂商**：

- 国内：阿里云 ECS/轻量应用服务器（生态最全）、腾讯云轻量（新用户优惠大、控制台友好）；
- 海外：AWS（节点多）、DigitalOcean（界面简洁）、搬瓦工（香港/日本节点对国内友好）、RackNerd（性价比高）。

国内厂商首年常见 1-3 折，买前先搜「XX云 新用户优惠」；海外关注黑五等年付促销。

**配置建议**：

| 参数 | 小型 Web 应用 | 中型全栈 |
|---|---|---|
| CPU | 2 核 | 2-4 核 |
| 内存 | 2GB | 4GB+ |
| 硬盘 | 40GB SSD | 60GB+ SSD |
| 带宽 | 3-5Mbps | 5-10Mbps |

**内存比 CPU 重要**：跑 Node.js + 数据库 + 1Panel，2GB 是底线；1GB 装完面板就没什么余量了。

> **带宽的坑**：国内云厂商多为独享带宽，1Mbps 意味着每秒最多传 128KB，加载一张手机照片要好几秒；海外常给共享大带宽（100Mbps-1Gbps），但可能按流量计费。买前看清。

**机房位置**：

| 位置 | 备案 | 国内速度 | 适合 |
|---|---|---|---|
| 中国大陆 | 需要 | 最快 | 正式商业项目 |
| 中国香港 | 免备案 | 快（30-60ms） | 个人项目首选 |
| 新加坡/东京 | 免备案 | 较快 | 东南亚/备选 |
| 美国西海岸 | 免备案 | 慢（150-200ms） | 海外用户 |

**操作系统**：Ubuntu 22.04/24.04 LTS（强烈推荐，社区大、1Panel 官方支持）、Debian 12（更轻量）、CentOS Stream（不推荐，生态萎缩）。生产环境一定选 **LTS**——厂商承诺多年安全补丁；非 LTS 只管 9 个月。

**购买注意**：薅新用户优惠；长期用选包年包月（租房签年约），临时测试选按量计费（住酒店）；看清流量限制与续费价格（首年优惠与续费可能差 3-5 倍）；**别忘了安全组**——很多新手买完装好应用却打不开，十有八九是端口没开放。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302205408945.png

<p>图 2：云服务器购买页面</p>

## 3. 初始化与安全加固

### 3.1 连接服务器

**方法一：云厂商 Web 终端（推荐新手）**：控制台服务器概览 → 远程登录 → 免密登录，浏览器直接打开终端。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-server-console-01.png

<p>图 3：腾讯云轻量服务器控制台</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-web-terminal-02.png

<p>图 4：Web 终端入口</p>

**方法二：本地 SSH**：

```bash
ssh root@你的服务器IP
```

首次连接会提示指纹确认，输入 `yes`，再输密码。看到 `root@xxx:~#` 即成功。密码输入时不显示任何字符（连星号都没有）是正常的安全设计。

**方法三：FinalShell**：带终端界面 + 文件管理器，填 IP、用户名、密码即可。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-finalssh-03.png

<p>图 5：FinalShell 连接配置</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-ssh-success-04.png

<p>图 6：SSH 连接成功</p>

### 3.2 系统更新与常用工具

```bash
apt update && apt upgrade -y
```

就像新电脑装完系统先打补丁：`apt update` 刷新软件列表，`apt upgrade` 安装所有更新。若弹出配置页面按回车保持默认，中途断开重连即可。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-apt-update-05.png

<p>图 7：apt update</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-apt-upgrade-06.png

<p>图 8：apt upgrade</p>

```bash
apt install vim git curl wget htop unzip rsync jq git-lfs lsof nload net-tools dnsutils -y
```

| 工具 | 用途 |
|---|---|
| vim | 终端文本编辑器 |
| git | 拉取/推送代码 |
| curl / wget | 下载文件、测试接口 |
| htop | 查看 CPU/内存占用 |
| unzip | 解压压缩包 |
| rsync | 高效同步与备份 |
| jq | JSON 格式化 |
| lsof | 查看端口占用 |
| nload | 实时网络流量 |
| net-tools | ifconfig/netstat |
| dnsutils | dig/nslookup |

### 3.3 安装 1Panel

```bash
curl -sSL https://resource.fit2cloud.com/1panel/package/quick_start.sh -o quick_start.sh && bash quick_start.sh
```

安装时设置：面板端口改不常见端口（如 `29876`）、用户名不要用 `admin`、设置强密码。完成后终端显示访问地址：`http://IP:端口/安全入口`。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-1panel-install-07.png

<p>图 9：1Panel 安装过程</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-1panel-install-10.png

<p>图 10：安装完成</p>

> **记住安全入口**：访问地址含随机路径（如 `/abc123`），别人知道 IP 和端口也打不开面板。

### 3.4 配置安全组

安全组是云厂商在机房门口设的保安：默认只放行 SSH（22），其他端口全锁。装了新应用要用新端口，得先去控制台「跟保安说一声」。

| 端口 | 用途 |
|---|---|
| 22 | SSH 远程连接 |
| 80 | HTTP |
| 443 | HTTPS |
| 面板端口 | 1Panel 访问 |

腾讯云叫「防火墙」，阿里云叫「安全组」。**安全组与系统防火墙是两道门，都要开**——很多新手只开一道，怎么都访问不了。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-security-group-14.png

<p>图 11：添加安全组规则</p>

配置后打开 1Panel 登录页：

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-1panel-login-11.png

<p>图 12：1Panel 登录页</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-1panel-dashboard-12.png

<p>图 13：1Panel 主界面</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-3-1-monitor-10.png

<p>图 14：容器监控面板</p>

### 3.5 Docker 与镜像加速

安装 Docker 时 1Panel 会引导一键安装，并提示是否自动配置镜像加速（按 y）。**国内必做**：不配加速，拉镜像大概率卡在「拉取镜像」直到超时。

### 3.6 安全加固（必做项，不是可选项）

「我的服务器又没什么重要数据，谁会来攻击？」打开 1Panel 日志，上线不到一小时已有几百次来自世界各地的 SSH 登录失败——自动化脚本不管你是谁，只要端口开着就会来试。

**启用防火墙**：`系统 → 防火墙`，放行 80、443、面板端口等。规则要同时加在 1Panel 与云厂商安全组。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-firewall-21.png

<p>图 15：1Panel 防火墙配置</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-firewall-rules-22.png

<p>图 16：防火墙规则列表</p>

**修改 SSH 端口**：默认 22 是扫描重点。`系统 > SSH 管理` 改为不常见端口（如 `24324`），然后在安全组和防火墙里**都开放新端口**，用新端口测试成功后再关 22。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-ssh-port-19.png

<p>图 17：修改 SSH 端口</p>

> **先测试再关旧端口**：新端口没验证成功就关 22，会把自己锁在门外。

**Fail2Ban**：自动拉黑，连续输错密码 5 次封 IP。`工具箱 > Fail2Ban` 一键安装，记得把监听端口改成新 SSH 端口。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-fail2ban-24.png

<p>图 18：安装 Fail2Ban</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-fail2ban-25.png

<p>图 19：Fail2Ban 配置</p>

**SSH 密钥登录**：本地生成密钥，把公钥上传到服务器，确认密钥可登录后再关闭密码认证。

```bash
ssh-keygen -t rsa -b 4096
cat ~/.ssh/id_rsa.pub
```

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-ssh-keygen-15.png

<p>图 20：生成 SSH 密钥</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-ssh-key-mgr-16.png

<p>图 21：1Panel 授权密钥</p>

> **先测试再禁用密码**：务必先用新终端确认密钥登录成功，再关闭密码认证；否则配置有误只能靠云厂商 Web 终端进去修。

**WAF（可选但推荐）**：绑定域名后，`高级功能 → WAF` 开启访问频率限制，自动拦截扫描、爆破、SQL 注入等恶意请求，不影响正常访问。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-2-waf-28.png

<p>图 22：1Panel WAF 配置</p>

**日常维护**：首页仪表盘看负载与磁盘；清理 Docker 垃圾用 `docker system prune -f`；系统更新 `apt update && apt upgrade -y`。

## 4. Docker 与 1Panel 应用商店

### 4.1 Docker 是什么

- **镜像（Image）**：应用商店里的安装包，含所有文件与配置；
- **容器（Container）**：安装包运行起来的实例，同一个镜像可以跑多个，互不影响；
- **仓库（Registry）**：应用商店，Docker Hub 有几十万个现成镜像。

不用 Docker 就要手动装 PostgreSQL、配置 Redis、调 OpenResty——版本冲突、依赖缺失是家常便饭。有了 Docker，每个应用在自己的「集装箱」里，装错删掉重来，不弄脏系统。

**铁律：永远不要直接在服务器上写代码。** 服务器是跑代码的生产环境，写代码和调试在本地完成，通过 Git 同步。

### 4.2 应用商店

常用应用：PostgreSQL（项目必装）、Redis（缓存/会话）、OpenResty（Web 服务器/反向代理，绑域名必装）、Node.js（跑 Next.js 等项目）。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-3-1-appstore-01.png

<p>图 23：1Panel 应用商店</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-3-1-app-install-02.png

<p>图 24：安装 PostgreSQL 配置页</p>

> 安装时设置的密码一定要记下来，后面配 `.env` 要用。

以 Halo 博客为例，应用商店的典型流程是：选择配置 → 等待安装 → 初始化设置 → 开始使用。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-3-1-halo-db-04.png

<p>图 25：Halo 数据库选择</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-3-1-halo-init-05.png

<p>图 26：Halo 初始化</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-3-1-halo-user-06.png

<p>图 27：创建管理员账号</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-3-1-halo-admin-07.png

<p>图 28：管理后台</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-3-1-halo-frontend-08.png

<p>图 29：前台页面</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-3-1-halo-plugin-09.png

<p>图 30：插件管理</p>

### 4.3 DBMS vs Database

装完 PostgreSQL 还要在「数据库 - PostgreSQL」新建一个具体数据库——**DBMS 是软件（Excel），Database 是文件（.xlsx）**。装一次 PostgreSQL 可以建无数个独立数据库供不同项目使用。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-3-1-db-conn-03.png

<p>图 31：数据库连接信息</p>

### 4.4 容器管理与数据持久化

`应用商店 - 已安装` 管理商店应用（重建、重启、参数）；`容器` 管理所有容器（终端、日志、监控、删除）。容器像临时工位，走了就清空；**Volume 是专属文件柜**，换多少次工位文件都在。1Panel 安装的应用默认配置 Volume，数据在 `/opt/1panel/` 下。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-3-1-container-list-20.png

<p>图 32：容器列表</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-3-1-container-log-11.png

<p>图 33：容器日志</p>

> 日志是应用最好的朋友：90% 的问题日志里都有答案。先看 Error，再看时间点前后的上下文。

### 4.5 容器网络（最大的坑）

`localhost` 在容器里指容器自己，不是服务器主机。你在家喊「妈」，你妈会应；在朋友家喊「妈」，应的是朋友的妈。

| 场景 | 地址 | 原因 |
|---|---|---|
| 容器 A 连容器 B | **容器名**（如 `1Panel-postgresql-ukow`） | 同一 Docker 网络内容器名即主机名 |
| 主机连容器 | `127.0.0.1` 或 `localhost` | 端口已映射到主机 |
| 外部浏览器 | 服务器公网 IP + 映射端口 | 端口映射转发 |

```bash
# 正确
DATABASE_URL="postgresql://用户:密码@1Panel-postgresql-ukow:5432/数据库名"
# 错误（容器里的 localhost 是它自己）
DATABASE_URL="postgresql://用户:密码@localhost:5432/数据库名"
```

Redis 同理：有密码 `redis://:密码@容器名:6379`，无密码 `redis://容器名:6379`。

> 填错连接地址是新手部署失败的头号原因，报错常是 `Connection refused`，看起来像数据库没启动，其实只是地址填错。**容器找邻居用名字，主机找容器用 localhost。**

## 5. 部署 Next.js 应用

### 5.1 上传代码与端口映射

先把代码放到服务器：`git clone` 或 1Panel 文件管理上传压缩包。公开仓库可省去配置凭证（敏感信息应放环境变量）。

端口映射：容器隔离，外部访问不到容器内的 3000，需要把服务器外部端口（如 3001）转发到容器 3000：

```text
用户访问 服务器IP:3001 → 转发到 容器内部:3000 → Next.js
```

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260303151924020.png

<p>图 34：端口映射示意</p>

**内存不足**（1-2GB 机器构建时 `JavaScript heap out of memory`）可以创建 Swap：

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

1GB 内存建议 2GB Swap；2GB 建议 1-2GB；4GB 以上通常不需要。

### 5.2 创建运行环境

1Panel 的「运行环境」本质是一个 Docker 容器，填的表单最终会变成 `docker-compose.yml` 参数。`网站 > 运行环境 > 创建`：

- 名称：`my-nextjs-app`
- 项目目录：代码路径
- Node 版本：与本地一致
- 启动命令：`git pull && pnpm build && pnpm start`（自定义模式）
- 容器名称：用于容器间通信

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302235637992.png

<p>图 35：运行环境基础配置</p>

**端口映射**：高级配置 → 添加 → 主机端口 3001、容器端口 3000。记得在安全组开放 3001。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260303151929448.png

<p>图 36：端口映射配置</p>

**启动命令**：`pnpm start` 只负责启动，要求已构建；用自定义模式串起拉代码、构建、启动。依赖安装建议上传代码后 SSH 提前做一次，不必每次重启都装。

> **生产环境永远用 `start`，不要用 `dev`**——dev 开热更新与调试工具，性能差且不安全。

### 5.3 环境变量与启动

环境变量区域逐行添加，主机名用容器名：

```bash
DATABASE_URL="postgresql://用户名:密码@1Panel-postgresql-xxxx:5432/数据库名"
REDIS_URL="redis://:密码@1Panel-redis-xxxx:6379"
BETTER_AUTH_SECRET="你的密钥"
BETTER_AUTH_URL="http://你的IP:3001"
NODE_ENV="production"
```

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302235711891.png

<p>图 37：环境变量配置</p>

点击确认后，1Panel 依次拉代码、构建、启动。2GB 机器首次构建可能要几分钟。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260302235748925.png

<p>图 38：构建与启动日志</p>

浏览器访问 `http://服务器IP:3001` 看到应用，注册、写数据、刷新，数据还在——前端、后端、数据库都在自己服务器上协同工作。

### 5.4 更新部署与排查

启动命令里已有 `git pull`，更新时在「运行环境」列表点「重启」即可；新增依赖时才需要 SSH 上去 `pnpm install`。彻底重建用「重建」，持久化数据保留。

| 现象 | 原因 | 解决 |
|---|---|---|
| 启动后立刻退出 | 环境变量缺失 | 看日志，补全变量 |
| 页面打不开 | 安全组没开端口 | 控制台开放端口 |
| 数据库连接失败 | 主机名填错 | 用容器名替代 localhost |
| 构建失败 | 内存不足/Node 版本不匹配 | 建 Swap 或对齐 Node 版本 |

## 6. 静态站点与前后端分离（简述）

**静态网站**：本地 `pnpm build` → 1Panel 创建网站（选静态）→ 上传 `dist` 产物 → 绑定域名。SPA 需要把未知路径回退到 `index.html`，并给带哈希的静态资源设置长缓存、`index.html` 短缓存。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-3-3-static-site-01.png

<p>图 39：静态站点配置</p>

**前后端分离**：部署数据库 → 部署后端服务（独立运行环境/容器）→ 构建并部署前端 → 配置**反向代理**（把 `/api` 转发到后端容器）→ 更新前端 API 地址。不同技术栈差异主要在启动命令与端口：Node/Next.js 用 `pnpm start`，Python/FastAPI 用 `uvicorn`，PHP 用 OpenResty + PHP 环境，Java 用 `java -jar`，Go 编译后跑二进制，.NET 用 `dotnet`。

## 7. 域名与证书

应用跑起来了，但 `http://IP:3001` 太丑。这次绑定域名和第十三章不同：**VPS 用 A 记录直拨 IP，Serverless 用 CNAME 转接**。

**第一步：DNS A 记录**：域名服务商 DNS 管理添加两条 A 记录，`@` 与 `www` 都指向服务器公网 IP。

| 类型 | 主机记录 | 记录值 | TTL |
|---|---|---|---|
| A | `@` | 服务器公网 IP | 600 |
| A | `www` | 服务器公网 IP | 600 |

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-4-dns-root-01.png

<p>图 40：根域名 A 记录</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-4-dns-www-02.png

<p>图 41：www 记录</p>

**第二步：创建网站**：在 1Panel 创建网站并填主域名，等于告诉 OpenResty「收到这个域名的请求交给我」。只做 DNS 不建网站，请求到了服务器没人接；只建网站不做 DNS，用户找不到服务器。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-4-openresty-03.png

<p>图 42：安装 OpenResty</p>

确认安全组开放 80/443，访问 `http://域名` 能看到网站即成功。DNS 生效需要几分钟到几小时，用 `nslookup`/`ping`/`dig` 检查。

**第三步：申请 Let's Encrypt 证书**：`网站 > 证书 > 申请证书`，填域名，验证方式选 HTTP（需 80 端口可访问），开启自动续签。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-4-http-insecure-05.png

<p>图 43：HTTP 的「不安全」提示</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-4-ssl-apply-06.png

<p>图 44：申请 SSL 证书</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-4-ssl-http-verify-07.png

<p>图 45：HTTP 验证配置</p>

**第四步：启用 HTTPS 并强制跳转**：网站 HTTPS 设置里选择证书，HTTP 选项选「访问 HTTP 自动跳转到 HTTPS」。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-4-ssl-success-08.png

<p>图 46：证书已签发</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_14-4-https-enable-09.png

<p>图 47：启用 HTTPS</p>

**第五步：自动续期**：Let's Encrypt 证书 90 天有效，勾选自动续签后 1Panel 会处理。续期失败常见原因：80 端口被占用/封禁、DNS 不再指向本机、防火墙关闭了 80。

**泛域名**：一张 `*.yourdomain.com` 覆盖所有子域名，但必须用 **DNS 验证**——在「证书 > DNS 账户」填入阿里云 AccessKey、Cloudflare API Token 或腾讯云 SecretId/SecretKey，申请时选 DNS 验证。

| 证书类型 | 覆盖 | 验证方式 | 适合 |
|---|---|---|---|
| 单域名 | `yourdomain.com` | HTTP/DNS | 一个域名 |
| 多域名 | 指定几个 | HTTP/DNS | 2-3 个 |
| 泛域名 | `*.yourdomain.com` | 仅 DNS | 子域名很多 |

## 8. 服务器上的「自留地」

有了自己的服务器，像有了一块自留地——除了种主粮（部署项目），还能种点有趣的东西。

**Umami**：隐私友好的网站统计，Google Analytics 的开源替代。应用商店一键安装，绑定已装的 PostgreSQL；在后台添加网站会生成追踪脚本，加到网站 `<head>` 即可。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260303152245410.png

<p>图 48：Umami 安装配置</p>

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260303152328354.png

<p>图 49：Umami 统计面板</p>

**n8n**：开源自动化工作流平台（类似 Zapier 但可自托管）。拖一个「定时触发」节点，再拖一个「GitHub」节点连起来，就能实现「每天 9 点检查 Issue 并发邮件」——不用写代码。

https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/vibe_image-20260303152359300.png

<p>图 50：n8n 工作流编辑器</p>

**Bitwarden**：开源密码管理，自托管替代 1Password。应用商店里的实际是 **Vaultwarden**（轻量 Rust 实现），功能完整、资源占用更少。

**Alist**：把阿里云盘、百度网盘、OneDrive 与服务器本地文件统一到一个页面浏览。安装后在容器终端执行 `./alist admin random` 生成管理员密码。

| 应用 | 用途 | 推荐度 |
|---|---|---|
| Umami | 网站统计 | 强烈推荐 |
| Bitwarden | 密码管理 | 按需 |
| n8n | 自动化工作流 | 按需 |
| Alist | 文件管理 | 按需 |

## 9. SRE 意识与日志

部署上线只是开始。**SRE（站点可靠性工程）** 的核心是用软件工程思维解决运维问题：监控（日志与统计了解状态）、自动化（Docker 一键部署、Git 一键回滚）、容灾（备份与恢复）。传统运维靠人熬夜修故障，SRE 靠自动化系统保持稳定——像把手动浇花升级成自动灌溉，不用每天惦记，但要定期检查系统本身。

看日志只看三个关键词：**Error**（具体报错）、**Warning**（值得检查）、**自己写的日志**（定位执行到哪一步）。常见模式：找不到模块 → 依赖没装好；连接被拒绝 → 地址错或没启动；超时 → 网络或逻辑问题。日志可能含敏感信息，别把连接串截屏发到公开地方。

技术在创造之外，更是长期守护——**部署上线不是结束，而是责任的开始**。

## 10. 本篇验收

- [ ] 能按需求选择云厂商、配置、机房与 LTS 系统；
- [ ] 会用 Web 终端/SSH/FinalShell 连接服务器；
- [ ] 完成系统更新、常用工具与 1Panel 安装；
- [ ] 理解安全组与系统防火墙是两道门；
- [ ] 能完成防火墙、SSH 端口、Fail2Ban、密钥登录、WAF 加固；
- [ ] 能解释镜像/容器/仓库、Volume 持久化与端口映射；
- [ ] 记住「容器找邻居用名字，主机找容器用 localhost」；
- [ ] 能部署 Next.js 并处理 Swap、环境变量、构建失败；
- [ ] 能用 A 记录 + 1Panel 网站 + Let's Encrypt 完成 HTTPS；
- [ ] 能按需安装 Umami/n8n/Vaultwarden/Alist，并建立日志与 SRE 意识。

> **补记**：这一章信息量最大，但可以压成四层：**买（配置与机房）→ 锁（安全加固）→ 跑（Docker/1Panel/端口）→ 接（域名与证书）**。最容易卡住的三处：安全组与系统防火墙两道门、容器网络里的 `localhost`、内存不足导致构建失败。把这三处记住，VPS 部署的 80% 问题都能自行排查。

## 写在最后

下一篇[SEO 与分析](15-SEO与分析)让更多人发现网站：Open Graph 分享卡片、搜索引擎优化、Umami 统计与合规基础。
