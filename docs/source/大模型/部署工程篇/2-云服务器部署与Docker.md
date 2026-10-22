# 云服务器部署与Docker Compose

## 1. 云服务器准备

本地 `uvicorn --reload` 只适合功能验证，真正对外服务要放到有公网 IP 的云服务器（ECS/VPS，阿里云、华为云、AWS 均可）。新用户/学生常有优惠；学习实验可选**按量计费**，但注意云硬盘、弹性公网 IP 等在关机后仍可能计费，不使用时应及时关机并定期查账单。

**登录凭证两种**：密码登录（简单，适合快速测试）；**SSH 密钥对**（更安全——公钥放服务器、私钥本地保管，连接时匹配验证，全程不传密码，可防暴力破解）。

**防火墙放行端口**（安全组/云防火墙/系统 ufw）：**22**（SSH）、**443**（apt 更新、curl 下载脚本等 HTTPS 访问）、**8000**（FastAPI 监听）。新服务器系统防火墙通常默认关闭，若已开放安全组仍无法访问，需同时检查系统级防火墙。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_2_1.png" width="80%" alt="阿里云 ECS 实例管理页面" />
  <p>图 1：阿里云 ECS 实例管理页面（复制公网 IP）</p>
</div>

以阿里云为例配置安全组：实例页左侧「网络与安全 → 安全组」→ 找到实例绑定的安全组 → 「管理规则」：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_2_2.png" width="80%" alt="安全组列表页面" />
  <p>图 2：安全组列表页面</p>
</div>

「入方向 → 增加规则」：协议类型「自定义 TCP」、端口范围 `8000`、授权对象测试可用 `0.0.0.0/0`（更安全的是仅授权自己的 IP）：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_2_3.png" width="80%" alt="新建安全组入方向规则" />
  <p>图 3：新建安全组入方向规则</p>
</div>

## 2. 连接服务器

三种方式：**控制台网页终端**（VNC，临时操作用）；**标准 SSH 客户端**（Windows 10+ 内置 OpenSSH，或 PuTTY/Xshell；macOS/Linux 直接 `ssh root@<IP>`）；**IDE 远程插件**（VS Code/PyCharm 远程开发）。

演示用免费 SSH 客户端 **FinalShell**（集成监控、文件管理、命令历史）：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_2_4.png" width="80%" alt="FinalShell 官网下载页面" />
  <p>图 4：FinalShell 官网下载页面</p>
</div>

新建连接：文件夹图标打开连接管理器 → 加号「新建连接」→「SSH连接(Linux)」：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_2_5.png" width="80%" alt="FinalShell 连接管理器" />
  <p>图 5：FinalShell 连接管理器</p>
</div>

填写：名称（如「阿里云-NER模型」）、主机（公网 IP）、端口 22、认证方法「密码」、用户名（阿里云 Ubuntu 默认 `root`；其他平台可能是 `ubuntu`/`ec2-user`，以镜像说明为准）、密码：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_2_6.png" width="80%" alt="FinalShell 新建 SSH 连接" />
  <p>图 6：FinalShell 新建 SSH 连接</p>
</div>

双击连接，首次可能提示接受主机密钥，点「接受并保存」：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_2_7.png" width="80%" alt="FinalShell 快速连接" />
  <p>图 7：FinalShell 连接过程</p>
</div>

连接成功进入命令行界面，左侧实时显示 CPU/内存/网络：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_2_8.png" width="80%" alt="FinalShell 命令行界面" />
  <p>图 8：FinalShell 命令行界面</p>
</div>

## 3. uv：更快的依赖管理

`uv` 是 Rust 编写的极速 Python 包安装/解析器（ruff 作者出品），作为 pip/pip-tools/venv 的统一高速替代品，解析安装复杂依赖尤其省时。本次部署全程用它管理虚拟环境与依赖。

### 3.1 pyproject.toml

uv 遵循 PEP 621，支持项目根目录 `pyproject.toml` 集中配置依赖与源，提升可移植性与可复现性。常用 `[tool.uv]` 配置：`native-tls`（用系统 TLS 还是 rustls，网络环境异常时切换可能有帮助）、`index-url`、`extra-index-url`、`no-index`、`find-links`。NER 项目的完整示例：

```toml
[project]
name = "ner-deployment-service"
version = "1.0.0"
dependencies = ["fastapi", "pydantic", "uvicorn", "gunicorn", "numpy<2", "torch==2.2.1"]

[[tool.uv.index]]
name = "aliyun"
url = "https://mirrors.aliyun.com/pypi/simple/"

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
```

多个 `[[tool.uv.index]]` 按出现顺序查找：先阿里云镜像找所有包，找不到再到 PyTorch CPU 索引。这种写法比 `index-url`/环境变量更集中清晰——既享国内镜像速度，又能拿官方特殊源；配置随代码托管，保证团队与 CI/CD 用完全一致的依赖源。

## 4. 配置服务器环境

```bash
sudo apt update && sudo apt upgrade -y       # 更新系统
python3 --version                             # 检查 Python（演示机自带 3.10.12）
sudo apt install -y python3.10                # 如无/过低则安装
pip install uv && uv --version                # 安装并验证 uv
```

> uv 官方也有免 Python 的一键安装 `curl -LsSf https://astral.sh/uv/install.sh | sh`，但演示服务器上遇到网络问题，故改用 pip。

**上传项目**：FinalShell 文件浏览器把本地 `ner_deployment` 拖到用户主目录；`cd ner_deployment`：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_2_9.png" width="80%" alt="FinalShell 文件上传" />
  <p>图 9：FinalShell 拖拽上传项目</p>
</div>

**建虚拟环境**：

```bash
uv venv                      # 生成 .venv
source .venv/bin/activate    # 提示符出现 (ner_deployment)
```

## 5. 部署 NER 应用

### 5.1 安装依赖

```bash
uv pip install -e .          # 可编辑模式：装依赖 + 项目自身
```

> **网络超时**：默认 30 秒超时可能不够，报 `Failed to download distribution due to network timeout` 时用 `UV_HTTP_TIMEOUT=300 uv pip install -e .` 延长到 5 分钟。

### 5.2 启动生产服务

`uvicorn --reload` 默认只监听 `127.0.0.1`，云端外部访问需 `--host 0.0.0.0`，但生产更推荐 **Gunicorn + UvicornWorker** 组合：Gunicorn 负责监听端口、管理多 worker 与日志，Uvicorn 负责异步处理请求。

```bash
gunicorn -w 3 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
```

- `-w 3`：worker 数。Gunicorn 通用推荐 `2*cpu+1`，但异步 UvicornWorker 通常 `cpu+1` 足够（2 核 → 3）；
- `-k uvicorn.workers.UvicornWorker`：用 Uvicorn 工作进程类支持 asyncio；
- `--bind 0.0.0.0:8000`：监听所有网卡，允许外网访问。

看到启动日志说明应用正常，`Ctrl+C` 先停掉。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_2_10.png" width="80%" alt="Gunicorn 启动日志" />
  <p>图 10：Gunicorn 启动日志</p>
</div>

### 5.3 systemd 持久化

终端里跑 gunicorn，SSH 一断服务就停。用 **systemd** 让服务后台常驻、开机自启：

```bash
sudo nano /etc/systemd/system/ner_api.service
```

```ini
[Unit]
Description=NER API Service
After=network.target

[Service]
User=root
Group=root
WorkingDirectory=/root/ner_deployment
ExecStart=/root/ner_deployment/.venv/bin/gunicorn -w 3 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
```

要点：`After=network.target` 等网络就绪再启动；`WorkingDirectory` 启动前切目录；`ExecStart` **务必用虚拟环境里 gunicorn 的绝对路径**；`Restart=on-failure` 失败自动重启；`WantedBy=multi-user.target` 多用户模式启用。

> 示例用 root 仅为演示。生产更推荐创建专用非特权用户（如 `neruser`），把项目目录权限授予它，并改 `User`/`Group`，降低被攻击后的系统风险。

常用管理命令（服务文件已写绝对路径，无需进出虚拟环境）：

```bash
sudo systemctl daemon-reload     # 重载配置
sudo systemctl start ner_api     # 启动
sudo systemctl enable ner_api    # 开机自启
sudo systemctl status ner_api    # 查看状态
sudo journalctl -u ner_api -f    # 实时日志
```

状态显示绿色 `active (running)` 即部署成功：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_2_11.png" width="80%" alt="systemctl status 显示服务正在运行" />
  <p>图 11：systemctl status 显示服务运行中</p>
</div>

### 5.4 测试云端服务

在**自己电脑**的新终端里：

```bash
curl -X POST "http://<服务器公网IP>:8000/predict/ner" -H "Content-Type: application/json" \
  -d "{\"text\":\"患者自述发热、咳嗽，伴有轻微头痛。\"}"
```

收到与本地一致的实体识别 JSON；浏览器访问 `http://<公网IP>:8000/docs` 也能用交互文档：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_2_12.png" width="80%" alt="curl 测试云端服务" />
  <p>图 12：curl 测试云端服务</p>
</div>

## 6. Docker Compose 部署

传统方式虽能跑通，但团队协作与多环境部署时仍会遇到「在我机器上明明是好的」。**Docker** 把应用及其所有依赖（含 Python 解释器）打包成可移植的**容器**，实现「一次构建，随处运行」。

三个核心概念：**镜像（Image）**——只读模板，像「安装包」；**容器（Container）**——镜像的运行实例，与宿主机共享内核、轻量、启动极快；**Dockerfile**——描述如何一步步构建镜像的文本文件。

**Docker Compose** 是官方多容器编排工具：用一个 `docker-compose.yml` 定义所有服务，一条 `docker compose up` 全部拉起。单服务也有收益——配置集中化（构建、端口、重启策略一处声明）、命令简化、未来加数据库只需加几行。

### 6.1 安装 Docker

```bash
sudo apt update
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
docker --version && docker compose version
```

> 国内服务器 `curl` 报 `Connection reset by peer` 时，先确认安全组放行了 **443 出站**；仍不行通常是链路不稳，可换国内镜像安装：`curl -sSL https://get.daocloud.io/docker | sh`。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_3_1.png" width="80%" alt="验证 Docker 安装成功" />
  <p>图 13：验证 Docker 与 Compose 版本</p>
</div>

**（可选）镜像加速**：默认从 Docker Hub 拉取慢，配置阿里云专属加速器（控制台搜「容器镜像服务」→「镜像工具」→「镜像加速器」）：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_3_2.png" width="80%" alt="获取阿里云镜像加速器" />
  <p>图 14：获取阿里云镜像加速器地址</p>
</div>

```bash
sudo mkdir -p /etc/docker
sudo nano /etc/docker/daemon.json
# {"registry-mirrors": ["https://<阿里云镜像ID>.mirror.aliyuncs.com"]}
sudo systemctl daemon-reload && sudo systemctl restart docker
```

（也可用网易公开加速器 `https://hub-mirror.c.163.com`。）

### 6.2 编写 Dockerfile

```dockerfile
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv pip install --system --no-cache .

COPY . .

RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["gunicorn", "-w", "3", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
```

设计要点：`python:3.10-slim` 轻量基础镜像；`ENV` 关输出缓冲、禁 `.pyc` 让日志更可读、镜像更整洁；镜像内持久配置阿里云 PyPI 源；**先复制 `pyproject.toml` 装依赖、再 `COPY . .` 复制代码**——充分利用 Docker 层缓存，依赖不变时不重复安装；创建 `appuser` 非 root 运行降风险；`CMD` 用与 systemd 相同的 Gunicorn 生产启动方式，只是封装在容器内。

### 6.3 编写 docker-compose.yml

```yaml
services:
  ner_api:
    build: .
    container_name: ner_api_service
    ports:
      - "8000:8000"
    restart: always
```

`build: .` 用当前目录 Dockerfile 构建；`container_name` 固定易识别的容器名；`ports` 把宿主机 8000 映射到容器 8000；`restart: always` 异常退出或服务器重启后自动恢复——**单服务场景下基本替代了 systemd 的持久化作用**，更简洁。

### 6.4 构建与启动

上传项目后确认 Docker 服务运行：

```bash
sudo systemctl status docker     # 未启动则 sudo systemctl start docker
```

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_3_3.png" width="80%" alt="检查 Docker 服务状态" />
  <p>图 15：检查 Docker 服务状态</p>
</div>

```bash
cd ner_deployment
sudo docker compose up --build -d
```

- `up` 启动服务；`--build` 强制重建镜像（首次或改了 Dockerfile/代码时）；`-d` 后台运行。

首次会逐条执行 Dockerfile 指令构建镜像再启动容器，耗时数分钟：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_3_4.png" width="80%" alt="Docker Compose 启动成功" />
  <p>图 16：Docker Compose 构建并启动</p>
</div>

### 6.5 管理服务

```bash
sudo docker compose ps           # 查看容器
sudo docker compose logs -f      # 实时日志
sudo docker compose down         # 停止并移除容器与网络
```

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_3_5.png" width="80%" alt="查看服务状态与日志" />
  <p>图 17：查看服务状态与日志</p>
</div>

### 6.6 测试

浏览器打开 `http://<公网IP>:8000/docs`：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_3_6.png" width="80%" alt="FastAPI 交互式 API 文档" />
  <p>图 18：云端 FastAPI 交互式文档</p>
</div>

展开 `/predict/ner` →「Try it out」：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_3_7.png" width="80%" alt="进入接口测试视图" />
  <p>图 19：进入接口测试视图</p>
</div>

输入文本 →「Execute」：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_3_8.png" width="80%" alt="在 API 文档中发起请求" />
  <p>图 20：在文档中发起请求</p>
</div>

页面下方立即显示实体识别 JSON，说明容器化部署成功：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_14_3_9.png" width="80%" alt="成功获取模型预测结果" />
  <p>图 21：成功获取预测结果</p>
</div>

## 7. 两种方案的取舍

- **手动 + uv + systemd**：完整、实用的传统服务器部署法——直接控制宿主机，排查问题直观，适合单机小服务；
- **Docker Compose**：以 `restart: always` 替代 systemd 的持久化，从根本上解决环境不一致，是业界主流方案，也是后续 CI/CD 的天然底座（部署服务器只需 `git pull` + `docker compose up --build -d`）。

> **补记**：这一篇的部署链路是「安全组放行 → SSH 登录 → uv 管依赖 → Gunicorn 托管 → systemd/Docker 保活」。其中最容易踩的还不是命令，而是**端口与防火墙**（安全组 + 系统防火墙两层）和**绝对路径**（systemd 里必须写虚拟环境中 gunicorn 的完整路径）。记住这两点，80% 的「服务起不来/外网访问不到」都能自查。

## 写在最后

服务上线了，但每次改代码都要手动「构建-上传-重启」，一天发版十几次就难以为继——下一篇[Git与GitHub版本控制基础](3-Git与GitHub版本控制基础)从 Git 三大区域讲到远程仓库协作，为自动化流水线打好地基。
