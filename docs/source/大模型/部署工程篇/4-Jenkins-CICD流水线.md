# Jenkins CI/CD流水线

## 1. Jenkins 入门

Git 管代码版本、远程仓库集中托管，**Jenkins 负责在代码变更后自动完成构建、测试与部署**，把「提交 → 构建 → 上线」串成可观测、可配置的流水线。虽然 GitHub Actions、GitLab CI 等托管平台自带 CI/CD，但很多企业内部仍选择自建 Jenkins，作为统一自动化平台管理多个仓库、对接内部系统、在私有网络完成构建部署。

Jenkins 定位「开源自动化服务器」：

- **开源、跨平台**：Java 开发，可跑在 Linux/Windows/macOS；
- **插件生态丰富**：构建工具、版本管理、消息通知、云平台几乎都有插件；
- **易于扩展**：既能图形化配置「自由风格项目」，也能用 Jenkinsfile 写「流水线脚本」。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_2_1.png" width="80%" alt="Jenkins 官网" />
  <p>图 1：Jenkins 官网</p>
</div>

Logo 是个服务员，角色也类似：研发提交代码后，Jenkins 通过 **Webhook 或定时任务**自动拉最新代码，在构建节点完成编译、测试、打包镜像，再部署到测试/准生产/生产环境；任一环节失败（如单测不过），团队可在控制台第一时间看日志与原因。**GitHub 解决「代码管理 + 托管」，Jenkins 把这些代码变成「可运行的服务」并自动部署到目标环境。**

为贴近企业「代码仓库 + CI/CD 平台 + 业务服务」的分层架构，需要**单独一台服务器部署 Jenkins**（企业里通常还有独立代码托管服务器；此处以 GitHub 作远程仓库）。只有一台服务器时，也可用 GitHub/Gitee 自带的 CI/CD 完成实践。

## 2. 安装部署

```bash
sudo apt update && sudo apt upgrade -y
java -version        # 检查 Java
```

**Jenkins 依赖 Java，2024 年后的 LTS 至少要求 JDK 17**，而很多发行版默认是 JDK 8/11，需手动升级：

```bash
sudo apt install -y openjdk-17-jdk
java -version        # 出现 openjdk version "17.x" 即满足要求
```

从 [jenkins.io/download](https://www.jenkins.io/download/) 选择安装方式：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_2_2.png" width="80%" alt="Jenkins 下载页面" />
  <p>图 2：Jenkins 下载页面</p>
</div>

Ubuntu/Debian 按官方指引：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_2_3.png" width="80%" alt="Jenkins Linux 安装指引" />
  <p>图 3：Jenkins Linux 安装指引</p>
</div>

```bash
sudo mkdir -p /etc/apt/keyrings
sudo wget -O /etc/apt/keyrings/jenkins-keyring.asc https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key
echo "deb [signed-by=/etc/apt/keyrings/jenkins-keyring.asc] https://pkg.jenkins.io/debian-stable binary/" | sudo tee /etc/apt/sources.list.d/jenkins.list > /dev/null
sudo apt update && sudo apt install jenkins -y

sudo systemctl enable jenkins && sudo systemctl start jenkins
sudo ss -ntlp | grep 8080    # 确认 Java 进程监听 0.0.0.0:8080
```

浏览器访问 `http://<服务器IP>:8080/` 进入初始化页；若设置了安全组/ufw，需放行 8080。

### 2.1 首次访问与插件

1. **输入初始密码**：`sudo cat /var/lib/jenkins/secrets/initialAdminPassword`，复制到浏览器解锁：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_2_4.png" width="80%" alt="Jenkins 解锁界面" />
  <p>图 4：Jenkins 解锁界面</p>
</div>

2. **安装插件**：入门推荐「安装推荐插件」。插件托管在国外，网络不佳时可能个别失败——先忽略，后续在插件管理重装即可。

3. **创建管理员**：填写用户名/密码/邮箱，或用默认 `admin` 继续：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_2_5.png" width="80%" alt="创建管理员用户" />
  <p>图 5：创建管理员用户</p>
</div>

4. 确认访问 URL（保持默认），保存并完成，进入主控制台：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_2_6.png" width="80%" alt="实例配置" />
  <p>图 6：实例配置</p>
</div>

### 2.2 常见管理问题

- **改管理员密码**：右上角用户名 → Security → 输入两遍新密码保存；或 Manage Jenkins → Manage Users 找到对应用户改密码。只有完全登不进去时，才需在服务器上临时关闭安全认证或用 Groovy 脚本重置（按官方文档操作）。
- **插件源慢**：Manage Jenkins → 插件管理 → Advanced Setting 的「升级站点」，把 `https://updates.jenkins.io/update-center.json` 换成国内镜像；或从镜像/同事实例下载 `.hpi`/`.jpi`，在「部署插件」处上传。
- **8080 被占用**：apt 安装方式改 `/etc/default/jenkins` 的 `HTTP_PORT=8080`（如 8090）后 `sudo systemctl daemon-reload && sudo systemctl restart jenkins`；RPM 安装则改 `/etc/sysconfig/jenkins`。

## 3. 准备示例项目与仓库

目标：**让 FastAPI + Docker 的 NER 服务，在代码提交到 GitHub 后被 Jenkins 自动拉取、构建并更新线上容器**。

### 3.1 复用 ner-demo 仓库

不新建仓库，直接把第 14 章的 `ner_deployment` 迁移进已有 `ner-demo`（保留其 Git 历史与远程配置），提交推送：

```bash
cd ner-demo
git add .
git commit -m "feat: add NER service deployment demo"
git push origin main
```

此后 GitHub 上的 `ner-demo` 就是「单一事实源」，包含源码、Dockerfile 与 docker-compose.yml。

> 实际项目里体积大的模型权重与数据一般不直接进 Git，而放对象存储/模型仓库并在部署阶段拉取；示例为「一次复制即可跑通」，才把 `checkpoints/` 一并放进仓库。

### 3.2 部署服务器改用 Git 拉代码

之前是 FinalShell 手工上传目录。为与流水线对齐，改为在部署服务器上 clone 仓库：

```bash
cd /root
git clone https://github.com/your-username/ner-demo.git
cd ner-demo
sudo docker compose up --build -d
```

> `git clone` 访问 GitHub 很慢时，可为 github.com 临时指定可用 IP（写入 `/etc/hosts`），IP 可能变化，需自行查询替换。

此后手工更新就是 `git pull origin main && sudo docker compose up --build -d`——接下来让 Jenkins 自动执行这一串。

## 4. 配置流水线项目

### 4.1 检查插件

流水线要做三件事：从 GitHub 拉代码、SSH 登录部署服务器、在服务器上执行 `git pull` + `docker compose up --build -d`。安装推荐插件时通常已含 **Git** 与 **Pipeline**；还需确认 **SSH Agent Plugin**（Manage Jenkins → 插件管理 → Available plugins 搜 `SSH Agent`）。

### 4.2 配置 SSH 凭据

让 Jenkins 免密登录部署服务器，用 SSH 密钥对：

```bash
# Jenkins 服务器上生成专用密钥
ssh-keygen -t ed25519 -C "jenkins-deploy"
# 公钥追加到部署服务器的 authorized_keys
cat ~/.ssh/id_ed25519.pub
echo "<公钥内容>" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
```

`-t ed25519` 用现代椭圆曲线算法，`-C` 是备注，便于日后识别「这把钥匙是 Jenkins 部署用的」。

Jenkins 里添加凭据：Manage Jenkins → 凭据管理 → 全局旁下拉「添加凭据」：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_2_7.png" width="80%" alt="添加凭据" />
  <p>图 7：添加凭据入口</p>
</div>

填写：类型 **SSH Username with private key**；ID 填 `ner-deploy-ssh`（流水线脚本引用用）；描述可选；Username 填部署服务器登录用户（如 `root`）；Private Key 选 **Enter directly**，粘贴**私钥** `id_ed25519` 内容（不是 `.pub`）；生成时设了密码则填 Passphrase，否则留空：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_2_8.png" width="100%" alt="填写凭据信息" />
  <p>图 8：填写凭据信息</p>
</div>

> 配置后先在 Jenkins 服务器终端手测 `ssh -i ~/.ssh/id_ed25519 root@<部署服务器IP>`，能登录说明通道打通（首次提示 `yes` 确认）。

### 4.3 创建 Pipeline 任务

Jenkins 首页「新建 Item」→ 名称如 `ner-deployment-pipeline` → 类型选**流水线** → 确定：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_2_9.png" width="80%" alt="创建流水线任务" />
  <p>图 9：创建流水线任务</p>
</div>

任务配置里找到「流水线」区域：Definition 选 **Pipeline script from SCM**；SCM 选 **Git**；Repository URL 填 `https://github.com/your-username/ner-demo.git`（私有仓库走 SSH 则填 `git@github.com:...` 并配凭据）：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_2_10.png" width="100%" alt="配置 SCM 与仓库地址" />
  <p>图 10：配置 SCM 与仓库地址</p>
</div>

Branches to build 填 `*/main`，脚本路径保持默认 `Jenkinsfile`，保存。

## 5. 编写 Jenkinsfile

在 `ner-demo` 根目录创建 `Jenkinsfile`（Groovy）：

```groovy
pipeline {
    agent any

    environment {
        DEPLOY_HOST = 'root@<部署服务器公网 IP>'
        DEPLOY_DIR  = '/root/ner-demo'
        BRANCH_NAME = 'main'
    }

    stages {
        stage('拉取代码') {
            steps { checkout scm }     // 与 "Pipeline script from SCM" 配置一致
        }

        stage('基础检查') {
            steps {
                echo "工作空间内容如下："
                sh 'pwd && ls -lah'
            }
        }

        stage('部署到 Docker 服务器') {
            steps {
                sshagent(credentials: ['ner-deploy-ssh']) {
                    sh '''
set -e
ssh -o StrictHostKeyChecking=no "$DEPLOY_HOST" << EOF
set -e
cd "$DEPLOY_DIR"
echo "[部署机] 拉取最新代码..."
git pull origin "$BRANCH_NAME"
echo "[部署机] 使用 Docker Compose 构建并重启服务..."
sudo docker compose up --build -d
EOF
                    '''
                }
            }
        }
    }
}
```

三段各司其职：**拉取代码**（`checkout scm`）；**基础检查**（打印工作空间路径与文件列表，便于看日志）；**部署到 Docker 服务器**（借 `sshagent` 用 `ner-deploy-ssh` 凭据 SSH 登录远端，执行 `git pull` 与 `docker compose up --build -d`）。

> 模板需按实际环境改：`DEPLOY_HOST` 换成真实地址与用户名（如 `ubuntu@your-server-ip`）；`DEPLOY_DIR` 对应部署机上项目路径；`BRANCH_NAME` 与部署分支一致（`main`/`master`）；`sshagent` 的凭据 ID 与 Jenkins 中创建的一致。

提交这份 Jenkinsfile，Jenkins 构建时才能读到最新脚本：

```bash
git add Jenkinsfile
git commit -m "chore: add Jenkins pipeline"
git push origin main
```

## 6. 自动触发

目前只能手动「立即构建」，要成为真正的 CI/CD 还需自动触发，两种方式：

### 6.1 轮询 SCM（简单）

任务配置 → Triggers → 勾选「轮询 SCM」，日程表填：

```text
H/5 * * * *
```

即每 5 分钟检查远程仓库是否有新提交，有则触发。优点是配置简单、无需在 GitHub 额外设置；缺点是有延迟，频繁轮询增加双方负载。

### 6.2 GitHub Webhook（实时）

确认装了 **GitHub 插件**，Triggers 勾选「GitHub hook trigger for GITScm polling」；GitHub 仓库 Settings → Webhooks → Add webhook：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_2_11.png" width="80%" alt="添加 Webhook" />
  <p>图 11：添加 Webhook</p>
</div>

Payload URL 填 `http://<Jenkins服务器IP或域名>:8080/github-webhook/`，Content type 选 `application/json`，触发事件选 **Just the push event**：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_2_12.png" width="100%" alt="Webhook 配置详情" />
  <p>图 12：Webhook 配置详情</p>
</div>

保存后，每次 push 都会通知 Jenkins 立即轮询并触发构建。至此整条链路打通：**本地提交 → 推送 GitHub → Webhook 通知 Jenkins → Jenkins SSH 到部署服务器执行 `git pull` + `docker compose up --build -d`**。

### 6.3 最小改动验证

把 `main.py` 根路径返回文案临时改掉（如「欢迎使用命名实体识别 (NER) API」→「不欢迎使用 API」），`git add . && git commit -m "chore: tweak NER welcome message" && git push origin main`；等待轮询或 Webhook 触发构建，在 Jenkins 控制台确认构建成功、容器已重启；浏览器访问 `http://<部署服务器公网IP>:8000/` 看到新文案，说明流水线完整跑通。

## 7. 练习

自行查阅文档，把 GitHub 上的 `ner-demo` 改为私有仓库，并在 Jenkins 中改用 SSH（Git）方式拉取代码，重新触发构建验证流水线仍正常。

> **补记**：Jenkins 流水线的本质是三段「信任链」——Jenkins 信任 GitHub（SCM 配置 + 凭据）、Jenkins 信任部署机（SSH 密钥对）、部署机信任镜像（Docker 构建）。排障时按链路逐段验证：先手动 `git pull` 能否拉、再手动 `ssh` 能否登、最后看 `docker compose logs`。把「手动能跑通」的每一步原样搬进 Jenkinsfile，是让流水线一次成功的最稳路径。

## 写在最后

「部署工程篇」四篇收尾：FastAPI 服务化、云服务器与容器化上线、Git 版本控制、Jenkins 自动化流水线。下一系列「大模型安全」转向模型可信与风险——从安全全景与威胁建模开始。
