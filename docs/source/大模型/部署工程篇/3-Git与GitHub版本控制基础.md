# Git与GitHub版本控制基础

## 1. DevOps 与 CI/CD

项目代码天天更新，每次改完手动「本地构建 → 推服务器 → 停旧容器 → 起新容器」低效且易错——一天发版十几次、几十台服务器时更难，误操作后回滚也痛苦。企业通常建设 **CI/CD 平台**（Git + Jenkins）把开发与运维打通：代码一提交，自动构建、测试、部署。

软件开发模式的演进解释了 DevOps 的由来：

- **原始时代（1950s-60s）**：个人经验驱动，写完就改，高度依赖个人；
- **瀑布模型（1970s-90s）**：需求→设计→编码→测试→维护严格分阶段，规范但僵化，改错成本极高；
- **敏捷开发（1990s-2000s）**：小步快跑、迭代交付，每个 Sprint 都产出可用软件；
- **DevOps（2000s-至今）**：打破开发与运维隔阂，用自动化工具链（CI/CD）把构建、测试、部署串成流水线。

**CI（持续集成）**：开发频繁把代码集成到主干，每次提交自动触发构建与自动化测试，以快速反馈避免「集成地狱」，强制代码规范与单测保证主干健康，团队通过构建状态实时了解项目健康度。

**CD（持续交付/部署）**：在 CI 基础上自动部署到测试/生产环境，涵盖自动化部署、环境配置管理、发布编排。价值是缩短上市时间、发布变更小易回滚、把运维从熬夜发版中解放。

目标流程一目了然：**本地提交（Git）→ 代码仓库（GitHub）→ Jenkins 自动构建 → 部署到服务器**。一切的基础是版本控制。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_1_1.svg" width="80%" alt="CI/CD 自动化流程示意图" />
  <p>图 1：CI/CD 自动化流程示意</p>
</div>

## 2. Git 基础

### 2.1 为什么需要版本控制

没有版本控制时：两人改同一份 `nginx.conf` 可能互相覆盖、改错无法回退、不知道谁在何时改了哪行。**版本控制系统（VCS）** 给每次修改打「版本号」，记录谁在何时改了什么，支持历史追踪、协作与回退。两类实现：**集中式**（如 SVN，依赖中央服务器、必须联网，服务器故障全员停工）；**分布式**（如 Git，每人本地都有完整版本库，断网也能提交与查看历史）。Git 由 Linus 为管理 Linux 内核而开发，两周用 C 写成，开源、免费、高效。

### 2.2 安装与身份配置

Windows 从 [git-scm.com](https://git-scm.com/) 下载（Intel/AMD 选常规安装包，ARM 设备选 ARM64），一路 Next 默认设置即可：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_1_2.png" width="80%" alt="Git for Windows 下载页面" />
  <p>图 2：Git for Windows 下载页面</p>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_1_3.png" width="70%" alt="Git 安装完成界面" />
  <p>图 3：Git 安装完成界面</p>
</div>

Linux 多数发行版内置，未装则 `sudo apt install git`（Ubuntu）或 `sudo yum install git`（CentOS）。验证：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_1_4.png" width="80%" alt="Git 版本验证" />
  <p>图 4：git --version 验证安装</p>
</div>

**配置身份**（分布式系统需要知道提交者是谁）：

```bash
git config --global user.name "YourName"
git config --global user.email "your_email@example.com"
git config --global color.ui true     # 可选：彩色输出
git config --list                      # 查看配置
```

> 这里的身份只用于在提交历史中记录开发者，**不是登录远程仓库的账号密码**；但建议与代码托管平台注册信息一致，便于识别。

### 2.3 三大区域与核心工作流

Git 的核心操作围绕**工作区、暂存区、版本库**三个区域展开：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_1_5.svg" width="80%" alt="Git 工作流" />
  <p>图 5：Git 三大区域工作流</p>
</div>

**初始化仓库**：

```bash
mkdir base-nlp && cd base-nlp
git init          # 生成隐藏的 .git 目录（版本库），不要手动删除或修改
```

> Windows 资源管理器默认隐藏 `.git`，可在文件夹选项勾选「显示隐藏文件」，或命令行 `dir /a` 查看。

**工作区 → 暂存区**：新建 `test.py` 后 `git status` 显示为 Untracked；`git add test.py`（或 `git add .`）后进入暂存区：

```bash
git status
# Changes to be committed: new file: test.py
```

暂存区像「购物车」，可在提交前确认这一批修改是否完整；改错了还能撤回：`git rm --cached test.py`。

**暂存区 → 版本库**：

```bash
git commit -m "first commit: add test.py"
```

`-m` 的备注信息很重要，推荐「**动词 + 对象**」：`feat: 新增 NER 预测接口`、`fix: 修复登录接口的 500 错误`、`refactor: 抽取公共的日志封装函数`；`update`、`test` 这类几乎零信息量的写法不推荐。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_1_6.png" width="80%" alt="反面教材" />
  <p>图 6：提交信息的反面教材</p>
</div>

### 2.4 差异对比与版本回退

修改文件后 `git status` 提示 modified，`git diff` 查看具体改动（绿色 `+` 新增、红色 `-` 删除）确认后再 `git add . && git commit -m "v2: update print output"`。

**查看历史**：`git log`（时间倒序，含 Commit ID、作者、说明，`HEAD -> master` 表示 HEAD 指向当前分支最新提交）；`git log --pretty=oneline` 每条压成一行。

**回退**：

```bash
git reset --hard HEAD^        # 回到上一版（HEAD~1），工作区+暂存区一起回滚
git reset --hard 3fa5369      # 按 Commit ID 精确切回指定版本（写前几位即可）
```

`--hard` 会丢弃未提交修改，用前务必确认。三种模式对比（了解即可）：

- `--soft`：只移动 HEAD，保留暂存区与工作区（适合合并多次提交）；
- `--mixed`（默认）：移动 HEAD、重置暂存区，保留工作区内容；
- `--hard`：HEAD、暂存区、工作区全部回滚，最干净也最危险，只建议在本地个人分支用。

团队协作中撤销**已推送**的提交，更安全的是 `git revert <commit>`——基于目标提交生成一个新的「回滚提交」，不改写历史，不影响已基于其开发的同事。

**reflog 黑匣子**：`git reset --hard` 之后 `git log` 看不到被回退的提交，但 `git reflog` 记录 **HEAD 的每一次移动**（commit/reset/checkout），找到目标 Commit ID 就能再次 `git reset --hard` 穿越回去。`log` 看版本历史，`reflog` 是所有关键操作的黑匣子，能救「手滑 reset」。

### 2.5 撤销修改与误删恢复

**未提交的修改**：

- 只改了工作区未 `git add`：`git restore test.py` 恢复到 HEAD 内容（等于这次编辑没发生）；
- 已 `git add` 未 `git commit`：先 `git restore --staged test.py`（第一次提交无 HEAD 时用 `git rm --cached test.py`），再 `git restore test.py`。

> 只要还没 commit，多数情况都能靠「编辑器 + `git restore`（老命令 `git checkout --`）」改回去。

**误删恢复**：文件只要**至少被 `git add` 过一次**，Git 就在暂存区/版本库里存了内容，`git restore demo.py` 即可找回。但如果之后执行过 `git reset --hard` 且内容未进入任何提交，就会被一并丢弃——重要文件尽快提交一次。**从未 `git add` 过的文件被删，Git 无能为力**，只能靠回收站或备份。

**主动删除并留痕**：`git rm old_script.py && git commit -m "remove old_script.py"`，当前及后续版本不再包含，但旧版本里仍可找回（前提是它曾出现在某次提交里）。

### 2.6 分支管理、合并与冲突

单分支开发时，大家都往主干提交容易把「半成品」混进去。**分支**的核心价值是在不影响主分支稳定性的前提下并行开发：小王开发「优惠券结算」时从主干拉出 `feature/coupon` 反复提交调试，自测通过后再合并回主干。

```bash
git branch                   # 查看分支，* 表示当前所在
git branch dev               # 创建 dev（仍停留在原分支）
git switch dev               # 切换（旧写法 git checkout dev）
git switch -c feature/live   # 创建并立即切换
```

分支是相互独立的代码快照，切换时工作区内容随之切换。在 `dev` 上新增并提交 `feature.py` 后切回 `master` 看不到它，直到：

```bash
git merge dev
# Updating 3fa5369..1d96be0  Fast-forward
```

**冲突**：两个分支改了同一文件同一行时 Git 无法自动判断，合并会报 `CONFLICT (content): Merge conflict in feature.py`，文件里出现：

```text
<<<<<<< HEAD
print("master_feature")
=======
print("dev_feature")
>>>>>>> dev
```

`<<<<<<< HEAD` 到 `=======` 是当前分支内容，`=======` 到 `>>>>>>> dev` 是被合并分支内容。解决步骤：按业务含义手工编辑冲突位置（保留其一或合并成新代码），**删除所有冲突标记**，然后 `git add feature.py && git commit -m "fix: resolve merge conflict in feature.py"`。

**删除与恢复分支**：`git branch -d dev`（在其他分支上删已合并的分支；不能站在要删的分支上删自己；未合并会警告，`-D` 强删需谨慎）。误删后 `git reflog` 找到最近一次 Commit ID，再 `git switch -c dev_recovered 0af27fc` 从历史时间点拉出新分支。

### 2.7 Tag：版本发布

一批 `0af27fcd9f...` 式的 Commit ID 难记也难看出「是哪个版本」。给关键版本打**语义化标签**（v1.0.0、v2.1.3），发布页以 Release 展示，回滚时说「退回 v1.0.0」即可。

```bash
git tag -a v1.0.0 -m "release v1.0.0"        # 给 HEAD 打带注释标签
git tag -a v1.0.0 8d4f871 -m "release v1.0.0" # 指定提交
git tag                                        # 列出标签
git show v1.0.0                                # 查看标签对应提交详情
git tag -d v1.0.0                              # 删除本地标签
git push origin v1.0.0                         # 推送单个标签
git push origin --tags                         # 推送所有标签
git push origin :refs/tags/v1.0.0              # 删除远程标签
```

`-a` 创建带注释标签（作为对象存入 Git 数据库），`-m` 是标签说明。

## 3. GitHub 远程仓库

本地版本库只在一台机器上：电脑损坏、硬盘丢失、想分享给团队都不方便。为解决**备份、协作、自动部署**三个问题，需要把本地仓库托管到远程平台（GitHub 为例，支持 Public/Private）。

**远程仓库的作用**：**集中托管**（所有人以远程为「单一真相源」）；**方便协作**（随时拉取最新代码、提交修改）；**自动部署入口**（Jenkins 等 CI/CD 工具从远程拉代码触发构建）。

**创建仓库**：访问 github.com 注册登录 → 右上角「+」→ New repository：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_1_7.png" width="80%" alt="GitHub 官网首页" />
  <p>图 7：GitHub 官网首页</p>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_1_8.png" width="80%" alt="GitHub 新建仓库菜单" />
  <p>图 8：GitHub 新建仓库菜单</p>
</div>

填写仓库名（如 `ner-demo`）、描述（可选）、可见性（Public/Private）、是否加 README、**.gitignore 选 Python**（自动忽略 `__pycache__/` 等）、许可证（可选，如 MIT/Apache 2.0）：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_1_9.png" width="80%" alt="GitHub 新建仓库页面" />
  <p>图 9：GitHub 新建仓库页面</p>
</div>

**SSH 免密推送（推荐）**：HTTPS 每次要输入账号密码/Token，SSH 配置密钥后免密，更适合日常。

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"   # 一路回车用默认路径
# 生成 ~/.ssh/id_rsa（私钥，勿泄露）与 ~/.ssh/id_rsa.pub（公钥）
```

复制公钥内容，GitHub 头像 → Settings → SSH and GPG keys → New SSH key：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_1_10.png" width="80%" alt="GitHub 配置 SSH Key" />
  <p>图 10：GitHub 配置 SSH Key 入口</p>
</div>

Title 填便于识别的名字，Key type 保持默认 `Authentication Key`（Signing Key 用于提交签名，暂不需要），粘贴公钥后 Add SSH key：

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_1_11.png" width="80%" alt="GitHub 添加 SSH Key 页面" />
  <p>图 11：GitHub 添加 SSH Key</p>
</div>

**关联本地与远程**（已有本地仓库的情况）：仓库页绿色 Code → SSH 选项卡复制地址 → 本地执行：

```bash
git remote add origin git@github.com:your-username/ner-demo.git
git push -u origin master      # 首次推送；默认分支为 main 则用 main
```

`origin` 是远程别名（约定俗成）；`-u` 建立本地分支与远程分支的跟踪关系，之后直接 `git push` 即可。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/basellm_15_1_12.png" width="80%" alt="GitHub 复制 SSH 地址" />
  <p>图 12：复制仓库 SSH 地址</p>
</div>

> 首次 SSH 连接会提示 `The authenticity of host 'github.com' can't be established...`，输入完整 `yes` 回车，之后不再询问。

**克隆远程仓库**（从零参与项目）：

```bash
git clone git@github.com:your-username/ner-demo.git
```

会得到含完整代码、`.git` 历史与已配置 `origin` 的目录。`clone` 只在第一次拿仓库时执行一次，后续更新用 `git pull`。

**日常协作**：开发前先 `git pull origin main` 拉最新 → 改代码、`git add` + `git commit -m "feat: ..."` → 完成后 `git push origin master/main`。**先 pull 再 push** 是安全习惯——始终基于远程最新状态开发与提交，减少冲突和覆盖风险。

> **补记**：Git 的所有「后悔药」都建立在同一件事上——**内容曾被 Git 记录过**。因此两条保命习惯：改完尽快 `git add`/`commit`；慎用 `git reset --hard`（它连暂存区一起清）。另外 `git log` 看版本历史、`git reflog` 看操作历史，后者能在 reset 后把人「捞」回来。

## 写在最后

本地提交、远程托管都就绪，最后一块拼图是「代码一旦推送就自动部署」——下一篇[Jenkins CI/CD流水线](4-Jenkins-CICD流水线)从安装配置、SSH 凭据到 Jenkinsfile 与 Webhook，把整条流水线真正跑通。
