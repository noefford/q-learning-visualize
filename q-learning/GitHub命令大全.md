# Git & GitHub 命令大全

> 从入门到精通的版本控制指南

---

## 📑 目录

1. [Git 基础配置](#一git-基础配置)
2. [仓库操作](#二仓库操作)
3. [文件操作](#三文件操作)
4. [分支管理](#四分支管理)
5. [远程仓库](#五远程仓库)
6. [提交历史](#六提交历史)
7. [撤销操作](#七撤销操作)
8. [标签管理](#八标签管理)
9. [GitHub 特有操作](#九github-特有操作)
10. [常见问题](#十常见问题)

---

## 一、Git 基础配置

### 1.1 设置用户信息

```bash
# 设置全局用户名（每次提交都会记录）
git config --global user.name "你的名字"

# 设置全局邮箱
git config --global user.email "your.email@example.com"

# 查看当前配置
git config --list

# 查看特定配置
git config user.name
```

**💡 说明**：
- `--global` 表示全局配置，对当前用户所有仓库生效
- 省略 `--global` 则只对当前仓库生效
- 用户名和邮箱会出现在每次提交记录中

### 1.2 配置编辑器

```bash
# 设置默认编辑器为 VS Code
git config --global core.editor "code --wait"

# 设置默认编辑器为 Vim
git config --global core.editor "vim"

# 设置默认编辑器为 Notepad++ (Windows)
git config --global core.editor "'C:/Program Files/Notepad++/notepad++.exe' -multiInst -nosession"
```

### 1.3 其他常用配置

```bash
# 设置默认分支名为 main
git config --global init.defaultBranch main

# 设置颜色输出
git config --global color.ui auto

# 设置命令别名（简化常用命令）
git config --global alias.st status      # git st = git status
git config --global alias.co checkout    # git co = git checkout
git config --global alias.br branch      # git br = git branch
git config --global alias.ci commit      # git ci = git commit
```

---

## 二、仓库操作

### 2.1 创建仓库

```bash
# 在当前目录初始化新仓库
git init

# 在指定目录初始化新仓库
git init 项目名

# 克隆远程仓库到本地
git clone https://github.com/用户名/仓库名.git

# 克隆指定分支
git clone -b 分支名 https://github.com/用户名/仓库名.git

# 克隆到指定目录
git clone https://github.com/用户名/仓库名.git 本地目录名
```

### 2.2 查看仓库状态

```bash
# 查看工作区状态（最常用）
git status

# 简短状态显示
git status -s

# 查看当前分支
git branch

# 查看远程仓库信息
git remote -v
```

---

## 三、文件操作

### 3.1 添加文件到暂存区

```bash
# 添加单个文件
git add 文件名.txt

# 添加多个文件
git add 文件1.txt 文件2.txt

# 添加所有修改的文件（最常用）
git add .

# 添加所有 .py 文件
git add *.py

# 交互式添加（逐个确认）
git add -i

# 查看将要添加的文件（ dry-run ）
git add --dry-run .
```

**📋 工作流说明**：
```
工作区(Working Directory) 
    ↓ git add
暂存区(Staging Area/Index)
    ↓ git commit
本地仓库(Local Repository)
    ↓ git push
远程仓库(Remote Repository)
```

### 3.2 提交更改

```bash
# 基础提交（会打开编辑器写提交信息）
git commit

# 带提交信息的提交（最常用）
git commit -m "提交信息"

# 提交并添加所有已跟踪的修改（跳过 git add）
git commit -am "提交信息"

# 修改最后一次提交（适用于未推送到远程时）
git commit --amend -m "新的提交信息"

# 只修改提交信息，不修改文件
git commit --amend --no-edit
```

**✍️ 提交信息规范**：
```
类型: 简短描述（不超过50字符）

详细描述（可选，可以换行多段）

相关Issue: #123
```

**常用类型**：
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 添加测试
- `chore`: 构建过程或辅助工具的变动

---

## 四、分支管理

### 4.1 查看分支

```bash
# 查看本地分支
git branch

# 查看远程分支
git branch -r

# 查看所有分支（本地+远程）
git branch -a

# 查看分支详情（含最后提交信息）
git branch -v

# 查看已合并的分支
git branch --merged

# 查看未合并的分支
git branch --no-merged
```

### 4.2 创建和切换分支

```bash
# 创建新分支
git branch 新分支名

# 切换到指定分支
git checkout 分支名

# 创建并切换到新分支（最常用）
git checkout -b 新分支名

# 创建分支并关联远程分支
git checkout -b 本地分支名 origin/远程分支名

# 切换到上一个分支
git checkout -

# 创建空分支（无历史记录）
git checkout --orphan 新分支名
```

### 4.3 合并分支

```bash
# 切换到目标分支（如 main）
git checkout main

# 合并指定分支到当前分支
git merge 要合并的分支

# 合并并创建合并提交（保留历史）
git merge --no-ff 要合并的分支

# 合并时压缩提交（将多个提交合并为一个）
git merge --squash 要合并的分支

# 终止合并（出现冲突时）
git merge --abort
```

### 4.4 删除分支

```bash
# 删除已合并的本地分支
git branch -d 分支名

# 强制删除分支（未合并也能删）
git branch -D 分支名

# 删除远程分支
git push origin --delete 分支名

# 简写形式删除远程分支
git push origin :分支名
```

### 4.5 分支重命名

```bash
# 重命名当前分支
git branch -m 新分支名

# 重命名指定分支
git branch -m 旧分支名 新分支名
```

---

## 五、远程仓库

### 5.1 添加远程仓库

```bash
# 添加远程仓库（默认名为 origin）
git remote add origin https://github.com/用户名/仓库名.git

# 添加多个远程仓库
git remote add upstream https://github.com/原作者/仓库名.git

# 查看远程仓库
git remote -v

# 查看远程仓库详情
git remote show origin

# 修改远程仓库地址
git remote set-url origin https://新的地址.git

# 删除远程仓库关联
git remote remove origin
```

### 5.2 推送代码

```bash
# 首次推送（建立关联）
git push -u origin main

# 推送到远程（已建立关联后）
git push

# 推送到指定分支
git push origin 分支名

# 强制推送（慎用！会覆盖远程历史）
git push -f origin main

# 推送所有分支
git push --all origin

# 推送标签
git push --tags
```

**⚠️ 警告**：`-f` 或 `--force` 会强制覆盖远程代码，团队协作时慎用！

### 5.3 拉取代码

```bash
# 拉取远程更新并合并（最常用）
git pull

# 拉取指定分支
git pull origin 分支名

# 只拉取不合并
git fetch

# 拉取所有远程分支
git fetch --all

# 拉取并变基（保持线性历史）
git pull --rebase origin main
```

**🔄 fetch vs pull**：
- `fetch`：只下载远程更新，不合并到本地
- `pull` = `fetch` + `merge`，下载并合并

### 5.4 同步远程分支

```bash
# 获取远程分支列表
git fetch origin

# 基于远程分支创建本地分支
git checkout -b 本地分支 origin/远程分支

# 删除本地已不存在的远程分支引用
git remote prune origin
```

---

## 六、提交历史

### 6.1 查看日志

```bash
# 查看提交历史（最常用）
git log

# 简化格式显示
git log --oneline

# 图形化显示分支合并历史
git log --graph --oneline --all

# 显示最近n条提交
git log -n 5

# 显示每次提交的文件变更统计
git log --stat

# 显示文件具体变更内容
git log -p

# 按作者筛选
git log --author="用户名"

# 按日期筛选
git log --since="2024-01-01" --until="2024-12-31"

# 查看某文件的修改历史
git log -p 文件名
```

### 6.2 查看具体提交

```bash
# 查看某次提交的详情
git show 提交ID

# 查看某次提交的文件列表
git show --stat 提交ID

# 查看某文件在某次提交时的内容
git show 提交ID:文件名
```

### 6.3 对比差异

```bash
# 查看工作区与暂存区的差异
git diff

# 查看暂存区与最新提交的差异
git diff --cached

# 查看工作区与最新提交的差异
git diff HEAD

# 查看两个提交之间的差异
git diff 提交ID1 提交ID2

# 查看某文件在两个版本间的差异
git diff 提交ID1 提交ID2 -- 文件名

# 查看某分支与当前分支的差异
git diff 分支名
```

---

## 七、撤销操作

### 7.1 撤销工作区修改

```bash
# 撤销某文件的修改（未添加到暂存区）
git checkout -- 文件名

# 撤销所有修改（危险！）
git checkout -- .

# 交互式选择要恢复的文件
git checkout -p

# 使用版本号恢复某文件
git checkout 提交ID -- 文件名
```

### 7.2 撤销暂存区修改

```bash
# 将文件从暂存区移回工作区（保留修改）
git reset HEAD 文件名

# 简写形式（Git 2.23+）
git restore --staged 文件名

# 撤销所有暂存
git reset HEAD .
```

### 7.3 撤销提交

```bash
# 软重置：撤销提交但保留修改到暂存区
git reset --soft HEAD~1

# 混合重置（默认）：撤销提交保留修改到工作区
git reset --mixed HEAD~1

# 硬重置：彻底丢弃修改（危险！）
git reset --hard HEAD~1

# 重置到指定版本
git reset --hard 提交ID
```

**🔔 注意**：`--hard` 会永久删除未提交的修改，使用前请确认！

### 7.4 查看和恢复已删除的提交

```bash
# 查看操作历史（包括已删除的提交）
git reflog

# 恢复到指定操作点
git reset --hard HEAD@{n}

# 恢复到指定提交
git reset --hard 提交ID
```

### 7.5 撤销合并

```bash
# 合并过程中出现冲突，放弃合并
git merge --abort

# 合并完成后想撤销
git reset --hard HEAD~1

# 或使用 reflog 找回之前的状态
git reflog
git reset --hard HEAD@{1}
```

---

## 八、标签管理

### 8.1 创建标签

```bash
# 创建轻量标签
git tag 标签名

# 创建附注标签（推荐）
git tag -a 标签名 -m "标签说明"

# 给指定提交打标签
git tag -a 标签名 提交ID -m "标签说明"

# 创建 GPG 签名标签
git tag -s 标签名 -m "标签说明"
```

### 8.2 查看标签

```bash
# 列出所有标签
git tag

# 列出符合条件的标签
git tag -l "v1.*"

# 查看标签详情
git show 标签名
```

### 8.3 推送和删除标签

```bash
# 推送单个标签到远程
git push origin 标签名

# 推送所有标签
git push --tags

# 删除本地标签
git tag -d 标签名

# 删除远程标签
git push origin --delete 标签名
```

### 8.4 检出标签

```bash
# 检出标签（进入分离头指针状态）
git checkout 标签名

# 基于标签创建分支
git checkout -b 新分支名 标签名
```

---

## 九、GitHub 特有操作

### 9.1 Fork 和 Clone

```bash
# 克隆自己的 Fork 仓库
git clone https://github.com/你的用户名/仓库名.git

# 添加上游仓库（原仓库）
git remote add upstream https://github.com/原作者/仓库名.git

# 同步上游更新
git fetch upstream
git checkout main
git merge upstream/main
```

### 9.2 GitHub CLI（可选）

```bash
# 安装 GitHub CLI 后可以使用

# 创建新仓库并推送
git init
git add .
git commit -m "Initial commit"
gh repo create 仓库名 --public --source=. --remote=origin --push

# 创建 Pull Request
gh pr create --title "标题" --body "描述"

# 查看 PR 列表
gh pr list

# 合并 PR
gh pr merge

# 创建 Issue
gh issue create --title "问题标题" --body "详细描述"
```

### 9.3 生成 SSH 密钥（推荐）

```bash
# 生成 SSH 密钥
ssh-keygen -t ed25519 -C "your.email@example.com"

# 或者使用 RSA
ssh-keygen -t rsa -b 4096 -C "your.email@example.com"

# 复制公钥到剪贴板（Mac）
pbcopy < ~/.ssh/id_ed25519.pub

# 复制公钥到剪贴板（Windows）
clip < ~/.ssh/id_ed25519.pub

# 测试连接
cdssh -T git@github.com
```

---

## 十、常见问题

### Q1: 提交时提示 "Please tell me who you are"

```bash
# 解决方法：配置用户信息
git config --global user.name "你的名字"
git config --global user.email "your.email@example.com"
```

### Q2: 推送时提示 "error: src refspec main does not match"

```bash
# 原因：本地分支名与远程不匹配
# 解决方法：查看本地分支名并推送
git branch                    # 查看分支名
git push -u origin master     # 如果是 master 分支
git push -u origin main       # 如果是 main 分支
```

### Q3: 出现合并冲突

```bash
# 1. 查看冲突文件
git status

# 2. 手动编辑冲突文件，删除冲突标记
# <<<<<<< HEAD
# 你的代码
# =======
# 别人的代码
# >>>>>>> branch-name

# 3. 添加解决冲突的文件
git add .

# 4. 完成合并
git commit -m "解决合并冲突"

# 或者放弃合并
git merge --abort
```

### Q4: 不小心提交了敏感信息

```bash
# 从所有历史中删除文件（慎用！会重写历史）
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch 文件名' \
HEAD

# 或使用 BFG Repo-Cleaner（更快）
# 然后强制推送
git push --force
```

### Q5: 仓库太大，只想克隆最新版本

```bash
# 浅克隆（只克隆最近一次提交）
git clone --depth 1 https://github.com/用户名/仓库名.git

# 后续可以拉取完整历史
git fetch --unshallow
```

### Q6: 忽略文件不生效

```bash
# 原因：文件已被跟踪，需要先从缓存移除
git rm -r --cached .
git add .
git commit -m "更新 .gitignore"
```

---

## 📚 学习资源

### 官方文档
- [Git 官方文档](https://git-scm.com/doc)
- [GitHub Docs](https://docs.github.com/)
- [Git 可视化练习](https://learngitbranching.js.org/)

### 推荐书籍
- 《Pro Git》（免费在线阅读）
- 《GitHub入门与实践》

### 图形化工具
- **GitHub Desktop** - 官方客户端
- **SourceTree** - 功能强大的免费工具
- **GitKraken** - 美观的跨平台工具

---

## 🎯 速查表

### 日常开发流程

```bash
# 1. 获取最新代码
git pull

# 2. 创建并切换到新分支
git checkout -b feature/xxx

# 3. 修改代码...

# 4. 查看修改
git status
git diff

# 5. 添加修改
git add .

# 6. 提交修改
git commit -m "feat: 添加新功能"

# 7. 推送到远程
git push -u origin feature/xxx

# 8. 在 GitHub 上创建 Pull Request

# 9. 合并后删除本地分支
git checkout main
git pull
git branch -d feature/xxx
```

### 常用命令速记

| 命令 | 作用 |
|------|------|
| `git status` | 查看状态 |
| `git add .` | 添加所有修改 |
| `git commit -m ""` | 提交更改 |
| `git push` | 推送到远程 |
| `git pull` | 拉取更新 |
| `git log --oneline` | 简洁历史 |
| `git checkout -b xxx` | 创建并切换分支 |
| `git merge xxx` | 合并分支 |
| `git clone url` | 克隆仓库 |

---

**Happy Coding! 🚀**

*Last Updated: 2024*
