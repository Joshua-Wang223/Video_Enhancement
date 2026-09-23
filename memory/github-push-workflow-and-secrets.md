---
name: GitHub 推送流程（SSH-over-443）与「密钥红线」
description: force_push_github.sh 的正确调用方式、跑完必须复核 ls-remote 的原因（曾因 SIGPIPE 静默中止）、以及 config/cc-switch-*.md 含真实 API Key 已被 .gitignore 排除（GitHub 只认 OpenRouter，DeepSeek 无检测器）
type: project
---

## 推送入口

工作区 `/workspace/Video_Enhancement` 平时**不是** git 仓库（每次推送由脚本现建 `.git`）。官方入口是仓库根自带的 `force_push_github.sh`：

```bash
cd /workspace/Video_Enhancement
GH_URL='git@github.com:Joshua-Wang223/Video_Enhancement.git' bash force_push_github.sh
PUSH=0 bash force_push_github.sh    # 只准备提交、不推送
```

- 远程 `Joshua-Wang223/Video_Enhancement`，分支 `main`。脚本 `commit-tree -p origin/main` + `--force-with-lease`，正常情况是**在远程基线之上新增一个提交**（历史保留），不是孤儿覆盖；每轮自动打本地备份 tag `backup/pre-force-push-*`。
- 本容器 `~/.ssh/config` 把 `github.com` 指向 **`ssh.github.com:443`**（22 端口被限），密钥 `~/.ssh/id_ed25519`；2026-09-23 实测 `ssh -T git@github.com` → `Hi Joshua-Wang223!`。容器内**没有** gh 登录 / credential helper / HTTPS 令牌文件 —— 别走 HTTPS 路径。
- **Why**：容器重建后这些凭据与 `.git` 都不持久，脚本是唯一被设计来兜住这一点的入口。

## 铁律：任何推送之后一律独立复核远程 ref

脚本曾有一处 `set -o pipefail` + `git diff … | head -30` 的 **SIGPIPE(141) 静默中止**缺陷 —— 变更条目 >30 时 `head` 提前关管道，脚本在**推送前**就死掉：不报错、不推送；而常见调用 `bash force_push_github.sh | tee log` 的退出码被 `tee` 掩盖成 0，看起来"跑完了"。已修 `[FIX-PIPEFAIL-SIGPIPE]`（改用 `sed -n '1,30p'`，读满才退出；同类 `printf … | head -20` 一并替换）。

⇒ **不论用哪个脚本或命令推送，推送完成后一律独立复核远程 ref** —— 这是唯一能确认"真的推出去了"的手段。**本地提交成功 ≠ 远程已更新**（本脚本的失败模式是不报错也不推送），也别把 `git push` 的退出码或"看起来跑完了"当证据：

```bash
git ls-remote origin refs/heads/main   # 与本地 git rev-parse refs/heads/main 逐字比对
```

**How to apply**：把这条当成推送动作的固定收尾步骤（省下的是"以为推上去了、其实没有"这类返工）。看到"脚本跑完了但远程没变"，先查是不是被 `set -e`+管道早闭静默中止，而不是怀疑网络或凭据。

## 密钥红线（2026-09-23 推送被 GitHub Push Protection 拦下）

- `config/cc-switch-{claude,codex}-{openrouter,deepseek}.md` 四个文件各含 **1 个真实 API Key**。它们**从未在远程**，本次新增才触发 `GH013 / Push cannot contain secrets`。
- ⚠️ **GitHub 只报了 OpenRouter 那两个**；DeepSeek 的 `sk-` 密钥**没有对应检测器**。只清被点名的文件会让 DeepSeek 密钥**静默进仓** —— 必须四个一起挡。四者已加入 `.gitignore`（`config/cc-switch-*.md`），本地文件原样保留。
- 推送前自查（不回显明文）：对**待推送的树**跑
  `git grep -I -E -l '(gh[pousr]_|sk-|AKIA|xox[baprs]-|PRIVATE KEY|AIza|glpat-)' <tree>`
- GitHub 提供 unblock 链接可放行单个 secret，但真密钥会公开并被历史永久记住 —— **不要**用。
- **用户 2026-09-23 拍板：选"加入 .gitignore 不推送"**（而非脱敏改写文件、也不是去 GitHub 放行）。
  **Why**：cc-switch 那几份本地文档要照常可用（脱敏会破坏正在使用的配置），目标是"密钥永不进仓"而不是"文件必须进仓"。
  **How to apply**：再遇到夹带凭据的本地配置/文档，默认按"**忽略而非改写**"处理，不要擅自修改用户文件里的密钥内容。

## 2026-09-23 本次推送记录

基线 `9518b40`；内容同步提交 **`274095d`**（118 条目变更 / 466 文件 / 30.7 MB / 无大目录泄漏）。该提交之后可能还有仅含 memory 追加的提交，**当前指针一律以 `git ls-remote origin refs/heads/main` 为准**。回滚 = `git push --force origin 9518b40:refs/heads/main`，或本地 tag `backup/pre-force-push-20260923-032845`。中途那个含密钥的本地提交 `19b5358` 从未推送，已 `reflog expire --all + gc --prune=now` 清除。
