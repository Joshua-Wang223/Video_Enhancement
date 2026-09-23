---
name: GitHub 推送流程（SSH-over-443）与「密钥红线」
description: force_push_github.sh 的正确调用方式、治理文件(.gitignore/.gitattributes)以 origin 为准的 [FIX-IGNORE-CANONICAL] 约定、推送后一律复核 ls-remote 的铁律与「怎么读推送日志」（0 报错≠推送成功、dry-run 行与真实行同格式）、models.del 类改名残留绕过 .gitignore 与哨兵的缺口、以及 config/cc-switch-*.md 含真实 API Key 已被 .gitignore 排除（GitHub 只认 OpenRouter，DeepSeek 无检测器）
type: project
---

## 推送入口

工作区 `/workspace/Video_Enhancement` **本身就是一个普通 git 仓库**（origin 已配好 SSH），日常直接 `git commit` + `git push origin main` 即可（2026-09-23 实测可用）。仓库根另有 `force_push_github.sh`，用于「拿某个环境的**完整工作区内容**全量覆盖远程」这条路（会 `read-tree --empty` 重建索引，见下方治理文件一节）：

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

**最强复核 = 独立浅克隆**（2026-09-23 两次用于收口，比 `ls-remote` 更硬）：

```bash
git clone -q --depth 1 --single-branch --branch main \
  git@github.com:Joshua-Wang223/Video_Enhancement.git /tmp/vc
# 然后在 /tmp/vc 里按 config.py 的方式跑目标链路（import 后端 / 加载模型架构）
```

clone 成功本身即证明**对象完整、推送未被截断**（`ls-remote` 只报 ref SHA，不校验对象可获取）；在 clone 里跑一遍导入链，还能证明"仓库内容足以让新机器跑起来"，而不只是"文件都在"。用完 `rm -rf /tmp/vc`。

**Provenance**：2026-09-23 用户在本会话明确要求「补充一条以后省事的经验：推送后一律独立复核 `git ls-remote origin refs/heads/main`」—— 即这条是**用户指定的长期经验**，不要因为"脚本的 SIGPIPE 缺陷已修"就把它删掉或降级。

### 怎么读推送日志（2026-09-23 用户要求「检查推送日志确认无报错」时实测出的两条反直觉）

1. **日志里 0 条报错 ≠ 推送成功。** 第 1 次运行（`force_push.log`）用 `error|fatal|failed|denied|rejected|⚠️` 扫描是 **0 命中**，但它**根本没走到推送阶段**——被 SIGPIPE 静默中止，日志里连一行错都没有。所以"日志干净"不能当证据（这正是本文件开头那条铁律的存在理由）。
2. **dry-run 的成功行与真实推送的成功行格式完全相同**：都是 `   <old>..<new>  <40位sha> -> main`。`force_push2.log` 里那 1 行 `-> main` 其实是 **dry-run**，真实那次紧跟其后被 `! [remote rejected]`。要区分必须看它是否出现在 `--- 预演通过，开始正式推送 ---` **之后**。

**可信的成功签名**（三条同时满足，但仍以 `ls-remote` 为准）：① 有 `=== 6. 预演 + 推送 ===` 小节；② 出现 **2 行** `-> main`（dry-run + 真实）；③ `rejected|declined` 为 0 行。

**How to apply**：复核历史推送日志时别只看"有没有 error"，按上面三条签名判定"到底推没推出去"；本会话 4 份日志（03:26/03:27/03:28/03:41）就是"0 报错但没推 / 走到推送但被拒 / 真成功"三种形态的现成样本。

## ⚠️ 大目录哨兵与 .gitignore 有同一个双重缺口：`.del` / `.bak` 类改名残留

2026-09-23 清理根目录 stray 权重目录时实测：把 `models/` 改名成 **`models.del/`**（"待删"的常见做法）后，**两道防线都拦不住它**——

- `.gitignore` 里是 `/models/`、`/models_*/`（2026-09-23 已**锚定到仓库根**；仍匹配不到 `models.del/`；且 `.gitignore` 只作用于未跟踪路径）；
  ⚠️ 反过来说：**未锚定**的 `models/` 会连带忽略 `external/**/models/` 这类**源码**目录 —— 2026-09-23 实测它把两个运行期必需目录长期锁在仓库外（IFRNet 架构 + vendored realesrgan 的 models 子包），任何全新 clone 都因此跑不起来，见 [报 No module named 'models' / 'realesrgan.models'](ifrnet-models-package-missing.md)；
- 脚本第 5 步哨兵正则 `^(models|models_[^/]*|temp|output|logs|gfpgan|\.trt_cache|\.t2_cache)/` 同样不匹配 `models.del/`。

⇒ 该目录 **128 MB（含两个 67 MB 的 .pth 权重）会直接进提交树被推上去**，而哨兵会照常打印 `✅ 大目录未被纳入`，给出假的安心感。

**How to apply**：清理大目录时**不要停留在改名**（`mv x x.del`），改名正是绕过防线的动作；要么真删、要么同时把新名字加进 `.gitignore` 与哨兵正则。推送前用 `git ls-tree -r --name-only <tree> | grep -E '^(models|temp|output|logs)'` 之类的**宽匹配**自查一遍，别只信脚本那句 ✅。

## ⚠️ `.gitignore` 对**已跟踪**文件无效 —— 全量重建索引会静默剔除它们

**规则**：`.gitignore` 只作用于**未跟踪**路径。给某模式加规则后，**此前已被跟踪**的匹配文件不会自动离仓；反过来，脚本第 4 步的 `git read-tree --empty && git add -A` 会全量重建索引，此时这些文件被**静默从提交树里剔除**（对远程表现为删除）。

**Why**：2026-09-23 实测——给 `Plan/*session*`、`Plan/*.txt` 加规则后，仍有 4 个 `Plan/session-ses_*.md`（合计 2.3 MB）与 `Plan/水彩花屏修复验证一次性执行Prompt.txt` 因**早已被跟踪**而滞留在远程；直到下一次全量覆盖推送才被剔除。极易误判为"我没动它们，怎么被删了"。

**How to apply**：
- 加/改 `.gitignore` 后立刻自查"被跟踪但已忽略"清单：
  `git ls-tree -r --name-only HEAD | git check-ignore --no-index --stdin`
  不想删的，就从规则里排除，或加 `!` 负向规则。
- 脚本第 4 步自带 `IGN_TRACKED` 守卫会**列出清单并要求 `ALLOW_BIG=1`** 才继续（原文 `如确认要移除它们，请加 ALLOW_BIG=1 重跑`）。**看到这条守卫不是故障**，是"你正在删以前跟踪的文件"的确认请求 —— 确认再放行。
- ⚠️ `ALLOW_BIG=1` 会**同时**放行大目录哨兵，不只 `IGN_TRACKED`；用它之前先独立确认树里确实没有大目录（见上一节）。

## 治理文件（.gitignore / .gitattributes）一律以 origin 为准 —— [FIX-IGNORE-CANONICAL]，2026-09-23

`force_push_github.sh` 第 4 步用 `git read-tree --empty && git add -A` 重建索引，**完全以本地 `.gitignore` 为准**；而原第 3 节只在 `.gitignore` 缺失时才从远程取。⇒ **任一环境带着旧规则跑脚本，就会用旧规则覆盖远程**——这正是 2026-09-23「另一环境缺 `.gitattributes` + 未锚定的 `models/` → 覆盖后 EOL 锁定丢失、`external/IFRNet/models/` 等架构源码被剔除、全新 clone 起不来」的机制。

现已改为：第 3 节对 `.gitignore` 与 `.gitattributes` **一律采用 `origin/$BRANCH` 版本**，本地有差异只告警、不采纳；需要本地优先时设 `IGNORE_LOCAL=1`。

**How to apply**：
- 要改这两个文件，**必须先 `git push` 到 origin** 再由各环境取用。只改本地不推 = 白改（下次跑脚本会被 origin 版本覆盖）。
- 在别的环境看到 `⚠️ .gitignore 与 origin 不一致 → 覆盖为 origin 版本` 是**预期行为**，不是故障。
- 验证手段：`bash -n force_push_github.sh` + 在**一次性克隆**里 `PUSH=0` 实跑。⚠️ 该脚本会 `update-ref refs/heads/main` 改写本地 main，**切勿**在主仓库直接实跑验证。
- 配套的仓库侧锚定（同一类隐患，2026-09-23 完成）：`.gitignore` 中所有「运行/权重目录」模式已锚定为 `/logs/ /models/ /models_*/ /gfpgan/ /output/ /temp/ /tmp/`；缓存/工具目录（`__pycache__/`、`.trt_cache/`、`.vscode/`、`.codebuddy/` 等）**有意保持未锚定**，勿再"修正"。

## ⚠️ `git reset --hard` 会删掉**已暂存**的原未跟踪文件

**事实**：`reset --hard` 清理的范围是"**索引/HEAD 里有、目标提交里没有**"的路径，**不看**它当初是不是未跟踪。因此只要在对比过程中跑过一次 `git add -A`（哪怕只是为了 `git diff origin/main <work-tree-tree>`），原本 untracked 的文件就变成**已暂存**，`reset --hard` 会把它从工作区**一并删除**。

**Why**：2026-09-23 对齐远程时实测 —— `memory/memory-write-discipline.md` 本是 `??` 未跟踪（纯 untracked 本可毫发无损地活过 `reset --hard`），但我在先前步骤里用 `git add -A` + `git write-tree` 做过全量对比，它已被暂存 → `reset --hard origin/main` 直接删了它。若没有提前快照，这份只存在于工作区的记忆就永久丢了。

**How to apply**：
- 执行 `reset --hard` / `checkout -f` / `read-tree --empty` 这类操作前，先**明确"工作区里哪些内容不在任何 commit 里"**（未跟踪 + 未提交修改），把它们 `cp -a` 到工作区外（如 `/tmp/`）后再动手。
- 想保留 untracked 安全性，对比工作区时优先用 `git diff --no-index` 或 `git stash create`，**避免**用 `git add -A` 把文件"升格"为已暂存。
- 事后想找回：`git show <old-sha>:<path>` 只能救**曾经被提交过**的；纯工作区内容只能靠预先前置的快照。

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
