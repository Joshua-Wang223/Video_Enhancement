---
name: 开发与运行环境
description: 开发环境 Windows，运行环境 Linux；含 2026-09-23 全仓 LF 策略、git 同步与并行会话覆盖警告
type: project
originSessionId: 549bd23c-b680-4b5e-b833-3852c89608f0
---
本项目的开发环境是 Windows（代码编辑、调试在 Windows），但实际运行/部署环境是 Linux（生产环境中执行视频处理任务）。

**Why:** 用户在 Windows 上开发和修改代码，但将这些代码部署到 Linux 服务器上运行。

**How to apply:** 编写代码时注意跨平台兼容（路径分隔符、换行符、shell 命令等），优先使用 POSIX 路径风格和可移植的 Python API（如 `pathlib.Path`）。

## memory 结论的归属：以 Linux 生产侧为准（2026-09-02 用户确认）

`memory/` 中的结论（尤其 `status: fixed` + "GPU 验证 ✅" 的条目）记录的是 **Linux 生产侧验证结果**。Windows 开发树是滞后的副本，两侧没有版本控制（仓库未初始化 git），靠手工拷贝同步，因此**开发树里搜不到某记忆描述的代码标记是正常现象，不构成否定该记忆结论的证据**。

**判定规则：**
- 在 Windows 开发树 grep 不到记忆声称的 FIX 标记 → 默认按"生产侧已改、开发树滞后"处理，**不得据此改写记忆结论或翻转 `status`**；
- 需要确认时，用文件 mtime 辅助判断方向：记忆 mtime **晚于**代码 mtime → 几乎可断定改动在生产侧或未回传；
- 只有拿到生产侧实证（日志/复跑结果）或用户明确说明，才可修订记忆结论。

**实例（2026-09-01/02）：** `ifrnet-hevc-la-slot-headroom-deadlock` 的 `la_depth + 2`、`ifrnet-f0-nv12-async-copy-race` 的 `[FIX-F0-NV12-STREAM-SYNC]`/`wait_on_event`、`ifrnet-f0-la0-double-consume` 的取用下放，在 Windows 开发树（代码 mtime 8/27、8/31）中均不存在，而记忆写于 9/1 17:34。经用户确认按"生产侧已验证、开发树滞后"处理，三篇记忆**保持原样**。

**注意反向情形：** 若某记忆描述的 bug 在开发树中已被**另一条代码路径**规避（如 `ifrnet-f0-la0-double-consume`：开发树 `main.py` 改为按 `_la_depth > 0` 分流，LA=0 直接 `encode_frame`，使该 bug 不再可达），这属于等价修复而非记忆失效，同样不应改写记忆——但值得在排查时留意"开发/生产路径不同"。

## 三套路径别名（2026-09-26 用户明确）

同一个项目 `Video_Enhancement` 在三处的绝对路径：

| 角色 | 路径 |
|---|---|
| 开发环境 · Windows | `D:\Workspace_Python\Video_Enhancement` |
| 开发环境 · WSL（挂载 D 盘） | `/mnt/d/Workspace_Python/Video_Enhancement` |
| **生产环境** · Linux | `/workspace/Video_Enhancement` |

**How to apply:**
- 写脚本/文档时**不要硬编码**这三者的任何一个；用 `Path(__file__).resolve().parents[n]` 推导项目根（本仓脚本大多在 `Accessory/<分类>/` 下，需回退两层）。
- 记忆里的路径一律以**生产侧 `/workspace/...`** 为准（与上文「以 Linux 生产侧为准」一致）；在 WSL 开发树看到的 `/mnt/d/...` 只是同一份代码的挂载别名。
- 生产侧的改动要经 git（origin `git@github.com`）推送后再拉取；不能用路径拷贝代替版本同步。

## 换行符策略：全仓锁定 LF（2026-09-23 用户确认）

仓库根有 `.gitattributes`：`* text=auto eol=lf` + 30 余种二进制类型（png/jpg/mp4/pt/onnx/mdb…）显式声明 `binary`。仓库（index/HEAD）与工作区一律 LF，**Windows 开发机检出也是 LF**，并已用 `git add --renormalize .` 收敛历史 CRLF blob。

**Why:** 此前仓库 blob 为 CRLF（Windows 侧提交）、Linux 工作区为 LF，导致全仓每个文件都显示 modified，真实改动被行尾噪音淹没——`git diff` 曾虚高到 44k 行插入 / 39k 删除，无法审阅。

**How to apply:**
- 本目录**是 git 仓库**（origin `git@github.com:Joshua-Wang223/Video_Enhancement.git`）。`force_push_github.sh`、`tar_excludes.txt` 是同步工作流工具，**保留 tracked**（详见 `github-push-workflow-and-secrets.md`）。
- ⚠️ **存在并行会话共用同一远程**：2026-09-23 实测另一会话用 `force_push_github.sh` 全量覆盖推送（提交标题 `Sync: /workspace latest complete code`），它会**回退本地未覆盖到的仓库级改动**——删掉 `.gitattributes`、恢复已 gitignore 的转储、删除报告类与历史文件，同时带来它自己的新工作。
  - 动手前先 `git fetch` + `git log --oneline HEAD..origin/main`，确认远程没被别人覆盖；
  - 发现分叉优先「合并取长」（远程为基底 + 恢复本地基础设施），不要直接强推；
  - **推送后必须 `git ls-remote origin refs/heads/main` 独立复核**：本地 push 输出成功 ≠ 远程已更新。
- Windows 侧若看到"整仓被修改"，先 `git add --renormalize .` 再看 `git status`，不要盲目提交或回滚。
- ⚠️ 陷阱：`git checkout-index -a -f` **不重写已存在文件**的行尾（实测无效）。要把工作区旧 CRLF 文件重写为 LF，用 `rm <file> && git checkout -- <file>`。
- git 会把含**孤立 CR**（非 CRLF）的文件判为 `-text` 二进制并拒绝归一化（实例 5 个：`Plan/session-ses_fb90.md`、`fb9a`、`fbdc`、`fcd1`、`Accessory/docs/diag_output_fixed.log`），属预期保护行为。

