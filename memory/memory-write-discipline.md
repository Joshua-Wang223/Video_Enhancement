---
name: memory/ 有多个写入方 —— 改动一律定点 Edit、写后复核 A/B
description: memory/ 下文件会被 auto-memory 抽取流程/并行会话追加内容；改 memory 时禁止整文件覆盖（会静默抹掉外部追加的记忆），必须定点 Edit，并在写后比对 A/B 双侧再决定同步方向
type: project
---

**事实**：`memory/` 不是只有我在写。2026-09-23 实测到三处证据：

- 我写完 `github-push-workflow-and-secrets.md` 后，文件里出现了我没撰写的段落（"用户 2026-09-23 拍板：选『加入 .gitignore 不推送』"那三行）—— 来自**外部写入方**（auto-memory 记忆抽取流程 / 并行会话）。
- `memory/gate-verify-plan-known-failures.md` 我整场没动过，却出现在待提交增量里；同批文件的 mtime 精确到**同一纳秒**（`03:35:18.753242268`），是批量写入的特征。
- 还有一个只出现在文件清单里、我从未创建的 `feedback_diagnostic_context_not_hidden.md`。

**Why**：若把"我刚读到的内容"当成"文件全部内容"再用 Write 整文件覆盖，会**静默抹掉**外部追加的记忆 —— 记忆丢失不可逆，且没有任何报错。

**How to apply**：

- 改 `memory/` 下**已存在**的文件一律用**定点 Edit**（锚定一小段唯一文本）；只有新建文件才用 Write。
- `old_string` 匹配失败本身就是"文件已被外部改过"的信号 —— 此时重新 Read 现状，而不是强行覆盖。
- 写完后按仓库约定同步 A（`/workspace/Video_Enhancement/memory`，canonical）↔ B（`/root/.codebuddy/projects/workspace-Video_Enhancement/memory`）并 `diff -r` 复核。⚠️ 若两侧**各自都有**对方没有的新增内容，先合并再同步，别无条件 `cp -a A/. B/` —— 那会把 B 侧的外部追加覆盖掉。
- 遇到陌生的文件或段落，先假定是外部写入方的产出，不要当残留删掉。
