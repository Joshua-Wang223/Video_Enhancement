---
name: memory/ 有多个写入方 —— 改动一律定点 Edit、写后复核 A/B
description: memory/ 下文件会被 auto-memory 抽取流程/并行会话追加内容；改 memory 时禁止整文件覆盖（会静默抹掉外部追加的记忆），必须定点 Edit，并在写后比对 A/B 双侧再决定同步方向；含 2026-09-28 A/B 整周脱同步 + HEAD 里带 stash 冲突标记的实测与 5 步同步手法
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

## 2026-09-28 实测：A/B 可能整周脱同步，且冲突标记会被提交进 HEAD

- 现象：A（仓库 `memory/`）比 B（会话侧）新整整一周（A mtime `09-28 06:27` / B `09-23 07:41`），`diff -rq` 报 49 个文件不同；**A 的 `MEMORY.md` 内还残留 stash 冲突标记**（`<<<<<<< Updated upstream` / `=======` / `>>>>>>> Stashed changes`），且该段残留在 `dc63317` 已被**提交进 HEAD**。
- 可复用手法（同步前按序做）：
  1. **逐文件方向核对**，别只看单点 mtime：对两侧同名文件逐个比内容 + mtime，统计 `A-newer / B-newer / ONLY-A / ONLY-B` 四类计数。本例 45 A-newer、0 B-newer、4 ONLY-A、0 ONLY-B ⇒ 可判定「B 是 A 的过期子集」，单向覆盖无损失。**只要 B-newer 或 ONLY-B 非 0，就必须先合并再同步。**
  2. **覆盖前快照对侧**（`cp -a <B> /tmp/mem_B_snapshot_<ts>`）。
  3. 清理冲突标记时**保留信息更全的一侧**（本例 upstream 侧含 2026-09-24 的 Windows 推送路径补充，stash 侧是较短旧版），删标记时一并删重复行。
  4. 删除用**字节级操作**（`open(...,'rb')` + 按行号删 ASCII 标记行），不经 shell 管道处理中文；写后校验 UTF-8 严格解码成功、`\ufffd` 计数为 0、无 `?{2,}` 串。
  5. 覆盖后 `diff -r` 必须报逐字节一致，并核对两侧文件数相等（本例 116/116）。
- ⚠️ 结论：**「上次同步过」不代表现在同步**；批量改 memory 后一律按上面 5 步走，别无条件 `cp -a A/. B/`。

## 2026-10-04 补充：`cp -a A/. B/` 的风险范围比上面写的更窄（澄清一条会误伤的规则）

上面第19 行写「别无条件 `cp -a A/. B/`」。本轮实测把这条**收窄**，避免它被当成"任何情况下都不能用"：

- **`cp -a A/. B/` 不会删除 B 侧任何文件** —— 它只覆盖**同名**文件。B 独有的文件会原样存活。
  ⇒ 「仅 B 有」的文件**不会**被 `cp -a` 抹掉，先合并的义务只针对**同名且内容分叉**的文件。
- 判别命令（`rsync` 未装时用）：`comm -13 <(cd A && ls -1|sort) <(cd B && ls -1|sort)` 得仅 B 有，
  `comm -23 ...` 得仅 A 有；**再对同名文件跑 `diff -rq`** 才知道有无内容分叉（本轮`diff -rq` 报的3 个差异全是"仅 B 有"的文件，**不是**同名分叉）。
- 本轮实测量：A仅 9 个 / B 仅 3 个；处置 = 先只同步自己改的那个文件并 `diff` 确认逐字节一致，
  再 `cp -a "$A/." "$B/"` 补齐 A 侧全部（**不删 B 侧**），最后把"B 仍独有的 3 个文件"列给用户裁定。
  ⚠️ 这 3 个是别处写入未回流到仓库的记忆（其中 `ffmpeg9-vbrhq-removal-impact.md` 与 A 侧
  `ffmpeg9-vbr_hq-removal-impact.md` **疑似改名前后两名**，内容不同）——**改名残留**判定见
  [GitHub 推送流程](github-push-workflow-and-secrets.md)一节，**别单凭"内容不同"就断定谁废弃**。

### 本容器（WSL/开发侧）的 A/B 实际路径与文档不一致

文档里写的 A=`/workspace/Video_Enhancement/memory`、B=`/root/.codebuddy/projects/workspace-Video_Enhancement/memory`
在本容器**都不存在**。实测本容器是：

- **A（canonical，仓库内）** = `/mnt/d/Workspace_Python/Video_Enhancement/memory`
- **B（会话侧镜像）** = `/home/administrator/.codebuddy/projects/mnt-d-Workspace_Python-Video_Enhancement/memory`

⇒ **动手前先 `ls -d` 确认两侧真实存在**，别照抄文档路径（照抄会误判成"镜像丢了"）。
判定纪律不变：**以 A 为准、双侧逐字节复核**。

## 2026-10-04 补充：同步「内容」之外还要同步「索引」

修完索引完整性后A/B 两侧 `diff -r` 一致，但**索引条目的增删本身也是要同步的内容**：
本轮 `cp -a "$A/." "$B/"` 把 B 侧 `MEMORY.md` 整份覆盖成 A 侧版本，**静默抹掉了 B 侧独有的索引条目**
（对应的三个memory 文件因不在 A 侧而成为「仅 B 有」，但索引指向它们的行被抹掉了 ⇒ 从此读不到）。
⇒ **同步后除了 `diff -r`，还要跑一次索引双向校验**（见
[索引双向完整](memory-index-integrity.md)）：「悬空 0 / 未收录 0」是同步完成的验收条件之一。

另：批量重写 `memory/` 下已有文件前，先读 [先算完再落盘](feedback_atomic_file_write.md)——
`open(p,'w')` 会立即清空原文件，内容生成与落盘不能放在同一语句序列里。
