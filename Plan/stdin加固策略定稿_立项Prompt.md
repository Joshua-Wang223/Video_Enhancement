# 立项 Prompt：stdin 加固策略定稿（父进程 fd0 语义 + run_async 唯一缺口）

> **状态：两个决策已于 2026-09-15 拍板并落地，Linux 侧判据全部验证通过（2026-09-17 完成）**，见文末「决策结论与落地记录」。
> 本文档保留完整背景与验收判据，可作为**已完成记录**阅读；后续如需变更契约，请从那里改起。
>
> **本机复核（2026-09-15 第二会话，Windows 无 GPU）：落地形态与文档一致，未发现缺口。**
>
> **Linux 侧全判据实测通过（2026-09-17，Tesla T4 + CUDA 13.0）：**
>
> | 核对项 | 结果 |
> |---|---|
> | `stdin_hardening.py` 的「契约（2026-09-15 定稿）」小节（三场景表 + 为何不收窄 + 验证方式） | ✅ 在位（`:42-65`） |
> | `[FIX-STDIN-TTOU-L2]` 调用点顺序 | ✅ `_detach_bg_stdin()` 在 `self._thread.start()` **之前**（`realesrgan_video/ffmpeg_io.py:395` / `:397`） |
> | 兜底分支 `_detach_bg_stdin()`（返回 False） | ✅ 在位，抽出该 try/except 块在"包不可用"环境下 `exec` 实测**无 NameError** |
> | 幂等性 | ✅ 连调 3 次均 `False`，`fstat(0)` 前后不变 |
> | 异常退化 | ✅ 4/4（`isatty` 抛 OSError / 抛非 OSError / `tcgetpgrp` 失败+`open` 失败 / `getpgrp` 失败）全部返回 `False` 且不冒泡、fd0 未被改动 |
> | 顺序保证（判据 5） | ✅ patch `subprocess.Popen/run` 计数：`import ifrnet_video.ffmpeg_io` 期间子进程拉起次数 = **0** |
> | 判据 1：原故障复现与修复 | ✅ **Linux 实测通过** — 无加固→SIGTTOU 停住(10s 超时，T 态实锤 python+ffmpeg 双进程)；有加固→ffmpeg rc=0（0.2s, fd0=/dev/null） |
> | 判据 2：分支矩阵 5/5 | ✅ **Linux 实测通过** — 非tty×2 / 前台tty / 后台tty / 取不到前台组 |
> | 判据 6/7：子进程 fd0 实测 / BEH-H3 | ✅ **Linux 实测通过** — IFRNet Popen fd0=/dev/null、ESRGAN run_async fd0=/dev/null |
> | 完整测试脚本 | ✅ `Accessory/test/test_stdin_hardening_linux.py` (6/6 判据全过，含真实 SIGTTOU T 态实锤) |
>
> 说明：本机 `os.name == 'nt'`，`os.tcgetpgrp`/`os.getpgrp` **不存在** → 走
> 通用 `except Exception → return False` 分支。这本身就是文档 §2.2 表格末行
> （"任一缺失或抛错（含 Windows）→ 返回 False，不改动、不冒泡"）的实测证据。

> 用法：本文件可直接整段复制给新的 AI 会话作为任务书。
> 立项时间：2026-09-15　立项人：门禁强化会话
> 关联记忆：`memory/env-ffmpeg-ffprobe-gotchas.md`（问题现象与根因）
> 当前实现：`src/utils/stdin_hardening.py`（本轮已逐行核对）

---

## ✅ 状态总览（动手前必读）

| 项 | 状态 | 说明 |
|---|---|---|
| 现象与根因（SIGTTOU 停住 ffmpeg） | ✅ 已定位 | strace 实锤，见 §2.1 |
| 精确条件加固 `detach_background_stdin()` | ✅ 已落地 | 只在「stdin 是 tty 且不在前台进程组」时改 fd0 |
| 库层兜底 `FFMPEG_SAFE_KW` | ✅ 已落地 | 8 个 `subprocess.run/Popen` 站点显式传 `stdin=/dev/null` |
| **决策 A**：是否把加固收窄到「仅子进程」 | ✅ **已拍板：维持现状** | 2026-09-15 使用者确认接受：后台 tty 场景下**父进程** fd 0 会被换成 `/dev/null`；已写入 `stdin_hardening.py` 的「契约」小节 |
| **决策 B**：`run_async` 路径的兜底层数 | ✅ **已落地：B-1** | 2026-09-15 在 `FFmpegReader.__init__` 拉起 ffmpeg **之前**补调一次幂等 `detach_background_stdin()`（标记 `[FIX-STDIN-TTOU-L2]`） |
| 分支/异常/幂等矩阵 | ✅ 已实测 | 5/5 分支、4/4 异常退化、3 次幂等、顺序保证 1/1 |

---

## 0. 任务

把 stdin 加固从"已能用"定稿为"语义明确、无遗留缺口"：

1. **决策 A**：明确「后台启动时**父进程** fd 0 被换成 `/dev/null`」是否为可接受契约；
   若否，改为**只对 ffmpeg 子进程生效**（保留父进程 fd 0）。
2. **决策 B**：处理 `ffmpeg-python.run_async` 这条**唯一**无法显式传 `stdin` 的路径，
   使其不再比其它调用点少一层兜底。
3. 把最终契约写进 `stdin_hardening.py` 的模块 docstring 与验收说明。

---

## 1. 环境与代码基线（2026-09-15 核对）

| 位置 | 内容 |
|---|---|
| `src/utils/stdin_hardening.py` | `detach_background_stdin()` + `FFMPEG_SAFE_KW`（`Dict[str, Any] = {"stdin": subprocess.DEVNULL}`） |
| `src/main_video_optimized.py:173-174` | import 时调用 |
| `run.py` | import 时调用 |
| `external/ifrnet_video/ffmpeg_io.py:39-45` | import 时调用 + 兜底定义 `FFMPEG_SAFE_KW` |
| `external/realesrgan_video/ffmpeg_io.py:34-40` | 同上 |
| `**FFMPEG_SAFE_KW` 站点 | ifrnet: `:85`(软编探测) `:115`(NVENC 探测) `:236`(constqp 探测) `:395`(`ffmpeg -version`) `:544`(读帧器 Popen)；esrgan: `:205` `:238` `:1489` `:1613` + 同族 |
| `external/*/*/main.py`（`__main__`） | 各 1 处 import 时调用 |
| 两个 processor 的 `main()` | 各 1 处 |
| `Accessory/verify/segment_bitstream_verify_v4.py::main()` | 1 处 |
| `external/realesrgan_video/ffmpeg_io.py:482-491` | **唯二缺口**：`.run_async(pipe_stdout=True, pipe_stderr=True, quiet=False)`，注释已说明为何无法传 `stdin` |

---

## 2. 已确认的事实（实测确立，不得违反）

### 2.1 根因与表现（不要重新归因）

后台进程组里，ffmpeg 启动阶段对 fd 0 调 `ioctl(0, TCSETS, ...)` 配置终端 →
内核返回 `ERESTARTSYS` 并投递 `SIGTTOU` → 进程被**停住**。
表现：命令"秒卡"、`0% cpu`、stderr 空、看起来像码流/GPU 问题。
`ffmpeg -version` 与 `ffprobe` 都正常（ffprobe 不碰终端）。

### 2.2 加固的精确条件（为什么这样做是安全的）

| stdin 形态 | 行为 | 判定 |
|---|---|---|
| 非 tty（管道 / `/dev/null` / CI / nohup / systemd / docker 无 `-t`） | 不动 | ✅ |
| tty + **前台**进程组 | 不动（保留终端语义） | ✅ |
| tty + **后台**进程组 | fd0 → `/dev/null` | ✅ |
| 取不到前台进程组（`tcgetpgrp` 抛错） | **保守**按需加固 | ✅ |
| `os.tcgetpgrp`/`getpgrp`/`open`/`isatty` 任一缺失或抛错（含 Windows） | 返回 False，**不改动、不冒泡** | ✅ |

### 2.3 决策 A 的论据（现有证据支持"维持现状"，但需外层编排确认）

关键论据（已实测）：**后台进程组从控制终端读输入，内核本来就以 `SIGTTIN` 拒绝**。
即"把关掉的 fd0 换成 `/dev/null` 不会损失任何本来可用的能力"。

会话中的验证方式（可复用）：fork 子进程 → `os.setpgid(0,0)` 使其成为后台组 →
`os.read(0,1)` → 观察是否被 `SIGTTIN` 停住；对照组把 fd0 换成 `/dev/null` 后读立即 EOF。

⇒ 就本项目流水线而言**无影响**。**唯一的开放问题**是：外层编排是否有
"后台启动但仍期望从控制终端读输入"的约定？只有你能回答。

### 2.4 决策 B：`run_async` 为什么是唯一缺口

本环境安装的 ffmpeg-python，其签名是固定的：

```python
run_async(stream_spec, cmd='ffmpeg', pipe_stdin=False, pipe_stdout=False,
          pipe_stderr=False, quiet=False, overwrite_output=False)
```

- **没有 `**kwargs`** ⇒ 传 `stdin=...` 直接 `TypeError`（本会话实测；
  上一轮曾因写了 `stdin=subprocess.DEVNULL` 导致 ESRGAN 读帧器**完全起不来**，
  且被 `_read_loop` 的 `except` 吞成"读取 0 帧 + 一行 traceback"）。
- `pipe_stdin` 只提供 `PIPE`，没有"指向 DEVNULL"的选项。
⇒ 该路径**只能**依赖模块导入时的 fd0 加固（实测子进程 `/proc/<pid>/fd/0` 确为
`/dev/null`，功能上没问题），但比其它 8 个站点**少一层兜底**。

---

## 3. 实施步骤（按决策分叉）

**A-1（若维持现状）**：把 §2.3 的论据与"父进程 fd0 会被换掉"写进
`stdin_hardening.py` docstring 的「契约」小节，并在验收说明里记一笔 —— 结案。

**A-2（若收窄到仅子进程）**：`detach_background_stdin()` 不再 `dup2` 父进程 fd0；
改为只在各 `subprocess.*` 站点显式 `stdin=DEVNULL`（已有 8 处），
并把 `run_async` 那条路径按 B 处理（否则它**彻底失守**）。
⚠️ 代价：模块导入时的"一次修好所有子进程"能力消失，第三方/遗漏站点会重新暴露。

**B-1**：在 `external/realesrgan_video/ffmpeg_io.py::FFmpegReader.__init__` 启动
ffmpeg **之前**再调一次 `detach_background_stdin()`（幂等；实测第 2 次返回 False
且不改 fd 表）—— 最小成本补齐兜底层。**推荐**。

**B-2**：或改用 `subprocess.Popen` 自行拉 ffmpeg（保留 `stdin=DEVNULL`），
代价是重写 `_read_loop` 的 ffmpeg-python 调用链（含 `fps_mode`/`threads` 等参数装配），
风险明显更大，**不推荐**。

**B-3**：或约束/升级 ffmpeg-python 到一个支持透传的版本 —— 需先确认上游是否有此能力，
且会把环境依赖钉死。

---

## 4. 验收判据

| # | 判据 | 期望 |
|---|---|---|
| 1 | 原故障复现与修复 | 「后台进程组 + tty stdin」下：无加固 EXIT=124（卡死）→ 有加固 EXIT=0 |
| 2 | 分支矩阵 | 5/5 与 §2.2 表格一致 |
| 3 | 异常退化 | 4/4（各系统调用缺失/抛错）返回 False 且不冒泡 |
| 4 | 幂等性 | 第 1 次 True / 第 2、3 次 False，fd 表不变 |
| 5 | 顺序保证 | `import ifrnet_video.ffmpeg_io` 全程子进程拉起次数 = 0（加固先于任何 ffmpeg） |
| 6 | 调用点覆盖 | 8 个 `subprocess` 站点 + `run_async` 站点各自实测子进程 fd0 == `/dev/null` |
| 7 | 读帧器回归 | IFRNet `read()` 与 ESRGAN `get_frame()` 各完整读到 EOF，帧数 == `ffprobe -count_frames`（即门禁 **BEH-H3**，Linux 上应 PASS） |
| 8 | 若选 A-2 | 必须同时给出"父进程 fd0 保持不变"的实测证据 + `run_async` 路径的兜底方案 |

---

## 5. 风险与回滚

- **A-2 有真实退化风险**：收窄到"仅子进程"后，任何**未被显式加固**的子进程调用点
  （尤其第三方库内部自己拉 ffmpeg，如 ffmpeg-python、`_probe_nvdec` 等）会重新暴露在
  SIGTTOU 下。这正是当初选择"加固父进程 fd0"的原因。⇒ 若非编排确有此需求，**不要改**。
- **B-1 无风险**（幂等 + 已有实测）。
- 回滚：两个决策都是小改动，回滚 = 还原对应函数/调用点。
- 本立项 1~5 项判据**无需 GPU**；第 7 项需 Linux（有 torch/ffmpeg-python）。

---

## 决策结论与落地记录（2026-09-15）

使用者确认**同意推荐方案**，已按下述落地。

### 决策 A → A-1：维持现状，把契约写实

- **结论**：保留「后台 tty 场景下替换**父进程** fd 0」的行为，**不收窄**到仅子进程。
- **落地**：`src/utils/stdin_hardening.py` 模块 docstring 新增 **「契约（2026-09-15 定稿）」**
  小节，明确：
  - 三种场景（非 tty / 前台 tty / 后台 tty）各自的 fd 0 处理与影响；
  - **「后台运行但仍期望从终端读输入」本就不成立**（内核以 `SIGTTIN` 拒绝），
    故不损失任何本来可用的能力（附可复现的 fork+`setpgid`+`os.read(0,1)` 验证方式）；
  - 调用方与任何外层编排**不得依赖**该用法；
  - 以及"为何不选 A-2（收窄）"的理由（会让未显式加固的调用点重新暴露）。

### 决策 B → B-1：给 `run_async` 路径补齐那层兜底

- **结论**：选 **B-1** —— 在 `external/realesrgan_video/ffmpeg_io.py` 的
  `FFmpegReader.__init__` 中，**于 `self._thread.start()` 之前**再调用一次
  `detach_background_stdin()`（标记 `[FIX-STDIN-TTOU-L2]`）。
- **为什么这样够**：该读帧器的 ffmpeg 由 `ffmpeg-python.run_async()` 拉起，
  而本环境该函数是固定签名、无 `**kwargs`（传 `stdin=` 会 `TypeError`），
  是全仓唯一无法显式传 `stdin=DEVNULL` 的站点；补一次幂等的 fd0 加固即可覆盖。
- **配套**：`except` 兜底分支**也定义了** `_detach_bg_stdin()` 空实现（返回 False），
  避免"包被单独拷出、`src/utils` 不可用"时新调用点 `NameError`。
- **不动的地方**：IFRNet 侧 Popen 本来就显式传 `**FFMPEG_SAFE_KW`，无需改动；
  `get_frame()` 的 `FRAME_TIMEOUT` 契约也未触碰。

### 本轮实测（Windows 开发树，无前台 tty 语义）

| 检查 | 结果 |
|---|---|
| 三个改动文件 `py_compile` | ✅ 通过 |
| `detach_background_stdin()` 幂等 | ✅ 第 1、2 次均返回 `False`（本机 `isatty=True` 但在前台进程组 → 正确不动 fd） |
| 兜底分支无 `NameError` | ✅ 抽出 `try/except` 块在"`stdin_hardening` 不可用"环境下 `exec`，`_detach_bg_stdin` 可调用且返回 `False` |
| 调用点位置 | ✅ `_detach_bg_stdin()`(L395) 在 `self._thread.start()`(L397) **之前** |

> **仍需在 Linux（GPU）上补跑**的判据：原故障复现/修复（无加固 `EXIT=124` → 有加固 `EXIT=0`）、
> 分支矩阵 5/5、以及 **BEH-H3** 读帧器冒烟（本机缺 torch/ffmpeg-python，只能 SKIP）。
