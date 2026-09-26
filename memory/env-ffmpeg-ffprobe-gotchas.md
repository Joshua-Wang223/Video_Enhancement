---
name: 本容器 ffmpeg/ffprobe 的两个环境坑（非代码缺陷）
description: ffmpeg 因 stdin 是后台终端会被 SIGTTOU 停住（必须重定向 /dev/null）；本 build 的 ffprobe 不提供 -hwaccel 选项
type: project
---

本容器（CloudStudio/T4）里跑 ffmpeg 实测时有两个**环境层**的坑，不是被测代码的问题。本会话两项各消耗了数十分钟，务必先记住。

## 1. ffmpeg 会被 SIGTTOU 停住 → 必须 `< /dev/null`

症状：`ffmpeg -f lavfi -i testsrc -f null -` 之类的命令**立即**卡死，`0.01s user 0.01s system 0% cpu`，stderr 一个字都没有；`ffmpeg -version` 与 `ffprobe` 却正常。

根因（strace 实锤）：进程处于**后台进程组**时，ffmpeg 在启动阶段对 fd 0 调用 `ioctl(0, TCSETS, ...)` 配置终端，内核回 `ERESTARTSYS` 并投递 `SIGTTOU` → 进程被停住：

```
ioctl(0, TCSETS, {..}) = ? ERESTARTSYS (To be restarted if SA_RESTART is set)
--- SIGTTOU {si_signo=SIGTTOU, si_code=SI_KERNEL} ---
--- stopped by SIGTTOU ---
```

**已落地永久修复（2026-09-14）** —— 不必再逐处手加重定向，代码里已覆盖：

- 新增 `src/utils/stdin_hardening.py::detach_background_stdin()`：**仅当** stdin 是 tty
  且本进程不在其前台进程组时，把 fd 0 换成 `/dev/null`（一次修好整个进程树的子进程；
  前台交互运行是纯 no-op）。
- 接线位置：
  - `run.py`、`src/main_video_optimized.py`（入口）；
  - 两个后端的 `ffmpeg_io.py` **模块导入时**（该模块是"拉起 ffmpeg"的归属地，一次覆盖
    能力探测 `_probe_nvdec`/`_probe_nvenc`、读帧器、写帧器，避免漏改调用点）；
  - 两个后端 `main.py` 的 `__main__`、两个 processor 的 `main()`、
    `Accessory/verify/segment_bitstream_verify_v4.py::main()`。
- 另在 IFRNet 读帧器 `Popen` 显式传 `**FFMPEG_SAFE_KW`（= `stdin=subprocess.DEVNULL`，
  单一真源在 `stdin_hardening.py`，库被直接 import 时同样安全）。

⚠️ **`run_async` 是全仓唯一的例外**（2026-09-15 定稿）：Real-ESRGAN 读帧器的 ffmpeg 由
`ffmpeg-python.run_async()` 拉起，而本环境该函数是**固定签名、无 `**kwargs`**
（`run_async(stream_spec, cmd, pipe_stdin, pipe_stdout, pipe_stderr, quiet, overwrite_output)`），
传 `stdin=` 会直接 `TypeError`。曾有一轮写了 `run_async(..., stdin=subprocess.DEVNULL)`
导致 ESRGAN 读帧器**完全起不来**（`_read_loop` 的 `except` 把它吞成"读取 0 帧 + 一行 traceback"）。
现以「补一次幂等加固」解决：`FFmpegReader.__init__` 在 `self._thread.start()` **之前**
再调一次 `detach_background_stdin()`（标记 `[FIX-STDIN-TTOU-L2]`）。

### 契约（2026-09-15 使用者拍板，勿再当作开放问题）

- 加固替换的是**父进程**的 fd 0（不只是子进程）。后台 tty 场景下，父进程 fd 0 会变成
  `/dev/null`；前台 tty 与非 tty 场景**完全不动**。
- **不损失能力**：后台进程组对控制终端发起**读**本来就被内核以 `SIGTTIN` 停住，
  该通路在加固前就不可用。可复现验证：fork → `os.setpgid(0,0)` → `os.read(0,1)`。
- ⇒ 调用方与任何外层编排**不得依赖**「后台启动但仍从终端读输入」。
  曾评估「收窄到仅子进程」，**已否决**（会让未显式加固的调用点重新暴露）。
- 完整背景、验收判据与落地记录：`Plan/stdin加固策略定稿_立项Prompt.md`。

⚠️ 教训：**只加固读帧器的 Popen 不够**。第一次只改了读帧器，脚本仍卡死 ——
`FFmpegFrameReader.__init__` 会经 `HardwareCapability.has_nvdec()` 调
`_probe_nvdec()`，那里**自己**会拉两次 ffmpeg（libx264 编码 + NVDEC 解码）用于能力探测，
没被加固就卡在这一步，且没有任何输出（连 `[读帧器]` 决策行都打不出来），极易误判为
"卡在读帧"。凡"某个模块会拉 ffmpeg"，就在该模块或全部入口统一加固。

排查时若见"ffmpeg 秒卡 + 0% CPU + 无输出"，先怀疑这条，别再怀疑码流或 GPU。
自检：`timeout 20 ffmpeg -v error -f lavfi -i testsrc=duration=1:size=64x64:rate=5 -f null -`

### ⚠️ 2026-09-16 新缺口：门禁自身曾整组被 SIGTTOU 停住（`[FIX-STDIN-TTOU-GATE]`）

**症状与上面同一类，但受害者是门禁进程本身**：`Accessory/verify/plan_implementation_gate.py`
在「后台进程组 + tty stdin」下跑到 A 阶段 R7「NVENC 环境探测」时**永久挂死、零输出**；
`ps` 显示**门禁主进程与其 ffmpeg 子进程双双为 `T`**（`wchan=do_signal_stop`），
即 SIGTTOU 把**整个进程组**停了 —— 因为 ffmpeg 在后台组里对 tty fd0 调
`ioctl(TCSETS)`，内核把 SIGTTOU 投给整个进程组。

**两层缺口**（缺任一层都不会中招，故此前未暴露）：

1. `run_cmd()`（本文件定义）**没有** `stdin=subprocess.DEVNULL` → 子 ffmpeg 继承 tty fd0。
2. `main()` 入口**没有**加固；唯一的 `detach_background_stdin()` 在
   `_setup_behavior_paths()`（行为阶段）里，而 R7 探测属 **A-前置条件**，跑在它**之前**。

**修复**：`run_cmd()` 加 `stdin=subprocess.DEVNULL` + `main()` 入口补一次加固（两层）。

**判别要点（可直接复用）**：

- 「命令跑得异常久 + 完全无输出」时，先看 `ps -o stat`：**`T` = 被停住**，不是卡在 I/O。
- 用 `wchan == do_signal_stop` 确认是 group-stop，而不是 D 态 I/O 等待。
- 用 `/proc/<pid>/stat` 的 `pgrp` 字段看**是不是整组都被停**（同组多进程同时 `T` ⇒ SIGTTOU）。
- 确定性复现/验收：`pty.fork()` 建控制终端 + 孙进程 `setpgid(0,0)` 造后台组，
  再 `exec` 目标命令；见本次会话的 `/tmp/repro_gate_stop.py`
  （broken 模式 45s 超时且 `T 态=[python, ffmpeg]`；fixed 模式 22.4s rc=0）。

**遗留（未修，属测试基建债）**：启发式扫描 `Accessory/` 下仍有 **28 个文件 / 102 处**
拉 ffmpeg 的子进程调用未显式传 `stdin=`。多数被「入口/模块导入时的加固」兜住，
但**凡在后台进程组被执行且入口未加固的脚本都可能整组被停住**。
结构化解法是抽一个共享 helper（等价于 `FFMPEG_SAFE_KW`）而不是逐处手改；
⚠️ 若改 `segment_bitstream_verify_v4.py`，必须**同步改 `_v5.py`**（两者须逐字节相同）。

## 2. `ffprobe` 不接受 `-hwaccel`

`ffprobe -h | grep hwaccel` 只列出 `-hwaccel_flags`；`-hwaccel` 不在 ffprobe 的 option 表里：

```
$ ffprobe -v error -hwaccel cuda -show_frames ... f.mp4
rc=1  Failed to set value 'cuda' for option 'hwaccel': Option not found
```

**How to apply:** 用 ffprobe 分析时不要加 `-hwaccel`；单帧/元数据读取本来也不需要它。
（这正是 `Accessory/verify/segment_bitstream_verify_v4.py::_decode_single_gpu_dual` 长期静默返回 None 的原因之一，另一个是 `-of csv=p=0` 与 `p=0` 前缀解析不匹配。）

## 附：两个文件事实

- `Accessory/verify/segment_bitstream_verify_v4.py` **未被 git 跟踪**（`git ls-files` 无记录），所以无法用 `git show HEAD:<path>` 取原版做 A/B 对照 —— 需要基线时请先自行备份副本。
- 仓库根目录的 `core.24882` / `core.60971` 与全量 `pytest Accessory/` 的 SIGSEGV 属**既有**现象：NVENC 硬件测试彼此状态隔离不足，单独跑各测试类可通过（实测 `TestLAAccumulation` 2 passed）。`nvenc_sdk.py` 只依赖 stdlib+numpy+torch+ctypes，与解码/读帧链路无耦合。
