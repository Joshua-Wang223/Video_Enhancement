# -*- coding: utf-8 -*-
"""子进程 stdin 加固：避免 ffmpeg 在后台进程组被 SIGTTOU 停住。

## 现象（2026-09-14 实测，strace 定位）

在容器 / CI / 调度器里以后台方式启动时，ffmpeg 会在启动阶段对 **fd 0** 调用
``ioctl(0, TCSETS, ...)`` 配置终端。若本进程不在**前台进程组**，内核会返回
``ERESTARTSYS`` 并投递 ``SIGTTOU``，进程随即被**停住**：

```
ioctl(0, TCSETS, {..}) = ? ERESTARTSYS
--- SIGTTOU {si_signo=SIGTTOU, si_code=SI_KERNEL} ---
--- stopped by SIGTTOU ---
```

表现极具误导性：命令"秒卡"，``0.01s user 0.01s system 0% cpu``，stderr 一个字都
没有，看起来像码流损坏 / GPU 挂死 / 死锁，实际与它们无关。``ffmpeg -version``
正常、``ffprobe`` 正常（ffprobe 不碰终端），只有真正解码的 ffmpeg 会卡。

## 处置

1. **入口点调用** :func:`detach_background_stdin` —— 仅当 stdin 是 tty 且本进程
   不在其前台进程组时，把 fd 0 换成 ``/dev/null``。一次修好**所有**子进程，
   且在前台交互运行时不改变任何行为（保留 ctrl-c 等终端语义）。
2. **显式给子进程 stdin 指向 ``/dev/null``**：使用本模块的 :data:`FFMPEG_SAFE_KW`
   展开进 ``subprocess.run/Popen``（见两个后端 ``ffmpeg_io.py`` 里各 ffmpeg 调用点）
   —— 库被直接 import 使用、绕过入口点时同样安全。

   不适用 / 被排除的调用点（都是有意的）：
   - ``subprocess.run(..., input=...)``：``input`` 与 ``stdin`` 互斥，同时传会
     ``ValueError``（见两后端 ``_probe_nvdec`` 的第二次调用）。
   - 写帧器的 ``Popen(..., stdin=subprocess.PIPE)``：它要用 stdin 灌帧，语义相反。
   - ``ffprobe`` 调用：实测 ffprobe 不碰终端，不受 SIGTTOU 影响。
   - ffmpeg-python 的 ``.run_async()``：本环境安装版本的签名是固定的
     ``run_async(stream_spec, cmd, pipe_stdin, pipe_stdout, pipe_stderr, quiet, overwrite_output)``，
     **没有** ``**kwargs``，无法透传 ``stdin``（传了会 ``TypeError``）；
     该路径只靠第 1 条（模块导入时的 fd0 加固）覆盖。
     另：`external/realesrgan_video/ffmpeg_io.py::FFmpegReader.__init__` 已在
     「真正拉起 ffmpeg 之前」**再补一次** :func:`detach_background_stdin`（标记
     ``[FIX-STDIN-TTOU-L2]``），把这条唯一缺口少掉的那层兜底补齐（幂等，零副作用）。

## 契约（2026-09-15 定稿，使用者已确认接受）

**加固会替换的是「父进程」的 fd 0**，不只是子进程的。具体语义：

| 场景 | fd 0 的处理 | 对调用方的影响 |
|---|---|---|
| stdin 非 tty（CI / nohup / systemd / docker 无 `-t` / 管道） | **不动** | 无 |
| stdin 是 tty 且本进程在**前台**进程组 | **不动** | 无（保留 ctrl-c、终端交互语义） |
| stdin 是 tty 且本进程在**后台**进程组 | 父进程 fd 0 → `/dev/null` | 见下 |
| 取不到前台进程组 | 保守按上一条处理 | 同上 |

**「后台运行但仍期望从控制终端读输入」这一用法本就不成立**：内核规定，后台进程组
（非前台进程组）对控制终端发起**读**操作会被 `SIGTTIN` 停住 —— 也就是说这条通路在加固
之前就是**不可用**的。因此把已被关掉的 fd 0 换成 `/dev/null` **不会损失任何本来可用的
能力**（实测方式：fork 子进程 → `os.setpgid(0, 0)` 使其成为后台组 → `os.read(0, 1)`，
观察是否被 `SIGTTIN` 停住；对照组把 fd 0 换成 `/dev/null` 后读立即 EOF）。

⇒ **调用方与任何外层编排都不应依赖**「以后台方式启动、进程仍从前台终端读取输入」。
本项目的流水线不用 stdin 读输入（全仓无 `input()` / `sys.stdin` 使用），故当前无影响。

> 决策记录：曾评估「是否把加固收窄到只作用于 ffmpeg 子进程、保留父进程 fd 0」，
> 结论是**维持现状** —— 收窄后所有**未被显式加固**的子进程调用点（含第三方库内部自己
> 拉 ffmpeg 的情况）会重新暴露在 SIGTTOU 下，收益不抵风险。
> 详见 `Plan/stdin加固策略定稿_立项Prompt.md`。
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict

#: 供 ffmpeg 等子进程复用的安全 kwargs（避免各处重复字面量）。
#: 用法：``subprocess.Popen(cmd, stdout=..., stderr=..., **FFMPEG_SAFE_KW)``。
#: 值类型取 ``Any``（而非 ``object``）：本字典用于 ``**`` 展开进 Popen/run，
#: ``Dict[str, object]`` 会让类型检查器判定实参类型不兼容而报错。
FFMPEG_SAFE_KW: Dict[str, Any] = {"stdin": subprocess.DEVNULL}


def detach_background_stdin() -> bool:
    """若 stdin 是「非前台」tty，则把 fd 0 重定向到 /dev/null。

    Returns:
        True = 做了重定向；False = 无需或无法重定向（前台 tty / 非 tty / 系统调用失败）。

    说明:
        - 前台 tty 不做处理：此时 ffmpeg 的 ``ioctl(TCSETS)`` 合法，且保留
          终端交互语义（如 ffmpeg 的交互式按键、tty 尺寸探测）。
        - ``os.tcgetpgrp`` 不会被 SIGTTOU 影响（只有 TCSETS/TCSETSW 这类
          **写侧**终端控制才是 SIGTTOU 的触发源），因此用它探测是安全的。
        - 全程吞异常：加固失败必须退化为原行为，绝不阻断主流程。
    """
    try:
        if not os.isatty(0):
            return False
        try:
            fg = os.tcgetpgrp(0)
        except OSError:
            # 无法获取前台进程组（无控制终端等）→ 保守视为需要加固
            fg = -1
        if fg == os.getpgrp():
            return False
        fd = os.open(os.devnull, os.O_RDONLY)
        try:
            os.dup2(fd, 0)
        finally:
            os.close(fd)
        return True
    except OSError:
        return False
    except Exception:
        return False
