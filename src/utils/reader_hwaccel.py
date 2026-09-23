# -*- coding: utf-8 -*-
"""读帧器 hwaccel 自适应决策 —— 两个后端（ifrnet_video / realesrgan_video）的唯一真源。

与 ``quality_map.py`` 同样的组织方式：本模块放在 ``src/utils``，由各后端
``main.py`` 在包入口把 ``src/utils`` 挂进 ``sys.path`` 后 import，避免在两个
独立包里各存一份阈值（否则刻度会漂移）。本模块不依赖 torch/ffmpeg，纯函数。

## 为什么需要这层：「有 NVDEC 就用」并不成立

NVDEC 只接管「熵解码 + 运动补偿」这一段。之后帧仍要：
    hwdownload（12.4MB/帧 @4K NV12）→ swscale nv12→rgb24（单线程）
    → 管道写 rgb24（24.9MB/帧）→ Python 侧逐帧读
而纯 CPU 软解把这些 CPU 环节合并成一条更短的通路，且能吃满多核。
因此**码率越低、物理核越多，软解越占优**；反之高码率（如高码率 HEVC）
熵解码开销大，NVDEC 的收益才体现出来。

## 实测标定（2026-09-14，Tesla T4 + 8 vCPU，真实 FFmpegFrameReader 消费者，
## 每档 3 轮取中位数）

| 素材 | bits/px/frame | NVDEC | CPU 软解 | 结论 |
|---|---|---|---|---|
| 3840x2160 23.98fps 8.5Mbps | 0.043 | 26.35s | 16.73s | **软解 1.58x** |
| 1920x1080 23.98fps 3.9Mbps | 0.079 | 4.33s | 3.94s | 软解 1.10x |
| 1920x1080 30fps 15.4Mbps | 0.248 | 6.38s | 6.22s | 持平（阈值两侧各差 ≤3%）|

阈值 0.15 落在「软解明确胜出(0.079)」与「持平(0.248)」之间，两侧误判代价都
很小（≤3%），故该阈值偏保守是可接受的。

## 决策输入只取两个单调可解释的量

    bits_per_px_frame = bit_rate / (width × height × fps)

再加物理核数下限。不做运行时试跑标定（代价高于收益）。
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

# 环境变量强制覆盖：auto（默认，走启发式）| on（强制硬解）| off（强制软解）
#
# [FIX-ENV-NAME-SCOPE] 正式名为中性的 READER_HWACCEL：本模块是 ifrnet_video 与
# realesrgan_video 两个后端读帧器共用的「单一真源」，用 IFRNET_ 前缀会让人误以为
# 只影响 IFRNet 侧（实际同时控制 ESRGAN 读帧器）。
# 旧名 IFRNET_READER_HWACCEL 保留兼容（历史命令/脚本/文档里用过），
# 两者都设时以新名优先，并在值冲突时提示一次。
READER_HWACCEL_ENV = 'READER_HWACCEL'
READER_HWACCEL_ENV_LEGACY = 'IFRNET_READER_HWACCEL'

_ENV_SCOPE_WARNED = False

# 每像素每帧比特数阈值（实测标定，见模块 docstring 表格）
BPP_THRESHOLD = 0.15

# 软解需要吃满的物理核数下限。核越多软解越占优（解码线程线性扩展），
# 核少时软解会与流水线其它 CPU 工作争抢，故不推荐。
CPU_MIN_CORES = 8


def cpu_core_count() -> int:
    """可用物理核数（考虑 cgroup/cpu affinity 亲和限制）。"""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _env_override() -> Tuple[Optional[str], str]:
    """解析环境变量覆盖 → (状态, 生效的变量名)。

    状态取值: 'on' / 'off' / None（未设置或 auto/非法值）。
    优先级: ``READER_HWACCEL`` > ``IFRNET_READER_HWACCEL``（旧名兼容）。
    两个变量都设置且取值冲突时，新名生效并提示一次（不失败）。
    """
    global _ENV_SCOPE_WARNED
    new = os.environ.get(READER_HWACCEL_ENV)
    old = os.environ.get(READER_HWACCEL_ENV_LEGACY)

    if new is None and old is None:
        return None, ''
    src = READER_HWACCEL_ENV if new is not None else READER_HWACCEL_ENV_LEGACY
    raw = (new if new is not None else old) or ''
    raw = raw.strip().lower()

    if (not _ENV_SCOPE_WARNED and new is not None and old is not None
            and new.strip().lower() != old.strip().lower()):
        _ENV_SCOPE_WARNED = True
        print('[读帧器] ⚠️  %s=%r 与旧名 %s=%r 取值冲突，以新名 %s 为准'
              % (READER_HWACCEL_ENV, new, READER_HWACCEL_ENV_LEGACY, old,
                 READER_HWACCEL_ENV), flush=True)
    elif new is None and old is not None and not _ENV_SCOPE_WARNED:
        _ENV_SCOPE_WARNED = True
        print('[读帧器] ℹ️  检测到旧变量名 %s；正式名已改为 %s（旧名仍兼容）'
              % (READER_HWACCEL_ENV_LEGACY, READER_HWACCEL_ENV), flush=True)

    if raw in ('off', '0', 'false', 'no'):
        return 'off', src
    if raw in ('on', '1', 'true', 'yes'):
        return 'on', src
    return None, src


def decide_reader_hwaccel(width: int, height: int, fps: float,
                          bit_rate: int = 0, nvdec_available: bool = True,
                          cpu_cores: Optional[int] = None,
                          label: str = '', quiet: bool = False) -> bool:
    """返回读帧器是否应使用 NVDEC 硬解。

    判据（按顺序）：
      1. ``READER_HWACCEL=off/0`` → 强制软解；``=on/1`` → 强制硬解。
         （旧名 ``IFRNET_READER_HWACCEL`` 仍兼容，新名优先，见 _env_override）
      2. 无可用 NVDEC → 软解。
      3. 码率未知（<=0）→ 维持硬解（保守：与历史行为一致，不引入回归）。
      4. 物理核数 >= CPU_MIN_CORES 且 bits/px/frame <= BPP_THRESHOLD
         → 软解更快，返回 False。
      5. 其余 → 硬解。

    Args:
        width/height/fps: 视频流参数（fps<=0 视为未知，退化为只看码率）。
        bit_rate:         视频流码率（bps），0/None 表示未知。
        nvdec_available:  调用方探测的 NVDEC 可用性。
        cpu_cores:        物理核数，None=自动探测。
        label:            日志附加标识（一般传文件名）。
        quiet:            True 时不打印决策行。

    Returns:
        True = 用 NVDEC；False = 用 CPU 软解。
    """
    env, env_src = _env_override()
    if env == 'off':
        if not quiet:
            print('[读帧器] hwaccel=off（%s 强制软解）%s'
                  % (env_src, (' ' + label) if label else ''), flush=True)
        return False

    force_on = (env == 'on')
    if force_on and not quiet:
        print('[读帧器] hwaccel=on（%s 强制硬解）%s'
              % (env_src, (' ' + label) if label else ''), flush=True)
    if not force_on and not nvdec_available:
        return False

    cores = cpu_cores or cpu_core_count()
    try:
        bpp = float(bit_rate) / float(max(1, int(width)) * max(1, int(height))
                                      * max(1.0, float(fps)))
    except (TypeError, ValueError, ZeroDivisionError):
        bpp = None

    if force_on:
        # 决策行已在上面按「哪个变量生效」打印过，此处不再重复
        return True

    if bpp is None or bpp <= 0:
        if not quiet:
            print('[读帧器] hwaccel=NVDEC（码率未知，保守维持硬解）'
                  + ((' ' + label) if label else ''), flush=True)
        return True

    use_hw = not (cores >= CPU_MIN_CORES and bpp <= BPP_THRESHOLD)
    if not quiet:
        print('[读帧器] hwaccel=%s（%dx%d %.1ffps %.1fMbps bits/px/frame=%.3f '
              'cores=%d，软解阈值 %.2f 且 >=%d 核）%s'
              % ('NVDEC' if use_hw else 'CPU软解', width, height, fps,
                 float(bit_rate) / 1e6, bpp, cores,
                 BPP_THRESHOLD, CPU_MIN_CORES,
                 (' ' + label) if label else ''), flush=True)
    return use_hw
