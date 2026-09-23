#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# IFRNet Video Enhancement - FFmpeg 读写模块（解码器 / 编码回退 / 硬件探测）。
# 镜像 external/realesrgan_video/ffmpeg_io.py 的职责。

from __future__ import annotations

import functools
import os
import queue
import subprocess
import sys
import threading
import time
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

# 质量参数换算的唯一真源（src/utils/quality_map.py → convert_crf.py）。
# 依赖 main.py 在包入口处把 src/utils 挂进 sys.path —— 本包只经 main.py 进入，
# 单独 import 本模块不成立。
from quality_map import to_constqp_qp

# [P2-2] 读帧器 hwaccel 自适应决策：与 quality_map 同样取自 src/utils 的唯一
# 真源（避免 ifrnet_video / realesrgan_video 两侧各存一份阈值而漂移）。
from reader_hwaccel import (decide_reader_hwaccel,   # noqa: F401
                            cpu_core_count as _reader_cpu_cores)

# [FIX-STDIN-TTOU] 本模块是「拉起 ffmpeg 子进程」的归属地，在此做一次幂等的
# fd0 加固，一次覆盖本模块内全部 ffmpeg 调用（能力探测 _probe_nvdec/_probe_nvenc、
# 读帧器、写帧器…），避免逐个调用点补 `stdin=subprocess.DEVNULL` 时漏改。
# 仅在「stdin 是 tty 且本进程不在其前台进程组」时动手，前台交互运行是纯 no-op；
# 原因与现象见 src/utils/stdin_hardening.py。导入失败不影响本包可用性。
try:
    from stdin_hardening import (FFMPEG_SAFE_KW,
                                 detach_background_stdin as _detach_bg_stdin)
    _detach_bg_stdin()
except Exception:
    # src/utils 不可用（包被单独拷出等）→ 本地兜底一份等价 kwargs，
    # 保证长驻子进程仍显式拿到 /dev/null。
    FFMPEG_SAFE_KW = {'stdin': subprocess.DEVNULL}

from ifrnet_video.ifrnet_utils import _clamp_decode_threads
from ifrnet_video.nvenc_sdk import _PRESET_P_INDEX

# [FIX-RANGE-VALIDATE] 非零 frame_start 的「整段解码再丢弃」告警只打印一次
# （批量处理时同一路径会反复构造 reader，逐次刷屏会淹没日志）。
_RANGE_SLOWPATH_WARNED = False


# ─────────────────────────────────────────────────────────────────────────────
# M2/M3: 硬件能力探测
# ─────────────────────────────────────────────────────────────────────────────

class HardwareCapability:
    _nvdec: Optional[bool] = None
    _nvenc: Dict[str, bool] = {}

    @classmethod
    def has_nvdec(cls) -> bool:
        if cls._nvdec is None:
            cls._nvdec = cls._probe_nvdec()
        return cls._nvdec

    @classmethod
    def has_nvenc(cls, codec: str = 'h264_nvenc') -> bool:
        if codec not in cls._nvenc:
            cls._nvenc[codec] = cls._probe_nvenc(codec)
        return cls._nvenc[codec]

    @staticmethod
    def _probe_nvdec() -> bool:
        """[FIX-NDV] 两阶段真实探测：先软件编码 H.264，再用 NVDEC 实际解码。"""
        try:
            enc_cmd = [
                'ffmpeg', '-f', 'lavfi',
                '-i', 'testsrc=size=64x64:duration=0.04:rate=25',
                '-vcodec', 'libx264', '-f', 'h264', 'pipe:1', '-loglevel', 'error',
            ]
            enc = subprocess.run(enc_cmd, capture_output=True, timeout=10,
                                  **FFMPEG_SAFE_KW)
            if enc.returncode != 0 or not enc.stdout:
                return False
            dec_cmd = [
                'ffmpeg', '-hwaccel', 'cuda',
                '-f', 'h264', '-i', 'pipe:0',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-frames:v', '1', 'pipe:1', '-loglevel', 'error',
            ]
            dec = subprocess.run(dec_cmd, input=enc.stdout, capture_output=True, timeout=10)
            return dec.returncode == 0 and len(dec.stdout) > 0
        except Exception:
            return False

    @staticmethod
    def _probe_nvenc(codec: str) -> bool:
        # [FIX-PROBE] 历次修复汇总（根因均由 stderr 诊断确认）：
        #   LAVFI  : color 源去掉 d=0.1（0帧问题）→ 无限源 + -frames:v 1 截帧
        #   PIXFMT : FFmpeg 5.0+ color 源默认 yuv444p → 显式 -pix_fmt yuv420p
        #   BFRAME : 单帧编码触发 B 帧 lookahead 报错 → -bf 0
        #   MINDIM : h264_nvenc 宽≥145px / hevc_nvenc 宽≥129px → 256×144
        cmd = [
            'ffmpeg', '-hide_banner', '-y', '-loglevel', 'error',
            '-f', 'lavfi', '-i', 'color=c=black:s=256x144:r=1',
            '-vcodec', codec, '-frames:v', '1',
            '-pix_fmt', 'yuv420p', '-bf', '0',
            '-f', 'null', '-',
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10,
                                    **FFMPEG_SAFE_KW)
            if result.returncode != 0:
                _err = result.stderr.decode('utf-8', errors='replace').strip()
                print(
                    f'  [PROBE-FAIL] {codec} probe 失败 (rc={result.returncode})\n'
                    f'  手动测试: {" ".join(cmd)}\n'
                    f'  stderr: {_err or "(空)"}',
                    flush=True,
                )
            return result.returncode == 0
        except Exception as e:
            print(
                f'  [PROBE-FAIL] {codec} probe 异常: {e}\n'
                f'  手动测试: {" ".join(cmd)}',
                flush=True,
            )
            return False

    @classmethod
    def best_encoder(cls, preferred: str = 'libx264',
                     hw_profile: Optional['_HWProfile'] = None) -> str:
        """
        [FIX-NVENC-UNIFIED] 统一 NVENC 检测路径。

        优先级（按 hw_profile 是否提供分两条路径）：

        路径 A（hw_profile 已提供，GPU 型号已知）：
          1. 直接采信 hw_profile.has_nvenc（静态 GPU 型号表，与 AUTO-TUNE 一致）。
          2. 若静态表确认可用且 ffmpeg probe 失败（Docker 设备映射缺失常见场景），
             打印明确警告后仍信任静态表——probe 失败不等于硬件不可用。
             可将 _NVENC_TRUST_STATIC = False 改为强制要求 probe 通过。
          3. 静态表否定（has_nvenc=False）时不做 probe，直接回退软件编码。

        路径 B（hw_profile 未提供，GPU 型号未知）：
          仅凭 ffmpeg 实际 probe 判断，probe 失败即回退软件编码。
          此路径适用于无法识别 GPU 型号的环境（非 NVIDIA GPU 等）。

        两套检测结果在 Docker 未映射 NVENC 设备时会不一致：AUTO-TUNE 显示 nvenc=True
        但 ffmpeg probe 失败，导致实际使用 libx264 引发 T3 瓶颈。本修复确保两者一致。
        """
        nvenc_map    = {'libx264': 'h264_nvenc', 'libx265': 'hevc_nvenc'}
        fallback_map = {'h264_nvenc': 'libx264', 'hevc_nvenc': 'libx265'}

        # [FIX-NVENC-TRUST-STATIC] 当 probe 失败但静态表确认可用时，是否信任静态表。
        # True（默认）：信任静态表，适用于 Docker 设备映射缺失但硬件真实存在的场景。
        # False：强制要求 probe 通过，适用于需要 100% 运行时验证的严格环境。
        _NVENC_TRUST_STATIC: bool = True

        def _nvenc_ok(codec_name: str) -> bool:
            # [FIX-NVENC-UNIFIED] 路径 A：hw_profile 已知时优先信任静态 GPU 型号表。
            # 这与 docstring 描述的优先级一致，解决了原实现 probe-first 导致
            # Docker 环境中 hw_profile.has_nvenc=True 被静默忽略的问题。
            if hw_profile is not None and hasattr(hw_profile, 'has_nvenc'):
                if not hw_profile.has_nvenc:
                    # 静态表明确否定：不做 probe，直接返回 False
                    return False
                # 静态表确认可用：尝试 probe 做二次验证
                probe_ok = cls.has_nvenc(codec_name)
                if not probe_ok:
                    # [FIX-NVENC-PROBE-WARN] probe 失败但静态表确认硬件存在。
                    # 常见于 Docker 容器 /dev/nvidia* 设备映射不完整，
                    # nvidia-smi 可见 GPU 但 ffmpeg 无法访问 NVENC 编码器。
                    # 根据 _NVENC_TRUST_STATIC 决定是否信任静态表。
                    if _NVENC_TRUST_STATIC:
                        print(
                            f'  [FIX-NVENC-UNIFIED] {codec_name} ffmpeg probe 失败，'
                            f'但静态 GPU 型号表确认硬件存在（hw_profile.has_nvenc=True）。\n'
                            f'  信任静态表，继续使用 {codec_name}（Docker 设备映射不完整时正常）。\n'
                            f'  若实际编码报错，可在代码中将 _NVENC_TRUST_STATIC 改为 False。'
                        )
                        return True
                    else:
                        print(
                            f'  [FIX-NVENC-UNIFIED] {codec_name} ffmpeg probe 失败。'
                            f'静态表显示硬件存在但 _NVENC_TRUST_STATIC=False，回退软件编码。'
                        )
                        return False
                return True  # 静态表 + probe 双重确认

            # [FIX-NVENC-UNIFIED] 路径 B：hw_profile 未知，仅凭 ffmpeg probe 判断。
            return cls.has_nvenc(codec_name)

        if preferred in fallback_map:
            if _nvenc_ok(preferred):
                return preferred
            fallback = fallback_map[preferred]
            # [FIX-NVENC-WARN] 明确说明回退原因（probe 失败 or 静态表否定 or 无 profile）
            if hw_profile is not None and hasattr(hw_profile, 'has_nvenc') and not hw_profile.has_nvenc:
                reason = '静态 GPU 型号表标记 has_nvenc=False'
            elif not cls.has_nvenc(preferred):
                reason = 'ffmpeg probe 失败（Docker 设备映射可能缺失 /dev/nvidia*）'
            else:
                reason = '未知原因'
            print(f'  [警告] {preferred} 不可用（{reason}），自动回退到 {fallback}')
            return fallback
        candidate = nvenc_map.get(preferred, preferred)
        if candidate != preferred and _nvenc_ok(candidate):
            return candidate
        return preferred

    @classmethod
    def lossless_encoder(cls) -> Tuple[str, List[str]]:
        """
        返回 (codec, extra_args) 用于无损中间段编码。
        优先 nvenc lossless（-rc constqp -qp 0），否则 libx264 lossless。

        与 best_encoder 的区别：best_encoder 只探测 NVENC 是否可用（VBR 编码），
        lossless_encoder 额外测试 constqp（常量 QP）模式——部分 GPU/driver 组合
        虽支持 NVENC VBR 但不支持 constqp 无损。
        """
        if cls.has_nvenc('h264_nvenc'):
            try:
                cmd = [
                    'ffmpeg', '-hide_banner', '-y', '-loglevel', 'error',
                    '-f', 'lavfi', '-i', 'color=c=black:s=256x144:r=1',
                    '-vcodec', 'h264_nvenc',
                    '-rc', 'constqp', '-qp', '0',
                    '-pix_fmt', 'yuv420p', '-bf', '0',
                    '-frames:v', '1', '-f', 'null', '-',
                ]
                if subprocess.run(cmd, capture_output=True, timeout=10,
                                  **FFMPEG_SAFE_KW).returncode == 0:
                    return 'h264_nvenc', ['-rc', 'constqp', '-qp', '0']
            except Exception:
                pass
        return 'libx264', ['-qp', '0', '-preset', 'medium']

_X264_PRESET_FACTOR = {
    'ultrafast': 8.0, 'superfast': 6.0, 'veryfast': 4.0,
    'faster': 2.5, 'fast': 2.0, 'medium': 1.0,
    'slow': 0.4, 'slower': 0.2, 'veryslow': 0.1,
}
# [FIX-PRESET-UNIFY] x264 名称 → NVENC p-index 统一映射（复用 nvenc_sdk._PRESET_P_INDEX）。
# NVENC 使用 p1(最快)~p7(最慢) 命名体系，与 x264 的 ultrafast~veryslow 不兼容。
# 统一口径后与 SDK Level 1 直通路径完全一致（medium→p5、fast→p4、slow→p6 等），
# 避免 FFmpeg CLI 层与 SDK 层映射不一致导致的后端降级档位跳变。
# [FIX-CRF0-CALIB] crf=0（lossless）实测校准因子。
# 理论模型（crf_factor = 2^((0-23)/12) ≈ 0.264）严重低估 lossless 编码成本：
#   · lossless 需维持精确像素，内存带宽和预测搜索开销远高于有损编码
#   · 实测（T4, libx264, ultrafast, 416×736, 8c）: ~150 fps output
#   · 理论估算（修正前）:  ~2860 fps → 偏差约 19×
# 乘以此因子后估算 ~157 fps，贴近实测正常（非热节流）状态。
_CRF0_X264_CALIB_FACTOR: float = 0.055


def _software_encode_fps(cpu_cores: int, H: int, W: int,
                         codec: str, preset: str, crf: int) -> float:
    base_pixels = 1920.0 * 1080.0
    current_pixels = float(H * W)
    scale_res = max(base_pixels / current_pixels, 1.0)
    factor = _X264_PRESET_FACTOR.get(preset, 1.0)
    crf_factor = 2.0 ** ((crf - 23) / 12.0)
    base_fps = 120.0 if 'x265' in codec.lower() else 200.0
    cores_factor = min(cpu_cores, 16) / 8.0
    fps = base_fps * scale_res * factor * crf_factor * cores_factor
    # [FIX-CRF0-CALIB] lossless（crf=0）时理论模型严重低估编码成本，乘以实测校准因子
    if crf == 0 and 'nvenc' not in codec.lower():
        fps *= _CRF0_X264_CALIB_FACTOR
    return min(fps, 3000.0)

# ─────────────────────────────────────────────────────────────────────────────
# [FIX-SLICE-THREAD] 编码并行度自动探测
# ─────────────────────────────────────────────────────────────────────────────

def _detect_encode_parallelism(n_threads_hint: Optional[int] = None) -> dict:
    """
    [FIX-SLICE-THREAD] 自动探测 CPU / 内存资源，返回最优软编码并行参数字典。

    返回字段
    ─────────────────────────────────────────────────────────────────────────
    cpu_logical    : int    逻辑核心数（含超线程，来自 os.cpu_count()）
    cpu_physical   : int    物理核心数（来自 /proc/cpuinfo；失败则 logical//2）
    mem_avail_gb   : float  系统当前可用内存 GiB（来自 /proc/meminfo MemAvailable）
    encode_threads : int    软编码线程数（x264/x265 frame-level parallelism），
                            = min(cpu_logical, 16)，超过 16 收益递减
    slices         : int    x264 intra-frame 分片数（slice-based threading）：
                            每片由独立线程并行编码，降低单帧编码延迟。
                            = min(encode_threads, 16)，同时受内存可用量约束
                            （大分片数需更多行缓冲区；低分辨率时此约束通常不触发）
    ffmpeg_threads : int    FFmpeg 全局 -threads 值，用于 demux/filter graph
                            = min(cpu_logical, 8)
    """
    cpu_logical = os.cpu_count() or 4

    # 物理核心数：从 /proc/cpuinfo 读 "core id" 去重；失败则估算
    cpu_physical = max(cpu_logical // 2, 1)
    try:
        _core_ids: set = set()
        _pkg_ids:  set = set()
        _cur_pkg       = None
        with open('/proc/cpuinfo', 'r') as _cpuf:
            for _line in _cpuf:
                _line = _line.strip()
                if _line.startswith('physical id'):
                    _cur_pkg = _line.split(':', 1)[1].strip()
                elif _line.startswith('core id') and _cur_pkg is not None:
                    _core_ids.add((_cur_pkg, _line.split(':', 1)[1].strip()))
        if _core_ids:
            cpu_physical = len(_core_ids)
    except Exception:
        pass

    # 系统可用内存（GiB）：读 /proc/meminfo MemAvailable；失败时尝试 psutil
    mem_avail_gb = 4.0
    try:
        with open('/proc/meminfo', 'r') as _memf:
            for _line in _memf:
                if _line.startswith('MemAvailable:'):
                    mem_avail_gb = int(_line.split()[1]) / (1024.0 ** 2)
                    break
    except Exception:
        try:
            import psutil as _psutil
            mem_avail_gb = _psutil.virtual_memory().available / (1024.0 ** 3)
        except ImportError:
            pass

    # 若外部传入 hint，直接使用（但仍不超过 16）
    if n_threads_hint is not None and n_threads_hint > 0:
        encode_threads = min(n_threads_hint, 16)
    else:
        encode_threads = min(cpu_logical, 16)

    # 分片数 = encode_threads（1 slice/thread），但：
    #   · 上限 16：slice 数越多压缩率越低（片间参考受限），16 是实用阈值
    #   · 内存约束：每个 slice 约需 0.25 GiB 额外行缓冲（高分辨率下），低分辨率（<1080p）可忽略
    #   · 下限 2：至少 2 片才有并行效果
    slices_by_cpu  = encode_threads
    slices_by_mem  = max(2, int(mem_avail_gb / 0.25))   # 每 slice 估算 0.25 GiB
    slices = max(2, min(slices_by_cpu, slices_by_mem, 16))

    ffmpeg_threads = min(cpu_logical, 8)

    return {
        'cpu_logical':    cpu_logical,
        'cpu_physical':   cpu_physical,
        'mem_avail_gb':   mem_avail_gb,
        'encode_threads': encode_threads,
        'slices':         slices,
        'ffmpeg_threads': ffmpeg_threads,
    }


# ─────────────────────────────────────────────────────────────────────────────
# [FIX-NVENC-PIPE] NVENC pipe 模式参数常量
# ─────────────────────────────────────────────────────────────────────────────

# NVENC 内部帧缓冲数（-surfaces N）：
#   NVENC 硬件编码器内部维护一个帧槽池（surfaces），每个 slot 存储一帧正在被硬件
#   编码的图像。默认值为 8，对于均匀帧率的文件输入已经足够；但 pipe 输入存在速率
#   抖动（T3-Writer _write_loop 批量写入 + T2-Infer 批量 D2H），当短时供帧速率超过
#   硬件编码速率时，较小的 surfaces 数会导致 FFmpeg 无法向 NVENC 提交新帧（硬件满载
#   等待回收），引发编码器停顿（stall）。扩大至 32 可覆盖约 1 秒的帧缓冲（@30fps），
#   配合 T3-Writer 的 _MAX_BATCH=8 批量写入，基本消除 pipe 速率抖动的影响。
_NVENC_SURFACES_PIPE: int = 32

# NVENC VBR 模式前向帧预看窗口（-rc-lookahead N）：
#   仅在 crf>0（-rc:v vbr 模式）下启用。NVENC 默认不使用前向预看（N=0），
#   设为 16 后编码器可向前分析 16 帧的运动复杂度，进行更精准的码率分配：
#   · 场景切换前预先降低相邻帧码率，切换后爆发较高 I 帧码率
#   · 高运动区域分配更多比特，静止区域节约比特
#   典型 PSNR 改善 0.2-0.5 dB（1080p VBR）。
#   注意：lookahead 需要 N 帧前瞻缓冲（内部 FIFO），因此输出有 N 帧延迟，
#   与 -delay 0（零输出延迟）互斥，故仅在 VBR 路径启用，QP=0 路径改用 -delay 0。
# [QUALITY-UNIFY] 8 与 realesrgan 侧同名常量对齐，也与新的统一默认
# lookahead_depth=8 对齐。此常量现仅作 FFmpegWriter 未显式传入时的兜底：
# 正常调用链由 main.py 透传 processor 的 lookahead_depth（config/CLI）。
_NVENC_LOOKAHEAD_VBR: int = 8


@functools.lru_cache(maxsize=8)
def _ffmpeg_has_fps_mode(ffmpeg_bin: str = 'ffmpeg') -> bool:
    """[FIX-VFR-READ] fps_mode 是 FFmpeg 5.0 引入的 vsync 替代品。

    旧版 FFmpeg 传入 -fps_mode 会因"未知选项"直接退出，读帧器拿不到首帧，
    因此这里按版本号决定用哪个参数。
    """
    try:
        out = subprocess.run([ffmpeg_bin, '-hide_banner', '-version'],
                             capture_output=True, text=True, timeout=10,
                             **FFMPEG_SAFE_KW).stdout
        for line in (out or '').splitlines():
            line = line.strip()
            if line.startswith('ffmpeg version'):
                parts = line.split()
                token = parts[2] if len(parts) > 2 else ''
                major = ''.join(ch for ch in token.split('-')[0].split('.')[0]
                                if ch.isdigit())
                return int(major) >= 5 if major else False
    except Exception:
        pass
    return False


# ── [FIX-READER-UNBOUND] 读帧器看门狗 ──────────────────────────────────────
# 历史缺陷：`FFmpegFrameReader.read()` 是裸 `self._queue.get()`（无 timeout，
# 无存活判定）。只要 `_read_loop` 不再投递任何东西 —— 例如子 ffmpeg 被 SIGTTOU
# 停住（见 src/utils/stdin_hardening.py）、或死锁而不退出 —— 调用方就**永久静默
# 挂起**：无超时、无日志、无法从现象区分「还在解码」与「已经死了」。
# 实测代价：2026-09-14 给门禁写 H3 读帧器冒烟时把 verify_plan_implementation.py
# 挂了 10 分钟，最终只能靠人工中断。
#
# 设计约束（不得违反）：
#   · **反压语义必须保住**：T1(读帧) 比 T2(推理) 快 25~30×，`queue.get()` 的阻塞
#     *就是* 背压机制。因此超时只允许把「无界挂起」变成「有界失败」，
#     绝不能把「暂时无帧」误判成失败。
#   · 死亡判据必须保守：**只有** 线程已停 或 子进程已退出 才立刻判死；
#     「线程活着 + 子进程在跑」时再多给一个观察窗（慢 ≠ 错），两个窗口都空
#     才抛 —— 此时现象已与死锁不可区分，继续等只会无限期挂死。
#   · ESRGAN 侧 `get_frame()` 的 FRAME_TIMEOUT 哨兵契约**不动**（两种 API 语义
#     不同：那边返回哨兵 + 消费端看门狗，这边抛异常）。
_READER_TIMEOUT_ENV = 'IFRNET_READER_TIMEOUT'
# 默认 120s：远大于单帧正常间隔（毫秒级），远小于人工发现挂死的时间。
_READER_DEFAULT_TIMEOUT = 120.0


def _reader_timeout_from_env() -> float:
    """解析 IFRNET_READER_TIMEOUT。

    缺省/非法 → `_READER_DEFAULT_TIMEOUT`；显式 0 或负数 → 0 表示**关闭看门狗**
    （退回原来的无界阻塞），供现场对照与回滚使用。
    """
    raw = os.environ.get(_READER_TIMEOUT_ENV)
    if raw is None or not str(raw).strip():
        return _READER_DEFAULT_TIMEOUT
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _READER_DEFAULT_TIMEOUT
    if val <= 0:
        return 0.0
    return val


class FFmpegFrameReader:
    _SENTINEL = object()

    def __init__(
        self,
        video_path:      str,
        frame_start:     int   = 0,
        frame_end:       int   = -1,
        width:           int   = -1,
        height:          int   = -1,
        fps_override:    float = 0.0,
        prefetch:        int   = 128,
        use_hwaccel:     bool  = True,
        ffmpeg_bin:      str   = 'ffmpeg',
        pad_stride:      int   = 0,
    ):
        meta = _probe_video(video_path)
        self.width     = meta['width']  if width  < 0 else width
        self.height    = meta['height'] if height < 0 else height
        self.fps       = fps_override  if fps_override > 0 else meta['fps']
        self.nb_frames = meta['nb_frames']
        self.has_audio = meta['has_audio']

        # ── [FIX-RANGE-VALIDATE] 区间参数校验 + 昂贵路径显式告警 ──────────────
        # 非零 frame_start 走的 `select='between(n,start,end)'` 是**帧号精确**的，
        # 代价是必须从第 0 帧解码到 end 再丢弃前面部分（整文件解码）。
        # 为什么不换成输入侧 `-ss` 快速 seek（省掉这段解码）：
        #   本仓库已有实测反证 —— tests/verify_segment_bitstream_v5.py 的
        #   `[FIX-BOUNDARY-LEAK]` 记录输入 `-ss` 会在分块边界泄漏/顶替帧，
        #   且「泄漏帧在输出中的位置不稳定（实测有时是第 N+1 帧、有时顶替第 N 帧）」，
        #   分块路径是靠 Python 侧按显示 pts 窗口精确过滤才得以正确。
        #   而本读帧器输出的是 rawvideo（无逐帧 pts 可供过滤），一旦 seek 边界
        #   偏一帧就会被静默当成真实帧喂给插帧/超分，属最难归因的一类缺陷。
        #   故此处**保留帧号精确的慢路径**，把代价显式化，不做不精确的加速。
        # 生产侧（_process_segment）恒传 frame_start=0, frame_end=-1，不触发。
        if frame_start < 0:
            raise ValueError(f'frame_start 不可为负: {frame_start}')
        if frame_end >= 0 and frame_end < frame_start:
            raise ValueError(
                f'帧区间非法: frame_start={frame_start} > frame_end={frame_end}')
        if frame_start > 0 and self.nb_frames > 0 and frame_start >= self.nb_frames:
            raise ValueError(
                f'frame_start={frame_start} 超出可解码帧数 {self.nb_frames}')

        actual_end = frame_end if frame_end >= 0 else self.nb_frames - 1
        if frame_start > 0:
            global _RANGE_SLOWPATH_WARNED
            if not _RANGE_SLOWPATH_WARNED:
                _RANGE_SLOWPATH_WARNED = True
                print(f'[读帧器] ⚠️  请求帧区间 [{frame_start}, {actual_end}] '
                      f'(non-zero frame_start)：为保证帧号精确，将从头解码整段再丢弃前 '
                      f'{frame_start} 帧（整文件解码，代价 ≈ 全片）。'
                      f'如只需整段可传 frame_start=0, frame_end=-1；'
                      f'如需高效切片请改用关键帧对齐分片（见 verify_segment_bitstream '
                      f'的 _build_chunk_plan + pts 窗口过滤）。', flush=True)
        self._segment_frames = actual_end - frame_start + 1
        self._frame_bytes    = self.width * self.height * 3

        self._pad_stride = pad_stride
        if pad_stride > 0:
            def _ceil(x, s): return x if x % s == 0 else x + (s - x % s)
            ph = _ceil(self.height, pad_stride) - self.height
            pw = _ceil(self.width,  pad_stride) - self.width
        else:
            ph = pw = 0
        self._pad_h  = ph
        self._pad_w  = pw
        self.need_pad = ph > 0 or pw > 0

        # [P2-2] 由 use_hwaccel 的静态请求改为「自适应决策」：调用方传 True
        # 仅表示「允许硬解」，最终用不用由码率/核数决定（实测标定见
        # src/utils/reader_hwaccel.py 的模块 docstring）。
        # use_hwaccel=False 仍为强制软解。
        self.use_hwaccel = bool(
            use_hwaccel and decide_reader_hwaccel(
                self.width, self.height, self.fps, meta.get('bit_rate', 0),
                nvdec_available=HardwareCapability.has_nvdec(),
                label='(%s)' % os.path.basename(str(video_path))))

        hw_args: List[str] = []
        if self.use_hwaccel:
            # nv12 是 NVDEC 合法的 hwaccel_output_format：
            # FFmpeg 会先将 CUDA NV12 surface download 到 CPU，
            # 再由 swscale 自动转换为 -pix_fmt rgb24 输出到管道
            hw_args = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'nv12']

        if frame_start == 0 and frame_end < 0:
            vf_args: List[str] = ['-vf', 'scale=in_color_matrix=bt601:in_range=tv']
        else:
            vf_args = [
                '-vf',
                f"select='between(n\\,{frame_start}\\,{actual_end})',setpts=N/FR/TB,scale=in_color_matrix=bt601:in_range=tv",
            ]

        # [FIX-VFR-READ] 必须显式 passthrough（输出侧选项，须在 -f rawvideo 之前）：
        # rawvideo 输出默认走 CFR，遇到 VFR 源（时间戳空洞，例如源视频尾部丢帧留下的
        # Δ=2 帧间隔）会复制帧补洞，使读入帧数 > 真实解码帧数。段级验收用
        # ffprobe -count_frames（真实解码帧数）推算期望值，二者口径不一致就会误判
        # decoded_frame_mismatch（表现为"最后一个分段总是失败"）。
        # 与 Real-ESRGAN 读帧器的 fps_mode=passthrough 保持一致。
        if _ffmpeg_has_fps_mode(ffmpeg_bin):
            sync_args = ['-fps_mode', 'passthrough']
        else:
            sync_args = ['-vsync', '0']          # ffmpeg < 5.0 回退

        # [FIX-NVDEC-THREAD-CAP] 解码线程钳位到 ≤8，避免多核机上
        # NVDEC 解码 surface 超过驱动 32 上限（-threads 9 → 33 surfaces 被拒）。
        # [P2-2] 该上限只对 NVDEC 成立：8 是驱动 surface 预算的映射，不是
        # CPU 软解的核数上限。软解路径按物理核数给线程，避免在 8 核以上的
        # 机器上把软解人为限制在 8 线程、白白浪费算力。
        if self.use_hwaccel:
            _dec_threads = _clamp_decode_threads()
        else:
            _dec_threads = max(1, _reader_cpu_cores())
        cmd = (
            [ffmpeg_bin]
            # [META-KEEP] 输入侧选项，须在 -i 之前：禁止 ffmpeg 把 display matrix
            # 烘焙成像素旋转。烘焙后 rawvideo 帧宽高被交换，而流水线按 ffprobe 的
            # 存储坐标系尺寸 reshape，旋转视频会静默错乱；不烘焙则由合并阶段统一
            # 把旋转标签写回产物。
            + ['-noautorotate']
            + hw_args
            + ['-threads', str(_dec_threads)]
            + ['-i', video_path]
            + vf_args
            + sync_args
            + ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-loglevel', 'error', 'pipe:1']
        )
        self._proc   = subprocess.Popen(
            # [FIX-STDIN-TTOU] 显式给子进程 stdin=/dev/null（经 FFMPEG_SAFE_KW，
            # 与 Real-ESRGAN 侧读帧器共用同一份定义）：
            # 若父进程 stdin 是非前台 tty，ffmpeg 启动时会对 fd0 调
            # ioctl(TCSETS) 触发 SIGTTOU 被停住（"秒卡、0% CPU、无输出"，
            # 极易误判为码流/GPU 故障）。详见 src/utils/stdin_hardening.py。
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            **FFMPEG_SAFE_KW
        )
        self._queue  = queue.Queue(maxsize=max(prefetch, 4))
        # [FIX-READER-UNBOUND] 看门狗参数与遥测。`_frames_read` 让超时异常能回答
        # 「卡在第几帧、队列积压多少」，否则现场只剩一句"没反应"。
        self._read_timeout = _reader_timeout_from_env()
        self._frames_read  = 0
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self):
        """Consume stderr to prevent pipe buffer deadlock."""
        try:
            while True:
                chunk = self._proc.stderr.read(8192)
                if not chunk:
                    break
        except Exception:
            pass

    def _read_loop(self):
        pad_h, pad_w = self._pad_h, self._pad_w
        do_pad = self.need_pad
        fb = self._frame_bytes
        try:
            while True:
                # Read exactly fb bytes (robust against partial pipe reads)
                buf = bytearray()
                while len(buf) < fb:
                    chunk = self._proc.stdout.read(fb - len(buf))
                    if not chunk:
                        break
                    buf.extend(chunk)
                if len(buf) < fb:
                    break
                arr = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(
                    self.height, self.width, 3)
                if do_pad:
                    padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
                    self._queue.put((arr, padded))
                else:
                    self._queue.put((arr, arr))
                self._frames_read += 1
        except Exception as e:
            self._queue.put(e)
            return
        self._queue.put(self._SENTINEL)

    # ── [FIX-READER-UNBOUND] 存活判定 + 有界 read() ────────────────────────
    def _producer_state(self) -> Tuple[bool, str]:
        """返回 (是否已死, 依据字符串)。

        判据刻意保守：只有「读线程已停」或「子 ffmpeg 已退出但没送哨兵」才算死。
        「线程活着 + 子进程在跑」一律算活 —— 那可能只是 T1 慢于 T2 的反压，
        误判会制造假失败。
        """
        if not self._thread.is_alive():
            return True, 'reader_thread_dead'
        rc = self._proc.poll()
        if rc is not None:
            return True, 'child_exited(rc=%s,no_sentinel)' % rc
        return False, 'thread_alive,child_running'

    def _stall_message(self, waited: float, state: str) -> str:
        return ('[FIX-READER-UNBOUND] 读帧器无产出（已等 %.1fs，阈值 %.0fs）: %s, '
                'thread_alive=%s, child_poll=%s, queue=%d/%d, frame=%d '
                '(设 %s=0 可退回无界阻塞)'
                % (waited, self._read_timeout, state, self._thread.is_alive(),
                   self._proc.poll(), self._queue.qsize(), self._queue.maxsize,
                   self._frames_read, _READER_TIMEOUT_ENV))

    def read(self, timeout: Optional[float] = None
             ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """取下一位帧；返回 `None` 表示正常读到 EOF。

        [FIX-READER-UNBOUND] `timeout` 语义（默认取 IFRNET_READER_TIMEOUT）：
          · `timeout <= 0` → **关闭看门狗**，退回原来的无界阻塞（现场对照/回滚用）；
          · `timeout  > 0` → 有界等待：先等 `timeout`，队列仍空时查存活；
            已死 → 立即抛 `RuntimeError`（含 thread/child/queue/frame 现场）；
            仍活 → 再给一个 `timeout` 观察窗（慢 ≠ 错，保住反压），
            第二个窗口也空才抛 —— 此时现象与死锁已不可区分。

        异常路径抛 `RuntimeError`（不是返回哨兵）：本读帧器的消费方是
        `main.py` 的 `while True: pair = reader.read()`，返回 None 会被当成
        "正常 EOF" 而静默少帧，故必须抛。ESRGAN 侧 `get_frame()` 的
        FRAME_TIMEOUT 哨兵契约与此不同，**不要**互相照搬。
        """
        t = self._read_timeout if timeout is None else float(timeout)
        if t <= 0:
            item = self._queue.get()
        else:
            try:
                item = self._queue.get(timeout=t)
            except queue.Empty:
                dead, state = self._producer_state()
                if not dead:
                    # 生产者仍在推进 → 反压中，不是错误；再给一个观察窗。
                    try:
                        item = self._queue.get(timeout=t)
                    except queue.Empty:
                        dead, state = self._producer_state()
                        if not dead:
                            state = ('producer_alive_but_silent_2xT'
                                     '(thread_alive,child_running)')
                        raise RuntimeError(self._stall_message(2.0 * t, state))
                else:
                    raise RuntimeError(self._stall_message(t, state))
        if item is self._SENTINEL:
            return None
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        try:
            self._proc.terminate()   # ✅ 先终止进程，防止 stdout.read() 因进程卡死而永久挂起
        except Exception:
            pass
        try:
            self._proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()


def _probe_video(video_path: str) -> dict:
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration,bit_rate',
        '-show_entries', 'format=nb_streams,bit_rate',
        '-of', 'json', video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f'ffprobe 失败: {result.stderr}')
    import json as _json
    data = _json.loads(result.stdout)
    vs   = data['streams'][0]
    fps_str = vs.get('r_frame_rate', '24/1')
    try:
        fps = float(Fraction(fps_str))
    except (ValueError, ZeroDivisionError):
        fps = 24.0
    nb = 0
    if 'duration' in vs:
        dur = float(vs['duration'])
        if dur > 0:
            nb = int(dur * fps)
            # 交叉验证：若 nb_frames 与 duration×fps 偏差 > 5%，警告
            if 'nb_frames' in vs and vs['nb_frames'] not in ('N/A', ''):
                nb_meta = int(vs['nb_frames'])
                if nb_meta > 0 and nb > 0 and abs(nb_meta - nb) / max(nb_meta, nb) > 0.05:
                    print(f'⚠️  ffprobe元数据 nb_frames={nb_meta} 与 duration×fps={nb} 不一致，'
                          f'使用后者（分段文件 -c copy 常见）', flush=True)
    elif 'nb_frames' in vs and vs['nb_frames'] not in ('N/A', ''):
        nb = int(vs['nb_frames'])
    cmd_audio = [
        'ffprobe', '-v', 'error', '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_type', '-of', 'json', video_path,
    ]
    a = subprocess.run(cmd_audio, capture_output=True, text=True, timeout=15)
    has_audio = (a.returncode == 0 and '"codec_type": "audio"' in a.stdout)

    # [P2-2] 码率用于读帧器 hwaccel 自适应决策（见 decide_reader_hwaccel）。
    # 视频流 bit_rate 优先，缺失时退到容器 overall bit_rate（含音频，作上界）。
    bit_rate = 0
    for _src in (vs.get('bit_rate'), data.get('format', {}).get('bit_rate')):
        try:
            _v = int(_src)
        except (TypeError, ValueError):
            continue
        if _v > 0:
            bit_rate = _v
            break

    return {
        'width': int(vs['width']), 'height': int(vs['height']),
        'fps': fps, 'nb_frames': nb, 'has_audio': has_audio,
        'bit_rate': bit_rate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# M3: FFmpeg Writer
# ─────────────────────────────────────────────────────────────────────────────

class FFmpegWriter:
    _SENTINEL  = object()
    _MAX_BATCH = 8
    _STDERR_IGNORE = (
        'x265 [info]:', 'x265 [warning]:', 'set_mempolicy:',
        'encoded ', 'Weighted P-Frames', 'consecutive B-frames',
        'frame I:', 'frame P:', 'frame B:',
        # [FIX-SLICE-THREAD] x264 slice-threading 信息行
        'using cpu capabilities:', 'slice threads:', 'frame threads:',
        'x264 [info]:', 'x264 [warning]:',
        # [FIX-NVENC-PIPE] NVENC 初始化 / 会话诊断信息行（非错误，无需打印）
        # · 'Initialized NPP'  : CUDA NPP 库初始化（h264_nvenc/hevc_nvenc 启动时打印）
        # · 'NVENC session'    : NVENC 编码会话创建日志
        # · 'GPU #'            : NVENC 选择 GPU 设备信息行
        'Initialized NPP', 'NVENC session', 'GPU #',
    )

    def __init__(
        self,
        output_path: str,
        width:  int,
        height: int,
        fps:    float,
        codec:  str = 'libx264',
        extra_codec_args: Optional[List[str]] = None,
        crf:    int  = 23,
        preset: str  = None,
        audio_src: Optional[str] = None,
        ffmpeg_bin: str = 'ffmpeg',
        quiet: bool = True,
        n_threads: Optional[int] = None,   # [FIX-SLICE-THREAD] None=自动探测
        # NVENC rate control mode for Level 2 fallback。默认与 realesrgan 侧同名
        # 参数一致；正常调用链由 main.py 透传生效的 rate_mode。
        rc_mode: str = "vbr_hq",
        lookahead_depth: Optional[int] = None,  # None → _NVENC_LOOKAHEAD_VBR（仅 VBR 生效）
    ):
        # [FIX-T3-V643] 去除内部 _queue 和 _write_loop 线程，直接管道写入
        self._error: Optional[Exception] = None
        self._write_count = 0

        if preset is None:
            preset = 'p4' if 'nvenc' in codec else 'medium'
        elif 'nvenc' in codec and preset in _PRESET_P_INDEX:
            # [FIX-PRESET-UNIFY] x264 名称（ultrafast 等）→ p1~p7 体系，统一 _PRESET_P_INDEX 口径，
            # 与 SDK Level 1 直通路径完全一致，避免 "Unable to parse option value" 错误。
            preset = f"p{_PRESET_P_INDEX[preset] + 1}"

        # [FIX-SLICE-THREAD] 自动探测 CPU / 内存，计算最优软编码并行参数
        _par = _detect_encode_parallelism(n_threads)
        _et  = _par['encode_threads']   # 编码线程数
        _s   = _par['slices']           # x264 分片数
        _ft  = _par['ffmpeg_threads']   # FFmpeg 全局线程数
        _mem = _par['mem_avail_gb']

        # [FIX-LOSSLESS] crf=0 → 按编解码器映射为正确的无损参数。
        # 背景：
        #   · libx264 : crf=0 恰好等于无损，但显式用 -qp 0 语义更清晰
        #   · libx265 : crf=0 ≠ 无损！仅为极高质量有损；无损需 -x265-params lossless=1
        #   · nvenc   : -cq:v 0 是 VBR 模式下的极低码率控制，不是无损；
        #               无损需去掉 -rc:v vbr，改用 -qp 0 -b:v 0（常量 QP 模式）
        # [FIX-SLICE-THREAD] x265 frame-threads：默认 min(4, cpu_logical//2)
        # x265 frame-threads 含义：同时编码的帧数（帧级并行），通常 2-4 最佳；
        # 过高会引入帧延迟，与 pipe 流式输入场景不符。
        _x265_ft = max(2, min(4, _par['cpu_logical'] // 2))
        # x265 pool：线程池总大小（所有 frame-threads 共享），= encode_threads
        _x265_pool = _et
        # CONSTQP 分支算出的 QP（None = 非 CONSTQP 模式）；在参数摘要中复用
        _nvenc_qp: Optional[int] = None

        if crf == 0:
            if 'nvenc' in codec:
                # [FIX-LOSSLESS] NVENC 无损：常量 QP=0，去掉 vbr 码率控制。
                # [FIX-NVENC-PIPE] 同时注入 pipe 场景三项优化参数：
                #   · -bf 0        禁用 B 帧：B 帧编码需要前后参考帧，编码器须缓存后续帧
                #                  才能输出，引入多帧流水线延迟；禁用后每帧独立编码即输出。
                #   · -surfaces N  扩大 NVENC 内部帧缓冲槽数（默认 8 → _NVENC_SURFACES_PIPE），
                #                  防止 pipe 写入速率抖动时编码器因 surface 耗尽而暂停。
                #   · -delay 0     零输出延迟：配合 -bf 0 强制 NVENC 在每帧编码完成后
                #                  立即写入输出流，不等待后续帧，最小化 pipe 端到端延迟。
                #                  注意：-delay 0 与 -rc-lookahead 互斥（lookahead 需要前瞻
                #                  缓冲），此处无损模式无需质量优化型预看，故可安全启用。
                quality_args = [
                    '-preset', preset,
                    '-rc', 'constqp',
                    '-qp', '0', '-b:v', '0',
                    '-bf', '0',
                    '-surfaces', str(_NVENC_SURFACES_PIPE),
                    '-delay', '0',
                ]
            elif codec == 'libx265':
                # x265 无损：lossless=1 + 多线程（替换旧 pools=none）
                # [FIX-SLICE-THREAD] pools={N} 启用线程池，frame-threads={F} 帧级并行
                quality_args = [
                    '-preset', preset,
                    '-x265-params',
                    f'lossless=1:pools={_x265_pool}:frame-threads={_x265_ft}',
                ]
            elif codec == 'libx264':
                # x264 无损：-qp 0 + slice-based threading
                # [FIX-SLICE-THREAD] threads=N 设置 x264 编码线程数；
                # slices=S 将单帧切为 S 片并行编码（intra-frame 并行），
                # 与 pipe 流式场景匹配（每帧编完即输出，无帧间延迟）。
                quality_args = [
                    '-preset', preset, '-qp', '0',
                    '-x264-params', f'threads={_et}:slices={_s}',
                ]
            else:
                # 其他编解码器（如 ffv1、utvideo 等）：回退到 -qp 0
                quality_args = ['-qp', '0']
        elif 'nvenc' in codec:
            # [FIX-NVENC-PIPE] NVENC VBR（cq）模式 pipe 场景优化：
            #   · -bf 0              禁用 B 帧，同 crf=0 路径，降低流水线缓冲延迟。
            #   · -rc-lookahead N    前向帧预看（N = _NVENC_LOOKAHEAD_VBR）：
            #                        VBR 模式下编码器向前分析 N 帧运动复杂度，优化帧间
            #                        码率分配，改善场景切换质量（PSNR +0.2-0.5 dB）。
            #                        与 -delay 0 互斥（预看需要 N 帧前瞻缓冲区），
            #                        因此 VBR 路径不设 -delay 0。
            #   · -surfaces N        扩大 NVENC 内部帧缓冲（同 crf=0 路径）。
            # [FIX-FFMPEGWRITER-RC] 根据 Level 1 的 RC 模式选择对应的 FFmpeg -rc:v 值
            # 注意：qvbr 在旧版 FFmpeg h264_nvenc 中不可用，回退到 vbr_hq
            _rc_v_map = {'vbr_hq': 'vbr_hq', 'qvbr': 'vbr_hq', 'constqp': 'constqp'}
            _rc_v = _rc_v_map.get(rc_mode, 'vbr_hq')
            if _rc_v == 'constqp':
                # [QUALITY-UNIFY] CONSTQP 专用参数是 -qp，不是 -cq:v：
                #   ffmpeg: -cq "…for constant quality mode in VBR rate control"（仅 VBR 有效）
                #           -qp "Constant quantization parameter rate control method"
                # 旧实现把 constqp 映射成 'vbr' + -cq:v，用户拿到的其实是普通 VBR+CQ，
                # 与 Level 1 SDK 直通路径（真 CONSTQP）不一致。此处对齐 Level 1，并按
                # constqp 轴把已算好的 CQ 值换算成 QP。CONSTQP 下 LA 被硬件静默禁用，
                # 故不发 -rc-lookahead；-b:v 0 对 CONSTQP 无意义，一并省略。
                _nvenc_qp = to_constqp_qp(codec, crf)
                quality_args = [
                    '-preset', preset,
                    '-rc:v', 'constqp', '-qp', str(_nvenc_qp),
                    '-bf', '0',
                    '-surfaces', str(_NVENC_SURFACES_PIPE),
                ]
            else:
                _la = _NVENC_LOOKAHEAD_VBR if lookahead_depth is None else int(lookahead_depth)
                quality_args = [
                    '-preset', preset,
                    '-rc:v', _rc_v, '-cq:v', str(crf), '-b:v', '0',
                    '-bf', '0',
                    '-rc-lookahead', str(_la),
                    '-surfaces', str(_NVENC_SURFACES_PIPE),
                ]
        elif codec == 'libx265':
            # [FIX-SLICE-THREAD] 替换旧 pools=none（完全禁用线程池）为正确多线程参数
            quality_args = [
                '-preset', preset, '-crf', str(crf),
                '-x265-params',
                f'pools={_x265_pool}:frame-threads={_x265_ft}',
            ]
        else:
            # libx264（及其他 x264 系列）：追加 slice-based threading 参数
            # [FIX-SLICE-THREAD] threads=N + slices=S：N 线程各负责 S/N 片，
            # 当 slices >= threads 时 x264 自动切换为 slice-based 模式。
            quality_args = [
                '-preset', preset, '-crf', str(crf),
                '-x264-params', f'threads={_et}:slices={_s}',
            ]

        cmd = [
            ffmpeg_bin, '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}',
            '-r', f'{fps:.6f}',
            '-i', 'pipe:0',
        ]
        # [FIX-SLICE-THREAD] FFmpeg 全局 -threads：仅软编码路径需要，
        # NVENC 是 GPU 固定功能硬件单元，不受 CPU 线程数控制。
        if 'nvenc' not in codec:
            cmd.insert(2, '-threads')
            cmd.insert(3, str(_ft))
        if audio_src:
            cmd += ['-i', audio_src, '-c:a', 'copy', '-map', '0:v', '-map', '1:a?']
        if extra_codec_args:
            # 合并 extra_args 与 quality_args，保留后者中不在前者内的关键 pipe 参数
            _extra_flags = set()
            for i in range(0, len(extra_codec_args), 2):
                _extra_flags.add(extra_codec_args[i])
            _merged = list(extra_codec_args)
            for i in range(0, len(quality_args), 2):
                if quality_args[i] not in _extra_flags:
                    _merged.extend(quality_args[i:i+2])
            cmd += ['-vcodec', codec] + _merged
        else:
            cmd += ['-vcodec', codec] + quality_args
        cmd += ['-pix_fmt', 'yuv420p', '-loglevel', 'error', output_path]

        # [FIX-SLICE-THREAD / FIX-NVENC-PIPE] 打印编码参数摘要
        # NVENC 路径：打印 GPU 硬件编码关键参数（无 CPU 线程参数，因 NVENC 不受其控制）
        # 软编码路径：打印 CPU 线程 / slice 并行配置（与旧行为一致）
        if 'nvenc' in codec:
            # [FIX-NVENC-PIPE] NVENC 参数摘要：显示 pipe 场景优化参数的实际生效值，
            # 便于用户确认 surfaces / lookahead / delay 参数是否符合预期。
            # 注：ffmpeg_threads 对 NVENC 本身无效，仅作用于 demux/filter graph，
            #     此处显示以完整呈现 FFmpeg 命令的全局线程配置。
            if crf == 0:
                _nvenc_info = (
                    f'[FIX-NVENC-PIPE] NVENC 无损(QP=0): '
                    f'preset={preset}  bf=0  '
                    f'surfaces={_NVENC_SURFACES_PIPE}  delay=0  '
                    f'ffmpeg_threads={_ft}(全局demux，不影响NVENC硬件单元)'
                )
            elif _nvenc_qp is not None:
                _nvenc_info = (
                    f'[FIX-NVENC-PIPE] NVENC CONSTQP(-qp {_nvenc_qp}): '
                    f'preset={preset}  bf=0  la=off(constqp 硬件禁用)  '
                    f'surfaces={_NVENC_SURFACES_PIPE}  '
                    f'ffmpeg_threads={_ft}(全局demux，不影响NVENC硬件单元)'
                )
            else:
                _nvenc_info = (
                    f'[FIX-NVENC-PIPE] NVENC VBR(cq={crf}): '
                    f'preset={preset}  bf=0  '
                    f'rc-lookahead={_NVENC_LOOKAHEAD_VBR if lookahead_depth is None else int(lookahead_depth)}  '
                    f'surfaces={_NVENC_SURFACES_PIPE}  '
                    f'ffmpeg_threads={_ft}(全局demux，不影响NVENC硬件单元)'
                )
            print(f'   {_nvenc_info}', flush=True)
        else:
            # [FIX-SLICE-THREAD] 软编码路径：打印 CPU 线程 / slice 并行配置摘要
            _codec_l = codec.lower()
            if 'x264' in _codec_l or codec not in ('libx265',):
                _thread_info = (
                    f'[FIX-SLICE-THREAD] 软编码并行: '
                    f'cpu={_par["cpu_logical"]}逻辑/{_par["cpu_physical"]}物理  '
                    f'mem_avail={_par["mem_avail_gb"]:.1f}GiB  '
                    f'encode_threads={_et}  slices={_s}  ffmpeg_threads={_ft}'
                )
            else:
                _thread_info = (
                    f'[FIX-SLICE-THREAD] 软编码并行: '
                    f'cpu={_par["cpu_logical"]}逻辑/{_par["cpu_physical"]}物理  '
                    f'mem_avail={_par["mem_avail_gb"]:.1f}GiB  '
                    f'encode_threads={_et}(frame-threads={_x265_ft})  ffmpeg_threads={_ft}'
                )
            print(f'   {_thread_info}', flush=True)

        # 打印完整 FFmpeg 命令，便于调试和确认编码参数（quiet=True 时跳过）
        print(f'   [FFmpegWriter] 命令: {" ".join(cmd)}', flush=True)
        # if not quiet:
        #     print(f'   [FFmpegWriter] 命令: {" ".join(cmd)}', flush=True)

        self._proc   = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        self._stderr_lines: List[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        # [FIX-T3-V643] 不再启动内部 _write_loop 线程，改为 write_direct() 直接写管道

    def _drain_stderr(self):
        try:
            for line in self._proc.stderr:
                decoded = line.decode(errors='ignore').rstrip()
                self._stderr_lines.append(decoded)
                if decoded and not any(decoded.lstrip().startswith(p)
                                       for p in self._STDERR_IGNORE):
                    print(f'[FFmpeg ERR] {decoded}')
        except Exception:
            pass

    def write_direct(self, data):
        """[FIX-T3-V643] 直接写 bytes/memoryview 到 FFmpeg stdin pipe，零中间拷贝。"""
        if self._error is not None:
            raise RuntimeError(f'FFmpegWriter 内部错误: {self._error}') from self._error
        try:
            self._proc.stdin.write(data)
            self._write_count += 1
        except BrokenPipeError:
            self._error = RuntimeError('FFmpeg stdin 管道已断开')
            raise self._error

    def write(self, frame):
        """[FIX-T3-V643] 兼容旧接口：numpy array → tobytes() → write_direct()。"""
        self.write_direct(frame.tobytes())

    def close(self):
        self._stderr_thread.join(timeout=5)
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        _killed = False
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
            _killed = True
        rc = self._proc.returncode
        if rc is not None and rc != 0:
            stderr_out = '\n'.join(self._stderr_lines[-20:])
            print(f'\n[Warning] FFmpeg 退出码={rc}, stderr: {stderr_out[:400]}')
        if self._error:
            print(f'[Warning] FFmpegWriter 累计写帧异常: {self._error}')
        # [P0-FIX-CLOSE-RC] 收尾失败（rc!=0 / 强杀）= 输出文件损坏（无 moov/截断）。
        # 正常收尾路径上抛使段判败；若当前已在异常清理中或写线程早已报错，
        # 则保留原语义仅打印，避免掩盖原始异常。
        if ((rc is not None and rc != 0) or _killed) and not self._error:
            import sys as _sys
            if _sys.exc_info()[0] is None:
                raise RuntimeError(
                    f'FFmpegWriter exited abnormally (rc={rc}, killed={_killed}); '
                    f'output file likely corrupted/truncated')
