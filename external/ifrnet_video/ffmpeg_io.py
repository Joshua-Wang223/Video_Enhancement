#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# IFRNet Video Enhancement - FFmpeg 读写模块（解码器 / 编码回退 / 硬件探测）。
# 镜像 external/realesrgan_video/ffmpeg_io.py 的职责。

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from ifrnet_utils import _clamp_decode_threads


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
            enc = subprocess.run(enc_cmd, capture_output=True, timeout=10)
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
            result = subprocess.run(cmd, capture_output=True, timeout=10)
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
                if subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0:
                    return 'h264_nvenc', ['-rc', 'constqp', '-qp', '0']
            except Exception:
                pass
        return 'libx264', ['-qp', '0', '-preset', 'medium']

_X264_PRESET_FACTOR = {
    'ultrafast': 8.0, 'superfast': 6.0, 'veryfast': 4.0,
    'faster': 2.5, 'fast': 2.0, 'medium': 1.0,
    'slow': 0.4, 'slower': 0.2, 'veryslow': 0.1,
}
# [FIX-NVENC-PRESET] x264 → NVENC preset 名称映射。
# NVENC 使用 p1(最快)~p7(最慢) 命名体系，与 x264 的 ultrafast~veryslow 不兼容。
# 当用户通过 --x264-preset 传入 x264 风格名称 + --codec h264_nvenc 时自动转换。
_X264_TO_NVENC_PRESET = {
    'ultrafast': 'p1', 'superfast': 'p1', 'veryfast': 'p2',
    'faster': 'p3', 'fast': 'p3', 'medium': 'p4',
    'slow': 'p5', 'slower': 'p6', 'veryslow': 'p7',
}
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
_NVENC_LOOKAHEAD_VBR: int = 16


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

        actual_end = frame_end if frame_end >= 0 else self.nb_frames - 1
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

        hw_args: List[str] = []
        if use_hwaccel and HardwareCapability.has_nvdec():
            # hw_args = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'nv12']
            # ✅ 正确：nv12 是 NVDEC 合法的 hwaccel_output_format
            # FFmpeg 会先将 CUDA NV12 surface download 到 CPU，
            # 再由 swscale 自动转换为 -pix_fmt rgb24 输出到管道
            hw_args = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'nv12']

        if frame_start == 0 and frame_end < 0:
            vf_args: List[str] = []
        else:
            vf_args = [
                '-vf',
                f"select='between(n\\,{frame_start}\\,{actual_end})',setpts=N/FR/TB",
                '-vsync', '0',
            ]

        # [FIX-NVDEC-THREAD-CAP] 解码线程钳位到 ≤8，避免多核机上
        # NVDEC 解码 surface 超过驱动 32 上限（-threads 9 → 33 surfaces 被拒）。
        cmd = (
            [ffmpeg_bin]
            + hw_args
            + ['-threads', str(_clamp_decode_threads())]
            + ['-i', video_path]
            + vf_args
            + ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-loglevel', 'error', 'pipe:1']
        )
        self._proc   = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._queue  = queue.Queue(maxsize=max(prefetch, 4))
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
        except Exception as e:
            self._queue.put(e)
            return
        self._queue.put(self._SENTINEL)

    def read(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        item = self._queue.get()
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
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
        '-show_entries', 'format=nb_streams',
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
    return {
        'width': int(vs['width']), 'height': int(vs['height']),
        'fps': fps, 'nb_frames': nb, 'has_audio': has_audio,
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
        rc_mode: str = "constqp",           # NVENC rate control mode for Level 2 fallback
    ):
        # [FIX-T3-V643] 去除内部 _queue 和 _write_loop 线程，直接管道写入
        self._error: Optional[Exception] = None
        self._write_count = 0

        if preset is None:
            preset = 'p4' if 'nvenc' in codec else 'medium'
        elif 'nvenc' in codec and preset in _X264_TO_NVENC_PRESET:
            # [FIX-NVENC-PRESET] --x264-preset 传入 ultrafast 等 x264 名称，
            # 但 nvenc 只认 p1~p7 体系，需自动映射避免 "Unable to parse option value" 错误
            preset = _X264_TO_NVENC_PRESET[preset]

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
            _rc_v_map = {'vbr_hq': 'vbr_hq', 'qvbr': 'vbr_hq', 'constqp': 'vbr'}
            _rc_v = _rc_v_map.get(rc_mode, 'vbr')
            quality_args = [
                '-preset', preset,
                '-rc:v', _rc_v, '-cq:v', str(crf), '-b:v', '0',
                '-bf', '0',
                '-rc-lookahead', str(_NVENC_LOOKAHEAD_VBR),
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
            else:
                _nvenc_info = (
                    f'[FIX-NVENC-PIPE] NVENC VBR(cq={crf}): '
                    f'preset={preset}  bf=0  '
                    f'rc-lookahead={_NVENC_LOOKAHEAD_VBR}  '
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
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        rc = self._proc.returncode
        if rc is not None and rc != 0:
            stderr_out = '\n'.join(self._stderr_lines[-20:])
            print(f'\n[Warning] FFmpeg 退出码={rc}, stderr: {stderr_out[:400]}')
        if self._error:
            print(f'[Warning] FFmpegWriter 累计写帧异常: {self._error}')
