"""
IFRNet 视频插帧处理脚本 —— 终极优化版 v6（单卡版）
==========================================================
基于 IFRNet（Intermediate Flow-based Recursive Network）的视频帧插值脚本，
面向单 GPU 生产环境的高性能实现。

【最终功能特性】
  推理加速：
    · FP16 半精度推理（use_fp16，默认开启）
    · torch.compile 内核融合（mode=default, dynamic=True；首次编译约 1~3 分钟）
    · CUDA Graph 捕获（use_cuda_graph；与 torch.compile 互斥，compile 优先）
    · TensorRT 可选加速（use_tensorrt；导出 ONNX → 构建 FP16 静态 Engine）
    · OOM 自动降级：batch_size 减半 → 深度清理 → 按空闲显存估算恢复

  I/O 加速：
    · NVDEC 硬件解码（use_hwaccel；自动探测，失败回退 CPU）
    · NVENC 硬件编码（有 h264_nvenc/hevc_nvenc 时自动升级）
    · 异步帧预取（FFmpegFrameReader 后台线程 + PinnedBufferPool）
    · 批量写帧（FFmpegWriter 攒 8 帧一次 write，减少 pipe syscall）

  稳定性与可观测性：
    · torch.compile 小形状（32×32）预热，避免大分辨率首帧编译卡顿
    · OOM 级联保护：首次 OOM 永久降低 max_batch_size，级联 OOM 不再修改上限
    · CUDA Graph 按 shape 缓存，batch_size 动态变化时安全捕获
    · JSON 性能报告（report_json；含 infer_latency_ms p95/mean/max、nvdec、nvenc）

【命令行使用示例】
  # 基础用法（FP16 + torch.compile + NVDEC/NVENC 自动启用）
  python process_video_v5_single.py \\
      --input input.mp4 --output output_2x.mp4 --scale 2

  # 4× 插帧，关闭 compile（跳过预热，启动更快，适合短视频）
  python process_video_v5_single.py \\
      --input input.mp4 --output output_4x.mp4 --scale 4 --no_compile

  # TensorRT 加速（首次构建 Engine，后续秒启动）
  python process_video_v5_single.py \\
      --input input.mp4 --output output.mp4 --scale 2 --use_tensorrt

  # 禁用所有加速（调试/CPU 环境）
  python process_video_v5_single.py \\
      --input input.mp4 --output output.mp4 --scale 2 \\
      --no_fp16 --no_compile --no_cuda_graph --no_hwaccel --device cpu

  # 输出性能报告
  python process_video_v5_single.py \\
      --input input.mp4 --output output.mp4 --scale 2 --report report.json

【关键参数说明】
  --scale        插帧倍数，≥2 整数（2=2×, 4=4×, 8=8×）
  --model        IFRNet_S_Vimeo90K（默认轻量）或 IFRNet_L_Vimeo90K（高质量）
  --batch_size   每批帧对数，默认 4；显存充裕时自动爬升
  --no_fp16      关闭 FP16（默认开启）
  --no_compile   关闭 torch.compile（默认开启）
  --no_cuda_graph  关闭手动 CUDA Graph（compile 激活时已自动接管）
  --use_tensorrt  启用 TRT Engine（首次构建耗时较长）
  --no_hwaccel   强制 CPU 解码（NVDEC 可用时默认启用硬解）
  --crf          编码质量，默认 23（18~28 常用范围）
  --report       JSON 报告路径（可选，记录推理延迟、硬件状态）

【注意事项】
  · 首次 torch.compile 约需 1~3 分钟，编译缓存默认存于 .torch_compile_cache/
  · TRT Engine 缓存于 .trt_cache/<tag>.trt，TRT 版本升级后需手动删除重建
  · --scale 对 Fraction(scale) 取整，非整数倍数会自动取近似整数
  · 模型路径由脚本顶部 base_dir / models_ifrnet 常量决定，部署前请修改

从多卡版精简为单 GPU 版本，移除 _ifrnet_segment_worker、
_process_multi_gpu、_process_single_fallback 及所有多进程
分段调度逻辑，保留全部单卡优化不变。

【保留全部优化标记（供代码内部定位）】
  M2. FFmpeg Pipe 替换 cv2.VideoCapture（NVDEC 解码）
  M3. NVENC 硬件编码输出
  M4. TensorRT 可选加速
  X1-X6: 批量写入/ThroughputMeter/PinnedBufferPool/异常传播/error flag/Event驱动
  B1-B5: CUDA Graph/FP16/torch.compile/OOM降级/流水线双流
  N1-N13: expand代replace/pad对齐/pinned内存/推理计时/...
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import warnings
from collections import deque
from contextlib import nullcontext
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

warnings.filterwarnings('ignore')

# ── 内存碎片优化：expandable_segments 允许 PyTorch 用不连续 VRAM 满足大分配请求
# 必须在任何 CUDA 分配之前设置（setdefault 不覆盖用户已设置的值）
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

# ── 抑制 torch.inductor / dynamo 的噪音警告 ────────────────────────────────
import logging as _logging
# "Not enough SMs to use max_autotune_gemm mode"
_logging.getLogger('torch._inductor.utils').setLevel(_logging.ERROR)
# "failed while executing pow_by_natural" —— 符号形状求解器边界问题，有 fallback，无害
# 当 dynamic=True + 跨 shape 推理时，inductor 符号范围偶尔命中负指数，自动回退正确路径
_logging.getLogger('torch.utils._sympy.interp').setLevel(_logging.ERROR)
# 同类符号求解警告（其他 sympy 子模块）
_logging.getLogger('torch.utils._sympy').setLevel(_logging.ERROR)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── 路径配置 ─────────────────────────────────────────────────────────────────
# 以本脚本所在目录（external/IFRNet/）为基准，向上两级到项目根
# 目录结构假设：<project_root>/external/IFRNet/process_video_v5_single.py
_SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
base_dir       = str(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
models_ifrnet  = os.path.join(base_dir, 'models_IFRNet', 'checkpoints')
sys.path.insert(0, os.path.join(base_dir, 'external', 'IFRNet'))
sys.path.insert(0, models_ifrnet)

from models.IFRNet_S import Model  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

MODEL_STRIDE = 32

MODEL_NAME_MAP: Dict[str, str] = {
    'IFRNet_Vimeo90K':   'IFRNet_Vimeo90K.pth',
    'IFRNet_S_Vimeo90K': 'IFRNet_S_Vimeo90K.pth',
    'IFRNet_L_Vimeo90K': 'IFRNet_L_Vimeo90K.pth',
}


# ─────────────────────────────────────────────────────────────────────────────
# M2/M3: 硬件能力探测
# ─────────────────────────────────────────────────────────────────────────────

class _NVMLFilter(_logging.Filter):
    """[FIX-NML] 过滤 PyTorch 将 NVML 断言包装为 stderr 输出的无害噪音。
    在子进程或容器环境中，NVML_SUCCESS 与 CUDACachingAllocator 相关日志
    常以 RuntimeError 或 logging.WARNING 形式出现，污染输出但不影响正确性。
    """
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return ('NVML_SUCCESS' not in msg and 'CUDACachingAllocator' not in msg)

# 在模块加载时全局安装过滤器
logging.getLogger().addFilter(_NVMLFilter())


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
        """[FIX-NDV] 两阶段探测：先编码测试源，再用 NVDEC 解码，
        避免单阶段 lavfi→NVDEC 在部分驱动/容器下误报成功。"""
        try:
            import tempfile, os
            # Stage 1: encode a tiny H.264 clip to pipe
            enc_cmd = [
                'ffmpeg', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=0.1:r=1',
                '-vcodec', 'libx264', '-frames:v', '1',
                '-f', 'mp4', '-movflags', 'frag_keyframe+empty_moov',
                '-loglevel', 'error', 'pipe:1',
            ]
            enc = subprocess.run(enc_cmd, capture_output=True, timeout=10)
            if enc.returncode != 0 or not enc.stdout:
                return False
            # Stage 2: decode with NVDEC
            dec_cmd = [
                'ffmpeg', '-hwaccel', 'cuda', '-hwaccel_output_format', 'bgr24',
                '-i', 'pipe:0', '-frames:v', '1', '-f', 'null', '-',
                '-loglevel', 'error',
            ]
            dec = subprocess.run(dec_cmd, input=enc.stdout,
                                 capture_output=True, timeout=10)
            return dec.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _probe_nvenc(codec: str) -> bool:
        try:
            cmd = [
                'ffmpeg', '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=0.1:r=1',
                '-vcodec', codec, '-frames:v', '1',
                '-f', 'null', '-', '-loglevel', 'error',
            ]
            return subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0
        except Exception:
            return False

    @classmethod
    def best_encoder(cls, preferred: str = 'libx264') -> str:
        nvenc_map = {'libx264': 'h264_nvenc', 'libx265': 'hevc_nvenc'}
        # 反向映射：若用户直接指定 nvenc 编码器但不可用，回退到软件编码器
        fallback_map = {'h264_nvenc': 'libx264', 'hevc_nvenc': 'libx265'}
        # 情况1：用户直接指定了 nvenc 编码器 → 检测可用性，不可用则回退
        if preferred in fallback_map:
            if cls.has_nvenc(preferred):
                return preferred
            fallback = fallback_map[preferred]
            print(f'  [警告] {preferred} 不可用，自动回退到 {fallback}')
            return fallback
        # 情况2：用户指定软件编码器 → 尝试升级到对应 nvenc
        candidate = nvenc_map.get(preferred, preferred)
        if candidate != preferred and cls.has_nvenc(candidate):
            return candidate
        return preferred


# ─────────────────────────────────────────────────────────────────────────────
# ThroughputMeter（v4 复用）
# ─────────────────────────────────────────────────────────────────────────────

class ThroughputMeter:
    def __init__(self, window: int = 20):
        self._times: deque = deque(maxlen=window)
        self._total = 0

    def update(self, n: int):
        self._times.append((time.perf_counter(), n))
        self._total += n

    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1][0] - self._times[0][0]
        return sum(t[1] for t in self._times) / dt if dt > 0 else 0.0

    def eta(self, total: int) -> float:
        f = self.fps()
        return (total - self._total) / f if f > 0 else float('inf')


# ─────────────────────────────────────────────────────────────────────────────
# PinnedBufferPool（v4 复用，线程本地）
# ─────────────────────────────────────────────────────────────────────────────

_thread_local = threading.local()


class PinnedBufferPool:
    def __init__(self):
        self._buf: Optional[torch.Tensor] = None

    def get_for_frames(self, frames: List[np.ndarray], to_rgb: bool = True) -> torch.Tensor:
        arr = np.stack(frames, axis=0)
        if to_rgb:
            arr = arr[:, :, :, ::-1]
        arr = np.ascontiguousarray(arr)
        src   = torch.from_numpy(arr)
        src_f = src.permute(0, 3, 1, 2).float().div_(255.0).contiguous()
        n = src_f.numel()
        if self._buf is None or self._buf.numel() < n:
            self._buf = torch.empty(n, dtype=torch.float32).pin_memory()
        dst = self._buf[:n].view_as(src_f)
        dst.copy_(src_f)
        return dst


def _get_pinned_pool() -> PinnedBufferPool:
    if not hasattr(_thread_local, 'pinned_pool'):
        _thread_local.pinned_pool = PinnedBufferPool()
    return _thread_local.pinned_pool


# ─────────────────────────────────────────────────────────────────────────────
# 张量工具
# ─────────────────────────────────────────────────────────────────────────────

def pad_to_stride(arr: np.ndarray, stride: int = MODEL_STRIDE):
    H, W = arr.shape[:2]
    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride
    if pad_h == 0 and pad_w == 0:
        return arr, 0, 0
    return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'), pad_h, pad_w


def frames_to_tensor(frames, device, stream=None, dtype=torch.float32):
    pool  = _get_pinned_pool()
    cpu_t = pool.get_for_frames(frames, to_rgb=True)
    ctx   = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with ctx:
        return cpu_t.to(device, non_blocking=True, dtype=dtype)


def tensor_to_np(t, orig_H, orig_W, sync_stream=None) -> List[np.ndarray]:
    if sync_stream is not None and torch.cuda.is_available():
        torch.cuda.current_stream().wait_stream(sync_stream)
    arr = t.clamp_(0.0, 1.0).mul_(255.0).byte()
    arr = arr.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return [arr[i, :orig_H, :orig_W, ::-1].copy() for i in range(arr.shape[0])]


# ─────────────────────────────────────────────────────────────────────────────
# TensorPool（v4 复用）
# ─────────────────────────────────────────────────────────────────────────────

class TensorPool:
    def __init__(self):
        self._cache: dict = {}

    def get(self, shape, dtype, device) -> torch.Tensor:
        key = (shape, dtype, device)
        if key not in self._cache:
            self._cache[key] = torch.empty(shape, dtype=dtype, device=device)
        return self._cache[key]

    def clear(self):
        self._cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# M2: FFmpeg Pipe 帧读取器（替换 cv2.VideoCapture）
# ─────────────────────────────────────────────────────────────────────────────

class FFmpegFrameReader:
    """
    通过 FFmpeg pipe 读取指定帧范围的视频帧，支持 NVDEC 硬件解码。
    异步预取：解码与推理并行，减少 GPU 等待。
    """
    _SENTINEL = object()

    def __init__(
        self,
        video_path:      str,
        frame_start:     int = 0,
        frame_end:       int = -1,        # -1 表示到末尾
        width:           int = -1,        # -1 表示从探测获取
        height:          int = -1,
        fps_override:    float = 0.0,
        prefetch:        int = 8,
        use_hwaccel:     bool = True,
        ffmpeg_bin:      str = 'ffmpeg',
    ):
        # 探测视频元信息
        meta = _probe_video(video_path)
        self.width     = meta['width']  if width  < 0 else width
        self.height    = meta['height'] if height < 0 else height
        self.fps       = fps_override  if fps_override > 0 else meta['fps']
        self.nb_frames = meta['nb_frames']
        self.has_audio = meta['has_audio']

        actual_end = frame_end if frame_end >= 0 else self.nb_frames - 1
        self._segment_frames = actual_end - frame_start + 1
        self._frame_bytes    = self.width * self.height * 3

        # 构建 FFmpeg 命令
        hw_args: List[str] = []
        if use_hwaccel and HardwareCapability.has_nvdec():
            hw_args = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'bgr24']

        # 帧范围 select filter
        if frame_start == 0 and frame_end < 0:
            vf_args: List[str] = []
        else:
            # between(n, start, end) 选取 [start, end] 闭区间帧（0-indexed）
            vf_args = [
                '-vf',
                f"select='between(n\\,{frame_start}\\,{actual_end})',setpts=N/FR/TB",
                '-vsync', '0',
            ]

        cmd = (
            [ffmpeg_bin]
            + hw_args
            + ['-i', video_path]
            + vf_args
            + ['-f', 'rawvideo', '-pix_fmt', 'bgr24', '-loglevel', 'error', 'pipe:1']
        )
        self._proc   = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._queue  = queue.Queue(maxsize=max(prefetch, 4))
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        try:
            while True:
                raw = self._proc.stdout.read(self._frame_bytes)
                if len(raw) < self._frame_bytes:
                    break
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
                self._queue.put(arr.copy())
        except Exception as e:
            self._queue.put(e)
            return
        self._queue.put(self._SENTINEL)

    def read(self) -> Optional[np.ndarray]:
        item = self._queue.get()
        if item is self._SENTINEL:
            return None
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        try:
            self._proc.stdout.read()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._proc.kill()


def _probe_video(video_path: str) -> dict:
    """轻量级 ffprobe 视频元信息探测（不依赖 ffmpeg-python）。"""
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

    # 帧率解析
    fps_str = vs.get('r_frame_rate', '24/1')
    try:
        fps = float(Fraction(fps_str))
    except (ValueError, ZeroDivisionError):
        fps = 24.0

    # 帧数
    if 'nb_frames' in vs and vs['nb_frames'] not in ('N/A', ''):
        nb = int(vs['nb_frames'])
    elif 'duration' in vs:
        nb = int(float(vs['duration']) * fps)
    else:
        nb = 0

    # 是否有音频：重新探测 audio 流
    cmd_audio = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_type',
        '-of', 'json', video_path,
    ]
    a = subprocess.run(cmd_audio, capture_output=True, text=True, timeout=15)
    has_audio = (a.returncode == 0 and '"codec_type": "audio"' in a.stdout)

    return {
        'width':     int(vs['width']),
        'height':    int(vs['height']),
        'fps':       fps,
        'nb_frames': nb,
        'has_audio': has_audio,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NVENC Writer（v5: 支持 NVENC + 批量写入 + error flag）
# ─────────────────────────────────────────────────────────────────────────────

class FFmpegWriter:
    """
    X1(v4): 批量攒帧写入（MAX_BATCH=8）。
    X5/error flag: 写帧线程异常 flag + print 双通知。
    M3(v5): 自动选择 NVENC 编码器。
    """
    _SENTINEL  = object()
    _MAX_BATCH = 8

    def __init__(
        self,
        output_path: str,
        width:  int,
        height: int,
        fps:    float,
        codec:  str = 'libx264',
        extra_codec_args: Optional[List[str]] = None,
        crf:    int = 23,
        audio_src: Optional[str] = None,
        ffmpeg_bin: str = 'ffmpeg',
    ):
        self._error: Optional[Exception] = None
        self._queue: queue.Queue = queue.Queue(maxsize=128)

        pix_fmt = 'yuv420p'
        if 'nvenc' in codec:
            # NVENC CQ 模式
            quality_args = ['-rc:v', 'vbr', '-cq:v', str(crf), '-b:v', '0']
        elif codec == 'libx265':
            # [FIX-NUMA] 容器内 set_mempolicy 受限，禁用 NUMA 线程池，
            # 消除 "set_mempolicy: Operation not permitted" 刷屏
            # 并防止线程池初始化异常导致的竖向条纹画面瑕疵。
            quality_args = ['-crf', str(crf),
                            '-x265-params', 'pools=none']
        else:
            quality_args = ['-crf', str(crf)]

        cmd = [
            ffmpeg_bin, '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', f'{fps:.6f}',
            '-i', 'pipe:0',
        ]
        if audio_src:
            # '1:a?' — trailing '?' makes the map optional: if audio_src has no
            # audio stream, FFmpeg silently skips it instead of aborting with
            # "Stream map '1:a' matches no streams."
            cmd += ['-i', audio_src, '-c:a', 'copy', '-map', '0:v', '-map', '1:a?']

        if extra_codec_args:
            cmd += ['-vcodec', codec] + extra_codec_args
        else:
            cmd += ['-vcodec', codec] + quality_args

        cmd += ['-pix_fmt', pix_fmt, '-loglevel', 'error', output_path]

        self._proc   = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        self._stderr_lines: List[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    # x265 正常 info/warning 前缀，以及容器内 NUMA 受限噪音，不应当作错误打印
    _STDERR_IGNORE = (
        'x265 [info]:', 'x265 [warning]:', 'set_mempolicy:',
        'encoded ', 'Weighted P-Frames', 'consecutive B-frames',
        'frame I:', 'frame P:', 'frame B:',
    )

    def _drain_stderr(self):
        """持续消费 FFmpeg stderr，防止 pipe buffer 满导致死锁；
        过滤 x265 info/set_mempolicy 噪音，只打印真正的错误行。"""
        try:
            for line in self._proc.stderr:
                decoded = line.decode(errors='ignore').rstrip()
                self._stderr_lines.append(decoded)
                if decoded and not any(decoded.lstrip().startswith(p)
                                       for p in self._STDERR_IGNORE):
                    print(f'[FFmpeg ERR] {decoded}')
        except Exception:
            pass

    def _write_loop(self):
        pending: List[bytes] = []
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.2)
                except queue.Empty:
                    if pending:
                        self._proc.stdin.write(b''.join(pending))
                        pending.clear()
                    continue

                if item is self._SENTINEL:
                    if pending:
                        self._proc.stdin.write(b''.join(pending))
                    break

                pending.append(item)
                if len(pending) >= self._MAX_BATCH or self._queue.empty():
                    self._proc.stdin.write(b''.join(pending))
                    pending.clear()
        except Exception as e:
            self._error = e
            print(f'[FFmpegWriter Error] {e}')

    def write(self, frame: np.ndarray):
        if self._error is not None:
            raise RuntimeError(f'FFmpegWriter 内部错误: {self._error}') from self._error
        self._queue.put(frame.tobytes())

    def close(self):
        self._queue.put(self._SENTINEL)
        self._thread.join(timeout=60)
        self._stderr_thread.join(timeout=5)
        # Close stdin FIRST to signal EOF to FFmpeg, then wait for it to exit.
        # Do NOT call communicate() after closing stdin — communicate() tries to
        # flush stdin internally and raises "flush of closed file".
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


# ─────────────────────────────────────────────────────────────────────────────
# 核心推理类（保留 v4 全部优化，v5 新增 TRT / NVDEC / NVENC）
# ─────────────────────────────────────────────────────────────────────────────

class IFRNetVideoProcessor:
    """
    IFRNet 插帧处理器（v5 单卡版）。
    保留全部单卡优化：FP16 / CUDA Graph / TRT / torch.compile /
    NVDEC 硬件解码 / NVENC 硬件编码 / OOM 自动降级 / PinnedBufferPool /
    批量写入 / 滑动窗口 FPS / JSON 报告。
    """

    def __init__(
        self,
        model_path:       str,
        device:           str = 'cuda',
        batch_size:       int = 4,
        max_batch_size:   int = 16,
        use_fp16:         bool = True,
        use_compile:      bool = True,
        use_cuda_graph:   bool = True,
        use_tensorrt:     bool = False,
        use_hwaccel:      bool = True,
        codec:            str = 'libx264',
        crf:              int = 23,
        keep_audio:       bool = True,
        ffmpeg_bin:       str = 'ffmpeg',
        report_json:      Optional[str] = None,
    ):
        self.model_path      = model_path
        self.device_str      = device
        self.batch_size      = batch_size
        self._max_batch_size = max(batch_size, max_batch_size)
        self._oom_cooldown   = 0
        self.use_fp16        = use_fp16 and (torch.cuda.is_available())
        self.use_cuda_graph  = use_cuda_graph and torch.cuda.is_available()
        self.use_tensorrt    = use_tensorrt
        self.use_hwaccel     = use_hwaccel
        self.codec           = codec
        self.crf             = crf
        self.keep_audio      = keep_audio
        self.ffmpeg_bin      = ffmpeg_bin
        self.report_json     = report_json
        self.dtype           = torch.float16 if self.use_fp16 else torch.float32

        self._pool          = TensorPool()
        self._graph:        dict = {}
        self._graph_inputs: dict = {}
        self._timing:       List[float] = []

        # 加载模型
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._load_model(self.device, use_compile)

    def _load_model(self, device: torch.device, use_compile: bool = True):
        """加载模型到指定设备。"""
        print(f'  加载模型: {self.model_path} → {device}')
        model = Model()
        ckpt  = torch.load(self.model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt)
        model = model.to(device).eval()
        # [FIX-CU] 开启 cuDNN benchmark：首次推理后自动选最优卷积算法
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        if self.use_fp16:
            model = model.half()
            print('  FP16 推理已启用')
        if use_compile and hasattr(torch, 'compile'):
            try:
                # Use 'default' mode instead of 'reduce-overhead'.
                #
                # Why NOT 'reduce-overhead':
                #   That mode enables torch._inductor's internal CUDA graph tree manager.
                #   This code dynamically ramps batch_size (4->5->...->N) on every
                #   successful inference, producing a different input shape each time.
                #   The inductor captures a new CUDA graph per shape. When OOM fires,
                #   empty_cache() frees GPU memory and invalidates the tensor weakrefs
                #   stored inside graph tree nodes. On the next call (even for a previously-
                #   seen shape) the assertion:
                #     assert len(node.tensor_weakrefs) == len(node.stack_traces)
                #   fails -> AssertionError crash (not caught by the OOM handler).
                #
                # 'default' mode still performs kernel fusion and Triton codegen (the bulk
                # of the speedup) but does NOT manage CUDA graphs internally, so dynamic
                # batch sizes and OOM recovery are fully safe.
                torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
                # Persist compiled kernels to disk so subsequent runs skip recompilation.
                # First run: ~2-3 min compile. Subsequent runs: near-instant startup.
                cache_dir = os.path.join(
                    os.path.dirname(os.path.abspath(self.model_path)),
                    '.torch_compile_cache',
                )
                os.makedirs(cache_dir, exist_ok=True)
                # Clear potentially corrupted /tmp cache on startup
                import tempfile, shutil
                tmp_inductor = os.path.join(tempfile.gettempdir(), 'torchinductor_root')
                if os.path.exists(tmp_inductor):
                    # Only clear if we suspect corruption (optional: always clear for safety)
                    pass  # or: shutil.rmtree(tmp_inductor, ignore_errors=True)
                os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', cache_dir)
                model = torch.compile(model, mode='default', dynamic=True)
                print(f'  torch.compile 加速已启用 (mode=default, dynamic=True)')
                print(f'  编译缓存目录: {cache_dir}')
                print(f'  首次运行将触发编译（约1-3分钟），后续运行秒启动')
                if self.use_cuda_graph:
                    self.use_cuda_graph = False
                    print('  手动 CUDA Graph 已禁用（由 torch.compile 接管）')
            except Exception as e:
                print(f'  torch.compile 不可用: {e}')
        self.model = model

        if device.type == 'cuda':
            self.stream_compute  = torch.cuda.Stream(device=device)
            self.stream_transfer = torch.cuda.Stream(device=device)
        else:
            self.stream_compute = self.stream_transfer = None

    # ──────────────────────────────────────────────────────────────────────────
    # M4: TensorRT 封装（可选）
    # ──────────────────────────────────────────────────────────────────────────

    def _build_trt_engine(self, input_shape: Tuple[int, int, int, int], cache_dir: str):
        """构建或加载 TRT Engine（单对帧推理形状）。"""
        try:
            import tensorrt as trt
        except ImportError:
            print('[TensorRT] 未安装，跳过 TRT 加速。')
            self.use_tensorrt = False
            return

        os.makedirs(cache_dir, exist_ok=True)
        B, C, H, W = input_shape
        tag      = f'ifrnet_B{B}_H{H}_W{W}_fp{"16" if self.use_fp16 else "32"}'
        trt_path = os.path.join(cache_dir, f'{tag}.trt')

        if os.path.exists(trt_path):
            print(f'[TensorRT] 加载缓存 Engine: {trt_path}')
        else:
            print(f'[TensorRT] 构建 Engine (shape={input_shape}) ...')
            onnx_path = os.path.join(cache_dir, f'{tag}.onnx')
            # IFRNet 有 4 个输入：img0, img1, embt, imgt_approx
            dummy0 = torch.randn(*input_shape, device=self.device)
            dummy1 = torch.randn(*input_shape, device=self.device)
            embt   = torch.full((B,), 0.5, dtype=torch.float32, device=self.device).view(B, 1, 1, 1)
            imgt_a = (dummy0 + dummy1) * 0.5
            if self.use_fp16:
                dummy0, dummy1, embt, imgt_a = (
                    dummy0.half(), dummy1.half(), embt.half(), imgt_a.half()
                )
            # [FIX-TRT] torch.compile 包装的模型导出时需解包 _orig_mod，
            # 否则权重名带 _orig_mod. 前缀导致 TRT parser 解析失败。
            export_model = getattr(self.model, '_orig_mod', self.model)
            with torch.no_grad():
                torch.onnx.export(
                    export_model,
                    (dummy0, dummy1, embt, imgt_a),
                    onnx_path,
                    input_names=['img0', 'img1', 'embt', 'imgt_approx'],
                    output_names=['output'],
                    opset_version=18,       # [FIX-TRT] torch.onnx >= 2.x 最低支持 18
                    dynamic_axes=None,
                )
            # [FIX-TRT] 确保无 .onnx.data 外部权重文件（TRT parser 只认单文件 ONNX）
            import onnx
            model_proto = onnx.load(onnx_path)
            onnx.save(model_proto, onnx_path,
                      save_as_external_data=False,
                      all_tensors_to_one_file=False)
            print(f'[TensorRT] ONNX 已导出: {onnx_path}')

            # 构建 Engine
            logger  = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(f'  [TRT ONNX Error] {parser.get_error(i)}')
                    self.use_tensorrt = False
                    return

            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
            if self.use_fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                print('[TensorRT] Engine 构建失败，回退 PyTorch 路径。')
                self.use_tensorrt = False
                return

            with open(trt_path, 'wb') as f:
                f.write(serialized)
            print(f'[TensorRT] Engine 已缓存: {trt_path}')

        # 加载 Engine
        try:
            logger  = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            with open(trt_path, 'rb') as f:
                self._trt_engine  = runtime.deserialize_cuda_engine(f.read())
            self._trt_context = self._trt_engine.create_execution_context()
            # [FIX-TRT] 动态查询 engine 的实际 tensor 名，不硬编码
            n = self._trt_engine.num_io_tensors
            inputs, outputs = [], []
            for i in range(n):
                name = self._trt_engine.get_tensor_name(i)
                mode = self._trt_engine.get_tensor_mode(name)
                import tensorrt as _trt
                if mode == _trt.TensorIOMode.INPUT:
                    inputs.append(name)
                else:
                    outputs.append(name)
            self._trt_input_names  = inputs   # e.g. ['img0','img1','embt','imgt_approx']
            self._trt_output_names = outputs  # e.g. ['add_25']
            print(f'[TensorRT] inputs={inputs} outputs={outputs}')
            self._trt_ok      = True
            print('[TensorRT] Engine 已激活，TRT 推理就绪。')
        except Exception as e:
            print(f'[TensorRT] Engine 加载失败: {e}，回退 PyTorch。')
            self.use_tensorrt = False

    # ──────────────────────────────────────────────────────────────────────────
    # CUDA Graph 推理（v4 B5 修复，完整保留）
    # ──────────────────────────────────────────────────────────────────────────

    def _get_cuda_graph(self, shape_key, img0, img1, embt, imgt_approx):
        if shape_key in self._graph:
            s = self._graph_inputs[shape_key]
            s['img0'].copy_(img0)
            s['img1'].copy_(img1)
            s['embt'].copy_(embt)
            s['imgt_approx'].copy_(imgt_approx)
            self._graph[shape_key].replay()
            return s['output']

        print(f'  [CUDA Graph] 捕获 shape={shape_key} ...')
        static_img0  = img0.clone()
        static_img1  = img1.clone()
        static_embt  = embt.clone()
        static_imgt  = imgt_approx.clone()

        for _ in range(3):
            with torch.cuda.stream(self.stream_compute):
                _ = self.model(static_img0, static_img1, static_embt, static_imgt)
        torch.cuda.current_stream().wait_stream(self.stream_compute)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=self.stream_compute):
            static_output = self.model(static_img0, static_img1, static_embt, static_imgt)
            if isinstance(static_output, tuple):
                static_output = static_output[0]

        self._graph[shape_key] = g
        self._graph_inputs[shape_key] = {
            'img0': static_img0, 'img1': static_img1,
            'embt': static_embt, 'imgt_approx': static_imgt,
            'output': static_output,
        }
        return static_output

    # ──────────────────────────────────────────────────────────────────────────
    # 核心批推理（v4 完整保留）
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer_batch(
        self,
        img0_list: List[np.ndarray],
        img1_list: List[np.ndarray],
        timesteps: List[float],
        orig_H: int,
        orig_W: int,
    ) -> List[List[np.ndarray]]:
        B = len(img0_list)
        T = len(timesteps)
        t0 = time.perf_counter()

        img0 = frames_to_tensor(img0_list, self.device, self.stream_transfer, self.dtype)
        img1 = frames_to_tensor(img1_list, self.device, self.stream_transfer, self.dtype)

        if self.stream_compute is not None:
            self.stream_compute.wait_stream(self.stream_transfer)

        img0_exp = img0.unsqueeze(1).expand(B, T, *img0.shape[1:]).reshape(B * T, *img0.shape[1:])
        img1_exp = img1.unsqueeze(1).expand(B, T, *img1.shape[1:]).reshape(B * T, *img1.shape[1:])
        shape_key = (B * T, 3, img0.shape[2], img0.shape[3], T)

        if self.use_cuda_graph:
            with torch.cuda.stream(self.stream_compute):
                t_vals     = timesteps * B
                embt       = torch.tensor(t_vals, dtype=self.dtype, device=self.device).view(-1, 1, 1, 1)
                img0_big   = img0_exp.contiguous()
                img1_big   = img1_exp.contiguous()
                imgt_approx = img0_big * (1 - embt) + img1_big * embt
                pred_big   = self._get_cuda_graph(shape_key, img0_big, img1_big, embt, imgt_approx)
        elif getattr(self, '_trt_ok', False):
            # [FIX-TRT] TensorRT 静态 Engine 推理路径
            import numpy as np
            t_vals      = timesteps * B
            embt_t      = torch.tensor(t_vals, dtype=torch.float32, device=self.device).view(-1, 1, 1, 1)
            imgt_approx = img0_exp.float() * (1 - embt_t) + img1_exp.float() * embt_t
            # 准备 fp16 或 fp32 输入
            dtype_np = np.float16 if self.use_fp16 else np.float32
            i0 = img0_exp.half().contiguous() if self.use_fp16 else img0_exp.float().contiguous()
            i1 = img1_exp.half().contiguous() if self.use_fp16 else img1_exp.float().contiguous()
            em = embt_t.half().contiguous() if self.use_fp16 else embt_t.contiguous()
            ia = imgt_approx.half().contiguous() if self.use_fp16 else imgt_approx.contiguous()
            BT = i0.shape[0]
            C, H_p, W_p = i0.shape[1], i0.shape[2], i0.shape[3]
            out_buf = torch.empty((BT, 3, H_p, W_p),
                                  dtype=torch.float16 if self.use_fp16 else torch.float32,
                                  device=self.device)
            ctx = self._trt_context
            # [FIX-TRT] 绑定所有输入 tensor
            in_names  = getattr(self, '_trt_input_names',
                                ['img0', 'img1', 'embt', 'imgt_approx'])
            out_names = getattr(self, '_trt_output_names', ['output'])
            for name, buf in zip(in_names, [i0, i1, em, ia]):
                ctx.set_tensor_address(name, buf.data_ptr())
            # [FIX-TRT] TRT 要求所有输出 tensor 都必须绑定地址（包括不需要的中间输出）
            # out_names[0] 是主输出，其余分配同形状临时 buffer
            import tensorrt as _trt2
            ctx.set_tensor_address(out_names[0], out_buf.data_ptr())
            _dummy_bufs = []
            for _out_name in out_names[1:]:
                _shape = tuple(ctx.get_tensor_shape(_out_name))
                _dtype = ctx.get_tensor_dtype(_out_name)
                _torch_dtype = torch.float16 if _dtype == _trt2.DataType.HALF else torch.float32
                _dummy = torch.empty(_shape, dtype=_torch_dtype, device=self.device)
                ctx.set_tensor_address(_out_name, _dummy.data_ptr())
                _dummy_bufs.append(_dummy)  # 保持引用，防止 GC
            stream = torch.cuda.current_stream().cuda_stream
            ctx.execute_async_v3(stream_handle=stream)
            torch.cuda.current_stream().synchronize()
            pred_big = out_buf.float() if self.use_fp16 else out_buf
        else:
            autocast_ctx = (
                torch.amp.autocast(device_type='cuda', dtype=torch.float16)
                if self.use_fp16 else nullcontext()
            )
            stream_ctx = (
                torch.cuda.stream(self.stream_compute)
                if self.stream_compute else nullcontext()
            )
            with stream_ctx, autocast_ctx:
                t_vals      = timesteps * B
                embt        = torch.tensor(t_vals, dtype=self.dtype, device=self.device).view(-1, 1, 1, 1)
                imgt_approx = img0_exp * (1 - embt) + img1_exp * embt
                out         = self.model(img0_exp, img1_exp, embt, imgt_approx)
                pred_big    = out[0] if isinstance(out, tuple) else out

        if self.use_fp16:
            pred_big = pred_big.float()

        all_np = tensor_to_np(pred_big, orig_H, orig_W, sync_stream=self.stream_compute)
        result = [[all_np[i * T + j] for j in range(T)] for i in range(B)]

        self._timing.append(time.perf_counter() - t0)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # OOM 自动降级 + 恢复（v4 完整保留）
    # ──────────────────────────────────────────────────────────────────────────

    def _estimate_safe_batch_size(self, H: int, W: int) -> int:
        """根据当前实测空闲显存估算安全的 batch_size。"""
        if not torch.cuda.is_available():
            return 1
        try:
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            # 单帧 FP16 字节数 × 6 倍激活系数（经验值）
            bytes_per_frame = H * W * 3 * 2 * 6
            # 只使用空闲显存的 70%，留 30% 给中间缓冲
            estimated = max(1, int(free_bytes * 0.7 / bytes_per_frame))
            return min(estimated, self._max_batch_size)
        except Exception:
            return 1

    def _safe_infer(self, img0_list, img1_list, timesteps, orig_H, orig_W):
        # _in_oom_cascade: True 表示当前处于同一次 OOM 的连锁降级中，
        # 此时不再更新 max_batch_size（避免级联惩罚）
        in_oom_cascade = False

        while True:
            try:
                result = self._infer_batch(img0_list, img1_list, timesteps, orig_H, orig_W)
                in_oom_cascade = False  # 推理成功，退出级联状态
                if self._oom_cooldown > 0:
                    self._oom_cooldown -= 1
                elif (self.batch_size < self._max_batch_size
                      and not getattr(self, '_trt_ok', False)):
                    # [FIX-TRT] TRT 静态 engine batch 必须固定，禁止扩张
                    # [FIX-COMPILE] torch.compile dynamic 每次新 shape 重编，步长改为保守值
                    new_bs = min(self.batch_size + 1, self._max_batch_size)
                    print(f'[恢复] 显存充裕，batch_size {self.batch_size} → {new_bs}')
                    self.batch_size = new_bs
                return result

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self._pool.clear()
                self._graph.clear()

                if not in_oom_cascade:
                    # ── 首次 OOM：永久更新天花板，此 batch_size 已证明不可用 ──
                    safe_ceiling = max(1, self.batch_size - 1)
                    if self._max_batch_size > safe_ceiling:
                        print(f'[OOM] 永久降低 max_batch_size: {self._max_batch_size} → {safe_ceiling}')
                        self._max_batch_size = safe_ceiling
                    in_oom_cascade = True  # 进入级联状态，后续降级不再修改上限
                else:
                    # ── 级联 OOM：内存仍脏，不修改 max_batch_size ──
                    pass

                if self.batch_size <= 1:
                    # 深度清理：同步 + 清空 + 重置 inductor
                    print(f'\n[OOM] batch_size=1 仍 OOM，深度清理后按剩余显存估算恢复...')
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        try:
                            torch._dynamo.reset()
                        except Exception:
                            pass
                        torch.cuda.empty_cache()  # 二次清理，inductor reset 后可能释放更多

                    # 深度清理后用实测空闲显存重新估算 batch_size
                    recovered_bs = self._estimate_safe_batch_size(orig_H, orig_W)
                    # 同时以此作为新的上限（深度清理后能用多少就上限多少）
                    if recovered_bs < self._max_batch_size:
                        print(f'[OOM] 深度清理后估算安全 batch_size={recovered_bs}，'
                              f'更新 max_batch_size: {self._max_batch_size} → {recovered_bs}')
                        self._max_batch_size = recovered_bs
                    self.batch_size = recovered_bs
                    self._oom_cooldown = 20  # 稳定一段时间再尝试爬升
                    in_oom_cascade = False   # 深度清理后重置级联标志
                    print(f'[OOM] 恢复 batch_size={self.batch_size}，继续处理...')
                    continue

                self.batch_size = max(1, self.batch_size // 2)
                self._oom_cooldown = 10
                print(f'\n[OOM] 自动降低 batch_size → {self.batch_size}')

    # ──────────────────────────────────────────────────────────────────────────
    # 单段处理核心
    # ──────────────────────────────────────────────────────────────────────────

    def _process_segment(
        self,
        input_path:         str,
        output_path:        str,
        scale:              float,
        frame_start:        int = 0,
        frame_end:          int = -1,
        skip_first_output:  bool = False,
        audio_src:          Optional[str] = None,
        codec_override:     Optional[str] = None,
        extra_codec_args:   Optional[List[str]] = None,
        worker_label:       str = '',
        preview:            bool = False,
        preview_interval:   int = 30,
    ) -> Tuple[bool, int, int]:
        """
        处理视频的一个帧范围段，写出到 output_path。
        返回 (成功, 原始帧数, 输出帧数)。
        """
        reader = FFmpegFrameReader(
            input_path,
            frame_start  = frame_start,
            frame_end    = frame_end,
            prefetch     = self.batch_size * 3,
            use_hwaccel  = self.use_hwaccel,
            ffmpeg_bin   = self.ffmpeg_bin,
        )
        W, H      = reader.width, reader.height
        fps       = reader.fps
        n_seg_est = reader._segment_frames   # 估计值

        # ── 根据分辨率限制最大 batch_size，防止大分辨率下 VRAM 溢出 ──────────
        # 估算单帧 FP16 字节数（×6 为模型中间激活的经验系数）
        bytes_per_frame = W * H * 3 * 2 * 6  # 3 channels, fp16=2B, ~6x activations
        free_bytes = 0
        if torch.cuda.is_available():
            free_bytes = torch.cuda.mem_get_info(self.device)[0]
        if free_bytes > 0:
            res_max_bs = max(1, int(free_bytes * 0.6 / bytes_per_frame))
            if self.batch_size > res_max_bs:
                print(f'[分辨率限制] {W}×{H} 下 batch_size {self.batch_size} → {res_max_bs}')
                self.batch_size = res_max_bs
            if self._max_batch_size > res_max_bs:
                self._max_batch_size = max(self.batch_size, res_max_bs)

        _, pad_h, pad_w = pad_to_stride(np.zeros((H, W, 3), dtype=np.uint8))
        need_pad = pad_h > 0 or pad_w > 0

        scale_frac = Fraction(scale).limit_denominator(64)
        n_interp   = int(scale_frac) - 1
        if n_interp < 1:
            print(f'[{worker_label}] 错误: scale 必须 ≥ 2，当前={scale}')
            reader.close()
            return False, 0, 0
        if n_interp > 32:
            scale_frac = Fraction(33)
            n_interp   = 32
        timesteps = [float(Fraction(i, int(scale_frac))) for i in range(1, int(scale_frac))]
        new_fps   = fps * float(scale_frac)

        use_codec = codec_override or HardwareCapability.best_encoder(self.codec)
        use_extra = extra_codec_args
        if 'nvenc' in use_codec:
            print(f'[{worker_label}] NVENC 编码: {use_codec}')

        # ── torch.compile 预热 ────────────────────────────────────────────────
        # torch.compile 的实际编译发生在第一次 forward 时，会阻塞数分钟。
        # 在开启 writer 和进度条之前做预热，让用户看到明确的"编译中"提示，
        # 而不是进度条卡在 4 帧假死。
        if hasattr(self.model, '_is_compiled') or (
            hasattr(self.model, '_orig_mod')  # torch.compile wraps with _orig_mod
        ):
            _need_warmup = True
        else:
            # check via dynamo
            try:
                _need_warmup = hasattr(self.model, '_dynamo_ctx')
            except Exception:
                _need_warmup = False
        # Simpler reliable check: always warmup once per segment if compile is active
        if not getattr(self, '_warmup_done', False):
            # ── 预热形状说明 ────────────────────────────────────────────────────
            # 使用固定小形状 (1×3×32×32) 而非真实分辨率做编译预热。
            # 原因：Triton 在超大 shape（如 1440×2560）首次编译时会生成巨大的 .so
            # 文件，某些 CUDA/Triton 版本组合下会触发 C 级堆损坏 → SIGABRT →
            # double free or corruption。
            # dynamic=True 模式下，Triton 编译出符号化 kernel，对任意后续 shape
            # 均有效，无需用真实分辨率触发编译。
            _WARM_H, _WARM_W = 32, 32   # 最小对齐单位，编译快速且稳定
            _bs_warm = 1
            print(f'  [预热] torch.compile 编译中 (小形状预热 {_bs_warm}×3×{_WARM_H}×{_WARM_W})...',
                  flush=True)
            _t_warm = time.perf_counter()
            try:
                with torch.no_grad():
                    _d0   = torch.zeros(_bs_warm, 3, _WARM_H, _WARM_W,
                                        dtype=self.dtype, device=self.device)
                    _d1   = torch.zeros_like(_d0)
                    _embt = torch.tensor([0.5] * _bs_warm,
                                         dtype=self.dtype, device=self.device).view(-1,1,1,1)
                    _imgt = (_d0 + _d1) * 0.5
                    _out  = self.model(_d0, _d1, _embt, _imgt)
                    # 显式释放所有引用后再 synchronize，防止析构顺序引发堆损坏
                    del _out, _d0, _d1, _embt, _imgt
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                _t_elapsed = time.perf_counter() - _t_warm
                print(f'  [预热] 编译完成，耗时 {_t_elapsed:.1f}s，后续帧将正常速度运行',
                      flush=True)
            except Exception as _we:
                print(f'  [预热] 编译失败，回退至 eager 模式: {_we}', flush=True)
                if hasattr(self.model, '_orig_mod'):
                    self.model = self.model._orig_mod
                else:
                    torch._dynamo.reset()
            self._warmup_done = True

        writer = FFmpegWriter(
            output_path, W, H, new_fps,
            codec      = use_codec,
            extra_codec_args = use_extra,
            crf        = self.crf,
            audio_src  = audio_src,
            ffmpeg_bin = self.ffmpeg_bin,
        )

        padded_buf: List[np.ndarray] = []
        raw_buf:    List[np.ndarray] = []
        frame_count  = 0
        output_count = 0
        meter        = ThroughputMeter(window=20)

        def maybe_pad(f: np.ndarray) -> np.ndarray:
            return np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge') if need_pad else f

        def flush_buf():
            nonlocal output_count
            if len(raw_buf) < 2:
                return
            n_pairs = len(raw_buf) - 1
            results = self._safe_infer(padded_buf[:-1], padded_buf[1:], timesteps, H, W)
            for i, interps in enumerate(results):
                for interp_frame in interps:
                    writer.write(interp_frame)
                    output_count += 1
                writer.write(raw_buf[i + 1])
                output_count += 1
            meter.update(n_pairs)

        desc = f'[{worker_label}] 插帧'
        pbar = tqdm(total=n_seg_est, unit='帧', desc=desc, dynamic_ncols=True) if HAS_TQDM else None

        # ── 读取第一帧 ─────────────────────────────────────────────────────
        first = reader.read()
        if first is None:
            print(f'[{worker_label}] 无法读取首帧')
            reader.close()
            writer.close()
            if pbar:
                pbar.close()
            return False, 0, 0

        # skip_first_output=False 时正常写入首帧
        if not skip_first_output:
            writer.write(first)
            output_count += 1

        raw_buf.append(first)
        padded_buf.append(maybe_pad(first))
        frame_count = 1
        if pbar:
            pbar.update(1)

        # ── 主读帧循环 ─────────────────────────────────────────────────────
        t_start = time.time()
        while True:
            frame = reader.read()
            if frame is None:
                break

            frame_count += 1
            raw_buf.append(frame)
            padded_buf.append(maybe_pad(frame))

            if len(raw_buf) == self.batch_size + 1:
                flush_buf()
                raw_buf    = [raw_buf[-1]]
                padded_buf = [padded_buf[-1]]

            if pbar:
                pbar.update(1)
                avg_t = np.mean(self._timing[-20:]) * 1000 if self._timing else 0
                pbar.set_postfix(
                    fps=f'{meter.fps():.1f}',
                    eta=f'{meter.eta(n_seg_est):.0f}s',
                    ms=f'{avg_t:.0f}',
                )

            if preview and frame_count % preview_interval == 0:
                import cv2
                cv2.imshow(f'IFRNet Preview [{worker_label}]', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if len(raw_buf) > 1:
            flush_buf()

        if pbar:
            pbar.close()
        writer.close()
        reader.close()

        elapsed = time.time() - t_start
        print(f'[{worker_label}] 完成 | 原始帧={frame_count} → 输出帧={output_count} | '
              f'{frame_count/elapsed:.1f} 原始帧/s')
        return True, frame_count, output_count

    # ──────────────────────────────────────────────────────────────────────────
    # 对外公开接口
    # ──────────────────────────────────────────────────────────────────────────

    def process_video(
        self,
        input_path:    str,
        output_path:   str,
        scale:         float = 2.0,
        preview:       bool  = False,
        preview_interval: int = 30,
    ) -> bool:
        """处理完整视频（单 GPU）。"""
        if not os.path.exists(input_path):
            print(f'错误: 输入不存在 - {input_path}')
            return False
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        audio_src = input_path if self.keep_audio else None

        # M4: TRT 初始化（如需要）
        if self.use_tensorrt:
            meta = _probe_video(input_path)
            sh   = (self.batch_size, 3, meta['height'], meta['width'])
            trt_dir = os.path.join(os.path.dirname(output_path) or '.', '.trt_cache')
            self._build_trt_engine(sh, trt_dir)

        ok, fc, oc = self._process_segment(
            input_path, output_path, scale,
            frame_start=0, frame_end=-1,
            skip_first_output=False,
            audio_src=audio_src,
            worker_label='GPU0',
            preview=preview,
            preview_interval=preview_interval,
        )
        if ok:
            self._print_summary(input_path, output_path, fc, oc, scale)
            self._dump_report(input_path, output_path, fc, oc, scale)
        return ok

    def _print_summary(self, input_path, output_path, fc, oc, scale):
        print(f'\n✅ 插帧完成！')
        if oc > 0:
            print(f'   原始帧数: {fc} → 输出帧数: {oc} (×{scale:.1f})')
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f'   输出: {output_path} ({size_mb:.1f} MB)')

    def _dump_report(self, input_path, output_path, fc, oc, scale):
        if not self.report_json or not self._timing:
            return
        report = {
            'input':      input_path,
            'output':     output_path,
            'scale':      scale,
            'batch_size': self.batch_size,
            'fp16':       self.use_fp16,
            'cuda_graph': self.use_cuda_graph,
            'tensorrt':   getattr(self, '_trt_ok', False),
            'nvdec':      HardwareCapability.has_nvdec(),
            'nvenc':      HardwareCapability.best_encoder(self.codec).endswith('nvenc'),
            'n_workers':  1,
            'frame_count':  fc,
            'output_count': oc,
            'infer_latency_ms': {
                'mean': round(float(np.mean(self._timing)) * 1000, 2),
                'p95':  round(float(np.percentile(self._timing, 95)) * 1000, 2),
                'max':  round(float(np.max(self._timing)) * 1000, 2),
            },
        }
        with open(self.report_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f'   性能报告: {self.report_json}')


# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='IFRNet 视频插帧 —— 终极优化版 v5（单卡版）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 基础参数
    parser.add_argument('--input',    required=True,  help='输入视频路径')
    parser.add_argument('--output',   required=True,  help='输出视频路径')
    parser.add_argument('--scale',    type=float, default=2.0, help='插帧倍数（≥2 整数）')
    parser.add_argument('--model',    default='IFRNet_S_Vimeo90K', help='模型名称或 .pth 路径')
    parser.add_argument('--device',   default='cuda',  choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size',  type=int, default=4)
    # 推理优化
    parser.add_argument('--no_fp16',     action='store_true', help='禁用 FP16')
    parser.add_argument('--no_compile',  action='store_true', help='禁用 torch.compile')
    parser.add_argument('--no_cuda_graph', action='store_true', help='禁用 CUDA Graph')
    parser.add_argument('--use_tensorrt', action='store_true',
                        help='[V5] 启用 TensorRT 加速（首次需构建 Engine）')
    # 硬件加速
    parser.add_argument('--no_hwaccel',  action='store_true',
                        help='[V5] 强制禁用 NVDEC 硬件解码')
    # 编码参数
    parser.add_argument('--codec',    default='libx264',
                        help='输出编码器（有 NVENC 时自动升级）')
    parser.add_argument('--crf',      type=int, default=23)
    parser.add_argument('--no_audio', action='store_true')
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg')
    # 调试
    parser.add_argument('--preview',  action='store_true')
    parser.add_argument('--preview_interval', type=int, default=30)
    parser.add_argument('--report',   default=None, help='JSON 性能报告路径')

    args = parser.parse_args()

    # 模型路径解析
    if args.model in MODEL_NAME_MAP:
        model_path = f'{models_ifrnet}/{MODEL_NAME_MAP[args.model]}'
    else:
        model_path = args.model
    if not os.path.exists(model_path):
        print(f'错误: 模型不存在 - {model_path}')
        sys.exit(1)

    # 打印启动信息
    print('=' * 65)
    print('  IFRNet 视频插帧 —— 终极优化版 v5（单卡版）')
    print('=' * 65)
    print(f'  模型:   {args.model}')
    print(f'  设备:   {args.device} | GPU: '
          f'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print(f'  FP16:   {not args.no_fp16} | '
          f'CUDA Graph: {not args.no_cuda_graph} | '
          f'TensorRT: {args.use_tensorrt}')
    print(f'  NVDEC:  {HardwareCapability.has_nvdec() and not args.no_hwaccel} | '
          f'NVENC(h264): {HardwareCapability.has_nvenc("h264_nvenc")} | '
          f'NVENC(hevc): {HardwareCapability.has_nvenc("hevc_nvenc")}')
    print(f'  编码器: {args.codec} → 实际: {HardwareCapability.best_encoder(args.codec)} | '
          f'CRF: {args.crf}')
    print()

    t_total = time.time()
    processor = IFRNetVideoProcessor(
        model_path         = model_path,
        device             = args.device,
        batch_size         = args.batch_size,
        max_batch_size     = args.batch_size * 4,
        use_fp16           = not args.no_fp16,
        use_compile        = not args.no_compile,
        use_cuda_graph     = not args.no_cuda_graph,
        use_tensorrt       = args.use_tensorrt,
        use_hwaccel        = not args.no_hwaccel,
        codec              = args.codec,
        crf                = args.crf,
        keep_audio         = not args.no_audio,
        ffmpeg_bin         = args.ffmpeg_bin,
        report_json        = args.report,
    )

    ok = processor.process_video(
        args.input, args.output,
        scale            = args.scale,
        preview          = args.preview,
        preview_interval = args.preview_interval,
    )

    m, s = divmod(int(time.time() - t_total), 60)
    print(f'\n⏱️  总耗时（含模型加载）: {m}分{s}秒')
    if ok and os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / 1024 / 1024
        print(f'✅ 输出: {args.output} ({size_mb:.1f} MB)')
    else:
        print('❌ 处理失败')
        sys.exit(1)


if __name__ == '__main__':
    main()
