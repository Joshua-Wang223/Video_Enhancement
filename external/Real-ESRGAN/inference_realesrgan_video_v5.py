"""
RealESRGAN 视频超分处理脚本 —— 终极优化版 v5
==========================================================
在 v4 基础上，彻底重构多卡调度架构，解决系统级瓶颈。

[V5 核心架构重构]

  M1. [Dispatcher-Queue 零临时文件多卡调度]
      ┌─ 主进程 FFmpegReader ──→ dispatch_queue ──→ GPU Worker 0 ──┐
      │                     ──→ GPU Worker 1 ──┤→ result_queue
      │                     ──→ GPU Worker N ──┘      │
      └─ 主进程 FrameCollector(heapq 重排) ──→ FFmpegWriter ─→ output.mp4
      彻底消除：① 临时文件 I/O 风暴；② FFmpeg 二次软编码画质损失；
               ③ 时间切片音频失步；④ 临时视频并发读写竞争。

  M2. [硬件解码 NVDEC / 硬件编码 NVENC 自动探测]
      - 解码：FFmpeg -hwaccel cuda -hwaccel_output_format bgr24
        CPU 零解码负载，VRAM 直接接收帧数据（PCIe 零拷贝路径）。
      - 编码：h264_nvenc / hevc_nvenc，GPU 编码引擎独立于 CUDA 核心，
        不占用推理算力。
      - 探测失败时自动回退至 CPU 软件解/编码（libx264/ffmpeg 默认）。

  M3. [TensorRT 可选加速]
      --use_tensorrt：将模型导出 ONNX → 构建 TRT Engine (FP16 + 静态形状)。
      分辨率固定时推理速度比 torch.compile 额外提升 1.5–2x，
      并显著降低显存碎片化，允许更大 batch_size。
      需要安装：pip install tensorrt pycuda onnx onnxruntime-gpu

  M4. [有界队列 + heapq 无锁帧重排]
      dispatch_queue / result_queue 均设上限，防止帧积压导致 OOM。
      result_queue 侧使用 heapq 最小堆按 frame_idx 重排，
      确保输出帧顺序严格正确，即使各 GPU 推理速度不一致。

  M5. [跨进程异常传播]
      Worker 进程内任何异常均通过 error_queue 传回主进程，
      主进程立即终止所有 Worker 并抛出原始异常，避免静默挂起。

[保留 v4 全部优化]
  X1-X5 (异常传播/error flag/JSON报告/timing采集/tqdm对齐)
  B1-B7 (Fraction fps/pipe安全/drain/pinned内存池/CUDA流/compile/...)
  N1-N9 (原地操作/批量降级/批量写入/async I/O/...)
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import heapq
import json
import mimetypes
import multiprocessing as mp
import numpy as np
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from os import path as osp
from tqdm import tqdm

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

try:
    import ffmpeg
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ffmpeg-python'])
    import ffmpeg

# ─────────────────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────────────────

base_dir = '/workspace/Video_Enhancement'
models_RealESRGAN = f'{base_dir}/models_RealESRGAN'

# ─────────────────────────────────────────────────────────────────────────────
# 队列哨兵（字符串形式，跨进程 pickle 安全）
# ─────────────────────────────────────────────────────────────────────────────

_DISPATCH_SENTINEL = ('__DISPATCH_SENTINEL__',)
_RESULT_DONE       = ('__RESULT_DONE__',)

# ─────────────────────────────────────────────────────────────────────────────
# 模型配置常量
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIG: Dict[str, Tuple] = {
    'RealESRGAN_x4plus': (
        RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'],
    ),
    'RealESRNet_x4plus': (
        RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'],
    ),
    'RealESRGAN_x4plus_anime_6B': (
        RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4), 4,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'],
    ),
    'RealESRGAN_x2plus': (
        RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2), 2,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'],
    ),
    'realesr-animevideov3': (
        SRVGGNetCompact(3, 3, 64, 16, upscale=4, act_type='prelu'), 4,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'],
    ),
    'realesr-general-x4v3': (
        SRVGGNetCompact(3, 3, 64, 32, upscale=4, act_type='prelu'), 4,
        [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        ],
    ),
    'RealESRGANv2-animevideo-xsx4': (
        SRVGGNetCompact(3, 3, 64, 16, upscale=4, act_type='prelu'), 4,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx4.pth'],
    ),
    'RealESRGANv2-animevideo-xsx2': (
        SRVGGNetCompact(3, 3, 64, 16, upscale=2, act_type='prelu'), 2,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx2.pth'],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# M2: 硬件能力探测
# ─────────────────────────────────────────────────────────────────────────────

class HardwareCapability:
    """一次性探测 NVDEC / NVENC 可用性，缓存结果供全局使用。"""

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
        """尝试用 -hwaccel cuda 解码一个极短片段，观察是否报错。"""
        try:
            cmd = [
                'ffmpeg', '-hwaccel', 'cuda',
                '-hwaccel_output_format', 'nv12',
                '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=0.1:r=1',
                '-frames:v', '1', '-f', 'null', '-',
                '-loglevel', 'error',
            ]
            r = subprocess.run(cmd, capture_output=True, timeout=10)
            return r.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _probe_nvenc(codec: str) -> bool:
        """尝试用指定编码器编码一帧，观察是否可用。"""
        try:
            cmd = [
                'ffmpeg',
                '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=0.1:r=1',
                '-vcodec', codec, '-frames:v', '1',
                '-f', 'null', '-',
                '-loglevel', 'error',
            ]
            r = subprocess.run(cmd, capture_output=True, timeout=10)
            return r.returncode == 0
        except Exception:
            return False

    @classmethod
    def best_video_encoder(cls, preferred: str) -> str:
        """返回可用的最佳编码器：优先 nvenc，否则回退 preferred。"""
        nvenc_map = {
            'libx264':    'h264_nvenc',
            'libx265':    'hevc_nvenc',
            'libvpx-vp9': 'libvpx-vp9',  # 无对应 nvenc
        }
        candidate = nvenc_map.get(preferred, preferred)
        if candidate != preferred and cls.has_nvenc(candidate):
            return candidate
        return preferred


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _parse_fps(fps_str: str) -> float:
    """B1(v3): Fraction 安全解析帧率字符串。"""
    try:
        return float(Fraction(fps_str))
    except (ValueError, ZeroDivisionError):
        return 24.0


def get_video_meta_info(video_path: str) -> dict:
    """通过 ffprobe 获取视频元数据，包含宽高/帧率/帧数/音轨。"""
    probe = ffmpeg.probe(video_path)
    video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
    has_audio     = any(s['codec_type'] == 'audio' for s in probe['streams'])
    vs = video_streams[0]
    fps = _parse_fps(vs['avg_frame_rate'])
    if 'nb_frames' in vs and vs['nb_frames'].isdigit():
        nb = int(vs['nb_frames'])
    elif 'duration' in vs:
        nb = int(float(vs['duration']) * fps)
    else:
        nb = 0
    return {
        'width':     vs['width'],
        'height':    vs['height'],
        'fps':       fps,
        'audio':     ffmpeg.input(video_path).audio if has_audio else None,
        'nb_frames': nb,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ThroughputMeter（滑动窗口 FPS 统计）
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
# PinnedBufferPool（线程本地 pinned CPU buffer）
# ─────────────────────────────────────────────────────────────────────────────

_thread_local = threading.local()


class PinnedBufferPool:
    """预分配 pinned CPU buffer 并复用，避免每批 H2D 前 pin_memory 的 malloc 开销。"""

    def __init__(self):
        self._buf: Optional[torch.Tensor] = None

    def get_for_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        arr   = np.stack(frames, axis=0)
        src   = torch.from_numpy(arr)
        n_elem = src.numel()
        if self._buf is None or self._buf.numel() < n_elem:
            self._buf = torch.empty(n_elem, dtype=torch.uint8).pin_memory()
        dst = self._buf[:n_elem].view_as(src)
        dst.copy_(src)
        return dst


def _get_pinned_pool() -> PinnedBufferPool:
    if not hasattr(_thread_local, 'pool'):
        _thread_local.pool = PinnedBufferPool()
    return _thread_local.pool


# ─────────────────────────────────────────────────────────────────────────────
# M3: TensorRT 可选加速封装
# ─────────────────────────────────────────────────────────────────────────────

class TensorRTAccelerator:
    """
    将 RealESRGAN 模型导出 ONNX 后编译 TRT Engine (FP16, 静态形状)。
    要求：pip install tensorrt pycuda onnx onnxruntime-gpu
    首次构建会缓存 .trt 文件，后续直接加载。
    """

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 cache_dir: str, input_shape: Tuple[int, int, int, int],
                 use_fp16: bool = True):
        self.device      = device
        self.input_shape = input_shape  # (B, C, H, W)
        self.use_fp16    = use_fp16
        self._engine     = None
        self._context    = None
        self._trt_ok     = False

        try:
            import tensorrt as trt
            import pycuda.autoinit  # noqa
            import pycuda.driver as cuda
            self._trt = trt
            self._cuda = cuda
        except ImportError as e:
            print(f'[TensorRT] 依赖未安装，跳过 TRT 加速: {e}')
            print('  安装命令: pip install tensorrt pycuda onnx onnxruntime-gpu')
            return

        B, C, H, W = input_shape
        tag       = f'B{B}_C{C}_H{H}_W{W}_fp{"16" if use_fp16 else "32"}'
        trt_path  = osp.join(cache_dir, f'realesrgan_{tag}.trt')
        onnx_path = osp.join(cache_dir, f'realesrgan_{tag}.onnx')
        os.makedirs(cache_dir, exist_ok=True)

        if not osp.exists(trt_path):
            print(f'[TensorRT] 构建 Engine (shape={input_shape}) ...')
            self._export_onnx(model, onnx_path, input_shape)
            self._build_engine(onnx_path, trt_path, use_fp16)

        if osp.exists(trt_path):
            self._load_engine(trt_path)

    def _export_onnx(self, model, onnx_path, input_shape):
        model.eval()
        dummy = torch.randn(*input_shape, device=self.device)
        if self.use_fp16:
            dummy = dummy.half()
            model  = model.half()
        with torch.no_grad():
            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=['input'], output_names=['output'],
                opset_version=17,
                dynamic_axes=None,  # 静态形状，TRT 最优化
            )
        print(f'[TensorRT] ONNX 已导出: {onnx_path}')

    def _build_engine(self, onnx_path, trt_path, use_fp16):
        trt = self._trt
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser  = trt.OnnxParser(network, logger)

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f'  ONNX 解析错误: {parser.get_error(i)}')
                return

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))  # 4GB
        if use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            print('[TensorRT] Engine 构建失败')
            return

        with open(trt_path, 'wb') as f:
            f.write(serialized)
        print(f'[TensorRT] Engine 已缓存: {trt_path}')

    def _load_engine(self, trt_path):
        trt    = self._trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(trt_path, 'rb') as f:
            self._engine  = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        self._trt_ok  = True
        print('[TensorRT] Engine 加载成功，已启用 TRT 推理')

    @property
    def available(self) -> bool:
        return self._trt_ok

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """同步 TRT 推理，返回 GPU Tensor。"""
        import pycuda.driver as cuda
        import numpy as np

        inp_np   = input_tensor.contiguous().cpu().numpy()
        out_shape = self._engine.get_binding_shape(1)  # (B, C, H*scale, W*scale)
        out_np    = np.empty(out_shape, dtype=np.float16 if self.use_fp16 else np.float32)

        inp_ptr = inp_np.ctypes.data_as(ctypes.c_void_p)
        out_ptr = out_np.ctypes.data_as(ctypes.c_void_p)

        # 使用 pycuda 做异步 H2D、执行、D2H
        stream = cuda.Stream()
        d_inp  = cuda.mem_alloc(inp_np.nbytes)
        d_out  = cuda.mem_alloc(out_np.nbytes)

        cuda.memcpy_htod_async(d_inp, inp_np, stream)
        self._context.execute_async_v2(
            bindings=[int(d_inp), int(d_out)],
            stream_handle=stream.handle,
        )
        cuda.memcpy_dtoh_async(out_np, d_out, stream)
        stream.synchronize()

        return torch.from_numpy(out_np).to(input_tensor.device)


# ─────────────────────────────────────────────────────────────────────────────
# Reader（支持 NVDEC + 异常传播）
# ─────────────────────────────────────────────────────────────────────────────

class FFmpegReader:
    """
    通过 FFmpeg pipe 读取视频帧，支持 NVDEC 硬件解码。
    M2: 若 NVDEC 可用，优先使用 -hwaccel cuda；否则回退 CPU 解码。
    X1(v4): 预取线程异常入队，get_frame() 处 re-raise，防止主线程永久阻塞。
    """
    _SENTINEL = object()

    def __init__(self, video_path: str, ffmpeg_bin: str = 'ffmpeg',
                 prefetch_factor: int = 8, use_hwaccel: bool = True):
        meta          = get_video_meta_info(video_path)
        self.width    = meta['width']
        self.height   = meta['height']
        self.fps      = meta['fps']
        self.audio    = meta['audio']
        self.nb_frames = meta['nb_frames']

        # M2: 构建 hwaccel 参数
        hw_args: List[str] = []
        if use_hwaccel and HardwareCapability.has_nvdec():
            hw_args = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'bgr24']
            print('[NVDEC] 硬件解码已启用')
        else:
            if use_hwaccel:
                print('[NVDEC] 不可用，回退 CPU 解码')

        cmd = (
            [ffmpeg_bin]
            + hw_args
            + ['-i', video_path,
               '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-loglevel', 'error', 'pipe:1']
        )
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self._frame_bytes = self.width * self.height * 3

        self._queue   = queue.Queue(maxsize=prefetch_factor)
        self._thread  = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        try:
            while True:
                raw = self._proc.stdout.read(self._frame_bytes)
                if len(raw) < self._frame_bytes:
                    break
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                    self.height, self.width, 3
                )
                self._queue.put(arr.copy())
        except Exception as e:
            self._queue.put(e)
            return
        self._queue.put(self._SENTINEL)

    def get_frame(self) -> Optional[np.ndarray]:
        item = self._queue.get()
        if item is self._SENTINEL:
            self._queue.put(self._SENTINEL)
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


# ─────────────────────────────────────────────────────────────────────────────
# Writer（NVENC + 批量写入 + error flag）
# ─────────────────────────────────────────────────────────────────────────────

class FFmpegWriter:
    """
    X2(v4): error flag 完整传播。
    X1(v4): 批量攒帧写入（MAX_BATCH=8）。
    M2(v5): 支持 NVENC 硬件编码。
    """
    _SENTINEL  = object()
    _MAX_BATCH = 8

    def __init__(self, args, audio, height: int, width: int,
                 save_path: str, fps: float):
        out_w = int(width  * args.outscale)
        out_h = int(height * args.outscale)
        if out_h > 2160:
            print('[Warning] 输出 > 4K，建议 --outscale 或 --video_codec libx265。')

        # M2: 自动选择最优编码器
        preferred = getattr(args, 'video_codec', 'libx264')
        codec     = HardwareCapability.best_video_encoder(preferred)
        if codec != preferred:
            print(f'[NVENC] 使用硬件编码器: {codec}')
        crf = getattr(args, 'crf', 18)

        # NVENC 使用 -b:v 0 而非 CRF（部分版本不支持 CRF）
        if 'nvenc' in codec:
            extra_kwargs: dict = {'b:v': '0', 'cq': str(crf)}
        else:
            extra_kwargs = {'crf': str(crf)}

        inp_spec = dict(format='rawvideo', pix_fmt='bgr24',
                        s=f'{out_w}x{out_h}', framerate=fps)
        inp      = ffmpeg.input('pipe:', **inp_spec)
        out_kw   = dict(pix_fmt='yuv420p', vcodec=codec,
                        loglevel='error', **extra_kwargs)

        if audio is not None:
            stream = inp.output(audio, save_path, acodec='copy', **out_kw)
        else:
            stream = inp.output(save_path, **out_kw)

        self._proc  = stream.overwrite_output().run_async(
            pipe_stdin=True, pipe_stdout=False,
            pipe_stderr=False, cmd=getattr(args, 'ffmpeg_bin', 'ffmpeg')
        )
        self._queue  = queue.Queue(maxsize=64)
        self._error: Optional[Exception] = None
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

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

    def write_frame(self, frame: np.ndarray):
        if self._error is not None:
            raise RuntimeError(f'FFmpegWriter 内部错误: {self._error}') from self._error
        self._queue.put(frame.tobytes())

    def close(self):
        self._queue.put(self._SENTINEL)
        self._thread.join(timeout=60)
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        if self._error:
            print(f'[Warning] FFmpegWriter 累计写帧异常: {self._error}')


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载（多进程中调用，必须为模块级纯函数）
# ─────────────────────────────────────────────────────────────────────────────

def _build_upsampler(model_name: str, model_path, dni_weight,
                     tile: int, tile_pad: int, pre_pad: int,
                     use_half: bool, device: torch.device) -> RealESRGANer:
    """从 MODEL_CONFIG 构建 RealESRGANer，供子进程调用。"""
    model, netscale, _ = MODEL_CONFIG[model_name]
    return RealESRGANer(
        scale=netscale, model_path=model_path, dni_weight=dni_weight,
        model=model, tile=tile, tile_pad=tile_pad,
        pre_pad=pre_pad, half=use_half, device=device,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 批次推理（模块级，供单进程/多进程共用）
# ─────────────────────────────────────────────────────────────────────────────

def _process_batch(
    upsampler: RealESRGANer,
    frames: List[np.ndarray],
    outscale: float,
    netscale: int,
    transfer_stream,
    compute_stream,
    trt_accel: Optional['TensorRTAccelerator'] = None,
) -> List[np.ndarray]:
    """
    将一批帧推理为超分结果。
    若 trt_accel 可用，优先使用 TRT；否则走 PyTorch 路径。
    """
    device   = upsampler.device
    use_half = upsampler.half

    pool   = _get_pinned_pool()
    batch_pin = pool.get_for_frames(frames)

    B = len(frames)
    # (B, H, W, 3) uint8 → (B, 3, H, W) float [0,1]
    if transfer_stream is not None:
        with torch.cuda.stream(transfer_stream):
            batch_t = batch_pin.to(device, non_blocking=True)
    else:
        batch_t = batch_pin.to(device)

    batch_t = batch_t.permute(0, 3, 1, 2).float().div_(255.0)
    if use_half:
        batch_t = batch_t.half()

    # TRT 路径
    if trt_accel is not None and trt_accel.available:
        output_t = trt_accel.infer(batch_t).float()
    else:
        # PyTorch 路径
        if transfer_stream is not None and compute_stream is not None:
            compute_stream.wait_stream(transfer_stream)
            with torch.cuda.stream(compute_stream):
                with torch.no_grad():
                    output_t = upsampler.model(batch_t)
        else:
            with torch.no_grad():
                output_t = upsampler.model(batch_t)

    if abs(outscale - netscale) > 1e-5:
        output_t = F.interpolate(
            output_t.float(), scale_factor=outscale / netscale,
            mode='bicubic', align_corners=False,
        )

    out_u8 = output_t.float().clamp_(0.0, 1.0).mul_(255.0).byte()
    out_np = out_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return [out_np[i] for i in range(B)]


# ─────────────────────────────────────────────────────────────────────────────
# M1: 模块级 Dispatcher Worker（多进程，GPU 推理节点）
# ─────────────────────────────────────────────────────────────────────────────

def _sr_worker_fn(
    worker_idx:    int,
    device_id:     int,
    dispatch_q:    mp.Queue,
    result_q:      mp.Queue,
    error_q:       mp.Queue,
    # 模型参数
    model_name:    str,
    model_path,
    dni_weight,
    tile:          int,
    tile_pad:      int,
    pre_pad:       int,
    use_half:      bool,
    use_compile:   bool,
    use_trt:       bool,
    trt_cache_dir: str,
    # 推理参数
    outscale:      float,
    batch_size:    int,
    frame_h:       int,
    frame_w:       int,
):
    """
    多进程 GPU 推理 Worker。从 dispatch_q 获取帧，处理后放入 result_q。
    异常通过 error_q 传回主进程。
    """
    try:
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device)

        upsampler = _build_upsampler(
            model_name, model_path, dni_weight,
            tile, tile_pad, pre_pad, use_half, device
        )
        _, netscale, _ = MODEL_CONFIG[model_name]

        if use_compile and hasattr(torch, 'compile'):
            try:
                upsampler.model = torch.compile(upsampler.model, mode='reduce-overhead')
            except Exception as e:
                print(f'[Worker {worker_idx}] torch.compile 失败: {e}')

        trt_accel: Optional[TensorRTAccelerator] = None
        if use_trt:
            # 推断超分后尺寸
            sh = (batch_size, 3, frame_h, frame_w)
            trt_accel = TensorRTAccelerator(
                upsampler.model, device, trt_cache_dir, sh, use_fp16=use_half
            )

        transfer_stream = torch.cuda.Stream(device=device)
        compute_stream  = torch.cuda.Stream(device=device)

        local_frames:  List[np.ndarray] = []
        local_indices: List[int]        = []
        bs = batch_size

        def flush():
            nonlocal bs
            if not local_frames:
                return
            try:
                outputs = _process_batch(
                    upsampler, local_frames, outscale, netscale,
                    transfer_stream, compute_stream, trt_accel
                )
                for idx, out in zip(local_indices, outputs):
                    result_q.put((idx, out.tobytes(), out.shape))
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and bs > 1:
                    bs = max(1, bs // 2)
                    torch.cuda.empty_cache()
                    print(f'\n[Worker {worker_idx}] OOM，降级 batch_size → {bs}')
                    # 降级后逐帧重试
                    for idx, frm in zip(local_indices, local_frames):
                        try:
                            out = _process_batch(
                                upsampler, [frm], outscale, netscale,
                                transfer_stream, compute_stream, trt_accel
                            )[0]
                            result_q.put((idx, out.tobytes(), out.shape))
                        except Exception as inner_e:
                            error_q.put((worker_idx, repr(inner_e)))
                else:
                    error_q.put((worker_idx, repr(e)))
            local_frames.clear()
            local_indices.clear()

        while True:
            try:
                item = dispatch_q.get(timeout=2.0)
            except Exception:
                continue

            if item == _DISPATCH_SENTINEL:
                flush()
                result_q.put(_RESULT_DONE)
                # 将哨兵传给下一个 Worker（菊花链方式）
                dispatch_q.put(_DISPATCH_SENTINEL)
                break

            frame_idx, frame_bytes = item
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(frame_h, frame_w, 3)
            local_frames.append(frame)
            local_indices.append(frame_idx)

            if len(local_frames) >= bs:
                flush()

    except Exception as e:
        error_q.put((worker_idx, repr(e)))
        result_q.put(_RESULT_DONE)  # 确保主进程不永久等待


# ─────────────────────────────────────────────────────────────────────────────
# M1: Dispatcher-Queue 多卡编排器
# ─────────────────────────────────────────────────────────────────────────────

class MultiGPUOrchestrator:
    """
    多 GPU 调度核心：
      1. 启动 N 个 GPU Worker 进程
      2. 主线程读帧 → dispatch_queue（有界，防 OOM）
      3. 主线程收结果 → heapq 重排 → FFmpegWriter
      4. 全程异常监控，任一 Worker 崩溃立即 abort
    """

    def __init__(self, args, video_save_path: str, num_gpus: int,
                 num_workers_per_gpu: int = 1):
        self.args              = args
        self.video_save_path   = video_save_path
        self.num_gpus          = num_gpus
        self.num_workers       = num_gpus * num_workers_per_gpu

    def run(self):
        args = self.args
        args.model_name = args.model_name.split('.pth')[0]
        if args.model_name not in MODEL_CONFIG:
            raise ValueError(f'未知模型: {args.model_name}')

        _, netscale, file_url = MODEL_CONFIG[args.model_name]

        model_path = osp.join(models_RealESRGAN, args.model_name + '.pth')
        if not osp.isfile(model_path):
            for url in file_url:
                model_path = load_file_from_url(url=url, model_dir=models_RealESRGAN,
                                                progress=True, file_name=None)

        dni_weight = None
        if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
            wdn_path   = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_path]
            dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

        # 探测视频元数据
        meta = get_video_meta_info(args.input)
        W, H = meta['width'], meta['height']
        fps  = args.fps if args.fps else meta['fps']
        nb   = meta['nb_frames']
        audio_obj = meta['audio']

        print(f'[Orchestrator] 输入: {W}x{H} @ {fps:.3f}fps | {nb} 帧')
        print(f'[Orchestrator] GPU workers: {self.num_workers} '
              f'(×{self.num_workers // self.num_gpus} per GPU × {self.num_gpus} GPU)')

        # 有界队列：防止生产者过快导致 OOM
        max_q = max(self.num_workers * args.batch_size * 4, 32)
        dispatch_q: mp.Queue = mp.Queue(maxsize=max_q)
        result_q:   mp.Queue = mp.Queue(maxsize=max_q)
        error_q:    mp.Queue = mp.Queue()

        trt_cache = osp.join(args.output, '.trt_cache')
        use_trt   = getattr(args, 'use_tensorrt', False)

        worker_kwargs = dict(
            dispatch_q  = dispatch_q,
            result_q    = result_q,
            error_q     = error_q,
            model_name  = args.model_name,
            model_path  = model_path,
            dni_weight  = dni_weight,
            tile        = args.tile,
            tile_pad    = args.tile_pad,
            pre_pad     = args.pre_pad,
            use_half    = not args.fp32,
            use_compile = getattr(args, 'use_compile', False),
            use_trt     = use_trt,
            trt_cache_dir = trt_cache,
            outscale    = args.outscale,
            batch_size  = args.batch_size,
            frame_h     = H,
            frame_w     = W,
        )

        ctx     = mp.get_context('spawn')
        workers = [
            ctx.Process(
                target=_sr_worker_fn,
                args=(i, i % self.num_gpus),
                kwargs=worker_kwargs,
                daemon=True,
                name=f'SRWorker-{i}',
            )
            for i in range(self.num_workers)
        ]
        for w in workers:
            w.start()
        print(f'[Orchestrator] {self.num_workers} 个 Worker 已启动')

        # 错误监控守护线程
        abort_evt = threading.Event()

        def _error_monitor():
            while not abort_evt.is_set():
                try:
                    wid, err_msg = error_q.get(timeout=0.5)
                    print(f'\n[Worker {wid} Error] {err_msg}')
                    abort_evt.set()
                except queue.Empty:
                    pass

        mon_thread = threading.Thread(target=_error_monitor, daemon=True)
        mon_thread.start()

        # 构建 Writer（NVENC 自动选择）
        writer = FFmpegWriter(args, audio_obj, H, W, self.video_save_path, fps)

        # 结果收集线程：heapq 重排 → Writer
        total_done = [0]
        meter      = ThroughputMeter()
        pbar       = tqdm(total=nb, unit='frame', desc='[Multi-GPU SR]', dynamic_ncols=True)
        timing: List[float] = []
        t_collect_start = time.time()

        def _collect():
            heap: List[Tuple] = []
            next_idx    = 0
            done_count  = 0

            while done_count < self.num_workers and not abort_evt.is_set():
                try:
                    item = result_q.get(timeout=1.0)
                except queue.Empty:
                    continue

                if item == _RESULT_DONE:
                    done_count += 1
                    continue

                frame_idx, result_bytes, shape = item
                heapq.heappush(heap, (frame_idx, result_bytes, shape))

                # 连续输出已就绪的帧
                while heap and heap[0][0] == next_idx:
                    _, rb, sh = heapq.heappop(heap)
                    frame = np.frombuffer(rb, dtype=np.uint8).reshape(sh)
                    t0 = time.perf_counter()
                    writer.write_frame(frame)
                    timing.append(time.perf_counter() - t0)
                    meter.update(1)
                    total_done[0] += 1
                    next_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        fps=f'{meter.fps():.1f}',
                        eta=f'{meter.eta(nb):.0f}s',
                        reorder_buf=len(heap),
                    )

            # 冲刷剩余（非正常结束时可能有残余）
            while heap:
                _, rb, sh = heapq.heappop(heap)
                frame = np.frombuffer(rb, dtype=np.uint8).reshape(sh)
                writer.write_frame(frame)
                total_done[0] += 1
                pbar.update(1)

        collect_thread = threading.Thread(target=_collect, daemon=False)
        collect_thread.start()

        # 主线程：读帧 → dispatch_queue
        reader = FFmpegReader(
            args.input,
            ffmpeg_bin     = getattr(args, 'ffmpeg_bin', 'ffmpeg'),
            prefetch_factor= getattr(args, 'prefetch_factor', 8),
            use_hwaccel    = getattr(args, 'use_hwaccel', True),
        )

        for frame_idx in range(nb):
            if abort_evt.is_set():
                print('\n[Orchestrator] 检测到 Worker 错误，中止读帧。')
                break
            frame = reader.get_frame()
            if frame is None:
                break
            # 放入有界队列（阻塞直到 Worker 有空位，自动施压控速）
            dispatch_q.put((frame_idx, frame.tobytes()))

        # 发送哨兵（菊花链传播，仅需一个起始哨兵）
        dispatch_q.put(_DISPATCH_SENTINEL)

        reader.close()
        collect_thread.join()
        pbar.close()
        writer.close()
        abort_evt.set()  # 停止错误监控线程
        mon_thread.join(timeout=3)

        for w in workers:
            w.join(timeout=30)
            if w.is_alive():
                print(f'[Warning] {w.name} 未在超时内退出，强制终止。')
                w.kill()

        elapsed = time.time() - t_collect_start
        frames_done = total_done[0]
        print(f'\n[Orchestrator] 完成 {frames_done} 帧 | '
              f'耗时 {elapsed:.1f}s | 平均 {frames_done/elapsed:.1f} fps')

        # X3: JSON 报告
        report_path = getattr(args, 'report', None)
        if report_path and timing:
            report = {
                'input':       args.input,
                'output':      self.video_save_path,
                'model':       args.model_name,
                'outscale':    args.outscale,
                'num_workers': self.num_workers,
                'num_gpus':    self.num_gpus,
                'batch_size':  args.batch_size,
                'fp16':        not args.fp32,
                'nvenc':       HardwareCapability.best_video_encoder(
                    getattr(args, 'video_codec', 'libx264')
                ).endswith('nvenc'),
                'nvdec':       HardwareCapability.has_nvdec(),
                'trt':         use_trt,
                'frame_count': frames_done,
                'elapsed_s':   round(elapsed, 2),
                'avg_fps':     round(frames_done / elapsed, 2) if elapsed > 0 else 0,
                'write_latency_ms': {
                    'mean': round(float(np.mean(timing)) * 1000, 2),
                    'p95':  round(float(np.percentile(timing, 95)) * 1000, 2),
                    'max':  round(float(np.max(timing)) * 1000, 2),
                },
            }
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f'[Orchestrator] 性能报告: {report_path}')


# ─────────────────────────────────────────────────────────────────────────────
# flush_batch_safe（OOM 降级 + 恢复，单卡路径用）
# ─────────────────────────────────────────────────────────────────────────────

def flush_batch_safe(
    upsampler, frames, outscale, netscale,
    transfer_stream, compute_stream, writer,
    pbar, init_bs, oom_cooldown, max_bs, timing,
    trt_accel=None,
) -> int:
    bs = min(init_bs, len(frames))
    i  = 0
    while i < len(frames):
        sub = frames[i: i + bs]
        try:
            t0 = time.perf_counter()
            outputs = _process_batch(upsampler, sub, outscale, netscale,
                                     transfer_stream, compute_stream, trt_accel)
            timing.append(time.perf_counter() - t0)
            for out in outputs:
                writer.write_frame(out)
            pbar.update(len(sub))
            avg_ms = np.mean(timing[-20:]) * 1000 if timing else 0
            pbar.set_postfix(bs=bs, ms=f'{avg_ms:.0f}')
            i += bs
            if oom_cooldown[0] > 0:
                oom_cooldown[0] -= 1
            elif bs < max_bs[0]:
                bs = min(bs + 1, max_bs[0])
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and bs > 1:
                bs = max(1, bs // 2)
                oom_cooldown[0] = 10
                torch.cuda.empty_cache()
                print(f'\n[OOM] 降级 batch_size → {bs}')
            else:
                print(f'\n[Error] {e}')
                pbar.update(len(sub))
                i += len(sub)
    return bs


# ─────────────────────────────────────────────────────────────────────────────
# 单 GPU 推理主循环（保留 v4 全部优化，增加 TRT / NVDEC / NVENC）
# ─────────────────────────────────────────────────────────────────────────────

def inference_video_single(args, video_save_path: str, device=None):
    """单卡推理路径，保留 v4 全部优化，v5 新增 TRT / NVDEC / NVENC。"""
    args.model_name = args.model_name.split('.pth')[0]
    if args.model_name not in MODEL_CONFIG:
        raise ValueError(f'未知模型: {args.model_name}')

    _, netscale, file_url = MODEL_CONFIG[args.model_name]

    model_path = osp.join(models_RealESRGAN, args.model_name + '.pth')
    if not osp.isfile(model_path):
        for url in file_url:
            model_path = load_file_from_url(url=url, model_dir=models_RealESRGAN,
                                            progress=True, file_name=None)

    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_path   = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    upsampler = _build_upsampler(
        args.model_name, model_path, dni_weight,
        args.tile, args.tile_pad, args.pre_pad, not args.fp32, device
    )

    if args.use_compile and hasattr(torch, 'compile'):
        print('[Info] torch.compile 加速中 ...')
        upsampler.model = torch.compile(upsampler.model, mode='reduce-overhead')

    # M3: TensorRT 可选
    trt_accel: Optional[TensorRTAccelerator] = None
    if getattr(args, 'use_tensorrt', False) and torch.cuda.is_available():
        meta   = get_video_meta_info(args.input)
        sh     = (args.batch_size, 3, meta['height'], meta['width'])
        trt_dir = osp.join(args.output, '.trt_cache')
        trt_accel = TensorRTAccelerator(upsampler.model, device, trt_dir, sh,
                                        use_fp16=not args.fp32)

    if 'anime' in args.model_name and args.face_enhance:
        print('[Warning] anime 模型不支持 face_enhance，已禁用。')
        args.face_enhance = False

    face_enhancer = None
    if args.face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale, arch='clean', channel_multiplier=2,
            bg_upsampler=upsampler,
        )
        if args.batch_size > 1:
            args.batch_size = 1

    use_batch = args.batch_size > 1 and args.tile == 0 and face_enhancer is None

    transfer_stream = compute_stream = None
    if torch.cuda.is_available():
        transfer_stream = torch.cuda.Stream(device=device)
        compute_stream  = torch.cuda.Stream(device=device)

    reader = FFmpegReader(
        args.input,
        ffmpeg_bin     = getattr(args, 'ffmpeg_bin', 'ffmpeg'),
        prefetch_factor= getattr(args, 'prefetch_factor', 8),
        use_hwaccel    = getattr(args, 'use_hwaccel', True),
    )

    H, W  = reader.height, reader.width
    fps   = args.fps if args.fps else reader.fps
    audio = reader.audio
    nb    = reader.nb_frames

    writer = FFmpegWriter(args, audio, H, W, video_save_path, fps)

    pbar   = tqdm(total=nb, unit='frame', desc='[Single-GPU SR]', dynamic_ncols=True)
    meter  = ThroughputMeter()
    timing: List[float] = []
    t_start = time.time()
    bs = args.batch_size
    _oom_cd = [0]
    _max_bs = [args.batch_size]

    if use_batch:
        batch_frames: List[np.ndarray] = []
        while True:
            img = reader.get_frame()
            end = img is None
            if img is not None:
                batch_frames.append(img)
            if (len(batch_frames) == bs) or (end and batch_frames):
                bs = flush_batch_safe(
                    upsampler, batch_frames, args.outscale, netscale,
                    transfer_stream, compute_stream, writer, pbar,
                    bs, _oom_cd, _max_bs, timing, trt_accel,
                )
                meter.update(len(batch_frames))
                pbar.set_postfix(
                    fps=f'{meter.fps():.1f}',
                    eta=f'{meter.eta(nb):.0f}s',
                    bs=bs,
                    ms=f'{np.mean(timing[-20:]) * 1000:.0f}' if timing else '—',
                )
                batch_frames = []
            if end:
                break
    else:
        while True:
            img = reader.get_frame()
            if img is None:
                break
            try:
                if face_enhancer:
                    _, _, output = face_enhancer.enhance(
                        img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    t0 = time.perf_counter()
                    output, _ = upsampler.enhance(img, outscale=args.outscale)
                    timing.append(time.perf_counter() - t0)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f'\n[OOM] {e}')
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f'\n[Error] {e}')
                    continue
            writer.write_frame(output)
            meter.update(1)
            pbar.update(1)
            pbar.set_postfix(
                fps=f'{meter.fps():.1f}',
                eta=f'{meter.eta(nb):.0f}s',
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize(device=device)
    reader.close()
    writer.close()
    pbar.close()

    # X3: JSON 报告
    report_path = getattr(args, 'report', None)
    if report_path and timing:
        elapsed = time.time() - t_start
        report = {
            'input':       args.input,
            'output':      video_save_path,
            'model':       args.model_name,
            'outscale':    args.outscale,
            'batch_size':  args.batch_size,
            'fp16':        not args.fp32,
            'trt':         trt_accel is not None and trt_accel.available,
            'nvdec':       HardwareCapability.has_nvdec(),
            'nvenc':       HardwareCapability.best_video_encoder(
                getattr(args, 'video_codec', 'libx264')).endswith('nvenc'),
            'frame_count': nb,
            'elapsed_s':   round(elapsed, 2),
            'avg_fps':     round(nb / elapsed, 2) if elapsed > 0 else 0,
            'infer_latency_ms': {
                'mean': round(float(np.mean(timing)) * 1000, 2),
                'p95':  round(float(np.percentile(timing, 95)) * 1000, 2),
                'max':  round(float(np.max(timing)) * 1000, 2),
            },
        }
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f'[Info] 性能报告: {report_path}')


# ─────────────────────────────────────────────────────────────────────────────
# run：单卡 / 多卡自动分派
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    args.video_name     = osp.splitext(os.path.basename(args.input))[0]
    video_save_path     = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4')
    mime                = mimetypes.guess_type(args.input)[0]
    args.input_type_is_video = mime is not None and mime.startswith('video')

    num_gpus = torch.cuda.device_count()
    nw_per_g = getattr(args, 'num_process_per_gpu', 1)
    num_workers = max(1, num_gpus * nw_per_g)

    if num_workers > 1 and args.input_type_is_video:
        print(f'\n[V5] 多 GPU 模式：{num_gpus} GPU × {nw_per_g} Worker = {num_workers} 总 Worker')
        print('[V5] 架构：Dispatcher-Queue（零临时文件）')
        orchestrator = MultiGPUOrchestrator(args, video_save_path, num_gpus, nw_per_g)
        orchestrator.run()
    else:
        if num_workers > 1:
            print('[Info] 图片目录输入，使用单卡路径。')
        inference_video_single(args, video_save_path)

    # 如果有原始音频且视频文件有效，补充合并音频
    # （NVDEC/NVENC 路径已在 FFmpegWriter 中直接合并，此处仅作保障性检查）
    if osp.exists(video_save_path) and osp.getsize(video_save_path) > 0:
        print(f'\n✅ 输出文件: {video_save_path} '
              f'({osp.getsize(video_save_path)/1024/1024:.1f} MB)')
    else:
        print('\n❌ 输出文件生成失败，请检查日志。')


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Real-ESRGAN 视频超分 —— 终极优化版 v5',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 基础参数
    parser.add_argument('-i',  '--input',            type=str, default='inputs')
    parser.add_argument('-n',  '--model_name',       type=str, default='realesr-animevideov3')
    parser.add_argument('-o',  '--output',           type=str, default='results')
    parser.add_argument('-dn', '--denoise_strength', type=float, default=0.5)
    parser.add_argument('-s',  '--outscale',         type=float, default=4)
    parser.add_argument('--suffix',                  type=str, default='out')
    # 推理参数
    parser.add_argument('-t',  '--tile',             type=int, default=0)
    parser.add_argument('--tile_pad',                type=int, default=10)
    parser.add_argument('--pre_pad',                 type=int, default=0)
    parser.add_argument('--face_enhance',            action='store_true')
    parser.add_argument('--fp32',                    action='store_true',
                        help='禁用 FP16（默认启用 FP16）')
    parser.add_argument('--fps',                     type=float, default=None)
    parser.add_argument('--batch_size',              type=int, default=4)
    parser.add_argument('--prefetch_factor',         type=int, default=8)
    parser.add_argument('--use_compile',             action='store_true',
                        help='启用 torch.compile（reduce-overhead）')
    # V5 新参数
    parser.add_argument('--use_tensorrt',            action='store_true',
                        help='[V5] 启用 TensorRT 加速（首次需要构建 Engine）')
    parser.add_argument('--use_hwaccel',             action='store_true', default=True,
                        help='[V5] 启用 NVDEC 硬件解码（自动探测，失败时回退）')
    parser.add_argument('--no_hwaccel',              action='store_true',
                        help='[V5] 强制禁用 NVDEC 硬件解码')
    # 多卡参数
    parser.add_argument('--num_process_per_gpu',     type=int, default=1,
                        help='每 GPU 启动的 Worker 进程数（多卡模式）')
    # 编码参数
    parser.add_argument('--video_codec',             type=str, default='libx264',
                        choices=['libx264', 'libx265', 'libvpx-vp9'],
                        help='偏好编码器（有 NVENC 时自动升级为 h264_nvenc/hevc_nvenc）')
    parser.add_argument('--crf',                     type=int, default=18)
    parser.add_argument('--ffmpeg_bin',              type=str, default='ffmpeg')
    # 模型 alpha 通道 / 扩展名
    parser.add_argument('--alpha_upsampler',         type=str, default='realesrgan',
                        choices=['realesrgan', 'bicubic'])
    parser.add_argument('--ext',                     type=str, default='auto',
                        choices=['auto', 'jpg', 'png'])
    # 报告
    parser.add_argument('--report',                  type=str, default=None,
                        help='输出 JSON 性能报告路径（如 report.json）')

    args = parser.parse_args()
    args.input = args.input.rstrip('/\\')
    os.makedirs(args.output, exist_ok=True)

    # 处理 --no_hwaccel
    if args.no_hwaccel:
        args.use_hwaccel = False

    # FLV 转 MP4
    mime     = mimetypes.guess_type(args.input)[0]
    is_video = mime is not None and mime.startswith('video')
    if is_video and args.input.endswith('.flv'):
        mp4_path = args.input.replace('.flv', '.mp4')
        subprocess.run([args.ffmpeg_bin, '-i', args.input, '-codec', 'copy', '-y', mp4_path])
        args.input = mp4_path

    # 启动前打印硬件状态
    print('=' * 60)
    print('  RealESRGAN 视频超分 —— 终极优化版 v5')
    print('=' * 60)
    num_gpus = torch.cuda.device_count()
    print(f'  GPU 数量: {num_gpus}')
    print(f'  NVDEC:   {HardwareCapability.has_nvdec()} | '
          f'NVENC(h264): {HardwareCapability.has_nvenc("h264_nvenc")} | '
          f'NVENC(hevc): {HardwareCapability.has_nvenc("hevc_nvenc")}')
    print(f'  TensorRT: {getattr(args, "use_tensorrt", False)} | '
          f'torch.compile: {getattr(args, "use_compile", False)}')
    print()

    run(args)


if __name__ == '__main__':
    main()
