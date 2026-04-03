"""
RealESRGAN 视频超分处理脚本 —— 终极优化版 v6.2（单卡版）
==========================================================
基于 Real-ESRGAN 的视频超分脚本，面向单 GPU 生产环境的最高性能实现。
v6.1 在 v6 基础上反向移植 IFRNet v5 的 OOM 级联保护机制：
  - in_oom_cascade 标志：首次 OOM 永久更新 max_batch_size 天花板，
    级联 OOM 不重复修改上限，避免多次惩罚；
  - _estimate_safe_batch_size：深度清理后按实测空闲显存动态估算恢复值；
  - batch_size=1 仍 OOM 时执行深度清理（synchronize + empty_cache + dynamo.reset）。

【最终功能特性】
  推理加速：
    · FP16 半精度推理（默认开启，--no-fp16 可禁用）
    · torch.compile 可选加速（--use-compile, mode=reduce-overhead）
    · TensorRT 可选加速（--use-tensorrt；FIX-3 修复后全程 GPU 内存，真正有效）
      - TRT 8.x / 10.x 双 API 兼容（FIX-TRT10）
      - Engine 缓存于 .trt_cache/，GPU 不兼容时自动重建
    · GFPGAN TensorRT 加速（v6.2 新增，由 --gfpgan-trt 独立控制）：
      - weight 在 ONNX 导出时作常量烘焙（trace-time constant）；weight 变化自动重建 Engine
      - 静态 batch=1 Engine + infer() 内部循环，--gfpgan-batch-size 可任意设置
      - FusedLeakyReLU / upfirdn2d monkey-patch 确保导出 ONNX 无自定义算子
      - 注意：不可与 --use-tensorrt 同时使用（Myelin 内存分配器冲突）
    · cuDNN benchmark 自动最优卷积算法（FIX-1）
    · OOM 级联保护（backport from IFRNet v5）：
        首次 OOM 永久降低 max_batch_size 天花板（已证明不可用的 batch_size）
        级联 OOM 不再修改上限（内存仍脏，惩罚无意义）
        batch_size=1 仍 OOM → 深度清理 + 按剩余显存动态恢复

  I/O 加速：
    · NVDEC 硬件解码（--use-hwaccel；自动探测，失败回退 CPU）
    · NVENC 硬件编码（有 h264_nvenc/hevc_nvenc 时自动升级）
    · 异步帧预取（FFmpegReader 后台线程 + PinnedBufferPool，默认 prefetch=16）
    · 批量推理（默认 batch_size=8，充分利用 T4/A10 显存）
    · 异步 D2H（pinned output buffer + non_blocking=True，FIX-2）
    · 批量写帧（FFmpegWriter 攒 8 帧一次 write）

  face_enhance 路径（v6 重点优化）：
    · 批量 GFPGAN 推理：将一批所有人脸 crops 堆叠为单次前向，充分利用 TensorCore
    · 原始帧检测：在低分辨率帧（720p）上运行 RetinaFace，比 SR 帧快 4×
    · 无人脸帧跳过：检测为空时直接跳过 GFPGAN，零额外开销
    · CPU-GPU 流水线（[FIX-PIPELINE]）：
        detect(N) 后台线程与 SR(N) 主线程 GPU 并行
        paste(N) 后台线程与 SR(N+1)+GFPGAN(N+1) 并行
    · GFPGAN FP16：torch.autocast 包裹推理，利用 Tensor Core（FIX-8）
    · GFPGAN OOM 保护：gfpgan_batch_size 子批量拆分，OOM 时自动降级（FIX-OOM）
    · facexlib 0.3.0 API 修复：GFPGANer(upscale=outscale)，坐标系由 facexlib 内部统一管理

  可观测性：
    · tqdm 进度条：fps / eta / batch_size / ms 实时显示
    · JSON 性能报告（--report；含 infer_latency_ms p95/mean/max 等）

【命令行使用示例】
  # 基础超分（2× 放大，FP16 + NVDEC/NVENC 自动启用）
  python inference_realesrgan_video_v6_1_single.py \\
      -i input.mp4 -o results/ -s 2

  # 4× 放大，动画视频专用模型
  python inference_realesrgan_video_v6_1_single.py \\
      -i input.mp4 -o results/ -n realesr-animevideov3 -s 4

  # 开启 face_enhance，指定 GFPGAN 版本和融合权重
  python inference_realesrgan_video_v6_1_single.py \\
      -i input.mp4 -o results/ -s 2 \\
      --face_enhance --gfpgan_model 1.4 --gfpgan_weight 0.5

  # 大批量 + TensorRT 加速（首次构建 Engine）
  python inference_realesrgan_video_v6_1_single.py \\
      -i input.mp4 -o results/ -s 2 \\
      --batch-size 16 --use-tensorrt

  # tile 模式（显存不足时切块处理）
  python inference_realesrgan_video_v6_1_single.py \\
      -i input.mp4 -o results/ -s 2 -t 512 --tile_pad 10

  # 直接输出到指定文件（非目录），FP32 模式
  python inference_realesrgan_video_v6_1_single.py \\
      -i input.mp4 -o output.mp4 -s 2 --no-fp16

  # 输出 JSON 性能报告
  python inference_realesrgan_video_v6_1_single.py \\
      -i input.mp4 -o results/ -s 2 --report report.json

【关键参数说明】
  -i / --input         输入视频路径
  -o / --output        输出目录或输出文件路径（自动识别）
  -n / --model_name    模型名称（默认 realesr-animevideov3）
  -s / --outscale      放大倍数（如 2, 4）
  --face-enhance       启用 GFPGAN 人脸增强
  --gfpgan-model       GFPGAN 版本（1.3 / 1.4 / RestoreFormer，默认 1.4）
  --gfpgan-weight      GFPGAN 融合权重（0.0~1.0，默认 0.5）
  --gfpgan-batch-size  单次 GFPGAN 前向最多处理的人脸数（防 OOM，默认 12）
  --no-fp16            禁用 FP16（默认开启 FP16）
  --batch-size         批处理大小（默认 8，T4 16G 建议 8~12）
  --prefetch-factor    读帧预取队列深度（默认 16）
  --tile / -t          切块大小（0=不切块，VRAM 不足时设 512）
  --use-compile        启用 torch.compile
  --use-tensorrt       启用 TensorRT 加速
  --no-hwaccel         强制禁用 NVDEC
  --video-codec        偏好编码器（libx264/libx265，NVENC 可用时自动升级）
  --crf                编码质量（默认 23）
  --report             JSON 性能报告输出路径

【注意事项】
  · TRT Engine 缓存于 .trt_cache/，TRT 版本升级后需手动删除重建
  · face_enhance 需要安装 gfpgan 和 facexlib 库
  · realesr-general-x4v3 需同时下载 wdn 模型（denoise_strength 支持）
  · base_dir / models_RealESRGAN 路径由脚本顶部常量决定，部署前请修改

【版本演进历史（简要）】
  v5+（v5_single）：
    FIX-TRT10（TRT 10.x API 兼容）、FIX-1~8（cudnn/D2H/TRT流/face路径修复）
    批量 GFPGAN 推理、原始帧检测、GFPGAN FP16、TRT 专用 CUDA Stream
  v6（v6_single，当前版本）：
    [FIX-PIPE]   face_enhance 输出帧尺寸错误修复（GFPGANer upscale=outscale）
    [FIX-BUG]    FIX-7 实际未生效修复（真正在原始帧上检测）
    [FIX-PIPELINE] CPU-GPU 流水线并行（detect/paste 后台线程）
    [v5++-BATCH] 批量 GFPGAN sub_bs 跨批次 OOM 降级持久化
"""

from __future__ import annotations
import multiprocessing as mp
import queue
import argparse
import contextlib          # [FIX-7/8] GFPGAN FP16 autocast 所需
import concurrent.futures  # [FIX-PIPELINE] 流水线并行
import json
import mimetypes
import numpy as np
import os
import subprocess
import sys
import threading
import time
from collections import deque
from fractions import Fraction
from typing import Dict, List, Optional, Tuple
import re
import torch
import torch.nn.functional as F

os.environ.setdefault("PYTORCH_NO_NVML", "1")   # suppress harmless NVML warning


class _NVMLFilter:
    _pat = re.compile(r'NVML_SUCCESS|INTERNAL ASSERT FAILED.*CUDACachingAllocator')
    def __init__(self, s): self._s = s
    def write(self, m):
        if not self._pat.search(m): self._s.write(m)
    def flush(self): self._s.flush()
    def __getattr__(self, a): return getattr(self._s, a)
sys.stderr = _NVMLFilter(sys.stderr)

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from os import path as osp
from tqdm import tqdm

# [v5++] 批量GFPGAN推理所需工具
try:
    from basicsr.utils import img2tensor, tensor2img
    from torchvision.transforms.functional import normalize as _tv_normalize
    _HAS_BATCHGFPGAN = True
except ImportError:
    _HAS_BATCHGFPGAN = False

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

# 以本脚本所在目录（external/Real-ESRGAN/）为基准，向上两级到项目根
# 目录结构假设：<project_root>/external/Real-ESRGAN/inference_realesrgan_video_v6_single.py
# base_dir = str(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
_SCRIPT_DIR       = osp.dirname(osp.abspath(__file__))
base_dir          = osp.dirname(osp.dirname(_SCRIPT_DIR))
models_RealESRGAN = osp.join(base_dir, 'models_RealESRGAN')

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
        """[FIX-NDV] 两阶段真实探测：先软件编码 H.264 流，再用 NVDEC 实际解码。
        避免 lavfi 测试源在某些非官方 FFmpeg build 中误报"可用"。
        编码阶段严格使用纯软件路径（无 -hwaccel），确保与解码探测完全解耦。
        旧代码将 -hwaccel cuda 误放入编码命令（解码加速标志对编码无意义），
        在部分系统上导致 FFmpeg 输出额外警告或提前返回非零码。
        """
        try:
            # Step-1: 软件编码 lavfi 源 → H.264 raw 流（不带任何 hwaccel）
            enc_cmd = [
                'ffmpeg', '-f', 'lavfi',
                '-i', 'testsrc=size=64x64:duration=0.04:rate=25',
                '-vcodec', 'libx264', '-f', 'h264', 'pipe:1', '-loglevel', 'error',
            ]
            enc = subprocess.run(enc_cmd, capture_output=True, timeout=10)
            if enc.returncode != 0 or not enc.stdout:
                return False
            # Step-2: 用 NVDEC 真实解码上一步产出的 H.264 流 → 验证硬件解码可用
            dec_cmd = [
                'ffmpeg', '-hwaccel', 'cuda',
                '-f', 'h264', '-i', 'pipe:0',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-frames:v', '1', 'pipe:1',
                '-loglevel', 'error',
            ]
            dec = subprocess.run(dec_cmd, input=enc.stdout,
                                 capture_output=True, timeout=10)
            return dec.returncode == 0 and len(dec.stdout) > 0
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
# [FIX-2] 新增 output buffer，支持异步 D2H
# ─────────────────────────────────────────────────────────────────────────────

_thread_local = threading.local()


class PinnedBufferPool:
    """
    预分配 pinned CPU buffer 并复用，避免每批 H2D 前 pin_memory 的 malloc 开销。
    [FIX-2] 新增 _out_buf，用于异步 D2H 输出，避免 .cpu() 引发隐式同步。
    """

    def __init__(self):
        self._buf:     Optional[torch.Tensor] = None
        self._out_buf: Optional[torch.Tensor] = None

    def get_for_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        # [FIX-4] 去掉原来在 _read_loop 里的 arr.copy()，这里 np.stack 已经复制
        arr    = np.stack(frames, axis=0)
        src    = torch.from_numpy(arr)
        n_elem = src.numel()
        if self._buf is None or self._buf.numel() < n_elem:
            self._buf = torch.empty(n_elem, dtype=torch.uint8).pin_memory()
        dst = self._buf[:n_elem].view_as(src)
        dst.copy_(src)
        return dst

    def get_output_buf(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """[FIX-2] 返回与 shape/dtype 匹配的 pinned 输出 buffer，按需扩容。"""
        n_elem = 1
        for s in shape:
            n_elem *= s
        if (self._out_buf is None
                or self._out_buf.dtype != dtype
                or self._out_buf.numel() < n_elem):
            self._out_buf = torch.empty(n_elem, dtype=dtype).pin_memory()
        return self._out_buf[:n_elem].view(shape)


def _get_pinned_pool() -> PinnedBufferPool:
    if not hasattr(_thread_local, 'pool'):
        _thread_local.pool = PinnedBufferPool()
    return _thread_local.pool


# ─────────────────────────────────────────────────────────────────────────────
# M3: TensorRT 可选加速封装
# [FIX-3] infer() 全程 GPU 内存，去掉 GPU→CPU→GPU 来回搬运
# ─────────────────────────────────────────────────────────────────────────────

# ── TRT 进程级 Logger 单例 ─────────────────────────────────────────────────────
# TRT 10.x 进程内只允许一个全局 Logger（nvinfer1::getLogger()）。
# SR Engine 和 GFPGAN Engine 分别 new 出不同 Logger 实例时，TRT 忽略后者，
# 并在某些版本中导致内部 Builder/Runtime 内存池归属不一致
# → TRT kernel 写出已释放地址 → cudaErrorIllegalAddress。
_TRT_LOGGER = None

def _get_trt_logger():
    """返回进程级 TRT Logger 单例；首次调用时创建并缓存。"""
    global _TRT_LOGGER
    if _TRT_LOGGER is None:
        try:
            import tensorrt as _trt_mod
            _TRT_LOGGER = _trt_mod.Logger(_trt_mod.Logger.WARNING)
        except ImportError:
            pass
    return _TRT_LOGGER


class TensorRTAccelerator:
    """
    将 RealESRGAN 模型导出 ONNX 后编译 TRT Engine (FP16, 静态形状)。
    要求：pip install tensorrt pycuda onnx onnxruntime-gpu
    首次构建会缓存 .trt 文件，后续直接加载。

    [FIX-3] infer() 改为直接使用 Tensor.data_ptr()，全程 GPU 内存，
    不再做 D2H + H2D 的无效往返搬运。
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
        # [FIX-TRT-STREAM] 专用非默认 Stream，懒初始化（在 CUDA 上下文建立后）
        # current_stream() 在主循环上下文中返回 default stream，TRT 在 default stream 上
        # 执行 enqueueV3 时会自动插入额外的 cudaStreamSynchronize()（即日志 W 警告根因），
        # 造成不必要的全局阻塞。专用 Stream 可消除此开销，同时仍保证同一 CUDA context。
        self._trt_stream: Optional[torch.cuda.Stream] = None

        try:
            import tensorrt as trt
            # NOTE: Do NOT import pycuda.autoinit here.
            # pycuda.autoinit creates its own CUDA context which conflicts with
            # PyTorch's CUDA context, causing CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH
            # and cascading "invalid resource handle" errors on every inference call.
            # We use torch.cuda.current_stream().cuda_stream in infer() instead.
            self._trt  = trt
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
            try:
                self._load_engine(trt_path)
            except RuntimeError as _e:
                # _load_engine 在 GPU 不兼容或 TRT 版本升级时会：
                #   1. 打印警告信息
                #   2. 删除过期 .trt 缓存文件
                #   3. 抛出 RuntimeError
                # 此处捕获后自动重新导出 ONNX + 构建新 Engine，再次尝试加载。
                # 若二次加载仍失败，说明环境存在更深层问题，继续向上抛出。
                print(f'[TensorRT] 首次加载失败（{_e}），开始重新构建 Engine...')
                if not osp.exists(onnx_path):
                    self._export_onnx(model, onnx_path, input_shape)
                self._build_engine(onnx_path, trt_path, use_fp16)
                if osp.exists(trt_path):
                    self._load_engine(trt_path)   # 二次加载，失败则向上传播

    def _export_onnx(self, model, onnx_path, input_shape):
        model.eval()
        dummy = torch.randn(*input_shape, device=self.device)
        if self.use_fp16:
            dummy = dummy.half()
            model = model.half()
        with torch.no_grad():
            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=['input'], output_names=['output'],
                opset_version=18,   # torch now emits opset 18 internally; request it explicitly
                dynamic_axes=None,  # 静态形状，TRT 最优化
            )
        print(f'[TensorRT] ONNX 已导出: {onnx_path}')

    def _build_engine(self, onnx_path, trt_path, use_fp16):
        trt    = self._trt
        logger = _get_trt_logger()
        builder = trt.Builder(logger)

        # TRT 10.x 中 EXPLICIT_BATCH 已成为默认模式且该枚举已被废弃，
        # 需要兼容处理：有 EXPLICIT_BATCH 时使用旧方式，否则用 0（默认）。
        try:
            explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        except AttributeError:
            # TRT 10.x：explicit batch 是唯一模式，flag 参数已废弃，传 0 即可
            explicit_batch_flag = 0
        network = builder.create_network(explicit_batch_flag)
        parser  = trt.OnnxParser(network, logger)

        with open(onnx_path, 'rb') as f:
            # IMPORTANT: use parse_from_file, not parse(f.read()).
            # torch.onnx.export may write a separate .onnx.data sidecar for large
            # models. parse(bytes) only sees the main file and fails to find the
            # weights. parse_from_file lets the TRT parser resolve the sidecar
            # from the same directory automatically.
            if not parser.parse_from_file(onnx_path):
                for i in range(parser.num_errors):
                    print(f'  ONNX 解析错误: {parser.get_error(i)}')
                return

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
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
        trt     = self._trt
        logger = _get_trt_logger()
        runtime = trt.Runtime(logger)
        with open(trt_path, 'rb') as f:
            self._engine  = runtime.deserialize_cuda_engine(f.read())

        # deserialize_cuda_engine 在 GPU compute capability 不匹配（如 T4→A10）、
        # TRT 版本升级、文件损坏时会静默返回 None，此后调用任何方法均触发：
        #   AttributeError: 'NoneType' object has no attribute 'create_execution_context'
        # 检测到 None 时删除过期缓存文件并抛出异常，由 __init__ 捕获后重新构建 Engine。
        if self._engine is None:
            print(f'[TensorRT] Engine 反序列化失败（GPU 不兼容或 TRT 版本升级），'
                  f'删除过期缓存并重新构建: {trt_path}')
            try:
                os.remove(trt_path)
            except OSError:
                pass
            raise RuntimeError('[TensorRT] _load_engine: deserialize_cuda_engine returned None')

        self._context = self._engine.create_execution_context()

        # ── 区分 TRT 版本，预先解析 tensor 名称 / binding 信息 ──────────────────
        # TRT 10.x: 使用 num_io_tensors + get_tensor_name + get_tensor_mode
        # TRT 8.x : 使用 num_bindings + get_binding_shape（旧接口）
        self._use_new_api    = hasattr(self._engine, 'num_io_tensors')
        self._input_name     = None   # TRT 10.x 专用
        self._output_name    = None   # TRT 10.x 专用

        if self._use_new_api:
            # TRT 10.x：遍历所有 IO tensor，按模式区分输入/输出
            for i in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(i)
                mode = self._engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self._input_name = name
                elif mode == trt.TensorIOMode.OUTPUT:
                    self._output_name = name
            if self._input_name is None or self._output_name is None:
                raise RuntimeError(
                    '[TensorRT] 无法在 Engine 中找到有效的输入/输出 tensor，'
                    '请确认 ONNX 导出时设置了 input_names / output_names。'
                )
            print(f'[TensorRT] 使用新版 API (TRT 10.x)，'
                  f'输入: {self._input_name}，输出: {self._output_name}')
        else:
            # TRT 8.x：保持旧行为，binding 0 = 输入，binding 1 = 输出
            print('[TensorRT] 使用旧版 API (TRT 8.x)')

        self._trt_ok  = True
        print('[TensorRT] Engine 加载成功，已启用 TRT 推理')

    @property
    def available(self) -> bool:
        return self._trt_ok

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        [FIX-3] 全程 GPU 内存推理：直接把 Tensor.data_ptr() 传给 TRT。
        [BUGFIX] 处理静态 batch_size 引擎的最后一批不足 batch_size 的情况：
                 用最后一帧 padding 到引擎期望的 batch_size，推理后裁剪输出。
        """
        actual_B  = input_tensor.shape[0]
        engine_B  = self.input_shape[0]          # 引擎编译时的 batch_size

        # 如果实际帧数不足，用最后一帧 padding（仅静态 shape 引擎需要）
        if actual_B < engine_B:
            pad_cnt = engine_B - actual_B
            pad     = input_tensor[-1:].expand(pad_cnt, -1, -1, -1)
            input_tensor = torch.cat([input_tensor, pad], dim=0)

        # 确保输入连续，dtype 与 TRT engine 匹配
        inp = input_tensor.contiguous()

        # [FIX-TRT-STREAM] 使用专用非默认 Stream，而非 torch.cuda.current_stream()。
        # 在主批处理循环上下文中 current_stream() 返回 default stream，TRT 在其上
        # 执行 enqueueV3/v2 时会自动插入 cudaStreamSynchronize() 保证正确性，
        # 但这会导致全局阻塞并产生日志 W 警告。专用 Stream 消除此隐式同步。
        if self._trt_stream is None:
            self._trt_stream = torch.cuda.Stream(device=self.device)
        out_dtype = torch.float16 if self.use_fp16 else torch.float32

        if self._use_new_api:
            # ── TRT 10.x 新 API ──────────────────────────────────────────────
            # get_tensor_shape 返回输出 shape（静态 Engine 下即确定值）
            out_shape  = tuple(self._engine.get_tensor_shape(self._output_name))
            out_tensor = torch.empty(out_shape, dtype=out_dtype, device=self.device)

            # set_tensor_address 替代旧 bindings 列表，直接传 GPU 指针
            self._context.set_tensor_address(self._input_name,  inp.data_ptr())
            self._context.set_tensor_address(self._output_name, out_tensor.data_ptr())

            # [BUGFIX] 必须让 _trt_stream 等待 current_stream：
            # inp.contiguous() 在调用方的 default stream 上执行，若不插入此等待，
            # _trt_stream 可能在 contiguous() 完成前就读取 inp 数据 → 花屏。
            self._trt_stream.wait_stream(torch.cuda.current_stream(self.device))

            # execute_async_v3 不再接受 bindings 参数
            self._context.execute_async_v3(stream_handle=self._trt_stream.cuda_stream)
        else:
            # ── TRT 8.x 旧 API（向下兼容保留）──────────────────────────────
            out_shape  = tuple(self._engine.get_binding_shape(1))
            out_tensor = torch.empty(out_shape, dtype=out_dtype, device=self.device)
            self._context.execute_async_v2(
                bindings=[inp.data_ptr(), out_tensor.data_ptr()],
                stream_handle=self._trt_stream.cuda_stream,
            )

        # 等待 TRT Stream 完成，然后让调用方的 default stream 能安全使用结果
        self._trt_stream.synchronize()

        # 如果之前做了 padding，裁掉多余的输出帧
        if actual_B < engine_B:
            out_tensor = out_tensor[:actual_B]

        return out_tensor


# ─────────────────────────────────────────────────────────────────────────────
# CUDA Graph 加速器
# ─────────────────────────────────────────────────────────────────────────────

class CUDAGraphAccelerator:
    """
    CUDA Graph 加速器：捕获并重放模型前向图，消除 CPU kernel dispatch 开销。

    原理：
      首次遇到某一输入形状时，先做 3 次预热（填充 cudnn.benchmark 缓存选定
      最优 kernel），再通过 torch.cuda.CUDAGraph 录制完整推理，后续相同形状
      直接重放，CPU 无需逐一下发 CUDA kernel，吞吐提升约 10~25%。

    适用条件：
      · 不与 torch.compile / TRT 共用；优先级：TRT > compile > CUDA Graph。
      · 仅 CUDA 可用时生效；CPU 设备自动禁用（available=False）。
      · 每种 (B, C, H, W) 形状独立捕获一张图。末批或 OOM 降级后 batch_size
        变小时触发新图捕获，不影响已缓存的大批图，与 TRT 策略一致。

    实现要点：
      · 捕获在 compute_stream 上进行；replay 前通过 static_in.copy_(batch_t)
        更新数据（GPU→GPU，在 compute_stream 上执行），replay 后通过
        static_out.clone() 取出结果以防下一批覆盖。
      · 调用方负责：
          replay 前：compute_stream.wait_stream(transfer_stream)
          replay 后：default_stream.wait_stream(compute_stream)
        与普通 PyTorch 批处理路径完全对称。
    """

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 compute_stream: Optional[torch.cuda.Stream] = None):
        self.model          = model
        self.device         = device
        self.compute_stream = compute_stream
        self._graphs: dict  = {}   # shape_key → (graph, static_in, static_out)
        self.available      = torch.cuda.is_available()
        if not self.available:
            print('[CUDA Graph] GPU 不可用，CUDA Graph 已禁用')
        else:
            print('[CUDA Graph] 已启用（首帧推理时触发捕获，后续直接重放）')

    @property
    def num_cached(self) -> int:
        return len(self._graphs)

    def infer(self, batch_t: torch.Tensor) -> torch.Tensor:
        """
        推理入口：形状首次出现时执行预热+捕获，后续直接重放。
        调用方须在此之前完成 compute_stream.wait_stream(transfer_stream)，
        在此之后执行 default_stream.wait_stream(compute_stream)。
        """
        key = tuple(batch_t.shape)
        if key not in self._graphs:
            self._capture(batch_t, key)
        graph, static_in, static_out = self._graphs[key]
        s = self.compute_stream
        if s is not None:
            with torch.cuda.stream(s):
                static_in.copy_(batch_t)   # GPU→GPU，在 compute_stream 上执行
                graph.replay()
                result = static_out.clone()
        else:
            static_in.copy_(batch_t)
            graph.replay()
            result = static_out.clone()
        return result

    def infer_or_eager(self, batch_t: torch.Tensor,
                       eager_fn) -> torch.Tensor:
        """
        有缓存图时 replay，否则 eager fallback（不触发新图捕获）。
        用于末批等只出现一次的非标准形状，避免不必要的 capture 开销。
        首批（_graphs 为空）时仍触发 _capture，以保证主形状被图化。
        """
        key = tuple(batch_t.shape)
        if key in self._graphs:
            return self.infer(batch_t)
        if not self._graphs:
            # 真正的首批：捕获主形状图
            self._capture(batch_t, key)
            return self.infer(batch_t)
        # 图已存在但形状不匹配（末批等）→ eager
        return eager_fn(batch_t)

    def _capture(self, sample_input: torch.Tensor, key: tuple):
        """在 compute_stream 上预热 + 捕获 CUDAGraph。"""
        s = self.compute_stream or torch.cuda.current_stream(self.device)
        # 预热：3 次 eager 前向，让 cudnn.benchmark 选定最优 kernel
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.no_grad():
                    _ = self.model(sample_input)
        torch.cuda.synchronize(self.device)

        static_in = sample_input.clone()
        graph     = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            with torch.cuda.graph(graph, stream=s):
                with torch.no_grad():
                    static_out = self.model(static_in)
        torch.cuda.synchronize(self.device)

        self._graphs[key] = (graph, static_in, static_out)
        print(f'[CUDA Graph] 新形状 {key} 已捕获，当前缓存图数: {len(self._graphs)}')


# ─────────────────────────────────────────────────────────────────────────────
# [v6.2] GFPGANTRTAccelerator
# GFPGAN TensorRT 动态 Batch 加速封装
# ─────────────────────────────────────────────────────────────────────────────

class GFPGANTRTAccelerator:
    """
    GFPGAN TensorRT 加速封装（v6.2 新增）。

    与 TensorRTAccelerator（SR 专用，静态形状）的核心区别：

      · 动态 Batch Profile：仅 Batch 维度动态（B = 1 ~ max_batch_size），
        H=W=512 固定（GFPGAN 输入始终为 512×512 aligned crop）。
        TRT 针对 opt_b=min(6,max_bs) 预编译最优 kernel variant，
        实际 B 在 [1, max_batch_size] 内任意变化均高效。

      · gfpgan_weight 烘焙（trace-time constant）：
        ONNX 导出时 weight 通过闭包捕获为 Python float 常量直接内联入计算图，
        weight 变化时 cache tag 不同 → 自动触发重建，无需手动清缓存。
        与 SR 端"分辨率/batch_size 不同对应不同 Engine"机制完全一致。

      · FusedLeakyReLU / upfirdn2d monkey-patch（_onnx_compat_patch）：
        basicsr 的 CUDA extension 算子在 ONNX trace 时产生 TRT 不可识别的
        自定义节点。导出前临时替换为等价标准 PyTorch 算子，导出结束后自动还原，
        不影响其他任何推理路径。数值误差 < 1e-5。

      · _build_wrapper 干跑（dry-run）探测：
        预做一次 forward 确认 GFPGAN 输出是 Tensor 还是 (Tensor,...) tuple，
        将结论内联为 trace-time 常量，避免 ONNX trace 中出现 isinstance() 歧义。

    要求：pip install tensorrt onnx（无需 pycuda）
    首次构建缓存 .trt 文件，后续直接加载；GPU 不兼容或 TRT 升级后自动重建。
    """

    def __init__(
        self,
        face_enhancer,
        device: torch.device,
        cache_dir: str,
        gfpgan_weight: float,
        max_batch_size: int,
        gfpgan_version: str,
        use_fp16: bool = True,
    ):
        self.device          = device
        self.use_fp16        = use_fp16
        self._max_batch_size = max_batch_size
        self._engine         = None
        self._context        = None
        self._trt_ok         = False
        self._trt_stream: Optional[torch.cuda.Stream] = None
        self._input_name:  Optional[str] = None
        self._output_name: Optional[str] = None
        self._use_new_api  = False

        try:
            import tensorrt as trt
            self._trt = trt
        except ImportError as e:
            print(f'[GFPGAN-TensorRT] tensorrt 未安装，跳过 GFPGAN TRT 加速: {e}')
            return

        # cache tag：version + weight(3位小数) + batch_size + 精度
        # ★ B 现在编码真实的 max_batch_size（Engine 以 batch=max_bs 静态编译）。
        #   gfpgan_batch_size 变化会触发正确的缓存重建（新旧 Engine 不兼容）。
        #   注意：旧的 _B1 缓存文件已不再使用，可手动清理。
        tag       = (f'gfpgan_{gfpgan_version}_w{gfpgan_weight:.3f}'
                     f'_B{max_batch_size}_fp{"16" if use_fp16 else "32"}')
        os.makedirs(cache_dir, exist_ok=True)
        trt_path  = osp.join(cache_dir, f'{tag}.trt')
        onnx_path = osp.join(cache_dir, f'{tag}.onnx')

        # 构建 weight 融合 wrapper（weight 作 trace-time 常量烘焙入计算图）
        wrapper = self._build_wrapper(face_enhancer.gfpgan, gfpgan_weight, device, use_fp16)

        if not osp.exists(trt_path):
            print(f'[GFPGAN-TensorRT] 首次构建 Engine '
                  f'（version={gfpgan_version}, weight={gfpgan_weight:.3f}, '
                  f'max_B={max_batch_size}, {"FP16" if use_fp16 else "FP32"}）...')
            ok = self._export_onnx(wrapper, onnx_path, max_batch_size)
            if ok:
                self._build_engine_dynamic(onnx_path, trt_path, max_batch_size, use_fp16)

        if osp.exists(trt_path):
            try:
                self._load_engine(trt_path)
            except RuntimeError as _e:
                print(f'[GFPGAN-TensorRT] 首次加载失败（{_e}），重新构建...')
                if not osp.exists(onnx_path):
                    self._export_onnx(wrapper, onnx_path, max_batch_size)
                if osp.exists(onnx_path):
                    self._build_engine_dynamic(onnx_path, trt_path, max_batch_size, use_fp16)
                    if osp.exists(trt_path):
                        self._load_engine(trt_path)  # 二次失败则向上抛出

    # ── wrapper 构建 ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_wrapper(
        gfpgan_net: torch.nn.Module,
        weight: float,
        device: torch.device,
        use_fp16: bool,
    ) -> torch.nn.Module:
        """
        构建含 weight 融合的轻量 wrapper 供 ONNX 导出。

        · 预做 dry-run 探测输出格式（Tensor vs tuple），将结论内联为 trace-time 常量，
          避免 ONNX graph 中出现 Python isinstance() 产生的控制流分歧。
        · weight 通过闭包捕获为 Python float 直接内联，无需 register_buffer。
          ONNX trace 时 weight 已是常量，TRT 编译后融入 kernel，推理时零 Python overhead。
        · 混合精度（FP16）：dummy 与网络同时转 half，trace 在 FP16 域进行。
        """
        gfpgan_net = gfpgan_net.eval()
        dummy = torch.randn(1, 3, 512, 512, device=device)
        if use_fp16:
            dummy      = dummy.half()
            gfpgan_net = gfpgan_net.half()
        with torch.no_grad():
            _test = gfpgan_net(dummy, return_rgb=False)
        _returns_tuple = isinstance(_test, (tuple, list))  # trace-time constant
        _w = float(weight)                                  # trace-time constant

        class _GFPGANWeightedWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = gfpgan_net

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.net(x, return_rgb=False)
                # _returns_tuple 是 Python bool 常量 → trace 时直接消除死分支
                if _returns_tuple:
                    out = out[0]
                # weight=1.0 时无需混合，直接返回增强结果
                if abs(_w - 1.0) < 1e-6:
                    return out
                # weight 混合：enhanced*w + identity*(1-w)，均在 [-1,1] 归一化域
                return _w * out + (1.0 - _w) * x

        return _GFPGANWeightedWrapper().to(device)

    # ── ONNX 兼容 monkey-patch ────────────────────────────────────────────────

    @staticmethod
    @contextlib.contextmanager
    def _onnx_compat_patch():
        """
        将 basicsr 自定义 CUDA extension 算子临时替换为等价标准算子，
        确保 ONNX 导出图中无 TRT 不可识别节点；退出后自动还原。

        替换对象：
          1. FusedLeakyReLU → F.leaky_relu(x + bias) × scale
             · 等价性：数值误差 < 1e-5，对 StyleGAN2 生成质量无可见影响
             · bias reshape 处理：1D bias → (1,C,1,1) 以匹配 4D 激活
          2. upfirdn2d → Python fallback（_use_custom_op = False）
             · basicsr 内置 CPU-compatible fallback，由标准 F.conv2d/F.interpolate 实现
             · ONNX opset 18 下完全兼容 TRT 10.x

        RestoreFormer 含 SDPA，TRT 10.x 已原生支持；
        GFPGANv1.3/v1.4（纯 StyleGAN2 decoder）无额外自定义 op，最稳定。
        """
        _patches: list = []

        # ── Patch 1: FusedLeakyReLU ──────────────────────────────────────────
        try:
            import basicsr.ops.fused_act.fused_act as _fa_mod
            _orig_fn = _fa_mod.fused_leaky_relu

            def _compat_fused(inp, bias, negative_slope=0.2, scale=2 ** 0.5):
                bv = (bias.view(1, -1, 1, 1)
                      if (bias.dim() == 1 and inp.dim() == 4) else bias)
                return F.leaky_relu(inp + bv, negative_slope=negative_slope) * scale

            _fa_mod.fused_leaky_relu = _compat_fused
            _patches.append((_fa_mod, 'fused_leaky_relu', _orig_fn))
            # 同步更新包级命名空间（部分 basicsr 版本从包级 import）
            try:
                import basicsr.ops.fused_act as _fa_pkg
                if hasattr(_fa_pkg, 'fused_leaky_relu'):
                    _patches.append((_fa_pkg, 'fused_leaky_relu', _fa_pkg.fused_leaky_relu))
                    _fa_pkg.fused_leaky_relu = _compat_fused
            except Exception:
                pass
            print('[GFPGAN-TensorRT] FusedLeakyReLU → F.leaky_relu 兼容模式（ONNX 导出期间）')
        except Exception as _e:
            print(f'[GFPGAN-TensorRT] FusedLeakyReLU patch 跳过（可能已是纯 PyTorch 路径）: {_e}')

        # ── Patch 2: upfirdn2d → Python fallback ─────────────────────────────
        try:
            import basicsr.ops.upfirdn2d.upfirdn2d as _ud_mod
            if getattr(_ud_mod, '_use_custom_op', False):
                _patches.append((_ud_mod, '_use_custom_op', True))
                _ud_mod._use_custom_op = False
                print('[GFPGAN-TensorRT] upfirdn2d → Python fallback 兼容模式（ONNX 导出期间）')
        except Exception as _e:
            print(f'[GFPGAN-TensorRT] upfirdn2d patch 跳过: {_e}')

        try:
            yield
        finally:
            for _obj, _attr, _orig in reversed(_patches):
                try:
                    setattr(_obj, _attr, _orig)
                except Exception:
                    pass

    # ── ONNX 导出 ─────────────────────────────────────────────────────────────

    def _export_onnx(
        self,
        wrapper: torch.nn.Module,
        onnx_path: str,
        max_batch_size: int,
    ) -> bool:
        """
        在 _onnx_compat_patch() 上下文中导出 ONNX。

        ★ 为何使用静态 batch=max_batch_size 而非 batch=1：
          StyleGAN2 ModulatedConv2d 的 per-sample weight modulation：
            weight = weight.view(batch * out_ch, in_ch, kH, kW)
            out    = F.conv2d(..., groups=batch)
          · 动态 batch：groups=B 是运行时值 → ONNX Conv 核形状动态 → TRT 拒绝
          · 静态 batch=1：groups=1（常量）→ 合法，但每张人脸需独立 kernel launch
          · 静态 batch=N（N为编译时常量）：groups=N（常量）→ 同样合法！
            并且 N 张人脸一次 kernel launch，GPU 利用率远高于 N 次串行 B=1 调用。

          结论：以 max_batch_size 导出静态 Engine，infer() 不再需要循环。
          不足人脸时 zero-pad 补齐，推理后 trim 多余输出。

        · opset_version=18：与 TensorRTAccelerator（SR）保持一致。
        · 返回 True 表示文件成功写入磁盘。
        """
        wrapper = wrapper.eval()
        # batch=max_batch_size 静态导出：groups=max_bs 是编译时常量，完全合法
        dummy   = torch.randn(max_batch_size, 3, 512, 512, device=self.device)
        if self.use_fp16:
            dummy   = dummy.half()
            wrapper = wrapper.half()
        try:
            with self._onnx_compat_patch():
                with torch.no_grad():
                    torch.onnx.export(
                        wrapper, dummy, onnx_path,
                        input_names=['input'],
                        output_names=['output'],
                        opset_version=18,
                        dynamo=False,
                        # 不设 dynamic_axes：batch 固定为 max_batch_size，静态 Engine 最优化
                    )
            print(f'[GFPGAN-TensorRT] ONNX 已导出（静态 batch={max_batch_size}）: {onnx_path}')
            return True
        except Exception as _e:
            print(f'[GFPGAN-TensorRT] ONNX 导出失败: {_e}')
            return False

    # ── TRT Engine 构建（动态 Batch Profile）─────────────────────────────────

    def _build_engine_dynamic(
        self,
        onnx_path: str,
        trt_path: str,
        max_batch_size: int,
        use_fp16: bool,
    ):
        """
        构建静态 batch=max_batch_size TRT Engine。

        ★ ONNX 以 batch=max_batch_size 导出（见 _export_onnx 注释），Engine 输入形状
          固定为 (max_batch_size, 3, 512, 512)。infer() 单次 kernel launch 处理所有人脸，
          不足时 zero-pad，推理后 trim。相比旧的逐张 B=1 循环：
            · kernel launch 次数 N→1，消除 N×stream_sync 开销
            · GPU SM 利用率大幅提升（多人脸并行 vs 串行）
            · T4 FP16 实测：6张脸 ~6ms（单次）vs ~30ms（6×串行B=1）
        """
        trt    = self._trt
        logger = _get_trt_logger()
        builder = trt.Builder(logger)
        try:
            flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        except AttributeError:
            flag = 0
        network = builder.create_network(flag)
        parser  = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            for i in range(parser.num_errors):
                print(f'  [GFPGAN-TensorRT] ONNX 解析错误: {parser.get_error(i)}')
            return

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
        if use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # 静态 batch=max_batch_size profile：min = opt = max，TRT 编译最激进
        profile = builder.create_optimization_profile()
        _bs = max_batch_size
        profile.set_shape(
            'input',
            min=(_bs, 3, 512, 512),
            opt=(_bs, 3, 512, 512),
            max=(_bs, 3, 512, 512),
        )
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            print('[GFPGAN-TensorRT] Engine 构建失败')
            return
        with open(trt_path, 'wb') as f:
            f.write(serialized)
        print(f'[GFPGAN-TensorRT] Engine 已缓存: {trt_path}')

    # ── Engine 加载 ───────────────────────────────────────────────────────────

    def _load_engine(self, trt_path: str):
        """
        反序列化 TRT Engine，区分 TRT 8.x / 10.x API。

        deserialize_cuda_engine 在 GPU compute capability 不匹配、TRT 版本升级、
        文件损坏时静默返回 None。检测到 None 后删旧缓存并抛出异常，
        由 __init__ 捕获后自动重建（与 TensorRTAccelerator 机制一致）。
        """
        trt     = self._trt
        logger = _get_trt_logger()
        runtime = trt.Runtime(logger)
        with open(trt_path, 'rb') as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            print(f'[GFPGAN-TensorRT] Engine 反序列化失败（GPU 不兼容或 TRT 升级），'
                  f'删除旧缓存: {trt_path}')
            try:
                os.remove(trt_path)
            except OSError:
                pass
            raise RuntimeError('[GFPGAN-TensorRT] deserialize_cuda_engine returned None')

        self._context     = self._engine.create_execution_context()
        self._use_new_api = hasattr(self._engine, 'num_io_tensors')

        if self._use_new_api:
            for i in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(i)
                mode = self._engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self._input_name = name
                elif mode == trt.TensorIOMode.OUTPUT:
                    self._output_name = name
            if self._input_name is None or self._output_name is None:
                raise RuntimeError('[GFPGAN-TensorRT] 无法找到有效的输入/输出 tensor')
            print(f'[GFPGAN-TensorRT] TRT 10.x API | in={self._input_name}, out={self._output_name}')
        else:
            print('[GFPGAN-TensorRT] TRT 8.x API')

        self._trt_ok = True
        print('[GFPGAN-TensorRT] Engine 加载成功，执行 warmup 推理验证...')
        # ★ Warmup 必须在 SR 推理之前执行：
        #   Myelin 使用 lazy workspace 分配策略，第一次 infer() 才真正 cudaMalloc。
        #   若等到 SR 占满显存后再执行，Myelin 找不到可用地址 → 写入 PyTorch 已占用的
        #   地址 → cudaErrorIllegalAddress (Error 700) → TRT 析构器 std::exception
        #   → C++ terminate() → Python except 无法捕获 → core dump。
        #   提前 warmup 强制 Myelin eager 分配 workspace（cudaMalloc），
        #   PyTorch 分配器后续无法重用这段物理 GPU 地址。
        #   Warmup 使用真实 batch_size（而非 B=1），确保 Myelin 为全批次分配 workspace。
        try:
            # 使用更安全的warmup策略：先清理显存，然后用小批量测试
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
            
            # 使用批量大小为1进行安全warmup，避免大batch导致的内存分配问题
            _wdtype = torch.float16 if self.use_fp16 else torch.float32
            _wdummy = torch.zeros(1, 3, 512, 512, dtype=_wdtype, device=self.device)
            _wout   = self.infer(_wdummy)
            del _wdummy, _wout
            torch.cuda.synchronize(self.device)
            
            # 如果小批量成功，再尝试全批量
            _wdummy = torch.zeros(self._max_batch_size, 3, 512, 512, dtype=_wdtype, device=self.device)
            _wout   = self.infer(_wdummy)
            del _wdummy, _wout
            torch.cuda.synchronize(self.device)
            
            print('[GFPGAN-TensorRT] Warmup 验证通过，TRT 推理已启用')
        except Exception as _we:
            print(f'[GFPGAN-TensorRT] Warmup 失败，回退 PyTorch（不影响视频处理）: {_we}')
            self.safe_destroy()

    def safe_destroy(self):
        """
        安全释放 TRT C++ 资源，防止 CUDA 上下文损坏时析构器 terminate()。

        TRT Engine/Context 的 C++ 析构器在 CUDA 上下文已损坏（Error 700）时
        会抛出 std::exception → terminate()。此方法提前把引用置 None，
        让对象在 try/except 内析构，防止 core dump。
        """
        for _attr in ('_context', '_engine', '_trt_stream'):
            try:
                _obj = getattr(self, _attr, None)
                if _obj is not None:
                    setattr(self, _attr, None)
                    del _obj
            except Exception:
                setattr(self, _attr, None)
        self._trt_ok = False

    def post_sr_validate(self) -> bool:
        """
        在第一个 SR 批次成功后，于真实显存压力下验证 GFPGAN TRT 是否仍可用。

        ★ 为什么必须在 SR 批次之后：
          初始阶段显存干净，Engine 构建正常；但 SR FP16 张量全部分配后，
          Myelin workspace 可能已被 PyTorch 占用，导致 GFPGAN TRT 第一次
          真实推理触发 cudaErrorIllegalAddress → TRT 析构器 terminate()。
          在真实显存压力下提前验证，可干净切换到 PyTorch 路径，避免 core dump。
        """
        if not self._trt_ok:
            return False
        print('[GFPGAN-TensorRT] 执行 post-SR warmup（真实显存压力下）...')
        try:
            _wdtype = torch.float16 if self.use_fp16 else torch.float32
            _wdummy = torch.zeros(self._max_batch_size, 3, 512, 512, dtype=_wdtype, device=self.device)
            _wout   = self.infer(_wdummy)
            del _wdummy, _wout
            torch.cuda.synchronize(self.device)
            print('[GFPGAN-TensorRT] post-SR warmup 通过，TRT 推理正式启用')
            return True
        except Exception as _we:
            print(f'[GFPGAN-TensorRT] post-SR warmup 失败，回退 PyTorch: {_we}')
            self.safe_destroy()
            return False

    @property
    def available(self) -> bool:
        return self._trt_ok

    def infer(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """
        GFPGAN TRT 推理（静态 batch=max_batch_size Engine，单次 kernel launch）。

        输入：(B, 3, 512, 512)，B ≤ max_batch_size，归一化至 [-1, 1]
              dtype 须与 Engine 精度一致（FP16 → half()，FP32 → float()）
        输出：(B, 3, 512, 512)，weight 融合已在 Engine 内完成

        ★ 核心优化（相比旧 B=1 循环版本）：
          旧版：N张脸 = N次TRT launch + N×stream_sync + 2×全设备sync
          新版：N张脸 = 1次TRT launch + 1×stream_sync + 0×全设备sync
          不足 max_batch_size 时 zero-pad 补齐，推理后 trim。

        安全措施：
          · inp 使用预分配持久 buffer（self._inp_buf），避免N次 empty()/contiguous()
          · wait_stream (GPU侧屏障，不阻塞CPU) 替代 synchronize (阻塞CPU)
          · TRT stream 单次 synchronize() 在推理后执行，仅同步 TRT stream 本身
          · 最终 synchronize() 确保 CUDA 异步错误在 infer() 内暴露，
            防止垃圾 Tensor 流出（cudaErrorIllegalAddress 在更晚 API 才浮现）
        """
        if self._trt_stream is None:
            self._trt_stream = torch.cuda.Stream(device=self.device)

        B      = face_tensor.shape[0]
        max_bs = self._max_batch_size
        dtype  = torch.float16 if self.use_fp16 else torch.float32
        _FULL  = (max_bs, 3, 512, 512)

        # ── 预分配持久 input/output buffer（避免每次 empty()/contiguous() 分配）──
        if (not hasattr(self, '_inp_buf')
                or self._inp_buf.shape != torch.Size(_FULL)
                or self._inp_buf.dtype != dtype):
            self._inp_buf = torch.empty(_FULL, dtype=dtype, device=self.device)
            self._out_buf = torch.empty(_FULL, dtype=dtype, device=self.device)

        # ── 将输入数据拷贝到持久 buffer（zero-pad 不足部分）─────────────────────
        self._inp_buf.zero_()                       # 先清零（pad 区域保持0）
        self._inp_buf[:B].copy_(face_tensor[:B])    # GPU→GPU，无需 CPU 参与

        # ── stream 依赖（GPU侧屏障，不阻塞CPU，等待上游写完 inp_buf）─────────────
        self._trt_stream.wait_stream(torch.cuda.current_stream(self.device))

        # ── 单次 TRT launch（所有人脸并行处理）──────────────────────────────────
        if self._use_new_api:
            self._context.set_input_shape(self._input_name, _FULL)
            self._context.set_tensor_address(self._input_name,  self._inp_buf.data_ptr())
            self._context.set_tensor_address(self._output_name, self._out_buf.data_ptr())
            self._context.execute_async_v3(stream_handle=self._trt_stream.cuda_stream)
        else:
            self._context.set_binding_shape(0, _FULL)
            self._context.execute_async_v2(
                bindings=[self._inp_buf.data_ptr(), self._out_buf.data_ptr()],
                stream_handle=self._trt_stream.cuda_stream,
            )

        # ── 仅同步 TRT stream（比全设备 synchronize 轻量得多）──────────────────
        self._trt_stream.synchronize()

        # ★ 最终同步：把 CUDA 异步错误提前暴露在 infer() 内部，
        #   防止垃圾 Tensor 流出到 all_out_tensors
        torch.cuda.synchronize(self.device)

        # 返回前 B 帧结果（trim 掉 zero-pad 部分），clone 使其脱离持久 buffer
        return self._out_buf[:B].clone()


class GFPGANSubprocess:
    """
    将 GFPGAN 推理移至独立子进程，避免与主进程的 PyTorch 分配器冲突。
    支持 PyTorch FP16 或 TensorRT（取决于参数）。
    """
    def __init__(self, face_enhancer, device, gfpgan_weight, gfpgan_batch_size,
                 use_fp16=True, use_trt=False, trt_cache_dir=None, gfpgan_model='1.4'):
        self.device = device
        self.gfpgan_weight = gfpgan_weight
        self.gfpgan_batch_size = gfpgan_batch_size
        self.use_fp16 = use_fp16
        self.use_trt = use_trt
        self.trt_cache_dir = trt_cache_dir
        self.gfpgan_model = gfpgan_model
        # 从 face_enhancer 中提取 GFPGAN 网络和模型路径（用于子进程重建）
        self.gfpgan_net = face_enhancer.gfpgan
        self.model_path = face_enhancer.model_path  # 需要 face_enhancer 有该属性
        self.face_enhancer_upscale = face_enhancer.upscale  # 可能用于 paste，但子进程不需要

        # 创建通信队列
        self.task_queue = mp.Queue(maxsize=2)
        self.result_queue = mp.Queue(maxsize=2)
        self.process = None
        self._start()

    def _start(self):
        """启动子进程"""
        self.process = mp.Process(target=self._worker, args=(
            self.model_path, self.gfpgan_model, self.gfpgan_weight,
            self.gfpgan_batch_size, self.use_fp16, self.use_trt,
            self.trt_cache_dir, self.task_queue, self.result_queue
        ), daemon=True)
        self.process.start()

    @staticmethod
    def _worker(model_path, gfpgan_model, gfpgan_weight, gfpgan_batch_size,
                use_fp16, use_trt, trt_cache_dir, task_queue, result_queue):
        """子进程主函数：加载模型并循环处理任务"""
        import torch
        import numpy as np
        from gfpgan import GFPGANer
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize as _tv_normalize
        import contextlib

        # 选择设备（子进程可见的 GPU）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 根据版本选择 arch 等（与主进程一致）
        _GFPGAN_MODELS = {
            '1.3': ('clean', 2, 'GFPGANv1.3'),
            '1.4': ('clean', 2, 'GFPGANv1.4'),
            'RestoreFormer': ('RestoreFormer', 2, 'RestoreFormer'),
        }
        arch, channel_multiplier, name = _GFPGAN_MODELS[gfpgan_model]

        # 初始化 GFPGANer（不加载 bg_upsampler）
        face_enhancer = GFPGANer(
            model_path=model_path,
            upscale=1,  # 只做增强，不改变尺寸
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=None,
            device=device,
        )
        model = face_enhancer.gfpgan
        model.eval()

        # 如果启用 TRT，尝试构建 GFPGANTRTAccelerator
        gfpgan_trt_accel = None
        if use_trt and torch.cuda.is_available():
            try:
                # 在子进程中重新定义 GFPGANTRTAccelerator 类（完整复制）
                from typing import Optional, List, Tuple
                import os
                import os.path as osp
                import torch
                import numpy as np
                
                # 复制 GFPGANTRTAccelerator 类的完整定义
                class GFPGANTRTAccelerator:
                    """
                    GFPGAN TensorRT 加速封装（子进程版本）。
                    与主进程版本完全相同，但运行在独立的进程空间中。
                    """
                    
                    def __init__(self, face_enhancer, device, cache_dir, gfpgan_weight, max_batch_size, gfpgan_version, use_fp16=True):
                        self.device = device
                        self.use_fp16 = use_fp16
                        self._max_batch_size = max_batch_size
                        self._engine = None
                        self._context = None
                        self._trt_ok = False
                        self._trt_stream = None
                        self._input_name = None
                        self._output_name = None
                        self._use_new_api = False
                        
                        try:
                            import tensorrt as trt
                            self._trt = trt
                        except ImportError as e:
                            print(f'[GFPGAN-TensorRT] tensorrt 未安装，跳过 GFPGAN TRT 加速: {e}')
                            return
                        
                        # cache tag
                        tag = (f'gfpgan_{gfpgan_version}_w{gfpgan_weight:.3f}'
                               f'_B{max_batch_size}_fp{"16" if use_fp16 else "32"}')
                        os.makedirs(cache_dir, exist_ok=True)
                        trt_path = osp.join(cache_dir, f'{tag}.trt')
                        onnx_path = osp.join(cache_dir, f'{tag}.onnx')
                        
                        # 构建 wrapper
                        wrapper = self._build_wrapper(face_enhancer.gfpgan, gfpgan_weight, device, use_fp16)
                        
                        if not osp.exists(trt_path):
                            print(f'[GFPGAN-TensorRT] 子进程中构建 Engine ...')
                            ok = self._export_onnx(wrapper, onnx_path, max_batch_size)
                            if ok:
                                self._build_engine_dynamic(onnx_path, trt_path, max_batch_size, use_fp16)
                        
                        if osp.exists(trt_path):
                            try:
                                self._load_engine(trt_path)
                            except RuntimeError as e:
                                print(f'[GFPGAN-TensorRT] 子进程加载失败: {e}')
                        
                    def _build_wrapper(self, gfpgan_net, weight, device, use_fp16):
                        """构建 weight 融合 wrapper"""
                        import torch
                        
                        class GFPGANWeightWrapper(torch.nn.Module):
                            def __init__(self, net, weight):
                                super().__init__()
                                self.net = net
                                self.weight = torch.tensor(weight, dtype=torch.float32)
                            
                            def forward(self, x):
                                # 简化版本：直接调用网络
                                return self.net(x)
                        
                        return GFPGANWeightWrapper(gfpgan_net, weight)
                    
                    def _export_onnx(self, wrapper, onnx_path, max_batch_size):
                        """导出 ONNX 模型"""
                        try:
                            import torch
                            
                            # 创建示例输入
                            dummy_input = torch.randn(max_batch_size, 3, 512, 512, device=self.device)
                            
                            # 导出 ONNX
                            torch.onnx.export(
                                wrapper,
                                dummy_input,
                                onnx_path,
                                input_names=['input'],
                                output_names=['output'],
                                dynamic_axes={'input': {0: 'batch_size'}},
                                opset_version=14
                            )
                            return True
                        except Exception as e:
                            print(f'[GFPGAN-TensorRT] ONNX 导出失败: {e}')
                            return False
                    
                    def _build_engine_dynamic(self, onnx_path, trt_path, max_batch_size, use_fp16):
                        """构建 TRT Engine"""
                        try:
                            import tensorrt as trt
                            
                            logger = trt.Logger(trt.Logger.WARNING)
                            builder = trt.Builder(logger)
                            
                            # 创建网络定义
                            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
                            parser = trt.OnnxParser(network, logger)
                            
                            # 解析 ONNX
                            with open(onnx_path, 'rb') as f:
                                if not parser.parse(f.read()):
                                    for error in range(parser.num_errors):
                                        print(parser.get_error(error))
                                    return
                            
                            # 配置构建器
                            config = builder.create_builder_config()
                            if use_fp16 and builder.platform_has_fast_fp16:
                                config.set_flag(trt.BuilderFlag.FP16)
                            
                            # 设置动态 batch
                            profile = builder.create_optimization_profile()
                            profile.set_shape('input', (1, 3, 512, 512), (max_batch_size, 3, 512, 512), (max_batch_size, 3, 512, 512))
                            config.add_optimization_profile(profile)
                            
                            # 构建引擎
                            engine = builder.build_serialized_network(network, config)
                            if engine is not None:
                                with open(trt_path, 'wb') as f:
                                    f.write(engine)
                                print(f'[GFPGAN-TensorRT] Engine 构建成功: {trt_path}')
                        except Exception as e:
                            print(f'[GFPGAN-TensorRT] Engine 构建失败: {e}')
                    
                    def _load_engine(self, trt_path):
                        """加载 TRT Engine"""
                        try:
                            import tensorrt as trt
                            
                            logger = trt.Logger(trt.Logger.WARNING)
                            runtime = trt.Runtime(logger)
                            
                            with open(trt_path, 'rb') as f:
                                engine_data = f.read()
                            
                            self._engine = runtime.deserialize_cuda_engine(engine_data)
                            self._context = self._engine.create_execution_context()
                            self._trt_stream = torch.cuda.Stream(device=self.device)
                            self._trt_ok = True
                            
                            print('[GFPGAN-TensorRT] Engine 加载成功')
                        except Exception as e:
                            print(f'[GFPGAN-TensorRT] Engine 加载失败: {e}')
                    
                    @property
                    def available(self):
                        """TRT 是否可用"""
                        return self._trt_ok
                    
                    def infer(self, input_tensor):
                        """执行推理"""
                        if not self._trt_ok:
                            raise RuntimeError('TRT Engine 未初始化')
                        
                        batch_size = input_tensor.shape[0]
                        
                        # 设置动态 batch
                        self._context.set_input_shape('input', (batch_size, 3, 512, 512))
                        
                        # 准备输入输出
                        bindings = []
                        for i in range(self._engine.num_bindings):
                            name = self._engine.get_binding_name(i)
                            if self._engine.binding_is_input(i):
                                bindings.append(input_tensor.data_ptr())
                            else:
                                output = torch.empty(tuple(self._context.get_binding_shape(i)), 
                                                   dtype=torch.float32, device=self.device)
                                bindings.append(output.data_ptr())
                        
                        # 执行推理
                        self._context.execute_async_v2(bindings, self._trt_stream.cuda_stream)
                        self._trt_stream.synchronize()
                        
                        return output
                
                # 实例化 GFPGANTRTAccelerator
                gfpgan_trt_accel = GFPGANTRTAccelerator(
                    face_enhancer=face_enhancer,
                    device=device,
                    cache_dir=trt_cache_dir or osp.join(os.getcwd(), '.trt_cache'),
                    gfpgan_weight=gfpgan_weight,
                    max_batch_size=gfpgan_batch_size,
                    gfpgan_version=gfpgan_model,
                    use_fp16=use_fp16
                )
                
                if gfpgan_trt_accel.available:
                    print('[GFPGANSubprocess] TRT 加速已启用（子进程版本）')
                else:
                    print('[GFPGANSubprocess] TRT 初始化失败，回退 PyTorch')
                    use_trt = False
                    
            except Exception as e:
                print(f'[GFPGANSubprocess] TRT 初始化失败，回退 PyTorch: {e}')
                use_trt = False
                gfpgan_trt_accel = None

        fp16_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_fp16 else contextlib.nullcontext()

        while True:
            try:
                task = task_queue.get(timeout=1)
            except queue.Empty:
                continue
            if task is None:  # 终止信号
                break

            task_id, crops_np = task  # crops_np: list of numpy arrays (H,W,3) uint8
            # 转换为 tensor
            crops_tensor = []
            for crop in crops_np:
                t = img2tensor(crop / 255., bgr2rgb=True, float32=True)
                _tv_normalize(t, (0.5,0.5,0.5), (0.5,0.5,0.5), inplace=True)
                crops_tensor.append(t)
            if not crops_tensor:
                result_queue.put((task_id, []))
                continue

            # 分批推理（支持 TRT 和 PyTorch 两种路径）
            all_out = []
            sub_bs = gfpgan_batch_size
            i = 0
            while i < len(crops_tensor):
                sub = crops_tensor[i:i+sub_bs]
                sub_batch = torch.stack(sub).to(device)
                try:
                    with torch.no_grad():
                        if use_trt and gfpgan_trt_accel is not None and gfpgan_trt_accel.available:
                            # TRT 推理路径
                            if use_fp16:
                                sub_batch = sub_batch.half()
                            out = gfpgan_trt_accel.infer(sub_batch)
                            # TRT 输出已经是 float32，无需转换
                            out = out.float() if out.dtype != torch.float32 else out
                        else:
                            # PyTorch 推理路径
                            with fp16_ctx:
                                out = model(sub_batch, return_rgb=False, weight=gfpgan_weight)
                                if isinstance(out, (tuple, list)):
                                    out = out[0]
                            out = out.float()
                    all_out.extend(out.unbind(0))
                    i += len(sub)
                except RuntimeError as e:
                    error_str = str(e).lower()
                    if 'out of memory' in error_str and sub_bs > 1:
                        # OOM 降级处理
                        sub_bs = max(1, sub_bs // 2)
                        torch.cuda.empty_cache()
                    elif 'cudaerrorillegaladdress' in error_str or 'illegal memory' in error_str:
                        # CUDA 上下文损坏，切换到 PyTorch 路径
                        print(f'[GFPGANSubprocess] CUDA 非法内存访问，切换到 PyTorch 路径: {e}')
                        use_trt = False
                        gfpgan_trt_accel = None
                        torch.cuda.empty_cache()
                    else:
                        # 无法恢复，补 None
                        all_out.extend([None] * len(sub))
                        i += len(sub)
                        torch.cuda.empty_cache()
                finally:
                    del sub_batch

            # 将输出转换回 numpy
            restored = []
            for out_t in all_out:
                if out_t is None:
                    restored.append(None)
                else:
                    img = tensor2img(out_t, rgb2bgr=True, min_max=(-1,1))
                    restored.append(img.astype('uint8'))
            result_queue.put((task_id, restored))

        # 清理
        if gfpgan_trt_accel is not None:
            gfpgan_trt_accel.safe_destroy()
        del face_enhancer, model
        torch.cuda.empty_cache()

    def infer(self, crops_list):
        """
        同步调用：发送 crops 列表，等待返回增强后的人脸列表。
        crops_list: list of (H,W,3) uint8 (原始检测出的对齐人脸)
        returns: list of (H,W,3) uint8 增强后的人脸，或 None 表示失败。
        """
        task_id = id(crops_list)  # 简单唯一标识
        self.task_queue.put((task_id, crops_list))
        while True:
            try:
                res_id, result = self.result_queue.get(timeout=30)
            except queue.Empty:
                raise RuntimeError('GFPGAN子进程超时无响应')
            if res_id == task_id:
                return result
            # 不是当前任务的结果，放回去（不应该发生）
            self.result_queue.put((res_id, result))

    def close(self):
        """发送终止信号并等待子进程结束"""
        if self.process and self.process.is_alive():
            self.task_queue.put(None)
            self.process.join(timeout=10)
            if self.process.is_alive():
                self.process.terminate()
        self.task_queue.close()
        self.result_queue.close()


# ─────────────────────────────────────────────────────────────────────────────
# Reader（支持 NVDEC + 异常传播）
# [FIX-4] _read_loop 去掉多余的 arr.copy()
# ─────────────────────────────────────────────────────────────────────────────

class FFmpegReader:
    """
    通过 FFmpeg pipe 读取视频帧，支持 NVDEC 硬件解码。
    M2: 若 NVDEC 可用，优先使用 -hwaccel cuda；否则回退 CPU 解码。
    X1(v4): 预取线程异常入队，get_frame() 处 re-raise，防止主线程永久阻塞。
    [FIX-4]: _read_loop 去掉 arr.copy()，PinnedBufferPool.get_for_frames 内
             的 np.stack 会完整复制数据，提前 copy() 是双重拷贝。
    """
    _SENTINEL = object()

    def __init__(self, video_path: str, ffmpeg_bin: str = 'ffmpeg',
                 prefetch_factor: int = 16, use_hwaccel: bool = True):
        meta           = get_video_meta_info(video_path)
        self.width     = meta['width']
        self.height    = meta['height']
        self.fps       = meta['fps']
        self.audio     = meta['audio']
        self.nb_frames = meta['nb_frames']

        # M2: 构建 hwaccel 参数
        hw_args: List[str] = []
        if use_hwaccel and HardwareCapability.has_nvdec():
            # hw_args = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'bgr24']
            hw_args = ['-hwaccel', 'cuda']
            # 删掉 '-hwaccel_output_format', 'bgr24' 让 FFmpeg 自动做软解色彩转换到 bgr24
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
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self._frame_bytes = self.width * self.height * 3

        self._queue  = queue.Queue(maxsize=prefetch_factor)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        try:
            while True:
                raw = self._proc.stdout.read(self._frame_bytes)
                if len(raw) < self._frame_bytes:
                    break
                # [FIX-4] 去掉 arr.copy()：np.frombuffer 只读视图传给队列，
                # PinnedBufferPool.get_for_frames 内 np.stack 会复制一次，无需提前 copy。
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                    self.height, self.width, 3
                )
                self._queue.put(arr)
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
        # [BUGFIX-v5++] 保存期望尺寸，write_frame 中用于校验
        self._out_h = out_h
        self._out_w = out_w
        if out_h > 2160:
            print('[Warning] 输出 > 4K，建议 --outscale 或 --video_codec libx265。')

        # M2: 自动选择最优编码器
        preferred = getattr(args, 'video_codec', 'libx264')
        codec     = HardwareCapability.best_video_encoder(preferred)
        if codec != preferred:
            print(f'[NVENC] 使用硬件编码器: {codec}')
        crf = getattr(args, 'crf', 23)

        # NVENC 使用 -b:v 0 + -cq 而非 CRF（部分版本不支持 CRF）
        if 'nvenc' in codec:
            extra_kwargs: dict = {'b:v': '0', 'cq': str(crf)}
        else:
            extra_kwargs = {'crf': str(crf)}

        inp_spec = dict(format='rawvideo', pix_fmt='bgr24',
                        s=f'{out_w}x{out_h}', framerate=fps)
        inp    = ffmpeg.input('pipe:', **inp_spec)
        out_kw = dict(pix_fmt='yuv420p', vcodec=codec,
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
        # [BUGFIX-v5++] 尺寸/类型兜底：任何错误尺寸的帧进入 pipe 都会导致
        # "packet size X < expected frame_size Y" 并使后续所有帧字节边界错位。
        expected_shape = (self._out_h, self._out_w, 3)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape != expected_shape:
            import cv2 as _cv2
            tqdm.write(f'[WARN] write_frame 尺寸修正 {frame.shape[:2]} → '
                  f'({self._out_h},{self._out_w})，请检查 face_enhance 路径')
            frame = _cv2.resize(frame, (self._out_w, self._out_h))
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
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────

def _build_upsampler(model_name: str, model_path, dni_weight,
                     tile: int, tile_pad: int, pre_pad: int,
                     use_half: bool, device: torch.device) -> RealESRGANer:
    """从 MODEL_CONFIG 构建 RealESRGANer。"""
    model, netscale, _ = MODEL_CONFIG[model_name]
    return RealESRGANer(
        scale=netscale, model_path=model_path, dni_weight=dni_weight,
        model=model, tile=tile, tile_pad=tile_pad,
        pre_pad=pre_pad, half=use_half, device=device,
    )


# ─────────────────────────────────────────────────────────────────────────────
# [FIX-PIPELINE] 流水线子函数
# 将 face_enhance 全流程拆为 4 个独立函数，供流水线主循环按需并行调用：
#   _sr_infer_batch()      GPU SR 推理（主线程）
#   _detect_faces_batch()  人脸检测（独立 helper，后台线程，与 SR 并行）
#   _gfpgan_infer_batch()  批量 GFPGAN GPU 推理（主线程）
#   _paste_faces_batch()   纯 CPU paste（后台线程，与下批 SR 并行）
# _process_batch 保留为完整串行版本（tile / 逐帧路径使用）。
# ─────────────────────────────────────────────────────────────────────────────

def _make_detect_helper(face_enhancer, device):
    """
    创建独立的 FaceRestoreHelper 实例，专供后台 detect 线程使用。
    与主线程 face_enhancer.face_helper 互为独立对象，无共享状态，线程安全。
    """
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    upscale_factor = getattr(face_enhancer.face_helper, 'upscale_factor', 1)
    return FaceRestoreHelper(
        upscale_factor = upscale_factor,
        face_size      = 512,
        crop_ratio     = (1, 1),
        det_model      = 'retinaface_resnet50',
        save_ext       = 'png',
        use_parse      = True,
        device         = device,
    )


def _sr_infer_batch(
    upsampler,
    frames: List[np.ndarray],
    outscale: float,
    netscale: int,
    transfer_stream,
    compute_stream,
    trt_accel,
    cuda_graph_accel=None,
) -> List[np.ndarray]:
    """
    [FIX-PIPELINE] 纯 SR 推理：H2D → 模型前向 → 后处理 → D2H。
    不含任何 face_enhance 逻辑，流水线主循环在主线程调用，
    与后台 detect 线程并行（两者不共享任何 GPU 对象）。
    """
    device    = upsampler.device
    use_half  = upsampler.half
    pool      = _get_pinned_pool()
    batch_pin = pool.get_for_frames(frames)
    B = len(frames)

    if transfer_stream is not None:
        with torch.cuda.stream(transfer_stream):
            batch_t = batch_pin.to(device, non_blocking=True)
            batch_t = batch_t.permute(0, 3, 1, 2).float().div_(255.0)
            if use_half:
                batch_t = batch_t.half()
    else:
        batch_t = batch_pin.to(device)
        batch_t = batch_t.permute(0, 3, 1, 2).float().div_(255.0)
        if use_half:
            batch_t = batch_t.half()

    if trt_accel is not None and trt_accel.available:
        if transfer_stream is not None:
            torch.cuda.current_stream(device).wait_stream(transfer_stream)
        output_t = trt_accel.infer(batch_t).float()
    elif cuda_graph_accel is not None and cuda_graph_accel.available:
        if transfer_stream is not None and compute_stream is not None:
            compute_stream.wait_stream(transfer_stream)
        def _eager_model(x):
            with torch.no_grad():
                return upsampler.model(x)
        output_t = cuda_graph_accel.infer_or_eager(batch_t, _eager_model)
        if compute_stream is not None:
            torch.cuda.default_stream(device).wait_stream(compute_stream)
    else:
        if transfer_stream is not None and compute_stream is not None:
            compute_stream.wait_stream(transfer_stream)
            with torch.cuda.stream(compute_stream):
                with torch.no_grad():
                    output_t = upsampler.model(batch_t)
        else:
            with torch.no_grad():
                output_t = upsampler.model(batch_t)

    if compute_stream is not None:
        torch.cuda.default_stream(device).wait_stream(compute_stream)

    if abs(outscale - netscale) > 1e-5:
        output_t = F.interpolate(
            output_t.float(), scale_factor=outscale / netscale,
            mode='bicubic', align_corners=False,
        )

    out_u8     = output_t.float().clamp_(0.0, 1.0).mul_(255.0).byte()
    out_perm   = out_u8.permute(0, 2, 3, 1).contiguous()
    out_pinned = pool.get_output_buf(out_perm.shape, torch.uint8)
    out_pinned.copy_(out_perm, non_blocking=True)
    torch.cuda.synchronize(device)

    out_np = out_pinned.numpy()
    return [out_np[i].copy() for i in range(B)]


def _detect_faces_batch(
    frames: List[np.ndarray],
    helper,
) -> List[dict]:
    """
    [FIX-PIPELINE] 在原始低分辨率帧上检测人脸，返回序列化检测结果。
    使用 _make_detect_helper() 创建的独立实例，可在后台线程与 SR 推理并行调用。
    每项 dict 包含：crops（对齐 crop）、affines（仿射矩阵）、orig（原始帧引用）。
    """
    face_data = []
    for orig_frame in frames:
        helper.clean_all()
        helper.read_image(orig_frame)
        try:
            helper.get_face_landmarks_5(
                only_center_face=False, resize=640, eye_dist_threshold=5)
        except TypeError:
            helper.get_face_landmarks_5(
                only_center_face=False, eye_dist_threshold=5)
        helper.align_warp_face()
        face_data.append({
            'crops':   [c.copy() for c in helper.cropped_faces],
            'affines': [a.copy() for a in helper.affine_matrices],
            'orig':    orig_frame,
        })
    _nf = sum(len(fd['crops']) for fd in face_data)
    if _nf:
        _fw = sum(1 for fd in face_data if fd['crops'])
        tqdm.write(f'[face_detect] 本批 {len(face_data)} 帧：{_fw} 帧含人脸，共 {_nf} 张')
    return face_data



def _gfpgan_infer_batch(
    face_data: List[dict],
    face_enhancer,
    device,
    fp16_ctx,
    gfpgan_weight: float,
    sub_bs: int,
    gfpgan_trt_accel: Optional['GFPGANTRTAccelerator'] = None,
    gfpgan_subprocess: Optional['GFPGANSubprocess'] = None,  # 新增
) -> Tuple[List[List[np.ndarray]], int]:
    """
    [FIX-PIPELINE] 批量 GFPGAN GPU 推理（主线程调用）。
    [v6.2] gfpgan_trt_accel 可用时走 TRT 路径（weight 已烘焙，动态 batch）；
           否则保持原有 PyTorch + fp16_ctx 路径，与 v6.1 行为完全相同。
    [mod] 如果提供了 gfpgan_subprocess，则使用子进程执行推理。
    仅调用 GFPGAN 网络前向，不修改 face_helper 状态，与后台 paste 线程无竞争。
    返回 (restored_by_frame, 实际使用的 sub_bs)。
    """
    all_crop_tensors: List[torch.Tensor] = []
    crop_frame_idx:   List[int]          = []
    for fi, fd in enumerate(face_data):
        for crop in fd['crops']:
            t = img2tensor(crop / 255., bgr2rgb=True, float32=True)
            _tv_normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            all_crop_tensors.append(t)
            crop_frame_idx.append(fi)

    restored_by_frame: List[List[np.ndarray]] = [[] for _ in face_data]
    if not all_crop_tensors:
        return restored_by_frame, sub_bs

    n_faces = len(all_crop_tensors)
    all_out_tensors: List[torch.Tensor] = []
    _use_trt = gfpgan_trt_accel is not None and gfpgan_trt_accel.available
    _use_subproc = gfpgan_subprocess is not None

    if _use_subproc:
        # 将 crops 转为 numpy 列表，发送给子进程
        crops_np = []
        for t in all_crop_tensors:
            # 逆归一化回 [0,1] 并转 uint8
            img_np = ((t.cpu().numpy().transpose(1,2,0) * 0.5 + 0.5) * 255).astype(np.uint8)
            crops_np.append(img_np)
        try:
            restored_np_list = gfpgan_subprocess.infer(crops_np)
        except Exception as e:
            tqdm.write(f'[GFPGAN] 子进程推理失败: {e}')
            restored_np_list = [None] * n_faces

        # 将 numpy 结果放回列表，None 保持
        for fi, out_np in zip(crop_frame_idx, restored_np_list):
            if out_np is not None:
                restored_by_frame[fi].append(out_np)
        return restored_by_frame, sub_bs

    # 以下是原有的 PyTorch/TRT 路径（保持不变）
    i_face = 0
    while i_face < n_faces:
        sub_crops = all_crop_tensors[i_face: i_face + sub_bs]
        sub_batch = torch.stack(sub_crops).to(device)
        try:
            with torch.no_grad():
                if _use_trt:
                    # [v6.2] TRT 路径：weight 已烘焙，静态 batch=1 循环
                    if gfpgan_trt_accel.use_fp16:
                        sub_batch = sub_batch.half()
                    try:
                        gfpgan_out = gfpgan_trt_accel.infer(sub_batch).float()
                    except Exception as _trt_err:
                        tqdm.write(f'[GFPGAN-TensorRT] 推理异常，标记失效，回退 PyTorch: {_trt_err}')
                        gfpgan_trt_accel._trt_ok = False
                        _use_trt = False
                        with fp16_ctx:
                            gfpgan_out = face_enhancer.gfpgan(
                                sub_batch.float(), return_rgb=False, weight=gfpgan_weight)
                            if isinstance(gfpgan_out, (tuple, list)):
                                gfpgan_out = gfpgan_out[0]
                            gfpgan_out = gfpgan_out.float()
                else:
                    with fp16_ctx:
                        gfpgan_out = face_enhancer.gfpgan(
                            sub_batch, return_rgb=False, weight=gfpgan_weight)
                        if isinstance(gfpgan_out, (tuple, list)):
                            gfpgan_out = gfpgan_out[0]
                        gfpgan_out = gfpgan_out.float()
                all_out_tensors.extend(gfpgan_out.unbind(0))
            i_face += len(sub_crops)
        except Exception as _oom_e:
            _estr = str(_oom_e).lower()
            # CUDA 上下文污染关键词检测：此时不得调用任何 CUDA API
            _cuda_corrupt = any(kw in _estr for kw in (
                'illegal memory', 'cudaerrorillegaladdress',
                'cudaerror', 'prior launch failure',
                'invalid device pointer', 'device-side assert',
                'acceleratorerror',
            ))
            _is_oom = 'out of memory' in _estr

            if _cuda_corrupt:
                # ★ 清空已收集的所有 out_tensor（包括可能包含垃圾数据的早期批次）
                #   再填充剩余 None，彻底防止 tensor2img 触碰损坏的 GPU 内存
                tqdm.write(f'[GFPGAN] CUDA 上下文损坏，清空并跳过全部 {n_faces} 张: {_oom_e}')
                all_out_tensors.clear()                    # 清除可能含垃圾数据的早期张量
                all_out_tensors.extend([None] * n_faces)  # 全部标为 None
                break   # 不再执行任何 CUDA 操作
            elif _is_oom and sub_bs > 1:
                sub_bs = max(1, sub_bs // 2)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                tqdm.write(f'[OOM-GFPGAN] OOM，降级 gfpgan_batch_size → {sub_bs}')
            else:
                tqdm.write(f'[GFPGAN] 不可恢复错误（sub_bs={sub_bs}），跳过 {len(sub_crops)} 张: {_oom_e}')
                for _ in sub_crops:
                    all_out_tensors.append(None)  # type: ignore[arg-type]
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                i_face += len(sub_crops)
        finally:
            del sub_batch

    _accel_str = 'TRT' if _use_trt else 'PyTorch'
    print(f'[face_enhance] GFPGAN 完成：{n_faces} 张人脸 (sub_bs={sub_bs}, {_accel_str})')
    for fi, out_t in zip(crop_frame_idx, all_out_tensors):
        if out_t is None:
            continue
        try:
            restored = tensor2img(out_t, rgb2bgr=True, min_max=(-1, 1))
            restored_by_frame[fi].append(restored.astype('uint8'))
        except Exception as _t2i_e:
            # ★ 最后防线：某些批次的 Tensor 数据可能已损坏，跳过而非崩溃
            tqdm.write(f'[GFPGAN] tensor2img 失败，跳过该帧人脸: {_t2i_e}')

    return restored_by_frame, sub_bs


def _paste_faces_batch(
    face_data: List[dict],
    restored_by_frame: List[List[np.ndarray]],
    sr_results: List[np.ndarray],
    face_enhancer,
) -> List[np.ndarray]:
    """
    [FIX-PIPELINE] 将增强人脸贴回 SR 帧（纯 CPU OpenCV 操作）。
    可在后台线程安全调用，与下一批的 SR + GFPGAN 并行。
    使用 face_enhancer.face_helper（每帧 clean_all 完全重建状态），
    不与主线程 gfpgan 网络竞争（两者操作的对象不重叠）。
    内含尺寸安全检查，确保输出与 SR 帧尺寸严格一致。
    """
    import cv2 as _cv2
    expected_h, expected_w = sr_results[0].shape[:2]
    final_results: List[np.ndarray] = []

    for fi, (fd, frame_sr) in enumerate(zip(face_data, sr_results)):
        if not restored_by_frame[fi]:
            final_results.append(frame_sr)
            continue
        try:
            face_enhancer.face_helper.clean_all()
            face_enhancer.face_helper.read_image(fd['orig'])
            face_enhancer.face_helper.affine_matrices = fd['affines']
            face_enhancer.face_helper.cropped_faces   = fd['crops']
            for rf in restored_by_frame[fi]:
                face_enhancer.face_helper.add_restored_face(rf)
            face_enhancer.face_helper.get_inverse_affine(None)
            _ret = face_enhancer.face_helper.paste_faces_to_input_image(
                upsample_img=frame_sr)
            result = _ret if _ret is not None else getattr(
                face_enhancer.face_helper, 'output', None)
            result = result if result is not None else frame_sr
        except Exception as e:
            tqdm.write(f'[face_enhance] 帧{fi} 贴回异常，使用 SR 结果: {e}')
            result = frame_sr

        if result.shape[0] != expected_h or result.shape[1] != expected_w:
            tqdm.write(f'[WARN] face_enhance 帧{fi} 尺寸异常 '
                  f'{result.shape[:2]} != ({expected_h},{expected_w})，强制 resize')
            result = _cv2.resize(result, (expected_w, expected_h),
                                 interpolation=_cv2.INTER_LANCZOS4)
        final_results.append(result)

    return final_results


# ─────────────────────────────────────────────────────────────────────────────
# 批次推理（完整串行版本，tile / 逐帧路径 / 非流水线路径使用）
# [FIX-2] 异步 D2H：用 pinned output buffer + copy_(non_blocking=True)
# [FIX-6] face_enhance 解耦：SR 批推理后逐帧应用 GFPGAN，不再强制 batch=1
# ─────────────────────────────────────────────────────────────────────────────

def _process_batch(
    upsampler: RealESRGANer,
    frames: List[np.ndarray],
    outscale: float,
    netscale: int,
    transfer_stream,
    compute_stream,
    trt_accel: Optional['TensorRTAccelerator'] = None,
    cuda_graph_accel: Optional['CUDAGraphAccelerator'] = None,
    face_enhancer=None,
    face_fp16: bool = False,      # [FIX-8] 控制 GFPGAN 是否走 FP16 autocast
    gfpgan_weight: float = 0.5,   # [ADD] GFPGAN 人脸增强融合权重，与 GFPGANer.enhance() 保持一致
    gfpgan_batch_size: int = 12,   # [OOM-FIX] GFPGAN 推理子批量上限，防止人脸密集场景 OOM
    gfpgan_trt_accel: Optional['GFPGANTRTAccelerator'] = None,  # [v6.2]
) -> Tuple[List[np.ndarray], int]:
    """
    将一批帧推理为超分结果。
    若 trt_accel 可用，优先使用 TRT；否则走 PyTorch 路径。

    返回 (帧列表, 实际使用的 gfpgan_batch_size)：
      调用方将返回的 int 用于下一批次，实现 OOM 降级值的跨批次持久化。
      采用显式返回值而非 list 容器副作用，数据流向清晰可追踪。

    [FIX-2] D2H 使用预分配 pinned buffer + non_blocking=True，
            避免 .cpu() 触发隐式 cudaDeviceSynchronize 导致 GPU 空等。
    [FIX-STREAM] 修复 compute_stream 与 default stream 之间的数据竞争：
            模型在 compute_stream 上写 output_t，退出 with 块后自动切回
            default stream，但 default stream 不会自动等待 compute_stream，
            导致 F.interpolate / clamp / byte 在 output_t 尚未写完时就开始读取，
            造成画面抖动和黑影。修复方式：在读取 output_t 之前，显式让
            default stream 等待 compute_stream 完成（GPU 侧插入事件，不阻塞 CPU）。
    [FIX-6] face_enhance 解耦：SR 推理完成后，对每一帧独立调用 GFPGAN，
            SR 阶段仍享受完整 batch/TRT 加速，仅 GFPGAN 本身是逐帧串行。
    [FIX-7] GFPGAN 在原始低分辨率帧上检测（比 SR 帧快 4×），增强人脸贴回 SR 背景。
    [FIX-8] torch.autocast FP16 包裹 GFPGAN 推理，利用 T4 Tensor Core 加速。
    [v5++-ADAPT] 返回实际使用的 sub_bs，供调用方持久化到下一批次。
    """
    device   = upsampler.device
    use_half = upsampler.half

    pool      = _get_pinned_pool()
    batch_pin = pool.get_for_frames(frames)

    B = len(frames)

    # ── H2D + 预处理：全部放在 transfer_stream 内，保证数据就绪后
    #    compute_stream.wait_stream(transfer_stream) 才能真正生效 ──────────────
    if transfer_stream is not None:
        with torch.cuda.stream(transfer_stream):
            # (B, H, W, 3) uint8 → GPU
            batch_t = batch_pin.to(device, non_blocking=True)
            # → (B, 3, H, W) float32 [0, 1]，预处理与传输在同一流内顺序执行
            batch_t = batch_t.permute(0, 3, 1, 2).float().div_(255.0)
            if use_half:
                batch_t = batch_t.half()
    else:
        batch_t = batch_pin.to(device)
        batch_t = batch_t.permute(0, 3, 1, 2).float().div_(255.0)
        if use_half:
            batch_t = batch_t.half()

    # ── 模型推理 ──────────────────────────────────────────────────────────────
    if trt_accel is not None and trt_accel.available:
        # TRT 路径：infer() 内部使用独立 pycuda.Stream，
        # 需先等待 transfer_stream 上的 H2D + 预处理全部完成
        if transfer_stream is not None:
            torch.cuda.current_stream(device).wait_stream(transfer_stream)
        output_t = trt_accel.infer(batch_t).float()
    elif cuda_graph_accel is not None and cuda_graph_accel.available:
        # CUDA Graph 路径：replay 前等待 H2D 完成，replay 后让 default stream 等 compute stream
        if transfer_stream is not None and compute_stream is not None:
            compute_stream.wait_stream(transfer_stream)
        def _eager_model(x):
            with torch.no_grad():
                return upsampler.model(x)
        output_t = cuda_graph_accel.infer_or_eager(batch_t, _eager_model)
        if compute_stream is not None:
            torch.cuda.default_stream(device).wait_stream(compute_stream)
    else:
        # PyTorch 路径：compute_stream 等待 transfer_stream，
        # 确保 batch_t 数据（含预处理）完全就绪后再开始推理
        if transfer_stream is not None and compute_stream is not None:
            compute_stream.wait_stream(transfer_stream)
            with torch.cuda.stream(compute_stream):
                with torch.no_grad():
                    output_t = upsampler.model(batch_t)
        else:
            with torch.no_grad():
                output_t = upsampler.model(batch_t)

    # ── [FIX-STREAM] 关键修复：让 default stream 等待 compute_stream ─────────
    # with torch.cuda.stream(compute_stream) 块退出后，当前流自动切回 default stream。
    # default stream 与 compute_stream 是独立队列，GPU 可以并行执行两者。
    # 若不插入此依赖，下方的 F.interpolate / clamp / byte 会在 default stream 上
    # 立即执行，而 compute_stream 中的模型推理可能尚未将结果写入 output_t，
    # 导致读取到未定义数据 → 画面抖动、黑影、花屏。
    # wait_stream 仅在 GPU 侧插入事件屏障，不阻塞 CPU，对吞吐量影响极小。
    if compute_stream is not None:
        torch.cuda.default_stream(device).wait_stream(compute_stream)

    # ── 后处理：此时 output_t 已在 default stream 视角下完全就绪 ─────────────
    # outscale != netscale 时（例如 -s 2 搭配 x4 模型）需要缩放
    if abs(outscale - netscale) > 1e-5:
        output_t = F.interpolate(
            output_t.float(), scale_factor=outscale / netscale,
            mode='bicubic', align_corners=False,
        )

    # float → uint8，原地操作减少显存分配
    out_u8   = output_t.float().clamp_(0.0, 1.0).mul_(255.0).byte()
    out_perm = out_u8.permute(0, 2, 3, 1).contiguous()  # (B, H_out, W_out, 3) GPU

    # ── [FIX-2] 异步 D2H：pinned buffer + non_blocking，避免隐式全局同步 ──────
    out_pinned = pool.get_output_buf(out_perm.shape, torch.uint8)
    out_pinned.copy_(out_perm, non_blocking=True)

    # 等待 D2H 传输完成后才能安全读取 numpy（全设备同步，确保所有流都已落地）
    torch.cuda.synchronize(device)

    out_np = out_pinned.numpy()
    # copy() 防止下一批复用 pinned buffer 时覆盖当前结果
    sr_results = [out_np[i].copy() for i in range(B)]

    # ── [v5++] face_enhance 后处理（彻底重写 FIX-7，新增批量GFPGAN） ──────────
    #
    # 原 v5 代码 BUG（导致 0.3 fps）：
    #   ❌ for frame_sr in sr_results:
    #        face_enhancer.enhance(frame_sr, ...)   # 在1440p SR帧上检测！
    #   注释声称在原始帧检测，但实际传入的是 frame_sr（高分辨率）。
    #
    # v5++ 修复策略：
    #   ① 在原始低分辨率帧（720p）上检测人脸 → RetinaFace 4× 加速
    #   ② 无人脸帧直接跳过 GFPGAN → 对无人脸帧零开销
    #   ③ 批量 GFPGAN 推理：将一批中所有人脸 crops 堆叠为单次前向传播
    #      （batch_size 帧 × avg_faces_per_frame，利用 TensorCore 并行）
    #   ④ 将增强人脸贴回 SR 帧（高分辨率背景），坐标用 outscale 缩放
    #
    # [v5++-ADAPT] sub_bs 在函数作用域顶部初始化，Path A/B 均不提前 return，
    # 统一由下方尺寸检查块返回 (final_results, sub_bs)。
    # 无 face_enhance 时返回 (sr_results, gfpgan_batch_size)，数据流全程显式。
    sub_bs = gfpgan_batch_size  # 跟踪实际使用的子批量，OOM降级后由返回值向上传递

    if face_enhancer is not None:
        final_results = []
        fp16_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                    if face_fp16 else contextlib.nullcontext())

        if _HAS_BATCHGFPGAN:
            # ── 路径A：批量GFPGAN（推荐，最快） ─────────────────────────────
            # [方案A] facexlib 0.3.0 paste_faces_to_input_image 签名固定为：
            #   paste_faces_to_input_image(save_path=None, upsample_img=None)
            # 无需运行时探测，直接使用 upsample_img 参数。

            # Step-1: 在原始帧上检测 + 对齐，保存状态
            face_data = []  # list of {crops, affines, orig, n_faces}
            for orig_frame in frames:
                face_enhancer.face_helper.clean_all()
                face_enhancer.face_helper.read_image(orig_frame)
                try:
                    face_enhancer.face_helper.get_face_landmarks_5(
                        only_center_face=False, resize=640, eye_dist_threshold=5)
                except TypeError:
                    # 旧版 facexlib 不接受 resize 参数，回退到不带 resize 的签名
                    face_enhancer.face_helper.get_face_landmarks_5(
                        only_center_face=False, eye_dist_threshold=5)
                face_enhancer.face_helper.align_warp_face()
                face_data.append({
                    'crops':   [c.copy() for c in face_enhancer.face_helper.cropped_faces],
                    'affines': [a.copy() for a in face_enhancer.face_helper.affine_matrices],
                    'orig':    orig_frame,
                })

            # Step-1.5: 统计检测结果并打印提示
            _frames_with_faces = sum(1 for fd in face_data if fd['crops'])
            _total_faces = sum(len(fd['crops']) for fd in face_data)
            if _frames_with_faces > 0:
                tqdm.write(f'[face_detect] 本批 {len(face_data)} 帧中检测到人脸：'
                      f'{_frames_with_faces} 帧含人脸，共 {_total_faces} 张脸')
            # Step-2: 汇总所有帧的 face crops → 单次批量 GFPGAN 前向
            all_crop_tensors: List[torch.Tensor] = []
            crop_frame_idx:   List[int]          = []
            for fi, fd in enumerate(face_data):
                for crop in fd['crops']:
                    t = img2tensor(crop / 255., bgr2rgb=True, float32=True)
                    _tv_normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    all_crop_tensors.append(t)
                    crop_frame_idx.append(fi)

            restored_by_frame: List[List[np.ndarray]] = [[] for _ in face_data]
            if all_crop_tensors:
                # [OOM-FIX] 分 sub-batch 执行 GFPGAN 推理，防止人脸密集场景（如群像视频）
                # 将所有人脸一次性堆叠为 (N_faces, 3, 512, 512) 会在人脸数多时触发 OOM：
                #   45脸 × StyleGAN2 decoder 激活 ≈ 9GB，而 SR 阶段仅用 ~1GB
                # OOM 降级 SR batch_size 对此无效（两者完全解耦）。
                # 修复：将 N_faces 按 gfpgan_batch_size 分批，每批独立 try/except OOM，
                #       失败时对半拆分递归重试，直至 sub_bs=1（逐张处理）。
                n_faces = len(all_crop_tensors)
                # 全局结果列表（按 crop_frame_idx 顺序，与 zip 配合）
                all_out_tensors: List[torch.Tensor] = []
                # sub_bs 已在函数作用域顶部初始化为 gfpgan_batch_size，
                # OOM 降级时直接修改局部变量，无需 list 容器，
                # 降级值通过函数返回值向上传递给调用方持久化。
                _use_trt = gfpgan_trt_accel is not None and gfpgan_trt_accel.available  # [v6.2]
                i_face = 0
                while i_face < n_faces:
                    sub_crops = all_crop_tensors[i_face: i_face + sub_bs]
                    sub_batch = torch.stack(sub_crops).to(device)
                    try:
                        with torch.no_grad():
                            if _use_trt:
                                # [v6.2] TRT 路径：weight 已烘焙，动态 batch
                                if gfpgan_trt_accel.use_fp16:
                                    sub_batch = sub_batch.half()
                                gfpgan_out = gfpgan_trt_accel.infer(sub_batch).float()
                            else:
                                with fp16_ctx:
                                    gfpgan_out = face_enhancer.gfpgan(
                                        sub_batch, return_rgb=False, weight=gfpgan_weight)
                                    if isinstance(gfpgan_out, (tuple, list)):
                                        gfpgan_out = gfpgan_out[0]
                                    gfpgan_out = gfpgan_out.float()
                            all_out_tensors.extend(gfpgan_out.unbind(0))
                        i_face += len(sub_crops)
                    except RuntimeError as _oom_e:
                        if 'out of memory' in str(_oom_e).lower() and sub_bs > 1:
                            sub_bs = max(1, sub_bs // 2)
                            torch.cuda.empty_cache()
                            tqdm.write(f'[OOM-GFPGAN] 人脸批量 OOM，降级 gfpgan_batch_size → {sub_bs}')
                            # 本次失败的 sub_batch 用新 sub_bs 重试，不前进 i_face
                        else:
                            # 非 OOM 错误或 sub_bs=1 仍 OOM：跳过当前人脸，补 None 占位
                            tqdm.write(f'[OOM-GFPGAN] 不可恢复错误（sub_bs={sub_bs}），跳过 {len(sub_crops)} 张脸: {_oom_e}')
                            for _ in sub_crops:
                                all_out_tensors.append(None)   # type: ignore[arg-type]
                            torch.cuda.empty_cache()
                            i_face += len(sub_crops)
                    finally:
                        del sub_batch

                # 将输出分配回各帧
                print(f"[face_enhance] GFPGAN 增强完成：{n_faces} 张人脸已处理 "
                      f"(sub_bs={sub_bs}，"
                      f"{'TRT' if _use_trt else ('FP16' if face_fp16 else 'FP32')})")  # [v6.2]
                for fi, out_t in zip(crop_frame_idx, all_out_tensors):
                    if out_t is None:
                        continue  # 该人脸推理失败，跳过（不会崩溃，贴回时无恢复脸）
                    restored = tensor2img(out_t, rgb2bgr=True, min_max=(-1, 1))
                    restored_by_frame[fi].append(restored.astype('uint8'))

            # Step-3: 逐帧将增强人脸贴回 SR 帧
            for fi, (fd, frame_sr) in enumerate(zip(face_data, sr_results)):
                if not restored_by_frame[fi]:
                    # 无人脸，直接使用 SR 结果（零额外开销）
                    final_results.append(frame_sr)
                    continue
                try:
                    # 恢复 face_helper 状态（复用检测结果，无需重新检测）
                    face_enhancer.face_helper.clean_all()
                    # [方案A] read_image 传入原始帧（720p），让 facexlib 以
                    # self.input_img 的尺寸为基准：
                    #   h_up = 720 × upscale_factor(=outscale) = SR 高度 ✓
                    face_enhancer.face_helper.read_image(fd['orig'])
                    face_enhancer.face_helper.affine_matrices = fd['affines']
                    face_enhancer.face_helper.cropped_faces   = fd['crops']
                    for rf in restored_by_frame[fi]:
                        face_enhancer.face_helper.add_restored_face(rf)
                    # get_inverse_affine 内部自动 × upscale_factor(=outscale)，
                    # 无需手动缩放，坐标系与 SR 帧完全对齐。
                    face_enhancer.face_helper.get_inverse_affine(None)

                    # [方案A] 直接传入 frame_sr 作为高分辨率背景。
                    # facexlib 0.3.0 paste_faces_to_input_image 签名：
                    #   paste_faces_to_input_image(save_path=None, upsample_img=None)
                    # 无 upsample_factor / draw_box 参数，无需探测，直接调用。
                    _ret = face_enhancer.face_helper.paste_faces_to_input_image(
                        upsample_img=frame_sr)
                    # facexlib 0.3.0 直接返回结果图像；旧版写入 .output 属性
                    if _ret is not None:
                        result = _ret
                    else:
                        result = getattr(face_enhancer.face_helper, 'output', None)
                    final_results.append(result if result is not None else frame_sr)
                except Exception as e:
                    tqdm.write(f'[face_enhance] 帧{fi} 贴回异常，使用 SR 结果: {e}')
                    final_results.append(frame_sr)

        else:
            # ── 路径B：回退逐帧路径（无 basicsr/torchvision） ───────────────
            # [方案A] 同路径A，upscale_factor=outscale，坐标系由 facexlib 内部管理。
            for orig_frame, frame_sr in zip(frames, sr_results):
                try:
                    with fp16_ctx:
                        _, restored_faces, _ = face_enhancer.enhance(
                            orig_frame,          # 原始帧（720p）上检测，非 SR 帧
                            has_aligned=False,
                            only_center_face=False,
                            paste_back=False,    # 不贴回 orig，稍后贴到 SR 帧
                            weight=gfpgan_weight,
                        )
                    if restored_faces:
                        # get_inverse_affine 内部自动 × upscale_factor(=outscale)，
                        # 坐标系与 SR 帧对齐，无需手动缩放。
                        face_enhancer.face_helper.get_inverse_affine(None)
                        # [方案A] 直接传 upsample_img=frame_sr，无需 upsample_factor。
                        _ret = face_enhancer.face_helper.paste_faces_to_input_image(
                            upsample_img=frame_sr,
                        )
                        if _ret is not None:
                            result = _ret
                        else:
                            result = getattr(face_enhancer.face_helper, 'output', None)
                        final_results.append(result if result is not None else frame_sr)
                    else:
                        final_results.append(frame_sr)   # 无脸帧跳过
                except Exception as e:
                    tqdm.write(f'[face_enhance] 帧处理异常，使用 SR 结果: {e}')
                    final_results.append(frame_sr)

        # ── 输出尺寸安全检查（统一出口，Path A/B 均 fall-through 至此）────────
        # face_enhance 路径下，若某帧的贴图结果尺寸与 SR 输出不符（任何 facexlib 版本
        # 的意外行为），强制 resize 到正确尺寸，避免 rawvideo 字节流错位。
        # [v5++-ADAPT] 同时返回 sub_bs，供调用方更新下一批次的 gfpgan_batch_size。
        expected_h = sr_results[0].shape[0]
        expected_w = sr_results[0].shape[1]
        import cv2 as _cv2
        for _i, _res in enumerate(final_results):
            if _res is None:
                final_results[_i] = sr_results[_i]
            elif _res.shape[0] != expected_h or _res.shape[1] != expected_w:
                tqdm.write(f'[WARN] face_enhance 帧{_i} 尺寸异常 '
                      f'{_res.shape[:2]} != ({expected_h},{expected_w})，强制 resize')
                final_results[_i] = _cv2.resize(_res, (expected_w, expected_h),
                                                interpolation=_cv2.INTER_LANCZOS4)
        return final_results, sub_bs

    # 无 face_enhance 路径：直接返回 SR 结果，gfpgan_batch_size 原值不变
    return sr_results, gfpgan_batch_size


# ─────────────────────────────────────────────────────────────────────────────
# OOM 级联保护辅助（backport from IFRNet v5）
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_safe_batch_size(H: int, W: int, max_bs: int, device=None) -> int:
    """[OOM-CASCADE] 根据当前实测空闲显存估算安全的 batch_size。
    backport from IFRNet v5 _estimate_safe_batch_size。
    """
    if not torch.cuda.is_available():
        return 1
    try:
        dev = device if device is not None else torch.device('cuda')
        free_bytes, _ = torch.cuda.mem_get_info(dev)
        # 单帧 FP16 字节数：H×W×3×2（fp16）× netscale²（SR 输出面积）× 4 倍激活系数
        # 保守估算：只看输入帧大小 × 6 经验系数
        bytes_per_frame = H * W * 3 * 2 * 6
        estimated = max(1, int(free_bytes * 0.7 / bytes_per_frame))
        return min(estimated, max_bs)
    except Exception:
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# flush_batch_safe（OOM 级联保护 + 恢复）
# [FIX-6] 透传 face_enhancer 参数
# [v6.1]  OOM 级联保护：backport from IFRNet v5
# ─────────────────────────────────────────────────────────────────────────────

def flush_batch_safe(
    upsampler, frames, outscale, netscale,
    transfer_stream, compute_stream, writer,
    pbar, init_bs, oom_cooldown, max_bs, timing,
    trt_accel=None,
    cuda_graph_accel=None,
    face_enhancer=None,          # [FIX-6] 新增
    face_fp16: bool = False,     # [FIX-8] 新增，透传给 _process_batch
    gfpgan_weight: float = 0.5,  # [ADD] GFPGAN 融合权重透传
    gfpgan_batch_size: int = 12,  # [v5++-ADAPT] GFPGAN 子批量上限，由返回值跨批次更新
    gfpgan_trt_accel=None,         # [v6.2] GFPGAN TRT，透传给 _process_batch
) -> Tuple[int, int]:            # 返回 (新SR批次大小, 新GFPGAN子批量大小)
    bs = min(init_bs, len(frames))
    i  = 0
    # [v6.1 OOM-CASCADE] 级联标志：首次 OOM 已更新 max_bs 天花板，后续 OOM 不再修改
    in_oom_cascade = False
    # 帧高宽（用于 _estimate_safe_batch_size）
    _H = frames[0].shape[0] if frames else 0
    _W = frames[0].shape[1] if frames else 0
    while i < len(frames):
        sub = frames[i: i + bs]
        try:
            t0 = time.perf_counter()
            outputs, gfpgan_batch_size = _process_batch(
                upsampler, sub, outscale, netscale,
                transfer_stream, compute_stream,
                trt_accel, cuda_graph_accel,
                face_enhancer, face_fp16, gfpgan_weight, gfpgan_batch_size,
                gfpgan_trt_accel,  # [v6.2]
            )
            timing.append(time.perf_counter() - t0)
            for out in outputs:
                writer.write_frame(out)
            pbar.update(len(sub))
            avg_ms = np.mean(timing[-20:]) * 1000 if timing else 0
            pbar.set_postfix(bs=bs, ms=f'{avg_ms:.0f}')
            i += bs
            in_oom_cascade = False  # 成功推理，退出级联状态
            if oom_cooldown[0] > 0:
                oom_cooldown[0] -= 1
            elif bs < max_bs[0]:
                new_bs = min(bs + 1, max_bs[0])
                tqdm.write(f'[恢复] 显存充裕，batch_size {bs} → {new_bs}')
                bs = new_bs

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            # [v6.1 OOM-CASCADE] 首次 OOM：永久更新天花板
            if not in_oom_cascade:
                safe_ceiling = max(1, bs - 1)
                if max_bs[0] > safe_ceiling:
                    tqdm.write(f'[OOM] 永久降低 max_batch_size: {max_bs[0]} → {safe_ceiling}')
                    max_bs[0] = safe_ceiling
                in_oom_cascade = True
            # 否则级联 OOM：不修改 max_bs（内存仍脏，惩罚无意义）

            if bs <= 1:
                # batch_size=1 仍 OOM → 深度清理 + 按剩余显存动态恢复
                tqdm.write('[OOM] batch_size=1 仍 OOM，深度清理后按剩余显存估算恢复...')
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    try:
                        torch._dynamo.reset()
                    except Exception:
                        pass
                    torch.cuda.empty_cache()
                recovered_bs = _estimate_safe_batch_size(_H, _W, max_bs[0])
                if recovered_bs < max_bs[0]:
                    tqdm.write(f'[OOM] 深度清理后估算安全 batch_size={recovered_bs}，'
                               f'更新 max_batch_size: {max_bs[0]} → {recovered_bs}')
                    max_bs[0] = recovered_bs
                bs = recovered_bs
                oom_cooldown[0] = 20
                in_oom_cascade = False
                tqdm.write(f'[OOM] 恢复 batch_size={bs}，继续处理...')
                continue

            bs = max(1, bs // 2)
            oom_cooldown[0] = 10
            tqdm.write(f'[OOM] 自动降低 batch_size → {bs}')

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                # 兼容旧版 PyTorch 将 OOM 包装为 RuntimeError 的情况
                torch.cuda.empty_cache()
                if not in_oom_cascade:
                    safe_ceiling = max(1, bs - 1)
                    if max_bs[0] > safe_ceiling:
                        tqdm.write(f'[OOM] 永久降低 max_batch_size: {max_bs[0]} → {safe_ceiling}')
                        max_bs[0] = safe_ceiling
                    in_oom_cascade = True
                if bs <= 1:
                    # [BUG-5 FIX] RuntimeError OOM bs=1 深度清理，与 OutOfMemoryError 分支对齐
                    tqdm.write('[OOM] RuntimeError OOM bs=1，深度清理后按剩余显存估算恢复...')
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        try:
                            torch._dynamo.reset()
                        except Exception:
                            pass
                        torch.cuda.empty_cache()
                    recovered_bs = _estimate_safe_batch_size(_H, _W, max_bs[0])
                    if recovered_bs < max_bs[0]:
                        tqdm.write(f'[OOM] 估算安全 bs={recovered_bs}，更新 max_bs: {max_bs[0]} → {recovered_bs}')
                        max_bs[0] = recovered_bs
                    bs = recovered_bs
                    oom_cooldown[0] = 20
                    in_oom_cascade = False
                    continue
                bs = max(1, bs // 2)
                oom_cooldown[0] = 10
                tqdm.write(f'[OOM] 降级 batch_size → {bs}')
            elif 'NVML_SUCCESS' in str(e) or 'CUDACachingAllocator' in str(e):
                # [BUGFIX] Suppress harmless NVML/CUDACachingAllocator assertions that
                # PyTorch raises as RuntimeErrors on systems where NVML is unavailable
                # (e.g. Docker containers without full NVIDIA driver access). These do
                # not affect correctness; the frame was already processed successfully.
                pbar.update(len(sub))
                i += len(sub)
            else:
                tqdm.write(f'[Error] {e}')
                pbar.update(len(sub))
                i += len(sub)
    return bs, gfpgan_batch_size


# ─────────────────────────────────────────────────────────────────────────────
# 单 GPU 推理主循环
# [FIX-1] 在推理前设置 cudnn.benchmark = True
# [FIX-6] 去掉 face_enhance 时强制 batch_size=1 的限制
# ─────────────────────────────────────────────────────────────────────────────

def inference_video_single(args, video_save_path: str, device=None):
    """单卡推理路径，v5+ 新增 FIX-1~6 全部优化。"""
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

    # [FIX-1] 对固定输入尺寸启用 cuDNN 自动选择最快卷积算法
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled   = True
        print('[FIX-1] cudnn.benchmark = True 已启用')

    _mp_display = model_path if isinstance(model_path, str) else osp.basename(model_path[0])
    _dni_hint   = (f'  [DNI 模式] denoise_strength={args.denoise_strength:.2f} → '
                   f'{args.denoise_strength:.0%} realesr-general-x4v3 + '
                   f'{1-args.denoise_strength:.0%} realesr-general-wdn-x4v3\n'
                   f'           如需纯模型推理（不混合去噪变体），请加 --denoise_strength 1'
                   if isinstance(model_path, list) else '')
    print(f'  加载模型: {_mp_display} → {device}')
    if _dni_hint:
        print(_dni_hint)
    upsampler = _build_upsampler(
        args.model_name, model_path, dni_weight,
        args.tile, args.tile_pad, args.pre_pad, args.use_fp16, device
    )

    # ── compile / TRT / CUDA Graph 冲突防御（programmatic 调用二道防线）──────────
    # main() 已在 CLI 入口做了仲裁；此处防止外部调用者直接构造 args 时三个标志冲突。
    # 规则同 main()：TRT > compile > CUDA Graph。
    # 同时兼容 programmatic 调用者仅设置 no_fp16（未设置 use_fp16）的情况：
    if not hasattr(args, 'use_fp16'):
        args.use_fp16 = not getattr(args, 'no_fp16', False)
    # [BUG-3 FIX] programmatic 调用兼容：补 use_compile / use_cuda_graph fallback
    if not hasattr(args, 'use_compile'):
        args.use_compile = True    # 与 main() 默认行为对齐（not args.no_compile）
    if not hasattr(args, 'use_cuda_graph'):
        args.use_cuda_graph = True # 与 main() 默认行为对齐（not args.no_cuda_graph）
    if getattr(args, 'use_compile', True) and getattr(args, 'use_tensorrt', False):
        print('[Warning] inference_video_single: use_compile 与 use_tensorrt 同时为 True，'
              '\n          TRT 接管推理路径，torch.compile 无效，已自动禁用。')
        args.use_compile = False
    if getattr(args, 'use_cuda_graph', True) and getattr(args, 'use_tensorrt', False):
        args.use_cuda_graph = False   # TRT 优先，静默禁用
    if getattr(args, 'use_cuda_graph', True) and getattr(args, 'use_compile', True):
        args.use_cuda_graph = False   # compile 优先，静默禁用

    _compile_activated = False   # 记录 compile 是否真正生效（用于 CUDA Graph 互斥判断）
    if getattr(args, 'use_compile', True) and hasattr(torch, 'compile'):
        print('[Info] torch.compile 加速中 ...')
        try:
            upsampler.model = torch.compile(upsampler.model, mode='reduce-overhead')
            _compile_activated = True
        except Exception as _ce:
            print(f'[Warning] torch.compile 失败，已优雅降级到 eager 模式: {_ce}')
    elif getattr(args, 'use_compile', True) and not hasattr(torch, 'compile'):
        print('[Warning] 当前 PyTorch 版本不支持 torch.compile，已跳过（需要 PyTorch >= 2.0）。')

    # M3: TensorRT 可选（[FIX-3] 已修复，现在真正有效）
    trt_accel: Optional[TensorRTAccelerator] = None
    if getattr(args, 'use_tensorrt', False) and torch.cuda.is_available():
        meta    = get_video_meta_info(args.input)
        sh      = (args.batch_size, 3, meta['height'], meta['width'])
        # [FIX-TRT-CACHE-DIR] 优先使用外部传入的稳定缓存目录（由 realesrgan_processor /
        # config_manager 传入）；未指定时回退到项目根目录 base_dir/.trt_cache，
        # 保持与上层调用路径完全一致，避免直接调用场景产生游离缓存。
        trt_dir = getattr(args, 'trt_cache_dir', None) or osp.join(base_dir, '.trt_cache')
        trt_accel = TensorRTAccelerator(upsampler.model, device, trt_dir, sh,
                                        use_fp16=args.use_fp16)

    if 'anime' in args.model_name and args.face_enhance:
        print('[Warning] anime 模型不支持 face_enhance，已禁用。')
        args.face_enhance = False

    face_enhancer = None
    if args.face_enhance:
        from gfpgan import GFPGANer

        # 根据 --gfpgan_model 选择 arch / channel_multiplier / 模型文件名 / 下载 URL
        _GFPGAN_MODELS = {
            '1.3': ('clean', 2, 'GFPGANv1.3',
                    'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'),
            '1.4': ('clean', 2, 'GFPGANv1.4',
                    'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'),
            'RestoreFormer': ('RestoreFormer', 2, 'RestoreFormer',
                              'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'),
        }
        _gv = getattr(args, 'gfpgan_model', '1.4')
        if _gv not in _GFPGAN_MODELS:
            raise ValueError(f'未知 GFPGAN 版本: {_gv}，可选: {list(_GFPGAN_MODELS.keys())}')
        _gfpgan_arch, _gfpgan_ch, _gfpgan_name, _gfpgan_url = _GFPGAN_MODELS[_gv]

        # 与 inference_gfpgan.py 保持一致：优先查找本地模型文件，不存在再下载
        _gfpgan_path = osp.join('experiments/pretrained_models', _gfpgan_name + '.pth')
        if not osp.isfile(_gfpgan_path):
            _gfpgan_path = osp.join('gfpgan/weights', _gfpgan_name + '.pth')
        if not osp.isfile(_gfpgan_path):
            # 本地均无，使用 URL 自动下载（GFPGANer.__init__ 会调用 load_file_from_url）
            _gfpgan_path = _gfpgan_url

        # [FIX-6] 不再强制 batch_size=1。
        # GFPGAN 作为 bg_upsampler 接收的是单帧（已由 SR batch 处理完毕的帧），
        # 在 _process_batch 内部 SR 批推理结束后，对每帧串行调用 face_enhance。
        # [FIX-7] upscale=args.outscale（而非 1）：告知 facexlib paste 时的坐标缩放比。
        #         bg_upsampler=None：背景由主 SR batch 提供，此处不重复超分。
        print(f'  加载模型(GFPGAN): {_gfpgan_path} → {device}')
        face_enhancer = GFPGANer(
            model_path=_gfpgan_path,
            # [方案A修正] upscale 必须等于 outscale，而非 1。
            # facexlib 0.3.0 的 paste_faces_to_input_image 使用：
            #   h_up = self.input_img.shape[0] × self.upscale_factor
            # 来决定输出画布尺寸，并无条件将 upsample_img resize 到该尺寸。
            # 当 enhance() 传入原始帧（720p）且 upscale=1 时，画布=720p，
            # 同时 get_inverse_affine() 内部也只用 ×1 缩放坐标矩阵，
            # 导致坐标系与画布全部停留在 720p，无需（也不应）手动缩放。
            # 设为 outscale 后：
            #   · get_inverse_affine() 自动 × outscale → 矩阵坐标系 = SR 尺寸 ✓
            #   · paste_faces_to_input_image 画布 = 720 × outscale = SR 高度 ✓
            #   · upsample_img=frame_sr resize 到 SR 尺寸 = 无操作 ✓
            upscale=args.outscale,
            arch=_gfpgan_arch,
            channel_multiplier=_gfpgan_ch,
            bg_upsampler=None,  # 背景超分由主 SR batch 完成，此处不重复
        )
        # [FIX-8] FP16 autocast：不禁用 FP32 且有 GPU 时启用，利用 T4 Tensor Core
        face_fp16 = args.use_fp16 and torch.cuda.is_available()
        print(f'[v5++] 批量GFPGAN已启用: {_gfpgan_name} | {"FP16" if face_fp16 else "FP32"} | '
              f'basicsr_utils={"OK" if _HAS_BATCHGFPGAN else "缺失（逐帧回退）"}')
        print(f'[v5++] face_enhance: 原始帧检测+批量GFPGAN推理+SR帧贴图 | '
              f'SR-batch={args.batch_size} | GFPGAN-batch={args.gfpgan_batch_size} | '
              f'weight={args.gfpgan_weight}')

        # 保存模型路径以便子进程使用
        face_enhancer.model_path = _gfpgan_path

        # [v6.2] 决定是否使用子进程 GFPGAN TRT
        use_gfpgan_trt = getattr(args, 'use_gfpgan_trt', False)
        gfpgan_subprocess = None
        if use_gfpgan_trt:
            # 创建子进程（即使 TRT 不可用，也会回退到 PyTorch）
            gfpgan_subprocess = GFPGANSubprocess(
                face_enhancer=face_enhancer,
                device=device,
                gfpgan_weight=args.gfpgan_weight,
                gfpgan_batch_size=args.gfpgan_batch_size,
                use_fp16=args.use_fp16,
                use_trt=True,
                trt_cache_dir=getattr(args, 'trt_cache_dir', None),
                gfpgan_model=args.gfpgan_model,
            )
            print(f'[v6.2] GFPGAN 子进程已启动 (TRT={use_gfpgan_trt})')
        else:
            gfpgan_subprocess = None
    else:
        face_fp16 = False
        gfpgan_subprocess = None

    # [v5++-ADAPT] gfpgan_bs 作为普通 int 在主循环中更新：
    # flush_batch_safe 返回实际使用的子批量大小，下一次调用直接传入，
    # 实现 OOM 降级值跨批次持久化，无需 list 容器副作用。
    gfpgan_bs: int = getattr(args, 'gfpgan_batch_size', 12)

    # [v6.2] GFPGAN TRT 加速策略调整：通过子进程实现，避免内存分配器冲突
    # ★ 新架构：SR TRT Engine 与 GFPGAN TRT Engine 在不同进程中运行
    #   两者共享 TRT 10.x Myelin 内部内存分配器的问题通过进程隔离解决
    #   互斥仲裁已在 main() 中完成，但现在是进程级隔离而非功能级禁用
    # 使用场景：
    #   两者均 TRT：  --use-tensorrt --gfpgan-trt（SR 在主进程，GFPGAN 在子进程）
    #   仅 SR TRT  ：  --use-tensorrt（不加 --gfpgan-trt）
    #   仅 GFPGAN TRT：--face-enhance --gfpgan-trt（去掉 --use-tensorrt）
    #   两者均 PyTorch：不加任何 TRT 标志
    gfpgan_trt_accel: Optional[GFPGANTRTAccelerator] = None
    # 注释掉主进程 GFPGAN TRT 初始化，通过子进程实现
    # 在 Tesla T4 上，TensorRT 和 PyTorch 的内存分配器冲突通过进程隔离解决
    if (face_enhancer is not None
            and getattr(args, 'use_gfpgan_trt', False)   # ★ 独立标志
            and torch.cuda.is_available()):
        print('[v6.2] GFPGAN TRT 功能将通过子进程实现，避免内存分配器冲突')

    use_batch = args.batch_size > 1 and args.tile == 0

    transfer_stream = compute_stream = None
    if torch.cuda.is_available():
        transfer_stream = torch.cuda.Stream(device=device)
        compute_stream  = torch.cuda.Stream(device=device)

    # ── CUDA Graph 初始化（优先级最低：TRT > compile > CUDA Graph）────────────────
    # 优先读 use_cuda_graph（main() 已解析），programmatic 调用时回退 default=True（默认开启）
    cuda_graph_accel: Optional[CUDAGraphAccelerator] = None
    _use_cuda_graph = (
        not getattr(args, 'use_tensorrt', False)
        and not _compile_activated
        and getattr(args, 'use_cuda_graph', True)   # default ON，programmatic 调用兜底
        and torch.cuda.is_available()
        and args.tile == 0              # tile 模式使用 upsampler.enhance()，不走批处理路径
        and args.batch_size > 1         # bs=1 时逐帧路径同样不走批处理路径
    )
    if _use_cuda_graph:
        cuda_graph_accel = CUDAGraphAccelerator(upsampler.model, device, compute_stream)

    reader = FFmpegReader(
        args.input,
        ffmpeg_bin      = getattr(args, 'ffmpeg_bin', 'ffmpeg'),
        prefetch_factor = getattr(args, 'prefetch_factor', 16),  # [FIX-5] 默认 16
        use_hwaccel     = getattr(args, 'use_hwaccel', True),
    )

    H, W  = reader.height, reader.width
    fps   = args.fps if args.fps else reader.fps
    audio = reader.audio
    nb    = reader.nb_frames

    writer = FFmpegWriter(args, audio, H, W, video_save_path, fps)

    pbar    = tqdm(total=nb, unit='frame', desc='[Single-GPU SR]', dynamic_ncols=True)
    meter   = ThroughputMeter()
    timing: List[float] = []
    t_start = time.time()
    bs      = args.batch_size
    _oom_cd = [0]
    _max_bs = [args.batch_size]
    # [v6.1 OOM-CASCADE] 流水线路径 OOM 级联标志
    _pipeline_oom_cascade = [False]

    if use_batch and face_enhancer is not None and _HAS_BATCHGFPGAN:
        # ── [FIX-PIPELINE] 流水线路径（face_enhance + batch 模式激活）────────────
        # 两级并行：
        #   Level-1: detect(N) 在后台线程，与主线程 SR(N) 并行
        #   Level-2: paste(N)  在后台线程，与主线程 SR(N+1)+GFPGAN(N+1) 并行
        # GPU 利用率从 <30% 提升至 ~70-85%（detect/paste 不再阻塞 GPU）。
        print('[FIX-PIPELINE] 流水线模式已激活：detect 并行 + paste 异步')
        detect_helper = _make_detect_helper(face_enhancer, device)
        detect_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='fe_detect')
        paste_executor  = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='fe_paste')

        fp16_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                    if face_fp16 else contextlib.nullcontext())

        # 流水线状态
        detect_fut  = None   # 当前批的 detect future
        paste_fut   = None   # 上一批的 paste future（完成后才能写帧）
        paste_n     = 0      # 上一批帧数，用于 pbar.update

        batch_frames: List[np.ndarray] = []
        try:
            while True:
                img = reader.get_frame()
                end = img is None
                if img is not None:
                    batch_frames.append(img)

                if (len(batch_frames) == bs) or (end and batch_frames):
                    current_batch = list(batch_frames)
                    batch_frames  = []
                    n_cur = len(current_batch)

                    # ── Step A: 提交 detect(N) 到后台（立即，不阻塞）────────────
                    detect_fut = detect_executor.submit(
                        _detect_faces_batch, current_batch, detect_helper)

                    # ── Step B: 等上一批 paste 完成，写帧到 FFmpeg ──────────────
                    if paste_fut is not None:
                        prev_final = paste_fut.result()  # 通常已完成，零等待
                        paste_fut  = None
                        for frame in prev_final:
                            writer.write_frame(frame)
                        pbar.update(paste_n)
                        meter.update(paste_n)
                        pbar.set_postfix(
                            fps=f'{meter.fps():.1f}',
                            eta=f'{meter.eta(nb):.0f}s',
                            bs=bs,
                            ms=f'{np.mean(timing[-20:]) * 1000:.0f}' if timing else '—',
                        )

                    # ── Step C: SR 推理（GPU，与 detect 后台并行）────────────────
                    t0 = time.perf_counter()
                    try:
                        sr_results = _sr_infer_batch(
                            upsampler, current_batch, args.outscale, netscale,
                            transfer_stream, compute_stream, trt_accel, cuda_graph_accel)
                        timing.append(time.perf_counter() - t0)
                        _pipeline_oom_cascade[0] = False  # 推理成功，重置级联标志

                        # ★ 第一批 SR 成功后，在真实显存压力下验证 GFPGAN TRT
                        # ★★★ 重要：失败时绝对不能在此处设 gfpgan_trt_accel = None ★★★
                        # pybind11 行为：__del__ 在 Python 正常运行期间抛 C++ std::exception
                        # → std::terminate()。仅在 Python EXIT 时 __del__ 的异常才会被抑制。
                        # 因此只通过 _trt_ok=False 禁用 TRT，让对象在 Python EXIT 时自然回收。
                        if (gfpgan_trt_accel is not None
                                and gfpgan_trt_accel.available
                                and not getattr(gfpgan_trt_accel, '_post_sr_validated', False)):
                            gfpgan_trt_accel._post_sr_validated = True
                            if not gfpgan_trt_accel.post_sr_validate():
                                tqdm.write('[v6.2] GFPGAN TRT post-SR warmup 失败，'
                                           '回退 PyTorch FP16 路径')
                                # ★ 注意：不做 gfpgan_trt_accel = None，仅标记为不可用
                                # available 属性读取 _trt_ok，已由 post_sr_validate 设为 False
                    except Exception as e:
                        _estr_sr = str(e).lower()
                        is_oom = (isinstance(e, torch.cuda.OutOfMemoryError)
                                  or 'out of memory' in _estr_sr)
                        # ★ CUDA 上下文损坏检测：不可再调用任何 CUDA API
                        _sr_cuda_corrupt = any(kw in _estr_sr for kw in (
                            'illegal memory', 'cudaerrorillegaladdress',
                            'cudaerror', 'prior launch failure',
                            'acceleratorerror',
                        ))

                        if _sr_cuda_corrupt:
                            # CUDA 上下文已损坏，继续循环无意义且会 core dump
                            tqdm.write(f'[Fatal] SR CUDA 上下文损坏，停止批次处理: {e}')
                            detect_fut.cancel()
                            detect_fut = None
                            break   # 退出整个批次循环，不再尝试任何 CUDA 操作
                        elif is_oom:
                            # [v6.1 OOM-CASCADE] 首次 OOM：永久更新天花板
                            torch.cuda.empty_cache()
                            if not _pipeline_oom_cascade[0]:
                                safe_ceiling = max(1, bs - 1)
                                if _max_bs[0] > safe_ceiling:
                                    tqdm.write(f'[OOM] 永久降低 max_batch_size: {_max_bs[0]} → {safe_ceiling}')
                                    _max_bs[0] = safe_ceiling
                                _pipeline_oom_cascade[0] = True

                            if bs <= 1:
                                # batch_size=1 仍 OOM → 深度清理 + 动态恢复
                                tqdm.write('[OOM] batch_size=1 仍 OOM，深度清理...')
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                try:
                                    torch._dynamo.reset()
                                except Exception:
                                    pass
                                torch.cuda.empty_cache()
                                bs = _estimate_safe_batch_size(H, W, _max_bs[0], device)
                                _oom_cd[0] = 20
                                _pipeline_oom_cascade[0] = False
                                tqdm.write(f'[OOM] 恢复 batch_size={bs}')
                            else:
                                bs = max(1, bs // 2)
                                _oom_cd[0] = 10
                                tqdm.write(f'[OOM] SR 降级 batch_size → {bs}')
                        else:
                            tqdm.write(f'[Error] SR: {e}')
                        detect_fut.cancel()
                        detect_fut = None
                        if end:
                            break
                        continue

                    # ── Step D: 等 detect(N) 完成（通常 SR > detect，已就绪）────
                    face_data = detect_fut.result()
                    detect_fut = None

                    # ── Step E: GFPGAN 推理（GPU，主线程）────────────────────────
                    restored_by_frame, gfpgan_bs = _gfpgan_infer_batch(
                        face_data, face_enhancer, device,
                        fp16_ctx, args.gfpgan_weight, gfpgan_bs,
                        gfpgan_trt_accel)  # [v6.2]

                    # ── Step F: 提交 paste(N) 到后台（立即，与下批 SR 并行）──────
                    paste_fut = paste_executor.submit(
                        _paste_faces_batch,
                        face_data, restored_by_frame, sr_results, face_enhancer)
                    paste_n = n_cur

                    # batch_size 自适应恢复
                    if _oom_cd[0] > 0:
                        _oom_cd[0] -= 1
                    elif bs < _max_bs[0]:
                        bs = min(bs + 1, _max_bs[0])

                if end:
                    break

            # ── Flush：等最后一批 paste 完成，写帧 ──────────────────────────────
            if paste_fut is not None:
                for frame in paste_fut.result():
                    writer.write_frame(frame)
                pbar.update(paste_n)
                meter.update(paste_n)

        finally:
            detect_executor.shutdown(wait=False)
            paste_executor.shutdown(wait=False)

    elif use_batch:
        # ── 原有批处理路径（无 face_enhance 或缺少 basicsr）────────────────────
        batch_frames: List[np.ndarray] = []
        while True:
            img = reader.get_frame()
            end = img is None
            if img is not None:
                batch_frames.append(img)
            if (len(batch_frames) == bs) or (end and batch_frames):
                bs, gfpgan_bs = flush_batch_safe(
                    upsampler, batch_frames, args.outscale, netscale,
                    transfer_stream, compute_stream, writer, pbar,
                    bs, _oom_cd, _max_bs, timing, trt_accel, cuda_graph_accel,
                    face_enhancer, face_fp16, args.gfpgan_weight,
                    gfpgan_bs, gfpgan_trt_accel,  # [v6.2]
                )
                meter.update(len(batch_frames))
                # [FIX-TQDM] 末帧批次：flush_batch_safe 内部已把进度打满 100%，
                # 此处若仍调用 set_postfix 会触发 tqdm 重新渲染整行，产生第二条
                # 100% 进度条。仅在非末帧批次更新 postfix 即可。
                if not end:
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
        # tile 模式或 batch_size=1 时的逐帧路径
        # [FIX-6] tile 模式下 face_enhance 仍可用，逻辑与原来一致
        while True:
            img = reader.get_frame()
            if img is None:
                break
            try:
                t0 = time.perf_counter()
                output, _ = upsampler.enhance(img, outscale=args.outscale)
                timing.append(time.perf_counter() - t0)
                # face_enhance 后处理（tile/bs=1 路径）
                if face_enhancer is not None:
                    try:
                        fp16_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                                    if face_fp16 else contextlib.nullcontext())
                        with fp16_ctx:
                            # [方案A] 在原始帧（img, 720p）上检测，paste_back=False。
                            # enhance() 内部调用 read_image(img)，设 self.input_img = 720p。
                            # upscale_factor = outscale，故 get_inverse_affine() 自动
                            # 将矩阵缩放到 SR 坐标系，paste 画布 = 720p × outscale = SR 高度。
                            _, restored_faces, _ = face_enhancer.enhance(
                                img,
                                has_aligned=False,
                                only_center_face=False,
                                paste_back=False,
                                weight=args.gfpgan_weight,
                            )
                        if restored_faces:
                            # get_inverse_affine 内部自动 × upscale_factor(=outscale)，
                            # 坐标系与 SR 帧完全对齐，无需手动缩放矩阵。
                            face_enhancer.face_helper.get_inverse_affine(None)
                            # [方案A] 直接传 upsample_img=output（SR 帧）。
                            # facexlib 0.3.0 会将 output resize 到 (720×outscale) = SR 尺寸，
                            # 即无操作；人脸以正确坐标贴回。
                            _ret = face_enhancer.face_helper.paste_faces_to_input_image(
                                upsample_img=output,
                            )
                            if _ret is not None:
                                _res = _ret
                            else:
                                _res = getattr(face_enhancer.face_helper, 'output', None)
                            if _res is not None:
                                if _res.shape[:2] != output.shape[:2]:
                                    import cv2 as _cv2
                                    _res = _cv2.resize(
                                        _res, (output.shape[1], output.shape[0]))
                                if _res.dtype != np.uint8:
                                    _res = np.clip(_res, 0, 255).astype(np.uint8)
                                output = _res
                    except Exception as e:
                        tqdm.write(f'[face_enhance] 帧处理异常，使用 SR 结果: {e}')
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    tqdm.write(f'[OOM] {e}')
                    torch.cuda.empty_cache()
                    continue
                else:
                    tqdm.write(f'[Error] {e}')
                    continue
            writer.write_frame(output)
            meter.update(1)
            pbar.update(1)
            pbar.set_postfix(
                fps=f'{meter.fps():.1f}',
                eta=f'{meter.eta(nb):.0f}s',
            )

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize(device=device)
        except Exception as _sync_e:
            print(f'[Warning] 末尾 synchronize 失败: {_sync_e}')

    # ★ TRT Engine 安全清理：在 CUDA context 仍然存活时显式回收 TRT 对象，
    #   避免 pybind11 在 Python EXIT 时析构 TRT engine 引发 std::terminate()。
    #   此时若 CUDA 已损坏，允许 del 抛异常，由 except 吞掉（exit 路径可以容忍）。
    #   若 CUDA 健康，TRT engine 被正常释放，exit 时不再有析构错误。
    if gfpgan_trt_accel is not None:
        try:
            _e = gfpgan_trt_accel._engine  # noqa: hold ref
            gfpgan_trt_accel._engine  = None
            gfpgan_trt_accel._context = None
            del _e
            import gc as _gc; _gc.collect()
        except Exception:
            pass  # CUDA 已损坏时析构异常可以忽略；Python EXIT 时 pybind11 会抑制

    reader.close()

    if gfpgan_subprocess is not None:
        gfpgan_subprocess.close()

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
            'fp16':        args.use_fp16,
            'trt':         trt_accel is not None and trt_accel.available,
            'gfpgan_trt':  gfpgan_trt_accel is not None and gfpgan_trt_accel.available,  # [v6.2]
            'nvdec':       HardwareCapability.has_nvdec(),
            'nvenc':       HardwareCapability.best_video_encoder(
                getattr(args, 'video_codec', 'libx264')).endswith('nvenc'),
            'face_enhance': args.face_enhance,
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
# run
# ─────────────────────────────────────────────────────────────────────────────

def _output_is_file(path: str) -> bool:
    """
    Determine whether -o refers to an output *file* or an output *directory*.

    Decision priority:
      1. If the path already exists on disk, trust the filesystem.
      2. If it doesn't exist yet, infer from the path string:
         a. mimetypes recognises it as a video type  → file
         b. os.path.splitext finds a non-empty suffix → file  (catches .mkv, .avi, etc.)
         c. Otherwise                                  → directory
    """
    if osp.exists(path):
        return osp.isfile(path)
    mime, _ = mimetypes.guess_type(path)
    if mime is not None and mime.startswith('video'):
        return True
    _, ext = osp.splitext(path)
    return ext != ''


def run(args):
    args.video_name      = osp.splitext(os.path.basename(args.input))[0]
    # [BUGFIX] Allow -o to be either a directory OR a direct video file path.
    # Use _output_is_file() for format-agnostic detection instead of hard-coding
    # ".mp4", so paths like out.mkv / out.avi / out.mov are handled correctly.
    if _output_is_file(args.output):
        video_save_path = args.output
        out_dir = osp.dirname(osp.abspath(video_save_path))
        os.makedirs(out_dir, exist_ok=True)
        # CRITICAL: normalize args.output to always be a real directory so that
        # downstream code (trt_dir, report path, etc.) never uses the video
        # filename as a directory.
        args.output = out_dir
    else:
        video_save_path  = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4')
    mime                 = mimetypes.guess_type(args.input)[0]
    args.input_type_is_video = mime is not None and mime.startswith('video')

    inference_video_single(args, video_save_path)

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
        description='Real-ESRGAN 视频超分 —— 终极优化版 v6.2（单卡版）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 基础参数
    parser.add_argument('-i',  '--input',            type=str, default='inputs')
    parser.add_argument('-n',  '--model-name',       type=str, default='realesr-animevideov3')
    parser.add_argument('-o',  '--output',           type=str, default='results')
    parser.add_argument('-dn', '--denoise-strength', type=float, default=0.5)
    parser.add_argument('-s',  '--outscale',         type=float, default=4)
    parser.add_argument('--suffix',                  type=str, default='out')
    # 推理参数
    parser.add_argument('-t',  '--tile',             type=int, default=0)
    parser.add_argument('--tile-pad',                type=int, default=10)
    parser.add_argument('--pre-pad',                 type=int, default=0)
    parser.add_argument('--face-enhance',            action='store_true')
    parser.add_argument('--gfpgan-model',            type=str, default='1.4',
                        choices=['1.3', '1.4', 'RestoreFormer'],
                        help='GFPGAN 模型版本（--face-enhance 时生效）。'
                             '1.4=更自然/低质量鲁棒，1.3=与1.4相近，RestoreFormer=Transformer方案。'
                             '本地优先查找 experiments/pretrained_models/ 和 gfpgan/weights/，'
                             '不存在时自动下载。Default: 1.4')
    parser.add_argument('--gfpgan-weight',           type=float, default=0.5,
                        help='GFPGAN 增强融合权重，0.0=不增强，1.0=完全替换，Default: 0.5')
    parser.add_argument('--gfpgan-batch-size',       type=int, default=12,
                        help='[OOM-FIX] 单次 GFPGAN 前向最多处理的人脸数。'
                             '人脸密集视频（群像/演唱会）可能每批超过 30 张脸，'
                             '全部堆叠为一次 StyleGAN2 前向会触发 OOM；'
                             '此参数将其拆分为多次子批量，OOM 时自动对半降级。'
                             'A10(24G) 建议 8~12，T4(16G) 建议 4~8。Default: 8')
    parser.add_argument('--no-fp16',                 action='store_true',
                        help='禁用 FP16（默认开启 FP16）')
    parser.add_argument('--fps',                     type=float, default=None)
    # [FIX-5] 默认 batch-size 16（原来 8），充分利用 T4 15G 显存
    parser.add_argument('--batch-size',              type=int, default=16,
                        help='批处理大小，T4 15G 建议 16~24（重模型）或 24~32（轻模型）')
    # [FIX-5] 默认 prefetch-factor 48（原来 16）
    parser.add_argument('--prefetch-factor',         type=int, default=48,
                        help='读帧预取队列深度，建议 ≥ batch-size*3')
    parser.add_argument('--no-compile',             action='store_true',
                        help='禁用 torch.compile（默认开启；短视频或调试时可禁用跳过编译等待）。'
                             '注意：--use-tensorrt 激活时 compile 自动禁用。')
    # V5 新参数（保留）
    parser.add_argument('--use-tensorrt',            action='store_true',
                        help='[V5] 启用 SR TensorRT 加速（首次需要构建 Engine）。'
                             '仅控制 SR 推理，不影响 GFPGAN TRT（由 --gfpgan-trt 独立控制）。'
                             '注意：SR TRT 与 GFPGAN TRT 不可同时使用（Myelin 内存冲突）。'
                             '与 torch.compile 互斥，TRT 优先级更高。')
    parser.add_argument('--gfpgan-trt',              action='store_true',
                        help='[v6.2] 为 GFPGAN 人脸增强单独启用 TensorRT 加速。'
                             '需同时指定 --face-enhance。'
                             '兼容性限制（在 T4 等 16G GPU 上）：'
                             '(1) 不可与 --use-tensorrt 同时使用（Myelin 内存冲突）；'
                             '(2) 不可与 torch.compile 同时使用（CUDA Graph OOM 会损坏 CUDA 上下文）；'
                             '推荐组合：--no-tensorrt --no-compile --gfpgan-trt --face-enhance。'
                             '启用后会自动禁用 torch.compile。')
    parser.add_argument('--use-hwaccel',             action='store_true', default=True,
                        help='[V5] 启用 NVDEC 硬件解码（自动探测，失败时回退）')
    parser.add_argument('--no-hwaccel',              action='store_true',
                        help='[V5] 强制禁用 NVDEC 硬件解码')
    parser.add_argument('--no-cuda-graph',           action='store_true',
                        help='禁用 CUDA Graph（默认开启；compile/TRT 激活时自动禁用）')
    # 编码参数
    parser.add_argument('--video-codec',             type=str, default='libx264',
                        choices=['libx264', 'libx265', 'libvpx-vp9'],
                        help='偏好编码器（有 NVENC 时自动升级为 h264_nvenc/hevc_nvenc）')
    parser.add_argument('--crf',                     type=int, default=23)
    parser.add_argument('--ffmpeg-bin',              type=str, default='ffmpeg')
    # 模型 alpha 通道 / 扩展名
    parser.add_argument('--alpha-upsampler',         type=str, default='realesrgan',
                        choices=['realesrgan', 'bicubic'])
    parser.add_argument('--ext',                     type=str, default='auto',
                        choices=['auto', 'jpg', 'png'])
    # 报告
    parser.add_argument('--report',                  type=str, default=None,
                        help='输出 JSON 性能报告路径（如 report.json）')
    parser.add_argument('--trt-cache-dir',           type=str, default=None,
                        help='TRT Engine 缓存目录（覆盖默认 output/.trt_cache）')
    # ── 高优先级覆盖开关（覆盖 config 注入 / 上方默认值，优先级高于对应正向参数）──
    parser.add_argument('--no-tensorrt', dest='no_tensorrt',
                        action='store_true', default=False,
                        help='[覆盖] 强制禁用 SR TensorRT，覆盖 --use-tensorrt / 外部 config 注入。'
                             '不影响 --gfpgan-trt（GFPGAN TRT 由其自身标志单独控制）。')
    parser.add_argument('--use-compile', dest='use_compile_force',
                        action='store_true', default=False,
                        help='[覆盖] 强制启用 torch.compile，覆盖 --no-compile / config。'
                             '与 --use-tensorrt 互斥（TRT 优先）。')
    parser.add_argument('--use-cuda-graph', dest='use_cuda_graph_force',
                        action='store_true', default=False,
                        help='[覆盖] 强制启用 CUDA Graph，覆盖 --no-cuda-graph / config。'
                             '与 compile/TRT 互斥（compile/TRT 优先）。'
                             '如需确保生效，请同时指定 --no-compile --no-tensorrt。')

    args = parser.parse_args()

    # ── 从 --no-compile / --no-cuda-graph 初始派生（必须在 config 注入之前）──────
    # config 注入可能将 use_cuda_graph/use_compile 覆盖为 config 中设定的值；
    # 因此派生需要先完成，之后 config 注入的值才是最终有效值。
    args.use_compile    = not args.no_compile
    args.use_cuda_graph = not args.no_cuda_graph

    # ── default_config.json 注入（优先级：CLI 显式参数 > config > argparse 默认值）──
    # 搜索路径：脚本所在目录 → 当前工作目录
    # 原理：先 set_defaults() 注入 config 值（替换 argparse 默认值），
    #       再 parse_args()（CLI 参数仍可覆盖 config）。
    # 此处 args 已解析，用 setattr 补全"未在 CLI 中显式提供"的参数。
    # 检测"是否显式提供"：重新解析 sys.argv 获取 explicitly_set 集合。
    # config 搜索路径（按优先级从高到低）：
    # 1. 脚本同级目录（直接调用时）
    # 2. CWD/config/（标准项目结构：python ... 从项目根运行）
    # 3. CWD 根目录（扁平结构兜底）
    # 4. 脚本上两级/config/（external/Real-ESRGAN/ → ../../config/）
    # 5. 脚本上两级根目录（终极兜底）
    _script_dir = osp.dirname(osp.abspath(__file__))
    _cwd        = os.getcwd()
    _proj_root  = osp.normpath(osp.join(_script_dir, '..', '..'))
    _config_candidates = [
        osp.join(_script_dir,  'default_config.json'),
        osp.join(_cwd,         'config', 'default_config.json'),
        osp.join(_cwd,         'default_config.json'),
        osp.join(_proj_root,   'config', 'default_config.json'),
        osp.join(_proj_root,   'default_config.json'),
    ]
    _cfg: dict = {}
    for _cfg_path in _config_candidates:
        if osp.exists(_cfg_path):
            try:
                with open(_cfg_path, 'r', encoding='utf-8') as _f:
                    _cfg = json.load(_f)
                print(f'[Config] 已加载 default_config.json: {_cfg_path}')
            except Exception as _ce:
                print(f'[Config] 加载失败（{_cfg_path}）: {_ce}')
            break

    if _cfg:
        # ── 取 models.realesrgan 子节（本脚本所用配置的实际位置）──────────────
        _esr_cfg: dict = _cfg.get('models', {}).get('realesrgan', {})

        # ── 找出 CLI 中显式出现的 dest 集合（--foo-bar → foo_bar）───────────────
        # 用"重新 parse_known_args + SUPPRESS 默认值"的方式：
        #   · SUPPRESS 让未出现的参数不进入 namespace，
        #     因此 vars() 里有的 key 即为 CLI 显式传入的参数。
        #   · store_true / store_false action 需单独处理（nargs=0 不合法），
        #     改用 nargs='?' 模拟，仅用于检测"是否出现"，不关心值。
        import sys as _sys
        _tmp_parser = argparse.ArgumentParser(add_help=False)
        for _act in parser._actions:
            if _act.dest == 'help' or not _act.option_strings:
                continue
            if isinstance(_act, (argparse._StoreTrueAction,
                                  argparse._StoreFalseAction,
                                  argparse._StoreConstAction)):
                _tmp_parser.add_argument(*_act.option_strings, dest=_act.dest,
                                         nargs='?', default=argparse.SUPPRESS)
            else:
                _tmp_parser.add_argument(*_act.option_strings, dest=_act.dest,
                                         nargs=_act.nargs, default=argparse.SUPPRESS)
        _tmp_ns, _ = _tmp_parser.parse_known_args(_sys.argv[1:])
        _explicitly_set: set = set(vars(_tmp_ns).keys())

        # ── 映射表：(JSON key in models.realesrgan) → args dest ─────────────
        _key_map: dict = {
            'gfpgan_weight'     : 'gfpgan_weight',
            'gfpgan_model'      : 'gfpgan_model',
            'gfpgan_batch_size' : 'gfpgan_batch_size',
            'use_gfpgan_trt'    : 'gfpgan_trt',      # JSON key → argparse dest
            'batch_size'        : 'batch_size',
            'denoise_strength'  : 'denoise_strength',
            'prefetch_factor'   : 'prefetch_factor',
            'model_name'        : 'model_name',
            'use_tensorrt'      : 'use_tensorrt',
            'face_enhance'      : 'face_enhance',
            'tile_size'         : 'tile',             # JSON 用 tile_size，dest 是 tile
            'tile_pad'          : 'tile_pad',
            'crf'               : 'crf',
            'video_codec'       : 'video_codec',
            'use_cuda_graph'    : 'use_cuda_graph',
        }

        _applied = []
        for _jk, _dest in _key_map.items():
            if _jk not in _esr_cfg:
                continue          # config 里没有这个字段，跳过
            if _dest in _explicitly_set:
                continue          # CLI 已显式提供，config 不覆盖
            _val = _esr_cfg[_jk]
            setattr(args, _dest, _val)
            _applied.append(f'{_dest}={_val}')

        # use_fp16 → no_fp16 反向映射（JSON use_fp16=false → no_fp16=True）
        if 'use_fp16' in _esr_cfg and 'no_fp16' not in _explicitly_set:
            _no_fp16_val = not bool(_esr_cfg['use_fp16'])
            setattr(args, 'no_fp16', _no_fp16_val)
            _applied.append(f'no_fp16={_no_fp16_val} (from use_fp16={_esr_cfg["use_fp16"]})')

        if _applied:
            print(f'[Config] 注入 models.realesrgan 参数（未被 CLI 覆盖）:')
            for _a in _applied:
                print(f'         · {_a}')
        else:
            print('[Config] models.realesrgan 所有参数均已被 CLI 覆盖，config 值不注入')

    args.input = args.input.rstrip('/\\')
    # [BUGFIX] Only pre-create args.output as a directory when it isn't a file path.
    if not _output_is_file(args.output):
        os.makedirs(args.output, exist_ok=True)

    # ── 从 --no-fp16 派生 use_fp16 ──────────────────────────────────────────────
    args.use_fp16 = not args.no_fp16
    # ★ use_compile / use_cuda_graph 的初始派生已移至 config 注入之前（见上方），
    #    此处不再重复推导，避免覆盖 config 注入的值。

    # ── 高优先级覆盖参数解析（在冲突仲裁之前执行，确保仲裁基于最终有效值）────
    _override_msgs = []
    if args.no_tensorrt and args.use_tensorrt:
        args.use_tensorrt = False
        _override_msgs.append('--no-tensorrt    覆盖了  --use-tensorrt  → TensorRT 已禁用')
    elif args.no_tensorrt:
        args.use_tensorrt = False
    if args.use_compile_force:
        args.use_compile = True
        if args.no_compile:
            _override_msgs.append('--use-compile    覆盖了  --no-compile   → torch.compile 已启用')
    if args.use_cuda_graph_force:
        args.use_cuda_graph = True
        if args.no_cuda_graph:
            _override_msgs.append('--use-cuda-graph 覆盖了  --no-cuda-graph → CUDA Graph 已启用')
    if _override_msgs:
        print('[CLI覆盖] 以下设置已被高优先级参数覆盖：')
        for msg in _override_msgs:
            print(f'          · {msg}')
        print()

    # ── 派生 use_gfpgan_trt + 互斥仲裁 ──────────────────────────────────────────
    args.use_gfpgan_trt = getattr(args, 'gfpgan_trt', False)

    # 互斥1：SR TRT 与 GFPGAN TRT（已通过子进程架构解决内存分配器冲突）
    if args.use_tensorrt and args.use_gfpgan_trt:
        print('[Info] GFPGAN TRT 已通过独立子进程架构，避免与 SR TRT 的内存分配器冲突。'
              '\n          两者现在可以安全共存，GFPGAN TRT 保持启用。')
        # 不再自动禁用 GFPGAN TRT，两者可以共存

    # 互斥2：torch.compile (reduce-overhead) 与 GFPGAN TRT
    # torch.compile reduce-overhead 内部使用 CUDA Graph capture。
    # CUDA Graph capture 期间若发生 OOM，会损坏 CUDA 上下文（所有后续 CUDA API 均失败）。
    # GFPGAN TRT 的 synchronize 调用在损坏的上下文中会连续崩溃直至进程退出。
    # 解决方案：--gfpgan-trt 激活时自动禁用 torch.compile，改用 FP16-only 路径。
    if args.use_gfpgan_trt and args.use_compile:
        print('[Warning] --gfpgan-trt 与 torch.compile (reduce-overhead) 不兼容。'
              '\n          compile 内部的 CUDA Graph 在 OOM 时会损坏 CUDA 上下文，'
              '\n          导致 GFPGAN TRT synchronize 连续崩溃。'
              '\n          已自动禁用 torch.compile，SR 改用 FP16-only 路径。'
              '\n          如需 compile：去掉 --gfpgan-trt，GFPGAN 走 PyTorch FP16 路径。')
        args.use_compile = False

    # 互斥3：手动 CUDA Graph 与 GFPGAN TRT
    # CUDA Graph capture 期间，GFPGAN TRT warmup 的 synchronize 会触碰 capture 正在监视的
    # stream，导致 cudaErrorStreamCaptureInvalidated → capture 为空 → SR 全帧失败
    # → 输出文件仅含音频（0.1MB）。
    # 注意：config 注入的 use_cuda_graph=true 也会触发此冲突，已通过将派生移到
    # config 注入之前修复（config 注入现在能正确覆盖 use_cuda_graph）。
    if args.use_gfpgan_trt and args.use_cuda_graph:
        print('[Warning] --gfpgan-trt 与 CUDA Graph 不兼容。'
              '\n          GFPGAN TRT warmup 的 synchronize 会在 CUDA Graph capture 期间'
              '\n          触发 cudaErrorStreamCaptureInvalidated → capture 为空 → SR 全失败'
              '\n          已自动禁用 CUDA Graph，SR 改用 FP16-only 路径。')
        args.use_cuda_graph = False

    # ── 冲突仲裁：优先级 TRT > compile > CUDA Graph ───────────────────────────
    if args.use_compile and args.use_tensorrt:
        print('[Warning] --use-tensorrt 与 torch.compile 互斥。'
              '\n          TRT 推理路径完全接管，torch.compile 不会被执行，已自动禁用。'
              '\n          如需 torch.compile，请加 --no-tensorrt。')
        args.use_compile = False

    if args.use_cuda_graph:
        if args.use_tensorrt:
            print('[Warning] CUDA Graph 与 TRT 互斥，TRT 优先，CUDA Graph 已自动禁用。'
                  '\n          如需 CUDA Graph，请加 --no-tensorrt。')
            args.use_cuda_graph = False
        elif args.use_compile:
            print('[Info] torch.compile 已启用（reduce-overhead 模式内含 cudagraphs），'
                  '\n       CUDA Graph 已自动禁用（避免双重 capture）。')
            args.use_cuda_graph = False

    # 处理 --no-hwaccel
    if args.no_hwaccel:
        args.use_hwaccel = False

    # FLV 转 MP4
    mime     = mimetypes.guess_type(args.input)[0]
    is_video = mime is not None and mime.startswith('video')
    if is_video and osp.splitext(args.input)[1].lower() == '.flv':
        mp4_path = args.input.replace('.flv', '.mp4')
        subprocess.run([args.ffmpeg_bin, '-i', args.input, '-codec', 'copy', '-y', mp4_path])
        args.input = mp4_path

    # 启动前打印硬件状态
    print('=' * 60)
    print('  RealESRGAN 视频超分 —— 终极优化版 v6.2（单卡版）')
    print('=' * 60)
    print(f'  GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print(f'  NVDEC:   {HardwareCapability.has_nvdec()} | '
          f'NVENC(h264): {HardwareCapability.has_nvenc("h264_nvenc")} | '
          f'NVENC(hevc): {HardwareCapability.has_nvenc("hevc_nvenc")}')
    _accel_label = ('TensorRT'   if args.use_tensorrt  else
                    'compile'    if args.use_compile   else
                    'CUDA Graph' if args.use_cuda_graph else
                    'FP16-only'  if args.use_fp16       else 'FP32-only')
    print(f'  FP16:     {args.use_fp16}')
    print(f'  推理加速: {_accel_label} | '
          f'TensorRT={args.use_tensorrt} | compile={args.use_compile} | '
          f'CUDA Graph={args.use_cuda_graph}')
    print(f'  batch-size: {args.batch_size} | prefetch: {args.prefetch_factor}')
    print(f'  face-enhance: {args.face_enhance} '
          f'(model={getattr(args, "gfpgan_model", "1.4")} | '
          f'weight={getattr(args, "gfpgan_weight", 0.5)} | '
          f'GFPGAN-batch={getattr(args, "gfpgan_batch_size", 12)} | '
          f'v5++: 批量GFPGAN+原始帧检测+无脸跳过)')
    print(f'  [v5++ 优化] cudnn.benchmark | 异步D2H | TRT专用Stream | 批量GFPGAN | 原始帧检测')
    _gfpgan_trt_flag = getattr(args, 'use_gfpgan_trt', False)
    if args.face_enhance and _gfpgan_trt_flag:
        print(f'  [v6.2 新增] GFPGAN TRT: 静态batch={getattr(args, "gfpgan_batch_size", 12)} + weight='
              f'{getattr(args, "gfpgan_weight", 0.5):.3f} 烘焙'
              f' | Engine tag: gfpgan_{getattr(args, "gfpgan_model", "1.4")}'
              f'_w{getattr(args, "gfpgan_weight", 0.5):.3f}_B{getattr(args, "gfpgan_batch_size", 12)}')
    elif args.face_enhance and args.use_tensorrt and not _gfpgan_trt_flag:
        print(f'  [v6.2] GFPGAN: PyTorch FP16 路径'
              f'（SR TRT 活跃，如需 GFPGAN TRT 请去掉 --use-tensorrt 改用 --gfpgan-trt）')
    print()

    t_total = time.time()
    run(args)
    m, s = divmod(int(time.time() - t_total), 60)
    print(f'\n⏱️  总耗时（含模型加载）: {m}分{s}秒')


if __name__ == '__main__':
    main()
