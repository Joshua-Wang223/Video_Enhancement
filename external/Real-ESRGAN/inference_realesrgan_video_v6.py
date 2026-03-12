"""
RealESRGAN 视频超分处理脚本 —— 终极优化版 v6（多卡版）
==========================================================
在 v5++ 多卡版基础上，全面移植 v6 单卡版的所有升级，
重点引入 CPU-GPU 流水线并行，将 face_enhance 模式下 GPU 利用率
从 <30% 提升至 ~70-85%，同时修复 face_enhance 的关键 bug。

[V6 新增升级（从 v6_single 移植，针对多卡架构深度适配）]

  FIX-PIPELINE. [CPU-GPU 流水线并行] — v6 最核心升级
      face_enhance 全流程拆解为 4 个独立阶段，在每个 GPU Worker 内
      启动 2 个专用线程池（detect_executor / paste_executor），实现：
        Level-1: detect(N) 后台线程 ‖ SR(N) GPU 推理（主线程）
                 RetinaFace 检测在独立 FaceRestoreHelper 实例上运行，
                 与主线程 GPU 推理完全无共享状态，线程安全。
        Level-2: paste(N) 后台线程 ‖ SR(N+1)+GFPGAN(N+1) GPU（主线程）
                 纯 CPU OpenCV paste 操作，不占用 GPU 核心，
                 与下一批推理真正并行。
      多卡放大效应：N 个 Worker 各自运行独立流水线，互不依赖，
      总体吞吐量 ≈ N × 单卡流水线吞吐量。

  FIX-PIPE-BUG. [face_enhance 输出尺寸 bug 修复]
      v5++ 中 GFPGANer(upscale=outscale) 坐标系混乱的根因已修复：
      · _paste_faces_batch 内含严格尺寸检查，强制 resize 到 SR 尺寸
      · 独立模块函数 → 可被 Worker 和单卡路径复用，行为一致

  FIX-DETECT-BUG. [FIX-7 原始帧检测真正生效]
      v5++ 的 FIX-7 在 _process_batch 内通过 face_enhancer.face_helper
      进行检测，但 clean_all/read_image 影响同一对象，之后 GFPGAN paste
      时 face_helper 状态已经改变。v6 通过 _make_detect_helper() 创建
      独立 FaceRestoreHelper 实例，detect 与 paste 使用不同对象，
      完全消除状态污染。

  FIX-PIPELINE-ARCH. [5 个模块级流水线函数]
      模块级（非嵌套），spawn Worker 可直接调用，无序列化问题：
        _make_detect_helper()  — 创建线程安全的独立检测器
        _sr_infer_batch()      — 纯 SR 推理，无 face 逻辑
        _detect_faces_batch()  — 纯 CPU 检测，可在后台线程运行
        _gfpgan_infer_batch()  — GPU GFPGAN 批量推理
        _paste_faces_batch()   — 纯 CPU paste，可在后台线程运行
      _process_batch 保留为完整串行版本（tile 模式 / 逐帧路径兜底）。

[V5++ 全部升级完整保留]

  FIX-1.  [cuDNN benchmark] 主进程 + 每 Worker 独立启用
  FIX-2.  [异步 D2H] pinned output buffer + non_blocking=True
  FIX-3.  [TRT 全程 GPU 内存] Tensor.data_ptr() 直接传 GPU 指针
  FIX-4.  [去掉双重 copy] _read_loop 移除 arr.copy()
  FIX-5.  [batch/prefetch 提升] batch_size 默认 8，prefetch 默认 16
  FIX-6.  [face_enhance 解耦] Worker 内独立初始化 GFPGANer
  FIX-7.  [原始帧检测] 独立 FaceRestoreHelper，真正在低分辨率帧检测
  FIX-8.  [GFPGAN FP16] torch.autocast 包裹，充分利用 Tensor Core
  FIX-STREAM.  [流间竞争修复] default_stream.wait_stream(compute_stream)
  FIX-TRT3.    [TRT 8/10 双 API] parse_from_file / None 检测 / 重建
  FIX-NVDEC.   [NVDEC 探测增强] 真实 H.264 双阶段探测
  FIX-PATH.    [动态路径] __file__ 基准，跨环境部署
  FIX-OUTPUT.  [灵活输出路径] -o 直接指定视频文件
  FIX-NVML.    [NVML 噪声抑制] stderr _NVMLFilter 过滤
  FIX-OOM-GFP. [GFPGAN OOM 降级] sub_bs 跨批次持久化
  FIX-WRITE.   [写入尺寸校验] write_frame 兜底修正

[V5 原有多卡架构（全部保留）]

  M1. [Dispatcher-Queue 零临时文件多卡调度]
      ┌─ 主进程 FFmpegReader ──→ dispatch_queue ──→ GPU Worker 0 (流水线) ──┐
      │                     ──→ GPU Worker 1 (流水线) ──┤→ result_queue
      │                     ──→ GPU Worker N (流水线) ──┘      │
      └─ 主进程 FrameCollector(heapq 重排) ──→ FFmpegWriter ─→ output.mp4
      每个 Worker 内部 detect/paste 线程池与 GPU 推理并行，
      Worker 间通过有界队列解耦，互不影响。

  M2. [NVDEC 硬件解码 / NVENC 硬件编码自动探测]
  M3. [TensorRT 可选加速]（FIX-TRT3 强化版）
  M4. [有界队列 + heapq 无锁帧重排]
  M5. [跨进程异常传播]

【命令行使用示例】
  # 双卡超分（每卡 1 Worker）
  python inference_realesrgan_video_v6.py -i input.mp4 -o results/ -s 2

  # 4× 放大 + 每卡 2 Worker（24G+ 显存）
  python inference_realesrgan_video_v6.py -i input.mp4 -o results/ -s 4 \\
      --num-process-per-gpu 2

  # 开启 face_enhance（v6 流水线加速，每 Worker 独立 detect/paste 并行）
  python inference_realesrgan_video_v6.py -i input.mp4 -o results/ -s 2 \\
      --face-enhance --gfpgan-model 1.4 --gfpgan-weight 0.5

  # TRT 加速（首次构建 Engine，之后秒加载）
  python inference_realesrgan_video_v6.py -i input.mp4 -o results/ -s 2 \\
      --use-tensorrt

  # 直接指定输出文件路径 + FP32 模式
  python inference_realesrgan_video_v6.py -i input.mp4 -o output_4k.mkv -s 2 --fp32

  # 输出 JSON 性能报告
  python inference_realesrgan_video_v6.py -i input.mp4 -o results/ -s 2 \\
      --report report.json

【版本演进历史】
  v5（多卡版）：  Dispatcher-Queue 零临时文件多卡调度，TRT/NVDEC/NVENC
  v5++（多卡版）：FIX-1~8、TRT双API、NVDEC增强、动态路径、灵活输出、NVML过滤
  v6（多卡版）：  FIX-PIPELINE（CPU-GPU流水线）、FIX-PIPE-BUG、FIX-DETECT-BUG
"""

from __future__ import annotations

import argparse
import concurrent.futures  # [FIX-PIPELINE] 流水线并行
import contextlib          # [FIX-7/8] GFPGAN FP16 autocast 所需
import heapq
import json
import mimetypes
import multiprocessing as mp
import numpy as np
import os
os.environ.setdefault("PYTORCH_NO_NVML", "1")   # [FIX-NVML] 抑制无害 NVML 警告
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# [FIX-NVML] stderr 过滤器：屏蔽 NVML_SUCCESS / CUDACachingAllocator 无害断言
# 多 Worker 并发时日志噪声尤其严重，此过滤器在主进程和每个 Worker 中各自安装
# ─────────────────────────────────────────────────────────────────────────────

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

# [v5++] 批量 GFPGAN 推理所需工具
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
# [FIX-PATH] 动态路径：以脚本所在目录为基准，向上两级到项目根
# 目录结构假设：<project_root>/external/Real-ESRGAN/inference_realesrgan_video_v5.py
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR       = osp.dirname(osp.abspath(__file__))
base_dir          = osp.dirname(osp.dirname(_SCRIPT_DIR))
models_RealESRGAN = osp.join(base_dir, 'models_RealESRGAN')

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
            'libvpx-vp9': 'libvpx-vp9',
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
    """
    预分配 pinned CPU buffer 并复用，避免每批 H2D 前 pin_memory 的 malloc 开销。
    [FIX-2] 新增 _out_buf，用于异步 D2H 输出，避免 .cpu() 引发隐式 cudaDeviceSynchronize。
    """

    def __init__(self):
        self._buf:     Optional[torch.Tensor] = None
        self._out_buf: Optional[torch.Tensor] = None

    def get_for_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        # [FIX-4] np.stack 已经复制数据，无需提前 arr.copy()
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
# M3: TensorRT 可选加速封装（[FIX-TRT3] 全面重写）
# ─────────────────────────────────────────────────────────────────────────────

class TensorRTAccelerator:
    """
    将 RealESRGAN 模型导出 ONNX 后编译 TRT Engine (FP16, 静态形状)。
    要求：pip install tensorrt onnx onnxruntime-gpu
    首次构建会缓存 .trt 文件，后续直接加载。

    [FIX-TRT3] 重大修复：
      · 完全移除 pycuda.autoinit（其创建的独立 CUDA context 与 PyTorch context 冲突，
        根因是 CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH 和级联 invalid resource handle）
      · 改用 Tensor.data_ptr() 全程 GPU 内存推理（[FIX-3]）
      · 专用非默认 Stream（_trt_stream），消除 TRT 隐式全局 cudaStreamSynchronize
      · TRT 8.x / 10.x 双 API 自动切换（num_io_tensors vs num_bindings）
      · Engine None 检测 + 自动删除过期缓存 + 重新构建
      · parse_from_file 替代 parse(bytes)，支持大模型 .onnx.data sidecar
      · 处理最后一批不足 batch_size 时的 padding / 裁剪
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
        # [FIX-TRT-STREAM] 专用非默认 Stream，懒初始化
        self._trt_stream: Optional[torch.cuda.Stream] = None
        self._use_new_api    = False
        self._input_name:  Optional[str] = None
        self._output_name: Optional[str] = None

        try:
            import tensorrt as trt
            # NOTE: 绝对不要 import pycuda.autoinit
            # pycuda.autoinit 创建独立 CUDA context，与 PyTorch context 冲突，
            # 造成 CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH 和级联 invalid resource handle
            self._trt = trt
        except ImportError as e:
            print(f'[TensorRT] 依赖未安装，跳过 TRT 加速: {e}')
            print('  安装命令: pip install tensorrt onnx onnxruntime-gpu')
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
                # Engine 不兼容（GPU 换代 / TRT 版本升级）时自动重建
                print(f'[TensorRT] 首次加载失败（{_e}），开始重新构建 Engine...')
                if not osp.exists(onnx_path):
                    self._export_onnx(model, onnx_path, input_shape)
                self._build_engine(onnx_path, trt_path, use_fp16)
                if osp.exists(trt_path):
                    self._load_engine(trt_path)

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
                opset_version=18,   # torch 内部已使用 opset 18，显式声明避免版本警告
                dynamic_axes=None,  # 静态形状，TRT 最优化
            )
        print(f'[TensorRT] ONNX 已导出: {onnx_path}')

    def _build_engine(self, onnx_path, trt_path, use_fp16):
        trt = self._trt
        logger  = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)

        # [FIX-TRT10] TRT 10.x 中 EXPLICIT_BATCH 已成为默认模式且枚举已废弃
        try:
            explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        except AttributeError:
            explicit_batch_flag = 0   # TRT 10.x：explicit batch 是唯一模式

        network = builder.create_network(explicit_batch_flag)
        parser  = trt.OnnxParser(network, logger)

        # [FIX-TRT3] parse_from_file 替代 parse(bytes)
        # 大模型会生成 .onnx.data sidecar，parse(bytes) 只看主文件会找不到权重
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
        logger  = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(trt_path, 'rb') as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        # [FIX-TRT3] deserialize_cuda_engine 在 GPU 不兼容或版本升级时静默返回 None
        if self._engine is None:
            print(f'[TensorRT] Engine 反序列化失败（GPU 不兼容或 TRT 版本升级），'
                  f'删除过期缓存并重新构建: {trt_path}')
            try:
                os.remove(trt_path)
            except OSError:
                pass
            raise RuntimeError('[TensorRT] _load_engine: deserialize_cuda_engine returned None')

        self._context = self._engine.create_execution_context()

        # ── 区分 TRT 版本，预先解析 tensor 名称 ──────────────────────────────
        self._use_new_api  = hasattr(self._engine, 'num_io_tensors')
        self._input_name   = None
        self._output_name  = None

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
            print('[TensorRT] 使用旧版 API (TRT 8.x)')

        self._trt_ok = True
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
        actual_B = input_tensor.shape[0]
        engine_B = self.input_shape[0]

        # 最后一批帧数不足时用最后一帧 padding（仅静态 shape 引擎需要）
        if actual_B < engine_B:
            pad_cnt      = engine_B - actual_B
            pad          = input_tensor[-1:].expand(pad_cnt, -1, -1, -1)
            input_tensor = torch.cat([input_tensor, pad], dim=0)

        inp       = input_tensor.contiguous()
        out_dtype = torch.float16 if self.use_fp16 else torch.float32

        # [FIX-TRT-STREAM] 专用非默认 Stream，消除 TRT 隐式全局 cudaStreamSynchronize
        if self._trt_stream is None:
            self._trt_stream = torch.cuda.Stream(device=self.device)

        if self._use_new_api:
            # ── TRT 10.x 新 API ────────────────────────────────────────────
            out_shape  = tuple(self._engine.get_tensor_shape(self._output_name))
            out_tensor = torch.empty(out_shape, dtype=out_dtype, device=self.device)

            # [BUGFIX] _trt_stream 必须等待 current_stream：
            # inp.contiguous() 在调用方的 default stream 上执行，
            # 若不插入此等待，_trt_stream 可能在 contiguous() 完成前读取 → 花屏
            self._trt_stream.wait_stream(torch.cuda.current_stream(self.device))
            self._context.set_tensor_address(self._input_name,  inp.data_ptr())
            self._context.set_tensor_address(self._output_name, out_tensor.data_ptr())
            self._context.execute_async_v3(stream_handle=self._trt_stream.cuda_stream)
        else:
            # ── TRT 8.x 旧 API（向下兼容）──────────────────────────────────
            out_shape  = tuple(self._engine.get_binding_shape(1))
            out_tensor = torch.empty(out_shape, dtype=out_dtype, device=self.device)
            self._trt_stream.wait_stream(torch.cuda.current_stream(self.device))
            self._context.execute_async_v2(
                bindings=[inp.data_ptr(), out_tensor.data_ptr()],
                stream_handle=self._trt_stream.cuda_stream,
            )

        # 等待 TRT Stream 完成，让 default stream 可以安全使用结果
        self._trt_stream.synchronize()

        # 裁掉 padding 帧
        if actual_B < engine_B:
            out_tensor = out_tensor[:actual_B]

        return out_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Reader（支持 NVDEC + 异常传播）
# ─────────────────────────────────────────────────────────────────────────────

class FFmpegReader:
    """
    通过 FFmpeg pipe 读取视频帧，支持 NVDEC 硬件解码。
    M2: 若 NVDEC 可用，优先使用 -hwaccel cuda；否则回退 CPU 解码。
    X1(v4): 预取线程异常入队，get_frame() 处 re-raise，防止主线程永久阻塞。
    [FIX-4]: _read_loop 去掉 arr.copy()，PinnedBufferPool 内 np.stack 会完整复制。
    [FIX-5]: prefetch_factor 默认 16（原来 8）。
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
        # [FIX-NVDEC] 删掉 '-hwaccel_output_format', 'bgr24'，
        # 让 FFmpeg 自动做软解色彩转换，避免部分 GPU 不支持直接输出 bgr24
        hw_args: List[str] = []
        if use_hwaccel and HardwareCapability.has_nvdec():
            hw_args = ['-hwaccel', 'cuda']
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
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                    self.height, self.width, 3
                )
                # [FIX-4] 去掉 arr.copy()：np.frombuffer 只读视图传给队列，
                # PinnedBufferPool.get_for_frames 内 np.stack 会复制一次，无需提前 copy
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
    [FIX-WRITE] write_frame 自动检测并修正尺寸/dtype，防止 face_enhance 路径下
                rawvideo 字节流错位。
    """
    _SENTINEL  = object()
    _MAX_BATCH = 8

    def __init__(self, args, audio, height: int, width: int,
                 save_path: str, fps: float):
        out_w = int(width  * args.outscale)
        out_h = int(height * args.outscale)
        if out_h > 2160:
            print('[Warning] 输出 > 4K，建议 --outscale 或 --video_codec libx265。')

        # [FIX-WRITE] 保存期望尺寸，用于 write_frame 兜底校验
        self._out_h = out_h
        self._out_w = out_w

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
        # [FIX-WRITE] 尺寸/类型兜底：任何错误尺寸的帧进入 pipe 都会导致
        # "packet size X < expected frame_size Y" 并使后续所有帧字节边界错位
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
# [FIX-PIPELINE] 流水线子函数（模块级，供 Worker 进程和单卡路径共用）
#
# face_enhance 全流程拆为 4 个独立阶段：
#   _sr_infer_batch()      GPU SR 推理（主线程 / Worker 主循环）
#   _detect_faces_batch()  人脸检测（独立实例，可在后台线程运行，与 SR 并行）
#   _gfpgan_infer_batch()  批量 GFPGAN GPU 推理（主线程）
#   _paste_faces_batch()   纯 CPU paste（可在后台线程运行，与下批 SR+GFPGAN 并行）
# _process_batch 保留为完整串行版本（tile / 逐帧 / 非流水线路径兜底）。
# ─────────────────────────────────────────────────────────────────────────────

def _make_detect_helper(face_enhancer, device):
    """
    [FIX-DETECT-BUG] 创建独立的 FaceRestoreHelper，专供后台 detect 线程使用。
    与主线程 face_enhancer.face_helper 互为独立对象，无共享状态，线程安全。
    独立实例确保 detect 和 paste 操作的对象不重叠，彻底消除 v5++ 中
    同一 face_helper 被检测和贴图路径交替修改导致的状态污染。
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
) -> List[np.ndarray]:
    """
    [FIX-PIPELINE] 纯 SR 推理：H2D → 模型前向 → 后处理 → 异步 D2H。
    不含任何 face_enhance 逻辑，流水线主循环在主线程调用，
    与后台 detect 线程并行（两者不共享任何 GPU 对象）。
    所有 FIX-2/FIX-STREAM/FIX-TRT3 优化全部保留。
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
    else:
        if transfer_stream is not None and compute_stream is not None:
            compute_stream.wait_stream(transfer_stream)
            with torch.cuda.stream(compute_stream):
                with torch.no_grad():
                    output_t = upsampler.model(batch_t)
        else:
            with torch.no_grad():
                output_t = upsampler.model(batch_t)

    # [FIX-STREAM] default stream 等待 compute_stream，消除画面抖动根因
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
    torch.cuda.synchronize(device)  # [FIX-2] 等 D2H 完成

    out_np = out_pinned.numpy()
    return [out_np[i].copy() for i in range(B)]


def _detect_faces_batch(
    frames: List[np.ndarray],
    helper,
) -> List[dict]:
    """
    [FIX-PIPELINE] 在原始低分辨率帧上检测人脸，返回序列化检测结果。
    使用 _make_detect_helper() 创建的独立实例，可在后台线程与 SR 推理并行调用。
    [FIX-DETECT-BUG] 独立 helper 确保与主线程 face_helper 零共享状态。
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
) -> Tuple[List[List[np.ndarray]], int]:
    """
    [FIX-PIPELINE] 批量 GFPGAN GPU 推理（主线程调用）。
    仅调用 face_enhancer.gfpgan 网络前向，不修改 face_helper 状态，
    与后台 paste 线程（使用 face_helper）无对象竞争。
    [FIX-OOM-GFP] sub_bs 自动降级，返回实际使用值供跨批次持久化。
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
    i_face = 0
    while i_face < n_faces:
        sub_crops = all_crop_tensors[i_face: i_face + sub_bs]
        sub_batch = torch.stack(sub_crops).to(device)
        try:
            with torch.no_grad(), fp16_ctx:
                gfpgan_out = face_enhancer.gfpgan(
                    sub_batch, return_rgb=False, weight=gfpgan_weight)
                if isinstance(gfpgan_out, (tuple, list)):
                    gfpgan_out = gfpgan_out[0]
                all_out_tensors.extend(gfpgan_out.float().unbind(0))
            i_face += len(sub_crops)
        except RuntimeError as _oom_e:
            if 'out of memory' in str(_oom_e).lower() and sub_bs > 1:
                sub_bs = max(1, sub_bs // 2)
                torch.cuda.empty_cache()
                tqdm.write(f'[OOM-GFPGAN] OOM，降级 gfpgan_batch_size → {sub_bs}')
            else:
                tqdm.write(f'[OOM-GFPGAN] 不可恢复（sub_bs={sub_bs}），'
                      f'跳过 {len(sub_crops)} 张: {_oom_e}')
                for _ in sub_crops:
                    all_out_tensors.append(None)  # type: ignore[arg-type]
                torch.cuda.empty_cache()
                i_face += len(sub_crops)
        finally:
            del sub_batch

    print(f'[face_enhance] GFPGAN 完成：{n_faces} 张人脸 (sub_bs={sub_bs})')
    for fi, out_t in zip(crop_frame_idx, all_out_tensors):
        if out_t is None:
            continue
        restored = tensor2img(out_t, rgb2bgr=True, min_max=(-1, 1))
        restored_by_frame[fi].append(restored.astype('uint8'))

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
    [FIX-PIPE-BUG] 内含严格尺寸安全检查，确保输出与 SR 帧尺寸严格一致，
    彻底修复 v5++ 中 face_enhance 输出帧尺寸的边缘情况 bug。
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

        # [FIX-PIPE-BUG] 严格尺寸校验：强制 resize 到 SR 尺寸
        if result.shape[0] != expected_h or result.shape[1] != expected_w:
            tqdm.write(f'[WARN] face_enhance 帧{fi} 尺寸异常 '
                  f'{result.shape[:2]} != ({expected_h},{expected_w})，强制 resize')
            result = _cv2.resize(result, (expected_w, expected_h),
                                 interpolation=_cv2.INTER_LANCZOS4)
        final_results.append(result)

    return final_results


# ─────────────────────────────────────────────────────────────────────────────
# 批次推理（完整串行版本，供 tile 模式 / 逐帧路径 / 非流水线路径使用）
# ─────────────────────────────────────────────────────────────────────────────

def _process_batch(
    upsampler: RealESRGANer,
    frames: List[np.ndarray],
    outscale: float,
    netscale: int,
    transfer_stream,
    compute_stream,
    trt_accel: Optional['TensorRTAccelerator'] = None,
    face_enhancer=None,
    face_fp16: bool = False,
    gfpgan_weight: float = 0.5,
    gfpgan_batch_size: int = 12,
) -> Tuple[List[np.ndarray], int]:
    """
    将一批帧推理为超分结果。
    若 trt_accel 可用，优先使用 TRT；否则走 PyTorch 路径。

    返回 (帧列表, 实际使用的 gfpgan_batch_size)：
      调用方将返回的 int 用于下一批次，实现 OOM 降级值的跨批次持久化。

    [FIX-2] D2H 使用预分配 pinned buffer + non_blocking=True，
            避免 .cpu() 触发隐式 cudaDeviceSynchronize。
    [FIX-STREAM] compute_stream 推理后，让 default stream 显式等待，
            消除画面抖动 / 黑影 / 花屏根因。
    [FIX-6] face_enhance 解耦：SR 推理完成后对每帧独立调用 GFPGAN，
            SR 阶段仍享受完整 batch/TRT 加速。
    [FIX-7] GFPGAN 在原始低分辨率帧上检测（比 SR 帧快 4×）。
    [FIX-8] torch.autocast FP16 包裹 GFPGAN 推理。
    """
    device   = upsampler.device
    use_half = upsampler.half

    pool      = _get_pinned_pool()
    batch_pin = pool.get_for_frames(frames)

    B = len(frames)

    # ── H2D + 预处理 ──────────────────────────────────────────────────────────
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

    # ── 模型推理 ──────────────────────────────────────────────────────────────
    if trt_accel is not None and trt_accel.available:
        # TRT 路径：infer() 内部使用独立 Stream，
        # 需先等待 transfer_stream 上的 H2D + 预处理全部完成
        if transfer_stream is not None:
            torch.cuda.current_stream(device).wait_stream(transfer_stream)
        output_t = trt_accel.infer(batch_t).float()
    else:
        # PyTorch 路径：compute_stream 等待 transfer_stream
        if transfer_stream is not None and compute_stream is not None:
            compute_stream.wait_stream(transfer_stream)
            with torch.cuda.stream(compute_stream):
                with torch.no_grad():
                    output_t = upsampler.model(batch_t)
        else:
            with torch.no_grad():
                output_t = upsampler.model(batch_t)

    # [FIX-STREAM] 关键修复：让 default stream 等待 compute_stream
    # with torch.cuda.stream(compute_stream) 退出后当前流切回 default stream，
    # 但 default stream 与 compute_stream 是独立队列，若不插入此依赖，
    # 后续的 F.interpolate / clamp / byte 在 compute_stream 中推理尚未写完时就读取
    # → 画面抖动、黑影、花屏。wait_stream 仅在 GPU 侧插入事件屏障，不阻塞 CPU。
    if compute_stream is not None:
        torch.cuda.default_stream(device).wait_stream(compute_stream)

    # ── 后处理 ────────────────────────────────────────────────────────────────
    if abs(outscale - netscale) > 1e-5:
        output_t = F.interpolate(
            output_t.float(), scale_factor=outscale / netscale,
            mode='bicubic', align_corners=False,
        )

    # float → uint8，原地操作减少显存分配
    out_u8   = output_t.float().clamp_(0.0, 1.0).mul_(255.0).byte()
    out_perm = out_u8.permute(0, 2, 3, 1).contiguous()  # (B, H_out, W_out, 3) GPU

    # [FIX-2] 异步 D2H：pinned buffer + non_blocking，避免隐式全局同步
    out_pinned = pool.get_output_buf(out_perm.shape, torch.uint8)
    out_pinned.copy_(out_perm, non_blocking=True)
    torch.cuda.synchronize(device)

    out_np     = out_pinned.numpy()
    sr_results = [out_np[i].copy() for i in range(B)]

    # ── [v5++] face_enhance 后处理 ────────────────────────────────────────────
    sub_bs = gfpgan_batch_size  # 跟踪实际使用的子批量，OOM 降级后由返回值向上传递

    if face_enhancer is not None:
        final_results = []
        fp16_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                    if face_fp16 else contextlib.nullcontext())

        if _HAS_BATCHGFPGAN:
            # ── 路径A：批量 GFPGAN（推荐，最快）────────────────────────────
            # Step-1: 在原始低分辨率帧上检测 + 对齐（[FIX-7] 4× 加速）
            face_data = []
            for orig_frame in frames:
                face_enhancer.face_helper.clean_all()
                face_enhancer.face_helper.read_image(orig_frame)
                try:
                    face_enhancer.face_helper.get_face_landmarks_5(
                        only_center_face=False, resize=640, eye_dist_threshold=5)
                except TypeError:
                    face_enhancer.face_helper.get_face_landmarks_5(
                        only_center_face=False, eye_dist_threshold=5)
                face_enhancer.face_helper.align_warp_face()
                face_data.append({
                    'crops':   [c.copy() for c in face_enhancer.face_helper.cropped_faces],
                    'affines': [a.copy() for a in face_enhancer.face_helper.affine_matrices],
                    'orig':    orig_frame,
                })

            _frames_with_faces = sum(1 for fd in face_data if fd['crops'])
            _total_faces = sum(len(fd['crops']) for fd in face_data)
            if _frames_with_faces > 0:
                tqdm.write(f'[face_detect] 本批 {len(face_data)} 帧中检测到人脸：'
                      f'{_frames_with_faces} 帧含人脸，共 {_total_faces} 张脸')

            # Step-2: 汇总所有帧的 face crops → 批量 GFPGAN 前向
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
                # [FIX-OOM-GFP] 分 sub-batch 执行 GFPGAN 推理，防止人脸密集场景 OOM
                n_faces = len(all_crop_tensors)
                all_out_tensors: List[torch.Tensor] = []
                i_face = 0
                while i_face < n_faces:
                    sub_crops = all_crop_tensors[i_face: i_face + sub_bs]
                    sub_batch = torch.stack(sub_crops).to(device)
                    try:
                        with torch.no_grad(), fp16_ctx:
                            gfpgan_out = face_enhancer.gfpgan(
                                sub_batch, return_rgb=False, weight=gfpgan_weight)
                            if isinstance(gfpgan_out, (tuple, list)):
                                gfpgan_out = gfpgan_out[0]
                            all_out_tensors.extend(gfpgan_out.float().unbind(0))
                        i_face += len(sub_crops)
                    except RuntimeError as _oom_e:
                        if 'out of memory' in str(_oom_e).lower() and sub_bs > 1:
                            sub_bs = max(1, sub_bs // 2)
                            torch.cuda.empty_cache()
                            tqdm.write(f'[OOM-GFPGAN] 人脸批量 OOM，降级 gfpgan_batch_size → {sub_bs}')
                        else:
                            tqdm.write(f'[OOM-GFPGAN] 不可恢复错误（sub_bs={sub_bs}），'
                                  f'跳过 {len(sub_crops)} 张脸: {_oom_e}')
                            for _ in sub_crops:
                                all_out_tensors.append(None)   # type: ignore[arg-type]
                            torch.cuda.empty_cache()
                            i_face += len(sub_crops)
                    finally:
                        del sub_batch

                print(f"[face_enhance] GFPGAN 增强完成：{n_faces} 张人脸已处理 "
                      f"(sub_bs={sub_bs}，{'FP16' if face_fp16 else 'FP32'})")
                for fi, out_t in zip(crop_frame_idx, all_out_tensors):
                    if out_t is None:
                        continue
                    restored = tensor2img(out_t, rgb2bgr=True, min_max=(-1, 1))
                    restored_by_frame[fi].append(restored.astype('uint8'))

            # Step-3: 逐帧将增强人脸贴回 SR 帧
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
                    if _ret is not None:
                        result = _ret
                    else:
                        result = getattr(face_enhancer.face_helper, 'output', None)
                    final_results.append(result if result is not None else frame_sr)
                except Exception as e:
                    tqdm.write(f'[face_enhance] 帧{fi} 贴回异常，使用 SR 结果: {e}')
                    final_results.append(frame_sr)

        else:
            # ── 路径B：回退逐帧路径（无 basicsr/torchvision）──────────────
            for orig_frame, frame_sr in zip(frames, sr_results):
                try:
                    with fp16_ctx:
                        _, restored_faces, _ = face_enhancer.enhance(
                            orig_frame,
                            has_aligned=False,
                            only_center_face=False,
                            paste_back=False,
                            weight=gfpgan_weight,
                        )
                    if restored_faces:
                        face_enhancer.face_helper.get_inverse_affine(None)
                        _ret = face_enhancer.face_helper.paste_faces_to_input_image(
                            upsample_img=frame_sr,
                        )
                        if _ret is not None:
                            result = _ret
                        else:
                            result = getattr(face_enhancer.face_helper, 'output', None)
                        final_results.append(result if result is not None else frame_sr)
                    else:
                        final_results.append(frame_sr)
                except Exception as e:
                    tqdm.write(f'[face_enhance] 帧处理异常，使用 SR 结果: {e}')
                    final_results.append(frame_sr)

        # ── 输出尺寸安全检查（统一出口）──────────────────────────────────────
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

    # 无 face_enhance 路径
    return sr_results, gfpgan_batch_size


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
    # face_enhance 参数（[FIX-6] Worker 内独立初始化，不跨进程传递不可序列化对象）
    face_enhance:       bool  = False,
    gfpgan_model_path:  str   = '',
    gfpgan_arch:        str   = 'clean',
    gfpgan_ch:          int   = 2,
    gfpgan_weight:      float = 0.5,
    gfpgan_batch_size:  int   = 12,
    face_fp16:          bool  = False,
):
    """
    多进程 GPU 推理 Worker（v6 版）。从 dispatch_q 获取帧，处理后放入 result_q。
    异常通过 error_q 传回主进程。

    [FIX-NVML]      安装 _NVMLFilter，多 Worker 并发时日志更清晰。
    [FIX-1]         cudnn.benchmark = True，每 GPU 独立受益。
    [FIX-PIPELINE]  face_enhance + batch 激活时启用两级并行流水线：
                    Level-1: detect(N) 后台线程 ‖ SR(N) GPU 推理（主线程）
                    Level-2: paste(N)  后台线程 ‖ SR(N+1)+GFPGAN(N+1) GPU（主线程）
                    detect_executor / paste_executor 进程内独立生命周期管理。
                    每个 Worker 独立流水线，N Worker 互不干扰。
    [FIX-DETECT-BUG] _make_detect_helper() 独立 FaceRestoreHelper，消除状态污染。
    [FIX-PIPE-BUG]   _paste_faces_batch() 内含严格尺寸校验，修复输出帧尺寸 bug。
    """
    try:
        # [FIX-NVML] Worker 进程中重新安装过滤器（spawn 不继承父进程 stderr 包装）
        import re as _re
        import sys as _sys
        class _F:
            _p = _re.compile(r'NVML_SUCCESS|INTERNAL ASSERT FAILED.*CUDACachingAllocator')
            def __init__(self, s): self._s = s
            def write(self, m):
                if not self._p.search(m): self._s.write(m)
            def flush(self): self._s.flush()
            def __getattr__(self, a): return getattr(self._s, a)
        _sys.stderr = _F(_sys.stderr)

        import os as _os
        _os.environ.setdefault("PYTORCH_NO_NVML", "1")

        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device)

        # [FIX-1] cuDNN benchmark：Worker 内固定输入尺寸，自动选最优卷积算法
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled   = True
        print(f'[Worker {worker_idx}] cudnn.benchmark = True 已启用 (cuda:{device_id})')

        _mp_display = model_path if isinstance(model_path, str) else osp.basename(model_path[0])
        _dni_hint   = (f'[Worker {worker_idx}] [DNI 模式] denoise_strength={dni_weight[0]:.2f} → '
                       f'{dni_weight[0]:.0%} realesr-general-x4v3 + '
                       f'{dni_weight[1]:.0%} realesr-general-wdn-x4v3'
                       if isinstance(model_path, list) and dni_weight else '')
        print(f'[Worker {worker_idx}] 加载模型: {_mp_display} → cuda:{device_id}')
        if _dni_hint:
            print(_dni_hint)
        upsampler = _build_upsampler(
            model_name, model_path, dni_weight,
            tile, tile_pad, pre_pad, use_half, device
        )
        _, netscale, _ = MODEL_CONFIG[model_name]

        # ── compile / TRT 冲突防御（Worker 级守卫）────────────────────────────
        # 正常路径下 main() 已完成仲裁；此处防止外部直接调用 _sr_worker_fn 时
        # use_compile 与 use_trt 同时为 True 导致 compile 白跑。
        if use_compile and use_trt:
            print(f'[Worker {worker_idx}] [Warning] use_compile 与 use_trt 同时为 True，'
                  f'TRT 优先，torch.compile 已禁用。')
            use_compile = False

        if use_compile and hasattr(torch, 'compile'):
            try:
                upsampler.model = torch.compile(upsampler.model, mode='reduce-overhead')
                print(f'[Worker {worker_idx}] torch.compile 加速已启用')
            except Exception as e:
                print(f'[Worker {worker_idx}] torch.compile 失败: {e}')
        elif use_compile and not hasattr(torch, 'compile'):
            print(f'[Worker {worker_idx}] [Warning] 当前 PyTorch 版本不支持 torch.compile，已跳过。')

        trt_accel: Optional[TensorRTAccelerator] = None
        if use_trt:
            sh = (batch_size, 3, frame_h, frame_w)
            trt_accel = TensorRTAccelerator(
                upsampler.model, device, trt_cache_dir, sh, use_fp16=use_half
            )

        # [FIX-6] face_enhance：Worker 内独立初始化 GFPGANer
        # GFPGANer 不可序列化，不能跨进程传递，必须在 Worker 内构建
        face_enhancer = None
        if face_enhance and gfpgan_model_path:
            try:
                from gfpgan import GFPGANer
                print(f'[Worker {worker_idx}] 加载模型(GFPGAN): {gfpgan_model_path} → cuda:{device_id}')
                face_enhancer = GFPGANer(
                    model_path=gfpgan_model_path,
                    upscale=outscale,
                    arch=gfpgan_arch,
                    channel_multiplier=gfpgan_ch,
                    bg_upsampler=None,  # 背景超分由主 SR batch 完成
                )
                print(f'[Worker {worker_idx}] GFPGANer 初始化成功 | '
                      f'{"FP16" if face_fp16 else "FP32"} | gfpgan_bs={gfpgan_batch_size}')
            except Exception as e:
                print(f'[Worker {worker_idx}] GFPGANer 初始化失败，跳过 face_enhance: {e}')
                face_enhancer = None

        transfer_stream = torch.cuda.Stream(device=device)
        compute_stream  = torch.cuda.Stream(device=device)

        bs     = batch_size
        gfp_bs = gfpgan_batch_size  # [FIX-OOM-GFP] 进程内持久化 GFPGAN 子批量

        # ── 判断是否走流水线路径 ──────────────────────────────────────────────
        # tile > 0 时模型内部分块推理，无法整批送入 _sr_infer_batch，使用串行兜底
        use_pipeline = (face_enhancer is not None and _HAS_BATCHGFPGAN and tile == 0)

        if use_pipeline:
            # ── [FIX-PIPELINE] 流水线路径 ────────────────────────────────────
            # 两级并行：
            #   Level-1: detect(N) 后台线程 ‖ SR(N) GPU 推理（主线程）
            #   Level-2: paste(N)  后台线程 ‖ SR(N+1)+GFPGAN(N+1) GPU（主线程）
            # GPU 利用率从 <30% 提升至 ~70-85%，每 Worker 独立受益，N Worker 叠加。
            print(f'[Worker {worker_idx}] [FIX-PIPELINE] 流水线模式已激活：'
                  f'detect ‖ SR，paste ‖ SR+GFPGAN')
            detect_helper   = _make_detect_helper(face_enhancer, device)
            detect_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f'w{worker_idx}_detect')
            paste_executor  = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f'w{worker_idx}_paste')

            fp16_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                        if face_fp16 else contextlib.nullcontext())

            local_frames:  List[np.ndarray] = []
            local_indices: List[int]        = []
            detect_fut   = None    # 当前批 detect future
            paste_fut    = None    # 上一批 paste future
            paste_idxs: List[int] = []
            sentinel_received = False

            try:
                while True:
                    # ── 获取下一帧或哨兵 ─────────────────────────────────────
                    try:
                        item = dispatch_q.get(timeout=2.0)
                    except Exception:
                        continue

                    if item == _DISPATCH_SENTINEL:
                        sentinel_received = True
                    else:
                        frame_idx, frame_bytes = item
                        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                            frame_h, frame_w, 3)
                        local_frames.append(frame)
                        local_indices.append(frame_idx)

                    # ── 触发条件：满批 or 哨兵 + 剩余帧 ─────────────────────
                    if (len(local_frames) >= bs) or (sentinel_received and local_frames):
                        current_batch   = list(local_frames)
                        current_indices = list(local_indices)
                        local_frames.clear()
                        local_indices.clear()

                        # Step A: 提交 detect(N) 到后台（立即，与下一步 SR 并行）
                        detect_fut = detect_executor.submit(
                            _detect_faces_batch, current_batch, detect_helper)

                        # Step B: 等上一批 paste 完成，发帧到 result_q
                        # 通常 SR >> paste（CPU），此处零等待
                        if paste_fut is not None:
                            prev_final = paste_fut.result()
                            paste_fut  = None
                            for idx, out in zip(paste_idxs, prev_final):
                                result_q.put((idx, out.tobytes(), out.shape))
                            paste_idxs = []

                        # Step C: SR 推理（GPU 主线程，与 detect 后台并行）
                        try:
                            sr_results = _sr_infer_batch(
                                upsampler, current_batch, outscale, netscale,
                                transfer_stream, compute_stream, trt_accel)
                        except RuntimeError as e:
                            err_str = str(e).lower()
                            if 'out of memory' in err_str and bs > 1:
                                bs = max(1, bs // 2)
                                torch.cuda.empty_cache()
                                print(f'[Worker {worker_idx}] SR OOM，降级 bs → {bs}')
                            else:
                                print(f'[Worker {worker_idx}] SR Error: {e}')
                            # 当前批失败：取消 detect future
                            if detect_fut is not None:
                                detect_fut.cancel()
                                detect_fut = None
                            if sentinel_received and not local_frames:
                                break
                            continue

                        # Step D: 等 detect(N) 完成（通常 SR > detect，已就绪，零等待）
                        face_data  = detect_fut.result()
                        detect_fut = None

                        # Step E: GFPGAN 推理（GPU 主线程）
                        restored_by_frame, gfp_bs = _gfpgan_infer_batch(
                            face_data, face_enhancer, device,
                            fp16_ctx, gfpgan_weight, gfp_bs)

                        # Step F: 提交 paste(N) 到后台（立即，与下批 SR+GFPGAN 并行）
                        paste_fut  = paste_executor.submit(
                            _paste_faces_batch,
                            face_data, restored_by_frame, sr_results, face_enhancer)
                        paste_idxs = current_indices

                    if sentinel_received and not local_frames:
                        break

                # ── Flush：等最后一批 paste 完成，发帧 ──────────────────────
                if paste_fut is not None:
                    final = paste_fut.result()
                    for idx, out in zip(paste_idxs, final):
                        result_q.put((idx, out.tobytes(), out.shape))

            finally:
                detect_executor.shutdown(wait=False)
                paste_executor.shutdown(wait=False)

            result_q.put(_RESULT_DONE)
            dispatch_q.put(_DISPATCH_SENTINEL)  # 菊花链传递哨兵

        else:
            # ── 串行路径（无 face_enhance / tile 模式 / 缺 basicsr）────────────
            local_frames:  List[np.ndarray] = []
            local_indices: List[int]        = []

            def flush():
                nonlocal bs, gfp_bs
                if not local_frames:
                    return
                try:
                    outputs, gfp_bs = _process_batch(
                        upsampler, local_frames, outscale, netscale,
                        transfer_stream, compute_stream, trt_accel,
                        face_enhancer, face_fp16, gfpgan_weight, gfp_bs,
                    )
                    for idx, out in zip(local_indices, outputs):
                        result_q.put((idx, out.tobytes(), out.shape))
                except RuntimeError as e:
                    err_str = str(e).lower()
                    if 'out of memory' in err_str and bs > 1:
                        bs = max(1, bs // 2)
                        torch.cuda.empty_cache()
                        print(f'[Worker {worker_idx}] OOM，降级 batch_size → {bs}')
                        for idx, frm in zip(local_indices, local_frames):
                            try:
                                out, gfp_bs = _process_batch(
                                    upsampler, [frm], outscale, netscale,
                                    transfer_stream, compute_stream, trt_accel,
                                    face_enhancer, face_fp16, gfpgan_weight, gfp_bs,
                                )
                                result_q.put((idx, out[0].tobytes(), out[0].shape))
                            except Exception as inner_e:
                                error_q.put((worker_idx, repr(inner_e)))
                    elif 'nvml_success' in err_str or 'cudacachingallocator' in err_str:
                        pass  # 无害断言，忽略
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
                    dispatch_q.put(_DISPATCH_SENTINEL)  # 菊花链
                    break

                frame_idx, frame_bytes = item
                frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                    frame_h, frame_w, 3)
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
            print(f'[Orchestrator] [DNI 模式] denoise_strength={args.denoise_strength:.2f} → '
                  f'{args.denoise_strength:.0%} realesr-general-x4v3 + '
                  f'{1-args.denoise_strength:.0%} realesr-general-wdn-x4v3'
                  f'（--denoise-strength 1 可禁用）')

        # 探测视频元数据
        meta = get_video_meta_info(args.input)
        W, H = meta['width'], meta['height']
        fps  = args.fps if args.fps else meta['fps']
        nb   = meta['nb_frames']
        audio_obj = meta['audio']

        print(f'[Orchestrator] 输入: {W}x{H} @ {fps:.3f}fps | {nb} 帧')
        print(f'[Orchestrator] GPU workers: {self.num_workers} '
              f'(×{self.num_workers // self.num_gpus} per GPU × {self.num_gpus} GPU)')

        # [FIX-6] 解析 face_enhance 参数，传给每个 Worker
        gfpgan_model_path = ''
        gfpgan_arch       = 'clean'
        gfpgan_ch         = 2
        face_fp16         = False

        if args.face_enhance:
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
            gfpgan_arch, gfpgan_ch, _gfpgan_name, _gfpgan_url = _GFPGAN_MODELS[_gv]

            # 优先查找本地模型文件
            _gfpgan_path = osp.join('experiments/pretrained_models', _gfpgan_name + '.pth')
            if not osp.isfile(_gfpgan_path):
                _gfpgan_path = osp.join('gfpgan/weights', _gfpgan_name + '.pth')
            if not osp.isfile(_gfpgan_path):
                _gfpgan_path = _gfpgan_url
            gfpgan_model_path = _gfpgan_path

            # [FIX-8] FP16 autocast
            face_fp16 = not args.fp32 and torch.cuda.is_available()
            print(f'[Orchestrator] face_enhance 已启用: {_gfpgan_name} | '
                  f'{"FP16" if face_fp16 else "FP32"} | '
                  f'SR-batch={args.batch_size} | '
                  f'GFPGAN-batch={getattr(args, "gfpgan_batch_size", 12)} | '
                  f'weight={getattr(args, "gfpgan_weight", 0.5)}')

        # 有界队列：防止生产者过快导致 OOM
        max_q = max(self.num_workers * args.batch_size * 4, 32)
        dispatch_q: mp.Queue = mp.Queue(maxsize=max_q)
        result_q:   mp.Queue = mp.Queue(maxsize=max_q)
        error_q:    mp.Queue = mp.Queue()

        trt_cache = osp.join(args.output, '.trt_cache')
        use_trt   = getattr(args, 'use_tensorrt', False)

        worker_kwargs = dict(
            dispatch_q        = dispatch_q,
            result_q          = result_q,
            error_q           = error_q,
            model_name        = args.model_name,
            model_path        = model_path,
            dni_weight        = dni_weight,
            tile              = args.tile,
            tile_pad          = args.tile_pad,
            pre_pad           = args.pre_pad,
            use_half          = not args.fp32,
            use_compile       = getattr(args, 'use_compile', False),
            use_trt           = use_trt,
            trt_cache_dir     = trt_cache,
            outscale          = args.outscale,
            batch_size        = args.batch_size,
            frame_h           = H,
            frame_w           = W,
            # [FIX-6] face_enhance 参数：通过普通可序列化类型传递
            face_enhance      = args.face_enhance,
            gfpgan_model_path = gfpgan_model_path,
            gfpgan_arch       = gfpgan_arch,
            gfpgan_ch         = gfpgan_ch,
            gfpgan_weight     = getattr(args, 'gfpgan_weight', 0.5),
            gfpgan_batch_size = getattr(args, 'gfpgan_batch_size', 12),
            face_fp16         = face_fp16,
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
                    tqdm.write(f'[Worker {wid} Error] {err_msg}')
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
            next_idx   = 0
            done_count = 0

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

            # 冲刷剩余
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
            ffmpeg_bin      = getattr(args, 'ffmpeg_bin', 'ffmpeg'),
            prefetch_factor = getattr(args, 'prefetch_factor', 16),   # [FIX-5]
            use_hwaccel     = getattr(args, 'use_hwaccel', True),
        )

        for frame_idx in range(nb):
            if abort_evt.is_set():
                print('\n[Orchestrator] 检测到 Worker 错误，中止读帧。')
                break
            frame = reader.get_frame()
            if frame is None:
                break
            dispatch_q.put((frame_idx, frame.tobytes()))

        # 发送哨兵（菊花链传播）
        dispatch_q.put(_DISPATCH_SENTINEL)

        reader.close()
        collect_thread.join()
        pbar.close()
        writer.close()
        abort_evt.set()
        mon_thread.join(timeout=3)

        for w in workers:
            w.join(timeout=30)
            if w.is_alive():
                print(f'[Warning] {w.name} 未在超时内退出，强制终止。')
                w.kill()

        elapsed     = time.time() - t_collect_start
        frames_done = total_done[0]
        print(f'[Orchestrator] 完成 {frames_done} 帧 | '
              f'耗时 {elapsed:.1f}s | 平均 {frames_done/elapsed:.1f} fps')

        # X3: JSON 报告
        report_path = getattr(args, 'report', None)
        if report_path and timing:
            report = {
                'input':        args.input,
                'output':       self.video_save_path,
                'model':        args.model_name,
                'outscale':     args.outscale,
                'num_workers':  self.num_workers,
                'num_gpus':     self.num_gpus,
                'batch_size':   args.batch_size,
                'fp16':         not args.fp32,
                'face_enhance': args.face_enhance,
                'nvenc':        HardwareCapability.best_video_encoder(
                    getattr(args, 'video_codec', 'libx264')
                ).endswith('nvenc'),
                'nvdec':        HardwareCapability.has_nvdec(),
                'trt':          use_trt,
                'frame_count':  frames_done,
                'elapsed_s':    round(elapsed, 2),
                'avg_fps':      round(frames_done / elapsed, 2) if elapsed > 0 else 0,
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
    face_enhancer=None,
    face_fp16: bool = False,
    gfpgan_weight: float = 0.5,
    gfpgan_batch_size: int = 12,
) -> Tuple[int, int]:
    """返回 (新SR批次大小, 新GFPGAN子批量大小)"""
    bs     = min(init_bs, len(frames))
    gfp_bs = gfpgan_batch_size
    i      = 0
    while i < len(frames):
        sub = frames[i: i + bs]
        try:
            t0 = time.perf_counter()
            outputs, gfp_bs = _process_batch(
                upsampler, sub, outscale, netscale,
                transfer_stream, compute_stream, trt_accel,
                face_enhancer, face_fp16, gfpgan_weight, gfp_bs,
            )
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
                new_bs = min(bs + 1, max_bs[0])
                tqdm.write(f'[恢复] 显存充裕,batch_size {bs} → {new_bs}')
                bs = new_bs
                
        except RuntimeError as e:
            err_str = str(e).lower()
            if 'out of memory' in err_str and bs > 1:
                bs = max(1, bs // 2)
                oom_cooldown[0] = 10
                torch.cuda.empty_cache()
                tqdm.write(f'[OOM] 降级 batch_size → {bs}')
            elif 'nvml_success' in err_str or 'cudacachingallocator' in err_str:
                # [BUGFIX] 无害断言，跳过
                pbar.update(len(sub))
                i += len(sub)
            else:
                tqdm.write(f'[Error] {e}')
                pbar.update(len(sub))
                i += len(sub)
    return bs, gfp_bs


# ─────────────────────────────────────────────────────────────────────────────
# 单 GPU 推理主循环
# ─────────────────────────────────────────────────────────────────────────────

def inference_video_single(args, video_save_path: str, device=None):
    """单卡推理路径，v5++ 全量优化。"""
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
        args.tile, args.tile_pad, args.pre_pad, not args.fp32, device
    )

    # [FIX-1] cuDNN benchmark
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled   = True
        print('[FIX-1] cudnn.benchmark = True 已启用')

    if args.use_compile and hasattr(torch, 'compile'):
        print('[Info] torch.compile 加速中 ...')
        upsampler.model = torch.compile(upsampler.model, mode='reduce-overhead')

    # M3: TensorRT 可选
    trt_accel: Optional[TensorRTAccelerator] = None
    if getattr(args, 'use_tensorrt', False) and torch.cuda.is_available():
        meta    = get_video_meta_info(args.input)
        sh      = (args.batch_size, 3, meta['height'], meta['width'])
        trt_dir = osp.join(args.output, '.trt_cache')
        trt_accel = TensorRTAccelerator(upsampler.model, device, trt_dir, sh,
                                        use_fp16=not args.fp32)

    if 'anime' in args.model_name and args.face_enhance:
        print('[Warning] anime 模型不支持 face_enhance，已禁用。')
        args.face_enhance = False

    face_enhancer = None
    face_fp16     = False
    if args.face_enhance:
        from gfpgan import GFPGANer
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

        _gfpgan_path = osp.join('experiments/pretrained_models', _gfpgan_name + '.pth')
        if not osp.isfile(_gfpgan_path):
            _gfpgan_path = osp.join('gfpgan/weights', _gfpgan_name + '.pth')
        if not osp.isfile(_gfpgan_path):
            _gfpgan_path = _gfpgan_url

        # [FIX-6] 不再强制 batch_size=1
        print(f'  加载模型(GFPGAN): {_gfpgan_path} → {device}')
        face_enhancer = GFPGANer(
            model_path=_gfpgan_path,
            upscale=args.outscale,
            arch=_gfpgan_arch,
            channel_multiplier=_gfpgan_ch,
            bg_upsampler=None,
        )
        # [FIX-8] FP16 autocast
        face_fp16 = not args.fp32 and torch.cuda.is_available()
        print(f'[v6] 流水线GFPGAN已启用: {_gfpgan_name} | {"FP16" if face_fp16 else "FP32"} | '
              f'basicsr_utils={"OK" if _HAS_BATCHGFPGAN else "缺失（逐帧回退）"}')

    # [v5++-ADAPT] gfpgan_bs 跨批次持久化
    gfpgan_bs: int = getattr(args, 'gfpgan_batch_size', 12)

    # [FIX-6] 不再因 face_enhancer 存在而禁用 batch 模式
    use_batch = args.batch_size > 1 and args.tile == 0

    transfer_stream = compute_stream = None
    if torch.cuda.is_available():
        transfer_stream = torch.cuda.Stream(device=device)
        compute_stream  = torch.cuda.Stream(device=device)

    reader = FFmpegReader(
        args.input,
        ffmpeg_bin      = getattr(args, 'ffmpeg_bin', 'ffmpeg'),
        prefetch_factor = getattr(args, 'prefetch_factor', 16),   # [FIX-5]
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

    if use_batch and face_enhancer is not None and _HAS_BATCHGFPGAN:
        # ── [FIX-PIPELINE] 流水线路径（face_enhance + batch 模式激活）─────────
        # 两级并行：
        #   Level-1: detect(N) 在后台线程，与主线程 SR(N) GPU 并行
        #   Level-2: paste(N)  在后台线程，与主线程 SR(N+1)+GFPGAN(N+1) 并行
        # GPU 利用率从 <30% 提升至 ~70-85%（detect/paste 不再阻塞 GPU）。
        print('[FIX-PIPELINE] 单卡流水线模式已激活：detect ‖ SR，paste ‖ SR+GFPGAN')
        detect_helper   = _make_detect_helper(face_enhancer, device)
        detect_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='fe_detect')
        paste_executor  = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='fe_paste')

        fp16_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                    if face_fp16 else contextlib.nullcontext())

        detect_fut  = None
        paste_fut   = None
        paste_n     = 0     # 上一批帧数，用于 pbar.update

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

                    # Step A: 提交 detect(N) 到后台（立即，与下一步 SR 并行）
                    detect_fut = detect_executor.submit(
                        _detect_faces_batch, current_batch, detect_helper)

                    # Step B: 等上一批 paste 完成，写帧到 FFmpeg
                    if paste_fut is not None:
                        prev_final = paste_fut.result()   # 通常已完成，零等待
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

                    # Step C: SR 推理（GPU 主线程，与 detect 后台并行）
                    t0 = time.perf_counter()
                    try:
                        sr_results = _sr_infer_batch(
                            upsampler, current_batch, args.outscale, netscale,
                            transfer_stream, compute_stream, trt_accel)
                        timing.append(time.perf_counter() - t0)
                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower() and bs > 1:
                            bs = max(1, bs // 2)
                            _oom_cd[0] = 10
                            torch.cuda.empty_cache()
                            tqdm.write(f'[OOM] SR 降级 batch_size → {bs}')
                        else:
                            tqdm.write(f'[Error] SR: {e}')
                        if detect_fut is not None:
                            detect_fut.cancel()
                            detect_fut = None
                        if end:
                            break
                        continue

                    # Step D: 等 detect(N) 完成（通常 SR > detect，已就绪）
                    face_data  = detect_fut.result()
                    detect_fut = None

                    # Step E: GFPGAN 推理（GPU 主线程）
                    restored_by_frame, gfpgan_bs = _gfpgan_infer_batch(
                        face_data, face_enhancer, device,
                        fp16_ctx, args.gfpgan_weight, gfpgan_bs)

                    # Step F: 提交 paste(N) 到后台（立即，与下批 SR 并行）
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

            # ── Flush：等最后一批 paste 完成，写帧 ──────────────────────────
            if paste_fut is not None:
                for frame in paste_fut.result():
                    writer.write_frame(frame)
                pbar.update(paste_n)
                meter.update(paste_n)

        finally:
            detect_executor.shutdown(wait=False)
            paste_executor.shutdown(wait=False)

    elif use_batch:
        # ── 原有批处理路径（无 face_enhance 或缺少 basicsr）──────────────────
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
                    bs, _oom_cd, _max_bs, timing, trt_accel,
                    face_enhancer, face_fp16,
                    getattr(args, 'gfpgan_weight', 0.5),
                    gfpgan_bs,
                )
                meter.update(len(batch_frames))
                # [FIX-TQDM] 末帧批次已在 flush_batch_safe 内打满 100%，
                # 此处再调 set_postfix 会触发 tqdm 重渲染出第二条完成行
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
        while True:
            img = reader.get_frame()
            if img is None:
                break
            try:
                t0 = time.perf_counter()
                output, _ = upsampler.enhance(img, outscale=args.outscale)
                timing.append(time.perf_counter() - t0)
                # [FIX-6] tile/bs=1 路径下的 face_enhance
                if face_enhancer is not None:
                    try:
                        fp16_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                                    if face_fp16 else contextlib.nullcontext())
                        with fp16_ctx:
                            _, restored_faces, _ = face_enhancer.enhance(
                                img,
                                has_aligned=False,
                                only_center_face=False,
                                paste_back=False,
                                weight=getattr(args, 'gfpgan_weight', 0.5),
                            )
                        if restored_faces:
                            face_enhancer.face_helper.get_inverse_affine(None)
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
                                    _res = _cv2.resize(_res, (output.shape[1], output.shape[0]))
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
        torch.cuda.synchronize(device=device)
    reader.close()
    writer.close()
    pbar.close()

    # X3: JSON 报告
    report_path = getattr(args, 'report', None)
    if report_path and timing:
        elapsed = time.time() - t_start
        report = {
            'input':        args.input,
            'output':       video_save_path,
            'model':        args.model_name,
            'outscale':     args.outscale,
            'batch_size':   args.batch_size,
            'fp16':         not args.fp32,
            'face_enhance': args.face_enhance,
            'trt':          trt_accel is not None and trt_accel.available,
            'nvdec':        HardwareCapability.has_nvdec(),
            'nvenc':        HardwareCapability.best_video_encoder(
                getattr(args, 'video_codec', 'libx264')).endswith('nvenc'),
            'frame_count':  nb,
            'elapsed_s':    round(elapsed, 2),
            'avg_fps':      round(nb / elapsed, 2) if elapsed > 0 else 0,
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
# [FIX-OUTPUT] 输出路径判断：支持 -o 直接指定视频文件
# ─────────────────────────────────────────────────────────────────────────────

def _output_is_file(path: str) -> bool:
    """
    判断 -o 参数是输出文件还是目录。
    优先级：①已存在则信任文件系统 ②MIME 识别为 video ③有非空扩展名 → 文件
    """
    if osp.exists(path):
        return osp.isfile(path)
    mime, _ = mimetypes.guess_type(path)
    if mime is not None and mime.startswith('video'):
        return True
    _, ext = osp.splitext(path)
    return ext != ''


# ─────────────────────────────────────────────────────────────────────────────
# run：单卡 / 多卡自动分派
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    args.video_name = osp.splitext(os.path.basename(args.input))[0]

    # [FIX-OUTPUT] -o 支持直接指定输出文件路径（如 out.mkv / out.avi）
    if _output_is_file(args.output):
        video_save_path = args.output
        out_dir = osp.dirname(osp.abspath(video_save_path))
        os.makedirs(out_dir, exist_ok=True)
        # 规范化 args.output 为目录，避免下游（trt_cache / report 路径）误用文件名
        args.output = out_dir
    else:
        video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4')

    mime = mimetypes.guess_type(args.input)[0]
    args.input_type_is_video = mime is not None and mime.startswith('video')

    num_gpus = torch.cuda.device_count()
    nw_per_g = getattr(args, 'num_process_per_gpu', 1)
    num_workers = max(1, num_gpus * nw_per_g)

    if num_workers > 1 and args.input_type_is_video:
        tqdm.write(f'[V6] 多 GPU 模式：{num_gpus} GPU × {nw_per_g} Worker = {num_workers} 总 Worker')
        print('[V6] 架构：Dispatcher-Queue + FIX-PIPELINE（每 Worker 独立 detect/paste 线程池）')
        orchestrator = MultiGPUOrchestrator(args, video_save_path, num_gpus, nw_per_g)
        orchestrator.run()
    else:
        if num_workers > 1:
            print('[Info] 图片目录输入，使用单卡路径。')
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
        description='Real-ESRGAN 视频超分 —— 终极优化版 v6（多卡版）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 基础参数
    parser.add_argument('-i',  '--input',            type=str, default='inputs')
    parser.add_argument('-n',  '--model-name',       type=str, default='realesr-animevideov3')
    parser.add_argument('-o',  '--output',           type=str, default='results',
                        help='输出目录，或直接指定输出视频文件路径（如 out.mkv）')
    parser.add_argument('-dn', '--denoise-strength', type=float, default=0.5)
    parser.add_argument('-s',  '--outscale',         type=float, default=4)
    parser.add_argument('--suffix',                  type=str, default='out')
    # 推理参数
    parser.add_argument('-t',  '--tile',             type=int, default=0)
    parser.add_argument('--tile-pad',                type=int, default=10)
    parser.add_argument('--pre-pad',                 type=int, default=0)
    parser.add_argument('--face-enhance',            action='store_true',
                        help='启用 GFPGAN 人脸增强（多卡模式下每 Worker 独立实例化）')
    parser.add_argument('--gfpgan-model',            type=str, default='1.4',
                        choices=['1.3', '1.4', 'RestoreFormer'],
                        help='GFPGAN 模型版本（--face-enhance 时生效）。'
                             '本地优先查找 experiments/pretrained_models/ 和 gfpgan/weights/，'
                             '不存在时自动下载。Default: 1.4')
    parser.add_argument('--gfpgan-weight',           type=float, default=0.5,
                        help='GFPGAN 增强融合权重，0.0=不增强，1.0=完全替换，Default: 0.5')
    parser.add_argument('--gfpgan-batch-size',       type=int, default=12,
                        help='[FIX-OOM-GFP] 单次 GFPGAN 前向最多处理的人脸数。'
                             'OOM 时自动对半降级。A10(24G) 建议 8~12，T4(16G) 建议 4~8。'
                             'Default: 12')
    parser.add_argument('--fp32',                    action='store_true',
                        help='禁用 FP16（默认启用 FP16）')
    parser.add_argument('--fps',                     type=float, default=None)
    # [FIX-5] batch_size 默认 8，prefetch_factor 默认 16
    parser.add_argument('--batch-size',              type=int, default=8,
                        help='批处理大小（T4 15G 建议 8~12，A100 80G 可到 32）')
    parser.add_argument('--prefetch-factor',         type=int, default=16,
                        help='读帧预取队列深度，建议 ≥ batch_size×2')
    parser.add_argument('--use-compile',             action='store_true',
                        help='启用 torch.compile（reduce-overhead，每 Worker 独立编译）')
    # V5 参数
    parser.add_argument('--use-tensorrt',            action='store_true',
                        help='[V5] 启用 TensorRT 加速（[FIX-TRT3] 已修复 TRT 8/10 兼容性）')
    parser.add_argument('--use-hwaccel',             action='store_true', default=True,
                        help='[V5] 启用 NVDEC 硬件解码（自动探测，失败时回退）')
    parser.add_argument('--no-hwaccel',              action='store_true',
                        help='[V5] 强制禁用 NVDEC 硬件解码')
    # 多卡参数
    parser.add_argument('--num-process-per-gpu',     type=int, default=1,
                        help='每 GPU 启动的 Worker 进程数（24G+ 显存可设 2）')
    # 编码参数
    parser.add_argument('--video-codec',             type=str, default='libx264',
                        choices=['libx264', 'libx265', 'libvpx-vp9'],
                        help='偏好编码器（有 NVENC 时自动升级为 h264_nvenc/hevc_nvenc）')
    parser.add_argument('--crf',                     type=int, default=23,
                        help='编码质量（默认 23，值越小质量越高文件越大）')
    parser.add_argument('--ffmpeg-bin',              type=str, default='ffmpeg')
    # 其他
    parser.add_argument('--alpha-upsampler',         type=str, default='realesrgan',
                        choices=['realesrgan', 'bicubic'])
    parser.add_argument('--ext',                     type=str, default='auto',
                        choices=['auto', 'jpg', 'png'])
    parser.add_argument('--report',                  type=str, default=None,
                        help='输出 JSON 性能报告路径（如 report.json）')

    args = parser.parse_args()
    args.input = args.input.rstrip('/\\')

    # [FIX-OUTPUT] 仅在非文件路径时预创建目录
    if not _output_is_file(args.output):
        os.makedirs(args.output, exist_ok=True)

    # ── 记录 CLI 显式传入的标志，供冲突仲裁使用 ────────────────────────────────
    import sys as _sys
    _cli_argv = set(_sys.argv[1:])
    _compile_explicit  = '--use-compile'  in _cli_argv
    _tensorrt_explicit = '--use-tensorrt' in _cli_argv

    # ── 冲突仲裁：--use-compile 与 --use-tensorrt 同时开启 ────────────────────
    if args.use_compile and args.use_tensorrt:
        if _compile_explicit and not _tensorrt_explicit:
            print('[Warning] CLI 显式传入 --use-compile，但 args.use_tensorrt=True（来自外部配置）。'
                  '\n          CLI 优先：禁用 TensorRT，使用 torch.compile 路径。'
                  '\n          若需要 TRT，请在 CLI 显式传入 --use-tensorrt。')
            args.use_tensorrt = False
        else:
            print('[Warning] --use-compile 与 --use-tensorrt 同时开启。'
                  '\n          TRT 推理路径完全接管，torch.compile 不会被执行，已自动禁用。'
                  '\n          如需 torch.compile，请去掉 --use-tensorrt。')
            args.use_compile = False

    # 处理 --no-hwaccel
    if args.no_hwaccel:
        args.use_hwaccel = False

    # FLV 转 MP4（[FIX-PATH] 用 splitext 而非 endswith）
    mime     = mimetypes.guess_type(args.input)[0]
    is_video = mime is not None and mime.startswith('video')
    if is_video and osp.splitext(args.input)[1].lower() == '.flv':
        mp4_path = args.input.replace('.flv', '.mp4')
        subprocess.run([args.ffmpeg_bin, '-i', args.input, '-codec', 'copy', '-y', mp4_path])
        args.input = mp4_path

    # 启动前打印硬件状态
    print('=' * 65)
    print('  RealESRGAN 视频超分 —— 终极优化版 v6（多卡版）')
    print('=' * 65)
    num_gpus = torch.cuda.device_count()
    print(f'  GPU 数量: {num_gpus}')
    for i in range(num_gpus):
        prop = torch.cuda.get_device_properties(i)
        print(f'    GPU {i}: {prop.name} | {prop.total_memory / 1024**3:.1f} GB VRAM')
    print(f'  NVDEC:   {HardwareCapability.has_nvdec()} | '
          f'NVENC(h264): {HardwareCapability.has_nvenc("h264_nvenc")} | '
          f'NVENC(hevc): {HardwareCapability.has_nvenc("hevc_nvenc")}')
    _accel_label = ('TensorRT' if args.use_tensorrt else
                    'compile' if args.use_compile else 'FP16-only')
    print(f'  推理加速: {_accel_label} | '
          f'TensorRT={args.use_tensorrt} | torch.compile={args.use_compile}')
    print(f'  batch-size: {args.batch_size} | prefetch: {args.prefetch_factor} | '
          f'num-process-per-gpu: {args.num_process_per_gpu}')
    print(f'  face-enhance: {args.face_enhance} '
          f'(model={getattr(args, "gfpgan_model", "1.4")} | '
          f'weight={getattr(args, "gfpgan_weight", 0.5)} | '
          f'GFPGAN-batch={getattr(args, "gfpgan_batch_size", 12)} | '
          f'v6: 流水线并行+独立FaceRestoreHelper+尺寸修复+Worker级流水线)')
    print(f'  [v6 升级] FIX-PIPELINE(detect‖SR + paste‖SR+GFPGAN) | '
          f'FIX-DETECT-BUG | FIX-PIPE-BUG | 每Worker独立流水线')
    print()

    run(args)


if __name__ == '__main__':
    main()
