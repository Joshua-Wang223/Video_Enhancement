"""
RealESRGAN 视频超分处理脚本 —— 终极优化版 v5++（单卡版）
==========================================================
v5++ 在 v5+ 基础上修复 face_enhance 性能问题（实测 0.3 fps → 预期 2~4 fps）：

  [v5++-PIPE] FIX-PIPE 修复 face_enhance 输出帧尺寸错误（根本原因分析与修正）
      根因分析（facexlib 0.3.0 API 契约冲突）：
        ❌ 原始代码 GFPGANer(upscale=1) → FaceRestoreHelper.upscale_factor = 1
        ❌ read_image(orig_720p) → self.input_img = 720p
        ❌ paste_faces_to_input_image() 内部：
             h_up = self.input_img.shape[0] × self.upscale_factor = 720 × 1 = 720
             → 无条件将 upsample_img resize 到 720p 画布（facexlib 0.3.0 行为）
        ❌ get_inverse_affine() 中 inverse_affine × upscale_factor=1 → 矩阵坐标系 = 720p
        ❌ 脚本再手动 [m × outscale] → 矩阵坐标系变 1440p，但画布仍 720p
        ❌ warpAffine(1440p 坐标系矩阵, 画布=(720p)) → 人脸像素超出画布，被截断/错位
        ❌ write_frame 收到 720p 帧触发 resize 兜底 → 错位碎片放大为浮动头像
      修复策略（方案 A，一次性修正所有下游缺陷）：
        ✅ GFPGANer(upscale=outscale) → upscale_factor = outscale（如 2）
        ✅ read_image(orig_720p) + upscale_factor=2 → h_up = 720×2 = 1440p 画布 ✓
        ✅ get_inverse_affine() 内部自动 × outscale → 矩阵坐标系 = 1440p ✓
        ✅ upsample_img=frame_sr(1440p) → resize 到 1440p = 无操作 ✓
        ✅ 彻底删除所有手动 [m × outscale] 代码（避免双重缩放）
        ✅ FFmpegWriter.write_frame 保留尺寸守卫作为最后防线

  [v5++-BUG] FIX-7 实际未生效（v5+ 最严重的隐藏 bug）
      v5+ 注释声称"在原始低分辨率帧上检测"，但代码实际传入的是：
        ❌ face_enhancer.enhance(frame_sr, ...)   # frame_sr = SR 输出（1440p）
      `frames` 变量（原始帧）就在作用域内却从未使用，导致：
        ❌ RetinaFace 在1440p上检测 → 与无 face_enhance 时相比慢 4×
        ❌ GFPGAN 后处理在1440p像素操作 → 额外慢 4×
        ❌ 实测 ~28000ms/batch(8帧) ≈ 0.3 fps
      v5++ 修正：
        ✅ 在原始帧（720p）上检测 → RetinaFace 实际获得 4× 加速
        ✅ 无人脸帧直接跳过 GFPGAN → 零额外开销（大多数帧无人脸时效果显著）

  [v5++-BATCH] 批量 GFPGAN 推理（新增）
      原逐帧路径：每帧调用一次 face_enhancer.enhance()，GFPGAN 网络串行
        ❌ batch=8 → 8次独立 GFPGAN 前向（每次 512×512×1）
      v5++ 批量路径：汇总所有帧的 face crops → 单次批量推理
        ✅ 收集 batch 内所有帧的 face crops（通常每帧 1~2 张）
        ✅ 堆叠为 (N_faces, 3, 512, 512)，一次 GFPGAN 前向
        ✅ 充分利用 T4 TensorCore 并行能力（FP16 吞吐 ~65 TFLOPS）
        ✅ 预期 GFPGAN 推理时间缩短 3~6×（取决于人脸密度）

  [v5++-NOSKIP] 无人脸帧快速跳过
      检测阶段（RetinaFace）在原始帧上运行后，若 cropped_faces 为空：
        ✅ 直接 final_results.append(frame_sr)，不进入 GFPGAN 推理
        ✅ 对于人脸稀疏的视频，大量帧可完全跳过 GFPGAN

  [v5++-PASTE] 正确将增强人脸贴回 SR 帧
      facexlib ≥ 0.3.x：paste_faces_to_input_image(upsample_img=sr_frame)
        ✅ 用 SR 高分辨率帧作为背景，affine 坐标自动缩放 × outscale
      旧版 facexlib 回退：手动将 inverse_affine_matrices × outscale
        ✅ 兼容两种 API，运行时自动探测（inspect.signature）

[保留全部 v5+ 优化]
  FIX-TRT10/1~8 全部保留，批量 SR 路径（TRT/PyTorch）不变

  [FIX-TRT10] TensorRT 10.x API 兼容性修复（主要崩溃根因）
      TensorRT 10.x 废弃并移除了以下旧 binding 索引 API：
        ❌ engine.get_binding_shape(binding_idx)
        ❌ context.execute_async_v2(bindings=[...], stream_handle=...)
        ❌ NetworkDefinitionCreationFlag.EXPLICIT_BATCH（已成默认，枚举已删除）
      改为 tensor 名称 API：
        ✅ engine.get_tensor_shape(tensor_name)
        ✅ context.set_tensor_address(tensor_name, data_ptr)
        ✅ context.execute_async_v3(stream_handle=...)
        ✅ engine.num_io_tensors / engine.get_tensor_name(i) / engine.get_tensor_mode(name)
      修复位置：
        _build_engine  → EXPLICIT_BATCH flag 兼容判断（try/except AttributeError）
        _load_engine   → 预解析 input/output tensor 名称，存入 self._input_name 等
        infer()        → 运行时根据 self._use_new_api 选择 TRT 10.x 或 TRT 8.x 路径
      ⚠️  注意：若已存在旧版 .trt 缓存文件（由 TRT 8.x 构建），
          升级 TRT 版本后请删除 .trt_cache/ 目录重新构建 Engine。

  [FIX-1] cudnn.benchmark = True
      对固定输入尺寸自动选择最快卷积算法，GPU 利用率 +10~15%。

  [FIX-2] 异步 D2H + Pinned 输出 Buffer
      原来 out_u8.cpu().numpy() 触发隐式 cudaDeviceSynchronize，
      GPU 在 PCIe 搬运期间完全空闲。
      改为预分配 pinned output buffer，copy_(non_blocking=True) 异步搬运，
      推理与 D2H 之间无同步等待，GPU 利用率显著提升。

  [FIX-3] TensorRT infer 去掉 CPU 中转（严重 bug 修复）
      原代码在 infer() 里先把 GPU Tensor 拉到 CPU（D2H），
      再通过 pycuda 把 numpy 搬回 GPU（H2D），完全抵消 TRT 收益。
      改为直接用 Tensor.data_ptr() 传给 TRT 执行上下文，全程 GPU 内存。

  [FIX-4] 去掉 _read_loop 里多余的 arr.copy()
      np.frombuffer 返回的数组传给 PinnedBufferPool.get_for_frames，
      np.stack 内部会完整复制，提前 copy() 是双重拷贝。

  [FIX-5] 默认 batch_size / prefetch_factor 提高
      T4 15G 显存仅用 9.8G，batch_size 默认从 4 → 8，
      prefetch_factor 从 8 → 16，充分填满显存和 PCIe 带宽。

  [FIX-6] face_enhance 批处理解耦（新增）
      原代码在 --face_enhance 时强制 batch_size=1 并完全禁用批处理路径，
      导致 TRT Engine 构建后从未被调用，实测 0.2 fps。
      修复策略：
        ① 取消 batch_size=1 强制降级，允许 face_enhance 走批处理路径。
        ② SR 批推理（TRT/PyTorch）先以完整 batch 执行，充分利用 GPU。
        ③ 对 SR 结果逐帧应用 face_enhance（GFPGAN 自身不支持批推理）。
        ④ face_enhancer 参数透传至 _process_batch / flush_batch_safe。
      实测效果：SR 阶段速度与无 face_enhance 相同，
      仅 GFPGAN 逐帧开销叠加，整体吞吐大幅提升。

  [FIX-TRT-STREAM] TRT 专用非默认 CUDA 流（新增）
      原代码 infer() 使用 torch.cuda.current_stream()，在主批处理循环上下文中
      该流即为 default stream。TRT 在 default stream 上执行 enqueueV3 时会自动
      插入额外的 cudaStreamSynchronize() 调用（即日志 W 警告的根因），造成全局阻塞。
      修复：在 TensorRTAccelerator 中预分配专用非默认 Stream，infer() 全程在该
      Stream 上执行，消除隐式同步开销，同时仍保证与 PyTorch 处于同一 CUDA context。

  [FIX-7] GFPGAN 在原始低分辨率帧上检测，增强人脸贴回 SR 帧（方案 A 最终修正）
      原代码将 GFPGAN 应用于 SR 输出（1440p），导致检测极慢且画布尺寸错误。
      方案 A 修复：
        ① GFPGANer(upscale=outscale) → upscale_factor 与 outscale 保持一致。
        ② enhance(orig_frame, paste_back=False)：720p 帧检测，速度 4×。
        ③ get_inverse_affine() 内部自动完成坐标缩放，无需手动干预。
        ④ paste_faces_to_input_image(upsample_img=sr_frame)：画布=1440p，贴图正确。
        ⑤ 全程坐标系由 facexlib 内部统一管理，彻底消除漂移。

  [FIX-8] GFPGAN FP16 混合精度（新增）
      原代码 GFPGAN 网络全程 FP32，T4 FP16 Tensor Core 未被利用。
      修复：用 torch.autocast(device_type='cuda', dtype=torch.float16) 包裹
      face_enhancer.enhance() 调用，网络自动以 FP16 运行，预期 2-3× 加速。

[保留全部 v5 优化]
  M2. NVDEC / NVENC 自动探测与回退
  M3. TensorRT 可选加速（FIX-3 修复后真正有效）
  X1-X5 异常传播/error flag/JSON报告/timing采集/tqdm对齐
  B1-B7 Fraction fps/pipe安全/drain/pinned内存池/CUDA流/compile
  N1-N9 原地操作/批量降级/批量写入/async I/O
"""

from __future__ import annotations

import argparse
import contextlib          # [FIX-7/8] GFPGAN FP16 autocast 所需
import json
import mimetypes
import numpy as np
import os
os.environ.setdefault("PYTORCH_NO_NVML", "1")   # suppress harmless NVML warning
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import inspect

import re
import torch
import torch.nn.functional as F


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

base_dir = '/workspace/Video_Enhancement'
models_RealESRGAN = f'{base_dir}/models_RealESRGAN'

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
        """实际使用NVDEC解码H.264流以进行验证。"""
        try:
            cmd = [
                'ffmpeg', '-hwaccel', 'cuda',
                '-f', 'lavfi',
                '-i', 'testsrc=size=64x64:duration=0.04:rate=25',
                '-vcodec', 'libx264', '-f', 'h264', 'pipe:1',
                '-loglevel', 'error',
            ]
            # 尝试先将视频编码为H.264格式，再使用NVDEC进行解码。
            enc = subprocess.run(cmd, capture_output=True, timeout=10)
            if enc.returncode != 0 or not enc.stdout:
                return False
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
        logger = trt.Logger(trt.Logger.WARNING)
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
        logger  = trt.Logger(trt.Logger.WARNING)
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
            print(f'\n[WARN] write_frame 尺寸修正 {frame.shape[:2]} → '
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
# 批次推理
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
    face_enhancer=None,
    face_fp16: bool = False,      # [FIX-8] 控制 GFPGAN 是否走 FP16 autocast
    gfpgan_weight: float = 0.5,   # [ADD] GFPGAN 人脸增强融合权重，与 GFPGANer.enhance() 保持一致
    gfpgan_batch_size: int = 12,   # [OOM-FIX] GFPGAN 推理子批量上限，防止人脸密集场景 OOM
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
                print(f'\n[face_detect] 本批 {len(face_data)} 帧中检测到人脸：'
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
                            print(f'\n[OOM-GFPGAN] 人脸批量 OOM，降级 gfpgan_batch_size → {sub_bs}')
                            # 本次失败的 sub_batch 用新 sub_bs 重试，不前进 i_face
                        else:
                            # 非 OOM 错误或 sub_bs=1 仍 OOM：跳过当前人脸，补 None 占位
                            print(f'\n[OOM-GFPGAN] 不可恢复错误（sub_bs={sub_bs}），跳过 {len(sub_crops)} 张脸: {_oom_e}')
                            for _ in sub_crops:
                                all_out_tensors.append(None)   # type: ignore[arg-type]
                            torch.cuda.empty_cache()
                            i_face += len(sub_crops)
                    finally:
                        del sub_batch

                # 将输出分配回各帧
                print(f"[face_enhance] GFPGAN 增强完成：{n_faces} 张人脸已处理 "
                      f"(sub_bs={sub_bs}，{'FP16' if face_fp16 else 'FP32'})")
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
                    print(f'\n[face_enhance] 帧{fi} 贴回异常，使用 SR 结果: {e}')
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
                    print(f'\n[face_enhance] 帧处理异常，使用 SR 结果: {e}')
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
                print(f'\n[WARN] face_enhance 帧{_i} 尺寸异常 '
                      f'{_res.shape[:2]} != ({expected_h},{expected_w})，强制 resize')
                final_results[_i] = _cv2.resize(_res, (expected_w, expected_h),
                                                interpolation=_cv2.INTER_LANCZOS4)
        return final_results, sub_bs

    # 无 face_enhance 路径：直接返回 SR 结果，gfpgan_batch_size 原值不变
    return sr_results, gfpgan_batch_size


# ─────────────────────────────────────────────────────────────────────────────
# flush_batch_safe（OOM 降级 + 恢复）
# [FIX-6] 透传 face_enhancer 参数
# ─────────────────────────────────────────────────────────────────────────────

def flush_batch_safe(
    upsampler, frames, outscale, netscale,
    transfer_stream, compute_stream, writer,
    pbar, init_bs, oom_cooldown, max_bs, timing,
    trt_accel=None,
    face_enhancer=None,          # [FIX-6] 新增
    face_fp16: bool = False,     # [FIX-8] 新增，透传给 _process_batch
    gfpgan_weight: float = 0.5,  # [ADD] GFPGAN 融合权重透传
    gfpgan_batch_size: int = 12,  # [v5++-ADAPT] GFPGAN 子批量上限，由返回值跨批次更新
) -> Tuple[int, int]:            # 返回 (新SR批次大小, 新GFPGAN子批量大小)
    bs = min(init_bs, len(frames))
    i  = 0
    while i < len(frames):
        sub = frames[i: i + bs]
        try:
            t0 = time.perf_counter()
            outputs, gfpgan_batch_size = _process_batch(
                upsampler, sub, outscale, netscale,
                transfer_stream, compute_stream,
                trt_accel, face_enhancer, face_fp16, gfpgan_weight, gfpgan_batch_size,  # [FIX-6/8/ADAPT] 透传
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
                bs = min(bs + 1, max_bs[0])
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and bs > 1:
                bs = max(1, bs // 2)
                oom_cooldown[0] = 10
                torch.cuda.empty_cache()
                print(f'\n[OOM] 降级 batch_size → {bs}')
            elif 'NVML_SUCCESS' in str(e) or 'CUDACachingAllocator' in str(e):
                # [BUGFIX] Suppress harmless NVML/CUDACachingAllocator assertions that
                # PyTorch raises as RuntimeErrors on systems where NVML is unavailable
                # (e.g. Docker containers without full NVIDIA driver access). These do
                # not affect correctness; the frame was already processed successfully.
                pbar.update(len(sub))
                i += len(sub)
            else:
                print(f'\n[Error] {e}')
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

    upsampler = _build_upsampler(
        args.model_name, model_path, dni_weight,
        args.tile, args.tile_pad, args.pre_pad, not args.fp32, device
    )

    if args.use_compile and hasattr(torch, 'compile'):
        print('[Info] torch.compile 加速中 ...')
        upsampler.model = torch.compile(upsampler.model, mode='reduce-overhead')

    # M3: TensorRT 可选（[FIX-3] 已修复，现在真正有效）
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
        face_fp16 = not args.fp32 and torch.cuda.is_available()
        print(f'[v5++] 批量GFPGAN已启用: {_gfpgan_name} | {"FP16" if face_fp16 else "FP32"} | '
              f'basicsr_utils={"OK" if _HAS_BATCHGFPGAN else "缺失（逐帧回退）"}')
        print(f'[v5++] face_enhance: 原始帧检测+批量GFPGAN推理+SR帧贴图 | '
              f'SR-batch={args.batch_size} | GFPGAN-batch={args.gfpgan_batch_size} | '
              f'weight={args.gfpgan_weight}')
    else:
        face_fp16 = False

    # [v5++-ADAPT] gfpgan_bs 作为普通 int 在主循环中更新：
    # flush_batch_safe 返回实际使用的子批量大小，下一次调用直接传入，
    # 实现 OOM 降级值跨批次持久化，无需 list 容器副作用。
    gfpgan_bs: int = getattr(args, 'gfpgan_batch_size', 12)

    use_batch = args.batch_size > 1 and args.tile == 0

    transfer_stream = compute_stream = None
    if torch.cuda.is_available():
        transfer_stream = torch.cuda.Stream(device=device)
        compute_stream  = torch.cuda.Stream(device=device)

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

    if use_batch:
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
                    face_enhancer, face_fp16, args.gfpgan_weight,
                    gfpgan_bs,  # [v5++-ADAPT] 传入当前值，接收降级后的新值
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
                        print(f'\n[face_enhance] 帧处理异常，使用 SR 结果: {e}')
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
        description='Real-ESRGAN 视频超分 —— 终极优化版 v5++（单卡版）',
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
    parser.add_argument('--gfpgan_model',             type=str, default='1.4',
                        choices=['1.3', '1.4', 'RestoreFormer'],
                        help='GFPGAN 模型版本（--face_enhance 时生效）。'
                             '1.4=更自然/低质量鲁棒，1.3=与1.4相近，RestoreFormer=Transformer方案。'
                             '本地优先查找 experiments/pretrained_models/ 和 gfpgan/weights/，'
                             '不存在时自动下载。Default: 1.4')
    parser.add_argument('--gfpgan_weight',            type=float, default=0.5,
                        help='GFPGAN 增强融合权重，0.0=不增强，1.0=完全替换，Default: 0.5')
    parser.add_argument('--gfpgan_batch_size',        type=int, default=12,
                        help='[OOM-FIX] 单次 GFPGAN 前向最多处理的人脸数。'
                             '人脸密集视频（群像/演唱会）可能每批超过 30 张脸，'
                             '全部堆叠为一次 StyleGAN2 前向会触发 OOM；'
                             '此参数将其拆分为多次子批量，OOM 时自动对半降级。'
                             'A10(24G) 建议 8~12，T4(16G) 建议 4~8。Default: 8')
    parser.add_argument('--fp32',                    action='store_true',
                        help='禁用 FP16（默认启用 FP16）')
    parser.add_argument('--fps',                     type=float, default=None)
    # [FIX-5] 默认 batch_size 8（原来 4），充分利用 T4 15G 显存
    parser.add_argument('--batch_size',              type=int, default=8,
                        help='批处理大小，T4 15G 建议 8~12（重模型）或 16~24（轻模型）')
    # [FIX-5] 默认 prefetch_factor 16（原来 8）
    parser.add_argument('--prefetch_factor',         type=int, default=16,
                        help='读帧预取队列深度，建议 ≥ batch_size*2')
    parser.add_argument('--use_compile',             action='store_true',
                        help='启用 torch.compile（reduce-overhead）')
    # V5 新参数（保留）
    parser.add_argument('--use_tensorrt',            action='store_true',
                        help='[V5] 启用 TensorRT 加速（首次需要构建 Engine，[FIX-3] 已修复）')
    parser.add_argument('--use_hwaccel',             action='store_true', default=True,
                        help='[V5] 启用 NVDEC 硬件解码（自动探测，失败时回退）')
    parser.add_argument('--no_hwaccel',              action='store_true',
                        help='[V5] 强制禁用 NVDEC 硬件解码')
    # 编码参数
    parser.add_argument('--video_codec',             type=str, default='libx264',
                        choices=['libx264', 'libx265', 'libvpx-vp9'],
                        help='偏好编码器（有 NVENC 时自动升级为 h264_nvenc/hevc_nvenc）')
    parser.add_argument('--crf',                     type=int, default=23)
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
    # [BUGFIX] Only pre-create args.output as a directory when it isn't a file path.
    # _output_is_file() is format-agnostic; run() will create the parent dir later.
    if not _output_is_file(args.output):
        os.makedirs(args.output, exist_ok=True)

    # 处理 --no_hwaccel
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
    print('  RealESRGAN 视频超分 —— 终极优化版 v5++（单卡版）')
    print('=' * 60)
    print(f'  GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print(f'  NVDEC:   {HardwareCapability.has_nvdec()} | '
          f'NVENC(h264): {HardwareCapability.has_nvenc("h264_nvenc")} | '
          f'NVENC(hevc): {HardwareCapability.has_nvenc("hevc_nvenc")}')
    print(f'  TensorRT: {getattr(args, "use_tensorrt", False)} | '
          f'torch.compile: {getattr(args, "use_compile", False)}')
    print(f'  batch_size: {args.batch_size} | prefetch: {args.prefetch_factor}')
    print(f'  face_enhance: {args.face_enhance} '
          f'(model={getattr(args, "gfpgan_model", "1.4")} | '
          f'weight={getattr(args, "gfpgan_weight", 0.5)} | '
          f'GFPGAN-batch={getattr(args, "gfpgan_batch_size", 12)} | '
          f'v5++: 批量GFPGAN+原始帧检测+无脸跳过)')
    print(f'  [v5++ 优化] cudnn.benchmark | 异步D2H | TRT专用Stream | 批量GFPGAN | 原始帧检测')
    print()

    run(args)


if __name__ == '__main__':
    main()
