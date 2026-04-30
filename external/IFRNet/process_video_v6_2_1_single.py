"""
IFRNet 视频插帧处理脚本 —— 终极优化版 v6.2.1（单卡版）
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
  python process_video_v6_2_1_single.py \\
      --input input.mp4 --output output_2x.mp4 --scale 2

  # 4× 插帧，关闭 compile（跳过预热，启动更快，适合短视频）
  python process_video_v6_2_1_single.py \\
      --input input.mp4 --output output_4x.mp4 --scale 4 --no-compile

  # TensorRT 加速（首次构建 Engine，后续秒启动）
  python process_video_v6_2_1_single.py \\
      --input input.mp4 --output output.mp4 --scale 2 --use-tensorrt

  # 禁用所有加速（调试/CPU 环境）
  python process_video_v6_2_1_single.py \\
      --input input.mp4 --output output.mp4 --scale 2 \\
      --no-fp16 --no-compile --no-cuda-graph --no-hwaccel --device cpu

  # 输出性能报告
  python process_video_v6_2_1_single.py \\
      --input input.mp4 --output output.mp4 --scale 2 --report report.json

【关键参数说明】
  --scale           插帧倍数，≥2 整数（2=2×, 4=4×, 8=8×）
  --model           IFRNet_S_Vimeo90K（默认轻量）或 IFRNet_L_Vimeo90K（高质量）
  --batch-size      每批帧对数，默认 4；显存充裕时自动爬升
  --no-fp16         关闭 FP16（默认开启）
  --no-compile      关闭 torch.compile（默认开启）
  --no-cuda-graph   关闭手动 CUDA Graph（compile 激活时已自动接管）
  --use-tensorrt    启用 TRT Engine（首次构建耗时较长）
  --no-hwaccel      强制 CPU 解码（NVDEC 可用时默认启用硬解）
  --crf             编码质量，默认 23（18~28 常用范围）
  --report          JSON 报告路径（可选，记录推理延迟、硬件状态）

【注意事项】
  · 首次 torch.compile 约需 1~3 分钟，编译缓存默认存于 .torch_compile_cache/
  · TRT Engine 缓存于 .trt_cache/<tag>.trt，TRT 版本升级后需手动删除重建
  · --scale 对 Fraction(scale) 取整，非整数倍数会自动取近似整数
  · 模型路径由脚本顶部 base_dir / models_ifrnet 常量决定，部署前请修改

从多卡版精简为单 GPU 版本，移除 _ifrnet_segment_worker、
_process_multi_gpu、_process_single_fallback 及所有多进程
分段调度逻辑，保留全部单卡优化不变。

【v6.2 新增升级（在 v6.1 基础上）】
  [PIPELINE]   三级深度流水线（IFRNetPipelineRunner）：
               T1-Reader / T2-Infer（主线程）/ T3-Writer 三线程并行；
               NVDEC 解码、GPU 推理、NVENC 编码全程无串行等待。
  [PREFETCH]   GPU Tensor 预取：batch-N 推理启动后立即在 transfer_stream
               上异步 H2D 上传 batch-N+1，_infer_batch 接受预取 tensor，
               跳过 H2D 等待，彻底将 PCIe 传输与 GPU compute 并行。
  [WATCHDOG]   全流水线死锁看门狗：全队列空转超 120s 时 dump 线程栈并退出。
  [QUEUE-VIS]  进度条队列水位显示 P（pair）/R（result），方便瓶颈定位。

【v6.1 新增升级（在 v6 基础上）】
  [FIX-D2H]    异步 D2H：PinnedBufferPool 增加输出 buffer，
               用 non_blocking=True copy_ + synchronize() 替代 .cpu()，
               消除每批推理后的隐式 GPU 全同步停顿。
  [FIX-PAD]    预取线程集成 padding：FFmpegFrameReader 在后台 _read_loop
               中同时完成 pad_to_stride，主线程直接取 (raw, padded) 元组，
               padding 的 CPU 计算与上批 GPU 推理完全并行。

【v6 升级（全部继承）】
  [FIX-CU]     cuDNN benchmark 自动最优卷积算法（_load_model 中启用）
  [FIX-STREAM] 修复 FP16 数据竞争：pred_big.float() 前插入 wait_stream
  [FIX-NDV]    NVDEC 两阶段真实 H.264 探测，消除 lavfi 误报
  [FIX-NML]    _NVMLFilter 过滤 NVML_SUCCESS / CUDACachingAllocator 日志噪音

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

# ── [FIX-NML] stderr 过滤器：屏蔽 NVML_SUCCESS / CUDACachingAllocator 无害断言 ──
import re as _re, sys as _sys
class _NVMLFilter:
    _pat = _re.compile(r'NVML_SUCCESS|INTERNAL ASSERT FAILED.*CUDACachingAllocator')
    def __init__(self, s): self._s = s
    def write(self, m):
        if not self._pat.search(m): self._s.write(m)
    def flush(self): self._s.flush()
    def __getattr__(self, a): return getattr(self._s, a)
_sys.stderr = _NVMLFilter(_sys.stderr)

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

# [FIX-CUDA-GRAPH-WARP] ───────────────────────────────────────────────────────
# utils.warp 每次调用都会在 CPU 上用 torch.arange 动态生成坐标网格并触发
# H2D 复制，CUDA Graph 捕获期间会崩溃。下面的 _cached_warp 在 GPU 上缓存
# 坐标网格，捕获期间不产生任何新的 malloc / H2D 复制。
# ─────────────────────────────────────────────────────────────────────────────
import torch.nn.functional as _F_warp

_warp_grid_cache: dict = {}

def _cached_warp(img: 'torch.Tensor', flow: 'torch.Tensor') -> 'torch.Tensor':
    B, _C, H, W = img.shape
    key = (B, H, W, str(img.device), img.dtype)
    if key not in _warp_grid_cache:
        xs = torch.arange(0, W, device=img.device, dtype=img.dtype)
        ys = torch.arange(0, H, device=img.device, dtype=img.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=0)
        _warp_grid_cache[key] = grid.unsqueeze(0).expand(B, -1, -1, -1)
    base_grid = _warp_grid_cache[key]
    vgrid = base_grid + flow
    vgrid_x = 2.0 * vgrid[:, 0:1] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, 1:2] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.cat([vgrid_x, vgrid_y], dim=1).permute(0, 2, 3, 1)
    return _F_warp.grid_sample(img, vgrid_scaled,
                               mode='bilinear', padding_mode='border', align_corners=True)

# ── 按模型名动态导入对应变体 ─────────────────────────────────────────────────
MODEL_MODULE_MAP: Dict[str, str] = {
    'IFRNet_Vimeo90K':   'models.IFRNet',
    'IFRNet_S_Vimeo90K': 'models.IFRNet_S',
    'IFRNet_L_Vimeo90K': 'models.IFRNet_L',
}

def _load_ifrnet_module(model_name: str):
    """按模型名动态导入对应变体的 Model 类，并把其 warp 替换为 CUDA-Graph 安全版本。"""
    import importlib
    module_name = MODEL_MODULE_MAP.get(model_name, 'models.IFRNet_S')
    mod = importlib.import_module(module_name)
    mod.warp = _cached_warp  # 三个变体都调用 warp，统一替换
    return mod.Model, mod

# 先按 S 版占位初始化；main() 里解析完 --model 后会再次重新绑定
Model, _ifrnet_s_mod = _load_ifrnet_module('IFRNet_S_Vimeo90K')
# ─────────────────────────────────────────────────────────────────────────────

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
        """[FIX-NDV] 两阶段真实探测：先软件编码 H.264 流，再用 NVDEC 实际解码。
        避免 lavfi 测试源在某些非官方 FFmpeg build 中误报"可用"。"""
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
# IFRNetPipelineRunner（参考 pipeline.py DeepPipelineOptimizer 移植）
# ─────────────────────────────────────────────────────────────────────────────

class IFRNetPipelineRunner:
    """
    IFRNet 三级深度流水线
    ─────────────────────────────────────────────────────────────────────────
    T1 Reader  : FFmpegFrameReader → 组装 pair batch → pair_queue
                 （NVDEC 解码 + numpy pad 已在 reader 后台线程完成，
                   与 GPU 推理完全并行，reader 常驻跑在 GPU 推理前面）
    T2 Infer   : pair_queue → GPU 推理 → result_queue           （主线程）
                 • GPU Tensor 预取（[PREFETCH]）：
                   batch-N 推理启动（compute_stream 开始执行）后，立即在
                   transfer_stream 上异步 H2D 上传 batch-N+1 的 img0/img1；
                   下次 _infer_batch 调用时跳过 H2D，直接进入 compute_stream，
                   使 PCIe 带宽与 GPU 算力真正重叠（overlap）。
    T3 Writer  : result_queue → FFmpegWriter                   （子线程）
                 （FFmpegWriter 内部已有异步写帧线程；此层在其前再加一层解耦
                   缓冲，让 T2-Infer 不被 result 组装与 pbar 更新阻塞）

    死锁看门狗（[WATCHDOG]）：全队列空转超 IDLE_DEADLOCK_TIMEOUT 秒时
    dump 所有线程调用栈并强制退出，防止无声卡死。
    """

    _SENTINEL             = object()
    IDLE_DEADLOCK_TIMEOUT = 120.0   # 秒

    def __init__(
        self,
        processor: 'IFRNetVideoProcessor',
        pair_queue_size:   int = 6,
        result_queue_size: int = 8,
    ):
        self.proc         = processor
        self.pair_queue   = queue.Queue(maxsize=pair_queue_size)
        self.result_queue = queue.Queue(maxsize=result_queue_size)
        self.running      = True

        # 预取状态（仅在 T2 推理主线程访问，无需锁）
        self._prefetch_item    = None                  # 已从 pair_queue 取出等待推理的 item
        self._prefetch_img0_t: Optional[torch.Tensor] = None   # 预取的 img0 GPU tensor
        self._prefetch_img1_t: Optional[torch.Tensor] = None   # 预取的 img1 GPU tensor

        # 预取命中率统计
        self._prefetch_hits  = 0
        self._prefetch_total = 0

        self._reader_th: Optional[threading.Thread] = None
        self._writer_th: Optional[threading.Thread] = None

    # ── T1 Reader 线程 ────────────────────────────────────────────────────────

    def _reader_loop(
        self,
        reader,
        effective_bs: int,
        first_raw:    np.ndarray,
        first_padded: np.ndarray,
    ):
        """
        读帧线程：将 FFmpegFrameReader 的输出组装成 pair batch 推入 pair_queue。
        first_raw/first_padded 是已在外部读取的首帧，作为每批的左锚。
        """
        raw_buf    = [first_raw]
        padded_buf = [first_padded]
        frames_read = 1
        try:
            while self.running:
                pair = reader.read()
                if pair is None:
                    # EOF：将剩余帧组成最后一批（is_end=True）
                    if len(raw_buf) >= 2:
                        self._enqueue_pair(
                            list(raw_buf[1:]),
                            list(padded_buf[:-1]), list(padded_buf[1:]),
                            True,
                        )
                    break

                raw_buf.append(pair[0])
                padded_buf.append(pair[1])
                frames_read += 1

                if len(raw_buf) == effective_bs + 1:
                    self._enqueue_pair(
                        list(raw_buf[1:]),
                        list(padded_buf[:-1]), list(padded_buf[1:]),
                        False,
                    )
                    # 保留最后一帧作为下批的 img0
                    raw_buf    = [raw_buf[-1]]
                    padded_buf = [padded_buf[-1]]

        except Exception as e:
            import traceback
            print(f'[IFRNet-Reader] 异常 @frame={frames_read}: '
                  f'{type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
        finally:
            if not self.proc.quiet:
                print(f'[IFRNet-Reader] 退出，已读 {frames_read} 帧', flush=True)
            # 发送终止哨兵（最多等 60s）
            for _ in range(60):
                try:
                    self.pair_queue.put(self._SENTINEL, timeout=1.0)
                    break
                except queue.Full:
                    continue

    def _enqueue_pair(
        self,
        img1_raw: list,
        img0_pad: list, img1_pad: list,
        is_end:   bool,
    ):
        """[FIX-QUEUE-MEM] img0_raw 移除（=上批 img1_raw），减少 1/4 队列内存"""
        item = (img1_raw, img0_pad, img1_pad, is_end)
        while self.running:
            try:
                self.pair_queue.put(item, timeout=1.0)
                return
            except queue.Full:
                continue

    # ── GPU 预取（在 T2 推理主线程中调用）───────────────────────────────────

    def _try_prefetch_next(self):
        """
        尝试从 pair_queue 非阻塞取出下一批，在 transfer_stream 上发起异步 H2D。
        应在当前 batch compute_stream 推理**已提交**（wait_stream 之后）时立即调用，
        使 H2D_{N+1} 与 compute_N 真正并行执行。
        """
        if self._prefetch_item is not None:
            return   # 已有预取中的 batch，不重复抢占
        if self.pair_queue.empty():
            return

        try:
            item = self.pair_queue.get_nowait()
        except queue.Empty:
            return

        # 哨兵放回队列，不做预取
        if item is self._SENTINEL:
            try:
                self.pair_queue.put(item)
            except Exception:
                pass
            return

        img1_raw, img0_pad, img1_pad, is_end = item
        if not img0_pad:   # 空批次
            try:
                self.pair_queue.put(item)
            except Exception:
                pass
            return

        proc   = self.proc
        pool   = _get_pinned_pool()
        stream = proc.stream_transfer
        device = proc.device
        dtype  = proc.dtype

        try:
            if stream is not None:
                with torch.cuda.stream(stream):
                    # slot=0/1 与 _infer_batch 保持一致，各用独立 pinned buffer
                    img0_pin = pool.get_for_frames(img0_pad, to_rgb=True, slot=0)
                    img0_t   = img0_pin.to(device, non_blocking=True, dtype=dtype)
                    img1_pin = pool.get_for_frames(img1_pad, to_rgb=True, slot=1)
                    img1_t   = img1_pin.to(device, non_blocking=True, dtype=dtype)
            else:
                img0_t = pool.get_for_frames(img0_pad, to_rgb=True, slot=0).to(device, dtype=dtype)
                img1_t = pool.get_for_frames(img1_pad, to_rgb=True, slot=1).to(device, dtype=dtype)

            self._prefetch_item    = item
            self._prefetch_img0_t  = img0_t
            self._prefetch_img1_t  = img1_t
        except Exception as e:
            print(f'[IFRNet-Prefetch] H2D 预取失败: {e}，放回队列', flush=True)
            try:
                self.pair_queue.put(item)
            except Exception:
                pass

    def _pop_prefetch_or_none(self) -> 'Optional[Tuple]':
        """
        若有预取好的 batch，弹出并返回 (item, img0_t, img1_t)；
        否则返回 None（调用方需从 pair_queue 阻塞取）。
        """
        if self._prefetch_item is None:
            return None
        item   = self._prefetch_item
        img0_t = self._prefetch_img0_t
        img1_t = self._prefetch_img1_t
        self._prefetch_item    = None
        self._prefetch_img0_t  = None
        self._prefetch_img1_t  = None
        self._prefetch_hits  += 1
        self._prefetch_total += 1
        return item, img0_t, img1_t

    # ── T3 Writer 线程 ────────────────────────────────────────────────────────

    def _writer_loop(
        self,
        writer:          'FFmpegWriter',
        pbar,
        n_seg_est:       int,
        meter:           'ThroughputMeter',
        timing_ref:      list,
    ):
        """
        写帧线程：从 result_queue 取结果，写入 FFmpegWriter，更新进度条。
        含全流水线死锁看门狗。
        """
        written          = 0     # 已写输出帧数
        _idle_since      = None
        received_sentinel = False

        try:
            while self.running or not self.result_queue.empty():
                try:
                    item = self.result_queue.get(timeout=2.0)
                except queue.Empty:
                    # ── 死锁看门狗 ──────────────────────────────────────────
                    all_empty = (
                        self.pair_queue.empty() and
                        self.result_queue.empty()
                    )
                    if all_empty and not received_sentinel:
                        if _idle_since is None:
                            _idle_since = time.time()
                            print(f'[IFRNet-Writer][看门狗] 流水线空转，'
                                  f'开始计时（阈值 {self.IDLE_DEADLOCK_TIMEOUT:.0f}s）',
                                  flush=True)
                        elif time.time() - _idle_since > self.IDLE_DEADLOCK_TIMEOUT:
                            print(f'[IFRNet-Writer][看门狗] ⚠️ 流水线空转超过 '
                                  f'{self.IDLE_DEADLOCK_TIMEOUT:.0f}s，'
                                  f'判定死锁，已写 {written} 帧，强制退出。', flush=True)
                            self._dump_threads()
                            self.running = False
                            break
                    else:
                        _idle_since = None

                    if received_sentinel and self.result_queue.empty():
                        break
                    continue

                if item is self._SENTINEL:
                    received_sentinel = True
                    break

                results, img1_raw_list, is_end = item
                _idle_since = None

                # 写出：T 个插帧 + img1 原始帧（每对）
                n_pairs = len(img1_raw_list)
                for i, interps in enumerate(results):
                    for fr in interps:
                        writer.write(fr)
                        written += 1
                    writer.write(img1_raw_list[i])
                    written += 1

                # 更新进度条（tracking 输入帧数）
                if pbar is not None:
                    pbar.update(n_pairs)
                    avg_t = np.mean(timing_ref[-20:]) * 1000 if timing_ref else 0
                    pbar.set_postfix(
                        fps=f'{meter.fps():.1f}',
                        eta=f'{meter.eta(n_seg_est):.0f}s',
                        ms=f'{avg_t:.0f}',
                        P=self.pair_queue.qsize(),
                        R=self.result_queue.qsize(),
                    )

        finally:
            if not self.proc.quiet:
                print(f'[IFRNet-Writer] 退出，已写 {written} 输出帧', flush=True)
            self._written = written

    def _dump_threads(self):
        import traceback, sys as _sys2
        for tid, frame in _sys2._current_frames().items():
            print(f'\n── Thread {tid} ──', flush=True)
            traceback.print_stack(frame)

    # ── 主入口（T2 推理主线程）───────────────────────────────────────────────

    def run(
        self,
        reader,
        writer:            'FFmpegWriter',
        timesteps:         List[float],
        H:                 int,
        W:                 int,
        effective_bs:      int,
        first_raw:         np.ndarray,
        first_padded:      np.ndarray,
        skip_first_output: bool,
        pbar,
        n_seg_est:         int,
        meter:             'ThroughputMeter',
    ) -> Tuple[int, int]:
        """
        启动流水线，阻塞直到全部帧处理完毕。
        返回 (额外原始帧数, 额外输出帧数)（不含已在外部写入的首帧）。
        """
        proc = self.proc

        proc._pipeline_runner = self  # [FIX-PREFETCH-TIMING]
        print(
            f'[IFRNet-Pipeline] 启动深度流水线 | '
            f'pair_queue={self.pair_queue.maxsize} '
            f'result_queue={self.result_queue.maxsize} '
            f'effective_bs={effective_bs} '
            f'T={len(timesteps)}×',
            flush=True,
        )

        # ── 启动 T1 Reader 线程 ──────────────────────────────────────────────
        self._reader_th = threading.Thread(
            target=self._reader_loop,
            args=(reader, effective_bs, first_raw, first_padded),
            daemon=True,
            name='IFRNet-Reader',
        )
        self._reader_th.start()

        # ── 启动 T3 Writer 线程 ──────────────────────────────────────────────
        self._written = 0
        self._writer_th = threading.Thread(
            target=self._writer_loop,
            args=(writer, pbar, n_seg_est, meter, proc._timing),
            daemon=True,
            name='IFRNet-Writer',
        )
        self._writer_th.start()

        # ── T2 推理主循环 ─────────────────────────────────────────────────────
        fc_extra = 0   # 额外处理的原始帧数（pair 数）
        oc_extra = 0   # 额外输出帧数

        try:
            while self.running:
                # 优先使用预取好的 batch
                prefetch_result = self._pop_prefetch_or_none()

                if prefetch_result is not None:
                    item, pfimg0_t, pfimg1_t = prefetch_result
                else:
                    pfimg0_t = pfimg1_t = None
                    self._prefetch_total += 1
                    # 阻塞等待 pair_queue
                    try:
                        item = self.pair_queue.get(timeout=2.0)
                    except queue.Empty:
                        # Reader 已死且队列空 → 正常结束
                        if not self._reader_th.is_alive():
                            break
                        continue

                # 哨兵 → 结束
                if item is self._SENTINEL:
                    break

                img1_raw, img0_pad, img1_pad, is_end = item
                if not img1_raw:
                    continue

                B = len(img0_pad)

                # ── GPU 推理（含 OOM 降级，预取 tensor 透传）────────────────
                results = proc._safe_infer(
                    img0_pad, img1_pad, timesteps, H, W,
                    prefetched_img0_t=pfimg0_t,
                    prefetched_img1_t=pfimg1_t,
                )

                # [FIX-PREFETCH-TIMING] 预取已在 _infer_batch compute 提交后触发

                # ── 结果入队 ──────────────────────────────────────────────────
                out_item = (results, img1_raw, is_end)
                _t0_full = None
                while self.running:
                    try:
                        self.result_queue.put(out_item, timeout=0.5)
                        break
                    except queue.Full:
                        # [FIX-WRITER-BACKPRESSURE] result_queue 满 → GPU 空转
                        if _t0_full is None:
                            _t0_full = time.time()
                        elif time.time() - _t0_full > 3.0:
                            print(f'[IFRNet][背压] result_queue 满 '
                                  f'{time.time()-_t0_full:.1f}s，GPU 空转等 Writer',
                                  flush=True)
                            _t0_full = time.time()

                fc_extra += B
                oc_extra += B * (len(timesteps) + 1)
                meter.update(B)

        except Exception as e:
            import traceback
            print(f'[IFRNet-Infer] 推理主循环异常: {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
        finally:
            self.running = False
            # 发送 Writer 哨兵
            for _ in range(10):
                try:
                    self.result_queue.put(self._SENTINEL, timeout=1.0)
                    break
                except queue.Full:
                    continue

        proc._pipeline_runner = None  # [FIX-PREFETCH-TIMING]

        # ── 等待 Writer/Reader 线程退出 ──────────────────────────────────────
        if self._writer_th and self._writer_th.is_alive():
            self._writer_th.join(timeout=30.0)
            if self._writer_th.is_alive():
                print('[IFRNet-Writer] ⚠️ 线程未在 30s 内退出', flush=True)

        if self._reader_th and self._reader_th.is_alive():
            self._reader_th.join(timeout=10.0)

        # ── 预取命中率报告 ────────────────────────────────────────────────────
        if self._prefetch_total > 0 and not self.proc.quiet:
            hit_pct = self._prefetch_hits / self._prefetch_total * 100
            print(
                f'[IFRNet-Pipeline] 预取命中率: '
                f'{self._prefetch_hits}/{self._prefetch_total} ({hit_pct:.1f}%)',
                flush=True,
            )

        oc_extra = self._written  # 以 Writer 线程实际写出为准
        return fc_extra, oc_extra

    def close(self):
        self.running = False

# ─────────────────────────────────────────────────────────────────────────────
# PinnedBufferPool（v4 复用，线程本地）
# ─────────────────────────────────────────────────────────────────────────────

_thread_local = threading.local()


class PinnedBufferPool:
    # [FIX-RACE] 双 buffer 方案：img0 和 img1 各用独立的 pinned buffer（slot 0/1），
    # 消除非阻塞 DMA 与 CPU 写入同一 buffer 的数据竞争，修复频闪根因。
    def __init__(self):
        self._bufs:    list = [None, None]  # slot 0 → img0, slot 1 → img1
        self._out_buf: Optional[torch.Tensor] = None   # [FIX-D2H] 异步 D2H 输出 buffer

    def get_for_frames(self, frames: List[np.ndarray],
                       to_rgb: bool = True, slot: int = 0) -> torch.Tensor:
        arr = np.stack(frames, axis=0)
        if to_rgb:
            arr = arr[:, :, :, ::-1]
        arr = np.ascontiguousarray(arr)
        src   = torch.from_numpy(arr)
        src_f = src.permute(0, 3, 1, 2).float().div_(255.0).contiguous()
        n = src_f.numel()
        if self._bufs[slot] is None or self._bufs[slot].numel() < n:
            self._bufs[slot] = torch.empty(n, dtype=torch.float32).pin_memory()
        dst = self._bufs[slot][:n].view_as(src_f)
        dst.copy_(src_f)
        return dst

    def get_output_buf(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """[FIX-D2H] 返回与 shape/dtype 匹配的 pinned 输出 buffer，按需扩容。"""
        n_elem = 1
        for s in shape:
            n_elem *= s
        if (self._out_buf is None
                or self._out_buf.dtype != dtype
                or self._out_buf.numel() < n_elem):
            self._out_buf = torch.empty(n_elem, dtype=dtype).pin_memory()
        return self._out_buf[:n_elem].view(shape)


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


def frames_to_tensor(frames, device, stream=None, dtype=torch.float32, slot: int = 0):
    # slot 0/1 selects which pinned buffer to use (img0 vs img1),
    # preventing DMA races when img0 and img1 are transferred concurrently.
    pool  = _get_pinned_pool()
    cpu_t = pool.get_for_frames(frames, to_rgb=True, slot=slot)
    ctx   = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with ctx:
        return cpu_t.to(device, non_blocking=True, dtype=dtype)


def tensor_to_np(t, orig_H, orig_W, sync_stream=None) -> List[np.ndarray]:
    """[FIX-D2H] 异步 D2H：用 non_blocking copy_ + synchronize() 替代 .cpu()，
    消除隐式 cudaDeviceSynchronize 阻塞（每批次节省 2-8ms）。"""
    if sync_stream is not None and torch.cuda.is_available():
        torch.cuda.current_stream().wait_stream(sync_stream)
    # GPU 上完成类型转换
    arr_gpu = t.clamp_(0.0, 1.0).mul_(255.0).round_().byte()  # [FIX-ROUND] 四舍五入
    arr_perm = arr_gpu.permute(0, 2, 3, 1).contiguous()   # [B, H, W, C]，仍在 GPU
    # 申请 pinned 输出 buffer，异步 DMA GPU→主机
    pool = _get_pinned_pool()
    out_pinned = pool.get_output_buf(arr_perm.shape, torch.uint8)
    out_pinned.copy_(arr_perm, non_blocking=True)          # 发起 DMA，不阻塞 CPU
    # 显式 synchronize：仅等待 DMA 完成，可与其他 CPU 工作重叠
    device = t.device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    arr = out_pinned.numpy()                               # 零拷贝视图
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
        pad_stride:      int = 0,         # [FIX-PAD] >0 则后台线程执行 padding
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

        # [FIX-PAD] 预计算 padding 量，后台线程直接产出 (raw, padded) 元组
        self._pad_stride = pad_stride
        if pad_stride > 0:
            def _ceil(x, s): return x if x % s == 0 else x + (s - x % s)
            ph = _ceil(self.height, pad_stride) - self.height
            pw = _ceil(self.width,  pad_stride) - self.width
        else:
            ph = pw = 0
        self._pad_h = ph
        self._pad_w = pw
        self.need_pad = ph > 0 or pw > 0

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
        # [FIX-PAD] 若 pad_stride>0，后台线程在解码后立即执行 padding，
        # 与上一批次 GPU 推理并行，消除主线程 CPU padding 等待。
        # 始终产出 (raw_frame, padded_frame) 元组。
        pad_h, pad_w = self._pad_h, self._pad_w
        do_pad = self.need_pad
        try:
            while True:
                raw = self._proc.stdout.read(self._frame_bytes)
                if len(raw) < self._frame_bytes:
                    break
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3).copy()
                if do_pad:
                    padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
                    self._queue.put((arr, padded))
                else:
                    self._queue.put((arr, arr))   # 无需 pad，两者共享同一数组
        except Exception as e:
            self._queue.put(e)
            return
        self._queue.put(self._SENTINEL)

    def read(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """返回 (raw_frame, padded_frame) 元组；视频结束时返回 None。
        [FIX-PAD] padded_frame 已在后台线程完成 padding，主线程零额外 CPU 开销。
        当 pad_stride=0 时，两个数组指向同一对象。"""
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
        preset: str = None,          # 新增
        audio_src: Optional[str] = None,
        ffmpeg_bin: str = 'ffmpeg',
    ):
        self._error: Optional[Exception] = None
        self._queue: queue.Queue = queue.Queue(maxsize=128)

        pix_fmt = 'yuv420p'

        # 确定 preset 默认值
        if preset is None:
            if 'nvenc' in codec:
                preset = 'p4'
            else:
                preset = 'medium'

        if 'nvenc' in codec:
            # NVENC CQ 模式，增加 -preset
            quality_args = ['-preset', preset, '-rc:v', 'vbr', '-cq:v', str(crf), '-b:v', '0']
        elif codec == 'libx265':
            # libx265 同时支持 -preset 和 x265-params
            quality_args = ['-preset', preset, '-crf', str(crf),
                            '-x265-params', 'pools=none']
        else:  # libx264 等其他软件编码器
            quality_args = ['-preset', preset, '-crf', str(crf)]

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
            stderr_out = '\n'.join(self._stderr_lines[-20:])  # 已由 _drain_stderr 实时消费
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
        x264_preset:      str = 'medium',   # 新增，软件编码默认 medium
        keep_audio:       bool = True,
        ffmpeg_bin:       str = 'ffmpeg',
        report_json:      Optional[str] = None,
        trt_cache_dir:    Optional[str] = None,
        quiet:            bool = True,
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
        self.x264_preset     = x264_preset
        self.keep_audio      = keep_audio
        self.ffmpeg_bin      = ffmpeg_bin
        self.report_json     = report_json
        self.dtype           = torch.float16 if self.use_fp16 else torch.float32
        # [FIX-TRT-CACHE-DIR] 允许外部指定 TRT 缓存目录（如 ifrnet_processor 传入稳定路径），
        # 不指定时在 process_video() 中回退到 base_dir/.trt_cache 默认规则。
        self.trt_cache_dir   = trt_cache_dir
        self.quiet           = quiet
        self._pipeline_runner: 'Optional[IFRNetPipelineRunner]' = None  # [FIX-PREFETCH-TIMING]

        self._pool          = TensorPool()
        self._graph:        dict = {}
        self._graph_inputs: dict = {}
        self._timing:       List[float] = []

        # [FIX-TRT-MUTEX] TRT 与手动 CUDA Graph / torch.compile 三选一严格互斥：
        # TRT 激活时推理全走 TRT 分支，手动 CUDA Graph 和 compile 路径永远不会执行，
        # 但若不在此处提前禁用，两者仍会被初始化（compile 触发耗时编译，
        # CUDA Graph 在 _infer_batch 中优先于 TRT 被执行），造成资源浪费或静默走错路径。
        # 在加载模型之前统一禁用，保证后续所有初始化逻辑行为一致。
        if self.use_tensorrt:
            if self.use_cuda_graph:
                self.use_cuda_graph = False
                print('  [FIX-TRT-MUTEX] use_tensorrt=True → 已禁用手动 CUDA Graph（互斥）')
            if use_compile:
                use_compile = False
                print('  [FIX-TRT-MUTEX] use_tensorrt=True → 已跳过 torch.compile（互斥，避免无效编译耗时）')

        self.use_compile = use_compile
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

        # [FIX-CU] cuDNN benchmark：IFRNet 输入尺寸在同段内固定，
        # cuDNN 首次遇到该 shape 时自动测速并缓存最优卷积算法，后续批次零开销。
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled   = True
            print('  [FIX-CU] cudnn.benchmark = True 已启用')
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
                if self.use_tensorrt:
                    print(f'  torch.compile 已加载（TRT 激活时推理走 TRT 分支，compile 不执行）')
                else:
                    print(f'  torch.compile 加速已启用 (mode=default, dynamic=True)')
                    print(f'  编译缓存目录: {cache_dir}')
                    print(f'  首次运行将触发编译（约1-3分钟），后续运行秒启动')
                if self.use_cuda_graph:
                    self.use_cuda_graph = False
                    if not self.use_tensorrt:
                        print('  手动 CUDA Graph 已禁用（由 torch.compile 接管）')
            except Exception as e:
                print(f'  torch.compile 不可用: {e}')
                # [FIX-TRT-MUTEX] compile 异常退出时若 use_tensorrt 同时开启，
                # use_cuda_graph 未经 compile 成功路径重置，需在此补充禁用，
                # 否则 _infer_batch 中 use_cuda_graph 分支会静默优先于 TRT 执行。
                if self.use_tensorrt and self.use_cuda_graph:
                    self.use_cuda_graph = False
                    print('  [FIX-TRT-MUTEX] compile 异常 + use_tensorrt=True → 补充禁用手动 CUDA Graph')
        self.model = model

        if device.type == 'cuda':
            self.stream_compute  = torch.cuda.Stream(device=device)
            self.stream_transfer = torch.cuda.Stream(device=device)
        else:
            self.stream_compute = self.stream_transfer = None

    # ──────────────────────────────────────────────────────────────────────────
    # M4: TensorRT 封装（可选）
    # ──────────────────────────────────────────────────────────────────────────

    def _build_trt_engine(self, input_shape: Tuple[int, int, int, int], cache_dir: str,
                          _rebuild_attempt: bool = False):
        """构建或加载 TRT Engine（单对帧推理形状）。

        包含 GPU SM 标记与构建进度心跳，确保跨架构缓存自动重建。
        """
        try:
            import tensorrt as trt
        except ImportError:
            print('[TensorRT] 未安装，跳过 TRT 加速。')
            self.use_tensorrt = False
            return

        os.makedirs(cache_dir, exist_ok=True)
        B, C, H, W = input_shape

        # ── 生成当前 GPU 的 SM 标记（防止跨架构缓存复用）─────────────────
        _sm_tag = ''
        if torch.cuda.is_available():
            _props = torch.cuda.get_device_properties(0)
            import re as _re_sm
            _gpu_slug = _re_sm.sub(r'[^a-z0-9]', '', _props.name.lower())[:16]
            _sm_tag = f'_sm{_props.major}{_props.minor}_{_gpu_slug}'
        # ──────────────────────────────────────────────────────────────────

        tag      = f'ifrnet_B{B}_H{H}_W{W}_fp{"16" if self.use_fp16 else "32"}{_sm_tag}'
        trt_path = os.path.join(cache_dir, f'{tag}.trt')
        onnx_path = os.path.join(cache_dir, f'{tag}.onnx')

        # ── 加载阶段：若缓存存在，先校验文件名中的 SM 标记 ───────────────
        if os.path.exists(trt_path):
            if _sm_tag and _sm_tag not in os.path.basename(trt_path):
                print(f'[TensorRT] 缓存文件缺少当前 GPU 标记 {_sm_tag}，'
                      f'可能是旧版本或跨 GPU 遗留，将删除并重建: {trt_path}')
                try:
                    os.remove(trt_path)
                except OSError:
                    pass
                # 也尝试删除对应的 onnx 缓存（如有）
                if os.path.exists(onnx_path):
                    try:
                        os.remove(onnx_path)
                    except OSError:
                        pass
                # 继续执行构建流程
            else:
                print(f'[TensorRT] 加载缓存 Engine: {trt_path}')
                # 直接跳到引擎加载部分
                pass
        # ──────────────────────────────────────────────────────────────────

        # 如果文件不存在（或刚被删除），执行构建
        if not os.path.exists(trt_path):
            print(f'[TensorRT] 构建 Engine (shape={input_shape}) ...')
            # 导出 ONNX
            dummy0 = torch.randn(*input_shape, device=self.device)
            dummy1 = torch.randn(*input_shape, device=self.device)
            embt   = torch.full((B,), 0.5, dtype=torch.float32, device=self.device).view(B, 1, 1, 1)
            if self.use_fp16:
                dummy0, dummy1, embt = dummy0.half(), dummy1.half(), embt.half()
            # [FIX-TRT] torch.compile 包装的模型导出时需解包 _orig_mod，
            # 否则权重名带 _orig_mod. 前缀导致 TRT parser 解析失败。
            # [FIX-ONNX-EXPORT] Model.forward() 是训练接口，签名为 (img0,img1,embt,imgt,flow=None)，
            # torch.onnx.export 仅传入 (img0,img1,embt) 会触发 "missing required argument: imgt" 错误。
            # 解决方案：用轻量 InferenceWrapper 封装 model.inference()，
            # 推理接口签名为 (img0,img1,embt)，返回单个 imgt_pred tensor，
            # 无 loss 计算，无额外输出节点，无 FP16/FP32 类型冲突。
            _base_model = getattr(self.model, '_orig_mod', self.model)

            class _InferenceWrapper(torch.nn.Module):
                """将 Model.inference() 包装为独立 nn.Module，供 ONNX 导出使用。"""
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, img0, img1, embt):
                    return self.m.inference(img0, img1, embt)

            export_model = _InferenceWrapper(_base_model)
            with torch.no_grad():
                torch.onnx.export(
                    export_model,
                    (dummy0, dummy1, embt),
                    onnx_path,
                    input_names=['img0', 'img1', 'embt'],
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
            # [FIX-TRT-LOGGER] TRT 全局只允许一个 Logger 实例；
            # 多次创建会触发 'logger ignored' WARNING。保存到 self._trt_logger 并复用。
            if not hasattr(self, '_trt_logger'):
                self._trt_logger = trt.Logger(trt.Logger.WARNING)
            logger  = self._trt_logger
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

            # 预估构建时间（参照 Real‑ESRGAN 脚本）
            _gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unknown'
            _sm_code = _props.major * 10 + _props.minor if torch.cuda.is_available() else 0
            _time_hint = {
                75: '约需 20~30 分钟（T4/RTX20系 SM7.5）',
                80: '约需 10~20 分钟（A100/A30 SM8.0）',
                86: '约需 5~15 分钟（A10/RTX30系 SM8.6）',
                89: '约需 5~10 分钟（RTX40系 SM8.9）',
                90: '约需 3~8 分钟（H100 SM9.0）',
            }.get(_sm_code, f'约需 5~20 分钟（{_gpu_name} SM{_props.major}.{_props.minor}）')
            print(f'[TensorRT] {_time_hint}')

            # 启动心跳线程，每 300 秒报告进度
            _build_start = time.time()
            _build_done = threading.Event()
            def _heartbeat():
                _last = time.time()
                while not _build_done.wait(timeout=5):
                    if time.time() - _last >= 300:
                        elapsed = time.time() - _build_start
                        print(f'[TensorRT] 编译中... {elapsed:.0f}s（仍在运行，请耐心等待）', flush=True)
                        _last = time.time()
            _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
            _hb_thread.start()

            serialized = builder.build_serialized_network(network, config)
            _build_done.set()
            _build_elapsed = time.time() - _build_start
            del config, parser, network, builder
            import gc; gc.collect()

            if serialized is None:
                print('[TensorRT] Engine 构建失败，回退 PyTorch 路径。')
                self.use_tensorrt = False
                return

            with open(trt_path, 'wb') as f:
                f.write(serialized)
            print(f'[TensorRT] Engine 已缓存（用时 {_build_elapsed:.0f}s）: {trt_path}')

        # ── 加载 Engine（含 deserialize / context 双重 None 防御）─────────
        try:
            if not hasattr(self, '_trt_logger'):
                self._trt_logger = trt.Logger(trt.Logger.WARNING)
            logger  = self._trt_logger
            runtime = trt.Runtime(logger)
            with open(trt_path, 'rb') as f:
                self._trt_engine = runtime.deserialize_cuda_engine(f.read())

            # ── [FIX-TRT-DESER] deserialize_cuda_engine 防御 ────────────────
            # deserialize_cuda_engine 在以下场景静默返回 None（不抛异常）：
            #   · GPU compute capability 不匹配（如 T4 → A10 迁移）
            #   · TRT 版本升级（8.x → 10.x），序列化格式不兼容
            #   · .trt 文件损坏（不完整写入 / 磁盘错误）
            # 若不检测，下方 create_execution_context() 在 NoneType 上调用 →
            #   AttributeError: 'NoneType' object has no attribute 'create_execution_context'
            #
            # 处理策略：
            #   首次失败 → 删除过期缓存 + 删除 ONNX + 递归重建（_rebuild_attempt=True）
            #   重建后仍失败 → 放弃 TRT，graceful 回退 PyTorch 路径
            if self._trt_engine is None:
                if _rebuild_attempt:
                    print('[TensorRT] ⚠️  重建后 Engine 仍反序列化失败，'
                          '回退 PyTorch 推理路径。')
                    self.use_tensorrt = False
                    self._trt_ok = False
                    return
                print(f'[TensorRT] Engine 反序列化失败（GPU 不兼容或 TRT 版本升级），'
                      f'删除过期缓存并重新构建: {trt_path}')
                try:
                    os.remove(trt_path)
                except OSError:
                    pass
                # 同步删除对应 ONNX（可能与旧 Engine 不匹配），重建时会重新导出
                if os.path.exists(onnx_path):
                    try:
                        os.remove(onnx_path)
                    except OSError:
                        pass
                # 递归调用：_rebuild_attempt=True 防止无限递归
                return self._build_trt_engine(input_shape, cache_dir,
                                              _rebuild_attempt=True)

            # ── [FIX-TRT-CTX-OOM] create_execution_context 防御 ────────────
            # create_execution_context() 在 GPU 显存不足时返回 None（不抛异常）。
            # 典型场景：upscale_then_interpolate 模式下，前序 ESRGan 步骤
            # 占用大量 GPU 显存（模型权重 + GFPGAN + 缓存分配器残留），
            # 导致 IFRNet TRT context 无法分配所需的激活内存。
            self._trt_context = self._trt_engine.create_execution_context()
            if self._trt_context is None:
                print('[TensorRT] ⚠️  create_execution_context() 失败'
                      '（GPU 显存不足），回退 PyTorch 推理路径。')
                print('[TensorRT] 提示: 前序处理步骤可能占用了大量显存。'
                      '可尝试减小 --batch-size-ifrnet 或使用 --no-tensorrt-ifrnet。')
                # 主动释放已加载的 engine，归还显存
                self._trt_engine = None
                self.use_tensorrt = False
                self._trt_ok = False
                # 尝试回收显存，为 PyTorch 推理路径腾出空间
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return

            # [FIX-TRT] 动态查询 engine 的实际 tensor 名，不硬编码
            n = self._trt_engine.num_io_tensors
            inputs, outputs = [], []
            for i in range(n):
                name = self._trt_engine.get_tensor_name(i)
                mode = self._trt_engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    inputs.append(name)
                else:
                    outputs.append(name)
            self._trt_input_names  = inputs   # e.g. ['img0','img1','embt']
            self._trt_output_names = outputs  # e.g. ['output']
            if not self.quiet:
                print(f'[TensorRT] inputs={inputs} outputs={outputs}')
            self._trt_ok = True
            print('[TensorRT] Engine 已激活，TRT 推理就绪。')
        except Exception as e:
            # 加载失败（如架构不匹配、文件损坏），删除缓存并标记不可用
            print(f'[TensorRT] Engine 加载失败: {e}，回退 PyTorch。')
            try:
                os.remove(trt_path)
            except OSError:
                pass
            self.use_tensorrt = False
            self._trt_ok = False

    # ──────────────────────────────────────────────────────────────────────────
    # CUDA Graph 推理（v4 B5 修复，完整保留）
    # ──────────────────────────────────────────────────────────────────────────

    def _get_cuda_graph(self, shape_key, img0, img1, embt, imgt_approx):
        if shape_key in self._graph:
            s = self._graph_inputs[shape_key]
            s['img0'].copy_(img0)
            s['img1'].copy_(img1)
            s['embt'].copy_(embt)
            self._graph[shape_key].replay()
            return s['output']

        print(f'  [CUDA Graph] 捕获 shape={shape_key} ...')
        static_img0 = img0.clone()
        static_img1 = img1.clone()
        static_embt = embt.clone()

        # [FIX-CUDA-GRAPH-INFERENCE] 必须用 model.inference()，forward() 含损失计算会触发 CPU 分配。
        #
        # [FIX-CG-BENCHMARK] 根本修复：cudnn.benchmark=True 导致 FIND 在 capture context 内
        # 重新测速而崩溃（FIND was unable to find an engine）。
        # 原理：
        #   · Warmup 阶段 benchmark=True → cuDNN 运行测速、选出最优算法并写入内部 cache。
        #   · 捕获阶段切换为 benchmark=False → cuDNN 走确定性 heuristic 路径，
        #     完全跳过 FIND/测速流程，避免在 capture context 内发起受限内存分配。
        #   · 捕获完成后恢复 benchmark=True，后续普通推理仍受益于 benchmark 缓存。
        # 增加至 5 次 warmup，确保内存碎片恢复后每层 conv 算法均已充分缓存。

        # ── Warmup（benchmark=True，充分缓存每层 conv 最优算法）────────────────
        for _ in range(5):
            with torch.cuda.stream(self.stream_compute):
                _ = self.model.inference(static_img0, static_img1, static_embt)
        # 捕获前完整同步，确保所有 warmup kernel 落盘、cuDNN 算法 cache 写入完成
        torch.cuda.synchronize(self.device)

        # ── CUDA Graph 捕获（benchmark=False 防止 FIND 在捕获内触发测速）─────────
        g = torch.cuda.CUDAGraph()
        _saved_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False  # [FIX-CG-BENCHMARK]
        try:
            with torch.cuda.graph(g, stream=self.stream_compute):
                static_output = self.model.inference(static_img0, static_img1, static_embt)
        except Exception as e:
            # [FIX-CG-FALLBACK] 捕获失败时的兜底：同步、清理、禁用 CUDA Graph、回退普通推理
            torch.backends.cudnn.benchmark = _saved_benchmark
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                pass
            torch.cuda.empty_cache()
            self.use_cuda_graph = False
            print(f'  [CUDA Graph] 捕获失败（{type(e).__name__}: {str(e)[:120]}），'
                  f'已禁用 CUDA Graph，本批及后续均走普通推理路径。')
            # warmup 已把权重数据写入 static tensor，直接用原始输入做一次普通推理返回
            with torch.cuda.stream(self.stream_compute):
                return self.model.inference(img0, img1, embt)
        finally:
            torch.backends.cudnn.benchmark = _saved_benchmark  # 无论成败都恢复

        # [FIX-CUDA-GRAPH-CAPTURE-REPLAY]
        # CUDA Graph 捕获期间操作只是被录制，并不执行
        # （cudaStreamBeginCapture 官方文档：work is not executed during capture）。
        # capture 路径直接返回 static_output 拿到的是未初始化内存 → 黑帧。
        # 必须在 capture 后立即 replay 一次，让 static_output 填入真实结果，
        # 再通过 wait_stream 让调用方安全读取。
        with torch.cuda.stream(self.stream_compute):
            g.replay()

        self._graph[shape_key] = g
        self._graph_inputs[shape_key] = {
            'img0': static_img0, 'img1': static_img1,
            'embt': static_embt,
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
        prefetched_img0_t: Optional[torch.Tensor] = None,
        prefetched_img1_t: Optional[torch.Tensor] = None,
    ) -> List[List[np.ndarray]]:
        B = len(img0_list)
        T = len(timesteps)
        t0 = time.perf_counter()

        # [PIPELINE-PREFETCH] 优先使用预取的 GPU 张量
        # （IFRNetPipelineRunner 在上批 compute_stream 推理期间异步 H2D 上传）
        # slot=0/1 确保 img0/img1 各用独立 pinned buffer，消除非阻塞 DMA 数据竞争
        _use_prefetch = (
            prefetched_img0_t is not None and
            prefetched_img1_t is not None and
            prefetched_img0_t.shape[0] == B and
            prefetched_img1_t.shape[0] == B
        )
        if _use_prefetch:
            img0 = prefetched_img0_t
            img1 = prefetched_img1_t
            # 预取在 transfer_stream 上异步完成，compute_stream 须等待 DMA 结束
            if self.stream_compute is not None and self.stream_transfer is not None:
                self.stream_compute.wait_stream(self.stream_transfer)
            elif self.stream_transfer is not None:
                self.stream_transfer.synchronize()
        else:
            # [FIX-RACE] slot=0/1 确保 img0/img1 各用独立 pinned buffer，消除非阻塞 DMA 数据竞争
            img0 = frames_to_tensor(img0_list, self.device, self.stream_transfer, self.dtype, slot=0)
            img1 = frames_to_tensor(img1_list, self.device, self.stream_transfer, self.dtype, slot=1)

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
            if self._pipeline_runner is not None:  # [FIX-PREFETCH-TIMING]
                self._pipeline_runner._try_prefetch_next()
        elif getattr(self, '_trt_ok', False):
            # [FIX-TRT] TensorRT 静态 Engine 推理路径
            # ─────────────────────────────────────────────────────────────────
            # [FIX-TRT-STREAM-RACE] 频闪根因（TRT 路径）：
            #
            # 旧代码在 default stream 上执行 img0_exp.half() / img1_exp.half() 的
            # 类型转换 GPU kernel，然后立即调用 execute_async_v3(stream=stream_compute)。
            # stream_compute 只 wait 了 stream_transfer（DMA完成），从未 wait default stream，
            # 因此 TRT 在 stream_compute 上读 i0/i1 时，default stream 上的 .half() kernel
            # 可能尚未写完 → TRT 读到未完成的半精度数据 → 每批帧概率性画面错误 → 频闪。
            #
            # 修复：把所有输入准备（.half()/.contiguous()/pad/bind）全部移入
            # torch.cuda.stream(stream_compute) 上下文，使 execute_async_v3 和其所有
            # 数据依赖都处于同一条流，彻底消除跨流竞争。
            # ─────────────────────────────────────────────────────────────────
            import tensorrt as _trt2
            in_names  = getattr(self, '_trt_input_names',  ['img0', 'img1', 'embt'])
            out_names = getattr(self, '_trt_output_names', ['output'])
            engine_BT = self._trt_engine.get_tensor_shape(in_names[0])[0]
            BT        = img0_exp.shape[0]
            # C, H_p, W_p = img0_exp.shape[1], img0_exp.shape[2], img0_exp.shape[3]
            out_dtype = torch.float16 if self.use_fp16 else torch.float32
            # # 输出 buffer 在 stream_compute 外分配（纯内存分配，无 GPU kernel）
            # out_buf   = torch.empty((engine_BT, 3, H_p, W_p), dtype=out_dtype, device=self.device)
            
            # [FIX-TRT-OUT-SHAPE] 直接从 Engine 查询输出 shape 分配 out_buf，
            # 替代原来手工推导的 (engine_BT, 3, H_p, W_p)。
            #
            # 与 FIX-TRT-PAD-DIMS（process_video 中用 pad 后尺寸构建 Engine）的关系：
            #   FIX-TRT-PAD-DIMS 是根基：确保 Engine 本身以正确的 padded shape 构建，
            #   因此 get_tensor_shape 返回的就是 padded shape（H_p, W_p）。
            #   本修复是在此基础上的加固：out_buf 的 shape 直接来源于 Engine，
            #   与 Engine 永远严格自洽，不再依赖 img0_exp.shape 的间接推导。
            #
            # 为什么手工推导有隐患（即使 FIX-TRT-PAD-DIMS 已修复）：
            #   IFRNet 是帧插值模型，输入输出恰好同尺寸，手工推导目前成立；
            #   但这是对模型结构的隐含假设，一旦换用输出通道数/尺寸不同的模型
            #  （如带 alpha 的 4 通道输出），手工推导会静默算错，而 get_tensor_shape
            #   能自动适应任何模型结构变化，无需修改推理代码。
            out_shape = tuple(self._trt_engine.get_tensor_shape(out_names[0]))
            out_buf   = torch.empty(out_shape, dtype=out_dtype, device=self.device)

            _trt_stream_ctx = (torch.cuda.stream(self.stream_compute)
                               if self.stream_compute is not None else nullcontext())
            with _trt_stream_ctx:
                # 在 stream_compute 上做类型转换和 pad，与 execute_async_v3 同流，无竞争
                t_vals = timesteps * B
                embt_t = torch.tensor(t_vals, dtype=torch.float32,
                                      device=self.device).view(-1, 1, 1, 1)
                i0 = img0_exp.half().contiguous() if self.use_fp16 else img0_exp.float().contiguous()
                i1 = img1_exp.half().contiguous() if self.use_fp16 else img1_exp.float().contiguous()
                em = embt_t.half().contiguous() if self.use_fp16 else embt_t.contiguous()
                # [FIX-TRT-PAD] 最后一批不足 engine_BT 时 pad，推理后只取前 BT 帧
                if BT < engine_BT:
                    pad_n = engine_BT - BT
                    def _pad(t):
                        return torch.cat([t, t[-1:].expand(pad_n, *t.shape[1:])], 0).contiguous()
                    i0, i1, em = _pad(i0), _pad(i1), _pad(em)
                ctx = self._trt_context
                for name, buf in zip(in_names, [i0, i1, em]):
                    ctx.set_tensor_address(name, buf.data_ptr())
                ctx.set_tensor_address(out_names[0], out_buf.data_ptr())
                _dummy_bufs = []
                for _out_name in out_names[1:]:
                    _shape  = tuple(self._trt_engine.get_tensor_shape(_out_name))
                    _dtype  = self._trt_engine.get_tensor_dtype(_out_name)
                    _tdtype = torch.float16 if _dtype == _trt2.DataType.HALF else torch.float32
                    _dummy  = torch.empty(_shape, dtype=_tdtype, device=self.device)
                    ctx.set_tensor_address(_out_name, _dummy.data_ptr())
                    _dummy_bufs.append(_dummy)
                _trt_stream_handle = (self.stream_compute.cuda_stream
                                      if self.stream_compute is not None
                                      else torch.cuda.current_stream().cuda_stream)
                ctx.execute_async_v3(stream_handle=_trt_stream_handle)

            # [FIX-PREFETCH-TIMING] TRT kernel 已入 stream_compute，立即预取下批
            if self._pipeline_runner is not None:
                self._pipeline_runner._try_prefetch_next()
            # default stream 等待 stream_compute 上的 TRT 推理完成后再读输出
            if self.stream_compute is not None:
                torch.cuda.default_stream(self.device).wait_stream(self.stream_compute)
            else:
                torch.cuda.current_stream().synchronize()
            # pad 时只取前 BT 帧有效结果
            result_buf = out_buf[:BT]
            pred_big   = result_buf.float() if self.use_fp16 else result_buf
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
                pred_big    = self.model.inference(img0_exp, img1_exp, embt)
            if self._pipeline_runner is not None:  # [FIX-PREFETCH-TIMING]
                self._pipeline_runner._try_prefetch_next()

        # [FIX-STREAM] compute_stream 上的推理可能尚未完成，
        # 必须在 .float() 之前让 default_stream 等待 compute_stream，
        # 否则 pred_big.float() 在 default_stream 上操作 compute_stream 的结果 → 数据竞争。
        if self.stream_compute is not None:
            torch.cuda.default_stream(self.device).wait_stream(self.stream_compute)

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

    def _safe_infer(self, img0_list, img1_list, timesteps, orig_H, orig_W,
                    prefetched_img0_t=None, prefetched_img1_t=None):
        # _in_oom_cascade: True 表示当前处于同一次 OOM 的连锁降级中，
        # 此时不再更新 max_batch_size（避免级联惩罚）
        in_oom_cascade = False
        _first_attempt = True   # [PREFETCH] 仅首次尝试使用预取 tensor；OOM 后 tensor 无效

        while True:
            try:
                # [PREFETCH] 首次尝试透传预取 tensor，重试时清空（OOM 后显存已清）
                _p0 = prefetched_img0_t if _first_attempt else None
                _p1 = prefetched_img1_t if _first_attempt else None
                result = self._infer_batch(img0_list, img1_list, timesteps, orig_H, orig_W,
                                           prefetched_img0_t=_p0, prefetched_img1_t=_p1)
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
                _first_attempt = False      # [PREFETCH] OOM 后 tensor 引用无效，后续勿用
                prefetched_img0_t = prefetched_img1_t = None
                torch.cuda.empty_cache()
                self._pool.clear()
                self._graph.clear()
                self._graph_inputs.clear()  # [FIX-OOM-STREAM] 清理 static tensor 引用

                # [FIX-OOM-STREAM] OOM 若发生在 CUDA Graph capture 内部会使 stream_compute
                # 残留 cudaErrorStreamCaptureInvalidated 状态，导致后续 capture 持续失败。
                # 重建流确保后续所有操作（capture / replay / 普通推理）在干净 stream 上运行。
                if self.stream_compute is not None:
                    try:
                        torch.cuda.synchronize(self.device)
                    except Exception:
                        pass
                    self.stream_compute  = torch.cuda.Stream(device=self.device)
                    self.stream_transfer = torch.cuda.Stream(device=self.device)

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

            except (RuntimeError, Exception) as _cg_err:
                _first_attempt = False      # [PREFETCH] 错误后 tensor 引用无效
                prefetched_img0_t = prefetched_img1_t = None
                # [FIX-CG-RTERR] 捕获 CUDA Graph capture 引发的非 OOM GPU 错误：
                #   · "FIND was unable to find an engine" — cudnn 在 capture 内 FIND 失败
                #   · "cudaErrorStreamCaptureInvalidated" — stream 被前序 OOM 污染
                #   · "operation failed due to a previous error during capture"
                # 对策：禁用 CUDA Graph，重建 CUDA 流，用普通推理重试当前 batch。
                # 非 CUDA Graph 相关错误则直接 reraise，不静默吞掉。
                _err_s = str(_cg_err)
                _is_cg = (
                    'FIND was unable to find an engine' in _err_s
                    or 'cudaErrorStreamCaptureInvalidated' in _err_s
                    or 'operation failed due to a previous error during capture' in _err_s
                    or 'cudaErrorIllegalState' in _err_s
                    or ('AcceleratorError' in type(_cg_err).__name__ and 'capture' in _err_s)
                )
                if self.use_cuda_graph and _is_cg:
                    print(f'[CUDA Graph 错误] {type(_cg_err).__name__}: {_err_s[:200]}')
                    print('  → 禁用 CUDA Graph，重建 CUDA 流，后续走普通推理路径...')
                    self.use_cuda_graph = False
                    self._graph.clear()
                    self._graph_inputs.clear()
                    self._pool.clear()
                    torch.cuda.empty_cache()
                    if self.stream_compute is not None:
                        try:
                            torch.cuda.synchronize(self.device)
                        except Exception:
                            pass
                        self.stream_compute  = torch.cuda.Stream(device=self.device)
                        self.stream_transfer = torch.cuda.Stream(device=self.device)
                    continue  # 以 use_cuda_graph=False 重试当前 batch
                raise  # 非 CUDA Graph 错误，原样上抛

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
        # [FIX-PAD] 将 pad_stride 传入 reader，padding 在后台线程完成
        # [FIX-PAD-VAR] 直接使用 MODEL_STRIDE，无需冗余中间变量
        reader = FFmpegFrameReader(
            input_path,
            frame_start  = frame_start,
            frame_end    = frame_end,
            prefetch     = self.batch_size * 3,
            use_hwaccel  = self.use_hwaccel,
            ffmpeg_bin   = self.ffmpeg_bin,
            pad_stride   = MODEL_STRIDE,      # [FIX-PAD] 交由后台线程 pad
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
        # [FIX-BS] 局部变量 effective_bs，不修改实例状态，避免多段处理状态泄漏
        effective_bs = self.batch_size
        if free_bytes > 0:
            res_max_bs = max(1, int(free_bytes * 0.6 / bytes_per_frame))
            if effective_bs > res_max_bs:
                print(f'[分辨率限制] {W}×{H} 下 batch_size {effective_bs} → {res_max_bs}')
                effective_bs = res_max_bs
            if self._max_batch_size > res_max_bs:
                self._max_batch_size = max(effective_bs, res_max_bs)

        # [FIX-PAD] padding 已由 FFmpegFrameReader 后台线程完成；
        # 此处仅从 reader 读取预计算好的值供 flush_buf 裁剪使用
        pad_h    = reader._pad_h
        pad_w    = reader._pad_w
        need_pad = reader.need_pad

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

        # [FIX-TSTART] 端到端计时，包含 warmup 耗时，避免计时虚高
        t_start = time.time()
        
		# ── torch.compile 预热 ────────────────────────────────────────────────
        # torch.compile 的实际编译发生在第一次 forward 时，会阻塞数分钟。
        # 在开启 writer 和进度条之前做预热，让用户看到明确的"编译中"提示，
        # 而不是进度条卡在 4 帧假死。
        # Simpler reliable check: always warmup once per segment if compile is active
        # [FIX-TRT-WARMUP] TRT 激活时推理全走 TRT 分支，torch.compile 路径不会执行，
        # 跳过预热避免浪费 ~30s 编译时间。
        if self.use_compile and not getattr(self, '_warmup_done', False) and not getattr(self, '_trt_ok', False):
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
                    _out  = self.model.inference(_d0, _d1, _embt)
                    # 显式释放所有引用后再 synchronize，防止析构顺序引发堆损坏
                    del _out, _d0, _d1, _embt
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
                    # [FIX-DYNAMO] 避免 dynamo 未初始化时 reset() 二次崩溃
                    try:
                        torch._dynamo.reset()
                    except Exception:
                        pass
            self._warmup_done = True

        writer = FFmpegWriter(
            output_path, W, H, new_fps,
            codec      = use_codec,
            extra_codec_args = use_extra,
            crf        = self.crf,
            preset     = self.x264_preset,      # 新增
            audio_src  = audio_src,
            ffmpeg_bin = self.ffmpeg_bin,
        )

        frame_count  = 0
        output_count = 0
        meter        = ThroughputMeter(window=20)

        desc = f'[{worker_label}] 插帧'
        pbar = tqdm(total=n_seg_est, unit='帧', desc=desc, dynamic_ncols=True) if HAS_TQDM else None

        # ── 读取第一帧 ─────────────────────────────────────────────────────
        # [FIX-PAD] reader.read() 返回 (raw, padded) 元组
        pair = reader.read()
        if pair is None:
            print(f'[{worker_label}] 无法读取首帧')
            reader.close()
            writer.close()
            if pbar:
                pbar.close()
            return False, 0, 0
        first, first_padded = pair

        # skip_first_output=False 时正常写入首帧
        if not skip_first_output:
            writer.write(first)
            output_count += 1

        frame_count = 1
        if pbar:
            pbar.update(1)

        # ── 主处理：GPU 深度流水线 / CPU 同步回退 ──────────────────────────
        if self.device.type == 'cuda':
            # ── [PIPELINE] 三级流水线（GPU 路径）────────────────────────────
            # T1-Reader 异步读帧组 batch → T2-Infer（主线程）推理 + GPU 预取
            # → T3-Writer 异步写帧，全程无串行等待。
            pipeline = IFRNetPipelineRunner(self)
            fc_extra, oc_extra = pipeline.run(
                reader            = reader,
                writer            = writer,
                timesteps         = timesteps,
                H                 = H,
                W                 = W,
                effective_bs      = effective_bs,
                first_raw         = first,
                first_padded      = first_padded,
                skip_first_output = skip_first_output,
                pbar              = pbar,
                n_seg_est         = n_seg_est,
                meter             = meter,
            )
            frame_count  += fc_extra
            output_count += oc_extra
        else:
            # ── 同步回退路径（CPU 或调试模式）────────────────────────────────
            padded_buf = [first_padded]
            raw_buf    = [first]

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

            # [FIX-PAD] reader.read() 已在后台线程完成 padding，主线程直接解包
            while True:
                pair = reader.read()
                if pair is None:
                    break
                frame, frame_padded = pair

                frame_count += 1
                raw_buf.append(frame)
                padded_buf.append(frame_padded)

                if len(raw_buf) == effective_bs + 1:
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

        # ── 收尾 ─────────────────────────────────────────────────────────────
        if pbar:
            pbar.close()
        writer.close()
        reader.close()

        elapsed = time.time() - t_start
        print(f'[{worker_label}] 完成 | 原始帧={frame_count} → 输出帧={output_count} | '
              f'{frame_count/elapsed:.1f} 原始帧/s（含 warmup/初始化）')
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
            # [FIX-TRT-CTX-OOM] 在双步模式（upscale_then_interpolate）下，
            # 前序 ESRGan 步骤可能在 PyTorch 缓存分配器中残留大量显存。
            # 主动清理，为 TRT execution context 腾出空间。
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            meta = _probe_video(input_path)
            # [FIX-TRT-PAD-DIMS] 必须用 padding 后的尺寸（对齐到 MODEL_STRIDE=32）构建 TRT Engine。
            #
            # 背景：_infer_batch 在推理前将帧 pad 到 MODEL_STRIDE=32 的倍数（如 W=720→736）。
            # 若 Engine 以原始尺寸构建，TRT 接收到 W=736 的输入但期望 W=720 → 输入不匹配。
            #
            # 历史 bug 说明（已通过两个互补修复彻底消除）：
            #   旧代码用 W=720 构建 Engine，out_buf 却按 W=736（img0_exp.shape）手工分配：
            #     ① TRT 写入 stride=720，PyTorch 读取 stride=736 → 最后一帧读到未初始化内存
            #       → 每 batch_size×scale=48 输出帧出现绿色/黑色上下分割（频闪）。
            #
            # 现已通过两个正交修复共同消除：
            #   本修复（构建期，FIX-TRT-PAD-DIMS）：用 _trt_H/_trt_W（pad 后尺寸）构建 Engine，
            #     确保 Engine 输入/输出 shape 与实际推理尺寸完全一致；
            #   推理期修复（FIX-TRT-OUT-SHAPE，见 _infer_batch）：out_buf 直接从
            #     get_tensor_shape(out_names[0]) 查询，与 Engine 永远严格自洽，
            #     消除手工推导出错的可能性（尤其应对未来模型结构变化）。
            _trt_ceil = lambda x, s: x if x % s == 0 else x + (s - x % s)
            _trt_H    = _trt_ceil(meta['height'], MODEL_STRIDE)
            _trt_W    = _trt_ceil(meta['width'],  MODEL_STRIDE)
            sh        = (self.batch_size, 3, _trt_H, _trt_W)
            # [FIX-TRT-CACHE-DIR] 优先使用外部传入的稳定缓存目录（由 ifrnet_processor /
            # config_manager 传入）；未指定时回退到项目根目录 base_dir/.trt_cache，
            # 保持与上层调用路径完全一致，避免直接调用场景产生游离缓存。
            trt_dir = self.trt_cache_dir or os.path.join(base_dir, '.trt_cache')
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
        description='IFRNet 视频插帧 —— 终极优化版 v6.2.1（单卡版）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 基础参数
    parser.add_argument('--input',      required=True,  help='输入视频路径')
    parser.add_argument('--output',     required=True,  help='输出视频路径')
    parser.add_argument('--scale',      type=float, default=2.0, help='插帧倍数（≥2 整数）')
    parser.add_argument('--model',      default='IFRNet_S_Vimeo90K', help='模型名称或 .pth 路径')
    parser.add_argument('--device',     default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch-size', type=int, default=4)
    # 推理优化
    parser.add_argument('--no-fp16',       action='store_true', help='禁用 FP16')
    parser.add_argument('--no-compile',    action='store_true', help='禁用 torch.compile')
    parser.add_argument('--no-cuda-graph', action='store_true', help='禁用 CUDA Graph')
    parser.add_argument('--use-tensorrt',  action='store_true',
                        help='[V5] 启用 TensorRT 加速（首次需构建 Engine）')
    # ── 高优先级覆盖参数（可覆盖 config / 默认值，优先级高于上方对应参数）──────────
    parser.add_argument('--use-cuda-graph', dest='use_cuda_graph_force',
                        action='store_true', default=False,
                        help='[覆盖] 强制启用 CUDA Graph，覆盖 --no-cuda-graph / config。'
                             '与 torch.compile 互斥（compile 成功时自动禁用 CUDA Graph）；'
                             '如需确保生效，请同时指定 --no-compile。')
    parser.add_argument('--use-compile', dest='use_compile_force',
                        action='store_true', default=False,
                        help='[覆盖] 强制启用 torch.compile，覆盖 --no-compile / config。'
                             '与 --use-tensorrt 互斥（TRT 激活时 compile 被跳过）。')
    parser.add_argument('--no-tensorrt', dest='no_tensorrt',
                        action='store_true', default=False,
                        help='[覆盖] 强制禁用 TensorRT，覆盖 --use-tensorrt / config。')
    # 硬件加速
    parser.add_argument('--no-hwaccel', action='store_true',
                        help='[V5] 强制禁用 NVDEC 硬件解码')
    # 编码参数
    parser.add_argument('--codec',      default='libx264',
                        help='输出编码器（有 NVENC 时自动升级）')
    parser.add_argument('--crf',        type=int, default=23)
    parser.add_argument('--x264-preset', type=str, default='medium',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast',
                                 'medium', 'slow', 'slower', 'veryslow'],
                        help='输出编码器预设（libx264/libx265 有效，NVENC 自动使用 p4）')
    parser.add_argument('--no-audio',   action='store_true')
    parser.add_argument('--ffmpeg-bin', type=str, default='ffmpeg')
    # 调试
    parser.add_argument('--preview',          action='store_true')
    parser.add_argument('--preview-interval', type=int, default=30)
    parser.add_argument('--report',           default=None, help='JSON 性能报告路径')
    parser.add_argument('--quiet', action=argparse.BooleanOptionalAction, default=True,
                        help='静默模式（默认开启），仅显示关键信息；--no-quiet 开启详细日志')
    parser.add_argument('--trt-cache-dir',    default=None,
                        help='TRT Engine 缓存目录（覆盖默认 dirname(output)/.trt_cache）')

    args = parser.parse_args()

    # ── 高优先级覆盖参数解析（优先级：新覆盖参数 > 旧参数 > 默认值）────────────────
    # 互斥优先级由底层逻辑决定，此处只做覆盖与冲突提示，不改变互斥裁定顺序：
    #   TensorRT（最高）> torch.compile > CUDA Graph（最低）
    _cli_overrides: list = []

    # 步骤 1：--no-tensorrt 覆盖 --use-tensorrt
    if args.no_tensorrt and args.use_tensorrt:
        args.use_tensorrt = False
        _cli_overrides.append('--no-tensorrt  覆盖了  --use-tensorrt  → TensorRT 已禁用')

    # 步骤 2：--use-compile 覆盖 --no-compile
    if args.use_compile_force and args.no_compile:
        args.no_compile = False
        _cli_overrides.append('--use-compile  覆盖了  --no-compile  → torch.compile 已启用')

    # 步骤 3：--use-cuda-graph 覆盖 --no-cuda-graph
    if args.use_cuda_graph_force and args.no_cuda_graph:
        args.no_cuda_graph = False
        _cli_overrides.append('--use-cuda-graph  覆盖了  --no-cuda-graph  → CUDA Graph 已启用')

    # 步骤 4：跨参数互斥冲突预警（行为由 __init__ / _load_model 最终裁定，此处仅告知）
    # 注意：以下判断均基于步骤 1-3 解析后的"有效值"
    _effective_trt     = args.use_tensorrt
    _effective_compile = not args.no_compile
    _effective_cugraph = not args.no_cuda_graph

    if args.use_cuda_graph_force and _effective_compile and not _effective_trt:
        # compile 成功时 _load_model 会把 use_cuda_graph 强制置 False
        print('[CLI警告] --use-cuda-graph 与 torch.compile 互斥：'
              'compile 成功后 CUDA Graph 将被自动禁用。')
        print('          若要确保 CUDA Graph 生效，请同时指定 --no-compile'
              '（或 --use-cuda-graph --no-compile）。')

    if args.use_cuda_graph_force and _effective_trt:
        # TRT 在 __init__ 中优先禁用 cuda_graph
        print('[CLI警告] --use-cuda-graph 与 --use-tensorrt 互斥：'
              'TensorRT 优先，CUDA Graph 将被禁用。')
        print('          如需 CUDA Graph，请同时指定 --no-tensorrt。')

    if args.use_compile_force and _effective_trt:
        # TRT 在 __init__ 中同时跳过 compile
        print('[CLI警告] --use-compile 与 --use-tensorrt 互斥：'
              'TensorRT 优先，compile 将被跳过。')
        print('          如需 torch.compile，请同时指定 --no-tensorrt。')

    if _cli_overrides:
        print('[CLI覆盖] 以下设置已被高优先级参数覆盖：')
        for msg in _cli_overrides:
            print(f'          · {msg}')
        print()

    # 模型路径解析
    if args.model in MODEL_NAME_MAP:
        model_path = f'{models_ifrnet}/{MODEL_NAME_MAP[args.model]}'
    else:
        model_path = args.model
    if not os.path.exists(model_path):
        print(f'错误: 模型不存在 - {model_path}')
        sys.exit(1)

    # 根据 --model 选择对应的模型变体
    global Model
    Model, _ = _load_ifrnet_module(args.model)

    # 打印启动信息
    print('=' * 65)
    print('  IFRNet 视频插帧 —— 终极优化版 v6.2（单卡版）')
    print('=' * 65)
    print(f'  模型:   {args.model}')
    print(f'  设备:   {args.device} | GPU: '
          f'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    # 以步骤 1-3 解析后的有效值为准打印（反映覆盖结果）
    print(f'  FP16:   {not args.no_fp16} | '
          f'Compile: {not args.no_compile} | '
          f'CUDA Graph: {not args.no_cuda_graph} | '
          f'TensorRT: {args.use_tensorrt}')
    print(f'  NVDEC:  {HardwareCapability.has_nvdec() and not args.no_hwaccel} | '
          f'NVENC(h264): {HardwareCapability.has_nvenc("h264_nvenc")} | '
          f'NVENC(hevc): {HardwareCapability.has_nvenc("hevc_nvenc")}')
    print(f'  编码器: {args.codec} → 实际: {HardwareCapability.best_encoder(args.codec)} | '
          f'CRF: {args.crf}')
    if args.use_tensorrt:
        _tcd = args.trt_cache_dir or f'(自动: {base_dir}/.trt_cache)'
        print(f'  TRT 缓存: {_tcd}')
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
        x264_preset        = args.x264_preset,
        keep_audio         = not args.no_audio,
        ffmpeg_bin         = args.ffmpeg_bin,
        report_json        = args.report,
        trt_cache_dir      = args.trt_cache_dir,
        quiet              = getattr(args, 'quiet', True),
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
