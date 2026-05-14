"""
IFRNet 视频插帧处理脚本 —— 终极优化版 v6.3.2（单卡版）
==========================================================
基于 IFRNet（Intermediate Flow-based Recursive Network）的视频帧插值脚本，
面向单 GPU 生产环境的高性能实现。

【v6.3.2 新增修复（基于 v6.3.1）】
  [FIX-CRF0-CALIB]  _software_encode_fps 无损编码校准因子修复：crf=0（lossless）时，
                   x264 实际吞吐远低于理论模型（实测约为估算的 1/18）。新增常量
                   _CRF0_X264_CALIB_FACTOR = 0.055，使 T3 静态估算更贴近实测，
                   从而改善初始 result_queue 深度。

  [FIX-T3-FPS]     T3 写入线程实测 fps 采样：_writer_loop 新增起止时间戳，
                   段结束后计算 _t3_fps_measured；通过 _next_t3_fps_measured 跨段
                   传递，供下一段 _auto_queue_depths 用实测 T3 速度代替静态估算。

  [FIX-T3-REPORT]  T3-bottleneck 诊断报告增强：[ADAPTIVE-QUEUE] T3-bottleneck 分支
                   新增实测 T3 fps、理论估算 fps 及偏差倍数显示；NVENC 可用时显示
                   预期加速比和 Docker 设备映射提示，否则建议降低 preset/crf 参数。

  [FIX-T3-DETECT]  GPU-MONITOR 误判修复：新增 _is_t3_bottleneck() 静态检测器；
                   当 GPU 空闲占比 > 60%、P95 > 85%、稳定均值 < 30% 时，判定编码
                   器（T3）是真正瓶颈。此时不再增大 result_queue（只增 PinnedPool
                   内存压力，对提速毫无帮助），并对超大值主动缩小以回收内存。

  [FIX-T3-MEMCAP]  PinnedPool 雪球效应修复：_auto_queue_depths() 和
                   get_queue_suggestions() 均新增 PinnedPool 内存上限约束
                   （默认 _PINNED_POOL_MAX_MB=2048 MiB）；result_queue 不再无限制增大，
                   防止锁页内存随段数累积至 2 GiB+ 导致 DMA 带宽压力恶化。

  [FIX-RETUNE-SKIP] T2 RETUNE 稳定性修复：引入 _CALIB_SKIP=3 跳过段初热身 batch，
                   避免流水线未稳定时的突发性快速采样污染 T2 测量值和 T2-CACHE，
                   同时过滤 < 1ms 的明显异常值（enqueue burst 假像）。

  [FIX-CALIB-KEY]  修复 _last_calib_config 缺少 model_name 的 bug：与 run() 中
                   _current_cfg 的构造不一致，导致跨模型切换时 t2_measured_ms
                   未正确清零，复用了上一模型的 T2 缓存值。

  [FIX-LOSSLESS]   crf=0 无损参数正确映射：
                   · libx264       → -qp 0（严格逐像素无损）
                   · libx265       → -x265-params lossless=1（crf=0 在 x265 中
                                     不是无损！仅为极高质量有损）
                   · h264/hevc_nvenc → -qp 0 -b:v 0（切换至常量 QP 无损模式，
                                     去掉 -rc:v vbr / -cq:v）

【v6.3.1 新增修复（基于 v6.3.0）】
  [FIX-INFER-THREAD]  T2 推理从 run() 主线程提取为独立线程 _infer_loop()，
                       仿 ESRGAN _sr_thread 架构，消除 GIL 竞争，波形趋于平顶。

  [FIX-DOUBLEBUF-H2D] 单槽预取（_prefetch_item）→ deque 双槽；
                       _try_prefetch_next() 以 while 循环填满至 2 个 in-flight，
                       大 bs 下 H2D 等待气泡消除，GPU 利用率更平滑。

【v6.3.0 核心升级（基于 v6.2.5）】
  [STREAM-DUAL]    双 transfer stream 架构：
                   · stream_h2d  专用 H2D 预取（原 stream_transfer 职责一拆二）
                   · stream_d2h  专用 D2H 输出
                   彻底消除旧版中同一条流上 D2H 阻塞 H2D 预取的根因；
                   stream_compute.wait_stream(stream_h2d) 只等 H2D，不再被 D2H 污染。
                   float() 类型转换从 default stream 移入 stream_d2h，主线程可立即
                   提交下一批推理，消除每批约 20-50ms 的 default stream 空档。
                   实现三流全重叠：compute(N) ‖ h2d(N+1) ‖ d2h(N-1)。

  [EVENT-POOL]     CudaEventPool 预分配 CUDA Event 对象池（默认 8 个），避免
                   每批次 cudaEventCreate/Destroy 带来的约 0.5-1ms 开销；
                   T3-Writer 写完后将 Event 归还池，形成完整的复用闭环。

  [BATCH-UP]       默认 batch_size 24 → 48，充分利用 T4 30% 空闲显存，
                   理论吞吐提升约 20-30%（TRT Engine 首次运行需重建缓存）。

  [GPU-MONITOR]    后台 GPU 监测线程（1 秒采样），运行结束后打印：
                   · 完整运行利用率：均值 / P50 / P95 / 峰值 / σ / 空闲占比
                   · 稳定段（去掉前 15% 预热）同上四项
                   · 最近 30s 滑动窗口：均值 / P95
                   · 显存：均值 / P95 / 峰值
                   · 三项调优建议：batch_size / pair_queue / result_queue。

【v6.2.5 完整特性（全部继承）】
  推理加速：FP16 / torch.compile / CUDA Graph / TensorRT / OOM 自动降级
  I/O 加速：NVDEC / NVENC / 异步预取 / 批量写帧
  三级深度流水线：T1-Reader / T2-Infer / T3-Writer
  AUTO-TUNE 队列深度 / T2 持久化缓存 / RETUNE 偏差报告
  PINNED-D2H 结果零拷贝 / 死锁看门狗 / JSON 性能报告

【命令行使用示例】
  # 基础用法（FP16 + torch.compile + NVDEC/NVENC 自动启用）
  python process_video_v6_3_0_single.py \\
      --input input.mp4 --output output_2x.mp4 --scale 2

  # TensorRT 加速（bs=48，首次构建 Engine）
  python process_video_v6_3_0_single.py \\
      --input input.mp4 --output output.mp4 --scale 2 --use-tensorrt

  # 输出性能报告
  python process_video_v6_3_0_single.py \\
      --input input.mp4 --output output.mp4 --scale 2 --report report.json

【注意事项】
  · v6.3.0 升级 batch_size 默认值为 48；TRT 用户若沿用旧 .trt 缓存（bs=24），
    首次运行会因 shape 不匹配而自动删除旧缓存并重建 Engine（约需 20-30 分钟）。
  · stream_transfer 属性已拆分为 stream_h2d / stream_d2h，上层调用方如有直接
    引用 processor.stream_transfer 需改为 stream_h2d（预取）或 stream_d2h（输出）。
"""

from __future__ import annotations

import argparse
import dataclasses
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

# ── [FIX-NML] stderr 过滤器 ──────────────────────────────────────────────────
import re as _re, sys as _sys

class _NVMLFilter:
    _pat = _re.compile(r'NVML_SUCCESS|INTERNAL ASSERT FAILED.*CUDACachingAllocator')
    def __init__(self, s): self._s = s
    def write(self, m):
        if not self._pat.search(m): self._s.write(m)
    def flush(self): self._s.flush()
    def __getattr__(self, a): return getattr(self._s, a)

_sys.stderr = _NVMLFilter(_sys.stderr)

os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import logging as _logging
_logging.getLogger('torch._inductor.utils').setLevel(_logging.ERROR)
_logging.getLogger('torch.utils._sympy.interp').setLevel(_logging.ERROR)
_logging.getLogger('torch.utils._sympy').setLevel(_logging.ERROR)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── 路径配置 ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
base_dir      = str(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
models_ifrnet = os.path.join(base_dir, 'models_IFRNet', 'checkpoints')
sys.path.insert(0, os.path.join(base_dir, 'external', 'IFRNet'))
sys.path.insert(0, models_ifrnet)

# ── [FIX-CUDA-GRAPH-WARP] CUDA-Graph 安全 warp ───────────────────────────────
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

MODEL_MODULE_MAP: Dict[str, str] = {
    'IFRNet_Vimeo90K':   'models.IFRNet',
    'IFRNet_S_Vimeo90K': 'models.IFRNet_S',
    'IFRNet_L_Vimeo90K': 'models.IFRNet_L',
}

def _load_ifrnet_module(model_name: str):
    import importlib
    module_name = MODEL_MODULE_MAP.get(model_name, 'models.IFRNet_S')
    mod = importlib.import_module(module_name)
    mod.warp = _cached_warp
    return mod.Model, mod

Model, _ifrnet_s_mod = _load_ifrnet_module('IFRNet_S_Vimeo90K')

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
        fallback_map = {'h264_nvenc': 'libx264', 'hevc_nvenc': 'libx265'}
        if preferred in fallback_map:
            if cls.has_nvenc(preferred):
                return preferred
            fallback = fallback_map[preferred]
            print(f'  [警告] {preferred} 不可用，自动回退到 {fallback}')
            return fallback
        candidate = nvenc_map.get(preferred, preferred)
        if candidate != preferred and cls.has_nvenc(candidate):
            return candidate
        return preferred


# ─────────────────────────────────────────────────────────────────────────────
# ThroughputMeter
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
# [PINNED-D2H] 结果队列结构体 + 预分配 Pinned Buffer 池
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _PinnedResultItem:
    buf:      torch.Tensor
    event:    torch.cuda.Event
    B:        int
    T:        int
    orig_H:   int
    orig_W:   int
    pool:     'PinnedResultPool'
    img1_raw: list = dataclasses.field(default_factory=list)


class PinnedResultPool:
    def __init__(self, pool_size: int, max_BT: int, H_pad: int, W_pad: int):
        self._q = queue.Queue(maxsize=pool_size)
        self.pool_size = pool_size
        _mb = pool_size * max_BT * H_pad * W_pad * 3 / 1e6
        print(
            f'[PinnedResultPool] 分配 {pool_size} × ({max_BT},{H_pad},{W_pad},3) '
            f'uint8 pinned，共 {_mb:.0f} MB', flush=True,
        )
        for _ in range(pool_size):
            self._q.put(torch.empty((max_BT, H_pad, W_pad, 3),
                                    dtype=torch.uint8).pin_memory())

    def acquire(self, timeout: float = 30.0) -> torch.Tensor:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError('[PinnedResultPool] acquire 超时，T3-Writer 可能卡死')

    def release(self, buf: torch.Tensor):
        self._q.put(buf)


# ─────────────────────────────────────────────────────────────────────────────
# [EVENT-POOL] CUDA Event 对象复用池
# ─────────────────────────────────────────────────────────────────────────────

class CudaEventPool:
    """
    预分配 CUDA Event 对象池，避免每批次 cudaEventCreate 开销（约 0.5-1ms/次）。
    T2-Infer 调用 acquire() 取出事件，T3-Writer synchronize() 后 release() 归还。
    线程安全：acquire/release 均持锁操作。
    """
    def __init__(self, max_size: int = 8):
        self._events: deque = deque()
        self._lock = threading.Lock()
        for _ in range(max_size):
            self._events.append(torch.cuda.Event())

    def acquire(self) -> torch.cuda.Event:
        with self._lock:
            return self._events.popleft() if self._events else torch.cuda.Event()

    def release(self, event: torch.cuda.Event):
        with self._lock:
            self._events.append(event)


# ─────────────────────────────────────────────────────────────────────────────
# [GPU-MONITOR] 后台 GPU 监测线程（v2：滑动窗口 + 精细统计 + 队列调优建议）
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class GPUStats:
    """
    GPU 采样统计结果（完整运行 + 稳定段 + 最近滑动窗口）。

    字段说明
    ─────────────────────────────────────────────────────────────────────────
    sample_count    : 总采样次数
    duration_s      : 采样时长（秒）
    total_vram_gib  : GPU 显存总量（GiB），0.0 表示未能获取

    【完整运行利用率】
    avg_util        : 全程均值 %
    max_util        : 全程峰值 %
    p50_util        : 中位数 %（比均值更抗异常抖动）
    p95_util        : 95 分位数 %（反映高负载上限）
    util_std        : 标准差（波动性指标；越高说明 GPU 供料越不稳定）
    low_util_frac   : util < 50% 的样本占比（空闲时间比；越高说明 GPU 饥饿越严重）

    【稳定段（剔除前 15% 预热样本后的统计）】
    stable_avg      : 稳定段均值 %
    stable_p50      : 稳定段中位数 %
    stable_p95      : 稳定段 95 分位数 %
    stable_std      : 稳定段标准差 %

    【显存】
    avg_mem_gib     : 全程均值 GiB
    max_mem_gib     : 全程峰值 GiB
    p95_mem_gib     : 95 分位数 GiB

    【最近滑动窗口段（最后 window_seconds 秒）】
    recent_avg      : 最近段均值 %（反映运行末尾 GPU 状态）
    recent_p95      : 最近段 95 分位数 %

    【推导属性（不存储，按需计算）】
    mem_headroom_gib: 显存余量 = total_vram_gib - max_mem_gib
    mem_frac        : 峰值显存占比 = max_mem_gib / total_vram_gib
    """
    sample_count:   int   = 0
    duration_s:     float = 0.0
    total_vram_gib: float = 0.0

    # 完整运行利用率
    avg_util:      float = 0.0
    max_util:      float = 0.0
    p50_util:      float = 0.0
    p95_util:      float = 0.0
    util_std:      float = 0.0
    low_util_frac: float = 0.0   # 空闲时间占比 [0, 1]

    # 稳定段（剔除前 15% 预热）
    stable_avg:    float = 0.0
    stable_p50:    float = 0.0
    stable_p95:    float = 0.0
    stable_std:    float = 0.0

    # 显存
    avg_mem_gib:   float = 0.0
    max_mem_gib:   float = 0.0
    p95_mem_gib:   float = 0.0

    # 最近滑动窗口段
    recent_avg:    float = 0.0
    recent_p95:    float = 0.0

    @property
    def mem_headroom_gib(self) -> float:
        """显存余量（GiB）= total_vram - max_mem。"""
        return max(self.total_vram_gib - self.max_mem_gib, 0.0)

    @property
    def mem_frac(self) -> float:
        """峰值显存占全局显存的比例 [0, 1]。"""
        return self.max_mem_gib / max(self.total_vram_gib, 1.0)

    def summary_str(self) -> str:
        """生成多行统计摘要字符串。"""
        vram_str = (f'{self.total_vram_gib:.1f} GiB'
                    if self.total_vram_gib > 0 else '未知')
        lines = [
            f'[GPU-MONITOR] 采样 {self.sample_count} 次  时长 {self.duration_s:.0f}s  '
            f'VRAM 总量: {vram_str}',
            f'  利用率(全程)  均值={self.avg_util:.1f}%  P50={self.p50_util:.1f}%  '
            f'P95={self.p95_util:.1f}%  峰值={self.max_util:.1f}%  σ={self.util_std:.1f}%  '
            f'空闲占比={self.low_util_frac*100:.1f}%',
            f'  利用率(稳定段) 均值={self.stable_avg:.1f}%  P50={self.stable_p50:.1f}%  '
            f'P95={self.stable_p95:.1f}%  σ={self.stable_std:.1f}%',
            f'  利用率(最近段) 均值={self.recent_avg:.1f}%  P95={self.recent_p95:.1f}%',
            f'  显存           均值={self.avg_mem_gib:.2f} GiB  '
            f'P95={self.p95_mem_gib:.2f} GiB  峰值={self.max_mem_gib:.2f} GiB  '
            f'({self.mem_frac*100:.0f}% 使用  余量 {self.mem_headroom_gib:.1f} GiB)',
        ]
        return '\n'.join(lines)


class GPUMonitor:
    """
    后台线程定期采样 GPU 利用率和显存占用（v2 滑动窗口精细统计版）。

    v2 改进（相较 v1）：
    · 原始样本改为 (timestamp, util%, mem_gib) 三元组，支持任意时间窗口切片。
    · 新增线程锁（_lock）保护 _raw_samples，消除残留数据竞争。
    · get_stats() 返回 GPUStats 数据类：完整运行 + 稳定段（去预热）+ 最近滑动窗口段。
    · print_report() 打印多维度统计并给出 batch_size / pair_queue / result_queue 调优建议。
    · summary() 保留为向后兼容接口（委托 get_stats() 实现）。

    继承 v1 全部修复标签：
    · FIX-NVML-LEAK / FIX-NVML-LOCK / FIX-THREAD-RST / FIX-SNAP / FIX-DEV-IDX
    """

    _LOW_UTIL_THRESHOLD = 50.0   # util < 此值视为"空闲"采样点
    _WARMUP_FRAC        = 0.15   # 稳定段：剔除前 15% 的预热样本

    def __init__(
        self,
        device:         torch.device,
        interval:       float = 1.0,
        window_seconds: float = 30.0,   # 最近滑动窗口时长（秒）
    ):
        self.device         = device
        self.interval       = interval
        self.window_seconds = window_seconds

        self._stop_event    = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml_help_printed = threading.Event()   # FIX-NVML-LOCK

        # 带时间戳原始样本列表：List[(monotonic_t, util%, mem_gib)]
        # 受 _lock 保护（FIX-SNAP 升级版）
        self._raw_samples: List[Tuple[float, float, float]] = []
        self._lock = threading.Lock()

    # ── 向后兼容属性 ─────────────────────────────────────────────────────────

    @property
    def util_samples(self) -> List[float]:
        """兼容 v1：返回纯利用率列表快照。"""
        with self._lock:
            return [s[1] for s in self._raw_samples]

    @property
    def mem_samples(self) -> List[float]:
        """兼容 v1：返回纯显存列表快照。"""
        with self._lock:
            return [s[2] for s in self._raw_samples]

    # ── 公开接口 ─────────────────────────────────────────────────────────────

    def start(self):
        """启动后台采样（仅对 CUDA 设备有效）。"""
        if self.device.type != 'cuda':
            return
        self._stop_event.clear()                   # FIX-THREAD-RST
        with self._lock:
            self._raw_samples.clear()
        self._thread = threading.Thread(
            target=self._sample, daemon=True, name='GPUMonitor'
        )
        self._thread.start()

    def stop(self):
        """通知采样线程停止并等待结束（最长 5 秒）。"""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5.0)
            self._thread = None                    # FIX-THREAD-RST

    # ── 采样线程 ─────────────────────────────────────────────────────────────

    def _sample(self):
        """
        采样线程主循环（继承 v1 全部 FIX 标签）：
        1. FIX-DEV-IDX : 解析目标 GPU 索引，支持 cuda:N 任意编号。
        2. FIX-NVML-LEAK: 独立 nvml_initialized 标志，确保 nvmlShutdown 必定被调用。
        3. FIX-NVML-LOCK: _nvml_help_printed 用 threading.Event 保证提示仅打印一次。
        4. 采样结果以 (monotonic_time, util, mem_gib) 存入 _raw_samples（受 _lock 保护）。
        """
        # FIX-DEV-IDX
        dev_idx = self.device.index if self.device.index is not None else 0

        pynvml_module    = None
        pynvml_handle    = None
        nvml_initialized = False   # FIX-NVML-LEAK

        try:
            import pynvml
            pynvml_module = pynvml
        except ImportError:
            pass

        if pynvml_module is not None:
            try:
                pynvml_module.nvmlInit()
                nvml_initialized = True            # FIX-NVML-LEAK
                pynvml_handle = pynvml_module.nvmlDeviceGetHandleByIndex(dev_idx)
            except Exception:
                pynvml_handle = None
                # nvml_initialized 保持现值：Init 成功但 GetHandle 失败时仍须 Shutdown

        if pynvml_handle is None and torch.cuda.is_available():
            if not self._nvml_help_printed.is_set():   # FIX-NVML-LOCK
                self._nvml_help_printed.set()
                print(
                    "[GPU-MONITOR] 未检测到 nvidia-ml-py，已回退至 torch.cuda 内置 API。\n"
                    "  推荐安装 nvidia-ml-py 以获得更精确的监控数据：\n"
                    "  pip install nvidia-ml-py==13.580.65   # 匹配驱动 580.65\n"
                    "  (若驱动不同，请根据 nvidia-smi 输出选择主版本号一致的 nvidia-ml-py 版本)",
                    flush=True,
                )

        while not self._stop_event.is_set():
            util         = 0.0
            mem_used_gib = 0.0
            try:
                if pynvml_handle is not None:
                    util     = float(pynvml_module.nvmlDeviceGetUtilizationRates(pynvml_handle).gpu)
                    mem_info = pynvml_module.nvmlDeviceGetMemoryInfo(pynvml_handle)
                    mem_used_gib = mem_info.used / (1024 ** 3)
                elif torch.cuda.is_available():
                    try:                           # FIX-DEV-IDX
                        util = float(torch.cuda.utilization(dev_idx))
                    except Exception:
                        util = 0.0
                    free, total = torch.cuda.mem_get_info(dev_idx)
                    mem_used_gib = (total - free) / (1024 ** 3)
                # 两者均不可用：保持 0.0（不影响推理流程）
            except Exception:
                pass

            with self._lock:
                self._raw_samples.append((time.monotonic(), util, mem_used_gib))
            self._stop_event.wait(self.interval)

        # FIX-NVML-LEAK: nvmlInit 成功过则必须 Shutdown
        if nvml_initialized and pynvml_module is not None:
            try:
                pynvml_module.nvmlShutdown()
            except Exception:
                pass

    # ── 精细统计 ─────────────────────────────────────────────────────────────

    def get_stats(self) -> GPUStats:
        """
        返回 GPUStats（线程安全快照 → 完整统计 + 稳定段 + 最近滑动窗口）。

        计算步骤：
          1. 加锁快照 _raw_samples（FIX-SNAP 升级版）。
          2. 全量 util_arr / mem_arr 计算完整运行统计（均值/P50/P95/σ/空闲占比）。
          3. 剔除前 WARMUP_FRAC（15%）样本，得稳定段统计。
          4. 从时间戳逆向切片最近 window_seconds 秒，得最近段统计。
        """
        with self._lock:
            snap = list(self._raw_samples)   # FIX-SNAP: 快照隔离

        stats = GPUStats()
        if not snap:
            return stats

        ts_arr   = np.array([s[0] for s in snap])
        util_arr = np.array([s[1] for s in snap])
        mem_arr  = np.array([s[2] for s in snap])
        n = len(snap)

        stats.sample_count = n
        stats.duration_s   = float(ts_arr[-1] - ts_arr[0]) if n > 1 else 0.0

        # 全局显存总量
        try:
            stats.total_vram_gib = (
                torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
            )
        except Exception:
            stats.total_vram_gib = 0.0

        # ── 完整运行利用率 ──────────────────────────────────────────────────
        stats.avg_util      = round(float(np.mean(util_arr)),                  1)
        stats.max_util      = round(float(np.max(util_arr)),                   1)
        stats.p50_util      = round(float(np.percentile(util_arr, 50)),        1)
        stats.p95_util      = round(float(np.percentile(util_arr, 95)),        1)
        stats.util_std      = round(float(np.std(util_arr)),                   1)
        stats.low_util_frac = round(float(np.mean(util_arr < self._LOW_UTIL_THRESHOLD)), 3)

        # ── 稳定段（剔除前 15% 预热）──────────────────────────────────────
        trim_n = max(1, int(n * self._WARMUP_FRAC))
        if n - trim_n >= 3:
            stable = util_arr[trim_n:]
            stats.stable_avg = round(float(np.mean(stable)),             1)
            stats.stable_p50 = round(float(np.percentile(stable, 50)),  1)
            stats.stable_p95 = round(float(np.percentile(stable, 95)),  1)
            stats.stable_std = round(float(np.std(stable)),              1)
        else:
            # 样本太少，以全量代替
            stats.stable_avg = stats.avg_util
            stats.stable_p50 = stats.p50_util
            stats.stable_p95 = stats.p95_util
            stats.stable_std = stats.util_std

        # ── 显存 ────────────────────────────────────────────────────────────
        stats.avg_mem_gib = round(float(np.mean(mem_arr)),             2)
        stats.max_mem_gib = round(float(np.max(mem_arr)),              2)
        stats.p95_mem_gib = round(float(np.percentile(mem_arr, 95)),   2)

        # ── 最近滑动窗口段 ──────────────────────────────────────────────────
        if n > 1 and stats.duration_s > 0:
            cutoff_t    = ts_arr[-1] - self.window_seconds
            recent_mask = ts_arr >= cutoff_t
            recent_util = util_arr[recent_mask]
            if len(recent_util) >= 2:
                stats.recent_avg = round(float(np.mean(recent_util)),            1)
                stats.recent_p95 = round(float(np.percentile(recent_util, 95)), 1)
            else:
                stats.recent_avg = stats.avg_util
                stats.recent_p95 = stats.p95_util
        else:
            stats.recent_avg = stats.avg_util
            stats.recent_p95 = stats.p95_util

        return stats

    # ── 调优建议报告 ─────────────────────────────────────────────────────────

    @staticmethod
    def _round_bs(bs: int) -> int:
        """将 batch_size 向上取整到最近 8 的倍数（TRT / CUDA Graph shape 对齐）。"""
        return max(8, (bs + 7) // 8 * 8)

    def print_report(
        self,
        stats:            GPUStats,
        current_bs:       int,
        current_pair_q:   int,
        current_result_q: int,
    ) -> None:
        """
        打印精细统计报告，并依据以下逻辑给出三项调优建议：

        ┌─────────────────────────────────────────────────────────────────────┐
        │  batch_size 建议（综合 VRAM 余量 + 稳定段利用率）                    │
        │  · 峰值 VRAM > 87%          → 缩小 bs，防 OOM                       │
        │  · stable_avg < 55% + VRAM < 60% → 增大 bs，提升 GPU 吞吐           │
        │  · stable_avg ≥ 80% + VRAM < 58% → 微增 bs，充分利用空闲显存        │
        │  · 其余                     → 当前 bs 合适                          │
        ├─────────────────────────────────────────────────────────────────────┤
        │  pair_queue / T2 输入缓冲深度建议（综合波动性 + 空闲占比）            │
        │  · stable_std > 25 或 low_util_frac > 30% → 增大，平滑 H2D 供料     │
        │  · stable_std < 10 且空闲低且队列偏大      → 适当缩小，节省显存      │
        │  · 其余                                   → 无需调整                │
        ├─────────────────────────────────────────────────────────────────────┤
        │  result_queue / T3 输出缓冲深度建议（综合 P95 + 波动性 + 显存余量）   │
        │  · stable_p95 > 85% 且 stable_std > 20%  → 增大，解耦 T3 瓶颈       │
        │  · 显存余量 > 2 GiB 且 result_q 较小      → 可增大，改善 T2/T3 解耦  │
        │  · 其余                                   → 无需调整                │
        └─────────────────────────────────────────────────────────────────────┘
        """
        print(stats.summary_str())

        if stats.sample_count == 0:
            print('[GPU-MONITOR] 无采样数据，跳过调优建议。')
            return

        mem_frac     = stats.mem_frac
        stable_avg   = stats.stable_avg
        stable_std   = stats.stable_std
        low_frac     = stats.low_util_frac
        headroom_gib = stats.mem_headroom_gib

        # ── batch_size 建议 ──────────────────────────────────────────────────
        if mem_frac > 0.87:
            sug_bs = self._round_bs(int(current_bs * 0.82 / max(mem_frac, 0.1)))
            sug_bs = max(8, min(sug_bs, current_bs - 8))
            print(f'[GPU-MONITOR] ⚠️  VRAM 使用率 {mem_frac*100:.0f}%，'
                  f'建议减小 batch_size: {current_bs} → {sug_bs}（防 OOM）')
        elif stable_avg < 55.0 and mem_frac < 0.60:
            factor = min(2.5, 0.60 / max(mem_frac, 0.05))
            sug_bs = self._round_bs(int(current_bs * factor))
            sug_bs = min(256, max(current_bs + 8, sug_bs))
            print(f'[GPU-MONITOR] 💡 batch_size 建议增大: {current_bs} → {sug_bs}'
                  f'  （稳定利用率 {stable_avg:.0f}% 偏低，VRAM 余量 {headroom_gib:.1f} GiB）')
        elif stable_avg >= 80.0 and mem_frac < 0.58:
            factor = min(1.8, 0.58 / max(mem_frac, 0.05))
            sug_bs = self._round_bs(int(current_bs * factor))
            sug_bs = min(256, max(current_bs + 8, sug_bs))
            print(f'[GPU-MONITOR] 💡 batch_size 可微增: {current_bs} → {sug_bs}'
                  f'  （利用率 {stable_avg:.0f}%，VRAM 余量 {headroom_gib:.1f} GiB 充裕）')
        else:
            print(f'[GPU-MONITOR] ✅ batch_size={current_bs} 配置合适'
                  f'  （稳定利用率 {stable_avg:.0f}%，VRAM 占用 {mem_frac*100:.0f}%）')

        # ── pair_queue（T2 输入缓冲）建议 ───────────────────────────────────
        _pq_reasons = []
        if stable_std > 25.0:
            _pq_reasons.append(f'利用率波动大 σ={stable_std:.0f}%')
        if low_frac > 0.30:
            _pq_reasons.append(f'GPU 空闲占比 {low_frac*100:.0f}%')

        if _pq_reasons:
            sug_pq = min(8, current_pair_q + 2)
            print(f'[GPU-MONITOR] 💡 pair_queue 建议增大: {current_pair_q} → {sug_pq}'
                  f'  （{"、".join(_pq_reasons)}，增大可平滑 H2D 供料气泡）')
        elif stable_std < 10.0 and low_frac < 0.10 and current_pair_q > 4:
            sug_pq = max(3, current_pair_q - 1)
            print(f'[GPU-MONITOR] 💡 pair_queue 可适当减小: {current_pair_q} → {sug_pq}'
                  f'  （利用率稳定，σ={stable_std:.0f}%，节省显存）')
        else:
            print(f'[GPU-MONITOR] ✅ pair_queue={current_pair_q} 无需调整'
                  f'  （σ={stable_std:.0f}%，空闲占比 {low_frac*100:.0f}%）')

        # ── [FIX-T3-DETECT] result_queue（T3 输出缓冲）建议 ─────────────────
        # 先判断是否 T3-bottleneck：若是，增大 result_queue 无助于提速，
        # 反而加重 PinnedPool 锁页内存压力，应保持或缩小。
        if self._is_t3_bottleneck(stats):
            if current_result_q > 16:
                sug_rq = max(16, current_result_q - 8)
                print(f'[GPU-MONITOR] ⚠️  检测到 T3-bottleneck（编码器是真正瓶颈）：'
                      f'result_queue 建议缩小 {current_result_q} → {sug_rq}'
                      f'  （空闲占比 {stats.low_util_frac*100:.0f}%，均值 {stable_avg:.0f}%，'
                      f'增大队列无助提速，仅增加 PinnedPool 内存压力）')
            else:
                print(f'[GPU-MONITOR] ⚠️  检测到 T3-bottleneck（编码器是真正瓶颈）：'
                      f'result_queue={current_result_q} 保持不变'
                      f'  （根本瓶颈在编码器速度，应考虑换用更快的 preset 或 NVENC）')
        elif stats.stable_p95 > 85.0 and stable_std > 20.0:
            sug_rq = min(64, current_result_q + 8)
            print(f'[GPU-MONITOR] 💡 result_queue 建议增大: {current_result_q} → {sug_rq}'
                  f'  （P95={stats.stable_p95:.0f}%，σ={stable_std:.0f}%，T3 可能拖累 T2）')
        elif headroom_gib > 2.0 and current_result_q < 32:
            sug_rq = min(48, current_result_q + 8)
            print(f'[GPU-MONITOR] 💡 result_queue 可增大: {current_result_q} → {sug_rq}'
                  f'  （VRAM 余量 {headroom_gib:.1f} GiB，增大可改善 T2/T3 解耦度）')
        else:
            print(f'[GPU-MONITOR] ✅ result_queue={current_result_q} 无需调整'
                  f'  （P95={stats.stable_p95:.0f}%，余量 {headroom_gib:.1f} GiB）')

    def get_queue_suggestions(
        self,
        stats:            GPUStats,
        current_pair_q:   int,
        current_result_q: int,
        slot_mb:          float = 0.0,   # [FIX-T3-MEMCAP] 每个 result slot 的 MiB 数
    ) -> Tuple[int, int]:
        """
        [FIX-T3-DETECT / FIX-T3-MEMCAP] 返回 (建议 pair_queue, 建议 result_queue)。
        用于跨段自适应队列调整，不打印信息。

        新增逻辑：
        · T3-bottleneck 时不增大 result_queue（否则 PinnedPool 雪球式积累）。
        · slot_mb > 0 时对 result_queue 施加 PinnedPool 内存上限约束（2 GiB）。
        """
        pair_q   = current_pair_q
        result_q = current_result_q
        stable_std   = stats.stable_std
        low_frac     = stats.low_util_frac
        headroom_gib = stats.mem_headroom_gib
        stable_p95   = stats.stable_p95

        # pair_queue 建议（T3-bottleneck 不影响 pair_queue 逻辑）
        if stable_std > 25.0 or low_frac > 0.30:
            pair_q = min(8, current_pair_q + 2)
        elif stable_std < 10.0 and low_frac < 0.10 and current_pair_q > 4:
            pair_q = max(3, current_pair_q - 1)

        # [FIX-T3-DETECT] result_queue 建议：T3-bottleneck 时保持或缩小
        if self._is_t3_bottleneck(stats):
            # T3 是真正瓶颈，增大 result_queue 无助于提速，反而增加内存压力
            if current_result_q > 16:
                result_q = max(16, current_result_q - 8)
            # else: 已经很小了，保持不变
        elif stable_p95 > 85.0 and stable_std > 20.0:
            result_q = min(64, current_result_q + 8)
        elif headroom_gib > 2.0 and current_result_q < 32:
            result_q = min(48, current_result_q + 8)

        # [FIX-T3-MEMCAP] PinnedPool 内存上限约束
        if slot_mb > 0.0:
            _max_rq_by_mem = max(8, int(_PINNED_POOL_MAX_MB / slot_mb) - 2)  # -2 留给 pool overhead
            result_q = min(result_q, _max_rq_by_mem)

        return pair_q, result_q

    # ── 向后兼容接口 ─────────────────────────────────────────────────────────

    def summary(self) -> Tuple[float, float, float, float, float]:
        """
        向后兼容（v1 接口）：返回
            (avg_util, max_util, avg_mem_gib, max_mem_gib, stable_p95)

        stable_p95 替代原"窗口分割峰值均值"，语义更精确（稳定段 95 分位利用率）。
        """
        s = self.get_stats()
        return s.avg_util, s.max_util, s.avg_mem_gib, s.max_mem_gib, s.stable_p95

    # ── [FIX-T3-DETECT] T3 瓶颈检测器 ──────────────────────────────────────

    @staticmethod
    def _is_t3_bottleneck(stats: 'GPUStats') -> bool:
        """
        [FIX-T3-DETECT] 判断流水线瓶颈是否在 T3（编码器）而非 T2（推理）。

        T3-bottleneck 的 GPU 采样特征（与 T2-bottleneck 截然不同）：
          · GPU 空闲占比极高（low_util_frac > 0.60）：GPU 大多数时间在等编码器腾出
            result_queue 空位，无法持续供料；
          · P95 利用率高（stable_p95 > 85%）：偶尔爆发到 100%，与高空闲同时存在；
          · 稳定段均值极低（stable_avg < 30%）：整体利用率远低于 GPU 能力。

        这是典型的「T3 背压→T2 阻塞→GPU 爆发/空转交替」波形。
        在此状态下增大 result_queue 毫无帮助（编码器速度不变），
        仅增加锁页内存压力、拖慢 DMA 带宽。

        与 T2-bottleneck 的区分：T2 慢时 GPU 持续均匀高负载（stable_avg 高）。
        """
        return (
            stats.low_util_frac > 0.60        # GPU 空闲 > 60%
            and stats.stable_p95 > 85.0       # 但 P95 仍然爆发到 85%+（阵发性）
            and stats.stable_avg < 30.0       # 均值极低，说明绝大多数时间 GPU 空转
        )


# ─────────────────────────────────────────────────────────────────────────────
# [AUTO-TUNE] 硬件感知队列深度自动调节
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _HWProfile:
    gpu_name:       str
    gpu_tier:       float
    has_nvdec:      bool
    has_nvenc:      bool
    pcie_bw_gbs:    float
    cpu_cores:      int
    t2_measured_ms: float = 0.0

    def __str__(self) -> str:
        return (f'{self.gpu_name} tier={self.gpu_tier:.1f} '
                f'nvdec={self.has_nvdec} nvenc={self.has_nvenc} '
                f'pcie={self.pcie_bw_gbs:.0f}GB/s cpu={self.cpu_cores}c')

_GPU_PROFILES_TABLE = [
    ('H100|H800',        15.2, True,  False, 63.0),
    ('L40S',             11.3, True,  True,  31.5),
    ('A100|A800',         4.8, True,  False, 31.5),
    ('L40(?!S)',          5.6, True,  True,  31.5),
    ('A30(?!\\d)',        2.5, True,  False, 31.5),
    ('A10(?!\\d)',        1.9, True,  True,  31.5),
    ('V100',              1.7, False, False, 31.5),
    ('T4(?!\\d)',         1.0, True,  True,  15.7),
    ('RTX\\s*4090',       3.3, True,  True,  31.5),
    ('RTX\\s*4080',       2.6, True,  True,  31.5),
    ('RTX\\s*4070\\s*Ti', 2.2, True,  True,  31.5),
    ('RTX\\s*4070',       1.9, True,  True,  31.5),
    ('RTX\\s*4060',       1.4, True,  True,  15.7),
    ('RTX\\s*3090\\s*Ti', 2.3, True,  True,  31.5),
    ('RTX\\s*3090',       2.1, True,  True,  15.7),
    ('RTX\\s*3080\\s*Ti', 1.8, True,  True,  15.7),
    ('RTX\\s*3080',       1.6, True,  True,  15.7),
    ('RTX\\s*3070',       1.3, True,  True,  15.7),
    ('RTX\\s*2080\\s*Ti', 1.1, True,  True,  15.7),
    ('RTX\\s*2080',       0.9, True,  True,  15.7),
    ('GTX\\s*1080\\s*Ti', 0.5, True,  True,  15.7),
    ('GTX\\s*1080',       0.4, True,  True,  15.7),
]
_GPU_FALLBACK = (1.0, True, True, 15.7)

def _get_gpu_slug() -> str:
    if torch.cuda.is_available():
        import re as _re_sm
        props = torch.cuda.get_device_properties(0)
        slug = _re_sm.sub(r'[^a-z0-9]', '', props.name.lower())[:16]
        return f'_sm{props.major}{props.minor}_{slug}'
    return '_cpu'

_T2_CACHE_DIR_DEFAULT = os.path.join(base_dir, '.t2_cache')

def _load_t2_cache(cache_dir: str) -> dict:
    path = os.path.join(cache_dir, 't2_measured.json')
    if os.path.isfile(path):
        try:
            with open(path, 'r', encoding='utf-8') as _f:
                return json.load(_f)
        except Exception:
            return {}
    return {}

def _save_t2_cache(cache_dir: str, cache: dict):
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, 't2_measured.json'), 'w', encoding='utf-8') as _f:
        json.dump(cache, _f, indent=2)

def _detect_hw_profile(device: torch.device) -> _HWProfile:
    import re as _re
    cpu_cores = os.cpu_count() or 4
    if device.type != 'cuda':
        return _HWProfile('CPU', 0.05, False, False, 0.0, cpu_cores)
    gpu_name = torch.cuda.get_device_name(device)
    tier, has_nvdec, has_nvenc, pcie_bw = _GPU_FALLBACK
    for pat, _t, _nd, _ne, _pb in _GPU_PROFILES_TABLE:
        if _re.search(pat, gpu_name, _re.IGNORECASE):
            tier, has_nvdec, has_nvenc, pcie_bw = _t, _nd, _ne, _pb
            break
    return _HWProfile(gpu_name, tier, has_nvdec, has_nvenc, pcie_bw, cpu_cores)

# [AUTO-TUNE-CALIB] T2 双分量模型
_T2_BASELINE_HWB = float(576 * 736 * 24)
_T2_FIXED_MS     = 240.0
_T2_VAR_MS       = 25.0

_X264_PRESET_FACTOR = {
    'ultrafast': 8.0, 'superfast': 6.0, 'veryfast': 4.0,
    'faster': 2.5, 'fast': 2.0, 'medium': 1.0,
    'slow': 0.4, 'slower': 0.2, 'veryslow': 0.1,
}
# [FIX-CRF0-CALIB] crf=0（lossless）实测校准因子。
# 理论模型（crf_factor = 2^((0-23)/12) ≈ 0.264）严重低估 lossless 编码成本：
#   · lossless 需维持精确像素，内存带宽和预测搜索开销远高于有损编码
#   · 实测（T4, libx264, ultrafast, 416×736, 8c）: ~150 fps output
#   · 理论估算（修正前）:  ~2860 fps → 偏差约 19×
# 乘以此因子后估算 ~157 fps，贴近实测正常（非热节流）状态。
_CRF0_X264_CALIB_FACTOR: float = 0.055
_PINNED_POOL_MAX_MB: float = 2048.0
_MODEL_T2_FACTOR = {
    'IFRNet_S_Vimeo90K': 1.0,      # 基准（最小模型）
    'IFRNet_Vimeo90K':   1.6,      # 中型
    'IFRNet_L_Vimeo90K': 3.0,      # 大型
}

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

# [AUTO-TUNE-BACKEND] 推理后端相对 TRT 的速度因子
_INFER_BACKEND_FACTORS = {
    'trt':        1.0,
    'cuda_graph': 1.5,
    'compile':    2.0,
    'eager':      3.5,
}

def _auto_queue_depths(
    profile: _HWProfile, codec: str, x264_preset: str, crf: int,
    H_pad: int, W_pad: int, effective_bs: int, T: int,
    infer_backend: str = 'eager',
    model_name: str = 'IFRNet_S_Vimeo90K',   # 新增参数
    verbose: bool = True,
    t3_fps_measured: float = 0.0,   # [FIX-T3-FPS] 跨段实测 T3 fps（0 表示无实测，用静态估算）
) -> Tuple[int, int, int]:
    import math as _math
    HWB = float(H_pad * W_pad * effective_bs)
    if profile.has_nvdec:
        nvdec_fps = min(440.0 * 1920.0 * 1080.0 / max(float(H_pad * W_pad), 1.0), 3000.0)
    else:
        nvdec_fps = min(60.0 * 1920.0 * 1080.0 / max(float(H_pad * W_pad), 1.0), 600.0)
    t1_ms = effective_bs / nvdec_fps * 1000.0

    infer_factor = _INFER_BACKEND_FACTORS.get(infer_backend, _INFER_BACKEND_FACTORS['eager'])
    model_factor = _MODEL_T2_FACTOR.get(model_name, 1.0)
    if profile.t2_measured_ms > 0:
        t2_ms  = profile.t2_measured_ms
        t2_src = 'measured'
    else:
        t2_base = (_T2_FIXED_MS + _T2_VAR_MS * HWB / _T2_BASELINE_HWB) / max(profile.gpu_tier, 0.05)
        t2_ms   = max(t2_base * infer_factor * model_factor, 1.0)   # 乘以模型系数
        t2_src  = f'estimated(×{infer_factor}, model×{model_factor})'

    out_frames = effective_bs * (T + 1)
    codec_l = codec.lower()
    if 'nvenc' in codec_l and profile.has_nvenc:
        t3_ms  = out_frames / 3000.0 * 1000.0
        t3_src = 'NVENC'
    elif 'copy' in codec_l:
        t3_ms  = out_frames / 5000.0 * 1000.0
        t3_src = 'copy'
    elif t3_fps_measured > 0.0:
        # [FIX-T3-FPS] 优先使用跨段实测 T3 fps（含热节流等实际因素）
        t3_ms  = out_frames / t3_fps_measured * 1000.0
        t3_src = f'{codec}({x264_preset}, crf={crf}, measured={t3_fps_measured:.0f}fps)'
    else:
        fps_est = _software_encode_fps(profile.cpu_cores, H_pad, W_pad, codec, x264_preset, crf)
        t3_ms   = out_frames / fps_est * 1000.0
        t3_src  = f'{codec}({x264_preset}, crf={crf})'

    pair_depth   = max(2, min(int(_math.ceil(t2_ms / max(t1_ms, 0.1))) + 2, 8))
    result_depth = max(8, min(int(_math.ceil(t3_ms / max(t2_ms, 0.1))) + 3, 64))

    # [FIX-T3-MEMCAP] 依据 PinnedPool 内存上限约束 result_depth。
    # 每个 result slot 持有 effective_bs * T 帧的 pinned uint8 buffer。
    # 若不加约束，T3 极慢（大 T3/T2 比）时 result_depth 会达到 50+，
    # 导致 PinnedPool 分配 2 GiB+ 锁页内存，反而拖慢 DMA 带宽，形成恶性循环。
    _slot_mb     = effective_bs * T * H_pad * W_pad * 3 / 1e6 # 每 slot 的 MiB
    if _slot_mb > 0.0:
        _max_result_by_mem = max(8, int(_PINNED_POOL_MAX_MB / _slot_mb) - 2)
        result_depth = min(result_depth, _max_result_by_mem)

    pool_size    = result_depth + 2
    if verbose:
        pair_mb = pair_depth * effective_bs * 3 * H_pad * W_pad * 3 / 1e6
        pool_mb = pool_size  * effective_bs * T * H_pad * W_pad * 3 / 1e6
        print(f'[AUTO-TUNE] {profile}  backend={infer_backend}(×{infer_factor}) model={model_name}(×{model_factor})\n'
              f'  T1={t1_ms:.1f}ms  T2[{t2_src}]={t2_ms:.1f}ms  T3({t3_src})={t3_ms:.1f}ms\n'
              f'  ratio T2/T1={t2_ms/max(t1_ms,0.1):.1f}x T3/T2={t3_ms/max(t2_ms,0.1):.1f}x\n'
              f'  pair_queue={pair_depth}(~{pair_mb:.0f}MB) '
              f'result_queue={result_depth} pool={pool_size}(~{pool_mb:.0f}MB pinned)',
              flush=True)
    return pair_depth, result_depth, pool_size


# ─────────────────────────────────────────────────────────────────────────────
# IFRNetPipelineRunner（三级流水线）
# ─────────────────────────────────────────────────────────────────────────────

class IFRNetPipelineRunner:
    """
    IFRNet 三级深度流水线
    ─────────────────────────────────────────────────────────────────────────
    T1 Reader  : FFmpegFrameReader → 组 batch → pair_queue（NVDEC + pad 后台线程）
    T2 Infer   : pair_queue → GPU 推理 → result_queue（主线程）
                 [STREAM-DUAL] 预取在 stream_h2d，D2H 在 stream_d2h，独立不阻塞。
    T3 Writer  : result_queue → FFmpegWriter（子线程）
    [EVENT-POOL] T2 从池中取 Event，T3 写完后归还，消除每批次创建销毁开销。
    [WATCHDOG]   空转超 120s dump 线程栈并强制退出。
    """

    _SENTINEL             = object()
    IDLE_DEADLOCK_TIMEOUT = 120.0

    def __init__(
        self,
        processor:         'IFRNetVideoProcessor',
        pair_queue_size:   int  = 4,
        result_queue_size: int  = 8,
        auto_tune:         bool = True,
        codec:             str  = 'libx264',
        x264_preset:       str  = 'medium',
        crf:               int  = 21,
        t2_cache_dir:      str  = '',
        # 新增：外部指定的队列深度（覆盖 AUTO-TUNE）
        pair_queue_override:   Optional[int] = None,
        result_queue_override: Optional[int] = None,
        t3_fps_measured:   float = 0.0,   # [FIX-T3-FPS] 跨段实测 T3 fps
    ):
        self.proc         = processor
        self.pair_queue   = queue.Queue(maxsize=pair_queue_size)
        self.result_queue = queue.Queue(maxsize=result_queue_size)
        self.running      = True
        self.auto_tune    = auto_tune
        self.codec        = codec
        self.x264_preset  = x264_preset
        self.crf          = crf
        self._hw_profile:       Optional[_HWProfile] = None
        self._t2_estimated_ms   = 0.0
        self._last_calib_config = None
        self._cache_key: Optional[str] = None
        self.t2_cache_dir = t2_cache_dir
        self._pair_queue_override = pair_queue_override
        self._result_queue_override = result_queue_override
        self._t3_fps_measured_input = t3_fps_measured   # [FIX-T3-FPS] 跨段实测值，传给 _auto_queue_depths

        # [FIX-DOUBLEBUF-H2D] 双槽飞行中 H2D，最多保持 2 个预取 in-flight
        self._prefetch_deque: deque = deque()   # 元素: (item, img0_t, img1_t)
        self._prefetch_slot  = 0                # 轮转 slot 组: 0→pinned(0,1), 1→pinned(2,3)
        self._prefetch_hits  = 0
        self._prefetch_total = 0
        self._reader_th: Optional[threading.Thread] = None
        self._infer_th:  Optional[threading.Thread] = None   # [FIX-INFER-THREAD]
        self._writer_th: Optional[threading.Thread] = None
        self._fc_extra   = 0                    # [FIX-INFER-THREAD] 推理线程累计输入帧
        self._oc_extra   = 0                    # [FIX-INFER-THREAD] 推理线程累计输出帧

        # [EVENT-POOL] 预分配 CUDA Event 对象池
        self._event_pool = CudaEventPool(max_size=8)

    def _get_infer_backend(self) -> str:
        proc = self.proc
        if getattr(proc, '_trt_ok', False):
            return 'trt'
        if proc.use_compile:
            if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'is_compiled'):
                try:
                    if torch._dynamo.is_compiled(proc.model):
                        return 'compile'
                except Exception:
                    pass
            if hasattr(proc.model, '_orig_mod'):
                return 'compile'
        if proc.use_cuda_graph:
            return 'cuda_graph'
        return 'eager'

    # ── T1 Reader 线程 ────────────────────────────────────────────────────────

    def _reader_loop(self, reader, effective_bs, first_raw, first_padded):
        raw_buf    = [first_raw]
        padded_buf = [first_padded]
        frames_read = 1
        try:
            while self.running:
                pair = reader.read()
                if pair is None:
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
                    raw_buf    = [raw_buf[-1]]
                    padded_buf = [padded_buf[-1]]
        except Exception as e:
            import traceback
            print(f'\n[IFRNet-Reader] 异常 @frame={frames_read}: {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
        finally:
            if not self.proc.quiet:
                print(f'\n[IFRNet-Reader] 退出，已读 {frames_read} 帧', flush=True)
            for _ in range(60):
                try:
                    self.pair_queue.put(self._SENTINEL, timeout=1.0)
                    break
                except queue.Full:
                    continue

    def _enqueue_pair(self, img1_raw, img0_pad, img1_pad, is_end):
        item = (img1_raw, img0_pad, img1_pad, is_end)
        while self.running:
            try:
                self.pair_queue.put(item, timeout=1.0)
                return
            except queue.Full:
                continue

    # ── GPU 预取（[STREAM-DUAL] 使用 stream_h2d）────────────────────────────

    def _try_prefetch_next(self):
        """
        [FIX-DOUBLEBUF-H2D] 在 stream_h2d 上异步预取下一批，最多维持 2 个 in-flight。
        · 每次调用以 while 循环填满至 2 个槽，消除大 bs 下 H2D 等待气泡。
        · 轮转 pinned-buffer slot 组（0/1 ↔ 2/3），确保飞行中 DMA 不被后续请求覆盖。
        · [STREAM-DUAL] 与 stream_d2h（D2H 输出）完全独立，PCIe 全双工利用。
        """
        while len(self._prefetch_deque) < 2:
            if self.pair_queue.empty():
                return
            try:
                item = self.pair_queue.get_nowait()
            except queue.Empty:
                return
            if item is self._SENTINEL:
                # [FIX-DOUBLEBUF-SLOT] timeout put，防止队列满时阻塞 T2
                try:
                    self.pair_queue.put(item, timeout=5.0)
                except Exception:
                    pass
                return
            img1_raw, img0_pad, img1_pad, is_end = item
            if not img0_pad:
                # [FIX-DOUBLEBUF-SLOT] timeout put
                try:
                    self.pair_queue.put(item, timeout=5.0)
                except Exception:
                    pass
                return
            proc       = self.proc
            pool       = _get_pinned_pool()
            stream_h2d = proc.stream_h2d
            device     = proc.device
            dtype      = proc.dtype
            # [FIX-DOUBLEBUF-H2D] 轮转 slot 组：0→pinned(0,1)，1→pinned(2,3)
            slot_base  = self._prefetch_slot * 2
            try:
                if stream_h2d is not None:
                    with torch.cuda.stream(stream_h2d):
                        img0_pin = pool.get_for_frames(img0_pad, to_rgb=True, slot=slot_base)
                        img0_t   = img0_pin.to(device, non_blocking=True, dtype=dtype)
                        img1_pin = pool.get_for_frames(img1_pad, to_rgb=True, slot=slot_base + 1)
                        img1_t   = img1_pin.to(device, non_blocking=True, dtype=dtype)
                else:
                    img0_t = pool.get_for_frames(img0_pad, to_rgb=True, slot=slot_base).to(
                        device, dtype=dtype)
                    img1_t = pool.get_for_frames(img1_pad, to_rgb=True, slot=slot_base + 1).to(
                        device, dtype=dtype)
                self._prefetch_deque.append((item, img0_t, img1_t))
                self._prefetch_slot = 1 - self._prefetch_slot   # 切换到另一 slot 组
            except Exception as e:
                print(f'[IFRNet-Prefetch] H2D 预取失败: {e}，放回队列', flush=True)
                # [FIX-DOUBLEBUF-SLOT] 用带超时的 put 防止 pair_queue 满时 T2 永久阻塞
                try:
                    self.pair_queue.put(item, timeout=5.0)
                except Exception:
                    pass
                return   # 出错停止继续填充

    def _pop_prefetch_or_none(self):
        """[FIX-DOUBLEBUF-H2D] 从双槽 deque 中按 FIFO 顺序弹出预取结果。"""
        if not self._prefetch_deque:
            return None
        item, img0_t, img1_t = self._prefetch_deque.popleft()
        self._prefetch_hits  += 1
        self._prefetch_total += 1
        return item, img0_t, img1_t

    # ── T3 Writer 线程 ────────────────────────────────────────────────────────

    def _writer_loop(self, writer, pbar, n_seg_est, meter, timing_ref):
        written           = 0
        _idle_since       = None
        received_sentinel = False
        # [FIX-T3-FPS] T3 实际写入吞吐计时（起止时间戳，单位 monotonic）
        _t3_t_first: Optional[float] = None
        _t3_t_last:  Optional[float] = None
        try:
            while self.running or not self.result_queue.empty():
                try:
                    item = self.result_queue.get(timeout=2.0)
                except queue.Empty:
                    all_empty = (
                        self.pair_queue.empty() and
                        self.result_queue.empty()
                    )
                    if all_empty and not received_sentinel:
                        if _idle_since is None:
                            _idle_since = time.time()
                            print(f'\n[IFRNet-Writer][看门狗] 流水线空转，'
                                  f'开始计时（阈值 {self.IDLE_DEADLOCK_TIMEOUT:.0f}s）', flush=True)
                        elif time.time() - _idle_since > self.IDLE_DEADLOCK_TIMEOUT:
                            print(f'\n[IFRNet-Writer][看门狗] ⚠️ 空转超过 '
                                  f'{self.IDLE_DEADLOCK_TIMEOUT:.0f}s，判定死锁，'
                                  f'已写 {written} 帧，强制退出。', flush=True)
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

                _idle_since = None

                # FIX Writer 增加预读取 + 批量处理
                # 批量取出更多待处理结果（最多 4 个）
                items = [item]
                for _ in range(3):
                    try:
                        items.append(self.result_queue.get_nowait())
                    except queue.Empty:
                        break

                # 批量等待所有 D2H DMA 完成
                _has_sentinel = False
                for _item in items:
                    if _item is self._SENTINEL:
                        _has_sentinel = True
                        continue
                    if isinstance(_item, _PinnedResultItem):
                        _item.event.synchronize()

                # 批量写入（跳过 sentinel）
                _n_pairs_total = 0
                for _item in items:
                    if _item is self._SENTINEL:
                        continue
                    if isinstance(_item, _PinnedResultItem):
                        n_pairs = _item.B
                        _n_pairs_total += n_pairs
                        arr = _item.buf[:_item.B * _item.T].numpy()
                        for i in range(_item.B):
                            for j in range(_item.T):
                                fr = arr[i * _item.T + j,
                                         :_item.orig_H, :_item.orig_W, ::-1].copy()
                                writer.write(fr)
                                written += 1
                            writer.write(_item.img1_raw[i])
                            written += 1
                        # [EVENT-POOL] 写完后归还 Event 和 buffer
                        self._event_pool.release(_item.event)
                        _item.pool.release(_item.buf)
                    else:
                        results, img1_raw_list, is_end = _item
                        n_pairs = len(img1_raw_list)
                        _n_pairs_total += n_pairs
                        for i, interps in enumerate(results):
                            for fr in interps:
                                writer.write(fr)
                                written += 1
                            writer.write(img1_raw_list[i])
                            written += 1

                if _has_sentinel:
                    received_sentinel = True
                    break

                # [FIX-T3-FPS] 记录第一帧/最后一帧写入时间，用于计算实际 T3 fps
                _t3_now = time.monotonic()
                if _t3_t_first is None:
                    _t3_t_first = _t3_now
                _t3_t_last = _t3_now

                if pbar is not None:
                    pbar.update(_n_pairs_total)
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
                print(f'\n[IFRNet-Writer] 退出，已写 {written} 输出帧', flush=True)
            # [FIX-T3-FPS] 计算并存储 T3 实测 fps（写入时长 < 1s 时视为不可靠）
            if _t3_t_first is not None and _t3_t_last is not None:
                _t3_elapsed = _t3_t_last - _t3_t_first
                self._t3_fps_measured = written / _t3_elapsed if _t3_elapsed > 1.0 else 0.0
            else:
                self._t3_fps_measured = 0.0
            self._written = written

    def _dump_threads(self):
        import traceback
        for tid, frame in sys._current_frames().items():
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
        H_pad:             int = 0,
        W_pad:             int = 0,
    ) -> Tuple[int, int]:
        proc = self.proc
        proc._pipeline_runner = self

        if self.auto_tune and H_pad > 0 and W_pad > 0:
            if self._hw_profile is None:
                self._hw_profile = _detect_hw_profile(proc.device)
            infer_be = self._get_infer_backend()
            gpu_slug = _get_gpu_slug()

            # ✅ KEY 中加入 model_name
            self._cache_key = (f'{proc.model_name}_{H_pad}x{W_pad}_bs{effective_bs}'
                               f'_{infer_be}{gpu_slug}')
            
            if self.t2_cache_dir and self._hw_profile.t2_measured_ms <= 0:
                _cached = _load_t2_cache(self.t2_cache_dir).get(self._cache_key, 0.0)
                if _cached > 0:
                    self._hw_profile.t2_measured_ms = _cached
                    print(f'[T2-CACHE] 加载缓存 T2={_cached:.1f}ms '
                          f'(key={self._cache_key})', flush=True)

            _current_cfg = (proc.model_name, H_pad, W_pad, effective_bs, infer_be)   # ✅ 加入模型名，跨模型切换时正确清零 t2_measured_ms
            if self._last_calib_config != _current_cfg:
                self._hw_profile.t2_measured_ms = 0.0

            # ── 使用外部建议覆盖 ──
            if (self._pair_queue_override is not None
                    and self._result_queue_override is not None):
                _pd = self._pair_queue_override
                _rd = self._result_queue_override
                if not self.proc.quiet:
                    print(f'[AUTO-TUNE] 使用外部建议队列: pair={_pd} result={_rd}')
            else:
                _pd, _rd, _ = _auto_queue_depths(
                    self._hw_profile, self.codec, self.x264_preset, self.crf,
                    H_pad, W_pad, effective_bs, len(timesteps),
                    infer_backend=infer_be,
                    model_name=proc.model_name,   # ✅ 传入模型名
                    t3_fps_measured=self._t3_fps_measured_input,   # [FIX-T3-FPS]
                )
            self.pair_queue   = queue.Queue(maxsize=_pd)
            self.result_queue = queue.Queue(maxsize=_rd)

            if self._hw_profile.t2_measured_ms > 0:
                self._t2_estimated_ms = self._hw_profile.t2_measured_ms
            else:
                _HWB = float(H_pad * W_pad * effective_bs)
                _ifactor = _INFER_BACKEND_FACTORS.get(infer_be, 3.5)
                _mfactor = _MODEL_T2_FACTOR.get(proc.model_name, 1.0)   # ✅ 补乘模型系数，与 _auto_queue_depths 保持一致
                _t2b = (_T2_FIXED_MS + _T2_VAR_MS * _HWB / _T2_BASELINE_HWB) \
                       / max(self._hw_profile.gpu_tier, 0.05)
                self._t2_estimated_ms = max(_t2b * _ifactor * _mfactor, 1.0)

        # [PINNED-D2H] 创建 PinnedResultPool（检查 stream_d2h 是否可用）
        _pool_ok = False
        if (H_pad > 0 and W_pad > 0
                and getattr(proc, 'stream_d2h', None) is not None   # [STREAM-DUAL]
                and proc.device.type == 'cuda'):
            _max_BT    = effective_bs * len(timesteps)
            _pool_size = self.result_queue.maxsize + 2
            try:
                proc._result_pool = PinnedResultPool(_pool_size, _max_BT, H_pad, W_pad)
                _pool_ok = True
            except Exception as _pe:
                print(f'[IFRNet-Pipeline] PinnedResultPool 分配失败: {_pe}，回退同步 D2H',
                      flush=True)
                proc._result_pool = None

        print(
            f'[IFRNet-Pipeline] 启动深度流水线 | '
            f'pair_queue={self.pair_queue.maxsize} '
            f'result_queue={self.result_queue.maxsize} '
            f'effective_bs={effective_bs} '
            f'T={len(timesteps)}× | '
            f'D2H={"pinned(stream_d2h)" if _pool_ok else "sync"}',
            flush=True,
        )

        self._reader_th = threading.Thread(
            target=self._reader_loop,
            args=(reader, effective_bs, first_raw, first_padded),
            daemon=True, name='IFRNet-Reader',
        )
        self._reader_th.start()

        self._written = 0
        self._writer_th = threading.Thread(
            target=self._writer_loop,
            args=(writer, pbar, n_seg_est, meter, proc._timing),
            daemon=True, name='IFRNet-Writer',
        )
        self._writer_th.start()

        # [FIX-INFER-THREAD] 启动独立 T2 推理线程（仿 ESRGAN _sr_thread）
        self._infer_th = threading.Thread(
            target=self._infer_loop,
            args=(timesteps, H, W, effective_bs, H_pad, W_pad, meter),
            daemon=True, name='IFRNet-Infer',
        )
        self._infer_th.start()

        # 等待推理线程完成（T2 结束后向 result_queue 发送 SENTINEL，Writer 随之退出）
        self._infer_th.join()

        if self._writer_th and self._writer_th.is_alive():
            self._writer_th.join(timeout=30.0)
            if self._writer_th.is_alive():
                print('\n[IFRNet-Writer] ⚠️ 线程未在 30s 内退出', flush=True)

        if self._reader_th and self._reader_th.is_alive():
            self._reader_th.join(timeout=10.0)

        if self._prefetch_total > 0 and not self.proc.quiet:
            hit_pct = self._prefetch_hits / self._prefetch_total * 100
            print(
                f'[IFRNet-Pipeline] 预取命中率: '
                f'{self._prefetch_hits}/{self._prefetch_total} ({hit_pct:.1f}%)',
                flush=True,
            )

        return self._fc_extra, self._written

    # ── T2 推理独立线程体 ──────────────────────────────────────────────────────
    # [FIX-INFER-THREAD] 从 run() 主线程提取为独立线程，消除 Python GIL 竞争。

    def _infer_loop(
        self,
        timesteps:    list,
        H:            int,
        W:            int,
        effective_bs: int,
        H_pad:        int,
        W_pad:        int,
        meter,
    ):
        """
        [FIX-INFER-THREAD] T2 推理独立线程体。
        · pair_queue → GPU 推理（_safe_infer）→ result_queue
        · [FIX-DOUBLEBUF-H2D] 每次推理完成后立即尝试补充双槽预取，
          确保 H2D(N+1) 与 compute(N) 全重叠。
        · 最终向 result_queue 投递 SENTINEL，通知 Writer 退出。
        """
        proc           = self.proc
        fc_extra       = 0
        oc_extra       = 0
        # [FIX-RETUNE-SKIP] 跳过前 _CALIB_SKIP 个 batch（流水线刚启动时
        # pair_queue/result_queue 均为空，GPU 呈 burst 爆发态，T2 测量值异常偏低）。
        # 从第 _CALIB_SKIP+1 个 batch 起连续取 _CALIB_BATCHES 个样本做中位数校准。
        _CALIB_SKIP    = 3
        _CALIB_BATCHES = 5
        _calib_done    = False

        # [FIX-DOUBLEBUF-H2D] 入口预热双槽预取
        self._try_prefetch_next()

        try:
            while self.running:
                prefetch_result = self._pop_prefetch_or_none()
                if prefetch_result is not None:
                    item, pfimg0_t, pfimg1_t = prefetch_result
                    # [FIX-DOUBLEBUF-H2D] 弹出一个槽 → 立即补充回 2 个
                    self._try_prefetch_next()
                else:
                    pfimg0_t = pfimg1_t = None
                    self._prefetch_total += 1
                    try:
                        item = self.pair_queue.get(timeout=2.0)
                    except queue.Empty:
                        if not self._reader_th.is_alive():
                            break
                        continue

                if item is self._SENTINEL:
                    break

                img1_raw, img0_pad, img1_pad, is_end = item
                if not img1_raw:
                    continue
                B = len(img0_pad)

                results = proc._safe_infer(
                    img0_pad, img1_pad, timesteps, H, W,
                    prefetched_img0_t=pfimg0_t,
                    prefetched_img1_t=pfimg1_t,
                )

                # [AUTO-TUNE-RETUNE] [FIX-RETUNE-SKIP] 实测校准（跳过热身 batch）
                if (not _calib_done and self.auto_tune
                        and len(proc._timing) >= _CALIB_SKIP + _CALIB_BATCHES):
                    # 取跳过热身后的稳定采样窗口（索引 [skip : skip+n]）
                    _samples = proc._timing[_CALIB_SKIP : _CALIB_SKIP + _CALIB_BATCHES]
                    t2_actual = float(np.median(_samples)) * 1000.0
                    # [FIX-RETUNE-SKIP] 过滤明显异常值（< 1ms = 纯 async enqueue overhead，
                    # 不代表真实推理时间；通常发生在段初 pair/result queue 均为空时）
                    if t2_actual < 1.0:
                        # 本轮样本不可信，等下一轮（允许重试，不置 _calib_done）
                        pass
                    else:
                        _calib_done = True
                        _infer_be2 = self._get_infer_backend()
                        if self._hw_profile is not None:
                            self._hw_profile.t2_measured_ms = t2_actual
                            # [FIX-CALIB-KEY] 修复：加入 model_name，与 run() 中
                            # _current_cfg = (model_name, H_pad, W_pad, bs, be) 保持一致
                            self._last_calib_config = (
                                proc.model_name, H_pad, W_pad, effective_bs, _infer_be2
                            )
                        if self.t2_cache_dir and self._cache_key:
                            _c = _load_t2_cache(self.t2_cache_dir)
                            _old = _c.get(self._cache_key, 0.0)
                            if _old <= 0 or abs(t2_actual - _old) / max(_old, 1.0) > 0.10:
                                _c[self._cache_key] = round(t2_actual, 1)
                                _save_t2_cache(self.t2_cache_dir, _c)
                                print(f'[T2-CACHE] 已更新缓存 T2={t2_actual:.1f}ms '
                                      f'(key={self._cache_key})', flush=True)
                        _dev = abs(t2_actual - self._t2_estimated_ms) / max(self._t2_estimated_ms, 1.0)
                        _pd_cal, _rd_cal, _ = _auto_queue_depths(
                            self._hw_profile, self.codec, self.x264_preset, self.crf,
                            H_pad, W_pad, effective_bs, len(timesteps),
                            infer_backend=_infer_be2, verbose=False,
                            model_name=proc.model_name,
                            t3_fps_measured=self._t3_fps_measured_input,   # [FIX-T3-FPS]
                        )
                        # 将 RETUNE 建议写回 processor，供 process_video 跨段使用
                        proc._retune_pair_q = _pd_cal
                        proc._retune_result_q = _rd_cal
                        print(
                            f'\n[AUTO-TUNE-RETUNE] 实测 T2={t2_actual:.1f}ms | '
                            f'静态估算={self._t2_estimated_ms:.1f}ms | '
                            f'偏差={_dev*100:.0f}% | '
                            f'当前 result_queue={self.result_queue.maxsize} | '
                            f'校准建议 pair={_pd_cal} result={_rd_cal}（下次生效）',
                            flush=True,
                        )

                if isinstance(results, _PinnedResultItem):
                    results.img1_raw = img1_raw
                    out_item = results
                else:
                    out_item = (results, img1_raw, is_end)

                # 非阻塞 put + 提前提交下一批预取（背压时先做预取再阻塞）
                try:
                    self.result_queue.put_nowait(out_item)
                    self._try_prefetch_next()
                except queue.Full:
                    # [FIX-DOUBLEBUF-H2D] 队列满时先触发预取，不让 GPU 闲着
                    self._try_prefetch_next()
                    self.result_queue.put(out_item, timeout=30.0)

                fc_extra += B
                oc_extra += B * (len(timesteps) + 1)
                meter.update(B)

        except Exception as e:
            import traceback
            print(f'[IFRNet-Infer] 推理线程异常: {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
        finally:
            self.running = False
            for _ in range(10):
                try:
                    self.result_queue.put(self._SENTINEL, timeout=1.0)
                    break
                except queue.Full:
                    continue
            proc._pipeline_runner = None
            proc._result_pool     = None
            self._fc_extra = fc_extra   # [FIX-INFER-THREAD] 供 run() 读取

    def close(self):
        self.running = False


# ─────────────────────────────────────────────────────────────────────────────
# PinnedBufferPool（线程本地）
# ─────────────────────────────────────────────────────────────────────────────

_thread_local = threading.local()


class PinnedBufferPool:
    """
    线程本地 pinned-memory 缓冲池。

    [FIX-DOUBLEBUF-SLOT] _try_prefetch_next 使用 slot 0/1（prefetch_slot=0）
    和 slot 2/3（prefetch_slot=1）交替轮转，需要 4 个 slot。
    v1 仅分配 [None, None]（2槽），slot=2/3 时触发 IndexError → 线程卡死死锁。
    修复：初始化为 4 槽；get_for_frames 动态扩容，彻底防御未来 slot 越界。
    """

    def __init__(self):
        self._bufs:    list = [None, None, None, None]   # [FIX-DOUBLEBUF-SLOT] 4 槽支持 slot 0-3
        self._out_buf: Optional[torch.Tensor] = None

    def get_for_frames(self, frames: List[np.ndarray],
                       to_rgb: bool = True, slot: int = 0) -> torch.Tensor:
        # [FIX-DOUBLEBUF-SLOT] 动态扩容：slot 超界时自动增长，永不 IndexError
        while len(self._bufs) <= slot:
            self._bufs.append(None)
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
    pool  = _get_pinned_pool()
    cpu_t = pool.get_for_frames(frames, to_rgb=True, slot=slot)
    ctx   = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with ctx:
        return cpu_t.to(device, non_blocking=True, dtype=dtype)


def tensor_to_np(t, orig_H, orig_W, sync_stream=None) -> List[np.ndarray]:
    """[FIX-D2H] 异步 D2H，用于同步回退路径。"""
    if sync_stream is not None and torch.cuda.is_available():
        torch.cuda.current_stream().wait_stream(sync_stream)
    arr_gpu    = t.clamp_(0.0, 1.0).mul_(255.0).round_().byte()
    arr_perm   = arr_gpu.permute(0, 2, 3, 1).contiguous()
    pool       = _get_pinned_pool()
    out_pinned = pool.get_output_buf(arr_perm.shape, torch.uint8)
    out_pinned.copy_(arr_perm, non_blocking=True)
    device = t.device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    arr = out_pinned.numpy()
    return [arr[i, :orig_H, :orig_W, ::-1].copy() for i in range(arr.shape[0])]


# ─────────────────────────────────────────────────────────────────────────────
# TensorPool
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
# M2: FFmpeg Pipe 帧读取器
# ─────────────────────────────────────────────────────────────────────────────

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
        prefetch:        int   = 8,
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
            hw_args = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'bgr24']

        if frame_start == 0 and frame_end < 0:
            vf_args: List[str] = []
        else:
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
        pad_h, pad_w = self._pad_h, self._pad_w
        do_pad = self.need_pad
        try:
            while True:
                raw = self._proc.stdout.read(self._frame_bytes)
                if len(raw) < self._frame_bytes:
                    break
                arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                    self.height, self.width, 3).copy()
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
    if 'nb_frames' in vs and vs['nb_frames'] not in ('N/A', ''):
        nb = int(vs['nb_frames'])
    elif 'duration' in vs:
        nb = int(float(vs['duration']) * fps)
    else:
        nb = 0
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
    ):
        self._error: Optional[Exception] = None
        self._queue: queue.Queue = queue.Queue(maxsize=128)

        if preset is None:
            preset = 'p4' if 'nvenc' in codec else 'medium'

        # [FIX-LOSSLESS] crf=0 → 按编解码器映射为正确的无损参数。
        # 背景：
        #   · libx264 : crf=0 恰好等于无损，但显式用 -qp 0 语义更清晰
        #   · libx265 : crf=0 ≠ 无损！仅为极高质量有损；无损需 -x265-params lossless=1
        #   · nvenc   : -cq:v 0 是 VBR 模式下的极低码率控制，不是无损；
        #               无损需去掉 -rc:v vbr，改用 -qp 0 -b:v 0（常量 QP 模式）
        if crf == 0:
            if 'nvenc' in codec:
                # NVENC 无损：常量 QP=0，去掉 vbr 码率控制
                quality_args = ['-preset', preset, '-qp', '0', '-b:v', '0']
            elif codec == 'libx265':
                # x265 无损：必须显式 lossless=1，pools=none 避免线程池 NUMA 警告
                quality_args = ['-preset', preset, '-x265-params', 'lossless=1:pools=none']
            elif codec == 'libx264':
                # x264 无损：-qp 0 严格逐像素无损
                quality_args = ['-preset', preset, '-qp', '0']
            else:
                # 其他编解码器（如 ffv1、utvideo 等）：回退到 -qp 0
                quality_args = ['-qp', '0']
        elif 'nvenc' in codec:
            quality_args = ['-preset', preset, '-rc:v', 'vbr', '-cq:v', str(crf), '-b:v', '0']
        elif codec == 'libx265':
            quality_args = ['-preset', preset, '-crf', str(crf), '-x265-params', 'pools=none']
        else:
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
            cmd += ['-i', audio_src, '-c:a', 'copy', '-map', '0:v', '-map', '1:a?']
        if extra_codec_args:
            cmd += ['-vcodec', codec] + extra_codec_args
        else:
            cmd += ['-vcodec', codec] + quality_args
        cmd += ['-pix_fmt', 'yuv420p', '-loglevel', 'error', output_path]

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
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

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
# 核心推理类
# ─────────────────────────────────────────────────────────────────────────────

class IFRNetVideoProcessor:

    def __init__(
        self,
        model_path:       str,
        device:           str  = 'cuda',
        batch_size:       int  = 48,     # [BATCH-UP] 默认 48（原 24）
        max_batch_size:   int  = 64,
        use_fp16:         bool = True,
        use_compile:      bool = True,
        use_cuda_graph:   bool = True,
        use_tensorrt:     bool = False,
        use_hwaccel:      bool = True,
        codec:            str  = 'libx264',
        crf:              int  = 23,
        x264_preset:      str  = 'medium',
        keep_audio:       bool = True,
        ffmpeg_bin:       str  = 'ffmpeg',
        report_json:      Optional[str] = None,
        trt_cache_dir:    Optional[str] = None,
        t2_cache_dir:     Optional[str] = None,
        model_name: str = 'IFRNet_S_Vimeo90K',   # 新增
        quiet:            bool = True,
    ):
        self.model_path      = model_path
        self.device_str      = device
        self.batch_size      = batch_size
        self._max_batch_size = max(batch_size, max_batch_size)
        self._oom_cooldown   = 0
        self.use_fp16        = use_fp16 and torch.cuda.is_available()
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
        self.trt_cache_dir   = trt_cache_dir
        self.t2_cache_dir    = t2_cache_dir or _T2_CACHE_DIR_DEFAULT
        self.model_name = model_name      # 保存模型名称
        self.quiet           = quiet
        self._pipeline_runner: Optional[IFRNetPipelineRunner] = None
        self._result_pool:     Optional[PinnedResultPool]     = None
        self._pool          = TensorPool()
        self._graph:        dict = {}
        self._graph_inputs: dict = {}
        self._timing:       List[float] = []

        # 跨段自适应队列（由上一次运行的综合建议决定）
        self._next_pair_queue = None      # int or None
        self._next_result_queue = None    # int or None
        self._next_t3_fps_measured = 0.0  # [FIX-T3-FPS] 跨段实测 T3 fps（0 表示无实测）

        # [FIX-TRT-MUTEX]
        if self.use_tensorrt:
            if self.use_cuda_graph:
                self.use_cuda_graph = False
                print('  [FIX-TRT-MUTEX] use_tensorrt=True → 已禁用手动 CUDA Graph（互斥）')
            if use_compile:
                use_compile = False
                print('  [FIX-TRT-MUTEX] use_tensorrt=True → 已跳过 torch.compile（互斥）')

        self.use_compile = use_compile
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._load_model(self.device, use_compile)

        # [GPU-MONITOR] 监测器（process_video 中启动）
        self._gpu_monitor = GPUMonitor(self.device, interval=1.0, window_seconds=30.0)

        self._trt_built   = False  # 标记 TRT Engine 是否已构建

    def _load_model(self, device: torch.device, use_compile: bool = True):
        print(f'  加载模型: {self.model_path} → {device}')
        model = Model()
        ckpt  = torch.load(self.model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt)
        model = model.to(device).eval()

        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled   = True
            print('  [FIX-CU] cudnn.benchmark = True 已启用')

        if self.use_fp16:
            model = model.half()
            print('  FP16 推理已启用')

        if use_compile and hasattr(torch, 'compile'):
            try:
                torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
                cache_dir = os.path.join(
                    os.path.dirname(os.path.abspath(self.model_path)),
                    '.torch_compile_cache',
                )
                os.makedirs(cache_dir, exist_ok=True)
                os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', cache_dir)
                model = torch.compile(model, mode='default', dynamic=True)
                if self.use_tensorrt:
                    print('  torch.compile 已加载（TRT 激活时推理走 TRT 分支，compile 不执行）')
                else:
                    print(f'  torch.compile 加速已启用 (mode=default, dynamic=True)')
                    print(f'  编译缓存目录: {cache_dir}')
                    print('  首次运行将触发编译（约1-3分钟），后续运行秒启动')
                if self.use_cuda_graph:
                    self.use_cuda_graph = False
                    if not self.use_tensorrt:
                        print('  手动 CUDA Graph 已禁用（由 torch.compile 接管）')
            except Exception as e:
                print(f'  torch.compile 不可用: {e}')
                if self.use_tensorrt and self.use_cuda_graph:
                    self.use_cuda_graph = False
                    print('  [FIX-TRT-MUTEX] compile 异常 + use_tensorrt=True → 补充禁用手动 CUDA Graph')

        self.model = model

        if device.type == 'cuda':
            self.stream_compute  = torch.cuda.Stream(device=device)
            # [STREAM-DUAL] H2D 预取专用流 / D2H 输出专用流
            self.stream_h2d = torch.cuda.Stream(device=device)
            self.stream_d2h = torch.cuda.Stream(device=device)
        else:
            self.stream_compute = self.stream_h2d = self.stream_d2h = None

    # ── M4: TensorRT ─────────────────────────────────────────────────────────

    def _build_trt_engine(self, input_shape: Tuple[int, int, int, int], cache_dir: str,
                          _rebuild_attempt: bool = False):
        try:
            import tensorrt as trt
        except ImportError:
            print('[TensorRT] 未安装，跳过 TRT 加速。')
            self.use_tensorrt = False
            return

        os.makedirs(cache_dir, exist_ok=True)
        B, C, H, W = input_shape

        _sm_tag = ''
        if torch.cuda.is_available():
            _props = torch.cuda.get_device_properties(0)
            import re as _re_sm
            _gpu_slug = _re_sm.sub(r'[^a-z0-9]', '', _props.name.lower())[:16]
            _sm_tag = f'_sm{_props.major}{_props.minor}_{_gpu_slug}'

        # ✅ 加入模型变体，避免跨模型加载错误 Engine
        tag       = f'{self.model_name}_B{B}_H{H}_W{W}_fp{"16" if self.use_fp16 else "32"}{_sm_tag}'
        trt_path  = os.path.join(cache_dir, f'{tag}.trt')
        onnx_path = os.path.join(cache_dir, f'{tag}.onnx')

        if os.path.exists(trt_path):
            if _sm_tag and _sm_tag not in os.path.basename(trt_path):
                print(f'[TensorRT] 缓存文件缺少当前 GPU 标记 {_sm_tag}，删除并重建: {trt_path}')
                try: os.remove(trt_path)
                except OSError: pass
                if os.path.exists(onnx_path):
                    try: os.remove(onnx_path)
                    except OSError: pass

        if not os.path.exists(trt_path):
            print(f'[TensorRT] 构建 Engine (shape={input_shape}) ...')
            dummy0 = torch.randn(*input_shape, device=self.device)
            dummy1 = torch.randn(*input_shape, device=self.device)
            embt   = torch.full((B,), 0.5, dtype=torch.float32,
                                device=self.device).view(B, 1, 1, 1)
            if self.use_fp16:
                dummy0, dummy1, embt = dummy0.half(), dummy1.half(), embt.half()

            _base_model = getattr(self.model, '_orig_mod', self.model)

            class _InferenceWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, img0, img1, embt):
                    return self.m.inference(img0, img1, embt)

            export_model = _InferenceWrapper(_base_model)
            with torch.no_grad():
                torch.onnx.export(
                    export_model, (dummy0, dummy1, embt), onnx_path,
                    input_names=['img0', 'img1', 'embt'],
                    output_names=['output'],
                    opset_version=18,
                    dynamic_axes=None,
                )
            import onnx
            model_proto = onnx.load(onnx_path)
            onnx.save(model_proto, onnx_path,
                      save_as_external_data=False, all_tensors_to_one_file=False)
            print(f'[TensorRT] ONNX 已导出: {onnx_path}')

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

            _gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unknown'
            _sm_code  = _props.major * 10 + _props.minor if torch.cuda.is_available() else 0
            _time_hint = {
                75: '约需 20~30 分钟（T4/RTX20系 SM7.5）',
                80: '约需 10~20 分钟（A100/A30 SM8.0）',
                86: '约需 5~15 分钟（A10/RTX30系 SM8.6）',
                89: '约需 5~10 分钟（RTX40系 SM8.9）',
                90: '约需 3~8 分钟（H100 SM9.0）',
            }.get(_sm_code, f'约需 5~20 分钟（{_gpu_name}）')
            print(f'[TensorRT] {_time_hint}')

            _build_start = time.time()
            _build_done  = threading.Event()

            def _heartbeat():
                _last = time.time()
                while not _build_done.wait(timeout=5):
                    if time.time() - _last >= 300:
                        elapsed = time.time() - _build_start
                        print(f'[TensorRT] 编译中... {elapsed:.0f}s（仍在运行）', flush=True)
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

        # 加载 Engine
        try:
            if not hasattr(self, '_trt_logger'):
                self._trt_logger = trt.Logger(trt.Logger.WARNING)
            logger  = self._trt_logger
            runtime = trt.Runtime(logger)
            with open(trt_path, 'rb') as f:
                self._trt_engine = runtime.deserialize_cuda_engine(f.read())

            if self._trt_engine is None:
                if _rebuild_attempt:
                    print('[TensorRT] ⚠️  重建后 Engine 仍反序列化失败，回退 PyTorch。')
                    self.use_tensorrt = False
                    self._trt_ok = False
                    return
                print(f'[TensorRT] Engine 反序列化失败，删除并重建: {trt_path}')
                try: os.remove(trt_path)
                except OSError: pass
                if os.path.exists(onnx_path):
                    try: os.remove(onnx_path)
                    except OSError: pass
                return self._build_trt_engine(input_shape, cache_dir, _rebuild_attempt=True)

            self._trt_context = self._trt_engine.create_execution_context()
            if self._trt_context is None:
                print('[TensorRT] ⚠️  create_execution_context() 失败（显存不足），回退 PyTorch。')
                self._trt_engine  = None
                self.use_tensorrt = False
                self._trt_ok      = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return

            n = self._trt_engine.num_io_tensors
            inputs, outputs = [], []
            for i in range(n):
                name = self._trt_engine.get_tensor_name(i)
                mode = self._trt_engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    inputs.append(name)
                else:
                    outputs.append(name)
            self._trt_input_names  = inputs
            self._trt_output_names = outputs
            if not self.quiet:
                print(f'[TensorRT] inputs={inputs} outputs={outputs}')
            self._trt_ok = True
            print('[TensorRT] Engine 已激活，TRT 推理就绪。')
        except Exception as e:
            print(f'[TensorRT] Engine 加载失败: {e}，回退 PyTorch。')
            try: os.remove(trt_path)
            except OSError: pass
            self.use_tensorrt = False
            self._trt_ok = False

    # ── CUDA Graph ────────────────────────────────────────────────────────────

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

        for _ in range(5):
            with torch.cuda.stream(self.stream_compute):
                _ = self.model.inference(static_img0, static_img1, static_embt)
        torch.cuda.synchronize(self.device)

        g = torch.cuda.CUDAGraph()
        _saved_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        try:
            with torch.cuda.graph(g, stream=self.stream_compute):
                static_output = self.model.inference(static_img0, static_img1, static_embt)
        except Exception as e:
            torch.backends.cudnn.benchmark = _saved_benchmark
            try: torch.cuda.synchronize(self.device)
            except Exception: pass
            torch.cuda.empty_cache()
            self.use_cuda_graph = False
            print(f'  [CUDA Graph] 捕获失败（{type(e).__name__}: {str(e)[:120]}），'
                  f'已禁用，后续走普通推理路径。')
            with torch.cuda.stream(self.stream_compute):
                return self.model.inference(img0, img1, embt)
        finally:
            torch.backends.cudnn.benchmark = _saved_benchmark

        with torch.cuda.stream(self.stream_compute):
            g.replay()

        self._graph[shape_key] = g
        self._graph_inputs[shape_key] = {
            'img0': static_img0, 'img1': static_img1,
            'embt': static_embt, 'output': static_output,
        }
        return static_output

    # ── 核心批推理 ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer_batch(
        self,
        img0_list: List[np.ndarray],
        img1_list: List[np.ndarray],
        timesteps: List[float],
        orig_H:    int,
        orig_W:    int,
        prefetched_img0_t: Optional[torch.Tensor] = None,
        prefetched_img1_t: Optional[torch.Tensor] = None,
    ):
        B  = len(img0_list)
        T  = len(timesteps)
        t0 = time.perf_counter()

        # ── H2D：优先使用预取 tensor ─────────────────────────────────────────
        _use_prefetch = (
            prefetched_img0_t is not None and
            prefetched_img1_t is not None and
            prefetched_img0_t.shape[0] == B and
            prefetched_img1_t.shape[0] == B
        )
        if _use_prefetch:
            img0 = prefetched_img0_t
            img1 = prefetched_img1_t
            # [STREAM-DUAL] compute 只需等待 H2D（stream_h2d），不等 D2H
            if self.stream_compute is not None and self.stream_h2d is not None:
                self.stream_compute.wait_stream(self.stream_h2d)
            elif self.stream_h2d is not None:
                self.stream_h2d.synchronize()
        else:
            # [STREAM-DUAL] H2D 在 stream_h2d 上执行
            img0 = frames_to_tensor(img0_list, self.device, self.stream_h2d, self.dtype, slot=0)
            img1 = frames_to_tensor(img1_list, self.device, self.stream_h2d, self.dtype, slot=1)
            if self.stream_compute is not None:
                self.stream_compute.wait_stream(self.stream_h2d)

        img0_exp  = img0.unsqueeze(1).expand(B, T, *img0.shape[1:]).reshape(B * T, *img0.shape[1:])
        img1_exp  = img1.unsqueeze(1).expand(B, T, *img1.shape[1:]).reshape(B * T, *img1.shape[1:])
        shape_key = (B * T, 3, img0.shape[2], img0.shape[3], T)

        # ── 推理分支 ──────────────────────────────────────────────────────────
        if self.use_cuda_graph:
            with torch.cuda.stream(self.stream_compute):
                t_vals      = timesteps * B
                embt        = torch.tensor(t_vals, dtype=self.dtype,
                                           device=self.device).view(-1, 1, 1, 1)
                img0_big    = img0_exp.contiguous()
                img1_big    = img1_exp.contiguous()
                imgt_approx = img0_big * (1 - embt) + img1_big * embt
                pred_big    = self._get_cuda_graph(shape_key, img0_big, img1_big,
                                                   embt, imgt_approx)
            if self._pipeline_runner is not None:
                self._pipeline_runner._try_prefetch_next()

        elif getattr(self, '_trt_ok', False):
            import tensorrt as _trt2
            in_names  = getattr(self, '_trt_input_names',  ['img0', 'img1', 'embt'])
            out_names = getattr(self, '_trt_output_names', ['output'])
            engine_BT = self._trt_engine.get_tensor_shape(in_names[0])[0]
            BT        = img0_exp.shape[0]
            out_dtype = torch.float16 if self.use_fp16 else torch.float32
            out_shape = tuple(self._trt_engine.get_tensor_shape(out_names[0]))
            out_buf   = torch.empty(out_shape, dtype=out_dtype, device=self.device)

            _trt_stream_ctx = (torch.cuda.stream(self.stream_compute)
                               if self.stream_compute is not None else nullcontext())
            with _trt_stream_ctx:
                t_vals = timesteps * B
                embt_t = torch.tensor(t_vals, dtype=torch.float32,
                                      device=self.device).view(-1, 1, 1, 1)
                i0 = img0_exp.half().contiguous() if self.use_fp16 else img0_exp.float().contiguous()
                i1 = img1_exp.half().contiguous() if self.use_fp16 else img1_exp.float().contiguous()
                em = embt_t.half().contiguous() if self.use_fp16 else embt_t.contiguous()
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

            # [STREAM-DUAL] 不再 wait default stream；
            # stream_d2h 将在 PINNED-D2H 路径内直接 wait stream_compute。
            # 保持原始类型（FP16），float() 转换移入 stream_d2h。
            result_buf = out_buf[:BT]
            pred_big   = result_buf

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
                t_vals   = timesteps * B
                embt     = torch.tensor(t_vals, dtype=self.dtype,
                                        device=self.device).view(-1, 1, 1, 1)
                pred_big = self.model.inference(img0_exp, img1_exp, embt)
            if self._pipeline_runner is not None:
                self._pipeline_runner._try_prefetch_next()

        # ── [STREAM-DUAL] PINNED-D2H 路径 ────────────────────────────────────
        # float() + 量化 + DMA 全在 stream_d2h 上执行，彻底绕开 default stream，
        # 主线程可立即提交下一批 TRT/compute kernel，消除每批空档。
        if (self._result_pool is not None
                and self.stream_d2h is not None
                and pred_big.device.type == 'cuda'):
            BT     = pred_big.shape[0]
            pinned = self._result_pool.acquire()
            ev = None   # ✅ [EVENT-POOL] 提前初始化，确保异常路径可安全归还
            try:
                with torch.cuda.stream(self.stream_d2h):
                    # 直接等 compute stream，不经 default stream
                    self.stream_d2h.wait_stream(self.stream_compute)
                    # float() 转换也在 stream_d2h 上排队，不阻塞主线程
                    if self.use_fp16:
                        pred_f = pred_big.float()
                    else:
                        pred_f = pred_big
                    pred_u8 = (
                        pred_f.clamp_(0.0, 1.0).mul_(255.0).byte()
                        .permute(0, 2, 3, 1).contiguous()   # (BT, H, W, 3) RGB uint8
                    )
                    pinned[:BT].copy_(pred_u8, non_blocking=True)
                # [EVENT-POOL] 从池中取 Event，记录在 stream_d2h
                ev = (self._pipeline_runner._event_pool.acquire()
                      if self._pipeline_runner is not None
                      else torch.cuda.Event())
                ev.record(self.stream_d2h)
            except Exception:
                self._result_pool.release(pinned)
                if ev is not None and self._pipeline_runner is not None:   # ✅ ev 已取出则归还，防止池耗尽
                    self._pipeline_runner._event_pool.release(ev)
                raise
            self._timing.append(time.perf_counter() - t0)
            return _PinnedResultItem(
                buf=pinned, event=ev,
                B=B, T=T, orig_H=orig_H, orig_W=orig_W,
                pool=self._result_pool,
            )

        # ── 同步回退路径（CPU / pool 不可用）─────────────────────────────────
        # 此处保留显式 wait + float()，保证同步路径的正确性。
        if self.stream_compute is not None:
            torch.cuda.default_stream(self.device).wait_stream(self.stream_compute)
        if self.use_fp16:
            pred_big = pred_big.float()
        all_np = tensor_to_np(pred_big, orig_H, orig_W, sync_stream=self.stream_compute)
        result = [[all_np[i * T + j] for j in range(T)] for i in range(B)]
        self._timing.append(time.perf_counter() - t0)
        return result

    # ── OOM 自动降级 ──────────────────────────────────────────────────────────

    def _estimate_safe_batch_size(self, H: int, W: int) -> int:
        if not torch.cuda.is_available():
            return 1
        try:
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            bytes_per_frame = H * W * 3 * 2 * 6
            estimated = max(1, int(free_bytes * 0.7 / bytes_per_frame))
            return min(estimated, self._max_batch_size)
        except Exception:
            return 1

    def _safe_infer(self, img0_list, img1_list, timesteps, orig_H, orig_W,
                    prefetched_img0_t=None, prefetched_img1_t=None):
        in_oom_cascade = False
        _first_attempt = True

        while True:
            try:
                _p0 = prefetched_img0_t if _first_attempt else None
                _p1 = prefetched_img1_t if _first_attempt else None
                result = self._infer_batch(img0_list, img1_list, timesteps, orig_H, orig_W,
                                           prefetched_img0_t=_p0, prefetched_img1_t=_p1)
                in_oom_cascade = False
                if self._oom_cooldown > 0:
                    self._oom_cooldown -= 1
                elif (self.batch_size < self._max_batch_size
                      and not getattr(self, '_trt_ok', False)):
                    new_bs = min(self.batch_size + 1, self._max_batch_size)
                    print(f'[恢复] 显存充裕，batch_size {self.batch_size} → {new_bs}')
                    self.batch_size = new_bs
                return result

            except torch.cuda.OutOfMemoryError:
                _first_attempt = False
                prefetched_img0_t = prefetched_img1_t = None
                torch.cuda.empty_cache()
                self._pool.clear()
                self._graph.clear()
                self._graph_inputs.clear()

                # [STREAM-DUAL] OOM 后重建全部三条流
                if self.stream_compute is not None:
                    try: torch.cuda.synchronize(self.device)
                    except Exception: pass
                    self.stream_compute = torch.cuda.Stream(device=self.device)
                    self.stream_h2d     = torch.cuda.Stream(device=self.device)
                    self.stream_d2h     = torch.cuda.Stream(device=self.device)

                if not in_oom_cascade:
                    safe_ceiling = max(1, self.batch_size - 1)
                    if self._max_batch_size > safe_ceiling:
                        print(f'[OOM] 永久降低 max_batch_size: {self._max_batch_size} → {safe_ceiling}')
                        self._max_batch_size = safe_ceiling
                    in_oom_cascade = True

                if self.batch_size <= 1:
                    print(f'\n[OOM] batch_size=1 仍 OOM，深度清理后按剩余显存估算恢复...')
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        try: torch._dynamo.reset()
                        except Exception: pass
                        torch.cuda.empty_cache()
                    recovered_bs = self._estimate_safe_batch_size(orig_H, orig_W)
                    if recovered_bs < self._max_batch_size:
                        print(f'[OOM] 深度清理后估算安全 batch_size={recovered_bs}，'
                              f'更新 max_batch_size: {self._max_batch_size} → {recovered_bs}')
                        self._max_batch_size = recovered_bs
                    self.batch_size    = recovered_bs
                    self._oom_cooldown = 20
                    in_oom_cascade     = False
                    print(f'[OOM] 恢复 batch_size={self.batch_size}，继续处理...')
                    continue

                self.batch_size    = max(1, self.batch_size // 2)
                self._oom_cooldown = 10
                print(f'\n[OOM] 自动降低 batch_size → {self.batch_size}')

            except (RuntimeError, Exception) as _cg_err:
                _first_attempt = False
                prefetched_img0_t = prefetched_img1_t = None
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
                        try: torch.cuda.synchronize(self.device)
                        except Exception: pass
                        self.stream_compute = torch.cuda.Stream(device=self.device)
                        self.stream_h2d     = torch.cuda.Stream(device=self.device)
                        self.stream_d2h     = torch.cuda.Stream(device=self.device)
                    continue
                raise

    # ── 单段处理核心 ──────────────────────────────────────────────────────────

    def _process_segment(
        self,
        input_path:         str,
        output_path:        str,
        scale:              float,
        frame_start:        int  = 0,
        frame_end:          int  = -1,
        skip_first_output:  bool = False,
        audio_src:          Optional[str] = None,
        codec_override:     Optional[str] = None,
        extra_codec_args:   Optional[List[str]] = None,
        worker_label:       str  = '',
        preview:            bool = False,
        preview_interval:   int  = 30,
        # 跨段自适应队列建议
        pair_queue_override:   Optional[int] = None,
        result_queue_override: Optional[int] = None,
        t3_fps_measured:       float = 0.0,   # [FIX-T3-FPS] 跨段实测 T3 fps
    ) -> Tuple[bool, int, int]:
        reader = FFmpegFrameReader(
            input_path,
            frame_start  = frame_start,
            frame_end    = frame_end,
            prefetch     = self.batch_size * 3,
            use_hwaccel  = self.use_hwaccel,
            ffmpeg_bin   = self.ffmpeg_bin,
            pad_stride   = MODEL_STRIDE,
        )
        W, H      = reader.width, reader.height
        fps       = reader.fps
        n_seg_est = reader._segment_frames

        bytes_per_frame = W * H * 3 * 2 * 6
        free_bytes = 0
        if torch.cuda.is_available():
            free_bytes = torch.cuda.mem_get_info(self.device)[0]
        effective_bs = self.batch_size
        if free_bytes > 0:
            res_max_bs = max(1, int(free_bytes * 0.6 / bytes_per_frame))
            if effective_bs > res_max_bs:
                print(f'[分辨率限制] {W}×{H} 下 batch_size {effective_bs} → {res_max_bs}')
                effective_bs = res_max_bs
            if self._max_batch_size > res_max_bs:
                self._max_batch_size = max(effective_bs, res_max_bs)

        pad_h    = reader._pad_h
        pad_w    = reader._pad_w

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

        # [FIX-TSTART] 含 warmup 的端到端计时
        t_start = time.time()

        # ── torch.compile 预热 ────────────────────────────────────────────────
        if (self.use_compile
                and not getattr(self, '_warmup_done', False)
                and not getattr(self, '_trt_ok', False)):
            _WARM_H, _WARM_W = 32, 32
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
                                         dtype=self.dtype, device=self.device).view(-1, 1, 1, 1)
                    _out  = self.model.inference(_d0, _d1, _embt)
                    del _out, _d0, _d1, _embt
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                print(f'  [预热] 编译完成，耗时 {time.perf_counter()-_t_warm:.1f}s', flush=True)
            except Exception as _we:
                print(f'  [预热] 编译失败，回退至 eager 模式: {_we}', flush=True)
                if hasattr(self.model, '_orig_mod'):
                    self.model = self.model._orig_mod
                else:
                    try: torch._dynamo.reset()
                    except Exception: pass
            self._warmup_done = True

        writer = FFmpegWriter(
            output_path, W, H, new_fps,
            codec            = use_codec,
            extra_codec_args = use_extra,
            crf              = self.crf,
            preset           = self.x264_preset,
            audio_src        = audio_src,
            ffmpeg_bin       = self.ffmpeg_bin,
            quiet            = self.quiet,
        )

        frame_count  = 0
        output_count = 0
        meter        = ThroughputMeter(window=20)
        desc         = f'[{worker_label}] 插帧'
        pbar = tqdm(total=n_seg_est, unit='帧', desc=desc,
                    dynamic_ncols=True) if HAS_TQDM else None

        # ── 读取第一帧 ────────────────────────────────────────────────────────
        pair = reader.read()
        if pair is None:
            print(f'[{worker_label}] 无法读取首帧')
            reader.close(); writer.close()
            if pbar: pbar.close()
            return False, 0, 0
        first, first_padded = pair

        if not skip_first_output:
            writer.write(first)
            output_count += 1

        frame_count = 1
        if pbar:
            pbar.update(1)

        # ── 主处理 ────────────────────────────────────────────────────────────
        if self.device.type == 'cuda':
            pipeline = IFRNetPipelineRunner(
                self,
                auto_tune    = True,
                codec        = use_codec,
                x264_preset  = self.x264_preset,
                crf          = self.crf,
                t2_cache_dir = self.t2_cache_dir,
                pair_queue_override   = pair_queue_override,
                result_queue_override = result_queue_override,
                t3_fps_measured       = t3_fps_measured,   # [FIX-T3-FPS]
            )
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
                H_pad             = H + pad_h,
                W_pad             = W + pad_w,
            )
            # [GPU-MONITOR-v2] 保存实际队列深度，供 print_report() 调优建议使用
            self._last_pair_q_size   = pipeline.pair_queue.maxsize
            self._last_result_q_size = pipeline.result_queue.maxsize
            # [FIX-T3-MEMCAP] 记录每个 result slot 的 MiB，供 get_queue_suggestions 约束
            _max_BT = effective_bs * len(timesteps)
            self._last_pool_slot_mb = (
                _max_BT * (H + pad_h) * (W + pad_w) * 3 / 1e6
            )
            # [FIX-T3-FPS] 保存实测 T3 fps + 编码分辨率，供 process_video 报告使用
            self._last_t3_fps_measured = getattr(pipeline, '_t3_fps_measured', 0.0)
            self._last_encode_hw = (H, W)
            frame_count  += fc_extra
            output_count += oc_extra
        else:
            # 同步回退路径
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

        # ── 收尾 ──────────────────────────────────────────────────────────────
        if pbar:
            pbar.close()
        writer.close()
        reader.close()

        elapsed = time.time() - t_start
        print(f'[{worker_label}] 完成 | 原始帧={frame_count} → 输出帧={output_count} | '
              f'{frame_count/elapsed:.1f} 原始帧/s（含 warmup/初始化）')
        return True, frame_count, output_count

    # ── 对外公开接口 ──────────────────────────────────────────────────────────

    def process_video(
        self,
        input_path:       str,
        output_path:      str,
        scale:            float = 2.0,
        preview:          bool  = False,
        preview_interval: int   = 30,
    ) -> bool:
        if not os.path.exists(input_path):
            print(f'错误: 输入不存在 - {input_path}')
            return False
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        audio_src = input_path if self.keep_audio else None

        # if self.use_tensorrt:
        if self.use_tensorrt and not self._trt_built:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            meta = _probe_video(input_path)
            _trt_ceil = lambda x, s: x if x % s == 0 else x + (s - x % s)
            _trt_H    = _trt_ceil(meta['height'], MODEL_STRIDE)
            _trt_W    = _trt_ceil(meta['width'],  MODEL_STRIDE)
            sh        = (self.batch_size, 3, _trt_H, _trt_W)
            trt_dir   = self.trt_cache_dir or os.path.join(base_dir, '.trt_cache')
            self._build_trt_engine(sh, trt_dir)
            # 无论成功或失败，标记已尝试，避免重复构建
            self._trt_built = True

        # [GPU-MONITOR] 启动后台监测
        self._gpu_monitor.start()

        ok, fc, oc = self._process_segment(
            input_path, output_path, scale,
            frame_start=0, frame_end=-1,
            skip_first_output=False,
            audio_src=audio_src,
            worker_label='GPU0',
            preview=preview,
            preview_interval=preview_interval,
            # 传入跨段自适应建议
            pair_queue_override=self._next_pair_queue,
            result_queue_override=self._next_result_queue,
            t3_fps_measured=self._next_t3_fps_measured,   # [FIX-T3-FPS]
        )

        # [GPU-MONITOR-v2] 停止采样，输出精细统计 + 三项调优建议
        self._gpu_monitor.stop()
        _gpu_stats = self._gpu_monitor.get_stats()
        if _gpu_stats.sample_count > 0:
            _cur_pair_q   = getattr(self, '_last_pair_q_size',   4)
            _cur_result_q = getattr(self, '_last_result_q_size', 16)
            print()
            self._gpu_monitor.print_report(
                _gpu_stats,
                current_bs       = self.batch_size,
                current_pair_q   = _cur_pair_q,
                current_result_q = _cur_result_q,
            )
            # [FIX-T3-DETECT] 获取 GPU-MONITOR 的队列建议（含 T3-bottleneck 检测）
            _slot_mb = getattr(self, '_last_pool_slot_mb', 0.0)
            pair_gpu_sug, result_gpu_sug = self._gpu_monitor.get_queue_suggestions(
                _gpu_stats, _cur_pair_q, _cur_result_q,
                slot_mb=_slot_mb,   # 传入每 slot 大小，用于 PinnedPool 内存约束
            )
            # 获取 AUTO-TUNE-RETUNE 的建议（如果存在）
            retune_pair_q   = getattr(self, '_retune_pair_q',   None)
            retune_result_q = getattr(self, '_retune_result_q', None)

            # [FIX-T3-DETECT] 先检测是否 T3-bottleneck，再决定综合策略
            _is_t3 = GPUMonitor._is_t3_bottleneck(_gpu_stats)
            if _is_t3:
                # T3 是真正瓶颈：不增大队列，result_queue 可适当缩小以回收 pinned 内存
                final_pair_q   = _cur_pair_q
                final_result_q = max(16, _cur_result_q - 8) if _cur_result_q > 16 else _cur_result_q
                print(
                    f'[ADAPTIVE-QUEUE] ⚠️  T3-bottleneck 确认（编码器是瓶颈）：'
                    f'pair_queue={final_pair_q}（不变）'
                    f' result_queue={_cur_result_q}->{final_result_q}（适当缩小，回收锁页内存）'
                )
                # [FIX-T3-REPORT] 增强诊断：实测 vs 理论 T3 fps + 具体编码建议
                _t3_fps_meas = getattr(self, '_last_t3_fps_measured', 0.0)
                _H_enc, _W_enc = getattr(self, '_last_encode_hw', (0, 0))
                _t3_fps_est = 0.0
                if _H_enc > 0 and _W_enc > 0:
                    _t3_fps_est = _software_encode_fps(
                        os.cpu_count() or 4, _H_enc, _W_enc,
                        self.codec, self.x264_preset, self.crf,
                    )
                _diag_parts = []
                if _t3_fps_meas > 0:
                    _diag_parts.append(f'实测 T3={_t3_fps_meas:.0f} fps')
                if _t3_fps_est > 0:
                    _diag_parts.append(f'理论估算={_t3_fps_est:.0f} fps')
                if _t3_fps_meas > 0 and _t3_fps_est > 0:
                    _degrade = _t3_fps_est / max(_t3_fps_meas, 1.0)
                    _diag_parts.append(f'偏差={_degrade:.1f}×（含热节流因素）')
                _diag_str = '  [' + '  '.join(_diag_parts) + ']' if _diag_parts else ''
                _has_nvenc_h264 = HardwareCapability.has_nvenc('h264_nvenc')
                if _has_nvenc_h264 and _t3_fps_meas > 0:
                    _nvenc_fps = 3000.0
                    if _H_enc > 0 and _W_enc > 0:
                        _nvenc_fps = min(3000.0, 3000.0 * 1920 * 1080 / (_H_enc * _W_enc))
                    _speedup = _nvenc_fps / max(_t3_fps_meas, 1.0)
                    _encoder_tip = (
                        f'建议切换 --codec h264_nvenc（理论 ~{_nvenc_fps:.0f} fps，'
                        f'约 {_speedup:.0f}× 加速）'
                        f'；注: Docker 环境需确认 NVENC 设备映射（--gpus）'
                    )
                elif _has_nvenc_h264:
                    _encoder_tip = (
                        '建议切换 --codec h264_nvenc（NVENC 约 10-20× 加速）'
                    )
                else:
                    _encoder_tip = (
                        '考虑降低编码参数：--x264-preset veryfast --crf 18'
                        '（实测约 5-10× 加速，画质略降但通常可接受）'
                    )
                print(f'[ADAPTIVE-QUEUE] 提示：真正瓶颈在编码器{_diag_str}  {_encoder_tip}')
                # [FIX-T3-FPS] 保存实测 T3 fps 供下段使用
                self._next_t3_fps_measured = _t3_fps_meas
            else:
                # 正常路径：综合 GPU-MONITOR 和 RETUNE 两方建议，取整数均值
                final_pair_q = max(
                    pair_gpu_sug,
                    retune_pair_q if retune_pair_q is not None else 0,
                    _cur_pair_q,    # 保持不低于当前
                )
                final_result_q = max(
                    ((result_gpu_sug + retune_result_q if retune_result_q is not None else 0) // 2),
                    _cur_result_q,
                )
                # 硬上限
                final_pair_q   = min(final_pair_q, 8)
                final_result_q = min(final_result_q, 64)
                # [FIX-T3-MEMCAP] 再次施加内存上限（防御性）
                if _slot_mb > 0.0:
                    _max_rq_mem = max(8, int(_PINNED_POOL_MAX_MB / _slot_mb) - 2)
                    final_result_q = min(final_result_q, _max_rq_mem)

                if final_pair_q != _cur_pair_q or final_result_q != _cur_result_q:
                    print(f'[ADAPTIVE-QUEUE] 下次将使用 pair_queue={final_pair_q} '
                          f'result_queue={final_result_q}（综合建议）')
                # [FIX-T3-FPS] 非 T3-bottleneck 时也更新实测 T3 fps（更可靠）
                self._next_t3_fps_measured = getattr(self, '_last_t3_fps_measured', 0.0)

            self._next_pair_queue   = final_pair_q
            self._next_result_queue = final_result_q
        else:
            print('[GPU-MONITOR] 警告：未能获取任何 GPU 采样数据，'
                  '请检查 nvidia-ml-py 安装或驱动状态。')

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
        description='IFRNet 视频插帧 —— 终极优化版 v6.3.2（单卡版）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 基础参数
    parser.add_argument('--input',      required=True,  help='输入视频路径')
    parser.add_argument('--output',     required=True,  help='输出视频路径')
    parser.add_argument('--scale',      type=float, default=2.0, help='插帧倍数（≥2 整数）')
    parser.add_argument('--model',      default='IFRNet_S_Vimeo90K', help='模型名称或 .pth 路径')
    parser.add_argument('--device',     default='cuda', choices=['cuda', 'cpu'])
    # [BATCH-UP] 默认 48
    parser.add_argument('--batch-size', type=int, default=48,
                        help='每批帧对数（默认 48，TRT 用户首次运行需重建 Engine）')
    # 推理优化
    parser.add_argument('--no-fp16',       action='store_true', help='禁用 FP16')
    parser.add_argument('--no-compile',    action='store_true', help='禁用 torch.compile')
    parser.add_argument('--no-cuda-graph', action='store_true', help='禁用 CUDA Graph')
    parser.add_argument('--use-tensorrt',  action='store_true',
                        help='启用 TensorRT 加速（首次需构建 Engine）')
    # 高优先级覆盖参数
    parser.add_argument('--use-cuda-graph', dest='use_cuda_graph_force',
                        action='store_true', default=False,
                        help='[覆盖] 强制启用 CUDA Graph，覆盖 --no-cuda-graph')
    parser.add_argument('--use-compile', dest='use_compile_force',
                        action='store_true', default=False,
                        help='[覆盖] 强制启用 torch.compile，覆盖 --no-compile')
    parser.add_argument('--no-tensorrt', dest='no_tensorrt',
                        action='store_true', default=False,
                        help='[覆盖] 强制禁用 TensorRT，覆盖 --use-tensorrt')
    # 硬件加速
    parser.add_argument('--no-hwaccel', action='store_true', help='强制禁用 NVDEC')
    # 编码参数
    parser.add_argument('--codec',       default='libx264')
    parser.add_argument('--crf',         type=int, default=23)
    parser.add_argument('--x264-preset', type=str, default='medium',
                        choices=['ultrafast','superfast','veryfast','faster','fast',
                                 'medium','slow','slower','veryslow'])
    parser.add_argument('--no-audio',    action='store_true')
    parser.add_argument('--ffmpeg-bin',  type=str, default='ffmpeg')
    # 调试
    parser.add_argument('--preview',           action='store_true')
    parser.add_argument('--preview-interval',  type=int, default=30)
    parser.add_argument('--report',            default=None, help='JSON 性能报告路径')
    parser.add_argument('--quiet', action=argparse.BooleanOptionalAction, default=True,
                        help='静默模式（默认开启），仅显示关键信息；--no-quiet 开启详细日志')
    parser.add_argument('--trt-cache-dir',   default=None)
    parser.add_argument('--t2-cache-dir',    default=None)

    args = parser.parse_args()

    # ── 高优先级覆盖参数解析 ──────────────────────────────────────────────────
    _cli_overrides: list = []

    if args.no_tensorrt and args.use_tensorrt:
        args.use_tensorrt = False
        _cli_overrides.append('--no-tensorrt  覆盖了  --use-tensorrt  → TensorRT 已禁用')

    if args.use_compile_force and args.no_compile:
        args.no_compile = False
        _cli_overrides.append('--use-compile  覆盖了  --no-compile  → torch.compile 已启用')

    if args.use_cuda_graph_force and args.no_cuda_graph:
        args.no_cuda_graph = False
        _cli_overrides.append('--use-cuda-graph  覆盖了  --no-cuda-graph  → CUDA Graph 已启用')

    _effective_trt     = args.use_tensorrt
    _effective_compile = not args.no_compile
    _effective_cugraph = not args.no_cuda_graph

    if args.use_cuda_graph_force and _effective_compile and not _effective_trt:
        print('[CLI警告] --use-cuda-graph 与 torch.compile 互斥：compile 成功后 CUDA Graph 将被自动禁用。')
    if args.use_cuda_graph_force and _effective_trt:
        print('[CLI警告] --use-cuda-graph 与 --use-tensorrt 互斥：TensorRT 优先。')
    if args.use_compile_force and _effective_trt:
        print('[CLI警告] --use-compile 与 --use-tensorrt 互斥：TensorRT 优先。')

    if _cli_overrides:
        print('[CLI覆盖] 以下设置已被高优先级参数覆盖：')
        for msg in _cli_overrides:
            print(f'          · {msg}')
        print()

    # 模型路径解析
    if args.model in MODEL_NAME_MAP:
        model_path = f'{models_ifrnet}/{MODEL_NAME_MAP[args.model]}'
        model_name = args.model
    else:
        model_path = args.model
        model_name = os.path.splitext(os.path.basename(args.model))[0]   # ✅ 自定义路径取 basename，防止斜杠污染 TRT/T2 缓存文件名
    if not os.path.exists(model_path):
        print(f'错误: 模型不存在 - {model_path}')
        sys.exit(1)

    global Model
    Model, _ = _load_ifrnet_module(args.model)

    print('=' * 65)
    print('  IFRNet 视频插帧 —— 终极优化版 v6.3.2（单卡版）')
    print('=' * 65)
    print(f'  模型:   {args.model}')
    print(f'  设备:   {args.device} | GPU: '
          f'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print(f'  FP16:   {not args.no_fp16} | '
          f'Compile: {not args.no_compile} | '
          f'CUDA Graph: {not args.no_cuda_graph} | '
          f'TensorRT: {args.use_tensorrt}')
    print(f'  NVDEC:  {HardwareCapability.has_nvdec() and not args.no_hwaccel} | '
          f'NVENC(h264): {HardwareCapability.has_nvenc("h264_nvenc")} | '
          f'NVENC(hevc): {HardwareCapability.has_nvenc("hevc_nvenc")}')
    _codec_actual = HardwareCapability.best_encoder(args.codec)
    if args.crf == 0:
        # [FIX-LOSSLESS] 提示用户实际使用的无损参数
        if 'nvenc' in _codec_actual:
            _lossless_note = '(-qp 0 无损，常量 QP 模式)'
        elif args.codec == 'libx265':
            _lossless_note = '(-x265-params lossless=1，注意：crf=0 在 x265 中不是无损！)'
        else:
            _lossless_note = '(-qp 0 严格逐像素无损)'
        print(f'  编码器: {args.codec} → 实际: {_codec_actual} | '
              f'CRF: 0 → 无损模式 {_lossless_note} | batch_size: {args.batch_size}')
    else:
        print(f'  编码器: {args.codec} → 实际: {_codec_actual} | '
              f'CRF: {args.crf} | batch_size: {args.batch_size}')
    if args.use_tensorrt:
        _tcd = args.trt_cache_dir or f'(自动: {base_dir}/.trt_cache)'
        print(f'  TRT 缓存: {_tcd}')
    print()

    t_total   = time.time()
    processor = IFRNetVideoProcessor(
        model_path     = model_path,
        device         = args.device,
        batch_size     = args.batch_size,
        max_batch_size = args.batch_size * 4,
        use_fp16       = not args.no_fp16,
        use_compile    = not args.no_compile,
        use_cuda_graph = not args.no_cuda_graph,
        use_tensorrt   = args.use_tensorrt,
        use_hwaccel    = not args.no_hwaccel,
        codec          = args.codec,
        crf            = args.crf,
        x264_preset    = args.x264_preset,
        keep_audio     = not args.no_audio,
        ffmpeg_bin     = args.ffmpeg_bin,
        report_json    = args.report,
        trt_cache_dir  = args.trt_cache_dir,
        t2_cache_dir   = getattr(args, 't2_cache_dir', None),
        model_name     = model_name,   # ✅ 使用规范化后的 model_name（自定义路径时为 basename）
        quiet          = getattr(args, 'quiet', True),
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