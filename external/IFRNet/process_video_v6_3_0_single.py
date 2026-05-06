"""
IFRNet 视频插帧处理脚本 —— 终极优化版 v6.3.0（单卡版）
==========================================================
基于 IFRNet（Intermediate Flow-based Recursive Network）的视频帧插值脚本，
面向单 GPU 生产环境的高性能实现。

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

  [GPU-MONITOR]    后台 GPU 监测线程（2 秒采样），运行结束后打印：
                   · 平均 / 最高 GPU 利用率（%）
                   · 平均 / 峰值显存占用（GiB）
                   · 基于利用率和显存余量的 batch_size 推荐。

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
# [GPU-MONITOR] 后台 GPU 监测线程
# ─────────────────────────────────────────────────────────────────────────────

class GPUMonitor:
    """
    后台线程定期采样 GPU 利用率和显存占用，运行结束后打印统计及 batch_size 推荐。

    · 采样间隔默认 1 秒，不显著增加 CPU 开销。
    · 采样间隔默认 10 秒
    · 优先使用 nvidia-ml-py（import pynvml），不可用时回退至 torch.cuda API。
    · 若完全无法获取 GPU 信息，采样值固定为 0，不影响推理流程。
    · 所有采样、NVML 初始化与清理均在子线程内完成，保证线程安全。

    修复记录：
    · FIX-NVML-LEAK : nvmlInit 成功但 GetHandleByIndex 失败时，用 nvml_initialized 标志
                      确保 nvmlShutdown 始终被调用，避免 NVML 句柄泄漏。
    · FIX-NVML-LOCK : _printed_nvml_help 改为 threading.Event，消除主线程与采样线程
                      之间对该标志的无锁竞态条件。
    · FIX-THREAD-RST: stop() 结束后将 _thread 重置为 None，防止重复 start/stop 时
                      旧线程引用残留导致逻辑错误。
    · FIX-SNAP      : summary() 先对 util_samples / mem_samples 做列表快照，避免
                      np.mean 遍历期间采样线程并发 append 引发数据竞争。
    · FIX-DEV-IDX   : NVML 及 torch.cuda API 均改用 self.device.index，正确支持
                      cuda:1、cuda:2 等非零设备，不再硬编码 GPU 0。
    """

    def __init__(self, device: torch.device, interval: float = 1.0, window_seconds: float = 10.0):
        self.device         = device
        self.interval       = interval
        self.window_seconds = window_seconds

        self._stop_event    = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # FIX-NVML-LOCK: 用 Event 替代普通 bool，线程安全地标记"已打印提示"
        self._nvml_help_printed = threading.Event()

        self.util_samples: List[float] = []
        self.mem_samples:  List[float] = []

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def start(self):
        """启动后台采样（仅对 CUDA 设备有效）。"""
        if self.device.type != "cuda":
            return
        # FIX-THREAD-RST: 每次 start 前清理旧状态，支持 stop→start 复用
        self._stop_event.clear()
        self.util_samples.clear()
        self.mem_samples.clear()
        self._thread = threading.Thread(
            target=self._sample, daemon=True, name="GPUMonitor"
        )
        self._thread.start()

    def stop(self):
        """通知采样线程停止并等待结束（最长 5 秒）。"""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5.0)
            self._thread = None  # FIX-THREAD-RST: 清空引用，避免残留

    # ------------------------------------------------------------------
    # 采样线程
    # ------------------------------------------------------------------

    def _sample(self):
        """
        采样线程主循环：
        1. 解析目标 GPU 索引（FIX-DEV-IDX）
        2. 优先初始化 NVML；失败时用 nvml_initialized 标志管理资源释放（FIX-NVML-LEAK）
        3. NVML 不可用时回退至 torch.cuda API，并用 Event 保证提示仅打印一次（FIX-NVML-LOCK）
        4. 循环采集 GPU 利用率与显存占用
        5. 退出前根据 nvml_initialized 决定是否调用 nvmlShutdown（FIX-NVML-LEAK）
        """
        # FIX-DEV-IDX: 取实际设备索引，兼容 cuda:0 / cuda:1 / cuda:N
        dev_idx = self.device.index if self.device.index is not None else 0

        pynvml_module   = None
        pynvml_handle   = None
        nvml_initialized = False  # FIX-NVML-LEAK: 独立跟踪 nvmlInit 是否成功

        # ----- 优先尝试 nvidia-ml-py -----
        try:
            import pynvml
            pynvml_module = pynvml
        except ImportError:
            pass

        if pynvml_module is not None:
            try:
                pynvml_module.nvmlInit()
                nvml_initialized = True  # FIX-NVML-LEAK: 标记初始化成功
                # FIX-DEV-IDX: 使用正确的设备索引
                pynvml_handle = pynvml_module.nvmlDeviceGetHandleByIndex(dev_idx)
            except Exception:
                pynvml_handle = None
                # nvml_initialized 保持其当前值：
                # 若 nvmlInit 已成功但 GetHandleByIndex 失败，仍需在退出时 Shutdown

        # ----- 回退到 torch.cuda API -----
        if pynvml_handle is None and torch.cuda.is_available():
            # FIX-NVML-LOCK: 用 Event.set() 替代 bool 赋值，线程安全
            if not self._nvml_help_printed.is_set():
                self._nvml_help_printed.set()
                print(
                    "[GPU-MONITOR] 未检测到 nvidia-ml-py，已回退至 torch.cuda 内置 API。\n"
                    "  推荐安装 nvidia-ml-py 以获得更精确的监控数据：\n"
                    "  pip install nvidia-ml-py==13.580.65   # 匹配驱动 580.65\n"
                    "  (若驱动不同，请根据 nvidia-smi 输出选择主版本号一致的 nvidia-ml-py 版本)",
                    flush=True,
                )

        # ----- 采样循环 -----
        while not self._stop_event.is_set():
            util        = 0.0
            mem_used_gib = 0.0
            try:
                if pynvml_handle is not None:
                    util     = float(pynvml_module.nvmlDeviceGetUtilizationRates(pynvml_handle).gpu)
                    mem_info = pynvml_module.nvmlDeviceGetMemoryInfo(pynvml_handle)
                    mem_used_gib = mem_info.used / (1024 ** 3)
                elif torch.cuda.is_available():
                    # FIX-DEV-IDX: 传入正确的设备索引
                    try:
                        util = float(torch.cuda.utilization(dev_idx))
                    except Exception:
                        util = 0.0
                    free, total = torch.cuda.mem_get_info(dev_idx)
                    mem_used_gib = (total - free) / (1024 ** 3)
                # 两者均不可用：保持 0.0
            except Exception:
                pass

            self.util_samples.append(util)
            self.mem_samples.append(mem_used_gib)
            self._stop_event.wait(self.interval)

        # ----- 释放 NVML（FIX-NVML-LEAK）-----
        # 只要 nvmlInit 成功过，无论 handle 是否有效，均需 Shutdown
        if nvml_initialized and pynvml_module is not None:
            try:
                pynvml_module.nvmlShutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 统计输出
    # ------------------------------------------------------------------

    def summary(self) -> Tuple[float, float, float, float, float]:
        """
        返回统计值：
            (平均利用率%, 最高利用率%, 平均显存GiB, 峰值显存GiB, 峰值利用率均值%)

        峰值利用率均值：将采样点按窗口（每10个样本）划分，取每窗口最大值再求平均；
        样本不足一个窗口时以全局最大值代替。

        FIX-SNAP: 先对共享列表做快照，避免 np 遍历期间采样线程并发 append。
        """
        # FIX-SNAP: 快照隔离，消除与采样线程的数据竞争
        util_snap = list(self.util_samples)
        mem_snap  = list(self.mem_samples)

        if not util_snap:
            print("[GPU-MONITOR] 警告：未能获取任何 GPU 利用数据，请检查 nvidia-ml-py 安装或驱动状态。")
            return 0.0, 0.0, 0.0, 0.0, 0.0

        avg_util = round(float(np.mean(util_snap)), 1)
        max_util = round(float(np.max(util_snap)), 1)
        avg_mem  = round(float(np.mean(mem_snap)),  2)
        max_mem  = round(float(np.max(mem_snap)),   2)

        # 峰值利用率均值：窗口覆盖固定时长（window_seconds 默认10秒），自动适配不同 interval（默认1秒）
        window_size = max(2, round(self.window_seconds / self.interval))
        util_arr    = np.array(util_snap)
        if len(util_arr) >= window_size:
            num_windows = len(util_arr) // window_size
            peaks   = [np.max(util_arr[i * window_size:(i + 1) * window_size])
                       for i in range(num_windows)]
            peak_avg = round(float(np.mean(peaks)), 1)
        else:
            peak_avg = max_util  # 样本太少，直接用全局最大值

        return avg_util, max_util, avg_mem, max_mem, peak_avg


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
    else:
        fps_est = _software_encode_fps(profile.cpu_cores, H_pad, W_pad, codec, x264_preset, crf)
        t3_ms   = out_frames / fps_est * 1000.0
        t3_src  = f'{codec}({x264_preset}, crf={crf})'

    pair_depth   = max(2, min(int(_math.ceil(t2_ms / max(t1_ms, 0.1))) + 2, 8))
    result_depth = max(8, min(int(_math.ceil(t3_ms / max(t2_ms, 0.1))) + 3, 64))
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

        self._prefetch_item    = None
        self._prefetch_img0_t: Optional[torch.Tensor] = None
        self._prefetch_img1_t: Optional[torch.Tensor] = None
        self._prefetch_hits  = 0
        self._prefetch_total = 0
        self._reader_th: Optional[threading.Thread] = None
        self._writer_th: Optional[threading.Thread] = None

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
            print(f'[IFRNet-Reader] 异常 @frame={frames_read}: {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
        finally:
            if not self.proc.quiet:
                print(f'[IFRNet-Reader] 退出，已读 {frames_read} 帧', flush=True)
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
        在 stream_h2d 上异步 H2D 上传下一批输入。
        [STREAM-DUAL] 与 stream_d2h（D2H 输出）完全独立，PCIe 全双工利用。
        """
        if self._prefetch_item is not None:
            return
        if self.pair_queue.empty():
            return
        try:
            item = self.pair_queue.get_nowait()
        except queue.Empty:
            return
        if item is self._SENTINEL:
            try:
                self.pair_queue.put(item)
            except Exception:
                pass
            return
        img1_raw, img0_pad, img1_pad, is_end = item
        if not img0_pad:
            try:
                self.pair_queue.put(item)
            except Exception:
                pass
            return
        proc      = self.proc
        pool      = _get_pinned_pool()
        # [STREAM-DUAL] 预取专用 stream_h2d
        stream_h2d = proc.stream_h2d
        device    = proc.device
        dtype     = proc.dtype
        try:
            if stream_h2d is not None:
                with torch.cuda.stream(stream_h2d):
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

    def _pop_prefetch_or_none(self):
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

    def _writer_loop(self, writer, pbar, n_seg_est, meter, timing_ref):
        written          = 0
        _idle_since      = None
        received_sentinel = False
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
                            print(f'[IFRNet-Writer][看门狗] 流水线空转，'
                                  f'开始计时（阈值 {self.IDLE_DEADLOCK_TIMEOUT:.0f}s）', flush=True)
                        elif time.time() - _idle_since > self.IDLE_DEADLOCK_TIMEOUT:
                            print(f'[IFRNet-Writer][看门狗] ⚠️ 空转超过 '
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
                print(f'[IFRNet-Writer] 退出，已写 {written} 输出帧', flush=True)
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

            _pd, _rd, _ = _auto_queue_depths(
                self._hw_profile, self.codec, self.x264_preset, self.crf,
                H_pad, W_pad, effective_bs, len(timesteps),
                infer_backend=infer_be,
                model_name=proc.model_name,   # ✅ 传入模型名
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

        fc_extra = 0
        oc_extra = 0
        _CALIB_BATCHES = 5
        _calib_done    = False

        try:
            while self.running:
                prefetch_result = self._pop_prefetch_or_none()
                if prefetch_result is not None:
                    item, pfimg0_t, pfimg1_t = prefetch_result
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

                # [AUTO-TUNE-RETUNE] 首批实测校准
                if (not _calib_done and self.auto_tune
                        and len(proc._timing) >= _CALIB_BATCHES):
                    _calib_done = True
                    t2_actual = float(np.median(proc._timing[-_CALIB_BATCHES:])) * 1000.0
                    _infer_be2 = self._get_infer_backend()
                    if self._hw_profile is not None:
                        self._hw_profile.t2_measured_ms = t2_actual
                        self._last_calib_config = (H_pad, W_pad, effective_bs, _infer_be2)
                    if self.t2_cache_dir and self._cache_key:
                        _c = _load_t2_cache(self.t2_cache_dir)
                        _old = _c.get(self._cache_key, 0.0)
                        if _old <= 0 or abs(t2_actual - _old) / max(_old, 1.0) > 0.10:
                            _c[self._cache_key] = round(t2_actual, 1)
                            _save_t2_cache(self.t2_cache_dir, _c)
                            print(f'[T2-CACHE] 已更新缓存 T2={t2_actual:.1f}ms '
                                  f'(key={self._cache_key})', flush=True)
                    _dev = abs(t2_actual - self._t2_estimated_ms) / max(self._t2_estimated_ms, 1.0)
                    _, _rd_cal, _ = _auto_queue_depths(
                        self._hw_profile, self.codec, self.x264_preset, self.crf,
                        H_pad, W_pad, effective_bs, len(timesteps),
                        infer_backend=_infer_be2, verbose=False,
                        model_name=proc.model_name,   # ✅ 传入模型名
                    )
                    print(
                        f'[AUTO-TUNE-RETUNE] 实测 T2={t2_actual:.1f}ms | '
                        f'静态估算={self._t2_estimated_ms:.1f}ms | '
                        f'偏差={_dev*100:.0f}% | '
                        f'当前 result_queue={self.result_queue.maxsize} | '
                        f'校准建议={_rd_cal}（下次生效）',
                        flush=True,
                    )

                if isinstance(results, _PinnedResultItem):
                    results.img1_raw = img1_raw
                    out_item = results
                else:
                    out_item = (results, img1_raw, is_end)

                # FIX 非阻塞式 result_queue.put + 提前提交下一批预取
                try:
                    # 尝试 put_nowait() 放入结果
                    self.result_queue.put_nowait(out_item)
                except queue.Full:
                    # 队列满时，先启动下一批的 GPU 预取，不让 GPU 闲着
                    self._try_prefetch_next()
                    # 然后阻塞等待
                    self.result_queue.put(out_item, timeout=30.0)

                fc_extra += B
                oc_extra += B * (len(timesteps) + 1)
                meter.update(B)

        except Exception as e:
            import traceback
            print(f'[IFRNet-Infer] 推理主循环异常: {type(e).__name__}: {e}', flush=True)
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

        if self._writer_th and self._writer_th.is_alive():
            self._writer_th.join(timeout=30.0)
            if self._writer_th.is_alive():
                print('[IFRNet-Writer] ⚠️ 线程未在 30s 内退出', flush=True)

        if self._reader_th and self._reader_th.is_alive():
            self._reader_th.join(timeout=10.0)

        if self._prefetch_total > 0 and not self.proc.quiet:
            hit_pct = self._prefetch_hits / self._prefetch_total * 100
            print(
                f'[IFRNet-Pipeline] 预取命中率: '
                f'{self._prefetch_hits}/{self._prefetch_total} ({hit_pct:.1f}%)',
                flush=True,
            )

        oc_extra = self._written
        return fc_extra, oc_extra

    def close(self):
        self.running = False


# ─────────────────────────────────────────────────────────────────────────────
# PinnedBufferPool（线程本地）
# ─────────────────────────────────────────────────────────────────────────────

_thread_local = threading.local()


class PinnedBufferPool:
    def __init__(self):
        self._bufs:    list = [None, None]
        self._out_buf: Optional[torch.Tensor] = None

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
    ):
        self._error: Optional[Exception] = None
        self._queue: queue.Queue = queue.Queue(maxsize=128)

        if preset is None:
            preset = 'p4' if 'nvenc' in codec else 'medium'

        if 'nvenc' in codec:
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
        self._gpu_monitor = GPUMonitor(self.device, interval=1.0)

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
        )

        # [GPU-MONITOR] 停止采样并打印统计
        self._gpu_monitor.stop()
        avg_u, max_u, avg_mem, max_mem, peak_avg_u = self._gpu_monitor.summary()
        if avg_u > 0 or avg_mem > 0:
            print(f'\n[GPU-MONITOR] 平均利用率: {avg_u}%  | 最高利用率: {max_u}%  | 峰值利用率均值: {peak_avg_u}%')
            print(f'[GPU-MONITOR] 平均显存: {avg_mem:.2f} GiB | 峰值显存: {max_mem:.2f} GiB')
            # batch_size 推荐
            if self.device.type == 'cuda':
                total_vram_gib = (torch.cuda.get_device_properties(self.device).total_memory
                                  / (1024 ** 3))
                mem_ratio = max_mem / max(total_vram_gib, 1.0)
                if avg_u < 60 and peak_avg_u < 90 and mem_ratio < 0.60:
                    # 保守估算：按显存余量比例放大，上限 256
                    suggested = min(256, int(self.batch_size * 0.60 / max(mem_ratio, 0.05)))
                    print(f'[GPU-MONITOR] 推荐增大 batch_size: 当前 {self.batch_size} → '
                          f'建议 {suggested}（GPU 利用率偏低，显存余量充足）')
                elif mem_ratio > 0.85:
                    print(f'[GPU-MONITOR] 显存使用率 {mem_ratio*100:.0f}%，'
                          f'建议减小 batch_size（当前 {self.batch_size}）以避免 OOM')
                else:
                    print(f'[GPU-MONITOR] 当前 batch_size={self.batch_size} 配置合适')

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
        description='IFRNet 视频插帧 —— 终极优化版 v6.3.0（单卡版）',
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
    parser.add_argument('--quiet', action=argparse.BooleanOptionalAction, default=True)
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
    print('  IFRNet 视频插帧 —— 终极优化版 v6.3.0（单卡版）')
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
    print(f'  编码器: {args.codec} → 实际: {HardwareCapability.best_encoder(args.codec)} | '
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