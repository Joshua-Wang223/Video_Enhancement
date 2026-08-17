#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# IFRNet Video Enhancement - 通用工具模块（模型加载 / 池 / tensor 辅助）。
# 镜像 external/realesrgan_video/realesrgan_utils.py 的职责。

from __future__ import annotations

import dataclasses
import os
import queue
import sys
import threading
import time
from collections import deque
from contextlib import nullcontext
from typing import List, Optional, Tuple

import numpy as np
import torch

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from ifrnet_video.config import MODEL_MODULE_MAP, MODEL_STRIDE

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



def _load_ifrnet_module(model_name: str):
    import importlib
    module_name = MODEL_MODULE_MAP.get(model_name, 'models.IFRNet_S')
    mod = importlib.import_module(module_name)
    mod.warp = _cached_warp
    return mod.Model, mod


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
            f'\n[PinnedResultPool] 分配 {pool_size} × ({max_BT},{H_pad},{W_pad},3) '
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

    def free(self) -> int:
        """
        [FIX-POOL-LEAK] 显式释放所有 pinned buffer，解除 CUDA 锁页内存占用。

        背景：Python GC 不会立即回收 cudaHostAlloc 分配的锁页内存（底层由 CUDA
        驱动管理，引用计数归零后仍可能驻留直至下次 GC 扫描）。若跨段不主动释放，
        各段 PinnedResultPool 会持续叠加（每段 ~1 GiB），累积 4-6 GiB 锁页内存，
        导致 DMA 带宽下降 → D2H 变慢 → result_queue 满载 → GPU P50 利用率从
        88% 跌至 8%，吞吐量减半。

        调用时机：_infer_loop finally 块，确保每段推理线程退出前完成释放。
        返回值：实际释放的 buffer 数量（用于日志确认）。
        """
        freed = 0
        while True:
            try:
                buf = self._q.get_nowait()
                del buf
                freed += 1
            except queue.Empty:
                break
        return freed


# ─────────────────────────────────────────────────────────────────────────────
# [FIX-T3-V643] Pinned Ring Buffer — T2→T3 零拷贝环形缓冲
# ─────────────────────────────────────────────────────────────────────────────

class PinnedRingBuffer:
    """
    [FIX-T3-V643] 预分配 N 个 pinned memory slot 的环形缓冲区。

    T2 (推理线程) 获取空 slot → GPU 裁剪 + D2H（保持 RGB）→ 存储 CUDA event → 标记为可读。
    T3 (写入线程) 获取可读 slot → 等待 event → memoryview 零拷贝写 pipe → 标记为空闲。

    每帧数据在 pinned memory 中完全连续（无 padding），memoryview 切片即得完整帧。
    """

    def __init__(self, num_slots: int, max_frames_per_slot: int, H: int, W: int):
        self.num_slots = num_slots
        self.H = H   # 原始高度（无 padding）
        self.W = W   # 原始宽度（无 padding）
        self.frame_bytes = H * W * 3
        self.max_frames = max_frames_per_slot
        self.slot_bytes = max_frames_per_slot * self.frame_bytes

        self.slots: List[torch.Tensor] = []
        _mb = num_slots * self.slot_bytes / 1e6
        print(
            f'[PinnedRingBuffer] 分配 {num_slots} × ({max_frames_per_slot},{H},{W},3) '
            f'uint8 pinned，共 {_mb:.0f} MB', flush=True
        )
        for _ in range(num_slots):
            t = torch.empty(max_frames_per_slot, H, W, 3, dtype=torch.uint8).pin_memory()
            self.slots.append(t)

        # 每 slot 的元数据: (n_frames, orig_H, orig_W, B, T, img1_list, cuda_event)
        self._meta: List[Optional[Tuple]] = [None] * num_slots

        self._write_sem = threading.Semaphore(num_slots)   # 可写 slot 数
        self._read_sem  = threading.Semaphore(0)           # 可读 slot 数
        self._write_idx = 0
        self._read_idx  = 0
        self._lock = threading.Lock()
        self._error: Optional[Exception] = None

    def writer_acquire(self, timeout: float = 30.0) -> Tuple[int, torch.Tensor]:
        """T2 获取一个空 slot 的 (slot_id, pinned_tensor)。"""
        if not self._write_sem.acquire(timeout=timeout):
            raise RuntimeError(
                f'[PinnedRingBuffer] writer_acquire 超时（{timeout}s），T3 可能卡死'
            )
        with self._lock:
            slot_id = self._write_idx % self.num_slots
            self._write_idx += 1
        return slot_id, self.slots[slot_id]

    def writer_commit(self, slot_id: int, n_frames: int, orig_H: int, orig_W: int,
                      B: int, T: int, img1_list: list, cuda_event):
        """T2 写完 slot + 记录 CUDA event，标记为可读。"""
        self._meta[slot_id] = (n_frames, orig_H, orig_W, B, T, img1_list, cuda_event)
        self._read_sem.release()

    def reader_acquire(self, timeout: float = 30.0) -> Tuple[int, memoryview, int, int, int, int, list, object]:
        """T3 获取可读 slot，返回 (slot_id, memoryview, n_frames, orig_H, orig_W, B, T, img1_list, event)。"""
        if not self._read_sem.acquire(timeout=timeout):
            raise RuntimeError(
                f'[PinnedRingBuffer] reader_acquire 超时（{timeout}s），T2 可能卡死'
            )
        with self._lock:
            slot_id = self._read_idx % self.num_slots
            self._read_idx += 1
        meta = self._meta[slot_id]
        self._meta[slot_id] = None
        n_frames, orig_H, orig_W, B, T, img1_list, cuda_event = meta
        mv = memoryview(self.slots[slot_id].numpy())
        return slot_id, mv, n_frames, orig_H, orig_W, B, T, img1_list, cuda_event

    def reader_release(self, slot_id: int):
        """T3 读完 slot，释放回可写池。"""
        self._write_sem.release()

    def free(self) -> int:
        """显式释放所有 pinned buffer，解除 CUDA 锁页内存占用。"""
        freed = 0
        for t in self.slots:
            del t
            freed += 1
        self.slots.clear()
        self._meta.clear()
        return freed


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
                       to_rgb: bool = False, slot: int = 0) -> torch.Tensor:
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
    cpu_t = pool.get_for_frames(frames, to_rgb=False, slot=slot)
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
    # [FIX-T3-V643-COLOR] 保持 RGB（不翻转），pix_fmt 已是 rgb24
    return [np.ascontiguousarray(arr[i, :orig_H, :orig_W]) for i in range(arr.shape[0])]


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

# ── [FIX-NVDEC-THREAD-CAP] NVDEC 解码线程钳位 ────────────────────────────────
# ffmpeg 6.1.1 的 NVDEC 解码 surface 计算公式：
#   ulNumDecodeSurfaces = ref_frame_count + num_reorder_frames
#                         + 2(deinterlace) + thread_count + 3(基础工作 surface)
# 32 是驱动硬上限（cudaVideoDecoder 拒绝），ffmpeg 6.1.1 无 FFMIN(pool,32) 钳位。
# 实测：-threads 8 → 32 surfaces 成功；-threads 9 → 33 surfaces 被驱动拒绝。
# 注意：仅影响解码侧命令；编码侧线程数由 _detect_encode_parallelism() 独立控制。
_MAX_DECODE_THREADS = 8
_DECODE_THREAD_HINT_SHOWN = False

def _clamp_decode_threads(requested: Optional[int] = None) -> int:
    """钳位 NVDEC 解码线程数（驱动 32-surface 硬上限），超限时打印一次提示。"""
    global _DECODE_THREAD_HINT_SHOWN
    n = int(requested) if requested else (os.cpu_count() or 4)
    if n > _MAX_DECODE_THREADS:
        if not _DECODE_THREAD_HINT_SHOWN:
            print(f"[decode] threads={n} exceeds max {_MAX_DECODE_THREADS}, "
                  f"clamped (NVDEC 32-surface limit)", flush=True)
            _DECODE_THREAD_HINT_SHOWN = True
        n = _MAX_DECODE_THREADS
    return n
