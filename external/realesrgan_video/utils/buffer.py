#!/usr/bin/env python3
import threading
import time
import numpy as np
import torch
from typing import List, Optional, Tuple, Dict, Any
from collections import deque

_thread_local = threading.local()

class PinnedBufferPool:
    def __init__(self):
        self._buf: Optional[torch.Tensor] = None
        self._out_buf: Optional[torch.Tensor] = None
    def get_for_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        arr = np.stack(frames, axis=0)
        src = torch.from_numpy(arr)
        n_elem = src.numel()
        if self._buf is None or self._buf.numel() < n_elem:
            self._buf = torch.empty(n_elem, dtype=torch.uint8).pin_memory()
        dst = self._buf[:n_elem].view_as(src)
        dst.copy_(src)
        return dst
    def get_output_buf(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        n_elem = 1
        for s in shape: n_elem *= s
        if (self._out_buf is None or self._out_buf.dtype != dtype or self._out_buf.numel() < n_elem):
            self._out_buf = torch.empty(n_elem, dtype=dtype).pin_memory()
        return self._out_buf[:n_elem].view(shape)

def get_pinned_pool() -> PinnedBufferPool:
    if not hasattr(_thread_local, 'pool'):
        _thread_local.pool = PinnedBufferPool()
    return _thread_local.pool

class ThroughputMeter:
    """滑动窗口FPS统计 - 修复版本"""
    def __init__(self, window: int = 20):
        self._times: deque = deque(maxlen=window)
        self._total = 0
        self._start_time = time.time()
    def update(self, n: int):
        current_time = time.time()
        self._times.append((current_time, n))
        self._total += n
    def fps(self) -> float:
        if len(self._times) < 2:
            if self._total == 0: return 0.0
            total_time = time.time() - self._start_time
            return self._total / total_time if total_time > 0 else 0.0
        t0, t1 = self._times[0][0], self._times[-1][0]
        dt = t1 - t0
        if dt <= 0: return 0.0
        window_frames = sum(n for _, n in self._times)
        return window_frames / dt
    def eta(self, total: int) -> float:
        fps = self.fps()
        if fps <= 0: return float('inf')
        remaining = total - self._total
        return max(0, remaining / fps)