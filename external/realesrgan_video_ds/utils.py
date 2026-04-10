#!/usr/bin/env python3
"""
Real-ESRGAN Video Enhancement - 通用工具模块
包含：ThroughputMeter, PinnedBufferPool, get_video_meta_info, _build_upsampler
"""

import os
import sys
import time
import fractions
import threading
from collections import deque
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import ffmpeg

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from config import MODEL_CONFIG, models_RealESRGAN

_thread_local = threading.local()


class PinnedBufferPool:
    """线程本地 pinned CPU buffer 池，避免每批 H2D 前的 pin_memory 开销。"""
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
            if self._total == 0:
                return 0.0
            total_time = time.time() - self._start_time
            return self._total / total_time if total_time > 0 else 0.0

        t0 = self._times[0][0]
        t1 = self._times[-1][0]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        window_frames = sum(n for _, n in self._times)
        return window_frames / dt

    def eta(self, total: int) -> float:
        fps = self.fps()
        if fps <= 0:
            return float('inf')
        remaining = total - self._total
        return max(0, remaining / fps)


def get_video_meta_info(video_path: str) -> dict:
    """通过 ffprobe 获取视频元数据，包含宽高/帧率/帧数/音轨。"""
    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        print(f"❌ ffprobe 失败: {video_path}")
        print(e.stderr.decode('utf-8', errors='ignore'))
        raise
    video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
    has_audio = any(s['codec_type'] == 'audio' for s in probe['streams'])
    vs = video_streams[0]

    fps_str = vs.get('avg_frame_rate', '24/1')
    try:
        fps = float(fractions.Fraction(fps_str))
    except:
        fps = 24.0

    if 'nb_frames' in vs and vs['nb_frames'].isdigit():
        nb = int(vs['nb_frames'])
    elif 'duration' in vs:
        nb = int(float(vs['duration']) * fps)
    else:
        nb = 0

    return {
        'width': vs['width'],
        'height': vs['height'],
        'fps': fps,
        'audio': ffmpeg.input(video_path).audio if has_audio else None,
        'nb_frames': nb,
    }


def _build_upsampler(model_name: str, dni_weight, tile: int, tile_pad: int, pre_pad: int,
                     use_half: bool, device: torch.device) -> RealESRGANer:
    """从 MODEL_CONFIG 构建 RealESRGANer。"""
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"未知模型名称: {model_name}")

    model, netscale, urls = MODEL_CONFIG[model_name]

    # 下载模型文件
    model_paths = []
    for url in urls:
        model_path = load_file_from_url(url, models_RealESRGAN, True)
        model_paths.append(model_path)

    model_path = model_paths[0] if model_paths else None

    return RealESRGANer(
        scale=netscale, model_path=model_path, dni_weight=dni_weight,
        model=model, tile=tile, tile_pad=tile_pad,
        pre_pad=pre_pad, half=use_half, device=device,
    )