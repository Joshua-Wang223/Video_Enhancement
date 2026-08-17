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
    """线程本地 pinned CPU buffer 池，避免每批 H2D/D2H 前后的 pin_memory 开销。

    [FIX-PINNED-POOL-RACE]
    旧实现每种用途（输入 H2D / 输出 D2H）只有**一块共享 buffer**，且没有任何同步保护：
        - get_for_frames(): CPU 端 `dst.copy_(src)` 是同步写入，但上一次调用把同一块
          buffer 交给了某个 stream 做 `.to(device, non_blocking=True)` 的异步 H2D 拷贝；
          如果那次拷贝在 GPU 上实际执行（DMA 读取 pinned 内存）还没完成，这次的写入就会
          和 GPU 的读取竞争同一块物理内存，导致拷到 GPU 上的数据是新旧混合甚至纯粹是
          上一批遗留的旧数据。
        - get_output_buf(): 调用方在 `out_pinned.copy_(out_perm, non_blocking=True)`
          （异步 D2H）之后往往立刻 `.numpy()` 读取数据，同样没有等待这次拷贝真正完成，
          读到的可能是还没写完 / 上一批遗留的旧数据。
    这正是分段超分视频里"开头/中途某几帧内容被更早批次顶替"现象的根因：输出帧的
    像素内容和更早批次的输出逐像素相同。

    修复方案：
        - 每种 buffer 改为 N 路环形缓冲（默认 2 路，可传参数放大做更深的流水线重叠）。
        - 用 torch.cuda.Event 追踪"这个 slot 上一次的异步拷贝是否已经真正执行完"：
          轮到某个 slot 被再次写入前，必须先等待它上一次关联的 event。
        - 调用方在发起异步拷贝后必须调用 mark_issued()/mark_output_issued()，
          在正确的 stream 上记录 event；读输出 buffer 前必须调用
          wait_output_ready() 确保 D2H 已完成。
        - 如果调用方忘记调用 mark_*（没有配合升级的旧代码），会退化为保守的
          `torch.cuda.synchronize()` 整卡同步兜底 —— 牺牲一点重叠性能，
          但绝不允许出现数据竞态。
    """

    def __init__(self, num_slots: int = 2):
        self._num_slots = max(2, num_slots)

        # 输入端（H2D 前的 pinned staging buffer）
        self._in_bufs: List[Optional[torch.Tensor]] = [None] * self._num_slots
        self._in_events: List[Optional[Any]] = [None] * self._num_slots
        self._in_slot: int = 0
        self._in_pending_slot: Optional[int] = None

        # 输出端（D2H 后的 pinned staging buffer）
        self._out_bufs: List[Optional[torch.Tensor]] = [None] * self._num_slots
        self._out_events: List[Optional[Any]] = [None] * self._num_slots
        self._out_slot: int = 0
        self._out_pending_slot: Optional[int] = None

    # ------------------------------------------------------------------
    # 输入端：H2D 前的 pinned staging buffer
    # ------------------------------------------------------------------

    def get_for_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """取一个可安全写入的 pinned buffer slot，写入 frames 数据后返回其 CPU 视图。

        调用方随后应在自己的 stream 上发起 `.to(device, non_blocking=True)`，
        然后**必须**调用 `mark_issued(stream)`，以便下次轮到这个 slot 时能
        正确等待该拷贝真正完成，而不是提前覆写。
        """
        arr = np.stack(frames, axis=0)
        src = torch.from_numpy(arr)
        n_elem = src.numel()

        slot = self._in_slot
        self._in_slot = (self._in_slot + 1) % self._num_slots

        self._wait_slot(self._in_bufs, self._in_events, slot)

        buf = self._in_bufs[slot]
        if buf is None or buf.numel() < n_elem:
            buf = torch.empty(n_elem, dtype=torch.uint8).pin_memory()
            self._in_bufs[slot] = buf

        dst = buf[:n_elem].view_as(src)
        dst.copy_(src)

        # 这次写入还没有关联的 event，等调用方 mark_issued() 才补上；
        # 在此之前如果这个 slot 又被轮到，_wait_slot 会用兜底同步保护。
        self._in_events[slot] = None
        self._in_pending_slot = slot
        return dst

    def mark_issued(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """在对 get_for_frames() 返回的 buffer 发起 H2D 拷贝后调用。"""
        if self._in_pending_slot is None:
            return
        ev = torch.cuda.Event()
        ev.record(stream if stream is not None else torch.cuda.current_stream())
        self._in_events[self._in_pending_slot] = ev
        self._in_pending_slot = None

    # ------------------------------------------------------------------
    # 输出端：D2H 后的 pinned staging buffer
    # ------------------------------------------------------------------

    def get_output_buf(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """取一个可安全写入（被 GPU D2H 拷贝写入）的 pinned buffer slot。

        调用方在发起 `.copy_(src, non_blocking=True)` 之后**必须**调用
        `mark_output_issued(stream)`；在把返回的 tensor 转成 numpy 读取
        数据前，**必须先调用 `wait_output_ready()`** 确保 D2H 拷贝真正完成，
        否则读到的是尚未写完 / 上一批遗留的旧数据。
        """
        n_elem = 1
        for s in shape:
            n_elem *= s

        slot = self._out_slot
        self._out_slot = (self._out_slot + 1) % self._num_slots

        self._wait_slot(self._out_bufs, self._out_events, slot)

        buf = self._out_bufs[slot]
        if (buf is None or buf.dtype != dtype or buf.numel() < n_elem):
            buf = torch.empty(n_elem, dtype=dtype).pin_memory()
            self._out_bufs[slot] = buf

        self._out_events[slot] = None
        self._out_pending_slot = slot
        return buf[:n_elem].view(shape)

    def mark_output_issued(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """在对 get_output_buf() 返回的 buffer 发起 D2H 拷贝后调用。

        注意：这里不会清空 pending slot 记录——wait_output_ready() 还需要
        用它定位到底该等哪个 slot 的 event，在那边才真正清空。
        """
        if self._out_pending_slot is None:
            return
        ev = torch.cuda.Event()
        ev.record(stream if stream is not None else torch.cuda.current_stream())
        self._out_events[self._out_pending_slot] = ev

    def wait_output_ready(self) -> None:
        """在把 get_output_buf() 返回的 tensor 转成 numpy / CPU 读取之前调用，
        阻塞直到对应的 D2H 拷贝真正完成。这是修复"读到未写完/旧数据"的关键一步。
        """
        if self._out_pending_slot is None:
            return
        ev = self._out_events[self._out_pending_slot]
        if ev is not None:
            ev.synchronize()
        else:
            # 调用方没有调用 mark_output_issued，无法精确判断拷贝是否完成，
            # 保守地整卡同步兜底，宁可牺牲性能也不允许读到脏数据。
            torch.cuda.synchronize()
        self._out_pending_slot = None

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _wait_slot(bufs: List[Optional[torch.Tensor]],
                    events: List[Optional[Any]], slot: int) -> None:
        """复用某个 slot 之前，确保它上一次关联的异步拷贝已经真正执行完。"""
        if bufs[slot] is None:
            return  # 这个 slot 从没被用过，天然安全，无需等待
        ev = events[slot]
        if ev is not None:
            ev.synchronize()
        else:
            # 该 slot 之前被写过，但没有关联 event（调用方没调用 mark_*），
            # 无法精确判断，保守地整卡同步兜底。
            torch.cuda.synchronize()


def _get_pinned_pool(num_slots: int = 2) -> PinnedBufferPool:
    """取线程本地 pinned buffer 池。首次调用时按 num_slots 创建
    （[FIX-DEFER-RESOLVE] SR 线程传 4：延迟解析路径要求输出环深 ≥ 在飞句柄数+1，
    保证某 slot 被 GPU 复用覆写前，上一占用者的 CPU 拷贝必然已完成）。"""
    if not hasattr(_thread_local, 'pool'):
        _thread_local.pool = PinnedBufferPool(num_slots=num_slots)
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

    nb = 0
    if 'duration' in vs:
        dur = float(vs['duration'])
        if dur > 0:
            nb = int(round(dur * fps))  # round 避免 int 截断导致的浮点精度 1 帧偏差
            # 交叉验证：若 nb_frames 与 duration×fps 偏差 > 5%，警告
            if 'nb_frames' in vs and str(vs['nb_frames']).isdigit():
                nb_meta = int(vs['nb_frames'])
                if nb_meta > 0 and nb > 0 and abs(nb_meta - nb) / max(nb_meta, nb) > 0.05:
                    print(f'⚠️  ffprobe元数据 nb_frames={nb_meta} 与 duration×fps={nb} 不一致，'
                          f'使用后者（分段文件 -c copy 常见）', flush=True)
    elif 'nb_frames' in vs and str(vs['nb_frames']).isdigit():
        nb = int(vs['nb_frames'])

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