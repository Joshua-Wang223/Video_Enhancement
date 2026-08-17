#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# IFRNet Video Enhancement - 三阶段流水线模块（T1 Reader / T2 GPU 推理 / T3 Writer）。
# 包含：IFRNetPipelineRunner、GPUMonitor/GPUStats、硬件画像与队列自动调优。
# 镜像 external/realesrgan_video/pipeline.py 的职责。

from __future__ import annotations

import dataclasses
import json
import os
import queue
import sys
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch

# [FIX-OOM-TRT] expandable_segments 减少 PyTorch CUDA allocator 碎片化。
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from config import base_dir
from ifrnet_utils import CudaEventPool, _PinnedResultItem, _get_pinned_pool
from ffmpeg_io import _detect_encode_parallelism, _software_encode_fps
from nvenc_sdk import (
    _NVENCEncodeThread,
    _rgb_to_nv12_gpu,
    _rgb_to_nv12_gpu_batch,
)


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
        codec:            str   = '',       # [FIX-NVENC-AWARE] 编码器名称，用于避开 NVENC 误判
        slot_mb:          float = 0.0,      # [FIX-MAXRQ-DYNAMIC] PinnedPool 每槽 MiB，用于内存上限约束
        t2_ms:            float = 0.0,      # [FIX-MAXRQ-DYNAMIC] T2 推理延迟（ms）
        t3_ms:            float = 0.0,      # [FIX-MAXRQ-DYNAMIC] T3 编码延迟（ms）
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
            # [FIX-MAXPQ-DYNAMIC] 建议上限随绝对上限 24 提升（原硬编码 8）；
            # 内存安全约束在消费端（ADAPTIVE-QUEUE 决策块）由 _compute_max_pair_queue 执行
            sug_pq = min(_PAIR_Q_ABS_CAP, current_pair_q + 2)
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
        sug_rq = None   # [FIX-MAXRQ-DYNAMIC] 统一收集建议值，用于后续内存上限约束
        if self._is_t3_bottleneck(stats, codec=codec):
            if current_result_q > 16:
                sug_rq = max(16, current_result_q - 8)
                print(f'[GPU-MONITOR] ⚠️  检测到 T3-bottleneck（编码器是真正瓶颈）：'
                      f'result_queue 建议缩小 {current_result_q} → {sug_rq}'
                      f'  （空闲占比 {stats.low_util_frac*100:.0f}%，均值 {stable_avg:.0f}%，'
                      f'增大队列无助提速，仅增加 PinnedPool 内存压力）')
            else:
                print(f'[GPU-MONITOR] ⚠️  检测到 T3-bottleneck（编码器是真正瓶颈）：'
                      f'result_queue={current_result_q} 保持不变'
                      f'  （根本瓶颈在编码器速度，应考虑换用更快的 preset'
                      f'{"" if "nvenc" in codec.lower() else " 或 NVENC"}）')
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

        # [FIX-MAXRQ-DYNAMIC] PinnedPool 内存上限约束（与 get_queue_suggestions 一致）
        if slot_mb > 0.0 and sug_rq is not None:
            _mem_avail_gb_pr = _detect_encode_parallelism()['mem_avail_gb']
            _max_rq_pr = _compute_max_result_queue(
                slot_mb=slot_mb, mem_avail_gb=_mem_avail_gb_pr,
                T2_ms=t2_ms, T3_ms=t3_ms,
            )
            if sug_rq > _max_rq_pr:
                print(f'[GPU-MONITOR] ℹ️  PinnedPool 内存上限约束: '
                      f'result_queue 理想值 {sug_rq} → {_max_rq_pr}'
                      f'  (slot={slot_mb:.1f} MB × {_max_rq_pr}'
                      f' ≈ {slot_mb * _max_rq_pr:.0f} MB'
                      f'  ≤ RAM预算 {_mem_avail_gb_pr * 1024 * 0.06:.0f} MB'
                      f'  [mem_avail={_mem_avail_gb_pr:.1f} GB × 6%])')

    def get_queue_suggestions(
        self,
        stats:            GPUStats,
        current_pair_q:   int,
        current_result_q: int,
        slot_mb:          float = 0.0,   # [FIX-T3-MEMCAP] 每个 result slot 的 MiB 数
        t2_ms:            float = 0.0,   # [FIX-MAXRQ-DYNAMIC] T2 推理延迟（ms），0=未知
        t3_ms:            float = 0.0,   # [FIX-MAXRQ-DYNAMIC] T3 编码延迟（ms），0=未知
        codec:            str   = '',    # [FIX-NVENC-AWARE] 编码器名称，用于避开 NVENC 误判
        ) -> Tuple[int, int]:
        """
        [FIX-T3-DETECT / FIX-T3-MEMCAP / FIX-POOL-AUTOSCALE / FIX-MAXRQ-DYNAMIC]
        返回 (建议 pair_queue, 建议 result_queue)。用于跨段自适应队列调整，不打印信息。

        新增逻辑：
        · T3-bottleneck 时不增大 result_queue（否则 PinnedPool 雪球式积累）。
        · slot_mb > 0 时对 result_queue 施加 PinnedPool 内存上限约束。
        · [FIX-MAXRQ-DYNAMIC] 上限改由 _compute_max_result_queue() 三轴动态计算
          （RAM 上限 / T3/T2 下限 / 绝对上限），替代静态 _PINNED_POOL_MAX_MB。
        """
        pair_q   = current_pair_q
        result_q = current_result_q
        stable_std   = stats.stable_std
        low_frac     = stats.low_util_frac
        headroom_gib = stats.mem_headroom_gib
        stable_p95   = stats.stable_p95

        # pair_queue 建议（T3-bottleneck 不影响 pair_queue 逻辑）
        # [FIX-MAXPQ-DYNAMIC] 建议上限随绝对上限 24 提升（原硬编码 8）
        if stable_std > 25.0 or low_frac > 0.30:
            pair_q = min(_PAIR_Q_ABS_CAP, current_pair_q + 2)
        elif stable_std < 10.0 and low_frac < 0.10 and current_pair_q > 4:
            pair_q = max(3, current_pair_q - 1)

        # [FIX-T3-DETECT] result_queue 建议：T3-bottleneck 时保持或缩小
        if self._is_t3_bottleneck(stats, codec=codec):
            # T3 是真正瓶颈，增大 result_queue 无助于提速，反而增加内存压力
            if current_result_q > 16:
                result_q = max(16, current_result_q - 8)
            # else: 已经很小了，保持不变
        elif stable_p95 > 85.0 and stable_std > 20.0:
            result_q = min(64, current_result_q + 8)
        elif headroom_gib > 2.0 and current_result_q < 32:
            result_q = min(48, current_result_q + 8)

        # [FIX-T3-MEMCAP / FIX-MAXRQ-DYNAMIC] PinnedPool 内存上限约束（三轴动态计算）
        if slot_mb > 0.0:
            _mem_avail_gb_gqs = _detect_encode_parallelism()['mem_avail_gb']
            _max_rq_by_mem = _compute_max_result_queue(
                slot_mb      = slot_mb,
                mem_avail_gb = _mem_avail_gb_gqs,
                T2_ms        = t2_ms,
                T3_ms        = t3_ms,
            )
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
    def _is_t3_bottleneck(stats: 'GPUStats', codec: str = '') -> bool:
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

        [FIX-NVENC-AWARE] NVENC 是独立硬件编码单元，不体现在 CUDA 利用率中；
        当编码器为 NVENC 时，低 CUDA 利用率是正常现象，不应判定为 T3-bottleneck。
        """
        if 'nvenc' in codec.lower():
            return False
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

_T2_BASELINE_HWB  = float(576 * 736 * 24)
_T2_FIXED_MS      = 240.0   # torch.compile / eager 路径固定 overhead（含 JIT 编译）
# [FIX-T2-TRT-CALIB] TRT 路径固定 overhead 实测约 2-5ms（无 JIT、纯硬件调度）
_T2_FIXED_MS_TRT  = 5.0
_T2_VAR_MS        = 25.0    # eager/compile 路径斜率 (保留不变)
_T2_VAR_MS_TRT    = 335.0   # [FIX-T2-VAR-TRT] TRT 路径基于 T4 实测校准 (Tesla T4, bs=24, 576×736)



def _pool_limit_mb_for_profile(profile: '_HWProfile') -> float:
    """
    [FIX-POOL-AUTOSCALE] 依据 GPU tier 自动计算合理的 PinnedPool 内存上限（MiB）。

    设计原则：
    · PinnedPool 使用系统 RAM 锁页内存（不占 VRAM），但过多会拖慢 DMA 带宽。
    · 上限随 GPU tier 递增（高端 GPU 往往配套大内存服务器）。
    · 兼顾实际可用 RAM：上限不超过 MemAvailable × 12%，最低保底 1024 MiB。

    tier 分档（来自 _GPU_PROFILES_TABLE）：
      GTX 1080 / 1080 Ti (tier ≤ 0.5)  → 1024 MiB
      T4 / RTX 2080 Ti   (tier ≤ 1.1)  → 2048 MiB   ← T4 推荐值
      RTX 3090 / 4070 Ti (tier ≤ 2.3)  → 3072 MiB
      A10 / L40S / 4080+ (tier ≤ 5.9)  → 4096 MiB
      A100 / A800         (tier ≤ 8.9)  → 6144 MiB
      H100 / H800         (tier > 8.9)  → 8192 MiB
    """
    tier = getattr(profile, 'gpu_tier', 1.0)
    if tier > 8.9:
        tier_limit = 8192.0
    elif tier > 4.7:
        tier_limit = 6144.0
    elif tier > 1.8:
        tier_limit = 4096.0
    elif tier > 1.1:
        tier_limit = 3072.0
    elif tier > 0.5:
        tier_limit = 2048.0
    else:
        tier_limit = 1024.0

    # 兼顾系统可用 RAM：上限 ≤ MemAvailable × 12%，最低 1024 MiB
    try:
        mem_avail_mb = 0.0
        try:
            with open('/proc/meminfo', 'r') as _mf:
                for _line in _mf:
                    if _line.startswith('MemAvailable:'):
                        mem_avail_mb = int(_line.split()[1]) / 1024.0  # kB → MiB
                        break
        except OSError:
            import psutil as _ps
            mem_avail_mb = _ps.virtual_memory().available / (1024.0 ** 2)
        ram_limit = max(1024.0, mem_avail_mb * 0.12)
    except Exception:
        ram_limit = tier_limit  # 无法读取时不做 RAM 约束

    return min(tier_limit, ram_limit)

# 模块级常量：PinnedPool 锁页内存上限（MiB），按 GPU 型号自动缩放，只初始化一次。
# [FIX-MAXRQ-DYNAMIC] 此常量仅保留供 PinnedPool 构建阶段参考；
# 运行时队列约束改由 _compute_max_result_queue() 动态计算，不再直接使用此常量。

_PINNED_POOL_MAX_MB: float = _pool_limit_mb_for_profile(
    _detect_hw_profile(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
)



def _compute_max_result_queue(
    slot_mb: float,
    mem_avail_gb: float,
    T2_ms: float = 0.0,
    T3_ms: float = 0.0,
    ram_budget_fraction: float = 0.06,
    abs_cap: int = 48,
) -> int:
    """
    [FIX-MAXRQ-DYNAMIC] 动态计算 result_queue 安全上限（三轴联合约束）。

    三轴约束原理
    ─────────────────────────────────────────────────────────────────────────
    轴1 RAM 上限（主要约束）：
      PinnedPool 使用系统 RAM 锁页内存（不占 VRAM），但过多会竞争 DMA 带宽。
      上限 = mem_avail_gb × ram_budget_fraction / slot_mb
      ram_budget_fraction 默认 6%：T4+20GiB RAM+47MB/slot ≈ 26 槽，与实测吻合。
      该参数是唯一需要人工校准的旋钮；无论换 GPU / 换分辨率 / 换机器都自适应。

    轴2 T3/T2 速度比（最小需求下限）：
      result_queue 存在是为了缓冲 T3 比 T2 慢这一事实，但 libx264 是流式编码器，
      并非每批阻塞整个 T3_ms；实际突发缓冲需求远小于 T3/T2 比值。
      经验系数 0.22（= 0.15 × 1.5）：在 T4+crf=0+ultrafast 场景实测吻合。
      floor_by_t3 = max(8, int(T3_ms / T2_ms × 0.22))
      T2_ms/T3_ms 均为 0 时（未知），跳过此轴，floor = 8。

    轴3 绝对上限（保险兜底）：
      防止内存估算本身有误（如 mem_avail_gb 异常偏大）时失控。
      固定为 abs_cap（默认 48）。

    三轴关系：RAM 决定上限，T3/T2 比决定下限，取二者之间合理值：
      result = max(floor_by_t3, min(cap_by_ram, abs_cap))

    参数
    ─────────────────────────────────────────────────────────────────────────
    slot_mb            : 每个 result slot 的锁页内存占用（MiB），
                         = effective_bs × T × H_pad × W_pad × 3 / 1e6
    mem_avail_gb       : 系统当前可用 RAM（GiB），来自 _detect_encode_parallelism()
    T2_ms              : GPU 推理延迟（ms），来自 AUTO-TUNE 测量；0 表示未知
    T3_ms              : 每批次编码延迟（ms），来自 AUTO-TUNE 估算；0 表示未知
    ram_budget_fraction: PinnedPool 允许占用可用 RAM 的比例（默认 6%）
    abs_cap            : result_queue 绝对上限（默认 48）
    """
    if slot_mb <= 0.0:
        return abs_cap

    # 轴1: RAM 上限（可用内存 × 预算比 / 每槽大小）
    ram_budget_mb = mem_avail_gb * 1024.0 * ram_budget_fraction
    cap_by_ram = max(8, int(ram_budget_mb / slot_mb))

    # 轴2: T3/T2 实际需求下限（仅在两者均已知时计算）
    # libx264 流式编码 burst 系数约 0.22（经验值，T4 实测 ~8 槽已足够解耦）
    if T2_ms > 0.0 and T3_ms > 0.0:
        floor_by_t3 = max(8, int(T3_ms / T2_ms * 0.22))
    else:
        floor_by_t3 = 8   # 未知时使用安全下限

    # 轴3: 绝对上限兜底，防止内存估算失控
    return max(floor_by_t3, min(cap_by_ram, abs_cap))


# [FIX-MAXPQ-DYNAMIC] pair_queue 绝对上限（原硬编码 8 的提升值）

_PAIR_Q_ABS_CAP: int = 24



def _compute_max_pair_queue(
    slot_mb: float,
    mem_avail_gb: float,
    ram_budget_fraction: float = 0.15,
    abs_cap: int = _PAIR_Q_ABS_CAP,
) -> int:
    """
    [FIX-MAXPQ-DYNAMIC] 动态计算 pair_queue 安全上限（两轴联合约束）。

    与 _compute_max_result_queue 同构，但有两处关键差异：
      1) pair slot 是普通 numpy 数组（pageable RAM），非 pinned 锁页内存，
         不参与 DMA 带宽竞争，因此预算比从 6% 放宽到 15%；
      2) pair_queue 缓冲的是 "T1(读取) 远快于 T2(推理)"（T2/T1 高达 25-30×），
         稳态下 8 槽已足够，更深的意义在于：
           · 吸收 T1 的瞬时 stall（pad CPU 突发 / NVDEC 抖动 / 预取槽竞争）；
           · 支撑 P 跟随 R 的对称缓冲（result_queue 可达 24-48 时 P 需 12-24）。
         因此需求下限固定为 8（实测无饥饿基线），不设 T2/T1 比例轴，
         由 GPU-MONITOR/RETUNE 建议驱动实际深度。

    轴1 RAM 上限：cap_by_ram = mem_avail_gb × ram_budget_fraction / slot_mb
    轴2 绝对上限：abs_cap（默认 24，替换原硬编码 8）

    参数
    ─────────────────────────────────────────────────────────────────────────
    slot_mb            : 每个 pair slot 的内存占用（MiB），
                         = effective_bs × 3帧(raw+2×pad) × H_pad × W_pad × 3 / 1e6
    mem_avail_gb       : 系统当前可用 RAM（GiB），来自 _detect_encode_parallelism()
    ram_budget_fraction: pair 缓冲允许占用可用 RAM 的比例（默认 15%，pageable）
    abs_cap            : pair_queue 绝对上限（默认 24）
    """
    if slot_mb <= 0.0:
        return abs_cap

    # 轴1: RAM 上限（可用内存 × 预算比 / 每槽大小），下限 8
    ram_budget_mb = mem_avail_gb * 1024.0 * ram_budget_fraction
    cap_by_ram = max(8, int(ram_budget_mb / slot_mb))

    # 轴2: 绝对上限兜底，防止内存估算失控
    return max(8, min(cap_by_ram, abs_cap))



_MODEL_T2_FACTOR = {
    'IFRNet_S_Vimeo90K': 1.0,      # 基准（最小模型）
    'IFRNet_Vimeo90K':   1.6,      # 中型
    'IFRNet_L_Vimeo90K': 3.0,      # 大型
}

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
        # [FIX-T2-TRT-CALIB] TRT 路径固定 overhead 仅 2-5ms（无 JIT），
        # 与 torch.compile/eager 的 240ms 差距 48×，必须分路处理。
        # [FIX-T2-VAR-TRT] TRT 路径可变斜率也需专用常量 (335ms vs 25ms)。
        _fixed_ms = _T2_FIXED_MS_TRT if infer_backend == 'trt' else _T2_FIXED_MS
        _var_ms   = _T2_VAR_MS_TRT if infer_backend == 'trt' else _T2_VAR_MS
        t2_base = (_fixed_ms + _var_ms * HWB / _T2_BASELINE_HWB) / max(profile.gpu_tier, 0.05)
        t2_ms   = max(t2_base * infer_factor * model_factor, 1.0)
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

    # [FIX-MAXPQ-DYNAMIC] pair 上限由动态两轴函数决定（RAM 约束 + 绝对上限 24），
    # 替换原硬编码 8：T2/T1 高达 25-30× 时 8 槽对 T1 瞬时 stall 余量不足，
    # 且无法支撑 P 跟随 R 的对称缓冲（result_queue 可达 24-48 → P 需 12-24）。
    _pair_slot_mb = effective_bs * 3 * H_pad * W_pad * 3 / 1e6  # 每 pair slot 的 MiB
    if _pair_slot_mb > 0.0:
        _pair_cap = _compute_max_pair_queue(
            slot_mb      = _pair_slot_mb,
            mem_avail_gb = _detect_encode_parallelism()['mem_avail_gb'],
        )
    else:
        _pair_cap = _PAIR_Q_ABS_CAP
    pair_depth   = max(2, min(int(_math.ceil(t2_ms / max(t1_ms, 0.1))) + 2, _pair_cap))
    result_depth = max(8, min(int(_math.ceil(t3_ms / max(t2_ms, 0.1))) + 3, 64))

    # [FIX-T3-MEMCAP / FIX-POOL-AUTOSCALE / FIX-MAXRQ-DYNAMIC] 动态约束 result_depth。
    # 每个 result slot 持有 effective_bs * T 帧的 pinned uint8 buffer。
    # 若不加约束，T3 极慢（大 T3/T2 比）时 result_depth 会达到 50+，
    # 导致 PinnedPool 分配 2 GiB+ 锁页内存，反而拖慢 DMA 带宽，形成恶性循环。
    # [FIX-MAXRQ-DYNAMIC] 改用三轴动态函数：RAM 上限 / T3/T2 下限 / 绝对上限联合约束，
    # 无论换 GPU / 换分辨率 / 换机器均自适应，无需修改硬编码常量。
    _slot_mb = effective_bs * T * H_pad * W_pad * 3 / 1e6  # 每 slot 的 MiB
    if _slot_mb > 0.0:
        _mem_avail_gb_aqd = _detect_encode_parallelism()['mem_avail_gb']
        _max_result_by_mem = _compute_max_result_queue(
            slot_mb      = _slot_mb,
            mem_avail_gb = _mem_avail_gb_aqd,
            T2_ms        = t2_ms,
            T3_ms        = t3_ms,
        )
        result_depth = min(result_depth, _max_result_by_mem)

    pool_size    = result_depth + 2
    if verbose:
        pair_mb = pair_depth * effective_bs * 3 * H_pad * W_pad * 3 / 1e6
        pool_mb = pool_size  * effective_bs * T * H_pad * W_pad * 3 / 1e6
        print(f'\n[AUTO-TUNE] {profile}  backend={infer_backend}(×{infer_factor}) model={model_name}(×{model_factor})\n'
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
        self._error: Optional[Exception] = None          # 流水线异常标志，供 run() 传播错误

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

        # [FIX-SENTINEL-V643] SENTINEL 无法放回 pair_queue 时的紧急标记
        self._pending_sentinel    = False

        # 诊断计数器：追踪 pipeline 各阶段的帧/批次数
        self._diag_reader_pairs   = 0
        self._diag_infer_batches  = 0
        self._diag_infer_pairs    = 0
        self._diag_gpu_stay_batches = 0  # GPU-STAY 路径处理的 batch 数
        self._diag_nvenc_frames   = 0     # NVENC 编码的总帧数
        self._diag_empty_h264     = 0     # 空 H.264 字节计数

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
            if self._error is None:
                self._error = e
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
            else:
                # 若队列满且无法放回，存入 _pending_sentinel 供 _infer_loop 读取
                self._pending_sentinel = True
                if not self.proc.quiet:
                    print('[IFRNet-Reader] pair_queue 持续满载，'
                          '存入 _pending_sentinel', flush=True)

    def _enqueue_pair(self, img1_raw, img0_pad, img1_pad, is_end):
        self._diag_reader_pairs += len(img1_raw)  # 诊断：计数 reader 发送的 pair 数
        item = (img1_raw, img0_pad, img1_pad, is_end)
        while self.running:
            try:
                self.pair_queue.put(item, timeout=1.0)
                return
            except queue.Full:
                infer_dead = self._infer_th is not None and not self._infer_th.is_alive()
                writer_dead = self._writer_th is not None and not self._writer_th.is_alive()
                if infer_dead or writer_dead:
                    break
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
                # [FIX-SENTINEL-LOSS] 将 SENTINEL 放回 pair_queue 供主循环消费。
                # 若 pair_queue 满载（GPU 热节流/编码阻塞），必须等待而非静默丢弃。
                # 否则 SENTINEL 永久丢失 → 推理线程提前退出 → 帧丢失。
                try:
                    self.pair_queue.put(item, timeout=5.0)
                except Exception:
                    if not self.proc.quiet:
                        print('[IFRNet-Prefetch] ⚠️ SENTINEL 放回失败（pair_queue 持续满载5s），'
                              '设置 _pending_sentinel', flush=True)
                    self._pending_sentinel = True
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
                        img0_pin = pool.get_for_frames(img0_pad, to_rgb=False, slot=slot_base)
                        img0_t   = img0_pin.to(device, non_blocking=True, dtype=dtype)
                        img1_pin = pool.get_for_frames(img1_pad, to_rgb=False, slot=slot_base + 1)
                        img1_t   = img1_pin.to(device, non_blocking=True, dtype=dtype)
                else:
                    img0_t = pool.get_for_frames(img0_pad, to_rgb=False, slot=slot_base).to(
                        device, dtype=dtype)
                    img1_t = pool.get_for_frames(img1_pad, to_rgb=False, slot=slot_base + 1).to(
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
        # [FIX-ENC-THREAD] 独立 NVENC 编码线程句柄（GPU_RAW 路径惰性初始化，其余路径为 None）
        _enc_thread: Optional[_NVENCEncodeThread] = None
        # [FIX-WRITTEN-TRUTH] 编码线程计数是否已汇总进 written（防止异常路径重复计数）
        _enc_counted = False
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
                    # [FIX-GPU-STAY] Level 1 GPU-STAY: GPU tensor → 批量RGB→NV12 → NVENC SDK 编码
                    # Infer 线程只做推理+传 GPU tensor，Writer 线程并行编码。
                    if isinstance(_item, tuple) and len(_item) >= 2 and _item[0] == 'GPU_RAW':
                        _, interp_gpu, img1_rgb, rB, rT, rorig_H, rorig_W = _item
                        _nvenc = self.proc._cached_nvenc_encoder
                        if _nvenc is None:
                            print('[IFRNet-Writer] GPU_RAW received but NVENC encoder lost, skip batch', flush=True)
                            continue
                        # 批量 RGB→NV12（单次 kernel launch 替代 N 次调用）
                        all_frames = torch.cat([interp_gpu, img1_rgb], dim=0)  # (B*T + B, H, W, 3)
                        # [DIAG-RAW] 环境变量 IFRNET_DIAG=1 时输出诊断统计，帮助定位花屏根因
                        import os as _os_diag
                        if _os_diag.environ.get('IFRNET_DIAG') == '1':
                            _di = interp_gpu.float(); _d2 = img1_rgb.float()
                            print(f'[DIAG-GPU_RAW] batch interp mean={_di.mean():.1f} std={_di.std():.1f} '
                                  f'min={_di.min():.0f} max={_di.max():.0f} | '
                                  f'img1 mean={_d2.mean():.1f} std={_d2.std():.1f} '
                                  f'min={_d2.min():.0f} max={_d2.max():.0f}', flush=True)
                        all_nv12 = _rgb_to_nv12_gpu_batch(all_frames)
                        n_interp = rB * rT

                        # [FIX-ENC-THREAD] CRITICAL: 等待当前 PyTorch stream 完成 NV12 写入，
                        # 再交给编码线程的 cuMemcpy2D 读取，防止 GPU 数据未就绪导致静默花帧。
                        torch.cuda.current_stream().synchronize()

                        # [FIX-LA-ACC-HOST] LA>0 时 _NVENCEncodeThread._loop() 会把整个 segment
                        # 的所有批次累积到 _acc_nv12（[FIX-LA-ACCUMULATE] 设计，为保证 LA FIFO
                        # 连续性 / _slot_pending 不跨批次错位，这是必要的正确性约束，不能改回
                        # per-batch 编码）。但若累积的元素继续是 GPU tensor，几千帧 NV12 数据
                        # （单帧约 (H+H//2)*W 字节，7000+ 帧可达 8~10GB+）会常驻显存不释放，
                        # 与 T2 推理批次抢显存，形成阶梯状显存上升直至 OOM。此处在入队前立即 D2H
                        # 搬到 pinned host 内存，GPU 侧引用随本次循环结束即可被 PyTorch allocator
                        # 回收复用；encode_frames_batch() 的 cuMemcpy2D 已同步支持 host 源
                        # （见 [FIX-LA-ACC-HOST] 在 encode_frames_batch 内的改动）。
                        # LA=0 走 per-batch ce_pipeline，帧不会跨批次滞留，无需搬移，继续留在 GPU
                        # 以保留原有 GPU-STAY 零拷贝性能。
                        if _nvenc._la_depth > 0:
                            # [FIX-LA-ACC-HOST-V2] 直接分配 pinned 缓冲区，一次 D2H 拷贝直接落地，
                            # 代替 .cpu().pin_memory() 的两步搬运（先 D2H→pageable，再 CPU
                            # pageable→pinned memcpy）。
                            try:
                                _pinned = torch.empty(all_nv12.shape, dtype=all_nv12.dtype,
                                                       device='cpu', pin_memory=True)
                                _pinned.copy_(all_nv12, non_blocking=False)
                                all_nv12 = _pinned
                            except RuntimeError:
                                # pinned 内存分配失败（如系统锁页内存额度耗尽），退化为普通
                                # 可分页内存，仅牺牲部分 H2D 拷贝带宽，正确性不受影响。
                                all_nv12 = all_nv12.cpu()

                        # [FIX-ENC-THREAD] 构建交叉交错顺序的帧列表，提交给独立编码线程。
                        # T3 Writer 做 RGB→NV12 kernel，编码线程做 encode_frames_batch，
                        # 两者在 T4 SM（CUDA 计算）与 NVENC 固定功能硬件上真正并行。
                        encode_order = []
                        for bi in range(rB):
                            for tj in range(rT):
                                encode_order.append(all_nv12[bi * rT + tj])
                            encode_order.append(all_nv12[n_interp + bi])

                        self.__dict__.setdefault('_diag_empty_h264', 0)
                        self.__dict__.setdefault('_diag_nvenc_frames', 0)
                        # 惰性初始化编码线程（首次 GPU_RAW 时创建）
                        if _enc_thread is None:
                            _enc_thread = _NVENCEncodeThread(
                                _nvenc, writer, batch_frames=len(encode_order))  # [FIX-CHUNK-SAFE]
                            # [FIX-DYN-TIMEOUT] 暴露到 pipeline 实例，run() 所在的主线程
                            # 据此读取 submitted_frames 动态估算 writer 线程 join 超时。
                            self._enc_thread_ref = _enc_thread
                        _is_first_submit = (_enc_thread._written == 0 and _enc_thread._empty == 0)
                        _enc_thread.submit(encode_order, force_idr_first=_is_first_submit)
                        _n_pairs_total += rB
                        # [FIX-LA-ACC-HOST] all_nv12/encode_order 的 GPU 版本源（all_frames/
                        # interp_gpu/img1_rgb）在 LA>0 时已不再需要，显式断开引用而不是等待
                        # 下一轮循环变量被覆盖，缩短显存峰值持有窗口。
                        if _nvenc._la_depth > 0:
                            all_frames = None
                        continue
                    # NVENC 直通编码 H.264 ES 输出
                    if isinstance(_item, tuple) and len(_item) == 5:
                        h264_frames, _rB, _rT, _oh, _ow = _item
                        _n_pairs_total += _rB
                        for _hitem in h264_frames:
                            if isinstance(_hitem, tuple) and _hitem[0] == b'RAW':
                                writer.write(_hitem[1].tobytes() if hasattr(_hitem[1], 'tobytes') else _hitem[1])
                                written += 1
                            elif isinstance(_hitem, bytes) and _hitem:
                                writer.write(_hitem)
                                written += 1
                        continue
                    if isinstance(_item, tuple) and len(_item) == 2 and _item[0] == 'RING':
                        # ── [FIX-T3-V643] Ring Buffer 零拷贝路径（Level 2/3） ──
                        _, _ring_slot = _item
                        _rb = self.proc._ring_buf
                        rv = _rb.reader_acquire(timeout=5.0)
                        if rv is None:
                            print('[IFRNet-Writer] ⚠️ ring buffer reader_acquire 返回 None，跳过', flush=True)
                            continue
                        _sid, _mv, _nf, _oh, _ow, _rB, _rT, _img1l, _ev = rv
                        # 等待 D2H 完成
                        if _ev is not None:
                            _ev.synchronize()
                            self._event_pool.release(_ev)
                        frame_sz = _oh * _ow * 3
                        _n_pairs_total += _rB
                        # [FIX-MEMORYVIEW-DIM] mv 为 4D memoryview（PEP 3118），
                        # 须 .cast('B') 转为 1D 字节视图后才能按字节偏移切片
                        _batch_data = bytearray()
                        mv_flat = _mv.cast('B')
                        for i in range(_rB):
                            interp_start = i * _rT * frame_sz
                            interp_end   = (i + 1) * _rT * frame_sz
                            _batch_data.extend(mv_flat[interp_start:interp_end])
                            written += _rT
                            # img1_raw: 来自 reader (RGB)，直接写入 rgb24
                            if i < len(_img1l):
                                _batch_data.extend(_img1l[i].copy().tobytes())
                                written += 1
                        writer.write_direct(bytes(_batch_data))
                        _rb.reader_release(_sid)
                        continue
                    if isinstance(_item, _PinnedResultItem):
                        n_pairs = _item.B
                        _n_pairs_total += n_pairs
                        arr = _item.buf[:_item.B * _item.T].numpy()
                        for i in range(_item.B):
                            for j in range(_item.T):
                                # [FIX-T3-V643-COLOR] 数据保持 RGB（不翻转），pix_fmt 已是 rgb24
                                fr = arr[i * _item.T + j,
                                         :_item.orig_H, :_item.orig_W]
                                if not fr.flags['C_CONTIGUOUS']:
                                    fr = np.ascontiguousarray(fr)
                                writer.write(fr)
                                written += 1
                            # img1_raw: 来自 reader (RGB)，直接写入 rgb24
                            writer.write(_item.img1_raw[i].copy())
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
                            # img1_raw: 来自 reader (RGB)，直接写入 rgb24
                            writer.write(img1_raw_list[i].copy())
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
                    avg_t = np.mean(timing_ref[-20:]) * 1000 if timing_ref else 0
                    pbar.set_postfix(
                        fps=f'{meter.fps():.1f}',
                        eta=f'{meter.eta(n_seg_est):.0f}s',
                        ms=f'{avg_t:.0f}',
                        P=self.pair_queue.qsize(),
                        R=self.result_queue.qsize(),
                        refresh=False,
                    )
                    pbar.update(_n_pairs_total)
            # [FIX-ENC-THREAD] SENTINEL 后收尾：等待编码线程排空队列 + NVENC EOS flush。
            # GPU_RAW 路径：_enc_thread.flush_and_join() 内部已调用 nvenc.flush()，
            #               并汇总写入计数到 written 及诊断计数器。
            # Level 2/3/4 路径（_enc_thread is None）：原逻辑直接 flush NVENC（如有）。
            if _enc_thread is not None:
                # [FIX-DYN-TIMEOUT] 内层等待就是真正阻塞编码收尾的地方，必须先在这里
                # 动态放宽，外层 run() 的 writer 线程 join 才有意义（否则内层仍按旧的
                # 固定 120s 提前放弃、外层放再宽的超时也等不到真正编码完成）。
                # LA=0 无段末编码脉冲，沿用原 30s 语义（走 estimate_timeout 的 min_timeout 分支）。
                _la_for_timeout = getattr(_enc_thread._nvenc, '_la_depth', 0) > 0
                if _la_for_timeout:
                    _flush_timeout = _enc_thread.estimate_timeout()
                else:
                    _flush_timeout = 30.0
                _enc_written, _enc_empty = _enc_thread.flush_and_join(timeout=_flush_timeout)
                written += _enc_written
                _enc_counted = True
                self._diag_empty_h264   = getattr(self, '_diag_empty_h264',   0) + _enc_empty
                self._diag_nvenc_frames = getattr(self, '_diag_nvenc_frames',  0) + _enc_written + _enc_empty
            else:
                # 非 GPU_RAW 路径（Level 2/3/4）：直接 flush NVENC encoder（如有）
                _nvenc_flush = getattr(self.proc, '_cached_nvenc_encoder', None)
                if _nvenc_flush is not None:
                    try:
                        _leftover = _nvenc_flush.flush()
                        if _leftover:
                            writer.write(_leftover)
                    except Exception:
                        pass
        except Exception as e:
            self._error = e
            self.running = False
            import traceback
            print(f'\n[IFRNet-Writer] 写线程异常: {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
        finally:
            # [FIX-WRITTEN-TRUTH] 异常路径下 flush_and_join 抛出导致计数未汇总，
            # 直接并入编码线程实际成功写入 muxer 的帧数，避免 pipeline._written=0
            # 触发理论值回退、掩盖输出截断。
            if _enc_thread is not None and not _enc_counted:
                written += getattr(_enc_thread, '_written', 0)
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
        nvenc_encoder:     Optional[object] = None,  # [FIX-T3-V643] GPU 直通编码器
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
                    if proc._is_first_segment():
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
                if not self.proc.quiet and proc._is_first_segment():
                    print(f'\n[AUTO-TUNE] 使用外部建议队列: pair={_pd} result={_rd}')
            else:
                _pd, _rd, _ = _auto_queue_depths(
                    self._hw_profile, self.codec, self.x264_preset, self.crf,
                    H_pad, W_pad, effective_bs, len(timesteps),
                    infer_backend=infer_be,
                    model_name=proc.model_name,   # ✅ 传入模型名
                    t3_fps_measured=self._t3_fps_measured_input,   # [FIX-T3-FPS]
                )
            self.pair_queue   = queue.Queue(maxsize=_pd)
            # [FIX-LA-RQ-BURST] LA>0 chunk encoding produces bursty consumption
            # from result_queue. Add headroom to prevent T2 blocking during
            # _encode_chunk() bursts (chunk≤1000 frames, ~5s each).
            _nvenc_la = getattr(getattr(self.proc, '_cached_nvenc_encoder', None), '_la_depth', 0)
            if _nvenc_la > 0 and _rd > 0:
                _rd = min(_rd + max(4, _rd // 4), 48)
            self.result_queue = queue.Queue(maxsize=_rd)

            if self._hw_profile.t2_measured_ms > 0:
                self._t2_estimated_ms = self._hw_profile.t2_measured_ms
            else:
                _HWB = float(H_pad * W_pad * effective_bs)
                _ifactor = _INFER_BACKEND_FACTORS.get(infer_be, 3.5)
                _mfactor = _MODEL_T2_FACTOR.get(proc.model_name, 1.0)
                # [FIX-T2-TRT-CALIB] TRT 路径使用专用固定 overhead 常量（5ms vs 240ms）
                # [FIX-T2-VAR-TRT] TRT 路径可变斜率也需专用常量 (335ms vs 25ms)
                _fixed_ms = _T2_FIXED_MS_TRT if infer_be == 'trt' else _T2_FIXED_MS
                _var_ms   = _T2_VAR_MS_TRT if infer_be == 'trt' else _T2_VAR_MS
                _t2b = (_fixed_ms + _var_ms * _HWB / _T2_BASELINE_HWB) \
                       / max(self._hw_profile.gpu_tier, 0.05)
                self._t2_estimated_ms = max(_t2b * _ifactor * _mfactor, 1.0)

        # [PINNED-D2H] 创建/复用 PinnedResultPool（检查 stream_d2h 是否可用）
        _pool_ok = False
        if (H_pad > 0 and W_pad > 0
                and getattr(proc, 'stream_d2h', None) is not None   # [STREAM-DUAL]
                and proc.device.type == 'cuda'):
            _max_BT    = effective_bs * len(timesteps)
            _pool_size = self.result_queue.maxsize + 2
            try:
                proc._result_pool, _pool_is_new = proc._get_or_create_result_pool(
                    _pool_size, _max_BT, H_pad, W_pad)
                _pool_ok = True
            except Exception as _pe:
                print(f'[IFRNet-Pipeline] PinnedResultPool 分配失败: {_pe}，回退同步 D2H',
                      flush=True)
                proc._result_pool = None

        # [SEGMENT-REUSE] 段 1 完整输出流水线详情，后续段简化
        if proc._is_first_segment():
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
            args=(timesteps, H, W, effective_bs, H_pad, W_pad, meter, nvenc_encoder),
            daemon=True, name='IFRNet-Infer',
        )
        self._infer_th.start()

        # 等待工作线程退出（先等 Writer，确保其错误先写入 self._error）
        if self._writer_th is not None:
            self._writer_th.join(timeout=10.0)

        # [FIX-JOIN-TIMEOUT] 动态超时：基于分段帧数估算，避免长分段误触发。
        # n_seg_est / 5.0 表示最慢 5 fps 仍可完成；下限 120s 保底，上限 7200s 防止
        # 极端大分段数值溢出。推理线程正常应在 ~n_seg_est / actual_fps 秒内完成。
        _infer_timeout = max(120.0, min(n_seg_est / 5.0, 7200.0))
        self._infer_th.join(timeout=_infer_timeout)
        if self._infer_th.is_alive():
            # 超时后发送停止信号，再给 60s 宽限期让 finally 块执行
            print(f'\n[IFRNet] ⚠️ 推理线程 {_infer_timeout:.0f}s 未退出，'
                  f'发送停止信号...', flush=True)
            self.running = False
            self._infer_th.join(timeout=60.0)
            if self._infer_th.is_alive():
                _msg = (f'推理线程在停止信号后 60s 仍未退出'
                        f'（总等待 {_infer_timeout + 60:.0f}s，n_seg_est={n_seg_est}）')
                self._error = RuntimeError(_msg)
                print(f'[IFRNet] ❌ {_msg}', flush=True)
            else:
                print('[IFRNet] 推理线程已响应停止信号并退出', flush=True)

        # 检查流水线是否有异常
        if self._error is not None:
            raise RuntimeError(
                f'流水线处理异常: {type(self._error).__name__}: {self._error}'
            ) from self._error

        # [FIX-RETUNE-POSTRUN] 段完成后用全段稳定 timing 做 RETUNE（精度优于早期 5-batch 采样）
        # · 采用 timing[_CALIB_SKIP:] 中位数，剔除流水线启动热身噪声
        # · 此时 T2-CACHE 早期写入已完成，此处仅更新队列建议（proc._retune_pair/result_q）
        # · 同步更新 hw_profile.t2_measured_ms 为全段最终值（供下段 _auto_queue_depths 使用）
        _RETUNE_SKIP = 3
        if self.auto_tune and len(proc._timing) > _RETUNE_SKIP:
            _stable = proc._timing[_RETUNE_SKIP:]
            _t2_post = float(np.median(_stable)) * 1000.0
            if _t2_post >= 1.0:
                _be_post = self._get_infer_backend()
                if self._hw_profile is not None:
                    self._hw_profile.t2_measured_ms = _t2_post
                    self._last_calib_config = (
                        proc.model_name, H_pad, W_pad, effective_bs, _be_post
                    )
                # 更新 T2-CACHE（以全段中位数覆盖早期估算，更稳定）
                if self.t2_cache_dir and self._cache_key:
                    _c2 = _load_t2_cache(self.t2_cache_dir)
                    _old2 = _c2.get(self._cache_key, 0.0)
                    if _old2 <= 0 or abs(_t2_post - _old2) / max(_old2, 1.0) > 0.05:
                        _c2[self._cache_key] = round(_t2_post, 1)
                        _save_t2_cache(self.t2_cache_dir, _c2)
                _pd_post, _rd_post, _ = _auto_queue_depths(
                    self._hw_profile, self.codec, self.x264_preset, self.crf,
                    H_pad, W_pad, effective_bs, len(timesteps),
                    infer_backend=_be_post, verbose=False,
                    model_name=proc.model_name,
                    t3_fps_measured=self._t3_fps_measured_input,
                )
                proc._retune_pair_q   = _pd_post
                proc._retune_result_q = _rd_post
                _dev_post = abs(_t2_post - self._t2_estimated_ms) / max(self._t2_estimated_ms, 1.0)
                print(
                    f'[AUTO-TUNE-RETUNE] 实测 T2={_t2_post:.1f}ms'
                    f'（全段 {len(_stable)} batches 中位数）| '
                    f'静态估算={self._t2_estimated_ms:.1f}ms | '
                    f'偏差={_dev_post*100:.0f}% | '
                    f'当前 result_queue={self.result_queue.maxsize} | '
                    f'校准建议 pair={_pd_post} result={_rd_post}（下次生效）',
                    flush=True,
                )

        if self._writer_th and self._writer_th.is_alive():
            # [FIX-LA-JOIN-TIMEOUT][FIX-DYN-TIMEOUT] LA>0 时段末需等待编码线程完成
            # 最后一块编码 + EOS 排空。30s 硬超时对长分段不可靠（段末收尾一旦超过
            # 30s，主线程放行后 writer.close() 会关闭 muxer stdin → 编码线程
            # write to closed file → 输出静默截断）。
            #
            # 原固定 150s 只是"看起来比 flush_and_join 内层 120s 上限宽"，但内层
            # 若真的按固定 120s 提前放弃，外层再宽的固定值也等不到真正编码完成——
            # 编码线程的自身 _th 会在后台继续跑，而 writer 线程却已经"正常退出"，
            # 段照样带着截断风险往下走。现改为按 _enc_thread.submitted_frames
            # 动态估算（与 _writer_loop 内 flush_and_join 调用处使用同一套
            # estimate_timeout 逻辑），外层在内层估算值基础上再加调度/收尾余量，
            # 确保外层等待窗口始终 ≥ 内层实际会等待的时长。
            _nv_join = getattr(self.proc, '_cached_nvenc_encoder', None)
            _la_join = getattr(_nv_join, '_la_depth', 0) if _nv_join is not None else 0
            if _la_join > 0:
                _enc_ref = getattr(self, '_enc_thread_ref', None)
                if _enc_ref is not None:
                    # 复用与内层完全一致的估算函数，外层额外加 20s 调度/收尾余量
                    # （线程唤醒、GIL 争用、flush_and_join 自身非编码开销等）。
                    _w_timeout = _enc_ref.estimate_timeout(margin=20.0)
                else:
                    # 理论上不应发生（LA>0 必走 GPU_RAW/_enc_thread 路径），
                    # 保底退回旧的固定值，不让超时估算缺失阻塞收尾判断。
                    _w_timeout = 150.0
            else:
                _w_timeout = 30.0
            self._writer_th.join(timeout=_w_timeout)
            if self._writer_th.is_alive():
                # [FIX-WRITER-TIMEOUT-FAIL] 超时仍存活 = 段失败：
                # 绝不在 writer/编码线程存活时放行继续收尾（muxer 会被提前关闭），
                # 标记错误并向上抛出，防止截断输出被误判成功写入 checkpoint。
                self._error = RuntimeError(
                    f'写线程在 {_w_timeout:.0f}s 内未退出（疑似死锁），段处理失败')
                print(f'\n[IFRNet-Writer] ❌ {self._error}', flush=True)

        # [FIX-WRITER-ERROR-PROP] writer join 之后必须复查错误：
        # _writer_loop 的异常（flush_and_join 抛出等）发生在推理线程 join 后的
        # 错误检查点之后，若不在此传播，段会被误判成功 → 截断文件带着
        # checkpoint 流入下游。
        if self._error is not None:
            raise RuntimeError(
                f'流水线处理异常: {type(self._error).__name__}: {self._error}'
            ) from self._error

        if self._reader_th and self._reader_th.is_alive():
            self._reader_th.join(timeout=10.0)

        if self._prefetch_total > 0 and not self.proc.quiet:
            hit_pct = self._prefetch_hits / self._prefetch_total * 100
            print(
                f'[IFRNet-Pipeline] 预取命中率: '
                f'{self._prefetch_hits}/{self._prefetch_total} ({hit_pct:.1f}%)',
                flush=True,
            )

        return self._fc_extra, self._oc_extra

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
        nvenc_encoder=None,  # [FIX-T3-V643]
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
                            # [DIAG-SENTINEL] pair_queue SENTINEL 丢失时的紧急退出路径
                            if self._pending_sentinel and self._prefetch_deque:
                                # 仍有预取数据待处理，消耗完再退出
                                continue
                            if self._pending_sentinel:
                                if not self.proc.quiet:
                                    print('[IFRNet-Infer] _pending_sentinel 触发，退出推理循环',
                                          flush=True)
                            break
                        continue

                if item is self._SENTINEL:
                    # 若 _prefetch_deque 仍有预取 batch，SENTINEL 放回 pair_queue
                    # 继续处理，避免 slot semaphore 泄漏 + 最后一批帧丢失
                    if self._prefetch_deque:
                        try:
                            self.pair_queue.put(item, timeout=2.0)
                        except queue.Full:
                            self._pending_sentinel = True
                        continue
                    break

                img1_raw, img0_pad, img1_pad, is_end = item
                if not img1_raw:
                    continue
                B = len(img0_pad)

                results = proc._safe_infer(
                    img0_pad, img1_pad, timesteps, H, W,
                    prefetched_img0_t=pfimg0_t,
                    prefetched_img1_t=pfimg1_t,
                    return_gpu=(nvenc_encoder is not None),  # [FIX-T3-V643-GPU] Level 1: 保持 GPU
                )

                # [FIX-RETUNE-POSTRUN] 早期 T2-CACHE 更新（仅更新缓存文件，不做队列建议）
                # 队列建议改在 run() 段完成后基于全段稳定数据统一计算，精度更高。
                if (not _calib_done and self.auto_tune
                        and len(proc._timing) >= _CALIB_SKIP + _CALIB_BATCHES):
                    # 取跳过热身后的稳定采样窗口（索引 [skip : skip+n]）
                    _samples = proc._timing[_CALIB_SKIP : _CALIB_SKIP + _CALIB_BATCHES]
                    t2_actual = float(np.median(_samples)) * 1000.0
                    if t2_actual >= 1.0:
                        _calib_done = True
                        _infer_be2 = self._get_infer_backend()
                        if self._hw_profile is not None:
                            self._hw_profile.t2_measured_ms = t2_actual
                            # [FIX-CALIB-KEY] 修复：加入 model_name，与 run() 中
                            # _current_cfg = (model_name, H_pad, W_pad, bs, be) 保持一致
                            self._last_calib_config = (
                                proc.model_name, H_pad, W_pad, effective_bs, _infer_be2
                            )
                        # T2-CACHE 早期写入：让下一个视频（非本段）尽快用上实测值
                        if self.t2_cache_dir and self._cache_key:
                            _c = _load_t2_cache(self.t2_cache_dir)
                            _old = _c.get(self._cache_key, 0.0)
                            if _old <= 0 or abs(t2_actual - _old) / max(_old, 1.0) > 0.10:
                                _c[self._cache_key] = round(t2_actual, 1)
                                _save_t2_cache(self.t2_cache_dir, _c)
                                print(f'\n[T2-CACHE] 已更新缓存 T2={t2_actual:.1f}ms '
                                      f'(key={self._cache_key})', flush=True)
                        # 队列建议不在此处输出，由 run() 完成后统一处理

                # [FIX-T3-V643] 结果分发：GPU > RING > NVENC > PinnedResultItem > fallback
                if isinstance(results, tuple) and len(results) > 0 and results[0] == 'GPU':
                    # ── Level 1: GPU-STAY 传递 GPU tensor 到 Writer 线程编码 ──
                    # [FIX-GPU-STAY] 不再在 Infer 线程做串行 NVENC 编码，
                    # 改为将 GPU tensor 传递到 result_queue，由 Writer 线程
                    # 并行做批量 RGB→NV12 + NVENC SDK 编码。
                    _, interp_gpu, img1_rgb_gpu, rB, rT, rorig_H, rorig_W = results
                    self._diag_gpu_stay_batches += 1
                    out_item = ('GPU_RAW', interp_gpu, img1_rgb_gpu, rB, rT, rorig_H, rorig_W)
                elif isinstance(results, tuple) and len(results) > 0 and results[0] == 'RING':
                    # Level 2/3: Ring Buffer 零拷贝路径
                    _, slot_id, ev, rB, rT, rorig_H, rorig_W, n_frames = results
                    proc._ring_buf.writer_commit(
                        slot_id, n_frames, rorig_H, rorig_W, rB, rT, img1_raw, ev
                    )
                    out_item = ('RING', slot_id)
                elif nvenc_encoder is not None and isinstance(results, _PinnedResultItem):
                    # [FIX-D2H-SYNC] 等待 stream_d2h 上的 D2H DMA 完成，确保 pinned buffer 数据就绪
                    # 否则第一个 batch 会因异步竞态读到全零数据 → 灰/黑帧
                    results.event.synchronize()
                    h264_frames = []
                    interp_arr = results.buf[:results.B * results.T]
                    for bi in range(results.B):
                        for tj in range(results.T):
                            rgb_gpu = interp_arr[bi * results.T + tj,
                                      :results.orig_H, :results.orig_W, :].cuda()
                            nv12_gpu = _rgb_to_nv12_gpu(rgb_gpu)
                            h264_data = nvenc_encoder.encode_frame(nv12_gpu)
                            h264_frames.append(h264_data)
                        # Encode img1_raw (original frame) through NVENC too
                        # img1_raw 来自 reader (RGB)，input_is_bgr=False 直接写 NV12
                        img1_gpu = torch.from_numpy(img1_raw[bi].copy()).cuda()
                        img1_nv12 = _rgb_to_nv12_gpu(img1_gpu, input_is_bgr=False)
                        h264_data = nvenc_encoder.encode_frame(img1_nv12)
                        h264_frames.append(h264_data)
                    out_item = (h264_frames, results.B, results.T,
                                results.orig_H, results.orig_W)
                    self._event_pool.release(results.event)
                    results.pool.release(results.buf)
                elif isinstance(results, _PinnedResultItem):
                    results.img1_raw = img1_raw
                    out_item = results
                else:
                    out_item = (results, img1_raw, is_end)

                # 非阻塞 put + 提前提交下一批预取（背压时先做预取再阻塞）
                try:
                    self.result_queue.put_nowait(out_item)
                    self._try_prefetch_next()
                except queue.Full:
                    if self._writer_th is not None and not self._writer_th.is_alive():
                        raise RuntimeError(
                            'Writer 线程已退出，result_queue 无消费者，推理线程中止'
                        )
                    # [FIX-DOUBLEBUF-H2D] 队列满时先触发预取，不让 GPU 闲着
                    self._try_prefetch_next()
                    self.result_queue.put(out_item, timeout=30.0)

                fc_extra += B
                oc_extra += B * (len(timesteps) + 1)
                self._diag_infer_batches += 1
                self._diag_infer_pairs   += B
                meter.update(B)

        except Exception as e:
            if self._error is None:
                self._error = e
            import traceback
            print(f'[IFRNet-Infer] 推理线程异常: {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
        finally:
            # [FIX-GPU-STAY] NVENC flush 已移到 Writer 线程 SENTINEL 之后执行，
            # 避免 T2 flush 与 T3 仍在进行的 GPU_RAW 编码产生竞态条件。
            writer_alive = self._writer_th is not None and self._writer_th.is_alive()
            if not writer_alive:
                if not self.proc.quiet:
                    print('[IFRNet-Infer] Writer 已退出，跳过 result_queue SENTINEL 投递', flush=True)
            else:
                for _ in range(5):
                    try:
                        self.result_queue.put(self._SENTINEL, timeout=1.0)
                        break
                    except queue.Full:
                        continue
            self.running = False
            proc._pipeline_runner = None
            # [FIX-POOL-LEAK / SEGMENT-REUSE] 跨段复用模式下，PinnedResultPool
            # 由 proc._cached_result_pool 持有（全局唯一），段间不释放。
            # 只清空段级引用，池实例存活至 cleanup() 统一销毁。
            # 无累积问题：全局只有一个池，不会像旧代码那样每段创建新池叠加。
            proc._result_pool = None
            import gc as _gc; _gc.collect()
            # [SEGMENT-REUSE] PinnedRingBuffer 由 proc._cached_ring_buf 持有，
            # Writer 线程排空队列期间仍需 _ring_buf，不清空（cleanup() 统一释放）
            self._fc_extra = fc_extra   # [FIX-INFER-THREAD] 供 run() 读取
            self._oc_extra = oc_extra   # [FIX-INFER-THREAD] 供 run() 读取

    def close(self):
        self.running = False
