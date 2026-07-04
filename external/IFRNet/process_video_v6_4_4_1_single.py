"""
IFRNet 视频插帧处理脚本 —— 四级回退统一编码版 v6.4.4.2（单卡版）
============================================================

【⚠️  LA>0 已知问题 — 已在 v6.4.5.1 中修复 ⚠️】
  此版本在 rate_mode=vbr_hq/qvbr + la_depth>0 时存在花屏和帧序错乱问题。
  根因: CE pipeline 的 NEED_MORE_INPUT 帧处理不当（6 个独立根因详见 v6.4.5.1）。
  如需 LA>0 功能，请使用 process_video_v6_4_5_1_single.py。
  LA=0 (constqp) 不受影响。

基于 IFRNet（Intermediate Flow-based Recursive Network）的视频帧插值脚本，
面向单 GPU 生产环境的极致性能实现。

【v6.4.4.2 新增修复（基于 v6.4.4.1）】
  [PHASE4-CE-PIPELINE] Backport 自 v6.4.5.1 — per-frame CE 异步流水线：
               · 新增 encode_frames_batch_ce_pipeline()：EncodePicture 附带 per-frame
                 CUDA completion event 异步提交，LockBitstream 延迟到 slot 下次轮转时
                 harvest（pipe_depth 帧后），CE 已触发→立即拿到数据，消除同步阻塞。
               · GPU 验证 (T4, 720×576): CONSTQP 612 FPS (+18%)、VBR_HQ 518 FPS (+39%)、
                 QVBR 568 FPS (+14%)，所有 RC 模式下 0% 空帧（LA=0）。
               · _NVENCEncodeThread._loop() 已切换默认使用此路径。
               · LA=0 保证与 pipe=4 兼容。

  [PHASE4-v645] 恢复多 slot 异步流水线 encode_frames_batch()：
               · v6.4.4.1 因 multi-slot 花屏问题禁用多 slot 架构；
                 根因修复（FIX-ENC-CTX + v6.4.4 synchronize）后恢复 4-slot 轮转。
               · pipeline_depth 默认 4，范围 1-8。

  [V6442-RC-ADAPTIVE] 按 RC 模式自适应 pipe/LA：
               · CONSTQP + LA=8 时保留 pipeline_depth=4（无空帧，FPS 与 LA=0 持平）。
               · VBR_HQ/QVBR + LA>0 时保守强制 pipe=1（避免 ~0.9% 空帧 + FPS 倒退）。

  [V6442-CONSTQP-FAST] CONSTQP 快速路径：
               · CONSTQP 模式下 LockBitstream 重试次数 5→2（实测 0% 空帧，跳过浪费）。
               · _NVENCEncodeThread._loop() 对 CONSTQP 跳过 Tier 1-B encode_frame 重试。

【v6.4.4 新增修复（基于 v6.4.3）】
  [FIX-ENC-CTX]  _NVENCEncodeThread CUDA context 缺失修复（cuMemcpy2D code=201 崩溃）：
               · 修复 A：在 _loop() 线程启动阶段调用 cuCtxSetCurrent(_primary)。
               · 修复 B：encode_frames_batch / encode_frame 两处 _need_pop = (r_push == 0)。

【v6.4.3 新增修复（基于 v6.3.5）】
  [FIX-T3-V643] 四级回退统一编码体系
  [FIX-GPU-STAY] NVENC SDK 直通性能修复 (Phase 1-3)
  [PHASE4] NVENC 批编码同步流水线（completionEvent async 因 ~7% 空帧率已移除）

【v6.3.5 新增修复（基于 v6.3.4）】
  [FIX-POOL-LEAK]     PinnedResultPool 跨段泄漏修复（后 3 段速度减半问题）：
                   · 根因：段结束后 PinnedResultPool 持有的 pinned 内存未显式释放，
                     Python GC 不会立即回收 cudaHostAlloc 分配的锁页内存，导致各
                     段 Pool 叠加累积（944MB → 1557MB → 1935MB → 1982MB+），
                     DMA 带宽被竞争，D2H 传输变慢，result_queue 长期满载，
                     GPU P50 利用率从 88% 跌至 8%，吞吐量减半（83fps → 41fps）。
                   · 修复：新增 PinnedResultPool.free() 方法，显式 del 全部 pinned
                     buffer；在 _infer_loop finally 块中调用，确保每段结束后立即
                     解除锁页内存占用；同时触发 gc.collect() 加速 Python 对象回收。

  [FIX-MAXRQ-DYNAMIC] result_queue 上限改为三轴动态计算（替代静态 _PINNED_POOL_MAX_MB）：
                   · 根因：_PINNED_POOL_MAX_MB 为 GPU-tier 静态常量（T4=2048MB），
                     无法反映实际 RAM 余量、分辨率变化、T3/T2 速度比等动态因素；
                     换机（RAM ≠ 32GB）或换分辨率时上限或过保守或过激进。
                   · 新增 _compute_max_result_queue(slot_mb, mem_avail_gb, T2_ms, T3_ms)
                     三轴联合约束：
                     轴1 RAM 上限  = mem_avail × 6% / slot_size（主要约束）
                     轴2 T3/T2 下限= T3_ms/T2_ms × 0.22（最小解耦需求，libx264 流式系数）
                     轴3 绝对上限  = 48（防估算失控保险）
                     结果 = max(floor_by_t3, min(cap_by_ram, 48))
                   · 替换 _auto_queue_depths / get_queue_suggestions /
                     ADAPTIVE-QUEUE 三处 _PINNED_POOL_MAX_MB 硬编码引用。
                   · _PINNED_POOL_MAX_MB 保留为模块级常量（PinnedPool 构建阶段
                     的参考值），不再用于运行时队列约束。

【v6.3.4 新增修复（基于 v6.3.3）】
  [FIX-BATCHCAP]     跨段 batch_size 误降修复（Segment 2+ bs 32→7 问题）：
                   · 根因：free_bytes = torch.cuda.mem_get_info()[0] 仅返回 OS 层面
                     真正空闲 VRAM，忽略 PyTorch allocator 已 reserved 但未 allocated
                     的可复用缓存（段结束后 TRT engine 保留 ~14 GB reserved pool）。
                   · 修复：effective_free = free_bytes + (reserved − allocated)，
                     使跨段 batch_size 估算与段内实际可分配量一致。
                   · 同时修复 _estimate_safe_batch_size() 中同一问题（OOM 恢复路径）。

  [FIX-NVENC-UNIFIED] NVENC 双检测路径统一：
                   · 根因：AUTO-TUNE 用静态 GPU 型号表（_HWProfile.has_nvenc=True），
                     HardwareCapability.best_encoder() 用 ffmpeg 实际 probe，两者结
                     果在 Docker 环境下不一致（probe 失败 → has_nvenc=False → 回退
                     libx264），导致 T3 实测 8fps、GPU 空闲 86% 的锯齿形瓶颈。
                   · 修复：best_encoder() 新增可选参数 hw_profile；当 hw_profile 提
                     供时优先信任静态表（GPU 型号已知）；仅在无 profile 时回退 probe。
                   · _run_segment() 在分辨率限制检查后缓存 hw_profile 并传给
                     best_encoder()，保证 NVENC 检测与 AUTO-TUNE 一致。

  [FIX-T2-TRT-CALIB] TRT 路径 T2 冷启动估算修正（8× 高估→精确）：
                   · 根因：_T2_FIXED_MS = 240ms 为 torch.compile/eager JIT overhead，
                     TRT 路径实测固定 overhead 仅 2-5ms，高估导致 Segment 1 初始
                     result_queue 过深（1150MB pinned），浪费锁页内存。
                   · 新增常量 _T2_FIXED_MS_TRT = 5.0ms（TRT 专用）。
                   · _auto_queue_depths() 和 pipeline.run() 中估算均依 infer_backend
                     分支选择对应固定 overhead 常量。
                   · 修复后 Segment 1 初始 pool 估算从 ~1150MB 降至 ~150MB。

  [FIX-POOL-AUTOSCALE] PinnedPool 上限依 GPU 型号自动缩放（替代硬编码 1024MB）：
                   · 根因：1024MB 对 T4（bs=32 时单 slot ~110MB × 10=1100MB 即超限）
                     过于保守，对 A100/H100（余量充裕）过于宽松。
                   · 新增 _pool_limit_mb_for_profile(profile) 函数，按 gpu_tier 分 6
                     档：GTX 1080=1024  T4/RTX2080=2048  RTX3090/4070=3072
                     A10/L40S/RTX4080+=4096  A100/A800=6144  H100/H800=8192（MiB）。
                   · 兼顾系统可用 RAM：上限不超过 MemAvailable × 12%（最低 1024MB）。
                   · get_queue_suggestions() 新增 hw_profile 参数接收动态上限；
                     _auto_queue_depths() 直接用 profile 计算上限。

【v6.3.3 新增修复（基于 v6.3.2）】
  [FIX-RETUNE-POSTRUN]  AUTO-TUNE-RETUNE 计算时机改为段完成后：
                   · 原实现在 _infer_loop 内 timing[3:8] 共 5 个 batch 做中位数，
                     约占全段 1% 的数据，受流水线启动状态影响较大。
                   · 新实现在 run() 中 _infer_th.join() 后，用 timing[3:] 全段
                     稳定 batch 的中位数（通常 100+ 样本），精度显著提升。
                   · T2-CACHE 文件更新策略不变（早期写入保留，段完成后再精校）。
                   · 与 GPU-MONITOR 统一在段完成后计算，日志顺序更直观。

  [FIX-MEMCAP-LOG]      PinnedPool 内存上限截断时输出显式 log：
                   · 原实现在 ADAPTIVE-QUEUE 合并建议时若 PinnedPool 动态上限
                     将建议值静默截断，用户无法得知"GPU-MONITOR 建议 23 但实际用 15"
                     的原因。
                   · 新增截断时打印：
                     [ADAPTIVE-QUEUE] PinnedPool 内存上限截断: result_queue 19 → 15
                      (slot=58.9 MB × 17 ≈ 动态上限 MB)

  [FIX-RETUNE-DISPLAY]  ADAPTIVE-QUEUE 综合建议打印推导链路：
                   · 原实现只打印最终值，无法追溯 GPU-MONITOR / RETUNE 各自贡献。
                   · 新增来源注释：
                     [ADAPTIVE-QUEUE] 下次将使用 pair_queue=8 result_queue=15
                       (GPU-MONITOR=23 RETUNE=15 → avg=19)

  [FIX-SLICE-THREAD] FFmpegWriter 软编码并行升级：自动探测 CPU 逻辑核心数、物理核心
                   数和系统可用内存，为 libx264/libx265 自动注入最优线程和分片参数。
                   · libx264: -threads N + -x264-params threads=N:slices=S，启用
                     intra-frame slice-based threading（单帧分 S 片并行编码），在
                     pipe 流式输入场景中比 frame-parallel 延迟更低、吞吐更高。
                   · libx265: -x265-params pools=N:frame-threads=F，替换旧
                     pools=none（完全禁用线程池）为正确的多线程配置。
                   · 新增 _detect_encode_parallelism() 函数：读取 /proc/cpuinfo
                     获取物理核心数，读取 /proc/meminfo 获取可用内存，综合计算
                     encode_threads / slices / ffmpeg_threads 三项参数。
                   · FFmpegWriter 新增 n_threads 参数（None=自动探测），兼容旧接口。

  [FIX-NVENC-PIPE]   FFmpegWriter NVENC pipe 模式优化：针对 pipe 流式输入场景补全四项
                   NVENC 硬件编码参数，显著改善编码吞吐与码率控制质量。
                   背景：NVENC 是 GPU 固定功能硬件单元，完全不受 FFmpeg CPU 线程数
                   控制（-threads 仅作用于 demux/filter graph），但以下 NVENC 自身
                   参数在 pipe 场景下对吞吐和质量有显著影响：
                   · -bf 0         禁用 B 帧，消除 NVENC 流水线多帧缓冲延迟
                                   （B 帧需双向参考，持有前后帧缓冲后再输出），
                                   两种模式（无损/VBR）均启用。
                   · -surfaces 32  扩大 NVENC 内部帧缓冲（默认 8 → 32），防止 pipe
                                   输入速率不均时编码器因缓冲不足频繁暂停（饥饿停顿）。
                                   两种模式均启用，值由 _NVENC_SURFACES_PIPE 常量控制。
                   · -delay 0      零延迟输出模式（仅 crf=0 无损/QP=0 路径）：与 -bf 0
                                   协同，使每帧编完即输出，最小化 pipe 端到端延迟。
                                   与 -rc-lookahead 互斥，故仅在无需前瞻的 QP 模式启用。
                   · -rc-lookahead 16  前向帧预看（仅 crf>0 VBR 模式）：编码器向前看
                                   16 帧进行码率分配，在场景切换和高运动区域有效降低
                                   码率浪费、提升 PSNR/SSIM。因需要前瞻缓冲与 -delay 0
                                   互斥，值由 _NVENC_LOOKAHEAD_VBR 常量控制。
                   新增 _NVENC_SURFACES_PIPE / _NVENC_LOOKAHEAD_VBR 模块级常量，
                   便于调优时统一修改。同步补全 NVENC 路径的 [FFmpegWriter] 参数摘要日志。

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
                   （当前已由 FIX-POOL-AUTOSCALE 按 GPU 型号自动缩放）；result_queue 不再无限制增大，
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
        # [FIX-PROBE] 历次修复汇总（根因均由 stderr 诊断确认）：
        #   LAVFI  : color 源去掉 d=0.1（0帧问题）→ 无限源 + -frames:v 1 截帧
        #   PIXFMT : FFmpeg 5.0+ color 源默认 yuv444p → 显式 -pix_fmt yuv420p
        #   BFRAME : 单帧编码触发 B 帧 lookahead 报错 → -bf 0
        #   MINDIM : h264_nvenc 宽≥145px / hevc_nvenc 宽≥129px → 256×144
        cmd = [
            'ffmpeg', '-hide_banner', '-y', '-loglevel', 'error',
            '-f', 'lavfi', '-i', 'color=c=black:s=256x144:r=1',
            '-vcodec', codec, '-frames:v', '1',
            '-pix_fmt', 'yuv420p', '-bf', '0',
            '-f', 'null', '-',
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode != 0:
                _err = result.stderr.decode('utf-8', errors='replace').strip()
                print(
                    f'  [PROBE-FAIL] {codec} probe 失败 (rc={result.returncode})\n'
                    f'  手动测试: {" ".join(cmd)}\n'
                    f'  stderr: {_err or "(空)"}',
                    flush=True,
                )
            return result.returncode == 0
        except Exception as e:
            print(
                f'  [PROBE-FAIL] {codec} probe 异常: {e}\n'
                f'  手动测试: {" ".join(cmd)}',
                flush=True,
            )
            return False

    @classmethod
    def best_encoder(cls, preferred: str = 'libx264',
                     hw_profile: Optional['_HWProfile'] = None) -> str:
        """
        [FIX-NVENC-UNIFIED] 统一 NVENC 检测路径。

        优先级（按 hw_profile 是否提供分两条路径）：

        路径 A（hw_profile 已提供，GPU 型号已知）：
          1. 直接采信 hw_profile.has_nvenc（静态 GPU 型号表，与 AUTO-TUNE 一致）。
          2. 若静态表确认可用且 ffmpeg probe 失败（Docker 设备映射缺失常见场景），
             打印明确警告后仍信任静态表——probe 失败不等于硬件不可用。
             可将 _NVENC_TRUST_STATIC = False 改为强制要求 probe 通过。
          3. 静态表否定（has_nvenc=False）时不做 probe，直接回退软件编码。

        路径 B（hw_profile 未提供，GPU 型号未知）：
          仅凭 ffmpeg 实际 probe 判断，probe 失败即回退软件编码。
          此路径适用于无法识别 GPU 型号的环境（非 NVIDIA GPU 等）。

        两套检测结果在 Docker 未映射 NVENC 设备时会不一致：AUTO-TUNE 显示 nvenc=True
        但 ffmpeg probe 失败，导致实际使用 libx264 引发 T3 瓶颈。本修复确保两者一致。
        """
        nvenc_map    = {'libx264': 'h264_nvenc', 'libx265': 'hevc_nvenc'}
        fallback_map = {'h264_nvenc': 'libx264', 'hevc_nvenc': 'libx265'}

        # [FIX-NVENC-TRUST-STATIC] 当 probe 失败但静态表确认可用时，是否信任静态表。
        # True（默认）：信任静态表，适用于 Docker 设备映射缺失但硬件真实存在的场景。
        # False：强制要求 probe 通过，适用于需要 100% 运行时验证的严格环境。
        _NVENC_TRUST_STATIC: bool = True

        def _nvenc_ok(codec_name: str) -> bool:
            # [FIX-NVENC-UNIFIED] 路径 A：hw_profile 已知时优先信任静态 GPU 型号表。
            # 这与 docstring 描述的优先级一致，解决了原实现 probe-first 导致
            # Docker 环境中 hw_profile.has_nvenc=True 被静默忽略的问题。
            if hw_profile is not None and hasattr(hw_profile, 'has_nvenc'):
                if not hw_profile.has_nvenc:
                    # 静态表明确否定：不做 probe，直接返回 False
                    return False
                # 静态表确认可用：尝试 probe 做二次验证
                probe_ok = cls.has_nvenc(codec_name)
                if not probe_ok:
                    # [FIX-NVENC-PROBE-WARN] probe 失败但静态表确认硬件存在。
                    # 常见于 Docker 容器 /dev/nvidia* 设备映射不完整，
                    # nvidia-smi 可见 GPU 但 ffmpeg 无法访问 NVENC 编码器。
                    # 根据 _NVENC_TRUST_STATIC 决定是否信任静态表。
                    if _NVENC_TRUST_STATIC:
                        print(
                            f'  [FIX-NVENC-UNIFIED] {codec_name} ffmpeg probe 失败，'
                            f'但静态 GPU 型号表确认硬件存在（hw_profile.has_nvenc=True）。\n'
                            f'  信任静态表，继续使用 {codec_name}（Docker 设备映射不完整时正常）。\n'
                            f'  若实际编码报错，可在代码中将 _NVENC_TRUST_STATIC 改为 False。'
                        )
                        return True
                    else:
                        print(
                            f'  [FIX-NVENC-UNIFIED] {codec_name} ffmpeg probe 失败。'
                            f'静态表显示硬件存在但 _NVENC_TRUST_STATIC=False，回退软件编码。'
                        )
                        return False
                return True  # 静态表 + probe 双重确认

            # [FIX-NVENC-UNIFIED] 路径 B：hw_profile 未知，仅凭 ffmpeg probe 判断。
            return cls.has_nvenc(codec_name)

        if preferred in fallback_map:
            if _nvenc_ok(preferred):
                return preferred
            fallback = fallback_map[preferred]
            # [FIX-NVENC-WARN] 明确说明回退原因（probe 失败 or 静态表否定 or 无 profile）
            if hw_profile is not None and hasattr(hw_profile, 'has_nvenc') and not hw_profile.has_nvenc:
                reason = '静态 GPU 型号表标记 has_nvenc=False'
            elif not cls.has_nvenc(preferred):
                reason = 'ffmpeg probe 失败（Docker 设备映射可能缺失 /dev/nvidia*）'
            else:
                reason = '未知原因'
            print(f'  [警告] {preferred} 不可用（{reason}），自动回退到 {fallback}')
            return fallback
        candidate = nvenc_map.get(preferred, preferred)
        if candidate != preferred and _nvenc_ok(candidate):
            return candidate
        return preferred

    @classmethod
    def lossless_encoder(cls) -> Tuple[str, List[str]]:
        """
        返回 (codec, extra_args) 用于无损中间段编码。
        优先 nvenc lossless（-rc constqp -qp 0），否则 libx264 lossless。

        与 best_encoder 的区别：best_encoder 只探测 NVENC 是否可用（VBR 编码），
        lossless_encoder 额外测试 constqp（常量 QP）模式——部分 GPU/driver 组合
        虽支持 NVENC VBR 但不支持 constqp 无损。
        """
        if cls.has_nvenc('h264_nvenc'):
            try:
                cmd = [
                    'ffmpeg', '-hide_banner', '-y', '-loglevel', 'error',
                    '-f', 'lavfi', '-i', 'color=c=black:s=256x144:r=1',
                    '-vcodec', 'h264_nvenc',
                    '-rc', 'constqp', '-qp', '0',
                    '-pix_fmt', 'yuv420p', '-bf', '0',
                    '-frames:v', '1', '-f', 'null', '-',
                ]
                if subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0:
                    return 'h264_nvenc', ['-rc', 'constqp', '-qp', '0']
            except Exception:
                pass
        return 'libx264', ['-qp', '0', '-preset', 'medium']


# ─────────────────────────────────────────────────────────────────────────────
# [FIX-T3-V643] NVENC GPU 直通编码器
# ─────────────────────────────────────────────────────────────────────────────

import ctypes
from ctypes import (c_uint8, c_uint16, c_uint32, c_int32, c_int, c_uint64, c_void_p,
                     c_char, c_size_t, c_double, Structure, POINTER, byref,
                     sizeof, cast, pointer, c_bool)
import os
import threading

# ==============================================================================
# NVENC GUID 结构体 (16 bytes)
# ==============================================================================

class _NvGuid(Structure):
    _pack_ = 1
    _fields_ = [
        ("Data1", c_uint32),
        ("Data2", c_uint16),
        ("Data3", c_uint16),
        ("Data4", c_uint8 * 8),
    ]

    def __eq__(self, other):
        if not isinstance(other, _NvGuid):
            return False
        return (self.Data1 == other.Data1 and self.Data2 == other.Data2
                and self.Data3 == other.Data3 and bytes(self.Data4) == bytes(other.Data4))

    def __hash__(self):
        return hash((self.Data1, self.Data2, self.Data3, bytes(self.Data4)))

# ==============================================================================
# NVENC 预定义 GUID 常量
# ==============================================================================

NV_ENC_CODEC_H264_GUID = _NvGuid(0x6b9c211b, 0x3fdd, 0x4a5a,
    (0x8d, 0x2e, 0x05, 0x0a, 0xbb, 0xb9, 0x1c, 0x6a))

NV_ENC_PRESET_P1_GUID = _NvGuid(0xfc9a8d6c, 0xa4e8, 0x4f03,
    (0xaa, 0xce, 0x91, 0x97, 0x6a, 0xc2, 0x74, 0x10))
NV_ENC_PRESET_P2_GUID = _NvGuid(0x711784c6, 0x34a6, 0x47e0,
    (0xaa, 0x06, 0x60, 0x9c, 0x72, 0xdb, 0x0c, 0x8a))
NV_ENC_PRESET_P3_GUID = _NvGuid(0xc678451b, 0x1b4f, 0x4b64,
    (0xbc, 0x03, 0x10, 0xbb, 0x22, 0xc3, 0x54, 0x63))
NV_ENC_PRESET_P4_GUID = _NvGuid(0xec59fb72, 0x14fb, 0x4e28,
    (0xbc, 0x04, 0xa1, 0x59, 0xd2, 0x6c, 0x01, 0xc5))
NV_ENC_PRESET_P5_GUID = _NvGuid(0xb0b9e4da, 0xb52e, 0x4edb,
    (0xa1, 0xcb, 0xd4, 0x70, 0x36, 0x48, 0x32, 0x95))
NV_ENC_PRESET_P6_GUID = _NvGuid(0x74c1e37a, 0x4b74, 0x4905,
    (0xb0, 0xe9, 0x00, 0x61, 0x15, 0x53, 0xb6, 0x3a))
NV_ENC_PRESET_P7_GUID = _NvGuid(0x5b7e7d04, 0xb7df, 0x4488,
    (0x82, 0x4e, 0x55, 0xbf, 0x41, 0x8a, 0x18, 0x29))

NV_ENC_H264_PROFILE_HIGH_GUID = _NvGuid(0x1a1a5a20, 0xf787, 0x4e5b,
    (0x9a, 0xab, 0x58, 0x76, 0xfa, 0x7a, 0xdc, 0x0f))

_PRESET_GUID_MAP = {
    "p1": NV_ENC_PRESET_P1_GUID, "p2": NV_ENC_PRESET_P2_GUID,
    "p3": NV_ENC_PRESET_P3_GUID, "p4": NV_ENC_PRESET_P4_GUID,
    "p5": NV_ENC_PRESET_P5_GUID, "p6": NV_ENC_PRESET_P6_GUID,
    "p7": NV_ENC_PRESET_P7_GUID,
}

# ==============================================================================
# NVENC API 常量
# ==============================================================================

# [FIX-NVENC-SDK13] SDK 13.0 struct version formula (verified against nv-codec-headers SDK 13.0.0):
#   NVENCAPI_VERSION = NVENCAPI_MAJOR | (NVENCAPI_MINOR << 24) = 13 | (0 << 24) = 0x0d
#   NVENCAPI_STRUCT_VERSION(ver) = NVENCAPI_VERSION | (ver << 16) | (0x7 << 28) | (0x80000000 if bit31 else 0)
#   低字节是 NVENCAPI_VERSION (0x0d)，不是 sizeof(struct)。这是 SDK 12+ 的正确公式。
_NVENCAPI_VERSION_FALLBACK = (13 << 4) | 0  # 13.0 = 0xD0

def NVENCAPI_STRUCT_VERSION(struct_or_size, api_ver=None):
    """保留兼容旧代码。"""
    if api_ver is None:
        api_ver = _NVENCAPI_VERSION_FALLBACK
    if isinstance(struct_or_size, int):
        size = struct_or_size
    else:
        size = sizeof(struct_or_size)
    return size | (api_ver << 16) | (0x7 << 28)

# SDK 13.0 struct version helper: NVENCAPI_VERSION | (ver << 16) | (0x7 << 28) | (bit31 ? 1<<31 : 0)
# NVENCAPI_VERSION = NVENCAPI_MAJOR | (NVENCAPI_MINOR << 24) = 13 | (0 << 24) = 0x0d
# This is the CORRECT formula per nv-codec-headers SDK 13.0, NOT sizeof-based.
_NVENCAPI_VERSION = 0x0d  # SDK 13.0

def _sdk13_ver(ver, bit31=False):
    v = _NVENCAPI_VERSION | (ver << 16) | (0x7 << 28)
    if bit31:
        v |= (1 << 31)
    return v
NV_ENC_CREATE_INPUT_BUFFER_VER = _sdk13_ver(2)           # NVENCAPI_STRUCT_VERSION(2) = 0x7002000d
NV_ENC_LOCK_INPUT_BUFFER_VER = _sdk13_ver(1)             # NVENCAPI_STRUCT_VERSION(1) = 0x7001000d
NV_ENC_CREATE_BITSTREAM_BUFFER_VER = _sdk13_ver(1)       # NVENCAPI_STRUCT_VERSION(1) = 0x7001000d

# func_table = 64 指针槽, sizeof(c_void_p) * 64 = 512 (64-bit)
_FUNC_TABLE_SIZE = sizeof(c_void_p) * 64

# 静态结构体版本常量 — 依赖 struct 定义，在 struct 定义完成后赋值 (#NVENC_STRUCT_VERS)
NV_ENC_PRESET_CONFIG_VER     = 0  # placeholder
NV_ENC_CONFIG_VER            = 0
NV_ENC_INITIALIZE_PARAMS_VER = 0
NV_ENC_PIC_PARAMS_VER        = 0
NV_ENC_LOCK_BITSTREAM_VER    = 0
NV_ENC_REGISTER_RESOURCE_VER = 0
NV_ENC_MAP_INPUT_RESOURCE_VER = 0

NV_ENC_SUCCESS = 0
NV_ENC_ERR_NEED_MORE_INPUT = 17  # lookahead: encoder needs more frames before producing output

# NV_ENC_DEVICE_TYPE enum per nv-codec-headers SDK 13.0:
#   DIRECTX=0, CUDA=1, OPENGL=2
NV_ENC_DEVICE_TYPE_CUDA = 1
NV_ENC_BUFFER_FORMAT_NV12 = 1
NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR = 2
NV_ENC_INPUT_IMAGE = 0
NV_ENC_PIC_STRUCT_FRAME = 1

# ==============================================================================
# NVENC API 结构体定义
# ==============================================================================

class _NvEncOpenEncodeSessionExParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",    c_uint32),
        ("deviceType", c_uint32),
        ("device",     c_void_p),
        ("reserved",   c_void_p),
        ("apiVersion", c_uint32),
        ("reserved1",  c_uint8 * 253 * 4),
        ("reserved2",  c_void_p * 64),
    ]

class _NvEncInitializeParams(Structure):
    """SDK 13.0 NV_ENC_INITIALIZE_PARAMS — sizeof=1800 bytes (C dump verified)"""
    _pack_ = 1
    _fields_ = [
        ("version",                 c_uint32),        # offset 0
        ("encodeGUID",              _NvGuid),         # offset 4 (16 bytes)
        ("presetGUID",              _NvGuid),         # offset 20 (16 bytes)
        ("encodeWidth",             c_uint32),        # offset 36
        ("encodeHeight",            c_uint32),        # offset 40
        ("darWidth",                c_uint32),        # offset 44
        ("darHeight",               c_uint32),        # offset 48
        ("frameRateNum",            c_uint32),        # offset 52
        ("frameRateDen",            c_uint32),        # offset 56
        ("enableEncodeAsync",       c_uint32),        # offset 60
        ("enablePTD",               c_uint32),        # offset 64
        ("bitfield",                c_uint32),        # offset 68
        ("privDataSize",            c_uint32),        # offset 72
        ("reserved_76",             c_uint32),        # offset 76
        ("privData",                c_void_p),        # offset 80
        ("encodeConfig",            c_void_p),        # offset 88 ★
        ("maxEncodeWidth",          c_uint32),        # offset 96
        ("maxEncodeHeight",         c_uint32),        # offset 100
        ("maxMEHintCountsPerBlock", c_uint8 * 32),    # offset 104
        ("tuningInfo",              c_uint32),        # offset 136
        ("bufferFormat",            c_uint32),        # offset 140
        ("numStateBuffers",         c_uint32),        # offset 144
        ("outputStatsLevel",        c_uint32),        # offset 148
        ("reserved1",               c_uint8 * 1136),  # offset 152 (284×4)
        ("reserved2",               c_void_p * 64),   # offset 1288 (64×8=512)
    ]

class _NvEncConfigH264VUIParameters(Structure):
    _pack_ = 1
    _fields_ = [
        ("overscanInfoPresentFlag", c_uint32),
        ("videoSignalTypePresentFlag", c_uint32),
        ("videoFormat", c_uint32),
        ("videoFullRangeFlag", c_uint32),
        ("colourDescriptionPresentFlag", c_uint32),
        ("colourPrimaries", c_uint32),
        ("transferCharacteristics", c_uint32),
        ("matrixCoefficients", c_uint32),
        ("chromaSampleLocationFlag", c_uint32),
        ("chromaSampleLocationTop", c_uint32),
        ("chromaSampleLocationBottom", c_uint32),
        ("bitstreamRestrictionFlag", c_uint32),
        ("reserved", c_uint32 * 16),
    ]

class _NvEncConfigH264(Structure):
    _pack_ = 1
    _fields_ = [
        ("enableTemporalSVC",         c_uint32),
        ("enableTemporalSVC_1",       c_uint32),
        ("profileLevel",              c_uint32),
        ("chromaFormatIDC",           c_uint32),   # [FIX-CHROMA] 1=4:2:0, 0=monochrome — was buried in reserved1
        ("reserved1",                 c_uint32 * 13),
        ("maxNumRefFramesInDPB",      c_uint32),
        ("reserved2",                 c_uint32 * 3),
        ("idrPeriod",                 c_uint32),
        ("repeatSPSPPS",              c_uint32),
        ("reserved10",                c_uint32 * 4),
        ("vuiParameters",             _NvEncConfigH264VUIParameters),
        ("reserved12",                c_uint32 * 222),
    ]

class _NvEncConfigHevc(Structure):
    _pack_ = 1
    _fields_ = [("reserved", c_uint32 * 288)]

class _NvEncConfigH264MeOnly(Structure):
    _pack_ = 1
    _fields_ = [("reserved", c_uint32 * 248)]

class _NvEncConfigHevcMeOnly(Structure):
    _pack_ = 1
    _fields_ = [("reserved", c_uint32 * 248)]

class _NvEncConfig(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",         c_uint32),
        ("profileGUID",     _NvGuid),
        ("gopLength",       c_uint32),
        ("frameIntervalP",  c_uint32),
        ("frameFieldMode",  c_uint32),
        ("enablePTD",       c_uint32),
        ("frameFieldMode_1",c_uint32),
        ("reserved3",       c_uint32 * 53),
        ("mvPrecision",     c_uint32),
        ("reserved4",       c_uint32 * 27),
        ("reserved5",       c_uint32 * 172),  # [V6441] absorb mis-mapped enableTemporalAQ (=rcParams bitfield bit7, not a standalone NV_ENC_CONFIG field)
        ("encodeCodecConfig", _NvEncConfigH264),
        ("reserved7",      c_uint32 * 252),
    ]

class _NvEncPresetConfig(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",       c_uint32),
        ("presetConfig",  _NvEncConfig),
        ("reserved",      c_uint32 * 256),
    ]

class _NvEncRegisterResource(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",          c_uint32),
        ("resourceType",     c_uint32),
        ("width",            c_uint32),
        ("height",           c_uint32),
        ("pitch",            c_uint32),
        ("subResourceIndex", c_uint32),
        ("bufferFormat",     c_uint32),
        ("bufferUsage",      c_uint32),
        ("pInputFencePoint", c_void_p),
        ("pOutputFencePoint",c_void_p),
        ("reserved",         c_uint32 * 248),
        ("registeredResource", c_void_p),
    ]

class _NvEncMapInputResource(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",            c_uint32),
        ("subResourceIndex",   c_uint32),
        ("reserved",           c_uint32 * 62),
        ("registeredResource", c_void_p),
        ("mappedResource",     c_void_p),
        ("reserved1",          c_uint32 * 62),
    ]

class _NvEncPicParamsH264(Structure):
    _pack_ = 1
    _fields_ = [
        ("reserved",        c_uint32 * 4),
        ("refFrameFlag",    c_uint32),
        ("reserved1",       c_uint32 * 257),
    ]

class _NvEncPicParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",           c_uint32),
        ("inputWidth",        c_uint32),
        ("inputHeight",       c_uint32),
        ("inputPitch",        c_uint32),
        ("inputBuffer",       c_void_p),
        ("inputTimeStamp",    c_uint64),
        ("pictureStruct",     c_uint32),
        ("encodePicFlags",    c_uint32),
        ("frameIdx",          c_uint32),
        ("inputFencePoint",   c_void_p),
        ("outputFencePoint",  c_void_p),
        ("inputDuration",     c_uint64),
        ("reserved",          c_uint32 * 8),
        ("codecPicParams",    _NvEncPicParamsH264),
        ("reserved1",         c_uint32 * 272),
    ]

class _NvEncLockBitstream(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",           c_uint32),
        ("doNotWait",         c_uint32),
        ("reserved",          c_uint32 * 30),
        ("outputBitstream",   c_void_p),
        ("sliceOffsets",      c_uint32 * 16),
        ("reserved1",         c_uint32 * 246),
        ("bitstreamSizeInBytes",c_uint32),
        ("bitstreamBufferPtr", c_void_p),
        ("reserved2",         c_uint32 * 174),
    ]

# ── #NVENC_STRUCT_VERS: SDK 13.0 version constants (NVENCAPI_STRUCT_VERSION based) ──
NV_ENC_PRESET_CONFIG_VER     = _sdk13_ver(5, True)   # 0xf005000d
NV_ENC_CONFIG_VER            = _sdk13_ver(9, True)   # 0xf009000d
NV_ENC_INITIALIZE_PARAMS_VER = _sdk13_ver(7, True)   # 0xf007000d
NV_ENC_PIC_PARAMS_VER        = _sdk13_ver(7, True)   # 0xf007000d
NV_ENC_LOCK_BITSTREAM_VER    = _sdk13_ver(2, True)   # 0xf002000d
NV_ENC_REGISTER_RESOURCE_VER = _sdk13_ver(5)          # 0x7005000d
NV_ENC_MAP_INPUT_RESOURCE_VER = _sdk13_ver(4)          # 0x7004000d

# ==============================================================================
# ctypes 函数原型定义
# ==============================================================================

_NvEncodeAPICreateInstanceProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p,
)

_NvEncOpenEncodeSessionExProto = ctypes.CFUNCTYPE(
    c_uint32, ctypes.POINTER(_NvEncOpenEncodeSessionExParams), ctypes.POINTER(c_void_p),
)
_NvEncGetEncodePresetConfigProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, _NvGuid, _NvGuid, ctypes.POINTER(_NvEncPresetConfig),
)
# [FIX-NVENC-VER-D] nvEncInitializeEncoder(void* encoder, NV_ENC_INITIALIZE_PARAMS*)
# 原参数顺序错误（params在前），修正为 encoder handle 在第一位
_NvEncCreateEncoderProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, ctypes.POINTER(_NvEncInitializeParams),
)
_NvEncDestroyEncoderProto = ctypes.CFUNCTYPE(c_uint32, c_void_p)
_NvEncRegisterResourceProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, ctypes.POINTER(_NvEncRegisterResource),
)
_NvEncUnregisterResourceProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, c_void_p,
)
_NvEncMapInputResourceProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, ctypes.POINTER(_NvEncMapInputResource),
)
_NvEncUnmapInputResourceProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, c_void_p,
)
_NvEncEncodePictureProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, ctypes.POINTER(_NvEncPicParams),
)
_NvEncLockBitstreamProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, ctypes.POINTER(_NvEncLockBitstream),
)
_NvEncUnlockBitstreamProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, c_void_p,
)

# [FIX-NVENC-SDK13] SDK 13.0 function table indices — verified via nv-codec-headers n13.0.19.0 C dump.
# Index = (offsetof(field) - offsetof(nvEncOpenEncodeSession)) / sizeof(void*)
# SDK 13.0 将 OpenEncodeSessionEx 从 index 24 → 29 (新增 5 个 reserved 槽)
_FUNC_IDX = {
    "GetEncodeGUIDCount":        1,   # nvEncGetEncodeGUIDCount
    "GetEncodeGUIDs":            4,   # nvEncGetEncodeGUIDs
    "GetEncodePresetGUIDs":      9,   # nvEncGetEncodePresetGUIDs
    "GetEncodePresetConfig":    10,   # nvEncGetEncodePresetConfig
    "InitializeEncoder":        11,   # nvEncInitializeEncoder (旧名 nvEncCreateEncoder)
    "CreateInputBuffer":        12,   # nvEncCreateInputBuffer
    "DestroyInputBuffer":       13,   # nvEncDestroyInputBuffer
    "CreateBitstreamBuffer":    14,   # nvEncCreateBitstreamBuffer
    "DestroyBitstreamBuffer":   15,   # nvEncDestroyBitstreamBuffer
    "EncodePicture":            16,   # nvEncEncodePicture
    "LockBitstream":            17,   # nvEncLockBitstream
    "UnlockBitstream":          18,   # nvEncUnlockBitstream
    "LockInputBuffer":          19,   # nvEncLockInputBuffer
    "UnlockInputBuffer":        20,   # nvEncUnlockInputBuffer
    "MapInputResource":         25,   # nvEncMapInputResource
    "UnmapInputResource":       26,   # nvEncUnmapInputResource
    "DestroyEncoder":           27,   # nvEncDestroyEncoder
    "OpenEncodeSessionEx":      29,   # nvEncOpenEncodeSessionEx (SDK 13.0)
    "RegisterResource":         30,   # nvEncRegisterResource
    "UnregisterResource":       31,   # nvEncUnregisterResource
    "GetEncodePresetConfigEx":  39,   # nvEncGetEncodePresetConfigEx (SDK 13.0)
}

# ── _NVENC_VBR_QUALITY_OFFSET ──
# [已废弃] 旧版 CRF→targetQuality 偏移公式的偏移值。
# 新公式（2026-06-09）：targetQuality = max(1, 51 - CRF)
#   · CRF=0 (lossless/best) → tq=51 (max quality)
#   · CRF=51 (worst)        → tq=1  (min quality)
#   · CRF=18 → tq=33, CRF=28 → tq=23 （差值 10，应明显区分文件大小）
# 不再使用此常量，保留作为历史参考。
# 历史值：7（CRF-offset 公式，文件仍偏大）、21（加法公式，文件极大 36.9 MB）、15（CRF-offset 公式）
_NVENC_VBR_QUALITY_OFFSET: int = 15

# ── _PRESET_P_INDEX ──
# x264 preset name → NVENC preset array index (p-index)
# NVENC driver returns presets in order: p1 (fastest) → p7 (slowest)
# p1=0, p2=1, ..., p7=6
# Also supports direct p1-p7 strings as preset names
_PRESET_P_INDEX: dict = {
    "ultrafast": 0, "superfast": 0,
    "veryfast": 1, "faster": 2,
    "fast": 3, "medium": 4,
    "slow": 5, "slower": 6, "veryslow": 6, "placebo": 6,
}

# ==============================================================================
# NVENCEncoder
# ==============================================================================

class NVENCEncoder:
    """GPU direct H.264 hardware encoder via NVENC SDK (ctypes) — SDK 13.0 verified.

    Input: GPU tensor (NV12 format, uint8, H_total x W, contiguous)
    Output: H.264 Elementary Stream bytes

    [FIX-NVENC-SDK13] Complete rewrite based on verified test_nvenc_pre_torch.py:
      - CreateInputBuffer + LockInputBuffer + cuMemcpyDtoD_v2 (no RegisterResource)
      - CreateBitstreamBuffer for encoder output
      - All structs via byte array + manual offset writes (verified offsets)
      - Dynamic GUID query from driver (GetEncodeGUIDs / GetEncodePresetGUIDs)
      - Primary context management (cuDevicePrimaryCtxRetain + cuCtxPushCurrent)
    """

    def __init__(self, width: int, height: int, fps: float,
                 preset: str = "p1", qp: int = 0,
                 codec: str = "h264", pipeline_depth: int = 4,
                 rate_mode: str = "constqp", la_depth: int = 0):
        """rate_mode: 'constqp' | 'vbr_hq' (CQ via VBR_HQ + targetQuality) | 'qvbr'.
           la_depth: lookahead depth (0=disabled, 8~32 for -rc-lookahead equivalent).
           pipeline_depth: NVENC multi-slot pipeline depth (1-8).
           将自动校准为 >= LA+1 (SDK 硬件安全要求)。"""
        if codec != "h264":
            raise ValueError("NVENCEncoder: only H.264 supported, got: " + codec)

        self._width = width
        self._height = height
        self._fps = fps
        self._qp = qp
        self._rate_mode = rate_mode
        self._la_depth = la_depth

        # NVENC SDK 规范: pipeline_depth >= LA+1 (硬件安全要求, 防止缓冲区覆盖)
        # NVIDIA SDK 声明: "The total number of output frames equals the number of input frames"
        # CONSTQP 下硬件静默禁用 LA，代码层面清零以与硬件行为一致
        # VBR_HQ/QVBR 需确保足够 slot 数以容纳 LA 缓冲区
        if rate_mode == "constqp":
            la_depth = 0  # 硬件静默禁用 LA，此处显式清零
        # pipeline_depth 自动校准为 >= LA+1 (SDK 硬件安全要求)
        _required_buffers = max(1, la_depth + 1)
        pipeline_depth = max(pipeline_depth, _required_buffers)

        print("[NVENCEncoder] %s + LA=%d: %d slots (HW pipeline buffers>=%d)" %
              (rate_mode.upper(), la_depth, pipeline_depth, _required_buffers), flush=True)

        self._preset_name = preset.lower()
        self._encoder = c_void_p(None)
        self._frame_idx = 0
        # [FIX-LA-OUTPTR] 独立输出槽位指针：跟踪下一个预期输出的 slot，
        # 确保 LA 延迟产出帧按正确顺序取回（参照 test_nvenc_la_frame_conservation.py）
        self._output_slot_idx = 0
        self._eos_sent = False  # [FIX-LA-EOS] 防重复送 EOS
        self._lock = threading.Lock()

        # 多 slot 缓冲池: _slot_count >= _required_buffers (SDK 硬件安全要求)
        # _pipeline_depth 为向后兼容别名
        self._slot_count = pipeline_depth
        self._pipeline_depth = pipeline_depth
        self._slots: list = []

        # Backward compat: legacy refs (initialized after slot creation)
        self._input_buf_handle = c_void_p(None)
        self._bs_handle = c_void_p(None)

        # [SEGMENT-REUSE] 缓存首段 SPS+PPS NAL 单元，后续段预挂到首帧前
        self._cached_sps_pps: Optional[bytes] = None
        self._sps_pps_injected: bool = False  # [FIX-SPS-PPS-V2] Writer-thread-side 注入已完成标志
        # [FIX-SPS-PPS] muxer 引用 — _cached_sps_pps 首次设置时通知 muxer 预注入
        self._muxer_ref: Optional[object] = None

        # 1. Load NVENC DLL
        self._dll_path = self._find_dll()
        try:
            self._dll = ctypes.CDLL(self._dll_path)
        except OSError as e:
            raise RuntimeError(
                "[NVENCEncoder] Cannot load NVENC DLL (%s): %s. "
                "Please verify NVIDIA driver is installed." % (self._dll_path, e))

        # 2. Load libcuda for GPU operations
        try:
            self._libcuda = ctypes.CDLL("libcuda.so.1" if sys.platform != "win32" else "nvcuda.dll")
            self._libcuda.cuInit(0)
        except Exception as e:
            raise RuntimeError("[NVENCEncoder] Cannot load CUDA library: %s" % e)

        # 3. Runtime API version detection
        try:
            _get_max_ver = self._dll.NvEncodeAPIGetMaxSupportedVersion
            _get_max_ver.restype = c_uint32
            _get_max_ver.argtypes = [ctypes.POINTER(c_uint32)]
            _max_ver_val = c_uint32(0)
            _get_max_ver(ctypes.byref(_max_ver_val))
            _nvenc_api_version = _max_ver_val.value if _max_ver_val.value > 0 else _NVENCAPI_VERSION_FALLBACK
        except Exception:
            _nvenc_api_version = _NVENCAPI_VERSION_FALLBACK
        print("[NVENCEncoder] NVENC API: 0x%x = v%d.%d" % (
            _nvenc_api_version, _nvenc_api_version >> 4, _nvenc_api_version & 0xF), flush=True)

        # 4. CUDA context setup — use primary context (cuCtxCreate fails code=2 on T4)
        self._saved_ctx = c_void_p(None)
        self._nvenc_own_ctx = False
        cuda_ctx = c_void_p(None)

        # Save current context
        self._libcuda.cuCtxGetCurrent.restype = c_uint32
        self._libcuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(c_void_p)]
        self._libcuda.cuCtxGetCurrent(ctypes.byref(self._saved_ctx))

        # Get primary context for device 0
        self._libcuda.cuDevicePrimaryCtxRetain.restype = c_uint32
        self._libcuda.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(c_void_p), c_int]
        primary_ctx = c_void_p(None)
        r = self._libcuda.cuDevicePrimaryCtxRetain(ctypes.byref(primary_ctx), c_int(0))
        if r == 0 and primary_ctx.value is not None:
            self._libcuda.cuCtxPushCurrent.restype = c_uint32
            self._libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
            self._libcuda.cuCtxPushCurrent(primary_ctx)
            cuda_ctx = primary_ctx
            self._primary_ctx = c_void_p(primary_ctx.value)  # [FIX-GPU-STAY] 跨线程 CUDA context 保护
            print("[NVENCEncoder] 使用 primary CUDA context (0x%x)" % primary_ctx.value, flush=True)
        elif self._saved_ctx.value is not None:
            cuda_ctx = self._saved_ctx
            print("[NVENCEncoder] 使用已存在的 CUDA context (0x%x)" % self._saved_ctx.value, flush=True)
        else:
            raise RuntimeError("[NVENCEncoder] 无法获取 CUDA context")

        # 5. NvEncodeAPICreateInstance
        _FUNC_TABLE_RAW_SIZE = 2552  # SDK 13.0 func_table size in bytes
        func_table = (c_uint8 * _FUNC_TABLE_RAW_SIZE)()
        _flist_ver = _sdk13_ver(2)  # NV_ENCODE_API_FUNCTION_LIST_VER = NVENCAPI_STRUCT_VERSION(2) = 0x7002000d
        cast(func_table, ctypes.POINTER(c_uint32))[0] = _flist_ver
        create_instance = _NvEncodeAPICreateInstanceProto(
            ("NvEncodeAPICreateInstance", self._dll))
        status = create_instance(cast(func_table, c_void_p))
        if status != NV_ENC_SUCCESS:
            raise RuntimeError(
                "[NVENCEncoder] NvEncodeAPICreateInstance failed, code=%d. "
                "Verify GPU supports NVENC." % status)

        # func_ptrs at offset 8 (skip version + reserved)
        self._func_ptrs = cast(byref(func_table, 8), ctypes.POINTER(c_void_p))
        # Keep ref to prevent GC
        self._func_table_raw = func_table

        def _get_func(idx):
            addr = self._func_ptrs[idx]
            if not addr or addr == 0:
                return None
            return addr

        # 6. OpenEncodeSessionEx
        open_func_addr = _get_func(_FUNC_IDX["OpenEncodeSessionEx"])
        if open_func_addr is None:
            raise RuntimeError("[NVENCEncoder] OpenEncodeSessionEx not available")
        open_session = _NvEncOpenEncodeSessionExProto(open_func_addr)

        # SDK 13.0 apiVersion uses NVENCAPI_VERSION = MAJOR | (MINOR << 24) = 0x0d
        # Also try old format (MAJOR << 4) | MINOR = 0xd0 for backward compat
        _api_try_list = sorted(set([
            _NVENCAPI_VERSION,       # 0x0d — SDK 13.0 new format
            _nvenc_api_version,      # from driver (0xd0)
            (13 << 4) | 0,           # 0xd0 — old format 13.0
            (12 << 4) | 0,           # 0xc0 — old format 12.0
        ]), reverse=True)
        status = 0xFFFF
        for _api_ver in _api_try_list:
            session_params = _NvEncOpenEncodeSessionExParams()
            session_params.version = _sdk13_ver(1)  # NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER = NVENCAPI_STRUCT_VERSION(1) = 0x7001000d
            session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA
            session_params.device = cuda_ctx
            session_params.apiVersion = _api_ver
            self._encoder = c_void_p(None)
            status = open_session(byref(session_params), byref(self._encoder))
            if status == NV_ENC_SUCCESS:
                print("[NVENCEncoder] OpenEncodeSessionEx OK: apiVersion=0x%x (%d.%d)" % (
                    _api_ver, _api_ver >> 4, _api_ver & 0xF), flush=True)
                break
            if status != 15:  # non-INVALID_VERSION, don't retry
                break
        if status != NV_ENC_SUCCESS:
            raise RuntimeError(
                "[NVENCEncoder] nvEncOpenEncodeSessionEx failed, code=%d" % status)

        # 7. Dynamic GUID query from driver
        _GetEncodeGUIDCountProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint32))
        _GetEncodeGUIDsProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(_NvGuid), c_uint32, ctypes.POINTER(c_uint32))
        _GetEncodePresetGUIDsProto = ctypes.CFUNCTYPE(
            c_uint32, c_void_p, _NvGuid, ctypes.POINTER(_NvGuid), c_uint32, ctypes.POINTER(c_uint32))

        count_val = c_uint32(0)
        s = _GetEncodeGUIDCountProto(_get_func(_FUNC_IDX["GetEncodeGUIDCount"]))(self._encoder, byref(count_val))
        if s != 0 or count_val.value == 0:
            raise RuntimeError("[NVENCEncoder] GetEncodeGUIDCount failed, code=%d" % s)

        n_guids = count_val.value
        guid_array = (_NvGuid * n_guids)()
        ctypes.memset(cast(guid_array, c_void_p), 0, sizeof(guid_array))
        actual_count = c_uint32(0)
        s = _GetEncodeGUIDsProto(_get_func(_FUNC_IDX["GetEncodeGUIDs"]))(self._encoder, guid_array, n_guids, byref(actual_count))
        if s != 0 or actual_count.value == 0:
            raise RuntimeError("[NVENCEncoder] GetEncodeGUIDs failed, code=%d" % s)
        codec_guid = guid_array[0]
        print("[NVENCEncoder] Driver codec GUID: %08x-%04x-%04x" % (
            codec_guid.Data1, codec_guid.Data2, codec_guid.Data3), flush=True)

        preset_guid_array = (_NvGuid * 64)()
        ctypes.memset(cast(preset_guid_array, c_void_p), 0, sizeof(preset_guid_array))
        preset_count = c_uint32(0)
        s = _GetEncodePresetGUIDsProto(_get_func(_FUNC_IDX["GetEncodePresetGUIDs"]))(
            self._encoder, codec_guid, preset_guid_array, 64, byref(preset_count))
        if s != 0 or preset_count.value == 0:
            raise RuntimeError("[NVENCEncoder] GetEncodePresetGUIDs failed, code=%d" % s)
        _p_idx = _PRESET_P_INDEX.get(self._preset_name, 4)
        _p_idx = min(_p_idx, preset_count.value - 1)
        preset_guid = preset_guid_array[_p_idx]
        print("[NVENCEncoder] Driver preset GUID: %08x-%04x-%04x (index=%d/%d)" % (
            preset_guid.Data1, preset_guid.Data2, preset_guid.Data3, _p_idx, preset_count.value - 1), flush=True)

        # 8. GetEncodePresetConfig
        get_preset_addr = _get_func(_FUNC_IDX["GetEncodePresetConfig"])
        if get_preset_addr is None:
            get_preset_addr = _get_func(_FUNC_IDX["GetEncodePresetConfigEx"])
        if get_preset_addr is None:
            raise RuntimeError("[NVENCEncoder] GetEncodePresetConfig not available")
        _GPC_fn = ctypes.CFUNCTYPE(c_uint32, c_void_p, _NvGuid, _NvGuid, ctypes.POINTER(_NvEncPresetConfig))

        preset_config = _NvEncPresetConfig()
        ctypes.memset(byref(preset_config), 0, sizeof(preset_config))
        preset_config.version = NV_ENC_PRESET_CONFIG_VER
        # SDK 中 presetCfg 从 offset 8 开始（version@0 + 4-byte padding）
        # ctypes _pack_=1 让它从 offset 4 开始，但 SDK 函数按 SDK 布局解析
        cast(byref(preset_config, 8), ctypes.POINTER(c_uint32))[0] = NV_ENC_CONFIG_VER

        status = _GPC_fn(get_preset_addr)(self._encoder, codec_guid, preset_guid, byref(preset_config))
        if status != NV_ENC_SUCCESS:
            raise RuntimeError("[NVENCEncoder] GetEncodePresetConfig failed, code=%d" % status)
        print("[NVENCEncoder] GetEncodePresetConfig OK", flush=True)

        # Configure encoding params (presetCfg at SDK offset 8)
        enc_cfg = cast(byref(preset_config, 8), ctypes.POINTER(_NvEncConfig)).contents
        enc_cfg.gopLength = int(fps)
        enc_cfg.frameIntervalP = 1
        enc_cfg.encodeCodecConfig.chromaFormatIDC = 1  # [FIX-CHROMA] 显式启用 chroma (1=4:2:0)，防止 SPS 声明为 monochrome → 灰色输出
        enc_cfg.encodeCodecConfig.idrPeriod = int(fps)
        enc_cfg.encodeCodecConfig.maxNumRefFramesInDPB = 4   # [FIX-GPU-STAY] 16→4: 减少运动估计开销 ~50%，文件体积更接近 v6.4.2
        enc_cfg.encodeCodecConfig.repeatSPSPPS = 1  # [SEGMENT-REUSE] 每个 IDR 前重发 SPS/PPS，确保新 muxer 能收到

        # [V6441-RC-FIXED] NV_ENC_RC_PARAMS at offset 40 in NV_ENC_CONFIG
        # SDK 13.0 layout (nvEncodeAPI.h, SEQUENTIAL — NO union at offset 8):
        #   NV_ENC_QP = {qpInterP@0, qpInterB@4, qpIntra@8} = 12 bytes sequential.
        #   version@0, rateControlMode@4,
        #   constQP@8 {qpInterP@8, qpInterB@12, qpIntra@16} (12B NV_ENC_QP struct),
        #   averageBitRate@20, maxBitRate@24,
        #   vbvBufferSize@28, vbvInitialDelay@32,
        #   bitfield@36 (enableMinQP:1, enableMaxQP:1, enableInitialRCQP:1,
        #     enableAQ:1=bit3, reserved:1=bit4, enableLookahead:1=bit5,
        #     disableIadapt:1=bit6, disableBadapt:1=bit7, enableTemporalAQ:1=bit8, ...)
        #   minQP@40 {InterP@40,InterB@44,Intra@48}, maxQP@52, initialRCQP@64,
        #   ── SEQUENTIAL layout (nvEncodeAPI.h master + GPU verified, 2026-06-09) ──
        #   temporallayerIdxMask@76, temporalLayerQP@80[8],
        #   targetQuality@88(uint8), targetQualityLSB@89(uint8),
        #   lookaheadDepth@90(uint16)  — qvbrQuality shares targetQuality slot
        #   lowDelayKeyFrameScale@92, yDcQPIndexOffset@93, uDcQPIndexOffset@94,
        #   vDcQPIndexOffset@95, qpMapMode@96, multiPass@100(uint32, rc_ptr[25]),
        #   alphaLayerBitrateRatio@104, cbQP@108, crQP@109, reserved2@110,
        #   lookaheadLevel@112, viewBitrateRatios@116[7], reserved3@123, reserved1@124
        rc_ptr = cast(byref(preset_config, 8 + 40), ctypes.POINTER(c_uint32))
        rc_ptr[0] = _sdk13_ver(1)                  # NV_ENC_RC_PARAMS_VER
        if self._rate_mode == 'constqp':
            _qp_val = qp          # CONSTQP: qp=0 合法（lossless），qp>0 正常
        elif qp <= 0:
            _qp_val = 1           # VBR_HQ/QVBR: targetQuality 范围 1-51, 1=最好 → crf=0 映射为 1
        else:
            _qp_val = qp          # VBR_HQ/QVBR: qp>0 保持直接映射

        if self._rate_mode == 'vbr_hq':
            # CQ mode: VBR_HQ + targetQuality (matches Level 2 -cq:v N behavior)
            # NV_ENC_PARAMS_RC_VBR_HQ = 32 (0x20) in SDK 12.0+.
            rc_ptr[1] = 32                           # NV_ENC_PARAMS_RC_VBR_HQ
            # [FIX-BR-CEILING] 提供合理的 avgBitRate 作为速度天花板。
            # targetQuality 仍主导质量决策，avgBitRate 仅防止 NVENC 在无约束下进入
            # 极慢的质量穷举搜索模式（GPU 闲置 65%，FPS 暴跌 2.2×）。
            # 参考 v6.4.3.1 的 _est_br 计算方式。
            _est_br = max(50000000, int(width * height * fps * 3.0))
            rc_ptr[5] = _est_br                      # averageBitRate @offset 20 (速度天花板)
            rc_ptr[6] = _est_br * 2                  # maxBitRate @offset 24
            _tq = max(1, _qp_val)  # VBR_HQ targetQuality = CRF (QP标度, 1=最好, 51=最差)
            # targetQuality: uint8_t at rcParams+88 (nvEncodeAPI.h SEQUENTIAL, GPU verified)
            _tq8_ptr = cast(byref(preset_config, 8 + 40 + 88), ctypes.POINTER(c_uint8))
            _tq8_ptr[0] = _tq & 0xFF
            print(f"[NVENCEncoder] VBR_HQ: crf={_qp_val} targetQuality={_tq} avgBitrate={_est_br//1000}kbps", flush=True)
        elif self._rate_mode == 'qvbr':
            # QVBR mode: NV_ENC_PARAMS_RC_QVBR = 64 (0x40)
            rc_ptr[1] = 64                           # NV_ENC_PARAMS_RC_QVBR (0x40)
            # [FIX-BR-CEILING] QVBR 也需要 avgBitRate 作为速度天花板，
            # 防止 NVENC 在无约束下进入极慢的质量搜索。
            _est_br = max(50000000, int(width * height * fps * 3.0))
            rc_ptr[5] = _est_br                      # averageBitRate @offset 20 (速度天花板)
            rc_ptr[6] = _est_br * 2                  # maxBitRate @offset 24 (码率上限)
            _tq = max(1, _qp_val)  # QVBR qvbrQuality = CRF (QP标度, 低值=高质量)
            _tq8_ptr = cast(byref(preset_config, 8 + 40 + 88), ctypes.POINTER(c_uint8))
            _tq8_ptr[0] = _tq & 0xFF
            # [OPT-VBV] VBV 缓冲合规由全局开关 _NVENC_QVBR_ENABLE_VBV 控制。
            # 离线编码不需要 VBV（虚拟解码器缓冲）合规约束，默认关闭。
            if _NVENC_QVBR_ENABLE_VBV:
                if rc_ptr[7] == 0:
                    rc_ptr[7] = 4194304              # vbvBufferSize @offset 28
                if rc_ptr[8] == 0:
                    rc_ptr[8] = 2097152              # vbvInitialDelay @offset 32
            print(f"[NVENCEncoder] QVBR: crf={_qp_val} qvbrQuality={_tq} maxBitrate={_est_br//1000}kbps", flush=True)
        else:
            # CONSTQP mode (default): direct QP control
            rc_ptr[1] = 0                            # NV_ENC_PARAMS_RC_CONSTQP
            rc_ptr[2] = _qp_val                      # constQP.qpInterP @offset 8
            rc_ptr[3] = _qp_val                      # constQP.qpInterB @offset 12
            rc_ptr[4] = _qp_val                      # constQP.qpIntra @offset 16

        # Enable AQ + Temporal AQ via rcParams bitfield (offset 36 = rc_ptr[9])
        # bit 3 = enableAQ, bit 8 = enableTemporalAQ (NOT bit 7!)
        rc_ptr[9] = rc_ptr[9] | (1 << 3) | (1 << 8)

        # Optional: lookahead for VBR_HQ/QVBR (matching Level 2 -rc-lookahead N)
        if self._la_depth > 0 and self._rate_mode in ('vbr_hq', 'qvbr'):
            _rc_bf = rc_ptr[9]                       # bitfield @offset 36
            _rc_bf |= (1 << 5)                       # enableLookahead (bit 5, NOT bit 4)
            rc_ptr[9] = _rc_bf
            # multiPass — uint32 at rc_ptr[25] (SEPARATE field, NOT in bitfield)
            rc_ptr[25] = 0                            # NV_ENC_MULTI_PASS_DISABLED (VBR_HQ/QVBR 不支持 two-pass)
            # lookaheadDepth — uint16 at rcParams+90 (nvEncodeAPI.h verified)
            _la_ptr = cast(byref(preset_config, 8 + 40 + 90), ctypes.POINTER(c_uint16))
            _la_ptr[0] = self._la_depth

        # 9. InitializeEncoder
        init_params = _NvEncInitializeParams()
        ctypes.memset(byref(init_params), 0, sizeof(init_params))
        init_params.version = NV_ENC_INITIALIZE_PARAMS_VER
        init_params.encodeGUID = codec_guid
        init_params.presetGUID = preset_guid
        init_params.encodeWidth = width
        init_params.encodeHeight = height
        init_params.darWidth = width
        init_params.darHeight = height
        init_params.frameRateNum = int(fps * 1000)
        init_params.frameRateDen = 1000
        init_params.maxEncodeWidth = width
        init_params.maxEncodeHeight = height
        init_params.enablePTD = 1
        init_params.encodeConfig = cast(byref(preset_config, 8), c_void_p)

        init_addr = _get_func(_FUNC_IDX["InitializeEncoder"])
        if init_addr is None:
            raise RuntimeError("[NVENCEncoder] InitializeEncoder not available")
        create_encoder = _NvEncCreateEncoderProto(init_addr)
        status = create_encoder(self._encoder, byref(init_params))
        if status != NV_ENC_SUCCESS:
            raise RuntimeError("[NVENCEncoder] InitializeEncoder failed, code=%d" % status)

        # 10. [PHASE4-v645] 创建多 slot 流水线：每 slot = input buffer + bitstream buffer + CUDA event。
        #     根因修复（v6.4.4 synchronize + FIX-ENC-CTX）后恢复，Multi-slot 让 NVENC HW 帧间流水线化：
        #     准备 slot N+1 的同时 NVENC 处理 slot N，Lock/Copy 与 Encode 重叠。
        nv12_h = height + height // 2
        _CreateInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 776))
        _CreateBitstreamBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 776))

        for slot_idx in range(self._slot_count):
            # 10a. Create input buffer
            create_buf = (c_uint8 * 776)()
            ctypes.memset(create_buf, 0, 776)
            cast(create_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_CREATE_INPUT_BUFFER_VER  # version@0
            cast(byref(create_buf, 4), ctypes.POINTER(c_uint32))[0] = width               # width@4
            cast(byref(create_buf, 8), ctypes.POINTER(c_uint32))[0] = height              # height@8 (luma only for NV12) [FIX-HEIGHT]
            cast(byref(create_buf, 16), ctypes.POINTER(c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12  # bufferFmt@16

            s = _CreateInputBufferProto(_get_func(_FUNC_IDX["CreateInputBuffer"]))(self._encoder, create_buf)
            if s != 0:
                self._destroy_all_slots()
                raise RuntimeError("[NVENCEncoder] CreateInputBuffer[%d] failed, code=%d" % (slot_idx, s))
            _raw_ptr = cast(byref(create_buf, 24), ctypes.POINTER(c_void_p))[0]  # inputBuffer@24
            input_handle = c_void_p(_raw_ptr if isinstance(_raw_ptr, int) else (_raw_ptr.value or 0))

            # 10b. Create bitstream buffer
            bs_buf = (c_uint8 * 776)()
            ctypes.memset(bs_buf, 0, 776)
            cast(bs_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_CREATE_BITSTREAM_BUFFER_VER  # version@0

            s = _CreateBitstreamBufferProto(_get_func(_FUNC_IDX["CreateBitstreamBuffer"]))(self._encoder, bs_buf)
            if s != 0:
                self._destroy_all_slots()
                raise RuntimeError("[NVENCEncoder] CreateBitstreamBuffer[%d] failed, code=%d" % (slot_idx, s))
            _raw_bs = cast(byref(bs_buf, 16), ctypes.POINTER(c_void_p))[0]  # bitstreamBuffer@16
            bs_handle = c_void_p(_raw_bs if isinstance(_raw_bs, int) else (_raw_bs.value or 0))

            # 10c. Create CUDA completion event (async ready signal for NVENC HW)
            event = c_void_p(None)
            r = self._libcuda.cuEventCreate(ctypes.byref(event), 0)  # 0 = cudaEventDefault
            if r != 0:
                self._destroy_all_slots()
                raise RuntimeError("[NVENCEncoder] cuEventCreate[%d] failed, code=%d" % (slot_idx, r))

            self._slots.append({
                'input_buf': input_handle,
                'bs_buf': bs_handle,
                'event': event,
            })

        # Backward compat: legacy refs point to slot 0
        self._input_buf_handle = self._slots[0]['input_buf']
        self._bs_handle = self._slots[0]['bs_buf']

        if self._slot_count > 1:
            print("[NVENCEncoder] %d slots created (0x%x..0x%x)" %
                  (self._slot_count, self._slots[0]['input_buf'].value,
                   self._slots[-1]['input_buf'].value), flush=True)

        # Setup cuMemcpyDtoD (1D linear, deprecated for pitched buffers)
        self._libcuda.cuMemcpyDtoD_v2.restype = c_uint32
        self._libcuda.cuMemcpyDtoD_v2.argtypes = [c_void_p, c_void_p, c_size_t]

        # Setup cuMemcpy2D_v2 (2D pitch-aware copy)
        self._libcuda.cuMemcpy2D_v2.restype = c_uint32
        self._libcuda.cuMemcpy2D_v2.argtypes = [c_void_p]

        # Setup context push/pop helpers
        self._libcuda.cuCtxPopCurrent.restype = c_uint32
        self._libcuda.cuCtxPopCurrent.argtypes = [ctypes.POINTER(c_void_p)]

        _mode_label = (self._rate_mode.upper() if self._rate_mode == 'vbr_hq'
                       else 'QVBR' if self._rate_mode == 'qvbr' else 'CONSTQP')
        _extra = ""
        if self._la_depth > 0:
            _extra += " la=%d" % self._la_depth
        if self._rate_mode in ('vbr_hq', 'qvbr'):
            _tq = max(1, _qp_val)  # QVBR qvbrQuality = CRF (QP标度)
            _extra += " tq=%d" % _tq
        print("[NVENCEncoder] Ready: %dx%d@%.1ffps H.264 %s QP=%d preset=%s slots=%d%s (GPU direct SDK 13.0)" %
              (width, height, fps, _mode_label, _qp_val, self._preset_name, self._slot_count, _extra), flush=True)

    def set_muxer_ref(self, muxer: object) -> None:
        """[FIX-SPS-PPS] 设置 muxer 引用，供 _cached_sps_pps 首次缓存时预注入用。"""
        self._muxer_ref = muxer

    def _destroy_slot(self, slot: dict):
        """Destroy a single pipeline slot's buffers and event."""
        if slot.get('input_buf') and slot['input_buf'].value is not None:
            try:
                _DestroyInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
                addr = self._func_ptrs[_FUNC_IDX["DestroyInputBuffer"]]
                if addr:
                    _DestroyInputBufferProto(addr)(self._encoder, slot['input_buf'])
            except Exception:
                pass
            slot['input_buf'] = c_void_p(None)
        if slot.get('bs_buf') and slot['bs_buf'].value is not None:
            try:
                _DestroyBitstreamBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
                addr = self._func_ptrs[_FUNC_IDX["DestroyBitstreamBuffer"]]
                if addr:
                    _DestroyBitstreamBufferProto(addr)(self._encoder, slot['bs_buf'])
            except Exception:
                pass
            slot['bs_buf'] = c_void_p(None)
        if slot.get('event') and slot['event'].value is not None:
            try:
                self._libcuda.cuEventDestroy(slot['event'])
            except Exception:
                pass
            slot['event'] = c_void_p(None)

    def _destroy_all_slots(self):
        """Destroy all pipeline slots (used on init failure or close)."""
        for slot in self._slots:
            self._destroy_slot(slot)
        self._slots.clear()
        self._input_buf_handle = c_void_p(None)
        self._bs_handle = c_void_p(None)

    def _find_dll(self) -> str:
        if sys.platform == "win32":
            candidates = [
                "nvEncodeAPI64.dll",
                os.path.join(os.environ.get("WINDIR", r"C:\Windows"),
                             "System32", "nvEncodeAPI64.dll"),
            ]
            prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
            nvidia_video = os.path.join(prog_files, "NVIDIA Corporation",
                                         "NVIDIA Video Codec SDK")
            if os.path.isdir(nvidia_video):
                for root, dirs, files in os.walk(nvidia_video):
                    if "nvEncodeAPI64.dll" in files:
                        candidates.insert(0, os.path.join(root, "nvEncodeAPI64.dll"))
                        break
            for c in candidates:
                if os.path.exists(c) or not os.path.dirname(c):
                    return c
            return "nvEncodeAPI64.dll"
        else:
            candidates = [
                "/usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1",
                "/usr/lib64/libnvidia-encode.so.1",
                "/usr/local/lib/libnvidia-encode.so.1",
                "libnvidia-encode.so.1",
            ]
            for search_dir in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/lib", "/usr/lib"):
                if os.path.isdir(search_dir):
                    for fname in os.listdir(search_dir):
                        if fname.startswith("libnvidia-encode.so"):
                            candidates.insert(0, os.path.join(search_dir, fname))
            for c in candidates:
                if os.path.exists(c):
                    return c
            return "libnvidia-encode.so.1"

    @staticmethod
    def _extract_sps_pps(h264_data: bytes) -> Optional[bytes]:
        """从 H.264 Annex B ES 中提取 SPS+PPS NAL 单元（含起始码）。"""
        result_parts: List[bytes] = []
        pos = 0
        n = len(h264_data)
        while pos < n - 3:
            # 查找 4-byte 起始码 0x00 0x00 0x00 0x01
            if h264_data[pos:pos+4] == b'\x00\x00\x00\x01':
                start = pos
                pos += 4
                if pos >= n:
                    break
                nal_byte = h264_data[pos]
                nal_type = nal_byte & 0x1f
                # 查找下一个起始码（3-byte 或 4-byte）
                end = n
                for j in range(pos, n - 2):
                    if h264_data[j:j+3] == b'\x00\x00\x01' or h264_data[j:j+4] == b'\x00\x00\x00\x01':
                        end = j
                        break
                if nal_type in (7, 8):  # SPS=7, PPS=8
                    result_parts.append(h264_data[start:end])
                pos = end
            elif h264_data[pos:pos+3] == b'\x00\x00\x01':
                start = pos
                pos += 3
                if pos >= n:
                    break
                nal_byte = h264_data[pos]
                nal_type = nal_byte & 0x1f
                end = n
                for j in range(pos, n - 2):
                    if h264_data[j:j+3] == b'\x00\x00\x01' or h264_data[j:j+4] == b'\x00\x00\x00\x01':
                        end = j
                        break
                if nal_type in (7, 8):
                    result_parts.append(h264_data[start:end])
                pos = end
            else:
                pos += 1
        if result_parts:
            return b''.join(result_parts)
        return None

    def _drain_outputs_blocking(self, max_slots: int = None) -> list:
        """[FIX-LA-REDRAIN] 按 _output_slot_idx 顺序循环 blocking LockBitstream
        排空所有已完成 slot，直到遇到 NEED_MORE_INPUT。

        参照 tests/test_nvenc_la_frame_conservation.py 的 _drain_outputs() 验证模式：
        - 从 _output_slot_idx 指向的 slot 开始，循环 blocking Lock
        - 每成功取回一帧则推进 _output_slot_idx
        - 遇到 NEED_MORE_INPUT 时退出循环

        返回 [(frame_index_estimate, h264_bytes), ...] 列表。
        """
        if max_slots is None:
            max_slots = self._slot_count
        outputs = []
        _LockBS = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
        _UnlockBS = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
        lock_bs_fn = _LockBS(self._func_ptrs[_FUNC_IDX["LockBitstream"]])
        unlock_fn = _UnlockBS(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])

        for _ in range(max_slots):
            slot_idx = self._output_slot_idx % self._slot_count
            bs_handle = self._slots[slot_idx]['bs_buf']

            lock_raw = (c_uint8 * 1544)()
            ctypes.memset(lock_raw, 0, 1544)
            cast(lock_raw, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
            cast(byref(lock_raw, 4), ctypes.POINTER(c_uint32))[0] = 0  # doNotWait=0 (blocking)
            cast(byref(lock_raw, 8), ctypes.POINTER(c_void_p))[0] = bs_handle

            bs_status = lock_bs_fn(self._encoder, lock_raw)
            if bs_status == NV_ENC_ERR_NEED_MORE_INPUT:
                break
            if bs_status != NV_ENC_SUCCESS:
                break

            bitstream_size = cast(byref(lock_raw, 36), ctypes.POINTER(c_uint32))[0]
            if bitstream_size == 0:
                unlock_fn(self._encoder, bs_handle)
                break

            _raw_bsptr = cast(byref(lock_raw, 56), ctypes.POINTER(c_void_p))[0]
            bitstream_ptr_val = _raw_bsptr if isinstance(_raw_bsptr, int) else (_raw_bsptr.value or 0)
            if bitstream_ptr_val:
                buf_type = c_uint8 * bitstream_size
                h264_data = bytes(buf_type.from_address(bitstream_ptr_val))
                est_fi = self._output_slot_idx
                outputs.append((est_fi, h264_data))
            unlock_fn(self._encoder, bs_handle)
            self._output_slot_idx += 1

        return outputs

    def _reset_output_slot_idx(self, start: int = 0):
        """[FIX-LA-OUTPTR] 重置输出槽位指针（新批次开始时调用）。"""
        self._output_slot_idx = start

    def _lock_bitstream_blocking(self, bs_handle, timeout_ms: int = 500):
        """[FIX-LA-BLOCKING] Blocking LockBitstream (doNotWait=0)。

        用于 NEED_MORE_INPUT 帧的可靠数据取回：CE 可能提前触发，
        用 blocking Lock 等待硬件实际完成编码并写入 bs_buf。
        超时后返回 (b"", status)。

        Returns (h264_data: bytes, status: int)
        """
        import time as _time
        _LockBS = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
        _UnlockBS = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
        lock_bs_fn = _LockBS(self._func_ptrs[_FUNC_IDX["LockBitstream"]])
        unlock_fn = _UnlockBS(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])
        _deadline = _time.monotonic() + timeout_ms / 1000.0

        while _time.monotonic() < _deadline:
            lock_raw = (c_uint8 * 1544)()
            ctypes.memset(lock_raw, 0, 1544)
            cast(lock_raw, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
            cast(byref(lock_raw, 4), ctypes.POINTER(c_uint32))[0] = 0  # doNotWait=0
            cast(byref(lock_raw, 8), ctypes.POINTER(c_void_p))[0] = bs_handle

            bs_status = lock_bs_fn(self._encoder, lock_raw)
            if bs_status == NV_ENC_ERR_NEED_MORE_INPUT:
                _time.sleep(0.001)
                continue
            if bs_status != NV_ENC_SUCCESS:
                return b"", bs_status

            bitstream_size = cast(byref(lock_raw, 36), ctypes.POINTER(c_uint32))[0]
            if bitstream_size > 0:
                _raw_bsptr = cast(byref(lock_raw, 56), ctypes.POINTER(c_void_p))[0]
                bitstream_ptr_val = _raw_bsptr if isinstance(_raw_bsptr, int) else (_raw_bsptr.value or 0)
                if bitstream_ptr_val:
                    buf_type = c_uint8 * bitstream_size
                    h264_data = bytes(buf_type.from_address(bitstream_ptr_val))
                    unlock_fn(self._encoder, bs_handle)
                    return h264_data, bs_status
            unlock_fn(self._encoder, bs_handle)
            _time.sleep(0.001)

        return b"", NV_ENC_ERR_NEED_MORE_INPUT

    def _lock_bitstream_with_retry(self, bs_handle, max_retries: int = 5, backoff_us: int = 1000):
        """[Tier 3-E] 带指数退避重试的 LockBitstream。

        应对 NVENC HW 的 bitstream DMA 竞态。
        [V6442-CONSTQP-FAST] CONSTQP 零空帧 → max_retries=2。
        [FIX-LA-LOCKBS] NEED_MORE_INPUT 立即返回不重试，让调用方正确区分
        "LA 缓冲中无数据"与"瞬时空帧"，避免静默丢失 LA 滞后产出帧。

        Returns (h264_data: bytes, status: int)
        """
        import time as _time
        # [V6442-CONSTQP-FAST] CONSTQP 零空帧 → 减少重试开销
        if self._rate_mode == 'constqp':
            max_retries = min(max_retries, 2)
        _LockBitstreamProto_raw = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
        lock_bs_fn = _LockBitstreamProto_raw(self._func_ptrs[_FUNC_IDX["LockBitstream"]])
        _unlock_fn = _NvEncUnlockBitstreamProto(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])

        for attempt in range(max_retries):
            lock_raw = (c_uint8 * 1544)()
            ctypes.memset(lock_raw, 0, 1544)
            cast(lock_raw, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
            cast(byref(lock_raw, 8), ctypes.POINTER(c_void_p))[0] = bs_handle

            bs_status = lock_bs_fn(self._encoder, lock_raw)
            # [FIX-LA-LOCKBS] NEED_MORE_INPUT 表示 LA 缓冲区中无可用输出，
            # 不是瞬态 DMA 竞态，不应重试。立即返回让调用方正确决策。
            if bs_status == NV_ENC_ERR_NEED_MORE_INPUT:
                return b"", NV_ENC_ERR_NEED_MORE_INPUT
            if bs_status != NV_ENC_SUCCESS:
                return b"", bs_status

            bitstream_size = cast(byref(lock_raw, 36), ctypes.POINTER(c_uint32))[0]
            if bitstream_size > 0:
                _raw_bsptr = cast(byref(lock_raw, 56), ctypes.POINTER(c_void_p))[0]
                bitstream_ptr_val = _raw_bsptr if isinstance(_raw_bsptr, int) else (_raw_bsptr.value or 0)
                if bitstream_ptr_val:
                    buf_type = c_uint8 * bitstream_size
                    h264_data = bytes(buf_type.from_address(bitstream_ptr_val))
                    _unlock_fn(self._encoder, bs_handle)
                    return h264_data, bs_status

            _unlock_fn(self._encoder, bs_handle)
            if attempt < max_retries - 1:
                _time.sleep(backoff_us / 1_000_000.0)
                backoff_us *= 2

        # [FIX-LA-LOCKBS] 重试耗尽且 bitstream_size 始终为 0 → 真正的空帧。
        # 返回 bs_status (SUCCESS) 以区分 NEED_MORE_INPUT。
        return b"", bs_status

    def encode_frames_batch(self, nv12_tensors: list, force_idr_first: bool = False,
                             send_eos: bool = False) -> list:
        """Encode multiple NV12 frames using synchronous per-slot encoding.

        Uses blocking EncodePicture (no completionEvent) + LockBitstream with retry.
        Frames distributed across pipeline slots round-robin to reduce buffer contention.

        Args:
            send_eos: If True, send EOS after all frames and drain ALL LA-buffered
                      frames into the results array. Required for single-batch-per-segment
                      encoding to achieve frame conservation. Default False (backward compat).

        Returns list of H.264 bytes in the same order as input tensors.
        """
        n_frames = len(nv12_tensors)
        if n_frames == 0:
            return []

        # [FIX-GPU-STAY] 跨线程 CUDA context 保护
        _need_pop = False
        _primary = getattr(self, '_primary_ctx', None)
        if _primary is not None and _primary.value is not None:
            try:
                self._libcuda.cuCtxPushCurrent.restype = c_uint32
                self._libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
                _r_push = self._libcuda.cuCtxPushCurrent(_primary)
                _need_pop = (_r_push == 0)
            except Exception:
                pass

        try:
            with self._lock:
                if self._encoder.value is None:
                    raise RuntimeError("[NVENCEncoder] Encoder not initialized or already closed")

                results = [None] * n_frames
                # [FIX-LA-DRAIN-SYNC] _slot_pending maps drained output to actual frame index.
                _slot_pending = [None] * self._slot_count  # (fi, bs_buf, force_idr, ep_status)
                # [FIX-PERSLOT-IDR] 每槽首帧强制 IDR 以初始化独立 DPB
                _slots_warmed = set()
                W = self._width
                nv12_h = self._height + self._height // 2

                _LockInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _UnlockInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
                _LockBitstreamProto_raw = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _CU_MEMORYTYPE_DEVICE = 2

                for fi in range(n_frames):
                    # [FIX-F0-MISSING] 使用全局 _frame_idx 分配 slot
                    slot_idx = self._frame_idx % self._slot_count
                    slot = self._slots[slot_idx]
                    force_idr = force_idr_first and (slot_idx not in _slots_warmed)

                    # ── LockInputBuffer ──
                    lock_buf = (c_uint8 * 1544)()
                    ctypes.memset(lock_buf, 0, 1544)
                    cast(lock_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_INPUT_BUFFER_VER
                    cast(byref(lock_buf, 8), ctypes.POINTER(c_void_p))[0] = slot['input_buf']

                    lock_addr = self._func_ptrs[_FUNC_IDX["LockInputBuffer"]]
                    s = _LockInputBufferProto(lock_addr)(self._encoder, lock_buf)
                    if s != 0:
                        raise RuntimeError("[NVENCEncoder] LockInputBuffer[%d] failed, code=%d" % (slot_idx, s))

                    _raw_map = cast(byref(lock_buf, 16), ctypes.POINTER(c_void_p))[0]
                    mapped_ptr = _raw_map if isinstance(_raw_map, int) else (_raw_map.value or 0)
                    actual_pitch = cast(byref(lock_buf, 24), ctypes.POINTER(c_uint32))[0]

                    if not mapped_ptr:
                        _UnlockInputBufferProto(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                            self._encoder, slot['input_buf'])
                        raise RuntimeError("[NVENCEncoder] LockInputBuffer[%d] returned NULL mapped ptr" % slot_idx)

                    # ── GPU→GPU copy (cuMemcpy2D, pitch-aware) ──
                    _cpy2d = (c_uint8 * 128)()
                    ctypes.memset(_cpy2d, 0, 128)
                    src_ptr = nv12_tensors[fi].data_ptr()
                    cast(byref(_cpy2d, 16), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
                    cast(byref(_cpy2d, 32), ctypes.POINTER(c_void_p))[0] = c_void_p(src_ptr)
                    cast(byref(_cpy2d, 48), ctypes.POINTER(c_size_t))[0] = W
                    cast(byref(_cpy2d, 72), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
                    cast(byref(_cpy2d, 88), ctypes.POINTER(c_void_p))[0] = c_void_p(mapped_ptr)
                    cast(byref(_cpy2d, 104), ctypes.POINTER(c_size_t))[0] = (
                        actual_pitch if actual_pitch > 0 else W)
                    cast(byref(_cpy2d, 112), ctypes.POINTER(c_size_t))[0] = W
                    cast(byref(_cpy2d, 120), ctypes.POINTER(c_size_t))[0] = nv12_h
                    r = self._libcuda.cuMemcpy2D_v2(cast(_cpy2d, c_void_p))
                    if r != 0:
                        _UnlockInputBufferProto(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                            self._encoder, slot['input_buf'])
                        raise RuntimeError("[NVENCEncoder] cuMemcpy2D[%d] failed, code=%d" % (slot_idx, r))

                    # ── UnlockInputBuffer ──
                    _UnlockInputBufferProto(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                        self._encoder, slot['input_buf'])

                    # ── EncodePicture (synchronous, NO completionEvent) ──
                    pic_buf = (c_uint8 * 3360)()
                    ctypes.memset(pic_buf, 0, 3360)
                    cast(pic_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
                    cast(byref(pic_buf, 4), ctypes.POINTER(c_uint32))[0] = W
                    cast(byref(pic_buf, 8), ctypes.POINTER(c_uint32))[0] = self._height
                    cast(byref(pic_buf, 12), ctypes.POINTER(c_uint32))[0] = (
                        actual_pitch if actual_pitch > 0 else W)
                    cast(byref(pic_buf, 24), ctypes.POINTER(c_uint64))[0] = self._frame_idx
                    cast(byref(pic_buf, 40), ctypes.POINTER(c_void_p))[0] = slot['input_buf']
                    cast(byref(pic_buf, 48), ctypes.POINTER(c_void_p))[0] = slot['bs_buf']
                    # NO completionEvent at offset 56 — synchronous encode
                    cast(byref(pic_buf, 64), ctypes.POINTER(c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12
                    cast(byref(pic_buf, 68), ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_STRUCT_FRAME
                    if force_idr:
                        cast(byref(pic_buf, 16), ctypes.POINTER(c_uint32))[0] = 0x2

                    encode_picture = _NvEncEncodePictureProto(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                    _ep_status = encode_picture(self._encoder, cast(pic_buf, ctypes.POINTER(_NvEncPicParams)))
                    self._frame_idx += 1
                    # [FIX-LA-DRAIN-SYNC] 保存提交上下文，供 drain 映射实际帧索引
                    _slot_pending[slot_idx] = (fi, slot['bs_buf'], force_idr, _ep_status)

                    if _ep_status == NV_ENC_ERR_NEED_MORE_INPUT:
                        # LA 缓冲: 编码器需要更多帧填入前向预看窗口才能产出数据。
                        # 此时 bs_buf 中无可用输出，不应做任何 Lock/Unlock 操作（SDK 规范：
                        # NEED_MORE_INPUT 时强制 Lock/Unlock 会静默丢弃延迟输出帧，破坏帧数守恒）。
                        self.__dict__.setdefault('_la_buffered', 0)
                        self._la_buffered += 1
                        if self._la_buffered <= 3 or self._la_buffered == self._la_depth:
                            print(f'[NVENC-Enc] 前向帧预看缓冲中 ({self._la_buffered}/{self._la_depth})',
                                  flush=True)
                        # ★ [FIX-LA-DRAIN-SYNC] NEED_MORE_INPUT 后仍需 drain
                    elif _ep_status != NV_ENC_SUCCESS:
                        raise RuntimeError("[NVENCEncoder] EncodePicture[%d] failed, code=%d" % (slot_idx, _ep_status))
                    else:
                        # [FIX-PERSLOT-IDR] 标记该 slot 已初始化 DPB
                        _slots_warmed.add(slot_idx)

                    # ★ [FIX-LA-DRAIN-SYNC] 全局 drain: 循环 LockBitstream 排空所有已完成 slot
                    _drained = self._drain_outputs_blocking()
                    for _est_fi, _h264_data in _drained:
                        _drain_slot = _est_fi % self._slot_count
                        _pending = _slot_pending[_drain_slot]
                        if _pending is None:
                            continue
                        _actual_fi, _, _is_idr, _ep_s = _pending
                        if _actual_fi >= n_frames:
                            continue
                        if results[_actual_fi] is not None and results[_actual_fi] != b"":
                            continue
                        if not _h264_data:
                            self.__dict__.setdefault('_diag_empty', 0)
                            self._diag_empty += 1
                            _nv12_t = nv12_tensors[_actual_fi]
                            _nv12_mean = float(_nv12_t.float().mean())
                            _nv12_std  = float(_nv12_t.float().std())
                            _nv12_min  = int(_nv12_t.min())
                            _nv12_max  = int(_nv12_t.max())
                            _la_buf = getattr(self, '_la_buffered', 0)
                            if self._diag_empty <= 5 or self._diag_empty % 50 == 0:
                                print(f'[NVENC-Enc] ⚠️ 空帧 #{self._diag_empty} (batch fi={_actual_fi}/{n_frames} '
                                      f'slot={_drain_slot} fidr={_is_idr} ep_status={_ep_s} '
                                      f'la_buf={_la_buf}) '
                                      f'nv12_mean={_nv12_mean:.1f} std={_nv12_std:.1f} '
                                      f'min={_nv12_min} max={_nv12_max}', flush=True)
                            results[_actual_fi] = b"" if _ep_s == NV_ENC_ERR_NEED_MORE_INPUT else None
                        else:
                            # [SEGMENT-REUSE] SPS/PPS caching and pre-pending
                            if _is_idr and self._cached_sps_pps is not None:
                                _h264_data = self._cached_sps_pps + _h264_data
                            elif _is_idr and self._cached_sps_pps is None:
                                self._cached_sps_pps = self._extract_sps_pps(_h264_data)
                                if self._cached_sps_pps:
                                    print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                          len(self._cached_sps_pps), flush=True)
                                    # [FIX-SPS-PPS-V3] 首帧已含 NVENC 初始化 SPS+PPS, 不 prepend, 仅预注入 muxer
                                    if self._muxer_ref is not None:
                                        try:
                                            self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                            self._sps_pps_injected = True
                                        except Exception: pass
                            elif not _is_idr and self._cached_sps_pps is None:
                                self._cached_sps_pps = self._extract_sps_pps(_h264_data)
                                if self._cached_sps_pps:
                                    print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                          len(self._cached_sps_pps), flush=True)
                            results[_actual_fi] = _h264_data
                        _slot_pending[_drain_slot] = None

                # ── [FIX-LA-DRAIN-SYNC] 最终排空 ──
                if send_eos:
                    # [FIX-LA-EOS-IN-BATCH] send_eos=True: 发送 EOS 后完整排空所有 slot。
                    # SDK 合规：EOS → 按 _output_slot_idx 顺序逐 slot blocking LockBitstream
                    # 直到 NEED_MORE_INPUT，回收全部 LA 滞留帧，保证帧数守恒。
                    eos_pic_buf = (c_uint8 * 3360)()
                    ctypes.memset(eos_pic_buf, 0, 3360)
                    cast(eos_pic_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
                    cast(byref(eos_pic_buf, 16), ctypes.POINTER(c_uint32))[0] = 0x8  # EOS flag
                    cast(byref(eos_pic_buf, 40), ctypes.POINTER(c_void_p))[0] = c_void_p(None)
                    cast(byref(eos_pic_buf, 48), ctypes.POINTER(c_void_p))[0] = self._bs_handle
                    _eos_ep = _NvEncEncodePictureProto(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                    _eos_ep(self._encoder, cast(eos_pic_buf, ctypes.POINTER(_NvEncPicParams)))

                    _LockBS = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                    _UnlockBS = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
                    _start_slot = self._output_slot_idx % self._slot_count
                    _drain_order = [(_start_slot + i) % self._slot_count
                                    for i in range(self._slot_count)]
                    for _ds in _drain_order:
                        _bs_h = self._slots[_ds]['bs_buf']
                        while True:
                            _lr = (c_uint8 * 1544)()
                            ctypes.memset(_lr, 0, 1544)
                            cast(_lr, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
                            cast(byref(_lr, 4), ctypes.POINTER(c_uint32))[0] = 0  # blocking
                            cast(byref(_lr, 8), ctypes.POINTER(c_void_p))[0] = _bs_h
                            _bs_s = _LockBS(self._func_ptrs[_FUNC_IDX["LockBitstream"]])(self._encoder, _lr)
                            if _bs_s == NV_ENC_ERR_NEED_MORE_INPUT:
                                break
                            if _bs_s != NV_ENC_SUCCESS:
                                break
                            _bs_size = cast(byref(_lr, 36), ctypes.POINTER(c_uint32))[0]
                            if _bs_size == 0:
                                _UnlockBS(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(self._encoder, _bs_h)
                                break
                            _raw_ptr = cast(byref(_lr, 56), ctypes.POINTER(c_void_p))[0]
                            _ptr_val = _raw_ptr if isinstance(_raw_ptr, int) else (_raw_ptr.value or 0)
                            if _ptr_val:
                                _buf = c_uint8 * _bs_size
                                _eos_data = bytes(_buf.from_address(_ptr_val))
                                _pend = _slot_pending[_ds]
                                if _pend is not None:
                                    _actual_fi_d, _, _is_idr_d, _ = _pend
                                    if _actual_fi_d < n_frames and (results[_actual_fi_d] is None or results[_actual_fi_d] == b""):
                                        if _is_idr_d and self._cached_sps_pps is not None:
                                            _eos_data = self._cached_sps_pps + _eos_data
                                        elif _is_idr_d and self._cached_sps_pps is None:
                                            self._cached_sps_pps = self._extract_sps_pps(_eos_data)
                                            if self._cached_sps_pps:
                                                print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                                      len(self._cached_sps_pps), flush=True)
                                                if self._muxer_ref is not None:
                                                    try:
                                                        self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                                        self._sps_pps_injected = True
                                                    except Exception:
                                                        pass
                                        results[_actual_fi_d] = _eos_data
                                        _slot_pending[_ds] = None
                                        self._output_slot_idx += 1
                            _UnlockBS(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(self._encoder, _bs_h)
                else:
                    _drained_final = self._drain_outputs_blocking()
                    for _est_fi, _h264_data in _drained_final:
                        _drain_slot = _est_fi % self._slot_count
                        _pending = _slot_pending[_drain_slot]
                        if _pending is None:
                            continue
                        _actual_fi, _, _is_idr, _ = _pending
                        if _actual_fi < n_frames and (results[_actual_fi] is None or results[_actual_fi] == b""):
                            if _h264_data:
                                results[_actual_fi] = _h264_data
                            _slot_pending[_drain_slot] = None
                for _fi in range(n_frames):
                    if results[_fi] is None:
                        results[_fi] = b""

                return results
        finally:
            if _need_pop:
                try:
                    _ctx_out = c_void_p()
                    self._libcuda.cuCtxPopCurrent.restype = c_uint32
                    self._libcuda.cuCtxPopCurrent.argtypes = [ctypes.POINTER(c_void_p)]
                    self._libcuda.cuCtxPopCurrent(ctypes.byref(_ctx_out))
                except Exception:
                    pass

    def encode_frames_batch_ce_pipeline(self, nv12_tensors: list, force_idr_first: bool = False) -> list:
        """[PHASE4-CE-PIPELINE] 带 per-frame CE 的异步流水线编码。

        EncodePicture 提交时附带 per-frame CUDA completion event，立即返回；
        LockBitstream 延迟到 slot 下次轮转时执行（pipeline_depth 帧后），此时 CE
        已触发、NVENC 硬件已完成编码 → LockBitstream 立即拿到数据，消除同步阻塞。

        Phase 1 (Harvest): slot 重用时 harvest 上一轮的 pending CE + LockBitstream
        Phase 2 (Submit):  LockInputBuffer → cuMemcpy2D → EncodePicture + 新建 CE
        Phase 3 (Drain):   批次结束 drain 所有 pending slots

        Returns list where: bytes = valid H.264, b"" = LA buffering frame, None = empty.
        GPU 验证 (T4, 720×576, VBR_HQ, pipe=4, LA=0): 523 FPS vs sync-batch 375 FPS (+39.5%)。
        """
        n_frames = len(nv12_tensors)
        if n_frames == 0:
            return []

        _need_pop = False
        _primary = getattr(self, '_primary_ctx', None)
        if _primary is not None and _primary.value is not None:
            try:
                self._libcuda.cuCtxPushCurrent.restype = c_uint32
                self._libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
                _r_push = self._libcuda.cuCtxPushCurrent(_primary)
                _need_pop = (_r_push == 0)
            except Exception:
                pass

        try:
            with self._lock:
                if self._encoder.value is None:
                    raise RuntimeError("[NVENCEncoder] Encoder not initialized or already closed")

                pd = self._slot_count
                W = self._width
                nv12_h = self._height + self._height // 2
                results = [None] * n_frames

                # [FIX-LA-OUTPTR] 每批次开始时重置输出槽位指针
                self._reset_output_slot_idx(0)

                # Per-slot pending state: (ce_handle, frame_index, ep_status, force_idr)
                _slot_pending = [None] * pd
                # [FIX-PIPE4-LA8] fi==0 only force_idr (单 IDR 替代 per-slot ×4 IDR)
                _LockIB = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _UnlockIB = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
                _CUDA_DEV = 2

                for fi in range(n_frames):
                    # [FIX-F0-MISSING] 使用全局 _frame_idx 分配 slot
                    slot_idx = self._frame_idx % pd
                    slot = self._slots[slot_idx]
                    # [FIX-PIPE4-LA8] fi==0 only force_idr (单 IDR 替代 per-slot ×4 IDR)
                    force_idr = force_idr_first and (fi == 0)

                    # ═══════════════════════════════════════════════
                    # Phase 1: Harvest pending frame from this slot (if any)
                    # ═══════════════════════════════════════════════
                    if _slot_pending[slot_idx] is not None:
                        _prev_ce, _prev_fi, _prev_ep_s, _prev_idr, _prev_bs = _slot_pending[slot_idx]
                        # Wait for CE → data is now in _prev_bs (the bs_buf this frame was submitted to)
                        if _prev_ce.value is not None:
                            self._libcuda.cuEventSynchronize.restype = c_uint32
                            self._libcuda.cuEventSynchronize.argtypes = [c_void_p]
                            self._libcuda.cuEventSynchronize(_prev_ce)
                            self._libcuda.cuEventDestroy.restype = c_uint32
                            self._libcuda.cuEventDestroy.argtypes = [c_void_p]
                            self._libcuda.cuEventDestroy(_prev_ce)
                        h264_data, bs_status = self._lock_bitstream_with_retry(_prev_bs)
                        if h264_data:
                            # [SEGMENT-REUSE] SPS/PPS caching and pre-pending
                            if _prev_idr and self._cached_sps_pps is not None:
                                h264_data = self._cached_sps_pps + h264_data
                            elif _prev_idr and self._cached_sps_pps is None and h264_data:
                                self._cached_sps_pps = self._extract_sps_pps(h264_data)
                                if self._cached_sps_pps:
                                    print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                          len(self._cached_sps_pps), flush=True)
                                    # [FIX-SPS-PPS-V3] 首帧已含 NVENC 初始化 SPS+PPS, 不 prepend, 仅预注入 muxer
                                    if self._muxer_ref is not None:
                                        try:
                                            self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                            self._sps_pps_injected = True
                                        except Exception:
                                            pass
                            elif not _prev_idr and self._cached_sps_pps is None and h264_data:
                                self._cached_sps_pps = self._extract_sps_pps(h264_data)
                                if self._cached_sps_pps:
                                    print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                          len(self._cached_sps_pps), flush=True)
                            results[_prev_fi] = h264_data
                            # [FIX-LA-OUTPTR] harvest 成功 → 推进输出指针
                            self._output_slot_idx += 1
                        elif _prev_ep_s == NV_ENC_ERR_NEED_MORE_INPUT:
                            # [FIX-LA-PENDING] LA 帧可能尚未完成编码（CE 在帧入队
                            # 时即触发，但编码尚未完成）。使用 blocking LockBitstream
                            # 再试一次取回实际数据，避免将有效帧误标为 b""。
                            # [FIX-LA-BLKRETRY-ALWAYS] 无论非阻塞 LockBitstream
                            # 返回什么 bs_status，NEED_MORE_INPUT 帧都尝试
                            # blocking LockBitstream。bs_status==SUCCESS 但
                            # bitstream_size==0 (LA 尚未产出数据)时旧逻辑
                            # 直接丢帧 → flush 写出错位帧破坏 GOP。
                            _h264_blk, _ = self._lock_bitstream_blocking(_prev_bs, timeout_ms=5000)
                            if _h264_blk:
                                # [FIX-LA-SPS-BLKRETRY] blocking retry 取回的
                                # 首个 IDR 帧必须同步缓存 SPS+PPS
                                if _prev_idr and self._cached_sps_pps is None:
                                    self._cached_sps_pps = self._extract_sps_pps(_h264_blk)
                                    if self._cached_sps_pps:
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                              len(self._cached_sps_pps), flush=True)
                                        if self._muxer_ref is not None:
                                            try:
                                                self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                                self._sps_pps_injected = True
                                            except Exception:
                                                pass
                                elif self._cached_sps_pps is None:
                                    self._cached_sps_pps = self._extract_sps_pps(_h264_blk)
                                    if self._cached_sps_pps:
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                              len(self._cached_sps_pps), flush=True)
                                results[_prev_fi] = _h264_blk
                                # [FIX-LA-OUTPTR] blocking 取回成功 → 推进输出指针
                                self._output_slot_idx += 1
                            else:
                                results[_prev_fi] = b""
                        else:
                            # 真正空帧：记录诊断
                            self.__dict__.setdefault('_diag_empty', 0)
                            self._diag_empty += 1
                            _la_buf = getattr(self, '_la_buffered', 0)
                            if self._diag_empty <= 5 or self._diag_empty % 50 == 0:
                                print(f'[NVENC-Enc] ⚠️ 空帧 #{self._diag_empty} (ce-pipe '
                                      f'fi={_prev_fi} slot={slot_idx} ep_s={_prev_ep_s} '
                                      f'bs_s={bs_status} la_buf={_la_buf})', flush=True)
                        _slot_pending[slot_idx] = None

                    # ═══════════════════════════════════════════════
                    # Phase 2: Submit new frame with per-frame CE
                    # ═══════════════════════════════════════════════
                    # LockInputBuffer
                    lock_buf = (c_uint8 * 1544)()
                    ctypes.memset(lock_buf, 0, 1544)
                    cast(lock_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_INPUT_BUFFER_VER
                    cast(byref(lock_buf, 8), ctypes.POINTER(c_void_p))[0] = slot['input_buf']

                    lock_addr = self._func_ptrs[_FUNC_IDX["LockInputBuffer"]]
                    s = _LockIB(lock_addr)(self._encoder, lock_buf)
                    if s != 0:
                        raise RuntimeError("[NVENCEncoder] LockInputBuffer[%d] failed, code=%d" %
                                           (slot_idx, s))

                    _raw_map = cast(byref(lock_buf, 16), ctypes.POINTER(c_void_p))[0]
                    mapped_ptr = _raw_map if isinstance(_raw_map, int) else (_raw_map.value or 0)
                    actual_pitch = cast(byref(lock_buf, 24), ctypes.POINTER(c_uint32))[0]

                    if not mapped_ptr:
                        _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                            self._encoder, slot['input_buf'])
                        raise RuntimeError("[NVENCEncoder] LockInputBuffer[%d] returned NULL mapped ptr" %
                                           slot_idx)

                    # ── GPU→GPU copy (cuMemcpy2D, pitch-aware) ──
                    _cpy2d = (c_uint8 * 128)()
                    ctypes.memset(_cpy2d, 0, 128)
                    src_ptr = nv12_tensors[fi].data_ptr()
                    cast(byref(_cpy2d, 16), ctypes.POINTER(c_uint32))[0] = _CUDA_DEV
                    cast(byref(_cpy2d, 32), ctypes.POINTER(c_void_p))[0] = c_void_p(src_ptr)
                    cast(byref(_cpy2d, 48), ctypes.POINTER(c_size_t))[0] = W
                    cast(byref(_cpy2d, 72), ctypes.POINTER(c_uint32))[0] = _CUDA_DEV
                    cast(byref(_cpy2d, 88), ctypes.POINTER(c_void_p))[0] = c_void_p(mapped_ptr)
                    cast(byref(_cpy2d, 104), ctypes.POINTER(c_size_t))[0] = (
                        actual_pitch if actual_pitch > 0 else W)
                    cast(byref(_cpy2d, 112), ctypes.POINTER(c_size_t))[0] = W
                    cast(byref(_cpy2d, 120), ctypes.POINTER(c_size_t))[0] = nv12_h
                    r = self._libcuda.cuMemcpy2D_v2(cast(_cpy2d, c_void_p))
                    if r != 0:
                        _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                            self._encoder, slot['input_buf'])
                        raise RuntimeError("[NVENCEncoder] cuMemcpy2D[%d] failed, code=%d" %
                                           (slot_idx, r))

                    # ── UnlockInputBuffer ──
                    _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                        self._encoder, slot['input_buf'])

                    # [FIX-LA-SYNC] LA>0 时跳过 CE 创建（同步编码）
                    _ce = c_void_p(None)
                    if self._la_depth == 0:
                        self._libcuda.cuEventCreate.restype = c_uint32
                        self._libcuda.cuEventCreate.argtypes = [ctypes.POINTER(c_void_p), c_uint32]
                        self._libcuda.cuEventCreate(ctypes.byref(_ce), 0)

                    # ── EncodePicture with per-frame completionEvent ──
                    pic_buf = (c_uint8 * 3360)()
                    ctypes.memset(pic_buf, 0, 3360)
                    cast(pic_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
                    cast(byref(pic_buf, 4), ctypes.POINTER(c_uint32))[0] = W
                    cast(byref(pic_buf, 8), ctypes.POINTER(c_uint32))[0] = self._height
                    cast(byref(pic_buf, 12), ctypes.POINTER(c_uint32))[0] = (
                        actual_pitch if actual_pitch > 0 else W)
                    cast(byref(pic_buf, 24), ctypes.POINTER(c_uint64))[0] = self._frame_idx
                    cast(byref(pic_buf, 40), ctypes.POINTER(c_void_p))[0] = slot['input_buf']
                    cast(byref(pic_buf, 48), ctypes.POINTER(c_void_p))[0] = slot['bs_buf']
                    # [FIX-LA-SYNC] LA>0 时禁用 CE，匹配测试脚本验证模式
                    if self._la_depth == 0:
                        cast(ctypes.byref(pic_buf, 56), ctypes.POINTER(c_void_p))[0] = _ce
                    cast(byref(pic_buf, 64), ctypes.POINTER(c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12
                    cast(byref(pic_buf, 68), ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_STRUCT_FRAME
                    if force_idr:
                        cast(byref(pic_buf, 16), ctypes.POINTER(c_uint32))[0] = 0x2

                    encode_picture = _NvEncEncodePictureProto(
                        self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                    _ep_status = encode_picture(self._encoder,
                                                cast(pic_buf, ctypes.POINTER(_NvEncPicParams)))
                    self._frame_idx += 1

                    if _ep_status != NV_ENC_SUCCESS and _ep_status != NV_ENC_ERR_NEED_MORE_INPUT:
                        raise RuntimeError("[NVENCEncoder] EncodePicture[%d] failed, code=%d" %
                                           (slot_idx, _ep_status))

                    # Handle LA buffering: encoder needs more frames before producing output.
                    # [FIX-LA-NEEDMORE-PENDING] NEED_MORE_INPUT frames are also recorded
                    # so Phase 1/3 correctly harvest them (returning b""). Aligned with
                    # test_nvenc_completion_event_v4.py NVENCEncoderMode5.
                    if _ep_status == NV_ENC_ERR_NEED_MORE_INPUT:
                        results[fi] = b""
                        self.__dict__.setdefault('_la_buffered', 0)
                        self._la_buffered += 1
                        if self._la_buffered <= 3 or self._la_buffered == self._la_depth:
                            print(f'[NVENC-Enc] 前向帧预看缓冲中 ({self._la_buffered}/{self._la_depth})',
                                  flush=True)

                    # Save pending state for harvest on next slot rotation.
                    # Tuple: (ce_handle, frame_index, ep_status, is_idr, bs_buf)
                    _slot_pending[slot_idx] = (_ce, fi, _ep_status, force_idr, slot['bs_buf'])

                    # [FIX-LA-INLINE-DRAIN] SDK 合规：每送入一帧后立即排空所有
                    # 已就绪的输出（LA>0 时帧产出可能早于 CE 轮转）。按
                    # _output_slot_idx 顺序循环 blocking LockBitstream，
                    # 收集 LA 管道中已完成但尚未被 Phase 1 harvest 的帧。
                    # 参照 tests/test_nvenc_la_frame_conservation.py _drain_outputs()。
                    if self._la_depth > 0:
                        _inlined = self._drain_outputs_blocking()
                        for _est_fi, _h264_data in _inlined:
                            # [FIX-LA-INDEX] 通过 _slot_pending 获取实际帧索引
                            _drain_slot = _est_fi % pd
                            _pending = _slot_pending[_drain_slot]
                            if _pending is None:
                                continue
                            _, _actual_fi, _, _actual_idr, _ = _pending
                            # [FIX-LA-INLINE-OVERWRITE] LA 缓冲帧被预设为 b""
                            if _actual_fi < n_frames and (results[_actual_fi] is None or results[_actual_fi] == b""):
                                # [FIX-LA-SPS-PPS] inline drain 必须同步做 SPS/PPS 缓存
                                if _actual_idr and self._cached_sps_pps is not None:
                                    _h264_data = self._cached_sps_pps + _h264_data
                                elif _actual_idr and self._cached_sps_pps is None:
                                    self._cached_sps_pps = self._extract_sps_pps(_h264_data)
                                    if self._cached_sps_pps:
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                              len(self._cached_sps_pps), flush=True)
                                        if self._muxer_ref is not None:
                                            try:
                                                self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                                self._sps_pps_injected = True
                                            except Exception:
                                                pass
                                elif self._cached_sps_pps is None:
                                    self._cached_sps_pps = self._extract_sps_pps(_h264_data)
                                    if self._cached_sps_pps:
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                              len(self._cached_sps_pps), flush=True)
                                results[_actual_fi] = _h264_data
                                # [FIX-LA-INLINE-CLEAR] 清除 _slot_pending
                                _slot_pending[_drain_slot] = None

                # [FIX-LA-EOS] LA>0 时在 Phase 3 之前发送 EOS，强制 NVENC
                # 排空前向预看缓冲区，否则 blocking retry 永远等不到数据。
                if self._la_depth > 0 and not self._eos_sent:
                    _eos_pic = (c_uint8 * 3360)()
                    ctypes.memset(_eos_pic, 0, 3360)
                    cast(_eos_pic, ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
                    cast(ctypes.byref(_eos_pic, 16), ctypes.POINTER(c_uint32))[0] = 0x8
                    cast(ctypes.byref(_eos_pic, 40), ctypes.POINTER(c_void_p))[0] = c_void_p(None)
                    cast(ctypes.byref(_eos_pic, 48), ctypes.POINTER(c_void_p))[0] = self._slots[0]['bs_buf']
                    _eos_ep = _NvEncEncodePictureProto(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                    _eos_ep(self._encoder, cast(_eos_pic, ctypes.POINTER(_NvEncPicParams)))
                    self._eos_sent = True

                # ═══════════════════════════════════════════════
                # Phase 3: Drain remaining pending slots
                # ═══════════════════════════════════════════════
                for slot_idx in range(pd):
                    if _slot_pending[slot_idx] is not None:
                        _pending_ce, _pending_fi, _pending_ep_s, _pending_idr, _pending_bs = \
                            _slot_pending[slot_idx]
                        if _pending_ce.value is not None:
                            self._libcuda.cuEventSynchronize.restype = c_uint32
                            self._libcuda.cuEventSynchronize.argtypes = [c_void_p]
                            self._libcuda.cuEventSynchronize(_pending_ce)
                            self._libcuda.cuEventDestroy.restype = c_uint32
                            self._libcuda.cuEventDestroy.argtypes = [c_void_p]
                            self._libcuda.cuEventDestroy(_pending_ce)
                        h264_data, bs_status = self._lock_bitstream_with_retry(_pending_bs)
                        if h264_data:
                            # [SEGMENT-REUSE] SPS/PPS caching
                            if _pending_idr and self._cached_sps_pps is not None:
                                h264_data = self._cached_sps_pps + h264_data
                            elif _pending_idr and self._cached_sps_pps is None and h264_data:
                                self._cached_sps_pps = self._extract_sps_pps(h264_data)
                                if self._cached_sps_pps:
                                    print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                          len(self._cached_sps_pps), flush=True)
                                    # [FIX-SPS-PPS-V3] 首帧已含 NVENC 初始化 SPS+PPS, 不 prepend, 仅预注入 muxer
                                    if self._muxer_ref is not None:
                                        try:
                                            self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                            self._sps_pps_injected = True
                                        except Exception:
                                            pass
                            elif not _pending_idr and self._cached_sps_pps is None and h264_data:
                                self._cached_sps_pps = self._extract_sps_pps(h264_data)
                                if self._cached_sps_pps:
                                    print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                          len(self._cached_sps_pps), flush=True)
                            results[_pending_fi] = h264_data
                            # [FIX-LA-OUTPTR] Phase 3 drain 成功 → 推进输出指针
                            self._output_slot_idx += 1
                        elif _pending_ep_s == NV_ENC_ERR_NEED_MORE_INPUT:
                            # [FIX-LA-BLKRETRY-ALWAYS] 同 Phase 1：无论 bs_status
                            # 为何值都尝试 blocking LockBitstream。
                            _h264_blk, _ = self._lock_bitstream_blocking(_pending_bs, timeout_ms=5000)
                            if _h264_blk:
                                # [FIX-LA-SPS-BLKRETRY-PH3] Phase 3 blocking retry
                                # 取回的 IDR 帧同样需要 SPS/PPS 缓存
                                if _pending_idr and self._cached_sps_pps is None:
                                    self._cached_sps_pps = self._extract_sps_pps(_h264_blk)
                                    if self._cached_sps_pps:
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                              len(self._cached_sps_pps), flush=True)
                                        if self._muxer_ref is not None:
                                            try:
                                                self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                                self._sps_pps_injected = True
                                            except Exception:
                                                pass
                                elif self._cached_sps_pps is None:
                                    self._cached_sps_pps = self._extract_sps_pps(_h264_blk)
                                    if self._cached_sps_pps:
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                              len(self._cached_sps_pps), flush=True)
                                results[_pending_fi] = _h264_blk
                                # [FIX-LA-OUTPTR] blocking 取回成功 → 推进输出指针
                                self._output_slot_idx += 1
                            else:
                                results[_pending_fi] = b""
                        # else: None stays — real empty frame
                        _slot_pending[slot_idx] = None

                # [FIX-LA-REDRAIN] 二次排空安全网：CE 可能因 LA 提前触发
                # 而导致 NEED_MORE_INPUT 帧在 Phase 1/3 被误标为 b""（CE
                # 在帧入队时即触发，但编码尚未完成）。按 _output_slot_idx
                # 顺序循环 blocking LockBitstream，收集所有实际已完成但
                # 被遗漏的帧，覆写 results 中的 b""/None 占位符。
                # 参照 tests/test_nvenc_la_frame_conservation.py 的
                # _drain_outputs() 验证模式。
                if self._la_depth > 0:
                    _redrained = self._drain_outputs_blocking()
                    if _redrained:
                        _recovered = 0
                        for _est_fi, _h264_data in _redrained:
                            if _est_fi < n_frames and (results[_est_fi] is None or results[_est_fi] == b""):
                                # [FIX-LA-SPS-REDRAIN] 二次排空安全网取回的帧也需
                                # SPS/PPS 缓存，作为所有路径均已失败的最终兜底
                                if _est_fi == 0 and self._cached_sps_pps is None:
                                    self._cached_sps_pps = self._extract_sps_pps(_h264_data)
                                    if self._cached_sps_pps:
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                              len(self._cached_sps_pps), flush=True)
                                        if self._muxer_ref is not None:
                                            try:
                                                self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                                self._sps_pps_injected = True
                                            except Exception:
                                                pass
                                elif self._cached_sps_pps is None:
                                    self._cached_sps_pps = self._extract_sps_pps(_h264_data)
                                    if self._cached_sps_pps:
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                              len(self._cached_sps_pps), flush=True)
                                results[_est_fi] = _h264_data
                                _recovered += 1
                        if _recovered > 0:
                            print(f'[NVENC-Enc] [FIX-LA-REDRAIN] 二次排空回收 {_recovered} '
                                  f'帧 (LA={self._la_depth}, pd={pd})', flush=True)

                return results
        finally:
            if _need_pop:
                try:
                    _ctx_out = c_void_p()
                    self._libcuda.cuCtxPopCurrent.restype = c_uint32
                    self._libcuda.cuCtxPopCurrent.argtypes = [ctypes.POINTER(c_void_p)]
                    self._libcuda.cuCtxPopCurrent(ctypes.byref(_ctx_out))
                except Exception:
                    pass

    def encode_frame(self, nv12_gpu_tensor, force_idr: bool = False) -> bytes:
        """Encode one NV12 GPU tensor → H.264 ES bytes (synchronous, blocking).

        Uses the single NVENC input/bitstream buffer with blocking EncodePicture +
        LockBitstream. Thread-safe via self._lock, CUDA context protected.

        Args:
            nv12_gpu_tensor: NV12 format GPU tensor (H+H//2, W) uint8
            force_idr: [SEGMENT-REUSE] Force IDR frame to re-emit SPS/PPS for new muxer
        """
        import torch

        # [FIX-GPU-STAY] 跨线程 CUDA context 保护: 确保 Writer 线程调用时
        # NVENC session 的 primary context 已 set current。与 __init__ 中
        # cuCtxPushCurrent(self._saved_ctx) 配对。
        # [FIX-ENC-CTX] _need_pop 依赖 cuCtxPushCurrent 实际返回值。
        _need_pop = False
        _primary = getattr(self, '_primary_ctx', None)
        if _primary is not None and _primary.value is not None:
            try:
                self._libcuda.cuCtxPushCurrent.restype = c_uint32
                self._libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
                _r_push = self._libcuda.cuCtxPushCurrent(_primary)
                _need_pop = (_r_push == 0)  # ✅ [FIX-ENC-CTX]
            except Exception:
                pass

        try:
            with self._lock:
                if self._encoder.value is None:
                    raise RuntimeError("[NVENCEncoder] Encoder not initialized or already closed")
                if self._input_buf_handle.value is None or self._bs_handle.value is None:
                    raise RuntimeError("[NVENCEncoder] Buffers not initialized")

                assert nv12_gpu_tensor.is_cuda
                assert nv12_gpu_tensor.dtype == torch.uint8
                assert nv12_gpu_tensor.is_contiguous()

                nv12_h = self._height + self._height // 2
                W = self._width

                # ── LockInputBuffer → get mapped pointer ──
                _LockInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _UnlockInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)

                lock_buf = (c_uint8 * 1544)()
                ctypes.memset(lock_buf, 0, 1544)
                cast(lock_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_INPUT_BUFFER_VER  # version@0
                cast(byref(lock_buf, 8), ctypes.POINTER(c_void_p))[0] = self._input_buf_handle  # inputBuffer@8

                lock_addr = self._func_ptrs[_FUNC_IDX["LockInputBuffer"]]
                s = _LockInputBufferProto(lock_addr)(self._encoder, lock_buf)
                if s != 0:
                    raise RuntimeError("[NVENCEncoder] LockInputBuffer failed, code=%d" % s)

                _raw_map = cast(byref(lock_buf, 16), ctypes.POINTER(c_void_p))[0]  # bufferDataPtr@16
                mapped_ptr = _raw_map if isinstance(_raw_map, int) else (_raw_map.value or 0)
                actual_pitch = cast(byref(lock_buf, 24), ctypes.POINTER(c_uint32))[0]  # pitch@24

                if not mapped_ptr:
                    _UnlockInputBufferProto(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                        self._encoder, self._input_buf_handle)
                    raise RuntimeError("[NVENCEncoder] LockInputBuffer returned NULL mapped ptr")

                # ── GPU→GPU copy (cuMemcpy2D, pitch-aware) ──
                # CUDA_MEMCPY2D (128B): srcX@0,srcY@8,srcMemType@16,srcHost@24,
                #   srcDevice@32,srcArray@40,srcPitch@48,
                #   dstX@56,dstY@64,dstMemType@72,dstHost@80,
                #   dstDevice@88,dstArray@96,dstPitch@104,
                #   WidthInBytes@112,Height@120
                _CU_MEMORYTYPE_DEVICE = 2
                _cpy2d = (c_uint8 * 128)()
                ctypes.memset(_cpy2d, 0, 128)
                src_ptr = nv12_gpu_tensor.data_ptr()
                cast(byref(_cpy2d, 16), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
                cast(byref(_cpy2d, 32), ctypes.POINTER(c_void_p))[0] = c_void_p(src_ptr)
                cast(byref(_cpy2d, 48), ctypes.POINTER(c_size_t))[0] = W
                cast(byref(_cpy2d, 72), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
                cast(byref(_cpy2d, 88), ctypes.POINTER(c_void_p))[0] = c_void_p(mapped_ptr)
                cast(byref(_cpy2d, 104), ctypes.POINTER(c_size_t))[0] = (
                    actual_pitch if actual_pitch > 0 else W)
                cast(byref(_cpy2d, 112), ctypes.POINTER(c_size_t))[0] = W
                cast(byref(_cpy2d, 120), ctypes.POINTER(c_size_t))[0] = nv12_h
                r = self._libcuda.cuMemcpy2D_v2(cast(_cpy2d, c_void_p))
                if r != 0:
                    _UnlockInputBufferProto(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                        self._encoder, self._input_buf_handle)
                    raise RuntimeError("[NVENCEncoder] cuMemcpy2D failed, code=%d" % r)

                # ── UnlockInputBuffer ──
                _UnlockInputBufferProto(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                    self._encoder, self._input_buf_handle)

                # ── EncodePicture (byte array, verified offsets) ──
                # NV_ENC_PIC_PARAMS (3360B): version@0, inputWidth@4, inputHeight@8, inputPitch@12,
                #   encodePicFlags@16, frameIdx@20, inputTimeStamp@24, inputDuration@32,
                #   inputBuffer@40, outputBitstream@48, completionEvent@56,
                #   bufferFmt@64, pictureStruct@68, pictureType@72, codecPicParams@76
                pic_buf = (c_uint8 * 3360)()
                ctypes.memset(pic_buf, 0, 3360)
                cast(pic_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER        # version@0
                cast(byref(pic_buf, 4), ctypes.POINTER(c_uint32))[0] = W                  # inputWidth@4
                cast(byref(pic_buf, 8), ctypes.POINTER(c_uint32))[0] = self._height        # inputHeight@8 (luma height only for NV12)
                cast(byref(pic_buf, 12), ctypes.POINTER(c_uint32))[0] = (
                    actual_pitch if actual_pitch > 0 else W)                               # inputPitch@12
                cast(byref(pic_buf, 24), ctypes.POINTER(c_uint64))[0] = self._frame_idx   # inputTimeStamp@24
                cast(byref(pic_buf, 40), ctypes.POINTER(c_void_p))[0] = self._input_buf_handle  # inputBuffer@40
                cast(byref(pic_buf, 48), ctypes.POINTER(c_void_p))[0] = self._bs_handle   # outputBitstream@48
                cast(byref(pic_buf, 64), ctypes.POINTER(c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12  # bufferFmt@64
                cast(byref(pic_buf, 68), ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_STRUCT_FRAME   # pictureStruct@68
                if force_idr:
                    cast(byref(pic_buf, 16), ctypes.POINTER(c_uint32))[0] = 0x2  # [SEGMENT-REUSE] NV_ENC_PIC_FLAG_FORCEIDR

                # ★ completionEvent: 创建 CUDA event 并设置到 pic_buf offset 56 ★
                _ce = c_void_p(None)
                self._libcuda.cuEventCreate.restype = c_uint32
                self._libcuda.cuEventCreate.argtypes = [ctypes.POINTER(c_void_p), c_uint32]
                self._libcuda.cuEventCreate(ctypes.byref(_ce), 0)
                cast(ctypes.byref(pic_buf, 56), ctypes.POINTER(c_void_p))[0] = _ce

                encode_picture = _NvEncEncodePictureProto(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                status = encode_picture(self._encoder, cast(pic_buf, ctypes.POINTER(_NvEncPicParams)))

                self._frame_idx += 1

                # ★ cuEventSynchronize — 等待 NVENC 硬件完成编码 ★
                if _ce.value is not None:
                    self._libcuda.cuEventSynchronize.restype = c_uint32
                    self._libcuda.cuEventSynchronize.argtypes = [c_void_p]
                    self._libcuda.cuEventSynchronize(_ce)
                    self._libcuda.cuEventDestroy.restype = c_uint32
                    self._libcuda.cuEventDestroy.argtypes = [c_void_p]
                    self._libcuda.cuEventDestroy(_ce)

                h264_data = b""

                # [FIX-LA-DRAIN-SYNC] ★ 全局 drain：循环 LockBitstream 排空所有已完成 slot
                _drained = self._drain_outputs_blocking()
                if _drained:
                    h264_data = _drained[0][1]
                if not h264_data:
                    h264_data = b""

                # ── LockBitstream with retry (Tier 3-E) ──
                # 带指数退避重试，应对 NVENC HW DMA 竞态导致的瞬时空帧
                h264_data, bs_status = self._lock_bitstream_with_retry(self._bs_handle)

                if status == NV_ENC_ERR_NEED_MORE_INPUT:
                    # lookahead: encoder needs more frames to fill the lookahead window.
                    # This is expected for the first N frames where N = lookahead depth.
                    # The frame was accepted; output will be produced once the window is full.
                    self.__dict__.setdefault('_la_buffered', 0)
                    self._la_buffered += 1
                    if self._la_buffered <= 3 or self._la_buffered == self._la_depth:
                        print(f'[NVENC-Enc] 前向帧预看缓冲中 ({self._la_buffered}/{self._la_depth})',
                              flush=True)
                    # Return empty bytes — frame accepted, no output yet
                    return b""
                elif not h264_data:
                    # Tier 0: 空帧诊断 — 记录 NV12 输入统计以区分根因类别
                    self.__dict__.setdefault('_diag_empty', 0)
                    self._diag_empty += 1
                    _nv12_mean = float(nv12_gpu_tensor.float().mean())
                    _nv12_std  = float(nv12_gpu_tensor.float().std())
                    _nv12_min  = int(nv12_gpu_tensor.min())
                    _nv12_max  = int(nv12_gpu_tensor.max())
                    if self._diag_empty <= 5 or self._diag_empty % 50 == 0:
                        print(f'[NVENC-Enc] ⚠️ 空帧 #{self._diag_empty} (encode_frame) '
                              f'frame_idx={self._frame_idx - 1} force_idr={force_idr} '
                              f'nv12_mean={_nv12_mean:.1f} std={_nv12_std:.1f} '
                              f'min={_nv12_min} max={_nv12_max}', flush=True)

                if status != NV_ENC_SUCCESS:
                    raise RuntimeError("[NVENCEncoder] EncodePicture failed, code=%d" % status)

                # [SEGMENT-REUSE] 首段首次编码时缓存 SPS+PPS，后续段 force_idr 帧预挂
                if force_idr and self._cached_sps_pps is not None:
                    h264_data = self._cached_sps_pps + h264_data
                elif force_idr and self._cached_sps_pps is None and h264_data:
                    self._cached_sps_pps = self._extract_sps_pps(h264_data)
                    if self._cached_sps_pps:
                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" % len(self._cached_sps_pps),
                              flush=True)
                        # [FIX-SPS-PPS-V3] 首帧已含 NVENC 初始化 SPS+PPS, 不 prepend, 仅预注入 muxer
                        if self._muxer_ref is not None:
                            try:
                                self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                self._sps_pps_injected = True
                            except Exception:
                                pass
                elif not force_idr and self._cached_sps_pps is None and h264_data:
                    self._cached_sps_pps = self._extract_sps_pps(h264_data)
                    if self._cached_sps_pps:
                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" % len(self._cached_sps_pps),
                              flush=True)
                return h264_data
        finally:
            if _need_pop:
                try:
                    _ctx_out = c_void_p()
                    self._libcuda.cuCtxPopCurrent.restype = c_uint32
                    self._libcuda.cuCtxPopCurrent.argtypes = [ctypes.POINTER(c_void_p)]
                    self._libcuda.cuCtxPopCurrent(ctypes.byref(_ctx_out))
                except Exception:
                    pass

    def flush(self) -> bytes:
        # [FIX-FLUSH-CTX] 跨线程 CUDA context 保护：flush() 可能由 Writer 线程
        # （而非 Encode 线程）调用，必须显式 push primary context 确保 NVENC API
        # 在正确上下文执行。与 encode_frames_batch_ce_pipeline() 中的保护逻辑一致。
        _need_pop = False
        _primary = getattr(self, '_primary_ctx', None)
        if _primary is not None and _primary.value is not None:
            try:
                self._libcuda.cuCtxPushCurrent.restype = c_uint32
                self._libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
                _r_push = self._libcuda.cuCtxPushCurrent(_primary)
                _need_pop = (_r_push == 0)
            except Exception:
                pass
        try:
            with self._lock:
                if self._encoder.value is None:
                    return b""

                # [FIX-LA-EOS] 若 CE pipeline 已送过 EOS 则跳过，避免 double-EOS 错误。
                if not self._eos_sent:
                    # Send EOS
                    pic_buf = (c_uint8 * 3360)()
                    ctypes.memset(pic_buf, 0, 3360)
                    cast(pic_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
                    cast(byref(pic_buf, 16), ctypes.POINTER(c_uint32))[0] = 0x8  # NV_ENC_PIC_FLAG_EOS
                    cast(byref(pic_buf, 40), ctypes.POINTER(c_void_p))[0] = c_void_p(None)
                    cast(byref(pic_buf, 48), ctypes.POINTER(c_void_p))[0] = self._bs_handle  # outputBitstream

                    encode_picture = _NvEncEncodePictureProto(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                    encode_picture(self._encoder, cast(pic_buf, ctypes.POINTER(_NvEncPicParams)))
                    self._eos_sent = True

                # Drain ALL slot bitstream buffers after EOS.
                # 4-slot 轮转下每个 slot 的 lookahead 滞留帧都需排空，
                # 仅排空 slot 0 会导致 slot 1/2/3 的帧永久丢失 (-6~-7 帧)。
                result_parts = []
                _LockBitstreamProto_raw = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _total_recovered_frames = 0
                _total_recovered_bytes = 0

                # [FIX-LA-FLUSH] 按 _output_slot_idx 起始顺序排空所有 slot，
                # 保证 EOS flush 的输出顺序与编码器内部输出顺序严格一致。
                # 参照 tests/test_nvenc_la_frame_conservation.py flush_eos()。
                _start_slot = self._output_slot_idx % self._slot_count
                _drain_order = [(_start_slot + i) % self._slot_count
                                for i in range(self._slot_count)]
                for _slot_idx in _drain_order:
                    _slot = self._slots[_slot_idx]
                    _bs_handle = _slot['bs_buf']
                    _slot_parts = []
                    # [FIX-LA-FLUSH] 首次 blocking Lock (doNotWait=0)，
                    # 后续用 non-blocking (doNotWait=1) 加速排空。
                    _first_lock = True
                    while True:
                        lock_raw = (c_uint8 * 1544)()
                        ctypes.memset(lock_raw, 0, 1544)
                        cast(lock_raw, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
                        cast(byref(lock_raw, 4), ctypes.POINTER(c_uint32))[0] = (
                            0 if _first_lock else 1)
                        cast(byref(lock_raw, 8), ctypes.POINTER(c_void_p))[0] = _bs_handle

                        lock_bs_fn = _LockBitstreamProto_raw(self._func_ptrs[_FUNC_IDX["LockBitstream"]])
                        bs_status = lock_bs_fn(self._encoder, lock_raw)
                        if bs_status == NV_ENC_ERR_NEED_MORE_INPUT:
                            break
                        if bs_status != NV_ENC_SUCCESS:
                            break

                        bitstream_size = cast(byref(lock_raw, 36), ctypes.POINTER(c_uint32))[0]
                        if bitstream_size == 0:
                            _NvEncUnlockBitstreamProto(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(
                                self._encoder, _bs_handle)
                            break

                        _raw_bsptr = cast(byref(lock_raw, 56), ctypes.POINTER(c_void_p))[0]
                        bitstream_ptr_val = _raw_bsptr if isinstance(_raw_bsptr, int) else (_raw_bsptr.value or 0)
                        if bitstream_ptr_val:
                            buf_type = c_uint8 * bitstream_size
                            _slot_parts.append(bytes(buf_type.from_address(bitstream_ptr_val)))

                        _NvEncUnlockBitstreamProto(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(
                            self._encoder, _bs_handle)
                        _first_lock = False

                    if _slot_parts:
                        _slot_bytes = sum(len(p) for p in _slot_parts)
                        _total_recovered_frames += len(_slot_parts)
                        _total_recovered_bytes += _slot_bytes
                        print(f'[NVENC-FLUSH] slot[{_slot_idx}] drained {len(_slot_parts)} frames, '
                              f'{_slot_bytes} bytes (SDK-compliant EOS drain)', flush=True)
                    result_parts.extend(_slot_parts)

                if _total_recovered_frames > 0:
                    print(f'[NVENC-FLUSH] total drained: {_total_recovered_frames} frames, '
                          f'{_total_recovered_bytes} bytes (SDK-compliant LA EOS flush — valid encoded frames)', flush=True)

                # NVENC EOS flush 排出的帧是有效编码帧（SDK 合规排空逻辑下帧数守恒）。
                # 已修正：旧版误认为花屏残片（错误排空导致缓冲区覆盖），正确实现下可正常使用。
                self._flush_frame_count = _total_recovered_frames
                return b"".join(result_parts)
        finally:
            if _need_pop:
                try:
                    _ctx_out = c_void_p()
                    self._libcuda.cuCtxPopCurrent.restype = c_uint32
                    self._libcuda.cuCtxPopCurrent.argtypes = [ctypes.POINTER(c_void_p)]
                    self._libcuda.cuCtxPopCurrent(ctypes.byref(_ctx_out))
                except Exception:
                    pass

    def close(self):
        with self._lock:
            if self._encoder.value is None:
                return

            self._destroy_all_slots()

            destroy_addr = self._func_ptrs[_FUNC_IDX["DestroyEncoder"]]
            if destroy_addr:
                _NvEncDestroyEncoderProto(destroy_addr)(self._encoder)
            self._encoder = c_void_p(None)
            self._sps_pps_injected = False   # [FIX-SPS-PPS-V2] 跨段重置换，支持 encoder 复用
            print("[NVENCEncoder] Encoder closed", flush=True)

            # Restore saved context
            if self._saved_ctx.value is not None:
                try:
                    self._libcuda.cuCtxPushCurrent.restype = c_uint32
                    self._libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
                    self._libcuda.cuCtxPushCurrent(self._saved_ctx)
                except Exception:
                    pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def frame_count(self) -> int:
        return self._frame_idx


# ==============================================================================
# [FIX-ENC-THREAD] 独立 NVENC 编码线程
# 将 NVENC 编码（encode_frames_batch_ce_pipeline）与 T3 Writer 的 RGB→NV12 kernel 并行化。
# T3 Writer 完成 event.sync + NV12 转换后将帧列表提交到队列，
# 编码线程独立调用 encode_frames_batch_ce_pipeline + muxer.write，
# 两者在 T4 SM（NV12 kernel）与 NVENC 固定功能硬件（编码）上真正并行。
# ==============================================================================

class _NVENCEncodeThread:
    """独立 NVENC 编码线程（GPU_RAW Level 1 路径专用）。

    设计约束：
      · T3 Writer 在调用 submit() 之前必须先调用
        torch.cuda.current_stream().synchronize()，确保 NV12 GPU tensor
        数据已完全写入 VRAM，防止编码线程 cuMemcpy2D 读到未完成的数据（静默花帧）。
      · encode_queue_depth=4：限制 VRAM 中积压的 NV12 tensor 批次数。
        T4 16GB 下每批约 300MB，depth=4 最多多占 ~1.2GB，不挤压 T2 推理空间。
      · 编码线程是 FFmpegMuxer pipe 的唯一写入者（GPU_RAW 路径），保证写入顺序正确。
      · 编码线程为 daemon 线程；正常路径通过 flush_and_join() 完成收尾。
    """

    _SENTINEL = object()

    def __init__(self, nvenc_encoder, writer, encode_queue_depth: int = 4):
        self._nvenc   = nvenc_encoder
        self._writer  = writer
        self._q: queue.Queue = queue.Queue(maxsize=encode_queue_depth)
        self.error: Optional[Exception] = None
        self._written = 0
        self._empty   = 0
        self._prev_h264: Optional[bytes] = None  # Tier 1-A: 空帧补偿用前一帧 H.264 数据
        self._th = threading.Thread(target=self._loop, daemon=True, name='NVENC-Enc')
        self._th.start()

    def submit(self, nv12_list: list, force_idr_first: bool = False):
        """T3 Writer 线程调用：提交一批 NV12 GPU tensor 给编码线程。

        ⚠️ 调用前必须已完成 torch.cuda.current_stream().synchronize()，
        确保 NV12 tensor 写入完成，防止编码线程 cuMemcpy2D 读到 GPU 未写完的数据。
        """
        if self.error is not None:
            raise self.error
        self._q.put((nv12_list, force_idr_first))

    def _loop(self):
        """编码线程主循环：阻塞等待帧批次，CE 异步流水线编码后写入 muxer。"""
        # [FIX-ENC-CTX] daemon 线程启动时 CUDA context stack 为空；
        # encode_frames_batch_ce_pipeline 内的 cuCtxPushCurrent 若静默失败会导致
        # cuMemcpy2D_v2 返回 CUDA_ERROR_INVALID_CONTEXT(201)。
        # 此处用 cuCtxSetCurrent 一次性将 primary context 绑定到本线程，
        # 后续 push/pop 循环在已激活的 context 上正常工作。
        _libcuda = getattr(self._nvenc, '_libcuda', None)
        _primary = getattr(self._nvenc, '_primary_ctx', None)
        if _libcuda is not None and _primary is not None and _primary.value is not None:
            try:
                _libcuda.cuCtxSetCurrent.restype  = c_uint32
                _libcuda.cuCtxSetCurrent.argtypes = [c_void_p]
                _r_set = _libcuda.cuCtxSetCurrent(_primary)  # ✅ [FIX-ENC-CTX]
                if _r_set != 0:
                    print(f'[NVENC-Enc] ⚠️ cuCtxSetCurrent 返回 {_r_set}，'
                          f'编码线程可能缺少有效 CUDA context', flush=True)
            except Exception as _e:
                print(f'[NVENC-Enc] ⚠️ cuCtxSetCurrent 异常: {_e}', flush=True)
        # [FIX-LA-ACCUMULATE] LA>0 时跨批次累积帧，保证 LA 缓冲区连续性。
        # 原实现在每个 batch 内独立调用 encode_frames_batch()，导致：
        #   1) local fi%pd 分配 slot 复位 → 覆盖上一批次的 LA 缓冲帧
        #   2) _slot_pending 是批次局部变量 → 跨批次帧映射失效
        #   3) 批次末尾未排空的 LA 帧被设为 b"" → 静默丢帧
        #   4) 最终 EOS flush 回收的帧被写在 segment 末尾 → 帧序错乱 + GOP 损坏
        # 修复：LA>0 时累积所有帧，segment 结束时一次性编码，保证 LA 连续流。
        # LA=0 继续使用 per-batch ce_pipeline 以利用 CE 异步流水线性能优势。
        _la_mode = (self._nvenc._la_depth > 0)
        _acc_nv12: list = []
        _acc_force_idr = False
        _first_batch = True
        # [FIX-F0-IN-BATCH] 从 encoder 取出暂存的 f0 NV12 tensor
        _pending_f0 = getattr(self._nvenc, '_pending_f0_nv12', None)
        if _pending_f0 is not None:
            self._nvenc._pending_f0_nv12 = None
            _f0_idr = getattr(self._nvenc, '_pending_f0_force_idr', False)
            self._nvenc._pending_f0_force_idr = False
        else:
            _f0_idr = False
        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                break
            nv12_list, force_idr = item
            if _la_mode:
                # LA>0: 累积帧，收到 SENTINEL 后一次性编码
                if _first_batch:
                    _acc_force_idr = force_idr
                    _first_batch = False
                _acc_nv12.extend(nv12_list)
            else:
                # LA=0: per-batch ce_pipeline（原有路径）
                try:
                    h264_list = self._nvenc.encode_frames_batch_ce_pipeline(nv12_list, force_idr)
                    if not self._nvenc._sps_pps_injected:
                        _sps = getattr(self._nvenc, '_cached_sps_pps', None)
                        if _sps:
                            self._writer.write_sps_pps(_sps)
                            self._nvenc._sps_pps_injected = True
                    for i, h264_data in enumerate(h264_list):
                        if h264_data is None:
                            self._empty += 1
                            if self._prev_h264 is not None:
                                self._writer.write(self._prev_h264)
                                self._written += 1
                        elif not h264_data:
                            pass
                        else:
                            self._writer.write(h264_data)
                            self._written += 1
                            self._prev_h264 = h264_data
                except Exception as e:
                    self.error = e
                    return

        # [FIX-LA-ACCUMULATE] LA>0: 编码所有累积帧 + 内置 EOS flush
        if _la_mode and _acc_nv12:
            if _pending_f0 is not None:
                _acc_nv12.insert(0, _pending_f0)
                if _f0_idr:
                    _acc_force_idr = True
            try:
                h264_list = self._nvenc.encode_frames_batch(_acc_nv12, _acc_force_idr,
                                                             send_eos=True)
                if not self._nvenc._sps_pps_injected:
                    _sps = getattr(self._nvenc, '_cached_sps_pps', None)
                    if _sps:
                        self._writer.write_sps_pps(_sps)
                        self._nvenc._sps_pps_injected = True
                for i, h264_data in enumerate(h264_list):
                    if h264_data is None:
                        self._empty += 1
                        if self._prev_h264 is not None:
                            self._writer.write(self._prev_h264)
                            self._written += 1
                    elif not h264_data:
                        self._empty += 1
                        if self._prev_h264 is not None:
                            self._writer.write(self._prev_h264)
                            self._written += 1
                    else:
                        self._writer.write(h264_data)
                        self._written += 1
                        self._prev_h264 = h264_data
            except Exception as e:
                self.error = e
                return

        # EOS flush: LA=0 (ce_pipeline) 路径的残余帧排空。
        if not _la_mode:
            try:
                flush_data = self._nvenc.flush()
                if flush_data and not self._nvenc._eos_sent:
                    self._writer.write(flush_data)
                    self._written += 1
                    self._prev_h264 = flush_data
            except Exception:
                pass

    def flush_and_join(self, timeout: float = 120.0):
        """等待编码线程完成所有已提交帧，并返回 (written, empty)。

        编码线程在 _loop() 末尾已执行 NVENC EOS flush，
        此处仅负责发送 SENTINEL 并 join 线程。
        必须在 SENTINEL 处理后、muxer.close() 之前调用。
        """
        self._q.put(self._SENTINEL)
        self._th.join(timeout=timeout)
        if self._th.is_alive():
            print(f'[NVENC-Enc] ⚠️ 编码线程未在 {timeout:.0f}s 内退出，可能死锁', flush=True)
        if self.error is not None:
            raise RuntimeError(f'[NVENC-Enc] 编码线程异常: {self.error}') from self.error
        # [FIX-ENC-FLUSH] NVENC EOS flush 已移至编码线程 _loop() 末尾执行，
        # 确保与 encode_frames_batch_ce_pipeline 在同一 CUDA context。此处不再重复 flush。
        return self._written, self._empty


# ==============================================================================
# GPU RGB to NV12 color space conversion (PyTorch)
# ==============================================================================

def _rgb_to_nv12_gpu(rgb_tensor, input_is_bgr: bool = False):
    # input: (H, W, 3) uint8 GPU tensor, RGB channel order (or BGR if input_is_bgr=True)
    # output: (H + H//2, W) uint8 GPU tensor, NV12 layout
    import torch

    H, W, C = rgb_tensor.shape
    assert C == 3
    assert rgb_tensor.dtype == torch.uint8
    assert rgb_tensor.is_cuda

    if input_is_bgr:
        r = rgb_tensor[..., 2].float()   # BGR: ch2 = R
        g = rgb_tensor[..., 1].float()   # BGR: ch1 = G
        b = rgb_tensor[..., 0].float()   # BGR: ch0 = B
    else:
        r = rgb_tensor[..., 0].float()
        g = rgb_tensor[..., 1].float()
        b = rgb_tensor[..., 2].float()

    # BT.601 limited range
    Y = (0.257 * r + 0.504 * g + 0.098 * b + 16.0).clamp_(0, 255).round_().to(torch.uint8)

    # 2x2 average downsample for chroma
    h2, w2 = H // 2, W // 2
    def _avg_down(x):
        return (x[:H - H % 2, :W - W % 2].reshape(h2, 2, w2, 2).mean(dim=(1, 3)))

    r_ds = _avg_down(r)
    g_ds = _avg_down(g)
    b_ds = _avg_down(b)

    Cb = (-0.148 * r_ds - 0.291 * g_ds + 0.439 * b_ds + 128.0).clamp_(0, 255).round_().to(torch.uint8)
    Cr = (0.439 * r_ds - 0.368 * g_ds - 0.071 * b_ds + 128.0).clamp_(0, 255).round_().to(torch.uint8)

    # NV12 UV interleave
    UV = torch.empty((h2, W), dtype=torch.uint8, device=rgb_tensor.device)
    UV[:, 0::2] = Cb
    UV[:, 1::2] = Cr

    return torch.cat([Y, UV], dim=0).contiguous()


def _rgb_to_nv12_gpu_batch(rgb_batch, input_is_bgr: bool = False):
    # [FIX-GPU-STAY] 批量 RGB→NV12: (N, H, W, 3) uint8 GPU tensor → (N, H+H//2, W) uint8 GPU tensor
    # 单次 kernel launch 替代 N 次 _rgb_to_nv12_gpu 调用，消除 kernel launch 累积开销
    import torch

    N, H, W, C = rgb_batch.shape
    assert C == 3
    assert rgb_batch.dtype == torch.uint8
    assert rgb_batch.is_cuda

    if input_is_bgr:
        r = rgb_batch[..., 2].float()   # BGR: ch2 = R
        g = rgb_batch[..., 1].float()   # BGR: ch1 = G
        b = rgb_batch[..., 0].float()   # BGR: ch0 = B
    else:
        r = rgb_batch[..., 0].float()
        g = rgb_batch[..., 1].float()
        b = rgb_batch[..., 2].float()

    # BT.601 limited range: (N, H, W)
    Y = (0.257 * r + 0.504 * g + 0.098 * b + 16.0).clamp_(0, 255).round_().to(torch.uint8)

    # 2x2 average downsample: (N, H//2, W//2)
    h2, w2 = H // 2, W // 2
    def _avg_down(x):
        return x[:, :H - H % 2, :W - W % 2].reshape(N, h2, 2, w2, 2).mean(dim=(2, 4))

    r_ds = _avg_down(r)
    g_ds = _avg_down(g)
    b_ds = _avg_down(b)

    Cb = (-0.148 * r_ds - 0.291 * g_ds + 0.439 * b_ds + 128.0).clamp_(0, 255).round_().to(torch.uint8)
    Cr = (0.439 * r_ds - 0.368 * g_ds - 0.071 * b_ds + 128.0).clamp_(0, 255).round_().to(torch.uint8)

    # NV12 UV interleave: (N, h2, W)
    UV = torch.empty((N, h2, W), dtype=torch.uint8, device=rgb_batch.device)
    UV[:, :, 0::2] = Cb
    UV[:, :, 1::2] = Cr

    return torch.cat([Y, UV], dim=1).contiguous()  # (N, H+h2, W)


# ==============================================================================
# FFmpegMuxer -- pure muxer (H.264 ES -> MP4, -c:v copy)
# ==============================================================================

class FFmpegMuxer:
    # Receives H.264 Elementary Stream, pipes to FFmpeg for MP4 muxing only.
    # No re-encoding -- NVENCEncoder already encoded on GPU.

    def __init__(
        self,
        output_path: str,
        fps:    float,
        audio_src: Optional[str] = None,
        ffmpeg_bin: str = "ffmpeg",
        quiet: bool = True,
    ):
        self._error: Optional[Exception] = None
        self._write_count = 0

        cmd = [
            ffmpeg_bin, "-y",
            "-f", "h264",
            "-r", f"{fps:.6f}",
            "-i", "pipe:0",
        ]
        if audio_src:
            cmd += ["-i", audio_src, "-c:a", "copy", "-map", "0:v", "-map", "1:a?"]
        cmd += ["-c:v", "copy"]
        cmd += ["-f", "mp4"]
        cmd += ["-movflags", "faststart"]
        cmd += ["-loglevel", "error"]
        cmd += [output_path]

        if not quiet:
            print("   [FFmpegMuxer] cmd: " + " ".join(cmd), flush=True)

        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        self._stderr_lines: List[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self):
        try:
            for line in self._proc.stderr:
                decoded = line.decode(errors="ignore").rstrip()
                self._stderr_lines.append(decoded)
                if decoded:
                    print("[FFmpegMuxer ERR] " + decoded)
        except Exception:
            pass

    def write_sps_pps(self, sps_pps: bytes):
        """[FIX-SPS-PPS] 在首帧数据前预注入 SPS+PPS NAL 单元。

        FFmpeg -f h264 parser 需要在收到首帧数据前建立解码器上下文。
        当 NVENC repeatSPSPPS 因 LA buffering 时序未能生效时，
        此方法提供软件层面的兜底防御。

        应在 _cached_sps_pps 首次设置后立即调用。"""
        if sps_pps:
            self._proc.stdin.write(sps_pps)
            self._proc.stdin.flush()

    def write(self, h264_es: bytes):
        if self._error is not None:
            raise RuntimeError("FFmpegMuxer error: " + str(self._error)) from self._error
        try:
            if h264_es:
                self._proc.stdin.write(h264_es)
                self._write_count += 1
        except BrokenPipeError:
            self._error = RuntimeError("FFmpeg muxer stdin pipe broken")
            raise self._error

    def close(self):
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._stderr_thread.join(timeout=5)
        rc = self._proc.returncode
        if rc is not None and rc != 0:
            stderr_out = "\\n".join(self._stderr_lines[-20:])
            print("\\n[FFmpegMuxer Warning] FFmpeg exit=%d, stderr: %s" % (rc, stderr_out[:400]))
        if self._error:
            print("[FFmpegMuxer Warning] write error: " + str(self._error))


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
        if stable_std > 25.0 or low_frac > 0.30:
            pair_q = min(8, current_pair_q + 2)
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


_X264_PRESET_FACTOR = {
    'ultrafast': 8.0, 'superfast': 6.0, 'veryfast': 4.0,
    'faster': 2.5, 'fast': 2.0, 'medium': 1.0,
    'slow': 0.4, 'slower': 0.2, 'veryslow': 0.1,
}
# [FIX-NVENC-PRESET] x264 → NVENC preset 名称映射。
# NVENC 使用 p1(最快)~p7(最慢) 命名体系，与 x264 的 ultrafast~veryslow 不兼容。
# 当用户通过 --x264-preset 传入 x264 风格名称 + --codec h264_nvenc 时自动转换。
_X264_TO_NVENC_PRESET = {
    'ultrafast': 'p1', 'superfast': 'p1', 'veryfast': 'p2',
    'faster': 'p3', 'fast': 'p3', 'medium': 'p4',
    'slow': 'p5', 'slower': 'p6', 'veryslow': 'p7',
}
# [FIX-CRF0-CALIB] crf=0（lossless）实测校准因子。
# 理论模型（crf_factor = 2^((0-23)/12) ≈ 0.264）严重低估 lossless 编码成本：
#   · lossless 需维持精确像素，内存带宽和预测搜索开销远高于有损编码
#   · 实测（T4, libx264, ultrafast, 416×736, 8c）: ~150 fps output
#   · 理论估算（修正前）:  ~2860 fps → 偏差约 19×
# 乘以此因子后估算 ~157 fps，贴近实测正常（非热节流）状态。
_CRF0_X264_CALIB_FACTOR: float = 0.055
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

# ─────────────────────────────────────────────────────────────────────────────
# [FIX-SLICE-THREAD] 编码并行度自动探测
# ─────────────────────────────────────────────────────────────────────────────

def _detect_encode_parallelism(n_threads_hint: Optional[int] = None) -> dict:
    """
    [FIX-SLICE-THREAD] 自动探测 CPU / 内存资源，返回最优软编码并行参数字典。

    返回字段
    ─────────────────────────────────────────────────────────────────────────
    cpu_logical    : int    逻辑核心数（含超线程，来自 os.cpu_count()）
    cpu_physical   : int    物理核心数（来自 /proc/cpuinfo；失败则 logical//2）
    mem_avail_gb   : float  系统当前可用内存 GiB（来自 /proc/meminfo MemAvailable）
    encode_threads : int    软编码线程数（x264/x265 frame-level parallelism），
                            = min(cpu_logical, 16)，超过 16 收益递减
    slices         : int    x264 intra-frame 分片数（slice-based threading）：
                            每片由独立线程并行编码，降低单帧编码延迟。
                            = min(encode_threads, 16)，同时受内存可用量约束
                            （大分片数需更多行缓冲区；低分辨率时此约束通常不触发）
    ffmpeg_threads : int    FFmpeg 全局 -threads 值，用于 demux/filter graph
                            = min(cpu_logical, 8)
    """
    cpu_logical = os.cpu_count() or 4

    # 物理核心数：从 /proc/cpuinfo 读 "core id" 去重；失败则估算
    cpu_physical = max(cpu_logical // 2, 1)
    try:
        _core_ids: set = set()
        _pkg_ids:  set = set()
        _cur_pkg       = None
        with open('/proc/cpuinfo', 'r') as _cpuf:
            for _line in _cpuf:
                _line = _line.strip()
                if _line.startswith('physical id'):
                    _cur_pkg = _line.split(':', 1)[1].strip()
                elif _line.startswith('core id') and _cur_pkg is not None:
                    _core_ids.add((_cur_pkg, _line.split(':', 1)[1].strip()))
        if _core_ids:
            cpu_physical = len(_core_ids)
    except Exception:
        pass

    # 系统可用内存（GiB）：读 /proc/meminfo MemAvailable；失败时尝试 psutil
    mem_avail_gb = 4.0
    try:
        with open('/proc/meminfo', 'r') as _memf:
            for _line in _memf:
                if _line.startswith('MemAvailable:'):
                    mem_avail_gb = int(_line.split()[1]) / (1024.0 ** 2)
                    break
    except Exception:
        try:
            import psutil as _psutil
            mem_avail_gb = _psutil.virtual_memory().available / (1024.0 ** 3)
        except ImportError:
            pass

    # 若外部传入 hint，直接使用（但仍不超过 16）
    if n_threads_hint is not None and n_threads_hint > 0:
        encode_threads = min(n_threads_hint, 16)
    else:
        encode_threads = min(cpu_logical, 16)

    # 分片数 = encode_threads（1 slice/thread），但：
    #   · 上限 16：slice 数越多压缩率越低（片间参考受限），16 是实用阈值
    #   · 内存约束：每个 slice 约需 0.25 GiB 额外行缓冲（高分辨率下），低分辨率（<1080p）可忽略
    #   · 下限 2：至少 2 片才有并行效果
    slices_by_cpu  = encode_threads
    slices_by_mem  = max(2, int(mem_avail_gb / 0.25))   # 每 slice 估算 0.25 GiB
    slices = max(2, min(slices_by_cpu, slices_by_mem, 16))

    ffmpeg_threads = min(cpu_logical, 8)

    return {
        'cpu_logical':    cpu_logical,
        'cpu_physical':   cpu_physical,
        'mem_avail_gb':   mem_avail_gb,
        'encode_threads': encode_threads,
        'slices':         slices,
        'ffmpeg_threads': ffmpeg_threads,
    }


# ─────────────────────────────────────────────────────────────────────────────
# [FIX-NVENC-PIPE] NVENC pipe 模式参数常量
# ─────────────────────────────────────────────────────────────────────────────

# NVENC 内部帧缓冲数（-surfaces N）：
#   NVENC 硬件编码器内部维护一个帧槽池（surfaces），每个 slot 存储一帧正在被硬件
#   编码的图像。默认值为 8，对于均匀帧率的文件输入已经足够；但 pipe 输入存在速率
#   抖动（T3-Writer _write_loop 批量写入 + T2-Infer 批量 D2H），当短时供帧速率超过
#   硬件编码速率时，较小的 surfaces 数会导致 FFmpeg 无法向 NVENC 提交新帧（硬件满载
#   等待回收），引发编码器停顿（stall）。扩大至 32 可覆盖约 1 秒的帧缓冲（@30fps），
#   配合 T3-Writer 的 _MAX_BATCH=8 批量写入，基本消除 pipe 速率抖动的影响。
_NVENC_SURFACES_PIPE: int = 32

# NVENC VBR 模式前向帧预看窗口（-rc-lookahead N）：
#   仅在 crf>0（-rc:v vbr 模式）下启用。NVENC 默认不使用前向预看（N=0），
#   设为 16 后编码器可向前分析 16 帧的运动复杂度，进行更精准的码率分配：
#   · 场景切换前预先降低相邻帧码率，切换后爆发较高 I 帧码率
#   · 高运动区域分配更多比特，静止区域节约比特
#   典型 PSNR 改善 0.2-0.5 dB（1080p VBR）。
#   注意：lookahead 需要 N 帧前瞻缓冲（内部 FIFO），因此输出有 N 帧延迟，
#   与 -delay 0（零输出延迟）互斥，故仅在 VBR 路径启用，QP=0 路径改用 -delay 0。
_NVENC_LOOKAHEAD_VBR: int = 16

# ═══════════════════════════════════════════════════════════════════════════════
# [V6441-LEVEL1-RC] Level 1 (NVENCEncoder ctypes) 码率控制模式
#
# Level 1 是 NVENC GPU 直通编码路径（ctypes 调用 NVENC SDK DLL），与 Level 2
# (FFmpegWriter 子进程) 并行探测，Level 1 成功则跳过 Level 2/3/4。
#
# ── _NVENC_LEVEL1_RATE_MODE ──
# Level 1 的码率控制模式，控制 NV_ENC_RC_PARAMS.rateControlMode：
#
#   "constqp" (默认) — NV_ENC_PARAMS_RC_CONSTQP (mode=0)
#     · 固定 QP 编码，输出码率随场景复杂度波动
#     · 等效 Level 2：-cq:v 0 -b:v 0（纯 CONSTQP）
#     · 端到端延迟最低，无需前瞻缓冲
#     · 与 v6.4.4 原始行为 100% 一致
#
#   "vbr_hq" — NV_ENC_PARAMS_RC_VBR_HQ (mode=32, 0x20 in SDK 12.0+) + targetQuality (CQ 模式)
#     · 质量驱动的 VBR（类似 CRF），通过 targetQuality 控制质量级别
#     · 等效 Level 2：-rc:v vbr_hq -cq:v N -b:v 0
#     · targetQuality 值由 --crf 参数传入（crf=0 时仍使用 CONSTQP）
#     · 输出码率不可预测（编码器按目标质量自适应分配）
#     · [GPU 验证]  header 解析确认 targetQuality@88(uint8) SEQUENTIAL 布局 (nvEncodeAPI.h)；经 116→76→88 三轮修正确认
#
#   "qvbr" — NV_ENC_PARAMS_RC_QVBR (mode=64, 0x40 in SDK 10.0+) + qvbrQuality
#     · 质量可变的 VBR，通过 qvbrQuality 控制质量级别，允许码率灵活波动
#     · 等效 Level 2：-rc:v qvbr -cq:v N -b:v <max_bitrate>
#     · qvbrQuality 值由 --crf 参数传入（crf=0 时仍使用 CONSTQP）
#     · [GPU 验证] qvbrQuality@88(uint8)（与 targetQuality 共用 SEQUENTIAL slot）；非 @rcParams+124
#
#   不支持的 mode：VBR(1) / CBR(2) — 未实现，无对应 Level 2 需求
#
# ── _NVENC_LEVEL1_LOOKAHEAD ──
# Level 1 VBR_HQ 的前向帧预看深度，控制 NV_ENC_RC_PARAMS.lookaheadDepth：
#
#   0   (默认) — 禁用前向预看
#   N>0 — 编码器前瞻 N 帧进行码率分配
#     · 启用条件：_NVENC_LEVEL1_RATE_MODE=="vbr_hq" 且 crf>0（同时满足）
#     · 可选值：0=禁用, 8=低延迟/适中质量(默认), 16=平衡, 32=最高质量/高延迟
#     · 值越大质量越好，延迟/显存开销越大 (N 帧编码延迟，与插帧流水线兼容)
#     · 等效 Level 2：-rc-lookahead N
#     · [GPU 验证] lookaheadDepth@90(uint16) SEQUENTIAL 布局 (nvEncodeAPI.h)
#
# ── 常用配置示例 ──
#
#   1) 默认行为（与原始 v6.4.4 完全一致）：
#        _NVENC_LEVEL1_RATE_MODE = "constqp"
#        _NVENC_LEVEL1_LOOKAHEAD = 0
#
#   2) CQ 质量优先（匹配 Level 2 -cq:v 23 语义，不启用 lookahead）：
#        _NVENC_LEVEL1_RATE_MODE = "vbr_hq"
#        _NVENC_LEVEL1_LOOKAHEAD = 0
#
#   3) CQ + lookahead（最高质量，匹配 Level 2 -cq:v 23 -rc-lookahead 8）：
#        _NVENC_LEVEL1_RATE_MODE = "vbr_hq"
#        _NVENC_LEVEL1_LOOKAHEAD = 8
#
#   4) QVBR（码率可控质量，匹配 Level 2 -rc:v qvbr -cq:v 23 -b:v 50M）：
#        _NVENC_LEVEL1_RATE_MODE = "qvbr"
#        _NVENC_LEVEL1_LOOKAHEAD = 8
#        # 注意：QVBR 额外需要 vbvBufferSize(@+28) 和 vbvInitialDelay(@+32)
#        # 需在 NVENCEncoder 或 call site 中设置（当前自动设为 vbvBuf=4MB, vbvDelay=2MB）
#
#   5) lookahead 仅在 crf>0 时生效。crf=0 时（--crf 0），call site 强制
#      la_depth=0 以保持零输出延迟兼容性，忽略此处的 LOOKAHEAD 设置。
#
#   修改常量后无需改动任何其他代码 — 运行管线和 call site 自动适配。
# ═══════════════════════════════════════════════════════════════════════════════
# ── _NVENC_QVBR_ENABLE_VBV ──
# QVBR 模式是否启用 VBV 缓冲合规约束（vbvBufferSize / vbvInitialDelay）。
#   True  — 启用：设置 VBV 约束，适合需要严格缓冲合规的流媒体/广播场景。
#           但每帧编码后需更新 VBV 水位 + QP 范围约束，增加 ~5% 延迟。
#   False — 关闭：跳过 VBV 设置，编码器不追踪虚拟解码器缓冲，离线编码
#           延迟更低、吞吐更高。
#   [GPU 验证] 移除 VBV 后 QVBR 编码延迟降低 ~5%。
_NVENC_QVBR_ENABLE_VBV: bool = False

_NVENC_LEVEL1_RATE_MODE: str = "vbr_hq"
_NVENC_LEVEL1_LOOKAHEAD: int = 8  # LA depth for VBR_HQ/QVBR. range 0-30.
# CE-pipeline 初始 slot 数，运行时会 auto-calibrate 为 >= LA+1 (SDK 硬件安全要求)
_NVENC_LEVEL1_DEFAULT_SLOTS: int = 4

# ── _NVENC_CRF0_FORCE_CONSTQP ──
# crf=0 时是否强制使用 CONSTQP + 禁用 lookahead。
#   True  (默认) — crf=0 时强制 rate_mode→"constqp", qp=0, la_depth=0
#                   覆盖 config/CLI 中的 rate_mode 配置（与历史行为 100% 一致）。
#                   运行时会打印日志说明覆盖原因。
#   False — crf=0 时不覆盖 rate_mode/lookahead，使用下方独立 quality 值。
#           仅对 vbr_hq/qvbr 有意义；constqp 模式下 crf=0 自然就是 qp=0。
_NVENC_CRF0_FORCE_CONSTQP: bool = True   # crf=0 时强制 CONSTQP qp=0（真无损，避免 VBR_HQ/QVBR RC 丢弃末帧导致少1帧）
# ── 以下常量仅在 _NVENC_CRF0_FORCE_CONSTQP=False 时生效 ──
# _NVENC_CRF0_QUALITY: crf=0 且使用配置 rate_mode 时的 qp 值。
#   值 0 具有双重语义，在 NVENCEncoder 内部按 rate_mode 分流处理：
#     CONSTQP:        qp=0 → 真逐像素无损
#     VBR_HB/QVBR:    qp<=0 → _qp_val=1 → targetQuality=1（NVENC scale: 1=最好）
#   不可改为 1：会破坏 CONSTQP 模式下的真无损（qp=0 变 qp=1 近无损）。
_NVENC_CRF0_QUALITY: int = 0
# crf=0 且 _NVENC_CRF0_FORCE_CONSTQP=False 时的 lookahead 深度。
_NVENC_CRF0_LOOKAHEAD: int = 0


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

    pair_depth   = max(2, min(int(_math.ceil(t2_ms / max(t1_ms, 0.1))) + 2, 8))
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
        # [FIX-ENC-THREAD] 独立 NVENC 编码线程句柄（GPU_RAW 路径惰性初始化，其余路径为 None）
        _enc_thread: Optional[_NVENCEncodeThread] = None
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

                        # [FIX-ENC-THREAD] 构建交叉交错顺序的帧列表，提交给独立编码线程。
                        # T3 Writer 做 RGB→NV12 kernel，编码线程做 encode_frame，
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
                            _enc_thread = _NVENCEncodeThread(_nvenc, writer)
                        _is_first_submit = (_enc_thread._written == 0 and _enc_thread._empty == 0)
                        # [FIX-ENC-THREAD] CRITICAL: 等待当前 PyTorch stream 完成 NV12 写入，
                        # 再交给编码线程的 cuMemcpy2D 读取，防止 GPU 数据未就绪导致静默花帧。
                        torch.cuda.current_stream().synchronize()
                        _enc_thread.submit(encode_order, force_idr_first=_is_first_submit)
                        _n_pairs_total += rB
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
                            # img1_raw: 来自 reader (BGR)，翻转为 RGB 以匹配 rgb24
                            if i < len(_img1l):
                                _batch_data.extend(_img1l[i][:, :, ::-1].copy().tobytes())
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
                            # img1_raw: 来自 reader (BGR)，翻转为 RGB
                            writer.write(_item.img1_raw[i][:, :, ::-1].copy())
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
                            # img1_raw: 来自 reader (BGR)，翻转为 RGB
                            writer.write(img1_raw_list[i][:, :, ::-1].copy())
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
                _enc_written, _enc_empty = _enc_thread.flush_and_join()
                written += _enc_written
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
                                print(f'[T2-CACHE] 已更新缓存 T2={t2_actual:.1f}ms '
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
                        # img1_raw 来自 reader (BGR)，用 input_is_bgr=True 跳过 numpy BGR→RGB 翻转
                        img1_gpu = torch.from_numpy(img1_raw[bi].copy()).cuda()
                        img1_nv12 = _rgb_to_nv12_gpu(img1_gpu, input_is_bgr=True)
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
        prefetch:        int   = 128,
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
            # hw_args = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'nv12']
            # ✅ 正确：nv12 是 NVDEC 合法的 hwaccel_output_format
            # FFmpeg 会先将 CUDA NV12 surface download 到 CPU，
            # 再由 swscale 自动转换为 -pix_fmt bgr24 输出到管道
            hw_args = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'nv12']

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
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self):
        """Consume stderr to prevent pipe buffer deadlock."""
        try:
            while True:
                chunk = self._proc.stderr.read(8192)
                if not chunk:
                    break
        except Exception:
            pass

    def _read_loop(self):
        pad_h, pad_w = self._pad_h, self._pad_w
        do_pad = self.need_pad
        fb = self._frame_bytes
        try:
            while True:
                # Read exactly fb bytes (robust against partial pipe reads)
                buf = bytearray()
                while len(buf) < fb:
                    chunk = self._proc.stdout.read(fb - len(buf))
                    if not chunk:
                        break
                    buf.extend(chunk)
                if len(buf) < fb:
                    break
                arr = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(
                    self.height, self.width, 3)
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
    nb = 0
    if 'duration' in vs:
        dur = float(vs['duration'])
        if dur > 0:
            nb = int(dur * fps)
            # 交叉验证：若 nb_frames 与 duration×fps 偏差 > 5%，警告
            if 'nb_frames' in vs and vs['nb_frames'] not in ('N/A', ''):
                nb_meta = int(vs['nb_frames'])
                if nb_meta > 0 and nb > 0 and abs(nb_meta - nb) / max(nb_meta, nb) > 0.05:
                    print(f'⚠️  ffprobe元数据 nb_frames={nb_meta} 与 duration×fps={nb} 不一致，'
                          f'使用后者（分段文件 -c copy 常见）', flush=True)
    elif 'nb_frames' in vs and vs['nb_frames'] not in ('N/A', ''):
        nb = int(vs['nb_frames'])
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
        # [FIX-SLICE-THREAD] x264 slice-threading 信息行
        'using cpu capabilities:', 'slice threads:', 'frame threads:',
        'x264 [info]:', 'x264 [warning]:',
        # [FIX-NVENC-PIPE] NVENC 初始化 / 会话诊断信息行（非错误，无需打印）
        # · 'Initialized NPP'  : CUDA NPP 库初始化（h264_nvenc/hevc_nvenc 启动时打印）
        # · 'NVENC session'    : NVENC 编码会话创建日志
        # · 'GPU #'            : NVENC 选择 GPU 设备信息行
        'Initialized NPP', 'NVENC session', 'GPU #',
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
        n_threads: Optional[int] = None,   # [FIX-SLICE-THREAD] None=自动探测
        rc_mode: str = "constqp",           # NVENC rate control mode for Level 2 fallback
    ):
        # [FIX-T3-V643] 去除内部 _queue 和 _write_loop 线程，直接管道写入
        self._error: Optional[Exception] = None
        self._write_count = 0

        if preset is None:
            preset = 'p4' if 'nvenc' in codec else 'medium'
        elif 'nvenc' in codec and preset in _X264_TO_NVENC_PRESET:
            # [FIX-NVENC-PRESET] --x264-preset 传入 ultrafast 等 x264 名称，
            # 但 nvenc 只认 p1~p7 体系，需自动映射避免 "Unable to parse option value" 错误
            preset = _X264_TO_NVENC_PRESET[preset]

        # [FIX-SLICE-THREAD] 自动探测 CPU / 内存，计算最优软编码并行参数
        _par = _detect_encode_parallelism(n_threads)
        _et  = _par['encode_threads']   # 编码线程数
        _s   = _par['slices']           # x264 分片数
        _ft  = _par['ffmpeg_threads']   # FFmpeg 全局线程数
        _mem = _par['mem_avail_gb']

        # [FIX-LOSSLESS] crf=0 → 按编解码器映射为正确的无损参数。
        # 背景：
        #   · libx264 : crf=0 恰好等于无损，但显式用 -qp 0 语义更清晰
        #   · libx265 : crf=0 ≠ 无损！仅为极高质量有损；无损需 -x265-params lossless=1
        #   · nvenc   : -cq:v 0 是 VBR 模式下的极低码率控制，不是无损；
        #               无损需去掉 -rc:v vbr，改用 -qp 0 -b:v 0（常量 QP 模式）
        # [FIX-SLICE-THREAD] x265 frame-threads：默认 min(4, cpu_logical//2)
        # x265 frame-threads 含义：同时编码的帧数（帧级并行），通常 2-4 最佳；
        # 过高会引入帧延迟，与 pipe 流式输入场景不符。
        _x265_ft = max(2, min(4, _par['cpu_logical'] // 2))
        # x265 pool：线程池总大小（所有 frame-threads 共享），= encode_threads
        _x265_pool = _et

        if crf == 0:
            if 'nvenc' in codec:
                # [FIX-LOSSLESS] NVENC 无损：常量 QP=0，去掉 vbr 码率控制。
                # [FIX-NVENC-PIPE] 同时注入 pipe 场景三项优化参数：
                #   · -bf 0        禁用 B 帧：B 帧编码需要前后参考帧，编码器须缓存后续帧
                #                  才能输出，引入多帧流水线延迟；禁用后每帧独立编码即输出。
                #   · -surfaces N  扩大 NVENC 内部帧缓冲槽数（默认 8 → _NVENC_SURFACES_PIPE），
                #                  防止 pipe 写入速率抖动时编码器因 surface 耗尽而暂停。
                #   · -delay 0     零输出延迟：配合 -bf 0 强制 NVENC 在每帧编码完成后
                #                  立即写入输出流，不等待后续帧，最小化 pipe 端到端延迟。
                #                  注意：-delay 0 与 -rc-lookahead 互斥（lookahead 需要前瞻
                #                  缓冲），此处无损模式无需质量优化型预看，故可安全启用。
                quality_args = [
                    '-preset', preset,
                    '-rc', 'constqp',
                    '-qp', '0', '-b:v', '0',
                    '-bf', '0',
                    '-surfaces', str(_NVENC_SURFACES_PIPE),
                    '-delay', '0',
                ]
            elif codec == 'libx265':
                # x265 无损：lossless=1 + 多线程（替换旧 pools=none）
                # [FIX-SLICE-THREAD] pools={N} 启用线程池，frame-threads={F} 帧级并行
                quality_args = [
                    '-preset', preset,
                    '-x265-params',
                    f'lossless=1:pools={_x265_pool}:frame-threads={_x265_ft}',
                ]
            elif codec == 'libx264':
                # x264 无损：-qp 0 + slice-based threading
                # [FIX-SLICE-THREAD] threads=N 设置 x264 编码线程数；
                # slices=S 将单帧切为 S 片并行编码（intra-frame 并行），
                # 与 pipe 流式场景匹配（每帧编完即输出，无帧间延迟）。
                quality_args = [
                    '-preset', preset, '-qp', '0',
                    '-x264-params', f'threads={_et}:slices={_s}',
                ]
            else:
                # 其他编解码器（如 ffv1、utvideo 等）：回退到 -qp 0
                quality_args = ['-qp', '0']
        elif 'nvenc' in codec:
            # [FIX-NVENC-PIPE] NVENC VBR（cq）模式 pipe 场景优化：
            #   · -bf 0              禁用 B 帧，同 crf=0 路径，降低流水线缓冲延迟。
            #   · -rc-lookahead N    前向帧预看（N = _NVENC_LOOKAHEAD_VBR）：
            #                        VBR 模式下编码器向前分析 N 帧运动复杂度，优化帧间
            #                        码率分配，改善场景切换质量（PSNR +0.2-0.5 dB）。
            #                        与 -delay 0 互斥（预看需要 N 帧前瞻缓冲区），
            #                        因此 VBR 路径不设 -delay 0。
            #   · -surfaces N        扩大 NVENC 内部帧缓冲（同 crf=0 路径）。
            # [FIX-FFMPEGWRITER-RC] 根据 Level 1 的 RC 模式选择对应的 FFmpeg -rc:v 值
            # 注意：qvbr 在旧版 FFmpeg h264_nvenc 中不可用，回退到 vbr_hq
            _rc_v_map = {'vbr_hq': 'vbr_hq', 'qvbr': 'vbr_hq', 'constqp': 'vbr'}
            _rc_v = _rc_v_map.get(rc_mode, 'vbr')
            quality_args = [
                '-preset', preset,
                '-rc:v', _rc_v, '-cq:v', str(crf), '-b:v', '0',
                '-bf', '0',
                '-rc-lookahead', str(_NVENC_LOOKAHEAD_VBR),
                '-surfaces', str(_NVENC_SURFACES_PIPE),
            ]
        elif codec == 'libx265':
            # [FIX-SLICE-THREAD] 替换旧 pools=none（完全禁用线程池）为正确多线程参数
            quality_args = [
                '-preset', preset, '-crf', str(crf),
                '-x265-params',
                f'pools={_x265_pool}:frame-threads={_x265_ft}',
            ]
        else:
            # libx264（及其他 x264 系列）：追加 slice-based threading 参数
            # [FIX-SLICE-THREAD] threads=N + slices=S：N 线程各负责 S/N 片，
            # 当 slices >= threads 时 x264 自动切换为 slice-based 模式。
            quality_args = [
                '-preset', preset, '-crf', str(crf),
                '-x264-params', f'threads={_et}:slices={_s}',
            ]

        cmd = [
            ffmpeg_bin, '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{width}x{height}',
            '-r', f'{fps:.6f}',
            '-i', 'pipe:0',
        ]
        # [FIX-SLICE-THREAD] FFmpeg 全局 -threads：仅软编码路径需要，
        # NVENC 是 GPU 固定功能硬件单元，不受 CPU 线程数控制。
        if 'nvenc' not in codec:
            cmd.insert(2, '-threads')
            cmd.insert(3, str(_ft))
        if audio_src:
            cmd += ['-i', audio_src, '-c:a', 'copy', '-map', '0:v', '-map', '1:a?']
        if extra_codec_args:
            # 合并 extra_args 与 quality_args，保留后者中不在前者内的关键 pipe 参数
            _extra_flags = set()
            for i in range(0, len(extra_codec_args), 2):
                _extra_flags.add(extra_codec_args[i])
            _merged = list(extra_codec_args)
            for i in range(0, len(quality_args), 2):
                if quality_args[i] not in _extra_flags:
                    _merged.extend(quality_args[i:i+2])
            cmd += ['-vcodec', codec] + _merged
        else:
            cmd += ['-vcodec', codec] + quality_args
        cmd += ['-pix_fmt', 'yuv420p', '-loglevel', 'error', output_path]

        # [FIX-SLICE-THREAD / FIX-NVENC-PIPE] 打印编码参数摘要
        # NVENC 路径：打印 GPU 硬件编码关键参数（无 CPU 线程参数，因 NVENC 不受其控制）
        # 软编码路径：打印 CPU 线程 / slice 并行配置（与旧行为一致）
        if 'nvenc' in codec:
            # [FIX-NVENC-PIPE] NVENC 参数摘要：显示 pipe 场景优化参数的实际生效值，
            # 便于用户确认 surfaces / lookahead / delay 参数是否符合预期。
            # 注：ffmpeg_threads 对 NVENC 本身无效，仅作用于 demux/filter graph，
            #     此处显示以完整呈现 FFmpeg 命令的全局线程配置。
            if crf == 0:
                _nvenc_info = (
                    f'[FIX-NVENC-PIPE] NVENC 无损(QP=0): '
                    f'preset={preset}  bf=0  '
                    f'surfaces={_NVENC_SURFACES_PIPE}  delay=0  '
                    f'ffmpeg_threads={_ft}(全局demux，不影响NVENC硬件单元)'
                )
            else:
                _nvenc_info = (
                    f'[FIX-NVENC-PIPE] NVENC VBR(cq={crf}): '
                    f'preset={preset}  bf=0  '
                    f'rc-lookahead={_NVENC_LOOKAHEAD_VBR}  '
                    f'surfaces={_NVENC_SURFACES_PIPE}  '
                    f'ffmpeg_threads={_ft}(全局demux，不影响NVENC硬件单元)'
                )
            print(f'   {_nvenc_info}', flush=True)
        else:
            # [FIX-SLICE-THREAD] 软编码路径：打印 CPU 线程 / slice 并行配置摘要
            _codec_l = codec.lower()
            if 'x264' in _codec_l or codec not in ('libx265',):
                _thread_info = (
                    f'[FIX-SLICE-THREAD] 软编码并行: '
                    f'cpu={_par["cpu_logical"]}逻辑/{_par["cpu_physical"]}物理  '
                    f'mem_avail={_par["mem_avail_gb"]:.1f}GiB  '
                    f'encode_threads={_et}  slices={_s}  ffmpeg_threads={_ft}'
                )
            else:
                _thread_info = (
                    f'[FIX-SLICE-THREAD] 软编码并行: '
                    f'cpu={_par["cpu_logical"]}逻辑/{_par["cpu_physical"]}物理  '
                    f'mem_avail={_par["mem_avail_gb"]:.1f}GiB  '
                    f'encode_threads={_et}(frame-threads={_x265_ft})  ffmpeg_threads={_ft}'
                )
            print(f'   {_thread_info}', flush=True)

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
        # [FIX-T3-V643] 不再启动内部 _write_loop 线程，改为 write_direct() 直接写管道

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

    def write_direct(self, data):
        """[FIX-T3-V643] 直接写 bytes/memoryview 到 FFmpeg stdin pipe，零中间拷贝。"""
        if self._error is not None:
            raise RuntimeError(f'FFmpegWriter 内部错误: {self._error}') from self._error
        try:
            self._proc.stdin.write(data)
            self._write_count += 1
        except BrokenPipeError:
            self._error = RuntimeError('FFmpeg stdin 管道已断开')
            raise self._error

    def write(self, frame):
        """[FIX-T3-V643] 兼容旧接口：numpy array → tobytes() → write_direct()。"""
        self.write_direct(frame.tobytes())

    def close(self):
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
        self._ring_buf:        Optional[PinnedRingBuffer]    = None  # [FIX-T3-V643] Level 2/3
        self._pool          = TensorPool()
        self._graph:        dict = {}
        self._graph_inputs: dict = {}
        self._timing:       List[float] = []

        # 跨段自适应队列（由上一次运行的综合建议决定）
        self._next_pair_queue = None      # int or None
        self._next_result_queue = None    # int or None
        self._next_t3_fps_measured = 0.0  # [FIX-T3-FPS] 跨段实测 T3 fps（0 表示无实测）

        # [SEGMENT-REUSE] 跨段复用追踪
        self._segment_index    = 0        # 当前分段序号（1-based，process_video 入口递增）
        self._total_segments   = 0        # 总分段数（由外部传入）
        self._last_seg_resolution: Optional[Tuple[int, int]] = None  # 上段分辨率缓存
        self._last_effective_bs = 0       # 上段有效 batch_size

        # [SEGMENT-REUSE] NVENC 编码器缓存
        self._cached_nvenc_encoder: Optional['NVENCEncoder'] = None
        self._cached_nvenc_key: Optional[Tuple[int, int, float, str, int]] = None

        # [SEGMENT-REUSE] PinnedResultPool / PinnedRingBuffer 缓存
        self._cached_result_pool: Optional['PinnedResultPool'] = None
        self._cached_pool_key: Optional[Tuple[int, int, int]] = None     # (max_BT, H_pad, W_pad)
        self._cached_pool_size: int = 0
        self._cached_ring_buf: Optional['PinnedRingBuffer'] = None
        self._cached_ring_key: Optional[Tuple[int, int, int, int]] = None  # (num_slots, max_frames, H, W)

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
        return_gpu: bool = False,
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

        # ── [FIX-T3-V643-GPU] GPU-STAY 路径（Level 1 NVENC 直通） ───────────
        # 推理结果保持在 GPU 上，仅做 float/clamp/byte/HWC 处理，
        # 跳过 D2H → 避免 GPU↔CPU round-trip。编码由 _infer_loop 完成。
        if return_gpu and pred_big.device.type == 'cuda':
            BT = pred_big.shape[0]
            with torch.cuda.stream(self.stream_compute):
                if self.use_fp16:
                    pred_f = pred_big.float()
                else:
                    pred_f = pred_big
                # GPU 侧: 量化 + CHW→HWC + 裁剪到 orig_H×orig_W（保持 RGB）
                pred_u8 = (
                    pred_f.clamp_(0.0, 1.0).mul_(255.0).byte()
                    .permute(0, 2, 3, 1)                              # CHW → HWC (BT,H,W,3) RGB
                    .contiguous()
                )
                interp_gpu = pred_u8[:, :orig_H, :orig_W, :].contiguous()
                # img1 (原始帧): 同样在 GPU 上做量化 + 裁剪
                # img1 来自 prefetch (已 to_rgb=True)，形状 (B, C, H_pad, W_pad)
                if self.use_fp16:
                    img1_gpu = img1.float()
                else:
                    img1_gpu = img1
                img1_u8 = (
                    img1_gpu.clamp_(0.0, 1.0).mul_(255.0).byte()
                    .permute(0, 2, 3, 1)                              # CHW → HWC (B,H,W,3) RGB
                    .contiguous()
                )
                img1_rgb = img1_u8[:, :orig_H, :orig_W, :].contiguous()
            # [FIX-STREAM-SYNC] 确保 default stream 等待 stream_compute 完成。
            # [FIX-STREAM-SYNC-v2] wait_stream 仅创建 GPU 侧依赖（Infer 线程 default stream），
            # 不阻塞 CPU 线程。Writer 线程使用 per-thread default stream，与 Infer 线程
            # 的 default stream 无同步关系。必须加 CPU 侧 synchronize() 确保 stream_compute
            # 上所有 GPU 写入（interp_gpu 和 img1_rgb 的 clamp/permute/contiguous）完全完成，
            # 再通过 result_queue 将 tensor 交给 Writer 线程，消除非确定性的隔帧花屏。
            # v6.4.3 不受影响因其同步编码耗时足够长 (>10ms) 掩盖了竞争窗口；
            # v6.4.4 异步 _NVENCEncodeThread 提交后 Writer 即刻进入下一批 → 窗口缩短。
            torch.cuda.default_stream(self.device).wait_stream(self.stream_compute)
            self.stream_compute.synchronize()
            self._timing.append(time.perf_counter() - t0)
            # 返回 GPU tensor: interp_gpu (BT,orig_H,orig_W,3) uint8 RGB
            #                  img1_rgb  (B, orig_H,orig_W,3) uint8 RGB
            # 元数据: B, T, orig_H, orig_W
            return ('GPU', interp_gpu, img1_rgb, B, T, orig_H, orig_W)

        # ── [FIX-T3-V643] RING-BUFFER D2H 路径（Level 2/3） ──────────────
        # float() + 量化 + DMA 全在 stream_d2h 上执行；GPU 侧保持 RGB + 裁剪；
        # 写入 PinnedRingBuffer slot（无 padding，每帧连续），返回轻量句柄。
        if (self._ring_buf is not None
                and self.stream_d2h is not None
                and pred_big.device.type == 'cuda'):
            BT = pred_big.shape[0]
            slot_id, slot_tensor = self._ring_buf.writer_acquire()
            ev = None
            try:
                with torch.cuda.stream(self.stream_d2h):
                    self.stream_d2h.wait_stream(self.stream_compute)
                    if self.use_fp16:
                        pred_f = pred_big.float()
                    else:
                        pred_f = pred_big
                    # [FIX-T3-V643] GPU 侧: 量化 + CHW→HWC + 裁剪到 orig_H×orig_W（保持 RGB）
                    pred_u8 = (
                        pred_f.clamp_(0.0, 1.0).mul_(255.0).byte()
                        .permute(0, 2, 3, 1)                              # CHW → HWC (BT,H,W,3) RGB
                        .contiguous()
                    )
                    # 裁剪到原始分辨率，保持 RGB（pix_fmt 已是 rgb24）
                    pred_rgb = pred_u8[:, :orig_H, :orig_W, :].contiguous()
                    slot_tensor[:BT].copy_(pred_rgb, non_blocking=True)
                ev = (self._pipeline_runner._event_pool.acquire()
                      if self._pipeline_runner is not None
                      else torch.cuda.Event())
                ev.record(self.stream_d2h)
            except Exception:
                # 归还已获取的 slot（通过 release write_sem）
                self._ring_buf._write_sem.release()
                if ev is not None and self._pipeline_runner is not None:
                    self._pipeline_runner._event_pool.release(ev)
                raise
            self._timing.append(time.perf_counter() - t0)
            # 返回轻量句柄：("RING", slot_id, event, B, T, orig_H, orig_W, BT)
            return ('RING', slot_id, ev, B, T, orig_H, orig_W, BT)

        # ── [STREAM-DUAL] PINNED-D2H 路径（Level 4） ──────────────────────
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
            # [FIX-BATCHCAP] mem_get_info 仅返回 OS 层面真正空闲 VRAM，
            # 不含 PyTorch allocator 已 reserved 但未 allocated 的可复用缓存。
            # 跨段时 TRT engine 保留大量 reserved 池，导致 free_bytes 严重低估，
            # 必须叠加 cached_free（reserved - allocated）才能得到真实可用量。
            cached_free    = (torch.cuda.memory_reserved(self.device)
                              - torch.cuda.memory_allocated(self.device))
            effective_free = free_bytes + cached_free
            bytes_per_frame = H * W * 3 * 2 * 6
            estimated = max(1, int(effective_free * 0.7 / bytes_per_frame))
            return min(estimated, self._max_batch_size)
        except Exception:
            return 1

    def _safe_infer(self, img0_list, img1_list, timesteps, orig_H, orig_W,
                    prefetched_img0_t=None, prefetched_img1_t=None,
                    return_gpu=False):
        in_oom_cascade = False
        _first_attempt = True

        while True:
            try:
                _p0 = prefetched_img0_t if _first_attempt else None
                _p1 = prefetched_img1_t if _first_attempt else None
                result = self._infer_batch(img0_list, img1_list, timesteps, orig_H, orig_W,
                                           prefetched_img0_t=_p0, prefetched_img1_t=_p1,
                                           return_gpu=return_gpu)
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

    # ── [SEGMENT-REUSE] 跨段资源复用方法 ────────────────────────────────────────

    def _is_first_segment(self) -> bool:
        return self._segment_index <= 1

    def _is_last_segment(self) -> bool:
        return self._segment_index >= self._total_segments

    def _get_or_create_nvenc_encoder(self, W: int, H: int, fps: float,
                                      preset: str, qp: int,
                                      rate_mode: str = "constqp", la_depth: int = 0,
                                      pipeline_depth: int = _NVENC_LEVEL1_DEFAULT_SLOTS) -> 'NVENCEncoder':
        """跨段复用 NVENC 编码器，参数不变时跳过 11 行初始化日志和 DLL 加载。"""
        key = (W, H, fps, preset, qp, rate_mode, la_depth, pipeline_depth)
        if self._cached_nvenc_encoder is not None and self._cached_nvenc_key == key:
            if not self.quiet:
                print(f'   [NVENC] 复用已激活编码器 ({W}x{H}@{fps:.1f}fps rate={rate_mode})', flush=True)
            return self._cached_nvenc_encoder
        if self._cached_nvenc_encoder is not None:
            self._cached_nvenc_encoder.close()
        encoder = NVENCEncoder(W, H, fps, preset=preset, qp=qp,
                               rate_mode=rate_mode, la_depth=la_depth,
                               pipeline_depth=pipeline_depth)
        self._cached_nvenc_encoder = encoder
        self._cached_nvenc_key = key
        return encoder

    def _get_or_create_result_pool(self, pool_size: int, max_BT: int,
                                    H_pad: int, W_pad: int) -> 'Tuple[PinnedResultPool, bool]':
        """跨段复用 PinnedResultPool，参数匹配时跳过锁页内存分配。

        保留 FIX-POOL-LEAK 语义：全局只有一个池实例，不会跨段累积。"""
        key = (max_BT, H_pad, W_pad)
        if (self._cached_result_pool is not None and self._cached_pool_key == key
                and self._cached_pool_size >= pool_size):
            return self._cached_result_pool, False
        if self._cached_result_pool is not None:
            self._cached_result_pool.free()
        pool = PinnedResultPool(pool_size, max_BT, H_pad, W_pad)
        self._cached_result_pool = pool
        self._cached_pool_key = key
        self._cached_pool_size = pool_size
        return pool, True

    def _get_or_create_ring_buf(self, num_slots: int, max_frames_per_slot: int,
                                 H: int, W: int) -> 'Tuple[PinnedRingBuffer, bool]':
        """跨段复用 PinnedRingBuffer。"""
        key = (num_slots, max_frames_per_slot, H, W)
        if self._cached_ring_buf is not None and self._cached_ring_key == key:
            return self._cached_ring_buf, False
        if self._cached_ring_buf is not None:
            self._cached_ring_buf.free()
        ring = PinnedRingBuffer(num_slots=num_slots, max_frames_per_slot=max_frames_per_slot,
                                H=H, W=W)
        self._cached_ring_buf = ring
        self._cached_ring_key = key
        return ring, True

    def cleanup(self):
        """统一释放所有跨段缓存资源。由处理器层在全部流程结束后调用。"""
        if self._cached_nvenc_encoder is not None:
            self._cached_nvenc_encoder.close()
            self._cached_nvenc_encoder = None
            self._cached_nvenc_key = None
        if self._cached_result_pool is not None:
            self._cached_result_pool.free()
            self._cached_result_pool = None
            self._cached_pool_key = None
            self._cached_pool_size = 0
        if self._cached_ring_buf is not None:
            self._cached_ring_buf.free()
            self._cached_ring_buf = None
            self._cached_ring_key = None

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
        # [SEGMENT-REUSE] 分辨率未变时跳过 VRAM 重算，复用上段 effective_bs
        if self._last_seg_resolution == (W, H) and self._last_effective_bs > 0:
            effective_bs = self._last_effective_bs
        else:
            # [FIX-BATCHCAP] mem_get_info()[0] 仅返回 OS 层面空闲 VRAM，跨段后 PyTorch
            # allocator 仍持有大量 reserved 缓存（TRT engine 等），导致估算严重偏低。
            # 修复：effective_free = OS空闲 + PyTorch可复用缓存（reserved - allocated）
            _seg_free = 0
            if torch.cuda.is_available():
                _raw_free, _  = torch.cuda.mem_get_info(self.device)
                _cached_free  = (torch.cuda.memory_reserved(self.device)
                                 - torch.cuda.memory_allocated(self.device))
                _seg_free = _raw_free + _cached_free
            effective_bs = self.batch_size
            if _seg_free > 0:
                res_max_bs = max(1, int(_seg_free * 0.6 / bytes_per_frame))
                if effective_bs > res_max_bs:
                    if self._segment_index <= 1:
                        print(f'[分辨率限制] {W}×{H} 下 batch_size {effective_bs} → {res_max_bs}')
                    effective_bs = res_max_bs
                if self._max_batch_size > res_max_bs:
                    self._max_batch_size = max(effective_bs, res_max_bs)
            self._last_seg_resolution = (W, H)
            self._last_effective_bs = effective_bs

        # [FIX-NVENC-UNIFIED] 在分辨率检查后缓存 hw_profile，
        # 作为 best_encoder() 的主判断依据，确保与 AUTO-TUNE 的 nvenc 检测一致
        if not hasattr(self, '_hw_profile_cache'):
            self._hw_profile_cache = _detect_hw_profile(self.device)

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

        # [FIX-NVENC-UNIFIED] 传入 hw_profile，统一两套 NVENC 检测路径
        _lossless_extra = None
        if self.crf == 0 and not codec_override:
            use_codec, _lossless_extra = HardwareCapability.lossless_encoder()
        else:
            use_codec = codec_override or HardwareCapability.best_encoder(
                self.codec, hw_profile=self._hw_profile_cache)
        # crf=0 时使用 lossless_encoder 的 extra_args（除非调用方显式传入了 extra_codec_args）
        use_extra = extra_codec_args if extra_codec_args is not None else _lossless_extra
        if 'nvenc' in use_codec:
            if self._segment_index <= 1:
                print(f'\n[{worker_label}] NVENC 编码已激活: {use_codec}')

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

        # [FIX-T3-V643] 四级自动回退编码路径探测
        _nvenc_encoder = None
        _active_level = 4  # 默认最低级别
        _use_nvenc_direct = False

        # [V6441-CRF0-DECOUPLE] 解耦 crf=0 与 rate_mode/lookahead：
        # crf=0 历史上强制 CONSTQP qp=0 + la_depth=0（三元表达式），
        # 现在通过 _NVENC_CRF0_FORCE_CONSTQP 常量控制该行为。
        # True (默认) → 行为与当前 100% 一致。
        # False → crf=0 不覆盖 rate_mode/lookahead，使用独立 quality 常量。
        _level1_pd = getattr(self, '_pipeline_depth', _NVENC_LEVEL1_DEFAULT_SLOTS)
        if self.crf == 0 and _NVENC_CRF0_FORCE_CONSTQP:
            _level1_qp = 0
            _level1_rate = "constqp"
            _level1_la = 0
        elif self.crf == 0:
            _level1_qp = _NVENC_CRF0_QUALITY
            _level1_rate = getattr(self, '_rate_mode', _NVENC_LEVEL1_RATE_MODE)
            _level1_la = _NVENC_CRF0_LOOKAHEAD
        else:
            _level1_qp = self.crf
            _level1_rate = getattr(self, '_rate_mode', _NVENC_LEVEL1_RATE_MODE)
            _level1_la = getattr(self, '_la_depth', _NVENC_LEVEL1_LOOKAHEAD)

        # [V6441-CRF0-LOG] 打印 CRF=0 时的 rate_mode 决策日志
        if self.crf == 0:
            _cfg_rate = getattr(self, '_rate_mode', _NVENC_LEVEL1_RATE_MODE)
            if _NVENC_CRF0_FORCE_CONSTQP:
                print(f'   [NVENC] CRF=0 + _NVENC_CRF0_FORCE_CONSTQP=True → '
                      f'强制 CONSTQP qp=0 LA=0（配置 rate_mode={_cfg_rate} 被覆盖）', flush=True)
            else:
                print(f'   [NVENC] CRF=0 + _NVENC_CRF0_FORCE_CONSTQP=False → '
                      f'rate_mode={_level1_rate} qp={_level1_qp} LA={_level1_la}', flush=True)

        # ── Level 1: NVENC SDK GPU 直通编码 ──
        if 'nvenc' in use_codec:
            try:
                _nvenc_encoder = self._get_or_create_nvenc_encoder(
                    W, H, new_fps,
                    preset=self.x264_preset if self.x264_preset else 'p1',
                    qp=_level1_qp,
                    rate_mode=_level1_rate,
                    la_depth=_level1_la,
                    pipeline_depth=_level1_pd,
                )
                writer = FFmpegMuxer(
                    output_path, new_fps,
                    audio_src=audio_src,
                    ffmpeg_bin=self.ffmpeg_bin,
                    quiet=self.quiet,
                )
                _nvenc_encoder.set_muxer_ref(writer)  # [FIX-SPS-PPS] 建立 muxer 回引用
                _use_nvenc_direct = True
                _active_level = 1
                if self._is_first_segment():
                    print(f'   [FIX-T3-V643] Level 1: NVENC GPU 直通编码 ({W}x{H}@{new_fps:.1f}fps)', flush=True)
            except Exception as _nv_err:
                if self._is_first_segment():
                    print(f'   [FIX-T3-V643] Level 1 失败: {_nv_err}', flush=True)

        # [DIAG] 记录最终激活的编码级别，供 benchmark 诊断使用
        self._diag_active_level = _active_level

        # ── Level 2/3: Pinned Ring Buffer + FFmpegWriter ──
        if _active_level == 4:
            _ring_max_frames = effective_bs * (n_interp + 1)  # B × (T+original)
            _ring_slot_bytes = _ring_max_frames * H * W * 3
            _ring_num_slots = min(8, max(2,
                (4 * 1024 * 1024 * 1024) // _ring_slot_bytes if _ring_slot_bytes > 0 else 4
            ))
            try:
                self._ring_buf, _ring_created = self._get_or_create_ring_buf(
                    num_slots=_ring_num_slots,
                    max_frames_per_slot=_ring_max_frames,
                    H=H, W=W,
                )
                _lv = 2 if 'nvenc' in use_codec else 3
                _active_level = _lv
                if _ring_created and not self.quiet:
                    print(f'   [FIX-T3-V643] Level {_lv}: Ring Buffer + '
                          f'{"NVENC" if _lv == 2 else "软编码"} ({_ring_num_slots} slots × {_ring_max_frames} frames)', flush=True)
            except Exception as _rb_err:
                if self._is_first_segment():
                    print(f'   [FIX-T3-V643] Level 2/3 (Ring Buffer) 初始化失败: {_rb_err}', flush=True)

        # ── Level 4: 标准路径 ──
        if _active_level == 4:
            if self._is_first_segment():
                print(f'   [FIX-T3-V643] Level 4: 标准 PinnedResultPool 路径', flush=True)

        if not _use_nvenc_direct:
            writer = FFmpegWriter(
                output_path, W, H, new_fps,
                codec            = use_codec,
                extra_codec_args = use_extra,
                crf              = self.crf,
                preset           = self.x264_preset,
                audio_src        = audio_src,
                ffmpeg_bin       = self.ffmpeg_bin,
                quiet            = self.quiet,
                rc_mode          = _NVENC_LEVEL1_RATE_MODE,
            )
        # [FIX-NVENC-AWARE] 保存实际使用的编码器，供段后诊断代码使用
        self._last_used_codec = use_codec

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

        # [FIX-F0-MISSING] 首帧 f0 必须显式编码并写入。CE pipeline 的 encode_order
        # 仅包含插值帧+img1（右帧），f0 从未被列入 — 不在此处写入则永久丢失。
        # LA>0: encode_frame 返回 0 bytes（LA 缓冲），暂存 encoder 由 encode thread
        #       插入累积 batch 开头。LA=0: 直接 encode_frame。
        if not skip_first_output:
            if _nvenc_encoder is not None:
                first_gpu = torch.from_numpy(first).cuda()
                first_nv12 = _rgb_to_nv12_gpu(first_gpu, input_is_bgr=True)
                force_idr_f0 = not self._is_first_segment()
                if _nvenc_encoder._la_depth > 0:
                    _nvenc_encoder._pending_f0_nv12 = first_nv12
                    _nvenc_encoder._pending_f0_force_idr = force_idr_f0
                else:
                    h264_data = _nvenc_encoder.encode_frame(first_nv12, force_idr=force_idr_f0)
                    writer.write(h264_data)
                    output_count += 1
            else:
                writer.write(first)
                output_count += 1

        frame_count = 1
        if pbar:
            pbar.update(1)

        # ── 主处理 ────────────────────────────────────────────────────────────
        preview_interrupted = False
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
            try:
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
                    nvenc_encoder     = _nvenc_encoder,  # [FIX-T3-V643]
                )
            except Exception as e:
                print(f'[{worker_label}] 流水线异常: {e}', flush=True)
                reader.close()
                writer.close()
                if pbar: pbar.close()
                return False, 0, 0
            if _nvenc_encoder:
                _leftover = _nvenc_encoder.flush()
                if _leftover:
                    writer.write(_leftover)
                # [SEGMENT-REUSE] 编码器跨段复用，仅 flush 不 close；由 cleanup() 统一销毁
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
            # 诊断计数器：reader/infer/writer 三级帧数追踪（供 benchmark 交叉验证）
            self._diag_reader_pairs  = getattr(pipeline, '_diag_reader_pairs', 0)
            self._diag_infer_batches = getattr(pipeline, '_diag_infer_batches', 0)
            self._diag_infer_pairs   = getattr(pipeline, '_diag_infer_pairs', 0)
            self._diag_writer_frames = getattr(pipeline, '_written', 0)
            self._diag_gpu_stay_batches = getattr(pipeline, '_diag_gpu_stay_batches', 0)
            self._diag_nvenc_frames   = getattr(pipeline, '_diag_nvenc_frames', 0)
            self._diag_empty_h264     = getattr(pipeline, '_diag_empty_h264', 0)
            self._diag_external_first_frame = 1  # f0 不在 pipeline._written 计数内（CE pipeline 首帧由 first_raw 传入但未计入 _written）
            frame_count  += fc_extra
            # [FIX-LA-COUNT] 使用 Writer 线程的实际写入帧数，而非推理线程的理论 oc_extra。
            # NVENC LA 缓冲消耗的帧不会写入 muxer（h264_data==b"" 不计入 written），
            # Writer 线程已正确排除。pipeline._written 在 writer thread join 后可用。
            if _nvenc_encoder is not None:
                _actual_written = getattr(pipeline, '_written', 0)
                if _actual_written > 0:
                    output_count += _actual_written  # [FIX-F0] 累加(含f0)，不再替换
                else:
                    output_count += oc_extra
            else:
                output_count += oc_extra
            if n_seg_est > 0:
                _shortfall = n_seg_est - frame_count
                if _shortfall > 1:
                    print(
                        f'[{worker_label}] ⚠️ 提前EOF！预期 {n_seg_est} 帧，实际读取 {frame_count} 帧 '
                        f'（缺失 {_shortfall} 帧，{_shortfall/n_seg_est*100:.1f}%）',
                        flush=True,
                    )
        else:
            # 同步回退路径
            preview_interrupted = False
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
                    avg_t = np.mean(self._timing[-20:]) * 1000 if self._timing else 0
                    pbar.set_postfix(
                        fps=f'{meter.fps():.1f}',
                        eta=f'{meter.eta(n_seg_est):.0f}s',
                        ms=f'{avg_t:.0f}',
                        refresh=False,
                    )
                    pbar.update(1)
                if preview and frame_count % preview_interval == 0:
                    import cv2
                    cv2.imshow(f'IFRNet Preview [{worker_label}]', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        preview_interrupted = True
                        break
            if len(raw_buf) > 1:
                flush_buf()

        # ── 收尾 ──────────────────────────────────────────────────────────────
        if pbar:
            pbar.close()
        writer.close()
        reader.close()

        if n_seg_est > 0:
            _shortfall = n_seg_est - frame_count
            if _shortfall > 1:
                print(
                    f'[{worker_label}] ⚠️ 提前EOF！预期 {n_seg_est} 帧，实际读取 {frame_count} 帧 '
                    f'（缺失 {_shortfall} 帧，{_shortfall/n_seg_est*100:.1f}%）',
                    flush=True,
                )

        elapsed = time.time() - t_start
        print(f'[{worker_label}] 完成 | 原始帧={frame_count} → 输出帧={output_count} | '
              f'{frame_count/elapsed:.1f} 原始帧/s（含 warmup/初始化）')
        if preview_interrupted:
            print(f'[{worker_label}] ⚠️  用户按 q 提前退出预览，输出不完整')
            return False, 0, 0
        return True, frame_count, output_count

    # ── 对外公开接口 ──────────────────────────────────────────────────────────

    def process_video(
        self,
        input_path:       str,
        output_path:      str,
        scale:            float = 2.0,
        preview:          bool  = False,
        preview_interval: int   = 30,
        total_segments:   int   = 1,       # [SEGMENT-REUSE] 总分段数
        segment_index:    int   = 1,       # [SEGMENT-REUSE] 当前分段序号（1-based）
    ) -> bool:
        if not os.path.exists(input_path):
            print(f'错误: 输入不存在 - {input_path}')
            return False
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        self._segment_index  = segment_index
        self._total_segments = total_segments
        _is_first = (segment_index == 1)
        _is_last  = (segment_index >= total_segments)

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
        # [SEGMENT-REUSE] 仅最后一段输出完整报告，中间段静默
        self._gpu_monitor.stop()
        _gpu_stats = self._gpu_monitor.get_stats()
        _verbose_report = _is_last or total_segments <= 1
        if _gpu_stats.sample_count > 0:
            _cur_pair_q   = getattr(self, '_last_pair_q_size',   4)
            _cur_result_q = getattr(self, '_last_result_q_size', 16)
            _slot_mb = getattr(self, '_last_pool_slot_mb', 0.0)
            if _verbose_report:
                print()
                self._gpu_monitor.print_report(
                    _gpu_stats,
                    current_bs       = self.batch_size,
                    current_pair_q   = _cur_pair_q,
                    current_result_q = _cur_result_q,
                    codec            = getattr(self, '_last_used_codec', self.codec),
                    slot_mb          = _slot_mb,   # [FIX-MAXRQ-DYNAMIC] 传入以应用内存上限约束
                )
            # [FIX-T3-DETECT] 获取 GPU-MONITOR 的队列建议（含 T3-bottleneck 检测）
            pair_gpu_sug, result_gpu_sug = self._gpu_monitor.get_queue_suggestions(
                _gpu_stats, _cur_pair_q, _cur_result_q,
                slot_mb=_slot_mb,           # 传入每 slot 大小，用于 PinnedPool 内存约束
                codec=getattr(self, '_last_used_codec', self.codec),           # [FIX-NVENC-AWARE] 使用实际编码器
            )
            # 获取 AUTO-TUNE-RETUNE 的建议（如果存在）
            retune_pair_q   = getattr(self, '_retune_pair_q',   None)
            retune_result_q = getattr(self, '_retune_result_q', None)

            # [FIX-T3-DETECT] 先检测是否 T3-bottleneck，再决定综合策略
            _is_t3 = GPUMonitor._is_t3_bottleneck(_gpu_stats, codec=getattr(self, '_last_used_codec', self.codec))
            if _is_t3:
                # T3 是真正瓶颈：不增大队列，result_queue 可适当缩小以回收 pinned 内存
                final_pair_q   = _cur_pair_q
                final_result_q = max(16, _cur_result_q - 8) if _cur_result_q > 16 else _cur_result_q
                if _verbose_report:
                    print(
                        f'[ADAPTIVE-QUEUE] ⚠️  T3-bottleneck 确认（编码器是瓶颈）：'
                        f'pair_queue={final_pair_q}（不变）'
                        f' result_queue={_cur_result_q}->{final_result_q}（适当缩小，回收锁页内存）'
                    )
                # [FIX-T3-REPORT] 增强诊断：实测 vs 理论 T3 fps + 具体编码建议
                _t3_fps_meas = getattr(self, '_last_t3_fps_measured', 0.0)
                _H_enc, _W_enc = getattr(self, '_last_encode_hw', (0, 0))
                _nvenc_already_active = 'nvenc' in getattr(self, '_last_used_codec', self.codec).lower()
                _t3_fps_est = 0.0
                # [FIX-NVENC-AWARE] _software_encode_fps 估算的是 x264 软编码速度，
                # 对 NVENC 硬编码毫无意义；当 NVENC 已激活时跳过此估算。
                if (not _nvenc_already_active
                        and _H_enc > 0 and _W_enc > 0):
                    _t3_fps_est = _software_encode_fps(
                        os.cpu_count() or 4, _H_enc, _W_enc,
                        self.codec, self.x264_preset, self.crf,
                    )
                if _verbose_report:
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
                    if _nvenc_already_active:
                        _encoder_tip = (
                            f'NVENC 已激活但 T3 仍为瓶颈（实测 {_t3_fps_meas:.0f} fps）。'
                            f'建议：1) 尝试 --x264-preset p1（最快 NVENC preset） '
                            f'2) 尝试 --crf 0 无损模式（跳过 VBR 前向预看） '
                            f'3) bgr24→yuv420p CPU 格式转换 / pipe 写入带宽 / 非标准分辨率'
                        )
                    elif _has_nvenc_h264 and _t3_fps_meas > 0:
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
                _rq_retune = retune_result_q if retune_result_q is not None else _cur_result_q
                _rq_combined_raw = max(
                    (result_gpu_sug + _rq_retune) // 2,
                    _cur_result_q,
                )
                final_result_q = _rq_combined_raw
                # 硬上限
                final_pair_q   = min(final_pair_q, 8)
                final_result_q = min(final_result_q, 64)
                # [FIX-T3-MEMCAP / FIX-POOL-AUTOSCALE / FIX-MAXRQ-DYNAMIC]
                # PinnedPool 内存上限约束：改用三轴动态函数，并显式 log 截断原因。
                if _slot_mb > 0.0:
                    _mem_avail_gb_aq = _detect_encode_parallelism()['mem_avail_gb']
                    _max_rq_mem = _compute_max_result_queue(
                        slot_mb      = _slot_mb,
                        mem_avail_gb = _mem_avail_gb_aq,
                    )
                    if final_result_q > _max_rq_mem:
                        if _verbose_report:
                            _ram_budget_mb = _mem_avail_gb_aq * 1024.0 * 0.06
                            print(
                                f'[ADAPTIVE-QUEUE] PinnedPool 动态上限截断: '
                                f'result_queue {final_result_q} → {_max_rq_mem}'
                                f'  (slot={_slot_mb:.1f} MB × {_max_rq_mem}'
                                f' ≈ {_slot_mb * _max_rq_mem:.0f} MB'
                                f'  ≤ RAM预算 {_ram_budget_mb:.0f} MB'
                                f'  [mem_avail={_mem_avail_gb_aq:.1f} GB × 6%])'
                            )
                    final_result_q = min(final_result_q, _max_rq_mem)

                # [FIX-RETUNE-DISPLAY] 打印推导链路，让建议来源透明可追溯
                _retune_str = (f'RETUNE={_rq_retune}' if retune_result_q is not None
                               else 'RETUNE=N/A')
                if _verbose_report:
                    print(
                        f'[ADAPTIVE-QUEUE] 下次将使用 pair_queue={final_pair_q} '
                        f'result_queue={final_result_q}'
                        f'  (GPU-MONITOR={result_gpu_sug} {_retune_str} → avg={_rq_combined_raw})'
                    )
                # [FIX-T3-FPS] 非 T3-bottleneck 时也更新实测 T3 fps（更可靠）
                self._next_t3_fps_measured = getattr(self, '_last_t3_fps_measured', 0.0)

            self._next_pair_queue   = final_pair_q
            self._next_result_queue = final_result_q
            # [SEGMENT-REUSE] 非末段输出简短队列调整摘要
            if not _verbose_report:
                print(f'[ADAPTIVE-QUEUE] 段{segment_index}→段{segment_index+1}: '
                      f'pair_queue={_cur_pair_q}→{final_pair_q} '
                      f'result_queue={_cur_result_q}→{final_result_q}')
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
            # 期望帧数 = 插值理论值 (fc-1 对产生 scale× 新帧) - NVENC前向预看帧损
            # LA 帧损是设计行为（见 memory/la-flush-recovery-is-harmful）, 非 bug
            _nvenc = getattr(self, '_cached_nvenc_encoder', None)
            _la_depth = getattr(_nvenc, '_la_depth', 0) if _nvenc is not None else 0
            _expected = (fc - 1) * scale  # NVENC 帧数守恒: Output == Input
            _loss = _expected - oc
            if _loss == 0:
                print(f'   原始帧: {fc} → 输出帧: {oc} (×{scale:.1f}, LA={_la_depth})')
            else:
                print(f'   原始帧: {fc} → 输出帧: {oc} '
                      f'(期望≈{_expected}, 差 {_loss} 帧 ⚠️)')
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
        description='IFRNet 视频插帧 —— 终极优化版 v6.4.4（单卡版）',
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
    print('  IFRNet 视频插帧 —— 终极优化版 v6.4.4（单卡版）')
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