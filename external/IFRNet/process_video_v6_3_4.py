"""
IFRNet 视频插帧处理脚本 —— 终极优化版 v6.3.4（多卡版）
==========================================================
基于 IFRNet（Intermediate Flow-based Recursive Network）的视频帧插值脚本，
面向单 GPU 生产环境的高性能实现。

【v6.3.4 新增修复（基于 v6.3.3 多卡版）】
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

  [FIX-POOL-AUTOSCALE] PinnedPool 上限依 GPU 型号自动缩放（替代硬编码 2048MB）：
                   · 根因：硬编码上限对 T4（bs=32 时单 slot ~110MB × 10=1100MB 即超限）
                     过于保守，对 A100/H100（余量充裕）过于宽松。
                   · 新增 _pool_limit_mb_for_profile(profile) 函数，按 gpu_tier 分 6
                     档：GTX 1080=1024  T4/RTX2080=2048  RTX3090/4070=3072
                     A10/L40S/RTX4080+=4096  A100/A800=6144  H100/H800=8192（MiB）。
                   · 兼顾系统可用 RAM：上限不超过 MemAvailable × 12%（最低 1024MB）。
                   · get_queue_suggestions() 新增 hw_profile 参数接收动态上限；
                     _auto_queue_depths() 直接用 profile 计算上限。

【v6.3.3 Worker 链路修复（基于原 v6.3.3 多卡版）】
  [FIX-WORKER-MODEL]    （严重 Bug）模型变体在多卡模式下完全失效：
                   · Worker 子进程以 spawn 模式启动，模块级 Model 重置为默认
                     IFRNet_S；原 worker_args_list 未传 model_name，函数体内
                     硬编码 model_name='IFRNet_S_Vimeo90K'，导致使用
                     --model IFRNet_L_Vimeo90K 等变体时 Worker 静默加载错误架构。
                   · 修复：_process_multi_gpu() worker_args_list 补传
                     model_name = self.model_name；_ifrnet_segment_worker()
                     签名补 model_name 参数；函数体内 try 块前重绑：
                       global Model; Model, _ = _load_ifrnet_module(model_name)
                     并将 processor 初始化的硬编码改为 model_name=model_name。

  [FIX-WORKER-COMPILE]  --no-compile 在多卡模式下无效：
                   · 原 worker_args_list 硬编码 use_compile=True，用户传
                     --no-compile 时单卡路径正确禁用，所有 Worker 仍强制编译。
                   · 修复：改为 use_compile = self.use_compile 透传真实值。

  [FIX-WORKER-TRT]      --use-tensorrt 在多卡模式下静默失效：
                   · 原 worker_args_list 无 use_tensorrt 字段，Worker 端
                     IFRNetVideoProcessor 默认 use_tensorrt=False，无报错。
                   · 修复：worker_args_list 补 use_tensorrt = self.use_tensorrt；
                     _ifrnet_segment_worker 签名补 use_tensorrt: bool，
                     processor 初始化补 use_tensorrt=use_tensorrt。

  [FIX-WORKER-T2CACHE]  T2 缓存目录未透传，接口不一致：
                   · 原 Worker 创建 IFRNetVideoProcessor 时未传 t2_cache_dir，
                     走 _T2_CACHE_DIR_DEFAULT；虽当前 Worker auto_tune=False
                     下 T2 缓存实际不被读写，但接口缺失影响后续扩展。
                   · 修复：worker_args_list 补 t2_cache_dir = self.t2_cache_dir；
                     _ifrnet_segment_worker 签名补 t2_cache_dir: str=''，
                     processor 初始化补 t2_cache_dir=t2_cache_dir。

  [FIX-WORKER-WRITER-NONE]  首帧读取失败时 AttributeError crash：
                   · _process_segment 在多卡 Worker 路径（_use_queue=True）下
                     writer=None，但首帧失败早返回代码调用 writer.close() 未判空，
                     导致 AttributeError。
                   · 修复：将 reader.close(); writer.close() 改为分行调用，
                     writer.close() 用 if writer is not None: 保护。

  [FIX-WORKER-QUIET]    --no-quiet 在多卡 Worker 下无效：
                   · 原 processor 初始化硬编码 quiet=True，用户 --no-quiet
                     仅对主进程生效，所有 Worker 均保持静默，影响调试体验。
                   · 修复：worker_args_list 补 quiet = self.quiet；
                     _ifrnet_segment_worker 签名补 quiet: bool=True（Worker
                     子进程建议默认静默），processor 初始化改为 quiet=quiet。

【v6.3.3 多卡路径特性补全（基于 Worker 链路修复补丁）】
  [GAP-MULTIGPU-TRT-PREBUILD]（严重 Bug）TRT 预构建被多卡路由跳过：
                   · 原 process_video() 中 TRT 预构建代码位于多卡路由之后，
                     使用 --use-tensorrt + 多 GPU 时主进程不构建 Engine，
                     所有 Worker TRT 完全静默失效。
                   · 修复：将 TRT 预构建移到多卡路由判断之前；主进程在 GPU0
                     上构建并缓存 .trt 文件，Workers 直接加载缓存。

  [GAP-WORKER-TRT-BUILD]（严重 Bug）Worker 内 TRT Engine 从不激活：
                   · Worker 调用 _process_segment() 而非 process_video()，
                     Engine 加载代码在 process_video() 里，Worker 路径永远
                     不触发，use_tensorrt=True 传入后 TRT 分支不走。
                   · 修复：_ifrnet_segment_worker 创建 processor 后、调用
                     _process_segment 前手动触发 _build_trt_engine()；
                     主进程预构建后 Worker 通常仅需加载缓存（< 5s）。

  [GAP-WORKER-TRT-CACHE-DIR]（轻微）trt_cache_dir 未透传给 Worker：
                   · 用户指定 --trt-cache-dir 时 Worker 走默认路径，
                     缓存未命中将触发 20-30 分钟重建。
                   · 修复：worker_args_list / _ifrnet_segment_worker 签名 /
                     IFRNetVideoProcessor 初始化均补传 trt_cache_dir。

  [GAP-MULTIGPU-GPUMONITOR]（中等）多卡路径无 GPU 监测：
                   · _process_multi_gpu() 未启动 _gpu_monitor，多卡运行
                     没有 GPU 利用率统计和 batch_size 调优建议。
                   · 修复：在 Worker 启动前 .start()，所有 Worker join 后
                     .stop() + print_report()（仅统计，不运行 ADAPTIVE-QUEUE）。

  [GAP-MULTIGPU-REPORT-TIMING]（中等）多卡模式 JSON 报告永远被跳过：
                   · _dump_report() 守卫条件 not self._timing，多卡模式
                     主进程 self._timing 始终为空（推理在 Worker 子进程），
                     --report 功能完全静默失效。
                   · 修复：新增 _dump_report_multigpu()，记录宏观吞吐指标
                     （帧数/总耗时/fps）；infer_latency_ms 注明不可用原因。

  [GAP-MULTIGPU-WRITER-QUIET]（轻微）主进程 FFmpegWriter 缺 quiet 参数：
                   · _process_multi_gpu() 构造 FFmpegWriter 时未传 quiet，
                     --no-quiet 对多卡编码器日志无效。
                   · 修复：补传 quiet=self.quiet。

  [GAP-MULTIGPU-PREVIEW]（轻微）--preview 在多卡路径静默丢失：
                   · process_video() 多卡路由未传 preview 参数，预览功能
                     对多卡模式完全无效且无提示。
                   · 修复：_process_multi_gpu() 补接 preview/preview_interval
                     参数；FrameCollector 按 preview_interval 调用 cv2.imshow；
                     Worker spawn 进程无显示上下文，预览仅在主线程侧实现。

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
  python process_video_v6_3_3.py \\
      --input input.mp4 --output output_2x.mp4 --scale 2

  # 每 GPU 2 Worker（适合 24GB+ 显存）
  python process_video_v6_3_3.py \\
      --input input.mp4 --output output.mp4 --scale 2 --num-process-per-gpu 2
	  
  # TensorRT 加速（bs=48，首次构建 Engine）
  python process_video_v6_3_3.py \\
      --input input.mp4 --output output.mp4 --scale 2 --use-tensorrt

  # 输出性能报告
  python process_video_v6_3_3.py \\
      --input input.mp4 --output output.mp4 --scale 2 --report report.json

【注意事项】
  · v6.3.0 升级 batch_size 默认值为 48；TRT 用户若沿用旧 .trt 缓存（bs=24），
    首次运行会因 shape 不匹配而自动删除旧缓存并重建 Engine（约需 20-30 分钟）。
  · stream_transfer 属性已拆分为 stream_h2d / stream_d2h，上层调用方如有直接
    引用 processor.stream_transfer 需改为 stream_h2d（预取）或 stream_d2h（输出）。

【多卡版专属功能（相对单卡版新增）】
  M1. [多 GPU 帧范围分割调度]
      按帧计数均分任务，每 GPU 独立处理一段。
      - 多 GPU 时 Worker 将帧以 (global_idx, bytes, shape) 放入 result_q
      - 主进程 FrameCollector 用 heapq 重排后直接写单一 FFmpegWriter
      - 单 GPU 时自动使用全套 v6.3.3 三级流水线优化

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
import multiprocessing as mp
import shutil
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

# ── U1: VRAM 碎片优化：expandable_segments 允许 PyTorch 用不连续 VRAM 满足大分配请求
# 必须在任何 CUDA 分配之前设置（setdefault 不覆盖用户已设置的值）
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

# ── U2: 抑制 torch.inductor / dynamo 的噪音警告 ─────────────────────────────
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

# ── U7: 动态路径配置 ──────────────────────────────────────────────────────────
# 目录结构假设：<project_root>/external/IFRNet/process_video_v5.py
# 如与实际部署结构不符，请直接修改 base_dir / models_ifrnet 两个变量
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
base_dir      = str(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
models_ifrnet = os.path.join(base_dir, 'models_IFRNet', 'checkpoints')
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
    def best_encoder(cls, preferred: str = 'libx264',
                     hw_profile: Optional['_HWProfile'] = None) -> str:
        """
        [FIX-NVENC-UNIFIED] 统一 NVENC 检测路径。

        优先级：
        1. hw_profile.has_nvenc（静态 GPU 型号表，与 AUTO-TUNE 一致，Docker 环境可靠）
        2. HardwareCapability.has_nvenc()（ffmpeg 实际 probe，Docker 无 /dev/nvidia*
           设备映射时可能失败，作为回退保底）

        两套检测结果在 Docker 未映射 NVENC 设备时会不一致：AUTO-TUNE 显示 nvenc=True
        但 ffmpeg probe 失败，导致实际使用 libx264 引发 T3 瓶颈。本修复确保两者一致。
        """
        nvenc_map    = {'libx264': 'h264_nvenc', 'libx265': 'hevc_nvenc'}
        # 反向映射：若用户直接指定 nvenc 编码器但不可用，回退到软件编码器
        fallback_map = {'h264_nvenc': 'libx264', 'hevc_nvenc': 'libx265'}

        def _nvenc_ok(codec_name: str) -> bool:
            # [FIX-NVENC-DOCKER] ffmpeg 实际 probe 是硬件可用性的唯一权威依据。
            # 静态表（hw_profile.has_nvenc）仅用于 AUTO-TUNE T3 性能估算，
            # 不能替代实际编码器能力检测：Docker 缺少 /dev/nvidia* 映射时
            # 静态表显示"GPU 支持 NVENC"但 ffmpeg 无法访问，盲目升级会编码失败。
            if not cls.has_nvenc(codec_name):
                return False
            # ffmpeg probe 通过 + hw_profile 确认（可选，仅作双重校验）
            if hw_profile is not None and hasattr(hw_profile, 'has_nvenc'):
                return hw_profile.has_nvenc
            return True

        # 情况1：用户直接指定了 nvenc 编码器 → 检测可用性，不可用则回退
        if preferred in fallback_map:
            if _nvenc_ok(preferred):
                return preferred
            fallback = fallback_map[preferred]
            print(f'  [警告] {preferred} 不可用，自动回退到 {fallback}')
            return fallback
        # 情况2：用户指定软件编码器 → 尝试升级到对应 nvenc
        candidate = nvenc_map.get(preferred, preferred)
        if candidate != preferred and _nvenc_ok(candidate):
            return candidate
        return preferred

    @classmethod
    def lossless_encoder(cls) -> Tuple[str, List[str]]:
        """
        返回 (codec, extra_args) 用于无损中间段编码。
        优先 nvenc lossless（-rc constqp -qp 0），否则 libx264 lossless。
        """
        if cls.has_nvenc('h264_nvenc'):
            try:
                cmd = [
                    'ffmpeg',
                    '-f', 'lavfi', '-i', 'color=c=black:s=64x64:d=0.1:r=1',
                    '-vcodec', 'h264_nvenc',
                    '-rc', 'constqp', '-qp', '0',
                    '-frames:v', '1', '-f', 'null', '-', '-loglevel', 'error',
                ]
                if subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0:
                    return 'h264_nvenc', ['-rc', 'constqp', '-qp', '0']
            except Exception:
                pass
        # 回退：libx264 无损
        return 'libx264', ['-qp', '0', '-preset', 'medium']


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
        [FIX-T3-DETECT / FIX-T3-MEMCAP / FIX-POOL-AUTOSCALE] 返回 (建议 pair_queue, 建议 result_queue)。
        用于跨段自适应队列调整，不打印信息。

        新增逻辑：
        · T3-bottleneck 时不增大 result_queue（否则 PinnedPool 雪球式积累）。
        · slot_mb > 0 时对 result_queue 施加 PinnedPool 内存上限约束。
        · [FIX-POOL-AUTOSCALE] 上限由模块级 _PINNED_POOL_MAX_MB 按 GPU 型号自动缩放。
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
_T2_BASELINE_HWB  = float(576 * 736 * 24)
_T2_FIXED_MS      = 240.0   # torch.compile / eager 路径固定 overhead（含 JIT 编译）
# [FIX-T2-TRT-CALIB] TRT 路径固定 overhead 实测约 2-5ms（无 JIT、纯硬件调度）
_T2_FIXED_MS_TRT  = 5.0
_T2_VAR_MS        = 25.0


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

# 模块级常量：PinnedPool 锁页内存上限（MiB），按 GPU 型号自动缩放，只初始化一次
_PINNED_POOL_MAX_MB: float = _pool_limit_mb_for_profile(
    _detect_hw_profile(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
)

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
        # 与 torch.compile/eager 的 240ms 差距 48×，必须分路处理
        _fixed_ms = _T2_FIXED_MS_TRT if infer_backend == 'trt' else _T2_FIXED_MS
        t2_base = (_fixed_ms + _T2_VAR_MS * HWB / _T2_BASELINE_HWB) / max(profile.gpu_tier, 0.05)
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

    # [FIX-T3-MEMCAP / FIX-POOL-AUTOSCALE] 依据模块级 _PINNED_POOL_MAX_MB 约束 result_depth。
    # 每个 result slot 持有 effective_bs * T 帧的 pinned uint8 buffer。
    # 若不加约束，T3 极慢（大 T3/T2 比）时 result_depth 会达到 50+，
    # 导致 PinnedPool 分配过多锁页内存，反而拖慢 DMA 带宽，形成恶性循环。
    # [FIX-POOL-AUTOSCALE] 上限由模块级常量按 GPU 型号自动缩放，只初始化一次。
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
                # [FIX-T2-TRT-CALIB] TRT 路径使用专用固定 overhead 常量（5ms vs 240ms）
                _fixed_ms = _T2_FIXED_MS_TRT if infer_be == 'trt' else _T2_FIXED_MS
                _t2b = (_fixed_ms + _T2_VAR_MS * _HWB / _T2_BASELINE_HWB) \
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
        self._out_buf: Optional[torch.Tensor] = None   # [FIX-D2H] 异步 D2H 输出 buffer

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
    """[FIX-D2H] 异步 D2H，用于同步回退路径。"""
    if sync_stream is not None and torch.cuda.is_available():
        torch.cuda.current_stream().wait_stream(sync_stream)
    # GPU 上完成类型转换
    arr_gpu    = t.clamp_(0.0, 1.0).mul_(255.0).round_().byte()
    arr_perm   = arr_gpu.permute(0, 2, 3, 1).contiguous()   # [B, H, W, C]，仍在 GPU
    # 申请 pinned 输出 buffer，异步 DMA GPU→主机
    pool       = _get_pinned_pool()
    out_pinned = pool.get_output_buf(arr_perm.shape, torch.uint8)
    out_pinned.copy_(arr_perm, non_blocking=True)          # 发起 DMA，不阻塞 CPU
    # 显式 synchronize：仅等待 DMA 完成，可与其他 CPU 工作重叠
    device = t.device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    arr = out_pinned.numpy()                               # 零拷贝视图
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
# [FIX-TMP-PIPELINE] _ResultQueueWriter：result_q 适配器
# 实现与 FFmpegWriter 相同的 write()/close() 接口，
# 使 IFRNetPipelineRunner.T3-Writer 在多卡 Worker 中也能透明使用三级流水线。
# ─────────────────────────────────────────────────────────────────────────────

class _ResultQueueWriter:
    """
    [FIX-TMP-PIPELINE] FFmpegWriter 兼容适配器（多卡模式）。

    将 IFRNetPipelineRunner.T3-Writer 的 writer.write(frame) 调用
    重定向为 result_q.put((global_idx, bytes, shape))，
    使每个 Worker 子进程也能享受完整三级流水线加速（T1/T2/T3 全重叠），
    而无需修改 IFRNetPipelineRunner 的任何内部逻辑。

    设计要点：
    · _error = None     : 兼容 IFRNetPipelineRunner 对 writer._error 的检查
    · close()           : 空操作，result_q 由主进程的 FrameCollector 管理
    · final_gidx        : 供调用方获取写完后的全局帧索引（用于 output_count 统计）
    """
    def __init__(self, result_q, start_gidx: int):
        self._q     = result_q
        self._gidx  = start_gidx
        self._error: Optional[Exception] = None   # FFmpegWriter 接口兼容

    def write(self, frame: np.ndarray):
        # frame 已是独立 copy（来自 T3-Writer 的 arr[...].copy()），tobytes() 安全
        self._q.put((self._gidx, frame.tobytes(), frame.shape))
        self._gidx += 1

    def close(self):
        pass   # result_q 生命周期由主进程管理，Worker 无需关闭

    @property
    def final_gidx(self) -> int:
        return self._gidx


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

        # [FIX-PAD] 预计算 padding 量，后台线程直接产出 (raw, padded) 元组
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
        # [FIX-PAD] 若 pad_stride>0，后台线程在解码后立即执行 padding，
        # 始终产出 (raw_frame, padded_frame) 元组。
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
        """返回 (raw_frame, padded_frame) 元组；视频结束时返回 None。
        [FIX-PAD] padded_frame 已在后台线程完成，主线程零额外 CPU 开销。"""
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
    ):
        self._error: Optional[Exception] = None
        self._queue: queue.Queue = queue.Queue(maxsize=128)

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
            quality_args = [
                '-preset', preset,
                '-rc:v', 'vbr', '-cq:v', str(crf), '-b:v', '0',
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

        # [FIX-SLICE-THREAD] FFmpeg 全局 -threads：
        #   · 放在第一个 -i 之前（全局选项位置），作用于 demux / filter graph
        #   · 对 libx264/libx265 编码器本身作用有限（由 -x264-params threads= 控制），
        #     但影响 rawvideo demux 的读取线程和 filter 并行度
        cmd = [
            ffmpeg_bin, '-y',
            '-threads', str(_ft),          # [FIX-SLICE-THREAD] FFmpeg 全局线程数
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
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

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
                pending.append(item if isinstance(item, bytes) else item.tobytes())
                if len(pending) >= self._MAX_BATCH or self._queue.empty():
                    self._proc.stdin.write(b''.join(pending))
                    pending.clear()
        except Exception as e:
            self._error = e
            print(f'[FFmpegWriter Error] {e}')

    def write(self, frame: np.ndarray):
        if self._error is not None:
            raise RuntimeError(f'FFmpegWriter 内部错误: {self._error}') from self._error
        self._queue.put(frame)

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
            stderr_out = '\n'.join(self._stderr_lines[-20:])
            print(f'\n[Warning] FFmpeg 退出码={rc}, stderr: {stderr_out[:400]}')
        if self._error:
            print(f'[Warning] FFmpegWriter 累计写帧异常: {self._error}')


# ─────────────────────────────────────────────────────────────────────────────
# 核心推理类（多 GPU 版本，携带全部单卡优化）
# ─────────────────────────────────────────────────────────────────────────────

class IFRNetVideoProcessor:
    """
    IFRNet 插帧处理器（v5 终极多卡版）。

    单 GPU 模式：直接处理完整视频（保留所有单卡优化）。
    多 GPU 模式：M1 帧范围分割 + 无损中间段 + ffmpeg concat。

    包含全部单卡 v5 演进升级：
      - VRAM 碎片优化（expandable_segments）
      - torch.compile 编译缓存持久化
      - 小形状（32×32）预热，避免大分辨率首编译崩溃
      - 分辨率自适应 batch_size 上限
      - OOM 级联保护与深度清理 + 按实测显存估算恢复
    """
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
        num_process_per_gpu: int  = 1,
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
        # [FIX-TRT-CACHE-DIR] 允许外部指定 TRT 缓存目录（如 ifrnet_processor 传入稳定路径），
        # 不指定时在 process_video() 中回退到 base_dir/.trt_cache 默认规则。
        self.trt_cache_dir   = trt_cache_dir
        self.t2_cache_dir    = t2_cache_dir or _T2_CACHE_DIR_DEFAULT
        self.model_name = model_name      # 保存模型名称
        self.quiet           = quiet
        self.num_process_per_gpu = num_process_per_gpu
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
        """加载模型到指定设备，供单卡路径和多卡子进程共用。"""
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

                # U3: 持久化编译缓存，首次约 2~3 分钟，后续运行秒启动
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
                # [FIX-TRT-MUTEX] compile 异常退出时若 use_tensorrt 同时开启，
                # use_cuda_graph 未经 compile 成功路径重置，需在此补充禁用，
                # 否则 _infer_batch 中 use_cuda_graph 分支会静默优先于 TRT 执行。
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
            # 导出 ONNX
            dummy0 = torch.randn(*input_shape, device=self.device)
            dummy1 = torch.randn(*input_shape, device=self.device)
            embt   = torch.full((B,), 0.5, dtype=torch.float32,
                                device=self.device).view(B, 1, 1, 1)
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
                    export_model, (dummy0, dummy1, embt), onnx_path,
                    input_names=['img0', 'img1', 'embt'],
                    output_names=['output'],
                    opset_version=18,       # [FIX-TRT] torch.onnx >= 2.x 最低支持 18
                    dynamic_axes=None,
                )
            # [FIX-TRT] 确保无 .onnx.data 外部权重文件（TRT parser 只认单文件 ONNX）
            import onnx
            model_proto = onnx.load(onnx_path)
            onnx.save(model_proto, onnx_path,
                      save_as_external_data=False, all_tensors_to_one_file=False)
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

            _gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unknown'
            _sm_code  = _props.major * 10 + _props.minor if torch.cuda.is_available() else 0
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

            # ── [FIX-TRT-CTX-OOM] create_execution_context 防御 ────────────
            # create_execution_context() 在 GPU 显存不足时返回 None（不抛异常）。
            # 典型场景：upscale_then_interpolate 模式下，前序 ESRGan 步骤
            # 占用大量 GPU 显存（模型权重 + GFPGAN + 缓存分配器残留），
            # 导致 IFRNet TRT context 无法分配所需的激活内存。
            self._trt_context = self._trt_engine.create_execution_context()
            if self._trt_context is None:
                print('[TensorRT] ⚠️  create_execution_context() 失败（显存不足），回退 PyTorch。')
                self._trt_engine  = None
                self.use_tensorrt = False
                self._trt_ok      = False
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
            try: os.remove(trt_path)
            except OSError: pass
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
            'embt': static_embt, 'output': static_output,
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
        """根据当前实测空闲显存估算安全的 batch_size。"""
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
            # 单帧 FP16 字节数 × 6 倍激活系数（经验值）
            bytes_per_frame = H * W * 3 * 2 * 6
            # 只使用空闲显存的 70%，留 30% 给中间缓冲
            estimated = max(1, int(effective_free * 0.7 / bytes_per_frame))
            return min(estimated, self._max_batch_size)
        except Exception:
            return 1

    # ──────────────────────────────────────────────────────────────────────────
    # U6: OOM 级联保护 + 深度清理 + 按实测显存恢复
    # ──────────────────────────────────────────────────────────────────────────
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
                    # ── 首次 OOM：永久更新天花板，此 batch_size 已证明不可用 ──
                    safe_ceiling = max(1, self.batch_size - 1)
                    if self._max_batch_size > safe_ceiling:
                        print(f'[OOM] 永久降低 max_batch_size: {self._max_batch_size} → {safe_ceiling}')
                        self._max_batch_size = safe_ceiling
                    in_oom_cascade = True  # 进入级联状态，后续降级不再修改上限

                if self.batch_size <= 1:
                    # 深度清理：同步 + 清空 + 重置 inductor
                    print(f'\n[OOM] batch_size=1 仍 OOM，深度清理后按剩余显存估算恢复...')
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        try: torch._dynamo.reset()
                        except Exception: pass
                        torch.cuda.empty_cache()  # 二次清理，inductor reset 后可能释放更多

                    # 深度清理后用实测空闲显存重新估算 batch_size
                    recovered_bs = self._estimate_safe_batch_size(orig_H, orig_W)
                    # 同时以此作为新的上限（深度清理后能用多少就上限多少）
                    if recovered_bs < self._max_batch_size:
                        print(f'[OOM] 深度清理后估算安全 batch_size={recovered_bs}，'
                              f'更新 max_batch_size: {self._max_batch_size} → {recovered_bs}')
                        self._max_batch_size = recovered_bs
                    self.batch_size    = recovered_bs
                    self._oom_cooldown = 20      # 稳定一段时间再尝试爬升
                    in_oom_cascade     = False   # 深度清理后重置级联标志
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

    # ──────────────────────────────────────────────────────────────────────────
    # 单段处理核心（单 GPU 或多 GPU 的单个 Worker 内部调用）
    # ──────────────────────────────────────────────────────────────────────────

    def _process_segment(
        self,
        input_path:         str,
        output_path:        str,
        scale:              float = 2.0,
        frame_start:        int   = 0,
        frame_end:          int   = -1,
        skip_first_output:  bool  = False,   # M1: 多 GPU 时首帧去重
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
        # [FIX-TMP] 多卡零临时文件模式
        result_q:              Optional[object] = None,
        scale_int_override:    int   = 0,
    ) -> Tuple[bool, int, int]:
        """
        处理视频的一个帧范围段。
        - result_q=None: 写出到 output_path（单 GPU / v5 兼容模式）
        - result_q!=None: [FIX-TMP] 帧数据放入 result_q，由主进程 FrameCollector 写出
        返回 (成功, 原始帧数, 输出帧数)。
        """
        # [FIX-PAD] 将 pad_stride 传入 reader，padding 在后台线程完成
        reader = FFmpegFrameReader(
            input_path,
            frame_start  = frame_start,
            frame_end    = frame_end,
            prefetch     = self.batch_size * 3,
            use_hwaccel  = self.use_hwaccel,
            ffmpeg_bin   = self.ffmpeg_bin,
            pad_stride   = MODEL_STRIDE,     # [FIX-PAD] 交由后台线程 pad
        )
        W, H      = reader.width, reader.height
        fps       = reader.fps
        n_seg_est = reader._segment_frames   # 估计值

        # ── U5: 根据分辨率限制最大 batch_size，防止大分辨率下 VRAM 溢出 ──────
        bytes_per_frame = W * H * 3 * 2 * 6  # 3 channels, fp16=2B, ~6x activations
        # [FIX-BATCHCAP] mem_get_info()[0] 仅返回 OS 层面空闲 VRAM，跨段后 PyTorch
        # allocator 仍持有大量 reserved 缓存（TRT engine 等），导致估算严重偏低。
        # 修复：effective_free = OS空闲 + PyTorch可复用缓存（reserved - allocated）
        _seg_free = 0
        if torch.cuda.is_available():
            _raw_free, _  = torch.cuda.mem_get_info(self.device)
            _cached_free  = (torch.cuda.memory_reserved(self.device)
                             - torch.cuda.memory_allocated(self.device))
            _seg_free = _raw_free + _cached_free
        # [FIX-BS] 局部变量 effective_bs，不修改实例状态，避免多段处理状态泄漏
        effective_bs = self.batch_size
        if _seg_free > 0:
            res_max_bs = max(1, int(_seg_free * 0.6 / bytes_per_frame))
            if effective_bs > res_max_bs:
                print(f'[{worker_label}][分辨率限制] {W}×{H} 下 batch_size {effective_bs} → {res_max_bs}')
                effective_bs = res_max_bs
            if self._max_batch_size > res_max_bs:
                self._max_batch_size = max(effective_bs, res_max_bs)

        # [FIX-NVENC-UNIFIED] 在分辨率检查后缓存 hw_profile，
        # 作为 best_encoder() 的主判断依据，确保与 AUTO-TUNE 的 nvenc 检测一致
        if not hasattr(self, '_hw_profile_cache'):
            self._hw_profile_cache = _detect_hw_profile(self.device)

        # [FIX-PAD] padding 已由 FFmpegFrameReader 后台线程完成
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

        use_codec = codec_override or HardwareCapability.best_encoder(
            self.codec, hw_profile=getattr(self, '_hw_profile_cache', None))
        use_extra = extra_codec_args
        if 'nvenc' in use_codec:
            print(f'[{worker_label}] NVENC 编码: {use_codec}')

        # [FIX-TMP] 多卡零临时文件：result_q 模式下不创建 writer，帧数据放入队列
        _use_queue = result_q is not None
        _s_int = scale_int_override if scale_int_override > 0 else int(scale_frac)
        if _use_queue:
            _out_gidx = [frame_start * _s_int + (1 if skip_first_output else 0)]
            def _write_frame(frame: np.ndarray):
                result_q.put((_out_gidx[0], frame.tobytes(), frame.shape))
                _out_gidx[0] += 1

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

        if not _use_queue:
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
        else:
            writer = None

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
            reader.close()
            if writer is not None:   # [FIX-WORKER-WRITER-NONE] _use_queue 模式下 writer=None
                writer.close()
            if pbar: pbar.close()
            return False, 0, 0
        first, first_padded = pair

        if not skip_first_output:
            if _use_queue:
                _write_frame(first)
            else:
                writer.write(first)
            output_count += 1

        frame_count = 1
        if pbar:
            pbar.update(1)

        # ── 主处理 ────────────────────────────────────────────────────────────
        preview_interrupted = False
        if self.device.type == 'cuda':
            # 单卡：直接用 FFmpegWriter；多卡 Worker：用 _ResultQueueWriter 适配器
            if _use_queue:
                _pipeline_writer = _ResultQueueWriter(result_q, _out_gidx[0])
            else:
                _pipeline_writer = writer

            pipeline = IFRNetPipelineRunner(
                self,
                auto_tune    = not _use_queue,   # 多卡 Worker 不需要 auto-tune
                codec        = use_codec,
                x264_preset  = self.x264_preset,
                crf          = self.crf,
                t2_cache_dir = self.t2_cache_dir if not _use_queue else '',
                pair_queue_override   = pair_queue_override,
                result_queue_override = result_queue_override,
                t3_fps_measured       = t3_fps_measured,   # [FIX-T3-FPS]
            )
            fc_extra, oc_extra = pipeline.run(
                reader            = reader,
                writer            = _pipeline_writer,
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
            if not _use_queue:
                # [GPU-MONITOR-v2] 单卡模式保存 pipeline 统计供 ADAPTIVE-QUEUE 使用
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
                        if _use_queue:
                            _write_frame(interp_frame)
                        else:
                            writer.write(interp_frame)
                        output_count += 1
                    if _use_queue:
                        _write_frame(raw_buf[i + 1])
                    else:
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
                        preview_interrupted = True
                        break
            if len(raw_buf) > 1:
                flush_buf()

        # ── 收尾 ──────────────────────────────────────────────────────────────
        if pbar:
            pbar.close()
        if writer is not None:
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

    # ──────────────────────────────────────────────────────────────────────────
    # 单 GPU 完整视频处理（对外公开，兼容 v4 接口）
    # ──────────────────────────────────────────────────────────────────────────

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

        # [GAP-MULTIGPU-TRT-PREBUILD] TRT 预构建必须在多卡路由之前执行：
        # 主进程在 GPU0 上构建并缓存 .trt 文件，Workers 启动后读取同一缓存（
        # 同架构 GPU 的缓存文件名相同），避免各 Worker 独立重建导致 20-30 分钟
        # 的串行等待，也避免原先 TRT 预构建被多卡路由直接跳过的沉默失效问题。
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

        # ── 多卡路由 ──────────────────────────────────────────────────────────
        num_gpus  = torch.cuda.device_count()
        nwpg      = self.num_process_per_gpu
        n_workers = max(1, num_gpus * nwpg)
        if n_workers > 1 and num_gpus > 1:
            # [GAP-MULTIGPU-PREVIEW] 透传 preview 参数，多卡路径由 FrameCollector 侧实现预览
            return self._process_multi_gpu(
                input_path, output_path, scale, num_gpus, nwpg,
                preview=preview, preview_interval=preview_interval,
            )
        # ── 单 GPU 路径（含全套 v6.3.3 优化）────────────────────────────────

        audio_src = input_path if self.keep_audio else None

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
                slot_mb=_slot_mb,             # 传入每 slot 大小，用于 PinnedPool 内存约束
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
                _rq_retune = retune_result_q if retune_result_q is not None else _cur_result_q
                _rq_combined_raw = max(
                    (result_gpu_sug + _rq_retune) // 2,
                    _cur_result_q,
                )
                final_result_q = _rq_combined_raw
                # 硬上限
                final_pair_q   = min(final_pair_q, 8)
                final_result_q = min(final_result_q, 64)
                # [FIX-T3-MEMCAP / FIX-POOL-AUTOSCALE] 施加 PinnedPool 内存上限约束，并显式 log 截断原因
                if _slot_mb > 0.0:
                    _max_rq_mem = max(8, int(_PINNED_POOL_MAX_MB / _slot_mb) - 2)
                    if final_result_q > _max_rq_mem:
                        print(
                            f'[ADAPTIVE-QUEUE] PinnedPool 内存上限截断: '
                            f'result_queue {final_result_q} → {_max_rq_mem}'
                            f'  (slot={_slot_mb:.1f} MB × {_max_rq_mem+2} ≈ {_PINNED_POOL_MAX_MB:.0f} MB上限)'
                        )
                    final_result_q = min(final_result_q, _max_rq_mem)

                # [FIX-RETUNE-DISPLAY] 打印推导链路，让建议来源透明可追溯
                _retune_str = (f'RETUNE={_rq_retune}' if retune_result_q is not None
                               else 'RETUNE=N/A')
                print(
                    f'[ADAPTIVE-QUEUE] 下次将使用 pair_queue={final_pair_q} '
                    f'result_queue={final_result_q}'
                    f'  (GPU-MONITOR={result_gpu_sug} {_retune_str} → avg={_rq_combined_raw})'
                )
                # [FIX-T3-FPS] 非 T3-bottleneck 时也更新实测 T3 fps（更可靠）
                self._next_t3_fps_measured = getattr(self, '_last_t3_fps_measured', 0.0)

            self._next_pair_queue   = final_pair_q
            self._next_result_queue = final_result_q
        else:
            print('[GPU-MONITOR] 警告：未能获取任何 GPU 采样数据，'
                  '请检查 nvidia-ml-py 安装或驱动状态。')

        if ok:
            self._print_summary(input_path, output_path, fc, oc, scale)
            self._dump_report(input_path, output_path, fc, oc, scale, n_workers=1)
        return ok

    # ──────────────────────────────────────────────────────────────────────────
    # M1 v6: 多 GPU 帧范围分割处理 [FIX-TMP 零临时文件架构]
    # ──────────────────────────────────────────────────────────────────────────

    def _process_multi_gpu(
        self,
        input_path:       str,
        output_path:      str,
        scale:            float,
        num_gpus:         int,
        nwpg:             int,
        preview:          bool = False,           # [GAP-MULTIGPU-PREVIEW]
        preview_interval: int  = 30,              # [GAP-MULTIGPU-PREVIEW]
    ) -> bool:
        """
        [FIX-TMP] 零临时文件多卡架构。

        Worker 不再写无损中间 seg_*.mp4，改为将每帧以
        (output_global_idx, frame.tobytes(), frame.shape) 放入有界 result_q；
        主进程 FrameCollector 线程用 heapq 按 output_global_idx 重排，
        再直接写入唯一的 FFmpegWriter（最终编码直接完成，无 concat 步骤）。

        消除了：
          · 无损段磁盘 IO（4K 10s 视频 ~3GB）
          · Worker 端 FFmpegWriter 编码 + 主进程端 concat 解码
          · ffmpeg concat 本身的读取时间
        """
        import heapq as _heapq

        # [GAP-MULTIGPU-PREVIEW] --preview 在多卡模式下由 FrameCollector 主线程处理。
        # Worker 子进程为 spawn 模式，无显示上下文，不支持 cv2.imshow。
        if preview:
            try:
                import cv2 as _cv2_preview
                _preview_ok = True
            except ImportError:
                print('[Multi-GPU] [GAP-MULTIGPU-PREVIEW] 警告：cv2 未安装，--preview 在多卡模式下无效。')
                _preview_ok = False
        else:
            _preview_ok = False

        n_workers = num_gpus * nwpg
        meta      = _probe_video(input_path)
        nb        = meta['nb_frames']
        fps       = meta['fps']
        if nb <= 0:
            print('[Multi-GPU] 无法获取帧数，回退单 GPU 处理。')
            return self._process_single_fallback(input_path, output_path, scale)

        scale_frac = Fraction(scale).limit_denominator(64)
        scale_int  = int(scale_frac)
        new_fps    = fps * scale_int

        print(f'\n[Multi-GPU v6] 零临时文件帧范围分割模式')
        print(f'  GPU 数: {num_gpus} × {nwpg} Worker = {n_workers} 总 Worker')
        print(f'  总帧数: {nb} → 每段约 {nb // n_workers} 帧 | 输出帧率: {new_fps:.2f}')

        # 计算帧范围（带 1 帧 overlap）
        seg_size = nb // n_workers
        ranges: List[Tuple[int, int]] = []
        for i in range(n_workers):
            s = i * seg_size
            e = (i + 1) * seg_size - 1 if i < n_workers - 1 else nb - 1
            if i < n_workers - 1:
                e = min(e + 1, nb - 1)  # +1 overlap
            ranges.append((s, e))

        worker_args_list = []
        for i, (s, e) in enumerate(ranges):
            worker_args_list.append(dict(
                worker_idx         = i,
                device_id          = i % num_gpus,
                input_path         = input_path,
                scale              = scale,
                frame_start        = s,
                frame_end          = e,
                skip_first_output  = (i > 0),
                scale_int          = scale_int,
                model_path         = self.model_path,
                batch_size         = self.batch_size,
                max_batch_size     = self._max_batch_size,
                use_fp16           = self.use_fp16,
                use_compile        = self.use_compile,        # [FIX-WORKER-COMPILE] 透传真实值，原为硬编码 True，--no-compile 在多卡模式下无效
                use_tensorrt       = self.use_tensorrt,       # [FIX-WORKER-TRT] 透传 TRT 开关，原缺失，--use-tensorrt 在多卡模式下静默失效
                use_cuda_graph     = self.use_cuda_graph,
                use_hwaccel        = self.use_hwaccel,
                model_name         = self.model_name,         # [FIX-WORKER-MODEL] 透传模型变体名，原缺失，Worker 内硬编码 IFRNet_S
                crf                = self.crf,
                x264_preset        = self.x264_preset,
                ffmpeg_bin         = self.ffmpeg_bin,
                t2_cache_dir       = self.t2_cache_dir,       # [FIX-WORKER-T2CACHE] 透传缓存目录，保持接口一致性
                trt_cache_dir      = self.trt_cache_dir,      # [GAP-WORKER-TRT-CACHE-DIR] 透传 TRT Engine 缓存目录；Worker 加载主进程预构建的 .trt 缓存
                quiet              = self.quiet,               # [FIX-WORKER-QUIET] 透传静默开关，原硬编码 True，--no-quiet 在 Worker 无效
            ))

        ctx         = mp.get_context('spawn')
        # result_q 有界（200 帧缓冲），Frame data + __DONE__ 共用同一队列
        result_q:   mp.Queue = ctx.Queue(maxsize=200)
        error_q:    mp.Queue = ctx.Queue()

        processes = [
            ctx.Process(
                target = _ifrnet_segment_worker,
                kwargs = dict(**wa, error_q=error_q, result_q=result_q),
                daemon = True,
                name   = f'IFRWorker-{wa["worker_idx"]}',
            )
            for wa in worker_args_list
        ]
        # [GAP-MULTIGPU-GPUMONITOR] 启动后台 GPU 监测（监测主进程所在 GPU0，
        # 反映 FrameCollector / 主进程侧整体 GPU 负载；Worker 推理负载可通过
        # pynvml 多设备接口扩展，当前阶段仅监测设备 0）。
        # 注：多卡模式不运行 ADAPTIVE-QUEUE 调优（Worker 内各自 auto-tune），
        # 仅打印统计信息供参考。
        self._gpu_monitor.start()
        for p in processes:
            p.start()
        print(f'[Multi-GPU] {n_workers} 个 Worker 已启动...')

        # 探测视频尺寸（用于 FrameCollector）
        meta_probe = _probe_video(input_path)
        W, H = meta_probe['width'], meta_probe['height']

        final_codec = HardwareCapability.best_encoder(self.codec)
        audio_src   = input_path if self.keep_audio else None

        writer = FFmpegWriter(
            output_path, W, H, new_fps,
            codec      = final_codec,
            crf        = self.crf,
            preset     = self.x264_preset,
            audio_src  = audio_src,
            ffmpeg_bin = self.ffmpeg_bin,
            quiet      = self.quiet,   # [GAP-MULTIGPU-WRITER-QUIET] 透传静默开关，与单卡路径保持一致
        )
        print(f'[Multi-GPU] FrameCollector 启动，输出编码器: {final_codec}')

        # FrameCollector：主线程同步处理（避免额外线程同步开销）
        heap         = []   # (global_idx, bytes)
        next_expected = 0
        done          = 0
        errors        = []
        total_written = 0
        t_collect     = time.time()

        while done < n_workers:
            # 检查 error_q
            try:
                wid, err = error_q.get_nowait()
                errors.append(f'Worker-{wid}: {err}')
                print(f'\n[Worker-{wid} Error] {err}')
            except Exception:
                pass

            try:
                item = result_q.get(timeout=30.0)
            except Exception:
                # 超时：检查进程是否还活着
                alive = any(p.is_alive() for p in processes)
                if not alive and done < n_workers:
                    print('[Multi-GPU] 警告：所有进程已退出但未收到全部 __DONE__，强制结束收集。')
                    break
                continue

            if isinstance(item, tuple) and len(item) == 2 and item[0] == '__DONE__':
                done += 1
                print(f'[Multi-GPU] Worker-{item[1]} 完成 ({done}/{n_workers})')
                continue

            global_idx, frame_bytes, shape = item
            _heapq.heappush(heap, (global_idx, frame_bytes, shape))

            # 按序写出已就绪帧
            while heap and heap[0][0] == next_expected:
                idx, fb, sh = _heapq.heappop(heap)
                frame = np.frombuffer(fb, dtype=np.uint8).reshape(sh)
                writer.write(frame)
                # [GAP-MULTIGPU-PREVIEW] FrameCollector 侧预览（BGR，来自 Worker D2H 输出）
                if _preview_ok and total_written % preview_interval == 0:
                    _cv2_preview.imshow('[Multi-GPU] IFRNet Preview', frame)
                    if _cv2_preview.waitKey(1) & 0xFF == ord('q'):
                        _preview_ok = False
                next_expected += 1
                total_written += 1

        # flush 剩余帧（理论上 heap 应已空，但防御性处理）
        if heap:
            heap.sort()
            for idx, fb, sh in heap:
                frame = np.frombuffer(fb, dtype=np.uint8).reshape(sh)
                writer.write(frame)
                total_written += 1

        for p in processes:
            p.join(timeout=30)
            if p.is_alive():
                print(f'[Warning] {p.name} 未在超时内退出，强制终止。')
                p.kill()

        writer.close()
        elapsed = time.time() - t_collect
        print(f'[Multi-GPU] 收集完成 | 总输出帧={total_written} | 耗时={elapsed:.1f}s')

        # [GAP-MULTIGPU-GPUMONITOR] 停止采样，打印 GPU 利用率统计
        # 多卡模式不运行 ADAPTIVE-QUEUE（Worker 内各自 auto-tune），
        # 仅打印统计数据供调优参考，不更新 _next_pair/result_queue。
        self._gpu_monitor.stop()
        _mgpu_stats = self._gpu_monitor.get_stats()
        if _mgpu_stats.sample_count > 0:
            print()
            self._gpu_monitor.print_report(
                _mgpu_stats,
                current_bs       = self.batch_size,
                current_pair_q   = 4,    # Worker 默认值（多卡模式主进程无 pipeline 深度数据）
                current_result_q = 8,
            )
        else:
            print('[GPU-MONITOR] 警告：未能获取 GPU 采样数据（多卡模式仅监测 GPU0）。')

        if errors:
            print(f'[Error] {len(errors)} 个 Worker 出现错误:\n' + '\n'.join(errors))
            return False

        self._print_summary(input_path, output_path, nb, total_written, scale)
        # [GAP-MULTIGPU-REPORT-TIMING] 多卡模式调用专用报告函数：
        # 主进程 self._timing 为空（推理计时在 Worker 子进程中），
        # 改写宏观吞吐指标，避免报告被 _dump_report 守卫条件静默跳过。
        self._dump_report_multigpu(
            input_path, output_path, nb, total_written, scale,
            n_workers=n_workers, elapsed_s=elapsed,
        )
        return True



    def _process_single_fallback(self, input_path, output_path, scale):
        audio_src = input_path if self.keep_audio else None
        ok, fc, oc = self._process_segment(
            input_path, output_path, scale,
            audio_src=audio_src, worker_label='GPU0',
        )
        if ok:
            self._print_summary(input_path, output_path, fc, oc, scale)
            self._dump_report(input_path, output_path, fc, oc, scale, n_workers=1)
        return ok


    def _print_summary(self, input_path, output_path, fc, oc, scale):
        print(f'\n✅ 插帧完成！')
        if oc > 0:
            print(f'   原始帧数: {fc} → 输出帧数: {oc} (×{scale:.1f})')
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f'   输出: {output_path} ({size_mb:.1f} MB)')

    def _dump_report(self, input_path, output_path, fc, oc, scale, n_workers: int = 1):
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
            'n_workers':  n_workers,
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

    def _dump_report_multigpu(
        self,
        input_path:  str,
        output_path: str,
        fc:          int,
        oc:          int,
        scale:       float,
        n_workers:   int,
        elapsed_s:   float,
    ):
        """
        [GAP-MULTIGPU-REPORT-TIMING] 多卡模式专用 JSON 性能报告。

        单卡 _dump_report() 依赖 self._timing（推理批次延迟）生成 infer_latency_ms，
        但多卡模式下推理发生在 Worker 子进程，主进程 self._timing 始终为空，
        导致单卡版报告守卫条件（not self._timing）永远触发，报告被静默跳过。

        本函数改为记录宏观吞吐指标（总帧数、输出帧数、总耗时、帧率），
        infer_latency_ms 字段标注为不可用，完整保留其余配置字段以保持格式兼容。
        """
        if not self.report_json:
            return
        report = {
            'input':       input_path,
            'output':      output_path,
            'scale':       scale,
            'batch_size':  self.batch_size,
            'fp16':        self.use_fp16,
            'cuda_graph':  self.use_cuda_graph,
            'tensorrt':    getattr(self, '_trt_ok', False),
            'nvdec':       HardwareCapability.has_nvdec(),
            'nvenc':       HardwareCapability.best_encoder(self.codec).endswith('nvenc'),
            'n_workers':   n_workers,
            'frame_count':  fc,
            'output_count': oc,
            'elapsed_s':    round(elapsed_s, 1),
            'throughput_fps': round(fc / max(elapsed_s, 0.1), 1),
            # 推理延迟数据在 Worker 子进程中，主进程无法获取；记录说明供参考。
            'infer_latency_ms': 'N/A（多卡模式：推理延迟数据在 Worker 子进程中）',
        }
        with open(self.report_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f'   性能报告（多卡宏观指标）: {self.report_json}')


# ─────────────────────────────────────────────────────────────────────────────
# M1: 多 GPU Worker 函数（FIX-TMP，spawn 安全）
# ─────────────────────────────────────────────────────────────────────────────

def _ifrnet_segment_worker(
    worker_idx:       int,
    device_id:        int,
    input_path:       str,
    scale:            float,
    frame_start:      int,
    frame_end:        int,
    skip_first_output: bool,
    scale_int:        int,          # [FIX-TMP] 全局帧索引计算所需
    model_path:       str,
    batch_size:       int,
    max_batch_size:   int,
    use_fp16:         bool,
    use_compile:      bool,
    use_tensorrt:     bool,         # [FIX-WORKER-TRT] 透传 TRT 开关
    use_cuda_graph:   bool,
    use_hwaccel:      bool,
    model_name:       str,          # [FIX-WORKER-MODEL] 透传模型变体名，spawn 子进程必须重绑
    crf:              int,
    x264_preset:      str,
    ffmpeg_bin:       str,
    t2_cache_dir:     str = '',          # [FIX-WORKER-T2CACHE] 透传缓存目录，保持接口一致性
    trt_cache_dir:    Optional[str] = None,  # [GAP-WORKER-TRT-CACHE-DIR] 透传 TRT Engine 缓存目录
    quiet:            bool = True,       # [FIX-WORKER-QUIET] 透传静默开关
    error_q:          mp.Queue = None,
    result_q:         mp.Queue = None,    # [FIX-TMP] 帧数据 + __DONE__ 共用同一队列
):
    """
    v6 多 GPU Worker：[FIX-TMP] 不再写临时文件，改为将帧放入 result_q。
    [FIX-NML] 在 Worker 进程中安装 _NVMLFilter，屏蔽 NVML 无害日志。
    spawn 模式启动，进程隔离，携带全部 v6 升级。
    """
    label = f'GPU{device_id}-W{worker_idx}'

    # ── U1: spawn 子进程中必须重设 PYTORCH_ALLOC_CONF ──────────────────────
    os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

    # ── U2: 子进程中重新抑制噪音日志 ──────────────────────────────────────
    import logging as _log
    _log.getLogger('torch._inductor.utils').setLevel(_log.ERROR)
    _log.getLogger('torch.utils._sympy.interp').setLevel(_log.ERROR)
    _log.getLogger('torch.utils._sympy').setLevel(_log.ERROR)

    # ── [FIX-NML] Worker 进程安装 _NVMLFilter ────────────────────────────
    import sys as _sys, re as _re
    class _F:
        _p = _re.compile(r'NVML_SUCCESS|INTERNAL ASSERT FAILED.*CUDACachingAllocator')
        def __init__(self, s): self._s = s
        def write(self, m):
            if not self._p.search(m): self._s.write(m)
        def flush(self): self._s.flush()
        def __getattr__(self, a): return getattr(self._s, a)
    _sys.stderr = _F(_sys.stderr)

    # ── [FIX-WORKER-MODEL] spawn 子进程模块级 Model 默认为 IFRNet_S，
    # 必须按传入的 model_name 重新绑定，否则与主进程模型架构不一致。
    global Model
    Model, _ = _load_ifrnet_module(model_name)

    try:
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device)

        processor = IFRNetVideoProcessor(
            model_path     = model_path,
            device         = f'cuda:{device_id}',
            batch_size     = batch_size,
            max_batch_size = max_batch_size,
            use_fp16       = use_fp16,
            use_compile    = use_compile,
            use_cuda_graph = use_cuda_graph,
            use_tensorrt   = use_tensorrt,        # [FIX-WORKER-TRT] 透传 TRT 开关
            use_hwaccel    = use_hwaccel,
            codec          = 'libx264',  # [FIX-TMP] codec 不再用于 Worker
            crf            = crf,
            x264_preset    = x264_preset,
            keep_audio     = False,
            ffmpeg_bin     = ffmpeg_bin,
            model_name     = model_name,          # [FIX-WORKER-MODEL] 使用传入值，而非硬编码 IFRNet_S
            t2_cache_dir   = t2_cache_dir,        # [FIX-WORKER-T2CACHE] 透传缓存目录
            trt_cache_dir  = trt_cache_dir,       # [GAP-WORKER-TRT-CACHE-DIR] 透传 TRT Engine 缓存目录
            quiet          = quiet,               # [FIX-WORKER-QUIET] 透传静默开关
            num_process_per_gpu = 1,
        )

        # [GAP-WORKER-TRT-BUILD] Worker 不走 process_video()，_process_segment 内
        # 没有 TRT Engine 构建逻辑，必须在此手动触发构建/加载。
        # 主进程已在路由前预构建并写入缓存（GAP-MULTIGPU-TRT-PREBUILD），
        # 同架构 GPU 的 Worker 直接加载缓存文件（通常 < 5s），不会重新编译。
        if use_tensorrt:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _meta_trt = _probe_video(input_path)
            _trt_ceil_w = lambda x, s: x if x % s == 0 else x + (s - x % s)
            _trt_H_w = _trt_ceil_w(_meta_trt['height'], MODEL_STRIDE)
            _trt_W_w = _trt_ceil_w(_meta_trt['width'],  MODEL_STRIDE)
            _sh_w    = (batch_size, 3, _trt_H_w, _trt_W_w)
            _trt_dir_w = trt_cache_dir if trt_cache_dir else os.path.join(base_dir, '.trt_cache')
            print(f'[{label}] [GAP-WORKER-TRT-BUILD] 加载 TRT Engine (shape={_sh_w})...', flush=True)
            processor._build_trt_engine(_sh_w, _trt_dir_w)
            processor._trt_built = True

        # [FIX-TMP] 使用 result_q 代替 FFmpegWriter
        ok, fc, oc = processor._process_segment(
            input_path          = input_path,
            output_path         = '',     # [FIX-TMP] 不使用
            scale               = scale,
            frame_start         = frame_start,
            frame_end           = frame_end,
            skip_first_output   = skip_first_output,
            audio_src           = None,
            worker_label        = label,
            result_q            = result_q,
            scale_int_override  = scale_int,
        )

        if not ok:
            error_q.put((worker_idx, '_process_segment 返回失败'))

    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f'[{label}] 异常: {err_msg}')
        error_q.put((worker_idx, err_msg))

    finally:
        # [FIX-TMP] 发送完成信号（与帧数据共用 result_q）
        result_q.put(('__DONE__', worker_idx))



# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='IFRNet 视频插帧 —— 终极优化版 v6.3.4（多卡版）',
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
                        help='[覆盖] 强制启用 CUDA Graph，覆盖 --no-cuda-graph / config。'
                             '与 torch.compile 互斥（compile 成功时自动禁用 CUDA Graph）；'
                             '如需确保生效，请同时指定 --no-compile。')
    parser.add_argument('--use-compile', dest='use_compile_force',
                        action='store_true', default=False,
                        help='[覆盖] 强制启用 torch.compile，覆盖 --no-compile / config。'
                             '与 --use-tensorrt 互斥（TRT 激活时 compile 被跳过）。')
    parser.add_argument('--no-tensorrt', dest='no_tensorrt',
                        action='store_true', default=False,
                        help='[覆盖] 强制禁用 TensorRT，覆盖 --use-tensorrt')
    # 硬件加速
    parser.add_argument('--no-hwaccel', action='store_true', help='强制禁用 NVDEC')
    # 多 GPU
    parser.add_argument('--num-process-per-gpu', type=int, default=1,
                        help='每 GPU Worker 数（多 GPU 模式，显存充裕时可设 2）')
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
    parser.add_argument('--trt-cache-dir',   default=None, help='TRT Engine 缓存目录')
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

    # 根据 --model 选择对应的模型变体
    global Model
    Model, _ = _load_ifrnet_module(args.model)

    # 打印启动信息
    print('=' * 65)
    print('  IFRNet 视频插帧 —— 终极优化版 v6.3.4（多卡版）')
    print('=' * 65)
    num_gpus  = torch.cuda.device_count()
    n_workers = max(1, num_gpus * args.num_process_per_gpu)
    print(f'  模型:   {args.model}')
    print(f'  设备:   {args.device} | GPU 数: {num_gpus} | Workers: {n_workers}')
    if torch.cuda.is_available():
        for _i in range(num_gpus):
            _props = torch.cuda.get_device_properties(_i)
            print(f'    GPU{_i}: {_props.name} ({_props.total_memory/1024**3:.1f} GB)')
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
        num_process_per_gpu = getattr(args, 'num_process_per_gpu', 1),
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