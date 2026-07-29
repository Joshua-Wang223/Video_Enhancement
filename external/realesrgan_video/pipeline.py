#!/usr/bin/env python3
"""
Real-ESRGAN Video Enhancement - 深度流水线模块
包含：GPUMemoryPool, DeepPipelineOptimizer, SR推理函数
"""

import gc
import os
import sys
import time
import queue
import threading
import concurrent.futures
from typing import List, Optional, Tuple, Dict, Any

# [FIX-NVML] 明确禁用 PyTorch 基于 NVML 的 CUDA 检测，
# 避免因系统 NVML/RM 版本不匹配导致 INTERNAL ASSERT FAILED。
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")

# [FIX-OOM-TRT] expandable_segments 使用 cudaMallocAsync 分配器，
# 大幅减少 PyTorch CUDA allocator 碎片化，防止 OOM 降级时批大小
# 因碎片而非真实需求持续塌方（参照 IFRNet v6.4.5.1 L322）。
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realesrgan_utils import ThroughputMeter, _get_pinned_pool
from face_utils import _detect_faces_batch, _paste_faces_batch, _make_detect_helper
from async_dispatcher import AsyncGFPGANDispatcher
from gfpgan_subprocess import SharedMemoryDoubleBuffer


def _sr_infer_batch(
    upsampler,
    frames: List[np.ndarray],
    outscale: float,
    netscale: int,
    transfer_stream,
    compute_stream,
    trt_accel,
    cuda_graph_accel=None,
    prefetched_batch_t=None,
    defer_resolve: bool = False,
):
    """纯 SR 推理：H2D → 模型前向 → 后处理 → D2H。"""
    device = upsampler.device
    use_half = upsampler.half
    pool = _get_pinned_pool()
    B = len(frames)
    t0 = time.perf_counter()

    if (prefetched_batch_t is not None and
            prefetched_batch_t.shape[0] == B):
        batch_t = prefetched_batch_t
        if transfer_stream is not None:
            torch.cuda.current_stream(device).wait_stream(transfer_stream)
    else:
        batch_pin = pool.get_for_frames(frames)
        if transfer_stream is not None:
            with torch.cuda.stream(transfer_stream):
                batch_t = batch_pin.to(device, non_blocking=True)
                batch_t = batch_t.permute(0, 3, 1, 2).float().div_(255.0)
                if use_half:
                    batch_t = batch_t.half()
            # [FIX-PINNED-POOL-RACE] 记录本次 H2D 拷贝所在 stream 的 event，
            # 供 pool 下次轮到这个 pinned buffer slot 时正确等待，
            # 避免下一批 CPU 写入和这次 GPU 异步读取竞争同一块内存。
            pool.mark_issued(transfer_stream)
        else:
            batch_t = batch_pin.to(device)
            batch_t = batch_t.permute(0, 3, 1, 2).float().div_(255.0)
            if use_half:
                batch_t = batch_t.half()

    if trt_accel is not None and trt_accel.available:
        if transfer_stream is not None:
            torch.cuda.current_stream(device).wait_stream(transfer_stream)
        output_t = trt_accel.infer(batch_t).float()
    elif cuda_graph_accel is not None and cuda_graph_accel.available:
        if transfer_stream is not None and compute_stream is not None:
            compute_stream.wait_stream(transfer_stream)

        def _eager_model(x):
            with torch.no_grad():
                return upsampler.model(x)

        output_t = cuda_graph_accel.infer_or_eager(batch_t, _eager_model)
        if compute_stream is not None:
            torch.cuda.default_stream(device).wait_stream(compute_stream)
    else:
        if transfer_stream is not None and compute_stream is not None:
            compute_stream.wait_stream(transfer_stream)
            with torch.cuda.stream(compute_stream):
                with torch.no_grad():
                    output_t = upsampler.model(batch_t)
        else:
            with torch.no_grad():
                output_t = upsampler.model(batch_t)

    if compute_stream is not None:
        torch.cuda.default_stream(device).wait_stream(compute_stream)

    if abs(outscale - netscale) > 1e-5:
        output_t = F.interpolate(
            output_t.float(), scale_factor=outscale / netscale,
            mode='bicubic', align_corners=False,
        )

    if defer_resolve:
        # [FIX-DEFER-RESOLVE] 延迟解析路径：发起 D2H 后立即返回句柄，
        # 不在此等待 event / CPU 拷贝 / stream 同步——这些由调用方的解析线程
        # 在下一批 TRT 推理（主机同步 ~700ms）期间执行（_sr_materialize），
        # 实现 CPU 拷贝与 GPU 推理跨批重叠，消除每批 ~25-45ms 串行尾部。
        # GPU 张量引用随句柄保留至解析时释放（比原路径晚 ~1 批，显存峰值
        # 多占 ~1 批）；句柄携带私有 event，不依赖池的 slot 事件注册表，
        # 避免 slot 复用后事件被覆写导致的错误等待。
        out_u8 = output_t.float().clamp_(0.0, 1.0).mul_(255.0).byte()
        out_perm = out_u8.permute(0, 2, 3, 1).contiguous()
        out_pinned = pool.get_output_buf(out_perm.shape, torch.uint8)
        out_pinned.copy_(out_perm, non_blocking=True)
        pool.mark_output_issued()
        _ev = torch.cuda.Event()
        _ev.record()   # 当前流，紧随 D2H 发起
        handle = {
            '_pending_sr': True,
            'event': _ev,
            'view': out_pinned,
            'B': B,
            'gpu_refs': (batch_t, output_t, out_u8, out_perm),
        }
        elapsed = time.perf_counter() - t0
        timing_info = {'batch_size': B, 'processing_time': elapsed}
        return [handle], timing_info, 'success'

    out_u8 = output_t.float().clamp_(0.0, 1.0).mul_(255.0).byte()
    out_perm = out_u8.permute(0, 2, 3, 1).contiguous()
    out_pinned = pool.get_output_buf(out_perm.shape, torch.uint8)
    out_pinned.copy_(out_perm, non_blocking=True)
    # [FIX-PINNED-POOL-RACE] 上面这行是异步 D2H 拷贝，之前紧接着就对 out_pinned
    # 做 .numpy() 读取——但 GPU 可能还没真正把数据写完（甚至还没开始写），读到的
    # 是"半新半旧"或纯粹上一批遗留的数据。这正是分段视频里出现"某几帧内容和
    # 更早批次逐像素相同"现象的根因。这里显式记录 event 并阻塞等待拷贝真正完成，
    # 再读取 numpy 数据。
    pool.mark_output_issued()
    pool.wait_output_ready()

    out_np = out_pinned.numpy()
    sr_results = [out_np[i].copy() for i in range(B)]

    # [FIX-CUDAFREEASYNC] expandable_segments 模式下 del 触发
    # cudaFreeAsync（非同步 cudaFree），必须在 del 后 synchronize
    # 排空异步释放，否则下一批 alloc 时旧内存尚未归还 →
    # CUDA pool 膨胀 → allocated 每批 +30MB 持续上升。
    # 原 synchronize 在 del 之前，只排空了计算流，不排空 free。
    #
    # [FIX-NARROW-SYNC] 原 torch.cuda.synchronize(device) 是全设备同步，
    # 会阻塞该 CUDA context 下**所有** stream（包括 NVENC 编码线程的专用
    # stream_encode、transfer_stream、gfpgan_stream），每个 SR batch 都
    # 触发一次——这是比 NVENC 那次 legacy-stream 拷贝更频繁的"隐式全局
    # 屏障"来源，会周期性阻塞编码/GFPGAN 与 SR 推理的并行重叠。
    # del 触发的 cudaFreeAsync 是 stream-ordered 的：本函数此处的 del 发生
    # 在 current_stream（此前的 compute_stream/transfer_stream 相关工作
    # 已通过 wait_stream 汇入 current_stream，且 pool.wait_output_ready()
    # 已经用 event 确认了 D2H 拷贝完成），因此只需同步 current_stream 即可
    # 保证这次 free 被排空，无需等待其他无关 stream 上的工作。
    del batch_t, output_t, out_u8, out_perm, out_pinned
    torch.cuda.current_stream(device).synchronize()

    elapsed = time.perf_counter() - t0
    timing_info = {'batch_size': B, 'processing_time': elapsed}
    return sr_results, timing_info, 'success'


def _sr_materialize(all_sr: list) -> list:
    """[FIX-DEFER-RESOLVE] 把 _sr_infer_batch 延迟句柄解析为 numpy 帧列表。

    严格按列表顺序处理，OOM 回退路径产出的普通 ndarray 元素原样透传，
    因此混合列表（句柄 + 帧）也能保持帧序不变。
    句柄解析：等待私有 D2H event → CPU 拷贝出 pinned buffer → 释放 GPU 引用。
    不在此做 stream synchronize：引用释放产生的 cudaFreeAsync 随后续批次
    流推进自然排空（_process_sr 每 100 批有深度清理兜底，OOM 路径亦有全同步）。
    """
    results = []
    for item in all_sr:
        if isinstance(item, dict) and item.get('_pending_sr'):
            item['event'].synchronize()
            out_np = item['view'].numpy()
            results.extend(out_np[i].copy() for i in range(item['B']))
            item['view'] = None
            item['gpu_refs'] = None   # 释放 GPU 张量引用（refcount 归零即释放）
        else:
            results.append(item)
    return results


class GPUMemoryPool:
    """流水线并发槽计数器（纯信号量）"""

    def __init__(self, max_batches: int = 4, batch_size: int = 4,
                 img_size: Tuple[int, int] = (540, 960), device: str = 'cuda'):
        self.max_batches = max_batches
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device
        self._slots = queue.Queue()
        for i in range(max_batches):
            self._slots.put(i)
        self.lock = threading.Lock()

    def acquire(self) -> Optional[Dict[str, Any]]:
        try:
            idx = self._slots.get_nowait()
            return {'index': idx}
        except queue.Empty:
            return None

    def release(self, idx: int):
        self._slots.put(idx)


class DeepPipelineOptimizer:
    """深度流水线优化器 - 4级并行处理"""

    def __init__(self, upsampler, face_enhancer, args, device, trt_accel=None,
                 input_h: int = 540, input_w: int = 960):
        self.upsampler = upsampler
        self.face_enhancer = face_enhancer
        self.args = args
        self.device = device

        # [FIX-BS-CAP-DYNAMIC] batch 硬顶由分辨率+显存联合决定，替换原固定 24：
        # 小分辨率（≤640x360）TRT 启动/调度开销占比高，允许 32-48 摊薄；
        # 高分辨率（>640x360）维持 24。VRAM 护栏沿用 _estimate_safe_batch_size
        # 的 6× 张量模型（B × H×W×3×2×6 ≤ 20% 总显存）。
        # 当前 config batch_size=24 时行为与原来完全一致。
        _bs_cap = 24
        if input_h * input_w <= 640 * 360:
            _bs_cap = 48
            if torch.cuda.is_available():
                _total_b = torch.cuda.get_device_properties(0).total_memory
                _bs_by_vram = int(_total_b * 0.20 / max(input_h * input_w * 36, 1))
                _bs_cap = max(24, min(48, _bs_by_vram))
        self.optimal_batch_size = min(args.batch_size, _bs_cap)

        # [FIX-QUEUE-RIGHTSIZE] 上游缓冲右置（参数化，可用 config 覆盖）：
        # SR 是唯一瓶颈时 F/D 队列常年全满，48/32 批在飞缓冲（≈1.3GB RAM）
        # 对吞吐零贡献，仅增加段尾延迟与内存压力；4-8 批已足够吸收
        # 读取/检测抖动（实测 σ ≈ 1 批）。S/G 保持 16：高分辨率下
        # 写端抖动需要更深出口缓冲。
        _frame_q_size = int(getattr(args, 'frame_queue_size', 8))
        _detect_q_size = int(getattr(args, 'detect_queue_size', 4))
        self.frame_queue = queue.Queue(maxsize=_frame_q_size)
        self.detect_queue = queue.Queue(maxsize=_detect_q_size)
        self.sr_queue = queue.Queue(maxsize=16)
        self.gfpgan_queue = queue.Queue(maxsize=16)

        self.memory_pool = GPUMemoryPool(
            max_batches=8,
            batch_size=self.optimal_batch_size,
            img_size=(input_h, input_w),
            device=device
        )

        self.detect_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix='opt_detect'
        )
        self.paste_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix='opt_paste'
        )

        # [FIX-DEFER-RESOLVE] 延迟解析执行器（单线程保证 sr_queue 入队顺序=提交顺序）。
        # 解析（D2H event 等待 + CPU 拷贝 + 入队）在此线程执行，与 SR 线程的
        # 下一批 TRT 推理重叠，消除每批 ~25-45ms 串行尾部（实测段 FPS +4-6%）。
        # 可用 args.defer_sr_resolve=False 回退原同步路径。
        self._defer_resolve = getattr(args, 'defer_sr_resolve', True)
        self._resolve_executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix='sr_resolve')
            if self._defer_resolve else None
        )

        self.transfer_stream = torch.cuda.Stream(device=device)
        self.sr_stream = torch.cuda.Stream(device=device)
        self.gfpgan_stream = torch.cuda.Stream(device=device)

        self.meter = ThroughputMeter()
        self.timing = []

        self.running = True
        self.trt_accel = trt_accel
        self.cuda_graph_accel = None
        self._face_frames_total = 0
        self._face_count_total = 0

        self.face_det_threshold = getattr(args, 'face_det_threshold', 0.5)
        self._face_filtered_total = 0

        self._face_density_ema = 0.0
        self._face_density_alpha = 0.3
        self._low_face_threshold = 2.0
        self._high_face_threshold = 5.0
        self._base_batch_size = self.optimal_batch_size
        # [FIX-ADAPTIVE-BS-INVERSION] 原硬顶 min(base*2, 12) 是 base=6 时代的遗留：
        # base=24 时 max_adaptive=12 < base，人脸低密度（便宜场景）反被压到 bs=12，
        # 自适应方向完全反转，SR 吞吐损失 ~50%。上限改为随 base 缩放（硬顶 48）。
        self._max_adaptive_batch = min(self._base_batch_size * 2, 48)
        self._min_adaptive_batch = max(2, self._base_batch_size // 2)
        self._adaptive_batch_lock = threading.Lock()
        self._adaptive_read_batch_size = self.optimal_batch_size
        self._enable_adaptive_batch = getattr(args, 'adaptive_batch', True)

        # ── OOM 冷却与统计（参照 IFRNet v6.4.5.1 _safe_infer）──
        self._oom_cooldown    = 0    # 正数 = 冷却轮数，期间不恢复 batch_size
        self._oom_total_count = 0    # 累计 OOM 次数（诊断用）
        self._oom_max_history = 5    # 保留最近 N 次 OOM 的 bs/retry_bs
        # ── TRT OOM 恢复跟踪 ──
        self._trt_working_bs     = 0  # OOM 后记住的成功临时 bs（避免每批都从 optimal 开始）
        self._trt_success_streak = 0  # 连续成功批次数，≥5 后尝试恢复 optimal_batch_size

        self.detect_helper = _make_detect_helper(face_enhancer, device) if face_enhancer else None

        # ----- 支持外部注入的 GFPGAN 子进程（多片段复用）-----
        self.gfpgan_subprocess = None
        try:
            _prestarted = getattr(args, '_early_gfpgan_subprocess', None)
        except Exception:
            _prestarted = None

        if _prestarted is not None:
            if hasattr(_prestarted, 'process') and _prestarted.process.is_alive():
                self._vlog('[优化架构] 使用外部预启动 GFPGAN 子进程（复用模式）')
                self.gfpgan_subprocess = _prestarted
                args._early_gfpgan_subprocess = None  # 避免重复使用
            else:
                self._vlog('[优化架构] 外部预启动 GFPGAN 子进程已死亡，关闭并回退')
                try:
                    _prestarted.close()
                except Exception as e:
                    print(f'[优化架构] 关闭死亡子进程错误: {e}')
                args._early_gfpgan_subprocess = None
                self.gfpgan_subprocess = None

        if (self.gfpgan_subprocess is None and
                getattr(args, 'gfpgan_trt', False) and
                face_enhancer is not None):
            if not getattr(args, '_gfpgan_trt_failed', False):
                self._vlog('[优化架构] 启用子进程GFPGAN TRT加速（非预启动路径）')
                try:
                    from gfpgan_subprocess import GFPGANSubprocess
                    self.gfpgan_subprocess = GFPGANSubprocess(
                        face_enhancer=face_enhancer, device=device,
                        gfpgan_weight=args.gfpgan_weight,
                        gfpgan_batch_size=args.gfpgan_batch_size,
                        use_fp16=not args.no_fp16, use_trt=True,
                        trt_cache_dir=getattr(args, 'trt_cache_dir', None),
                        gfpgan_model=args.gfpgan_model,
                    )
                except Exception as e:
                    print(f'[优化架构] GFPGAN 子进程创建失败: {e}')
                    self.gfpgan_subprocess = None
                    args._gfpgan_trt_failed = True

        self._async_dispatcher: Optional[AsyncGFPGANDispatcher] = None
        self._task_id_counter = 0
        self._task_id_lock = threading.Lock()

        # [FIX-QUEUE-LABEL] G 队列实际为 writer 编码前的最终输出缓冲；
        # face_enhance 关闭时改名为 O(Output)，避免误解为"人脸增强队列仍有排队"。
        self._g_label = 'G' if (self.face_enhancer is not None
                                or self.gfpgan_subprocess is not None) else 'O'

        # 线程句柄
        self._read_thread = None
        self._detect_thread = None
        self._sr_thread = None
        self._gfpgan_thread = None

        # 上游 reader 句柄（由 optimize_pipeline 填充，供监控/看门狗使用）
        self.reader = None

        # 静默模式：从 args 读取，默认开启
        self._quiet = getattr(args, 'quiet', True)

    def _vlog(self, *args, **kwargs):
        """受静默模式控制的日志打印（仅在 --no-quiet 时输出）"""
        if not getattr(self, '_quiet', True):
            print(*args, **kwargs)

    def optimize_pipeline(self, reader, writer, pbar, total_frames):
        """运行优化的深度流水线"""
        self.reader = reader  # 保存引用，供 _write_frames 显示 prefetch 水位
        self._vlog("[优化架构] 启动深度流水线处理...")
        self._vlog(f"[优化架构] 队列深度: "
              f"P{reader.get_queue_capacity() if reader is not None else '?'}"
              f"/F{self.frame_queue.maxsize}"
              f"/D{self.detect_queue.maxsize}"
              f"/S{self.sr_queue.maxsize}"
              f"/{self._g_label}{self.gfpgan_queue.maxsize}")
        if self._g_label == 'O':
            self._vlog("[优化架构] 人脸增强关闭，O 队列为编码输出缓冲（非 GFPGAN 队列）")
        self._vlog(f"[优化架构] 内存池: {self.memory_pool.max_batches}批次")
        self._vlog(f"[优化架构] 最优batch_size: {self.optimal_batch_size}")
        if self.face_enhancer is not None:
            self._vlog(f"[优化架构] 人脸检测置信度阈值: {self.face_det_threshold}")
            if self._enable_adaptive_batch:
                self._vlog(f"[优化架构] 自适应批处理: 开启 (范围 {self._min_adaptive_batch}~{self._max_adaptive_batch}, "
                      f"低密度阈值={self._low_face_threshold}, 高密度阈值={self._high_face_threshold})")
        else:
            self._vlog(f"[优化架构] 自适应批处理: 关闭")

        if self.gfpgan_subprocess is not None:
            self._vlog('[优化架构] 等待 GFPGAN Inference 进程初始化（加载 .trt + warmup）...')
            max_elapsed = 2700
            deadline = time.time() + max_elapsed
            ready = False
            _poll_interval = 5
            _report_every = 300
            _last_report = time.time() - _report_every
            while time.time() < deadline:
                if not self.gfpgan_subprocess.process.is_alive():
                    exitcode = self.gfpgan_subprocess.process.exitcode
                    if exitcode == 0:
                        print('[优化架构] GFPGAN 子进程因 CUDA context 污染主动退出，'
                              '降级到主进程内 GFPGAN（PyTorch FP16）路径')
                    else:
                        print(f'[优化架构] GFPGAN 子进程意外退出（exitcode={exitcode}），回退 PyTorch')
                    break
                if self.gfpgan_subprocess.ready_event.wait(timeout=_poll_interval):
                    time.sleep(1.0)
                    if not self.gfpgan_subprocess.process.is_alive():
                        exitcode = self.gfpgan_subprocess.process.exitcode
                        if exitcode == 0:
                            print('[优化架构] GFPGAN TRT warmup 失败，子进程主动退出（exitcode=0），'
                                  '降级到主进程内 GFPGAN PyTorch 路径')
                        else:
                            print(f'[优化架构] GFPGAN 子进程 ready 后意外退出（exitcode={exitcode}），'
                                  '回退 PyTorch 路径')
                        break
                    ready = True
                    break
                now = time.time()
                if now - _last_report >= _report_every:
                    elapsed = now - (deadline - max_elapsed)
                    self._vlog(f'[优化架构] 等待中... {elapsed:.0f}s（Inference 进程初始化中）', flush=True)
                    _last_report = now
            if ready:
                self._vlog('[优化架构] GFPGAN 子进程已就绪，启动流水线')
                _shm = getattr(self.gfpgan_subprocess, 'shm_buf', None)
                self._async_dispatcher = AsyncGFPGANDispatcher(
                    self.gfpgan_subprocess, shm_buf=_shm)
                self._vlog('[优化架构] AsyncGFPGANDispatcher 已创建'
                      f' (shm={"是" if _shm else "否"})')
            else:
                self._vlog('[优化架构] GFPGAN 子进程未就绪，回退 PyTorch 路径')
                self.gfpgan_subprocess = None

        # 启动读取线程
        read_thread = threading.Thread(target=self._read_frames, args=(reader,), daemon=True)
        read_thread.start()

        # 启动检测线程
        detect_thread = threading.Thread(target=self._detect_faces, daemon=True)
        detect_thread.start()

        # 启动SR处理线程
        sr_thread = threading.Thread(target=self._process_sr, daemon=True)
        sr_thread.start()

        # 启动GFPGAN处理线程
        gfpgan_thread = threading.Thread(target=self._process_gfpgan, daemon=True)
        gfpgan_thread.start()

        self._read_thread = read_thread
        self._detect_thread = detect_thread
        self._sr_thread = sr_thread
        self._gfpgan_thread = gfpgan_thread

        # 主线程处理写入
        self._write_frames(writer, pbar, total_frames)

        # [EARLY-FLUSH] 提前触发 NVENC 编码收尾，在管线线程清理之前启动。
        # 此时所有帧已由主线程同步提交到编码队列（write_frame_batch →
        # NVENCWriter.submit → _NVENCEncodeThread._q.put），发送 SENTINEL
        # 排在队尾绝对安全。begin_flush() 为非阻塞（仅 queue.put, ~μs），
        # 编码线程收到后立即异步执行最终 chunk 编码 + per-slot EOS drain
        # （GPU 操作），主线程同步继续下方 CPU 线程 join/清理，两者真并行。
        # PreviewWriter 经 __getattr__ 透明代理到内部 NVENCWriter：
        # hasattr 对 NVENCWriter → True，对 FFmpegWriter → False 自动跳过。
        if hasattr(writer, 'begin_flush'):
            writer.begin_flush()

        # 终止所有流水线线程
        self.running = False

        for q_name, q in [('frame', self.frame_queue),
                          ('detect', self.detect_queue),
                          ('sr', self.sr_queue)]:
            try:
                q.put(None, timeout=1.0)
            except (queue.Full, Exception):
                pass

        _JOIN_TIMEOUT = 15.0
        for name, t in [('read', read_thread), ('detect', detect_thread),
                        ('sr', sr_thread), ('gfpgan', gfpgan_thread)]:
            t.join(timeout=_JOIN_TIMEOUT)
            if t.is_alive():
                print(f"\n[Pipeline] 警告: {name} 线程未在 {_JOIN_TIMEOUT:.0f}s 内退出",
                      flush=True)

    def _get_reader_state(self):
        """获取上游 reader 的当前水位/状态。返回 (p_size, p_cap, alive, eof_sent, produced)"""
        if self.reader is None:
            return (-1, 0, False, False, -1)
        try:
            p_size = self.reader.get_queue_size()
            p_cap = self.reader.get_queue_capacity()
            alive = self.reader.is_reader_alive()
            eof_sent = self.reader.is_eof_sent()
            produced = self.reader.get_frames_produced() if hasattr(
                self.reader, 'get_frames_produced') else -1
            return (p_size, p_cap, alive, eof_sent, produced)
        except Exception:
            return (-1, 0, False, False, -1)

    def _queue_status_str(self) -> str:
        """统一的队列状态字符串，含 prefetch"""
        p_size, p_cap, alive, eof_sent, _ = self._get_reader_state()
        state = 'alive' if alive else ('eof' if eof_sent else 'DEAD')
        return (f"P:{p_size}/{p_cap}[{state}]/"
                f"F:{self.frame_queue.qsize()}/"
                f"D:{self.detect_queue.qsize()}/"
                f"S:{self.sr_queue.qsize()}/"
                f"{self._g_label}:{self.gfpgan_queue.qsize()}")

    def _dump_all_queues(self):
        import sys, queue, collections
        seen = set()
        lines = []
        def visit(obj, path, depth=0):
            if depth > 3 or id(obj) in seen:
                return
            seen.add(id(obj))
            if isinstance(obj, (queue.Queue, queue.LifoQueue, queue.PriorityQueue)):
                try:
                    lines.append(f"  {path}: {obj.qsize()}/{obj.maxsize}  [{type(obj).__name__}]")
                except Exception as e:
                    lines.append(f"  {path}: <err {e}>")
                return
            if isinstance(obj, collections.deque):
                lines.append(f"  {path}: len={len(obj)} maxlen={obj.maxlen}  [deque]")
                return
            if hasattr(obj, '__dict__'):
                for k, v in vars(obj).items():
                    if k.startswith('_'):
                        continue
                    visit(v, f"{path}.{k}", depth+1)
        visit(self, "self")

        # 额外补上 reader 的 prefetch 队列
        p_size, p_cap, alive, eof_sent, produced = self._get_reader_state()
        state = 'alive' if alive else ('eof' if eof_sent else 'DEAD')
        lines.append(f"  self.reader._frame_queue: {p_size}/{p_cap}  "
                     f"[Queue, state={state}, produced={produced}]")

        sys.stderr.write("[QDUMP-FULL]\n" + "\n".join(lines) + "\n")
        sys.stderr.flush()

    def _dump_all_threads_stack(self):
        """打印所有线程的栈，用于死锁现场诊断"""
        import sys, traceback, threading
        try:
            for tid, frame in sys._current_frames().items():
                name = next((t.name for t in threading.enumerate()
                             if t.ident == tid), str(tid))
                sys.stderr.write(f"\n--- Thread {name} ({tid}) ---\n")
                traceback.print_stack(frame, file=sys.stderr)
            sys.stderr.flush()
        except Exception as e:
            print(f"\n[Pipeline] dump 线程栈失败: {e}", flush=True)

    def _read_frames(self, reader):
        """读取视频帧到队列（修复版：
        1) 只 catch Exception，不再吞 BaseException；
        2) 连续 FRAME_TIMEOUT 看门狗 + reader 探活；
        3) finally 无条件发送哨兵；
        4) ★ 不再在 finally 中设置 self.running = False ★
           —— 让 sentinel 正常沿流水线传播，避免下游在队列里的帧被丢弃。
        """
        import traceback
        frames_read = 0
        batch_frames = []
        consecutive_timeouts = 0
        # 约 60s 没拿到任何帧 → 判定 reader 死锁
        MAX_CONSECUTIVE_TIMEOUTS = 30   # get_frame() timeout 是 2s，30 次 ≈ 60s
        _reader_dead_reported = False

        try:
            while self.running:
                try:
                    img = reader.get_frame()
                except Exception as e:
                    print(f"\n[Reader] ❌ reader.get_frame() 抛异常 "
                          f"@frame={frames_read}: {type(e).__name__}: {e}",
                          flush=True)
                    traceback.print_exc()
                    break

                # 超时哨兵：队列暂时为空
                if img is reader.FRAME_TIMEOUT:
                    consecutive_timeouts += 1

                    # 探活：如果 reader 线程已经死了却没送 EOF → 主动兜底
                    if hasattr(reader, 'is_reader_alive') and not reader.is_reader_alive():
                        if hasattr(reader, 'is_eof_sent') and reader.is_eof_sent():
                            # 预计下一次 get 就能拿到 None，继续循环即可
                            continue
                        if not _reader_dead_reported:
                            print(f"\n[Reader] ⚠️ reader 线程已死亡但未送 EOF "
                                  f"@frame={frames_read}，强制收尾",
                                  flush=True)
                            _reader_dead_reported = True
                        break

                    # 超时看门狗
                    if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                        print(f"\n[Reader] ❌ 连续 {consecutive_timeouts} 次 FRAME_TIMEOUT "
                              f"(~{consecutive_timeouts*2}s) 无帧 "
                              f"@frame={frames_read}，判定 reader 死锁，主动终止",
                              flush=True)
                        # 再次打印上游状态
                        if hasattr(reader, 'is_reader_alive'):
                            print(f"\n[Reader]   reader 线程存活: "
                                  f"{reader.is_reader_alive()}", flush=True)
                        break

                    # 定期轻量心跳，帮助定位
                    if consecutive_timeouts % 10 == 0:
                        self._vlog(f"\n[Reader] FRAME_TIMEOUT 已累计 {consecutive_timeouts} 次 "
                              f"(~{consecutive_timeouts*2}s) @frame={frames_read}，"
                              f"等待上游...", flush=True)
                    continue

                # 收到真正的数据，重置超时计数
                consecutive_timeouts = 0

                # EOF
                if img is None:
                    self._vlog(f"\n[Reader] EOF reached at frame {frames_read}", flush=True)
                    if batch_frames:
                        put_ok = False
                        for _ in range(30):            # 给末批更多耐心（30s）
                            if not self.running:
                                break
                            try:
                                self.frame_queue.put((batch_frames, True), timeout=1.0)
                                put_ok = True
                                break
                            except queue.Full:
                                continue
                        if not put_ok:
                            print(f"\n[Reader] 警告: 最后一批 {len(batch_frames)} 帧未能入队",
                                  flush=True)
                    break

                frames_read += 1
                batch_frames.append(img)

                _current_bs = (self._adaptive_read_batch_size
                               if self._enable_adaptive_batch
                               else self.optimal_batch_size)

                if len(batch_frames) >= _current_bs:
                    while self.running:
                        try:
                            self.frame_queue.put((batch_frames.copy(), False), timeout=1.0)
                            break
                        except queue.Full:
                            continue
                    batch_frames = []

        except Exception as e:
            print(f"\n[Reader] ❌ _read_frames 异常 @frame={frames_read}: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        finally:
            self._vlog(f"\n[Reader] 线程退出，frames_read={frames_read}, "
                  f"running={self.running}", flush=True)
            # 关键：发送终止哨兵，避免下游干等
            # ★ 给足耐心（最多 60s），因为此时 F/D 队列可能仍然接近满
            _sent = False
            for _ in range(60):
                try:
                    self.frame_queue.put((None, True), timeout=1.0)
                    _sent = True
                    break
                except queue.Full:
                    continue
                except Exception:
                    break
            if not _sent:
                print(f"\n[Reader] ❌ 致命: 终止哨兵未能送入 frame_queue，"
                      f"下游可能需要靠超时退出", flush=True)
            else:
                self._vlog(f"\n[Reader] ✓ 终止哨兵已送达 frame_queue", flush=True)

            # ★★★ 关键修复：不再设置 self.running = False ★★★
            # 下游线程必须继续运行以消化 F/D/S/G 队列中已积压的帧，
            # 然后依靠哨兵自然退出。self.running=False 只应由 close() 触发。

    def _detect_faces(self):
        """人脸检测处理"""
        _sentinel_sent = False
        try:
            while self.running:
                try:
                    batch_data = self.frame_queue.get(timeout=1.0)

                    if batch_data is None:
                        self.detect_queue.put(None)
                        _sentinel_sent = True
                        break

                    batch_frames, is_end = batch_data

                    if batch_frames is None:
                        self.detect_queue.put((None, None, True))
                        _sentinel_sent = True
                        break

                    if self.detect_helper:
                        future = self.detect_executor.submit(
                            _detect_faces_batch, batch_frames, self.detect_helper,
                            self.face_det_threshold
                        )
                        face_data, _fw, _nf, _filtered = future.result()
                        self._face_frames_total += _fw
                        self._face_count_total += _nf
                        self._face_filtered_total += _filtered

                        while self.running:
                            try:
                                self.detect_queue.put((batch_frames, face_data, is_end), timeout=1.0)
                                break
                            except queue.Full:
                                continue
                    else:
                        while self.running:
                            try:
                                self.detect_queue.put((batch_frames, None, is_end), timeout=1.0)
                                break
                            except queue.Full:
                                continue

                except queue.Empty:
                    continue
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"人脸检测错误: {e}")
        finally:
            if not _sentinel_sent:
                try:
                    self.detect_queue.put((None, None, True), timeout=3.0)
                except Exception:
                    pass

    def _estimate_safe_batch_size(self, H: int, W: int) -> int:
        """[FIX-ESTIMATE-BS] bs=1 OOM 时估算安全 batch_size（吸收 IFRNet
        v6.4.5.1 _estimate_safe_batch_size L6592-6608）。

        综合 OS 层面空闲 VRAM + PyTorch allocator 已 reserved 但未 allocated
        的可复用缓存，给出 70% 保守估算。避免硬编码逐帧回退造成速度惩罚。
        """
        if not torch.cuda.is_available():
            return 1
        try:
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            cached_free = (torch.cuda.memory_reserved(self.device)
                           - torch.cuda.memory_allocated(self.device))
            effective_free = free_bytes + cached_free
            # SR 输入+输出+中间特征 ≈ H*W*3 byte * 2(fp16) * 6(tensors)
            bytes_per_frame = H * W * 3 * 2 * 6
            estimated = max(1, int(effective_free * 0.7 / bytes_per_frame))
            return min(estimated, self.optimal_batch_size)
        except Exception:
            return 1

    def _clamp_adaptive_bs(self, new_bs: int) -> int:
        """[FIX-TRT-ALIGN-READ-BS] 自适应读批收敛约束。

        TRT 激活时 engine B 编译时固定，小于 engine B 的批次会被
        tensorrt_accel.infer 内部 padding 到 engine B（pad 帧纯浪费算力），
        缩小 read_bs 不省 SR 算力、反而降低 GFPGAN 侧批效率 → 锁定 engine B。
        非 TRT 路径维持原收敛逻辑：不超过 [min_adaptive, optimal] 区间。
        """
        new_bs = min(new_bs, max(self.optimal_batch_size, self._min_adaptive_batch))
        if self.trt_accel is not None and self.trt_accel.available:
            return self.trt_accel.input_shape[0]
        return new_bs

    def _process_sr(self):
        """SR推理处理"""
        _first_batch_done = False
        _sentinel_sent = False
        _prefetched_item = None
        _prefetched_tensor = None
        # [FIX-DEFER-RESOLVE] 延迟解析要求输出环深 ≥ 在飞句柄数+1（此处 4），
        # 保证 slot 被 GPU 复用覆写前上一占用者的 CPU 拷贝已完成
        _prefetch_pool = _get_pinned_pool(4)
        _resolve_futs = []   # 在飞解析 future（按提交顺序，最多 2 个未完成）
        _batch_cnt = 0  # 周期清理计数器
        _use_half = self.upsampler.half

        try:
            while self.running:
                try:
                    _pre_gpu_t = None
                    if _prefetched_item is not None:
                        item = _prefetched_item
                        _pre_gpu_t = _prefetched_tensor
                        _prefetched_item = None
                        _prefetched_tensor = None
                    else:
                        item = self.detect_queue.get(timeout=1.0)

                    if item is None:
                        # [FIX-DEFER-RESOLVE] 转发哨兵前按序冲刷全部在飞解析，
                        # 保证末批帧先于哨兵入队（帧序守恒）
                        self._drain_resolve_futs(_resolve_futs)
                        self.sr_queue.put(None)
                        _sentinel_sent = True
                        break

                    batch_frames, face_data, is_end = item

                    if batch_frames is None:
                        self._drain_resolve_futs(_resolve_futs)
                        self.sr_queue.put((None, None, None, None, True))
                        _sentinel_sent = True
                        break

                    memory_block = None
                    while self.running and memory_block is None:
                        memory_block = self.memory_pool.acquire()
                        if memory_block is None:
                            time.sleep(0.005)
                    if not self.running:
                        break

                    t0 = time.perf_counter()

                    def _sr_with_oom_fallback(frames, prefetched_batch_t=None):
                        # [FIX-OOM-TRT] TRT engine 的 batch 维度是编译时固定的，
                        # OOM 时不持久化降低 optimal_batch_size，否则后续所有批次
                        # 都用小 batch 去 pad 大 engine，性能塌方（参照 IFRNet
                        # v6.4.5.1 _safe_infer L6627: TRT 启用时抑制 batch_size 变更）。
                        _is_trt = (self.trt_accel is not None
                                   and self.trt_accel.available)
                        _trt_engine_B = (self.trt_accel.input_shape[0]
                                         if _is_trt else None)
                        # [FIX-TRT-WORKING-BS] TRT OOM 后记住成功 bs，
                        # 避免每批都从 optimal_batch_size 重试 → 必然 OOM → 浪费冷却时间
                        if _is_trt and self._trt_working_bs > 0:
                            retry_bs = min(self._trt_working_bs, len(frames))
                        else:
                            retry_bs = min(self.optimal_batch_size, len(frames))
                        _can_use_prefetch = (prefetched_batch_t is not None and
                                             retry_bs >= len(frames) and
                                             prefetched_batch_t.shape[0] == len(frames))
                        _had_real_oom = False
                        _oom_count = 0
                        _max_oom_retries = 4 if _is_trt else 12
                        while True:
                            try:
                                all_sr = []
                                i = 0
                                # [FIX-DEFER-RESOLVE] 单批多子批（OOM 降级后 retry_bs < len）
                                # 时在飞句柄上限 = 2：跨批 slot 复用距离 ≥ 2 次调用，
                                # 保证下次复用前其 future 已被 reap（解析完成），
                                # 防止 pinned slot 复用覆写未拷贝数据
                                _defer_this = self._defer_resolve
                                _ring_keep = min(2, max(1, getattr(_prefetch_pool, '_num_slots', 2) - 1))
                                while i < len(frames):
                                    if _defer_this and sum(
                                            1 for _x in all_sr
                                            if isinstance(_x, dict)) >= _ring_keep:
                                        all_sr = _sr_materialize(all_sr)
                                    sub = frames[i:i + retry_bs]
                                    _pt = (prefetched_batch_t
                                           if (_can_use_prefetch and i == 0)
                                           else None)
                                    sub_sr, _, _ = _sr_infer_batch(
                                        self.upsampler, sub, self.args.outscale,
                                        getattr(self.args, 'netscale', 4),
                                        self.transfer_stream, self.sr_stream,
                                        self.trt_accel, self.cuda_graph_accel,
                                        prefetched_batch_t=_pt,
                                        defer_resolve=_defer_this,
                                    )
                                    all_sr.extend(sub_sr)
                                    i += retry_bs

                                # ── 成功：TRT 模式不持久化 batch_size ──
                                # [FIX-RECOVER-THRESHOLD] 原阈值 5 太低（~0.3s 就触发恢复），
                                # 恢复到 engine B 后立即 OOM → 死亡螺旋。提升到 50 批，
                                # 确保显存压力真正缓解后才尝试扩大 batch。
                                _recover_threshold = 50
                                if _had_real_oom:
                                    if _is_trt:
                                        # 记录本次成功的临时 bs，后续批次从它开始
                                        self._trt_working_bs = retry_bs
                                        self._trt_success_streak = 1
                                        print(f'[SR-OOM] TRT OOM 恢复（临时 bs={retry_bs}，'
                                              f'engine B={_trt_engine_B} 不变，稳定={self._trt_success_streak}/{_recover_threshold}）',
                                              flush=True)
                                    elif retry_bs < self.optimal_batch_size:
                                        print(f'[SR-OOM] batch_size 降级至 {retry_bs}，持久生效', flush=True)
                                        self.optimal_batch_size = retry_bs
                                elif _is_trt and self._trt_working_bs > 0:
                                    # 无 OOM 的正常批次（使用中 reduced bs），递增成功计数
                                    self._trt_success_streak += 1
                                    if self._trt_success_streak >= _recover_threshold:
                                        # [FIX-GRADUAL-RECOVER] 渐进恢复：×2 而非直接跳
                                        # engine B，避免跳跃过大导致立即 OOM（参照 IFRNet
                                        # batch_size+1 渐进模式）。
                                        # [FIX-NO-RESET] 到达 engine B 后不再清零 _trt_working_bs，
                                        # 保持 = engine B（非 0），后续批次继续走 tracking 路径。
                                        # 清零导致下一批从 optimal_batch_size 开始 → 立即 OOM → 死亡螺旋。
                                        _next_bs = min(self._trt_working_bs * 2, _trt_engine_B)
                                        if _next_bs > self._trt_working_bs:
                                            # [FIX-MEM-GUARD] 显存压力检查：free < 20% 时跳过恢复，
                                            # 维持当前 bs 并重置计数。避免在显存近满时盲目扩大 batch。
                                            _free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
                                            if _free_bytes < _total_bytes * 0.20:
                                                self._trt_success_streak = 0
                                                # 不打印，避免日志洪水（每 50 批触发一次检查）
                                            else:
                                                _old_bs = self._trt_working_bs
                                                self._trt_working_bs = _next_bs
                                                self._trt_success_streak = 0
                                                self._oom_cooldown = 0
                                                _label = '完全恢复' if _next_bs >= _trt_engine_B else '渐进恢复'
                                                print(f'[SR-OOM] {_label}（bs={_old_bs}'
                                                      f'→{_next_bs}，engine B={_trt_engine_B}）', flush=True)
                                                # [FIX-NO-RESET] 不再清零。_trt_working_bs 保持 = engine_B。
                                                # 下一批走 tracking 路径 (L726-727)，而非从 optimal 开始。

                                # ── TRT padding 模式：碎片整理 ──
                                # [FIX-NO-EMPTY-CACHE] expandable_segments 模式下
                                # empty_cache() 释放整个内存池回 OS，下一次 alloc
                                # 需重新走 cudaMallocAsync → 反而增加碎片+延迟。
                                # 不再每批调用。低频兜底由 _batch_cnt % 100 处理。
                                # 原逻辑（每批 empty_cache）已移除。

                                # ── OOM 冷却递减 ──
                                if self._oom_cooldown > 0:
                                    self._oom_cooldown -= 1
                                return all_sr

                            except RuntimeError as _oom_e:
                                _es = str(_oom_e).lower()
                                if 'out of memory' not in _es:
                                    raise

                                _had_real_oom = True
                                _oom_count += 1
                                self._oom_total_count += 1
                                self._trt_success_streak = 0  # OOM 打破稳定计数
                                _can_use_prefetch = False
                                prefetched_batch_t = None

                                # ── Stage 1: 轻量清理 ──
                                # [FIX-SYNC-BEFORE-CACHE] expandable_segments 下
                                # cudaFreeAsync 是异步的 — 必须先 synchronize 排空异步释放，
                                # 再 empty_cache() 将回收内存返回 OS（参照
                                # expandable-segments-synchronize-after-del.md）。
                                torch.cuda.synchronize(self.device)
                                torch.cuda.empty_cache()

                                # ── OOM 冷却期：防连续震荡 ──
                                if self._oom_cooldown > 0:
                                    if _is_trt:
                                        # [FIX-TRT-NO-SLEEP] TRT 模式不睡眠：
                                        # engine B 固定不变 → 无 batch_size 震荡风险；
                                        # sleep 只会浪费 wall-clock 时间，不释放显存
                                        self._oom_cooldown = min(self._oom_cooldown + 5, 60)
                                        retry_bs = max(1, retry_bs // 2)
                                    else:
                                        _wait_s = min(self._oom_cooldown * 0.15, 3.0)
                                        print(f'[SR-OOM] 冷却中 (cooldown={self._oom_cooldown})，'
                                              f'等待 {_wait_s:.1f}s ...', flush=True)
                                        time.sleep(_wait_s)
                                        torch.cuda.synchronize(self.device)
                                        torch.cuda.empty_cache()
                                        self._oom_cooldown = min(self._oom_cooldown + 5, 60)
                                        retry_bs = max(1, retry_bs // 2)
                                    continue

                                # ── 暂停 GFPGAN 子进程释放显存 ──
                                if self.gfpgan_subprocess is not None:
                                    self.gfpgan_subprocess.pause(duration=5.0)

                                # ── 超过最大重试次数 → 深度清理后逐帧回退 ──
                                if _oom_count > _max_oom_retries:
                                    if _is_trt:
                                        print(f'[SR-OOM] TRT 重试 {_max_oom_retries} 次仍 OOM，'
                                              f'深度清理后逐帧推理（engine B={_trt_engine_B} 不变）...', flush=True)
                                    else:
                                        print(f'[SR-OOM] 重试 {_max_oom_retries} 次仍 OOM，'
                                              f'深度清理后逐帧推理...', flush=True)
                                    self._oom_cooldown = 20
                                    torch.cuda.synchronize(self.device)
                                    torch.cuda.empty_cache()
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    time.sleep(3.0)
                                    # 逐帧推理
                                    all_sr = []
                                    for fi in range(len(frames)):
                                        s, _, _ = _sr_infer_batch(
                                            self.upsampler, [frames[fi]], self.args.outscale,
                                            getattr(self.args, 'netscale', 4),
                                            self.transfer_stream, self.sr_stream,
                                            self.trt_accel, self.cuda_graph_accel,
                                        )
                                        all_sr.extend(s)
                                    if not _is_trt:
                                        self.optimal_batch_size = 1
                                    return all_sr

                                # ── Stage 2: 减半重试 ──
                                if retry_bs > 1:
                                    retry_bs = max(1, retry_bs // 2)
                                    if _is_trt:
                                        # [FIX-TRT-WORKING-BS] 记录本次成功的临时 bs，
                                        # 下一批从它开始，避免每批都从 optimal 重试
                                        self._trt_working_bs = retry_bs
                                        print(f'[SR-OOM] TRT OOM → 临时拆分 bs={retry_bs}'
                                              f'（engine B={_trt_engine_B} 不变），重试...', flush=True)
                                    else:
                                        self._oom_cooldown = 8
                                        print(f'[SR-OOM] OOM，降级 batch_size → {retry_bs}，重试...', flush=True)
                                else:
                                    # bs=1 仍 OOM：深度清理 + 估算安全 bs 重试
                                    self._oom_cooldown = 15
                                    torch.cuda.synchronize(self.device)
                                    torch.cuda.empty_cache()
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    # [STREAM-DUAL] OOM 后重建 sr_stream（参照 IFRNet L6646-6651）
                                    if self.sr_stream is not None:
                                        try:
                                            torch.cuda.synchronize(self.device)
                                        except Exception:
                                            pass
                                        self.sr_stream = torch.cuda.Stream(device=self.device)
                                    time.sleep(3.0)
                                    if self.gfpgan_subprocess is not None:
                                        self.gfpgan_subprocess.pause(duration=5.0)
                                    # [FIX-ESTIMATE-BS] 深度清理后用 IFRNet 方法估算
                                    # 安全 batch_size，避免硬编码逐帧回退的速度惩罚
                                    _h = frames[0].shape[0] if frames else 540
                                    _w = frames[0].shape[1] if frames else 960
                                    _safe_bs = self._estimate_safe_batch_size(_h, _w)
                                    if _safe_bs > 1:
                                        if _is_trt:
                                            self._trt_working_bs = _safe_bs
                                            print(f'[SR-OOM] bs=1 OOM 深度清理后估算'
                                                  f' safe_bs={_safe_bs}，以临时 bs={_safe_bs} 重试...', flush=True)
                                        else:
                                            self.optimal_batch_size = _safe_bs
                                            print(f'[SR-OOM] bs=1 OOM 深度清理后估算'
                                                  f' safe_bs={_safe_bs}，重试...', flush=True)
                                        retry_bs = _safe_bs
                                        continue
                                    # 估算 ≤1：逐帧推理兜底
                                    print('[SR-OOM] 估算 safe_bs≤1，逐帧推理兜底...', flush=True)
                                    all_sr = []
                                    for fi in range(len(frames)):
                                        s, _, _ = _sr_infer_batch(
                                            self.upsampler, [frames[fi]], self.args.outscale,
                                            getattr(self.args, 'netscale', 4),
                                            self.transfer_stream, self.sr_stream,
                                            self.trt_accel, self.cuda_graph_accel,
                                        )
                                        all_sr.extend(s)
                                    if _is_trt:
                                        self._trt_working_bs = 1
                                        print(f'[SR-OOM] 逐帧推理完成，TRT engine B={_trt_engine_B} 保持不变', flush=True)
                                    else:
                                        self.optimal_batch_size = 1
                                    return all_sr

                    try:
                        sr_results = _sr_with_oom_fallback(batch_frames, _pre_gpu_t)

                        timing = time.perf_counter() - t0
                        self.timing.append(timing)

                        if not _first_batch_done and self.gfpgan_subprocess is not None:
                            _first_batch_done = True
                            self._vlog('[优化架构] 第一个 SR 批次完成，触发 GFPGAN TRT post-SR 验证...', flush=True)
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                            if self._async_dispatcher is not None:
                                _val_id = id(self)
                                self._async_dispatcher.submit_validate(_val_id)
                                _val_ok = self._async_dispatcher.wait_validate(
                                    _val_id, timeout=180.0)
                            else:
                                _val_ok = self.gfpgan_subprocess.post_sr_validate()

                            if _val_ok:
                                print('[优化架构] GFPGAN TRT post-SR 验证通过，TRT 推理正式启用', flush=True)
                            else:
                                self.gfpgan_subprocess.process.join(timeout=1.5)
                                if not self.gfpgan_subprocess.process.is_alive():
                                    print('[优化架构] GFPGAN 子进程因 CUDA context 损坏已退出，'
                                          '降级到主进程内 GFPGAN PyTorch 路径', flush=True)
                                    self.gfpgan_subprocess = None
                                    self._async_dispatcher = None
                                else:
                                    print('[优化架构] GFPGAN 子进程以 PyTorch FP16 路径服务'
                                          '（TRT 未启用：SM不兼容 / build失败 / OOM）', flush=True)

                        # 预取下一批
                        if (not is_end and _prefetched_item is None and
                                self.detect_queue.qsize() > 0):
                            try:
                                _peek_item = self.detect_queue.get_nowait()
                                if _peek_item is not None:
                                    _pk_frames, _pk_face_data, _pk_is_end = _peek_item
                                    if _pk_frames is not None:
                                        # [FIX-PINNED-POOL-RACE] 原来这里用
                                        # self.transfer_stream.synchronize() 笨重地整条
                                        # stream 同步来规避 pinned buffer 复用竞态，代价是
                                        # 抵消了预取本该带来的重叠收益。现在 PinnedBufferPool
                                        # 内部按 slot 用 event 精确追踪，get_for_frames() 会
                                        # 自动等待"这个 slot 上次的拷贝"完成，不需要在外面
                                        # 再做一次全量同步。
                                        with torch.cuda.stream(self.transfer_stream):
                                            _pk_pin = _prefetch_pool.get_for_frames(_pk_frames)
                                            _pk_gpu = _pk_pin.to(self.device, non_blocking=True)
                                            _pk_gpu = _pk_gpu.permute(0, 3, 1, 2).float().div_(255.0)
                                            if _use_half:
                                                _pk_gpu = _pk_gpu.half()
                                        _prefetch_pool.mark_issued(self.transfer_stream)
                                        _prefetched_item = _peek_item
                                        _prefetched_tensor = _pk_gpu
                                    else:
                                        self.detect_queue.put(_peek_item)
                                else:
                                    self.detect_queue.put(_peek_item)
                            except queue.Empty:
                                pass

                        if self._defer_resolve:
                            # [FIX-DEFER-RESOLVE] 解析+入队交给解析线程，
                            # 与下一批 TRT 推理重叠；限深 2 个未完成 future
                            # （下一迭代启动新推理前会 reap 最旧 future，
                            # 保证 pinned slot 复用时上一占用者已拷贝完成）
                            _resolve_futs.append(self._resolve_executor.submit(
                                self._resolve_and_put,
                                sr_results, batch_frames, face_data,
                                memory_block, is_end))
                            while len(_resolve_futs) > 2:
                                try:
                                    _resolve_futs.pop(0).result()
                                except Exception as _rp_e:
                                    # 旧批次解析失败不级联：当前批 future 已提交，
                                    # 其 memory_block 由 GFPGAN 线程消费时释放
                                    print(f'[SR] 解析任务 reap 异常: {_rp_e}', flush=True)
                        else:
                            while self.running:
                                try:
                                    self.sr_queue.put((batch_frames, face_data, memory_block, sr_results, is_end), timeout=1.0)
                                    break
                                except queue.Full:
                                    continue

                        # [FIX-MEM-PERIODIC] 低频深层清理。非延迟解析路径的主清理
                        # 已由 _sr_infer_batch 的 del+synchronize 覆盖；
                        # [FIX-DEFER-RESOLVE] 延迟解析路径下句柄引用在解析线程释放，
                        # 此处的周期性全同步同时承担其 cudaFreeAsync 排空职责。
                        # 频率降至 100 批以降低开销。
                        _batch_cnt += 1
                        if _batch_cnt % 100 == 0:
                            torch.cuda.synchronize(self.device)
                            gc.collect()
                            torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"SR推理错误（不可恢复）: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                        try:
                            self.memory_pool.release(memory_block['index'])
                        except Exception:
                            pass

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"SR处理错误: {e}")
        finally:
            # [FIX-DEFER-RESOLVE] 先按序冲刷全部在飞解析（其中的批次早于
            # 下方预取残留批次），保证 sr_queue 内帧序不乱
            try:
                self._drain_resolve_futs(_resolve_futs)
            except Exception as _dr_e:
                print(f'[SR] finally: 解析冲刷异常: {_dr_e}', flush=True)
            # 处理预取残留
            if _prefetched_item is not None:
                _pk_frames, _pk_face_data, _pk_is_end = _prefetched_item
                if _pk_frames is not None:
                    _pk_count = len(_pk_frames)
                    print(f'[SR] 检测到预取残留帧: {_pk_count} 帧，尝试补处理...',
                          flush=True)
                    try:
                        _pk_sr, _, _ = _sr_infer_batch(
                            self.upsampler, _pk_frames, self.args.outscale,
                            getattr(self.args, 'netscale', 4),
                            self.transfer_stream, self.sr_stream,
                            self.trt_accel, self.cuda_graph_accel,
                            prefetched_batch_t=None,
                        )
                        self.sr_queue.put(
                            (_pk_frames, _pk_face_data, None, _pk_sr, _pk_is_end),
                            timeout=5.0)
                        print(f'[SR] 预取残留帧 SR 推理成功: {_pk_count} 帧已送入 sr_queue',
                              flush=True)
                    except Exception as _pf_e:
                        print(f'[SR] 预取残留帧 SR 推理失败 ({_pf_e})，'
                              f'回退 CPU resize 保帧...', flush=True)
                        try:
                            import cv2 as _cv2_fb
                            _out_h = int(_pk_frames[0].shape[0] * self.args.outscale)
                            _out_w = int(_pk_frames[0].shape[1] * self.args.outscale)
                            _fallback_sr = [
                                _cv2_fb.resize(f, (_out_w, _out_h),
                                               interpolation=_cv2_fb.INTER_LANCZOS4)
                                for f in _pk_frames
                            ]
                            self.sr_queue.put(
                                (_pk_frames, _pk_face_data, None,
                                 _fallback_sr, _pk_is_end),
                                timeout=5.0)
                            print(f'[SR] 预取残留帧已用 CPU resize 替代: '
                                  f'{_pk_count} 帧（质量降级但不丢帧）', flush=True)
                        except Exception as _fb_e:
                            print(f'[SR] 预取残留帧彻底丢失: {_pk_count} 帧 '
                                  f'({_fb_e})', flush=True)

                _prefetched_item = None
                if _prefetched_tensor is not None:
                    del _prefetched_tensor
                    _prefetched_tensor = None

            if not _sentinel_sent:
                try:
                    self.sr_queue.put((None, None, None, None, True), timeout=3.0)
                except Exception:
                    pass

    def _resolve_and_put(self, all_sr, batch_frames, face_data, memory_block, is_end):
        """[FIX-DEFER-RESOLVE] 解析线程入口：物化延迟句柄 → 保序入队 sr_queue。

        单线程执行器保证各批次的入队顺序与提交顺序一致。
        解析失败时回退 CPU resize 保帧（质量降级但不丢帧，守住帧数守恒）。
        """
        try:
            results = _sr_materialize(all_sr)
        except Exception as _m_e:
            print(f'[SR] 延迟解析失败: {_m_e}，回退 CPU resize 保帧', flush=True)
            import cv2 as _cv2_fb
            _out_h = int(batch_frames[0].shape[0] * self.args.outscale)
            _out_w = int(batch_frames[0].shape[1] * self.args.outscale)
            results = [_cv2_fb.resize(f, (_out_w, _out_h),
                                      interpolation=_cv2_fb.INTER_LANCZOS4)
                       for f in batch_frames]
        for _ in range(300):   # 最多 300s，覆盖下游长时间背压
            try:
                self.sr_queue.put(
                    (batch_frames, face_data, memory_block, results, is_end),
                    timeout=1.0)
                return
            except queue.Full:
                if not self.running:
                    break
        print('[SR] 延迟解析入队失败（pipeline 已停止），批次可能丢失', flush=True)

    def _drain_resolve_futs(self, futs):
        """[FIX-DEFER-RESOLVE] 按提交顺序等待全部在飞解析完成（含入队）。
        单个 future 异常不阻断后续冲刷（批次丢失会大声记录）。"""
        while futs:
            _f = futs.pop(0)
            try:
                _f.result()
            except Exception as _fe:
                print(f'[SR] 解析任务异常: {_fe}', flush=True)

    def _process_gfpgan(self):
        """GFPGAN处理 - 优化2(提前释放) + 优化5B(异步派发)"""
        _sentinel_sent = False
        _current_sr_item = None

        _shm: Optional[SharedMemoryDoubleBuffer] = (
            getattr(self.gfpgan_subprocess, 'shm_buf', None)
            if self.gfpgan_subprocess is not None else None)

        def _release_slot(slot):
            if slot is not None and _shm is not None:
                _shm.release_slot(slot)

        def _pop_and_output_head():
            if not _pending_tasks:
                return
            _h_tid, _h_fd, _h_sr, _h_slot, _h_is_end = _pending_tasks.pop(0)
            try:
                try:
                    if _h_tid is None:
                        _h_final = _h_sr
                    elif self._async_dispatcher is not None:
                        _h_restored = self._async_dispatcher.wait_result(
                            _h_tid, timeout=120.0, slot=_h_slot)
                        _h_final = self._assemble_result(
                            _h_fd, _h_restored, _h_sr)
                    else:
                        _h_final = _h_sr
                except Exception as _he:
                    print(f'[GFPGAN] _pop_and_output_head 等待结果失败: {_he}',
                          flush=True)
                    _h_final = _h_sr
                _put_ok = False
                while self.running:
                    try:
                        self.gfpgan_queue.put(
                            (_h_final, None, _h_is_end), timeout=1.0)
                        _put_ok = True
                        break
                    except queue.Full:
                        continue
                if not _put_ok:
                    try:
                        self.gfpgan_queue.put(
                            (_h_final, None, _h_is_end), timeout=5.0)
                    except Exception:
                        pass
            finally:
                _release_slot(_h_slot)

        def _drain_all_pending():
            while _pending_tasks:
                _pop_and_output_head()

        _pending_tasks: List[Tuple] = []
        _MAX_IN_FLIGHT = 2

        try:
            while self.running:
                # 阶段A: 按序输出已就绪的 pending 任务
                while _pending_tasks:
                    _oldest = _pending_tasks[0]
                    _tid, _fd, _sr, _slot, _is_end = _oldest

                    if _tid is None:
                        _pending_tasks.pop(0)
                        _release_slot(_slot)
                        while self.running:
                            try:
                                self.gfpgan_queue.put(
                                    (_sr, None, _is_end), timeout=1.0)
                                break
                            except queue.Full:
                                continue
                        continue

                    if self._async_dispatcher is None:
                        break

                    with self._async_dispatcher._lock:
                        if _tid not in self._async_dispatcher._results:
                            break

                    _pending_tasks.pop(0)
                    try:
                        all_restored = self._async_dispatcher.wait_result(
                            _tid, timeout=0.1, slot=_slot)
                        final_frames = self._assemble_result(
                            _fd, all_restored, _sr)
                    except Exception as _phase_a_err:
                        print(f'[GFPGAN] Phase A 结果获取异常: {_phase_a_err}，'
                              f'降级发送 SR 结果', flush=True)
                        final_frames = _sr
                    finally:
                        _release_slot(_slot)

                    while self.running:
                        try:
                            self.gfpgan_queue.put(
                                (final_frames, None, _is_end), timeout=1.0)
                            break
                        except queue.Full:
                            continue

                # 阶段B: 从 sr_queue 取新的 SR 结果
                if len(_pending_tasks) >= _MAX_IN_FLIGHT:
                    _pop_and_output_head()

                _has_async_pending = any(
                    t[0] is not None for t in _pending_tasks)
                _sr_timeout = 0.05 if _has_async_pending else 0.5

                try:
                    item = self.sr_queue.get(timeout=_sr_timeout)
                except queue.Empty:
                    continue

                _current_sr_item = item

                if item is None:
                    _drain_all_pending()
                    self.gfpgan_queue.put(None)
                    _sentinel_sent = True
                    _current_sr_item = None
                    break

                batch_frames, face_data, memory_block, sr_results, is_end = item

                if memory_block is not None:
                    try:
                        self.memory_pool.release(memory_block['index'])
                    except Exception:
                        pass

                if batch_frames is None:
                    _drain_all_pending()
                    self.gfpgan_queue.put((None, None, True))
                    _sentinel_sent = True
                    _current_sr_item = None
                    break

                has_valid_faces = (face_data is not None and
                                   len(face_data) > 0 and
                                   any(fd.get('crops') for fd in face_data if fd))

                _gfpgan_sub_alive = (self.gfpgan_subprocess is not None
                                     and self.gfpgan_subprocess.process.is_alive())
                _gfpgan_main_ok = (self.face_enhancer is not None
                                   and getattr(self.face_enhancer, 'gfpgan', None) is not None)

                _n_faces = (sum(len(fd.get('crops', [])) for fd in face_data)
                            if face_data else 0)

                if has_valid_faces and (_gfpgan_sub_alive or _gfpgan_main_ok):
                    all_crops = []
                    crops_per_frame = []
                    for fd in face_data:
                        crops = fd.get('crops', [])
                        crops_per_frame.append(len(crops))
                        all_crops.extend(crops)

                    num_frames = len(face_data) if face_data else 0
                    _n_faces_this_batch = sum(crops_per_frame)
                    avg_faces = _n_faces_this_batch / num_frames if num_frames > 0 else 0.0

                    if _gfpgan_sub_alive and all_crops:
                        self._vlog(f'[GFPGAN] 使用子进程TRT处理 {_n_faces_this_batch} 个人脸。当前批次共 {num_frames} 帧，平均每帧 {avg_faces:.2f} 个人脸')
                        if self._async_dispatcher is not None:
                            _slot = None
                            _submitted = False
                            task_id = self._next_task_id()

                            if (_shm is not None
                                    and _n_faces <= SharedMemoryDoubleBuffer.MAX_FACES):
                                _slot = _shm.try_acquire_slot()
                                if _slot is None:
                                    while _pending_tasks and _slot is None:
                                        _pop_and_output_head()
                                        _slot = _shm.try_acquire_slot()
                                if _slot is None:
                                    try:
                                        _slot = _shm.acquire_slot(timeout=30.0)
                                    except TimeoutError as _te:
                                        print(f'[GFPGAN] slot 获取超时: {_te}，'
                                              f'回退 pickle 路径', flush=True)
                                        _slot = None

                                if _slot is not None:
                                    try:
                                        _shm.write_input(_slot, all_crops)
                                        self.gfpgan_subprocess.task_queue.put(
                                            (task_id, _n_faces, _slot),
                                            timeout=10.0)
                                        _submitted = True
                                    except Exception:
                                        _release_slot(_slot)
                                        _slot = None

                            if not _submitted:
                                _slot = None
                                try:
                                    self.gfpgan_subprocess.task_queue.put(
                                        (task_id, all_crops), timeout=10.0)
                                except Exception as _submit_e:
                                    print(f'[GFPGAN] 异步提交完全失败: {_submit_e}，'
                                          f'降级直通', flush=True)
                                    _pending_tasks.append(
                                        (None, None, sr_results, None, is_end))
                                    _current_sr_item = None
                                    if self._enable_adaptive_batch and face_data is not None:
                                        _frames_in_batch = max(1, len(face_data))
                                        _cur_density = _n_faces / _frames_in_batch
                                        with self._adaptive_batch_lock:
                                            self._face_density_ema = (
                                                (1.0 - self._face_density_alpha) * self._face_density_ema +
                                                self._face_density_alpha * _cur_density
                                            )
                                    continue

                            _pending_tasks.append(
                                (task_id, face_data, sr_results, _slot, is_end))
                            _current_sr_item = None

                            if self._enable_adaptive_batch and face_data is not None:
                                _frames_in_batch = max(1, len(face_data))
                                _cur_density = _n_faces / _frames_in_batch
                                with self._adaptive_batch_lock:
                                    self._face_density_ema = (
                                        (1.0 - self._face_density_alpha) * self._face_density_ema +
                                        self._face_density_alpha * _cur_density
                                    )
                                    _prev_adaptive = self._adaptive_read_batch_size
                                    if self._face_density_ema < self._low_face_threshold:
                                        _new_bs = self._max_adaptive_batch
                                    elif self._face_density_ema > self._high_face_threshold:
                                        _new_bs = self._min_adaptive_batch
                                    else:
                                        _new_bs = self._base_batch_size
                                    # [FIX-TRT-ALIGN-READ-BS] 收敛约束集中到辅助方法（TRT 锁定 engine B）
                                    self._adaptive_read_batch_size = self._clamp_adaptive_bs(_new_bs)
                            continue
                        else:
                            all_restored = self.gfpgan_subprocess.infer(all_crops)
                            restored_by_frame = self._split_restored(all_restored, crops_per_frame, face_data)
                    elif _gfpgan_main_ok and all_crops:
                        from face_utils import _gfpgan_infer_batch
                        restored_by_frame, _ = _gfpgan_infer_batch(
                            face_data, self.face_enhancer, self.device,
                            None, self.args.gfpgan_weight,
                            getattr(self.args, 'gfpgan_batch_size', 8), None, None)
                        all_restored = None
                        self._vlog(f'[GFPGAN] 使用主进程PyTorch处理 {_n_faces_this_batch} 个人脸。当前批次共 {num_frames} 帧，平均每帧 {avg_faces:.2f} 个人脸')
                    else:
                        restored_by_frame = [[] for _ in face_data]
                        all_restored = None
                        print(f'[GFPGAN] GFPGAN不可用，跳过人脸增强')

                    final_frames = self._assemble_result(
                        face_data, restored_by_frame, sr_results)
                else:
                    if _n_faces > 0:
                        print(f'[GFPGAN] GFPGAN不可用，{_n_faces} 个人脸未处理')
                    final_frames = sr_results

                if self._enable_adaptive_batch and face_data is not None:
                    _frames_in_batch = max(1, len(face_data))
                    _cur_density = _n_faces / _frames_in_batch
                    with self._adaptive_batch_lock:
                        self._face_density_ema = (
                            (1.0 - self._face_density_alpha) * self._face_density_ema +
                            self._face_density_alpha * _cur_density
                        )
                        if self._face_density_ema < self._low_face_threshold:
                            _new_bs = self._max_adaptive_batch
                        elif self._face_density_ema > self._high_face_threshold:
                            _new_bs = self._min_adaptive_batch
                        else:
                            _new_bs = self._base_batch_size
                        # [FIX-TRT-ALIGN-READ-BS] 收敛约束集中到辅助方法（TRT 锁定 engine B）
                        self._adaptive_read_batch_size = self._clamp_adaptive_bs(_new_bs)

                _pending_tasks.append((None, None, final_frames, None, is_end))
                _current_sr_item = None

        finally:
            # 处理残留项
            if _current_sr_item is not None:
                try:
                    _ci_batch, _, _ci_mem, _ci_sr, _ci_is_end = _current_sr_item
                    if _ci_mem is not None:
                        try:
                            self.memory_pool.release(_ci_mem['index'])
                        except Exception:
                            pass
                    if _ci_batch is not None and _ci_sr is not None:
                        _ci_count = len(_ci_sr) if isinstance(_ci_sr, list) else 0
                        print(f'[GFPGAN] finally: 发现未转发的 SR 项 '
                              f'({_ci_count} 帧)，降级发送（跳过人脸增强）',
                              flush=True)
                        self.gfpgan_queue.put(
                            (_ci_sr, None, _ci_is_end), timeout=5.0)
                    elif _ci_batch is None:
                        print(f'[GFPGAN] finally: 发现未转发的哨兵项，补发',
                              flush=True)
                        self.gfpgan_queue.put((None, None, True), timeout=5.0)
                        _sentinel_sent = True
                except Exception as _ci_e:
                    print(f'[GFPGAN] finally: 处理残留 SR 项失败: {_ci_e}',
                          flush=True)
                _current_sr_item = None

            for _tid, _fd, _sr, _slot, _is_end in _pending_tasks:
                try:
                    if _tid is None:
                        self.gfpgan_queue.put((_sr, None, _is_end), timeout=5.0)
                    elif self._async_dispatcher is not None:
                        all_restored = self._async_dispatcher.wait_result(
                            _tid, timeout=30.0, slot=_slot)
                        final = self._assemble_result(_fd, all_restored, _sr)
                        self.gfpgan_queue.put((final, None, _is_end), timeout=5.0)
                    else:
                        self.gfpgan_queue.put((_sr, None, _is_end), timeout=5.0)
                except Exception:
                    try:
                        self.gfpgan_queue.put((_sr, None, _is_end), timeout=5.0)
                    except Exception:
                        pass
                finally:
                    _release_slot(_slot)

            _pending_tasks.clear()

            if not _sentinel_sent:
                try:
                    self.gfpgan_queue.put((None, None, True), timeout=5.0)
                except Exception:
                    pass

    @staticmethod
    def _split_restored(all_restored, crops_per_frame, face_data):
        restored_by_frame = []
        idx = 0
        for count in crops_per_frame:
            if all_restored is not None and count > 0:
                restored_by_frame.append(all_restored[idx:idx + count])
            else:
                restored_by_frame.append([])
            idx += count
        return restored_by_frame

    def _assemble_result(self, face_data, restored_or_list, sr_results):
        if restored_or_list is None or not face_data:
            return sr_results

        if (isinstance(restored_or_list, list) and
                len(restored_or_list) > 0 and
                not isinstance(restored_or_list[0], list)):
            crops_per_frame = [len(fd.get('crops', [])) for fd in face_data]
            restored_by_frame = self._split_restored(
                restored_or_list, crops_per_frame, face_data)
        else:
            restored_by_frame = restored_or_list

        if not restored_by_frame or all(
                r is None or len(r) == 0 for r in restored_by_frame):
            return sr_results

        try:
            future = self.paste_executor.submit(
                _paste_faces_batch, face_data, restored_by_frame,
                sr_results, self.face_enhancer)
            return future.result(timeout=60)
        except Exception:
            return sr_results

    def _next_task_id(self) -> int:
        with self._task_id_lock:
            self._task_id_counter += 1
            return self._task_id_counter

    def close(self):
        self._vlog("\n[Pipeline] 正在停止流水线...", flush=True)
        self.running = False

        if self._async_dispatcher is not None:
            self._async_dispatcher.close()
            self._async_dispatcher = None

        # [FIX-DEFER-RESOLVE] 关闭解析执行器（正常流程 futures 已在哨兵处冲刷，
        # 此处不等待以避免异常关闭路径挂死）
        if self._resolve_executor is not None:
            self._resolve_executor.shutdown(wait=False)
            self._resolve_executor = None

        for q_name, q in [('frame', self.frame_queue), ('detect', self.detect_queue),
                          ('sr', self.sr_queue), ('gfpgan', self.gfpgan_queue)]:
            try:
                q.put(None, timeout=1.0)
            except queue.Full:
                pass
            except Exception:
                pass
        self._vlog("\n[Pipeline] 已发送停止信号到所有队列", flush=True)

        if self.gfpgan_subprocess is not None:
            if self.gfpgan_subprocess.process.is_alive():
                # [FIX-GFPGAN-KEEPALIVE] 子进程跨段保活：存活时回注到 args 供下一段复用，
                # 避免每段重新 spawn + TRT warmup（典型几十秒/段）。
                # 下一段 DeepPipelineOptimizer.__init__ 的预启动消费路径
                # （is_alive() 检查 + 死亡回退）天然接住回注的存活子进程。
                self.args._early_gfpgan_subprocess = self.gfpgan_subprocess
                self.gfpgan_subprocess = None  # 断开本 pipeline 引用，防止重复处置
                print("[FIX-GFPGAN-KEEPALIVE] GFPGAN 子进程保活，注入 args._early_gfpgan_subprocess 供下一段复用",
                      flush=True)
            else:
                self._vlog("\n[Pipeline] 正在关闭GFPGAN子进程（已死亡）...", flush=True)
                self.gfpgan_subprocess.close()
                self._vlog("\n[Pipeline] GFPGAN子进程已关闭", flush=True)

        self.detect_executor.shutdown(wait=False)
        self.paste_executor.shutdown(wait=False)

        thread_names = ['_read_thread', '_detect_thread', '_sr_thread', '_gfpgan_thread']
        for name in thread_names:
            thread = getattr(self, name, None)
            if thread and thread.is_alive():
                self._vlog(f"\n[Pipeline] 等待线程 {name} 结束...", flush=True)
                thread.join(timeout=5.0)
                if thread.is_alive():
                    print(f"\n[Pipeline] 线程 {name} 未响应，已放弃等待", flush=True)
                    if not thread.is_alive():
                        thread.daemon = True
        self._vlog("\n[Pipeline] 所有流水线线程已关闭", flush=True)

    def _write_frames(self, writer, pbar, total_frames):
        """写帧 + 全流水线死锁看门狗（含 prefetch 监控）"""
        written_count = 0
        end_sentinel_count = 0
        received_end_sentinel = False

        # 全流水线空转看门狗
        _idle_since = None
        IDLE_DEADLOCK_TIMEOUT = 120.0  # 120s 全空且无哨兵 → 强制退出并 dump 栈

        try:
            while self.running:
                try:
                    item = self.gfpgan_queue.get(timeout=10.0)

                    if item is None:
                        end_sentinel_count += 1
                        received_end_sentinel = True
                        print(f"\n[Pipeline] 写入线程收到第{end_sentinel_count}个结束哨兵，"
                              f"队列积压: {self._queue_status_str()}", flush=True)
                        continue

                    final_frames, memory_block, is_end = item

                    if final_frames is None:
                        if is_end:
                            end_sentinel_count += 1
                            received_end_sentinel = True
                            self._vlog(f"\n[Pipeline] 写入线程收到结束信号，"
                                  f"队列积压: {self._queue_status_str()}", flush=True)
                            continue
                        continue

                    # 收到有效数据，重置空转计时
                    _idle_since = None

                    # [SDK-NVENC] 检测批量写入接口: 一次性提交整批帧以获得更好的编码吞吐
                    if hasattr(writer, 'write_frame_batch'):
                        try:
                            writer.write_frame_batch(final_frames)
                            written_count += len(final_frames)
                        except Exception:
                            # 批量写入失败时回退到逐帧写入
                            for frame in final_frames:
                                if getattr(writer, '_broken', False):
                                    break
                                writer.write_frame(frame)
                                written_count += 1
                    else:
                        for frame in final_frames:
                            if getattr(writer, '_broken', False):
                                print("\n[致命错误] FFmpeg 后台写入进程已崩溃!", flush=True)
                                self.running = False
                                break
                            writer.write_frame(frame)
                            written_count += 1

                    if getattr(writer, '_broken', False):
                        break

                    pbar.update(len(final_frames))
                    self.meter.update(len(final_frames))

                    current_fps = self.meter.fps()
                    eta = self.meter.eta(total_frames)
                    avg_ms = np.mean(self.timing[-10:]) * 1000 if self.timing else 0

                    # 读取 reader / prefetch 水位
                    p_size, p_cap, reader_alive, reader_eof, _produced = (
                        self._get_reader_state())
                    reader_state = ('alive' if reader_alive
                                    else ('eof' if reader_eof else 'DEAD'))

                    # 注意：下面注释代码保留不要删除
                    # postfix = {
                    #     'fps': f'{current_fps:.1f}',
                    #     'eta': f'{eta:.0f}s',
                    #     'bs': self.optimal_batch_size,
                    #     'ms': f'{avg_ms:.0f}',
                    # }
                    # if not self._quiet:
                    #     postfix['queue_sizes'] = (
                    #         f"P:{p_size}/{p_cap}[{reader_state}]/"
                    #         f"F:{self.frame_queue.qsize()}/"
                    #         f"D:{self.detect_queue.qsize()}/"
                    #         f"S:{self.sr_queue.qsize()}/"           # 注意修正为 self.sr_queue
                    #         f"G:{self.gfpgan_queue.qsize()}"
                    #     )
                    # pbar.set_postfix(postfix)

                    pbar.set_postfix(
                        fps=f'{current_fps:.1f}',
                        eta=f'{eta:.0f}s',
                        bs=self.optimal_batch_size,
                        ms=f'{avg_ms:.0f}',
                        queue_sizes=(f"P:{p_size}/{p_cap}[{reader_state}]/"
                                     f"F:{self.frame_queue.qsize()}/"
                                     f"D:{self.detect_queue.qsize()}/"
                                     f"S:{self.sr_queue.qsize()}/"
                                     f"{self._g_label}:{self.gfpgan_queue.qsize()}")
                    )

                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024 ** 3
                        reserved = torch.cuda.memory_reserved() / 1024 ** 3
                        # [FIX-THRESHOLD] 原阈值 allocated > 0.9 * reserved 在 T4 等
                        # 小显存 GPU 上因 reserved 很小（~2GB）几乎每帧都触发误报。
                        # 改用物理总显存：> 85% 物理上限才报警，对应实际 OOM 风险。
                        # [FIX-RATELIMIT] 200 帧才报告一次（对齐 IFRNet 不报实时压力的设计），
                        # 避免日志洪水。阈值仍保留作为早期预警。
                        _total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                        if allocated > 0.85 * _total_mem and written_count % 200 < len(final_frames):
                            print(f'\n[资源警告] GPU内存压力过高: '
                                  f'{allocated:.2f}GB / {_total_mem:.1f}GB total '
                                  f'(reserved={reserved:.2f}GB)')

                    if written_count // 20 > (written_count - len(final_frames)) // 20:
                        # monitor_msg = (f"\n[性能监控] 帧{written_count}/{total_frames} | "
                        #                f"fps={current_fps:.1f} | eta={eta:.0f}s | "
                        #                f"bs={self.optimal_batch_size} | ms={avg_ms:.0f} | "
                        #                f"队列 P:{p_size}/{p_cap}[{reader_state}]/"
                        #                f"F:{self.frame_queue.qsize()}/"
                        #                f"D:{self.detect_queue.qsize()}/"
                        #                f"S:{self.sr_queue.qsize()}/"
                        #                f"G:{self.gfpgan_queue.qsize()}")

                        # if self.face_enhancer is not None:
                        #     monitor_msg += (f" | 人脸 {self._face_count_total}张"
                        #                     f"/{self._face_frames_total}帧")
                        #     if self._face_filtered_total > 0:
                        #         monitor_msg += f" | 过滤{self._face_filtered_total}"
                        #     if self._enable_adaptive_batch:
                        #         monitor_msg += f" | 密度EMA={self._face_density_ema:.1f}"
                        #         monitor_msg += f" | 自适应arbs={self._adaptive_read_batch_size}"
                        if self.face_enhancer is not None:
                            monitor_msg = (f"\n[性能监控] 帧{written_count}/{total_frames}")
                            monitor_msg += (f" | 人脸 {self._face_count_total}张"
                                            f"/{self._face_frames_total}帧")
                            if self._face_filtered_total > 0:
                                monitor_msg += f" | 过滤{self._face_filtered_total}"
                            if self._enable_adaptive_batch:
                                monitor_msg += f" | 密度EMA={self._face_density_ema:.1f}"
                                monitor_msg += f" | 自适应arbs={self._adaptive_read_batch_size}"

                            self._vlog(monitor_msg, flush=True)

                    # self._dump_all_queues()

                except queue.Empty:
                    # 正常终止条件
                    if received_end_sentinel and self.gfpgan_queue.qsize() == 0:
                        print(f"\n[Pipeline] 收到哨兵且 gfpgan_queue 已清空，退出。"
                              f"已写入 {written_count}/{total_frames} 帧", flush=True)
                        break
                    if written_count >= total_frames and received_end_sentinel:
                        print(f"\n[Pipeline] 所有帧已写入且收到结束信号，退出。"
                              f"已写入 {written_count}/{total_frames} 帧", flush=True)
                        break
                    if (written_count >= total_frames
                            and self.sr_queue.qsize() == 0
                            and self.gfpgan_queue.qsize() == 0):
                        self._vlog(f"\n[Pipeline] 所有帧已写入且上游队列清空，强制退出",
                              flush=True)
                        break

                    # 全流水线空转死锁看门狗（含 prefetch）
                    p_size, p_cap, reader_alive, reader_eof, _produced = (
                        self._get_reader_state())
                    reader_state = ('alive' if reader_alive
                                    else ('eof' if reader_eof else 'DEAD'))

                    all_q_empty = (
                        (p_size == 0 or p_size == -1) and
                        self.frame_queue.qsize() == 0 and
                        self.detect_queue.qsize() == 0 and
                        self.sr_queue.qsize() == 0 and
                        self.gfpgan_queue.qsize() == 0
                    )
                    if all_q_empty and not received_end_sentinel:
                        if _idle_since is None:
                            _idle_since = time.time()
                            self._vlog(f"\n[Pipeline][看门狗] 检测到全流水线空转 "
                                  f"(P:{p_size}/{p_cap}[{reader_state}])，"
                                  f"开始计时（阈值 {IDLE_DEADLOCK_TIMEOUT:.0f}s）；"
                                  f"已写入 {written_count}/{total_frames}",
                                  flush=True)
                        elif time.time() - _idle_since > IDLE_DEADLOCK_TIMEOUT:
                            print(f"\n[Pipeline][看门狗] ⚠️ 全流水线空转超过 "
                                  f"{IDLE_DEADLOCK_TIMEOUT:.0f}s 无进展，"
                                  f"判定上游死锁。"
                                  f"P:{p_size}/{p_cap}[{reader_state}] "
                                  f"produced={_produced}；"
                                  f"已写入 {written_count}/{total_frames} 帧，"
                                  f"强制退出。", flush=True)
                            # dump 所有线程栈 + 队列
                            print(f"\n[Pipeline][看门狗] 打印所有队列水位：",
                                  flush=True)
                            self._dump_all_queues()
                            print(f"\n[Pipeline][看门狗] 打印所有线程调用栈：",
                                  flush=True)
                            self._dump_all_threads_stack()
                            # 尝试让上游线程自行收尾
                            self.running = False
                            try:
                                self.frame_queue.put((None, True), timeout=1.0)
                            except Exception:
                                pass
                            break
                    else:
                        _idle_since = None
                    continue
                except Exception as e:
                    print(f"写入帧错误: {e}")
                    if 'memory_block' in locals() and memory_block is not None:
                        try:
                            self.memory_pool.release(memory_block['index'])
                        except Exception:
                            pass
        finally:
            self._vlog(f"\n[Pipeline] 写入线程退出，已写入 {written_count}/{total_frames} 帧 "
                  f"(最终队列: {self._queue_status_str()})", flush=True)