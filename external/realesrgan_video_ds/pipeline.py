#!/usr/bin/env python3
"""
Real-ESRGAN Video Enhancement - 深度流水线模块
包含：GPUMemoryPool, DeepPipelineOptimizer, SR推理函数
"""

import os
import sys
import time
import queue
import threading
import concurrent.futures
from typing import List, Optional, Tuple, Dict, Any

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

    out_u8 = output_t.float().clamp_(0.0, 1.0).mul_(255.0).byte()
    out_perm = out_u8.permute(0, 2, 3, 1).contiguous()
    out_pinned = pool.get_output_buf(out_perm.shape, torch.uint8)
    out_pinned.copy_(out_perm, non_blocking=True)

    torch.cuda.synchronize(device)

    out_np = out_pinned.numpy()
    sr_results = [out_np[i].copy() for i in range(B)]

    elapsed = time.perf_counter() - t0
    timing_info = {'batch_size': B, 'processing_time': elapsed}
    return sr_results, timing_info, 'success'


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

        self.optimal_batch_size = min(args.batch_size, 24)

        self.frame_queue = queue.Queue(maxsize=48)
        self.detect_queue = queue.Queue(maxsize=32)
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
        self._max_adaptive_batch = min(self._base_batch_size * 2, 12)
        self._min_adaptive_batch = max(2, self._base_batch_size // 2)
        self._adaptive_batch_lock = threading.Lock()
        self._adaptive_read_batch_size = self.optimal_batch_size
        self._enable_adaptive_batch = getattr(args, 'adaptive_batch', True)

        self.detect_helper = _make_detect_helper(face_enhancer, device) if face_enhancer else None

        self.gfpgan_subprocess = None
        try:
            _prestarted = getattr(args, '_early_gfpgan_subprocess', None)
        except Exception:
            _prestarted = None

        if _prestarted is not None:
            if hasattr(_prestarted, 'process') and _prestarted.process.is_alive():
                print('[优化架构] 使用预启动 GFPGAN 子进程（FIX-EARLY-SPAWN）')
                self.gfpgan_subprocess = _prestarted
                args._early_gfpgan_subprocess = None
            else:
                print('[优化架构] 预启动 GFPGAN 子进程已死亡，关闭并回退')
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
                print('[优化架构] 启用子进程GFPGAN TRT加速（非预启动路径）')
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

        # 线程句柄
        self._read_thread = None
        self._detect_thread = None
        self._sr_thread = None
        self._gfpgan_thread = None

    def optimize_pipeline(self, reader, writer, pbar, total_frames):
        """运行优化的深度流水线"""
        print("[优化架构] 启动深度流水线处理...")
        print(f"[优化架构] 队列深度: F{self.frame_queue.maxsize}/D{self.detect_queue.maxsize}/S{self.sr_queue.maxsize}/G{self.gfpgan_queue.maxsize}")
        print(f"[优化架构] 内存池: {self.memory_pool.max_batches}批次")
        print(f"[优化架构] 最优batch_size: {self.optimal_batch_size}")
        print(f"[优化架构] 人脸检测置信度阈值: {self.face_det_threshold}")
        if self._enable_adaptive_batch:
            print(f"[优化架构] 自适应批处理: 开启 (范围 {self._min_adaptive_batch}~{self._max_adaptive_batch}, "
                  f"低密度阈值={self._low_face_threshold}, 高密度阈值={self._high_face_threshold})")
        else:
            print(f"[优化架构] 自适应批处理: 关闭")

        if self.gfpgan_subprocess is not None:
            print('[优化架构] 等待 GFPGAN Inference 进程初始化（加载 .trt + warmup）...')
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
                    print(f'[优化架构] 等待中... {elapsed:.0f}s（Inference 进程初始化中）', flush=True)
                    _last_report = now
            if ready:
                print('[优化架构] GFPGAN 子进程已就绪，启动流水线')
                _shm = getattr(self.gfpgan_subprocess, 'shm_buf', None)
                self._async_dispatcher = AsyncGFPGANDispatcher(
                    self.gfpgan_subprocess, shm_buf=_shm)
                print('[优化架构] AsyncGFPGANDispatcher 已创建'
                      f' (shm={"是" if _shm else "否"})')
            else:
                print('[优化架构] GFPGAN 子进程未就绪，回退 PyTorch 路径')
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

        # 终止所有流水线线程
        # print("[Pipeline] _write_frames 已退出，通知所有流水线线程终止...", flush=True)
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
                print(f"[Pipeline] 警告: {name} 线程未在 {_JOIN_TIMEOUT:.0f}s 内退出",
                      flush=True)
            # else:
            #     print(f"[Pipeline] {name} 线程已退出", flush=True)

    def _read_frames(self, reader):
        """读取视频帧到队列"""
        batch_frames = []
        try:
            while self.running:
                try:
                    img = reader.get_frame()
                    if img is reader.FRAME_TIMEOUT:
                        continue
                    if img is None:
                        if batch_frames:
                            while self.running:
                                try:
                                    self.frame_queue.put((batch_frames, True), timeout=1.0)
                                    break
                                except queue.Full:
                                    continue
                        break

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
                    print(f"读取帧错误: {e}")
                    break
        finally:
            try:
                self.frame_queue.put((None, True), timeout=3.0)
            except Exception:
                pass

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
                    print(f"人脸检测错误: {e}")
        finally:
            if not _sentinel_sent:
                try:
                    self.detect_queue.put((None, None, True), timeout=3.0)
                except Exception:
                    pass

    def _process_sr(self):
        """SR推理处理"""
        _first_batch_done = False
        _sentinel_sent = False
        _prefetched_item = None
        _prefetched_tensor = None
        _prefetch_pool = _get_pinned_pool()
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
                        self.sr_queue.put(None)
                        _sentinel_sent = True
                        break

                    batch_frames, face_data, is_end = item

                    if batch_frames is None:
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
                        retry_bs = min(self.optimal_batch_size, len(frames))
                        _can_use_prefetch = (prefetched_batch_t is not None and
                                             retry_bs >= len(frames) and
                                             prefetched_batch_t.shape[0] == len(frames))
                        _had_real_oom = False
                        while True:
                            try:
                                all_sr = []
                                i = 0
                                while i < len(frames):
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
                                    )
                                    all_sr.extend(sub_sr)
                                    i += retry_bs
                                if _had_real_oom and retry_bs < self.optimal_batch_size:
                                    print(f'[SR-OOM] batch_size 降级至 {retry_bs}，持久生效', flush=True)
                                    self.optimal_batch_size = retry_bs
                                return all_sr
                            except RuntimeError as _oom_e:
                                _es = str(_oom_e).lower()
                                if 'out of memory' not in _es:
                                    raise

                                _had_real_oom = True
                                _can_use_prefetch = False
                                prefetched_batch_t = None

                                torch.cuda.empty_cache()
                                if self.gfpgan_subprocess is not None:
                                    self.gfpgan_subprocess.pause(duration=5.0)

                                if retry_bs > 1:
                                    retry_bs = max(1, retry_bs // 2)
                                    print(f'[SR-OOM] OOM，降级 batch_size → {retry_bs}，重试...', flush=True)
                                else:
                                    print('[SR-OOM] bs=1 仍 OOM，等待 2s 后最终尝试...', flush=True)
                                    time.sleep(2.0)
                                    torch.cuda.empty_cache()
                                    if self.gfpgan_subprocess is not None:
                                        self.gfpgan_subprocess.pause(duration=5.0)
                                    sub_sr, _, _ = _sr_infer_batch(
                                        self.upsampler, frames[:1], self.args.outscale,
                                        getattr(self.args, 'netscale', 4),
                                        self.transfer_stream, self.sr_stream,
                                        self.trt_accel, self.cuda_graph_accel,
                                    )
                                    all_sr = sub_sr
                                    for fi in range(1, len(frames)):
                                        s, _, _ = _sr_infer_batch(
                                            self.upsampler, [frames[fi]], self.args.outscale,
                                            getattr(self.args, 'netscale', 4),
                                            self.transfer_stream, self.sr_stream,
                                            self.trt_accel, self.cuda_graph_accel,
                                        )
                                        all_sr.extend(s)
                                    self.optimal_batch_size = 1
                                    return all_sr

                    try:
                        sr_results = _sr_with_oom_fallback(batch_frames, _pre_gpu_t)

                        timing = time.perf_counter() - t0
                        self.timing.append(timing)

                        if not _first_batch_done and self.gfpgan_subprocess is not None:
                            _first_batch_done = True
                            print('[优化架构] 第一个 SR 批次完成，触发 GFPGAN TRT post-SR 验证...', flush=True)
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
                                        self.transfer_stream.synchronize()
                                        with torch.cuda.stream(self.transfer_stream):
                                            _pk_pin = _prefetch_pool.get_for_frames(_pk_frames)
                                            _pk_gpu = _pk_pin.to(self.device, non_blocking=True)
                                            _pk_gpu = _pk_gpu.permute(0, 3, 1, 2).float().div_(255.0)
                                            if _use_half:
                                                _pk_gpu = _pk_gpu.half()
                                        _prefetched_item = _peek_item
                                        _prefetched_tensor = _pk_gpu
                                    else:
                                        self.detect_queue.put(_peek_item)
                                else:
                                    self.detect_queue.put(_peek_item)
                            except queue.Empty:
                                pass

                        while self.running:
                            try:
                                self.sr_queue.put((batch_frames, face_data, memory_block, sr_results, is_end), timeout=1.0)
                                break
                            except queue.Full:
                                continue

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
                        print(f'[GFPGAN] 使用子进程TRT处理 {_n_faces_this_batch} 个人脸。当前批次共 {num_frames} 帧，平均每帧 {avg_faces:.2f} 个人脸')
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
                                    _new_bs = min(_new_bs, max(self.optimal_batch_size, self._min_adaptive_batch))
                                    self._adaptive_read_batch_size = _new_bs
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
                        print(f'[GFPGAN] 使用主进程PyTorch处理 {_n_faces_this_batch} 个人脸。当前批次共 {num_frames} 帧，平均每帧 {avg_faces:.2f} 个人脸')
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
                        _new_bs = min(_new_bs, max(self.optimal_batch_size, self._min_adaptive_batch))
                        self._adaptive_read_batch_size = _new_bs

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
        print("[Pipeline] 正在停止流水线...", flush=True)
        self.running = False

        if self._async_dispatcher is not None:
            self._async_dispatcher.close()
            self._async_dispatcher = None

        for q_name, q in [('frame', self.frame_queue), ('detect', self.detect_queue),
                          ('sr', self.sr_queue), ('gfpgan', self.gfpgan_queue)]:
            try:
                q.put(None, timeout=1.0)
            except queue.Full:
                pass
            except Exception:
                pass
        print("[Pipeline] 已发送停止信号到所有队列", flush=True)

        if self.gfpgan_subprocess:
            print("[Pipeline] 正在关闭GFPGAN子进程...", flush=True)
            self.gfpgan_subprocess.close()
            print("[Pipeline] GFPGAN子进程已关闭", flush=True)

        self.detect_executor.shutdown(wait=False)
        self.paste_executor.shutdown(wait=False)

        thread_names = ['_read_thread', '_detect_thread', '_sr_thread', '_gfpgan_thread']
        for name in thread_names:
            thread = getattr(self, name, None)
            if thread and thread.is_alive():
                print(f"[Pipeline] 等待线程 {name} 结束...", flush=True)
                thread.join(timeout=5.0)
                if thread.is_alive():
                    print(f"[Pipeline] 线程 {name} 未响应，已放弃等待", flush=True)
                    if not thread.is_alive():
                        thread.daemon = True
        print("[Pipeline] 所有流水线线程已关闭", flush=True)

    def _write_frames(self, writer, pbar, total_frames):
        written_count = 0
        end_sentinel_count = 0
        received_end_sentinel = False

        try:
            while self.running:
                try:
                    item = self.gfpgan_queue.get(timeout=10.0)

                    if item is None:
                        end_sentinel_count += 1
                        received_end_sentinel = True
                        print(f"[Pipeline] 写入线程收到第{end_sentinel_count}个结束哨兵，队列积压: S:{self.sr_queue.qsize()}/G:{self.gfpgan_queue.qsize()}", flush=True)
                        continue

                    final_frames, memory_block, is_end = item

                    if final_frames is None:
                        if is_end:
                            end_sentinel_count += 1
                            received_end_sentinel = True
                            print(f"[Pipeline] 写入线程收到结束信号，队列积压: S:{self.sr_queue.qsize()}/G:{self.gfpgan_queue.qsize()}", flush=True)
                            continue
                        continue

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

                    pbar.set_postfix(
                        fps=f'{current_fps:.1f}',
                        eta=f'{eta:.0f}s',
                        bs=self.optimal_batch_size,
                        ms=f'{avg_ms:.0f}',
                        queue_sizes=f"F:{self.frame_queue.qsize()}/D:{self.detect_queue.qsize()}/S:{self.sr_queue.qsize()}/G:{self.gfpgan_queue.qsize()}"
                    )

                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024 ** 3
                        reserved = torch.cuda.memory_reserved() / 1024 ** 3
                        if allocated > 0.9 * reserved:
                            print(f'\n[资源警告] GPU内存压力过高: {allocated:.2f}GB / {reserved:.2f}GB')

                    if written_count // 20 > (written_count - len(final_frames)) // 20:
                        _density_str = f' | 密度EMA={self._face_density_ema:.1f}' if self._enable_adaptive_batch else ''
                        _filtered_str = f' | 过滤{self._face_filtered_total}' if self._face_filtered_total > 0 else ''
                        _adaptive_str = f' | 自适应arbs={self._adaptive_read_batch_size}' if self._enable_adaptive_batch else ''
                        print(f"[性能监控] 帧{written_count}/{total_frames} | fps={current_fps:.1f} | eta={eta:.0f}s | bs={self.optimal_batch_size} | ms={avg_ms:.0f} | 队列 F:{self.frame_queue.qsize()}/D:{self.detect_queue.qsize()}/S:{self.sr_queue.qsize()}/G:{self.gfpgan_queue.qsize()} | 人脸 {self._face_count_total}张/{self._face_frames_total}帧{_filtered_str}{_density_str}{_adaptive_str}")

                except queue.Empty:
                    if received_end_sentinel and self.gfpgan_queue.qsize() == 0:
                        print(f"[Pipeline] 收到哨兵且 gfpgan_queue 已清空，退出。"
                              f"已写入 {written_count}/{total_frames} 帧", flush=True)
                        break
                    if written_count >= total_frames and received_end_sentinel:
                        print(f"[Pipeline] 所有帧已写入且收到结束信号，退出。"
                              f"已写入 {written_count}/{total_frames} 帧", flush=True)
                        break
                    if (written_count >= total_frames
                            and self.sr_queue.qsize() == 0
                            and self.gfpgan_queue.qsize() == 0):
                        print(f"[Pipeline] 所有帧已写入且上游队列清空，强制退出", flush=True)
                        break
                    continue
                except Exception as e:
                    print(f"写入帧错误: {e}")
                    if 'memory_block' in locals() and memory_block is not None:
                        try:
                            self.memory_pool.release(memory_block['index'])
                        except Exception:
                            pass
        finally:
            print(f"[Pipeline] 写入线程退出，已写入 {written_count}/{total_frames} 帧", flush=True)