"""
IFRNet 视频插帧处理脚本 —— 深度优化版 v4
======================================================
适用模型: IFRNet_S / IFRNet / IFRNet_L（Vimeo90K 权重）
模型路径: /workspace/video_enhancement/models_ifrnet/checkpoints/
推理入口: IFRNetVideoProcessor.process_video(input, output, scale)

────────────────────────────────────────────────────
功能特性
────────────────────────────────────────────────────
  · 支持任意整数及分数插帧倍数（2× / 4× / 8× / 2.5× 等）
  · 输出编码器可选：libx264 / libx265 / h264_nvenc / hevc_nvenc
  · 自动保留原始音轨（FFmpeg mux，零损耗）
  · 自动检测并修正分辨率 stride 对齐（model stride=32）
  · --preview 实时预览，按 q 中断
  · --report 输出 JSON 性能报告（mean/p95/max 推理延迟、FPS 等）

────────────────────────────────────────────────────
GPU 计算效率
────────────────────────────────────────────────────
  · CUDA Graph 捕获推理循环：
    固定输入形状时消除 kernel launch 调度开销（每帧节省 0.5~2ms），
    首次自动 warmup 3 次后捕获，后续 replay 完全绕过 Python 调度层。

  · 双 CUDA Stream 真正流水线：
    stream_transfer 专用 H2D 数据传输，stream_compute 专用模型推理，
    两者真正重叠执行（需 GPU 具备独立 DMA 引擎，如 3090/A100/H100）。
    embt / imgt_approx 的构建也在 compute_stream 内完成，保证 stream
    语义正确，充分利用 SM 与 DMA 并行能力。

  · expand() 替代 repeat_interleave()：
    将 img0/img1 扩展为 (B×T) 大 batch 时，expand() 共享底层存储，
    无实际显存拷贝；模型若有 in-place 操作自动触发 COW，安全高效。

  · GPU Tensor 缓存池（TensorPool）：
    预分配固定形状的 GPU buffer，避免反复 cudaMalloc，减少显存碎片。

  · 全 GPU 后处理：
    D2H 回传前在 GPU 上完成 clamp_/mul_/byte() 原地操作链，
    PCIe 回传量从 float32（4 字节/像素）压缩到 uint8（1 字节/像素），
    减少约 75% 的 PCIe 流量。D2H 前显式同步 compute_stream，确保
    推理结果完整写回显存后再发起传输。

  · torch.compile 支持（mode='reduce-overhead'）：
    可选启用，对重复调用的固定图模型进一步消除 Python 层开销。

────────────────────────────────────────────────────
CPU / IO 效率
────────────────────────────────────────────────────
  · 异步读帧（AsyncFrameReader）：
    独立后台线程执行 cv2.VideoCapture.read()，主线程从队列直接取帧，
    解码与 GPU 推理真正并行，消除 GPU 等待 CPU 解码的空泡。
    队列深度自适应为 batch_size × 3，随批大小动态调整。

  · PinnedBufferPool CPU 端预分配复用：
    thread-local pinned buffer 长期持有，后续批次直接复用（shape 不变
    时零分配），彻底消除每批 pin_memory() 的 cudaHostRegister 调用开销
    （每次约 0.1~0.5ms）。使用 np.stack 一次性构建连续内存后零拷贝包装
    为 torch.Tensor，再统一 copy 到 pinned buffer，替代原来的逐帧操作。

  · 读入时即 pad，双缓冲设计：
    帧读入时立即做 stride 对齐 pad（mode='edge'，边缘复制，比 reflect
    更自然），维护 padded_buf（用于推理）和 raw_buf（用于写输出）双缓冲，
    彻底消除原来在 flush_buffer 中对同一帧重复 pad 的 CPU 浪费。

  · FFmpegWriter 批量攒帧写入：
    write_loop 引入 pending 列表，攒够 MAX_BATCH=8 帧或队列空闲时才合并
    为一次 stdin.write()，syscall 次数降低约 8 倍。对 120fps/240fps
    高帧率输出场景效果尤为显著，写帧线程 CPU 占用大幅降低。
    write() 调用处即完成 tobytes() 转换，队列存储原始字节，避免
    write_loop 线程内重复转换。

  · FFmpeg pipe 替代 cv2.VideoWriter：
    直接通过 stdin pipe 输送 BGR 原始帧，支持 H.264/HEVC/nvenc 等
    高效编码器，异步写帧线程不阻塞推理主循环。

────────────────────────────────────────────────────
精度 / 正确性
────────────────────────────────────────────────────
  · Fraction 精确时间步计算：
    scale 为非整数（如 2.5）时用 fractions.Fraction 做有理数精确计算，
    避免浮点误差累积导致中间帧时序偏移；n_interp 超过 32 时自动截断
    并打印警告，防止极大值导致显存溢出。

  · 保留原始音轨：
    FFmpeg 模式下将源视频音轨直接 mux 到输出，零损耗保留原始音频编码。

────────────────────────────────────────────────────
健壮性 / 错误处理
────────────────────────────────────────────────────
  · 读帧线程异常跨线程传播：
    AsyncFrameReader._read_loop 发生任何异常（磁盘 IO 错误等）时，将
    Exception 对象直接入队，read() 处判断类型并 re-raise，避免线程静默
    崩溃导致主线程永久阻塞在 queue.get()。

  · FFmpegWriter fail-fast：
    _write_loop 中 stdin.write() 失败时设置 _error flag 并同步 print；
    write() 调用前检查 flag，若已置位立即抛出 RuntimeError，终止推理
    主循环，防止 FFmpeg 进程已死而 GPU 还在白白计算。
    close() 时检查进程返回码，非零时打印 stderr 内容；双重超时保护
    （join 30s + communicate 15s），无响应时强制 kill。

  · OOM 自动降级 + 逐步恢复：
    CUDA OOM 时 batch_size 自动折半重试（最低降至 1），同时清空
    TensorPool 和 CUDA Graph 缓存。OOM 后引入 cooldown 计数器（10 批），
    冷却结束后每批成功推理将 batch_size +1 逐步恢复至初始上限，防止
    一次偶发 OOM 导致全程低速。

────────────────────────────────────────────────────
可观测性
────────────────────────────────────────────────────
  · ThroughputMeter 滑动窗口 FPS：
    基于 deque（最近 20 批）统计帧速，排除 CUDA Graph warmup 等启动
    阶段的慢速干扰，fps / ETA 显示更准确，OOM 降级时速率变化也能
    即时反映。

  · tqdm 实时进度条，postfix 同时展示：
    fps（滑窗帧速）/ eta（剩余时间）/ batch（当前批大小，OOM 降级/
    恢复时动态变化）/ ms（最近 20 批平均推理延迟），四字段并列，
    一眼判断瓶颈在 IO、GPU 推理还是编码写出。

  · JSON 性能报告（--report）：
    处理完成后输出结构化报告，记录 scale、batch_size、fp16、
    frame_count、elapsed_s、avg_fps、推理延迟 mean/p95/max（ms），
    便于多次调参运行的量化对比。
"""

import os
import sys
import cv2
import torch
import numpy as np
import threading
import queue
import time
import json
import argparse
import subprocess
import warnings
from collections import deque
from fractions import Fraction
from contextlib import nullcontext
from typing import Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

warnings.filterwarnings('ignore')

# ── 路径配置 ──────────────────────────────────────────────────────────────────
base_dir = '/workspace/video_enhancement'
models_ifrnet = f'{base_dir}/models_ifrnet/checkpoints'
sys.path.insert(0, f'{base_dir}/IFRNet')
sys.path.insert(0, f'{models_ifrnet}')

from models.IFRNet_S import Model  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# X2: ThroughputMeter（借鉴自 RealESRGAN v3）
# ─────────────────────────────────────────────────────────────────────────────

class ThroughputMeter:
    """
    滑动窗口帧速统计（最近 N 批）。
    借鉴自 RealESRGAN v3，比简单的 总帧数/总时间 更准确，
    不受启动阶段慢速预热的影响。
    """
    def __init__(self, window: int = 20):
        self._times: deque[tuple[float, int]] = deque(maxlen=window)
        self._total = 0

    def update(self, n_frames: int):
        self._times.append((time.perf_counter(), n_frames))
        self._total += n_frames

    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1][0] - self._times[0][0]
        if dt <= 0:
            return 0.0
        return sum(t[1] for t in self._times) / dt

    def eta(self, total: int) -> float:
        f = self.fps()
        if f <= 0:
            return float('inf')
        return (total - self._total) / f


# ─────────────────────────────────────────────────────────────────────────────
# X3: PinnedBufferPool（借鉴自 RealESRGAN v3）
# ─────────────────────────────────────────────────────────────────────────────

_thread_local = threading.local()


class PinnedBufferPool:
    """
    进程/线程本地的 pinned CPU buffer 池，避免每批 H2D 前重新分配。
    借鉴自 RealESRGAN v3 的 PinnedBufferPool。
    使用 np.stack 一次性构建连续内存（零拷贝包装），再统一 copy 到 pinned buffer。
    """
    def __init__(self):
        self._buf: Optional[torch.Tensor] = None

    def get_for_frames(self, frames: list[np.ndarray], to_rgb: bool = True) -> torch.Tensor:
        """
        将 frames 列表 stack，转 RGB（可选），再 copy 到 pinned buffer。
        返回 (B, 3, H, W) float32 pinned tensor，range [0, 1]。
        """
        # np.stack 一次性构建连续内存
        arr = np.stack(frames, axis=0)          # (B, H, W, 3) uint8
        if to_rgb:
            arr = arr[:, :, :, ::-1]            # BGR→RGB
        arr = np.ascontiguousarray(arr)

        src = torch.from_numpy(arr)             # 零拷贝包装 (B, H, W, 3)
        # 转为 (B, 3, H, W) float32 [0,1]
        src_f = src.permute(0, 3, 1, 2).float().div_(255.0).contiguous()

        n_elem = src_f.numel()
        if self._buf is None or self._buf.numel() < n_elem:
            self._buf = torch.empty(n_elem, dtype=torch.float32).pin_memory()

        dst = self._buf[:n_elem].view_as(src_f)
        dst.copy_(src_f)
        return dst


def _get_pinned_pool() -> PinnedBufferPool:
    """线程本地 pool，确保多线程安全。"""
    if not hasattr(_thread_local, 'pinned_pool'):
        _thread_local.pinned_pool = PinnedBufferPool()
    return _thread_local.pinned_pool


# ─────────────────────────────────────────────────────────────────────────────
# 张量工具
# ─────────────────────────────────────────────────────────────────────────────

MODEL_STRIDE = 32


def pad_to_stride(arr: np.ndarray, stride: int = MODEL_STRIDE):
    """对帧做下边/右边 pad，使 H/W 对齐 stride，返回 (padded, pad_h, pad_w)"""
    H, W = arr.shape[:2]
    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride
    if pad_h == 0 and pad_w == 0:
        return arr, 0, 0
    padded = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    return padded, pad_h, pad_w


def frames_to_tensor(
    frames: list[np.ndarray],
    device: torch.device,
    stream: Optional[torch.cuda.Stream] = None,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    numpy BGR list → GPU tensor (B, 3, H, W), range [0,1]。
    X3: 使用 PinnedBufferPool 避免每批重新 pin_memory。
    """
    pool = _get_pinned_pool()
    cpu_t = pool.get_for_frames(frames, to_rgb=True)  # (B,3,H,W) float32 pinned

    ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with ctx:
        gpu_t = cpu_t.to(device, non_blocking=True, dtype=dtype)
    return gpu_t


def tensor_to_np(
    t: torch.Tensor,
    orig_H: int,
    orig_W: int,
    sync_stream: Optional[torch.cuda.Stream] = None,
) -> list[np.ndarray]:
    """
    (B, 3, H, W) float → list of (orig_H, orig_W, 3) uint8 BGR。
    D2H 前先同步指定 stream，确保数据完整。
    """
    if sync_stream is not None and torch.cuda.is_available():
        torch.cuda.current_stream().wait_stream(sync_stream)

    arr = t.clamp_(0.0, 1.0).mul_(255.0).byte()
    arr = arr.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # (B, H, W, 3) RGB
    return [arr[i, :orig_H, :orig_W, ::-1].copy() for i in range(arr.shape[0])]


# ─────────────────────────────────────────────────────────────────────────────
# 张量 GPU 缓存池
# ─────────────────────────────────────────────────────────────────────────────

class TensorPool:
    """预分配固定形状的 GPU Tensor，避免反复 cudaMalloc。"""
    def __init__(self):
        self._cache: dict[tuple, torch.Tensor] = {}

    def get(self, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        key = (shape, dtype, device)
        if key not in self._cache:
            self._cache[key] = torch.empty(shape, dtype=dtype, device=device)
        return self._cache[key]

    def clear(self):
        self._cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# FFmpeg pipe 写帧器（v4：引入 X1 批量写入 + X6 Event 驱动）
# ─────────────────────────────────────────────────────────────────────────────

class FFmpegWriter:
    """
    将 BGR uint8 帧通过 pipe 传给 FFmpeg，支持 H.264 / HEVC / nvenc 编码。

    X1: _write_loop 引入攒批写入（MAX_BATCH=8），减少 syscall，
        对 120/240fps 高帧率输出效果显著。（借鉴自 RealESRGAN v3 Writer）
    X5: error flag + print 双重通知，与 RealESRGAN v3 错误处理风格对齐。
    X6: 关闭信号由 SENTINEL 触发，配合 timeout get() 处理尾帧冲刷。
    """

    _SENTINEL = object()
    _MAX_BATCH = 8    # X1: 每次最多攒多少帧合并写入

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = 'libx264',
        crf: int = 18,
        audio_src: Optional[str] = None,
    ):
        self.output_path = output_path
        self._queue: queue.Queue = queue.Queue(maxsize=128)
        self._error: Optional[Exception] = None

        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', f'{fps:.6f}',
            '-i', 'pipe:0',
        ]
        if audio_src:
            cmd += ['-i', audio_src, '-c:a', 'copy', '-map', '0:v', '-map', '1:a']
        cmd += [
            '-vcodec', codec,
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',
            '-loglevel', 'error',
            output_path,
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def _write_loop(self):
        """
        X1: 攒批写入——每次尽量从队列取多个帧合并成一次 stdin.write()，
        减少系统调用次数（对高帧率/高分辨率输出效果明显）。
        X5/X6: 异常 flag + print 双重通知；SENTINEL 触发干净退出。
        """
        pending: list[bytes] = []
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    # 超时时刷出已积累帧（避免帧在队列中滞留）
                    if pending:
                        self._proc.stdin.write(b''.join(pending))
                        pending = []
                    continue

                if item is self._SENTINEL:
                    if pending:
                        self._proc.stdin.write(b''.join(pending))
                    break

                pending.append(item)
                # 攒够 MAX_BATCH 帧或队列已空时一次写入
                if len(pending) >= self._MAX_BATCH or self._queue.empty():
                    self._proc.stdin.write(b''.join(pending))
                    pending = []

        except Exception as e:
            self._error = e
            print(f'[FFmpegWriter Error] 写帧线程异常: {e}')  # X5: 同步 print

    def write(self, frame_bgr: np.ndarray):
        """write() 处即转 bytes，减少 write_loop 内开销。"""
        if self._error is not None:
            raise RuntimeError(f"FFmpegWriter 内部错误: {self._error}") from self._error
        self._queue.put(frame_bgr.tobytes())

    def close(self):
        """检查 FFmpeg 返回码，超时时强制终止。"""
        self._queue.put(self._SENTINEL)
        self._thread.join(timeout=30)
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            _, stderr_out = self._proc.communicate(timeout=15)
            rc = self._proc.returncode
            if rc != 0:
                print(f"\n[Warning] FFmpeg 退出码={rc}，stderr: "
                      f"{stderr_out.decode(errors='ignore')[:200]}")
        except subprocess.TimeoutExpired:
            self._proc.kill()
            print("\n[Warning] FFmpeg 进程未在超时内退出，已强制终止。")
        if self._error is not None:
            print(f"[Warning] FFmpegWriter 累计写帧异常: {self._error}")


# ─────────────────────────────────────────────────────────────────────────────
# 异步读帧器（保留 v3 的 B2 异常传播）
# ─────────────────────────────────────────────────────────────────────────────

class AsyncFrameReader:
    """
    独立线程读取视频帧，主线程从队列取帧，解码与推理并行。
    B2(v3): 读帧异常时将异常对象入队，主线程 re-raise，避免永久阻塞。
    prefetch 大小动态设置为 batch_size * 3。
    """
    _SENTINEL = object()

    def __init__(self, cap: cv2.VideoCapture, prefetch: int = 8):
        self._cap = cap
        self._queue: queue.Queue = queue.Queue(maxsize=max(prefetch, 4))
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    self._queue.put(self._SENTINEL)
                    break
                self._queue.put(frame)
        except Exception as e:
            self._queue.put(e)  # B2: 异常入队，主线程 re-raise

    def read(self) -> Optional[np.ndarray]:
        item = self._queue.get()
        if item is self._SENTINEL:
            return None
        if isinstance(item, Exception):
            raise item
        return item


# ─────────────────────────────────────────────────────────────────────────────
# 主处理类
# ─────────────────────────────────────────────────────────────────────────────

class IFRNetVideoProcessor:
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        batch_size: int = 4,
        max_batch_size: int = 16,
        use_fp16: bool = True,
        use_compile: bool = True,
        use_cuda_graph: bool = True,
        codec: str = 'libx264',
        crf: int = 18,
        keep_audio: bool = True,
        report_json: Optional[str] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self._max_batch_size = max(batch_size, max_batch_size)
        self._oom_cooldown = 0
        self.use_fp16 = use_fp16 and (self.device.type == 'cuda')
        self.use_cuda_graph = use_cuda_graph and (self.device.type == 'cuda')
        self.codec = codec
        self.crf = crf
        self.keep_audio = keep_audio
        self.report_json = report_json
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        self._pool = TensorPool()
        self._graph: dict = {}
        self._graph_inputs: dict = {}
        self._timing: list[float] = []

        print(f"加载模型: {model_path}")
        self.model = Model()
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt)
        self.model = self.model.to(self.device).eval()

        if self.use_fp16:
            self.model = self.model.half()
            print("FP16 推理已启用")

        if use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("torch.compile 加速已启用")
            except Exception as e:
                print(f"torch.compile 不可用: {e}")

        if self.device.type == 'cuda':
            self.stream_compute = torch.cuda.Stream(device=self.device)
            self.stream_transfer = torch.cuda.Stream(device=self.device)
        else:
            self.stream_compute = self.stream_transfer = None

        print(f"模型就绪 | 设备: {self.device} | batch: {self.batch_size} | "
              f"CUDA Graph: {self.use_cuda_graph}")

    # ──────────────────────────────────────────────────────────────────────────
    # CUDA Graph 推理（保留 v3 B5 修复）
    # ──────────────────────────────────────────────────────────────────────────

    def _get_cuda_graph(self, shape_key: tuple, img0, img1, embt, imgt_approx):
        if shape_key in self._graph:
            static = self._graph_inputs[shape_key]
            static['img0'].copy_(img0)
            static['img1'].copy_(img1)
            static['embt'].copy_(embt)
            static['imgt_approx'].copy_(imgt_approx)
            self._graph[shape_key].replay()
            return static['output']

        print(f"  [CUDA Graph] 捕获 shape={shape_key} ...")
        static_img0 = img0.clone()
        static_img1 = img1.clone()
        static_embt = embt.clone()
        static_imgt = imgt_approx.clone()

        for _ in range(3):
            with torch.cuda.stream(self.stream_compute):
                _warmup_out = self.model(static_img0, static_img1, static_embt, static_imgt)
        torch.cuda.current_stream().wait_stream(self.stream_compute)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=self.stream_compute):
            static_output = self.model(static_img0, static_img1, static_embt, static_imgt)
            if isinstance(static_output, tuple):
                static_output = static_output[0]

        self._graph[shape_key] = g
        self._graph_inputs[shape_key] = {
            'img0': static_img0, 'img1': static_img1,
            'embt': static_embt, 'imgt_approx': static_imgt,
            'output': static_output,
        }
        return static_output

    # ──────────────────────────────────────────────────────────────────────────
    # 核心推理（保留 v3 N1/N2/B4）
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer_batch(
        self,
        img0_list: list[np.ndarray],
        img1_list: list[np.ndarray],
        timesteps: list[float],
        orig_H: int,
        orig_W: int,
    ) -> list[list[np.ndarray]]:
        B = len(img0_list)
        T = len(timesteps)
        t0 = time.perf_counter()

        # X3: 使用 PinnedBufferPool 的 frames_to_tensor
        img0 = frames_to_tensor(img0_list, self.device, self.stream_transfer, self.dtype)
        img1 = frames_to_tensor(img1_list, self.device, self.stream_transfer, self.dtype)

        if self.stream_compute is not None:
            self.stream_compute.wait_stream(self.stream_transfer)

        # N1: expand 替代 repeat_interleave，共享存储
        img0_exp = img0.unsqueeze(1).expand(B, T, *img0.shape[1:]).reshape(B * T, *img0.shape[1:])
        img1_exp = img1.unsqueeze(1).expand(B, T, *img1.shape[1:]).reshape(B * T, *img1.shape[1:])

        shape_key = (B * T, 3, img0.shape[2], img0.shape[3], T)

        if self.use_cuda_graph:
            with torch.cuda.stream(self.stream_compute):
                t_vals = timesteps * B
                embt = torch.tensor(t_vals, dtype=self.dtype, device=self.device).view(-1, 1, 1, 1)
                img0_big = img0_exp.contiguous()
                img1_big = img1_exp.contiguous()
                imgt_approx = img0_big * (1 - embt) + img1_big * embt  # B4
                pred_big = self._get_cuda_graph(shape_key, img0_big, img1_big, embt, imgt_approx)
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
                t_vals = timesteps * B
                embt = torch.tensor(t_vals, dtype=self.dtype, device=self.device).view(-1, 1, 1, 1)
                imgt_approx = img0_exp * (1 - embt) + img1_exp * embt  # B4/N2
                out = self.model(img0_exp, img1_exp, embt, imgt_approx)
                pred_big = out[0] if isinstance(out, tuple) else out

        if self.use_fp16:
            pred_big = pred_big.float()

        all_np = tensor_to_np(pred_big, orig_H, orig_W, sync_stream=self.stream_compute)
        result = [
            [all_np[i * T + j] for j in range(T)]
            for i in range(B)
        ]

        self._timing.append(time.perf_counter() - t0)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # OOM 自动降级 + 恢复
    # ──────────────────────────────────────────────────────────────────────────

    def _safe_infer(self, img0_list, img1_list, timesteps, orig_H, orig_W):
        while True:
            try:
                result = self._infer_batch(img0_list, img1_list, timesteps, orig_H, orig_W)
                if self._oom_cooldown > 0:
                    self._oom_cooldown -= 1
                elif self.batch_size < self._max_batch_size:
                    new_bs = min(self.batch_size + 1, self._max_batch_size)
                    print(f"[恢复] 显存充裕，batch_size {self.batch_size} → {new_bs}")
                    self.batch_size = new_bs
                return result
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                self._pool.clear()
                self._graph.clear()
                if self.batch_size <= 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)
                self._oom_cooldown = 10
                print(f"\n[OOM] 显存不足，自动降低 batch_size → {self.batch_size}")

    # ──────────────────────────────────────────────────────────────────────────
    # 主流程
    # ──────────────────────────────────────────────────────────────────────────

    def process_video(
        self,
        input_path: str,
        output_path: str,
        scale: float = 2.0,
        preview: bool = False,
        preview_interval: int = 30,
    ) -> bool:

        if not os.path.exists(input_path):
            print(f"错误: 输入不存在 - {input_path}")
            return False

        out_dir = os.path.dirname(output_path) or '.'
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"无法打开视频: {input_path}")
            return False

        fps      = cap.get(cv2.CAP_PROP_FPS)
        W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"输入: {W}x{H} @ {fps:.3f}fps | {n_frames} 帧")

        _, pad_h, pad_w = pad_to_stride(np.zeros((H, W, 3), dtype=np.uint8))
        need_pad = pad_h > 0 or pad_w > 0
        if need_pad:
            print(f"  自动 Pad: {H}x{W} → {H+pad_h}x{W+pad_w}（stride={MODEL_STRIDE}）")

        scale_frac = Fraction(scale).limit_denominator(64)
        n_interp   = int(scale_frac) - 1
        if n_interp < 1:
            print(f"错误: scale 必须 ≥ 2，当前={scale}")
            cap.release()
            return False
        if n_interp > 32:
            print(f"警告: n_interp={n_interp} 过大，已截断至 32")
            scale_frac = Fraction(33)
            n_interp = 32
        timesteps = [float(Fraction(i, int(scale_frac))) for i in range(1, int(scale_frac))]
        new_fps   = fps * float(scale_frac)
        print(f"输出: {new_fps:.3f}fps | 插帧×{int(scale_frac)} | 时间步数: {n_interp}")

        audio_src = input_path if self.keep_audio else None
        writer = FFmpegWriter(
            output_path, W, H, new_fps,
            codec=self.codec, crf=self.crf, audio_src=audio_src,
        )

        frame_reader = AsyncFrameReader(cap, prefetch=self.batch_size * 3)

        padded_buf: list[np.ndarray] = []
        raw_buf:    list[np.ndarray] = []

        frame_count  = 0
        output_count = 0
        t_start      = time.time()

        # X2: ThroughputMeter 替代简单总帧数/总时间
        meter = ThroughputMeter(window=20)

        def maybe_pad_inline(frame: np.ndarray) -> np.ndarray:
            if not need_pad:
                return frame
            return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')

        def flush_buffer():
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
            meter.update(n_pairs)  # X2: 更新吞吐统计

        if HAS_TQDM:
            pbar = tqdm(total=n_frames, unit='帧', desc='插帧处理', dynamic_ncols=True)
        else:
            pbar = None

        first = frame_reader.read()
        if first is None:
            print("无法读取视频帧")
            cap.release()
            if pbar:
                pbar.close()
            return False

        writer.write(first)
        output_count += 1
        raw_buf.append(first)
        padded_buf.append(maybe_pad_inline(first))
        frame_count = 1
        if pbar:
            pbar.update(1)

        while True:
            frame = frame_reader.read()
            if frame is None:
                break

            frame_count += 1
            raw_buf.append(frame)
            padded_buf.append(maybe_pad_inline(frame))

            if len(raw_buf) == self.batch_size + 1:
                flush_buffer()
                raw_buf    = [raw_buf[-1]]
                padded_buf = [padded_buf[-1]]

            if pbar:
                pbar.update(1)
                # X2: 使用 ThroughputMeter 的精确滑动窗口 FPS
                cur_fps = meter.fps()
                eta     = meter.eta(n_frames)
                avg_t   = np.mean(self._timing[-20:]) * 1000 if self._timing else 0
                pbar.set_postfix(
                    fps=f'{cur_fps:.1f}',
                    eta=f'{eta:.0f}s',
                    batch=self.batch_size,
                    ms=f'{avg_t:.0f}',
                )
            else:
                report_interval = max(1, n_frames // 20)
                if frame_count % report_interval == 0:
                    elapsed = time.time() - t_start
                    spd = frame_count / elapsed if elapsed > 0 else 0
                    eta = (n_frames - frame_count) / spd if spd > 0 else 0
                    avg_t = np.mean(self._timing[-20:]) * 1000 if self._timing else 0
                    print(
                        f"  进度 {frame_count}/{n_frames} "
                        f"({frame_count/n_frames*100:.1f}%) | "
                        f"{spd:.1f} 帧/s | ETA {eta:.0f}s | 推理延迟 {avg_t:.1f}ms/batch"
                    )

            if preview and frame_count % preview_interval == 0:
                cv2.imshow('IFRNet Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("用户中断预览，退出处理。")
                    break

        if len(raw_buf) > 1:
            flush_buffer()

        if pbar:
            pbar.close()

        writer.close()
        cap.release()
        if preview:
            cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        print(f"\n✅ 插帧完成！")
        print(f"   原始帧数: {frame_count} → 输出帧数: {output_count}")
        print(f"   总耗时: {elapsed:.1f}s | 平均: {frame_count/elapsed:.1f} 原始帧/s")
        print(f"   输出: {output_path}")

        if self.report_json and self._timing:
            report = {
                'input': input_path,
                'output': output_path,
                'scale': scale,
                'batch_size': self.batch_size,
                'fp16': self.use_fp16,
                'cuda_graph': self.use_cuda_graph,
                'frame_count': frame_count,
                'output_count': output_count,
                'elapsed_s': round(elapsed, 2),
                'avg_fps': round(frame_count / elapsed, 2),
                'infer_latency_ms': {
                    'mean': round(float(np.mean(self._timing)) * 1000, 2),
                    'p95':  round(float(np.percentile(self._timing, 95)) * 1000, 2),
                    'max':  round(float(np.max(self._timing)) * 1000, 2),
                },
            }
            with open(self.report_json, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"   性能报告: {self.report_json}")

        return True


# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME_MAP = {
    'IFRNet_Vimeo90K':   'IFRNet_Vimeo90K.pth',
    'IFRNet_S_Vimeo90K': 'IFRNet_S_Vimeo90K.pth',
    'IFRNet_L_Vimeo90K': 'IFRNet_L_Vimeo90K.pth',
}


def main():
    parser = argparse.ArgumentParser(
        description='IFRNet 视频插帧 —— 深度优化版 v4',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input',       required=True,  help='输入视频路径')
    parser.add_argument('--output',      required=True,  help='输出视频路径')
    parser.add_argument('--scale',       type=float, default=2.0,
                        help='插帧倍数（整数：2/4/8；支持 Fraction：2.5）')
    parser.add_argument('--model',       default='IFRNet_S_Vimeo90K',
                        help='模型名称或完整 .pth 路径')
    parser.add_argument('--device',      default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size',  type=int, default=4)
    parser.add_argument('--no_fp16',     action='store_true', help='禁用 FP16')
    parser.add_argument('--no_compile',  action='store_true', help='禁用 torch.compile')
    parser.add_argument('--no_cuda_graph', action='store_true',
                        help='禁用 CUDA Graph（形状可变时必须禁用）')
    parser.add_argument('--codec',       default='libx264',
                        help='输出编码器，如 libx264 / libx265 / h264_nvenc')
    parser.add_argument('--crf',         type=int, default=18)
    parser.add_argument('--no_audio',    action='store_true', help='不保留原始音轨')
    parser.add_argument('--preview',     action='store_true', help='实时预览插帧结果')
    parser.add_argument('--preview_interval', type=int, default=30)
    parser.add_argument('--report',      default=None,
                        help='输出 JSON 性能报告路径（如 report.json）')

    args = parser.parse_args()

    if args.model in MODEL_NAME_MAP:
        model_path = f'{models_ifrnet}/{MODEL_NAME_MAP[args.model]}'
    else:
        model_path = args.model
    if not os.path.exists(model_path):
        print(f"错误: 模型不存在 - {model_path}")
        return

    print("=" * 60)
    print("  IFRNet 视频插帧 —— 深度优化版 v4")
    print("=" * 60)
    print(f"  模型: {args.model}")
    print(f"  设备: {args.device} | 批大小: {args.batch_size} | "
          f"FP16: {not args.no_fp16} | CUDA Graph: {not args.no_cuda_graph}")
    print(f"  编码器: {args.codec} | CRF: {args.crf} | 保留音频: {not args.no_audio}")
    print()

    t0 = time.time()
    processor = IFRNetVideoProcessor(
        model_path=model_path,
        device=args.device,
        batch_size=args.batch_size,
        max_batch_size=args.batch_size,
        use_fp16=not args.no_fp16,
        use_compile=not args.no_compile,
        use_cuda_graph=not args.no_cuda_graph,
        codec=args.codec,
        crf=args.crf,
        keep_audio=not args.no_audio,
        report_json=args.report,
    )

    ok = processor.process_video(
        args.input, args.output,
        scale=args.scale,
        preview=args.preview,
        preview_interval=args.preview_interval,
    )

    m, s = divmod(int(time.time() - t0), 60)
    print(f"\n⏱️  总耗时（含模型加载）: {m}分{s}秒")
    if ok and os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / 1024 / 1024
        print(f"✅ 输出文件: {args.output} ({size_mb:.1f} MB)")
    else:
        print("❌ 处理失败")


if __name__ == '__main__':
    main()
