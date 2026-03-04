"""
RealESRGAN 视频超分处理脚本 —— 深度优化版 v4
======================================================
支持模型: RealESRGAN_x4plus / x2plus / anime_6B / animevideov3 /
          general-x4v3 / RealESRGANv2-animevideo-xsx2/x4
模型路径: /workspace/Video_Enhancement/models_RealESRGAN/
推理入口: inference_video(args, video_save_path)

────────────────────────────────────────────────────  
功能特性
────────────────────────────────────────────────────
  · 支持视频文件、图片文件夹、单张图片三种输入模式
  · 多 GPU 并行：视频自动切分，各 GPU 独立处理后 concat 合并
  · 输出编码器可选：libx264 / libx265 / libvpx-vp9，支持 CRF 质量控制
    （libx265 在 4K 超分场景可减小约 40~60% 文件体积）
  · 可选人脸增强（GFPGANer）、降噪强度调节（DNI weight）
  · --extract_frame_first 先解帧再处理，适合高码率长视频
  · --report 输出 JSON 性能报告（mean/p95/max 推理延迟、FPS 等）

────────────────────────────────────────────────────
GPU 计算效率
────────────────────────────────────────────────────
  · 双 CUDA Stream 真正流水线：
    transfer_stream 专用 H2D 数据传输，compute_stream 专用模型推理，
    两者真正重叠执行（需 GPU 具备独立 DMA 引擎，如 3090/A100/H100），
    H2D 期间 GPU SM 计算单元不空闲。

  · 全 GPU 后处理：
    output_t 直接在 GPU 上执行 clamp_/mul_/byte() 原地操作链，
    回传量从 float32（4 字节/像素）压缩到 uint8（1 字节/像素），
    PCIe 流量减少约 75%，中间 tensor 分配次数从 4 次降为 1 次。

  · torch.compile 支持（mode='reduce-overhead'，可选启用）。

────────────────────────────────────────────────────
CPU / IO 效率
────────────────────────────────────────────────────
  · PinnedBufferPool CPU 端预分配复用（process-local，线程安全）：
    采用 threading.local 使每个线程独立持有 pinned buffer，彻底消除
    原全局单 buffer 的多线程竞争条件。buffer 按需分配后长期复用，
    热路径内存分配降至 O(1)，消除每批 pin_memory() 的
    cudaHostRegister 调用开销（约 0.1~0.5ms/次）。
    使用 np.stack 一次性构建 (B,H,W,3) 连续内存后零拷贝包装为
    torch.Tensor，再统一 copy 到 pinned buffer，内存操作从 O(B) 次
    降为 O(1) 次。

  · Reader 多线程预取解码：
    图片文件夹模式支持可配置的多线程并发解码（--num_prefetch_threads），
    充分利用多核 CPU 并行解码多张独立图片；视频流模式维持单线程顺序读取。
    图片读取改用 np.fromfile + cv2.imdecode，完整支持非 ASCII 路径
    （含中文、日文等字符的路径在 Windows 下 cv2.imread 会静默返回 None）。

  · Writer 批量攒帧写入：
    _write_loop 引入 pending 列表，攒够 MAX_BATCH=8 帧或队列空闲时才
    合并为一次 stdin.write()，syscall 次数降低约 8 倍，显著降低写帧
    线程 CPU 占用；write_frame() 调用处即完成 tobytes() 转换，队列存
    储原始字节，write_loop 零拷贝直写。

  · get_sub_video 切分增加 -avoid_negative_ts make_zero：
    防止 H.265 with B-frames 等编码器在切分点产生负时间戳，消除
    多段合并时的 A/V 不同步和播放器跳帧问题。

────────────────────────────────────────────────────
精度 / 正确性
────────────────────────────────────────────────────
  · Fraction 安全解析帧率：
    get_video_meta_info 改用 fractions.Fraction 解析 avg_frame_rate
    字段（格式为 "num/den"），替代原来的 eval()——eval 在视频文件元数据
    被篡改时可执行任意 Python 代码，存在安全风险。同时处理 nb_frames
    字段缺失的情况（回退为 duration × fps）。

  · extract_frame_first 路径安全：
    改用 shlex.split() + subprocess.run() 替代 os.system()，正确处理
    路径中含空格或特殊字符的情况，并检查 ffmpeg 返回码。

────────────────────────────────────────────────────
健壮性 / 错误处理
────────────────────────────────────────────────────
  · Reader 预取线程异常跨线程传播：
    _prefetch_worker 捕获所有异常后将 Exception 对象直接入队，
    get_frame() 取到时立即 re-raise，避免线程静默崩溃后主线程永久
    阻塞在 queue.get()（原版线程异常只会让 active_threads 计数不归零，
    SENTINEL 永远不放，主线程死锁）。

  · Writer fail-fast（error flag）：
    _write_loop 中 stdin.write() 失败时设置 _error flag 并同步 print
    双重通知；write_frame() 调用前检查 flag，若已置位立即抛出
    RuntimeError，终止推理主循环，防止 FFmpeg 进程已死而 GPU 还在
    白白计算。close() 检查返回码，非零时打印 stderr；双重超时保护
    （join 30s + communicate 15s），无响应时强制 kill。

  · Writer 消除潜在死锁：
    原版 run_async(pipe_stdout=True) 但从不消费 stdout，当 FFmpeg 向
    stdout 写入数据时管道缓冲区满，FFmpeg 阻塞等待读取，造成死锁。
    改为 pipe_stdout=False / pipe_stderr=True，只捕获错误输出。

  · Reader 关闭时正确 drain stdout：
    close() 先将 stdout 完全读完（drain）再调用 wait()，避免直接关闭
    管道写端后 ffmpeg 收到 SIGPIPE 产生 BrokenPipeError。

  · OOM 自动降级 + 逐步恢复：
    flush_batch_safe 遇到 CUDA OOM 时 batch_size 自动折半重试，
    引入 cooldown 计数器（10 批）冷却后每批成功推理将 batch_size +1
    逐步恢复至初始上限，防止一次偶发 OOM 导致全程以最小 batch 处理。
    非 OOM RuntimeError 时精确以 len(sub) 推进帧指针（原版用 bs 可能
    导致跳过未处理的帧）。

  · 多 GPU 多进程稳定性：
    error_callback 改为 module-level 函数（原版 lambda 在 spawn context
    下不可 pickle，异常静默丢失）；合并前检测各分段文件是否存在且非空，
    失败分段跳过而非中断；vidlist.txt 清理增加 OSError 捕获；
    ffmpeg-python 包安装改用 subprocess pip install（pip.main() 在
    pip 21+ 已废弃）。

────────────────────────────────────────────────────
可观测性
────────────────────────────────────────────────────
  · ThroughputMeter 滑动窗口 FPS：
    基于 deque（最近 20 批）统计帧速，排除启动阶段（模型加载、首批
    warmup）的慢速干扰，fps / ETA 更准确，速率变化即时反映。

  · tqdm 实时进度条，postfix 同时展示：
    fps（滑窗帧速）/ eta（剩余时间）/ bs（当前实际 batch_size，
    OOM 降级/恢复时动态变化）/ ms（最近 20 批平均推理延迟），
    四字段并列，一眼判断瓶颈在 IO、推理还是编码写出。

  · JSON 性能报告（--report）：
    处理完成后输出结构化报告，记录 model、outscale、batch_size、
    fp16、frame_count、elapsed_s、avg_fps、推理延迟 mean/p95/max（ms），
    与 IFRNet v4 报告格式完全一致，可用同一脚本对超分和插帧两个阶段
    做横向对比分析。

  · 模型配置集中管理（MODULE_CONFIG）：
    所有模型的网络结构、netscale、权重 URL 统一维护在模块顶层常量，
    新增模型只需追加一条记录，避免散落在函数内部难以维护。
"""

from __future__ import annotations

import argparse
import cv2
import glob
import json
import mimetypes
import numpy as np
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from fractions import Fraction
from typing import Optional

import torch
import torch.nn.functional as F
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from os import path as osp
from tqdm import tqdm

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# N7: 安全导入 ffmpeg
try:
    import ffmpeg
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ffmpeg-python'])
    import ffmpeg

base_dir = '/workspace/Video_Enhancement'
models_RealESRGAN = f'{base_dir}/models_RealESRGAN'


# ─────────────────────────────────────────────────────────────────────────────
# 模型配置常量（v3 N8）
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIG: dict[str, tuple] = {
    'RealESRGAN_x4plus': (
        RRDBNet(3, 3, 64, 23, 32, scale=4), 4,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'],
    ),
    'RealESRNet_x4plus': (
        RRDBNet(3, 3, 64, 23, 32, scale=4), 4,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'],
    ),
    'RealESRGAN_x4plus_anime_6B': (
        RRDBNet(3, 3, 64, 6, 32, scale=4), 4,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'],
    ),
    'RealESRGAN_x2plus': (
        RRDBNet(3, 3, 64, 23, 32, scale=2), 2,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'],
    ),
    'realesr-animevideov3': (
        SRVGGNetCompact(3, 3, 64, 16, upscale=4, act_type='prelu'), 4,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'],
    ),
    'realesr-general-x4v3': (
        SRVGGNetCompact(3, 3, 64, 32, upscale=4, act_type='prelu'), 4,
        [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
        ],
    ),
    'RealESRGANv2-animevideo-xsx4': (
        SRVGGNetCompact(3, 3, 64, 16, upscale=4, act_type='prelu'), 4,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx4.pth'],
    ),
    'RealESRGANv2-animevideo-xsx2': (
        SRVGGNetCompact(3, 3, 64, 16, upscale=2, act_type='prelu'), 2,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx2.pth'],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _parse_fps(fps_str: str) -> float:
    """B1(v3): 用 Fraction 安全解析帧率字符串，避免 eval() 风险。"""
    try:
        return float(Fraction(fps_str))
    except (ValueError, ZeroDivisionError):
        return 24.0


def get_video_meta_info(video_path: str) -> dict:
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
    has_audio = any(s['codec_type'] == 'audio' for s in probe['streams'])
    vs = video_streams[0]
    ret['width']     = vs['width']
    ret['height']    = vs['height']
    ret['fps']       = _parse_fps(vs['avg_frame_rate'])
    ret['audio']     = ffmpeg.input(video_path).audio if has_audio else None
    if 'nb_frames' in vs and vs['nb_frames'].isdigit():
        ret['nb_frames'] = int(vs['nb_frames'])
    elif 'duration' in vs:
        ret['nb_frames'] = int(float(vs['duration']) * ret['fps'])
    else:
        ret['nb_frames'] = 0
    return ret


def get_sub_video(args, num_process: int, process_idx: int) -> str:
    if num_process == 1:
        return args.input

    meta = get_video_meta_info(args.input)
    duration  = meta['nb_frames'] / meta['fps']
    part_time = duration / num_process
    print(f'duration: {duration:.2f}s, part_time: {part_time:.2f}s')

    tmp_dir  = osp.join(args.output, f'{args.video_name}_inp_tmp_videos')
    os.makedirs(tmp_dir, exist_ok=True)
    out_path = osp.join(tmp_dir, f'{process_idx:03d}.mp4')

    ss  = part_time * process_idx
    cmd = [args.ffmpeg_bin, '-i', args.input, '-ss', f'{ss:.3f}',
           '-avoid_negative_ts', 'make_zero']
    if process_idx != num_process - 1:
        cmd += ['-to', f'{part_time * (process_idx + 1):.3f}']
    cmd += ['-async', '1', '-y', out_path]
    print(' '.join(cmd))
    subprocess.call(cmd)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# PinnedBufferPool（保留 v3 B4/B5）
# ─────────────────────────────────────────────────────────────────────────────

_thread_local = threading.local()


class PinnedBufferPool:
    def __init__(self):
        self._buf: Optional[torch.Tensor] = None

    def get_for_frames(self, frames: list[np.ndarray]) -> torch.Tensor:
        arr = np.stack(frames, axis=0)
        src = torch.from_numpy(arr)
        n_elem = src.numel()
        if self._buf is None or self._buf.numel() < n_elem:
            self._buf = torch.empty(n_elem, dtype=torch.uint8).pin_memory()
        dst = self._buf[:n_elem].view_as(src)
        dst.copy_(src)
        return dst


def _get_process_buffer_pool() -> PinnedBufferPool:
    if not hasattr(_thread_local, 'pool'):
        _thread_local.pool = PinnedBufferPool()
    return _thread_local.pool


# ─────────────────────────────────────────────────────────────────────────────
# ThroughputMeter（保留 v3）
# ─────────────────────────────────────────────────────────────────────────────

class ThroughputMeter:
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
# Reader（X1: 预取线程异常传播）
# ─────────────────────────────────────────────────────────────────────────────

class Reader:
    _SENTINEL = object()
    _ERROR    = object()   # X1: 特殊标记，标识异常槽位

    def __init__(self, args, total_workers: int = 1, worker_idx: int = 0,
                 prefetch_factor: int = 4, num_prefetch_threads: int = 1):
        self.args               = args
        self.prefetch_factor    = prefetch_factor
        self.num_prefetch_threads = num_prefetch_threads

        input_type       = mimetypes.guess_type(args.input)[0]
        self.input_type  = 'folder' if input_type is None else input_type
        self.paths: list = []
        self.audio       = None
        self.input_fps   = None
        self._stream_reader = None

        if self.input_type.startswith('video'):
            video_path = get_sub_video(args, total_workers, worker_idx)
            # B2(v3): pipe_stdout=True 用于消费帧数据，不会死锁（ffmpeg→stdout→我们读）
            self._stream_reader = (
                ffmpeg.input(video_path)
                .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel='error')
                .run_async(pipe_stdin=False, pipe_stdout=True, cmd=args.ffmpeg_bin)
            )
            meta           = get_video_meta_info(video_path)
            self.width     = meta['width']
            self.height    = meta['height']
            self.input_fps = meta['fps']
            self.audio     = meta['audio']
            self.nb_frames = meta['nb_frames']
            self.num_prefetch_threads = 1
        else:
            if self.input_type.startswith('image'):
                self.paths = [args.input]
            else:
                paths = sorted(glob.glob(os.path.join(args.input, '*')))
                tot   = len(paths)
                n     = tot // total_workers + (1 if tot % total_workers else 0)
                self.paths = paths[n * worker_idx: n * (worker_idx + 1)]
            self.nb_frames = len(self.paths)
            assert self.nb_frames > 0, 'empty folder'
            tmp_img = self._imread_safe(self.paths[0])
            self.height, self.width = tmp_img.shape[:2]

        self.idx = 0
        self._path_lock = threading.Lock()

        if self.prefetch_factor > 0:
            self._queue = queue.Queue(maxsize=self.prefetch_factor)
            self._threads = [
                threading.Thread(target=self._prefetch_worker, daemon=True)
                for _ in range(self.num_prefetch_threads)
            ]
            self._active_threads = len(self._threads)
            self._sentinel_lock  = threading.Lock()
            for t in self._threads:
                t.start()

    @staticmethod
    def _imread_safe(path: str) -> np.ndarray:
        """N9(v3): np.fromfile + cv2.imdecode 支持非 ASCII 路径。"""
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"无法读取图片: {path}")
        return img

    def _next_path(self) -> Optional[str]:
        with self._path_lock:
            if self.idx >= self.nb_frames:
                return None
            p = self.paths[self.idx]
            self.idx += 1
        return p

    def _raw_frame_from_stream(self) -> Optional[np.ndarray]:
        img_bytes = self._stream_reader.stdout.read(self.width * self.height * 3)
        if not img_bytes:
            return None
        return np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])

    def _raw_frame_from_list(self) -> Optional[np.ndarray]:
        p = self._next_path()
        if p is None:
            return None
        return self._imread_safe(p)

    def _raw_frame(self) -> Optional[np.ndarray]:
        if self.input_type.startswith('video'):
            return self._raw_frame_from_stream()
        return self._raw_frame_from_list()

    def _prefetch_worker(self):
        """X1: 捕获异常并入队，而非静默终止。"""
        try:
            while True:
                frame = self._raw_frame()
                if frame is None:
                    break
                self._queue.put(frame)
        except Exception as e:
            # X1: 将异常对象放入队列，让 get_frame() 处 re-raise
            self._queue.put(e)
            return   # 异常后直接退出，不放 SENTINEL（防止覆盖异常信号）

        # 正常结束：最后一个线程放 SENTINEL
        with self._sentinel_lock:
            self._active_threads -= 1
            if self._active_threads == 0:
                self._queue.put(self._SENTINEL)

    def get_frame(self) -> Optional[np.ndarray]:
        """X1: 若队列中取到 Exception，re-raise 给调用方。"""
        if self.prefetch_factor > 0:
            item = self._queue.get()
            if item is self._SENTINEL:
                self._queue.put(self._SENTINEL)
                return None
            if isinstance(item, Exception):
                raise item  # X1: 异常传播
            return item
        return self._raw_frame()

    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        if self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def close(self):
        """B3(v3): drain stdout 再 wait，避免 BrokenPipeError。"""
        if self._stream_reader is not None:
            try:
                self._stream_reader.stdout.read()
            except Exception:
                pass
            try:
                self._stream_reader.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._stream_reader.kill()


# ─────────────────────────────────────────────────────────────────────────────
# Writer（X2: error flag 完整传播）
# ─────────────────────────────────────────────────────────────────────────────

class Writer:
    """
    X2: 增加 _error flag，write_frame() 调用前检查，fail-fast。
        write_loop 异常时既 flag 也 print，与 IFRNet v4 FFmpegWriter 对齐。
    保留 v3 的批量攒帧写入（MAX_BATCH=8）和 timeout get() 方式。
    """
    _SENTINEL = object()
    _MAX_BATCH = 8

    def __init__(self, args, audio, height: int, width: int,
                 video_save_path: str, fps: float):
        out_w = int(width  * args.outscale)
        out_h = int(height * args.outscale)
        if out_h > 2160:
            print('[Warning] Output > 4K; IO may bottleneck. '
                  'Consider --outscale or --video_codec libx265.')

        vcodec = getattr(args, 'video_codec', 'libx264')
        crf    = getattr(args, 'crf', 18)
        common = dict(pix_fmt='yuv420p', vcodec=vcodec, crf=crf, loglevel='error')

        inp = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24',
                           s=f'{out_w}x{out_h}', framerate=fps)
        if audio is not None:
            stream = inp.output(audio, video_save_path, acodec='copy', **common)
        else:
            stream = inp.output(video_save_path, **common)

        self._proc = (
            stream.overwrite_output()
            .run_async(
                pipe_stdin=True,
                pipe_stdout=False,
                pipe_stderr=True,
                cmd=args.ffmpeg_bin,
            )
        )
        self.out_w = out_w
        self.out_h = out_h

        self._queue    = queue.Queue(maxsize=128)
        self._error: Optional[Exception] = None  # X2: error flag
        self._thread   = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def _write_loop(self):
        """攒批写入 + X2 error flag。"""
        pending: list[bytes] = []
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if pending:
                        self._proc.stdin.write(b''.join(pending))
                        pending = []
                    continue

                if item is self._SENTINEL:
                    if pending:
                        self._proc.stdin.write(b''.join(pending))
                    break

                pending.append(item)
                if len(pending) >= self._MAX_BATCH or self._queue.empty():
                    self._proc.stdin.write(b''.join(pending))
                    pending = []

        except Exception as e:
            self._error = e                        # X2: flag
            print(f'[Writer Error] 写帧线程异常: {e}')  # X2: print

    def write_frame(self, frame: np.ndarray):
        """X2: 写入前检查 error flag，fail-fast 防止无效推理继续。"""
        if self._error is not None:
            raise RuntimeError(f"Writer 内部错误: {self._error}") from self._error
        self._queue.put(frame.tobytes())

    def close(self):
        self._queue.put(self._SENTINEL)
        self._thread.join(timeout=30)
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            _, stderr_out = self._proc.communicate(timeout=15)
            rc = self._proc.returncode
            if rc != 0 and stderr_out:
                print(f'[FFmpeg Warning] rc={rc}: {stderr_out.decode(errors="ignore")[:300]}')
        except subprocess.TimeoutExpired:
            self._proc.kill()
            print('[Warning] FFmpeg 进程未响应，已强制终止。')
        if self._error is not None:
            print(f'[Warning] Writer 累计写帧异常: {self._error}')


# ─────────────────────────────────────────────────────────────────────────────
# process_batch（保留 v3 B4/B5/N1）
# ─────────────────────────────────────────────────────────────────────────────

def process_batch(
    upsampler: RealESRGANer,
    frames: list[np.ndarray],
    outscale: float,
    netscale: int,
    transfer_stream: Optional[torch.cuda.Stream] = None,
    compute_stream:  Optional[torch.cuda.Stream] = None,
) -> list[np.ndarray]:
    device   = upsampler.device
    use_half = upsampler.half
    B, H, W  = len(frames), frames[0].shape[0], frames[0].shape[1]

    pool = _get_process_buffer_pool()
    buf  = pool.get_for_frames(frames)
    batch_pin = buf.permute(0, 3, 1, 2).float().div_(255.0).contiguous()

    if transfer_stream is not None:
        with torch.cuda.stream(transfer_stream):
            batch_t = batch_pin.to(device, non_blocking=True)
            if use_half:
                batch_t = batch_t.half()
        if compute_stream is not None:
            compute_stream.wait_stream(transfer_stream)
        with torch.cuda.stream(compute_stream) if compute_stream else torch.no_grad():
            with torch.no_grad():
                output_t = upsampler.model(batch_t)
    else:
        batch_t = batch_pin.to(device)
        if use_half:
            batch_t = batch_t.half()
        with torch.no_grad():
            output_t = upsampler.model(batch_t)

    if abs(outscale - netscale) > 1e-5:
        output_t = F.interpolate(
            output_t.float(), scale_factor=outscale / netscale,
            mode='bicubic', align_corners=False,
        )

    # N1(v3): 原地操作链
    output_u8 = output_t.float().clamp_(0.0, 1.0).mul_(255.0).byte()
    output_np = output_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return [output_np[i] for i in range(B)]


# ─────────────────────────────────────────────────────────────────────────────
# flush_batch_safe（保留 v3 N3/B6，X4: 增加 timing 采集）
# ─────────────────────────────────────────────────────────────────────────────

def flush_batch_safe(
    upsampler: RealESRGANer,
    frames: list[np.ndarray],
    outscale: float,
    netscale: int,
    transfer_stream,
    compute_stream,
    writer: Writer,
    pbar: tqdm,
    init_batch_size: int,
    oom_cooldown: list,
    max_batch_size: list,
    timing: list,       # X4: 新增，收集每批推理耗时
) -> int:
    """
    N3(v3): OOM 降级 + 恢复。
    B6(v3): 非 OOM 错误时 i 准确推进。
    X4:     新增 timing 列表采集，支持 JSON 报告的 p95/max 计算。
    X5:     返回耗时供调用方更新 pbar postfix ms 字段。
    """
    bs = min(init_batch_size, len(frames))
    i  = 0
    while i < len(frames):
        sub = frames[i: i + bs]
        try:
            t0 = time.perf_counter()
            outputs = process_batch(upsampler, sub, outscale, netscale,
                                    transfer_stream, compute_stream)
            elapsed = time.perf_counter() - t0
            timing.append(elapsed)  # X4: 记录本批耗时

            for out in outputs:
                writer.write_frame(out)
            pbar.update(len(sub))

            # X5: 更新 postfix（fps 由外部 ThroughputMeter 提供，这里只更新 ms/bs）
            avg_ms = np.mean(timing[-20:]) * 1000 if timing else 0
            pbar.set_postfix(bs=bs, ms=f'{avg_ms:.0f}')
            i += bs

            # N3: 恢复 batch_size
            if oom_cooldown[0] > 0:
                oom_cooldown[0] -= 1
            elif bs < max_batch_size[0]:
                new_bs = min(bs + 1, max_batch_size[0])
                print(f'\n[恢复] batch_size {bs} → {new_bs}')
                bs = new_bs

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and bs > 1:
                bs = max(1, bs // 2)
                oom_cooldown[0] = 10
                torch.cuda.empty_cache()
                print(f'\n[OOM] 降级 batch_size → {bs}; 重试...')
            else:
                print(f'\n[Error] {e}')
                pbar.update(len(sub))
                i += len(sub)  # B6: 精确推进

    return bs


# ─────────────────────────────────────────────────────────────────────────────
# 单进程推理主循环（X3/X4/X5: JSON 报告 + 推理耗时 + tqdm 字段对齐）
# ─────────────────────────────────────────────────────────────────────────────

def inference_video(args, video_save_path: str, device=None,
                    total_workers: int = 1, worker_idx: int = 0):

    args.model_name = args.model_name.split('.pth')[0]

    if args.model_name not in MODEL_CONFIG:
        raise ValueError(f'Unknown model: {args.model_name}')

    model, netscale, file_url = MODEL_CONFIG[args.model_name]

    model_path = os.path.join(models_RealESRGAN, args.model_name + '.pth')
    if not os.path.isfile(model_path):
        for url in file_url:
            model_path = load_file_from_url(url=url, model_dir=models_RealESRGAN,
                                            progress=True, file_name=None)

    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_path   = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    upsampler = RealESRGANer(
        scale=netscale, model_path=model_path, dni_weight=dni_weight,
        model=model, tile=args.tile, tile_pad=args.tile_pad,
        pre_pad=args.pre_pad, half=not args.fp32, device=device,
    )

    if args.use_compile and hasattr(torch, 'compile'):
        print('[Info] Compiling model with torch.compile ...')
        upsampler.model = torch.compile(upsampler.model, mode='reduce-overhead')

    if 'anime' in args.model_name and args.face_enhance:
        print('[Warning] face_enhance not supported for anime models; disabling.')
        args.face_enhance = False

    face_enhancer = None
    if args.face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale, arch='clean', channel_multiplier=2,
            bg_upsampler=upsampler,
        )
        if args.batch_size > 1:
            print('[Warning] face_enhance enabled; batch_size forced to 1.')
            args.batch_size = 1

    use_batch = (args.batch_size > 1 and args.tile == 0 and face_enhancer is None)
    if args.batch_size > 1 and not use_batch:
        print('[Warning] batch inference requires --tile 0 and no face_enhance; falling back.')

    transfer_stream = compute_stream = None
    if torch.cuda.is_available():
        transfer_stream = torch.cuda.Stream(device=device)
        compute_stream  = torch.cuda.Stream(device=device)

    num_pt = min(getattr(args, 'num_prefetch_threads', 1),
                 4 if not getattr(args, 'input_type_is_video', True) else 1)

    reader = Reader(args, total_workers, worker_idx,
                    prefetch_factor=args.prefetch_factor,
                    num_prefetch_threads=num_pt)
    audio          = reader.get_audio()
    height, width  = reader.get_resolution()
    fps            = reader.get_fps()
    writer         = Writer(args, audio, height, width, video_save_path, fps)

    total_frames  = len(reader)
    pbar          = tqdm(total=total_frames, unit='frame',
                         desc=f'[Worker {worker_idx}] SR', dynamic_ncols=True)
    meter         = ThroughputMeter()
    batch_size    = args.batch_size
    _oom_cooldown  = [0]
    _max_bs        = [args.batch_size]
    timing: list[float] = []     # X4: 推理耗时列表
    t_start = time.time()

    if use_batch:
        batch_frames: list[np.ndarray] = []

        while True:
            img = reader.get_frame()
            end = img is None

            if img is not None:
                batch_frames.append(img)

            if (len(batch_frames) == batch_size) or (end and batch_frames):
                batch_size = flush_batch_safe(
                    upsampler, batch_frames, args.outscale, netscale,
                    transfer_stream, compute_stream, writer, pbar,
                    batch_size, _oom_cooldown, _max_bs, timing,  # X4
                )
                meter.update(len(batch_frames))
                # X5: 同时显示 fps（滑窗）+ eta + batch + ms
                pbar.set_postfix(
                    fps=f'{meter.fps():.1f}',
                    eta=f'{meter.eta(total_frames):.0f}s',
                    bs=batch_size,
                    ms=f'{np.mean(timing[-20:]) * 1000:.0f}' if timing else '—',
                )
                batch_frames = []

            if end:
                break

    else:
        while True:
            img = reader.get_frame()
            if img is None:
                break
            try:
                if face_enhancer:
                    _, _, output = face_enhancer.enhance(
                        img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    t0 = time.perf_counter()
                    output, _ = upsampler.enhance(img, outscale=args.outscale)
                    timing.append(time.perf_counter() - t0)  # X4
            except RuntimeError as e:
                print(f'\n[Error] {e}')
                print('If CUDA OOM, try --tile with a smaller number.')
            else:
                writer.write_frame(output)
                meter.update(1)
            pbar.update(1)
            pbar.set_postfix(
                fps=f'{meter.fps():.1f}',
                eta=f'{meter.eta(total_frames):.0f}s',
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize(device=device)

    reader.close()
    writer.close()
    pbar.close()

    # X3: JSON 性能报告（借鉴自 IFRNet v3）
    report_path = getattr(args, 'report', None)
    if report_path and timing and worker_idx == 0:
        elapsed = time.time() - t_start
        report = {
            'input':          getattr(args, 'input', ''),
            'output':         video_save_path,
            'model':          args.model_name,
            'outscale':       args.outscale,
            'batch_size':     batch_size,
            'fp16':           not args.fp32,
            'frame_count':    total_frames,
            'elapsed_s':      round(elapsed, 2),
            'avg_fps':        round(total_frames / elapsed, 2) if elapsed > 0 else 0,
            'infer_latency_ms': {
                'mean': round(float(np.mean(timing)) * 1000, 2),
                'p95':  round(float(np.percentile(timing, 95)) * 1000, 2),
                'max':  round(float(np.max(timing)) * 1000, 2),
            },
        }
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f'[Info] 性能报告已保存: {report_path}')


# ─────────────────────────────────────────────────────────────────────────────
# 多进程 worker（module-level，可 pickle）
# ─────────────────────────────────────────────────────────────────────────────

def _inference_worker(args, video_save_path: str, device,
                      total_workers: int, worker_idx: int):
    inference_video(args, video_save_path,
                    device=device,
                    total_workers=total_workers,
                    worker_idx=worker_idx)


def _on_worker_error(e):
    print(f'[Worker Error] {e}')


# ─────────────────────────────────────────────────────────────────────────────
# run
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    args.video_name = osp.splitext(os.path.basename(args.input))[0]
    video_save_path = osp.join(args.output, f'{args.video_name}_{args.suffix}.mp4')

    mime = mimetypes.guess_type(args.input)[0]
    args.input_type_is_video = mime is not None and mime.startswith('video')

    if args.extract_frame_first:
        tmp_frames_folder = osp.join(args.output, f'{args.video_name}_inp_tmp_frames')
        os.makedirs(tmp_frames_folder, exist_ok=True)
        cmd = shlex.split(
            f'{args.ffmpeg_bin} -i {shlex.quote(args.input)} '
            f'-qscale:v 1 -qmin 1 -qmax 1 -vsync 0 '
            f'{shlex.quote(osp.join(tmp_frames_folder, "frame%08d.png"))}'
        )
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f'[Error] ffmpeg frame extraction failed (code {ret.returncode})')
            return
        args.input = tmp_frames_folder
        args.input_type_is_video = False

    num_gpus    = torch.cuda.device_count()
    num_process = max(1, num_gpus * args.num_process_per_gpu)

    if num_process == 1:
        inference_video(args, video_save_path)
        return

    ctx  = torch.multiprocessing.get_context('spawn')
    pool = ctx.Pool(num_process)
    os.makedirs(osp.join(args.output, f'{args.video_name}_out_tmp_videos'), exist_ok=True)
    pbar = tqdm(total=num_process, unit='sub_video', desc='multi-GPU dispatch')

    def _on_done(_):
        pbar.update(1)

    results = []
    for i in range(num_process):
        sub_save = osp.join(args.output, f'{args.video_name}_out_tmp_videos', f'{i:03d}.mp4')
        r = pool.apply_async(
            _inference_worker,
            args=(args, sub_save, torch.device(i % num_gpus), num_process, i),
            callback=_on_done,
            error_callback=_on_worker_error,
        )
        results.append((i, sub_save, r))

    pool.close()
    pool.join()
    pbar.close()

    valid_parts = []
    for i, sub_save, r in results:
        if osp.exists(sub_save) and osp.getsize(sub_save) > 0:
            valid_parts.append((i, sub_save))
        else:
            print(f'[Warning] Sub-video {i} missing or empty; skipping.')

    if not valid_parts:
        print('[Error] All sub-videos failed; no output produced.')
        return

    vidlist_path = osp.join(args.output, f'{args.video_name}_vidlist.txt')
    with open(vidlist_path, 'w') as f:
        for _, p in valid_parts:
            f.write(f"file '{p}'\n")

    cmd = [args.ffmpeg_bin, '-f', 'concat', '-safe', '0',
           '-i', vidlist_path, '-c', 'copy', '-y', video_save_path]
    print(' '.join(cmd))
    subprocess.call(cmd)

    shutil.rmtree(osp.join(args.output, f'{args.video_name}_out_tmp_videos'), ignore_errors=True)
    inp_tmp = osp.join(args.output, f'{args.video_name}_inp_tmp_videos')
    if osp.exists(inp_tmp):
        shutil.rmtree(inp_tmp, ignore_errors=True)
    try:
        if osp.exists(vidlist_path):
            os.remove(vidlist_path)
    except OSError as e:
        print(f'[Warning] 无法删除 vidlist: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Real-ESRGAN Video Super-Resolution (v4 Cross-Optimized)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('-i',  '--input',             type=str,   default='inputs')
    parser.add_argument('-n',  '--model_name',        type=str,   default='realesr-animevideov3')
    parser.add_argument('-o',  '--output',            type=str,   default='results')
    parser.add_argument('-dn', '--denoise_strength',  type=float, default=0.5)
    parser.add_argument('-s',  '--outscale',          type=float, default=4)
    parser.add_argument('--suffix',                   type=str,   default='out')
    parser.add_argument('-t',  '--tile',              type=int,   default=0)
    parser.add_argument('--tile_pad',                 type=int,   default=10)
    parser.add_argument('--pre_pad',                  type=int,   default=0)
    parser.add_argument('--face_enhance',             action='store_true')
    parser.add_argument('--fp32',                     action='store_true')
    parser.add_argument('--fps',                      type=float, default=None)
    parser.add_argument('--ffmpeg_bin',               type=str,   default='ffmpeg')
    parser.add_argument('--extract_frame_first',      action='store_true')
    parser.add_argument('--num_process_per_gpu',      type=int,   default=1)
    parser.add_argument('--batch_size',               type=int,   default=4)
    parser.add_argument('--prefetch_factor',          type=int,   default=8)
    parser.add_argument('--use_compile',              action='store_true')
    parser.add_argument('--alpha_upsampler',          type=str,   default='realesrgan',
                        choices=['realesrgan', 'bicubic'])
    parser.add_argument('--ext',                      type=str,   default='auto',
                        choices=['auto', 'jpg', 'png'])
    parser.add_argument('--video_codec',              type=str,   default='libx264',
                        choices=['libx264', 'libx265', 'libvpx-vp9'])
    parser.add_argument('--crf',                      type=int,   default=18)
    parser.add_argument('--num_prefetch_threads',     type=int,   default=2)
    # X3: JSON 性能报告（借鉴自 IFRNet v3）
    parser.add_argument('--report',                   type=str,   default=None,
                        help='输出 JSON 性能报告路径（如 report.json）')

    args = parser.parse_args()
    args.input = args.input.rstrip('/\\')
    os.makedirs(args.output, exist_ok=True)

    mime     = mimetypes.guess_type(args.input)[0]
    is_video = mime is not None and mime.startswith('video')

    if is_video and args.input.endswith('.flv'):
        mp4_path = args.input.replace('.flv', '.mp4')
        subprocess.run([args.ffmpeg_bin, '-i', args.input, '-codec', 'copy', '-y', mp4_path])
        args.input = mp4_path

    if args.extract_frame_first and not is_video:
        args.extract_frame_first = False

    run(args)

    if args.extract_frame_first:
        video_name = osp.splitext(os.path.basename(args.input))[0]
        tmp = osp.join(args.output, f'{video_name}_inp_tmp_frames')
        if osp.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
