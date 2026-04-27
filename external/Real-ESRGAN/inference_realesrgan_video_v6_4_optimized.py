#!/usr/bin/env python3
"""
Real-ESRGAN Video Enhancement v6.4 - 架构优化版 (最终修复版)
基于v6.2代码重构，实现深度流水线、GPU内存池和异步计算优化
修复了视频末尾卡死的问题，增强退出机制。

FIX-DET-THRESHOLD: 增加人脸检测置信度阈值，过滤低质量检测，减少不必要的 GFPGAN 推理
FIX-ADAPTIVE-BATCH: 基于人脸密度的自适应批处理大小，提升 GPU 利用率
FIX-GPU-PREFETCH: SR 推理完成后预取下一批 H2D 传输，重叠传输与计算

FIX-BGR:        FFmpeg 读写统一 bgr24，与 OpenCV/GFPGAN BGR 约定一致
FIX-INV-AFFINE: 预计算逆仿射矩阵乘以 upscale_factor，人脸贴回位置正确
FIX-TASK-ID:    单调递增 task_id 替代 id()，杜绝内存地址复用导致结果错配
FIX-SLOT-POOL:  Queue 池管理共享内存 slot，与 pending 出队联动释放
FIX-SYNC-PATH:  清理同步回退路径不可靠判断

FIX-READER-DIAG: FFmpegReader 增加 stderr drain 线程 + _send_eof_guaranteed
                 + 四路径清晰诊断 + is_reader_alive()/is_eof_sent() 探活接口
FIX-READ-WATCHDOG: _read_frames 删除 except BaseException，改为超时看门狗 +
                   reader 探活，连续 60s 无帧 → 主动终止；finally 不再置
                   self.running=False，让哨兵沿流水线自然传播
FIX-WRITE-WATCHDOG: _write_frames 增加全流水线空转看门狗，120s 全空且无哨兵
                    → dump 所有线程栈 + 强制退出，消灭静默死锁
"""

import os
import sys
import time
import select
import queue
import threading
import concurrent.futures
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import ffmpeg
import fractions
import re
import traceback
from typing import List, Optional, Tuple, Dict, Any
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入必要的模块
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url
from os import path as osp

try:
    from gfpgan import GFPGANer
except ImportError:
    GFPGANer = None

# 导入必要的类定义（这些类在原始v6.2脚本中定义）
import subprocess
from collections import deque

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

# 路径配置
_SCRIPT_DIR       = osp.dirname(osp.abspath(__file__))
base_dir          = osp.dirname(osp.dirname(_SCRIPT_DIR))
models_RealESRGAN = osp.join(base_dir, 'models_RealESRGAN')

# 模型配置常量
MODEL_CONFIG = {
    'RealESRGAN_x4plus': (
        RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'],
    ),
    'RealESRNet_x4plus': (
        RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth'],
    ),
    'RealESRGAN_x4plus_anime_6B': (
        RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4), 4,
        ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'],
    ),
    'RealESRGAN_x2plus': (
        RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2), 2,
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
# PinnedBufferPool（线程本地 pinned CPU buffer）
# 预分配 pinned CPU buffer 并复用，避免每批 H2D 前 pin_memory 的 malloc 开销。
# _out_buf 用于异步 D2H 输出，避免 .cpu() 引发隐式同步。
# ─────────────────────────────────────────────────────────────────────────────

_thread_local = threading.local()


class PinnedBufferPool:
    def __init__(self):
        self._buf:     Optional[torch.Tensor] = None
        self._out_buf: Optional[torch.Tensor] = None

    def get_for_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        arr    = np.stack(frames, axis=0)
        src    = torch.from_numpy(arr)
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
            # 数据不足，用总时间兜底
            if self._total == 0:
                return 0.0
            total_time = time.time() - self._start_time
            return self._total / total_time if total_time > 0 else 0.0

        # FIX-FPS: _times 存的是 (timestamp, n_increment)
        # 窗口内总帧数 = sum(所有增量)，而不是 (最后增量 - 第一增量)
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
    
    # 对于多权重模型，使用第一个权重文件
    model_path = model_paths[0] if model_paths else None
    
    return RealESRGANer(
        scale=netscale, model_path=model_path, dni_weight=dni_weight,
        model=model, tile=tile, tile_pad=tile_pad,
        pre_pad=pre_pad, half=use_half, device=device,
    )


def get_video_meta_info(video_path: str) -> dict:
    """通过 ffprobe 获取视频元数据，包含宽高/帧率/帧数/音轨。"""
    probe = ffmpeg.probe(video_path)
    video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
    has_audio     = any(s['codec_type'] == 'audio' for s in probe['streams'])
    vs = video_streams[0]
    
    # 解析帧率
    fps_str = vs.get('avg_frame_rate', '24/1')
    try:
        fps = float(fractions.Fraction(fps_str))
    except:
        fps = 24.0
    
    # 获取帧数
    if 'nb_frames' in vs and vs['nb_frames'].isdigit():
        nb = int(vs['nb_frames'])
    elif 'duration' in vs:
        nb = int(float(vs['duration']) * fps)
    else:
        nb = 0
    
    return {
        'width':     vs['width'],
        'height':    vs['height'],
        'fps':       fps,
        'audio':     ffmpeg.input(video_path).audio if has_audio else None,
        'nb_frames': nb,
    }


class FFmpegReader:
    """通过FFmpeg pipe读取视频帧

    FIX-READER-DIAG: 四大增强：
      1. _drain_stderr 后台线程持续抓取 ffmpeg stderr，避免 pipe 满导致 ffmpeg 阻塞，
         同时供 EOF/短读/退出场景打印诊断。
      2. _send_eof_guaranteed 在 finally 中反复重试送 None 到下游，直到送达或
         pipeline 主动关闭，杜绝下游静默等待。
      3. _read_loop 对 stdout EOF、短读、reshape 失败、下游满 + ffmpeg 已退出
         四种路径分别打清晰诊断日志，附带 ffmpeg 返回码和 stderr 尾部。
      4. 新增 is_reader_alive() / is_eof_sent() / get_frames_produced() 等探活
         接口，供下游看门狗判断上游健康状态。
    """

    # FIX-PREMATURE-EOF: 超时哨兵，与真正的 EOF None 区分
    FRAME_TIMEOUT = object()

    def __init__(self, input_path, ffmpeg_bin='ffmpeg', prefetch_factor=16, use_hwaccel=True):
        self.input_path = input_path
        self.ffmpeg_bin = ffmpeg_bin
        self.prefetch_factor = prefetch_factor
        self.use_hwaccel = use_hwaccel

        # 获取视频元数据
        meta = get_video_meta_info(input_path)
        self.width = meta['width']
        self.height = meta['height']
        self.fps = meta['fps']
        self.nb_frames = meta['nb_frames']
        self.audio = meta['audio']

        # 构建输入流时注入 hwaccel 参数
        input_kwargs = {}
        if self.use_hwaccel:
            input_kwargs['hwaccel'] = 'auto'  # 自动选择最佳硬件解码器 (cuda, qsv, dxva2等)

        # 创建FFmpeg输入流
        self._ffmpeg_input = ffmpeg.input(input_path, **input_kwargs)

        # 帧队列
        self._frame_queue = queue.Queue(maxsize=prefetch_factor)
        self._running = True

        # FIX-READER-DIAG: 诊断 / 状态
        self._stderr_lines: List[str] = []
        self._stderr_lock = threading.Lock()
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._eof_sent = False
        self._frames_produced = 0
        self._last_error: Optional[str] = None

        # 启动读取线程
        self._thread = threading.Thread(target=self._read_loop, daemon=True,
                                        name='ffmpeg_reader_loop')
        self._thread.start()

    # ----------------------------------------------------------------------
    # FIX-READER-DIAG: 内部 stderr drain 线程，防止 ffmpeg 因 stderr pipe 满阻塞
    # ----------------------------------------------------------------------
    def _drain_stderr(self, process):
        try:
            while True:
                line = process.stderr.readline()
                if not line:
                    break
                try:
                    text = line.decode('utf-8', errors='replace')
                except Exception:
                    text = str(line)
                with self._stderr_lock:
                    self._stderr_lines.append(text)
                    # 限长，避免无限增长
                    if len(self._stderr_lines) > 400:
                        del self._stderr_lines[:200]
        except Exception:
            pass

    def _get_stderr_tail(self, n: int = 30) -> str:
        """FIX-READER-DIAG: 返回最后 n 行 ffmpeg stderr 输出"""
        with self._stderr_lock:
            if not self._stderr_lines:
                return '(无 stderr 输出)'
            lines = self._stderr_lines[-n:]
        return ''.join(lines)

    # ----------------------------------------------------------------------
    # FIX-READER-DIAG: 保证 EOF 一定送到下游（修死锁的关键）
    # ----------------------------------------------------------------------
    def _send_eof_guaranteed(self, max_wait_s: float = 600.0):
        """
        向下游队列发送 None 作为 EOF 标记。
        无论下游是否消费，都要尽最大努力送达；若 pipeline 已被关闭则退出。
        """
        if self._eof_sent:
            return
        retries = 0
        while True:
            try:
                self._frame_queue.put(None, timeout=1.0)
                self._eof_sent = True
                print(f"[FFmpegReader] EOF 已发送到下游（frames_produced={self._frames_produced}）",
                      flush=True)
                return
            except queue.Full:
                retries += 1
                # pipeline 主动关闭，不再坚持
                if not self._running:
                    print(f"[FFmpegReader] 放弃发送 EOF：_running=False",
                          flush=True)
                    return
                if retries % 10 == 0:
                    print(f"[FFmpegReader] EOF 发送受阻 {retries}s，"
                          f"下游队列持续满（frame_queue={self._frame_queue.qsize()}/"
                          f"{self._frame_queue.maxsize}）", flush=True)
                if retries >= max_wait_s:
                    print(f"[FFmpegReader] 放弃发送 EOF：超过 {max_wait_s:.0f}s 仍无法入队",
                          flush=True)
                    return
            except Exception as e:
                print(f"[FFmpegReader] 发送 EOF 异常: {e}", flush=True)
                return

    # ----------------------------------------------------------------------
    # 主读取循环（FIX-READER-DIAG 增强版）
    # ----------------------------------------------------------------------
    def _read_loop(self):
        """后台读取帧的线程"""
        process = None
        stderr_thread = None

        try:
            # 设置FFmpeg输出格式
            # 注意：如果开启了 hwaccel，FFmpeg 会尝试在硬件层解码并传回到系统内存
            process = (
                self._ffmpeg_input
                .output(
                    'pipe:',
                    format='rawvideo',
                    pix_fmt='rgb24',
                    an=None,                               # 不处理音频，顺便消除 [mp3float] Header missing 警告
                    **{'fps_mode': 'passthrough'},         # 替代已弃用的 vsync=0，消除 -vsync deprecated 警告
                )
                .global_args('-hide_banner', '-loglevel', 'error', '-nostats')  # 屏蔽 banner / info 行 / 进度行
                .run_async(pipe_stdout=True, pipe_stderr=True, quiet=False)
            )
            self._ffmpeg_process = process

            # FIX-READER-DIAG: 启动 stderr drain 线程
            stderr_thread = threading.Thread(
                target=self._drain_stderr, args=(process,),
                daemon=True, name='ffmpeg_reader_stderr')
            stderr_thread.start()

            frame_size = self.width * self.height * 3
            print(f"[FFmpegReader] FFmpeg 进程启动（pid={process.pid}, "
                  f"frame_size={frame_size}B, queue_size={self.prefetch_factor}）",
                  flush=True)

            while self._running:
                # 读一帧
                try:
                    in_bytes = process.stdout.read(frame_size)
                except Exception as e:
                    self._last_error = f'stdout.read 异常: {e}'
                    print(f"[FFmpegReader] 致命: stdout.read 异常 "
                          f"@frame={self._frames_produced}: {e}", flush=True)
                    break

                # FIX-READER-DIAG: 正常 EOF 诊断
                if not in_bytes:
                    rc = process.poll()
                    print(f"[FFmpegReader] stdout EOF @frame={self._frames_produced}, "
                          f"ffmpeg_rc={rc}", flush=True)
                    if rc is not None and rc != 0:
                        # 异常退出：打诊断
                        tail = self._get_stderr_tail(40)
                        print(f"[FFmpegReader] ffmpeg 非零退出 (rc={rc})，stderr 尾部:\n{tail}",
                              flush=True)
                        self._last_error = f'ffmpeg 非零退出 rc={rc}'
                    break

                # FIX-READER-DIAG: 短读 — ffmpeg 多半已异常退出
                if len(in_bytes) != frame_size:
                    rc = process.poll()
                    tail = self._get_stderr_tail(40)
                    print(f"[FFmpegReader] 致命: 短读! 期望={frame_size} "
                          f"实际={len(in_bytes)} @frame={self._frames_produced} "
                          f"ffmpeg_rc={rc}", flush=True)
                    print(f"[FFmpegReader] ffmpeg stderr 尾部:\n{tail}",
                          flush=True)
                    self._last_error = (f'短读 got={len(in_bytes)} '
                                        f'want={frame_size} rc={rc}')
                    break

                # reshape
                try:
                    frame = np.frombuffer(in_bytes, np.uint8).reshape(
                        [self.height, self.width, 3])
                except Exception as e:
                    print(f"[FFmpegReader] reshape 失败 "
                          f"@frame={self._frames_produced}: {e}", flush=True)
                    self._last_error = f'reshape 失败: {e}'
                    break

                # FIX-READER-DIAG: 入队（阻塞，但若 ffmpeg 已退出则主动终止，避免死守）
                while self._running:
                    try:
                        self._frame_queue.put(frame, timeout=1.0)
                        self._frames_produced += 1
                        break
                    except queue.Full:
                        if process.poll() is not None:
                            rc = process.returncode
                            tail = self._get_stderr_tail(20)
                            print(f"[FFmpegReader] ffmpeg 退出但下游满 "
                                  f"@frame={self._frames_produced} "
                                  f"rc={rc}, stderr:\n{tail}", flush=True)
                            self._last_error = f'下游满时 ffmpeg 退出 rc={rc}'
                            raise RuntimeError('ffmpeg exited while queue full')
                        continue
                else:
                    # self._running 变为 False
                    break

        except Exception as e:
            print(f"[FFmpegReader] _read_loop 异常: {e}", flush=True)
            traceback.print_exc()
            self._last_error = f'_read_loop 异常: {e}'

        finally:
            # 1) 终止 ffmpeg 进程
            if process is not None:
                try:
                    if process.poll() is None:
                        try:
                            if process.stdout and not process.stdout.closed:
                                process.stdout.close()
                        except Exception:
                            pass
                        try:
                            process.terminate()
                        except Exception:
                            pass
                        try:
                            process.wait(timeout=3.0)
                        except subprocess.TimeoutExpired:
                            try:
                                process.kill()
                                process.wait(timeout=2.0)
                            except Exception:
                                pass
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[FFmpegReader] 终止 ffmpeg 异常: {e}", flush=True)

            # 2) 打印最终 stderr：仅在 ffmpeg 异常退出或 Reader 线程捕获到错误时才打印
            rc_final = None
            if process is not None:
                try:
                    rc_final = process.poll()
                    if rc_final is None:
                        # 走到这里时 ffmpeg 通常已经被上面的 terminate/kill 收尾过，
                        # 再尝试等一小段时间拿到真正的 returncode
                        try:
                            rc_final = process.wait(timeout=1.0)
                        except Exception:
                            rc_final = process.returncode
                except Exception:
                    rc_final = None

            abnormal = (
                (rc_final is not None and rc_final != 0)   # ffmpeg 非零退出
                or self._last_error is not None            # Reader 线程自身记录过错误
            )

            if abnormal:
                tail = self._get_stderr_tail(40)
                print(f"[FFmpegReader] ffmpeg 异常退出 rc={rc_final}, "
                      f"last_error={self._last_error}", flush=True)
                if tail and tail != '(无 stderr 输出)':
                    print(f"[FFmpegReader] ffmpeg stderr:\n{tail}", flush=True)
            else:
                # 正常结束：只留一行摘要，不再刷屏
                print(f"[FFmpegReader] ffmpeg 正常结束 rc={rc_final}, "
                      f"frames_produced={self._frames_produced}", flush=True)

            # 3) FIX-READER-DIAG: 关键：无论如何把 EOF 送到下游
            self._send_eof_guaranteed()

            # 4) 等 stderr drain 线程结束（非强制）
            if stderr_thread is not None and stderr_thread.is_alive():
                stderr_thread.join(timeout=1.0)

            print(f"[FFmpegReader] _read_loop 退出 "
                  f"(frames_produced={self._frames_produced}, "
                  f"last_error={self._last_error})", flush=True)

    # ----------------------------------------------------------------------
    # 消费者接口
    # ----------------------------------------------------------------------
    def get_frame(self):
        """获取一帧。返回 None=EOF，返回 FRAME_TIMEOUT=队列暂时为空（继续重试）"""
        try:
            return self._frame_queue.get(timeout=2.0)
        except queue.Empty:
            return FFmpegReader.FRAME_TIMEOUT  # FIX-PREMATURE-EOF

    # ── FIX-READER-DIAG: 探活接口 ─────────────────────────────────────
    def is_reader_alive(self) -> bool:
        """供消费者探活：reader 线程仍在读帧吗？"""
        return self._thread.is_alive()

    def is_eof_sent(self) -> bool:
        """EOF 是否已经送到下游队列？"""
        return self._eof_sent

    def get_queue_size(self) -> int:
        """prefetch 队列当前帧数（供监控使用）"""
        try:
            return self._frame_queue.qsize()
        except Exception:
            return -1

    def get_queue_capacity(self) -> int:
        """prefetch 队列容量"""
        return self.prefetch_factor

    def get_frames_produced(self) -> int:
        """已从 ffmpeg 读取并入队的总帧数"""
        return self._frames_produced

    def close(self):
        """关闭读取器"""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)  # 允许 read_loop 处理完当前帧


class FFmpegWriter:
    """通过FFmpeg pipe写入视频帧"""

    # 写入超时时间（秒）
    # FIX-WRITE-TIMEOUT-180S:
    #   原 30s 在 2560×1440 高分辨率下不足：libx264/h264_nvenc 遇到复杂场景（I帧/
    #   场景切换）时编码器内部队列短暂饱和，导致 FFmpeg stdin pipe buffer（64KB）
    #   填满，select() 持续返回 not-ready 超过 30s → 误判管道断裂。
    #   改为 180s，并在循环内部增加 FFmpeg 存活检测，真正崩溃时仍能在 ≤1s 内发现。
    WRITE_TIMEOUT = 300.0
    # 线程终止等待时间
    THREAD_JOIN_TIMEOUT = 5.0
    # FFmpeg 进程终止等待时间
    PROCESS_TERMINATE_TIMEOUT = 10.0
    # 每次写入的块大小（64KB，匹配 Linux pipe buffer）
    WRITE_CHUNK_SIZE = 65536

    def __init__(self, args, audio, height, width, output_path, fps):
        self.args = args
        self.audio = audio
        self.height = height
        self.width = width
        self.output_path = output_path
        self.fps = fps

        # FIX-FRAME-QUEUE-SIZE: 从256缩小到64，匹配实际流水线深度
        # 防止长时间帧积压掩盖nvenc stall，让问题更早暴露
        self._frame_queue = queue.Queue(maxsize=64)
        self._running = True
        self._broken = False   # 管道断裂标志
        self._write_error = None  # 记录写入错误
        self._stderr_buffer = []  # 缓存 FFmpeg stderr 输出
        # FIX-DIAG: 记录写入统计
        self._frames_written_to_pipe = 0
        self._bytes_written_to_pipe = 0

        self._process = None
        self._init_ffmpeg_process()

        # 启动 stderr 后台读取线程，防止 stderr pipe 满导致 FFmpeg 阻塞
        self._stderr_thread = threading.Thread(
            target=self._stderr_reader_loop, daemon=True,
            name='ffmpeg_stderr_reader')
        self._stderr_thread.start()

        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def _stderr_reader_loop(self):
        """后台线程持续读取 FFmpeg stderr，防止 pipe buffer 满导致 FFmpeg 阻塞

        FIX-STDERR-BLOCK:
          原实现 read(4096) 调用底层 os.read() 多次直到凑齐 4096 字节才返回。
          FFmpeg banner ~2KB → 第一次 os.read() 读 2KB → 还需 2KB → 再次 os.read()
          → 永久阻塞（FFmpeg 不再主动写 stderr）→ _stderr_buffer 始终为空 →
          诊断信息全部丢失，且 FFmpeg 一旦写 stderr 时管道积压导致 FFmpeg 本身阻塞。
          修复：改用 readline()，有换行符即返回，非阻塞地持续消费 stderr。
        """
        try:
            if not self._process or not self._process.stderr:
                return
            while True:
                line = self._process.stderr.readline()  # 按行读，有数据就返回
                if not line:
                    break
                text = line.decode('utf-8', errors='replace')
                self._stderr_buffer.append(text)
        except Exception:
            pass

    def _get_ffmpeg_stderr(self, tail_lines=10):
        """获取缓存的 FFmpeg stderr 输出（最后 N 行）"""
        if not self._stderr_buffer:
            return '(无 stderr 输出)'
        full = ''.join(self._stderr_buffer)
        lines = full.strip().split('\n')
        if len(lines) > tail_lines:
            lines = lines[-tail_lines:]
        return '\n'.join(lines)

    @staticmethod
    def _check_nvenc_available(ffmpeg_bin='ffmpeg', codec='h264_nvenc') -> bool:
        """预检测 NVENC 编码器是否可用。
        
        使用 nullsrc 生成 1 帧测试视频，尝试用指定 NVENC 编码器编码。
        成功 → True；失败（unsupported device / no capable devices）→ False。
        整个过程 <1s，不写磁盘。
        """
        test_cmd = [
            ffmpeg_bin, '-y',
            '-f', 'lavfi', '-i', 'nullsrc=s=64x64:d=0.04:r=25',  # 1 帧
            '-frames:v', '1',
            '-c:v', codec,
            '-f', 'null', '-',
        ]
        try:
            result = subprocess.run(
                test_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=10,
            )
            stderr_text = result.stderr.decode('utf-8', errors='replace').lower()
            if result.returncode != 0:
                # 检查是否是 NVENC 特有错误
                nvenc_errors = [
                    'openencodesessionex failed',
                    'no capable devices found',
                    'unsupported device',
                    'cannot load libnvidia-encode',
                    'nvenc',
                ]
                is_nvenc_error = any(kw in stderr_text for kw in nvenc_errors)
                if is_nvenc_error:
                    print(f'[FFmpegWriter] NVENC 预检测失败: {codec} 不可用', flush=True)
                    # 打印关键 stderr 行
                    for line in stderr_text.split('\n'):
                        line = line.strip()
                        if any(kw in line for kw in nvenc_errors):
                            print(f'[FFmpegWriter]   {line}', flush=True)
                    return False
                else:
                    # 非 NVENC 错误（可能是 lavfi 问题），保守地认为可用
                    print(f'[FFmpegWriter] NVENC 预检测返回 rc={result.returncode}，'
                          f'但非 NVENC 特有错误，继续使用', flush=True)
                    return True
            return True
        except subprocess.TimeoutExpired:
            print(f'[FFmpegWriter] NVENC 预检测超时（>10s），假定可用', flush=True)
            return True
        except FileNotFoundError:
            print(f'[FFmpegWriter] FFmpeg 未找到: {ffmpeg_bin}', flush=True)
            return True  # 让后续正式启动时报错
        except Exception as e:
            print(f'[FFmpegWriter] NVENC 预检测异常: {e}，假定可用', flush=True)
            return True

    def _init_ffmpeg_process(self):
        """初始化FFmpeg进程 - FIX-MANUAL-CMD: 手动构造命令行，绕过 ffmpeg-python 参数排序问题"""

        # 获取用户指定的编码器
        video_codec = getattr(self.args, 'video_codec', 'libx264')
        crf = getattr(self.args, 'crf', 23)
        x264_preset = getattr(self.args, 'x264_preset', 'medium')

        # ── FIX-NVENC-PRECHECK: NVENC 预检测 + 自动降级 ──────────────
        if video_codec in ('h264_nvenc', 'hevc_nvenc'):
            _ffbin = getattr(self.args, 'ffmpeg_bin', 'ffmpeg')
            print(f'[FFmpegWriter] 预检测 {video_codec} 可用性...', flush=True)
            if not self._check_nvenc_available(_ffbin, video_codec):
                _fallback = 'libx264' if 'h264' in video_codec else 'libx265'
                print(f'[FFmpegWriter] {video_codec} 不可用，自动降级到 {_fallback}',
                      flush=True)
                video_codec = _fallback
                self.args.video_codec = _fallback
            else:
                print(f'[FFmpegWriter] {video_codec} 预检测通过', flush=True)

        # FIX-AUDIO-SEPARATE: 始终写无音轨临时文件，close() 后单独 mux 音轨
        _base, _ext = os.path.splitext(self.output_path)
        self._tmp_video_path = f'{_base}.tmp_novid{_ext}'
        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        tmp_dir = os.path.dirname(self._tmp_video_path)
        if tmp_dir:
            os.makedirs(tmp_dir, exist_ok=True)

        # ── 手动构造命令行 ────────────────────────────────────────────
        cmd_args = [
            getattr(self.args, 'ffmpeg_bin', 'ffmpeg'),
            '-y',
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', 'pipe:',
        ]

        if video_codec in ('h264_nvenc', 'hevc_nvenc'):
            cmd_args += [
                '-vcodec', video_codec,
                '-pix_fmt', 'yuv420p',
                '-preset', 'p4',
                '-cq', str(crf),
                '-surfaces', '4',
                '-delay', '0',
                '-rc-lookahead', '0',
                '-bf', '0',
            ]
        elif video_codec == 'libx265':
            cmd_args += [
                '-vcodec', video_codec,
                '-pix_fmt', 'yuv420p',
                '-crf', str(crf),
                '-preset', x264_preset,
            ]
        else:
            cmd_args += [
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', str(crf),
                '-preset', x264_preset,
            ]

        cmd_args += ['-an']
        cmd_args += [self._tmp_video_path]

        print(f"[FFmpegWriter] 命令: {' '.join(cmd_args)}", flush=True)

        self._process = subprocess.Popen(
            cmd_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # FIX-PIPE-SZ: 尝试扩大管道缓冲区
        try:
            import fcntl
            fcntl.fcntl(self._process.stdin.fileno(), fcntl.F_SETPIPE_SZ, 4 * 1024 * 1024)
        except PermissionError:
            pass
        except Exception as _e:
            print(f"[FFmpegWriter] 管道缓冲区扩大失败（使用默认 64KB）: {_e}", flush=True)
            pass

        time.sleep(0.5)
        if self._process.poll() is not None:
            stderr_text = self._get_ffmpeg_stderr(tail_lines=20)
            print(f"[FFmpegWriter] FFmpeg 启动失败! "
                  f"(returncode={self._process.returncode})", flush=True)
            print(f"[FFmpegWriter] 命令: {' '.join(cmd_args)}", flush=True)
            for line in stderr_text.split('\n'):
                print(f"[FFmpegWriter]   {line}", flush=True)
            raise RuntimeError(
                f"FFmpeg 启动失败 (rc={self._process.returncode}): {stderr_text}")
        else:
            print(f"[FFmpegWriter] FFmpeg 进程已启动 "
                  f"(pid={self._process.pid}, codec={video_codec})", flush=True)

    def _write_with_timeout(self, data: bytes, timeout: float = WRITE_TIMEOUT) -> bool:
        """带超时的写入操作 — 分块写入版（增加诊断日志）"""
        if not self._process or not self._process.stdin:
            self._write_error = 'process 或 stdin 不可用'
            return False

        if self._process.poll() is not None:
            rc = self._process.returncode
            self._write_error = f'FFmpeg 已退出 (rc={rc})'
            return False

        try:
            fd = self._process.stdin.fileno()
            deadline = time.time() + timeout
            offset = 0
            total = len(data)
            _diag_frame = self._frames_written_to_pipe < 5

            while offset < total:
                now = time.time()
                if now >= deadline:
                    self._write_error = (
                        f'写入超时 ({offset}/{total} 字节, '
                        f'frame_size={total})')
                    return False

                remaining = deadline - now
                try:
                    _rready, wready, xready = select.select(
                        [], [fd], [fd], min(remaining, 1.0))
                except (OSError, ValueError):
                    self._write_error = 'select() 调用失败 (fd 可能已关闭)'
                    return False

                if xready:
                    time.sleep(0.1)
                    if self._process.poll() is not None:
                        rc = self._process.returncode
                        self._write_error = f'FFmpeg 异常退出 (rc={rc})'
                    else:
                        self._write_error = (
                            f'select 异常条件 (已写 {offset}/{total} 字节)')
                    return False

                if not wready:
                    if self._process.poll() is not None:
                        rc = self._process.returncode
                        stderr_text = self._get_ffmpeg_stderr(tail_lines=5)
                        self._write_error = (
                            f'FFmpeg 进程已退出 (rc={rc}, 已写 {offset}/{total} 字节)'
                            f'\n{stderr_text}')
                        return False
                    if _diag_frame and offset == 0:
                        print(f'[FFmpegWriter][DIAG] 帧{self._frames_written_to_pipe}: '
                              f'select 返回 not-ready (offset={offset}/{total}), '
                              f'FFmpeg alive={self._process.poll() is None}, '
                              f'stderr_lines={len(self._stderr_buffer)}',
                              flush=True)
                        _diag_frame = False
                    continue

                chunk = data[offset:offset + self.WRITE_CHUNK_SIZE]
                try:
                    n = os.write(fd, chunk)
                    if n == 0:
                        self._write_error = 'os.write() 返回 0 (fd 已关闭?)'
                        return False
                    offset += n
                    self._bytes_written_to_pipe += n
                except OSError as e:
                    if self._process.poll() is not None:
                        rc = self._process.returncode
                        self._write_error = (
                            f'FFmpeg 退出 (rc={rc}), OSError: {e}')
                    else:
                        self._write_error = f'OSError: {e}'
                    return False

            self._frames_written_to_pipe += 1
            return True

        except (BrokenPipeError, OSError) as e:
            self._write_error = str(e)
            return False
        except Exception as e:
            self._write_error = str(e)
            return False

    def _write_loop(self):
        """后台写入帧的线程 - 增强版 v3（FIX-NVENC-STALL）"""
        consecutive_errors = 0
        max_consecutive_errors = 3
        _vc = getattr(self.args, 'video_codec', 'libx264')
        if _vc in ('h264_nvenc', 'hevc_nvenc'):
            MAX_NVENC_STALL_S = 600.0
            NVENC_RETRY_SLEEP = 20.0
            SINGLE_WRITE_TIMEOUT = 300.0
        else:
            MAX_NVENC_STALL_S = 180.0
            NVENC_RETRY_SLEEP = 5.0
            SINGLE_WRITE_TIMEOUT = 60.0

        while self._running:
            try:
                frame = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                if self._process and self._process.poll() is not None:
                    rc = self._process.returncode
                    stderr_text = self._get_ffmpeg_stderr(tail_lines=10)
                    print(f"[FFmpegWriter] FFmpeg 进程意外退出 "
                          f"(returncode={rc})", flush=True)
                    print(f"[FFmpegWriter] 最后的 stderr:\n{stderr_text}", flush=True)
                    self._broken = True
                    break
                continue

            if frame is None:
                break

            try:
                if self._process and self._process.stdin:
                    frame_bytes = frame.tobytes()

                    _expected_bytes = self.width * self.height * 3
                    if len(frame_bytes) != _expected_bytes:
                        print(
                            f'[FFmpegWriter] [致命诊断] 帧尺寸错误: '
                            f'实际={len(frame_bytes)}B '
                            f'期望={_expected_bytes}B '
                            f'(shape={frame.shape}, '
                            f'writer_wh={self.width}x{self.height})',
                            flush=True,
                        )
                        self._broken = True
                        return

                    stall_elapsed = 0.0
                    wrote_ok = False
                    while True:
                        if not self._running:
                            return

                        if self._write_with_timeout(frame_bytes, timeout=SINGLE_WRITE_TIMEOUT):
                            consecutive_errors = 0
                            wrote_ok = True

                            if self._frames_written_to_pipe <= 5:
                                _early_stderr = ''.join(self._stderr_buffer).lower()
                                _nvenc_dead = any(kw in _early_stderr for kw in (
                                    'openencodesessionex failed',
                                    'no capable devices found',
                                    'error while opening encoder',
                                ))
                                if _nvenc_dead:
                                    print(f'[FFmpegWriter] [首帧 stderr 检测] 编码器初始化失败，'
                                          f'标记管道断裂', flush=True)
                                    _full_stderr = self._get_ffmpeg_stderr(tail_lines=10)
                                    print(f'[FFmpegWriter] FFmpeg stderr:\n{_full_stderr}',
                                          flush=True)
                                    self._broken = True
                                    return

                            break

                        err_detail = self._write_error or '(未知错误)'
                        ffmpeg_alive = (self._process.poll() is None)

                        if not ffmpeg_alive:
                            consecutive_errors += 1
                            print(f"[FFmpegWriter] 写入失败（FFmpeg 已退出）"
                                  f" ({consecutive_errors}/{max_consecutive_errors})"
                                  f" | {err_detail}", flush=True)
                            if consecutive_errors >= max_consecutive_errors:
                                stderr_text = self._get_ffmpeg_stderr(tail_lines=20)
                                if stderr_text:
                                    print(f"[FFmpegWriter] FFmpeg stderr (共{len(self._stderr_buffer)}行):\n{stderr_text}",
                                      flush=True)
                                    print("[FFmpegWriter] FFmpeg 已退出，标记管道断裂", flush=True)

                                self._broken = True
                                return
                            break

                        stall_elapsed += SINGLE_WRITE_TIMEOUT
                        if stall_elapsed >= MAX_NVENC_STALL_S:
                            if _vc in ('h264_nvenc', 'hevc_nvenc'):
                                _suggestion = (f'建议改用 --video-codec libx264 避免 '
                                               f'SR+GFPGAN 与 {_vc} 的 GPU 资源竞争。')
                            else:
                                _suggestion = (f'编码器 {_vc} stdin 阻塞超过 '
                                               f'{MAX_NVENC_STALL_S:.0f}s，'
                                               f'请检查磁盘空间或增大 --crf 降低码率。')
                            print(f'[FFmpegWriter] stall 超过 {MAX_NVENC_STALL_S:.0f}s，'
                                  f'放弃写入。{_suggestion}',
                                  flush=True)
                            stderr_text = self._get_ffmpeg_stderr(tail_lines=20)
                            if stderr_text:
                                print(f'[FFmpegWriter] FFmpeg stderr (共{len(self._stderr_buffer)}行):\n{stderr_text}',
                                  flush=True)

                            self._broken = True
                            return

                        if stall_elapsed <= self.WRITE_TIMEOUT + 0.1:
                            _early_stderr = self._get_ffmpeg_stderr(tail_lines=5)
                            if _early_stderr:
                                print(f'[FFmpegWriter] stall 首次发生，FFmpeg stderr:\n{_early_stderr}',
                                      flush=True)
                        print(f'[FFmpegWriter] stdin 暂时阻塞（已等待 {stall_elapsed:.0f}s'
                              f' / {MAX_NVENC_STALL_S:.0f}s），'
                              f'{NVENC_RETRY_SLEEP:.0f}s 后重试同一帧...',
                              flush=True)
                        time.sleep(NVENC_RETRY_SLEEP)

            except (BrokenPipeError, OSError) as e:
                print(f"[FFmpegWriter] 管道断裂: {e}", flush=True)
                self._broken = True
                break
            except Exception as e:
                print(f"[FFmpegWriter] 写入错误: {e}", flush=True)
                self._broken = True
                break
    
    def write_frame(self, frame):
        """写入一帧"""
        if not self._running or self._broken:
            return False

        _vc = getattr(self.args, 'video_codec', 'libx264')
        if _vc in ('h264_nvenc', 'hevc_nvenc'):
            _total_timeout = 660.0
        else:
            _total_timeout = 240.0

        deadline = time.time() + _total_timeout

        while True:
            if self._broken:
                return False
            remaining = deadline - time.time()
            if remaining <= 0:
                print(f"[FFmpegWriter] 警告: 帧队列已满超时 ({_total_timeout:.0f}s)，"
                      f"标记管道断裂 (codec={_vc})", flush=True)
                print(f"[FFmpegWriter][DIAG] 管道写入统计: "
                      f"帧={self._frames_written_to_pipe}, "
                      f"字节={self._bytes_written_to_pipe/1024/1024:.1f}MB, "
                      f"stderr_lines={len(self._stderr_buffer)}", flush=True)
                print(f"[FFmpegWriter][DIAG] FFmpeg stderr:\n{self._get_ffmpeg_stderr(tail_lines=20)}", flush=True)
                self._broken = True
                return False
            try:
                self._frame_queue.put(frame, timeout=min(1.0, remaining))
                return True
            except queue.Full:
                continue

    def _has_video_stream(self, filepath: str) -> bool:
        """使用 ffmpeg 探测文件是否包含视频流（不依赖 ffprobe）"""
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return False
        ffmpeg_bin = getattr(self.args, 'ffmpeg_bin', 'ffmpeg')
        cmd = [ffmpeg_bin, '-i', filepath]
        try:
            # ffmpeg 会将元信息输出到 stderr，我们需要捕获它
            result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE,
                                    text=True, timeout=10)
            stderr = result.stderr
            # 查找视频流标识：例如 "Stream #0:0(und): Video: h264"
            if re.search(r'Stream\s+#\d+:\d+.*Video:', stderr, re.IGNORECASE):
                return True
            return False
        except subprocess.TimeoutExpired:
            print(f'[FFmpegWriter] 检查视频流超时: {filepath}', flush=True)
            return False
        except Exception as e:
            print(f'[FFmpegWriter] 检查视频流异常: {e}', flush=True)
            return False

    def close(self):
        """关闭写入器 - 增强版"""
        print("[FFmpegWriter] 阶段1/5: 等待写入线程完成...", flush=True)

        if not self._broken:
            for _ in range(3):
                try:
                    self._frame_queue.put(None, timeout=2.0)
                except queue.Full:
                    print("[FFmpegWriter] 警告: 队列已满，无法发送结束信号", flush=True)
                    self._broken = True
                    break

        if self._thread.is_alive():
            self._thread.join(timeout=self.THREAD_JOIN_TIMEOUT)

        if self._thread.is_alive():
            print("[FFmpegWriter] 警告: 写入线程未响应，强制终止 FFmpeg 进程...", flush=True)

            if self._process and self._process.poll() is None:
                try:
                    if self._process.stdin and not self._process.stdin.closed:
                        self._process.stdin.close()
                except Exception:
                    pass

                try:
                    self._process.terminate()
                    self._process.wait(timeout=self.PROCESS_TERMINATE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    print("[FFmpegWriter] SIGTERM 超时，发送 SIGKILL...", flush=True)
                    try:
                        self._process.kill()
                        self._process.wait(timeout=5)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[FFmpegWriter] 终止 FFmpeg 进程异常: {e}", flush=True)

            self._thread.join(timeout=2.0)

            if self._thread.is_alive():
                print("[FFmpegWriter] 错误: 写入线程仍无法停止，标记为 daemon", flush=True)
                if self._process and self._process.poll() is None:
                    try:
                        self._process.kill()
                    except Exception:
                        pass

        print("[FFmpegWriter] 阶段2/5: 写入线程已结束", flush=True)
        self._running = False

        if self._process and self._process.poll() is None:
            print("[FFmpegWriter] 阶段3/5: 等待 FFmpeg 完成编码...", flush=True)

            if self._process.stdin and not self._process.stdin.closed:
                try:
                    self._process.stdin.flush()
                    self._process.stdin.close()
                except Exception:
                    pass

            try:
                self._process.wait(timeout=300)
                print("[FFmpegWriter] 阶段4/5: FFmpeg 编码完成", flush=True)
            except subprocess.TimeoutExpired:
                print("[FFmpegWriter] FFmpeg 编码超时（>300s），强制终止", flush=True)
                try:
                    self._process.kill()
                    self._process.wait(timeout=10)
                except Exception:
                    pass
        else:
            print("[FFmpegWriter] 阶段3-4/5: FFmpeg 进程已终止", flush=True)

        print("[FFmpegWriter] 阶段5/5: 清理完成", flush=True)

        # FIX-AUDIO-SEPARATE: 视频写入完成后，单独 mux 音轨
        _tmp = getattr(self, '_tmp_video_path', None)
        if _tmp and not self._broken and self.audio is not None:
            _src = getattr(self.args, 'input', None)
            if _src and os.path.exists(_tmp) and os.path.exists(_src):
                # 增加临时文件有效性检查：确保临时文件包含视频流
                if not self._has_video_stream(_tmp):
                    print(f'[FFmpegWriter] 警告: 临时文件 {_tmp} 不包含视频流，跳过音轨合并', flush=True)
                    # 若临时文件无效，直接将其重命名为输出（可能无视频，但保留现场供调试）
                    try:
                        os.rename(_tmp, self.output_path)
                        print(f'[FFmpegWriter] 无有效视频流，已保留临时文件为 {self.output_path}', flush=True)
                    except Exception as _e:
                        print(f'[FFmpegWriter] 重命名失败: {_e}', flush=True)
                    _tmp = None
                else:
                    print(f'[FFmpegWriter] 合并音轨: {_tmp} + {_src} → {self.output_path}',
                          flush=True)
                    _mux_cmd = [
                        getattr(self.args, 'ffmpeg_bin', 'ffmpeg'), '-y',
                        '-i', _tmp,
                        '-i', _src,
                        '-map', '0:v:0',
                        '-map', '1:a:0',
                        '-c:v', 'copy',
                        '-c:a', 'aac',
                        '-shortest',
                        self.output_path,
                    ]
                    try:
                        _r = subprocess.run(
                            _mux_cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            timeout=300,
                        )
                        if _r.returncode == 0:
                            print(f'[FFmpegWriter] 音轨合并成功: {self.output_path}', flush=True)
                        else:
                            _mux_err = _r.stderr.decode('utf-8', errors='replace')[-500:]
                            print(f'[FFmpegWriter] 音轨合并失败 (rc={_r.returncode}): {_mux_err}',
                                  flush=True)
                            print(f'[FFmpegWriter] 保留无音轨文件: {_tmp}', flush=True)
                            _tmp = None
                    except subprocess.TimeoutExpired:
                        print('[FFmpegWriter] 音轨合并超时（>300s）', flush=True)
                        _tmp = None
                    except Exception as _e:
                        print(f'[FFmpegWriter] 音轨合并异常: {_e}', flush=True)
                        _tmp = None
            elif _src is None:
                try:
                    os.rename(_tmp, self.output_path)
                    _tmp = None
                    print(f'[FFmpegWriter] 无音轨模式: 已输出 {self.output_path}', flush=True)
                except Exception as _e:
                    print(f'[FFmpegWriter] rename 失败: {_e}', flush=True)
                    _tmp = None
        elif _tmp and (self._broken or self.audio is None):
            if os.path.exists(_tmp):
                try:
                    os.rename(_tmp, self.output_path)
                    print(f'[FFmpegWriter] 无音轨输出: {self.output_path}', flush=True)
                except Exception as _e:
                    print(f'[FFmpegWriter] rename 失败: {_e}', flush=True)
                _tmp = None

        if _tmp and os.path.exists(_tmp):
            try:
                os.remove(_tmp)
            except Exception:
                pass

def _make_detect_helper(face_enhancer, device):
    """
    创建独立的 FaceRestoreHelper 实例，专供后台 detect 线程使用。
    与主线程 face_enhancer.face_helper 互为独立对象，无共享状态，线程安全。

    FIX-DET-CPU (优化4): 强制在 CPU 上运行人脸检测（retinaface_resnet50）。
    原因：人脸检测与 SR/GFPGAN 共享 GPU，导致互相抢占，GPU 利用率出现谷值。
    CPU 上 retinaface 对 640px resize 的推理约 30-50ms/帧，远快于 SR 的 ~300ms/帧，
    不会成为瓶颈，但释放了 GPU 带宽给 SR 和 GFPGAN。
    """
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    upscale_factor = getattr(face_enhancer.face_helper, 'upscale_factor', 1)
    return FaceRestoreHelper(
        upscale_factor = upscale_factor,
        face_size      = 512,
        crop_ratio     = (1, 1),
        det_model      = 'retinaface_resnet50',
        save_ext       = 'png',
        use_parse      = True,
        device         = torch.device('cpu'),  # FIX-DET-CPU: 强制 CPU
    )


def _detect_faces_batch(frames: List[np.ndarray], helper,
                         det_threshold: float = 0.5) -> Tuple[List[dict], int, int, int]:
    """
    在原始低分辨率帧上检测人脸，返回序列化检测结果。
    FIX-DET-THRESHOLD + FIX-PRECOMPUTE-INV-AFFINE (优化6): 在检测阶段预计算逆仿射矩阵，
    避免 _paste_faces_batch 中重复调用 get_inverse_affine()。
    对于人脸密集场景（每帧 5+ 个人脸），节省 ~10-20ms/帧的 CPU 开销。
    """
    import cv2 as _cv2
    face_data = []
    _total_filtered = 0   # FIX-DET-THRESHOLD: 被阈值过滤的人脸总数
    for orig_frame in frames:
        helper.clean_all()
        bgr_frame = _cv2.cvtColor(orig_frame, _cv2.COLOR_RGB2BGR)
        helper.read_image(bgr_frame)
        try:
            helper.get_face_landmarks_5(
                only_center_face=False, resize=640, eye_dist_threshold=5)
        except TypeError:
            helper.get_face_landmarks_5(
                only_center_face=False, eye_dist_threshold=5)

        # FIX-DET-THRESHOLD: 按置信度阈值过滤低质量人脸检测
        # 在 align_warp_face() 之前过滤，避免对低置信度人脸做无意义的对齐+裁切+GFPGAN推理
        if (det_threshold > 0 and
                hasattr(helper, 'det_faces') and
                helper.det_faces is not None and
                len(helper.det_faces) > 0):
            _before_count = len(helper.det_faces)
            keep_indices = []
            # 过滤面积过小的检测（通常是误检）
            MIN_FACE_AREA = 48 * 48  # 最小人脸面积
            for _fi, _face in enumerate(helper.det_faces):
                # det_faces 格式: [x1, y1, x2, y2, confidence] 或 numpy array
                x1, y1, x2, y2 = _face[:4]
                area = (x2 - x1) * (y2 - y1)
                if area < MIN_FACE_AREA:
                    continue  # 跳过过小的检测
                _score = float(_face[4]) if len(_face) > 4 else float(_face[-1])
                if _score >= det_threshold:
                    keep_indices.append(_fi)
            _after_count = len(keep_indices)
            if _after_count < _before_count:
                _total_filtered += _before_count - _after_count
                # 同步过滤 all_landmarks_5 和 det_faces，保持索引一致
                if hasattr(helper, 'all_landmarks_5') and helper.all_landmarks_5:
                    helper.all_landmarks_5 = [helper.all_landmarks_5[_ki]
                                              for _ki in keep_indices]
                helper.det_faces = [helper.det_faces[_ki] for _ki in keep_indices]

        helper.align_warp_face()

        # FIX-PRECOMPUTE-INV-AFFINE (优化6): 预计算逆仿射矩阵
        inv_affines = []
        _upscale = getattr(helper, 'upscale_factor', 1)
        for a in helper.affine_matrices:
            inv = _cv2.invertAffineTransform(a)
            inv *= _upscale  # FIX-INV-AFFINE: 缩放到 SR 图像坐标系
            inv_affines.append(inv)

        face_data.append({
            'crops':       [c.copy() for c in helper.cropped_faces],  # 现在是 BGR
            'affines':     [a.copy() for a in helper.affine_matrices],
            'inv_affines': inv_affines,  # 优化6: 预计算
            'orig':        bgr_frame,  # 存 BGR 版本
        })
    _nf = sum(len(fd['crops']) for fd in face_data)
    _fw = sum(1 for fd in face_data if fd['crops'])
    # FIX-DET-THRESHOLD: 返回过滤计数供调用方累加统计
    return face_data, _fw, _nf, _total_filtered


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
    """
    纯 SR 推理：H2D → 模型前向 → 后处理 → D2H。
    trt_accel 可用时走 TRT 路径（全程 GPU 内存，data_ptr()）；
    否则走普通 PyTorch 路径（compute_stream 上）。
    返回 (sr_results, timing_info, status_flag) 以兼容 DeepPipelineOptimizer 调用约定。

    FIX-GPU-PREFETCH: 增加 prefetched_batch_t 参数。
    当上一轮 SR 推理完成后预取了下一批的 H2D 传输（在 transfer_stream 上异步执行），
    本轮可直接使用该预传输 tensor，跳过 H2D 阶段，节省 ~1-3ms/batch 的传输延迟。
    预取在人脸稀疏时最有价值：GPU 计算单元处于 SR 推理→GFPGAN 交接的空闲窗口，
    利用空闲内存总线提前完成下一批数据的搬运。

    FIX-SYNC: 修复 D2H 同步不足导致的鬼脸漂移。
    原代码仅同步 compute_stream，但 out_pinned.copy_(non_blocking=True) 的 D2H 拷贝
    运行在 default stream 上，compute_stream.synchronize() 无法保证 D2H 完成。
    改为 torch.cuda.synchronize(device) 同步所有流，与 v6.1 行为一致。
    """
    device   = upsampler.device
    use_half = upsampler.half
    pool     = _get_pinned_pool()
    B        = len(frames)
    t0       = time.perf_counter()

    # FIX-GPU-PREFETCH: 优先使用预传输的 GPU tensor，跳过 H2D 传输
    if (prefetched_batch_t is not None and
            prefetched_batch_t.shape[0] == B):
        batch_t = prefetched_batch_t
        # 预传输在 transfer_stream 上完成，等待其完成后再进入计算
        if transfer_stream is not None:
            torch.cuda.current_stream(device).wait_stream(transfer_stream)
    else:
        # 原有 H2D 传输路径
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
        # TRT 路径：全程 GPU 内存，不经过 CPU
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
        # 普通 PyTorch 路径
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

    # outscale != netscale 时（例如 -s 2 搭配 x4 模型）需要缩放
    if abs(outscale - netscale) > 1e-5:
        output_t = F.interpolate(
            output_t.float(), scale_factor=outscale / netscale,
            mode='bicubic', align_corners=False,
        )

    out_u8     = output_t.float().clamp_(0.0, 1.0).mul_(255.0).byte()
    out_perm   = out_u8.permute(0, 2, 3, 1).contiguous()
    out_pinned = pool.get_output_buf(out_perm.shape, torch.uint8)
    out_pinned.copy_(out_perm, non_blocking=True)

    # FIX-SYNC: 必须同步所有流（包括 default stream 上的 D2H 异步拷贝）。
    # 原代码仅 compute_stream.synchronize()，但 out_pinned.copy_(non_blocking=True)
    # 运行在 default stream 上，compute_stream 同步无法保证 D2H 完成，
    # 导致 out_pinned.numpy() 读到上一批残留数据与本批的混合 → 帧间鬼影叠加。
    # 改为 torch.cuda.synchronize(device)，与 v6.1 行为完全一致。
    torch.cuda.synchronize(device)

    out_np     = out_pinned.numpy()
    sr_results = [out_np[i].copy() for i in range(B)]

    elapsed     = time.perf_counter() - t0
    timing_info = {'batch_size': B, 'processing_time': elapsed}
    return sr_results, timing_info, 'success'


def _gfpgan_infer_batch(face_data, face_enhancer, device, fp16_ctx, gfpgan_weight, sub_bs, gfpgan_trt_accel=None, gfpgan_subprocess=None):
    """
    GFPGAN 批量推理 —— 直接对预检测 crops 做网络前向。

    关键修复：不调用 face_enhancer.enhance(has_aligned=False)。
    该方法内部会重新检测人脸，覆盖预先设置的 cropped_faces/affine_matrices，
    导致 restored_faces 数量与 _detect_faces_batch 检测出的 affine_matrices 不一致
    → _paste_faces_batch 报 "length of restored_faces and affine_matrices are different"。

    正确做法：直接把预检测的 crops 送入 gfpgan 网络，保证输出数量与 affines 严格对齐。
    """
    import contextlib as _ctx
    if not face_data:
        return [], sub_bs

    try:
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize as _tv_normalize
    except ImportError:
        restored_by_frame = []
        for fd in face_data:
            restored_by_frame.append([])
        return restored_by_frame, sub_bs

    _fp16 = fp16_ctx if fp16_ctx is not None else _ctx.nullcontext()

    restored_by_frame = []
    for fd in face_data:
        crops = fd.get('crops', [])
        if not crops:
            restored_by_frame.append([])
            continue

        all_restored = []
        i = 0
        _cur_sub_bs = sub_bs
        while i < len(crops):
            sub_crops = crops[i:i + _cur_sub_bs]
            tensors = []
            for crop in sub_crops:
                t = img2tensor(crop / 255., bgr2rgb=True, float32=True)
                _tv_normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                tensors.append(t)
            sub_batch = torch.stack(tensors).to(device)
            try:
                with torch.no_grad():
                    with _fp16:
                        out = face_enhancer.gfpgan(sub_batch, return_rgb=True, weight=gfpgan_weight)
                        if isinstance(out, (tuple, list)):
                            out = out[0]
                    out = out.float()

                for out_t in out.unbind(0):
                    if out_t is None:
                        all_restored.append(None)
                    else:
                        # --- 修复开始：处理 TRT FP16 导致的 NaN 和数值溢出 ---
                        # 1. 将 NaN 转换为 0，将正负无穷转换为对应边界值
                        out_t = torch.nan_to_num(out_t, nan=0.0, posinf=1.0, neginf=-1.0)
                        # 2. 强制截断到 GFPGAN 预期的 -1 到 1 范围，防止超出该范围的值在后处理时产生异常色块
                        out_t = torch.clamp(out_t, min=-1.0, max=1.0)
                        # --- 修复结束 ---

                        restored = tensor2img(out_t, rgb2bgr=True, min_max=(-1, 1))  # 现在正确：RGB → BGR
                        all_restored.append(restored.astype('uint8'))

                i += len(sub_crops)
            except RuntimeError as e:
                _estr = str(e).lower()
                if 'out of memory' in _estr and _cur_sub_bs > 1:
                    _cur_sub_bs = max(1, _cur_sub_bs // 2)
                    sub_bs = _cur_sub_bs
                    torch.cuda.empty_cache()
                else:
                    all_restored.extend([None] * len(sub_crops))
                    i += len(sub_crops)
                    torch.cuda.empty_cache()
            finally:
                del sub_batch

        restored_by_frame.append(all_restored)

    return restored_by_frame, sub_bs


def _paste_faces_batch(face_data, restored_by_frame, sr_results, face_enhancer):
    """人脸贴回函数

    FIX-PRECOMPUTE-INV-AFFINE (优化6): 使用预计算的逆仿射矩阵，
    跳过 face_helper.get_inverse_affine()，减少 CPU 开销。
    """
    import cv2 as _cv2
    expected_h, expected_w = sr_results[0].shape[:2]
    final_frames = []

    for fi, (fd, sr_frame) in enumerate(zip(face_data, sr_results)):
        if not restored_by_frame[fi]:
            final_frames.append(sr_frame)
            continue
        try:
            face_enhancer.face_helper.clean_all()
            sr_bgr = _cv2.cvtColor(sr_frame, _cv2.COLOR_RGB2BGR)
            face_enhancer.face_helper.read_image(fd['orig'])   # 传入低分辨率原始帧（BGR）

            # # 直接传入 SR 尺寸的参考图而非低分辨率原图
            # _orig_upscaled = _cv2.resize(
            #     fd['orig'],
            #     (sr_bgr.shape[1], sr_bgr.shape[0]),
            #     interpolation=_cv2.INTER_LANCZOS4
            # )
            # face_enhancer.face_helper.read_image(_orig_upscaled)  # 传入与SR同分辨率的BGR帧

            # FIX-ALIGN-AFFINE: 过滤掉 None 位置，保持 affines 与 valid_restored 数量一致
            # None 由 _gfpgan_infer_batch 在不可恢复错误时填入，表示该人脸推理失败
            _raw = restored_by_frame[fi]
            _affines = fd['affines']
            _crops   = fd['crops']
            _inv_affines = fd.get('inv_affines', [])  # 优化6: 预计算的逆仿射
            _n = min(len(_raw), len(_affines), len(_crops))  # 若长度不一致（理论上不应发生），取最短对齐
            valid_pairs = [(rf, _affines[j], _crops[j],
                            _inv_affines[j] if j < len(_inv_affines) else None)
                           for j, rf in enumerate(_raw[:_n]) if rf is not None]
            if not valid_pairs:
                # 所有人脸推理均失败，返回 RGB sr_frame
                final_frames.append(sr_frame)
                continue

            valid_restored, valid_affines, valid_crops, valid_inv = zip(*valid_pairs)
            face_enhancer.face_helper.affine_matrices = list(valid_affines)
            face_enhancer.face_helper.cropped_faces   = list(valid_crops)
            for rf in valid_restored:
                face_enhancer.face_helper.add_restored_face(rf)

            # FIX-PRECOMPUTE-INV-AFFINE (优化6): 直接设置预计算的逆仿射矩阵
            if all(v is not None for v in valid_inv):
                face_enhancer.face_helper.inverse_affine_matrices = list(valid_inv)
            else:
                face_enhancer.face_helper.get_inverse_affine(None)

            # FIX-BGR: 传入 sr_bgr（BGR）而非 sr_frame（RGB）
            _ret = face_enhancer.face_helper.paste_faces_to_input_image(
                upsample_img=sr_bgr)

            # facexlib 0.3.0+ 直接返回结果；旧版写入 .output 属性
            result_bgr = _ret if _ret is not None else getattr(
                face_enhancer.face_helper, 'output', None)

            if result_bgr is not None:
                # FIX-BGR: facexlib 输出 BGR，转回 RGB 保持与 sr_results 一致
                result = _cv2.cvtColor(result_bgr, _cv2.COLOR_BGR2RGB)
            else:
                # paste 返回 None（理论上不应发生），回退原始 RGB sr_frame
                result = sr_frame

        except Exception as e:
            print(f'[face_enhance] 帧{fi} 贴回异常，使用 SR 结果: {e}')
            result = sr_frame  # 回退 RGB sr_frame

        # 尺寸安全检查：确保输出与 SR 帧严格一致
        if result.shape[0] != expected_h or result.shape[1] != expected_w:
            print(f'[WARN] face_enhance 帧{fi} 尺寸异常 '
                  f'{result.shape[:2]} != ({expected_h},{expected_w})，强制 resize')
            result = _cv2.resize(result, (expected_w, expected_h),
                                 interpolation=_cv2.INTER_LANCZOS4)
        final_frames.append(result)

    return final_frames

# ─────────────────────────────────────────────────────────────────────────────
# TRT 进程级 Logger 单例
# ─────────────────────────────────────────────────────────────────────────────
_TRT_LOGGER = None

def _get_trt_logger():
    """返回进程级 TRT Logger 单例；首次调用时创建并缓存。"""
    global _TRT_LOGGER
    if _TRT_LOGGER is None:
        try:
            import tensorrt as _trt_mod
            _TRT_LOGGER = _trt_mod.Logger(_trt_mod.Logger.ERROR)
        except ImportError:
            pass
    return _TRT_LOGGER


class TensorRTAccelerator:
    """
    将 RealESRGAN 模型导出 ONNX 后编译 TRT Engine (FP16, 静态形状)。
    """

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 cache_dir: str, input_shape: Tuple[int, int, int, int],
                 use_fp16: bool = True):
        self.device      = device
        self.input_shape = input_shape
        self.use_fp16    = use_fp16
        self._engine     = None
        self._context    = None
        self._trt_ok     = False
        self._trt_stream: Optional[torch.cuda.Stream] = None

        try:
            import tensorrt as trt
            self._trt  = trt
        except ImportError as e:
            print(f'[TensorRT] 依赖未安装，跳过 TRT 加速: {e}')
            print('  安装命令: pip install tensorrt onnx onnxruntime-gpu')
            return

        _sm_tag = ''
        if torch.cuda.is_available():
            _p = torch.cuda.get_device_properties(0)
            import re as _re_sr
            _gpu_slug_sr = _re_sr.sub(r'[^a-z0-9]', '', _p.name.lower())[:16]
            _sm_tag = f'_sm{_p.major}{_p.minor}_{_gpu_slug_sr}'

        B, C, H, W = input_shape
        tag       = f'B{B}_C{C}_H{H}_W{W}_fp{"16" if use_fp16 else "32"}{_sm_tag}'
        trt_path  = osp.join(cache_dir, f'realesrgan_{tag}.trt')
        onnx_path = osp.join(cache_dir, f'realesrgan_{tag}.onnx')
        os.makedirs(cache_dir, exist_ok=True)

        if not osp.exists(trt_path):
            print(f'[TensorRT] 构建 Engine (shape={input_shape}, tag={tag}) ...')
            self._export_onnx(model, onnx_path, input_shape)
            self._build_engine(onnx_path, trt_path, use_fp16)

        if osp.exists(trt_path):
            try:
                self._load_engine(trt_path)
            except RuntimeError as _e:
                print(f'[TensorRT] 首次加载失败（{_e}），开始重新构建 Engine...')
                if not osp.exists(onnx_path):
                    self._export_onnx(model, onnx_path, input_shape)
                self._build_engine(onnx_path, trt_path, use_fp16)
                if osp.exists(trt_path):
                    self._load_engine(trt_path)

    def _export_onnx(self, model, onnx_path, input_shape):
        model.eval()
        dummy = torch.randn(*input_shape, device=self.device)
        if self.use_fp16:
            dummy = dummy.half()
            model = model.half()
        with torch.no_grad():
            torch.onnx.export(
                model, dummy, onnx_path,
                input_names=['input'], output_names=['output'],
                opset_version=18,
                dynamic_axes=None,
            )
        print(f'[TensorRT] ONNX 已导出: {onnx_path}')

    def _build_engine(self, onnx_path, trt_path, use_fp16):
        trt    = self._trt
        logger = _get_trt_logger()
        builder = trt.Builder(logger)
        try:
            explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        except AttributeError:
            explicit_batch_flag = 0
        network = builder.create_network(explicit_batch_flag)
        parser  = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(onnx_path):
            for i in range(parser.num_errors):
                print(f'  [TensorRT] ONNX 解析错误: {parser.get_error(i)}')
            del parser, network, builder
            return
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
        if use_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # 添加 SM 检测和时间预估
        _gpu_name = 'unknown'
        _sm_major = 0
        _sm_minor = 0
        if torch.cuda.is_available():
            _props = torch.cuda.get_device_properties(0)
            _gpu_name = _props.name
            _sm_major = _props.major
            _sm_minor = _props.minor
            if _sm_major < 7 or (_sm_major == 7 and _sm_minor < 5):
                print(f'[TensorRT] 警告: {_gpu_name} (SM {_sm_major}.{_sm_minor}) '
                      f'可能不受当前 TRT 版本支持（通常需要 SM 7.5+）')
        _sm_code = _sm_major * 10 + _sm_minor
        _time_hint = {
            75: '约需 20~30 分钟（T4/RTX20系 SM7.5）',
            80: '约需 8~15 分钟（A100/A30 SM8.0）',
            86: '约需 10~18 分钟（A10/RTX30系 SM8.6）',
            89: '约需 8~12 分钟（RTX40系 SM8.9）',
            90: '约需 5~10 分钟（H100 SM9.0）',
        }.get(_sm_code, f'约需 10~30 分钟（{_gpu_name} SM{_sm_major}.{_sm_minor}）')
        print(f'[TensorRT] {_time_hint}')
        
        # 添加心跳线程，每300秒报告一次状态
        _build_start = time.time()
        _build_done = threading.Event()
        def _heartbeat():
            _last = time.time()
            while not _build_done.wait(timeout=5):
                if time.time() - _last >= 300:
                    elapsed = time.time() - _build_start
                    print(f'[TensorRT] 编译中... {elapsed:.0f}s（仍在运行，请耐心等待）')
                    _last = time.time()
        _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
        _hb_thread.start()

        serialized = builder.build_serialized_network(network, config)
        _build_done.set()
        _build_elapsed = time.time() - _build_start
        del config, parser, network, builder
        import gc; gc.collect()
        if serialized is None:
            _sm_str = f'SM {_sm_major}.{_sm_minor if _sm_major else "?"}' if torch.cuda.is_available() else ''
            _sm_hint = (f'\n[TensorRT] 提示: {_gpu_name} ({_sm_str}) 可能不受此 TRT 版本支持'
                        if _sm_major < 8 else '')
            print(f'[TensorRT] Engine 构建失败（返回 None，用时 {_build_elapsed:.0f}s）{_sm_hint}')
            return
        with open(trt_path, 'wb') as f:
            f.write(serialized)
        del serialized
        print(f'[TensorRT] Engine 已缓存（用时 {_build_elapsed:.0f}s）: {trt_path}')

    def _load_engine(self, trt_path):
        _cur_sm_tag = ''
        if torch.cuda.is_available():
            _pp = torch.cuda.get_device_properties(0)
            import re as _re
            _gpu_slug = _re.sub(r'[^a-z0-9]', '', _pp.name.lower())[:16]
            _cur_sm_tag = f'_sm{_pp.major}{_pp.minor}_{_gpu_slug}'
        if _cur_sm_tag:
            _basename = osp.basename(trt_path)
            if _cur_sm_tag not in _basename:
                print(f'[TensorRT] .trt 文件名不含当前 GPU SM tag {_cur_sm_tag}，'
                      f'可能是旧版本缓存或跨 GPU 遗留文件: {_basename}')
                print(f'[TensorRT] 删除过期缓存，触发针对当前 GPU 的重建')
                try:
                    os.remove(trt_path)
                except OSError:
                    pass
                raise RuntimeError(f'[TensorRT] 过期缓存 {_basename} 已删除，需重建')
        trt     = self._trt
        logger  = _get_trt_logger()
        runtime = trt.Runtime(logger)
        with open(trt_path, 'rb') as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        del runtime
        if self._engine is None:
            print(f'[TensorRT] Engine 反序列化失败，删除过期缓存并重新构建: {trt_path}')
            try:
                os.remove(trt_path)
            except OSError:
                pass
            raise RuntimeError('[TensorRT] _load_engine: deserialize_cuda_engine returned None')

        # [FIX-TRT-CTX-OOM] create_execution_context() 在 GPU 显存不足时
        # 返回 None 而非抛出 Python 异常（与 deserialize_cuda_engine 行为一致）。
        # 典型场景：interpolate_then_upscale 模式下，前序 IFRNet 步骤的
        # PyTorch 缓存分配器残留大量显存，导致 TRT 无法分配 context 所需的
        # 激活内存（通常为数 GB 量级）。
        # 若不检测，后续 infer() 中 self._context.set_tensor_address() /
        # execute_async_v3() 会在 NoneType 上调用 → AttributeError 崩溃。
        self._context = self._engine.create_execution_context()
        if self._context is None:
            print('[TensorRT] ⚠️  create_execution_context() 失败'
                  '（GPU 显存不足），回退 PyTorch 推理路径。')
            print('[TensorRT] 提示: 前序处理步骤可能占用了大量显存。'
                  '可尝试减小 --batch-size 或移除 --use-tensorrt。')
            # 释放已加载的 engine，归还显存
            self._engine = None
            self._trt_ok = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return  # 不抛异常，__init__ 中 self._trt_ok 保持 False，自动走 PyTorch 路径

        self._use_new_api  = hasattr(self._engine, 'num_io_tensors')
        self._input_name   = None
        self._output_name  = None
        trt = self._trt
        if self._use_new_api:
            for i in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(i)
                mode = self._engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    self._input_name = name
                elif mode == trt.TensorIOMode.OUTPUT:
                    self._output_name = name
            if self._input_name is None or self._output_name is None:
                raise RuntimeError(
                    '[TensorRT] 无法在 Engine 中找到有效输入/输出 tensor')
            print(f'[TensorRT] 使用新版 API (TRT 10.x)，'
                  f'输入: {self._input_name}，输出: {self._output_name}')
        else:
            print('[TensorRT] 使用旧版 API (TRT 8.x)')
        self._trt_ok = True
        print('[TensorRT] Engine 加载成功，已启用 TRT 推理')

    @property
    def available(self) -> bool:
        return self._trt_ok

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        actual_B  = input_tensor.shape[0]
        engine_B  = self.input_shape[0]
        if actual_B < engine_B:
            pad_cnt = engine_B - actual_B
            pad     = input_tensor[-1:].expand(pad_cnt, -1, -1, -1)
            input_tensor = torch.cat([input_tensor, pad], dim=0)
        inp      = input_tensor.contiguous()
        out_dtype = torch.float16 if self.use_fp16 else torch.float32
        if self._trt_stream is None:
            self._trt_stream = torch.cuda.Stream(device=self.device)
        if self._use_new_api:
            out_shape  = tuple(self._engine.get_tensor_shape(self._output_name))
            out_tensor = torch.empty(out_shape, dtype=out_dtype, device=self.device)
            self._trt_stream.wait_stream(torch.cuda.current_stream(self.device))
            self._context.set_tensor_address(self._input_name,  inp.data_ptr())
            self._context.set_tensor_address(self._output_name, out_tensor.data_ptr())
            self._context.execute_async_v3(stream_handle=self._trt_stream.cuda_stream)
        else:
            out_shape  = tuple(self._engine.get_binding_shape(1))
            out_tensor = torch.empty(out_shape, dtype=out_dtype, device=self.device)
            self._trt_stream.wait_stream(torch.cuda.current_stream(self.device))
            self._context.execute_async_v2(
                bindings=[inp.data_ptr(), out_tensor.data_ptr()],
                stream_handle=self._trt_stream.cuda_stream,
            )
        self._trt_stream.synchronize()
        if actual_B < engine_B:
            out_tensor = out_tensor[:actual_B]
        return out_tensor


class GFPGANSubprocess:
    """
    将 GFPGAN 推理移至独立子进程，避免与主进程的 PyTorch 分配器冲突。
    支持 PyTorch FP16 或 TensorRT（取决于参数）。

    FIX-EARLY-SPAWN: 支持在 SR 模型加载之前独立启动。
    """
    def __init__(self, face_enhancer=None, device=None, gfpgan_weight=0.5, gfpgan_batch_size=4,
                 use_fp16=True, use_trt=False, trt_cache_dir=None, gfpgan_model='1.4',
                 model_path=None):
        self.device = device
        self.gfpgan_weight = gfpgan_weight
        self.gfpgan_batch_size = gfpgan_batch_size
        self.use_fp16 = use_fp16
        self.use_trt = use_trt
        self.trt_cache_dir = trt_cache_dir
        self.gfpgan_model = gfpgan_model

        if model_path is not None:
            self.model_path = model_path
        elif face_enhancer is not None:
            try:
                self.model_path = face_enhancer.model_path
            except AttributeError:
                pass

        if not hasattr(self, 'model_path'):
            if face_enhancer is not None:
                try:
                    self.model_path = face_enhancer.model_path
                except AttributeError:
                    pass
            if not hasattr(self, 'model_path'):
                model_paths = {
                    '1.3': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                    '1.4': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                    'RestoreFormer': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
                }
                model_url = model_paths.get(gfpgan_model, model_paths['1.4'])
                model_dir = osp.join(models_RealESRGAN, 'GFPGAN')
                os.makedirs(model_dir, exist_ok=True)
                model_filename = osp.basename(model_url)
                self.model_path = osp.join(model_dir, model_filename)
                if not osp.exists(self.model_path):
                    print(f'[GFPGANSubprocess] 下载模型: {model_filename}')
                    self.model_path = load_file_from_url(model_url, model_dir, True)

        if face_enhancer is not None:
            self.gfpgan_net = face_enhancer.gfpgan
            self.face_enhancer_upscale = face_enhancer.upscale
        else:
            self.gfpgan_net = None
            self.face_enhancer_upscale = 1

        self._mp_ctx = mp.get_context('spawn')
        self.task_queue   = self._mp_ctx.Queue(maxsize=2)
        self.result_queue = self._mp_ctx.Queue(maxsize=2)
        self.ready_event  = self._mp_ctx.Event()
        self.process = None

        # ── 两阶段子进程架构（FIX-TWO-PHASE）────────────────────────
        if use_trt:
            _sm_tag = ''
            if torch.cuda.is_available():
                _p = torch.cuda.get_device_properties(0)
                import re as _re_coord
                _gpu_slug_coord = _re_coord.sub(r'[^a-z0-9]', '', _p.name.lower())[:16]
                _sm_tag = f'_sm{_p.major}{_p.minor}_{_gpu_slug_coord}'
            tag       = (f'gfpgan_{gfpgan_model}_w{gfpgan_weight:.3f}'
                         f'_B{gfpgan_batch_size}_fp{"16" if use_fp16 else "32"}{_sm_tag}')
            cache_dir = trt_cache_dir or osp.join(os.getcwd(), '.trt_cache')
            trt_path  = osp.join(cache_dir, f'{tag}.trt')

            if not osp.exists(trt_path):
                print(f'[GFPGANSubprocess] Phase 1: 启动独立 Builder 进程构建 TRT Engine...', flush=True)
                print(f'[GFPGANSubprocess] 构建完成后 Builder 自动退出，再启动干净 Inference 进程', flush=True)
                builder = self._mp_ctx.Process(
                    target=GFPGANSubprocess._build_only_worker,
                    args=(self.model_path, gfpgan_model, gfpgan_weight,
                          gfpgan_batch_size, use_fp16, trt_cache_dir),
                    daemon=False,
                )
                builder.start()
                _p1_start      = time.time()
                _p1_max        = 5400
                _p1_poll       = 5
                _p1_reported   = False
                _p1_deadline   = _p1_start + _p1_max
                while time.time() < _p1_deadline:
                    builder.join(timeout=_p1_poll)
                    if not builder.is_alive():
                        break
                    if not _p1_reported:
                        elapsed = time.time() - _p1_start
                        print(f'[GFPGANSubprocess] Phase 1 编译中... {elapsed:.0f}s（Builder 进程运行中）', flush=True)
                        _p1_reported = True
                _p1_elapsed = time.time() - _p1_start
                if builder.is_alive():
                    builder.terminate()
                    print(f'[GFPGANSubprocess] Builder 超时（>{_p1_max//60}min），TRT 构建失败', flush=True)
                elif osp.exists(trt_path):
                    print(f'[GFPGANSubprocess] Phase 1 完成，用时 {_p1_elapsed:.0f}s，.trt 已生成，启动 Phase 2 Inference 进程', flush=True)
                else:
                    print(f'[GFPGANSubprocess] Phase 1 失败（用时 {_p1_elapsed:.0f}s，.trt 未生成），Phase 2 将走 PyTorch 路径', flush=True)

        # ── FIX-SHM-IPC (优化3): 创建双缓冲共享内存 ──────────────────
        self.shm_buf: Optional[SharedMemoryDoubleBuffer] = None
        try:
            self.shm_buf = SharedMemoryDoubleBuffer()
            print(f'[GFPGANSubprocess] 共享内存双缓冲已创建 '
                  f'(input: {self.shm_buf.input_names}, '
                  f'output: {self.shm_buf.output_names})', flush=True)
        except Exception as _shm_e:
            print(f'[GFPGANSubprocess] 共享内存创建失败，回退 pickle: {_shm_e}',
                  flush=True)
            self.shm_buf = None

        # Phase 2: 始终启动 Inference 进程（干净 CUDA context）
        self._start()

    @staticmethod
    def _build_only_worker(model_path, gfpgan_model, gfpgan_weight,
                           gfpgan_batch_size, use_fp16, trt_cache_dir):
        """Phase 1 Builder 进程：仅做 TRT build，完成后立即退出。"""
        import warnings; warnings.filterwarnings('ignore')
        import os, sys, gc, time, threading
        import os.path as osp
        import torch
        from gfpgan import GFPGANer
        import contextlib, torch.nn.functional as _F

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _GFPGAN_MODELS = {
            '1.3': ('clean', 2, 'GFPGANv1.3'),
            '1.4': ('clean', 2, 'GFPGANv1.4'),
            'RestoreFormer': ('RestoreFormer', 2, 'RestoreFormer'),
        }
        arch, channel_multiplier, _ = _GFPGAN_MODELS[gfpgan_model]
        face_enhancer = GFPGANer(
            model_path=model_path, upscale=1, arch=arch,
            channel_multiplier=channel_multiplier, bg_upsampler=None, device=device,
        )

        _sm_tag_b = ''
        if torch.cuda.is_available():
            _pb = torch.cuda.get_device_properties(0)
            import re as _re_b
            _gpu_slug_b = _re_b.sub(r'[^a-z0-9]', '', _pb.name.lower())[:16]
            _sm_tag_b = f'_sm{_pb.major}{_pb.minor}_{_gpu_slug_b}'
        tag       = (f'gfpgan_{gfpgan_model}_w{gfpgan_weight:.3f}'
                     f'_B{gfpgan_batch_size}_fp{"16" if use_fp16 else "32"}{_sm_tag_b}')
        cache_dir = trt_cache_dir or osp.join(os.getcwd(), '.trt_cache')
        trt_path  = osp.join(cache_dir, f'{tag}.trt')
        onnx_path = osp.join(cache_dir, f'{tag}.onnx')
        os.makedirs(cache_dir, exist_ok=True)

        if osp.exists(trt_path):
            print(f'[Builder] .trt 已存在，跳过构建: {trt_path}', flush=True)
            import os as _os; _os._exit(0)

        try:
            import tensorrt as trt
        except ImportError as e:
            print(f'[Builder] tensorrt 未安装: {e}', flush=True)
            import os as _os; _os._exit(1)

        @contextlib.contextmanager
        def _onnx_compat_patch():
            _patches = []
            try:
                import basicsr.ops.fused_act.fused_act as _fa_mod
                _orig = _fa_mod.fused_leaky_relu
                def _compat(inp, bias, negative_slope=0.2, scale=2**0.5):
                    bv = (bias.view(1,-1,1,1) if (bias.dim()==1 and inp.dim()==4) else bias)
                    return _F.leaky_relu(inp + bv, negative_slope=negative_slope) * scale
                _fa_mod.fused_leaky_relu = _compat
                _patches.append((_fa_mod, 'fused_leaky_relu', _orig))
                try:
                    import basicsr.ops.fused_act as _fa_pkg
                    if hasattr(_fa_pkg, 'fused_leaky_relu'):
                        _patches.append((_fa_pkg, 'fused_leaky_relu', _fa_pkg.fused_leaky_relu))
                        _fa_pkg.fused_leaky_relu = _compat
                except Exception: pass
            except Exception: pass
            try:
                import basicsr.ops.upfirdn2d.upfirdn2d as _ud_mod
                if getattr(_ud_mod, '_use_custom_op', False):
                    _patches.append((_ud_mod, '_use_custom_op', True))
                    _ud_mod._use_custom_op = False
            except Exception: pass
            try:
                yield
            finally:
                for _obj, _attr, _orig in reversed(_patches):
                    try: setattr(_obj, _attr, _orig)
                    except Exception: pass

        gfpgan_net = face_enhancer.gfpgan.eval()
        dummy_d = torch.randn(1, 3, 512, 512, device=device)
        if use_fp16:
            dummy_d = dummy_d.half()
            gfpgan_net = gfpgan_net.half()
        with torch.no_grad():
            _test = gfpgan_net(dummy_d, return_rgb=True)
        _returns_tuple = isinstance(_test, (tuple, list))
        _w = float(gfpgan_weight)
        
        # FIX-TRT-WEIGHT: 移除 TRT wrapper 中的 weight blending。
        # v6.1 不在 GFPGAN 推理阶段做 weight blending（weight 参数被网络忽略），
        # blending 语义仅在 paste_faces_to_input_image 的 mask 融合中体现。
        # TRT wrapper 内嵌 blending 导致输出是 (w*enhanced + (1-w)*input)，
        # 增强脸与原始裁切亮度/色彩差异在 paste 边界产生花斑。
        class _W(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = gfpgan_net
            def forward(self, x):
                out = self.net(x, return_rgb=False)
                if _returns_tuple: out = out[0]
                return out
        wrapper = _W().to(device)

        if use_fp16:
            wrapper = wrapper.half()
        wrapper.eval()

        print(f'[Builder] ONNX 导出 (静态 batch={gfpgan_batch_size})...', flush=True)
        dummy = torch.randn(gfpgan_batch_size, 3, 512, 512, device=device)
        if use_fp16:
            dummy = dummy.half()
        try:
            with _onnx_compat_patch():
                with torch.no_grad():
                    torch.onnx.export(
                        wrapper, dummy, onnx_path,
                        input_names=['input'], output_names=['output'],
                        opset_version=18, dynamo=False,
                    )
            print(f'[Builder] ONNX 已导出: {onnx_path}', flush=True)
        except Exception as e:
            print(f'[Builder] ONNX 导出失败: {e}', flush=True)
            import os as _os; _os._exit(1)
        del wrapper, dummy, dummy_d, face_enhancer
        gc.collect()
        torch.cuda.empty_cache()

        _sm_ok = True
        _gpu_name_b = 'unknown'
        _sm_major = 0
        if torch.cuda.is_available():
            _props = torch.cuda.get_device_properties(0)
            _gpu_name_b = _props.name
            _sm_major = _props.major
            _sm_minor = _props.minor
            if _sm_major < 7 or (_sm_major == 7 and _sm_minor < 5):
                print(f'[Builder] 警告: {_gpu_name_b} (SM {_sm_major}.{_sm_minor}) '
                      f'可能不受当前 TRT 版本支持（通常需要 SM 7.5+）', flush=True)

        _sm_minor = 0
        if torch.cuda.is_available():
            _sm_minor = torch.cuda.get_device_properties(0).minor
        _sm_code = _sm_major * 10 + _sm_minor
        _time_hint = {
            75: '约需 20~30 分钟（T4/RTX20系 SM7.5）',
            80: '约需 8~15 分钟（A100/A30 SM8.0）',
            86: '约需 10~18 分钟（A10/RTX30系 SM8.6）',
            89: '约需 8~12 分钟（RTX40系 SM8.9）',
            90: '约需 5~10 分钟（H100 SM9.0）',
        }.get(_sm_code, f'约需 10~30 分钟（{_gpu_name_b} SM{_sm_major}.{_sm_minor}）')
        print(f'[Builder] 构建 TRT Engine (B={gfpgan_batch_size}, fp16={use_fp16})...', flush=True)
        print(f'[Builder] {_time_hint}', flush=True)
        try:
            logger  = trt.Logger(trt.Logger.ERROR)
            builder = trt.Builder(logger)
            try:   flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            except AttributeError: flag = 0
            network = builder.create_network(flag)
            parser  = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(onnx_path):
                for i in range(parser.num_errors):
                    print(f'  [Builder] 解析错误: {parser.get_error(i)}', flush=True)
                import os as _os; _os._exit(1)
            print('[Builder] ONNX 解析完成，开始编译...', flush=True)
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
            if use_fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            profile = builder.create_optimization_profile()
            _bs = gfpgan_batch_size
            profile.set_shape('input',
                min=(_bs,3,512,512), opt=(_bs,3,512,512), max=(_bs,3,512,512))
            config.add_optimization_profile(profile)
            _build_start  = time.time()
            _build_done   = threading.Event()
            _report_every = 300
            def _heartbeat():
                _last = time.time()
                while not _build_done.wait(timeout=5):
                    if time.time() - _last >= _report_every:
                        elapsed = time.time() - _build_start
                        print(f'[Builder] 编译中... {elapsed:.0f}s（仍在运行，请耐心等待）', flush=True)
                        _last = time.time()
            _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
            _hb_thread.start()
            serialized = builder.build_serialized_network(network, config)
            _build_done.set()
            _build_elapsed = time.time() - _build_start
            del config, profile, parser, network, builder
            gc.collect()
            if serialized is None:
                _sm_str = f'SM {_sm_major}.{_sm_minor if _sm_major else "?"}' if torch.cuda.is_available() else ''
                _sm_hint = (f'\n[Builder] 提示: {_gpu_name_b} ({_sm_str}) 可能不受此 TRT 版本支持，'
                            f'请降级 TRT 或改用 PyTorch FP16（去掉 --gfpgan-trt）'
                            if _sm_major < 8 else '')
                print(f'[Builder] Engine 构建失败（返回 None，用时 {_build_elapsed:.0f}s）{_sm_hint}', flush=True)
                import os as _os; _os._exit(1)
            with open(trt_path, 'wb') as f:
                f.write(serialized)
            del serialized
            print(f'[Builder] Engine 已缓存（用时 {_build_elapsed:.0f}s）: {trt_path}', flush=True)
        except Exception as e:
            print(f'[Builder] Engine 构建异常: {e}', flush=True)
            import os as _os; _os._exit(1)
        import os as _os; _os._exit(0)

    def _start(self):
        """启动 Phase 2 Inference 子进程（spawn，CUDA context 干净）"""
        # FIX-SHM-IPC: 传递共享内存名称给子进程
        _shm_input_names = self.shm_buf.input_names if self.shm_buf else None
        _shm_output_names = self.shm_buf.output_names if self.shm_buf else None
        self.process = self._mp_ctx.Process(target=self._worker, args=(
            self.model_path, self.gfpgan_model, self.gfpgan_weight,
            self.gfpgan_batch_size, self.use_fp16, self.use_trt,
            self.trt_cache_dir, self.task_queue, self.result_queue,
            self.ready_event,
            _shm_input_names, _shm_output_names,  # 优化：共享内存 + 异步支持
        ), daemon=True)
        self.process.start()

    @staticmethod
    def _worker(model_path, gfpgan_model, gfpgan_weight, gfpgan_batch_size,
                use_fp16, use_trt, trt_cache_dir, task_queue, result_queue,
                ready_event=None,
                shm_input_names=None, shm_output_names=None):  # 优化：共享内存 + 异步支持
        """子进程主函数：加载模型并循环处理任务"""
        import warnings
        warnings.filterwarnings('ignore')
        import os
        os.environ['PYTHONWARNINGS'] = 'ignore'
        import torch
        import numpy as np
        from gfpgan import GFPGANer
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize as _tv_normalize
        import contextlib

        cuda_context_dead = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            torch.cuda.init()

        _GFPGAN_MODELS = {
            '1.3': ('clean', 2, 'GFPGANv1.3'),
            '1.4': ('clean', 2, 'GFPGANv1.4'),
            'RestoreFormer': ('RestoreFormer', 2, 'RestoreFormer'),
        }
        arch, channel_multiplier, name = _GFPGAN_MODELS[gfpgan_model]

        if use_trt and torch.cuda.is_available():
            print('[GFPGANSubprocess] FIX-INIT-ORDER: TRT 路径，延迟 GFPGANer GPU 加载', flush=True)
            face_enhancer = GFPGANer(
                model_path=model_path, upscale=1, arch=arch,
                channel_multiplier=channel_multiplier, bg_upsampler=None,
                device=torch.device('cpu'),
            )
            model = face_enhancer.gfpgan
            model.eval()
        else:
            face_enhancer = GFPGANer(
                model_path=model_path, upscale=1, arch=arch,
                channel_multiplier=channel_multiplier, bg_upsampler=None,
                device=device,
            )
            model = face_enhancer.gfpgan
            model.eval()

        gfpgan_trt_accel = None
        if use_trt and torch.cuda.is_available():
            try:
                from typing import Optional, List, Tuple
                import os
                import os.path as osp
                import torch
                import numpy as np
                import contextlib as _ctx
                import torch.nn.functional as _F

                def _get_subprocess_trt_logger():
                    import tensorrt as _trt
                    if not hasattr(_get_subprocess_trt_logger, '_inst'):
                        _get_subprocess_trt_logger._inst = _trt.Logger(_trt.Logger.ERROR)
                    return _get_subprocess_trt_logger._inst

                class GFPGANTRTAccelerator:
                    """GFPGAN TRT 加速（v6.2 验证版，移植入子进程）"""

                    def __init__(self, face_enhancer, device, cache_dir, gfpgan_weight,
                                 max_batch_size, gfpgan_version, use_fp16=True):
                        self.device          = device
                        self.use_fp16        = use_fp16
                        self._max_batch_size = max_batch_size
                        self._engine         = None
                        self._context        = None
                        self._trt_ok         = False
                        self._trt_stream     = None
                        self._input_name     = None
                        self._output_name    = None
                        self._use_new_api    = False
                        self._cuda_context_dead = False
                        self._warmup_failed = False

                        try:
                            import tensorrt as trt
                            self._trt = trt
                        except ImportError as e:
                            print(f'[GFPGAN-TensorRT] tensorrt 未安装: {e}', flush=True)
                            return

                        _sm_tag_w = ''
                        try:
                            import torch as _t
                            if _t.cuda.is_available():
                                _pw = _t.cuda.get_device_properties(0)
                                import re as _re_w
                                _gpu_slug_w = _re_w.sub(r'[^a-z0-9]', '', _pw.name.lower())[:16]
                                _sm_tag_w = f'_sm{_pw.major}{_pw.minor}_{_gpu_slug_w}'
                        except Exception:
                            pass
                        tag      = (f'gfpgan_{gfpgan_version}_w{gfpgan_weight:.3f}'
                                    f'_B{max_batch_size}_fp{"16" if use_fp16 else "32"}{_sm_tag_w}')
                        os.makedirs(cache_dir, exist_ok=True)
                        trt_path  = osp.join(cache_dir, f'{tag}.trt')
                        onnx_path = osp.join(cache_dir, f'{tag}.onnx')
                        self._trt_path = trt_path
                        if not osp.exists(trt_path):
                            print(f'[GFPGAN-TensorRT] .trt 不存在，跳过构建，走 PyTorch 路径', flush=True)
                            return
                        if osp.exists(trt_path):
                            try:
                                self._load_engine(trt_path)
                            except RuntimeError as _e:
                                print(f'[GFPGAN-TensorRT] 首次加载失败({_e})，重建...', flush=True)
                                wrapper = self._build_wrapper(face_enhancer.gfpgan, gfpgan_weight, device, use_fp16)
                                if not osp.exists(onnx_path):
                                    self._export_onnx(wrapper, onnx_path, max_batch_size)
                                if osp.exists(onnx_path):
                                    self._build_engine_dynamic(onnx_path, trt_path, max_batch_size, use_fp16)
                                    if osp.exists(trt_path):
                                        self._load_engine(trt_path)

                    @staticmethod
                    def _build_wrapper(gfpgan_net, weight, device, use_fp16):
                        gfpgan_net = gfpgan_net.eval()
                        actual_device = next(gfpgan_net.parameters()).device
                        dummy = torch.randn(1, 3, 512, 512, device=actual_device)
                        if use_fp16:
                            dummy      = dummy.half()
                            gfpgan_net = gfpgan_net.half()
                        with torch.no_grad():
                            _test = gfpgan_net(dummy, return_rgb=True)
                        _returns_tuple = isinstance(_test, (tuple, list))
                        # FIX-TRT-WEIGHT: 移除 weight blending，与 v6.1 一致。
                        class _W(torch.nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.net = gfpgan_net
                            def forward(self, x):
                                out = self.net(x, return_rgb=False)
                                if _returns_tuple: out = out[0]
                                return out
                        return _W().to(device)

                    @staticmethod
                    @_ctx.contextmanager
                    def _onnx_compat_patch():
                        _patches = []
                        try:
                            import basicsr.ops.fused_act.fused_act as _fa_mod
                            _orig = _fa_mod.fused_leaky_relu
                            def _compat(inp, bias, negative_slope=0.2, scale=2**0.5):
                                bv = (bias.view(1,-1,1,1) if (bias.dim()==1 and inp.dim()==4) else bias)
                                return _F.leaky_relu(inp + bv, negative_slope=negative_slope) * scale
                            _fa_mod.fused_leaky_relu = _compat
                            _patches.append((_fa_mod, 'fused_leaky_relu', _orig))
                            try:
                                import basicsr.ops.fused_act as _fa_pkg
                                if hasattr(_fa_pkg, 'fused_leaky_relu'):
                                    _patches.append((_fa_pkg, 'fused_leaky_relu', _fa_pkg.fused_leaky_relu))
                                    _fa_pkg.fused_leaky_relu = _compat
                            except Exception: pass
                        except Exception: pass
                        try:
                            import basicsr.ops.upfirdn2d.upfirdn2d as _ud_mod
                            if getattr(_ud_mod, '_use_custom_op', False):
                                _patches.append((_ud_mod, '_use_custom_op', True))
                                _ud_mod._use_custom_op = False
                        except Exception: pass
                        try:
                            yield
                        finally:
                            for _obj, _attr, _orig in reversed(_patches):
                                try: setattr(_obj, _attr, _orig)
                                except Exception: pass

                    def _export_onnx(self, wrapper, onnx_path, max_batch_size):
                        wrapper = wrapper.eval()
                        dummy   = torch.randn(max_batch_size, 3, 512, 512, device=self.device)
                        if self.use_fp16:
                            dummy   = dummy.half()
                            wrapper = wrapper.half()
                        try:
                            with self._onnx_compat_patch():
                                with torch.no_grad():
                                    torch.onnx.export(wrapper, dummy, onnx_path,
                                        input_names=['input'], output_names=['output'],
                                        opset_version=18, dynamo=False)
                            return True
                        except Exception as _e:
                            print(f'[GFPGAN-TensorRT] ONNX 导出失败: {_e}', flush=True)
                            return False

                    def _build_engine_dynamic(self, onnx_path, trt_path, max_batch_size, use_fp16):
                        trt    = self._trt
                        logger = _get_subprocess_trt_logger()
                        builder = trt.Builder(logger)
                        try: flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                        except AttributeError: flag = 0
                        network = builder.create_network(flag)
                        parser  = trt.OnnxParser(network, logger)
                        if not parser.parse_from_file(onnx_path):
                            del parser, network, builder
                            return
                        config = builder.create_builder_config()
                        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
                        if use_fp16 and builder.platform_has_fast_fp16:
                            config.set_flag(trt.BuilderFlag.FP16)
                        profile = builder.create_optimization_profile()
                        _bs = max_batch_size
                        profile.set_shape('input', min=(_bs,3,512,512), opt=(_bs,3,512,512), max=(_bs,3,512,512))
                        config.add_optimization_profile(profile)
                        serialized = builder.build_serialized_network(network, config)
                        del config, profile, parser, network, builder
                        import gc; gc.collect()
                        if serialized is None:
                            return
                        with open(trt_path, 'wb') as f:
                            f.write(serialized)
                        del serialized

                    def _load_engine(self, trt_path):
                        _cur_sm_tag = ''
                        if torch.cuda.is_available():
                            _pp = torch.cuda.get_device_properties(0)
                            import re as _re
                            _gpu_slug = _re.sub(r'[^a-z0-9]', '', _pp.name.lower())[:16]
                            _cur_sm_tag = f'_sm{_pp.major}{_pp.minor}_{_gpu_slug}'
                        if _cur_sm_tag:
                            import os.path as _osp2
                            _basename = _osp2.basename(trt_path)
                            if _cur_sm_tag not in _basename:
                                try: os.remove(trt_path)
                                except OSError: pass
                                raise RuntimeError(f'过期缓存 {_basename} 已删除，需重建')
                        trt     = self._trt
                        logger  = _get_subprocess_trt_logger()
                        runtime = trt.Runtime(logger)
                        with open(trt_path, 'rb') as f:
                            self._engine = runtime.deserialize_cuda_engine(f.read())
                        del runtime
                        if self._engine is None:
                            try: os.remove(trt_path)
                            except OSError: pass
                            raise RuntimeError('deserialize returned None')
                        self._context = self._engine.create_execution_context()
                        if self._context is None:
                            # raise RuntimeError('create_execution_context returned None')
                            print('[GFPGAN-TensorRT] ⚠️  create_execution_context() 失败'
                                  '（GPU 显存不足），回退 PyTorch 推理路径。')
                            print('[GFPGAN-TensorRT] 提示: 前序处理步骤可能占用了大量显存。'
                                  '可尝试减小 --gfpgan-batch-size 或移除 --gfpgan-trt。')
                            # 释放已加载的 engine，归还显存
                            self._engine = None
                            self._trt_ok = False
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            return  # 不抛异常，__init__ 中 self._trt_ok 保持 False，自动走 PyTorch 路径

                        try:
                            torch.cuda.synchronize(self.device)
                        except Exception as _ce:
                            self._cuda_context_dead = True
                            return
                        self._use_new_api = hasattr(self._engine, 'num_io_tensors')
                        if self._use_new_api:
                            for i in range(self._engine.num_io_tensors):
                                name = self._engine.get_tensor_name(i)
                                mode = self._engine.get_tensor_mode(name)
                                if mode == trt.TensorIOMode.INPUT:
                                    self._input_name = name
                                elif mode == trt.TensorIOMode.OUTPUT:
                                    self._output_name = name
                            print(f'[GFPGAN-TRT] TRT 10.x | in={self._input_name}, out={self._output_name}', flush=True)
                        else:
                            print('[GFPGAN-TRT] TRT 8.x API', flush=True)
                        self._trt_ok = True
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize(self.device)
                            _dt   = torch.float16 if self.use_fp16 else torch.float32
                            _B    = self._max_batch_size
                            _FULL = (_B, 3, 512, 512)
                            _inp  = torch.zeros(_FULL, dtype=_dt, device=self.device)
                            _out  = torch.zeros(_FULL, dtype=_dt, device=self.device)
                            _warmup_stream = torch.cuda.Stream(device=self.device)
                            _warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
                            if self._use_new_api:
                                self._context.set_input_shape(self._input_name, _FULL)
                                self._context.set_tensor_address(self._input_name, _inp.data_ptr())
                                self._context.set_tensor_address(self._output_name, _out.data_ptr())
                                self._context.execute_async_v3(stream_handle=_warmup_stream.cuda_stream)
                            else:
                                self._context.set_binding_shape(0, _FULL)
                                self._context.execute_async_v2(
                                    bindings=[_inp.data_ptr(), _out.data_ptr()],
                                    stream_handle=_warmup_stream.cuda_stream)
                            _warmup_stream.synchronize()
                            del _inp, _out, _warmup_stream
                            print('[GFPGAN-TensorRT] Warmup 通过', flush=True)
                        except Exception as _we:
                            print(f'[GFPGAN-TensorRT] Warmup 失败: {_we}', flush=True)
                            self._warmup_failed = True
                            self._cuda_context_dead = True
                            return

                    @property
                    def available(self):
                        return self._trt_ok

                    def infer(self, face_tensor):
                        if self._trt_stream is None:
                            self._trt_stream = torch.cuda.Stream(device=self.device)
                        B      = face_tensor.shape[0]
                        max_bs = self._max_batch_size
                        dtype  = torch.float16 if self.use_fp16 else torch.float32
                        _FULL  = (max_bs, 3, 512, 512)
                        if (not hasattr(self, '_inp_buf')
                                or self._inp_buf.shape != torch.Size(_FULL)
                                or self._inp_buf.dtype != dtype):
                            self._inp_buf = torch.empty(_FULL, dtype=dtype, device=self.device)
                            self._out_buf = torch.empty(_FULL, dtype=dtype, device=self.device)
                        self._inp_buf.zero_()
                        self._inp_buf[:B].copy_(face_tensor[:B])
                        self._trt_stream.wait_stream(torch.cuda.current_stream(self.device))
                        if self._use_new_api:
                            self._context.set_input_shape(self._input_name, _FULL)
                            self._context.set_tensor_address(self._input_name, self._inp_buf.data_ptr())
                            self._context.set_tensor_address(self._output_name, self._out_buf.data_ptr())
                            self._context.execute_async_v3(stream_handle=self._trt_stream.cuda_stream)
                        else:
                            self._context.set_binding_shape(0, _FULL)
                            self._context.execute_async_v2(
                                bindings=[self._inp_buf.data_ptr(), self._out_buf.data_ptr()],
                                stream_handle=self._trt_stream.cuda_stream)
                        self._trt_stream.synchronize()
                        return self._out_buf[:B].clone()

                gfpgan_trt_accel = GFPGANTRTAccelerator(
                    face_enhancer=face_enhancer, device=device,
                    cache_dir=trt_cache_dir or osp.join(os.getcwd(), '.trt_cache'),
                    gfpgan_weight=gfpgan_weight, max_batch_size=gfpgan_batch_size,
                    gfpgan_version=gfpgan_model, use_fp16=use_fp16)
                if gfpgan_trt_accel.available:
                    print('[GFPGANSubprocess] TRT 加速已启用（子进程版本）', flush=True)
                else:
                    print('[GFPGANSubprocess] TRT 初始化失败，回退 PyTorch', flush=True)
                    use_trt = False
            except Exception as e:
                print(f'[GFPGANSubprocess] TRT 初始化失败，回退 PyTorch: {e}', flush=True)
                use_trt = False
                gfpgan_trt_accel = None

        if gfpgan_trt_accel is not None and getattr(gfpgan_trt_accel, '_cuda_context_dead', False):
            print('[GFPGANSubprocess] TRT warmup 失败导致 CUDA context 损坏', flush=True)
            import time; time.sleep(0.5)
            import os as _os; _os._exit(0)

        _model_needs_gpu = False
        if use_trt and gfpgan_trt_accel is not None and gfpgan_trt_accel.available:
            print('[GFPGANSubprocess] TRT warmup 通过，迁移 GFPGANer 到 GPU...', flush=True)
            _model_needs_gpu = True
        elif use_trt and not (gfpgan_trt_accel is not None
                              and getattr(gfpgan_trt_accel, '_cuda_context_dead', False)):
            print('[GFPGANSubprocess] TRT 失败，迁移 GFPGANer 到 GPU...', flush=True)
            _model_needs_gpu = True

        if _model_needs_gpu:
            face_enhancer.gfpgan = face_enhancer.gfpgan.to(device)
            model = face_enhancer.gfpgan
            if use_fp16:
                face_enhancer.gfpgan = face_enhancer.gfpgan.half()
                model = face_enhancer.gfpgan
            face_enhancer.face_helper = None
            print('[GFPGANSubprocess] GFPGANer 已迁移到 GPU', flush=True)

        fp16_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_fp16 else contextlib.nullcontext()

        if gfpgan_trt_accel is not None and getattr(gfpgan_trt_accel, '_cuda_context_dead', False):
            if ready_event is not None:
                ready_event.set()
            import os as _os, time; time.sleep(0.5); _os._exit(0)

        if cuda_context_dead:
            pass
        elif use_trt and gfpgan_trt_accel is not None and gfpgan_trt_accel.available:
            print('[GFPGANSubprocess] TRT 成功，进入任务循环', flush=True)
            if ready_event is not None:
                ready_event.set()
        elif use_trt and (gfpgan_trt_accel is None or not gfpgan_trt_accel.available):
            print('[GFPGANSubprocess] TRT 失败但 context 正常，以 PyTorch 模式服务', flush=True)
            if ready_event is not None:
                ready_event.set()
        else:
            print('[GFPGANSubprocess] PyTorch 模式，进入任务循环', flush=True)
            if ready_event is not None:
                ready_event.set()

        # ── FIX-SHM-IPC (优化3): Attach 到主进程创建的共享内存 ────────
        import multiprocessing.shared_memory as shm
        _shm_inputs = []
        _shm_outputs = []
        _shm_available = False
        _shm_max_faces = 64
        _shm_face_shape = (512, 512, 3)
        if shm_input_names and shm_output_names:
            try:
                for _sname in shm_input_names:
                    _shm_inputs.append(shm.SharedMemory(name=_sname))
                for _sname in shm_output_names:
                    _shm_outputs.append(shm.SharedMemory(name=_sname))
                _shm_available = True
                print(f'[GFPGANSubprocess] 共享内存 attach 成功 '
                      f'({len(_shm_inputs)} input slots + '
                      f'{len(_shm_outputs)} output slots)', flush=True)
            except Exception as _shm_e:
                print(f'[GFPGANSubprocess] 共享内存 attach 失败，回退 pickle: '
                      f'{_shm_e}', flush=True)
                _shm_available = False
                _shm_inputs = []
                _shm_outputs = []
        else:
            print('[GFPGANSubprocess] 未传入共享内存名称，使用 pickle 传输',
                  flush=True)

        while True:
            try:
                task = task_queue.get(timeout=1)
            except queue.Empty:
                continue
            if task is None:
                break

            if isinstance(task, tuple) and len(task) == 2 and task[0] == '__validate__':
                _val_id = task[1]
                if use_trt and gfpgan_trt_accel is not None and gfpgan_trt_accel.available:
                    print('[GFPGANSubprocess] post-SR warmup（真实显存压力下）...', flush=True)
                    try:
                        torch.cuda.empty_cache()
                        _vdt = torch.float16 if use_fp16 else torch.float32
                        _vdmy = torch.zeros(gfpgan_batch_size, 3, 512, 512, dtype=_vdt, device=device)
                        _vout = gfpgan_trt_accel.infer(_vdmy)
                        del _vdmy, _vout
                        torch.cuda.synchronize(device)
                        print('[GFPGANSubprocess] post-SR warmup 通过，TRT 推理正式启用', flush=True)
                        result_queue.put(('__validate__', _val_id, True), timeout=5.0)
                    except Exception as _ve:
                        _ve_str = str(_ve).lower()
                        _ctx_dead = any(kw in _ve_str for kw in (
                            'illegal memory', 'cudaerrorillegaladdress',
                            'illegal instruction', 'prior launch failure',
                            'acceleratorerror', 'cudaerror'))
                        _is_oom = 'out of memory' in _ve_str
                        if _ctx_dead:
                            # 区分是 OOM 还是真正的 context 损坏
                            if 'out of memory' in _ve_str:
                                # OOM: 降级到 PyTorch，继续服务
                                print('[GFPGANSubprocess] post-SR OOM，降级 PyTorch...')
                                gfpgan_trt_accel._trt_ok = False
                                use_trt = False
                                result_queue.put(('__validate__', _val_id, False), timeout=5.0)
                            else:
                                # 真正的 context 损坏: 必须退出
                                result_queue.put(('__validate__', _val_id, False), timeout=5.0)
                                os._exit(0)
                        else:
                            # OOM 或其他可恢复错误：降级 PyTorch，子进程继续服务
                            print(f'[GFPGANSubprocess] post-SR warmup 失败（{"OOM" if _is_oom else "其他"}），'
                                  f'降级 PyTorch 路径继续: {_ve}', flush=True)
                            gfpgan_trt_accel._trt_ok = False
                            use_trt = False
                            result_queue.put(('__validate__', _val_id, False), timeout=5.0)
                else:
                    # TRT 从未成功初始化（build 失败 / SM 不支持 / .trt 不存在）
                    # 子进程已在 PyTorch FP16 路径服务，直接回报 False，无需任何处理
                    print('[GFPGANSubprocess] post-SR validate: TRT 未初始化，'
                          '子进程以 PyTorch FP16 路径服务', flush=True)
                    result_queue.put(('__validate__', _val_id, False), timeout=5.0)
                continue

            if isinstance(task, tuple) and len(task) == 2 and task[0] == '__pause__':
                _pause_duration = task[1]
                print(f'[GFPGANSubprocess] 收到暂停信号，休眠 {_pause_duration}s 释放显存...', flush=True)
                torch.cuda.empty_cache()  # 立即释放显存
                time.sleep(_pause_duration)
                torch.cuda.empty_cache()  # 再次清理
                print(f'[GFPGANSubprocess] 暂停结束，恢复处理', flush=True)
                continue

            # ── FIX-SHM-IPC (优化3): 兼容共享内存和 pickle 两种传输格式 ──
            # 共享内存格式: (task_id, n_faces, slot_id) — 3-tuple, slot_id 是 int
            # pickle 格式:  (task_id, crops_np)          — 2-tuple, crops_np 是 list
            _use_shm_output = False
            _shm_slot_id = -1
            if (isinstance(task, tuple) and len(task) == 3
                    and isinstance(task[1], int) and isinstance(task[2], int)):
                # 共享内存路径
                task_id, _n_faces, _shm_slot_id = task
                if _shm_available and _shm_slot_id < len(_shm_inputs):
                    _shm_face_shape = (512, 512, 3)
                    _shm_max_faces = 64
                    _inp_buf = np.ndarray(
                        (_shm_max_faces, *_shm_face_shape), dtype=np.uint8,
                        buffer=_shm_inputs[_shm_slot_id].buf)
                    crops_np = [_inp_buf[i].copy() for i in range(_n_faces)]
                    _use_shm_output = True
                else:
                    # shm attach 失败，此 task 无法处理
                    print(f'[GFPGANSubprocess] 共享内存 slot {_shm_slot_id} 不可用，跳过',
                          flush=True)
                    try:
                        result_queue.put((task_id, []), timeout=5.0)
                    except queue.Full:
                        pass
                    continue
            else:
                # pickle 路径（原有逻辑）
                task_id, crops_np = task

            # 转换为 tensor
            crops_tensor = []
            for crop in crops_np:
                t = img2tensor(crop / 255., bgr2rgb=True, float32=True)
                _tv_normalize(t, (0.5,0.5,0.5), (0.5,0.5,0.5), inplace=True)
                crops_tensor.append(t)
            if not crops_tensor:
                result_queue.put((task_id, []), timeout=5.0)
                continue

            # 分批推理（支持 TRT 和 PyTorch 两种路径）
            all_out = []
            sub_bs = gfpgan_batch_size
            i = 0
            while i < len(crops_tensor):
                sub = crops_tensor[i:i+sub_bs]
                sub_batch = torch.stack(sub).to(device)
                try:
                    with torch.no_grad():
                        if use_trt and gfpgan_trt_accel is not None and gfpgan_trt_accel.available:
                            if use_fp16: sub_batch = sub_batch.half()
                            out = gfpgan_trt_accel.infer(sub_batch)
                            out = out.float() if out.dtype != torch.float32 else out
                            # --- [修复] 处理 FP16 数值溢出 (NaN/Inf) 并补充缺失的 Weight Blending ---
                            # 1. 将 NaN 转为 0，Inf 转为边界值，防止转换图像时出现噪点
                            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
                            
                            # 2. 强制截断到 GFPGAN 预期的 -1 到 1 范围
                            out = torch.clamp(out, min=-1.0, max=1.0)
                            
                            # 3. 补充缺失的 Weight Blending (权重融合)
                            # TRT 导出的是纯网络前向，缺少这一步会导致结果不一致且容易产生噪声
                            if abs(gfpgan_weight - 1.0) > 1e-6:
                                _sub_b = sub_batch.float() if sub_batch.dtype == torch.float16 else sub_batch
                                out = gfpgan_weight * out + (1.0 - gfpgan_weight) * _sub_b
                            
                            # 4. 混合后再次 clamp 确保范围正确
                            out = torch.clamp(out, min=-1.0, max=1.0)
                            # ------------------------------------------------------------------------
                        else:
                            with fp16_ctx:
                                out = model(sub_batch, return_rgb=False)
                                if isinstance(out, (tuple, list)): out = out[0]
                            out = out.float()
                            # 手动 weight 融合（与 TRT wrapper 行为对齐）
                            if abs(gfpgan_weight - 1.0) > 1e-6:
                                # sub_batch 在 [-1,1] RGB 空间，out 也在 [-1,1] RGB 空间
                                out = gfpgan_weight * out + (1.0 - gfpgan_weight) * sub_batch.float()
                    all_out.extend(out.unbind(0))
                    i += len(sub)
                except RuntimeError as e:
                    error_str = str(e).lower()
                    if 'out of memory' in error_str and sub_bs > 1:
                        sub_bs = max(1, sub_bs // 2)
                        torch.cuda.empty_cache()
                        print(f'[GFPGANSubprocess] GFPGAN OOM，sub_bs 降级至 {sub_bs}，重试...', flush=True)
                    elif 'cudaerrorillegaladdress' in error_str or 'illegal memory' in error_str:
                        # CUDA 上下文损坏，切换到 PyTorch 路径
                        print(f'[GFPGANSubprocess] CUDA 非法内存访问，切换到 PyTorch 路径: {e}')
                        use_trt = False
                        gfpgan_trt_accel = None
                        torch.cuda.empty_cache()
                    else:
                        all_out.extend([None] * len(sub))
                        i += len(sub)
                        torch.cuda.empty_cache()
                finally:
                    del sub_batch

            # 将输出转换回 numpy
            restored = []
            for out_t in all_out:
                if out_t is None:
                    restored.append(None)
                else:
                    # --- 修复开始：处理 TRT FP16 导致的 NaN 和数值溢出 ---
                    # 1. 将 NaN 转换为 0，将正负无穷转换为对应边界值
                    out_t = torch.nan_to_num(out_t, nan=0.0, posinf=1.0, neginf=-1.0)
                    # 2. 强制截断到 GFPGAN 预期的 -1 到 1 范围，防止超出该范围的值在后处理时产生异常色块
                    out_t = torch.clamp(out_t, min=-1.0, max=1.0)
                    # --- 修复结束 ---

                    img = tensor2img(out_t, rgb2bgr=True, min_max=(-1, 1))  # 现在正确：RGB → BGR
                    restored.append(img.astype('uint8'))

            # ── FIX-SHM-IPC (优化3): 共享内存写回或 pickle 返回 ──────
            if _use_shm_output and _shm_slot_id >= 0 and _shm_slot_id < len(_shm_outputs):
                # 共享内存路径：将结果写入 output shm，queue 只传 (task_id, n_restored)
                _face_shape = (512, 512, 3)
                _max_faces = 64
                _out_buf = np.ndarray(
                    (_max_faces, *_face_shape), dtype=np.uint8,
                    buffer=_shm_outputs[_shm_slot_id].buf)
                _n_valid = 0
                for i, r in enumerate(restored):
                    if r is not None and i < _max_faces:
                        if r.shape == _face_shape:
                            _out_buf[i] = r
                        else:
                            # 尺寸不匹配时 resize（理论上不应发生）
                            import cv2 as _cv_w
                            _out_buf[i] = _cv_w.resize(
                                r, (_face_shape[1], _face_shape[0]))
                        _n_valid += 1
                    elif i < _max_faces:
                        _out_buf[i] = 0  # None 位置清零
                try:
                    # 只传元数据：(task_id, 有效结果数量)
                    # AsyncGFPGANDispatcher._collect_loop 收到 int 型 result
                    # 时知道从 output shm 读取
                    result_queue.put((task_id, len(restored)), timeout=5.0)
                    # result_queue.put((task_id, _n_valid), timeout=5.0)
                except queue.Full:
                    pass
            else:
                # pickle 路径（原有逻辑）
                try:
                    result_queue.put((task_id, restored), timeout=5.0)
                except queue.Full:
                    pass

        # ── FIX-SHM-IPC (优化3): 关闭共享内存 attach（不 unlink，由主进程负责）──
        for _s in _shm_inputs + _shm_outputs:
            try:
                _s.close()
            except Exception:
                pass

        if gfpgan_trt_accel is not None:
            gfpgan_trt_accel._trt_ok = False   # 禁用，不 safe_destroy
        try:
            del face_enhancer, model
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        import os as _os_worker_exit
        _os_worker_exit._exit(0)   # FIX-WORKER-EXIT: 跳过 Python GC，子进程立即退出

    def infer(self, crops_list):
        if not self.process or not self.process.is_alive():
            return [None] * len(crops_list)
        task_id = id(crops_list)
        try:
            self.task_queue.put((task_id, crops_list), timeout=10.0)
        except queue.Full:
            return [None] * len(crops_list)
        while True:
            try:
                res = self.result_queue.get(timeout=60)
            except queue.Empty:
                raise RuntimeError('GFPGAN子进程超时无响应')
            if isinstance(res, tuple) and len(res) == 3 and res[0] == '__validate__':
                continue
            res_id, result = res
            if res_id == task_id:
                return result
            self.result_queue.put((res_id, result))

    def post_sr_validate(self) -> bool:
        if not self.process or not self.process.is_alive():
            return False
        val_id = id(self)
        try:
            self.task_queue.put(('__validate__', val_id), timeout=5.0)
        except queue.Full:
            return False
        deadline = time.time() + 180
        while time.time() < deadline:
            if not self.process.is_alive():
                return False
            try:
                res = self.result_queue.get(timeout=5)
            except queue.Empty:
                continue
            if isinstance(res, tuple) and len(res) == 3 and res[0] == '__validate__' and res[1] == val_id:
                return res[2]
            self.result_queue.put(res)
        return False

    def pause(self, duration: float = 5.0):
        """通知 GFPGAN 子进程暂停处理一段时间。"""
        if not self.process or not self.process.is_alive():
            return
        try:
            self.task_queue.put(('__pause__', duration), timeout=2.0)
            print(f'[GFPGANSubprocess] 已发送暂停信号 ({duration}s)', flush=True)
        except queue.Full:
            pass

    def close(self):
        """发送终止信号并等待子进程结束
        
        FIX-CLOSE-KILL:
          原实现：terminate() 之后不再 join，不调 kill()，直接 task_queue.close()。
          若子进程深陷 CUDA C 调用忽略 SIGTERM，进程活着但 close() 已经返回，
          主进程继续但子进程一直驻留显存（8.2GB 不释放）。
          修复：terminate() → join(5s) → kill() 三级终止，确保子进程一定死透。
        """
        if self.process and self.process.is_alive():
            # 1. 发送正常终止信号（子进程走 os._exit(0) 路径，此处通常立即返回）
            try:
                self.task_queue.put(None, timeout=3)   # FIX: 加 timeout 防止满队列死锁
            except Exception:
                pass
            # 2. 等待正常退出（FIX-WORKER-EXIT 后子进程会立刻 _exit(0)，join 几乎即时）
            self.process.join(timeout=15)
            # 3. 仍存活 → SIGTERM
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
            # 4. 仍存活 → SIGKILL（FIX-CLOSE-KILL: 最终兜底，无视 CUDA C 调用）
            if self.process.is_alive():
                print('[GFPGANSubprocess] 子进程未响应 SIGTERM，发送 SIGKILL...', flush=True)
                self.process.kill()
                self.process.join(timeout=5)
                
        # FIX-SHM-IPC (优化3): 清理共享内存
        if self.shm_buf is not None:
            self.shm_buf.close()
            self.shm_buf = None

        try:
            self.task_queue.close()
        except Exception:
            pass
        try:
            self.result_queue.close()
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# SharedMemoryDoubleBuffer（优化3：零拷贝 IPC）
# 预分配双缓冲共享内存，替代 pickle 序列化，减少 GFPGAN IPC 开销 20-30%。
# 双缓冲配合 AsyncGFPGANDispatcher（优化5B）：最多 2 批 in-flight，
# 各占一个 slot，由 task_queue maxsize=2 保证不会同时写同一 slot。
# ─────────────────────────────────────────────────────────────────────────────

class SharedMemoryDoubleBuffer:
    """双缓冲共享内存，用于主进程与 GFPGAN 子进程之间的零拷贝数据传输。

    FIX-SHM-IPC: 原实现通过 multiprocessing.Queue 传递人脸 crops，
    底层 pickle 序列化 20+ 张 512×512×3 ≈ 15MB，序列化+反序列化 ~200-400ms。
    修复：使用 POSIX 共享内存预分配双缓冲区，task_queue 只传元数据
    (task_id, n_faces, slot_id)，不传数据本身。

    FIX-SLOT-POOL: 替代原 _slot_counter 盲目轮转。
    使用 queue.Queue 池管理 slot 的借出/归还，确保：
      1. slot 仅在空闲时被分配（acquire_slot）
      2. slot 仅在对应任务结果取回后才归还（release_slot）
      3. queue.Queue 是线程安全的，无需额外加锁
      4. acquire_slot 的阻塞天然提供背压，不再需要外部 _MAX_IN_FLIGHT 常量
    """

    N_SLOTS = 2          # 双缓冲
    MAX_FACES = 64       # 单 slot 最大人脸数（覆盖极端密集场景）
    FACE_SHAPE = (512, 512, 3)

    def __init__(self):
        face_bytes = int(np.prod(self.FACE_SHAPE))
        slot_bytes = self.MAX_FACES * face_bytes
        self._input_shms = []
        self._output_shms = []
        for _ in range(self.N_SLOTS):
            self._input_shms.append(shm.SharedMemory(create=True, size=slot_bytes))
            self._output_shms.append(shm.SharedMemory(create=True, size=slot_bytes))
        # FIX-SLOT-POOL: Queue 池管理替代盲目轮转计数器
        # 初始时所有 slot 均可用，放入池中
        self._slot_pool = queue.Queue(maxsize=self.N_SLOTS)
        for i in range(self.N_SLOTS):
            self._slot_pool.put(i)

    # ── 名称属性（传给子进程 _worker 用于 attach）──────────────────
    @property
    def input_names(self) -> List[str]:
        return [s.name for s in self._input_shms]

    @property
    def output_names(self) -> List[str]:
        return [s.name for s in self._output_shms]

    def acquire_slot(self, timeout: float = 30.0) -> int:
        """从池中借出一个空闲 slot，阻塞直到有可用 slot。

        替代原 next_slot() 的盲目轮转。
        注意：在与 _pending_tasks 出队同一线程中调用时，
        应优先使用 try_acquire_slot() + 手动排空 pending 的模式，
        避免单线程自死锁（acquire 阻塞 → 无人 release）。
        """
        try:
            return self._slot_pool.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(
                f'无法在 {timeout}s 内获取空闲 slot，'
                f'可能存在 slot 泄漏（未调用 release_slot）')

    def try_acquire_slot(self) -> Optional[int]:
        """非阻塞尝试获取空闲 slot，无可用则返回 None。

        用于 _process_gfpgan 单线程场景：先 try_acquire，
        失败则手动排空 _pending_tasks（释放 slot），再重试。
        """
        try:
            return self._slot_pool.get_nowait()
        except queue.Empty:
            return None

    def release_slot(self, slot: int):
        """归还 slot 到池中。slot 为 None 时无操作（幂等）。"""
        if slot is not None:
            self._slot_pool.put(slot)

    def write_input(self, slot: int, crops: List[np.ndarray]) -> int:
        """将 crops 写入指定 slot 的输入共享内存，返回实际写入数量"""
        n = min(len(crops), self.MAX_FACES)
        buf = np.ndarray(
            (self.MAX_FACES, *self.FACE_SHAPE), dtype=np.uint8,
            buffer=self._input_shms[slot].buf)
        for i in range(n):
            c = crops[i]
            if c.shape == self.FACE_SHAPE:
                buf[i] = c
            else:
                # 尺寸不匹配（理论上不应发生，align_warp_face 输出固定 512x512）
                import cv2 as _cv
                buf[i] = _cv.resize(c, (self.FACE_SHAPE[1], self.FACE_SHAPE[0]))
        return n

    def read_output(self, slot: int, n: int) -> List[np.ndarray]:
        """从指定 slot 的输出共享内存读取 n 个结果（深拷贝）"""
        buf = np.ndarray(
            (self.MAX_FACES, *self.FACE_SHAPE), dtype=np.uint8,
            buffer=self._output_shms[slot].buf)
        return [buf[i].copy() for i in range(n)]

    def close(self):
        """清理共享内存（主进程负责 unlink）"""
        for s_list in [self._input_shms, self._output_shms]:
            for s in s_list:
                try:
                    s.close()
                except Exception:
                    pass
                try:
                    s.unlink()
                except Exception:
                    pass

# ─────────────────────────────────────────────────────────────────────────────
# AsyncGFPGANDispatcher（优化5B：异步 GFPGAN 调度）
# 将 GFPGAN IPC 从同步调用改为异步提交+后台收集。
# 主线程提交人脸 crops 后不等结果，立即返回，允许 SR 继续下一批推理。
# 后台 _collector 线程从 result_queue 收集结果。
# wait_result() 按 task_id 等待特定批次完成。
# ─────────────────────────────────────────────────────────────────────────────

class AsyncGFPGANDispatcher:
    """异步 GFPGAN 调度器

    FIX-ASYNC-GFPGAN: 解耦 SR → GFPGAN → Writer 的串行依赖。
    SR 推理完成后，将人脸 crops 异步提交给 GFPGAN 子进程，
    不等待结果即开始下一批 SR 推理。GPU 利用率从锯齿状脉冲
    变为持续高占用（SR 和 GFPGAN 在同一 GPU 上时分复用）。

    线程安全：_lock + _cv 保护 _results/_pending；
    _validate_lock + _validate_cv 保护 _validate_results。
    """

    def __init__(self, subprocess_obj, shm_buf: Optional[SharedMemoryDoubleBuffer] = None):
        self.subprocess = subprocess_obj
        self.shm_buf = shm_buf          # FIX-SHM-IPC: 共享内存双缓冲
        self._results: Dict[int, Any] = {}
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._validate_results: Dict[int, bool] = {}
        self._validate_lock = threading.Lock()
        self._validate_cv = threading.Condition(self._validate_lock)
        self._running = True
        # 后台结果收集线程
        self._collector = threading.Thread(
            target=self._collect_loop, daemon=True,
            name='async_gfpgan_collector')
        self._collector.start()

    def _collect_loop(self):
        """后台线程：持续从子进程 result_queue 收集结果"""
        while self._running:
            try:
                res = self.subprocess.result_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if res is None:
                break
            # ── __validate__ 响应路由 ──
            if (isinstance(res, tuple) and len(res) == 3
                    and res[0] == '__validate__'):
                _, val_id, ok = res
                with self._validate_lock:
                    self._validate_results[val_id] = ok
                    self._validate_cv.notify_all()
                continue
            # ── 正常推理结果 ──
            if isinstance(res, tuple) and len(res) == 2:
                task_id, result = res
                with self._lock:
                    self._results[task_id] = result
                    self._cv.notify_all()

    # ── 异步提交（非阻塞）──────────────────────────────────────────
    def _submit_blocking(self, crops_list: List[np.ndarray], task_id: Optional[int] = None) -> Tuple[int, Optional[int]]:
        """不用于 _process_gfpgan（后者需要非阻塞+排空模式）

        非阻塞提交 crops，返回 (task_id, slot)。
        使用共享内存时，先 acquire slot → 写入 shm → 发元数据；否则走 pickle。
        调用方负责在结果取回后调用 shm_buf.release_slot(slot) 归还槽位。

        FIX-SLOT-POOL: 替代原 next_slot() 盲目轮转。
        提交失败时自动归还 slot，不泄漏。
        """
        if task_id is None:
            task_id = id(crops_list)
        _slot = None

        if (self.shm_buf is not None
                and len(crops_list) <= SharedMemoryDoubleBuffer.MAX_FACES):
            # FIX-SLOT-POOL: 共享内存路径 — acquire → write → put
            _slot = self.shm_buf.acquire_slot(timeout=30.0)
            try:
                n = self.shm_buf.write_input(_slot, crops_list)
                self.subprocess.task_queue.put(
                    (task_id, n, _slot), timeout=10.0)
            except Exception:
                # 提交失败，立即归还 slot 防止泄漏
                self.shm_buf.release_slot(_slot)
                _slot = None
                raise
        else:
            # Fallback: pickle 路径（crops 过多或无 shm）
            self.subprocess.task_queue.put(
                (task_id, crops_list), timeout=10.0)

        return task_id, _slot
    # ── 阻塞等待结果 ──────────────────────────────────────────────

    def wait_result(self, task_id: int, timeout: float = 120.0,
                    slot: Optional[int] = None) -> Optional[list]:
        """阻塞等待 task_id 的结果。
        共享内存路径时 slot 非 None，从 output_shm 读取结果。
        """
        deadline = time.time() + timeout
        with self._lock:
            while task_id not in self._results:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cv.wait(timeout=min(1.0, remaining))
                if not self._running:
                    return None
            raw = self._results.pop(task_id)

        # FIX-SHM-IPC: 如果 raw 是 int（n_faces），从 output_shm 读取
        if isinstance(raw, int) and self.shm_buf is not None and slot is not None:
            # return self.shm_buf.read_output(slot, raw)
            # 调整读取后的处理：将全零视为 None
            restored_list = self.shm_buf.read_output(slot, raw)
            restored_list = [r if r.any() else None for r in restored_list]
            return restored_list
        return raw  # pickle 路径直接返回

    # ── validate 结果等待 ─────────────────────────────────────────

    def wait_validate(self, val_id: int, timeout: float = 180.0) -> bool:
        """阻塞等待 validate 结果"""
        deadline = time.time() + timeout
        with self._validate_lock:
            while val_id not in self._validate_results:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                if (self.subprocess.process is not None
                        and not self.subprocess.process.is_alive()):
                    return False
                self._validate_cv.wait(timeout=min(5.0, remaining))
            return self._validate_results.pop(val_id)

    def submit_validate(self, val_id: int):
        """非阻塞提交 validate 请求"""
        try:
            self.subprocess.task_queue.put(
                ('__validate__', val_id), timeout=5.0)
        except queue.Full:
            pass

    def in_flight(self) -> int:
        """当前在途的推理任务数"""
        with self._lock:
            # result_queue 中的结果已被 _collector 收入 _results
            # task_queue 中的 + 正在处理的 = subprocess.task_queue.qsize() + 1(maybe)
            return self.subprocess.task_queue.qsize()

    def close(self):
        self._running = False
        with self._lock:
            self._cv.notify_all()
        with self._validate_lock:
            self._validate_cv.notify_all()
        if self._collector.is_alive():
            self._collector.join(timeout=5.0)

class GPUMemoryPool:
    """流水线并发槽计数器（纯信号量）"""

    def __init__(self, max_batches: int = 4, batch_size: int = 4,
                 img_size: Tuple[int, int] = (540, 960), device: str = 'cuda'):
        self.max_batches = max_batches
        self.batch_size = batch_size
        self.img_size   = img_size
        self.device     = device
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
    """深度流水线优化器 - 4级并行处理
    
    FIX-DET-THRESHOLD: 支持人脸检测置信度阈值过滤
    FIX-ADAPTIVE-BATCH: 基于人脸密度自适应调整批处理大小
    FIX-GPU-PREFETCH: SR 推理完成后预取下一批 H2D 传输

    FIX-READ-WATCHDOG: _read_frames 增加 reader 探活 + 连续超时看门狗，
                       并在 finally 中不再置 self.running=False，让哨兵沿流水线自然传播
    FIX-WRITE-WATCHDOG: _write_frames 增加全流水线空转看门狗，120s 全空且无哨兵
                        → dump 所有线程栈 + 强制退出
    """
    
    def __init__(self, upsampler, face_enhancer, args, device, trt_accel=None,
                 input_h: int = 540, input_w: int = 960):
        self.upsampler = upsampler
        self.face_enhancer = face_enhancer
        self.args = args
        self.device = device
        
        # 优化参数
        self.optimal_batch_size = min(args.batch_size, 24)  # 限制最大batch_size
        
        # 深度缓冲队列（4级流水线）
        self.frame_queue = queue.Queue(maxsize=48)        # 原始帧队列
        self.detect_queue = queue.Queue(maxsize=32)       # 检测结果队列  
        self.sr_queue = queue.Queue(maxsize=16)           # SR结果队列
        self.gfpgan_queue = queue.Queue(maxsize=16)       # GFPGAN结果队列
        
        # GPU内存池（减少内存分配以避免OOM）
        self.memory_pool = GPUMemoryPool(
            max_batches=8,                          # 足够深，避免spin-wait
            batch_size=self.optimal_batch_size,     # 匹配实际 batch 大小
            img_size=(input_h, input_w),          # 从视频元数据读取，不要硬编码
            device=device
        )
                
        # 优化的线程池
        self.detect_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix='opt_detect'
        )
        self.paste_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix='opt_paste'
        )
        
        # 异步CUDA流
        self.transfer_stream = torch.cuda.Stream(device=device)
        self.sr_stream = torch.cuda.Stream(device=device)
        self.gfpgan_stream = torch.cuda.Stream(device=device)
        
        # 性能监控
        self.meter = ThroughputMeter()
        self.timing = []
        
        # 控制标志
        self.running = True
        self.trt_accel        = trt_accel
        self.cuda_graph_accel = None
        # FIX-VERBOSE: 人脸检测累计统计
        self._face_frames_total = 0
        self._face_count_total  = 0
        
        # FIX-DET-THRESHOLD: 人脸检测置信度阈值
        self.face_det_threshold = getattr(args, 'face_det_threshold', 0.5)
        # FIX-DET-THRESHOLD: 被阈值过滤的人脸总数（用于性能监控日志）
        self._face_filtered_total = 0
        
        # ── FIX-ADAPTIVE-BATCH: 基于人脸密度的自适应批处理 ──────────────
        # 核心思路：人脸稀疏时 GFPGAN 快速完成 → GPU 空闲 → 增大 SR 批次提升吞吐
        #           人脸密集时 GFPGAN 是瓶颈 → 减小 SR 批次 → 省出显存给 GFPGAN
        self._face_density_ema = 0.0                    # 人脸密度指数移动平均（每帧平均人脸数）
        self._face_density_alpha = 0.3                   # EMA 平滑系数（0-1，越大越敏感）
        self._low_face_threshold = 2.0                   # 低人脸密度阈值（每帧 <2 个人脸 = 稀疏）
        self._high_face_threshold = 5.0                  # 高人脸密度阈值（每帧 >5 个人脸 = 密集）
        self._base_batch_size = self.optimal_batch_size  # 用户指定的基础批处理大小（不随 OOM 变化）
        self._max_adaptive_batch = min(self._base_batch_size * 2, 12)  # 自适应上限
        self._min_adaptive_batch = max(2, self._base_batch_size // 2)  # 自适应下限
        self._adaptive_batch_lock = threading.Lock()     # 线程安全锁
        self._adaptive_read_batch_size = self.optimal_batch_size  # 当前自适应读取批处理大小
        self._enable_adaptive_batch = getattr(args, 'adaptive_batch', True)
        # ── FIX-ADAPTIVE-BATCH 结束 ──────────────────────────────────
        
        # 人脸检测helper
        self.detect_helper = _make_detect_helper(face_enhancer, device) if face_enhancer else None

        # FIX-EARLY-SPAWN: 优先使用预启动的 GFPGAN 子进程
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
                try: _prestarted.close()
                except Exception as e: print(f'[优化架构] 关闭死亡子进程错误: {e}')
                args._early_gfpgan_subprocess = None
                self.gfpgan_subprocess = None

        if (self.gfpgan_subprocess is None and 
            getattr(args, 'gfpgan_trt', False) and 
            face_enhancer is not None):
            if not getattr(args, '_gfpgan_trt_failed', False):
                print('[优化架构] 启用子进程GFPGAN TRT加速（非预启动路径）')
                try:
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

        # ── FIX-ASYNC-GFPGAN (优化5B): 创建异步调度器 ────────────────
        self._async_dispatcher: Optional[AsyncGFPGANDispatcher] = None
        # 调度器在 optimize_pipeline() 中 GFPGAN 子进程就绪后创建

        # ── FIX id() 复用导致异步结果错配
        self._task_id_counter = 0
        self._task_id_lock = threading.Lock()

        # 线程句柄（供 close() / join 使用）
        self._read_thread = None
        self._detect_thread = None
        self._sr_thread = None
        self._gfpgan_thread = None

        # FIX-READ-WATCHDOG: 上游 reader 句柄（由 optimize_pipeline 填充，
        # 供 _read_frames 探活、_write_frames 看门狗显示 prefetch 水位使用）
        self.reader = None
    
    # ────────────────────────────────────────────────────────────────────
    # FIX-WRITE-WATCHDOG: 统一的 reader / queue 状态辅助方法
    # ────────────────────────────────────────────────────────────────────
    def _get_reader_state(self):
        """获取上游 reader 的当前水位/状态。返回 (p_size, p_cap, alive, eof_sent, produced)"""
        if self.reader is None:
            return (-1, 0, False, False, -1)
        try:
            p_size = self.reader.get_queue_size()
            p_cap = self.reader.get_queue_capacity()
            alive = self.reader.is_reader_alive()
            eof_sent = self.reader.is_eof_sent()
            produced = (self.reader.get_frames_produced()
                        if hasattr(self.reader, 'get_frames_produced') else -1)
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
                f"G:{self.gfpgan_queue.qsize()}")

    def _dump_all_queues(self):
        """FIX-WRITE-WATCHDOG: 遍历对象属性，dump 所有 Queue/deque 的水位"""
        seen = set()
        lines = []

        def visit(obj, path, depth=0):
            if depth > 3 or id(obj) in seen:
                return
            seen.add(id(obj))
            if isinstance(obj, (queue.Queue, queue.LifoQueue, queue.PriorityQueue)):
                try:
                    lines.append(f"  {path}: {obj.qsize()}/{obj.maxsize}  "
                                 f"[{type(obj).__name__}]")
                except Exception as e:
                    lines.append(f"  {path}: <err {e}>")
                return
            if isinstance(obj, deque):
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
        """FIX-WRITE-WATCHDOG: 打印所有线程的栈，用于死锁现场诊断"""
        try:
            for tid, frame in sys._current_frames().items():
                name = next((t.name for t in threading.enumerate()
                             if t.ident == tid), str(tid))
                sys.stderr.write(f"\n--- Thread {name} ({tid}) ---\n")
                traceback.print_stack(frame, file=sys.stderr)
            sys.stderr.flush()
        except Exception as e:
            print(f"[Pipeline] dump 线程栈失败: {e}", flush=True)
    
    def optimize_pipeline(self, reader, writer, pbar, total_frames):
        """运行优化的深度流水线"""

        # FIX-READ-WATCHDOG: 保存 reader 引用，供 _read_frames 探活、
        # _write_frames 看门狗显示 prefetch 水位使用
        self.reader = reader

        print("[优化架构] 启动深度流水线处理...")
        print(f"[优化架构] 队列深度: "
              f"P{reader.get_queue_capacity() if reader is not None else '?'}"
              f"/F{self.frame_queue.maxsize}"
              f"/D{self.detect_queue.maxsize}"
              f"/S{self.sr_queue.maxsize}"
              f"/G{self.gfpgan_queue.maxsize}")
        print(f"[优化架构] 内存池: {self.memory_pool.max_batches}批次")
        print(f"[优化架构] 最优batch_size: {self.optimal_batch_size}")
        # FIX-DET-THRESHOLD: 打印置信度阈值
        print(f"[优化架构] 人脸检测置信度阈值: {self.face_det_threshold}")
        # FIX-ADAPTIVE-BATCH: 打印自适应批处理配置
        if self._enable_adaptive_batch:
            print(f"[优化架构] 自适应批处理: 开启 (范围 {self._min_adaptive_batch}~{self._max_adaptive_batch}, "
                  f"低密度阈值={self._low_face_threshold}, 高密度阈值={self._high_face_threshold})")
        else:
            print(f"[优化架构] 自适应批处理: 关闭")

        if self.gfpgan_subprocess is not None:
            print('[优化架构] 等待 GFPGAN Inference 进程初始化（加载 .trt + warmup）...')
            max_elapsed = 2700   # 最多等 45 分钟
            deadline = time.time() + max_elapsed
            ready = False
            _poll_interval = 5   # FIX-FAST-DETECT: 每 5s 轮询一次进程存活状态
            _report_every  = 300  # 每 300s 打印一次等待进度
            _last_report   = time.time() - _report_every  # 立即允许第一次打印
            while time.time() < deadline:
                # FIX-FAST-DETECT: 优先检测进程是否已死（崩溃），避免傻等超时
                if not self.gfpgan_subprocess.process.is_alive():
                    exitcode = self.gfpgan_subprocess.process.exitcode
                    if exitcode == 0:
                        print('[优化架构] GFPGAN 子进程因 CUDA context 污染主动退出，'
                              '降级到主进程内 GFPGAN（PyTorch FP16）路径')
                    else:
                        print(f'[优化架构] GFPGAN 子进程意外退出（exitcode={exitcode}），回退 PyTorch')
                    break
                if self.gfpgan_subprocess.ready_event.wait(timeout=_poll_interval):
                    # ready_event 有两种触发方：
                    #   (A) 初始化成功 → 进入 task loop，进程持续存活
                    #   (B) warmup 失败 → 进程在 ready_event.set() 后 sleep(0.5) 再 os._exit(0)
                    # 等待 1.0s（> 0.5s 子进程 sleep），再检查进程是否仍然存活以区分两者。
                    time.sleep(1.0)
                    if not self.gfpgan_subprocess.process.is_alive():
                        exitcode = self.gfpgan_subprocess.process.exitcode
                        if exitcode == 0:
                            print('[优化架构] GFPGAN TRT warmup 失败，子进程主动退出（exitcode=0），'
                                  '降级到主进程内 GFPGAN PyTorch 路径')
                        else:
                            print(f'[优化架构] GFPGAN 子进程 ready 后意外退出（exitcode={exitcode}），'
                                  '回退 PyTorch 路径')
                        break   # ready 保持 False → 后面设 gfpgan_subprocess = None
                    ready = True
                    break
                # 每 300s 打印一次进度（不影响每 5s 轮询）
                now = time.time()
                if now - _last_report >= _report_every:
                    elapsed = now - (deadline - max_elapsed)
                    print(f'[优化架构] 等待中... {elapsed:.0f}s（Inference 进程初始化中）', flush=True)
                    _last_report = now
            if ready:
                print('[优化架构] GFPGAN 子进程已就绪，启动流水线')
                # FIX-ASYNC-GFPGAN (优化5B): 创建异步调度器
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
        
        # 记录线程句柄以便 close 时等待
        self._read_thread = read_thread
        self._detect_thread = detect_thread
        self._sr_thread = sr_thread
        self._gfpgan_thread = gfpgan_thread
        
        # 主线程处理写入
        self._write_frames(writer, pbar, total_frames)

        # ── FIX-JOIN-HANG: _write_frames 退出后立即终止所有流水线线程 ─────
        # 根因：_write_frames 通过"强制退出"条件退出时，哨兵可能尚未传播到
        # 所有线程。线程仍在 while self.running 循环中，join() 无限等待。
        # 修复：设置 running=False → 发送哨兵解除队列阻塞 → 带超时 join。
        print("[Pipeline] _write_frames 已退出，通知所有流水线线程终止...", flush=True)
        self.running = False

        # 向每个队列注入 None 哨兵，解除线程在 queue.get() 上的阻塞
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
            else:
                print(f"[Pipeline] {name} 线程已退出", flush=True)
        # ── FIX-JOIN-HANG 结束 ───────────────────────────────────────────
    
    def _read_frames(self, reader):
        """读取视频帧到队列
        
        FIX-ADAPTIVE-BATCH: 使用 _adaptive_read_batch_size 替代固定 optimal_batch_size。
        当人脸密度低时读取更大批次（提升 SR GPU 利用率），
        当人脸密度高时读取更小批次（为 GFPGAN 腾出显存）。

        FIX-READ-WATCHDOG: 修复要点：
          1) 只 catch Exception，不再吞 BaseException；
          2) FRAME_TIMEOUT 分支增加 reader 探活 + 连续超时看门狗
             （~60s 无帧则强制终止）+ 周期性心跳；
          3) finally 里多次重试送哨兵，保证下游不干等；
          4) ★ 不再在 finally 中设置 self.running = False ★
             —— 让 sentinel 正常沿流水线传播，避免下游在队列里的帧被丢弃。
        """
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
                    print(f"[Reader] ❌ reader.get_frame() 抛异常 "
                          f"@frame={frames_read}: {type(e).__name__}: {e}",
                          flush=True)
                    traceback.print_exc()
                    break

                # FIX-READ-WATCHDOG: 超时哨兵 — 队列暂时为空
                if img is FFmpegReader.FRAME_TIMEOUT:
                    consecutive_timeouts += 1

                    # 探活：如果 reader 线程已经死了却没送 EOF → 主动兜底
                    if hasattr(reader, 'is_reader_alive') and not reader.is_reader_alive():
                        if hasattr(reader, 'is_eof_sent') and reader.is_eof_sent():
                            # 预计下一次 get 就能拿到 None，继续循环即可
                            continue
                        if not _reader_dead_reported:
                            print(f"[Reader] ⚠️ reader 线程已死亡但未送 EOF "
                                  f"@frame={frames_read}，强制收尾",
                                  flush=True)
                            _reader_dead_reported = True
                        break

                    # 超时看门狗
                    if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                        print(f"[Reader] ❌ 连续 {consecutive_timeouts} 次 FRAME_TIMEOUT "
                              f"(~{consecutive_timeouts*2}s) 无帧 "
                              f"@frame={frames_read}，判定 reader 死锁，主动终止",
                              flush=True)
                        # 再次打印上游状态
                        if hasattr(reader, 'is_reader_alive'):
                            print(f"[Reader]   reader 线程存活: "
                                  f"{reader.is_reader_alive()}", flush=True)
                        break

                    # 定期轻量心跳，帮助定位
                    if consecutive_timeouts % 10 == 0:
                        print(f"[Reader] FRAME_TIMEOUT 已累计 {consecutive_timeouts} 次 "
                              f"(~{consecutive_timeouts*2}s) @frame={frames_read}，"
                              f"等待上游...", flush=True)
                    continue

                # 收到真正的数据，重置超时计数
                consecutive_timeouts = 0

                # EOF
                if img is None:
                    print(f"[Reader] EOF reached at frame {frames_read}", flush=True)
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
                            print(f"[Reader] 警告: 最后一批 {len(batch_frames)} 帧未能入队",
                                  flush=True)
                    break

                frames_read += 1
                batch_frames.append(img)

                # FIX-ADAPTIVE-BATCH: 使用自适应批处理大小
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
            print(f"[Reader] ❌ _read_frames 异常 @frame={frames_read}: "
                  f"{type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        finally:
            print(f"[Reader] 线程退出，frames_read={frames_read}, "
                  f"running={self.running}", flush=True)
            # FIX-READ-WATCHDOG: 关键：发送终止哨兵，避免下游干等
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
                print(f"[Reader] ❌ 致命: 终止哨兵未能送入 frame_queue，"
                      f"下游可能需要靠超时退出", flush=True)
            else:
                print(f"[Reader] ✓ 终止哨兵已送达 frame_queue", flush=True)

            # ★★★ FIX-READ-WATCHDOG: 关键修复：不再设置 self.running = False ★★★
            # 下游线程必须继续运行以消化 F/D/S/G 队列中已积压的帧，
            # 然后依靠哨兵自然退出。self.running=False 只应由 close() 触发。
    
    def _detect_faces(self):
        """人脸检测处理
        
        FIX-DET-THRESHOLD: 传递置信度阈值到 _detect_faces_batch
        """
        _sentinel_sent = False  # FIX-DUP-SENTINEL
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
                    
                    # 批量人脸检测
                    if self.detect_helper:
                        # FIX-DET-THRESHOLD: 传递置信度阈值参数
                        future = self.detect_executor.submit(
                            _detect_faces_batch, batch_frames, self.detect_helper,
                            self.face_det_threshold
                        )
                        # FIX-DET-THRESHOLD: 接收 4-tuple（含过滤计数）
                        face_data, _fw, _nf, _filtered = future.result()
                        self._face_frames_total += _fw
                        self._face_count_total  += _nf
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
        """SR推理处理
        
        FIX-GPU-PREFETCH: SR 推理完成后，尝试预取下一批帧的 H2D 传输。
        利用 SR→GFPGAN 交接期间的 GPU 内存总线空闲时间，将下一批帧数据
        从 CPU pinned memory 异步传输到 GPU。下一轮 SR 可跳过 H2D 阶段。
        当人脸密度低时（GFPGAN 快速完成），这个重叠尤其有价值：
        GPU 计算单元和内存总线同时被利用，减少整体空闲时间。
        """
        _first_batch_done = False
        _sentinel_sent = False  # FIX-DUP-SENTINEL
        # FIX-GPU-PREFETCH: 预取状态
        _prefetched_item = None      # 预取的 detect_queue item: (batch_frames, face_data, is_end)
        _prefetched_tensor = None    # 预取的 GPU tensor（已完成 H2D + permute + normalize）
        # FIX-GPU-PREFETCH: 专用 pinned buffer pool（与主 H2D 路径的 pool 独立，避免 buffer 覆盖）
        _prefetch_pool = PinnedBufferPool()
        _use_half = self.upsampler.half  # 缓存，避免线程中重复访问

        try:
            while self.running:
                try:
                    # FIX-GPU-PREFETCH: 优先使用上一轮预取的数据
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

                    # 获取GPU内存块
                    memory_block = None
                    while self.running and memory_block is None:
                        memory_block = self.memory_pool.acquire()
                        if memory_block is None:
                            time.sleep(0.005)
                    if not self.running:
                        break
                    
                    t0 = time.perf_counter()

                    def _sr_with_oom_fallback(frames, prefetched_batch_t=None):
                        """对当前 batch 做 OOM 级联降级推理，返回 sr_results 列表。
                        
                        FIX-GPU-PREFETCH: 接受 prefetched_batch_t 参数。
                        仅在全批次一次处理（无需子批拆分）且形状匹配时使用预取 tensor。
                        OOM 导致子批拆分时自动放弃预取（形状不匹配）。
                        """
                        retry_bs = min(self.optimal_batch_size, len(frames))
                        # FIX-GPU-PREFETCH: 检查预取 tensor 是否可用于当前批次
                        _can_use_prefetch = (prefetched_batch_t is not None and
                                            retry_bs >= len(frames) and
                                            prefetched_batch_t.shape[0] == len(frames))
                        _had_real_oom = False          # FIX-FALSE-OOM: 区分真实 OOM 与尾批截断
                        while True:
                            try:
                                all_sr = []
                                i = 0
                                while i < len(frames):
                                    sub = frames[i:i + retry_bs]
                                    # FIX-GPU-PREFETCH: 第一个子批次且可用预取时传入
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
                                # 成功：持久化降级后的 batch_size
                                # 只有真正经历过 OOM 重试才降级
                                if _had_real_oom and retry_bs < self.optimal_batch_size:
                                    print(f'[SR-OOM] batch_size 降级至 {retry_bs}，持久生效', flush=True)
                                    self.optimal_batch_size = retry_bs
                                return all_sr
                            except RuntimeError as _oom_e:
                                _es = str(_oom_e).lower()
                                if 'out of memory' not in _es:
                                    raise

                                _had_real_oom = True   # ← 标记真实 OOM

                                # OOM → 释放预取 tensor（回收显存）
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

                        # POST-SR-VALIDATE: 第一个 SR 批次完成 → 真实显存压力下验证 GFPGAN TRT
                        if not _first_batch_done and self.gfpgan_subprocess is not None:
                            _first_batch_done = True
                            print('[优化架构] 第一个 SR 批次完成，触发 GFPGAN TRT post-SR 验证...', flush=True)
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                            # FIX-ASYNC-GFPGAN: 通过 dispatcher 路由 validate
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

                        # ── FIX-GPU-PREFETCH: SR 完成后预取下一批 H2D ──────────────
                        # 时机：SR 推理刚完成，结果尚未传给 GFPGAN，GPU 计算单元短暂空闲。
                        # 在此窗口启动下一批帧的 CPU→GPU 异步传输（transfer_stream 上），
                        # 与主线程的 sr_queue.put / GFPGAN 处理并行执行。
                        # 下一轮 _process_sr 迭代可直接使用预取的 GPU tensor，跳过 H2D。
                        if (not is_end and _prefetched_item is None and
                                self.detect_queue.qsize() > 0):
                            try:
                                _peek_item = self.detect_queue.get_nowait()
                                if _peek_item is not None:
                                    _pk_frames, _pk_face_data, _pk_is_end = _peek_item
                                    if _pk_frames is not None:
                                        # 确保 transfer_stream 上一轮操作完成后再复用
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
                                        # 哨兵项，放回队列
                                        self.detect_queue.put(_peek_item)
                                else:
                                    # None 哨兵，放回队列
                                    self.detect_queue.put(_peek_item)
                            except queue.Empty:
                                pass
                        # ── FIX-GPU-PREFETCH 结束 ──────────────────────────────────

                        while self.running:
                            try:
                                self.sr_queue.put((batch_frames, face_data, memory_block, sr_results, is_end), timeout=1.0)
                                break
                            except queue.Full:
                                continue

                    except Exception as e:
                        print(f"SR推理错误（不可恢复）: {e}", flush=True)
                        import traceback; traceback.print_exc()
                        try:
                            self.memory_pool.release(memory_block['index'])
                        except Exception: pass
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"SR处理错误: {e}")
        finally:
            # ── FIX-PREFETCH-LOSS: 处理预取但未消费的帧，避免帧丢失 ────────
            # 场景：预取成功（_prefetched_item 非 None），但下一轮循环未执行
            # （异常/self.running=False/break），帧已从 detect_queue 取出却未处理。
            if _prefetched_item is not None:
                _pk_frames, _pk_face_data, _pk_is_end = _prefetched_item
                if _pk_frames is not None:
                    _pk_count = len(_pk_frames)
                    print(f'[SR] 检测到预取残留帧: {_pk_count} 帧，尝试补处理...',
                          flush=True)
                    try:
                        # 尝试正常 SR 推理（不使用预取 tensor，它可能已失效）
                        _pk_sr, _, _ = _sr_infer_batch(
                            self.upsampler, _pk_frames, self.args.outscale,
                            getattr(self.args, 'netscale', 4),
                            self.transfer_stream, self.sr_stream,
                            self.trt_accel, self.cuda_graph_accel,
                            prefetched_batch_t=None,  # 显式不用预取 tensor
                        )
                        self.sr_queue.put(
                            (_pk_frames, _pk_face_data, None, _pk_sr, _pk_is_end),
                            timeout=5.0)
                        print(f'[SR] 预取残留帧 SR 推理成功: {_pk_count} 帧已送入 sr_queue',
                              flush=True)
                    except Exception as _pf_e:
                        # SR 推理失败（GPU 状态异常 / OOM），回退到 CPU resize
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

                # 释放预取 GPU tensor 显存
                _prefetched_item = None
                if _prefetched_tensor is not None:
                    del _prefetched_tensor
                    _prefetched_tensor = None
            # ── FIX-PREFETCH-LOSS 结束 ────────────────────────────────────
            if not _sentinel_sent:
                try:
                    self.sr_queue.put((None, None, None, None, True), timeout=3.0)
                except Exception:
                    pass
    
    def _process_gfpgan(self):
        """GFPGAN处理 - 优化2(提前释放) + 优化5B(异步派发)

        FIX-EARLY-RELEASE (优化2): SR 结果已是 CPU numpy，不再需要 GPU memory_block。
        在取出 sr_queue 项后立即释放，解除 memory_pool 对 SR 的反压。

        FIX-ASYNC-GFPGAN (优化5B): 使用 AsyncGFPGANDispatcher 异步提交人脸 crops，
        不等待 GFPGAN 结果即处理下一个 SR 输出。GFPGAN 结果到达后在主循环中收集。
        实现 SR 和 GFPGAN 在时间上的重叠执行。

        FIX-ORDER-PRESERVE: 所有批次（有脸异步/有脸同步/无脸直通）统一通过
        _pending_tasks 有序队列输出，杜绝无人脸批次"插队"导致帧序颠倒。
        task_id=None 的"直通"项表示结果已就绪，阶段A 中可立即输出，
        但必须等前面的任务先完成才能出队，从而严格保序。

        FIX-SLOT-TRACKING (Bug 3): 显式跟踪共享内存双缓冲槽位占用状态，
        替代 SharedMemoryDoubleBuffer.next_slot() 的盲目轮转。
        slot 仅从 {0,1} - _slots_in_use 中分配，仅在对应任务从 _pending_tasks
        出队且结果已取回后才释放。task_queue.put() 失败时立即回收，不泄漏。

        FIX-ADAPTIVE-POLL (Bug 4): sr_queue.get() 的超时时间根据 _pending_tasks
        中是否存在未完成的异步任务动态调整。有异步等待时缩短至 50ms 以便快速
        回到 Phase A 检查结果；无异步任务时保持 500ms 避免空转。
        """
        _sentinel_sent = False

        # FIX-INFLIGHT-LOSS: 跟踪当前从 sr_queue 取出但尚未完成处理的项
        # 异常发生在 sr_queue.get() 和 gfpgan_queue.put() 之间时，
        # 没有此跟踪帧会永久丢失。finally 中据此做降级转发。
        _current_sr_item = None  # 类型: item 元组 或 None

        # FIX-ASYNC-GFPGAN + FIX-ORDER-PRESERVE: 维护所有批次的有序输出队列
        # 每个元素: (task_id, face_data, sr_results, slot_or_none, is_end)
        #
        # task_id != None → "异步批次"：需等待 AsyncGFPGANDispatcher 返回结果后组装。
        # task_id == None → "直通批次"：结果已就绪（sr_results 字段存放 final_frames），
        #                    阶段A 中可立即输出，但必须等前面的任务先完成才能出队。
        _pending_tasks: List[Tuple] = []
        _MAX_IN_FLIGHT = 2  # 与 task_queue maxsize 对齐

        # ── FIX-SLOT-POOL (Bug 3 修复): Queue 池化槽位管理 ────────────
        # 早期捕获 _shm 引用：即使后续 self.gfpgan_subprocess 被置 None
        # （如 post-SR validate 失败），局部引用仍指向同一 SharedMemoryDoubleBuffer。
        # _release_slot / _pop_and_output_head 通过闭包访问 _shm。
        _shm: Optional[SharedMemoryDoubleBuffer] = (
            getattr(self.gfpgan_subprocess, 'shm_buf', None)
            if self.gfpgan_subprocess is not None else None)

        def _release_slot(slot):
            """归还共享内存槽位到池中。

            FIX-SLOT-POOL: 委托 _shm.release_slot()，由 Queue.put() 实现。
            幂等：slot=None 或 _shm=None 时无操作。
            """
            if slot is not None and _shm is not None:
                _shm.release_slot(slot)

        # FIX-SLOT-POOL: 公共弹出+输出辅助函数
        # Phase B / slot-ensure 循环 / _drain_all_pending 共用，
        # 确保每个出队点都在 finally 中调用 _release_slot，消除遗漏风险。
        def _pop_and_output_head():
            """弹出 _pending_tasks[0]，阻塞等待结果(如需)，发送到 gfpgan_queue。
            异常时降级发送 sr_results。无论成功/失败，finally 中自动 _release_slot。"""
            if not _pending_tasks:
                return
            _h_tid, _h_fd, _h_sr, _h_slot, _h_is_end = _pending_tasks.pop(0)
            try:
                try:
                    if _h_tid is None:
                        # 直通批次，_h_sr 已是组装好的 final_frames
                        _h_final = _h_sr
                    elif self._async_dispatcher is not None:
                        _h_restored = self._async_dispatcher.wait_result(
                            _h_tid, timeout=120.0, slot=_h_slot)
                        _h_final = self._assemble_result(
                            _h_fd, _h_restored, _h_sr)
                    else:
                        # 无 dispatcher，降级输出 sr_results
                        _h_final = _h_sr
                except Exception as _he:
                    print(f'[GFPGAN] _pop_and_output_head 等待结果失败: {_he}',
                          flush=True)
                    _h_final = _h_sr  # 降级：发送未增强的 SR 帧
                # 发送到 gfpgan_queue
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
                    # self.running 已 False，最后尝试一次
                    try:
                        self.gfpgan_queue.put(
                            (_h_final, None, _h_is_end), timeout=5.0)
                    except Exception:
                        pass
            finally:
                # FIX-SLOT-POOL: 无论成功/失败/异常，归还 slot 到池中
                _release_slot(_h_slot)

        # FIX-ORDER-PRESERVE + FIX-SLOT-TRACKING: 排空 _pending_tasks 的局部辅助函数
        # 用于发送结束哨兵前，确保所有已入队批次按序输出。
        # 改用 _pop_and_output_head 统一处理，确保每项释放槽位。
        def _drain_all_pending():
            """排空所有 _pending_tasks，按序输出。用于发送哨兵前。"""
            while _pending_tasks:
                _pop_and_output_head()

        try:
            while self.running:
                # ── 阶段A: 按序输出已就绪的 pending 任务 ──────────────
                while _pending_tasks:
                    _oldest = _pending_tasks[0]
                    _tid, _fd, _sr, _slot, _is_end = _oldest

                    # FIX-ORDER-PRESERVE: 直通批次（无人脸/同步完成），立即可输出
                    if _tid is None:
                        _pending_tasks.pop(0)
                        _release_slot(_slot)  # FIX-SLOT-TRACKING: 防御性释放（直通 slot 应为 None）
                        # _sr 中已是组装好的 final_frames
                        while self.running:
                            try:
                                self.gfpgan_queue.put(
                                    (_sr, None, _is_end), timeout=1.0)
                                break
                            except queue.Full:
                                continue
                        continue  # 继续检查队列中下一个任务

                    if self._async_dispatcher is None:
                        # 同步路径结果已在 _oldest 中，直接处理
                        break

                    # 检查最早提交的任务是否完成（非阻塞 peek）
                    with self._async_dispatcher._lock:
                        if _tid not in self._async_dispatcher._results:
                            break  # 还没完成，不处理

                    # 已完成：取出结果，贴回人脸，发送到 writer
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
                        _release_slot(_slot)  # 无论成功失败都归还 slot

                    while self.running:
                        try:
                            self.gfpgan_queue.put(
                                (final_frames, None, _is_end), timeout=1.0)
                            break
                        except queue.Full:
                            continue

                # ── 阶段B: 从 sr_queue 取新的 SR 结果 ────────────────
                # 如果 in-flight 已满，先等待一个完成
                if len(_pending_tasks) >= _MAX_IN_FLIGHT:
                    # FIX-SLOT-TRACKING: 使用 _pop_and_output_head 统一处理，
                    # 替代原有内联代码，确保槽位释放不遗漏。
                    _pop_and_output_head()

                # ── FIX-ADAPTIVE-POLL (Bug 4): 动态超时 ──────────────
                # 有未完成异步任务时 50ms 快速轮询以便及时回到阶段A 检查结果，
                # 否则 500ms 节能等待。避免异步结果就绪后白等 ≤500ms。
                _has_async_pending = any(
                    t[0] is not None for t in _pending_tasks)
                _sr_timeout = 0.05 if _has_async_pending else 0.5
                # ── FIX-ADAPTIVE-POLL 结束 ────────────────────────────

                try:
                    item = self.sr_queue.get(timeout=_sr_timeout)
                except queue.Empty:
                    continue

                # FIX-INFLIGHT-LOSS: 记录当前取出的项，确保异常时可在 finally 中恢复
                _current_sr_item = item

                if item is None:
                    # FIX-ORDER-PRESERVE: 排空所有 pending 任务后再发送哨兵，
                    # 否则 writer 收到哨兵后停止接收，pending 帧丢失。
                    _drain_all_pending()

                    self.gfpgan_queue.put(None)
                    _sentinel_sent = True
                    _current_sr_item = None  # FIX-INFLIGHT-LOSS: 哨兵已转发
                    break

                batch_frames, face_data, memory_block, sr_results, is_end = item

                # ── 优化2: 立即释放 memory_pool 槽位 ─────────────────
                if memory_block is not None:
                    try:
                        self.memory_pool.release(memory_block['index'])
                    except Exception:
                        pass

                if batch_frames is None:
                    # FIX-ORDER-PRESERVE: 排空所有 pending 任务后再发送哨兵
                    _drain_all_pending()

                    self.gfpgan_queue.put((None, None, True))
                    _sentinel_sent = True
                    _current_sr_item = None  # FIX-INFLIGHT-LOSS: 哨兵已转发
                    break

                has_valid_faces = (face_data is not None and
                                   len(face_data) > 0 and
                                   any(fd.get('crops') for fd in face_data if fd))

                # FIX: 检查 GFPGAN 是否可用（子进程或主进程）
                _gfpgan_sub_alive = (self.gfpgan_subprocess is not None
                                     and self.gfpgan_subprocess.process.is_alive())
                _gfpgan_main_ok = (self.face_enhancer is not None
                                   and getattr(self.face_enhancer, 'gfpgan', None) is not None)

                _n_faces = (sum(len(fd.get('crops', [])) for fd in face_data)
                            if face_data else 0)

                if has_valid_faces and (_gfpgan_sub_alive or _gfpgan_main_ok):
                    # 收集所有 crops
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
                        # ── 优化5B: 异步提交 ────────────────────
                        if self._async_dispatcher is not None:
                            # ── FIX-SLOT-POOL: 异步提交（Queue 池化重构）──
                            _slot = None
                            _submitted = False
                            task_id = self._next_task_id()      # 单调递增，永不重复

                            if (_shm is not None
                                    and _n_faces <= SharedMemoryDoubleBuffer.MAX_FACES):

                                # FIX-SLOT-POOL: 非阻塞尝试获取空闲 slot
                                _slot = _shm.try_acquire_slot()
                                if _slot is None:
                                    # 池已耗尽：逐个排出 _pending_tasks 最旧任务以释放 slot。
                                    # _pop_and_output_head 的 finally 中调用 _release_slot，
                                    # 把 slot 归还到 _slot_pool，下次 try_acquire 即可成功。
                                    while _pending_tasks and _slot is None:
                                        _pop_and_output_head()
                                        _slot = _shm.try_acquire_slot()
                                if _slot is None:
                                    # 所有 pending 已排空仍无 slot（不应发生），
                                    # 阻塞等待兜底（超时后抛 TimeoutError，外层捕获降级）
                                    try:
                                        _slot = _shm.acquire_slot(timeout=30.0)
                                    except TimeoutError as _te:
                                        print(f'[GFPGAN] slot 获取超时: {_te}，'
                                              f'回退 pickle 路径', flush=True)
                                        _slot = None

                                if _slot is not None:
                                    # FIX-SLOT-POOL: write_input + put 整体 try/except，
                                    # 任一失败都归还 slot 到池中，防止泄漏。
                                    try:
                                        _shm.write_input(_slot, all_crops)
                                        self.gfpgan_subprocess.task_queue.put(
                                            (task_id, _n_faces, _slot),
                                            timeout=10.0)
                                        _submitted = True
                                    except Exception:
                                        # 提交失败，立即归还 slot 到池
                                        _release_slot(_slot)
                                        _slot = None
                            # ── FIX-SLOT-POOL: shm 路径结束 ──────────────

                            # _submitted 仍为 False: shm 不可用 / 人脸超限 /
                            # slot 获取失败 / write_input 异常 / put 超时
                            # → 回退到非共享内存 pickle 路径
                            if not _submitted:
                                _slot = None  # 确保 pending 中记录为无槽位
                                try:
                                    self.gfpgan_subprocess.task_queue.put(
                                        (task_id, all_crops), timeout=10.0)
                                except Exception as _submit_e:
                                    # 非 shm 提交也失败 → 降级为直通批次（跳过人脸增强）
                                    print(f'[GFPGAN] 异步提交完全失败: {_submit_e}，'
                                          f'降级直通', flush=True)
                                    _pending_tasks.append(
                                        (None, None, sr_results, None, is_end))
                                    _current_sr_item = None
                                    # FIX-EMA-COMMON: 即使降级直通，也更新 EMA
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

                            # FIX-INFLIGHT-LOSS: 已移交 _pending_tasks 管理，
                            # finally 中由 _pending_tasks 循环负责此批帧
                            _current_sr_item = None

                            # ── FIX-ADAPTIVE-BATCH: 更新人脸密度 EMA 并调整自适应批处理大小 ──
                            # 提交后、continue 前立即更新
                            # 每批 GFPGAN 处理完成后，根据实际人脸数更新密度估计。
                            # 低密度 → 增大读取批次（SR 处理更多帧/GPU调用，摊薄 kernel launch 开销）
                            # 高密度 → 减小读取批次（减少单次 SR 占用的显存，为 GFPGAN TRT 留出空间）
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
                                        # 人脸稀疏：GFPGAN 空闲多 → 增大 SR 批次利用空闲 GPU
                                        _new_bs = self._max_adaptive_batch
                                    elif self._face_density_ema > self._high_face_threshold:
                                        # 人脸密集：GFPGAN 是瓶颈 → 减小 SR 批次省显存
                                        _new_bs = self._min_adaptive_batch
                                    else:
                                        _new_bs = self._base_batch_size
                                    # 不超过 optimal_batch_size（可能已被 OOM 降级）
                                    _new_bs = min(_new_bs, max(self.optimal_batch_size, self._min_adaptive_batch))
                                    self._adaptive_read_batch_size = _new_bs
                            # ── FIX-ADAPTIVE-BATCH 结束 ──────────────────────────────────

                            # 不等结果，继续处理下一个 SR 输出！
                            continue
                        else:
                            # 无 dispatcher（降级同步路径）
                            all_restored = self.gfpgan_subprocess.infer(all_crops)
                            restored_by_frame = self._split_restored(all_restored, crops_per_frame, face_data)
                    elif _gfpgan_main_ok and all_crops:
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

                    # # 同步路径组装结果
                    # if all_restored is not None:
                    #     restored_by_frame = self._split_restored(all_restored, crops_per_frame, face_data)

                    final_frames = self._assemble_result(
                        face_data, restored_by_frame, sr_results)
                else:
                    if _n_faces > 0:
                        print(f'[GFPGAN] GFPGAN不可用，{_n_faces} 个人脸未处理')
                    final_frames = sr_results

                # FIX-EMA-COMMON: 将 EMA 更新移至所有同步路径的公共出口。
                # 原代码仅在 async dispatcher 分支内更新，导致主进程 PyTorch 路径、
                # 同步子进程路径、无人脸直通路径均不更新 EMA，始终为 0。
                # 此处位于 _pending_tasks.append 之前，所有非 async-continue 路径均经过。
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

                # FIX-ORDER-PRESERVE: 同步完成/无人脸批次统一进入 _pending_tasks 排队
                _pending_tasks.append((None, None, final_frames, None, is_end))
                _current_sr_item = None  # FIX-INFLIGHT-LOSS: 已移交 _pending_tasks 管理

        finally:
            # ── FIX-INFLIGHT-LOSS: 处理从 sr_queue 取出但未完成转发的项 ──────
            # 场景：sr_queue.get() 成功后，GFPGAN 推理异常 / self.running=False
            #       导致 gfpgan_queue.put() 从未执行，该批帧丢失。
            # 策略：降级转发 sr_results（跳过人脸增强），保帧优先于保质量。
            if _current_sr_item is not None:
                try:
                    _ci_batch, _, _ci_mem, _ci_sr, _ci_is_end = _current_sr_item
                    # 释放可能尚未释放的 memory_pool 槽位
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
                        # batch_frames 为 None 的哨兵项，put 未成功就异常了
                        print(f'[GFPGAN] finally: 发现未转发的哨兵项，补发',
                              flush=True)
                        self.gfpgan_queue.put((None, None, True), timeout=5.0)
                        _sentinel_sent = True
                except Exception as _ci_e:
                    print(f'[GFPGAN] finally: 处理残留 SR 项失败: {_ci_e}',
                          flush=True)
                _current_sr_item = None
            # ── FIX-INFLIGHT-LOSS 结束 ───────────────────────────────────

            # 清理所有 in-flight 任务（FIX-ORDER-PRESERVE: 包括直通批次）
            for _tid, _fd, _sr, _slot, _is_end in _pending_tasks:
                try:
                    if _tid is None:
                        # FIX-ORDER-PRESERVE: 直通批次，_sr 已是 final_frames
                        self.gfpgan_queue.put((_sr, None, _is_end), timeout=5.0)
                    elif self._async_dispatcher is not None:
                        all_restored = self._async_dispatcher.wait_result(
                            _tid, timeout=30.0, slot=_slot)
                        final = self._assemble_result(_fd, all_restored, _sr)
                        self.gfpgan_queue.put((final, None, _is_end), timeout=5.0)
                    else:
                        # 无 dispatcher，降级输出 sr_results
                        self.gfpgan_queue.put((_sr, None, _is_end), timeout=5.0)
                except Exception:
                    try:
                        # 降级：异步批次发 sr_results，直通批次 _sr 本身就是 final
                        self.gfpgan_queue.put((_sr, None, _is_end), timeout=5.0)
                    except Exception:
                        pass
                finally:
                    # FIX-SLOT-TRACKING: finally 中也释放每个槽位
                    _release_slot(_slot)

            _pending_tasks.clear()   # FIX-SLOT-TRACKING: 显式清空，防止悬挂引用

            if not _sentinel_sent:
                try:
                    self.gfpgan_queue.put((None, None, True), timeout=5.0)
                except Exception:
                    pass
    
    
    @staticmethod
    def _split_restored(all_restored, crops_per_frame, face_data):
        """将扁平化的 restored 列表按帧切分"""
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
        """将 GFPGAN 结果（restored_by_frame 或 flat list）与 SR 结果组装"""
        if restored_or_list is None or not face_data:
            return sr_results

        # 如果是 flat list（从异步 dispatcher 来），先切分
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
        """FIX-TASK-ID: 单调递增 task_id，替代 id() 避免内存地址复用导致结果错配"""
        with self._task_id_lock:
            self._task_id_counter += 1
            return self._task_id_counter

    def close(self):
        """清理资源，发送哨兵唤醒所有线程，强制关闭"""
        print("[Pipeline] 正在停止流水线...", flush=True)
        self.running = False

        # FIX-ASYNC-GFPGAN (优化5B): 先关闭 dispatcher
        if self._async_dispatcher is not None:
            self._async_dispatcher.close()
            self._async_dispatcher = None
        
        # 1. 向所有队列发送 None 哨兵，唤醒可能阻塞的线程
        for q_name, q in [('frame', self.frame_queue), ('detect', self.detect_queue),
                          ('sr', self.sr_queue), ('gfpgan', self.gfpgan_queue)]:
            try:
                q.put(None, timeout=1.0)
            except queue.Full:
                pass  # 队列满时忽略，继续尝试其他队列
            except Exception:
                pass
        print("[Pipeline] 已发送停止信号到所有队列", flush=True)
        
        # 2. 关闭子进程
        if self.gfpgan_subprocess:
            print("[Pipeline] 正在关闭GFPGAN子进程...", flush=True)
            self.gfpgan_subprocess.close()
            print("[Pipeline] GFPGAN子进程已关闭", flush=True)
        
        # 3. 关闭线程池，等待任务完成（超时后不再等待）
        self.detect_executor.shutdown(wait=False)
        self.paste_executor.shutdown(wait=False)
        
        # 4. 等待所有后台线程结束（带超时）
        thread_names = ['_read_thread', '_detect_thread', '_sr_thread', '_gfpgan_thread']
        for name in thread_names:
            thread = getattr(self, name, None)
            if thread and thread.is_alive():
                print(f"[Pipeline] 等待线程 {name} 结束...", flush=True)
                thread.join(timeout=5.0)
                # 若仍未结束，标记为 daemon 让 Python 退出时自动终止
                if thread.is_alive():
                    print(f"[Pipeline] 线程 {name} 未响应，已放弃等待", flush=True)
                    if not thread.is_alive():
                        thread.daemon = True
        print("[Pipeline] 所有流水线线程已关闭", flush=True)
    
    def _write_frames(self, writer, pbar, total_frames):
        """写入帧处理

        FIX-WRITE-WATCHDOG: 增加全流水线空转死锁看门狗：
          - 所有队列（含 reader prefetch）同时为空且未收到哨兵持续 120s
            → dump 所有线程栈 + 强制退出
          - 彻底消灭静默死锁，无论 ffmpeg 还是任一线程僵死，都能在 120s 内被识别
        """
        written_count = 0
        end_sentinel_count = 0
        received_end_sentinel = False  # 初始化标志变量

        # FIX-WRITE-WATCHDOG: 全流水线空转看门狗
        _idle_since = None
        IDLE_DEADLOCK_TIMEOUT = 120.0  # 120s 全空且无哨兵 → 强制退出并 dump 栈

        try:
            while self.running:
                try:
                    item = self.gfpgan_queue.get(timeout=10.0)   # 增加超时，避免死等
                    
                    if item is None:
                        end_sentinel_count += 1
                        received_end_sentinel = True
                        print(f"[Pipeline] 写入线程收到第{end_sentinel_count}个结束哨兵，"
                              f"队列积压: {self._queue_status_str()}", flush=True)
                        continue
                    
                    final_frames, memory_block, is_end = item
                    
                    if final_frames is None:
                        if is_end:
                            end_sentinel_count += 1
                            received_end_sentinel = True
                            print(f"[Pipeline] 写入线程收到结束信号，"
                                  f"队列积压: {self._queue_status_str()}", flush=True)
                            continue
                        continue

                    # FIX-WRITE-WATCHDOG: 收到有效数据，重置空转计时
                    _idle_since = None

                    # 写入帧
                    for frame in final_frames:
                        # FIX-DEADLOCK-4: 检测 FFmpeg 后台写入器是否已经崩溃
                        if getattr(writer, '_broken', False):
                            print("\n[致命错误] FFmpeg 后台写入进程已崩溃!", flush=True)
                            self.running = False  # 立即阻断整条流水线
                            break
                            
                        writer.write_frame(frame)
                        written_count += 1
                    
                    # 如果写入中途崩溃，立刻跳出大循环
                    if getattr(writer, '_broken', False):
                        break
                    
                    # 更新进度
                    pbar.update(len(final_frames))
                    self.meter.update(len(final_frames))
                    
                    # 每批次都更新 postfix（不受 batch_size 整除影响）
                    current_fps = self.meter.fps()
                    eta = self.meter.eta(total_frames)
                    avg_ms = np.mean(self.timing[-10:]) * 1000 if self.timing else 0

                    # FIX-WRITE-WATCHDOG: 读取 reader / prefetch 水位
                    p_size, p_cap, reader_alive, reader_eof, _produced = (
                        self._get_reader_state())
                    reader_state = ('alive' if reader_alive
                                    else ('eof' if reader_eof else 'DEAD'))

                    pbar.set_postfix(
                        fps=f'{current_fps:.1f}',
                        eta=f'{eta:.0f}s',
                        bs=self.optimal_batch_size,      # FIX-BS-DISPLAY: 显示 SR 实际 batch_size，OOM 降级后立即体现
                        ms=f'{avg_ms:.0f}',
                        queue_sizes=(f"P:{p_size}/{p_cap}[{reader_state}]/"
                                     f"F:{self.frame_queue.qsize()}/"
                                     f"D:{self.detect_queue.qsize()}/"
                                     f"S:{self.sr_queue.qsize()}/"
                                     f"G:{self.gfpgan_queue.qsize()}")
                    )

                    # 动态检测GPU内存压力
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        if allocated > 0.9 * reserved:
                            print(f'\n[资源警告] GPU内存压力过高: {allocated:.2f}GB / {reserved:.2f}GB')
                    
                    # 每跨越 20 帧输出一次详细日志（跨越检测，不受 batch_size 整除影响）
                    if written_count // 20 > (written_count - len(final_frames)) // 20:
                        # FIX-DET-THRESHOLD + FIX-ADAPTIVE-BATCH: 增加过滤计数和密度信息
                        _density_str = f' | 密度EMA={self._face_density_ema:.1f}' if self._enable_adaptive_batch else ''
                        _filtered_str = f' | 过滤{self._face_filtered_total}' if self._face_filtered_total > 0 else ''
                        _adaptive_str = f' | 自适应arbs={self._adaptive_read_batch_size}' if self._enable_adaptive_batch else ''
                        print(f"[性能监控] 帧{written_count}/{total_frames} | fps={current_fps:.1f} | eta={eta:.0f}s | bs={self.optimal_batch_size} | ms={avg_ms:.0f} | "
                              f"队列 P:{p_size}/{p_cap}[{reader_state}]/"
                              f"F:{self.frame_queue.qsize()}/D:{self.detect_queue.qsize()}/"
                              f"S:{self.sr_queue.qsize()}/G:{self.gfpgan_queue.qsize()} | "
                              f"人脸 {self._face_count_total}张/{self._face_frames_total}帧"
                              f"{_filtered_str}{_density_str}{_adaptive_str}", flush=True)
                    
                except queue.Empty:
                    # FIX-EXIT-HANG: 退出判断只检查 gfpgan_queue（_write_frames 直接消费的队列）。
                    # sr_queue 由 _process_gfpgan 消费，可能残留 finally 安全哨兵，
                    # _write_frames 永远不会去消费它，检查它会导致死循环。
                    if received_end_sentinel and self.gfpgan_queue.qsize() == 0:
                        print(f"[Pipeline] 收到哨兵且 gfpgan_queue 已清空，退出。"
                              f"已写入 {written_count}/{total_frames} 帧", flush=True)
                        break
                    # 所有帧已写入 + 收到结束信号 → 无需再等待上游队列
                    if written_count >= total_frames and received_end_sentinel:
                        print(f"[Pipeline] 所有帧已写入且收到结束信号，退出。"
                              f"已写入 {written_count}/{total_frames} 帧", flush=True)
                        break
                    # 新增：所有帧已写入 + 所有上游队列为空 → 强制退出
                    if (written_count >= total_frames
                            and self.sr_queue.qsize() == 0
                            and self.gfpgan_queue.qsize() == 0):
                        print(f"[Pipeline] 所有帧已写入且上游队列清空，强制退出", flush=True)
                        break

                    # ── FIX-WRITE-WATCHDOG: 全流水线空转死锁看门狗（含 prefetch）──
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
                            print(f"[Pipeline][看门狗] 检测到全流水线空转 "
                                  f"(P:{p_size}/{p_cap}[{reader_state}])，"
                                  f"开始计时（阈值 {IDLE_DEADLOCK_TIMEOUT:.0f}s）；"
                                  f"已写入 {written_count}/{total_frames}",
                                  flush=True)
                        elif time.time() - _idle_since > IDLE_DEADLOCK_TIMEOUT:
                            print(f"[Pipeline][看门狗] ⚠️ 全流水线空转超过 "
                                  f"{IDLE_DEADLOCK_TIMEOUT:.0f}s 无进展，"
                                  f"判定上游死锁。"
                                  f"P:{p_size}/{p_cap}[{reader_state}] "
                                  f"produced={_produced}；"
                                  f"已写入 {written_count}/{total_frames} 帧，"
                                  f"强制退出。", flush=True)
                            # dump 所有线程栈 + 队列
                            print(f"[Pipeline][看门狗] 打印所有队列水位：",
                                  flush=True)
                            self._dump_all_queues()
                            print(f"[Pipeline][看门狗] 打印所有线程调用栈：",
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
                    # ── FIX-WRITE-WATCHDOG 结束 ──────────────────────────────
                    continue
                except Exception as e:
                    print(f"写入帧错误: {e}")
                    # FIX: 增加 memory_block is not None 判断，防止处理哨兵时抛出异常导致泄漏
                    if 'memory_block' in locals() and memory_block is not None:  
                        try:
                            self.memory_pool.release(memory_block['index'])
                        except Exception:
                            pass
        finally:
            print(f"[Pipeline] 写入线程退出，已写入 {written_count}/{total_frames} 帧 "
                  f"(最终队列: {self._queue_status_str()})", flush=True)
            # 注意：不要在这里调用 self.close()！
            # close() 应该在 main_optimized 的 finally 中由主线程统一调用
            # 否则会导致递归关闭和流水线提前终止


def main_optimized(args):
    """优化版主函数"""
    
    print("[优化架构] 修复版: 改进 GFPGAN TRT 就绪判断")
    # FIX-DET-THRESHOLD: 打印检测阈值配置
    print(f"[优化架构] 人脸检测置信度阈值: {getattr(args, 'face_det_threshold', 0.5)}")
    # FIX-ADAPTIVE-BATCH: 打印自适应批处理配置
    if getattr(args, 'adaptive_batch', True):
        print(f"[优化架构] 自适应批处理: 已启用")
    print("[优化架构] 阶段 0: 准备环境（不初始化 CUDA）...")
    
    cuda_available = torch.backends.cuda.is_built() and torch.cuda.device_count() > 0
    if not cuda_available:
        print("[优化架构] CUDA 不可用，使用 CPU 模式")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"[优化架构] CUDA 编译支持: 是")
        print(f"[优化架构] 延迟 CUDA Runtime 初始化直到 GFPGAN 子进程就绪")

    _early_gfpgan_subprocess = None
    gfpgan_ready = False
    use_gfpgan_subprocess = False
    gfpgan_mode = "disabled"
    
    if args.face_enhance and getattr(args, 'gfpgan_trt', False) and GFPGANer is not None:
        if not cuda_available:
            print("[优化架构] 警告: CUDA 不可用，跳过 GFPGAN TRT")
        else:
            print("[优化架构] 阶段 1: 预启动 GFPGAN 子进程（GPU 干净状态）...")
            _model_paths_early = {
                '1.3': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                '1.4': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                'RestoreFormer': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth',
            }
            _model_url_early = _model_paths_early.get(args.gfpgan_model, _model_paths_early['1.4'])
            _model_dir_early = osp.join(models_RealESRGAN, 'GFPGAN')
            os.makedirs(_model_dir_early, exist_ok=True)
            _model_filename_early = osp.basename(_model_url_early)
            _model_path_early = osp.join(_model_dir_early, _model_filename_early)
            if not osp.exists(_model_path_early):
                _model_path_early = load_file_from_url(_model_url_early, _model_dir_early, True)

            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            
            print('[优化架构] 启动 GFPGAN 子进程...')
            _early_gfpgan_subprocess = GFPGANSubprocess(
                model_path=_model_path_early, device=device,
                gfpgan_weight=args.gfpgan_weight, gfpgan_batch_size=args.gfpgan_batch_size,
                use_fp16=not args.no_fp16, use_trt=True,
                trt_cache_dir=getattr(args, 'trt_cache_dir', None),
                gfpgan_model=args.gfpgan_model,
            )
            
            print('[优化架构] 等待 GFPGAN 子进程完成初始化...')
            max_wait = 5400
            deadline = time.time() + max_wait
            _poll_interval = 3
            _last_report = time.time()
            _report_every = 60
            
            while time.time() < deadline:
                if not _early_gfpgan_subprocess.process.is_alive():
                    exitcode = _early_gfpgan_subprocess.process.exitcode
                    _early_gfpgan_subprocess.close()
                    _early_gfpgan_subprocess = None
                    gfpgan_ready = False
                    gfpgan_mode = "failed_subprocess"
                    break
                if _early_gfpgan_subprocess.ready_event.wait(timeout=_poll_interval):
                    time.sleep(0.5)
                    if not _early_gfpgan_subprocess.process.is_alive():
                        _early_gfpgan_subprocess.close()
                        _early_gfpgan_subprocess = None
                        gfpgan_ready = False
                        gfpgan_mode = "failed_trt_warmup"
                        break
                    print('[优化架构] GFPGAN 子进程 signaled ready 且进程稳定')
                    gfpgan_ready = True
                    use_gfpgan_subprocess = True
                    gfpgan_mode = "subprocess_trt"
                    print(f'[优化架构] GFPGAN 子进程验证通过，模式: {gfpgan_mode}')
                    break
                now = time.time()
                if now - _last_report >= _report_every:
                    elapsed = now - (deadline - max_wait)
                    print(f'[优化架构] 等待 GFPGAN 初始化... {elapsed:.0f}s', flush=True)
                    _last_report = now
            
            if not gfpgan_ready:
                print('[优化架构] GFPGAN 子进程初始化失败，将使用主进程 PyTorch 路径')
                if _early_gfpgan_subprocess:
                    _early_gfpgan_subprocess.close()
                    _early_gfpgan_subprocess = None
                gfpgan_mode = "main_pytorch_fallback"
            else:
                print('[优化架构] GFPGAN 子进程准备就绪')
    
    print("[优化架构] 阶段 2: 初始化主进程 CUDA 并加载 RealESRGAN...")
    
    if cuda_available:
        try:
            torch.cuda.init()
            device_name = torch.cuda.get_device_name(0)
            device_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[优化架构] 主进程 CUDA 初始化完成: {device_name} ({device_mem:.1f}GB)")
            torch.cuda.empty_cache()
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[优化架构] 当前显存: 已分配 {mem_allocated:.2f}GB, 预留 {mem_reserved:.2f}GB")
        except Exception as e:
            print(f"[优化架构] CUDA 初始化失败: {e}")
            cuda_available = False
            device = torch.device('cpu')
    
    print(f"[优化架构] 加载模型: {args.model_name}")
    use_half = not args.no_fp16
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and hasattr(args, 'denoise_strength'):
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]
    
    try:
        upsampler = _build_upsampler(
            args.model_name, dni_weight, args.tile, args.tile_pad, 
            args.pre_pad, use_half, device
        )
        print("[优化架构] RealESRGAN模型加载成功")
        _, _netscale, _ = MODEL_CONFIG.get(args.model_name, (None, 4, None))
        args.netscale = _netscale
        if getattr(args, 'use_tensorrt', False) and not getattr(args, 'no_compile', False):
            args.no_compile = True
    except Exception as e:
        print(f"[优化架构] RealESRGAN模型加载失败: {e}")
        import traceback; traceback.print_exc()
        return
    
    face_enhancer = None
    if args.face_enhance and GFPGANer is not None:
        if use_gfpgan_subprocess and gfpgan_ready and _early_gfpgan_subprocess is not None:
            print("[优化架构] GFPGAN 由子进程处理，主进程创建 detect helper...")
            try:
                from facexlib.utils.face_restoration_helper import FaceRestoreHelper
                class DummyGFPGANer:
                    def __init__(self, device, upscale):
                        self.device = device
                        self.upscale = upscale
                        self.face_helper = FaceRestoreHelper(
                            upscale_factor=upscale, face_size=512, crop_ratio=(1, 1),
                            det_model='retinaface_resnet50', save_ext='png',
                            use_parse=True, device=device,
                        )
                        self.gfpgan = None
                        self.model_path = None
                face_enhancer = DummyGFPGANer(device, args.outscale)
                print("[优化架构] Detect helper 创建成功（GFPGAN 推理由子进程处理）")
            except Exception as e:
                print(f"[优化架构] Detect helper 创建失败: {e}")
                face_enhancer = None
                use_gfpgan_subprocess = False
                gfpgan_ready = False
        else:
            print("[优化架构] 加载GFPGAN主进程模型...")
            try:
                model_paths = {
                    '1.3': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                    '1.4': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                    'RestoreFormer': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
                }
                model_url = model_paths.get(args.gfpgan_model, model_paths['1.4'])
                model_dir = osp.join(models_RealESRGAN, 'GFPGAN')
                os.makedirs(model_dir, exist_ok=True)
                model_filename = osp.basename(model_url)
                model_path = osp.join(model_dir, model_filename)
                if not osp.exists(model_path):
                    model_path = load_file_from_url(model_url, model_dir, True)
                face_enhancer = GFPGANer(
                    model_path=model_path, upscale=args.outscale, arch='clean',
                    channel_multiplier=2, bg_upsampler=None, device=device
                )
                if face_enhancer.gfpgan is None:
                    raise RuntimeError("GFPGAN 模型加载失败: gfpgan 网络为 None")
                print("[优化架构] GFPGAN主进程模型加载成功")
                gfpgan_mode = "main_pytorch"
                if getattr(args, 'gfpgan_trt', False):
                    args.gfpgan_trt = False
            except Exception as e:
                print(f"[优化架构] GFPGAN模型加载失败: {e}")
                import traceback; traceback.print_exc()
                face_enhancer = None
                gfpgan_mode = "disabled"
    
    print("[优化架构] 阶段 3: 创建视频读写器...")
    reader = FFmpegReader(
        args.input, ffmpeg_bin=getattr(args, 'ffmpeg_bin', 'ffmpeg'),
        prefetch_factor=getattr(args, 'prefetch_factor', 48),
        use_hwaccel=getattr(args, 'use_hwaccel', True),
    )
    
    out_h = int(reader.height * args.outscale)
    out_w = int(reader.width * args.outscale)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    writer = FFmpegWriter(args, reader.audio, out_h, out_w, args.output, reader.fps)
    
    pbar = tqdm(total=reader.nb_frames, unit='frame', desc='[优化流水线]',
                dynamic_ncols=False, ncols=180,
                bar_format='{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]\n')
    
    trt_accel = None
    if getattr(args, 'use_tensorrt', False) and cuda_available:
        meta = get_video_meta_info(args.input)
        sh = (args.batch_size, 3, meta['height'], meta['width'])
        trt_dir = getattr(args, 'trt_cache_dir', None) or osp.join(base_dir, '.trt_cache')
        print(f'[优化架构] 初始化 SR TensorRT Engine (shape={sh})...')
        try:
            trt_accel = TensorRTAccelerator(
                upsampler.model, device, trt_dir, sh, use_fp16=not args.no_fp16)
            if trt_accel.available:
                print('[优化架构] SR TensorRT Engine 加载成功')
            else:
                trt_accel = None
        except Exception as _te:
            print(f'[优化架构] SR TensorRT 初始化异常: {_te}')
            trt_accel = None
    
    if _early_gfpgan_subprocess is not None and use_gfpgan_subprocess and gfpgan_ready:
        args._early_gfpgan_subprocess = _early_gfpgan_subprocess
        print(f'[优化架构] GFPGAN 子进程已绑定到流水线（模式: {gfpgan_mode}）')
    else:
        args._early_gfpgan_subprocess = None
        if args.face_enhance:
            print(f'[优化架构] GFPGAN 使用主进程模式: {gfpgan_mode}')
    
    print("[优化架构] 阶段 4: 启动优化流水线...")
    pipeline = DeepPipelineOptimizer(upsampler, face_enhancer, args, device, trt_accel=trt_accel,
                                     input_h=reader.height, input_w=reader.width)
    
    start_time = time.time()
    try:
        pipeline.optimize_pipeline(reader, writer, pbar, reader.nb_frames)
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"流水线错误: {e}")
        import traceback; traceback.print_exc()
    finally:
        print("\n[优化架构] ===========================================")
        print("[优化架构] 视频推理完成，正在清理资源...")
        print("[优化架构] ===========================================\n")
        
        print("[优化架构] 步骤1/4: 关闭流水线线程...")
        pipeline.close()
        print("[优化架构] 流水线线程已关闭")
        
        print("[优化架构] 步骤2/4: 关闭FFmpeg写入器...")
        writer.close()
        print("[优化架构] FFmpeg写入器已关闭")
        
        print("[优化架构] 步骤3/4: 关闭视频读取器...")
        reader.close()
        print("[优化架构] 视频读取器已关闭")
        
        print("[优化架构] 步骤4/4: 关闭进度条...")
        pbar.close()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # ── 生成性能报告（如果指定了 --report）─────────────────────
        if getattr(args, 'report', None) and pipeline.timing:
            import json
            report_path = args.report
            os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)
            elapsed = total_time
            report = {
                'input': args.input,
                'output': args.output,
                'model': args.model_name,
                'outscale': args.outscale,
                'batch_size': args.batch_size,
                'fp16': not args.no_fp16,
                'trt': trt_accel is not None and trt_accel.available,
                'nvdec': getattr(args, 'use_hwaccel', True),
                'nvenc': getattr(args, 'video_codec', 'libx264') in ('h264_nvenc', 'hevc_nvenc'),
                'face_enhance': args.face_enhance,
                'frame_count': reader.nb_frames,
                'elapsed_s': round(elapsed, 2),
                'avg_fps': round(reader.nb_frames / elapsed, 2) if elapsed > 0 else 0,
                'infer_latency_ms': {
                    'mean': round(float(np.mean(pipeline.timing)) * 1000, 2),
                    'p95': round(float(np.percentile(pipeline.timing, 95)) * 1000, 2),
                    'max': round(float(np.max(pipeline.timing)) * 1000, 2),
                },
                'gfpgan_mode': gfpgan_mode,
                'face_filtered': pipeline._face_filtered_total,
                'adaptive_batch': pipeline._enable_adaptive_batch,
            }
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f'[Info] 性能报告已保存: {report_path}')
        # ──────────────────────────────────────────────────────────

        if pipeline.timing:
            avg_time = np.mean(pipeline.timing) * 1000
            actual_fps = reader.nb_frames / total_time if total_time > 0 else 0
            print(f"\n[性能统计] 总时间: {total_time:.1f}秒 | 平均: {avg_time:.1f}ms | FPS: {actual_fps:.2f}")
            print(f"[性能统计] GFPGAN 模式: {gfpgan_mode}")
            # FIX-DET-THRESHOLD: 打印过滤统计
            if pipeline._face_filtered_total > 0:
                print(f"[性能统计] 人脸检测: 保留 {pipeline._face_count_total} 个, "
                      f"过滤 {pipeline._face_filtered_total} 个 "
                      f"(阈值={pipeline.face_det_threshold})")
            # FIX-ADAPTIVE-BATCH: 打印最终密度和自适应状态
            if pipeline._enable_adaptive_batch:
                print(f"[性能统计] 最终人脸密度EMA: {pipeline._face_density_ema:.2f} 人脸/帧, "
                      f"最终自适应arbs: {pipeline._adaptive_read_batch_size}")


def main():
    """主函数 - 参数解析"""
    parser = argparse.ArgumentParser(
        description='Real-ESRGAN 视频超分 —— 架构优化版 v6.4',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # 基础参数
    parser.add_argument('-i', '--input', type=str, required=True, help='输入视频路径')
    parser.add_argument('-o', '--output', type=str, required=True, help='输出视频路径')
    parser.add_argument('-n', '--model-name', type=str, default='realesr-animevideov3', help='模型名称')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='输出缩放比例')
    parser.add_argument('-dn', '--denoise-strength', type=float, default=0.5, help='去噪强度')
    parser.add_argument('--suffix', type=str, default='out', help='输出文件后缀')
    
    # 优化参数
    parser.add_argument('--batch-size', type=int, default=6, help='批处理大小（优化版本推荐6-8）')
    parser.add_argument('--prefetch-factor', type=int, default=48, help='读帧预取队列深度')
    
    # 人脸增强参数
    parser.add_argument('--face-enhance', action='store_true', help='启用人脸增强')
    parser.add_argument('--gfpgan-model', type=str, default='1.4', choices=['1.3', '1.4', 'RestoreFormer'],
                        help='GFPGAN 模型版本')
    parser.add_argument('--gfpgan-weight', type=float, default=0.5, help='GFPGAN 增强融合权重')
    parser.add_argument('--gfpgan-batch-size', type=int, default=8, help='GFPGAN 单次最多处理人脸数')
    # FIX-DET-THRESHOLD: 人脸检测置信度阈值参数
    # 检测模型（retinaface_resnet50）为每个检测结果输出置信度分数 [0,1]。
    # 提高此阈值可过滤模糊、远景、遮挡人脸及非人脸误检，减少 GFPGAN 调用次数。
    # 建议范围：
    #   0.5（默认）—— 保留大部分人脸，包括略模糊的
    #   0.7        —— 过滤模糊/远景人脸，适合人脸密集视频加速
    #   0.9        —— 仅保留高置信度清晰人脸，最大化加速效果
    parser.add_argument('--face-det-threshold', type=float, default=0.5,
                        help='人脸检测置信度阈值 [0.0-1.0]，越高过滤越多低质量检测。'
                             '0.5=保留多数人脸，0.7=过滤模糊远景，0.9=仅保留清晰人脸')
    # FIX-ADAPTIVE-BATCH: 自适应批处理开关
    # 根据视频内容中人脸密度的变化，动态调整每批读取的帧数。
    # 人脸稀疏段 → 增大批次（提升 SR GPU 利用率，减少 kernel launch 开销）
    # 人脸密集段 → 减小批次（为 GFPGAN TRT 预留更多显存）
    parser.add_argument('--adaptive-batch', action='store_true', default=True,
                        help='启用基于人脸密度的自适应批处理大小（默认开启）')
    parser.add_argument('--no-adaptive-batch', action='store_true', default=False,
                        help='禁用自适应批处理大小（使用固定 --batch-size）')
    
    # 加速参数
    parser.add_argument('--no-fp16', action='store_true', help='禁用 FP16')
    parser.add_argument('--no-compile', action='store_true', help='禁用 torch.compile')
    parser.add_argument('--use-tensorrt', action='store_true', help='启用 SR TensorRT 加速')
    parser.add_argument('--gfpgan-trt', action='store_true', help='GFPGAN TensorRT 加速')
    parser.add_argument('--no-cuda-graph', action='store_true', help='禁用 CUDA Graph')
    
    # 硬件加速参数
    parser.add_argument('--use-hwaccel', action='store_true', default=True, help='启用 NVDEC')
    parser.add_argument('--no-hwaccel', action='store_true', help='禁用 NVDEC')
    
    # 其他参数
    parser.add_argument('-t', '--tile', type=int, default=0, help='分块大小')
    parser.add_argument('--tile-pad', type=int, default=10, help='分块填充')
    parser.add_argument('--pre-pad', type=int, default=0, help='预填充')
    parser.add_argument('--fps', type=float, default=None, help='输出帧率')
    
    # 编码参数
    parser.add_argument('--video-codec', type=str, default='libx264', 
                        choices=['libx264', 'libx265', 'libvpx-vp9', 'libvpx-vp9', 'h264_nvenc'],
                        help='偏好编码器')
    parser.add_argument('--crf', type=int, default=23, help='编码质量')
    parser.add_argument('--x264-preset', type=str, default='medium',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast',
                                 'medium', 'slow', 'slower', 'veryslow'],
                        help='libx264/libx265 preset')
    parser.add_argument('--ffmpeg-bin', type=str, default='ffmpeg', help='FFmpeg 二进制路径')
    
    # 性能报告
    parser.add_argument('--report', type=str, default=None,
                        help='输出 JSON 性能报告路径（如 report.json）')

    # 高优先级覆盖参数
    parser.add_argument('--no-tensorrt', dest='no_tensorrt', action='store_true', default=False,
                        help='强制禁用 TensorRT（覆盖 --use-tensorrt 或外部 config）')
    parser.add_argument('--use-compile', dest='use_compile_force', action='store_true', default=False,
                        help='强制启用 torch.compile（覆盖 --no-compile 或 config）')
    parser.add_argument('--use-cuda-graph', dest='use_cuda_graph_force', action='store_true', default=False,
                        help='强制启用 CUDA Graph（覆盖 --no-cuda-graph 或 config）')
    
    args = parser.parse_args()

    # 处理硬件加速开关
    if args.no_hwaccel:
        args.use_hwaccel = False

    # 派生 use_compile / use_cuda_graph（用于后续仲裁）
    args.use_compile = not getattr(args, 'no_compile', False)
    args.use_cuda_graph = not getattr(args, 'no_cuda_graph', False)

    # 高优先级覆盖参数处理
    _override_msgs = []
    if args.no_tensorrt and args.use_tensorrt:
        args.use_tensorrt = False
        _override_msgs.append('--no-tensorrt    覆盖了  --use-tensorrt  → TensorRT 已禁用')
    elif args.no_tensorrt:
        args.use_tensorrt = False
    if args.use_compile_force:
        args.use_compile = True
        if args.no_compile:
            _override_msgs.append('--use-compile    覆盖了  --no-compile   → torch.compile 已启用')
    if args.use_cuda_graph_force:
        args.use_cuda_graph = True
        if args.no_cuda_graph:
            _override_msgs.append('--use-cuda-graph 覆盖了  --no-cuda-graph → CUDA Graph 已启用')
    if _override_msgs:
        print('[CLI覆盖] 以下设置已被高优先级参数覆盖：')
        for msg in _override_msgs:
            print(f'          · {msg}')
        print()

    # 冲突仲裁：TRT > compile > CUDA Graph
    if args.use_compile and args.use_tensorrt:
        print('[Warning] --use-tensorrt 与 torch.compile 互斥，TRT 优先，compile 已自动禁用。')
        args.use_compile = False
    if args.use_cuda_graph:
        if args.use_tensorrt:
            print('[Warning] CUDA Graph 与 TRT 互斥，TRT 优先，CUDA Graph 已自动禁用。')
            args.use_cuda_graph = False
        elif args.use_compile:
            print('[Info] torch.compile 已启用（内含 cudagraphs），CUDA Graph 已自动禁用。')
            args.use_cuda_graph = False

    # 更新对应的 --no-xxx 标志以保持一致性（可选）
    args.no_compile = not args.use_compile
    args.no_cuda_graph = not args.use_cuda_graph

    # FIX-ADAPTIVE-BATCH: 处理 --no-adaptive-batch 覆盖
    if args.no_adaptive_batch:
        args.adaptive_batch = False

    print("Real-ESRGAN Video Enhancement v6.4 - 架构优化版")
    print("主要优化特性:")
    print("1. 深度流水线架构（4级并行处理）")
    print("2. GPU内存池优化（避免频繁分配释放）")
    print("3. 异步计算模式（多CUDA流并行）")
    print("4. 多级缓冲队列（深度缓冲减少等待）")
    print("5. 优化线程池配置（提高并发效率）")
    # FIX-DET-THRESHOLD + FIX-ADAPTIVE-BATCH + FIX-GPU-PREFETCH: 新增优化特性
    print("6. 人脸检测置信度过滤（减少无效 GFPGAN 推理）")
    print("7. 自适应批处理（根据人脸密度动态调整）")
    print("8. SR H2D 预取重叠（利用空闲 GPU 内存总线）")
    # FIX-READER-DIAG + FIX-READ-WATCHDOG + FIX-WRITE-WATCHDOG: 死锁诊断增强
    print("9. FFmpegReader 四路径诊断 + EOF 保证送达 + 探活接口")
    print("10. _read_frames 超时看门狗（~60s 无帧 → 主动终止）")
    print("11. _write_frames 全流水线空转看门狗（120s → dump 栈 + 强制退出）")
    # ── FIX-T4-FP16: T4 GPU + GFPGAN TRT 自动禁用 FP16 防止溢出噪斑 ────
    if args.gfpgan_trt and torch.cuda.is_available():
        # try:
        #     props = torch.cuda.get_device_properties(0)
        #     # 识别 T4 级别 GPU（名称含 T4 或 计算能力 SM 7.5）
        #     is_t4_level = "t4" in props.name.lower() or (props.major == 7 and props.minor == 5)
        #     if is_t4_level and not args.no_fp16:
        #         args.no_fp16 = True
        #         print("\n[T4/GFPGAN-TRT] 为防止 FP16 溢出噪斑，已自动禁用半精度并启用 FP32 推理。")
        # except Exception:
        #     pass
        # 实测 A10 GPU 存在同样的问题，禁用所有 GPU 的 GFPGAN-TRT + FP16 组合
        args.no_fp16 = True
        print("\n[GFPGAN-TRT] 为防止 FP16 溢出噪斑，已自动禁用半精度并启用 FP32 推理。")
    # ────────────────────────────────────────────────────────────────────
    print()
    
    main_optimized(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()