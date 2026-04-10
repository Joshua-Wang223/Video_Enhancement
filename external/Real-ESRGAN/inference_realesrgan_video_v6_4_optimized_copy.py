#!/usr/bin/env python3
"""
Real-ESRGAN Video Enhancement v6.3 - 架构优化版 (最终修复版)
基于v6.2代码重构，实现深度流水线、GPU内存池和异步计算优化
修复了视频末尾卡死的问题，增强退出机制。

FIX-DET-THRESHOLD: 增加人脸检测置信度阈值，过滤低质量检测，减少不必要的 GFPGAN 推理
FIX-ADAPTIVE-BATCH: 基于人脸密度的自适应批处理大小，提升 GPU 利用率
FIX-GPU-PREFETCH: SR 推理完成后预取下一批 H2D 传输，重叠传输与计算
"""

import os
import sys
import time
import select
import queue
import threading
import concurrent.futures
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import ffmpeg
import fractions
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
    """通过FFmpeg pipe读取视频帧"""
    
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
        
        # 创建FFmpeg输入流
        self._ffmpeg_input = ffmpeg.input(input_path)
        
        # 帧队列
        self._frame_queue = queue.Queue(maxsize=prefetch_factor)
        self._running = True
        
        # 启动读取线程
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
    
    def _read_loop(self):
        """后台读取帧的线程"""
        try:
            # 设置FFmpeg输出格式
            process = (
                self._ffmpeg_input
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', vsync=0)
                .run_async(pipe_stdout=True, quiet=True)
            )
            
            frame_size = self.width * self.height * 3
            
            while self._running:
                # 读取一帧
                in_bytes = process.stdout.read(frame_size)
                if not in_bytes:
                    break
                
                # 转换为numpy数组
                frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
                # FIX: 带超时和运行检测的安全推入
                while self._running:
                    try:
                        self._frame_queue.put(frame, timeout=1.0)
                        break
                    except queue.Full:
                        continue
            
            while self._running:
                try:
                    self._frame_queue.put(None, timeout=1.0)
                    break
                except queue.Full:
                    continue
            process.wait()
            
        except Exception as e:
            print(f"FFmpegReader读取错误: {e}")
            try:
                self._frame_queue.put(None, timeout=1.0)
            except Exception:
                pass
    
    # FIX-PREMATURE-EOF: 超时哨兵，与真正的 EOF None 区分
    FRAME_TIMEOUT = object()

    def get_frame(self):
        """获取一帧。返回 None=EOF，返回 FRAME_TIMEOUT=队列暂时为空（继续重试）"""
        try:
            return self._frame_queue.get(timeout=2.0)
        except queue.Empty:
            return FFmpegReader.FRAME_TIMEOUT  # FIX-PREMATURE-EOF
    
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
        video_stream = ffmpeg.input(
            'pipe:',
            format='rawvideo',
            pix_fmt='rgb24',
            s=f'{self.width}x{self.height}',
            r=self.fps,
        )

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
                print(f'[FFmpegWriter] 合并音轨: {_tmp} + {_src} → {self.output_path}',
                      flush=True)
                _mux_cmd = [
                    'ffmpeg', '-y',
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

# 添加缺失的函数定义
def _make_detect_helper(face_enhancer, device):
    """
    创建独立的 FaceRestoreHelper 实例，专供后台 detect 线程使用。
    与主线程 face_enhancer.face_helper 互为独立对象，无共享状态，线程安全。
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
        device         = device,
    )


def _detect_faces_batch(frames: List[np.ndarray], helper,
                         det_threshold: float = 0.5) -> Tuple[List[dict], int, int, int]:
    """
    在原始低分辨率帧上检测人脸，返回序列化检测结果。
    使用 _make_detect_helper() 创建的独立实例，可在后台线程与 SR 推理并行调用。
    每项 dict 包含：crops（对齐 crop）、affines（仿射矩阵）、orig（原始帧引用）。

    FIX-DET-THRESHOLD: 增加 det_threshold 参数，过滤低于该置信度的检测结果。
    检测模型（retinaface_resnet50）输出的 det_faces 每行格式为
    [x1, y1, x2, y2, confidence]，confidence ∈ [0, 1]。
    提高阈值可有效过滤：
      · 模糊/远景人脸（检测器不确定，GFPGAN 增强效果也差）
      · 非人脸误检（海报、雕像、动物等）
    减少 GFPGAN 推理次数 → 直接提升整体处理速度，尤其在人脸密集场景中效果显著。

    返回 4-tuple: (face_data, faces_with_face_count, total_face_count, filtered_count)
    """
    face_data = []
    _total_filtered = 0   # FIX-DET-THRESHOLD: 被阈值过滤的人脸总数
    for orig_frame in frames:
        helper.clean_all()
        helper.read_image(orig_frame)
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
            for _fi, _face in enumerate(helper.det_faces):
                # det_faces 格式: [x1, y1, x2, y2, confidence] 或 numpy array
                _score = float(_face[4]) if len(_face) > 4 else float(_face[-1])
                if _score >= det_threshold:
                    keep_indices.append(_fi)
            _after_count = len(keep_indices)
            if _after_count < _before_count:
                _n_removed = _before_count - _after_count
                _total_filtered += _n_removed
                # 同步过滤 all_landmarks_5 和 det_faces，保持索引一致
                if hasattr(helper, 'all_landmarks_5') and helper.all_landmarks_5:
                    helper.all_landmarks_5 = [helper.all_landmarks_5[_ki]
                                              for _ki in keep_indices]
                helper.det_faces = [helper.det_faces[_ki] for _ki in keep_indices]

        helper.align_warp_face()
        face_data.append({
            'crops':   [c.copy() for c in helper.cropped_faces],
            'affines': [a.copy() for a in helper.affine_matrices],
            'orig':    orig_frame,
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

    if compute_stream is not None:
        compute_stream.synchronize() # 仅同步计算流
    else:
        torch.cuda.current_stream(device).synchronize()

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
                        out = face_enhancer.gfpgan(sub_batch, return_rgb=False, weight=gfpgan_weight)
                        if isinstance(out, (tuple, list)):
                            out = out[0]
                    out = out.float()
                for out_t in out.unbind(0):
                    restored = tensor2img(out_t, rgb2bgr=True, min_max=(-1, 1))
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
    """人脸贴回函数"""
    import cv2 as _cv2
    expected_h, expected_w = sr_results[0].shape[:2]
    final_frames = []

    for fi, (fd, sr_frame) in enumerate(zip(face_data, sr_results)):
        if not restored_by_frame[fi]:
            final_frames.append(sr_frame)
            continue
        try:
            face_enhancer.face_helper.clean_all()
            # FIX-INPUT-IMG: 必须传入原始低分辨率帧（非 SR 帧）
            face_enhancer.face_helper.read_image(fd['orig'])
            _raw = restored_by_frame[fi]
            _affines = fd['affines']
            _crops   = fd['crops']
            _n = min(len(_raw), len(_affines), len(_crops))
            valid_pairs = [(rf, _affines[j], _crops[j])
                           for j, rf in enumerate(_raw[:_n]) if rf is not None]
            if not valid_pairs:
                final_frames.append(sr_frame)
                continue
            valid_restored, valid_affines, valid_crops = zip(*valid_pairs)
            face_enhancer.face_helper.affine_matrices = list(valid_affines)
            face_enhancer.face_helper.cropped_faces   = list(valid_crops)
            for rf in valid_restored:
                face_enhancer.face_helper.add_restored_face(rf)
            face_enhancer.face_helper.get_inverse_affine(None)
            _ret = face_enhancer.face_helper.paste_faces_to_input_image(
                upsample_img=sr_frame)
            result = _ret if _ret is not None else getattr(
                face_enhancer.face_helper, 'output', None)
            result = result if result is not None else sr_frame
        except Exception as e:
            print(f'[face_enhance] 帧{fi} 贴回异常，使用 SR 结果: {e}')
            result = sr_frame

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
        serialized = builder.build_serialized_network(network, config)
        del config, parser, network, builder
        import gc; gc.collect()
        if serialized is None:
            print('[TensorRT] Engine 构建失败')
            return
        with open(trt_path, 'wb') as f:
            f.write(serialized)
        del serialized
        print(f'[TensorRT] Engine 已缓存: {trt_path}')

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
        self._context = self._engine.create_execution_context()
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
            _test = gfpgan_net(dummy_d, return_rgb=False)
        _returns_tuple = isinstance(_test, (tuple, list))
        _w = float(gfpgan_weight)
        class _W(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = gfpgan_net
            def forward(self, x):
                out = self.net(x, return_rgb=False)
                if _returns_tuple: out = out[0]
                if abs(_w - 1.0) < 1e-6: return out
                return _w * out + (1.0 - _w) * x
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
                _sm_hint = (f'\n[Builder] 提示: {_gpu_name_b} ({_sm_str}) 可能不受此 TRT 版本支持'
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
        """启动 Phase 2 Inference 子进程"""
        self.process = self._mp_ctx.Process(target=self._worker, args=(
            self.model_path, self.gfpgan_model, self.gfpgan_weight,
            self.gfpgan_batch_size, self.use_fp16, self.use_trt,
            self.trt_cache_dir, self.task_queue, self.result_queue,
            self.ready_event,
        ), daemon=True)
        self.process.start()

    @staticmethod
    def _worker(model_path, gfpgan_model, gfpgan_weight, gfpgan_batch_size,
                use_fp16, use_trt, trt_cache_dir, task_queue, result_queue,
                ready_event=None):
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
                            _test = gfpgan_net(dummy, return_rgb=False)
                        _returns_tuple = isinstance(_test, (tuple, list))
                        _w = float(weight)
                        class _W(torch.nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.net = gfpgan_net
                            def forward(self, x):
                                out = self.net(x, return_rgb=False)
                                if _returns_tuple: out = out[0]
                                if abs(_w - 1.0) < 1e-6: return out
                                return _w * out + (1.0 - _w) * x
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
                            raise RuntimeError('create_execution_context returned None')
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
            if ready_event is not None: ready_event.set()
        elif use_trt and (gfpgan_trt_accel is None or not gfpgan_trt_accel.available):
            print('[GFPGANSubprocess] TRT 失败但 context 正常，以 PyTorch 模式服务', flush=True)
            if ready_event is not None: ready_event.set()
        else:
            print('[GFPGANSubprocess] PyTorch 模式，进入任务循环', flush=True)
            if ready_event is not None: ready_event.set()

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
                            if 'out of memory' in _ve_str:
                                gfpgan_trt_accel._trt_ok = False
                                use_trt = False
                                result_queue.put(('__validate__', _val_id, False), timeout=5.0)
                            else:
                                result_queue.put(('__validate__', _val_id, False), timeout=5.0)
                                os._exit(0)
                        else:
                            gfpgan_trt_accel._trt_ok = False
                            use_trt = False
                            result_queue.put(('__validate__', _val_id, False), timeout=5.0)
                else:
                    result_queue.put(('__validate__', _val_id, False), timeout=5.0)
                continue

            if isinstance(task, tuple) and len(task) == 2 and task[0] == '__pause__':
                _pause_duration = task[1]
                torch.cuda.empty_cache()
                time.sleep(_pause_duration)
                torch.cuda.empty_cache()
                continue

            task_id, crops_np = task
            crops_tensor = []
            for crop in crops_np:
                t = img2tensor(crop / 255., bgr2rgb=True, float32=True)
                _tv_normalize(t, (0.5,0.5,0.5), (0.5,0.5,0.5), inplace=True)
                crops_tensor.append(t)
            if not crops_tensor:
                result_queue.put((task_id, []), timeout=5.0)
                continue

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
                        else:
                            with fp16_ctx:
                                out = model(sub_batch, return_rgb=False, weight=gfpgan_weight)
                                if isinstance(out, (tuple, list)): out = out[0]
                            out = out.float()
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

            restored = []
            for out_t in all_out:
                if out_t is None:
                    restored.append(None)
                else:
                    img = tensor2img(out_t, rgb2bgr=True, min_max=(-1,1))
                    restored.append(img.astype('uint8'))
            try:
                result_queue.put((task_id, restored), timeout=5.0)
            except queue.Full:
                pass

        if gfpgan_trt_accel is not None:
            gfpgan_trt_accel._trt_ok = False
        try: del face_enhancer, model
        except Exception: pass
        try: torch.cuda.empty_cache()
        except Exception: pass
        import os as _os_worker_exit
        _os_worker_exit._exit(0)

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
        if not self.process or not self.process.is_alive():
            return
        try:
            self.task_queue.put(('__pause__', duration), timeout=2.0)
        except queue.Full:
            pass

    def close(self):
        if self.process and self.process.is_alive():
            try: self.task_queue.put(None, timeout=3)
            except Exception: pass
            self.process.join(timeout=15)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)
        try: self.task_queue.close()
        except Exception: pass
        try: self.result_queue.close()
        except Exception: pass


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
        self.frame_queue = queue.Queue(maxsize=48)
        self.detect_queue = queue.Queue(maxsize=32)
        self.sr_queue = queue.Queue(maxsize=16)
        self.gfpgan_queue = queue.Queue(maxsize=16)
        
        # GPU内存池
        self.memory_pool = GPUMemoryPool(
            max_batches=8,
            batch_size=self.optimal_batch_size,
            img_size=(input_h, input_w),
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
    
    def optimize_pipeline(self, reader, writer, pbar, total_frames):
        """运行优化的深度流水线"""
        
        print("[优化架构] 启动深度流水线处理...")
        print(f"[优化架构] 队列深度: F{self.frame_queue.maxsize}/D{self.detect_queue.maxsize}/S{self.sr_queue.maxsize}/G{self.gfpgan_queue.maxsize}")
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
            max_elapsed = 2700
            deadline = time.time() + max_elapsed
            ready = False
            _poll_interval = 5
            _report_every  = 300
            _last_report   = time.time() - _report_every
            while time.time() < deadline:
                if not self.gfpgan_subprocess.process.is_alive():
                    exitcode = self.gfpgan_subprocess.process.exitcode
                    if exitcode == 0:
                        print('[优化架构] GFPGAN 子进程因 CUDA context 污染主动退出')
                    else:
                        print(f'[优化架构] GFPGAN 子进程意外退出（exitcode={exitcode}），回退 PyTorch')
                    break
                if self.gfpgan_subprocess.ready_event.wait(timeout=_poll_interval):
                    time.sleep(1.0)
                    if not self.gfpgan_subprocess.process.is_alive():
                        exitcode = self.gfpgan_subprocess.process.exitcode
                        if exitcode == 0:
                            print('[优化架构] GFPGAN TRT warmup 失败，子进程主动退出')
                        else:
                            print(f'[优化架构] GFPGAN 子进程 ready 后意外退出（exitcode={exitcode}）')
                        break
                    ready = True
                    break
                now = time.time()
                if now - _last_report >= _report_every:
                    elapsed = now - (deadline - max_elapsed)
                    print(f'[优化架构] 等待中... {elapsed:.0f}s', flush=True)
                    _last_report = now
            if ready:
                print('[优化架构] GFPGAN 子进程已就绪，启动流水线')
            else:
                print('[优化架构] GFPGAN 子进程未就绪，回退 PyTorch 路径')
                self.gfpgan_subprocess = None

        read_thread = threading.Thread(target=self._read_frames, args=(reader,), daemon=True)
        read_thread.start()
        detect_thread = threading.Thread(target=self._detect_faces, daemon=True)
        detect_thread.start()
        sr_thread = threading.Thread(target=self._process_sr, daemon=True)
        sr_thread.start()
        gfpgan_thread = threading.Thread(target=self._process_gfpgan, daemon=True)
        gfpgan_thread.start()
        
        self._read_thread = read_thread
        self._detect_thread = detect_thread
        self._sr_thread = sr_thread
        self._gfpgan_thread = gfpgan_thread
        
        self._write_frames(writer, pbar, total_frames)
        
        read_thread.join()
        detect_thread.join()
        sr_thread.join()
        gfpgan_thread.join()
    
    def _read_frames(self, reader):
        """读取视频帧到队列
        
        FIX-ADAPTIVE-BATCH: 使用 _adaptive_read_batch_size 替代固定 optimal_batch_size。
        当人脸密度低时读取更大批次（提升 SR GPU 利用率），
        当人脸密度高时读取更小批次（为 GFPGAN 腾出显存）。
        """
        batch_frames = []
        try:
            while self.running:
                try:
                    img = reader.get_frame()
                    if img is FFmpegReader.FRAME_TIMEOUT:
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
                    
                    # FIX-ADAPTIVE-BATCH: 使用自适应批处理大小
                    # 读取线程动态感知 GFPGAN 反馈的人脸密度，调整每批帧数
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

                        # POST-SR-VALIDATE
                        if not _first_batch_done and self.gfpgan_subprocess is not None:
                            _first_batch_done = True
                            print('[优化架构] 第一个 SR 批次完成，触发 GFPGAN TRT post-SR 验证...', flush=True)
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            _val_ok = self.gfpgan_subprocess.post_sr_validate()
                            if _val_ok:
                                print('[优化架构] GFPGAN TRT post-SR 验证通过，TRT 推理正式启用', flush=True)
                            else:
                                self.gfpgan_subprocess.process.join(timeout=1.5)
                                if not self.gfpgan_subprocess.process.is_alive():
                                    print('[优化架构] GFPGAN 子进程因 CUDA context 损坏已退出，降级', flush=True)
                                    self.gfpgan_subprocess = None
                                else:
                                    print('[优化架构] GFPGAN 子进程以 PyTorch FP16 路径服务', flush=True)

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
            if not _sentinel_sent:
                try:
                    self.sr_queue.put((None, None, None, None, True), timeout=3.0)
                except Exception:
                    pass
    
    
    def _process_gfpgan(self):
        """GFPGAN处理 - 支持子进程TRT
        
        FIX-ADAPTIVE-BATCH: 每批处理后更新人脸密度 EMA，
        动态调整 _adaptive_read_batch_size 反馈给 _read_frames 线程。
        """
        _sentinel_sent = False  # FIX-DUP-SENTINEL
        try:
            while self.running:
                try:
                    item = self.sr_queue.get(timeout=1.0)
                    
                    if item is None:
                        self.gfpgan_queue.put(None)
                        _sentinel_sent = True
                        break
                    
                    batch_frames, face_data, memory_block, sr_results, is_end = item
                    
                    if batch_frames is None:
                        self.gfpgan_queue.put((None, None, True))
                        _sentinel_sent = True
                        break
                    # FIX: 提前检查 face_data 是否有效
                    has_valid_faces = (face_data is not None and 
                                      len(face_data) > 0 and 
                                      any(fd.get('crops') for fd in face_data if fd))

                    # FIX: 检查 GFPGAN 是否可用（子进程或主进程）
                    gfpgan_available = False
                    num_frames = len(face_data) if face_data else 0
                    _n_faces_this_batch = sum(len(fd.get("crops", [])) for fd in face_data or [])
                    avg_faces = _n_faces_this_batch / num_frames if num_frames > 0 else 0.0
                    if self.gfpgan_subprocess and self.gfpgan_subprocess.process.is_alive():
                        gfpgan_available = True
                        print(f'[GFPGAN] 使用子进程TRT处理 {_n_faces_this_batch} 个人脸。当前批次共 {num_frames} 帧，平均每帧 {avg_faces:.2f} 个人脸')
                    elif (self.face_enhancer is not None and 
                          getattr(self.face_enhancer, 'gfpgan', None) is not None):
                        gfpgan_available = True
                        print(f'[GFPGAN] 使用主进程PyTorch处理 {_n_faces_this_batch} 个人脸。当前批次共 {num_frames} 帧，平均每帧 {avg_faces:.2f} 个人脸')
                    else:
                        if has_valid_faces:
                            print(f'[GFPGAN] GFPGAN不可用，跳过人脸增强')
                    
                    # GFPGAN处理
                    if has_valid_faces and gfpgan_available:
                        try:
                            restored_by_frame = []
                            
                            if self.gfpgan_subprocess and self.gfpgan_subprocess.process.is_alive():
                                all_crops = []
                                crops_per_frame = []
                                for fd in face_data:
                                    crops = fd.get('crops', [])
                                    crops_per_frame.append(len(crops))
                                    all_crops.extend(crops)

                                if all_crops:
                                    all_restored = self.gfpgan_subprocess.infer(all_crops)
                                    idx = 0
                                    for count in crops_per_frame:
                                        restored_by_frame.append(
                                            all_restored[idx:idx + count] if count else []
                                        )
                                        idx += count
                                else:
                                    restored_by_frame = [[] for _ in face_data]
                            else:
                                if (self.face_enhancer is None or 
                                    getattr(self.face_enhancer, 'gfpgan', None) is None):
                                    raise RuntimeError("face_enhancer 或 gfpgan 为 None")
                                    
                                restored_by_frame, _ = _gfpgan_infer_batch(
                                    face_data, self.face_enhancer, self.device,
                                    None, self.args.gfpgan_weight, 
                                    getattr(self.args, 'gfpgan_batch_size', 4), None, None
                                )
                            
                            if not restored_by_frame or all(r is None or len(r) == 0 for r in restored_by_frame):
                                final_frames = sr_results
                            else:
                                future = self.paste_executor.submit(
                                    _paste_faces_batch, face_data, restored_by_frame, 
                                    sr_results, self.face_enhancer
                                )
                                try:
                                    final_frames = future.result(timeout=60)
                                except concurrent.futures.TimeoutError:
                                    final_frames = sr_results
                                
                        except Exception as e:
                            print(f"GFPGAN处理错误: {e}")
                            import traceback; traceback.print_exc()
                            final_frames = sr_results
                    else:
                        final_frames = sr_results
                    
                    # ── FIX-ADAPTIVE-BATCH: 更新人脸密度 EMA 并调整自适应批处理大小 ──
                    # 每批 GFPGAN 处理完成后，根据实际人脸数更新密度估计。
                    # 低密度 → 增大读取批次（SR 处理更多帧/GPU调用，摊薄 kernel launch 开销）
                    # 高密度 → 减小读取批次（减少单次 SR 占用的显存，为 GFPGAN TRT 留出空间）
                    if self._enable_adaptive_batch and face_data is not None:
                        _frames_in_batch = max(1, len(face_data))
                        _cur_density = _n_faces_this_batch / _frames_in_batch
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
                    
                    while self.running:
                        try:
                            self.gfpgan_queue.put((final_frames, memory_block, is_end), timeout=1.0)
                            break
                        except queue.Full:
                            continue
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"GFPGAN处理错误: {e}")
                    import traceback; traceback.print_exc()
                    if 'memory_block' in locals() and memory_block is not None:
                        try:
                            self.memory_pool.release(memory_block['index'])
                        except Exception: pass
        finally:
            if not _sentinel_sent:
                try:
                    for _ in range(3):
                        try:
                            self.gfpgan_queue.put((None, None, True), timeout=5.0)
                            break
                        except queue.Full:
                            continue
                except Exception:
                    pass
                
    def close(self):
        """清理资源"""
        print("[Pipeline] 正在停止流水线...", flush=True)
        self.running = False
        
        for q_name, q in [('frame', self.frame_queue), ('detect', self.detect_queue),
                          ('sr', self.sr_queue), ('gfpgan', self.gfpgan_queue)]:
            try:
                q.put(None, timeout=1.0)
            except (queue.Full, Exception):
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
                thread.join(timeout=5.0)
                if thread.is_alive():
                    print(f"[Pipeline] 线程 {name} 未响应，已放弃等待", flush=True)
                    if not thread.is_alive():
                        thread.daemon = True
        print("[Pipeline] 所有流水线线程已关闭", flush=True)
    
    def _write_frames(self, writer, pbar, total_frames):
        """写入帧处理"""
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
                    
                    if memory_block is not None:
                        try:
                            self.memory_pool.release(memory_block['index'])
                        except Exception: pass
                    
                    pbar.update(len(final_frames))
                    self.meter.update(len(final_frames))
                    
                    current_fps = self.meter.fps()
                    eta = self.meter.eta(total_frames)
                    avg_ms = np.mean(self.timing[-10:]) * 1000 if self.timing else 0
                    
                    pbar.set_postfix(
                        fps=f'{current_fps:.1f}',
                        eta=f'{eta:.0f}s',
                        bs=self.optimal_batch_size,      # FIX-BS-DISPLAY: 显示 SR 实际 batch_size，OOM 降级后立即体现
                        ms=f'{avg_ms:.0f}',
                        queue_sizes=f"F:{self.frame_queue.qsize()}/D:{self.detect_queue.qsize()}/S:{self.sr_queue.qsize()}/G:{self.gfpgan_queue.qsize()}"
                    )

                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        if allocated > 0.9 * reserved:
                            print(f'\n[资源警告] GPU内存压力过高: {allocated:.2f}GB / {reserved:.2f}GB')
                    
                    # 每跨越 20 帧输出一次详细日志
                    if written_count // 20 > (written_count - len(final_frames)) // 20:
                        # FIX-DET-THRESHOLD + FIX-ADAPTIVE-BATCH: 增加过滤计数和密度信息
                        _density_str = f' | 密度EMA={self._face_density_ema:.1f}' if self._enable_adaptive_batch else ''
                        _filtered_str = f' | 过滤{self._face_filtered_total}' if self._face_filtered_total > 0 else ''
                        _adaptive_str = f' | 自适应bs={self._adaptive_read_batch_size}' if self._enable_adaptive_batch else ''
                        print(f"\n[性能监控] 帧{written_count}/{total_frames} | fps={current_fps:.1f} | eta={eta:.0f}s | bs={self.optimal_batch_size} | ms={avg_ms:.0f} | 队列 F:{self.frame_queue.qsize()}/D:{self.detect_queue.qsize()}/S:{self.sr_queue.qsize()}/G:{self.gfpgan_queue.qsize()} | 人脸 {self._face_count_total}张/{self._face_frames_total}帧{_filtered_str}{_density_str}{_adaptive_str}")
                    
                except queue.Empty:
                    if received_end_sentinel and self.gfpgan_queue.qsize() == 0:
                        print(f"[Pipeline] 收到哨兵且 gfpgan_queue 已清空，退出。"
                              f"已写入 {written_count}/{total_frames} 帧", flush=True)
                        break
                    if written_count >= total_frames and received_end_sentinel:
                        print(f"[Pipeline] 所有帧已写入且收到结束信号，退出。"
                              f"已写入 {written_count}/{total_frames} 帧", flush=True)
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
        print(f'\n[优化架构] GFPGAN 子进程已绑定到流水线（模式: {gfpgan_mode}）')
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
                      f"最终自适应bs: {pipeline._adaptive_read_batch_size}")


def main():
    """主函数 - 参数解析"""
    parser = argparse.ArgumentParser(
        description='Real-ESRGAN 视频超分 —— 架构优化版 v6.3',
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
    
    args = parser.parse_args()

    # FIX-ADAPTIVE-BATCH: 处理 --no-adaptive-batch 覆盖
    if args.no_adaptive_batch:
        args.adaptive_batch = False
    
    print("Real-ESRGAN Video Enhancement v6.3 - 架构优化版")
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
    print()
    
    main_optimized(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()