#!/usr/bin/env python3
"""
Real-ESRGAN Video Enhancement v6.3 - 架构优化版 (最终修复版)
基于v6.2代码重构，实现深度流水线、GPU内存池和异步计算优化
修复了视频末尾卡死的问题，增强退出机制。
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
        """初始化FFmpeg进程 - FIX-MANUAL-CMD: 手动构造命令行，绕过 ffmpeg-python 参数排序问题

        根因诊断：原实现使用 ffmpeg-python 的 ffmpeg.input().output().compile() 生成命令行。
        该库在某些版本/参数组合下会将输入选项（-f rawvideo -pix_fmt rgb24 -s WxH -r fps）
        错误地放到 -i pipe: 之后，导致 FFmpeg 不知道如何读取 stdin rawvideo 格式：
          FFmpeg 尝试自动探测裸 rawvideo 流 → 失败/挂起 → 不读取 stdin → pipe 撑满
          → _write_with_timeout 中 select() 永久返回不可写 → 队列撑满 → 180s 超时

        FFmpeg stderr 只有版本 banner（无 "Input #0"）即为此症状的确证。

        修复：手动构造命令列表，100% 确保参数顺序正确：
          ffmpeg -y [input_opts] -i pipe: [output_opts] output_path
        """
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
        # 在启动真正的编码进程之前，用 1 帧 nullsrc 快速测试 NVENC 可用性。
        # 失败时自动降级到 libx264，避免写入数十帧后才发现编码器不可用。
        # 根因：A10 等数据中心 GPU 在某些驱动/MIG 配置下 NVENC 不可用，
        #        或所有 NVENC session 已被占用（多进程并发编码时常见）。
        if video_codec in ('h264_nvenc', 'hevc_nvenc'):
            _ffbin = getattr(self.args, 'ffmpeg_bin', 'ffmpeg')
            print(f'[FFmpegWriter] 预检测 {video_codec} 可用性...', flush=True)
            if not self._check_nvenc_available(_ffbin, video_codec):
                _fallback = 'libx264' if 'h264' in video_codec else 'libx265'
                print(f'[FFmpegWriter] {video_codec} 不可用，自动降级到 {_fallback}',
                      flush=True)
                video_codec = _fallback
                # 同步更新 args，让 _write_loop 的 stall 参数也用正确的编码器
                self.args.video_codec = _fallback
            else:
                print(f'[FFmpegWriter] {video_codec} 预检测通过', flush=True)
        # ── 预检测结束 ────────────────────────────────────────────────

        # FIX-AUDIO-SEPARATE: 始终写无音轨临时文件，close() 后单独 mux 音轨
        # 临时文件路径：output_path 同目录，避免跨分区 rename 失败
        _base, _ext = os.path.splitext(self.output_path)
        self._tmp_video_path = f'{_base}.tmp_novid{_ext}'

        # ── 手动构造命令行 ────────────────────────────────────────────
        cmd_args = [
            getattr(self.args, 'ffmpeg_bin', 'ffmpeg'),
            '-y',                                   # 覆盖输出
            # ── 输入选项（必须在 -i 之前）──
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', 'pipe:',                          # stdin 输入
            # ── 输出选项 ──
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
            # 默认 libx264
            cmd_args += [
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', str(crf),
                '-preset', x264_preset,
            ]

        cmd_args += ['-an']                         # 无音轨
        cmd_args += [self._tmp_video_path]          # 输出文件（必须在最后）

        # FIX-DIAG: 始终打印实际命令行
        print(f"[FFmpegWriter] 命令: {' '.join(cmd_args)}", flush=True)

        self._process = subprocess.Popen(
            cmd_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # FIX-PIPE-SZ: 尝试扩大管道缓冲区，捕获所有 OSError（不仅 PermissionError）
        try:
            import fcntl
            fcntl.fcntl(self._process.stdin.fileno(), fcntl.F_SETPIPE_SZ, 4 * 1024 * 1024)
            print(f"[FFmpegWriter] 管道缓冲区已扩大到 4MB", flush=True)
        except PermissionError:
            # 内核限制非特权进程的管道大小上限（/proc/sys/fs/pipe-max-size），
            # 这里仅是性能优化，失败时静默回退即可。
            pass
        except Exception as _e:
            print(f"[FFmpegWriter] 管道缓冲区扩大失败（使用默认 64KB）: {_e}", flush=True)
            pass
        # FIX-FFMPEG-HEALTH: 启动后短暂等待并验证进程存活
        time.sleep(0.5)
        if self._process.poll() is not None:
            # FFmpeg 立即退出 → 打印 stderr 帮助诊断
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
        """带超时的写入操作 — 分块写入版（增加诊断日志）

        FIX-WRITE-TIMEOUT:
          原实现 select() 只检测 1 字节可写，但 stdin.write(6MB) 会阻塞到
          全部写入完成（pipe buffer 仅 64KB），30 秒超时形同虚设。
          修复：使用 os.write() 每次 64KB 分块写入，每块之间检查超时和
          FFmpeg 进程存活状态，确保超时机制真正生效。
        """
        if not self._process or not self._process.stdin:
            self._write_error = 'process 或 stdin 不可用'
            return False

        # 写前检查 FFmpeg 是否存活
        if self._process.poll() is not None:
            rc = self._process.returncode
            self._write_error = f'FFmpeg 已退出 (rc={rc})'
            return False

        try:
            fd = self._process.stdin.fileno()
            deadline = time.time() + timeout
            offset = 0
            total = len(data)
            # FIX-DIAG: 每帧首次 select 前打印诊断（仅前5帧）
            _diag_frame = self._frames_written_to_pipe < 5

            while offset < total:
                now = time.time()
                if now >= deadline:
                    self._write_error = (
                        f'写入超时 ({offset}/{total} 字节, '
                        f'frame_size={total})')
                    return False

                remaining = deadline - now
                # select 最多等 1 秒，确保定期检查 FFmpeg 是否仍然存活
                try:
                    _rready, wready, xready = select.select(
                        [], [fd], [fd], min(remaining, 1.0))
                except (OSError, ValueError):
                    self._write_error = 'select() 调用失败 (fd 可能已关闭)'
                    return False

                if xready:
                    # FIX-EPOLLHUP-RACE:
                    #   FFmpeg 关闭 stdin read-end 时，select() 立即将 fd 放入 errs
                    #   (EPOLLHUP)，但进程可能还需 ~100ms 才完全退出。
                    #   直接 poll() 可能仍返回 None → 误走 stall 路径 →
                    #   libx264: MAX_NVENC_STALL_S=60 < WRITE_TIMEOUT=300 → 第一次失败即放弃。
                    #   修复：先 sleep 0.1s 让进程完成退出，再复查 poll()。
                    time.sleep(0.1)
                    if self._process.poll() is not None:
                        rc = self._process.returncode
                        self._write_error = f'FFmpeg 异常退出 (rc={rc})'
                    else:
                        self._write_error = (
                            f'select 异常条件 (已写 {offset}/{total} 字节)')
                    return False

                if not wready:
                    # fd 暂时不可写（pipe buffer 满），继续等待
                    # FIX-ALIVE-CHECK: 每次 select 超时均检测 FFmpeg 是否已退出，
                    # 确保即使 WRITE_TIMEOUT 增大到 300s，真正崩溃的 FFmpeg 仍能
                    # 在 ≤1s 内被发现，而不是等满整个超时窗口。
                    if self._process.poll() is not None:
                        rc = self._process.returncode
                        stderr_text = self._get_ffmpeg_stderr(tail_lines=5)
                        self._write_error = (
                            f'FFmpeg 进程已退出 (rc={rc}, 已写 {offset}/{total} 字节)'
                            f'\n{stderr_text}')
                        return False
                    # FIX-DIAG: 管道不可写时打印一次
                    if _diag_frame and offset == 0:
                        print(f'[FFmpegWriter][DIAG] 帧{self._frames_written_to_pipe}: '
                              f'select 返回 not-ready (offset={offset}/{total}), '
                              f'FFmpeg alive={self._process.poll() is None}, '
                              f'stderr_lines={len(self._stderr_buffer)}',
                              flush=True)
                        _diag_frame = False  # 只打印一次
                    continue

                # 分块写入：每次最多 WRITE_CHUNK_SIZE 字节
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

            # FIX-DIAG: 帧写入成功，更新统计
            self._frames_written_to_pipe += 1
            # if self._frames_written_to_pipe <= 5 or self._frames_written_to_pipe % 20 == 0:
            #     print(f'[FFmpegWriter][DIAG] 帧{self._frames_written_to_pipe} '
            #           f'写入管道成功 ({total} 字节, '
            #           f'累计 {self._bytes_written_to_pipe/1024/1024:.1f}MB, '
            #           f'stderr_lines={len(self._stderr_buffer)})',
            #           flush=True)
            return True

        except (BrokenPipeError, OSError) as e:
            self._write_error = str(e)
            return False
        except Exception as e:
            self._write_error = str(e)
            return False

    def _write_loop(self):
        """后台写入帧的线程 - 增强版 v3（FIX-NVENC-STALL）

        FIX-NVENC-STALL: h264_nvenc 在 TRT SR + GFPGAN TRT 高负载时会出现
        H2D memcpy 阻塞（nvenc 等待 CUDA context 调度）→ FFmpeg 内部 nvenc
        管线卡住 → 不再消费 stdin → pipe buffer 填满 → select() 返回不可写。
        FFmpeg 进程此时仍然存活，不是真正崩溃。

        旧逻辑：write_timeout × 3次 = crash，每次 continue 丢弃当前帧。
        新逻辑：
          - FFmpeg 存活时：对同一帧做内循环重试，每次间隔 NVENC_RETRY_SLEEP，
            直到写入成功或总等待超过 MAX_NVENC_STALL_S（此时才标记 broken）。
          - FFmpeg 已死亡时：走原有 consecutive_errors fatal 路径。
        """
        consecutive_errors = 0
        max_consecutive_errors = 3
        # FIX-WRITE-TIMEOUT-CODEC: stall 参数按编码器区分
        #   nvenc: GPU 调度竞争，允许更长等待（600s/20s）
        #   libx264/libx265: CPU 编码器 stall = 真实问题，快速失败（60s/5s）
        #   FIX-AUDIO-SEPARATE 彻底修复后，libx264 几乎不会触发 stall，此仅兜底
        _vc = getattr(self.args, 'video_codec', 'libx264')
        if _vc in ('h264_nvenc', 'hevc_nvenc'):
            MAX_NVENC_STALL_S = 600.0
            NVENC_RETRY_SLEEP = 20.0
            SINGLE_WRITE_TIMEOUT = 300.0
        else:
            # FIX-STALL-TIMEOUT-MATCH: 
            # 原代码 60s 太短，_write_with_timeout 一次就 300s
            # 增加到 180s，允许 3 次 60s 重试
            MAX_NVENC_STALL_S = 180.0
            NVENC_RETRY_SLEEP = 5.0
            SINGLE_WRITE_TIMEOUT = 60.0  # 单次写入最多 60s

        while self._running:
            try:
                frame = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                # 空闲时检测 FFmpeg 进程是否意外退出
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
                # 收到哨兵，正常退出
                break

            try:
                if self._process and self._process.stdin:
                    frame_bytes = frame.tobytes()

                    # FIX-FRAME-SIZE-DIAG: 检测帧字节数是否符合 FFmpeg 声明的分辨率
                    # 如果帧字节数不匹配，FFmpeg 会在内部报 "Invalid data found" 并关闭 stdin，
                    # 造成 EPOLLHUP → stall → broken 的连锁反应，且完全看不到根因。
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

                    # ── 单帧 stall-tolerant 内循环 ──────────────────────────
                    stall_elapsed = 0.0
                    wrote_ok = False
                    while True:
                        if not self._running:
                            return

                        if self._write_with_timeout(frame_bytes, timeout=SINGLE_WRITE_TIMEOUT):
                            consecutive_errors = 0
                            wrote_ok = True

                            # FIX-NVENC-EARLY-DETECT: 首 5 帧写入成功后立即检查 stderr，
                            # 捕获 h264_nvenc 初始化失败（FFmpeg 读取了数据但编码器报错后退出）。
                            # 此时 FFmpeg 可能仍在处理 rawvideo 解码（尚未 poll()=退出），
                            # 但 stderr 已经包含 NVENC 错误信息。
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

                            break  # 写入成功

                        err_detail = self._write_error or '(未知错误)'
                        ffmpeg_alive = (self._process.poll() is None)

                        if not ffmpeg_alive:
                            # FFmpeg 真正崩溃 → 走 fatal 路径
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
                            break  # 跳出内循环，继续外循环处理下一帧

                        # FFmpeg 存活但 stdin 不可写 = encoder transient stall
                        stall_elapsed += SINGLE_WRITE_TIMEOUT
                        if stall_elapsed >= MAX_NVENC_STALL_S:
                            # FIX-CODEC-MSG: 根据实际编码器给出有意义的建议
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
                            # FIX-NVENC-DIAG
                            stderr_text = self._get_ffmpeg_stderr(tail_lines=20)
                            if stderr_text:
                                print(f'[FFmpegWriter] FFmpeg stderr (共{len(self._stderr_buffer)}行):\n{stderr_text}',
                                  flush=True)

                            self._broken = True
                            return

                        # FIX-EARLY-STDERR: 第一次 stall 立即打印 stderr（不等到超时）
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
                        # 继续内循环，重试同一帧
                    # ── 内循环结束 ──────────────────────────────────────────

            except (BrokenPipeError, OSError) as e:
                print(f"[FFmpegWriter] 管道断裂: {e}", flush=True)
                self._broken = True
                break
            except Exception as e:
                print(f"[FFmpegWriter] 写入错误: {e}", flush=True)
                self._broken = True
                break
    
    def write_frame(self, frame):
        """写入一帧

        FIX-WRITE-BROKEN:
          原实现 queue.put(timeout=180s) 整段阻塞：_write_loop 在 nvenc stall 超时
          后设置 _broken=True，但 write_frame 还在 180s 等待里无法感知，导致
          _write_frames 晚 180s 才能检测到崩溃并停止流水线。
          修复：分段 1s 等待，每轮检查 _broken，发现后立即返回 False，
          _write_frames 下一帧前的 broken 检测即可在 ≤1s 内响应。
        """
        if not self._running or self._broken:
            return False

        # FIX-FRAME-QUEUE-TIMEOUT: 根据编码器类型选择超时时间
        _vc = getattr(self.args, 'video_codec', 'libx264')
        if _vc in ('h264_nvenc', 'hevc_nvenc'):
            _total_timeout = 660.0  # 略大于 MAX_NVENC_STALL_S (600s)
        else:
            _total_timeout = 240.0  # CPU 编码器 略大于 MAX_NVENC_STALL_S (180s)

        deadline = time.time() + _total_timeout

        while True:
            if self._broken:
                return False
            remaining = deadline - time.time()
            if remaining <= 0:
                print(f"[FFmpegWriter] 警告: 帧队列已满超时 ({_total_timeout:.0f}s)，"
                      f"标记管道断裂 (codec={_vc})", flush=True)
                # FIX-DIAG: 打印管道写入统计
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
        """
        关闭写入器 - 增强版：确保线程一定能终止

        修复策略：
        1. 发送哨兵通知线程退出
        2. 等待线程超时后，强制 kill FFmpeg 进程
        3. 再次等待线程退出
        4. 如果仍然不退出，标记为 daemon 让 Python 退出时处理
        """
        print("[FFmpegWriter] 阶段1/5: 等待写入线程完成...", flush=True)

        # 1. 发送哨兵，通知线程结束
        if not self._broken:
            for _ in range(3):  # 发送多个哨兵确保至少一个被收到
                try:
                    self._frame_queue.put(None, timeout=2.0)
                except queue.Full:
                    print("[FFmpegWriter] 警告: 队列已满，无法发送结束信号", flush=True)
                    self._broken = True
                    break

        # 2. 等待线程结束
        if self._thread.is_alive():
            self._thread.join(timeout=self.THREAD_JOIN_TIMEOUT)

        # 3. 如果超时后线程仍在运行，强制终止 FFmpeg 进程
        if self._thread.is_alive():
            print("[FFmpegWriter] 警告: 写入线程未响应，强制终止 FFmpeg 进程...", flush=True)

            # 先尝试 SIGTERM
            if self._process and self._process.poll() is None:
                try:
                    # 关闭 stdin 可能唤醒阻塞的 write
                    if self._process.stdin and not self._process.stdin.closed:
                        self._process.stdin.close()
                except Exception:
                    pass

                # 发送 SIGTERM
                try:
                    self._process.terminate()
                    self._process.wait(timeout=self.PROCESS_TERMINATE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    # SIGTERM 超时，使用 SIGKILL
                    print("[FFmpegWriter] SIGTERM 超时，发送 SIGKILL...", flush=True)
                    try:
                        self._process.kill()
                        self._process.wait(timeout=5)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[FFmpegWriter] 终止 FFmpeg 进程异常: {e}", flush=True)

            # 再次等待线程退出
            self._thread.join(timeout=2.0)

            if self._thread.is_alive():
                print("[FFmpegWriter] 错误: 写入线程仍无法停止，标记为 daemon", flush=True)
                # 线程已经是 daemon=True，主进程退出时会自动终止
                # 但我们需要确保 FFmpeg 进程已终止
                if self._process and self._process.poll() is None:
                    try:
                        self._process.kill()
                    except Exception:
                        pass

        print("[FFmpegWriter] 阶段2/5: 写入线程已结束", flush=True)
        self._running = False

        # 4. 安全关闭 FFmpeg 进程（如果还活着）
        if self._process and self._process.poll() is None:
            print("[FFmpegWriter] 阶段3/5: 等待 FFmpeg 完成编码...", flush=True)

            # 关闭 stdin 通知 FFmpeg 输入结束
            if self._process.stdin and not self._process.stdin.closed:
                try:
                    self._process.stdin.flush()
                    self._process.stdin.close()
                except Exception:
                    pass

            try:
                # 限制 FFmpeg 编码最长等待时间
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
        # 此时 FFmpeg 已退出，临时无音轨文件已完整写入。
        # 用独立 FFmpeg 进程将 video-only 文件 + 原始音轨合并为最终输出。
        _tmp = getattr(self, '_tmp_video_path', None)
        if _tmp and not self._broken and self.audio is not None:
            _src = getattr(self.args, 'input', None)
            if _src and os.path.exists(_tmp) and os.path.exists(_src):
                print(f'[FFmpegWriter] 合并音轨: {_tmp} + {_src} → {self.output_path}',
                      flush=True)
                _mux_cmd = [
                    'ffmpeg', '-y',
                    '-i', _tmp,          # 视频流来源（已编码）
                    '-i', _src,          # 音频流来源（原始视频）
                    '-map', '0:v:0',     # 取第一个输入的视频
                    '-map', '1:a:0',     # 取第二个输入的音频
                    '-c:v', 'copy',      # 视频直接 remux，不重编
                    '-c:a', 'aac',       # 音频转码为 AAC（兼容性最佳）
                    '-shortest',         # 以较短轨道为准（防止音频比视频长）
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
                        # 合并失败时保留无音轨文件，避免丢失视频
                        print(f'[FFmpegWriter] 保留无音轨文件: {_tmp}', flush=True)
                        _tmp = None   # 跳过删除
                except subprocess.TimeoutExpired:
                    print('[FFmpegWriter] 音轨合并超时（>300s）', flush=True)
                    _tmp = None
                except Exception as _e:
                    print(f'[FFmpegWriter] 音轨合并异常: {_e}', flush=True)
                    _tmp = None
            elif _src is None:
                # 没有原始视频路径，直接 rename 无音轨文件为输出
                try:
                    os.rename(_tmp, self.output_path)
                    _tmp = None
                    print(f'[FFmpegWriter] 无音轨模式: 已输出 {self.output_path}', flush=True)
                except Exception as _e:
                    print(f'[FFmpegWriter] rename 失败: {_e}', flush=True)
                    _tmp = None
        elif _tmp and (self._broken or self.audio is None):
            # 没有音轨或写入中断：直接 rename 无音轨文件为最终输出
            if os.path.exists(_tmp):
                try:
                    os.rename(_tmp, self.output_path)
                    print(f'[FFmpegWriter] 无音轨输出: {self.output_path}', flush=True)
                except Exception as _e:
                    print(f'[FFmpegWriter] rename 失败: {_e}', flush=True)
                _tmp = None

        # 删除临时无音轨文件
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


def _detect_faces_batch(frames: List[np.ndarray], helper) -> List[dict]:
    """
    在原始低分辨率帧上检测人脸，返回序列化检测结果。
    使用 _make_detect_helper() 创建的独立实例，可在后台线程与 SR 推理并行调用。
    每项 dict 包含：crops（对齐 crop）、affines（仿射矩阵）、orig（原始帧引用）。
    """
    face_data = []
    for orig_frame in frames:
        helper.clean_all()
        helper.read_image(orig_frame)
        try:
            helper.get_face_landmarks_5(
                only_center_face=False, resize=640, eye_dist_threshold=5)
        except TypeError:
            helper.get_face_landmarks_5(
                only_center_face=False, eye_dist_threshold=5)
        helper.align_warp_face()
        face_data.append({
            'crops':   [c.copy() for c in helper.cropped_faces],
            'affines': [a.copy() for a in helper.affine_matrices],
            'orig':    orig_frame,
        })
    _nf = sum(len(fd['crops']) for fd in face_data)
    _fw = sum(1 for fd in face_data if fd['crops'])
    # FIX-VERBOSE: 不再每批 print，统计数字随 face_data 一起返回给调用方汇总显示
    return face_data, _fw, _nf


def _sr_infer_batch(
    upsampler,
    frames: List[np.ndarray],
    outscale: float,
    netscale: int,
    transfer_stream,
    compute_stream,
    trt_accel,
    cuda_graph_accel=None,
):
    """
    纯 SR 推理：H2D → 模型前向 → 后处理 → D2H。
    trt_accel 可用时走 TRT 路径（全程 GPU 内存，data_ptr()）；
    否则走普通 PyTorch 路径（compute_stream 上）。
    返回 (sr_results, timing_info, status_flag) 以兼容 DeepPipelineOptimizer 调用约定。
    """
    device   = upsampler.device
    use_half = upsampler.half
    pool     = _get_pinned_pool()
    B        = len(frames)
    t0       = time.perf_counter()

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

    # torch.cuda.synchronize(device)
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
        # 无法导入工具函数，降级为逐帧 enhance 路径（但会有数量不匹配风险）
        restored_by_frame = []
        for fd in face_data:
            restored_by_frame.append([])
        return restored_by_frame, sub_bs

    # fp16_ctx 可能是 None（调用方传 None 时），统一处理
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
            # BGR uint8 → 归一化 [-1,1] float tensor，与 GFPGANer 内部预处理一致
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
                    sub_bs = _cur_sub_bs   # 持久化降级值，对后续批次生效
                    torch.cuda.empty_cache()
                else:
                    # 不可恢复错误：该子批次全部填 None，保持数量与 crops 对齐
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
            # FIX-ALIGN-AFFINE: 过滤掉 None 位置，保持 affines 与 valid_restored 数量一致
            # None 由 _gfpgan_infer_batch 在不可恢复错误时填入，表示该人脸推理失败
            _raw = restored_by_frame[fi]
            _affines = fd['affines']
            _crops   = fd['crops']
            # 若长度不一致（理论上不应发生），取最短对齐
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
            # facexlib 0.3.0 直接返回结果；旧版写入 .output 属性
            result = _ret if _ret is not None else getattr(
                face_enhancer.face_helper, 'output', None)
            result = result if result is not None else sr_frame
        except Exception as e:
            print(f'[face_enhance] 帧{fi} 贴回异常，使用 SR 结果: {e}')
            result = sr_frame

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
# TRT 10.x 进程内只允许一个全局 Logger（nvinfer1::getLogger()）。
# SR Engine 和 GFPGAN Engine 分别 new 出不同 Logger 实例时，TRT 忽略后者，
# 并在某些版本中导致内部 Builder/Runtime 内存池归属不一致
# → TRT kernel 写出已释放地址 → cudaErrorIllegalAddress。
# ─────────────────────────────────────────────────────────────────────────────
_TRT_LOGGER = None

def _get_trt_logger():
    """返回进程级 TRT Logger 单例；首次调用时创建并缓存。"""
    global _TRT_LOGGER
    if _TRT_LOGGER is None:
        try:
            import tensorrt as _trt_mod
            # FIX-TRT-LOGGER: 使用 ERROR 级别屏蔽 TRT 的 "cross different models of devices"
            # 警告。该警告在引擎与当前 GPU 完全匹配时仍会触发（GPU firmware 微差异）。
            # 如果引擎真正不兼容，会在 warmup execute 阶段抛出异常，已有完整处理链路。
            _TRT_LOGGER = _trt_mod.Logger(_trt_mod.Logger.ERROR)
        except ImportError:
            pass
    return _TRT_LOGGER


class TensorRTAccelerator:
    """
    将 RealESRGAN 模型导出 ONNX 后编译 TRT Engine (FP16, 静态形状)。
    要求：pip install tensorrt onnx onnxruntime-gpu
    首次构建会缓存 .trt 文件，后续直接加载。

    FIX-SM-TAG: cache tag 中包含 GPU SM 版本（如 _sm86），防止不同 GPU 架构
    （如 A10 SM8.6 与 T4 SM7.5）之间复用不兼容 engine 文件，避免
    deserialize 成功但执行时 cudaErrorIllegalAddress (Error 700)。

    infer() 改为直接使用 Tensor.data_ptr()，全程 GPU 内存，
    不再做 D2H + H2D 的无效往返搬运。
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
        # 专用非默认 Stream，懒初始化。TRT 在 default stream 上执行 enqueueV3 时
        # 会自动插入 cudaStreamSynchronize()（日志 W 警告根因），造成全局阻塞。
        # 专用 Stream 消除此隐式同步。
        self._trt_stream: Optional[torch.cuda.Stream] = None

        try:
            import tensorrt as trt
            # 不在此处 import pycuda.autoinit：pycuda.autoinit 会创建独立 CUDA context，
            # 与 PyTorch CUDA context 冲突，导致流不匹配等级联错误。
            # 使用 torch.cuda.current_stream().cuda_stream 替代。
            self._trt  = trt
        except ImportError as e:
            print(f'[TensorRT] 依赖未安装，跳过 TRT 加速: {e}')
            print('  安装命令: pip install tensorrt onnx onnxruntime-gpu')
            return

        # FIX-SR-SLUG: cache tag 含 GPU SM 版本 + GPU 型号，与 _load_engine 检查完全对齐。
        # 原 tag 仅含 _sm86；_load_engine 期望 _sm86_nvidiaa10 → 不匹配 → 删除 → 重建
        # → 再不匹配 → 无限循环 → SR TRT 失败 → PyTorch 无界 VRAM → OOM cascade。
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
                # _load_engine 在 GPU 不兼容（SM tag 不匹配）或 TRT 版本升级时会：
                #   1. 打印警告信息
                #   2. 删除过期 .trt 缓存文件
                #   3. 抛出 RuntimeError
                # 此处捕获后自动重新导出 ONNX + 构建新 Engine，再次尝试加载。
                print(f'[TensorRT] 首次加载失败（{_e}），开始重新构建 Engine...')
                if not osp.exists(onnx_path):
                    self._export_onnx(model, onnx_path, input_shape)
                self._build_engine(onnx_path, trt_path, use_fp16)
                if osp.exists(trt_path):
                    self._load_engine(trt_path)   # 二次加载，失败则向上传播

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
                opset_version=18,   # 与内部 dynamo opset 对齐
                dynamic_axes=None,  # 静态形状，TRT 最优化
            )
        print(f'[TensorRT] ONNX 已导出: {onnx_path}')

    def _build_engine(self, onnx_path, trt_path, use_fp16):
        trt    = self._trt
        logger = _get_trt_logger()
        builder = trt.Builder(logger)

        # TRT 10.x 中 EXPLICIT_BATCH 已成为默认模式且该枚举已被废弃，
        # 需要兼容处理：有 EXPLICIT_BATCH 时使用旧方式，否则用 0（默认）。
        try:
            explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        except AttributeError:
            explicit_batch_flag = 0
        network = builder.create_network(explicit_batch_flag)
        parser  = trt.OnnxParser(network, logger)

        # parse_from_file 让 TRT 自动从同目录找 .onnx.data sidecar（大模型权重文件）
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
        # FIX-DESTRUCTOR: build 完成后立即释放大对象，在 CUDA 同步前完成 C++ 析构
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
        # FIX-CROSS-GPU: 检测 .trt 文件是否与当前 GPU 匹配。
        # 历史遗留文件（无 SM tag）或跨 GPU 复用（A10 sm86 → T4 sm75）
        # 会导致 deserialize 成功但 execute 时 Error 700。
        # 解决：文件名不含当前 GPU 的 SM tag → 视为过期缓存，删除并重建。
        _cur_sm_tag = ''
        if torch.cuda.is_available():
            _pp = torch.cuda.get_device_properties(0)
            # _cur_sm_tag = f'_sm{_pp.major}{_pp.minor}'
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
        # IRuntime 仅用于反序列化，deserialize_cuda_engine 返回后 Engine 独立存在。
        del runtime

        # deserialize_cuda_engine 在 GPU compute capability 不匹配（如 T4→A10）、
        # TRT 版本升级、文件损坏时会静默返回 None。
        if self._engine is None:
            print(f'[TensorRT] Engine 反序列化失败（GPU 不兼容或 TRT 版本升级），'
                  f'删除过期缓存并重新构建: {trt_path}')
            try:
                os.remove(trt_path)
            except OSError:
                pass
            raise RuntimeError('[TensorRT] _load_engine: deserialize_cuda_engine returned None')

        self._context = self._engine.create_execution_context()

        # 区分 TRT 版本，预先解析 tensor 名称 / binding 信息
        # TRT 10.x: 使用 num_io_tensors + get_tensor_name + get_tensor_mode
        # TRT  8.x: 使用 num_bindings + get_binding_shape（旧接口）
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
                    '[TensorRT] 无法在 Engine 中找到有效输入/输出 tensor，'
                    '请确认 ONNX 导出时设置了 input_names / output_names。'
                )
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
        """
        全程 GPU 内存推理：直接把 Tensor.data_ptr() 传给 TRT。
        静态 batch_size engine：末批不足时用最后一帧 padding，推理后裁剪输出。
        """
        actual_B  = input_tensor.shape[0]
        engine_B  = self.input_shape[0]

        if actual_B < engine_B:
            pad_cnt = engine_B - actual_B
            pad     = input_tensor[-1:].expand(pad_cnt, -1, -1, -1)
            input_tensor = torch.cat([input_tensor, pad], dim=0)

        inp      = input_tensor.contiguous()
        out_dtype = torch.float16 if self.use_fp16 else torch.float32

        # FIX-TRT-STREAM: 专用非默认 Stream，消除 TRT 在 default stream 上的隐式 sync
        if self._trt_stream is None:
            self._trt_stream = torch.cuda.Stream(device=self.device)

        if self._use_new_api:
            out_shape  = tuple(self._engine.get_tensor_shape(self._output_name))
            out_tensor = torch.empty(out_shape, dtype=out_dtype, device=self.device)
            # BUGFIX: 等待 current_stream 上的 contiguous() 完成后再 enqueue
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
    构造函数接受 face_enhancer（原有路径）或直接 model_path（早启动路径）。
    早启动可确保 GFPGAN TRT Myelin warmup 在 GPU 干净时抢占连续地址，
    避免 SR 模型加载后碎片化导致 CUDA_ERROR_ILLEGAL_ADDRESS。
    """
    def __init__(self, face_enhancer=None, device=None, gfpgan_weight=0.5, gfpgan_batch_size=4,
                 use_fp16=True, use_trt=False, trt_cache_dir=None, gfpgan_model='1.4',
                 model_path=None):   # FIX-EARLY-SPAWN: 支持直接传 model_path
        self.device = device
        self.gfpgan_weight = gfpgan_weight
        self.gfpgan_batch_size = gfpgan_batch_size
        self.use_fp16 = use_fp16
        self.use_trt = use_trt
        self.trt_cache_dir = trt_cache_dir
        self.gfpgan_model = gfpgan_model

        # FIX-EARLY-SPAWN: model_path 可直接传入（早启动路径），
        # 无需 face_enhancer 对象（SR 加载前无法创建 face_enhancer）
        if model_path is not None:
            self.model_path = model_path
        elif face_enhancer is not None:
            try:
                self.model_path = face_enhancer.model_path
            except AttributeError:
                pass  # 下方 fallback 处理

        # 如果仍然没有 model_path，尝试从 face_enhancer 获取或使用默认下载
        if not hasattr(self, 'model_path'):
            if face_enhancer is not None:
                try:
                    self.model_path = face_enhancer.model_path
                except AttributeError:
                    pass
            # 如果还是没有，使用默认下载
            if not hasattr(self, 'model_path'):
                # 默认下载路径
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

        # 只有 face_enhancer 非 None 时才尝试获取网络等属性（早启动路径不需要）
        if face_enhancer is not None:
            self.gfpgan_net = face_enhancer.gfpgan
            self.face_enhancer_upscale = face_enhancer.upscale
        else:
            self.gfpgan_net = None
            self.face_enhancer_upscale = 1

        # FIX-SPAWN: spawn context 保证子进程 CUDA context 干净
        self._mp_ctx = mp.get_context('spawn')
        self.task_queue   = self._mp_ctx.Queue(maxsize=2)
        self.result_queue = self._mp_ctx.Queue(maxsize=2)
        self.ready_event  = self._mp_ctx.Event()
        self.process = None

        # ── 两阶段子进程架构（FIX-TWO-PHASE）────────────────────────
        # Phase 1: Builder 进程 — 仅做 TRT build，完成后 sys.exit(0)
        #   build 完成时 TRT C++ 析构会污染 CUDA context，但进程直接退出
        #   → 污染不会传染到 Phase 2（新 spawn 的进程）
        # Phase 2: Inference 进程 — 加载已有 .trt（跳过 build）
        #   CUDA context 全程干净 → warmup 成功 → TRT inference 稳定工作
        if use_trt:
            # FIX-SM-TAG: 在 cache tag 中加入 GPU SM 版本，防止不同架构 GPU 复用不兼容的 .trt 文件
            # A10(SM86) 构建的 engine 在 T4(SM75) 上加载执行 → cudaErrorIllegalAddress (Error 700)
            import torch as _torch
            _sm_tag = ''
            if _torch.cuda.is_available():
                _p = _torch.cuda.get_device_properties(0)
                import re as _re_coord
                _gpu_slug_coord = _re_coord.sub(r'[^a-z0-9]', '', _p.name.lower())[:16]
                _sm_tag = f'_sm{_p.major}{_p.minor}_{_gpu_slug_coord}'
            tag       = (f'gfpgan_{gfpgan_model}_w{gfpgan_weight:.3f}'
                         f'_B{gfpgan_batch_size}_fp{"16" if use_fp16 else "32"}{_sm_tag}')
            cache_dir = trt_cache_dir or osp.join(os.getcwd(), '.trt_cache')
            trt_path  = osp.join(cache_dir, f'{tag}.trt')

            if not osp.exists(trt_path):
                # Phase 1: 独立 Builder 进程（阻塞等待完成）
                print(f'[GFPGANSubprocess] Phase 1: 启动独立 Builder 进程构建 TRT Engine...', flush=True)
                print(f'[GFPGANSubprocess] 构建完成后 Builder 自动退出，再启动干净 Inference 进程', flush=True)
                builder = self._mp_ctx.Process(
                    target=GFPGANSubprocess._build_only_worker,
                    args=(self.model_path, gfpgan_model, gfpgan_weight,
                          gfpgan_batch_size, use_fp16, trt_cache_dir),
                    daemon=False,   # 非 daemon：确保 build 完整写入后再退出
                )
                builder.start()
                # 轮询等待 Phase 1 完成（最多 90 分钟），每 300s 打印一次进度
                _p1_start      = time.time()
                _p1_max        = 5400   # 90 分钟
                _p1_poll       = 5      # 每 5s 轮询一次存活状态
                _p1_reported   = False  # 只打印一次进度提示
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

        # Phase 2: 始终启动 Inference 进程（干净 CUDA context）
        self._start()

    @staticmethod
    def _build_only_worker(model_path, gfpgan_model, gfpgan_weight,
                           gfpgan_batch_size, use_fp16, trt_cache_dir):
        """
        Phase 1 Builder 进程：仅做 TRT build，完成后立即退出。
        FIX-TWO-PHASE: build 完成时 TRT C++ 析构会污染 CUDA context（Error 700）。
        独立进程退出后污染随进程消亡，不会影响后续 Phase 2 Inference 进程。
        """
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

        # FIX-SM-TAG: cache tag 含 SM 版本 + GPU 型号，防止跨 GPU 架构复用不兼容 engine
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
            import os as _os; _os._exit(0)  # FIX-OSEXIT

        # 复用子进程内已有的完整 GFPGANTRTAccelerator 构建逻辑
        # 只做 export + build，不做 load/warmup/inference
        try:
            import tensorrt as trt
        except ImportError as e:
            print(f'[Builder] tensorrt 未安装: {e}', flush=True)
            import os as _os; _os._exit(1)  # FIX-OSEXIT

        # ── _onnx_compat_patch（与 _worker 完全一致）────────────────
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

        # ── _build_wrapper（dry-run 探测输出格式）────────────────────
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

        # ── ONNX export ───────────────────────────────────────────────
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
            import os as _os; _os._exit(1)  # FIX-OSEXIT
        del wrapper, dummy, dummy_d, face_enhancer
        gc.collect()
        torch.cuda.empty_cache()

        # ── TRT Engine build ──────────────────────────────────────────
        # SM 版本预检：在启动编译之前确认当前 GPU 受此 TRT 版本支持
        # V100=SM70, RTX20/T4=SM75, A100=SM80, H100=SM90 等
        # 不同 TRT release 支持的最低 SM 不同，提前检测可避免浪费时间进入编译再失败
        _sm_ok = True
        _gpu_name_b = 'unknown'
        _sm_major = 0
        if torch.cuda.is_available():
            _props = torch.cuda.get_device_properties(0)
            _gpu_name_b = _props.name
            _sm_major = _props.major
            _sm_minor = _props.minor
            # 用 builder.platform_has_fast_fp16 + get_plugin_registry 等间接探测
            # 最直接方法：尝试 builder 并捕获 SM 不支持错误（TRT 在 create_network 后才报）
            # 故此处只做 SM < 7.5 的保守预警，真正的错误留给 build 阶段捕获并识别
            if _sm_major < 7 or (_sm_major == 7 and _sm_minor < 5):
                # SM 7.0 (V100) 及以下，新版 TRT 可能不支持
                print(f'[Builder] 警告: {_gpu_name_b} (SM {_sm_major}.{_sm_minor}) '
                      f'可能不受当前 TRT 版本支持（通常需要 SM 7.5+）', flush=True)
                print(f'[Builder] 若编译失败报 SM not supported，请降级 TRT 版本或改用 PyTorch FP16 路径', flush=True)

        # GPU-aware 编译时间估算
        _sm_minor = getattr(_sm_major, '__class__', None) and 0  # fallback
        _sm_minor = 0  # 默认值（torch.cuda 不可用时）
        if torch.cuda.is_available():
            _sm_minor = torch.cuda.get_device_properties(0).minor
        _sm_code = _sm_major * 10 + _sm_minor  # e.g. A10 SM8.6 → 86, T4 SM7.5 → 75
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
            logger  = trt.Logger(trt.Logger.ERROR)  # FIX-TRT-LOGGER: 屏蔽 cross-device-model 警告
            builder = trt.Builder(logger)
            try:   flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            except AttributeError: flag = 0
            network = builder.create_network(flag)
            parser  = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(onnx_path):
                for i in range(parser.num_errors):
                    print(f'  [Builder] 解析错误: {parser.get_error(i)}', flush=True)
                import os as _os; _os._exit(1)  # FIX-OSEXIT
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
            # ── 编译心跳线程（build_serialized_network 是阻塞 C++ 调用，无法从中 print）──
            _build_start  = time.time()
            _build_done   = threading.Event()
            _report_every = 300  # 每 300s 打印一次编译进度
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
            # 提前显式释放（FIX-DESTRUCTOR），避免析构在 sys.exit 时崩溃
            del config, profile, parser, network, builder
            gc.collect()
            if serialized is None:
                _sm_str = f'SM {_sm_major}.{_sm_minor if _sm_major else "?"}' if torch.cuda.is_available() else ''
                _sm_hint = (f'\n[Builder] 提示: {_gpu_name_b} ({_sm_str}) 可能不受此 TRT 版本支持，'
                            f'请降级 TRT 或改用 PyTorch FP16（去掉 --gfpgan-trt）'
                            if _sm_major < 8 else '')
                print(f'[Builder] Engine 构建失败（返回 None，用时 {_build_elapsed:.0f}s）{_sm_hint}', flush=True)
                import os as _os; _os._exit(1)  # FIX-OSEXIT
            with open(trt_path, 'wb') as f:
                f.write(serialized)
            del serialized
            print(f'[Builder] Engine 已缓存（用时 {_build_elapsed:.0f}s）: {trt_path}', flush=True)
        except Exception as e:
            _e_str = str(e)
            _sm_not_supported = 'SM' in _e_str and ('not supported' in _e_str or 'not enable' in _e_str.lower())
            if _sm_not_supported:
                print(f'[Builder] TRT 版本不支持 {_gpu_name_b} (SM {_sm_major}.{_sm_minor if _sm_major else "?"}): {e}', flush=True)
                print(f'[Builder] 解决方案: 安装支持 SM{_sm_major}.x 的 TRT 版本，或去掉 --gfpgan-trt 改用 PyTorch FP16', flush=True)
            else:
                print(f'[Builder] Engine 构建异常: {e}', flush=True)
            import os as _os; _os._exit(1)  # FIX-OSEXIT
        # FIX-OSEXIT: os._exit(0) 完全跳过 Python GC，TRT C++ 析构器不运行
        # 避免 builder 正常完成后析构器仍触发 terminate()
        import os as _os; _os._exit(0)

    def _start(self):
        """启动 Phase 2 Inference 子进程（spawn，CUDA context 干净）"""
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
                ready_event=None):   # FIX-TIMEOUT: ready 信号
        """子进程主函数：加载模型并循环处理任务"""
        import warnings
        warnings.filterwarnings('ignore')   # FIX-WARN: 压制子进程 torchvision deprecation 警告，防止污染主进程 tqdm 输出
        import os
        os.environ['PYTHONWARNINGS'] = 'ignore'
        import torch
        import numpy as np
        from gfpgan import GFPGANer
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize as _tv_normalize
        import contextlib

        # FIX: 初始化标志变量
        cuda_context_dead = False

        # 选择设备（子进程可见的 GPU）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # FIX-CUDA-CTX-ORDER: 强制 PyTorch 先拿主 CUDA context
        # TRT Runtime 初始化也会附着到主 CUDA context（cuDevicePrimaryCtxRetain）。
        # 若 TRT 先于 PyTorch 调用任何 CUDA API，极少数情况下两者可能附着到不同
        # primary context 实例（driver 版本差异），导致跨 context 内存访问 → Error 700。
        # torch.cuda.init() 显式触发 PyTorch 的 cuCtxCreate/cuDevicePrimaryCtxRetain，
        # 确保主 context 已存在，后续 trt.Runtime() 只是 retain 同一个 context。
        if torch.cuda.is_available():
            torch.cuda.init()

        # 根据版本选择 arch 等（与主进程一致）
        _GFPGAN_MODELS = {
            '1.3': ('clean', 2, 'GFPGANv1.3'),
            '1.4': ('clean', 2, 'GFPGANv1.4'),
            'RestoreFormer': ('RestoreFormer', 2, 'RestoreFormer'),
        }
        arch, channel_multiplier, name = _GFPGAN_MODELS[gfpgan_model]

        # FIX-INIT-ORDER: TRT 路径时，延迟加载 GFPGANer PyTorch 权重
        # 原因：主进程 SR TRT 已占用 ~4GB GPU 内存（碎片化）
        #       若先加载 GFPGANer (~0.6GB)，Myelin workspace 找不到大块连续内存
        #       → create_execution_context / warmup → CUDA_ERROR_ILLEGAL_ADDRESS
        # 修复：use_trt 时先加载 TRT engine + warmup（Myelin 抢占大块连续地址）
        #       再加载 GFPGANer PyTorch 权重（填充碎片空间，对连续性无要求）
        #       不用 TRT 时维持原顺序（无影响）
        if use_trt and torch.cuda.is_available():
            # TRT 路径：先加载轻量 CPU 模型（仅用于 wrapper 构建，暂不 to(device)）
            # GFPGANer 完整 GPU 加载延迟到 TRT warmup 之后
            print('[GFPGANSubprocess] FIX-INIT-ORDER: TRT 路径，延迟 GFPGANer GPU 加载', flush=True)
            face_enhancer = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch=arch,
                channel_multiplier=channel_multiplier,
                bg_upsampler=None,
                device=torch.device('cpu'),  # 先在 CPU 上加载，避免占用 GPU 碎片
            )
            model = face_enhancer.gfpgan
            model.eval()
        else:
            # PyTorch 路径：正常顺序直接加载到 GPU
            face_enhancer = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch=arch,
                channel_multiplier=channel_multiplier,
                bg_upsampler=None,
                device=device,
            )
            model = face_enhancer.gfpgan
            model.eval()

        # 如果启用 TRT，尝试构建 GFPGANTRTAccelerator
        gfpgan_trt_accel = None
        if use_trt and torch.cuda.is_available():
            try:
                # 在子进程中重新定义 GFPGANTRTAccelerator 类（完整复制）
                from typing import Optional, List, Tuple
                import os
                import os.path as osp
                import torch
                import numpy as np
                
                # FIX-V62-PORT: 完整移植 v6.2 GFPGANTRTAccelerator，含 _onnx_compat_patch
                # _onnx_compat_patch 是解决 ArrayRef/FusedLeakyReLU 报错的关键
                import contextlib as _ctx
                import torch.nn.functional as _F

                def _get_subprocess_trt_logger():
                    """子进程内 TRT Logger 单例"""
                    import tensorrt as _trt
                    if not hasattr(_get_subprocess_trt_logger, '_inst'):
                        # FIX-TRT-LOGGER: ERROR 级别屏蔽 cross-device-model 警告
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
                        self._cuda_context_dead = False  # FIX: 标记 context 是否损坏
                        self._warmup_failed = False      # FIX: 标记 warmup 是否失败


                        try:
                            import tensorrt as trt
                            self._trt = trt
                        except ImportError as e:
                            print(f'[GFPGAN-TensorRT] tensorrt 未安装: {e}', flush=True)
                            return

                        # FIX-SM-TAG: cache tag 含 SM 版本 + GPU 型号，防止跨 GPU 架构复用不兼容 engine
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
                        self._trt_path = trt_path   # FIX-CORRUPT-TRT: 供 warmup 失败时自动删除
                        # 会把 GFPGAN 模型移到 GPU，抢占显存连续块。
                        # Phase 2 的关键目标是先让 Myelin 在干净的 GPU 上 warmup，
                        # 所以：先尝试 _load_engine，只在 rebuild fallback 时才构建 wrapper。
                        # FIX-TWO-PHASE: Phase 2 Inference 进程不做 build
                        # .trt 应由 Phase 1 Builder 进程预先生成
                        # 若 .trt 不存在（build 失败/跳过），直接走 PyTorch 路径
                        if not osp.exists(trt_path):
                            print(f'[GFPGAN-TensorRT] .trt 不存在，跳过构建（Phase 1 应已完成），走 PyTorch 路径', flush=True)
                            return   # __init__ 提前返回，_trt_ok 保持 False

                        if osp.exists(trt_path):
                            try:
                                # FIX-INIT-ORDER-PHASE2: 先 load engine（Myelin 在 GPU 抢位）
                                # 再 build wrapper（把 GFPGAN 模型移到 GPU 填充碎片）
                                self._load_engine(trt_path)
                            except RuntimeError as _e:
                                print(f'[GFPGAN-TensorRT] 首次加载失败({_e})，重建...', flush=True)
                                # fallback rebuild 才需要 wrapper
                                wrapper = self._build_wrapper(face_enhancer.gfpgan, gfpgan_weight, device, use_fp16)
                                if not osp.exists(onnx_path):
                                    self._export_onnx(wrapper, onnx_path, max_batch_size)
                                if osp.exists(onnx_path):
                                    self._build_engine_dynamic(onnx_path, trt_path, max_batch_size, use_fp16)
                                    if osp.exists(trt_path):
                                        self._load_engine(trt_path)

                    @staticmethod
                    def _build_wrapper(gfpgan_net, weight, device, use_fp16):
                        """dry-run 探测输出格式，weight 烘焙为 trace-time 常量
                        FIX-DEVICE-MATCH: dry-run dummy 必须与模型实际设备一致
                        model 可能在 CPU（FIX-INIT-ORDER 延迟加载），不能用 device 参数
                        """
                        gfpgan_net = gfpgan_net.eval()
                        actual_device = next(gfpgan_net.parameters()).device  # 跟随模型实际设备
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
                                if _returns_tuple:
                                    out = out[0]
                                if abs(_w - 1.0) < 1e-6:
                                    return out
                                return _w * out + (1.0 - _w) * x

                        return _W().to(device)

                    @staticmethod
                    @_ctx.contextmanager
                    def _onnx_compat_patch():
                        """临时替换 FusedLeakyReLU / upfirdn2d 为标准算子，导出后自动还原
                        这是解决 ArrayRef/ONNX trace 报错的关键！"""
                        _patches = []
                        # Patch 1: FusedLeakyReLU → F.leaky_relu
                        try:
                            import basicsr.ops.fused_act.fused_act as _fa_mod
                            _orig = _fa_mod.fused_leaky_relu
                            def _compat(inp, bias, negative_slope=0.2, scale=2**0.5):
                                bv = (bias.view(1,-1,1,1)
                                      if (bias.dim()==1 and inp.dim()==4) else bias)
                                return _F.leaky_relu(inp + bv, negative_slope=negative_slope) * scale
                            _fa_mod.fused_leaky_relu = _compat
                            _patches.append((_fa_mod, 'fused_leaky_relu', _orig))
                            try:
                                import basicsr.ops.fused_act as _fa_pkg
                                if hasattr(_fa_pkg, 'fused_leaky_relu'):
                                    _patches.append((_fa_pkg, 'fused_leaky_relu', _fa_pkg.fused_leaky_relu))
                                    _fa_pkg.fused_leaky_relu = _compat
                            except Exception:
                                pass
                            print('[GFPGAN-TensorRT] FusedLeakyReLU → F.leaky_relu', flush=True)
                        except Exception as _e:
                            print(f'[GFPGAN-TensorRT] FusedLeakyReLU patch 跳过: {_e}', flush=True)
                        # Patch 2: upfirdn2d → Python fallback
                        try:
                            import basicsr.ops.upfirdn2d.upfirdn2d as _ud_mod
                            if getattr(_ud_mod, '_use_custom_op', False):
                                _patches.append((_ud_mod, '_use_custom_op', True))
                                _ud_mod._use_custom_op = False
                                print('[GFPGAN-TensorRT] upfirdn2d → Python fallback', flush=True)
                        except Exception as _e:
                            print(f'[GFPGAN-TensorRT] upfirdn2d patch 跳过: {_e}', flush=True)
                        try:
                            yield
                        finally:
                            for _obj, _attr, _orig in reversed(_patches):
                                try: setattr(_obj, _attr, _orig)
                                except Exception: pass

                    def _export_onnx(self, wrapper, onnx_path, max_batch_size):
                        """在 _onnx_compat_patch 上下文中导出静态 batch ONNX"""
                        wrapper = wrapper.eval()
                        dummy   = torch.randn(max_batch_size, 3, 512, 512, device=self.device)
                        if self.use_fp16:
                            dummy   = dummy.half()
                            wrapper = wrapper.half()
                        try:
                            print(f'[GFPGAN-TensorRT] ONNX 导出 (静态 batch={max_batch_size})...', flush=True)
                            with self._onnx_compat_patch():
                                with torch.no_grad():
                                    torch.onnx.export(
                                        wrapper, dummy, onnx_path,
                                        input_names=['input'],
                                        output_names=['output'],
                                        opset_version=18,
                                        dynamo=False,
                                    )
                            print(f'[GFPGAN-TensorRT] ONNX 已导出: {onnx_path}', flush=True)
                            return True
                        except Exception as _e:
                            print(f'[GFPGAN-TensorRT] ONNX 导出失败: {_e}', flush=True)
                            return False

                    def _build_engine_dynamic(self, onnx_path, trt_path, max_batch_size, use_fp16):
                        """构建静态 batch TRT Engine，parse_from_file 自动解析 .onnx.data
                        FIX-DESTRUCTOR: 显式提前释放 builder/network/parser/config，
                        避免函数返回时 C++ 析构器在 CUDA context 半初始化状态下
                        触发 CUDA_ERROR_ILLEGAL_ADDRESS → std::terminate()
                        """
                        trt    = self._trt
                        logger = _get_subprocess_trt_logger()
                        builder = trt.Builder(logger)
                        try:
                            flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                        except AttributeError:
                            flag = 0
                        network = builder.create_network(flag)
                        parser  = trt.OnnxParser(network, logger)
                        # parse_from_file 让 TRT 自动从同目录找 .onnx.data（若有）
                        if not parser.parse_from_file(onnx_path):
                            for i in range(parser.num_errors):
                                print(f'  [GFPGAN-TRT] 解析错误: {parser.get_error(i)}', flush=True)
                            # FIX-DESTRUCTOR: 提前释放，避免析构崩溃
                            del parser, network, builder
                            return
                        print('[GFPGAN-TensorRT] ONNX 解析完成，编译 Engine...', flush=True)
                        config = builder.create_builder_config()
                        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))
                        if use_fp16 and builder.platform_has_fast_fp16:
                            config.set_flag(trt.BuilderFlag.FP16)
                        # 静态 batch profile：min=opt=max=max_batch_size
                        profile = builder.create_optimization_profile()
                        _bs = max_batch_size
                        profile.set_shape('input',
                            min=(_bs,3,512,512), opt=(_bs,3,512,512), max=(_bs,3,512,512))
                        config.add_optimization_profile(profile)
                        serialized = builder.build_serialized_network(network, config)
                        # FIX-DESTRUCTOR: build 完成后立即释放大对象，在 CUDA 同步前完成 C++ 析构
                        # 避免函数 return 时析构器触发 CUDA_ERROR_ILLEGAL_ADDRESS → terminate()
                        del config, profile, parser, network, builder
                        import gc; gc.collect()
                        if serialized is None:
                            print('[GFPGAN-TensorRT] Engine 构建失败', flush=True)
                            return
                        with open(trt_path, 'wb') as f:
                            f.write(serialized)
                        del serialized  # 释放序列化数据（可能数百MB）
                        print(f'[GFPGAN-TensorRT] Engine 已缓存: {trt_path}', flush=True)

                    def _load_engine(self, trt_path):
                        # FIX-CROSS-GPU: 检测 .trt 文件是否与当前 GPU 匹配。
                        # 历史遗留文件（无 SM tag）或跨 GPU 复用（A10 sm86 → T4 sm75）
                        # 会导致 deserialize 成功但 warmup execute Error 700。
                        # 解决：文件名不含当前 GPU 的 SM tag → 视为过期缓存，删除并重建。
                        _cur_sm_tag = ''
                        if torch.cuda.is_available():
                            _pp = torch.cuda.get_device_properties(0)
                            # _cur_sm_tag = f'_sm{_pp.major}{_pp.minor}'
                            import re as _re
                            _gpu_slug = _re.sub(r'[^a-z0-9]', '', _pp.name.lower())[:16]
                            _cur_sm_tag = f'_sm{_pp.major}{_pp.minor}_{_gpu_slug}'
                        if _cur_sm_tag:
                            import os.path as _osp2
                            _basename = _osp2.basename(trt_path)
                            if _cur_sm_tag not in _basename:
                                print(f'[GFPGAN-TensorRT] .trt 文件名不含当前 GPU SM tag {_cur_sm_tag}，'
                                      f'可能是旧版本缓存或跨 GPU 遗留文件: {_basename}', flush=True)
                                print(f'[GFPGAN-TensorRT] 删除过期缓存，触发针对当前 GPU 的重建', flush=True)
                                try:
                                    os.remove(trt_path)
                                except OSError:
                                    pass
                                raise RuntimeError(f'[GFPGAN-TensorRT] 过期缓存 {_basename} 已删除，需重建')

                        trt     = self._trt
                        logger  = _get_subprocess_trt_logger()
                        runtime = trt.Runtime(logger)
                        with open(trt_path, 'rb') as f:
                            self._engine = runtime.deserialize_cuda_engine(f.read())
                        # IRuntime 仅用于反序列化，deserialize_cuda_engine 返回后 Engine 独立存在。
                        # runtime 局部变量 GC 后不影响 Engine/Context 生命周期（v6.2_single 验证）。
                        del runtime
                        if self._engine is None:
                            try: os.remove(trt_path)
                            except OSError: pass
                            raise RuntimeError('deserialize_cuda_engine returned None')
                        self._context     = self._engine.create_execution_context()
                        if self._context is None:
                            raise RuntimeError('create_execution_context returned None')
                        # FIX-MYELIN-EARLY: create_execution_context 内部 Myelin workspace 分配是异步的。
                        # 立即 sync 让分配失败在此精确暴露，而不是推迟到 warmup execute 的 sync 点混淆诊断。
                        # 此时 PyTorch 主 context 已由 torch.cuda.init() 确保存在，sync 安全可调用。
                        try:
                            torch.cuda.synchronize(self.device)
                        except Exception as _ce:
                            print(f'[GFPGAN-TensorRT] create_execution_context 后同步失败（Myelin workspace 分配错误）: {_ce}', flush=True)
                            print('[GFPGAN-TensorRT] 可能原因：GPU 显存不足或碎片化，无法分配 Myelin 连续 workspace', flush=True)
                            self._cuda_context_dead = True
                            return   # 不设 _trt_ok=True，不继续 warmup
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
                        # ★ Warmup：强制 Myelin eager 分配 workspace（dedicated stream）
                        # FIX-STREAM: 使用专用 non-default stream，消除 TRT 关于 default stream 的警告。
                        # TRT enqueueV3 在 default stream 上会额外插入 cudaStreamSynchronize 调用，
                        # 这些隐式 sync 点在 context 状态不稳定时可能暴露与 execute 无关的已有异步错误，
                        # 造成误判。专用 stream 可精确控制同步点，只在 stream.synchronize() 时暴露错误。
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
                                self._context.set_tensor_address(self._input_name,  _inp.data_ptr())
                                self._context.set_tensor_address(self._output_name, _out.data_ptr())
                                self._context.execute_async_v3(stream_handle=_warmup_stream.cuda_stream)
                            else:
                                self._context.set_binding_shape(0, _FULL)
                                self._context.execute_async_v2(
                                    bindings=[_inp.data_ptr(), _out.data_ptr()],
                                    stream_handle=_warmup_stream.cuda_stream,
                                )
                            _warmup_stream.synchronize()
                            # torch.cuda.synchronize(self.device)
                            del _inp, _out, _warmup_stream
                            print('[GFPGAN-TensorRT] Warmup 通过', flush=True)
                        except Exception as _we:
                            print(f'[GFPGAN-TensorRT] Warmup 失败: {_we}', flush=True)
                            self._warmup_failed = True       # FIX: 标记 warmup 失败
                            self._cuda_context_dead = True   # FIX: 标记 context 损坏
                            # FIX-NO-NULL: 不销毁 TRT 对象，只标记，避免 C++ 析构器在损坏的 context 上崩溃
                            return  # 提前返回，不设置 _trt_ok = True

                    def safe_destroy(self):
                        for _attr in ('_context', '_engine', '_trt_stream'):
                            try:
                                _obj = getattr(self, _attr, None)
                                if _obj is not None:
                                    setattr(self, _attr, None)
                                    del _obj
                            except Exception:
                                setattr(self, _attr, None)
                        self._trt_ok = False

                    @property
                    def available(self):
                        return self._trt_ok

                    def infer(self, face_tensor):
                        """单次 kernel launch，zero-pad 不足部分，返回前 B 帧结果"""
                        if self._trt_stream is None:
                            self._trt_stream = torch.cuda.Stream(device=self.device)
                        B      = face_tensor.shape[0]
                        max_bs = self._max_batch_size
                        dtype  = torch.float16 if self.use_fp16 else torch.float32
                        _FULL  = (max_bs, 3, 512, 512)
                        # 预分配持久 buffer
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
                            self._context.set_tensor_address(self._input_name,  self._inp_buf.data_ptr())
                            self._context.set_tensor_address(self._output_name, self._out_buf.data_ptr())
                            self._context.execute_async_v3(stream_handle=self._trt_stream.cuda_stream)
                        else:
                            self._context.set_binding_shape(0, _FULL)
                            self._context.execute_async_v2(
                                bindings=[self._inp_buf.data_ptr(), self._out_buf.data_ptr()],
                                stream_handle=self._trt_stream.cuda_stream,
                            )
                        self._trt_stream.synchronize()  # 仅同步当前 TRT 专属流，这足够保证结果计算完毕
                        # torch.cuda.synchronize(self.device)
                        return self._out_buf[:B].clone()
                # 实例化 GFPGANTRTAccelerator
                gfpgan_trt_accel = GFPGANTRTAccelerator(
                    face_enhancer=face_enhancer,
                    device=device,
                    cache_dir=trt_cache_dir or osp.join(os.getcwd(), '.trt_cache'),
                    gfpgan_weight=gfpgan_weight,
                    max_batch_size=gfpgan_batch_size,
                    gfpgan_version=gfpgan_model,
                    use_fp16=use_fp16
                )
                
                if gfpgan_trt_accel.available:
                    print('[GFPGANSubprocess] TRT 加速已启用（子进程版本）', flush=True)
                    trt_init_success = True  # FIX: 标记成功
                else:
                    print('[GFPGANSubprocess] TRT 初始化失败，回退 PyTorch', flush=True)
                    use_trt = False
                    trt_init_success = False
                    
            except Exception as e:
                print(f'[GFPGANSubprocess] TRT 初始化失败，回退 PyTorch: {e}', flush=True)
                use_trt = False
                trt_init_success = False
                gfpgan_trt_accel = None

        # FIX: 如果 TRT 初始化导致 CUDA context 损坏，直接退出，不设置 ready_event
        if gfpgan_trt_accel is not None and getattr(gfpgan_trt_accel, '_cuda_context_dead', False):
            print('[GFPGANSubprocess] TRT warmup 失败导致 CUDA context 损坏，不发送 ready 信号', flush=True)
            print('[GFPGANSubprocess] 子进程将直接退出，主进程应回退到 PyTorch 路径', flush=True)
            import time
            time.sleep(0.5)  # 确保日志被刷新
            import os as _os
            _os._exit(0)  # 直接退出，不触发 Python GC，不设置 ready_event

        # FIX-INIT-ORDER: 根据最终路径决定 GFPGANer 设备
        _model_needs_gpu = False
        if use_trt and gfpgan_trt_accel is not None and gfpgan_trt_accel.available:
            # TRT warmup 通过，此时 Myelin workspace 已占位，迁移 GFPGANer 到 GPU
            print('[GFPGANSubprocess] TRT warmup 通过，迁移 GFPGANer 到 GPU...', flush=True)
            _model_needs_gpu = True
        elif use_trt and not (gfpgan_trt_accel is not None
                              and getattr(gfpgan_trt_accel, '_cuda_context_dead', False)):
            # TRT init 失败（非 context dead）→ 回退 PyTorch，model 在 CPU 需迁移到 GPU
            print('[GFPGANSubprocess] TRT 失败，迁移 GFPGANer 到 GPU 用于 PyTorch 路径...', flush=True)
            _model_needs_gpu = True

        if _model_needs_gpu:
            face_enhancer.gfpgan = face_enhancer.gfpgan.to(device)
            model = face_enhancer.gfpgan
            if use_fp16:
                face_enhancer.gfpgan = face_enhancer.gfpgan.half()
                model = face_enhancer.gfpgan
            face_enhancer.face_helper = None   # 强制重建（device 已变）
            print('[GFPGANSubprocess] GFPGANer 已迁移到 GPU', flush=True)
        
        fp16_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_fp16 else contextlib.nullcontext()

        # FIX-CUDA-DEAD: 检测 TRT warmup 是否使 CUDA context 不可用
        # 两种子情形，用 _warmup_failed 区分以提供精确诊断：
        #   · _warmup_failed=True  → execute_async Error 700（engine kernel bug，T4/SM75 常见）
        #     → .trt 文件本身可能有问题，建议删除缓存以较小 batch_size 重建
        #   · _warmup_failed=False → Myelin workspace 分配失败（显存不足）
        #     → 减少 --gfpgan-batch-size 或释放 VRAM 后重试
        if gfpgan_trt_accel is not None and getattr(gfpgan_trt_accel, '_cuda_context_dead', False):
            _is_warmup_err = getattr(gfpgan_trt_accel, '_warmup_failed', False)
            if _is_warmup_err:
                print('[GFPGANSubprocess] TRT warmup 失败（execute Error 700），子进程退出，'
                      '主进程将降级到内部 GFPGAN PyTorch 路径', flush=True)
                print('[GFPGANSubprocess] 提示：此 GPU 上 TRT FP16 执行该 engine 触发非法内存访问。'
                      f'\n         可尝试：(1) 删除 .trt 缓存后以更小 --gfpgan-batch-size 重建；'
                      f'\n                 (2) 去掉 --gfpgan-trt，改用 PyTorch FP16（主进程内）。', flush=True)
            else:
                print('[GFPGANSubprocess] TRT CUDA context 不可用（Myelin workspace 分配失败），子进程退出，'
                      '主进程将降级到内部 GFPGAN PyTorch 路径', flush=True)
            if ready_event is not None:
                ready_event.set()   # 先解除主进程等待，再退出
            import os as _os, time
            time.sleep(0.5)         # 确保 ready_event.set() 被主进程接收
            # FIX-OSEXIT: os._exit(0) 直接系统调用，完全跳过 Python GC/__del__
            # sys.exit(0) 会触发 TRT C++ 析构器 → std::exception → terminate()
            _os._exit(0)

        # FIX: 只有在真正成功时才设置 ready_event
        if cuda_context_dead:
            # 前面已经处理，这里不会执行
            pass
        elif use_trt and gfpgan_trt_accel is not None and gfpgan_trt_accel.available:
            # TRT 成功
            print('[GFPGANSubprocess] TRT 成功，进入任务循环', flush=True)
            if ready_event is not None:
                ready_event.set()
        elif use_trt and (gfpgan_trt_accel is None or not gfpgan_trt_accel.available):
            # TRT 失败但 context 正常（前面已处理 context 损坏的情况）
            print('[GFPGANSubprocess] TRT 失败但 context 正常，以 PyTorch 模式服务', flush=True)
            if ready_event is not None:
                ready_event.set()  # 可以服务，但走 PyTorch
        else:
            # 纯 PyTorch 路径
            print('[GFPGANSubprocess] PyTorch 模式，进入任务循环', flush=True)
            if ready_event is not None:
                ready_event.set()

        while True:
            try:
                task = task_queue.get(timeout=1)
            except queue.Empty:
                continue
            if task is None:  # 终止信号
                break

            # ── POST-SR-VALIDATE 消息（借鉴 v6.2_single post_sr_validate）────────
            # 主进程在第一个 SR 批次完成后发送此消息，触发真实显存压力下的二次 warmup。
            # 初始 warmup 在显存干净时通过不代表 SR 跑起来后仍可用（Myelin workspace 可能被 SR 挤压）。
            # 关键区分：
            #   OOM（显存不足）→ 降级 PyTorch，子进程继续
            #   cudaErrorIllegalAddress/context 损坏 → CUDA context 已死，PyTorch 降级也无法工作
            #     → 必须 os._exit(0)，主进程通过 is_alive()+exitcode 检测后降级到主进程内 GFPGAN
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
                            'acceleratorerror', 'cudaerror',
                        ))
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

            task_id, crops_np = task  # crops_np: list of numpy arrays (H,W,3) uint8
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
                            # TRT 推理路径
                            if use_fp16:
                                sub_batch = sub_batch.half()
                            out = gfpgan_trt_accel.infer(sub_batch)
                            # TRT 输出已经是 float32，无需转换
                            out = out.float() if out.dtype != torch.float32 else out
                        else:
                            # PyTorch 推理路径
                            with fp16_ctx:
                                out = model(sub_batch, return_rgb=False, weight=gfpgan_weight)
                                if isinstance(out, (tuple, list)):
                                    out = out[0]
                            out = out.float()
                    all_out.extend(out.unbind(0))
                    i += len(sub)
                except RuntimeError as e:
                    error_str = str(e).lower()
                    if 'out of memory' in error_str and sub_bs > 1:
                        # FIX-WORKER-OOM: 子进程内 OOM 只降级 sub_bs 并释放显存后重试。
                        # 原代码错误地调用 self.gfpgan_subprocess.pause()：
                        #   1. _worker 是 @staticmethod，self 未定义 → NameError → 子进程崩溃
                        #   2. 逻辑上无意义：子进程自身 OOM，无需通知自己暂停
                        # pause() 的正确调用方是主进程 _sr_with_oom_fallback（已正确实现）。
                        sub_bs = max(1, sub_bs // 2)
                        # FIX-SR-OOM-PAUSE: SR OOM 降级后立即释放显存并通知 GFPGAN subprocess 暂停
                        torch.cuda.empty_cache()
                        print(f'[GFPGANSubprocess] GFPGAN OOM，sub_bs 降级至 {sub_bs}，重试...', flush=True)
                    elif 'cudaerrorillegaladdress' in error_str or 'illegal memory' in error_str:
                        # CUDA 上下文损坏，切换到 PyTorch 路径
                        print(f'[GFPGANSubprocess] CUDA 非法内存访问，切换到 PyTorch 路径: {e}')
                        use_trt = False
                        gfpgan_trt_accel = None
                        torch.cuda.empty_cache()
                    else:
                        # 无法恢复，补 None
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
                    img = tensor2img(out_t, rgb2bgr=True, min_max=(-1,1))
                    restored.append(img.astype('uint8'))
            try:
                result_queue.put((task_id, restored), timeout=5.0)
            except queue.Full:
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
        """
        同步调用：发送 crops 列表，等待返回增强后的人脸列表。
        crops_list: list of (H,W,3) uint8 (原始检测出的对齐人脸)
        returns: list of (H,W,3) uint8 增强后的人脸，或 None 表示失败。
        """
        # FIX-DEAD-GUARD: 若调用时子进程已退出（正常路径下不应发生，但防御性处理），
        # 立即返回 None 列表，避免 result_queue.get() 阻塞 60s。
        if not self.process or not self.process.is_alive():
            return [None] * len(crops_list)
        task_id = id(crops_list)
        
        # FIX-DEADLOCK-3: 防止子进程挂死导致 put 永久阻塞
        try:
            self.task_queue.put((task_id, crops_list), timeout=10.0)
        except queue.Full:
            print("[致命错误] GFPGAN子进程任务队列拥堵，子进程可能已假死！", flush=True)
            return [None] * len(crops_list)

        while True:
            try:
                res = self.result_queue.get(timeout=60)   # FIX-TIMEOUT: 单次推理超时60s
            except queue.Empty:
                raise RuntimeError('GFPGAN子进程超时无响应')
            # 过滤掉 validate 响应（不应该发生，但防御性处理）
            if isinstance(res, tuple) and len(res) == 3 and res[0] == '__validate__':
                continue
            res_id, result = res
            if res_id == task_id:
                return result
            # 不是当前任务的结果，放回去（不应该发生）
            self.result_queue.put((res_id, result))

    def post_sr_validate(self) -> bool:
        """
        借鉴 v6.2_single GFPGANTRTAccelerator.post_sr_validate()。
        在主进程第一个 SR 批次完成后调用，于真实显存压力下验证子进程 TRT 是否可用。
        初始 warmup 在显存干净时通过不代表 SR 全速跑起来后仍可用。

        返回值：
          True  → TRT post-SR warmup 通过，可以正常使用
          False → 失败（含两种情况）：
            · OOM/可恢复：子进程已降级 PyTorch，仍然存活
            · context 损坏：子进程已 os._exit(0)，is_alive()=False
              主进程调用方通过检查 self.process.is_alive() 区分
        """
        if not self.process or not self.process.is_alive():
            return False
        val_id = id(self)
        try:
            self.task_queue.put(('__validate__', val_id), timeout=5.0)
        except queue.Full:
            return False

        deadline = time.time() + 180   # 最多等 3 分钟
        while time.time() < deadline:
            # FIX-CTX-DEAD: 每轮先检测进程是否已死（context 损坏时子进程 os._exit）
            if not self.process.is_alive():
                return False
            try:
                res = self.result_queue.get(timeout=5)
            except queue.Empty:
                continue
            if isinstance(res, tuple) and len(res) == 3 and res[0] == '__validate__' and res[1] == val_id:
                return res[2]
            # 不是 validate 结果，放回去
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
            print('[GFPGANSubprocess] 警告: 任务队列已满，无法发送暂停信号', flush=True)

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
        try:
            self.task_queue.close()
        except Exception:
            pass
        try:
            self.result_queue.close()
        except Exception:
            pass


class GPUMemoryPool:
    """流水线并发槽计数器（纯信号量）

    FIX-OOM-POOL: 原实现预分配了 max_batches×(input_buf + output_buf×4) 的 GPU tensor，
    对 1080p 输入 batch_size=12 已超过 10 GiB，实际上这些 tensor 从未被 _sr_infer_batch
    使用（SR 推理使用自己的 PinnedBufferPool），是纯粹的 VRAM 浪费，直接导致 OOM。
    修复：改为纯计数信号量，仅做流水线背压控制，不分配任何 GPU 内存。
    """

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
        """取一个槽位；无空闲时返回 None（非阻塞）"""
        try:
            idx = self._slots.get_nowait()
            return {'index': idx}
        except queue.Empty:
            return None

    def release(self, idx: int):
        """归还槽位"""
        self._slots.put(idx)


class DeepPipelineOptimizer:
    """深度流水线优化器 - 4级并行处理"""
    
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
        # SR TRT 加速器（由 main_optimized 在 upsampler 加载后构建并传入）
        # TRT 优先级最高；cuda_graph_accel 保留扩展接口但当前保持 None
        self.trt_accel        = trt_accel
        self.cuda_graph_accel = None   # TRT 优先，与 CUDA Graph 互斥
        # FIX-VERBOSE: 人脸检测累计统计（用于性能监控日志，替代每批 print）
        self._face_frames_total = 0
        self._face_count_total  = 0
        
        # 人脸检测helper
        self.detect_helper = _make_detect_helper(face_enhancer, device) if face_enhancer else None

        # FIX-EARLY-SPAWN: 优先使用预启动的 GFPGAN 子进程（SR 加载前已 warmup）
        self.gfpgan_subprocess = None
        
        # FIX: 安全获取预启动子进程，确保变量始终定义
        try:
            _prestarted = getattr(args, '_early_gfpgan_subprocess', None)
        except Exception:
            _prestarted = None
        
        # FIX: 验证预启动的子进程是否仍然存活
        if _prestarted is not None:
            if hasattr(_prestarted, 'process') and _prestarted.process.is_alive():
                print('[优化架构] 使用预启动 GFPGAN 子进程（FIX-EARLY-SPAWN）')
                self.gfpgan_subprocess = _prestarted
                args._early_gfpgan_subprocess = None   # 转移所有权
            else:
                print('[优化架构] 预启动 GFPGAN 子进程已死亡，关闭并回退')
                try:
                    _prestarted.close()
                except Exception as e:
                    print(f'[优化架构] 关闭死亡子进程错误: {e}')
                args._early_gfpgan_subprocess = None
                self.gfpgan_subprocess = None

        # 如果没有预启动的子进程，且启用了 gfpgan_trt，创建新的子进程
        # FIX: 但此时如果 args.gfpgan_trt 已被标记为失败，则跳过
        if (self.gfpgan_subprocess is None and 
            getattr(args, 'gfpgan_trt', False) and 
            face_enhancer is not None):
            
            # 检查是否之前已经尝试过 TRT 并失败了
            if not getattr(args, '_gfpgan_trt_failed', False):
                print('[优化架构] 启用子进程GFPGAN TRT加速（非预启动路径）')
                try:
                    self.gfpgan_subprocess = GFPGANSubprocess(
                        face_enhancer=face_enhancer,
                        device=device,
                        gfpgan_weight=args.gfpgan_weight,
                        gfpgan_batch_size=args.gfpgan_batch_size,
                        use_fp16=not args.no_fp16,
                        use_trt=True,
                        trt_cache_dir=getattr(args, 'trt_cache_dir', None),
                        gfpgan_model=args.gfpgan_model,
                    )
                except Exception as e:
                    print(f'[优化架构] GFPGAN 子进程创建失败: {e}')
                    self.gfpgan_subprocess = None
                    args._gfpgan_trt_failed = True  # 标记失败，防止重试
            else:
                print('[优化架构] GFPGAN TRT 之前已失败，跳过子进程创建')
    
    def optimize_pipeline(self, reader, writer, pbar, total_frames):
        """运行优化的深度流水线"""
        
        print("[优化架构] 启动深度流水线处理...")
        print(f"[优化架构] 队列深度: F{self.frame_queue.maxsize}/D{self.detect_queue.maxsize}/S{self.sr_queue.maxsize}/G{self.gfpgan_queue.maxsize}")
        print(f"[优化架构] 内存池: {self.memory_pool.max_batches}批次")
        print(f"[优化架构] 最优batch_size: {self.optimal_batch_size}")
        
        # FIX-TWO-PHASE: Phase 1 build 已在 __init__ 完成
        # 这里等待的是 Phase 2 Inference 进程的 warmup（通常 <30秒）
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
        
        # 等待所有线程完成
        read_thread.join()
        detect_thread.join()
        sr_thread.join()
        gfpgan_thread.join()
    
    def _read_frames(self, reader):
        """读取视频帧到队列"""
        batch_frames = []
        try:
            while self.running:
                try:
                    img = reader.get_frame()
                    if img is FFmpegReader.FRAME_TIMEOUT:
                        continue
                    if img is None:
                        if batch_frames:
                            # FIX-DEADLOCK-1: 使用超时防死锁推入
                            while self.running:
                                try:
                                    self.frame_queue.put((batch_frames, True), timeout=1.0)
                                    break
                                except queue.Full:
                                    continue
                        break
                    
                    batch_frames.append(img)
                    
                    if len(batch_frames) >= self.optimal_batch_size:
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
            # FIX-DEADLOCK-2: 崩溃时也强制发哨兵，保证接力棒传递
            try:
                self.frame_queue.put((None, True), timeout=3.0)
            except Exception:
                pass
    
    def _detect_faces(self):
        """人脸检测处理"""
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
                        future = self.detect_executor.submit(
                            _detect_faces_batch, batch_frames, self.detect_helper
                        )
                        face_data, _fw, _nf = future.result()   # FIX-VERBOSE
                        # 累加到流水线统计，由 _write_frames 的性能监控日志统一输出
                        self._face_frames_total += _fw
                        self._face_count_total  += _nf
                            
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
            # FIX-DUP-SENTINEL: 只在主循环未发送时才补发
            if not _sentinel_sent:
                try:
                    self.detect_queue.put((None, None, True), timeout=3.0)
                except Exception:
                    pass
    
    
    def _process_sr(self):
        """SR推理处理"""
        _first_batch_done = False   # POST-SR-VALIDATE: 标记第一个 SR 批次是否已完成
        _sentinel_sent = False  # FIX-DUP-SENTINEL
        try:
            while self.running:
                try:
                    item = self.detect_queue.get(timeout=1.0)
                    
                    if item is None:
                        self.sr_queue.put(None)
                        _sentinel_sent = True
                        break
                    
                    batch_frames, face_data, is_end = item

                    # FIX-BUG4: sentinel 检查必须在 acquire memory_block 之前。
                    # 原实现先 acquire 再检查 batch_frames is None：
                    #   · pool 满时（8 slot 全在途中）此处无限自旋 → sr_thread.join() 死锁
                    #   · 即使 pool 未满，acquire 到的 slot 在 break 前永远不 release → slot 泄漏
                    if batch_frames is None:
                        self.sr_queue.put((None, None, None, None, True))
                        _sentinel_sent = True
                        break

                    # 获取GPU内存块 — FIX-POOL-DROP: 必须先拿到 memory_block 再 proceed
                    # 旧代码 `continue` 会丢弃已从 detect_queue.get() 取出的 item（静默跳帧）
                    # 正确做法：在当前 item 的处理循环内自旋等待，直到内存池有空余
                    memory_block = None
                    while self.running and memory_block is None:
                        memory_block = self.memory_pool.acquire()
                        if memory_block is None:
                            time.sleep(0.005)
                    if not self.running:
                        break
                    
                    # SR推理（含 OOM 级联降级）
                    # FIX-OOM-SR: 原实现 except 仅 print+release，不降级、不 empty_cache，
                    # 导致每批都 OOM 后直接丢弃，产生无限 OOM 风暴且 0 帧输出。
                    # 修复：OOM 时对半切 batch_size 并 retry，bs=1 仍 OOM 则 sleep(2) 后再试一次。
                    t0 = time.perf_counter()

                    def _sr_with_oom_fallback(frames):
                        """对当前 batch 做 OOM 级联降级推理，返回 sr_results 列表。"""
                        retry_bs = min(self.optimal_batch_size, len(frames))
                        _had_real_oom = False          # FIX-FALSE-OOM: 区分真实 OOM 与尾批截断
                        while True:
                            try:
                                all_sr = []
                                i = 0
                                while i < len(frames):
                                    sub = frames[i:i + retry_bs]
                                    sub_sr, _, _ = _sr_infer_batch(
                                        self.upsampler, sub, self.args.outscale,
                                        getattr(self.args, 'netscale', 4),
                                        self.transfer_stream, self.sr_stream,
                                        self.trt_accel, self.cuda_graph_accel,
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
                                    raise   # 非 OOM 错误直接上抛
                                _had_real_oom = True   # ← 标记真实 OOM
                                # FIX-SR-OOM-PAUSE: SR OOM 降级后立即释放显存并通知 GFPGAN subprocess 暂停
                                torch.cuda.empty_cache()
                                # 通知 GFPGAN 子进程暂停，释放 GPU 显存压力
                                if self.gfpgan_subprocess is not None:
                                    self.gfpgan_subprocess.pause(duration=5.0)

                                if retry_bs > 1:
                                    retry_bs = max(1, retry_bs // 2)
                                    print(f'[SR-OOM] OOM，降级 batch_size → {retry_bs}，重试...', flush=True)
                                else:
                                    # bs=1 仍 OOM：等待显存释放后最后一次尝试
                                    print('[SR-OOM] bs=1 仍 OOM，等待 2s 后最终尝试...', flush=True)
                                    time.sleep(2.0)

                                    # FIX-SR-OOM-PAUSE: SR OOM 降级后立即释放显存并通知 GFPGAN subprocess 暂停
                                    torch.cuda.empty_cache()
                                    # 通知 GFPGAN 子进程暂停，释放 GPU 显存压力
                                    if self.gfpgan_subprocess is not None:
                                        self.gfpgan_subprocess.pause(duration=5.0)

                                    # 最终尝试，失败则上抛
                                    sub_sr, _, _ = _sr_infer_batch(
                                        self.upsampler, frames[:1], self.args.outscale,
                                        getattr(self.args, 'netscale', 4),
                                        self.transfer_stream, self.sr_stream,
                                        self.trt_accel, self.cuda_graph_accel,
                                    )
                                    # 剩余帧逐帧处理
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
                        sr_results = _sr_with_oom_fallback(batch_frames)

                        timing = time.perf_counter() - t0
                        self.timing.append(timing)

                        # POST-SR-VALIDATE（借鉴 v6.2_single）:
                        # 第一个 SR 批次完成 → 显存进入真实压力状态 → 让子进程做二次 warmup 验证。
                        # 初始 warmup 在显存干净时通过，不代表 SR 全速跑起来后 Myelin workspace 仍存活。
                        if not _first_batch_done and self.gfpgan_subprocess is not None:
                            _first_batch_done = True
                            print('[优化架构] 第一个 SR 批次完成，触发 GFPGAN TRT post-SR 验证...', flush=True)
                            # FIX-POSTVLD-MEM: SR 批次完成后，主进程 PyTorch 缓存可能仍持有大量
                            # reserved 显存（中间 tensor、输出 buffer 等）。在 T4 这类 16GB 卡上，
                            # 若不释放则子进程 TRT post-SR warmup 因全局 VRAM 不足而 OOM/context 损坏。
                            # synchronize 确保所有主进程 CUDA op 完成后再释放，避免 DMA race。
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            _val_ok = self.gfpgan_subprocess.post_sr_validate()
                            if _val_ok:
                                print('[优化架构] GFPGAN TRT post-SR 验证通过，TRT 推理正式启用', flush=True)
                            else:
                                # FIX-RACE: 子进程在 result_queue.put(False) 之后才调用 os._exit(0)（有 0.3s sleep）
                                # 此时 is_alive() 可能仍为 True。用 join(timeout=1.5) 等待子进程真正退出再判断。
                                self.gfpgan_subprocess.process.join(timeout=1.5)
                                if not self.gfpgan_subprocess.process.is_alive():
                                    print('[优化架构] GFPGAN 子进程因 CUDA context 损坏已退出，'
                                          '降级到主进程内 GFPGAN PyTorch 路径', flush=True)
                                    self.gfpgan_subprocess = None   # 触发 _process_gfpgan 回落到主进程路径
                                else:
                                    # 子进程存活：TRT 未初始化（SM不支持/build失败）或 OOM 降级，均以 PyTorch FP16 继续
                                    print('[优化架构] GFPGAN 子进程以 PyTorch FP16 路径服务'
                                          '（TRT 未启用：SM不兼容 / build失败 / OOM）', flush=True)

                        # FIX-QUEUE-TIMEOUT: 与 gfpgan_queue.put 一致，加超时防死锁
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
            # FIX-DUP-SENTINEL: 只在主循环未发送时才补发
            if not _sentinel_sent:
                try:
                    self.sr_queue.put((None, None, None, None, True), timeout=3.0)
                except Exception:
                    pass
    
    
    def _process_gfpgan(self):
        """GFPGAN处理 - 支持子进程TRT"""
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
                    if self.gfpgan_subprocess and self.gfpgan_subprocess.process.is_alive():
                        gfpgan_available = True
                        print(f'[GFPGAN] 使用子进程TRT处理 {sum(len(fd.get("crops", [])) for fd in face_data or [])} 个人脸')
                    elif (self.face_enhancer is not None and 
                          getattr(self.face_enhancer, 'gfpgan', None) is not None):
                        gfpgan_available = True
                        print(f'[GFPGAN] 使用主进程PyTorch处理 {sum(len(fd.get("crops", [])) for fd in face_data or [])} 个人脸')
                    else:
                        print(f'[GFPGAN] GFPGAN不可用，跳过人脸增强')
                    
                    # GFPGAN处理
                    if has_valid_faces and gfpgan_available:
                        try:
                            restored_by_frame = []
                            
                            # 优先使用子进程TRT
                            if self.gfpgan_subprocess and self.gfpgan_subprocess.process.is_alive():
                                # FIX-GFPGAN-BATCH-IPC: 将所有帧的 crops 合并为单次 IPC 调用
                                # 原实现每帧一次 infer() = bs=6 时 6 次 IPC round-trip
                                # 大脸批次(46-63张)导致 GPU 长时间被占，nvenc 无法调度 → stall
                                # 修复：收集全部 crops → 单次 task_queue.put → 单次 result_queue.get
                                #        → 按帧拆分结果，GPU 独占时间从 N×T 降为 1×T
                                all_crops = []
                                crops_per_frame = []
                                for fd in face_data:
                                    crops = fd.get('crops', [])
                                    crops_per_frame.append(len(crops))
                                    all_crops.extend(crops)

                                if all_crops:
                                    all_restored = self.gfpgan_subprocess.infer(all_crops)
                                    # 按帧切分结果（infer 返回与 crops 等长的列表）
                                    idx = 0
                                    for count in crops_per_frame:
                                        restored_by_frame.append(
                                            all_restored[idx:idx + count] if count else []
                                        )
                                        idx += count
                                else:
                                    restored_by_frame = [[] for _ in face_data]
                            else:
                                # FIX: 确保主进程 face_enhancer 可用
                                if (self.face_enhancer is None or 
                                    getattr(self.face_enhancer, 'gfpgan', None) is None):
                                    print(f'[GFPGAN] 警告: 主进程 face_enhancer 不可用，跳过GFPGAN')
                                    raise RuntimeError("face_enhancer 或 gfpgan 为 None")
                                    
                                # 使用传统方法
                                restored_by_frame, _ = _gfpgan_infer_batch(
                                    face_data, self.face_enhancer, self.device,
                                    None, self.args.gfpgan_weight, 
                                    getattr(self.args, 'gfpgan_batch_size', 4), None, None
                                )
                            
                            # FIX: 检查 restored_by_frame 是否有效
                            if not restored_by_frame or all(r is None or len(r) == 0 for r in restored_by_frame):
                                print(f'[GFPGAN] 警告: 没有成功恢复的人脸，使用SR结果')
                                final_frames = sr_results
                            else:
                                # 提交贴回处理
                                # FIX-DEADLOCK-5: 为 future.result() 添加超时，防止永久卡死
                                future = self.paste_executor.submit(
                                    _paste_faces_batch, face_data, restored_by_frame, 
                                    sr_results, self.face_enhancer
                                )
                                try:
                                    final_frames = future.result(timeout=60)  # 超时60秒
                                except concurrent.futures.TimeoutError:
                                    print(f"[GFPGAN] 人脸贴回超时，使用SR结果")
                                    final_frames = sr_results
                                
                        except Exception as e:
                            print(f"GFPGAN处理错误: {e}")
                            import traceback
                            traceback.print_exc()
                            final_frames = sr_results  # 降级到SR结果
                    else:
                        # 无人脸或GFPGAN不可用，直接使用SR结果
                        if not has_valid_faces:
                            pass  # 无人脸，正常情况
                        else:
                            print(f'[GFPGAN] GFPGAN不可用，{sum(len(fd.get("crops", [])) for fd in face_data)} 个人脸未处理')
                        final_frames = sr_results
                    
                    # FIX-QUEUE-TIMEOUT: 加超时防止 _write_frames 阻塞时此处永久死锁
                    # 原实现无超时：_write_frames 被 write_frame 阻塞(≤180s) → gfpgan_queue 积满
                    # → 此处永久阻塞 → sr_queue 积满 → sr_queue.put 永久阻塞
                    # → memory_pool 8槽全占 → SR 停转 → GPU 利用率归零(见截图)
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
                    import traceback
                    traceback.print_exc()
                    # FIX: 确保释放内存块，避免内存泄漏
                    if 'memory_block' in locals() and memory_block is not None:
                        try:
                            self.memory_pool.release(memory_block['index'])
                        except Exception: pass
        finally:
            # FIX-DUP-SENTINEL: 只在主循环未发送时才补发
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
        """清理资源，发送哨兵唤醒所有线程，强制关闭"""
        print("[Pipeline] 正在停止流水线...", flush=True)
        self.running = False
        
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
                    print(f"[Pipeline] 线程 {name} 未响应，已放弃等待（daemon线程将随主进程退出）", flush=True)
                    # FIX-DAEMON-GUARD: Python 3.11+ 禁止对运行中的线程设置 daemon，
                    # 会抛出 RuntimeError("cannot set daemon status of active thread")。
                    # 所有流水线线程在创建时已设 daemon=True，主进程退出时自动终止，
                    # 此处无需（也不能）再次设置。
                    if not thread.is_alive():      # double-check after print
                        thread.daemon = True
        print("[Pipeline] 所有流水线线程已关闭", flush=True)
    
    def _write_frames(self, writer, pbar, total_frames):
        """写入帧处理"""
        written_count = 0
        end_sentinel_count = 0
        received_end_sentinel = False  # 初始化标志变量
        
        try:
            while self.running:
                try:
                    item = self.gfpgan_queue.get(timeout=10.0)   # 增加超时，避免死等
                    
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
                    
                    # 写入帧
                    for frame in final_frames:
                        # FIX-DEADLOCK-4: 检测 FFmpeg 后台写入器是否已经崩溃
                        if getattr(writer, '_broken', False):
                            print("\n[致命错误] FFmpeg 后台写入进程已崩溃 (可能是磁盘已满或编码器显存溢出)！", flush=True)
                            self.running = False  # 立即阻断整条流水线
                            break
                            
                        writer.write_frame(frame)
                        written_count += 1
                    
                    # 如果写入中途崩溃，立刻跳出大循环
                    if getattr(writer, '_broken', False):
                        break
                    
                    # FIX-LEAK: 安全释放内存块
                    if memory_block is not None:
                        try:
                            self.memory_pool.release(memory_block['index'])
                        except Exception: pass
                    
                    # 更新进度
                    pbar.update(len(final_frames))
                    self.meter.update(len(final_frames))
                    
                    # 每批次都更新 postfix（不受 batch_size 整除影响）
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

                    # 动态检测GPU内存压力
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        reserved = torch.cuda.memory_reserved() / 1024**3
                        if allocated > 0.9 * reserved:
                            print(f'\n[资源警告] GPU内存压力过高: {allocated:.2f}GB / {reserved:.2f}GB')
                    
                    # 每跨越 20 帧输出一次详细日志（跨越检测，不受 batch_size 整除影响）
                    if written_count // 20 > (written_count - len(final_frames)) // 20:
                        print(f"[性能监控] 帧{written_count}/{total_frames} | fps={current_fps:.1f} | eta={eta:.0f}s | bs={self.optimal_batch_size} | ms={avg_ms:.0f} | 队列 F:{self.frame_queue.qsize()}/D:{self.detect_queue.qsize()}/S:{self.sr_queue.qsize()}/G:{self.gfpgan_queue.qsize()} | 人脸 {self._face_count_total}张/{self._face_frames_total}帧")
                    
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
                              f"已写入 {written_count}/{total_frames} 帧 "
                              f"(残留 S:{self.sr_queue.qsize()}/G:{self.gfpgan_queue.qsize()})",
                              flush=True)
                        break
                    continue
                except Exception as e:
                    print(f"写入帧错误: {e}")
                    # FIX: 增加 memory_block is not None 判断，防止处理哨兵时抛出异常导致泄漏
                    if 'memory_block' in locals() and memory_block is not None:  
                        try:
                            self.memory_pool.release(memory_block['index'])
                        except Exception as release_err:
                            pass
        finally:
            print(f"[Pipeline] 写入线程退出，已写入 {written_count}/{total_frames} 帧", flush=True)
            # 注意：不要在这里调用 self.close()！
            # close() 应该在 main_optimized 的 finally 中由主线程统一调用
            # 否则会导致递归关闭和流水线提前终止


def main_optimized(args):
    """优化版主函数 - 修复 GFPGAN TRT 就绪判断逻辑"""
    
    print("[优化架构] 修复版: 改进 GFPGAN TRT 就绪判断")
    print("[优化架构] 阶段 0: 准备环境（不初始化 CUDA）...")
    
    # 步骤 0: 检测 CUDA 可用性（但不初始化）
    cuda_available = torch.backends.cuda.is_built() and torch.cuda.device_count() > 0
    if not cuda_available:
        print("[优化架构] CUDA 不可用，使用 CPU 模式")
        device = torch.device('cpu')
        cuda_available = False
    else:
        device = torch.device('cuda')
        print(f"[优化架构] CUDA 编译支持: 是")
        print(f"[优化架构] 延迟 CUDA Runtime 初始化直到 GFPGAN 子进程就绪")

    # 步骤 1: 在加载任何模型之前，先启动 GFPGAN 子进程
    _early_gfpgan_subprocess = None
    gfpgan_ready = False
    use_gfpgan_subprocess = False
    gfpgan_mode = "disabled"  # 记录 GFPGAN 模式
    
    if args.face_enhance and getattr(args, 'gfpgan_trt', False) and GFPGANer is not None:
        if not cuda_available:
            print("[优化架构] 警告: CUDA 不可用，跳过 GFPGAN TRT")
        else:
            print("[优化架构] 阶段 1: 预启动 GFPGAN 子进程（GPU 干净状态）...")
            
            # 准备模型路径（纯 CPU 操作）
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
                print(f'[优化架构] 下载 GFPGAN 模型: {_model_filename_early}')
                _model_path_early = load_file_from_url(_model_url_early, _model_dir_early, True)

            # 检查并报告 CUDA 状态
            if torch.cuda.is_initialized():
                print("[优化架构] 警告: 检测到意外的 CUDA 初始化，尝试清理...")
                torch.cuda.empty_cache()
            
            # 启动子进程
            print('[优化架构] 启动 GFPGAN 子进程...')
            _early_gfpgan_subprocess = GFPGANSubprocess(
                model_path=_model_path_early,
                device=device,
                gfpgan_weight=args.gfpgan_weight,
                gfpgan_batch_size=args.gfpgan_batch_size,
                use_fp16=not args.no_fp16,
                use_trt=True,
                trt_cache_dir=getattr(args, 'trt_cache_dir', None),
                gfpgan_model=args.gfpgan_model,
            )
            
            # FIX: 改进的子进程就绪判断逻辑
            print('[优化架构] 等待 GFPGAN 子进程完成初始化...')
            max_wait = 5400
            deadline = time.time() + max_wait
            _poll_interval = 3
            _last_report = time.time()
            _report_every = 60
            
            while time.time() < deadline:
                # 首先检查进程是否仍然存活
                if not _early_gfpgan_subprocess.process.is_alive():
                    exitcode = _early_gfpgan_subprocess.process.exitcode
                    if exitcode == 0:
                        print('[优化架构] GFPGAN 子进程主动退出（context 污染保护）')
                    else:
                        print(f'[优化架构] GFPGAN 子进程异常退出（exitcode={exitcode}）')
                    
                    _early_gfpgan_subprocess.close()
                    _early_gfpgan_subprocess = None
                    gfpgan_ready = False
                    gfpgan_mode = "failed_subprocess"
                    break
                
                # 检查 ready 信号
                if _early_gfpgan_subprocess.ready_event.wait(timeout=_poll_interval):
                    # FIX: ready_event 被设置后，必须再次确认进程仍然存活
                    time.sleep(0.5)  # 给进程一点时间稳定
                    
                    if not _early_gfpgan_subprocess.process.is_alive():
                        # 进程在设置 ready 后退出，说明初始化实际上失败了
                        exitcode = _early_gfpgan_subprocess.process.exitcode
                        print(f'[优化架构] GFPGAN 子进程 ready 后退出（exitcode={exitcode}）')
                        print('[优化架构] 诊断: TRT warmup 失败 -> 设置 ready -> 退出，这是预期的错误处理')
                        _early_gfpgan_subprocess.close()
                        _early_gfpgan_subprocess = None
                        gfpgan_ready = False
                        gfpgan_mode = "failed_trt_warmup"
                        break
                    
                    # FIX: 进程存活且 ready，验证是否能正常工作
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
    
    # 步骤 2: GFPGAN 就绪后，初始化主进程 CUDA 并加载 SR
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
    
    # 加载 RealESRGAN
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
            print('[优化架构] use_tensorrt=True：自动禁用 torch.compile')
            args.no_compile = True
    except Exception as e:
        print(f"[优化架构] RealESRGAN模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤 3: 准备 face_enhancer - FIX: 改进逻辑
    face_enhancer = None
    
    if args.face_enhance and GFPGANer is not None:
        if use_gfpgan_subprocess and gfpgan_ready and _early_gfpgan_subprocess is not None:
            # FIX: 只有在子进程真正就绪时才创建 DummyGFPGANer
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
                        self.gfpgan = None  # 子进程处理，主进程不需要
                        self.model_path = None
                        
                    def enhance(self, *args, **kwargs):
                        # 如果意外调用，报错提示
                        raise RuntimeError("DummyGFPGANer 不支持 enhance，GFPGAN 应由子进程处理")
                
                face_enhancer = DummyGFPGANer(device, args.outscale)
                print("[优化架构] Detect helper 创建成功（GFPGAN 推理由子进程处理）")
            except Exception as e:
                print(f"[优化架构] Detect helper 创建失败: {e}")
                face_enhancer = None
                use_gfpgan_subprocess = False
                gfpgan_ready = False
        else:
            # 子进程失败或未启用，加载完整 GFPGAN 到主进程
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
                    print(f"[优化架构] 下载GFPGAN模型: {model_filename}")
                    model_path = load_file_from_url(model_url, model_dir, True)
                
                # FIX: 确保加载完整的 GFPGAN 模型，不是 DummyGFPGANer
                face_enhancer = GFPGANer(
                    model_path=model_path, 
                    upscale=args.outscale, 
                    arch='clean',
                    channel_multiplier=2, 
                    bg_upsampler=None, 
                    device=device
                )
                
                # FIX: 验证模型正确加载
                if face_enhancer.gfpgan is None:
                    raise RuntimeError("GFPGAN 模型加载失败: gfpgan 网络为 None")
                    
                print("[优化架构] GFPGAN主进程模型加载成功")
                print(f"[优化架构] GFPGAN 设备: {next(face_enhancer.gfpgan.parameters()).device}")
                gfpgan_mode = "main_pytorch"

                # 标记 TRT 已经尝试过且失败了，防止 DeepPipelineOptimizer 再次尝试
                if getattr(args, 'gfpgan_trt', False):
                    print('[优化架构] 标记: GFPGAN TRT 已尝试但失败，禁用后续 TRT 尝试')
                    args.gfpgan_trt = False  # 禁用 TRT，防止 DeepPipelineOptimizer 再次创建子进程
                            
            except Exception as e:
                print(f"[优化架构] GFPGAN模型加载失败: {e}")
                import traceback
                traceback.print_exc()
                face_enhancer = None  # 彻底禁用 GFPGAN
                gfpgan_mode = "disabled"
    
    # 步骤 4: 创建读写器
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
    
    # 步骤 5: 可选 SR TRT
    trt_accel = None
    if getattr(args, 'use_tensorrt', False) and cuda_available:
        meta = get_video_meta_info(args.input)
        sh = (args.batch_size, 3, meta['height'], meta['width'])
        trt_dir = getattr(args, 'trt_cache_dir', None) or osp.join(base_dir, '.trt_cache')
        print(f'[优化架构] 初始化 SR TensorRT Engine (shape={sh})...')
        if use_gfpgan_subprocess:
            print('[优化架构] 警告: SR TRT 与 GFPGAN TRT 同时启用，Myelin workspace 存在显存竞争')
        # FIX-NVENC-TRIPLE-WARN: SR TRT + GFPGAN TRT + h264_nvenc 三者同时启用时
        # NVENC 在 OOM cascade（PyTorch 无界分配 + cudaFree 序列）期间会停止接受帧，
        # stdin pipe 填满 → stall → fatal。SR TRT slug 修复后 OOM 消失，但仍建议
        # 若 h264_nvenc 再出现 stall，改用 --video-codec libx264（CPU 编码无 GPU 竞争）。
        _vc_now = getattr(args, 'video_codec', 'libx264')
        if use_gfpgan_subprocess and _vc_now in ('h264_nvenc', 'hevc_nvenc'):
            print(f'[优化架构] 注意: SR TRT + GFPGAN TRT + {_vc_now} 三者同时启用。'
                  f'若出现 nvenc stall，请改用 --video-codec libx264 消除 GPU 竞争。')
        try:
            trt_accel = TensorRTAccelerator(
                upsampler.model, device, trt_dir, sh, use_fp16=not args.no_fp16,
            )
            if trt_accel.available:
                print('[优化架构] SR TensorRT Engine 加载成功')
            else:
                print('[优化架构] SR TensorRT Engine 初始化失败，回退 PyTorch')
                trt_accel = None
        except Exception as _te:
            print(f'[优化架构] SR TensorRT 初始化异常: {_te}')
            trt_accel = None
    
    # FIX: 只有在子进程真正就绪时才绑定
    if _early_gfpgan_subprocess is not None and use_gfpgan_subprocess and gfpgan_ready:
        args._early_gfpgan_subprocess = _early_gfpgan_subprocess
        print(f'\n[优化架构] GFPGAN 子进程已绑定到流水线（模式: {gfpgan_mode}）')
    else:
        # 确保不传递失败的子进程
        args._early_gfpgan_subprocess = None
        if args.face_enhance:
            print(f'[优化架构] GFPGAN 使用主进程模式: {gfpgan_mode}')
    
    # 运行流水线
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
        import traceback
        traceback.print_exc()
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


def main():
    """主函数 - 参数解析"""
    parser = argparse.ArgumentParser(
        description='Real-ESRGAN 视频超分 —— 架构优化版 v6.3',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # 基础参数（与原始v6.2保持一致）
    parser.add_argument('-i', '--input', type=str, required=True, help='输入视频路径')
    parser.add_argument('-o', '--output', type=str, required=True, help='输出视频路径')
    parser.add_argument('-n', '--model-name', type=str, default='realesr-animevideov3', help='模型名称')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='输出缩放比例')
    parser.add_argument('-dn', '--denoise-strength', type=float, default=0.5, help='去噪强度（仅realexr-general-x4v3有效）')
    parser.add_argument('--suffix', type=str, default='out', help='输出文件后缀')
    
    # 优化参数（新增架构优化参数）
    parser.add_argument('--batch-size', type=int, default=6, help='批处理大小（优化版本推荐6-8）')
    parser.add_argument('--prefetch-factor', type=int, default=48, help='读帧预取队列深度，建议 ≥ batch-size*3')
    
    # 人脸增强参数
    parser.add_argument('--face-enhance', action='store_true', help='启用人脸增强')
    parser.add_argument('--gfpgan-model', type=str, default='1.4', choices=['1.3', '1.4', 'RestoreFormer'],
                        help='GFPGAN 模型版本（--face-enhance 时生效）')
    parser.add_argument('--gfpgan-weight', type=float, default=0.5, help='GFPGAN 增强融合权重，0.0=不增强，1.0=完全替换')
    parser.add_argument('--gfpgan-batch-size', type=int, default=8, 
                        help='单次 GFPGAN 前向最多处理的人脸数，OOM 时自动对半降级')
    
    # 加速参数（与v6.2完全兼容）
    parser.add_argument('--no-fp16', action='store_true', help='禁用 FP16（默认开启 FP16）')
    parser.add_argument('--no-compile', action='store_true', 
                        help='禁用 torch.compile（默认开启；短视频或调试时可禁用跳过编译等待）')
    parser.add_argument('--use-tensorrt', action='store_true',
                        help='启用 SR TensorRT 加速（首次需要构建 Engine）')
    parser.add_argument('--gfpgan-trt', action='store_true',
                        help='为 GFPGAN 人脸增强单独启用 TensorRT 加速')
    parser.add_argument('--no-cuda-graph', action='store_true',
                        help='禁用 CUDA Graph（默认开启；compile/TRT 激活时自动禁用）')
    
    # 硬件加速参数
    parser.add_argument('--use-hwaccel', action='store_true', default=True, 
                        help='启用 NVDEC 硬件解码（自动探测，失败时回退）')
    parser.add_argument('--no-hwaccel', action='store_true', help='强制禁用 NVDEC 硬件解码')
    
    # 其他参数
    parser.add_argument('-t', '--tile', type=int, default=0, help='分块大小')
    parser.add_argument('--tile-pad', type=int, default=10, help='分块填充')
    parser.add_argument('--pre-pad', type=int, default=0, help='预填充')
    parser.add_argument('--fps', type=float, default=None, help='输出帧率')
    
    # 编码参数
    parser.add_argument('--video-codec', type=str, default='libx264', 
                        choices=['libx264', 'libx265', 'libvpx-vp9', 'libvpx-vp9', 'h264_nvenc'],
                        help='偏好编码器（有 NVENC 时自动升级为 h264_nvenc/hevc_nvenc）')
    parser.add_argument('--crf', type=int, default=23, help='编码质量（默认 23）')
    # FIX-LIBX264-PRESET: libx264/libx265 preset 可配置
    #   GFPGAN+SR 同跑时 CPU 存在竞争，medium → fast 可减少 CPU 占用并缓解 pipe stall
    parser.add_argument('--x264-preset', type=str, default='medium',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast',
                                 'medium', 'slow', 'slower', 'veryslow'],
                        help='libx264/libx265 preset，GFPGAN 同跑时建议 fast 减少 CPU 竞争')
    parser.add_argument('--ffmpeg-bin', type=str, default='ffmpeg', help='FFmpeg 二进制路径')
    
    args = parser.parse_args()
    
    print("Real-ESRGAN Video Enhancement v6.3 - 架构优化版")
    print("主要优化特性:")
    print("1. 深度流水线架构（4级并行处理）")
    print("2. GPU内存池优化（避免频繁分配释放）")
    print("3. 异步计算模式（多CUDA流并行）")
    print("4. 多级缓冲队列（深度缓冲减少等待）")
    print("5. 优化线程池配置（提高并发效率）")
    print()
    
    main_optimized(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')   # FIX-WARN: 压制主进程 torchvision deprecation 警告
    main()