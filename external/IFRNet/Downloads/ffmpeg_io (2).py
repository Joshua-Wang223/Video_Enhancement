#!/usr/bin/env python3
"""
Real-ESRGAN Video Enhancement - FFmpeg 读写模块
包含 FFmpegReader 和 FFmpegWriter 类
"""

import os
import sys
import time
import select
import queue
import threading
import subprocess
from typing import Optional
import re

import numpy as np
import ffmpeg

from realesrgan_utils import get_video_meta_info

# ── [FIX-NVENC-PIPE] NVENC pipe 模式参数常量 ──────────────────────────────────
# NVENC 内部帧缓冲数（-surfaces N）：
#   NVENC 硬件编码器内部维护一个帧槽池（surfaces），每个 slot 存储一帧正在被硬件
#   编码的图像。默认值为 8，对于均匀帧率的文件输入已经足够；但 pipe 输入存在速率
#   抖动，当短时供帧速率超过硬件编码速率时，较小的 surfaces 数会导致 FFmpeg 无法
#   向 NVENC 提交新帧（硬件满载等待回收），引发编码器停顿（stall）。
#   扩大至 32 可覆盖约 1 秒的帧缓冲（@30fps），基本消除 pipe 速率抖动的影响。
_NVENC_SURFACES_PIPE: int = 32

# NVENC VBR 模式前向帧预看窗口（-rc-lookahead N）：
#   仅在 crf>0（-rc:v vbr 模式）下启用。NVENC 默认不使用前向预看（N=0），
#   设为 16 后编码器可向前分析 16 帧的运动复杂度，进行更精准的码率分配。
#   典型 PSNR 改善 0.2-0.5 dB（1080p VBR）。
#   注意：lookahead 需要 N 帧前瞻缓冲，与 -delay 0（零输出延迟）互斥，
#   故仅在 VBR 路径启用，QP=0 路径改用 -delay 0。
_NVENC_LOOKAHEAD_VBR: int = 16


# ── [FIX-SLICE-THREAD] 编码并行度自动探测 ──────────────────────────────────────
def _detect_encode_parallelisms(n_threads_hint: Optional[int] = None) -> dict:
    """
    自动探测 CPU / 内存资源，返回最优软编码并行参数字典。

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
    ffmpeg_threads : int    FFmpeg 全局 -threads 值，用于 demux/filter graph
                            = min(cpu_logical, 8)
    """
    cpu_logical = os.cpu_count() or 4

    # 物理核心数：从 /proc/cpuinfo 读 "core id" 去重；失败则估算
    cpu_physical = max(cpu_logical // 2, 1)
    try:
        _core_ids: set = set()
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
    #   · 内存约束：每个 slice 约需 0.25 GiB 额外行缓冲
    #   · 下限 2：至少 2 片才有并行效果
    slices_by_cpu  = encode_threads
    slices_by_mem  = max(2, int(mem_avail_gb / 0.25))
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


class FFmpegReader:
    """通过FFmpeg pipe读取视频帧"""

    FRAME_TIMEOUT = object()  # 超时哨兵，与真正的 EOF None 区分

    def __init__(self, input_path, ffmpeg_bin='ffmpeg', prefetch_factor=16, use_hwaccel=True, quiet=True):
        self.input_path = input_path
        self.ffmpeg_bin = ffmpeg_bin
        self.prefetch_factor = prefetch_factor
        self.use_hwaccel = use_hwaccel

        meta = get_video_meta_info(input_path)
        self.width = meta['width']
        self.height = meta['height']
        self.fps = meta['fps']
        self.nb_frames = meta['nb_frames']
        self.audio = meta['audio']

        input_kwargs = {}
        if self.use_hwaccel:
            input_kwargs['hwaccel'] = 'auto'

        self._ffmpeg_input = ffmpeg.input(input_path, **input_kwargs)

        self._frame_queue = queue.Queue(maxsize=prefetch_factor)
        self._running = True

        # 诊断 / 状态
        self._stderr_lines = []
        self._stderr_lock = threading.Lock()
        self._ffmpeg_process = None
        self._eof_sent = False
        self._frames_produced = 0
        self._last_error = None

        self._quiet = quiet

        self._thread = threading.Thread(target=self._read_loop, daemon=True,
                                        name='ffmpeg_reader_loop')
        self._thread.start()

    def _vlog(self, *args, **kwargs):
        """受静默模式控制的日志打印（仅在 --no-quiet 时输出）"""
        if not self._quiet:
            print(*args, **kwargs)

    # ----------------------------------------------------------------------
    # 内部：stderr drain 线程
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

    def _get_stderr_tail(self, n=30) -> str:
        with self._stderr_lock:
            if not self._stderr_lines:
                return '(无 stderr 输出)'
            lines = self._stderr_lines[-n:]
        return ''.join(lines)

    # ----------------------------------------------------------------------
    # 保证 EOF 一定送到下游（修死锁的关键）
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
                self._vlog(f"\n[FFmpegReader] EOF 已发送到下游（frames_produced={self._frames_produced}）",
                           flush=True)
                return
            except queue.Full:
                retries += 1
                # pipeline 主动关闭，不再坚持
                if not self._running:
                    print(f"\n[FFmpegReader] 放弃发送 EOF：_running=False",
                          flush=True)
                    return
                if retries % 10 == 0:
                    print(f"\n[FFmpegReader] EOF 发送受阻 {retries}s，"
                          f"下游队列持续满（frame_queue={self._frame_queue.qsize()}/"
                          f"{self._frame_queue.maxsize}）", flush=True)
                if retries >= max_wait_s:
                    print(f"\n[FFmpegReader] 放弃发送 EOF：超过 {max_wait_s:.0f}s 仍无法入队",
                          flush=True)
                    return
            except Exception as e:
                print(f"\n[FFmpegReader] 发送 EOF 异常: {e}", flush=True)
                return

    # ----------------------------------------------------------------------
    # 主读取循环
    # ----------------------------------------------------------------------
    def _read_loop(self):
        process = None
        stderr_thread = None

        # try:
        #     process = (
        #         self._ffmpeg_input
        #         .output('pipe:', format='rawvideo', pix_fmt='rgb24', vsync=0)
        #         .run_async(pipe_stdout=True, pipe_stderr=True, quiet=False)
        #     )
        #     self._ffmpeg_process = process
        try:
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

            stderr_thread = threading.Thread(
                target=self._drain_stderr, args=(process,),
                daemon=True, name='ffmpeg_reader_stderr')
            stderr_thread.start()

            frame_size = self.width * self.height * 3
            print(f"\n[FFmpegReader] FFmpeg 进程启动（pid={process.pid}, "
                  f"frame_size={frame_size}B, queue_size={self.prefetch_factor}）",
                  flush=True)

            while self._running:
                # 读一帧
                try:
                    in_bytes = process.stdout.read(frame_size)
                except Exception as e:
                    self._last_error = f'stdout.read 异常: {e}'
                    print(f"\n[FFmpegReader] 致命: stdout.read 异常 "
                          f"@frame={self._frames_produced}: {e}", flush=True)
                    break

                # 正常 EOF
                if not in_bytes:
                    rc = process.poll()
                    self._vlog(f"\n[FFmpegReader] stdout EOF @frame={self._frames_produced}, "
                               f"ffmpeg_rc={rc}", flush=True)
                    if rc is not None and rc != 0:
                        # 异常退出：打诊断
                        tail = self._get_stderr_tail(40)
                        print(f"\n[FFmpegReader] ffmpeg 非零退出 (rc={rc})，stderr 尾部:\n{tail}",
                              flush=True)
                        self._last_error = f'ffmpeg 非零退出 rc={rc}'
                    break

                # 短读：ffmpeg 多半已异常退出
                if len(in_bytes) != frame_size:
                    rc = process.poll()
                    tail = self._get_stderr_tail(40)
                    print(f"\n[FFmpegReader] 致命: 短读! 期望={frame_size} "
                          f"实际={len(in_bytes)} @frame={self._frames_produced} "
                          f"ffmpeg_rc={rc}", flush=True)
                    print(f"\n[FFmpegReader] ffmpeg stderr 尾部:\n{tail}",
                          flush=True)
                    self._last_error = (f'短读 got={len(in_bytes)} '
                                        f'want={frame_size} rc={rc}')
                    break

                # reshape
                try:
                    frame = np.frombuffer(in_bytes, np.uint8).reshape(
                        [self.height, self.width, 3])
                except Exception as e:
                    print(f"\n[FFmpegReader] reshape 失败 "
                          f"@frame={self._frames_produced}: {e}", flush=True)
                    self._last_error = f'reshape 失败: {e}'
                    break

                # 入队（阻塞等队列有空位）
                # FIX-READER: ffmpeg 以 rc=0 正常退出后队列仍可能满（下游消费滞后）。
                # 旧逻辑对 rc=0 也抛 RuntimeError，导致最后若干帧丢失。
                # 新逻辑：仅 rc≠0（真正的异常退出）才报错退出；rc=0 继续等待入队。
                while self._running:
                    try:
                        self._frame_queue.put(frame, timeout=1.0)
                        self._frames_produced += 1
                        break
                    except queue.Full:
                        rc = process.poll()
                        if rc is not None and rc != 0:
                            # 真正的异常退出：立即终止，避免无限等待
                            tail = self._get_stderr_tail(20)
                            print(f"\n[FFmpegReader] ffmpeg 异常退出但下游满 "
                                  f"@frame={self._frames_produced} "
                                  f"rc={rc}, stderr:\n{tail}", flush=True)
                            self._last_error = f'下游满时 ffmpeg 异常退出 rc={rc}'
                            raise RuntimeError('ffmpeg exited with error while queue full')
                        # rc=0（正常完成）或 rc=None（仍在运行）：继续等空位
                        continue
                else:
                    # self._running 变为 False
                    break

        except Exception as e:
            import traceback
            print(f"\n[FFmpegReader] _read_loop 异常: {e}", flush=True)
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
                    print(f"\n[FFmpegReader] 终止 ffmpeg 异常: {e}", flush=True)

            # # 2) 打印最终 stderr（若尚未打印过）
            # tail = self._get_stderr_tail(40)
            # if tail and tail != '(无 stderr 输出)':
            #     print(f"\n[FFmpegReader] 最终 ffmpeg stderr:\n{tail}",
            #           flush=True)

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
                print(f"\n[FFmpegReader] ffmpeg 异常退出 rc={rc_final}, "
                      f"last_error={self._last_error}", flush=True)
                if tail and tail != '(无 stderr 输出)':
                    print(f"\n[FFmpegReader] ffmpeg stderr:\n{tail}", flush=True)
            else:
                # 正常结束：只留一行摘要，不再刷屏
                self._vlog(f"\n[FFmpegReader] ffmpeg 正常结束 rc={rc_final}, "
                           f"frames_produced={self._frames_produced}", flush=True)

            # 3) 关键：无论如何把 EOF 送到下游
            self._send_eof_guaranteed()

            # 4) 等 stderr drain 线程结束（非强制）
            if stderr_thread is not None and stderr_thread.is_alive():
                stderr_thread.join(timeout=1.0)

            self._vlog(f"\n[FFmpegReader] _read_loop 退出 "
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
            return FFmpegReader.FRAME_TIMEOUT

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
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)


class FFmpegWriter:
    """通过FFmpeg pipe写入视频帧"""

    _SENTINEL  = object()
    _MAX_BATCH = 8
    _STDERR_IGNORE = (
        'x265 [info]:', 'x265 [warning]:', 'set_mempolicy:',
        'encoded ', 'Weighted P-Frames', 'consecutive B-frames',
        'frame I:', 'frame P:', 'frame B:',
        'using cpu capabilities:', 'slice threads:', 'frame threads:',
        'x264 [info]:', 'x264 [warning]:',
        'Initialized NPP', 'NVENC session', 'GPU #',
    )

    WRITE_TIMEOUT = 300.0
    THREAD_JOIN_TIMEOUT = 5.0
    PROCESS_TERMINATE_TIMEOUT = 10.0
    WRITE_CHUNK_SIZE = 65536

    def __init__(self, args, audio, height, width, output_path, fps,
                 n_threads=None, extra_codec_args=None, audio_src=None):
        self.args = args
        self.audio = audio
        self.height = height
        self.width = width
        self.output_path = output_path
        self.fps = fps

        self._frame_queue = queue.Queue(maxsize=24)
        self._running = True
        self._broken = False
        self._error = None
        self._write_error = None
        self._stderr_buffer = []
        self._frames_written_to_pipe = 0
        self._bytes_written_to_pipe = 0
        self._n_threads = n_threads
        self._extra_codec_args = extra_codec_args
        self._audio_src = audio_src

        self._quiet = getattr(args, 'quiet', True)
        self._process = None

        # [FIX-SLICE-THREAD] 自动探测 CPU / 内存，计算最优软编码并行参数
        self._encode_par = _detect_encode_parallelisms(n_threads)

        self._init_ffmpeg_process()

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True,
            name='ffmpeg_stderr_reader')
        self._stderr_thread.start()

        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def _vlog(self, *args, **kwargs):
        """受静默模式控制的日志打印（仅在 --no-quiet 时输出）"""
        if not self._quiet:
            print(*args, **kwargs)

    def _drain_stderr(self):
        try:
            if not self._process or not self._process.stderr:
                return
            for line in self._process.stderr:
                decoded = line.decode(errors='ignore').rstrip()
                self._stderr_buffer.append(decoded)
                if decoded and not any(decoded.lstrip().startswith(p)
                                       for p in self._STDERR_IGNORE):
                    print(f'[FFmpeg ERR] {decoded}')
        except Exception:
            pass

    def _get_ffmpeg_stderr(self, tail_lines=10):
        if not self._stderr_buffer:
            return '(无 stderr 输出)'
        full = ''.join(self._stderr_buffer)
        lines = full.strip().split('\n')
        if len(lines) > tail_lines:
            lines = lines[-tail_lines:]
        return '\n'.join(lines)

    @staticmethod
    def _check_nvenc_available(ffmpeg_bin='ffmpeg', codec='h264_nvenc') -> bool:
        test_cmd = [
            ffmpeg_bin, '-y',
            '-f', 'lavfi', '-i', 'nullsrc=s=64x64:d=0.04:r=25',
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
                    for line in stderr_text.split('\n'):
                        line = line.strip()
                        if any(kw in line for kw in nvenc_errors):
                            print(f'[FFmpegWriter]   {line}', flush=True)
                    return False
                else:
                    print(f'[FFmpegWriter] NVENC 预检测返回 rc={result.returncode}，'
                          f'但非 NVENC 特有错误，继续使用', flush=True)
                    return True
            return True
        except subprocess.TimeoutExpired:
            print(f'[FFmpegWriter] NVENC 预检测超时（>10s），假定可用', flush=True)
            return True
        except FileNotFoundError:
            print(f'[FFmpegWriter] FFmpeg 未找到: {ffmpeg_bin}', flush=True)
            return True
        except Exception as e:
            print(f'[FFmpegWriter] NVENC 预检测异常: {e}，假定可用', flush=True)
            return True

    def _init_ffmpeg_process(self):
        video_codec = getattr(self.args, 'video_codec', 'libx264')
        crf = getattr(self.args, 'crf', 23)
        x264_preset = getattr(self.args, 'x264_preset', 'medium')

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

        _base, _ext = os.path.splitext(self.output_path)
        self._tmp_video_path = f'{_base}.tmp_novid{_ext}'
        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        tmp_dir = os.path.dirname(self._tmp_video_path)
        if tmp_dir:
            os.makedirs(tmp_dir, exist_ok=True)

        # [FIX-SLICE-THREAD] 读取自动探测的并行参数
        _par = getattr(self, '_encode_par', _detect_encode_parallelisms(self._n_threads))
        _et  = _par['encode_threads']   # 编码线程数
        _s   = _par['slices']           # x264 分片数
        _ft  = _par['ffmpeg_threads']   # FFmpeg 全局线程数

        # x265 frame-threads：默认 min(4, cpu_logical//2)
        _x265_ft = max(2, min(4, _par['cpu_logical'] // 2))
        # x265 pool：线程池总大小 = encode_threads
        _x265_pool = _et

        # [FIX-PRESET] NVENC preset 默认 p4（最快），允许通过 x264_preset 调优
        if video_codec in ('h264_nvenc', 'hevc_nvenc'):
            nvenc_preset = x264_preset if x264_preset != 'medium' else 'p4'
        else:
            nvenc_preset = x264_preset

        cmd_args = [
            getattr(self.args, 'ffmpeg_bin', 'ffmpeg'),
            '-y',
            # [FIX-SLICE-THREAD] FFmpeg 全局 -threads：作用于 demux / filter graph
            '-threads', str(_ft),
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', 'pipe:',
        ]

        # [FIX-AUDIO-MUX] audio_src 内联音轨：在编码时直接 mux，省去 post-encode 合并
        if self._audio_src and os.path.exists(self._audio_src):
            cmd_args += [
                '-i', self._audio_src,
                '-c:a', 'copy',
                '-map', '0:v',
                '-map', '1:a?',
            ]

        # [FIX-EXTRA-CODEC-ARGS] extra_codec_args 完全替换 quality_args
        if self._extra_codec_args:
            cmd_args += ['-vcodec', video_codec, '-pix_fmt', 'yuv420p'] + self._extra_codec_args
        else:
            # [FIX-SLICE-THREAD / FIX-NVENC-PIPE / FIX-LOSSLESS]
            # 依据编解码器和 crf 构造最优编码参数
            if video_codec in ('h264_nvenc', 'hevc_nvenc'):
                if crf == 0:
                    # [FIX-LOSSLESS] NVENC 无损：常量 QP=0，去掉 vbr 码率控制
                    # [FIX-NVENC-PIPE] pipe 场景优化：-bf 0 + -surfaces N + -delay 0
                    quality_args = [
                        '-preset', nvenc_preset,
                        '-qp', '0', '-b:v', '0',
                        '-bf', '0',
                        '-surfaces', str(_NVENC_SURFACES_PIPE),
                        '-delay', '0',
                    ]
                else:
                    # [FIX-NVENC-PIPE] NVENC VBR（cq）模式 pipe 场景优化
                    #   -bf 0 禁用 B 帧，降低流水线缓冲延迟
                    #   -rc-lookahead N 前向帧预看（与 -delay 0 互斥）
                    quality_args = [
                        '-preset', nvenc_preset,
                        '-rc:v', 'vbr', '-cq:v', str(crf), '-b:v', '0',
                        '-bf', '0',
                        '-rc-lookahead', str(_NVENC_LOOKAHEAD_VBR),
                        '-surfaces', str(_NVENC_SURFACES_PIPE),
                    ]
                cmd_args += ['-vcodec', video_codec, '-pix_fmt', 'yuv420p'] + quality_args
            elif video_codec == 'libx265':
                # [FIX-LOSSLESS] crf=0 在 x265 中不是无损！仅为极高质量有损。
                # 无损需显式 lossless=1，同时移除 -crf 参数（互斥）。
                if crf == 0:
                    cmd_args += [
                        '-vcodec', video_codec,
                        '-pix_fmt', 'yuv420p',
                        '-preset', nvenc_preset,
                        '-x265-params',
                        f'lossless=1:pools={_x265_pool}:frame-threads={_x265_ft}',
                    ]
                else:
                    cmd_args += [
                        '-vcodec', video_codec,
                        '-pix_fmt', 'yuv420p',
                        '-preset', nvenc_preset,
                        '-crf', str(crf),
                        '-x265-params',
                        f'pools={_x265_pool}:frame-threads={_x265_ft}',
                    ]
            elif video_codec == 'copy':
                # 管道输入只能是原始流，无法 copy 到 mp4，直接回退
                print("[FFmpegWriter] 警告：copy 模式不适用于管道原始输入，自动改用 libx264")
                if crf == 0:
                    cmd_args += [
                        '-vcodec', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-preset', nvenc_preset,
                        '-qp', '0',
                        '-x264-params', f'threads={_et}:slices={_s}',
                    ]
                else:
                    cmd_args += [
                        '-vcodec', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-preset', nvenc_preset,
                        '-crf', str(crf),
                        '-x264-params', f'threads={_et}:slices={_s}',
                    ]
            else:
                # [FIX-LOSSLESS] libx264 crf=0 语义等价于无损，但显式 -qp 0 更清晰
                if crf == 0:
                    cmd_args += [
                        '-vcodec', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-preset', nvenc_preset,
                        '-qp', '0',
                        '-x264-params', f'threads={_et}:slices={_s}',
                    ]
                else:
                    cmd_args += [
                        '-vcodec', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-preset', nvenc_preset,
                        '-crf', str(crf),
                        '-x264-params', f'threads={_et}:slices={_s}',
                    ]

        # [FIX-AUDIO-MUX] 使用 audio_src 内联音轨时跳过 -an
        if not self._audio_src or not os.path.exists(self._audio_src or ''):
            cmd_args += ['-an']

        cmd_args += ['-loglevel', 'error', self._tmp_video_path]

        # [FIX-SLICE-THREAD / FIX-NVENC-PIPE] 打印编码参数摘要
        if 'nvenc' in video_codec:
            if crf == 0:
                _enc_info = (
                    f'[FIX-NVENC-PIPE] NVENC 无损(QP=0): '
                    f'preset={nvenc_preset}  bf=0  '
                    f'surfaces={_NVENC_SURFACES_PIPE}  delay=0  '
                    f'ffmpeg_threads={_ft}(全局demux，不影响NVENC硬件单元)'
                )
            else:
                _enc_info = (
                    f'[FIX-NVENC-PIPE] NVENC VBR(cq={crf}): '
                    f'preset={nvenc_preset}  bf=0  '
                    f'rc-lookahead={_NVENC_LOOKAHEAD_VBR}  '
                    f'surfaces={_NVENC_SURFACES_PIPE}  '
                    f'ffmpeg_threads={_ft}(全局demux，不影响NVENC硬件单元)'
                )
            print(f'   {_enc_info}', flush=True)
        else:
            _codec_l = video_codec.lower()
            if 'x265' in _codec_l:
                _thread_info = (
                    f'[FIX-SLICE-THREAD] 软编码并行: '
                    f'cpu={_par["cpu_logical"]}逻辑/{_par["cpu_physical"]}物理  '
                    f'mem_avail={_par["mem_avail_gb"]:.1f}GiB  '
                    f'encode_threads={_et}(frame-threads={_x265_ft})  ffmpeg_threads={_ft}'
                )
            else:
                _thread_info = (
                    f'[FIX-SLICE-THREAD] 软编码并行: '
                    f'cpu={_par["cpu_logical"]}逻辑/{_par["cpu_physical"]}物理  '
                    f'mem_avail={_par["mem_avail_gb"]:.1f}GiB  '
                    f'encode_threads={_et}  slices={_s}  ffmpeg_threads={_ft}'
                )
            print(f'   {_thread_info}', flush=True)

        print(f"[FFmpegWriter] 命令: {' '.join(cmd_args)}", flush=True)

        self._process = subprocess.Popen(
            cmd_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        try:
            import fcntl
            fcntl.fcntl(self._process.stdin.fileno(), fcntl.F_SETPIPE_SZ, 4 * 1024 * 1024)
        except PermissionError:
            pass
        except Exception as _e:
            print(f"[FFmpegWriter] 管道缓冲区扩大失败（使用默认 64KB）: {_e}", flush=True)

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
        """
        [FIX-BATCH-WRITE] 写线程主循环：
          · 攒够 _MAX_BATCH（8）帧后批量写入，减少 syscall 约 8×
          · 超时 0.2s 自动刷出，防止帧饥饿
          · 用 _SENTINEL 哨兵替代 None 作为结束信号
          · 保留完整的 stall / NVENC 死亡检测 / 重试容错逻辑（原代码 130 行）
        """
        pending: list[bytes] = []
        _vc = getattr(self.args, 'video_codec', 'libx264')
        if _vc in ('h264_nvenc', 'hevc_nvenc'):
            _single_timeout = 300.0
            _max_stall_s   = 600.0
            _retry_sleep   = 20.0
        else:
            _single_timeout = 60.0
            _max_stall_s   = 180.0
            _retry_sleep   = 5.0
        _max_consecutive_err = 3

        # ── 内联辅助：刷空 pending 列表 ──────────────────────────────────
        def _flush():
            nonlocal pending
            if not pending:
                return
            data = b''.join(pending)
            n    = len(pending)
            pending = []
            self._write_batch_with_stall_check(
                data, n,
                _single_timeout, _max_stall_s, _retry_sleep, _max_consecutive_err,
            )

        try:
            while self._running:
                try:
                    frame = self._frame_queue.get(timeout=0.2)
                except queue.Empty:
                    # 超时：刷出已累积帧，防止帧饥饿
                    _flush()
                    if self._broken:
                        return
                    # 检查 FFmpeg 是否意外退出
                    if self._process and self._process.poll() is not None:
                        rc = self._process.returncode
                        stderr_text = self._get_ffmpeg_stderr(tail_lines=10)
                        print(f"[FFmpegWriter] FFmpeg 进程意外退出 "
                              f"(returncode={rc})", flush=True)
                        print(f"[FFmpegWriter] 最后的 stderr:\n{stderr_text}", flush=True)
                        self._broken = True
                        break
                    continue

                # ── _SENTINEL 哨兵 ──────────────────────────────────────────
                if frame is self._SENTINEL:
                    _flush()
                    break

                # ── 帧处理 ──────────────────────────────────────────────────
                try:
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

                    # [首帧 NVENC 死亡检测] 前 6 帧检查 stderr 中是否有编码器初始化失败标志
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

                    pending.append(frame_bytes)
                except (BrokenPipeError, OSError) as e:
                    print(f"[FFmpegWriter] 管道断裂: {e}", flush=True)
                    self._broken = True
                    break
                except Exception as e:
                    print(f"[FFmpegWriter] 处理帧异常: {e}", flush=True)
                    self._broken = True
                    break

                # ── 批量写触发 ──────────────────────────────────────────────
                if len(pending) >= self._MAX_BATCH:
                    _flush()
                    if self._broken:
                        return

        except Exception as e:
            # [FIX-ERROR-PROPAGATION] 捕获所有未预期的异常，存入 self._error 供 write_frame() 传播
            self._error = e
            print(f"[FFmpegWriter] _write_loop 未捕获异常: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self._broken = True
            return

        # ── 循环结束：刷出残余帧（仅非 broken 时） ──────────────────────
        if not self._broken and pending:
            self._write_batch_with_stall_check(
                b''.join(pending), len(pending),
                _single_timeout, _max_stall_s, _retry_sleep, _max_consecutive_err,
            )

    def _write_batch_with_stall_check(
        self,
        data: bytes,
        n_frames: int,
        single_write_timeout: float,
        max_stall_s: float,
        retry_sleep: float,
        max_consecutive_errors: int,
    ) -> bool:
        """
        [FIX-BATCH-WRITE] 批量写入 data（多帧 bytes 拼接），带 stall 检测和重试逻辑。

        用于 NVENC 场景下 SR+GFPGAN 与硬件编码器同 GPU 资源竞争导致的 pipe 阻塞。
        保留原 _write_loop 中全套容错逻辑：FFmpeg 退出检测、stall 超时判断、
        首 stall 首次报告、诊断提示。

        参数
        ──────────────────────────────────────────────────────────
        data                   : 多帧 bytes 拼接（b''.join(pending)）
        n_frames               : data 中包含的帧数（用于 frames_written 计数）
        single_write_timeout   : 单次 _write_with_timeout 超时（NVENC=300s, 软编码=60s）
        max_stall_s            : 最大累积 stall 时间（NVENC=600s, 软编码=180s）
        retry_sleep            : stall 重试间隔（NVENC=20s, 软编码=5s）
        max_consecutive_errors : 最大连续失败次数（默认 3）

        返回
        ──────────────────────────────────────────────────────────
        True  = 写入成功（self._frames_written_to_pipe += n_frames）
        False = 写入失败（self._broken 可能已被设为 True）
        """
        stall_elapsed = 0.0
        consecutive_errors = 0
        stall_first_reported = False
        _vc = getattr(self.args, 'video_codec', 'libx264')

        while True:
            if not self._running:
                return False

            if self._write_with_timeout(data, timeout=single_write_timeout):
                self._frames_written_to_pipe += n_frames
                return True

            err_detail = self._write_error or '(未知错误)'
            ffmpeg_alive = (self._process.poll() is None)

            # ── FFmpeg 已退出 ──────────────────────────────────────────
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
                    return False
                # FFmpeg 已退出，返回 False 但不标记 broken（外层可能继续处理）
                return False

            # ── Stall 累积检测 ─────────────────────────────────────────
            stall_elapsed += single_write_timeout
            if stall_elapsed >= max_stall_s:
                if _vc in ('h264_nvenc', 'hevc_nvenc'):
                    _suggestion = (f'建议改用 --video-codec libx264 避免 '
                                   f'SR+GFPGAN 与 {_vc} 的 GPU 资源竞争。')
                else:
                    _suggestion = (f'编码器 {_vc} stdin 阻塞超过 '
                                   f'{max_stall_s:.0f}s，'
                                   f'请检查磁盘空间或增大 --crf 降低码率。')
                print(f'[FFmpegWriter] stall 超过 {max_stall_s:.0f}s，'
                      f'放弃写入。{_suggestion}',
                      flush=True)
                stderr_text = self._get_ffmpeg_stderr(tail_lines=20)
                if stderr_text:
                    print(f'[FFmpegWriter] FFmpeg stderr (共{len(self._stderr_buffer)}行):\n{stderr_text}',
                          flush=True)
                self._broken = True
                return False

            # ── Stall 首次报告 ─────────────────────────────────────────
            if not stall_first_reported:
                stall_first_reported = True
                _early_stderr = self._get_ffmpeg_stderr(tail_lines=5)
                if _early_stderr:
                    print(f'[FFmpegWriter] stall 首次发生，FFmpeg stderr:\n{_early_stderr}',
                          flush=True)
            print(f'[FFmpegWriter] stdin 暂时阻塞（已等待 {stall_elapsed:.0f}s'
                  f' / {max_stall_s:.0f}s），'
                  f'{retry_sleep:.0f}s 后重试同一批...',
                  flush=True)
            time.sleep(retry_sleep)

    def write_frame(self, frame):
        # [FIX-ERROR-PROPAGATION] 写线程异常 → 调用方 raise，避免静默失败
        if self._error is not None:
            raise RuntimeError(f'FFmpegWriter 内部写错误: {self._error}') from self._error

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
        self._vlog("[FFmpegWriter] 阶段1/5: 等待写入线程完成...", flush=True)

        if not self._broken:
            for _ in range(3):
                try:
                    self._frame_queue.put(self._SENTINEL, timeout=2.0)
                    break  # 成功入队则退出
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

        self._vlog("[FFmpegWriter] 阶段2/5: 写入线程已结束", flush=True)
        self._running = False

        if self._process and self._process.poll() is None:
            self._vlog("[FFmpegWriter] 阶段3/5: 等待 FFmpeg 完成编码...", flush=True)

            if self._process.stdin and not self._process.stdin.closed:
                try:
                    self._process.stdin.flush()
                    self._process.stdin.close()
                except Exception:
                    pass

            try:
                self._process.wait(timeout=300)
                self._vlog("[FFmpegWriter] 阶段4/5: FFmpeg 编码完成", flush=True)
            except subprocess.TimeoutExpired:
                print("[FFmpegWriter] FFmpeg 编码超时（>300s），强制终止", flush=True)
                try:
                    self._process.kill()
                    self._process.wait(timeout=10)
                except Exception:
                    pass
        else:
            print("[FFmpegWriter] 阶段3-4/5: FFmpeg 进程已终止", flush=True)

        self._vlog("[FFmpegWriter] 阶段5/5: 清理完成", flush=True)

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
                    print(f'[FFmpegWriter] 合并音轨: {_tmp} + {_src} → {self.output_path}', flush=True)
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
                            print(f'[FFmpegWriter] 音轨合并失败 (rc={_r.returncode}): {_mux_err}', flush=True)
                            print(f'[FFmpegWriter] 保留无音轨文件: {_tmp}', flush=True)
                            _tmp = None
                    except subprocess.TimeoutExpired:
                        print('[FFmpegWriter] 音轨合并超时（>300s）', flush=True)
                        _tmp = None
                    except Exception as _e:
                        print(f'[FFmpegWriter] 音轨合并异常: {_e}', flush=True)
                        _tmp = None
            elif _src is None:
                # 无音轨模式：直接重命名临时文件
                try:
                    os.rename(_tmp, self.output_path)
                    _tmp = None
                    print(f'[FFmpegWriter] 无音轨模式: 已输出 {self.output_path}', flush=True)
                except Exception as _e:
                    print(f'[FFmpegWriter] rename 失败: {_e}', flush=True)
                    _tmp = None
        elif _tmp and (self._broken or self.audio is None) and os.path.exists(_tmp):
            if os.path.exists(_tmp):
                # 破损或无音频情况，直接重命名（可能无视频流，但保留）
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