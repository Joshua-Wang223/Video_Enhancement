#!/usr/bin/env python3
import subprocess, select, time, os, queue, threading, re
from typing import Any, Optional, Dict

class FFmpegWriter:
    WRITE_TIMEOUT = 300.0
    THREAD_JOIN_TIMEOUT = 5.0
    PROCESS_TERMINATE_TIMEOUT = 10.0
    WRITE_CHUNK_SIZE = 65536
    def __init__(self, args, audio, height, width, output_path, fps):
        self.args, self.audio, self.height, self.width, self.output_path, self.fps = args, audio, height, width, output_path, fps
        self._frame_queue = queue.Queue(maxsize=64)
        self._running, self._broken, self._write_error = True, False, None
        self._stderr_buffer, self._frames_written_to_pipe, self._bytes_written_to_pipe = [], 0, 0
        self._process = None
        self._init_ffmpeg_process()
        self._stderr_thread = threading.Thread(target=self._stderr_reader_loop, daemon=True, name='ffmpeg_stderr_reader')
        self._stderr_thread.start()
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def _stderr_reader_loop(self):
        try:
            if not self._process or not self._process.stderr: return
            while True:
                line = self._process.stderr.readline()
                if not line: break
                self._stderr_buffer.append(line.decode('utf-8', errors='replace'))
        except Exception: pass

    def _get_ffmpeg_stderr(self, tail_lines=10):
        if not self._stderr_buffer: return '(无 stderr 输出)'
        full = ''.join(self._stderr_buffer)
        lines = full.strip().split('\n')
        return '\n'.join(lines[-tail_lines:]) if len(lines) > tail_lines else '\n'.join(lines)

    @staticmethod
    def _check_nvenc_available(ffmpeg_bin='ffmpeg', codec='h264_nvenc') -> bool:
        test_cmd = [ffmpeg_bin, '-y', '-f', 'lavfi', '-i', 'nullsrc=s=64x64:d=0.04:r=25', '-frames:v', '1', '-c:v', codec, '-f', 'null', '-']
        try:
            result = subprocess.run(test_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=10)
            stderr_text = result.stderr.decode('utf-8', errors='replace').lower()
            if result.returncode != 0:
                nvenc_errors = ['openencodesessionex failed', 'no capable devices found', 'unsupported device', 'cannot load libnvidia-encode', 'nvenc']
                if any(kw in stderr_text for kw in nvenc_errors):
                    print(f'[FFmpegWriter] NVENC 预检测失败: {codec} 不可用', flush=True)
                    for line in stderr_text.split('\n'):
                        if any(kw in line for kw in nvenc_errors): print(f'[FFmpegWriter]   {line}', flush=True)
                    return False
                print(f'[FFmpegWriter] NVENC 预检测返回 rc={result.returncode}，但非 NVENC 特有错误，继续使用', flush=True)
            return True
        except Exception: return True

    def _init_ffmpeg_process(self):
        # 确保输出目录存在
        output_dir = os.path.dirname(self.output_path)
        if output_dir:  # 如果路径包含目录部分
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"[FFmpegWriter] 创建输出目录: {output_dir}", flush=True)
        
        video_codec = getattr(self.args, 'video_codec', 'libx264')
        crf, x264_preset = getattr(self.args, 'crf', 23), getattr(self.args, 'x264_preset', 'medium')
        if video_codec in ('h264_nvenc', 'hevc_nvenc'):
            print(f'[FFmpegWriter] 预检测 {video_codec} 可用性...', flush=True)
            if not self._check_nvenc_available(getattr(self.args, 'ffmpeg_bin', 'ffmpeg'), video_codec):
                _fallback = 'libx264' if 'h264' in video_codec else 'libx265'
                print(f'[FFmpegWriter] {video_codec} 不可用，自动降级到 {_fallback}', flush=True)
                video_codec = _fallback; self.args.video_codec = _fallback
            else: print(f'[FFmpegWriter] {video_codec} 预检测通过', flush=True)
        _base, _ext = os.path.splitext(self.output_path)
        self._tmp_video_path = f'{_base}.tmp_novid{_ext}'
        cmd_args = [getattr(self.args, 'ffmpeg_bin', 'ffmpeg'), '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{self.width}x{self.height}', '-r', str(self.fps), '-i', 'pipe:']
        if video_codec in ('h264_nvenc', 'hevc_nvenc'):
            cmd_args += ['-vcodec', video_codec, '-pix_fmt', 'yuv420p', '-preset', 'p4', '-cq', str(crf), '-surfaces', '4', '-delay', '0', '-rc-lookahead', '0', '-bf', '0']
        elif video_codec == 'libx265': cmd_args += ['-vcodec', video_codec, '-pix_fmt', 'yuv420p', '-crf', str(crf), '-preset', x264_preset]
        else: cmd_args += ['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(crf), '-preset', x264_preset]
        cmd_args += ['-an', self._tmp_video_path]
        print(f"[FFmpegWriter] 命令: {' '.join(cmd_args)}", flush=True)
        self._process = subprocess.Popen(cmd_args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            import fcntl; fcntl.fcntl(self._process.stdin.fileno(), fcntl.F_SETPIPE_SZ, 4 * 1024 * 1024)
        except: pass
        time.sleep(0.5)
        if self._process.poll() is not None:
            print(f"[FFmpegWriter] FFmpeg 启动失败! (returncode={self._process.returncode})", flush=True)
            print(f"[FFmpegWriter] 命令: {' '.join(cmd_args)}", flush=True)
            for line in self._get_ffmpeg_stderr(20).split('\n'): print(f"[FFmpegWriter]   {line}", flush=True)
            raise RuntimeError(f"FFmpeg 启动失败 (rc={self._process.returncode})")
        print(f"[FFmpegWriter] FFmpeg 进程已启动 (pid={self._process.pid}, codec={video_codec})", flush=True)

    def _write_with_timeout(self, data: bytes, timeout: float = WRITE_TIMEOUT) -> bool:
        if not self._process or not self._process.stdin: self._write_error = 'process 或 stdin 不可用'; return False
        if self._process.poll() is not None: self._write_error = f'FFmpeg 已退出 (rc={self._process.returncode})'; return False
        fd, deadline, offset, total = self._process.stdin.fileno(), time.time() + timeout, 0, len(data)
        _diag = self._frames_written_to_pipe < 5
        while offset < total:
            now = time.time()
            if now >= deadline: self._write_error = f'写入超时 ({offset}/{total} 字节)'; return False
            remaining = deadline - now
            try: _r, wready, xready = select.select([], [fd], [fd], min(remaining, 1.0))
            except: self._write_error = 'select() 调用失败'; return False
            if xready:
                time.sleep(0.1)
                if self._process.poll() is not None: self._write_error = f'FFmpeg 异常退出 (rc={self._process.returncode})'; return False
                self._write_error = f'select 异常条件'; return False
            if not wready:
                if self._process.poll() is not None:
                    self._write_error = f'FFmpeg 进程已退出 (rc={self._process.returncode})'; return False
                if _diag and offset == 0:
                    print(f'[FFmpegWriter][DIAG] 帧{self._frames_written_to_pipe}: select not-ready', flush=True); _diag = False
                continue
            chunk = data[offset:offset+self.WRITE_CHUNK_SIZE]
            try:
                n = os.write(fd, chunk)
                if n == 0: self._write_error = 'os.write() 返回 0'; return False
                offset += n; self._bytes_written_to_pipe += n; self._frames_written_to_pipe += 1
            except Exception as e: self._write_error = str(e); return False
        return True

    def _write_loop(self):
        consecutive_errors, max_consecutive_errors = 0, 3
        _vc = getattr(self.args, 'video_codec', 'libx264')
        if _vc in ('h264_nvenc', 'hevc_nvenc'): MAX_S, RETRY, SINGLE = 600.0, 20.0, 300.0
        else: MAX_S, RETRY, SINGLE = 180.0, 5.0, 60.0
        while self._running:
            try: frame = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                if self._process and self._process.poll() is not None:
                    print(f"[FFmpegWriter] FFmpeg 进程意外退出 (returncode={self._process.returncode})", flush=True)
                    print(f"[FFmpegWriter] 最后的 stderr:\n{self._get_ffmpeg_stderr(10)}", flush=True); self._broken = True; break
                continue
            if frame is None: break
            try:
                if self._process and self._process.stdin:
                    frame_bytes = frame.tobytes()
                    if len(frame_bytes) != self.width * self.height * 3:
                        print(f'[FFmpegWriter] [致命诊断] 帧尺寸错误', flush=True); self._broken = True; return
                    stall_elapsed, wrote_ok = 0.0, False
                    while True:
                        if not self._running: return
                        if self._write_with_timeout(frame_bytes, timeout=SINGLE):
                            consecutive_errors = 0; wrote_ok = True
                            if self._frames_written_to_pipe <= 5 and any(kw in ''.join(self._stderr_buffer).lower() for kw in ('openencodesessionex failed', 'no capable devices found', 'error while opening encoder')):
                                print(f'[FFmpegWriter] [首帧 stderr 检测] 编码器初始化失败', flush=True); self._broken = True; return
                            break
                        err_detail = self._write_error or '(未知错误)'; ffmpeg_alive = self._process.poll() is None
                        if not ffmpeg_alive:
                            consecutive_errors += 1
                            print(f"[FFmpegWriter] 写入失败（FFmpeg 已退出） ({consecutive_errors}/{max_consecutive_errors}) | {err_detail}", flush=True)
                            if consecutive_errors >= max_consecutive_errors: print(f"[FFmpegWriter] FFmpeg stderr:\n{self._get_ffmpeg_stderr(20)}", flush=True); self._broken = True; return
                            break
                        stall_elapsed += SINGLE
                        if stall_elapsed >= MAX_S:
                            print(f'[FFmpegWriter] stall 超过 {MAX_S:.0f}s，放弃写入。', flush=True); self._broken = True; return
                        if stall_elapsed <= self.WRITE_TIMEOUT + 0.1:
                            _early = self._get_ffmpeg_stderr(5)
                            if _early: print(f'[FFmpegWriter] stall 首次发生，FFmpeg stderr:\n{_early}', flush=True)
                            print(f'[FFmpegWriter] stdin 暂时阻塞（已等待 {stall_elapsed:.0f}s / {MAX_S:.0f}s），{RETRY:.0f}s 后重试...', flush=True)
                            time.sleep(RETRY)
            except Exception as e: print(f"[FFmpegWriter] 写入错误: {e}", flush=True); self._broken = True; break

    def write_frame(self, frame) -> bool:
        if not self._running or self._broken: return False
        _vc = getattr(self.args, 'video_codec', 'libx264')
        _total_timeout = 660.0 if _vc in ('h264_nvenc', 'hevc_nvenc') else 240.0
        deadline = time.time() + _total_timeout
        while True:
            if self._broken: return False
            if deadline - time.time() <= 0:
                print(f"[FFmpegWriter] 警告: 帧队列已满超时 ({_total_timeout:.0f}s)，标记管道断裂", flush=True); self._broken = True; return False
            try: self._frame_queue.put(frame, timeout=min(1.0, deadline - time.time())); return True
            except queue.Full: continue

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
        print("[FFmpegWriter] 阶段1/5: 等待写入线程完成...", flush=True)
        if not self._broken:
            for _ in range(3):
                try: self._frame_queue.put(None, timeout=2.0); break
                except queue.Full: print("[FFmpegWriter] 警告: 队列已满", flush=True); self._broken = True; break
            if self._thread.is_alive(): self._thread.join(timeout=self.THREAD_JOIN_TIMEOUT)
            if self._thread.is_alive():
                print("[FFmpegWriter] 警告: 写入线程未响应，强制终止...", flush=True)
                if self._process and self._process.poll() is None:
                    try: self._process.stdin.close() if self._process.stdin else None
                    except: pass
                    try: self._process.terminate(); self._process.wait(timeout=self.PROCESS_TERMINATE_TIMEOUT)
                    except: 
                        print("[FFmpegWriter] SIGTERM 超时，发送 SIGKILL...", flush=True)
                        try: self._process.kill(); self._process.wait(timeout=5)
                        except: pass
                self._thread.join(timeout=2.0)
                if self._thread.is_alive(): self._thread.daemon = True
                if self._process and self._process.poll() is None:
                    try: self._process.kill()
                    except: pass
        print("[FFmpegWriter] 阶段2/5: 写入线程已结束", flush=True)
        self._running = False
        if self._process and self._process.poll() is None:
            print("[FFmpegWriter] 阶段3/5: 等待 FFmpeg 完成编码...", flush=True)
            try: self._process.stdin.flush(); self._process.stdin.close()
            except: pass
            try: self._process.wait(timeout=300); print("[FFmpegWriter] 阶段4/5: FFmpeg 编码完成", flush=True)
            except: print("[FFmpegWriter] FFmpeg 编码超时（>300s），强制终止", flush=True); self._process.kill(); self._process.wait(timeout=10)
        else: print("[FFmpegWriter] 阶段3-4/5: FFmpeg 进程已终止", flush=True)
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
                    print(f'[FFmpegWriter] 合并音轨: {_tmp} + {_src} → {self.output_path}', flush=True)
                    _mux_cmd = [getattr(self.args, 'ffmpeg_bin', 'ffmpeg'), '-y', '-i', _tmp, '-i', _src, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-shortest', self.output_path]
                    try:
                        _r = subprocess.run(_mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
                        if _r.returncode == 0: print(f'[FFmpegWriter] 音轨合并成功', flush=True)
                        else: print(f'[FFmpegWriter] 音轨合并失败 (rc={_r.returncode}): {_r.stderr.decode("utf-8", errors="replace")[-500:]}', flush=True); print(f'[FFmpegWriter] 保留无音轨文件: {_tmp}', flush=True); _tmp = None
                    except Exception as _e: print(f'[FFmpegWriter] 音轨合并异常: {_e}', flush=True); _tmp = None
            elif _src is None:
                # 无音轨模式：直接重命名临时文件
                try:
                    os.rename(_tmp, self.output_path)
                    _tmp = None
                    print(f'[FFmpegWriter] 无音轨模式: 已输出 {self.output_path}', flush=True)
                except Exception as _e:
                    print(f'[FFmpegWriter] rename 失败: {_e}', flush=True)
                    _tmp = None
        if _tmp and (self._broken or self.audio is None) and os.path.exists(_tmp):
            try: os.rename(_tmp, self.output_path); print(f'[FFmpegWriter] 无音轨输出: {self.output_path}', flush=True)
            except Exception as _e: print(f'[FFmpegWriter] rename 失败: {_e}', flush=True)
            _tmp = None
        if _tmp and os.path.exists(_tmp):
            try: os.remove(_tmp)
            except: pass