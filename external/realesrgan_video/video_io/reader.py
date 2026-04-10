#!/usr/bin/env python3
import queue
import threading
import numpy as np
import ffmpeg
import fractions
import os.path as osp
from typing import Optional, Dict, Any

def get_video_meta_info(video_path: str) -> dict:
    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        print(f"❌ ffprobe 失败: {video_path}")
        print(e.stderr.decode('utf-8', errors='ignore'))
        raise
    video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
    has_audio = any(s['codec_type'] == 'audio' for s in probe['streams'])
    vs = video_streams[0]
    fps_str = vs.get('avg_frame_rate', '24/1')
    try: fps = float(fractions.Fraction(fps_str))
    except: fps = 24.0
    if 'nb_frames' in vs and vs['nb_frames'].isdigit(): nb = int(vs['nb_frames'])
    elif 'duration' in vs: nb = int(float(vs['duration']) * fps)
    else: nb = 0
    return {'width': vs['width'], 'height': vs['height'], 'fps': fps, 'audio': ffmpeg.input(video_path).audio if has_audio else None, 'nb_frames': nb}

class FFmpegReader:
    FRAME_TIMEOUT = object()
    def __init__(self, input_path, ffmpeg_bin='ffmpeg', prefetch_factor=16, use_hwaccel=True):
        self.input_path, self.ffmpeg_bin, self.prefetch_factor, self.use_hwaccel = input_path, ffmpeg_bin, prefetch_factor, use_hwaccel
        meta = get_video_meta_info(input_path)
        self.width, self.height, self.fps, self.nb_frames, self.audio = meta['width'], meta['height'], meta['fps'], meta['nb_frames'], meta['audio']
        input_kwargs = {}
        if self.use_hwaccel: input_kwargs['hwaccel'] = 'auto'
        self._ffmpeg_input = ffmpeg.input(input_path, **input_kwargs)
        self._frame_queue = queue.Queue(maxsize=prefetch_factor)
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        try:
            process = self._ffmpeg_input.output('pipe:', format='rawvideo', pix_fmt='rgb24', vsync=0).run_async(pipe_stdout=True, quiet=True)
            frame_size = self.width * self.height * 3
            while self._running:
                in_bytes = process.stdout.read(frame_size)
                if not in_bytes: break
                frame = np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3])
                while self._running:
                    try: self._frame_queue.put(frame, timeout=1.0); break
                    except queue.Full: continue
            while self._running:
                try: self._frame_queue.put(None, timeout=1.0); break
                except queue.Full: continue
            process.wait()
        except Exception as e:
            print(f"FFmpegReader读取错误: {e}")
            try: self._frame_queue.put(None, timeout=1.0)
            except Exception: pass

    def get_frame(self):
        try: return self._frame_queue.get(timeout=2.0)
        except queue.Empty: return FFmpegReader.FRAME_TIMEOUT
    def close(self):
        self._running = False
        if self._thread.is_alive(): self._thread.join(timeout=3.0)