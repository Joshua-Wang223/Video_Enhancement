#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe_Optimisation_batch_size —— Real-ESRGAN 超分 batch_size 探测工具（第一期）

第一期范围（固定条件，不做多变量扫描）：
  · 输入分辨率固定：960x540（动画视频源）
  · 超分模型固定：realesr-animevideov3（SRVGGNetCompact，netscale=4）
  · 只探 batch_size 一个变量

探测方法（粗扫找拐点 → 步进折半精调 → 回归确认）：
  阶段 A 粗扫：batch_size 从 8 起，逐级 +8，直到命中「拐点」：
      - 相对上一级的吞吐增益 < --gain-threshold（默认 2%），或
      - 吞吐不升反降，或
      - GPU 平均利用率已饱和（增量 < --util-gain-threshold，默认 1 个百分点），或
      - 触发 OOM / 显存护栏（峰值显存 > 总显存 × --vram-limit）
  阶段 B 精调：步进折半（4 → 2 → 1），在当前最优点的 ±step 上补测，
      取吞吐更高者移动（若两点都不如当前点则原地不动 → 回归保持）；
      平手（差距 < --min-improve）时依次比 GPU 利用率高 → 显存低 → batch_size 小。
  阶段 C 确认：对最终最优点重测一轮（不复用缓存），输出建议值。

最佳点判据（三者同时记录，共同决定排序）：
  1) 超分处理 FPS      —— 主判据，越高越好（默认取 e2e 口径）
  2) GPU 平均利用率     —— 测量窗口内 NVML 采样均值；已饱和即不再往上探
  3) 处理时长/视频时长  —— realtime factor = 源帧率 / 超分 FPS，越小越好
     （例：0.35 表示 1 秒源视频需 0.35 秒超分；该值与 FPS 反相关，用于直观汇报）

计时口径（两种都测，默认用 e2e 决策）：
  e2e  ：H2D + 前向 + clamp/byte + D2H + 同步，等价 _sr_infer_batch 的真实成本
  pure ：张量常驻显存，只计时前向（H2D/D2H 均已排除），反映算力饱和点

用法：
  # 合成 960x540 动画帧（跨机器可比）
  python Accessory/probe/batch_size_optimizer_probe.py

  # 真实 960x540 动画视频
  python Accessory/probe/batch_size_optimizer_probe.py --input input/anime_960x540.mp4

  # 自定义：粗扫起点 8、步进 8、上限 96、护栏 0.85
  python Accessory/probe/batch_size_optimizer_probe.py --start-bs 8 --coarse-step 8 \
         --max-bs 96 --vram-limit 0.85
"""

import argparse
import fractions
import gc
import json
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# 环境与生产管线保持一致（必须在 import torch 之前设置）
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402  （必须在上面两个 setdefault 之后导入）

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_TESTS_DIR))
for _p in (_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, "external")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540
DEFAULT_MODEL = "realesr-animevideov3"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    """单个 batch_size 的探测结果。"""
    batch_size: int
    status: str = "ok"            # ok | oom | vram_guard | skipped
    fps_e2e: float = 0.0          # 完整批次链路吞吐（帧/秒）——判据 1
    fps_pure: float = 0.0         # 纯前向吞吐（帧/秒）
    gpu_util_avg: float = -1.0    # 测量窗口 GPU 平均利用率(%)，-1=未采到——判据 2
    gpu_util_samples: int = 0
    proc_per_video_sec: float = 0.0  # 处理时长/视频时长（realtime factor）——判据 3
    ms_per_batch_e2e: float = 0.0
    ms_per_batch_pure: float = 0.0
    peak_alloc_gb: float = 0.0    # 峰值 allocated 显存
    peak_reserved_gb: float = 0.0  # 峰值 reserved（含缓存分配器）
    vram_frac: float = 0.0        # peak_alloc / 总显存
    fps_e2e_runs: List[float] = field(default_factory=list)
    fps_pure_runs: List[float] = field(default_factory=list)
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class GPUSampler:
    """GPU 利用率采样器（后台线程定时采样，返回窗口均值）。

    后端优先级：pynvml（NVML 直读，最准）→ torch.cuda.utilization → nvidia-smi。
    注意：这里只用 NVML 的**只读**接口，不会创建 CUDA context，
    因此不存在 pycuda.autoinit 那种与 PyTorch 抢 context 的问题。
    """

    def __init__(self, device_index: int, nvml_index: Optional[int] = None,
                 enabled: bool = True):
        self.enabled = enabled
        self.device_index = device_index
        self.nvml_index = nvml_index if nvml_index is not None else device_index
        self.backend = "disabled"
        self._handle = None
        self._samples: List[float] = []
        self._stop = None
        self._thread = None

        if not enabled:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.nvml_index)
            self._nvml = pynvml
            self.backend = "pynvml"
            return
        except Exception:
            self._handle = None
        try:
            if hasattr(torch.cuda, "utilization"):
                torch.cuda.utilization(device_index)  # 探活，不可用会抛
                self.backend = "torch"
                return
        except Exception:
            pass
        self.backend = "nvidia-smi"  # 兜底：每样本一次进程调用，较粗

    # ---------------- 单点采样 ----------------

    def sample(self) -> Optional[float]:
        if not self.enabled:
            return None
        try:
            if self.backend == "pynvml":
                return float(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
            if self.backend == "torch":
                return float(torch.cuda.utilization(self.device_index))
            if self.backend == "nvidia-smi":
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits", "-i", str(self.nvml_index)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, timeout=5)
                return float(out.stdout.decode().strip().splitlines()[0])
        except Exception:
            return None
        return None

    def idle_baseline(self, seconds: float = 1.0, interval: float = 0.1) -> Optional[float]:
        """测前空转基线：GPU 被别的任务占用时，利用率数据会失真。"""
        vals: List[float] = []
        deadline = time.time() + seconds
        while time.time() < deadline:
            v = self.sample()
            if v is not None:
                vals.append(v)
            time.sleep(interval)
        return float(np.mean(vals)) if vals else None

    # ---------------- 窗口采样 ----------------

    def start(self, interval: float) -> None:
        if not self.enabled or interval <= 0:
            return
        self._samples = []
        self._stop = threading.Event()

        def _loop():
            while not self._stop.is_set():
                v = self.sample()
                if v is not None:
                    self._samples.append(v)
                self._stop.wait(interval)

        self._thread = threading.Thread(target=_loop, name="gpu-util-sampler",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> Tuple[float, int]:
        if self._thread is None:
            return -1.0, 0
        self._stop.set()
        self._thread.join(timeout=5)
        self._thread = None
        if not self._samples:
            return -1.0, 0
        return float(np.mean(self._samples)), len(self._samples)


class FrameSource:
    """探测用帧池：合成动画帧，或来自真实视频的解码帧（循环喂入）。"""

    def __init__(self, frames: List[np.ndarray], origin: str, width: int, height: int):
        self.frames = frames
        self.origin = origin
        self.width = width
        self.height = height

    def take(self, batch_size: int, cursor: int) -> List[np.ndarray]:
        n = len(self.frames)
        return [self.frames[(cursor + i) % n] for i in range(batch_size)]


# ---------------------------------------------------------------------------
# 帧源：合成 / 真实视频
# ---------------------------------------------------------------------------

def make_synthetic_frames(count: int, height: int, width: int, seed: int) -> List[np.ndarray]:
    """生成动画风格合成帧：大面积平滑渐变 + 移动条纹 + 边缘 + 轻微噪点。

    动画内容以平坦色块为主，若用纯噪声会高估访存压力、用纯色块又会低估，
    这里混合三类特征，尽量贴近真实动画帧的统计特性。
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx, ny = xx / width, yy / height

    frames: List[np.ndarray] = []
    for i in range(count):
        t = i / max(count - 1, 1)
        # 平滑渐变背景（动画常见的大面积色块）
        base_r = 110.0 + 90.0 * np.sin(6.0 * nx + t * 6.283)
        base_g = 120.0 + 80.0 * np.sin(4.0 * ny - t * 6.283 + 1.0)
        base_b = 130.0 + 70.0 * np.cos(3.0 * (nx + ny) + t * 6.283)
        # 移动条纹（模拟平移镜头的高频细节）
        stripe = 18.0 * np.sin(40.0 * (nx + 0.35 * t))
        # 硬边缘块（模拟线条稿）
        edge = np.where(((xx // 64).astype(np.int32) + (yy // 48).astype(np.int32) + i) % 7 == 0,
                        45.0, 0.0)
        frame = np.stack([base_r + stripe + edge, base_g + stripe, base_b + stripe * 0.5],
                         axis=-1)
        frame += rng.normal(0.0, 3.0, frame.shape).astype(np.float32)
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return frames


def _ffprobe_video(path: str) -> Tuple[int, int, float]:
    """返回 (width, height, fps)。"""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,avg_frame_rate", "-of", "csv=p=0", path]
    out = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {out.stderr.decode('utf-8', 'ignore').strip()}")
    fields = out.stdout.decode().strip().split(",")
    w, h = int(fields[0]), int(fields[1])
    fps = 0.0
    if len(fields) > 2 and fields[2]:
        try:
            fps = float(fractions.Fraction(fields[2]))
        except (ValueError, ZeroDivisionError):
            fps = 0.0
    return w, h, fps


def load_video_frames(path: str, count: int, width: int,
                      height: int) -> Tuple[List[np.ndarray], float]:
    """用 ffmpeg 解码视频前 count 帧，缩放到目标分辨率后返回 (rgb24 帧列表, 源帧率)。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"输入视频不存在: {path}")
    src_w, src_h, src_fps = _ffprobe_video(path)
    if (src_w, src_h) != (width, height):
        print(f"[帧源] 输入为 {src_w}x{src_h}，第一期固定分辨率，强制缩放到 {width}x{height}")
    frame_bytes = width * height * 3
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-vf", f"scale={width}:{height}:flags=bicubic", "-pix_fmt", "rgb24",
           "-f", "rawvideo", "-"]
    # stdin=DEVNULL：避免 ffmpeg 在后台进程组对 fd0 调 ioctl 被 SIGTTOU 停住
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    frames: List[np.ndarray] = []
    try:
        while len(frames) < count:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frames.append(np.frombuffer(buf, dtype=np.uint8)
                          .reshape(height, width, 3).copy())
    finally:
        if proc.stdout:
            proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "ignore").strip() if proc.stderr else ""
        proc.stderr.close()
        proc.wait(timeout=30)
    if not frames:
        raise RuntimeError(f"未解出任何帧（ffmpeg: {err}）")
    print(f"[帧源] 从 {os.path.basename(path)} 解出 {len(frames)} 帧 @ {width}x{height}"
          f"，源帧率 {src_fps:g} fps（处理时长/视频时长按此折算）")
    return frames, (src_fps if src_fps > 0 else 0.0)


# ---------------------------------------------------------------------------
# 探测器
# ---------------------------------------------------------------------------

class BatchSizeProber:
    def __init__(self, args):
        self.args = args
        self.width = args.width
        self.height = args.height
        self.results: Dict[int, ProbeResult] = {}
        self.failed: set = set()
        self.device = self._setup_device()
        self.total_vram_gb = (torch.cuda.get_device_properties(self.device).total_memory
                              / (1024 ** 3))
        self.upsampler = self._build_upsampler()
        self.model = self.upsampler.model
        self.use_half = bool(self.upsampler.half)
        self.use_tile = int(getattr(self.upsampler, "tile_size", 0) or 0) > 0
        self._tile_forward = None
        if self.use_tile:
            # pipeline 模块会连带 import facexlib（人脸链路），仅切块模式才需要
            try:
                from realesrgan_video.pipeline import _sr_tile_forward
                self._tile_forward = _sr_tile_forward
            except ImportError as exc:  # 不吞上下文
                raise ImportError(
                    f"--tile-size>0 需要 realesrgan_video.pipeline（依赖 facexlib）: {exc}"
                ) from exc
        # 输入/输出各用**一块**共享 pinned staging buffer（随 bs 增大而扩容）。
        # 不按 bs 各自缓存：输出是 4x 超分，bs=96 时单块就 ~2.4GB，逐 bs 缓存会
        # 把主机内存吃光。输入/输出分开是必须的——H2D 与 D2H 是异步的，
        # 共用一块会出现 GPU 还在读、CPU 已经覆写的竞态。
        self._pin_in: Optional[torch.Tensor] = None
        self._pin_out: Optional[torch.Tensor] = None
        # 纯前向用的常驻显存输入，只保留当前 bs 的一份（否则显存会被探测自身
        # 占用，污染显存护栏判定）
        self._gpu_input: Dict[int, torch.Tensor] = {}
        self._cursor = 0
        self.src_fps = float(args.src_fps)          # 源视频帧率，用于 realtime factor
        self.sampler = GPUSampler(args.gpu, args.nvml_index,
                                  enabled=not args.no_util)
        self.source: Optional[FrameSource] = None   # 由 main() 在构造后注入

    # ---------------- 初始化 ----------------

    def _setup_device(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，本探测工具需要 NVIDIA GPU 环境")
        idx = self.args.gpu
        if idx >= torch.cuda.device_count():
            raise RuntimeError(f"--gpu {idx} 越界，可见设备数 {torch.cuda.device_count()}")
        return torch.device("cuda", idx)

    def _build_upsampler(self):
        from realesrgan_video.realesrgan_utils import _build_upsampler
        return _build_upsampler(
            model_name=self.args.model,
            dni_weight=None,
            tile=self.args.tile_size,
            tile_pad=self.args.tile_pad,
            pre_pad=self.args.pre_pad,
            use_half=not self.args.fp32,
            device=self.device,
        )

    # ---------------- 单次批次执行 ----------------

    def _forward(self, x):
        with torch.no_grad():
            if self.use_tile:
                return self._tile_forward(self.upsampler, x)
            return self.model(x)

    def _pinned(self, which: str, numel: int, shape, dtype) -> torch.Tensor:
        """取（必要时扩容）共享 pinned staging buffer 的视图。"""
        attr = "_pin_in" if which == "in" else "_pin_out"
        buf = getattr(self, attr)
        if buf is None or buf.numel() < numel or buf.dtype != dtype:
            try:
                buf = torch.empty(numel, dtype=dtype).pin_memory()
            except RuntimeError:
                # 主机 pinned 内存不足 → 退化为普通页内存（慢一点但不失败）
                print(f"  [warn] pinned 内存分配失败（{numel/1024**3:.2f}GB），"
                      f"{which} 端退化为普通内存")
                buf = torch.empty(numel, dtype=dtype)
            setattr(self, attr, buf)
        return buf[:numel].view(shape)

    def _run_e2e_batch(self, frames: List[np.ndarray]) -> None:
        """完整批次链路：np.stack → pinned → H2D → 前向 → clamp/byte → D2H → 同步。

        与 _sr_infer_batch 同构（单流简化版，无 GFPGAN / TRT 分支）。
        """
        bs = len(frames)
        arr = np.stack(frames, axis=0)                       # (B,H,W,3) uint8
        src = torch.from_numpy(arr)
        pin_in = self._pinned("in", src.numel(), src.shape, torch.uint8)
        pin_in.copy_(src)
        x = pin_in.to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        if self.use_half:
            x = x.half()
        y = self._forward(x)
        out = y.float().clamp_(0.0, 1.0).mul_(255.0).byte().permute(0, 2, 3, 1).contiguous()
        pin_out = self._pinned("out", out.numel(), out.shape, torch.uint8)
        pin_out.copy_(out, non_blocking=True)
        torch.cuda.synchronize(self.device)

    def _run_pure_batch(self, bs: int) -> None:
        """纯前向：输入常驻显存，只跑模型（H2D/D2H 不计入）。"""
        x = self._gpu_input.get(bs)
        if x is None:
            frames = self.source.take(bs, self._cursor)
            arr = np.stack(frames, axis=0)
            x = torch.from_numpy(arr).to(self.device)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            if self.use_half:
                x = x.half()
            x = x.contiguous()
            torch.cuda.synchronize(self.device)
            self._gpu_input[bs] = x
        self._forward(x)

    # ---------------- 一个 batch_size 的完整测量 ----------------

    def measure(self, bs: int, fresh: bool = False) -> ProbeResult:
        """测量指定 batch_size；结果缓存，fresh=True 时强制重测（确认轮用）。"""
        if (not fresh) and bs in self.results:
            return self.results[bs]

        res = ProbeResult(batch_size=bs)
        util_avg, util_n = -1.0, 0
        self._evict_gpu_input(bs)
        try:
            self._warmup(bs)
            torch.cuda.reset_peak_memory_stats(self.device)
            # 利用率采样覆盖整个计时窗口（e2e + pure）
            self.sampler.start(self.args.util_interval)
            try:
                res.fps_e2e_runs, res.fps_pure_runs = self._timed_runs(bs)
            finally:
                util_avg, util_n = self.sampler.stop()
            res.gpu_util_avg = util_avg
            res.gpu_util_samples = util_n
            peak_alloc = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
            res.peak_alloc_gb = peak_alloc / (1024 ** 3)
            res.peak_reserved_gb = peak_reserved / (1024 ** 3)
            res.vram_frac = peak_alloc / (torch.cuda.get_device_properties(self.device).total_memory)

            res.fps_e2e = float(np.median(res.fps_e2e_runs))
            res.fps_pure = float(np.median(res.fps_pure_runs))
            res.ms_per_batch_e2e = 1000.0 * bs / res.fps_e2e if res.fps_e2e > 0 else 0.0
            res.ms_per_batch_pure = 1000.0 * bs / res.fps_pure if res.fps_pure > 0 else 0.0
            # 处理时长 / 视频时长：1 秒源视频所需的超分秒数（越小越好）
            res.proc_per_video_sec = self.src_fps / res.fps_e2e if res.fps_e2e > 0 else 0.0

            if res.vram_frac > self.args.vram_limit:
                res.status = "vram_guard"
                res.note = (f"峰值显存 {res.peak_alloc_gb:.2f}GB "
                            f"({res.vram_frac*100:.1f}%) 超过护栏 "
                            f"{self.args.vram_limit*100:.0f}%")
        except RuntimeError as exc:
            # torch.cuda.OutOfMemoryError 是 RuntimeError 子类，这里统一用
            # 消息匹配，避免旧版本 torch 无该属性时在异常匹配阶段再抛异常
            if "out of memory" not in str(exc).lower():
                raise
            res.status = "oom"
            res.note = str(exc).splitlines()[0][:160]
            self._release(bs)
        finally:
            torch.cuda.empty_cache()

        if res.status != "ok":
            self.failed.add(bs)
        if fresh:
            self.results[bs] = res      # 覆盖缓存，保留确认轮数据
        else:
            self.results.setdefault(bs, res)
        return res

    def _warmup(self, bs: int) -> None:
        for _ in range(self.args.warmup):
            self._run_e2e_batch(self.source.take(bs, self._next_cursor(bs)))
        for _ in range(self.args.warmup):
            self._run_pure_batch(bs)
        torch.cuda.synchronize(self.device)

    def _timed_runs(self, bs: int) -> Tuple[List[float], List[float]]:
        e2e_runs: List[float] = []
        pure_runs: List[float] = []
        for _ in range(self.args.repeats):
            # --- e2e：整批墙钟时间（含 H2D/前向/D2H/同步）---
            t0 = time.perf_counter()
            for _ in range(self.args.batches_per_repeat):
                self._run_e2e_batch(self.source.take(bs, self._next_cursor(bs)))
            dt = time.perf_counter() - t0
            n = bs * self.args.batches_per_repeat
            e2e_runs.append(n / dt if dt > 0 else 0.0)

            # --- pure：连续 K 次前向的墙钟时间（末尾一次同步）---
            torch.cuda.synchronize(self.device)
            t0 = time.perf_counter()
            for _ in range(self.args.batches_per_repeat):
                self._run_pure_batch(bs)
            torch.cuda.synchronize(self.device)
            dt = time.perf_counter() - t0
            pure_runs.append(n / dt if dt > 0 else 0.0)
        return e2e_runs, pure_runs

    def _next_cursor(self, bs: int) -> int:
        cur = self._cursor
        self._cursor = (cur + bs) % max(len(self.source.frames), 1)
        return cur

    def _evict_gpu_input(self, bs: int) -> None:
        """只保留当前 batch_size 的常驻显存输入，避免探测自身占用干扰显存护栏。"""
        if self._gpu_input and bs not in self._gpu_input:
            self._gpu_input.clear()
            gc.collect()
            torch.cuda.empty_cache()

    def _release(self, bs: int) -> None:
        """OOM 后清理该 batch_size 占用的常驻/缓存显存。"""
        self._gpu_input.pop(bs, None)
        gc.collect()
        try:
            torch.cuda.synchronize(self.device)
        except RuntimeError:
            pass
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 搜索：粗扫找拐点 → 步进折半精调 + 回归
# ---------------------------------------------------------------------------

def metric_of(res: "ProbeResult", decide_by: str) -> float:
    """主判据：超分处理 FPS。"""
    return res.fps_e2e if decide_by == "e2e" else res.fps_pure


def pick_best(prober: BatchSizeProber, pool: List[int], args) -> int:
    """在候选集中挑最优点。

    判据优先级：
      1) FPS 最高（主判据，处理时长/视频时长与之反相关，不重复计权）
      2) FPS 差距在 --min-improve 内视为平手 → GPU 平均利用率更高者
      3) 仍平手 → 峰值显存更低者 → batch_size 更小者
    """
    def fps(b: int) -> float:
        return metric_of(prober.results[b], args.decide_by)

    top = max(fps(b) for b in pool)
    tied = [b for b in pool if fps(b) >= top * (1.0 - args.min_improve)]
    return max(tied, key=lambda b: (prober.results[b].gpu_util_avg,
                                    -prober.results[b].peak_alloc_gb, -b))


def coarse_scan(prober: BatchSizeProber, args) -> Tuple[int, List[int]]:
    """阶段 A：从 start-bs 起逐级 +coarse-step，直到命中拐点/护栏/上限。

    拐点判据（任一命中即停）：
      · FPS 相对增益 < --gain-threshold（含负增益）
      · GPU 平均利用率增量 < --util-gain-threshold（GPU 已喂饱，再加 batch 只堆显存）
      · OOM / 显存护栏

    返回 (拐点 batch_size, 已测序列)。
    """
    print(f"\n=== 阶段 A 粗扫：{args.start_bs} 起，步进 +{args.coarse_step}，"
          f"FPS 增益阈值 {args.gain_threshold*100:.1f}%，"
          f"利用率增量阈值 {args.util_gain_threshold:.1f}pp，"
          f"护栏 {args.vram_limit*100:.0f}% 显存 ===")
    prev: Optional[ProbeResult] = None
    bs = args.start_bs
    while bs <= args.max_bs:
        res = prober.measure(bs)
        _print_row(res, args.decide_by, prober.total_vram_gb)
        if not res.ok:
            print(f"   ↳ 命中{_status_cn(res.status)}，粗扫停止，拐点回退到 "
                  f"{prev.batch_size if prev else args.start_bs}")
            return (prev.batch_size if prev else args.start_bs), sorted(prober.results)
        if prev is not None:
            gain = (metric_of(res, args.decide_by) - metric_of(prev, args.decide_by)) / \
                max(metric_of(prev, args.decide_by), 1e-9)
            print(f"   ↳ 相对 bs={prev.batch_size}：FPS 增益 {gain*100:+.2f}%")
            if gain < args.gain_threshold:
                print(f"   ↳ 低于 FPS 增益阈值 → 拐点出现在 bs={bs} 附近，粗扫停止")
                return bs, sorted(prober.results)
            if res.gpu_util_avg >= 0 and prev.gpu_util_avg >= 0:
                du = res.gpu_util_avg - prev.gpu_util_avg
                print(f"   ↳ 相对 bs={prev.batch_size}：GPU 利用率 "
                      f"{prev.gpu_util_avg:.1f}% → {res.gpu_util_avg:.1f}% ({du:+.1f}pp)")
                if du < args.util_gain_threshold:
                    print("   ↳ GPU 利用率已饱和 → 拐点出现，粗扫停止")
                    return bs, sorted(prober.results)
        prev = res
        bs += args.coarse_step
    print(f"   ↳ 已到 --max-bs {args.max_bs}，粗扫结束")
    return prev.batch_size, sorted(prober.results)


def refine(prober: BatchSizeProber, args, knee: int) -> int:
    """阶段 B：步进折半（4→2→1），在 ±step 上补测并移动；更优则进，否则回归保持。"""
    lo = max(1, knee - args.coarse_step)
    hi = min(args.max_bs, knee + args.coarse_step)
    # 护栏/OOM 之上的点不再探
    for bad in sorted(prober.failed):
        if bad <= knee:
            lo = max(lo, bad + 1)
        else:
            hi = min(hi, bad - 1)

    candidates = [b for b in prober.results
                  if prober.results[b].ok and lo <= b <= hi]
    if not candidates:
        return knee
    best = pick_best(prober, candidates, args)
    print(f"\n=== 阶段 B 精调：区间 [{lo}, {hi}]，起点 bs={best} ===")

    step = max(args.coarse_step // 2, 1)
    while step >= 1:
        print(f"--- 步进 {step} ---")
        for _ in range(args.max_moves):
            cands = [c for c in (best - step, best + step)
                     if lo <= c <= hi and c not in prober.failed]
            for c in cands:
                res = prober.measure(c)
                _print_row(res, args.decide_by, prober.total_vram_gb)
            pool = [best] + [c for c in cands if prober.results[c].ok]
            cur_fps = metric_of(prober.results[best], args.decide_by)
            nxt = pick_best(prober, pool, args)
            nxt_fps = metric_of(prober.results[nxt], args.decide_by)
            if nxt == best or nxt_fps <= cur_fps * (1.0 + args.min_improve):
                print(f"   ↳ 最优保持 bs={best}（{cur_fps:.1f} fps，"
                      f"GPU {prober.results[best].gpu_util_avg:.1f}%）")
                break
            print(f"   ↳ 移动到 bs={nxt}（{cur_fps:.1f} → {nxt_fps:.1f} fps，"
                  f"GPU {prober.results[nxt].gpu_util_avg:.1f}%）")
            best = nxt
        step //= 2
    return best


def _status_cn(status: str) -> str:
    return {"ok": "正常", "oom": "OOM", "vram_guard": "显存护栏",
            "skipped": "跳过"}.get(status, status)


def _util_text(v: float) -> str:
    return "  n/a" if v < 0 else f"{v:5.1f}"


def _print_row(res: ProbeResult, decide_by: str, total_vram_gb: float) -> None:
    flag = "" if res.ok else f"  [{_status_cn(res.status)}]"
    mark = "*" if decide_by == "e2e" else " "
    print(f"  bs={res.batch_size:>3} | e2e {res.fps_e2e:>7.1f} fps{mark} "
          f"({res.ms_per_batch_e2e:>6.1f} ms/批) | pure {res.fps_pure:>7.1f} fps "
          f"| GPU {_util_text(res.gpu_util_avg)}% "
          f"| 处理/视频 {res.proc_per_video_sec:>5.2f}x "
          f"| 峰值 {res.peak_alloc_gb:>5.2f}GB "
          f"({res.vram_frac*100:>4.1f}% / {total_vram_gb:.0f}GB){flag}")
    if res.note:
        print(f"        note: {res.note}")


def print_summary(prober: BatchSizeProber, args, best: int, elapsed: float) -> None:
    print("\n=== 全部测点 ===")
    print("  bs  |  e2e fps | pure fps | GPU 利用率 | 处理/视频 | 峰值显存      | 状态")
    for bs in sorted(prober.results):
        r = prober.results[bs]
        print(f"  {bs:>3} | {r.fps_e2e:>8.1f} | {r.fps_pure:>8.1f} | "
              f"{_util_text(r.gpu_util_avg):>8}% | {r.proc_per_video_sec:>7.2f}x | "
              f"{r.peak_alloc_gb:>6.2f}GB {r.vram_frac*100:>5.1f}% | {_status_cn(r.status)}")

    base = prober.results.get(args.start_bs)
    b = prober.results[best]
    print("\n=== 结论（第一期：960x540 动画源 + realesr-animevideov3）===")
    print(f"  最佳 batch_size      : {best}")
    print(f"  ① 超分处理 FPS       : {b.fps_e2e:.1f} fps（e2e，{b.ms_per_batch_e2e:.1f} ms/批）"
          f" / {b.fps_pure:.1f} fps（pure）")
    print(f"  ② GPU 平均利用率     : {_util_text(b.gpu_util_avg).strip()}%"
          f"（{b.gpu_util_samples} 个采样，后端 {prober.sampler.backend}）")
    print(f"  ③ 处理时长/视频时长  : {b.proc_per_video_sec:.3f}x"
          f"（源 {prober.src_fps:g}fps；1 秒源视频需 {b.proc_per_video_sec:.3f} 秒超分）")
    print(f"  峰值显存             : {b.peak_alloc_gb:.2f}GB "
          f"（{b.vram_frac*100:.1f}% / {prober.total_vram_gb:.0f}GB）")
    if base is not None and base.ok and best != base.batch_size:
        print(f"  相对 bs={base.batch_size} 提升      : "
              f"{(b.fps_e2e / base.fps_e2e - 1) * 100:+.1f}% (e2e) / "
              f"{(b.fps_pure / base.fps_pure - 1) * 100:+.1f}% (pure)，"
              f"处理时长 {base.proc_per_video_sec:.3f}x → {b.proc_per_video_sec:.3f}x")
    print(f"  耗时                 : {elapsed:.1f}s")
    print("\n  建议落地：config/default_config.json → realesrgan.batch_size = "
          f"{best}（max_batch_size 建议 ≥ {best}）")
    if best > 48:
        print("  ⚠️  生产管线 adaptive 上限 _max_adaptive_batch = min(batch_size*2, 48)，"
              f"若要真正用上 bs={best} 需同步放宽该上限（external/realesrgan_video/pipeline.py）")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Probe_Optimisation_batch_size —— Real-ESRGAN batch_size 探测（第一期）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default=None,
                   help="真实动画视频；缺省使用 960x540 合成帧（跨机器可比）")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="超分模型名（MODEL_CONFIG 中的 key）")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="输入宽")
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="输入高")
    p.add_argument("--frames", type=int, default=64, help="帧池大小（循环喂入）")
    p.add_argument("--seed", type=int, default=20260926, help="合成帧随机种子")
    p.add_argument("--src-fps", type=float, default=24.0,
                   help="源视频帧率（用于「处理时长/视频时长」；--input 时自动取 ffprobe 值）")

    p.add_argument("--start-bs", type=int, default=8, help="粗扫起始 batch_size")
    p.add_argument("--coarse-step", type=int, default=8, help="粗扫步进")
    p.add_argument("--max-bs", type=int, default=96, help="batch_size 上限")
    p.add_argument("--gain-threshold", type=float, default=0.02,
                   help="粗扫拐点判据：相对增益低于此值即停止")
    p.add_argument("--min-improve", type=float, default=0.005,
                   help="精调移动判据：新点需超过当前最优至少该比例")
    p.add_argument("--max-moves", type=int, default=3, help="每个步进下最多移动次数")
    p.add_argument("--util-gain-threshold", type=float, default=1.0,
                   help="粗扫拐点判据：GPU 利用率增量低于该百分点即视为饱和")
    p.add_argument("--vram-limit", type=float, default=0.85,
                   help="显存护栏：峰值显存 / 总显存 超过即判定越界")

    p.add_argument("--warmup", type=int, default=2, help="每个测点的预热批数")
    p.add_argument("--repeats", type=int, default=3, help="计时轮数（取中位数抗噪）")
    p.add_argument("--batches-per-repeat", type=int, default=3, help="每轮计时的批数")
    p.add_argument("--decide-by", choices=["e2e", "pure"], default="e2e",
                   help="用哪种口径的吞吐做决策")

    p.add_argument("--fp32", action="store_true", help="使用 FP32（默认 FP16）")
    p.add_argument("--tile-size", type=int, default=0, help="tile 切块大小（0=不切块）")
    p.add_argument("--tile-pad", type=int, default=10, help="tile 重叠填充")
    p.add_argument("--pre-pad", type=int, default=0, help="前置反射填充")
    p.add_argument("--gpu", type=int, default=0, help="GPU 设备序号")
    p.add_argument("--nvml-index", type=int, default=None,
                   help="NVML 设备序号（CUDA_VISIBLE_DEVICES 重排时手工指定）")
    p.add_argument("--util-interval", type=float, default=0.1,
                   help="GPU 利用率采样间隔（秒）")
    p.add_argument("--no-util", action="store_true", help="不采集 GPU 利用率（判据退化为 ① ③）")
    p.add_argument("--report", default="", help="JSON 报告路径（缺省自动命名到 benchmark_output/）")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.input is None and (args.width, args.height) != (DEFAULT_WIDTH, DEFAULT_HEIGHT):
        print(f"[提示] 第一期固定 {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}，"
              f"当前 --width/--height = {args.width}x{args.height}")
    if args.max_bs < args.start_bs:
        print("[错误] --max-bs 必须 >= --start-bs")
        return 2

    print("Probe_Optimisation_batch_size —— Real-ESRGAN batch_size 探测（第一期）")
    print(f"  模型      : {args.model}   FP{'32' if args.fp32 else '16'}"
          f"   tile={args.tile_size}")
    print(f"  输入分辨率: {args.width}x{args.height}")
    print("  最佳点判据 : ① 超分 FPS（主） ② GPU 平均利用率 ③ 处理时长/视频时长")

    if args.input:
        frames, src_fps = load_video_frames(args.input, args.frames,
                                            args.width, args.height)
        if src_fps > 0:
            args.src_fps = src_fps
        origin = f"video:{args.input}"
    else:
        frames = make_synthetic_frames(args.frames, args.height, args.width, args.seed)
        origin = f"synthetic(seed={args.seed})"
        print(f"[帧源] 合成 {len(frames)} 帧 @ {args.width}x{args.height}（{origin}）")

    t_start = time.time()
    prober = BatchSizeProber(args)
    prober.source = FrameSource(frames, origin, args.width, args.height)
    print(f"[环境] GPU {torch.cuda.get_device_name(prober.device)}，"
          f"显存 {prober.total_vram_gb:.1f}GB，"
          f"CUDA {torch.version.cuda}，torch {torch.__version__}")

    # 共享 GPU 主机上，别的任务会把利用率顶上去 —— 先量一次空转基线再决定可信度
    idle = prober.sampler.idle_baseline()
    print(f"[环境] 利用率采样后端: {prober.sampler.backend}，"
          f"探测前空转基线: {'n/a' if idle is None else f'{idle:.1f}%'}")
    if idle is not None and idle > 5.0:
        print("  ⚠️  空转基线 >5%，疑似其它任务正在占用该 GPU，"
              "利用率与 FPS 都会失真，建议空闲时重跑")

    knee, _ = coarse_scan(prober, args)
    best = refine(prober, args, knee)

    print(f"\n=== 阶段 C 确认：bs={best} 重测一轮（不复用缓存）===")
    confirmed = prober.measure(best, fresh=True)
    _print_row(confirmed, args.decide_by, prober.total_vram_gb)
    elapsed = time.time() - t_start
    print_summary(prober, args, best, elapsed)

    _write_report(prober, args, best, origin, elapsed)
    return 0


def _write_report(prober: BatchSizeProber, args, best: int, origin: str,
                  elapsed: float) -> None:
    if args.report:
        path = args.report
    else:
        out_dir = os.path.join(_PROJECT_ROOT, "benchmark_output")
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"probe_batch_size_{args.model}_"
                                     f"{args.width}x{args.height}_{stamp}.json")
    payload = {
        "probe": "Probe_Optimisation_batch_size",
        "phase": 1,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "criteria": {
            "1_fps": "超分处理 FPS（主判据，默认 e2e 口径）",
            "2_gpu_util": "测量窗口内 GPU 平均利用率（%，pynvml 采样）",
            "3_proc_per_video": "处理时长/视频时长（源帧率 / 超分 FPS，越小越好）",
        },
        "config": {
            "model": args.model, "fp16": not args.fp32,
            "width": args.width, "height": args.height,
            "frame_source": origin, "frame_pool": len(prober.source.frames),
            "src_fps": prober.src_fps,
            "start_bs": args.start_bs, "coarse_step": args.coarse_step,
            "max_bs": args.max_bs, "gain_threshold": args.gain_threshold,
            "util_gain_threshold": args.util_gain_threshold,
            "min_improve": args.min_improve, "vram_limit": args.vram_limit,
            "warmup": args.warmup, "repeats": args.repeats,
            "batches_per_repeat": args.batches_per_repeat,
            "decide_by": args.decide_by, "tile_size": args.tile_size,
        },
        "env": {
            "gpu": torch.cuda.get_device_name(prober.device),
            "total_vram_gb": round(prober.total_vram_gb, 2),
            "util_backend": prober.sampler.backend,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
        "recommended_batch_size": best,
        "elapsed_sec": round(elapsed, 1),
        "measurements": [asdict(prober.results[b]) for b in sorted(prober.results)],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\n[报告] {path}")


if __name__ == "__main__":
    sys.exit(main())
