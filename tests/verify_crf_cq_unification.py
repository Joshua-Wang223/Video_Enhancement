#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_crf_cq_unification.py — 编码质量参数（CRF/CQ）统一优化 · Linux/GPU 实测验证
================================================================================

验证对象
--------
本次会话对 Video_Enhancement 的 CRF/CQ 统一化改造，方案见：
  · Plan/Video_Enhancement 视频编码转换统一优化_Prompt.md
  · Plan/Video_Enhancement_crf_cq统一优化_对比分析报告.md

核心待验证命题
--------------
1. **量纲换算正确**：libx264 CRF 与 NVENC CQ / CONSTQP QP 是三套不同刻度，
   `src/utils/quality_map.py` 的换算是否把同一"画质基准"正确落到各编码器，
   且 CONSTQP 轴（`-qp`）与 CQ 轴（`-cq:v`）不混淆。
2. **下发路径一致**：IFRNet / Real-ESRGAN 两侧、SDK Level 1 直通与 FFmpeg CLI
   两条路径，在同一 `rate_mode` 下发出的参数是否等价、无静默丢弃。
3. **avgBitRate 天花板（分辨率自适应：≤1080p 50 Mbps / >1080p 100 Mbps）合理**：
   该钳制是否只在极端高码率需求下生效，是否对画面质量造成明显损失。
4. **输入量程校验严格**（`[P0-FIX-QUALITY-RANGE]`）：所有质量输入（字面量 crf/cq
   与基准 crf_ref/cq_ref、CLI 与 JSON 两条来源）都必须按**技术规范定义的实际可用
   范围**校验；超限必须给出明确可读错误并拒绝执行，不得静默放行或自动截断。
   字面量量程随生效编码器而定（libx264 0~51、libvpx-vp9 0~63、h264_qsv 1~51 …）。

分组与依赖
----------
  G0 前置        PREREQ   环境探测（Python/ffmpeg/ffprobe/NVENC/模型无关）
  G1 换算表      TABLE    纯逻辑，无需 GPU/ffmpeg（含 literal_range 量程表）
  G2 解析顺序    RESOLVE  纯逻辑，无需 GPU/ffmpeg
  G3 CONSTQP 轴  CONSTQP  纯逻辑，无需 GPU/ffmpeg
  G4 CLI 契约    CLI      进程内导入主模块，无需 GPU
                          （含 [P0-FIX-QUALITY-RANGE] 严格量程校验：所有质量
                           输入按编码器技术规范量程判定，超限明确报错退出，
                           不静默放行、不自动截断）
  G5 代码落点    STATIC   源码结构断言（标签即契约），无需 GPU
  G6 命令下发    EMIT     尽力而为：导入后端并捕获构建出的 ffmpeg 命令
  G7 画质统一性  QUALITY  需 GPU(NVENC) + ffmpeg（可用则 VMAF，否则 PSNR/SSIM）
  G8 码率天花板  BITRATE  需 GPU(NVENC) + ffmpeg
  G9 范围外闭环  SCOPE    原 G9 的 5 项遗留项逐项闭环核对（保留编号便于与历史报告对照）
  G10 环节①/③契约 PIPE    合并/归一化 codec-aware、copy-by-default、CLI 与量程、
                           旧版辅助 API（add_audio/merge_videos/encode_video）统一换算

运行方式
--------
    # 生产环境（GPU 机器）全量实测，输出 Markdown + JSON 报告
    python tests/verify_crf_cq_unification.py --gpu

    # 只跑纯逻辑 / CLI / 静态核验（CPU 即可，秒级）
    python tests/verify_crf_cq_unification.py --quick

    # 用真实素材替代合成素材（更贴近生产画质结论）
    python tests/verify_crf_cq_unification.py --gpu \
        --source /data/clip_720p.mp4 --bitrate-source /data/clip_1080p60.mp4

    # G8-4H 默认用合成 4K60 验证 >1080p 高档（100M）天花板是否真正绑定；
    # 可调尺寸/帧率，或用 --no-br-high 关闭以省去 4K 编码/度量开销
    python tests/verify_crf_cq_unification.py --gpu \
        --br-high-w 2560 --br-high-h 1440 --br-high-fps 120
    python tests/verify_crf_cq_unification.py --gpu --no-br-high

    # 自定义报告路径
    python tests/verify_crf_cq_unification.py --gpu \
        --report verification_report/CRF_CQ统一验证报告_$(date +%Y%m%d).md \
        --json   verification_report/CRF_CQ统一验证结果_$(date +%Y%m%d).json

    # 并行度：默认 0=自动探测系统资源（CPU 物理核 / 可用内存 / GPU 会话数）
    python tests/verify_crf_cq_unification.py --gpu --jobs 8
    python tests/verify_crf_cq_unification.py --gpu --jobs 1   # 强制串行，复现旧行为

并行说明
--------
  · G7 的 6 次编码与全部 PSNR/SSIM/VMAF 度量、G8 的两组度量均并行执行，
    复用 src/utils/parallel_executor.py + system_resources.py 自动定档：
    workers=None 时按 CPU 物理核数与可用内存自动计算，GPU 编码另有会话闸门
    （容量 = 本机 NVENC 会话上限），避免并发撞 OpenEncodeSession。
  · 单任务内存预留按**素材分辨率**推导（_task_ram_mb，基于实测 peak RSS）：
    ≤720p 192MB / 1080p 640MB / 1440p 1024MB / 4K 2048MB。固定值要么在小
    分辨率下白白掐住并发，要么在 4K 下超售内存。
  · G8 的两次编码刻意保持串行计时：G8-5「钳制不拖慢编码」依赖独占计时。
  · `--jobs 1` 走串行分支，结果与并行路径逐项可比，便于定位问题。

退出码
------
    0 = 无 FAIL（PASS / WARN / SKIP 均可接受）
    1 = 存在 FAIL
    2 = 脚本自身异常终止

维护约定
--------
  · 断言锚定"结构性证据"（源码正则 / 真实命令 / 真实编码指标），禁止只匹配注释文案。
  · 新增修复项：在对应分组追加一条 Check；标签 [*-FIX-*] 是代码与脚本的契约，
    改名须两侧同步。
  · 质量类断言一律采用"相对改进 + 绝对对齐"双判据：相比修复前的朴素下发是否
    更接近软编基准（相对），以及差距是否落在容忍带内（绝对）。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# =============================================================================
# 路径与常量
# =============================================================================

TESTS_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT   = TESTS_DIR.parent
SRC_DIR        = PROJECT_ROOT / "src"
UTILS_DIR      = SRC_DIR / "utils"
PROC_DIR       = SRC_DIR / "processors"
EXTERNAL_DIR   = PROJECT_ROOT / "external"
CONFIG_JSON    = PROJECT_ROOT / "config" / "default_config.json"
MAIN_CLI       = SRC_DIR / "main_video_optimized.py"

IFRNET_PKG     = EXTERNAL_DIR / "ifrnet_video"
ESRGAN_PKG     = EXTERNAL_DIR / "realesrgan_video"
QUALITY_MAP_PY = UTILS_DIR / "quality_map.py"
CONVERT_CRF_PY = UTILS_DIR / "convert_crf.py"

# 本次会话改动过的活跃文件（用于 py_compile 编译核验）
CHANGED_FILES = [
    "src/utils/quality_map.py",
    "src/utils/video_utils.py",
    "src/main_video_optimized.py",
    "src/utils/config_manager.py",
    "config/default_config.json",
    "src/processors/ifrnet_processor_video_optimized.py",
    "src/processors/realesrgan_processor_video_optimized.py",
    "external/ifrnet_video/main.py",
    "external/ifrnet_video/ffmpeg_io.py",
    "external/ifrnet_video/nvenc_sdk.py",
    "external/realesrgan_video/main.py",
    "external/realesrgan_video/ffmpeg_io.py",
    "external/realesrgan_video/nvenc_sdk.py",
]

# 基准轴参考值（报告 §4.3 的统一默认）：libx264 CRF 21 的等效映射
REF21_EXPECTED: Dict[str, Tuple[str, int]] = {
    "libx264":    ("-crf", 21),
    "libx265":    ("-crf", 24),
    "h264_nvenc": ("-cq:v", 26),
    "hevc_nvenc": ("-cq:v", 28),
    "av1_nvenc":  ("-cq:v", 27),
    "libsvtav1":  ("-crf", 27),
    "libvpx-vp9": ("-crf", 27),
    "librav1e":   ("-qp", 80),
}

# avgBitRate 天花板（两侧 nvenc_sdk.py 必须一致）
# [FIX-BR-RES-ADAPT] 分辨率自适应：≤1080p 用 50 Mbps 基线，>1080p 用 100 Mbps 高档
BR_CEILING      = 50_000_000
BR_CEILING_HIGH = 100_000_000
BR_FLOOR        = 5_000_000
BR_ADAPT_PIXELS = 1920 * 1080

# 画质对齐容忍带（相对软编基准）
TOL_PSNR_DB   = 1.5     # 换算值相对软编基准的**单向下探** ≤ 1.5 dB 视为对齐
TOL_PSNR_WARN = 3.0     # 单向下探 ≤ 3.0 dB 记为 WARN（超出则 FAIL）
TOL_SSIM      = 0.010   # |ΔSSIM| ≤ 0.010 视为对齐
TOL_VMAF      = 2.5     # |ΔVMAF| ≤ 2.5（若可用）视为对齐
# [FIX-VMAF-LONGFORM] VMAF 采样帧数上限。libvmaf 极慢，长视频（数万帧）整段打分
# 极易超过单命令超时而返回 None（082057 报告 G7 组执行中断的根因）。截断到前 N 帧
# 既能让长视频跑完，又不影响短合成素材（n < N 时等于整段）。
VMAF_MAX_FRAMES = 300
# 码率保真度带（换算值码率 / 软编基准码率）。固定线性映射允许一定偏离，
# 但必须显著优于"朴素下发"（后者常见 1.7×+ 的码率暴涨）。
RATE_PASS = (0.65, 1.50)
RATE_WARN = (0.55, 1.65)
# 天花板影响容忍带（钳制后相对未钳制）
TOL_PSNR_LOSS = 0.5     # 钳制导致 PSNR 下降 ≥0.5 dB 视为"有损失"
TOL_SSIM_LOSS = 0.002

# [FIX-BR-HIGH-BIND] >1080p（100M 档）绑定验证默认合成素材。
# 经验值 ~0.338 bit/pixel/frame（1080p60 CQ26 ≈42Mbps），据此估算：
#   1440p60 ≈ 75Mbps（<100M，永远绑不定）  2160p60 ≈ 168Mbps（绑定，余量最大）。
# 故用 4K60、时长压到 1.0s（60 帧）以约束编码/度量开销。
BR_HIGH_W, BR_HIGH_H, BR_HIGH_FPS, BR_HIGH_DUR = 3840, 2160, 60, 1.0


# =============================================================================
# 结果模型
# =============================================================================

class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


_ICON = {Status.PASS: "✅", Status.FAIL: "❌", Status.WARN: "⚠️ ", Status.SKIP: "⏭️ "}


@dataclass
class Check:
    cid: str
    group: str
    title: str
    status: Status = Status.SKIP
    detail: str = ""
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.cid,
            "group": self.group,
            "title": self.title,
            "status": self.status.value,
            "detail": self.detail,
            "evidence": self.evidence,
        }


class Verifier:
    """收集检查项；支持把异常兜底成 FAIL，保证单点故障不中断整体验证。"""

    def __init__(self) -> None:
        self.checks: List[Check] = []

    def add(self, cid: str, group: str, title: str, status: Status,
            detail: str = "", evidence: Optional[Sequence[str]] = None) -> Check:
        c = Check(cid, group, title, status, detail, list(evidence or []))
        self.checks.append(c)
        print(f"  {_ICON[status]} [{cid}] {title}"
              + (f" — {detail}" if detail else ""))
        for e in c.evidence[:_EVIDENCE_PRINT_LIMIT]:
            print(f"        · {e}")
        if len(c.evidence) > _EVIDENCE_PRINT_LIMIT:
            print(f"        · …（另有 {len(c.evidence) - _EVIDENCE_PRINT_LIMIT} 条，见报告）")
        return c

    def guard(self, cid: str, group: str, title: str, fn: Callable[[], Check]) -> Check:
        """执行 fn；异常时记为 FAIL（避免一个检查崩溃导致整轮无结论）。"""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            import traceback
            return self.add(cid, group, title, Status.FAIL,
                            detail=f"执行异常：{type(exc).__name__}: {exc}",
                            evidence=[traceback.format_exc().splitlines()[-1]])

    def counts(self) -> Dict[str, int]:
        out = {s.value: 0 for s in Status}
        for c in self.checks:
            out[c.status.value] += 1
        return out


_EVIDENCE_PRINT_LIMIT = 6


# =============================================================================
# 通用工具：子进程 / 媒体 / 指标
# =============================================================================

class Ctx:
    """运行上下文：缓存环境探测结果，避免重复调用外部命令。"""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tmp: Optional[Path] = None
        self.ffmpeg = shutil.which("ffmpeg")
        self.ffprobe = shutil.which("ffprobe")
        self.encoders: Optional[set] = None
        self.filters: Optional[set] = None
        self._nvenc_ok: Optional[bool] = None
        self._torch: Optional[bool] = None
        self._gpu_ok: Optional[bool] = None
        self.quality_map = None
        self.main_mod = None
        self.metrics: Dict[str, dict] = {}
        self._executors: Dict[str, Any] = {}

    # ── 外部命令 ─────────────────────────────────────────────────────────────
    def run(self, cmd: Sequence[str], timeout: Optional[int] = None,
            ) -> Tuple[int, str, str]:
        timeout = timeout or self.args.timeout
        try:
            p = subprocess.run(list(cmd), capture_output=True, text=True,
                               timeout=timeout, encoding="utf-8", errors="ignore")
            return p.returncode, p.stdout or "", p.stderr or ""
        except subprocess.TimeoutExpired:
            return 124, "", f"timeout after {timeout}s"
        except FileNotFoundError as exc:
            return 127, "", str(exc)

    def ffmpeg_encoders(self) -> set:
        if self.encoders is None:
            self.encoders = set()
            if self.ffmpeg:
                rc, out, err = self.run([self.ffmpeg, "-hide_banner", "-encoders"], 60)
                if rc == 0:
                    for line in (out + err).splitlines():
                        m = re.match(r"\s*[VAS][\w.]*\s+(\S+)", line)
                        if m:
                            self.encoders.add(m.group(1))
        return self.encoders

    def ffmpeg_filters(self) -> set:
        if self.filters is None:
            self.filters = set()
            if self.ffmpeg:
                rc, out, err = self.run([self.ffmpeg, "-hide_banner", "-filters"], 60)
                if rc == 0:
                    for line in (out + err).splitlines():
                        parts = line.split()
                        if len(parts) >= 2:
                            self.filters.add(parts[1])
        return self.filters

    # ── 并行执行 ─────────────────────────────────────────────────────────────
    def executor(self, gpu: bool, task_ram_mb: int):
        """惰性构建项目统一的并行执行引擎（src/utils/parallel_executor.py）。

        · 复用既有基础设施而非自造线程池：workers=None 时引擎内部调用
          compute_auto_workers()，按 CPU 物理核数 / 可用内存自动定档并封顶；
          task_ram_mb 决定"内存"这一档的松紧，由 _task_ram_mb() 按源分辨率推导。
        · gpu=True 时额外挂 GPU 闸门（get_gpu_semaphore），容量取该机所有 GPU 的
          NVDEC/NVENC 会话数上限（消费级卡通常为 2），避免并发编码撞
          OpenEncodeSession 失败。
        · 线程模式：本脚本的负载全是 subprocess.run（等待期释放 GIL），无需
          process 模式，也就不受 Ctx/模块不可 pickle 的限制。
        · 缓存按 (用途, 预留量) 区分：分辨率变化时不复用旧档位。
        """
        for p in (str(UTILS_DIR), str(SRC_DIR)):
            if p not in sys.path:
                sys.path.insert(0, p)
        import importlib
        importlib.import_module("system_resources")
        pe = importlib.import_module("parallel_executor")
        key = ("gpu" if gpu else "cpu", int(task_ram_mb))
        ex = self._executors.get(key)
        if ex is None:
            jobs = int(getattr(self.args, "jobs", 0) or 0)
            ex = pe.ParallelExecutor(
                workers=(jobs if jobs > 1 else None),
                parallel_mode="thread",
                gpu_task=gpu,
                task_ram_mb=int(task_ram_mb),
            )
            self._executors[key] = ex
        return ex

    # ── 能力探测 ─────────────────────────────────────────────────────────────
    def torch_available(self) -> bool:
        if self._torch is None:
            try:
                import torch  # noqa: F401
                self._torch = True
            except Exception:  # noqa: BLE001
                self._torch = False
        return self._torch

    def nvenc_ok(self) -> bool:
        """真机试编一帧，确认 NVENC 可用（比只看 encoder 列表可靠）。"""
        if self._nvenc_ok is None:
            self._nvenc_ok = False
            if self.ffmpeg and "h264_nvenc" in self.ffmpeg_encoders():
                rc, _, _ = self.run(
                    [self.ffmpeg, "-hide_banner", "-v", "error",
                     "-f", "lavfi", "-i", "color=black:s=256x256:d=0.2:r=30",
                     "-c:v", "h264_nvenc", "-frames:v", "4", "-f", "null", "-"], 90)
                self._nvenc_ok = (rc == 0)
        return self._nvenc_ok

    def gpu_mode(self) -> bool:
        """是否执行 GPU 组：--gpu 强制；--quick/--no-gpu 禁止；否则自动探测。"""
        if self.args.quick or self.args.no_gpu:
            return False
        if self.args.gpu:
            return True
        return self.ffmpeg is not None and self.nvenc_ok()

    # ── 媒体信息与画质指标 ───────────────────────────────────────────────────
    def media_info(self, path: Path) -> dict:
        info = {"size": path.stat().st_size if path.exists() else 0}
        if not self.ffprobe:
            return info
        rc, out, _ = self.run(
            [self.ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,nb_frames,avg_frame_rate",
             "-show_entries", "format=duration,bit_rate",
             "-of", "json", str(path)], 60)
        if rc != 0:
            return info
        try:
            d = json.loads(out)
        except json.JSONDecodeError:
            return info
        st = (d.get("streams") or [{}])[0]
        fmt = d.get("format") or {}
        info.update({
            "width": int(st.get("width") or 0),
            "height": int(st.get("height") or 0),
            "nbf": int(st.get("nb_frames") or 0),
            "fps": _parse_fraction(st.get("avg_frame_rate") or st.get("r_frame_rate")),
            "duration": float(fmt.get("duration") or 0.0),
        })
        dur = info.get("duration") or 0.0
        info["bitrate"] = int(info["size"] * 8 / dur) if dur > 0 else 0
        info["kbps"] = info["bitrate"] // 1000
        return info

    def psnr(self, enc: Path, ref: Path, nframes: Optional[int] = None) -> Optional[float]:
        return self._metric(enc, ref, "psnr", r"average:\s*([0-9.]+|inf)", nframes)

    def ssim(self, enc: Path, ref: Path, nframes: Optional[int] = None) -> Optional[float]:
        return self._metric(enc, ref, "ssim", r"All:\s*([0-9.]+)", nframes)

    def vmaf(self, enc: Path, ref: Path, nframes: Optional[int] = None) -> Optional[float]:
        if "libvmaf" not in self.ffmpeg_filters():
            return None
        return self._metric(enc, ref, "libvmaf", r"VMAF score:\s*([0-9.]+)", nframes)

    def _metric(self, enc: Path, ref: Path, filt: str, pat: str,
                nframes: Optional[int]) -> Optional[float]:
        if not self.ffmpeg:
            return None
        cmd = [self.ffmpeg, "-hide_banner", "-v", "info", "-i", str(enc), "-i", str(ref)]
        if nframes:
            cmd += ["-frames:v", str(nframes)]
        cmd += ["-lavfi", f"[0:v][1:v]{filt}", "-f", "null", "-"]
        rc, _, err = self.run(cmd)
        hits = re.findall(pat, err)
        if not hits:
            return None
        val = hits[-1]
        if val == "inf":
            return float("inf")
        try:
            return float(val)
        except ValueError:
            return None


def _parse_fraction(s: str) -> float:
    try:
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b) if float(b) else 0.0
        return float(s)
    except (ValueError, ZeroDivisionError):
        return 0.0


def read_text(p: Path) -> str:
    """读取文本并剥离 UTF-8 BOM（部分源文件带 BOM，否则 compile()/json 会误报）。"""
    try:
        return p.read_text(encoding="utf-8-sig", errors="ignore")
    except OSError:
        return ""


def has(text: str, pattern: str, flags: int = 0) -> bool:
    return re.search(pattern, text, flags) is not None


# =============================================================================
# 模块加载
# =============================================================================

def load_quality_map(ctx: Ctx):
    if ctx.quality_map is None:
        for p in (str(UTILS_DIR), str(SRC_DIR)):
            if p not in sys.path:
                sys.path.insert(0, p)
        import importlib
        ctx.quality_map = importlib.import_module("quality_map")
    return ctx.quality_map


def load_main_module(ctx: Ctx):
    if ctx.main_mod is None:
        for p in (str(SRC_DIR), str(UTILS_DIR), str(PROC_DIR)):
            if p not in sys.path:
                sys.path.insert(0, p)
        import importlib
        ctx.main_mod = importlib.import_module("main_video_optimized")
    return ctx.main_mod


# =============================================================================
# 素材生成
# =============================================================================

def make_quality_source(ctx: Ctx) -> Path:
    """画质对比用素材：无损编码，保证"对源 PSNR"是有效基准。"""
    if ctx.args.source:
        return Path(ctx.args.source)
    out = ctx.tmp / "src_quality.mp4"
    if out.exists():
        return out
    cmd = [ctx.ffmpeg, "-y", "-v", "error",
           "-f", "lavfi", "-i",
           f"testsrc2=size={ctx.args.qw}x{ctx.args.qh}:rate={ctx.args.fps}",
           "-frames:v", str(int(ctx.args.fps * ctx.args.duration)),
           "-c:v", "libx264", "-preset", "veryfast", "-qp", "0",
           "-pix_fmt", "yuv420p", str(out)]
    rc, _, err = ctx.run(cmd)
    if rc != 0 or not out.exists():
        raise RuntimeError(f"画质素材生成失败：{err[-400:]}")
    return out


def _make_noise_source(ctx: Ctx, out: Path, w: int, h: int, fps: int,
                       dur: float) -> Path:
    """合成高熵（噪声）素材：保留足够细节把实际码率推高，但非无损以免产生 GB 级中间文件。"""
    if out.exists():
        return out
    cmd = [ctx.ffmpeg, "-y", "-v", "error",
           "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate={fps}",
           "-vf", "noise=alls=45:allf=t+u",
           "-frames:v", str(int(fps * dur)),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "10",
           "-pix_fmt", "yuv420p", str(out)]
    rc, _, err = ctx.run(cmd)
    if rc != 0 or not out.exists():
        raise RuntimeError(f"码率素材生成失败：{err[-400:]}")
    return out


def make_bitrate_source(ctx: Ctx) -> Path:
    """G8-4 码率天花板用素材：默认 1080p 高熵高帧率；`--bitrate-source` 优先。"""
    if ctx.args.bitrate_source:
        return Path(ctx.args.bitrate_source)
    return _make_noise_source(ctx, ctx.tmp / "src_bitrate.mp4",
                              ctx.args.bw, ctx.args.bh,
                              ctx.args.bfps, ctx.args.duration)


def make_bitrate_source_high(ctx: Ctx) -> Path:
    """G8-4H >1080p 高档绑定验证素材：默认 4K60 高熵（`--br-high-*` 可调）。"""
    return _make_noise_source(ctx, ctx.tmp / "src_br_high.mp4",
                              ctx.args.br_high_w, ctx.args.br_high_h,
                              ctx.args.br_high_fps, ctx.args.br_high_dur)


# =============================================================================
# 编码封装（直接调用 ffmpeg，模拟各下发路径的参数形状）
# =============================================================================

def enc_soft(ctx: Ctx, src: Path, out: Path, crf: int, preset: str = "medium",
             timeout: Optional[int] = None) -> Path:
    cmd = [ctx.ffmpeg, "-y", "-v", "error", "-i", str(src),
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-pix_fmt", "yuv420p", str(out)]
    rc, _, err = ctx.run(cmd, timeout)
    if rc != 0:
        raise RuntimeError(f"libx264 编码失败：{err[-300:]}")
    return out


def enc_nvenc(ctx: Ctx, src: Path, out: Path, codec: str, rc_mode: str,
              value: int, preset: str = "p4",
              avg_br: Optional[int] = None,
              timeout: Optional[int] = None) -> Path:
    """按 Level 2（FFmpeg CLI）形状编码。

    rc_mode='vbr_hq'  → -rc:v vbr_hq -cq:v <value> -b:v <avg_br or 0>
    rc_mode='constqp' → -rc:v constqp -qp <value>
    avg_br 非 None 时用于模拟 SDK Level 1 的 averageBitRate 字段（天花板测试）。
    """
    cmd = [ctx.ffmpeg, "-y", "-v", "error", "-i", str(src),
           "-c:v", codec, "-preset", preset]
    if rc_mode == "constqp":
        cmd += ["-rc:v", "constqp", "-qp", str(value)]
    else:
        cmd += ["-rc:v", "vbr_hq", "-cq:v", str(value)]
        if avg_br is None:
            cmd += ["-b:v", "0"]
        else:
            # 模拟 SDK Level 1：averageBitRate=avg_br，maxBitRate=avg_br*2
            cmd += ["-b:v", str(avg_br), "-maxrate", str(avg_br * 2),
                    "-bufsize", str(avg_br * 2)]
    cmd += ["-bf", "0", "-pix_fmt", "yuv420p", str(out)]
    rc, _, err = ctx.run(cmd, timeout)
    if rc != 0:
        raise RuntimeError(f"{codec} 编码失败：{err[-300:]}")
    return out


def mirror_clamp_bitrate(raw_bps: int, width: int = 0, height: int = 0) -> int:
    """脚本内独立镜像 avgBitRate 钳制公式（用于交叉校验 nvenc_sdk 的真实实现）。

    [FIX-BR-RES-ADAPT] 分辨率自适应：>1080p 用 100 Mbps 高档，其余用 50 Mbps 基线。
    """
    if raw_bps <= 0:
        return BR_FLOOR
    _cap = (BR_CEILING_HIGH
            if (width > 0 and height > 0 and width * height > BR_ADAPT_PIXELS)
            else BR_CEILING)
    return max(BR_FLOOR, min(_cap, raw_bps))


def _probe_raw_driver_illegal(ctx: Ctx, w: int, h: int, fps: int,
                              raw: int, tq: int) -> Tuple[bool, str]:
    """复刻被弃用的非法控制臂 `-b:v raw -maxrate 2*raw`，验证驱动会拒绝。

    不需要真实素材：2 帧 lavfi 色块即可在编码器初始化阶段撞上量程/驱动上限
    （1440p60 maxrate=1.33Gbps→EINVAL；4K60=2.99Gbps>INT32_MAX→ERANGE）。
    返回 (是否被拒, stderr 末行)。
    """
    cmd = [ctx.ffmpeg, "-y", "-v", "error",
           "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r={fps}:d=0.1",
           "-frames:v", "2", "-c:v", "h264_nvenc",
           "-rc:v", "vbr_hq", "-cq:v", str(tq),
           "-b:v", str(raw), "-maxrate", str(raw * 2), "-bufsize", str(raw * 2),
           "-f", "null", "-"]
    rc, _, err = ctx.run(cmd, min(int(ctx.args.timeout), 120))
    lines = [ln.strip() for ln in err.strip().splitlines() if ln.strip()]
    # 优先抓关键错误行（编码器打开失败/量程错误），否则退回末行
    tail = next((ln for ln in lines if re.search(
        r"opening encoder|Numerical result|Invalid argument|bit_rate", ln)),
        lines[-1] if lines else "")
    return rc != 0, tail


# =============================================================================
# G0 前置条件
# =============================================================================

def group_prereq(ctx: Ctx, v: Verifier) -> None:
    print("\n【G0】前置条件")
    v.add("G0-1", "PREREQ", "Python 版本 ≥ 3.9",
          Status.PASS if sys.version_info >= (3, 9) else Status.FAIL,
          detail=f"{sys.version.split()[0]}")
    v.add("G0-2", "PREREQ", "ffmpeg 可用", 
          Status.PASS if ctx.ffmpeg else Status.FAIL,
          detail=ctx.ffmpeg or "未在 PATH 找到")
    v.add("G0-3", "PREREQ", "ffprobe 可用",
          Status.PASS if ctx.ffprobe else Status.WARN,
          detail=ctx.ffprobe or "未找到（码率/帧数统计将退化为文件大小估算）")

    encs = ctx.ffmpeg_encoders()
    need = [c for c in ("libx264", "h264_nvenc", "hevc_nvenc") if c in encs]
    v.add("G0-4", "PREREQ", "ffmpeg 编码器清单",
          Status.PASS if "libx264" in encs else Status.FAIL,
          detail=", ".join(sorted(need)) or "无",
          evidence=[f"共 {len(encs)} 个编码器"])

    gpu = ctx.gpu_mode()
    nv = ctx.nvenc_ok()
    if gpu and not nv:
        v.add("G0-5", "PREREQ", "NVENC 真机可用", Status.FAIL,
              detail="--gpu 已指定但 NVENC 试编失败（驱动/权限/容器 GPU 透传）")
    elif nv:
        v.add("G0-5", "PREREQ", "NVENC 真机可用", Status.PASS, detail="试编 4 帧成功")
    else:
        v.add("G0-5", "PREREQ", "NVENC 真机可用", Status.SKIP,
              detail="未启用 GPU 组（--quick / --no-gpu 或自动探测未命中）")

    v.add("G0-6", "PREREQ", "quality_map 模块可导入",
          Status.PASS if _try(load_quality_map, ctx) else Status.FAIL,
          detail=str(QUALITY_MAP_PY.relative_to(PROJECT_ROOT)))
    v.add("G0-7", "PREREQ", "libvmaf 滤镜可用（可选）",
          Status.PASS if "libvmaf" in ctx.ffmpeg_filters() else Status.SKIP,
          detail="可用，将额外采集 VMAF" if "libvmaf" in ctx.ffmpeg_filters()
                 else "不可用，画质判定使用 PSNR + SSIM")
    v.add("G0-8", "PREREQ", "torch 可用（决定 EMIT/GPU 组）",
          Status.PASS if ctx.torch_available() else Status.SKIP,
          detail="可用" if ctx.torch_available() else "不可用，EMIT 组将 SKIP")


def _try(fn, *a) -> bool:
    try:
        fn(*a)
        return True
    except Exception:  # noqa: BLE001
        return False


# =============================================================================
# G1 换算表正确性
# =============================================================================

def group_table(ctx: Ctx, v: Verifier) -> None:
    print("\n【G1】换算表正确性")
    Q = load_quality_map(ctx)

    v.add("G1-1", "TABLE", "单一真源：convert_crf.py 提供 QUALITY_MAP",
          Status.PASS if (CONVERT_CRF_PY.exists()
                          and isinstance(getattr(Q, "QUALITY_MAP", None), dict)
                          and len(Q.QUALITY_MAP) >= 15) else Status.FAIL,
          detail=f"{len(getattr(Q, 'QUALITY_MAP', {}))} 个编码器条目",
          evidence=[f"quality_map 自 {CONVERT_CRF_PY.name} 导入并 re-export"])

    rows, bad = [], []
    for codec, (param_exp, val_exp) in REF21_EXPECTED.items():
        try:
            p, val, extra, _ = Q.resolve_quality(codec)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{codec}: 异常 {exc}")
            continue
        ok = (p == param_exp and val == val_exp)
        if not ok:
            bad.append(f"{codec}: 期望 {param_exp} {val_exp}，实际 {p} {val}")
        rows.append(f"{codec:<12} {p:<6} {val:>3}  {' '.join(extra)}")
    v.add("G1-2", "TABLE", "REF=21 等效映射与报告 §4.3 一致",
          Status.PASS if not bad else Status.FAIL,
          detail="全部命中" if not bad else "; ".join(bad),
          evidence=rows)

    # 单调性：正 a 递增，VideoToolbox（a<0）递减
    mono_bad = []
    for codec in ("libx264", "h264_nvenc", "hevc_nvenc", "libx265", "libsvtav1"):
        a = Q.QUALITY_MAP[codec][0]
        prev = None
        for ref in range(5, 45):
            val = Q.from_x264_crf(codec, ref)
            if prev is not None and not (val >= prev if a > 0 else val <= prev):
                mono_bad.append(f"{codec}@{ref}")
            prev = val
    v.add("G1-3", "TABLE", "质量-数值单调性（正轴递增 / VideoToolbox 反向）",
          Status.PASS if not mono_bad else Status.FAIL,
          detail="单调" if not mono_bad else f"异常点 {mono_bad[:5]}")

    # 往返一致：from∘to ≈ identity（避开 clamp 边界）
    rt_bad = []
    for codec in ("libx265", "h264_nvenc", "hevc_nvenc", "av1_nvenc", "libsvtav1"):
        a, b, lo, hi = Q.QUALITY_MAP[codec]
        for ref in (10, 16, 21, 28, 35):
            back = Q.to_x264_crf(codec, Q.from_x264_crf(codec, ref))
            if back is None or abs(back - ref) > 0.6:
                rt_bad.append(f"{codec}@{ref}→{back}")
    v.add("G1-4", "TABLE", "基准轴往返一致（to∘from ≈ identity）",
          Status.PASS if not rt_bad else Status.FAIL,
          detail="往返误差 ≤0.6" if not rt_bad else "; ".join(rt_bad))

    # 边界 clamp
    clamp_bad = []
    for codec, (a, b, lo, hi) in Q.QUALITY_MAP.items():
        for ref in (0, 51):
            val = Q.from_x264_crf(codec, ref)
            if val is None or not (lo - 1e-6 <= val <= hi + 1e-6):
                clamp_bad.append(f"{codec}@{ref}={val} 越界 [{lo},{hi}]")
    v.add("G1-5", "TABLE", "边界值 clamp 到各编码器合法量程",
          Status.PASS if not clamp_bad else Status.FAIL,
          detail="全部在量程内" if not clamp_bad else "; ".join(clamp_bad[:4]))

    v.add("G1-6", "TABLE", "未知编码器不猜测（返回 None）",
          Status.PASS if (Q.from_x264_crf("no_such_codec", 21) is None
                          and Q.to_x264_crf("no_such_codec", 21) is None) else Status.FAIL)

    # G1-7 literal_range：输入校验用的"技术规范可用量程"必须与 QUALITY_MAP 一致，
    #      且跨族字面量回退到源轴量程（crf→libx264，cq→h264_nvenc，均 0~51）
    if not hasattr(Q, "literal_range"):
        v.add("G1-7", "TABLE", "literal_range 提供编码器可用量程", Status.FAIL,
              detail="quality_map 未导出 literal_range（[P0-FIX-QUALITY-RANGE] 缺失）")
    else:
        exp = {
            ("libx264", "crf"): (0, 51), ("libx265", "crf"): (0, 51),
            ("libvpx-vp9", "crf"): (0, 63), ("librav1e", "crf"): (0, 51),
            ("h264_nvenc", "crf"): (0, 51), ("libx264", "cq"): (0, 51),
            ("h264_nvenc", "cq"): (0, 51), ("hevc_nvenc", "cq"): (0, 51),
            ("h264_qsv", "cq"): (1, 51), ("hevc_qsv", "cq"): (1, 51),
            ("h264_videotoolbox", "cq"): (1, 100),
            ("no_such_codec", "crf"): (0, 51), ("no_such_codec", "cq"): (0, 51),
        }
        bad = [f"{c}/{k}: {tuple(Q.literal_range(c, k))} ≠ {r}"
               for (c, k), r in exp.items() if tuple(Q.literal_range(c, k)) != r]
        v.add("G1-7", "TABLE", "literal_range 与 QUALITY_MAP 量程一致",
              Status.PASS if not bad else Status.FAIL,
              detail="全部一致" if not bad else "; ".join(bad))


# =============================================================================
# G2 resolve_quality 判定顺序
# =============================================================================

def group_resolve(ctx: Ctx, v: Verifier) -> None:
    print("\n【G2】resolve_quality 判定顺序")
    Q = load_quality_map(ctx)

    def expect(cid, title, codec, kwargs, param, value, extra=None):
        try:
            p, val, ex, _ = Q.resolve_quality(codec, **kwargs)
        except Exception as exc:  # noqa: BLE001
            v.add(cid, "RESOLVE", title, Status.FAIL, detail=f"异常 {exc}")
            return
        ok = (p == param and val == value and (extra is None or ex == extra))
        v.add(cid, "RESOLVE", title,
              Status.PASS if ok else Status.FAIL,
              detail=f"{codec} {kwargs} → {p} {val} {' '.join(ex)}"
                     + ("" if ok else f"（期望 {param} {value} {extra}）"))

    expect("G2-1", "无输入 → 默认基准 21 换算", "hevc_nvenc", {}, "-cq:v", 28)
    expect("G2-2", "crf_ref → 按基准轴换算", "hevc_nvenc", {"crf_ref": 21}, "-cq:v", 28)
    expect("G2-3", "cq_ref → 经 h264_nvenc 归一再换算", "hevc_nvenc",
           {"cq_ref": 21}, "-cq:v", 24, ["-b:v", "0"])
    expect("G2-4", "crf 字面量·同族原样下发", "libx264", {"crf": 18}, "-crf", 18)
    expect("G2-5", "cq 字面量·同族原样下发", "h264_nvenc", {"cq": 26},
           "-cq:v", 26, ["-b:v", "0"])
    expect("G2-6", "cq 字面量·跨族换算到软编", "libx264", {"cq": 26}, "-crf", 21)
    expect("G2-7", "crf 字面量·跨族换算到硬编", "h264_nvenc", {"crf": 21},
           "-cq:v", 26, ["-b:v", "0"])
    expect("G2-8", "crf_ref=0 无损意图不套线性映射", "hevc_nvenc",
           {"crf_ref": 0}, "-cq:v", 0, ["-b:v", "0"])
    expect("G2-9", "cq_ref=0 无损意图", "libx265", {"cq_ref": 0}, "-crf", 0)
    expect("G2-10", "librav1e 用 -qp 而非 -crf", "librav1e", {}, "-qp", 80)
    expect("G2-11", "libvpx-vp9 必须配 -b:v 0", "libvpx-vp9", {},
           "-crf", 27, ["-b:v", "0"])

    # 未知编码器：不猜测，原样下发（不抛异常）
    try:
        p, val, ex, note = Q.resolve_quality("no_such_codec", default_ref=21)
        v.add("G2-12", "RESOLVE", "未知编码器不抛异常（原样兜底）",
              Status.PASS if "无等效表" in note else Status.WARN,
              detail=f"→ {p} {val}", evidence=[note])
    except Exception as exc:  # noqa: BLE001
        v.add("G2-12", "RESOLVE", "未知编码器不抛异常（原样兜底）",
              Status.FAIL, detail=f"异常 {exc}")


# =============================================================================
# G3 CONSTQP 轴
# =============================================================================

def group_constqp(ctx: Ctx, v: Verifier) -> None:
    print("\n【G3】CONSTQP 轴换算")
    Q = load_quality_map(ctx)

    q1 = Q.to_constqp_qp("h264_nvenc", 26)
    v.add("G3-1", "CONSTQP", "h264_nvenc：CQ 26 → QP 21（回到基准轴）",
          Status.PASS if q1 == 21 else Status.FAIL, detail=f"QP={q1}", )

    q2 = Q.to_constqp_qp("hevc_nvenc", 28)
    # 已知取整伪影：21 → 28 → 20.5 → 银行家取整 20；±1 内接受
    if q2 == 20:
        st, det = Status.PASS, "QP=20（21↔28 往返的已知取整伪影，±1 内）"
    elif q2 in (19, 21):
        st, det = Status.WARN, f"QP={q2}（期望 20，偏差 1）"
    else:
        st, det = Status.FAIL, f"QP={q2}（期望 20）"
    v.add("G3-2", "CONSTQP", "hevc_nvenc：CQ 28 → QP ≈20", st, detail=det)

    v.add("G3-3", "CONSTQP", "未知编码器原样返回（不猜测）",
          Status.PASS if Q.to_constqp_qp("no_such_codec", 26) == 26 else Status.FAIL)
    v.add("G3-4", "CONSTQP", "无损 QP=0 保持 0",
          Status.PASS if Q.to_constqp_qp("h264_nvenc", 0) == 0 else Status.FAIL)
    v.add("G3-5", "CONSTQP", "CONSTQP_QP_OFFSET 为可调口且当前为 0",
          Status.PASS if getattr(Q, "CONSTQP_QP_OFFSET", None) == 0 else Status.WARN,
          detail=f"CONSTQP_QP_OFFSET={getattr(Q, 'CONSTQP_QP_OFFSET', 'N/A')}")

    # 与 CQ 轴明确区分：constqp 值必须不同于 -cq:v 值（否则说明没做轴换算）
    cq_val = Q.resolve_quality("hevc_nvenc", crf_ref=21)[1]
    v.add("G3-6", "CONSTQP", "CONSTQP 轴与 CQ 轴不混用（值不同）",
          Status.PASS if q2 != cq_val else Status.FAIL,
          detail=f"cq={cq_val} vs qp={q2}")


# =============================================================================
# G4 CLI 契约（进程内）
# =============================================================================

def _cli_case(ctx: Ctx, argv: List[str]):
    """返回 (accepted, message, config)。进程内执行，不触发环境检查/模型加载。"""
    M = load_main_module(ctx)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        args = M._build_parser().parse_args(argv)
        cfg = M.Config(str(CONFIG_JSON))
        M._apply_cli_overrides(cfg, args)
        ok = M._validate_effective_config(cfg, args)
    return ok, buf.getvalue(), cfg, args


def group_cli(ctx: Ctx, v: Verifier) -> None:
    print("\n【G4】CLI 契约 / 互斥 / 量程 / 优先级")
    M = load_main_module(ctx)

    # G4-1 六个新参数存在且为 int
    parser = M._build_parser()
    dests = {a.dest for a in parser._actions}
    need = ["crf_ifrnet", "cq_ifrnet", "crf_ifrnet_ref", "cq_ifrnet_ref",
            "crf_esrgan", "cq_esrgan", "crf_esrgan_ref", "cq_esrgan_ref"]
    missing = [d for d in need if d not in dests]
    v.add("G4-1", "CLI", "新增 IFRNet/ESRGAN 质量参数齐全",
          Status.PASS if not missing else Status.FAIL,
          detail="全部存在" if not missing else f"缺失 {missing}",
          evidence=["--cq-ifrnet / --crf-ifrnet-ref / --cq-ifrnet-ref",
                    "--cq-esrgan / --crf-esrgan-ref / --cq-esrgan-ref"])

    # G4-2 字面量 + 基准 互斥
    ok, msg, _, _ = _cli_case(ctx, ["--crf-ifrnet", "18", "--crf-ifrnet-ref", "21"])
    v.add("G4-2", "CLI", "IFRNet 字面量与基准互斥 → 拒绝启动",
          Status.PASS if (not ok and "互斥" in msg) else Status.FAIL,
          detail="已拒绝并提示互斥" if not ok else "未拒绝（校验缺失）")

    # G4-3 基准轴超量程
    ok, msg, _, _ = _cli_case(ctx, ["--crf-ifrnet-ref", "60"])
    v.add("G4-3", "CLI", "IFRNet 基准轴量程 0~51 → 60 拒绝",
          Status.PASS if (not ok and "0~51" in msg) else Status.FAIL,
          detail="已拒绝" if not ok else "未拒绝")

    # G4-4 cq_ref 优先且清掉配置里的 crf_ref 默认
    ok, _, cfg, _ = _cli_case(ctx, ["--cq-ifrnet-ref", "26"])
    got = (cfg.get("models", "ifrnet", "cq_ref"),
           cfg.get("models", "ifrnet", "crf_ref"))
    v.add("G4-4", "CLI", "--cq-ifrnet-ref 不被默认 crf_ref 吞掉",
          Status.PASS if (ok and got == (26, None)) else Status.FAIL,
          detail=f"cq_ref={got[0]} crf_ref={got[1]}")

    # G4-5 字面量清掉两个基准
    ok, _, cfg, _ = _cli_case(ctx, ["--cq-ifrnet", "26"])
    got = (cfg.get("models", "ifrnet", "cq"),
           cfg.get("models", "ifrnet", "crf_ref"),
           cfg.get("models", "ifrnet", "cq_ref"))
    v.add("G4-5", "CLI", "--cq-ifrnet 清掉配置默认基准",
          Status.PASS if (ok and got == (26, None, None)) else Status.FAIL,
          detail=f"cq={got[0]} crf_ref={got[1]} cq_ref={got[2]}")

    # G4-6 默认放行且 crf_ref=21
    ok, _, cfg, _ = _cli_case(ctx, [])
    v.add("G4-6", "CLI", "默认配置放行且基准 crf_ref=21",
          Status.PASS if (ok and cfg.get("models", "ifrnet", "crf_ref") == 21
                          and cfg.get("models", "realesrgan", "crf_ref") == 21)
          else Status.FAIL,
          detail=f"ifrnet={cfg.get('models', 'ifrnet', 'crf_ref')} "
                 f"esrgan={cfg.get('models', 'realesrgan', 'crf_ref')}")

    # G4-7 ESRGAN 字面量 + 基准 互斥（验证 dest 名 crf_esrgan_ref 拼写）
    ok, msg, _, _ = _cli_case(ctx, ["--crf-esrgan", "18", "--crf-esrgan-ref", "21"])
    v.add("G4-7", "CLI", "ESRGAN 字面量与基准互斥（dest 名正确）",
          Status.PASS if (not ok and "互斥" in msg) else Status.FAIL,
          detail="已拒绝" if not ok else "未拒绝（dest 名可能拼错）")

    # G4-8 ESRGAN 基准超量程
    ok, msg, _, _ = _cli_case(ctx, ["--crf-esrgan-ref", "60"])
    v.add("G4-8", "CLI", "ESRGAN 基准轴量程 0~51 → 60 拒绝",
          Status.PASS if (not ok and "0~51" in msg) else Status.FAIL)

    # G4-9 ESRGAN cq_ref 清 crf_ref
    ok, _, cfg, _ = _cli_case(ctx, ["--cq-esrgan-ref", "26"])
    got = (cfg.get("models", "realesrgan", "cq_ref"),
           cfg.get("models", "realesrgan", "crf_ref"))
    v.add("G4-9", "CLI", "--cq-esrgan-ref 不被默认 crf_ref 吞掉",
          Status.PASS if (ok and got == (26, None)) else Status.FAIL,
          detail=f"cq_ref={got[0]} crf_ref={got[1]}")

    # G4-10 字面量在基准量程内放行
    ok, _, cfg, _ = _cli_case(ctx, ["--crf-ifrnet", "51"])
    v.add("G4-10", "CLI", "字面量 --crf-ifrnet 51 放行",
          Status.PASS if (ok and cfg.get("models", "ifrnet", "crf") == 51) else Status.FAIL)

    # ── G4-12~G4-17 [P0-FIX-QUALITY-RANGE] 严格量程校验 ─────────────────────
    # 要求：所有质量输入（含字面量）按"技术规范定义的实际可用范围"校验；
    # 超限必须给出明确可读错误并拒绝执行，不得静默放行、不得自动截断。
    def _err_lines(msg: str) -> List[str]:
        return [ln.strip() for ln in msg.splitlines() if "·" in ln]

    # G4-12 默认编码器 libx264：63 超限必须拒绝，且错误点名参数/编码器/范围
    ok, msg, _, _ = _cli_case(ctx, ["--crf-ifrnet", "63"])
    errs = _err_lines(msg)
    text = " ".join(errs)
    readable = ("--crf-ifrnet" in text and "libx264" in text
                and "0~51" in text and "63" in text)
    v.add("G4-12", "CLI", "--crf-ifrnet 63（libx264 量程 0~51）拒绝且提示明确",
          Status.PASS if (not ok and readable) else Status.FAIL,
          detail=(errs[0] if errs else ("未拒绝" if ok else "错误信息不含关键要素")),
          evidence=errs)

    # G4-13 边界严格：51 接受 / 52 拒绝（证明是边界判定而非粗放范围）
    ok51, _, _, _ = _cli_case(ctx, ["--crf-ifrnet", "51"])
    ok52, msg52, _, _ = _cli_case(ctx, ["--crf-ifrnet", "52"])
    v.add("G4-13", "CLI", "字面量边界严格：51 放行 / 52 拒绝（无静默截断）",
          Status.PASS if (ok51 and not ok52) else Status.FAIL,
          detail=f"51→{'ACCEPT' if ok51 else 'REJECT'} / 52→{'ACCEPT' if ok52 else 'REJECT'}",
          evidence=_err_lines(msg52))

    # G4-14 量程随生效编码器变化：libvpx-vp9 规范量程 0~63
    ok63, _, _, _ = _cli_case(ctx, ["--crf-ifrnet", "63", "--codec-ifrnet", "libvpx-vp9"])
    ok64, msg64, _, _ = _cli_case(ctx, ["--crf-ifrnet", "64", "--codec-ifrnet", "libvpx-vp9"])
    e64 = " ".join(_err_lines(msg64))
    v.add("G4-14", "CLI", "量程随编码器而定：libvpx-vp9 接受 63 / 拒绝 64",
          Status.PASS if (ok63 and not ok64 and "0~63" in e64) else Status.FAIL,
          detail=(f"63→{'ACCEPT' if ok63 else 'REJECT'} / 64→{'ACCEPT' if ok64 else 'REJECT'}"
                  + ("" if "0~63" in e64 else "；64 的错误未给出 0~63")),
          evidence=_err_lines(msg64))

    # G4-15 CQ 轴量程：默认（软编）按 h264_nvenc 0~51；QSV 下界为 1
    okc0, msgc0, _, _ = _cli_case(ctx, ["--cq-ifrnet", "52"])
    okq0, msgq0, _, _ = _cli_case(ctx, ["--cq-ifrnet", "0", "--codec-ifrnet", "h264_qsv"])
    r15 = (not okc0 and not okq0
           and "0~51" in " ".join(_err_lines(msgc0))
           and "1~51" in " ".join(_err_lines(msgq0)))
    v.add("G4-15", "CLI", "CQ 轴量程：52 拒绝（0~51）/ QSV 0 拒绝（1~51）",
          Status.PASS if r15 else Status.FAIL,
          detail=(_err_lines(msgq0) or _err_lines(msgc0) or ["未按预期拒绝"])[0],
          evidence=_err_lines(msgc0) + _err_lines(msgq0))

    # G4-16 ESRGan 侧同规则：63 拒绝，且可读
    ok, msg, _, _ = _cli_case(ctx, ["--crf-esrgan", "63"])
    e16 = " ".join(_err_lines(msg))
    v.add("G4-16", "CLI", "ESRGan 侧同规则：--crf-esrgan 63 拒绝",
          Status.PASS if (not ok and "--crf-esrgan" in e16 and "0~51" in e16) else Status.FAIL,
          detail=(_err_lines(msg) or ["未拒绝"])[0])

    # G4-17 JSON 配置来源同样校验（无 CLI 时按配置值判定）
    #   直接把 config 的 crf 改成超限值，模拟"配置文件写错"，应被拒绝
    M = load_main_module(ctx)
    args = M._build_parser().parse_args([])
    cfg = M.Config(str(CONFIG_JSON))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        cfg.set("models", "ifrnet", "crf", value=99)
        ok_cfg = M._validate_effective_config(cfg, args)
    v.add("G4-17", "CLI", "配置来源超限量（models.ifrnet.crf=99）同样拒绝",
          Status.PASS if not ok_cfg else Status.FAIL,
          detail="已拒绝" if not ok_cfg else "未拒绝（存在静默放行路径）")

    # G4-11 _quality_label 四种渲染
    def _lbl(vals: dict) -> str:
        return M._quality_label(lambda k, d=None: vals.get(k, d))
    cases = [
        ({"crf_ref": 21}, "CRF-ref: 21"),
        ({"cq_ref": 26}, "CQ-ref: 26"),
        ({"cq": 30}, "CQ: 30"),
        ({"crf": 18}, "CRF: 18"),
    ]
    bad = [f"{vals}→{_lbl(vals)}" for vals, exp in cases if exp not in _lbl(vals)]
    v.add("G4-11", "CLI", "摘要 _quality_label 四态渲染正确",
          Status.PASS if not bad else Status.FAIL,
          detail="四种输入均正确" if not bad else "; ".join(bad))


# =============================================================================
# G5 代码落点（静态结构核验）
# =============================================================================

def group_static(ctx: Ctx, v: Verifier) -> None:
    print("\n【G5】代码落点静态核验")

    def src(rel: str) -> str:
        return read_text(PROJECT_ROOT / rel)

    # G5-1 quality_map 契约
    qm = src("src/utils/quality_map.py")
    checks = [
        ("def resolve_quality", "resolve_quality 存在"),
        ("def to_constqp_qp", "to_constqp_qp 存在"),
        (r"DEFAULT_REF\s*:\s*int\s*=\s*21", "DEFAULT_REF=21"),
        (r"CONSTQP_QP_OFFSET\s*:\s*int\s*=", "CONSTQP_QP_OFFSET 可调口"),
        (r"def literal_range", "literal_range 量程表（输入校验单一真源）"),
        (r"from convert_crf import", "换算表来自单一真源 convert_crf"),
    ]
    bad = [d for pat, d in checks if not has(qm, pat)]
    v.add("G5-1", "STATIC", "quality_map.py 对外契约完整",
          Status.PASS if not bad else Status.FAIL,
          detail="全部存在" if not bad else f"缺失 {bad}")

    # G5-11 [P0-FIX-QUALITY-RANGE] 主入口确实调用了 literal_range 做量程校验
    mv = src("src/main_video_optimized.py")
    ok = (has(mv, r"from quality_map import .*literal_range")
          and has(mv, r"literal_range\(\s*_codec\s*,\s*_kind\s*\)"))
    v.add("G5-11", "STATIC", "主入口接线 literal_range（[P0-FIX-QUALITY-RANGE]）",
          Status.PASS if ok else Status.FAIL,
          detail="已导入并在校验循环中调用" if ok else "未接线")

    # G5-2 IFRNet 后端接入
    imain = src("external/ifrnet_video/main.py")
    checks = [
        (r"_UTILS_DIR\s*=", "src/utils 路径注入（quality_map 可导入）"),
        (r"from quality_map import .*resolve_quality", "导入 resolve_quality"),
        (r"def _resolve_effective_crf", "_resolve_effective_crf 存在"),
        (r"self\._crf_original\s*=", "保留原始字面量（防跨段二次换算）"),
        (r"self\._resolved_crf_cache", "按编码器缓存解析结果"),
        (r"to_constqp_qp\(\s*use_codec", "Level 1 constqp 轴换算"),
    ]
    bad = [d for pat, d in checks if not has(imain, pat)]
    v.add("G5-2", "STATIC", "IFRNet 后端接入统一换算",
          Status.PASS if not bad else Status.FAIL,
          detail="全部命中" if not bad else f"缺失 {bad}")

    # G5-3 IFRNet 下发路径
    ifio = src("external/ifrnet_video/ffmpeg_io.py")
    checks = [
        (r"'_?constqp'\s*:\s*'constqp'", "constqp 不再被错映射为 vbr"),
        (r"'-rc:v',\s*'constqp',\s*'-qp',\s*str\(\s*_nvenc_qp\s*\)", "CONSTQP 发 -qp"),
        (r"'-rc:v',\s*_rc_v,\s*'-cq:v',\s*str\(\s*crf\s*\),\s*'-b:v',\s*'0'", "VBR 发 -cq:v + -b:v 0"),
        (r"_nvenc_qp\s*=\s*to_constqp_qp\(\s*codec\s*,\s*crf\s*\)", "constqp 轴换算调用"),
    ]
    bad = [d for pat, d in checks if not has(ifio, pat, re.S)]
    v.add("G5-3", "STATIC", "IFRNet FFmpeg 下发路径正确",
          Status.PASS if not bad else Status.FAIL,
          detail="全部命中" if not bad else f"缺失 {bad}")

    # G5-4 ESRGAN 后端接入
    emain = src("external/realesrgan_video/main.py")
    checks = [
        (r"_UTILS_DIR\s*=", "src/utils 路径注入"),
        (r"resolve_quality\(", "[QUALITY-UNIFY] 解析调用"),
        (r"\[FIX-RATEMODE\]", "constqp 显式清 LA"),
        (r"to_constqp_qp\(\s*args\.codec", "SDK 路径 constqp 轴换算"),
    ]
    bad = [d for pat, d in checks if not has(emain, pat)]
    v.add("G5-4", "STATIC", "Real-ESRGAN 后端接入统一换算",
          Status.PASS if not bad else Status.FAIL,
          detail="全部命中" if not bad else f"缺失 {bad}")

    # G5-5 ESRGAN 下发路径（含无损分支显式 -rc constqp）
    efio = src("external/realesrgan_video/ffmpeg_io.py")
    checks = [
        (r"'_?constqp'\s*:\s*'constqp'", "constqp 映射正确"),
        (r"nvenc_rc\s*==\s*'constqp'", "constqp 分支存在"),
        (r"'-rc:v',\s*'constqp',\s*'-qp',\s*str\(\s*_nvenc_qp\s*\)", "CONSTQP 发 -qp"),
        (r"'-rc',\s*'constqp',\s*\n\s*'-qp',\s*'0'", "无损分支显式 -rc constqp"),
    ]
    bad = [d for pat, d in checks if not has(efio, pat)]
    v.add("G5-5", "STATIC", "Real-ESRGAN FFmpeg 下发路径正确",
          Status.PASS if not bad else Status.FAIL,
          detail="全部命中" if not bad else f"缺失 {bad}")

    # G5-6 avgBitRate 天花板（分辨率自适应）：两侧同公式、同常量、无旧式无上限
    bad = []
    for rel in ("external/ifrnet_video/nvenc_sdk.py",
                "external/realesrgan_video/nvenc_sdk.py"):
        t = src(rel)
        if not has(t, r"_NVENC_BR_CLAMP_MAX\s*:\s*int\s*=\s*50_000_000"):
            bad.append(f"{rel}: 缺 50M 基线常量")
        if not has(t, r"_NVENC_BR_CLAMP_MAX_HIGH\s*:\s*int\s*=\s*100_000_000"):
            bad.append(f"{rel}: 缺 100M 高档常量")
        if not has(t, r"_NVENC_BR_CLAMP_MIN\s*:\s*int\s*=\s*5_000_000"):
            bad.append(f"{rel}: 缺 5M 常量")
        if not has(t, r"_NVENC_BR_ADAPT_PIXELS\s*:\s*int\s*=\s*1920\s*\*\s*1080"):
            bad.append(f"{rel}: 缺分辨率分档阈值")
        if not has(t, r"def _max_avg_bitrate"):
            bad.append(f"{rel}: 缺 _max_avg_bitrate")
        if not has(t, r"def _clamp_bitrate"):
            bad.append(f"{rel}: 缺 _clamp_bitrate")
        if not has(t, r"_clamp_bitrate\(\s*int\(\s*width\s*\*\s*height\s*\*\s*fps"
                      r"\s*\*\s*3\.0\s*\)\s*,\s*width\s*,\s*height\s*\)"):
            bad.append(f"{rel}: 调用点未传分辨率")
        if has(t, r"max\(\s*50_?000_?000\s*,"):
            bad.append(f"{rel}: 残留旧式 max(50M, …) 无上限写法")
    v.add("G5-6", "STATIC", "avgBitRate 天花板分辨率自适应（两侧同公式同常量）",
          Status.PASS if not bad else Status.FAIL,
          detail="两侧一致且无旧式残留" if not bad else "; ".join(bad))

    # G5-7 默认值对齐
    try:
        cfg_json = json.loads(read_text(CONFIG_JSON))
        j_ifr = cfg_json["models"]["ifrnet"]
        j_esr = cfg_json["models"]["realesrgan"]
    except Exception as exc:  # noqa: BLE001
        j_ifr = j_esr = {}
        v.add("G5-7", "STATIC", "默认值对齐", Status.FAIL, detail=f"JSON 解析失败 {exc}")
    else:
        def _d(d):
            return (d.get("crf_ref"), d.get("rate_mode"), d.get("lookahead_depth"))
        ok = _d(j_ifr) == (21, "vbr_hq", 8) and _d(j_esr) == (21, "vbr_hq", 8)
        v.add("G5-7", "STATIC", "JSON 默认 crf_ref/rate_mode/LA 两侧一致",
              Status.PASS if ok else Status.FAIL,
              detail=f"ifrnet={_d(j_ifr)} esrgan={_d(j_esr)}")

    cm = src("src/utils/config_manager.py")
    ok = (has(cm, r"\"crf_ref\":\s*21") and has(cm, r"\"rate_mode\":\s*\"vbr_hq\"")
          and has(cm, r"\"lookahead_depth\":\s*8"))
    v.add("G5-8", "STATIC", "config_manager 内置默认与 JSON 一致",
          Status.PASS if ok else Status.FAIL)

    # G5-9 编译核验（用 compile() 而非 py_compile，避免 Windows 上写 nul 的坑）
    bad = []
    for rel in CHANGED_FILES:
        p = PROJECT_ROOT / rel
        if not p.exists():
            bad.append(f"{rel}: 文件不存在")
            continue
        if p.suffix != ".py":
            continue
        try:
            compile(read_text(p), rel, "exec")
        except SyntaxError as exc:
            bad.append(f"{rel}:{exc.lineno}: {exc.msg}")
    v.add("G5-9", "STATIC", "全部改动 Python 文件编译通过",
          Status.PASS if not bad else Status.FAIL,
          detail="OK" if not bad else "; ".join(bad))

    # G5-10 镜像包一致性（两侧 RC 映射表内容一致）
    def rc_map(t: str) -> Optional[str]:
        m = re.search(r"_rc_v_map\s*=\s*\{([^}]*)\}", t, re.S)
        if m:
            return m.group(1)
        m = re.search(r"_NVENC_RC_MAP\s*=\s*\{([^}]*)\}", t, re.S)
        return m.group(1) if m else None
    a, b = rc_map(ifio), rc_map(efio)
    norm = lambda s: sorted(re.findall(r"'(\w+)'\s*:\s*'(\w+)'", s or ""))
    v.add("G5-10", "STATIC", "两侧后端 NVENC RC 映射表语义一致",
          Status.PASS if (a and b and norm(a) == norm(b)) else Status.FAIL,
          detail=f"ifrnet={norm(a)} | esrgan={norm(b)}")


# =============================================================================
# G6 命令下发动态捕获（尽力而为）
# =============================================================================

class _FakePopen:
    """替换 subprocess.Popen，仅捕获命令行，不真正启动 ffmpeg。

    [FIX-HARNESS-POLL] 必须模拟"仍在运行"的进程：poll() 存活期间返回 None，
    否则 realesrgan 的 FFmpegWriter 在构造末尾 `if self._process.poll() is not None:`
    会误判"FFmpeg 启动失败(rc=0)"并抛错，导致 ESRGAN 侧命令永远捕获不到
    （见 3 份报告中 G6-4/5/6 恒为 SKIP）。wait()/terminate()/kill() 之后才落定
    returncode，close() 中的 poll()/wait() 分支才能正常走完。
    """

    def __init__(self, cmd, *a, **k):
        self.cmd = list(cmd)
        self.pid = -1
        self.returncode = None      # None = 仍在运行
        self.stdin = io.BytesIO()
        self.stderr = iter(())

    def wait(self, timeout=None):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def poll(self):
        return self.returncode      # 存活 → None

    def kill(self):
        self.returncode = -9

    def terminate(self):
        self.returncode = -15


def _capture_writer(ctx: Ctx, pkg: str, codec: str, crf: int, rc_mode: str,
                    lookahead: Optional[int]) -> Optional[List[str]]:
    """构造后端 FFmpegWriter 并捕获其将执行的 ffmpeg 命令（不真正拉起进程）。

    两个后端的 FFmpegWriter 构造签名不同：IFRNet 为位置参数 + codec/crf/rc_mode，
    Real-ESRGAN 为 (args_namespace, audio, h, w, path, fps)。此处分别适配。
    同时把 HardwareCapability.best_encoder 临时替换为恒等函数，确保捕获的是
    "传入该 codec 时的命令形状"，而不是自动升降级之后的结果。
    """
    for p in (str(UTILS_DIR), str(EXTERNAL_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import importlib
    fio = importlib.import_module(f"{pkg}.ffmpeg_io")
    captured: Dict[str, List[str]] = {}
    real_popen = subprocess.Popen
    hw = getattr(fio, "HardwareCapability", None)
    real_best = hw.__dict__.get("best_encoder") if hw is not None else None
    if hw is not None:
        hw.best_encoder = staticmethod(lambda c, **k: c)  # type: ignore[assignment]

    def fake(cmd, *a, **k):
        captured["cmd"] = list(cmd)
        return _FakePopen(cmd)

    subprocess.Popen = fake  # type: ignore[assignment]
    try:
        if pkg == "ifrnet_video":
            w = fio.FFmpegWriter(str(ctx.tmp / "emit.mp4"), 320, 240, 30.0,
                                 codec=codec, crf=crf, rc_mode=rc_mode,
                                 lookahead_depth=lookahead, quiet=True)
        else:
            ns = types.SimpleNamespace(
                codec=codec, crf=crf, rate_mode=rc_mode,
                lookahead_depth=lookahead, encode_preset="medium",
                ffmpeg_bin="ffmpeg", quiet=True)
            w = fio.FFmpegWriter(ns, None, 240, 320, str(ctx.tmp / "emit.mp4"), 30.0)
        with contextlib.suppress(Exception):
            w.close()
    finally:
        subprocess.Popen = real_popen  # type: ignore[assignment]
        if hw is not None and real_best is not None:
            hw.best_encoder = real_best  # type: ignore[assignment]
    return captured.get("cmd")


def _has_pair(cmd: Optional[List[str]], key: str, val: str) -> bool:
    if not cmd:
        return False
    for i, tok in enumerate(cmd[:-1]):
        if tok == key and cmd[i + 1] == val:
            return True
    return False


def group_emit(ctx: Ctx, v: Verifier) -> None:
    print("\n【G6】下发命令捕获（ffmpeg 参数形状）")

    cases = [
        ("G6-1", "IFRNet", "ifrnet_video", "h264_nvenc", 26, "vbr_hq", 8,
         [("-rc:v", "vbr_hq"), ("-cq:v", "26"), ("-b:v", "0"), ("-rc-lookahead", "8")],
         [("-qp", None)]),
        ("G6-2", "IFRNet", "ifrnet_video", "h264_nvenc", 26, "constqp", 8,
         [("-rc:v", "constqp"), ("-qp", "21")],
         [("-cq:v", None), ("-b:v", None)]),
        ("G6-3", "IFRNet", "ifrnet_video", "libx264", 21, "vbr_hq", 8,
         [("-crf", "21")], [("-cq:v", None)]),
        ("G6-4", "ESRGAN", "realesrgan_video", "h264_nvenc", 26, "vbr_hq", 8,
         [("-rc:v", "vbr_hq"), ("-cq:v", "26"), ("-b:v", "0")], []),
        ("G6-5", "ESRGAN", "realesrgan_video", "h264_nvenc", 26, "constqp", 0,
         [("-rc:v", "constqp"), ("-qp", "21")], [("-cq:v", None)]),
        ("G6-6", "ESRGAN", "realesrgan_video", "libx265", 24, "vbr_hq", 8,
         [("-crf", "24")], [("-cq:v", None)]),
    ]

    for cid, stage, pkg, codec, crf, rc, la, want, forbid in cases:
        title = f"{stage} {codec} rc={rc} → 命令形状"
        try:
            cmd = _capture_writer(ctx, pkg, codec, crf, rc, la)
        except Exception as exc:  # noqa: BLE001
            v.add(cid, "EMIT", title, Status.SKIP,
                  detail=f"后端不可导入/需 GPU：{type(exc).__name__}: {exc}")
            continue
        if not cmd:
            v.add(cid, "EMIT", title, Status.FAIL, detail="未捕获到命令")
            continue
        miss = [f"{k} {val or ''}".strip() for k, val in want if not _has_pair(cmd, k, val)]
        ban = [k for k, val in forbid if _has_pair(cmd, k, val if val else _next_of(cmd, k))]
        v.add(cid, "EMIT", title,
              Status.PASS if (not miss and not ban) else Status.FAIL,
              detail=("命令形状正确" if not (miss or ban)
                      else f"缺少 {miss} / 多出 {ban}"),
              evidence=[" ".join(cmd)[:220]])


def _next_of(cmd: List[str], key: str) -> str:
    for i, t in enumerate(cmd[:-1]):
        if t == key:
            return cmd[i + 1]
    return ""


# =============================================================================
# G7 画质统一性实测
# =============================================================================

def _fmt(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return "n/a"
    if x == float("inf"):
        return "inf"
    return f"{x:.{nd}f}"


def _task_ram_mb(width: int, height: int) -> int:
    """按实测峰值 RSS 推导单个 ffmpeg 任务要预留的内存（MB）。

    实测（psutil 采样子进程 peak RSS；RSS 只随分辨率增长，与片段时长无关）：

        分辨率          psnr    ssim   libvmaf(默认)  libvmaf(8线程)  编码(libx264 medium)
        640x360          41      53       63            ~120              81
        1920x1080       205     205      278             484             448
        3840x2160       696     683      985            1885            1435

    两点结论决定了档位取值：
      1. libvmaf 是三类指标里最重的，且对线程数敏感（1080p 每线程约 +30MB，
         4K 约 +129MB）。本脚本发的是裸 `libvmaf`（默认 1 线程档，1080p 稳定
         275~278MB），但为兼容默认线程数更高的 ffmpeg 构建，按多线程上界定档。
      2. 编码池主机侧以解码缓冲为主，与软编同量级 —— 于是两池可用同一张表，
         取"最重任务 × 约 1.1~1.5 倍余量"。

    旧值固定 1024MB 在 640x360（实际 63MB）下把并发掐到 1/16，又对 4K 偏紧；
    改为随分辨率走，既放得开小分辨率并发，也不超售 4K 的内存。
    1440p/8K 两档无直接实测，按像素数在 1080p↔4K 之间线性插值 + 余量给出。
    """
    px = max(0, int(width)) * max(0, int(height))
    for limit_px, mb in ((1280 * 720, 192), (1920 * 1080, 640),
                         (2560 * 1440, 1024), (3840 * 2160, 2048)):
        if px <= limit_px:
            return mb
    return 3072  # 超过 4K：按 4K 档再放宽，宁保守不超售


def _encode_timeout(ctx: Ctx, width: int, height: int, nframes: int) -> int:
    """单条编码命令超时（秒）。

    `--encode-timeout` 显式覆盖；否则按帧数自适应：以保守软编速度 ~5 fps
    （0.2 s/帧）+120s 固定开销估算。下限沿用 `--timeout`（合成素材 90~180 帧
    → 600s，行为不变），上限 3600s，避免真实长片在 600s 被误杀（见 123938）。
    """
    if int(getattr(ctx.args, "encode_timeout", 0) or 0) > 0:
        return int(ctx.args.encode_timeout)
    est = 120 + int(max(0, nframes) * 0.2)
    return max(int(ctx.args.timeout), min(3600, est))


def _run_jobs(ctx: Ctx, jobs: Sequence[Tuple[str, Callable[[], Any]]], *,
              gpu: bool = False, task_ram_mb: int = 512,
              label: str = "任务") -> Dict[str, Any]:
    """执行 jobs=[(任务名, 零参可调用)]，返回 {任务名: 返回值}（与输入等长）。

    · `--jobs 1` 或仅 1 个任务 → 串行就地执行：既保证可与旧行为逐项比对，也让
      需要独占计时的调用方（如 G8-5 的编码耗时）不被并行干扰。
    · 其余情况交给 ParallelExecutor：worker 数自动探测系统资源决定，
      gpu=True 时额外走 GPU 会话闸门；task_ram_mb 由 _task_ram_mb() 按分辨率给。
    · 任一任务异常 → 抛携带原始错误首行的 RuntimeError，与旧串行版一致，
      不把失败静默吞成 None。
    """
    if not jobs:
        return {}
    names = [n for n, _ in jobs]
    if len(names) != len(set(names)):
        raise ValueError(f"{label} 任务名重复：{names}")
    if int(getattr(ctx.args, "jobs", 0) or 0) == 1 or len(jobs) == 1:
        return {name: fn() for name, fn in jobs}
    results = ctx.executor(gpu, task_ram_mb).submit_tasks(
        [fn for _, fn in jobs], names)
    out: Dict[str, Any] = {}
    errs: List[str] = []
    for name, res in zip(names, results):
        if res.success:
            out[name] = res.result
        else:
            errs.append(f"{name}: {(res.error or '').splitlines()[0]}")
    if errs:
        raise RuntimeError(f"{label} 并行执行失败 {len(errs)}/{len(jobs)} 项："
                           + "; ".join(errs))
    return out


def _measure_many(ctx: Ctx, targets: Dict[str, Path], ref: Path, n: int,
                  res: Tuple[int, int], *, gpu: bool = False,
                  filters: Sequence[str] = ("psnr", "ssim", "vmaf")) -> Dict[str, dict]:
    """并行度量多份编码产物，返回 {逻辑名: {psnr, ssim, vmaf, ...media_info}}。

    把"每份文件 × 每种指标"拆成独立任务，而不是"每份文件一个任务"：VMAF 远慢于
    PSNR/SSIM，细粒度切分才能让快指标填满慢指标留下的空档，避免长尾。
    res=(w,h) 为素材分辨率，用于按实际画幅给内存档位（见 _task_ram_mb）。
    filters 可按需裁剪（4K 高档绑定只测 PSNR/SSIM，省掉最重的 VMAF）。
    media_info 只是轻量 ffprobe，留在主线程串行补齐即可。
    """
    _all = {"psnr": ctx.psnr, "ssim": ctx.ssim, "vmaf": ctx.vmaf}
    jobs: List[Tuple[str, Callable[[], Any]]] = []
    for name, path in targets.items():
        for filt in filters:
            fn = _all[filt]
            # [FIX-VMAF-LONGFORM] VMAF 截断到前 VMAF_MAX_FRAMES 帧，避免长视频超时返回 None
            _nf = n
            if filt == "vmaf" and n:
                _nf = min(int(n), VMAF_MAX_FRAMES)
            jobs.append((f"{name}|{filt}",
                         (lambda p=path, f=fn, k=_nf: f(p, ref, k))))
    raw = _run_jobs(ctx, jobs, gpu=gpu, task_ram_mb=_task_ram_mb(*res),
                    label="画质度量")
    out: Dict[str, dict] = {}
    for name, path in targets.items():
        m: Dict[str, Any] = {filt: raw.get(f"{name}|{filt}")
                             for filt in ("psnr", "ssim", "vmaf")}
        m.update(ctx.media_info(path))
        out[name] = m
    return out


def _rate_verdict(d_signed: float, ratio_c: float,
                  ratio_n: Optional[float] = None) -> Tuple[Status, str]:
    """[FIX-G7-CRITERION] "有符号 ΔPSNR 单向容差 + 码率保真度" 判据。

    旧判据用无方向的 |ΔPSNR| 且忽略码率，会把"朴素下发多花 1.7× 码率、PSNR 反而
    高出软编 1.5dB"误判为"更接近基准"（3 份报告 Run3 的 FAIL 即源于此）。

    新判据：
      · 质量：换算值不得比软编基准低超过 TOL_PSNR_DB（单向下探）；过配（正值）不罚。
      · 码率：换算值码率须落在 RATE_PASS 带内（= 限码率刻度已校准）。
      PASS = 质量与码率皆达标；WARN = 落在放宽带内且比朴素值更接近 1.0；否则 FAIL。
    """
    _r_ok   = RATE_PASS[0] <= ratio_c <= RATE_PASS[1]
    _r_warn = RATE_WARN[0] <= ratio_c <= RATE_WARN[1]
    _q_ok   = d_signed >= -TOL_PSNR_DB
    _q_warn = d_signed >= -TOL_PSNR_WARN
    _improve = (ratio_n is None) or (abs(ratio_c - 1.0) <= abs(ratio_n - 1.0))
    if _q_ok and _r_ok:
        st = Status.PASS
    elif _q_warn and _r_warn and _improve:
        st = Status.WARN
    else:
        st = Status.FAIL
    detail = (f"ΔPSNR(有符号)={d_signed:+.2f} dB，码率比={ratio_c:.2f}×"
              + (f"（朴素 {ratio_n:.2f}×）" if ratio_n is not None else "")
              + f"；质量{'达标' if _q_ok else ('略松' if _q_warn else '过松')}"
              + f"，码率{'达标' if _r_ok else ('偏离' if _r_warn else '失控')}")
    return st, detail


def group_quality(ctx: Ctx, v: Verifier) -> None:
    print("\n【G7】画质统一性实测（同一基准 → 不同编码器）")
    Q = load_quality_map(ctx)
    if not ctx.gpu_mode():
        for cid in ("G7-1", "G7-2", "G7-3", "G7-4", "G7-5"):
            v.add(cid, "QUALITY", "画质统一性实测", Status.SKIP,
                  detail="未启用 GPU 组（--quick / 无 NVENC）")
        return

    try:
        src = make_quality_source(ctx)
    except Exception as exc:  # noqa: BLE001
        for cid in ("G7-1", "G7-2", "G7-3", "G7-4", "G7-5"):
            v.add(cid, "QUALITY", "画质统一性实测", Status.FAIL, detail=f"素材准备失败：{exc}")
        return

    ref_info = ctx.media_info(src)
    n = ref_info.get("nbf") or int(ctx.args.fps * ctx.args.duration)
    res = (ref_info.get("width", 0), ref_info.get("height", 0))
    out = ctx.tmp

    # ── 阶段①：并行编码。6 个任务只依赖 src、彼此独立；经 GPU 会话闸门限流
    #    （闸门容量 = 本机 NVENC 会话上限，消费级卡通常为 2）────────────────
    NAIVE = 21  # 修复前：基准轴数值被原样当作硬编 CQ 下发
    pairs = [("h264_nvenc", "G7-1"), ("hevc_nvenc", "G7-2")]
    resolved = {codec: Q.resolve_quality(codec, default_ref=21) for codec, _ in pairs}
    qp = Q.to_constqp_qp("h264_nvenc", resolved["h264_nvenc"][1])

    # 真实长素材的整段软编可能远超 --timeout(600s)，按帧数自适应放宽（见 _encode_timeout）
    enc_to = _encode_timeout(ctx, *res, n)
    enc_jobs: List[Tuple[str, Callable[[], Path]]] = [
        ("soft", lambda: enc_soft(ctx, src, out / "q_libx264_r21.mp4", 21,
                                  timeout=enc_to)),
    ]
    for codec, _cid in pairs:
        val = resolved[codec][1]
        enc_jobs.append((f"{codec}__c{val}",
                         (lambda c=codec, v=val: enc_nvenc(
                             ctx, src, out / f"q_{c}_c{v}.mp4", c, "vbr_hq", v,
                             timeout=enc_to))))
        enc_jobs.append((f"{codec}__n{NAIVE}",
                         (lambda c=codec, v=NAIVE: enc_nvenc(
                             ctx, src, out / f"q_{c}_c{v}.mp4", c, "vbr_hq", v,
                             timeout=enc_to))))
    enc_jobs.append(("constqp",
                     (lambda q=qp: enc_nvenc(ctx, src, out / f"q_h264_nvenc_qp{q}.mp4",
                                             "h264_nvenc", "constqp", q,
                                             timeout=enc_to))))
    enc_files = _run_jobs(ctx, enc_jobs, gpu=True,
                          task_ram_mb=_task_ram_mb(*res), label="G7 编码")

    # ── 阶段②：并行度量（PSNR/SSIM/VMAF 逐指标拆任务；无 libvmaf 时秒回 None）
    measured = _measure_many(ctx, enc_files, src, n, res)
    m_soft = measured["soft"]
    ctx.metrics["libx264_crf21"] = m_soft

    # 逐硬编编码器验证：换算值 vs 朴素下发值（旧行为）
    for codec, cid in pairs:
        param, val, extra, note = resolved[codec]
        m_c = measured[f"{codec}__c{val}"]
        m_n = measured[f"{codec}__n{NAIVE}"]
        ctx.metrics[f"{codec}_rescued_{val}"] = m_c
        ctx.metrics[f"{codec}_naive_{NAIVE}"] = m_n

        _psnr_soft = m_soft["psnr"] or 0
        _kbps_soft = m_soft.get("kbps") or 0
        d_c = (m_c["psnr"] or 0) - _psnr_soft                       # 有符号
        ratio_c = (m_c.get("kbps") or 0) / _kbps_soft if _kbps_soft else 0.0
        ratio_n = (m_n.get("kbps") or 0) / _kbps_soft if _kbps_soft else 0.0
        st, vd = _rate_verdict(d_c, ratio_c, ratio_n)
        v.add(cid, "QUALITY", f"{codec} 基准21→{param} {val} 对齐 libx264 crf21",
              st,
              detail=f"{vd}（软编 {_fmt(_psnr_soft)} dB / {_kbps_soft} kbps）",
              evidence=[f"换算值 {param} {val}：PSNR {_fmt(m_c['psnr'])} "
                        f"SSIM {_fmt(m_c['ssim'], 4)} 码率 {m_c.get('kbps', 0)} kbps",
                        f"朴素值 -cq:v {NAIVE}：PSNR {_fmt(m_n['psnr'])} "
                        f"SSIM {_fmt(m_n['ssim'], 4)} 码率 {m_n.get('kbps', 0)} kbps"
                        f"（{ratio_n:.2f}×，仅用于对照）",
                        f"换算说明：{note}"])

    # constqp 轴验证（h264_nvenc：CQ 26 → QP 21）
    m_qp = measured["constqp"]
    ctx.metrics["h264_nvenc_constqp"] = m_qp
    _psnr_soft = m_soft["psnr"] or 0
    _kbps_soft = m_soft.get("kbps") or 0
    d_qp = (m_qp["psnr"] or 0) - _psnr_soft
    ratio_qp = (m_qp.get("kbps") or 0) / _kbps_soft if _kbps_soft else 0.0
    _m_naive = ctx.metrics.get(f"h264_nvenc_naive_{NAIVE}", {})
    ratio_naive = ((_m_naive.get("kbps") or 0) / _kbps_soft
                   if _kbps_soft else None)
    st, vd = _rate_verdict(d_qp, ratio_qp, ratio_naive)
    # 合成 testsrc2 下 constqp 是"过配"（ΔPSNR>0），比值仅轻微越过 RATE_PASS 上界；
    # 真实素材（131802 / 102225）均 PASS（1.40×/ΔPSNR −0.26）。仅在未显式提供真实
    # --source 时标注为预期，避免把合成素材假阳性误读为回归；喂真实素材仍 WARN
    # 时注解不出现 ⇒ WARN 即回归信号，判据守护不被削弱。
    _synthetic = not getattr(ctx.args, "source", None)
    if st is Status.WARN and d_qp > 0 and ratio_qp <= RATE_WARN[1] and _synthetic:
        vd += ("；【合成素材已知过配边界】真实素材已验证 PASS"
               "（131802 G7-3 1.40×/ΔPSNR −0.26），此处 WARN 为预期、非回归")
    v.add("G7-3", "QUALITY", f"CONSTQP 轴：-qp {qp} 对齐 libx264 crf21", st,
          detail=vd,
          evidence=[f"-rc:v constqp -qp {qp}：PSNR {_fmt(m_qp['psnr'])} "
                    f"SSIM {_fmt(m_qp['ssim'], 4)} 码率 {m_qp.get('kbps', 0)} kbps"
                    f"（{ratio_qp:.2f}×）"])

    # 码率合理性：换算后不应相对软编暴涨（修复前正是暴涨）
    bad_br = []
    for codec in ("h264_nvenc", "hevc_nvenc"):
        m = ctx.metrics.get(f"{codec}_rescued_"
                            f"{Q.resolve_quality(codec, default_ref=21)[1]}")
        if not m or not m_soft.get("kbps"):
            continue
        ratio = m["kbps"] / m_soft["kbps"]
        if ratio > 2.5:
            bad_br.append(f"{codec} 码率比 {ratio:.2f}×")
        ctx.metrics.setdefault("bitrate_ratio", {})[codec] = round(ratio, 3)
    v.add("G7-4", "QUALITY", "换算后码率与软编同量级（≤2.5×）",
          Status.PASS if not bad_br else Status.WARN,
          detail="量级一致" if not bad_br else "; ".join(bad_br),
          evidence=[f"libx264 crf21 码率 {m_soft.get('kbps', 0)} kbps"])

    # VMAF 可选补充（直接复用阶段②已测值，不再重复解码 + 打分）
    # [FIX-VMAF-NONE] 任一侧 VMAF 可能为 None（长视频超时/采样失败），必须逐项判空，
    # 否则 `abs(None - vm_soft)` 会抛 TypeError 让整个 G7 组执行中断（见 082057）。
    vm_soft = m_soft.get("vmaf")
    if vm_soft is None:
        # 单帧探测区分"无 libvmaf 滤镜"与"有滤镜但整段采样失败"
        if ctx.vmaf(enc_files["soft"], src, 1) is None:
            v.add("G7-5", "QUALITY", "VMAF 对齐检查（如可用）", Status.SKIP,
                  detail="ffmpeg 无 libvmaf，已用 PSNR/SSIM 判定")
        else:
            v.add("G7-5", "QUALITY", "VMAF 对齐检查（如可用）", Status.WARN,
                  detail=("libvmaf 可用但软编基准整段采样未取得分数"
                          "（超时/解码受限），已用 PSNR/SSIM 判定"))
    else:
        rows: List[str] = []
        worst = 0.0
        got = False
        for codec, _cid in pairs:
            val = resolved[codec][1]
            vm = measured.get(f"{codec}__c{val}", {}).get("vmaf")
            if vm is None:
                rows.append(f"{codec} cq{val}: VMAF n/a（采样失败，不参与判定）")
                continue
            got = True
            _d = abs(vm - vm_soft)
            worst = max(worst, _d)
            rows.append(f"{codec} cq{val}: VMAF {_fmt(vm, 2)}（Δ {_fmt(_d, 2)}）")
        # constqp 仅作参考行，不参与 worst（保持与原 PASS/WARN 语义一致）
        if m_qp.get("vmaf") is not None:
            rows.append(f"h264_nvenc constqp qp{qp}: VMAF {_fmt(m_qp['vmaf'], 2)}"
                        f"（Δ {_fmt(abs(m_qp['vmaf'] - vm_soft), 2)}，仅参考）")
        if not got:
            v.add("G7-5", "QUALITY", "VMAF 对齐检查（如可用）", Status.WARN,
                  detail="libvmaf 可用但硬编产物均未取得整段 VMAF，已用 PSNR/SSIM 判定",
                  evidence=rows)
        else:
            v.add("G7-5", "QUALITY", "VMAF 对齐检查（如可用）",
                  Status.PASS if worst <= TOL_VMAF else Status.WARN,
                  detail=f"最大偏差 {_fmt(worst)}（软编基准 VMAF {_fmt(vm_soft)}）",
                  evidence=rows)


# =============================================================================
# G8 avgBitRate 天花板（分辨率自适应）合理性
# =============================================================================

_BR_CASES = [(640, 360, 30, "360p30"), (854, 480, 30, "480p30"),
             (1280, 720, 30, "720p30"), (1280, 720, 60, "720p60"),
             (1920, 1080, 30, "1080p30"), (1920, 1080, 60, "1080p60"),
             (2560, 1440, 30, "1440p30"), (2560, 1440, 60, "1440p60"),
             (3840, 2160, 30, "2160p30"), (3840, 2160, 60, "2160p60")]


def _br_cap(width: int, height: int) -> int:
    """镜像分辨率分档上限（≤1080p→50M，>1080p→100M）。"""
    return (BR_CEILING_HIGH
            if (width > 0 and height > 0 and width * height > BR_ADAPT_PIXELS)
            else BR_CEILING)


def _br_rows() -> List[Tuple[str, float, float, bool]]:
    out = []
    for w, h, fps, name in _BR_CASES:
        raw = int(w * h * fps * 3.0)
        out.append((name, raw / 1e6, mirror_clamp_bitrate(raw, w, h) / 1e6,
                    raw > _br_cap(w, h)))
    return out


def _br_table() -> List[str]:
    return [f"{n:<9} raw={raw:7.1f}M → clamp={cl:5.1f}M"
            f"{'  【触发上限】' if b else ''}"
            for n, raw, cl, b in _br_rows()]


def group_bitrate(ctx: Ctx, v: Verifier) -> None:
    print("\n【G8】avgBitRate 天花板（分辨率自适应）合理性")

    # G8-1 公式表格（纯 CPU）
    rows = _br_table()
    v.add("G8-1", "BITRATE", "钳制公式分档（镜像公式）",
          Status.PASS,
          detail=(f"≤1080p: [{BR_FLOOR/1e6:.0f}M, {BR_CEILING/1e6:.0f}M]，"
                  f">1080p: 上限 {BR_CEILING_HIGH/1e6:.0f}M"),
          evidence=rows)

    # G8-2 真实实现与镜像一致（若可导入 nvenc_sdk）
    try:
        for p in (str(UTILS_DIR), str(EXTERNAL_DIR)):
            if p not in sys.path:
                sys.path.insert(0, p)
        from realesrgan_video.nvenc_sdk import _clamp_bitrate  # type: ignore
        mism = []
        for w, h, fps in [(320, 240, 15), (640, 360, 24), (1280, 720, 30),
                          (1920, 1080, 30), (1920, 1080, 60),
                          (2560, 1440, 30), (2560, 1440, 60),
                          (3840, 2160, 30), (3840, 2160, 60)]:
            raw = int(w * h * fps * 3.0)
            _got, _exp = _clamp_bitrate(raw, w, h), mirror_clamp_bitrate(raw, w, h)
            if _got != _exp:
                mism.append(f"{w}x{h}@{fps}: {_got} vs {_exp}")
        # 分辨率未知(0,0)时按基线处理（保持旧行为）
        if _clamp_bitrate(int(1920 * 1080 * 60 * 3.0)) != mirror_clamp_bitrate(
                int(1920 * 1080 * 60 * 3.0)):
            mism.append("未知分辨率回退基线不一致")
        v.add("G8-2", "BITRATE", "nvenc_sdk 真实钳制实现与公式一致",
              Status.PASS if not mism else Status.FAIL,
              detail="一致" if not mism else "; ".join(mism))
    except Exception as exc:  # noqa: BLE001
        v.add("G8-2", "BITRATE", "nvenc_sdk 真实钳制实现与公式一致", Status.SKIP,
              detail=f"nvenc_sdk 不可导入（{type(exc).__name__}），已由 G5-6 静态核验覆盖")

    # G8-3 低需求档位：天花板确实不生效（纯逻辑）
    low = [(640, 360, 24), (640, 360, 30), (854, 480, 24), (640, 480, 30)]
    binding = [(w, h, f) for w, h, f in low if int(w * h * f * 3.0) > _br_cap(w, h)]
    v.add("G8-3", "BITRATE", "低需求档位（≈480p30 及以下）天花板不生效",
          Status.PASS if not binding else Status.FAIL,
          detail="raw ≤ 上限，钳制不改变取值" if not binding else f"意外触发 {binding}")

    # G8-3b 天花板触发边界（分档）
    #   raw = w·h·fps·3 ≤ cap  ⇒  w·h·fps ≤ cap/3
    #   720p/1080p 用 50M 基线；1440p/2160p 用 100M 高档（raw 随像素线性增长）。
    thr = []
    for name, w, h in (("720p", 1280, 720), ("1080p", 1920, 1080),
                       ("1440p", 2560, 1440), ("2160p", 3840, 2160)):
        _cap = _br_cap(w, h)
        thr.append(f"{name}: 上限 {_cap/1e6:.0f}M，触发阈值 fps > "
                   f"{_cap / (w * h * 3.0):.1f}")
    v.add("G8-3b", "BITRATE", "天花板分档触发边界（≤1080p 50M / >1080p 100M）",
          Status.PASS,
          detail="分辨率自适应后，高分辨率获得更大头部空间，故 G8-4 的实测仍必要",
          evidence=thr)

    # G8-4 / G8-4H / G8-6 高需求档位实测（需 NVENC）
    if not ctx.gpu_mode():
        for cid, title in (("G8-4", "高熵内容：钳制前后画质对比"),
                           ("G8-4H", ">1080p 高档天花板绑定验证"),
                           ("G8-6", "证据：原样下发 raw 的 maxrate 为驱动非法值")):
            v.add(cid, "BITRATE", title, Status.SKIP,
                  detail="未启用 GPU 组（--quick / 无 NVENC）")
        return
    try:
        src = make_bitrate_source(ctx)
    except Exception as exc:  # noqa: BLE001
        v.add("G8-4", "BITRATE", "高熵内容：钳制前后画质对比",
              Status.FAIL, detail=f"素材准备失败：{exc}")
        return

    info = ctx.media_info(src)
    w, h, fps = info.get("width", 0), info.get("height", 0), info.get("fps", 0)
    n = info.get("nbf") or int(ctx.args.bfps * ctx.args.duration)
    raw = int(w * h * fps * 3.0)
    _cap = _br_cap(w, h)
    clamped = mirror_clamp_bitrate(raw, w, h)
    tq = load_quality_map(ctx).resolve_quality("h264_nvenc", default_ref=21)[1]

    # G8-6 独立证据：原样下发 raw 的 maxrate 是驱动非法值（旧控制臂缺陷）。
    # 放在编码前，确保即便后续 G8-4 异常也能留下这条根因证据。
    _raw_hi = int(BR_HIGH_W * BR_HIGH_H * BR_HIGH_FPS * 3.0)
    illegal, tail = _probe_raw_driver_illegal(
        ctx, BR_HIGH_W, BR_HIGH_H, BR_HIGH_FPS, _raw_hi, tq)
    v.add("G8-6", "BITRATE", "证据：原样下发 raw 的 maxrate 为驱动非法值",
          Status.PASS if illegal else Status.WARN,
          detail=(f"raw={_raw_hi/1e6:.1f}M → maxrate={2*_raw_hi/1e6:.1f}M 时 NVENC "
                  f"拒绝（{tail}），证实旧 G8-4 控制臂为脚本缺陷" if illegal
                  else f"maxrate={2*_raw_hi/1e6:.1f}M 竟被接受，需复核驱动上限假设"),
          evidence=[f"素材 {BR_HIGH_W}x{BR_HIGH_H}@{BR_HIGH_FPS}", f"stderr: {tail}"])

    out = ctx.tmp
    enc_to = _encode_timeout(ctx, w, h, n)
    # 两个编码保持串行：G8-5 断言"钳制不拖慢编码"，依赖独占计时，不可并行。
    # 控制臂用纯 CQ(-b:v 0) 测"内容自然码率"；原把 raw 当 target 会下发
    # maxrate=2*raw（1440p60=1.33Gbps / 4K60=2.99Gbps>INT32_MAX），驱动必拒（见 G8-6）。
    t_cl = t_un = 0.0
    enc_cl: Optional[Path] = None
    enc_un: Optional[Path] = None
    errs: List[str] = []
    try:
        t0 = time.time()
        enc_cl = enc_nvenc(ctx, src, out / "br_clamped.mp4", "h264_nvenc",
                           "vbr_hq", tq, avg_br=clamped, timeout=enc_to)
        t_cl = time.time() - t0
    except Exception as exc:  # noqa: BLE001
        errs.append(f"钳制臂(avg={clamped//1000}kbps)：{exc}")
    try:
        t0 = time.time()
        enc_un = enc_nvenc(ctx, src, out / "br_purecq.mp4", "h264_nvenc",
                           "vbr_hq", tq, avg_br=None, timeout=enc_to)
        t_un = time.time() - t0
    except Exception as exc:  # noqa: BLE001
        errs.append(f"控制臂(纯CQ,-b:v 0)：{exc}")

    if errs:
        v.add("G8-4", "BITRATE", "高熵内容：钳制前后画质对比", Status.FAIL,
              detail="；".join(errs),
              evidence=[f"素材 {w}x{h}@{fps:.0f}，raw 估算 {raw/1e6:.1f}M，"
                        f"钳制 {clamped/1e6:.1f}M，分档上限 {_cap/1e6:.0f}M",
                        f"单条编码超时 {enc_to}s（可用 --encode-timeout 放大）",
                        "控制臂已改纯 CQ，不再发送 maxrate=2*raw"])
        v.add("G8-5", "BITRATE", "钳制保留速度天花板语义（不拖慢编码）",
              Status.SKIP, detail="G8-4 编码失败，无计时数据")
        return

    # 度量可并行（两路互不依赖，且不参与 G8-5 计时）
    measured = _measure_many(ctx, {"clamped": enc_cl, "unclamped": enc_un}, src, n,
                             (w, h))
    m_cl = measured["clamped"]
    m_un = measured["unclamped"]
    ctx.metrics["bitrate_clamped"] = m_cl
    ctx.metrics["bitrate_unclamped"] = m_un

    natural = m_un.get("kbps", 0)
    binds = natural > _cap / 1000
    clamp_effective = m_cl.get("kbps", 0) < natural * 0.9
    d_psnr = (m_cl["psnr"] or 0) - (m_un["psnr"] or 0)
    d_ssim = (m_cl["ssim"] or 0) - (m_un["ssim"] or 0)

    ev = [
        f"素材 {w}x{h}@{fps:.0f}  raw 估算 {raw/1e6:.1f}M → 钳制 {clamped/1e6:.1f}M"
        f"（分档上限 {_cap/1e6:.0f}M）",
        f"未钳制（纯 CQ 自然码率）：{natural} kbps，PSNR {_fmt(m_un['psnr'])}，"
        f"SSIM {_fmt(m_un['ssim'], 4)}，体积 {m_un['size']/1e6:.1f}MB",
        f"已钳制：实际码率 {m_cl.get('kbps', 0)} kbps，PSNR {_fmt(m_cl['psnr'])}，"
        f"SSIM {_fmt(m_cl['ssim'], 4)}，体积 {m_cl['size']/1e6:.1f}MB",
        f"ΔPSNR={_fmt(d_psnr)} dB，ΔSSIM={_fmt(d_ssim, 4)}；"
        f"binds={binds}，clamp_effective={clamp_effective}",
        f"编码耗时：未钳制 {t_un:.1f}s / 已钳制 {t_cl:.1f}s",
    ]
    if binds and not clamp_effective:
        ev.append("注意：产品 maxBitRate=2×cap（高档=200M），自然码率落在 100~200M "
                  "时平均目标被钳制但硬上限未生效")
    ctx.metrics["bitrate_delta"] = {"d_psnr": d_psnr, "d_ssim": d_ssim,
                                    "binds": binds,
                                    "clamp_effective": clamp_effective,
                                    "raw_mbps": raw / 1e6,
                                    "clamped_mbps": clamped / 1e6,
                                    "natural_kbps": natural,
                                    "t_clamped_s": round(t_cl, 2),
                                    "t_unclamped_s": round(t_un, 2)}
    if not binds:
        v.add("G8-4", "BITRATE", "高熵内容：钳制前后画质对比", Status.PASS,
              detail=(f"自然码率 {natural} kbps < 上限 {_cap/1000:.0f} kbps，"
                      f"天花板未生效，无质量影响"),
              evidence=ev)
    else:
        ok = (d_psnr >= -TOL_PSNR_LOSS) and (d_ssim >= -TOL_SSIM_LOSS)
        warn = (d_psnr >= -2 * TOL_PSNR_LOSS)
        st = Status.PASS if ok else (Status.WARN if warn else Status.FAIL)
        v.add("G8-4", "BITRATE", "高熵内容：钳制前后画质对比", st,
              detail=("天花板生效但无明显质量损失" if ok
                      else f"钳制导致质量下降 ΔPSNR={_fmt(d_psnr)} dB"),
              evidence=ev)

    # G8-5 钳制不应显著拖慢编码（速度天花板初衷：防止无约束质量搜索）
    ratio_t = (t_cl / t_un) if t_un > 0 else 1.0
    st = Status.PASS if ratio_t <= 1.5 else Status.WARN
    v.add("G8-5", "BITRATE", "钳制保留速度天花板语义（不拖慢编码）", st,
          detail=f"耗时比 钳制/纯CQ = {ratio_t:.2f}×（≤1.5× 视为无拖慢）",
          evidence=[f"纯CQ {t_un:.1f}s，已钳制 {t_cl:.1f}s"])

    # ── G8-4H >1080p（100M 档）天花板绑定验证（4K60 高熵，纯 CQ 对照）────────
    if getattr(ctx.args, "no_br_high", False):
        v.add("G8-4H", "BITRATE", ">1080p 高档天花板绑定验证", Status.SKIP,
              detail="已用 --no-br-high 关闭")
    else:
        try:
            hsrc = make_bitrate_source_high(ctx)
        except Exception as exc:  # noqa: BLE001
            v.add("G8-4H", "BITRATE", ">1080p 高档天花板绑定验证",
                  Status.FAIL, detail=f"高档素材准备失败：{exc}")
        else:
            hi = ctx.media_info(hsrc)
            hw, hh = hi.get("width", 0), hi.get("height", 0)
            hfps = hi.get("fps", 0)
            hn = hi.get("nbf") or int(ctx.args.br_high_fps * ctx.args.br_high_dur)
            hcap = _br_cap(hw, hh)
            henc_to = _encode_timeout(ctx, hw, hh, hn)
            if hcap <= BR_CEILING:
                v.add("G8-4H", "BITRATE", ">1080p 高档天花板绑定验证",
                      Status.WARN,
                      detail=f"高档素材仅 {hw}x{hh}（未越过 1920x1080 阈值），"
                             f"cap={hcap/1e6:.0f}M 仍为基线档")
            else:
                try:
                    h_cl = enc_nvenc(ctx, hsrc, out / "brh_clamped.mp4",
                                     "h264_nvenc", "vbr_hq", tq, avg_br=hcap,
                                     timeout=henc_to)
                    h_un = enc_nvenc(ctx, hsrc, out / "brh_purecq.mp4",
                                     "h264_nvenc", "vbr_hq", tq, avg_br=None,
                                     timeout=henc_to)
                except Exception as exc:  # noqa: BLE001
                    v.add("G8-4H", "BITRATE", ">1080p 高档天花板绑定验证",
                          Status.FAIL, detail=f"高档编码失败：{exc}",
                          evidence=[f"素材 {hw}x{hh}@{hfps:.0f}，cap={hcap/1e6:.0f}M",
                                    f"单条编码超时 {henc_to}s"
                                    f"（--encode-timeout 可放大）"])
                else:
                    hm = _measure_many(ctx, {"clamped": h_cl, "purecq": h_un},
                                       hsrc, hn, (hw, hh),
                                       filters=("psnr", "ssim"))
                    m_c, m_n = hm["clamped"], hm["purecq"]
                    nat = m_n.get("kbps", 0)
                    hbinds = nat > hcap / 1000
                    hclamp_eff = m_c.get("kbps", 0) < nat * 0.9
                    hd_psnr = (m_c["psnr"] or 0) - (m_n["psnr"] or 0)
                    hd_ssim = (m_c["ssim"] or 0) - (m_n["ssim"] or 0)
                    hev = [
                        f"素材 {hw}x{hh}@{hfps:.0f}（{hn} 帧），cap={hcap/1e6:.0f}M",
                        f"纯 CQ 自然码率 {nat} kbps"
                        f"（raw 估算 {hw*hh*hfps*3/1e6:.1f}M）",
                        f"钳制 avg={hcap/1e6:.0f}M 后实际 {m_c.get('kbps', 0)} kbps，"
                        f"ΔPSNR={_fmt(hd_psnr)} dB，ΔSSIM={_fmt(hd_ssim, 4)}",
                    ]
                    ctx.metrics["bitrate_high"] = {
                        "w": hw, "h": hh, "fps": hfps, "cap_bps": hcap,
                        "natural_kbps": nat, "binds": hbinds,
                        "clamped_kbps": m_c.get("kbps", 0),
                        "clamp_effective": hclamp_eff,
                        "d_psnr": hd_psnr, "d_ssim": hd_ssim}
                    if not hbinds:
                        v.add("G8-4H", "BITRATE", ">1080p 高档天花板绑定验证",
                              Status.WARN,
                              detail=(f"素材自然码率 {nat} kbps ≤ 上限 "
                                      f"{hcap/1000:.0f} kbps，未能验证绑定；"
                                      f"请提高分辨率/熵或降低 CQ（--br-high-* 可调）"),
                              evidence=hev)
                    else:
                        ok = (hd_psnr >= -TOL_PSNR_LOSS
                              and hd_ssim >= -TOL_SSIM_LOSS)
                        if hclamp_eff:
                            _eff = "钳制实际压低码率"
                        else:
                            _eff = ("钳制未实际压低码率（avgBitRate 为软速度目标，"
                                    "maxBitRate=2×cap 才是硬上限）")
                            hev.append("注意：产品 maxBitRate=2×cap=200M，"
                                       "自然码率落在 100~200M 时硬上限未生效")
                        v.add("G8-4H", "BITRATE", ">1080p 高档天花板绑定验证",
                              Status.PASS if ok else Status.WARN,
                              detail=(f"自然码率 {nat} > 上限 {hcap/1000:.0f} kbps；"
                                      f"{_eff}，ΔPSNR={_fmt(hd_psnr)} dB"),
                              evidence=hev)


# =============================================================================
# G9 原「范围外/遗留项」闭环核对（保留 G9 编号，便于与历史报告逐项对照）
# =============================================================================

# 原 3 份报告中 G9-1..G9-5 的 5 项 SKIP，本次修复后逐项闭环；此处保留编号，
# 使新旧报告可直接对照（不再出现"G9 消失"的困惑）。
def group_scope_closure(ctx: Ctx, v: Verifier) -> None:
    print("\n【G9】原范围外项闭环核对（原 5 项 SKIP → 已修复）")
    vu = read_text(PROJECT_ROOT / "src/utils/video_utils.py")
    M = load_main_module(ctx)
    _dests = {a.dest for a in M._build_parser()._actions}
    try:
        _cfg = json.loads(read_text(CONFIG_JSON))
    except Exception:  # noqa: BLE001
        _cfg = {}
    _o = _cfg.get("output") or {}
    _ifr = ((_cfg.get("models") or {}).get("ifrnet") or {}).get("crf_ref")
    _esr = ((_cfg.get("models") or {}).get("realesrgan") or {}).get("crf_ref")

    # G9-1 原 P0-③a：merge_videos_by_codec 不再无条件发 -crf
    ok = (has(vu, r"def merge_videos_by_codec")
          and has(vu, r"_resolve_quality_args\(")
          and not has(vu, r"'-crf',\s*crf_str"))
    v.add("G9-1", "SCOPE", "P0-③a 合并环节已 codec-aware（原无条件 -crf 已消除）",
          Status.PASS if ok else Status.FAIL,
          detail="硬编走 -cq:v/-qp，不再被静默丢弃" if ok else "仍存在无条件 -crf")

    # G9-2 原 P1-③b：分段/合并质量倒挂消除
    ok = (_o.get("use_copy") is True and _o.get("crf_ref") == 21
          and _ifr == _esr == 21 and _o.get("crf") is None)
    v.add("G9-2", "SCOPE", "P1-③b 质量倒挂已消除（默认 copy 且基准统一 21）",
          Status.PASS if ok else Status.FAIL,
          detail=(f"output.use_copy={_o.get('use_copy')} / output.crf_ref={_o.get('crf_ref')} "
                  f"/ models.crf_ref={_ifr},{_esr}"))

    # G9-3 原 P1-①：normalize_video_timeline 去 crf=18 硬编码 + --split-* 入口
    _mfn = re.search(r"def normalize_video_timeline\(.*?(?=\ndef )", vu, re.S)
    _fn = _mfn.group(0) if _mfn else ""
    ok = (bool(_fn) and has(_fn, r"_resolve_quality_args\(")
          and not has(_fn, r"crf:\s*int\s*=\s*18")
          and "split_codec" in _dests and "split_crf_ref" in _dests)
    v.add("G9-3", "SCOPE", "P1-① 归一化去 crf=18 硬编码且建立 --split-* 入口",
          Status.PASS if ok else Status.FAIL,
          detail="默认基准 21 + --split-codec/--split-crf-ref" if ok else "未闭环")

    # G9-4 原 --output-*：环节③ 基准轴/CQ 入口
    ok = all(d in _dests for d in ("output_cq", "output_crf_ref", "output_cq_ref"))
    v.add("G9-4", "SCOPE", "环节③ --output-crf-ref/--output-cq-ref/--output-cq 已实现",
          Status.PASS if ok else Status.FAIL,
          detail="已实现" if ok else "缺失")

    # G9-5 原 --split-*：环节① 入口
    ok = all(d in _dests for d in ("split_codec", "split_crf_ref", "split_cq_ref",
                                   "split_preset"))
    v.add("G9-5", "SCOPE", "环节① --split-codec/--split-crf-ref/--split-cq-ref 已实现",
          Status.PASS if ok else Status.FAIL,
          detail="已实现" if ok else "缺失")


# =============================================================================
# G10 环节①/③ 契约（[QUALITY-UNIFY] 闭环：合并/归一化 codec-aware + CLI + 默认）
# =============================================================================

def _pipe_err_lines(msg: str) -> List[str]:
    return [ln.strip() for ln in msg.splitlines() if "·" in ln]


def group_pipeline(ctx: Ctx, v: Verifier) -> None:
    """把原 G9 的"范围外 SKIP"转为可判定的真实检查：合并/归一化 codec-aware。

    覆盖本次修复的环节①（归一化质量）与环节③（最终合并输出质量）：
    CLI 契约、量程校验、copy-by-default、无裸 -crf、去 crf=18 硬编码、CQ 偏移口。
    """
    print("\n【G10】环节①/③ 契约（合并/归一化 codec-aware）")
    M = load_main_module(ctx)

    # G10-1 新增 CLI 参数齐全
    parser = M._build_parser()
    dests = {a.dest for a in parser._actions}
    need = ["output_cq", "output_crf_ref", "output_cq_ref",
            "split_codec", "split_crf_ref", "split_cq_ref", "split_preset"]
    missing = [d for d in need if d not in dests]
    v.add("G10-1", "PIPE", "环节①/③ 新增 CLI 参数齐全",
          Status.PASS if not missing else Status.FAIL,
          detail="全部存在" if not missing else f"缺失 {missing}",
          evidence=["--output-cq / --output-crf-ref / --output-cq-ref",
                    "--split-codec / --split-crf-ref / --split-cq-ref / --split-preset"])

    # G10-2 output 字面量 vs 基准轴互斥
    ok, msg, _, _ = _cli_case(ctx, ["--output-crf", "18", "--output-crf-ref", "21"])
    v.add("G10-2", "PIPE", "output 字面量与基准轴互斥 → 拒绝",
          Status.PASS if (not ok and "互斥" in msg) else Status.FAIL,
          detail="已拒绝并提示互斥" if not ok else "未拒绝（校验缺失）")

    # G10-3 output 量程随编码器：libvpx-vp9 规范量程 0~63
    ok63, _, _, _ = _cli_case(ctx, ["--output-codec", "libvpx-vp9", "--output-crf", "63"])
    ok64, msg64, _, _ = _cli_case(ctx, ["--output-codec", "libvpx-vp9", "--output-crf", "64"])
    _e64 = " ".join(_pipe_err_lines(msg64))
    v.add("G10-3", "PIPE", "output 量程随编码器：libvpx-vp9 63 放行 / 64 拒绝",
          Status.PASS if (ok63 and not ok64 and "0~63" in _e64) else Status.FAIL,
          detail=f"63→{'ACCEPT' if ok63 else 'REJECT'} / 64→{'ACCEPT' if ok64 else 'REJECT'}",
          evidence=_pipe_err_lines(msg64))

    # G10-4 --output-codec copy 不得与质量/preset 同时给
    ok, msg, _, _ = _cli_case(ctx, ["--output-codec", "copy", "--output-crf", "18"])
    v.add("G10-4", "PIPE", "--output-codec copy 与质量参数互斥 → 拒绝",
          Status.PASS if not ok else Status.FAIL,
          detail="已拒绝" if not ok else "未拒绝")

    # G10-5 默认 copy-by-default + 消除质量倒挂（配置层）
    try:
        cfg_json = json.loads(read_text(CONFIG_JSON))
        _o = cfg_json["output"]
        _ifr_ref = cfg_json["models"]["ifrnet"].get("crf_ref")
        _esr_ref = cfg_json["models"]["realesrgan"].get("crf_ref")
        ok = (_o.get("use_copy") is True and _o.get("crf_ref") == 21
              and _ifr_ref == _esr_ref == 21 and _o.get("crf") is None)
        det = (f"output.use_copy={_o.get('use_copy')} / output.crf_ref={_o.get('crf_ref')} "
               f"/ output.crf={_o.get('crf')} / models.crf_ref={_ifr_ref},{_esr_ref}")
    except Exception as exc:  # noqa: BLE001
        ok, det = False, f"JSON 解析失败 {exc}"
    v.add("G10-5", "PIPE", "默认 copy 且消除质量倒挂（output.crf_ref=models.crf_ref=21）",
          Status.PASS if ok else Status.FAIL, detail=det)

    # G10-6 merge_videos_by_codec codec-aware（静态结构）
    vu = read_text(PROJECT_ROOT / "src/utils/video_utils.py")
    checks = [
        (r"from quality_map import resolve_quality", "导入 resolve_quality"),
        (r"def merge_videos_by_codec", "merge_videos_by_codec 存在"),
        (r"reencode:\s*Optional\[bool\]", "新增 reencode 路由参数"),
        (r"_resolve_quality\(", "重编码走 resolve_quality"),
        (r"'-c:v',\s*'copy'", "保留 copy 分支"),
    ]
    bad = [d for pat, d in checks if not has(vu, pat)]
    if has(vu, r"'-crf',\s*crf_str"):
        bad.append("残留无条件 '-crf', crf_str")
    v.add("G10-6", "PIPE", "merge_videos_by_codec 已 codec-aware（无裸 -crf）",
          Status.PASS if not bad else Status.FAIL,
          detail="全部命中" if not bad else f"缺失/残留 {bad}")

    # G10-7 normalize_video_timeline 去 crf=18 硬编码（仅在本函数体内断言，
    #        避免误伤 merge_videos/encode_video 等无调用方旧 API 的同名默认值）
    _mfn = re.search(r"def normalize_video_timeline\(.*?(?=\ndef )", vu, re.S)
    _fn = _mfn.group(0) if _mfn else ""
    ok = (bool(_fn) and has(_fn, r"crf:\s*Optional\[int\]\s*=\s*None")
          and not has(_fn, r"crf:\s*int\s*=\s*18"))
    v.add("G10-7", "PIPE", "normalize_video_timeline 去 crf=18 硬编码（默认基准 21）",
          Status.PASS if ok else Status.FAIL,
          detail="默认 None → 基准 21" if ok else "函数体内仍硬编码 crf=18")

    # G10-8 quality_map CQ_OFFSET 可调口存在且默认全 0（仍严格等于 §4.3）
    Q = load_quality_map(ctx)
    off = getattr(Q, "CQ_OFFSET", None)
    ok = isinstance(off, dict) and len(off) >= 1 and all(int(x) == 0 for x in off.values())
    v.add("G10-8", "PIPE", "quality_map.CQ_OFFSET 可调口存在且默认全 0（= §4.3）",
          Status.PASS if ok else Status.FAIL,
          detail=(f"{len(off)} 个编码器，全部为 0" if ok else f"CQ_OFFSET={off}"))

    # G10-9 旧版辅助 API 也统一走 _resolve_quality_args（不留裸 -crf 隐患）
    def _fn_body(name: str) -> str:
        _mm = re.search(rf"def {name}\(.*?(?=\ndef |\Z)", vu, re.S)
        return _mm.group(0) if _mm else ""

    _bad = []
    for _name in ("add_audio_to_video", "merge_videos", "encode_video"):
        _body = _fn_body(_name)
        if not _body:
            _bad.append(f"{_name}: 未定位到函数体")
            continue
        if not has(_body, r"_resolve_quality_args\("):
            _bad.append(f"{_name}: 未接 resolve")
        if has(_body, r"'-crf',\s*str\("):
            _bad.append(f"{_name}: 残留裸 -crf")
    v.add("G10-9", "PIPE", "旧版 add_audio_to_video/merge_videos/encode_video 已统一换算",
          Status.PASS if not _bad else Status.FAIL,
          detail=("三处均走 _resolve_quality_args，无裸 -crf"
                  if not _bad else "; ".join(_bad)))


# =============================================================================
# 报告渲染
# =============================================================================

def render_console(v: Verifier) -> None:
    c = v.counts()
    print("\n" + "=" * 78)
    print("  验证汇总")
    print("=" * 78)
    groups: Dict[str, List[Check]] = {}
    for chk in v.checks:
        groups.setdefault(chk.group, []).append(chk)
    for g, items in groups.items():
        cc = {s.value: 0 for s in Status}
        for it in items:
            cc[it.status.value] += 1
        print(f"  {g:<9} 共 {len(items):>2} 项  "
              f"✅{cc['PASS']:>2}  ❌{cc['FAIL']:>2}  ⚠️{cc['WARN']:>2}  ⏭️{cc['SKIP']:>2}")
    print("-" * 78)
    print(f"  合计：PASS={c['PASS']}  FAIL={c['FAIL']}  WARN={c['WARN']}  SKIP={c['SKIP']}")
    fails = [x for x in v.checks if x.status == Status.FAIL]
    if fails:
        print("\n  ❌ 未通过项：")
        for x in fails:
            print(f"     · [{x.cid}] {x.title} — {x.detail}")
    warns = [x for x in v.checks if x.status == Status.WARN]
    if warns:
        print("\n  ⚠️  需关注项：")
        for x in warns:
            print(f"     · [{x.cid}] {x.title} — {x.detail}")
    print("=" * 78)


def render_markdown(ctx: Ctx, v: Verifier, argv: Sequence[str]) -> str:
    c = v.counts()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L: List[str] = []
    L.append("# Video_Enhancement CRF/CQ 统一优化 · 实测验证报告\n")
    L.append(f"- 生成时间：{now}")
    L.append(f"- 主机平台：{sys.platform}")
    _cmd = " ".join(["python", "tests/verify_crf_cq_unification.py", *argv])
    L.append(f"- 命令：`{_cmd}`")
    L.append(f"- ffmpeg：`{ctx.ffmpeg or '未找到'}`")
    L.append(f"- NVENC：{'可用' if ctx.nvenc_ok() else '不可用/未启用'}"
             f"　libvmaf：{'可用' if 'libvmaf' in ctx.ffmpeg_filters() else '不可用'}")
    _j = int(getattr(ctx.args, "jobs", 0) or 0)
    _jdesc = ("自动探测系统资源（ParallelExecutor/auto workers）" if _j == 0
              else ("串行（--jobs 1）" if _j == 1 else f"指定 {_j} workers"))
    L.append(f"- 并行：{_jdesc}；G7 编码/度量、G8 度量并行，G8 编码计时保持串行")
    verdict = "❌ 存在 FAIL" if c["FAIL"] else ("⚠️ 有 WARN" if c["WARN"] else "✅ 全部通过")
    L.append(f"\n**总判定：{verdict}**（PASS={c['PASS']} / FAIL={c['FAIL']} "
             f"/ WARN={c['WARN']} / SKIP={c['SKIP']}）\n")

    L.append("## 一、结论总览\n")
    L.append("| 分组 | 说明 | 共 | ✅ | ❌ | ⚠️ | ⏭️ |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    desc = {
        "PREREQ": "前置条件", "TABLE": "换算表正确性", "RESOLVE": "resolve_quality 判定顺序",
        "CONSTQP": "CONSTQP 轴换算", "CLI": "CLI 互斥/量程/优先级",
        "STATIC": "代码落点静态核验", "EMIT": "下发命令捕获",
        "QUALITY": "画质统一性实测", "BITRATE": "avgBitRate 天花板合理性",
        "SCOPE": "原范围外项闭环核对", "PIPE": "环节①/③ 契约",
    }
    groups: Dict[str, List[Check]] = {}
    for chk in v.checks:
        groups.setdefault(chk.group, []).append(chk)
    for g in ("PREREQ", "TABLE", "RESOLVE", "CONSTQP", "CLI", "STATIC",
              "EMIT", "QUALITY", "BITRATE", "SCOPE", "PIPE"):
        items = groups.get(g)
        if not items:
            continue
        cc = {s.value: 0 for s in Status}
        for it in items:
            cc[it.status.value] += 1
        L.append(f"| {g} | {desc.get(g, g)} | {len(items)} | {cc['PASS']} | "
                 f"{cc['FAIL']} | {cc['WARN']} | {cc['SKIP']} |")
    L.append("")

    if c["FAIL"]:
        L.append("### 未通过项\n")
        L.append("| ID | 检查项 | 结论/证据 |")
        L.append("|---|---|---|")
        for x in v.checks:
            if x.status == Status.FAIL:
                L.append(f"| {x.cid} | {x.title} | {x.detail} |")
        L.append("")
    if c["WARN"]:
        L.append("### 需关注项\n")
        L.append("| ID | 检查项 | 结论/证据 |")
        L.append("|---|---|---|")
        for x in v.checks:
            if x.status == Status.WARN:
                L.append(f"| {x.cid} | {x.title} | {x.detail} |")
        L.append("")

    L.append("## 二、逐项明细\n")
    for g in ("PREREQ", "TABLE", "RESOLVE", "CONSTQP", "CLI", "STATIC",
              "EMIT", "QUALITY", "BITRATE", "SCOPE", "PIPE"):
        items = groups.get(g)
        if not items:
            continue
        L.append(f"### {g} — {desc.get(g, g)}\n")
        L.append("| ID | 检查项 | 结论 | 说明 |")
        L.append("|---|---|:--:|---|")
        for x in items:
            L.append(f"| {x.cid} | {x.title} | {_ICON[x.status].strip()} {x.status.value} "
                     f"| {_escape_md(x.detail)} |")
        L.append("")

    L.append("## 三、关键指标原始数据\n")
    L.append("```json")
    L.append(json.dumps(ctx.metrics, ensure_ascii=False, indent=2, default=str))
    L.append("```\n")

    L.append("## 四、统一参数矩阵（换算表实测输出）\n")
    L.append("| 编码器 | 基准 CRF 18 | 基准 CRF 21 | 基准 CRF 23 | 基准 CRF 26 |")
    L.append("|---|---|---|---|---|")
    try:
        Q = load_quality_map(ctx)
        for codec in ("libx264", "libx265", "h264_nvenc", "hevc_nvenc",
                      "av1_nvenc", "libsvtav1", "libvpx-vp9", "librav1e"):
            cells = []
            for ref in (18, 21, 23, 26):
                p, val, extra, _ = Q.resolve_quality(codec, crf_ref=ref)
                cells.append(f"`{p} {val}`" + (" +`-b:v 0`" if extra else ""))
            L.append(f"| {codec} | " + " | ".join(cells) + " |")
        L.append("\n> 由 `quality_map.resolve_quality(codec, crf_ref=N)` 实测生成，"
                 "即各环节实际发给 ffmpeg 的质量参数。\n")
    except Exception as exc:  # noqa: BLE001
        L.append(f"\n> 换算表不可用：{exc}\n")

    L.append("## 五、avgBitRate 天花板公式分档（分辨率自适应）\n")
    L.append("| 分辨率档 | raw 估算 | 钳制后 | 分档上限 | 是否触发上限 |")
    L.append("|---|---:|---:|---:|:--:|")
    for (n, raw, cl, b), (_w, _h, _f, _) in zip(_br_rows(), _BR_CASES):
        L.append(f"| {n} | {raw:.1f} Mbps | {cl:.1f} Mbps | "
                 f"{_br_cap(_w, _h)/1e6:.0f} Mbps | {'是' if b else '否'} |")
    L.append("\n> 公式：`raw = w × h × fps × 3.0`，"
             "`cap = 50M`（≤1080p）/ `100M`（>1080p），`clamp = min(max(raw, 5M), cap)`。"
             "两侧 `nvenc_sdk.py` 使用同一分档与公式（见 G5-6/G8-2）。")
    # 实测结论按真实 metrics 生成，避免硬编码随运行结果失真
    _bd = ctx.metrics.get("bitrate_delta") or {}
    _bh = ctx.metrics.get("bitrate_high") or {}
    if _bd:
        L.append(f"> G8-4 实测：raw 估算 {_bd.get('raw_mbps', 0):.1f}M → 钳制 "
                 f"{_bd.get('clamped_mbps', 0):.0f}M；纯 CQ 自然码率 "
                 f"{_bd.get('natural_kbps', 0)} kbps，binds={_bd.get('binds')}，"
                 f"clamp_effective={_bd.get('clamp_effective')}。")
    if _bh and _bh.get("binds"):
        _eff = ("钳制实际压低码率（clamp_effective=True）"
                if _bh.get("clamp_effective")
                else "钳制未实际压低码率（clamp_effective=False：avgBitRate 为软目标）")
        L.append(f"> G8-4H >1080p 绑定实测：{_bh['w']}x{_bh['h']}@{_bh['fps']:.0f} "
                 f"自然码率 {_bh['natural_kbps']} kbps > 上限 "
                 f"{_bh['cap_bps']/1000:.0f} kbps；{_eff}，"
                 f"ΔPSNR={_bh['d_psnr']:+.2f} dB、ΔSSIM={_bh['d_ssim']:+.4f}。")
    elif _bh:
        L.append(f"> G8-4H >1080p 绑定实测：自然码率 {_bh.get('natural_kbps', 0)} kbps "
                 f"未越上限 {_bh.get('cap_bps', 0)/1000:.0f} kbps（**未绑定**）。")
    else:
        L.append("> G8-4/G8-4H 未执行（SKIP）：请在具备 NVENC 的机器上加 `--gpu` 重跑。")
    L.append("")

    L.append("## 六、结论解释与后续\n")
    L.append("- **换算正确性**：G1/G2/G3 覆盖换算表、判定顺序与 CONSTQP 轴；"
             "G7 用真实编码证明换算后的 CQ 值比修复前的朴素值更接近软编基准。")
    L.append("- **下发一致性**：G5 静态核验两侧后端 + G6 捕获真实命令形状，"
             "确认 `-cq:v`（CQ 轴）与 `-qp`（CONSTQP 轴）不再混用。")
    L.append("- **avgBitRate 天花板（50M/100M 分档）**：G8 给出分档表与实测；"
             "G8-4 验证基线档、G8-4H 验证 >1080p 高档是否真正绑定，"
             "绑定后以 ΔPSNR/ΔSSIM 量化质量影响。")
    L.append("- 若 G7/G8 未执行（SKIP），请在具备 NVENC 的机器上加 `--gpu` 重跑。\n")
    return "\n".join(L)


def _escape_md(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ")


# =============================================================================
# main
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Video_Enhancement CRF/CQ 统一优化 —— Linux/GPU 实测验证脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("运行方式")[-1] if __doc__ else None,
    )
    p.add_argument("--gpu", action="store_true",
                   help="强制执行 GPU 组（NVENC 不可用时对应项 FAIL）")
    p.add_argument("--no-gpu", action="store_true", help="禁止执行 GPU 组")
    p.add_argument("--quick", action="store_true",
                   help="等价于 --no-gpu：仅静态/逻辑/CLI 核验")
    p.add_argument("--source", metavar="PATH",
                   help="画质对比用的真实素材（默认合成 testsrc2 无损片段）")
    p.add_argument("--bitrate-source", metavar="PATH", dest="bitrate_source",
                   help="码率天花板测试用的 1080p 高熵素材（默认合成）")
    p.add_argument("--duration", type=float, default=3.0, help="合成素材时长（秒，默认 3）")
    p.add_argument("--fps", type=int, default=30, help="画质素材帧率（默认 30）")
    p.add_argument("--qw", type=int, default=640, help="画质素材宽（默认 640）")
    p.add_argument("--qh", type=int, default=360, help="画质素材高（默认 360）")
    p.add_argument("--bfps", type=int, default=60, help="码率素材帧率（默认 60）")
    p.add_argument("--bw", type=int, default=1920, help="码率素材宽（默认 1920）")
    p.add_argument("--bh", type=int, default=1080, help="码率素材高（默认 1080）")
    p.add_argument("--encode-timeout", type=int, default=0, dest="encode_timeout",
                   help="单条编码命令超时秒；0=按帧数自适应（下限 --timeout，上限 3600）")
    p.add_argument("--no-br-high", action="store_true", dest="no_br_high",
                   help="跳过 G8-4H >1080p 高档绑定验证（省 4K 编码/度量时间）")
    p.add_argument("--br-high-w", type=int, default=BR_HIGH_W,
                   help="高档绑定素材宽（默认 3840）")
    p.add_argument("--br-high-h", type=int, default=BR_HIGH_H,
                   help="高档绑定素材高（默认 2160）")
    p.add_argument("--br-high-fps", type=int, default=BR_HIGH_FPS,
                   help="高档绑定素材帧率（默认 60）")
    p.add_argument("--br-high-dur", type=float, default=BR_HIGH_DUR,
                   help="高档绑定素材时长秒（默认 1.0）")
    p.add_argument("--timeout", type=int, default=600, help="单条外部命令超时秒数")
    p.add_argument("--jobs", type=int, default=0, metavar="N",
                   help="并行 worker 数：0=自动探测系统资源（默认），1=串行（复现旧行为），"
                        "N=指定上限。GPU 编码另有会话闸门，不受此值放宽")
    p.add_argument("--report", metavar="PATH",
                   help="Markdown 报告输出路径（默认 verification_report/…）")
    p.add_argument("--json", metavar="PATH", dest="json_path",
                   help="JSON 结果输出路径（默认 verification_report/…）")
    p.add_argument("--keep-temp", action="store_true", help="保留临时编码产物")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    args = build_parser().parse_args(argv)

    print("=" * 78)
    print("  Video_Enhancement · CRF/CQ 统一优化实测验证")
    print(f"  项目根：{PROJECT_ROOT}")
    print(f"  时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)

    ctx = Ctx(args)
    ctx.tmp = Path(tempfile.mkdtemp(prefix="ve_crfcq_"))
    v = Verifier()

    try:
        v.guard("G0", "PREREQ", "前置条件组", lambda: (group_prereq(ctx, v), v.checks[-1])[1])
        for cid, grp, title, fn in (
            ("G1", "TABLE", "换算表正确性组", group_table),
            ("G2", "RESOLVE", "resolve_quality 判定顺序组", group_resolve),
            ("G3", "CONSTQP", "CONSTQP 轴组", group_constqp),
            ("G4", "CLI", "CLI 契约组", group_cli),
            ("G5", "STATIC", "代码落点组", group_static),
            ("G6", "EMIT", "命令下发捕获组", group_emit),
            ("G7", "QUALITY", "画质统一性组", group_quality),
            ("G8", "BITRATE", "码率天花板组", group_bitrate),
            ("G9", "SCOPE", "原范围外项闭环核对组", group_scope_closure),
            ("G10", "PIPE", "环节①/③ 契约组", group_pipeline),
        ):
            try:
                fn(ctx, v)
            except Exception as exc:  # noqa: BLE001
                import traceback
                v.add(cid, grp, f"{title} 执行中断", Status.FAIL,
                      detail=f"{type(exc).__name__}: {exc}",
                      evidence=[traceback.format_exc().splitlines()[-1]])

        render_console(v)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rep_dir = PROJECT_ROOT / "verification_report"
        rep_dir.mkdir(parents=True, exist_ok=True)
        report_path = Path(args.report) if args.report else \
            rep_dir / f"CRF_CQ统一验证报告_{ts}.md"
        json_path = Path(args.json_path) if args.json_path else \
            rep_dir / f"CRF_CQ统一验证结果_{ts}.json"

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_markdown(ctx, v, argv), encoding="utf-8")
        json_path.write_text(json.dumps({
            "generated_at": datetime.now().isoformat(),
            "project_root": str(PROJECT_ROOT),
            "argv": argv,
            "nvenc": ctx.nvenc_ok(),
            "vmaf": "libvmaf" in ctx.ffmpeg_filters(),
            "counts": v.counts(),
            "checks": [c.to_dict() for c in v.checks],
            "metrics": ctx.metrics,
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        print(f"\n  报告：{report_path}")
        print(f"  数据：{json_path}")
        return 1 if v.counts()["FAIL"] else 0
    finally:
        if not args.keep_temp:
            shutil.rmtree(ctx.tmp, ignore_errors=True)
        else:
            print(f"  临时目录（保留）：{ctx.tmp}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断。")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"\n❌ 验证脚本自身异常：{exc}")
        traceback.print_exc()
        sys.exit(2)
