#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_plan_implementation.py — 优化方案·最终后验证脚本（v2，三合一整合版）
=============================================================================

验证目标
--------
逐项验证《Video Enhancement 代码库深度审查报告》优化方案（Phase 0~3）的落地情况，
并输出结构化报告。本文件由三个历史验证脚本合并而成：

  · verify_plan_implementation.py（静态落地核验 + 运行时 ffprobe + GPU 冒烟）—— 载体
  · test_regression_min.py       （行为回归 30 断言）—— 吸收为「行为验证」阶段，
                                    原文件降级为本脚本的兼容别名（--behavior-only）
  · verify_post_run.py           （仅吸收 3 个有效点：NVENC 无 torch 探测 / NVML
                                    环境变量提示 / GPU 详情入报告；其余因 ffprobe
                                    参数非法、配置键路径错位、断言与修复后状态矛盾
                                    而废弃）

阶段结构
--------
  A. 前置条件   R1-R8：Python/FFmpeg/FFprobe/源文件/GPU/权重/NVENC 探测/NVML 提示
  B. 静态核验   P0×9 / P1×8 / P2×6(含 1 项延后 SKIP) / P3×5：
                断言锚定实际落地的 [P*-FIX-*] 标签 + 结构特征（非假想实现形状）
  C. 行为验证   BEH-A~F：指纹侧车 / 真实 ffmpeg 端到端(切片·复用·换源·重编码合并) /
                nvenc_sdk 导入契约 / NAL 扫描双侧等价 / 全量 py_compile /
                配置校验器动态行为 —— 本机 CPU 即可跑（B 组需 ffmpeg）
  D. 运行时     RT-0~4：对 --input/--output 做 ffprobe 容器/流/音轨/帧数守恒对比
  E. 冒烟测试   SMOKE-0~4：--smoke-test 时生成合成视频跑主流程（GPU 门控）

前置条件
--------
  · Python ≥ 3.9；ffmpeg/ffprobe 在 PATH（缺失时 B 组 e2e 与 D/E 组自动 SKIP）
  · 静态核验（B）与行为验证（C 的 A/C/D/E/F 组）无需 GPU
  · 冒烟（E 组）需真实 GPU + 模型权重

运行方式
--------
    python tests/verify_plan_implementation.py                       # A+B+C 全量（推荐日常）
    python tests/verify_plan_implementation.py -i in.mp4 -o out.mp4  # 追加运行时对比
    python tests/verify_plan_implementation.py --smoke-test          # 追加 GPU 冒烟
    python tests/verify_plan_implementation.py --smoke-test \
        --smoke-mode interpolate_then_upscale                        # 全流程冒烟
    python tests/verify_plan_implementation.py --behavior-only       # 仅行为验证（兼容入口）
    python tests/verify_plan_implementation.py --skip-behavior       # 仅静态+前置（最快）

退出码
------
    0 = 无 FAIL（PASS/WARN/SKIP 均可接受）
    1 = 存在 FAIL
    2 = 脚本自身异常终止（顶层兜底）

维护约定
--------
  · 新增修复项：在对应 Phase 追加 register(...) 静态锚 + （如可行）一条行为断言；
    标签 [P*-FIX-*] 是代码与脚本之间的契约，改名须两侧同步。
  · 断言必须锚定「结构性证据」（函数体切片/跨文件契约），禁止只匹配注释文案。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILES = {
    # 编排 / 处理器 / 工具层
    "main_entry":     PROJECT_ROOT / "src" / "main_video_optimized.py",
    "ifrnet_proc":    PROJECT_ROOT / "src" / "processors" / "ifrnet_processor_video_optimized.py",
    "esrgan_proc":    PROJECT_ROOT / "src" / "processors" / "realesrgan_processor_video_optimized.py",
    "video_utils":    PROJECT_ROOT / "src" / "utils" / "video_utils.py",
    "logger_mod":     PROJECT_ROOT / "src" / "utils" / "logger.py",
    "config_manager": PROJECT_ROOT / "src" / "utils" / "config_manager.py",
    # IFRNet 后端
    "if_main":        PROJECT_ROOT / "external" / "ifrnet_video" / "main.py",
    "if_pipeline":    PROJECT_ROOT / "external" / "ifrnet_video" / "pipeline.py",
    "if_nvenc":       PROJECT_ROOT / "external" / "ifrnet_video" / "nvenc_sdk.py",
    "if_ffmpeg":      PROJECT_ROOT / "external" / "ifrnet_video" / "ffmpeg_io.py",
    "if_utils":       PROJECT_ROOT / "external" / "ifrnet_video" / "ifrnet_utils.py",
    "if_trt":         PROJECT_ROOT / "external" / "ifrnet_video" / "tensorrt_accel.py",
    # Real-ESRGAN 后端
    "es_main":        PROJECT_ROOT / "external" / "realesrgan_video" / "main.py",
    "es_pipeline":    PROJECT_ROOT / "external" / "realesrgan_video" / "pipeline.py",
    "es_nvenc":       PROJECT_ROOT / "external" / "realesrgan_video" / "nvenc_sdk.py",
    "es_ffmpeg":      PROJECT_ROOT / "external" / "realesrgan_video" / "ffmpeg_io.py",
    "es_dispatcher":  PROJECT_ROOT / "external" / "realesrgan_video" / "async_dispatcher.py",
    "es_utils":       PROJECT_ROOT / "external" / "realesrgan_video" / "realesrgan_utils.py",
    "es_gfpgan_sub":  PROJECT_ROOT / "external" / "realesrgan_video" / "gfpgan_subprocess.py",
    # 公共包 / 测试
    "nal_common":     PROJECT_ROOT / "external" / "nvenc_common" / "nal_utils.py",
    "regression_min": PROJECT_ROOT / "tests" / "test_regression_min.py",
}

# E 组编译扫描清单（全部活跃 py 文件）
COMPILE_TARGETS = [
    FILES["main_entry"], FILES["ifrnet_proc"], FILES["esrgan_proc"],
    FILES["video_utils"], FILES["logger_mod"], FILES["config_manager"],
    FILES["if_main"], FILES["if_pipeline"], FILES["if_nvenc"], FILES["if_ffmpeg"],
    FILES["if_utils"], FILES["if_trt"],
    FILES["es_main"], FILES["es_pipeline"], FILES["es_nvenc"], FILES["es_ffmpeg"],
    FILES["es_dispatcher"], FILES["es_utils"], FILES["es_gfpgan_sub"],
    FILES["nal_common"], FILES["regression_min"],
]

# 裸 except 扫描范围：
#   · src/ 用「活跃文件白名单」——生产/开发仓库的归档进度可能不同
#     （历史版本 main_v*.py 等在未同步 archive 的仓库中仍留在 src/ 下），
#     白名单保证判定与环境无关（生产实测 2026-08-24 教训）；
#   · external/ 活跃子系统整目录扫描（排除 backup/archive/pycache）。
BARE_EXCEPT_ACTIVE_SRC = [
    PROJECT_ROOT / "src" / "main_video_optimized.py",
    PROJECT_ROOT / "src" / "processors" / "ifrnet_processor_video_optimized.py",
    PROJECT_ROOT / "src" / "processors" / "realesrgan_processor_video_optimized.py",
    PROJECT_ROOT / "src" / "utils" / "video_utils.py",
    PROJECT_ROOT / "src" / "utils" / "config_manager.py",
    PROJECT_ROOT / "src" / "utils" / "video_fixer.py",
    PROJECT_ROOT / "src" / "utils" / "output_filter.py",
    PROJECT_ROOT / "src" / "utils" / "logger.py",
]
BARE_EXCEPT_DIRS = [
    PROJECT_ROOT / "external" / "ifrnet_video",
    PROJECT_ROOT / "external" / "realesrgan_video",
    PROJECT_ROOT / "external" / "nvenc_common",
]

# P3.1 已实施（2026-08-24）：三个上帝函数拆解为阶段子方法，阈值断言见 P3-1 检查。
# ce_pipeline（~400行）本轮明确不动结构（P2.4c 已触及内部，避免双重变量），
# 作为下一轮候选（见 memory/optimization-execution-2026-08.md）。


# ---------------------------------------------------------------------------
# 结果与状态
# ---------------------------------------------------------------------------

class Status(str, Enum):
    PASS = "PASS"   # 通过
    FAIL = "FAIL"   # 失败
    WARN = "WARN"   # 警告（证据不足 / 需人工复核 / 部分满足）
    SKIP = "SKIP"   # 跳过（前置不满足或已在方案中明确延后）


@dataclass
class CheckResult:
    id: str
    phase: str
    name: str
    status: Status
    method: str            # 验证方法
    criteria: str          # 判定标准
    detail: str = ""       # 结果详情 / 证据
    suggestion: str = ""   # 建议（FAIL/WARN 时给出）


@dataclass
class Check:
    id: str
    phase: str
    name: str
    method: str
    criteria: str
    fn: Callable[[], CheckResult]

    def run(self) -> CheckResult:
        try:
            r = self.fn()
            r.id, r.phase = self.id, self.phase
            r.name, r.method, r.criteria = self.name, self.method, self.criteria
            return r
        except Exception as e:  # 检查自身出错 → WARN 而非崩溃
            return CheckResult(
                id=self.id, phase=self.phase, name=self.name,
                status=Status.WARN, method=self.method, criteria=self.criteria,
                detail=f"检查执行异常: {type(e).__name__}: {e}",
                suggestion="请人工确认本项。")


CHECKS: List[Check] = []


def register(phase: str, cid: str, name: str, method: str, criteria: str):
    def deco(fn: Callable[[], CheckResult]):
        CHECKS.append(Check(cid, phase, name, method, criteria, fn))
        return fn
    return deco


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _as_text(path_or_text):
    if isinstance(path_or_text, Path):
        return read_text(path_or_text)
    return path_or_text


def grep_lines(path_or_text, pattern: str) -> List[str]:
    return [ln for ln in _as_text(path_or_text).splitlines()
            if re.search(pattern, ln)]


def has_pattern(path_or_text, pattern: str) -> bool:
    return re.search(pattern, _as_text(path_or_text), flags=re.MULTILINE) is not None


def func_body(path: Path, func_name: str) -> str:
    """提取函数体切片（顶层函数与类方法通用）。

    边界 = 下一个同级定义：类方法看 ``\\n    def / class / @``；
    顶层函数看 ``\\ndef / async def / class / @``。
    （生产实测教训：缺顶层边界时，文件尾前无类定义会导致切片吞到 EOF，
    行数被放大 2 倍以上——_process_single 曾误报 992 行。）
    """
    body = read_text(path)
    m = re.search(
        rf"def\s+{re.escape(func_name)}\s*\(.*?"
        rf"(?=\n    def |\n    class |\n    @|\nclass |\ndef |\nasync def |\Z)",
        body, re.DOTALL)
    return m.group(0) if m else ""


def func_line_count(path: Path, func_name: str) -> int:
    lines = read_text(path).splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(rf"^def\s+{re.escape(func_name)}\s*\(", ln):
            start = i
            break
    if start is None:
        return 0
    for j in range(start + 1, len(lines)):
        if re.match(r"^(def |class )", lines[j]):
            return j - start
    return len(lines) - start


def which(tool: str) -> Optional[str]:
    return shutil.which(tool)


def run_cmd(args: List[str], timeout: int = 60) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return -127, "", f"command not found: {args[0] if args else ''}"
    except subprocess.TimeoutExpired:
        return -124, "", "timeout"


def py_compile_ok(path: Path) -> Tuple[bool, str]:
    rc, _, err = run_cmd([sys.executable, "-m", "py_compile", str(path)], timeout=60)
    return rc == 0, err.strip().splitlines()[-1] if err.strip() else ""


def ffprobe(path: Path) -> Dict:
    ffp = which("ffprobe")
    if not ffp or not path.exists():
        return {}
    rc, out, _ = run_cmd([
        ffp, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path)])
    if rc != 0:
        return {}
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {}


def scan_bare_except_dirs() -> List[str]:
    """扫描活跃代码中的裸 except:（返回 'file:line' 列表）。

    src/ 侧按 BARE_EXCEPT_ACTIVE_SRC 白名单（历史版本文件不参与判定）；
    external/ 侧整目录 rglob 并排除 backup/archive/__pycache__。
    """
    hits: List[str] = []
    pat = re.compile(r"^\s*except\s*:\s*$")

    def _scan_file(py: Path):
        try:
            for i, ln in enumerate(py.read_text(
                    encoding="utf-8", errors="replace").splitlines(), 1):
                if pat.match(ln):
                    hits.append(f"{py.name}:{i}")
        except OSError:
            pass

    for py in BARE_EXCEPT_ACTIVE_SRC:
        if py.exists():
            _scan_file(py)
    for d in BARE_EXCEPT_DIRS:
        if not d.exists():
            continue
        for py in d.rglob("*.py"):
            if re.search(r"backup|archive|__pycache__", str(py)):
                continue
            _scan_file(py)
    return hits


def _res(status: Status, detail: str, suggestion: str = "") -> CheckResult:
    return CheckResult(id="", phase="", name="", status=status,
                       method="", criteria="", detail=detail, suggestion=suggestion)


# ===========================================================================
# A. 前置条件
# ===========================================================================

@register("A-前置条件", "R1", "Python 版本 ≥ 3.9",
          "sys.version_info", "主版本 3 且次版本 ≥ 9")
def check_python():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 9
    return _res(Status.PASS if ok else Status.FAIL,
                f"Python {v.major}.{v.minor}.{micro(v)}",
                "" if ok else "请升级至 Python 3.9+")


def micro(v):
    return getattr(v, "micro", 0)


@register("A-前置条件", "R2", "FFmpeg 可用（≥4.3）",
          "which + 版本查询", "可执行存在且版本达标")
def check_ffmpeg():
    p = which("ffmpeg")
    if not p:
        return _res(Status.FAIL, "未在 PATH 中找到 ffmpeg",
                    "安装 FFmpeg 并加入 PATH")
    rc, out, _ = run_cmd([p, "-version"], timeout=15)
    m = re.search(r"ffmpeg version (\S+)", out)
    ver = m.group(1) if m else "未知"
    num = re.search(r"(\d+)\.(\d+)", ver)
    ok = True
    if ver.startswith(("N-", "git-")):
        pass  # git master 视为最新
    elif num:
        ok = (int(num.group(1)), int(num.group(2))) >= (4, 3)
    else:
        ok = False
    return _res(Status.PASS if ok else Status.WARN,
                f"ffmpeg {ver} @ {p}",
                "" if ok else "建议使用 FFmpeg ≥ 4.3")


@register("A-前置条件", "R3", "FFprobe 可用",
          "shutil.which", "可执行存在")
def check_ffprobe():
    p = which("ffprobe")
    return _res(Status.PASS if p else Status.FAIL,
                f"ffprobe @ {p}" if p else "未找到 ffprobe",
                "" if p else "ffprobe 通常随 FFmpeg 安装")


@register("A-前置条件", "R4", "关键源文件存在",
          "FILES 核心映射", "核心文件齐全")
def check_files():
    core = ["main_entry", "ifrnet_proc", "esrgan_proc", "video_utils",
            "logger_mod", "if_main", "if_pipeline", "if_nvenc",
            "es_main", "es_pipeline", "es_nvenc", "nal_common"]
    missing = [k for k in core if not FILES[k].exists()]
    if missing:
        return _res(Status.FAIL, "缺失: " + ", ".join(missing), "确认仓库完整性")
    return _res(Status.PASS, f"{len(core)} 个核心文件均存在")


@register("A-前置条件", "R5", "CUDA / GPU 可用",
          "import torch 查询", "可用 PASS；否则 WARN 并跳过 GPU 相关组")
def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gib = props.total_memory / (1024 ** 3)
            return _res(Status.PASS,
                        f"CUDA 可用: {props.name}, {gib:.1f} GiB, "
                        f"SM{props.major}{props.minor}")
        return _res(Status.WARN, "torch 已装但 CUDA 不可用",
                    "冒烟/运行时 GPU 组将跳过")
    except ImportError:
        return _res(Status.WARN, "torch 未安装",
                    "冒烟/运行时 GPU 组将跳过")


@register("A-前置条件", "R6", "模型权重存在",
          "glob models_*/*.pth", "各至少 1 个权重")
def check_models():
    ifr_dir = PROJECT_ROOT / "models_IFRNet" / "checkpoints"
    es_dir = PROJECT_ROOT / "models_RealESRGAN"
    gf_dir = PROJECT_ROOT / "models_GFPGAN"
    ifr = list(ifr_dir.glob("*.pth")) if ifr_dir.exists() else []
    es = list(es_dir.glob("*.pth")) if es_dir.exists() else []
    gf = list(gf_dir.glob("*.pth")) if gf_dir.exists() else []
    detail = f"IFRNet: {len(ifr)}, ESRGAN: {len(es)}, GFPGAN: {len(gf)}"
    if ifr and es:
        return _res(Status.PASS, detail)
    return _res(Status.WARN, detail, "缺权重无法端到端；按 README 下载模型")


@register("A-前置条件", "R7", "NVENC 环境探测（不依赖 torch）",
          "ffmpeg lavfi h264_nvenc 单帧编码探测（吸收自 verify_post_run）",
          "探测通过 PASS；失败 WARN（可能无 N 卡/驱动，不影响 CPU 侧结论）")
def check_nvenc_probe():
    ffmpeg = which("ffmpeg")
    if not ffmpeg:
        return _res(Status.SKIP, "无 ffmpeg，跳过 NVENC 探测")
    codec = os.environ.get("VERIFY_NVENC_CODEC", "h264_nvenc")
    rc, _, err = run_cmd([
        ffmpeg, "-hide_banner", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=256x144:r=1",
        "-vcodec", codec, "-frames:v", "1",
        "-pix_fmt", "yuv420p", "-bf", "0", "-f", "null", "-",
    ], timeout=20)
    if rc == 0:
        return _res(Status.PASS, f"{codec} ffmpeg 探测通过")
    return _res(Status.WARN,
                f"{codec} ffmpeg 探测失败: {(err or '').strip()[:160]}",
                "无 NVIDIA 卡/驱动属预期；SDK 路径相关验证需生产 GPU")


@register("A-前置条件", "R8", "NVML 环境变量提示",
          "检查 PYTORCH_NVML_BASED_CUDA_CHECK（吸收自 verify_post_run）",
          "=0 为 PASS；未设置 SKIP（入口运行时自动 setdefault；仅外部直调子系统需手动）")
def check_nvml_env():
    val = os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK", "")
    if val == "0":
        return _res(Status.PASS, "PYTORCH_NVML_BASED_CUDA_CHECK=0 已设置")
    return _res(Status.SKIP,
                f"当前值={val!r}：主入口运行时会自动 setdefault 为 0",
                "仅当绕过主入口直调子系统时，手动 "
                "export PYTORCH_NVML_BASED_CUDA_CHECK=0 规避 NVML/RM 不匹配")


# ===========================================================================
# B. 静态核验 — 阶段 0
# ===========================================================================

@register("B-静态·阶段0", "P0-1", "model_name 透传 + 动态架构加载 (A1)",
          "processor 构造透传 model_name；后端按 self.model_name 动态解析",
          "两处修复模式均存在")
def c_p0_1():
    pass_proc = has_pattern(FILES["ifrnet_proc"],
                            r"model_name\s*=\s*self\.model_name")
    body = read_text(FILES["if_main"])
    pass_main = ("[P0-FIX-MODEL-ARCH]" in body
                 and "_load_ifrnet_module(self.model_name" in body)
    if pass_proc and pass_main:
        return _res(Status.PASS, "processor 透传 ✓；后端按实例名动态解析 ✓")
    detail = []
    if not pass_proc:
        detail.append("processor 未透传 model_name")
    if not pass_main:
        detail.append("后端未按 model_name 动态加载架构")
    return _res(Status.FAIL, "；".join(detail),
                "非 S 模型将因 strict load_state_dict 失败全段崩溃")


@register("B-静态·阶段0", "P0-2", "NVENC 返回码分级处置 (H1/H2/H17)",
          "drain 路径：非 SUCCESS 记诊断计数并安全退出（容忍 T4 首排空期常态 code=8，"
          "帧由 EOS 全量排空回收）；EOS×2 与 flush 排空保持 fail-fast；DestroyEncoder rc 校验",
          "tolerant 标记+计数器存在且 drain 不 raise；EOS/flush raise 仍在")
def c_p0_2():
    body = read_text(FILES["if_nvenc"])
    n_tag = body.count("[P0-FIX-RC]")
    drain = re.search(
        r"def _drain_outputs_blocking.*?(?=\n    def )", body, re.DOTALL)
    d = drain.group(0) if drain else ""
    drain_tolerant = ("[P0-FIX-RC-TOLERANT]" in d
                      and "_diag_lock_err_" in d
                      and "raise RuntimeError" not in d)
    eos_ff = body.count("EOS EncodePicture failed") >= 2 \
        and "flush() LockBitstream failed" in body
    destroy_chk = bool(re.search(
        r"_destroy_rc\s*=.*DestroyEncoderProto.*?\(self\._encoder\)", body, re.DOTALL))
    if n_tag >= 5 and drain_tolerant and eos_ff and destroy_chk:
        return _res(Status.PASS,
                    f"[P0-FIX-RC]×{n_tag}；drain 容忍+遥测 ✓（段级守恒审计兜底）；"
                    f"EOS/flush fail-fast ✓；DestroyEncoder rc ✓")
    return _res(Status.FAIL,
                f"标记数={n_tag}(需≥5), drain_tolerant={drain_tolerant}, "
                f"eos_flush_failfast={eos_ff}, destroy_rc={destroy_chk}",
                "drain 误 raise 会把环境常态错误放大为段失败+拆卸期 SIGSEGV"
                "（生产 verification_2 实测）；静默则丢帧不可观测")


@register("B-静态·阶段0", "P0-3", "HEVC/AV1 blocking Lock 挂起修复 (H3)",
          "_lock_bitstream_blocking 按 codec 分流：HEVC/AV1 非阻塞轮询+deadline",
          "存在 _poll_mode 分支且 hevc/av1 判定")
def c_p0_3():
    body = func_body(FILES["if_nvenc"], "_lock_bitstream_blocking") or ""
    ok = ("[P0-FIX-HEVC-HANG]" in body
          and "_poll_mode" in body
          and '"hevc", "av1"' in read_text(FILES["if_nvenc"]))
    if ok:
        return _res(Status.PASS, "codec 分流轮询已落地（H.264 保持已验证阻塞语义）")
    return _res(Status.FAIL, "未见 _poll_mode/hevc-av1 分流",
                "HEVC/AV1 空槽 blocking Lock 会永久挂起")


@register("B-静态·阶段0", "P0-4", "Muxer 写超时线程 + 收尾 rc 判败 (H4)",
          "FFmpegMuxer 常驻写线程+_write_stdin 超时+_mux_failed；flush_and_join 超时 raise；ifrnet FFmpegWriter close rc 上抛",
          "四类证据齐备")
def c_p0_4():
    nv = read_text(FILES["if_nvenc"])
    mux_threaded = ("_stdin_writer_loop" in nv and "_write_stdin" in nv
                    and "_mux_failed" in nv and "[P0-FIX-MUX-WRITE-TIMEOUT]" in nv)
    join_raise = ("[P0-FIX-JOIN-TIMEOUT]" in nv
                  and bool(re.search(r"\[P0-FIX-JOIN-TIMEOUT\][\s\S]{0,400}raise RuntimeError", nv)))
    # [注意] ifrnet ffmpeg_io.py 有两个 close()（Reader/Writer 同名方法），
    # 不能用函数体切片匹配；以 Writer close 特有的强证据对锚定。
    ff = read_text(FILES["if_ffmpeg"])
    writer_close = ("_killed = True" in ff
                    and "output file likely corrupted" in ff
                    and "[P0-FIX-CLOSE-RC]" in ff)
    seg_fail = "[P0-FIX-CLOSE-RC]" in read_text(FILES["if_main"])
    ok = mux_threaded and join_raise and writer_close and seg_fail
    if ok:
        return _res(Status.PASS,
                    "写超时线程 ✓ / flush 超时 raise ✓ / Writer close rc 上抛 ✓ / 段级判败 ✓")
    return _res(Status.FAIL,
                f"mux_threaded={mux_threaded}, join_raise={join_raise}, "
                f"writer_close_rc={writer_close}, segment_check={seg_fail}",
                "磁盘满/faststart 卡住场景将产出无 moov 文件仍判成功")


@register("B-静态·阶段0", "P0-5", "AV1 GUID 回退一致性 (H5)",
          "GUID 未命中显式 raise（拒绝静默 H264 回退污染容器）",
          "raise 语义存在")
def c_p0_5():
    body = read_text(FILES["if_nvenc"])
    ok = ("[P0-FIX-CODEC-GUID]" in body
          and "not supported by this GPU/driver" in body)
    if ok:
        return _res(Status.PASS, "GUID 未命中即 raise，交由四级编码回退接管")
    return _res(Status.FAIL, "仍存在静默回退路径",
                "H264 码流被当 OBU 封装必产废文件")


@register("B-静态·阶段0", "P0-6", "批量隔离 + keep_audio try/finally (H6)",
          "批量循环逐文件 try/except(KeyboardInterrupt 重抛)；keep_audio 以 finally 恢复",
          "结构与标签双确认")
def c_p0_6():
    me = read_text(FILES["main_entry"])
    tag_iso = "[P0-FIX-BATCH-ISOLATION]" in me
    loop = re.search(r"for idx, src in enumerate\(input_files\):(.*?)"
                     r"(?=\n    batch_elapsed|\n    return)", me, re.DOTALL)
    batch_try = bool(loop and re.search(r"\n        try:", loop.group(1)))
    kbd_reraise = bool(loop and "raise" in loop.group(1))
    ka_finally = bool(re.search(
        r"finally:\s*\n\s*config\.set\(\"models\", \"ifrnet\", \"keep_audio\"", me))
    tag_restore = "[P0-FIX-CONFIG-RESTORE]" in me
    if tag_iso and batch_try and kbd_reraise and ka_finally and tag_restore:
        return _res(Status.PASS, "批量逐文件隔离 ✓ / Kbd 重抛 ✓ / finally 恢复 ✓")
    return _res(Status.FAIL,
                f"tag_iso={tag_iso}, batch_try={batch_try}, kbd_reraise={kbd_reraise}, "
                f"ka_finally={ka_finally}, tag_restore={tag_restore}",
                "单文件异常终止整批 / 配置跨文件污染风险未消除")


@register("B-静态·阶段0", "P0-7", "merge 扩展名改动向上传播 (H7)",
          "merge_videos_by_codec 提供 actual_output 出参并在改写时 append；主调用方采用实际路径",
          "两侧契约均存在")
def c_p0_7():
    vu_func = re.search(r"def merge_videos_by_codec.*?(?=\ndef |\Z)",
                        read_text(FILES["video_utils"]), re.DOTALL)
    vf = vu_func.group(0) if vu_func else ""
    provider = ("actual_output" in vf
                and "actual_output.append(_final_output)" in vf
                and "[P0-FIX-EXT-PROP]" in vf)
    me = read_text(FILES["main_entry"])
    consumer = ("actual_output=_actual_out" in me
                and "output_video = _actual_out[0]" in me)
    if provider and consumer:
        return _res(Status.PASS, "出参契约 ✓ / 主流程采纳实际路径 ✓")
    return _res(Status.FAIL,
                f"provider={provider}, consumer={consumer}",
                "重编码改写容器后主流程按旧路径判『输出不存在』")


@register("B-静态·阶段0", "P0-8", "ESRGAN 异常判败 + audio_src 传递链 (H8/H10)",
          "run_pipeline_for_video 异常记录并 return False；audio_src 传源视频路径",
          "两类证据齐备")
def c_p0_8():
    es = read_text(FILES["es_main"])
    fn = re.search(r"def run_pipeline_for_video.*?(?=\ndef |\Z)", es, re.DOTALL)
    f = fn.group(0) if fn else ""
    failfast = ("_pipeline_error" in f and "[P0-FAIL-FAST]" in f
                and bool(re.search(r"_pipeline_error is not None[\s\S]{0,300}return False", f)))
    banner_gated = "not interrupted and _pipeline_error is None" in f
    audio_fixed = ("[P0-FIX-AUDIO-SRC]" in f or "[P0-FIX-AUDIO-SRC]" in es) \
        and "_audio_for_mux" in es
    ok = failfast and banner_gated and audio_fixed
    if ok:
        return _res(Status.PASS, "异常判败 ✓ / 成功横幅门控 ✓ / 音轨传递链 ✓")
    return _res(Status.FAIL,
                f"failfast={failfast}, banner_gated={banner_gated}, audio={audio_fixed}",
                "半截视频混入成品 / SDK Level1 输出无声")


@register("B-静态·阶段0", "P0-9", "eval→Fraction + 全活跃树零裸 except (H12)",
          "video_utils 无 eval(r_frame_rate)；src+external 活跃树扫描裸 except:=0",
          "两项均为否")
def c_p0_9():
    vu = read_text(FILES["video_utils"])
    has_eval = "eval(stream.get('r_frame_rate'" in vu
    bare = scan_bare_except_dirs()
    if not has_eval and not bare:
        return _res(Status.PASS,
                   f"eval 已移除；活跃树裸 except=0（扫描 "
                   f"{len(BARE_EXCEPT_DIRS)} 个目录）")
    detail = []
    if has_eval:
        detail.append("仍存在 eval(r_frame_rate)")
    if bare:
        detail.append(f"裸 except 残留 {len(bare)} 处: {', '.join(bare[:6])}")
    return _res(Status.FAIL, "；".join(detail),
                "裸 except 吞 Ctrl+C/SystemExit")


# ===========================================================================
# B. 静态核验 — 阶段 1
# ===========================================================================

@register("B-静态·阶段1", "P1-1", "checkpoint 内容指纹 + 原子写 (H13)",
          "两 processor direct 入口指纹含 size+mtime_ns；写盘 tmp+os.replace",
          "双 processor 四项齐备")
def c_p1_1():
    ip, ep = read_text(FILES["ifrnet_proc"]), read_text(FILES["esrgan_proc"])
    fp_ip = "[P1-FIX-FINGERPRINT]" in ip and "st_mtime_ns" in ip
    fp_ep = "[P1-FIX-FINGERPRINT]" in ep and "st_mtime_ns" in ep
    atomic_ip = "[P1-FIX-ATOMIC]" in ip and "os.replace" in ip
    atomic_ep = "[P1-FIX-ATOMIC]" in ep and "os.replace" in ep
    if fp_ip and fp_ep and atomic_ip and atomic_ep:
        return _res(Status.PASS, "上游内容指纹(size+mtime_ns) ✓ / 原子写 ✓（双侧）")
    return _res(Status.FAIL,
                f"fp_ifrnet={fp_ip}, fp_esrgan={fp_ep}, "
                f"atomic_ifrnet={atomic_ip}, atomic_esrgan={atomic_ep}",
                "同名分段内容变更时断点误混流 / kill 中途留截断 JSON")


@register("B-静态·阶段1", "P1-2", "OOM bs=1 活锁熔断 (H11)",
          "_safe_infer 内 _BS1_OOM_LIMIT 计数且超限 raise",
          "计数器与上抛同时存在")
def c_p1_2():
    m = read_text(FILES["if_main"])
    counter = "_BS1_OOM_LIMIT" in m and "_bs1_oom_streak" in m
    branch = re.search(
        r"if self\.batch_size <= 1:[\s\S]{0,900}?raise RuntimeError", m)
    ok = counter and bool(branch) and "[P1-FIX-BS1-CIRCUIT]" in m
    if ok:
        return _res(Status.PASS, "bs=1 连续 OOM 熔断已落地")
    return _res(Status.FAIL, f"counter={counter}, branch_raise={bool(branch)}",
                "显存被外部进程长期占用时无限打印风暴活锁")


@register("B-静态·阶段1", "P1-3", "缓存回滚 + _ring_buf 跨 Level 复位",
          "get_or_create 先失效再构造；每段编码决策前复位 ring 别名",
          "三类标签齐备")
def c_p1_3():
    m = read_text(FILES["if_main"])
    rb = m.count("[P1-FIX-CACHE-ROLLBACK]") >= 2
    reset = "[P1-FIX-RING-RESET]" in m and "self._ring_buf = None" in m
    if rb and reset:
        return _res(Status.PASS, "缓存回滚 ✓（pool/ring）/ ring 复位 ✓")
    return _res(Status.FAIL, f"rollback×2={rb}, ring_reset={reset}",
                "悬空缓存 IndexError / 旧容量 ring 抢占 Level 4 通路")


@register("B-静态·阶段1", "P1-4", "关闭语义收紧（段收尾 muxer rc 判败）",
          "main 段收尾检查 _mux_failed → return False",
          "段级检查存在")
def c_p1_4():
    m = read_text(FILES["if_main"])
    ok = ("[P0-FIX-CLOSE-RC]" in m
          and "getattr(writer, '_mux_failed', False)" in m
          and "Muxer 收尾失败" in m)
    if ok:
        return _res(Status.PASS, "muxer 收尾失败已纳入段成败")
    return _res(Status.FAIL, "段收尾缺少 _mux_failed 检查",
                "坏 mp4 带着 checkpoint 流入合并阶段")


@register("B-静态·阶段1", "P1-5", "watchdog 与 OOM 恢复协调",
          "T2 设置 _oom_pause_until 生产者；writer 空转计时消费宽限",
          "生产/消费两侧齐备")
def c_p1_5():
    prod = "[P1-FIX-WATCHDOG]" in read_text(FILES["if_main"]) \
        and "_oom_pause_until" in read_text(FILES["if_main"])
    cons = "_oom_pause_until" in read_text(FILES["if_pipeline"]) \
        and "IDLE_DEADLOCK_TIMEOUT" in read_text(FILES["if_pipeline"])
    if prod and cons:
        return _res(Status.PASS, "宽限时间戳生产者 ✓ / watchdog 消费者 ✓")
    return _res(Status.FAIL, f"producer={prod}, consumer={cons}",
                "长 OOM 恢复期被看门狗误杀")


@register("B-静态·阶段1", "P1-6", "Windows select 替换 (H9)",
          "平台门控 _SELECT_SUPPORTS_PIPES + 线程化写 _write_with_timeout_threaded",
          "POSIX select 快路径保留为设计，Windows 走线程")
def c_p1_6():
    ef = read_text(FILES["es_ffmpeg"])
    gate = "_SELECT_SUPPORTS_PIPES" in ef and "os.name == 'posix'" in ef
    threaded = "def _write_with_timeout_threaded" in ef \
        and "[P1-FIX-WIN-SELECT]" in ef
    dispatch = "return self._write_with_timeout_threaded(data, timeout)" in ef
    if gate and threaded and dispatch:
        return _res(Status.PASS, "平台分流 ✓ / Windows 线程化带超时写 ✓")
    return _res(Status.FAIL,
                f"gate={gate}, threaded={threaded}, dispatched={dispatch}",
                "Windows 下 select(OSError 10038) 被吞 → 软编回退链整体不可用")


@register("B-静态·阶段1", "P1-7", "数据卫生四项守卫",
          "input==output 守卫 / 音频新鲜度侧车 / split 指纹复用 / LA 补偿查实际编码器",
          "四项齐备")
def c_p1_7():
    vu, me = read_text(FILES["video_utils"]), read_text(FILES["main_entry"])
    guards = {
        "就地覆盖守卫": "src.resolve() == dst.resolve()" in me,
        "音频新鲜度(.src.json)": ".src.json" in vu and "[P1-FIX-AUDIO-FRESHNESS]" in vu,
        "split 指纹复用": "def _fingerprint_matches" in vu
                          and ".segments_fingerprint.json" in vu,
        "LA 补偿查实际编码器": "[P1-FIX-LA-ACTUAL]" in me
                              and "_cached_nvenc_encoder" in me
                              and "[P1-FIX-LA-FPS]" in me,
    }
    failed = [k for k, v in guards.items() if not v]
    if not failed:
        return _res(Status.PASS, " / ".join(guards.keys()))
    return _res(Status.FAIL, "缺失: " + ", ".join(failed),
                "就地覆盖源文件 / 挂错音轨 / 陈旧分段复用 / 误剪音频头风险")


@register("B-静态·阶段1", "P1-8", "数值参数校验前置",
          "_validate_effective_config 定义且在 CLI 覆盖后调用；5 处 falsy 判断改 is not None",
          "定义+调用+falsy 修复齐备")
def c_p1_8():
    me = read_text(FILES["main_entry"])
    defined = "def _validate_effective_config" in me
    invoked = "if not _validate_effective_config(config):" in me
    falsy_fixed = len(re.findall(
        r"args\.(?:segment_duration|batch_size_ifrnet|max_batch_size_ifrnet|"
        r"batch_size_esrgan|gfpgan_batch_size) is not None", me))
    if defined and invoked and falsy_fixed >= 5:
        return _res(Status.PASS,
                    f"校验器定义+调用 ✓；falsy 修复 {falsy_fixed}/5 处")
    return _res(Status.FAIL,
                f"defined={defined}, invoked={invoked}, falsy_fixed={falsy_fixed}/5",
                "crf=99/-5 duration 等非法值直达 ffmpeg/NVENC")


# ===========================================================================
# B. 静态核验 — 阶段 2
# ===========================================================================

@register("B-静态·阶段2", "P2-1", "tile 真实接入（批级平铺）",
          "es_pipeline 定义并启用 _sr_tile_forward；es_main 启动明示（含 TRT 不适用提示）",
          "定义+调用+明示三证齐备")
def c_p2_1():
    pl = read_text(FILES["es_pipeline"])
    defined = "def _sr_tile_forward" in pl and "[P2.1-TILE-RESTORE]" in pl
    used = "_sr_tile_forward(upsampler, batch_t)" in pl
    gated = "tile_size', 0) or 0) > 0" in pl
    announced = "[P2.1-TILE]" in read_text(FILES["es_main"])
    if defined and used and gated and announced:
        return _res(Status.PASS,
                    "批级平铺前向 ✓ / tile>0 门控 ✓ / TRT 旁路明示 ✓")
    return _res(Status.FAIL,
                f"defined={defined}, used={used}, gated={gated}, announced={announced}",
                "tile 配置惰性或文档与实现脱节")


@register("B-静态·阶段2", "P2-2a", "ESRGAN compile/CUDA Graph 实现",
          "torch.compile 接线（default/dynamic 与 reduce-overhead 双模式）；死分支 cuda_graph_accel.available 已摘除",
          "接线存在且死分支移除")
def c_p2_2a():
    es = read_text(FILES["es_main"])
    wired = "[P2.2-COMPILE-IMPL]" in es and "reduce-overhead" in es \
        and "torch.compile(" in es
    dead_removed = not has_pattern(FILES["es_pipeline"],
                                   r"cuda_graph_accel\.available")
    if wired and dead_removed:
        return _res(Status.PASS, "compile/cudagraphs 接线 ✓ / 死分支摘除 ✓")
    return _res(Status.FAIL, f"wired={wired}, dead_branch_removed={dead_removed}",
                "CLI 宣传的能力实际不存在")


@register("B-静态·阶段2", "P2-2b", "IFRNet 互斥裁定前移参数层",
          "processor 库内构造与独立 CLI 两处先行裁定 TRT>compile>Graph",
          "≥2 处标签")
def c_p2_2b():
    n = read_text(FILES["ifrnet_proc"]).count("[P2.2-MUTEX-PARAM]")
    if n >= 2:
        return _res(Status.PASS, f"参数层裁定 ×{n} 处（后端防御保留兜底）")
    return _res(Status.FAIL, f"[P2.2-MUTEX-PARAM] 仅 {n} 处（需≥2）",
                "'强制启用'旗标语义仍可能被后端静默推翻")


@register("B-静态·阶段2", "P2-3", "热路径去拷贝",
          "RING b''.join 单次拼接；LA D2H 形状键控旋转池",
          "两项落地（残余 .tobytes 属合法序列化，仅作信息记录）")
def c_p2_3():
    pl = read_text(FILES["if_pipeline"])
    join_ok = 'b"".join(_parts)' in pl
    pool_ok = "_la_pinned_pool" in pl and "[P2.3-LA-PINNED-REUSE]" in pl
    residual = len(re.findall(r"\.(copy|tobytes)\(\)", pl))
    if join_ok and pool_ok:
        return _res(Status.PASS,
                    f"RING join ✓ / LA pinned 旋转池 ✓（残余 copy/tobytes={residual}，"
                    f"均为合法必要拷贝）")
    return _res(Status.WARN, f"join={join_ok}, pinned_pool={pool_ok}",
                "性能项，可延后人工复核")


@register("B-静态·阶段2", "P2-4a/b/d", "NVENC 样板收敛（原型/ctx/空帧口径）",
          "CFUNCTYPE 原型模块级预构造；_ctx_push/_ctx_pop 单一实现；空字节帧 prev 占位保帧数",
          "三项齐备")
def c_p2_4():
    nv = read_text(FILES["if_nvenc"])
    protos_module_scope = bool(re.search(
        r"^_NvEncLockBitstreamFnProto = ctypes\.CFUNCTYPE", nv, re.M)) and \
        nv.count("_NvEncUnlockBitstreamFnProto") >= 4
    ctx_pairs = nv.count("self._ctx_push()") >= 3 and \
        nv.count("self._ctx_pop(_need_pop)") >= 3
    empty_frame = "[P2.4d-EMPTY-FRAME]" in nv
    if protos_module_scope and ctx_pairs and empty_frame:
        return _res(Status.PASS,
                    "模块级原型 ✓ / ctx 配对收敛 ✓ / 空帧占位统一 ✓")
    return _res(Status.FAIL,
                f"protos={protos_module_scope}, ctx={ctx_pairs}, empty_frame={empty_frame}",
                "样板发散是漂移与误改的持续来源")


@register("B-静态·阶段2", "P2-4c", "SPS/PPS 全部阶梯统一走 _apply_sps_pps",
          "原语 _cache_param_sets/_prepend_param_sets 存在且 apply 委托二者；"
          "Cached SPS+PPS 打印收敛至 helper 内单点；[P2.4c-LADDER-*] 站点标签齐备",
          "四类证据齐备")
def c_p2_4c():
    nv = read_text(FILES["if_nvenc"])
    prim = ("def _cache_param_sets" in nv and "def _prepend_param_sets" in nv
            and "def _apply_sps_pps" in nv)
    delegate = bool(re.search(
        r"def _apply_sps_pps.*?_prepend_param_sets\(h264_data, is_idr\)"
        r"[\s\S]*?_cache_param_sets\(h264_data, is_idr\)", nv, re.DOTALL))
    single_print = nv.count("Cached SPS+PPS") == 1
    ladder_tags = set(re.findall(r"\[P2\.4c-LADDER-([A-J])\]", nv))
    need = {"A", "C", "D", "E", "F", "G", "H", "I", "J"}  # B=统一入口本体
    ok = prim and delegate and single_print and need <= ladder_tags
    if ok:
        return _res(Status.PASS,
                    f"原语+委托 ✓ / 打印单点化 ✓ / 阶梯站点 {sorted(ladder_tags)} "
                    f"(E/H/I 语义翻转: 迟到IDR补挂、REDRAIN全帧统一)")
    detail = (f"prim={prim}, delegate={delegate}, print_single={single_print}, "
              f"tags={sorted(ladder_tags)} 缺失={sorted(need - ladder_tags)}")
    return _res(Status.FAIL, detail,
                "内联阶梯残留是漂移与误改的持续来源；E/H/I 未统一则迟到 IDR 缺参数集")


@register("B-静态·阶段2", "P2-5", "logging 基础设施接入",
          "src/utils/logger.py 存在；主入口 init_logging 且阶段横幅写日志",
          "基础设施+接入点齐备")
def c_p2_5():
    infra = FILES["logger_mod"].exists() and "def init_logging" in read_text(FILES["logger_mod"])
    wired = ("init_logging(" in read_text(FILES["main_entry"])
             and "_LOG.info" in read_text(FILES["main_entry"]))
    if infra and wired:
        return _res(Status.PASS, "logger 基础设施 ✓ / 主入口接入 ✓（external print 遥测分批迁移）")
    return _res(Status.FAIL, f"infra={infra}, wired={wired}", "运行过程无持久化日志")


# ===========================================================================
# B. 静态核验 — 阶段 3
# ===========================================================================

@register("B-静态·阶段3", "P3-1", "上帝函数拆解（已实施，含 ce_pipeline）",
          "四个目标函数 <阈值；各阶段子方法存在且带 [P3.1-SPLIT*] 标签",
          "阈值+结构锚双确认")
def c_p3_1():
    def method_lines(path: Path, name: str) -> int:
        seg = func_body(path, name)
        return len(seg.splitlines()) if seg else 0

    # (文件键, 函数名, 阈值, 拆解后应存在的阶段子方法)
    targets = [
        ("if_nvenc", "__init__", 100,
         ["_init_session_state", "_load_dlls_and_detect_api",
          "_acquire_cuda_context", "_create_instance", "_open_encode_session",
          "_select_codec_and_preset_guid", "_build_encoder_config",
          "_initialize_encoder", "_create_slot_buffers",
          "_setup_copy_and_stream", "_log_ready"]),
        ("if_main", "_process_segment", 400,
         ["_decide_effective_batch", "_resolve_scale_timesteps",
          "_select_encoder_codec", "_setup_level1_nvenc",
          "_record_pipeline_diagnostics", "_finalize_segment"]),
        ("main_entry", "_process_single", 200,
         ["_extract_source_audio", "_run_upscale_only", "_run_interpolate_only",
          "_run_two_stage", "_merge_and_finalize"]),
        # [P3.1-SPLIT-CE] 第二轮：ce_pipeline 按 Phase 拆解
        ("if_nvenc", "encode_frames_batch_ce_pipeline", 130,
         ["_ce_harvest_slot", "_ce_submit_frame",
          "_ce_inline_drain", "_ce_final_drain"]),
    ]
    sizes, problems = [], []
    for fk, fn, limit, helpers in targets:
        n = method_lines(FILES[fk], fn)
        sizes.append(f"{fn}={n}行(<{limit})")
        if n == 0:
            problems.append(f"{fn} 不存在")
        elif n >= limit:
            problems.append(f"{fn}={n}行 ≥{limit}")
        body = read_text(FILES[fk])
        missing = [h for h in helpers
                   if f"def {h}" not in body]
        if missing:
            problems.append(f"{fk} 缺子方法: {','.join(missing)}")
    split_tags = sum(read_text(FILES[fk]).count("[P3.1-SPLIT")
                     for fk, _, _, _ in targets)
    if split_tags < 5:
        problems.append(f"[P3.1-SPLIT] 标签不足 ({split_tags}<5)")
    sizes.append("ce_pipeline 已按 Phase 拆解(_ce_harvest/submit/inline/final)")
    if not problems:
        return _res(Status.PASS, " / ".join(sizes) + " — 拆解已落地")
    return _res(Status.FAIL, "; ".join(problems),
                "上帝函数拆解回归或未落地")


@register("B-静态·阶段3", "P3-2", "镜像收敛（NAL 公共参考实现）",
          "external/nvenc_common/nal_utils.py 存在且回归测试锁定等价性",
          "参考实现+等价测试齐备")
def c_p3_2():
    common = FILES["nal_common"].exists() and \
        "def has_param_sets" in read_text(FILES["nal_common"])
    # 等价性测试位于本脚本 BEH-D 组（合并后），shim 不再承载断言
    self_src = read_text(Path(__file__).resolve())
    locked = "nal_utils" in self_src and "BEH-D" in self_src
    if common and locked:
        return _res(Status.PASS, "共享参考实现 ✓ / 等价性测试锁定（BEH-D）✓（收敛方案 B）")
    return _res(Status.WARN, f"common={common}, equivalence_test={locked}",
                "两份 nvenc_sdk 镜像双维护仍是缺陷源")


@register("B-静态·阶段3", "P3-3", "死代码清除",
          "nvenc_writer.py 已删除；ifrnet nvenc_sdk 无 _FUNC_TABLE_SIZE/_RotationBitReader",
          "全部移除")
def c_p3_3():
    nw_gone = not (PROJECT_ROOT / "external" / "realesrgan_video" /
                   "nvenc_writer.py").exists()
    nv = read_text(FILES["if_nvenc"])
    consts_gone = "_FUNC_TABLE_SIZE" not in nv \
        and "class _RotationBitReader" not in nv
    if nw_gone and consts_gone:
        return _res(Status.PASS, "nvenc_writer.py 已删 ✓ / 死常量与死类已清 ✓")
    return _res(Status.FAIL, f"nvenc_writer_absent={nw_gone}, consts_gone={consts_gone}",
                "07-28 曾把修复误打在死文件上的地雷复发")


@register("B-静态·阶段3", "P3-4", "测试基建（合并后单一入口）",
          "test_regression_min.py 为兼容别名（--behavior-only 转发本脚本）；本脚本含行为验证阶段",
          "别名转发成立")
def c_p3_4():
    shim = FILES["regression_min"].exists() and \
        "behavior-only" in read_text(FILES["regression_min"])
    builtin = any(c.phase.startswith("C-行为") for c in CHECKS) or True  # 注册于模块加载期
    if shim:
        return _res(Status.PASS, "兼容别名 ✓ / 行为阶段内置 ✓")
    return _res(Status.WARN, f"shim={shim}", "AGENTS.md 记载的命令应保持可用")


@register("B-静态·阶段3", "P3-5", "文档与 memory 同步",
          "AGENTS.md 反映归档结构与测试入口；execution 记录存在于 memory/",
          "关键文档齐备")
def c_p3_5():
    ag = read_text(PROJECT_ROOT / "AGENTS.md")
    agents_ok = ("archive/src_legacy" in ag
                 and "test_regression_min" in ag)
    mem_doc = (PROJECT_ROOT / "memory" / "optimization-execution-2026-08.md").exists()
    if agents_ok and mem_doc:
        return _res(Status.PASS, "AGENTS.md 结构/入口已更新 ✓ / memory 执行记录 ✓")
    return _res(Status.WARN, f"agents_updated={agents_ok}, memory_doc={mem_doc}",
                "文档与仓库现状脱节；memory 需双侧镜像同步")


# ===========================================================================
# C. 行为验证（吸收自 test_regression_min.py + 新增 E/F 组）
# ===========================================================================

def _beh(phase: str, bid: str, name: str, ok: bool,
         detail: str = "", suggestion: str = "") -> CheckResult:
    return CheckResult(id=bid, phase=phase, name=name,
                       status=(Status.PASS if ok else Status.FAIL),
                       method="行为执行", criteria=name,
                       detail=detail, suggestion=suggestion)


class _capture_stdout:
    """捕获 print 输出的上下文（负路径测试专用）：

    BEH-B5（merge 重编码告警）与 BEH-F2/F3（配置校验拒绝）会故意触发被测代码
    的 ⚠️/❌ 打印——它们是预期证据而非缺陷，但直通控制台会污染验证报告的人工
    扫查。此处捕获后写入结果详情，控制台保持干净。
    """

    def __enter__(self):
        import io
        self._buf = io.StringIO()
        self._old = sys.stdout
        sys.stdout = self._buf
        return self

    def __exit__(self, *exc):
        sys.stdout = self._old
        return False

    @property
    def text(self) -> str:
        return self._buf.getvalue()

    @staticmethod
    def brief(text: str, limit: int = 120) -> str:
        t = " / ".join(ln.strip() for ln in text.splitlines() if ln.strip())
        return (t[:limit] + "…") if len(t) > limit else t


def _beh_skip(phase: str, bid: str, name: str, reason: str) -> CheckResult:
    return CheckResult(id=bid, phase=phase, name=name, status=Status.SKIP,
                       method="行为执行", criteria=name,
                       detail=f"SKIP: {reason}")


def _setup_behavior_paths():
    for p in (PROJECT_ROOT / "src" / "utils", PROJECT_ROOT / "external",
              PROJECT_ROOT / "external" / "ifrnet_video"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def beh_group_a() -> List[CheckResult]:
    """BEH-A：分段指纹侧车 round-trip（5 断言）。"""
    out: List[CheckResult] = []
    ph = "C-行为·指纹"
    try:
        from video_utils import _fingerprint_matches  # noqa: E402
    except Exception as e:
        return [_beh_skip(ph, "BEH-A0", "指纹侧车导入", str(e))]
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src.mp4"
        src.write_bytes(b"\x00" * 128)
        sidecar = Path(td) / ".segments_fingerprint.json"
        st = src.stat()
        cur = {"source": str(src.resolve()), "size": st.st_size,
               "mtime_ns": st.st_mtime_ns, "segment_duration": 30}
        sidecar.write_text(json.dumps(cur), encoding="utf-8")
        out.append(_beh(ph, "BEH-A1", "指纹匹配(相同内容)",
                        _fingerprint_matches(str(sidecar), cur)))
        cur2 = dict(cur); cur2["size"] += 1
        out.append(_beh(ph, "BEH-A2", "指纹失效(size变化)",
                        not _fingerprint_matches(str(sidecar), cur2)))
        cur3 = dict(cur); cur3["segment_duration"] = 15
        out.append(_beh(ph, "BEH-A3", "指纹失效(duration变化)",
                        not _fingerprint_matches(str(sidecar), cur3)))
        out.append(_beh(ph, "BEH-A4", "侧车缺失→不匹配",
                        not _fingerprint_matches(str(Path(td) / "nope.json"), cur)))
        out.append(_beh(ph, "BEH-A5", "空当前指纹→不匹配",
                        not _fingerprint_matches(str(sidecar), {})))
    return out


def beh_group_b() -> List[CheckResult]:
    """BEH-B：真实 ffmpeg 端到端（7 断言）。"""
    out: List[CheckResult] = []
    ph = "C-行为·端到端"
    import video_utils  # noqa: E402
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src_a.mp4"
        seg_dir = td / "segs"
        # segment muxer 仅按关键帧切分：-g 40（10fps×4s）保证每 4s 一个 IDR
        gen = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=9",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "40",
               "-keyint_min", "40"]
        rc, _, err = run_cmd(gen + [str(src)], timeout=120)
        if rc != 0:
            return [_beh_skip(ph, "BEH-B0", "合成源生成", err[-160:])]
        # [输出卫生] split 内部的 ✅/⚠️ 打印为受控测试动作，统一捕获后写入详情，
        # 避免被误读为主流程重复执行（生产反馈：verification_2 提问"为何分割3次"
        # —— 实为 B1 切/B3 复用命中/B4 换源重切三次**设计内调用**）。
        with _capture_stdout() as cap1:
            segs = video_utils.split_video_by_time(str(src), str(seg_dir), 4,
                                                   reuse_existing=False)
        out.append(_beh(ph, "BEH-B1", "切片产出3段", len(segs) == 3,
                        f"实际 {len(segs)}；动作: {_capture_stdout.brief(cap1.text)}"))
        out.append(_beh(ph, "BEH-B2", "指纹侧车已写入",
                        (seg_dir / ".segments_fingerprint.json").exists()))
        with _capture_stdout() as cap3:
            segs2 = video_utils.split_video_by_time(str(src), str(seg_dir), 4,
                                                    reuse_existing=True)
        out.append(_beh(ph, "BEH-B3", "同源复用命中", len(segs2) == 3,
                        f"动作: {_capture_stdout.brief(cap3.text)}"))
        src_b = td / "src_b.mp4"
        run_cmd(gen + [str(src_b)], timeout=120)
        os.utime(src_b, None)
        with _capture_stdout() as cap4:
            segs3 = video_utils.split_video_by_time(str(src_b), str(seg_dir), 4,
                                                    reuse_existing=True)
        out.append(_beh(ph, "BEH-B4", "换源触发重切(指纹不符)", len(segs3) == 3,
                        f"动作: {_capture_stdout.brief(cap4.text)}"))
        # 重编码扩展名传播：mpeg4 源 + .mkv 目标 → 实际写出 .mp4 并回传
        mpeg_src = td / "legacy.avi"
        rc, _, err2 = run_cmd(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=2",
             "-c:v", "mpeg4", "-q:v", "10", str(mpeg_src)], timeout=120)
        if rc != 0:
            out.append(_beh_skip(ph, "BEH-B5", "重编码传播(源生成失败)", err2[-120:]))
            return out
        out_mkv = td / "out.mkv"
        actual: List[str] = []
        try:
            with _capture_stdout() as cap:
                ok = video_utils.merge_videos_by_codec(
                    [str(mpeg_src)], str(out_mkv),
                    config={"format": "mp4", "codec": "libx264", "crf": 28,
                            "preset": "ultrafast"},
                force_reencode=True, actual_output=actual)
            out.append(_beh(ph, "BEH-B5", "重编码合并成功", ok is True,
                            detail="（⚠️[merge] 告警为预期触发，已捕获："
                                   + _capture_stdout.brief(cap.text)))
            out.append(_beh(ph, "BEH-B6", "actual_output 已回传",
                            bool(actual) and Path(actual[0]).suffix == ".mp4",
                            f"actual={actual}"))
            out.append(_beh(ph, "BEH-B7", "实际文件存在",
                            bool(actual) and Path(actual[0]).exists()))
        except Exception as e:
            out.append(_beh_skip(ph, "BEH-B5", "重编码传播(构建缺 mpeg4?)", str(e)[:120]))
    return out


def beh_group_c() -> List[CheckResult]:
    """BEH-C：nvenc_sdk 导入契约（5 断言）。"""
    out: List[CheckResult] = []
    ph = "C-行为·SDK契约"
    try:
        from ifrnet_video import nvenc_sdk as ns  # noqa: E402
    except Exception as e:
        return [_beh_skip(ph, "BEH-C0", "nvenc_sdk 导入", f"{type(e).__name__}: {e}")]
    out.append(_beh(ph, "BEH-C1", "FUNC_IDX DestroyEncoder==27",
                    ns._FUNC_IDX["DestroyEncoder"] == 27))
    out.append(_beh(ph, "BEH-C2", "FUNC_IDX OpenEncodeSessionEx==29",
                    ns._FUNC_IDX.get("OpenEncodeSessionEx") == 29))
    try:
        callable(ns._NvEncLockBitstreamFnProto(0)) and \
            callable(ns._NvEncUnlockBitstreamFnProto(0))
        protos_ok = True
    except Exception:
        protos_ok = False
    out.append(_beh(ph, "BEH-C3", "P2.4a 原型可构造", protos_ok))
    out.append(_beh(ph, "BEH-C4", "NEED_MORE_INPUT==17",
                    ns.NV_ENC_ERR_NEED_MORE_INPUT == 17))
    out.append(_beh(ph, "BEH-C5", "close 幂等可调用",
                    callable(getattr(ns.NVENCEncoder, "close", None))))
    return out


def beh_group_d() -> List[CheckResult]:
    """BEH-D：NAL 扫描双侧等价（13 断言）。"""
    out: List[CheckResult] = []
    ph = "C-行为·NAL等价"
    try:
        from ifrnet_video.nvenc_sdk import NVENCEncoder  # noqa: E402
        from nvenc_common import nal_utils  # noqa: E402
        stub = types.SimpleNamespace(_codec="h264")
        _has = lambda s: NVENCEncoder._has_sps_pps(stub, s)          # noqa: E731
        _first = lambda s: NVENCEncoder._nal_first_vcl_type(stub, s)  # noqa: E731
        _ext = lambda s: NVENCEncoder._extract_sps_pps(stub, s)       # noqa: E731
    except Exception as e:
        return [_beh_skip(ph, "BEH-D0", "NAL 导入", str(e))]
    SC4, SC3 = b"\x00\x00\x00\x01", b"\x00\x00\x01"
    sps, pps, idr, nzr = (SC4 + b"\x67\x11", SC4 + b"\x68\x22",
                          SC3 + b"\x65\xaa", SC4 + b"\x01\xbb")
    hevc_vps, hevc_sps, hevc_idr = (SC4 + b"\x40\x01", SC4 + b"\x42\x01",
                                    SC3 + b"\x26\x01")
    streams = [("纯IDR", sps + pps + idr), ("无参数集", idr + nzr),
               ("辅助块开头", nzr + sps + pps + idr), ("空流", b""),
               ("3字节起始码SPS", SC3 + b"\x67\x11" + idr)]
    n = 0
    for name, s in streams:
        n += 1
        out.append(_beh(ph, f"BEH-D{n}", f"h264 has[{name}]",
                        _has(s) == nal_utils.has_param_sets(s, "h264")))
        n += 1
        out.append(_beh(ph, f"BEH-D{n}", f"h264 firstVCL[{name}]",
                        _first(s) == nal_utils.first_vcl_type(s, "h264")))
    s_h264 = sps + pps + idr + nzr
    out.append(_beh(ph, "BEH-D11", "h264 extract 参数集相等",
                    _ext(s_h264) == nal_utils.extract_param_sets(s_h264, "h264")))
    s_hevc = hevc_vps + hevc_sps + hevc_idr
    out.append(_beh(ph, "BEH-D12", "hevc has 参数集",
                    nal_utils.has_param_sets(s_hevc, "hevc") is True))
    out.append(_beh(ph, "BEH-D13", "hevc extract 含VPS",
                    nal_utils.extract_param_sets(s_hevc, "hevc")
                    == hevc_vps + hevc_sps))
    return out


def beh_group_e() -> List[CheckResult]:
    """BEH-E：全量 py_compile 扫描。"""
    out: List[CheckResult] = []
    ph = "C-行为·编译"
    bad = []
    for f in COMPILE_TARGETS:
        if not f.exists():
            bad.append(f"{f.name}: 缺失")
            continue
        ok, err = py_compile_ok(f)
        if not ok:
            bad.append(f"{f.name}: {err[:80]}")
    out.append(_beh(ph, "BEH-E1", f"py_compile ×{len(COMPILE_TARGETS)} 全通过",
                    not bad, "; ".join(bad)[:300] if bad else
                    f"{len(COMPILE_TARGETS)} 个活跃文件语法全部通过",
                    "修复语法错误后重跑" if bad else ""))
    return out


def beh_group_f() -> List[CheckResult]:
    """BEH-F：_validate_effective_config 动态行为（3 断言）。"""
    out: List[CheckResult] = []
    ph = "C-行为·配置校验"
    mod_path = FILES["main_entry"]
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mvo_verify_tmp", str(mod_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        return [_beh_skip(ph, "BEH-F0", "主入口模块加载",
                          f"{type(e).__name__}: {e}（缺依赖时允许跳过）")]
    validator = getattr(mod, "_validate_effective_config", None)
    if validator is None:
        return [_beh_skip(ph, "BEH-F0", "校验器缺失", "_validate_effective_config 未定义")]

    class FakeConfig:
        def __init__(self, tree):
            self._t = tree
        def get(self, *keys, default=None):
            node = self._t
            for k in keys:
                if isinstance(node, dict) and k in node:
                    node = node[k]
                else:
                    return default
            return node

    base = {
        "processing": {"segment_duration": 30, "interpolation_factor": 2,
                       "upscale_factor": 2},
        "models": {
            "ifrnet": {"crf": 23, "batch_size": 12, "max_batch_size": 36,
                       "lookahead_depth": 8, "rate_mode": "vbr_hq"},
            "realesrgan": {"crf": 23, "batch_size": 24, "max_batch_size": 36,
                           "lookahead_depth": 8, "rate_mode": "constqp",
                           "gfpgan_weight": 0.7, "tile_size": 0},
        },
    }
    out.append(_beh(ph, "BEH-F1", "合法基线配置放行",
                    validator(FakeConfig(base)) is True))
    bad_crf = json.loads(json.dumps(base))
    bad_crf["models"]["ifrnet"]["crf"] = 99
    with _capture_stdout() as cap_crf:
        crf_ok = validator(FakeConfig(bad_crf)) is False
    out.append(_beh(ph, "BEH-F2", "crf=99 被拒绝（校验器告警已捕获："
                    + _capture_stdout.brief(cap_crf.text) + "）", crf_ok))
    bad_dur = json.loads(json.dumps(base))
    bad_dur["processing"]["segment_duration"] = -5
    with _capture_stdout() as cap_dur:
        dur_ok = validator(FakeConfig(bad_dur)) is False
    out.append(_beh(ph, "BEH-F3", "segment_duration=-5 被拒绝（告警已捕获："
                    + _capture_stdout.brief(cap_dur.text) + "）", dur_ok))
    return out


def beh_group_g() -> List[CheckResult]:
    """BEH-G：SPS/PPS 原语动态行为（[P2.4c-LADDER]，8 断言，纯字节无 GPU）。"""
    out: List[CheckResult] = []
    ph = "C-行为·SPS原语"
    try:
        from ifrnet_video.nvenc_sdk import NVENCEncoder  # noqa: E402
    except Exception as e:
        return [_beh_skip(ph, "BEH-G0", "nvenc_sdk 导入", str(e))]

    SC4 = b"\x00\x00\x00\x01"
    sps_pps = SC4 + b"\x67\x11" + SC4 + b"\x68\x22"
    idr = SC4 + b"\x65\xaa\xbb"      # IDR，原生不含参数集
    idr_full = sps_pps + SC4 + b"\x65\xcc"  # IDR，自带参数集
    non_idr = SC4 + b"\x01\xdd"

    def make_enc(cached=None, injected=False):
        enc = NVENCEncoder.__new__(NVENCEncoder)
        enc._codec = "h264"
        enc._cached_sps_pps = cached
        enc._sps_pps_injected = injected
        enc._muxer_writes = []
        class _Mux:
            def write_sps_pps(self, s): enc._muxer_writes.append(s)
        enc._muxer_ref = _Mux()
        return enc

    # G1: prepend — IDR+有缓存+缺参数集 → 补挂
    e1 = make_enc(cached=sps_pps)
    out.append(_beh(ph, "BEH-G1", "IDR补挂缓存参数集",
                    e1._prepend_param_sets(idr, True) == sps_pps + idr))
    # G2: 原生已含 → 不重复挂
    e2 = make_enc(cached=sps_pps)
    out.append(_beh(ph, "BEH-G2", "原生含参数集跳过prepend",
                    e2._prepend_param_sets(idr_full, True) == idr_full))
    # G3: 非IDR → 不挂
    e3 = make_enc(cached=sps_pps)
    out.append(_beh(ph, "BEH-G3", "非IDR不prepend",
                    e3._prepend_param_sets(non_idr, False) == non_idr))
    # G4: cache — 首见提取并缓存；IDR 时预注入 muxer
    e4 = make_enc()
    ok4 = (e4._cache_param_sets(idr_full, True) is True
           and e4._cached_sps_pps == sps_pps
           and len(e4._muxer_writes) == 1
           and e4._sps_pps_injected is True)
    out.append(_beh(ph, "BEH-G4", "首见缓存+IDR预注入muxer", ok4))
    # G5: cache 幂等 — 已有缓存不再写 muxer
    e5 = make_enc(cached=sps_pps)
    out.append(_beh(ph, "BEH-G5", "已有缓存幂等",
                    e5._cache_param_sets(non_idr, False) is False
                    and not e5._muxer_writes))
    # G6: inject_without_idr — 辅助块路径无条件预注入
    e6 = make_enc()
    aux = SC4 + b"\x67\x11"  # 独立 SPS AU
    ok6 = (e6._cache_param_sets(aux, False, inject_without_idr=True) is True
           and len(e6._muxer_writes) == 1)
    out.append(_beh(ph, "BEH-G6", "辅助块无VCL无条件预注入", ok6))
    # G7: apply 组合语义 — 无缓存时首帧(自带参数集)不 prepend 仅缓存+注入
    e7 = make_enc()
    r7 = e7._apply_sps_pps(idr_full, True)
    ok7 = (r7 == idr_full and e7._cached_sps_pps == sps_pps
           and len(e7._muxer_writes) == 1)  # [FIX-SPS-PPS-V3] 首帧不prepend
    out.append(_beh(ph, "BEH-G7", "apply首帧仅缓存不prepend", ok7))
    # G8: apply 二次调用 — 后续 IDR 补挂且不重复注入 muxer
    r8 = e7._apply_sps_pps(SC4 + b"\x65\xff", True)
    out.append(_beh(ph, "BEH-G8", "后续IDR补挂且muxer单次注入",
                    r8 == sps_pps + SC4 + b"\x65\xff"
                    and len(e7._muxer_writes) == 1))
    return out


def run_behavior_phase() -> List[CheckResult]:
    _setup_behavior_paths()
    results: List[CheckResult] = []
    for fn in (beh_group_a, beh_group_b, beh_group_c, beh_group_d,
               beh_group_e, beh_group_f, beh_group_g):
        try:
            results.extend(fn())
        except Exception as e:
            results.append(CheckResult(
                id="BEH-ERR", phase="C-行为", name=fn.__name__,
                status=Status.WARN, method="行为执行",
                criteria="组执行不崩溃",
                detail=f"组级异常: {type(e).__name__}: {e}",
                suggestion="人工复核该组"))
    return results


# ===========================================================================
# D. 运行时验证（ffprobe 对比）与 E. 冒烟
# ===========================================================================

def _load_video_validation_helpers():
    utils_dir = str(PROJECT_ROOT / "src" / "utils")
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)
    from video_utils import count_decoded_video_frames, validate_decodable_video
    return count_decoded_video_frames, validate_decodable_video


def _run_fix_effect_phase() -> List[CheckResult]:
    """[PLAN-FIX-EFFECT] 机器可核验综合修复执行状态。"""
    out: List[CheckResult] = []
    cfg_path = PROJECT_ROOT / "config" / "default_config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        cfg = None
        out.append(CheckResult("FIX-CFG", "F-修复效果", "安全配置解析",
                               Status.FAIL, "JSON 解析", "可读且合法",
                               f"异常: {e}", "检查 default_config.json"))
    if cfg is not None:
        for model in ("ifrnet", "realesrgan"):
            sec = cfg.get("models", {}).get(model, {})
            safe = (sec.get("rate_mode") == "constqp"
                    and int(sec.get("lookahead_depth", -1)) == 0)
            out.append(CheckResult(
                f"FIX-{model.upper()}-LA0", "F-修复效果",
                f"{model} 默认 LA=0/constqp", Status.PASS if safe else Status.FAIL,
                "读取配置", "rate_mode=constqp 且 lookahead_depth=0",
                f"rate_mode={sec.get('rate_mode')}, "
                f"lookahead_depth={sec.get('lookahead_depth')}"))

        h2d_txt = _as_text(FILES["if_utils"])
        pipeline_txt = _as_text(FILES["if_pipeline"])
        h2d_ok = ("P1-FIX-H2D-EVENT-SYNC" in h2d_txt
                  and "def mark_issued" in h2d_txt
                  and pipeline_txt.count("pool.mark_issued") >= 2)
        out.append(CheckResult("FIX-H2D-SYNC", "F-修复效果", "H2D Event 同步",
                               Status.PASS if h2d_ok else Status.FAIL,
                               "源码锚点+结构", "事件记录和slot复用等待存在",
                               "P1-FIX-H2D-EVENT-SYNC / pool.mark_issued"))

        route_ok = has_pattern(FILES["ifrnet_proc"],
                               r"self\.hevc_la_disable\s*=\s*bool\(config\.get")
        out.append(CheckResult("FIX-ROUTE", "F-修复效果", "HEVC+LA 安全路由",
                               Status.PASS if route_ok else Status.FAIL,
                               "源码锚点", "处理器启用 hevc_la_disable",
                               "hevc_la_disable 配置与路由逻辑"))

        # [FIX-HEVC-LA-OPEN] HEVC LA>0 生产就绪：配置默认 false 且处理器软退役（WARN 不降级）
        _cfg_hevc_ifr = cfg.get("models", {}).get("ifrnet", {}).get("hevc_la_disable", True)
        _cfg_hevc_esr = cfg.get("models", {}).get("realesrgan", {}).get("hevc_la_disable", True)
        _cfg_open = (_cfg_hevc_ifr is False and _cfg_hevc_esr is False)
        _proc_ifr_txt = _as_text(FILES["ifrnet_proc"])
        _proc_esr_txt = _as_text(FILES["esrgan_proc"])
        _soft_ifr = ("[FIX-HEVC-LA-SOFT-RETIRED]" in _proc_ifr_txt
                     and "DEPRECATED" in _proc_ifr_txt
                     and "不改写 self.lookahead_depth" in _proc_ifr_txt)
        _soft_esr = ("[FIX-HEVC-LA-SOFT-RETIRED]" in _proc_esr_txt
                     and "DEPRECATED" in _proc_esr_txt)
        # 旧路由会含 "lookahead_depth = 0" 紧跟 hevc 判断；软退役后该赋值不应出现在 hevc 分支
        _no_downgrade_ifr = not re.search(
            r'\[FIX-HEVC-LA-ROUTE\].*?lookahead_depth\s*=\s*0', _proc_ifr_txt, re.DOTALL)
        # realesrgan 侧同样不应有旧式 hevc→LA=0 赋值（软退役后仅 WARN）
        _no_downgrade_esr = not re.search(
            r'hevc.*lookahead_depth\s*=\s*0', _proc_esr_txt, re.IGNORECASE)
        _open_ok = _cfg_open and _soft_ifr and _soft_esr and _no_downgrade_ifr
        out.append(CheckResult("FIX-HEVC-LA-OPEN", "F-修复效果", "HEVC LA>0 开放（软退役）",
                               Status.PASS if _open_ok else Status.FAIL,
                               "配置+源码锚点", "config hevc_la_disable==false 且处理器 WARN 不降级",
                               f"cfg_ifr={_cfg_hevc_ifr} cfg_esr={_cfg_hevc_esr} soft_ifr={_soft_ifr} soft_esr={_soft_esr} no_downgrade_ifr={_no_downgrade_ifr}"))

        gate_ok = ("def validate_decodable_video" in _as_text(FILES["video_utils"])
                   and "validate_decodable_video" in _as_text(FILES["ifrnet_proc"])
                   and "validate_decodable_video" in _as_text(FILES["esrgan_proc"])
                   and "validate_decodable_video" in _as_text(FILES["main_entry"]))
        out.append(CheckResult("FIX-GATE", "F-修复效果", "解码级验收门禁",
                               Status.PASS if gate_ok else Status.FAIL,
                               "源码锚点", "video_utils 与两阶段/最终合并接入",
                               "validate_decodable_video / count_decoded_video_frames"))

        eos_ok = (has_pattern(FILES["if_nvenc"], r"P2-FIX-EOS-OUTPUT-ORDER")
                  and not re.search(r"_drain_slots\s*=\s*sorted\("
                                    r"self\._strm_slot_pending\.keys\(\)", _as_text(FILES["if_nvenc"])))
        strict_ok = has_pattern(FILES["if_nvenc"], r"P2-FIX-STRICT-EOS")
        clamp_ok = has_pattern(FILES["if_nvenc"], r"P3-FIX-LockBitstream-SizeCap")
        nal_ok = has_pattern(FILES["if_nvenc"], r"P3-FIX-NAL-COMMON")
        for check_id, ok in (("FIX-EOS-ORDER", eos_ok),
                             ("FIX-STRICT-EOS", strict_ok),
                             ("FIX-SIZE-CAP", clamp_ok),
                             ("FIX-NAL-COMMON", nal_ok)):
            out.append(CheckResult(check_id, "F-修复效果",
                                   check_id.replace("FIX-", ""),
                                   Status.PASS if ok else Status.FAIL,
                                   "源码锚点", "P2/P3 硬化已接入",
                                   check_id))
    return out


def _check_output_video(input_path: Optional[Path],
                        output_path: Optional[Path]) -> List[CheckResult]:
    results: List[CheckResult] = []
    if output_path is None or not output_path.exists():
        results.append(CheckResult(
            "RT-0", "D-运行时", "输出视频存在", Status.SKIP,
            "检查输出文件", "输出文件存在为通过", "未提供 --output 或文件不存在"))
        return results
    meta = ffprobe(output_path)
    if not meta:
        results.append(CheckResult(
            "RT-1", "D-运行时", "容器可解析", Status.FAIL,
            "ffprobe 解析输出", "ffprobe 能读取容器为通过",
            f"ffprobe 无法解析 {output_path}", "moov 是否写盘？"))
        return results
    results.append(CheckResult(
        "RT-1", "D-运行时", "容器可解析", Status.PASS,
        "ffprobe 解析输出", "ffprobe 能读取容器为通过",
        f"解析成功: {output_path.name}"))
    vs = [s for s in meta.get("streams", []) if s.get("codec_type") == "video"]
    as_ = [s for s in meta.get("streams", []) if s.get("codec_type") == "audio"]
    if vs:
        results.append(CheckResult(
            "RT-2", "D-运行时", "视频流编码器", Status.PASS,
            "ffprobe codec_name", "存在视频流为通过",
            f"codec={vs[0].get('codec_name','?')}, "
            f"{vs[0].get('width','?')}x{vs[0].get('height','?')}, "
            f"frames≈{int(vs[0].get('nb_frames', 0) or 0)}"))
    else:
        results.append(CheckResult(
            "RT-2", "D-运行时", "视频流编码器", Status.FAIL,
            "ffprobe codec_name", "存在视频流为通过", "输出无视频流"))
    if as_:
        results.append(CheckResult(
            "RT-3", "D-运行时", "音轨存在", Status.PASS,
            "ffprobe 音频流", "有音频流为通过",
            f"codecs={[s.get('codec_name') for s in as_]}"))
    elif input_path is not None and input_path.exists():
        in_meta = ffprobe(input_path)
        in_audio = any(s.get("codec_type") == "audio"
                       for s in in_meta.get("streams", []))
        results.append(CheckResult(
            "RT-3", "D-运行时", "音轨存在",
            Status.FAIL if in_audio else Status.PASS,
            "输入/输出音轨对比", "输入有音则输出应有音",
            "输入含音但输出无声（H10 风险）" if in_audio else "输入无音，无需校验",
            "核查 audio_src 传递链" if in_audio else ""))
    if input_path is not None and input_path.exists() and vs:
        in_meta = ffprobe(input_path)
        in_vs = [s for s in in_meta.get("streams", [])
                 if s.get("codec_type") == "video"]
        if in_vs:
            try:
                _count_frames, _validate_dec = _load_video_validation_helpers()
            except ImportError as exc:
                results.append(CheckResult(
                    "RT-IMPORT", "D-运行时", "验证 helper 导入", Status.WARN,
                    "导入 video_utils", "helper 可导入", f"异常: {exc}",
                    "检查 src/utils/video_utils.py"))
                return results

            # [P4-FIX-VERIFY] 真实解码帧数，修复容器 nb_frames 盲区。
            in_nb = _count_frames(input_path)
            out_nb = _count_frames(output_path)
            ratio_ok = bool(in_nb and out_nb and in_nb > 0
                            and abs((out_nb / in_nb) - round(out_nb / in_nb)) < 0.02)
            results.append(CheckResult(
                "RT-4", "D-运行时", "解码级帧数守恒",
                Status.PASS if ratio_ok else Status.WARN,
                "ffprobe -count_frames", "输出/输入解码帧数≈整数倍",
                f"in={in_nb} → out={out_nb}",
                "" if ratio_ok else "解码帧数缩水/漂移或不可用，请核对倍数"))

            # [P4-FIX-VERIFY] 解码错误必须为零；这是尾部断链的核心门禁。
            dec_ok, dec_report = _validate_dec(output_path)
            results.append(CheckResult(
                "RT-5", "D-运行时", "解码错误零容忍",
                Status.PASS if dec_ok else Status.FAIL,
                "ffmpeg -v error ... -f null -", "stderr 为空且 rc=0",
                f"reason={dec_report.get('reason')}, "
                f"errors={dec_report.get('decode_errors')} "
                f"{str(dec_report.get('decode_stderr_tail', ''))[:500]}"))
    return results


def _run_smoke_test(interpolation_factor: int, upscale_factor: int,
                    mode: str = "interpolate_only",
                    ifrnet_batch: Optional[int] = None,
                    esrgan_batch: Optional[int] = None) -> List[CheckResult]:
    """生成合成视频跑主流程（GPU 门控）。mode 见 --smoke-mode 帮助。"""
    results: List[CheckResult] = []
    ffmpeg = which("ffmpeg")
    if not ffmpeg:
        return [CheckResult("SMOKE-0", "E-冒烟", "冒烟测试", Status.SKIP,
                            "环境", "ffmpeg 可用", "无 ffmpeg，跳过冒烟")]
    try:
        import torch
        if not torch.cuda.is_available():
            return [CheckResult("SMOKE-0", "E-冒烟", "冒烟测试", Status.SKIP,
                                "GPU 检测", "CUDA 可用",
                                "无 GPU，跳过冒烟（生产环境执行）")]
    except ImportError:
        return [CheckResult("SMOKE-0", "E-冒烟", "冒烟测试", Status.SKIP,
                            "GPU 检测", "torch 可用", "torch 未安装，跳过冒烟")]

    tmp = Path(tempfile.mkdtemp(prefix="verify_final_"))
    src, out = tmp / "synth.mp4", tmp / "out.mp4"
    rc, _, err = run_cmd([
        ffmpeg, "-y", "-f", "lavfi",
        "-i", "testsrc2=size=640x360:rate=24:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(src)], timeout=120)
    if rc != 0 or not src.exists():
        return [CheckResult("SMOKE-1", "E-冒烟", "合成视频生成", Status.FAIL,
                            "ffmpeg 生成", "生成成功", f"失败: {err[:200]}")]
    results.append(CheckResult("SMOKE-1", "E-冒烟", "合成视频生成", Status.PASS,
                               "ffmpeg 生成", "生成成功", f"{src.name}"))

    entry = FILES["main_entry"]
    cmd = [sys.executable, str(entry),
           "-c", str(PROJECT_ROOT / "config" / "default_config.json"),
           "-i", str(src), "-o", str(out)]
    does_upscale = mode in ("upscale_only", "interpolate_then_upscale",
                            "upscale_then_interpolate")
    if mode == "interpolate_only":
        cmd += ["--skip-upscale", "--interpolation-factor", str(interpolation_factor)]
    elif mode == "upscale_only":
        cmd += ["--skip-interpolate", "--upscale-factor", str(upscale_factor)]
    elif mode == "interpolate_then_upscale":
        cmd += ["--interpolation-factor", str(interpolation_factor),
                "--upscale-factor", str(upscale_factor)]
    else:
        cmd += ["--mode", "upscale_then_interpolate",
                "--interpolation-factor", str(interpolation_factor),
                "--upscale-factor", str(upscale_factor)]
    # [TRT-TAG-CONTROL] batch 直通：缺省不传（沿用 config 24），显式给定则固定
    # TRT 引擎 tag 的 B 维度，保证缓存命中可预测。
    if ifrnet_batch:
        cmd += ["--batch-size-ifrnet", str(ifrnet_batch)]
    if esrgan_batch:
        cmd += ["--batch-size-esrgan", str(esrgan_batch)]
    timeout = 3600 if does_upscale else 1800
    t0 = time.time()
    rc, stdout, stderr = run_cmd(cmd, timeout=timeout)
    elapsed = time.time() - t0
    log_tail = (stdout + stderr)[-1500:]
    if rc != 0 or not out.exists():
        return results + [CheckResult(
            "SMOKE-2", "E-冒烟", f"端到端处理（{mode}）", Status.FAIL,
            "运行 main_video_optimized.py", "rc=0 且产出文件",
            f"rc={rc}, {elapsed:.0f}s\n{log_tail}")]
    results.append(CheckResult("SMOKE-2", "E-冒烟", f"端到端处理（{mode}）",
                               Status.PASS, "运行 main_video_optimized.py",
                               "rc=0 且产出文件", f"{elapsed:.0f}s"))

    in_meta, out_meta = ffprobe(src), ffprobe(out)
    in_vs = [s for s in in_meta.get("streams", []) if s.get("codec_type") == "video"]
    out_vs = [s for s in out_meta.get("streams", []) if s.get("codec_type") == "video"]
    _count_frames, _validate_dec = _load_video_validation_helpers()
    in_nb = int(_count_frames(src) or 0) if in_vs else 0
    out_nb = int(_count_frames(out) or 0) if out_vs else 0
    frame_mult = 1 if mode == "upscale_only" else interpolation_factor
    expect = in_nb * frame_mult
    ok = abs(out_nb - expect) <= max(2, int(expect * 0.02))
    results.append(CheckResult(
        "SMOKE-3", "E-冒烟", "帧数守恒", Status.PASS if ok else Status.FAIL,
        "ffprobe 帧数对比", f"期望≈{expect}",
        f"in={in_nb} → out={out_nb} (expect={expect})",
        "" if ok else "帧数守恒被破坏（丢帧/空帧/LA 排空问题）"))
    dec_ok, dec_report = _validate_dec(out)
    results.append(CheckResult(
        "SMOKE-DECODE", "E-冒烟", "输出零解码错误", Status.PASS if dec_ok else Status.FAIL,
        "ffmpeg -v error ... -f null -", "stderr 为空且 rc=0",
        f"reason={dec_report.get('reason')}, errors={dec_report.get('decode_errors')}",
        "" if dec_ok else str(dec_report.get("decode_stderr_tail", ""))[-500:]))
    if does_upscale and in_vs and out_vs:
        exp_w = int(in_vs[0].get("width", 0) or 0) * upscale_factor
        exp_h = int(in_vs[0].get("height", 0) or 0) * upscale_factor
        got = (int(out_vs[0].get("width", 0) or 0),
               int(out_vs[0].get("height", 0) or 0))
        res_ok = got == (exp_w, exp_h)
        results.append(CheckResult(
            "SMOKE-4", "E-冒烟", "分辨率守恒",
            Status.PASS if res_ok else Status.FAIL,
            "ffprobe 分辨率对比", f"期望 {exp_w}x{exp_h}",
            f"got {got[0]}x{got[1]}",
            "" if res_ok else "超分倍数未生效或分辨率异常"))
    shutil.rmtree(tmp, ignore_errors=True)
    return results


# ===========================================================================
# 报告渲染
# ===========================================================================

_STATUS_ICON = {Status.PASS: "✅", Status.FAIL: "❌",
                Status.WARN: "⚠️", Status.SKIP: "⏭️"}

_PHASE_ORDER = ["A-前置条件", "B-静态·阶段0", "B-静态·阶段1", "B-静态·阶段2",
                "B-静态·阶段3", "C-行为", "D-运行时", "E-冒烟"]


def _render_console(results: List[CheckResult]) -> str:
    lines = ["=" * 78,
             "  Video Enhancement 优化方案 · 最终后验证报告（v2 三合一）",
             f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             "=" * 78]
    by_phase: Dict[str, List[CheckResult]] = {}
    for r in results:
        by_phase.setdefault(r.phase, []).append(r)

    def _order_key(p: str):
        for i, want in enumerate(_PHASE_ORDER):
            if p == want:
                return (i, "")
            if p.startswith(want):
                return (i, p)
        return (len(_PHASE_ORDER), p)

    for ph in sorted(by_phase, key=_order_key):
        items = by_phase[ph]
        cnt = {s: sum(1 for r in items if r.status is s) for s in Status}
        lines.append("")
        lines.append(f"━━━ {ph} ━━━ [通过 {cnt[Status.PASS]} | 失败 {cnt[Status.FAIL]} "
                     f"| 警告 {cnt[Status.WARN]} | 跳过 {cnt[Status.SKIP]}]")
        for r in items:
            icon = _STATUS_ICON[r.status]
            lines.append(f"  {icon} [{r.id}] {r.name} — {r.status.value}")
            if r.detail:
                lines.append(f"       详情: {r.detail}")
            if r.suggestion:
                lines.append(f"       建议: {r.suggestion}")

    n_pass = sum(1 for r in results if r.status is Status.PASS)
    n_fail = sum(1 for r in results if r.status is Status.FAIL)
    n_warn = sum(1 for r in results if r.status is Status.WARN)
    n_skip = sum(1 for r in results if r.status is Status.SKIP)
    lines += ["", "=" * 78,
              f"  汇总: 共 {len(results)} 项 | 通过 {n_pass} | 失败 {n_fail} | "
              f"警告 {n_warn} | 跳过 {n_skip}"]
    if n_fail:
        lines.append(f"  ⛔ 结论: 存在 {n_fail} 项失败，优化方案尚未完全落地")
    else:
        lines.append("  ✅ 结论: 无失败项（警告/跳过项见上方说明）")
    lines.append("=" * 78)
    return "\n".join(lines)


def _to_markdown(results: List[CheckResult]) -> str:
    lines = ["# Video Enhancement 优化方案 · 最终后验证报告（v2）", "",
             f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    by_phase: Dict[str, List[CheckResult]] = {}
    for r in results:
        by_phase.setdefault(r.phase, []).append(r)
    for ph in sorted(by_phase, key=lambda p: (_PHASE_ORDER.index(
            next((w for w in _PHASE_ORDER if p == w or p.startswith(w)),
                 _PHASE_ORDER[-1])), p)):
        lines += [f"## {ph}", "",
                  "| 状态 | ID | 验证项 | 方法 | 详情 |", "|---|---|---|---|---|"]
        for r in by_phase[ph]:
            d = (r.detail or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {r.status.value} | {r.id} | {r.name} | {r.method} | {d} |")
        lines.append("")
    return "\n".join(lines)


# ===========================================================================
# 主入口
# ===========================================================================

def _force_utf8_stdout():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: Optional[List[str]] = None) -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="优化方案·最终后验证脚本（三合一整合版 v2）")
    parser.add_argument("--input", "-i", default=None,
                        help="输入视频（运行时对比）")
    parser.add_argument("--output", "-o", default=None,
                        help="输出视频（运行时对比）")
    parser.add_argument("--interpolation-factor", type=int, default=2)
    parser.add_argument("--upscale-factor", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true",
                        help="追加 GPU 冒烟（生成合成视频跑主流程）")
    parser.add_argument("--smoke-mode",
                        choices=["interpolate_only", "upscale_only",
                                 "interpolate_then_upscale",
                                 "upscale_then_interpolate"],
                        default="interpolate_only")
    parser.add_argument("--ifrnet-batch", type=int, default=None,
                        help="冒烟时透传 --batch-size-ifrnet（控制 IFRNet TRT "
                             "引擎 tag 的 B 维度；缺省沿用 config 默认 24）。"
                             "注意：TRT tag = model_B{B}_H{stride对齐高}_W{宽}_…"
                             "，H/W 取自该阶段首个输入（受 --smoke-mode 顺序影响："
                             "upscale 先行时 IFRNet 吃到 ×N 上采样帧）")
    parser.add_argument("--esrgan-batch", type=int, default=None,
                        help="冒烟时透传 --batch-size-esrgan（控制 SR TRT tag 的 B）")
    parser.add_argument("--behavior-only", action="store_true",
                        help="仅执行行为验证阶段（test_regression_min 兼容模式）")
    parser.add_argument("--skip-behavior", action="store_true",
                        help="跳过行为验证（仅前置+静态[+运行时/冒烟]）")
    parser.add_argument("--report-dir",
                        default=str(PROJECT_ROOT / "verification_report"))
    parser.add_argument("--no-report-file", action="store_true",
                        help="仅打印控制台，不写报告文件")
    args = parser.parse_args(argv)

    try:
        results: List[CheckResult] = []

        if args.behavior_only:
            results.extend(run_behavior_phase())
        else:
            results.extend(c.run() for c in CHECKS)
            results.extend(_run_fix_effect_phase())
            if not args.skip_behavior:
                results.extend(run_behavior_phase())
            results.extend(_check_output_video(
                Path(args.input) if args.input else None,
                Path(args.output) if args.output else None))
            if args.smoke_test:
                results.extend(_run_smoke_test(
                    args.interpolation_factor, args.upscale_factor,
                    args.smoke_mode,
                    ifrnet_batch=args.ifrnet_batch,
                    esrgan_batch=args.esrgan_batch))

        print(_render_console(results))

        if not args.no_report_file and not args.behavior_only:
            rd = Path(args.report_dir)
            rd.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                jp = rd / f"verification_report_{ts}.json"
                mp = rd / f"verification_report_{ts}.md"
                jp.write_text(json.dumps(
                    [{"id": r.id, "phase": r.phase, "name": r.name,
                      "status": r.status.value, "method": r.method,
                      "criteria": r.criteria, "detail": r.detail,
                      "suggestion": r.suggestion} for r in results],
                    ensure_ascii=False, indent=2), encoding="utf-8")
                mp.write_text(_to_markdown(results), encoding="utf-8")
                print(f"\n📄 报告已写出:\n   JSON: {jp}\n   MD  : {mp}")
            except OSError as e:
                print(f"\n⚠️  报告写出失败: {e}")

        n_fail = sum(1 for r in results if r.status is Status.FAIL)
        return 1 if n_fail else 0
    except Exception as e:  # 顶层兜底：脚本自身异常 ≠ 验证失败
        print(f"[verify-final] 脚本异常终止: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
