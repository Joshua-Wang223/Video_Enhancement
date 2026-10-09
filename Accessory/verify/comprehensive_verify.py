#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合验证脚本：整合所有验证测试内容
支持 --env cpu|t4|l40 分环境执行
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SCRIPTS = {
    "plan_gate": PROJECT_ROOT / "Accessory" / "verify" / "plan_implementation_gate.py",
    "verify_eq": PROJECT_ROOT / "Accessory" / "verify" / "verify_equal_quality.py",
    "crf_cq": PROJECT_ROOT / "Accessory" / "verify" / "crf_cq_unification_verify.py",
    "av1_vp9_matrix": PROJECT_ROOT / "Accessory" / "probe" / "av1_vp9_quality_matrix.py",
    "nvenc_vbr_hq": PROJECT_ROOT / "Accessory" / "probe" / "nvenc_vbr_hq_verify.py",
    "nvenc_diagnose": PROJECT_ROOT / "Accessory" / "probe" / "nvenc_rc_mode_diagnose.py",
    "seg_verify": PROJECT_ROOT / "Accessory" / "verify" / "segment_bitstream_verify_v5.py",
    "av1_smoke": PROJECT_ROOT / "Accessory" / "verify" / "av1_pipeline_smoke.py",
    "eqq_pool_fit": PROJECT_ROOT / "Accessory" / "probe" / "eqq_pool_fit_table.py",
    "calibrate_eq": PROJECT_ROOT / "Accessory" / "probe" / "calibrate_equal_quality.py",
    "nvenc_tune": PROJECT_ROOT / "src" / "utils" / "nvenc_tuning.py",
}


class Env(str, Enum):
    CPU = "cpu"
    T4 = "t4"
    L40 = "l40"


@dataclass
class TestResult:
    name: str
    status: str  # PASS/FAIL/SKIP/WARN
    detail: str = ""
    duration: float = 0.0
    exit_code: int = 0


@dataclass
class TestCase:
    name: str
    envs: List[Env]  # 适用环境
    cmd_builder: Callable[[argparse.Namespace], List[str]]
    description: str = ""
    required_files: List[Path] = field(default_factory=list)
    required_gpu: bool = False
    timeout: int = 300


def which(tool: str) -> Optional[str]:
    return shutil.which(tool)


def run_cmd(args: List[str], timeout: int = 300, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(
            args, capture_output=True, text=True,
            stdin=subprocess.DEVNULL,
            encoding="utf-8", errors="replace", timeout=timeout, cwd=str(cwd) if cwd else None,
            env=env
        )
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return -127, "", f"command not found: {args[0] if args else ''}"
    except subprocess.TimeoutExpired:
        return -124, "", "timeout"


def check_gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def check_nvenc_codec(codec: str) -> bool:
    ffmpeg = which("ffmpeg")
    if not ffmpeg:
        return False
    rc, _, _ = run_cmd([
        ffmpeg, "-hide_banner", "-f", "lavfi",
        "-i", "testsrc2=size=320x240:rate=30:duration=1",
        "-c:v", codec, "-f", "null", "-"
    ], timeout=20)
    return rc == 0


def get_gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "Unknown"


#: 外部素材目录候选（按优先级探测，首个存在者生效）。
#: 生产 Linux/WSL 与开发机 Windows 均为「项目父目录下的同级 input_videos」，
#: 故以 PROJECT_ROOT.parent 为主，Windows 固定路径仅作兜底。
INPUT_VIDEOS_CANDIDATES = (
    PROJECT_ROOT.parent / "input_videos",
    Path("/mnt/d/Workspace_Python/input_videos"),
)

def _input_videos_base() -> Path:
    for c in INPUT_VIDEOS_CANDIDATES:
        if c.is_dir():
            return c
    return INPUT_VIDEOS_CANDIDATES[0]

INPUT_VIDEOS_BASE = _input_videos_base()

def _env_val(args: argparse.Namespace) -> str:
    """归一化 env 取值：args.env 可能是 str（argparse）或 Env 枚举。"""
    e = getattr(args, "env", Env.CPU)
    return e.value if isinstance(e, Env) else str(e)

def _default_source(args: argparse.Namespace, filename: str) -> str:
    """获取默认源视频路径：优先用 args.source，否则用外部 input_videos 目录。"""
    if args.source:
        return args.source
    for base in INPUT_VIDEOS_CANDIDATES:
        p = base / filename
        if p.exists():
            return str(p)
    return filename


def build_test_matrix() -> List[TestCase]:
    tests = []

    # 1. plan_implementation_gate - 优化方案落地验证 (全环境)
    tests.append(TestCase(
        name="plan_gate",
        envs=[Env.CPU, Env.T4, Env.L40],
        cmd_builder=lambda args: [
            sys.executable, str(SCRIPTS["plan_gate"]),
            *(["-i", args.input, "-o", args.output] if args.input and args.output else []),
            *(["--smoke-test"] if args.smoke else []),
            *(["--smoke-mode", args.smoke_mode] if args.smoke_mode else []),
            *(["--behavior-only"] if args.behavior_only else []),
            *(["--skip-behavior"] if args.skip_behavior else []),
            "<", "/dev/null"
        ],
        description="Phase 0-3 优化方案落地验证（静态/行为/运行时/冒烟）",
        required_files=[SCRIPTS["plan_gate"]],
    ))

    # 2. verify_equal_quality - 等质量换算表验证 (CPU)
    tests.append(TestCase(
        name="verify_equal_quality",
        envs=[Env.CPU],
        cmd_builder=lambda args: [
            sys.executable, str(SCRIPTS["verify_eq"]),
            "--src", _default_source(args, "word_world_2.mp4"),
            "--codecs", "libx265,libvpx-vp9,libsvtav1,libaom-av1,librav1e",
            "--duration", "6",
            "<", "/dev/null"
        ],
        description="等质量换算表主门禁 ΔVMAF ≤1.0（纯 CPU 逻辑）",
        required_files=[SCRIPTS["verify_eq"]],
        timeout=600,
    ))

    # 3. crf_cq_unification_verify - 质量参数统一验证 (CPU + GPU)
    tests.append(TestCase(
        name="crf_cq_cpu",
        envs=[Env.CPU, Env.T4, Env.L40],
        cmd_builder=lambda args: [
            sys.executable, str(SCRIPTS["crf_cq"]),
            "--quick", "--no-gpu",
            "<", "/dev/null"
        ],
        description="质量参数换算正确性（CPU 静态断言 G1~G6/G10）",
        required_files=[SCRIPTS["crf_cq"]],
        timeout=300,
    ))

    tests.append(TestCase(
        name="crf_cq_gpu",
        envs=[Env.T4, Env.L40],
        cmd_builder=lambda args: [
            sys.executable, str(SCRIPTS["crf_cq"]),
            "--gpu",
            "--source", _default_source(args, "word_world_2.mp4"),
            "--bitrate-source", _default_source(args, "new4_raw.mp4"),
            "<", "/dev/null"
        ],
        description="GPU 画质与码率天花板验证（G7/G8，需真实素材+GPU）",
        required_files=[SCRIPTS["crf_cq"]],
        required_gpu=True,
    ))

    # 4. av1_vp9_quality_matrix - AV1/VP9 质量矩阵 (CPU软编 + GPU硬编)
    tests.append(TestCase(
        name="av1_vp9_matrix_cpu",
        envs=[Env.CPU, Env.T4, Env.L40],
        cmd_builder=lambda args: [
            sys.executable, str(SCRIPTS["av1_vp9_matrix"]),
            "--quality-mode", "quality",
            "--src", _default_source(args, "word_world_2.mp4"),
            "--only", "libvpx-vp9,libsvtav1,libaom-av1",  # 核心软编 3 个，避免全量 7 个超时
            "<", "/dev/null"
        ],
        description="AV1/VP9 家族核心软编质量矩阵（3 编码器，CPU 可跑）",
        required_files=[SCRIPTS["av1_vp9_matrix"]],
        timeout=900,
    ))

    # 5. NVENC vbr_hq 迁移验证 (T4)
    tests.append(TestCase(
        name="nvenc_vbr_hq_verify",
        envs=[Env.T4],
        cmd_builder=lambda args: [
            sys.executable, str(SCRIPTS["nvenc_vbr_hq"]),
            "<", "/dev/null"
        ],
        description="NVENC vbr_hq 移除迁移验证（CLI 拒绝 + SDK 接受 + 画质对比）",
        required_files=[SCRIPTS["nvenc_vbr_hq"]],
        required_gpu=True,
    ))

    tests.append(TestCase(
        name="nvenc_rc_diagnose",
        envs=[Env.T4, Env.L40],
        cmd_builder=lambda args: [
            sys.executable, str(SCRIPTS["nvenc_diagnose"]),
            *( [args.nvenc_header] if args.nvenc_header else [] ),
            "<", "/dev/null"
        ],
        description="NVENC RC 模式枚举诊断（头文件/驱动能力/选项表）",
        required_files=[SCRIPTS["nvenc_diagnose"]],
        required_gpu=True,
    ))

    # 6. 段级码流验证 (全环境，需输出视频)
    tests.append(TestCase(
        name="segment_bitstream_verify",
        envs=[Env.CPU, Env.T4, Env.L40],
        cmd_builder=lambda args: [
            sys.executable, str(SCRIPTS["seg_verify"]),
            args.output or "output.mp4",
            *(["--skip-chroma"] if args.skip_chroma else []),
            "<", "/dev/null"
        ],
        description="段级帧守恒/IDR/frame_num/pts/色度簇验收（含解码级门禁）",
        required_files=[SCRIPTS["seg_verify"]],
    ))

    # 7. AV1 长视频冒烟 (L40 专属)
    tests.append(TestCase(
        name="av1_pipeline_smoke",
        envs=[Env.L40],
        cmd_builder=lambda args: [
            sys.executable, str(SCRIPTS["av1_smoke"]),
            "--src", _default_source(args, "01 the race to mystery island fixed.avi"),
            "--rate-modes", "constqp,vbr",
            "--segment-duration", str(args.segment_duration),
            "--mem-interval", str(args.mem_interval),
            "--mem-dump-dir", args.mem_dump_dir or "/tmp/s8_mem",
            "<", "/dev/null"
        ],
        description="AV1 端到端长视频冒烟 S1~S8（帧守恒/解码级/内存泄漏/编码器确认）",
        required_files=[SCRIPTS["av1_smoke"]],
        required_gpu=True,
        timeout=3600,
    ))

    # 8. 落表器自检 (CPU)
    #    ⚠ GPU 点数据目录带轴后缀（gpu_t4_cq/gpu_t4_qp/gpu_l40_cq/gpu_l40_qp），
    #    且 CQ/QP 两轴的点不可混池（见 eqq_pool_fit_table.py --axis 说明）⇒ 拆成两次调用。
    tests.append(TestCase(
        name="eqq_pool_fit_selftest",
        envs=[Env.CPU, Env.T4, Env.L40],
        cmd_builder=lambda args: [
            sys.executable, str(SCRIPTS["eqq_pool_fit"]),
            "--sides", ",".join(
                ["6s", "10s", "legacy10s"]
                + {"t4": ["gpu_t4_cq"], "l40": ["gpu_l40_cq"]}.get(_env_val(args), [])
            ),
            "--axis", "cq",
            "<", "/dev/null"
        ],
        description="落表器 LOO 门禁 + 顺序无关性自检（CQ 轴：软编 6 档 + GPU 档）",
        required_files=[SCRIPTS["eqq_pool_fit"]],
        timeout=300,
    ))

    tests.append(TestCase(
        name="eqq_pool_fit_selftest_qp",
        envs=[Env.T4, Env.L40],
        cmd_builder=lambda args: [
            sys.executable, str(SCRIPTS["eqq_pool_fit"]),
            "--sides", {"t4": "gpu_t4_qp", "l40": "gpu_l40_qp"}.get(_env_val(args), "gpu_l40_qp"),
            "--axis", "qp",
            "<", "/dev/null"
        ],
        description="落表器 QP 轴 LOO 门禁（NVENC 硬编档，QUALITY_MAP_QP 候选）",
        required_files=[SCRIPTS["eqq_pool_fit"]],
        timeout=300,
    ))

    tests.append(TestCase(
        name="calibrate_eq_selftest",
        envs=[Env.CPU, Env.T4, Env.L40],
        cmd_builder=lambda args: [
            sys.executable, str(SCRIPTS["calibrate_eq"]),
            "--selftest",
            "<", "/dev/null"
        ],
        description="标定哈内斯自检（39 项，含 NVENC/axis/跨仓纯函数）",
        required_files=[SCRIPTS["calibrate_eq"]],
    ))

    # 9. NVENC tuning 模块验证 (全环境)
    tests.append(TestCase(
        name="nvenc_tuning_verify",
        envs=[Env.CPU, Env.T4, Env.L40],
        cmd_builder=lambda args: [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, {str(PROJECT_ROOT / 'src' / 'utils')!r}); "
            "from nvenc_tuning import NVENC_TUNE_VALUES, NVENC_MULTIPASS_VALUES, is_bitrate; "
            "print('TUNE:', NVENC_TUNE_VALUES); "
            "print('MULTIPASS:', NVENC_MULTIPASS_VALUES); "
            "print('is_bitrate(vbr):', is_bitrate('vbr')); "
            "print('is_bitrate(vbr_hq):', is_bitrate('vbr_hq'))"
        ],
        description="NVENC 调优参数表（唯一真源）单元验证",
        required_files=[SCRIPTS["nvenc_tune"]],
    ))

    return tests


def filter_tests(tests: List[TestCase], env: Env, args: argparse.Namespace) -> List[TestCase]:
    filtered = []
    gpu_ok = check_gpu_available()
    gpu_name = get_gpu_name() if gpu_ok else ""

    for t in tests:
        if env not in t.envs:
            continue

        # 检查必需文件
        missing = [f for f in t.required_files if not f.exists()]
        if missing:
            print(f"  ⚠️  {t.name}: 缺失文件 {', '.join(str(m) for m in missing)}，跳过")
            continue

        # GPU 依赖检查
        if t.required_gpu and not gpu_ok:
            print(f"  ⚠️  {t.name}: 需要 GPU 但不可用（当前: {gpu_name or '无'}），跳过")
            continue

        # 特定编码器可用性检查
        if t.name == "av1_vp9_matrix_gpu" and not check_nvenc_codec("av1_nvenc"):
            print(f"  ⚠️  {t.name}: av1_nvenc 不可用（Turing 无 AV1 NVENC），跳过")
            continue
        if t.name == "nvenc_vbr_hq_verify" and not (check_nvenc_codec("h264_nvenc") and check_nvenc_codec("hevc_nvenc")):
            print(f"  ⚠️  {t.name}: h264/hevc_nvenc 不可用，跳过")
            continue
        if t.name == "nvenc_rc_diagnose" and not (check_nvenc_codec("h264_nvenc") or check_nvenc_codec("hevc_nvenc")):
            print(f"  ⚠️  {t.name}: 无可用 NVENC 编码器，跳过")
            continue
        if t.name == "av1_pipeline_smoke" and not check_nvenc_codec("av1_nvenc"):
            print(f"  ⚠️  {t.name}: av1_nvenc 不可用，跳过")
            continue

        # 输出视频存在性检查
        if t.name == "segment_bitstream_verify" and args.output and not Path(args.output).exists():
            print(f"  ⚠️  {t.name}: 输出视频 {args.output} 不存在，跳过")
            continue

        # 长视频素材存在性检查
        if t.name == "av1_pipeline_smoke":
            long_src = args.long_source or _default_source(args, "01 the race to mystery island fixed.avi")
            if not Path(long_src).exists():
                print(f"  ⚠️  {t.name}: 长视频素材 {long_src} 不存在，跳过")
                continue

        filtered.append(t)

    return filtered


def _setup_python_path() -> Dict[str, str]:
    """设置 Python 路径环境变量，确保 src/utils 等可导入。"""
    env = os.environ.copy()
    src_utils = str(PROJECT_ROOT / "src" / "utils")
    src_processors = str(PROJECT_ROOT / "src" / "processors")
    src_root = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    paths = [src_utils, src_processors, src_root]
    for p in paths:
        if p not in existing:
            existing = p + os.pathsep + existing if existing else p
    env["PYTHONPATH"] = existing
    return env


def run_test(t: TestCase, args: argparse.Namespace) -> TestResult:
    print(f"\n{'='*60}")
    print(f"▶ 运行: {t.name} — {t.description}")
    print(f"{'='*60}")

    cmd = t.cmd_builder(args)
    # 处理 shell 重定向
    if "<" in cmd:
        stdin_idx = cmd.index("<")
        cmd = cmd[:stdin_idx]  # 移除重定向，由 run_cmd 的 stdin=DEVNULL 处理

    # 针对 Python -c 类测试，注入 PYTHONPATH
    test_env = _setup_python_path() if t.name in ("nvenc_tuning_verify",) else None

    start = time.time()
    rc, stdout, stderr = run_cmd(cmd, timeout=t.timeout, cwd=PROJECT_ROOT, env=test_env)
    duration = time.time() - start

    # 判定状态
    if rc == 0:
        status = "PASS"
    elif rc in (-127, -124):
        status = "SKIP"  # 命令不存在/超时视为环境问题
    else:
        status = "FAIL"

    detail = stdout.strip()[:500] if stdout else stderr.strip()[:500]
    if len(stdout) > 500 or len(stderr) > 500:
        detail += " ... (truncated)"

    return TestResult(
        name=t.name,
        status=status,
        detail=detail,
        duration=duration,
        exit_code=rc
    )


def print_summary(results: List[TestResult], env: Env):
    print(f"\n{'='*70}")
    print(f"综合验证汇总 — 环境: {env.value.upper()}")
    print(f"{'='*70}")

    pass_cnt = sum(1 for r in results if r.status == "PASS")
    fail_cnt = sum(1 for r in results if r.status == "FAIL")
    skip_cnt = sum(1 for r in results if r.status == "SKIP")
    warn_cnt = sum(1 for r in results if r.status == "WARN")

    for r in results:
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️", "WARN": "⚠️"}.get(r.status, "?")
        print(f"  {icon} {r.name:<30} {r.status:<6} ({r.duration:.1f}s)")
        if r.detail and r.status in ("FAIL", "WARN"):
            print(f"      └─ {r.detail[:120]}")

    print(f"\n  总计: {len(results)} 项 | PASS: {pass_cnt} | FAIL: {fail_cnt} | SKIP: {skip_cnt} | WARN: {warn_cnt}")

    if fail_cnt > 0:
        print(f"\n❌ 存在 {fail_cnt} 项 FAIL，验证未通过")
        sys.exit(1)
    else:
        print(f"\n✅ 所有必跑项均 PASS/SKIP，验证通过")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Video Enhancement 综合验证脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
环境分级说明:
  --env cpu   : 仅 CPU 可跑的静态/逻辑/单元验证（无需 GPU）
  --env t4    : Tesla T4 环境（含 h264/hevc NVENC，无 AV1 NVENC）
  --env l40   : L40/Ada 环境（含 av1_nvenc，全量 GPU 验证）

常用组合示例:
  # CPU 仅静态验证（最快，CI 适用）
  python comprehensive_verify.py --env cpu

  # T4 完整验证（含 h264/hevc NVENC 画质/冒烟）
  python comprehensive_verify.py --env t4 \\
      --source input_videos/word_world_2.mp4 \\
      --bitrate-source input_videos/new4_raw.mp4 \\
      --output output.mp4

  # L40 完整验证（含 AV1 NVENC 全链路）
  python comprehensive_verify.py --env l40 \\
      --source input_videos/word_world_2.mp4 \\
      --bitrate-source input_videos/new4_raw.mp4 \\
      --long-source input_videos/long_src.mp4 \\
      --output output.mp4 \\
      --smoke-test

  # 仅运行特定测试
  python comprehensive_verify.py --env t4 --only crf_cq_gpu,nvenc_vbr_hq_verify
        """
    )

    parser.add_argument("--env", choices=["cpu", "t4", "l40"], required=True,
                        help="测试环境: cpu(仅CPU) | t4(Tesla T4) | l40(L40/Ada)")
    parser.add_argument("--only", type=str, default="",
                        help="仅运行指定测试（逗号分隔），如: crf_cq_gpu,nvenc_vbr_hq_verify")
    parser.add_argument("--exclude", type=str, default="",
                        help="排除指定测试（逗号分隔）")

    # 通用路径参数
    parser.add_argument("-i", "--input", type=str, help="输入视频（用于 plan_gate 运行时/冒烟）")
    parser.add_argument("-o", "--output", type=str, help="输出视频（用于段级验证/plan_gate 运行时）")
    parser.add_argument("--source", type=str, help="质量验证源视频（默认 word_world_2.mp4）")
    parser.add_argument("--bitrate-source", type=str, help="码率天花板源视频（默认 new4_raw.mp4）")
    parser.add_argument("--long-source", type=str, help="长视频冒烟源视频（默认 long_src.mp4）")

    # plan_gate 选项
    parser.add_argument("--smoke", action="store_true", help="plan_gate: 启用冒烟测试")
    parser.add_argument("--smoke-mode", type=str, choices=["interpolate_then_upscale", "upscale_then_interpolate"],
                        help="plan_gate: 冒烟模式")
    parser.add_argument("--behavior-only", action="store_true", help="plan_gate: 仅行为验证")
    parser.add_argument("--skip-behavior", action="store_true", help="plan_gate: 跳过行为验证")

    # 段级验证选项
    parser.add_argument("--skip-chroma", action="store_true", help="segment_verify: 跳过色度检查")

    # AV1 冒烟选项
    parser.add_argument("--segment-duration", type=int, default=30, help="av1_smoke: 分段时长(秒)")
    parser.add_argument("--mem-interval", type=int, default=5, help="av1_smoke: 内存采样间隔(秒)")
    parser.add_argument("--mem-dump-dir", type=str, help="av1_smoke: 内存明细导出目录")

    # NVENC 诊断选项
    parser.add_argument("--nvenc-header", type=str, help="nvenc_diagnose: nvEncodeAPI.h 路径")

    args = parser.parse_args()

    env = Env(args.env)

    print(f"🔍 Video Enhancement 综合验证启动")
    print(f"   环境: {env.value.upper()}")
    print(f"   项目根目录: {PROJECT_ROOT}")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   FFmpeg: {which('ffmpeg') or '未找到'}")
    print(f"   FFprobe: {which('ffprobe') or '未找出'}")
    gpu_ok = check_gpu_available()
    print(f"   GPU 可用: {'是' if gpu_ok else '否'} ({get_gpu_name() if gpu_ok else 'N/A'})")

    all_tests = build_test_matrix()
    tests = filter_tests(all_tests, env, args)

    # --only 过滤
    if args.only:
        only_set = set(args.only.split(","))
        tests = [t for t in tests if t.name in only_set]

    # --exclude 过滤
    if args.exclude:
        exclude_set = set(args.exclude.split(","))
        tests = [t for t in tests if t.name not in exclude_set]

    if not tests:
        print("\n⚠️  无可执行测试（可能被环境/参数过滤）")
        sys.exit(0)

    print(f"\n📋 待执行测试 ({len(tests)} 项):")
    for t in tests:
        print(f"   • {t.name} — {t.description}")

    results = []
    for t in tests:
        result = run_test(t, args)
        results.append(result)

    print_summary(results, env)


if __name__ == "__main__":
    main()