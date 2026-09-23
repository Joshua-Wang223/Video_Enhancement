#!/usr/bin/env python3
"""
视频增强主流程
=============================================
统一调度 IFRNet 插帧 + Real-ESRGAN 超分（深度优化版），支持两种处理顺序：
  interpolate_then_upscale  : 先插帧（IFRNet）→ 再超分（ESRGan）
  upscale_then_interpolate  : 先超分（ESRGan）→ 再插帧（IFRNet）

【完整处理流程】
  阶段 0: 输入验证与环境检查（GPU / CUDA / FFmpeg）
  阶段 1: [可选] 视频预去噪（NAFNet / DnCNN / SCUNet）
  阶段 2: 主处理流水线
         ├─ 模式 A: 先 IFRNet 插帧 → 再 Real-ESRGAN 超分
         ├─ 模式 B: 先 Real-ESRGAN 超分 → 再 IFRNet 插帧
         ├─ 仅超分 : --skip-interpolate
         └─ 仅插帧 : --skip-upscale
  阶段 3: 分段合并 + 音频无损回写 + 完整性校验 + 统计
  阶段 4: [可选] 临时 / 中间文件清理

【底层架构】
  IFRNet     : src/processors/ifrnet_processor_video_optimized.py
               → external/ifrnet_video/main.py (IFRNetVideoProcessor, v6.4.5.1)
  Real-ESRGAN: src/processors/realesrgan_processor_video_optimized.py
               → external/realesrgan_video/main.py (main_optimized, v6.4)

【核心特性（v6 亮点 + 优化版增强）】
  ✓ 全部底层控制参数均可命令行透传（见下方参数列表）
  ✓ 批量模式（--batch-mode）：从 --input-dir 批量读取，输出到 --output-dir
  ✓ 断点恢复：各处理器内部独立管理分段断点，中断后可从断点继续
  ✓ 原始音频从输入一次性提取，合并时无损回写
  ✓ --skip-interpolate / --skip-upscale 可单独运行任一步骤
  ✓ 深度流水线架构（读帧→SR→GFPGAN→写帧 4级并行）+ SR H2D 预取重叠
  ✓ FP16 / torch.compile / CUDA Graph / TensorRT 可选（IFRNet & ESRGan）
  ✓ 可选预去噪阶段（NAFNet / DnCNN / SCUNet）
  ✓ face_enhance 精细控制：批量GFPGAN + 人脸检测阈值 + 自适应批处理 + TRT加速
  ✓ 一次性音频提取 + 最终无损回写
  ✓ 详细环境检查 + 配置摘要 + 后处理完整性校验 / 时长体积统计
  ✓ 配置加载容错（文件缺失时自动回退默认值）
  ✓ Dry-run 模式（--dry-run） + 失败恢复提示 + GPU 峰值显存报告

【v6 透传参数列表】
  IFRNet:
    --use-tensorrt-ifrnet / --no-fp16-ifrnet / --no-compile-ifrnet /
    --no-cuda-graph-ifrnet / --no-hwaccel-ifrnet / --batch-size-ifrnet /
    --max-batch-size-ifrnet / --crf-ifrnet / --codec-ifrnet /
    --ifrnet-model / --ifrnet-model-path / --report-ifrnet /
    --preview-ifrnet / --preview-interval-ifrnet /
    --use-cuda-graph-ifrnet / --use-compile-ifrnet / --no-tensorrt-ifrnet
  ESRGan:
    --use-tensorrt-esrgan / --no-compile-esrgan / --no-cuda-graph-esrgan /
    --no-fp16-esrgan / --no-hwaccel-esrgan / --batch-size-esrgan /
    --prefetch-factor-esrgan / --crf-esrgan / --codec-esrgan / --tile-size /
    --tile-pad / --pre-pad / --denoise-strength / --face-enhance /
    --gfpgan-model / --gfpgan-weight / --gfpgan-batch-size / --report-esrgan /
    --no-tensorrt-esrgan / --use-compile-esrgan / --use-cuda-graph-esrgan /
    --preview-esrgan / --preview-interval-esrgan / --encode-preset-esrgan
  共用:
    --trt-cache-dir（IFRNet 与 ESRGan 共享同一 TRT Engine 缓存目录）
    --quiet-ifrnet   静默 IFRNet 底层输出（如 [FFmpegWriter] 命令行）
    --quiet-esrgan   静默 ESRGAN 底层输出

【优化版相对 v6 的 ESRGan 侧变更】
  · 底层脚本更换为 realesrgan_video/main.py（深度模块化架构 v6.4）
  · 新增 CLI 参数：
      --face-det-threshold         人脸检测置信度阈值
      --no-adaptive-batch-esrgan   禁用自适应批处理
      --gfpgan-trt                 GFPGAN TensorRT 子进程加速
      --encode-preset-esrgan       libx264/libx265 编码预设
      --report-esrgan              JSON 性能报告输出路径
      --preview-esrgan             启用实时预览窗口
      --preview-interval-esrgan    预览帧间隔
  · 默认值请以 config/default_config.json 为准

【IFRNet 侧无变化】
  IFRNet 调用链、参数、覆盖系统与 main_video_v6_single.py 完全一致。

【处理流程示意图】
  interpolate_then_upscale（默认）：
    ┌─────────────┐   IFRNet(×N)  ┌──────────────────┐  ESRGan(×M)  ┌──────────┐
    │ input_video │ ─────────────► │ interpolated segs │ ────────────► │ output   │
    └─────────────┘               └──────────────────┘               └──────────┘

  upscale_then_interpolate：
    ┌─────────────┐  ESRGan(×M)  ┌────────────────┐  IFRNet(×N)  ┌──────────┐
    │ input_video │ ────────────► │ upscaled segs  │ ────────────► │ output   │
    └─────────────┘              └────────────────┘               └──────────┘

【使用示例】

  # 默认：先插帧 2× 再超分 2×
  python main_video_optimized.py -i input.mp4 -o output.mp4

  # 反转顺序 + 不同倍数
  python main_video_optimized.py -i input.mp4 -o output.mp4 \\
      --mode upscale_then_interpolate --upscale-factor 4 --interpolation-factor 2

  # 开启 IFRNet + ESRGan TRT 加速（优化版可额外加 --gfpgan-trt）
  python main_video_optimized.py -i input.mp4 -o output.mp4 \\
      --use-tensorrt-ifrnet --use-tensorrt-esrgan \\
      --batch-size-ifrnet 8 --batch-size-esrgan 16

  # 仅超分（跳过插帧）+ 人脸增强 + 阈值控制
  python main_video_optimized.py -i face.mp4 -o face_4x.mp4 \\
      --skip-interpolate --face-enhance --face-det-threshold 0.7

  # 全 TensorRT 加速（含 GFPGAN TRT）
  python main_video_optimized.py -i input.mp4 -o output.mp4 \\
      --use-tensorrt-ifrnet --use-tensorrt-esrgan --gfpgan-trt --face-enhance

  # 去噪 + 插帧 + 超分 全流水线
  python main_video_optimized.py -i noisy.mp4 -o clean_4x.mp4 \\
      --denoise --denoise-model nafnet --denoise-strength-pre 0.5

  # 低显存模式（分块 + 小批量 + 禁用高级优化）
  python main_video_optimized.py -i input.mp4 -o output.mp4 \\
      --tile-size 512 --batch-size-esrgan 2 --no-cuda-graph-esrgan --no-fp16-esrgan

  # 仅插帧（跳过超分）
  python main_video_optimized.py -i input.mp4 -o output.mp4 \\
      --skip-upscale --interpolation-factor 4

  # 批量模式
  python main_video_optimized.py --batch-mode \\
      --input-dir /data/raw/ --output-dir /data/enhanced/

  # 指定配置文件 + 覆盖关键参数
  python main_video_optimized.py -c my_config.json \\
      -i input.mp4 -o output.mp4 \\
      --interpolation-factor 4 --upscale-factor 2 \\
      --no-fp16-ifrnet --no-compile-esrgan

  # Dry-run（仅查看配置，不实际处理）
  python main_video_optimized.py -i input.mp4 -o output.mp4 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# [FIX-NVML] 明确禁用 PyTorch 基于 NVML 的 CUDA 检测，
# 避免因系统 NVML/RM 版本不匹配导致 INTERNAL ASSERT FAILED。
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")

# ─────────────────────────────────────────────────────────────────────────────
# 路径设置：确保项目内模块可被导入
# ─────────────────────────────────────────────────────────────────────────────
_SRC_DIR  = Path(os.path.abspath(__file__)).parent          # …/src
_BASE_DIR = _SRC_DIR.parent                                 # …/Video_Enhancement

_utils_path      = str(_BASE_DIR / "src" / "utils")
_processors_path = str(_BASE_DIR / "src" / "processors")
for _p in (_utils_path, _processors_path, str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─────────────────────────────────────────────────────────────────────────────
# 项目内部导入
# ─────────────────────────────────────────────────────────────────────────────
from config_manager import Config                    # noqa: E402
# [FIX-STDIN-TTOU] 后台进程组 + tty stdin 时，子 ffmpeg 会对 fd0 调 ioctl(TCSETS)
# 触发 SIGTTOU 被停住（表现为"秒卡、0% CPU、无输出"，极易误判为码流/GPU 故障）。
# 入口处把 fd0 换成 /dev/null，一次修好本进程树内全部子进程。
# 详见 src/utils/stdin_hardening.py。
from stdin_hardening import detach_background_stdin   # noqa: E402
detach_background_stdin()
# [P0-FIX-QUALITY-RANGE] 质量参数量程取编码器技术规范（单一真源 QUALITY_MAP）
from quality_map import literal_range, supports_cq, supports_crf   # noqa: E402
from video_utils import (                            # noqa: E402
    format_time,
    get_video_duration,
    get_video_content_duration,
    merge_videos_by_codec,
    build_color_args,
    smart_extract_audio,
    validate_decodable_video,
    validate_source_video_structurally,
    VideoInfo,
)

# [V1] 可选导入：verify_video_integrity 可能不存在于所有版本
try:
    from video_utils import verify_video_integrity as _verify_integrity
except ImportError:
    _verify_integrity = None


# =============================================================================
# 全局常量                                                              [V1]
# =============================================================================

VERSION = "2.0.0"  # 与 realesrgan_video v6.4 架构对齐

_DEFAULT_CFG = str(_BASE_DIR / "config" / "default_config.json")

SUPPORTED_VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv",
    ".webm", ".ts", ".m4v", ".mpg", ".mpeg", ".3gp",
}

SUPPORTED_ESRGAN_MODELS = [
    "realesr-general-x4v3",
    "RealESRGAN_x4plus",
    "RealESRGAN_x2plus",
    "realesr-animevideov3",
    "RealESRGANv2-animevideo-xsx2",
    "RealESRGAN_x4plus_anime_6B",
]


def _select_optimal_mode(
    config: Config,
    input_video: str,
    mode: str,
    quiet: bool = False
) -> str:
    """
    根据输入分辨率和超分倍数，自动选择最优处理模式。

    Args:
        config: 配置对象
        input_video: 输入视频路径
        mode: 用户指定的模式
        quiet: 是否静默模式

    Returns:
        优化后的模式字符串
    """
    # 仅对 upscale_then_interpolate 模式进行自动优化
    if mode != "upscale_then_interpolate":
        return mode

    # 读取可配置阈值（0 = 禁用自动切换）
    # 默认 4K UHD (3840×2160) = 8294400：超分后达到 4K 才自动切换
    max_pixels = config.get("processing", "max_upscale_then_interpolate_pixels", default=8294400)
    if max_pixels <= 0:
        return mode

    try:
        vi = VideoInfo(input_video)
        if not vi.width or not vi.height:
            return mode
    except Exception:
        return mode

    upscale_factor = config.get("processing", "upscale_factor", default=2)
    post_w = vi.width * upscale_factor
    post_h = vi.height * upscale_factor
    post_pixels = post_w * post_h

    if post_pixels > max_pixels:
        if not quiet:
            print()
            print("⚠️" + "=" * 68 + "⚠️")
            print(f"  检测到超分后分辨率 {post_w}×{post_h} ({post_pixels/1e6:.1f}M 像素)")
            print(f"  超过阈值 {max_pixels/1e6:.1f}M 像素 (config: max_upscale_then_interpolate_pixels)")
            print(f"  原因: upscale_then_interpolate 会在 {post_w}×{post_h} 下运行 IFRNet 插帧")
            print(f"        T4 显存/算力不足，极易导致早期 EOF / 帧丢失")
            print(f"  动作: 自动切换为 interpolate_then_upscale 模式（先插帧再超分）")
            print(f"        插帧在 {vi.width}×{vi.height} 下进行，像素吞吐降低 {post_pixels/(vi.width*vi.height):.1f}×")
            print("⚠️" + "=" * 68 + "⚠️")
            print()
        return "interpolate_then_upscale"

    return mode


# =============================================================================
# UI 辅助函数                                                          [V1]
# =============================================================================

def _print_banner():
    """打印启动横幅。"""
    print()
    print("═══════════════════════════════════════════════════════════════════")
    print("    🎬  视频增强流水线 (优化版) v{:<18s}".format(VERSION))
    print("    IFRNet + Real-ESRGAN  ·  realesrgan_video v6.4")
    print("═══════════════════════════════════════════════════════════════════")
    print()


_LOG = None  # [P2.5] 延迟初始化（init_logging 之后可用）


def _print_stage(stage_num: int, title: str, emoji: str = "🔷", quiet: bool = False):
    """打印阶段标题。"""
    if not quiet:
        print()
        print(f"{'─' * 64}")
        print(f"  {emoji}  阶段 {stage_num}: {title}")
        print(f"{'─' * 64}")


def _fmt_time(seconds: float) -> str:
    """将秒数格式化为中文可读时间字符串。"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}分{s:.1f}秒"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}时{int(m)}分{s:.0f}秒"


# =============================================================================
# 验证辅助函数                                                         [V1]
# =============================================================================

def _validate_input(input_path: str) -> bool:
    """验证输入视频文件：是否存在、是否文件、扩展名、文件大小。"""
    p = Path(input_path)
    if not p.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        return False
    if not p.is_file():
        print(f"❌ 输入路径不是文件: {input_path}")
        return False
    if p.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
        print(f"⚠️  文件扩展名 '{p.suffix}' 不在常见视频格式列表中，将尝试处理...")
    if p.stat().st_size == 0:
        print(f"❌ 输入文件大小为 0: {input_path}")
        return False
    return True


def _ensure_output_dir(output_path: str) -> bool:
    """确保输出目录存在。"""
    out_dir = Path(output_path).parent
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"❌ 无法创建输出目录 {out_dir}: {e}")
        return False


# =============================================================================
# 环境检查                                                              [V1]
# =============================================================================

def _check_environment() -> dict:
    """检查运行环境：Python / PyTorch / CUDA / GPU / FFmpeg。"""
    import torch

    env_info: dict = {
        "python_version":   sys.version.split()[0],
        "torch_version":    torch.__version__,
        "cuda_available":   torch.cuda.is_available(),
        "gpu_name":         None,
        "gpu_memory_gb":    None,
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
    }
    if env_info["cuda_available"]:
        env_info["gpu_name"]      = torch.cuda.get_device_name(0)
        mem_bytes                  = torch.cuda.get_device_properties(0).total_memory
        env_info["gpu_memory_gb"] = round(mem_bytes / (1024 ** 3), 1)
    return env_info


def _print_environment(env_info: dict):
    """打印环境信息。"""
    print("🖥️  运行环境:")
    print(f"   Python : {env_info['python_version']}")
    print(f"   PyTorch: {env_info['torch_version']}")
    if env_info["cuda_available"]:
        print(f"   GPU    : {env_info['gpu_name']}")
        print(f"   显存   : {env_info['gpu_memory_gb']} GB")
    else:
        print("   GPU    : 不可用（将使用 CPU，速度极慢）")
    print(f"   FFmpeg : {'✅ 可用' if env_info['ffmpeg_available'] else '❌ 未找到'}")


# =============================================================================
# 配置摘要
# =============================================================================

def _quality_label(getter) -> str:
    """把某环节的质量参数渲染成摘要里的一小段（基准参数优先于字面量）。

    实际下发的数值要等编码器自动升级/降级之后才能确定，由
    ``quality_map.resolve_quality()`` 在后端换算；这里只呈现用户的输入意图。
    """
    ref = getter("crf_ref")
    if ref is not None:
        return f" | CRF-ref: {ref}（libx264 基准，按等效表换算）"
    cq_ref = getter("cq_ref")
    if cq_ref is not None:
        return f" | CQ-ref: {cq_ref}（h264_nvenc 基准，按等效表换算）"
    cq = getter("cq")
    if cq is not None:
        return f" | CQ: {cq}"
    _crf = getter("crf")
    if _crf is not None:
        return f" | CRF: {_crf}"
    # [QUALITY-UNIFY] 未显式给质量 → 后端按 libx264 CRF 基准 21 换算
    return " | 质量: 默认基准 21"


def _print_startup_info(config: Config, args: argparse.Namespace, mode: str) -> None:
    """打印完整启动配置摘要（覆盖 V1 的 print_config_summary 与 V2 的 _print_startup_info）。"""
    ifr = lambda k, d=None: config.get("models", "ifrnet",     k, default=d)
    esr = lambda k, d=None: config.get("models", "realesrgan", k, default=d)

    print("\n" + "─" * 70)
    print("  📋 配置摘要")
    print("─" * 70)

    # 全局
    _in  = getattr(args, "input",  None) or "(批量模式)"
    _out = getattr(args, "output", None) or "(批量模式)"
    print(f"  输入          : {_in}")
    print(f"  输出          : {_out}")
    print(f"  处理模式      : {mode}")
    if getattr(args, "skip_interpolate", False):
        print("  ⚠️  跳过插帧步骤（仅超分）")
    if getattr(args, "skip_upscale", False):
        print("  ⚠️  跳过超分步骤（仅插帧）")
    print(f"  插帧倍数      : "
          f"{config.get('processing', 'interpolation_factor', default=2)}×"
          f"  |  超分倍数: "
          f"{config.get('processing', 'upscale_factor', default=2)}×")
    print(f"  分段时长      : "
          f"{config.get('processing', 'segment_duration', default=30)}秒")

    # ── IFRNet ────────────────────────────────────────────────────────────────
    print()
    print("  ── IFRNet ──")
    print(f"     模型       : {ifr('model_name', 'IFRNet_S_Vimeo90K')}")
    _ifr_path = ifr("model_path", "")
    if _ifr_path:
        print(f"     模型路径   : {_ifr_path}")
    print(f"     FP16       : {ifr('use_fp16', True)}"
          f"  |  compile: {ifr('use_compile', True)}"
          f"  |  CUDA Graph: {ifr('use_cuda_graph', True)}"
          f"  |  TRT: {ifr('use_tensorrt', False)}")
    print(f"     batch_size : {ifr('batch_size', 4)}"
          f" (上限 {ifr('max_batch_size', 8)})"
          f"  |  NVDEC: {ifr('use_hwaccel', True)}")

    if getattr(args, "codec_ifrnet", None) == 'copy':
        print(f"     编码器     : copy")
    else:
        print(f"     编码器     : {ifr('codec', 'libx264')}"
              f"{_quality_label(ifr)}"
              f" | preset: {ifr('encode_preset', 'medium')}")

    # ── Real-ESRGAN ──────────────────────────────────────────────────────────
    print()
    print("  ── Real-ESRGAN（优化版）──")
    print(f"     模型       : {esr('model_name', 'realesr-general-x4v3')}")
    print(f"     降噪强度   : {esr('denoise_strength', 0.5)}")
    _tile = esr("tile_size", 0)
    print(f"     Tile       : "
          f"{_tile if _tile and int(_tile) > 0 else '禁用(整图推理)'}")
    print(f"     FP16       : {esr('use_fp16', True)}"
          f"  |  compile: {esr('use_compile', True)}"
          f"  |  CUDA Graph: {esr('use_cuda_graph', True)}"
          f"  |  TRT: {esr('use_tensorrt', False)}")
    print(f"     batch_size : {esr('batch_size', 6)}"
          f"  |  prefetch: {esr('prefetch_factor', 48)}"
          f"  |  NVDEC: {esr('use_hwaccel', True)}")

    if getattr(args, "codec_esrgan", None) == 'copy':
        print(f"     编码器     : copy")
    else:
        print(f"     编码器     : {esr('codec', 'libx264')}"
              f"{_quality_label(esr)}"
              f" | preset: {esr('encode_preset', 'medium')}")

    # face_enhance
    face_on = esr("face_enhance", False)
    if face_on:
        print(f"     👤 face_enhance: GFPGAN-{esr('gfpgan_model', '1.4')}"
              f" | weight={esr('gfpgan_weight', 0.5)}"
              f" | gfpgan_batch={esr('gfpgan_batch_size', 8)}")
        print(f"        det_threshold={esr('face_det_threshold', 0.5)}"
              f" | adaptive_batch={esr('adaptive_batch', True)}"
              f" | gfpgan_trt={esr('gfpgan_trt', False)}")
    else:
        print("     face_enhance: 关闭")

    # TRT 缓存
    _use_trt = ifr("use_tensorrt", False) or esr("use_tensorrt", False)
    if _use_trt:
        _tcd = (config.get("paths", "trt_cache_dir", default="")
                or f"(自动: {_BASE_DIR}/.trt_cache)")
        print(f"\n  TRT Engine 缓存: {_tcd}")

    # 预去噪
    if getattr(args, "denoise", False):
        print(f"\n  🧹 预去噪阶段 : {getattr(args, 'denoise_model', 'nafnet')}"
              f" (strength={getattr(args, 'denoise_strength_pre', 0.5)})")
    else:
        print(f"\n  预去噪阶段    : 关闭")

    # 最终合并输出
    _use_copy = config.get("output", "use_copy", default=True)
    if _use_copy:
        # --output-codec/crf/preset 均未在 CLI 指定 → stream copy
        # 显示第二阶段处理器的编码参数（决定实际画质的是第二阶段）
        if mode == "upscale_then_interpolate":
            _2nd_codec  = ifr("codec",       "libx264")
            _2nd_q      = _quality_label(ifr)
            _2nd_preset = ifr("encode_preset", "medium")
            _2nd_label  = "IFRNet"
        else:  # interpolate_then_upscale / skip_*
            _2nd_codec  = esr("codec",       "libx264")
            _2nd_q      = _quality_label(esr)
            _2nd_preset = esr("encode_preset", "medium")
            _2nd_label  = "ESRGan"
        _norm_skip = getattr(args, "skip_seg_normalize", False)
        _norm_note = ("已禁用 --skip-seg-normalize" if _norm_skip
                      else "含分段 timescale 归一化")
        print(f"  最终合并输出  : -c:v copy ({_norm_note})"
              f"（继承 {_2nd_label}: {_2nd_codec}"
              f"{_2nd_q}"
              f" | preset: {_2nd_preset}）")
    else:
        _oc   = config.get("output", "codec",  default="")
        _op   = config.get("output", "preset", default="")
        _out_get = lambda k, d=None: config.get("output", k, default=d)
        print(f"  最终合并输出  : codec={_oc or '默认'}"
              f"{_quality_label(_out_get)}"
              f" | preset={_op or '默认'}")

    # 预览与报告
    _preview = esr("preview", False)
    _report = esr("report_json", "")
    if _preview:
        print(f"  实时预览      : 启用 (间隔 {esr('preview_interval', 30)} 帧)")
    if _report:
        print(f"  性能报告(ESR) : {_report}")
    _main_report = config.get("output", "report_path", default="")
    if _main_report:
        print(f"  流水线报告    : {_main_report}")
    print("─" * 70 + "\n")


# =============================================================================
# 去噪阶段（可选前处理）                                                [V1]
# =============================================================================

def _run_denoise_stage(input_path: str, output_path: str,
                       config: Config,
                       args: argparse.Namespace) -> Optional[str]:
    """
    可选视频预去噪。

    未启用（--denoise）时直接返回原始路径。
    去噪器不可用 / 失败时回退到原始路径（不中断流水线）。

    Returns:
        供后续阶段使用的视频路径；None 表示不可恢复的错误。
    """
    if not getattr(args, "denoise", False):
        return input_path

    _print_stage(1, "视频预去噪", "🧹", quiet=args.quiet)

    dn_model    = getattr(args, "denoise_model",       "nafnet")
    dn_strength = getattr(args, "denoise_strength_pre", 0.5)

    print(f"   去噪模型: {dn_model}")
    print(f"   去噪强度: {dn_strength}")

    out_dir  = Path(output_path).parent
    out_stem = Path(input_path).stem
    denoised = str(out_dir / f"{out_stem}_denoised.mp4")

    try:
        try:
            from denoise_processor import DenoiseProcessor        # noqa
            proc = DenoiseProcessor(config)
            proc.denoise_strength = dn_strength
            proc.model_name       = dn_model
            ok = proc.process_video(input_path, denoised)
        except ImportError:
            print("   ⚠️  去噪处理器 (denoise_processor) 未找到")
            print("   📝 跳过去噪阶段，直接使用原始输入")
            return input_path

        if ok and Path(denoised).exists():
            sz = Path(denoised).stat().st_size / (1024 * 1024)
            print(f"   ✅ 去噪完成: {denoised} ({sz:.1f} MB)")
            return denoised
        else:
            print("   ⚠️  去噪失败，使用原始输入继续")
            return input_path

    except Exception as e:
        print(f"   ⚠️  去噪阶段异常: {e}，使用原始输入继续")
        return input_path


# =============================================================================
# LA 音频同步修正                                                       [V2]
# =============================================================================

def _trim_audio_start(audio_path: str, offset_seconds: float) -> Optional[str]:
    """用 ffmpeg 修剪音频开头指定秒数，返回修剪后文件路径。

    使用 -ss X -i input -c copy（输入 seek + 流复制），快速不重编码。
    AAC 帧精度 ~21ms，对人耳不可感知。
    """
    import subprocess as _subprocess
    p = Path(audio_path)
    out = p.parent / f"trimmed_{p.name}"
    try:
        _subprocess.run([
            "ffmpeg", "-y",
            "-ss", f"{offset_seconds:.6f}",
            "-i", str(p),
            "-c", "copy",
            str(out)
        ], check=True, capture_output=True, text=True, timeout=60)
        if out.exists() and out.stat().st_size > 0:
            return str(out)
        return None
    except Exception as e:
        print(f"   ⚠️ 音频修剪失败 ({Path(audio_path).name}): {e}")
        return None


def _rewrite_audio_and_cleanup(output_video: str, audio_path: Optional[str],
                               input_video: str, config) -> bool:
    """[P1-FIX-AUDIO-CLEAN] skip 单步模式的音频回写 + 中间产物回收。

    两条 skip 分支原为近乎复制的内联块，且异常时残留 <output>.with_audio.mp4。
    统一收口：成功后 move；任何路径 finally 清理中间文件。返回回写是否成功。
    """
    if not audio_path:
        return False
    _tmp_out = output_video + ".with_audio.mp4"
    try:
        print("🎵 回写音频到输出视频...")
        _merge_cfg = config.get_section("output", {})
        # [COLOR-FIX] 合并输出注入源视频色彩元数据（有值透传，无值回退 BT.709+Full Range）
        _merge_cfg = {
            **_merge_cfg,
            # [QUALITY-UNIFY] 音频回写只是容器级 remux，强制 -c:v copy，避免二次代损
            "codec": "copy",
            "extra_args": list(_merge_cfg.get("extra_args", []))
                          + build_color_args(input_video),
        }
        audio_ok = merge_videos_by_codec(
            [output_video], _tmp_out,
            audio_path=audio_path,
            config=_merge_cfg,
            reencode=False,
            # [META-KEEP] 回写原片容器级元数据（tags / creation_time / 旋转 / 位深）
            source_video=input_video,
        )
        if audio_ok:
            shutil.move(_tmp_out, output_video)
            print("   ✅ 音频已回写")
        else:
            print("   ⚠️  音频回写失败，输出将无音频")
        return bool(audio_ok)
    except Exception as e:
        print(f"   ⚠️  音频回写异常: {e}")
        return False
    finally:
        try:
            Path(_tmp_out).unlink(missing_ok=True)
        except Exception:
            pass


def _cleanup_audio_temp(audio_path: Optional[str]):
    """[P1-FIX-AUDIO-CLEAN] 回收提取的音频临时文件（含指纹侧车）。"""
    if not audio_path:
        return
    for _f in (audio_path, audio_path + ".src.json"):
        try:
            Path(_f).unlink(missing_ok=True)
        except Exception:
            pass


# =============================================================================
# 后处理与验证阶段                                                      [V1]
# =============================================================================

def _run_postprocess_stage(output_path: str,
                           original_input: str,
                           denoised_path: Optional[str],
                           args: argparse.Namespace,
                           env_info: dict):
    """
    后处理：完整性验证 → 体积/时长统计 → GPU 峰值显存 → 中间文件清理。
    """
    _print_stage(3, "后处理与验证", "🔍", quiet=args.quiet)

    if not Path(output_path).exists():
        print("   ❌ 最终输出文件不存在")
        return

    # ── 完整性验证 ────────────────────────────────────────────────────────────
    if _verify_integrity is not None:
        try:
            if _verify_integrity(output_path):
                print("   ✅ 输出视频完整性验证通过")
            else:
                print("   ⚠️  输出视频可能不完整，请检查")
        except Exception as e:
            print(f"   ⚠️  完整性验证异常: {e}")
    else:
        print("   ℹ️  完整性验证不可用（verify_video_integrity 未导入）")

    # ── 统计信息 ──────────────────────────────────────────────────────────────
    try:
        in_sz  = Path(original_input).stat().st_size / (1024 * 1024)
        out_sz = Path(output_path).stat().st_size    / (1024 * 1024)
        in_dur  = get_video_duration(original_input)
        out_dur = get_video_duration(output_path)

        print(f"\n   📊 统计:")
        print(f"      输入 : {in_sz:.1f} MB"
              f"{f', {_fmt_time(in_dur)}'  if in_dur  else ''}")
        print(f"      输出 : {out_sz:.1f} MB"
              f"{f', {_fmt_time(out_dur)}' if out_dur else ''}")
        if in_sz > 0:
            print(f"      体积比: {out_sz / in_sz:.2f}x")
        if in_dur and out_dur:
            diff = abs(out_dur - in_dur)
            if diff > 1.0:
                print(f"      ⚠️  时长差异: {diff:.1f}秒")
            else:
                print(f"      时长差异: {diff:.2f}秒 ✅")
    except Exception as e:
        print(f"   ⚠️  统计信息获取异常: {e}")

    # ── GPU 峰值显存 ──────────────────────────────────────────────────────────
    if env_info.get("cuda_available"):
        try:
            import torch
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"      🖥️  GPU峰值显存: {peak:.2f} GB")
        except Exception:
            pass

    # ── 清理去噪中间文件 ──────────────────────────────────────────────────────
    if (denoised_path
            and denoised_path != original_input
            and Path(denoised_path).exists()):
        if getattr(args, "keep_intermediate", False):
            print(f"\n   📂 去噪中间文件保留: {denoised_path}")
        else:
            try:
                Path(denoised_path).unlink()
                print(f"\n   🧹 已清理去噪中间文件: {Path(denoised_path).name}")
            except Exception as e:
                print(f"   ⚠️  清理去噪中间文件失败: {e}")


# =============================================================================
# 完成 / 失败提示                                                       [V1]
# =============================================================================

def _print_completion(output_path: str, elapsed: float, env_info: dict):
    """打印成功完成横幅。"""
    print()
    print("═══════════════════════════════════════════════════════════════════")
    print("                   🎉  处理完成！")
    print("═══════════════════════════════════════════════════════════════════")
    print(f"   📤 输出文件: {output_path}")
    print(f"   ⏱️  总用时  : {_fmt_time(elapsed)}")
    if os.path.exists(output_path):
        print(f"   📦 文件大小: "
              f"{os.path.getsize(output_path) / (1024 * 1024):.1f} MB")
    if env_info.get("cuda_available"):
        try:
            import torch
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"   🖥️  GPU峰值显存: {peak:.2f} GB")
        except Exception:
            pass
    print()



# =============================================================================
# 最终报告写出                                                  [--report]
# =============================================================================

def _write_final_report(
    report_path:  str,
    input_video:  str,
    output_video: str,
    mode:         str,
    elapsed:      float,
    success:      bool,
    env_info:     dict,
    args:         argparse.Namespace,
    extra:        Optional[dict] = None,
) -> None:
    """
    将本次流水线运行摘要写入 JSON 文件（由 --report 指定路径）。

    字段说明
    --------
    version             脚本版本
    timestamp           写入时间（ISO-8601）
    success             是否成功
    input / output      输入 / 输出路径
    mode                处理模式
    elapsed_seconds     总耗时（秒）
    elapsed_human       格式化耗时
    input_size_mb / output_size_mb / size_ratio
    input_duration_s / output_duration_s
    gpu_peak_memory_gb  GPU 峰值显存（GB）
    env                 运行环境（Python / PyTorch / GPU）
    params              本次运行的关键 CLI 参数快照
    extra               调用方附加的额外 kv（如批量索引）
    """
    import json
    import datetime

    report: dict = {
        "version":        VERSION,
        "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
        "success":        success,
        "input":          input_video,
        "output":         output_video,
        "mode":           mode,
        "elapsed_seconds": round(elapsed, 2),
        "elapsed_human":   _fmt_time(elapsed),
    }

    # ── 文件大小 / 时长 ───────────────────────────────────────────────────────
    try:
        report["input_size_mb"] = round(
            Path(input_video).stat().st_size / (1024 * 1024), 2)
    except Exception:
        report["input_size_mb"] = None

    if success and os.path.exists(output_video):
        try:
            out_sz = Path(output_video).stat().st_size / (1024 * 1024)
            report["output_size_mb"] = round(out_sz, 2)
            in_sz = report["input_size_mb"] or 0
            report["size_ratio"] = round(out_sz / in_sz, 3) if in_sz else None
        except Exception:
            report["output_size_mb"] = None
            report["size_ratio"]     = None
    else:
        report["output_size_mb"] = None
        report["size_ratio"]     = None

    try:
        report["input_duration_s"] = round(
            get_video_duration(input_video) or 0, 3)
    except Exception:
        report["input_duration_s"] = None

    if success and os.path.exists(output_video):
        try:
            report["output_duration_s"] = round(
                get_video_duration(output_video) or 0, 3)
        except Exception:
            report["output_duration_s"] = None
    else:
        report["output_duration_s"] = None

    # ── GPU 峰值显存 ──────────────────────────────────────────────────────────
    report["gpu_peak_memory_gb"] = None
    if env_info.get("cuda_available"):
        try:
            import torch
            report["gpu_peak_memory_gb"] = round(
                torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        except Exception:
            pass

    # ── 运行环境 ──────────────────────────────────────────────────────────────
    report["env"] = {
        "python":        env_info.get("python_version"),
        "torch":         env_info.get("torch_version"),
        "gpu_name":      env_info.get("gpu_name"),
        "gpu_memory_gb": env_info.get("gpu_memory_gb"),
        "ffmpeg":        env_info.get("ffmpeg_available"),
    }

    # ── 关键 CLI 参数快照 ─────────────────────────────────────────────────────
    _snap_keys = [
        "mode", "interpolation_factor", "upscale_factor", "segment_duration",
        "skip_interpolate", "skip_upscale",
        "batch_size_ifrnet", "max_batch_size_ifrnet",
        "no_fp16_ifrnet", "no_compile_ifrnet", "no_cuda_graph_ifrnet",
        "use_tensorrt_ifrnet", "no_tensorrt_ifrnet",
        # 分段输出质量参数（字面量 / 基准轴，两侧同构）
        "crf_ifrnet", "cq_ifrnet", "crf_ifrnet_ref", "cq_ifrnet_ref",
        "codec_ifrnet", "encode_preset_ifrnet",
        "rate_mode_ifrnet", "lookahead_depth_ifrnet",
        "batch_size_esrgan", "prefetch_factor_esrgan",
        "no_fp16_esrgan", "no_compile_esrgan", "no_cuda_graph_esrgan",
        "use_tensorrt_esrgan", "no_tensorrt_esrgan",
        "crf_esrgan", "cq_esrgan", "crf_esrgan_ref", "cq_esrgan_ref",
        "codec_esrgan", "encode_preset_esrgan",
        "rate_mode_esrgan", "lookahead_depth_esrgan",
        "tile_size", "tile_pad", "pre_pad", "denoise_strength",
        "face_enhance", "gfpgan_model", "gfpgan_weight",
        "gfpgan_batch_size", "face_det_threshold",
        "no_adaptive_batch_esrgan", "gfpgan_trt",
        "output_codec", "output_crf", "output_cq",
        "output_crf_ref", "output_cq_ref", "output_preset",
        "split_codec", "split_crf_ref", "split_cq_ref", "split_preset",
        "denoise", "denoise_model", "denoise_strength_pre",
        "auto_cleanup", "keep_intermediate",
    ]
    report["params"] = {
        k: getattr(args, k, None)
        for k in _snap_keys
        if hasattr(args, k)
    }

    # ── 附加字段（如批量索引）────────────────────────────────────────────────
    if extra:
        report["extra"] = extra

    # ── 写出 JSON ─────────────────────────────────────────────────────────────
    try:
        out_p = Path(report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        print(f"   📄 流水线报告已写出: {report_path}")
    except Exception as e:
        print(f"   ⚠️  报告写出失败: {e}")

def _print_failure_hints(elapsed: float):
    """打印失败恢复提示。"""
    print(f"\n❌ 处理失败（用时 {_fmt_time(elapsed)}）")
    print("   💡 提示:")
    print("      · 断点已自动保存，重新运行相同命令可从断点恢复")
    print("      · 如遇 OOM: --tile-size 512 --batch-size-esrgan 2"
          " --no-cuda-graph-esrgan")
    print("      · 如遇 TRT 构建失败: 移除 --use-tensorrt-esrgan"
          " / --use-tensorrt-ifrnet")
    print("      · 如遇编码错误: --codec-esrgan libx264 --crf-esrgan 23")


# =============================================================================
# 命令行参数覆盖配置                                                     [V2]
# =============================================================================

def _validate_effective_config(config: Config,
                               args: Optional[argparse.Namespace] = None) -> bool:
    """[P1-FIX-VALIDATE] 对"JSON + CLI 覆盖"后的生效配置做范围校验。

    原实现仅在 Config 加载期执行 _validate_config，CLI 数值参数（argparse 多数
    无范围约束）覆盖后直达 ffmpeg/NVENC。任一项不合法即拒绝启动。

    args 用于"用户是否显式给了某参数"的判定（互斥校验依赖它）；缺省时只做
    范围校验（此时所有质量参数按 JSON 配置值校验）。

    [P0-FIX-QUALITY-RANGE] 质量量程一律取编码器技术规范定义的实际可用范围，
    超限即拒绝启动并给出明确错误，不静默放行、不自动截断。
    """
    if args is None:
        args = argparse.Namespace()      # 缺省：全部质量参数走 JSON 配置值分支
    _errors = []
    seg = config.get("processing", "segment_duration", default=30)
    if not isinstance(seg, (int, float)) or isinstance(seg, bool) or seg <= 0:
        _errors.append(f"processing.segment_duration 必须为正数（当前 {seg!r}）")
    fac = config.get("processing", "interpolation_factor", default=2)
    if not (isinstance(fac, (int, float)) and not isinstance(fac, bool) and fac >= 1.0):
        _errors.append(f"processing.interpolation_factor 必须 ≥1（当前 {fac!r}）")
    up = config.get("processing", "upscale_factor", default=2)
    if not (isinstance(up, (int, float)) and not isinstance(up, bool) and 1.0 <= up <= 4.0):
        _errors.append(f"processing.upscale_factor 需在 1~4（当前 {up!r}）")
    for sect in ("ifrnet", "realesrgan"):
        bs = config.get("models", sect, "batch_size", default=1)
        if not isinstance(bs, int) or isinstance(bs, bool) or bs < 1:
            _errors.append(f"models.{sect}.batch_size 必须 ≥1（当前 {bs!r}）")
        mbs = config.get("models", sect, "max_batch_size", default=None)
        if mbs is not None:
            if not isinstance(mbs, int) or isinstance(mbs, bool) or mbs < 1:
                _errors.append(f"models.{sect}.max_batch_size 必须 ≥1（当前 {mbs!r}）")
            elif isinstance(bs, int) and mbs < bs:
                _errors.append(
                    f"models.{sect}.max_batch_size({mbs}) 不得小于 batch_size({bs})")
        la = config.get("models", sect, "lookahead_depth", default=0)
        if not isinstance(la, int) or isinstance(la, bool) or not (0 <= la <= 32):
            _errors.append(f"models.{sect}.lookahead_depth 需在 0~32（当前 {la!r}）")
        rate = config.get("models", sect, "rate_mode", default="vbr_hq")
        if rate not in ("constqp", "vbr_hq", "qvbr"):
            _errors.append(f"models.{sect}.rate_mode 需为 constqp/vbr_hq/qvbr（当前 {rate!r}）")

    # ── 分段输出质量参数（IFRNet / ESRGan 两侧同规则）─────────────────────────
    # [P0-FIX-QUALITY-RANGE] 所有质量输入（字面量 crf/cq 与基准 crf_ref/cq_ref，
    # CLI 与 JSON 配置两条来源）都必须校验，且量程取"技术规范定义的实际可用范围"：
    #   · 字面量：按**生效编码器**的 QUALITY_MAP 量程（libx264 0~51、libvpx-vp9 0~63、
    #     h264_qsv 1~51 …）。跨族时值会经换算，故取源轴量程（crf→libx264，
    #     cq→h264_nvenc，均为 0~51）。
    #   · 基准轴：统一为 libx264 CRF / h264_nvenc CQ 刻度 0~51。
    # 超限一律**拒绝执行**并给出明确错误，绝不静默放行或自动截断。
    # 互斥判定基于 CLI 实参：config 里 crf 恒有默认值（23），无法区分"用户显式给了"
    # 与"配置默认"，用 config 判互斥会把默认配置误判成冲突。
    for _stage, _sfx in (("IFRNet", "ifrnet"), ("ESRGan", "esrgan")):
        _codec = config.get("models", _sfx, "codec", default="libx264") or "libx264"
        _lit = [(f"--{_n}-{_sfx}", _v) for _n, _v in
                (("crf", getattr(args, f"crf_{_sfx}", None)),
                 ("cq",  getattr(args, f"cq_{_sfx}",  None))) if _v is not None]
        _refs = [(f"--{_n}-{_sfx}-ref", _v) for _n, _v in
                 (("crf", getattr(args, f"crf_{_sfx}_ref", None)),
                  ("cq",  getattr(args, f"cq_{_sfx}_ref",  None))) if _v is not None]
        _ln = [n for n, _ in _lit]
        _rn = [n for n, _ in _refs]
        if len(_rn) > 1:
            _errors.append(f"{_stage} 质量基准参数互斥：{' / '.join(_rn)} 只能给一个")
        if _rn and _ln:
            _errors.append(f"{_stage} 质量参数互斥：基准参数 {' / '.join(_rn)}"
                           f" 不能与字面量参数 {' / '.join(_ln)} 同时使用")

        # 字面量量程：随生效编码器而定；CLI 显式给出的用 CLI 标签，否则校验配置值
        for _kind in ("crf", "cq"):
            _cli_v = getattr(args, f"{_kind}_{_sfx}", None)
            if _cli_v is not None:
                _label, _val = f"--{_kind}-{_sfx}", _cli_v
            else:
                _val = config.get("models", _sfx, _kind, default=None)
                _label = f"models.{_sfx}.{_kind}"
                if _val is None:
                    continue
            _lo, _hi = literal_range(_codec, _kind)
            if not (isinstance(_val, int) and not isinstance(_val, bool)
                    and _lo <= _val <= _hi):
                _native = (supports_cq(_codec) if _kind == "cq"
                           else supports_crf(_codec))
                if _native:
                    _errors.append(
                        f"{_label} 超出编码器 {_codec} 的可用质量范围 "
                        f"{_lo}~{_hi}（当前 {_val!r}）")
                else:
                    _axis = "h264_nvenc CQ" if _kind == "cq" else "libx264 CRF"
                    _errors.append(
                        f"{_label} 超出 {_axis} 量纲的可用质量范围 {_lo}~{_hi}"
                        f"（当前 {_val!r}）：生效编码器 {_codec} 不使用该参数，"
                        f"值会按等效表换算，故须落在源轴量程内")

        # 基准轴量程：统一为 0~51（libx264 CRF / h264_nvenc CQ 刻度）
        for _kind in ("crf", "cq"):
            _cli_v = getattr(args, f"{_kind}_{_sfx}_ref", None)
            if _cli_v is not None:
                _label, _val = f"--{_kind}-{_sfx}-ref", _cli_v
            else:
                _val = config.get("models", _sfx, f"{_kind}_ref", default=None)
                _label = f"models.{_sfx}.{_kind}_ref"
                if _val is None:
                    continue
            if not (isinstance(_val, int) and not isinstance(_val, bool)
                    and 0 <= _val <= 51):
                _errors.append(
                    f"{_label} 超出质量基准轴可用范围 0~51（当前 {_val!r}）")

    # ── 环节③ 最终合并输出质量（[QUALITY-UNIFY] / [P0-FIX-QUALITY-RANGE]）─────
    _out_codec = str(config.get("output", "codec", default="libx264") or "libx264")
    _out_is_copy = _out_codec.lower() == "copy"
    _o_lit = [(f"--output-{n}", getattr(args, f"output_{n}", None))
              for n in ("crf", "cq")
              if getattr(args, f"output_{n}", None) is not None]
    _o_ref = [(f"--output-{n}-ref", getattr(args, f"output_{n}_ref", None))
              for n in ("crf", "cq")
              if getattr(args, f"output_{n}_ref", None) is not None]
    if len(_o_ref) > 1:
        _errors.append("最终合并质量基准参数互斥："
                       + " / ".join(n for n, _ in _o_ref) + " 只能给一个")
    if _o_ref and _o_lit:
        _errors.append("最终合并质量参数互斥：基准参数 "
                       + " / ".join(n for n, _ in _o_ref)
                       + " 不能与字面量参数 " + " / ".join(n for n, _ in _o_lit)
                       + " 同时使用")
    if _out_is_copy and (_o_lit or _o_ref or getattr(args, "output_preset", None)):
        _errors.append("--output-codec copy 不能与 --output-crf/cq/-ref/preset 同时使用")
    if not _out_is_copy:
        for _kind in ("crf", "cq"):
            _cli_v = getattr(args, f"output_{_kind}", None)
            _val = (_cli_v if _cli_v is not None
                    else config.get("output", _kind, default=None))
            _label = f"--output-{_kind}" if _cli_v is not None else f"output.{_kind}"
            if _val is None:
                continue
            _lo, _hi = literal_range(_out_codec, _kind)
            if not (isinstance(_val, int) and not isinstance(_val, bool)
                    and _lo <= _val <= _hi):
                _errors.append(f"{_label} 超出编码器 {_out_codec} 的可用质量范围 "
                               f"{_lo}~{_hi}（当前 {_val!r}）")
        for _kind in ("crf", "cq"):
            _cli_v = getattr(args, f"output_{_kind}_ref", None)
            _val = (_cli_v if _cli_v is not None
                    else config.get("output", f"{_kind}_ref", default=None))
            _label = (f"--output-{_kind}-ref" if _cli_v is not None
                      else f"output.{_kind}_ref")
            if _val is None:
                continue
            if not (isinstance(_val, int) and not isinstance(_val, bool)
                    and 0 <= _val <= 51):
                _errors.append(
                    f"{_label} 超出质量基准轴可用范围 0~51（当前 {_val!r}）")

    # ── 环节① 归一化质量（[QUALITY-UNIFY] / [P0-FIX-QUALITY-RANGE]）──────────
    _split_codec = str(config.get("split", "codec", default="libx264") or "libx264")
    _s_ref = [(f"--split-{n}-ref", getattr(args, f"split_{n}_ref", None))
              for n in ("crf", "cq")
              if getattr(args, f"split_{n}_ref", None) is not None]
    if len(_s_ref) > 1:
        _errors.append("归一化质量基准参数互斥："
                       + " / ".join(n for n, _ in _s_ref) + " 只能给一个")
    for _kind in ("crf", "cq"):
        _cli_v = getattr(args, f"split_{_kind}_ref", None)
        _val = (_cli_v if _cli_v is not None
                else config.get("split", f"{_kind}_ref", default=None))
        _label = (f"--split-{_kind}-ref" if _cli_v is not None
                  else f"split.{_kind}_ref")
        if _val is None:
            continue
        if not (isinstance(_val, int) and not isinstance(_val, bool)
                and 0 <= _val <= 51):
            _errors.append(
                f"{_label} 超出质量基准轴可用范围 0~51（当前 {_val!r}）")
    for _kind in ("crf", "cq"):
        _val = config.get("split", _kind, default=None)
        if _val is None:
            continue
        _lo, _hi = literal_range(_split_codec, _kind)
        if not (isinstance(_val, int) and not isinstance(_val, bool)
                and _lo <= _val <= _hi):
            _errors.append(f"split.{_kind} 超出编码器 {_split_codec} 的可用质量范围 "
                           f"{_lo}~{_hi}（当前 {_val!r}）")

    gw = config.get("models", "realesrgan", "gfpgan_weight", default=0.7)
    if not (isinstance(gw, (int, float)) and not isinstance(gw, bool) and 0.0 <= gw <= 1.0):
        _errors.append(f"models.realesrgan.gfpgan_weight 需在 0~1（当前 {gw!r}）")
    ts = config.get("models", "realesrgan", "tile_size", default=0)
    if not (ts == 0 or (isinstance(ts, int) and not isinstance(ts, bool) and ts >= 128)):
        _errors.append(f"models.realesrgan.tile_size 为 0（禁用）或 ≥128（当前 {ts!r}）")

    if _errors:
        print("❌ 生效配置未通过范围校验：")
        for e in _errors:
            print(f"   · {e}")
        return False
    return True


def _apply_cli_overrides(config: Config, args: argparse.Namespace) -> None:
    """将命令行参数写入 config 对象，供两个处理器读取。"""

    # ── 全局处理参数 ─────────────────────────────────────────────────────────
    if args.mode:
        config.set("processing", "mode",                 value=args.mode)
    if args.interpolation_factor:
        config.set("processing", "interpolation_factor", value=args.interpolation_factor)
    if args.upscale_factor:
        config.set("processing", "upscale_factor",       value=args.upscale_factor)
    if args.segment_duration is not None:  # [P1-FIX-VALIDATE] falsy 值(0)不再被静默忽略
        config.set("processing", "segment_duration",     value=args.segment_duration)
    if args.skip_validate:
        config.set("processing", "skip_validate", value=True)
    if args.validate_workers is not None:
        config.set("processing", "validate_workers", value=args.validate_workers)
    if args.validate_mode:
        config.set("processing", "validate_mode", value=args.validate_mode)
    if args.validate_gpu_workers is not None:
        config.set("processing", "validate_gpu_workers", value=args.validate_gpu_workers)
    if args.auto_parallel is not None:
        config.set("processing", "auto_parallel", value=args.auto_parallel)
    if args.max_parallel_workers:
        config.set("processing", "max_parallel_workers", value=args.max_parallel_workers)
    if args.auto_cleanup:
        config.set("processing", "auto_cleanup_temp",    value=True)
    if getattr(args, "no_auto_cleanup", False):
        config.set("processing", "auto_cleanup_temp",    value=False)

    # 注：Video_Enhancement 侧不做 color_range 强制覆盖，固定走 auto
    # （源有值取源值、unknown 取 tv）。强制 tv/pc 并在 AI 编码阶段做真实值域
    # 转换的能力已单独立项：Plan/Video_Enhancement_color_range_强制转换_立项Prompt.md

    # ── IFRNet 模型参数（与 v6 完全一致）─────────────────────────────────────
    if args.ifrnet_model_path:
        config.set("models", "ifrnet", "model_path", value=args.ifrnet_model_path)
        config.set("models", "ifrnet", "model_name", value="")
    elif args.ifrnet_model:
        config.set("models", "ifrnet", "model_name", value=args.ifrnet_model)
        config.set("models", "ifrnet", "model_path", value="")
    
    # 重新派生模型路径（基于可能被 CLI 覆盖的 model_name）
    _base_dir = config.get("paths", "base_dir", default="") or os.getcwd()
    config._derive_model_paths(Path(_base_dir))

    # ── IFRNet 推理优化（与 v6 完全一致）─────────────────────────────────────
    if args.no_fp16_ifrnet:
        config.set("models", "ifrnet", "use_fp16",       value=False)
    if args.no_compile_ifrnet:
        config.set("models", "ifrnet", "use_compile",    value=False)
    if args.no_cuda_graph_ifrnet:
        config.set("models", "ifrnet", "use_cuda_graph", value=False)
    if args.use_tensorrt_ifrnet:
        config.set("models", "ifrnet", "use_tensorrt",   value=True)
    if args.no_hwaccel_ifrnet:
        config.set("models", "ifrnet", "use_hwaccel",    value=False)
    if args.batch_size_ifrnet is not None:  # [P1-FIX-VALIDATE]
        config.set("models", "ifrnet", "batch_size",     value=args.batch_size_ifrnet)
    if args.max_batch_size_ifrnet is not None:  # [P1-FIX-VALIDATE]
        config.set("models", "ifrnet", "max_batch_size", value=args.max_batch_size_ifrnet)
    if args.crf_ifrnet is not None:
        config.set("models", "ifrnet", "crf",            value=args.crf_ifrnet)
    if args.cq_ifrnet is not None:
        config.set("models", "ifrnet", "cq",             value=args.cq_ifrnet)
    if args.crf_ifrnet_ref is not None:
        config.set("models", "ifrnet", "crf_ref",        value=args.crf_ifrnet_ref)
        # 配置里 crf_ref 有默认值（21），给了 cq_ref 就必须摘掉它，否则基准轴
        # 判定顺序（crf_ref 优先）会把 --cq-ifrnet-ref 静默吞掉
        config.set("models", "ifrnet", "cq_ref",         value=None)
    if args.cq_ifrnet_ref is not None:
        config.set("models", "ifrnet", "cq_ref",         value=args.cq_ifrnet_ref)
        config.set("models", "ifrnet", "crf_ref",        value=None)
    if (args.crf_ifrnet is not None or args.cq_ifrnet is not None):
        # 字面量与基准轴量纲不同：显式给了字面量就摘掉配置里的默认基准，
        # 否则 crf_ref（默认 21）会静默覆盖用户的 --crf-ifrnet。
        if args.crf_ifrnet_ref is None:
            config.set("models", "ifrnet", "crf_ref",    value=None)
        if args.cq_ifrnet_ref is None:
            config.set("models", "ifrnet", "cq_ref",     value=None)
    if args.codec_ifrnet:
        config.set("models", "ifrnet", "codec",          value=args.codec_ifrnet)
    if args.encode_preset_ifrnet:
        config.set("models", "ifrnet", "encode_preset",  value=args.encode_preset_ifrnet)
    if args.rate_mode_ifrnet:
        config.set("models", "ifrnet", "rate_mode",       value=args.rate_mode_ifrnet)
    if args.lookahead_depth_ifrnet is not None:
        config.set("models", "ifrnet", "lookahead_depth", value=args.lookahead_depth_ifrnet)
    if args.no_audio_ifrnet:
        config.set("models", "ifrnet", "keep_audio",     value=False)
    if args.report_ifrnet:
        config.set("models", "ifrnet", "report_json",    value=args.report_ifrnet)

    # ── IFRNet 高优先级覆盖（与 v6 完全一致）────────────────────────────────
    _ifrnet_overrides: list[str] = []
    if args.no_tensorrt_ifrnet and args.use_tensorrt_ifrnet:
        config.set("models", "ifrnet", "use_tensorrt", value=False)
        config.set("models", "ifrnet", "force_no_tensorrt", value=True)
        _ifrnet_overrides.append(
            "--no-tensorrt-ifrnet  覆盖了  --use-tensorrt-ifrnet  → IFRNet TensorRT 已禁用")
    elif args.no_tensorrt_ifrnet:
        config.set("models", "ifrnet", "use_tensorrt", value=False)
        config.set("models", "ifrnet", "force_no_tensorrt", value=True)

    if args.use_compile_force_ifrnet:
        config.set("models", "ifrnet", "use_compile", value=True)
        config.set("models", "ifrnet", "force_use_compile", value=True)
        if args.no_compile_ifrnet:
            _ifrnet_overrides.append(
                "--use-compile-ifrnet  覆盖了  --no-compile-ifrnet  → IFRNet torch.compile 已启用")
    if args.use_cuda_graph_force_ifrnet:
        config.set("models", "ifrnet", "use_cuda_graph", value=True)
        config.set("models", "ifrnet", "force_use_cuda_graph", value=True)
        if args.no_cuda_graph_ifrnet:
            _ifrnet_overrides.append(
                "--use-cuda-graph-ifrnet  覆盖了  --no-cuda-graph-ifrnet  → IFRNet CUDA Graph 已启用")
    # 互斥冲突预警（基于最终写入 config 的有效值）
    _eff_trt_ifr     = config.get("models", "ifrnet", "use_tensorrt",   default=False)
    _eff_compile_ifr = config.get("models", "ifrnet", "use_compile",    default=True)
    if args.use_cuda_graph_force_ifrnet and _eff_compile_ifr and not _eff_trt_ifr:
        print("[CLI警告] --use-cuda-graph-ifrnet 与 torch.compile 互斥："
              "compile 成功后 CUDA Graph 将被自动禁用。")
        print("          若要确保 CUDA Graph 生效，请同时指定 --no-compile-ifrnet。")
    if args.use_cuda_graph_force_ifrnet and _eff_trt_ifr:
        print("[CLI警告] --use-cuda-graph-ifrnet 与 --use-tensorrt-ifrnet 互斥："
              "TensorRT 优先，CUDA Graph 将被禁用。")
        print("          如需 CUDA Graph，请同时指定 --no-tensorrt-ifrnet。")
    if args.use_compile_force_ifrnet and _eff_trt_ifr:
        print("[CLI警告] --use-compile-ifrnet 与 --use-tensorrt-ifrnet 互斥："
              "TensorRT 优先，compile 将被跳过。")
        print("          如需 torch.compile，请同时指定 --no-tensorrt-ifrnet。")
    if _ifrnet_overrides:
        print("[CLI覆盖] IFRNet 以下设置已被高优先级参数覆盖：")
        for msg in _ifrnet_overrides:
            print(f"          · {msg}")
        print()

    # ── ESRGan 模型参数 ──────────────────────────────────────────────────────
    if args.esrgan_model:
        config.set("models", "realesrgan", "model_name",       value=args.esrgan_model)
    if args.denoise_strength is not None:
        config.set("models", "realesrgan", "denoise_strength", value=args.denoise_strength)

    # ── ESRGan 推理优化 ──────────────────────────────────────────────────────
    if args.no_fp16_esrgan:
        config.set("models", "realesrgan", "use_fp16",        value=False)
    if args.no_compile_esrgan:
        config.set("models", "realesrgan", "use_compile",     value=False)
    if args.no_cuda_graph_esrgan:
        config.set("models", "realesrgan", "use_cuda_graph",  value=False)
    if args.use_tensorrt_esrgan:
        config.set("models", "realesrgan", "use_tensorrt",    value=True)
    if args.no_hwaccel_esrgan:
        config.set("models", "realesrgan", "use_hwaccel",     value=False)
    if args.batch_size_esrgan is not None:  # [P1-FIX-VALIDATE]
        config.set("models", "realesrgan", "batch_size",      value=args.batch_size_esrgan)
    if args.prefetch_factor_esrgan:
        config.set("models", "realesrgan", "prefetch_factor", value=args.prefetch_factor_esrgan)
    if args.tile_size is not None:
        config.set("models", "realesrgan", "tile_size",       value=args.tile_size)
    if args.tile_pad is not None:
        config.set("models", "realesrgan", "tile_pad",        value=args.tile_pad)
    if args.pre_pad is not None:
        config.set("models", "realesrgan", "pre_pad",         value=args.pre_pad)
    if args.crf_esrgan is not None:
        config.set("models", "realesrgan", "crf",             value=args.crf_esrgan)
    if args.cq_esrgan is not None:
        config.set("models", "realesrgan", "cq",              value=args.cq_esrgan)
    if args.crf_esrgan_ref is not None:
        config.set("models", "realesrgan", "crf_ref",         value=args.crf_esrgan_ref)
        # 配置里 crf_ref 有默认值（21），给了 cq_ref 就必须摘掉它，否则基准轴
        # 判定顺序（crf_ref 优先）会把 --cq-esrgan-ref 静默吞掉
        config.set("models", "realesrgan", "cq_ref",          value=None)
    if args.cq_esrgan_ref is not None:
        config.set("models", "realesrgan", "cq_ref",          value=args.cq_esrgan_ref)
        config.set("models", "realesrgan", "crf_ref",         value=None)
    if args.crf_esrgan is not None or args.cq_esrgan is not None:
        # 给了字面量就摘掉配置里的默认基准，否则 crf_ref 会静默覆盖 --crf-esrgan
        if args.crf_esrgan_ref is None:
            config.set("models", "realesrgan", "crf_ref",     value=None)
        if args.cq_esrgan_ref is None:
            config.set("models", "realesrgan", "cq_ref",      value=None)
    if args.codec_esrgan:
        config.set("models", "realesrgan", "codec",           value=args.codec_esrgan)
    if args.encode_preset_esrgan:
        config.set("models", "realesrgan", "encode_preset",   value=args.encode_preset_esrgan)
    if args.rate_mode_esrgan:
        config.set("models", "realesrgan", "rate_mode",       value=args.rate_mode_esrgan)
    if args.lookahead_depth_esrgan is not None:
        config.set("models", "realesrgan", "lookahead_depth", value=args.lookahead_depth_esrgan)
    if getattr(args, "ffmpeg_bin", None):
        config.set("models", "realesrgan", "ffmpeg_bin",      value=args.ffmpeg_bin)

    # ── ESRGan 高优先级覆盖（后写入，覆盖上方 --no-* / config 的值）────────────
    _esr_overrides: list[str] = []
    if args.no_tensorrt_esrgan and args.use_tensorrt_esrgan:
        config.set("models", "realesrgan", "use_tensorrt", value=False)
        _esr_overrides.append(
            "--no-tensorrt-esrgan  覆盖了  --use-tensorrt-esrgan  → ESRGan TensorRT 已禁用")
    elif args.no_tensorrt_esrgan:
        config.set("models", "realesrgan", "use_tensorrt", value=False)

    if args.use_compile_force_esrgan:
        config.set("models", "realesrgan", "use_compile", value=True)
        if args.no_compile_esrgan:
            _esr_overrides.append(
                "--use-compile-esrgan  覆盖了  --no-compile-esrgan  → ESRGan torch.compile 已启用")
    if args.use_cuda_graph_force_esrgan:
        config.set("models", "realesrgan", "use_cuda_graph", value=True)
        if args.no_cuda_graph_esrgan:
            _esr_overrides.append(
                "--use-cuda-graph-esrgan  覆盖了  --no-cuda-graph-esrgan  → ESRGan CUDA Graph 已启用")
    # 互斥冲突预警（基于最终写入 config 的有效值）
    _eff_trt_esr     = config.get("models", "realesrgan", "use_tensorrt",   default=False)
    _eff_compile_esr = config.get("models", "realesrgan", "use_compile",    default=True)
    if args.use_cuda_graph_force_esrgan and _eff_compile_esr and not _eff_trt_esr:
        print("[CLI警告] --use-cuda-graph-esrgan 与 torch.compile 互斥："
              "compile 成功后 CUDA Graph 将被自动禁用。")
        print("          若要确保 CUDA Graph 生效，请同时指定 --no-compile-esrgan。")
    if args.use_cuda_graph_force_esrgan and _eff_trt_esr:
        print("[CLI警告] --use-cuda-graph-esrgan 与 --use-tensorrt-esrgan 互斥："
              "TensorRT 优先，CUDA Graph 将被禁用。")
        print("          如需 CUDA Graph，请同时指定 --no-tensorrt-esrgan。")
    if args.use_compile_force_esrgan and _eff_trt_esr:
        print("[CLI警告] --use-compile-esrgan 与 --use-tensorrt-esrgan 互斥："
              "TensorRT 优先，torch.compile 将被禁用。")
        print("          如需 torch.compile，请同时指定 --no-tensorrt-esrgan。")
    if _esr_overrides:
        print("[CLI覆盖] ESRGan 以下设置已被高优先级参数覆盖：")
        for msg in _esr_overrides:
            print(f"          · {msg}")
        print()

    # ── face_enhance ─────────────────────────────────────────────────────────
    if args.face_enhance is not None:
        config.set("models", "realesrgan", "face_enhance",       value=args.face_enhance)
    if args.gfpgan_model:
        config.set("models", "realesrgan", "gfpgan_model",       value=args.gfpgan_model)
    if args.gfpgan_weight is not None:
        config.set("models", "realesrgan", "gfpgan_weight",      value=args.gfpgan_weight)
    if args.gfpgan_batch_size is not None:  # [P1-FIX-VALIDATE]
        config.set("models", "realesrgan", "gfpgan_batch_size",  value=args.gfpgan_batch_size)
    if args.face_det_threshold is not None:
        config.set("models", "realesrgan", "face_det_threshold", value=args.face_det_threshold)
    if args.no_adaptive_batch_esrgan:
        config.set("models", "realesrgan", "adaptive_batch",     value=False)
    if args.gfpgan_trt:
        config.set("models", "realesrgan", "gfpgan_trt",         value=True)

    # ── 新增：预览与报告参数 ─────────────────────────────────────────────────
    if getattr(args, "report_esrgan", None):
        config.set("models", "realesrgan", "report_json", value=args.report_esrgan)
    if getattr(args, "preview_esrgan", False):
        config.set("models", "realesrgan", "preview", value=True)
    if getattr(args, "preview_interval_esrgan", None) is not None:
        config.set("models", "realesrgan", "preview_interval", value=args.preview_interval_esrgan)

    # ── 最终合并输出参数：copy-by-default ──────────────────────────────────
    # [QUALITY-UNIFY] 四类质量输入互斥（字面量 crf/cq vs 基准轴 crf_ref/cq_ref）。
    # 只要 CLI 显式给了 codec 或任一质量参数，即触发重编码；否则保持 -c:v copy。
    _output_quality_cli = any(getattr(args, _n, None) is not None for _n in (
        "output_crf", "output_cq", "output_crf_ref", "output_cq_ref"))
    _output_codec_copy = (str(getattr(args, "output_codec", "") or "")
                          .strip().lower() == "copy")

    if args.output_codec:
        config.set("output", "codec", value=args.output_codec)
    if args.output_crf is not None:
        config.set("output", "crf", value=args.output_crf)
        config.set("output", "cq", value=None)
        config.set("output", "crf_ref", value=None)
        config.set("output", "cq_ref", value=None)
    if getattr(args, "output_cq", None) is not None:
        config.set("output", "cq", value=args.output_cq)
        config.set("output", "crf", value=None)
        config.set("output", "crf_ref", value=None)
        config.set("output", "cq_ref", value=None)
    if getattr(args, "output_crf_ref", None) is not None:
        config.set("output", "crf_ref", value=args.output_crf_ref)
        config.set("output", "crf", value=None)
        config.set("output", "cq", value=None)
        config.set("output", "cq_ref", value=None)
    if getattr(args, "output_cq_ref", None) is not None:
        config.set("output", "cq_ref", value=args.output_cq_ref)
        config.set("output", "crf", value=None)
        config.set("output", "cq", value=None)
        config.set("output", "crf_ref", value=None)
    if args.output_preset:
        config.set("output", "preset", value=args.output_preset)
    # 写入 use_copy：显式 --output-codec copy（且无质量/preset）也算 copy
    _output_cli_any = (bool(args.output_codec) or _output_quality_cli
                       or bool(args.output_preset))
    if _output_codec_copy and not _output_quality_cli and not args.output_preset:
        _output_cli_any = False
    config.set("output", "use_copy", value=(not _output_cli_any))

    # ── 环节① 归一化质量参数（--normalize-source 重编码时使用）──────────────
    if getattr(args, "split_codec", None):
        config.set("split", "codec", value=args.split_codec)
    if getattr(args, "split_crf_ref", None) is not None:
        config.set("split", "crf_ref", value=args.split_crf_ref)
        config.set("split", "cq_ref", value=None)
    if getattr(args, "split_cq_ref", None) is not None:
        config.set("split", "cq_ref", value=args.split_cq_ref)
        config.set("split", "crf_ref", value=None)
    if getattr(args, "split_preset", None):
        config.set("split", "preset", value=args.split_preset)

    # ── 最终报告输出路径（--report）──────────────────────────────────────
    if getattr(args, "report", None):
        config.set("output", "report_path", value=args.report)

    # ── TRT 缓存目录（全局，IFRNet 与 ESRGan 共享）────────────────────────
    if args.trt_cache_dir:
        config.set("paths", "trt_cache_dir", value=args.trt_cache_dir)



# =============================================================================
# [FIX-C] 分段 timescale 归一化 —— 已迁移至 src/utils/video_utils.py
# =============================================================================
# 原 _get_seg_codec / _normalize_segs_for_copy 已迁移为
# video_utils._detect_seg_codec / normalize_segments_timescale，并由
# merge_videos_by_codec 在 copy 合并前统一调用，使「单阶段」（各 processor
# 内部合并）与「两阶段」（main 层最终合并）路径获得一致保护。
# 旧名 "extradata 归一化" 系早期 -bsf:v dump_extra 实现的遗留，实际动作
# 一直是 timescale 归一化，迁移时一并更正。


# =============================================================================
# 单文件处理
# =============================================================================

def _process_single(
    config:           Config,
    input_video:      str,
    output_video:     str,
    mode:             str,
    skip_interpolate: bool,
    skip_upscale:     bool,
    args:             argparse.Namespace,
    env_info:         dict,
) -> bool:
    """
    处理单个视频文件：
      1. 提取原始音频（若存在）
      2. [可选] 预去噪
      3. 按 mode 链式调用 IFRNetProcessor / RealESRGANVideoProcessor
      4. 合并分段 + 音频无损回写
      5. 后处理验证 + 统计

    Returns:
        是否成功
    """
    from ifrnet_processor_video_optimized       import IFRNetProcessor           # noqa
    from realesrgan_processor_video_optimized   import RealESRGANVideoProcessor  # noqa

    # [P5-FIX-SOURCE-STRUCT-GATE] 结构坏源直接拒绝，防止生成失败/污染中间产物。
    src_ok, src_reason = validate_source_video_structurally(input_video)
    if not src_ok:
        print(f"❌ 源视频结构校验失败: {src_reason}")
        print("   请修复/更换片源；如确需处理请先排除 ffmpeg 报告的码流问题。")
        return False

    t0         = time.time()
    video_name = Path(input_video).stem

    _preview_ifr      = getattr(args, "preview_ifrnet", False)
    _preview_ifr_intv = getattr(args, "preview_interval_ifrnet", 30)

    # ── 打印头部信息 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  🚀 视频增强主流程（优化版）")
    print("=" * 70)
    print(f"  输入  : {input_video}")
    print(f"  输出  : {output_video}")
    print(f"  模式  : {mode}")
    if skip_interpolate:
        print("  ⚠️  跳过插帧步骤（仅超分）")
    if skip_upscale:
        print("  ⚠️  跳过超分步骤（仅插帧）")
    if getattr(args, "denoise", False):
        print(f"  🧹 预去噪: {getattr(args, 'denoise_model', 'nafnet')}"
              f" (strength={getattr(args, 'denoise_strength_pre', 0.5)})")
    print("=" * 70 + "\n")

    # ── 0. [可选] 源时间轴归一化 ─────────────────────────────────────────────
    # [P3-FIX-NORM] VFR 源（时间戳空洞）与非零起始偏移会让段级帧数验收和时长
    # 统计失真；归一化后一次性根除。未开启时只做检测与提示，不改变行为。
    input_video = _run_normalize_source(config, input_video, args)
    if input_video is None:
        return False
    video_name = Path(input_video).stem

    # [P2-FIX-MODE-AUTO] 自动选择最优模式：防止 upscale_then_interpolate 在高分辨率下 OOM/EOF
    mode = _select_optimal_mode(config, input_video, mode, quiet=args.quiet)

    if skip_interpolate and skip_upscale:
        print("❌ --skip-interpolate 与 --skip-upscale 不能同时指定")
        return False

    # ── 1. 提取原始音频（从原始输入一次性提取，后续合并步骤无损回写）────────
    # [P3.1-SPLIT] 阶段方法化：语句顺序与失败路径逐字保持
    audio_path = _extract_source_audio(config, input_video, quiet=args.quiet)

    # ── 2. [可选] 预去噪 ──────────────────────────────────────────────────────
    denoised_path: Optional[str] = None
    actual_input = _run_denoise_stage(input_video, output_video, config, args)
    if actual_input is None:
        print("❌ 去噪阶段失败且无法回退")
        return False
    if actual_input != input_video:
        denoised_path = actual_input
        print(f"   📎 后续处理将使用去噪后文件: {Path(actual_input).name}")

    # ── 3a. 单步模式 —— 仅超分 ───────────────────────────────────────────────
    # [P3.1-SPLIT] 阶段方法化
    if skip_interpolate:
        return _run_upscale_only(config, actual_input, output_video,
                                 audio_path, denoised_path, input_video,
                                 args, env_info, t0)

    # ── 3b. 单步模式 —— 仅插帧 ───────────────────────────────────────────────
    # [P3.1-SPLIT] 阶段方法化（keep_audio 暂存/finally 恢复在方法内闭合）
    if skip_upscale:
        return _run_interpolate_only(config, actual_input, output_video,
                                     audio_path, denoised_path, input_video,
                                     args, env_info, t0,
                                     _preview_ifr, _preview_ifr_intv)

    # ── 3c. 双步模式 —— 处理器创建 + 链式执行 + VRAM 对称释放 ────────────────
    # [P3.1-SPLIT] 阶段方法化；失败返回 None（调用方判败）
    _two_stage = _run_two_stage(config, actual_input, video_name, mode,
                                args, env_info, t0,
                                _preview_ifr, _preview_ifr_intv)
    if _two_stage is None:
        return False
    final_segs, ifrnet_proc, esrgan_proc = _two_stage

    # ── 4. 合并最终分段（含音频无损回写 / LA 修正 / 色彩元数据）──────────────
    # [P3.1-SPLIT] 阶段方法化；audio_path/output_video 可能被方法内改写（LA 修剪/EXT-PROP）
    success, output_video, audio_path = _merge_and_finalize(
        config, final_segs, mode,
        audio_path, input_video, output_video,
        args, ifrnet_proc, esrgan_proc)

    elapsed = time.time() - t0
    _report_path = config.get("output", "report_path", default="")

    if success:
        print(f"\n✅ 全流程完成！总耗时: {format_time(elapsed)}")
        print(f"📤 输出: {output_video}")
        if os.path.exists(output_video):
            print(f"   文件大小: {os.path.getsize(output_video)/1024/1024:.1f} MB")

        # 5. 后处理
        _run_postprocess_stage(output_video, input_video,
                               denoised_path, args, env_info)
        _print_completion(output_video, elapsed, env_info)

        # 清理断点文件（全流程成功完成后无条件删除）
        for proc in (ifrnet_proc, esrgan_proc):
            try:
                proc._delete_checkpoint()
            except Exception:
                pass
        # 自动清理临时分段文件及音频
        if config.get("processing", "auto_cleanup_temp", default=False):
            print("🧹 自动清理临时分段文件...")
            for proc in (ifrnet_proc, esrgan_proc):
                try:
                    proc._cleanup_temp_files()
                except Exception:
                    pass
            # 清理本次提取的临时音频文件（精准删除，避免误删并发流程的文件）
            if audio_path:
                try:
                    audio_file = Path(audio_path)
                    if audio_file.exists():
                        audio_file.unlink()
                except Exception:
                    pass
            print("\n   ✅ 清理完成")
    else:
        print("❌ 最终合并失败")
        # [P1-FIX-AUDIO-CLEAN] 失败路径同样回收音频临时文件，避免磁盘缓慢泄漏
        if audio_path:
            try:
                Path(audio_path).unlink(missing_ok=True)
            except Exception:
                pass
        _print_failure_hints(elapsed)

    # ── 写出最终流水线报告（成功或失败均写）──────────────────────────────
    if _report_path:
        _write_final_report(
            report_path=_report_path,
            input_video=input_video,
            output_video=output_video,
            mode=mode,
            elapsed=elapsed,
            success=success,
            env_info=env_info,
            args=args,
        )

    return success


# ── [P3.1-SPLIT] _process_single 阶段子函数 ───────────────────────────────────

def _run_normalize_source(config: Config, input_video: str,
                          args: argparse.Namespace) -> Optional[str]:
    """[P3-FIX-NORM] 可选前处理：把源时间轴归一化为均匀 CFR。

    · --normalize-source：检测到异常即重编码归一化（一次软编码开销），产物落在
      temp/normalized/<name>_norm.mp4，同名同源且产物不旧于源时直接复用。
    · 未开启：仅检测并提示，返回原路径，行为与之前完全一致。
    """
    from video_utils import detect_timestamp_anomaly, normalize_video_timeline

    enable = bool(getattr(args, "normalize_source", False))
    quiet = bool(getattr(args, "quiet", False))

    try:
        rep = detect_timestamp_anomaly(input_video)
    except Exception as e:
        if not quiet:
            print(f"   ⚠️  源时间轴检测异常（跳过）: {e}")
        return input_video

    if not rep.get("ok"):
        if not quiet:
            print("   ℹ️  源时间轴无法判定，跳过归一化")
        return input_video

    if not rep.get("anomaly"):
        if not quiet:
            print(f"   ✅ 源时间轴均匀（{rep.get('frame_count')} 帧，"
                  f"起始 {rep.get('start_time')}s），无需归一化")
        return input_video

    if not quiet:
        print(f"   ⚠️  检测到源时间轴异常: {rep.get('detail')}")

    if not enable:
        print("      → 如需修复请加 --normalize-source（会做一次重编码）")
        return input_video

    out_dir = Path(config.get_temp_dir("normalized"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(input_video).stem}_norm.mp4"

    try:
        if (out_path.exists()
                and out_path.stat().st_mtime >= Path(input_video).stat().st_mtime):
            print(f"   ♻️  复用已归一化文件: {out_path.name}")
            return str(out_path)
    except OSError:
        pass

    print(f"   🔧 归一化源时间轴 → {out_path.name} …")
    # [QUALITY-UNIFY] 环节① 质量参数来自 --split-* / config.split（默认基准 21）
    _sp = config.get_section("split", {}) or {}
    if not normalize_video_timeline(
            input_video, str(out_path),
            encoder=_sp.get("codec", "libx264"),
            preset=_sp.get("preset", "veryfast"),
            crf=_sp.get("crf"), cq=_sp.get("cq"),
            crf_ref=_sp.get("crf_ref"), cq_ref=_sp.get("cq_ref")):
        print("   ❌ 源时间轴归一化失败，回退使用原始输入")
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass
        return input_video

    if not quiet:
        after = detect_timestamp_anomaly(str(out_path))
        if after.get("ok"):
            print(f"   ✅ 归一化完成: {after.get('frame_count')} 帧 | "
                  f"起始 {after.get('start_time')}s | "
                  f"空洞 {len(after.get('gaps') or [])} 处")
    return str(out_path)


def _extract_source_audio(config: Config, input_video: str, quiet: bool = False) -> Optional[str]:
    """[P3.1-SPLIT] 从原始输入一次性提取音频；失败仅告警不阻断（返回 None）。"""
    audio_path: Optional[str] = None
    try:
        info = VideoInfo(input_video)
        if info.has_audio:
            if not quiet:
                print("🎵 提取原始音频...")
            audio_path = smart_extract_audio(
                input_video,
                str(config.get_temp_dir("main_audio")),
                quiet=quiet,
            )
            if audio_path and not quiet:
                print(f"   ✅ 音频已暂存: {audio_path}")
            elif not audio_path and not quiet:
                print("   ⚠️  音频提取失败，输出将无音频")
    except Exception as e:
        if not quiet:
            print(f"   ⚠️  音频提取异常（继续处理）: {e}")
    return audio_path


def _run_upscale_only(config: Config, actual_input: str, output_video: str,
                      audio_path: Optional[str], denoised_path: Optional[str],
                      input_video: str, args: argparse.Namespace,
                      env_info: dict, t0: float) -> bool:
    """[P3.1-SPLIT] 单步模式——仅超分（含音频回写/后处理/报告）。"""
    from realesrgan_processor_video_optimized import RealESRGANVideoProcessor  # noqa
    _print_stage(2, "Real-ESRGAN 视频超分（优化版）", "🎨", quiet=args.quiet)
    try:
        proc = RealESRGANVideoProcessor(config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return False
    except Exception as e:
        print(f"❌ 初始化 RealESRGAN 处理器失败: {e}")
        traceback.print_exc()
        return False
    proc.quiet = getattr(args, "quiet_esrgan", True)  # 透传静默开关
    try:
        success = proc.process_video(actual_input, output_video)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断（Ctrl+C），当前分段将在下次运行时重新处理。")
        return False
    _elapsed_sk = time.time() - t0

    # [FIX] --skip-interpolate 模式下，process_video() 内部不会回写音频，
    # 需要在此处将已提取的音频合并到最终输出中。
    if success and audio_path:
        _rewrite_audio_and_cleanup(output_video, audio_path,
                                   input_video, config)

    if success:
        _run_postprocess_stage(output_video, input_video,
                               denoised_path, args, env_info)
        _print_completion(output_video, _elapsed_sk, env_info)
    else:
        _print_failure_hints(_elapsed_sk)
    # [P1-FIX-AUDIO-CLEAN] 流程结束回收音频临时文件（原实现此分支不清理）
    _cleanup_audio_temp(audio_path)
    _rpt = config.get("output", "report_path", default="")
    if _rpt:
        _write_final_report(
            report_path=_rpt,
            input_video=input_video,
            output_video=output_video,
            mode="skip_interpolate",
            elapsed=_elapsed_sk,
            success=success,
            env_info=env_info,
            args=args,
        )
    return success


def _run_interpolate_only(config: Config, actual_input: str, output_video: str,
                          audio_path: Optional[str], denoised_path: Optional[str],
                          input_video: str, args: argparse.Namespace,
                          env_info: dict, t0: float,
                          preview_ifr: bool, preview_ifr_intv: int) -> bool:
    """[P3.1-SPLIT] 单步模式——仅插帧（keep_audio finally 恢复在方法内闭合）。"""
    from ifrnet_processor_video_optimized import IFRNetProcessor  # noqa
    _print_stage(2, "IFRNet 视频插帧", "🎞️", quiet=args.quiet)
    # [FIX] 禁用 IFRNet 内部音频处理，统一由本层使用已提取的 audio_path 回写，
    # 避免重复提取音频或音频冲突。
    proc_keep_audio = config.get("models", "ifrnet", "keep_audio", default=True)
    config.set("models", "ifrnet", "keep_audio", value=False)
    try:
        try:
            proc = IFRNetProcessor(config)
            proc.preview          = preview_ifr
            proc.preview_interval = preview_ifr_intv
            proc.quiet            = getattr(args, "quiet_ifrnet", True)  # 透传静默开关
        except Exception as e:
            print(f"❌ 初始化 IFRNet 处理器失败: {e}")
            traceback.print_exc()
            return False
        # [P0-FIX-CONFIG-RESTORE] 原实现仅在正常返回/KeyboardInterrupt 两条路径
        # 恢复 keep_audio；其他异常会冲出本函数（批量模式下污染后续文件的配置）。
        # 统一由 finally 保证恢复。
        success = proc.process_video(actual_input, output_video)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断（Ctrl+C），当前分段将在下次运行时重新处理。")
        return False
    finally:
        config.set("models", "ifrnet", "keep_audio", value=proc_keep_audio)
    _elapsed_sk = time.time() - t0

    # [FIX] 统一使用顶层提取的 audio_path 回写音频（与 skip-interpolate 对齐）
    if success and audio_path:
        _rewrite_audio_and_cleanup(output_video, audio_path,
                                   input_video, config)

    if success:
        _run_postprocess_stage(output_video, input_video,
                               denoised_path, args, env_info)
        _print_completion(output_video, _elapsed_sk, env_info)
    else:
        _print_failure_hints(_elapsed_sk)
    # [P1-FIX-AUDIO-CLEAN] 流程结束回收音频临时文件（原实现此分支不清理）
    _cleanup_audio_temp(audio_path)
    _rpt = config.get("output", "report_path", default="")
    if _rpt:
        _write_final_report(
            report_path=_rpt,
            input_video=input_video,
            output_video=output_video,
            mode="skip_upscale",
            elapsed=_elapsed_sk,
            success=success,
            env_info=env_info,
            args=args,
        )
    return success


def _run_two_stage(config: Config, actual_input: str, video_name: str,
                   mode: str, args: argparse.Namespace, env_info: dict,
                   t0: float, preview_ifr: bool, preview_ifr_intv: int):
    """[P3.1-SPLIT] 双步模式：处理器创建 + 链式执行 + VRAM 对称释放。

    返回 (final_segs, ifrnet_proc, esrgan_proc)；任一步失败返回 None。
    """
    from ifrnet_processor_video_optimized       import IFRNetProcessor           # noqa
    from realesrgan_processor_video_optimized   import RealESRGANVideoProcessor  # noqa

    try:
        ifrnet_proc = IFRNetProcessor(config)
        ifrnet_proc.preview          = preview_ifr
        ifrnet_proc.preview_interval = preview_ifr_intv
        ifrnet_proc.quiet            = getattr(args, "quiet_ifrnet", True)  # 透传静默开关
        esrgan_proc = RealESRGANVideoProcessor(config)
        esrgan_proc.quiet            = getattr(args, "quiet_esrgan", True)  # 透传静默开关
    except Exception as e:
        print(f"❌ 初始化处理器失败: {e}")
        traceback.print_exc()
        return None

    final_segs = None

    if mode == "interpolate_then_upscale":
        # Step 1: IFRNet
        _print_stage(1, "Step 1/2 — IFRNet 插帧", "🎞️", quiet=args.quiet)
        try:
            step1_segs = ifrnet_proc.process_video_segments(actual_input)
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断（Ctrl+C），当前分段将在下次运行时重新处理。")
            return None
        if not step1_segs or getattr(ifrnet_proc, '_has_failure', False):
            print("❌ IFRNet 插帧未产生有效分段或部分分段失败，流程终止")
            _print_failure_hints(time.time() - t0)
            return None
        print(f"\n   ✅ Step 1 完成，产生 {len(step1_segs)} 个分段")

        # [VRAM-CLEANUP] 释放 IFRNet 持有的全部 GPU/pinned 资源，为 ESRGAN 腾出显存。
        # ifrnet_proc 本体保留：末尾 _delete_checkpoint/_cleanup_temp_files 不依赖 _video_processor。
        ifrnet_proc._cleanup_video_processor()
        print("   🧹 IFRNet GPU 资源已释放（模型/TRT/pinned 池/torch 缓存）")

        # Step 2: ESRGan
        _print_stage(2, "Step 2/2 — Real-ESRGAN 超分（优化版）", "🎨", quiet=args.quiet)
        try:
            final_segs = esrgan_proc.process_segments_directly(
                step1_segs, video_name)
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断（Ctrl+C），当前分段将在下次运行时重新处理。")
            return None

    elif mode == "upscale_then_interpolate":
        # Step 1: ESRGan
        _print_stage(1, "Step 1/2 — Real-ESRGAN 超分（优化版）", "🎨", quiet=args.quiet)
        try:
            step1_segs = esrgan_proc.process_video_segments(actual_input)
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断（Ctrl+C），当前分段将在下次运行时重新处理。")
            return None
        if not step1_segs or getattr(esrgan_proc, '_has_failure', False):
            print("❌ Real-ESRGAN 超分未产生有效分段或部分分段失败，流程终止")
            _print_failure_hints(time.time() - t0)
            return None
        print(f"\n   ✅ Step 1 完成，产生 {len(step1_segs)} 个分段")

        # [VRAM-CLEANUP] 对称释放 ESRGAN enhancer（SR 模型/TRT/GFPGAN 子进程），为 IFRNet 腾出显存。
        # esrgan_proc 本体保留：末尾 _delete_checkpoint/_cleanup_temp_files 不依赖 _enhancer。
        esrgan_proc.close_enhancer()
        print("   🧹 ESRGAN GPU 资源已释放（enhancer/TRT/GFPGAN 子进程/torch 缓存）")

        # [FIX-STAGE2-VRAM] 进入 Step 2 前的显存水位闸门。
        # 实测：本模式下插帧紧接超分同进程启动时，若阶段间显存未真正归还，
        # 推理速率会从 9.6 帧/s 掉到 2.2 帧/s（4.4 倍），并最终触发
        # 「推理线程 1822s 未退出」→ 提前 EOF → 缺帧 → exit=1。
        # 这里做：同步 → 二次回收 → 打印水位 → 可用显存不足阈值时告警并等待回落。
        try:
            import gc as _gc
            import time as _time
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.synchronize()
                _gc.collect()
                _torch.cuda.empty_cache()
                _torch.cuda.synchronize()
                _free, _total = _torch.cuda.mem_get_info()
                _free_gib = _free / 2 ** 30
                _total_gib = _total / 2 ** 30
                # 阈值：可用显存需 ≥ 总量的 55%（IFRNet 在 1536x1152 下
                # 需要 TRT engine + 模型 + 多批 pinned/显存缓冲的连续空间）。
                _need_gib = _total_gib * 0.55
                if _free_gib < _need_gib:
                    print(f"   ⚠️ [显存水位] 进入插帧前可用显存偏低: "
                          f"{_free_gib:.2f} GiB < 需求 {_need_gib:.2f} GiB"
                          f"（total {_total_gib:.2f} GiB）— 等待回落…", flush=True)
                    for _i in range(12):          # 最多等 60s
                        _time.sleep(5)
                        _gc.collect()
                        _torch.cuda.empty_cache()
                        _torch.cuda.synchronize()
                        _free, _total = _torch.cuda.mem_get_info()
                        _free_gib = _free / 2 ** 30
                        if _free_gib >= _need_gib:
                            break
                    print(f"   {'✅' if _free_gib >= _need_gib else '⚠️'} [显存水位] "
                          f"最终可用 {_free_gib:.2f} GiB / {_total_gib:.2f} GiB", flush=True)
                else:
                    print(f"   ✅ [显存水位] 进入插帧前可用 {_free_gib:.2f} GiB / "
                          f"{_total_gib:.2f} GiB（满足 ≥{_need_gib:.2f} GiB）", flush=True)
        except Exception as _e:
            print(f"   ⚠️ [显存水位] 检查异常（不阻断流程）: {_e}", flush=True)

        # Step 2: IFRNet
        _print_stage(2, "Step 2/2 — IFRNet 插帧", "🎞️", quiet=args.quiet)
        try:
            final_segs = ifrnet_proc.process_segments_directly(
                step1_segs, video_name)
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断（Ctrl+C），当前分段将在下次运行时重新处理。")
            return None

    else:
        print(f"❌ 未知模式: {mode}，可选值: "
              f"interpolate_then_upscale | upscale_then_interpolate")
        return None

    if not final_segs:
        print("❌ 第二步处理未产生有效分段，流程终止")
        _print_failure_hints(time.time() - t0)
        return None

    # [VRAM-CLEANUP] 第二阶段资源在合并/后处理（纯 CPU ffmpeg）前确定释放。
    # 两个清理方法均幂等：本阶段此前若已清理（或另一阶段已清理）重复调用无副作用。
    # 批量模式下避免上一文件的残留资源累积到下一文件。
    if mode == "interpolate_then_upscale":
        esrgan_proc.close_enhancer()
    else:
        ifrnet_proc._cleanup_video_processor()

    return final_segs, ifrnet_proc, esrgan_proc


def _merge_and_finalize(config: Config, final_segs, mode: str,
                        audio_path: Optional[str], input_video: str,
                        output_video: str, args: argparse.Namespace,
                        ifrnet_proc, esrgan_proc):
    """[P3.1-SPLIT] 合并最终分段 + FIX-C 归一化 + LA 音频修正 + COLOR-FIX + EXT-PROP。

    返回 (success, output_video, audio_path)：LA 修剪与容器改写会更新后两者。
    """
    print(f"\n🔗 合并 {len(final_segs)} 个最终分段 → {output_video}")
    output_config = config.get_section("output", {})
    # copy-by-default：三个输出参数均未在 CLI 中指定时，直接 stream copy
    _use_copy = config.get("output", "use_copy", default=True)
    if _use_copy:
        print("   ℹ️  未指定 --output-codec/质量/preset，最终合并使用 -c:v copy（不重新编码）")
        _merge_cfg = {**output_config, "codec": "copy"}
    else:
        _merge_cfg = output_config

    # ── [FIX-C] 分段 timescale 归一化（已下沉至 merge_videos_by_codec）──────
    # 归一化统一由 merge_videos_by_codec 在 copy 合并前执行，使"单阶段"路径
    # （各 processor 内部合并）与"两阶段"路径（此处最终合并）获得一致保护。
    # 旧实现只挂在 main 层，单阶段完全无保护；此处仅保留开关。
    _skip_ts_norm = bool(getattr(args, "skip_seg_normalize", False))

    # ── LA 音频同步修正：比较输出时长与预期时长，修剪音频开头 ──────────
    if audio_path and final_segs:
        input_dur = get_video_duration(input_video)
        if input_dur and input_dur > 0:
            # 累加所有最终分段时长
            output_dur = 0.0
            for seg in final_segs:
                d = get_video_duration(str(seg))
                if d:
                    output_dur += d

            # 预期时长 = 源"内容时长"（插帧改变帧率，不改变时长）
            # [FIX-DUR-CONTENT] 源若带非零 start_time（-c copy 剪辑残留），
            # format.duration 会把它一并算入，而各阶段输出时间轴都从 0 开始，
            # 直接用会凭空多出这段起始偏移（实测虚增 0.988s）。
            expected_dur = get_video_content_duration(input_video) or input_dur

            # [P1-FIX-LA-ACTUAL] 原实现以"SDK 模块可导入"推断"本次分段确实走了
            # SDK NVENC"——输出≥1080p 时后端自动切 constqp+LA=0，此时高估丢帧；
            # 反之 FFmpeg CLI 引入丢帧则漏估。现直接查询两阶段处理器的实际编码器：
            # encoder 实例内的 _la_depth 已是生效值（CRF=0 强制清零、高分辨率
            # 自动切 constqp 都反映在内），且仅 SDK Level1 路径会真正滞留帧。
            _total_la_depth = 0
            if not args.skip_interpolate:
                _ivp = getattr(ifrnet_proc, '_video_processor', None)
                _i_enc = getattr(_ivp, '_cached_nvenc_encoder', None)
                if (_i_enc is not None
                        and getattr(_ivp, '_diag_active_level', 0) == 1):
                    _total_la_depth += int(getattr(_i_enc, '_la_depth', 0) or 0)
            if not args.skip_upscale:
                _enh = getattr(esrgan_proc, '_enhancer', None) or {}
                _e_enc = _enh.get('_sdk_nvenc_encoder')
                if _e_enc is not None:
                    _total_la_depth += int(getattr(_e_enc, '_la_depth', 0) or 0)
            _la_enabled = _total_la_depth > 0
            # [P1-FIX-LA-FPS] 阈值换算改用真实输出帧率（原硬编码 60fps 对 24/30fps
            # 源的物理含义漂移）：输出 fps = 源 fps × 插帧倍数。
            _out_fps = 30.0
            try:
                _vi_la = VideoInfo(input_video)
                if _vi_la.fps and _vi_la.fps > 0:
                    _out_fps = float(_vi_la.fps)
            except Exception:
                pass
            if not args.skip_interpolate:
                _out_fps *= max(1, config.get("processing", "interpolation_factor",
                                              default=1))
            _la_threshold = max(0.3, _total_la_depth / max(_out_fps, 1.0) * 0.5)

            audio_offset = expected_dur - output_dur
            if audio_offset > _la_threshold and _la_enabled:
                print(f"\n🔊 检测到 LA 导致输出视频缺失 {audio_offset:.2f}秒")
                print(f"   预期时长 {expected_dur:.2f}s，实际 {output_dur:.2f}s")
                print(f"   正在修剪音频开头 {audio_offset:.2f}秒...")
                trimmed = _trim_audio_start(audio_path, audio_offset)
                if trimmed:
                    # [P1-FIX-AUDIO-CLEAN] 修剪成功后删除原始音频临时文件
                    try:
                        Path(audio_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                    audio_path = trimmed
                    print(f"   ✅ 音频已修剪: {Path(trimmed).name}")
                else:
                    print("   ⚠️ 音频修剪失败，将使用原始音频（可能音画不同步）")
            elif audio_offset > 0.1 and not _la_enabled:
                # 非 LA 原因导致的时长偏差（如编码器行为、时间基精度），记录但不修剪
                print(f"   ℹ️ 输出时长偏差 {audio_offset:.2f}s（非 LA 原因，不修剪音频）")
            elif audio_offset < -0.1:
                # 防御：输出反而更长（不应发生，但记录一下）
                print(f"   ℹ️ 输出视频比预期长 {abs(audio_offset):.2f}s，无需修剪音频")

    # [COLOR-FIX] 最终合并注入源视频色彩元数据（有值透传，无值回退 BT.709+Full Range）
    _merge_cfg = {
        **_merge_cfg,
        "extra_args": list(_merge_cfg.get("extra_args", []))
                      + build_color_args(input_video),
    }

    # [P0-FIX-EXT-PROP] 重编码触发容器改写时（如 out.mkv→out.mp4），后续所有
    # 逻辑必须使用实际输出路径，否则出现"任务成功但报告文件不存在"的假阴性链。
    _actual_out: list = []
    success = merge_videos_by_codec(
        final_segs, output_video,
        audio_path=audio_path,
        config=_merge_cfg,
        # [QUALITY-UNIFY] 显式请求输出编码/质量才重编码；否则 -c:v copy（无损、快）
        reencode=(not _use_copy),
        actual_output=_actual_out,
        normalize_timescale=not _skip_ts_norm,
        # [META-KEEP] 回写原片容器级元数据（tags / creation_time / 旋转 / 位深）
        source_video=input_video,
    )
    if success and _actual_out:
        output_video = _actual_out[0]
    # [P4-FIX-GATE] 分段通过不代表 concat/remux 没有引入新错误。
    if success and output_video and os.path.exists(output_video):
        # [P3-1] 最终输出验收严格度按「是否分段 + 分段是否已逐段解码级验收通过」决定：
        #   · 分段模式且全部分段已通过逐段解码级验收
        #       → 合并阶段只做 -c:v copy 不重编码，缺陷只可能出在容器层，
        #         故最终产物走「容器级校验（不解码）」，避免对同一视频再付
        #         一次全量解码代价（实测 4K 每次 7~15s，大文件更甚）。
        #   · 整体处理不分段（时长 <= segment_duration 的直接处理路径），
        #     或分段验收未全部通过/被 skip_validate 跳过
        #       → 最终输出是唯一的严格门槛，必须解码级验收（count_mode='decode'）。
        # 依据由各处理器在 _process_segments 末尾写入（_was_segmented /
        # _segments_decode_verified）。
        # 走到本函数的只有两个双步模式（单步模式由 _run_*_only 自行收尾），
        # 因此 ifrnet_proc 与 esrgan_proc 都实际参与了处理链 —— 要求**两者**
        # 的产出分段都通过逐段解码级验收，才允许最终产物降级为容器级校验。
        _participants = [p for p in (ifrnet_proc, esrgan_proc) if p is not None]
        _seg_verified = bool(_participants) and all(
            getattr(p, '_was_segmented', False)
            and getattr(p, '_segments_decode_verified', False)
            for p in _participants)
        if _seg_verified:
            print("🔎 最终输出验收: 分段已逐段解码级验收通过 → 走容器级校验（不解码）")
            dec_ok, dec_report = validate_decodable_video(
                output_video, skip_decode_check=True)
            if not dec_ok:
                print("❌ 最终合并输出容器级校验失败: "
                      f"packets={dec_report.get('packets')} "
                      f"reason={dec_report.get('reason')}")
                tail = str(dec_report.get('decode_stderr_tail', '')).strip()
                if tail:
                    print(f"   ↳ {tail[-500:]}")
                success = False
        else:
            print("🔎 最终输出验收: 未分段或分段验收未全覆盖 → 解码级验收（全解码）")
            # [PROBE-OPT] 最终交付门强制全解码计数（count_mode='decode'）：
            # 容器 nb_frames 对"包存在但解码失败/参考链断裂"完全盲区，中间分段
            # 可用 auto 提速，但最终合并产出必须保留 [P4-FIX-COUNT] 的严格语义。
            dec_ok, dec_report = validate_decodable_video(output_video,
                                                          count_mode="decode")
            if not dec_ok:
                print("❌ 最终合并输出解码级验收失败: "
                      f"decoded={dec_report.get('decoded_frames')} "
                      f"reason={dec_report.get('reason')}")
                tail = str(dec_report.get('decode_stderr_tail', '')).strip()
                if tail:
                    print(f"   ↳ {tail[-500:]}")
                success = False
    # [P5-FIX-PROVENANCE] 成功时输出 QA/代际侧车，禁止把增强输出当无损原片。
    if success and output_video and os.path.exists(output_video):
        qa_path = Path(str(output_video) + ".qa.json")
        generation = 1
        if qa_path.exists():
            try:
                old_qa = json.loads(qa_path.read_text(encoding="utf-8"))
                generation = int(old_qa.get("encoding_generation", 0)) + 1
            except (OSError, ValueError):
                pass
        qa_payload = {
            "encoding_generation": generation,
            "generated_by": "video-enhancement-comprehensive-fix-v1",
            "source": os.path.abspath(input_video),
            "mode": mode,
            "ifrnet_factor": int(config.get(
                "processing", "interpolation_factor", default=2)),
            "realesrgan_factor": int(config.get(
                "models", "realesrgan", "upscale_factor", default=4)),
            "codec_hint": config.get("models", "realesrgan", "codec", default=""),
            "rate_mode_ifrnet": config.get("models", "ifrnet", "rate_mode", default=""),
            "lookahead_depth_ifrnet": int(config.get(
                "models", "ifrnet", "lookahead_depth", default=0)),
            "fixes_applied": [
                "H2D_EVENT_SYNC", "HEVC_LA_SAFE_ROUTE", "DECODABLE_GATE",
                "EOS_OUTPUT_ORDER", "STRICT_EOS", "LOCKBITSTREAM_SIZE_CAP",
                "NAL_COMMON"
            ],
            "validated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }
        try:
            qa_path.write_text(json.dumps(qa_payload, ensure_ascii=False,
                                          indent=2), encoding="utf-8")
        except OSError as e:
            print(f"⚠️ QA sidecar 写入失败（不影响媒体文件）: {e}")
    return success, output_video, audio_path


# =============================================================================
# 批量模式                                                              [V2]
# =============================================================================

def _process_batch(
    config:           Config,
    input_dir:        str,
    output_dir:       str,
    mode:             str,
    skip_interpolate: bool,
    skip_upscale:     bool,
    args:             argparse.Namespace,
    env_info:         dict,
) -> int:
    """批量处理目录下所有视频文件，返回成功数量。"""
    input_files = sorted([
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in SUPPORTED_VIDEO_EXTS
    ])
    if not input_files:
        print(f"⚠️  输入目录 {input_dir} 中未找到视频文件")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    total    = len(input_files)
    ok_count = 0

    print(f"\n📦 批量模式: 共 {total} 个文件")
    print(f"   输入目录: {input_dir}")
    print(f"   输出目录: {output_dir}")
    print(f"   处理模式: {mode}\n")

    batch_t0 = time.time()

    for idx, src in enumerate(input_files):
        dst = Path(output_dir) / src.name
        # [P1-FIX-INPLACE-GUARD] 输入目录与输出目录相同时会就地覆盖源文件。
        if src.resolve() == dst.resolve():
            print(f"\n  ⏭️  [{idx + 1}/{total}] {src.name}: 输入输出为同一文件，跳过"
                  f"（请为 --output-dir 指定其他目录）")
            continue
        print(f"\n{'=' * 70}")
        print(f"  [{idx + 1}/{total}] {src.name}")
        print(f"{'=' * 70}")

        config.set("paths", "input_video", value=str(src))
        config.set("paths", "output_dir",  value=output_dir)

        # [P0-FIX-BATCH-ISOLATION] 单文件未捕获异常（磁盘满 OSError、cv2 崩溃等）
        # 原实现会冲出循环终止整批，与"继续下一个"的承诺矛盾。逐文件隔离：
        try:
            ok = _process_single(
                config, str(src), str(dst), mode,
                skip_interpolate, skip_upscale,
                args, env_info,
            )
        except KeyboardInterrupt:
            print("  ⏹️ 用户中断，停止批量处理")
            raise
        except Exception as e:
            import traceback as _tb
            print(f"  ❌ 异常 ({idx + 1}/{total}): {type(e).__name__}: {e}")
            _tb.print_exc()
            ok = False
        if ok:
            ok_count += 1
            print(f"  ✅ 完成 ({idx + 1}/{total})")
        else:
            print(f"  ❌ 失败 ({idx + 1}/{total})，继续下一个")

    batch_elapsed = time.time() - batch_t0
    print(f"\n📊 批量处理完成: 成功 {ok_count}/{total}，"
          f"总耗时 {_fmt_time(batch_elapsed)}")
    return ok_count


# =============================================================================
# CLI 参数定义                                              [V2 骨架 + V1 新增]
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(f"🎬 视频增强主流程（优化版）v{VERSION}"
                     f" —— IFRNet 插帧 + Real-ESRGAN 超分"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
底层架构：
  IFRNet     : src/processors/ifrnet_processor_video_optimized.py
               → external/ifrnet_video/main.py (IFRNetVideoProcessor, v6.4.5.1)
  Real-ESRGAN: src/processors/realesrgan_processor_video_optimized.py
               → external/realesrgan_video/main.py (main_optimized, v6.4)

ESRGan 模型选项 (--esrgan-model):
  realesr-general-x4v3          通用高质量（推荐，支持 --denoise-strength）
  RealESRGAN_x4plus             经典 4× 模型
  RealESRGAN_x2plus             经典 2× 模型
  realesr-animevideov3          动漫视频专用
  RealESRGANv2-animevideo-xsx2  动漫视频 2× 轻量版
  RealESRGAN_x4plus_anime_6B    动漫图像 4×

参数分组：
  基础参数   : -i / -o / -c / --batch-mode / --input-dir / --output-dir
  处理控制   : --mode / --interpolation-factor / --upscale-factor /
               --segment-duration / --skip-interpolate / --skip-upscale
  IFRNet参数 : --ifrnet-model* / --use-tensorrt-ifrnet / --no-fp16-ifrnet /
               --no-compile-ifrnet / --no-cuda-graph-ifrnet / --no-hwaccel-ifrnet /
               --batch-size-ifrnet / --max-batch-size-ifrnet / --crf-ifrnet /
               --codec-ifrnet / --report-ifrnet / --preview-ifrnet /
               --use-cuda-graph-ifrnet / --use-compile-ifrnet / --no-tensorrt-ifrnet
  ESRGan参数 : --esrgan-model / --use-tensorrt-esrgan / --no-compile-esrgan /
               --no-cuda-graph-esrgan / --no-fp16-esrgan / --no-hwaccel-esrgan /
               --batch-size-esrgan / --prefetch-factor-esrgan / --tile-size /
               --tile-pad / --pre-pad / --denoise-strength / --crf-esrgan /
               --codec-esrgan / --encode-preset-esrgan /
               --face-det-threshold / --no-adaptive-batch-esrgan / --gfpgan-trt /
               --report-esrgan / --preview-esrgan / --preview-interval-esrgan
  人脸增强   : --face-enhance / --gfpgan-model / --gfpgan-weight / --gfpgan-batch-size
  合并输出   : --output-codec / --output-crf / --output-cq /
               --output-crf-ref / --output-cq-ref / --output-preset
  归一化分段 : --split-codec / --split-crf-ref / --split-cq-ref / --split-preset
  TRT 缓存   : --trt-cache-dir（IFRNet 与 ESRGan 共享同一目录）
  
使用示例：
  # 基本：先插帧 → 再超分
  python main_video_optimized.py -i input.mp4 -o output.mp4

  # 仅超分 + 人脸增强
  python main_video_optimized.py -i face.mp4 -o face_4x.mp4 \\
         --skip-interpolate --face-enhance --face-det-threshold 0.7

  # 去噪 + 插帧 + 超分
  python main_video_optimized.py -i noisy.mp4 -o clean.mp4 --denoise

  # TensorRT 全加速
  python main_video_optimized.py -i input.mp4 -o output.mp4 \\
         --use-tensorrt-ifrnet --use-tensorrt-esrgan

  # 低显存设备
  python main_video_optimized.py -i input.mp4 -o output.mp4 \\
         --tile-size 512 --batch-size-esrgan 2 --no-cuda-graph-esrgan

  # Dry-run（仅打印配置）
  python main_video_optimized.py -i input.mp4 -o output.mp4 --dry-run

  # 批量处理
  python main_video_optimized.py --batch-mode \\
         --input-dir /data/raw/ --output-dir /data/enhanced/
""",
    )

    # ── 基础参数 ─────────────────────────────────────────────────────────────
    g = parser.add_argument_group("基础参数")
    g.add_argument("--config", "-c", default=_DEFAULT_CFG,
                   help=f"配置文件路径（默认: {_DEFAULT_CFG}）")
    g.add_argument("--input", "-i",
                   help="输入视频路径（单文件模式必填）")
    g.add_argument("--output", "-o",
                   help="输出视频路径（含文件名，单文件模式必填）")

    # ── 批量模式 ─────────────────────────────────────────────────────────────
    g = parser.add_argument_group("批量模式")
    g.add_argument("--batch-mode", action="store_true",
                   help="启用批量模式（扫描 --input-dir 目录下所有视频）")
    g.add_argument("--input-dir",  metavar="DIR",
                   help="批量输入视频目录（batch-mode 必填）")
    g.add_argument("--output-dir", metavar="DIR",
                   help="批量输出目录（batch-mode 必填）")

    # ── 全局处理控制 ─────────────────────────────────────────────────────────
    g = parser.add_argument_group("处理控制")
    g.add_argument("--mode", "-m",
                   choices=["interpolate_then_upscale", "upscale_then_interpolate"],
                   help="处理顺序（覆盖配置，默认 interpolate_then_upscale）")
    g.add_argument("--interpolation-factor", type=int, choices=[2, 4, 8, 16],
                   help="插帧倍数（覆盖配置，默认 2）")
    g.add_argument("--upscale-factor", type=int, choices=[2, 4],
                   help="超分倍数（覆盖配置，默认 2）")
    g.add_argument("--segment-duration", type=int, metavar="SEC",
                   help="分段时长（秒，覆盖配置，默认 30）")
    g.add_argument("--skip-interpolate", action="store_true",
                   help="跳过 IFRNet 插帧，仅执行超分")
    g.add_argument("--skip-upscale", action="store_true",
                   help="跳过 Real-ESRGAN 超分，仅执行插帧")
    g.add_argument("--skip-validate", action="store_true",
               default=False, help="跳过解码级验收（加速流程，不推荐生产使用）")
    g.add_argument("--validate-workers", type=int, metavar="N",
               default=None, help="验收阶段并行度（默认自动：基于 CPU/RAM/GPU 计算）")
    g.add_argument("--validate-mode", choices=["thread", "process"],
               default="thread", help="验收并行模式（默认 thread）")
    g.add_argument("--validate-gpu-workers", type=int, metavar="N",
               default=None, help="验收 GPU 并发上限（默认自动按 GPU 型号，0=禁用 GPU）")
    g.add_argument("--auto-parallel", action="store_true",
               default=True, help="启用自动并行度计算（默认开启）")
    g.add_argument("--max-parallel-workers", type=int, default=0, metavar="N",
               help="全局并行度上限（0=不限制，仅受资源约束）")
    g.add_argument("--normalize-source", action="store_true",
                   help="处理前把源时间轴归一化为均匀 CFR（按帧号重编号 pts，需一次"
                        "重编码）。用于修复 VFR 源（时间戳空洞）与首帧非零起始偏移"
                        "导致的段级帧数验收失败/时长偏差")

    # ── 去噪参数（可选前处理阶段）──────────────────────────────────   [V1]
    g = parser.add_argument_group("去噪参数（可选前处理阶段）")
    g.add_argument("--denoise", action="store_true",
                   help="启用预去噪阶段（在插帧/超分之前执行）")
    g.add_argument("--denoise-model", type=str, default="nafnet",
                   choices=["nafnet", "dncnn", "scunet"],
                   help="去噪模型（默认 nafnet）")
    g.add_argument("--denoise-strength-pre", type=float, default=0.5,
                   help="预去噪强度 [0.0-1.0]（默认 0.5）")

    # ── IFRNet 参数（与 v6 完全一致）─────────────────────────────────────────
    g = parser.add_argument_group("IFRNet 参数")
    g.add_argument("--ifrnet-model",
                   choices=["IFRNet_Vimeo90K", "IFRNet_S_Vimeo90K", "IFRNet_L_Vimeo90K"],
                   help="IFRNet 模型名称（覆盖配置）")
    g.add_argument("--ifrnet-model-path", metavar="PATH",
                   help="IFRNet .pth 权重绝对路径（优先级高于 --ifrnet-model）")
    g.add_argument("--batch-size-ifrnet", type=int, metavar="N",
                   help="IFRNet 批处理大小（覆盖配置，默认 24）")
    g.add_argument("--max-batch-size-ifrnet", type=int, metavar="N",
                   help="IFRNet 批大小上限（OOM 天花板，覆盖配置，默认 8）")
    g.add_argument("--no-fp16-ifrnet",       action="store_true",
                   help="禁用 IFRNet FP16（默认开启）")
    g.add_argument("--no-compile-ifrnet",    action="store_true",
                   help="禁用 IFRNet torch.compile（短视频可禁用跳过预热）")
    g.add_argument("--no-cuda-graph-ifrnet", action="store_true",
                   help="禁用 IFRNet CUDA Graph（compile 激活时已接管，可安全禁用）")
    g.add_argument("--use-tensorrt-ifrnet",  action="store_true",
                   help="IFRNet TensorRT 加速（JSON 配置默认开启；显式添加确保开启；--no-tensorrt-ifrnet 可禁用）")
    g.add_argument("--no-hwaccel-ifrnet",    action="store_true",
                   help="禁用 IFRNet NVDEC 硬件解码")
    g.add_argument("--no-audio-ifrnet",      action="store_true",
                   help="IFRNet 分段处理时不保留音轨（主流程会统一处理音频）")
    g.add_argument("--crf-ifrnet",  type=int, metavar="N",
                   help="IFRNet 分段输出 CRF 字面量（软编原样下发）。量程随编码器而定"
                        "（libx264/libx265 0~51、libvpx-vp9 0~63），超限报错退出")
    g.add_argument("--cq-ifrnet",   type=int, metavar="N",
                   help="IFRNet 分段输出 CQ 字面量（硬编原样下发）。量程随编码器而定"
                        "（NVENC 0~51、QSV 1~51、VideoToolbox 1~100），超限报错退出")
    g.add_argument("--crf-ifrnet-ref", type=int, metavar="N",
                   help="IFRNet 分段输出质量：以 libx264 CRF 为统一基准（0~51，默认 21），"
                        "按等效表换算到实际编码器。"
                        "例：--codec-ifrnet hevc_nvenc --crf-ifrnet-ref 21 → -cq:v 28。"
                        "与 --crf-ifrnet / --cq-ifrnet 互斥")
    g.add_argument("--cq-ifrnet-ref",  type=int, metavar="N",
                   help="IFRNet 分段输出质量：以 h264_nvenc CQ 为统一基准（0~51），"
                        "按等效表换算到实际编码器。"
                        "例：--codec-ifrnet hevc_nvenc --cq-ifrnet-ref 26 → -cq:v 28。"
                        "与 --crf-ifrnet / --cq-ifrnet 互斥")
    g.add_argument("--codec-ifrnet", metavar="CODEC",
                   help="IFRNet 分段输出编码器（默认 libx264，有 NVENC 时自动升级）")
    g.add_argument("--encode-preset-ifrnet", metavar="PRESET",
               choices=["ultrafast", "superfast", "veryfast", "faster", "fast",
                        "medium", "slow", "slower", "veryslow"],
               help="IFRNet 编码预设（libx264/libx265 名称，NVENC 自动映射为 p1~p7）")
    g.add_argument("--rate-mode-ifrnet", metavar="MODE",
               choices=["constqp", "vbr_hq", "qvbr"],
               help="IFRNet NVENC 码率控制模式（默认 vbr_hq）")
    g.add_argument("--lookahead-depth-ifrnet", type=int, metavar="N",
               choices=[0, 8, 16, 32],
               help="IFRNet NVENC 前向帧预看深度（默认 8）")
    g.add_argument("--report-ifrnet", metavar="PATH",
                   help="IFRNet JSON 性能报告输出路径")
    g.add_argument("--preview-ifrnet", action="store_true",
                   help="IFRNet 处理时弹出帧预览窗口（调试用）")
    g.add_argument("--preview-interval-ifrnet", type=int, default=30, metavar="N",
                   help="IFRNet 帧预览间隔（每隔 N 帧弹出一次，默认 30）")
    # ── 高优先级覆盖开关（覆盖 config / --no-* 默认值）────────────────────────
    g.add_argument("--use-cuda-graph-ifrnet", dest="use_cuda_graph_force_ifrnet",
                   action="store_true", default=False,
                   help="[覆盖] 强制启用 IFRNet CUDA Graph，覆盖 --no-cuda-graph-ifrnet / config。"
                        "与 torch.compile 互斥；如需确保生效请同时指定 --no-compile-ifrnet。")
    g.add_argument("--use-compile-ifrnet", dest="use_compile_force_ifrnet",
                   action="store_true", default=False,
                   help="[覆盖] 强制启用 IFRNet torch.compile，覆盖 --no-compile-ifrnet / config。"
                        "与 --use-tensorrt-ifrnet 互斥。")
    g.add_argument("--no-tensorrt-ifrnet", dest="no_tensorrt_ifrnet",
                   action="store_true", default=False,
                   help="[覆盖] 强制禁用 IFRNet TensorRT，覆盖 --use-tensorrt-ifrnet / config。")

    # ── Real-ESRGAN 参数（已更新，对齐 realesrgan_video/main.py）──────────
    g = parser.add_argument_group("Real-ESRGAN 参数（优化版）")
    g.add_argument("--esrgan-model", metavar="MODEL_NAME",
                   help="ESRGan 模型名称（覆盖配置；不存在时自动下载）\n"
                        "可选: realesr-general-x4v3 | RealESRGAN_x4plus | "
                        "RealESRGAN_x2plus | realesr-animevideov3 | "
                        "RealESRGANv2-animevideo-xsx2")
    g.add_argument("--denoise-strength", type=float, metavar="F",
                   help="SR 模型降噪强度 0~1"
                        "（仅 realesr-general-x4v3）")
    g.add_argument("--batch-size-esrgan", type=int, metavar="N",
                   help="ESRGan SR 批处理大小（默认 24）")
    g.add_argument("--prefetch-factor-esrgan", type=int, metavar="N",
                   help="ESRGan 读帧预取深度（默认 48）")
    g.add_argument("--tile-size", type=int, metavar="N",
                   help="tile 切块大小（0=不切块；VRAM 不足时设 512）")
    g.add_argument("--tile-pad", type=int, metavar="N",
                   help="tile 边缘填充（默认 10）")
    g.add_argument("--pre-pad", type=int, metavar="N",
                   help="预处理填充（默认 0）")
    g.add_argument("--no-fp16-esrgan", action="store_true",
                   help="ESRGan 禁用 FP16（默认开启）")
    g.add_argument("--no-compile-esrgan", action="store_true",
                   help="禁用 ESRGan torch.compile（默认开启；短视频或调试时可禁用）")
    g.add_argument("--no-cuda-graph-esrgan", action="store_true",
                   help="禁用 ESRGan CUDA Graph（默认开启；compile/TRT 激活时自动禁用）")
    g.add_argument("--use-tensorrt-esrgan", action="store_true",
                   help="ESRGan TensorRT 加速（JSON 配置默认开启；显式添加确保开启；--no-tensorrt-esrgan 可禁用）")
    g.add_argument("--no-hwaccel-esrgan", action="store_true",
                   help="禁用 ESRGan NVDEC 硬件解码")
    g.add_argument("--crf-esrgan", type=int, metavar="N",
                   help="ESRGan 分段输出 CRF 字面量（软编原样下发）。量程随编码器而定"
                        "（libx264/libx265 0~51、libvpx-vp9 0~63），超限报错退出")
    g.add_argument("--cq-esrgan",   type=int, metavar="N",
                   help="ESRGan 分段输出 CQ 字面量（硬编原样下发）。量程随编码器而定"
                        "（NVENC 0~51、QSV 1~51、VideoToolbox 1~100），超限报错退出")
    g.add_argument("--crf-esrgan-ref", type=int, metavar="N",
                   help="ESRGan 分段输出质量：以 libx264 CRF 为统一基准（0~51，默认 21），"
                        "按等效表换算到实际编码器。"
                        "例：--codec-esrgan hevc_nvenc --crf-esrgan-ref 21 → -cq:v 28。"
                        "与 --crf-esrgan / --cq-esrgan 互斥")
    g.add_argument("--cq-esrgan-ref",  type=int, metavar="N",
                   help="ESRGan 分段输出质量：以 h264_nvenc CQ 为统一基准（0~51），"
                        "按等效表换算到实际编码器。"
                        "例：--codec-esrgan hevc_nvenc --cq-esrgan-ref 26 → -cq:v 28。"
                        "与 --crf-esrgan / --cq-esrgan 互斥")
    g.add_argument("--codec-esrgan", metavar="CODEC",
                   help="ESRGan 分段输出编码器（默认 libx264，有 NVENC 时自动升级；可选 libx265/h264_nvenc）")
    g.add_argument("--encode-preset-esrgan", metavar="PRESET",
                   choices=["ultrafast", "superfast", "veryfast",
                            "faster", "fast", "medium",
                            "slow", "slower", "veryslow"],
                   help="ESRGan libx264/libx265 编码预设（默认 medium，NVENC 自动映射为 p1~p7）")
    g.add_argument("--rate-mode-esrgan", metavar="MODE",
                   choices=["constqp", "vbr_hq", "qvbr"],
                   help="ESRGan NVENC 码率控制模式（默认 vbr_hq）")
    g.add_argument("--lookahead-depth-esrgan", type=int, metavar="N",
                   choices=[0, 8, 16, 32],
                   help="ESRGan NVENC 前向帧预看深度（默认 8）")
    g.add_argument("--ffmpeg-bin", type=str,
                   help="ffmpeg 可执行文件路径（默认 ffmpeg）")
    # ── 高优先级覆盖开关（强制启用，覆盖 --no-* / config 中的禁用设置）──────────
    g.add_argument("--no-tensorrt-esrgan", dest="no_tensorrt_esrgan",
                   action="store_true", default=False,
                   help="[覆盖] 强制禁用 ESRGan TensorRT，覆盖 --use-tensorrt-esrgan / config。"
                        "适用于 config 中 use_tensorrt=true 但本次不希望启用 TRT 的场景。")
    g.add_argument("--use-compile-esrgan", dest="use_compile_force_esrgan",
                   action="store_true", default=False,
                   help="[覆盖] 强制启用 ESRGan torch.compile，覆盖 --no-compile-esrgan / config。"
                        "与 --use-tensorrt-esrgan 互斥（TRT 优先）。")
    g.add_argument("--use-cuda-graph-esrgan", dest="use_cuda_graph_force_esrgan",
                   action="store_true", default=False,
                   help="[覆盖] 强制启用 ESRGan CUDA Graph，覆盖 --no-cuda-graph-esrgan / config。"
                        "与 compile/TRT 互斥（compile/TRT 优先）。"
                        "如需确保生效，请同时指定 --no-compile-esrgan --no-tensorrt。")
    # ── face_enhance 参数（已更新，新增置信度过滤 + 自适应批处理 + GFPGAN TRT）─────────────
    g = parser.add_argument_group("face_enhance 参数（Real-ESRGAN 优化版）")
    _fe = g.add_mutually_exclusive_group()
    _fe.add_argument("--face-enhance", dest="face_enhance",
                     action="store_true", default=None,
                     help="开启人脸增强（GFPGAN）")
    _fe.add_argument("--no-face-enhance", dest="face_enhance",
                     action="store_false",
                     help="关闭人脸增强（覆盖配置）")
    g.add_argument("--gfpgan-model",
                   choices=["1.3", "1.4", "RestoreFormer"],
                   help="GFPGAN 版本（默认 1.4）")
    g.add_argument("--gfpgan-weight", type=float, metavar="F",
                   help="GFPGAN 融合权重 0.0~1.0（0=不增强，1=完全替换，默认 0.5）")
    g.add_argument("--gfpgan-batch-size", type=int, metavar="N",
                   help="单次 GFPGAN 前向最多处理的人脸数（OOM 保护，默认 8）")
    g.add_argument("--face-det-threshold", type=float, metavar="F",
                   help="人脸检测置信度阈值 [0.0-1.0]（默认 0.5）。"
                        "0.5=保留多数人脸，0.7=过滤模糊远景，0.9=仅保留清晰人脸")
    g.add_argument("--no-adaptive-batch-esrgan", action="store_true",
                   help="禁用基于人脸密度的自适应批处理（默认开启）")
    g.add_argument("--gfpgan-trt", action="store_true",
                   help="GFPGAN TensorRT 子进程加速（启用时自动禁用 FP16 改用 FP32）")

    # ── 新增：预览与报告参数 ─────────────────────────────────────────────────
    g = parser.add_argument_group("Real-ESRGAN 预览与报告")
    g.add_argument("--report-esrgan", metavar="PATH",
                   help="ESRGan JSON 性能报告输出路径")
    g.add_argument("--preview-esrgan", action="store_true",
                   help="启用 ESRGan 实时预览窗口（显示最终输出，按 q 退出）")
    g.add_argument("--preview-interval-esrgan", type=int, default=30, metavar="N",
                   help="ESRGan 预览帧间隔（每多少帧刷新一次，默认 30）")

    # ── 最终合并输出参数（环节③）────────────────────────────────────────────
    g = parser.add_argument_group("最终合并输出参数")
    g.add_argument("--output-codec",  metavar="CODEC",
                   help="最终合并编码器（显式指定即触发重编码；'copy' 表示强制直接复制）")
    g.add_argument("--output-crf",    type=int, metavar="N",
                   help="最终合并 CRF 字面量（软编原样下发，量程随编码器）")
    g.add_argument("--output-cq",     type=int, metavar="N",
                   help="最终合并 CQ 字面量（硬编原样下发，如 h264_nvenc 0~51）")
    g.add_argument("--output-crf-ref", type=int, metavar="N",
                   help="最终合并质量：libx264 CRF 基准 0~51（默认 21），按等效表换算")
    g.add_argument("--output-cq-ref",  type=int, metavar="N",
                   help="最终合并质量：h264_nvenc CQ 基准 0~51，按等效表换算")
    g.add_argument("--output-preset", metavar="PRESET",
                   help="最终合并编码预设（如 medium / slow，覆盖配置）")

    # ── 归一化 / 分段参数（环节①）──────────────────────────────────────────
    g = parser.add_argument_group("归一化 / 分段参数（环节①）")
    g.add_argument("--split-codec",   metavar="CODEC",
                   help="源时间轴归一化重编码编码器（默认 libx264）")
    g.add_argument("--split-crf-ref", type=int, metavar="N",
                   help="归一化质量：libx264 CRF 基准 0~51（默认 21），按等效表换算")
    g.add_argument("--split-cq-ref",  type=int, metavar="N",
                   help="归一化质量：h264_nvenc CQ 基准 0~51，按等效表换算")
    g.add_argument("--split-preset",  metavar="PRESET",
                   help="归一化重编码预设（默认 veryfast）")

    # ── TRT 缓存目录（全局）─────────────────────────────────────────────────
    g = parser.add_argument_group("TRT Engine 缓存（IFRNet / ESRGan 共用）")
    g.add_argument("--trt-cache-dir", metavar="DIR",
                   help="TRT Engine 缓存目录（覆盖配置 paths.trt_cache_dir；"
                        "未指定时从配置读取；配置为空时自动使用 base_dir/.trt_cache）")

    # ── 杂项 ──────────────────────────────────────────────────────   [V1+V2]
    g = parser.add_argument_group("杂项")
    g.add_argument("--auto-cleanup", action="store_true",
                   help="全流程结束后自动删除所有临时分段文件")
    g.add_argument("--no-auto-cleanup", action="store_true",
                   help="保留所有临时分段文件（覆盖配置文件中的 auto_cleanup_temp）")
    g.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False,
                   help="静默主流程日志（音频提取、阶段标题等）；--no-quiet 开启详细日志")
    g.add_argument("--quiet-ifrnet", action=argparse.BooleanOptionalAction, default=True,
                   help="静默 IFRNet 底层冗余输出（底层 --quiet 参数）；--no-quiet-ifrnet 开启详细日志")
    g.add_argument("--quiet-esrgan", action=argparse.BooleanOptionalAction, default=True,
                   help="静默 ESRGAN 底层冗余输出（底层 --quiet 参数）；--no-quiet-esrgan 开启详细日志")
    g.add_argument("--keep-intermediate", action="store_true",
                   help="保留去噪等中间文件（调试用）")
    g.add_argument("--skip-seg-normalize", action="store_true",
                   help="跳过合并前的分段 timescale 归一化（FIX-C：统一 MP4 "
                        "时间基为 90kHz 的零损耗 remux）；调试用，或已确认各分段"
                        "时间基完全一致时可跳过以节省 I/O")
    g.add_argument("--dry-run", action="store_true",
                   help="仅打印配置和环境信息，不实际处理")
    g.add_argument("--report", metavar="PATH",
                   help="流水线完成后将整体运行摘要写入指定 JSON 文件 "
                        "（含耗时 / 大小 / GPU峰值显存 / 参数快照；成功或失败均写出）")
    g.add_argument("--version", "-V", action="version",
                   version=f"%(prog)s {VERSION}")

    return parser


# =============================================================================
# main()                                                    [V1 结构 + V2 逻辑]
# =============================================================================

def main() -> int:
    """
    主入口函数。

    Returns:
        退出码  0=成功  1=运行时失败  2=参数/输入错误
    """
    parser = _build_parser()
    args   = parser.parse_args()

    # ── 启动横幅 ──────────────────────────────────────────────────────────────
    _print_banner()

    # ── 阶段 0: 输入验证与环境检查 ────────────────────────────────────────────
    _print_stage(0, "输入验证与环境检查", "🔍", quiet=args.quiet)

    # 环境检查（不依赖配置文件）
    env_info = _check_environment()
    _print_environment(env_info)

    if not env_info["ffmpeg_available"]:
        print("\n❌ FFmpeg 未找到，无法进行视频处理。")
        print("   请安装 FFmpeg: https://ffmpeg.org/download.html")
        return 1
    if not env_info["cuda_available"]:
        print("\n⚠️  CUDA 不可用，将使用 CPU 进行推理，速度极慢。")

    # ── 加载配置（含容错回退）─────────────────────────────────────────────────
    print(f"\n⚙️  加载配置: {args.config}")
    try:
        config = Config(args.config)
    except FileNotFoundError:
        print(f"⚠️  配置文件未找到: {args.config}，尝试使用内置默认值")
        try:
            config = Config(None)
        except Exception as e:
            print(f"❌ 无法初始化配置: {e}")
            return 1
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        return 1

    # ── 命令行参数覆盖 ────────────────────────────────────────────────────────
    _apply_cli_overrides(config, args)
    # [P1-FIX-VALIDATE] CLI 数值参数无 argparse 范围约束，覆盖会绕过加载期校验，
    # 非法值可直达 ffmpeg/NVENC（故障点远离出错原因）。覆盖后重验关键范围。
    if not _validate_effective_config(config, args):
        return 2

    # [P2.5] 初始化统一日志（控制台兼容 + 可选 UTF-8 文件落盘）
    global _LOG
    try:
        from logger import init_logging, get_logger, log_file_path
        _log_dir = config.get("paths", "log_dir", default="")
        if not _log_dir:
            _base = config.get("paths", "base_dir", default="") or os.getcwd()
            _log_dir = os.path.join(_base, "logs")
        init_logging(
            level=config.get("logging", "level", default="INFO"),
            log_dir=_log_dir,
            log_to_file=config.get("logging", "log_to_file", default=True),
            log_to_console=config.get("logging", "log_to_console", default=True),
        )
        _LOG = get_logger("main")
        if log_file_path():
            print(f"📝 运行日志: {log_file_path()}")
        _LOG.info("[P5-FIX-RUN] 视频增强运行开始：日志基础设施已启用")
    except Exception as e:
        print(f"⚠️  日志初始化失败（不影响主流程）: {e}")

    # ── 确定处理模式 ──────────────────────────────────────────────────────────
    mode = config.get("processing", "mode",
                      default="interpolate_then_upscale")

    # ── 解析并验证路径（配置已就绪，可回退至 config 中的路径）────────────────
    is_batch = (args.batch_mode
                or config.get("processing", "batch_mode", default=False))

    if is_batch:
        input_dir = (args.input_dir
                     or config.get("paths", "input_dir", default=""))
        output_dir = (args.output_dir
                      or config.get("paths", "output_dir", default=""))
        if not input_dir:
            print("❌ 批量模式需指定 --input-dir 或配置 paths.input_dir")
            return 2
        if not output_dir:
            print("❌ 批量模式需指定 --output-dir 或配置 paths.output_dir")
            return 2
        if not Path(input_dir).is_dir():
            print(f"❌ 输入目录不存在: {input_dir}")
            return 2
    else:
        input_video = (args.input
                       or config.get("paths", "input_video", default="")).strip()
        output_video = (args.output or "").strip()
        if not input_video:
            print("❌ 单文件模式需指定 --input / -i"
                  "（或配置 paths.input_video）")
            return 2
        if not output_video:
            print("❌ 单文件模式需指定 --output / -o（含文件名）")
            return 2
        if not _validate_input(input_video):
            return 2
        if not _ensure_output_dir(output_video):
            return 2

    # [P2-FIX-MODE-AUTO] 单文件模式：提前进行自动模式选择（用于配置摘要显示）
    if not is_batch and input_video:
        mode = _select_optimal_mode(config, input_video, mode, quiet=args.quiet)

    # ── 打印完整配置摘要 ──────────────────────────────────────────────────────
    _print_startup_info(config, args, mode)

    # ── Dry-run 模式 ──────────────────────────────────────────────────────────
    if getattr(args, "dry_run", False):
        print("🏁 [Dry-run] 仅显示配置和环境信息，不实际处理。")
        return 0

    # ── 记录流水线开始时间 ────────────────────────────────────────────────────
    pipeline_t0 = time.time()

    # ── 分支：批量 / 单文件 ───────────────────────────────────────────────────
    if is_batch:
        n = _process_batch(
            config, input_dir, output_dir, mode,
            args.skip_interpolate, args.skip_upscale,
            args, env_info,
        )
        elapsed = time.time() - pipeline_t0
        print(f"\n⏱️  总流水线耗时: {_fmt_time(elapsed)}")
        return 0 if n > 0 else 1

    else:
        # 将路径写入 config（供处理器 get_temp_dir 等使用）
        config.set("paths", "input_video", value=input_video)
        config.set("paths", "output_dir",
                   value=str(Path(output_video).parent))
        config.set("processing", "batch_mode", value=False)

        ok = _process_single(
            config, input_video, output_video, mode,
            args.skip_interpolate, args.skip_upscale,
            args, env_info,
        )
        return 0 if ok else 1


# =============================================================================
# 入口                                                                  [V1]
# =============================================================================

if __name__ == "__main__":
    try:
        _exit_code = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断（Ctrl+C），当前分段将在下次运行时重新处理。")
        _exit_code = 130
    except Exception as _exc:
        print(f"\n❌ 未捕获的异常: {_exc}")
        traceback.print_exc()
        _exit_code = 1

    sys.exit(_exit_code)
