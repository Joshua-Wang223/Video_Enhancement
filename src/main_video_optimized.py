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
  IFRNet     : src/processors/ifrnet_processor_v6_single.py
  Real-ESRGAN: src/processors/realesrgan_processor_video_optimized.py
               → external/realesrgan_video_ds/main.py (main_optimized, v6.4)

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
    --preview-esrgan / --preview-interval-esrgan / --x264-preset-esrgan
  共用:
    --trt-cache-dir（IFRNet 与 ESRGan 共享同一 TRT Engine 缓存目录）

【优化版相对 v6 的 ESRGan 侧变更】
  · 底层脚本更换为 realesrgan_video_ds/main.py（深度模块化架构 v6.4）
  · 新增 CLI 参数：
      --face-det-threshold         人脸检测置信度阈值
      --no-adaptive-batch-esrgan   禁用自适应批处理
      --gfpgan-trt                 GFPGAN TensorRT 子进程加速
      --x264-preset-esrgan         libx264/libx265 编码预设
      --report-esrgan              JSON 性能报告输出路径
      --preview-esrgan             启用实时预览窗口
      --preview-interval-esrgan    预览帧间隔
  · 默认值调整：batch_size 12→6, prefetch 24→48, gfpgan_batch 12→8

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
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

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
from video_utils import (                            # noqa: E402
    format_time,
    get_video_duration,
    merge_videos_by_codec,
    smart_extract_audio,
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

VERSION = "2.0.0"  # 与 realesrgan_video_ds v6.4 架构对齐

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


# =============================================================================
# UI 辅助函数                                                          [V1]
# =============================================================================

def _print_banner():
    """打印启动横幅。"""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║    🎬  视频增强流水线 (优化版) v{:<18s} ║".format(VERSION))
    print("║    IFRNet + Real-ESRGAN  ·  realesrgan_video_ds v6.4       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


def _print_stage(stage_num: int, title: str, emoji: str = "🔷"):
    """打印阶段标题。"""
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
    print(f"     编码器     : {ifr('codec', 'libx264')}"
          f" | CRF: {ifr('crf', 23)}")

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
    print(f"     编码器     : {esr('codec', 'libx264')}"
          f" | CRF: {esr('crf', 23)}"
          f" | preset: {esr('x264_preset', 'medium')}")

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
    _oc = config.get("output", "codec",  default="")
    _ocrf = config.get("output", "crf",  default="")
    _op = config.get("output", "preset", default="")
    if _oc or _ocrf or _op:
        print(f"  最终合并输出  : codec={_oc or '默认'}"
              f" | CRF={_ocrf or '默认'}"
              f" | preset={_op or '默认'}")

    # 预览与报告
    _preview = esr("preview", False)
    _report = esr("report_json", "")
    if _preview:
        print(f"  实时预览      : 启用 (间隔 {esr('preview_interval', 30)} 帧)")
    if _report:
        print(f"  性能报告      : {_report}")

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

    _print_stage(1, "视频预去噪", "🧹")

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
    _print_stage(3, "后处理与验证", "🔍")

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
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                   🎉  处理完成！                            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
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

def _apply_cli_overrides(config: Config, args: argparse.Namespace) -> None:
    """将命令行参数写入 config 对象，供两个处理器读取。"""

    # ── 全局处理参数 ─────────────────────────────────────────────────────────
    if args.mode:
        config.set("processing", "mode",                 value=args.mode)
    if args.interpolation_factor:
        config.set("processing", "interpolation_factor", value=args.interpolation_factor)
    if args.upscale_factor:
        config.set("processing", "upscale_factor",       value=args.upscale_factor)
    if args.segment_duration:
        config.set("processing", "segment_duration",     value=args.segment_duration)
    if args.auto_cleanup:
        config.set("processing", "auto_cleanup_temp",    value=True)

    # ── IFRNet 模型参数（与 v6 完全一致）─────────────────────────────────────
    if args.ifrnet_model_path:
        config.set("models", "ifrnet", "model_path", value=args.ifrnet_model_path)
        config.set("models", "ifrnet", "model_name", value="")
    elif args.ifrnet_model:
        config.set("models", "ifrnet", "model_name", value=args.ifrnet_model)
        config.set("models", "ifrnet", "model_path", value="")

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
    if args.batch_size_ifrnet:
        config.set("models", "ifrnet", "batch_size",     value=args.batch_size_ifrnet)
    if args.max_batch_size_ifrnet:
        config.set("models", "ifrnet", "max_batch_size", value=args.max_batch_size_ifrnet)
    if args.crf_ifrnet is not None:
        config.set("models", "ifrnet", "crf",            value=args.crf_ifrnet)
    if args.codec_ifrnet:
        config.set("models", "ifrnet", "codec",          value=args.codec_ifrnet)
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
    if args.batch_size_esrgan:
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
    if args.codec_esrgan:
        config.set("models", "realesrgan", "codec",           value=args.codec_esrgan)
    if args.x264_preset_esrgan:
        config.set("models", "realesrgan", "x264_preset",     value=args.x264_preset_esrgan)
    if getattr(args, "ffmpeg_bin", None):
        config.set("models", "realesrgan", "ffmpeg_bin",      value=args.ffmpeg_bin)

    # ── face_enhance ─────────────────────────────────────────────────────────
    if args.face_enhance is not None:
        config.set("models", "realesrgan", "face_enhance",       value=args.face_enhance)
    if args.gfpgan_model:
        config.set("models", "realesrgan", "gfpgan_model",       value=args.gfpgan_model)
    if args.gfpgan_weight is not None:
        config.set("models", "realesrgan", "gfpgan_weight",      value=args.gfpgan_weight)
    if args.gfpgan_batch_size:
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

    # ── 最终合并输出参数 ─────────────────────────────────────────────────────
    if args.output_codec:
        config.set("output", "codec",  value=args.output_codec)
    if args.output_crf is not None:
        config.set("output", "crf",    value=args.output_crf)
    if args.output_preset:
        config.set("output", "preset", value=args.output_preset)

    # ── TRT 缓存目录（全局，IFRNet 与 ESRGan 共享）────────────────────────
    if args.trt_cache_dir:
        config.set("paths", "trt_cache_dir", value=args.trt_cache_dir)


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
    from ifrnet_processor_v6_single           import IFRNetProcessor          # noqa
    from realesrgan_processor_video_optimized import RealESRGANVideoProcessor  # noqa

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

    if skip_interpolate and skip_upscale:
        print("❌ --skip-interpolate 与 --skip-upscale 不能同时指定")
        return False

    # ── 1. 提取原始音频（从原始输入一次性提取，后续合并步骤无损回写）────────
    audio_path: Optional[str] = None
    try:
        info = VideoInfo(input_video)
        if info.has_audio:
            print("🎵 提取原始音频...")
            audio_path = smart_extract_audio(
                input_video,
                str(config.get_temp_dir("main_audio")),
            )
            if audio_path:
                print(f"   ✅ 音频已暂存: {audio_path}")
            else:
                print("   ⚠️  音频提取失败，输出将无音频")
    except Exception as e:
        print(f"   ⚠️  音频提取异常（继续处理）: {e}")

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
    if skip_interpolate:
        _print_stage(2, "Real-ESRGAN 视频超分（优化版）", "🎨")
        try:
            proc = RealESRGANVideoProcessor(config)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return False
        except Exception as e:
            print(f"❌ 初始化 RealESRGAN 处理器失败: {e}")
            traceback.print_exc()
            return False
        try:
            success = proc.process_video(actual_input, output_video)
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断（Ctrl+C），断点已保存。")
            return False
        if success:
            _run_postprocess_stage(output_video, input_video,
                                   denoised_path, args, env_info)
            _print_completion(output_video, time.time() - t0, env_info)
        else:
            _print_failure_hints(time.time() - t0)
        return success

    # ── 3b. 单步模式 —— 仅插帧 ───────────────────────────────────────────────
    if skip_upscale:
        _print_stage(2, "IFRNet 视频插帧", "🎞️")
        try:
            proc = IFRNetProcessor(config)
            proc.preview          = _preview_ifr
            proc.preview_interval = _preview_ifr_intv
        except Exception as e:
            print(f"❌ 初始化 IFRNet 处理器失败: {e}")
            traceback.print_exc()
            return False
        try:
            success = proc.process_video(actual_input, output_video)
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断（Ctrl+C），断点已保存。")
            return False
        if success:
            _run_postprocess_stage(output_video, input_video,
                                   denoised_path, args, env_info)
            _print_completion(output_video, time.time() - t0, env_info)
        else:
            _print_failure_hints(time.time() - t0)
        return success

    # ── 3c. 双步模式 —— 创建处理器 ───────────────────────────────────────────
    try:
        ifrnet_proc = IFRNetProcessor(config)
        ifrnet_proc.preview          = _preview_ifr
        ifrnet_proc.preview_interval = _preview_ifr_intv
        esrgan_proc = RealESRGANVideoProcessor(config)
    except Exception as e:
        print(f"❌ 初始化处理器失败: {e}")
        traceback.print_exc()
        return False

    # ── 双步执行 ──────────────────────────────────────────────────────────────
    final_segs = None

    if mode == "interpolate_then_upscale":
        # Step 1: IFRNet
        _print_stage(2, "Step 1/2 — IFRNet 插帧", "🎞️")
        try:
            step1_segs = ifrnet_proc.process_video_segments(actual_input)
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断（Ctrl+C），断点已保存。")
            return False
        if not step1_segs:
            print("❌ IFRNet 插帧未产生有效分段，流程终止")
            _print_failure_hints(time.time() - t0)
            return False
        print(f"\n   ✅ Step 1 完成，产生 {len(step1_segs)} 个分段")

        # Step 2: ESRGan
        _print_stage(2, "Step 2/2 — Real-ESRGAN 超分（优化版）", "🎨")
        try:
            final_segs = esrgan_proc.process_segments_directly(
                step1_segs, video_name)
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断（Ctrl+C），断点已保存。")
            return False

    elif mode == "upscale_then_interpolate":
        # Step 1: ESRGan
        _print_stage(2, "Step 1/2 — Real-ESRGAN 超分（优化版）", "🎨")
        try:
            step1_segs = esrgan_proc.process_video_segments(actual_input)
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断（Ctrl+C），断点已保存。")
            return False
        if not step1_segs:
            print("❌ Real-ESRGAN 超分未产生有效分段，流程终止")
            _print_failure_hints(time.time() - t0)
            return False
        print(f"\n   ✅ Step 1 完成，产生 {len(step1_segs)} 个分段")

        # Step 2: IFRNet
        _print_stage(2, "Step 2/2 — IFRNet 插帧", "🎞️")
        try:
            final_segs = ifrnet_proc.process_segments_directly(
                step1_segs, video_name)
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断（Ctrl+C），断点已保存。")
            return False

    else:
        print(f"❌ 未知模式: {mode}，可选值: "
              f"interpolate_then_upscale | upscale_then_interpolate")
        return False

    if not final_segs:
        print("❌ 第二步处理未产生有效分段，流程终止")
        _print_failure_hints(time.time() - t0)
        return False

    # ── 4. 合并最终分段（含音频无损回写）──────────────────────────────────────
    print(f"\n🔗 合并 {len(final_segs)} 个最终分段 → {output_video}")
    output_config = config.get_section("output", {})
    success = merge_videos_by_codec(
        final_segs, output_video,
        audio_path=audio_path,
        config=output_config,
    )

    if success:
        elapsed = time.time() - t0
        print(f"\n✅ 全流程完成！总耗时: {format_time(elapsed)}")
        print(f"📤 输出: {output_video}")
        if os.path.exists(output_video):
            print(f"   文件大小: {os.path.getsize(output_video)/1024/1024:.1f} MB")
            
        # 5. 后处理
        _run_postprocess_stage(output_video, input_video,
                               denoised_path, args, env_info)
        _print_completion(output_video, time.time() - t0, env_info)

        # 自动清理临时分段文件
        if config.get("processing", "auto_cleanup_temp", default=False):
            print("🧹 自动清理临时分段文件...")
            for proc in (ifrnet_proc, esrgan_proc):
                try:
                    proc._cleanup_temp_files()
                except Exception:
                    pass
            print("   ✅ 清理完成")
    else:
        print("❌ 最终合并失败")
        _print_failure_hints(time.time() - t0)

    return success


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
        print(f"\n{'=' * 70}")
        print(f"  [{idx + 1}/{total}] {src.name}")
        print(f"{'=' * 70}")

        config.set("paths", "input_video", value=str(src))
        config.set("paths", "output_dir",  value=output_dir)

        ok = _process_single(
            config, str(src), str(dst), mode,
            skip_interpolate, skip_upscale,
            args, env_info,
        )
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
  IFRNet     : src/processors/ifrnet_processor_v6_single.py
  Real-ESRGAN: src/processors/realesrgan_processor_video_optimized.py
               → external/realesrgan_video_ds/main.py (main_optimized, v6.4)

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
               --codec-esrgan / --x264-preset-esrgan /
               --face-det-threshold / --no-adaptive-batch-esrgan / --gfpgan-trt /
               --report-esrgan / --preview-esrgan / --preview-interval-esrgan
  人脸增强   : --face-enhance / --gfpgan-model / --gfpgan-weight / --gfpgan-batch-size
  合并输出   : --output-codec / --output-crf / --output-preset
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
                   help="IFRNet 批处理大小（覆盖配置，默认 4）")
    g.add_argument("--max-batch-size-ifrnet", type=int, metavar="N",
                   help="IFRNet 批大小上限（OOM 天花板，覆盖配置，默认 8）")
    g.add_argument("--no-fp16-ifrnet",       action="store_true",
                   help="禁用 IFRNet FP16（默认开启）")
    g.add_argument("--no-compile-ifrnet",    action="store_true",
                   help="禁用 IFRNet torch.compile（短视频可禁用跳过预热）")
    g.add_argument("--no-cuda-graph-ifrnet", action="store_true",
                   help="禁用 IFRNet CUDA Graph（compile 激活时已接管，可安全禁用）")
    g.add_argument("--use-tensorrt-ifrnet",  action="store_true",
                   help="启用 IFRNet TensorRT 加速（首次需构建 Engine，缓存于 .trt_cache/）")
    g.add_argument("--no-hwaccel-ifrnet",    action="store_true",
                   help="禁用 IFRNet NVDEC 硬件解码")
    g.add_argument("--no-audio-ifrnet",      action="store_true",
                   help="IFRNet 分段处理时不保留音轨（主流程会统一处理音频）")
    g.add_argument("--crf-ifrnet",  type=int, metavar="N",
                   help="IFRNet 分段输出 CRF（0~51，默认 23）")
    g.add_argument("--codec-ifrnet", metavar="CODEC",
                   help="IFRNet 分段输出编码器（默认 libx264，有 NVENC 时自动升级）")
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

    # ── Real-ESRGAN 参数（已更新，对齐 realesrgan_video_ds/main.py）──────────
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
                   help="ESRGan SR 批处理大小（默认 6）")
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
                   help="启用 ESRGan TensorRT 加速（首次需构建 Engine，缓存于 .trt_cache/）")
    g.add_argument("--no-hwaccel-esrgan", action="store_true",
                   help="禁用 ESRGan NVDEC 硬件解码")
    g.add_argument("--crf-esrgan", type=int, metavar="N",
                   help="ESRGan 分段输出 CRF（0~51，默认 23）")
    g.add_argument("--codec-esrgan", metavar="CODEC",
                   help="ESRGan 分段输出编码器（默认 libx264，有 NVENC 时自动升级；可选 libx265/h264_nvenc）")
    g.add_argument("--x264-preset-esrgan", metavar="PRESET",
                   choices=["ultrafast", "superfast", "veryfast",
                            "faster", "fast", "medium",
                            "slow", "slower", "veryslow"],
                   help="ESRGan libx264/libx265 编码预设（默认 medium）")
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

    # ── 最终合并输出参数 ─────────────────────────────────────────────────────
    g = parser.add_argument_group("最终合并输出参数")
    g.add_argument("--output-codec",  metavar="CODEC",
                   help="最终合并编码器（覆盖配置，默认 libx264）")
    g.add_argument("--output-crf",    type=int, metavar="N",
                   help="最终合并 CRF 质量（0~51，覆盖配置，默认 23）")
    g.add_argument("--output-preset", metavar="PRESET",
                   help="最终合并编码预设（如 medium / slow，覆盖配置）")

    # ── TRT 缓存目录（全局）─────────────────────────────────────────────────
    g = parser.add_argument_group("TRT Engine 缓存（IFRNet / ESRGan 共用）")
    g.add_argument("--trt-cache-dir", metavar="DIR",
                   help="TRT Engine 缓存目录（覆盖配置 paths.trt_cache_dir；"
                        "未指定时从配置读取；配置为空时自动使用 base_dir/.trt_cache）")

    # ── 杂项 ──────────────────────────────────────────────────────   [V1+V2]
    g = parser.add_argument_group("杂项")
    g.add_argument("--auto-cleanup", action="store_true",
                   help="全流程结束后自动删除所有临时分段文件")
    g.add_argument("--keep-intermediate", action="store_true",
                   help="保留去噪等中间文件（调试用）")
    g.add_argument("--dry-run", action="store_true",
                   help="仅打印配置和环境信息，不实际处理")
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
    _print_stage(0, "输入验证与环境检查", "🔍")

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
                       or config.get("paths", "input_video", default=""))
        output_video = args.output or ""
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
        print("\n\n⚠️  用户中断（Ctrl+C）")
        print("   断点已保存，下次运行相同参数可从断点恢复。")
        _exit_code = 130
    except Exception as _exc:
        print(f"\n❌ 未捕获的异常: {_exc}")
        traceback.print_exc()
        _exit_code = 1

    sys.exit(_exit_code)