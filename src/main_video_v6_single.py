"""
视频增强主流程 —— v6（单卡版）
==============================
统一调度 IFRNet 插帧 + Real-ESRGAN 超分，支持两种处理顺序：
  interpolate_then_upscale  : 先插帧（IFRNet）→ 再超分（ESRGan）
  upscale_then_interpolate  : 先超分（ESRGan）→ 再插帧（IFRNet）

【v6 亮点（相对 v5）】
  · 全部底层控制参数均可命令行透传：
      IFRNet: --use-tensorrt-ifrnet / --no-fp16-ifrnet / --no-compile-ifrnet /
              --no-cuda-graph-ifrnet / --no-hwaccel-ifrnet / --batch-size-ifrnet /
              --max-batch-size-ifrnet / --crf-ifrnet / --codec-ifrnet /
              --ifrnet-model / --ifrnet-model-path / --report-ifrnet /
              --preview-ifrnet / --preview-interval-ifrnet
      ESRGan: --use-tensorrt-esrgan / --use-compile-esrgan / --fp32-esrgan /
              --no-hwaccel-esrgan / --batch-size-esrgan / --prefetch-factor-esrgan /
              --crf-esrgan / --codec-esrgan / --tile-size / --tile-pad /
              --pre-pad / --denoise-strength / --face-enhance / --gfpgan-model /
              --gfpgan-weight / --gfpgan-batch-size / --report-esrgan
  · 批量模式（--batch-mode）：从 --input-dir 批量读取，输出到 --output-dir
  · 断点恢复：各处理器内部独立管理分段断点
  · 原始音频从输入一次性提取，合并时无损回写
  · --skip-interpolate / --skip-upscale 可单独运行任一步骤

【处理流程】

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
  python main_video_v6_single.py -i input.mp4 -o output.mp4

  # 反转顺序 + 不同倍数
  python main_video_v6_single.py -i input.mp4 -o output.mp4 \\
      --mode upscale_then_interpolate --upscale-factor 4 --interpolation-factor 2

  # 开启 IFRNet + ESRGan TRT 加速
  python main_video_v6_single.py -i input.mp4 -o output.mp4 \\
      --use-tensorrt-ifrnet --use-tensorrt-esrgan \\
      --batch-size-ifrnet 8 --batch-size-esrgan 16

  # 仅超分（跳过插帧）+ 开启人脸增强
  python main_video_v6_single.py -i input.mp4 -o output.mp4 \\
      --skip-interpolate --use-tensorrt-esrgan \\
      --face-enhance --gfpgan-model 1.4 --gfpgan-weight 0.5

  # 仅插帧（跳过超分）
  python main_video_v6_single.py -i input.mp4 -o output.mp4 \\
      --skip-upscale --interpolation-factor 4

  # 批量模式
  python main_video_v6_single.py --batch-mode \\
      --input-dir /data/raw/ --output-dir /data/enhanced/

  # 指定配置文件 + 覆盖关键参数
  python main_video_v6_single.py -c my_config.json \\
      -i input.mp4 -o output.mp4 \\
      --interpolation-factor 4 --upscale-factor 2 \\
      --no-fp16-ifrnet --use-compile-esrgan
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ----- Add the utils directory so that video_utils and config_manager can be found -----
# _SCRIPT_DIR   = Path(__file__).resolve().parent          # src/processors
# _PROJECT_ROOT = _SCRIPT_DIR.parent.parent                # Video_Enhancement/
# _UTILS_PATH   = str(_PROJECT_ROOT / "src" / "utils")
# if _UTILS_PATH not in sys.path:
#     sys.path.insert(0, _UTILS_PATH)
# ----------------------------------------------------------------------------------------

# 动态定位项目根目录（本文件在 src/，根目录在上一层）
_SRC_DIR  = Path(os.path.abspath(__file__)).parent        # …/src
_BASE_DIR = _SRC_DIR.parent                               # …/Video_Enhancement

# 导入自定义工具
_utils_path      = str(_BASE_DIR / "src" / "utils")
_processors_path = str(_BASE_DIR / "src" / "processors")
for _p in (_utils_path, _processors_path):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config_manager import Config
from video_utils import (
    format_time, get_video_duration, merge_videos_by_codec,
    smart_extract_audio, VideoInfo,
)

_DEFAULT_CFG = str(_BASE_DIR / "config" / "default_config.json")


# =============================================================================
# 命令行参数覆盖配置
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

    # ── IFRNet 模型参数 ──────────────────────────────────────────────────────
    if args.ifrnet_model_path:
        config.set("models", "ifrnet", "model_path", value=args.ifrnet_model_path)
        config.set("models", "ifrnet", "model_name", value="")
    elif args.ifrnet_model:
        config.set("models", "ifrnet", "model_name", value=args.ifrnet_model)
        config.set("models", "ifrnet", "model_path", value="")

    # ── IFRNet 推理优化 ──────────────────────────────────────────────────────
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

    # ── ESRGan 模型参数 ──────────────────────────────────────────────────────
    if args.esrgan_model:
        config.set("models", "realesrgan", "model_name",       value=args.esrgan_model)
    if args.denoise_strength is not None:
        config.set("models", "realesrgan", "denoise_strength", value=args.denoise_strength)

    # ── ESRGan 推理优化 ──────────────────────────────────────────────────────
    if args.fp32_esrgan:
        config.set("models", "realesrgan", "fp32",            value=True)
    if args.use_compile_esrgan:
        config.set("models", "realesrgan", "use_compile",     value=True)
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
    if args.report_esrgan:
        config.set("models", "realesrgan", "report_json",     value=args.report_esrgan)

    # ── face_enhance ─────────────────────────────────────────────────────────
    if args.face_enhance is not None:
        config.set("models", "realesrgan", "face_enhance",      value=args.face_enhance)
    if args.gfpgan_model:
        config.set("models", "realesrgan", "gfpgan_model",      value=args.gfpgan_model)
    if args.gfpgan_weight is not None:
        config.set("models", "realesrgan", "gfpgan_weight",     value=args.gfpgan_weight)
    if args.gfpgan_batch_size:
        config.set("models", "realesrgan", "gfpgan_batch_size", value=args.gfpgan_batch_size)

    # ── 最终合并输出参数 ─────────────────────────────────────────────────────
    if args.output_codec:
        config.set("output", "codec",  value=args.output_codec)
    if args.output_crf is not None:
        config.set("output", "crf",    value=args.output_crf)
    if args.output_preset:
        config.set("output", "preset", value=args.output_preset)


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
    # IFRNet preview 参数（不通过 Config 传递，而是在实例化后直接注入）
    preview_ifrnet:          bool = False,
    preview_interval_ifrnet: int  = 30,
) -> bool:
    """
    处理单个视频文件：
      1. 提取原始音频（若存在）
      2. 按 mode 链式调用 IFRNetProcessor / RealESRGANVideoProcessor
      3. 合并最终分段并回写音频

    Returns:
        是否成功
    """
    from ifrnet_processor_v6_single           import IFRNetProcessor
    from realesrgan_processor_video_v6_single import RealESRGANVideoProcessor

    t0         = time.time()
    video_name = Path(input_video).stem

    print("\n" + "=" * 70)
    print("  🚀 视频增强主流程 v6（单卡版）")
    print("=" * 70)
    print(f"  输入  : {input_video}")
    print(f"  输出  : {output_video}")
    print(f"  模式  : {mode}")
    if skip_interpolate:
        print("  ⚠️  跳过插帧步骤（仅超分）")
    if skip_upscale:
        print("  ⚠️  跳过超分步骤（仅插帧）")
    print("=" * 70 + "\n")

    if skip_interpolate and skip_upscale:
        print("❌ --skip-interpolate 与 --skip-upscale 不能同时指定")
        return False

    # ── 提取原始音频（从输入一次性提取，后续合并步骤无损回写）────────────────
    audio_path: Optional[str] = None
    try:
        info = VideoInfo(input_video)
        if info.has_audio:
            print("🎵 提取原始音频...")
            audio_path = smart_extract_audio(
                input_video,
                str(config.get_temp_dir("main_audio"))
            )
            if audio_path:
                print(f"   ✅ 音频已暂存: {audio_path}")
            else:
                print("   ⚠️  音频提取失败，输出将无音频")
    except Exception as e:
        print(f"   ⚠️  音频提取异常（继续处理）: {e}")

    # ── 单步模式 ──────────────────────────────────────────────────────────────
    if skip_interpolate:
        try:
            proc = RealESRGANVideoProcessor(config)
            return proc.process_video(input_video, output_video)
        except Exception as e:
            print(f"❌ 初始化 RealESRGAN 处理器失败: {e}")
            return False

    if skip_upscale:
        try:
            proc = IFRNetProcessor(config)
            proc.preview          = preview_ifrnet
            proc.preview_interval = preview_interval_ifrnet
            return proc.process_video(input_video, output_video)
        except Exception as e:
            print(f"❌ 初始化 IFRNet 处理器失败: {e}")
            return False

    # ── 双步模式：创建处理器 ──────────────────────────────────────────────────
    try:
        ifrnet_proc  = IFRNetProcessor(config)
        ifrnet_proc.preview          = preview_ifrnet
        ifrnet_proc.preview_interval = preview_interval_ifrnet
        esrgan_proc  = RealESRGANVideoProcessor(config)
    except Exception as e:
        print(f"❌ 初始化处理器失败: {e}")
        return False

    # ── 按模式执行两步处理 ────────────────────────────────────────────────────
    if mode == "interpolate_then_upscale":
        print("\n" + "─" * 60)
        print("  📌 Step 1/2 — IFRNet 插帧")
        print("─" * 60)
        step1_segs = ifrnet_proc.process_video_segments(input_video)
        if not step1_segs:
            print("❌ IFRNet 插帧未产生有效分段，流程终止")
            return False

        print("\n" + "─" * 60)
        print("  📌 Step 2/2 — Real-ESRGAN 超分")
        print("─" * 60)
        final_segs = esrgan_proc.process_segments_directly(step1_segs, video_name)

    elif mode == "upscale_then_interpolate":
        print("\n" + "─" * 60)
        print("  📌 Step 1/2 — Real-ESRGAN 超分")
        print("─" * 60)
        step1_segs = esrgan_proc.process_video_segments(input_video)
        if not step1_segs:
            print("❌ Real-ESRGAN 超分未产生有效分段，流程终止")
            return False

        print("\n" + "─" * 60)
        print("  📌 Step 2/2 — IFRNet 插帧")
        print("─" * 60)
        final_segs = ifrnet_proc.process_segments_directly(step1_segs, video_name)

    else:
        print(f"❌ 未知模式: {mode}，可选值: "
              f"interpolate_then_upscale | upscale_then_interpolate")
        return False

    if not final_segs:
        print("❌ 第二步处理未产生有效分段，流程终止")
        return False

    # ── 合并最终分段（含音频回写）────────────────────────────────────────────
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
        if config.get("processing", "auto_cleanup_temp", default=False):
            for proc in (ifrnet_proc, esrgan_proc):
                try:
                    proc._cleanup_temp_files()
                except Exception:
                    pass
    else:
        print("❌ 最终合并失败")

    return success


# =============================================================================
# 批量模式
# =============================================================================

def _process_batch(
    config:           Config,
    input_dir:        str,
    output_dir:       str,
    mode:             str,
    skip_interpolate: bool,
    skip_upscale:     bool,
    preview_ifrnet:          bool,
    preview_interval_ifrnet: int,
) -> int:
    """批量处理目录下所有视频文件，返回成功数量。"""
    VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".ts"}
    input_files = sorted([
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in VIDEO_EXTS
    ])
    if not input_files:
        print(f"⚠️  输入目录 {input_dir} 中未找到视频文件")
        return 0

    os.makedirs(output_dir, exist_ok=True)
    total   = len(input_files)
    success = 0

    print(f"\n📦 批量模式: 共 {total} 个文件")
    print(f"   输入目录: {input_dir}")
    print(f"   输出目录: {output_dir}")
    print(f"   处理模式: {mode}\n")

    for idx, src_path in enumerate(input_files):
        dst_path = Path(output_dir) / src_path.name
        print(f"\n{'='*70}")
        print(f"  [{idx+1}/{total}] {src_path.name}")
        print(f"{'='*70}")
        ok = _process_single(
            config, str(src_path), str(dst_path),
            mode, skip_interpolate, skip_upscale,
            preview_ifrnet=preview_ifrnet,
            preview_interval_ifrnet=preview_interval_ifrnet,
        )
        if ok:
            success += 1
            print(f"  ✅ 完成 ({idx+1}/{total})")
        else:
            print(f"  ❌ 失败 ({idx+1}/{total})，继续下一个")

    print(f"\n📊 批量处理完成: 成功 {success}/{total}")
    return success


# =============================================================================
# CLI 参数定义
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="视频增强主流程 v6（单卡版）—— IFRNet 插帧 + Real-ESRGAN 超分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
参数分组：
  基础参数   : -i / -o / -c / --batch-mode / --input-dir / --output-dir
  处理控制   : --mode / --interpolation-factor / --upscale-factor /
               --segment-duration / --skip-interpolate / --skip-upscale
  IFRNet参数 : --ifrnet-model* / --use-tensorrt-ifrnet / --no-fp16-ifrnet /
               --no-compile-ifrnet / --no-cuda-graph-ifrnet / --no-hwaccel-ifrnet /
               --batch-size-ifrnet / --max-batch-size-ifrnet / --crf-ifrnet /
               --codec-ifrnet / --report-ifrnet / --preview-ifrnet
  ESRGan参数 : --esrgan-model / --use-tensorrt-esrgan / --use-compile-esrgan /
               --fp32-esrgan / --no-hwaccel-esrgan / --batch-size-esrgan /
               --prefetch-factor-esrgan / --tile-size / --tile-pad / --pre-pad /
               --denoise-strength / --crf-esrgan / --codec-esrgan /
               --report-esrgan
  人脸增强   : --face-enhance / --gfpgan-model / --gfpgan-weight / --gfpgan-batch-size
  合并输出   : --output-codec / --output-crf / --output-preset
""",
    )

    # ── 基础参数 ─────────────────────────────────────────────────────────────
    g = parser.add_argument_group("基础参数")
    g.add_argument("--config", "-c", default=_DEFAULT_CFG,
                   help=f"配置文件路径（默认: {_DEFAULT_CFG}）")
    g.add_argument("--input",  "-i",
                   help="输入视频路径（单文件模式必填）")
    g.add_argument("--output", "-o",
                   help="输出视频路径（含文件名，单文件模式必填）")

    # ── 批量模式 ─────────────────────────────────────────────────────────────
    g = parser.add_argument_group("批量模式")
    g.add_argument("--batch-mode",  action="store_true",
                   help="启用批量模式（扫描 --input-dir 目录下所有视频）")
    g.add_argument("--input-dir",   metavar="DIR",
                   help="批量输入视频目录（batch-mode 必填）")
    g.add_argument("--output-dir",  metavar="DIR",
                   help="批量输出目录（batch-mode 必填）")

    # ── 全局处理控制 ─────────────────────────────────────────────────────────
    g = parser.add_argument_group("处理控制")
    g.add_argument("--mode",
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
    g.add_argument("--auto-cleanup", action="store_true",
                   help="全流程结束后自动删除所有临时分段文件")

    # ── IFRNet 参数 ──────────────────────────────────────────────────────────
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

    # ── Real-ESRGAN 参数 ─────────────────────────────────────────────────────
    g = parser.add_argument_group("Real-ESRGAN 参数")
    g.add_argument("--esrgan-model", metavar="MODEL_NAME",
                   help="ESRGan 模型名称（覆盖配置；不存在时自动下载）\n"
                        "可选: realesr-general-x4v3 | RealESRGAN_x4plus | "
                        "RealESRGAN_x2plus | realesr-animevideov3 | "
                        "RealESRGANv2-animevideo-xsx2")
    g.add_argument("--denoise-strength", type=float, metavar="F",
                   help="降噪强度 0~1（仅 realesr-general-x4v3，覆盖配置）")
    g.add_argument("--batch-size-esrgan", type=int, metavar="N",
                   help="ESRGan SR 批处理大小（覆盖配置，默认 12）")
    g.add_argument("--prefetch-factor-esrgan", type=int, metavar="N",
                   help="ESRGan 读帧预取深度（建议 ≥ batch_size×2，默认 24）")
    g.add_argument("--tile-size",  type=int, metavar="N",
                   help="tile 切块大小（0=不切块；VRAM 不足时设 512）")
    g.add_argument("--tile-pad",   type=int, metavar="N",
                   help="tile 边缘填充（默认 10）")
    g.add_argument("--pre-pad",    type=int, metavar="N",
                   help="预处理填充（默认 0）")
    g.add_argument("--fp32-esrgan", action="store_true",
                   help="ESRGan 使用 FP32 精度（默认 FP16）")
    g.add_argument("--use-compile-esrgan", action="store_true",
                   help="启用 ESRGan torch.compile（reduce-overhead 模式）")
    g.add_argument("--use-tensorrt-esrgan", action="store_true",
                   help="启用 ESRGan TensorRT 加速（首次需构建 Engine，缓存于 .trt_cache/）")
    g.add_argument("--no-hwaccel-esrgan", action="store_true",
                   help="禁用 ESRGan NVDEC 硬件解码")
    g.add_argument("--crf-esrgan",  type=int, metavar="N",
                   help="ESRGan 分段输出 CRF（0~51，默认 23）")
    g.add_argument("--codec-esrgan", metavar="CODEC",
                   help="ESRGan 分段输出编码器（默认 libx264，有 NVENC 时自动升级）")
    g.add_argument("--report-esrgan", metavar="PATH",
                   help="ESRGan JSON 性能报告输出路径")

    # ── face_enhance 参数 ─────────────────────────────────────────────────────
    g = parser.add_argument_group("face_enhance 参数（Real-ESRGAN）")
    _fe = g.add_mutually_exclusive_group()
    _fe.add_argument("--face-enhance",    dest="face_enhance", action="store_true",
                     default=None, help="开启人脸增强（GFPGAN）")
    _fe.add_argument("--no-face-enhance", dest="face_enhance", action="store_false",
                     help="关闭人脸增强（覆盖配置）")
    g.add_argument("--gfpgan-model", choices=["1.3", "1.4", "RestoreFormer"],
                   help="GFPGAN 版本（默认 1.4）")
    g.add_argument("--gfpgan-weight", type=float, metavar="F",
                   help="GFPGAN 融合权重 0.0~1.0（0=不增强，1=完全替换，默认 0.5）")
    g.add_argument("--gfpgan-batch-size", type=int, metavar="N",
                   help="单次 GFPGAN 前向最多处理的人脸数（OOM 保护，默认 12）")

    # ── 最终合并输出参数 ─────────────────────────────────────────────────────
    g = parser.add_argument_group("最终合并输出参数")
    g.add_argument("--output-codec",  metavar="CODEC",
                   help="最终合并编码器（覆盖配置，默认 libx264）")
    g.add_argument("--output-crf",    type=int, metavar="N",
                   help="最终合并 CRF 质量（0~51，覆盖配置，默认 23）")
    g.add_argument("--output-preset", metavar="PRESET",
                   help="最终合并编码预设（如 medium / slow，覆盖配置）")

    return parser


# =============================================================================
# 启动信息打印
# =============================================================================

def _print_startup_info(config: Config, args: argparse.Namespace, mode: str) -> None:
    """打印启动配置摘要（便于快速核对参数）。"""
    ifr = lambda k, d=None: config.get("models", "ifrnet", k, default=d)
    esr = lambda k, d=None: config.get("models", "realesrgan", k, default=d)

    print("\n" + "─" * 70)
    print("  配置摘要")
    print("─" * 70)
    print(f"  处理模式      : {mode}")
    print(f"  插帧倍数      : {config.get('processing','interpolation_factor',default=2)}×"
          f"  |  超分倍数: {config.get('processing','upscale_factor',default=2)}×")
    print(f"  分段时长      : {config.get('processing','segment_duration',default=30)}秒")
    print()
    print(f"  ── IFRNet ──")
    print(f"     模型       : {ifr('model_name','IFRNet_S_Vimeo90K')}")
    print(f"     FP16       : {ifr('use_fp16',True)}"
          f"  |  compile: {ifr('use_compile',True)}"
          f"  |  CUDA Graph: {ifr('use_cuda_graph',True)}"
          f"  |  TRT: {ifr('use_tensorrt',False)}")
    print(f"     batch_size : {ifr('batch_size',4)} (上限 {ifr('max_batch_size',8)})"
          f"  |  NVDEC: {ifr('use_hwaccel',True)}")
    print()
    print(f"  ── Real-ESRGAN ──")
    print(f"     模型       : {esr('model_name','realesr-general-x4v3')}")
    print(f"     FP16       : {not esr('fp32',False)}"
          f"  |  compile: {esr('use_compile',False)}"
          f"  |  TRT: {esr('use_tensorrt',False)}")
    print(f"     batch_size : {esr('batch_size',12)}"
          f"  |  prefetch: {esr('prefetch_factor',24)}"
          f"  |  NVDEC: {esr('use_hwaccel',True)}")
    face_on = esr('face_enhance', False)
    print(f"     face_enhance: {face_on}" + (
        f" (GFPGAN-{esr('gfpgan_model','1.4')}"
        f", w={esr('gfpgan_weight',0.5)}"
        f", batch={esr('gfpgan_batch_size',12)})"
        if face_on else ""
    ))
    print("─" * 70 + "\n")


# =============================================================================
# main()
# =============================================================================

def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────────────────────────────
    try:
        config = Config(args.config)
        print(f"⚙️  配置文件: {args.config}")
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        return 1

    # ── 命令行参数覆盖写入配置 ────────────────────────────────────────────────
    _apply_cli_overrides(config, args)

    # ── 确定处理模式 ──────────────────────────────────────────────────────────
    mode = config.get("processing", "mode", default="interpolate_then_upscale")

    # ── 打印启动摘要 ──────────────────────────────────────────────────────────
    _print_startup_info(config, args, mode)

    # ── 分支：批量 / 单文件 ───────────────────────────────────────────────────
    is_batch = args.batch_mode or config.get("processing", "batch_mode", default=False)

    if is_batch:
        input_dir  = args.input_dir  or config.get("paths", "input_dir",  default="")
        output_dir = args.output_dir or config.get("paths", "output_dir", default="")
        if not input_dir:
            print("❌ 批量模式需指定 --input-dir 或在配置中设置 paths.input_dir")
            return 1
        if not output_dir:
            print("❌ 批量模式需指定 --output-dir 或在配置中设置 paths.output_dir")
            return 1
        n = _process_batch(
            config, input_dir, output_dir, mode,
            args.skip_interpolate, args.skip_upscale,
            preview_ifrnet=args.preview_ifrnet,
            preview_interval_ifrnet=args.preview_interval_ifrnet,
        )
        return 0 if n > 0 else 1

    else:
        input_video  = args.input  or config.get("paths", "input_video", default="")
        output_video = args.output or ""
        if not input_video:
            print("❌ 单文件模式需指定 --input 或在配置中设置 paths.input_video")
            return 1
        if not output_video:
            print("❌ 单文件模式需指定 --output（含文件名）")
            return 1
        if not os.path.isfile(input_video):
            print(f"❌ 输入文件不存在: {input_video}")
            return 1

        # 将路径写入 config（供处理器内部 get_temp_dir 等使用）
        config.set("paths", "input_video", value=input_video)
        config.set("paths", "output_dir",  value=str(Path(output_video).parent))
        config.set("processing", "batch_mode", value=False)

        ok = _process_single(
            config, input_video, output_video, mode,
            args.skip_interpolate, args.skip_upscale,
            preview_ifrnet=args.preview_ifrnet,
            preview_interval_ifrnet=args.preview_interval_ifrnet,
        )
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
