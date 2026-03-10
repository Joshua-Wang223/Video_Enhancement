"""
Real-ESRGAN 视频超分处理器 v5（单卡版）
==========================================
对接 inference_realesrgan_video_v6_1_single.py（inference_video_single），
保留分段直接对接与断点恢复逻辑，支持 v5/v6 全部硬件加速参数：
  - FP16 / torch.compile
  - TensorRT 可选加速（TRT 8.x / 10.x 双 API 兼容）
  - NVDEC 硬件解码 / NVENC 硬件编码
  - OOM 自动降级
  - 批量推理（batch_size 默认 8）/ JSON 性能报告
  - face_enhance：批量 GFPGAN 推理 + 原始帧检测 + 无脸跳过（v6 新增）
  - CPU-GPU 流水线并行：detect/paste 异步，SR 推理不停顿（v6 新增）
"""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from typing import List, Optional

import torch

# ----- Add the utils directory so that video_utils and config_manager can be found -----
script_dir = Path(__file__).resolve().parent          # src/processors
project_root = script_dir.parent.parent                # Video_Enhancement/
utils_path = str(project_root / "src" / "utils")
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
# ----------------------------------------------------------------------------------------

from video_utils import (
    get_video_duration, format_time, verify_video_integrity,
    split_video_by_time, merge_videos_by_codec
)


class RealESRGANVideoProcessor:
    """Real-ESRGAN 视频超分处理器 v5（单卡版）"""

    def __init__(self, config):
        """
        初始化处理器

        Args:
            config: 配置对象（应包含 paths, models.realesrgan, processing 等节）
        """
        self.config = config
        self.esrgan_dir = Path(config.get("paths", "base_dir")) / "external" / "Real-ESRGAN"
        self.model_name = config.get("models", "realesrgan", "model_name",
                                     default="RealESRGAN_x4plus")
        self.model_path = config.get("models", "realesrgan", "model_path")

        # 推理设备与基础参数
        use_gpu = config.get("models", "realesrgan", "use_gpu", default=True)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        self.upscale_factor   = config.get("processing", "upscale_factor",   default=4)
        self.segment_duration = config.get("processing", "segment_duration",  default=30)

        # Real-ESRGAN 基础参数
        self.denoise_strength = config.get("models", "realesrgan", "denoise_strength", default=0.5)
        self.tile_size        = config.get("models", "realesrgan", "tile_size",        default=0)
        self.tile_pad         = config.get("models", "realesrgan", "tile_pad",         default=10)
        self.pre_pad          = config.get("models", "realesrgan", "pre_pad",          default=0)
        self.fp32             = config.get("models", "realesrgan", "fp32",             default=False)
        self.face_enhance     = config.get("models", "realesrgan", "face_enhance",     default=False)

        # v5 新增：推理优化参数（batch_size/prefetch_factor 对齐 v6 默认值）
        self.batch_size       = config.get("models", "realesrgan", "batch_size",       default=8)
        self.prefetch_factor  = config.get("models", "realesrgan", "prefetch_factor",  default=16)
        self.use_compile      = config.get("models", "realesrgan", "use_compile",      default=False)
        self.use_tensorrt     = config.get("models", "realesrgan", "use_tensorrt",     default=False)

        # v6 新增：face_enhance 精细控制参数
        self.gfpgan_model       = config.get("models", "realesrgan", "gfpgan_model",       default="1.4")
        self.gfpgan_weight      = config.get("models", "realesrgan", "gfpgan_weight",      default=0.5)
        self.gfpgan_batch_size  = config.get("models", "realesrgan", "gfpgan_batch_size",  default=8)

        # v5 新增：硬件解/编码参数
        self.use_hwaccel = config.get("models", "realesrgan", "use_hwaccel", default=True)
        self.video_codec = config.get("models", "realesrgan", "video_codec", default="libx264")
        self.crf         = config.get("models", "realesrgan", "crf",         default=18)
        self.ffmpeg_bin  = config.get("models", "realesrgan", "ffmpeg_bin",  default="ffmpeg")
        self.report_json = config.get("models", "realesrgan", "report_json", default=None)

        # 验证 Real-ESRGAN 目录
        if not self.esrgan_dir.exists():
            raise FileNotFoundError(f"Real-ESRGAN 目录不存在: {self.esrgan_dir}")

        # 将 Real-ESRGAN 加入 Python 路径
        sys.path.insert(0, str(self.esrgan_dir))

        # 临时目录句柄（在 _setup_temp_dirs 中初始化）
        self.checkpoint_file: Optional[Path] = None
        self.segment_dir:     Optional[Path] = None
        self.processed_dir:   Optional[Path] = None

    # -------------------------------------------------------------------------
    # 公共接口
    # -------------------------------------------------------------------------

    def process_video(self, input_video: str, output_video: str) -> bool:
        """
        完整处理视频（分段 → 超分 → 合并），支持断点恢复。

        Args:
            input_video:  输入视频路径
            output_video: 最终输出视频路径

        Returns:
            是否成功
        """
        print("\n" + "=" * 60)
        print("🎨 Real-ESRGAN 视频超分处理（完整流程）")
        print(f"📹 输入: {input_video}")
        print(f"📤 输出: {output_video}")
        print(f"⚡ 超分倍数: {self.upscale_factor}x")
        print(f"🖥️  设备: {self.device}")
        print(f"🧩 分段时长: {self.segment_duration}秒")
        print("=" * 60 + "\n")

        total_start = time.time()

        processed_segments = self.process_video_segments(input_video)
        if not processed_segments:
            print("❌ 未成功处理任何分段")
            return False

        print(f"\n🔗 合并 {len(processed_segments)} 个处理后的分段...")
        output_config = self.config.get_section("output", {})
        success = merge_videos_by_codec(processed_segments, output_video,
                                        config=output_config)

        if success:
            total_time = time.time() - total_start
            print(f"\n✅ 超分处理完成！总用时: {format_time(total_time)}")
            print(f"📤 输出: {output_video}")
            if self.config.get("processing", "auto_cleanup_temp", default=False):
                self._cleanup_temp_files()
            return True
        else:
            print("❌ 视频合并失败")
            return False

    def process_video_segments(self, input_video: str) -> List[str]:
        """
        对完整视频执行超分，返回处理后的分段列表（不合并）。

        Args:
            input_video: 输入视频路径

        Returns:
            处理后的分段文件路径列表
        """
        print(f"\n🎨 Real-ESRGAN 超分处理（分段模式）")
        print(f"📹 输入: {input_video}")
        print(f"⚡ 超分倍数: {self.upscale_factor}x")
        print(f"🖥️  设备: {self.device}")

        video_name = Path(input_video).stem
        self._setup_temp_dirs(video_name, prefix="esrgan_video")
        checkpoint = self._load_checkpoint()

        duration = get_video_duration(input_video)
        if duration is None:
            print("❌ 无法获取视频时长")
            return []
        print(f"📊 时长: {format_time(duration)}, 分段: {self.segment_duration}秒")

        # 视频较短时直接整体处理
        if duration <= self.segment_duration:
            print("📦 视频较短，直接处理整个视频...")
            output_file = self.processed_dir / f"upscaled_{Path(input_video).name}"
            success = self._process_segment(input_video, str(output_file), segment_idx=0)
            return [str(output_file)] if success else []

        print(f"\n🔪 分割视频...")
        segment_files = split_video_by_time(
            input_video,
            str(self.segment_dir),
            self.segment_duration
        )
        if not segment_files:
            print("❌ 视频分割失败")
            return []
        print(f"✅ 共 {len(segment_files)} 个片段")

        return self._process_segments(segment_files, checkpoint)

    def process_segments_directly(self, input_segments: List[str],
                                  video_name: str) -> List[str]:
        """
        直接对已有分段执行超分（用于对接上游处理器输出）。

        Args:
            input_segments: 输入分段文件路径列表
            video_name:     视频名称（用于临时目录命名）

        Returns:
            处理后的分段文件路径列表
        """
        print(f"\n🎨 Real-ESRGAN 超分处理（接收分段输入）")
        print(f"📹 输入分段数: {len(input_segments)}")
        print(f"⚡ 超分倍数: {self.upscale_factor}x")
        print(f"🖥️  设备: {self.device}")

        self._setup_temp_dirs(video_name, prefix="esrgan_from_segments")
        checkpoint = self._load_checkpoint()
        return self._process_segments(input_segments, checkpoint)

    # -------------------------------------------------------------------------
    # 内部核心方法
    # -------------------------------------------------------------------------

    def _setup_temp_dirs(self, video_name: str, prefix: str):
        """创建并记录临时目录路径。"""
        temp_base = (self.config.get_temp_dir("esrgan_video")
                     / f"{prefix}_{video_name}")
        self.segment_dir     = temp_base / "segments"
        self.processed_dir   = temp_base / "processed"
        self.checkpoint_file = temp_base / "checkpoint.json"

        self.segment_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _load_checkpoint(self) -> dict:
        """加载断点信息；文件不存在或损坏时返回空断点。"""
        if self.checkpoint_file and self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    checkpoint = json.load(f)
                print(f"📌 发现断点: "
                      f"已完成 {len(checkpoint.get('processed_segments', []))} 个分段")
                return checkpoint
            except Exception:
                pass
        return {"processed_segments": [], "last_segment": -1}

    def _save_checkpoint(self, checkpoint: dict):
        """将断点信息持久化到磁盘。"""
        if self.checkpoint_file:
            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint, f, indent=2)

    def _process_segments(self, segment_files: List[str],
                          checkpoint: dict) -> List[str]:
        """
        遍历分段列表，跳过已处理项，处理并更新断点。

        Args:
            segment_files: 分段文件路径列表
            checkpoint:    断点字典

        Returns:
            处理成功的分段输出路径列表
        """
        print(f"\n⚙️  开始处理片段...")
        processed_files: List[str] = []
        start_time = time.time()

        for idx, seg_path in enumerate(segment_files):
            seg_name = Path(seg_path).name

            # 断点跳过
            if idx in checkpoint["processed_segments"]:
                print(f"\n⏭️  片段 {idx+1}/{len(segment_files)}: {seg_name} (已处理)")
                out_path = self.processed_dir / f"upscaled_{seg_name}"
                if out_path.exists():
                    processed_files.append(str(out_path))
                    continue

            print(f"\n🎨 片段 {idx+1}/{len(segment_files)}: {seg_name}")
            out_path = self.processed_dir / f"upscaled_{seg_name}"

            success = self._process_segment(seg_path, str(out_path), segment_idx=idx)
            if success:
                processed_files.append(str(out_path))
                checkpoint["processed_segments"].append(idx)
                checkpoint["last_segment"] = idx
                self._save_checkpoint(checkpoint)
            else:
                print(f"⚠️  片段 {idx+1} 处理失败，跳过")
                continue

            # 估算剩余时间
            elapsed = time.time() - start_time
            completed = len(checkpoint["processed_segments"])
            if completed > 0:
                avg_time  = elapsed / completed
                remaining = (len(segment_files) - completed) * avg_time
                print(f"   ⏱️  已用时: {format_time(elapsed)}, "
                      f"预计剩余: {format_time(remaining)}")

        if processed_files:
            print(f"\n✅ Real-ESRGAN 处理完成: "
                  f"{len(processed_files)}/{len(segment_files)} 个分段")
        else:
            print("\n❌ 没有成功处理的片段")

        return processed_files

    def _process_segment(self, input_path: str, output_path: str,
                         segment_idx: int) -> bool:
        """
        处理单个视频片段（调用 inference_realesrgan_video_v6_1_single）。

        Args:
            input_path:   输入片段路径
            output_path:  期望的输出路径
            segment_idx:  片段索引

        Returns:
            是否成功
        """
        try:
            print(f"   🎬 处理片段 {segment_idx+1}: {Path(input_path).name}")
            duration = get_video_duration(input_path)
            if duration:
                print(f"   📊 片段时长: {format_time(duration)}")

            success = self._run_esrgan_video(input_path, output_path, segment_idx)

            if success:
                if verify_video_integrity(output_path):
                    out_duration = get_video_duration(output_path)
                    print(f"   ✅ 处理完成: {format_time(out_duration)}")
                    return True
                else:
                    print("   ❌ 输出文件验证失败")
                    return False
            return False

        except Exception as e:
            print(f"   ❌ 处理片段时发生异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _run_esrgan_video(self, input_path: str, output_path: str,
                          segment_idx: int) -> bool:
        """
        构建参数命名空间并调用 inference_realesrgan_video_v6_1_single.run()。

        v5 变更：
          - 新增 batch_size / prefetch_factor / use_compile
          - 新增 use_tensorrt / use_hwaccel / no_hwaccel
          - 新增 video_codec / crf / report
          - 移除 extract_frame_first / num_process_per_gpu
        v6 变更（新增 face_enhance 精细控制参数）：
          - 新增 gfpgan_model（1.3 / 1.4 / RestoreFormer）
          - 新增 gfpgan_weight（融合权重，0.0~1.0）
          - 新增 gfpgan_batch_size（单次 GFPGAN 前向最多处理的人脸数，防 OOM）

        Args:
            input_path:   输入视频路径
            output_path:  最终输出路径
            segment_idx:  片段索引（用于后缀命名）

        Returns:
            是否成功
        """
        try:
            args = argparse.Namespace()

            # 基本路径
            args.input  = input_path
            args.output = str(Path(output_path).parent)

            # 模型与基础处理参数
            args.model_name       = self.model_name
            args.denoise_strength = self.denoise_strength
            args.outscale         = self.upscale_factor
            args.suffix           = f"processed_{segment_idx:03d}"

            args.tile        = self.tile_size
            args.tile_pad    = self.tile_pad
            args.pre_pad     = self.pre_pad
            args.face_enhance= self.face_enhance
            args.fp32        = self.fp32
            args.fps         = None          # 保持原帧率

            # v5 新增：推理优化参数
            args.batch_size      = self.batch_size
            args.prefetch_factor = self.prefetch_factor
            args.use_compile     = self.use_compile
            args.use_tensorrt    = self.use_tensorrt

            # v5 新增：硬件解/编码参数
            args.use_hwaccel = self.use_hwaccel
            args.no_hwaccel  = not self.use_hwaccel
            args.video_codec = self.video_codec
            args.crf         = self.crf
            args.ffmpeg_bin  = self.ffmpeg_bin

            # v5 新增：性能报告
            args.report = self.report_json

            # v6 新增：face_enhance 精细控制参数
            args.gfpgan_model      = self.gfpgan_model
            args.gfpgan_weight     = self.gfpgan_weight
            args.gfpgan_batch_size = self.gfpgan_batch_size

            # 其他 inference_realesrgan_video_v5_single 所需参数
            args.alpha_upsampler = "realesrgan"
            args.ext             = "auto"

            # 视频名称（用于临时文件前缀）
            video_name    = Path(input_path).stem
            args.video_name = f"{video_name}_{segment_idx:03d}"

            os.makedirs(args.output, exist_ok=True)

            # 动态导入并运行 v6 单卡版
            from inference_realesrgan_video_v6_1_single import run

            print(f"   🔧 加载模型: {self.model_name}")
            print(f"   🖥️  设备: {self.device} | "
                  f"FP16: {not self.fp32} | "
                  f"compile: {self.use_compile} | "
                  f"TRT: {self.use_tensorrt}")
            start_time = time.time()

            run(args)

            # run() 生成文件路径：{output_dir}/{video_name}_{suffix}.mp4
            temp_output = os.path.join(
                args.output, f"{args.video_name}_{args.suffix}.mp4"
            )
            if os.path.exists(temp_output):
                shutil.move(temp_output, output_path)
                elapsed = time.time() - start_time
                print(f"   ✅ 处理完成 ({format_time(elapsed)})")
                return True
            else:
                print(f"   ❌ 输出文件未生成: {temp_output}")
                return False

        except ImportError as e:
            print(f"   ❌ 无法导入 inference_realesrgan_video_v6_1_single: {e}")
            return False
        except Exception as e:
            print(f"   ❌ 调用 Real-ESRGAN 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _cleanup_temp_files(self):
        """清理临时目录（分段和中间处理文件）。"""
        print("\n🧹 清理临时文件...")
        try:
            if self.segment_dir and self.segment_dir.exists():
                shutil.rmtree(self.segment_dir)
                print("✅ 已删除分段文件")
            if self.processed_dir and self.processed_dir.exists():
                shutil.rmtree(self.processed_dir)
                print("✅ 已删除处理文件")
        except Exception as e:
            print(f"⚠️  清理失败: {e}")


# =============================================================================
# 独立命令行入口
# =============================================================================

def main():
    """
    独立调用入口：直接驱动 RealESRGANVideoProcessor，
    底层对接 inference_realesrgan_video_v6_1_single.run()。

    示例：
      # 使用默认配置，直接超分
      python realesrgan_processor_video_v5_single.py -i input.mp4 -o output.mp4

      # 指定配置文件 + 开启人脸增强
      python realesrgan_processor_video_v5_single.py -c config.json \\
             -i input.mp4 -o output.mp4 --face-enhance

      # 覆盖超分倍数、模型、GFPGAN 精细控制
      python realesrgan_processor_video_v5_single.py -i input.mp4 -o output.mp4 \\
             --upscale-factor 4 --model realesr-animevideov3 \\
             --face-enhance --gfpgan-model 1.4 --gfpgan-weight 0.5 \\
             --gfpgan-batch-size 8
    """
    import argparse

    # 自动定位默认配置文件（假设脚本在 src/processors/，config 在项目根/config/）
    _script_dir  = Path(os.path.abspath(__file__)).parent          # src/processors
    _base_dir    = _script_dir.parent.parent                       # project root
    _default_cfg = str(_base_dir / "config" / "default_config.json")

    parser = argparse.ArgumentParser(
        description="Real-ESRGAN 视频超分处理器（单卡版）—— 独立入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
底层脚本：external/Real-ESRGAN/inference_realesrgan_video_v6_1_single.py

特性：
  · 分段处理 + 断点恢复
  · FP16 / NVDEC / NVENC / torch.compile / TensorRT 可选
  · face_enhance：批量GFPGAN + 原始帧检测 + CPU-GPU流水线
  · OOM 自动降级（SR batch 与 GFPGAN sub-batch 独立降级）
""",
    )

    # 基础参数
    parser.add_argument("--config", "-c", default=_default_cfg,
                        help=f"配置文件路径（默认: {_default_cfg}）")
    parser.add_argument("--input",  "-i", required=True,
                        help="输入视频路径")
    parser.add_argument("--output", "-o", required=True,
                        help="输出视频路径（含文件名）")

    # 覆盖配置的常用参数
    parser.add_argument("--upscale-factor", type=int, choices=[2, 4],
                        help="超分倍数（覆盖配置文件）")
    parser.add_argument("--model",
                        help="模型名称，如 realesr-general-x4v3 / RealESRGAN_x4plus（覆盖配置）")
    parser.add_argument("--segment-duration", type=int,
                        help="分段时长（秒，覆盖配置）")
    parser.add_argument("--batch-size", type=int,
                        help="SR 批处理大小（覆盖配置）")
    parser.add_argument("--tile-size", type=int,
                        help="tile 切块大小（0=不切块，覆盖配置）")

    # face_enhance 精细控制
    _fe = parser.add_mutually_exclusive_group()
    _fe.add_argument("--face-enhance",    dest="face_enhance", action="store_true",
                     default=None, help="开启人脸增强（GFPGAN）")
    _fe.add_argument("--no-face-enhance", dest="face_enhance", action="store_false",
                     help="关闭人脸增强")
    parser.add_argument("--gfpgan-model", choices=["1.3", "1.4", "RestoreFormer"],
                        help="GFPGAN 版本（覆盖配置，--face-enhance 时生效）")
    parser.add_argument("--gfpgan-weight", type=float,
                        help="GFPGAN 融合权重 0.0~1.0（覆盖配置）")
    parser.add_argument("--gfpgan-batch-size", type=int,
                        help="单次 GFPGAN 前向最多处理的人脸数（覆盖配置）")

    # 推理/编码开关
    parser.add_argument("--no-hwaccel", action="store_true",
                        help="禁用 NVDEC 硬件解码")
    parser.add_argument("--use-compile", action="store_true",
                        help="启用 torch.compile 加速")
    parser.add_argument("--use-tensorrt", action="store_true",
                        help="启用 TensorRT 加速")
    parser.add_argument("--fp32", action="store_true",
                        help="使用 FP32 精度（默认 FP16）")
    parser.add_argument("--crf", type=int,
                        help="分段输出视频质量 CRF（覆盖配置）")
    parser.add_argument("--report", metavar="PATH",
                        help="输出 JSON 性能报告路径")
    parser.add_argument("--auto-cleanup", action="store_true",
                        help="处理完成后自动清理临时文件")

    args = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────────────────────────────
    # 需要将 config_manager 所在目录加入路径
    sys.path.insert(0, str(_base_dir / "src" / "utils"))
    from config_manager import Config

    try:
        config = Config(args.config)
        print(f"⚙️  配置文件: {args.config}")
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        return 1

    # ── 命令行参数覆盖配置 ────────────────────────────────────────────────────
    config.set("paths", "input_video",  value=args.input)
    config.set("paths", "output_dir",   value=str(Path(args.output).parent))
    config.set("processing", "batch_mode", value=False)

    if args.upscale_factor:
        config.set("processing", "upscale_factor", value=args.upscale_factor)
    if args.model:
        config.set("models", "realesrgan", "model_name", value=args.model)
    if args.segment_duration:
        config.set("processing", "segment_duration", value=args.segment_duration)
    if args.batch_size:
        config.set("models", "realesrgan", "batch_size", value=args.batch_size)
    if args.tile_size is not None:
        config.set("models", "realesrgan", "tile_size", value=args.tile_size)

    # face_enhance 精细控制
    if args.face_enhance is not None:
        config.set("models", "realesrgan", "face_enhance", value=args.face_enhance)
    if args.gfpgan_model:
        config.set("models", "realesrgan", "gfpgan_model", value=args.gfpgan_model)
    if args.gfpgan_weight is not None:
        config.set("models", "realesrgan", "gfpgan_weight", value=args.gfpgan_weight)
    if args.gfpgan_batch_size:
        config.set("models", "realesrgan", "gfpgan_batch_size", value=args.gfpgan_batch_size)

    # 推理/编码开关
    if args.no_hwaccel:
        config.set("models", "realesrgan", "use_hwaccel", value=False)
    if args.use_compile:
        config.set("models", "realesrgan", "use_compile", value=True)
    if args.use_tensorrt:
        config.set("models", "realesrgan", "use_tensorrt", value=True)
    if args.fp32:
        config.set("models", "realesrgan", "fp32", value=True)
    if args.crf is not None:
        config.set("models", "realesrgan", "crf", value=args.crf)
    if args.report:
        config.set("models", "realesrgan", "report_json", value=args.report)
    if args.auto_cleanup:
        config.set("processing", "auto_cleanup_temp", value=True)

    # ── 创建处理器并执行 ──────────────────────────────────────────────────────
    try:
        processor = RealESRGANVideoProcessor(config)
    except Exception as e:
        print(f"❌ 初始化处理器失败: {e}")
        return 1

    face_on = config.get("models", "realesrgan", "face_enhance", default=False)
    print(f"\n🎨 Real-ESRGAN 独立超分")
    print(f"   输入  : {args.input}")
    print(f"   输出  : {args.output}")
    print(f"   模型  : {processor.model_name}")
    print(f"   倍数  : {processor.upscale_factor}x")
    print(f"   设备  : {processor.device}")
    print(f"   face_enhance: {face_on}" +
          (f" (model={processor.gfpgan_model}, weight={processor.gfpgan_weight},"
           f" batch={processor.gfpgan_batch_size})" if face_on else ""))

    success = processor.process_video(args.input, args.output)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

