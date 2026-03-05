"""
视频处理主程序 v5（单卡版）
=====================================
整合插帧（IFRNet v5）和超分（Real-ESRGAN v6）处理流程。
采用分段直接对接，避免中间合并，支持断点恢复。

对接模块：
  - ifrnet_processor_v5_single.IFRNetProcessor
    → 底层: external/IFRNet/process_video_v5_single.py
  - realesrgan_processor_video_v5_single.RealESRGANVideoProcessor
    → 底层: external/Real-ESRGAN/inference_realesrgan_video_v6_single.py
"""

import os
import sys
import time
import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple, List

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
    VideoInfo, get_video_duration, format_time,
    extract_audio, smart_extract_audio, add_audio_to_video,
    verify_video_integrity, merge_videos_by_codec
)

# 导入 v5 单卡版处理器
from ifrnet_processor_v5_single import IFRNetProcessor
from realesrgan_processor_video_v5_single import RealESRGANVideoProcessor


class VideoProcessor:
    """视频处理主类 v5（单卡版）——分段直接对接，避免中间合并"""

    def __init__(self, config: Config):
        """
        初始化处理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.mode   = config.get("processing", "mode")

        # 创建 v5 单卡版子处理器
        self.ifrnet = IFRNetProcessor(config)
        self.esrgan = RealESRGANVideoProcessor(config)

        # 临时文件追踪（用于最终清理）
        self.temp_files_tracker: dict = {
            "audio_files":       [],
            "segment_dirs":      [],
            "intermediate_dirs": [],
            "checkpoint_files":  [],
        }

    # -------------------------------------------------------------------------
    # 单视频处理
    # -------------------------------------------------------------------------

    def process_single_video(self, input_video: str) -> bool:
        """
        处理单个视频。

        Args:
            input_video: 输入视频路径

        Returns:
            是否成功
        """
        print("\n" + "=" * 70)
        print("🎬 开始处理视频（v5 单卡版）")
        print(f"📹 输入: {input_video}")
        print(f"⚙️  模式: {self.mode}")
        print("=" * 70 + "\n")

        start_time = time.time()

        if not os.path.exists(input_video):
            print(f"❌ 输入文件不存在: {input_video}")
            return False

        # 打印视频信息
        video_info = self._print_video_info(input_video)

        # 提取音频
        audio_path = self._extract_audio(input_video, video_info)

        # 根据模式分发处理
        if self.mode == "interpolate_then_upscale":
            success, output_path = self._process_interpolate_then_upscale(
                input_video, audio_path
            )
        elif self.mode == "upscale_then_interpolate":
            success, output_path = self._process_upscale_then_interpolate(
                input_video, audio_path
            )
        else:
            print(f"❌ 未知的处理模式: {self.mode}")
            return False

        if success:
            elapsed = time.time() - start_time
            print("\n" + "=" * 70)
            print("✅ 视频处理完成！")
            print(f"⏱️  总用时: {format_time(elapsed)}")
            self._print_video_info(output_path, label="最终")
            print("=" * 70 + "\n")
        else:
            print("\n" + "=" * 70)
            print("❌ 视频处理失败")
            print("=" * 70 + "\n")

        return success

    # -------------------------------------------------------------------------
    # 处理流程（插帧 → 超分）
    # -------------------------------------------------------------------------

    def _process_interpolate_then_upscale(
        self,
        input_video: str,
        audio_path: Optional[str],
    ) -> Tuple[bool, Optional[str]]:
        """
        先插帧再超分（分段直接对接，避免中间合并）。

        Args:
            input_video: 输入视频路径
            audio_path:  提取的音频路径（可为 None）

        Returns:
            (成功标志, 输出视频路径或 None)
        """
        print("\n" + "=" * 60)
        print("📋 处理流程: 先插帧 → 再超分（v5 单卡版）")
        print("💡 优化策略: 分段直接对接，避免中间合并")
        print("=" * 60)

        video_name = Path(input_video).stem

        # 步骤 1：插帧（分段模式）
        print("\n" + "-" * 60)
        print("步骤 1/3: 视频插帧处理（分段模式）")
        print("-" * 60)

        interpolated_segments = self.ifrnet.process_video_segments(input_video)
        if not interpolated_segments:
            print("❌ 插帧处理失败")
            return False, None

        print(f"✅ 插帧完成，生成 {len(interpolated_segments)} 个分段")
        self.temp_files_tracker["segment_dirs"].append(self.ifrnet.segment_dir)
        self.temp_files_tracker["checkpoint_files"].append(self.ifrnet.checkpoint_file)

        # 步骤 2：超分（直接接收插帧分段）
        print("\n" + "-" * 60)
        print("步骤 2/3: 视频超分处理（接收分段输入）")
        print("-" * 60)

        upscaled_segments = self.esrgan.process_segments_directly(
            interpolated_segments, video_name
        )
        if not upscaled_segments:
            print("❌ 超分处理失败")
            return False, None

        print(f"✅ 超分完成，生成 {len(upscaled_segments)} 个分段")
        self.temp_files_tracker["segment_dirs"].append(self.esrgan.processed_dir)
        self.temp_files_tracker["checkpoint_files"].append(self.esrgan.checkpoint_file)

        # 步骤 3：合并
        print("\n" + "-" * 60)
        print("步骤 3/3: 合并最终视频")
        print("-" * 60)

        final_output  = self.config.get_output_path(input_video, "_processed")
        output_config = self.config.get_section("output", {})

        print(f"🔗 合并 {len(upscaled_segments)} 个处理后的分段...")
        if not merge_videos_by_codec(upscaled_segments, final_output,
                                     audio_path, config=output_config):
            print("❌ 视频合并失败")
            return False, None

        print(f"✅ 视频合并完成: {final_output}")
        self._cleanup_checkpoints()
        return True, final_output

    # -------------------------------------------------------------------------
    # 处理流程（超分 → 插帧）
    # -------------------------------------------------------------------------

    def _process_upscale_then_interpolate(
        self,
        input_video: str,
        audio_path: Optional[str],
    ) -> Tuple[bool, Optional[str]]:
        """
        先超分再插帧（分段直接对接，避免中间合并）。

        Args:
            input_video: 输入视频路径
            audio_path:  提取的音频路径（可为 None）

        Returns:
            (成功标志, 输出视频路径或 None)
        """
        print("\n" + "=" * 60)
        print("📋 处理流程: 先超分 → 再插帧（v5 单卡版）")
        print("💡 优化策略: 分段直接对接，避免中间合并")
        print("=" * 60)

        video_name = Path(input_video).stem

        # 步骤 1：超分（分段模式）
        print("\n" + "-" * 60)
        print("步骤 1/3: 视频超分处理（分段模式）")
        print("-" * 60)

        upscaled_segments = self.esrgan.process_video_segments(input_video)
        if not upscaled_segments:
            print("❌ 超分处理失败")
            return False, None

        print(f"✅ 超分完成，生成 {len(upscaled_segments)} 个分段")
        self.temp_files_tracker["segment_dirs"].append(self.esrgan.segment_dir)
        self.temp_files_tracker["checkpoint_files"].append(self.esrgan.checkpoint_file)

        # 步骤 2：插帧（直接接收超分分段）
        print("\n" + "-" * 60)
        print("步骤 2/3: 视频插帧处理（接收分段输入）")
        print("-" * 60)

        interpolated_segments = self.ifrnet.process_segments_directly(
            upscaled_segments, video_name
        )
        if not interpolated_segments:
            print("❌ 插帧处理失败")
            return False, None

        print(f"✅ 插帧完成，生成 {len(interpolated_segments)} 个分段")
        self.temp_files_tracker["segment_dirs"].append(self.ifrnet.processed_dir)
        self.temp_files_tracker["checkpoint_files"].append(self.ifrnet.checkpoint_file)

        # 步骤 3：合并
        print("\n" + "-" * 60)
        print("步骤 3/3: 合并最终视频")
        print("-" * 60)

        final_output  = self.config.get_output_path(input_video, "_processed")
        output_config = self.config.get_section("output", {})

        print(f"🔗 合并 {len(interpolated_segments)} 个处理后的分段...")
        if not merge_videos_by_codec(interpolated_segments, final_output,
                                     audio_path, config=output_config):
            print("❌ 视频合并失败")
            return False, None

        print(f"✅ 视频合并完成: {final_output}")
        self._cleanup_checkpoints()
        return True, final_output

    # -------------------------------------------------------------------------
    # 批量处理
    # -------------------------------------------------------------------------

    def process_batch(self) -> bool:
        """
        批量处理视频目录中的所有视频。

        Returns:
            是否全部成功
        """
        print("\n" + "=" * 70)
        print("📦 批量处理模式（v5 单卡版）")
        print("=" * 70)

        videos = self.config.get_input_videos()
        if not videos:
            print("❌ 未找到任何视频文件")
            return False

        print(f"\n📋 找到 {len(videos)} 个视频文件:")
        for i, video in enumerate(videos, 1):
            print(f"   {i}. {Path(video).name}")

        success_count = 0
        failed_videos: List[str] = []

        for i, video in enumerate(videos, 1):
            print(f"\n{'=' * 70}")
            print(f"处理进度: {i}/{len(videos)}")
            print(f"{'=' * 70}")
            try:
                if self.process_single_video(video):
                    success_count += 1
                else:
                    failed_videos.append(video)
            except Exception as e:
                print(f"❌ 处理失败: {e}")
                failed_videos.append(video)

        print("\n" + "=" * 70)
        print("📊 批量处理总结")
        print("=" * 70)
        print(f"✅ 成功: {success_count}/{len(videos)}")
        print(f"❌ 失败: {len(failed_videos)}/{len(videos)}")
        if failed_videos:
            print("\n失败的视频:")
            for video in failed_videos:
                print(f"   - {Path(video).name}")
        print("=" * 70 + "\n")

        return len(failed_videos) == 0

    # -------------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------------

    def _print_video_info(self, video_path: str,
                          label: str = "输入") -> Optional["VideoInfo"]:
        """打印视频基本信息，返回 VideoInfo 对象（获取失败时返回 None）。"""
        try:
            info = VideoInfo(video_path)
            print(f"📊 {label}视频信息:")
            print(f"   - 分辨率: {info.width}x{info.height}")
            print(f"   - 视频编码: {info.codec}")
            print(f"   - 帧率: {info.fps:.2f} fps")
            print(f"   - 时长: {format_time(info.duration)}")
            print(f"   - 总帧数: {info.frame_count}")
            print(f"   - 音频: {'有' if info.has_audio else '无'}")
            if info.has_audio:
                print(f"   - 音频编码: {info.audio_codec}")
            return info
        except Exception as e:
            print(f"⚠️  获取{label}视频信息失败: {e}")
            return None

    def _extract_audio(self, input_video: str,
                       video_info: Optional["VideoInfo"]) -> Optional[str]:
        """
        提取音频轨道并记录路径；无音频或提取失败时返回 None。

        Args:
            input_video: 输入视频路径
            video_info:  视频信息对象（可为 None）

        Returns:
            音频文件路径或 None
        """
        if not (video_info and video_info.has_audio):
            return None

        print(f"\n🎵 提取音频...")
        output_config = self.config.get_section("output", {})
        # "smart" = 保留原始编码（推荐）；其他值视为目标格式扩展名
        audio_format  = output_config.get("audio_format", "smart")

        if audio_format == "smart":
            audio_path = smart_extract_audio(
                input_video, str(self.config.get_temp_dir())
            )
        else:
            audio_path = str(
                self.config.get_temp_dir()
                / f"{Path(input_video).stem}_audio.{audio_format}"
            )
            if not extract_audio(input_video, audio_path, config=output_config):
                audio_path = None

        if audio_path:
            print(f"✅ 音频已保存: {audio_path}")
            self.temp_files_tracker["audio_files"].append(audio_path)
        else:
            print("⚠️  音频提取失败")

        return audio_path

    def _cleanup_checkpoints(self):
        """处理成功后删除断点文件。"""
        for checkpoint_file in self.temp_files_tracker["checkpoint_files"]:
            if checkpoint_file and Path(checkpoint_file).exists():
                try:
                    Path(checkpoint_file).unlink()
                except Exception:
                    pass

    def prompt_cleanup(self):
        """处理完成后询问（或自动）清理临时文件。"""
        if not self.config.get("processing", "auto_cleanup_temp", default=False):
            print("\n" + "=" * 70)
            print("🧹 临时文件清理")
            print("=" * 70)

            total_dirs  = len(self.temp_files_tracker["segment_dirs"])
            total_audio = len(self.temp_files_tracker["audio_files"])

            if total_dirs == 0 and total_audio == 0:
                print("✅ 没有需要清理的临时文件")
                return

            print(f"\n📁 临时目录: {total_dirs} 个")
            print(f"🎵 音频文件: {total_audio} 个")

            response = input("\n是否清理所有临时文件? (Y/n，默认 Y): ").strip().lower()
            if response == "n":
                print("⏭️  保留临时文件")
                return

            self._execute_cleanup()
        else:
            print("\n🧹 自动清理临时文件...")
            self._execute_cleanup()

    def _execute_cleanup(self):
        """执行实际清理操作。"""
        cleaned_dirs  = 0
        cleaned_files = 0

        for dir_path in self.temp_files_tracker["segment_dirs"]:
            if dir_path and Path(dir_path).exists():
                try:
                    shutil.rmtree(dir_path)
                    cleaned_dirs += 1
                except Exception as e:
                    print(f"⚠️  清理目录失败 {dir_path}: {e}")

        for audio_file in self.temp_files_tracker["audio_files"]:
            if audio_file and Path(audio_file).exists():
                try:
                    Path(audio_file).unlink()
                    cleaned_files += 1
                except Exception as e:
                    print(f"⚠️  清理文件失败 {audio_file}: {e}")

        for checkpoint_file in self.temp_files_tracker["checkpoint_files"]:
            if checkpoint_file and Path(checkpoint_file).exists():
                try:
                    Path(checkpoint_file).unlink()
                except Exception:
                    pass

        if cleaned_dirs > 0 or cleaned_files > 0:
            print(f"✅ 清理完成: {cleaned_dirs} 个目录, {cleaned_files} 个文件")
        else:
            print("✅ 清理完成")


# =============================================================================
# 命令行入口
# =============================================================================

def main():
    """
    主函数 —— 完整流程入口（v5 单卡版）

    调用链：
      main_video_v5_single.py
        ├── IFRNetProcessor          (src/processors/ifrnet_processor_v5_single.py)
        │     └── IFRNetVideoProcessor  (external/IFRNet/process_video_v5_single.py)
        └── RealESRGANVideoProcessor  (src/processors/realesrgan_processor_video_v5_single.py)
              └── run()               (external/Real-ESRGAN/inference_realesrgan_video_v6_single.py)
    """
    default_config = str(_BASE_DIR / "config" / "default_config.json")

    parser = argparse.ArgumentParser(
        description="视频处理程序 v5（单卡版）——分段直接对接，避免中间合并",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
特性（v5 单卡版）:
  · IFRNet v5 插帧 / Real-ESRGAN v6 超分，分段直接对接，省去中间合并
  · NVDEC/NVENC 硬件编解码自动探测
  · FP16 / torch.compile / CUDA Graph / TensorRT 可选加速
  · OOM 自动降级（batch_size 减半持久化）
  · 断点续传（segment 级别 checkpoint.json）
  · face_enhance：批量 GFPGAN 推理 + 原始帧检测 + CPU-GPU 流水线（v6 新增）

示例:
  # 单视频，默认配置
  python main_video_v5_single.py -i video.mp4

  # 先超分再插帧，4x+4x
  python main_video_v5_single.py -i video.mp4 -m upscale_then_interpolate \\
         --interpolation-factor 4 --upscale-factor 4

  # 指定 IFRNet 模型名称（L 版质量更高）
  python main_video_v5_single.py -i video.mp4 --ifrnet-model IFRNet_L_Vimeo90K

  # 直接指定 IFRNet .pth 路径（覆盖自动拼接）
  python main_video_v5_single.py -i video.mp4 \\
         --ifrnet-model-path /workspace/models/IFRNet_S_Vimeo90K.pth

  # 同时调整 IFRNet 和 ESRGAN 批大小（根据显存调优）
  python main_video_v5_single.py -i video.mp4 \\
         --ifrnet-batch-size 8 --esrgan-batch-size 16

  # 开启人脸增强（使用 GFPGAN 1.4，融合权重 0.5）
  python main_video_v5_single.py -i video.mp4 --face-enhance

  # 人脸增强精细控制
  python main_video_v5_single.py -i video.mp4 --face-enhance \\
         --gfpgan-model 1.4 --gfpgan-weight 0.7 --gfpgan-batch-size 4

  # 批量处理目录
  python main_video_v5_single.py --input-dir ./videos --batch

  # 指定配置文件 + 自定义输出目录
  python main_video_v5_single.py -c my_config.json -i video.mp4 -o /data/output
""",
    )

    # ── 基础参数 ──────────────────────────────────────────────────────────────
    parser.add_argument("--config", "-c", type=str, default=default_config,
                        help=f"配置文件路径（默认: {default_config}）")
    parser.add_argument("--input", "-i", type=str,
                        help="输入视频文件路径（单视频模式）")
    parser.add_argument("--input-dir", type=str,
                        help="输入视频目录（批量处理，自动启用 --batch）")
    parser.add_argument("--output-dir", "-o", type=str,
                        help="输出目录（覆盖配置文件）")
    parser.add_argument("--mode", "-m", type=str,
                        choices=["interpolate_then_upscale", "upscale_then_interpolate"],
                        help="处理模式（覆盖配置文件）")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="批量处理模式（处理 --input-dir 下所有视频）")

    # ── 倍数参数 ──────────────────────────────────────────────────────────────
    parser.add_argument("--interpolation-factor", type=int, choices=[2, 4, 8, 16],
                        help="插帧倍数，覆盖配置（默认 2）")
    parser.add_argument("--upscale-factor", type=int, choices=[2, 4],
                        help="超分倍数，覆盖配置（默认 2）")
    parser.add_argument("--segment-duration", type=int,
                        help="视频分段时长（秒），覆盖配置（默认 30）")

    # ── Real-ESRGAN 模型参数 ───────────────────────────────────────────────────
    parser.add_argument("--esrgan-model", type=str,
                        metavar="MODEL_NAME",
                        help="Real-ESRGAN 模型名称，如 realesr-general-x4v3 / "
                             "RealESRGAN_x4plus / realesr-animevideov3（覆盖配置）")
    parser.add_argument("--tile-size", type=int,
                        help="Real-ESRGAN tile 切块大小（0=不切块，显存不足时设 512，覆盖配置）")
    parser.add_argument("--esrgan-batch-size", type=int, metavar="N",
                        help="Real-ESRGAN 超分推理批大小（覆盖配置，T4/16G 建议 8~12）")

    # ── IFRNet 模型参数 ────────────────────────────────────────────────────────
    parser.add_argument("--ifrnet-model", metavar="MODEL_NAME",
                        choices=["IFRNet_Vimeo90K", "IFRNet_S_Vimeo90K", "IFRNet_L_Vimeo90K"],
                        help="IFRNet 模型名称（覆盖配置；processor 自动在 "
                             "models_IFRNet/checkpoints/ 下查找对应 .pth）：\n"
                             "  IFRNet_S_Vimeo90K（默认，轻量快速）\n"
                             "  IFRNet_L_Vimeo90K（高质量，速度更慢）")
    parser.add_argument("--ifrnet-model-path", metavar="PATH",
                        help="IFRNet .pth 权重文件绝对路径（优先级高于 --ifrnet-model，覆盖配置）")
    parser.add_argument("--ifrnet-batch-size", type=int, metavar="N",
                        help="IFRNet 插帧推理批大小（覆盖配置，默认 4）")

    # ── face_enhance 人脸增强参数 ─────────────────────────────────────────────
    _fe = parser.add_mutually_exclusive_group()
    _fe.add_argument("--face-enhance", dest="face_enhance", action="store_true",
                     default=None,
                     help="开启 GFPGAN 人脸增强（anime 系模型自动禁用）")
    _fe.add_argument("--no-face-enhance", dest="face_enhance", action="store_false",
                     help="关闭人脸增强（覆盖配置中 face_enhance=true）")
    parser.add_argument("--gfpgan-model", choices=["1.3", "1.4", "RestoreFormer"],
                        help="GFPGAN 版本（默认 1.4）；--face-enhance 时生效")
    parser.add_argument("--gfpgan-weight", type=float, metavar="W",
                        help="GFPGAN 融合权重 0.0~1.0（0=不增强，1=完全替换，默认 0.5）")
    parser.add_argument("--gfpgan-batch-size", type=int, metavar="N",
                        help="单次 GFPGAN 前向最多处理的人脸数，防 OOM（默认 8）")

    # ── 硬件加速 / 精度开关 ────────────────────────────────────────────────────
    parser.add_argument("--no-hwaccel", action="store_true",
                        help="禁用 NVDEC 硬件解码（IFRNet + ESRGAN 同时生效）")
    parser.add_argument("--no-fp16", action="store_true",
                        help="禁用 FP16 半精度推理（IFRNet），改用 FP32")
    parser.add_argument("--no-compile", action="store_true",
                        help="禁用 torch.compile（IFRNet），适合短视频跳过预热")
    parser.add_argument("--no-cuda-graph", action="store_true",
                        help="禁用 CUDA Graph（IFRNet）")

    # ── 其他行为参数 ──────────────────────────────────────────────────────────
    parser.add_argument("--auto-cleanup", action="store_true",
                        help="处理完成后自动清理临时文件，不再询问")

    args = parser.parse_args()

    print(f"⚙️  使用配置文件: {args.config}")

    # ── 加载配置 ──────────────────────────────────────────────────────────────
    try:
        config = Config(args.config)
        print(f"✅ 配置文件已加载: {args.config}")
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        print(f"💡 请确保配置文件存在: {args.config}")
        return 1

    # ── 命令行参数覆盖配置 ────────────────────────────────────────────────────

    # 输入 / 输出 / 模式
    if args.input:
        config.set("paths", "input_video", value=args.input)
        config.set("processing", "batch_mode", value=False)
    if args.input_dir:
        config.set("paths", "input_dir", value=args.input_dir)
        config.set("processing", "batch_mode", value=True)
    if args.batch:
        config.set("processing", "batch_mode", value=True)
    if args.output_dir:
        config.set("paths", "output_dir", value=args.output_dir)
    if args.mode:
        config.set("processing", "mode", value=args.mode)

    # 倍数 / 分段
    if args.interpolation_factor:
        config.set("processing", "interpolation_factor", value=args.interpolation_factor)
    if args.upscale_factor:
        config.set("processing", "upscale_factor", value=args.upscale_factor)
    if args.segment_duration:
        config.set("processing", "segment_duration", value=args.segment_duration)

    # Real-ESRGAN 模型
    if args.esrgan_model:
        config.set("models", "realesrgan", "model_name", value=args.esrgan_model)
    if args.tile_size is not None:
        config.set("models", "realesrgan", "tile_size", value=args.tile_size)
    if args.esrgan_batch_size:
        config.set("models", "realesrgan", "batch_size", value=args.esrgan_batch_size)

    # IFRNet 模型（model_path 优先级高于 model_name）
    if args.ifrnet_model_path:
        config.set("models", "ifrnet", "model_path", value=args.ifrnet_model_path)
        config.set("models", "ifrnet", "model_name", value="")  # 禁用 name 拼接
    elif args.ifrnet_model:
        config.set("models", "ifrnet", "model_name", value=args.ifrnet_model)
        config.set("models", "ifrnet", "model_path", value="")  # 让 processor 按 name 拼接
    if args.ifrnet_batch_size:
        config.set("models", "ifrnet", "batch_size", value=args.ifrnet_batch_size)

    # face_enhance 精细控制
    if args.face_enhance is not None:
        config.set("models", "realesrgan", "face_enhance", value=args.face_enhance)
    if args.gfpgan_model:
        config.set("models", "realesrgan", "gfpgan_model", value=args.gfpgan_model)
    if args.gfpgan_weight is not None:
        config.set("models", "realesrgan", "gfpgan_weight", value=args.gfpgan_weight)
    if args.gfpgan_batch_size:
        config.set("models", "realesrgan", "gfpgan_batch_size", value=args.gfpgan_batch_size)

    # 硬件加速 / 精度
    if args.no_hwaccel:
        config.set("models", "ifrnet",     "use_hwaccel", value=False)
        config.set("models", "realesrgan", "use_hwaccel", value=False)
    if args.no_fp16:
        config.set("models", "ifrnet", "use_fp16", value=False)
    if args.no_compile:
        config.set("models", "ifrnet", "use_compile", value=False)
    if args.no_cuda_graph:
        config.set("models", "ifrnet", "use_cuda_graph", value=False)

    # 其他行为
    if args.auto_cleanup:
        config.set("processing", "auto_cleanup_temp", value=True)

    # ── 打印当前关键参数摘要 ───────────────────────────────────────────────────
    face_on = config.get("models", "realesrgan", "face_enhance", default=False)
    print(f"\n📋 处理参数摘要:")
    print(f"   模式          : {config.get('processing', 'mode')}")
    print(f"   插帧倍数      : {config.get('processing', 'interpolation_factor')}x")
    print(f"   超分倍数      : {config.get('processing', 'upscale_factor')}x")
    # IFRNet 模型信息
    _ifrnet_path = config.get("models", "ifrnet", "model_path", default="")
    _ifrnet_name = config.get("models", "ifrnet", "model_name", default="IFRNet_S_Vimeo90K")
    if _ifrnet_path:
        print(f"   IFRNet 模型   : (显式路径) {_ifrnet_path}")
    else:
        print(f"   IFRNet 模型   : {_ifrnet_name}")
    print(f"   IFRNet batch  : {config.get('models', 'ifrnet', 'batch_size', default=4)}")
    print(f"   ESRGAN 模型   : {config.get('models', 'realesrgan', 'model_name')}")
    print(f"   ESRGAN batch  : {config.get('models', 'realesrgan', 'batch_size', default=8)}")
    print(f"   face_enhance  : {face_on}", end="")
    if face_on:
        print(f" (GFPGAN {config.get('models','realesrgan','gfpgan_model')},"
              f" weight={config.get('models','realesrgan','gfpgan_weight')},"
              f" batch={config.get('models','realesrgan','gfpgan_batch_size')})")
    else:
        print()
    print(f"   批量模式      : {config.get('processing', 'batch_mode')}")
    print()

    # ── 创建处理器 ────────────────────────────────────────────────────────────
    try:
        processor = VideoProcessor(config)
    except Exception as e:
        print(f"❌ 创建处理器失败: {e}")
        return 1

    # ── 执行处理 ──────────────────────────────────────────────────────────────
    try:
        if config.get("processing", "batch_mode"):
            success = processor.process_batch()
        else:
            videos = config.get_input_videos()
            if not videos:
                print("❌ 未指定输入视频，请使用 -i <video> 或 --input-dir <dir> --batch")
                return 1
            success = processor.process_single_video(videos[0])

        processor.prompt_cleanup()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断处理")
        processor.prompt_cleanup()
        return 130
    except Exception as e:
        print(f"\n❌ 处理过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
