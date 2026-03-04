"""
视频处理主程序 v2.0（优化版）
整合插帧和超分处理流程，采用分段直接对接，避免中间合并
"""

import os
import sys
import time
import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple, List

# 导入自定义模块
sys.path.insert(0, "/workspace/Video_Enhancement/src/utils")
from config_manager import Config
from video_utils import (
    VideoInfo, get_video_duration, format_time,
    extract_audio, smart_extract_audio, add_audio_to_video,
    verify_video_integrity, merge_videos_by_codec
)

# 导入 v2 版本的处理器
sys.path.insert(0, "/workspace/Video_Enhancement/src/processors")
from ifrnet_processor_v2 import IFRNetProcessor
from realesrgan_processor_video_v2 import RealESRGANVideoProcessor


class VideoProcessor:
    """视频处理主类 v2.0 - 优化版（分段直接对接）"""

    def __init__(self, config: Config):
        """
        初始化处理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.mode = config.get("processing", "mode")

        # 创建子处理器（v2版本）
        self.ifrnet = IFRNetProcessor(config)
        self.esrgan = RealESRGANVideoProcessor(config)

        # 临时文件追踪（用于最后清理）
        self.temp_files_tracker = {
            'audio_files': [],
            'segment_dirs': [],
            'intermediate_dirs': [],
            'checkpoint_files': []
        }

    def process_single_video(self, input_video: str) -> bool:
        """
        处理单个视频

        Args:
            input_video: 输入视频路径

        Returns:
            是否成功
        """
        print("\n" + "=" * 70)
        print(f"🎬 开始处理视频 (v2.0 优化版)")
        print(f"📹 输入: {input_video}")
        print(f"⚙️  模式: {self.mode}")
        print("=" * 70 + "\n")

        start_time = time.time()

        # 验证输入文件
        if not os.path.exists(input_video):
            print(f"❌ 输入文件不存在: {input_video}")
            return False

        # 获取视频信息
        try:
            video_info = VideoInfo(input_video)
            print(f"📊 视频信息:")
            print(f"   - 分辨率: {video_info.width}x{video_info.height}")
            print(f"   - 视频编码: {video_info.codec}")
            print(f"   - 帧率: {video_info.fps:.2f} fps")
            print(f"   - 时长: {format_time(video_info.duration)}")
            print(f"   - 总帧数: {video_info.frame_count}")
            print(f"   - 音频: {'有' if video_info.has_audio else '无'}")
            if video_info.has_audio:
                print(f"   - 音频编码: {video_info.audio_codec}")
        except Exception as e:
            print(f"⚠️  获取视频信息失败: {e}")
            video_info = None

        # 提取音频（如果有）
        audio_path = None
        if video_info and video_info.has_audio:
            print(f"\n🎵 提取音频...")
            output_config = self.config.get_section("output", {})
            audio_format = output_config.get("audio_format", "aac")
            if audio_format == "smart":
                audio_output_dir = str(self.config.get_temp_dir())
                audio_path = smart_extract_audio(input_video, audio_output_dir)
                if audio_path:
                    print(f"✅ 音频已保存: {audio_path}")
                    self.temp_files_tracker['audio_files'].append(audio_path)
                else:
                    print("⚠️ 音频提取失败")
                    audio_path = None
            else:
                audio_path = str(self.config.get_temp_dir() / f"{Path(input_video).stem}_audio.{audio_format}")
                if extract_audio(input_video, audio_path, config=output_config):
                    print(f"✅ 音频已保存: {audio_path}")
                    self.temp_files_tracker['audio_files'].append(audio_path)
                else:
                    print("⚠️ 音频提取失败")
                    audio_path = None

        # 根据模式选择处理顺序
        if self.mode == "interpolate_then_upscale":
            success, output_path = self._process_interpolate_then_upscale_optimized(input_video, audio_path)
        elif self.mode == "upscale_then_interpolate":
            success, output_path = self._process_upscale_then_interpolate_optimized(input_video, audio_path)
        else:
            print(f"❌ 未知的处理模式: {self.mode}")
            return False

        if success:
            elapsed_time = time.time() - start_time
            print("\n" + "=" * 70)
            print("✅ 视频处理完成！")
            print(f"⏱️  总用时: {format_time(elapsed_time)}")

            # 获取处理完成后的视频信息
            try:
                video_info = VideoInfo(output_path)
                print(f"📊 最终视频信息:")
                print(f"   - 分辨率: {video_info.width}x{video_info.height}")
                print(f"   - 视频编码: {video_info.codec}")
                print(f"   - 帧率: {video_info.fps:.2f} fps")
                print(f"   - 时长: {format_time(video_info.duration)}")
                print(f"   - 总帧数: {video_info.frame_count}")
                print(f"   - 音频: {'有' if video_info.has_audio else '无'}")
                if video_info.has_audio:
                    print(f"   - 音频编码: {video_info.audio_codec}")
            except Exception as e:
                print(f"⚠️  获取最终视频信息失败: {e}")
            print("=" * 70 + "\n")
        else:
            print("\n" + "=" * 70)
            print("❌ 视频处理失败")
            print("=" * 70 + "\n")

        return success

    def _process_interpolate_then_upscale_optimized(self, input_video: str,
                                                    audio_path: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        先插帧再超分（优化版：分段直接对接）

        Args:
            input_video: 输入视频
            audio_path: 音频路径

        Returns:
            (成功标志, 输出视频路径或None)
        """
        print("\n" + "=" * 60)
        print("📋 处理流程: 先插帧 → 再超分 (优化版)")
        print("💡 优化策略: 分段直接对接，避免中间合并")
        print("=" * 60)

        video_name = Path(input_video).stem

        # 步骤1: 插帧处理（分段模式）
        print("\n" + "-" * 60)
        print("步骤 1/2: 视频插帧处理（分段模式）")
        print("-" * 60)

        interpolated_segments = self.ifrnet.process_video_segments(input_video)

        if not interpolated_segments:
            print("❌ 插帧处理失败")
            return False, None

        print(f"✅ 插帧完成，生成 {len(interpolated_segments)} 个分段")

        # 追踪临时目录
        self.temp_files_tracker['segment_dirs'].append(self.ifrnet.segment_dir)
        self.temp_files_tracker['checkpoint_files'].append(self.ifrnet.checkpoint_file)

        # 步骤2: 直接对插帧后的分段进行超分（接收分段输入）
        print("\n" + "-" * 60)
        print("步骤 2/2: 视频超分处理（接收分段输入）")
        print("-" * 60)

        final_output = self.config.get_output_path(input_video, "_processed")

        upscaled_segments = self.esrgan.process_segments_directly(
            interpolated_segments,
            video_name
        )

        if not upscaled_segments:
            print("❌ 超分处理失败")
            return False, None

        print(f"✅ 超分完成，生成 {len(upscaled_segments)} 个分段")

        # 追踪临时目录
        self.temp_files_tracker['segment_dirs'].append(self.esrgan.processed_dir)
        self.temp_files_tracker['checkpoint_files'].append(self.esrgan.checkpoint_file)

        # 步骤3: 只在最后合并一次
        print("\n" + "-" * 60)
        print("步骤 3/3: 合并最终视频")
        print("-" * 60)

        print(f"🔗 合并 {len(upscaled_segments)} 个处理后的分段...")
        output_config = self.config.get_section("output", {})

        if not merge_videos_by_codec(upscaled_segments, final_output, audio_path, config=output_config):
            print("❌ 视频合并失败")
            return False, None

        print(f"✅ 视频合并完成: {final_output}")

        # 清理断点文件
        self._cleanup_checkpoints()

        return True, final_output

    def _process_upscale_then_interpolate_optimized(self, input_video: str,
                                                audio_path: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        先超分再插帧（优化版：分段直接对接）

        Args:
            input_video: 输入视频
            audio_path: 音频路径

        Returns:
            (成功标志, 输出视频路径或None)
        """
        print("\n" + "=" * 60)
        print("📋 处理流程: 先超分 → 再插帧 (优化版)")
        print("💡 优化策略: 分段直接对接，避免中间合并")
        print("=" * 60)

        video_name = Path(input_video).stem

        # 步骤1: 超分处理（分段模式）
        print("\n" + "-" * 60)
        print("步骤 1/2: 视频超分处理（分段模式）")
        print("-" * 60)

        upscaled_segments = self.esrgan.process_video_segments(input_video)

        if not upscaled_segments:
            print("❌ 超分处理失败")
            return False, None

        print(f"✅ 超分完成，生成 {len(upscaled_segments)} 个分段")

        # 追踪临时目录
        self.temp_files_tracker['segment_dirs'].append(self.esrgan.segment_dir)
        self.temp_files_tracker['checkpoint_files'].append(self.esrgan.checkpoint_file)

        # 步骤2: 直接对超分后的分段进行插帧（接收分段输入）
        print("\n" + "-" * 60)
        print("步骤 2/2: 视频插帧处理（接收分段输入）")
        print("-" * 60)

        final_output = self.config.get_output_path(input_video, "_processed")

        interpolated_segments = self.ifrnet.process_segments_directly(
            upscaled_segments,
            video_name
        )

        if not interpolated_segments:
            print("❌ 插帧处理失败")
            return False, None

        print(f"✅ 插帧完成，生成 {len(interpolated_segments)} 个分段")

        # 追踪临时目录
        self.temp_files_tracker['segment_dirs'].append(self.ifrnet.processed_dir)
        self.temp_files_tracker['checkpoint_files'].append(self.ifrnet.checkpoint_file)

        # 步骤3: 只在最后合并一次
        print("\n" + "-" * 60)
        print("步骤 3/3: 合并最终视频")
        print("-" * 60)

        print(f"🔗 合并 {len(interpolated_segments)} 个处理后的分段...")
        output_config = self.config.get_section("output", {})

        if not merge_videos_by_codec(interpolated_segments, final_output, audio_path, config=output_config):
            print("❌ 视频合并失败")
            return False, None

        print(f"✅ 视频合并完成: {final_output}")

        # 清理断点文件
        self._cleanup_checkpoints()

        return True, final_output

    def _cleanup_checkpoints(self):
        """清理断点文件（处理成功后）"""
        for checkpoint_file in self.temp_files_tracker['checkpoint_files']:
            if checkpoint_file and Path(checkpoint_file).exists():
                try:
                    Path(checkpoint_file).unlink()
                except:
                    pass

    def process_batch(self) -> bool:
        """
        批量处理视频

        Returns:
            是否全部成功
        """
        print("\n" + "=" * 70)
        print("📦 批量处理模式 (v2.0 优化版)")
        print("=" * 70)

        # 获取所有视频
        videos = self.config.get_input_videos()

        if not videos:
            print("❌ 未找到任何视频文件")
            return False

        print(f"\n📋 找到 {len(videos)} 个视频文件:")
        for i, video in enumerate(videos, 1):
            print(f"   {i}. {Path(video).name}")

        # 处理每个视频
        success_count = 0
        failed_videos = []

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

        # 总结
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

    def prompt_cleanup(self):
        """
        处理完成后询问是否清理临时文件
        """
        if not self.config.get("processing", "auto_cleanup_temp", default=False):
            print("\n" + "=" * 70)
            print("🧹 临时文件清理")
            print("=" * 70)

            total_dirs = len(self.temp_files_tracker['segment_dirs'])
            total_audio = len(self.temp_files_tracker['audio_files'])

            if total_dirs == 0 and total_audio == 0:
                print("✅ 没有需要清理的临时文件")
                return

            print(f"\n📁 临时目录: {total_dirs} 个")
            print(f"🎵 音频文件: {total_audio} 个")

            response = input("\n是否清理所有临时文件? (Y/n，默认Y): ").strip().lower()

            if response == 'n':
                print("⏭️  保留临时文件")
                return

            self._execute_cleanup()
        else:
            print("\n🧹 自动清理临时文件...")
            self._execute_cleanup()

    def _execute_cleanup(self):
        """执行实际的清理操作"""
        cleaned_dirs = 0
        cleaned_files = 0

        # 清理目录
        for dir_path in self.temp_files_tracker['segment_dirs']:
            if dir_path and Path(dir_path).exists():
                try:
                    shutil.rmtree(dir_path)
                    cleaned_dirs += 1
                except Exception as e:
                    print(f"⚠️  清理目录失败 {dir_path}: {e}")

        # 清理音频文件
        for audio_file in self.temp_files_tracker['audio_files']:
            if audio_file and Path(audio_file).exists():
                try:
                    Path(audio_file).unlink()
                    cleaned_files += 1
                except Exception as e:
                    print(f"⚠️  清理文件失败 {audio_file}: {e}")

        # 清理断点文件
        for checkpoint_file in self.temp_files_tracker['checkpoint_files']:
            if checkpoint_file and Path(checkpoint_file).exists():
                try:
                    Path(checkpoint_file).unlink()
                except:
                    pass

        if cleaned_dirs > 0 or cleaned_files > 0:
            print(f"✅ 清理完成: {cleaned_dirs} 个目录, {cleaned_files} 个文件")
        else:
            print("✅ 清理完成")


def main():
    """主函数"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_config = os.path.join(base_dir, "config", "default_config.json")

    parser = argparse.ArgumentParser(
        description="视频处理程序 v2.0 - 优化版（分段直接对接）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    优化功能:
      - 分段直接对接，避免中间合并
      - 超分和插帧均支持断点恢复
      - 处理完成后统一询问清理

    示例:
      python main_video_v2.py -c config.json -i video.mp4
      python main_video_v2.py -c config.json --input-dir ./videos --batch
      python main_video_v2.py -i video.mp4 --mode upscale_then_interpolate --interpolation-factor 4 --upscale-factor 4
    """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=default_config,
        help=f"配置文件路径 (默认: {default_config})"
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        help="输入视频文件路径"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        help="输入视频目录（批量处理）"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="输出目录（覆盖配置文件中的设置）"
    )

    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["interpolate_then_upscale", "upscale_then_interpolate"],
        help="处理模式（覆盖配置文件中的设置）"
    )

    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="批量处理模式"
    )

    parser.add_argument(
        "--interpolation-factor",
        type=int,
        choices=[2, 4, 8, 16],
        help="插帧倍数"
    )

    parser.add_argument(
        "--upscale-factor",
        type=int,
        choices=[2, 4],
        help="超分倍数"
    )

    args = parser.parse_args()

    print(f"⚙️  使用配置文件: {args.config}")

    # 加载配置
    try:
        config = Config(args.config)
        print(f"✅ 配置文件已加载: {args.config}")
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        print(f"💡 请确保配置文件存在: {args.config}")
        return 1

    # 覆盖配置参数
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

    if args.interpolation_factor:
        config.set("processing", "interpolation_factor", value=args.interpolation_factor)

    if args.upscale_factor:
        config.set("processing", "upscale_factor", value=args.upscale_factor)

    # 创建处理器
    try:
        processor = VideoProcessor(config)
    except Exception as e:
        print(f"❌ 创建处理器失败: {e}")
        return 1

    # 执行处理
    try:
        if config.get("processing", "batch_mode"):
            success = processor.process_batch()
        else:
            videos = config.get_input_videos()
            if not videos:
                print("❌ 未指定输入视频")
                return 1
            success = processor.process_single_video(videos[0])

        # 处理完成后询问清理
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