"""
视频处理主程序
整合插帧和超分处理流程
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple

# 导入自定义模块
sys.path.insert(0, "/workspace/Video_Enhancement/src/utils")
from config_manager import Config
from video_utils import (
    VideoInfo, get_video_duration, format_time,
    extract_audio, smart_extract_audio, get_audio_codec, 
    add_audio_to_video, verify_video_integrity
)

sys.path.insert(0, "/workspace/Video_Enhancement/src/processors")
from ifrnet_processor import IFRNetProcessor
from realesrgan_processor_video import RealESRGANVideoProcessor

class VideoProcessor:
    """视频处理主类"""
    
    def __init__(self, config: Config):
        """
        初始化处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.mode = config.get("processing", "mode")

        self.base_dir = config.get("base_dir", "/workspace/Video_Enhancement")
        
        # 创建子处理器
        self.ifrnet = IFRNetProcessor(config)
        self.esrgan = RealESRGANVideoProcessor(config)
    
    def process_single_video(self, input_video: str) -> bool:
        """
        处理单个视频
        
        Args:
            input_video: 输入视频路径
        
        Returns:
            是否成功
        """
        print("\n" + "=" * 70)
        print(f"🎬 开始处理视频")
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
                # 使用智能提取功能
                audio_output_dir = str(self.config.get_temp_dir())
                audio_path = smart_extract_audio(input_video, audio_output_dir)
                if audio_path:
                    print(f"✅ 音频已保存: {audio_path}")
                else:
                    print("⚠️ 音频提取失败")
                    audio_path = None

            else:
                audio_path = str(self.config.get_temp_dir() / f"{Path(input_video).stem}_audio.{audio_format}")
                if extract_audio(input_video, audio_path, config=output_config):
                    print(f"✅ 音频已保存: {audio_path}")
                else:
                    audio_path = None

            
                
        # 根据模式选择处理顺序
        if self.mode == "interpolate_then_upscale":
            success, output_path = self._process_interpolate_then_upscale(input_video, audio_path)
        elif self.mode == "upscale_then_interpolate":
            success, output_path = self._process_upscale_then_interpolate(input_video, audio_path)
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
                video_info = None
            print("=" * 70 + "\n")
        else:
            print("\n" + "=" * 70)
            print("❌ 视频处理失败")
            print("=" * 70 + "\n")
        
        return success
    
    def _process_interpolate_then_upscale(self, input_video: str, 
                                         audio_path: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        先插帧再超分
        
        Args:
            input_video: 输入视频
            audio_path: 音频路径
        
        Returns:
            Tuple[是否成功, 输出视频路径]
        """
        print("\n" + "=" * 60)
        print("📋 处理流程: 先插帧 → 再超分")
        print("=" * 60)
        
        # 生成中间文件路径
        temp_dir = self.config.get_temp_dir()
        video_name = Path(input_video).stem
        
        interpolated_video = temp_dir / f"{video_name}_interpolated.mp4"
        upscaled_video = temp_dir / f"{video_name}_upscaled.mp4"
        
        # 步骤1: 插帧
        print("\n" + "-" * 60)
        print("步骤 1/2: 视频插帧")
        print("-" * 60)
        
        if not self.ifrnet.process_video(input_video, str(interpolated_video)):
            print("❌ 插帧处理失败")
            return False, None
        
        # 步骤2: 超分
        print("\n" + "-" * 60)
        print("步骤 2/2: 视频超分")
        print("-" * 60)
        
        if not self.esrgan.process_video(str(interpolated_video), str(upscaled_video)):
            print("❌ 超分处理失败")
            return False, None
        
        # 生成最终输出
        output_video = self.config.get_output_path(input_video, "_processed")
        
        # 添加音频
        if audio_path and os.path.exists(audio_path):
            print("\n🎵 添加音频到最终视频...")
            output_config = self.config.get_section("output", {})
            if add_audio_to_video(str(upscaled_video), audio_path, output_video, config=output_config):
                print(f"✅ 最终视频: {output_video}")
            else:
                print("⚠️  音频添加失败，使用无音频版本")
                import shutil
                shutil.copy2(str(upscaled_video), output_video)
        else:
            print("\n📹 保存最终视频...")
            import shutil
            shutil.copy2(str(upscaled_video), output_video)
            print(f"✅ 最终视频: {output_video}")
        
        # 清理临时文件
        if self.config.get("processing", "auto_cleanup_temp", default=False):
            self._cleanup_intermediate_files(interpolated_video, upscaled_video, audio_path)
        
        return True, str(output_video)
    
    def _process_upscale_then_interpolate(self, input_video: str, 
                                         audio_path: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        先超分再插帧
        
        Args:
            input_video: 输入视频
            audio_path: 音频路径
        
        Returns:
            Tuple[是否成功, 输出视频路径]
        """
        print("\n" + "=" * 60)
        print("📋 处理流程: 先超分 → 再插帧")
        print("=" * 60)
        
        # 生成中间文件路径
        temp_dir = self.config.get_temp_dir()
        video_name = Path(input_video).stem
        
        upscaled_video = temp_dir / f"{video_name}_upscaled.mp4"
        interpolated_video = temp_dir / f"{video_name}_interpolated.mp4"
        
        # 步骤1: 超分
        print("\n" + "-" * 60)
        print("步骤 1/2: 视频超分")
        print("-" * 60)
        
        if not self.esrgan.process_video(input_video, str(upscaled_video)):
            print("❌ 超分处理失败")
            return False, None
        
        # 步骤2: 插帧
        print("\n" + "-" * 60)
        print("步骤 2/2: 视频插帧")
        print("-" * 60)
        
        if not self.ifrnet.process_video(str(upscaled_video), str(interpolated_video)):
            print("❌ 插帧处理失败")
            return False, None
        
        # 生成最终输出
        output_video = self.config.get_output_path(input_video, "_processed")
        
        # 添加音频
        if audio_path and os.path.exists(audio_path):
            print("\n🎵 添加音频到最终视频...")
            output_config = self.config.get_section("output", {})
            if add_audio_to_video(str(interpolated_video), audio_path, output_video, config=output_config):
                print(f"✅ 最终视频: {output_video}")
            else:
                print("⚠️  音频添加失败，使用无音频版本")
                import shutil
                shutil.copy2(str(interpolated_video), output_video)
        else:
            print("\n📹 保存最终视频...")
            import shutil
            shutil.copy2(str(interpolated_video), output_video)
            print(f"✅ 最终视频: {output_video}")
        
        # 清理临时文件
        if self.config.get("processing", "auto_cleanup_temp", default=False):
            self._cleanup_intermediate_files(upscaled_video, interpolated_video, audio_path)
        
        return True, str(output_video)
    
    def _cleanup_intermediate_files(self, *files):
        """清理中间文件"""
        print("\n🧹 清理中间文件...")
        for file_path in files:
            if file_path and os.path.exists(str(file_path)):
                try:
                    os.remove(str(file_path))
                    print(f"✅ 已删除: {Path(file_path).name}")
                except Exception as e:
                    print(f"⚠️  删除失败 {Path(file_path).name}: {e}")
    
    def process_batch(self) -> bool:
        """
        批量处理视频
        
        Returns:
            是否全部成功
        """
        print("\n" + "=" * 70)
        print("📦 批量处理模式")
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


def main():
    """主函数"""
    # 获取当前文件所在目录的父目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_config = os.path.join(base_dir, "config", "default_config.json")

    parser = argparse.ArgumentParser(
        description="视频处理程序 - 插帧和超分",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置处理单个视频
  python main.py --input video.mp4
  
  # 使用自定义配置文件
  python main.py --config custom_config.json --input video.mp4
  
  # 批量处理目录中的所有视频
  python main.py --input-dir ./videos --batch
  
  # 使用先超分再插帧模式
  python main.py --input video.mp4 --mode upscale_then_interpolate
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
    
    # 修改点2: 显示实际使用的配置文件路径
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
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断处理")
        return 130
    except Exception as e:
        print(f"\n❌ 处理过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())