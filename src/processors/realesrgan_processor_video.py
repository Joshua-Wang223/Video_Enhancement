"""
Real-ESRGAN视频超分处理器（视频直接处理版本）
直接调用 inference_realesrgan_video.py 进行视频超分处理
"""

import os
import sys
import subprocess
import json
import time
import torch
from pathlib import Path
from typing import List, Optional
import shutil
import argparse

from video_utils import (
    get_video_duration, format_time, verify_video_integrity,
    split_video_by_time, merge_videos, merge_videos_by_codec
)


class RealESRGANVideoProcessor:
    """Real-ESRGAN视频超分处理器（直接处理视频版本）"""
    
    def __init__(self, config):
        """
        初始化处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.esrgan_dir = Path(config.get("paths", "base_dir")) / "external" / "Real-ESRGAN"
        self.model_name = config.get("models", "realesrgan", "model_name", default="RealESRGAN_x4plus")
        self.model_path = config.get("models", "realesrgan", "model_path")
        self.use_gpu = config.get("models", "realesrgan", "use_gpu", default=True)
        self.upscale_factor = config.get("processing", "upscale_factor", default=4)
        self.segment_duration = config.get("processing", "segment_duration", default=30)
        
        # Real-ESRGAN参数
        self.denoise_strength = config.get("models", "realesrgan", "denoise_strength", default=0.5)
        self.tile_size = config.get("models", "realesrgan", "tile_size", default=0)
        self.tile_pad = config.get("models", "realesrgan", "tile_pad", default=10)
        self.pre_pad = config.get("models", "realesrgan", "pre_pad", default=0)
        self.fp32 = config.get("models", "realesrgan", "fp32", default=False)
        self.face_enhance = config.get("models", "realesrgan", "face_enhance", default=False)
        self.num_process_per_gpu = config.get("models", "realesrgan", "num_process_per_gpu", default=1)
        
        # 验证Real-ESRGAN是否存在
        if not self.esrgan_dir.exists():
            raise FileNotFoundError(f"Real-ESRGAN目录不存在: {self.esrgan_dir}")
        
        # 添加Real-ESRGAN到Python路径
        sys.path.insert(0, str(self.esrgan_dir))
        
        # 检测设备
        self.device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
        
        self.checkpoint_file = None
        self.segment_dir = None
        self.processed_dir = None
    
    def _setup_temp_dirs(self, video_path: str):
        """设置临时目录"""
        video_name = Path(video_path).stem
        temp_base = self.config.get_temp_dir("esrgan_video") / video_name
        
        self.segment_dir = temp_base / "segments"
        self.processed_dir = temp_base / "processed"
        self.checkpoint_file = temp_base / "checkpoint.json"
        
        self.segment_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_checkpoint(self) -> dict:
        """加载断点信息"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {"processed_segments": [], "last_segment": -1}
    
    def _save_checkpoint(self, checkpoint: dict):
        """保存断点信息"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _run_esrgan_video(self, input_path: str, output_path: str, segment_idx: int = 0) -> bool:
        """
        调用 inference_realesrgan_video.py 处理视频
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            segment_idx: 片段索引（用于多进程）
        
        Returns:
            是否成功
        """
        try:
            # 准备参数
            args = argparse.Namespace()
            
            # 视频参数
            args.input = input_path
            args.output = str(Path(output_path).parent)
            
            # 模型参数
            args.model_name = self.model_name
            args.denoise_strength = self.denoise_strength
            args.outscale = self.upscale_factor
            args.suffix = f"processed_{segment_idx:03d}"
            
            # 处理参数
            args.tile = self.tile_size
            args.tile_pad = self.tile_pad
            args.pre_pad = self.pre_pad
            args.face_enhance = self.face_enhance
            args.fp32 = self.fp32
            args.fps = None  # 保持原帧率
            args.ffmpeg_bin = "ffmpeg"
            args.extract_frame_first = False  # 不先提取帧
            args.num_process_per_gpu = self.num_process_per_gpu
            
            # 其他参数
            args.alpha_upsampler = "realesrgan"
            args.ext = "auto"
            
            # 设置视频名称（用于临时文件）
            video_name = Path(input_path).stem
            args.video_name = f"{video_name}_{segment_idx:03d}"
            
            # 确保输出目录存在
            os.makedirs(args.output, exist_ok=True)
            
            # 导入并运行 inference_realesrgan_video
            try:
                from inference_realesrgan_video import run
                
                print(f"   运行Real-ESRGAN视频超分...")
                print(f"   加载模型权重: {self.model_name}")
                start_time = time.time()
                
                # 直接调用run函数
                run(args)
                
                # 检查输出文件
                expected_output = os.path.join(args.output, f"{args.video_name}_{args.suffix}.mp4")
                if os.path.exists(expected_output):
                    # 重命名到指定输出路径
                    shutil.move(expected_output, output_path)
                    
                    elapsed = time.time() - start_time
                    print(f"   ✅ 处理完成 ({format_time(elapsed)})")
                    return True
                else:
                    print(f"   ❌ 输出文件未生成")
                    return False
                    
            except ImportError as e:
                print(f"   ❌ 无法导入inference_realesrgan_video: {e}")
                return False
            except Exception as e:
                print(f"   ❌ 处理失败: {e}")
                import traceback
                traceback.print_exc()
                return False
                
        except Exception as e:
            print(f"   ❌ 运行Real-ESRGAN失败: {e}")
            return False
    
    def _process_segment(self, segment_path: str, output_path: str, segment_idx: int) -> bool:
        """
        处理单个视频片段
        
        Args:
            segment_path: 输入片段路径
            output_path: 输出路径
            segment_idx: 片段索引
        
        Returns:
            是否成功
        """
        try:
            print(f"   🎬 处理片段 {segment_idx+1}: {Path(segment_path).name}")
            
            # 获取视频信息
            duration = get_video_duration(segment_path)
            if duration:
                print(f"   📊 片段时长: {format_time(duration)}")
            
            # 调用Real-ESRGAN视频处理
            success = self._run_esrgan_video(segment_path, output_path, segment_idx)
            
            if success:
                # 验证输出
                if verify_video_integrity(output_path):
                    out_duration = get_video_duration(output_path)
                    print(f"   ✅ 处理完成: {format_time(out_duration)}")
                    return True
                else:
                    print(f"   ❌ 输出文件验证失败")
                    return False
            else:
                return False
                
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_video(self, input_video: str, output_video: str) -> bool:
        """
        处理完整视频（支持分段和断点恢复）
        
        Args:
            input_video: 输入视频路径
            output_video: 输出视频路径
        
        Returns:
            是否成功
        """
        print("\n" + "=" * 60)
        print(f"🎨 Real-ESRGAN 视频超分处理 (直接处理模式)")
        print(f"📹 输入: {input_video}")
        print(f"📤 输出: {output_video}")
        print(f"⚡ 超分倍数: {self.upscale_factor}x")
        print(f"🖥️  设备: {self.device}")
        print(f"🧩 分段时长: {self.segment_duration}秒")
        print("=" * 60 + "\n")
        
        # 记录开始时间
        total_start_time = time.time()  # 添加这行
        
        # 设置临时目录
        self._setup_temp_dirs(input_video)
        
        # 加载断点
        checkpoint = self._load_checkpoint()
        
        # 获取视频时长
        duration = get_video_duration(input_video)
        if duration is None:
            print("❌ 无法获取视频时长")
            return False
        
        print(f"📊 视频信息:")
        print(f"   - 时长: {format_time(duration)}")
        print(f"   - 分段时长: {self.segment_duration}秒")
        
        # 如果视频较短，直接处理整个视频
        if duration <= self.segment_duration:
            print(f"\n📦 视频较短，直接处理整个视频...")
            
            success = self._run_esrgan_video(input_video, output_video, 0)
            
            if success:
                # 清理断点文件
                if self.checkpoint_file.exists():
                    self.checkpoint_file.unlink()
                
                total_time = time.time() - total_start_time  # 修改这行
                print(f"\n✅ 超分处理完成！")
                print(f"⏱️  总用时: {format_time(total_time)}")
                print(f"📤 输出: {output_video}")
                
                # 清理临时文件（如果配置允许）
                if self.config.get("processing", "auto_cleanup_temp", default=False):
                    self._cleanup_temp_files()
                
                return True
            else:
                print("\n❌ 视频处理失败")
                return False
        
        # 分割视频
        print(f"\n🔪 分割视频...")
        segment_files = split_video_by_time(
            input_video, 
            str(self.segment_dir), 
            self.segment_duration
        )
        
        if not segment_files:
            print("❌ 视频分割失败")
            return False
        
        print(f"✅ 共 {len(segment_files)} 个片段")
        
        # 处理每个片段
        print(f"\n⚙️  开始处理片段...")
        processed_files = []
        segment_start_time = time.time()  # 添加这行
        
        for i, segment_file in enumerate(segment_files):
            segment_name = Path(segment_file).name
            
            # 检查是否已处理
            if i in checkpoint["processed_segments"]:
                print(f"\n⏭️  片段 {i+1}/{len(segment_files)}: {segment_name} (已处理)")
                output_file = self.processed_dir / f"processed_{segment_name}"
                if output_file.exists():
                    processed_files.append(str(output_file))
                    continue
            
            print(f"\n🎨 片段 {i+1}/{len(segment_files)}: {segment_name}")
            
            # 处理片段
            output_file = self.processed_dir / f"processed_{segment_name}"
            success = self._process_segment(segment_file, str(output_file), i)
            
            if success:
                processed_files.append(str(output_file))
                checkpoint["processed_segments"].append(i)
                checkpoint["last_segment"] = i
                self._save_checkpoint(checkpoint)
            else:
                print(f"⚠️  片段 {i+1} 处理失败，跳过")
                continue
            
            # 估算剩余时间
            elapsed = time.time() - segment_start_time  # 修改这行
            completed = len(checkpoint["processed_segments"])
            if completed > 0:
                avg_time = elapsed / completed
                remaining = (len(segment_files) - completed) * avg_time
                print(f"   ⏱️  已用时: {format_time(elapsed)}, 预计剩余: {format_time(remaining)}")
        
        if not processed_files:
            print("\n❌ 没有成功处理的片段")
            return False
        
        # 合并处理后的片段
        print(f"\n🔗 合并处理后的片段...")
        # success = merge_videos(processed_files, output_video)
        # output_config = self.config.get("output", default={})
        output_config = self.config.get_section("output", {})
        # success = merge_videos(processed_files, output_video, config=output_config)
        success = merge_videos_by_codec(processed_files, output_video, config=output_config)
        
        if success:
            # 清理断点文件
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            
            total_time = time.time() - total_start_time  # 修改这行
            print(f"\n✅ 超分处理完成！")
            print(f"⏱️  总用时: {format_time(total_time)}")
            print(f"📤 输出: {output_video}")
            
            # 清理临时文件（如果配置允许）
            if self.config.get("processing", "auto_cleanup_temp", default=False):
                self._cleanup_temp_files()
            
            return True
        else:
            print("\n❌ 视频合并失败")
            return False
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
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


if __name__ == "__main__":
    print("Real-ESRGAN视频超分处理器模块 (直接处理视频版本)")
    print("✅ 模块加载成功，请在主程序中调用。")