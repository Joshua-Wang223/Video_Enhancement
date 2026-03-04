"""
IFRNet视频插帧处理器
负责视频帧插值处理
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Optional
import shutil

from video_utils import (
    get_video_duration, format_time, verify_video_integrity,
    split_video_by_time, merge_videos, merge_videos_by_codec
)


class IFRNetProcessor:
    """IFRNet插帧处理器"""
    
    def __init__(self, config):
        """
        初始化处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.ifrnet_dir = Path(config.get("paths", "base_dir")) / "external" / "IFRNet"
        self.model_path = config.get("models", "ifrnet", "model_path")
        self.use_gpu = config.get("models", "ifrnet", "use_gpu", default=True)
        self.interpolation_factor = config.get("processing", "interpolation_factor", default=2)
        self.segment_duration = config.get("processing", "segment_duration", default=30)
        
        # 验证IFRNet是否存在
        if not self.ifrnet_dir.exists():
            raise FileNotFoundError(f"IFRNet目录不存在: {self.ifrnet_dir}")
        
        # 添加IFRNet到Python路径
        sys.path.insert(0, str(self.ifrnet_dir))
        
        self.checkpoint_file = None
        self.segment_dir = None
        self.processed_dir = None
    
    def _setup_temp_dirs(self, video_path: str):
        """设置临时目录"""
        video_name = Path(video_path).stem
        temp_base = self.config.get_temp_dir("ifrnet") / video_name
        
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
    
    def _process_segment(self, segment_path: str, output_path: str) -> bool:
        """
        处理单个视频片段
        
        Args:
            segment_path: 输入片段路径
            output_path: 输出路径
        
        Returns:
            是否成功
        """
        try:
            # 导入IFRNet处理器
            from process_video import IFRNetVideoProcessor

            # 根据use_gpu设置设备
            device = 'cuda' if self.use_gpu else 'cpu'
            
            # 创建处理器
            processor = IFRNetVideoProcessor(
                model_path=self.model_path,
                # use_gpu=self.use_gpu
                device=device  # 使用device参数而不是use_gpu
            )
            
            # 计算输出帧数
            # # 对于2倍插帧：每两帧之间插入1帧
            # # 对于4倍插帧：每两帧之间插入3帧
            # times = self.interpolation_factor - 1
            scale = self.interpolation_factor
            
            print(f"   🎬 处理片段: {Path(segment_path).name}")
            # print(f"   📊 插帧倍数: {self.interpolation_factor}x (每两帧间插入{times}帧)")
            print(f"   📊 插帧倍数: {self.interpolation_factor}")
            
            # 处理视频
            processor.process_video(
                # input_video=segment_path,
                input_path=segment_path,
                # output_video=output_path,
                output_path=output_path,
                # times=times
                scale=scale
            )
            
            # 验证输出
            if verify_video_integrity(output_path):
                seg_duration = get_video_duration(output_path)
                print(f"   ✅ 处理完成: {format_time(seg_duration)}")
                return True
            else:
                print(f"   ❌ 输出文件验证失败")
                return False
                
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
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
        print(f"🎬 IFRNet 视频插帧处理")
        print(f"📹 输入: {input_video}")
        print(f"📤 输出: {output_video}")
        print(f"⚡ 插帧倍数: {self.interpolation_factor}x")
        print("=" * 60 + "\n")
        
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
        start_time = time.time()
        
        for i, segment_file in enumerate(segment_files):
            segment_name = Path(segment_file).name
            
            # 检查是否已处理
            if i in checkpoint["processed_segments"]:
                print(f"\n⏭️  片段 {i+1}/{len(segment_files)}: {segment_name} (已处理)")
                output_file = self.processed_dir / f"processed_{segment_name}"
                if output_file.exists():
                    processed_files.append(str(output_file))
                    continue
            
            print(f"\n🎬 片段 {i+1}/{len(segment_files)}: {segment_name}")
            
            # 处理片段
            output_file = self.processed_dir / f"processed_{segment_name}"
            success = self._process_segment(segment_file, str(output_file))
            
            if success:
                processed_files.append(str(output_file))
                checkpoint["processed_segments"].append(i)
                checkpoint["last_segment"] = i
                self._save_checkpoint(checkpoint)
            else:
                print(f"⚠️  片段 {i+1} 处理失败，跳过")
                continue
            
            # 估算剩余时间
            elapsed = time.time() - start_time
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
            
            total_time = time.time() - start_time
            print(f"\n✅ 插帧处理完成！")
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
    print("IFRNet插帧处理器模块")
    print("✅ 模块加载成功，请在主程序中调用。")