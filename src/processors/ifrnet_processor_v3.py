"""
IFRNet视频插帧处理器 v2.0
优化版本：支持分段直接对接，完善断点恢复
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Optional

from video_utils import (
    get_video_duration, format_time, verify_video_integrity,
    split_video_by_time
)


class IFRNetProcessor:
    """IFRNet插帧处理器 v2.0"""
    
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
        self.batch_size = config.get("models", "ifrnet", "batch_size", default=4)
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
    
    def process_video_segments(self, input_video: str) -> List[str]:
        """
        处理视频并返回处理后的分段列表（不合并）
        
        Args:
            input_video: 输入视频路径
        
        Returns:
            处理后的分段文件列表
        """
        print(f"\n🎬 IFRNet 插帧处理 (分段模式)")
        print(f"📹 输入: {input_video}")
        print(f"⚡ 插帧倍数: {self.interpolation_factor}x")
        
        # 设置临时目录
        video_name = Path(input_video).stem
        self._setup_temp_dirs(video_name, "ifrnet_source")
        
        # 加载断点
        checkpoint = self._load_checkpoint()
        
        # 获取视频时长
        duration = get_video_duration(input_video)
        if duration is None:
            print("❌ 无法获取视频时长")
            return []
        
        print(f"📊 时长: {format_time(duration)}, 分段: {self.segment_duration}秒")
        
        # 分割视频
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
        
        # 处理每个片段
        return self._process_segments(segment_files, checkpoint)
    
    def process_segments_directly(self, input_segments: List[str], 
                                  video_name: str) -> List[str]:
        """
        直接处理已有的分段（用于对接上一步处理）
        
        Args:
            input_segments: 输入分段列表
            video_name: 视频名称（用于命名）
        
        Returns:
            处理后的分段文件列表
        """
        print(f"\n🎬 IFRNet 插帧处理 (接收分段输入)")
        print(f"📹 输入分段数: {len(input_segments)}")
        print(f"⚡ 插帧倍数: {self.interpolation_factor}x")
        
        # 设置临时目录
        self._setup_temp_dirs(video_name, "ifrnet_from_segments")
        
        # 加载断点
        checkpoint = self._load_checkpoint()
        
        # 直接处理分段
        return self._process_segments(input_segments, checkpoint)
    
    def _setup_temp_dirs(self, video_name: str, prefix: str):
        """设置临时目录"""
        temp_base = self.config.get_temp_dir("ifrnet") / f"{prefix}_{video_name}"
        
        self.segment_dir = temp_base / "segments"
        self.processed_dir = temp_base / "processed"
        self.checkpoint_file = temp_base / "checkpoint.json"
        
        self.segment_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_checkpoint(self) -> dict:
        """加载断点信息"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                print(f"📌 发现断点: 已完成 {len(checkpoint['processed_segments'])} 个分段")
                return checkpoint
            except:
                pass
        return {"processed_segments": [], "last_segment": -1}
    
    def _save_checkpoint(self, checkpoint: dict):
        """保存断点信息"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _process_segments(self, segment_files: List[str], 
                         checkpoint: dict) -> List[str]:
        """
        处理分段列表
        
        Args:
            segment_files: 分段文件列表
            checkpoint: 断点信息
        
        Returns:
            处理后的分段列表
        """
        print(f"\n⚙️  开始处理片段...")
        processed_files = []
        start_time = time.time()
        
        for i, segment_file in enumerate(segment_files):
            segment_name = Path(segment_file).name
            
            # 检查是否已处理
            if i in checkpoint["processed_segments"]:
                print(f"\n⏭️  片段 {i+1}/{len(segment_files)}: {segment_name} (已处理)")
                output_file = self.processed_dir / f"interpolated_{segment_name}"
                if output_file.exists():
                    processed_files.append(str(output_file))
                    continue
            
            print(f"\n🎬 片段 {i+1}/{len(segment_files)}: {segment_name}")
            
            # 处理片段
            output_file = self.processed_dir / f"interpolated_{segment_name}"
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
        
        if processed_files:
            print(f"\n✅ IFRNet处理完成: {len(processed_files)}/{len(segment_files)} 个分段")
        else:
            print(f"\n❌ 没有成功处理的片段")
        
        return processed_files
    
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
            from process_video_v3 import IFRNetVideoProcessor

            # 根据use_gpu设置设备
            device = 'cuda' if self.use_gpu else 'cpu'
            
            # 创建处理器
            processor = IFRNetVideoProcessor(
                model_path=self.model_path,
                # use_gpu=self.use_gpu
                device=device,
                batch_size=args.batch_size
            )
            
            # 计算输出帧数
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


if __name__ == "__main__":
    print("IFRNet插帧处理器模块 v2.0")
