"""
Real-ESRGAN视频超分处理器 v2.0
优化版本：支持分段直接对接，完善断点恢复
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

from video_utils import (
    get_video_duration, format_time, verify_video_integrity,
    split_video_by_time
)
from output_filter import filter_tile_output

sys.path.insert(0, "/workspace/Video_Enhancement/external/Real-ESRGAN")
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


class RealESRGANProcessor:
    """Real-ESRGAN超分处理器 v2.0"""
    
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
    
    def process_video_segments(self, input_video: str) -> List[str]:
        """
        处理视频并返回处理后的分段列表（不合并）
        
        Args:
            input_video: 输入视频路径
        
        Returns:
            处理后的分段文件列表
        """
        print(f"\n🎨 Real-ESRGAN 超分处理 (分段模式)")
        print(f"📹 输入: {input_video}")
        print(f"⚡ 超分倍数: {self.upscale_factor}x")
        print(f"🖥️  设备: {self.device}")
        
        # 设置临时目录
        video_name = Path(input_video).stem
        self._setup_temp_dirs(video_name, "esrgan_source")
        
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
        print(f"\n🎨 Real-ESRGAN 超分处理 (接收分段输入)")
        print(f"📹 输入分段数: {len(input_segments)}")
        print(f"⚡ 超分倍数: {self.upscale_factor}x")
        print(f"🖥️  设备: {self.device}")
        
        # 设置临时目录
        self._setup_temp_dirs(video_name, "esrgan_from_segments")
        
        # 加载断点
        checkpoint = self._load_checkpoint()
        
        # 直接处理分段
        return self._process_segments(input_segments, checkpoint)
    
    def _setup_temp_dirs(self, video_name: str, prefix: str):
        """设置临时目录"""
        temp_base = self.config.get_temp_dir("esrgan") / f"{prefix}_{video_name}"
        
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
                output_file = self.processed_dir / f"upscaled_{segment_name}"
                if output_file.exists():
                    processed_files.append(str(output_file))
                    continue
            
            print(f"\n🎨 片段 {i+1}/{len(segment_files)}: {segment_name}")
            
            # 处理片段
            output_file = self.processed_dir / f"upscaled_{segment_name}"
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
            print(f"\n✅ Real-ESRGAN处理完成: {len(processed_files)}/{len(segment_files)} 个分段")
        else:
            print(f"\n❌ 没有成功处理的片段")
        
        return processed_files
    
    def _extract_frames(self, video_path: str, output_dir: str) -> bool:
        """提取视频帧"""
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            cmd = [
                'ffmpeg', '-i', video_path,
                '-qscale:v', '1',  # 高质量
                '-qmin', '1',
                '-qmax', '1',
                '-vsync', '0',  # 不丢帧
                os.path.join(output_dir, 'frame_%08d.png')
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 统计帧数
            frame_count = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
            print(f"   ✅ 提取 {frame_count} 帧")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ 帧提取失败: {e}")
            return False
    
    def _upscale_frames(self, input_dir: str, output_dir: str) -> bool:
        """
        超分视频帧
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
        
        Returns:
            是否成功
        """
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 导入Real-ESRGAN推理脚本
            from inference_realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            print(f"   🔧 加载模型: {self.model_name}")
            
            # 创建模型
            if 'x4' in self.model_name:
                num_blocks = 23
                netscale = 4
            elif 'x2' in self.model_name:
                num_blocks = 23
                netscale = 2
            else:
                num_blocks = 6
                netscale = 4
            
            # model = RRDBNet(
            #     num_in_ch=3,
            #     num_out_ch=3,
            #     num_feat=64,
            #     num_block=num_blocks,
            #     num_grow_ch=32,
            #     scale=netscale
            # )
            # 加载精简版模型(适应RealESRGANv2-animevideo-xsx2.pth)
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=16,  # animevideo-xsx2 通常是16层
                upscale=2,
                act_type='prelu'
            )
            
            # 创建upsampler
            upsampler = RealESRGANer(
                scale=netscale,
                model_path=self.model_path,
                model=model,
                tile=self.tile_size,
                tile_pad=self.tile_pad,
                pre_pad=self.pre_pad,
                half=not self.fp32,
                device=self.device
            )
            
            # 获取所有帧
            frames = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
            total_frames = len(frames)
            
            print(f"   🎨 处理 {total_frames} 帧...")
            
            # 使用过滤器屏蔽Tile输出
            with filter_tile_output():
                for i, frame_name in enumerate(frames):
                    if (i + 1) % 100 == 0:
                        print(f"   进度: {i+1}/{total_frames}")
                    
                    input_path = os.path.join(input_dir, frame_name)
                    output_path = os.path.join(output_dir, frame_name)
                    
                    # 读取图像
                    import cv2
                    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
                    
                    # 超分
                    output, _ = upsampler.enhance(img, outscale=self.upscale_factor)
                    
                    # 保存
                    cv2.imwrite(output_path, output)
            
            print(f"   ✅ 超分完成")
            return True
            
        except Exception as e:
            print(f"   ❌ 超分失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _frames_to_video(self, frames_dir: str, output_video: str, fps: float) -> bool:
        """
        将帧合成为视频
        
        Args:
            frames_dir: 帧目录
            output_video: 输出视频
            fps: 帧率
        
        Returns:
            是否成功
        """
        try:
            cmd = [
                'ffmpeg',
                '-framerate', str(fps),
                '-i', os.path.join(frames_dir, 'frame_%08d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '18',
                '-y', output_video
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"   ✅ 视频合成完成")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ 视频合成失败: {e}")
            return False
    
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
            print(f"   🎬 处理片段: {Path(segment_path).name}")
            
            # 获取视频信息
            import cv2
            cap = cv2.VideoCapture(segment_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # 创建临时目录
            temp_dir = Path(output_path).parent / f"temp_{Path(segment_path).stem}"
            input_frames = temp_dir / "input_frames"
            output_frames = temp_dir / "output_frames"
            
            # 提取帧
            print(f"   📸 提取帧...")
            if not self._extract_frames(segment_path, str(input_frames)):
                return False
            
            # 超分
            print(f"   🎨 超分处理 (倍数: {self.upscale_factor}x)...")
            if not self._upscale_frames(str(input_frames), str(output_frames)):
                return False
            
            # 合成视频
            print(f"   🎞️  合成视频...")
            if not self._frames_to_video(str(output_frames), output_path, fps):
                return False
            
            # 清理临时帧
            shutil.rmtree(temp_dir)
            
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
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    print("Real-ESRGAN超分处理器模块 v2.0")
