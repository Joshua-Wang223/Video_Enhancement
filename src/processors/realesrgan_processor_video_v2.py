"""
Real-ESRGAN视频超分处理器 v2.0（视频直接处理版本）
优化版本：支持分段直接输出、断点恢复完善、兼容多种模型
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

from video_utils import (
    get_video_duration, format_time, verify_video_integrity,
    split_video_by_time, merge_videos_by_codec
)


class RealESRGANVideoProcessor:
    """Real-ESRGAN视频超分处理器 v2.0（直接处理视频版本）"""

    def __init__(self, config):
        """
        初始化处理器

        Args:
            config: 配置对象（应包含 paths, models.realesrgan, processing 等节）
        """
        self.config = config
        self.esrgan_dir = Path(config.get("paths", "base_dir")) / "external" / "Real-ESRGAN"
        self.model_name = config.get("models", "realesrgan", "model_name", default="RealESRGAN_x4plus")
        self.model_path = config.get("models", "realesrgan", "model_path")
        self.use_gpu = config.get("models", "realesrgan", "use_gpu", default=True)
        self.upscale_factor = config.get("processing", "upscale_factor", default=4)
        self.segment_duration = config.get("processing", "segment_duration", default=30)

        # Real-ESRGAN 特有参数（用于传递给 inference_realesrgan_video）
        self.denoise_strength = config.get("models", "realesrgan", "denoise_strength", default=0.5)
        self.tile_size = config.get("models", "realesrgan", "tile_size", default=0)
        self.tile_pad = config.get("models", "realesrgan", "tile_pad", default=10)
        self.pre_pad = config.get("models", "realesrgan", "pre_pad", default=0)
        self.fp32 = config.get("models", "realesrgan", "fp32", default=False)
        self.face_enhance = config.get("models", "realesrgan", "face_enhance", default=False)
        self.num_process_per_gpu = config.get("models", "realesrgan", "num_process_per_gpu", default=1)

        # 验证 Real-ESRGAN 目录
        if not self.esrgan_dir.exists():
            raise FileNotFoundError(f"Real-ESRGAN目录不存在: {self.esrgan_dir}")

        # 添加 Real-ESRGAN 到 Python 路径（以便导入 inference_realesrgan_video）
        sys.path.insert(0, str(self.esrgan_dir))

        # 设备检测
        self.device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"

        # 以下属性在设置临时目录时初始化
        self.checkpoint_file = None
        self.segment_dir = None
        self.processed_dir = None

    # ----------------------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------------------
    def process_video(self, input_video: str, output_video: str) -> bool:
        """
        完整处理视频（分段 -> 超分 -> 合并），支持断点恢复

        Args:
            input_video: 输入视频路径
            output_video: 最终输出视频路径

        Returns:
            是否成功
        """
        print("\n" + "=" * 60)
        print(f"🎨 Real-ESRGAN 视频超分处理 (完整流程)")
        print(f"📹 输入: {input_video}")
        print(f"📤 输出: {output_video}")
        print(f"⚡ 超分倍数: {self.upscale_factor}x")
        print(f"🖥️  设备: {self.device}")
        print(f"🧩 分段时长: {self.segment_duration}秒")
        print("=" * 60 + "\n")

        total_start = time.time()

        # 1. 获取分段列表（并处理）
        processed_segments = self.process_video_segments(input_video)
        if not processed_segments:
            print("❌ 未成功处理任何分段")
            return False

        # 2. 合并分段
        print(f"\n🔗 合并 {len(processed_segments)} 个处理后的分段...")
        output_config = self.config.get_section("output", {})
        success = merge_videos_by_codec(processed_segments, output_video, config=output_config)

        if success:
            total_time = time.time() - total_start
            print(f"\n✅ 超分处理完成！总用时: {format_time(total_time)}")
            print(f"📤 输出: {output_video}")

            # 清理临时文件（可选）
            if self.config.get("processing", "auto_cleanup_temp", default=False):
                self._cleanup_temp_files()
            return True
        else:
            print("❌ 视频合并失败")
            return False

    def process_video_segments(self, input_video: str) -> List[str]:
        """
        处理视频，返回超分后的分段文件列表（不合并）

        Args:
            input_video: 输入视频路径

        Returns:
            处理后的分段文件路径列表
        """
        print(f"\n🎨 Real-ESRGAN 超分处理 (分段模式)")
        print(f"📹 输入: {input_video}")
        print(f"⚡ 超分倍数: {self.upscale_factor}x")
        print(f"🖥️  设备: {self.device}")

        # 设置临时目录（使用视频名和固定前缀）
        video_name = Path(input_video).stem
        self._setup_temp_dirs(video_name, prefix="esrgan_video")

        # 加载断点
        checkpoint = self._load_checkpoint()

        # 获取视频时长
        duration = get_video_duration(input_video)
        if duration is None:
            print("❌ 无法获取视频时长")
            return []
        print(f"📊 时长: {format_time(duration)}, 分段: {self.segment_duration}秒")

        # 如果视频短于分段时长，视为一个分段直接处理
        if duration <= self.segment_duration:
            print("📦 视频较短，直接处理整个视频...")
            output_file = self.processed_dir / f"upscaled_{Path(input_video).name}"
            success = self._process_segment(input_video, str(output_file), segment_idx=0)
            return [str(output_file)] if success else []

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

        # 处理所有分段
        return self._process_segments(segment_files, checkpoint)

    def process_segments_directly(self, input_segments: List[str], video_name: str) -> List[str]:
        """
        直接处理已有的分段视频（用于对接上游分段结果）

        Args:
            input_segments: 输入分段文件路径列表
            video_name: 视频名称（用于临时目录命名）

        Returns:
            处理后的分段文件路径列表
        """
        print(f"\n🎨 Real-ESRGAN 超分处理 (接收分段输入)")
        print(f"📹 输入分段数: {len(input_segments)}")
        print(f"⚡ 超分倍数: {self.upscale_factor}x")
        print(f"🖥️  设备: {self.device}")

        # 设置临时目录
        self._setup_temp_dirs(video_name, prefix="esrgan_from_segments")

        # 加载断点
        checkpoint = self._load_checkpoint()

        return self._process_segments(input_segments, checkpoint)

    # ----------------------------------------------------------------------
    # 内部核心方法
    # ----------------------------------------------------------------------
    def _setup_temp_dirs(self, video_name: str, prefix: str):
        """创建临时目录及断点文件路径"""
        temp_base = self.config.get_temp_dir("esrgan_video") / f"{prefix}_{video_name}"
        self.segment_dir = temp_base / "segments"
        self.processed_dir = temp_base / "processed"
        self.checkpoint_file = temp_base / "checkpoint.json"

        self.segment_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _load_checkpoint(self) -> dict:
        """加载断点信息"""
        if self.checkpoint_file and self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                print(f"📌 发现断点: 已完成 {len(checkpoint.get('processed_segments', []))} 个分段")
                return checkpoint
            except Exception:
                pass
        return {"processed_segments": [], "last_segment": -1}

    def _save_checkpoint(self, checkpoint: dict):
        """保存断点信息"""
        if self.checkpoint_file:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)

    def _process_segments(self, segment_files: List[str], checkpoint: dict) -> List[str]:
        """
        通用分段处理循环（支持断点恢复）

        Args:
            segment_files: 输入分段文件列表
            checkpoint: 断点字典

        Returns:
            处理成功的分段输出路径列表
        """
        print(f"\n⚙️  开始处理片段...")
        processed_files = []
        start_time = time.time()

        for idx, seg_path in enumerate(segment_files):
            seg_name = Path(seg_path).name

            # 检查是否已处理
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
                avg_time = elapsed / completed
                remaining = (len(segment_files) - completed) * avg_time
                print(f"   ⏱️  已用时: {format_time(elapsed)}, 预计剩余: {format_time(remaining)}")

        if processed_files:
            print(f"\n✅ Real-ESRGAN 处理完成: {len(processed_files)}/{len(segment_files)} 个分段")
        else:
            print("\n❌ 没有成功处理的片段")
        return processed_files

    def _process_segment(self, input_path: str, output_path: str, segment_idx: int) -> bool:
        """
        处理单个视频片段（直接调用 inference_realesrgan_video）

        Args:
            input_path: 输入片段路径
            output_path: 期望的输出路径
            segment_idx: 片段索引

        Returns:
            是否成功
        """
        try:
            print(f"   🎬 处理片段 {segment_idx+1}: {Path(input_path).name}")

            # 获取片段时长（仅用于显示）
            duration = get_video_duration(input_path)
            if duration:
                print(f"   📊 片段时长: {format_time(duration)}")

            # 调用 Real-ESRGAN 视频处理
            success = self._run_esrgan_video(input_path, output_path, segment_idx)

            if success:
                # 验证输出文件完整性
                if verify_video_integrity(output_path):
                    out_duration = get_video_duration(output_path)
                    print(f"   ✅ 处理完成: {format_time(out_duration)}")
                    return True
                else:
                    print(f"   ❌ 输出文件验证失败")
                    return False
            return False

        except Exception as e:
            print(f"   ❌ 处理片段时发生异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _run_esrgan_video(self, input_path: str, output_path: str, segment_idx: int) -> bool:
        """
        调用 inference_realesrgan_video.py 执行视频超分

        Args:
            input_path: 输入视频路径
            output_path: 最终输出路径
            segment_idx: 片段索引（用于临时文件名）

        Returns:
            是否成功
        """
        try:
            # 准备参数命名空间
            args = argparse.Namespace()

            # 基本路径
            args.input = input_path
            args.output = str(Path(output_path).parent)  # 输出目录

            # 模型与处理参数
            args.model_name = self.model_name
            args.denoise_strength = self.denoise_strength
            args.outscale = self.upscale_factor
            args.suffix = f"processed_{segment_idx:03d}"  # 后缀用于临时文件

            args.tile = self.tile_size
            args.tile_pad = self.tile_pad
            args.pre_pad = self.pre_pad
            args.face_enhance = self.face_enhance
            args.fp32 = self.fp32
            args.fps = None  # 保持原帧率
            args.ffmpeg_bin = "ffmpeg"
            args.extract_frame_first = False
            args.num_process_per_gpu = self.num_process_per_gpu

            # 其他 inference_realesrgan_video 所需参数
            args.alpha_upsampler = "realesrgan"
            args.ext = "auto"

            # 构造视频名称（用于临时文件前缀）
            video_name = Path(input_path).stem
            args.video_name = f"{video_name}_{segment_idx:03d}"

            # 确保输出目录存在
            os.makedirs(args.output, exist_ok=True)

            # 动态导入并运行
            from inference_realesrgan_video import run

            print(f"   🔧 加载模型: {self.model_name}")
            start_time = time.time()

            run(args)

            # inference_realesrgan_video 生成的默认文件路径格式：
            # {output_dir}/{args.video_name}_{args.suffix}.mp4
            temp_output = os.path.join(args.output, f"{args.video_name}_{args.suffix}.mp4")
            if os.path.exists(temp_output):
                shutil.move(temp_output, output_path)
                elapsed = time.time() - start_time
                print(f"   ✅ 处理完成 ({format_time(elapsed)})")
                return True
            else:
                print(f"   ❌ 输出文件未生成: {temp_output}")
                return False

        except ImportError as e:
            print(f"   ❌ 无法导入 inference_realesrgan_video: {e}")
            return False
        except Exception as e:
            print(f"   ❌ 调用 Real-ESRGAN 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _cleanup_temp_files(self):
        """清理临时目录（分段和中间处理文件）"""
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
    print("Real-ESRGAN视频超分处理器模块 v2.0 (视频直接处理版)")
    print("✅ 模块加载成功，请在主程序中调用。")