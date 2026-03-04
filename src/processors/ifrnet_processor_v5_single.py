"""
IFRNet 视频插帧处理器 v5（单卡版）
=====================================
对接 process_video_v5_single.py（IFRNetVideoProcessor），
保留分段直接对接与断点恢复逻辑，支持 v5 全部硬件加速参数：
  - FP16 / torch.compile / CUDA Graph（compile 激活时自动接管 Graph）
  - TensorRT 可选加速（首次构建需缓存 .trt Engine）
  - NVDEC 硬件解码 / NVENC 硬件编码（自动探测，失败时回退软解/软编）
  - OOM 自动降级：batch_size 减半 → 深度清理 → 显存估算恢复
  - torch.compile 预热：小形状（32×32）触发编译，避免大分辨率首次卡顿
  - JSON 性能报告（可选，含 infer_latency_ms / nvdec / nvenc 等字段）
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
    """IFRNet 插帧处理器 v5（单卡版）"""

    def __init__(self, config):
        """
        初始化处理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.ifrnet_dir = Path(config.get("paths", "base_dir")) / "external" / "IFRNet"
        self.model_path = config.get("models", "ifrnet", "model_path")

        # 推理设备与基础参数
        use_gpu = config.get("models", "ifrnet", "use_gpu", default=True)
        self.device = "cuda" if use_gpu else "cpu"
        self.batch_size = config.get("models", "ifrnet", "batch_size", default=4)
        self.max_batch_size = config.get("models", "ifrnet", "max_batch_size", default=16)
        self.interpolation_factor = config.get("processing", "interpolation_factor", default=2)
        self.segment_duration = config.get("processing", "segment_duration", default=30)

        # v5 新增：推理优化参数
        self.use_fp16       = config.get("models", "ifrnet", "use_fp16",       default=True)
        self.use_compile    = config.get("models", "ifrnet", "use_compile",    default=True)
        self.use_cuda_graph = config.get("models", "ifrnet", "use_cuda_graph", default=True)
        self.use_tensorrt   = config.get("models", "ifrnet", "use_tensorrt",   default=False)

        # v5 新增：硬件解/编码参数
        self.use_hwaccel = config.get("models", "ifrnet", "use_hwaccel", default=True)
        self.codec       = config.get("models", "ifrnet", "codec",       default="libx264")
        self.crf         = config.get("models", "ifrnet", "crf",         default=18)
        self.keep_audio  = config.get("models", "ifrnet", "keep_audio",  default=True)
        self.ffmpeg_bin  = config.get("models", "ifrnet", "ffmpeg_bin",  default="ffmpeg")
        self.report_json = config.get("models", "ifrnet", "report_json", default=None)

        # 验证 IFRNet 目录
        if not self.ifrnet_dir.exists():
            raise FileNotFoundError(f"IFRNet 目录不存在: {self.ifrnet_dir}")

        # 将 IFRNet 加入 Python 路径
        sys.path.insert(0, str(self.ifrnet_dir))

        # 临时目录句柄（在 _setup_temp_dirs 中初始化）
        self.checkpoint_file: Optional[Path] = None
        self.segment_dir:     Optional[Path] = None
        self.processed_dir:   Optional[Path] = None

    # -------------------------------------------------------------------------
    # 公共接口
    # -------------------------------------------------------------------------

    def process_video_segments(self, input_video: str) -> List[str]:
        """
        对完整视频执行插帧，返回处理后的分段列表（不合并）。

        Args:
            input_video: 输入视频路径

        Returns:
            处理后的分段文件路径列表
        """
        print(f"\n🎬 IFRNet 插帧处理（分段模式）")
        print(f"📹 输入: {input_video}")
        print(f"⚡ 插帧倍数: {self.interpolation_factor}x")

        video_name = Path(input_video).stem
        self._setup_temp_dirs(video_name, "ifrnet_source")
        checkpoint = self._load_checkpoint()

        duration = get_video_duration(input_video)
        if duration is None:
            print("❌ 无法获取视频时长")
            return []

        print(f"📊 时长: {format_time(duration)}, 分段: {self.segment_duration}秒")

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
        直接对已有分段执行插帧（用于对接上游处理器输出）。

        Args:
            input_segments: 输入分段文件路径列表
            video_name:     视频名称（用于临时目录命名）

        Returns:
            处理后的分段文件路径列表
        """
        print(f"\n🎬 IFRNet 插帧处理（接收分段输入）")
        print(f"📹 输入分段数: {len(input_segments)}")
        print(f"⚡ 插帧倍数: {self.interpolation_factor}x")

        self._setup_temp_dirs(video_name, "ifrnet_from_segments")
        checkpoint = self._load_checkpoint()
        return self._process_segments(input_segments, checkpoint)

    # -------------------------------------------------------------------------
    # 内部方法
    # -------------------------------------------------------------------------

    def _setup_temp_dirs(self, video_name: str, prefix: str):
        """创建并记录临时目录路径。"""
        temp_base = self.config.get_temp_dir("ifrnet") / f"{prefix}_{video_name}"

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
                print(f"📌 发现断点: 已完成 {len(checkpoint['processed_segments'])} 个分段")
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

        for i, segment_file in enumerate(segment_files):
            segment_name = Path(segment_file).name

            # 断点跳过
            if i in checkpoint["processed_segments"]:
                print(f"\n⏭️  片段 {i+1}/{len(segment_files)}: {segment_name} (已处理)")
                output_file = self.processed_dir / f"interpolated_{segment_name}"
                if output_file.exists():
                    processed_files.append(str(output_file))
                    continue

            print(f"\n🎬 片段 {i+1}/{len(segment_files)}: {segment_name}")
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
                avg_time  = elapsed / completed
                remaining = (len(segment_files) - completed) * avg_time
                print(f"   ⏱️  已用时: {format_time(elapsed)}, "
                      f"预计剩余: {format_time(remaining)}")

        if processed_files:
            print(f"\n✅ IFRNet 处理完成: "
                  f"{len(processed_files)}/{len(segment_files)} 个分段")
        else:
            print("\n❌ 没有成功处理的片段")

        return processed_files

    def _process_segment(self, segment_path: str, output_path: str) -> bool:
        """
        调用 process_video_v5_single.IFRNetVideoProcessor 处理单个分段。

        Args:
            segment_path: 输入片段路径
            output_path:  输出路径

        Returns:
            是否成功
        """
        try:
            # 导入 v5 单卡版处理器
            from process_video_v5_single import IFRNetVideoProcessor

            print(f"   🎬 处理片段: {Path(segment_path).name}")
            print(f"   📊 插帧倍数: {self.interpolation_factor}x")
            print(f"   🖥️  设备: {self.device} | "
                  f"FP16: {self.use_fp16} | "
                  f"CUDA Graph: {self.use_cuda_graph} | "
                  f"TRT: {self.use_tensorrt}")

            processor = IFRNetVideoProcessor(
                model_path    = self.model_path,
                device        = self.device,
                batch_size    = self.batch_size,
                max_batch_size= self.max_batch_size,
                use_fp16      = self.use_fp16,
                use_compile   = self.use_compile,
                use_cuda_graph= self.use_cuda_graph,
                use_tensorrt  = self.use_tensorrt,
                use_hwaccel   = self.use_hwaccel,
                codec         = self.codec,
                crf           = self.crf,
                keep_audio    = self.keep_audio,
                ffmpeg_bin    = self.ffmpeg_bin,
                report_json   = self.report_json,
            )

            ok = processor.process_video(
                input_path  = segment_path,
                output_path = output_path,
                scale       = float(self.interpolation_factor),
            )

            if ok and verify_video_integrity(output_path):
                seg_duration = get_video_duration(output_path)
                print(f"   ✅ 处理完成: {format_time(seg_duration)}")
                return True
            else:
                print(f"   ❌ 输出文件验证失败")
                return False

        except ImportError as e:
            print(f"   ❌ 无法导入 process_video_v5_single: {e}")
            return False
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    print("IFRNet 插帧处理器模块 v5（单卡版）")
