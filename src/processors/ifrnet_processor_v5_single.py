"""
IFRNet 视频插帧处理器 v5（单卡版）
=====================================
对接 process_video_v6_1_single.py（IFRNetVideoProcessor），
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

# ----- Add the utils directory so that video_utils and config_manager can be found -----
script_dir = Path(__file__).resolve().parent          # src/processors
project_root = script_dir.parent.parent                # Video_Enhancement/
utils_path = str(project_root / "src" / "utils")
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
# ----------------------------------------------------------------------------------------

from video_utils import (
    get_video_duration, format_time, verify_video_integrity,
    split_video_by_time
)


class IFRNetProcessor:
    """IFRNet 插帧处理器 v5（单卡版）"""

    # 支持的模型名称 → 文件名映射（与 process_video_v6_1_single.py 保持一致）
    MODEL_NAME_MAP = {
        "IFRNet_Vimeo90K":   "IFRNet_Vimeo90K.pth",
        "IFRNet_S_Vimeo90K": "IFRNet_S_Vimeo90K.pth",
        "IFRNet_L_Vimeo90K": "IFRNet_L_Vimeo90K.pth",
    }

    def __init__(self, config):
        """
        初始化处理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.ifrnet_dir = Path(config.get("paths", "base_dir")) / "external" / "IFRNet"

        # ── 模型路径解析（优先级: model_path 显式路径 > model_name 自动拼接 > 默认名称）
        # model_name: 用于从约定目录自动拼接路径
        self.model_name = config.get("models", "ifrnet", "model_name",
                                     default="IFRNet_S_Vimeo90K")
        # model_path: 显式指定时直接使用，为空则按 model_name 拼接
        _model_path = config.get("models", "ifrnet", "model_path", default="")
        if _model_path:
            # 用户显式指定了路径，直接使用
            self.model_path = _model_path
        else:
            # 按 model_name 在约定目录下拼接绝对路径
            base_dir       = Path(config.get("paths", "base_dir"))
            checkpoints_dir = base_dir / "models_IFRNet" / "checkpoints"
            pth_filename   = self.MODEL_NAME_MAP.get(
                self.model_name, f"{self.model_name}.pth"
            )
            self.model_path = str(checkpoints_dir / pth_filename)

        # 提前校验模型文件是否存在，给出清晰的错误提示
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"IFRNet 模型文件不存在: {self.model_path}\n"
                f"  · 请将 .pth 权重文件放置到上述路径，或\n"
                f"  · 使用 --ifrnet-model-path /绝对/路径/model.pth 直接指定，或\n"
                f"  · 使用 --ifrnet-model IFRNet_S_Vimeo90K 指定模型名称（需存在对应 .pth）。\n"
                f"  · 可选模型名: {', '.join(self.MODEL_NAME_MAP.keys())}"
            )

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
        调用 process_video_v6_1_single.IFRNetVideoProcessor 处理单个分段。

        Args:
            segment_path: 输入片段路径
            output_path:  输出路径

        Returns:
            是否成功
        """
        try:
            # 导入 v5 单卡版处理器
            from process_video_v6_1_single import IFRNetVideoProcessor

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
            print(f"   ❌ 无法导入 process_video_v6_1_single: {e}")
            return False
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    # -------------------------------------------------------------------------
    # 完整流程：分段 → 插帧 → 合并（含音频），供独立调用使用
    # -------------------------------------------------------------------------

    def process_video(self, input_video: str, output_video: str) -> bool:
        """
        完整处理视频（分段插帧 → 合并），支持断点恢复。
        与 RealESRGANVideoProcessor.process_video() 对称，供独立调用或测试。

        Args:
            input_video:  输入视频路径
            output_video: 最终输出视频路径

        Returns:
            是否成功
        """
        from video_utils import (
            VideoInfo, smart_extract_audio, merge_videos_by_codec, format_time
        )
        import time

        print("\n" + "=" * 60)
        print("🎬 IFRNet 视频插帧处理（完整流程）")
        print(f"📹 输入  : {input_video}")
        print(f"📤 输出  : {output_video}")
        print(f"⚡ 插帧倍数: {self.interpolation_factor}x")
        print(f"🖥️  设备  : {self.device}")
        print(f"🧩 分段时长: {self.segment_duration}秒")
        print("=" * 60 + "\n")

        total_start = time.time()

        # 提取音频
        audio_path = None
        try:
            info = VideoInfo(input_video)
            if info.has_audio:
                print("🎵 提取音频...")
                audio_path = smart_extract_audio(
                    input_video,
                    str(self.config.get_temp_dir("ifrnet_audio"))
                )
                if audio_path:
                    print(f"✅ 音频已保存: {audio_path}")
                else:
                    print("⚠️  音频提取失败，输出将无音频")
        except Exception as e:
            print(f"⚠️  获取视频信息或提取音频失败: {e}")

        # 插帧（分段）
        processed_segments = self.process_video_segments(input_video)
        if not processed_segments:
            print("❌ 未成功处理任何分段")
            return False

        # 合并
        print(f"\n🔗 合并 {len(processed_segments)} 个插帧分段...")
        output_config = self.config.get_section("output", {})
        success = merge_videos_by_codec(
            processed_segments, output_video,
            audio_path=audio_path,
            config=output_config,
        )

        if success:
            total_time = time.time() - total_start
            print(f"\n✅ 插帧处理完成！总用时: {format_time(total_time)}")
            print(f"📤 输出: {output_video}")
            if self.config.get("processing", "auto_cleanup_temp", default=False):
                self._cleanup_temp_files()
        else:
            print("❌ 视频合并失败")

        return success

    def _cleanup_temp_files(self):
        """清理临时目录（分段和中间处理文件）。"""
        import shutil as _shutil
        print("\n🧹 清理临时文件...")
        try:
            if self.segment_dir and self.segment_dir.exists():
                _shutil.rmtree(self.segment_dir)
                print("✅ 已删除分段文件")
            if self.processed_dir and self.processed_dir.exists():
                _shutil.rmtree(self.processed_dir)
                print("✅ 已删除处理文件")
        except Exception as e:
            print(f"⚠️  清理失败: {e}")


# =============================================================================
# 独立命令行入口
# =============================================================================

def main():
    """
    独立调用入口：直接驱动 IFRNetProcessor，
    底层对接 process_video_v6_1_single.IFRNetVideoProcessor。

    示例：
      # 使用默认配置，直接插帧
      python ifrnet_processor_v5_single.py -i input.mp4 -o output_2x.mp4

      # 指定配置文件 + 覆盖插帧倍数
      python ifrnet_processor_v5_single.py -c config.json \\
             -i input.mp4 -o output_4x.mp4 --interpolation-factor 4

      # 关闭 compile（短视频跳过预热，启动更快）
      python ifrnet_processor_v5_single.py -i input.mp4 -o output.mp4 \\
             --no-compile --no-cuda-graph
    """
    import argparse
    import sys

    _script_dir  = Path(os.path.abspath(__file__)).parent        # src/processors
    _base_dir    = _script_dir.parent.parent                     # project root
    _default_cfg = str(_base_dir / "config" / "default_config.json")

    sys.path.insert(0, str(_base_dir / "src" / "utils"))
    from config_manager import Config

    parser = argparse.ArgumentParser(
        description="IFRNet 视频插帧处理器（单卡版）—— 独立入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
底层脚本：external/IFRNet/process_video_v6_1_single.py

特性：
  · 分段处理 + 断点恢复
  · FP16 / CUDA Graph / torch.compile / TensorRT 可选
  · NVDEC 硬件解码 / NVENC 硬件编码自动探测
  · OOM 自动降级（batch_size 减半 → 深度清理 → 显存估算恢复）

模型选项（二选一，model-path 优先）：
  --ifrnet-model IFRNet_S_Vimeo90K   轻量默认（推荐）
  --ifrnet-model IFRNet_L_Vimeo90K   高质量，速度更慢
  --ifrnet-model-path /path/to/x.pth 直接指定 .pth 路径
""",
    )

    # 基础参数
    parser.add_argument("--config", "-c", default=_default_cfg,
                        help=f"配置文件路径（默认: {_default_cfg}）")
    parser.add_argument("--input",  "-i", required=True,
                        help="输入视频路径")
    parser.add_argument("--output", "-o", required=True,
                        help="输出视频路径（含文件名）")

    # ── IFRNet 模型参数 ──────────────────────────────────────────────────────
    parser.add_argument("--ifrnet-model", metavar="MODEL_NAME",
                        choices=["IFRNet_Vimeo90K", "IFRNet_S_Vimeo90K", "IFRNet_L_Vimeo90K"],
                        help="IFRNet 模型名称（processor 自动在 models_IFRNet/checkpoints/ 下查找对应 .pth）；"
                             "覆盖配置中 models.ifrnet.model_name")
    parser.add_argument("--ifrnet-model-path", metavar="PATH",
                        help="IFRNet .pth 权重文件绝对路径（优先级高于 --ifrnet-model）；"
                             "覆盖配置中 models.ifrnet.model_path")

    # 覆盖配置
    parser.add_argument("--interpolation-factor", type=int, choices=[2, 4, 8, 16],
                        help="插帧倍数（覆盖配置）")
    parser.add_argument("--segment-duration", type=int,
                        help="分段时长（秒，覆盖配置）")
    parser.add_argument("--batch-size", type=int,
                        help="推理批处理大小（覆盖配置）")
    parser.add_argument("--max-batch-size", type=int,
                        help="批大小上限（覆盖配置）")

    # 推理优化开关
    parser.add_argument("--no-fp16",       action="store_true", help="禁用 FP16")
    parser.add_argument("--no-compile",    action="store_true", help="禁用 torch.compile")
    parser.add_argument("--no-cuda-graph", action="store_true", help="禁用 CUDA Graph")
    parser.add_argument("--use-tensorrt",  action="store_true", help="启用 TensorRT 加速")
    parser.add_argument("--no-hwaccel",    action="store_true", help="禁用 NVDEC 硬件解码")
    parser.add_argument("--no-audio",      action="store_true", help="不提取/保留音频")
    parser.add_argument("--crf", type=int, help="分段输出视频质量 CRF（覆盖配置）")
    parser.add_argument("--report", metavar="PATH", help="输出 JSON 性能报告路径")
    parser.add_argument("--auto-cleanup",  action="store_true",
                        help="处理完成后自动清理临时文件")

    args = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────────────────────────────
    try:
        config = Config(args.config)
        print(f"⚙️  配置文件: {args.config}")
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        return 1

    # ── 命令行参数覆盖配置 ────────────────────────────────────────────────────
    config.set("paths", "input_video", value=args.input)
    config.set("paths", "output_dir",  value=str(Path(args.output).parent))
    config.set("processing", "batch_mode", value=False)

    # IFRNet 模型参数（model_path 优先级高于 model_name）
    if args.ifrnet_model_path:
        config.set("models", "ifrnet", "model_path", value=args.ifrnet_model_path)
        # 同时清空 model_name，避免 processor 按 name 拼路径覆盖显式路径
        config.set("models", "ifrnet", "model_name", value="")
    elif args.ifrnet_model:
        config.set("models", "ifrnet", "model_name", value=args.ifrnet_model)
        config.set("models", "ifrnet", "model_path", value="")  # 让 processor 按 name 拼接

    if args.interpolation_factor:
        config.set("processing", "interpolation_factor", value=args.interpolation_factor)
    if args.segment_duration:
        config.set("processing", "segment_duration", value=args.segment_duration)
    if args.batch_size:
        config.set("models", "ifrnet", "batch_size", value=args.batch_size)
    if args.max_batch_size:
        config.set("models", "ifrnet", "max_batch_size", value=args.max_batch_size)

    # 推理优化
    if args.no_fp16:
        config.set("models", "ifrnet", "use_fp16", value=False)
    if args.no_compile:
        config.set("models", "ifrnet", "use_compile", value=False)
    if args.no_cuda_graph:
        config.set("models", "ifrnet", "use_cuda_graph", value=False)
    if args.use_tensorrt:
        config.set("models", "ifrnet", "use_tensorrt", value=True)
    if args.no_hwaccel:
        config.set("models", "ifrnet", "use_hwaccel", value=False)
    if args.no_audio:
        config.set("models", "ifrnet", "keep_audio", value=False)
    if args.crf is not None:
        config.set("models", "ifrnet", "crf", value=args.crf)
    if args.report:
        config.set("models", "ifrnet", "report_json", value=args.report)
    if args.auto_cleanup:
        config.set("processing", "auto_cleanup_temp", value=True)

    # ── 创建处理器并执行 ──────────────────────────────────────────────────────
    try:
        processor = IFRNetProcessor(config)
    except Exception as e:
        print(f"❌ 初始化处理器失败: {e}")
        return 1

    print(f"\n🎬 IFRNet 独立插帧")
    print(f"   输入  : {args.input}")
    print(f"   输出  : {args.output}")
    print(f"   模型  : {processor.model_name or '(直接路径)'} → {processor.model_path}")
    print(f"   倍数  : {processor.interpolation_factor}x")
    print(f"   设备  : {processor.device}")
    print(f"   批大小: {processor.batch_size} (上限 {processor.max_batch_size})")
    print(f"   FP16  : {processor.use_fp16} | compile: {processor.use_compile}"
          f" | CUDAGraph: {processor.use_cuda_graph} | TRT: {processor.use_tensorrt}")

    success = processor.process_video(args.input, args.output)
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

