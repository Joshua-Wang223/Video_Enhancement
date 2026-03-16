"""
Real-ESRGAN 视频超分处理器 v6（单卡版）
==========================================
对接 inference_realesrgan_video_v6_1_single.py（run / inference_video_single），
保留分段直接对接与断点恢复逻辑，支持 v6.1 全部硬件加速参数：
  - FP16 / torch.compile
  - TensorRT 可选加速（TRT 8.x / 10.x 双 API 兼容；FIX-3 已修复，真正有效）
  - NVDEC 硬件解码 / NVENC 硬件编码（自动探测，失败时回退软解/软编）
  - OOM 级联保护（backport from IFRNet v5）：
      首次 OOM 永久降低 max_batch_size 天花板
      级联 OOM 不重复修改上限（内存仍脏，惩罚无意义）
      batch_size=1 仍 OOM → 深度清理 + 按剩余显存动态恢复
  - 批量推理（batch_size 默认 12）/ JSON 性能报告
  - face_enhance（v6 重点优化）：
      批量 GFPGAN 推理（堆叠一批所有人脸 crops → 单次前向）
      原始帧检测（低分辨率帧上 RetinaFace，比 SR 帧快 4×）
      无人脸帧跳过（检测为空时直接跳过 GFPGAN，零额外开销）
      CPU-GPU 流水线并行（detect/paste 后台线程与 SR 主线程并行）
      GFPGAN FP16（torch.autocast，利用 Tensor Core）
      GFPGAN OOM 保护（gfpgan_batch_size 子批量 + OOM 自动降级）

【v6 变更说明（相对 v5）】
  - 对齐底层 inference_realesrgan_video_v6_1_single.py v6.1
  - 构造函数新增参数与 default_config.json 完全对齐
  - main() CLI 新增 --prefetch-factor（v5 中有属性但 CLI 缺失）
  - main() CLI 参数文档更新，覆盖 face_enhance 流水线说明
  - _run_esrgan_video Namespace 字段与底层 v6.1 argparse 定义严格对齐
  - 输出路径处理：兼容 run() 新增的 _output_is_file() 逻辑
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

# ----- Add the utils directory so that video_utils and config_manager can be found -----
script_dir   = Path(__file__).resolve().parent          # src/processors
project_root = script_dir.parent.parent                 # Video_Enhancement/
utils_path   = str(project_root / "src" / "utils")
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)
# ----------------------------------------------------------------------------------------

from video_utils import (
    get_video_duration, format_time, verify_video_integrity,
    split_video_by_time, merge_videos_by_codec,
)


class RealESRGANVideoProcessor:
    """Real-ESRGAN 视频超分处理器 v6（单卡版）"""

    def __init__(self, config):
        """
        初始化处理器

        Args:
            config: 配置对象（应包含 paths, models.realesrgan, processing 等节）
        """
        self.config    = config
        self.esrgan_dir = Path(config.get("paths", "base_dir")) / "external" / "Real-ESRGAN"
        self.model_name = config.get("models", "realesrgan", "model_name",
                                     default="realesr-general-x4v3")
        self.model_path = config.get("models", "realesrgan", "model_path", default="")

        # 推理设备与基础参数
        use_gpu = config.get("models", "realesrgan", "use_gpu", default=True)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        self.upscale_factor   = config.get("processing",    "upscale_factor",   default=4)
        self.segment_duration = config.get("processing",    "segment_duration",  default=30)

        # Real-ESRGAN 基础推理参数
        self.denoise_strength = config.get("models", "realesrgan", "denoise_strength", default=0.5)
        self.tile_size        = config.get("models", "realesrgan", "tile_size",        default=0)
        self.tile_pad         = config.get("models", "realesrgan", "tile_pad",         default=10)
        self.pre_pad          = config.get("models", "realesrgan", "pre_pad",          default=0)
        self.use_fp16         = config.get("models", "realesrgan", "use_fp16",         default=True)
        self.face_enhance     = config.get("models", "realesrgan", "face_enhance",     default=False)

        # 推理优化参数（v5+，对齐 v6.1 默认值）
        self.batch_size      = config.get("models", "realesrgan", "batch_size",      default=12)
        self.prefetch_factor = config.get("models", "realesrgan", "prefetch_factor", default=24)
        self.use_compile     = config.get("models", "realesrgan", "use_compile",     default=True)
        self.use_cuda_graph  = config.get("models", "realesrgan", "use_cuda_graph",  default=True)
        self.use_tensorrt    = config.get("models", "realesrgan", "use_tensorrt",    default=False)

        # face_enhance 精细控制参数（v6）
        self.gfpgan_model      = config.get("models", "realesrgan", "gfpgan_model",      default="1.4")
        self.gfpgan_weight     = config.get("models", "realesrgan", "gfpgan_weight",     default=0.5)
        self.gfpgan_batch_size = config.get("models", "realesrgan", "gfpgan_batch_size", default=12)

        # 硬件解/编码参数（v5+）
        self.use_hwaccel = config.get("models", "realesrgan", "use_hwaccel", default=True)
        self.codec       = config.get("models", "realesrgan", "codec",       default="libx264")
        self.crf         = config.get("models", "realesrgan", "crf",         default=23)
        self.ffmpeg_bin  = config.get("models", "realesrgan", "ffmpeg_bin",  default="ffmpeg")
        self.report_json = config.get("models", "realesrgan", "report_json", default=None)

        # TRT Engine 缓存目录（v6+）
        # 优先级：config.paths.trt_cache_dir（由 config_manager 自动派生为 base_dir/.trt_cache，
        # 或用户在 config / CLI --trt-cache-dir 中显式指定）；空时兜底为 base_dir/.trt_cache。
        self.trt_cache_dir = config.get("paths", "trt_cache_dir", default="") or ""

        # 验证 Real-ESRGAN 目录
        if not self.esrgan_dir.exists():
            raise FileNotFoundError(f"Real-ESRGAN 目录不存在: {self.esrgan_dir}")

        # 将 Real-ESRGAN 加入 Python 路径
        sys.path.insert(0, str(self.esrgan_dir))

        # 临时目录句柄（在 _setup_temp_dirs 中初始化）
        self.checkpoint_file: Optional[Path] = None
        self.segment_dir:     Optional[Path] = None
        self.processed_dir:   Optional[Path] = None

    # -------------------------------------------------------------------------
    # 公共接口
    # -------------------------------------------------------------------------

    def process_video(self, input_video: str, output_video: str) -> bool:
        """
        完整处理视频（分段 → 超分 → 合并），支持断点恢复。

        Args:
            input_video:  输入视频路径
            output_video: 最终输出视频路径

        Returns:
            是否成功
        """
        print("\n" + "=" * 65)
        print("🎨 Real-ESRGAN 视频超分处理（完整流程）—— v6.1")
        print(f"📹 输入: {input_video}")
        print(f"📤 输出: {output_video}")
        print(f"⚡ 超分倍数: {self.upscale_factor}x")
        print(f"🖥️  设备: {self.device}")
        print(f"🧩 分段时长: {self.segment_duration}秒")
        print(f"   FP16: {self.use_fp16} | compile: {self.use_compile}"
              f" | CUDA Graph: {self.use_cuda_graph} | TRT: {self.use_tensorrt}")
        face_hint = (
            f" (model={self.gfpgan_model}, weight={self.gfpgan_weight},"
            f" gfpgan_batch={self.gfpgan_batch_size})"
            if self.face_enhance else ""
        )
        print(f"   face_enhance: {self.face_enhance}{face_hint}")
        print("=" * 65 + "\n")

        total_start = time.time()

        processed_segments = self.process_video_segments(input_video)
        if not processed_segments:
            print("❌ 未成功处理任何分段")
            return False

        print(f"\n🔗 合并 {len(processed_segments)} 个处理后的分段...")
        output_config = self.config.get_section("output", {})
        success = merge_videos_by_codec(processed_segments, output_video,
                                        config=output_config)

        if success:
            total_time = time.time() - total_start
            print(f"\n✅ 超分处理完成！总用时: {format_time(total_time)}")
            print(f"📤 输出: {output_video}")
            if self.config.get("processing", "auto_cleanup_temp", default=False):
                self._cleanup_temp_files()
            return True
        else:
            print("❌ 视频合并失败")
            return False

    def process_video_segments(self, input_video: str) -> List[str]:
        """
        对完整视频执行超分，返回处理后的分段列表（不合并）。

        Args:
            input_video: 输入视频路径

        Returns:
            处理后的分段文件路径列表
        """
        print(f"\n🎨 Real-ESRGAN 超分处理（分段模式）—— v6.1")
        print(f"📹 输入: {input_video}")
        print(f"⚡ 超分倍数: {self.upscale_factor}x | 模型: {self.model_name}")
        print(f"🖥️  设备: {self.device} | "
              f"FP16: {self.use_fp16} | "
              f"compile: {self.use_compile} | "
              f"CUDA Graph: {self.use_cuda_graph} | "
              f"TRT: {self.use_tensorrt}")

        video_name = Path(input_video).stem
        self._setup_temp_dirs(video_name, prefix="esrgan_video")
        checkpoint = self._load_checkpoint()

        duration = get_video_duration(input_video)
        if duration is None:
            print("❌ 无法获取视频时长")
            return []
        print(f"📊 时长: {format_time(duration)}, 分段: {self.segment_duration}秒")

        # 视频较短时直接整体处理
        if duration <= self.segment_duration:
            print("📦 视频较短，直接处理整个视频...")
            output_file = self.processed_dir / f"upscaled_{Path(input_video).name}"
            success = self._process_segment(input_video, str(output_file), segment_idx=0)
            return [str(output_file)] if success else []

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
        直接对已有分段执行超分（用于对接上游处理器输出）。

        Args:
            input_segments: 输入分段文件路径列表
            video_name:     视频名称（用于临时目录命名）

        Returns:
            处理后的分段文件路径列表
        """
        print(f"\n🎨 Real-ESRGAN 超分处理（接收分段输入）—— v6.1")
        print(f"📹 输入分段数: {len(input_segments)}")
        print(f"⚡ 超分倍数: {self.upscale_factor}x")
        print(f"🖥️  设备: {self.device}")

        self._setup_temp_dirs(video_name, prefix="esrgan_from_segments")
        checkpoint = self._load_checkpoint()
        return self._process_segments(input_segments, checkpoint)

    # -------------------------------------------------------------------------
    # 内部核心方法
    # -------------------------------------------------------------------------

    def _setup_temp_dirs(self, video_name: str, prefix: str):
        """创建并记录临时目录路径。"""
        temp_base = (self.config.get_temp_dir("esrgan_video")
                     / f"{prefix}_{video_name}")
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
                print(f"📌 发现断点: "
                      f"已完成 {len(checkpoint.get('processed_segments', []))} 个分段")
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

        for idx, seg_path in enumerate(segment_files):
            seg_name = Path(seg_path).name

            # 断点跳过
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
            elapsed   = time.time() - start_time
            completed = len(checkpoint["processed_segments"])
            if completed > 0:
                avg_time  = elapsed / completed
                remaining = (len(segment_files) - completed) * avg_time
                print(f"   ⏱️  已用时: {format_time(elapsed)}, "
                      f"预计剩余: {format_time(remaining)}")

        if processed_files:
            print(f"\n✅ Real-ESRGAN 处理完成: "
                  f"{len(processed_files)}/{len(segment_files)} 个分段")
        else:
            print("\n❌ 没有成功处理的片段")

        return processed_files

    def _process_segment(self, input_path: str, output_path: str,
                          segment_idx: int) -> bool:
        """
        处理单个视频片段（调用 inference_realesrgan_video_v6_1_single）。

        Args:
            input_path:   输入片段路径
            output_path:  期望的输出路径
            segment_idx:  片段索引

        Returns:
            是否成功
        """
        try:
            print(f"   🎬 处理片段 {segment_idx+1}: {Path(input_path).name}")
            duration = get_video_duration(input_path)
            if duration:
                print(f"   📊 片段时长: {format_time(duration)}")

            success = self._run_esrgan_video(input_path, output_path, segment_idx)

            if success:
                if verify_video_integrity(output_path):
                    out_duration = get_video_duration(output_path)
                    print(f"   ✅ 处理完成: {format_time(out_duration)}")
                    return True
                else:
                    print("   ❌ 输出文件验证失败")
                    return False
            return False

        except Exception as e:
            print(f"   ❌ 处理片段时发生异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _run_esrgan_video(self, input_path: str, output_path: str,
                           segment_idx: int) -> bool:
        """
        构建参数命名空间并调用 inference_realesrgan_video_v6_1_single.run()。

        Namespace 字段与底层 v6.1 argparse 严格对齐：
          基础: input, output, model_name, denoise_strength, outscale, suffix
          推理: tile, tile_pad, pre_pad, face_enhance, use_fp16/no_fp16, fps
          face_enhance: gfpgan_model, gfpgan_weight, gfpgan_batch_size
          优化: batch_size, prefetch_factor, use_compile, use_cuda_graph, use_tensorrt
          硬件: use_hwaccel, no_hwaccel, codec, crf, ffmpeg_bin
          输出: alpha_upsampler, ext
          报告: report

        Args:
            input_path:   输入视频路径
            output_path:  最终输出路径
            segment_idx:  片段索引（用于后缀命名）

        Returns:
            是否成功
        """
        try:
            ns = argparse.Namespace()

            # ── 基本路径 ─────────────────────────────────────────────────────
            # run() 支持 output 既可以是目录也可以是文件（_output_is_file 自动判断）。
            # 传入 output_path（目标文件路径），让 run() 直接写到该路径，
            # 规避后续 shutil.move 时路径不匹配的问题。
            ns.input  = input_path
            ns.output = output_path      # v6: 直接传文件路径，run() 内部识别

            # ── 模型与基础处理参数 ──────────────────────────────────────────
            ns.model_name       = self.model_name
            ns.denoise_strength = self.denoise_strength
            ns.outscale         = float(self.upscale_factor)
            ns.suffix           = f"processed_{segment_idx:03d}"

            ns.tile         = self.tile_size
            ns.tile_pad     = self.tile_pad
            ns.pre_pad      = self.pre_pad
            ns.face_enhance = self.face_enhance
            ns.use_fp16     = self.use_fp16
            ns.no_fp16      = not self.use_fp16  # 推理层 shim 从 no_fp16 派生 use_fp16
            ns.fps          = None          # 保持原帧率

            # ── face_enhance 精细控制参数（v6）─────────────────────────────
            ns.gfpgan_model      = self.gfpgan_model
            ns.gfpgan_weight     = self.gfpgan_weight
            ns.gfpgan_batch_size = self.gfpgan_batch_size

            # ── 推理优化参数（v5+）─────────────────────────────────────────
            ns.batch_size      = self.batch_size
            ns.prefetch_factor = self.prefetch_factor
            ns.use_compile     = self.use_compile
            ns.use_cuda_graph  = self.use_cuda_graph
            ns.use_tensorrt    = self.use_tensorrt
            # 反向标志对齐：底层 inference_video_single 通过 getattr 读取，
            # processor 已解析最终有效值，直接同步对应的 no_* 字段
            ns.no_compile     = not self.use_compile
            ns.no_cuda_graph  = not self.use_cuda_graph
            ns.no_tensorrt    = not self.use_tensorrt
            # TRT Engine 缓存目录；底层 inference_video_single() 通过 getattr 读取
            ns.trt_cache_dir   = self.trt_cache_dir or None

            # ── 硬件解/编码参数（v5+）──────────────────────────────────────
            ns.use_hwaccel = self.use_hwaccel
            ns.no_hwaccel  = not self.use_hwaccel  # 底层同时读取两个标志
            ns.codec       = self.codec
            ns.crf         = self.crf
            ns.ffmpeg_bin  = self.ffmpeg_bin

            # ── 性能报告 ────────────────────────────────────────────────────
            ns.report = self.report_json

            # ── 其他 inference_realesrgan_video_v6_1_single 所需参数 ────────
            ns.alpha_upsampler = "realesrgan"
            ns.ext             = "auto"

            # 动态导入并运行 v6.1 单卡版
            from inference_realesrgan_video_v6_1_single import run

            print(f"   🔧 加载模型: {self.model_name}")
            print(f"   🖥️  设备: {self.device} | "
                  f"FP16: {self.use_fp16} | "
                  f"compile: {self.use_compile} | "
                  f"CUDA Graph: {self.use_cuda_graph} | "
                  f"TRT: {self.use_tensorrt}")
            print(f"   📦 batch_size: {self.batch_size} | "
                  f"prefetch: {self.prefetch_factor}")
            if self.face_enhance:
                print(f"   👤 face_enhance: GFPGAN-{self.gfpgan_model} | "
                      f"weight={self.gfpgan_weight} | "
                      f"gfpgan_batch={self.gfpgan_batch_size}")

            start_time = time.time()
            run(ns)
            elapsed = time.time() - start_time

            # run() 在 output_path 是文件路径时直接写入该文件
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"   ✅ 处理完成 ({format_time(elapsed)})")
                return True
            else:
                # 兼容旧版 run()：output_path 是目录时，拼接生成的文件名
                video_name   = Path(input_path).stem
                temp_output  = str(
                    Path(output_path).parent
                    / f"{video_name}_{ns.suffix}.mp4"
                )
                if os.path.exists(temp_output):
                    shutil.move(temp_output, output_path)
                    print(f"   ✅ 处理完成 ({format_time(elapsed)})")
                    return True
                else:
                    print(f"   ❌ 输出文件未生成: {output_path}")
                    return False

        except ImportError as e:
            print(f"   ❌ 无法导入 inference_realesrgan_video_v6_1_single: {e}")
            return False
        except Exception as e:
            print(f"   ❌ 调用 Real-ESRGAN 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _cleanup_temp_files(self):
        """清理临时目录（分段和中间处理文件）。"""
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


# =============================================================================
# 独立命令行入口
# =============================================================================

def main():
    """
    独立调用入口：直接驱动 RealESRGANVideoProcessor，
    底层对接 inference_realesrgan_video_v6_1_single.run()。

    示例：
      # 使用默认配置，直接超分（compile + CUDA Graph 默认开启）
      python realesrgan_processor_v6_single.py -i input.mp4 -o output.mp4

      # 指定配置文件 + 开启人脸增强
      python realesrgan_processor_v6_single.py -c config.json \\
             -i input.mp4 -o output.mp4 --face-enhance

      # 覆盖超分倍数、模型、GFPGAN 精细控制
      python realesrgan_processor_v6_single.py -i input.mp4 -o output.mp4 \\
             --upscale-factor 4 --model realesr-animevideov3 \\
             --face-enhance --gfpgan-model 1.4 --gfpgan-weight 0.5 \\
             --gfpgan-batch-size 8

      # TensorRT 加速 + 大批量（TRT 优先，compile/CUDA Graph 自动禁用）
      python realesrgan_processor_v6_single.py -i input.mp4 -o output.mp4 \\
             --use-tensorrt --batch-size 16 --prefetch-factor 32

      # 禁用所有 GPU 加速（调试模式）
      python realesrgan_processor_v6_single.py -i input.mp4 -o output.mp4 \\
             --no-compile --no-cuda-graph --no-fp16
    """

    # 自动定位默认配置文件（假设脚本在 src/processors/，config 在项目根/config/）
    _script_dir  = Path(os.path.abspath(__file__)).parent          # src/processors
    _base_dir    = _script_dir.parent.parent                       # project root
    _default_cfg = str(_base_dir / "config" / "default_config.json")

    sys.path.insert(0, str(_base_dir / "src" / "utils"))
    from config_manager import Config

    parser = argparse.ArgumentParser(
        description="Real-ESRGAN 视频超分处理器 v6（单卡版）—— 独立入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
底层脚本：external/Real-ESRGAN/inference_realesrgan_video_v6_1_single.py

特性：
  · 分段处理 + 断点恢复
  · FP16 / NVDEC / NVENC / torch.compile / TensorRT 可选
  · face_enhance：批量GFPGAN + 原始帧检测 + CPU-GPU流水线 + GFPGAN FP16
  · OOM 级联保护（首次 OOM 永久更新 max_batch 天花板）

模型选项（--model 指定名称，脚本自动拼路径 + 不存在时自动下载）：
  realesr-general-x4v3   通用高质量（推荐，支持 denoise_strength）
  RealESRGAN_x4plus      经典 4× 模型
  RealESRGAN_x2plus      经典 2× 模型
  realesr-animevideov3   动漫视频专用（不支持 face_enhance）
  RealESRGANv2-animevideo-xsx2  动漫视频 2× 轻量版
""",
    )

    # ── 基础参数 ─────────────────────────────────────────────────────────────
    parser.add_argument("--config", "-c", default=_default_cfg,
                        help=f"配置文件路径（默认: {_default_cfg}）")
    parser.add_argument("--input",  "-i", required=True,
                        help="输入视频路径")
    parser.add_argument("--output", "-o", required=True,
                        help="输出视频路径（含文件名）")

    # ── 模型参数 ─────────────────────────────────────────────────────────────
    parser.add_argument("--model",
                        help="模型名称，如 realesr-general-x4v3 / RealESRGAN_x4plus（覆盖配置）")
    parser.add_argument("--upscale-factor", type=int, choices=[2, 4],
                        help="超分倍数（覆盖配置文件）")
    parser.add_argument("--denoise-strength", type=float,
                        help="降噪强度 0~1（仅 realesr-general-x4v3 有效，覆盖配置）")

    # ── 处理参数 ─────────────────────────────────────────────────────────────
    parser.add_argument("--segment-duration", type=int,
                        help="分段时长（秒，覆盖配置，默认 30）")
    parser.add_argument("--batch-size", type=int,
                        help="SR 批处理大小（覆盖配置，默认 12）")
    parser.add_argument("--prefetch-factor", type=int,
                        help="读帧预取队列深度（建议 ≥ batch_size×2，覆盖配置，默认 24）")
    parser.add_argument("--tile-size", type=int,
                        help="tile 切块大小（0=不切块；VRAM 不足时设 512，覆盖配置）")
    parser.add_argument("--tile-pad", type=int,
                        help="tile 边缘填充大小（默认 10）")
    parser.add_argument("--pre-pad", type=int,
                        help="预处理填充大小（默认 0）")

    # ── face_enhance 精细控制 ─────────────────────────────────────────────────
    _fe = parser.add_mutually_exclusive_group()
    _fe.add_argument("--face-enhance",    dest="face_enhance", action="store_true",
                     default=None, help="开启人脸增强（GFPGAN）")
    _fe.add_argument("--no-face-enhance", dest="face_enhance", action="store_false",
                     help="关闭人脸增强（覆盖配置）")
    parser.add_argument("--gfpgan-model", choices=["1.3", "1.4", "RestoreFormer"],
                        help="GFPGAN 版本（覆盖配置，--face-enhance 时生效）")
    parser.add_argument("--gfpgan-weight", type=float,
                        help="GFPGAN 融合权重 0.0~1.0（0=不增强，1=完全替换，覆盖配置）")
    parser.add_argument("--gfpgan-batch-size", type=int,
                        help="单次 GFPGAN 前向最多处理的人脸数（OOM 保护，覆盖配置）")

    # ── 推理/编码开关 ─────────────────────────────────────────────────────────
    parser.add_argument("--no-hwaccel",    action="store_true",
                        help="禁用 NVDEC 硬件解码（默认自动探测）")
    parser.add_argument("--no-compile",    action="store_true",
                        help="禁用 torch.compile（默认开启；短视频或调试时可禁用跳过编译等待）")
    parser.add_argument("--no-cuda-graph", action="store_true",
                        help="禁用 CUDA Graph（默认开启；compile/TRT 激活时自动禁用）")
    parser.add_argument("--use-tensorrt",  action="store_true",
                        help="启用 TensorRT 加速（首次需构建 Engine，缓存于 .trt_cache/）")
    parser.add_argument("--trt-cache-dir", metavar="DIR",
                        help="TRT Engine 缓存目录（覆盖配置 paths.trt_cache_dir；"
                             "默认 base_dir/.trt_cache）")
    parser.add_argument("--no-fp16",       action="store_true",
                        help="禁用 FP16（默认开启 FP16）")
    parser.add_argument("--crf",           type=int,
                        help="分段输出视频质量 CRF（0~51，默认 23）")
    parser.add_argument("--codec",         type=str,
                        help="分段输出编码器（默认 libx264；有 NVENC 时自动升级）")
    parser.add_argument("--ffmpeg-bin",    type=str,
                        help="ffmpeg 可执行文件路径（默认 ffmpeg）")
    # ── 高优先级覆盖开关（强制启用，覆盖上方 --no-* / config 中的禁用设置）────────
    parser.add_argument("--no-tensorrt", dest="no_tensorrt",
                        action="store_true", default=False,
                        help="[覆盖] 强制禁用 TensorRT，覆盖 --use-tensorrt / config。"
                             "适用于 config 中 use_tensorrt=true 但本次不希望启用 TRT 的场景。")
    parser.add_argument("--use-compile", dest="use_compile_force",
                        action="store_true", default=False,
                        help="[覆盖] 强制启用 torch.compile，覆盖 --no-compile / config。"
                             "与 --use-tensorrt 互斥（TRT 优先）。")
    parser.add_argument("--use-cuda-graph", dest="use_cuda_graph_force",
                        action="store_true", default=False,
                        help="[覆盖] 强制启用 CUDA Graph，覆盖 --no-cuda-graph / config。"
                             "与 compile/TRT 互斥（compile/TRT 优先）。"
                             "如需确保生效，请同时指定 --no-compile --no-tensorrt。")

    # ── 性能报告 ─────────────────────────────────────────────────────────────
    parser.add_argument("--report", metavar="PATH",
                        help="输出 JSON 性能报告路径（含 infer_latency_ms/nvdec/nvenc）")

    # ── 杂项 ─────────────────────────────────────────────────────────────────
    parser.add_argument("--auto-cleanup", action="store_true",
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
    config.set("paths", "input_video",  value=args.input)
    config.set("paths", "output_dir",   value=str(Path(args.output).parent))
    config.set("processing", "batch_mode", value=False)

    if args.upscale_factor:
        config.set("processing", "upscale_factor", value=args.upscale_factor)
    if args.model:
        config.set("models", "realesrgan", "model_name",        value=args.model)
    if args.segment_duration:
        config.set("processing", "segment_duration",            value=args.segment_duration)
    if args.batch_size:
        config.set("models", "realesrgan", "batch_size",        value=args.batch_size)
    if args.prefetch_factor:
        config.set("models", "realesrgan", "prefetch_factor",   value=args.prefetch_factor)
    if args.tile_size is not None:
        config.set("models", "realesrgan", "tile_size",         value=args.tile_size)
    if args.tile_pad is not None:
        config.set("models", "realesrgan", "tile_pad",          value=args.tile_pad)
    if args.pre_pad is not None:
        config.set("models", "realesrgan", "pre_pad",           value=args.pre_pad)
    if args.denoise_strength is not None:
        config.set("models", "realesrgan", "denoise_strength",  value=args.denoise_strength)

    # face_enhance 精细控制
    if args.face_enhance is not None:
        config.set("models", "realesrgan", "face_enhance",      value=args.face_enhance)
    if args.gfpgan_model:
        config.set("models", "realesrgan", "gfpgan_model",      value=args.gfpgan_model)
    if args.gfpgan_weight is not None:
        config.set("models", "realesrgan", "gfpgan_weight",     value=args.gfpgan_weight)
    if args.gfpgan_batch_size:
        config.set("models", "realesrgan", "gfpgan_batch_size", value=args.gfpgan_batch_size)

    # 推理/编码开关（--no-* 时显式写 False，其余留给 __init__ default=True 兜底）
    if args.no_hwaccel:
        config.set("models", "realesrgan", "use_hwaccel",   value=False)
    if args.no_compile:
        config.set("models", "realesrgan", "use_compile",   value=False)
    if args.no_cuda_graph:
        config.set("models", "realesrgan", "use_cuda_graph", value=False)
    if args.use_tensorrt:
        config.set("models", "realesrgan", "use_tensorrt",  value=True)
    if args.trt_cache_dir:
        config.set("paths", "trt_cache_dir", value=args.trt_cache_dir)
    if args.no_fp16:
        config.set("models", "realesrgan", "use_fp16",   value=False)
    if args.crf is not None:
        config.set("models", "realesrgan", "crf",           value=args.crf)
    if args.codec:
        config.set("models", "realesrgan", "codec",         value=args.codec)
    if args.ffmpeg_bin:
        config.set("models", "realesrgan", "ffmpeg_bin",    value=args.ffmpeg_bin)
    if args.report:
        config.set("models", "realesrgan", "report_json",   value=args.report)
    if args.auto_cleanup:
        config.set("processing", "auto_cleanup_temp",       value=True)
    # ── 高优先级覆盖（后写入，覆盖上方 --no-* / config 的值）─────────────────
    _esr_overrides = []
    if args.no_tensorrt and args.use_tensorrt:
        config.set("models", "realesrgan", "use_tensorrt", value=False)
        _esr_overrides.append("--no-tensorrt    覆盖了  --use-tensorrt  → TensorRT 已禁用")
    elif args.no_tensorrt:
        config.set("models", "realesrgan", "use_tensorrt", value=False)
    if args.use_compile_force:
        config.set("models", "realesrgan", "use_compile", value=True)
        if args.no_compile:
            _esr_overrides.append("--use-compile    覆盖了  --no-compile   → torch.compile 已启用")
    if args.use_cuda_graph_force:
        config.set("models", "realesrgan", "use_cuda_graph", value=True)
        if args.no_cuda_graph:
            _esr_overrides.append("--use-cuda-graph 覆盖了  --no-cuda-graph → CUDA Graph 已启用")
    if _esr_overrides:
        print("[CLI覆盖] Real-ESRGAN 以下设置已被高优先级参数覆盖：")
        for msg in _esr_overrides:
            print(f"          · {msg}")
        print()

    # ── 创建处理器并执行 ──────────────────────────────────────────────────────
    try:
        processor = RealESRGANVideoProcessor(config)
    except Exception as e:
        print(f"❌ 初始化处理器失败: {e}")
        return 1

    face_on = config.get("models", "realesrgan", "face_enhance", default=False)
    print(f"\n🎨 Real-ESRGAN v6 独立超分")
    print(f"   输入   : {args.input}")
    print(f"   输出   : {args.output}")
    print(f"   模型   : {processor.model_name}")
    print(f"   倍数   : {processor.upscale_factor}x")
    print(f"   设备   : {processor.device}")
    print(f"   FP16   : {processor.use_fp16} | compile: {processor.use_compile}"
          f" | CUDA Graph: {processor.use_cuda_graph} | TensorRT: {processor.use_tensorrt}")
    print(f"   batch  : {processor.batch_size} | prefetch: {processor.prefetch_factor}")
    if processor.use_tensorrt:
        print(f"   TRT 缓存: {processor.trt_cache_dir or '(自动: base_dir/.trt_cache)'}")
    face_hint = (
        f" (model={processor.gfpgan_model}, weight={processor.gfpgan_weight},"
        f" batch={processor.gfpgan_batch_size})"
        if face_on else ""
    )
    print(f"   face_enhance: {face_on}{face_hint}")

    success = processor.process_video(args.input, args.output)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
