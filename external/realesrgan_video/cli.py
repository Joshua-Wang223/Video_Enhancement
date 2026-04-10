#!/usr/bin/env python3
import argparse

def parse_args():
    """主函数 - 参数解析 (100%保留原版CLI)"""
    parser = argparse.ArgumentParser(
        description='Real-ESRGAN 视频超分 —— 架构优化版 v6.4',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 基础参数
    parser.add_argument('-i', '--input', type=str, required=True, help='输入视频路径')
    parser.add_argument('-o', '--output', type=str, required=True, help='输出视频路径')
    parser.add_argument('-n', '--model-name', type=str, default='realesr-animevideov3', help='模型名称')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='输出缩放比例')
    parser.add_argument('-dn', '--denoise-strength', type=float, default=0.5, help='去噪强度')
    parser.add_argument('--suffix', type=str, default='out', help='输出文件后缀')
    # 优化参数
    parser.add_argument('--batch-size', type=int, default=6, help='批处理大小（优化版本推荐6-8）')
    parser.add_argument('--prefetch-factor', type=int, default=48, help='读帧预取队列深度')
    # 人脸增强参数
    parser.add_argument('--face-enhance', action='store_true', help='启用人脸增强')
    parser.add_argument('--gfpgan-model', type=str, default='1.4', choices=['1.3', '1.4', 'RestoreFormer'],
                        help='GFPGAN 模型版本')
    parser.add_argument('--gfpgan-weight', type=float, default=0.5, help='GFPGAN 增强融合权重')
    parser.add_argument('--gfpgan-batch-size', type=int, default=8, help='GFPGAN 单次最多处理人脸数')
    # FIX-DET-THRESHOLD: 人脸检测置信度阈值参数
    parser.add_argument('--face-det-threshold', type=float, default=0.5,
                        help='人脸检测置信度阈值 [0.0-1.0]，越高过滤越多低质量检测。'
                             '0.5=保留多数人脸，0.7=过滤模糊远景，0.9=仅保留清晰人脸')
    # FIX-ADAPTIVE-BATCH: 自适应批处理开关
    parser.add_argument('--adaptive-batch', action='store_true', default=True,
                        help='启用基于人脸密度的自适应批处理大小（默认开启）')
    parser.add_argument('--no-adaptive-batch', action='store_true', default=False,
                        help='禁用自适应批处理大小（使用固定 --batch-size）')
    # 加速参数
    parser.add_argument('--no-fp16', action='store_true', help='禁用 FP16')
    parser.add_argument('--no-compile', action='store_true', help='禁用 torch.compile')
    parser.add_argument('--use-tensorrt', action='store_true', help='启用 SR TensorRT 加速')
    parser.add_argument('--gfpgan-trt', action='store_true', help='GFPGAN TensorRT 加速')
    parser.add_argument('--no-cuda-graph', action='store_true', help='禁用 CUDA Graph')
    # 硬件加速参数
    parser.add_argument('--use-hwaccel', action='store_true', default=True, help='启用 NVDEC')
    parser.add_argument('--no-hwaccel', action='store_true', help='禁用 NVDEC')
    # 其他参数
    parser.add_argument('-t', '--tile', type=int, default=0, help='分块大小')
    parser.add_argument('--tile-pad', type=int, default=10, help='分块填充')
    parser.add_argument('--pre-pad', type=int, default=0, help='预填充')
    parser.add_argument('--fps', type=float, default=None, help='输出帧率')
    # 编码参数
    parser.add_argument('--video-codec', type=str, default='libx264',
                        choices=['libx264', 'libx265', 'libvpx-vp9', 'libvpx-vp9', 'h264_nvenc'],
                        help='偏好编码器')
    parser.add_argument('--crf', type=int, default=23, help='编码质量')
    parser.add_argument('--x264-preset', type=str, default='medium',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast',
                                 'medium', 'slow', 'slower', 'veryslow'],
                        help='libx264/libx265 preset')
    parser.add_argument('--ffmpeg-bin', type=str, default='ffmpeg', help='FFmpeg 二进制路径')
    
    args = parser.parse_args()
    if args.no_adaptive_batch:
        args.adaptive_batch = False
        print("已禁用自适应批处理大小（使用固定 --batch-size）")
    return args