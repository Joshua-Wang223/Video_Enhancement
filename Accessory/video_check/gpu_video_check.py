#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 加速视频质量检测脚本（最高实时性能方案）
检测：黑屏、白屏、花屏（拉普拉斯方差异常）、静帧
优先使用 GPU 硬件解码 (NVDEC)，失败自动回退 CPU 解码 + GPU 处理
"""

import argparse
import os
import sys
import time
from math import log2

import cv2
import numpy as np

# --------------------------- GPU 工具函数 ---------------------------
def gpu_available():
    """检查 GPU 可用性"""
    try:
        _ = cv2.cuda.getCudaEnabledDeviceCount()
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except:
        return False

def has_gpu_decoder():
    """检查是否可用 cudacodec（硬件解码）"""
    try:
        # 尝试创建一个 cudacodec 解码器对象，看是否抛出异常
        dummy = cv2.cudacodec.VideoCapture()
        return True
    except:
        return False

# --------------------------- GPU 加速检测函数 ---------------------------
class GPUFrameChecker:
    """
    将检测逻辑全部放在 GPU 上，输入统一为 cv2.cuda.GpuMat
    """
    def __init__(self, opts):
        self.black_bright = opts.black_bright
        self.black_var = opts.black_var
        self.white_bright = opts.white_bright
        self.white_var = opts.white_var
        self.still_mse = opts.still_mse
        self.laplacian_var_thresh = opts.laplacian_var  # 花屏阈值

    def _upload_if_needed(self, frame):
        """若传入 numpy 数组，自动上传至 GPU"""
        if isinstance(frame, np.ndarray):
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(frame)
            return gpu_mat
        return frame  # 已经是 GpuMat

    def is_black_frame(self, gpu_frame):
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
        mean_mat, stddev_mat = cv2.cuda.meanStdDev(gpu_gray)
        mean_val = mean_mat.download()[0][0]
        stddev_val = stddev_mat.download()[0][0]
        var_val = stddev_val ** 2
        return mean_val < self.black_bright and var_val < self.black_var

    def is_white_frame(self, gpu_frame):
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
        mean_mat, stddev_mat = cv2.cuda.meanStdDev(gpu_gray)
        mean_val = mean_mat.download()[0][0]
        stddev_val = stddev_mat.download()[0][0]
        var_val = stddev_val ** 2
        return mean_val > self.white_bright and var_val < self.white_var

    def is_still(self, prev_gpu, curr_gpu):
        """计算 MSE，GPU 上完成"""
        diff = cv2.cuda.absdiff(prev_gpu, curr_gpu)
        diff_gray = cv2.cuda.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # 使用 meanStdDev 得到平均值，MSE = mean**2 + stddev**2？ 不是直接 MSE。
        # 更准确：sum(abs(diff)) / 像素数
        # 但这里我们采用近似：平均绝对差 (MAD) 代替 MSE，阈值相应调整。
        mean_mat, stddev_mat = cv2.cuda.meanStdDev(diff_gray)
        mad = mean_mat.download()[0][0]  # 平均绝对差
        # 静态帧阈值 (mad < threshold)
        return mad < self.still_mse

    def is_corrupted_laplacian(self, gpu_frame):
        """基于拉普拉斯方差的花屏检测"""
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
        # 拉普拉斯算子 (3x3)
        gpu_lap = cv2.cuda.Laplacian(gpu_gray, cv2.CV_32F, ksize=3)
        # 计算拉普拉斯图的方差
        mean_mat, stddev_mat = cv2.cuda.meanStdDev(gpu_lap)
        stddev_val = stddev_mat.download()[0][0]
        var_lap = stddev_val ** 2
        # 花屏通常拉普拉斯方差极高
        return var_lap > self.laplacian_var_thresh, var_lap

    def check_frame(self, frame, prev_frame=None):
        """
        对单帧进行全面检测，返回 (issue_type, detail)
        frame 可以是 numpy array 或 GpuMat
        """
        gpu_frame = self._upload_if_needed(frame)
        if prev_frame is not None:
            prev_gpu = self._upload_if_needed(prev_frame)
        else:
            prev_gpu = None

        # 1. 黑屏
        if self.is_black_frame(gpu_frame):
            return "BlackScreen", None
        # 2. 白屏
        if self.is_white_frame(gpu_frame):
            return "WhiteScreen", None
        # 3. 花屏 (拉普拉斯方差)
        corrupted, var_lap = self.is_corrupted_laplacian(gpu_frame)
        if corrupted:
            return "Corruption", f"LaplacianVar={var_lap:.2f}"
        # 4. 静帧
        if prev_gpu is not None and self.is_still(prev_gpu, gpu_frame):
            return "StillFrame", None
        return None, None

# --------------------------- 主流程 ---------------------------
def detect_video_gpu(video_path, opts):
    if not gpu_available():
        print("错误：未检测到支持 CUDA 的 GPU，或 OpenCV 未编译 CUDA 模块。")
        sys.exit(1)

    print("GPU 可用，设备数量:", cv2.cuda.getCudaEnabledDeviceCount())

    # 尝试使用 GPU 硬件解码
    use_gpu_decoder = has_gpu_decoder()
    if use_gpu_decoder:
        print("使用 GPU 硬件解码 (cudacodec)")
        cap = cv2.cudacodec.VideoCapture(video_path)
        # 获取视频属性（cudacodec 可能不支持 get()，使用 cv2.CAP_PROP_* 也可能部分有效）
        # 我们用 fallback 方式获取 fps 和总帧数
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            # 回退使用 CPU 解码器获取元数据
            cpu_cap = cv2.VideoCapture(video_path)
            fps = cpu_cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cpu_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cpu_cap.release()
    else:
        print("cudacodec 不可用，使用 CPU 解码 + GPU 处理")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0  # 默认假设

    print(f"视频帧率: {fps:.2f} fps, 总帧数: {total_frames}")
    if opts.max_frames > 0:
        total_frames = min(total_frames, opts.max_frames)

    checker = GPUFrameChecker(opts)
    prev_frame_gpu = None  # 上一帧的 GpuMat（或 numpy，会在内部上传）
    frame_idx = 0
    issues = []
    t_start = time.time()

    if opts.save_dir:
        os.makedirs(opts.save_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= total_frames:
            break

        # frame 类型：cudacodec 返回的是 GpuMat，CPU 解码是 numpy
        timestamp = frame_idx / fps
        issue_type, detail = checker.check_frame(frame, prev_frame_gpu)

        if issue_type is not None:
            info = f"Frame {frame_idx} ({timestamp:.2f}s) - {issue_type}"
            if detail:
                info += f" ({detail})"
            print(info)
            issues.append((timestamp, issue_type))

            # 保存异常帧（需要下载到 CPU）
            if opts.save_dir:
                if isinstance(frame, cv2.cuda.GpuMat):
                    cpu_frame = frame.download()
                else:
                    cpu_frame = frame
                fname = f"frame_{frame_idx:06d}_{timestamp:.2f}s_{issue_type}.jpg"
                cv2.imwrite(os.path.join(opts.save_dir, fname), cpu_frame)

        # 更新前一帧（在 GPU 上保留一份，避免多次上传）
        if isinstance(frame, cv2.cuda.GpuMat):
            prev_frame_gpu = frame.clone()  # 深拷贝
        else:
            # 上传到 GPU 并保存
            gpu = cv2.cuda_GpuMat()
            gpu.upload(frame)
            prev_frame_gpu = gpu

        frame_idx += 1

    cap.release()
    elapsed = time.time() - t_start
    print("\n===== 检测完成 =====")
    print(f"总检查帧数: {frame_idx}, 耗时: {elapsed:.2f} 秒")
    if issues:
        print(f"发现 {len(issues)} 处异常:")
        for t, tp in issues:
            print(f"  时间 {t:.2f}s, 类型: {tp}")
    else:
        print("未发现异常。")

    return issues

# --------------------------- 命令行接口 ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU 加速视频质量检测 (花屏/黑屏/静帧)")
    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument("--max-frames", type=int, default=0, help="最大检测帧数，0 表示全部")
    parser.add_argument("--black-bright", type=float, default=10.0, help="黑屏亮度阈值")
    parser.add_argument("--black-var", type=float, default=10.0, help="黑屏方差阈值")
    parser.add_argument("--white-bright", type=float, default=245.0, help="白屏亮度阈值")
    parser.add_argument("--white-var", type=float, default=10.0, help="白屏方差阈值")
    parser.add_argument("--still-mse", type=float, default=1.0, help="静帧平均绝对差阈值 (MAD)")
    parser.add_argument("--laplacian-var", type=float, default=600.0,
                        help="拉普拉斯方差阈值，超过视为花屏 (默认 600)")
    parser.add_argument("--save-dir", type=str, default=None, help="保存异常帧的目录")
    args = parser.parse_args()

    detect_video_gpu(args.video, args)