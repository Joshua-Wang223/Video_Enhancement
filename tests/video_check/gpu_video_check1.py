#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 加速视频质量检测脚本（最高实时性能方案）
支持 --config 配置文件，以及 --laplacian-var 花屏阈值
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np


# --------------------------- GPU 工具函数 ---------------------------
def gpu_available():
    try:
        _ = cv2.cuda.getCudaEnabledDeviceCount()
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except:
        return False


def has_gpu_decoder():
    try:
        _ = cv2.cudacodec.VideoCapture()
        return True
    except:
        return False


# --------------------------- GPU 检测类 ---------------------------
class GPUFrameChecker:
    def __init__(self, opts):
        self.black_bright = opts.black_bright
        self.black_var = opts.black_var
        self.white_bright = opts.white_bright
        self.white_var = opts.white_var
        self.still_mse = opts.still_mse
        self.laplacian_var_thresh = opts.laplacian_var

    def _upload(self, frame):
        if isinstance(frame, np.ndarray):
            gpu = cv2.cuda_GpuMat()
            gpu.upload(frame)
            return gpu
        return frame

    def is_black_frame(self, gpu_frame):
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
        mean_mat, stddev_mat = cv2.cuda.meanStdDev(gpu_gray)
        mean_val = mean_mat.download()[0][0]
        var_val = stddev_mat.download()[0][0] ** 2
        return mean_val < self.black_bright and var_val < self.black_var

    def is_white_frame(self, gpu_frame):
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
        mean_mat, stddev_mat = cv2.cuda.meanStdDev(gpu_gray)
        mean_val = mean_mat.download()[0][0]
        var_val = stddev_mat.download()[0][0] ** 2
        return mean_val > self.white_bright and var_val < self.white_var

    def is_still(self, prev_gpu, curr_gpu):
        diff = cv2.cuda.absdiff(prev_gpu, curr_gpu)
        diff_gray = cv2.cuda.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        mean_mat, _ = cv2.cuda.meanStdDev(diff_gray)
        mad = mean_mat.download()[0][0]
        return mad < self.still_mse

    def is_corrupted_laplacian(self, gpu_frame):
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
        gpu_lap = cv2.cuda.Laplacian(gpu_gray, cv2.CV_32F, ksize=3)
        _, stddev_mat = cv2.cuda.meanStdDev(gpu_lap)
        stddev_val = stddev_mat.download()[0][0]
        var_lap = stddev_val ** 2
        return var_lap > self.laplacian_var_thresh, var_lap

    def check_frame(self, frame, prev_frame=None):
        gpu_frame = self._upload(frame)
        prev_gpu = self._upload(prev_frame) if prev_frame is not None else None

        if self.is_black_frame(gpu_frame):
            return "BlackScreen", None
        if self.is_white_frame(gpu_frame):
            return "WhiteScreen", None

        corrupted, var_lap = self.is_corrupted_laplacian(gpu_frame)
        if corrupted:
            return "Corruption", f"LaplacianVar={var_lap:.2f}"

        if prev_gpu is not None and self.is_still(prev_gpu, gpu_frame):
            return "StillFrame", None

        return None, None


# --------------------------- 主流程 ---------------------------
def detect_video_gpu(video_path, opts):
    if not gpu_available():
        print("错误：未检测到支持 CUDA 的 GPU，或 OpenCV 未编译 CUDA 模块。")
        sys.exit(1)

    print("GPU 可用，设备数量:", cv2.cuda.getCudaEnabledDeviceCount())

    use_gpu_decoder = has_gpu_decoder()
    if use_gpu_decoder:
        print("使用 GPU 硬件解码 (cudacodec)")
        cap = cv2.cudacodec.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
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
        fps = 30.0
    print(f"视频帧率: {fps:.2f} fps, 总帧数: {total_frames}")

    if opts.max_frames > 0:
        total_frames = min(total_frames, opts.max_frames)

    checker = GPUFrameChecker(opts)
    prev_frame_gpu = None
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

        timestamp = frame_idx / fps
        issue_type, detail = checker.check_frame(frame, prev_frame_gpu)

        if issue_type is not None:
            info = f"Frame {frame_idx} ({timestamp:.2f}s) - {issue_type}"
            if detail:
                info += f" ({detail})"
            print(info)
            issues.append((timestamp, issue_type))

            if opts.save_dir:
                cpu_frame = frame.download() if isinstance(frame, cv2.cuda.GpuMat) else frame
                fname = f"frame_{frame_idx:06d}_{timestamp:.2f}s_{issue_type}.jpg"
                cv2.imwrite(os.path.join(opts.save_dir, fname), cpu_frame)

        if isinstance(frame, cv2.cuda.GpuMat):
            prev_frame_gpu = frame.clone()
        else:
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
    parser = argparse.ArgumentParser(description="GPU 加速视频质量检测（花屏/黑屏/静帧）")

    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument("--config", type=str, default=None, help="JSON 配置文件路径")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default=None)

    parser.add_argument("--black-bright", type=float, default=10.0)
    parser.add_argument("--black-var", type=float, default=10.0)
    parser.add_argument("--white-bright", type=float, default=245.0)
    parser.add_argument("--white-var", type=float, default=10.0)
    parser.add_argument("--still-mse", type=float, default=1.0)
    parser.add_argument("--laplacian-var", type=float, default=600.0)

    # 加载配置文件
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        with open(temp_args.config, 'r') as f:
            config_data = json.load(f)
        config_data.pop('config', None)
        parser.set_defaults(**config_data)

    args = parser.parse_args()

    detect_video_gpu(args.video, args)