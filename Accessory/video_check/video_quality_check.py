#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频质量检测脚本：检测花屏、黑屏、白屏、静帧等问题
"""

import argparse
import os
import sys
from math import log2

import cv2
import numpy as np

# --------------------------- 工具函数 ---------------------------
def compute_mse(img1, img2):
    """计算两幅图像(灰度)的均方误差"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    err = np.sum((gray1.astype("float") - gray2.astype("float")) ** 2)
    err /= float(gray1.shape[0] * gray1.shape[1])
    return err

def image_entropy(img):
    """计算彩色图像的整体熵(基于灰度直方图，也可按通道)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    # 去除零值避免log2(0)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def frame_brightness(frame):
    """返回帧的平均亮度（0-255）"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def frame_saturation(frame):
    """返回帧的平均饱和度（HSV的S通道均值）"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 1])

# --------------------------- 检测函数 ---------------------------
def is_black_frame(frame, bright_thresh=10, var_thresh=10):
    """
    检测黑屏：平均亮度极低，且方差极小。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_bright = np.mean(gray)
    var_bright = np.var(gray)
    return mean_bright < bright_thresh and var_bright < var_thresh

def is_white_frame(frame, bright_thresh=245, var_thresh=10):
    """
    检测白屏：平均亮度极高，且方差极小。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_bright = np.mean(gray)
    var_bright = np.var(gray)
    return mean_bright > bright_thresh and var_bright < var_thresh

def is_still_frame(prev_frame, curr_frame, mse_thresh=1.0):
    """
    检测静止帧（卡顿）：相邻帧几乎完全相同。
    返回静止程度(True/False)以及MSE值。
    """
    mse = compute_mse(prev_frame, curr_frame)
    return mse < mse_thresh, mse

def is_corrupted_blocky(frame, block_size=16, var_thresh=800, ratio_thresh=0.25):
    """
    花屏/乱帧检测：基于块的高方差比例。
    原理：正常图像局部方差适中，花屏区域通常像素值随机跳变，局部方差极高。
    将帧分成 block_size 大小的块，统计方差超过 var_thresh 的块的比例。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # 分块
    blocks = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = gray[y:min(y+block_size, h), x:min(x+block_size, w)]
            if block.size > 0:
                blocks.append(np.var(block))
    if not blocks:
        return False, 0.0
    high_var_blocks = sum(1 for v in blocks if v > var_thresh)
    ratio = high_var_blocks / len(blocks)
    return ratio > ratio_thresh, ratio

def is_low_entropy_color(frame, entropy_thresh=3.0):
    """
    颜色异常单一检测：计算图像熵，极低的熵可能指示大面积纯色或花屏中的单调区域。
    但需结合其他条件。
    """
    ent = image_entropy(frame)
    return ent < entropy_thresh, ent

# --------------------------- 主检测逻辑 ---------------------------
def detect_video_quality(video_path, opts):
    """
    主函数：遍历视频，输出异常时间点。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频帧率: {fps:.2f} fps, 总帧数: {total_frames}")

    if opts.max_frames > 0:
        total_frames = min(total_frames, opts.max_frames)

    prev_frame = None
    frame_idx = 0
    issues = []  # 存储 (时间戳, 类型, 详情)

    # 如果需要保存异常帧，创建输出目录
    if opts.save_dir:
        os.makedirs(opts.save_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= total_frames:
            break

        timestamp = frame_idx / fps if fps > 0 else frame_idx
        issue_type = None

        # 1. 黑屏检测
        if is_black_frame(frame, opts.black_bright, opts.black_var):
            issue_type = "BlackScreen"

        # 2. 白屏检测
        if issue_type is None and is_white_frame(frame, opts.white_bright, opts.white_var):
            issue_type = "WhiteScreen"

        # 3. 花屏/乱帧检测 (组合规则)
        corrupted, ratio = is_corrupted_blocky(frame, opts.block_size, opts.var_thresh, opts.ratio_thresh)
        if corrupted and issue_type is None:
            # 进一步排除极端熵情况？花屏常伴随一定程度的纹理，但熵不一定很低。
            # 可加颜色熵辅助判断：如果熵极低且不是黑/白屏，可能是单色花屏
            low_ent, ent = is_low_entropy_color(frame, opts.entropy_thresh)
            if low_ent:
                # 低熵且不是黑/白屏，可能为异常单色帧（绿屏、蓝屏等）
                issue_type = "AbnormalSolidColor"
            else:
                issue_type = "Corruption"

        # 4. 静帧检测
        if prev_frame is not None:
            still, mse = is_still_frame(prev_frame, frame, opts.still_mse)
            if still:
                # 如果当前已经是黑屏/白屏/花屏，则优先报告那些问题，静帧可能伴随。
                if issue_type is None:
                    issue_type = "StillFrame"
                # 也可以记录连续静帧的开始结束，这里简单每帧都报
        # 更新前一帧
        prev_frame = frame.copy()

        # 如果发现问题，记录并可选保存图像
        if issue_type is not None:
            info = f"Frame {frame_idx} ({timestamp:.2f}s) - {issue_type}"
            if issue_type == "Corruption":
                info += f" (high-var block ratio: {ratio:.3f})"
            print(info)
            issues.append((timestamp, issue_type))

            # 保存异常帧
            if opts.save_dir:
                fname = f"frame_{frame_idx:06d}_{timestamp:.2f}s_{issue_type}.jpg"
                cv2.imwrite(os.path.join(opts.save_dir, fname), frame)

        frame_idx += 1

    cap.release()

    # 输出汇总
    print("\n===== 检测完成 =====")
    print(f"总检查帧数: {frame_idx}")
    if issues:
        print(f"发现 {len(issues)} 处异常:")
        for t, tp in issues:
            print(f"  时间 {t:.2f}s, 类型: {tp}")
    else:
        print("未发现异常。")

    return issues

# --------------------------- 命令行接口 ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频质量检测脚本：检测花屏、黑屏、白屏、静帧等问题")
    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument("--max-frames", type=int, default=0, help="最大检测帧数，0表示全部")
    parser.add_argument("--block-size", type=int, default=16, help="花屏检测分块大小 (默认16)")
    parser.add_argument("--var-thresh", type=float, default=800.0, help="块方差阈值，超过视为异常块 (默认800)")
    parser.add_argument("--ratio-thresh", type=float, default=0.25, help="异常块比例阈值 (默认0.25)")
    parser.add_argument("--black-bright", type=float, default=10.0, help="黑屏亮度上限 (默认10)")
    parser.add_argument("--black-var", type=float, default=10.0, help="黑屏方差上限 (默认10)")
    parser.add_argument("--white-bright", type=float, default=245.0, help="白屏亮度下限 (默认245)")
    parser.add_argument("--white-var", type=float, default=10.0, help="白屏方差上限 (默认10)")
    parser.add_argument("--still-mse", type=float, default=1.0, help="静帧MSE阈值 (默认1.0)")
    parser.add_argument("--entropy-thresh", type=float, default=3.0, help="低熵阈值，用于识别单色异常 (默认3.0)")
    parser.add_argument("--save-dir", type=str, default=None, help="保存异常帧的目录（可选）")
    args = parser.parse_args()

    detect_video_quality(args.video, args)