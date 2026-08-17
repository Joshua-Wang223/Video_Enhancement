#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU 视频质量检测脚本：检测花屏、黑屏、白屏、静帧等问题
新增 --laplacian-var 拉普拉斯方差花屏检测，以及 --config 配置文件支持
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


# --------------------------- 工具函数 ---------------------------
def compute_mse(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    err = np.sum((gray1.astype("float") - gray2.astype("float")) ** 2)
    err /= float(gray1.shape[0] * gray1.shape[1])
    return err


def image_entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


# --------------------------- 检测函数 ---------------------------
def is_black_frame(frame, bright_thresh=10, var_thresh=10):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < bright_thresh and np.var(gray) < var_thresh


def is_white_frame(frame, bright_thresh=245, var_thresh=10):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) > bright_thresh and np.var(gray) < var_thresh


def is_still_frame(prev_frame, curr_frame, mse_thresh=1.0):
    mse = compute_mse(prev_frame, curr_frame)
    return mse < mse_thresh, mse


def is_corrupted_blocky(frame, block_size=16, var_thresh=800, ratio_thresh=0.25):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
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


def is_corrupted_laplacian(frame, var_thresh=600):
    """拉普拉斯方差花屏检测（GPU 版同款）"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    var = lap.var()
    return var > var_thresh, var


def is_low_entropy_color(frame, entropy_thresh=3.0):
    ent = image_entropy(frame)
    return ent < entropy_thresh, ent


# --------------------------- 主流程 ---------------------------
def detect_video_quality(video_path, opts):
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
    issues = []

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
        detail = None

        # 1. 黑屏
        if is_black_frame(frame, opts.black_bright, opts.black_var):
            issue_type = "BlackScreen"
        # 2. 白屏
        elif is_white_frame(frame, opts.white_bright, opts.white_var):
            issue_type = "WhiteScreen"
        # 3. 花屏检测（根据参数选择方法）
        elif opts.laplacian_var > 0:
            corrupted, var_val = is_corrupted_laplacian(frame, opts.laplacian_var)
            if corrupted:
                issue_type = "Corruption"
                detail = f"LaplacianVar={var_val:.2f}"
        else:
            corrupted, ratio = is_corrupted_blocky(frame, opts.block_size, opts.var_thresh, opts.ratio_thresh)
            if corrupted:
                low_ent, ent = is_low_entropy_color(frame, opts.entropy_thresh)
                if low_ent:
                    issue_type = "AbnormalSolidColor"
                else:
                    issue_type = "Corruption"
                    detail = f"HighVarBlockRatio={ratio:.3f}"
        # 4. 静帧
        if prev_frame is not None:
            still, mse = is_still_frame(prev_frame, frame, opts.still_mse)
            if still and issue_type is None:
                issue_type = "StillFrame"
                detail = f"MSE={mse:.3f}"

        prev_frame = frame.copy()

        if issue_type is not None:
            info = f"Frame {frame_idx} ({timestamp:.2f}s) - {issue_type}"
            if detail:
                info += f" ({detail})"
            print(info)
            issues.append((timestamp, issue_type))

            if opts.save_dir:
                fname = f"frame_{frame_idx:06d}_{timestamp:.2f}s_{issue_type}.jpg"
                cv2.imwrite(os.path.join(opts.save_dir, fname), frame)

        frame_idx += 1

    cap.release()

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
    parser = argparse.ArgumentParser(description="CPU 视频质量检测（花屏/黑屏/静帧）")

    # 基础参数
    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument("--config", type=str, default=None, help="JSON 配置文件路径（覆盖默认参数）")
    parser.add_argument("--max-frames", type=int, default=0, help="最大检测帧数")
    parser.add_argument("--save-dir", type=str, default=None, help="保存异常帧的目录")

    # 花屏检测 - 分块方差法（默认）
    parser.add_argument("--block-size", type=int, default=16, help="分块大小")
    parser.add_argument("--var-thresh", type=float, default=800.0, help="块方差阈值")
    parser.add_argument("--ratio-thresh", type=float, default=0.25, help="异常块比例阈值")

    # 花屏检测 - 拉普拉斯方差法（新增，默认禁用）
    parser.add_argument("--laplacian-var", type=float, default=0.0,
                        help="拉普拉斯方差阈值，>0 时启用该法并覆盖分块检测")

    # 黑屏 / 白屏
    parser.add_argument("--black-bright", type=float, default=10.0)
    parser.add_argument("--black-var", type=float, default=10.0)
    parser.add_argument("--white-bright", type=float, default=245.0)
    parser.add_argument("--white-var", type=float, default=10.0)

    # 静帧
    parser.add_argument("--still-mse", type=float, default=1.0)

    # 低熵单色帧（仅分块法时使用）
    parser.add_argument("--entropy-thresh", type=float, default=3.0)

    # ---- 加载配置文件 ----
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        with open(temp_args.config, 'r') as f:
            config_data = json.load(f)
        # 禁止通过配置文件设置 config 本身
        config_data.pop('config', None)
        parser.set_defaults(**config_data)

    args = parser.parse_args()

    detect_video_quality(args.video, args)