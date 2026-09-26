#!/usr/bin/env python
"""scene_cut_threshold_calibrator.py —— 用已知切镜 ground truth 标定切镜判定阈值。

**结论（重要）**：单纯「相邻帧 MAD」无法分离切镜与快速运动——
segment_000 切镜处 MAD 低至 0.071，而普通快速运动帧高达 0.186，
满召回时误报 63 个。误报的代价（正常运动帧被复制成重复帧→可见卡顿）
比漏报（保留鬼影）更糟。因此生产改用 ffmpeg 的 scene 滤镜做预扫描，
本脚本保留用于说明该结论与后续阈值研究。


ground truth（ffmpeg scene>0.25，0-based 源帧号 c，表示 src[c-1]→src[c] 之间是切镜）:
  seg000: 282 287 292 297 302 307 312 317 322 1308 1309 1384 1697 2087 5210 8323
  seg001: 207 969 1030 1543 1682 2326 2409 2794 3414 3633 3808 6408 8405 8468 8521

输出：在 0 误报前提下召回率最高的阈值区间。
"""
import os
import subprocess
import sys

import numpy as np

PKG = os.environ.get(
    "R2_ARTIFACTS", "/workspace/retest_package_20260904/r2_artifacts")
GT = {
    0: [282, 287, 292, 297, 302, 307, 312, 317, 322, 1308, 1309, 1384, 1697, 2087, 5210, 8323],
    1: [207, 969, 1030, 1543, 1682, 2326, 2409, 2794, 3414, 3633, 3808, 6408, 8405, 8468, 8521],
}


def load_gray(seg, w=48, h=36):
    """解码为 w×h 灰度 rawvideo，返回 (N, h*w) uint8。"""
    cmd = ["ffmpeg", "-v", "error", "-i", f"{PKG}/segments/segment_{seg:03d}.mp4",
           "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    r = subprocess.run(cmd, capture_output=True)
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    n = a.size // (w * h)
    return a[:n * (w * h)].reshape(n, w * h)


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 1   # 空间下采样步长（模拟更粗的采样）
    for seg in (0, 1):
        g = load_gray(seg)
        if stride > 1:
            g = g.reshape(-1, 36, 48)[:, ::stride, ::stride].reshape(len(g), -1)
        f = g.astype(np.float32)
        mad = np.abs(f[1:] - f[:-1]).mean(axis=1) / 255.0
        # mad[i] 对应 src[i] → src[i+1] 这对帧（即 pair bi=(src[i], src[i+1])）
        gt = set(c - 1 for c in GT[seg])     # pair 索引
        print(f"=== segment_{seg:03d}  帧数={len(g)}  GT 切镜 pair 数={len(gt)} ===")
        pos = mad[np.array(sorted(gt))]
        mask = np.ones(len(mad), dtype=bool)
        mask[np.array(sorted(gt))] = False
        neg = mad[mask]
        print(f"  切镜处  MAD: min={pos.min():.4f} max={pos.max():.4f}")
        print(f"  非切镜  MAD: mean={neg.mean():.4f} p99={np.percentile(neg,99):.4f} "
              f"p999={np.percentile(neg,99.9):.4f} max={neg.max():.4f}")
        best = None
        for th in np.arange(0.02, 0.60, 0.005):
            tp = int((pos >= th).sum())
            fp = int((neg >= th).sum())
            if fp == 0 and tp == len(pos):
                if best is None:
                    best = th
                last = th
        if best is not None:
            print(f"  ✅ 零误报且满召回的阈值区间: [{best:.3f}, {last:.3f}]  (建议取中值 {(best+last)/2:.3f})")
        else:
            # 找误报最少的满召回点
            cand = [(int((neg >= th).sum()), th) for th in np.arange(0.02, 0.60, 0.005)
                    if int((pos >= th).sum()) == len(pos)]
            if cand:
                fp, th = min(cand)
                print(f"  ⚠️ 无零误报区间；满召回下最小误报 = {fp} @ th={th:.3f}")
            else:
                print("  ❌ 该采样步长下无法同时满召回")


if __name__ == "__main__":
    main()
