#!/usr/bin/env python
"""diagnose_scene_cut_ghost.py —— 扫描分段所有切镜处的插值帧质量，量化鬼影总量。

实测结论（本仓库）：每一个硬切镜都恰好产生一张「水彩画」鬼影插值帧
（segment_000 16/16、segment_001 15/15、segment_002 19/19），
而源帧位 out[2k] 始终守恒。故「鬼影总量 ≈ 切镜数」。

用法:
  python tests/diagnose_scene_cut_ghost.py 1        # 扫 r2_artifacts 的 segment_001
  python tests/diagnose_scene_cut_ghost.py 1 20 --pkg /path/to/artifacts
  python tests/diagnose_scene_cut_ghost.py --src a.mp4 --out b.mp4
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

PKG = os.environ.get(
    "R2_ARTIFACTS", "/workspace/retest_package_20260904/r2_artifacts")


def grab(mp4, idx, tag, tmp):
    out = os.path.join(tmp, tag + ".png")
    r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", mp4,
                        "-vf", "select=eq(n\\,%d)" % idx, "-vsync", "0",
                        "-frames:v", "1", out], capture_output=True)
    if r.returncode != 0 or not os.path.exists(out):
        return None
    return np.asarray(Image.open(out).convert("RGB"))


def psnr(a, b):
    if a is None or b is None or a.shape != b.shape:
        return -1
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = (d ** 2).mean()
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seg", type=int, nargs="?", default=None)
    ap.add_argument("thr", type=float, nargs="?", default=20.0)
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--src", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    seg, thr = a.seg, a.thr
    tmp = tempfile.mkdtemp(prefix="cutscan_")
    if a.src and a.out:
        src_m, out_m = a.src, a.out
    elif seg is not None:
        src_m = "%s/segments/segment_%03d.mp4" % (a.pkg, seg)
        out_m = "%s/processed/interpolated_segment_%03d.mp4" % (a.pkg, seg)
    else:
        ap.error("必须提供 --src/--out，或分段号")
    tag = "seg%03d" % seg if seg is not None else os.path.basename(src_m)
    cf = "/tmp/cuts_%s.txt" % re.sub(r"[^0-9A-Za-z_.-]", "_", tag)
    subprocess.run(["ffmpeg", "-hide_banner", "-i", src_m,
                    "-vf", "select='gt(scene,0.25)',metadata=print:file=" + cf,
                    "-f", "null", "-"], capture_output=True)
    # 用实际帧率换算，避免硬编码 30fps
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", src_m],
                       capture_output=True, text=True)
    num, den = (r.stdout or "30/1").strip().split("/")[:2]
    fps = float(num) / float(den) if float(den) else 30.0
    cuts = [round(float(m.group(1)) * fps)
            for m in re.finditer(r"pts_time:([0-9.]+)", open(cf).read())]
    print("%s: %d 个切镜，扫描插值帧 out[2c-1]（阈值 PSNR<%.0f, fps=%.3f）"
          % (tag, len(cuts), thr, fps))
    print("%12s %14s %18s  判定" % ("切镜src[c]", "out[2c-1]帧号", "PSNR vs src[c-1]"))
    bad = 0
    for c in cuts:
        p = psnr(grab(out_m, 2 * c - 1, "o%d" % (2 * c - 1), tmp),
                 grab(src_m, c - 1, "s%d" % (c - 1), tmp))
        isbad = p < thr
        bad += isbad
        print("%12d %14d %18.2f  %s" % (c, 2 * c - 1, p, "❌ 鬼影" if isbad else "✅ 正常"))
    print("\n→ 切镜处鬼影帧数: %d / %d" % (bad, len(cuts)))


if __name__ == "__main__":
    main()
