#!/usr/bin/env python
"""scene_cut_fix_verify.py —— 校验切镜鬼影修复效果（不依赖包内路径）。

对 (src, out) 一对文件：
  1. 在 src 上做 ffmpeg scene 检测，得到切镜源帧号集合 C
  2. 对每个 c ∈ C，计算插值帧 out[2c-1] 的 PSNR：
       · vs src[c]   （修复后应 ≈ 复制，PSNR 高，判定阈值 ≥30dB）
       · vs src[c-1] （修复前是鬼影，PSNR 低，作为对照）
  3. 校验帧数守恒 out 帧数 == 2N-1

用法:
  python Accessory/verify/scene_cut_fix_verify.py <src.mp4> <out.mp4>
"""
import sys

import numpy as np
from PIL import Image

from diagnose_scene_cut_ghost import grab, psnr   # 复用抽取与 PSNR


def cuts_of(src):
    import re
    import subprocess
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tf:
        meta_path = tf.name
    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-i", src,
                        "-vf", "select='gt(scene,0.25)',metadata=print:file=" + meta_path,
                        "-f", "null", "-"], capture_output=True, text=True)
        meta = open(meta_path, errors='replace').read()
    finally:
        try:
            os.unlink(meta_path)
        except OSError:
            pass
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", src],
                       capture_output=True, text=True)
    num, den = (r.stdout or "30/1").strip().split("/")[:2]
    fps = float(num) / float(den) if float(den) else 30.0
    cuts = sorted({int(round(float(m.group(1)) * fps))
                   for m in re.finditer(r"pts_time:([0-9.]+)", meta)})
    return [c for c in cuts if c >= 1], fps


def main():
    import subprocess
    import tempfile
    src, out = sys.argv[1], sys.argv[2]
    tmp = tempfile.mkdtemp(prefix="verify_")

    def nframes(p):
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-count_frames", "-show_entries", "stream=nb_read_frames",
                            "-of", "csv=p=0", p], capture_output=True, text=True)
        return int((r.stdout or "0").strip() or 0)

    n_src, n_out = nframes(src), nframes(out)
    expect = 2 * n_src - 1
    print(f"src 帧数 = {n_src}   out 帧数 = {n_out}   期望 2N-1 = {expect}  "
          f"{'✅ 守恒' if n_out == expect else '❌ 不守恒'}")

    cuts, fps = cuts_of(src)
    print(f"检测到 {len(cuts)} 处切镜（源帧号）: {cuts}\n")
    print(f"{'切镜src[c]':>10} {'out[2c-1]':>10} {'PSNR vs src[c]':>15} "
          f"{'PSNR vs src[c-1]':>17}  判定")
    ghost = 0
    for c in cuts:
        o = grab(out, 2 * c - 1, f"o{2*c-1}", tmp)
        p_new = psnr(o, grab(src, c, f"s{c}", tmp))        # 修复后应匹配 src[c]
        p_old = psnr(o, grab(src, c - 1, f"s{c-1}", tmp))  # 鬼影参照
        ok = p_new >= 30.0
        ghost += (not ok)
        print(f"{c:10d} {2*c-1:10d} {p_new:15.2f} {p_old:17.2f}  "
              f"{'✅ 已修复（=源帧副本）' if ok else '❌ 仍是鬼影'}")
    print(f"\n→ 鬼影残留: {ghost}/{len(cuts)}  "
          f"{'✅ 全部消除' if ghost == 0 else '❌ 未完全修复'}")


if __name__ == "__main__":
    main()
