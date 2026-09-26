#!/usr/bin/env python
"""interp_ghost_analyzer.py —— 定位分段插帧产物的「水彩画样润开」帧来源。

核心判定（帧守恒映射 2x）：
    out[2k]   必须等于源帧 src[k]   （重复的原始帧）
    out[2k+1] 是 IFRNet 插值帧
若 out[2k] 与 src[k] 差异巨大 → 源帧被替换成鬼影 = 管线 bug
若鬼影只出现在 out[2k+1]        → IFRNet 插值伪影（切镜处正常）

产物根目录由 --pkg 指定，默认取环境变量 R2_ARTIFACTS，
再退到 /workspace/retest_package_20260904/r2_artifacts。
也可用 --src/--out 直接指定任意一对文件。

用法:
  python Accessory/analyze/interp_ghost_analyzer.py --seg 1 --around 414,2060,1938
  python Accessory/analyze/interp_ghost_analyzer.py --seg 0 --around 10420 --scan 0:600:2
  python Accessory/analyze/interp_ghost_analyzer.py --src a.mp4 --out b.mp4 --around 413
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

PKG = os.environ.get(
    "R2_ARTIFACTS", "/workspace/retest_package_20260904/r2_artifacts")


def grab(mp4, idx, tmpdir, tag):
    """用 ffmpeg select 按帧号抽帧，返回 BGR ndarray。"""
    out = os.path.join(tmpdir, f"{tag}.png")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", mp4, "-vf", f"select=eq(n\\,{idx})", "-vsync", "0",
           "-frames:v", "1", out]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not os.path.exists(out):
        return None
    try:
        import cv2
        return cv2.imread(out)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(out).convert("RGB"))[:, :, ::-1].copy()


def psnr(a, b):
    if a is None or b is None or a.shape != b.shape:
        return -1.0
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float((d ** 2).mean())
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=None)
    ap.add_argument("--pkg", default=PKG, help="产物根目录（含 segments/ 与 processed/）")
    ap.add_argument("--src", default="", help="直接指定源分段（优先于 --seg）")
    ap.add_argument("--out", default="", help="直接指定插帧产物（优先于 --seg）")
    ap.add_argument("--around", default="")
    ap.add_argument("--scan", default="", help="start:end:step 对 out[2k] vs src[k] 做守恒扫描")
    args = ap.parse_args()

    if args.src and args.out:
        src, out = args.src, args.out
    elif args.seg is not None:
        src = f"{args.pkg}/segments/segment_{args.seg:03d}.mp4"
        out = f"{args.pkg}/processed/interpolated_segment_{args.seg:03d}.mp4"
    else:
        ap.error("必须提供 --src/--out，或 --seg")
    tmp = tempfile.mkdtemp(prefix="ghost_")
    print(f"[seg{args.seg}] src={src}\n          out={out}\n          tmp={tmp}", flush=True)

    # ---- 1) 映射判定：out[0]/src[0], out[1]/src[?], out[2]/src[1] ----
    print("\n=== A. 帧映射判定（out[0..6] vs src[0..3]） ===", flush=True)
    mats = {}
    for i in range(6):
        oi = grab(out, i, tmp, f"o{i}")
        row = []
        for j in range(4):
            if f"s{j}" not in mats:
                mats[f"s{j}"] = grab(src, j, tmp, f"s{j}")
            row.append(psnr(oi, mats[f"s{j}"]))
        print(f"  out[{i}] vs src[0..3] PSNR = " +
              " ".join(f"{v:6.2f}" for v in row), flush=True)

    # ---- 2) 鬼影帧定位：out[i] 与 src[k] 的最佳匹配 ----
    if args.around:
        print("\n=== B. 鬼影帧定位（每个鬼影索引 ±2，与最近 4 个源帧比对） ===", flush=True)
        for a in [int(x) for x in args.around.split(",") if x.strip()]:
            print(f"  --- 鬼影 out[{a}] ---", flush=True)
            k = a // 2
            srcs = {}
            for j in range(max(0, k - 1), k + 3):
                srcs[j] = grab(src, j, tmp, f"sa{j}")
            for i in range(a - 2, a + 3):
                oi = grab(out, i, tmp, f"oa{i}")
                best, bj = -1, None
                for j, sv in srcs.items():
                    p = psnr(oi, sv)
                    if p > best:
                        best, bj = p, j
                tag = "SRC" if i % 2 == 0 else "INTERP"
                print(f"    out[{i}]({tag:6s}) 最佳匹配 src[{bj}] PSNR={best:6.2f}", flush=True)

    # ---- 3) 守恒扫描：out[2k] 是否严格等于 src[k] ----
    if args.scan:
        s, e, st = [int(x) for x in args.scan.split(":")]
        print(f"\n=== C. 守恒扫描 out[2k] vs src[k]  (k={s}..{e} step={st}) ===", flush=True)
        bad = []
        vals = []
        for k in range(s, e, st):
            p = psnr(grab(out, 2 * k, tmp, f"scan_o{2*k}"), grab(src, k, tmp, f"scan_s{k}"))
            vals.append(p)
            if p < 25.0:
                bad.append((k, p))
            if (k - s) // st % 25 == 0:
                print(f"    k={k:6d}  out[{2*k:6d}] vs src[{k:6d}]  PSNR={p:6.2f}", flush=True)
        if vals:
            print(f"  → 扫描 {len(vals)} 点：PSNR min={min(vals):.2f} max={max(vals):.2f} "
                  f"mean={sum(vals)/len(vals):.2f}", flush=True)
            print(f"  → 异常（PSNR<25）: {len(bad)} 个 {bad[:20]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
