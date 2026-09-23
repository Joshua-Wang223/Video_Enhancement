#!/usr/bin/env python
"""真实素材编码回归：用真实视频解码出的 NV12 帧压 encode_frames_stream。

用途：验证 **正常工况下尺寸钳制 0 次触发**（立项 §8 第 4 项）。
与 tests/repro_ifrnet_lookahead.py 的区别：输入是真实视频帧而非合成噪声，
会走真实的参数集/帧序/码率分布路径。

用法:
  python tests/repro_real_frames.py -i benchmark_output/xxx.mp4 --codec hevc --la 8
  python tests/repro_real_frames.py -i benchmark_output/xxx.mp4 --codec h264 --la 8
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "external"))

from ifrnet_video.nvenc_sdk import NVENCEncoder  # noqa: E402


def probe_wh(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "csv=p=0",
                        str(path)], capture_output=True, text=True)
    w, h = (r.stdout or "").strip().split(",")[:2]
    return int(w), int(h)


def decode_nv12(path, w, h, n):
    """解码前 n 帧为 NV12，返回 list[CUDA tensor (h*3//2, w)]。"""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vframes", str(n),
           "-pix_fmt", "nv12", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    fs = w * h * 3 // 2
    out = []
    for i in range(len(raw) // fs):
        arr = np.frombuffer(raw[i * fs:(i + 1) * fs], dtype=np.uint8).reshape(h * 3 // 2, w)
        out.append(torch.from_numpy(arr.copy()).contiguous().to("cuda", non_blocking=True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--codec", default="hevc")
    ap.add_argument("--rc", default="vbr_hq")
    ap.add_argument("--qp", type=int, default=21)
    ap.add_argument("--la", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=128)
    args = ap.parse_args()

    src = Path(_ROOT) / args.input if not Path(args.input).is_absolute() else Path(args.input)
    w0, h0 = probe_wh(src)
    w, h = (w0 // 2) * 2, (h0 // 2) * 2
    print(f"[real] {src.name} {w}x{h} codec={args.codec} la={args.la} "
          f"frames={args.frames} chunk={args.chunk}", flush=True)

    frames = decode_nv12(src, w, h, args.frames)
    print(f"[real] 解码得到 {len(frames)} 帧", flush=True)
    if not frames:
        print("[real] ❌ 无帧可编码", flush=True)
        return 1

    enc = NVENCEncoder(width=w, height=h, fps=30.0, preset="veryslow", qp=args.qp,
                       codec=args.codec, pipeline_depth=4,
                       rate_mode=args.rc, la_depth=args.la)
    collected = []
    try:
        idx = 0
        while idx < len(frames):
            chunk = frames[idx:idx + args.chunk]
            idx += len(chunk)
            pairs = enc.encode_frames_stream(chunk, force_idr_first=(idx == len(chunk)),
                                             send_eos=(idx >= len(frames)))
            collected.extend(pairs)
    finally:
        enc.close()

    dropped = getattr(enc, '_sizecap_force_dropped', 0)
    sites = getattr(enc, '_diag_sizecap_site', {})
    print(f"[real] pairs={len(collected)} expected={len(frames)} "
          f"钳制强制消费={dropped} 站点命中={sites}", flush=True)
    ok = (len(collected) == len(frames) and dropped == 0 and not sites)
    print(f"[real] {'PASS' if ok else 'FAIL'}（正常工况钳制必须为 0）", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
