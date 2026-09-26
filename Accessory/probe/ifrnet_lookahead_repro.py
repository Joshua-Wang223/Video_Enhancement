#!/usr/bin/env python
"""IFRNet 侧 NVENC 独立复现：绕开插帧推理，直接按生产时序压测 encode_frames_stream。

目标：定位"Duplicate POC"重复帧的来源（段验收 decoded==expected 但 decode_errors）。

生产时序（external/ifrnet_video/nvenc_sdk.py _loop/_encode_chunk）：
  · 段首：可能传入 _pending_f0_nv12（此处简化为 force_idr_first=True）
  · LA>0：有界分块流式编码，chunk=128，仅段末最后一块 send_eos=True
  · 每段 hevc/av1 新建编码器（IFRNET_NVENC_CROSS_SEGMENT_REUSE 默认禁用）

用法:
  python temp/retest/repro_ifrnet_la.py --frames 400 --codec hevc --la 8 \
      --rc vbr_hq --qp 21 --chunk 128 --segments 2 --out temp/retest/ril.mp4
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent.parent   # Accessory/<分类>/*.py → Video_Enhancement
sys.path.insert(0, str(_ROOT / "external"))

from ifrnet_video.nvenc_sdk import NVENCEncoder  # noqa: E402


def make_nv12(n, h, w, seed=0):
    """生成 n 帧 NV12 CUDA tensor，形状 (h*3//2, w)。帧间有明显差异便于识别重复。"""
    rng = np.random.default_rng(seed)
    y = np.empty((n, h, w), dtype=np.uint8)
    for i in range(n):
        row = np.full((h, w), (i * 7) % 256, dtype=np.uint8)
        row[:, max(0, (i * 11) % max(1, w - 32)):] = 255   # 移动亮块，制造帧间差异
        y[i] = np.clip(row.astype(np.int16) + rng.integers(-2, 3, size=(h, w)), 0, 255).astype(np.uint8)
    uv = np.full((n, h // 2, w // 2), 128, dtype=np.uint8).repeat(2, axis=2).repeat(2, axis=1)
    nv12 = np.concatenate([y, uv], axis=1)                  # (n, h*3//2, w)
    return [torch.from_numpy(f).contiguous().to("cuda", non_blocking=True) for f in nv12]


def write_es(path, parts):
    with open(path, "wb") as f:
        for p in parts:
            f.write(p)


def mux(es_path, out_path, fps, codec="hevc"):
    # [FIX-MUX-CODEC] 原实现硬编码 "-f hevc"：跑 --codec h264 时会把 H.264 裸流
    # 当 HEVC 解析（missing picture in access unit, rc=234），导致 H.264 用例
    # 一律假失败。改为按 codec 选择解复用器。
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", codec, "-r", str(fps), "-i", str(es_path),
           "-c:v", "copy", "-f", "mp4", "-movflags", "+faststart", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stderr or "").strip()


def decode_check(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-i", str(path),
                        "-map", "0:v:0", "-f", "null", "-"],
                       capture_output=True, text=True)
    err = (r.stderr or "").strip().splitlines()
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-count_frames", "-show_entries", "stream=nb_read_frames",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return (p.stdout or "").strip(), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--codec", default="hevc")
    ap.add_argument("--rc", default="vbr_hq")
    ap.add_argument("--qp", type=int, default=21)
    ap.add_argument("--preset", default="veryslow")
    ap.add_argument("--la", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--segments", type=int, default=2)
    ap.add_argument("--pipeline-depth", type=int, default=4)
    ap.add_argument("--out", default="temp/retest/ril.mp4")
    args = ap.parse_args()

    W, H = args.width, args.height
    out_base = Path(_ROOT) / args.out
    out_base.parent.mkdir(parents=True, exist_ok=True)

    print(f"[repro] {W}x{H}@{args.fps} codec={args.codec} rc={args.rc} la={args.la} "
          f"preset={args.preset} qp={args.qp} frames={args.frames} "
          f"chunk={args.chunk} segments={args.segments}", flush=True)

    all_ok = True
    for seg in range(args.segments):
        t0 = time.time()
        enc = NVENCEncoder(width=W, height=H, fps=args.fps, preset=args.preset,
                           qp=args.qp, codec=args.codec,
                           pipeline_depth=args.pipeline_depth,
                           rate_mode=args.rc, la_depth=args.la)
        frames = make_nv12(args.frames, H, W, seed=seg)

        # ── 按生产时序分块提交，仅末块 send_eos ──
        # [FIX-REPRO-CLOSE] 编码异常时也必须 close：否则编码器对象留到解释器退出
        # 阶段才被 GC，DestroyEncoder 在未排空状态下触发 SIGSEGV（exit=139），
        # 掩盖真实失败原因。生产路径由 pipeline 的 finally 保证，此处对齐。
        collected = []          # [(fi, bytes)]
        idx = 0
        try:
            while idx < len(frames):
                chunk = frames[idx:idx + args.chunk]
                idx += len(chunk)
                last = (idx >= len(frames))
                pairs = enc.encode_frames_stream(chunk, force_idr_first=(len(collected) == 0),
                                                 send_eos=last)
                collected.extend(pairs)
                print(f"[seg{seg}] chunk -> +{len(pairs)} pairs "
                      f"(submitted {idx}/{len(frames)}, eos={last})", flush=True)
        finally:
            enc.close()

        # ── 帧级校验 ──
        fis = [fi for fi, _ in collected]
        empty = sum(1 for _, b in collected if not b)
        dup_fi = len(fis) - len(set(fis))
        # 相邻完全相同的码流 = 重复帧（非 IDR 帧不应与上帧逐字节相同）
        dup_bytes = sum(1 for i in range(1, len(collected))
                        if collected[i][1] and collected[i][1] == collected[i - 1][1])
        expected_fi = list(range(args.frames))
        print(f"[seg{seg}] pairs={len(collected)} expected={args.frames} "
              f"empty={empty} 重复fi={dup_fi} 相邻逐字节重复={dup_bytes} "
              f"fi序列正确={fis == expected_fi}", flush=True)
        if fis != expected_fi and len(fis) == args.frames:
            bad = [(i, a, b) for i, (a, b) in enumerate(zip(fis, expected_fi)) if a != b][:10]
            print(f"[seg{seg}] ⚠️ fi 序列错位样例: {bad}", flush=True)

        # ── 落盘 + 解码校验 ──
        es = out_base.with_name(f"{out_base.stem}_seg{seg}.hevc")
        mp4 = out_base.with_name(f"{out_base.stem}_seg{seg}.mp4")
        write_es(es, [b for _, b in collected])
        rc, err = mux(es, mp4, args.fps, args.codec)
        print(f"[seg{seg}] mux rc={rc} {err[:200]}", flush=True)
        nread, derr = decode_check(mp4)
        print(f"[seg{seg}] nb_read_frames={nread} (期望 {args.frames}) "
              f"decode_errors={len(derr)}", flush=True)
        for line in derr[:10]:
            print(f"    {line}", flush=True)
        ok = (len(derr) == 0 and str(nread) == str(args.frames)
              and len(collected) == args.frames and empty == 0 and dup_bytes == 0)
        all_ok = all_ok and ok
        print(f"[seg{seg}] {'PASS' if ok else 'FAIL'}  用时 {time.time()-t0:.1f}s", flush=True)

        # [FIX-REPRO-CLOSE] close 已在上方 finally 完成（异常路径同样覆盖）
        del enc

    print(f"\n=== 总结: {'ALL PASS' if all_ok else 'HAS FAILURE'} ===", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
