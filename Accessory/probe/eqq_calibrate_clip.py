#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""等质量标定 —— 单素材测量器（CRF/QP 扫描 → points.json）。

用harness `calibrate_equal_quality.py` 的 encode/measure/make_prep，
对**一条**素材扫完所有档位，产出 `points.json`。断点续跑：已有 key 直接跳过。

口径（改动前务必先读 Accessory/data/eqq_calibration/MANIFEST.md）
--------------------------------------------------------------------
· 参考片 = 720p / yuv420p / lanczos 降采样 / libx264 crf10 veryfast（`make_prep`）
· 锚点 = libx264 CRF `ANCHOR_CRFS`（两仓统一 18/21/24/27/30）
· VMAF 必须 `subsample=1` —— >1 会偏置 1.9~3.0（历史踩坑，见 superseded/n3）
· `duration` 必须 ≤ 素材实际时长。**素材短于 duration 时 ffmpeg 静默截断且不报错**，
  会污染整个口径 ⇒ 开跑前先 `ffprobe` 核实（历史 5 条 10s 素材实测 9.958~10.010s）。

⚠ **每素材必须独立 workdir**：`make_prep()` 固定写 `work/prep.mp4`（先unlink 再建），
  `_save_points()` 是 `write_text()` 直接覆盖、无锁 ⇒ 同 workdir 并行会互删 prep / 丢点。
  并行请用 `eqq_calibrate_batch.py`。

用法
----
    # 单条
    python3 eqq_calibrate_clip.py --src input_videos/eqq_calib/6s/live_kids_play_src1280x720.mp4 \
        --out /tmp/eqq_run/live_kids_play --duration 6

    # 只跑部分档位（rav1e最慢，调试时可先跳过）
    --tiers libx265,libvpx-vp9,libaom-av1,libsvtav1

    # 自定义扫描点（默认用 harness 的 SWEEP + rav1e QP 梯度）
    --sweep libx265=10,14,18 --sweep librav1e=10,30,50
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('eqq_harness', _HERE / 'calibrate_equal_quality.py')
CH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CH)

DEFAULT_TIERS = 'libx265,libvpx-vp9,libaom-av1,libsvtav1,librav1e,librav1e@10'
#: librav1e 的 QP 梯度（0~255 刻度，跨度需覆盖 native / -speed10 两档）
RAV1E_QP_SWEEP = [10, 30, 50, 70, 90, 110, 130, 155, 180, 210]


def parse_sweep(items, soft):
    """--sweep libx265=10,14 / --sweep librav1e=10,30 → {tier: [值]}；未指定的沿用默认。"""
    out = dict(soft)
    for it in items or []:
        tier, _, vals = it.partition('=')
        if not vals:
            raise SystemExit(f'--sweep 需为tier=v1,v2 形式：{it!r}')
        out[tier.strip()] = [float(v) for v in vals.split(',') if v.strip()]
    return out


def main():
    ap = argparse.ArgumentParser(description='等质量标定单素材测量器',
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='源视频（建议用 eqq_calib/ 下的切片）')
    ap.add_argument('--out', required=True, help='该素材的独立 workdir')
    ap.add_argument('--tiers', default=DEFAULT_TIERS,
                    help=f'逗号分隔档位键（默认 {DEFAULT_TIERS}）')
    ap.add_argument('--duration', type=float, required=True, help='切片时长（秒）')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--sweep', action='append', metavar='TIER=v1,v2',
                    help='覆盖某档位的扫描点，可重复；默认用 harness SWEEP')
    ap.add_argument('--src-is-prep', action='store_true',
                    help='--src 已是 720p 参考片（eqq_calib/ 下的切片）⇒ 跳过 make_prep。'
                         '**用切片复核时必须加**：再 prep 一次会多一轮 crf10 重编码，'
                         '实测锚点 VMAF 偏移 0.02~0.29，与库内原始数据不可比')
    ap.add_argument('--keep-prep', action='store_true',
                    help='保留 work/prep.mp4（默认跑完删除）')
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.is_file():
        raise SystemExit(f'[FATAL] 源不存在：{src}')
    out.mkdir(parents=True, exist_ok=True)
    pts_path = out / 'points.json'
    pts = json.loads(pts_path.read_text(encoding='utf-8')) if pts_path.is_file() else {}
    tiers = [t for t in args.tiers.split(',') if t]
    sweep = parse_sweep(args.sweep, CH.SWEEP)

    print(f'== {src.name} -> {out} ==', flush=True)
    src_frames, src_dur, src_fps, sw, sh, _, _ = CH.ffprobe_video(src)
    print(f'   源 {sw}x{sh} {src_frames}帧 {src_dur:.3f}s @{src_fps:.3f}', flush=True)
    # ★ 提前拦「素材短于口径」——ffmpeg 只会静默截断
    if src_dur < args.duration - 0.05:
        print(f'   ⚠ 素材仅 {src_dur:.3f}s < 口径 {args.duration}s：'
              f'ffmpeg 会静默截断，VMAF 不可比。建议重采集更长素材。', flush=True)

    if args.src_is_prep:
        prep, hdr = src, False
        print(f'   [src-is-prep] 直接用切片作参考片，跳过 make_prep', flush=True)
    else:
        prep, hdr = CH.make_prep(src, out, args.duration, args.width, args.height)
    nframes = CH.ffprobe_video(prep)[0]
    print(f'   参考片 {nframes}帧 md5={CH.md5(prep)[:12]} hdr_tonemap={hdr}', flush=True)

    def one(tier, value):
        key = f'{src.name}|{tier}|{value:g}'
        if key in pts:
            print(f'   [skip] {tier} {value:g}', flush=True)
            return
        enc_out = out / f'{tier.replace("@", "_spd")}_{value:g}.mp4'
        size, dt = CH.encode(prep, tier, value, enc_out)
        m = CH.measure(enc_out, prep, nframes, out, with_filters=False, subsample=1)
        pts[key] = {'m': m, 'sec': round(dt, 2)}
        pts_path.write_text(json.dumps(pts, ensure_ascii=False, indent=1), encoding='utf-8')
        enc_out.unlink(missing_ok=True)
        print(f'   {tier:<14}{value:>6g} -> {size / 1024:8.1f} KiB  '
              f'vmaf={m["vmaf"]:.3f}  ({dt:.1f}s)', flush=True)

    t0 = time.time()
    for tier in tiers:
        for crf in CH.ANCHOR_CRFS:
            one('libx264', crf)
        for value in sweep.get(tier, RAV1E_QP_SWEEP):
            one(tier, value)
        print(f'   -- {tier} 完成，累计 {len(pts)} 点，'
              f'用时 {(time.time() - t0) / 60:.1f} min', flush=True)

    if not args.keep_prep and not args.src_is_prep:
        prep.unlink(missing_ok=True)
    print(f'== {src.name} 完成：{len(pts)} 点，用时 {(time.time() - t0) / 60:.1f} min', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())