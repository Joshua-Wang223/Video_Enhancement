#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""等质量标定 —— 切片与素材清单生成器。

标定的参考片不是原片，而是「720p /固定时长 / lanczos / crf10」切片
（harness 的 `make_prep()`）。本脚本把原片转成切片并产出 manifest，
供 `eqq_calibrate_batch.py` 驱动，也作为素材入库的溯源清单。

⚠ **必须复用 harness 的 `make_prep()`**，不要自己拼 ffmpeg 命令 ——
   切片与已落表数据的口径必须逐字一致，否则复核得到的表值与在库表值不可比。

⚠ **素材短于口径时ffmpeg 静默截断且不报错**，污染整个口径且无告警。
   本脚本默认在切片后**强校验**实际时长，低于口径 0.05s 即报错退出。

用法
----
    # 素材规格表（JSON 数组）：src / name / side / category / topic / origin
    python3 eqq_slice_prep.py --spec my_clips.json --outdir ../input_videos/eqq_calib
"""
import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('eqq_harness', _HERE / 'calibrate_equal_quality.py')
CH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CH)

DUR_BY_SIDE = {'6s': 6.0, '10s': 10.0}


def resolve(p):
    """展开符号链接并确认存在——BBC 素材常用软链到 /mnt/f 网络盘。"""
    rp = Path(p).resolve()
    if not rp.is_file():
        raise SystemExit(f'[FATAL] 原片不可达：{p}→ {rp}')
    return rp


def slice_one(spec, outroot, workroot, width, height, duration, dry_run):
    src = resolve(spec['src'])
    side = spec['side']
    dst_dir = outroot / side
    dst = dst_dir / spec['name']

    sf, sd, sfps, sw, sh, _, stransfer = CH.ffprobe_video(src)
    if sd < duration - 0.05 and not dry_run:
        raise SystemExit(
            f'[FATAL] {src.name} 仅 {sd:.3f}s < 口径 {duration}s。'
            f'ffmpeg 会静默截断、VMAF 不可比 ⇒ 请重新采集更长素材，不要靠 --force。')

    wd = workroot / (side + '_' + Path(spec['name']).stem)
    wd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    prep, hdr = CH.make_prep(src, wd, duration, width, height)
    nf, dd, fps, w, h, pix, _ = CH.ffprobe_video(prep)
    # ★ 事后强校验（make_prep 内部不报错）
    if dd < duration - 0.05 and not dry_run:
        raise SystemExit(f'[FATAL] {spec["name"]} 切片实际 {dd:.3f}s < 口径 {duration}s')
    size = prep.stat().st_size
    md5 = CH.md5(prep)

    if dry_run:
        prep.unlink(missing_ok=True)
        shutil.rmtree(wd, ignore_errors=True)
    else:
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst.unlink(missing_ok=True)
        shutil.move(str(prep), str(dst))
        shutil.rmtree(wd, ignore_errors=True)

    return dict(
        side=side, name=spec['name'],
        # 绝对路径：manifest 可能被复制到别处（仓内外都可能），相对路径会随 cwd 失效
        slice_path=str(dst if not dry_run else outroot / side / spec['name']),
        category=spec.get('category', ''), topic=spec.get('topic', ''),
        origin=spec.get('origin', ''),
        source_path_resolved=str(src), source_res=f'{sw}x{sh}',
        source_fps=round(sfps, 3), source_codec_duration=round(sd, 3),
        source_frames=sf, source_transfer=stransfer or '',
        slice_target_seconds=duration, slice_res=f'{w}x{h}', slice_frames=nf,
        slice_duration=round(dd, 3), slice_fps=round(fps, 3), slice_pix_fmt=pix,
        slice_bytes=size, slice_md5=md5, hdr_tonemap=bool(hdr))


def main():
    ap = argparse.ArgumentParser(description='等质量标定切片与清单生成器')
    ap.add_argument('--spec', required=True,
                    help='素材规格 JSON 数组（src/name/side/category/topic/origin）')
    ap.add_argument('--outdir', required=True, help='切片输出根目录（通常是 input_videos/eqq_calib）')
    ap.add_argument('--workroot', default='/tmp/eqq_slice_work')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--duration', type=float, default=0,
                    help='覆盖切片时长（0 = 按 side 取 6/ 10s）')
    ap.add_argument('--dry-run', action='store_true', help='只报告，不落盘')
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding='utf-8'))
    outroot, workroot = Path(args.outdir), Path(args.workroot)
    workroot.mkdir(parents=True, exist_ok=True)
    by_side = {}
    for s in spec:
        dur = args.duration or DUR_BY_SIDE.get(s['side'])
        if not dur:
            raise SystemExit(f'[FATAL] side={s["side"]!r} 未在DUR_BY_SIDE 中，且未给 --duration')
        m = slice_one(s, outroot, workroot, args.width, args.height, dur, args.dry_run)
        print(f'[{"dry" if args.dry_run else "ok"}] {m["side"]}/{m["name"]:<44}'
              f'{m["slice_res"]} {m["slice_frames"]:>4}帧 {m["slice_duration"]:.3f}s '
              f'{m["slice_bytes"] / 1024:>7.0f}KiB hdr={m["hdr_tonemap"]} '
              f'src={m["source_res"]}@{m["source_fps"]}', flush=True)
        by_side.setdefault(m['side'], []).append(m)

    if not args.dry_run:
        for side, ms in by_side.items():
            (outroot / f'manifest_{side}.json').write_text(
                json.dumps(ms, ensure_ascii=False, indent=1), encoding='utf-8')
            print(f'→ manifest_{side}.json（{len(ms)} 条）')
    print(f'完成 {sum(len(v) for v in by_side.values())} 条')
    return 0


if __name__ == '__main__':
    sys.exit(main())