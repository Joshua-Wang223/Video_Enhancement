#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""等质量标定 —— 批量测量器（manifest 驱动 · 每素材独立 workdir · 并行）。

历史脚本 `run_gen.sh` / `run_ji.sh` 把素材清单硬编码在shell 数组里，换批次就得改脚本。
本脚本改为读 `input_videos/eqq_calib/manifest_*.json`，新增素材只需更新 manifest。

⚠ **并行安全**：每素材一个独立 workdir（`eqq_calibrate_clip.py` 内部约定）。
   harness 的 `make_prep()` 固定写`work/prep.mp4` 且 `_save_points()` 是无锁
   `write_text()` 覆盖 ⇒ 同 workdir 并行会互删 prep / 丢点。

用法
----
    # 跑 6s 侧全部 12 条，4 路并行
    python3 eqq_calibrate_batch.py \
        --manifest ../input_videos/eqq_calib/manifest_6s.json \
        --outroot /tmp/eqq_run --jobs 4

    # 串行 + 只跑软编（rav1e 单点 ~60~180s，调试时先跳过）
    python3 eqq_calibrate_batch.py --manifest ... --outroot ... --jobs 1 \
        --tiers libx265,libvpx-vp9,libaom-av1,libsvtav1
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CLIP = _HERE / 'eqq_calibrate_clip.py'


def load(manifest):
    """读 manifest → [(tag, src, duration)]。

    manifest 由 `eqq_slice_prep.py` 生成，每条含 slice_path（绝对）/
    slice_target_seconds / source_path_resolved。tag 取切片名主干（workdir 名可读且唯一）。

    兼容历史 manifest：①缺 slice_path ⇒ 按 `<manifest 所在目录>/<side>/<name>` 推导；
    ②slice_path 为相对 `input_videos/...` ⇒ 从 manifest 上溯到 `input_videos` 再拼。
    """
    mp = Path(manifest).resolve()
    for m in json.loads(mp.read_text(encoding='utf-8')):
        raw = (m.get('slice_path') or '').strip()
        if not raw:
            src = mp.parent / m['side'] / m['name']
        else:
            src = Path(raw)
            if not src.is_absolute():
                parts = Path(raw).parts
                if 'input_videos' in parts:
                    idx = len(parts) - 1 - parts[::-1].index('input_videos')
                    src = mp.parents[idx] / src
                else:
                    src = mp.parent / src
        if not src.is_file():
            print(f'[skip] 切片缺失：{src}', file=sys.stderr)
            continue
        dur = float(m.get('slice_target_seconds') or m.get('slice_seconds') or 0)
        if dur <= 0:
            print(f'[skip] manifest 缺时长字段：{src.name}', file=sys.stderr)
            continue
        yield Path(m['name']).stem, src, dur


def run_one(tag, src, dur, outroot, tiers, sweeps, src_is_prep, keep_prep):
    wd = Path(outroot) / tag
    wd.mkdir(parents=True, exist_ok=True)
    # 供 eqq_watch_batch.py --restart-on-abnormal 定位源
    (wd / 'src.txt').write_text(str(src) + '\n', encoding='utf-8')
    (wd / 'duration.txt').write_text(f'{dur}\n', encoding='utf-8')
    cmd = [sys.executable, str(_CLIP), '--src', str(src), '--out', str(wd),
           '--duration', str(dur)]
    if tiers:
        cmd += ['--tiers', tiers]
    for s in sweeps:
        cmd += ['--sweep', s]
    if src_is_prep:
        cmd.append('--src-is-prep')
    if keep_prep:
        cmd.append('--keep-prep')
    t0 = time.time()
    with (wd / 'run.log').open('w', encoding='utf-8') as lg:
        rc = subprocess.call(cmd, stdout=lg, stderr=subprocess.STDOUT)
    n = 0
    pf = wd / 'points.json'
    if pf.is_file():
        n = len(json.loads(pf.read_text(encoding='utf-8')))
    return tag, rc, n, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description='等质量标定批量测量器')
    ap.add_argument('--manifest', required=True, help='eqq_calib/manifest_*.json')
    ap.add_argument('--outroot', required=True, help='批次根目录，每素材一个子 workdir')
    ap.add_argument('--jobs', type=int, default=1, help='并行路数（默认 1）')
    ap.add_argument('--tiers', default='', help='覆盖档位（空 = 全部 6 档）')
    ap.add_argument('--sweep', action='append', metavar='TIER=v1,v2',
                    help='覆盖某档位扫描点，逐素材透传（调试小样本时用）')
    ap.add_argument('--src-is-prep', action='store_true',
                    help='manifest 里的切片已是参考片 ⇒ 透传 --src-is-prep 跳过 make_prep。'
                         '**复核库内数据时必须加**（否则多一轮 crf10 重编码，实测锚点 VMAF 偏移 0.02~0.29）')
    ap.add_argument('--keep-prep', action='store_true')
    ap.add_argument('--only', default='', help='只跑这些 tag（逗号分隔，调试用）')
    args = ap.parse_args()

    items = load(args.manifest)
    if args.only:
        want = {t.strip() for t in args.only.split(',')}
        items = [i for i in items if i[0] in want]
    if not items:
        raise SystemExit('没有可跑的素材（检查 manifest 路径与切片是否存在）')
    print(f'== {len(items)} 条素材，{args.jobs} 路并行，outroot={args.outroot}')
    for tag, src, dur in items:
        print(f'   {tag:<44}{dur}s{"" if dur >= 5.95 else "  ⚠ 素材短于口径"}  {src.name}')

    t0 = time.time()
    fails = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = [ex.submit(run_one, tag, src, dur, args.outroot,
                          args.tiers, args.sweep, args.src_is_prep, args.keep_prep)
                for tag, src, dur in items]
        for fu in futs:
            tag, rc, n, dt = fu.result()
            flag = 'OK ' if rc == 0 else f'rc={rc}'
            print(f'[{flag}] {tag:<44}{n:>4} 点  {dt / 60:6.1f} min', flush=True)
            if rc != 0:
                fails.append(tag)

    print(f'\n墙钟 {(time.time() - t0) / 60:.1f} min；'
          f'失败 {len(fails)} 条 {fails if fails else ""}')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())