#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""留一交叉验证（LOO）：2 素材拟合 → 预测第 3 素材的锚点参数 → 回查 VMAF 算 ΔVMAF。

不重编码：用该素材已测的 (param, vmaf) 扫描曲线反查。
"""
import importlib.util
import json
import statistics
from pathlib import Path

W = Path('/tmp/eqq_calib/1280x720_10s_n3')
spec = importlib.util.spec_from_file_location(
    'eqq', '/mnt/d/Workspace_Python/Video_Enhancement/Accessory/probe/calibrate_equal_quality.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

points = json.loads((W / 'points.json').read_text(encoding='utf-8'))
TIERS = ['libx265', 'libsvtav1', 'libvpx-vp9', 'libaom-av1', 'librav1e@10', 'librav1e']

data = {}
for k, v in points.items():
    mat, tier, val = k.split('|')
    data.setdefault(mat, {}).setdefault(tier, []).append((float(val), v['m']['vmaf']))
for m in data:
    for t in data[m]:
        data[m][t].sort()

mats = sorted(data)
print(f'素材 {len(mats)}：{mats}\n')
print(f'{"tier":<14}{"fold":<22}{"预测参数":>9}  {"实测等VMAF":>10}  {"Δparam":>8}  ΔVMAF')
worst = {}
for t in TIERS:
    for hold in mats:
        train = [m for m in mats if m != hold]
        # 用训练素材的 points 池化拟合
        xs, ys = [], []
        for m in train:
            r = C.calibrate_tier(t, data[m]['libx264'], data[m][t])
            if 'a' in r:
                xs += [x for x, _ in r['points']]
                ys += [y for _, y in r['points']]
        if len(xs) < 2:
            print(f'{t:<14}{hold:<22} 训练点不足')
            continue
        a, _ = C.fit_line(xs, ys)
        bs = []
        for m in train:
            r = C.calibrate_tier(t, data[m]['libx264'], data[m][t])
            if 'a' in r:
                bs.append(statistics.median([y - a * x for x, y in r['points']]))
        b = statistics.median(bs)
        # 用留出素材实测的 (param, vmaf) 曲线评估
        iso_v = C.pava_nonincreasing([v for _, v in data[hold][t]])
        iso = [(p, v) for (p, _), v in zip(data[hold][t], iso_v)]
        for crf, avmaf in data[hold]['libx264']:
            pred = a * crf + b
            got = C.vmaf_at_param(iso, pred)
            if got is None:
                continue
            dv = abs(got - avmaf)
            exp = C.interp_iso(iso, avmaf)
            worst[t] = max(worst.get(t, 0.0), dv)
            mark = '  ⚠' if dv > 1.0 else ''
            exp_s = '—' if exp is None else f'{exp:.1f}'
            dp_s = '—' if exp is None else f'{abs(pred - exp):.1f}'
            print(f'{t:<14}{("train=" + ",".join(x[:6] for x in train)):<22}'
                  f'{pred:>9.1f}  {exp_s:>10}  {dp_s:>8}  {dv:.3f}{mark}')
    print()

print('=== LOO 最差 ΔVMAF / 编码器 ===')
for t in TIERS:
    print(f'  {t:<14} {worst.get(t, float("nan")):.3f}   '
          f'{"✅ <1.0" if worst.get(t, 9) < 1.0 else "❌ ≥1.0"}')
