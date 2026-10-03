#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 points.json 重新聚合等质量表（report.json 被 pass-2 覆盖，只剩 rav1e 两档）。"""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

W = Path('/tmp/eqq_calib/1280x720_10s_n3')
HARNESS = Path('/mnt/d/Workspace_Python/Video_Enhancement/Accessory/probe/calibrate_equal_quality.py')

spec = importlib.util.spec_from_file_location('eqq', HARNESS)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

points = json.loads((W / 'points.json').read_text(encoding='utf-8'))
TIERS = ['libx265', 'libsvtav1', 'libvpx-vp9', 'libaom-av1', 'librav1e@10', 'librav1e']

# 组织成 {material: {tier: [(val, vmaf)]}}
data = {}
for k, v in points.items():
    mat, tier, val = k.split('|')
    data.setdefault(mat, {}).setdefault(tier, []).append((float(val), v['m']['vmaf']))
for mat in data:
    for t in data[mat]:
        data[mat][t].sort()

print(f'素材: {list(data)}\n')

per_tier_pts = {t: [] for t in TIERS}
detail = {}
for mat, tiers in data.items():
    ref = tiers['libx264']
    detail[mat] = {}
    for t in TIERS:
        r = C.calibrate_tier(t, ref, tiers.get(t, []))
        detail[mat][t] = r
        if 'a' in r:
            per_tier_pts[t] += list(r['points'])

print(f'{"tier":<14}{"a":>10}{"b":>11}{"lo":>5}{"hi":>5}'
      f'{"resid_max":>11}{"dVMAF_max":>11}  模型')
table = {}
for t in TIERS:
    pts = per_tier_pts[t]
    if len(pts) < 2:
        print(f'{t:<14} 点不足({len(pts)})')
        continue
    a, _ = C.fit_line([x for x, _ in pts], [y for _, y in pts])
    bs = [statistics.median([y - a * x for x, y in detail[m][t]['points']])
          for m in detail if 'a' in detail[m][t]]
    b = statistics.median(bs) if bs else 0.0
    codec = C._ffcodec(t)
    lo, hi = C.CRF.QUALITY_MAP[codec][2], C.CRF.QUALITY_MAP[codec][3]
    resid = max(detail[m][t]['max_resid_param'] for m in detail if 'a' in detail[m][t])
    dv = max((detail[m][t]['max_delta_vmaf'] or 0) for m in detail if 'a' in detail[m][t])
    models = {detail[m][t]['model'] for m in detail if 'a' in detail[m][t]}
    table[t] = [round(a, 4), round(b, 4), int(lo), int(hi)]
    print(f'{t:<14}{a:>10.4f}{b:>+11.3f}{lo:>5}{hi:>5}{resid:>11.3f}{dv:>11.3f}  {",".join(sorted(models))}')

print('\n=== 各素材逐点（等 VMAF 参数 vs 锚点）===')
for mat, r in detail.items():
    print(f'[{mat}]')
    for t in TIERS:
        d = r[t]
        if 'a' not in d:
            continue
        pts = ' '.join(f'{x:g}→{y:.1f}' for x, y in d['points'])
        print(f'   {t:<14} {pts}   mono={d["monotonic_violation"]:.4f} {d["method"]}')

print('\n=== 候选表（写入 convert_crf.py）===')
for t, v in table.items():
    print(f"    '{t}': ({v[0]}, {v[1]}, {v[2]}, {v[3]}),")

(W / 'harvested.json').write_text(
    json.dumps({'table': table, 'detail': {m: {t: {k: v for k, v in d.items()
                if k != 'piecewise'} for t, d in r.items()} for m, r in detail.items()}},
               ensure_ascii=False, indent=1), encoding='utf-8')
print(f'\n已写 {W/"harvested.json"}')
