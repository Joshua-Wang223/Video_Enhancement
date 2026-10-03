"""为甲方案选锚点：找出「x264 VMAF 陡峭且各素材共同覆盖」的 crf 区间。

用已有实测数据（VE R2 的 x264 锚点 18-34 + 各素材 svtav1/x265 扫描曲线）
外推/内插 x264 的 VMAF(crf) 曲线，评估候选锚点集：
  ① 各素材在锚点处的 VMAF 是否落在**共同可分辨区间**（避免某个素材已到顶/到底）
  ② 锚点处目标曲线的局部斜率（越大越好）
  ③ 锚点区间是否被目标编码器的扫描点包住（否则反解会落区间外 ⇒ 0 评估点）

判据：目标 = 每素材锚点 VMAF 落在各自曲线 VMAF 幅度的 20%~85% 区间内，
      且 dVMAF/dcrf ≥ 0.8（陡峭区）。
"""
import importlib.util, json
from collections import defaultdict
from pathlib import Path

R = Path('/mnt/d/Workspace_Python/Video_Enhancement')
spec = importlib.util.spec_from_file_location('eqq', R / 'Accessory/probe/calibrate_equal_quality.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

D = defaultdict(lambda: defaultdict(list))
for k, v in json.loads(Path('/tmp/eqq2/1280x720_10s_n4/points.json').read_text()).items():
    vm = (v.get('m') or {}).get('vmaf')
    if vm is None:
        continue
    m, t, val = k.split('|')
    D[m][t].append((float(val), float(vm)))
for m in D:
    for t in D[m]:
        D[m][t].sort()
D = dict(D)
MATS = sorted(D)

print('=== x264 锚点 VMAF 实测 + 外插（n_subsample=1, 10s）===')
print(f'  {"素材":<18}' + ''.join(f'crf{c:<7.0f}' for c in (18, 22, 26, 30, 34, 38, 42, 46)))
for m in MATS:
    pts = D[m]['libx264']
    xs = [p for p, _ in pts]
    ys = [v for _, v in pts]
    # 在 crf 区间内线性插值/外插（仅用于选点评估，不作为标定手段）
    row = []
    for c in (18, 22, 26, 30, 34, 38, 42, 46):
        if c <= xs[-1]:
            v = C.vmaf_at_param(pts, c)
        else:
            # 外插：用最后两点斜率
            s = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            v = ys[-1] + s * (c - xs[-1])
        row.append(f'{v:>10.2f}' if v is not None else f'{"—":>10}')
    print(f'  {m:<18}' + ''.join(row))

print('\n=== 目标曲线覆盖率检查（锚点 VMAF 是否落在曲线幅度 15%~90%）===')
for tier in ('libx265', 'libsvtav1', 'librav1e@10'):
    print(f'\n  {tier}')
    print(f'    {"素材":<18}' + ''.join(f'{"crf"+str(c):>10}' for c in (18, 26, 30, 34, 38, 42, 46)))
    for m in MATS:
        curve = D[m][tier]
        vs = [v for _, v in curve]
        lo, hi = min(vs), max(vs)
        xs = [p for p, _ in D[m]['libx264']]
        ys = [v for _, v in D[m]['libx264']]
        row = []
        for c in (18, 26, 30, 34, 38, 42, 46):
            if c <= xs[-1]:
                v = C.vmaf_at_param(D[m]['libx264'], c)
            else:
                s = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
                v = ys[-1] + s * (c - xs[-1])
            if v is None:
                row.append(f'{"—":>10}')
                continue
            frac = (v - lo) / (hi - lo) if hi > lo else 0
            mark = '' if 0.15 <= frac <= 0.90 else '✗'
            row.append(f'{frac*100:>8.0f}%{mark:<1}')
        print(f'    {m:<18}' + ''.join(row))
    print(f'    （曲线 VMAF 幅度：' +
          ', '.join(f'{m[:6]}:{min(v for _,v in D[m][tier]):.0f}~{max(v for _,v in D[m][tier]):.0f}' for m in MATS) + '）')

print('\n=== 各素材目标曲线的 param 上限（锚点 VMAF 能否反解到参数）===')
for tier in ('libx265', 'libsvtav1'):
    print(f'\n  {tier}')
    for m in MATS:
        curve = D[m][tier]
        ps = [p for p, _ in curve]
        xs = [p for p, _ in D[m]['libx264']]
        ys = [v for _, v in D[m]['libx264']]
        out = []
        for c in (26, 30, 34, 38, 42, 46):
            if c <= xs[-1]:
                v = C.vmaf_at_param(D[m]['libx264'], c)
            else:
                s = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
                v = ys[-1] + s * (c - xs[-1])
            if v is None:
                out.append('--')
                continue
            r = C.calibrate_tier(tier, D[m]['libx264'], curve)
            own = dict(r['points']) if 'a' in r else {}
            hit = own.get(c)
            out.append(f'{hit:.0f}' if hit is not None else
                       (f'<{ps[0]:.0f}' if v >= max(y for _, y in curve) else '>?'))
        print(f'    {m:<18} 参数@{26,30,34,38,42,46}: ' + ' '.join(f'{x:>5}' for x in out))