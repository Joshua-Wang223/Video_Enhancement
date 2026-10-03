"""A仓锚点 (18/22/26/30/34) vs B 仓 (18/21/24/27/30) 对比。

同一批已测数据（VE R2 4 素材 / 10s），仅锚点集不同 ⇒ 纯口径对比。
"""
import importlib.util, json, statistics
from collections import defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'eqq', '/mnt/d/Workspace_Python/Video_Enhancement/Accessory/probe/calibrate_equal_quality.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

D = defaultdict(lambda: defaultdict(dict))
for p in ('/tmp/eqq2/1280x720_10s_n4/points.json', '/tmp/eqq2/1280x720_10s_n2/points.json'):
    for k, v in json.loads(Path(p).read_text()).items():
        vm = (v.get('m') or {}).get('vmaf')
        if vm is None:
            continue
        m, t, val = k.split('|')
        D[m][t][float(val)] = float(vm)
for m in D:
    for t in D[m]:
        D[m][t] = sorted(D[m][t].items())
D = dict(D)

A = [18.0, 22.0, 26.0, 30.0, 34.0]
Bv = [18.0, 21.0, 24.0, 27.0, 30.0]

print('=== A 仓锚点 18/22/26/30/34 的实测VMAF 与局部斜率 ===')
for m in sorted(D):
    pts = D[m]['libx264']
    xs = [p for p, _ in pts]
    ys = [y for _, y in pts]
    v = dict(pts)
    row = ' '.join(f'{v[c]:.2f}' for c in A if c in v)
    sl = ' '.join(f'{(ys[i+1]-ys[i])/(xs[i+1]-xs[i]):+.2f}' for i in range(len(xs) - 1))
    print(f'  {m:<18} VMAF: {row}   斜率: {sl}   跨度={ys[0]-ys[-1]:.1f}')

print('\n=== 覆盖度：各素材在两套锚点下的可用点数（须 ≥3 否则拟合不可靠）===')
for m in sorted(D):
    have = {c for c, _ in D[m]['libx264']}
    na = len([c for c in A if c in have])
    nb = len([c for c in Bv if c in have])
    print(f'  {m:<18} A 锚点命中 {na}/5   B 锚点命中 {nb}/5')


def loo(tier, anchors, mats):
    worst = 0.0
    for hold in mats:
        train = [m for m in mats if m != hold]
        xs, ys, bs = [], [], []
        for m in train:
            sub = [(c, v) for c, v in D[m]['libx264'] if c in anchors]
            r = C.calibrate_tier(tier, sub, D[m][tier])
            if 'a' in r:
                xs += [x for x, _ in r['points']]
                ys += [y for _, y in r['points']]
        if len(xs) < 2:
            continue
        a, _ = C.fit_line(xs, ys)
        for m in train:
            sub = [(c, v) for c, v in D[m]['libx264'] if c in anchors]
            r = C.calibrate_tier(tier, sub, D[m][tier])
            if 'a' in r:
                bs += [y - a * x for x, y in r['points']]
        if not bs:
            continue
        b = statistics.median(bs)
        av = C.pava_nonincreasing([y for _, y in D[hold][tier]])
        iso = [(p, y) for (p, _), y in zip(D[hold][tier], av)]
        for crf, want in D[hold]['libx264']:
            if crf not in anchors:
                continue
            g = C.vmaf_at_param(iso, a * crf + b)
            if g is not None:
                worst = max(worst, abs(g - want))
    return worst


print('\n=== LOO 对比（同 4 素材/10s 数据，仅锚点集不同）===')
print(f'  {"档位":<14}{"A:18/22/26/30/34":>18}{"B:18/21/24/27/30":>18}   差异')
for t in ('libx265', 'libvpx-vp9', 'libaom-av1', 'libsvtav1', 'librav1e@10'):
    mats = [m for m in sorted(D) if t in D[m] and 'libx264' in D[m]]
    if len(mats) < 3:
        continue
    wa = loo(t, A, mats)
    wb = loo(t, Bv, mats)
    fa = f'{wa:.2f}' + ('✅' if wa < 1 else '❌')
    fb = f'{wb:.2f}' + ('✅' if wb < 1 else '❌')
    better = 'B 更优' if wb < wa else ('A 更优' if wa < wb else '持平')
    print(f'  {t:<14}{fa:>18}{fb:>18}   {better}')

print('\n=== 结论依据：B 锚点在 A 侧数据上「不可用」的锚点数===')
for m in sorted(D):
    have = {c for c, _ in D[m]['libx264']}
    miss = [int(c) for c in Bv if c not in have]
    print(f'  {m:<18} B 锚点缺测: {miss if miss else "无"}')
