"""R2 收尾聚合：合并两个 workdir 的 points.json → 完整表 + LOO worst。

注意 report.json 不可信（被最后阶段覆盖），一切以 points.json 为准。
"""
import importlib.util, json, statistics
from collections import defaultdict
from pathlib import Path

R = Path('/mnt/d/Workspace_Python/Video_Enhancement')
spec = importlib.util.spec_from_file_location('eqq', R / 'Accessory/probe/calibrate_equal_quality.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
CRF = C.CRF

# ── 合并载入（跨workdir 去重：同(素材,档位,参数)取首次）──
raw = {}
for tag in ('1280x720_10s_n4', '1280x720_10s_n2'):
    p = Path('/tmp/eqq2') / tag / 'points.json'
    d = json.loads(p.read_text())
    print(f'  {tag}: {len(d)} 点')
    for k, v in d.items():
        raw.setdefault(k, v)
D = defaultdict(lambda: defaultdict(list))
for k, v in raw.items():
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
print(f'  合并去重后：{len(raw)} 点/ {len(MATS)} 素材\n')

TIERS = ['libx265', 'libvpx-vp9', 'libaom-av1', 'libsvtav1', 'librav1e@10', 'librav1e']


def slope_at(curve, p):
    for (p0, v0), (p1, v1) in zip(curve, curve[1:]):
        if p0 <= p <= p1 and p1 > p0:
            return abs(v1 - v0) / (p1 - p0)
    return 1e-3


def pooled(tier, mats, gamma=0.0):
    xs, ys, ws, per = [], [], [], []
    for m in mats:
        curve = D[m][tier]
        r = C.calibrate_tier(tier, D[m]['libx264'], curve)
        if 'a' not in r:
            continue
        pts = r['points']
        if len(pts) < 2:
            continue
        g = [slope_at(curve, y) for _, y in pts]
        per.append((pts, g))
        xs += [x for x, _ in pts]
        ys += [y for _, y in pts]
        ws += [v ** gamma for v in g]
    if len(xs) < 2:
        return None
    W = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / W
    my = sum(w * y for w, y in zip(ws, ys)) / W
    den = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if den == 0:
        return None
    a = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / den
    bs = []
    for pts, g in per:
        parts = sorted((y - a * x, max(w, 1e-6)) for (x, y), w in zip(pts, g))
        tot = sum(w for _, w in parts)
        acc, pick = 0.0, parts[-1][0]
        for val, w in parts:
            acc += w
            if acc >= tot / 2:
                pick = val
                break
        bs.append(pick)
    b = statistics.median(bs) if bs else 0.0
    resid = max(abs(y - (a * x + b)) for x, y in zip(xs, ys))
    return a, b, resid, len(xs)


def loo(tier, mats, gamma=0.0):
    if len(mats) < 3:
        return None, {}
    worst, per = 0.0, {}
    for hold in mats:
        p = pooled(tier, [m for m in mats if m != hold], gamma)
        if not p:
            continue
        a, b = p[0], p[1]
        v = C.pava_nonincreasing([y for _, y in D[hold][tier]])
        iso = [(pp, y) for (pp, _), y in zip(D[hold][tier], v)]
        hw, ne = 0.0, 0
        for crf, want in D[hold]['libx264']:
            got = C.vmaf_at_param(iso, a * crf + b)
            if got is None:
                continue
            ne += 1
            hw = max(hw, abs(got - want))
        if ne == 0:
            per[hold] = float('inf')
            continue
        per[hold] = hw
        worst = max(worst, hw)
    return worst, per


print('=' * 92)
print('R2 完整聚合（4 素材 / 10s / 锚点 18/22/26/30/34 / subsample=1）')
print('=' * 92)
print(f'{"档位":<13}{"a":>9}{"b":>10}{"lo":>5}{"hi":>5}{"池化残差":>10}{"同素材Δ":>9}'
      f'{"LOO(γ=0)":>11}{"LOO(γ=2)":>11}  素材')
rows = []
for t in TIERS:
    mats = [m for m in MATS if t in D[m]]
    if len(mats) < 2:
        continue
    p = pooled(t, mats, 0.0)
    if not p:
        continue
    a, b, resid, npts = p
    dvs = [C.calibrate_tier(t, D[m]['libx264'], D[m][t]).get('max_delta_vmaf', 0)
           for m in mats]
    dvs = [d for d in dvs if d is not None]
    w0, _ = loo(t, mats, 0.0)
    w2, _ = loo(t, mats, 2.0)
    lo, hi = CRF.QUALITY_MAP[C._ffcodec(t)][2], CRF.QUALITY_MAP[C._ffcodec(t)][3]
    rows.append((t, a, b, lo, hi, w0, w2))
    f = lambda v: '—' if v is None else (f'{v:.2f}' + ('✅' if v < 1 else '❌'))
    print(f'{t:<13}{a:>9.4f}{b:>+10.3f}{lo:>5}{hi:>5}{resid:>10.2f}'
          f'{(max(dvs) if dvs else 0):>9.2f}{f(w0):>11}{f(w2):>11}  {len(mats)}')

print('\n候选 QUALITY_MAP 行（γ=0，生产口径；ra1e 两档须分开存）:')
for t, a, b, lo, hi, w0, w2 in rows:
    key = t if t in CRF.QUALITY_MAP else C._ffcodec(t)
    print(f"    '{key}': ({round(a, 4)}, {round(b, 4)}, {lo}, {hi}),"
          f"   # LOO={w0:.2f}{'✅' if (w0 or 9) < 1 else '❌'}"
          + (f' γ2LOO={w2:.2f}' if w2 is not None else ''))
