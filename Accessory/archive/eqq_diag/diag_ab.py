"""根因分解：LOO 的 ΔVMAF 主要来自 a（斜率）还是 b（截距）不可迁移？

方法：对每个留出素材，四种组合预测
  1. a_train, b_train  （真 LOO，全迁移）
  2. a_train, b_hold   （只有斜率迁移）
  3. a_hold,   b_train （只有截距迁移）
  4. a_hold,   b_hold   （oracle，不迁移但自洽——即素材自身拟合，已知 ΔVMAF<1）
若 3 的误差远小于 1 ⇒ 根因是**截距**；若 2 远小于 1 ⇒ 根因是**斜率**。
同时打印 VMAF 曲线在锚点区的局部斜率（平缓区⇒反解不稳定）。
"""
import importlib.util, json, statistics
from collections import defaultdict
from pathlib import Path

R = Path('/mnt/d/Workspace_Python/Video_Enhancement')
spec = importlib.util.spec_from_file_location('eqq', R / 'Accessory/probe/calibrate_equal_quality.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

raw = {}
for tag in ('1280x720_10s_n4', '1280x720_10s_n2'):
    for k, v in json.loads((Path('/tmp/eqq2') / tag / 'points.json').read_text()).items():
        vm = (v.get('m') or {}).get('vmaf')
        if vm is not None:
            raw[k] = float(vm)
D = defaultdict(lambda: defaultdict(dict))
for k, vm in raw.items():
    m, t, val = k.split('|')
    D[m][t][float(val)] = vm
for m in D:
    for t in list(D[m]):
        if t != 'libx264':
            D[m][t] = sorted(D[m][t].items())


def own_ab(mat, tier):
    r = C.calibrate_tier(tier, sorted(D[mat]['libx264'].items()), D[mat][tier])
    return (r.get('a'), r.get('b')) if 'a' in r else (None, None)


def pooled_ab(train, tier):
    xs, ys = [], []
    for m in train:
        a_, b_ = own_ab(m, tier)
        if a_ is None:
            continue
        r = C.calibrate_tier(tier, sorted(D[m]['libx264'].items()), D[m][tier])
        xs += [x for x, _ in r['points']]
        ys += [y for _, y in r['points']]
    if len(xs) < 2:
        return None, None
    a, _ = C.fit_line(xs, ys)
    bs = []
    for m in train:
        a_, b_ = own_ab(m, tier)
        r = C.calibrate_tier(tier, sorted(D[m]['libx264'].items()), D[m][tier])
        if 'a' in r:
            bs.append(statistics.median([y - a * x for x, y in r['points']]))
    return a, statistics.median(bs) if bs else None


def dv(hold, tier, a, b):
    av = C.pava_nonincreasing([v for _, v in D[hold][tier]])
    iso = [(p, v) for (p, _), v in zip(D[hold][tier], av)]
    worst, ne = 0.0, 0
    for crf, want in sorted(D[hold]['libx264'].items()):
        got = C.vmaf_at_param(iso, a * crf + b)
        if got is None:
            continue
        ne += 1
        worst = max(worst, abs(got - want))
    return (worst if ne else float('inf')), ne


print('=== 每素材自身 (a, b) ===')
for t in ('libx265', 'libvpx-vp9', 'libaom-av1', 'libsvtav1', 'librav1e@10'):
    s = []
    for m in sorted(D):
        a_, b_ = own_ab(m, t)
        s.append(f'{m[:8]}:a={a_:.3f},b={b_:+.1f}' if a_ else f'{m[:8]}:—')
    print(f'  {t:<14} ' + '  '.join(s))
    if t == 'libx265':
        as_ = [own_ab(m, t)[0] for m in sorted(D) if own_ab(m, t)[0]]
        bs_ = [own_ab(m, t)[1] for m in sorted(D) if own_ab(m, t)[1]]
        print(f'  {"":<14} a 极差={max(as_)-min(as_):.4f}  b 极差={max(bs_)-min(bs_):.2f}')

print('\n=== ΔVMAF 归因（留出素材；1=全迁移 2=仅a迁移 3=仅b迁移 4=oracle）===')
for t in ('libx265', 'libvpx-vp9', 'libaom-av1', 'libsvtav1', 'librav1e@10'):
    mats = [m for m in sorted(D) if t in D[m]]
    print(f'  {t}')
    for hold in mats:
        train = [m for m in mats if m != hold]
        a_tr, b_tr = pooled_ab(train, t)
        a_ho, b_ho = own_ab(hold, t)
        if a_tr is None or a_ho is None:
            continue
        d1, _ = dv(hold, t, a_tr, b_tr)
        d2, _ = dv(hold, t, a_tr, b_ho)
        d3, _ = dv(hold, t, a_ho, b_tr)
        d4, _ = dv(hold, t, a_ho, b_ho)
        f = lambda v: 'inf' if v == float('inf') else f'{v:.2f}'
        print(f'    {hold:<18} 全迁移={f(d1):>6}  仅a迁移={f(d2):>6}  '
              f'仅b迁移={f(d3):>6}  oracle={f(d4):>6}')

print('\n=== 锚点区 x264 局部斜率 dVMAF/dcrf（平缓 ⇒ 参数反解不稳）===')
for m in sorted(D):
    pts = sorted(D[m]['libx264'].items())
    segs = [f'{p0:.0f}-{p1:.0f}:{(v1-v0)/(p1-p0):.2f}'
            for (p0, v0), (p1, v1) in zip(pts, pts[1:])]
    print(f'  {m:<18} ' + '  '.join(segs))
