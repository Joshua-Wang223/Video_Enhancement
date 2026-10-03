"""两套锚点在**同一批 7 素材 / 同一 6s 口径**下的可比对照。

数据源（全部 7 素材共有）：
  · B 套锚点 18/21/24/27/30← m2_7src（软编）+ m2_7src_6s_rav1e / _s10（rav1e）
  · A 套锚点 18/22/26/30/34 ← m2_7src（18/30）+ m2_anchorA（补测 22/26/34）
⇒ 两套锚点现在**共用同一批素材与同一时长口径**，LOO 可直接比较。
"""
import importlib.util, json, statistics
from collections import defaultdict
from pathlib import Path

_h = '/mnt/d/Workspace_Python/VidUtils/probe/calibrate_equal_quality.py'
_s = importlib.util.spec_from_file_location('eqq', _h)
C = importlib.util.module_from_spec(_s)
_s.loader.exec_module(C)

W = Path('/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib')
# (workdir, 是否旧格式)  —— 只取 libx264 锚点，A/B 两套都需要
D = defaultdict(lambda: defaultdict(dict))
for tag in ('m2_7src', 'm2_anchorA', 'm2_7src_6s_rav1e', 'm2_7src_6s_rav1e_s10'):
    p = W / tag / 'points.json'
    if not p.is_file():
        p = W / tag / 'points_cache.json'
        if not p.is_file():
            continue
    for k, v in json.loads(p.read_text()).items():
        parts = k.split('|')
        mat, tier, val = parts[0], parts[-2], parts[-1]
        vm = (v.get('m') or v).get('vmaf')
        if vm is None:
            continue
        # 锚点键值统一成g（去 .0）
        D[mat][tier][float(f'{float(val):g}')] = float(vm)

MATS = sorted(D)
have = {m: {int(c) for c in D[m]['libx264']} for m in MATS}
A = [18, 22, 26, 30, 34]
Bv = [18, 21, 24, 27, 30]
print(f'素材 {len(MATS)} 条\n')
print('=== 各素材锚点覆盖（补测后）===')
for m in MATS:
    print(f'  {m:<22} A 命中 {len(set(A)&have[m])}/5   B 命中 {len(set(Bv)&have[m])}/5   '
          f'并集 {sorted(have[m])}')
full = [m for m in MATS if set(A) <= have[m] and set(Bv) <= have[m]]
print(f'\n两套锚点都全测的素材: {len(full)}/{len(MATS)} → {full}')

A_ = sorted(set.intersection(*[have[m] for m in full]))
print(f'共同可用锚点: {A_}')

# 目标编码器曲线也需两套齐全（目标曲线与锚点无关，但要素材齐全）
TIERS = sorted({t for m in MATS for t in D[m] if t != 'libx264'})
print(f'档位: {TIERS}\n')


def loo(tier, anchors, mats):
    worst, n_ev, n_mat = 0.0, 0, 0
    for hold in mats:
        train = [m for m in mats if m != hold]
        xs, ys, per = [], [], []
        for m in train:
            sub = [(float(c), v) for c, v in D[m]['libx264'].items() if int(c) in anchors]
            curve = [(p, v) for p, v in sorted(D[m][tier].items())]
            r = C.calibrate_tier(tier, sub, curve)
            if 'a' in r:
                xs += [x for x, _ in r['points']]
                ys += [y for _, y in r['points']]
                per.append(r['points'])
        if len(xs) < 2 or not per:
            continue
        a, _ = C.fit_line(xs, ys)
        bs = [statistics.median([y - a * x for x, y in pts]) for pts in per]
        b = statistics.median(bs)
        iso_v = C.pava_nonincreasing([v for _, v in sorted(D[hold][tier].items())])
        iso = [(p, v) for (p, _), v in zip(sorted(D[hold][tier].items()), iso_v)]
        n_mat += 1
        for crf, want in D[hold]['libx264'].items():
            if int(crf) not in anchors:
                continue
            got = C.vmaf_at_param(iso, a * crf + b)
            if got is None:
                continue
            n_ev += 1
            worst = max(worst, abs(got - want))
    return (worst if n_ev else None), n_ev, n_mat


print('=== LOO 可比对照（同 7 素材 / 同 6s / 仅锚点集不同）===')
print(f'  {"档位":<14}{"A:18/22/26/30/34":>19}{"B:18/21/24/27/30":>19}   结论')
summary = []
for t in TIERS:
    wa, ea, ma = loo(t, A, full)
    wb, eb, mb = loo(t, Bv, full)
    f = lambda v: '—' if v is None else f'{v:.2f}'
    if wa is not None and wb is not None:
        d = wb - wa
        cmp = f'B 更优 (Δ={d:+.2f})' if d < -0.01 else (f'A 更优 (Δ={d:+.2f})' if d > 0.01 else '持平')
    else:
        cmp = '不可比'
    summary.append((t, wa, wb))
    print(f'  {t:<14}{f(wa):>19}{f(wb):>19}   {cmp}')

# 各素材单独拟合（看训练内，作为上限参考）
print('\n=== 训练内（素材自身拟合）ΔVMAF 上限参考 ===')
print(f'  {"素材":<22}' + ''.join(f'{t[:11]:>13}' for t in TIERS))
for m in full:
    row = []
    for t in TIERS:
        sub = [(float(c), v) for c, v in D[m]['libx264'].items()]
        curve = [(p, v) for p, v in sorted(D[m][t].items())]
        r = C.calibrate_tier(t, sub, curve)
        row.append(f'{r["max_delta_vmaf"]:.2f}' if 'a' in r and r.get('max_delta_vmaf') is not None else '—')
    print(f'  {m:<22}' + ''.join(f'{x:>13}' for x in row))

print('\n=== 两套锚点的 VMAF 跨度（各素材 crf18→crf34 或 crf18→crf30）===')
for m in full:
    d = D[m]['libx264']
    va = (d[18.0], d[34.0]) if 18.0 in d and 34.0 in d else None
    vb = (d[18.0], d[30.0]) if 18.0 in d and 30.0 in d else None
    sa = f'{va[0]:.1f}→{va[1]:.1f} (跨度{va[0]-va[1]:.1f})' if va else '—'
    sb = f'{vb[0]:.1f}→{vb[1]:.1f} (跨度{vb[0]-vb[1]:.1f})' if vb else '—'
    print(f'  {m:<22} A: {sa:<30} B: {sb}')