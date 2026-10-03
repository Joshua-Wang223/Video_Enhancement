"""用 B 套锚点（18/21/24/27/30）在 A 侧重标，并做可比 LOO 对照。

数据：
  · A 侧 4 素材 × 目标曲线：/tmp/eqq2/1280x720_10s_n4（软编4档）+ _n2（rav1e native）
    ⚠ _n2 含 BBC 实拍 3 素材（10s 口径），需与 n4 的 4 素材合并
  · A 侧补测的 B 套锚点：/tmp/eqq2/1280x720_10s_anchorB（crf21/24/27）
  · VU 侧 7 素材：m2_7src（软编）+ m2_7src_6s_rav1e/_s10（rav1e 两档）
  · VU 侧补测的 A 套锚点：m2_anchorA（crf22/26/34）

目标：给出「A 侧用 B 套锚点」的 LOO，并对比现状（A 侧用 A 套锚点、9 素材合并）。
"""
import importlib.util, json, statistics
from collections import defaultdict
from pathlib import Path

def load_harness(p):
    s = importlib.util.spec_from_file_location('eqq_' + Path(p).parent.name, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

CH = load_harness('/mnt/d/Workspace_Python/Video_Enhancement/Accessory/probe/calibrate_equal_quality.py')
CV = load_harness('/mnt/d/Workspace_Python/VidUtils/probe/calibrate_equal_quality.py')


def load(sources, H):
    D = defaultdict(lambda: defaultdict(dict))
    for path in sources:
        p = Path(path)
        if not p.is_file():
            continue
        for k, v in json.loads(p.read_text()).items():
            parts = k.split('|')
            mat, tier, val = parts[0], parts[-2], parts[-1]
            vm = (v.get('m') or v).get('vmaf')
            if vm is None:
                continue
            D[mat][tier][float(f'{float(val):g}')] = float(vm)
    return dict(D)


VE = load(['/tmp/eqq2/1280x720_10s_n4/points.json',
           '/tmp/eqq2/1280x720_10s_n2/points.json',
           '/tmp/eqq2/1280x720_10s_anchorB/points.json'], CH)
VU = load(['/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src/points.json',
           '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src_6s_rav1e/points.json',
           '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src_6s_rav1e_s10/points.json',
           '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_anchorA/points.json'], CV)

A = [18, 22, 26, 30, 34]
Bv = [18, 21, 24, 27, 30]


def loo(D, H, tier, anchors, mats):
    ms = [m for m in mats if tier in D.get(m, {}) and 'libx264' in D[m]
          and len([c for c in D[m]['libx264'] if int(c) in anchors]) >= 3]
    if len(ms) < 4:
        return None, 0, 0
    worst, ev = 0.0, 0
    for hold in ms:
        train = [m for m in ms if m != hold]
        xs, ys, per = [], [], []
        for m in train:
            sub = [(float(c), v) for c, v in D[m]['libx264'].items() if int(c) in anchors]
            r = H.calibrate_tier(tier, sub, [(p, v) for p, v in sorted(D[m][tier].items())])
            if 'a' in r:
                xs += [x for x, _ in r['points']]
                ys += [y for _, y in r['points']]
                per.append(r['points'])
        if len(xs) < 2 or not per:
            continue
        a, _ = H.fit_line(xs, ys)
        b = statistics.median([statistics.median([y - a * x for x, y in pts]) for pts in per])
        curve = sorted(D[hold][tier].items())
        iso = [(p, v) for (p, _), v in zip(curve, H.pava_nonincreasing([v for _, v in curve]))]
        for crf, want in D[hold]['libx264'].items():
            if int(crf) not in anchors:
                continue
            got = H.vmaf_at_param(iso, a * crf + b)
            if got is not None:
                ev += 1
                worst = max(worst, abs(got - want))
    return (worst if ev else None), ev, len(ms)


def pooled(D, H, tier, anchors, mats):
    ms = [m for m in mats if tier in D.get(m, {}) and 'libx264' in D[m]]
    xs, ys, per = [], [], []
    for m in ms:
        sub = [(float(c), v) for c, v in D[m]['libx264'].items() if int(c) in anchors]
        r = H.calibrate_tier(tier, sub, [(p, v) for p, v in sorted(D[m][tier].items())])
        if 'a' in r:
            xs += [x for x, _ in r['points']]
            ys += [y for _, y in r['points']]
            per.append(r['points'])
    if len(xs) < 2:
        return None
    a, _ = H.fit_line(xs, ys)
    b = statistics.median([statistics.median([y - a * x for x, y in pts]) for pts in per])
    return a, b, len(ms)


VE_M = sorted(VE)
VU_M = sorted(VU)
TIERS = sorted({t for m in VU for t in VU[m] if t != 'libx264'})
print(f'VE 素材 {len(VE_M)}: {VE_M}')
print(f'VU 素材 {len(VU_M)}: {VU_M}')
print(f'档位: {TIERS}\n')

print('=== VU 侧 7 素材（两套锚点都已全测）===')
print(f'  {"档位":<14}{"A:18/22/26/30/34":>19}{"B:18/21/24/27/30":>19}   结论')
for t in TIERS:
    wa, _, _ = loo(VU, CV, t, A, VU_M)
    wb, _, _ = loo(VU, CV, t, Bv, VU_M)
    f = lambda v: '—' if v is None else f'{v:.2f}'
    cm = 'B 更优' if (wa and wb and wb < wa - 0.01) else ('A 更优' if (wa and wb and wa < wb - 0.01) else '持平')
    print(f'  {t:<14}{f(wa):>19}{f(wb):>19}   {cm}')

print('\n=== A 侧素材（VE 4+3=7，含 BBC 实拍）===')
VE_MATS = sorted({m for m in VE})
print(f'  素材: {VE_MATS}')
common = [m for m in VE_MATS
          if set(A) <= {int(c) for c in VE[m]['libx264']}
          and set(Bv) <= {int(c) for c in VE[m]['libx264']}]
print(f'  两套锚点都全测的: {len(common)} → {common}')
print(f'\n  {"档位":<14}{"A:18/22/26/30/34":>19}{"B:18/21/24/27/30":>19}   结论')
for t in sorted({t for m in VE_MATS for t in VE[m] if t != 'libx264'}):
    wa, _, _ = loo(VE, CH, t, A, common)
    wb, _, _ = loo(VE, CH, t, Bv, common)
    f = lambda v: '—' if v is None else f'{v:.2f}'
    cm = 'B 更优' if (wa and wb and wb < wa - 0.01) else ('A 更优' if (wa and wb and wa < wb - 0.01) else '持平')
    print(f'  {t:<14}{f(wa):>19}{f(wb):>19}   {cm}')

print('\n=== 若统一到 B 套：各仓表值（pooled 斜率 + 中位截距）===')
print(f'  {"档位":<14}{"VU 7素材(B套)":>22}{"VE (B套)":>22}')
for t in TIERS:
    pv = pooled(VU, CV, t, Bv, VU_M)
    pe = pooled(VE, CH, t, Bv, common)
    fv = '—' if not pv else f'({round(pv[0],4)}, {round(pv[1],4)}) n={pv[2]}'
    fe = '—' if not pe else f'({round(pe[0],4)}, {round(pe[1],4)}) n={pe[2]}'
    print(f'  {t:<14}{fv:>22}{fe:>22}')