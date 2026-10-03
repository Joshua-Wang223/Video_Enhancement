"""A/B 两套锚点在**同一批 7 素材**上的可比对照（真正的可比口径）。

数据源（VU 仓 temp/eqq_calib 下三个 workdir 合并）：
  - m2_anchorA          : libx264 A 锚点22/26/34        （21 点 / 7 素材）
  - m2_7src_6s_rav1e    : libx264 B 锚点18/21/24/27/30 + librav1e native（105 点）
  - m2_7src_6s_rav1e_s10: 同上 + librav1e@10              （105 点）
⇒ libx264 合并后同素材拥有 A∪B 全部 8 个锚点，可直接按两套锚点分别拟合 + LOO。
"""
import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'eqq', '/mnt/d/Workspace_Python/VidUtils/probe/calibrate_equal_quality.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

ROOT = Path('/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib')
TAGS = ['m2_anchorA', 'm2_7src_6s_rav1e', 'm2_7src_6s_rav1e_s10']

D = defaultdict(lambda: defaultdict(dict))
for tag in TAGS:
    for k, v in json.loads((ROOT / tag / 'points.json').read_text()).items():
        vm = (v.get('m') or {}).get('vmaf')
        if vm is None:
            continue
        m, t, val = k.split('|')
        D[m][t][float(val)] = float(vm)
for m in D:
    for t in D[m]:
        D[m][t] = sorted(D[m][t].items())
D = dict(D)
MATS = sorted(D)

A = [18.0, 22.0, 26.0, 30.0, 34.0]
B = [18.0, 21.0, 24.0, 27.0, 30.0]

print('═' * 78)
print('【1】同批 7 素材 libx264 锚点实测 VMAF（A∪B 全部 8 点）')
print('═' * 78)
print(f'{"素材":<20}{"A:18":>7}{"21":>7}{"22":>7}{"24":>7}{"26":>7}{"27":>7}{"30":>7}{"34":>7}   B跨度   A跨度')
for m in MATS:
    v = dict(D[m]['libx264'])
    def g(c):
        return f'{v[c]:.2f}' if c in v else '  --  '
    bspan = v[B[0]] - v[B[-1]]
    aspan = v[A[0]] - v[A[-1]]
    print(f'  {m:<20}' + ''.join(g(c) for c in [18, 21, 22, 24, 26, 27, 30, 34])
          + f'{bspan:9.2f}{aspan:8.2f}')

print()
print('═' * 78)
print('【2】A 集合 B 集各自缺测的锚点（须0 才算公平可比）')
print('═' * 78)
for name, anc in (('A', A), ('B', B)):
    miss = []
    for m in MATS:
        have = {c for c, _ in D[m]['libx264']}
        gaps = [int(c) for c in anc if c not in have]
        if gaps:
            miss.append(f'{m}:{gaps}')
    print(f'  {name} 锚点缺测: {miss if miss else "无（7 素材全覆盖）"}')


def loo(tier, anchors, mats):
    worst, detail = 0.0, {}
    for hold in mats:
        train = [m for m in mats if m != hold]
        xs, ys, bs = [], [], []
        ok = True
        for m in train:
            sub = [(c, v) for c, v in D[m]['libx264'] if c in anchors]
            if len(sub) < 2:
                ok = False
                break
            r = C.calibrate_tier(tier, sub, D[m].get(tier, []))
            if 'a' not in r:
                ok = False
                break
            xs += [x for x, _ in r['points']]
            ys += [y for _, y in r['points']]
        if not ok or len(xs) < 2:
            continue
        a, _ = C.fit_line(xs, ys)
        for m in train:
            sub = [(c, v) for c, v in D[m]['libx264'] if c in anchors]
            r = C.calibrate_tier(tier, sub, D[m].get(tier, []))
            if 'a' in r:
                bs += [y - a * x for x, y in r['points']]
        if not bs:
            continue
        b = statistics.median(bs)
        tp = D[hold].get(tier, [])
        if not tp:
            continue
        av = C.pava_nonincreasing([y for _, y in tp])
        iso = [(p, y) for (p, _), y in zip(tp, av)]
        w = 0.0
        for crf, want in D[hold]['libx264']:
            if crf not in anchors:
                continue
            g = C.vmaf_at_param(iso, a * crf + b)
            if g is not None:
                w = max(w, abs(g - want))
        detail[hold] = w
        worst = max(worst, w)
    return worst, detail


print()
print('═' * 78)
print('【3】LOO 可比对照（同 7 素材 / 同数据 / 唯一变量 = 锚点集）')
print('═' * 78)
print(f'  {"档位":<16}{"A:18/22/26/30/34":>19}{"B:18/21/24/27/30":>19}   差异')
for tier in ('librav1e', 'librav1e@10'):
    mats = [m for m in MATS if tier in D[m]]
    if len(mats) < 3:
        continue
    wa, da = loo(tier, A, mats)
    wb, db = loo(tier, B, mats)
    fa = f'{wa:.3f}' + ('✅' if wa < 1 else '❌')
    fb = f'{wb:.3f}' + ('✅' if wb < 1 else '❌')
    d = f'B 优于 A {(wa-wb)/wa*100:+.1f}%' if wb < wa else (f'A 优于 B {(wb-wa)/wb*100:+.1f}%' if wa < wb else '持平')
    print(f'  {tier:<16}{fa:>19}{fb:>19}   {d}')
    print(f'{"":<18}{"A 最差: " + ", ".join(f"{k[:12]}:{v:.2f}" for k, v in sorted(da.items(), key=lambda x: -x[1])[:2])}')
    print(f'{"":<18}{"B 最差: " + ", ".join(f"{k[:12]}:{v:.2f}" for k, v in sorted(db.items(), key=lambda x: -x[1])[:2])}')

print()
print('═' * 78)
print('【4】锚点集敏感性（同档位，改锚点子集）—— 若 worst 变化小则锚点非瓶颈')
print('═' * 78)
for tier in ('librav1e', 'librav1e@10'):
    mats = [m for m in MATS if tier in D[m]]
    if len(mats) < 3:
        continue
    print(f'  ── {tier}')
    for name, anc in (
        ('A  18/22/26/30/34', A),
        ('B  18/21/24/27/30', B),
        ('A∪B 8 点全集', sorted(set(A) | set(B))),
        ('仅两端 18/34', [18.0, 34.0]),
        ('仅两端 18/30', [18.0, 30.0]),
    ):
        w, _ = loo(tier, anc, mats)
        print(f'     {name:<22} worst={w:7.3f} ' + ('✅' if w < 1 else '❌'))