"""同批 7 素材上对比 A/B 两套锚点（VU M2 数据，锚点 18/21/24/27/30全测）。

背景：此前用 A 侧数据比较不成立 —— A 侧实测锚点是 18/22/26/30/34，
B 套的 21/24/27 在 A 侧数据里**缺测**，只有 2 个点可评估，误差区间被截断。
VU 的 m2_7src 数据是 B 套锚点（18/21/24/27/30）全测，但**同时也含 18/22/26/30/34
中的一部分**？—— 先查实际可用的锚点交集，再决定可比性。
"""
import importlib.util, json, statistics
from collections import defaultdict
from pathlib import Path

# 直接复用 harness 的拟合/插值函数（权威口径，与 loo_equal_quality.py 同源）
_h = '/mnt/d/Workspace_Python/VidUtils/probe/calibrate_equal_quality.py'
_s = importlib.util.spec_from_file_location('eqq', _h)
C = importlib.util.module_from_spec(_s)
_s.loader.exec_module(C)

W = Path('/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib')
D = defaultdict(lambda: defaultdict(list))
seen = set()
for tag in ('m2_7src', 'm2_7src_6s_rav1e', 'm2_7src_6s_rav1e_s10'):
    p = W / tag / 'points.json'
    if not p.is_file():
        p2 = W / tag / 'points_cache.json'
        if not p2.is_file():
            continue
        p = p2
    for k, v in json.loads(p.read_text()).items():
        parts = k.split('|')
        mat, tier, val = parts[0], parts[-2], parts[-1]
        vm = (v.get('m') or v).get('vmaf')
        if vm is None:
            continue
        key = (mat, tier, float(val))
        if key in seen:
            continue
        seen.add(key)
        D[mat][tier].append((float(val), float(vm)))
for m in D:
    for t in D[m]:
        D[m][t].sort()
D = dict(D)
MATS = sorted(D)
print(f'素材 {len(MATS)}: {MATS}\n')

# 各素材实际有哪些 x264 锚点
have = defaultdict(set)
for m in MATS:
    for c, _ in D[m].get('libx264', []):
        have[m].add(int(c))
allc = sorted(set().union(*have.values()))
print(f'全部素材的 x264 锚点并集: {allc}')
for m in MATS:
    print(f'  {m:<22} {sorted(have[m])}')
print(f'\n⇒ 只有**全部 7 素材都有**的锚点才能用于统一拟合：'
      f'{sorted(set.intersection(*have.values()))}')


def loo(tier, anchors, mats):
    """按给定锚点集做 LOO（与 loo_equal_quality.py 同口径）。"""
    usable = [m for m in mats if tier in D[m] and 'libx264' in D[m]
              and len([c for c, _ in D[m]['libx264'] if int(c) in anchors]) >= 3]
    if len(usable) < 4:
        return None, 0
    worst, n_anchor_eval = 0.0, 0
    for hold in usable:
        train = [m for m in usable if m != hold]
        xs, ys = [], []
        for m in train:
            sub = [(c, v) for c, v in D[m]['libx264'] if int(c) in anchors]
            r = C.calibrate_tier(tier, sub, D[m][tier])
            if 'a' in r:
                xs += [x for x, _ in r['points']]
                ys += [y for _, y in r['points']]
        if len(xs) < 2:
            continue
        a, _ = C.fit_line(xs, ys)
        bs = []
        for m in train:
            sub = [(c, v) for c, v in D[m]['libx264'] if int(c) in anchors]
            r = C.calibrate_tier(tier, sub, D[m][tier])
            if 'a' in r:
                bs.append(statistics.median([y - a * x for x, y in r['points']]))
        if not bs:
            continue
        b = statistics.median(bs)
        iso_v = C.pava_nonincreasing([v for _, v in D[hold][tier]])
        iso = [(p, v) for (p, _), v in zip(D[hold][tier], iso_v)]
        for crf, want in D[hold]['libx264']:
            if int(crf) not in anchors:
                continue
            got = C.vmaf_at_param(iso, a * crf + b)
            if got is None:
                continue
            n_anchor_eval += 1
            worst = max(worst, abs(got - want))
    return (worst if n_anchor_eval else None), n_anchor_eval


A = [18, 22, 26, 30, 34]
Bv = [18, 21, 24, 27, 30]
common = sorted(set.intersection(*have.values()))
print(f'\n共同可用锚点: {common}')
print(f'\n=== LOO 对比（同一批 7 素材，仅锚点集不同）===')
print(f'  {"档位":<16}{"A:18/22/26/30/34":>20}{"B:18/21/24/27/30":>20}   差异')
tiers = sorted({t for m in D for t in D[m] if t not in ('libx264',)})
for t in tiers:
    wa, na = loo(t, A, MATS)
    wb, nb = loo(t, Bv, MATS)
    f = lambda v, n: '—' if v is None else f'{v:.2f}' + ('✅' if v < 1 else '❌')
    cmp = ''
    if wa is not None and wb is not None:
        cmp = 'B 更优' if wb < wa else ('A 更优' if wa < wb else '持平')
    print(f'  {t:<16}{f(wa,na):>20}{f(wb,nb):>20}   {cmp}')