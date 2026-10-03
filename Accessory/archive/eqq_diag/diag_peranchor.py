"""甲方案的前置验证：LOO 误差是否集中在「VMAF 平坦区」的锚点？

若误差集中在高 VMAF（平坦区）锚点 ⇒ 甲（锚点移出平坦区）有效，值得重跑。
若误差均匀分布 或 集中在陡峭区 ⇒ 甲无效，重跑 10 小时会白费 ⇒ 必须先报告。

同时输出：若只用现有可得的陡峭锚点子集拟合，worst 变成多少。
"""
import importlib.util, json
from collections import defaultdict
from pathlib import Path

R = Path('/mnt/d/Workspace_Python/Video_Enhancement')
spec = importlib.util.spec_from_file_location('eqq', R / 'Accessory/probe/calibrate_equal_quality.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def load(paths):
    D = defaultdict(lambda: defaultdict(list))
    for p in paths:
        for k, v in json.loads(Path(p).read_text()).items():
            vm = (v.get('m') or {}).get('vmaf')
            if vm is None:
                continue
            m, t, val = k.split('|')
            D[m][t].append((float(val), float(vm)))
    for m in D:
        for t in D[m]:
            D[m][t].sort()
    return dict(D)


SETS = {
    'VU M2 (7 素材/6s)': load(['/tmp/eqq_vu/m2_7src/points.json']),
    'VE R2 (4 素材/10s)': load(['/tmp/eqq2/1280x720_10s_n4/points.json']),
}
ANCH = [18.0, 21.0, 24.0, 27.0, 30.0]   # VU 口径
ANCH2 = [18.0, 22.0, 26.0, 30.0, 34.0]  # VE 口径


def run(D, tier, anchors, tag=''):
    """返回 {anchor: worst ΔVMAF at that anchor}（LOO 口径）。"""
    mats = [m for m in sorted(D) if tier in D[m] and 'libx264' in D[m]]
    if len(mats) < 3:
        return None, None
    per_anchor = {a: 0.0 for a in anchors}
    worst = 0.0
    for hold in mats:
        train = [m for m in mats if m != hold]
        xs, ys = [], []
        for m in train:
            sub = [(c, v) for c, v in D[m]['libx264'] if c in anchors]
            r = C.calibrate_tier(tier, sub, D[m][tier])
            if 'a' in r:
                xs += [x for x, _ in r['points']]
                ys += [y for _, y in r['points']]
        if len(xs) < 2:
            continue
        a, _ = C.fit_line(xs, ys)
        bs = []
        for m in train:
            sub = [(c, v) for c, v in D[m]['libx264'] if c in anchors]
            r = C.calibrate_tier(tier, sub, D[m][tier])
            if 'a' in r:
                bs += [y - a * x for x, y in r['points']]
        if not bs:
            continue
        b = sorted(bs)[len(bs) // 2]
        v = C.pava_nonincreasing([y for _, y in D[hold][tier]])
        iso = [(p, y) for (p, _), y in zip(D[hold][tier], v)]
        for crf, want in D[hold]['libx264']:
            if crf not in anchors:
                continue
            got = C.vmaf_at_param(iso, a * crf + b)
            if got is None:
                continue
            dv = abs(got - want)
            per_anchor[crf] = max(per_anchor[crf], dv)
            worst = max(worst, dv)
    return worst, per_anchor


for name, D in SETS.items():
    A = ANCH if 'VU' in name else ANCH2
    tiers = sorted({t for m in D for t in D[m] if t != 'libx264'})
    print(f'\n{"="*66}\n== {name}')
    for t in tiers:
        w, pa = run(D, t, A)
        if w is None:
            continue
        print(f'\n  {t}   全锚点 worst={w:.2f}')
        for a_ in A:
            n = sum(1 for m in D if t in D[m]
                    and any(abs(c - a_) < 1e-6 for c, _ in D[m]['libx264']))
            bar = '#' * int(pa[a_] * 4)
            print(f'    crf{a_:>4.0f} (n={n})  ΔVMAF={pa[a_]:6.2f} {bar}')

    # 只用陡峭区子集（后 3 个锚点）重测
    print(f'\n  ---- 只用后 3 个锚点（陡峭区）----')
    for t in tiers:
        w3, _ = run(D, t, A[-3:])
        w5, _ = run(D, t, A)
        if w3 is None:
            continue
        gain = (w5 - w3) / w5 * 100 if w5 else 0
        print(f'    {t:<13} 5锚点={w5:6.2f}  →  3锚点={w3:6.2f}  '
              f'{"改善" if w3 < w5 else "恶化"} {abs(gain):.0f}%')