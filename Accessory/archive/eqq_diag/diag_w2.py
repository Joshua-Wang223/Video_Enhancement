"""甲方案的最后一击：用「crf26 起 + γ 加权」在**现有实测锚点**上验证方向。

甲方案提议：锚点 crf 26/30/34/38/42。
现有实测只有 18/22/26/30/34 ⇒ 只能测到 26/30/34（3 点，欠定但可看趋势）。
若 26/30/34 + γ 加权已明显优于 18-34 全窗口 ⇒ 方向成立，值得到 38/42 重跑。

另测：把 crf18/22 **丢弃**（而非下移）会不会更好 —— 因为它们的斜率极小
（0.02~0.25），是误差放大倍数最低、贡献最差的锚点。
"""
import importlib.util, json, itertools, statistics
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


def slope_at(curve, p):
    for (p0, v0), (p1, v1) in zip(curve, curve[1:]):
        if p0 <= p <= p1 and p1 > p0:
            return abs(v1 - v0) / (p1 - p0)
    return 1e-3


def loo(D, tier, anchors, gamma):
    mats = [m for m in sorted(D) if tier in D[m] and 'libx264' in D[m]]
    if len(mats) < 3:
        return None
    worst, used = 0.0, 0
    for hold in mats:
        train = [m for m in mats if m != hold]
        xs, ys, ws, per = [], [], [], []
        for m in train:
            sub = [(c, v) for c, v in D[m]['libx264'] if c in anchors]
            curve = D[m][tier]
            r = C.calibrate_tier(tier, sub, curve)
            if 'a' not in r:
                continue
            pts = r['points']
            if not pts:
                continue
            g = [slope_at(curve, y) for _, y in pts]
            per.append((pts, g))
            xs += [x for x, _ in pts]
            ys += [y for _, y in pts]
            ws += [v ** gamma for v in g]
        if len(xs) < 2:
            continue
        W = sum(ws)
        mx = sum(w * x for w, x in zip(ws, xs)) / W
        my = sum(w * y for w, y in zip(ws, ys)) / W
        den = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
        if den == 0:
            continue
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
        if not bs:
            continue
        b = statistics.median(bs)
        v = C.pava_nonincreasing([y for _, y in D[hold][tier]])
        iso = [(p, y) for (p, _), y in zip(D[hold][tier], v)]
        for crf, want in D[hold]['libx264']:
            if crf not in anchors:
                continue
            got = C.vmaf_at_param(iso, a * crf + b)
            if got is None:
                continue
            used += 1
            worst = max(worst, abs(got - want))
    return (worst if used else None)


SETS = {
    'VU M2 (7 素材/6s)': (load(['/tmp/eqq_vu/m2_7src/points.json']), [18.0, 21.0, 24.0, 27.0, 30.0]),
    'VE R2 (4 素材/10s)': (load(['/tmp/eqq2/1280x720_10s_n4/points.json']), [18.0, 22.0, 26.0, 30.0, 34.0]),
}

for name, (D, A) in SETS.items():
    tiers = sorted({t for m in D for t in D[m] if t != 'libx264'})
    print(f'\n{"="*74}\n== {name}')
    print(f'  {"锚点集":<26}{"γ":>3}{"worst":>9}   评价')
    combos = [c for k in range(3, len(A) + 1) for c in itertools.combinations(A, k)]
    for c in sorted(combos, key=lambda x: x[0]):
        for g in (0, 2):
            w = loo(D, tiers[0] if len(tiers) == 1 else c and tiers[0], list(c), g)
        # 全档位取最差
        ws = []
        for t in tiers:
            r = loo(D, t, list(c), g)
            if r is not None:
                ws.append(r)
        if not ws:
            continue
        worst = max(ws)
        label = ('✅ <1.0' if worst < 1 else
                 ('⚠ 1~2' if worst < 2 else '❌'))
        print(f'  crf{",".join(str(int(x)) for x in c):<22}{g:>3}{worst:>9.2f}   {label}')