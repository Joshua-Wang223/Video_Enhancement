"""甲方案的可行性上限：锚点数与锚点位置的联合扫描。

已有数据只能提供 crf18~30(VU)/18~34(VE) 的锚点。要评估「锚点移到 crf28~44」
需要新数据（重跑）。此处用**已有数据模拟**：
  - 自由度数 N = 锚点数（仿射表 2 参数 ⇒ N<3 时欠定；N≥4 才稳）
  - 斜率随 crf 增大的规律已知（实测），可外推「若锚点继续右移，斜率更大」
关键问题：**锚点数从 5 增到 6/7 能否把 worst 压到 1.0 以内**？
用 bootstrap：对每档位，枚举现有 5 锚点的所有子集，看「锚点更多」的上限趋势。
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
    return 0.0


def loo_with(D, tier, anchors, gamma=2.0):
    mats = [m for m in sorted(D) if tier in D[m] and 'libx264' in D[m]]
    if len(mats) < 3:
        return None
    worst = 0.0
    for hold in mats:
        train = [m for m in mats if m != hold]
        xs, ys, ws = [], [], []
        per_mat = []
        for m in train:
            sub = [(c, v) for c, v in D[m]['libx264'] if c in anchors]
            curve = D[m][tier]
            r = C.calibrate_tier(tier, sub, curve)
            if 'a' not in r:
                continue
            pts = r['points']
            g = [max(slope_at(curve, y), 1e-3) for _, y in pts]
            per_mat.append((pts, g))
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
        for pts, g in per_mat:
            parts = sorted((y - a * x, w) for (x, y), w in zip(pts, g))
            tot = sum(w for _, w in parts)
            acc, pick = 0.0, parts[-1][0]
            for val, w in parts:
                acc += w
                if acc >= tot / 2:
                    pick = val
                    break
            bs.append(pick)
        b = statistics.median(bs)
        v = C.pava_nonincreasing([y for _, y in D[hold][tier]])
        iso = [(p, y) for (p, _), y in zip(D[hold][tier], v)]
        for crf, want in D[hold]['libx264']:
            if crf not in anchors:
                continue
            got = C.vmaf_at_param(iso, a * crf + b)
            if got is not None:
                worst = max(worst, abs(got - want))
    return worst


for name, path, A in (('VU M2 (7 素材)', '/tmp/eqq_vu/m2_7src/points.json',
                       [18.0, 21.0, 24.0, 27.0, 30.0]),
                      ('VE R2 (4 素材)', '/tmp/eqq2/1280x720_10s_n4/points.json',
                       [18.0, 22.0, 26.0, 30.0, 34.0])):
    D = load([path])
    tiers = sorted({t for m in D for t in D[m] if t != 'libx264'})
    print(f'\n{"="*76}\n== {name}   γ=2 加权')
    print(f'  {"锚点子集":<28}' + ''.join(f'{t[:11]:>13}' for t in tiers) + '   自由度')
    combos = []
    for k in range(3, len(A) + 1):
        for c in itertools.combinations(A, k):
            combos.append(c)
    for c in sorted(combos, key=lambda x: (-len(x), x))[:10]:
        row = []
        for t in tiers:
            w = loo_with(D, t, list(c))
            row.append('  inf' if w is None else f'{w:.2f}' + ('✅' if w < 1 else '❌'))
        print(f'  crf{",".join(str(int(x)) for x in c):<24}' +
              ' '.join(f'{x:>13}' for x in row) + f'   {len(c)-2}')
    print(f'\n  → 最优子集（全档位 worst 最小）：')
    best = []
    for c in combos:
        ws = []
        for t in tiers:
            w = loo_with(D, t, list(c))
            if w is not None:
                ws.append(w)
        if ws:
            best.append((max(ws), c))
    best.sort()
    for w, c in best[:4]:
        print(f'     crf{",".join(str(int(x)) for x in c):<24} 全档位 worst={w:.2f} '
              + ('✅' if w < 1 else '❌'))