"""甲方案的真正形态：斜率加权拟合（斜率 ∝ 误差放大倍数）。

推导：表参数误差 Δb 引起的 ΔVMAF ≈ |Δb| × |dVMAF/dparam|。
     ⇒ 要最小化**最大** ΔVMAF，应对高斜率点（放大倍数大）给更高拟合权重。
     ⇒ 目标从「最小化参数残差」改为「最小化 VMAF 域残差」。

比较（LOO）：
  L1  普通最小二乘（当前生产）
  W   以 |dVMAF/dparam| 为权重的加权最小二乘（权重 ∝ 斜率^γ，γ 可调）
  W+  W 且锚点限定到「斜率 ≥ 阈值」（即 crf18 附近的达标区，对应「甲」的锚点移出平坦区）
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


def slope_at(curve, p):
    for (p0, v0), (p1, v1) in zip(curve, curve[1:]):
        if p0 <= p <= p1 and p1 > p0:
            return abs(v1 - v0) / (p1 - p0)
    return None


def wfit(xs, ys, ws):
    """加权最小二乘。"""
    W = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / W
    my = sum(w * y for w, y in zip(ws, ys)) / W
    den = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if den == 0:
        return None, None
    a = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / den
    return a, my - a * mx


def run(D, tier, gamma, anchor_keep=None):
    mats = [m for m in sorted(D) if tier in D[m] and 'libx264' in D[m]]
    if len(mats) < 3:
        return None
    worst = 0.0
    for hold in mats:
        train = [m for m in mats if m != hold]
        xs, ys, ws = [], [], []
        for m in train:
            sub = [(c, v) for c, v in D[m]['libx264']
                   if anchor_keep is None or c in anchor_keep]
            curve = D[m][tier]
            r = C.calibrate_tier(tier, sub, curve)
            if 'a' not in r:
                continue
            for x, y in r['points']:
                g = slope_at(curve, y) or 0.0
                xs.append(x)
                ys.append(y)
                ws.append(max(g, 1e-3) ** gamma)
        if len(xs) < 2:
            continue
        a, _b0 = wfit(xs, ys, [1.0] * len(xs))     # a 用普通拟合（斜率只影响 b 的权重？）
        if a is None:
            continue
        # b 用加权中位（与现有口径一致：各素材中位数）
        bs = []
        for m in train:
            sub = [(c, v) for c, v in D[m]['libx264']
                   if anchor_keep is None or c in anchor_keep]
            curve = D[m][tier]
            r = C.calibrate_tier(tier, sub, curve)
            if 'a' not in r:
                continue
            parts = []
            for x, y in r['points']:
                g = slope_at(curve, y) or 0.0
                parts.append(((y - a * x), max(g, 1e-3) ** gamma))
            if parts:
                parts.sort()
                tot = sum(w for _, w in parts)
                acc = 0.0
                pick = parts[-1][0]
                for val, w in parts:
                    acc += w
                    if acc >= tot / 2:
                        pick = val
                        break
                bs.append(pick)
        if not bs:
            continue
        b = statistics_median(bs)
        v = C.pava_nonincreasing([y for _, y in D[hold][tier]])
        iso = [(p, y) for (p, _), y in zip(D[hold][tier], v)]
        for crf, want in D[hold]['libx264']:
            if anchor_keep is not None and crf not in anchor_keep:
                continue
            got = C.vmaf_at_param(iso, a * crf + b)
            if got is None:
                continue
            worst = max(worst, abs(got - want))
    return worst


def statistics_median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


SETS = {
    'VU M2 (7 素材/6s)': (load(['/tmp/eqq_vu/m2_7src/points.json']), [18.0, 21.0, 24.0, 27.0, 30.0]),
    'VE R2 (4 素材/10s)': (load(['/tmp/eqq2/1280x720_10s_n4/points.json']), [18.0, 22.0, 26.0, 30.0, 34.0]),
}

for name, (D, A) in SETS.items():
    tiers = sorted({t for m in D for t in D[m] if t != 'libx264'})
    print(f'\n{"="*72}\n== {name}')
    print(f'  {"档位":<13}' + ''.join(f'{"γ="+str(g):>9}' for g in (0, 1, 2, 3)) +
          f'{"γ=2+低crf限定":>15}')
    for t in tiers:
        row = []
        for g in (0, 1, 2, 3):
            w = run(D, t, g)
            row.append('  inf' if w is None else f'{w:.2f}' + ('✅' if w < 1 else '❌'))
        w2 = run(D, t, 2, anchor_keep=A[:2])
        last = 'inf' if w2 is None else f'{w2:.2f}' + ('✅' if w2 < 1 else '❌')
        print(f'  {t:<13}' + ' '.join(f'{x:>9}' for x in row) + f'{last:>15}')
print('\n注：γ=0 即当前生产口径（普通 LS + 中位截距）；γ↑ 表示更强调高斜率点。')
print('「γ=2+低crf限定」= 只用前 2 个锚点 + γ=2，即「甲」的锚点移出平坦区 + 加权。')