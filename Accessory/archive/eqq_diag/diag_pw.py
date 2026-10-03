"""评估「方案 A：分段表」能否达到 ΔVMAF < 1.0。

用两批已测数据（零重编码）：
  集①VE R2   4 素材 / 10s / 锚点 18-34 / 5 档
  集②VU M2   7 素材 /  6s / 锚点 18-30 / 4 档

模型形式（生产表格式 (a,b,lo,hi) 只能承载直线，这里评估更丰富的形式）：
  L1 直线（当前生产格式）
  L2 分段2段（split = 锚点中位，报表格两行）
  L3 分段3段
  L4 分段4段（锚点数-1）

分段的关键约束：**表必须是「若干段 (a,b,lo,hi)」**，换算时按 x264_crf 落在哪段选行。
评估时严格模拟这个约束（不留全局外推自由度）。
"""
import importlib.util, json, statistics
from collections import defaultdict
from pathlib import Path

R = Path('/mnt/d/Workspace_Python/Video_Enhancement')
spec = importlib.util.spec_from_file_location('eqq', R / 'Accessory/probe/calibrate_equal_quality.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def load(paths):
    D = defaultdict(lambda: defaultdict(dict))
    for p in paths:
        for k, v in json.loads(Path(p).read_text()).items():
            vm = (v.get('m') or {}).get('vmaf')
            if vm is not None:
                m, t, val = k.split('|')
                D[m][t][float(val)] = float(vm)
    for m in D:
        for t in list(D[m]):
            if t != 'libx264':
                D[m][t] = sorted(D[m][t].items())
            else:
                D[m][t] = sorted(D[m][t].items())
    return D


SETS = {
    'VE R2（4 素材/10s）': load(['/tmp/eqq2/1280x720_10s_n4/points.json']),
    'VU M2（7 素材/6s）': load(['/tmp/eqq_vu/m2_7src/points.json']),
}


def segments(xs, ys, lo, hi):
    """按 x 区间切段，每段独立最小二乘；返回 [(a,b,lo,hi)]。"""
    segs, cur = [], []
    for x, y in sorted(zip(xs, ys)):
        if cur and x > cur[-1][0] and x != cur[-1][0]:
            # 以相邻 x 的中点为界
            cut = (cur[-1][0] + x) / 2
            if segs:
                prev = segs[-1]
                prev = (prev[0], prev[1], prev[2], cut)
                segs[-1] = prev
            cur = []
        cur.append((x, y))
    if cur:
        segs.append((cur[0][0], cur[-1][0], None))
    out = []
    for i, (xl, xh, _c) in enumerate(segs):
        px = [x for x, y in zip(xs, ys) if xl <= x <= xh]
        py = [y for x, y in zip(xs, ys) if xl <= x <= xh]
        if len(px) < 2:
            if len(px) == 1:
                out.append((0.0, py[0], xl, xh))
            continue
        a, b = C.fit_line(px, py)
        out.append((a, b, xl, xh))
    # 用中点重新划界，使各段无缝覆盖
    fin = []
    for i, (a, b, xl, xh) in enumerate(out):
        hi_bound = out[i + 1][2] if i + 1 < len(out) else xh
        lo_bound = xl
        if i > 0:
            lo_bound = (out[i - 1][3] + xl) / 2
        hi_bound = (xh + hi_bound) / 2 if i + 1 < len(out) else xh
        fin.append((a, b, lo_bound, hi_bound))
    return fin


def predict(segs, x):
    for a, b, lo, hi in segs:
        if lo <= x <= hi:
            return a * x + b
    return segs[-1][0] * x + segs[-1][1] if x > segs[-1][3] else \
        segs[0][0] * x + segs[0][1]


def build(tier, mats, D, nseg):
    """按 nseg 段拟合（用全部素材的池化点），返回段表。"""
    xs, ys = [], []
    anchors = set()
    for m in mats:
        r = C.calibrate_tier(tier, D[m]['libx264'], D[m][tier])
        if 'a' in r:
            xs += [x for x, _ in r['points']]
            ys += [y for _, y in r['points']]
        anchors |= {x for x, _ in D[m]['libx264']}
    if len(xs) < 2:
        return None
    if nseg == 1:
        a, b = C.fit_line(xs, ys)
        return [(a, b, -1e9, 1e9)]
    # 按锚点分桶：每个锚点区间一段
    ss = sorted(anchors)
    cuts = [ss[0] - 1] + [(ss[i] + ss[i + 1]) / 2 for i in range(len(ss) - 1)] + [ss[-1] + 1]
    wanted = min(nseg, len(ss))
    if wanted == 1:
        a, b = C.fit_line(xs, ys)
        return [(a, b, -1e9, 1e9)]
    # 选 wanted-1 个切点（均分锚点）
    idx = [round(i * (len(cuts) - 1) / wanted) for i in range(1, wanted)]
    bounds = [cuts[0]] + [cuts[i] for i in idx] + [cuts[-1]]
    segs = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        if i == len(bounds) - 2:
            hi = 1e9
        px = [x for x in xs if (x >= lo and (x < hi if i < len(bounds) - 2 else True))]
        py = [y for x, y in zip(xs, ys) if x in px]
        if len(px) >= 2:
            a, b = C.fit_line(px, py)
        elif len(px) == 1:
            a, b = 0.0, py[0]
        else:
            a, b = None, None
        if a is None:          # 空段：并入前一段
            continue
        segs.append((a, b, lo, hi))
    return segs


def eval_tier(tier, D, nseg):
    mats = [m for m in sorted(D) if tier in D[m] and 'libx264' in D[m]]
    if len(mats) < 3:
        return None, None
    worst = 0.0
    detail = []
    for hold in mats:
        train = [m for m in mats if m != hold]
        segs = build(tier, train, D, nseg)
        if not segs:
            continue
        av = C.pava_nonincreasing([v for _, v in D[hold][tier]])
        iso = [(p, v) for (p, _), v in zip(D[hold][tier], av)]
        hw, ne = 0.0, 0
        for crf, want in D[hold]['libx264']:
            pr = predict(segs, crf)
            got = C.vmaf_at_param(iso, pr)
            if got is None:
                continue
            ne += 1
            hw = max(hw, abs(got - want))
        if ne == 0:
            return float('inf'), '模型失效'
        worst = max(worst, hw)
        detail.append((hold, hw))
    return worst, detail


for sname, D in SETS.items():
    tiers = [t for t in sorted({t for m in D for t in D[m]}) if t != 'libx264']
    print(f'\n══ {sname}（{len({m for m in D})} 素材）══')
    print(f'  {"档位":<13}' + ''.join(f'{("L%d" % n):>9}' for n in (1, 2, 3, 4)) + '   段数样本')
    for t in tiers:
        row = []
        for nseg in (1, 2, 3, 4):
            w, _ = eval_tier(t, D, nseg)
            row.append('  inf' if w == float('inf') else f'{w:>6.3f}' + ('✅' if w < 1 else '❌'))
        segs = build(t, [m for m in sorted(D) if t in D[m]], D, 3)
        ns = len(segs) if segs else 0
        print(f'  {t:<13}' + ' '.join(row) + f'   L3={ns} 段')