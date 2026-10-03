"""决定性检验：分段表到底能救多少？

对比（全部 LOO，零重编码）：
  L1     1 段直线（当前生产格式）
  Lk     k 段（按锚点均匀切）
  PER    **每个素材单独一张表**（不可用于生产 —— 生产不知道素材身份，
         但它给出「仿射模型族 + 任意素材集」的**误差下界**）
  ORACLE 素材自身拟合（训练内）

若 PER 也 > 1.0 ⇒ 连「每素材专属表」都达不到 1.0，说明问题不在表的**分段数**
而在**素材本身的质量跨度超出 VMAF 可分辨范围** ⇒ 方案 A 无法达标。
"""
import importlib.util, json, statistics
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
    'VE R2 (4)': load(['/tmp/eqq2/1280x720_10s_n4/points.json']),
    'VU M2 (7)': load(['/tmp/eqq_vu/m2_7src/points.json']),
}


def iso_of(mat, tier):
    v = C.pava_nonincreasing([y for _, y in D[mat][tier]])
    return [(p, y) for (p, _), y in zip(D[mat][tier], v)]


def fit_nseg(tier, mats, nseg):
    """用给定素材集拟合 n 段表；段界取锚点区间的均匀切分。"""
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
    if nseg <= 1:
        a, b = C.fit_line(xs, ys)
        return [(a, b, -1e9, 1e9)]
    ss = sorted(anchors)
    if nseg >= len(ss):
        # 每锚点一段（用相邻锚点中点为界）
        bounds = []
        for i in range(len(ss)):
            lo = -1e9 if i == 0 else (ss[i - 1] + ss[i]) / 2
            hi = 1e9 if i == len(ss) - 1 else (ss[i] + ss[i + 1]) / 2
            bounds.append((lo, hi))
    else:
        idx = sorted({round(i * (len(ss) - 1) / nseg) for i in range(nseg)})
        bnds = [-1e9] + [(ss[idx[i]] + ss[idx[i + 1]]) / 2 for i in range(len(idx) - 1)] + [1e9]
        bounds = [(bnds[i], bnds[i + 1]) for i in range(len(bnds) - 1)]
    segs = []
    for lo, hi in bounds:
        px = [x for x in xs if lo <= x < hi]
        py = [y for x, y in zip(xs, ys) if lo <= x < hi]
        if len(px) >= 2:
            a, b = C.fit_line(px, py)
            if a is None:
                continue
        elif len(px) == 1:
            a, b = 0.0, py[0]
        else:
            continue
        segs.append((a, b, lo, hi))
    return segs or None


def pred(segs, x):
    for a, b, lo, hi in segs:
        if lo <= x < hi:
            return a * x + b
    a, b = segs[-1][0], segs[-1][1]
    return a * x + b


def worst_of(segs, hold, tier):
    iso = iso_of(hold, tier)
    w, ne = 0.0, 0
    for crf, want in D[hold]['libx264']:
        got = C.vmaf_at_param(iso, pred(segs, crf))
        if got is None:
            continue
        ne += 1
        w = max(w, abs(got - want))
    return (w if ne else float('inf'))


for name, D in SETS.items():
    tiers = sorted({t for m in D for t in D[m] if t != 'libx264'})
    print(f'\n══ {name}：{len(D)} 素材 ══')
    print(f'  {"档位":<13}{"L1":>9}{"L2":>9}{"L3":>9}{"L5":>9}'
          f'{"PER素材表":>11}{"ORACLE":>9}')
    for t in tiers:
        mats = [m for m in sorted(D) if t in D[m] and 'libx264' in D[m]]
        if len(mats) < 3:
            continue
        row = []
        for n in (1, 2, 3, 5):
            w = 0.0
            ok = True
            for hold in mats:
                s = fit_nseg(t, [m for m in mats if m != hold], n)
                if not s:
                    ok = False
                    break
                v = worst_of(s, hold, t)
                if v == float('inf'):
                    ok = False
                    break
                w = max(w, v)
            row.append(f'{w:.2f}' + ('✅' if w < 1 else '❌') if ok else 'inf')
        # PER：每素材单独表预测**其它**素材 ⇒ 取所有配对最差
        pw, pok = 0.0, True
        for a_mat in mats:
            s = fit_nseg(t, [a_mat], 1)
            if not s:
                pok = False
                break
            for b_mat in mats:
                if a_mat == b_mat:
                    continue
                v = worst_of(s, b_mat, t)
                if v == float('inf'):
                    pok = False
                    break
                pw = max(pw, v)
            if not pok:
                break
        per = f'{pw:.2f}' + ('✅' if pw < 1 else '❌') if pok else 'inf'
        # ORACLE
        ow = 0.0
        for m in mats:
            r = C.calibrate_tier(t, D[m]['libx264'], D[m][t])
            if 'max_delta_vmaf' in r:
                ow = max(ow, r['max_delta_vmaf'])
        print(f'  {t:<13}' + ' '.join(f'{x:>9}' for x in row) +
              f'{per:>11}' + f'{ow:>7.2f}' + ('✅' if ow < 1 else '❌'))