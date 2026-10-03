"""最后一问：失败是否由少数「VMAF 不可分辨」素材主导？

素材分两类：
  - VMAF 可分辨：锚点区 x264 VMAF 斜率足够（crf18-22 不是平台）
  - VMAF 不可分辨：锚点挤在顶部（接近 100），参数反解条件数极差

做法：对每个留出素材，报告
  (a) 该素材的锚点 VMAF 顶部拥挤度
  (b) 用**其余素材**拟合的表预测它的 worst ΔVMAF
然后按拥挤度排序，看误差是否单调相关 ⇒ 若是，则「素材筛选/限定质量区」是有效修法。
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


def pooled(tier, mats):
    xs, ys, bs = [], [], []
    for m in mats:
        r = C.calibrate_tier(tier, D[m]['libx264'], D[m][tier])
        if 'a' in r:
            xs += [x for x, _ in r['points']]
            ys += [y for _, y in r['points']]
    if len(xs) < 2:
        return None
    a, _ = C.fit_line(xs, ys)
    for m in mats:
        r = C.calibrate_tier(tier, D[m]['libx264'], D[m][tier])
        if 'a' in r:
            bs.append(sum(y - a * x for x, y in r['points']) / len(r['points']))
    return a, (sum(bs) / len(bs) if bs else 0.0)


def dvm(hold, tier, a, b):
    v = C.pava_nonincreasing([y for _, y in D[hold][tier]])
    iso = [(p, y) for (p, _), y in zip(D[hold][tier], v)]
    w, ne = 0.0, 0
    for crf, want in D[hold]['libx264']:
        got = C.vmaf_at_param(iso, a * crf + b)
        if got is None:
            continue
        ne += 1
        w = max(w, abs(got - want))
    return (w if ne else float('inf'))


for name, path in (('VU M2 (7 素材)', ['/tmp/eqq_vu/m2_7src/points.json']),
                   ('VE R2 (4 素材)', ['/tmp/eqq2/1280x720_10s_n4/points.json'])):
    D = load(path)
    tiers = sorted({t for m in D for t in D[m] if t != 'libx264'})
    print(f'\n══ {name} ══')
    for t in tiers:
        mats = [m for m in sorted(D) if t in D[m] and 'libx264' in D[m]]
        if len(mats) < 3:
            continue
        rows = []
        for hold in mats:
            p = pooled(t, [m for m in mats if m != hold])
            if not p:
                continue
            a, b = p
            anc = D[hold]['libx264']
            top = anc[0][1]
            crowded = sum(1 for _, v in anc if v >= 98.0)
            slope_hi = (anc[1][1] - anc[0][1]) / (anc[1][0] - anc[0][0]) if len(anc) > 1 else 0
            rows.append((hold, top, crowded, slope_hi, dvm(hold, t, a, b)))
        rows.sort(key=lambda r: -r[4])
        print(f'  {t}')
        print(f'    {"素材":<22}{"锚顶VMAF":>9}{"≥98个数":>9}{"高端斜率":>9}{"LOO ΔVMAF":>11}')
        for h, top, cr, sh, w in rows:
            print(f'    {h:<22}{top:>9.2f}{cr:>9}{sh:>9.2f}{w:>11.2f}')
        # 剔除「锚顶 ≥98」或「高端斜率 >-0.3」的素材后重测
        keep = [h for h, top, cr, sh, w in rows if top < 98.0 or sh <= -0.3]
        if len(keep) >= 2 and len(keep) < len(mats):
            worst = 0.0
            for hold in keep:
                p = pooled(t, [m for m in keep if m != hold])
                if p:
                    worst = max(worst, dvm(hold, t, *p))
            print(f'    ⇒ 剔除可分辨性差的素材后（{len(keep)}/{len(mats)} 留出）'
                  f'worst={worst:.2f} ' + ('✅ <1.0' if worst < 1 else '❌'))
        else:
            print(f'    ⇒ 无可剔除素材（{len(keep)}/{len(mats)} 符合剔除条件）')