"""诊断 libvpx-vp9 LOO 6.71 的来源，判断「重标」能否收敛。

关键问题：6.71 是**训练内**就高（模型形式不足 ⇒ 重标无效），
还是**跨素材**才高（离群素材主导 ⇒ 剔除/加权可收敛）？
"""
import importlib.util, json, statistics
from collections import defaultdict
from pathlib import Path


def lh(p, tag):
    s = importlib.util.spec_from_file_location(tag, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


CH = lh('/mnt/d/Workspace_Python/Video_Enhancement/Accessory/probe/calibrate_equal_quality.py', 'ch')
CV = lh('/mnt/d/Workspace_Python/VidUtils/probe/calibrate_equal_quality.py', 'cv')
Bv = [18, 21, 24, 27, 30]


def load(srcs):
    D = defaultdict(lambda: defaultdict(dict))
    for f in srcs:
        p = Path(f)
        if not p.is_file():
            continue
        for k, v in json.loads(p.read_text()).items():
            pr = k.split('|')
            vm = (v.get('m') or v).get('vmaf')
            if vm is None:
                continue
            D[pr[0]][pr[-2]][float(f'{float(pr[-1]):g}')] = float(vm)
    return dict(D)


VU = load(['/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src/points_cache.json',
           '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_anchorA/points.json'])
VE = load(['/tmp/eqq2/1280x720_10s_n4/points.json',
           '/tmp/eqq2/1280x720_10s_anchorB/points.json'])

# 合并（素材名去重：new5_raw/new4_raw 在两仓都有，但锚点口径已统一为 B 套 ⇒ 可合并）
POOL = defaultdict(lambda: defaultdict(dict))
for src in (VU, VE):
    for m, tiers in src.items():
        for t, d in tiers.items():
            if t not in ('libx264', 'libvpx-vp9'):
                continue
            for k, v in d.items():
                if k not in POOL[m][t] or t == 'libvpx-vp9':
                    POOL[m][t][k] = v
POOL = dict(POOL)
MATS = sorted(m for m in POOL if 'libvpx-vp9' in POOL[m] and 'libx264' in POOL[m])
print(f'vp9 样本 {len(MATS)} 条: {MATS}\n')

print('=== 各样本自身拟合（训练内）ΔVMAF ===')
print(f'  {"素材":<22}{"a":>9}{"b":>10}{"ΔVMAF":>9}{"VMAF跨度":>10}')
INS = {}
for m in MATS:
    sub = [(float(c), v) for c, v in sorted(POOL[m]['libx264'].items()) if int(c) in Bv]
    curve = sorted(POOL[m]['libvpx-vp9'].items())
    r = CH.calibrate_tier('libvpx-vp9', sub, curve)
    if 'a' not in r:
        print(f'  {m:<22} 拟合失败')
        continue
    INS[m] = r['max_delta_vmaf']
    span = sub[0][1] - sub[-1][1]
    print(f'  {m:<22}{r["a"]:>9.4f}{r["b"]:>+10.3f}{r["max_delta_vmaf"]:>9.3f}{span:>10.1f}')

worst_in = max(INS.values()) if INS else 0
print(f'\n  训练内 worst = {worst_in:.3f}  ⇒  {"模型形式不足，重标无效" if worst_in > 1.0 else "训练内达标，误差来自跨素材"}')


def loo(mats, anchors):
    worst, per = 0.0, {}
    for hold in mats:
        train = [m for m in mats if m != hold]
        xs, ys, per_m = [], [], []
        for m in train:
            sub = [(float(c), v) for c, v in sorted(POOL[m]['libx264'].items()) if int(c) in anchors]
            r = CH.calibrate_tier('libvpx-vp9', sub, sorted(POOL[m]['libvpx-vp9'].items()))
            if 'a' in r:
                xs += [x for x, _ in r['points']]
                ys += [y for _, y in r['points']]
                per_m.append(r['points'])
        if len(xs) < 2 or not per_m:
            continue
        a, _ = CH.fit_line(xs, ys)
        b = statistics.median([statistics.median([y - a * x for x, y in p]) for p in per_m])
        curve = sorted(POOL[hold]['libvpx-vp9'].items())
        iso = [(p, v) for (p, _), v in zip(curve, CH.pava_nonincreasing([v for _, v in curve]))]
        hw = 0.0
        for crf, want in sorted(POOL[hold]['libx264'].items()):
            if int(crf) not in anchors:
                continue
            g = CH.vmaf_at_param(iso, a * crf + b)
            if g is not None:
                hw = max(hw, abs(g - want))
        per[hold] = hw
        worst = max(worst, hw)
    return worst, per


w, per = loo(MATS, Bv)
print(f'\n=== LOO（B套锚点，{len(MATS)} 样本）worst = {w:.3f} ===')
for m, v in sorted(per.items(), key=lambda x: -x[1]):
    ins = INS.get(m, 0)
    print(f'  {m:<22} LOO={v:>6.2f}   (自身 {ins:.2f}){"← 离群" if v > 5.9 else ""}')

# 剔除离群后重测
good = [m for m in MATS if per.get(m, 9) <= 5.9]
if 3 <= len(good) < len(MATS):
    w2, per2 = loo(good, Bv)
    print(f'\n  剔除超 5.9 的 {len(MATS)-len(good)} 条后（剩 {len(good)} 条）：worst = {w2:.3f}')
    print(f'    剔除: {[m for m in MATS if m not in good]}')