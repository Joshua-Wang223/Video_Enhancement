"""v8 表值差异根因定位：逐层 dump 池化快照 + 逐折 LOO，穷尽差异源。

目标：解释「我的 x265=1.0979」vs「cron 的 x265=1.0908」。
方法：把每个可能成为差异源的环节单独打印/落盘，逐项哈希比对。
"""
import hashlib
import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path

_s = importlib.util.spec_from_file_location(
    'eqq', '/mnt/d/Workspace_Python/Video_Enhancement/Accessory/probe/calibrate_equal_quality.py')
CH = importlib.util.module_from_spec(_s)
_s.loader.exec_module(CH)
Bv = [18, 21, 24, 27, 30]

CRON = [
    '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src/points_cache.json',
    '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src_6s_rav1e/points.json',
    '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src_6s_rav1e_s10/points.json',
    '/tmp/eqq_uni_6s/new1/points.json',
    '/tmp/eqq_uni_6s/word_world_2/points.json',
    '/tmp/eqq_uni_6s/bbc_s01e01/points.json',
    '/tmp/eqq_uni_6s/bbc_s03e01/points.json',
    '/tmp/eqq_uni_6s/bbc_s05e01/points.json',
    '/tmp/eqq_uni_10s/anim_10s/points.json',
    '/tmp/eqq_uni_10s/anim_subs_10s/points.json',
    '/tmp/eqq_uni_10s/dark_10s/points.json',
    '/tmp/eqq_uni_10s/ui_10s/points.json',
    '/tmp/eqq_uni_10s/texture_10s/points.json',
    '/tmp/eqq2/1280x720_10s_n4/points.json',
    '/tmp/eqq2/1280x720_10s_n2/points.json',
    '/tmp/eqq2/1280x720_10s_anchorB/points.json',
    '/tmp/eqq2/1280x720_10s_bbc_anchorB/points.json',
]
ANCHOR_A = '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_anchorA/points.json'


def build(files):
    acc = defaultdict(list)
    for f in files:
        p = Path(f)
        if not p.is_file():
            continue
        for k, v in json.loads(p.read_text()).items():
            pr = k.split('|')
            vm = (v.get('m') or v).get('vmaf')
            if vm is None:
                continue
            acc[(pr[0], pr[-2], float(f'{float(pr[-1]):g}'))].append(float(vm))
    pool = defaultdict(lambda: defaultdict(dict))
    for (m, t, pm), vals in acc.items():
        pool[m][t][pm] = sum(vals) / len(vals)
    return dict(pool), acc


def fit(tier, pool, mats):
    ms = [m for m in mats if tier in pool.get(m, {}) and 'libx264' in pool[m]]
    xs, ys, per = [], [], []
    for m in ms:
        sub = [(float(c), v) for c, v in sorted(pool[m]['libx264'].items()) if int(c) in Bv]
        r = CH.calibrate_tier(tier, sub, sorted(pool[m][tier].items()))
        if 'a' in r and len(r['points']) >= 2:
            xs += [x for x, _ in r['points']]
            ys += [y for _, y in r['points']]
            per.append(r['points'])
    if not per:
        return None, 0, 0
    a, _ = CH.fit_line(xs, ys)
    b = statistics.median([statistics.median([y - a * x for x, y in p]) for p in per])
    return (a, b), len(xs), len(ms)


def h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     default=str).encode()).hexdigest()[:12]


print('=' * 70)
print('第1 层：池化快照哈希')
print('=' * 70)
variants = {
    'cron(16文件)': CRON,
    'mine(16+m2_anchorA)': CRON + [ANCHOR_A],
}
pools = {}
for name, files in variants.items():
    P, acc = build(files)
    pools[name] = P
    # 只哈希 B 套相关的点（拟合真正使用的部分）
    bset = {}
    for (m, t, pm), vs in sorted(acc.items()):
        if int(pm) in Bv or t != 'libx264':
            bset[f'{m}|{t}|{pm:g}'] = round(sum(vs) / len(vs), 9)
    print(f'  {name:<22} 素材 {len(P):>2}  ACC {len(acc):>4}  '
          f'B套相关点 {len(bset):>4}  哈希={h(bset)}')
print()
a1 = {f'{m}|{t}|{pm:g}': sum(vs) / len(vs) for (m, t, pm), vs in build(CRON)[1].items()}
a2 = {f'{m}|{t}|{pm:g}': sum(vs) / len(vs) for (m, t, pm), vs in build(CRON + [ANCHOR_A])[1].items()}
only_a2 = {k: v for k, v in a2.items() if k not in a1}
print(f'  仅 mine 有的键 {len(only_a2)} 个（应全为 A 套 crf22/26/34）：')
for k in sorted(only_a2)[:6]:
    print(f'    {k}')
in_b = [k for k in only_a2 if int(k.split("|")[-1]) in Bv]
print(f'  ⇒ 其中落在 B 套内的：**{len(in_b)} 个**' + ('（不应有）' if in_b else '✅ 零贡献'))

print()
print('=' * 70)
print('第 2 层：拟合输入（xs, ys）哈希')
print('=' * 70)
tier = 'libx265'
for name, P in pools.items():
    MATS = sorted(P)
    a, npts, nmat = fit(tier, P, MATS)
    # 重建 xs/ys 以便哈希
    xs, ys = [], []
    for m in MATS:
        sub = [(float(c), v) for c, v in sorted(P[m]['libx264'].items()) if int(c) in Bv]
        r = CH.calibrate_tier(tier, sub, sorted(P[m][tier].items()))
        if 'a' in r and len(r['points']) >= 2:
            xs += [x for x, _ in r['points']]
            ys += [y for _, y in r['points']]
    print(f'  {name:<22} a={a[0]:.10f} b={a[1]:+.6f}  '
          f'拟合点 {npts}  素材 {nmat}  哈希={h([round(x, 9) for x in xs])}')

print()
print('=' * 70)
print('第 3 层：素材是否被过滤（calibrate_tier 返回 a 的素材集合）')
print('=' * 70)
for name, P in pools.items():
    MATS = sorted(P)
    ok = []
    for m in MATS:
        sub = [(float(c), v) for c, v in sorted(P[m]['libx264'].items()) if int(c) in Bv]
        r = CH.calibrate_tier(tier, sub, sorted(P[m][tier].items()))
        if 'a' in r and len(r['points']) >= 2:
            ok.append((m, round(r['a'], 4)))
    print(f'  {name}:参与拟合 {len(ok)} 条')
    for m, a_ in ok:
        print(f'    {m:<36} a={a_}')

print()
print('=' * 70)
print('第 4 层：LOO 逐折明细（cron 声称 x265 LOO 3.37）')
print('=' * 70)
for name, P in pools.items():
    MATS = sorted(P)
    worst = 0.0
    rows = []
    for hold in MATS:
        r, _, _ = fit(tier, P, [m for m in MATS if m != hold])
        if not r:
            continue
        a, b = r
        curve = sorted(P[hold][tier].items())
        iso = [(p, v) for (p, _), v in
               zip(curve, CH.pava_nonincreasing([v for _, v in curve]))]
        hw, ne = 0.0, 0
        for crf, want in sorted(P[hold]['libx264'].items()):
            if int(crf) not in Bv:
                continue
            g = CH.vmaf_at_param(iso, a * crf + b)
            if g is not None:
                hw = max(hw, abs(g - want))
                ne += 1
        rows.append((hold, hw, ne))
        worst = max(worst, hw)
    rows.sort(key=lambda x: -x[1])
    print(f'  {name}: worst={worst:.2f}')
    for m, v, ne in rows[:5]:
        print(f'    {m:<36} {v:.2f}（{ne} 评估点）')

print()
print('=' * 70)
print('第 5 层：cron 的 1.0908 能否由某种子集复现？')
print('=' * 70)
POOL = pools['cron(16文件)']
ALL = sorted(POOL)
target = 1.0908
#尝试：排除某素材 / 限定某时长侧 / 只用部分档位
import itertools
found = []
# (a) 排除单条素材
for m in ALL:
    sub = [x for x in ALL if x != m]
    rr, _, _ = fit(tier, POOL, sub)
    if rr and abs(rr[0] - target) < 5e-5:
        found.append(f'排除 {m} => a={rr[0]:.4f}')
# (b) 只用 6s 侧（排除带 _10s 与 eqq2 独有的）
six = [m for m in ALL if not m.endswith('_10s.mp4')]
r, _, _ = fit(tier, POOL, six)
print(f'  只用非 _10s 素材（{len(six)} 条）: a={r[0]:.4f}' if r else '  n/a')
# (c) 只用 10s 侧
ten = [m for m in ALL if m.endswith('_10s.mp4') or m in
       ('new1.mp4', 'new4_raw.mp4', 'new5_raw.mp4', 'word_world_2.mp4',
        'S01E01._The_New_Stall.mp4', 'S03E01._James_and_Alice.mp4',
        'S05E01._Forever_Friends.mp4')]
r, _, _ = fit(tier, POOL, ten)
print(f'  只用 10s 侧素材（{len(ten)} 条）: a={r[0]:.4f}' if r else '  n/a')
for f in found:
    print(f'  {f}')