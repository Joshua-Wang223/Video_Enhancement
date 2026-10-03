"""统一到 B 套锚点（18/21/24/27/30）后的**9+3 素材合并**池化表值。

素材池（12 条，跨两仓实测）：
  · VU 7 素材（6s）：new5_raw / new4_raw / cc_anim_300s / cc_subs_105s /
    earth_dark_80s / ui_screen_10s / natgeo_grass_40s
    —— 软编 4 档 + rav1e native + rav1e@10 全测
  · VE 4 素材（10s）：new5_raw / new4_raw / new1 / word_world_2
    —— 软编 4 档 + rav1e@10（+ rav1e native 仅 2 素材，见 _n2）
  ⚠ new5_raw / new4_raw 在两仓都有（6s 与 10s 两个口径），按**分开计入**处理
    （下方按 tier 去重后统计），避免同素材双权重。

输出：每个档位的 pooled 斜率 a + 合并中位截距 b，以及 pooled LOO worst。
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
    """→ {(素材, 口径): {tier: {param: vmaf}}}；口径 = prep 时长（6s/10s）"""
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
            mat, tier, val = pr[0], pr[-2], pr[-1]
            if len(pr) == 4:
                dur = pr[1]          # 旧格式：素材|时长|档位|参数
            else:
                dur = '10' if '10s' in str(f) else '6'
            D[(mat, dur)][tier][float(f'{float(val):g}')] = float(vm)
    return dict(D)


VU = load(['/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src/points_cache.json',
           '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src_6s_rav1e/points.json',
           '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src_6s_rav1e_s10/points.json',
           '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_anchorA/points.json'])
VE = load(['/tmp/eqq2/1280x720_10s_n4/points.json',
           '/tmp/eqq2/1280x720_10s_n2/points.json',
           '/tmp/eqq2/1280x720_10s_anchorB/points.json'])

# (素材, 时长) 视作独立样本；同素材跨时长算两个口径
KEYS = sorted(set(VU) | set(VE))
print(f'样本（素材, 时长）共 {len(KEYS)} 条:')
for k in KEYS:
    src = 'VU' if k in VU else 'VE'
    tiers = sorted((VU.get(k) or VE[k]).keys())
    print(f'  {k[0]:<24} {k[1]:>3}s  [{src}]  {tiers}')

# 每 tier 收集有足够锚点的样本
print()
rows = []
for t in ('libx265', 'libvpx-vp9', 'libaom-av1', 'libsvtav1', 'librav1e@10', 'librav1e'):
    pool = []
    for k in KEYS:
        D, H = (VU, CV) if k in VU else (VE, CH)
        m = D.get(k) or VE.get(k)
        if t not in m or 'libx264' not in m:
            continue
        if len([c for c in m['libx264'] if int(c) in Bv]) < 4:
            continue
        pool.append((k, D, H, m))
    if len(pool) < 3:
        print(f'{t:<14} 样本不足（{len(pool)}）')
        continue
    # pooled a
    xs, ys = [], []
    for k, D, H, m in pool:
        sub = [(float(c), v) for c, v in m['libx264'].items() if int(c) in Bv]
        r = H.calibrate_tier(t, sub, [(p, v) for p, v in sorted(m[t].items())])
        if 'a' in r:
            xs += [x for x, _ in r['points']]
            ys += [y for _, y in r['points']]
    if len(xs) < 2:
        print(f'{t:<14} 拟合点不足')
        continue
    a, _ = CH.fit_line(xs, ys)
    # 各样本中位截距 → 合并中位数
    bs = []
    for k, D, H, m in pool:
        sub = [(float(c), v) for c, v in m['libx264'].items() if int(c) in Bv]
        r = H.calibrate_tier(t, sub, [(p, v) for p, v in sorted(m[t].items())])
        if 'a' in r:
            bs.append(statistics.median([y - a * x for x, y in r['points']]))
    b = statistics.median(bs)
    # pooled LOO
    worst = 0.0
    for hold in pool:
        tr = [p for p in pool if p[0] != hold[0]]     # 按素材留出（跨时长一起留）
        if len(tr) < 3:
            continue
        xs2, ys2, per = [], [], []
        for k, D, H, m in tr:
            sub = [(float(c), v) for c, v in m['libx264'].items() if int(c) in Bv]
            r = H.calibrate_tier(t, sub, [(p, v) for p, v in sorted(m[t].items())])
            if 'a' in r:
                xs2 += [x for x, _ in r['points']]
                ys2 += [y for _, y in r['points']]
                per.append(r['points'])
        if len(xs2) < 2 or not per:
            continue
        a2, _ = CH.fit_line(xs2, ys2)
        b2 = statistics.median([statistics.median([y - a2 * x for x, y in p]) for p in per])
        hk, _, _, hm = hold
        curve = sorted((VU.get(hk) or VE[hk])[t].items())
        iso = [(p, v) for (p, _), v in zip(curve, CH.pava_nonincreasing([v for _, v in curve]))]
        for crf, want in hm['libx264'].items():
            if int(crf) not in Bv:
                continue
            got = CH.vmaf_at_param(iso, a2 * crf + b2)
            if got is not None:
                worst = max(worst, abs(got - want))
    rows.append((t, a, b, len(pool), worst))
    print(f'{t:<14} a={a:>7.4f} b={b:>+9.4f}  样本={len(pool):>2}  LOO={worst:>6.2f}  '
          f'crf21→{a*21+b:.2f}')

print('\n=== 最终表值（统一 B 套锚点 18/21/24/27/30）===')
for t, a, b, n, w in rows:
    key = t if t != 'librav1e@10' else 'librav1e@10(override)'
    print(f"    '{key}': ({round(a,4)}, {round(b,4)}, ...),# LOO {w:.2f}")