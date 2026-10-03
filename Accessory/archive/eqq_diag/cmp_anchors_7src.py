"""同批 7 素材上「锚点集」敏感性分析。

背景：/tmp/eqq_vu/m2_7src（VU 侧 7 素材）的 libx264 锚点实测为 18/21/24/27/30（B 集），
      A 集（18/22/26/30/34）在这 7 素材上**没有实测数据** ⇒ 无法直接拟合 A。
      本脚本先用「可用锚点子集」做敏感性分析，回答根因问题：
        ΔVMAF 大，究竟来自「锚点取值」，还是来自「跨素材结构差异」？

方法：对同一批数据、同一档位，只改锚点子集，看 LOO worst 如何变化。
      若worst 对锚点集不敏感 ⇒ 根因是结构上限（与 R2 结论一致）。
      若对锚点集高度敏感 ⇒ 锚点选择才是可优化项。
"""
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HARNESS = '/mnt/d/Workspace_Python/VidUtils/probe/calibrate_equal_quality.py'
spec = importlib.util.spec_from_file_location('eqq', HARNESS)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

POINTS = {
    'VU-7src-soft': '/tmp/eqq_vu/m2_7src/points.json',
    'VU-7src-rav1e-native': '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src_6s_rav1e/points.json',
    'VU-7src-rav1e-s10': '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src_6s_rav1e_s10/points.json',
}


def load(path):
    D = defaultdict(lambda: defaultdict(dict))
    for k, v in json.loads(Path(path).read_text()).items():
        vm = (v.get('m') or {}).get('vmaf')
        if vm is None:
            continue
        m, t, val = k.split('|')
        D[m][t][float(val)] = float(vm)
    out = {}
    for m, tiers in D.items():
        out[m] = {t: sorted(d.items()) for t, d in tiers.items()}
    return out


def loo(D, tier, anchors, mats):
    """留一：用其余素材拟合 (a,b)，预测留出素材在 anchors 处的 VMAF。"""
    worst, detail = 0.0, {}
    for hold in mats:
        train = [m for m in mats if m != hold]
        xs, ys, bs = [], [], []
        ok = True
        for m in train:
            have = {c for c, _ in D[m].get('libx264', [])}
            sub = [(c, v) for c, v in D[m]['libx264'] if c in anchors]
            if len(sub) < 2:
                ok = False
                break
            r = C.calibrate_tier(tier, sub, D[m].get(tier, []))
            if 'a' not in r:
                ok = False
                break
            xs += [x for x, _ in r['points']]
            ys += [y for _, y in r['points']]
        if not ok or len(xs) < 2:
            continue
        a, _ = C.fit_line(xs, ys)
        for m in train:
            sub = [(c, v) for c, v in D[m]['libx264'] if c in anchors]
            r = C.calibrate_tier(tier, sub, D[m].get(tier, []))
            if 'a' in r:
                bs += [y - a * x for x, y in r['points']]
        if not bs:
            continue
        b = statistics.median(bs)
        tier_pts = D[hold].get(tier, [])
        if not tier_pts:
            continue
        av = C.pava_nonincreasing([y for _, y in tier_pts])
        iso = [(p, y) for (p, _), y in zip(tier_pts, av)]
        w = 0.0
        for crf, want in D[hold]['libx264']:
            if crf not in anchors:
                continue
            g = C.vmaf_at_param(iso, a * crf + b)
            if g is not None:
                w = max(w, abs(g - want))
        detail[hold] = w
        worst = max(worst, w)
    return worst, detail


B5 = [18.0, 21.0, 24.0, 27.0, 30.0]
SUBSETS = [
    ('B5  18/21/24/27/30 (实测全集)', B5),
    ('间隔3  18/24/30', [18.0, 24.0, 30.0]),
    ('宽跨度 18/21/27/30', [18.0, 21.0, 27.0, 30.0]),
    ('两端   18/30', [18.0, 30.0]),
    ('低端密 18/21/24/30', [18.0, 21.0, 24.0, 30.0]),
]

for tag, path in POINTS.items():
    D = load(path)
    mats = sorted(m for m in D if 'libx264' in D[m])
    have = sorted({c for m in mats for c, _ in D[m]['libx264']})
    print(f'\n{"="*74}\n数据集 {tag}  素材 {len(mats)}  实测libx264 锚点 {have}')
    for tier in sorted({t for m in mats for t in D[m] if t != 'libx264'}):
        if sum(1 for m in mats if tier in D[m]) < 3:
            continue
        print(f'  ── 档位 {tier}')
        for name, anc in SUBSETS:
            anc_use = [c for c in anc if c in have]
            if len(anc_use) < 2:
                continue
            w, det = loo(D, tier, anc_use, mats)
            flag = 'PASS' if w < 1.0 else 'FAIL'
            top = sorted(det.items(), key=lambda x: -x[1])[:2]
            tops = ', '.join(f'{k[:14]}:{v:.2f}' for k, v in top)
            print(f'     {name:<30} n={len(anc_use)} worst={w:6.3f} {flag}   最差: {tops}')