"""第五版落表值：**按素材去重**的池化（修正第四版的同素材双倍权重缺陷）。

第四版缺陷：把 `new5_raw`/`new4_raw` 的 6s（VU）与 10s（VE）口径当作两个独立样本
⇒ 这两条素材在池化中被计两次，等于双倍权重 ⇒ vp9 LOO 被从 4.59 推到 6.71（超≤5.9）。
修正：锚点已两仓统一到 18/21/24/27/30 ⇒ 同素材的两口径**可合并**（锚点曲线取并集），
按素材去重后每条只计一次。
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
           '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src_6s_rav1e/points.json',
           '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src_6s_rav1e_s10/points.json',
           '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_anchorA/points.json'])
VE = load(['/tmp/eqq2/1280x720_10s_n4/points.json',
           '/tmp/eqq2/1280x720_10s_n2/points.json',
           '/tmp/eqq2/1280x720_10s_anchorB/points.json'])

# ★ 按素材名去重合并（同素材两口径的锚点/曲线取并集）
POOL = defaultdict(lambda: defaultdict(dict))
for src in (VU, VE):
    for m, tiers in src.items():
        for t, d in tiers.items():
            for k, v in d.items():
                POOL[m][t][k] = v          # 同 key 取后者覆盖（值相同，因锚点口径已统一）
POOL = dict(POOL)
MATS = sorted(POOL)
print(f'★ 按素材去重后：{len(MATS)} 条 —— {MATS}\n')


def fit_pool(tier, anchors, mats, leave_out=None):
    """池化拟合 + LOO（leave_out=None 时只返回 a/b）。"""
    ms = [m for m in mats if tier in POOL.get(m, {}) and 'libx264' in POOL[m]]
    if leave_out is not None:
        ms = [m for m in ms if m != leave_out]
    xs, ys, per = [], [], []
    for m in ms:
        sub = [(float(c), v) for c, v in sorted(POOL[m]['libx264'].items()) if int(c) in anchors]
        r = CH.calibrate_tier(tier, sub, sorted(POOL[m][tier].items()))
        if 'a' in r and len(r['points']) >= 2:
            xs += [x for x, _ in r['points']]
            ys += [y for _, y in r['points']]
            per.append(r['points'])
    if len(xs) < 2 or not per:
        return None
    a, _ = CH.fit_line(xs, ys)
    b = statistics.median([statistics.median([y - a * x for x, y in p]) for p in per])
    return a, b, per


def loo(tier, anchors, mats):
    worst, per = 0.0, {}
    for hold in mats:
        if tier not in POOL.get(hold, {}) or 'libx264' not in POOL[hold]:
            continue
        r = fit_pool(tier, anchors, mats, leave_out=hold)
        if not r:
            continue
        a, b, _ = r
        curve = sorted(POOL[hold][tier].items())
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


TIERS = ['libx265', 'libvpx-vp9', 'libaom-av1', 'libsvtav1', 'librav1e@10', 'librav1e']
print(f'{"档位":<14}{"a":>9}{"b":>11}{"样本":>5}{"LOO":>8}   门禁')
print(f'{"":<14}{"":>9}{"":>11}{"":>5}{"":>8}   (软编≤5.9 / rav1e≤7.5)')
rows = []
for t in TIERS:
    ms = [m for m in MATS if t in POOL.get(m, {}) and 'libx264' in POOL[m]]
    if len(ms) < 4:
        print(f'{t:<14} 样本不足（{len(ms)}）')
        continue
    r = fit_pool(t, Bv, ms)
    if not r:
        print(f'{t:<14} 拟合失败')
        continue
    a, b, _ = r
    w, per = loo(t, Bv, ms)
    gate = 7.5 if t.startswith('librav1e') else 5.9
    ok = w <= gate
    rows.append((t, a, b, len(ms), w, gate, ok))
    print(f'{t:<14}{a:>9.4f}{b:>+11.4f}{len(ms):>5}{w:>8.2f}   ≤{gate} {"✅" if ok else "❌"}')

print('\n=== 各档最差留出素材 ===')
for t, a, b, n, w, gate, ok in rows:
    _, per = loo(t, Bv, [m for m in MATS if t in POOL.get(m, {}) and 'libx264' in POOL[m]])
    if per:
        m, v = max(per.items(), key=lambda x: x[1])
        print(f'  {t:<14} {m:<22} {v:.2f}')

print('\n=== 第五版落表候选 ===')
for t, a, b, n, w, gate, ok in rows:
    print(f"    '{t}': ({round(a,4)}, {round(b,4)}),   # LOO {w:.2f} ≤{gate} {'✅' if ok else '❌'}")

print('\n=== 与第四版对比 ===')
V4 = {'libx265': (1.0943, -2.4562), 'libvpx-vp9': (1.9736, -12.3252),
      'libaom-av1': (2.2531, -20.5904), 'libsvtav1': (2.2391, -16.7060),
      'librav1e@10': (7.8373, -95.5520), 'librav1e': (7.9348, -96.0822)}
V4LOO = {'libx265': 4.49, 'libvpx-vp9': 6.71, 'libaom-av1': 5.00,
         'libsvtav1': 5.19, 'librav1e@10': 7.15, 'librav1e': 7.41}
print(f'  {"档位":<14}{"第四版 (a,b)":>24}{"第五版 (a,b)":>24}{"LOO变化":>12}')
for t, a, b, n, w, gate, ok in rows:
    o = V4.get(t, (0, 0))
    print(f'  {t:<14}({o[0]:.4f},{o[1]:+.3f})      ({a:.4f},{b:+.3f})      '
          f'{V4LOO[t]:.2f}→{w:.2f}')