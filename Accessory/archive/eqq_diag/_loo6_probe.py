"""第七版落表值：**同 key 取均值**合并（消除第六版的顺序依赖）。

第六版遗留隐患：`new5_raw`/`new4_raw` 在 VU(6s) 与 VE(10s) 两侧都有同名锚点，
但 VMAF 不同（`crf30` 差1.75 —— VMAF 曲线随片段长度变化，非测量误差）。
合并时「后者覆盖前者」⇒ **表值依赖文件加载顺序**（实测三规则 a差 ~1.5%）。

本版用**同 key 取算术均值**（顺序无关，可复现）：
  · 同素材同参数的所有重复测量视为该点的多个独立观测，取均值；
  · 各素材仍**只计一次**（保持第五版起的「按素材去重」口径）。

口径不变：
  · 锚点统一 `18/21/24/27/30`（B 套，两仓一致）
  · 720p prep / `n_subsample=1`
  · 按素材去重（不把同素材的 6s 与 10s 当两个独立样本）
  · 门禁按编码器分档：软编 ≤5.9 / rav1e ≤7.5
"""
import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path


def lh(p, tag):
    s = importlib.util.spec_from_file_location(tag, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


CH = lh('/mnt/d/Workspace_Python/Video_Enhancement/Accessory/probe/calibrate_equal_quality.py', 'ch')
Bv = [18, 21, 24, 27, 30]

FILES = [
    # ── 仅 VU 6s 侧（探针：不含庚/己新数据）──
    '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src/points_cache.json',
    '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src_6s_rav1e/points.json',
    '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_7src_6s_rav1e_s10/points.json',
    '/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_anchorA/points.json',
]

# ★ 同 key 收集全部观测，最后取均值 ⇒ 与文件加载顺序无关
ACC = defaultdict(list)
SRC_TAGS = defaultdict(set)
for f in FILES:
    p = Path(f)
    if not p.is_file():
        print(f'  [skip缺失] {f}')
        continue
    tag = p.parent.name
    for k, v in json.loads(p.read_text()).items():
        pr = k.split('|')
        mat, tier, param = pr[0], pr[-2], float(f'{float(pr[-1]):g}')
        vm = (v.get('m') or v).get('vmaf')
        if vm is None:
            continue
        ACC[(mat, tier, param)].append(float(vm))
        SRC_TAGS[(mat, tier, param)].add(tag)

POOL = defaultdict(lambda: defaultdict(dict))
dup_count = 0
for (mat, tier, param), vals in ACC.items():
    POOL[mat][tier][param] = sum(vals) / len(vals)
    if len(vals) > 1:
        dup_count += 1
POOL = dict(POOL)
MATS = sorted(POOL)

print(f'\n★ 同 key 取均值后：{len(MATS)} 条素材；合并重复点 {dup_count}/{len(ACC)} 个')
print(f'  素材: {MATS}\n')

#重复点的实际跨度（确认合并合理，不是把不同参数混在一起）
print('=== 重复测量的点（已取均值）===')
worst = []
for (mat, tier, param), vals in ACC.items():
    if len(vals) > 1 and (tier == 'libx264' or tier.startswith('librav1e')):
        sp = max(vals) - min(vals)
        if sp > 0.05:
            worst.append((sp, mat, tier, param, vals))
worst.sort(reverse=True)
for sp, mat, tier, param, vals in worst[:12]:
    print(f'  {mat:<34}{tier:<13}{param:>6g}  n={len(vals)} 跨度={sp:.3f}  {sorted(vals)}')
if not worst:
    print('  （无显著差异）')


def fit_pool(tier, mats, leave_out=None):
    ms = [m for m in mats if tier in POOL.get(m, {}) and 'libx264' in POOL[m]]
    if leave_out is not None:
        ms = [m for m in ms if m != leave_out]
    xs, ys, per = [], [], []
    for m in ms:
        sub = [(float(c), v) for c, v in sorted(POOL[m]['libx264'].items())
               if int(c) in Bv]
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


def loo(tier, mats):
    worst_dv, per = 0.0, {}
    for hold in mats:
        if tier not in POOL.get(hold, {}) or 'libx264' not in POOL[hold]:
            continue
        r = fit_pool(tier, mats, leave_out=hold)
        if not r:
            continue
        a, b, _ = r
        curve = sorted(POOL[hold][tier].items())
        iso = [(p, v) for (p, _), v in
               zip(curve, CH.pava_nonincreasing([v for _, v in curve]))]
        hw, n_eval = 0.0, 0
        for crf, want in sorted(POOL[hold]['libx264'].items()):
            if int(crf) not in Bv:
                continue
            g = CH.vmaf_at_param(iso, a * crf + b)
            if g is not None:          # ★ 0 评估点 ⇒ 判 inf（模型失效），非PASS
                hw = max(hw, abs(g - want))
                n_eval += 1
        # 留出素材若所有 B 套锚点都反查不到 ⇒ 该折无效，记 inf 而非 0
        if n_eval == 0:
            hw = float('inf')
        per[hold] = hw
        worst_dv = max(worst_dv, hw)
    return worst_dv, per


TIERS = ['libx265', 'libvpx-vp9', 'libaom-av1', 'libsvtav1', 'librav1e@10', 'librav1e']
print('\n=== 第七版（顺序无关）===')
print(f'{"档位":<14}{"a":>9}{"b":>11}{"样本":>5}{"LOO":>8}   门禁')
rows = []
for t in TIERS:
    ms = [m for m in MATS if t in POOL.get(m, {}) and 'libx264' in POOL[m]]
    if len(ms) < 4:
        print(f'{t:<14} 样本不足（{len(ms)}）')
        continue
    r = fit_pool(t, ms)
    if not r:
        print(f'{t:<14} 拟合失败')
        continue
    a, b, _ = r
    w, per = loo(t, ms)
    gate = 7.5 if t.startswith('librav1e') else 5.9
    ok = w <= gate
    rows.append((t, a, b, len(ms), w, gate, ok))
    print(f'{t:<14}{a:>9.4f}{b:>+11.4f}{len(ms):>5}{w:>8.2f}   ≤{gate} {"✅" if ok else "❌"}')

print('\n=== 各档最差留出素材 ===')
for t, a, b, n, w, gate, ok in rows:
    _, per = loo(t, [m for m in MATS if t in POOL.get(m, {}) and 'libx264' in POOL[m]])
    if per:
        m, v = max(per.items(), key=lambda x: x[1])
        print(f'  {t:<14} {m:<34} {v:.2f}')

# 与第六版对比
SIXTH = {'libx265': (1.0877, -2.4279), 'libvpx-vp9': (1.9933, -15.6126),
         'libaom-av1': (2.2671, -20.9112), 'libsvtav1': (2.1886, -16.3312),
         'librav1e@10': (7.8982, -104.5072), 'librav1e': (7.9494, -101.2811)}
SIXTH_LOO = {'libx265': 3.98, 'libvpx-vp9': 4.59, 'libaom-av1': 5.13,
             'libsvtav1': 4.66, 'librav1e@10': 5.92, 'librav1e': 5.37}
print('\n=== 与第六版对比 ===')
print(f'  {"档位":<14}{"第六版(a,b)":>22}{"第七版(a,b)":>22}   LOO 变化')
for t, a, b, n, w, gate, ok in rows:
    a0, b0 = SIXTH[t]
    print(f'  {t:<14}({a0:.4f},{b0:+.3f}){"":>4}({a:.4f},{b:+.3f}){"":<4}'
          f'{SIXTH_LOO[t]:.2f} → {w:.2f} ({w - SIXTH_LOO[t]:+.2f})')

print('\n=== 第七版落表候选 ===')
for t, a, b, n, w, gate, ok in rows:
    print(f"    '{t}': ({a:.4f}, {b:.4f}, 0, {255 if 'rav1e' in t else (51 if t == 'libx265' else 63)}),"
          f'   # LOO {w:.2f} ≤{gate} {"✅" if ok else "❌"}')