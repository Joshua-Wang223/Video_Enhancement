#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""等质量表落表器 —— 池化 + 拟合 + LOO 门禁 + 顺序无关断言。

从 `Accessory/data/eqq_calibration/points/`读 points，聚合成
`QUALITY_MAP` 的 `(a, b, quality, tag)` 四元组，并跑 3 项硬断言。

三项断言（任一不过 ⇒ `exit 1`，**不自行放水**）
------------------------------------------------
1. **数据完整性**：跨口径同名素材数只作**告警**，不作失败条件。
   ⚠ 历史上这里曾断言「必须为 0」，是**错的** —— 那 4 个 `legacy10s` 文件
   （316 点）是7 条素材的 10s 侧**有效观测**，不是脏数据。舍弃它们换来的更低 LOO
   是样本覆盖变窄导致的**虚假改善**（见 git `eda957d`）。真正的保证来自第 2 项。
2. **顺序无关性回归**：3 个随机种子打乱文件顺序复算，6 档表值须逐位一致（1e-12）。
   合并规则「同 key 取均值」本身是对称的，故顺序无关成立 —— 这是**实测**结论，
   不是「结构性保证」那种口头声明。
3. **LOO 门禁**：留一素材 worst-case ΔVMAF，软编 ≤5.9 / rav1e ≤7.5。
   ⚠ 0 评估点必须记 `inf`（模型失效），**不能当 PASS** —— 曾出现「预测越界⇒回查 None⇒被当 PASS」。

用法
----
    python3 eqq_pool_fit_table.py \
        --points-dir ../data/eqq_calibration/points \
        --out /tmp/table.txt
"""
import argparse
import importlib.util
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('eqq_harness', _HERE / 'calibrate_equal_quality.py')
CH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CH)

TIERS = ['libx265', 'libvpx-vp9', 'libsvtav1', 'libaom-av1', 'librav1e@10', 'librav1e',
         # NVENC 硬编（需 GPU 点数据；无数据时自动打印「拟合失败」并跳过）
         'h264_nvenc', 'hevc_nvenc', 'av1_nvenc']
#: LOO 门限。软编 0~63 刻度、rav1e 0~255 刻度，rav1e 天然更宽⇒ 单独放宽。
#: NVENC 的 CQ 轴同为 0~63 刻度 ⇒ 按软编门限。
GATE = {t: (7.5 if t.startswith('librav1e') else 5.9) for t in TIERS}
SEEDS = (1, 2, 3)


def find_points(points_dir, side):
    """收集一个口径下的所有 points 文件（跳过 vmaf 原始输出与映射表）。"""
    d = Path(points_dir) / side
    return sorted(f for f in d.glob('*.json')
                  if '_vmaf' not in f.name and 'clip_name_mapping' not in f.name) \
        if d.is_dir() else []


def build_pool(files):
    """同 (素材, 档位, 参数) 收集全部观测取均值。

    ⚠ 均值法只适合**真噪声**下的重复观测。若重复来自系统性差异（时长/分辨率/锚点集），
    必须选边舍弃而非平均 —— 见 MANIFEST.md「三条数据完整性教训」。
    """
    acc = defaultdict(list)
    for f in files:
        p = Path(f)
        if not p.is_file():
            print(f'  [skip缺失] {f}')
            continue
        for k, v in json.loads(p.read_text(encoding='utf-8')).items():
            pr = k.split('|')
            mat, tier, param = pr[0], pr[-2], float(f'{float(pr[-1]):g}')
            vm = (v.get('m') or v).get('vmaf')
            if vm is None:
                continue
            acc[(mat, tier, param)].append(float(vm))
    pool = defaultdict(lambda: defaultdict(dict))
    dup = 0
    for (mat, tier, param), vals in acc.items():
        pool[mat][tier][param] = sum(vals) / len(vals)
        if len(vals) > 1:
            dup += 1
    return dict(pool), dup, acc


def side_of(path):
    p = Path(path).parent.name
    return p if p in ('6s', '10s', 'legacy10s') else 'other'


def check_cross_side(files):
    """同一素材名出现在多个口径 ⇒ 返回 {素材: {口径,...}}。

    ⚠ 只作告警。这些是**有效观测**，合并时按「同 key 取均值」处理。
    曾把它们当脏数据舍弃，导致 LOO 虚假改善 —— 见 git `eda957d`。
    """
    side = defaultdict(set)
    for f in files:
        sd = side_of(f)
        for k in json.loads(Path(f).read_text(encoding='utf-8')):
            side[k.split('|')[0]].add(sd)
    return {m: s for m, s in side.items() if len(s) > 1}


def fit_pool(pool, tier, mats, leave_out=None):
    """逐素材标定 tier → 汇集 (cq, vmaf) 点 → 直线拟合。

    截距用「各素材残差中位数的中位数」（robust），不用均值 —— 单素材的离群点
    （如UI 素材的极陡段）会带偏均值截距。
    """
    ms = [m for m in mats if tier in pool.get(m, {}) and 'libx264' in pool[m]]
    if leave_out is not None:
        ms = [m for m in ms if m != leave_out]
    xs, ys, per = [], [], []
    for m in ms:
        sub = [(float(c), v) for c, v in sorted(pool[m]['libx264'].items())
               if int(c) in CH.ANCHOR_CRFS]
        r = CH.calibrate_tier(tier, sub, sorted(pool[m][tier].items()))
        if 'a' in r and len(r['points']) >= 2:
            xs += [x for x, _ in r['points']]
            ys += [y for _, y in r['points']]
            per.append(r['points'])
    if len(xs) < 2 or not per:
        return None
    a, _ = CH.fit_line(xs, ys)
    b = statistics.median([statistics.median([y - a * x for x, y in p]) for p in per])
    return a, b


def loo(pool, tier, mats):
    """留一法：每次hold out 一条素材，用其余拟合后在该素材上算 worst ΔVMAF。"""
    worst, per = 0.0, {}
    for hold in mats:
        if tier not in pool.get(hold, {}) or 'libx264' not in pool[hold]:
            continue
        r = fit_pool(pool, tier, mats, leave_out=hold)
        if not r:
            continue
        a, b = r
        curve = sorted(pool[hold][tier].items())
        iso = [(p, v) for (p, _), v in
               zip(curve, CH.pava_nonincreasing([v for _, v in curve]))]
        hw, n_eval = 0.0, 0
        for crf, want in sorted(pool[hold]['libx264'].items()):
            if int(crf) not in CH.ANCHOR_CRFS:
                continue
            g = CH.vmaf_at_param(iso, a * crf + b)
            if g is not None:          # ★ 0 评估点 ⇒ inf（模型失效），非 PASS
                hw = max(hw, abs(g - want))
                n_eval += 1
        per[hold] = hw if n_eval else float('inf')
        worst = max(worst, per[hold])
    return worst, per


def compute(pool, mats, tiers):
    out = {}
    for t in tiers:
        ms = [m for m in mats if t in pool.get(m, {}) and 'libx264' in pool.get(m, {})]
        out[t] = fit_pool(pool, t, ms) if len(ms) >= 4 else None
    return out


def main():
    ap = argparse.ArgumentParser(description='等质量表落表器（池化+拟合+LOO+断言）')
    ap.add_argument('--points-dir', default=str(_HERE.parent / 'data' / 'eqq_calibration' / 'points'))
    ap.add_argument('--sides', default='6s,10s,legacy10s',
                    help='参与聚合的口径目录（逗号分隔）。'
                         'legacy10s = 4 个 eqq2 10s 文件 + m2_anchorA，'
                         '是7 条素材的 10s 侧有效观测，**不可剔除**')
    ap.add_argument('--out', default='', help='落表候选写入此文件（默认只打印）')
    ap.add_argument('--axis', choices=('cq', 'qp'), default='cq',
                    help='质量轴：cq → 候选 QUALITY_MAP / 量程取 CQ 轴；'
                         'qp → 候选 QUALITY_MAP_QP（VE 特有 D2b）/ 量程取 QP 轴。'
                         '⚠ 两条轴的点数据必须分目录（--sides 分开），勿混池。')
    args = ap.parse_args()

    files = [f for s in args.sides.split(',') if s for f in find_points(args.points_dir, s.strip())]
    if not files:
        raise SystemExit(f'未找到 points 文件：{args.points_dir}')
    # ★ 池化前必须打印这四个数并与预期比对（memory 教训③：素材名去重 ≠ 数据完整）
    print(f'文件数 = {len(files)}')
    per_file = {f: len(json.loads(Path(f).read_text(encoding='utf-8'))) for f in files}
    for f, n in per_file.items():
        print(f'  {n:>5}  {f}')
    print(f'逐文件点数求和 = {sum(per_file.values())}')

    pool, dup, acc = build_pool(files)
    mats = sorted(pool)
    print(f'ACC 总点数 = {len(acc)}')
    print(f'合并重复点 = {dup}')
    print(f'素材池 = {len(mats)} 条')

    cross = check_cross_side(files)
    print(f'\n跨口径同名素材：{len(cross)} 条'
          f'{"（有效观测，按同 key 取均值合并）" if cross else ""}')
    for mname, sds in sorted(cross.items()):
        print(f'    {mname:<40}{sorted(sds)}')

    print(f'\n{"档位":<14}{"a":>9}{"b":>11}{"样本":>5}{"LOO":>8}   门禁')
    rows, base, fails = [], {}, []
    for t in TIERS:
        ms = [m for m in mats if t in pool.get(m, {}) and 'libx264' in pool[m]]
        r = fit_pool(pool, t, ms)
        if not r:
            print(f'{t:<14} 拟合失败（样本 {len(ms)} < 4）')
            continue
        a, b = r
        w, per = loo(pool, t, ms)
        g = GATE[t]
        ok = w <= g
        base[t] = (a, b)
        rows.append((t, a, b, len(ms), w, g, ok, per))
        if not ok:
            fails.append(t)
        print(f'{t:<14}{a:>9.4f}{b:>+11.4f}{len(ms):>5}{w:>8.2f}   ≤{g} {"✅" if ok else "❌"}')

    print('\n各档最差留出素材')
    for t, a, b, n, w, g, ok, per in rows:
        if per:
            m, v = max(per.items(), key=lambda x: x[1])
            print(f'  {t:<14} {m:<40}{v:.2f}')

    print(f'\n=== 顺序无关性回归断言（{len(SEEDS)} seed 打乱文件顺序）===')
    assert_fail = []
    for seed in SEEDS:
        sh = files[:]
        random.Random(seed).shuffle(sh)
        p2, _, _ = build_pool(sh)
        got = compute(p2, mats, TIERS)
        bad = [f'{t}: {got[t]} != {base[t]}' for t in TIERS
               if base.get(t) and got.get(t)
               and (abs(got[t][0] - base[t][0]) > 1e-12 or abs(got[t][1] - base[t][1]) > 1e-12)]
        assert_fail += [f'seed{seed}/{b}' for b in bad]
        print(f'  seed {seed}: {"✅ 6 档逐位一致" if not bad else "❌ " + "; ".join(bad)}')

    tbl_name = 'QUALITY_MAP_QP' if args.axis == 'qp' else 'QUALITY_MAP'
    print(f'\n=== 落表候选（轴={args.axis}；写入 src/utils/quality_map.py 的 {tbl_name}）===')
    lines = []
    for t, a, b, n, w, g, ok, per in rows:
        # hi 从**目标轴**量程取（不再硬编码 255/51/63）：
        #   cq → QUALITY_MAP→SIZE_MAP 回退；qp → QP_LIMITS（AV1=255，其余 51）
        tag = CH._axis_range(t, args.axis)[1]
        line = f"    '{t}': ({a:.4f}, {b:.4f}, 0, {tag}),   # LOO {w:.2f} ≤{g} {'✅' if ok else '❌'}"
        lines.append(line)
        print(line)

    if args.out:
        Path(args.out).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(f'\n→ {args.out}')

    print('\n=== 汇总 ===')
    if cross:
        print(f'⚠ {len(cross)} 条素材跨口径（有效观测，已合并；'
              f'若 LOO 明显变好请先核对此处数据完整性）')
    print('❌ 顺序无关性断言失败：' + '; '.join(assert_fail) if assert_fail
          else f'✅ 顺序无关性：{len(SEEDS)} seed × 6 档逐位一致')
    print(f'❌ 超门禁档位：{fails}（**不自行放水**，须先写清失败项与根因）' if fails
          else '✅ 全部档位 LOO 达标')
    return 1 if (assert_fail or fails) else 0


if __name__ == '__main__':
    sys.exit(main())