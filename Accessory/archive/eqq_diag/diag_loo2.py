"""根因验证：LOO 失败是「模型形式」还是「素材池被污染」？

对比 4 个素材集（同一批已测点，零重编码）：
  S1 全 4 素材（当前 R2 口径）
  S2 剔除 word_world_2（720x576 → 720p 上采样，VMAF 由缩放主导）
  S3 只 new5_raw + new4_raw（同为 1080p 实拍）
  S4 全 4 但锚点收窄到 crf22/26/30（去掉两端）
"""
import importlib.util, json, statistics
from collections import defaultdict
from pathlib import Path

R = Path('/mnt/d/Workspace_Python/Video_Enhancement')
spec = importlib.util.spec_from_file_location('eqq', R / 'Accessory/probe/calibrate_equal_quality.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

raw = {}
for tag in ('1280x720_10s_n4', '1280x720_10s_n2'):
    for k, v in json.loads((Path('/tmp/eqq2') / tag / 'points.json').read_text()).items():
        vm = (v.get('m') or {}).get('vmaf')
        if vm is not None:
            raw[k] = float(vm)
D = defaultdict(lambda: defaultdict(dict))
for k, vm in raw.items():
    m, t, val = k.split('|')
    D[m][t][float(val)] = vm
for m in D:
    for t in list(D[m]):
        if t != 'libx264':
            D[m][t] = sorted(D[m][t].items())


def loo(tier, mats, anchors=None):
    worst, per, bad = 0.0, {}, 0
    for hold in mats:
        train = [m for m in mats if m != hold]
        if len(train) < 1:
            continue
        xs, ys = [], []
        for m in train:
            a_ = [(c, v) for c, v in sorted(D[m]['libx264'].items())
                  if anchors is None or c in anchors]
            r = C.calibrate_tier(tier, a_, D[m][tier])
            if 'a' in r:
                xs += [x for x, _ in r['points']]
                ys += [y for _, y in r['points']]
        if len(xs) < 2:
            continue
        a, _ = C.fit_line(xs, ys)
        bs = []
        for m in train:
            a_ = [(c, v) for c, v in sorted(D[m]['libx264'].items())
                  if anchors is None or c in anchors]
            r = C.calibrate_tier(tier, a_, D[m][tier])
            if 'a' in r:
                bs.append(statistics.median([y - a * x for x, y in r['points']]))
        if not bs:
            continue
        b = statistics.median(bs)
        av = C.pava_nonincreasing([v for _, v in D[hold][tier]])
        iso = [(p, v) for (p, _), v in zip(D[hold][tier], av)]
        hw, ne = 0.0, 0
        for crf, want in sorted(D[hold]['libx264'].items()):
            if anchors is not None and crf not in anchors:
                continue
            got = C.vmaf_at_param(iso, a * crf + b)
            if got is None:
                continue
            ne += 1
            hw = max(hw, abs(got - want))
        if ne == 0:
            per[hold] = float('inf')
            bad += 1
            continue
        per[hold] = hw
        worst = max(worst, hw)
    return (float('inf') if bad else worst), per


ALL = ['new1.mp4', 'new4_raw.mp4', 'new5_raw.mp4', 'word_world_2.mp4']
SETS = {
    'S1 全4素材': (ALL, None),
    'S2 剔除 word_world_2': ([m for m in ALL if m != 'word_world_2.mp4'], None),
    'S3 仅1080p两素材': (['new4_raw.mp4', 'new5_raw.mp4'], None),
    'S4 全4/锚点22-30': (ALL, {22.0, 26.0, 30.0}),
}
TIERS = ['libx265', 'libvpx-vp9', 'libaom-av1', 'libsvtav1', 'librav1e@10']
for name, (mats, anc) in SETS.items():
    print(f'\n══ {name}：{mats}' + (f'  锚点{sorted(anc)}' if anc else ''))
    for t in TIERS:
        ms = [m for m in mats if t in D[m]]
        if len(ms) < 2:
            continue
        w, per = loo(t, ms, anc)
        s = '  (折数=%d)' % len(ms)
        if w == float('inf'):
            print(f'  {t:<14} inf ❌ 模型失效  {per}')
        else:
            det = ' '.join(f'{k[:8]}={v:.2f}' for k, v in per.items())
            print(f'  {t:<14} {w:6.3f} {"✅" if w < 1.0 else "❌"}{s}  {det}')
