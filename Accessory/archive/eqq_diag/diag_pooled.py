"""核算 R2 已完成 5 档的**池化** (a, b, lo, hi) + 残差 + 同素材 ΔVMAF（供报告回填）。"""
import importlib.util, json, statistics
from collections import defaultdict
from pathlib import Path

R = Path('/mnt/d/Workspace_Python/Video_Enhancement')
spec = importlib.util.spec_from_file_location('eqq', R / 'Accessory/probe/calibrate_equal_quality.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
CRF = C.CRF

raw = {}
for tag in ('1280x720_10s_n4', '1280x720_10s_n2'):
    p = Path('/tmp/eqq2') / tag / 'points.json'
    if p.is_file():
        for k, v in json.loads(p.read_text()).items():
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

print(f'{"档位":<14}{"a":>9}{"b":>10}{"lo":>5}{"hi":>5}'
      f'{"池化残差":>10}{"同素材maxΔVMAF":>16}  素材数')
for t in ('libx265', 'libvpx-vp9', 'libaom-av1', 'libsvtav1', 'librav1e@10'):
    pts, bs, resid, dvs, nm = [], [], 0.0, [], 0
    for m in sorted(D):
        if t not in D[m]:
            continue
        r = C.calibrate_tier(t, sorted(D[m]['libx264'].items()), D[m][t])
        if 'a' not in r:
            continue
        nm += 1
        pts += list(r['points'])
        dvs.append(r['max_delta_vmaf'])
        resid = max(resid, r['max_resid_param'])
    if len(pts) < 2:
        continue
    a, _ = C.fit_line([x for x, _ in pts], [y for _, y in pts])
    for m in sorted(D):
        if t not in D[m]:
            continue
        r = C.calibrate_tier(t, sorted(D[m]['libx264'].items()), D[m][t])
        if 'a' in r:
            bs.append(statistics.median([y - a * x for x, y in r['points']]))
    b = statistics.median(bs)
    lo, hi = CRF.QUALITY_MAP[C._ffcodec(t)][2], CRF.QUALITY_MAP[C._ffcodec(t)][3]
    print(f'{t:<14}{a:>9.4f}{b:>+10.3f}{lo:>5}{hi:>5}{resid:>10.2f}{max(dvs):>16.2f}  {nm}')

print('\n池化直线在**全体池化点**上的残差（与素材内 resid 不同口径）：')
for t in ('libx265', 'libvpx-vp9', 'libaom-av1', 'libsvtav1', 'librav1e@10'):
    pts, bs = [], []
    for m in sorted(D):
        if t not in D[m]:
            continue
        r = C.calibrate_tier(t, sorted(D[m]['libx264'].items()), D[m][t])
        if 'a' in r:
            pts += list(r['points'])
    if len(pts) < 2:
        continue
    a, _ = C.fit_line([x for x, _ in pts], [y for _, y in pts])
    for m in sorted(D):
        if t not in D[m]:
            continue
        r = C.calibrate_tier(t, sorted(D[m]['libx264'].items()), D[m][t])
        if 'a' in r:
            bs.append(statistics.median([y - a * x for x, y in r['points']]))
    b = statistics.median(bs)
    pr = max(abs(y - (a * x + b)) for x, y in pts)
    print(f'  {t:<14} a={a:.4f} b={b:+.3f}  池化残差={pr:.2f}  b_m范围=[{min(bs):+.2f},{max(bs):+.2f}]')
