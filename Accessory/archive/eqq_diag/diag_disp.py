"""结构性反证：等 VMAF 参数解在素材间的离散度随 crf 如何变化。

若离散度随 crf **增大** ⇒ 单一线性式（任何分段）在结构上都无法收敛，
问题不在锚点位置，而在「用一个与素材无关的线性式逼近素材相关的解」。
"""
import importlib.util, json, statistics
from collections import defaultdict
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'eqq', '/mnt/d/Workspace_Python/Video_Enhancement/Accessory/probe/calibrate_equal_quality.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

D = defaultdict(lambda: defaultdict(list))
for k, v in json.loads(Path('/tmp/eqq2/1280x720_10s_n4/points.json').read_text()).items():
    vm = (v.get('m') or {}).get('vmaf')
    if vm is None:
        continue
    m, t, val = k.split('|')
    D[m][t].append((float(val), float(vm)))
for m in D:
    for t in D[m]:
        D[m][t].sort()
D = dict(D)
ANCH = [18.0, 22.0, 26.0, 30.0, 34.0]

print('=== 各素材在每个锚点处的「等 VMAF 参数」（绝对值）===')
print(f'  {"档位":<13}{"素材":<18}' + ''.join(f'{"crf"+str(int(c)):>9}' for c in ANCH))
for tier in ('libx265', 'libvpx-vp9', 'libaom-av1', 'libsvtav1', 'librav1e@10'):
    fits = {}
    for m in sorted(D):
        if tier not in D[m]:
            continue
        r = C.calibrate_tier(tier, D[m]['libx264'], D[m][tier])
        if 'a' in r:
            fits[m] = dict(r['points'])
    for m, d in fits.items():
        print(f'  {tier:<13}{m:<18}' +
              ''.join(f'{d.get(c, float("nan")):>9.1f}' for c in ANCH))
    # 离散度（相对极差）
    disp = []
    for c in ANCH:
        ps = [d[c] for d in fits.values() if c in d]
        if len(ps) >= 2:
            disp.append((c, (max(ps) - min(ps)) / (sum(ps) / len(ps))))
    if disp:
        s = '  '.join(f'crf{int(c)}={x*100:.0f}%' for c, x in disp)
        print(f'  {"":<13}{"→素材间离散度":<18}{s}')
    print()