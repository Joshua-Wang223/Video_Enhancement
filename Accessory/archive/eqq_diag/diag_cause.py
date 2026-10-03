"""两件事定性：
1. svtav1 连「素材自身拟合」(oracle, ΔVMAF=4~5.6) 都超门禁 ⇒ 模型形式不足，
   验证：分段/更高阶能否在**训练内**救回。
2. 锚点区 VMAF 饱和度：crf18/22 处各素材 VMAF 挤在 96~100，
   VMAF 分辨率不足 ⇒ 参数反解噪声大。量化：把锚点限制到敏感区(26/30/34)
   时的 LOO。
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


def anchors(m, keep=None):
    return [(c, v) for c, v in sorted(D[m]['libx264'].items()) if keep is None or c in keep]


print('=== 1. 训练内：直线 vs 分段（素材自身）===')
for t in ('libsvtav1', 'libx265', 'librav1e@10'):
    print(f'  {t}')
    for m in sorted(D):
        if t not in D[m]:
            continue
        a_ = anchors(m)
        r = C.calibrate_tier(t, a_, D[m][t])
        pw = r.get('piecewise', {})
        segs = pw.get('segments')
        dvw = pw.get('max_delta_vmaf')
        print(f'    {m:<18} 直线 ΔVMAF={r["max_delta_vmaf"]:>6.3f} resid={r["max_resid_param"]:>5.2f}  '
              f'分段 ΔVMAF={"—" if dvw is None else f"{dvw:.3f}"}  '
              f'needs_piecewise={r.get("needs_piecewise")}'
              + (f'  segs={[(round(s[0],3),round(s[1],2)) for s in segs]}' if segs else ''))

print('\n=== 2. 锚点窗口对 LOO 的影响（全 4 素材）===')
WIN = {'全窗口 18-34': None, '敏感区 26-34': {26.0, 30.0, 34.0},
       '敏感区 22-34': {22.0, 26.0, 30.0, 34.0}}
for t in ('libx265', 'libsvtav1', 'librav1e@10'):
    print(f'  {t}')
    mats = [m for m in sorted(D) if t in D[m]]
    for wname, keep in WIN.items():
        worst, det = 0.0, []
        for hold in mats:
            train = [x for x in mats if x != hold]
            xs, ys = [], []
            for x in train:
                rr = C.calibrate_tier(t, anchors(x, keep), D[x][t])
                if 'a' in rr:
                    xs += [p for p, _ in rr['points']]
                    ys += [q for _, q in rr['points']]
            if len(xs) < 2:
                continue
            a, _ = C.fit_line(xs, ys)
            bs = []
            for x in train:
                rr = C.calibrate_tier(t, anchors(x, keep), D[x][t])
                if 'a' in rr:
                    bs.append(statistics.median([q - a * p for p, q in rr['points']]))
            b = statistics.median(bs)
            av = C.pava_nonincreasing([v for _, v in D[hold][t]])
            iso = [(p, v) for (p, _), v in zip(D[hold][t], av)]
            hw, ne = 0.0, 0
            for crf, want in anchors(hold, keep):
                g = C.vmaf_at_param(iso, a * crf + b)
                if g is None:
                    continue
                ne += 1
                hw = max(hw, abs(g - want))
            if ne == 0:
                det.append(f'{hold[:8]}=inf')
                continue
            det.append(f'{hold[:8]}={hw:.2f}')
            worst = max(worst, hw)
        print(f'    {wname:<16} worst={worst:6.3f} {"✅" if worst < 1.0 else "❌"}  {" ".join(det)}')

print('\n=== 3. crf18 处各素材 VMAF 拥挤度（饱和 ⇒ 反解噪声）===')
for m in sorted(D):
    a_ = anchors(m)
    v18 = dict(a_).get(18.0)
    near = sum(1 for _, v in a_ if v >= 98.0)
    print(f'  {m:<18} crf18 VMAF={v18:.2f}   锚点中 VMAF≥98 的个数={near}/5')
