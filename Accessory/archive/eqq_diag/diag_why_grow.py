"""误差为何随 crf 单调递增？

假设：目标编码器的 (param→VMAF) 曲线在**高 VMAF 端斜率大**（同样Δparam 引起更大 ΔVMAF），
     于是 x264 crf 越高、目标 VMAF 越低，但**反向插值的条件数越差**。
验证：算每个锚点处
  ① 目标曲线的局部斜率 dVMAF/dparam（越陡 ⇒ param 反解越稳）
  ② 锚点 VMAF 落在曲线什么位置（相对该素材该曲线的 VMAF 全幅）
  ③ 用**该素材自身**拟合时同一锚点的 ΔVMAF（训练内）
若③ 也随 crf 递增 ⇒ 是曲线本身的性质，与素材无关。
"""
import importlib.util, json
from collections import defaultdict
from pathlib import Path

R = Path('/mnt/d/Workspace_Python/Video_Enhancement')
spec = importlib.util.spec_from_file_location('eqq', R / 'Accessory/probe/calibrate_equal_quality.py')
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

tier = 'libsvtav1'
print('=== svtav1：锚点落在目标曲线什么位置 + 局部斜率 + 训练内误差 ===')
for mat in sorted(D):
    if tier not in D[mat]:
        continue
    curve = D[mat][tier]
    vs = [v for _, v in curve]
    lo, hi = min(vs), max(vs)
    print(f'\n  {mat}   曲线 VMAF 范围 {lo:.1f}~{hi:.1f}')
    print(f'    {"锚crf":>6}{"锚VMAF":>8}{"等VMAF参数":>11}{"曲线位置%":>10}'
          f'{"局部斜率":>10}{"训练内Δ":>9}')
    r = C.calibrate_tier(tier, D[mat]['libx264'], curve)
    own = dict(r['points']) if 'a' in r else {}
    for crf, want in D[mat]['libx264']:
        p = own.get(crf)
        if p is None:
            print(f'    {crf:>6.0f}{want:>8.2f}{"—":>11}  （锚点VMAF 超出目标曲线范围）')
            continue
        pos = (hi - want) / (hi - lo) * 100 if hi > lo else 0
        # 局部斜率：在 p 处 dVMAF/dparam
        v_at = C.vmaf_at_param(curve, p)
        g = None
        for (p0, v0), (p1, v1) in zip(curve, curve[1:]):
            if p0 <= p <= p1 and p1 > p0:
                g = abs(v1 - v0) / (p1 - p0)
                break
        # 训练内 ΔVMAF：把 p 当参数回查
        dv = abs(v_at - want) if v_at is not None else None
        print(f'    {crf:>6.0f}{want:>8.2f}{p:>11.2f}{pos:>9.0f}%'
              f'{(f"{g:.3f}" if g else "—"):>10}'
              f'{(f"{dv:.3f}" if dv is not None else "—"):>9}')

print('\n\n=== 关键量：目标曲线在「等 VMAF 参数处」的陡峭度（跨素材/跨档位）===')
print('（陡 ⇒ 参数反解稳；平 ⇒ 一个参数变动引起大 VMAF 变化 ⇒ 表的 b 误差被放大）')
for tier in ('libx265', 'libvpx-vp9', 'libaom-av1', 'libsvtav1', 'librav1e@10'):
    slopes = []
    for mat in sorted(D):
        if tier not in D[mat]:
            continue
        curve = D[mat][tier]
        r = C.calibrate_tier(tier, D[mat]['libx264'], curve)
        if 'a' not in r:
            continue
        for crf, p in r['points']:
            g = None
            for (p0, v0), (p1, v1) in zip(curve, curve[1:]):
                if p0 <= p <= p1 and p1 > p0:
                    g = abs(v1 - v0) / (p1 - p0)
                    break
            if g:
                slopes.append(g)
    if slopes:
        slopes.sort()
        print(f'  {tier:<13} n={len(slopes):<3} 中位 dVMAF/dparam={slopes[len(slopes)//2]:.4f}'
              f'  最小={slopes[0]:.4f}  最大={slopes[-1]:.4f}')
print('\n⇒ 中位斜率越小（曲线越平），同样 Δparam ⇒ ΔVMAF 越大 ⇒ 表的微小 b 误差被放大。')