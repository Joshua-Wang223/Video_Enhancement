"""验证 M3 二次是否真过门禁，还是 fit_quad 的 bug。

做法：把二次拟合的 (a,b,c) 打印出来，并直接对比
  「训练素材拟合出的 f(crf)」 vs 「留出素材的真实等质量参数」
若两者真的逐点相同 ⇒ 是 bug（训练集与留出集数据串了）；
若只是接近 ⇒ 二次确实捕获了某种真实结构，需要另查泛化含义。
"""
import importlib.util, json
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
data = defaultdict(lambda: defaultdict(dict))
for k, vm in raw.items():
    m, t, val = k.split('|')
    data[m][t][float(val)] = vm
for m in data:
    for t in list(data[m]):
        if t != 'libx264':
            data[m][t] = sorted(data[m][t].items())


def fit_quad(xs, ys):
    A = [[sum(x ** (i + j) for x in xs) for j in range(3)] for i in range(3)]
    B = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(3)]
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(A[r][i]))
        A[i], A[p] = A[p], A[i]
        B[i], B[p] = B[p], B[i]
        for r in range(i + 1, 3):
            f = A[r][i] / A[i][i]
            for c in range(i, 3):
                A[r][c] -= f * A[i][c]
            B[r] -= f * B[i]
    c = [0.0, 0.0, 0.0]
    for i in (2, 1, 0):
        c[i] = (B[i] - sum(A[i][j] * c[j] for j in range(i + 1, 3))) / A[i][i]
    return c


tier = 'libx265'
mats = sorted(data)
for hold in mats:
    train = [m for m in mats if m != hold]
    xs, ys = [], []
    for m in train:
        r = C.calibrate_tier(tier, sorted(data[m]['libx264'].items()), data[m][tier])
        xs += [x for x, _ in r['points']]
        ys += [y for _, y in r['points']]
    c = fit_quad(xs, ys)
    f = lambda x: c[0] * x * x + c[1] * x + c[2]
    r_hold = C.calibrate_tier(tier, sorted(data[hold]['libx264'].items()), data[hold][tier])
    print(f'--- 留出 {hold} ---')
    print(f'  二次系数 a2={c[0]:+.6f} a1={c[1]:+.4f} a0={c[2]:+.3f}')
    print(f'  {"crf":>5}{"训练拟合":>10}{"留出真实":>10}{"差":>8}')
    for (crf, av), (tcrf, truep) in zip(sorted(data[hold]['libx264'].items()),
                                       r_hold['points']):
        assert abs(tcrf - crf) < 1e-6, (tcrf, crf)
        print(f'  {crf:>5.0f}{f(crf):>10.2f}{truep:>10.2f}{f(crf)-truep:>8.2f}')
    # 训练集自身残差（应有非零，说明拟合没把训练点也吃成 0）
    tr_res = max(abs(f(x) - y) for x, y in zip(xs, ys))
    print(f'  训练集最大残差 = {tr_res:.3f}（若≈0 说明 xs 有重复导致过约束）')
