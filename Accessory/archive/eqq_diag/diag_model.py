"""诊断 LOO 失败根因：区分「模型形式不足」与「素材本身不一致」。

纯 CPU、零重编码：只用已测 (param, vmaf) 点。
比较 4 种模型形式的 LOO 泛化误差：
  M1 仿射（当前生产格式 (a,b,lo,hi)）
  M2 分段仿射（2 段，split=26）
  M3 二次
  M4 逐素材独立仿射（上界 oracle——素材不一致时的理论最好）
"""
import importlib.util, json, statistics
from collections import defaultdict
from pathlib import Path

R = Path('/mnt/d/Workspace_Python/Video_Enhancement')
spec = importlib.util.spec_from_file_location('eqq', R / 'Accessory/probe/calibrate_equal_quality.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

# ── 载入（去掉跨 workdir 重复：同 (素材,档位,参数) 取首次）──
raw = {}
for tag in ('1280x720_10s_n4', '1280x720_10s_n2'):
    p = Path('/tmp/eqq2') / tag / 'points.json'
    for k, v in json.loads(p.read_text()).items():
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

ANCH = C.ANCHOR_CRFS


def iso_of(mat, tier):
    pts = data[mat][tier]
    v = C.pava_nonincreasing([y for _, y in pts])
    return [(p, y) for (p, _), y in zip(pts, v)]


def anchor_of(mat):
    return list(data[mat]['libx264'].items())


def fit_affine(xs, ys):
    a, b = C.fit_line(xs, ys)
    return lambda x: a * x + b


def fit_quad(xs, ys):
    """最小二乘二次（纯 Python 3x3 正规方程 + 高斯消元；无 numpy 依赖）。"""
    n = len(xs)
    A = [[sum(x ** (i + j) for x in xs) for j in range(3)] for i in range(3)]
    B = [sum(y * x ** i for x, y in zip(xs, ys)) for i in range(3)]
    for i in range(3):                       # 高斯消元（带部分主元）
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
    return lambda x: c[0] * x * x + c[1] * x + c[2]


def fit_piece(xs, ys, split=26):
    segs = C.fit_piecewise(xs, ys, split)
    return lambda x: C._predict_piecewise(x, segs)


def fit_2pt(xs, ys):
    """参数量极限：2 点定一线（oracle 上界之一）。"""
    (x0, y0), (x1, y1) = (xs[0], ys[0]), (xs[-1], ys[-1])
    a = (y1 - y0) / (x1 - x0)
    b = y0 - a * x0
    return lambda x: a * x + b


MODELS = {'M1 仿射(生产)': fit_affine, 'M2 分段(2段)': fit_piece,
          'M3 二次': fit_quad, 'M4 2点定线(oracle)': fit_2pt}

print(f'素材: {sorted(data)}\n')
print(f'{"档位":<14}{"模型":<20}{"LOO worst ΔVMAF":>16}   逐留出')
for tier in ('libx265', 'libvpx-vp9', 'libaom-av1', 'libsvtav1', 'librav1e@10'):
    mats = [m for m in sorted(data) if tier in data[m] and 'libx264' in data[m]]
    if len(mats) < 2:
        continue
    for name, fitter in MODELS.items():
        worst, per = 0.0, []
        for hold in mats:
            train = [m for m in mats if m != hold]
            xs, ys = [], []
            for m in train:
                r = C.calibrate_tier(tier, anchor_of(m), data[m][tier])
                xs += [x for x, _ in r['points']]
                ys += [y for _, y in r['points']]
            if len(xs) < 3:
                continue
            f = fitter(xs, ys)
            iso = iso_of(hold, tier)
            hw = 0.0
            for crf, avmaf in anchor_of(hold):
                got = C.vmaf_at_param(iso, f(crf))
                if got is not None:
                    hw = max(hw, abs(got - avmaf))
            worst = max(worst, hw)
            per.append(f'{hw:.2f}')
        flag = '✅' if worst < 1.0 else '❌'
        print(f'{tier:<14}{name:<20}{worst:>13.3f} {flag}  {" ".join(per)}')
    print()

# ── 素材两两之间的一致性（oracle：每素材用**全体**点拟合，再互相预测）──
print('=== 素材间不一致性（用其余素材拟合 → 预测留出；= LOO 本质）===')
print('每素材的锚点 VMAF 区间跨度（判「素材是否落在同一质量区」）：')
for m in sorted(data):
    a_ = anchor_of(m)
    print(f'  {m:<18} crf18={a_[0][1]:.2f} → crf34={a_[-1][1]:.2f}  '
          f'跨度={a_[0][1]-a_[-1][1]:.2f} VMAF  素材自身 VMAF 中位={statistics.median([y for _,y in a_]):.2f}')
