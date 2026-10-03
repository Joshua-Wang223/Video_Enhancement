import importlib.util, json, statistics, subprocess
from collections import defaultdict
from pathlib import Path

R = Path('/mnt/d/Workspace_Python/Video_Enhancement')
spec = importlib.util.spec_from_file_location('eqq', R / 'Accessory/probe/calibrate_equal_quality.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

data = defaultdict(lambda: defaultdict(list))
for tag in ('1280x720_10s_n4', '1280x720_10s_n2'):
    p = Path('/tmp/eqq2') / tag / 'points.json'
    for k, v in json.loads(p.read_text()).items():
        m, t, val = k.split('|')
        vm = (v.get('m') or {}).get('vmaf')
        if vm is not None:
            data[m][t].append((float(val), float(vm)))
for m in data:
    for t in data[m]:
        data[m][t].sort()

print('=== 素材特征 + 锚点 VMAF 曲线 ===')
for m in sorted(data):
    src = Path('/mnt/d/Workspace_Python/input_videos') / m
    if not src.is_file():
        cand = list(Path('/mnt/d/Workspace_Python/input_videos').rglob(m))
        src = cand[0] if cand else None
    info = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'stream=width,height,nb_frames,r_frame_rate', '-of', 'csv=p=0', str(src)],
        capture_output=True, text=True).stdout.strip() if src else '?'
    xs = [v for _, v in data[m]['libx264']]
    pts = data[m]['libx264']
    print(f'  {m:<18} {info:<30} n={len(pts)}  ' +
          ' '.join(f'crf{p:.0f}={v:.2f}' for p, v in pts))

print('\n=== 各素材单独拟合 param = a*crf + b ===')
for t in ('libx265', 'libvpx-vp9', 'libaom-av1', 'libsvtav1', 'librav1e@10'):
    print(f'  {t}')
    for m in sorted(data):
        if t not in data[m]:
            continue
        r = C.calibrate_tier(t, data[m]['libx264'], data[m][t])
        if 'a' in r:
            print(f'    {m:<18} a={r["a"]:.4f} b={r["b"]:+9.2f} '
                  f'pts={len(r["points"])}/5 dVMAF={r["max_delta_vmaf"]:.3f} '
                  f'resid={r["max_resid_param"]:.2f} mono={r["monotonic_violation"]:.3f}')
        else:
            print(f'    {m:<18} 拟合失败 {r.get("error")} pts={len(r.get("points", []))}/5')

print('\n=== 各素材目标编码器扫描点 (param -> vmaf) ===')
for m in sorted(data):
    for t in ('libx265', 'libsvtav1'):
        if t not in data[m]:
            continue
        print(f'  {m:<18} {t:<11} ' +
              ' '.join(f'{p:.0f}:{v:.1f}' for p, v in data[m][t]))
