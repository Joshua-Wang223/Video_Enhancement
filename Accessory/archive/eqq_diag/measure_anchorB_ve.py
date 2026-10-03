"""在 A 侧（VE R2 的 4 素材）补测 B 套锚点 crf21/24/27，使 A 侧也能用 B 套锚点。

背景：LOO 可比对照（同 7 素材/同 6s）显示 B 套 `18/21/24/27/30` **6 个档位全部更优**
（rav1e两档优 4.0~4.5）。A 侧现有锚点是 `18/22/26/30/34`，缺 21/24/27 ⇒ 需补测
这三个锚点，才能用 B 套口径重标 A 侧表值。

口径：A 侧 R2 = 720p / yuv420p / **10s** / n_subsample=1（与 /tmp/eqq2 一致）。
成本：libx264 锚点单点 2~5s ⇒ 4 素材 × 3 点 = 12 点 ≈ 1 分钟。
"""
import importlib.util, json
from pathlib import Path

_h = '/mnt/d/Workspace_Python/Video_Enhancement/Accessory/probe/calibrate_equal_quality.py'
_s = importlib.util.spec_from_file_location('eqq', _h)
C = importlib.util.module_from_spec(_s)
_s.loader.exec_module(C)

IV = Path('/mnt/d/Workspace_Python/input_videos')
SRCS = [IV / 'new5_raw.mp4', IV / 'new4_raw.mp4', IV / 'new1.mp4',
        IV / 'word_world_2.mp4']
OUT = Path('/tmp/eqq2/1280x720_10s_anchorB')
OUT.mkdir(parents=True, exist_ok=True)
PTS = OUT / 'points.json'

WANT = [21.0, 24.0, 27.0]
pts = json.loads(PTS.read_text()) if PTS.is_file() else {}
print(f'workdir={OUT}\n已有 {len(pts)} 点；目标 {len(SRCS)}×{len(WANT)} = {len(SRCS)*len(WANT)} 点')

for src in SRCS:
    if not src.is_file():
        print(f'  跳过缺失 {src}')
        continue
    prep, hdr = C.make_prep(src, OUT, 10.0, 1280, 720)
    nframes = C.ffprobe_video(prep)[0]
    print(f'\n══ {src.name} ══  prep {nframes} 帧 md5={C.md5(prep)[:12]}')
    for crf in WANT:
        key = f'{src.name}|libx264|{crf:g}'
        if key in pts:
            print(f'  [skip] crf{crf:g}')
            continue
        out = OUT / f'x264_{crf:g}.mp4'
        sz, dt = C.encode(prep, 'libx264', crf, out)
        m = C.measure(out, prep, nframes, OUT, with_filters=False, subsample=1)
        pts[key] = {'m': m, 'sec': round(dt, 2)}
        PTS.write_text(json.dumps(pts, ensure_ascii=False, indent=1), encoding='utf-8')
        out.unlink(missing_ok=True)
        print(f'  libx264 crf{crf:<3g} → {sz/1024:7.1f} KiB  vmaf={m["vmaf"]:.3f}  ({dt:.1f}s)',
              flush=True)

print(f'\n落盘 {len(pts)} 点 → {PTS}')