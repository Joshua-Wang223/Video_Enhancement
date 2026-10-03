"""补测 A 套锚点 crf22/26/34（7 素材 × 6s），使两套锚点可在**同一批素材**上比较。

背景：VU 的m2_7src 只测了 B 套锚点 18/21/24/27/30；A 侧的 18/22/26/30/34 在这批
素材上缺 crf22/26/34 ⇒ 此前任何"A 套 vs B 套"的比较都不可比（A 侧缺 21/24/27，
本方向缺 22/26/34）。补齐这 3 个锚点后，两套锚点共用同一批 7 素材与同一 6s 口径。

成本：libx264 锚点单点均值 3.4s（实测 70 点）⇒ 21 点 ≈ 1.2 分钟。
只测 x264 锚点，不测目标编码器（目标曲线复用各档位已有的 points.json）。
"""
import importlib.util, json, sys
from pathlib import Path

_h = '/mnt/d/Workspace_Python/VidUtils/probe/calibrate_equal_quality.py'
_s = importlib.util.spec_from_file_location('eqq', _h)
C = importlib.util.module_from_spec(_s)
_s.loader.exec_module(C)

IV = Path('/mnt/d/Workspace_Python/input_videos')
M = Path('/mnt/d/Workspace_Python/VidUtils/temp/m2_srcs')
SRCS = [IV / 'new5_raw.mp4', IV / 'new4_raw.mp4', M / 'cc_anim_300s.mkv',
        M / 'cc_subs_105s.mp4', M / 'earth_dark_80s.mp4', M / 'ui_screen_10s.mp4',
        M / 'natgeo_grass_40s.mp4']
OUT = Path('/mnt/d/Workspace_Python/VidUtils/temp/eqq_calib/m2_anchorA')
OUT.mkdir(parents=True, exist_ok=True)
PTS = OUT / 'points.json'

WANT = [22.0, 26.0, 34.0]          # A 套缺的三个（B 套已有 18/21/24/27/30）
pts = json.loads(PTS.read_text()) if PTS.is_file() else {}
DUR, W, H = 6.0, 1280, 720
print(f'workdir={OUT}\n已有 {len(pts)} 点；目标 {len(SRCS)} 素材 × {len(WANT)} 锚点 = '
      f'{len(SRCS)*len(WANT)} 点')

for src in SRCS:
    if not src.is_file():
        print(f'  跳过缺失: {src}')
        continue
    prep, hdr = C.make_prep(src, OUT, DUR, W, H)
    nframes = C.ffprobe_video(prep)[0]
    for crf in WANT:
        key = f'{src.name}|libx264|{crf:g}'
        if key in pts:
            print(f'  [skip] {src.name:<22} crf{crf:g}')
            continue
        out = OUT / f'x264_{crf:g}.mp4'
        sz, dt = C.encode(prep, 'libx264', crf, out)
        m = C.measure(out, prep, nframes, OUT, with_filters=False, subsample=1)
        pts[key] = {'m': m, 'sec': round(dt, 2)}
        PTS.write_text(json.dumps(pts, ensure_ascii=False, indent=1), encoding='utf-8')
        out.unlink(missing_ok=True)
        print(f'  {src.name:<22} crf{crf:<3g} vmaf={m["vmaf"]:.3f}  ({dt:.1f}s)', flush=True)

print(f'\n落盘 {len(pts)} 点 → {PTS}')