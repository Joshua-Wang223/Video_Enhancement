#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""己方案：采集 5 类素材的 **10s** 版本（现有 6s 版本物理上不够长，无法补 10s 口径）。

背景（2026-10-02 实测）
----------------------
「统一到 10s 口径」需要 10s 版的 5 类内容，但现有 clip 时长为：
  cc_anim_300s 8.08s / cc_subs_105s 6.00s / earth_dark_80s 6.15s
  / natgeo_grass_40s 6.16s / ui_screen_10s 10.00s
⇒ 前 4 条**物理上无法**做 10s 口径（`make_prep(duration=10)` 会截断到素材末尾，
   得到的仍是 6~8s，VMAF 曲线不可比）。

故从素材库/其它源**重新采集** 10s clip。⚠ 新 clip 内容与原 6s clip **不同**
⇒ 它们是**新增素材**而非「同一素材补另一时长」，使素材池从 12 扩到 17 条，
   且每条只存在于**单一时长** ⇒ 合并时天然无同 key 冲突。

5 类映射（内容与原 6s 池尽量对应，但源不同）：
  动画平涂   → Disney Amphibia S01E02（720p 动画剧，2D 平涂风格）
  动画+字幕  → Super Simple Song 'Tobee' 003（自带歌词字幕）
  暗场       → Earth.at.Night 2160p（夜拍纪录片，天然低照度）
  屏幕 UI    → ui_screen_10s.mp4（合成 UI，恰10.00s，直接复用）
  高细节纹理 → NatGeo Our World video-1（960x540 实拍，剪 10s 后上采样 720p）

产出：/tmp/eqq_uni_10s/src/<name>.mp4，均 1280x720 / yuv420p / ≥10.0s / 无音轨
"""
import subprocess
import sys
from pathlib import Path

B = Path('/mnt/f/English Enlightenment')
IV = Path('/mnt/d/Workspace_Python/input_videos')
OUT = Path('/tmp/eqq_uni_10s/src')
OUT.mkdir(parents=True, exist_ok=True)

# (标签, 源路径, 起始秒, 时长秒, 说明)
JOBS = [
    ('anim_10s', B / 'Disney 迪士尼/Amphibia 奇幻沼泽/S1'
     / '奇幻沼泽.Amphibia.S01E02.WEBRip.720p.Chs.Eng-Deefun迪幻字幕组.mp4', 30.0, 10.0,
     '动画平涂（2D 动画剧）'),
    ('anim_subs_10s', B / 'Super Simple Song/01 - Super Simple Songs 歌曲视频/03 - Sing Along With Tobee'
     / '003 Skidamarink - 271064439.mp4', 20.0, 10.0,
     '动画 + 歌词字幕'),
    ('dark_10s', IV / 'Earth.at.Night.in.Color.S02E03.Kangaroo.Valley.2160p_T100_2xfps.mp4',
     30.0, 10.0, '暗场（夜拍纪录片，天然低照度）'),
    ('ui_10s', Path('/mnt/d/Workspace_Python/VidUtils/temp/m2_srcs/ui_screen_10s.mp4'),
     None, None, '屏幕 UI（合成界面，恰 10.00s，直接复用）'),
    ('texture_10s', B / 'National Geographic/Our World 教材（Age 6-12）/OW 1E/10.视频/video-1'
     / 'ow_l1u1_ame_scene1.mp4', 15.0, 10.0,
     '高细节纹理（实拍，960x540 → 上采样 720p）'),
]


def probe(p):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,duration:format=duration',
         '-of', 'default=nw=1', str(p)],
        capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            out[k] = v
    return out


def main():
    for tag, src, ss, dur, desc in JOBS:
        dst = OUT / f'{tag}.mp4'
        if not src.is_file():
            print(f'  [skip] {tag} 源缺失: {src}', file=sys.stderr)
            continue
        if dst.is_file() and float(probe(dst).get('duration', 0)) >= 9.9:
            print(f'  [skip] {tag} 已存在且 ≥10s', file=sys.stderr)
            continue
        info = probe(src)
        sw, sh = info.get('width', '?'), info.get('height', '?')
        sdur = float(info.get('duration', 0) or 0)
        if sdur and dur is not None and (ss + dur) > sdur:
            print(f'  [warn] {tag} 源仅 {sdur:.1f}s，ss={ss}+{dur} 越界，改为 ss=0', file=sys.stderr)
            ss = 0.0
        dst.unlink(missing_ok=True)
        cmd = ['ffmpeg', '-nostdin', '-y', '-hide_banner', '-loglevel', 'error']
        if ss is not None:
            cmd += ['-ss', f'{ss}']
        cmd += ['-i', str(src)]
        if dur is not None:
            cmd += ['-t', f'{dur}']
        cmd += ['-an', '-vf', 'scale=1280:720:flags=lanczos',
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '10',
                '-pix_fmt', 'yuv420p', str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not dst.is_file():
            print(f'  [FAIL] {tag} rc={r.returncode} {r.stderr[:200]}', file=sys.stderr)
            continue
        di = probe(dst)
        print(f'  [ok] {tag:<16}{desc:<26} {sw}x{sh}→1280x720  '
              f'{float(di.get("duration", 0)):.2f}s', file=sys.stderr)
    print('全部完成', file=sys.stderr)


if __name__ == '__main__':
    main()
