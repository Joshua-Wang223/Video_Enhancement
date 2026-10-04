#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NVENC 调优轴（`--nvenc-tune` / `--nvenc-multipass` / 目标码率）的**唯一真源**。

依据 `Plan/ffmpeg_nvenc_knowledge.md` §5/§5.1/§5.2 与 T4 实测（2026-10-04）：

  · `-tune` 的 ffmpeg 默认值就是 `hq`（h264 1~4 / hevc 1~5，default hq）⇒ 写 `hq`
    等于没写，**默认不下发**；`uhq` 是 hevc/av1 专属（`h264_nvenc` 无此档，实测 rc=234）。
  · `-multipass` 默认 `disabled`。固定 `-cq` 下 T4 A/B：fullres ΔVMAF −0.01~−0.11、
    qres −0.07~−0.34（码率 ×0.98~0.997）——**不升 VMAF**。故 CQ 路径**默认不下发**；
    它真正有用的是 CBR / 受限码率（提升码率命中精度），故 `rc_mode=='cbr'` 或给了
    目标码率时自动补 `fullres`（**显式值优先，含 `disabled`**）。constqp 无码率目标
    ⇒ 驱动会忽略，忽略并告知。
  · 现代 `p1~p7` 别名**不带** multipass 标记（只有 legacy `slow` 别名会开两遍）
    ⇒ 别指望 `-preset p7` 自带 two-pass。

本模块被 `external/ifrnet_video/ffmpeg_io.py` 与 `external/realesrgan_video/ffmpeg_io.py`
共同依赖（与 `quality_map` 同样经包入口把 `src/utils` 挂进 sys.path）。**两个 writer 的
命令形状必须逐字一致**（由 `Accessory/verify/crf_cq_unification_verify.py` 的 G5-10 校验）。
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional

NVENC_TUNE_VALUES = ('hq', 'll', 'ull', 'lossless', 'uhq')
NVENC_MULTIPASS_VALUES = ('disabled', 'qres', 'fullres')
#: `uhq` 仅 hevc/av1 NVENC 有该档（h264_nvenc 传 uhq → ffmpeg rc=234）。
NVENC_UHQ_CODECS = {'hevc_nvenc', 'av1_nvenc'}

#: 目标码率的 ffmpeg 记法：`8M` / `8000k` / `12000000`（裸数字按 bps 解释）。
BITRATE_RE = re.compile(r'^\d+(\.\d+)?[kKmM]?$')


def is_nvenc(codec: Optional[str]) -> bool:
    """编码器名是否属于 NVENC 家族。"""
    return 'nvenc' in (codec or '').lower()


def is_bitrate(spec: object) -> bool:
    """目标码率写法是否合法（`8M` / `8000k` / `12000000`）。"""
    return bool(BITRATE_RE.match(str(spec if spec is not None else '').strip()))


def resolve_tune_token(codec: str, tune: Optional[str],
                       inform: Callable[[str], None] = print) -> List[str]:
    """`-tune` 的 token（独立于 `-rc`：constqp / 无损分支也可原样下发）。

    `uhq` 落到非 hevc/av1 → 忽略并告知（返回空表）。
    """
    if tune is None:
        return []
    if tune == 'uhq' and codec not in NVENC_UHQ_CODECS:
        inform(f'--nvenc-tune {tune} 未生效（{codec} 没有 uhq 档，'
               f'仅 hevc / av1 NVENC 有），已忽略')
        return []
    return ['-tune', tune]


def resolve_nvenc_tokens(rc_mode: str, *, codec: str,
                         tune: Optional[str] = None,
                         multipass: Optional[str] = None,
                         bitrate: Optional[str] = None,
                         crf: Optional[float] = None,
                         inform: Callable[[str], None] = print) -> List[str]:
    """VBR/CBR 分支的 quality token（**不含** `-rc:v` 与 `-preset`：由调用方按原位下发）。

    返回顺序固定，便于命令形状判据与「默认命令逐字对齐」::

        [-tune X] [-multipass Y] ('-b:v' <bitrate> | '-cq:v' <crf> '-b:v' '0')

    规则：
      · `tune`：`uhq` 落到非 hevc/av1 → 忽略并告知；否则原样下发。
      · `multipass`：`None` 且（`rc_mode=='cbr'` 或给了 bitrate）→ 自动 `fullres`；
        显式值（含 `disabled`）优先。
      · `bitrate`：给了目标码率 ⇒ 改发 `-b:v <bitrate>` 并**去掉 `-cq:v`**。

    ⚠ 调用方须**已排除** `constqp` / 无损（`crf==0`）分支；那些分支请改用
    :func:`note_suppressed` 告知忽略，不要调用本函数（否则会发出无意义的 multipass）。
    """
    toks: List[str] = []

    toks += resolve_tune_token(codec, tune, inform)

    mp = multipass
    if mp is None and (rc_mode == 'cbr' or bitrate):
        # 自动 multipass：只有 CBR / 受限码率场景才有意义（提升码率命中精度）。
        mp = 'fullres'
    if mp is not None:
        toks += ['-multipass', mp]

    if bitrate:
        toks += ['-b:v', str(bitrate)]
    else:
        toks += ['-cq:v', str(crf), '-b:v', '0']
    return toks


def note_suppressed(*, tune: Optional[str] = None, multipass: Optional[str] = None,
                    reason: str, inform: Callable[[str], None] = print) -> None:
    """NVENC 专属但当前分支不适用（constqp / 无损）⇒ 忽略并告知。"""
    if tune is not None:
        inform(f'--nvenc-tune {tune} 未生效（{reason}），已忽略')
    if multipass is not None:
        inform(f'--nvenc-multipass {multipass} 未生效（{reason}），已忽略')


def note_non_nvenc(codec: str, *, tune: Optional[str] = None,
                   multipass: Optional[str] = None,
                   inform: Callable[[str], None] = print) -> None:
    """`-tune` / `-multipass` 落到软件编码器 ⇒ 忽略并告知（编码器没有这个开关）。"""
    if tune is not None:
        inform(f'--nvenc-tune {tune} 未生效（-tune 是 NVENC 专属选项，'
               f'编码器 {codec} 没有这个开关），已忽略')
    if multipass is not None:
        inform(f'--nvenc-multipass {multipass} 未生效（-multipass 是 NVENC 专属选项，'
               f'编码器 {codec} 没有这个开关），已忽略')
