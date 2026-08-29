#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# [P3.2] NVENC 公共工具包 —— 供 ifrnet_video 与 realesrgan_video 两子系统共享。
#
# 背景：两份 nvenc_sdk.py 互为镜像但已开始漂移，其中 Annex B NAL 扫描逻辑存在
# 四份变体（_extract_sps_pps / _has_sps_pps / _nal_first_vcl_type /
# FFmpegMuxer._es_has_param_sets）。本包收敛参考实现；两侧通过等价性测试
# （tests/test_regression_min.py::test_nal_scanner_equivalence）锁定行为一致，
# 后续逐步切换为直接引用本模块。

from __future__ import annotations

from typing import List, Optional, Tuple

_START_CODE_3 = b"\x00\x00\x01"
_START_CODE_4 = b"\x00\x00\x00\x01"

# 参数集 NAL 类型集合（按 codec）
PARAM_SET_TYPES = {
    "h264": frozenset({7, 8}),          # SPS / PPS
    "hevc": frozenset({32, 33, 34}),    # VPS / SPS / PPS
}


def iter_nals(es: bytes) -> List[Tuple[int, int, int]]:
    """扫描 Annex B 码流，返回 [(start_code_offset, payload_offset, nal_type), ...]。

    - 兼容 3/4 字节起始码；
    - h264: nal_type = header[0] & 0x1F（1 字节头）；
      hevc:  nal_type = (header[0] >> 1) & 0x3F（2 字节头）；
      av1 : 无 NAL 结构，返回空列表。
    """
    n = len(es)
    out: List[Tuple[int, int, int]] = []
    pos = 0
    while pos < n - 3:
        if es[pos:pos + 4] == _START_CODE_4:
            hdr = pos + 4
        elif es[pos:pos + 3] == _START_CODE_3:
            hdr = pos + 3
        else:
            pos += 1
            continue
        if hdr >= n:
            break
        out.append((pos, hdr, -1))
        pos = hdr
    # 类型解析需要 codec，放到专用函数里做（保持本函数 codec 无关）
    return out


def first_vcl_type(es: bytes, codec: str) -> Optional[int]:
    """返回码流中首个 VCL NAL 的类型；无 VCL 返回 None（即"辅助块/参数集-only"）。

    h264 VCL 类型: 1-5；hevc VCL 类型: 0-31（BLA/IDR/TRAIL 等）。
    """
    if codec not in PARAM_SET_TYPES:
        return None  # av1 或未知：调用方按"含参数集"处理
    vcl_range = range(1, 6) if codec == "h264" else range(0, 32)
    shift_mask = (0x1F, 0x3F << 1)
    n = len(es)
    pos = 0
    while pos < n - 3:
        if es[pos:pos + 4] == _START_CODE_4:
            hdr = pos + 4
        elif es[pos:pos + 3] == _START_CODE_3:
            hdr = pos + 3
        else:
            pos += 1
            continue
        if hdr >= n:
            break
        if codec == "h264":
            t = es[hdr] & 0x1F
        else:
            t = (es[hdr] >> 1) & 0x3F
        if t in vcl_range:
            return t
        pos = hdr
    return None


def has_param_sets(es: bytes, codec: str) -> bool:
    """码流是否已包含参数集 NAL（av1 恒 True：无 NAL，由 -f obu 解析器处理）。"""
    types = PARAM_SET_TYPES.get(codec)
    if types is None:
        return True
    n = len(es)
    pos = 0
    while pos < n - 3:
        if es[pos:pos + 4] == _START_CODE_4:
            hdr = pos + 4
        elif es[pos:pos + 3] == _START_CODE_3:
            hdr = pos + 3
        else:
            pos += 1
            continue
        if hdr >= n:
            break
        t = (es[hdr] & 0x1F) if codec == "h264" else ((es[hdr] >> 1) & 0x3F)
        if t in types:
            return True
        pos = hdr
    return False


def extract_param_sets(es: bytes, codec: str,
                       vps_too: bool = True) -> Optional[bytes]:
    """提取参数集 NAL（含起始码）连续拼接；不存在返回 None。

    hevc 默认连 VPS 一起取（vps_too），与 IFRNet 侧 _extract_sps_pps 行为对齐。
    """
    types = PARAM_SET_TYPES.get(codec)
    if types is None:
        return None
    want = set(types)
    if vps_too and codec == "hevc":
        want.add(32)
    parts: List[bytes] = []
    n = len(es)
    pos = 0
    while pos < n - 3:
        if es[pos:pos + 4] == _START_CODE_4:
            sc_len = 4
        elif es[pos:pos + 3] == _START_CODE_3:
            sc_len = 3
        else:
            pos += 1
            continue
        hdr = pos + sc_len
        if hdr >= n:
            break
        t = (es[hdr] & 0x1F) if codec == "h264" else ((es[hdr] >> 1) & 0x3F)
        if t in want:
            # [P3-FIX-NAL-BOUNDARY] 结束位置必须从 NAL header 后扫描。
            # 直接在 es.find(START_CODE_3) 会命中 4 字节起始码的后缀，拆错连续参数集。
            end = n
            for j in range(hdr, n - 2):
                if es[j:j + 4] == _START_CODE_4 or es[j:j + 3] == _START_CODE_3:
                    end = j
                    break
            parts.append(es[pos:end])
            pos = end
        else:
            pos = hdr
    return b"".join(parts) if parts else None
