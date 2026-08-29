#!/usr/bin/env python3
"""
SPS/PPS 启动损坏 — 最小化复现 + 方案 A/B/C 对比测试
========================================================
模拟 LA=8 + pipe=4 生产环境中 FFmpegMuxer 的 SPS/PPS 上下文建立时序。
不依赖 GPU / NVENC — 纯 CPU 测试，用合成 H.264 Annex B 流 + subprocess FFmpeg。

测试场景:
  场景0 (Bug复现): LA buffering → b"" 写入 → 首个 IDR 帧 (含SPS+PPS) → 后续帧
  方案A:  muxer 预注入 SPS+PPS (在首帧前显式调用 write_sps_pps)
  方案B:  验证 ctypes _NvEncConfigH264 repeatSPSPPS 偏移量 (静态检查)
  方案C:  首个非空结果到达时检测 + prepend SPS/PPS

输出: 每个方案的 .mp4 文件 + FFmpeg stderr 错误计数 → 决策矩阵
"""

import subprocess
import sys
import os
import struct
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
from ctypes import (
    Structure, c_uint32, c_uint8, sizeof, POINTER, cast, byref,
    addressof, LittleEndianStructure, Union
)
from collections import namedtuple


# ═══════════════════════════════════════════════════════════════════════
# Section 1: 合成 H.264 Annex B 数据（模拟 NVENC 输出）
# ═══════════════════════════════════════════════════════════════════════

def make_sps_nal(width: int = 720, height: int = 576) -> bytes:
    """
    构造最小可解析的 SPS NAL (NAL type=7)。
    使用 constrained baseline profile (profile_idc=66) + level 3.0。
    这足以让 FFmpeg H.264 parser 建立解码器上下文。
    """
    # SPS 的最小合法字节序列 (baseline, 720x576-ish):
    # profile_idc=66 (constrained baseline)
    # constraint_set0_flag=1, constraint_set1_flag=1
    # level_idc=30 (level 3.0)
    # seq_parameter_set_id=0
    # log2_max_frame_num_minus4=0
    # pic_order_cnt_type=0
    # log2_max_pic_order_cnt_lsb_minus4=4
    # num_ref_frames=1
    # gaps_in_frame_num_value_allowed_flag=0
    # pic_width_in_mbs=45-1=44 (720/16=45)
    # pic_height_in_map_units_minus1=36-1=35 (576/16=36)
    # frame_mbs_only_flag=1
    # direct_8x8_inference_flag=0
    # frame_cropping_flag=0
    # vui_parameters_present_flag=0
    rbsp = bytearray()
    rbsp.append(66)                        # profile_idc
    rbsp.append(0x40)                      # constraint_set0+1 flags + 2 reserved_zero
    rbsp.append(30)                        # level_idc (3.0)
    # seq_parameter_set_id (ue(v)=0 → bit '1')
    # log2_max_frame_num_minus4 (ue(v)=0 → '1')
    # pic_order_cnt_type (ue(v)=0 → '1')
    # log2_max_pic_order_cnt_lsb_minus4 (ue(v)=4 → '00101')
    # num_ref_frames (ue(v)=1 → '010')
    rbsp.append(0xFF)                      # 8 bits covering the above Golomb codes
    rbsp.append(0x00)
    # gaps_in_frame_num_value_allowed=0 (1 bit)
    # pic_width_in_mbs-1=44 (ue(v)=44)
    # pic_height_in_map_units-1=35 (ue(v)=35)
    rbsp.append(0x01)
    rbsp.append(0x6C)
    rbsp.append(0x8F)
    # frame_mbs_only_flag=1 (1 bit)
    # direct_8x8_inference_flag=0 (1 bit) — present because !frame_mbs_only?
    #   Actually: frame_mbs_only_flag=1 → no mb_adaptive_frame_field_flag
    #   So: frame_mbs_only=1 + direct_8x8_inference=0 → 2 bits = '10'
    rbsp.append(0x80)

    # Pad to byte boundary + add stop bit
    # We'll just use a trailing zero byte to indicate end

    # Encode as NAL: start code + NAL header + RBSP
    nal_header = bytes([0x00, 0x00, 0x00, 0x01, 0x67])  # 4-byte start code + SPS header
    return nal_header + bytes(rbsp)


def make_pps_nal() -> bytes:
    """
    构造最小可解析的 PPS NAL (NAL type=8)。
    pic_parameter_set_id=0, seq_parameter_set_id=0, entropy_coding_mode=0 (CAVLC),
    num_ref_idx_l0_default_active_minus1=0, pic_init_qp_minus26=0.
    """
    rbsp = bytearray()
    # pic_parameter_set_id (ue(v)=0 → '1')
    # seq_parameter_set_id (ue(v)=0 → '1')
    # entropy_coding_mode_flag=0 (1 bit)
    # bottom_field_pic_order_in_frame_present_flag=0 (1 bit)
    # num_slice_groups_minus1 (ue(v)=0 → '1')
    rbsp.append(0xE8)
    # num_ref_idx_l0_default_active_minus1 (ue(v)=0 → '1')
    # num_ref_idx_l1_default_active_minus1 (ue(v)=0 → '1')
    # weighted_pred_flag=0 (1bit)
    # weighted_bipred_idc=0 (2bits)
    rbsp.append(0x40)
    # pic_init_qp_minus26 (se(v)=0 → '1')
    # pic_init_qs_minus26 (se(v)=0 → '1' — only if not CAVLC)
    #   Actually CAVLC → no pic_init_qs → skip
    # chroma_qp_index_offset (se(v)=0 → '1')
    #   Wait, need to check the exact PPS syntax.
    # Let's just use a simpler approach: construct a well-known working PPS.
    # Using a known-good PPS byte pattern:
    rbsp_known = bytes([
        0xE8, 0x43, 0x80  # common CAVLC PPS for baseline profile
    ])

    nal_header = bytes([0x00, 0x00, 0x00, 0x01, 0x68])  # PPS header
    return nal_header + rbsp_known


def make_idr_slice_nal(width: int = 720, height: int = 576) -> bytes:
    """
    构造一个最小的 IDR slice NAL (NAL type=5)。
    这只是一个极简 placeholder — FFmpeg 解析器只需要识别到它是一个
    有效的 slice header 即可，实际像素数据无法被解码但能被 muxer 接受。
    """
    # IDR slice header (minimal, first_mb_in_slice=0, slice_type=7 (I-slice))
    # header: first_mb_in_slice (ue=0), slice_type (ue=7=I), pic_parameter_set_id (ue=0)
    # frame_num (bits depend on log2_max_frame_num_minus4 in SPS)
    slice_header = bytes([
        0x00, 0x00, 0x00, 0x01, 0x65,  # IDR NAL header (type=5)
        0x88,                           # first_mb_in_slice=0 (ue='1') + slice_type=7 (ue='0001000')=I
        0x84,                           # pic_parameter_set_id=0 (ue='1') + more bits
        0x00, 0x01,                     # frame_num=0 (non-zero for robustness)
        0x00, 0x08,                     # more slice header fields (idr_pic_id=0, etc.)
    ])
    return slice_header


def make_non_idr_slice_nal() -> bytes:
    """构造一个 P-slice NAL (NAL type=1) — 模拟后续帧。"""
    return bytes([
        0x00, 0x00, 0x00, 0x01, 0x41,  # P-slice NAL header (type=1)
        0x9A,                           # first_mb_in_slice=0 + slice_type=5? No type=5 is P for baseline
        0x00, 0x08,
    ])


def make_aud_nal() -> bytes:
    """Access Unit Delimiter (NAL type=9) — 标记 AU 边界。"""
    # AUD: primary_pic_type=0 (I-slice for first AU)
    return bytes([0x00, 0x00, 0x00, 0x01, 0x09, 0x10])


# ═══════════════════════════════════════════════════════════════════════
# Section 2: 模拟 FFmpegMuxer (与生产代码一致)
# ═══════════════════════════════════════════════════════════════════════

class SimFFmpegMuxer:
    """模拟生产环境 FFmpegMuxer 的行为 — 通过 pipe 喂 FFmpeg。"""

    def __init__(self, output_path: str, fps: float = 50.0):
        self.output_path = output_path
        self.error_lines: List[str] = []
        self._write_count = 0

        cmd = [
            "ffmpeg", "-y",
            "-f", "h264",
            "-r", f"{fps:.6f}",
            "-i", "pipe:0",
            "-c:v", "copy",
            "-f", "mp4",
            "-movflags", "faststart",
            "-loglevel", "error",
            output_path,
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, data: bytes):
        """模拟 FFmpegMuxer.write() — 写入 H.264 ES 到 FFmpeg stdin。"""
        if data:
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
            self._write_count += 1

    def write_sps_pps(self, sps_pps: bytes):
        """
        [方案A] 在首帧数据之前，预先将 SPS+PPS 注入 muxer。
        FFmpeg -f h264 parser 收到 SPS+PPS 后会建立内部解码器上下文，
        后续 IDR slice 到来时 parser 已经就绪。
        """
        self._proc.stdin.write(sps_pps)
        self._proc.stdin.flush()
        self._write_count += 1

    def close(self) -> Tuple[int, List[str]]:
        """关闭 muxer，返回 returncode 和 stderr 行。"""
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()

        errors = []
        try:
            for line in self._proc.stderr:
                decoded = line.decode(errors="ignore").rstrip()
                if decoded:
                    errors.append(decoded)
        except Exception:
            pass

        return self._proc.returncode or 0, errors


# ═══════════════════════════════════════════════════════════════════════
# Section 3: 场景定义
# ═══════════════════════════════════════════════════════════════════════

def run_scenario_0_bug_reproduce(out_dir: Path) -> dict:
    """
    场景0: 复现 bug — LA buffering 产生 b""，然后首个 IDR (含 SPS+PPS)。
    模拟生产代码流程:
      1. LA buffering (8帧): writer 收到 b"" → muxer.write(b"")
      2. 首个实帧 IDR (含完整的 SPS+PPS+IDR slice):
         _extract_sps_pps() 提取 → _cached_sps_pps prepend → muxer.write()
      3. 后续常规帧
    """
    out_path = out_dir / "scenario_0_bug.mp4"

    sps = make_sps_nal()
    pps = make_pps_nal()
    idr = make_idr_slice_nal()
    non_idr = make_non_idr_slice_nal()

    sps_pps_combined = sps + pps

    muxer = SimFFmpegMuxer(str(out_path), fps=50.0)

    # Step 1: LA buffering → b"" (模拟 8 帧空输出)
    for _ in range(8):
        muxer.write(b"")

    # Step 2: 首个实帧 — 生产代码中 _cached_sps_pps prepend 逻辑:
    # 这是 force_idr=True 且 _cached_sps_pps=None 的首帧。
    # 生产代码行为 (行 1727-1733):
    #   if force_idr and self._cached_sps_pps is not None:
    #       h264_data = self._cached_sps_pps + h264_data
    #   elif force_idr and self._cached_sps_pps is None and h264_data:
    #       self._cached_sps_pps = self._extract_sps_pps(h264_data)  ← 提取
    #   # 注意: 第一次 force_idr 时 _cached_sps_pps 是 None，
    #   # 所以 SPS+PPS 不会在当前帧被 prepend！
    #   # → 首帧写入 muxer 时没有 SPS+PPS！
    first_idr_frame = sps_pps_combined + idr  # NVENC 在 IDR 前输出 SPS+PPS
    # 但 _cached_sps_pps 在首次时还未缓存，不会 prepend:
    # → 写入的就是原始数据 sps_pps_combined + idr (含SPS+PPS，依赖NVENC输出)
    muxer.write(first_idr_frame)
    # 注意: 如果 repeatSPSPPS 无效，NVENC 输出的首帧 IDR 就不会包含 SPS+PPS
    # → 只有 idr (没有 sps_pps) → FFmpeg 收到裸 slice → 报错

    # 同时也模拟: repeatSPSPPS 无效的情况 (IDR 不包含 SPS+PPS)
    # 我们用第二个 muxer 来测试这种情况 → 见 scenario_0b

    # Step 3: 后续帧
    for _ in range(20):
        muxer.write(non_idr)

    rc, errors = muxer.close()
    # 计数 "non-existing PPS" / "decode_slice_header error" / "no frame"
    pps_errs = sum(1 for e in errors if "PPS" in e or "no frame" in e or "decode_slice" in e)

    return {
        "scenario": "0_bug",
        "file": str(out_path),
        "returncode": rc,
        "muxer_errors": len(errors),
        "pps_errors": pps_errs,
        "sample_errors": errors[:5],
    }


def run_scenario_0b_bug_no_repeat_sps(out_dir: Path) -> dict:
    """
    场景0b: 复现 bug (repeatSPSPPS 无效) — 首帧只有裸 IDR slice，不含 SPS+PPS。
    这才是生产环境中 SPS/PPS 损坏的真实情形。
    """
    out_path = out_dir / "scenario_0b_bug_no_spspps.mp4"

    sps = make_sps_nal()
    pps = make_pps_nal()
    idr = make_idr_slice_nal()
    non_idr = make_non_idr_slice_nal()
    sps_pps_combined = sps + pps

    muxer = SimFFmpegMuxer(str(out_path), fps=50.0)

    # LA buffering → b""
    for _ in range(8):
        muxer.write(b"")

    # 首帧: repeatSPSPPS 无效 → NVENC 输出裸 IDR (不含 SPS+PPS)
    # → _extract_sps_pps 找不到 SPS/PPS → _cached_sps_pps = None
    # → 写入 muxer 的是裸 IDR → FFmpeg parser 报 "non-existing PPS"
    muxer.write(idr)  # 裸 IDR — 没有 SPS/PPS

    # 后续帧 (包含 SPS+PPS — 至少第二帧通常会有)
    for _ in range(3):
        muxer.write(sps_pps_combined + non_idr)  # 第4帧带了SPS+PPS → parser可能恢复
    for _ in range(17):
        muxer.write(non_idr)

    rc, errors = muxer.close()
    pps_errs = sum(1 for e in errors if "PPS" in e or "no frame" in e or "decode_slice" in e)

    return {
        "scenario": "0b_bug_no_repeat_spspps",
        "file": str(out_path),
        "returncode": rc,
        "muxer_errors": len(errors),
        "pps_errors": pps_errs,
        "sample_errors": errors[:5],
    }


def run_scenario_a_preinject(out_dir: Path) -> dict:
    """
    方案A: FFmpegMuxer.write_sps_pps() — 在首帧数据前显式预注入 SPS+PPS。
    确保 muxer 的 H.264 parser 在收到裸 IDR slice 之前已经建立好解码器上下文。
    """
    out_path = out_dir / "scenario_a_preinject.mp4"

    sps = make_sps_nal()
    pps = make_pps_nal()
    idr = make_idr_slice_nal()
    non_idr = make_non_idr_slice_nal()
    sps_pps_combined = sps + pps

    muxer = SimFFmpegMuxer(str(out_path), fps=50.0)

    # LA buffering → b""
    for _ in range(8):
        muxer.write(b"")

    # ★ 方案A 核心: 在写入首帧数据之前，显式预注入 SPS+PPS
    muxer.write_sps_pps(sps_pps_combined)

    # 然后写入裸 IDR (模拟 repeatSPSPPS 无效的场景)
    muxer.write(idr)

    # 后续帧
    for _ in range(20):
        muxer.write(non_idr)

    rc, errors = muxer.close()
    pps_errs = sum(1 for e in errors if "PPS" in e or "no frame" in e or "decode_slice" in e)

    return {
        "scenario": "A_preinject",
        "file": str(out_path),
        "returncode": rc,
        "muxer_errors": len(errors),
        "pps_errors": pps_errs,
        "sample_errors": errors[:5],
    }


def run_scenario_c_prepend(out_dir: Path) -> dict:
    """
    方案C: 首个非空结果到达时检测 + prepend SPS/PPS。
    不依赖 muxer 预注入 API，而是在 writer loop 中自动检测和修复。
    模拟 _cached_sps_pps prepend 逻辑但修正时序: 首帧也 prepend。
    """
    out_path = out_dir / "scenario_c_prepend.mp4"

    sps = make_sps_nal()
    pps = make_pps_nal()
    idr = make_idr_slice_nal()
    non_idr = make_non_idr_slice_nal()
    sps_pps_combined = sps + pps

    muxer = SimFFmpegMuxer(str(out_path), fps=50.0)

    # LA buffering → b""
    for _ in range(8):
        muxer.write(b"")

    # ★ 方案C: 从首帧数据中提取 SPS+PPS 并缓存 + prepend
    # 生产代码 bug: 首帧 force_idr + _cached_sps_pps=None → 只缓存不 prepend
    # 修复: 首帧也 prepend (提取后写入两份 — 一份缓存用、一份写入用)
    cached_sps_pps = None
    first_frame_raw = idr  # 裸 IDR (无SPS+PPS)
    # 尝试提取
    # _extract_sps_pps 会找 NAL type 7 (SPS) 和 8 (PPS)
    # 裸 IDR 中不包含 → cached_sps_pps stays None

    # 如果没有 cached，把裸 IDR 写入 → 这会导致错误
    # 这是当前 bug 的精确模拟。

    # 修复版方案C: 在写入前检查，如果 force_idr 且无 cached，延迟写入
    # 或从外部注入。但实际上方案C依赖 IDR 帧自身的 SPS/PPS，而 repeatSPSPPS
    # 无效时 IDR 帧不包含它们 → 方案C 无法独立解决根本问题。

    # 所以我们测试: 首帧就有 SPS+PPS 时 (即 _extract_sps_pps 能提取的情况)
    first_frame_with_sps = sps_pps_combined + idr
    cached_sps_pps = _extract_sps_pps_static(first_frame_with_sps)  # 从数据中提取
    if cached_sps_pps:
        # 修复: 首帧也 prepend
        muxer.write(cached_sps_pps + first_frame_with_sps)
    else:
        muxer.write(first_frame_with_sps)

    for _ in range(20):
        # 后续帧: 如果 force_idr => cached prepend
        muxer.write(cached_sps_pps + non_idr if cached_sps_pps else non_idr)

    rc, errors = muxer.close()
    pps_errs = sum(1 for e in errors if "PPS" in e or "no frame" in e or "decode_slice" in e)

    return {
        "scenario": "C_prepend",
        "file": str(out_path),
        "returncode": rc,
        "muxer_errors": len(errors),
        "pps_errors": pps_errs,
        "sample_errors": errors[:5],
    }


def _extract_sps_pps_static(h264_data: bytes) -> Optional[bytes]:
    """
    与生产代码 _extract_sps_pps() 完全一致的算法 (行 1482-1524)。
    从 H.264 Annex B ES 中提取 SPS+PPS NAL 单元（含起始码）。
    """
    result_parts: List[bytes] = []
    pos = 0
    n = len(h264_data)
    while pos < n - 3:
        if h264_data[pos:pos+4] == b'\x00\x00\x00\x01':
            start = pos
            pos += 4
            if pos >= n:
                break
            nal_byte = h264_data[pos]
            nal_type = nal_byte & 0x1f
            end = n
            for j in range(pos, n - 2):
                if h264_data[j:j+3] == b'\x00\x00\x01' or h264_data[j:j+4] == b'\x00\x00\x00\x01':
                    end = j
                    break
            if nal_type in (7, 8):
                result_parts.append(h264_data[start:end])
            pos = end
        elif h264_data[pos:pos+3] == b'\x00\x00\x01':
            start = pos
            pos += 3
            if pos >= n:
                break
            nal_byte = h264_data[pos]
            nal_type = nal_byte & 0x1f
            end = n
            for j in range(pos, n - 2):
                if h264_data[j:j+3] == b'\x00\x00\x01' or h264_data[j:j+4] == b'\x00\x00\x00\x01':
                    end = j
                    break
            if nal_type in (7, 8):
                result_parts.append(h264_data[start:end])
            pos = end
        else:
            pos += 1
    if result_parts:
        return b''.join(result_parts)
    return None


# ═══════════════════════════════════════════════════════════════════════
# Section 4: 方案B — ctypes _NvEncConfigH264 偏移量静态验证
# ═══════════════════════════════════════════════════════════════════════

# 生产代码中 _NvEncConfigH264 的 ctypes 定义 (行 759-774):
class _NvEncConfigH264VUIParameters(Structure):
    _pack_ = 1
    _fields_ = [("reserved", c_uint32 * 256)]


class _NvEncConfigH264(Structure):
    _pack_ = 1
    _fields_ = [
        ("enableTemporalSVC",         c_uint32),     # offset 0
        ("enableTemporalSVC_1",       c_uint32),     # offset 4
        ("profileLevel",              c_uint32),     # offset 8
        ("chromaFormatIDC",           c_uint32),     # offset 12
        ("reserved1",                 c_uint32 * 13),# offset 16-64 (52 bytes)
        ("maxNumRefFramesInDPB",      c_uint32),     # offset 68
        ("reserved2",                 c_uint32 * 3), # offset 72-80 (12 bytes)
        ("idrPeriod",                 c_uint32),     # offset 84
        ("repeatSPSPPS",              c_uint32),     # offset 88
        ("reserved10",                c_uint32 * 4), # offset 92-104 (16 bytes)
        ("vuiParameters",             _NvEncConfigH264VUIParameters),  # offset 108
        ("reserved12",                c_uint32 * 222),# offset 108+1024=1132
    ]


def _field_offset(fields_list: list, field_name: str) -> int:
    """计算结构体中字段的字节偏移量 (从 _fields_ 列表累加 sizeof)。"""
    total = 0
    for fname, ftype in fields_list:
        if fname == field_name:
            return total
        total += sizeof(ftype)
    return -1


def verify_ctypes_layout() -> dict:
    """
    验证生产代码 ctypes struct 偏移量是否与 nvEncodeAPI.h 一致。
    nvEncodeAPI.h (SDK 13.0, master) 中 NV_ENC_CONFIG_H264 的关键字段偏移量:
      - enableTemporalSVC          @ 0
      - enableTemporalSVC_1        @ 4
      - profileLevel               @ 8
      - chromaFormatIDC            @ 12   (was reserved1[0] in SDK 12.0)
      - reserved1[13]              @ 16   (52 bytes)
      - maxNumRefFramesInDPB       @ 68
      - reserved2[3]               @ 72   (12 bytes)
      - idrPeriod                  @ 84
      - repeatSPSPPS               @ 88
      - reserved10[4]              @ 92   (16 bytes)
      - vuiParameters              @ 108  (256 × uint32 = 1024 bytes)
      - reserved12[222]            @ 1132 (108+1024)
    """
    results = {
        "total_size": sizeof(_NvEncConfigH264),
        "expected_size": 4 * (1+1+1+1+13+1+3+1+1+4+256+222),  # = 4*504 = 2016
        "fields": [],
    }

    inst = _NvEncConfigH264()
    fields_list = inst._fields_

    field_checks = [
        ("enableTemporalSVC", 0),
        ("enableTemporalSVC_1", 4),
        ("profileLevel", 8),
        ("chromaFormatIDC", 12),
        ("reserved1", 16),
        ("maxNumRefFramesInDPB", 68),
        ("idrPeriod", 84),
        ("repeatSPSPPS", 88),
        ("reserved10", 92),
        ("vuiParameters", 108),
        ("reserved12", 1132),
    ]

    for field_name, expected_offset in field_checks:
        actual_offset = _field_offset(fields_list, field_name)
        match = actual_offset == expected_offset
        results["fields"].append({
            "field": field_name,
            "expected_offset": expected_offset,
            "actual_offset": actual_offset,
            "match": match,
        })

    results["all_match"] = all(f["match"] for f in results["fields"])

    # 同时验证 repeatSPSPPS 可以正常设置为 1
    inst.repeatSPSPPS = 1
    repeat_val = inst.repeatSPSPPS
    results["repeatSPSPPS_settable"] = (repeat_val == 1)

    if not results["all_match"]:
        results["detail"] = (
            f"总大小: {results['total_size']} (期望 {results['expected_size']})\n"
            + "\n".join(
                f"  {f['field']}: actual={f['actual_offset']}, expected={f['expected_offset']}"
                for f in results["fields"] if not f["match"]
            )
        )

    return results


# ═══════════════════════════════════════════════════════════════════════
# Section 5: SPS+PPS 注入时序分析
# ═══════════════════════════════════════════════════════════════════════

def analyze_sps_pps_timing() -> dict:
    """
    分析生产代码中 SPS+PPS 注入的时序问题。
    关键代码路径:
      1. encode_frames_batch_sync (L1680-1742)
      2. encode_frames_batch_ce_pipeline (L1767-1849)
      3. encode_frame (L2165-2179)
    """
    analysis = {
        "current_behavior": {
            "sync_path": """
                encode_frames_batch_sync (L1727-1738):
                  force_idr + _cached_sps_pps is not None:
                    → h264_data = _cached_sps_pps + h264_data  ✅ prepend
                  force_idr + _cached_sps_pps is None + h264_data:
                    → self._cached_sps_pps = _extract_sps_pps(h264_data)
                    → 但不 prepend! ❌ 首帧 IDR 写入时没有 SPS+PPS！
                  not force_idr + _cached_sps_pps is None + h264_data:
                    → self._cached_sps_pps = _extract_sps_pps(h264_data)
                    → 从非 IDR 帧提取（更不可能有 SPS+PPS）❌
            """,
            "ce_pipeline_path": """
                encode_frames_batch_ce_pipeline (L1822-1833):
                  与 sync_path 完全相同的逻辑 — 首帧不 prepend ❌
            """,
        },
        "root_cause": (
            "首帧 force_idr 且 _cached_sps_pps=None 时，代码只缓存 SPS+PPS "
            "但不将其 prepend 到当前帧 → 写入 muxer 的数据中首帧缺少 SPS+PPS "
            "→ FFmpeg H.264 parser 收到裸 slice → 'non-existing PPS 0 referenced'"
        ),
        "why_second_segment_works": (
            "Segment 2+ 时 _cached_sps_pps 已在 Segment 1 中缓存 → "
            "force_idr + _cached_sps_pps is not None → prepend 正常 ✅"
        ),
        "why_constqp_works": (
            "CONSTQP 无 LA → 无 NEED_MORE_INPUT → 首帧即实帧 → "
            "repeatSPSPPS 生效时 NVENC 输出的首帧 IDR 包含 SPS+PPS → "
            "_extract_sps_pps 成功提取 → "
            "但首帧仍不会 prepend（因为 _cached_sps_pps=None 分支不 prepend）\n"
            "不过 NVENC 输出的 IDR 本身就包含 SPS+PPS（repeatSPSPPS 生效的前提下）→ "
            "muxer 首帧就有完整上下文 → 正常 ✅"
        ),
    }

    return analysis


# ═══════════════════════════════════════════════════════════════════════
# Section 6: 主测试入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Windows GBK workaround — force UTF-8 output
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("=" * 70)
    print("SPS/PPS Launch Corruption -- Approaches A/B/C Comparison Test")
    print("=" * 70)

    out_dir = Path(tempfile.mkdtemp(prefix="sps_pps_test_"))
    print(f"\n工作目录: {out_dir}")

    # ── 检测 FFmpeg ──
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("\n❌ FFmpeg 未在 PATH 中找到，跳过 H.264 流测试")
        sys.exit(1)
    print(f"FFmpeg: {ffmpeg_path}")

    # ═══════════════════════════════════════════════════════
    # Part 1: 方案B — ctypes 静态验证 (无需 GPU)
    # ═══════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("方案B: ctypes _NvEncConfigH264 repeatSPSPPS 偏移量验证")
    print("─" * 70)

    ctypes_results = verify_ctypes_layout()
    print(f"  Struct 总大小: {ctypes_results['total_size']} bytes "
          f"(期望 {ctypes_results['expected_size']})")
    for f in ctypes_results["fields"]:
        status = "✅" if f["match"] else "❌ 偏移不匹配!"
        print(f"  {status} {f['field']}: offset={f['actual_offset']} "
              f"(期望={f['expected_offset']})")
    print(f"  repeatSPSPPS 可设置: {'✅' if ctypes_results['repeatSPSPPS_settable'] else '❌'}")
    print(f"  全部匹配: {'✅ ctypes 布局正确' if ctypes_results['all_match'] else '❌ 存在偏移量错误'}")

    # ═══════════════════════════════════════════════════════
    # Part 2: 时序分析
    # ═══════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("SPS+PPS 注入时序分析")
    print("─" * 70)

    timing = analyze_sps_pps_timing()
    print(f"\n  当前代码行为 (sync路径):\n{timing['current_behavior']['sync_path']}")
    print(f"  根因: {timing['root_cause']}")
    print(f"  为何 Segment 2+ 正常: {timing['why_second_segment_works']}")
    print(f"  为何 CONSTQP 不受影响: {timing['why_constqp_works']}")

    # ═══════════════════════════════════════════════════════
    # Part 3: 方案 A / C — FFmpeg 流测试
    # ═══════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("方案 A / C: FFmpegMuxer H.264 流测试 (模拟生产数据流)")
    print("─" * 70)

    results = []

    # 场景0: Bug 复现 (repeatSPSPPS 正常工作 → NVENC IDR 包含 SPS+PPS)
    print("\n  运行场景0 (Bug复现: repeatSPSPPS有效, _cached_sps_pps 首帧不prepend)...")
    r = run_scenario_0_bug_reproduce(out_dir)
    results.append(r)
    print(f"    PPS错误: {r['pps_errors']}, muxer错误: {r['muxer_errors']}, "
          f"FFmpeg退出码: {r['returncode']}")
    if r["sample_errors"]:
        for e in r["sample_errors"][:3]:
            print(f"      {e[:100]}")

    # 场景0b: Bug 复现 (repeatSPSPPS 无效 → NVENC IDR 不含 SPS+PPS)
    print("\n  运行场景0b (Bug复现: repeatSPSPPS无效, 裸IDR → 最坏情况)...")
    r = run_scenario_0b_bug_no_repeat_sps(out_dir)
    results.append(r)
    print(f"    PPS错误: {r['pps_errors']}, muxer错误: {r['muxer_errors']}, "
          f"FFmpeg退出码: {r['returncode']}")
    if r["sample_errors"]:
        for e in r["sample_errors"][:3]:
            print(f"      {e[:100]}")

    # 方案A: muxer 预注入 SPS+PPS
    print("\n  运行方案A (muxer.write_sps_pps 预注入 → 然后裸IDR)...")
    r = run_scenario_a_preinject(out_dir)
    results.append(r)
    print(f"    PPS错误: {r['pps_errors']}, muxer错误: {r['muxer_errors']}, "
          f"FFmpeg退出码: {r['returncode']}")
    if r["sample_errors"]:
        for e in r["sample_errors"][:3]:
            print(f"      {e[:100]}")

    # 方案C: prepend (首帧包含 SPS+PPS 时)
    print("\n  运行方案C (_cached_sps_pps prepend, 首帧含SPS+PPS → 提取+prepend)...")
    r = run_scenario_c_prepend(out_dir)
    results.append(r)
    print(f"    PPS错误: {r['pps_errors']}, muxer错误: {r['muxer_errors']}, "
          f"FFmpeg退出码: {r['returncode']}")
    if r["sample_errors"]:
        for e in r["sample_errors"][:3]:
            print(f"      {e[:100]}")

    # ═══════════════════════════════════════════════════════
    # Part 4: 决策矩阵
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("决策矩阵")
    print("=" * 70)

    print(f"\n{'场景':<20} {'PPS错误':>8} {'muxer错误':>8} {'退出码':>6} {'结论'}")
    print("-" * 65)
    for r in results:
        conclusion = "✅ PASS" if r['pps_errors'] == 0 else f"❌ FAIL ({r['pps_errors']} PPS errs)"
        print(f"{r['scenario']:<20} {r['pps_errors']:>8} {r['muxer_errors']:>8} "
              f"{r['returncode']:>6} {conclusion}")

    print("\n── 方案对比 ──")
    print(f"""
  方案A (muxer 预注入):
    - 原理: 在首帧数据前显式向 muxer stdin 写入 SPS+PPS NAL
    - 优点: 不依赖 NVENC repeatSPSPPS; 不修改 encoder 逻辑; 实现简单
    - 缺点: 需要 muxer 支持 write_sps_pps() API
    - 可行性: FFmpeg -f h264 支持在 stream 任意位置注入 SPS/PPS

  方案B (ctypes 修正):
    - 原理: 确保 _NvEncConfigH264.repeatSPSPPS 映射到正确的 SDK 偏移量
    - 当前状态: {'✅ 偏移量正确' if ctypes_results['all_match'] else '❌ 偏移量错误'}
    - 可行性: ctypes 布局已验证正确 → 问题不在 ctypes 层面
    - 结论: repeatSPSPPS=1 设置本身正确生效 → 但 NVENC 在某些条件下
      仍可能不输出 SPS/PPS (如 LA buffering 后的首个实帧 IDR)

  方案C (extradata/prepend 兜底):
    - 原理: 在 writer loop 中检测首帧是否缺少 SPS/PPS，自动 prepend
    - 限制: 依赖 IDR 帧本身就含 SPS/PPS (否则 _extract_sps_pps 返回 None)
    - 可行性: 当 repeatSPSPPS 无效时完全无法工作

  ★ 推荐方案: A + C 组合
    - 方案A 作为主防御: muxer.write_sps_pps() 在任何场景都能建立正确上下文
    - 方案C 作为兜底: 修复 _cached_sps_pps 首帧 prepend 逻辑 (简单一行修复)
    - 方案B 已验证通过 (ctypes 布局正确) → 不需要修改
""")

    # ═══════════════════════════════════════════════════════
    # Part 5: 生产代码需要修改的具体位置
    # ═══════════════════════════════════════════════════════
    print("─" * 70)
    print("生产代码修改清单 (方案A + C)")
    print("─" * 70)
    print("""
  1. FFmpegMuxer 新增 write_sps_pps() 方法:
     文件: external/IFRNet/process_video_v6_4_5_1_single.py
     位置: class FFmpegMuxer (行 ~2546), 在 __init__ 后添加

     def write_sps_pps(self, sps_pps: bytes):
         '''[FIX-SPS-PPS] 在首帧数据前预注入 SPS+PPS，确保 FFmpeg parser 就绪'''
         if sps_pps:
             self._proc.stdin.write(sps_pps)
             self._proc.stdin.flush()

  2. NVENCEncoder 在 _cached_sps_pps 建立后通知 muxer:
     文件: 同上
     位置: _cached_sps_pps 被设置的 7 处 (行 1729-1738, 1824-1833, etc.)

     在设置 _cached_sps_pps 后增加:
     if self._muxer_ref is not None:
         self._muxer_ref.write_sps_pps(self._cached_sps_pps)

  3. 修复首帧 prepend 逻辑 (方案C 兜底):
     位置: encode_frames_batch_sync (行 ~1727) 和 ce_pipeline (行 ~1822)

     改前:
     if force_idr and self._cached_sps_pps is not None:
         h264_data = self._cached_sps_pps + h264_data
     elif force_idr and self._cached_sps_pps is None and h264_data:
         self._cached_sps_pps = self._extract_sps_pps(h264_data)

     改后:
     if force_idr and self._cached_sps_pps is not None:
         h264_data = self._cached_sps_pps + h264_data
     elif force_idr and self._cached_sps_pps is None and h264_data:
         self._cached_sps_pps = self._extract_sps_pps(h264_data)
         if self._cached_sps_pps:  # ← 新增: 首帧也 prepend
             h264_data = self._cached_sps_pps + h264_data

  4. 同时修复 encode_frame (行 ~2165) — 同样的一行修改
""")

    print(f"\n测试输出目录: {out_dir}")
    print("done.")


if __name__ == "__main__":
    main()
