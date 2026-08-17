#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# IFRNet Video Enhancement - NVENC SDK Level 1 编码模块（ctypes CUDA/NVENC SDK 13.0）。
# 组件：NVENCEncoder / _NVENCEncodeThread / _rgb_to_nv12_gpu(_batch) / FFmpegMuxer。
# 镜像 external/realesrgan_video/nvenc_sdk.py 的职责。

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
from collections import deque
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# [FIX-T3-V643] NVENC GPU 直通编码器
# ─────────────────────────────────────────────────────────────────────────────

import ctypes
from ctypes import (c_uint8, c_uint16, c_uint32, c_int32, c_int, c_uint64, c_void_p,
                     c_char, c_size_t, c_double, Structure, POINTER, byref,
                     sizeof, cast, pointer, c_bool)
import os
import threading

# ==============================================================================
# NVENC GUID 结构体 (16 bytes)
# ==============================================================================

class _NvGuid(Structure):
    _pack_ = 1
    _fields_ = [
        ("Data1", c_uint32),
        ("Data2", c_uint16),
        ("Data3", c_uint16),
        ("Data4", c_uint8 * 8),
    ]

    def __eq__(self, other):
        if not isinstance(other, _NvGuid):
            return False
        return (self.Data1 == other.Data1 and self.Data2 == other.Data2
                and self.Data3 == other.Data3 and bytes(self.Data4) == bytes(other.Data4))

    def __hash__(self):
        return hash((self.Data1, self.Data2, self.Data3, bytes(self.Data4)))

# ==============================================================================
# NVENC 预定义 GUID 常量
# ==============================================================================

NV_ENC_CODEC_H264_GUID = _NvGuid(0x6b9c211b, 0x3fdd, 0x4a5a,
    (0x8d, 0x2e, 0x05, 0x0a, 0xbb, 0xb9, 0x1c, 0x6a))

NV_ENC_PRESET_P1_GUID = _NvGuid(0xfc9a8d6c, 0xa4e8, 0x4f03,
    (0xaa, 0xce, 0x91, 0x97, 0x6a, 0xc2, 0x74, 0x10))
NV_ENC_PRESET_P2_GUID = _NvGuid(0x711784c6, 0x34a6, 0x47e0,
    (0xaa, 0x06, 0x60, 0x9c, 0x72, 0xdb, 0x0c, 0x8a))
NV_ENC_PRESET_P3_GUID = _NvGuid(0xc678451b, 0x1b4f, 0x4b64,
    (0xbc, 0x03, 0x10, 0xbb, 0x22, 0xc3, 0x54, 0x63))
NV_ENC_PRESET_P4_GUID = _NvGuid(0xec59fb72, 0x14fb, 0x4e28,
    (0xbc, 0x04, 0xa1, 0x59, 0xd2, 0x6c, 0x01, 0xc5))
NV_ENC_PRESET_P5_GUID = _NvGuid(0xb0b9e4da, 0xb52e, 0x4edb,
    (0xa1, 0xcb, 0xd4, 0x70, 0x36, 0x48, 0x32, 0x95))
NV_ENC_PRESET_P6_GUID = _NvGuid(0x74c1e37a, 0x4b74, 0x4905,
    (0xb0, 0xe9, 0x00, 0x61, 0x15, 0x53, 0xb6, 0x3a))
NV_ENC_PRESET_P7_GUID = _NvGuid(0x5b7e7d04, 0xb7df, 0x4488,
    (0x82, 0x4e, 0x55, 0xbf, 0x41, 0x8a, 0x18, 0x29))

NV_ENC_H264_PROFILE_HIGH_GUID = _NvGuid(0x1a1a5a20, 0xf787, 0x4e5b,
    (0x9a, 0xab, 0x58, 0x76, 0xfa, 0x7a, 0xdc, 0x0f))

_PRESET_GUID_MAP = {
    "p1": NV_ENC_PRESET_P1_GUID, "p2": NV_ENC_PRESET_P2_GUID,
    "p3": NV_ENC_PRESET_P3_GUID, "p4": NV_ENC_PRESET_P4_GUID,
    "p5": NV_ENC_PRESET_P5_GUID, "p6": NV_ENC_PRESET_P6_GUID,
    "p7": NV_ENC_PRESET_P7_GUID,
}

# ==============================================================================
# NVENC API 常量
# ==============================================================================

# [FIX-NVENC-SDK13] SDK 13.0 struct version formula (verified against nv-codec-headers SDK 13.0.0):
#   NVENCAPI_VERSION = NVENCAPI_MAJOR | (NVENCAPI_MINOR << 24) = 13 | (0 << 24) = 0x0d
#   NVENCAPI_STRUCT_VERSION(ver) = NVENCAPI_VERSION | (ver << 16) | (0x7 << 28) | (0x80000000 if bit31 else 0)
#   低字节是 NVENCAPI_VERSION (0x0d)，不是 sizeof(struct)。这是 SDK 12+ 的正确公式。
_NVENCAPI_VERSION_FALLBACK = (13 << 4) | 0  # 13.0 = 0xD0

def NVENCAPI_STRUCT_VERSION(struct_or_size, api_ver=None):
    """保留兼容旧代码。"""
    if api_ver is None:
        api_ver = _NVENCAPI_VERSION_FALLBACK
    if isinstance(struct_or_size, int):
        size = struct_or_size
    else:
        size = sizeof(struct_or_size)
    return size | (api_ver << 16) | (0x7 << 28)

# SDK 13.0 struct version helper: NVENCAPI_VERSION | (ver << 16) | (0x7 << 28) | (bit31 ? 1<<31 : 0)
# NVENCAPI_VERSION = NVENCAPI_MAJOR | (NVENCAPI_MINOR << 24) = 13 | (0 << 24) = 0x0d
# This is the CORRECT formula per nv-codec-headers SDK 13.0, NOT sizeof-based.
_NVENCAPI_VERSION = 0x0d  # SDK 13.0

def _sdk13_ver(ver, bit31=False):
    v = _NVENCAPI_VERSION | (ver << 16) | (0x7 << 28)
    if bit31:
        v |= (1 << 31)
    return v
NV_ENC_CREATE_INPUT_BUFFER_VER = _sdk13_ver(2)           # NVENCAPI_STRUCT_VERSION(2) = 0x7002000d
NV_ENC_LOCK_INPUT_BUFFER_VER = _sdk13_ver(1)             # NVENCAPI_STRUCT_VERSION(1) = 0x7001000d
NV_ENC_CREATE_BITSTREAM_BUFFER_VER = _sdk13_ver(1)       # NVENCAPI_STRUCT_VERSION(1) = 0x7001000d

# func_table = 64 指针槽, sizeof(c_void_p) * 64 = 512 (64-bit)
_FUNC_TABLE_SIZE = sizeof(c_void_p) * 64

# 静态结构体版本常量 — 依赖 struct 定义，在 struct 定义完成后赋值 (#NVENC_STRUCT_VERS)
NV_ENC_PRESET_CONFIG_VER     = 0  # placeholder
NV_ENC_CONFIG_VER            = 0
NV_ENC_INITIALIZE_PARAMS_VER = 0
NV_ENC_PIC_PARAMS_VER        = 0
NV_ENC_LOCK_BITSTREAM_VER    = 0
NV_ENC_REGISTER_RESOURCE_VER = 0
NV_ENC_MAP_INPUT_RESOURCE_VER = 0

NV_ENC_SUCCESS = 0
NV_ENC_ERR_NEED_MORE_INPUT = 17  # lookahead: encoder needs more frames before producing output

# NV_ENC_DEVICE_TYPE enum per nv-codec-headers SDK 13.0:
#   DIRECTX=0, CUDA=1, OPENGL=2
NV_ENC_DEVICE_TYPE_CUDA = 1
NV_ENC_BUFFER_FORMAT_NV12 = 1
NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR = 2
NV_ENC_INPUT_IMAGE = 0
NV_ENC_PIC_STRUCT_FRAME = 1

# ==============================================================================
# NVENC API 结构体定义
# ==============================================================================

class _NvEncOpenEncodeSessionExParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",    c_uint32),
        ("deviceType", c_uint32),
        ("device",     c_void_p),
        ("reserved",   c_void_p),
        ("apiVersion", c_uint32),
        ("reserved1",  c_uint8 * 253 * 4),
        ("reserved2",  c_void_p * 64),
    ]

class _NvEncInitializeParams(Structure):
    """SDK 13.0 NV_ENC_INITIALIZE_PARAMS — sizeof=1800 bytes (C dump verified)"""
    _pack_ = 1
    _fields_ = [
        ("version",                 c_uint32),        # offset 0
        ("encodeGUID",              _NvGuid),         # offset 4 (16 bytes)
        ("presetGUID",              _NvGuid),         # offset 20 (16 bytes)
        ("encodeWidth",             c_uint32),        # offset 36
        ("encodeHeight",            c_uint32),        # offset 40
        ("darWidth",                c_uint32),        # offset 44
        ("darHeight",               c_uint32),        # offset 48
        ("frameRateNum",            c_uint32),        # offset 52
        ("frameRateDen",            c_uint32),        # offset 56
        ("enableEncodeAsync",       c_uint32),        # offset 60
        ("enablePTD",               c_uint32),        # offset 64
        ("bitfield",                c_uint32),        # offset 68
        ("privDataSize",            c_uint32),        # offset 72
        ("reserved_76",             c_uint32),        # offset 76
        ("privData",                c_void_p),        # offset 80
        ("encodeConfig",            c_void_p),        # offset 88 ★
        ("maxEncodeWidth",          c_uint32),        # offset 96
        ("maxEncodeHeight",         c_uint32),        # offset 100
        ("maxMEHintCountsPerBlock", c_uint8 * 32),    # offset 104
        ("tuningInfo",              c_uint32),        # offset 136
        ("bufferFormat",            c_uint32),        # offset 140
        ("numStateBuffers",         c_uint32),        # offset 144
        ("outputStatsLevel",        c_uint32),        # offset 148
        ("reserved1",               c_uint8 * 1136),  # offset 152 (284×4)
        ("reserved2",               c_void_p * 64),   # offset 1288 (64×8=512)
    ]

class _NvEncConfigH264VUIParameters(Structure):
    _pack_ = 1
    _fields_ = [
        ("overscanInfoPresentFlag", c_uint32),
        ("videoSignalTypePresentFlag", c_uint32),
        ("videoFormat", c_uint32),
        ("videoFullRangeFlag", c_uint32),
        ("colourDescriptionPresentFlag", c_uint32),
        ("colourPrimaries", c_uint32),
        ("transferCharacteristics", c_uint32),
        ("matrixCoefficients", c_uint32),
        ("chromaSampleLocationFlag", c_uint32),
        ("chromaSampleLocationTop", c_uint32),
        ("chromaSampleLocationBottom", c_uint32),
        ("bitstreamRestrictionFlag", c_uint32),
        ("reserved", c_uint32 * 16),
    ]

class _NvEncConfigH264(Structure):
    _pack_ = 1
    _fields_ = [
        ("enableTemporalSVC",         c_uint32),
        ("enableTemporalSVC_1",       c_uint32),
        ("profileLevel",              c_uint32),
        ("chromaFormatIDC",           c_uint32),   # [FIX-CHROMA] 1=4:2:0, 0=monochrome — was buried in reserved1
        ("reserved1",                 c_uint32 * 13),
        ("maxNumRefFramesInDPB",      c_uint32),
        ("reserved2",                 c_uint32 * 3),
        ("idrPeriod",                 c_uint32),
        ("repeatSPSPPS",              c_uint32),
        ("reserved10",                c_uint32 * 4),
        ("vuiParameters",             _NvEncConfigH264VUIParameters),
        ("reserved12",                c_uint32 * 222),
    ]

class _NvEncConfigHevc(Structure):
    _pack_ = 1
    _fields_ = [("reserved", c_uint32 * 288)]

class _NvEncConfigH264MeOnly(Structure):
    _pack_ = 1
    _fields_ = [("reserved", c_uint32 * 248)]

class _NvEncConfigHevcMeOnly(Structure):
    _pack_ = 1
    _fields_ = [("reserved", c_uint32 * 248)]

class _NvEncConfig(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",         c_uint32),
        ("profileGUID",     _NvGuid),
        ("gopLength",       c_uint32),
        ("frameIntervalP",  c_uint32),
        ("frameFieldMode",  c_uint32),
        ("enablePTD",       c_uint32),
        ("frameFieldMode_1",c_uint32),
        ("reserved3",       c_uint32 * 53),
        ("mvPrecision",     c_uint32),
        ("reserved4",       c_uint32 * 27),
        ("reserved5",       c_uint32 * 172),  # [V6451] absorb mis-mapped enableTemporalAQ (=rcParams bitfield bit7, not a standalone NV_ENC_CONFIG field)
        ("encodeCodecConfig", _NvEncConfigH264),
        ("reserved7",      c_uint32 * 252),
    ]

class _NvEncPresetConfig(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",       c_uint32),
        ("presetConfig",  _NvEncConfig),
        ("reserved",      c_uint32 * 256),
    ]

class _NvEncRegisterResource(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",          c_uint32),
        ("resourceType",     c_uint32),
        ("width",            c_uint32),
        ("height",           c_uint32),
        ("pitch",            c_uint32),
        ("subResourceIndex", c_uint32),
        ("bufferFormat",     c_uint32),
        ("bufferUsage",      c_uint32),
        ("pInputFencePoint", c_void_p),
        ("pOutputFencePoint",c_void_p),
        ("reserved",         c_uint32 * 248),
        ("registeredResource", c_void_p),
    ]

class _NvEncMapInputResource(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",            c_uint32),
        ("subResourceIndex",   c_uint32),
        ("reserved",           c_uint32 * 62),
        ("registeredResource", c_void_p),
        ("mappedResource",     c_void_p),
        ("reserved1",          c_uint32 * 62),
    ]

class _NvEncPicParamsH264(Structure):
    _pack_ = 1
    _fields_ = [
        ("reserved",        c_uint32 * 4),
        ("refFrameFlag",    c_uint32),
        ("reserved1",       c_uint32 * 257),
    ]

class _NvEncPicParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",           c_uint32),
        ("inputWidth",        c_uint32),
        ("inputHeight",       c_uint32),
        ("inputPitch",        c_uint32),
        ("inputBuffer",       c_void_p),
        ("inputTimeStamp",    c_uint64),
        ("pictureStruct",     c_uint32),
        ("encodePicFlags",    c_uint32),
        ("frameIdx",          c_uint32),
        ("inputFencePoint",   c_void_p),
        ("outputFencePoint",  c_void_p),
        ("inputDuration",     c_uint64),
        ("reserved",          c_uint32 * 8),
        ("codecPicParams",    _NvEncPicParamsH264),
        ("reserved1",         c_uint32 * 272),
    ]

class _NvEncLockBitstream(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",           c_uint32),
        ("doNotWait",         c_uint32),
        ("reserved",          c_uint32 * 30),
        ("outputBitstream",   c_void_p),
        ("sliceOffsets",      c_uint32 * 16),
        ("reserved1",         c_uint32 * 246),
        ("bitstreamSizeInBytes",c_uint32),
        ("bitstreamBufferPtr", c_void_p),
        ("reserved2",         c_uint32 * 174),
    ]

# ── #NVENC_STRUCT_VERS: SDK 13.0 version constants (NVENCAPI_STRUCT_VERSION based) ──
NV_ENC_PRESET_CONFIG_VER     = _sdk13_ver(5, True)   # 0xf005000d
NV_ENC_CONFIG_VER            = _sdk13_ver(9, True)   # 0xf009000d
NV_ENC_INITIALIZE_PARAMS_VER = _sdk13_ver(7, True)   # 0xf007000d
NV_ENC_PIC_PARAMS_VER        = _sdk13_ver(7, True)   # 0xf007000d
NV_ENC_LOCK_BITSTREAM_VER    = _sdk13_ver(2, True)   # 0xf002000d
NV_ENC_REGISTER_RESOURCE_VER = _sdk13_ver(5)          # 0x7005000d
NV_ENC_MAP_INPUT_RESOURCE_VER = _sdk13_ver(4)          # 0x7004000d

# ==============================================================================
# ctypes 函数原型定义
# ==============================================================================

_NvEncodeAPICreateInstanceProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p,
)

_NvEncOpenEncodeSessionExProto = ctypes.CFUNCTYPE(
    c_uint32, ctypes.POINTER(_NvEncOpenEncodeSessionExParams), ctypes.POINTER(c_void_p),
)
_NvEncGetEncodePresetConfigProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, _NvGuid, _NvGuid, ctypes.POINTER(_NvEncPresetConfig),
)
# [FIX-NVENC-VER-D] nvEncInitializeEncoder(void* encoder, NV_ENC_INITIALIZE_PARAMS*)
# 原参数顺序错误（params在前），修正为 encoder handle 在第一位
_NvEncCreateEncoderProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, ctypes.POINTER(_NvEncInitializeParams),
)
_NvEncDestroyEncoderProto = ctypes.CFUNCTYPE(c_uint32, c_void_p)
_NvEncRegisterResourceProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, ctypes.POINTER(_NvEncRegisterResource),
)
_NvEncUnregisterResourceProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, c_void_p,
)
_NvEncMapInputResourceProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, ctypes.POINTER(_NvEncMapInputResource),
)
_NvEncUnmapInputResourceProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, c_void_p,
)
_NvEncEncodePictureProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, ctypes.POINTER(_NvEncPicParams),
)
_NvEncLockBitstreamProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, ctypes.POINTER(_NvEncLockBitstream),
)
_NvEncUnlockBitstreamProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, c_void_p,
)
# Raw LockBitstream proto using byte buffer (no typed struct needed).
# Exported for external test scripts that construct NV_ENC_LOCK_BITSTREAM manually.
_LockBitstreamProto_raw = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, ctypes.POINTER(ctypes.c_uint8 * 1544),
)

# [FIX-NVENC-SDK13] SDK 13.0 function table indices — verified via nv-codec-headers n13.0.19.0 C dump.
# Index = (offsetof(field) - offsetof(nvEncOpenEncodeSession)) / sizeof(void*)
# SDK 13.0 将 OpenEncodeSessionEx 从 index 24 → 29 (新增 5 个 reserved 槽)
_FUNC_IDX = {
    "GetEncodeGUIDCount":        1,   # nvEncGetEncodeGUIDCount
    "GetEncodeGUIDs":            4,   # nvEncGetEncodeGUIDs
    "GetEncodePresetGUIDs":      9,   # nvEncGetEncodePresetGUIDs
    "GetEncodePresetConfig":    10,   # nvEncGetEncodePresetConfig
    "InitializeEncoder":        11,   # nvEncInitializeEncoder (旧名 nvEncCreateEncoder)
    "CreateInputBuffer":        12,   # nvEncCreateInputBuffer
    "DestroyInputBuffer":       13,   # nvEncDestroyInputBuffer
    "CreateBitstreamBuffer":    14,   # nvEncCreateBitstreamBuffer
    "DestroyBitstreamBuffer":   15,   # nvEncDestroyBitstreamBuffer
    "EncodePicture":            16,   # nvEncEncodePicture
    "LockBitstream":            17,   # nvEncLockBitstream
    "UnlockBitstream":          18,   # nvEncUnlockBitstream
    "LockInputBuffer":          19,   # nvEncLockInputBuffer
    "UnlockInputBuffer":        20,   # nvEncUnlockInputBuffer
    "MapInputResource":         25,   # nvEncMapInputResource
    "UnmapInputResource":       26,   # nvEncUnmapInputResource
    "DestroyEncoder":           27,   # nvEncDestroyEncoder
    "OpenEncodeSessionEx":      29,   # nvEncOpenEncodeSessionEx (SDK 13.0)
    "RegisterResource":         30,   # nvEncRegisterResource
    "UnregisterResource":       31,   # nvEncUnregisterResource
    "GetEncodePresetConfigEx":  39,   # nvEncGetEncodePresetConfigEx (SDK 13.0)
}

# ── _NVENC_VBR_QUALITY_OFFSET ──
# [已废弃] 旧版 CRF→targetQuality 偏移公式的偏移值。
# 注意（术语澄清）：targetQuality / qvbrQuality 是 RC 质量参数（CRF 语义，
# 范围 1-51），【不是】H.264 Level。真正的 Level 是 "H.264 level requested: 51"
# 与 SPS level_idc=51，二者不可混为一谈。
# 当前实际行为（与代码 line 1367/1380 一致）：targetQuality = max(1, CRF) 直接映射
# （CRF=18 → tq=18；CRF=23 → tq=23），并非 51-CRF 反置。
#   · 值越小质量越高（1=best），越大质量越低（51=worst）
# 不再使用此常量，保留作为历史参考。
# 历史值：7（CRF-offset 公式，文件仍偏大）、21（加法公式，文件极大 36.9 MB）、15（CRF-offset 公式）
_NVENC_VBR_QUALITY_OFFSET: int = 15

# ── _PRESET_P_INDEX ──
# x264 preset name → NVENC preset array index (p-index)
# NVENC driver returns presets in order: p1 (fastest) → p7 (slowest)
# p1=0, p2=1, ..., p7=6
# Also supports direct p1-p7 strings as preset names
_PRESET_P_INDEX: dict = {
    "ultrafast": 0, "superfast": 0,
    "veryfast": 1, "faster": 2,
    "fast": 3, "medium": 4,
    "slow": 5, "slower": 6, "veryslow": 6, "placebo": 6,
}

# [FIX-BR-CLAMP] avgBitrate 估算钳制上限（bps）。驱动对单会话码率有上限，
# 高分辨率×高帧率（如 720p48 算出 132.6Mbps）超限会触发 InitializeEncoder
# code=8（关注点4）。目标仅作为 NVENC 的速度天花板（防止无约束质量搜索
# 导致 GPU 空闲/FPS 塌方），钳制不影响实际编码质量语义。
_NVENC_BR_CLAMP_MAX: int = 50_000_000   # 50 Mbps
_NVENC_BR_CLAMP_MIN: int = 5_000_000    # 5 Mbps（保底，同旧 max() 下限）


def _clamp_bitrate(raw_bps: int) -> int:
    """钳制 NVENC avgBitrate 估算值到 [min, max] 区间。"""
    if raw_bps <= 0:
        return _NVENC_BR_CLAMP_MIN
    return max(_NVENC_BR_CLAMP_MIN, min(_NVENC_BR_CLAMP_MAX, raw_bps))


# ==============================================================================
# NVENCEncoder
# ==============================================================================

class _RotationBitReader:
    """H.264 RBSP 位读取器（用于 [FIX-LA-WINDOW-ROTATION] 的 slice/SPS 解析）。
    与 tests/verify_segment_bitstream_v2.py 的 _BitReader 等价：字节序大端，
    exp-golomb ue(v)/se(v) 支持。payload 必须是已去 emulation prevention 的 RBSP。"""

    __slots__ = ('_data', '_pos')

    def __init__(self, payload: bytes):
        self._data = payload
        self._pos = 0

    def _bit(self) -> int:
        byte_idx = self._pos >> 3
        if byte_idx >= len(self._data):
            raise ValueError('RBSP 位越界')
        v = (self._data[byte_idx] >> (7 - (self._pos & 7))) & 1
        self._pos += 1
        return v

    def bits(self, n: int) -> int:
        v = 0
        for _ in range(n):
            v = (v << 1) | self._bit()
        return v

    def ue(self) -> int:
        leading = 0
        while self._bit() == 0:
            leading += 1
            if leading > 31:
                raise ValueError('ue(v) 溢出')
        return (1 << leading) - 1 + (self.bits(leading) if leading else 0)

    def se(self) -> int:
        code_num = self.ue()
        return (code_num + 1) // 2 if code_num & 1 else -(code_num // 2)



class NVENCEncoder:
    """GPU direct H.264 hardware encoder via NVENC SDK (ctypes) — SDK 13.0 verified.

    Input: GPU tensor (NV12 format, uint8, H_total x W, contiguous)
    Output: H.264 Elementary Stream bytes

    [FIX-NVENC-SDK13] Complete rewrite based on verified test_nvenc_pre_torch.py:
      - CreateInputBuffer + LockInputBuffer + cuMemcpyDtoD_v2 (no RegisterResource)
      - CreateBitstreamBuffer for encoder output
      - All structs via byte array + manual offset writes (verified offsets)
      - Dynamic GUID query from driver (GetEncodeGUIDs / GetEncodePresetGUIDs)
      - Primary context management (cuDevicePrimaryCtxRetain + cuCtxPushCurrent)
    """

    def __init__(self, width: int, height: int, fps: float,
                 preset: str = "p1", qp: int = 0,
                 codec: str = "h264", pipeline_depth: int = 4,
                 rate_mode: str = "constqp", la_depth: int = 0):
        """rate_mode: 'constqp' | 'vbr_hq' (CQ via VBR_HQ + targetQuality) | 'qvbr'.
           la_depth: lookahead depth (0=disabled, 8~32 for -rc-lookahead equivalent).
           将自动校准为 >= LA+1 (SDK 硬件安全要求)。"""
        if codec != "h264":
            raise ValueError("NVENCEncoder: only H.264 supported, got: " + codec)

        self._width = width
        self._height = height
        self._fps = fps
        self._qp = qp
        self._rate_mode = rate_mode
        self._la_depth = la_depth

        # ── 概念分离（参见 pipeline-depth-slot-rotation-confusion.md）──
        # _required_buffers: SDK 硬件安全下限 — la_depth==0 时为 1，否则 >= LA+1
        # _slot_count: 实际分配的 slot 对数（buffer pool 大小 + 轮转模数）
        # LA=0 时 _slot_count=2 (ce_pipeline 真正的 2 帧流水线深度)
        # LA>0 时 _slot_count=max(2, LA+1) (膨胀为 buffer pool，累积模式不参与轮转)
        # CONSTQP 下硬件静默禁用 LA，代码层面清零以与硬件行为一致
        if rate_mode == "constqp":
            la_depth = 0  # 硬件静默禁用 LA，此处显式清零
            self._la_depth = 0  # 同步更新实例变量，确保后续代码路径与 Ready 日志一致
        _required_buffers = max(1, la_depth + 1)  # SDK 硬件安全要求: buffer 数 >= LA+1
        _slot_count = max(pipeline_depth, _required_buffers)

        print("[NVENCEncoder] %s + LA=%d: %d slots (HW pipeline buffers>=%d)" %
              (rate_mode.upper(), la_depth, _slot_count, _required_buffers), flush=True)

        self._preset_name = preset.lower()
        self._encoder = c_void_p(None)
        self._frame_idx = 0
        # [FIX-LA-OUTPTR] 独立输出槽位指针：跟踪下一个预期输出的 slot，
        # 确保 LA 延迟产出帧按正确顺序取回（参照 test_nvenc_la_frame_conservation.py）
        self._output_slot_idx = 0
        # [FIX-LA-CHUNK-STREAM] LA 分块流式编码的跨调用持久状态。
        # 原 [FIX-LA-ACCUMULATE] 设计要求整段累积后单次 encode_frames_batch()，
        # 因为 _slot_pending/_slots_warmed 是批次局部变量，跨批次调用会导致
        # LA 窗口帧映射失效。现将二者提升为实例状态，配合 _strm_next_fi（流内
        # 全局帧号），使 lookahead 窗口帧可跨 chunk 正确映射回其帧号，
        # 从而支持有界分块编码（见 encode_frames_stream）。
        # [FIX-FIFO-DRAIN] per-slot FIFO deque 表（对齐 realesrgan_video _slot_pending）：
        # slot_idx -> deque[(gfi, bs_buf, force_idr, ep_status)]。同槽多条记录按提交
        # 顺序排队，drain 时队首即该槽下一次必取回的帧；append 永不覆盖。
        self._strm_slot_pending: dict = {}
        # [FIX-PIPE4-LA8] 原 _strm_slots_warmed per-slot IDR 表已删除（死状态）：
        # IDR 判定收敛为 fi==0 单 IDR，不再依赖 slot warm 标记。
        self._strm_next_fi: int = 0    # 当前流内下一个提交帧的流内序号（0 起）
        self._strm_active: bool = False
        # [FIX-FIFO-DRAIN] 段首全局 fi 基线（对齐 realesrgan_video chunk_start_global）：
        # drain 标签 = 队首条目全局 fi - _strm_ts_base = 流内帧号。ts 重关联
        # （outputTimeStamp 信任机制，[FIX-STREAM-TS-REASSOC]）已整体删除：
        # 辅助块场景下 ts 回显不可信（test7 seg1 abort 后标签错位根因）。
        # 原 [FIX-LA-WINDOW-ROTATION] 状态此前已移除（被证明为失败方案）。
        self._strm_ts_base: int = 0
        # [FIX-DRAIN-ORDER-DEFENSE] 上次已消费的队首全局 gfi（段内单调检测基准）
        self._last_drained_gfi = None
        # [FIX-EMPTY-PREV-FILL] 流内顺序最后一帧的 H.264 码流（空帧兜底填充用）
        self._prev_stream_h264 = None
        # [FIX-DRAIN-ORDER-DEFENSE] 一致性诊断计数器（沿用 _diag_empty 命名风格）
        self._diag_slot_mismatch = 0
        self._diag_gfi_regress = 0
        # [FIX-AUX-NO-CLEAR] / [FIX-DRAIN-ORDER-DEFENSE] 诊断计数
        self._diag_aux_block = 0
        self._diag_phase_shift = 0
        self._diag_slot_drain_fallback = 0
        self._lock = threading.Lock()
        # [FIX-ASYNC-COPY] 专用非默认 CUDA stream，用于 NVENC 输入拷贝
        # (cuMemcpy2DAsync)。提前声明为 None，防止初始化中途失败时
        # close() 访问未定义属性。
        self._stream_encode = c_void_p(None)

        # 多 slot 缓冲池: _slot_count >= _required_buffers (SDK 硬件安全要求)
        self._slot_count = _slot_count
        self._slots: list = []

        # Backward compat: legacy refs (initialized after slot creation)
        self._input_buf_handle = c_void_p(None)
        self._bs_handle = c_void_p(None)

        # [SEGMENT-REUSE] 缓存首段 SPS+PPS NAL 单元，后续段预挂到首帧前
        self._cached_sps_pps: Optional[bytes] = None
        self._sps_pps_injected: bool = False  # [FIX-SPS-PPS-V2] Writer-thread-side 注入已完成标志
        # [FIX-SPS-PPS] muxer 引用 — _cached_sps_pps 首次设置时通知 muxer 预注入
        self._muxer_ref: Optional[object] = None

        # 1. Load NVENC DLL
        self._dll_path = self._find_dll()
        try:
            self._dll = ctypes.CDLL(self._dll_path)
        except OSError as e:
            raise RuntimeError(
                "[NVENCEncoder] Cannot load NVENC DLL (%s): %s. "
                "Please verify NVIDIA driver is installed." % (self._dll_path, e))

        # 2. Load libcuda for GPU operations
        try:
            self._libcuda = ctypes.CDLL("libcuda.so.1" if sys.platform != "win32" else "nvcuda.dll")
            self._libcuda.cuInit(0)
        except Exception as e:
            raise RuntimeError("[NVENCEncoder] Cannot load CUDA library: %s" % e)

        # 3. Runtime API version detection
        try:
            _get_max_ver = self._dll.NvEncodeAPIGetMaxSupportedVersion
            _get_max_ver.restype = c_uint32
            _get_max_ver.argtypes = [ctypes.POINTER(c_uint32)]
            _max_ver_val = c_uint32(0)
            _get_max_ver(ctypes.byref(_max_ver_val))
            _nvenc_api_version = _max_ver_val.value if _max_ver_val.value > 0 else _NVENCAPI_VERSION_FALLBACK
        except Exception:
            _nvenc_api_version = _NVENCAPI_VERSION_FALLBACK
        print("[NVENCEncoder] NVENC API: 0x%x = v%d.%d" % (
            _nvenc_api_version, _nvenc_api_version >> 4, _nvenc_api_version & 0xF), flush=True)

        # 4. CUDA context setup — use primary context (cuCtxCreate fails code=2 on T4)
        self._saved_ctx = c_void_p(None)
        self._nvenc_own_ctx = False
        cuda_ctx = c_void_p(None)

        # Save current context
        self._libcuda.cuCtxGetCurrent.restype = c_uint32
        self._libcuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(c_void_p)]
        self._libcuda.cuCtxGetCurrent(ctypes.byref(self._saved_ctx))

        # Get primary context for device 0
        self._libcuda.cuDevicePrimaryCtxRetain.restype = c_uint32
        self._libcuda.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(c_void_p), c_int]
        primary_ctx = c_void_p(None)
        r = self._libcuda.cuDevicePrimaryCtxRetain(ctypes.byref(primary_ctx), c_int(0))
        if r == 0 and primary_ctx.value is not None:
            self._libcuda.cuCtxPushCurrent.restype = c_uint32
            self._libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
            self._libcuda.cuCtxPushCurrent(primary_ctx)
            cuda_ctx = primary_ctx
            self._primary_ctx = c_void_p(primary_ctx.value)  # [FIX-GPU-STAY] 跨线程 CUDA context 保护
            print("[NVENCEncoder] 使用 primary CUDA context (0x%x)" % primary_ctx.value, flush=True)
        elif self._saved_ctx.value is not None:
            cuda_ctx = self._saved_ctx
            print("[NVENCEncoder] 使用已存在的 CUDA context (0x%x)" % self._saved_ctx.value, flush=True)
        else:
            raise RuntimeError("[NVENCEncoder] 无法获取 CUDA context")

        # 5. NvEncodeAPICreateInstance
        _FUNC_TABLE_RAW_SIZE = 2552  # SDK 13.0 func_table size in bytes
        func_table = (c_uint8 * _FUNC_TABLE_RAW_SIZE)()
        _flist_ver = _sdk13_ver(2)  # NV_ENCODE_API_FUNCTION_LIST_VER = NVENCAPI_STRUCT_VERSION(2) = 0x7002000d
        cast(func_table, ctypes.POINTER(c_uint32))[0] = _flist_ver
        create_instance = _NvEncodeAPICreateInstanceProto(
            ("NvEncodeAPICreateInstance", self._dll))
        status = create_instance(cast(func_table, c_void_p))
        if status != NV_ENC_SUCCESS:
            raise RuntimeError(
                "[NVENCEncoder] NvEncodeAPICreateInstance failed, code=%d. "
                "Verify GPU supports NVENC." % status)

        # func_ptrs at offset 8 (skip version + reserved)
        self._func_ptrs = cast(byref(func_table, 8), ctypes.POINTER(c_void_p))
        # Keep ref to prevent GC
        self._func_table_raw = func_table

        def _get_func(idx):
            addr = self._func_ptrs[idx]
            if not addr or addr == 0:
                return None
            return addr

        # 6. OpenEncodeSessionEx
        open_func_addr = _get_func(_FUNC_IDX["OpenEncodeSessionEx"])
        if open_func_addr is None:
            raise RuntimeError("[NVENCEncoder] OpenEncodeSessionEx not available")
        open_session = _NvEncOpenEncodeSessionExProto(open_func_addr)

        # SDK 13.0 apiVersion uses NVENCAPI_VERSION = MAJOR | (MINOR << 24) = 0x0d
        # Also try old format (MAJOR << 4) | MINOR = 0xd0 for backward compat
        _api_try_list = sorted(set([
            _NVENCAPI_VERSION,       # 0x0d — SDK 13.0 new format
            _nvenc_api_version,      # from driver (0xd0)
            (13 << 4) | 0,           # 0xd0 — old format 13.0
            (12 << 4) | 0,           # 0xc0 — old format 12.0
        ]), reverse=True)
        status = 0xFFFF
        for _api_ver in _api_try_list:
            session_params = _NvEncOpenEncodeSessionExParams()
            session_params.version = _sdk13_ver(1)  # NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER = NVENCAPI_STRUCT_VERSION(1) = 0x7001000d
            session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA
            session_params.device = cuda_ctx
            session_params.apiVersion = _api_ver
            self._encoder = c_void_p(None)
            status = open_session(byref(session_params), byref(self._encoder))
            if status == NV_ENC_SUCCESS:
                print("[NVENCEncoder] OpenEncodeSessionEx OK: apiVersion=0x%x (%d.%d)" % (
                    _api_ver, _api_ver >> 4, _api_ver & 0xF), flush=True)
                break
            if status != 15:  # non-INVALID_VERSION, don't retry
                break
        if status != NV_ENC_SUCCESS:
            raise RuntimeError(
                "[NVENCEncoder] nvEncOpenEncodeSessionEx failed, code=%d" % status)

        # 7. Dynamic GUID query from driver
        _GetEncodeGUIDCountProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint32))
        _GetEncodeGUIDsProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(_NvGuid), c_uint32, ctypes.POINTER(c_uint32))
        _GetEncodePresetGUIDsProto = ctypes.CFUNCTYPE(
            c_uint32, c_void_p, _NvGuid, ctypes.POINTER(_NvGuid), c_uint32, ctypes.POINTER(c_uint32))

        count_val = c_uint32(0)
        s = _GetEncodeGUIDCountProto(_get_func(_FUNC_IDX["GetEncodeGUIDCount"]))(self._encoder, byref(count_val))
        if s != 0 or count_val.value == 0:
            raise RuntimeError("[NVENCEncoder] GetEncodeGUIDCount failed, code=%d" % s)

        n_guids = count_val.value
        guid_array = (_NvGuid * n_guids)()
        ctypes.memset(cast(guid_array, c_void_p), 0, sizeof(guid_array))
        actual_count = c_uint32(0)
        s = _GetEncodeGUIDsProto(_get_func(_FUNC_IDX["GetEncodeGUIDs"]))(self._encoder, guid_array, n_guids, byref(actual_count))
        if s != 0 or actual_count.value == 0:
            raise RuntimeError("[NVENCEncoder] GetEncodeGUIDs failed, code=%d" % s)
        codec_guid = guid_array[0]
        print("[NVENCEncoder] Driver codec GUID: %08x-%04x-%04x" % (
            codec_guid.Data1, codec_guid.Data2, codec_guid.Data3), flush=True)

        preset_guid_array = (_NvGuid * 64)()
        ctypes.memset(cast(preset_guid_array, c_void_p), 0, sizeof(preset_guid_array))
        preset_count = c_uint32(0)
        s = _GetEncodePresetGUIDsProto(_get_func(_FUNC_IDX["GetEncodePresetGUIDs"]))(
            self._encoder, codec_guid, preset_guid_array, 64, byref(preset_count))
        if s != 0 or preset_count.value == 0:
            raise RuntimeError("[NVENCEncoder] GetEncodePresetGUIDs failed, code=%d" % s)
        _p_idx = _PRESET_P_INDEX.get(self._preset_name, 4)
        _p_idx = min(_p_idx, preset_count.value - 1)
        preset_guid = preset_guid_array[_p_idx]
        print("[NVENCEncoder] Driver preset GUID: %08x-%04x-%04x (index=%d/%d)" % (
            preset_guid.Data1, preset_guid.Data2, preset_guid.Data3, _p_idx, preset_count.value - 1), flush=True)

        # 8. GetEncodePresetConfig
        get_preset_addr = _get_func(_FUNC_IDX["GetEncodePresetConfig"])
        if get_preset_addr is None:
            get_preset_addr = _get_func(_FUNC_IDX["GetEncodePresetConfigEx"])
        if get_preset_addr is None:
            raise RuntimeError("[NVENCEncoder] GetEncodePresetConfig not available")
        _GPC_fn = ctypes.CFUNCTYPE(c_uint32, c_void_p, _NvGuid, _NvGuid, ctypes.POINTER(_NvEncPresetConfig))

        preset_config = _NvEncPresetConfig()
        ctypes.memset(byref(preset_config), 0, sizeof(preset_config))
        preset_config.version = NV_ENC_PRESET_CONFIG_VER
        # SDK 中 presetCfg 从 offset 8 开始（version@0 + 4-byte padding）
        # ctypes _pack_=1 让它从 offset 4 开始，但 SDK 函数按 SDK 布局解析
        cast(byref(preset_config, 8), ctypes.POINTER(c_uint32))[0] = NV_ENC_CONFIG_VER

        status = _GPC_fn(get_preset_addr)(self._encoder, codec_guid, preset_guid, byref(preset_config))
        if status != NV_ENC_SUCCESS:
            raise RuntimeError("[NVENCEncoder] GetEncodePresetConfig failed, code=%d" % status)
        print("[NVENCEncoder] GetEncodePresetConfig OK", flush=True)

        # Configure encoding params (presetCfg at SDK offset 8)
        enc_cfg = cast(byref(preset_config, 8), ctypes.POINTER(_NvEncConfig)).contents
        enc_cfg.gopLength = int(fps)
        enc_cfg.frameIntervalP = 1
        # [FIX-SDK13-CODEC] NV_ENC_CONFIG_H264 字段按 SDK 13.0 真实偏移硬编码写入（对齐 rc_params 模式）。
        # 背景: _NvEncConfig.encodeCodecConfig@1052 是错误布局 (SDK 真实 @168=绝对 176;
        #   reserved3[53] 覆盖 rcParams 区 + reserved5[172] 推至 1052)，经结构体写入的
        #   chromaFormatIDC/idrPeriod/maxNumRefFrames/repeatSPSPPS/profileLevel 全部落在
        #   NV_ENC_CONFIG.reserved[278] 保留区 → 驱动忽略。sweep 实测: 绝对 180=level 命中
        #   level_idc=51; 绝对 176=config 起始写坏 → InitializeEncoder 拒绝。
        # 偏移基准: _h264_cfg_off = 8(presetCfg) + 168(encodeCodecConfig)。
        _h264_cfg_off = 8 + 168
        # chromaFormatIDC@192 (VUI 112B 之后): 1=4:2:0，防止 SPS 声明 monochrome → 灰色输出 [FIX-CHROMA]
        cast(byref(preset_config, _h264_cfg_off + 192), ctypes.POINTER(c_uint32))[0] = 1
        # idrPeriod@8: IDR 周期 = fps（驱动默认回退 gopLength）
        cast(byref(preset_config, _h264_cfg_off + 8), ctypes.POINTER(c_uint32))[0] = int(fps)
        # maxNumRefFrames@60 (DPB): 16→4 减少运动估计开销 ~50% [FIX-GPU-STAY]
        cast(byref(preset_config, _h264_cfg_off + 60), ctypes.POINTER(c_uint32))[0] = 4
        # profileLevel@4: H.264 Level 5.1 — 限制 NVENC 自动 level 不超 T4 NVDEC 上限 (L5.2)
        cast(byref(preset_config, _h264_cfg_off + 4), ctypes.POINTER(c_uint32))[0] = 51
        # repeatSPSPPS (bitfield@0 bit12) 不写: 历史上从未生效，SPS/PPS 由 _apply_sps_pps()
        #   手动缓存/预挂替代 ([SEGMENT-REUSE][FIX-SPS-PPS-V2])，行为保持不变。

        # [V6441-RC-FIXED] NV_ENC_RC_PARAMS at offset 40 in NV_ENC_CONFIG
        # SDK 13.0 layout (nvEncodeAPI.h, SEQUENTIAL — NO union at offset 8):
        #   NV_ENC_QP = {qpInterP@0, qpInterB@4, qpIntra@8} = 12 bytes sequential.
        #   version@0, rateControlMode@4,
        #   constQP@8 {qpInterP@8, qpInterB@12, qpIntra@16} (12B NV_ENC_QP struct),
        #   averageBitRate@20, maxBitRate@24,
        #   vbvBufferSize@28, vbvInitialDelay@32,
        #   bitfield@36 (enableMinQP:1, enableMaxQP:1, enableInitialRCQP:1,
        #     enableAQ:1=bit3, reserved:1=bit4, enableLookahead:1=bit5,
        #     disableIadapt:1=bit6, disableBadapt:1=bit7, enableTemporalAQ:1=bit8, ...)
        #   minQP@40 {InterP@40,InterB@44,Intra@48}, maxQP@52, initialRCQP@64,
        #   ── SEQUENTIAL layout (nvEncodeAPI.h master + GPU verified, 2026-06-09) ──
        #   temporallayerIdxMask@76, temporalLayerQP@80[8],
        #   targetQuality@88(uint8), targetQualityLSB@89(uint8),
        #   lookaheadDepth@90(uint16)  — qvbrQuality shares targetQuality slot
        #   lowDelayKeyFrameScale@92, yDcQPIndexOffset@93, uDcQPIndexOffset@94,
        #   vDcQPIndexOffset@95, qpMapMode@96, multiPass@100(uint32, rc_ptr[25]),
        #   alphaLayerBitrateRatio@104, cbQP@108, crQP@109, reserved2@110,
        #   lookaheadLevel@112, viewBitrateRatios@116[7], reserved3@123, reserved1@124
        rc_ptr = cast(byref(preset_config, 8 + 40), ctypes.POINTER(c_uint32))
        rc_ptr[0] = _sdk13_ver(1)                  # NV_ENC_RC_PARAMS_VER
        if self._rate_mode == 'constqp':
            _qp_val = qp          # CONSTQP: qp=0 合法（lossless），qp>0 正常
        elif qp <= 0:
            _qp_val = 1           # VBR_HQ/QVBR: targetQuality 范围 1-51, 1=最好 → crf=0 映射为 1
        else:
            _qp_val = qp          # VBR_HQ/QVBR: qp>0 保持直接映射

        if self._rate_mode == 'vbr_hq':
            # CQ mode: VBR_HQ + targetQuality (matches Level 2 -cq:v N behavior)
            # NV_ENC_PARAMS_RC_VBR_HQ = 32 (0x20) in SDK 12.0+.
            rc_ptr[1] = 32                           # NV_ENC_PARAMS_RC_VBR_HQ
            # [FIX-BR-CEILING] 提供合理的 avgBitRate 作为速度天花板。
            # targetQuality 仍主导质量决策，avgBitRate 仅防止 NVENC 在无约束下进入
            # 极慢的质量穷举搜索模式（GPU 闲置 65%，FPS 暴跌 2.2×）。
            # 参考 v6.4.3.1 的 _est_br 计算方式。
            # [FIX-BR-CLAMP] avgBitrate 钳制：估算值对高分辨率×帧率会失控（如 720p48
            # 算出 132.6Mbps），超驱动 NVENC 合法范围可能触发 InitializeEncoder code=8
            # （关注点4 场景）。钳制到 _NVENC_BR_CLAMP_MAX，maxBitRate 同步。
            _est_br = _clamp_bitrate(int(width * height * fps * 3.0))
            rc_ptr[5] = _est_br                      # averageBitRate @offset 20 (速度天花板)
            rc_ptr[6] = _est_br * 2                  # maxBitRate @offset 24
            _tq = max(1, _qp_val)  # VBR_HQ targetQuality = CRF (QP标度, 1=最好, 51=最差)
            # targetQuality: uint8_t at rcParams+88 (nvEncodeAPI.h SEQUENTIAL, GPU verified)
            _tq8_ptr = cast(byref(preset_config, 8 + 40 + 88), ctypes.POINTER(c_uint8))
            _tq8_ptr[0] = _tq & 0xFF
            # [DIAG-TERM] targetQuality 是 RC 质量参数（CRF 语义 1-51），不是 H.264 Level。
            print(f"[NVENCEncoder] VBR_HQ: crf={_qp_val} targetQuality(CRF)={_tq} "
                  f"avgBitrate={_est_br//1000}kbps", flush=True)
        elif self._rate_mode == 'qvbr':
            # QVBR mode: NV_ENC_PARAMS_RC_QVBR = 64 (0x40)
            rc_ptr[1] = 64                           # NV_ENC_PARAMS_RC_QVBR (0x40)
            # [FIX-BR-CEILING] QVBR 也需要 avgBitRate 作为速度天花板，
            # 防止 NVENC 在无约束下进入极慢的质量搜索。
            # [FIX-BR-CLAMP] 同 VBR_HQ：估算值钳制，避免超驱动合法范围触发 code=8。
            _est_br = _clamp_bitrate(int(width * height * fps * 3.0))
            rc_ptr[5] = _est_br                      # averageBitRate @offset 20 (速度天花板)
            rc_ptr[6] = _est_br * 2                  # maxBitRate @offset 24 (码率上限)
            _tq = max(1, _qp_val)  # QVBR qvbrQuality = CRF (QP标度, 低值=高质量)
            _tq8_ptr = cast(byref(preset_config, 8 + 40 + 88), ctypes.POINTER(c_uint8))
            _tq8_ptr[0] = _tq & 0xFF
            # [OPT-VBV] VBV 缓冲合规由全局开关 _NVENC_QVBR_ENABLE_VBV 控制。
            # 离线编码不需要 VBV（虚拟解码器缓冲）合规约束，默认关闭。
            if _NVENC_QVBR_ENABLE_VBV:
                if rc_ptr[7] == 0:
                    rc_ptr[7] = 4194304              # vbvBufferSize @offset 28
                if rc_ptr[8] == 0:
                    rc_ptr[8] = 2097152              # vbvInitialDelay @offset 32
            # [DIAG-TERM] qvbrQuality 是 RC 质量参数（CRF 语义 1-51），不是 H.264 Level。
            print(f"[NVENCEncoder] QVBR: crf={_qp_val} qvbrQuality(CRF)={_tq} maxBitrate={_est_br//1000}kbps", flush=True)
        else:
            # CONSTQP mode (default): direct QP control
            rc_ptr[1] = 0                            # NV_ENC_PARAMS_RC_CONSTQP
            rc_ptr[2] = _qp_val                      # constQP.qpInterP @offset 8
            rc_ptr[3] = _qp_val                      # constQP.qpInterB @offset 12
            rc_ptr[4] = _qp_val                      # constQP.qpIntra @offset 16

        # Enable AQ + Temporal AQ via rcParams bitfield (offset 36 = rc_ptr[9])
        # bit 3 = enableAQ, bit 8 = enableTemporalAQ (NOT bit 7!)
        rc_ptr[9] = rc_ptr[9] | (1 << 3) | (1 << 8)

        # Optional: lookahead for VBR_HQ/QVBR (matching Level 2 -rc-lookahead N)
        if self._la_depth > 0 and self._rate_mode in ('vbr_hq', 'qvbr'):
            _rc_bf = rc_ptr[9]                       # bitfield @offset 36
            _rc_bf |= (1 << 5)                       # enableLookahead (bit 5, NOT bit 4)
            rc_ptr[9] = _rc_bf
            # multiPass — uint32 at rc_ptr[25] (SEPARATE field, NOT in bitfield)
            rc_ptr[25] = 0                            # NV_ENC_MULTI_PASS_DISABLED (VBR_HQ/QVBR 不支持 two-pass)
            # lookaheadDepth — uint16 at rcParams+90 (nvEncodeAPI.h verified)
            _la_ptr = cast(byref(preset_config, 8 + 40 + 90), ctypes.POINTER(c_uint16))
            _la_ptr[0] = self._la_depth

        # 9. InitializeEncoder
        init_params = _NvEncInitializeParams()
        ctypes.memset(byref(init_params), 0, sizeof(init_params))
        init_params.version = NV_ENC_INITIALIZE_PARAMS_VER
        init_params.encodeGUID = codec_guid
        init_params.presetGUID = preset_guid
        init_params.encodeWidth = width
        init_params.encodeHeight = height
        init_params.darWidth = width
        init_params.darHeight = height
        init_params.frameRateNum = int(fps * 1000)
        init_params.frameRateDen = 1000
        init_params.maxEncodeWidth = width
        init_params.maxEncodeHeight = height
        init_params.enablePTD = 1
        init_params.encodeConfig = cast(byref(preset_config, 8), c_void_p)

        init_addr = _get_func(_FUNC_IDX["InitializeEncoder"])
        if init_addr is None:
            raise RuntimeError("[NVENCEncoder] InitializeEncoder not available")
        create_encoder = _NvEncCreateEncoderProto(init_addr)
        status = create_encoder(self._encoder, byref(init_params))
        if status != NV_ENC_SUCCESS:
            # [FIX-CODEC8-DIAG] code=8=INVALID_PARAM 时输出完整参数诊断，便于定位
            # 非法组合（高分辨率×高帧率 + 严格 preset + 失控 avgBitrate 等）。
            # 参考关注点4：IFRNet 720p48 曾因 avgBitrate 估算失控(132.6Mbps) 触发。
            _diag_rc = getattr(self, '_rate_mode', '?')
            _diag_br = cast(byref(preset_config, 8 + 40 + 20), ctypes.POINTER(c_uint32))[0]
            _diag_brmax = cast(byref(preset_config, 8 + 40 + 24), ctypes.POINTER(c_uint32))[0]
            _diag_tq = cast(byref(preset_config, 8 + 40 + 88), ctypes.POINTER(c_uint8))[0]
            _diag_la = cast(byref(preset_config, 8 + 40 + 90), ctypes.POINTER(c_uint16))[0]
            print(f"[NVENCEncoder] InitializeEncoder FAILED code={status} | "
                  f"cfg: {width}x{height}@{fps:.2f}fps rc={_diag_rc} preset={self._preset_name} "
                  f"slots={self._slot_count} | avgBR={_diag_br}b maxBR={_diag_brmax}b "
                  f"targetQuality={_diag_tq} LA={_diag_la}", flush=True)
            raise RuntimeError("[NVENCEncoder] InitializeEncoder failed, code=%d" % status)
        # [DIAG-LEVEL] 请求的 H.264 level（0=AUTO：NVENC 自动选择可能超 T4 NVDEC 上限 L5.2 → 软解回退）。
        # 实际生效值见首帧 "[NVENCEncoder] H.264 SPS actual" 日志——两者不一致 = profileLevel 写入未生效。
        # [FIX-SDK13-CODEC] 请求 level 已硬编码写入 SDK 真实偏移 (绝对 180)，此处直接回显 51。
        _req_lvl = 51
        print("[NVENCEncoder] H.264 level requested: %d (0=AUTO; 51=L5.1)" % _req_lvl, flush=True)

        # 10. [PHASE4-v645] 创建多 slot 流水线：每 slot = input buffer + bitstream buffer + CUDA event。
        #     根因修复（v6.4.4 synchronize + FIX-ENC-CTX）后恢复，Multi-slot 让 NVENC HW 帧间流水线化：
        #     准备 slot N+1 的同时 NVENC 处理 slot N，Lock/Copy 与 Encode 重叠。
        nv12_h = height + height // 2
        _CreateInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 776))
        _CreateBitstreamBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 776))

        for slot_idx in range(self._slot_count):
            # 10a. Create input buffer
            create_buf = (c_uint8 * 776)()
            ctypes.memset(create_buf, 0, 776)
            cast(create_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_CREATE_INPUT_BUFFER_VER  # version@0
            cast(byref(create_buf, 4), ctypes.POINTER(c_uint32))[0] = width               # width@4
            cast(byref(create_buf, 8), ctypes.POINTER(c_uint32))[0] = height              # height@8 (luma only for NV12) [FIX-HEIGHT]
            cast(byref(create_buf, 16), ctypes.POINTER(c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12  # bufferFmt@16

            s = _CreateInputBufferProto(_get_func(_FUNC_IDX["CreateInputBuffer"]))(self._encoder, create_buf)
            if s != 0:
                self._destroy_all_slots()
                raise RuntimeError("[NVENCEncoder] CreateInputBuffer[%d] failed, code=%d" % (slot_idx, s))
            _raw_ptr = cast(byref(create_buf, 24), ctypes.POINTER(c_void_p))[0]  # inputBuffer@24
            input_handle = c_void_p(_raw_ptr if isinstance(_raw_ptr, int) else (_raw_ptr.value or 0))

            # 10b. Create bitstream buffer
            bs_buf = (c_uint8 * 776)()
            ctypes.memset(bs_buf, 0, 776)
            cast(bs_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_CREATE_BITSTREAM_BUFFER_VER  # version@0

            s = _CreateBitstreamBufferProto(_get_func(_FUNC_IDX["CreateBitstreamBuffer"]))(self._encoder, bs_buf)
            if s != 0:
                self._destroy_all_slots()
                raise RuntimeError("[NVENCEncoder] CreateBitstreamBuffer[%d] failed, code=%d" % (slot_idx, s))
            _raw_bs = cast(byref(bs_buf, 16), ctypes.POINTER(c_void_p))[0]  # bitstreamBuffer@16
            bs_handle = c_void_p(_raw_bs if isinstance(_raw_bs, int) else (_raw_bs.value or 0))

            # 10c. Create CUDA completion event (async ready signal for NVENC HW)
            event = c_void_p(None)
            r = self._libcuda.cuEventCreate(ctypes.byref(event), 0)  # 0 = cudaEventDefault
            if r != 0:
                self._destroy_all_slots()
                raise RuntimeError("[NVENCEncoder] cuEventCreate[%d] failed, code=%d" % (slot_idx, r))

            self._slots.append({
                'input_buf': input_handle,
                'bs_buf': bs_handle,
                'event': event,
            })

        # Backward compat: legacy refs point to slot 0
        self._input_buf_handle = self._slots[0]['input_buf']
        self._bs_handle = self._slots[0]['bs_buf']

        if self._slot_count > 1:
            print("[NVENCEncoder] %d slots created (0x%x..0x%x)" %
                  (self._slot_count, self._slots[0]['input_buf'].value,
                   self._slots[-1]['input_buf'].value), flush=True)

        # Setup cuMemcpyDtoD (1D linear, deprecated for pitched buffers)
        self._libcuda.cuMemcpyDtoD_v2.restype = c_uint32
        self._libcuda.cuMemcpyDtoD_v2.argtypes = [c_void_p, c_void_p, c_size_t]

        # Setup cuMemcpy2D_v2 (2D pitch-aware copy, SYNC — 仅保留作为异常回退用)
        self._libcuda.cuMemcpy2D_v2.restype = c_uint32
        self._libcuda.cuMemcpy2D_v2.argtypes = [c_void_p]

        # [FIX-ASYNC-COPY] cuMemcpy2D_v2（同步、走 legacy default/null stream）
        # 会隐式地在本 CUDA context 下的**所有** stream 之间做全局同步——
        # 每次 NVENC 输入拷贝都会强制等待推理用的 stream_h2d/stream_compute/
        # stream_d2h 排空，反之亦然。在 LA 分块流式编码（chunk=768 帧）场景下，
        # 编码线程连续、逐帧执行这个同步拷贝，导致推理线程的 GPU kernel 被
        # 周期性阻塞，表现为 nvtop/nvitop 中规律出现的 GPU 利用率骤降。
        #
        # 修复：改用 cuMemcpy2DAsync_v2 + 专用非默认 CUDA stream
        # (self._stream_encode)。拷贝仍然入队后立即 cuStreamSynchronize，
        # 编码线程自身的等待时间不变，但**只同步这一个私有 stream**，
        # 不再对其他 stream 施加隐式全局屏障，从而让 T2(推理) 与 T3(编码)
        # 在 GPU 上真正并行重叠。
        self._libcuda.cuMemcpy2DAsync_v2.restype = c_uint32
        self._libcuda.cuMemcpy2DAsync_v2.argtypes = [c_void_p, c_void_p]
        self._libcuda.cuStreamSynchronize.restype = c_uint32
        self._libcuda.cuStreamSynchronize.argtypes = [c_void_p]
        self._libcuda.cuStreamCreate.restype = c_uint32
        self._libcuda.cuStreamCreate.argtypes = [ctypes.POINTER(c_void_p), c_uint32]
        _CU_STREAM_NON_BLOCKING = 1  # 不与 legacy stream 0 隐式同步
        _r_stream = self._libcuda.cuStreamCreate(
            ctypes.byref(self._stream_encode), c_uint32(_CU_STREAM_NON_BLOCKING))
        if _r_stream != 0 or self._stream_encode.value is None:
            print(f"[NVENCEncoder] ⚠️ cuStreamCreate 失败(code={_r_stream})，"
                  f"回退到同步 cuMemcpy2D（可能出现周期性 GPU 阻塞）", flush=True)
            self._stream_encode = c_void_p(None)
        else:
            print(f"[NVENCEncoder] [FIX-ASYNC-COPY] 专用编码拷贝 stream 已创建 "
                  f"(0x{self._stream_encode.value:x}, non-blocking)", flush=True)

        # Setup context push/pop helpers
        self._libcuda.cuCtxPopCurrent.restype = c_uint32
        self._libcuda.cuCtxPopCurrent.argtypes = [ctypes.POINTER(c_void_p)]

        _mode_label = (self._rate_mode.upper() if self._rate_mode == 'vbr_hq'
                       else 'QVBR' if self._rate_mode == 'qvbr' else 'CONSTQP')
        _extra = ""
        if self._la_depth > 0:
            _extra += " la=%d" % self._la_depth
        if self._rate_mode in ('vbr_hq', 'qvbr'):
            _tq = max(1, _qp_val)  # QVBR qvbrQuality = CRF (QP标度)
            _extra += " tq=%d" % _tq
        print("[NVENCEncoder] Ready: %dx%d@%.1ffps H.264 %s QP=%d preset=%s slots=%d%s (GPU direct SDK 13.0)" %
              (width, height, fps, _mode_label, _qp_val, self._preset_name, self._slot_count, _extra), flush=True)

    def _copy_into_input_buffer(self, cpy2d_struct) -> int:
        """[FIX-ASYNC-COPY] 把 NV12 源数据拷贝进 NVENC 输入缓冲区。

        优先走 cuMemcpy2DAsync_v2 + 专用非默认 stream(self._stream_encode)，
        入队后立即 cuStreamSynchronize —— 编码线程自身仍然同步等待拷贝完成
        (语义与之前一致，帧顺序/正确性不变)，但只同步这一个私有 stream，
        不再像 cuMemcpy2D_v2(legacy null stream) 那样对同 context 下所有
        stream(包括推理用的 stream_h2d/stream_compute/stream_d2h)施加隐式
        全局屏障。这是解决"推理 GPU 利用率周期性骤降"的核心修复点。

        若专用 stream 创建失败(见 __init__)，回退到原始同步 cuMemcpy2D_v2，
        保证功能不中断，仅退化为修复前的行为。

        Returns:
            CUDA error code (0 = success)，语义与原 cuMemcpy2D_v2 返回值一致。
        """
        _ptr = cast(cpy2d_struct, c_void_p)
        if self._stream_encode.value is not None:
            r = self._libcuda.cuMemcpy2DAsync_v2(_ptr, self._stream_encode)
            if r != 0:
                return r
            return self._libcuda.cuStreamSynchronize(self._stream_encode)
        # Fallback: 专用 stream 不可用，退回同步拷贝（旧行为）。
        return self._libcuda.cuMemcpy2D_v2(_ptr)

    def set_muxer_ref(self, muxer: object) -> None:
        """[FIX-SPS-PPS] 设置 muxer 引用，供 _cached_sps_pps 首次缓存时预注入用。"""
        self._muxer_ref = muxer

    def _destroy_slot(self, slot: dict):
        """Destroy a single pipeline slot's buffers and event."""
        if slot.get('input_buf') and slot['input_buf'].value is not None:
            try:
                _DestroyInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
                addr = self._func_ptrs[_FUNC_IDX["DestroyInputBuffer"]]
                if addr:
                    _DestroyInputBufferProto(addr)(self._encoder, slot['input_buf'])
            except Exception:
                pass
            slot['input_buf'] = c_void_p(None)
        if slot.get('bs_buf') and slot['bs_buf'].value is not None:
            try:
                _DestroyBitstreamBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
                addr = self._func_ptrs[_FUNC_IDX["DestroyBitstreamBuffer"]]
                if addr:
                    _DestroyBitstreamBufferProto(addr)(self._encoder, slot['bs_buf'])
            except Exception:
                pass
            slot['bs_buf'] = c_void_p(None)
        if slot.get('event') and slot['event'].value is not None:
            try:
                self._libcuda.cuEventDestroy(slot['event'])
            except Exception:
                pass
            slot['event'] = c_void_p(None)

    def _destroy_all_slots(self):
        """Destroy all pipeline slots (used on init failure or close)."""
        for slot in self._slots:
            self._destroy_slot(slot)
        self._slots.clear()
        self._input_buf_handle = c_void_p(None)
        self._bs_handle = c_void_p(None)

    def _find_dll(self) -> str:
        if sys.platform == "win32":
            candidates = [
                "nvEncodeAPI64.dll",
                os.path.join(os.environ.get("WINDIR", r"C:\Windows"),
                             "System32", "nvEncodeAPI64.dll"),
            ]
            prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
            nvidia_video = os.path.join(prog_files, "NVIDIA Corporation",
                                         "NVIDIA Video Codec SDK")
            if os.path.isdir(nvidia_video):
                for root, dirs, files in os.walk(nvidia_video):
                    if "nvEncodeAPI64.dll" in files:
                        candidates.insert(0, os.path.join(root, "nvEncodeAPI64.dll"))
                        break
            for c in candidates:
                if os.path.exists(c) or not os.path.dirname(c):
                    return c
            return "nvEncodeAPI64.dll"
        else:
            candidates = [
                "/usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1",
                "/usr/lib64/libnvidia-encode.so.1",
                "/usr/local/lib/libnvidia-encode.so.1",
                "libnvidia-encode.so.1",
            ]
            for search_dir in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/lib", "/usr/lib"):
                if os.path.isdir(search_dir):
                    for fname in os.listdir(search_dir):
                        if fname.startswith("libnvidia-encode.so"):
                            candidates.insert(0, os.path.join(search_dir, fname))
            for c in candidates:
                if os.path.exists(c):
                    return c
            return "libnvidia-encode.so.1"

    @staticmethod
    def _extract_sps_pps(h264_data: bytes) -> Optional[bytes]:
        """从 H.264 Annex B ES 中提取 SPS+PPS NAL 单元（含起始码）。"""
        result_parts: List[bytes] = []
        pos = 0
        n = len(h264_data)
        while pos < n - 3:
            # 查找 4-byte 起始码 0x00 0x00 0x00 0x01
            if h264_data[pos:pos+4] == b'\x00\x00\x00\x01':
                start = pos
                pos += 4
                if pos >= n:
                    break
                nal_byte = h264_data[pos]
                nal_type = nal_byte & 0x1f
                # 查找下一个起始码（3-byte 或 4-byte）
                end = n
                for j in range(pos, n - 2):
                    if h264_data[j:j+3] == b'\x00\x00\x01' or h264_data[j:j+4] == b'\x00\x00\x00\x01':
                        end = j
                        break
                if nal_type in (7, 8):  # SPS=7, PPS=8
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

    @staticmethod
    def _has_sps_pps(h264_data: bytes) -> bool:
        """检测 H.264 ES 是否已含 SPS/PPS NAL（避免与 NVENC 原生参数集重复注入 avcC）。

        NVENC 原生 IDR 帧已自带 SPS/PPS（repeatSPSPPS 或首帧行为），再手动 prepend
        _cached_sps_pps 会导致 ffmpeg muxer 把两份参数集收进 avcC → numSPS/numPPS=2
        → NVDEC cuvidCreateDecoder 初始化失败。注入前先探测原生是否已含。
        """
        pos = 0
        n = len(h264_data)
        while pos < n - 3:
            if h264_data[pos:pos+4] == b'\x00\x00\x00\x01':
                pos += 4
            elif h264_data[pos:pos+3] == b'\x00\x00\x01':
                pos += 3
            else:
                pos += 1
                continue
            if pos < n and (h264_data[pos] & 0x1f) in (7, 8):
                return True
        return False

    @staticmethod
    def _nal_first_vcl_type(h264_data: bytes) -> Optional[int]:
        """[FIX-AUX-NO-CLEAR] 返回码流块中第一个 VCL NAL 类型（1=P / 5=IDR），
        无 VCL 时返回 None。NVENC LA 预热期会把 SPS/PPS/AUD 作为独立辅助块
        drain 出来，不对应任何已提交帧：VCL 块占帧槽，辅助块仅缓存参数集、
        不占 fi、不进 pairs、不清 pending（tt6 FIX-AUX-NO-CLEAR）。
        """
        pos = 0
        n = len(h264_data)
        while pos < n - 3:
            if h264_data[pos:pos+4] == b'\x00\x00\x00\x01':
                pos += 4
            elif h264_data[pos:pos+3] == b'\x00\x00\x01':
                pos += 3
            else:
                pos += 1
                continue
            if pos >= n:
                break
            nal_type = h264_data[pos] & 0x1f
            if nal_type in (1, 5):
                return nal_type
            end = pos
            while end < n - 3:
                if h264_data[end:end+4] == b'\x00\x00\x00\x01' or \
                        h264_data[end:end+3] == b'\x00\x00\x01':
                    break
                end += 1
            pos = end
        return None

    def _drain_outputs_blocking(self, max_slots: int = None) -> list:
        """[FIX-LA-REDRAIN] 按 _output_slot_idx 顺序循环 blocking LockBitstream
        排空所有已完成 slot，直到遇到 NEED_MORE_INPUT。

        参照 tests/test_nvenc_la_frame_conservation.py 的 _drain_outputs() 验证模式：
        - 从 _output_slot_idx 指向的 slot 开始，循环 blocking Lock
        - 每成功取回一帧则推进 _output_slot_idx
        - 遇到 NEED_MORE_INPUT 时退出循环

        返回 [(frame_index_estimate, output_timestamp, h264_bytes), ...] 列表。
        frame_index 为估算值（基于 output_slot_idx 和 pipeline_depth）。
        output_timestamp 为 NV_ENC_LOCK_BITSTREAM.outputTimeStamp@40 回显的
        提交时 inputTimeStamp（仅用于 [FIX-DRAIN-ORDER-DEFENSE] 相位漂移诊断，
        不作为标签来源 —— 标签一律来自 per-slot FIFO 记账）。
        """
        if max_slots is None:
            max_slots = self._slot_count
        outputs = []
        _LockBS = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
        _UnlockBS = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
        lock_bs_fn = _LockBS(self._func_ptrs[_FUNC_IDX["LockBitstream"]])
        unlock_fn = _UnlockBS(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])

        for _ in range(max_slots):
            slot_idx = self._output_slot_idx % self._slot_count
            bs_handle = self._slots[slot_idx]['bs_buf']

            lock_raw = (c_uint8 * 1544)()
            ctypes.memset(lock_raw, 0, 1544)
            cast(lock_raw, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
            cast(byref(lock_raw, 4), ctypes.POINTER(c_uint32))[0] = 0  # doNotWait=0 (blocking)
            cast(byref(lock_raw, 8), ctypes.POINTER(c_void_p))[0] = bs_handle

            bs_status = lock_bs_fn(self._encoder, lock_raw)
            if bs_status == NV_ENC_ERR_NEED_MORE_INPUT:
                break  # 无更多已完成帧
            if bs_status != NV_ENC_SUCCESS:
                break

            bitstream_size = cast(byref(lock_raw, 36), ctypes.POINTER(c_uint32))[0]
            if bitstream_size == 0:
                unlock_fn(self._encoder, bs_handle)
                break

            _raw_bsptr = cast(byref(lock_raw, 56), ctypes.POINTER(c_void_p))[0]
            bitstream_ptr_val = _raw_bsptr if isinstance(_raw_bsptr, int) else (_raw_bsptr.value or 0)
            out_ts = int(cast(byref(lock_raw, 40), ctypes.POINTER(c_uint64))[0])
            if bitstream_ptr_val:
                buf_type = c_uint8 * bitstream_size
                h264_data = bytes(buf_type.from_address(bitstream_ptr_val))
                # 估算 frame_index: 当前 output_slot_idx 对应的 slot（物理槽位指针）
                est_fi = self._output_slot_idx
                outputs.append((est_fi, out_ts, h264_data))
                unlock_fn(self._encoder, bs_handle)
                self._output_slot_idx += 1
            else:
                # [FIX-DRAIN-COUNTER-DRIFT] size>0 但数据指针为空：未取到码流。
                # 此时推进 _output_slot_idx 会造成"帧计数超前于实际取回"的
                # 相位漂移（记忆文件审核观察），必须 break 且不推进。
                unlock_fn(self._encoder, bs_handle)
                break

        return outputs

    def _reset_output_slot_idx(self, start: int = 0):
        """[FIX-LA-OUTPTR] 重置输出槽位指针（新批次开始时调用）。"""
        self._output_slot_idx = start

    def _apply_drained_entries(self, drained: list, pairs: list) -> None:
        """[FIX-FIFO-DRAIN] 把 _drain_outputs_blocking() 返回的条目按
        per-slot FIFO 队首映射分发到 pairs（对齐 realesrgan_video _apply_drained_entries）。

        与 ESRGAN 已验证设计逐条对齐（同驱动同配置 5099/5099 解码级干净）：
        - drain 条目顺序 = LockBitstream 顺序 = 物理槽位轮转顺序；
        - 每物理 slot 的未决记录为 FIFO deque，队首 = 最旧提交 = 该槽下一次
          必然取回的帧（驱动按提交顺序输出，辅助块也按数据流顺序消费）；
        - 标签 = 队首条目全局 fi - _strm_ts_base（段基线）= 流内帧号。
          绝不读取 outputTimeStamp：ts 回显在辅助块场景下不可信（V2 重关联
          已被 test7 证伪）；
        - 无 VCL 辅助块（独立 SPS/PPS AU）按数据消费：pop 队首条目并缓存参数集，
          不占 fi、不进 pairs —— 与 FIX-AUX-NO-CLEAR（保留 pending 致提交 9
          反压死锁 → 槽位带病复用 → 帧序错位）完全相反；
        - 数据块永不丢弃：有数据必进 pairs（标签 = 队首 fi）；
        - is_idr 用 pending 记录的 force_idr（提交时确定，队列顺序与提交一致，
          不再依赖 NAL 实测 —— _nal_first_vcl_type 已删除）。

        [FIX-AUX-NO-CLEAR]（2026-08-14，test11 复现 tt6 根因）：
        - 无 VCL 辅助块（独立 SPS/PPS/AUD）**必须保留 pending 记录**，仅缓存
          参数集，不占 fi、不进 pairs。若像旧代码那样 pop 队首，则该槽真实
          帧的 VCL 数据仍滞留在驱动输出队列中，pending 却已清空 →
          _ensure_slot_free 误判槽位空闲 → 输入 buffer 未排空即复用 → LA
          延迟消费读到被覆盖的像素 → 数据与标签错位 → 每 9 帧迟一窗的
          frame_num 回退（test11: 549 次、解码 1411/4960）。
        """
        for _est_fi, _out_ts, _h264_data in drained:
            # [FIX-AUX-NO-CLEAR] 先判 VCL 再消费：辅助块不占帧槽。
            if self._nal_first_vcl_type(_h264_data) is None:
                self.__dict__.setdefault('_diag_aux_block', 0)
                self._diag_aux_block += 1
                if self._diag_aux_block <= 5 or self._diag_aux_block % 50 == 0:
                    print(f'[NVENC-Enc] ℹ️ 辅助块 #{self._diag_aux_block} '
                          f'(est={_est_fi} slot={_est_fi % self._slot_count} '
                          f'{len(_h264_data)}B) 仅缓存参数集，不占帧槽', flush=True)
                if self._cached_sps_pps is None:
                    _sps_a = self._extract_sps_pps(_h264_data)
                    if _sps_a:
                        self._cached_sps_pps = _sps_a
                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" % len(_sps_a),
                              flush=True)
                        if self._muxer_ref is not None:
                            try:
                                self._muxer_ref.write_sps_pps(_sps_a)
                                self._sps_pps_injected = True
                            except Exception:
                                pass
                continue
            _drain_slot = _est_fi % self._slot_count
            _entry_deque = self._strm_slot_pending.get(_drain_slot)
            if not _entry_deque:
                # [FIX-DRAIN-ORDER-DEFENSE] 原静默 continue：驱动输出槽与 FIFO 记账
                # 错位（LA 重路由）时无痕错配 → 帧序损坏。计数 + 诊断可见。
                self.__dict__.setdefault('_diag_slot_mismatch', 0)
                self._diag_slot_mismatch += 1
                if self._diag_slot_mismatch <= 5 or self._diag_slot_mismatch % 50 == 0:
                    print(f'[NVENC-Enc] ⚠️ slot 记账错配 #{self._diag_slot_mismatch} '
                          f'(drain_slot={_drain_slot} 无 pending，est_fi={_est_fi} '
                          f'out_slot={self._output_slot_idx})', flush=True)
                continue
            _global_fi, _, _is_idr, _ep_s = _entry_deque[0]  # peek 最旧条目
            _actual_fi = _global_fi - self._strm_ts_base
            # [FIX-DRAIN-ORDER-DEFENSE] 相位漂移检测：VCL 块 outputTimeStamp@40
            # 回显提交时 inputTimeStamp（tt6/tt7 生产双射验证 18/18、34/34）。
            # 与 FIFO 队首 gfi 不一致 = LA 重路由/输入覆盖征兆；标签仍以 FIFO
            # 队首为准（对齐 ESRGAN），但错配不再静默。
            if _out_ts is not None and _out_ts != _global_fi:
                self.__dict__.setdefault('_diag_phase_shift', 0)
                self._diag_phase_shift += 1
                if self._diag_phase_shift <= 5 or self._diag_phase_shift % 50 == 0:
                    print(f'[NVENC-Enc] ⚠️ 相位漂移 #{self._diag_phase_shift} '
                          f'(ts={_out_ts} != fifo_gfi={_global_fi} '
                          f'slot={_drain_slot} est={_est_fi})', flush=True)
            # [FIX-DRAIN-ORDER-DEFENSE] gfi 回退检测：LA 重路由使驱动输出顺序 ≠
            # 提交顺序时，队首 gfi 可能 ≤ 上一已消费 gfi。仍按 FIFO popleft 消费
            # 推进（不丢数据、不卡死），但错配不再静默。
            _last = getattr(self, '_last_drained_gfi', None)
            if _last is not None and _global_fi <= _last:
                self.__dict__.setdefault('_diag_gfi_regress', 0)
                self._diag_gfi_regress += 1
                if self._diag_gfi_regress <= 5 or self._diag_gfi_regress % 50 == 0:
                    print(f'[NVENC-Enc] ⚠️ gfi 回退 #{self._diag_gfi_regress} '
                          f'(gfi={_global_fi} <= last={_last} slot={_drain_slot})', flush=True)
            self._last_drained_gfi = _global_fi
            if _h264_data:
                _out_data = self._apply_sps_pps(_h264_data, _is_idr)
                pairs.append((_actual_fi, _out_data))
                # [FIX-EMPTY-PREV-FILL] 跟踪流内顺序最后一帧（空帧兜底填充基准）
                self._prev_stream_h264 = _out_data
            else:
                self.__dict__.setdefault('_diag_empty', 0)
                self._diag_empty += 1
                if self._diag_empty <= 5 or self._diag_empty % 50 == 0:
                    print(f'[NVENC-Enc] ⚠️ 空帧 #{self._diag_empty} (stream fi={_actual_fi} '
                          f'slot={_drain_slot} fidr={_is_idr} '
                          f'la_buf={getattr(self, "_la_buffered", 0)})', flush=True)
                # [FIX-EMPTY-PREV-FILL] b"" → 前一帧码流兜底（仅"真缺失"帧走此分支：
                # EOS 残留 / 槽位排空超限占位；LA 正常缓冲帧不会产出 b""）。
                pairs.append((_actual_fi, self._prev_stream_h264 or b""))
            _entry_deque.popleft()
            if not _entry_deque:
                del self._strm_slot_pending[_drain_slot]

    def _ensure_slot_free(self, slot_idx: int, pairs: list) -> None:
        """[FIX-ENSURE-SLOT-ROTATION] Drain pending outputs in output-slot
        rotation order before reusing a physical bitstream buffer.

        This mirrors the verified realesrgan_video implementation: repeatedly
        drain via _drain_outputs_blocking(max_slots=1) until the target slot
        per-slot FIFO is empty. Never lock the target slot directly and never
        advance _output_slot_idx manually.
        """
        _guard = 0
        while self._strm_slot_pending.get(slot_idx):
            _drained = self._drain_outputs_blocking(max_slots=1)
            if _drained:
                # [FIX-AUX-NO-CLEAR] 辅助块会被 _apply_drained_entries 跳过
                # （不 pop），但物理轮转指针已推进 —— 循环继续直至目标槽排空。
                _guard = 0
                self._apply_drained_entries(_drained, pairs)
                continue
            _guard += 1
            if _guard > self._slot_count * 4:
                # [FIX-SLOT-DRAIN-TARGET] 轮转探测无法触达目标槽（LA 重路由/
                # 辅助块使目标槽输出晚于其他槽就绪）→ 直接对目标槽自身做
                # blocking LockBitstream（doNotWait=0）。
                _h264_t, _st_t = self._lock_bitstream_blocking(
                    self._slots[slot_idx]['bs_buf'], timeout_ms=2000)
                if _h264_t:
                    self._apply_drained_entries([(slot_idx, None, _h264_t)], pairs)
                    _guard = 0
                    continue
                # [FIX-SLOT-BACKPRESSURE-B] 硬件仍无数据：绝不带 pending 复用，
                # 将目标槽 pending 逐条以空帧占位（prev 填充）消费并推进 FIFO；
                # 宁可写占位也绝不覆盖未取回的码流。
                _dq = self._strm_slot_pending.get(slot_idx)
                if _dq:
                    self.__dict__.setdefault('_diag_slot_drain_fallback', 0)
                    self._diag_slot_drain_fallback += 1
                    if self._diag_slot_drain_fallback <= 5 or \
                            self._diag_slot_drain_fallback % 50 == 0:
                        print(f'[NVENC-Enc] ⚠️ slot={slot_idx} 排空超限 '
                              f'#{self._diag_slot_drain_fallback}：'
                              f'空帧占位兜底（prev 填充）', flush=True)
                    _n_fill = len(_dq)
                    while _dq:
                        _gfi_f, _, _is_idr_f, _ep_s_f = _dq[0]
                        _actual_fi_f = _gfi_f - self._strm_ts_base
                        pairs.append((_actual_fi_f, self._prev_stream_h264 or b""))
                        _dq.popleft()
                    del self._strm_slot_pending[slot_idx]
                    self._output_slot_idx += _n_fill
                break
            continue

    def _lock_bitstream_blocking(self, bs_handle, timeout_ms: int = 500):
        """[FIX-LA-BLOCKING] Blocking LockBitstream (doNotWait=0)。

        用于 NEED_MORE_INPUT 帧的可靠数据取回：CE 可能提前触发，
        用 blocking Lock 等待硬件实际完成编码并写入 bs_buf。
        超时后返回 (b"", status)。

        Returns (h264_data: bytes, status: int)
        """
        import time as _time
        _LockBS = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
        _UnlockBS = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
        lock_bs_fn = _LockBS(self._func_ptrs[_FUNC_IDX["LockBitstream"]])
        unlock_fn = _UnlockBS(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])
        _deadline = _time.monotonic() + timeout_ms / 1000.0

        while _time.monotonic() < _deadline:
            lock_raw = (c_uint8 * 1544)()
            ctypes.memset(lock_raw, 0, 1544)
            cast(lock_raw, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
            cast(byref(lock_raw, 4), ctypes.POINTER(c_uint32))[0] = 0  # doNotWait=0
            cast(byref(lock_raw, 8), ctypes.POINTER(c_void_p))[0] = bs_handle

            bs_status = lock_bs_fn(self._encoder, lock_raw)
            if bs_status == NV_ENC_ERR_NEED_MORE_INPUT:
                # 仍未完成，短暂等待后重试
                _time.sleep(0.001)
                continue
            if bs_status != NV_ENC_SUCCESS:
                return b"", bs_status

            bitstream_size = cast(byref(lock_raw, 36), ctypes.POINTER(c_uint32))[0]
            if bitstream_size > 0:
                _raw_bsptr = cast(byref(lock_raw, 56), ctypes.POINTER(c_void_p))[0]
                bitstream_ptr_val = _raw_bsptr if isinstance(_raw_bsptr, int) else (_raw_bsptr.value or 0)
                if bitstream_ptr_val:
                    buf_type = c_uint8 * bitstream_size
                    h264_data = bytes(buf_type.from_address(bitstream_ptr_val))
                    unlock_fn(self._encoder, bs_handle)
                    return h264_data, bs_status
            unlock_fn(self._encoder, bs_handle)
            _time.sleep(0.001)

        return b"", NV_ENC_ERR_NEED_MORE_INPUT

    def _lock_bitstream_with_retry(self, bs_handle, max_retries: int = 5, backoff_us: int = 1000):
        """[Tier 3-E] 带指数退避重试的 LockBitstream。

        应对 NVENC HW 的 bitstream DMA 竞态。
        [V6442-CONSTQP-FAST] CONSTQP 零空帧 → max_retries=2。
        [FIX-LA-LOCKBS] NEED_MORE_INPUT 立即返回不重试，让调用方正确区分
        "LA 缓冲中无数据"与"瞬时空帧"，避免静默丢失 LA 滞后产出帧。

        Returns (h264_data: bytes, status: int)
        """
        import time as _time
        # [V6442-CONSTQP-FAST] CONSTQP 零空帧 → 减少重试开销
        if self._rate_mode == 'constqp':
            max_retries = min(max_retries, 2)
        _LockBitstreamProto_raw = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
        lock_bs_fn = _LockBitstreamProto_raw(self._func_ptrs[_FUNC_IDX["LockBitstream"]])
        _unlock_fn = _NvEncUnlockBitstreamProto(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])

        for attempt in range(max_retries):
            lock_raw = (c_uint8 * 1544)()
            ctypes.memset(lock_raw, 0, 1544)
            cast(lock_raw, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
            cast(byref(lock_raw, 8), ctypes.POINTER(c_void_p))[0] = bs_handle

            bs_status = lock_bs_fn(self._encoder, lock_raw)
            # [FIX-LA-LOCKBS] NEED_MORE_INPUT 表示 LA 缓冲区中无可用输出，
            # 不是瞬态 DMA 竞态，不应重试。立即返回让调用方正确决策。
            if bs_status == NV_ENC_ERR_NEED_MORE_INPUT:
                return b"", NV_ENC_ERR_NEED_MORE_INPUT
            if bs_status != NV_ENC_SUCCESS:
                return b"", bs_status

            bitstream_size = cast(byref(lock_raw, 36), ctypes.POINTER(c_uint32))[0]
            if bitstream_size > 0:
                _raw_bsptr = cast(byref(lock_raw, 56), ctypes.POINTER(c_void_p))[0]
                bitstream_ptr_val = _raw_bsptr if isinstance(_raw_bsptr, int) else (_raw_bsptr.value or 0)
                if bitstream_ptr_val:
                    buf_type = c_uint8 * bitstream_size
                    h264_data = bytes(buf_type.from_address(bitstream_ptr_val))
                    _unlock_fn(self._encoder, bs_handle)
                    return h264_data, bs_status

            _unlock_fn(self._encoder, bs_handle)
            if attempt < max_retries - 1:
                _time.sleep(backoff_us / 1_000_000.0)
                backoff_us *= 2

        # [FIX-LA-LOCKBS] 重试耗尽且 bitstream_size 始终为 0 → 真正的空帧。
        # 返回 bs_status (SUCCESS) 以区分 NEED_MORE_INPUT。
        return b"", bs_status

    def encode_frames_batch(self, nv12_tensors: list, force_idr_first: bool = False,
                             send_eos: bool = False) -> list:
        """Encode multiple NV12 frames using synchronous per-slot encoding.

        [FIX-LA-CHUNK-STREAM] 本方法现在是 encode_frames_stream() 的兼容封装：
        整个批次视为一个独立流（调用前强制 _stream_begin 重置），返回值与输入
        逐帧对齐；未取回数据的帧（LA 窗口滞留 / 真空帧）以 b"" 占位，语义与
        原实现一致。

        Args:
            send_eos: If True, send EOS after all frames and drain ALL LA-buffered
                      frames into the results array. Required for single-batch-per-segment
                      encoding to achieve frame conservation. Default False (backward compat).

        Returns list of H.264 bytes in the same order as input tensors.
        """
        n_frames = len(nv12_tensors)
        # [FIX-EOS-EMPTY-CHUNK] 空批次快速路径不能无视 send_eos：
        # 当 segment 总帧数恰好被 chunk size 整除时最终块可能为空，
        # 但 send_eos=True 仍需往下走到 EOS 发送+per-slot 阻塞 drain。
        if n_frames == 0 and not send_eos:
            return []

        self._stream_begin(force=True)
        pairs = self.encode_frames_stream(nv12_tensors, force_idr_first=force_idr_first,
                                          send_eos=send_eos)
        results = [b""] * n_frames
        for _fi, _data in pairs:
            if 0 <= _fi < n_frames and _data:
                results[_fi] = _data
        return results

    def _stream_begin(self, force: bool = False):
        """[FIX-LA-CHUNK-STREAM] 开始一个新的流式编码序列（每段一次）。

        重置跨 chunk 持久的 slot 未决表 / 流内帧号。
        [FIX-PIPE4-LA8] IDR 判定基于 fi==0，故 _strm_next_fi 必须每段归零，
        保证新段首帧（fi=0）强制 IDR 重建 DPB（防止跨段污染花屏）。
        EOS 完整排空后 encode_frames_stream 会将 _strm_active 置 False；
        下一段首块前必须重新 begin。
        """
        if force or not self._strm_active:
            self._strm_slot_pending.clear()
            self._strm_next_fi = 0
            self._strm_active = True
            # [FIX-FIFO-DRAIN] 段首基线：drain 标签 = 队首全局 fi - _strm_ts_base
            # （对齐 realesrgan_video 的 chunk_start_global 语义）。_frame_idx
            # 跨段单调（严禁重置），段首提交前记录即可。ts 重关联
            # （[FIX-STREAM-TS-REASSOC] outputTimeStamp 信任机制）已整体删除：
            # 辅助块场景下 ts 回显不可信（test7 seg1 abort 后标签错位），标签
            # 一律来自自身 FIFO 记账。原 [FIX-LA-WINDOW-ROTATION] 状态此前已
            # 移除（失败方案，见 memory/stream-ts-reassociation-fix.md）。
            self._strm_ts_base = self._frame_idx
            # [FIX-OUTSLOT-SEGMENT-RESET] cross-segment reset: EOS leftover empty
            # frames do not advance _output_slot_idx; without reset the next segment
            # drains the wrong physical slot (segment-boundary frame_num regression).
            # Slot assignment is _frame_idx % _slot_count, so align = _frame_idx.
            self._output_slot_idx = self._frame_idx
            # [FIX-DRAIN-ORDER-DEFENSE] / [FIX-EMPTY-PREV-FILL] 每段重置：
            # gfi 单调基准、prev 填充缓存与错配诊断计数。
            self._last_drained_gfi = None
            self._prev_stream_h264 = None
            self._diag_slot_mismatch = 0
            self._diag_gfi_regress = 0
            self._diag_aux_block = 0
            self._diag_phase_shift = 0
            self._diag_slot_drain_fallback = 0

    def _apply_sps_pps(self, h264_data: bytes, is_idr: bool) -> bytes:
        """[SEGMENT-REUSE] SPS/PPS 缓存、预挂与 muxer 预注入（drain 路径统一入口）。

        - IDR 且已有缓存：缓存 SPS+PPS 预挂到帧前（NVENC repeatSPSPPS 不可靠）。
          若原生已含 SPS/PPS（NVENC 自带），跳过 prepend 避免 avcC 重复参数集。
        - 首次见到数据（无论是否 IDR）：提取并缓存；IDR 时额外预注入 muxer。
          首帧本身已含 NVENC 初始化 SPS+PPS，不再预挂（[FIX-SPS-PPS-V3]）。
        """
        if is_idr and self._cached_sps_pps is not None and \
                not self._has_sps_pps(h264_data):
            return self._cached_sps_pps + h264_data
        if self._cached_sps_pps is None:
            _sps = self._extract_sps_pps(h264_data)
            if _sps:
                self._cached_sps_pps = _sps
                print("[NVENCEncoder] Cached SPS+PPS: %d bytes" % len(_sps), flush=True)
                if is_idr and self._muxer_ref is not None:
                    try:
                        self._muxer_ref.write_sps_pps(_sps)
                        self._sps_pps_injected = True
                    except Exception:
                        pass
        return h264_data

    # ── [FIX-FIFO-DRAIN] 已删除的失败方案 ──────────────────────────────
    # [FIX-LA-WINDOW-ROTATION]（9 帧窗口旋转假说 → 双重洗牌）与
    # [FIX-STREAM-TS-REASSOC]（outputTimeStamp 重关联）均被证伪并移除：
    # 标签一律来自 per-slot FIFO 记账（见 _apply_drained_entries）。
    # _RotationBitReader 类保留供备用方向 B（frame_num 码内解析重关联）复用。

    def encode_frames_stream(self, nv12_tensors: list, force_idr_first: bool = False,
                             send_eos: bool = False) -> list:
        """[FIX-LA-CHUNK-STREAM] LA 有界分块流式编码（核心实现）。

        与原 encode_frames_batch() 相同的逐帧同步编码 + 每帧后全局 drain 逻辑，
        但 _slot_pending/_slots_warmed 为实例持久状态，帧以流内全局帧号
        (_strm_next_fi) 标识，因此 lookahead 窗口帧可跨调用正确取回——
        调用方可将一个 segment 分成多个有界 chunk 顺序提交（中间不得发送
        EOS/flush），消除段末一次性编码脉冲及其 pinned host 内存峰值。

        Args:
            send_eos: 段末最后一块必须 True：发送 EOS 并完整排空所有 slot，
                      保证流内全部已提交帧都被取回（帧数守恒）。

        Returns:
            [(fi, h264_bytes), ...]，按取回顺序（即提交顺序）排列。
            仅包含本次调用期间实际取回数据的帧；LA 窗口滞留帧由后续调用或
            EOS 取回。EOS 后仍无数据的帧以 (fi, b"") 占位（调用方做空帧防御）。
        """
        n_frames = len(nv12_tensors)
        if n_frames == 0 and not send_eos:
            return []

        # [FIX-GPU-STAY] 跨线程 CUDA context 保护
        _need_pop = False
        _primary = getattr(self, '_primary_ctx', None)
        if _primary is not None and _primary.value is not None:
            try:
                self._libcuda.cuCtxPushCurrent.restype = c_uint32
                self._libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
                _r_push = self._libcuda.cuCtxPushCurrent(_primary)
                _need_pop = (_r_push == 0)
            except Exception:
                pass

        try:
            with self._lock:
                if self._encoder.value is None:
                    raise RuntimeError("[NVENCEncoder] Encoder not initialized or already closed")

                if not self._strm_active:
                    self._stream_begin()

                # (fi, h264) 取回对；LA 滞留帧不在本次返回中
                pairs = []
                W = self._width
                nv12_h = self._height + self._height // 2

                _LockInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _UnlockInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
                _LockBitstreamProto_raw = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _CU_MEMORYTYPE_HOST   = 1
                _CU_MEMORYTYPE_DEVICE = 2

                for _i in range(n_frames):
                    fi = self._strm_next_fi
                    # [FIX-FIFO-DRAIN] 全局提交序号（= EncodePicture inputTimeStamp，
                    # 段内单调；_frame_idx 在 EncodePicture 后自增，此处先取用）。
                    fi_global = self._frame_idx
                    # [FIX-F0-MISSING] 使用全局 _frame_idx 分配 slot，
                    # 与 encode_frame 首帧编码的 slot 0 不冲突。
                    slot_idx = self._frame_idx % self._slot_count
                    slot = self._slots[slot_idx]
                    # [FIX-PIPE4-LA8] fi==0 only force_idr（单 IDR 替代 per-slot IDR）。
                    # 原 [FIX-PERSLOT-IDR] 以 slot 是否 warm 判定：LA 预热期前帧返回
                    # NEED_MORE_INPUT 使 slot 永不 warm → 首 LA+1 帧全强制 IDR →
                    # 触发 NVENC LA 输出 buffer 重路由错位 → 帧序乱 → 段首花屏。
                    # 与 ce_pipeline 已验证方案（line 2274）对齐。
                    force_idr = force_idr_first and (fi == 0)

                    # [FIX-SLOT-BACKPRESSURE] 复用物理 slot bs_buf 前，强制确保
                    # 上一次占用者的码流已被排空，避免硬件层面覆盖丢帧。
                    # [FIX-FIFO-DRAIN] 对齐 realesrgan_video _ensure_slot_free()：
                    # 排出的帧经 _apply_drained_entries 按 per-slot FIFO 队首消费
                    # （写入 pairs 并 pop 队首），guard 超限仅告警不 abort。
                    self._ensure_slot_free(slot_idx, pairs)

                    # ── LockInputBuffer ──
                    lock_buf = (c_uint8 * 1544)()
                    ctypes.memset(lock_buf, 0, 1544)
                    cast(lock_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_INPUT_BUFFER_VER
                    cast(byref(lock_buf, 8), ctypes.POINTER(c_void_p))[0] = slot['input_buf']

                    lock_addr = self._func_ptrs[_FUNC_IDX["LockInputBuffer"]]
                    s = _LockInputBufferProto(lock_addr)(self._encoder, lock_buf)
                    if s != 0:
                        raise RuntimeError("[NVENCEncoder] LockInputBuffer[%d] failed, code=%d" % (slot_idx, s))

                    _raw_map = cast(byref(lock_buf, 16), ctypes.POINTER(c_void_p))[0]
                    mapped_ptr = _raw_map if isinstance(_raw_map, int) else (_raw_map.value or 0)
                    actual_pitch = cast(byref(lock_buf, 24), ctypes.POINTER(c_uint32))[0]

                    if not mapped_ptr:
                        _UnlockInputBufferProto(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                            self._encoder, slot['input_buf'])
                        raise RuntimeError("[NVENCEncoder] LockInputBuffer[%d] returned NULL mapped ptr" % slot_idx)

                    # ── GPU / host → GPU copy (cuMemcpy2D, pitch-aware) ──
                    # [FIX-LA-ACC-HOST] LA>0 累积模式下 nv12_tensors 可能已被 writer 线程
                    # 搬到 pinned host 内存（见 _writer_loop [FIX-LA-ACC-HOST]），
                    # 此处按来源张量是否仍在 GPU 决定 srcMemoryType。
                    _src_t = nv12_tensors[_i]
                    _src_is_cuda = _src_t.is_cuda
                    _cpy2d = (c_uint8 * 128)()
                    ctypes.memset(_cpy2d, 0, 128)
                    src_ptr = _src_t.data_ptr()
                    if _src_is_cuda:
                        cast(byref(_cpy2d, 16), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
                        cast(byref(_cpy2d, 32), ctypes.POINTER(c_void_p))[0] = c_void_p(src_ptr)  # srcDevice
                    else:
                        cast(byref(_cpy2d, 16), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_HOST
                        cast(byref(_cpy2d, 24), ctypes.POINTER(c_void_p))[0] = c_void_p(src_ptr)  # srcHost
                    cast(byref(_cpy2d, 48), ctypes.POINTER(c_size_t))[0] = W
                    cast(byref(_cpy2d, 72), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
                    cast(byref(_cpy2d, 88), ctypes.POINTER(c_void_p))[0] = c_void_p(mapped_ptr)
                    cast(byref(_cpy2d, 104), ctypes.POINTER(c_size_t))[0] = (
                        actual_pitch if actual_pitch > 0 else W)
                    cast(byref(_cpy2d, 112), ctypes.POINTER(c_size_t))[0] = W
                    cast(byref(_cpy2d, 120), ctypes.POINTER(c_size_t))[0] = nv12_h
                    r = self._copy_into_input_buffer(_cpy2d)  # [FIX-ASYNC-COPY]
                    if r != 0:
                        _UnlockInputBufferProto(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                            self._encoder, slot['input_buf'])
                        raise RuntimeError("[NVENCEncoder] cuMemcpy2D[%d] failed, code=%d" % (slot_idx, r))
                    # [FIX-LA-ACC-HOST] 该帧数据已提交给 NVENC（H2D/D2D 拷贝已同步完成），
                    # 立即释放累积列表里对应元素的引用，降低峰值内存。
                    nv12_tensors[_i] = None

                    # ── UnlockInputBuffer ──
                    _UnlockInputBufferProto(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                        self._encoder, slot['input_buf'])

                    # ── EncodePicture (synchronous, NO completionEvent) ──
                    pic_buf = (c_uint8 * 3360)()
                    ctypes.memset(pic_buf, 0, 3360)
                    cast(pic_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
                    cast(byref(pic_buf, 4), ctypes.POINTER(c_uint32))[0] = W
                    cast(byref(pic_buf, 8), ctypes.POINTER(c_uint32))[0] = self._height
                    cast(byref(pic_buf, 12), ctypes.POINTER(c_uint32))[0] = (
                        actual_pitch if actual_pitch > 0 else W)
                    cast(byref(pic_buf, 24), ctypes.POINTER(c_uint64))[0] = self._frame_idx
                    cast(byref(pic_buf, 40), ctypes.POINTER(c_void_p))[0] = slot['input_buf']
                    cast(byref(pic_buf, 48), ctypes.POINTER(c_void_p))[0] = slot['bs_buf']
                    # [FIX-CONSTQP-FRAME-CE] constqp+la=0: set completionEvent before
                    # EncodePicture（同 encode_frame 已验证模式）。无 CE 的同步 EncodePicture
                    # 在 Tesla T4 上 LockBitstream 时 segfault。
                    _ep_ce = c_void_p(None)
                    if (self._rate_mode, self._la_depth) == ('constqp', 0):
                        try:
                            self._libcuda.cuEventCreate.restype = c_uint32
                            self._libcuda.cuEventCreate.argtypes = [ctypes.POINTER(c_void_p), c_uint32]
                            self._libcuda.cuEventCreate(ctypes.byref(_ep_ce), 0)
                            if _ep_ce.value is not None:
                                cast(byref(pic_buf, 56), ctypes.POINTER(c_void_p))[0] = _ep_ce
                        except Exception:
                            _ep_ce = c_void_p(None)
                    cast(byref(pic_buf, 64), ctypes.POINTER(c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12
                    cast(byref(pic_buf, 68), ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_STRUCT_FRAME
                    if force_idr:
                        cast(byref(pic_buf, 16), ctypes.POINTER(c_uint32))[0] = 0x2

                    encode_picture = _NvEncEncodePictureProto(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                    _ep_status = encode_picture(self._encoder, cast(pic_buf, ctypes.POINTER(_NvEncPicParams)))
                    self._frame_idx += 1
                    self._strm_next_fi += 1

                    # [FIX-CONSTQP-FRAME-CE] sync completionEvent for constqp+la=0
                    if _ep_ce.value is not None:
                        try:
                            self._libcuda.cuEventSynchronize.restype = c_uint32
                            self._libcuda.cuEventSynchronize.argtypes = [c_void_p]
                            self._libcuda.cuEventSynchronize(_ep_ce)
                            self._libcuda.cuEventDestroy.restype = c_uint32
                            self._libcuda.cuEventDestroy.argtypes = [c_void_p]
                            self._libcuda.cuEventDestroy(_ep_ce)
                        except Exception:
                            pass
                        cast(byref(pic_buf, 56), ctypes.POINTER(c_void_p))[0] = c_void_p(None)

                    # [FIX-LA-DRAIN-SYNC] 保存提交上下文，供 drain 映射实际帧索引。
                    # NEED_MORE_INPUT 帧也记录到 pending，后续 drain 时正确取回。
                    # [FIX-FIFO-DRAIN] per-slot FIFO deque（对齐 ESRGAN）：同槽可
                    # 累积多条记录，append 永不覆盖 —— 单元组覆盖会丢失辅助块
                    # 滞留期间的提交记录（test7 seg1 帧序错位根因之一）。
                    _entry = (fi_global, slot['bs_buf'], force_idr, _ep_status)
                    if slot_idx not in self._strm_slot_pending:
                        self._strm_slot_pending[slot_idx] = deque()
                    self._strm_slot_pending[slot_idx].append(_entry)

                    if _ep_status == NV_ENC_ERR_NEED_MORE_INPUT:
                        # LA 缓冲: 编码器需要更多帧填入前向预看窗口才能产出数据。
                        # 此时 bs_buf 中无可用输出，不应做任何 Lock/Unlock 操作（SDK 规范：
                        # NEED_MORE_INPUT 时强制 Lock/Unlock 会静默丢弃延迟输出帧，破坏帧数守恒）。
                        self.__dict__.setdefault('_la_buffered', 0)
                        self._la_buffered += 1
                        if self._la_buffered <= 3 or self._la_buffered == self._la_depth:
                            print(f'[NVENC-Enc] 前向帧预看缓冲中 ({self._la_buffered}/{self._la_depth})',
                                  flush=True)
                        # ★ 关键修复: NEED_MORE_INPUT 后仍需 drain —
                        # 其他 slot 可能在 LA 窗口填满后产生了已完成帧。
                        # 不 continue，继续执行下面的全局 drain 逻辑。
                    elif _ep_status != NV_ENC_SUCCESS:
                        raise RuntimeError("[NVENCEncoder] EncodePicture[%d] failed, code=%d" % (slot_idx, _ep_status))
                    else:
                        pass  # [FIX-PIPE4-LA8] 单 IDR 由 fi==0 控制，无需 per-slot 标记

                    # ── [FIX-LA-DRAIN-SYNC] ★ 全局 drain: 循环 LockBitstream 排空所有已完成 slot ──
                    # [FIX-DRAIN-UNSUBMITTED-SLOT] 只 drain 已提交但未排空的 slot。
                    # CONSTQP+LA=0 下 LockBitstream 未提交的 bitstream buffer 会导致
                    # NVENC driver segfault（不返回 NEED_MORE_INPUT）。
                    _pending_cnt = self._frame_idx - self._output_slot_idx
                    _max_drain = min(_pending_cnt, self._slot_count)
                    _drained = self._drain_outputs_blocking(max_slots=_max_drain) if _max_drain > 0 else []
                    # [FIX-FIFO-DRAIN] 统一经 _apply_drained_entries 消费（FIFO 队首映射）
                    self._apply_drained_entries(_drained, pairs)

                # ── [FIX-LA-DRAIN-SYNC] 最终排空 ──
                if send_eos:
                    # [FIX-LA-EOS-IN-BATCH] send_eos=True: 发送 EOS 后完整排空所有 slot。
                    # SDK 合规：EOS → 按 _output_slot_idx 顺序逐 slot blocking LockBitstream
                    # 直到 NEED_MORE_INPUT，回收全部 LA 滞留帧，保证帧数守恒。
                    # 参照 tests/test_nvenc_la_frame_conservation.py flush_eos()。
                    eos_pic_buf = (c_uint8 * 3360)()
                    ctypes.memset(eos_pic_buf, 0, 3360)
                    cast(eos_pic_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
                    cast(byref(eos_pic_buf, 16), ctypes.POINTER(c_uint32))[0] = 0x8  # EOS flag
                    cast(byref(eos_pic_buf, 40), ctypes.POINTER(c_void_p))[0] = c_void_p(None)
                    cast(byref(eos_pic_buf, 48), ctypes.POINTER(c_void_p))[0] = self._bs_handle
                    _eos_ep = _NvEncEncodePictureProto(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                    _eos_ep(self._encoder, cast(eos_pic_buf, ctypes.POINTER(_NvEncPicParams)))

                    # 按 _output_slot_idx 起始顺序逐 slot blocking LockBitstream 排空
                    _LockBS = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                    _UnlockBS = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
                    _start_slot = self._output_slot_idx % self._slot_count
                    _drain_order = [(_start_slot + i) % self._slot_count
                                    for i in range(self._slot_count)]
                    for _ds in _drain_order:
                        _bs_h = self._slots[_ds]['bs_buf']
                        while True:
                            _lr = (c_uint8 * 1544)()
                            ctypes.memset(_lr, 0, 1544)
                            cast(_lr, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
                            cast(byref(_lr, 4), ctypes.POINTER(c_uint32))[0] = 0  # blocking
                            cast(byref(_lr, 8), ctypes.POINTER(c_void_p))[0] = _bs_h
                            _bs_s = _LockBS(self._func_ptrs[_FUNC_IDX["LockBitstream"]])(self._encoder, _lr)
                            if _bs_s == NV_ENC_ERR_NEED_MORE_INPUT:
                                break
                            if _bs_s != NV_ENC_SUCCESS:
                                break
                            _bs_size = cast(byref(_lr, 36), ctypes.POINTER(c_uint32))[0]
                            if _bs_size == 0:
                                _UnlockBS(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(self._encoder, _bs_h)
                                break
                            _raw_ptr = cast(byref(_lr, 56), ctypes.POINTER(c_void_p))[0]
                            _ptr_val = _raw_ptr if isinstance(_raw_ptr, int) else (_raw_ptr.value or 0)
                            if _ptr_val:
                                _buf = c_uint8 * _bs_size
                                _eos_data = bytes(_buf.from_address(_ptr_val))
                                # [FIX-AUX-NO-CLEAR] EOS 排空同样先判 VCL：
                                # 无 VCL 辅助块仅缓存参数集，不 pop pending、
                                # 不写 pairs、不推进帧计数（保留队首供真实帧）。
                                if self._nal_first_vcl_type(_eos_data) is None:
                                    self.__dict__.setdefault('_diag_aux_block', 0)
                                    self._diag_aux_block += 1
                                    if self._diag_aux_block <= 5 or \
                                            self._diag_aux_block % 50 == 0:
                                        print(f'[NVENC-Enc] ℹ️ EOS 辅助块 '
                                              f'#{self._diag_aux_block} '
                                              f'(slot={_ds}) 跳过帧记账', flush=True)
                                    if self._cached_sps_pps is None:
                                        _sps_e = self._extract_sps_pps(_eos_data)
                                        if _sps_e:
                                            self._cached_sps_pps = _sps_e
                                            print("[NVENCEncoder] Cached SPS+PPS: %d bytes"
                                                  % len(_sps_e), flush=True)
                                    _UnlockBS(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(self._encoder, _bs_h)
                                    continue
                                # [FIX-FIFO-DRAIN] EOS 排空同样按 per-slot FIFO 队首消费
                                # （物理 slot = _ds；while True 同槽连续取回多帧时，
                                # 队首即该槽下一次必取回的帧，与驱动输出顺序一致）。
                                _entry_deque = self._strm_slot_pending.get(_ds)
                                if _entry_deque:
                                    _gfi_e, _, _idr_e, _ = _entry_deque[0]
                                    _eos_out = self._apply_sps_pps(_eos_data, _idr_e)
                                    pairs.append((_gfi_e - self._strm_ts_base, _eos_out))
                                    # [FIX-EMPTY-PREV-FILL] advance prev cache so later
                                    # leftover empty placeholders use the correct prev frame.
                                    if _eos_out:
                                        self._prev_stream_h264 = _eos_out
                                    _entry_deque.popleft()
                                    if not _entry_deque:
                                        del self._strm_slot_pending[_ds]
                                    self._output_slot_idx += 1
                                else:
                                    # [FIX-DRAIN-ORDER-DEFENSE] slot has data but FIFO
                                    # empty: misrouted (LA reorder). Do not advance
                                    # _output_slot_idx, to avoid amplifying misalignment.
                                    self.__dict__.setdefault('_diag_slot_mismatch', 0)
                                    self._diag_slot_mismatch += 1
                                    if self._diag_slot_mismatch <= 5 or self._diag_slot_mismatch % 50 == 0:
                                        print('[NVENC-Enc] WARN EOS slot accounting mismatch #%d (slot=%d no pending)'
                                              % (self._diag_slot_mismatch, _ds), flush=True)
                            _UnlockBS(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(self._encoder, _bs_h)
                    # EOS 已发送 + 全槽排空，残留未决为真正的空帧 → (fi, b"") 占位
                    # [FIX-EOS-LEFTOVER-G-NAMEERROR] _g → _ent[0]（4 元组首元素 gfi）
                    _leftover_fis = sorted(
                        _ent[0] - self._strm_ts_base
                        for _dq in self._strm_slot_pending.values()
                        for _ent in _dq)
                    for _fi_lo in _leftover_fis:
                        self.__dict__.setdefault('_diag_empty', 0)
                        self._diag_empty += 1
                        if self._diag_empty <= 5 or self._diag_empty % 50 == 0:
                            print(f'[NVENC-Enc] ⚠️ EOS 后仍无数据帧 (stream fi={_fi_lo})，'
                                  f'记为空帧 #{self._diag_empty}', flush=True)
                        # [FIX-EMPTY-PREV-FILL] EOS 残留占位用前一帧码流兜底
                        pairs.append((_fi_lo, self._prev_stream_h264 or b""))
                    # [FIX-ENSURE-SLOT-ROTATION] EOS output order is
                    # defined by _drain_outputs_blocking rotation; do not
                    # re-sort H.264 elementary stream.
                    self._strm_slot_pending.clear()
                    self._strm_active = False
                else:
                    # 非 EOS：最终尝试排空（LA 窗口帧可能尚未就绪，由后续 chunk 取回）
                    # [FIX-DRAIN-UNSUBMITTED-SLOT] 限制只 drain 剩余未排空的帧
                    _final_pending = self._frame_idx - self._output_slot_idx
                    _final_max = min(_final_pending, self._slot_count)
                    _drained_final = self._drain_outputs_blocking(
                        max_slots=_final_max) if _final_max > 0 else []
                    # [FIX-FIFO-DRAIN] 统一经 _apply_drained_entries 消费（FIFO 队首映射）
                    self._apply_drained_entries(_drained_final, pairs)

                return pairs
        finally:
            if _need_pop:
                try:
                    _ctx_out = c_void_p()
                    self._libcuda.cuCtxPopCurrent.restype = c_uint32
                    self._libcuda.cuCtxPopCurrent.argtypes = [ctypes.POINTER(c_void_p)]
                    self._libcuda.cuCtxPopCurrent(ctypes.byref(_ctx_out))
                except Exception:
                    pass

    def encode_frames_batch_ce_pipeline(self, nv12_tensors: list, force_idr_first: bool = False) -> list:
        """[PHASE4-CE-PIPELINE] 带 per-frame CE 的异步流水线编码。

        EncodePicture 提交时附带 per-frame CUDA completion event，立即返回；
        LockBitstream 延迟到 slot 下次轮转时执行（pipeline_depth 帧后），此时 CE
        已触发、NVENC 硬件已完成编码 → LockBitstream 立即拿到数据，消除同步阻塞。

        Phase 1 (Harvest): slot 重用时 harvest 上一轮的 pending CE + LockBitstream
        Phase 2 (Submit):  LockInputBuffer → cuMemcpy2D → EncodePicture + 新建 CE
        Phase 3 (Drain):   批次结束 drain 所有 pending slots

        Returns list where: bytes = valid H.264, b"" = LA buffering frame, None = empty.
        GPU 验证 (T4, 720×576, VBR_HQ, pipe=4, LA=0): 523 FPS vs sync-batch 375 FPS (+39.5%)。
        """
        n_frames = len(nv12_tensors)
        if n_frames == 0:
            return []

        _need_pop = False
        _primary = getattr(self, '_primary_ctx', None)
        if _primary is not None and _primary.value is not None:
            try:
                self._libcuda.cuCtxPushCurrent.restype = c_uint32
                self._libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
                _r_push = self._libcuda.cuCtxPushCurrent(_primary)
                _need_pop = (_r_push == 0)
            except Exception:
                pass

        try:
            with self._lock:
                if self._encoder.value is None:
                    raise RuntimeError("[NVENCEncoder] Encoder not initialized or already closed")

                pd = self._slot_count
                W = self._width
                nv12_h = self._height + self._height // 2
                results = [None] * n_frames

                # [FIX-LA-OUTPTR] 每批次开始时重置输出槽位指针
                self._reset_output_slot_idx(0)

                # Per-slot pending state: (ce_handle, frame_index, ep_status, force_idr)
                _slot_pending = [None] * pd
                # [FIX-PIPE4-LA8] fi==0 only force_idr (单 IDR 替代 per-slot ×4 IDR)
                _LockIB = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _UnlockIB = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
                _CUDA_DEV = 2

                for fi in range(n_frames):
                    # [FIX-F0-MISSING] 使用全局 _frame_idx 分配 slot，
                    # 与 encode_frame 首帧编码的 slot 0 不冲突。
                    slot_idx = self._frame_idx % pd
                    slot = self._slots[slot_idx]
                    # [FIX-PIPE4-LA8] fi==0 only force_idr (单 IDR 替代 per-slot ×4 IDR)
                    force_idr = force_idr_first and (fi == 0)

                    # ═══════════════════════════════════════════════
                    # Phase 1: Harvest pending frame from this slot (if any)
                    # ═══════════════════════════════════════════════
                    if _slot_pending[slot_idx] is not None:
                        _prev_ce, _prev_fi, _prev_ep_s, _prev_idr, _prev_bs = _slot_pending[slot_idx]
                        # Wait for CE → data is now in _prev_bs (the bs_buf this frame was submitted to)
                        if _prev_ce.value is not None:
                            self._libcuda.cuEventSynchronize.restype = c_uint32
                            self._libcuda.cuEventSynchronize.argtypes = [c_void_p]
                            self._libcuda.cuEventSynchronize(_prev_ce)
                            self._libcuda.cuEventDestroy.restype = c_uint32
                            self._libcuda.cuEventDestroy.argtypes = [c_void_p]
                            self._libcuda.cuEventDestroy(_prev_ce)
                        # harvest from stored bs_buf (not current slot's bs_buf)
                        h264_data, bs_status = self._lock_bitstream_with_retry(_prev_bs)
                        if h264_data:
                            # [SEGMENT-REUSE] SPS/PPS caching and pre-pending
                            if _prev_idr and self._cached_sps_pps is not None and \
                                    not self._has_sps_pps(h264_data):
                                h264_data = self._cached_sps_pps + h264_data
                            elif _prev_idr and self._cached_sps_pps is None and h264_data:
                                self._cached_sps_pps = self._extract_sps_pps(h264_data)
                                if self._cached_sps_pps:
                                    print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                          len(self._cached_sps_pps), flush=True)
                                    # [FIX-SPS-PPS-V3] 首帧已含 NVENC 初始化 SPS+PPS, 不 prepend, 仅预注入 muxer
                                    if self._muxer_ref is not None:
                                        try:
                                            self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                            self._sps_pps_injected = True
                                        except Exception:
                                            pass
                            elif not _prev_idr and self._cached_sps_pps is None and h264_data:
                                self._cached_sps_pps = self._extract_sps_pps(h264_data)
                                if self._cached_sps_pps:
                                    print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                          len(self._cached_sps_pps), flush=True)
                            results[_prev_fi] = h264_data
                            # [FIX-LA-OUTPTR] harvest 成功 → 推进输出指针
                            self._output_slot_idx += 1
                        elif _prev_ep_s == NV_ENC_ERR_NEED_MORE_INPUT:
                            # [FIX-LA-PENDING] LA 帧可能尚未完成编码（CE 在帧入队
                            # 时即触发，但编码尚未完成）。使用 blocking LockBitstream
                            # 再试一次取回实际数据，避免将有效帧误标为 b""。
                            # [FIX-LA-BLKRETRY-ALWAYS] 无论非阻塞 LockBitstream
                            # 返回什么 bs_status，NEED_MORE_INPUT 帧都尝试
                            # blocking LockBitstream。当 bs_status==SUCCESS 但
                            # bitstream_size==0 (LA 尚未产出数据)时，旧逻辑跳过
                            # blocking retry 直接丢帧 → 帧数据被 flush 写入
                            # 流末尾而非正确位置 → GOP 参考帧错乱 → 花屏/PPS错误。
                            _h264_blk, _ = self._lock_bitstream_blocking(_prev_bs, timeout_ms=5000)
                            if _h264_blk:
                                # [FIX-LA-SPS-BLKRETRY] blocking retry 取回的
                                # 首个 IDR 帧必须同步缓存 SPS+PPS，否则 _loop()
                                # 写入 muxer 时无 extradata → FFmpeg "non-existing PPS 0"
                                if _prev_idr and self._cached_sps_pps is None:
                                    self._cached_sps_pps = self._extract_sps_pps(_h264_blk)
                                    if self._cached_sps_pps:
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                              len(self._cached_sps_pps), flush=True)
                                        if self._muxer_ref is not None:
                                            try:
                                                self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                                self._sps_pps_injected = True
                                            except Exception:
                                                pass
                                elif self._cached_sps_pps is None:
                                    self._cached_sps_pps = self._extract_sps_pps(_h264_blk)
                                    if self._cached_sps_pps:
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                              len(self._cached_sps_pps), flush=True)
                                results[_prev_fi] = _h264_blk
                                # [FIX-LA-OUTPTR] blocking 取回成功 → 推进输出指针
                                self._output_slot_idx += 1
                            else:
                                results[_prev_fi] = b""
                            self.__dict__.setdefault('_diag_empty', 0)
                            self._diag_empty += 1
                            _la_buf = getattr(self, '_la_buffered', 0)
                            if self._diag_empty <= 5 or self._diag_empty % 50 == 0:
                                print(f'[NVENC-Enc] ⚠️ 空帧 #{self._diag_empty} (ce-pipe '
                                      f'fi={_prev_fi} slot={slot_idx} ep_s={_prev_ep_s} '
                                      f'bs_s={bs_status} la_buf={_la_buf})', flush=True)
                        _slot_pending[slot_idx] = None

                    # ═══════════════════════════════════════════════
                    # Phase 2: Submit new frame with per-frame CE
                    # ═══════════════════════════════════════════════
                    # LockInputBuffer
                    lock_buf = (c_uint8 * 1544)()
                    ctypes.memset(lock_buf, 0, 1544)
                    cast(lock_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_INPUT_BUFFER_VER
                    cast(byref(lock_buf, 8), ctypes.POINTER(c_void_p))[0] = slot['input_buf']

                    lock_addr = self._func_ptrs[_FUNC_IDX["LockInputBuffer"]]
                    s = _LockIB(lock_addr)(self._encoder, lock_buf)
                    if s != 0:
                        raise RuntimeError("[NVENCEncoder] LockInputBuffer[%d] failed, code=%d" %
                                           (slot_idx, s))

                    _raw_map = cast(byref(lock_buf, 16), ctypes.POINTER(c_void_p))[0]
                    mapped_ptr = _raw_map if isinstance(_raw_map, int) else (_raw_map.value or 0)
                    actual_pitch = cast(byref(lock_buf, 24), ctypes.POINTER(c_uint32))[0]

                    if not mapped_ptr:
                        _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                            self._encoder, slot['input_buf'])
                        raise RuntimeError("[NVENCEncoder] LockInputBuffer[%d] returned NULL mapped ptr" %
                                           slot_idx)

                    # ── GPU→GPU copy (cuMemcpy2D, pitch-aware) ──
                    _cpy2d = (c_uint8 * 128)()
                    ctypes.memset(_cpy2d, 0, 128)
                    src_ptr = nv12_tensors[fi].data_ptr()
                    cast(byref(_cpy2d, 16), ctypes.POINTER(c_uint32))[0] = _CUDA_DEV
                    cast(byref(_cpy2d, 32), ctypes.POINTER(c_void_p))[0] = c_void_p(src_ptr)
                    cast(byref(_cpy2d, 48), ctypes.POINTER(c_size_t))[0] = W
                    cast(byref(_cpy2d, 72), ctypes.POINTER(c_uint32))[0] = _CUDA_DEV
                    cast(byref(_cpy2d, 88), ctypes.POINTER(c_void_p))[0] = c_void_p(mapped_ptr)
                    cast(byref(_cpy2d, 104), ctypes.POINTER(c_size_t))[0] = (
                        actual_pitch if actual_pitch > 0 else W)
                    cast(byref(_cpy2d, 112), ctypes.POINTER(c_size_t))[0] = W
                    cast(byref(_cpy2d, 120), ctypes.POINTER(c_size_t))[0] = nv12_h
                    r = self._copy_into_input_buffer(_cpy2d)  # [FIX-ASYNC-COPY]
                    if r != 0:
                        _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                            self._encoder, slot['input_buf'])
                        raise RuntimeError("[NVENCEncoder] cuMemcpy2D[%d] failed, code=%d" %
                                           (slot_idx, r))

                    # ── UnlockInputBuffer ──
                    _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                        self._encoder, slot['input_buf'])

                    # [FIX-LA-SYNC] LA>0 时跳过 CE 创建（同步编码）
                    _ce = c_void_p(None)
                    if self._la_depth == 0:
                        self._libcuda.cuEventCreate.restype = c_uint32
                        self._libcuda.cuEventCreate.argtypes = [ctypes.POINTER(c_void_p), c_uint32]
                        self._libcuda.cuEventCreate(ctypes.byref(_ce), 0)

                    # ── EncodePicture with per-frame completionEvent ──
                    pic_buf = (c_uint8 * 3360)()
                    ctypes.memset(pic_buf, 0, 3360)
                    cast(pic_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
                    cast(byref(pic_buf, 4), ctypes.POINTER(c_uint32))[0] = W
                    cast(byref(pic_buf, 8), ctypes.POINTER(c_uint32))[0] = self._height
                    cast(byref(pic_buf, 12), ctypes.POINTER(c_uint32))[0] = (
                        actual_pitch if actual_pitch > 0 else W)
                    cast(byref(pic_buf, 24), ctypes.POINTER(c_uint64))[0] = self._frame_idx
                    cast(byref(pic_buf, 40), ctypes.POINTER(c_void_p))[0] = slot['input_buf']
                    cast(byref(pic_buf, 48), ctypes.POINTER(c_void_p))[0] = slot['bs_buf']
                    # [FIX-LA-SYNC] LA>0 时禁用 CE (completionEvent)，
                    # 使用同步 EncodePicture 匹配测试脚本的验证模式。
                    # CE 在 LA 场景下提前触发但数据未就绪 → 帧丢失。
                    if self._la_depth == 0:
                        cast(ctypes.byref(pic_buf, 56), ctypes.POINTER(c_void_p))[0] = _ce
                    cast(byref(pic_buf, 64), ctypes.POINTER(c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12
                    cast(byref(pic_buf, 68), ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_STRUCT_FRAME
                    if force_idr:
                        cast(byref(pic_buf, 16), ctypes.POINTER(c_uint32))[0] = 0x2

                    encode_picture = _NvEncEncodePictureProto(
                        self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                    _ep_status = encode_picture(self._encoder,
                                                cast(pic_buf, ctypes.POINTER(_NvEncPicParams)))
                    self._frame_idx += 1

                    if _ep_status != NV_ENC_SUCCESS and _ep_status != NV_ENC_ERR_NEED_MORE_INPUT:
                        raise RuntimeError("[NVENCEncoder] EncodePicture[%d] failed, code=%d" %
                                           (slot_idx, _ep_status))

                    # Handle LA buffering: encoder needs more frames before producing output.
                    # [FIX-LA-NEEDMORE-PENDING] NEED_MORE_INPUT frames are also recorded in
                    # _slot_pending so Phase 1/3 correctly harvest them (returning b"").
                    # This aligns with test_nvenc_completion_event_v4.py NVENCEncoderMode5.
                    if _ep_status == NV_ENC_ERR_NEED_MORE_INPUT:
                        results[fi] = b""
                        self.__dict__.setdefault('_la_buffered', 0)
                        self._la_buffered += 1
                        if self._la_buffered <= 3 or self._la_buffered == self._la_depth:
                            print(f'[NVENC-Enc] 前向帧预看缓冲中 ({self._la_buffered}/{self._la_depth})',
                                  flush=True)

                    # Save pending state for harvest on next slot rotation.
                    # Tuple: (ce_handle, frame_index, ep_status, is_idr, bs_buf)
                    _slot_pending[slot_idx] = (_ce, fi, _ep_status, force_idr, slot['bs_buf'])

                    # [FIX-LA-INLINE-DRAIN] SDK 合规：每送入一帧后立即排空所有
                    # 已就绪的输出（LA>0 时帧产出可能早于 CE 轮转）。按
                    # _output_slot_idx 顺序循环 blocking LockBitstream，
                    # 收集 LA 管道中已完成但尚未被 Phase 1 harvest 的帧。
                    # 参照 tests/test_nvenc_la_frame_conservation.py _drain_outputs()。
                    if self._la_depth > 0:
                        _inlined = self._drain_outputs_blocking()
                        for _est_fi, _out_ts_i, _h264_data in _inlined:
                            # [FIX-LA-INDEX] 通过 _slot_pending 获取实际帧索引，
                            # 替代 _output_slot_idx 计数器估算。当 Phase 1 harvest
                            # 已消费部分 slot 时 _output_slot_idx≠实际帧序 → 花屏。
                            _drain_slot = _est_fi % pd
                            _pending = _slot_pending[_drain_slot]
                            if _pending is None:
                                continue
                            _, _actual_fi, _, _actual_idr, _ = _pending
                            # [FIX-LA-INLINE-OVERWRITE] LA 缓冲帧被预设为 b""
                            if _actual_fi < n_frames and (results[_actual_fi] is None or results[_actual_fi] == b""):
                                # [FIX-LA-SPS-PPS] inline drain 必须同步做 SPS/PPS 缓存
                                if _actual_idr and self._cached_sps_pps is not None and \
                                        not self._has_sps_pps(_h264_data):
                                    _h264_data = self._cached_sps_pps + _h264_data
                                elif _actual_idr and self._cached_sps_pps is None:
                                    self._cached_sps_pps = self._extract_sps_pps(_h264_data)
                                    if self._cached_sps_pps:
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                              len(self._cached_sps_pps), flush=True)
                                        if self._muxer_ref is not None:
                                            try:
                                                self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                                self._sps_pps_injected = True
                                            except Exception:
                                                pass
                                elif self._cached_sps_pps is None:
                                    self._cached_sps_pps = self._extract_sps_pps(_h264_data)
                                    if self._cached_sps_pps:
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                              len(self._cached_sps_pps), flush=True)
                                results[_actual_fi] = _h264_data
                                # [FIX-LA-INLINE-CLEAR] 清除 _slot_pending
                                _slot_pending[_drain_slot] = None

                # [FIX-LA-BATCH-ROUTE] LA>0 时 _loop() 已路由到 encode_frames_batch()
                # （同步全局 drain），ce_pipeline 仅在 LA=0 时被调用。
                # LA=0 下无前向预看缓冲，Phase 3 drain 足够排空所有 pending slot。
                # EOS 由 flush() 在 encoder 生命周期结束时统一发送。

                # ═══════════════════════════════════════════════
                # Phase 3: Drain remaining pending slots
                # ═══════════════════════════════════════════════
                for slot_idx in range(pd):
                    if _slot_pending[slot_idx] is not None:
                        _pending_ce, _pending_fi, _pending_ep_s, _pending_idr, _pending_bs = \
                            _slot_pending[slot_idx]
                        if _pending_ce.value is not None:
                            self._libcuda.cuEventSynchronize.restype = c_uint32
                            self._libcuda.cuEventSynchronize.argtypes = [c_void_p]
                            self._libcuda.cuEventSynchronize(_pending_ce)
                            self._libcuda.cuEventDestroy.restype = c_uint32
                            self._libcuda.cuEventDestroy.argtypes = [c_void_p]
                            self._libcuda.cuEventDestroy(_pending_ce)
                        h264_data, bs_status = self._lock_bitstream_with_retry(_pending_bs)
                        if h264_data:
                            # [SEGMENT-REUSE] SPS/PPS caching
                            if _pending_idr and self._cached_sps_pps is not None and \
                                    not self._has_sps_pps(h264_data):
                                h264_data = self._cached_sps_pps + h264_data
                            elif _pending_idr and self._cached_sps_pps is None and h264_data:
                                self._cached_sps_pps = self._extract_sps_pps(h264_data)
                                if self._cached_sps_pps:
                                    print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                          len(self._cached_sps_pps), flush=True)
                                    # [FIX-SPS-PPS-V3] 首帧已含 NVENC 初始化 SPS+PPS, 不 prepend, 仅预注入 muxer
                                    if self._muxer_ref is not None:
                                        try:
                                            self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                            self._sps_pps_injected = True
                                        except Exception:
                                            pass
                            elif not _pending_idr and self._cached_sps_pps is None and h264_data:
                                self._cached_sps_pps = self._extract_sps_pps(h264_data)
                                if self._cached_sps_pps:
                                    print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                          len(self._cached_sps_pps), flush=True)
                            results[_pending_fi] = h264_data
                            # [FIX-LA-OUTPTR] Phase 3 drain 成功 → 推进输出指针
                            self._output_slot_idx += 1
                        elif _pending_ep_s == NV_ENC_ERR_NEED_MORE_INPUT:
                            # [FIX-LA-BLKRETRY-ALWAYS] 同 Phase 1：无论 bs_status
                            # 为何值都尝试 blocking LockBitstream，防止 SUCCESS+空
                            # 数据被静默丢弃 → flush 写出错位帧破坏 GOP。
                            _h264_blk, _ = self._lock_bitstream_blocking(_pending_bs, timeout_ms=5000)
                            if _h264_blk:
                                # [FIX-LA-SPS-BLKRETRY-PH3] Phase 3 blocking retry
                                # 取回的 IDR 帧同样需要 SPS/PPS 缓存
                                if _pending_idr and self._cached_sps_pps is None:
                                    self._cached_sps_pps = self._extract_sps_pps(_h264_blk)
                                    if self._cached_sps_pps:
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                              len(self._cached_sps_pps), flush=True)
                                        if self._muxer_ref is not None:
                                            try:
                                                self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                                self._sps_pps_injected = True
                                            except Exception:
                                                pass
                                elif self._cached_sps_pps is None:
                                    self._cached_sps_pps = self._extract_sps_pps(_h264_blk)
                                    if self._cached_sps_pps:
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                              len(self._cached_sps_pps), flush=True)
                                results[_pending_fi] = _h264_blk
                                # [FIX-LA-OUTPTR] blocking 取回成功 → 推进输出指针
                                self._output_slot_idx += 1
                            else:
                                results[_pending_fi] = b""
                        # else: None stays — real empty frame
                        _slot_pending[slot_idx] = None

                # [FIX-LA-REDRAIN] 二次排空安全网：CE 可能因 LA 提前触发
                # 而导致 NEED_MORE_INPUT 帧在 Phase 1/3 被误标为 b""（CE
                # 在帧入队时即触发，但编码尚未完成）。按 _output_slot_idx
                # 顺序循环 blocking LockBitstream，收集所有实际已完成但
                # 被遗漏的帧，覆写 results 中的 b""/None 占位符。
                # 参照 tests/test_nvenc_la_frame_conservation.py 的
                # _drain_outputs() 验证模式。
                if self._la_depth > 0:
                    _redrained = self._drain_outputs_blocking()
                    if _redrained:
                        _recovered = 0
                        for _est_fi, _out_ts_r, _h264_data in _redrained:
                            if _est_fi < n_frames and (results[_est_fi] is None or results[_est_fi] == b""):
                                # [FIX-LA-SPS-REDRAIN] 二次排空安全网取回的帧也需
                                # SPS/PPS 缓存，作为所有路径均已失败的最终兜底
                                if _est_fi == 0 and self._cached_sps_pps is None:
                                    self._cached_sps_pps = self._extract_sps_pps(_h264_data)
                                    if self._cached_sps_pps:
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                              len(self._cached_sps_pps), flush=True)
                                        if self._muxer_ref is not None:
                                            try:
                                                self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                                self._sps_pps_injected = True
                                            except Exception:
                                                pass
                                elif self._cached_sps_pps is None:
                                    self._cached_sps_pps = self._extract_sps_pps(_h264_data)
                                    if self._cached_sps_pps:
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                              len(self._cached_sps_pps), flush=True)
                                results[_est_fi] = _h264_data
                                _recovered += 1
                        if _recovered > 0:
                            print(f'[NVENC-Enc] [FIX-LA-REDRAIN] 二次排空回收 {_recovered} '
                                  f'帧 (LA={self._la_depth}, pd={pd})', flush=True)

                return results
        finally:
            if _need_pop:
                try:
                    _ctx_out = c_void_p()
                    self._libcuda.cuCtxPopCurrent.restype = c_uint32
                    self._libcuda.cuCtxPopCurrent.argtypes = [ctypes.POINTER(c_void_p)]
                    self._libcuda.cuCtxPopCurrent(ctypes.byref(_ctx_out))
                except Exception:
                    pass

    def encode_frame(self, nv12_gpu_tensor, force_idr: bool = False) -> bytes:
        """Encode one NV12 GPU tensor → H.264 ES bytes (synchronous, backward compat).

        For single-frame encoding (first frame, flush, etc.). Uses slot 0 with
        blocking EncodePicture + LockBitstream — same behavior as pre-multi-slot.

        For batch encoding, use encode_frames_batch() which pipelines across all slots.
        """
        import torch

        # [FIX-GPU-STAY] 跨线程 CUDA context 保护: 确保 Writer 线程调用时
        # NVENC session 的 primary context 已 set current。与 __init__ 中
        # cuCtxPushCurrent(self._saved_ctx) 配对。
        # [FIX-ENC-CTX] _need_pop 依赖 cuCtxPushCurrent 实际返回值。
        _need_pop = False
        _primary = getattr(self, '_primary_ctx', None)
        if _primary is not None and _primary.value is not None:
            try:
                self._libcuda.cuCtxPushCurrent.restype = c_uint32
                self._libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
                _r_push = self._libcuda.cuCtxPushCurrent(_primary)
                _need_pop = (_r_push == 0)  # ✅ [FIX-ENC-CTX]
            except Exception:
                pass

        try:
            with self._lock:
                if self._encoder.value is None:
                    raise RuntimeError("[NVENCEncoder] Encoder not initialized or already closed")
                if self._input_buf_handle.value is None or self._bs_handle.value is None:
                    raise RuntimeError("[NVENCEncoder] Buffers not initialized")

                assert nv12_gpu_tensor.is_cuda
                assert nv12_gpu_tensor.dtype == torch.uint8
                assert nv12_gpu_tensor.is_contiguous()

                nv12_h = self._height + self._height // 2
                W = self._width

                # ── LockInputBuffer → get mapped pointer ──
                _LockInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _UnlockInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)

                lock_buf = (c_uint8 * 1544)()
                ctypes.memset(lock_buf, 0, 1544)
                cast(lock_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_INPUT_BUFFER_VER  # version@0
                cast(byref(lock_buf, 8), ctypes.POINTER(c_void_p))[0] = self._input_buf_handle  # inputBuffer@8

                lock_addr = self._func_ptrs[_FUNC_IDX["LockInputBuffer"]]
                s = _LockInputBufferProto(lock_addr)(self._encoder, lock_buf)
                if s != 0:
                    raise RuntimeError("[NVENCEncoder] LockInputBuffer failed, code=%d" % s)

                _raw_map = cast(byref(lock_buf, 16), ctypes.POINTER(c_void_p))[0]  # bufferDataPtr@16
                mapped_ptr = _raw_map if isinstance(_raw_map, int) else (_raw_map.value or 0)
                actual_pitch = cast(byref(lock_buf, 24), ctypes.POINTER(c_uint32))[0]  # pitch@24

                if not mapped_ptr:
                    _UnlockInputBufferProto(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                        self._encoder, self._input_buf_handle)
                    raise RuntimeError("[NVENCEncoder] LockInputBuffer returned NULL mapped ptr")

                # ── GPU / host → GPU copy (cuMemcpy2D, pitch-aware) ──
                # [FIX-LA-ACC-HOST] LA>0 累积模式下 nv12_gpu_tensor 可能已被 writer 线程
                # 搬到 pinned host 内存，此处按来源张量是否仍在 GPU 决定 srcMemoryType。
                # CUDA_MEMCPY2D (128B): srcX@0,srcY@8,srcMemType@16,srcHost@24,
                #   srcDevice@32,srcArray@40,srcPitch@48,
                #   dstX@56,dstY@64,dstMemType@72,dstHost@80,
                #   dstDevice@88,dstArray@96,dstPitch@104,
                #   WidthInBytes@112,Height@120
                _CU_MEMORYTYPE_HOST   = 1
                _CU_MEMORYTYPE_DEVICE = 2
                _cpy2d = (c_uint8 * 128)()
                ctypes.memset(_cpy2d, 0, 128)
                _src_is_cuda = nv12_gpu_tensor.is_cuda
                src_ptr = nv12_gpu_tensor.data_ptr()
                if _src_is_cuda:
                    cast(byref(_cpy2d, 16), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
                    cast(byref(_cpy2d, 32), ctypes.POINTER(c_void_p))[0] = c_void_p(src_ptr)  # srcDevice
                else:
                    cast(byref(_cpy2d, 16), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_HOST
                    cast(byref(_cpy2d, 24), ctypes.POINTER(c_void_p))[0] = c_void_p(src_ptr)  # srcHost
                cast(byref(_cpy2d, 48), ctypes.POINTER(c_size_t))[0] = W
                cast(byref(_cpy2d, 72), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
                cast(byref(_cpy2d, 88), ctypes.POINTER(c_void_p))[0] = c_void_p(mapped_ptr)
                cast(byref(_cpy2d, 104), ctypes.POINTER(c_size_t))[0] = (
                    actual_pitch if actual_pitch > 0 else W)
                cast(byref(_cpy2d, 112), ctypes.POINTER(c_size_t))[0] = W
                cast(byref(_cpy2d, 120), ctypes.POINTER(c_size_t))[0] = nv12_h
                r = self._copy_into_input_buffer(_cpy2d)  # [FIX-ASYNC-COPY]
                if r != 0:
                    _UnlockInputBufferProto(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                        self._encoder, self._input_buf_handle)
                    raise RuntimeError("[NVENCEncoder] cuMemcpy2D failed, code=%d" % r)

                # ── UnlockInputBuffer ──
                _UnlockInputBufferProto(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                    self._encoder, self._input_buf_handle)

                # ── EncodePicture (byte array, verified offsets) ──
                # NV_ENC_PIC_PARAMS (3360B): version@0, inputWidth@4, inputHeight@8, inputPitch@12,
                #   encodePicFlags@16, frameIdx@20, inputTimeStamp@24, inputDuration@32,
                #   inputBuffer@40, outputBitstream@48, completionEvent@56,
                #   bufferFmt@64, pictureStruct@68, pictureType@72, codecPicParams@76
                pic_buf = (c_uint8 * 3360)()
                ctypes.memset(pic_buf, 0, 3360)
                cast(pic_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER        # version@0
                cast(byref(pic_buf, 4), ctypes.POINTER(c_uint32))[0] = W                  # inputWidth@4
                cast(byref(pic_buf, 8), ctypes.POINTER(c_uint32))[0] = self._height        # inputHeight@8 (luma height only for NV12)
                cast(byref(pic_buf, 12), ctypes.POINTER(c_uint32))[0] = (
                    actual_pitch if actual_pitch > 0 else W)                               # inputPitch@12
                cast(byref(pic_buf, 24), ctypes.POINTER(c_uint64))[0] = self._frame_idx   # inputTimeStamp@24
                cast(byref(pic_buf, 40), ctypes.POINTER(c_void_p))[0] = self._input_buf_handle  # inputBuffer@40
                cast(byref(pic_buf, 48), ctypes.POINTER(c_void_p))[0] = self._bs_handle   # outputBitstream@48
                cast(byref(pic_buf, 64), ctypes.POINTER(c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12  # bufferFmt@64
                cast(byref(pic_buf, 68), ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_STRUCT_FRAME   # pictureStruct@68
                if force_idr:
                    cast(byref(pic_buf, 16), ctypes.POINTER(c_uint32))[0] = 0x2  # [SEGMENT-REUSE] NV_ENC_PIC_FLAG_FORCEIDR

                # ★ completionEvent: 创建 CUDA event 并设置到 pic_buf offset 56 ★
                _ce = c_void_p(None)
                self._libcuda.cuEventCreate.restype = c_uint32
                self._libcuda.cuEventCreate.argtypes = [ctypes.POINTER(c_void_p), c_uint32]
                self._libcuda.cuEventCreate(ctypes.byref(_ce), 0)
                cast(ctypes.byref(pic_buf, 56), ctypes.POINTER(c_void_p))[0] = _ce

                encode_picture = _NvEncEncodePictureProto(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                status = encode_picture(self._encoder, cast(pic_buf, ctypes.POINTER(_NvEncPicParams)))

                self._frame_idx += 1

                # ★ cuEventSynchronize — 等待 NVENC 硬件完成编码 ★
                if _ce.value is not None:
                    self._libcuda.cuEventSynchronize.restype = c_uint32
                    self._libcuda.cuEventSynchronize.argtypes = [c_void_p]
                    self._libcuda.cuEventSynchronize(_ce)
                    self._libcuda.cuEventDestroy.restype = c_uint32
                    self._libcuda.cuEventDestroy.argtypes = [c_void_p]
                    self._libcuda.cuEventDestroy(_ce)

                h264_data = b""

                # [FIX-LA-DRAIN-SYNC] ★ 全局 drain: 循环 LockBitstream 排空所有已完成 slot ──
                # 参照 test_nvenc_la_frame_conservation.py: 每送入一帧后立即排空所有输出。
                # encode_frame() 返回单帧 bytes，取首个有效帧；若多次 drain 产出多帧
                # 应优先使用 encode_frames_batch() 或 encode_frames_batch_ce_pipeline()。
                _drained = self._drain_outputs_blocking()
                if _drained:
                    h264_data = _drained[0][2]
                if not h264_data:
                    h264_data = b""

                if status == NV_ENC_ERR_NEED_MORE_INPUT:
                    # lookahead: encoder needs more frames to fill the lookahead window.
                    self.__dict__.setdefault('_la_buffered', 0)
                    self._la_buffered += 1
                    if self._la_buffered <= 3 or self._la_buffered == self._la_depth:
                        print(f'[NVENC-Enc] 前向帧预看缓冲中 ({self._la_buffered}/{self._la_depth})',
                              flush=True)
                    return h264_data if h264_data else b""
                elif not h264_data:
                    # Tier 0: 空帧诊断 — 记录 NV12 输入统计以区分根因类别
                    self.__dict__.setdefault('_diag_empty', 0)
                    self._diag_empty += 1
                    _nv12_mean = float(nv12_gpu_tensor.float().mean())
                    _nv12_std  = float(nv12_gpu_tensor.float().std())
                    _nv12_min  = int(nv12_gpu_tensor.min())
                    _nv12_max  = int(nv12_gpu_tensor.max())
                    if self._diag_empty <= 5 or self._diag_empty % 50 == 0:
                        print(f'[NVENC-Enc] ⚠️ 空帧 #{self._diag_empty} (encode_frame) '
                              f'frame_idx={self._frame_idx - 1} force_idr={force_idr} '
                              f'nv12_mean={_nv12_mean:.1f} std={_nv12_std:.1f} '
                              f'min={_nv12_min} max={_nv12_max}', flush=True)

                if status != NV_ENC_SUCCESS:
                    raise RuntimeError("[NVENCEncoder] EncodePicture failed, code=%d" % status)

                # [SEGMENT-REUSE] 首段首次编码时缓存 SPS+PPS，后续段 force_idr 帧预挂
                if force_idr and self._cached_sps_pps is not None and \
                        not self._has_sps_pps(h264_data):
                    h264_data = self._cached_sps_pps + h264_data
                elif force_idr and self._cached_sps_pps is None and h264_data:
                    # [FIX-SPS-CACHE] 首个 IDR 帧: 从码流提取并缓存 SPS+PPS
                    self._cached_sps_pps = self._extract_sps_pps(h264_data)
                    if self._cached_sps_pps:
                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" % len(self._cached_sps_pps),
                              flush=True)
                        # [FIX-SPS-PPS-V3] 首帧已含 NVENC 初始化 SPS+PPS, 不 prepend, 仅预注入 muxer
                        if self._muxer_ref is not None:
                            try:
                                self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                self._sps_pps_injected = True
                            except Exception:
                                pass
                elif not force_idr and self._cached_sps_pps is None and h264_data:
                    self._cached_sps_pps = self._extract_sps_pps(h264_data)
                    if self._cached_sps_pps:
                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes" % len(self._cached_sps_pps),
                              flush=True)
                return h264_data
        finally:
            if _need_pop:
                try:
                    _ctx_out = c_void_p()
                    self._libcuda.cuCtxPopCurrent.restype = c_uint32
                    self._libcuda.cuCtxPopCurrent.argtypes = [ctypes.POINTER(c_void_p)]
                    self._libcuda.cuCtxPopCurrent(ctypes.byref(_ctx_out))
                except Exception:
                    pass

    def flush(self) -> bytes:
        # [FIX-FLUSH-CTX] 跨线程 CUDA context 保护：flush() 可能由 Writer 线程
        # （而非 Encode 线程）调用，必须显式 push primary context 确保 NVENC API
        # 在正确上下文执行。与 encode_frames_batch() 中的保护逻辑一致。
        _need_pop = False
        _primary = getattr(self, '_primary_ctx', None)
        if _primary is not None and _primary.value is not None:
            try:
                self._libcuda.cuCtxPushCurrent.restype = c_uint32
                self._libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
                _r_push = self._libcuda.cuCtxPushCurrent(_primary)
                _need_pop = (_r_push == 0)
            except Exception:
                pass
        try:
            with self._lock:
                if self._encoder.value is None:
                    return b""

                # [FIX-LA-EOS] EOS 仅在 flush() 中发送一次，ce_pipeline 不再发送。
                # 发送 EOS → 循环排空所有 slot → 回收 LA 滞留帧。
                # Send EOS
                pic_buf = (c_uint8 * 3360)()
                ctypes.memset(pic_buf, 0, 3360)
                cast(pic_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
                cast(byref(pic_buf, 16), ctypes.POINTER(c_uint32))[0] = 0x8  # encodePicFlags@16 = NV_ENC_PIC_FLAG_EOS
                cast(byref(pic_buf, 40), ctypes.POINTER(c_void_p))[0] = c_void_p(None)
                cast(byref(pic_buf, 48), ctypes.POINTER(c_void_p))[0] = self._bs_handle  # outputBitstream

                encode_picture = _NvEncEncodePictureProto(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                encode_picture(self._encoder, cast(pic_buf, ctypes.POINTER(_NvEncPicParams)))

                # Drain ALL slot bitstream buffers after EOS.
                # 4-slot 轮转下每个 slot 的 lookahead 滞留帧都需排空，
                # 仅排空 slot 0 会导致 slot 1/2/3 的帧永久丢失 (-6~-7 帧)。
                result_parts = []
                _LockBitstreamProto_raw = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _total_recovered_frames = 0
                _total_recovered_bytes = 0

                # [FIX-LA-FLUSH] 按 _output_slot_idx 起始顺序排空所有 slot，
                # 保证 EOS flush 的输出顺序与编码器内部输出顺序严格一致。
                # 参照 tests/test_nvenc_la_frame_conservation.py flush_eos()。
                _start_slot = self._output_slot_idx % self._slot_count
                _drain_order = [(_start_slot + i) % self._slot_count
                                for i in range(self._slot_count)]
                for _slot_idx in _drain_order:
                    _slot = self._slots[_slot_idx]
                    _bs_handle = _slot['bs_buf']
                    _slot_parts = []
                    # [FIX-LA-FLUSH] 首个 LockBitstream 使用 blocking (doNotWait=0)
                    # 确保 LA 滞留帧全部取回。EOS 后 NVENC 可能仍需时间将 LA
                    # 缓冲区中的帧写入 bs_buf，非阻塞 Lock 可能遗漏未完成帧。
                    _first_lock = True
                    while True:
                        lock_raw = (c_uint8 * 1544)()
                        ctypes.memset(lock_raw, 0, 1544)
                        cast(lock_raw, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
                        # [FIX-LA-FLUSH] 首次 blocking Lock (doNotWait=0)，
                        # 后续用 non-blocking (doNotWait=1) 加速排空
                        cast(byref(lock_raw, 4), ctypes.POINTER(c_uint32))[0] = (
                            0 if _first_lock else 1)
                        cast(byref(lock_raw, 8), ctypes.POINTER(c_void_p))[0] = _bs_handle

                        lock_bs_fn = _LockBitstreamProto_raw(self._func_ptrs[_FUNC_IDX["LockBitstream"]])
                        bs_status = lock_bs_fn(self._encoder, lock_raw)
                        if bs_status == NV_ENC_ERR_NEED_MORE_INPUT:
                            break
                        if bs_status != NV_ENC_SUCCESS:
                            break

                        bitstream_size = cast(byref(lock_raw, 36), ctypes.POINTER(c_uint32))[0]
                        if bitstream_size == 0:
                            _NvEncUnlockBitstreamProto(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(
                                self._encoder, _bs_handle)
                            break

                        _raw_bsptr = cast(byref(lock_raw, 56), ctypes.POINTER(c_void_p))[0]
                        bitstream_ptr_val = _raw_bsptr if isinstance(_raw_bsptr, int) else (_raw_bsptr.value or 0)
                        if bitstream_ptr_val:
                            buf_type = c_uint8 * bitstream_size
                            _slot_parts.append(bytes(buf_type.from_address(bitstream_ptr_val)))

                        _NvEncUnlockBitstreamProto(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(
                            self._encoder, _bs_handle)
                        _first_lock = False

                    if _slot_parts:
                        _slot_bytes = sum(len(p) for p in _slot_parts)
                        _total_recovered_frames += len(_slot_parts)
                        _total_recovered_bytes += _slot_bytes
                        print(f'[NVENC-FLUSH] slot[{_slot_idx}] drained {len(_slot_parts)} frames, '
                              f'{_slot_bytes} bytes (SDK-compliant EOS drain)', flush=True)
                    result_parts.extend(_slot_parts)

                if _total_recovered_frames > 0:
                    print(f'[NVENC-FLUSH] total drained: {_total_recovered_frames} frames, '
                          f'{_total_recovered_bytes} bytes (SDK-compliant LA EOS flush — valid encoded frames)', flush=True)

                # NVENC EOS flush 排出的帧是有效编码帧（SDK 合规排空逻辑下帧数守恒）。
                # 已修正：旧版误认为花屏残片（错误排空导致缓冲区覆盖），正确实现下可正常使用。
                self._flush_frame_count = _total_recovered_frames
                return b"".join(result_parts)
        finally:
            if _need_pop:
                try:
                    _ctx_out = c_void_p()
                    self._libcuda.cuCtxPopCurrent.restype = c_uint32
                    self._libcuda.cuCtxPopCurrent.argtypes = [ctypes.POINTER(c_void_p)]
                    self._libcuda.cuCtxPopCurrent(ctypes.byref(_ctx_out))
                except Exception:
                    pass

    def close(self):
        with self._lock:
            if self._encoder.value is None:
                return

            self._destroy_all_slots()

            # [FIX-ASYNC-COPY] 销毁专用编码拷贝 stream
            if self._stream_encode.value is not None:
                try:
                    self._libcuda.cuStreamDestroy_v2.restype = c_uint32
                    self._libcuda.cuStreamDestroy_v2.argtypes = [c_void_p]
                    self._libcuda.cuStreamDestroy_v2(self._stream_encode)
                except Exception:
                    pass
                self._stream_encode = c_void_p(None)

            destroy_addr = self._func_ptrs[_FUNC_IDX["DestroyEncoder"]]
            if destroy_addr:
                _NvEncDestroyEncoderProto(destroy_addr)(self._encoder)
            self._encoder = c_void_p(None)
            self._sps_pps_injected = False   # [FIX-SPS-PPS-V2] 跨段重置换，支持 encoder 复用
            print("[NVENCEncoder] Encoder closed", flush=True)

            # Restore saved context
            if self._saved_ctx.value is not None:
                try:
                    self._libcuda.cuCtxPushCurrent.restype = c_uint32
                    self._libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
                    self._libcuda.cuCtxPushCurrent(self._saved_ctx)
                except Exception:
                    pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def frame_count(self) -> int:
        return self._frame_idx


# ==============================================================================
# [FIX-ENC-THREAD] 独立 NVENC 编码线程
# 将 NVENC 编码（encode_frames_batch）与 T3 Writer 的 RGB→NV12 kernel 并行化。
# T3 Writer 完成 event.sync + NV12 转换后将帧列表提交到队列，
# 编码线程独立调用 encode_frames_batch + muxer.write，
# 两者在 T4 SM（NV12 kernel）与 NVENC 固定功能硬件（编码）上真正并行。
# ==============================================================================


class _NVENCEncodeThread:
    """独立 NVENC 编码线程（GPU_RAW Level 1 路径专用）。

    设计约束：
      · T3 Writer 在调用 submit() 之前必须先调用
        torch.cuda.current_stream().synchronize()，确保 NV12 GPU tensor
        数据已完全写入 VRAM，防止编码线程 cuMemcpy2D 读到未完成的数据（静默花帧）。
      · encode_queue_depth: 编码线程内部排队批次数。
        LA=0: 4 (VRAM NV12 tensors, ~1.2GB on T4 16GB)
        LA>0: 默认自适应增大（pinned host NV12 tensors），见 [FIX-LA-QUEUE-DEPTH]
      · 编码线程是 FFmpegMuxer pipe 的唯一写入者（GPU_RAW 路径），保证写入顺序正确。
      · 编码线程为 daemon 线程；正常路径通过 flush_and_join() 完成收尾。
    """

    _SENTINEL = object()

    def __init__(self, nvenc_encoder, writer, encode_queue_depth: int = 4,
                 batch_frames: int = 48,  # [FIX-CHUNK-SAFE] 新增参数
                 flush_chunk_frames: int = 128):  # [FIX-FLUSH-GRANULARITY]
        self._nvenc   = nvenc_encoder
        self._writer  = writer
        # [FIX-LA-QUEUE-DEPTH] LA>0 chunk encoding batches can be queued while
        # a chunk is being encoded. Default depth=4 is too small for LA>0 mode.
        # Increase to 16 so chunk≤16×batch_frames = zero Writer blocking.
        # Explicitly-passed encode_queue_depth is always honored (no override).
        _la = getattr(nvenc_encoder, '_la_depth', 0)
        if _la > 0 and encode_queue_depth == 4:
            encode_queue_depth = 16
        self._q: queue.Queue = queue.Queue(maxsize=encode_queue_depth)
        # [FIX-CHUNK-SAFE] chunk ≤ queue_depth × batch_frames → Writer 永不阻塞
        # 16×48=768 帧，在 640×360 下 ≈ 3.8s 编码时间，result_queue(15) 不溢出
        # （安全上限，防止累积无界；不再作为实际触发编码的粒度，见下方
        #  [FIX-FLUSH-GRANULARITY]）。
        self._la_chunk_safe = encode_queue_depth * batch_frames
        # [FIX-FLUSH-GRANULARITY] v6.4.5.1: 原设计"攒够 768 帧才编码一整块"
        # 会把 NVENC 的 Lock/Copy/Encode/LockBitstream 逐帧同步调用压缩成一次
        # 持续 1-2s+ 的连续脉冲。即使 [FIX-ASYNC-COPY] 消除了 cuMemcpy2D 走
        # legacy stream 造成的跨 stream 隐式全局同步，这个脉冲本身仍然是
        # encode 线程独占、持续较长时间的一段串行 CPU/NVENC 工作——期间
        # self._q（编码线程输入队列，maxsize=encode_queue_depth）迅速被
        # Writer 写满，Writer 阻塞在 submit()，进而阻塞它消费 result_queue，
        # result_queue 触顶后 T2 推理线程也被阻塞产帧，GPU 计算利用率随之
        # 骤降——这是队列反压（backpressure），与 CUDA stream 选择无关，
        # 是 FIX-ASYNC-COPY 修复后残留周期性掉底的主因。
        #
        # 段首之所以平稳：_acc_nv12 从空开始累积，尚未触发第一次 flush，
        # 期间队列有充足余量吸收 Writer 的产出，因此看不到反压。一旦跨过
        # 第一个 chunk 边界、出现第一次同步脉冲，才开始出现周期性凹陷。
        #
        # 修复：把"触发编码"的粒度从安全上限(768)大幅调小到
        # flush_chunk_frames(默认 128 ≈ 1 batch 的 2-3 倍)，让 NVENC 工作
        # 均匀摊薄到许多次小脉冲上，而不是少数几次大脉冲。_la_chunk_safe
        # 仍然保留作为硬上限（三者取最小值），双重保险，正确性不变——
        # encode_frames_stream() 本就是为跨 chunk 连续调用设计的
        # （slot_pending/slots_warmed 已提升为实例状态）。
        self._flush_chunk_frames = flush_chunk_frames
        self.error: Optional[Exception] = None
        self._written = 0
        self._empty   = 0
        self._prev_h264: Optional[bytes] = None  # Tier 1-A: 空帧补偿用前一帧 H.264 数据
        # [FIX-DYN-TIMEOUT] 已提交给编码线程但不一定已编码完成的帧数，供外层
        # （_writer_loop 内的 flush_and_join 调用 / run() 里的 writer 线程 join）
        # 动态估算等待超时，替代原先针对"段末编码脉冲"的固定 150s。
        # 单线程生产者（本 Writer 线程）+ CPython GIL 下的 += 足够安全，
        # 读者（run() 所在的 T2 主线程）读到的是近似值，允许有少量滞后。
        self.submitted_frames = 0
        self._th = threading.Thread(target=self._loop, daemon=True, name='NVENC-Enc')
        self._th.start()

    def submit(self, nv12_list: list, force_idr_first: bool = False):
        """T3 Writer 线程调用：提交一批 NV12 GPU tensor 给编码线程。

        ⚠️ 调用前必须已完成 torch.cuda.current_stream().synchronize()，
        确保 NV12 tensor 写入完成，防止编码线程 cuMemcpy2D 读到 GPU 未写完的数据。
        """
        if self.error is not None:
            raise self.error
        self._q.put((nv12_list, force_idr_first))
        # [FIX-DYN-TIMEOUT] 提交即计数（而非编码完成后才计数），这样在 SENTINEL
        # 尚未处理完、编码线程仍在追赶队列积压时，外层也能读到接近真实的总量。
        self.submitted_frames += len(nv12_list)

    def estimate_timeout(self, min_timeout: float = 30.0,
                          floor_timeout: float = 120.0,
                          assumed_fps: float = 200.0,
                          margin: float = 0.0) -> float:
        """[FIX-DYN-TIMEOUT] 按已提交帧数动态估算编码线程收尾还需要的等待时间。

        assumed_fps=200 为 NVENC VBR_HQ + LA 编码吞吐的保守下限估算
        （实测通常远高于此值，取保守值避免误判超时）。
        floor_timeout 保证短分段/低帧数场景下也至少留出与旧固定值相当的余量，
        避免因估算过短引入新的误报。
        LA=0（无段末编码脉冲）时不需要这条估算，调用方应直接使用 min_timeout。
        """
        _est = self.submitted_frames / assumed_fps + min_timeout
        return max(min_timeout, floor_timeout, _est) + margin

    def _loop(self):
        """编码线程主循环：阻塞等待帧批次，同步编码后写入 muxer。"""
        # [FIX-ENC-CTX] daemon 线程启动时 CUDA context stack 为空；
        # encode_frames_batch 内的 cuCtxPushCurrent 若静默失败会导致
        # cuMemcpy2D_v2 返回 CUDA_ERROR_INVALID_CONTEXT(201)。
        # 此处用 cuCtxSetCurrent 一次性将 primary context 绑定到本线程，
        # 后续 push/pop 循环在已激活的 context 上正常工作。
        _libcuda = getattr(self._nvenc, '_libcuda', None)
        _primary = getattr(self._nvenc, '_primary_ctx', None)
        if _libcuda is not None and _primary is not None and _primary.value is not None:
            try:
                _libcuda.cuCtxSetCurrent.restype  = c_uint32
                _libcuda.cuCtxSetCurrent.argtypes = [c_void_p]
                _r_set = _libcuda.cuCtxSetCurrent(_primary)  # ✅ [FIX-ENC-CTX]
                if _r_set != 0:
                    print(f'[NVENC-Enc] ⚠️ cuCtxSetCurrent 返回 {_r_set}，'
                          f'编码线程可能缺少有效 CUDA context', flush=True)
            except Exception as _e:
                print(f'[NVENC-Enc] ⚠️ cuCtxSetCurrent 异常: {_e}', flush=True)
        # [FIX-LA-CHUNK-STREAM] LA>0 有界分块流式编码。
        # 原 [FIX-LA-ACCUMULATE] 为避免 per-batch 编码破坏 LA 连续性
        # （_slot_pending 批次局部 → 跨批次帧映射失效 → 静默丢帧/帧序错乱），
        # 将整段帧全部累积、段末一次性编码。长分段（如 300s）下产生两大问题：
        #   1) 段末编码脉冲：14k+ 帧一次性编码耗时 25-35s，超过主线程 writer
        #      join 30s 超时 → muxer 被提前关闭 → write to closed file
        #      → 输出静默截断；
        #   2) pinned host 内存随段长线性增长（14k 帧 ≈ 5GB），挤压系统内存。
        # 现 NVENCEncoder 的 slot 未决表已提升为实例持久状态
        # （encode_frames_stream + _stream_begin），lookahead 窗口帧可跨 chunk
        # 正确取回，因此改为有界分块：累积满 _chunk_frames 即编码一块
        # （不发送 EOS，编码器看到的仍是连续流），仅段末最后一块
        # send_eos=True 完整排空。LA=0 继续 per-batch ce_pipeline。
        _la_mode = (self._nvenc._la_depth > 0)
        _acc_nv12: list = []
        _acc_force_idr = False
        _first_batch = True
        _chunk_frames = 2000   # 首块时按帧字节数重算（~1.5GB pinned 预算）
        _strm_submitted = 0    # 本段已提交编码帧数（帧数守恒诊断）
        _strm_returned  = 0    # 本段已取回帧数（帧数守恒诊断）
        # [FIX-F0-IN-BATCH] 从 encoder 取出暂存的 f0 NV12 tensor，
        # 在首个 chunk 开头插入，确保首帧参与 LA 缓冲 + EOS flush。
        _pending_f0 = getattr(self._nvenc, '_pending_f0_nv12', None)
        if _pending_f0 is not None:
            self._nvenc._pending_f0_nv12 = None
            _f0_idr = getattr(self._nvenc, '_pending_f0_force_idr', False)
            self._nvenc._pending_f0_force_idr = False
        else:
            _f0_idr = False

        # [FIX-FIFO-DRAIN] 流式重排缓冲已整体删除（对齐 realesrgan_video 已验证设计）：
        # drain 顺序 = per-slot FIFO 队首映射顺序 = 提交顺序 → 各 chunk 的 pairs 已
        # 严格按流内帧号递增（LA 滞留帧在后续 chunk 以连续 fi 接续写出），跨 chunk
        # 无需重排。旧重排缓冲在标签错位时每 ~132 帧触发一次强制排空风暴
        # （test7 seg0/seg1 frame_num 回退 76/74 次的周期来源）。
        def _wrap_exc(e, where):
            # [FIX-ENC-EXC-CONTEXT] 编码线程异常包装 (frame_idx, pending_slots,
            # pending 计数) 诊断上下文，便于定位 LA 路径崩溃（裸 e 丢失现场）。
            _pending = getattr(self._nvenc, '_strm_slot_pending', {})
            _p_cnt = sum(len(_dq) for _dq in _pending.values())
            return RuntimeError(
                f'[NVENC-Enc] 编码线程异常({where}) '
                f'frame_idx={getattr(self._nvenc, "_frame_idx", -1)} '
                f'pending_slots={len(_pending)} pending_count={_p_cnt}: {e}')

        def _drain_write(pairs):
            """[FIX-SPS-PPS-V2] 写入任何帧之前确保 muxer 已收到 SPS+PPS；
            随后按 pairs 顺序直接写入（fi 严格递增），空帧执行 Tier 1-A 前一帧补偿。"""
            if not self._nvenc._sps_pps_injected:
                _sps = getattr(self._nvenc, '_cached_sps_pps', None)
                if _sps:
                    self._writer.write_sps_pps(_sps)
                    self._nvenc._sps_pps_injected = True
            for _fi, _h264 in pairs:
                if not _h264:
                    self._empty += 1
                    if self._prev_h264 is not None:
                        self._writer.write(self._prev_h264)
                        self._written += 1
                else:
                    self._writer.write(_h264)
                    self._written += 1
                    self._prev_h264 = _h264

        def _encode_chunk(send_eos: bool):
            """编码当前累积块。send_eos=True 仅用于段末最后一块：
            发送 EOS 并完整排空所有 slot，保证帧数守恒。"""
            nonlocal _acc_force_idr, _strm_submitted, _strm_returned
            if not _acc_nv12 and not send_eos:
                return
            pairs = self._nvenc.encode_frames_stream(_acc_nv12, _acc_force_idr,
                                                     send_eos=send_eos)
            _strm_submitted += len(_acc_nv12)
            _strm_returned  += len(pairs)
            # 各元素已被 encoder 提交后置 None（释放 pinned 引用），此处丢弃列表
            _acc_nv12.clear()
            _acc_force_idr = False
            _drain_write(pairs)

        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                break
            nv12_list, force_idr = item
            if _la_mode:
                # LA>0: 有界累积，满一块编码一块（不发送 EOS，LA 状态连续）
                if _first_batch:
                    # 每段新流：重置 slot 未决表 / 流内帧号（IDR 由 fi==0 判定）
                    self._nvenc._stream_begin(force=True)
                    _fb = nv12_list[0].numel() if nv12_list else 0
                    if _fb > 0:
                        _mem_budget = max(200, int(5e8 / _fb))  # ~500MB pinned 内存
                        # [FIX-FLUSH-GRANULARITY] 三者取最小：内存预算 / 队列安全上限 /
                        # 期望的小粒度 flush 阈值。默认情况下 flush_chunk_frames(128)
                        # 远小于 la_chunk_safe(768)，实际生效值 = flush_chunk_frames，
                        # 把大脉冲拆成多次小脉冲，避免编码线程长时间独占导致
                        # 下游队列反压、阻塞 T2 推理产帧。
                        _chunk_frames = min(_mem_budget, self._la_chunk_safe,
                                             self._flush_chunk_frames)
                    print(f'\n[NVENC-Enc] LA={self._nvenc._la_depth} 分块流式编码: '
                          f'chunk={_chunk_frames} 帧/块 (mem≤{_mem_budget if _fb>0 else "?"} '
                          f'safe≤{self._la_chunk_safe} flush≤{self._flush_chunk_frames} '
                          f'qd={self._q.maxsize})', flush=True)
                    if _pending_f0 is not None:
                        _acc_nv12.append(_pending_f0)
                        # f0 编码时的 force_idr 优先于 batch 的 force_idr_first
                        if _f0_idr:
                            _acc_force_idr = True
                    _acc_force_idr = _acc_force_idr or force_idr
                    _first_batch = False
                _acc_nv12.extend(nv12_list)
                if len(_acc_nv12) >= _chunk_frames:
                    try:
                        _encode_chunk(send_eos=False)
                    except Exception as e:
                        self.error = _wrap_exc(e, 'LA chunk encode')
                        return
            else:
                # LA=0: per-batch ce_pipeline（原有路径）
                try:
                    h264_list = self._nvenc.encode_frames_batch_ce_pipeline(nv12_list, force_idr)
                    # [FIX-SPS-PPS-V2] encode 返回后、写入任何帧之前，确保 muxer 已收到 SPS+PPS。
                    if not self._nvenc._sps_pps_injected:
                        _sps = getattr(self._nvenc, '_cached_sps_pps', None)
                        if _sps:
                            self._writer.write_sps_pps(_sps)
                            self._nvenc._sps_pps_injected = True
                    for i, h264_data in enumerate(h264_list):
                        if h264_data is None:
                            self._empty += 1
                            if self._prev_h264 is not None:
                                self._writer.write(self._prev_h264)
                                self._written += 1
                        elif not h264_data:
                            pass
                        else:
                            self._writer.write(h264_data)
                            self._written += 1
                            self._prev_h264 = h264_data
                except Exception as e:
                    self.error = _wrap_exc(e, 'LA=0 ce_pipeline')
                    return

        # [FIX-LA-CHUNK-STREAM] 段末：最后一块 + EOS 完整排空 LA 滞留帧。
        # 即使 _acc_nv12 为空（总帧数恰被整除）也必须走 send_eos=True，
        # 以取回上一块滞留在 LA 窗口的帧（[FIX-EOS-EMPTY-CHUNK]）。
        if _la_mode and not _first_batch:
            try:
                _encode_chunk(send_eos=True)
                # 帧数守恒诊断：取回数应等于提交数（不等则说明有静默丢帧）
                if _strm_returned != _strm_submitted:
                    print(f'[NVENC-Enc] ⚠️ 帧数守恒校验失败: 提交 {_strm_submitted} '
                          f'vs 取回 {_strm_returned} (差 {_strm_submitted - _strm_returned})',
                          flush=True)
            except Exception as e:
                self.error = _wrap_exc(e, 'EOS final chunk')
                return

        # EOS flush: LA=0 (ce_pipeline) 路径的残余帧排空。
        # LA>0 路径已在 encode_frames_stream(send_eos=True) 中完成 EOS + 全槽排空，
        # 此处仅处理 LA=0 的情况。
        if not _la_mode:
            try:
                flush_data = self._nvenc.flush()
                if flush_data:
                    self._writer.write(flush_data)
                    self._written += 1
                    self._prev_h264 = flush_data
            except Exception:
                pass

    def flush_and_join(self, timeout: float = 120.0):
        """等待编码线程完成所有已提交帧，并返回 (written, empty)。

        编码线程在 _loop() 末尾已执行 NVENC EOS flush，
        此处仅负责发送 SENTINEL 并 join 线程。
        必须在 SENTINEL 处理后、muxer.close() 之前调用。
        """
        self._q.put(self._SENTINEL)
        self._th.join(timeout=timeout)
        if self._th.is_alive():
            print(f'[NVENC-Enc] ⚠️ 编码线程未在 {timeout:.0f}s 内退出，可能死锁', flush=True)
        if self.error is not None:
            raise RuntimeError(f'[NVENC-Enc] 编码线程异常: {self.error}') from self.error
        # [FIX-ENC-FLUSH] NVENC EOS flush 已移至编码线程 _loop() 末尾执行，
        # 确保与 encode_frames_batch 在同一 CUDA context。此处不再重复 flush。
        return self._written, self._empty


# ==============================================================================
# GPU RGB to NV12 color space conversion (PyTorch)
# ==============================================================================


def _rgb_to_nv12_gpu(rgb_tensor, input_is_bgr: bool = False):
    # input: (H, W, 3) uint8 GPU tensor, RGB channel order (or BGR if input_is_bgr=False)
    # output: (H + H//2, W) uint8 GPU tensor, NV12 layout
    import torch

    H, W, C = rgb_tensor.shape
    assert C == 3
    assert rgb_tensor.dtype == torch.uint8
    assert rgb_tensor.is_cuda

    if input_is_bgr:
        r = rgb_tensor[..., 2].float()   # BGR: ch2 = R
        g = rgb_tensor[..., 1].float()   # BGR: ch1 = G
        b = rgb_tensor[..., 0].float()   # BGR: ch0 = B
    else:
        r = rgb_tensor[..., 0].float()
        g = rgb_tensor[..., 1].float()
        b = rgb_tensor[..., 2].float()

    # BT.601 limited range
    Y = (0.257 * r + 0.504 * g + 0.098 * b + 16.0).clamp_(0, 255).round_().to(torch.uint8)

    # 2x2 average downsample for chroma
    h2, w2 = H // 2, W // 2
    def _avg_down(x):
        return (x[:H - H % 2, :W - W % 2].reshape(h2, 2, w2, 2).mean(dim=(1, 3)))

    r_ds = _avg_down(r)
    g_ds = _avg_down(g)
    b_ds = _avg_down(b)

    Cb = (-0.148 * r_ds - 0.291 * g_ds + 0.439 * b_ds + 128.0).clamp_(0, 255).round_().to(torch.uint8)
    Cr = (0.439 * r_ds - 0.368 * g_ds - 0.071 * b_ds + 128.0).clamp_(0, 255).round_().to(torch.uint8)

    # NV12 UV interleave
    UV = torch.empty((h2, W), dtype=torch.uint8, device=rgb_tensor.device)
    UV[:, 0::2] = Cb
    UV[:, 1::2] = Cr

    return torch.cat([Y, UV], dim=0).contiguous()


def _rgb_to_nv12_gpu_batch(rgb_batch, input_is_bgr: bool = False):
    # [FIX-GPU-STAY] 批量 RGB→NV12: (N, H, W, 3) uint8 GPU tensor → (N, H+H//2, W) uint8 GPU tensor
    # 单次 kernel launch 替代 N 次 _rgb_to_nv12_gpu 调用，消除 kernel launch 累积开销
    import torch

    N, H, W, C = rgb_batch.shape
    assert C == 3
    assert rgb_batch.dtype == torch.uint8
    assert rgb_batch.is_cuda

    if input_is_bgr:
        r = rgb_batch[..., 2].float()   # BGR: ch2 = R
        g = rgb_batch[..., 1].float()   # BGR: ch1 = G
        b = rgb_batch[..., 0].float()   # BGR: ch0 = B
    else:
        r = rgb_batch[..., 0].float()
        g = rgb_batch[..., 1].float()
        b = rgb_batch[..., 2].float()

    # BT.601 limited range: (N, H, W)
    Y = (0.257 * r + 0.504 * g + 0.098 * b + 16.0).clamp_(0, 255).round_().to(torch.uint8)

    # 2x2 average downsample: (N, H//2, W//2)
    h2, w2 = H // 2, W // 2
    def _avg_down(x):
        return x[:, :H - H % 2, :W - W % 2].reshape(N, h2, 2, w2, 2).mean(dim=(2, 4))

    r_ds = _avg_down(r)
    g_ds = _avg_down(g)
    b_ds = _avg_down(b)

    Cb = (-0.148 * r_ds - 0.291 * g_ds + 0.439 * b_ds + 128.0).clamp_(0, 255).round_().to(torch.uint8)
    Cr = (0.439 * r_ds - 0.368 * g_ds - 0.071 * b_ds + 128.0).clamp_(0, 255).round_().to(torch.uint8)

    # NV12 UV interleave: (N, h2, W)
    UV = torch.empty((N, h2, W), dtype=torch.uint8, device=rgb_batch.device)
    UV[:, :, 0::2] = Cb
    UV[:, :, 1::2] = Cr

    return torch.cat([Y, UV], dim=1).contiguous()  # (N, H+h2, W)


# ==============================================================================
# FFmpegMuxer -- pure muxer (H.264 ES -> MP4, -c:v copy)
# ==============================================================================


class FFmpegMuxer:
    # Receives H.264 Elementary Stream, pipes to FFmpeg for MP4 muxing only.
    # No re-encoding -- NVENCEncoder already encoded on GPU.

    def __init__(
        self,
        output_path: str,
        fps:    float,
        audio_src: Optional[str] = None,
        ffmpeg_bin: str = "ffmpeg",
        quiet: bool = True,
    ):
        self._error: Optional[Exception] = None
        self._write_count = 0

        cmd = [
            ffmpeg_bin, "-y",
            "-f", "h264",
            "-r", f"{fps:.6f}",
            "-i", "pipe:0",
        ]
        if audio_src:
            cmd += ["-i", audio_src, "-c:a", "copy", "-map", "0:v", "-map", "1:a?"]
        cmd += ["-c:v", "copy"]
        cmd += ["-f", "mp4"]
        # [FIX-FASTSTART] 中间分段文件不需要 faststart（moov 二次写入可能
        # 因磁盘 I/O 波动超过 60s 超时被 SIGKILL，导致 "moov atom not found"）。
        # 最终合并步骤会重新生成完整的输出文件，moov atom 由合并步骤正确写入。
        cmd += ["-movflags", "+faststart"]
        cmd += ["-loglevel", "error"]
        cmd += [output_path]

        if not quiet:
            print("   [FFmpegMuxer] cmd: " + " ".join(cmd), flush=True)

        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        self._stderr_lines: List[str] = []
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self):
        try:
            for line in self._proc.stderr:
                decoded = line.decode(errors="ignore").rstrip()
                self._stderr_lines.append(decoded)
                if decoded:
                    print("[FFmpegMuxer ERR] " + decoded)
        except Exception:
            pass

    def write_sps_pps(self, sps_pps: bytes):
        """[FIX-SPS-PPS] 在首帧数据前预注入 SPS+PPS NAL 单元。

        FFmpeg -f h264 parser 需要在收到首帧数据前建立解码器上下文。
        当 NVENC repeatSPSPPS 因 LA buffering 时序未能生效时，
        此方法提供软件层面的兜底防御。

        应在 _cached_sps_pps 首次设置后立即调用。"""
        if sps_pps:
            self._proc.stdin.write(sps_pps)
            self._proc.stdin.flush()

    def write(self, h264_es: bytes):
        if self._error is not None:
            raise RuntimeError("FFmpegMuxer error: " + str(self._error)) from self._error
        try:
            if h264_es:
                self._proc.stdin.write(h264_es)
                self._write_count += 1
        except BrokenPipeError:
            self._error = RuntimeError("FFmpeg muxer stdin pipe broken")
            raise self._error

    def close(self):
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._stderr_thread.join(timeout=5)
        rc = self._proc.returncode
        if rc is not None and rc != 0:
            stderr_out = "\\n".join(self._stderr_lines[-20:])
            print("\\n[FFmpegMuxer Warning] FFmpeg exit=%d, stderr: %s" % (rc, stderr_out[:400]))
        if self._error:
            print("[FFmpegMuxer Warning] write error: " + str(self._error))

# ═══════════════════════════════════════════════════════════════════════════════
# [V6451-LEVEL1-RC] Level 1 (NVENCEncoder ctypes) 码率控制模式
#
# Level 1 是 NVENC GPU 直通编码路径（ctypes 调用 NVENC SDK DLL），与 Level 2
# (FFmpegWriter 子进程) 并行探测，Level 1 成功则跳过 Level 2/3/4。
#
# ── _NVENC_LEVEL1_RATE_MODE ──
# Level 1 的码率控制模式，控制 NV_ENC_RC_PARAMS.rateControlMode：
#
#   "constqp" (默认) — NV_ENC_PARAMS_RC_CONSTQP (mode=0)
#     · 固定 QP 编码，输出码率随场景复杂度波动
#     · 等效 Level 2：-cq:v 0 -b:v 0（纯 CONSTQP）
#     · 端到端延迟最低，无需前瞻缓冲
#     · 与 v6.4.5 原始行为 100% 一致
#
#   "vbr_hq" — NV_ENC_PARAMS_RC_VBR_HQ (mode=32, 0x20 in SDK 12.0+) + targetQuality (CQ 模式)
#     · 质量驱动的 VBR（类似 CRF），通过 targetQuality 控制质量级别
#     · 等效 Level 2：-rc:v vbr_hq -cq:v N -b:v 0
#     · targetQuality 值由 --crf 参数传入（crf=0 时仍使用 CONSTQP）
#     · 输出码率不可预测（编码器按目标质量自适应分配）
#     · [GPU 验证]  header 解析确认 targetQuality@88(uint8) SEQUENTIAL 布局 (nvEncodeAPI.h)；经 116→76→88 三轮修正确认
#
#   "qvbr" — NV_ENC_PARAMS_RC_QVBR (mode=64, 0x40 in SDK 10.0+) + qvbrQuality
#     · 质量可变的 VBR，通过 qvbrQuality 控制质量级别，允许码率灵活波动
#     · 等效 Level 2：-rc:v qvbr -cq:v N -b:v <max_bitrate>
#     · qvbrQuality 值由 --crf 参数传入（crf=0 时仍使用 CONSTQP）
#     · [GPU 验证] qvbrQuality@88(uint8)（与 targetQuality 共用 SEQUENTIAL slot）；非 @rcParams+124
#
#   不支持的 mode：VBR(1) / CBR(2) — 未实现，无对应 Level 2 需求
#
# ── _NVENC_LEVEL1_LOOKAHEAD ──
# Level 1 VBR_HQ 的前向帧预看深度，控制 NV_ENC_RC_PARAMS.lookaheadDepth：
#
#   0   (默认) — 禁用前向预看
#   N>0 — 编码器前瞻 N 帧进行码率分配
#     · 启用条件：_NVENC_LEVEL1_RATE_MODE=="vbr_hq" 且 crf>0（同时满足）
#     · 可选值：0=禁用, 8=低延迟/适中质量, 16=平衡(推荐), 32=最高质量/高延迟
#     · 值越大质量越好，延迟/显存开销越大 (N 帧编码延迟，与插帧流水线兼容)
#     · 等效 Level 2：-rc-lookahead N
#     · [GPU 验证] lookaheadDepth@90(uint16) SEQUENTIAL 布局 (nvEncodeAPI.h)
#
# ── 常用配置示例 ──
#
#   1) 默认行为（与原始 v6.4.5 完全一致）：
#        _NVENC_LEVEL1_RATE_MODE = "constqp"
#        _NVENC_LEVEL1_LOOKAHEAD = 0
#
#   2) CQ 质量优先（匹配 Level 2 -cq:v 23 语义，不启用 lookahead）：
#        _NVENC_LEVEL1_RATE_MODE = "vbr_hq"
#        _NVENC_LEVEL1_LOOKAHEAD = 0
#
#   3) CQ + lookahead（最高质量，匹配 Level 2 -cq:v 23 -rc-lookahead 16）：
#        _NVENC_LEVEL1_RATE_MODE = "vbr_hq"
#        _NVENC_LEVEL1_LOOKAHEAD = 16
#
#   4) QVBR（码率可控质量，匹配 Level 2 -rc:v qvbr -cq:v 23 -b:v 50M）：
#        _NVENC_LEVEL1_RATE_MODE = "qvbr"
#        _NVENC_LEVEL1_LOOKAHEAD = 16
#        # 注意：QVBR 额外需要 vbvBufferSize(@+28) 和 vbvInitialDelay(@+32)
#        # 需在 NVENCEncoder 或 call site 中设置（当前自动设为 vbvBuf=4MB, vbvDelay=2MB）
#
#   5) lookahead 仅在 crf>0 时生效。crf=0 时（--crf 0），call site 强制
#      la_depth=0 以保持零输出延迟兼容性，忽略此处的 LOOKAHEAD 设置。
#
#   修改常量后无需改动任何其他代码 — 运行管线和 call site 自动适配。
# ═══════════════════════════════════════════════════════════════════════════════
_NVENC_LEVEL1_RATE_MODE: str = "vbr_hq"
_NVENC_LEVEL1_LOOKAHEAD: int = 8  # LA depth for VBR_HQ/QVBR. range 0-30.
# CE-pipeline 初始 slot 数，运行时会 auto-calibrate 为 >= LA+1 (SDK 硬件安全要求)
_NVENC_LEVEL1_DEFAULT_SLOTS: int = 4

# ── _NVENC_CRF0_FORCE_CONSTQP ──
# crf=0 时是否强制使用 CONSTQP + 禁用 lookahead。
#   True  (默认) — crf=0 时强制 rate_mode→"constqp", qp=0, la_depth=0
#                   覆盖 config/CLI 中的 rate_mode 配置（与历史行为 100% 一致）。
#                   运行时会打印日志说明覆盖原因。
#   False — crf=0 时不覆盖 rate_mode/lookahead，使用下方独立 quality 值。
#           仅对 vbr_hq/qvbr 有意义；constqp 模式下 crf=0 自然就是 qp=0。
_NVENC_CRF0_FORCE_CONSTQP: bool = True   # crf=0 时强制 CONSTQP qp=0（真无损，避免 VBR_HQ/QVBR RC 丢弃末帧导致少1帧）
# ── 以下常量仅在 _NVENC_CRF0_FORCE_CONSTQP=False 时生效 ──
# _NVENC_CRF0_QUALITY: crf=0 且使用配置 rate_mode 时的 qp 值。
#   值 0 具有双重语义，在 NVENCEncoder 内部按 rate_mode 分流处理：
#     CONSTQP:        qp=0 → 真逐像素无损
#     VBR_HB/QVBR:    qp<=0 → _qp_val=1 → targetQuality=1（NVENC scale: 1=最好）
#   不可改为 1：会破坏 CONSTQP 模式下的真无损（qp=0 变 qp=1 近无损）。
_NVENC_CRF0_QUALITY: int = 0
# crf=0 且 _NVENC_CRF0_FORCE_CONSTQP=False 时的 lookahead 深度。
# 默认 8 与 _NVENC_LEVEL1_LOOKAHEAD 一致，FORCE_CONSTQP=False 时
# vbr_hq/qvbr 享有正常前向预看质量提升。
_NVENC_CRF0_LOOKAHEAD: int = 8

# ── _NVENC_QVBR_ENABLE_VBV ──
# QVBR 模式是否启用 VBV 缓冲合规约束（vbvBufferSize / vbvInitialDelay）。
#   True  — 启用：设置 4MB buffer + 2MB initial delay，保证实时流合规，
#           但每帧编码后需更新 VBV 水位 + QP 范围约束，增加 ~5% 延迟。
#   False — 关闭：跳过 VBV 设置，编码器不追踪虚拟解码器缓冲，离线编码
#           无合规需求，质量无明显差异。
#   [GPU 验证] 移除 VBV 后 QVBR 编码延迟降低 ~5%。
_NVENC_QVBR_ENABLE_VBV: bool = False
