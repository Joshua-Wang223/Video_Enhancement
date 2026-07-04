#!/usr/bin/env python3
"""
NVENC SDK Level 1 编码模块 (ctypes CUDA/NVENC SDK 13.0)

从 IFRNet v6.4.5.1 提取，供 Real-ESRGAN 复用。

组件:
  - NVENCEncoder: GPU 直接 H.264 硬件编码器 (CE-pipeline, LA 支持)
  - _NVENCEncodeThread: daemon 编码线程 (异步编码 + 空帧防御)
  - _rgb_to_nv12_gpu / _rgb_to_nv12_gpu_batch: GPU RGB→NV12 色彩空间转换
  - FFmpegMuxer: H.264 ES → MP4 纯 muxer (-c:v copy)

跨段复用:
  - NVENCEncoder 实例可在多个视频段间复用，保持 LA FIFO 连续性
  - _frame_idx 不重置，SPS+PPS 首段缓存后续段自动 prepend
  - FFmpegMuxer 每段独立创建，段结束即关闭
"""

import os
import sys
import time
import queue
import threading
import subprocess
import ctypes

import numpy as np
from ctypes import (
    c_uint8, c_uint16, c_uint32, c_uint64, c_int, c_size_t,
    c_void_p, c_bool, c_float, Structure, sizeof, cast, byref, POINTER,
    pointer,
)
from typing import Optional, List, Dict, Tuple, Any, Deque
from collections import deque

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
# 新公式（2026-06-09）：targetQuality = max(1, 51 - CRF)
#   · CRF=0 (lossless/best) → tq=51 (max quality)
#   · CRF=51 (worst)        → tq=1  (min quality)
#   · CRF=18 → tq=33, CRF=28 → tq=23 （差值 10，应明显区分文件大小）
# 不再使用此常量，保留作为历史参考。
# 历史值：7（CRF-offset 公式，文件仍偏大）、21（加法公式，文件极大 36.9 MB）、15（CRF-offset 公式）
_NVENC_VBR_QUALITY_OFFSET: int = 15

# ── _NVENC_CRF0_FORCE_CONSTQP ──
# crf=0 时是否强制使用 CONSTQP + 禁用 lookahead。
#   True  (默认) — crf=0 时强制 rate_mode→"constqp", la_depth=0
#                   覆盖调用方传入的 rate_mode 配置。
#   False — crf=0 时不覆盖 rate_mode/lookahead，使用下方独立 quality 值。
_NVENC_CRF0_FORCE_CONSTQP: bool = True   # crf=0 时强制 CONSTQP qp=0（真无损，避免 VBR_HQ/QVBR RC 丢弃末帧）
# ── 以下常量仅在 _NVENC_CRF0_FORCE_CONSTQP=False 时生效 ──
# _NVENC_CRF0_QUALITY: crf=0 且使用配置 rate_mode 时的 qp 值。
#   值 0 具有双重语义，在 NVENCEncoder 内部按 rate_mode 分流处理：
#     CONSTQP:        qp=0 → 真逐像素无损
#     VBR_HQ/QVBR:    qp<=0 → _qp_val=1 → targetQuality=1（NVENC scale: 1=最好）
#   不可改为 1：会破坏 CONSTQP 模式下的真无损（qp=0 变 qp=1 近无损）。
_NVENC_CRF0_QUALITY: int = 0
# crf=0 且 _NVENC_CRF0_FORCE_CONSTQP=False 时的 lookahead 深度。
_NVENC_CRF0_LOOKAHEAD: int = 0

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

# ==============================================================================
# NVENCEncoder
# ==============================================================================

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
           pipeline_depth: NVENC multi-slot pipeline depth (1-8, default 4)."""
        if codec != "h264":
            raise ValueError("NVENCEncoder: only H.264 supported, got: " + codec)

        self._width = width
        self._height = height
        self._fps = fps
        self._qp = qp
        self._rate_mode = rate_mode
        self._la_depth = la_depth

        # [CRF0-FORCE-CONSTQP] CRF=0 时根据 _NVENC_CRF0_FORCE_CONSTQP 决策
        # 是否强制 CONSTQP qp=0 LA=0（真无损，避免 VBR_HQ/QVBR RC 丢弃末帧）。
        # 必须在 self._rate_mode 赋值之后、pipe/LA 调整之前执行，
        # 以便后续 constqp→la_depth=0 逻辑自然生效。
        if self._qp == 0 and _NVENC_CRF0_FORCE_CONSTQP:
            _cfg_rate = rate_mode
            rate_mode = "constqp"
            la_depth = 0
            self._rate_mode = rate_mode  # 同步更新实例属性（RC params 使用 self._rate_mode 决策）
            print(f"   [NVENCEncoder] CRF=0 + _NVENC_CRF0_FORCE_CONSTQP=True → "
                  f"强制 CONSTQP qp=0 LA=0（配置 rate_mode={_cfg_rate} 被覆盖）", flush=True)
        elif self._qp == 0:
            la_depth = _NVENC_CRF0_LOOKAHEAD
            print(f"   [NVENCEncoder] CRF=0 + _NVENC_CRF0_FORCE_CONSTQP=False → "
                  f"rate_mode={rate_mode} qp={self._qp} LA={la_depth}", flush=True)

        # ── 概念分离（参见 pipeline-depth-slot-rotation-confusion.md）──
        # _required_buffers: SDK 硬件安全下限 — la_depth==0 时为 1，否则 >= LA+1
        # _slot_count: 实际分配的 slot 对数（buffer pool 大小 + 轮转模数）
        # LA=0 时 _slot_count=2 (ce_pipeline 真正的 2 帧流水线深度)
        # LA>0 时 _slot_count=max(user_default, LA+1) (膨胀为 buffer pool，累积模式不参与轮转)
        # CONSTQP 下硬件静默禁用 LA，代码层面清零以与硬件行为一致
        if rate_mode == "constqp":
            la_depth = 0  # 硬件静默禁用 LA，此处显式清零
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
        self._lock = threading.Lock()

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
        enc_cfg.encodeCodecConfig.chromaFormatIDC = 1  # [FIX-CHROMA] 显式启用 chroma (1=4:2:0)，防止 SPS 声明为 monochrome → 灰色输出
        enc_cfg.encodeCodecConfig.idrPeriod = int(fps)
        enc_cfg.encodeCodecConfig.maxNumRefFramesInDPB = 4   # [FIX-GPU-STAY] 16→4: 减少运动估计开销 ~50%，文件体积更接近 v6.4.2
        enc_cfg.encodeCodecConfig.repeatSPSPPS = 1  # [SEGMENT-REUSE] 每个 IDR 前重发 SPS/PPS，确保新 muxer 能收到

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
            _est_br = max(50000000, int(width * height * fps * 3.0))
            rc_ptr[5] = _est_br                      # averageBitRate @offset 20 (速度天花板)
            rc_ptr[6] = _est_br * 2                  # maxBitRate @offset 24
            _tq = max(1, _qp_val)  # VBR_HQ targetQuality = CRF (QP标度, 1=最好, 51=最差)
            # targetQuality: uint8_t at rcParams+88 (nvEncodeAPI.h SEQUENTIAL, GPU verified)
            _tq8_ptr = cast(byref(preset_config, 8 + 40 + 88), ctypes.POINTER(c_uint8))
            _tq8_ptr[0] = _tq & 0xFF
            print(f"[NVENCEncoder] VBR_HQ: crf={_qp_val} targetQuality={_tq} avgBitrate={_est_br//1000}kbps", flush=True)
        elif self._rate_mode == 'qvbr':
            # QVBR mode: NV_ENC_PARAMS_RC_QVBR = 64 (0x40)
            rc_ptr[1] = 64                           # NV_ENC_PARAMS_RC_QVBR (0x40)
            # [FIX-BR-CEILING] QVBR 也需要 avgBitRate 作为速度天花板，
            # 防止 NVENC 在无约束下进入极慢的质量搜索。
            _est_br = max(50000000, int(width * height * fps * 3.0))
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
            print(f"[NVENCEncoder] QVBR: crf={_qp_val} qvbrQuality={_tq} maxBitrate={_est_br//1000}kbps", flush=True)
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
            raise RuntimeError("[NVENCEncoder] InitializeEncoder failed, code=%d" % status)

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

        # Setup cuMemcpy2D_v2 (2D pitch-aware copy)
        self._libcuda.cuMemcpy2D_v2.restype = c_uint32
        self._libcuda.cuMemcpy2D_v2.argtypes = [c_void_p]

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

    def _lock_bitstream_with_retry(self, bs_handle, max_retries: int = 5, backoff_us: int = 1000):
        """[Tier 3-E] 带指数退避重试的 LockBitstream。

        应对 NVENC HW 的 bitstream DMA 竞态。
        [V6442-CONSTQP-FAST] CONSTQP 零空帧 → max_retries=2。

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

        return b"", bs_status

    def _drain_outputs_blocking(self, max_slots: int = None) -> list:
        """按 _output_slot_idx 顺序循环 blocking LockBitstream
        排空所有已完成 slot，直到遇到 NEED_MORE_INPUT。

        参照 tests/test_nvenc_la_frame_conservation.py 的 _drain_outputs() 验证模式：
        - 从 _output_slot_idx 指向的 slot 开始，循环 blocking Lock
        - 每成功取回一帧则推进 _output_slot_idx
        - 遇到 NEED_MORE_INPUT 时退出循环

        返回 [(frame_index_estimate, h264_bytes), ...] 列表。
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
            if bitstream_ptr_val:
                buf_type = c_uint8 * bitstream_size
                h264_data = bytes(buf_type.from_address(bitstream_ptr_val))
                est_fi = self._output_slot_idx
                outputs.append((est_fi, h264_data))
            unlock_fn(self._encoder, bs_handle)
            self._output_slot_idx += 1

        return outputs

    def _reset_output_slot_idx(self, start: int = 0):
        """重置输出槽位指针（新批次开始时调用）。"""
        self._output_slot_idx = start

    def _lock_bitstream_blocking(self, bs_handle, timeout_ms: int = 500):
        """Blocking LockBitstream with timeout — 等待 LA 缓冲突帧完成编码。"""
        _LockBS = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
        _UnlockBS = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
        lock_bs_fn = _LockBS(self._func_ptrs[_FUNC_IDX["LockBitstream"]])
        unlock_fn = _UnlockBS(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])

        lock_raw = (c_uint8 * 1544)()
        ctypes.memset(lock_raw, 0, 1544)
        cast(lock_raw, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
        cast(byref(lock_raw, 4), ctypes.POINTER(c_uint32))[0] = 0  # doNotWait=0 (blocking)
        cast(byref(lock_raw, 8), ctypes.POINTER(c_void_p))[0] = bs_handle

        bs_status = lock_bs_fn(self._encoder, lock_raw)
        if bs_status == NV_ENC_ERR_NEED_MORE_INPUT:
            return b"", bs_status
        if bs_status != NV_ENC_SUCCESS:
            return b"", bs_status

        bitstream_size = cast(byref(lock_raw, 36), ctypes.POINTER(c_uint32))[0]
        if bitstream_size == 0:
            unlock_fn(self._encoder, bs_handle)
            return b"", bs_status

        _raw_bsptr = cast(byref(lock_raw, 56), ctypes.POINTER(c_void_p))[0]
        bitstream_ptr_val = _raw_bsptr if isinstance(_raw_bsptr, int) else (_raw_bsptr.value or 0)
        if bitstream_ptr_val:
            buf_type = c_uint8 * bitstream_size
            h264_data = bytes(buf_type.from_address(bitstream_ptr_val))
            unlock_fn(self._encoder, bs_handle)
            return h264_data, bs_status
        unlock_fn(self._encoder, bs_handle)
        return b"", bs_status

    def encode_frames_batch(self, nv12_tensors: list, force_idr_first: bool = False,
                             send_eos: bool = False) -> list:
        """[FIX-LA-DRAIN-SYNC] SDK-compliant batch encode with global drain + EOS support.

        LA=0: per-batch synchronous encode with per-frame global drain.
        LA>0 + send_eos=True: EOS send + per-slot blocking drain for frame conservation.
        LA>0 without send_eos: frames accumulate; caller must eventually call with send_eos=True.

        Uses self._frame_idx for global slot allocation (prevents cross-batch slot clobber).
        Returns list of H.264 bytes (b"" = LA buffering, None = empty).
        """
        n_frames = len(nv12_tensors)
        if n_frames == 0:
            return []

        # [FIX-GPU-STAY] Cross-thread CUDA context protection
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

                results = [None] * n_frames
                # _slot_pending maps drained output to actual frame index.
                _slot_pending = [None] * self._slot_count  # (fi, bs_buf, force_idr, ep_status)
                _slots_warmed = set()
                W = self._width
                nv12_h = self._height + self._height // 2

                _LockInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _UnlockInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
                _CU_MEMORYTYPE_DEVICE = 2

                for fi in range(n_frames):
                    slot_idx = self._frame_idx % self._slot_count
                    slot = self._slots[slot_idx]
                    force_idr = force_idr_first and (slot_idx not in _slots_warmed)

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

                    # ── GPU→GPU copy (cuMemcpy2D, pitch-aware) ──
                    _cpy2d = (c_uint8 * 128)()
                    ctypes.memset(_cpy2d, 0, 128)
                    src_ptr = nv12_tensors[fi].data_ptr()
                    cast(byref(_cpy2d, 16), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
                    cast(byref(_cpy2d, 32), ctypes.POINTER(c_void_p))[0] = c_void_p(src_ptr)
                    cast(byref(_cpy2d, 48), ctypes.POINTER(c_size_t))[0] = W
                    cast(byref(_cpy2d, 72), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
                    cast(byref(_cpy2d, 88), ctypes.POINTER(c_void_p))[0] = c_void_p(mapped_ptr)
                    cast(byref(_cpy2d, 104), ctypes.POINTER(c_size_t))[0] = (
                        actual_pitch if actual_pitch > 0 else W)
                    cast(byref(_cpy2d, 112), ctypes.POINTER(c_size_t))[0] = W
                    cast(byref(_cpy2d, 120), ctypes.POINTER(c_size_t))[0] = nv12_h
                    r = self._libcuda.cuMemcpy2D_v2(cast(_cpy2d, c_void_p))
                    if r != 0:
                        _UnlockInputBufferProto(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                            self._encoder, slot['input_buf'])
                        raise RuntimeError("[NVENCEncoder] cuMemcpy2D[%d] failed, code=%d" % (slot_idx, r))

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
                    # NO completionEvent at offset 56 — synchronous encode
                    cast(byref(pic_buf, 64), ctypes.POINTER(c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12
                    cast(byref(pic_buf, 68), ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_STRUCT_FRAME
                    if force_idr:
                        cast(byref(pic_buf, 16), ctypes.POINTER(c_uint32))[0] = 0x2

                    encode_picture = _NvEncEncodePictureProto(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                    _ep_status = encode_picture(self._encoder, cast(pic_buf, ctypes.POINTER(_NvEncPicParams)))
                    self._frame_idx += 1

                    # Record submission context for drain-to-index mapping
                    _slot_pending[slot_idx] = (fi, slot['bs_buf'], force_idr, _ep_status)

                    if _ep_status == NV_ENC_ERR_NEED_MORE_INPUT:
                        # LA buffering: encoder needs more frames. Do NOT Lock/Unlock
                        # (SDK spec: Lock/Unlock on NEED_MORE_INPUT silently drops delayed output).
                        self.__dict__.setdefault('_la_buffered', 0)
                        self._la_buffered += 1
                        if self._la_buffered <= 3 or self._la_buffered == self._la_depth:
                            print(f'[NVENC-Enc] Forward lookahead buffering ({self._la_buffered}/{self._la_depth})',
                                  flush=True)
                        # Fall through to global drain — other slots may have completed frames.
                    elif _ep_status != NV_ENC_SUCCESS:
                        raise RuntimeError("[NVENCEncoder] EncodePicture[%d] failed, code=%d" % (slot_idx, _ep_status))
                    else:
                        _slots_warmed.add(slot_idx)

                    # ── Global drain: Loop blocking LockBitstream to drain ALL completed slots ──
                    _drained = self._drain_outputs_blocking()
                    for _est_fi, _h264_data in _drained:
                        _drain_slot = _est_fi % self._slot_count
                        _pending = _slot_pending[_drain_slot]
                        if _pending is None:
                            continue
                        _actual_fi, _, _is_idr, _ep_s = _pending
                        if _actual_fi >= n_frames:
                            continue
                        if results[_actual_fi] is not None and results[_actual_fi] != b"":
                            continue

                        if _h264_data:
                            if _is_idr and self._cached_sps_pps is not None:
                                _h264_data = self._cached_sps_pps + _h264_data
                            elif _is_idr and self._cached_sps_pps is None and _h264_data:
                                self._cached_sps_pps = self._extract_sps_pps(_h264_data)
                                if self._cached_sps_pps:
                                    print("[NVENCEncoder] Cached SPS+PPS: %d bytes" % len(self._cached_sps_pps),
                                          flush=True)
                                    if self._muxer_ref is not None:
                                        try:
                                            self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                            self._sps_pps_injected = True
                                        except Exception:
                                            pass
                            elif not _is_idr and self._cached_sps_pps is None and _h264_data:
                                self._cached_sps_pps = self._extract_sps_pps(_h264_data)
                                if self._cached_sps_pps:
                                    print("[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                          len(self._cached_sps_pps), flush=True)
                            results[_actual_fi] = _h264_data
                        elif _ep_s == NV_ENC_ERR_NEED_MORE_INPUT:
                            results[_actual_fi] = b""
                        _slot_pending[_drain_slot] = None

                # ═══════════════════════════════════════════════
                # send_eos: EOS + per-slot blocking drain
                # ═══════════════════════════════════════════════
                if send_eos:
                    # Send EOS picture
                    eos_pic = (c_uint8 * 3360)()
                    ctypes.memset(eos_pic, 0, 3360)
                    cast(eos_pic, ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
                    cast(byref(eos_pic, 16), ctypes.POINTER(c_uint32))[0] = 0x8  # EOS flag
                    cast(byref(eos_pic, 40), ctypes.POINTER(c_void_p))[0] = c_void_p(None)
                    cast(byref(eos_pic, 48), ctypes.POINTER(c_void_p))[0] = self._bs_handle
                    _ep_eos_fn = _NvEncEncodePictureProto(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                    _ep_eos_fn(self._encoder, cast(eos_pic, ctypes.POINTER(_NvEncPicParams)))

                    # Drain each slot in output order (blocking LockBitstream)
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
                            _bs_ptr_raw = cast(byref(_lr, 56), ctypes.POINTER(c_void_p))[0]
                            _bs_ptr = _bs_ptr_raw if isinstance(_bs_ptr_raw, int) else (_bs_ptr_raw.value or 0)
                            if _bs_ptr:
                                _eos_data = bytes((c_uint8 * _bs_size).from_address(_bs_ptr))
                                _pend = _slot_pending[_ds]
                                if _pend is not None:
                                    _actual_fi_d, _, _is_idr_d, _ep_s_d = _pend
                                    if _actual_fi_d < n_frames:
                                        results[_actual_fi_d] = _eos_data
                                        _slot_pending[_ds] = None
                                        self._output_slot_idx += 1
                            _UnlockBS(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(self._encoder, _bs_h)

                else:
                    # Without EOS: final drain attempt (LA frames may not be ready)
                    _drained_final = self._drain_outputs_blocking()
                    for _est_fi, _h264_data in _drained_final:
                        _drain_slot = _est_fi % self._slot_count
                        _pending = _slot_pending[_drain_slot]
                        if _pending is None:
                            continue
                        _actual_fi, _, _is_idr, _ep_s = _pending
                        if _actual_fi >= n_frames:
                            continue
                        if results[_actual_fi] is not None:
                            continue
                        if _h264_data:
                            results[_actual_fi] = _h264_data
                        elif _ep_s == NV_ENC_ERR_NEED_MORE_INPUT:
                            results[_actual_fi] = b""
                        _slot_pending[_drain_slot] = None

                # Fill remaining None entries
                for _fi in range(n_frames):
                    if results[_fi] is None:
                        results[_fi] = b""

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

                # [FIX-LA-OUTPTR] Reset output slot idx at start of each batch
                self._reset_output_slot_idx(0)

                # Per-slot pending state: (ce_handle, frame_index, ep_status, force_idr)
                _slot_pending = [None] * pd
                # [FIX-PIPE4-LA8] fi==0 only force_idr (单 IDR 替代 per-slot ×4 IDR)
                _LockIB = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _UnlockIB = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
                _CUDA_DEV = 2

                for fi in range(n_frames):
                    # [FIX-F0-MISSING] Use global _frame_idx for slot allocation
                    # to avoid conflict with encode_frame's slot usage
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
                            if _prev_idr and self._cached_sps_pps is not None:
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
                        elif _prev_ep_s == NV_ENC_ERR_NEED_MORE_INPUT:
                            results[_prev_fi] = b""
                        else:
                            # 真正空帧：记录诊断
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
                    r = self._libcuda.cuMemcpy2D_v2(cast(_cpy2d, c_void_p))
                    if r != 0:
                        _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                            self._encoder, slot['input_buf'])
                        raise RuntimeError("[NVENCEncoder] cuMemcpy2D[%d] failed, code=%d" %
                                           (slot_idx, r))

                    # ── UnlockInputBuffer ──
                    _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                        self._encoder, slot['input_buf'])

                    # ★ Per-frame CE: create fresh CUDA event ★
                    # [FIX-LA-NOCE] LA>0: disable CE (completion event fires prematurely
                    # in lookahead mode — NVENC signals CE at input acceptance, not output).
                    _la_disable_ce = (self._la_depth > 0)
                    if _la_disable_ce:
                        _ce = c_void_p(None)
                    else:
                        _ce = c_void_p(None)
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
                    # ★ completionEvent at offset 56 ★
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
                    if _ep_status == NV_ENC_ERR_NEED_MORE_INPUT:
                        results[fi] = b""
                        self.__dict__.setdefault('_la_buffered', 0)
                        self._la_buffered += 1
                        if self._la_buffered <= 3 or self._la_buffered == self._la_depth:
                            print(f'[NVENC-Enc] Forward lookahead buffering ({self._la_buffered}/{self._la_depth})',
                                  flush=True)

                    # Save pending state for harvest on next slot rotation.
                    # Tuple: (ce_handle, frame_index, ep_status, is_idr, bs_buf)
                    _slot_pending[slot_idx] = (_ce, fi, _ep_status, force_idr, slot['bs_buf'])

                    # [FIX-LA-CE-DRAIN] LA>0: inline global drain after each frame.
                    # Without CE, completed frames from other slots must be collected
                    # immediately via synchronous drain.
                    if _la_disable_ce:
                        _drained_inline = self._drain_outputs_blocking()
                        for _est_fi, _h264_data in _drained_inline:
                            _drain_slot = _est_fi % pd
                            _pending = _slot_pending[_drain_slot]
                            if _pending is None:
                                continue
                            _actual_fi, _, _is_idr, _ep_s = _pending
                            if _actual_fi >= n_frames:
                                continue
                            if results[_actual_fi] is not None and results[_actual_fi] != b"":
                                continue
                            if _h264_data:
                                results[_actual_fi] = _h264_data
                            elif _ep_s == NV_ENC_ERR_NEED_MORE_INPUT:
                                results[_actual_fi] = b""
                            _slot_pending[_drain_slot] = None

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
                            if _pending_idr and self._cached_sps_pps is not None:
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
                        elif _pending_ep_s == NV_ENC_ERR_NEED_MORE_INPUT:
                            # [FIX-LA-BLKRETRY] LA frame may have completed by Phase 3.
                            # Use blocking LockBitstream to get delayed output.
                            _retry_data, _retry_bs = self._lock_bitstream_blocking(_pending_bs)
                            if _retry_data:
                                results[_pending_fi] = _retry_data
                            else:
                                results[_pending_fi] = b""
                        # else: None stays — real empty frame
                        _slot_pending[slot_idx] = None

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

                # ── GPU→GPU copy (cuMemcpy2D, pitch-aware) ──
                # CUDA_MEMCPY2D (128B): srcX@0,srcY@8,srcMemType@16,srcHost@24,
                #   srcDevice@32,srcArray@40,srcPitch@48,
                #   dstX@56,dstY@64,dstMemType@72,dstHost@80,
                #   dstDevice@88,dstArray@96,dstPitch@104,
                #   WidthInBytes@112,Height@120
                _CU_MEMORYTYPE_DEVICE = 2
                _cpy2d = (c_uint8 * 128)()
                ctypes.memset(_cpy2d, 0, 128)
                src_ptr = nv12_gpu_tensor.data_ptr()
                cast(byref(_cpy2d, 16), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
                cast(byref(_cpy2d, 32), ctypes.POINTER(c_void_p))[0] = c_void_p(src_ptr)
                cast(byref(_cpy2d, 48), ctypes.POINTER(c_size_t))[0] = W
                cast(byref(_cpy2d, 72), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
                cast(byref(_cpy2d, 88), ctypes.POINTER(c_void_p))[0] = c_void_p(mapped_ptr)
                cast(byref(_cpy2d, 104), ctypes.POINTER(c_size_t))[0] = (
                    actual_pitch if actual_pitch > 0 else W)
                cast(byref(_cpy2d, 112), ctypes.POINTER(c_size_t))[0] = W
                cast(byref(_cpy2d, 120), ctypes.POINTER(c_size_t))[0] = nv12_h
                r = self._libcuda.cuMemcpy2D_v2(cast(_cpy2d, c_void_p))
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

                # ── Global drain: 排空所有已完成 slot（替代单 slot 0 LockBitstream）──
                _drained = self._drain_outputs_blocking()
                if _drained:
                    h264_data = _drained[0][1]

                if status == NV_ENC_ERR_NEED_MORE_INPUT:
                    # lookahead: encoder needs more frames to fill the lookahead window.
                    self.__dict__.setdefault('_la_buffered', 0)
                    self._la_buffered += 1
                    if self._la_buffered <= 3 or self._la_buffered == self._la_depth:
                        print(f'[NVENC-Enc] 前向帧预看缓冲中 ({self._la_buffered}/{self._la_depth})',
                              flush=True)
                    return b""
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
                if force_idr and self._cached_sps_pps is not None:
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
                # Per-slot blocking drain in _output_slot_idx order.
                # SDK-compliant: EOS flush output frames are valid encoded frames.
                result_parts = []
                _LockBitstreamProto_raw = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _total_recovered_frames = 0
                _total_recovered_bytes = 0

                _start_slot = self._output_slot_idx % self._slot_count
                _drain_order = [(_start_slot + i) % self._slot_count
                                for i in range(self._slot_count)]
                for _slot_idx in _drain_order:
                    _slot = self._slots[_slot_idx]
                    _bs_handle = _slot['bs_buf']
                    _slot_parts = []
                    _first_lock = True
                    while True:
                        lock_raw = (c_uint8 * 1544)()
                        ctypes.memset(lock_raw, 0, 1544)
                        cast(lock_raw, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
                        cast(byref(lock_raw, 4), ctypes.POINTER(c_uint32))[0] = (
                            0 if _first_lock else 1)  # blocking first, then non-blocking
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
      · encode_queue_depth=4：限制 VRAM 中积压的 NV12 tensor 批次数。
        T4 16GB 下每批约 300MB，depth=2 最多多占 ~600MB，不挤压 T2 推理空间。
      · 编码线程是 FFmpegMuxer pipe 的唯一写入者（GPU_RAW 路径），保证写入顺序正确。
      · 编码线程为 daemon 线程；正常路径通过 flush_and_join() 完成收尾。
    """

    _SENTINEL = object()

    def __init__(self, nvenc_encoder, writer, encode_queue_depth: int = 4):
        self._nvenc   = nvenc_encoder
        self._writer  = writer
        self._q: queue.Queue = queue.Queue(maxsize=encode_queue_depth)
        self.error: Optional[Exception] = None
        self._written = 0
        self._empty   = 0
        self._prev_h264: Optional[bytes] = None  # Tier 1-A: 空帧补偿用前一帧 H.264 数据
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

    def _loop(self):
        """编码线程主循环：LA>0 全段累积 + send_eos，LA=0 per-batch ce_pipeline。"""
        # [FIX-ENC-CTX] daemon 线程 CUDA context 绑定
        _libcuda = getattr(self._nvenc, '_libcuda', None)
        _primary = getattr(self._nvenc, '_primary_ctx', None)
        if _libcuda is not None and _primary is not None and _primary.value is not None:
            try:
                _libcuda.cuCtxSetCurrent.restype  = c_uint32
                _libcuda.cuCtxSetCurrent.argtypes = [c_void_p]
                _r_set = _libcuda.cuCtxSetCurrent(_primary)
                if _r_set != 0:
                    print(f'[NVENC-Enc] WARNING cuCtxSetCurrent returned {_r_set}, '
                          f'encode thread may lack valid CUDA context', flush=True)
            except Exception as _e:
                print(f'[NVENC-Enc] WARNING cuCtxSetCurrent exception: {_e}', flush=True)

        # [FIX-LA-ACCUMULATE] LA>0: cross-batch frame accumulation to preserve LA buffer continuity.
        # Per-batch encoding with local fi%pd slot assignment causes LA buffered frames
        # from one batch to overwrite slots in the next. Fix: accumulate all frames
        # across batches, encode once at SENTINEL with send_eos=True.
        # LA=0: continue per-batch ce_pipeline for async CE performance advantage.
        _la_mode = (self._nvenc._la_depth > 0)
        _acc_nv12: list = []
        _acc_force_idr = False
        _first_batch = True
        # [FIX-F0-IN-BATCH] Retrieve stashed f0 NV12 tensor from encoder,
        # insert at head of accumulated batch so first frame participates in LA buffering.
        _pending_f0 = getattr(self._nvenc, '_pending_f0_nv12', None)
        if _pending_f0 is not None:
            self._nvenc._pending_f0_nv12 = None
            _f0_idr = getattr(self._nvenc, '_pending_f0_force_idr', False)
            self._nvenc._pending_f0_force_idr = False
        else:
            _f0_idr = False

        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                break
            nv12_list, force_idr = item
            if _la_mode:
                # LA>0: accumulate frames, encode once at SENTINEL
                if _first_batch:
                    _acc_force_idr = force_idr
                    _first_batch = False
                _acc_nv12.extend(nv12_list)
            else:
                # LA=0: per-batch ce_pipeline (original path)
                try:
                    h264_list = self._nvenc.encode_frames_batch_ce_pipeline(nv12_list, force_idr)
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
                    self.error = e
                    return

        # LA>0: encode all accumulated frames + built-in EOS flush
        if _la_mode and _acc_nv12:
            if _pending_f0 is not None:
                _acc_nv12.insert(0, _pending_f0)
                if _f0_idr:
                    _acc_force_idr = True
            try:
                h264_list = self._nvenc.encode_frames_batch(_acc_nv12, _acc_force_idr,
                                                             send_eos=True)
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
                        self._empty += 1
                        if self._prev_h264 is not None:
                            self._writer.write(self._prev_h264)
                            self._written += 1
                    else:
                        self._writer.write(h264_data)
                        self._written += 1
                        self._prev_h264 = h264_data
            except Exception as e:
                self.error = e
                return

        # EOS flush: LA=0 (ce_pipeline) path residual frame drain.
        # LA>0 path already completed EOS + full-slot drain in encode_frames_batch(send_eos=True).
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
    # input: (H, W, 3) uint8 GPU tensor, RGB channel order (or BGR if input_is_bgr=True)
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
        cmd += ["-movflags", "faststart"]
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


# ==============================================================================
# NVENCWriter — drop-in replacement for FFmpegWriter using NVENC SDK ctypes
# ==============================================================================

class NVENCWriter:
    """NVENC SDK 直接编码 Writer，兼容 pipeline.py 的 write_frame/write_frame_batch API。

    封装 NVENCEncoder + _NVENCEncodeThread + FFmpegMuxer 为一个完整的 writer，
    可替代 FFmpegWriter 作为 DeepPipelineOptimizer 的输出目标。

    数据流：
        write_frame(numpy RGB) → torch GPU → _rgb_to_nv12_gpu → encode_thread.submit
        write_frame_batch(numpy RGB list) → torch GPU batch → _rgb_to_nv12_gpu_batch → submit
        close → flush_and_join → muxer.close
    """

    _MINI_BATCH = 4  # 累积 mini-batch 大小以减少 encode thread 提交次数

    def __init__(self, nvenc_encoder, args, audio, width, height, output_path, fps,
                 audio_src=None, is_last_segment=True, la_carryover=False):
        self._enc = nvenc_encoder
        self._args = args
        self._width = width
        self._height = height
        self._fps = fps
        self._quiet = getattr(args, 'quiet', True)
        self._broken = False
        self._frames_submitted = 0
        self._frames_completed = 0

        # Create muxer for H.264 ES → MP4 muxing
        _qf = getattr(args, 'quiet', True)
        self._muxer = FFmpegMuxer(
            audio=audio if audio else b"",
            height=height, width=width,
            output_path=output_path, fps=fps,
            ffmpeg_bin=getattr(args, 'ffmpeg_bin', 'ffmpeg'),
            audio_src=audio_src,
            quiet=_qf,
        )
        # Wire muxer ref for SPS/PPS pre-injection
        self._enc.set_muxer_ref(self._muxer)

        # Create encode thread
        _q_depth = getattr(args, 'encode_queue_depth', 4)
        self._enc_thread = _NVENCEncodeThread(self._enc, self._muxer,
                                               encode_queue_depth=_q_depth)
        self._enc_thread._is_last_segment = is_last_segment
        self._enc_thread._la_carryover = la_carryover

        # Mini-batch accumulator
        self._mini_batch: list = []
        self._first_batch = True
        self._pending_force_idr = False

        if not self._quiet:
            print(f'[NVENCWriter] Ready: {width}x{height}@{fps:.1f}fps, '
                  f'slots={self._enc._slot_count}, la_depth={self._enc._la_depth}', flush=True)

    def write_frame(self, frame):
        """写入单帧 numpy RGB24 数组 (H, W, 3) uint8."""
        import torch
        if self._broken:
            return False
        if self._enc_thread.error is not None:
            self._broken = True
            raise RuntimeError(f'[NVENCWriter] Encode thread error: {self._enc_thread.error}')

        self._mini_batch.append(frame)
        if len(self._mini_batch) >= self._MINI_BATCH:
            self._flush_mini_batch()
        return True

    def write_frame_batch(self, frames):
        """批量写入 numpy RGB24 帧列表 [(H, W, 3) uint8, ...]."""
        import torch
        if self._broken:
            return False
        if self._enc_thread.error is not None:
            self._broken = True
            raise RuntimeError(f'[NVENCWriter] Encode thread error: {self._enc_thread.error}')

        # Flush pending mini-batch first, then write this batch
        if self._mini_batch:
            self._flush_mini_batch()

        if not frames:
            return True

        n = len(frames)
        # Stack into GPU tensor
        batch_t = torch.from_numpy(np.stack(frames)).cuda()
        nv12_list = []
        for i in range(n):
            nv12 = _rgb_to_nv12_gpu(batch_t[i], input_is_bgr=True)
            nv12_list.append(nv12)
        del batch_t

        force_idr = self._first_batch
        self._enc_thread.submit(nv12_list, force_idr_first=force_idr)
        self._frames_submitted += n
        self._first_batch = False
        return True

    def _flush_mini_batch(self):
        """将累积的 mini-batch 转换为 NV12 GPU tensor 并提交到编码线程。"""
        if not self._mini_batch:
            return
        n = len(self._mini_batch)
        batch_t = torch.from_numpy(np.stack(self._mini_batch)).cuda()
        nv12_list = []
        for i in range(n):
            nv12 = _rgb_to_nv12_gpu(batch_t[i], input_is_bgr=True)
            nv12_list.append(nv12)
        del batch_t

        force_idr = self._first_batch
        self._enc_thread.submit(nv12_list, force_idr_first=force_idr)
        self._frames_submitted += n
        self._first_batch = False
        self._mini_batch.clear()

    def close(self):
        """完成编码并关闭所有资源。"""
        if self._broken:
            return

        # Flush remaining mini-batch
        if self._mini_batch:
            self._flush_mini_batch()

        # Wait for encode thread to finish
        _written, _empty = self._enc_thread.flush_and_join()
        self._frames_completed = _written

        # Close muxer
        self._muxer.close()

        if not self._quiet:
            print(f'[NVENCWriter] Closed: submitted={self._frames_submitted}, '
                  f'written={_written}, empty={_empty}', flush=True)

    @property
    def frames_written(self) -> int:
        return self._frames_completed
