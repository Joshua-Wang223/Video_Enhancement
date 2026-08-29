#!/usr/bin/env python3
"""
NVENC completionEvent multi-mode verification — 3D full matrix comparison.

3-dimensional matrix: 6 techniques × 2 pipes × 2 LAs = 24 combinations.

  Technique       Description
  ----------      ----------
  sync-batch      Synchronous batch (NO CE) — v6.4.5.1 production baseline
  single-ce       Single-slot per-frame CE (create→sync→destroy)
  batch-ce        Batch per-frame CE across all slots, sync harvest
  phase4-slot     PHASE4 async submit/harvest (slot-reuse event, broken for pipe=4)
  phase4-pfce     PHASE4 async + per-frame CE (fresh CE per frame)
  ce-pipeline     Per-frame CE with deferred harvest at slot reuse

Configurable via top-level constants: W/H, N_FRAMES, PRESET, QP, RATE_MODE.

Usage (on GPU server):
  python test_nvenc_completion_event.py

Self-contained version — NVENC SDK inlined from v6.4.5.1 (no external import).

"""


# ============================================================================
# Part 1: NVENC SDK Library (inlined from process_video_v6_4_5_1_single.py)
# ============================================================================

import ctypes
from ctypes import (c_uint8, c_uint16, c_uint32, c_int32, c_int, c_uint64, c_void_p,
                    c_char, c_size_t, c_double, Structure, POINTER, byref,
                    sizeof, cast, pointer, c_bool)
import os
import threading
import re
import sys
import time
from typing import Optional, List

import torch

# ============================================================================
# NVENC GUID / 常量 / 结构体 / 函数原型 (from v6.4.5.1)
# ============================================================================
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

_NVENC_QVBR_ENABLE_VBV: bool = False
# Offline encoding does not require VBV compliance; disabled by default.
# QVBR mode whether to enable VBV buffer compliance constraints (vbvBufferSize / vbvInitialDelay).
# ── _NVENC_QVBR_ENABLE_VBV ──

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
        """rate_mode: 'constqp' | 'vbr_hq' (CQ via VBR_HQ + targetQuality).
           la_depth: lookahead depth (0=disabled, 8~32 for -rc-lookahead equivalent)."""
        if codec != "h264":
            raise ValueError("NVENCEncoder: only H.264 supported, got: " + codec)

        self._width = width
        self._height = height
        self._fps = fps
        self._qp = qp
        self._rate_mode = rate_mode
        self._la_depth = la_depth

        self._preset_name = preset.lower()
        self._encoder = c_void_p(None)
        self._frame_idx = 0
        self._lock = threading.Lock()

        # [PHASE4-v645] 多 slot 异步流水线（4-slot 轮转），根因修复后恢复
        self._pipeline_depth = max(1, min(8, pipeline_depth))
        self._slots: list = []

        # Backward compat: legacy refs (initialized after slot creation)
        self._input_buf_handle = c_void_p(None)
        self._bs_handle = c_void_p(None)

        # [SEGMENT-REUSE] 缓存首段 SPS+PPS NAL 单元，后续段预挂到首帧前
        self._cached_sps_pps: Optional[bytes] = None

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

        # [V6451-RC-FIXED] NV_ENC_RC_PARAMS at offset 40 in NV_ENC_CONFIG
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
        _qp_val = max(1, qp) if qp > 0 else 28

        if self._rate_mode == 'vbr_hq':
            # CQ mode: VBR_HQ + targetQuality (matches Level 2 -cq:v N behavior)
            # NV_ENC_PARAMS_RC_VBR_HQ = 32 (0x20) in SDK 12.0+.
            rc_ptr[1] = 32                           # NV_ENC_PARAMS_RC_VBR_HQ
            # targetQuality range 1(worst)→51(best). Map: max(1, 51-CRF).
            # 当 targetQuality 启用时，averageBitRate/maxBitRate 置零，
            # 让编码器完全由 targetQuality 驱动质量（与 FFmpeg -b:v 0 语义一致）
            rc_ptr[5] = 0                            # averageBitRate=0 (由targetQuality驱动)
            rc_ptr[6] = 0                            # maxBitRate=0 (无上限)
            _tq = max(1, _qp_val)  # VBR_HQ targetQuality = CRF (QP标度, 1=最好, 51=最差)
            # targetQuality: uint8_t at rcParams+88 (nvEncodeAPI.h SEQUENTIAL, GPU verified)
            _tq8_ptr = cast(byref(preset_config, 8 + 40 + 88), ctypes.POINTER(c_uint8))
            _tq8_ptr[0] = _tq & 0xFF
            print(f"[NVENCEncoder] VBR_HQ: crf={_qp_val} targetQuality={_tq} bitrate=unconstrained", flush=True)
        elif self._rate_mode == 'qvbr':
            # QVBR mode: NV_ENC_PARAMS_RC_QVBR = 64 (0x40)
            rc_ptr[1] = 64                           # NV_ENC_PARAMS_RC_QVBR (0x40)
            rc_ptr[5] = 0                            # averageBitRate @offset 20
            # NVENC 要求 QVBR 必须设置 maxBitRate 作为码率上限
            _est_br = max(50000000, int(width * height * fps * 3.0))
            rc_ptr[6] = _est_br                      # maxBitRate @offset 24
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

        # 11. [PHASE4-v645] 创建多 slot 流水线：每 slot = input buffer + bitstream buffer + CUDA event。
        #     根因修复（v6.4.4 synchronize）后恢复，Multi-slot 让 NVENC HW 帧间流水线化：
        #     准备 slot N+1 的同时 NVENC 处理 slot N，Lock/Copy 与 Encode 重叠。
        nv12_h = height + height // 2
        _CreateInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 776))
        _CreateBitstreamBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 776))

        for slot_idx in range(self._pipeline_depth):
            # 11a. Create input buffer
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

            # 11b. Create bitstream buffer
            bs_buf = (c_uint8 * 776)()
            ctypes.memset(bs_buf, 0, 776)
            cast(bs_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_CREATE_BITSTREAM_BUFFER_VER  # version@0

            s = _CreateBitstreamBufferProto(_get_func(_FUNC_IDX["CreateBitstreamBuffer"]))(self._encoder, bs_buf)
            if s != 0:
                self._destroy_all_slots()
                raise RuntimeError("[NVENCEncoder] CreateBitstreamBuffer[%d] failed, code=%d" % (slot_idx, s))
            _raw_bs = cast(byref(bs_buf, 16), ctypes.POINTER(c_void_p))[0]  # bitstreamBuffer@16
            bs_handle = c_void_p(_raw_bs if isinstance(_raw_bs, int) else (_raw_bs.value or 0))

            # 11c. Create CUDA completion event (async ready signal for NVENC HW)
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

        if self._pipeline_depth > 1:
            print("[NVENCEncoder] %d pipeline slots created (0x%x..0x%x)" %
                  (self._pipeline_depth, self._slots[0]['input_buf'].value,
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
        print("[NVENCEncoder] Ready: %dx%d@%.1ffps H.264 %s QP=%d preset=%s pipeline=%d%s (GPU direct SDK 13.0)" %
              (width, height, fps, _mode_label, _qp_val, self._preset_name, self._pipeline_depth, _extra), flush=True)

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

        应对 NVENC HW 的 bitstream DMA 与 completion signaling 之间的
        底层竞态：偶发 LockBitstream 返回 bitstream_size=0 但数据在
        几百 μs 后可读。重试 3 次（500μs / 1000μs / 2000μs）覆盖
        典型竞态窗口。

        Returns (h264_data: bytes, status: int)
        """
        import time as _time
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

    def encode_frames_batch(self, nv12_tensors: list, force_idr_first: bool = False) -> list:
        """Encode multiple NV12 frames using synchronous per-slot encoding.

        Uses blocking EncodePicture (no completionEvent) + LockBitstream with retry.
        Frames distributed across pipeline slots round-robin to reduce buffer contention.

        Returns list of H.264 bytes in the same order as input tensors.
        """
        n_frames = len(nv12_tensors)
        if n_frames == 0:
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

                results = []
                # [FIX-PERSLOT-IDR] 每槽首帧强制 IDR 以初始化独立 DPB（pipeline_depth>1 时 slot 1/2/3 需各自 IDR）
                _slots_warmed = set()
                W = self._width
                nv12_h = self._height + self._height // 2

                _LockInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _UnlockInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
                _LockBitstreamProto_raw = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _CU_MEMORYTYPE_DEVICE = 2

                for fi in range(n_frames):
                    slot_idx = fi % self._pipeline_depth
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

                    if _ep_status == NV_ENC_ERR_NEED_MORE_INPUT:
                        # [FIX-BS-RELEASE] NVENC API合约: 每次EncodePicture后必须LockBitstream→UnlockBitstream
                        try:
                            _lock_bs = (c_uint8 * 1544)()
                            ctypes.memset(_lock_bs, 0, 1544)
                            cast(_lock_bs, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
                            cast(byref(_lock_bs, 8), ctypes.POINTER(c_void_p))[0] = slot['bs_buf']
                            _lock_bs_fn = _LockBitstreamProto_raw(self._func_ptrs[_FUNC_IDX["LockBitstream"]])
                            _lock_bs_fn(self._encoder, cast(_lock_bs, ctypes.POINTER(c_uint8 * 1544)))
                            _unlock_bs_fn = _NvEncUnlockBitstreamProto(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])
                            _unlock_bs_fn(self._encoder, slot['bs_buf'])
                        except Exception:
                            pass
                        self.__dict__.setdefault('_la_buffered', 0)
                        self._la_buffered += 1
                        if self._la_buffered <= 3 or self._la_buffered == self._la_depth:
                            print(f'[NVENC-Enc] 前向帧预看缓冲中 ({self._la_buffered}/{self._la_depth})',
                                  flush=True)
                        results.append(b"")
                        continue
                    elif _ep_status != NV_ENC_SUCCESS:
                        raise RuntimeError("[NVENCEncoder] EncodePicture[%d] failed, code=%d" % (slot_idx, _ep_status))

                    # [FIX-PERSLOT-IDR] 标记该 slot 已初始化 DPB（后续帧不再 IDR）
                    _slots_warmed.add(slot_idx)

                    # ── LockBitstream with retry (Tier 3-E) ──
                    h264_data, bs_status = self._lock_bitstream_with_retry(slot['bs_buf'])
                    if not h264_data:
                        self.__dict__.setdefault('_diag_empty', 0)
                        self._diag_empty += 1
                        _nv12_t = nv12_tensors[fi]
                        _nv12_mean = float(_nv12_t.float().mean())
                        _nv12_std  = float(_nv12_t.float().std())
                        _nv12_min  = int(_nv12_t.min())
                        _nv12_max  = int(_nv12_t.max())
                        # [DIAG-EMPTY] 增强空帧诊断: 记录完整的编码上下文
                        _la_buf = getattr(self, '_la_buffered', 0)
                        _fidx = self._frame_idx - 1
                        if self._diag_empty <= 5 or self._diag_empty % 50 == 0:
                            print(f'[NVENC-Enc] ⚠️ 空帧 #{self._diag_empty} (batch fi={fi}/{n_frames} '
                                  f'slot={slot_idx} fidr={force_idr} ep_status={_ep_status} '
                                  f'bs_status={bs_status} la_buf={_la_buf} frame_idx={_fidx}) '
                                  f'nv12_mean={_nv12_mean:.1f} std={_nv12_std:.1f} '
                                  f'min={_nv12_min} max={_nv12_max}', flush=True)
                        results.append(None)
                        continue

                    # [SEGMENT-REUSE] SPS/PPS caching and pre-pending
                    if force_idr and self._cached_sps_pps is not None:
                        h264_data = self._cached_sps_pps + h264_data
                    elif force_idr and self._cached_sps_pps is None and h264_data:
                        self._cached_sps_pps = self._extract_sps_pps(h264_data)
                        if self._cached_sps_pps:
                            print("[NVENCEncoder] Cached SPS+PPS: %d bytes" % len(self._cached_sps_pps),
                                  flush=True)
                    elif not force_idr and self._cached_sps_pps is None and h264_data:
                        self._cached_sps_pps = self._extract_sps_pps(h264_data)
                        if self._cached_sps_pps:
                            print("[NVENCEncoder] Cached SPS+PPS: %d bytes" % len(self._cached_sps_pps),
                                  flush=True)

                    results.append(h264_data)

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

                # ── LockBitstream with retry (Tier 3-E) ──
                # 带指数退避重试，应对 NVENC HW DMA 竞态导致的瞬时空帧
                h264_data, bs_status = self._lock_bitstream_with_retry(self._bs_handle)

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
                # 4-slot 轮转下每个 slot 的 lookahead 滞留帧都需排空，
                # 仅排空 slot 0 会导致 slot 1/2/3 的帧永久丢失 (-6~-7 帧)。
                result_parts = []
                _LockBitstreamProto_raw = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _total_recovered_frames = 0
                _total_recovered_bytes = 0

                for _slot_idx, _slot in enumerate(self._slots):
                    _bs_handle = _slot['bs_buf']
                    _slot_parts = []
                    while True:
                        lock_raw = (c_uint8 * 1544)()
                        ctypes.memset(lock_raw, 0, 1544)
                        cast(lock_raw, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
                        cast(byref(lock_raw, 4), ctypes.POINTER(c_uint32))[0] = 1  # doNotWait=1
                        cast(byref(lock_raw, 8), ctypes.POINTER(c_void_p))[0] = _bs_handle

                        lock_bs_fn = _LockBitstreamProto_raw(self._func_ptrs[_FUNC_IDX["LockBitstream"]])
                        bs_status = lock_bs_fn(self._encoder, lock_raw)
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

                    if _slot_parts:
                        _slot_bytes = sum(len(p) for p in _slot_parts)
                        _total_recovered_frames += len(_slot_parts)
                        _total_recovered_bytes += _slot_bytes
                        print(f'[NVENC-FLUSH] slot[{_slot_idx}] recovered {len(_slot_parts)} frames, '
                              f'{_slot_bytes} bytes', flush=True)
                    result_parts.extend(_slot_parts)

                if _total_recovered_frames > 0:
                    print(f'[NVENC-FLUSH] total recovered: {_total_recovered_frames} frames, '
                          f'{_total_recovered_bytes} bytes', flush=True)

                # Store exact frame count for _NVENCEncodeThread.flush_and_join() to use,
                # avoiding NAL heuristic overcounting (5 frames ≠ 42 NAL units).
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


# ── Test config ──
# 720x576 matches production input resolution (word_world_2.mp4).
# VBR_HQ + la_depth=8 = strictest DMA concurrency condition for NVENC completionEvent testing.
# N=600 gives enough runtime at realistic resolution for stable, reproducible FPS measurements.
W, H = 720, 576
FPS = 30.0
N_FRAMES = 2000
N_WARMUP = 30
PRESET = "p1"        # NVENC preset: p1(poor)=lowest latency, p7=highest quality
QP = 23              # H.264 QP / VBR target quality
RATE_MODE = "vbr_hq"  # constqp / vbr_hq / cbr
LA_DEPTH = 8
LA_DEPTH0 = 0  # No lookahead (isolation test)


# ============================================================================
# Helpers
# ============================================================================
def _make_test_rgb(H, W, seed):
    torch.manual_seed(seed)
    return torch.randint(16, 240, (H, W, 3), dtype=torch.uint8, device='cuda')


# ============================================================================
# Mode 1: pipeline_depth=1 single-slot CE (existing, verified)
# ============================================================================
class NVENCEncoderMode1(NVENCEncoder):
    """Per-frame CE encoding (pipe-agnostic, uses slot[0] like encode_frame)."""

    def __init__(self, *args, **kwargs):
        # pipeline_depth flows through naturally from caller; no forced override.
        super().__init__(*args, **kwargs)

    def encode_frame_with_ce(self, nv12_gpu_tensor, force_idr=False):
        """Single frame with per-frame CE (create→encode→sync→destroy→LockBitstream)."""
        import ctypes as _ct
        NV_ENC_LOCK_INPUT_BUFFER_VER = _ct.c_uint32(0x7001000d).value

        _need_pop = False
        _primary = getattr(self, '_primary_ctx', None)
        if _primary is not None and _primary.value is not None:
            try:
                self._libcuda.cuCtxPushCurrent.restype = _ct.c_uint32
                self._libcuda.cuCtxPushCurrent.argtypes = [_ct.c_void_p]
                _r_push = self._libcuda.cuCtxPushCurrent(_primary)
                _need_pop = (_r_push == 0)
            except Exception:
                pass

        try:
            with self._lock:
                if self._encoder.value is None:
                    raise RuntimeError("Encoder not initialized")

                _W = self._width
                nv12_h = self._height + self._height // 2

                _LockInputBufferProto = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p,
                                                        _ct.POINTER(_ct.c_uint8 * 1544))
                _UnlockInputBufferProto = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p,
                                                          _ct.c_void_p)
                _CU_MEMORYTYPE_DEVICE = 2

                # LockInputBuffer
                lock_buf = (_ct.c_uint8 * 1544)()
                _ct.memset(lock_buf, 0, 1544)
                _ct.cast(lock_buf, _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_LOCK_INPUT_BUFFER_VER
                _ct.cast(_ct.byref(lock_buf, 8), _ct.POINTER(_ct.c_void_p))[0] = \
                    self._input_buf_handle

                _lock_addr = self._func_ptrs[_FUNC_IDX["LockInputBuffer"]]
                s = _LockInputBufferProto(_lock_addr)(self._encoder, lock_buf)
                if s != 0:
                    raise RuntimeError("LockInputBuffer failed: %d" % s)

                _raw_map = _ct.cast(_ct.byref(lock_buf, 16), _ct.POINTER(_ct.c_void_p))[0]
                mapped_ptr = _raw_map if isinstance(_raw_map, int) else (_raw_map.value or 0)
                actual_pitch = _ct.cast(_ct.byref(lock_buf, 24), _ct.POINTER(_ct.c_uint32))[0]

                # cuMemcpy2D
                _cpy2d = (_ct.c_uint8 * 128)()
                _ct.memset(_cpy2d, 0, 128)
                src_ptr = nv12_gpu_tensor.data_ptr()
                _ct.cast(_ct.byref(_cpy2d, 16), _ct.POINTER(_ct.c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
                _ct.cast(_ct.byref(_cpy2d, 32), _ct.POINTER(_ct.c_void_p))[0] = _ct.c_void_p(src_ptr)
                _ct.cast(_ct.byref(_cpy2d, 48), _ct.POINTER(_ct.c_size_t))[0] = _W
                _ct.cast(_ct.byref(_cpy2d, 72), _ct.POINTER(_ct.c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
                _ct.cast(_ct.byref(_cpy2d, 88), _ct.POINTER(_ct.c_void_p))[0] = _ct.c_void_p(mapped_ptr)
                _ct.cast(_ct.byref(_cpy2d, 104), _ct.POINTER(_ct.c_size_t))[0] = (
                    actual_pitch if actual_pitch > 0 else _W)
                _ct.cast(_ct.byref(_cpy2d, 112), _ct.POINTER(_ct.c_size_t))[0] = _W
                _ct.cast(_ct.byref(_cpy2d, 120), _ct.POINTER(_ct.c_size_t))[0] = nv12_h
                r = self._libcuda.cuMemcpy2D_v2(_ct.cast(_cpy2d, _ct.c_void_p))
                if r != 0:
                    _UnlockInputBufferProto(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                        self._encoder, self._input_buf_handle)
                    raise RuntimeError("cuMemcpy2D failed: %d" % r)

                _UnlockInputBufferProto(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                    self._encoder, self._input_buf_handle)

                # ★ Per-frame CE: create → use → sync → destroy ★
                _ce = _ct.c_void_p(None)
                self._libcuda.cuEventCreate.restype = _ct.c_uint32
                self._libcuda.cuEventCreate.argtypes = [_ct.POINTER(_ct.c_void_p), _ct.c_uint32]
                self._libcuda.cuEventCreate(_ct.byref(_ce), 0)

                pic_buf = (_ct.c_uint8 * 3360)()
                _ct.memset(pic_buf, 0, 3360)
                _ct.cast(pic_buf, _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
                _ct.cast(_ct.byref(pic_buf, 4), _ct.POINTER(_ct.c_uint32))[0] = _W
                _ct.cast(_ct.byref(pic_buf, 8), _ct.POINTER(_ct.c_uint32))[0] = self._height
                _ct.cast(_ct.byref(pic_buf, 12), _ct.POINTER(_ct.c_uint32))[0] = (
                    actual_pitch if actual_pitch > 0 else _W)
                _ct.cast(_ct.byref(pic_buf, 24), _ct.POINTER(_ct.c_uint64))[0] = self._frame_idx
                _ct.cast(_ct.byref(pic_buf, 40), _ct.POINTER(_ct.c_void_p))[0] = self._input_buf_handle
                _ct.cast(_ct.byref(pic_buf, 48), _ct.POINTER(_ct.c_void_p))[0] = self._bs_handle
                _ct.cast(_ct.byref(pic_buf, 56), _ct.POINTER(_ct.c_void_p))[0] = _ce
                _ct.cast(_ct.byref(pic_buf, 64), _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12
                _ct.cast(_ct.byref(pic_buf, 68), _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_PIC_STRUCT_FRAME
                if force_idr:
                    _ct.cast(_ct.byref(pic_buf, 16), _ct.POINTER(_ct.c_uint32))[0] = 0x2

                _NvEncEPProto = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p,
                                              _ct.POINTER(_ct.c_uint8 * 3360))
                encode_picture = _NvEncEPProto(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                status = encode_picture(self._encoder, _ct.cast(pic_buf,
                                        _ct.POINTER(_ct.c_uint8 * 3360)))
                self._frame_idx += 1

                # cuEventSynchronize → destroy
                if _ce.value is not None:
                    self._libcuda.cuEventSynchronize.restype = _ct.c_uint32
                    self._libcuda.cuEventSynchronize.argtypes = [_ct.c_void_p]
                    self._libcuda.cuEventSynchronize(_ce)
                    self._libcuda.cuEventDestroy.restype = _ct.c_uint32
                    self._libcuda.cuEventDestroy.argtypes = [_ct.c_void_p]
                    self._libcuda.cuEventDestroy(_ce)

                if status == NV_ENC_ERR_NEED_MORE_INPUT:
                    return b""

                h264_data, _ = self._lock_bitstream_with_retry(self._bs_handle)
                if status != NV_ENC_SUCCESS:
                    raise RuntimeError("EncodePicture failed: %d" % status)

                return h264_data
        finally:
            if _need_pop:
                try:
                    _ctx_out = _ct.c_void_p()
                    self._libcuda.cuCtxPopCurrent.restype = _ct.c_uint32
                    self._libcuda.cuCtxPopCurrent.argtypes = [_ct.POINTER(_ct.c_void_p)]
                    self._libcuda.cuCtxPopCurrent(_ct.byref(_ctx_out))
                except Exception:
                    pass


# ============================================================================
# Mode 2: pipeline_depth=4 batch per-frame CE
# ============================================================================
class NVENCEncoderMode2(NVENCEncoder):
    """pipeline_depth=4, batch loop with per-frame CE (v6.4.5.1 attempted pattern)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('pipeline_depth', 4)
        super().__init__(*args, **kwargs)

    def encode_frames_batch_with_ce(self, nv12_tensors, force_idr_first=False):
        """Multi-slot batch: each frame gets its own CE, synchronous per-frame."""
        import ctypes as _ct
        n_frames = len(nv12_tensors)
        if n_frames == 0:
            return []

        _need_pop = False
        _primary = getattr(self, '_primary_ctx', None)
        if _primary is not None and _primary.value is not None:
            try:
                self._libcuda.cuCtxPushCurrent.restype = _ct.c_uint32
                self._libcuda.cuCtxPushCurrent.argtypes = [_ct.c_void_p]
                _r_push = self._libcuda.cuCtxPushCurrent(_primary)
                _need_pop = (_r_push == 0)
            except Exception:
                pass

        try:
            with self._lock:
                results = []
                # [FIX-PERSLOT-IDR] 每槽首帧强制 IDR 以初始化独立 DPB
                _slots_warmed = set()
                _W = self._width
                nv12_h = self._height + self._height // 2

                _LockIB = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p,
                                        _ct.POINTER(_ct.c_uint8 * 1544))
                _UnlockIB = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p, _ct.c_void_p)
                _CUDA_DEV = 2

                for fi in range(n_frames):
                    slot_idx = fi % self._pipeline_depth
                    force_idr = force_idr_first and (slot_idx not in _slots_warmed)
                    slot = self._slots[slot_idx]

                    # LockInputBuffer
                    lock_buf = (_ct.c_uint8 * 1544)()
                    _ct.memset(lock_buf, 0, 1544)
                    _ct.cast(lock_buf, _ct.POINTER(_ct.c_uint32))[0] = \
                        _ct.c_uint32(0x7001000d).value  # NV_ENC_LOCK_INPUT_BUFFER_VER
                    _ct.cast(_ct.byref(lock_buf, 8), _ct.POINTER(_ct.c_void_p))[0] = \
                        slot['input_buf']
                    s = _LockIB(self._func_ptrs[_FUNC_IDX["LockInputBuffer"]])(self._encoder, lock_buf)
                    if s != 0:
                        raise RuntimeError("LockInputBuffer[%d] failed: %d" % (slot_idx, s))
                    _raw = _ct.cast(_ct.byref(lock_buf, 16), _ct.POINTER(_ct.c_void_p))[0]
                    mptr = _raw if isinstance(_raw, int) else (_raw.value or 0)
                    apitch = _ct.cast(_ct.byref(lock_buf, 24), _ct.POINTER(_ct.c_uint32))[0]

                    # cuMemcpy2D
                    _cpy = (_ct.c_uint8 * 128)()
                    _ct.memset(_cpy, 0, 128)
                    src = nv12_tensors[fi].data_ptr()
                    _ct.cast(_ct.byref(_cpy, 16), _ct.POINTER(_ct.c_uint32))[0] = _CUDA_DEV
                    _ct.cast(_ct.byref(_cpy, 32), _ct.POINTER(_ct.c_void_p))[0] = _ct.c_void_p(src)
                    _ct.cast(_ct.byref(_cpy, 48), _ct.POINTER(_ct.c_size_t))[0] = _W
                    _ct.cast(_ct.byref(_cpy, 72), _ct.POINTER(_ct.c_uint32))[0] = _CUDA_DEV
                    _ct.cast(_ct.byref(_cpy, 88), _ct.POINTER(_ct.c_void_p))[0] = _ct.c_void_p(mptr)
                    _ct.cast(_ct.byref(_cpy, 104), _ct.POINTER(_ct.c_size_t))[0] = (
                        apitch if apitch > 0 else _W)
                    _ct.cast(_ct.byref(_cpy, 112), _ct.POINTER(_ct.c_size_t))[0] = _W
                    _ct.cast(_ct.byref(_cpy, 120), _ct.POINTER(_ct.c_size_t))[0] = nv12_h
                    r = self._libcuda.cuMemcpy2D_v2(_ct.cast(_cpy, _ct.c_void_p))
                    if r != 0:
                        _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                            self._encoder, slot['input_buf'])
                        raise RuntimeError("cuMemcpy2D[%d] failed: %d" % (slot_idx, r))

                    _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                        self._encoder, slot['input_buf'])

                    # ★ Per-frame CE (same as Mode 1, but across 4 slots) ★
                    _ce = _ct.c_void_p(None)
                    self._libcuda.cuEventCreate.restype = _ct.c_uint32
                    self._libcuda.cuEventCreate.argtypes = [_ct.POINTER(_ct.c_void_p), _ct.c_uint32]
                    self._libcuda.cuEventCreate(_ct.byref(_ce), 0)

                    pic_buf = (_ct.c_uint8 * 3360)()
                    _ct.memset(pic_buf, 0, 3360)
                    _ct.cast(pic_buf, _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
                    _ct.cast(_ct.byref(pic_buf, 4), _ct.POINTER(_ct.c_uint32))[0] = _W
                    _ct.cast(_ct.byref(pic_buf, 8), _ct.POINTER(_ct.c_uint32))[0] = self._height
                    _ct.cast(_ct.byref(pic_buf, 12), _ct.POINTER(_ct.c_uint32))[0] = (
                        apitch if apitch > 0 else _W)
                    _ct.cast(_ct.byref(pic_buf, 24), _ct.POINTER(_ct.c_uint64))[0] = self._frame_idx
                    _ct.cast(_ct.byref(pic_buf, 40), _ct.POINTER(_ct.c_void_p))[0] = slot['input_buf']
                    _ct.cast(_ct.byref(pic_buf, 48), _ct.POINTER(_ct.c_void_p))[0] = slot['bs_buf']
                    _ct.cast(_ct.byref(pic_buf, 56), _ct.POINTER(_ct.c_void_p))[0] = _ce
                    _ct.cast(_ct.byref(pic_buf, 64), _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12
                    _ct.cast(_ct.byref(pic_buf, 68), _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_PIC_STRUCT_FRAME
                    if force_idr:
                        _ct.cast(_ct.byref(pic_buf, 16), _ct.POINTER(_ct.c_uint32))[0] = 0x2

                    _NvEP = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p,
                                          _ct.POINTER(_ct.c_uint8 * 3360))
                    ep = _NvEP(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                    status = ep(self._encoder, _ct.cast(pic_buf,
                                _ct.POINTER(_ct.c_uint8 * 3360)))
                    self._frame_idx += 1

                    # cuEventSynchronize → destroy
                    if _ce.value is not None:
                        self._libcuda.cuEventSynchronize.restype = _ct.c_uint32
                        self._libcuda.cuEventSynchronize.argtypes = [_ct.c_void_p]
                        self._libcuda.cuEventSynchronize(_ce)
                        self._libcuda.cuEventDestroy.restype = _ct.c_uint32
                        self._libcuda.cuEventDestroy.argtypes = [_ct.c_void_p]
                        self._libcuda.cuEventDestroy(_ce)

                    if status == NV_ENC_ERR_NEED_MORE_INPUT:
                        # Release bitstream buffer even for lookahead frames
                        try:
                            _lb = (_ct.c_uint8 * 1544)()
                            _ct.memset(_lb, 0, 1544)
                            _ct.cast(_lb, _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
                            _ct.cast(_ct.byref(_lb, 8), _ct.POINTER(_ct.c_void_p))[0] = slot['bs_buf']
                            _LockBS_raw = _LockBitstreamProto_raw(self._func_ptrs[_FUNC_IDX["LockBitstream"]])
                            _LockBS_raw(self._encoder, _ct.cast(_lb, _ct.POINTER(_ct.c_uint8 * 1544)))
                            _UnlockBS = _NvEncUnlockBitstreamProto(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])
                            _UnlockBS(self._encoder, slot['bs_buf'])
                        except Exception:
                            pass
                        results.append(b"")
                        continue
                    elif status != NV_ENC_SUCCESS:
                        raise RuntimeError("EncodePicture[%d] failed: %d" % (slot_idx, status))

                    # [FIX-PERSLOT-IDR] 标记该 slot 已初始化 DPB
                    _slots_warmed.add(slot_idx)

                    h264_data, _ = self._lock_bitstream_with_retry(slot['bs_buf'])
                    if not h264_data:
                        results.append(None)  # empty
                    else:
                        results.append(h264_data)

                return results
        finally:
            if _need_pop:
                try:
                    _ctx_out = _ct.c_void_p()
                    self._libcuda.cuCtxPopCurrent.restype = _ct.c_uint32
                    self._libcuda.cuCtxPopCurrent.argtypes = [_ct.POINTER(_ct.c_void_p)]
                    self._libcuda.cuCtxPopCurrent(_ct.byref(_ctx_out))
                except Exception:
                    pass


# ============================================================================
# Mode 3: pipeline_depth=4 PHASE4 async submit/harvest
# ============================================================================
class NVENCEncoderMode3(NVENCEncoder):
    """PHASE4 async: _submit_to_slot (non-blocking, uses pre-created slot['event'])
    + _harvest_slot (blocks on slot['event'], LockBitstream doNotWait=1 + fallback)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('pipeline_depth', 4)
        super().__init__(*args, **kwargs)

    def submit_to_slot(self, slot_idx, nv12_gpu_tensor, force_idr=False):
        """[PHASE4] Copy NV12 to slot's input buffer → async EncodePicture with slot['event'].

        Does NOT block — caller must later harvest_slot() to retrieve bytes.
        Caller must hold self._lock.
        """
        import ctypes as _ct
        slot = self._slots[slot_idx]
        _W = self._width
        nv12_h = self._height + self._height // 2

        _LockIB = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p,
                                _ct.POINTER(_ct.c_uint8 * 1544))
        _UnlockIB = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p, _ct.c_void_p)
        _CUDA_DEV = 2

        # LockInputBuffer
        lock_buf = (_ct.c_uint8 * 1544)()
        _ct.memset(lock_buf, 0, 1544)
        _ct.cast(lock_buf, _ct.POINTER(_ct.c_uint32))[0] = _ct.c_uint32(0x7001000d).value
        _ct.cast(_ct.byref(lock_buf, 8), _ct.POINTER(_ct.c_void_p))[0] = slot['input_buf']
        s = _LockIB(self._func_ptrs[_FUNC_IDX["LockInputBuffer"]])(self._encoder, lock_buf)
        if s != 0:
            raise RuntimeError("LockInputBuffer[%d] failed: %d" % (slot_idx, s))
        _raw = _ct.cast(_ct.byref(lock_buf, 16), _ct.POINTER(_ct.c_void_p))[0]
        mptr = _raw if isinstance(_raw, int) else (_raw.value or 0)
        apitch = _ct.cast(_ct.byref(lock_buf, 24), _ct.POINTER(_ct.c_uint32))[0]

        # cuMemcpy2D
        _cpy = (_ct.c_uint8 * 128)()
        _ct.memset(_cpy, 0, 128)
        src = nv12_gpu_tensor.data_ptr()
        _ct.cast(_ct.byref(_cpy, 16), _ct.POINTER(_ct.c_uint32))[0] = _CUDA_DEV
        _ct.cast(_ct.byref(_cpy, 32), _ct.POINTER(_ct.c_void_p))[0] = _ct.c_void_p(src)
        _ct.cast(_ct.byref(_cpy, 48), _ct.POINTER(_ct.c_size_t))[0] = _W
        _ct.cast(_ct.byref(_cpy, 72), _ct.POINTER(_ct.c_uint32))[0] = _CUDA_DEV
        _ct.cast(_ct.byref(_cpy, 88), _ct.POINTER(_ct.c_void_p))[0] = _ct.c_void_p(mptr)
        _ct.cast(_ct.byref(_cpy, 104), _ct.POINTER(_ct.c_size_t))[0] = (
            apitch if apitch > 0 else _W)
        _ct.cast(_ct.byref(_cpy, 112), _ct.POINTER(_ct.c_size_t))[0] = _W
        _ct.cast(_ct.byref(_cpy, 120), _ct.POINTER(_ct.c_size_t))[0] = nv12_h
        r = self._libcuda.cuMemcpy2D_v2(_ct.cast(_cpy, _ct.c_void_p))
        if r != 0:
            _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                self._encoder, slot['input_buf'])
            raise RuntimeError("cuMemcpy2D[%d] failed: %d" % (slot_idx, r))

        _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
            self._encoder, slot['input_buf'])

        # ★ PHASE4: EncodePicture with pre-created slot['event'] ★
        pic_buf = (_ct.c_uint8 * 3360)()
        _ct.memset(pic_buf, 0, 3360)
        _ct.cast(pic_buf, _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
        _ct.cast(_ct.byref(pic_buf, 4), _ct.POINTER(_ct.c_uint32))[0] = _W
        _ct.cast(_ct.byref(pic_buf, 8), _ct.POINTER(_ct.c_uint32))[0] = self._height
        _ct.cast(_ct.byref(pic_buf, 12), _ct.POINTER(_ct.c_uint32))[0] = (
            apitch if apitch > 0 else _W)
        _ct.cast(_ct.byref(pic_buf, 24), _ct.POINTER(_ct.c_uint64))[0] = self._frame_idx
        _ct.cast(_ct.byref(pic_buf, 40), _ct.POINTER(_ct.c_void_p))[0] = slot['input_buf']
        _ct.cast(_ct.byref(pic_buf, 48), _ct.POINTER(_ct.c_void_p))[0] = slot['bs_buf']
        # ★ Use pre-created slot event (persistent, reused across frames) ★
        _ct.cast(_ct.byref(pic_buf, 56), _ct.POINTER(_ct.c_void_p))[0] = slot['event']
        _ct.cast(_ct.byref(pic_buf, 64), _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12
        _ct.cast(_ct.byref(pic_buf, 68), _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_PIC_STRUCT_FRAME
        if force_idr:
            _ct.cast(_ct.byref(pic_buf, 16), _ct.POINTER(_ct.c_uint32))[0] = 0x2

        _NvEP = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p,
                              _ct.POINTER(_ct.c_uint8 * 3360))
        ep = _NvEP(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
        status = ep(self._encoder, _ct.cast(pic_buf,
                    _ct.POINTER(_ct.c_uint8 * 3360)))
        self._frame_idx += 1

        if status != NV_ENC_SUCCESS and status != NV_ENC_ERR_NEED_MORE_INPUT:
            raise RuntimeError("EncodePicture[%d] failed: %d" % (slot_idx, status))

    def harvest_slot(self, slot_idx):
        """[PHASE4] Wait for slot's completionEvent, then retrieve H.264 bytes.

        Blocks on cuEventSynchronize → LockBitstream(doNotWait=1 fast path) →
        if failed, LockBitstream(doNotWait=0 fallback). Returns (h264_data, status).
        Caller must hold self._lock.
        """
        import ctypes as _ct
        slot = self._slots[slot_idx]

        # ★ cuEventSynchronize on pre-created slot event ★
        r = self._libcuda.cuEventSynchronize(slot['event'])
        if r != 0:
            raise RuntimeError("cuEventSynchronize[%d] failed: %d" % (slot_idx, r))

        # LockBitstream — doNotWait=1 first (fast path)
        lock_raw = (_ct.c_uint8 * 1544)()
        _ct.memset(lock_raw, 0, 1544)
        _ct.cast(lock_raw, _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
        _ct.cast(_ct.byref(lock_raw, 4), _ct.POINTER(_ct.c_uint32))[0] = 1  # doNotWait=1
        _ct.cast(_ct.byref(lock_raw, 8), _ct.POINTER(_ct.c_void_p))[0] = slot['bs_buf']

        lock_fn = _LockBitstreamProto_raw(self._func_ptrs[_FUNC_IDX["LockBitstream"]])
        bs_status = lock_fn(self._encoder, _ct.cast(lock_raw,
                            _ct.POINTER(_ct.c_uint8 * 1544)))

        # ★ PHASE4 fallback: doNotWait=1 failed → retry doNotWait=0 ★
        if bs_status != NV_ENC_SUCCESS:
            _ct.memset(lock_raw, 0, 1544)
            _ct.cast(lock_raw, _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
            _ct.cast(_ct.byref(lock_raw, 4), _ct.POINTER(_ct.c_uint32))[0] = 0  # blocking
            _ct.cast(_ct.byref(lock_raw, 8), _ct.POINTER(_ct.c_void_p))[0] = slot['bs_buf']
            bs_status = lock_fn(self._encoder, _ct.cast(lock_raw,
                                _ct.POINTER(_ct.c_uint8 * 1544)))

        h264_data = b""
        if bs_status == NV_ENC_SUCCESS:
            bitstream_size = _ct.cast(_ct.byref(lock_raw, 36), _ct.POINTER(_ct.c_uint32))[0]
            _raw_bsptr = _ct.cast(_ct.byref(lock_raw, 56), _ct.POINTER(_ct.c_void_p))[0]
            ptr_val = _raw_bsptr if isinstance(_raw_bsptr, int) else (_raw_bsptr.value or 0)
            if bitstream_size > 0 and ptr_val:
                buf_type = _ct.c_uint8 * bitstream_size
                h264_data = bytes(buf_type.from_address(ptr_val))

            _UnlockBS = _NvEncUnlockBitstreamProto(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])
            _UnlockBS(self._encoder, slot['bs_buf'])

        return h264_data, bs_status

    def encode_frames_batch_phase4(self, nv12_tensors, force_idr_first=False):
        """PHASE4 async pipeline: submit→harvest with 4-slot staggered overlap.

        For pipeline_depth=4: submit first 4 frames (non-blocking), then for each
        subsequent frame, submit frame N and harvest frame N-4 in the same iteration,
        achieving true NVENC pipeline overlap.
        """
        import ctypes as _ct
        n_frames = len(nv12_tensors)
        if n_frames == 0:
            return []

        _need_pop = False
        _primary = getattr(self, '_primary_ctx', None)
        if _primary is not None and _primary.value is not None:
            try:
                self._libcuda.cuCtxPushCurrent.restype = _ct.c_uint32
                self._libcuda.cuCtxPushCurrent.argtypes = [_ct.c_void_p]
                _r_push = self._libcuda.cuCtxPushCurrent(_primary)
                _need_pop = (_r_push == 0)
            except Exception:
                pass

        try:
            with self._lock:
                results = [None] * n_frames
                # [FIX-PERSLOT-IDR] 每槽首帧强制 IDR 以初始化独立 DPB
                _slots_warmed = set()
                _pd = self._pipeline_depth  # = 4

                # Phase 1: Fill pipeline — submit first pd frames (non-blocking)
                for fi in range(min(_pd, n_frames)):
                    slot_idx = fi % _pd
                    force_idr = force_idr_first and (slot_idx not in _slots_warmed)
                    self.submit_to_slot(slot_idx, nv12_tensors[fi], force_idr)

                # Phase 2: Steady state — submit fi, harvest fi-pd
                for fi in range(_pd, n_frames):
                    slot_idx = fi % _pd
                    harvest_slot_idx = (fi - _pd) % _pd
                    force_idr = force_idr_first and (slot_idx not in _slots_warmed)

                    # Submit new frame
                    self.submit_to_slot(slot_idx, nv12_tensors[fi], force_idr)

                    # Harvest completed frame from pd steps ago
                    h264_data, bs_status = self.harvest_slot(harvest_slot_idx)
                    if bs_status != NV_ENC_SUCCESS:
                        results[fi - _pd] = None  # empty
                    else:
                        results[fi - _pd] = h264_data
                        _slots_warmed.add(harvest_slot_idx)

                # Phase 3: Drain — harvest remaining pd frames
                for drain_i in range(_pd):
                    fi = n_frames - _pd + drain_i
                    if fi < 0:
                        continue
                    harvest_slot_idx = fi % _pd
                    h264_data, bs_status = self.harvest_slot(harvest_slot_idx)
                    if bs_status != NV_ENC_SUCCESS:
                        results[fi] = None
                    else:
                        results[fi] = h264_data
                        _slots_warmed.add(harvest_slot_idx)

                return results
        finally:
            if _need_pop:
                try:
                    _ctx_out = _ct.c_void_p()
                    self._libcuda.cuCtxPopCurrent.restype = _ct.c_uint32
                    self._libcuda.cuCtxPopCurrent.argtypes = [_ct.POINTER(_ct.c_void_p)]
                    self._libcuda.cuCtxPopCurrent(_ct.byref(_ctx_out))
                except Exception:
                    pass


# ============================================================================
# Test runners
# ============================================================================
def _summarize_empty(label, empty_frames):
    """Print a single compact line for empty frames instead of one per frame.

    Detect well-known patterns, otherwise show count + first/last N ranges.
    """
    if not empty_frames:
        return
    n = len(empty_frames)
    # Group into runs of consecutive fi
    runs = []
    cur_start = empty_frames[0][0]
    cur_end = empty_frames[0][0]
    cur_slots = {empty_frames[0][1]}
    for fi, slot in empty_frames[1:]:
        if fi == cur_end + 1:
            cur_end = fi
        else:
            runs.append((cur_start, cur_end, cur_slots))
            cur_start = fi
            cur_end = fi
            cur_slots = {slot}
        cur_slots.add(slot)
    runs.append((cur_start, cur_end, cur_slots))

    # Build compact description
    if len(runs) == 1:
        _desc = f"fi={runs[0][0]}..{runs[0][1]}"
    elif len(runs) <= 6:
        _desc = ", ".join(f"{s}..{e}" if s != e else str(s) for s, e, _ in runs)
    else:
        _desc = f"fi={runs[0][0]}..{runs[-1][1]} ({len(runs)} bursts)"

    # Slot info
    _all_slots = set()
    for _, _, slots in runs:
        _all_slots |= slots
    _slot_desc = f" slot={','.join(map(str, sorted(_all_slots)))}" if len(_all_slots) < 4 else " slots=all"

    _interval = runs[1][0] - runs[0][0] if len(runs) > 1 else None
    _interval_str = f" every {_interval}" if _interval and _interval > 0 else ""

    print(f"  [{label}] {n} empty frames{_interval_str}: {_desc}{_slot_desc}", flush=True)


def _flush_and_count(enc):
    """Flush encoder and return number of frames recovered."""
    flush_data = enc.flush()
    if not flush_data:
        return 0, 0
    # Use NAL start code counting (same as production _loop())
    _nal4 = flush_data.count(b'\x00\x00\x00\x01')
    _nal3 = flush_data.count(b'\x00\x00\x01')
    return max(1, (_nal4 + _nal3) // 2), len(flush_data)



# ============================================================================

class NVENCEncoderMode4(NVENCEncoder):
    """PHASE4 async with per-frame CE (no event reuse) → submit+harvest with unique CE per frame."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('pipeline_depth', 4)
        super().__init__(*args, **kwargs)

    def submit_to_slot_pfce(self, slot_idx, nv12_gpu_tensor, force_idr=False):
        """Lock→Copy→Unlock + EncodePicture with per-frame CE. Returns (ce_handle, ep_status)."""
        import ctypes as _ct
        slot = self._slots[slot_idx]
        _W = self._width
        nv12_h = self._height + self._height // 2
        _CUDA_DEV = 2

        _LockIB = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p,
                                _ct.POINTER(_ct.c_uint8 * 1544))
        _UnlockIB = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p, _ct.c_void_p)

        # LockInputBuffer
        lock_buf = (_ct.c_uint8 * 1544)()
        _ct.memset(lock_buf, 0, 1544)
        _ct.cast(lock_buf, _ct.POINTER(_ct.c_uint32))[0] = _ct.c_uint32(0x7001000d).value
        _ct.cast(_ct.byref(lock_buf, 8), _ct.POINTER(_ct.c_void_p))[0] = slot['input_buf']
        s = _LockIB(self._func_ptrs[_FUNC_IDX["LockInputBuffer"]])(self._encoder, lock_buf)
        if s != 0:
            raise RuntimeError("LockInputBuffer[%d] failed: %d" % (slot_idx, s))
        _raw = _ct.cast(_ct.byref(lock_buf, 16), _ct.POINTER(_ct.c_void_p))[0]
        mptr = _raw if isinstance(_raw, int) else (_raw.value or 0)
        apitch = _ct.cast(_ct.byref(lock_buf, 24), _ct.POINTER(_ct.c_uint32))[0]

        # cuMemcpy2D
        _cpy = (_ct.c_uint8 * 128)()
        _ct.memset(_cpy, 0, 128)
        src = nv12_gpu_tensor.data_ptr()
        _ct.cast(_ct.byref(_cpy, 16), _ct.POINTER(_ct.c_uint32))[0] = _CUDA_DEV
        _ct.cast(_ct.byref(_cpy, 32), _ct.POINTER(_ct.c_void_p))[0] = _ct.c_void_p(src)
        _ct.cast(_ct.byref(_cpy, 48), _ct.POINTER(_ct.c_size_t))[0] = _W
        _ct.cast(_ct.byref(_cpy, 72), _ct.POINTER(_ct.c_uint32))[0] = _CUDA_DEV
        _ct.cast(_ct.byref(_cpy, 88), _ct.POINTER(_ct.c_void_p))[0] = _ct.c_void_p(mptr)
        _ct.cast(_ct.byref(_cpy, 104), _ct.POINTER(_ct.c_size_t))[0] = (
            apitch if apitch > 0 else _W)
        _ct.cast(_ct.byref(_cpy, 112), _ct.POINTER(_ct.c_size_t))[0] = _W
        _ct.cast(_ct.byref(_cpy, 120), _ct.POINTER(_ct.c_size_t))[0] = nv12_h
        r = self._libcuda.cuMemcpy2D_v2(_ct.cast(_cpy, _ct.c_void_p))
        if r != 0:
            _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(self._encoder, slot['input_buf'])
            raise RuntimeError("cuMemcpy2D[%d] failed: %d" % (slot_idx, r))

        _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(self._encoder, slot['input_buf'])

        # ★ Per-frame CE: create fresh event (no slot event reuse) ★
        _ce = _ct.c_void_p(None)
        self._libcuda.cuEventCreate.restype = _ct.c_uint32
        self._libcuda.cuEventCreate.argtypes = [_ct.POINTER(_ct.c_void_p), _ct.c_uint32]
        self._libcuda.cuEventCreate(_ct.byref(_ce), 0)

        pic_buf = (_ct.c_uint8 * 3360)()
        _ct.memset(pic_buf, 0, 3360)
        _ct.cast(pic_buf, _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
        _ct.cast(_ct.byref(pic_buf, 4), _ct.POINTER(_ct.c_uint32))[0] = _W
        _ct.cast(_ct.byref(pic_buf, 8), _ct.POINTER(_ct.c_uint32))[0] = self._height
        _ct.cast(_ct.byref(pic_buf, 12), _ct.POINTER(_ct.c_uint32))[0] = (
            apitch if apitch > 0 else _W)
        _ct.cast(_ct.byref(pic_buf, 24), _ct.POINTER(_ct.c_uint64))[0] = self._frame_idx
        _ct.cast(_ct.byref(pic_buf, 40), _ct.POINTER(_ct.c_void_p))[0] = slot['input_buf']
        _ct.cast(_ct.byref(pic_buf, 48), _ct.POINTER(_ct.c_void_p))[0] = slot['bs_buf']
        _ct.cast(_ct.byref(pic_buf, 56), _ct.POINTER(_ct.c_void_p))[0] = _ce  # per-frame CE
        _ct.cast(_ct.byref(pic_buf, 64), _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12
        _ct.cast(_ct.byref(pic_buf, 68), _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_PIC_STRUCT_FRAME
        if force_idr:
            _ct.cast(_ct.byref(pic_buf, 16), _ct.POINTER(_ct.c_uint32))[0] = 0x2

        _NvEP = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p,
                              _ct.POINTER(_ct.c_uint8 * 3360))
        ep = _NvEP(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
        status = ep(self._encoder, _ct.cast(pic_buf, _ct.POINTER(_ct.c_uint8 * 3360)))
        self._frame_idx += 1

        return _ce, status  # return CE for later harvest

    def harvest_slot_pfce(self, slot_idx, ce_handle):
        """Wait for per-frame CE → LockBitstream → destroy CE. Returns (h264_data, status)."""
        import ctypes as _ct
        slot = self._slots[slot_idx]

        # cuEventSynchronize on per-frame CE
        if ce_handle.value is not None:
            self._libcuda.cuEventSynchronize.restype = _ct.c_uint32
            self._libcuda.cuEventSynchronize.argtypes = [_ct.c_void_p]
            sync_r = self._libcuda.cuEventSynchronize(ce_handle)
            self._libcuda.cuEventDestroy.restype = _ct.c_uint32
            self._libcuda.cuEventDestroy.argtypes = [_ct.c_void_p]
            self._libcuda.cuEventDestroy(ce_handle)

        # LockBitstream: doNotWait=1 fast path, then doNotWait=0 fallback
        _LockBS_raw = _LockBitstreamProto_raw
        _UnlockBS = _NvEncUnlockBitstreamProto

        for _attempt in range(2):
            do_not_wait = 1 if _attempt == 0 else 0
            _lb = (_ct.c_uint8 * 1544)()
            _ct.memset(_lb, 0, 1544)
            _ct.cast(_lb, _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
            _ct.cast(_ct.byref(_lb, 4), _ct.POINTER(_ct.c_uint32))[0] = do_not_wait
            _ct.cast(_ct.byref(_lb, 8), _ct.POINTER(_ct.c_void_p))[0] = slot['bs_buf']

            bs_status = _LockBS_raw(self._func_ptrs[_FUNC_IDX["LockBitstream"]])(
                self._encoder, _ct.cast(_lb, _ct.POINTER(_ct.c_uint8 * 1544)))

            bitstream_size = _ct.cast(_ct.byref(_lb, 36), _ct.POINTER(_ct.c_uint32))[0]
            if bitstream_size > 0:
                _raw_bsptr = _ct.cast(_ct.byref(_lb, 56), _ct.POINTER(_ct.c_void_p))[0]
                bsptr = _raw_bsptr if isinstance(_raw_bsptr, int) else (_raw_bsptr.value or 0)
                if bsptr:
                    buf_type = _ct.c_uint8 * bitstream_size
                    h264_data = bytes(buf_type.from_address(bsptr))
                    _UnlockBS(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(self._encoder, slot['bs_buf'])
                    return h264_data, bs_status

            _UnlockBS(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(self._encoder, slot['bs_buf'])
            if do_not_wait == 1 and bs_status != NV_ENC_SUCCESS:
                continue  # try doNotWait=0
            break

        return b"", bs_status

    def encode_frames_batch_phase4_pfce(self, nv12_tensors, force_idr_first=False):
        """PHASE4 pipeline with per-frame CE: submit with fresh CE each frame, harvest independently."""
        import ctypes as _ct
        n_frames = len(nv12_tensors)
        if n_frames == 0:
            return []

        _need_pop = False
        _primary = getattr(self, '_primary_ctx', None)
        if _primary is not None and _primary.value is not None:
            try:
                self._libcuda.cuCtxPushCurrent.restype = _ct.c_uint32
                self._libcuda.cuCtxPushCurrent.argtypes = [_ct.c_void_p]
                _r_push = self._libcuda.cuCtxPushCurrent(_primary)
                _need_pop = (_r_push == 0)
            except Exception:
                pass

        try:
            with self._lock:
                results = [None] * n_frames
                _pd = self._pipeline_depth

                # Phase 1: Fill pipeline — submit first pd frames (non-blocking)
                _pending = {}  # slot_idx → ce_handle
                for fi in range(min(_pd, n_frames)):
                    slot_idx = fi % _pd
                    force_idr = force_idr_first and fi == 0
                    ce_h, ep_s = self.submit_to_slot_pfce(slot_idx, nv12_tensors[fi], force_idr)
                    _pending[slot_idx] = ce_h
                    if ep_s != NV_ENC_SUCCESS and ep_s != NV_ENC_ERR_NEED_MORE_INPUT:
                        raise RuntimeError("EncodePicture[%d] failed: %d" % (slot_idx, ep_s))

                # Phase 2: Steady state — submit fi, harvest fi-pd
                for fi in range(_pd, n_frames):
                    slot_idx = fi % _pd
                    harvest_slot_idx = (fi - _pd) % _pd
                    force_idr = force_idr_first and fi == 0

                    # Submit new frame
                    ce_h, ep_s = self.submit_to_slot_pfce(slot_idx, nv12_tensors[fi], force_idr)
                    _pending[slot_idx] = ce_h
                    if ep_s != NV_ENC_SUCCESS and ep_s != NV_ENC_ERR_NEED_MORE_INPUT:
                        raise RuntimeError("EncodePicture[%d] failed: %d" % (slot_idx, ep_s))

                    # Harvest completed frame from pd steps ago
                    ce_to_harvest = _pending.get(harvest_slot_idx)
                    if ce_to_harvest is not None:
                        h264_data, bs_status = self.harvest_slot_pfce(harvest_slot_idx, ce_to_harvest)
                        del _pending[harvest_slot_idx]
                        if bs_status != NV_ENC_SUCCESS:
                            results[fi - _pd] = None
                        else:
                            results[fi - _pd] = h264_data

                # Phase 3: Drain — harvest remaining pd frames
                for drain_i in range(_pd):
                    fi = n_frames - _pd + drain_i
                    if fi < 0:
                        continue
                    harvest_slot_idx = fi % _pd
                    ce_to_harvest = _pending.get(harvest_slot_idx)
                    if ce_to_harvest is not None:
                        h264_data, bs_status = self.harvest_slot_pfce(harvest_slot_idx, ce_to_harvest)
                        del _pending[harvest_slot_idx]
                        if bs_status != NV_ENC_SUCCESS:
                            results[fi] = None
                        else:
                            results[fi] = h264_data

                return results
        finally:
            if _need_pop:
                try:
                    _ctx_out = _ct.c_void_p()
                    self._libcuda.cuCtxPopCurrent.restype = _ct.c_uint32
                    self._libcuda.cuCtxPopCurrent.argtypes = [_ct.POINTER(_ct.c_void_p)]
                    self._libcuda.cuCtxPopCurrent(_ct.byref(_ctx_out))
                except Exception:
                    pass



class NVENCEncoderMode5(NVENCEncoder):
    """CE pipeline: per-frame CE + deferred harvest. Solves LA + pipe=4 routing issue.

    Key innovation: LockBitstream is deferred until the NEXT time the same slot
    is used (= pd frames later). At that point, the per-frame CE has fired and
    the frame's data is safely in slot[N%4].bs_buf.
    """

    def __init__(self, *args, pipeline_depth=4, la_depth=8, **kwargs):
        kwargs.setdefault('pipeline_depth', pipeline_depth)
        kwargs.setdefault('la_depth', la_depth)
        super().__init__(*args, **kwargs)

    def encode_frames_batch_ce_pipeline(self, nv12_tensors, force_idr_first=False):
        """Pipe=4 CE pipeline: submit with per-frame CE, harvest before slot reuse."""
        import ctypes as _ct
        n_frames = len(nv12_tensors)
        if n_frames == 0:
            return []

        _need_pop = False
        _primary = getattr(self, '_primary_ctx', None)
        if _primary is not None and _primary.value is not None:
            try:
                self._libcuda.cuCtxPushCurrent.restype = _ct.c_uint32
                self._libcuda.cuCtxPushCurrent.argtypes = [_ct.c_void_p]
                _r_push = self._libcuda.cuCtxPushCurrent(_primary)
                _need_pop = (_r_push == 0)
            except Exception:
                pass

        try:
            with self._lock:
                pd = self._pipeline_depth
                _W = self._width
                nv12_h = self._height + self._height // 2
                results = [None] * n_frames

                # Per-slot pending state: (ce_handle, frame_index, submit_status)
                _slot_pending = [None] * pd

                _LockIB = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p,
                                        _ct.POINTER(_ct.c_uint8 * 1544))
                _UnlockIB = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p, _ct.c_void_p)
                _CUDA_DEV = 2

                for fi in range(n_frames):
                    slot_idx = fi % pd
                    slot = self._slots[slot_idx]
                    force_idr = force_idr_first and fi == 0

                    # ── Phase 1: Harvest pending frame from this slot (if any) ──
                    if _slot_pending[slot_idx] is not None:
                        _prev_ce, _prev_fi, _prev_ep_s = _slot_pending[slot_idx]
                        # Wait for CE → data is now in slot.bs_buf
                        if _prev_ce.value is not None:
                            self._libcuda.cuEventSynchronize.restype = _ct.c_uint32
                            self._libcuda.cuEventSynchronize.argtypes = [_ct.c_void_p]
                            self._libcuda.cuEventSynchronize(_prev_ce)
                            self._libcuda.cuEventDestroy.restype = _ct.c_uint32
                            self._libcuda.cuEventDestroy.argtypes = [_ct.c_void_p]
                            self._libcuda.cuEventDestroy(_prev_ce)
                        h264_data, _bs = self._lock_bitstream_with_retry(slot['bs_buf'])
                        if h264_data:
                            results[_prev_fi] = h264_data
                        elif _prev_ep_s == NV_ENC_ERR_NEED_MORE_INPUT:
                            # LA帧：bitstream为空是预期行为，NVENC缓冲做码控
                            results[_prev_fi] = b""
                        # else: results[_prev_fi] stays None → 真正的空帧
                        _slot_pending[slot_idx] = None

                    # ── Phase 2: Submit new frame ──
                    # LockInputBuffer
                    lock_buf = (_ct.c_uint8 * 1544)()
                    _ct.memset(lock_buf, 0, 1544)
                    _ct.cast(lock_buf, _ct.POINTER(_ct.c_uint32))[0] = _ct.c_uint32(0x7001000d).value
                    _ct.cast(_ct.byref(lock_buf, 8), _ct.POINTER(_ct.c_void_p))[0] = slot['input_buf']
                    s = _LockIB(self._func_ptrs[_FUNC_IDX["LockInputBuffer"]])(self._encoder, lock_buf)
                    if s != 0:
                        raise RuntimeError("LockIB[%d] failed: %d" % (slot_idx, s))
                    _raw = _ct.cast(_ct.byref(lock_buf, 16), _ct.POINTER(_ct.c_void_p))[0]
                    mptr = _raw if isinstance(_raw, int) else (_raw.value or 0)
                    apitch = _ct.cast(_ct.byref(lock_buf, 24), _ct.POINTER(_ct.c_uint32))[0]

                    # cuMemcpy2D
                    _cpy = (_ct.c_uint8 * 128)()
                    _ct.memset(_cpy, 0, 128)
                    src = nv12_tensors[fi].data_ptr()
                    _ct.cast(_ct.byref(_cpy, 16), _ct.POINTER(_ct.c_uint32))[0] = _CUDA_DEV
                    _ct.cast(_ct.byref(_cpy, 32), _ct.POINTER(_ct.c_void_p))[0] = _ct.c_void_p(src)
                    _ct.cast(_ct.byref(_cpy, 48), _ct.POINTER(_ct.c_size_t))[0] = _W
                    _ct.cast(_ct.byref(_cpy, 72), _ct.POINTER(_ct.c_uint32))[0] = _CUDA_DEV
                    _ct.cast(_ct.byref(_cpy, 88), _ct.POINTER(_ct.c_void_p))[0] = _ct.c_void_p(mptr)
                    _ct.cast(_ct.byref(_cpy, 104), _ct.POINTER(_ct.c_size_t))[0] = (
                        apitch if apitch > 0 else _W)
                    _ct.cast(_ct.byref(_cpy, 112), _ct.POINTER(_ct.c_size_t))[0] = _W
                    _ct.cast(_ct.byref(_cpy, 120), _ct.POINTER(_ct.c_size_t))[0] = nv12_h
                    r = self._libcuda.cuMemcpy2D_v2(_ct.cast(_cpy, _ct.c_void_p))
                    if r != 0:
                        _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                            self._encoder, slot['input_buf'])
                        raise RuntimeError("cuMemcpy2D[%d] failed: %d" % (slot_idx, r))
                    _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                        self._encoder, slot['input_buf'])

                    # ★ Per-frame CE (create fresh event) ★
                    _ce = _ct.c_void_p(None)
                    self._libcuda.cuEventCreate.restype = _ct.c_uint32
                    self._libcuda.cuEventCreate.argtypes = [_ct.POINTER(_ct.c_void_p), _ct.c_uint32]
                    self._libcuda.cuEventCreate(_ct.byref(_ce), 0)

                    pic_buf = (_ct.c_uint8 * 3360)()
                    _ct.memset(pic_buf, 0, 3360)
                    _ct.cast(pic_buf, _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
                    _ct.cast(_ct.byref(pic_buf, 4), _ct.POINTER(_ct.c_uint32))[0] = _W
                    _ct.cast(_ct.byref(pic_buf, 8), _ct.POINTER(_ct.c_uint32))[0] = self._height
                    _ct.cast(_ct.byref(pic_buf, 12), _ct.POINTER(_ct.c_uint32))[0] = (
                        apitch if apitch > 0 else _W)
                    _ct.cast(_ct.byref(pic_buf, 24), _ct.POINTER(_ct.c_uint64))[0] = self._frame_idx
                    _ct.cast(_ct.byref(pic_buf, 40), _ct.POINTER(_ct.c_void_p))[0] = slot['input_buf']
                    _ct.cast(_ct.byref(pic_buf, 48), _ct.POINTER(_ct.c_void_p))[0] = slot['bs_buf']
                    _ct.cast(_ct.byref(pic_buf, 56), _ct.POINTER(_ct.c_void_p))[0] = _ce
                    _ct.cast(_ct.byref(pic_buf, 64), _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12
                    _ct.cast(_ct.byref(pic_buf, 68), _ct.POINTER(_ct.c_uint32))[0] = NV_ENC_PIC_STRUCT_FRAME
                    if force_idr:
                        _ct.cast(_ct.byref(pic_buf, 16), _ct.POINTER(_ct.c_uint32))[0] = 0x2

                    _NvEP = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p,
                                          _ct.POINTER(_ct.c_uint8 * 3360))
                    ep = _NvEP(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                    _ep_s = ep(self._encoder, _ct.cast(pic_buf, _ct.POINTER(_ct.c_uint8 * 3360)))
                    self._frame_idx += 1

                    if _ep_s != NV_ENC_SUCCESS and _ep_s != NV_ENC_ERR_NEED_MORE_INPUT:
                        raise RuntimeError("EncodePicture[%d] failed: %d" % (slot_idx, _ep_s))

                    _slot_pending[slot_idx] = (_ce, fi, _ep_s)
                # ── Phase 3: Drain remaining pending slots ──
                for slot_idx in range(pd):
                    if _slot_pending[slot_idx] is not None:
                        _pending_ce, _pending_fi, _pending_ep_s = _slot_pending[slot_idx]
                        if _pending_ce.value is not None:
                            self._libcuda.cuEventSynchronize.restype = _ct.c_uint32
                            self._libcuda.cuEventSynchronize.argtypes = [_ct.c_void_p]
                            self._libcuda.cuEventSynchronize(_pending_ce)
                            self._libcuda.cuEventDestroy.restype = _ct.c_uint32
                            self._libcuda.cuEventDestroy.argtypes = [_ct.c_void_p]
                            self._libcuda.cuEventDestroy(_pending_ce)
                        h264_data, _bs = self._lock_bitstream_with_retry(
                            self._slots[slot_idx]['bs_buf'])
                        if h264_data:
                            results[_pending_fi] = h264_data
                        elif _pending_ep_s == NV_ENC_ERR_NEED_MORE_INPUT:
                            # LA帧：bitstream为空是预期行为，NVENC缓冲做码控
                            results[_pending_fi] = b""
                        # else: results[_pending_fi] stays None → 真正的空帧
                        _slot_pending[slot_idx] = None

                return results
        finally:
            if _need_pop:
                try:
                    _ctx_out = _ct.c_void_p()
                    self._libcuda.cuCtxPopCurrent.restype = _ct.c_uint32
                    self._libcuda.cuCtxPopCurrent.argtypes = [_ct.POINTER(_ct.c_void_p)]
                    self._libcuda.cuCtxPopCurrent(_ct.byref(_ctx_out))
                except Exception:
                    pass



# Technique Registry — encoding strategy × pipeline_depth × lookahead
# ============================================================================
TECHNIQUES = {
    'sync-batch': {
        'encoder_cls': NVENCEncoder,
        'encode_fn_name': 'encode_frames_batch',
        'warmup_fn_name': 'encode_frame',
        'per_frame': False,
        'ce_info': 'none',
        'is_baseline': True,
        'desc': 'sync batch (NO CE) — v6.4.5.1 production baseline',
    },
    'single-ce': {
        'encoder_cls': NVENCEncoderMode1,
        'encode_fn_name': 'encode_frame_with_ce',
        'warmup_fn_name': 'encode_frame_with_ce',
        'per_frame': True,
        'ce_info': 'per-frame',
        'is_baseline': False,
        'desc': 'single-slot CE — per-frame create→sync→destroy',
    },
    'batch-ce': {
        'encoder_cls': NVENCEncoderMode2,
        'encode_fn_name': 'encode_frames_batch_with_ce',
        'warmup_fn_name': 'encode_frame',
        'per_frame': False,
        'ce_info': 'batch',
        'is_baseline': False,
        'desc': 'batch per-frame CE — sync harvest across all slots',
    },
    'phase4-slot': {
        'encoder_cls': NVENCEncoderMode3,
        'encode_fn_name': 'encode_frames_batch_phase4',
        'warmup_fn_name': 'encode_frame',
        'per_frame': False,
        'ce_info': 'slot-reuse',
        'is_baseline': False,
        'desc': 'PHASE4 async — slot-reuse event (broken for pipe=4)',
    },
    'phase4-pfce': {
        'encoder_cls': NVENCEncoderMode4,
        'encode_fn_name': 'encode_frames_batch_phase4_pfce',
        'warmup_fn_name': 'encode_frame',
        'per_frame': False,
        'ce_info': 'per-frame',
        'is_baseline': False,
        'desc': 'PHASE4 async — per-frame CE (no event reuse)',
    },
    'ce-pipeline': {
        'encoder_cls': NVENCEncoderMode5,
        'encode_fn_name': 'encode_frames_batch_ce_pipeline',
        'warmup_fn_name': 'encode_frame',
        'per_frame': False,
        'ce_info': 'deferred',
        'is_baseline': False,
        'desc': 'CE-pipeline — defer LockBitstream to next slot reuse',
    },
}


# ============================================================================
# Unified test runner — dispatches technique × pipe × la
# ============================================================================
def run_technique(tech_key, W, H, fps, rate_mode, la_depth, n_frames, n_warmup,
                  pipeline_depth=4, preset="p1", qp=23):
    """Run one (technique, pipe, la) combination via technique registry."""
    spec = TECHNIQUES[tech_key]
    EncoderCls   = spec['encoder_cls']
    encode_fn    = spec['encode_fn_name']
    warmup_fn    = spec['warmup_fn_name']
    per_frame    = spec['per_frame']
    ce_info      = spec['ce_info']
    is_baseline  = spec.get('is_baseline', False)

    label = f"{tech_key}: pipe={pipeline_depth} LA={la_depth} {preset} QP{qp}"
    print(f"\n{'='*60}")
    print(f"[{label}]  {spec['desc']}")
    print(f"  {W}x{H}, {rate_mode}, {preset} QP{qp}, la={la_depth}, pipe={pipeline_depth}, N={n_frames}")
    print(f"{'='*60}")

    enc = EncoderCls(W, H, fps, preset=preset, qp=qp,
                     rate_mode=rate_mode, la_depth=la_depth,
                     pipeline_depth=pipeline_depth)

    # ── Warmup ──
    _warmup_func = getattr(enc, warmup_fn)
    for wi in range(n_warmup):
        rgb = _make_test_rgb(H, W, wi)
        nv12 = _rgb_to_nv12_gpu(rgb, input_is_bgr=False)
        torch.cuda.synchronize()
        _warmup_func(nv12)
    torch.cuda.synchronize()

    empty_count = 0
    empty_frames = []
    la_count = 0
    success_count = 0
    batch_size = 24
    t_start = time.time()

    _encode_func = getattr(enc, encode_fn)

    for batch_start in range(0, n_frames, batch_size):
        batch_end = min(batch_start + batch_size, n_frames)
        batch_tensors = []
        for fi in range(batch_start, batch_end):
            rgb = _make_test_rgb(H, W, fi + n_warmup)
            nv12 = _rgb_to_nv12_gpu(rgb, input_is_bgr=False)
            torch.cuda.synchronize()
            batch_tensors.append(nv12)

        if per_frame:
            # ★ Per-frame CE loop (single-ce technique) ★
            for i, nv12 in enumerate(batch_tensors):
                fi = batch_start + i
                h264_data = _encode_func(nv12)
                if not h264_data:
                    # h264_data == b"" from NEED_MORE_INPUT, or None in edge cases
                    if fi < la_depth and la_depth > 0:
                        la_count += 1
                    else:
                        empty_frames.append((fi, fi % pipeline_depth))
                        empty_count += 1
                else:
                    success_count += 1
        else:
            # ★ Batch encode (all other techniques) ★
            results = _encode_func(batch_tensors,
                                   force_idr_first=(batch_start == 0))
            for i, h264_data in enumerate(results):
                fi = batch_start + i
                if h264_data is None:
                    # ★ True empty frame — LockBitstream returned no data ★
                    empty_frames.append((fi, fi % pipeline_depth))
                    empty_count += 1
                elif h264_data == b"":
                    # Lookahead buffer fill — NVENC needs more input before producing output
                    if fi < la_depth and la_depth > 0:
                        la_count += 1
                    else:
                        # Post-LA empty frame is unexpected
                        empty_frames.append((fi, fi % pipeline_depth))
                        empty_count += 1
                else:
                    success_count += 1

    _summarize_empty(tech_key, empty_frames)
    flush_frames, flush_bytes = _flush_and_count(enc)
    success_count += flush_frames
    t_elapsed = time.time() - t_start
    enc.close()
    torch.cuda.synchronize()

    return _make_result(label, success_count, empty_count, la_count,
                        flush_frames, flush_bytes, t_elapsed, n_frames,
                        ce_info=ce_info, is_baseline=is_baseline)


def _make_result(label, success, empty, la, flush_f, flush_b, t, n,
                 ce_info="?", is_baseline=False):
    """Build a result dict with embedded technique metadata."""
    total = success + empty
    return {
        'label': label,
        'success': success,
        'empty': empty,
        'lookahead': la,
        'flush_frames': flush_f,
        'flush_bytes': flush_b,
        't_elapsed': t,
        'empty_rate': empty / max(1, total),
        'fps': n / t,
        'ce_info': ce_info,
        'is_baseline': is_baseline,
    }


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("NVENC completionEvent 3D Full-Matrix Verification")
    print(f"Config: {W}x{H}, VBR_HQ, {len(TECHNIQUES)} techniques × 2 pipes × 2 LAs = {len(TECHNIQUES)*4} combos")
    print(f"Techniques: {', '.join(TECHNIQUES.keys())}")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")

    results = []

    def _run(label_prefix, fn, *args, **kwargs):
        """Run a test variant and handle exceptions gracefully."""
        try:
            r = fn(*args, **kwargs)
            results.append(r)
        except Exception as e:
            print(f"  ❌ {label_prefix} FAILED: {e}")
            import traceback; traceback.print_exc()

    # ── 3D full matrix: technique × pipe × la ──
    _LAs = [LA_DEPTH, LA_DEPTH0]
    _Pipes = [4, 1]

    for tech_key in TECHNIQUES:
        for pipe in _Pipes:
            for la in _LAs:
                _label = f"{tech_key}: pipe={pipe} LA={la}"
                _run(_label, run_technique, tech_key, W, H, FPS, RATE_MODE, la,
                     N_FRAMES, N_WARMUP, pipeline_depth=pipe, preset=PRESET, qp=QP)

    # ── Summary Table ──
    # ── Sort: Verdict first (OK → unstable → broken), then FPS descending within each tier ──
    def _sort_key(r):
        er = r['empty_rate']
        verdict_order = 0 if er < 0.005 else (1 if er < 0.02 else 2)
        return (verdict_order, -r['fps'])
    _sorted = sorted(results, key=_sort_key)
    _best_fps = _sorted[0]['fps'] if _sorted else 1.0

    print("\n" + "=" * 120)
    print("3D FULL MATRIX — Technique × Pipe × LA — FPS Ranking")
    print("=" * 120)
    _hdr = (f"  {'Rank':>4}  {'Test Case':<58} {'CE':>11} {'Empty':>9} "
            f"  {'FPS':>7} {'Rel%':>6}  {'Verdict'}")
    _sep = (f"  {'-'*4}  {'-'*58} {'-'*11} {'-'*9} "
            f"  {'-'*7} {'-'*6}  {'-'*16}")
    print(_hdr)
    print(_sep)

    for _rank, r in enumerate(_sorted):
        _rel = r['fps'] / max(1, _best_fps) * 100
        _crown = '👑' if _rank == 0 else '  '

        _er = r['empty_rate']
        if _er < 0.005:        _badge = '✅ OK'
        elif _er < 0.02:       _badge = '⚠️ unstable'
        else:                  _badge = '❌ broken'

        if r.get('is_baseline'):
            _badge += ' 🏭'

        print(f"  {_crown}{_rank+1:>2}  {r['label']:<58} {r['ce_info']:>11} "
              f"{_er:>8.1%} {r['fps']:>7.1f} {_rel:>5.1f}%  {_badge}")

    print(_sep)
    legend_parts = [
        '👑=fastest',
        '✅ OK=0% empty',
        '⚠️=<2% empty',
        '❌=broken',
        '🏭=baseline',
    ]
    print(f"  Legend: {', '.join(legend_parts)}")
    print(f"  CE: none=sync LockBitstream  per-frame=create/sync/destroy  slot-reuse=pre-created(broken)")
    print(f"       batch=per-frame across slots  deferred=harvest on next slot reuse")
    print()


if __name__ == "__main__":
    main()