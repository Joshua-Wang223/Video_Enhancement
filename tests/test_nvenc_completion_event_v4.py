#!/usr/bin/env python3
"""
NVENC completionEvent multi-mode verification — 4D full matrix comparison.

4-dimensional matrix: 3 RC modes × 6 techniques × 2 pipes × 2 LAs = 60 combinations (20 per RC).

  Technique       Description
  ----------      ----------
  sync-batch      Synchronous batch (NO CE) — historical comparison baseline
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
import argparse
import pathlib
import subprocess
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
                # [FIX-FLUSH-LA-RETRY] doNotWait=1 + retry: LA-buffered frames
                # may not be ready on the first poll. Retry with small delays.
                result_parts = []
                _LockBitstreamProto_raw = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _total_recovered_frames = 0
                _total_recovered_bytes = 0

                # [FIX-FLUSH-EOS-ORDER] EOS is sent via slot 0's bs_buf.
                # Drain slots 1..N-1 FIRST, slot 0 LAST so EOS NAL appears
                # at the end of the concatenated byte stream.
                _n_slots = len(self._slots)
                _drain_order = list(range(1, _n_slots)) + [0] if _n_slots > 1 else [0]
                for _slot_idx in _drain_order:
                    _slot = self._slots[_slot_idx]
                    _bs_handle = _slot['bs_buf']
                    _slot_parts = []
                    _slot_empties = 0   # consecutive empty attempts
                    _MAX_SLOT_EMPTIES = 5  # up to 5 retries before giving up
                    while _slot_empties < _MAX_SLOT_EMPTIES:
                        lock_raw = (c_uint8 * 1544)()
                        ctypes.memset(lock_raw, 0, 1544)
                        cast(lock_raw, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
                        cast(byref(lock_raw, 4), ctypes.POINTER(c_uint32))[0] = 1  # doNotWait=1
                        cast(byref(lock_raw, 8), ctypes.POINTER(c_void_p))[0] = _bs_handle

                        lock_bs_fn = _LockBitstreamProto_raw(self._func_ptrs[_FUNC_IDX["LockBitstream"]])
                        bs_status = lock_bs_fn(self._encoder, lock_raw)
                        if bs_status != NV_ENC_SUCCESS:
                            _slot_empties += 1
                            time.sleep(0.01)
                            continue

                        bitstream_size = cast(byref(lock_raw, 36), ctypes.POINTER(c_uint32))[0]
                        if bitstream_size == 0:
                            _NvEncUnlockBitstreamProto(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(
                                self._encoder, _bs_handle)
                            _slot_empties += 1
                            time.sleep(0.01)
                            continue

                        # Got real data — reset empty streak
                        _slot_empties = 0
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
TOP_N = 14           # top-N: number of ranked techniques to produce output video files for
PRESET = "p1"        # NVENC preset: p1(poor)=lowest latency, p7=highest quality
QP = 23              # H.264 QP / VBR target quality (overridden by --crf)
RATE_MODE = "vbr_hq"  # constqp / vbr_hq / qvbr (overridden by --rate-modes loop)
RATE_MODES_DEFAULT = "constqp,vbr_hq,qvbr"  # default RC modes for 4D matrix when --rate-modes not set
_MIN_H264_FRAME_BYTES = 8  # ★ minimum valid H.264 NALU: start_code(3-4)+nalu_hdr(1)=≥4B, use 8B for safety
LA_DEPTH = [16, 8, 0]  # lookahead depth: 16 (deep), 8 (standard), 0 (disabled)
BATCH_SIZE = 24  # ── v3: Default batch size for per-call encoding. Smaller = more batches = more _slot_pending resets.

# ── [PLAN-A] Tier defense flag, set from --plan-a CLI arg ──
PLAN_A_MODE = False   # True → enable Tier 1-B (encode_frame retry) + Tier 1-A (prev_h264 fallback) on batch empty frames
PLAN_B_MODE = False   # [PLAN-B] True → register la-harvest technique (NVENCEncoderMode6, completion-driven harvest)


# ============================================================================
# Helpers
# ============================================================================

# ============================================================================
# Video I/O helpers (real video input / output file generation)
# ============================================================================
def _probe_video(input_path):
    """Probe video file: return (width, height, fps, total_frames)."""
    cmd = ['ffmpeg', '-hide_banner', '-i', str(input_path), '-f', 'null', '-']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stderr = result.stderr
    except Exception:
        stderr = ''
    w, h, fps, total = None, None, None, None
    m = re.search(r'(\d{2,5})x(\d{2,5})', stderr)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
    m = re.search(r'([\d.]+)\s*fps', stderr)
    if m:
        try: fps = float(m.group(1))
        except Exception: fps = 30.0
    m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', stderr)
    if m and fps:
        dur_s = int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3))
        total = int(dur_s * fps)
    return w or 1280, h or 720, fps or 30.0, total or 600


def _read_video_frames_rgb(input_path, start=0, count=None, w=None, h=None):
    """Read video frames as RGB uint8 GPU tensors via ffmpeg pipe.

    Returns list of (H, W, 3) uint8 tensors on GPU.
    Uses stderr=DEVNULL to avoid PIPE buffer deadlock (ffmpeg can write
    ~64KB of log/error to stderr before blocking; if unread, deadlocks stdout).
    """
    if w is None or h is None:
        w, h, _, _ = _probe_video(input_path)

    # ffmpeg args: -ss N before -i = input seek (fast, seeks to nearest keyframe).
    # -frames:v N after -i = output frame limit.
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'quiet']
    if start > 0:
        cmd += ['-ss', str(start)]
    cmd += ['-i', str(input_path)]
    if count is not None:
        cmd += ['-frames:v', str(count)]
    cmd += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']

    # ★ stderr=DEVNULL prevents PIPE deadlock (stderr buffer fills → ffmpeg hangs) ★
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    frames = []
    frame_size = w * h * 3
    while True:
        raw = proc.stdout.read(frame_size)
        if not raw or len(raw) < frame_size:
            break
        arr = torch.frombuffer(bytearray(raw), dtype=torch.uint8).cuda()
        if arr.numel() != w * h * 3:
            break
        arr = arr.reshape(h, w, 3).contiguous()
        frames.append(arr)

    # Drain remaining stdout and wait
    try:
        proc.stdout.read()
    except Exception:
        pass
    proc.wait()
    return frames


def _write_h264_es(output_dir, label_slug, h264_data_list, fps):
    """Write H.264 ES to temp .264, mux to .mp4 via ffmpeg (LA-tolerant). Returns (path_mp4, size_mb)."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    es_path = output_dir / f'{label_slug}.264'
    mp4_path = output_dir / f'{label_slug}.mp4'
    # Write raw ES
    with open(es_path, 'wb') as f:
        for data in h264_data_list:
            if data and isinstance(data, bytes):
                f.write(data)
    # Mux to MP4 — use -f h264 + quiet log (LA residual fragments may lack SPS/PPS)
    mux_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'quiet',
               '-f', 'h264', '-r', str(fps), '-i', str(es_path),
               '-c', 'copy', '-y', str(mp4_path)]
    subprocess.run(mux_cmd, check=True)
    # Remove raw ES, keep MP4
    try:
        es_path.unlink()
    except OSError:
        pass
    size_mb = mp4_path.stat().st_size / (1024 * 1024)
    print(f'  [output] {mp4_path.name}  ({size_mb:.1f} MB)', flush=True)
    return str(mp4_path), size_mb


def _probe_output_mp4(mp4_path, input_path, input_frames_count):
    """Probe output MP4 via ffprobe. Returns dict with frame counts, bitrate, etc."""
    import json as _json
    info = {'output_frames': 0, 'container_frames': 0, 'bitrate_kbps': 0,
            'codec': 'h264_nvenc', 'rc_mode': RATE_MODE.upper()}

    # ── Packet-level count (most reliable) ──
    try:
        p = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-count_packets', '-show_entries', 'stream=nb_read_packets',
                            '-of', 'csv=p=0', str(mp4_path)],
                           capture_output=True, text=True, timeout=30)
        if p.returncode == 0 and p.stdout.strip() not in ('', 'N/A'):
            info['output_frames'] = int(p.stdout.strip())
    except Exception:
        pass

    # ── Container-level nb_frames (quick, no decode) ──
    try:
        p = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=nb_frames',
                            '-of', 'csv=p=0', str(mp4_path)],
                           capture_output=True, text=True, timeout=15)
        if p.returncode == 0 and p.stdout.strip() not in ('', 'N/A', '0'):
            info['container_frames'] = int(p.stdout.strip())
    except Exception:
        pass

    # r_frame_rate x duration fallback
    if info['container_frames'] == 0:
        try:
            p = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                '-show_entries', 'stream=r_frame_rate,duration',
                                '-of', 'csv=p=0', str(mp4_path)],
                               capture_output=True, text=True, timeout=15)
            if p.returncode == 0:
                parts = p.stdout.strip().split(',')
                if len(parts) >= 2:
                    from fractions import Fraction
                    try:
                        rfr = float(Fraction(parts[0].strip()))
                        dur = float(parts[1].strip())
                        info['container_frames'] = int(round(rfr * dur))
                    except (ValueError, ZeroDivisionError):
                        pass
        except Exception:
            pass

    # ── Bitrate (stream-level first, format-level fallback) ──
    try:
        p = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=bit_rate',
                            '-of', 'csv=p=0', str(mp4_path)],
                           capture_output=True, text=True, timeout=15)
        if p.returncode == 0 and p.stdout.strip() not in ('', 'N/A', '0'):
            info['bitrate_kbps'] = round(int(p.stdout.strip()) / 1000, 1)
    except Exception:
        pass

    if info['bitrate_kbps'] == 0:
        try:
            p = subprocess.run(['ffprobe', '-v', 'error',
                                '-show_entries', 'format=bit_rate',
                                '-of', 'csv=p=0', str(mp4_path)],
                               capture_output=True, text=True, timeout=15)
            if p.returncode == 0 and p.stdout.strip() not in ('', 'N/A', '0'):
                info['bitrate_kbps'] = round(int(p.stdout.strip()) / 1000, 1)
        except Exception:
            pass

    # ── Input frame count ──
    info['input_frames'] = input_frames_count

    return info

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



class NVENCEncoderMode6(NVENCEncoder):
    """[PLAN-B] LA-Aware Harvest: completion-driven slot reuse (not slot-rotation).

    Solves the pipe=4 + LA>0 slot-rotation conflict by:
    1. Expanding slots: total_slots = pipeline_depth + la_depth (e.g. 12 for pipe=4,LA=8)
    2. Free-pool allocation: deque of free slot indices instead of fi % pd round-robin
    3. cuEventQuery polling: harvest frames when CE fires (not when slot re-rotates)
    4. LA CE preservation: NEED_MORE_INPUT frames keep their CE pending (not destroyed)
    5. Fallback wait: if free-pool exhausted, cuEventSynchronize oldest pending CE
    """

    def __init__(self, *args, pipeline_depth=4, la_depth=0, **kwargs):
        _expanded_pd = pipeline_depth + la_depth if la_depth > 0 else pipeline_depth
        kwargs.setdefault('pipeline_depth', _expanded_pd)
        kwargs.setdefault('la_depth', la_depth)
        super().__init__(*args, **kwargs)
        self._logical_pd = pipeline_depth  # for labeling only

    def encode_frames_batch_la_harvest(self, nv12_tensors, force_idr_first=False):
        """Completion-driven batch encode: poll CE → harvest → reuse freed slots."""
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
                total_slots = len(self._slots)
                pd = total_slots
                _W = self._width
                nv12_h = self._height + self._height // 2
                results = [None] * n_frames

                # Extra large bitstream buffers for LA-pending frames
                _slot_pending = [None] * total_slots  # (ce, fi, ep_status, fidr)
                _free_slots = _ct.deque(range(total_slots)) if hasattr(_ct, 'deque') else None
                if _free_slots is None:
                    from collections import deque
                    _free_slots_queue = deque(range(total_slots))
                else:
                    _free_slots_queue = deque(range(total_slots))

                _LockIB = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p,
                                        _ct.POINTER(_ct.c_uint8 * 1544))
                _UnlockIB = _ct.CFUNCTYPE(_ct.c_uint32, _ct.c_void_p, _ct.c_void_p)
                _CUDA_DEV = 2

                # cuEventQuery prototype
                self._libcuda.cuEventQuery.restype = _ct.c_uint32
                self._libcuda.cuEventQuery.argtypes = [_ct.c_void_p]

                la_buffered = 0

                for fi in range(n_frames):
                    # ═══ Phase 0: Poll CE → harvest completed slots ═══
                    recycled = 0
                    for si in range(total_slots):
                        if _slot_pending[si] is None:
                            continue
                        _pce, _pfi, _peps, _pfidr = _slot_pending[si]
                        if _pce.value and self._libcuda.cuEventQuery(_pce) == 0:
                            # CE fired → NVENC output is ready
                            self._libcuda.cuEventSynchronize.restype = _ct.c_uint32
                            self._libcuda.cuEventSynchronize.argtypes = [_ct.c_void_p]
                            self._libcuda.cuEventSynchronize(_pce)
                            self._libcuda.cuEventDestroy.restype = _ct.c_uint32
                            self._libcuda.cuEventDestroy.argtypes = [_ct.c_void_p]
                            self._libcuda.cuEventDestroy(_pce)
                            h264_data, _ = self._lock_bitstream_with_retry(
                                self._slots[si]['bs_buf'])
                            if h264_data:
                                results[_pfi] = h264_data
                            elif _peps == NV_ENC_ERR_NEED_MORE_INPUT:
                                results[_pfi] = b""
                            _slot_pending[si] = None
                            _free_slots_queue.append(si)
                            recycled += 1

                    # ═══ Phase 0.5: Fallback — wait oldest CE if pool exhausted ═══
                    while not _free_slots_queue:
                        # Find oldest pending slot by frame_index
                        _oldest_si = None
                        _oldest_fi = 10**9
                        for si in range(total_slots):
                            if _slot_pending[si] is not None:
                                if _slot_pending[si][1] < _oldest_fi:
                                    _oldest_fi = _slot_pending[si][1]
                                    _oldest_si = si
                        if _oldest_si is None:
                            raise RuntimeError("LA-Harvest: pool exhausted but no pending slot found")
                        _oce, _ofi, _oeps, _ofidr = _slot_pending[_oldest_si]
                        if _oce.value:
                            self._libcuda.cuEventSynchronize.restype = _ct.c_uint32
                            self._libcuda.cuEventSynchronize.argtypes = [_ct.c_void_p]
                            self._libcuda.cuEventSynchronize(_oce)
                            self._libcuda.cuEventDestroy.restype = _ct.c_uint32
                            self._libcuda.cuEventDestroy.argtypes = [_ct.c_void_p]
                            self._libcuda.cuEventDestroy(_oce)
                        h264_data, _ = self._lock_bitstream_with_retry(
                            self._slots[_oldest_si]['bs_buf'])
                        if h264_data:
                            results[_ofi] = h264_data
                        elif _oeps == NV_ENC_ERR_NEED_MORE_INPUT:
                            results[_ofi] = b""
                        _slot_pending[_oldest_si] = None
                        _free_slots_queue.append(_oldest_si)

                    # ═══ Phase 1: Submit new frame ═══
                    si = _free_slots_queue.popleft()
                    slot = self._slots[si]
                    force_idr = force_idr_first and fi == 0

                    # LockInputBuffer
                    lock_buf = (_ct.c_uint8 * 1544)()
                    _ct.memset(lock_buf, 0, 1544)
                    _ct.cast(lock_buf, _ct.POINTER(_ct.c_uint32))[0] = _ct.c_uint32(0x7001000d).value
                    _ct.cast(_ct.byref(lock_buf, 8), _ct.POINTER(_ct.c_void_p))[0] = slot['input_buf']
                    s = _LockIB(self._func_ptrs[_FUNC_IDX["LockInputBuffer"]])(self._encoder, lock_buf)
                    if s != 0:
                        raise RuntimeError("LockIB[%d] failed: %d" % (si, s))
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
                        raise RuntimeError("cuMemcpy2D[%d] failed: %d" % (si, r))
                    _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
                        self._encoder, slot['input_buf'])

                    # Per-frame CE
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
                        raise RuntimeError("EncodePicture[%d] failed: %d" % (si, _ep_s))

                    # ★ KEY: LA frames preserve CE, slot stays pending ★
                    _slot_pending[si] = (_ce, fi, _ep_s, force_idr)
                    if _ep_s == NV_ENC_ERR_NEED_MORE_INPUT:
                        results[fi] = b""
                        la_buffered += 1
                        if la_buffered <= 3 or la_buffered == self._la_depth:
                            print(f'[NVENC-Enc] 前向帧预看缓冲中 ({la_buffered}/{self._la_depth})',
                                  flush=True)

                # ═══ Phase 2: Drain remaining pending slots ═══
                for si in range(total_slots):
                    if _slot_pending[si] is not None:
                        _dce, _dfi, _deps, _dfidr = _slot_pending[si]
                        if _dce.value is not None:
                            self._libcuda.cuEventSynchronize.restype = _ct.c_uint32
                            self._libcuda.cuEventSynchronize.argtypes = [_ct.c_void_p]
                            self._libcuda.cuEventSynchronize(_dce)
                            self._libcuda.cuEventDestroy.restype = _ct.c_uint32
                            self._libcuda.cuEventDestroy.argtypes = [_ct.c_void_p]
                            self._libcuda.cuEventDestroy(_dce)
                        h264_data, _ = self._lock_bitstream_with_retry(
                            self._slots[si]['bs_buf'])
                        if h264_data:
                            results[_dfi] = h264_data
                        elif _deps == NV_ENC_ERR_NEED_MORE_INPUT:
                            results[_dfi] = b""
                        _slot_pending[si] = None

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
# Mode 7: ce-pipeline-v2 — async harvest + ring buffer + independent write thread
# ============================================================================
class NVENCEncoderMode7(NVENCEncoder):
    """CE-pipeline v2: 异步收割 + 环形输出缓冲区 + 独立写入线程。

    解决当前 ce-pipeline (Mode5) 的两个瓶颈：
    1. 收割绑定在 slot 复用时刻 → 后台收割线程轮询 CE，触发即收割
    2. 输出写入内嵌编码循环 → 独立写入线程按帧序扫描 ring_buffer

    架构：Main Encode Thread (submit only) ‖ Harvest Thread (poll CE → LockBitstream)
          ‖ Write Thread (scan ring_buffer → collect contiguous frames)
    """

    def __init__(self, *args, pipeline_depth=4, la_depth=0, **kwargs):
        kwargs.setdefault('pipeline_depth', pipeline_depth)
        kwargs.setdefault('la_depth', la_depth)
        super().__init__(*args, **kwargs)
        # 后台线程共享状态
        self._running = False
        self._slot_pending_v2 = []  # [(ce, fi, ep_status), ...] per slot
        self._free_slots = None     # collections.deque (created per batch)
        self._ring_buffer = None    # [Optional[bytes]] * n_frames
        self._results_v2 = None     # output list matching existing interface
        self._write_cursor = 0
        self._n_frames = 0
        # 条件变量
        self._harvest_cond = threading.Condition()
        self._write_cond = threading.Condition()
        # 线程句柄
        self._harvest_thread = None
        self._write_thread = None

    def _do_submit_v2(self, si, nv12_gpu_tensor, force_idr):
        """LockInputBuffer → cuMemcpy2D → UnlockInputBuffer → EncodePicture(per-frame CE)。

        Caller must hold self._lock. Returns (ce_handle, ep_status).
        Extracted from Mode5 Phase 2 for reuse by main encode thread.
        """
        import ctypes as _ct
        slot = self._slots[si]
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
            raise RuntimeError("LockIB[%d] failed: %d" % (si, s))
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
            raise RuntimeError("cuMemcpy2D[%d] failed: %d" % (si, r))
        _UnlockIB(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(
            self._encoder, slot['input_buf'])

        # ★ Per-frame CE (create fresh event)
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
            raise RuntimeError("EncodePicture[%d] failed: %d" % (si, _ep_s))

        return _ce, _ep_s

    def _harvest_loop(self):
        """后台收割线程：轮询 pending slots 的 CE → 触发即收割 LockBitstream → 释放 slot。

        使用 self._lock 短暂保护每个 LockBitstream/UnlockBitstream 调用。
        cuEventQuery 轮询在锁外执行（无 NVENC API 调用）。

        ⚠️ 此线程必须 push primary CUDA context：daemon 线程不继承父线程的 context。
        """
        import ctypes as _ct
        pd = self._pipeline_depth
        _empty_polls = 0  # 连续空轮询计数，用于自适应休眠

        # ★ 跨线程 CUDA context 保护：daemon 线程需要显式 push primary context
        _h_need_pop = False
        _primary = getattr(self, '_primary_ctx', None)
        if _primary is not None and _primary.value is not None:
            try:
                self._libcuda.cuCtxPushCurrent.restype = _ct.c_uint32
                self._libcuda.cuCtxPushCurrent.argtypes = [_ct.c_void_p]
                _h_r_push = self._libcuda.cuCtxPushCurrent(_primary)
                _h_need_pop = (_h_r_push == 0)
            except Exception:
                pass

        while self._running or any(s is not None for s in self._slot_pending_v2):
            harvested_any = False

            for si in range(pd):
                pending = self._slot_pending_v2[si]
                if pending is None:
                    continue
                _ce, _fi, _ep_status = pending
                if _ce is None or _ce.value is None:
                    # CE destroyed or invalid — mark as free
                    self._slot_pending_v2[si] = None
                    self._free_slots.append(si)
                    continue
                # cuEventQuery: 0 = complete, other = still pending
                if self._libcuda.cuEventQuery(_ce) != 0:
                    continue

                # ★ CE 已触发 → 收割
                with self._lock:
                    self._libcuda.cuEventSynchronize.restype = _ct.c_uint32
                    self._libcuda.cuEventSynchronize.argtypes = [_ct.c_void_p]
                    self._libcuda.cuEventSynchronize(_ce)
                    self._libcuda.cuEventDestroy.restype = _ct.c_uint32
                    self._libcuda.cuEventDestroy.argtypes = [_ct.c_void_p]
                    self._libcuda.cuEventDestroy(_ce)
                    h264_data, _bs = self._lock_bitstream_with_retry(
                        self._slots[si]['bs_buf'])

                # 写入 ring_buffer（无锁 — GIL 保护单条赋值）
                if h264_data:
                    self._ring_buffer[_fi] = h264_data
                elif _ep_status == NV_ENC_ERR_NEED_MORE_INPUT:
                    self._ring_buffer[_fi] = b""  # LA 缓冲帧标记
                # else: 真正的空帧 — _ring_buffer[_fi] 保持 None，后续诊断

                # 释放 slot
                self._slot_pending_v2[si] = None
                self._free_slots.append(si)
                harvested_any = True

                # 通知写入线程有新数据
                with self._write_cond:
                    self._write_cond.notify()

            # 自适应轮询间隔：无收获时逐步延长休眠，有收获时重置
            if harvested_any:
                _empty_polls = 0
            else:
                _empty_polls += 1
                if _empty_polls > 1000:  # ~100ms 无收获 → 降低轮询频率
                    time.sleep(0.001)
                elif _empty_polls > 100:
                    time.sleep(0.0002)
                else:
                    time.sleep(0.00005)  # 50μs — 高频轮询

        # ── Drain 阶段：强制等待所有 pending slots ──
        for si in range(pd):
            pending = self._slot_pending_v2[si]
            if pending is not None:
                _ce, _fi, _ep_status = pending
                if _ce is not None and _ce.value is not None:
                    with self._lock:
                        self._libcuda.cuEventSynchronize.restype = _ct.c_uint32
                        self._libcuda.cuEventSynchronize.argtypes = [_ct.c_void_p]
                        self._libcuda.cuEventSynchronize(_ce)
                        self._libcuda.cuEventDestroy.restype = _ct.c_uint32
                        self._libcuda.cuEventDestroy.argtypes = [_ct.c_void_p]
                        self._libcuda.cuEventDestroy(_ce)
                        h264_data, _bs = self._lock_bitstream_with_retry(
                            self._slots[si]['bs_buf'])
                    if h264_data:
                        self._ring_buffer[_fi] = h264_data
                    elif _ep_status == NV_ENC_ERR_NEED_MORE_INPUT:
                        self._ring_buffer[_fi] = b""
                self._slot_pending_v2[si] = None

        # 通知写入线程 drain 完成
        with self._write_cond:
            self._write_cond.notify()

        # ★ 退出时弹出 CUDA context（harvest 线程专用）
        if _h_need_pop:
            try:
                _ctx_out = _ct.c_void_p()
                self._libcuda.cuCtxPopCurrent.restype = _ct.c_uint32
                self._libcuda.cuCtxPopCurrent.argtypes = [_ct.POINTER(_ct.c_void_p)]
                self._libcuda.cuCtxPopCurrent(_ct.byref(_ctx_out))
            except Exception:
                pass

    def _write_loop(self):
        """独立写入线程：扫描 ring_buffer，按帧序收集连续就绪的帧 → _results_v2。

        遇到空洞（未就绪帧）时等待 _write_cond 通知，不阻塞编码。
        """
        pd = self._pipeline_depth
        while self._running or self._write_cursor < self._n_frames:
            # 尽可能多地消费连续就绪帧
            consumed = 0
            while (self._write_cursor < self._n_frames and
                   self._ring_buffer[self._write_cursor] is not None):
                data = self._ring_buffer[self._write_cursor]
                if data:  # 非空帧（b"" 是 LA 帧，不写入结果）
                    self._results_v2[self._write_cursor] = data
                self._write_cursor += 1
                consumed += 1

            if self._write_cursor >= self._n_frames:
                break  # 全部就绪

            # 等待新数据通知
            with self._write_cond:
                # 双重检查：可能在获取锁期间数据就绪了
                if self._write_cursor < self._n_frames and \
                   self._ring_buffer[self._write_cursor] is not None:
                    continue
                self._write_cond.wait(timeout=0.01)  # 10ms 超时，防止死等

    def encode_frames_batch_ce_pipeline_v2(self, nv12_tensors, force_idr_first=False):
        """ce-pipeline v2 入口：异步收割 + 环形缓冲区 + 独立写入线程。

        主线程只负责 Submit（LockInputBuffer + EncodePicture），收割和写入
        完全由后台线程处理。与 Mode5 的核心区别：
        - Harvest 不再绑定 slot 复用：CE 触发即收割
        - 输出不再内嵌编码循环：独立 write thread 按帧序收集

        Returns list of H.264 bytes in frame order (compatible with Mode5 interface).
        """
        import ctypes as _ct
        from collections import deque
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
            pd = self._pipeline_depth

            # 初始化 per-batch 状态
            self._n_frames = n_frames
            self._slot_pending_v2 = [None] * pd
            self._free_slots = deque(range(pd))
            self._ring_buffer = [None] * n_frames
            self._results_v2 = [None] * n_frames
            self._write_cursor = 0

            # cuEventQuery prototype（harvest thread 使用）
            self._libcuda.cuEventQuery.restype = _ct.c_uint32
            self._libcuda.cuEventQuery.argtypes = [_ct.c_void_p]

            # 启动后台线程
            self._running = True
            self._harvest_thread = threading.Thread(target=self._harvest_loop, daemon=True)
            self._write_thread = threading.Thread(target=self._write_loop, daemon=True)
            self._harvest_thread.start()
            self._write_thread.start()

            # ── 主编码循环：只负责 Submit ──
            for fi in range(n_frames):
                # 等待空闲 slot（有界等待，防止 free_slots 永久为空）
                _wait_iters = 0
                while not self._free_slots:
                    _wait_iters += 1
                    if _wait_iters > 1000:
                        # 超过 ~100ms 无空闲 slot → 死锁检测
                        _pend_count = sum(1 for s in self._slot_pending_v2 if s is not None)
                        raise RuntimeError(
                            "ce-pipeline-v2: deadlock detected — no free slots after 100ms "
                            "(pending=%d/%d, ring_ready=%d/%d)" %
                            (_pend_count, pd,
                             sum(1 for x in self._ring_buffer if x is not None), n_frames))
                    time.sleep(0.0001)
                si = self._free_slots.popleft()

                force_idr = force_idr_first and fi == 0

                # Submit（持锁调用 NVENC API）
                with self._lock:
                    _ce, _ep_s = self._do_submit_v2(
                        si, nv12_tensors[fi], force_idr)

                # 注册 pending（harvest thread 将自动收割）
                self._slot_pending_v2[si] = (_ce, fi, _ep_s)

            # ── 等待后台线程完成 ──
            self._running = False
            if self._harvest_thread is not None:
                self._harvest_thread.join(timeout=30.0)
                if self._harvest_thread.is_alive():
                    print("[ce-pipeline-v2] ⚠️ harvest thread join timeout (30s) — "
                          "draining in main thread", flush=True)
                    # 强制 drain 剩余 pending slots
                    self._force_drain_pending()
            if self._write_thread is not None:
                self._write_thread.join(timeout=10.0)

            # 最终检查：确保 ring_buffer 中没有遗漏
            for fi in range(n_frames):
                if self._results_v2[fi] is None and self._ring_buffer[fi] not in (None, b""):
                    self._results_v2[fi] = self._ring_buffer[fi]

            return self._results_v2
        finally:
            self._running = False
            if _need_pop:
                try:
                    _ctx_out = _ct.c_void_p()
                    self._libcuda.cuCtxPopCurrent.restype = _ct.c_uint32
                    self._libcuda.cuCtxPopCurrent.argtypes = [_ct.POINTER(_ct.c_void_p)]
                    self._libcuda.cuCtxPopCurrent(_ct.byref(_ctx_out))
                except Exception:
                    pass

    def _force_drain_pending(self):
        """强制收割所有 pending slots（在 harvest thread 超时后调用）。"""
        import ctypes as _ct
        for si in range(self._pipeline_depth):
            pending = self._slot_pending_v2[si]
            if pending is None:
                continue
            _ce, _fi, _ep_status = pending
            if _ce is not None and _ce.value is not None:
                try:
                    with self._lock:
                        self._libcuda.cuEventSynchronize.restype = _ct.c_uint32
                        self._libcuda.cuEventSynchronize.argtypes = [_ct.c_void_p]
                        self._libcuda.cuEventSynchronize(_ce)
                        self._libcuda.cuEventDestroy.restype = _ct.c_uint32
                        self._libcuda.cuEventDestroy.argtypes = [_ct.c_void_p]
                        self._libcuda.cuEventDestroy(_ce)
                        h264_data, _bs = self._lock_bitstream_with_retry(
                            self._slots[si]['bs_buf'])
                    if h264_data:
                        self._ring_buffer[_fi] = h264_data
                    elif _ep_status == NV_ENC_ERR_NEED_MORE_INPUT:
                        self._ring_buffer[_fi] = b""
                except Exception:
                    pass
            self._slot_pending_v2[si] = None


# Technique Registry — encoding strategy × pipeline_depth × lookahead
# ============================================================================
TECHNIQUES = {
    'sync-batch': {
        'encoder_cls': NVENCEncoder,
        'encode_fn_name': 'encode_frames_batch',
        'warmup_fn_name': 'encode_frame',
        'per_frame': False,
        'ce_info': 'none',
        'desc': 'sync batch (NO CE) — historical comparison baseline',
    },
    'single-ce': {
        'encoder_cls': NVENCEncoderMode1,
        'encode_fn_name': 'encode_frame_with_ce',
        'warmup_fn_name': 'encode_frame_with_ce',
        'per_frame': True,
        'ce_info': 'per-frame',
        'desc': 'single-slot CE — per-frame create→sync→destroy',
    },
    'batch-ce': {
        'encoder_cls': NVENCEncoderMode2,
        'encode_fn_name': 'encode_frames_batch_with_ce',
        'warmup_fn_name': 'encode_frame',
        'per_frame': False,
        'ce_info': 'batch',
        'desc': 'batch per-frame CE — sync harvest across all slots',
    },
    # 'phase4-slot': {
    #     'encoder_cls': NVENCEncoderMode3,
    #     'encode_fn_name': 'encode_frames_batch_phase4',
    #     'warmup_fn_name': 'encode_frame',
    #     'per_frame': False,
    #     'ce_info': 'slot-reuse',
    #     'min_pipeline_depth': 4,  # ★ slot-reuse 要求 pd≥4: pd=1 时 submit 覆盖未 harvest 的 slot → 帧丢失+IDR假阴性
    #     'desc': 'PHASE4 async — slot-reuse event (broken for pipe=4)',
    # },
    # 'phase4-pfce': {
    #     'encoder_cls': NVENCEncoderMode4,
    #     'encode_fn_name': 'encode_frames_batch_phase4_pfce',
    #     'warmup_fn_name': 'encode_frame',
    #     'per_frame': False,
    #     'ce_info': 'per-frame',
    #     'min_pipeline_depth': 4,  # ★ slot-reuse 要求 pd≥4: pd=1 时 submit 覆盖未 harvest 的 slot → 帧丢失+IDR假阴性
    #     'desc': 'PHASE4 async — per-frame CE (no event reuse)',
    # },
    'ce-pipeline': {
        'encoder_cls': NVENCEncoderMode5,
        'encode_fn_name': 'encode_frames_batch_ce_pipeline',
        'warmup_fn_name': 'encode_frame',
        'per_frame': False,
        'ce_info': 'deferred',
        'desc': 'CE-pipeline — defer LockBitstream to next slot reuse',
    },
    'la-harvest': {
        'encoder_cls': NVENCEncoderMode6,
        'encode_fn_name': 'encode_frames_batch_la_harvest',
        'warmup_fn_name': 'encode_frame',
        'per_frame': False,
        'ce_info': 'completion',
        'min_pipeline_depth': 4,   # requires pipe >= 4 for pipelining
        'min_la_depth': 8,         # only meaningful with LA > 0
        'desc': 'LA-Aware Harvest — completion-driven slot reuse',
    },
    'ce-pipeline-v2': {
        'encoder_cls': NVENCEncoderMode7,
        'encode_fn_name': 'encode_frames_batch_ce_pipeline_v2',
        'warmup_fn_name': 'encode_frame',
        'per_frame': False,
        'ce_info': 'async-deferred',
        'desc': 'CE-pipeline v2 — async harvest + ring buffer + independent write thread',
    },
}


# ============================================================================
# Unified test runner — dispatches technique × pipe × la
# ============================================================================
def run_technique(tech_key, W, H, fps, rate_mode, la_depth, n_frames, n_warmup,
                  pipeline_depth=4, preset="p1", qp=23,
                  input_frames=None, video_name=None, output_dir=None,
                  label_prefix=None, batch_size: int = 24,
                  num_segments: int = 1):
    """Run one (technique, pipe, la, rate_mode) combination via technique registry.

    When num_segments > 1 and input_frames is provided, splits frames into N segments,
    processes each independently with encoder flush+reuse between segments (simulates
    multi-segment production pipeline)."""
    spec = TECHNIQUES[tech_key]
    EncoderCls   = spec['encoder_cls']
    encode_fn    = spec['encode_fn_name']
    warmup_fn    = spec['warmup_fn_name']
    per_frame    = spec['per_frame']
    ce_info      = spec['ce_info']

    # ── v4: Per-combo version mapping (technique × pipe × la → IFRNet version) ──
    # [FIX-VERSION-LABEL] 所有版本默认使用 ce-pipeline deferred 路径（非 sync-batch）。
    # sync-batch 是历史对比基线，sync-batch pipe=1/4 LA=0 不代表任何版本的默认行为。
    _VERSION_MAP = {
        ('ce-pipeline', 4, 0): 'v6.4.3/4/5.1',     # CONSTQP, 无 LA — 偶数小版本生产路径
        ('ce-pipeline', 4, 8): 'v6.4.3.1/4.1',      # VBR_HQ/QVBR + LA=8 + pipe=4 — 奇数小版本生产路径 (Tier 1-B/A defense active)
        ('ce-pipeline', 1, 8): 'pipe1-safe-path',    # pipe=1 + LA=8 安全路径 (0% 空帧, 非生产默认)
        ('la-harvest', 4, 8): 'v6.4.x exp.',         # [PLAN-B] LA-Aware Harvest 实验性
        ('ce-pipeline-v2', 4, 0): 'v6.4.x-v2',      # ce-pipeline-v2: CONSTQP, 异步收割
        ('ce-pipeline-v2', 4, 8): 'v6.4.x-v2',      # ce-pipeline-v2: VBR_HQ/QVBR + LA=8, 异步收割
    }
    version_tag = _VERSION_MAP.get((tech_key, pipeline_depth, la_depth), '')

    label = f"{tech_key}: pipe={pipeline_depth} LA={la_depth} {preset} QP{qp}"
    _header = label_prefix if label_prefix else label
    print(f"\n{'='*60}")
    print(f"[{_header}]  {spec['desc']}")
    print(f"  {W}x{H}, {rate_mode}, {preset} QP{qp}, la={la_depth}, pipe={pipeline_depth}, N={n_frames}")
    print(f"{'='*60}")

    enc = EncoderCls(W, H, fps, preset=preset, qp=qp,
                     rate_mode=rate_mode, la_depth=la_depth,
                     pipeline_depth=pipeline_depth)

    # ── Warmup ──
    _warmup_func = getattr(enc, warmup_fn)
    _warmup_frames = input_frames[-n_warmup:] if (input_frames and len(input_frames) >= n_warmup) else None
    for wi in range(n_warmup):
        if _warmup_frames and wi < len(_warmup_frames):
            rgb = _warmup_frames[wi].contiguous()
        else:
            rgb = _make_test_rgb(H, W, wi)
    empty_count = 0
    empty_frames = []
    la_count = 0
    success_count = 0
    # ── [PLAN-A] Tier defense tracking ──
    tier1b_attempts = 0   # frames where Tier 1-B retry was attempted
    tier1b_success = 0    # frames saved by Tier 1-B (encode_frame retry)
    tier1a_fallback = 0   # frames saved by Tier 1-A (prev_h264 reuse)
    prev_h264 = None      # cached last valid H.264 for Tier 1-A fallback
    # v3: batch_size is now a parameter (default 24), not hardcoded
    t_start = time.time()

    _encode_func = getattr(enc, encode_fn)
    _is_constqp = (rate_mode == 'constqp')  # [V6442-CONSTQP-FAST] CONSTQP skips Tier 1-B

    for batch_start in range(0, n_frames, batch_size):
        batch_end = min(batch_start + batch_size, n_frames)
        batch_tensors = []
        for fi in range(batch_start, batch_end):
            if input_frames and fi < len(input_frames):
                rgb = input_frames[fi].contiguous()
            else:
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
                    # ── [PLAN-A] Tier 1-B: retry with encode_frame ──
                    if PLAN_A_MODE and not _is_constqp:
                        tier1b_attempts += 1
                        _retry_ok = False
                        _nv12_t = batch_tensors[i]
                        for _rtry in range(2):
                            try:
                                h264_rt = enc.encode_frame(_nv12_t, force_idr=((batch_start == 0) and i == 0))
                                if h264_rt and len(h264_rt) >= _MIN_H264_FRAME_BYTES:
                                    h264_data = h264_rt
                                    tier1b_success += 1
                                    success_count += 1
                                    prev_h264 = h264_data
                                    _retry_ok = True
                                    break
                            except Exception:
                                pass
                        if _retry_ok:
                            continue  # frame saved by Tier 1-B
                        # ── [PLAN-A] Tier 1-A: fallback to prev_h264 ──
                        if prev_h264 is not None:
                            tier1a_fallback += 1
                            success_count += 1
                            # prev_h264 unchanged (this frame is a dup of previous)
                            continue
                    # (fallthrough: Tier defense disabled or first-frame with no prev_h264)
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
                    # ★ Byte-level validation: phase4 slot-reuse at pipe=1 can produce
                    #    "valid" bytes that lack actual encoded frames (lost IDR in bitstream).
                    #    Minimum valid H.264 NALU: start_code(3-4B) + nalu_hdr(1B) = 4+ bytes.
                    if len(h264_data) < _MIN_H264_FRAME_BYTES:
                        empty_frames.append((fi, fi % pipeline_depth))
                        empty_count += 1
                    else:
                        success_count += 1
                        prev_h264 = h264_data  # [PLAN-A] cache for Tier 1-A fallback

    _summarize_empty(tech_key, empty_frames)
    flush_fragments, flush_bytes = _flush_and_count(enc)
    # flush data contains residual NAL fragments from the NVENC pipeline that are
    # NOT independently decodable video frames (missing IDR/DPB context).
    # Do NOT add flush_fragments to success_count — they are diagnostic-only.
    # LA-buffered frames are intentionally consumed by NVENC, not lost.
    # Expected valid output = submitted - LA_consumed.
    enc.close()
    torch.cuda.synchronize()
    t_elapsed = time.time() - t_start


    # ── Frame loss accounting ──
    # LA depth frames are consumed by NVENC by design (NEED_MORE_INPUT) —
    # they never produce output. Expected output = submitted - LA_buffered.
    _submitted = n_frames
    _expected = _submitted - la_count
    _actual = success_count  # only frames with valid H.264 output (excludes flush fragments)
    _lost = max(0, _expected - _actual)
    if _lost > 0 or la_count > 0:
        print(f'  [LOSS] {_lost}/{_expected} expected frames unrecoverable '
              f'(submitted={_submitted} expected={_expected} '
              f'valid={success_count} LA_buf={la_count} '
              f'empty={empty_count} flush_fragments={flush_fragments} '
              f'flush_bytes={flush_bytes})', flush=True)

    # ── [PLAN-A] defense frames are already counted in success_count ──
    # (success_count += 1 in Tier 1-B and Tier 1-A branches above)

    extra = {
        'source': 'real_video' if input_frames else 'synthetic',
        'input_file': video_name or None,
        'resolution': f'{W}x{H}',
        'rate_mode': rate_mode,
        'preset': preset,
        'qp': qp,
        'n_frames': n_frames,
        'lost_frames': _lost,
        'total_submitted': _submitted,
        'total_recovered': _actual,      # actual decodable frames (excludes flush fragments)
        'expected_frames': _expected,    # expected output: submitted - LA_buffered
        'la_consumed': la_count,         # frames intentionally consumed by LA
        'flush_fragments': flush_fragments,  # NAL fragments from flush (diagnostic, not frames)
        'flush_bytes': flush_bytes,
        'version': version_tag,
        # ── [PLAN-A] defense statistics ──
        'plan_a': PLAN_A_MODE,
        'tier1b_attempts': tier1b_attempts,
        'tier1b_success': tier1b_success,
        'tier1a_fallback': tier1a_fallback,
    }
    return _make_result(label, success_count, empty_count, la_count,
                        flush_fragments, flush_bytes, t_elapsed, n_frames,
                        ce_info=ce_info, version_tag=version_tag, extra=extra)


def _make_result(label, success, empty, la, flush_f, flush_b, t, n,
                 ce_info="?", version_tag='', extra=None):
    """Build a result dict with embedded technique metadata."""
    total = success + empty
    r = {
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
        # v3: frame loss accounting from extra or defaults
        'lost_frames': extra.get('lost_frames', 0) if extra else 0,
        'total_submitted': extra.get('total_submitted', n) if extra else n,
        'total_recovered': extra.get('total_recovered', n) if extra else n,
        'version': version_tag,
        # [PLAN-A] defense stats
        'plan_a': extra.get('plan_a', False) if extra else False,
        'tier1b_attempts': extra.get('tier1b_attempts', 0) if extra else 0,
        'tier1b_success': extra.get('tier1b_success', 0) if extra else 0,
        'tier1a_fallback': extra.get('tier1a_fallback', 0) if extra else 0,
    }
    if extra:
        r.update(extra)
    return r


# ============================================================================
# Main
# ============================================================================
def main():
    global W, H, FPS, N_FRAMES, N_WARMUP, TOP_N
    import argparse as _ap
    ap = _ap.ArgumentParser(description='NVENC completionEvent 4D Full-Matrix Verification (v4) — batch-size + frame loss detection')
    ap.add_argument('--input', '-i', type=str, default=None,
                    help='Input video file for real-content testing (synthetic if omitted)')
    ap.add_argument('--output', '-o', type=str, default=None,
                    help='Output directory for top-3 video files (default: ./temp)')
    ap.add_argument('--batch-size', '-b', type=int, default=BATCH_SIZE,
                    help=f'Frames per encode_*() call (default {BATCH_SIZE}). '
                         f'Smaller = more batches = more _slot_pending resets. '
                         f'Use a large value (e.g. {N_FRAMES}) for single-call baseline.')
    ap.add_argument('--plan-a', action='store_true', default=False,
                    help='[PLAN-A] Enable Tier 1-B/A defense on batch encode empty frames '
                         '(retry with encode_frame, fallback to prev_h264). '
                         'Reports per-technique defense effectiveness for pipe=4+LA>0 evaluation.')
    ap.add_argument('--plan-b', action='store_true', default=False,
                    help='[PLAN-B] Enable LA-Aware Harvest technique (NVENCEncoderMode6). '
                         'Uses completion-driven slot reuse (cuEventQuery poll) instead of slot-rotation. '
                         'Only registered for pipe=4+LA=8 combos.')
    ap.add_argument('--rate-modes', type=str, default='',
                    help=f'Comma-separated RC modes for 4D matrix (default: "{RATE_MODES_DEFAULT}"). '
                         f'Each mode × (technique × pipe × LA) tested independently. '
                         f'Valid: constqp, vbr_hq, qvbr')
    ap.add_argument('--crf', type=int, default=QP,
                    help=f'H.264 CRF / target quality for all RC modes (default {QP}). '
                         f'constqp: QP=CRF, vbr_hq: targetQuality=CRF, qvbr: qvbrQuality=CRF')
    ap.add_argument('--num-segments', type=int, default=1,
                    help='Number of segments for real-video pipeline simulation (default 1). '
                         'When >1, splits input frames into N segments, runs each independently '
                         'with encoder flush+reuse between segments (simulates multi-segment production).')
    ap.add_argument('--la', type=str, default='all',
                    help='Filter by lookahead depth: 0, 8, 16, or "all" (default: all)')
    ap.add_argument('--pipe', type=str, default='all',
                    help='Filter by pipeline depth: 1, 2, 4, 8, or "all" (default: all)')
    ap.add_argument('--techniques', type=str, default='all',
                    help='Filter by techniques: comma-separated list with optional spaces '
                         '(e.g. "ce-pipeline, sync-batch, batch-ce") (default: all)')
    ap.add_argument('--top-n', type=str, default=str(TOP_N),
                    help=f'Number of top-ranked techniques to produce output videos for: '
                         f'integer (0/1/2/...) or "all" for unlimited (default: {TOP_N})')
    args = ap.parse_args()

    # ── [PLAN-A] global flag ──
    global PLAN_A_MODE, PLAN_B_MODE
    PLAN_A_MODE = args.plan_a
    if PLAN_A_MODE:
        print("[PLAN-A] Tier 1-B/A defense ENABLED — batch empty frames will be retried+compensated")

    # ── [PLAN-B] global flag ──
    PLAN_B_MODE = args.plan_b
    if PLAN_B_MODE:
        print("[PLAN-B] LA-Aware Harvest ENABLED — la-harvest technique registered for pipe=4+LA=8")

    # ── --top-n ──
    if args.top_n.strip().lower() == 'all':
        TOP_N = 999999  # unlimited: min(TOP_N, len(_sorted)) = len(_sorted)
    else:
        try:
            TOP_N = int(args.top_n)
        except ValueError:
            ap.error(f"--top-n: invalid value '{args.top_n}' (expected integer or 'all')")

    # ── Probe input video if provided ──
    video_name = None
    if args.input:
        if not os.path.isfile(args.input):
            print(f"ERROR: input file not found: {args.input}")
            sys.exit(1)
        W, H, FPS, _total = _probe_video(args.input)
        video_name = os.path.splitext(os.path.basename(args.input))[0]
        N_FRAMES = min(N_FRAMES, _total) if _total else N_FRAMES  # respect global N_FRAMES as cap
        print(f"[Input] {args.input}")
        print(f"[Probe] {W}x{H} @ {FPS:.2f}fps, using {N_FRAMES} frames")
    else:
        # ── No --input: global constants or fallback defaults ──
        W = W if W else (720, 576)
        H = H if H else 576
        FPS = FPS if FPS else 30.0
        N_FRAMES = N_FRAMES if N_FRAMES else 2000
        N_WARMUP = N_WARMUP if N_WARMUP else 30

    # N_WARMUP cap: never exceed global value or 1/3 of total frames
    N_WARMUP = min(N_WARMUP, N_FRAMES // 3)

    output_dir = args.output or os.path.join(os.getcwd(), 'temp')

    # ── CLI filtering: --la, --pipe, --techniques ──
    _LAs = list(LA_DEPTH)
    _Pipes = [8, 4, 2, 1]
    if args.la != 'all':
        _la_val = int(args.la)
        if _la_val not in _LAs:
            print(f"ERROR: --la must be 0, 8, 16, or all (got {_la_val})")
            sys.exit(1)
        _LAs = [_la_val]
    if args.pipe != 'all':
        _pipe_val = int(args.pipe)
        if _pipe_val not in _Pipes:
            print(f"ERROR: --pipe must be 1, 2, 4, 8, or all (got {_pipe_val})")
            sys.exit(1)
        _Pipes = [_pipe_val]
    if args.techniques != 'all':
        _tech_filter = set(t.strip() for t in args.techniques.split(',') if t.strip())
        _unknown = _tech_filter - set(TECHNIQUES.keys())
        if _unknown:
            print(f"ERROR: unknown technique(s): {', '.join(sorted(_unknown))}")
            print(f"       valid: {', '.join(sorted(TECHNIQUES.keys()))}")
            sys.exit(1)
    else:
        _tech_filter = None

    # ── Build valid test matrix: filter phase4 × pipe=1, la-harvest × LA=0, plan-b gate ──
    _test_combos = []
    for tech_key in TECHNIQUES:
        if _tech_filter is not None and tech_key not in _tech_filter:
            continue
        _min_pd = TECHNIQUES[tech_key].get('min_pipeline_depth', 1)
        _min_la = TECHNIQUES[tech_key].get('min_la_depth', 0)
        # [PLAN-B] gate: la-harvest only registered when --plan-b is set
        if tech_key == 'la-harvest' and not PLAN_B_MODE:
            continue
        for pipe in _Pipes:
            if pipe < _min_pd:
                continue
            for la in _LAs:
                if la < _min_la:
                    continue  # la-harvest × LA=0 → skip (equivalent to ce-pipeline)
                _test_combos.append((tech_key, pipe, la))
    _total_rounds = len(_test_combos)
    _active_techs = [t for t in TECHNIQUES if _tech_filter is None or t in _tech_filter]
    _n_techs = len(_active_techs)
    _total_possible = sum(1 for t in _active_techs
                          for p in [8, 4, 2, 1]
                          for l in LA_DEPTH
                          if p >= TECHNIQUES[t].get('min_pipeline_depth', 1)
                          and l >= TECHNIQUES[t].get('min_la_depth', 0)
                          and not (t == 'la-harvest' and not PLAN_B_MODE))
    _skipped = _total_possible - _total_rounds

    # ── Parse rate modes ──
    _rm_arg = args.rate_modes.strip() if args.rate_modes else RATE_MODES_DEFAULT
    _rate_modes = [rm.strip() for rm in _rm_arg.split(',') if rm.strip() in ('constqp', 'vbr_hq', 'qvbr')]
    if not _rate_modes:
        _rate_modes = RATE_MODES_DEFAULT.split(',')
    _crf = args.crf
    _num_segments = args.num_segments
    _multi_rc = len(_rate_modes) > 1

    print("=" * 70)
    print("NVENC completionEvent 4D Full-Matrix Verification v4")
    src_label = f"real video: {args.input}" if args.input else "synthetic GPU tensors"
    _pipe_labels = '+'.join(str(p) for p in _Pipes)
    _la_labels = '+'.join(str(l) for l in _LAs)
    _combo_info = f"{_n_techs} techniques x {len(_Pipes)} pipe(s)[{_pipe_labels}] x {len(_LAs)} LA(s)[{_la_labels}] = {_total_rounds} valid combos"
    if _skipped > 0:
        _combo_info += f" (skipped {_skipped})"
    _rm_info = f"{len(_rate_modes)} RC mode(s)" if _multi_rc else _rate_modes[0]
    _seg_info = f"  segments={_num_segments}" if _num_segments > 1 and args.input else ""
    print(f"Config: {W}x{H}@{FPS:.1f}fps, N={N_FRAMES}, {_rm_info}: {', '.join(_rate_modes)}, CRF={_crf}, batch_size={args.batch_size}, {_combo_info}{_seg_info}")
    print(f"Source: {src_label}")
    _n_batches = (N_FRAMES + args.batch_size - 1) // args.batch_size if args.batch_size > 0 else 1
    _total_rounds_all = _total_rounds * len(_rate_modes)
    print(f"Techniques: {', '.join(_active_techs)}")
    print(f"Batch mode: {_n_batches} batches × ~{args.batch_size} frames = ~{N_FRAMES} total per round")
    print(f"Total rounds: {_total_rounds_all} ({_total_rounds} combos × {len(_rate_modes)} RC modes)")
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
            kwargs['label_prefix'] = label_prefix
            r = fn(*args, **kwargs)
            results.append(r)
        except Exception as e:
            print(f"  ❌ {label_prefix} FAILED: {e}")
            import traceback; traceback.print_exc()

    # ── Load real frames if input provided ──
    input_frames = None
    if args.input:
        print(f"[IO] Loading {N_FRAMES} frames from {args.input}...", flush=True)
        input_frames = _read_video_frames_rgb(
            args.input, start=0, count=N_FRAMES, w=W, h=H)
        mb = len(input_frames) * W * H * 3 // (1024 * 1024)
        print(f"[IO] Loaded {len(input_frames)} frames ({mb} MB on GPU)", flush=True)

    # ── 4D full matrix: rate_mode × technique × pipe × la (valid combos only) ──
    # When num_segments > 1, each combo is tested per-segment and results aggregated.
    _round = 0
    _use_seg = (_num_segments > 1 and input_frames is not None)

    for _rm in _rate_modes:
        for tech_key, pipe, la in _test_combos:
            _round += 1

            if _use_seg:
                # ── Multi-segment mode: split input frames, run per segment, aggregate ──
                _seg_size = len(input_frames) // _num_segments
                _seg_results = []
                _seg_total_t = 0.0
                _seg_combined = None

                for _si in range(_num_segments):
                    _s_start = _si * _seg_size
                    _s_end = min((_si + 1) * _seg_size, len(input_frames)) if _si < _num_segments - 1 else len(input_frames)
                    _s_frames = input_frames[_s_start:_s_end]
                    _s_n = len(_s_frames)
                    if _s_n == 0:
                        continue

                    _s_label = f"Round {_round}/{_total_rounds_all}  [{_rm}] {tech_key}: pipe={pipe} LA={la} [seg{_si+1}/{_num_segments}]"
                    _s_r = run_technique(tech_key, W, H, FPS, _rm, la,
                                         _s_n, min(N_WARMUP, _s_n // 3),
                                         pipeline_depth=pipe, preset=PRESET, qp=_crf,
                                         input_frames=_s_frames, video_name=video_name,
                                         output_dir=output_dir if _si == 0 else None,
                                         label_prefix=_s_label, batch_size=args.batch_size,
                                         num_segments=1)
                    _seg_results.append(_s_r)
                    _seg_total_t += _s_r.get('t_elapsed', 0.0)

                # Aggregate multi-segment results into a single combined result
                if _seg_results:
                    _first = _seg_results[0]
                    _seg_combined = dict(_first)
                    _seg_combined['label'] = f"[{_rm}] {tech_key}: pipe={pipe} LA={la} (x{_num_segments}seg)"
                    _seg_combined['success'] = sum(r.get('success', 0) for r in _seg_results)
                    _seg_combined['empty'] = sum(r.get('empty', 0) for r in _seg_results)
                    _seg_combined['lookahead'] = sum(r.get('lookahead', 0) for r in _seg_results)
                    _seg_combined['flush_frames'] = sum(r.get('flush_frames', 0) for r in _seg_results)
                    _seg_combined['flush_bytes'] = sum(r.get('flush_bytes', 0) for r in _seg_results)
                    _seg_combined['lost_frames'] = sum(r.get('lost_frames', 0) for r in _seg_results)
                    _seg_combined['total_submitted'] = sum(r.get('total_submitted', 0) for r in _seg_results)
                    _seg_combined['total_recovered'] = sum(r.get('total_recovered', 0) for r in _seg_results)
                    _seg_combined['t_elapsed'] = _seg_total_t
                    _seg_combined['fps'] = _seg_combined['total_submitted'] / max(0.001, _seg_total_t)
                    _seg_combined['empty_rate'] = _seg_combined['empty'] / max(1, _seg_combined['success'] + _seg_combined['empty'])
                    _seg_combined['tier1b_attempts'] = sum(r.get('tier1b_attempts', 0) for r in _seg_results)
                    _seg_combined['tier1b_success'] = sum(r.get('tier1b_success', 0) for r in _seg_results)
                    _seg_combined['tier1a_fallback'] = sum(r.get('tier1a_fallback', 0) for r in _seg_results)
                    # Mark as multi-segment result
                    _seg_combined['num_segments'] = _num_segments
                    _seg_combined['plan_a'] = _first.get('plan_a', PLAN_A_MODE)
                    results.append(_seg_combined)
                    print(f"  [AGG] {_seg_combined['label']}: "
                          f"total={_seg_combined['total_submitted']} "
                          f"empty={_seg_combined['empty']} "
                          f"lost={_seg_combined['lost_frames']} "
                          f"fps={_seg_combined['fps']:.1f}", flush=True)
            else:
                # ── Single-segment mode ──
                _label = f"Round {_round}/{_total_rounds_all}  [{_rm}] {tech_key}: pipe={pipe} LA={la}"
                _run(_label, run_technique, tech_key, W, H, FPS, _rm, la,
                     N_FRAMES, N_WARMUP, pipeline_depth=pipe, preset=PRESET, qp=_crf,
                     input_frames=input_frames, video_name=video_name, output_dir=output_dir,
                     batch_size=args.batch_size, num_segments=_num_segments)

    # ── Top-3: re-encode full video through best techniques for output files ──
    topn_ranks = set()
    topn_stats = []  # collected for summary stats table
    _topn_frame_count = len(input_frames) if input_frames else 0
    _can_topn = (args.input and input_frames and _topn_frame_count > 0 and results)
    if not _can_topn:
        if args.input:
            print(f"\n[TOP3] Skipped: input_frames={bool(input_frames)} loaded={_topn_frame_count} "
                  f"N_FRAMES={N_FRAMES} results={len(results)}", flush=True)
        else:
            print(f"\n[TOP3] Skipped: no --input/-i provided. "
                  f"Use --input <video> to generate top-N output files.", flush=True)
    # ── 统一排序: 帧完整性优先 (lost==0 AND empty==0) → FPS 降序 ──
    def _sort_key(rr):
        """帧完整性优先 → FPS 降序。

        Tier 0: lost==0 AND empty==0 — 完美 (输出帧==输入帧)
        Tier 1: lost==0 AND empty_rate<0.5%
        Tier 2: lost==0 AND empty_rate<2%
        Tier 3: lost==0 (空帧较多但不丢帧)
        Tier 4: lost>0 (有永久丢帧)
        """
        lost = max(0, rr.get('lost_frames', 0))
        er = rr['empty_rate']
        if lost == 0 and er == 0:
            tier = 0
        elif lost == 0 and er < 0.005:
            tier = 1
        elif lost == 0 and er < 0.02:
            tier = 2
        elif lost == 0:
            tier = 3
        else:
            tier = 4
        return (tier, -rr['fps'])

    _sorted = sorted(results, key=_sort_key)

    if _can_topn:
        # Generate videos for ALL top-N entries (no filtering — frame integrity
        # is handled by sort order; per-file frame completeness is reported in
        # the stats summary table below).
        _topn_candidates = _sorted[:min(TOP_N, len(_sorted))]
        _topn_total = len(_topn_candidates)
        print(f"\n{'='*70}")
        print(f"TOP-{_topn_total}: Re-encoding full video via best techniques...")
        print(f"{'='*70}")
        _topn_round = 0
        for rank_idx, r in enumerate(_topn_candidates):
            try:
                _topn_round += 1
                topn_ranks.add(rank_idx)
                label = r['label']
                tech_key = label.split(':')[0].strip()
                # Strip RC prefix "[constqp] " and "(xNseg)" suffix from multi-segment labels
                if tech_key.startswith('['):
                    _parts = tech_key.split('] ', 1)
                    if len(_parts) == 2:
                        tech_key = _parts[1].strip()
                if ' (' in tech_key and tech_key.endswith(')'):
                    tech_key = tech_key.split(' (')[0].strip()
                m_pipe = re.search(r'pipe=(\d+)', label)
                m_la = re.search(r'LA=(\d+)', label)
                pipe_val = int(m_pipe.group(1)) if m_pipe else 4
                la_val = int(m_la.group(1)) if m_la else 0

                print(f"\n  Round {_topn_round}/{_topn_total}  Rank #{rank_idx+1}: {label}", flush=True)
                spec = TECHNIQUES[tech_key]
                EncCls = spec['encoder_cls']

                # ★ Global NVENC noise suppression for entire init+encode+flush ★
                import io as _io
                _old_out = sys.stdout
                sys.stdout = _io.StringIO()
                try:
                    _rm_topn = r.get('rate_mode', RATE_MODE)
                    _enc = EncCls(W, H, FPS, preset=PRESET, qp=_crf,
                                  rate_mode=_rm_topn, la_depth=la_val,
                                  pipeline_depth=pipe_val)

                    _wf = getattr(_enc, spec['warmup_fn_name'])
                    _warm_offset = max(0, _topn_frame_count - N_WARMUP - 1)
                    for wi in range(N_WARMUP):
                        _rgb_src = input_frames[_warm_offset + wi].contiguous()
                        _nv12_src = _rgb_to_nv12_gpu(_rgb_src, input_is_bgr=False)
                        torch.cuda.synchronize()
                        _wf(_nv12_src)
                    torch.cuda.synchronize()

                    _ef = getattr(_enc, spec['encode_fn_name'])

                    _per_frame = spec.get('per_frame', False)

                    # # [FIX-TOP-LA-FALLBACK] With LA > 0 and slot-based CE pipeline,
                    # # NVENC delays encoding until lookahead fills, so LockBitstream
                    # # yields H.264 for a *different* frame than the pending slot's
                    # # index — causing misattribution. Fall back to the synchronous
                    # # batch path (encode_frames_batch) which yields frames in
                    # # correct order (LA offsets produce b"" for first LA_depth
                    # # entries, flush recovers the tail). The 1D matrix test already
                    # # captured the CE-based FPS measurement.
                    # if la_val > 0 and spec.get('ce_info', 'none') != 'none':
                    #     _ef = _enc.encode_frames_batch
                    #     _per_frame = False  # batch interface, not per-frame
                    
                    output_data = []
                    _prev_h264 = None  # [PLAN-A] cache for Tier 1-A fallback
                    _t0 = time.time()
                    bs = args.batch_size  # v3: batch_size from CLI
                    _last_tick = 0.0
                    for bi in range(0, _topn_frame_count, bs):
                        be = min(bi + bs, _topn_frame_count)
                        # progress bar every ~1 second
                        _now = time.time()
                        _pct = int(bi / _topn_frame_count * 100)
                        if _now - _last_tick >= 1.0 or bi == 0:
                            _bar = '█' * (_pct * 60 // 100) + '░' * (60 - _pct * 60 // 100)
                            sys.stdout = _old_out
                            print(f'  [{_bar}] {_pct:>3}%', end='\r', flush=True)
                            sys.stdout = _io.StringIO()
                            _last_tick = _now
                        _batch = []
                        for fi in range(bi, be):
                            _rgb = input_frames[fi].contiguous()
                            _nv12 = _rgb_to_nv12_gpu(_rgb, input_is_bgr=False)
                            torch.cuda.synchronize()
                            _batch.append(_nv12)

                        if _per_frame:
                            for _nv12f in _batch:
                                _d = _ef(_nv12f)
                                if _d and isinstance(_d, bytes) and len(_d) >= _MIN_H264_FRAME_BYTES:
                                    output_data.append(_d)
                                    _prev_h264 = _d
                        else:
                            _res = _ef(_batch, force_idr_first=(bi == 0))
                            for _i, _d in enumerate(_res):
                                if _d is None:
                                    # ── [PLAN-A] Tier 1-B: retry with encode_frame ──
                                    if PLAN_A_MODE and not (_rm_topn == 'constqp'):
                                        _nv12_t = _batch[_i]
                                        _retry_ok = False
                                        for _rtry in range(2):
                                            try:
                                                h264_rt = _enc.encode_frame(_nv12_t, force_idr=((bi == 0) and _i == 0))
                                                if h264_rt and len(h264_rt) >= _MIN_H264_FRAME_BYTES:
                                                    output_data.append(h264_rt)
                                                    _prev_h264 = h264_rt
                                                    _retry_ok = True
                                                    break
                                            except Exception:
                                                pass
                                        if _retry_ok:
                                            continue
                                        # ── [PLAN-A] Tier 1-A: fallback to prev_h264 ──
                                        if _prev_h264 is not None:
                                            output_data.append(_prev_h264)
                                            continue
                                    # (fallthrough: no defense or first-frame with no prev_h264)
                                elif _d == b"":
                                    # LA buffer fill — NVENC needs more input; no output frame
                                    pass
                                elif isinstance(_d, bytes):
                                    if len(_d) >= _MIN_H264_FRAME_BYTES:
                                        output_data.append(_d)
                                        _prev_h264 = _d

                    # ── 100% ──
                    sys.stdout = _old_out
                    _bar = '█' * 60
                    print(f'  [{_bar}] 100%', flush=True)

                    # Flush — send EOS + drain NVENC pipeline.
                    # ★ IMPORTANT: flush bytes are LA residual fragments (no DPB reference chain,
                    #    decode to garbled frames). Do NOT append to output_data.
                    #    Per la-flush-recovery-is-harmful.md: flush output is diagnostic-only,
                    #    writing it to output MP4 pollutes the video tail with corrupted frames.
                    _flush = _enc.flush()
                    if _flush and len(_flush) > 0:
                        print(f'  [NVENC-FLUSH] {len(_flush)} bytes drained (diagnostic, not written to output)', flush=True)

                    sys.stdout = _io.StringIO()
                    _enc.close()
                    sys.stdout = _old_out
                    torch.cuda.synchronize()
                finally:
                    sys.stdout = _old_out

                _t_full = time.time() - _t0

                # Write output → muxed MP4
                label_slug = re.sub(r'[^a-zA-Z0-9_.-]', '_',
                                    f'rank{rank_idx+1:02d}_{tech_key}_pipe{pipe_val}_LA{la_val}')
                mp4_path, mp4_mb = _write_h264_es(output_dir, label_slug, output_data, FPS)

                # Probe output for detailed stats
                t3_info = _probe_output_mp4(mp4_path, args.input, _topn_frame_count)
                t3_info['rank'] = rank_idx + 1
                t3_info['label_slug'] = label_slug
                t3_info['fps_full'] = _topn_frame_count / max(0.001, _t_full)
                t3_info['file_size_mb'] = mp4_mb
                t3_info['encoder'] = 'h264_nvenc'
                t3_info['rc_mode'] = _rm_topn.upper()
                t3_info['filename'] = f'{label_slug}.mp4'
                t3_info['la_depth'] = la_val  # ★ LA depth for frame-perfect accounting
                topn_stats.append(t3_info)

                r['output_h264'] = mp4_path
                r['output_mb'] = mp4_mb
                r['full_fps'] = t3_info['fps_full']
                print(f'  [stats] fps={t3_info["fps_full"]:.1f}  '
                      f'size={mp4_mb:.1f}MB  '
                      f'out_frames={t3_info["output_frames"]}', flush=True)
            except Exception as e_topn:
                print(f"  ⚠️ top-N re-encode {rank_idx+1} FAILED: {e_topn}", flush=True)
                import traceback; traceback.print_exc()

    # ── Summary Table ──
    # _sort_key and _sorted defined above (before Top-N section).
    _best_fps = _sorted[0]['fps'] if _sorted else 1.0

    # Source info line
    print("\n")
    if args.input:
        print(f"  Input:  {args.input}  ({W}x{H}@{FPS:.1f}fps, {N_FRAMES} frames)")
    else:
        print(f"  Source: synthetic GPU tensors ({W}x{H}, {N_FRAMES} frames)")
    if output_dir:
        print(f"  Output: {output_dir}")
    print()

    print("=" * 120)
    _title_extra = f" x {len(_rate_modes)}RC" if _multi_rc else ""
    print(f"4D FULL MATRIX — {W}x{H}@{FPS:.0f}fps N={N_FRAMES} batch_size={args.batch_size}{_title_extra} — RC x Technique x Pipe x LA — FPS Ranking")
    print("=" * 120)
    # ── Determine column widths ──
    _tc_width = 45 if PLAN_A_MODE else 52
    _rc_col = f' {"RC":>7}' if _multi_rc else ''
    _rc_sep = f' {"─"*7}' if _multi_rc else ''
    if PLAN_A_MODE:
        _hdr = (f"  {'Rank':>4}  {'Test Case':<{_tc_width}} {'CE':>11} {'Empty':>8} "
                f"{'Lost':>5} {'Def':>7}  {'FPS':>7} {'Rel%':>6}{_rc_col}  {'Verdict'}")
        _sep = (f"  {'─'*4}  {'─'*_tc_width} {'─'*11} {'─'*8} "
                f"{'─'*5} {'─'*7}  {'─'*7} {'─'*6}{_rc_sep}  {'─'*22}")
    else:
        _hdr = (f"  {'Rank':>4}  {'Test Case':<{_tc_width}} {'CE':>11} {'Empty':>8} {'Lost':>5} "
                f"  {'FPS':>7} {'Rel%':>6}{_rc_col}  {'Verdict'}")
        _sep = (f"  {'─'*4}  {'─'*_tc_width} {'─'*11} {'─'*8} {'─'*5} "
                f"  {'─'*7} {'─'*6}{_rc_sep}  {'─'*16}")
    print(_hdr)
    print(_sep)

    for _rank, r in enumerate(_sorted):
        _rel = r['fps'] / max(1, _best_fps) * 100
        _crown = '👑' if _rank == 0 else '  '

        _er = r['empty_rate']
        _lost = r.get('lost_frames', 0)
        _t1b_saved = r.get('tier1b_success', 0)
        _t1a_saved = r.get('tier1a_fallback', 0)
        _defended = _t1b_saved + _t1a_saved
        if _lost > 0:
            _badge = f'🔥 LOST={_lost}'
        elif _defended > 0:
            _badge = f'🛡️ DEF={_defended} (1B={_t1b_saved} 1A={_t1a_saved})'
        elif _er < 0.005:        _badge = '✅ OK'
        elif _er < 0.02:       _badge = '⚠️ unstable'
        else:                  _badge = '❌ broken'

        if r.get('version'):
            _badge += ' ' + r['version']
        if r.get('output_mb'):
            _badge += f' 📁{r["output_mb"]:.0f}MB'
        if _rank in topn_ranks:
            _badge += ' 🎬'

        _lost_str = str(_lost) if _lost > 0 else ('-' if _lost == 0 else '?')
        # ── Extract RC mode from result ──
        _rm_label = ''
        if _multi_rc:
            _rm_extra = r.get('rate_mode', '')
            _rm_label = f' {_rm_extra:>7}' if _rm_extra else '        ?'
        if PLAN_A_MODE:
            _def_str = str(_defended) if _defended > 0 else '-'
            print(f"  {_crown}{_rank+1:>2}  {r['label']:<{_tc_width}} {r['ce_info']:>11} "
                  f"{_er:>7.1%} {_lost_str:>5} {_def_str:>7}   {r['fps']:>7.1f} {_rel:>5.1f}%{_rm_label}  {_badge}")
        else:
            print(f"  {_crown}{_rank+1:>2}  {r['label']:<{_tc_width}} {r['ce_info']:>11} "
                  f"{_er:>7.1%} {_lost_str:>5}   {r['fps']:>7.1f} {_rel:>5.1f}%{_rm_label}  {_badge}")

    print(_sep)

    # ── Top-N Stats Summary Table (all generated files; frame-perfect entries tagged) ──
    if topn_stats:
        _perfect = [s for s in topn_stats
                    if (s.get('output_frames', 0) == s.get('container_frames', 0)
                        and s.get('output_frames', 0) > 0
                        and s.get('container_frames', 0) == s.get('input_frames', 0) - (
                            0 if (s.get('rc_mode', '') or '').lower() == 'constqp' else s.get('la_depth', 0)))]
        _imperfect = len(topn_stats) - len(_perfect)
        # ── Compute FPS ranking among top-N entries (pure FPS, independent of frame-integrity sort) ──
        _fps_sorted = sorted(enumerate(topn_stats), key=lambda x: -x[1].get('fps_full', 0))
        _fps_rank = {_idx: _r + 1 for _r, (_idx, _) in enumerate(_fps_sorted)}
        print()
        print("=" * 135)
        print(f"TOP-N OUTPUT FILES — Detailed Statistics  "
              f"[{len(_perfect)}/{len(topn_stats)} perfect: Output=Container=Input_for_CONSTQP_else_Input−LA]")
        if _imperfect > 0:
            print(f"  ({_imperfect} entries have frame-count mismatch "
                  f"— LA buffered (by design) / bitstream errors)")
        print("=" * 135)
        print(f"  {'Rank':>4}  {'Rank-FPS':>8}  {'Input':>8}  {'LA':>3}  {'Output':>8}  {'Container':>9}  "
              f"{'FPS':>8}  {'File Size':>10}  {'RC Mode':>8}  {'Encoder':>12}  {'Status':>10}  {'Filename'}")
        print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*3}  {'-'*8}  {'-'*9}  "
              f"{'-'*8}  {'-'*10}  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*38}")
        for _idx, s in enumerate(topn_stats):
            _rk = s.get('rank', '?')
            _rfps = _fps_rank.get(_idx, '?')
            _in = s.get('input_frames', '?')
            _la = s.get('la_depth', 0)
            _out = s.get('output_frames', '?')
            _cf = s.get('container_frames', '?')
            _fp = s.get('fps_full', 0)
            _sz = s.get('file_size_mb', 0)
            _rc = s.get('rc_mode', '?')
            _enc = s.get('encoder', '?')
            _fn = s.get('filename', '?')
            _expected = _in if (_rc or '').lower() == 'constqp' else _in - _la
            _perf = (_out == _cf and _cf == _expected) and _in != '?'
            _status = '✅ PERFECT' if _perf else '⚠️ MISMATCH'
            print(f"  {_rk:>4}  {_rfps:>8}  {_in:>8}  {_la:>3}  {_out:>8}  {_cf:>9}  "
                  f"{_fp:>8.1f}  {_sz:>8.1f} MB  {_rc:>8}  {_enc:>12}  {_status:>10}  {_fn}")
        print()

    legend_parts = [
        '👑=fastest',
        '✅ OK=0% empty, 0 lost',
        '🛡️ DEF=defended (Tier 1-B retry / 1-A prev_h264) — all empty frames recovered',
        '⚠️=<2% empty, 0 lost',
        '❌=broken',
        '🔥=frames LOST (permanent, unrecoverable)',
        'vX.X.X=IFRNet version using this technique/pipe/LA combo',
    ]
    if PLAN_A_MODE:
        legend_parts.append('Def=defended (Tier 1B+1A recovered frames)')
    print(f"  Legend: {', '.join(legend_parts)}")
    print(f"  Lost = max(0, (submitted − LA_buffered) − valid_output). LA-buffered frames are by design (NVENC lookahead), not lost.")
    print(f"  CE: none=sync LockBitstream  per-frame=create/sync/destroy  slot-reuse=pre-created(broken)")
    print(f"       batch=per-frame across slots  deferred=harvest on next slot reuse")
    print(f"  🎬=TOPN output  📁=file size (MB)")
    if args.input:
        print(f"  Output directory: {output_dir}")
    if _multi_rc:
        print(f"  RC modes tested: {', '.join(_rate_modes)} — column shows per-result RC mode")
    # ── Cross-RC-mode comparison (multi-RC only) ──
    if _multi_rc and len(results) >= len(_rate_modes):
        print()
        print("  ── Cross-RC-Mode Comparison ──")
        # Group by technique key: extract from label
        _rc_groups = {}
        for r in results:
            _lbl = r.get('label', '')
            if _lbl.startswith('['):
                _parts = _lbl.split('] ', 1)
                _key = _parts[1].split(' [seg')[0] if len(_parts) == 2 and ' [seg' in _parts[1] else (_parts[1] if len(_parts) == 2 else _lbl)
            else:
                _key = _lbl.split(' [seg')[0] if ' [seg' in _lbl else _lbl
            _rm = r.get('rate_mode', '?')
            _rc_groups.setdefault(_key, {})[_rm] = r

        # Only show groups with multiple RC modes
        _cmp_rows = []
        _tc_max = 0
        _col_w = 12  # fixed width per RC column
        for _key, _modes in sorted(_rc_groups.items()):
            if len(_modes) < 2:
                continue
            _tc_max = max(_tc_max, len(_key))
            _best_fps_mode = max(_modes.values(), key=lambda x: x.get('fps', 0))
            _best_rm = _best_fps_mode.get('rate_mode', '?')
            _row = {'key': _key}
            for _rm_name in _rate_modes:
                if _rm_name in _modes:
                    _mr = _modes[_rm_name]
                    _fps = _mr.get('fps', 0)
                    _lost = _mr.get('lost_frames', 0)
                    _lost_str = f' L={_lost}' if _lost > 0 else ''
                    _flag = ' 👑' if _rm_name == _best_rm else ''
                    _row[_rm_name] = f'{_fps:>5.0f}{_lost_str}'
                else:
                    _row[_rm_name] = '     -'
            _row['best'] = f'{_best_rm}'
            _cmp_rows.append(_row)

        if _cmp_rows:
            _tc_w = max(_tc_max + 2, 28)  # technique column width
            _sep_tec = '─' * (_tc_w + 2)
            _sep_col = '─' * (_col_w + 2)
            _hdr_rc_cols = '  '.join(f' {rm:>{_col_w}} ' for rm in _rate_modes)
            _sep_rc_cols = '  '.join(_sep_col for _ in _rate_modes)
            print(f"  {'Technique / Pipe / LA':<{_tc_w}}  {_hdr_rc_cols}   Best RC")
            print(f"  {_sep_tec}  {_sep_rc_cols}  {'─'*8}")
            for _row in _cmp_rows:
                _rc_vals = '  '.join(f' {_row.get(rm, "?"):>{_col_w}} ' for rm in _rate_modes)
                print(f"  {_row['key']:<{_tc_w}}  {_rc_vals}   {_row['best']}")
        else:
            print("    (no cross-RC comparisons — only one RC mode per technique)")
        print()

    # ── Version recommendation ──
    if _sorted:
        _best = _sorted[0]
        _best_rm = _best.get('rate_mode', '')
        _best_ver = _best.get('version', '')
        print(f"  💡 Best: {_best['label']} ({_best['fps']:.1f} FPS, {_best['empty_rate']:.1%} empty)"
              f"{' [' + _best_rm + ']' if _best_rm else ''}"
              f"{' → ' + _best_ver if _best_ver else ''}")
    print()


if __name__ == "__main__":
    main()