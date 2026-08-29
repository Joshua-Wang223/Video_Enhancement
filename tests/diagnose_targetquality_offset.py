#!/usr/bin/env python3
"""
诊断 NV_ENC_RC_PARAMS 字段偏移 — GPU 实测验证工具。

基于 nvEncodeAPI.h master (nv-codec-headers, SEQUENTIAL layout, NO union) 验证：
  targetQuality@88(uint8), averageBitRate@20(rc_ptr[5]), multiPass@100(rc_ptr[25])

历史：此脚本曾使用错误的 UNION 假设 (avgbr@12, tq@76)，已被 GPU 测试推翻。
      正确布局经 nvEncodeAPI.h 原文+VBR_HQ CRF=18/28 GPU 验证通过 (2026-06-09)。

用法:
  python diagnose_targetquality_offset.py          # 验证已知正确偏移
  python diagnose_targetquality_offset.py --sweep  # 扫描候选偏移（未来驱动变化时）

要求: NVIDIA GPU, CUDA driver, NVENC SDK DLL, PyTorch
"""

import ctypes
import sys
import os
import time
import threading
from ctypes import (
    c_uint32, c_uint16, c_uint8, c_int32, c_int8, c_uint64,
    c_size_t, c_void_p, c_bool, POINTER, byref, cast, sizeof,
    memset, CFUNCTYPE, CDLL,
)

# ═══════════════════════════════════════════════════════════════════════════════
# NVENC SDK 13.0 constants
# ═══════════════════════════════════════════════════════════════════════════════

def _sdk13_ver(ver, bit31=False):
    v = _NVENCAPI_VERSION | (ver << 16) | (0x7 << 28)
    if bit31:
        v |= (1 << 31)
    return v

_NVENCAPI_VERSION = 0x0d  # SDK 13.0 MAJOR | (MINOR << 24)

NV_ENC_SUCCESS = 0
NV_ENC_DEVICE_TYPE_CUDA = 1
NV_ENC_BUFFER_FORMAT_NV12 = 1
NV_ENC_PIC_STRUCT_FRAME = 2
NV_ENC_PARAMS_RC_CONSTQP = 0
NV_ENC_PARAMS_RC_VBR_HQ = 32
NV_ENC_PARAMS_RC_QVBR = 64

NV_ENC_PRESET_CONFIG_VER = _sdk13_ver(5, True)   # 0xf005000d
NV_ENC_CONFIG_VER = _sdk13_ver(9, True)           # 0xf009000d
NV_ENC_INITIALIZE_PARAMS_VER = _sdk13_ver(7, True)  # 0xf007000d
NV_ENC_PIC_PARAMS_VER = _sdk13_ver(7, True)       # 0xf007000d
NV_ENC_LOCK_BITSTREAM_VER = _sdk13_ver(2, True)   # 0xf002000d
NV_ENC_CREATE_INPUT_BUFFER_VER = _sdk13_ver(2)    # 0x7002000d
NV_ENC_CREATE_BITSTREAM_BUFFER_VER = _sdk13_ver(1)  # 0x7001000d
NV_ENC_LOCK_INPUT_BUFFER_VER = _sdk13_ver(1)      # 0x7001000d

# ═══════════════════════════════════════════════════════════════════════════════
# Correct NV_ENC_RC_PARAMS layout (nvEncodeAPI.h master, GPU-verified 2026-06-09)
# ═══════════════════════════════════════════════════════════════════════════════
#
# NV_ENC_RC_PARAMS is a 128-byte SEQUENTIAL struct. NV_ENC_QP is a 12-byte
# sequential sub-struct {qpInterP, qpInterB, qpIntra}, NOT a 4-byte union.
#
# Offset  rc_ptr   Type      Field
# ────────────────────────────────────────────────────────────────────
#   0       [0]    uint32    version                  _sdk13_ver(1)
#   4       [1]    uint32    rateControlMode          0=CONSTQP, 32=VBR_HQ, 64=QVBR
#   8       [2]    uint32    constQP.qpInterP         NV_ENC_QP struct (12B)
#  12       [3]    uint32    constQP.qpInterB
#  16       [4]    uint32    constQP.qpIntra
#  20       [5]    uint32    averageBitRate           ← was rc_ptr[3] in old wrong code
#  24       [6]    uint32    maxBitRate
#  28       [7]    uint32    vbvBufferSize
#  32       [8]    uint32    vbvInitialDelay
#  36       [9]    uint32    bitfield                 bits: AQ=3, LA=5, TemporalAQ=8
#  40      [10]    uint32    minQP.qpInterP
#  ... (minQP/maxQP/initialRCQP 各 12B)
#  76      [19]    uint32    temporallayerIdxMask
#  80       —       uint8    temporalLayerQP[8]
#  88       —       uint8    targetQuality            ← KEY! uint8 at byte 88
#  89       —       uint8    targetQualityLSB
#  90       —       uint16   lookaheadDepth           ← uint16 at byte 90
#  92       —       uint8    lowDelayKeyFrameScale
#  93       —       int8     yDcQPIndexOffset
#  94       —       int8     uDcQPIndexOffset
#  95       —       int8     vDcQPIndexOffset
#  96      [24]    uint32    qpMapMode
# 100      [25]    uint32    multiPass                ← SEPARATE field, NOT in bitfield!
# 104      [26]    uint32    alphaLayerBitrateRatio
# 108       —       int8     cbQPIndexOffset
# 109       —       int8     crQPIndexOffset
# 110       —       uint16   reserved2
# 112      [28]    uint32    lookaheadLevel
# 116       —       uint8    viewBitrateRatios[7]     ← old wrong tq target!
# 123       —       uint8    reserved3
# 124      [31]    uint32    reserved1
# ────────────────────────────────────────────────────────────────────
# TOTAL: 128 bytes

# Verified correct rcParams byte offsets (nvEncodeAPI.h master, 2026-06-09)
RC_CORRECT = {
    "version":              0,   # uint32, rc_ptr[0]
    "rateControlMode":      4,   # uint32, rc_ptr[1]
    "constQP.qpInterP":     8,   # uint32, rc_ptr[2]
    "constQP.qpInterB":    12,   # uint32, rc_ptr[3]
    "constQP.qpIntra":     16,   # uint32, rc_ptr[4]
    "averageBitRate":      20,   # uint32, rc_ptr[5]
    "maxBitRate":          24,   # uint32, rc_ptr[6]
    "vbvBufferSize":       28,   # uint32, rc_ptr[7]
    "vbvInitialDelay":     32,   # uint32, rc_ptr[8]
    "bitfield":            36,   # uint32, rc_ptr[9]
    "minQP.qpInterP":      40,   # uint32, rc_ptr[10]
    "maxQP.qpInterP":      52,   # uint32, rc_ptr[13]
    "initialRCQP.qpInterP":64,   # uint32, rc_ptr[16]
    "temporallayerIdxMask":76,   # uint32, rc_ptr[19]
    "temporalLayerQP":     80,   # uint8[8]
    "targetQuality":       88,   # uint8  ← VERIFIED: used to be 76 (union) or 116 (obsolete)
    "targetQualityLSB":    89,   # uint8
    "lookaheadDepth":      90,   # uint16
    "qpMapMode":           96,   # uint32, rc_ptr[24]
    "multiPass":          100,   # uint32, rc_ptr[25]  ← SEPARATE field
    "alphaLayerBitrateRatio":104,  # uint32, rc_ptr[26]
    "lookaheadLevel":     112,   # uint32, rc_ptr[28]
    "viewBitrateRatios":  116,   # uint8[7]  ← old wrong tq target!
}

# Old wrong offsets — for comparison testing only
RC_OLD_WRONG = {
    "averageBitRate":  12,   # was rc_ptr[3]  → overwrote qpInterB
    "maxBitRate":      16,   # was rc_ptr[4]  → overwrote qpIntra
    "targetQuality":  116,   # was byte 116   → viewBitrateRatios (no-op)
    "targetQualityLSB":120,
    "lookaheadDepth": 121,
}

# Default test parameters
TEST_WIDTH = 640
TEST_HEIGHT = 480
TEST_FPS = 30
TEST_FRAMES = 30
TEST_PRESET = "medium"
TEST_CRF = 28  # → targetQuality = 51 - 28 = 23 (but test uses tq=1 and tq=51 extremes)


# ═══════════════════════════════════════════════════════════════════════════════
# NVENC ctypes structs (minimal)
# ═══════════════════════════════════════════════════════════════════════════════

class _NvGuid(ctypes.Structure):
    _fields_ = [
        ("Data1", c_uint32),
        ("Data2", c_uint16),
        ("Data3", c_uint16),
        ("Data4", c_uint8 * 8),
    ]


# [FIX-STRUCTS] Copied from v6.4.3.1 production code — correct sizes so
# GetEncodePresetConfig doesn't overflow the ctypes buffer.
class _NvEncConfigH264VUIParameters(ctypes.Structure):
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

class _NvEncConfigH264(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("enableTemporalSVC",         c_uint32),
        ("enableTemporalSVC_1",       c_uint32),
        ("profileLevel",              c_uint32),
        ("chromaFormatIDC",           c_uint32),
        ("reserved1",                 c_uint32 * 13),
        ("maxNumRefFramesInDPB",      c_uint32),
        ("reserved2",                 c_uint32 * 3),
        ("idrPeriod",                 c_uint32),
        ("repeatSPSPPS",              c_uint32),
        ("reserved10",                c_uint32 * 4),
        ("vuiParameters",             _NvEncConfigH264VUIParameters),
        ("reserved12",                c_uint32 * 222),
    ]

class _NvEncConfig(ctypes.Structure):
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
        ("reserved5",       c_uint32 * 172),
        ("encodeCodecConfig", _NvEncConfigH264),
        ("reserved7",      c_uint32 * 252),
    ]

class _NvEncPresetConfig(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("version",       c_uint32),
        ("presetConfig",  _NvEncConfig),
        ("reserved",      c_uint32 * 256),
    ]


class _NvEncInitializeParams(ctypes.Structure):
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
        ("reserved1",               c_uint8 * 1136),  # offset 152
        ("reserved2",               c_void_p * 64),   # offset 1288
    ]


class _NvEncOpenEncodeSessionExParams(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("version", c_uint32),
        ("deviceType", c_uint32),
        ("device", c_void_p),
        ("reserved", c_void_p),
        ("apiVersion", c_uint32),
        ("reserved1", c_uint8 * 253 * 4),
        ("reserved2", c_void_p * 64),
    ]


class _NvEncPicParams(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("version", c_uint32),
        ("inputWidth", c_uint32),
        ("inputHeight", c_uint32),
        ("inputPitch", c_uint32),
        ("encodePicFlags", c_uint32),
        ("frameIdx", c_uint32),
        ("inputTimeStamp", c_uint64),
        ("inputDuration", c_uint64),
        ("inputBuffer", c_void_p),
        ("outputBitstream", c_void_p),
        ("completionEvent", c_void_p),
        ("bufferFmt", c_uint32),
        ("pictureStruct", c_uint32),
        ("pictureType", c_uint32),
        ("codecPicParams", c_uint8 * 1048),
        ("reserved", c_uint32 * 544),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Function prototypes & indices
# ═══════════════════════════════════════════════════════════════════════════════

_NvEncodeAPICreateInstanceProto = CFUNCTYPE(c_uint32, c_void_p)
_NvEncOpenEncodeSessionExProto = CFUNCTYPE(c_uint32, POINTER(_NvEncOpenEncodeSessionExParams), POINTER(c_void_p))
_NvEncCreateEncoderProto = CFUNCTYPE(c_uint32, c_void_p, POINTER(_NvEncInitializeParams))
_NvEncDestroyEncoderProto = CFUNCTYPE(c_uint32, c_void_p)
_NvEncEncodePictureProto = CFUNCTYPE(c_uint32, c_void_p, POINTER(_NvEncPicParams))

_FUNC_IDX = {
    "GetEncodeGUIDCount": 1,
    "GetEncodeGUIDs": 4,
    "GetEncodePresetGUIDs": 9,
    "GetEncodePresetConfig": 10,
    "InitializeEncoder": 11,
    "CreateInputBuffer": 12,
    "DestroyInputBuffer": 13,
    "CreateBitstreamBuffer": 14,
    "DestroyBitstreamBuffer": 15,
    "EncodePicture": 16,
    "LockBitstream": 17,
    "UnlockBitstream": 18,
    "LockInputBuffer": 19,
    "UnlockInputBuffer": 20,
    "MapInputResource": 25,
    "UnmapInputResource": 26,
    "DestroyEncoder": 27,
    "OpenEncodeSessionEx": 29,
    "RegisterResource": 30,
    "UnregisterResource": 31,
    "GetEncodePresetConfigEx": 39,
}

_PRESET_P_INDEX = {
    "ultrafast": 0, "superfast": 0,
    "veryfast": 1, "faster": 2,
    "fast": 3, "medium": 4,
    "slow": 5, "slower": 6, "veryslow": 6, "placebo": 6,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level CUDA context (v7: one-time init, all encoders share)
# ═══════════════════════════════════════════════════════════════════════════════
#
# CRITICAL: DO NOT push/pop context in encode_frame or close(). The module-level
# context stays current for the entire process lifetime. Driver API cuCtxPushCurrent
# after cuDevicePrimaryCtxRetain succeeds exactly ONCE; subsequent pushes cause
# CUDA_ERROR_LAUNCH_FAILURE (201) → context stack corruption → segfault.
#
# Strategy: establish Driver API primary context FIRST, then let PyTorch Runtime
# reuse it. All encoders share this one context — no per-encoder or per-frame
# context management.

_GLOBAL_CUDA_CTX = c_void_p(None)
_GLOBAL_LIBCUDA = None
_GLOBAL_NVENC_DLL_PATH = None
_GLOBAL_NVENC_DLL = None
_INITIALIZED = False


def _find_nvenc_dll():
    if sys.platform == "win32":
        for c in ["nvEncodeAPI64.dll"]:
            if os.path.exists(c) or not os.path.dirname(c):
                try:
                    return c, CDLL(c)
                except OSError:
                    continue
    else:
        for c in [
            "/usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1",
            "/usr/lib64/libnvidia-encode.so.1",
            "libnvidia-encode.so.1",
        ]:
            try:
                return c, CDLL(c)
            except OSError:
                continue
    raise RuntimeError("Cannot find NVENC DLL")


def _ensure_global_cuda_context():
    """One-time CUDA context setup. Idempotent — safe to call repeatedly."""
    global _GLOBAL_CUDA_CTX, _GLOBAL_LIBCUDA, _GLOBAL_NVENC_DLL_PATH, _GLOBAL_NVENC_DLL, _INITIALIZED
    if _INITIALIZED and _GLOBAL_CUDA_CTX.value is not None:
        return  # already done

    import torch
    # Step 1: Driver API — establish primary context BEFORE PyTorch touches CUDA
    _GLOBAL_NVENC_DLL_PATH, _GLOBAL_NVENC_DLL = _find_nvenc_dll()

    libname = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
    _GLOBAL_LIBCUDA = CDLL(libname)
    _GLOBAL_LIBCUDA.cuInit(0)

    _GLOBAL_LIBCUDA.cuDeviceGet.restype = c_uint32
    _GLOBAL_LIBCUDA.cuDeviceGet.argtypes = [POINTER(c_int32), c_int32]
    _dev = c_int32(0)
    r = _GLOBAL_LIBCUDA.cuDeviceGet(byref(_dev), 0)
    if r != 0:
        raise RuntimeError(f"cuDeviceGet(0) failed: {r}")

    _GLOBAL_LIBCUDA.cuDevicePrimaryCtxRetain.restype = c_uint32
    _GLOBAL_LIBCUDA.cuDevicePrimaryCtxRetain.argtypes = [POINTER(c_void_p), c_int32]
    r = _GLOBAL_LIBCUDA.cuDevicePrimaryCtxRetain(byref(_GLOBAL_CUDA_CTX), _dev)
    if r != 0 or _GLOBAL_CUDA_CTX.value is None:
        raise RuntimeError(f"cuDevicePrimaryCtxRetain failed: {r}")

    _GLOBAL_LIBCUDA.cuCtxPushCurrent.restype = c_uint32
    _GLOBAL_LIBCUDA.cuCtxPushCurrent.argtypes = [c_void_p]
    r = _GLOBAL_LIBCUDA.cuCtxPushCurrent(_GLOBAL_CUDA_CTX)
    if r != 0:
        raise RuntimeError(f"cuCtxPushCurrent failed: {r}")

    # Step 2: Warm up PyTorch — it reuses the already-current primary context
    _a = torch.randn(128, 128, device='cuda')
    _b = torch.randn(128, 128, device='cuda')
    _c = torch.matmul(_a, _b)
    del _a, _b, _c
    torch.cuda.synchronize()

    # Step 3: Setup cuMemcpy2D (used by encode_frame)
    _GLOBAL_LIBCUDA.cuMemcpy2D_v2.restype = c_uint32
    _GLOBAL_LIBCUDA.cuMemcpy2D_v2.argtypes = [c_void_p]

    _INITIALIZED = True
    print(f"  [GPU context ready]", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DiagEncoder — minimal H.264 encoder with configurable rcParams writes
# ═══════════════════════════════════════════════════════════════════════════════

class DiagEncoder:
    """Minimal NVENC encoder that writes targetQuality at a specified byte offset.

    By comparing total_output_bytes for tq=1 vs tq=51 at the same offset,
    we can determine if the driver actually reads from that position."""

    def __init__(self, width, height, fps, qp, rate_mode,
                 tq_offset, tq_value, preset="medium",
                 use_correct_layout=True):
        self._width = width
        self._height = height
        self._fps = fps
        self._qp = qp
        self._rate_mode = rate_mode
        self._tq_offset = tq_offset
        self._tq_value = tq_value
        self._use_correct = use_correct_layout
        self._preset_name = preset.lower()
        self._encoder = c_void_p(None)
        self._frame_idx = 0
        self._lock = threading.Lock()
        self._input_buf_handle = c_void_p(None)
        self._bs_handle = c_void_p(None)
        self._total_output_bytes = 0

        # ── v7: use module-level global CUDA context (one-time init) ──
        _ensure_global_cuda_context()
        self._libcuda = _GLOBAL_LIBCUDA
        self._dll = _GLOBAL_NVENC_DLL
        cuda_ctx = _GLOBAL_CUDA_CTX

        try:
            _get_max_ver = self._dll.NvEncodeAPIGetMaxSupportedVersion
            _get_max_ver.restype = c_uint32
            _get_max_ver.argtypes = [POINTER(c_uint32)]
            _mv = c_uint32(0)
            _get_max_ver(byref(_mv))
            _nvenc_api_version = _mv.value if _mv.value > 0 else _NVENCAPI_VERSION
        except Exception:
            _nvenc_api_version = _NVENCAPI_VERSION
        _FT_SIZE = 2552
        func_table = (c_uint8 * _FT_SIZE)()
        cast(func_table, POINTER(c_uint32))[0] = _sdk13_ver(2)
        s = _NvEncodeAPICreateInstanceProto(("NvEncodeAPICreateInstance", self._dll))(cast(func_table, c_void_p))
        if s != NV_ENC_SUCCESS:
            raise RuntimeError(f"NvEncodeAPICreateInstance failed code={s}")
        self._func_ptrs = cast(byref(func_table, 8), POINTER(c_void_p))
        self._func_table_raw = func_table

        def gf(idx):
            a = self._func_ptrs[idx]
            return a if (a and a != 0) else None

        open_addr = gf(_FUNC_IDX["OpenEncodeSessionEx"])
        if open_addr is None:
            raise RuntimeError("OpenEncodeSessionEx NA")
        open_sess = _NvEncOpenEncodeSessionExProto(open_addr)

        for _av in sorted({_NVENCAPI_VERSION, _nvenc_api_version, 0xd0, 0xc0}, reverse=True):
            sp = _NvEncOpenEncodeSessionExParams()
            sp.version = _sdk13_ver(1)
            sp.deviceType = NV_ENC_DEVICE_TYPE_CUDA
            sp.device = cuda_ctx
            sp.apiVersion = _av
            self._encoder = c_void_p(None)
            if open_sess(byref(sp), byref(self._encoder)) == NV_ENC_SUCCESS:
                break

        _GetEncodeGUIDCountProto = CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint32))
        cv = c_uint32(0)
        _GetEncodeGUIDCountProto(gf(_FUNC_IDX["GetEncodeGUIDCount"]))(self._encoder, byref(cv))

        _GetEncodeGUIDsProto = CFUNCTYPE(c_uint32, c_void_p, POINTER(_NvGuid), c_uint32, POINTER(c_uint32))
        ga = (_NvGuid * cv.value)()
        memset(cast(ga, c_void_p), 0, sizeof(ga))
        ac = c_uint32(0)
        _GetEncodeGUIDsProto(gf(_FUNC_IDX["GetEncodeGUIDs"]))(self._encoder, ga, cv.value, byref(ac))
        codec_guid = ga[0]

        _GetEncodePresetGUIDsProto = CFUNCTYPE(c_uint32, c_void_p, _NvGuid, POINTER(_NvGuid), c_uint32, POINTER(c_uint32))
        pga = (_NvGuid * 64)()
        memset(cast(pga, c_void_p), 0, sizeof(pga))
        pcnt = c_uint32(0)
        _GetEncodePresetGUIDsProto(gf(_FUNC_IDX["GetEncodePresetGUIDs"]))(self._encoder, codec_guid, pga, 64, byref(pcnt))
        pi = _PRESET_P_INDEX.get(self._preset_name, 4)
        pi = min(pi, pcnt.value - 1)
        preset_guid = pga[pi]

        gpa = gf(_FUNC_IDX["GetEncodePresetConfig"])
        if gpa is None:
            gpa = gf(_FUNC_IDX["GetEncodePresetConfigEx"])
        _GP = CFUNCTYPE(c_uint32, c_void_p, _NvGuid, _NvGuid, POINTER(_NvEncPresetConfig))

        pcfg = _NvEncPresetConfig()
        memset(byref(pcfg), 0, sizeof(pcfg))
        pcfg.version = NV_ENC_PRESET_CONFIG_VER
        cast(byref(pcfg, 8), POINTER(c_uint32))[0] = NV_ENC_CONFIG_VER
        s = _GP(gpa)(self._encoder, codec_guid, preset_guid, byref(pcfg))
        if s != NV_ENC_SUCCESS:
            raise RuntimeError(f"GetEncodePresetConfig failed code={s}")

        ec = cast(byref(pcfg, 8), POINTER(_NvEncConfig)).contents
        ec.gopLength = int(fps)
        ec.frameIntervalP = 1
        ec.encodeCodecConfig.chromaFormatIDC = 1
        ec.encodeCodecConfig.idrPeriod = int(fps)
        ec.encodeCodecConfig.maxNumRefFramesInDPB = 4
        ec.encodeCodecConfig.repeatSPSPPS = 1

        # ── rcParams: correct SEQUENTIAL layout (nvEncodeAPI.h master, GPU-verified) ──
        # preset_config byte offset = 8 + 40  (pcfg SDK padding + NV_ENC_CONFIG.rcParams)
        rc_ptr = cast(byref(pcfg, 8 + 40), POINTER(c_uint32))
        rc_ptr[0] = _sdk13_ver(1)  # NV_ENC_RC_PARAMS_VER
        _qp_val = max(1, qp) if qp > 0 else 28

        if self._rate_mode == 'vbr_hq':
            rc_ptr[1] = NV_ENC_PARAMS_RC_VBR_HQ  # rc_ptr[1]@4

            _est_br = max(50000000, int(width * height * fps * 3.0))

            if self._use_correct:
                # ── CORRECT layout (nvEncodeAPI.h, GPU-verified) ──
                rc_ptr[5] = _est_br               # averageBitRate @20 ✅
                rc_ptr[6] = _est_br * 2           # maxBitRate @24 ✅
                # targetQuality: uint8 at rcParams byte 88
                tq8_ptr = cast(byref(pcfg, 8 + 40 + self._tq_offset), POINTER(c_uint8))
            else:
                # ── Target offset into THE specific candidate ──
                # For sweeping: avgbr/maxbr at correct positions, only tq varies
                rc_ptr[5] = _est_br               # averageBitRate @20 (always correct)
                rc_ptr[6] = _est_br * 2           # maxBitRate @24 (always correct)
                tq8_ptr = cast(byref(pcfg, 8 + 40 + self._tq_offset), POINTER(c_uint8))

            tq8_ptr[0] = self._tq_value & 0xFF

            # Enable AQ (bit3) + Temporal AQ (bit8) in bitfield @36
            rc_ptr[9] = rc_ptr[9] | (1 << 3) | (1 << 8)

        elif self._rate_mode == 'constqp':
            rc_ptr[1] = NV_ENC_PARAMS_RC_CONSTQP
            rc_ptr[2] = _qp_val  # qpInterP@8
            rc_ptr[3] = _qp_val  # qpInterB@12
            rc_ptr[4] = _qp_val  # qpIntra@16
        else:
            raise ValueError(f"Unknown rate_mode: {self._rate_mode}")

        # multiPass — SEPARATE field at offset 100, NOT in bitfield
        # (Disabled for VBR_HQ testing; enable if testing lookahead)
        # rc_ptr[25] = 1  # multiPass=1pass

        ip = _NvEncInitializeParams()
        memset(byref(ip), 0, sizeof(ip))
        ip.version = NV_ENC_INITIALIZE_PARAMS_VER
        ip.encodeGUID = codec_guid
        ip.presetGUID = preset_guid
        ip.encodeWidth = width
        ip.encodeHeight = height
        ip.darWidth = width
        ip.darHeight = height
        ip.frameRateNum = int(fps * 1000)
        ip.frameRateDen = 1000
        ip.maxEncodeWidth = width
        ip.maxEncodeHeight = height
        ip.enablePTD = 1
        ip.encodeConfig = cast(byref(pcfg, 8), c_void_p)

        ia = gf(_FUNC_IDX["InitializeEncoder"])
        s = _NvEncCreateEncoderProto(ia)(self._encoder, byref(ip))
        if s != NV_ENC_SUCCESS:
            raise RuntimeError(f"InitializeEncoder failed code={s}")

        cb = (c_uint8 * 776)()
        memset(cb, 0, 776)
        cast(cb, POINTER(c_uint32))[0] = NV_ENC_CREATE_INPUT_BUFFER_VER
        cast(byref(cb, 4), POINTER(c_uint32))[0] = width
        cast(byref(cb, 8), POINTER(c_uint32))[0] = height
        cast(byref(cb, 16), POINTER(c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12
        s = CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 776))(gf(_FUNC_IDX["CreateInputBuffer"]))(self._encoder, cb)
        if s != 0:
            raise RuntimeError(f"CreateInputBuffer failed code={s}")
        rp = cast(byref(cb, 24), POINTER(c_void_p))[0]
        self._input_buf_handle = c_void_p(rp if isinstance(rp, int) else (rp.value or 0))

        bb = (c_uint8 * 776)()
        memset(bb, 0, 776)
        cast(bb, POINTER(c_uint32))[0] = NV_ENC_CREATE_BITSTREAM_BUFFER_VER
        s = CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 776))(gf(_FUNC_IDX["CreateBitstreamBuffer"]))(self._encoder, bb)
        if s != 0:
            self._destroy_ib()
            raise RuntimeError(f"CreateBitstreamBuffer failed code={s}")
        rbs = cast(byref(bb, 16), POINTER(c_void_p))[0]
        self._bs_handle = c_void_p(rbs if isinstance(rbs, int) else (rbs.value or 0))

        # cuMemcpy2D already set up in _ensure_global_cuda_context()
        layout_tag = "correct-sequential" if self._use_correct else "CANDIDATE-SWEEP"
        print(f" | OK tq={self._tq_value}@byte{self._tq_offset} [{layout_tag}]", flush=True)

    def _destroy_ib(self):
        if self._input_buf_handle.value is None:
            return
        try:
            a = self._func_ptrs[_FUNC_IDX["DestroyInputBuffer"]]
            if a:
                CFUNCTYPE(c_uint32, c_void_p, c_void_p)(a)(self._encoder, self._input_buf_handle)
        except Exception:
            pass
        self._input_buf_handle = c_void_p(None)

    def _destroy_bb(self):
        if self._bs_handle.value is None:
            return
        try:
            a = self._func_ptrs[_FUNC_IDX["DestroyBitstreamBuffer"]]
            if a:
                CFUNCTYPE(c_uint32, c_void_p, c_void_p)(a)(self._encoder, self._bs_handle)
        except Exception:
            pass
        self._bs_handle = c_void_p(None)

    def close(self):
        with self._lock:
            if self._encoder.value is None:
                return
            self._destroy_ib()
            self._destroy_bb()
            da = self._func_ptrs[_FUNC_IDX["DestroyEncoder"]]
            if da:
                _NvEncDestroyEncoderProto(da)(self._encoder)
            self._encoder = c_void_p(None)
            # No context restore — global context stays current for the process lifetime

    def encode_frame(self, nv12_gpu_tensor):
        # Global CUDA context is already current — no push/pop needed
        import torch
        try:
            with self._lock:
                nvh = self._height + self._height // 2
                W = self._width

                lb = (c_uint8 * 1544)()
                memset(lb, 0, 1544)
                cast(lb, POINTER(c_uint32))[0] = NV_ENC_LOCK_INPUT_BUFFER_VER
                cast(byref(lb, 8), POINTER(c_void_p))[0] = self._input_buf_handle

                _Li = CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 1544))
                _Ui = CFUNCTYPE(c_uint32, c_void_p, c_void_p)

                s = _Li(self._func_ptrs[_FUNC_IDX["LockInputBuffer"]])(self._encoder, lb)
                if s != 0:
                    raise RuntimeError(f"LockInputBuffer code={s}")

                rm = cast(byref(lb, 16), POINTER(c_void_p))[0]
                mp = rm if isinstance(rm, int) else (rm.value or 0)
                ap = cast(byref(lb, 24), POINTER(c_uint32))[0]
                if not mp:
                    _Ui(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(self._encoder, self._input_buf_handle)
                    raise RuntimeError("LockInputBuffer NULL ptr")

                _cp = (c_uint8 * 128)()
                memset(_cp, 0, 128)
                cast(byref(_cp, 16), POINTER(c_uint32))[0] = 2  # CU_MEMORYTYPE_DEVICE
                cast(byref(_cp, 32), POINTER(c_void_p))[0] = c_void_p(nv12_gpu_tensor.data_ptr())
                cast(byref(_cp, 48), POINTER(c_size_t))[0] = W
                cast(byref(_cp, 72), POINTER(c_uint32))[0] = 2
                cast(byref(_cp, 88), POINTER(c_void_p))[0] = c_void_p(mp)
                cast(byref(_cp, 104), POINTER(c_size_t))[0] = ap if ap > 0 else W
                cast(byref(_cp, 112), POINTER(c_size_t))[0] = W
                cast(byref(_cp, 120), POINTER(c_size_t))[0] = nvh
                r = self._libcuda.cuMemcpy2D_v2(cast(_cp, c_void_p))
                if r != 0:
                    _Ui(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(self._encoder, self._input_buf_handle)
                    raise RuntimeError(f"cuMemcpy2D code={r}")

                _Ui(self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])(self._encoder, self._input_buf_handle)

                pb = (c_uint8 * 3360)()
                memset(pb, 0, 3360)
                cast(pb, POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
                cast(byref(pb, 4), POINTER(c_uint32))[0] = W
                cast(byref(pb, 8), POINTER(c_uint32))[0] = self._height
                cast(byref(pb, 12), POINTER(c_uint32))[0] = ap if ap > 0 else W
                cast(byref(pb, 24), POINTER(c_uint64))[0] = self._frame_idx
                cast(byref(pb, 40), POINTER(c_void_p))[0] = self._input_buf_handle
                cast(byref(pb, 48), POINTER(c_void_p))[0] = self._bs_handle
                cast(byref(pb, 64), POINTER(c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12
                cast(byref(pb, 68), POINTER(c_uint32))[0] = NV_ENC_PIC_STRUCT_FRAME

                epf = _NvEncEncodePictureProto(self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                epf(self._encoder, cast(pb, POINTER(_NvEncPicParams)))
                self._frame_idx += 1

                _Lb = CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 1544))
                _Ub = CFUNCTYPE(c_uint32, c_void_p, c_void_p)
                lr = (c_uint8 * 1544)()
                memset(lr, 0, 1544)
                cast(lr, POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
                cast(byref(lr, 8), POINTER(c_void_p))[0] = self._bs_handle
                bs = _Lb(self._func_ptrs[_FUNC_IDX["LockBitstream"]])(self._encoder, lr)
                if bs == NV_ENC_SUCCESS:
                    bsz = cast(byref(lr, 36), POINTER(c_uint32))[0]
                    self._total_output_bytes += bsz
                    _Ub(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(self._encoder, self._bs_handle)

        except Exception:
            # Re-raise NVENC errors — no context cleanup needed (global stays current)
            raise

    def flush(self):
        with self._lock:
            if self._encoder.value is None:
                return
            pb = (c_uint8 * 3360)()
            memset(pb, 0, 3360)
            cast(pb, POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
            cast(byref(pb, 16), POINTER(c_uint32))[0] = 0x8  # EOS
            cast(byref(pb, 40), POINTER(c_void_p))[0] = c_void_p(None)
            cast(byref(pb, 48), POINTER(c_void_p))[0] = self._bs_handle
            _NvEncEncodePictureProto(self._func_ptrs[_FUNC_IDX["EncodePicture"]])(self._encoder, cast(pb, POINTER(_NvEncPicParams)))

            _Lb = CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 1544))
            _Ub = CFUNCTYPE(c_uint32, c_void_p, c_void_p)
            while True:
                lr = (c_uint8 * 1544)()
                memset(lr, 0, 1544)
                cast(lr, POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
                cast(byref(lr, 4), POINTER(c_uint32))[0] = 1
                cast(byref(lr, 8), POINTER(c_void_p))[0] = self._bs_handle
                bs = _Lb(self._func_ptrs[_FUNC_IDX["LockBitstream"]])(self._encoder, lr)
                if bs != NV_ENC_SUCCESS:
                    break
                bsz = cast(byref(lr, 36), POINTER(c_uint32))[0]
                if bsz == 0:
                    _Ub(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(self._encoder, self._bs_handle)
                    break
                self._total_output_bytes += bsz
                _Ub(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(self._encoder, self._bs_handle)


# ═══════════════════════════════════════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _create_test_nv12_frame(width, height, frame_idx, seed=42):
    import numpy as np
    np.random.seed(seed + frame_idx)
    H, W = height, width
    Hh = H // 2

    y = np.zeros((H, W), dtype=np.uint8)
    bw = W // 8
    for b in range(8):
        x0, x1 = b * bw, (b + 1) * bw if b < 7 else W
        y[:, x0:x1] = [30, 80, 130, 180, 220, 160, 100, 50][b]
    tn = np.random.randint(-8, 9, (H, W)).astype(np.float32)
    y = np.clip(y.astype(np.float32) + tn * (1 + frame_idx % 5) * 0.3, 0, 255).astype(np.uint8)

    uv = np.zeros((Hh, W), dtype=np.uint8)
    for b in range(8):
        x0, x1 = b * bw, (b + 1) * bw if b < 7 else W
        uv[:, x0:x1] = 108 if b % 2 == 0 else 148

    nv12 = np.zeros((H + Hh, W), dtype=np.uint8)
    nv12[:H] = y
    nv12[H:] = uv
    import torch
    return torch.from_numpy(nv12.copy()).cuda()


def test_encoder(tq_offset, tq_value, use_correct=True):
    """Create encoder, encode frames, return (total_bytes, error_code).
    error_code is None for success, or the NVENC error code string for diagnostic display."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    enc = None
    error_code = None
    try:
        enc = DiagEncoder(
            TEST_WIDTH, TEST_HEIGHT, TEST_FPS,
            qp=TEST_CRF, rate_mode='vbr_hq',
            tq_offset=tq_offset, tq_value=tq_value,
            preset=TEST_PRESET,
            use_correct_layout=use_correct)
        for i in range(TEST_FRAMES):
            frm = _create_test_nv12_frame(TEST_WIDTH, TEST_HEIGHT, i)
            enc.encode_frame(frm)
        enc.flush()
        result = enc._total_output_bytes
    except RuntimeError as e:
        msg = str(e)
        if "code=8" in msg:
            error_code = "NV_ENC_ERR_INVALID_PARAM (code=8)"
        elif "code=15" in msg:
            error_code = "NV_ENC_ERR_INVALID_VERSION (code=15)"
        elif "code=201" in msg or "LAUNCH_FAILURE" in msg:
            error_code = "CUDA_ERROR_LAUNCH_FAILURE (201)"
        else:
            error_code = msg[:80]
        result = -1
    except Exception as e:
        error_code = str(e)[:80]
        result = -1
    finally:
        if enc:
            try:
                enc.close()
            except Exception:
                pass
    time.sleep(0.3)
    return result, error_code


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_verify(verify_offsets):
    """For each offset, encode with tq=1 and tq=51. Returns {offset: {'works': bool, ...}}."""
    results = {}
    for offset in verify_offsets:
        print(f"  offset={offset:>4} tq=1:  ", end="", flush=True)
        b1, ec1 = test_encoder(offset, 1, use_correct=True)
        print(f"  offset={offset:>4} tq=51: ", end="", flush=True)
        b51, ec2 = test_encoder(offset, 51, use_correct=True)

        if b1 > 0 and b51 > 0:
            diff_pct = abs(b51 - b1) / max(b1, b51) * 100
            works = diff_pct > 50.0  # Real targetQuality: tq=1 vs tq=51 shows >90% diff.
                                        # False positives (config corruption side-effects): ~30% diff.
                                        # Threshold at 50% cleanly separates the two.
            status = "VALID (targetQuality works here)" if works else "INVALID (no effect — not targetQuality)"
            print(f"  => tq=1→{b1:,}B  tq=51→{b51:,}B  diff={diff_pct:.1f}%  {status}")
        elif ec1 == "NV_ENC_ERR_INVALID_PARAM (code=8)" or ec2 == "NV_ENC_ERR_INVALID_PARAM (code=8)":
            works = False
            diff_pct = 0
            # code=8 = this offset is a critical config field, writing tq to it corrupts the config
            # This is expected for uint32 fields (qpMapMode@96, multiPass@100, lookaheadLevel@112)
            print(f"  => INVALID_PARAM (code=8) — this offset is a sensitive config field, not targetQuality")
        else:
            works = False
            diff_pct = 0
            err_info = ec1 or ec2 or "unknown"
            print(f"  => ERROR: tq=1={b1}({ec1}) tq=51={b51}({ec2})")

        results[offset] = {
            "works": works,
            "diff_pct": diff_pct,
            "bytes_tq1": b1,
            "bytes_tq51": b51,
            "error_tq1": ec1,
            "error_tq2": ec2,
        }
        print()

    return results


def main():
    import torch
    print(f"PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")
    print()
    print("=" * 72)
    print("NV_ENC_RC_PARAMS 字段偏移诊断 (nvEncodeAPI.h master, SEQUENTIAL layout)")
    print(f"基础参数: {TEST_WIDTH}x{TEST_HEIGHT}@{TEST_FPS}fps, {TEST_FRAMES}帧, VBR_HQ")
    print()
    print("已 GPU 验证的正确布局 (2026-06-09):")
    print("  averageBitRate@20(rc_ptr[5]), maxBitRate@24(rc_ptr[6])")
    print("  targetQuality@88(uint8), lookaheadDepth@90(uint16)")
    print("  bitfield@36(rc_ptr[9]): AQ=bit3, TemporalAQ=bit8")
    print("  multiPass@100(rc_ptr[25])  ← 独立字段，不在 bitfield 中")
    print("=" * 72)
    print()

    # ── Verify known correct offset (88) ──
    print("═══ TEST: Verify correct layout, targetQuality@88 ═══")
    correct = run_verify([88])

    # ── Verify old wrong offset (116) ──
    print("═══ TEST: Old wrong offset, targetQuality@116 (should NOT work) ═══")
    wrong = run_verify([116])

    # ── Sweep mode ──
    if "--sweep" in sys.argv:
        print("═══ SWEEP: Scan all candidate offsets ═══")
        # Scan plausible targetQuality locations: byte 76-128
        sweep_offsets = [76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124]
        sweep_results = run_verify(sweep_offsets)
    else:
        sweep_results = {}

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    all_results = {**correct, **wrong, **sweep_results}
    working = [off for off, r in all_results.items() if r["works"]]
    no_effect = [off for off, r in all_results.items()
                 if not r["works"] and r["bytes_tq1"] > 0 and r["bytes_tq51"] > 0]
    invalid_param = [off for off, r in all_results.items()
                     if (r.get("error_tq1") or "").startswith("NV_ENC_ERR_INVALID_PARAM")
                     or (r.get("error_tq2") or "").startswith("NV_ENC_ERR_INVALID_PARAM")]
    other_error = [off for off, r in all_results.items()
                   if off not in working and off not in no_effect and off not in invalid_param]

    expected_working = 88  # nvEncodeAPI.h verified
    expected_not = 116      # old wrong code

    # offset → field name mapping for diagnostic context
    RC_OFFSET_FIELD_MAP = {
        76:  "temporallayerIdxMask[0] (uint32)",  88: "targetQuality (uint8)",
        80:  "temporalLayerQP[0] (uint8)",         92: "lowDelayKeyFrameScale (uint8)",
        84:  "temporalLayerQP[4] (uint8)",         96: "qpMapMode[0] (uint32)",
        100: "multiPass[0] (uint32)",              104: "alphaLayerBitrateRatio[0] (uint32)",
        108: "cbQPIndexOffset (int8)",             112: "lookaheadLevel[0] (uint32)",
        116: "viewBitrateRatios[0] (uint8)",       120: "viewBitrateRatios[4] (uint8)",
        124: "reserved1[0] (uint32)",
    }

    print(f"  targetQuality 真正生效:  {working if working else '(none)'}")
    if no_effect:
        print(f"  无效果 (写入不受影响):")
        for off in sorted(no_effect):
            fname = RC_OFFSET_FIELD_MAP.get(off, "unknown")
            r = all_results[off]
            diff_pct = abs(r["bytes_tq1"] - r["bytes_tq51"]) / max(r["bytes_tq1"], r["bytes_tq51"]) * 100 if max(r["bytes_tq1"], r["bytes_tq51"]) > 0 else 0
            print(f"      offset {off:3d} → {fname}  (diff={diff_pct:.1f}%)")
    else:
        print(f"  无效果 (写入不受影响):      (none)")
    if invalid_param:
        print(f"  写入破坏配置 (code=8):")
        for off in sorted(invalid_param):
            fname = RC_OFFSET_FIELD_MAP.get(off, "unknown")
            print(f"      offset {off:3d} → {fname}  ← uint8 写入破坏该字段 → InitializeEncoder code=8")
    else:
        print(f"  写入破坏配置 (code=8):      (none)")
    if other_error:
        print(f"  其他错误:                     {other_error}")
    print()

    if expected_working in working:
        print(f"  PASS targetQuality@88 → CORRECT (nvEncodeAPI.h + GPU 双重验证)")
    else:
        print(f"  FAIL targetQuality@88 未生效 — 需要调查")

    if expected_not in no_effect:
        print(f"  PASS targetQuality@116 → 确认无效 (旧代码的 bug 已证实)")
    elif expected_not in invalid_param:
        print(f"  WARN targetQuality@116 → 触发 code=8 (写入保留字段)")
    else:
        print(f"  WARN targetQuality@116 → 意外生效?")

    # Sweep-specific analysis
    if sweep_results:
        false_positives = [off for off in working if off != 88]
        if false_positives:
            print(f"\n  WARN 假阳性偏移 (副作用非真正 targetQuality): {false_positives}")
            print(f"       这些偏移对应其他 rcParams 字段 — 写入 tq 值破坏了它们的功能")

    return 0


if __name__ == '__main__':
    sys.exit(main())
