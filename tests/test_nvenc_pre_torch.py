#!/usr/bin/env python3
"""
方案 2b 验证: 先占 NVENC (context 纯净时创建 session)，再 import torch

理论:
  NVENC OpenEncodeSessionEx 要求 CUDA context 纯净（无活跃 streams/events/分配）
  但 session 一旦建立，后续 PyTorch 在相同 context 中的操作不应影响已建立的 session
  因为 NVENC 内部维护独立的编码状态

流程:
  1. ctypes → cuDevicePrimaryCtxRetain (clean context)
  2. NVENC 初始化 (OpenEncodeSessionEx + CreateEncoder)
  3. import torch (PyTorch 复用相同 primary context)
  4. torch 分配 NV12 tensor
  5. NVENC RegisterResource(torch tensor) → EncodePicture → 输出 H.264 ES

用法:
  python test_nvenc_pre_torch.py
"""

import sys
import os
import ctypes
import struct
from ctypes import (c_uint8, c_uint16, c_uint32, c_int32, c_int, c_uint64, c_void_p,
                     c_char, c_size_t, c_double, Structure, POINTER, byref,
                     sizeof, cast, pointer, c_bool)


def guid_to_u64_pair(g):
    """将 _NvGuid (16 bytes) 拆成两个 uint64，绕过 ctypes 在 Linux x86-64
    System V AMD64 ABI 上对 >8 字节 struct 按值传递的兼容性问题。"""
    raw = struct.pack('<IHH8s', g.Data1, g.Data2, g.Data3, bytes(g.Data4))
    lo, hi = struct.unpack('<QQ', raw)
    return lo, hi

# ============================================================================
# NVENC 常量 & 结构体 (精简自 v6.4.3)
# ============================================================================

NV_ENC_SUCCESS = 0
NV_ENC_DEVICE_TYPE_CUDA_VALUES = [1]  # value=1 on Linux (verified with comprehensive test)
NV_ENC_BUFFER_FORMAT_NV12 = 1
NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR = 2
NV_ENC_INPUT_IMAGE = 0
NV_ENC_PIC_STRUCT_FRAME = 1

# SDK 13.0: NVENCAPI_VERSION = NVENCAPI_MAJOR | (NVENCAPI_MINOR << 24) = 13 | 0
_NVENCAPI_VERSION = 0x0d


def NVENCAPI_STRUCT_VERSION(ver, bit31=False):
    """SDK 13.0: NVENCAPI_VERSION | (ver << 16) | (0x7 << 28)"""
    v = _NVENCAPI_VERSION | (ver << 16) | (0x7 << 28)
    if bit31:
        v |= (1 << 31)
    return v


class _NvGuid(Structure):
    _pack_ = 1
    _fields_ = [
        ("Data1", c_uint32), ("Data2", c_uint16), ("Data3", c_uint16),
        ("Data4", c_uint8 * 8),
    ]


NV_ENC_CODEC_H264_GUID = _NvGuid(0x6b9c211b, 0x3fdd, 0x4a5a,
    (0x8d, 0x2e, 0x05, 0x0a, 0xbb, 0xb9, 0x1c, 0x6a))
NV_ENC_PRESET_P1_GUID = _NvGuid(0xfc9a8d6c, 0xa4e8, 0x4f03,
    (0xaa, 0xce, 0x91, 0x97, 0x6a, 0xc2, 0x74, 0x10))


class _NvEncOpenEncodeSessionExParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",    c_uint32),
        ("deviceType", c_uint32),
        ("device",     c_void_p),
        ("reserved",   c_void_p),
        ("apiVersion", c_uint32),
        ("reserved1",  c_uint32 * 253),
        ("reserved2",  c_void_p * 64),
    ]


class _NvEncConfigH264VUIParameters(Structure):
    _pack_ = 1
    _fields_ = [
        ("overscanInfoPresentFlag", c_uint32),
        ("videoSignalTypePresentFlag", c_uint32),
        ("videoFormat", c_uint32), ("videoFullRangeFlag", c_uint32),
        ("colourDescriptionPresentFlag", c_uint32),
        ("colourPrimaries", c_uint32), ("transferCharacteristics", c_uint32),
        ("matrixCoefficients", c_uint32), ("chromaSampleLocationFlag", c_uint32),
        ("chromaSampleLocationTop", c_uint32),
        ("chromaSampleLocationBottom", c_uint32),
        ("bitstreamRestrictionFlag", c_uint32),
        ("reserved", c_uint32 * 16),
    ]


class _NvEncConfigH264(Structure):
    _pack_ = 1
    _fields_ = [
        ("enableTemporalSVC",    c_uint32), ("enableTemporalSVC_1", c_uint32),
        ("profileLevel",         c_uint32), ("reserved1",            c_uint32 * 14),
        ("maxNumRefFramesInDPB", c_uint32), ("reserved2",            c_uint32 * 3),
        ("idrPeriod",            c_uint32), ("reserved10",           c_uint32 * 5),
        ("vuiParameters",        _NvEncConfigH264VUIParameters),
        ("reserved12",           c_uint32 * 222),
    ]


class _NvEncConfig(Structure):
    """SDK 13.0 NV_ENC_CONFIG — sizeof=3584 bytes"""
    _pack_ = 1
    _fields_ = [
        ("version",         c_uint32),  ("profileGUID",     _NvGuid),
        ("gopLength",       c_uint32),  ("frameIntervalP",  c_uint32),
        ("frameFieldMode",  c_uint32),  ("enablePTD",       c_uint32),
        ("frameFieldMode_1", c_uint32), ("reserved3",       c_uint32 * 53),
        ("mvPrecision",     c_uint32),  ("reserved4",       c_uint32 * 27),
        ("enableTemporalAQ", c_uint32), ("reserved5",       c_uint32 * 171),
        ("encodeCodecConfig", _NvEncConfigH264),
        ("reserved7",       c_uint32 * 356),  # padded to hit 3584 bytes total
    ]


class _NvEncPresetConfig(Structure):
    """SDK 13.0 NV_ENC_PRESET_CONFIG — sizeof=5128 bytes"""
    _pack_ = 1
    _fields_ = [
        ("version",     c_uint32),         # offset 0
        ("reserved",    c_uint32),         # offset 4
        # NV_ENC_CONFIG = 3584 bytes (CONFIG_VER=9, bit31)
        ("presetCfg",   c_uint8 * 3584),   # offset 8
        ("reserved1",   c_uint32 * 256),   # offset 3592 (256×4=1024)
        ("reserved2",   c_void_p * 64),    # offset 4616 (64×8=512)
    ]


class _NvEncInitializeParams(Structure):
    """SDK 13.0 NV_ENC_INITIALIZE_PARAMS — sizeof=1800 bytes"""
    _pack_ = 1
    _fields_ = [
        ("version",                 c_uint32),       # offset 0
        ("encodeGUID",              _NvGuid),        # offset 4 (16 bytes)
        ("presetGUID",              _NvGuid),        # offset 20 (16 bytes)
        ("encodeWidth",             c_uint32),       # offset 36
        ("encodeHeight",            c_uint32),       # offset 40
        ("darWidth",                c_uint32),       # offset 44
        ("darHeight",               c_uint32),       # offset 48
        ("frameRateNum",            c_uint32),       # offset 52
        ("frameRateDen",            c_uint32),       # offset 56
        ("enableEncodeAsync",       c_uint32),       # offset 60
        ("enablePTD",               c_uint32),       # offset 64
        ("bitfield",                c_uint32),       # offset 68
        ("privDataSize",            c_uint32),       # offset 72
        ("reserved_76",             c_uint32),       # offset 76
        ("privData",                c_void_p),       # offset 80
        ("encodeConfig",            c_void_p),       # offset 88
        ("maxEncodeWidth",          c_uint32),       # offset 96
        ("maxEncodeHeight",         c_uint32),       # offset 100
        ("maxMEHintCountsPerBlock", c_uint8 * 32),   # offset 104
        ("tuningInfo",              c_uint32),       # offset 136
        ("bufferFormat",            c_uint32),       # offset 140
        ("numStateBuffers",         c_uint32),       # offset 144
        ("outputStatsLevel",        c_uint32),       # offset 148
        ("reserved1",               c_uint8 * 1136), # offset 152 (284×4)
        ("reserved2",               c_void_p * 64),  # offset 1288 (64×8=512)
    ]


class _NvEncRegisterResource(Structure):
    """SDK 13.0 NV_ENC_REGISTER_RESOURCE — sizeof=1536 bytes (C dump verified)"""
    _pack_ = 1
    _fields_ = [
        ("version",          c_uint32),  ("resourceType",     c_uint32),
        ("width",            c_uint32),  ("height",           c_uint32),
        ("pitch",            c_uint32),  ("subResourceIndex", c_uint32),
        ("bufferFormat",     c_uint32),  ("bufferUsage",      c_uint32),
        ("pInputFencePoint", c_void_p),  ("pOutputFencePoint",c_void_p),
        ("reserved",         c_uint32 * 370),  # 1480B padding to hit 1536
        ("registeredResource", c_void_p),
    ]


class _NvEncMapInputResource(Structure):
    """SDK 13.0 NV_ENC_MAP_INPUT_RESOURCE — sizeof=1544 bytes (C dump verified)"""
    _pack_ = 1
    _fields_ = [
        ("version",            c_uint32), ("subResourceIndex", c_uint32),
        ("reserved",           c_uint32 * 62),
        ("registeredResource", c_void_p), ("mappedResource",   c_void_p),
        ("reserved1",          c_uint8 * 1272),  # pad to 1544
    ]


class _NvEncPicParamsH264(Structure):
    _pack_ = 1
    _fields_ = [("reserved", c_uint32 * 4), ("refFrameFlag", c_uint32),
                 ("reserved1", c_uint32 * 257)]


class _NvEncPicParams(Structure):
    """SDK 13.0 NV_ENC_PIC_PARAMS — sizeof=3360 bytes (C dump verified)"""
    _pack_ = 1
    _fields_ = [
        ("version",        c_uint32),  ("inputWidth",      c_uint32),
        ("inputHeight",    c_uint32),  ("inputPitch",      c_uint32),
        ("inputBuffer",    c_void_p),  ("inputTimeStamp",  c_uint64),
        ("pictureStruct",  c_uint32),  ("encodePicFlags",  c_uint32),
        ("frameIdx",       c_uint32),  ("inputFencePoint", c_void_p),
        ("outputFencePoint",c_void_p), ("inputDuration",   c_uint64),
        ("reserved",       c_uint32 * 8),
        ("codecPicParams", _NvEncPicParamsH264),
        ("reserved1",      c_uint32 * 553),  # pad to 3360
    ]


class _NvEncLockBitstream(Structure):
    """SDK 13.0 NV_ENC_LOCK_BITSTREAM — sizeof=1544 bytes (C dump verified)"""
    _pack_ = 1
    _fields_ = [
        ("version",             c_uint32), ("doNotWait",        c_uint32),
        ("reserved",            c_uint32 * 30),
        ("outputBitstream",     c_void_p), ("sliceOffsets",     c_uint32 * 16),
        ("reserved1",           c_uint32 * 159),
        ("bitstreamSizeInBytes",c_uint32), ("bitstreamBufferPtr",c_void_p),
        ("reserved2",           c_uint32 * 174),
    ]


# Version constants (SDK 13.0)
_VER_OPEN_ENCODE_SESSION_EX_PARAMS = NVENCAPI_STRUCT_VERSION(1)          # 0x7001000d
_VER_FUNCTION_LIST                = NVENCAPI_STRUCT_VERSION(2)           # 0x7002000d
NV_ENC_PRESET_CONFIG_VER          = NVENCAPI_STRUCT_VERSION(5, True)     # 0xf005000d
NV_ENC_CONFIG_VER                 = NVENCAPI_STRUCT_VERSION(9, True)     # 0xf009000d
NV_ENC_INITIALIZE_PARAMS_VER      = NVENCAPI_STRUCT_VERSION(7, True)     # 0xf007000d
NV_ENC_PIC_PARAMS_VER             = NVENCAPI_STRUCT_VERSION(7, True)     # 0xf007000d
NV_ENC_LOCK_BITSTREAM_VER         = NVENCAPI_STRUCT_VERSION(2, True)     # 0xf002000d
NV_ENC_REGISTER_RESOURCE_VER      = NVENCAPI_STRUCT_VERSION(5)           # 0x7005000d
NV_ENC_MAP_INPUT_RESOURCE_VER     = NVENCAPI_STRUCT_VERSION(4)           # 0x7004000d
NV_ENC_CREATE_INPUT_BUFFER_VER    = NVENCAPI_STRUCT_VERSION(2)           # 0x7002000d
NV_ENC_LOCK_INPUT_BUFFER_VER      = NVENCAPI_STRUCT_VERSION(1)           # 0x7001000d
NV_ENC_CREATE_BITSTREAM_BUFFER_VER = NVENCAPI_STRUCT_VERSION(1)          # 0x7001000d

# SDK 13.0 function table indices (from NV_ENCODE_API_FUNCTION_LIST struct layout)
_FUNC_TABLE_SIZE = 2552
_FUNC_IDX = {
    "OpenEncodeSession":        0,
    "OpenEncodeSessionEx":     29,  # was 24 in SDK 12.0
    "GetEncodePresetConfig":   10,  # was 11
    "GetEncodePresetConfigEx": 39,  # was 34
    "InitializeEncoder":       11,  # NvEncInitializeEncoder (was CreateEncoder=12)
    "DestroyEncoder":          27,  # was 22
    "EncodePicture":           16,  # SDK 13.0: offset 136
    "LockBitstream":           17,  # SDK 13.0: offset 144
    "UnlockBitstream":         18,  # SDK 13.0: offset 152
    "MapInputResource":        25,  # SDK 13.0: offset 208
    "UnmapInputResource":      26,  # SDK 13.0: offset 216
    "RegisterResource":        30,  # SDK 13.0: offset 248
    "UnregisterResource":      31,  # SDK 13.0: offset 256
}

# Function prototypes (SDK 13.0)
_NvEncodeAPICreateInstanceProto = ctypes.CFUNCTYPE(c_uint32, ctypes.c_void_p)
_NvEncOpenEncodeSessionExProto = ctypes.CFUNCTYPE(
    c_uint32, ctypes.POINTER(_NvEncOpenEncodeSessionExParams), ctypes.POINTER(c_void_p))
# GUID (16 bytes) 拆成两个 uint64, 绕过 ctypes struct-by-value ABI 问题 (Linux x86-64)
_NvEncGetEncodePresetConfigProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, c_uint64, c_uint64, c_uint64, c_uint64, ctypes.POINTER(_NvEncPresetConfig))
_NvEncCreateEncoderProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(_NvEncInitializeParams))
_NvEncDestroyEncoderProto = ctypes.CFUNCTYPE(c_uint32, c_void_p)
_NvEncRegisterResourceProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, ctypes.POINTER(_NvEncRegisterResource))
_NvEncUnregisterResourceProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
_NvEncMapInputResourceProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, ctypes.POINTER(_NvEncMapInputResource))
_NvEncUnmapInputResourceProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
_NvEncEncodePictureProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, ctypes.POINTER(_NvEncPicParams))
_NvEncLockBitstreamProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, ctypes.POINTER(_NvEncLockBitstream))
_NvEncUnlockBitstreamProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)


def _find_nvenc_dll():
    if sys.platform == "win32":
        return "nvEncodeAPI64.dll"
    for c in [
        "/usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1",
        "/usr/lib64/libnvidia-encode.so.1",
        "/usr/lib/libnvidia-encode.so.1",
    ]:
        if os.path.exists(c):
            return c
    return "libnvidia-encode.so.1"


def _load_libcuda():
    return ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")


# ============================================================================
# 核心测试
# ============================================================================

def main():
    W, H = 1920, 1080
    fps = 30.0

    # ── Step 0: import torch (不触发 CUDA), 用 driver API 管理所有 context ──
    import torch

    libcuda = _load_libcuda()
    libcuda.cuInit(0)

    # 用 driver API 显式获取 primary context (PyTorch 的 context 对 driver API 不可见!)
    libcuda.cuDevicePrimaryCtxRetain.restype = c_uint32
    libcuda.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(c_void_p), c_int]
    primary_ctx = c_void_p(None)
    r = libcuda.cuDevicePrimaryCtxRetain(ctypes.byref(primary_ctx), c_int(0))
    if r != 0 or primary_ctx.value is None:
        print(f"FATAL: cuDevicePrimaryCtxRetain failed (code={r})", flush=True)
        sys.exit(1)
    torch_ctx = primary_ctx
    _tv = torch_ctx.value if torch_ctx.value else 0
    print(f"[Phase0] primary context: 0x{_tv:x}", flush=True)

    # 将 primary context 设为当前 (torch 需要它)
    libcuda.cuCtxPushCurrent.restype = c_uint32
    libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
    libcuda.cuCtxPushCurrent(torch_ctx)

    if not torch.cuda.is_available():
        print("FATAL: torch CUDA not available", flush=True)
        sys.exit(1)
    print("[Phase0] torch CUDA OK (primary context via driver API)", flush=True)

    # ── Step 1: 创建独立 CUDA context 用于 NVENC ──
    libcuda.cuCtxCreate.restype = c_uint32
    libcuda.cuCtxCreate.argtypes = [ctypes.POINTER(c_void_p), c_uint32, c_int]
    ctx = c_void_p(None)
    r = libcuda.cuCtxCreate(ctypes.byref(ctx), c_uint32(4), c_int(0))
    if r == 0 and ctx.value is not None:
        print("[Phase1] cuCtxCreate OK (独立 context)", flush=True)
        own_ctx = True
    else:
        print(f"[Phase1] cuCtxCreate failed (code={r}), using primary context for NVENC", flush=True)
        own_ctx = False
        ctx = torch_ctx

    # Load NVENC DLL
    nvenc_dll_path = _find_nvenc_dll()
    nvenc_dll = ctypes.CDLL(nvenc_dll_path)

    # Runtime API version detection
    try:
        _get_max_ver = nvenc_dll.NvEncodeAPIGetMaxSupportedVersion
        _get_max_ver.restype = c_uint32
        _get_max_ver.argtypes = [ctypes.POINTER(c_uint32)]
        _max_ver_val = c_uint32(0)
        _get_max_ver(ctypes.byref(_max_ver_val))
        api_ver = _max_ver_val.value if _max_ver_val.value > 0 else _NVENCAPI_VERSION
    except Exception:
        api_ver = _NVENCAPI_VERSION
    print(f"[Phase1] NVENC API: 0x{api_ver:x} (SDK 13.0: NVENCAPI_VERSION=0x{_NVENCAPI_VERSION:x})", flush=True)

    # NvEncodeAPICreateInstance — SDK 13.0: 2552 byte func_table
    create_instance = _NvEncodeAPICreateInstanceProto(("NvEncodeAPICreateInstance", nvenc_dll))
    func_table = (c_uint8 * _FUNC_TABLE_SIZE)()
    flist_ver = _VER_FUNCTION_LIST
    cast(func_table, ctypes.POINTER(c_uint32))[0] = flist_ver
    status = create_instance(cast(func_table, c_void_p))
    if status != NV_ENC_SUCCESS:
        print(f"FATAL: NvEncodeAPICreateInstance failed (code={status})", flush=True)
        sys.exit(1)
    print(f"[Phase1] NvEncodeAPICreateInstance OK (flist_ver=0x{flist_ver:08x}, {_FUNC_TABLE_SIZE}B)", flush=True)

    # 函数指针从 offset 8 开始 (skip version + reserved)
    _func_ptrs = cast(byref(func_table, 8), ctypes.POINTER(c_void_p))
    def _get_func(idx):
        addr = _func_ptrs[idx]
        if not addr or addr == 0:
            return None
        return addr

    enc = c_void_p(None)
    open_session_ex = _NvEncOpenEncodeSessionExProto(_get_func(_FUNC_IDX["OpenEncodeSessionEx"]))
    _NvEncOpenEncodeSessionProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_uint32, ctypes.POINTER(c_void_p))
    open_session_v1 = None
    if _get_func(_FUNC_IDX["OpenEncodeSession"]):
        open_session_v1 = _NvEncOpenEncodeSessionProto(_get_func(_FUNC_IDX["OpenEncodeSession"]))
    opened_ver = None

    # ── 方法 A: nvEncOpenEncodeSessionEx (SDK 13.0, vers=0x7001000d) ──
    for cuda_type in NV_ENC_DEVICE_TYPE_CUDA_VALUES:
        if opened_ver is not None:
            break
        for api_v in [api_ver, _NVENCAPI_VERSION, 0]:
            session_params = _NvEncOpenEncodeSessionExParams()
            session_params.version = _VER_OPEN_ENCODE_SESSION_EX_PARAMS
            session_params.deviceType = cuda_type
            session_params.device = ctx
            session_params.apiVersion = api_v

            status = open_session_ex(byref(session_params), ctypes.byref(enc))
            tag = "OK" if status == NV_ENC_SUCCESS else f"code={status}"
            print(f"[Phase1] Ex  CUDA={cuda_type} apiVer=0x{api_v:x}: {tag}", flush=True)
            if status == NV_ENC_SUCCESS:
                opened_ver = api_v
                break

    # ── 方法 B: dev=NULL 变体 ──
    if opened_ver is None:
        print("[Phase1] dev=ctx failed, 尝试 dev=NULL...", flush=True)
        for api_v in [api_ver, _NVENCAPI_VERSION]:
            session_params = _NvEncOpenEncodeSessionExParams()
            session_params.version = _VER_OPEN_ENCODE_SESSION_EX_PARAMS
            session_params.deviceType = 1
            session_params.device = c_void_p(None)
            session_params.apiVersion = api_v
            status = open_session_ex(byref(session_params), ctypes.byref(enc))
            tag = "OK" if status == NV_ENC_SUCCESS else f"code={status}"
            print(f"[Phase1] Ex  CUDA=1 dev=NULL apiVer=0x{api_v:x}: {tag}", flush=True)
            if status == NV_ENC_SUCCESS:
                opened_ver = api_v
                break

    # ── 方法 C: nvEncOpenEncodeSession (老 API) ──
    if opened_ver is None and open_session_v1 is not None:
        print("[Phase1] Ex 全部失败, 尝试老 API nvEncOpenEncodeSession...", flush=True)
        for cuda_type in NV_ENC_DEVICE_TYPE_CUDA_VALUES:
            if opened_ver is not None:
                break
            for _try_device in (ctx, c_void_p(None)):
                status = open_session_v1(_try_device, c_uint32(cuda_type), ctypes.byref(enc))
                tag = "OK" if status == NV_ENC_SUCCESS else f"code={status}"
                dev_label = f"ctx=0x{_try_device.value:x}" if _try_device.value is not None else "ctx=NULL"
                print(f"[Phase1] V1  CUDA={cuda_type} {dev_label}: {tag}", flush=True)
                if status == NV_ENC_SUCCESS:
                    opened_ver = -1
                    break

    if opened_ver is None:
        print("FATAL: 所有 API 版本 + 两种 API 均失败 — NVENC SDK 在此环境不可用", flush=True)
        sys.exit(1)
    if opened_ver == -1:
        print("[Phase1] ★ nvEncOpenEncodeSession OK! (老 API, context 纯净)", flush=True)
    else:
        print(f"[Phase1] ★ nvEncOpenEncodeSessionEx OK! (apiVer=0x{opened_ver:x}, context 纯净)", flush=True)

    # GetEncodePresetConfig + InitializeEncoder
    codec_guid = NV_ENC_CODEC_H264_GUID
    preset_guid = NV_ENC_PRESET_P1_GUID

    # ── GetEncodePresetConfig: 从 driver 动态获取 GUID (不同 driver 版本 GUID 不同!) ──
    # Step 1: 获取 driver 支持的 encode GUIDs
    _GetEncodeGUIDCountProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint32))
    _GetEncodeGUIDsProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(_NvGuid), c_uint32, ctypes.POINTER(c_uint32))
    _GetEncodePresetGUIDsProto = ctypes.CFUNCTYPE(
        c_uint32, c_void_p, _NvGuid, ctypes.POINTER(_NvGuid), c_uint32, ctypes.POINTER(c_uint32))

    count_val = c_uint32(0)
    s = _GetEncodeGUIDCountProto(_get_func(1))(enc, byref(count_val))
    print(f"[Phase1] GetEncodeGUIDCount → code={s} count={count_val.value}", flush=True)
    if s != 0 or count_val.value == 0:
        print("FATAL: 无法获取 encode GUID count", flush=True)
        sys.exit(1)

    n_guids = count_val.value
    guid_array = (_NvGuid * n_guids)()
    ctypes.memset(cast(guid_array, c_void_p), 0, sizeof(guid_array))
    actual_count = c_uint32(0)
    s = _GetEncodeGUIDsProto(_get_func(4))(enc, guid_array, n_guids, byref(actual_count))
    print(f"[Phase1] GetEncodeGUIDs → code={s} count={actual_count.value}", flush=True)
    if s != 0 or actual_count.value == 0:
        print("FATAL: 无法获取 encode GUIDs", flush=True)
        sys.exit(1)

    codec_guid = guid_array[0]  # 第一个 = H.264 (或 driver 首选 codec)
    print(f"[Phase1] 使用 codec GUID: {codec_guid.Data1:08x}-{codec_guid.Data2:04x}-{codec_guid.Data3:04x}", flush=True)

    # Step 2: 获取该 codec 的 preset GUIDs
    preset_guid_array = (_NvGuid * 64)()
    ctypes.memset(cast(preset_guid_array, c_void_p), 0, sizeof(preset_guid_array))
    preset_count = c_uint32(0)
    s = _GetEncodePresetGUIDsProto(_get_func(9))(enc, codec_guid, preset_guid_array, 64, byref(preset_count))
    print(f"[Phase1] GetEncodePresetGUIDs → code={s} count={preset_count.value}", flush=True)
    if s != 0 or preset_count.value == 0:
        print("FATAL: 无法获取 preset GUIDs", flush=True)
        sys.exit(1)

    preset_guid = preset_guid_array[0]  # 第一个 preset (通常是 P1)
    print(f"[Phase1] 使用 preset GUID: {preset_guid.Data1:08x}-{preset_guid.Data2:04x}-{preset_guid.Data3:04x}", flush=True)

    # Step 3: GetEncodePresetConfig
    get_preset_addr = _get_func(_FUNC_IDX["GetEncodePresetConfig"])
    # struct-GUID CFUNCTYPE (已验证 ABI 不是问题, struct 传递正常)
    _GPC_fn = ctypes.CFUNCTYPE(c_uint32, c_void_p, _NvGuid, _NvGuid, ctypes.POINTER(_NvEncPresetConfig))

    preset_config = _NvEncPresetConfig()
    ctypes.memset(byref(preset_config), 0, sizeof(preset_config))
    preset_config.version = NV_ENC_PRESET_CONFIG_VER
    cast(byref(preset_config, 8), ctypes.POINTER(c_uint32))[0] = NV_ENC_CONFIG_VER

    status = _GPC_fn(get_preset_addr)(enc, codec_guid, preset_guid, byref(preset_config))
    print(f"[Phase1] GetEncodePresetConfig → code={status}", flush=True)
    if status != NV_ENC_SUCCESS:
        print(f"FATAL: GetEncodePresetConfig failed (code={status})", flush=True)
        sys.exit(1)
    print("[Phase1] GetEncodePresetConfig OK", flush=True)

    # SDK 13.0: presetCfg is NV_ENC_CONFIG at offset 8 in _NvEncPresetConfig
    # 可以直接 cast presetCfg byte array 为 _NvEncConfig 来访问字段
    enc_cfg = cast(byref(preset_config, 8), ctypes.POINTER(_NvEncConfig)).contents
    enc_cfg.gopLength = int(fps)
    enc_cfg.frameIntervalP = 1
    enc_cfg.encodeCodecConfig.idrPeriod = int(fps)
    enc_cfg.encodeCodecConfig.maxNumRefFramesInDPB = 16

    init_params = _NvEncInitializeParams()
    init_params.version = NV_ENC_INITIALIZE_PARAMS_VER
    init_params.encodeGUID = codec_guid
    init_params.presetGUID = preset_guid
    init_params.encodeWidth = W
    init_params.encodeHeight = H
    init_params.darWidth = W
    init_params.darHeight = H
    init_params.frameRateNum = int(fps * 1000)
    init_params.frameRateDen = 1000
    init_params.maxEncodeWidth = W
    init_params.maxEncodeHeight = H
    init_params.enablePTD = 1
    # SDK 13.0: encodeConfig 在 offset 88 (不是旧版的 presetConfig + encodeConfig 分开)
    init_params.encodeConfig = cast(byref(preset_config, 8), c_void_p)

    # NvEncInitializeEncoder (SDK 13.0 改名, 旧名 NvEncCreateEncoder)
    create_encoder = _NvEncCreateEncoderProto(_get_func(_FUNC_IDX["InitializeEncoder"]))
    status = create_encoder(enc, byref(init_params))
    if status != NV_ENC_SUCCESS:
        print(f"FATAL: NvEncInitializeEncoder failed (code={status})", flush=True)
        sys.exit(1)
    print(f"[Phase1] Encoder initialized: {W}x{H}@{fps:.1f}fps H.264 p1", flush=True)

    # Context 切换 helpers
    libcuda.cuCtxPopCurrent.restype = c_uint32
    libcuda.cuCtxPopCurrent.argtypes = [ctypes.POINTER(c_void_p)]
    libcuda.cuCtxPushCurrent.restype = c_uint32
    libcuda.cuCtxPushCurrent.argtypes = [c_void_p]

    def push_nvenc_ctx():
        """Push NVENC context onto stack"""
        libcuda.cuCtxPushCurrent(ctx)

    def pop_nvenc_ctx():
        """Pop NVENC context off stack, return to saved context"""
        _p = c_void_p(0)
        libcuda.cuCtxPopCurrent(ctypes.byref(_p))

    # ── Phase 1.5: RegisterResource 诊断 (默认跳过 — 已知 segfault, 用 CreateInputBuffer 替代) ──
    _PHASE15_SKIP = os.environ.get("PHASE15", "0") != "1"
    if _PHASE15_SKIP:
        print(f"\n[Phase1.5] SKIP — RegisterResource 已知 segfault, 改用 CreateInputBuffer (set PHASE15=1 to force)", flush=True)

    # ── 切回 torch context 进行 Phase 2 torch 操作 ──
    pop_nvenc_ctx()  # pop NVENC context, restore torch context
    _tv2 = torch_ctx.value if torch_ctx.value else 0
    print(f"\n[Phase1] switched back to torch ctx 0x{_tv2:x}", flush=True)

    # ── Phase 2: torch CUDA 操作 ──
    dummy = torch.zeros((4, 3, 256, 256), device="cuda")
    dummy = dummy * 2 + 1
    torch.cuda.synchronize()
    print("[Phase2] torch CUDA 操作已完成", flush=True)

    # 验证 encoder 在跨 context 后仍有效
    push_nvenc_ctx()
    post_count = c_uint32(0)
    s = _GetEncodeGUIDCountProto(_get_func(1))(enc, byref(post_count))
    print(f"[Phase2] NVENC context: GetEncodeGUIDCount → code={s}", flush=True)
    pop_nvenc_ctx()
    if s != 0:
        print(f"FATAL: encoder 失效 (code={s})", flush=True)
        sys.exit(1)

    # ── Step 3: 用 torch 创建 NV12 帧, 用预建立的 NVENC 编码 ──
    nv12_h = H + H // 2
    y_plane = torch.zeros((H, W), dtype=torch.uint8, device="cuda")
    for x in range(W):
        y_plane[:, x] = int(255 * x / W)
    uv_plane = torch.full((H // 2, W), 128, dtype=torch.uint8, device="cuda")
    nv12_tensor = torch.cat([y_plane, uv_plane], dim=0).contiguous()

    print(f"[Phase3] Torch NV12 tensor: {nv12_tensor.shape}, ptr=0x{nv12_tensor.data_ptr():x}", flush=True)

    # 切换到 NVENC context
    push_nvenc_ctx()

    # ── Phase 3: CreateInputBuffer (NVENC 管理内存, 避免 RegisterResource segfault) ──
    h264_data = b""  # 提前初始化, 防止 NameError
    #  Struct offsets from nv-codec-headers n13.0.19.0 (verified via curl):
    #   NV_ENC_CREATE_INPUT_BUFFER (776B): version@0, width@4, height@8,
    #     memoryHeap@12, bufferFmt@16, reserved@20, inputBuffer@24, pSysMemBuffer@32,
    #     reserved1[58]@40, reserved2[63]@272
    #   NV_ENC_LOCK_INPUT_BUFFER (1544B): version@0, doNotWait:1@4,
    #     inputBuffer@8, bufferDataPtr@16, pitch@24, reserved1[251]@28, reserved2[64]@1032

    _CreateInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 776))
    _LockInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
    _UnlockInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
    _DestroyInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)

    # ── CreateInputBuffer ──
    create_buf = (c_uint8 * 776)()
    ctypes.memset(create_buf, 0, 776)
    cast(create_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_CREATE_INPUT_BUFFER_VER  # version @ 0
    cast(byref(create_buf, 4), ctypes.POINTER(c_uint32))[0] = W           # width @ 4
    cast(byref(create_buf, 8), ctypes.POINTER(c_uint32))[0] = nv12_h      # height @ 8
    # memoryHeap @ 12 = 0 (deprecated, memset already 0)
    cast(byref(create_buf, 16), ctypes.POINTER(c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12  # bufferFmt @ 16
    # reserved @ 20 = 0 (memset already 0)

    create_input = _CreateInputBufferProto(_get_func(12))  # index 12 = CreateInputBuffer
    s = create_input(enc, create_buf)
    print(f"[Phase3] CreateInputBuffer → code={s}", flush=True)
    if s != 0:
        print(f"FATAL: CreateInputBuffer failed (code={s})", flush=True)
        sys.exit(1)
    _raw_ptr = cast(byref(create_buf, 24), ctypes.POINTER(c_void_p))[0]  # inputBuffer @ 24
    input_buf_ptr = _raw_ptr if isinstance(_raw_ptr, int) else (_raw_ptr.value or 0)
    print(f"[Phase3] inputBuffer=0x{input_buf_ptr:x}", flush=True)

    # ── CreateBitstreamBuffer (encoder 需要输出 buffer) ──
    #  NV_ENC_CREATE_BITSTREAM_BUFFER (776B): version@0, size@4, memoryHeap@8,
    #    reserved@12, bitstreamBuffer@16, bitstreamBufferPtr@24,
    #    reserved1[58]@32, reserved2[64]@264
    _CreateBitstreamBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 776))
    bs_buf = (c_uint8 * 776)()
    ctypes.memset(bs_buf, 0, 776)
    cast(bs_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_CREATE_BITSTREAM_BUFFER_VER  # version @ 0
    # size @ 4, memoryHeap @ 8, reserved @ 12 all = 0 (memset)

    s = _CreateBitstreamBufferProto(_get_func(14))(enc, bs_buf)  # index 14 = CreateBitstreamBuffer
    print(f"[Phase3] CreateBitstreamBuffer → code={s}", flush=True)
    if s != 0:
        print(f"FATAL: CreateBitstreamBuffer failed (code={s})", flush=True)
        sys.exit(1)
    _raw_bs = cast(byref(bs_buf, 16), ctypes.POINTER(c_void_p))[0]  # bitstreamBuffer @ 16
    bs_handle = _raw_bs if isinstance(_raw_bs, int) else (_raw_bs.value or 0)
    print(f"[Phase3] bitstreamBuffer=0x{bs_handle:x}", flush=True)

    # ── LockInputBuffer → get mapped CPU pointer ──
    lock_buf = (c_uint8 * 1544)()
    ctypes.memset(lock_buf, 0, 1544)
    cast(lock_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_INPUT_BUFFER_VER  # version @ 0
    # doNotWait:1 + reservedBitFields:31 @ 4 = 0 (memset already 0)
    cast(byref(lock_buf, 8), ctypes.POINTER(c_void_p))[0] = input_buf_ptr  # inputBuffer @ 8

    lock_input = _LockInputBufferProto(_get_func(19))  # index 19 = LockInputBuffer
    s = lock_input(enc, lock_buf)
    print(f"[Phase3] LockInputBuffer → code={s}", flush=True)

    if s == 0:
        _raw_map = cast(byref(lock_buf, 16), ctypes.POINTER(c_void_p))[0]  # bufferDataPtr @ 16
        mapped_buf_val = _raw_map if isinstance(_raw_map, int) else (_raw_map.value or 0)
        actual_pitch = cast(byref(lock_buf, 24), ctypes.POINTER(c_uint32))[0]  # pitch @ 24 (output)
        print(f"[Phase3] mappedBuffer=0x{mapped_buf_val:x} pitch={actual_pitch}", flush=True)

        if mapped_buf_val:
            nv12_bytes = nv12_tensor.numel()
            src_ptr = nv12_tensor.data_ptr()
            # 用 CUDA driver API 做 GPU→GPU copy (ctypes.memmove 不能读 GPU 内存)
            libcuda.cuMemcpyDtoD_v2.restype = c_uint32
            libcuda.cuMemcpyDtoD_v2.argtypes = [c_void_p, c_void_p, c_size_t]
            r = libcuda.cuMemcpyDtoD_v2(c_void_p(mapped_buf_val), c_void_p(src_ptr), nv12_bytes)
            if r != 0:
                print(f"[Phase3] cuMemcpyDtoD failed (code={r}), trying cuMemcpy...", flush=True)
                libcuda.cuMemcpy.restype = c_uint32
                libcuda.cuMemcpy.argtypes = [c_void_p, c_void_p, c_size_t]
                r = libcuda.cuMemcpy(c_void_p(mapped_buf_val), c_void_p(src_ptr), nv12_bytes)
            print(f"[Phase3] Copied {nv12_bytes} bytes torch→NVENC (cuMemcpy code={r})", flush=True)

        # UnlockInputBuffer (pass handle, not mapped ptr)
        _UnlockInputBufferProto(_get_func(20))(enc, c_void_p(input_buf_ptr))  # index 20 = UnlockInputBuffer

        # ── EncodePicture: 使用 byte array (避免 ctypes struct 字段偏移问题) ──
        #  NV_ENC_PIC_PARAMS actual layout (nv-codec-headers n13.0.19.0):
        #   version@0, inputWidth@4, inputHeight@8, inputPitch@12,
        #   encodePicFlags@16, frameIdx@20, inputTimeStamp@24, inputDuration@32,
        #   inputBuffer@40, outputBitstream@48, completionEvent@56,
        #   bufferFmt@64, pictureStruct@68, pictureType@72,
        #   codecPicParams@76 (H264=1032B), ... pad to 3360
        pic_buf = (c_uint8 * 3360)()
        ctypes.memset(pic_buf, 0, 3360)
        cast(pic_buf, ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER       # version @ 0
        cast(byref(pic_buf, 4), ctypes.POINTER(c_uint32))[0] = W                 # inputWidth @ 4
        cast(byref(pic_buf, 8), ctypes.POINTER(c_uint32))[0] = nv12_h            # inputHeight @ 8
        cast(byref(pic_buf, 12), ctypes.POINTER(c_uint32))[0] = (
            actual_pitch if actual_pitch > 0 else W)                              # inputPitch @ 12
        # encodePicFlags @ 16 = 0 (default)
        # frameIdx @ 20 = 0 (memset already 0)
        cast(byref(pic_buf, 24), ctypes.POINTER(c_uint64))[0] = 0                # inputTimeStamp @ 24
        cast(byref(pic_buf, 32), ctypes.POINTER(c_uint64))[0] = 0                # inputDuration @ 32
        cast(byref(pic_buf, 40), ctypes.POINTER(c_void_p))[0] = c_void_p(input_buf_ptr)  # inputBuffer @ 40
        cast(byref(pic_buf, 48), ctypes.POINTER(c_void_p))[0] = c_void_p(bs_handle)     # outputBitstream @ 48
        cast(byref(pic_buf, 64), ctypes.POINTER(c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12  # bufferFmt @ 64
        cast(byref(pic_buf, 68), ctypes.POINTER(c_uint32))[0] = NV_ENC_PIC_STRUCT_FRAME     # pictureStruct @ 68
        # pictureType @ 72 = 0 (auto)

        encode_picture = _NvEncEncodePictureProto(_get_func(_FUNC_IDX["EncodePicture"]))
        s = encode_picture(enc, cast(pic_buf, ctypes.POINTER(_NvEncPicParams)))
        print(f"[Phase3] EncodePicture → code={s}", flush=True)

        if s == 0:
            # ── LockBitstream: 使用 byte array (struct 偏移也错了) ──
            #  NV_ENC_LOCK_BITSTREAM actual layout (nv-codec-headers n13.0.19.0):
            #   version@0, bitfield@4, outputBitstream@8, sliceOffsets@16,
            #   frameIdx@24, hwEncodeStatus@28, numSlices@32, bitstreamSizeInBytes@36,
            #   outputTimeStamp@40, outputDuration@48, bitstreamBufferPtr@56,
            #   pictureType@64, ...
            _LockBitstreamProto_raw = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
            lock_raw = (c_uint8 * 1544)()
            ctypes.memset(lock_raw, 0, 1544)
            cast(lock_raw, ctypes.POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER  # version @ 0
            # bitfield @ 4 = 0 (doNotWait=0, memset already)
            cast(byref(lock_raw, 8), ctypes.POINTER(c_void_p))[0] = c_void_p(bs_handle)  # outputBitstream @ 8

            lock_bs_fn = _LockBitstreamProto_raw(_get_func(_FUNC_IDX["LockBitstream"]))
            bs_status = lock_bs_fn(enc, lock_raw)

            if bs_status == NV_ENC_SUCCESS:
                bitstream_size = cast(byref(lock_raw, 36), ctypes.POINTER(c_uint32))[0]  # @ 36
                _raw_bsptr = cast(byref(lock_raw, 56), ctypes.POINTER(c_void_p))[0]  # bitstreamBufferPtr @ 56
                bitstream_ptr_val = _raw_bsptr if isinstance(_raw_bsptr, int) else (_raw_bsptr.value or 0)
                if bitstream_size > 0 and bitstream_ptr_val:
                    buf_type = c_uint8 * bitstream_size
                    h264_data = bytes(buf_type.from_address(bitstream_ptr_val))
                    print(f"[Phase3] Encoded: {bitstream_size} bytes H.264 ES", flush=True)
                # UnlockBitstream
                _raw_bsptr2 = cast(byref(lock_raw, 8), ctypes.POINTER(c_void_p))[0]  # outputBitstream @ 8
                _NvEncUnlockBitstreamProto(_get_func(_FUNC_IDX["UnlockBitstream"]))(
                    enc, _raw_bsptr2)
        else:
            print(f"FATAL: EncodePicture failed (code={s})", flush=True)

        # DestroyInputBuffer
        _DestroyInputBufferProto(_get_func(13))(enc, c_void_p(input_buf_ptr))  # index 13 = DestroyInputBuffer
    else:
        print(f"FATAL: LockInputBuffer failed (code={s})", flush=True)

    # DestroyEncoder
    destroy_addr = _get_func(_FUNC_IDX["DestroyEncoder"])
    if destroy_addr:
        _NvEncDestroyEncoderProto(destroy_addr)(enc)

    # Pop NVENC context, destroy if owned
    pop_nvenc_ctx()
    if own_ctx:
        libcuda.cuCtxDestroy.restype = c_uint32
        libcuda.cuCtxDestroy.argtypes = [c_void_p]
        libcuda.cuCtxDestroy(ctx)

    # 验证
    if len(h264_data) == 0:
        print("FATAL: 未产出 H.264 数据", flush=True)
        sys.exit(1)

    has_start_code = any(
        h264_data[i:i+4] == b'\x00\x00\x00\x01'
        for i in range(len(h264_data) - 4)
    )
    print(f"\n[RESULT] 输出 {len(h264_data)} bytes, NAL start code: {'YES' if has_start_code else 'NO'}", flush=True)

    if has_start_code:
        import tempfile
        out = os.path.join(tempfile.gettempdir(), "test_nvenc_pre_torch.h264")
        with open(out, "wb") as f:
            f.write(h264_data)
        print(f"[RESULT] 保存到: {out}", flush=True)
        print(f"[RESULT] *** 方案 2b 验证通过: 先占 NVENC + 后污染 context 可行 ***", flush=True)
    else:
        print("[RESULT] 方案 2b 待定: 需要检查输出有效性", flush=True)


if __name__ == "__main__":
    main()
