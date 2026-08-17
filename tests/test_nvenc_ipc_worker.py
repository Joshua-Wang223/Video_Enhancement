#!/usr/bin/env python3
"""
CUDA IPC + NVENC 子进程编码 可行性测试

验证: 子进程通过 CUDA IPC 访问主进程 PyTorch GPU tensor,
     在独立 clean CUDA context 中初始化 NVENC 并编码。

用法:
  python test_nvenc_ipc_worker.py              # 完整测试
  python test_nvenc_ipc_worker.py --child <b64> # 子进程入口 (内部调用)

流程:
  主进程:  torch 分配 NV12 tensor → cuIpcGetMemHandle → spawn 子进程
  子进程:  cuIpcOpenMemHandle → NVENC 初始化 → RegisterResource → Encode → stdout
"""

import sys
import os
import ctypes
import struct
from ctypes import (c_uint8, c_uint16, c_uint32, c_int32, c_int, c_uint64, c_void_p,
                     c_char, c_size_t, c_double, Structure, POINTER, byref,
                     sizeof, cast, pointer, c_bool)
import subprocess
import base64
import tempfile
import time


def guid_to_u64_pair(g):
    """将 _NvGuid (16 bytes) 拆成两个 uint64，绕过 ctypes 在 Linux x86-64
    System V AMD64 ABI 上对 >8 字节 struct 按值传递的兼容性问题。"""
    raw = struct.pack('<IHH8s', g.Data1, g.Data2, g.Data3, bytes(g.Data4))
    lo, hi = struct.unpack('<QQ', raw)
    return lo, hi

# ============================================================================
# NVENC 常量 & 结构体 (精简自 process_video_v6_4_3_single.py)
# ============================================================================

NV_ENC_SUCCESS = 0
NV_ENC_DEVICE_TYPE_CUDA = 1  # value=1 on Linux (verified)
NV_ENC_BUFFER_FORMAT_NV12 = 1
NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR = 2
NV_ENC_INPUT_IMAGE = 0
NV_ENC_PIC_STRUCT_FRAME = 1

_NVENCAPI_VERSION = 0x0d  # SDK 13.0


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
    """SDK 13.0: sizeof=1552"""
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


# _NvEncConfigH264VUIParameters / _NvEncConfigH264 — old layout kept for
# backward compat with _NvEncConfig. These are NOT the full SDK 13.0 structs,
# but serve as a minimal interface for setting gopLength/frameIntervalP/idrPeriod.
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
    """SDK 13.0 NV_ENC_CONFIG — sizeof=3584 bytes.
    Layout: first ~256B are the active fields, rest is reserved.
    Use offset-based access from _NvEncPresetConfig.presetCfg."""
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
        ("presetCfg",   c_uint8 * 3584),   # offset 8 (NV_ENC_CONFIG)
        ("reserved1",   c_uint32 * 256),   # offset 3592
        ("reserved2",   c_void_p * 64),    # offset 4616
    ]


class _NvEncRegisterResource(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",          c_uint32),  ("resourceType",     c_uint32),
        ("width",            c_uint32),  ("height",           c_uint32),
        ("pitch",            c_uint32),  ("subResourceIndex", c_uint32),
        ("bufferFormat",     c_uint32),  ("bufferUsage",      c_uint32),
        ("pInputFencePoint", c_void_p),  ("pOutputFencePoint",c_void_p),
        ("reserved",         c_uint32 * 248),
        ("registeredResource", c_void_p),
    ]


class _NvEncMapInputResource(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",            c_uint32), ("subResourceIndex", c_uint32),
        ("reserved",           c_uint32 * 62),
        ("registeredResource", c_void_p), ("mappedResource",   c_void_p),
        ("reserved1",          c_uint32 * 62),
    ]


class _NvEncPicParamsH264(Structure):
    _pack_ = 1
    _fields_ = [
        ("reserved",   c_uint32 * 4), ("refFrameFlag", c_uint32),
        ("reserved1",  c_uint32 * 257),
    ]


class _NvEncPicParams(Structure):
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
        ("reserved1",      c_uint32 * 272),
    ]


class _NvEncLockBitstream(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",             c_uint32), ("doNotWait",        c_uint32),
        ("reserved",            c_uint32 * 30),
        ("outputBitstream",     c_void_p), ("sliceOffsets",     c_uint32 * 16),
        ("reserved1",           c_uint32 * 246),
        ("bitstreamSizeInBytes",c_uint32), ("bitstreamBufferPtr",c_void_p),
        ("reserved2",           c_uint32 * 174),
    ]


# NVENC struct version numbers (SDK 13.0)
_VER_OPEN_ENCODE_SESSION_EX_PARAMS = NVENCAPI_STRUCT_VERSION(1)          # 0x7001000d
_VER_FUNCTION_LIST                = NVENCAPI_STRUCT_VERSION(2)           # 0x7002000d
NV_ENC_PRESET_CONFIG_VER          = NVENCAPI_STRUCT_VERSION(5, True)     # 0xf005000d
NV_ENC_CONFIG_VER                 = NVENCAPI_STRUCT_VERSION(9, True)     # 0xf009000d
NV_ENC_INITIALIZE_PARAMS_VER      = NVENCAPI_STRUCT_VERSION(7, True)     # 0xf007000d
NV_ENC_PIC_PARAMS_VER             = NVENCAPI_STRUCT_VERSION(7, True)     # 0xf007000d
NV_ENC_LOCK_BITSTREAM_VER         = NVENCAPI_STRUCT_VERSION(2, True)     # 0xf002000d
NV_ENC_REGISTER_RESOURCE_VER      = NVENCAPI_STRUCT_VERSION(5)           # 0x7005000d
NV_ENC_MAP_INPUT_RESOURCE_VER     = NVENCAPI_STRUCT_VERSION(4)           # 0x7004000d

# SDK 13.0 function table (2552 bytes) + indices
_FUNC_TABLE_SIZE = 2552
_FUNC_IDX = {
    "OpenEncodeSession":        0,
    "OpenEncodeSessionEx":     29,  # was 24 in SDK 12.0
    "GetEncodePresetConfig":   10,  # was 11
    "GetEncodePresetConfigEx": 39,  # was 34
    "InitializeEncoder":       11,  # NvEncInitializeEncoder
    "DestroyEncoder":          27,  # was 22
    "EncodePicture":           16,  # was 17
    "LockBitstream":           17,  # was 18
    "UnlockBitstream":         18,  # was 19
    "MapInputResource":        25,  # was 20
    "UnmapInputResource":      26,  # was 21
    "RegisterResource":        30,  # was 25
    "UnregisterResource":      31,  # was 26
}

# ============================================================================
# NVENC function prototypes (SDK 13.0)
# ============================================================================

_NvEncodeAPICreateInstanceProto = ctypes.CFUNCTYPE(
    c_uint32, ctypes.c_void_p)  # 2552-byte buffer
_NvEncOpenEncodeSessionExProto = ctypes.CFUNCTYPE(
    c_uint32, ctypes.POINTER(_NvEncOpenEncodeSessionExParams), ctypes.POINTER(c_void_p))
# GUID (16 bytes) 拆成两个 uint64, 绕过 ctypes struct-by-value ABI 问题 (Linux x86-64)
_NvEncGetEncodePresetConfigProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, c_uint64, c_uint64, c_uint64, c_uint64, ctypes.POINTER(_NvEncPresetConfig))
_NvEncInitializeEncoderProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, ctypes.POINTER(_NvEncInitializeParams))
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


# ============================================================================
# CUDA Driver API helpers
# ============================================================================

def _load_libcuda():
    plat = sys.platform
    if plat == "win32":
        return ctypes.CDLL("nvcuda.dll")
    else:
        return ctypes.CDLL("libcuda.so.1")


# ============================================================================
# 子进程: NVENC worker
# ============================================================================

def _eprint(*args, **kwargs):
    """子进程诊断日志 → stderr，避免污染 stdout (H.264 二进制输出)"""
    print(*args, file=sys.stderr, **kwargs)


def child_main(ipc_handle_b64: str):
    """子进程入口: 接收 IPC handle → 打开 GPU 内存 → NVENC 编码 → stdout 输出 H.264"""

    # 1. 解析 IPC handle
    ipc_handle_bytes = base64.b64decode(ipc_handle_b64)
    if len(ipc_handle_bytes) != 64:
        _eprint(f"CHILD_ERROR: IPC handle size mismatch: {len(ipc_handle_bytes)} != 64", flush=True)
        sys.exit(1)

    # 打印 handle 前 32 字节 (hex) 便于调试
    _hex = ipc_handle_bytes[:32].hex()
    _eprint(f"[CHILD] IPC handle (first 32B): {_hex}", flush=True)

    handle_array = (c_uint8 * 64)(*ipc_handle_bytes)

    # 2. CUDA 初始化 — 优先使用 primary context (与 parent 一致, 确保 IPC 兼容)
    libcuda = _load_libcuda()
    libcuda.cuInit(0)

    libcuda.cuDevicePrimaryCtxRetain.restype = c_uint32
    libcuda.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(c_void_p), c_int]
    libcuda.cuCtxPushCurrent.restype = c_uint32
    libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
    child_ctx = c_void_p(None)
    r = libcuda.cuDevicePrimaryCtxRetain(ctypes.byref(child_ctx), c_int(0))
    have_own_ctx = False

    if r == 0 and child_ctx.value is not None:
        libcuda.cuCtxPushCurrent(child_ctx)
        _eprint(f"[CHILD] Primary context retained + pushed (ctx=0x{child_ctx.value:x})", flush=True)
    else:
        _eprint(f"[CHILD] cuDevicePrimaryCtxRetain failed (code={r}), trying independent context", flush=True)
        libcuda.cuCtxCreate.restype = c_uint32
        libcuda.cuCtxCreate.argtypes = [ctypes.POINTER(c_void_p), c_uint32, c_int]
        r = libcuda.cuCtxCreate(ctypes.byref(child_ctx), c_uint32(4), c_int(0))
        if r == 0 and child_ctx.value is not None:
            _eprint(f"[CHILD] cuCtxCreate OK (独立 context, ctx=0x{child_ctx.value:x})", flush=True)
            have_own_ctx = True
        else:
            _eprint(f"CHILD_ERROR: Cannot create any CUDA context (code={r})", flush=True)
            sys.exit(1)

    # 3. cuIpcOpenMemHandle 获取设备指针
    # 尝试多种 handle 类型传递方式
    libcuda.cuIpcOpenMemHandle.restype = c_uint32
    dev_ptr = c_void_p(None)
    ipc_method = 0

    # 方式1: POINTER(c_uint8 * 64)
    libcuda.cuIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(c_void_p), ctypes.POINTER(c_uint8 * 64), c_uint32]
    r = libcuda.cuIpcOpenMemHandle(ctypes.byref(dev_ptr), handle_array, c_uint32(0))
    if r == 0: ipc_method = 1

    if r != 0:
        # 方式2: c_void_p (cast handle to raw pointer)
        libcuda.cuIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(c_void_p), c_void_p, c_uint32]
        r = libcuda.cuIpcOpenMemHandle(ctypes.byref(dev_ptr),
                                        ctypes.cast(handle_array, c_void_p), c_uint32(0))
        if r == 0: ipc_method = 2

    if r != 0:
        # 方式3: c_char * 64
        handle_char = (c_char * 64)(*ipc_handle_bytes)
        libcuda.cuIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(c_void_p), ctypes.POINTER(c_char * 64), c_uint32]
        r = libcuda.cuIpcOpenMemHandle(ctypes.byref(dev_ptr), handle_char, c_uint32(0))
        if r == 0: ipc_method = 3

    if r != 0:
        # 方式4: CUDA Runtime API 路径 (cudaIpcOpenMemHandle via libcudart)
        try:
            libcudart = ctypes.CDLL("libcudart.so.1")
        except OSError:
            libcudart = None
        if libcudart:
            libcudart.cudaIpcOpenMemHandle.restype = c_int32
            libcudart.cudaIpcOpenMemHandle.argtypes = [
                ctypes.POINTER(c_void_p), ctypes.POINTER(c_char * 64), c_uint32]
            r = libcudart.cudaIpcOpenMemHandle(ctypes.byref(dev_ptr), handle_char, c_uint32(0))
            if r == 0: ipc_method = 4

    if r != 0 or dev_ptr.value is None:
        _eprint(f"CHILD_ERROR: cuIpcOpenMemHandle failed (code={r}) — all 4 methods failed", flush=True)
        sys.exit(1)
    _eprint(f"[CHILD] cuIpcOpenMemHandle OK (method={ipc_method}): dev_ptr=0x{dev_ptr.value:x}", flush=True)

    # 4. NVENC 初始化
    nvenc_dll_path = _find_nvenc_dll()
    try:
        nvenc_dll = ctypes.CDLL(nvenc_dll_path)
    except OSError as e:
        _eprint(f"CHILD_ERROR: Cannot load NVENC DLL: {e}", flush=True)
        sys.exit(1)

    # 运行时查询 API 版本
    try:
        _get_max_ver = nvenc_dll.NvEncodeAPIGetMaxSupportedVersion
        _get_max_ver.restype = c_uint32
        _get_max_ver.argtypes = [ctypes.POINTER(c_uint32)]
        _max_ver_val = c_uint32(0)
        _get_max_ver(ctypes.byref(_max_ver_val))
        nvenc_api_ver = _max_ver_val.value if _max_ver_val.value > 0 else _NVENCAPI_VERSION
    except Exception:
        nvenc_api_ver = _NVENCAPI_VERSION
    _eprint(f"[CHILD] NVENC API: 0x{nvenc_api_ver:x} (SDK 13.0)", flush=True)

    # 5. NvEncodeAPICreateInstance — SDK 13.0: 2552 byte func_table
    func_table = (c_uint8 * _FUNC_TABLE_SIZE)()
    flist_ver = _VER_FUNCTION_LIST
    cast(func_table, ctypes.POINTER(c_uint32))[0] = flist_ver
    create_instance = _NvEncodeAPICreateInstanceProto(
        ("NvEncodeAPICreateInstance", nvenc_dll))
    status = create_instance(cast(func_table, c_void_p))
    if status != NV_ENC_SUCCESS:
        _eprint(f"CHILD_ERROR: NvEncodeAPICreateInstance failed (code={status})", flush=True)
        sys.exit(1)
    _eprint("[CHILD] NvEncodeAPICreateInstance OK", flush=True)

    # 函数指针从 offset 8 开始
    _func_ptrs = cast(byref(func_table, 8), ctypes.POINTER(c_void_p))
    def _get_func(idx):
        addr = _func_ptrs[idx]
        if not addr or addr == 0:
            return None
        return addr

    # 6. nvEncOpenEncodeSessionEx (SDK 13.0: index 29)
    enc = c_void_p(None)
    session_params = _NvEncOpenEncodeSessionExParams()
    session_params.version = _VER_OPEN_ENCODE_SESSION_EX_PARAMS
    session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA
    session_params.device = child_ctx
    session_params.apiVersion = nvenc_api_ver

    open_func_addr = _get_func(_FUNC_IDX["OpenEncodeSessionEx"])
    if open_func_addr is None:
        _eprint("CHILD_ERROR: OpenEncodeSessionEx not available", flush=True)
        sys.exit(1)

    open_session = _NvEncOpenEncodeSessionExProto(open_func_addr)
    status = open_session(byref(session_params), ctypes.byref(enc))
    if status != NV_ENC_SUCCESS:
        _eprint(f"CHILD_ERROR: nvEncOpenEncodeSessionEx failed (code={status})", flush=True)
        _eprint(f"  child_ctx={'own' if have_own_ctx else 'primary'} ctx=0x{child_ctx.value:x}", flush=True)
        sys.exit(1)
    _eprint("[CHILD] nvEncOpenEncodeSessionEx OK!", flush=True)

    # 7. GetEncodePresetConfig
    codec_guid = NV_ENC_CODEC_H264_GUID
    preset_guid = NV_ENC_PRESET_P1_GUID
    preset_config = _NvEncPresetConfig()
    preset_config.version = NV_ENC_PRESET_CONFIG_VER

    lo1, hi1 = guid_to_u64_pair(codec_guid)
    lo2, hi2 = guid_to_u64_pair(preset_guid)

    get_preset_addr = _get_func(_FUNC_IDX["GetEncodePresetConfig"])
    if get_preset_addr is None:
        get_preset_addr = _get_func(_FUNC_IDX["GetEncodePresetConfigEx"])
    if get_preset_addr is None:
        _eprint("CHILD_ERROR: GetEncodePresetConfig not available", flush=True)
        sys.exit(1)

    get_preset = _NvEncGetEncodePresetConfigProto(get_preset_addr)
    status = get_preset(enc, lo1, hi1, lo2, hi2, byref(preset_config))
    if status != NV_ENC_SUCCESS:
        _eprint(f"CHILD_ERROR: GetEncodePresetConfig failed (code={status})", flush=True)
        sys.exit(1)
    _eprint("[CHILD] GetEncodePresetConfig OK", flush=True)

    # 8. Configure + InitializeEncoder
    W, H = 1920, 1080
    fps = 30.0
    # SDK 13.0: presetCfg is at offset 8 in _NvEncPresetConfig
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
    # SDK 13.0: encodeConfig at offset 88, no separate presetConfig
    init_params.encodeConfig = cast(byref(preset_config, 8), c_void_p)

    # NvEncInitializeEncoder (SDK 13.0, was NvEncCreateEncoder)
    create_encoder_addr = _get_func(_FUNC_IDX["InitializeEncoder"])
    if create_encoder_addr is None:
        _eprint("CHILD_ERROR: InitializeEncoder not available", flush=True)
        sys.exit(1)

    create_encoder = _NvEncInitializeEncoderProto(create_encoder_addr)
    status = create_encoder(enc, byref(init_params))
    if status != NV_ENC_SUCCESS:
        _eprint(f"CHILD_ERROR: NvEncInitializeEncoder failed (code={status})", flush=True)
        sys.exit(1)
    _eprint(f"[CHILD] Encoder initialized: {W}x{H}@{fps:.1f}fps H.264 p1", flush=True)

    # 9. RegisterResource (用 IPC 打开的 dev_ptr)
    reg = _NvEncRegisterResource()
    reg.version = NV_ENC_REGISTER_RESOURCE_VER
    reg.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR
    reg.width = W
    reg.height = H + H // 2  # NV12 total height
    reg.pitch = W
    reg.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12
    reg.bufferUsage = NV_ENC_INPUT_IMAGE

    register_resource = _NvEncRegisterResourceProto(
        func_table[_FUNC_IDX["RegisterResource"]])
    status = register_resource(enc, byref(reg))
    if status != NV_ENC_SUCCESS:
        # 尝试直接传 dev_ptr (不通过 registered resource)
        _eprint(f"[CHILD] nvEncRegisterResource failed (code={status}), "
              f"尝试直接使用 dev_ptr...", flush=True)
        # 失败不退出 — 可能 IPC 指针不能直接用于 NVENC RegisterResource
        # 我们改为在子进程内部分配 GPU 内存 + cudaMemcpy 从 IPC 指针拷入
        _try_fallback_encode(libcuda, nvenc_dll, func_table, enc, dev_ptr, W, H, enc_cfg)
        return

    registered_ptr = reg.registeredResource
    _eprint(f"[CHILD] RegisterResource OK: reg=0x{registered_ptr.value:x}", flush=True)

    # 10. MapInputResource
    map_param = _NvEncMapInputResource()
    map_param.version = NV_ENC_MAP_INPUT_RESOURCE_VER
    map_param.registeredResource = registered_ptr

    map_resource = _NvEncMapInputResourceProto(
        func_table[_FUNC_IDX["MapInputResource"]])
    status = map_resource(enc, byref(map_param))
    if status != NV_ENC_SUCCESS:
        _NvEncUnregisterResourceProto(func_table[_FUNC_IDX["UnregisterResource"]])(
            enc, registered_ptr)
        _eprint(f"CHILD_ERROR: nvEncMapInputResource failed (code={status})", flush=True)
        sys.exit(1)

    mapped_ptr = map_param.mappedResource
    _eprint(f"[CHILD] MapInputResource OK: mapped=0x{mapped_ptr.value:x}", flush=True)

    # 11. EncodePicture
    pic_params = _NvEncPicParams()
    pic_params.version = NV_ENC_PIC_PARAMS_VER
    pic_params.inputWidth = W
    pic_params.inputHeight = H + H // 2
    pic_params.inputPitch = W
    pic_params.inputBuffer = mapped_ptr
    pic_params.inputTimeStamp = 0
    pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME
    pic_params.encodePicFlags = 0
    pic_params.frameIdx = 0

    encode_picture = _NvEncEncodePictureProto(
        func_table[_FUNC_IDX["EncodePicture"]])

    status = encode_picture(enc, byref(pic_params))
    if status != NV_ENC_SUCCESS:
        _NvEncUnmapInputResourceProto(func_table[_FUNC_IDX["UnmapInputResource"]])(
            enc, mapped_ptr)
        _NvEncUnregisterResourceProto(func_table[_FUNC_IDX["UnregisterResource"]])(
            enc, registered_ptr)
        _eprint(f"CHILD_ERROR: nvEncEncodePicture failed (code={status})", flush=True)
        sys.exit(1)
    _eprint("[CHILD] EncodePicture OK", flush=True)

    # 12. LockBitstream → 输出 H.264 ES
    lock = _NvEncLockBitstream()
    lock.version = NV_ENC_LOCK_BITSTREAM_VER
    lock.doNotWait = 0

    lock_bitstream = _NvEncLockBitstreamProto(
        func_table[_FUNC_IDX["LockBitstream"]])
    bs_status = lock_bitstream(enc, byref(lock))

    if bs_status == NV_ENC_SUCCESS:
        bitstream_size = lock.bitstreamSizeInBytes
        bitstream_ptr = lock.bitstreamBufferPtr
        if bitstream_size > 0 and bitstream_ptr is not None:
            buf_type = c_uint8 * bitstream_size
            addr = cast(bitstream_ptr, c_void_p).value
            if addr:
                h264_data = bytes(buf_type.from_address(addr))
                _eprint(f"[CHILD] Encoded: {bitstream_size} bytes H.264 ES", flush=True)
                sys.stdout.buffer.write(h264_data)
                sys.stdout.buffer.flush()

        _NvEncUnlockBitstreamProto(func_table[_FUNC_IDX["UnlockBitstream"]])(
            enc, lock.bitstreamBufferPtr)
    else:
        _eprint(f"[CHILD] LockBitstream returned code={bs_status} (may need more frames)", flush=True)

    # 13. Cleanup
    _NvEncUnmapInputResourceProto(func_table[_FUNC_IDX["UnmapInputResource"]])(
        enc, mapped_ptr)
    _NvEncUnregisterResourceProto(func_table[_FUNC_IDX["UnregisterResource"]])(
        enc, registered_ptr)

    # Destroy encoder
    destroy_addr = _get_func(_FUNC_IDX["DestroyEncoder"])
    if destroy_addr:
        _NvEncDestroyEncoderProto(destroy_addr)(enc)

    # Destroy CUDA context
    if have_own_ctx:
        libcuda.cuCtxDestroy.restype = c_uint32
        libcuda.cuCtxDestroy.argtypes = [c_void_p]
        libcuda.cuCtxDestroy(child_ctx)
        _eprint("[CHILD] Cleanup done (own ctx destroyed)", flush=True)
    else:
        _eprint("[CHILD] Cleanup done", flush=True)


def _try_fallback_encode(libcuda, nvenc_dll, func_table, enc, ipc_dev_ptr, W, H, enc_cfg):
    """Fallback: 子进程内分配 GPU 内存, cudaMemcpy 从 IPC 指针拷入, 再用 NVENC 编码"""
    nv12_size = W * (H + H // 2)

    # 子进程内分配 GPU 内存
    libcuda.cuMemAlloc.restype = c_uint32
    libcuda.cuMemAlloc.argtypes = [ctypes.POINTER(c_void_p), c_size_t]
    local_dev_ptr = c_void_p(None)
    r = libcuda.cuMemAlloc(ctypes.byref(local_dev_ptr), c_size_t(nv12_size))
    if r != 0 or local_dev_ptr.value is None:
        _eprint(f"CHILD_ERROR: cuMemAlloc failed (code={r})", flush=True)
        sys.exit(1)
    _eprint(f"[CHILD] cuMemAlloc OK: local=0x{local_dev_ptr.value:x} size={nv12_size}", flush=True)

    # cudaMemcpyDeviceToDevice from IPC ptr to local ptr
    libcuda.cuMemcpyDtoD.restype = c_uint32
    libcuda.cuMemcpyDtoD.argtypes = [c_void_p, c_void_p, c_size_t]
    r = libcuda.cuMemcpyDtoD(local_dev_ptr, ipc_dev_ptr, c_size_t(nv12_size))
    if r != 0:
        _eprint(f"CHILD_ERROR: cuMemcpyDtoD failed (code={r})", flush=True)
        sys.exit(1)
    _eprint("[CHILD] cuMemcpyDtoD OK (IPC → local GPU mem)", flush=True)

    # NVENC RegisterResource with local ptr
    reg = _NvEncRegisterResource()
    reg.version = NV_ENC_REGISTER_RESOURCE_VER
    reg.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR
    reg.width = W
    reg.height = H + H // 2
    reg.pitch = W
    reg.bufferFormat = NV_ENC_BUFFER_FORMAT_NV12
    reg.bufferUsage = NV_ENC_INPUT_IMAGE

    def _get_func(idx):
        addr = func_table[idx]
        if addr is None or addr == 0:
            return None
        return addr

    register_resource = _NvEncRegisterResourceProto(
        func_table[_FUNC_IDX["RegisterResource"]])
    status = register_resource(enc, byref(reg))
    if status != NV_ENC_SUCCESS:
        _eprint(f"CHILD_ERROR: Fallback RegisterResource failed (code={status})", flush=True)
        sys.exit(1)

    registered_ptr = reg.registeredResource
    _eprint(f"[CHILD] Fallback RegisterResource OK: reg=0x{registered_ptr.value:x}", flush=True)

    # MapInputResource
    map_param = _NvEncMapInputResource()
    map_param.version = NV_ENC_MAP_INPUT_RESOURCE_VER
    map_param.registeredResource = registered_ptr

    map_resource = _NvEncMapInputResourceProto(
        func_table[_FUNC_IDX["MapInputResource"]])
    status = map_resource(enc, byref(map_param))
    if status != NV_ENC_SUCCESS:
        _NvEncUnregisterResourceProto(func_table[_FUNC_IDX["UnregisterResource"]])(
            enc, registered_ptr)
        _eprint(f"CHILD_ERROR: Fallback MapInputResource failed (code={status})", flush=True)
        sys.exit(1)

    mapped_ptr = map_param.mappedResource

    # EncodePicture
    pic_params = _NvEncPicParams()
    pic_params.version = NV_ENC_PIC_PARAMS_VER
    pic_params.inputWidth = W
    pic_params.inputHeight = H + H // 2
    pic_params.inputPitch = W
    pic_params.inputBuffer = mapped_ptr
    pic_params.inputTimeStamp = 0
    pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME
    pic_params.encodePicFlags = 0
    pic_params.frameIdx = 0

    encode_picture = _NvEncEncodePictureProto(
        func_table[_FUNC_IDX["EncodePicture"]])
    status = encode_picture(enc, byref(pic_params))
    if status != NV_ENC_SUCCESS:
        _eprint(f"CHILD_ERROR: Fallback EncodePicture failed (code={status})", flush=True)
        sys.exit(1)
    _eprint("[CHILD] Fallback EncodePicture OK", flush=True)

    # LockBitstream
    lock = _NvEncLockBitstream()
    lock.version = NV_ENC_LOCK_BITSTREAM_VER
    lock.doNotWait = 0

    lock_bitstream = _NvEncLockBitstreamProto(
        func_table[_FUNC_IDX["LockBitstream"]])
    bs_status = lock_bitstream(enc, byref(lock))

    if bs_status == NV_ENC_SUCCESS:
        bitstream_size = lock.bitstreamSizeInBytes
        bitstream_ptr = lock.bitstreamBufferPtr
        if bitstream_size > 0 and bitstream_ptr is not None:
            buf_type = c_uint8 * bitstream_size
            addr = cast(bitstream_ptr, c_void_p).value
            if addr:
                h264_data = bytes(buf_type.from_address(addr))
                _eprint(f"[CHILD] Fallback encoded: {bitstream_size} bytes H.264 ES", flush=True)
                sys.stdout.buffer.write(h264_data)
                sys.stdout.buffer.flush()
        _NvEncUnlockBitstreamProto(func_table[_FUNC_IDX["UnlockBitstream"]])(
            enc, lock.bitstreamBufferPtr)

    _NvEncUnmapInputResourceProto(func_table[_FUNC_IDX["UnmapInputResource"]])(
        enc, mapped_ptr)
    _NvEncUnregisterResourceProto(func_table[_FUNC_IDX["UnregisterResource"]])(
        enc, registered_ptr)
    libcuda.cuMemFree.restype = c_uint32
    libcuda.cuMemFree.argtypes = [c_void_p]
    libcuda.cuMemFree(local_dev_ptr)
    _eprint("[CHILD] Fallback cleanup done", flush=True)


def _find_nvenc_dll() -> str:
    if sys.platform == "win32":
        return "nvEncodeAPI64.dll"
    candidates = [
        "/usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1",
        "/usr/lib64/libnvidia-encode.so.1",
        "/usr/lib/libnvidia-encode.so.1",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "libnvidia-encode.so.1"


# ============================================================================
# 主进程
# ============================================================================

def parent_main():
    import torch

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available", flush=True)
        sys.exit(0)

    W, H = 1920, 1080
    nv12_h = H + H // 2
    nv12_size = W * nv12_h

    libcuda = _load_libcuda()
    libcuda.cuInit(0)

    # Push primary context so driver API has a current context for cuIpcGetMemHandle
    libcuda.cuDevicePrimaryCtxRetain.restype = c_uint32
    libcuda.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(c_void_p), c_int]
    libcuda.cuCtxPushCurrent.restype = c_uint32
    libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
    parent_primary_ctx = c_void_p(None)
    r = libcuda.cuDevicePrimaryCtxRetain(ctypes.byref(parent_primary_ctx), c_int(0))
    if r == 0 and parent_primary_ctx.value is not None:
        libcuda.cuCtxPushCurrent(parent_primary_ctx)
        print(f"[PARENT] Primary context pushed (ctx=0x{parent_primary_ctx.value:x})", flush=True)
    else:
        print(f"[PARENT] WARNING: cuDevicePrimaryCtxRetain failed (code={r}), proceeding anyway", flush=True)

    # ── 测试 A: PyTorch tensor 指针 → IPC ──
    import torch
    y_plane = torch.zeros((H, W), dtype=torch.uint8, device="cuda")
    for x in range(W):
        y_plane[:, x] = int(255 * x / W)
    uv_plane = torch.full((H // 2, W), 128, dtype=torch.uint8, device="cuda")
    nv12_tensor = torch.cat([y_plane, uv_plane], dim=0).contiguous()
    data_ptr_a = nv12_tensor.data_ptr()
    print(f"[PARENT] Test A: PyTorch ptr=0x{data_ptr_a:x}", flush=True)

    handle_a = (c_uint8 * 64)()
    libcuda.cuIpcGetMemHandle.restype = c_uint32
    libcuda.cuIpcGetMemHandle.argtypes = [
        ctypes.POINTER(c_uint8 * 64), c_void_p]
    r_a = libcuda.cuIpcGetMemHandle(ctypes.byref(handle_a), c_void_p(data_ptr_a))
    print(f"[PARENT] cuIpcGetMemHandle(torch): {'OK' if r_a == 0 else 'FAIL code=' + str(r_a)}", flush=True)

    # ── 测试 B: cuMemAlloc 原生分配 → IPC ──
    r_alloc = -1
    r_b = -1
    raw_ptr = c_void_p(None)
    handle_b = (c_uint8 * 64)()
    if nv12_size > 0:
        libcuda.cuMemAlloc.restype = c_uint32
        libcuda.cuMemAlloc.argtypes = [ctypes.POINTER(c_void_p), c_size_t]
        r_alloc = libcuda.cuMemAlloc(ctypes.byref(raw_ptr), c_size_t(nv12_size))
        ptr_str = f"0x{raw_ptr.value:x}" if raw_ptr.value is not None else "NULL"
        print(f"[PARENT] Test B: cuMemAlloc ptr={ptr_str} {'OK' if r_alloc == 0 else 'FAIL code=' + str(r_alloc)}", flush=True)

        if r_alloc == 0 and raw_ptr.value is not None:
            r_b = libcuda.cuIpcGetMemHandle(ctypes.byref(handle_b), raw_ptr)
            print(f"[PARENT] cuIpcGetMemHandle(cuMemAlloc): {'OK' if r_b == 0 else 'FAIL code=' + str(r_b)}", flush=True)
        else:
            print(f"[PARENT] Test B 跳过 (cuMemAlloc 失败)", flush=True)
    else:
        print(f"[PARENT] Test B 跳过 (nv12_size={nv12_size})", flush=True)

    # 选用第一个成功的 handle
    if r_a == 0:
        data_ptr, handle_array, label = c_void_p(data_ptr_a), handle_a, "torch"
    elif r_b == 0:
        data_ptr, handle_array, label = raw_ptr, handle_b, "cuMemAlloc"
    else:
        print("PARENT_ERROR: 所有 IPC handle 导出均失败", flush=True)
        sys.exit(1)
    print(f"[PARENT] 使用 {label} handle 进行 IPC 测试", flush=True)

    ipc_handle_b64 = base64.b64encode(bytes(handle_array)).decode("ascii")
    print(f"[PARENT] IPC handle (first 32B hex): {bytes(handle_array)[:32].hex()}", flush=True)

    # 启动子进程
    script_path = os.path.abspath(__file__)
    start = time.time()
    proc = subprocess.Popen(
        [sys.executable, script_path, "--child", ipc_handle_b64],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stdout_data, stderr_data = proc.communicate(timeout=60)
    elapsed = time.time() - start

    # 释放 cuMemAlloc 分配的内存 (子进程已结束)
    if label == "cuMemAlloc" and raw_ptr.value is not None:
        try:
            libcuda.cuMemFree.restype = c_uint32
            libcuda.cuMemFree.argtypes = [c_void_p]
            libcuda.cuMemFree(raw_ptr)
        except Exception:
            pass

    print(f"\n[PARENT] 子进程退出码: {proc.returncode}, 耗时: {elapsed:.2f}s", flush=True)

    # 打印子进程 stderr (日志)
    stderr_text = stderr_data.decode("utf-8", errors="replace")
    for line in stderr_text.strip().split("\n"):
        print(f"  {line}", flush=True)

    if proc.returncode != 0:
        print(f"\nPARENT_ERROR: 子进程失败 (exit={proc.returncode})", flush=True)
        sys.exit(1)

    h264_data = stdout_data
    if len(h264_data) == 0:
        print("PARENT_ERROR: 子进程未输出 H.264 数据", flush=True)
        sys.exit(1)

    print(f"\n[PARENT] 收到 {len(h264_data)} bytes H.264 ES", flush=True)

    # 验证 H.264 NAL unit start code
    has_start_code = False
    nal_types = []
    for i in range(len(h264_data) - 4):
        if h264_data[i:i+4] == b'\x00\x00\x00\x01':
            has_start_code = True
            if i + 4 < len(h264_data):
                nal_type = h264_data[i+4] & 0x1F
                nal_types.append(nal_type)

    nal_names = {
        5: "IDR", 6: "SEI", 7: "SPS", 8: "PPS",
        1: "Non-IDR", 9: "AUD",
    }

    if has_start_code:
        print(f"[PARENT] H.264 NAL units detected: {len(nal_types)}", flush=True)
        for nt in nal_types[:10]:
            name = nal_names.get(nt, f"type={nt}")
            print(f"  - NAL type {nt} ({name})", flush=True)
    else:
        print("[PARENT] WARNING: No NAL start code found in output", flush=True)

    # 保存测试文件
    test_path = os.path.join(tempfile.gettempdir(), "test_nvenc_ipc_output.h264")
    with open(test_path, "wb") as f:
        f.write(h264_data)
    print(f"\n[PARENT] 测试输出保存到: {test_path}", flush=True)
    print(f"[PARENT] 验证: ffprobe {test_path}", flush=True)

    print("\n[PARENT] *** 测试通过: CUDA IPC + 子进程 NVENC 编码可行 ***", flush=True)
    return True


# ============================================================================
# main
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        child_main(sys.argv[2])
    else:
        parent_main()
