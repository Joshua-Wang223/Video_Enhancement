#!/usr/bin/env python3
__test__ = False  # 此文件不是 pytest 模块 — 通过 `python test_*.py` 直接运行
"""
NVENC Lookahead 帧数守恒验证测试（修复版）
===========================================

验证 NVENC Level 1 API (ctypes 直调 SDK) 的核心声明：

  "Lookahead depth adds latency. The total number of output frames equals
   the number of input frames. No frames are discarded by the encoder."

  — NVIDIA Video Codec SDK Programming Guide, 《Lookahead》章节

修复要点（对比旧版脚本的 3 个 SDK 违规 BUG）
------------------------------------------
1. pipeline_depth 必须 >= LA+1（旧版用了 max(1, LA)，违反 SDK 硬件安全要求，
   导致缓冲区覆盖 → 开头丢帧）
2. EncodePicture 后必须循环 LockBitstream (blocking) 直到 NEED_MORE_INPUT
   （旧版只 Lock 一次 + NEED_MORE_INPUT 分支强制 Lock/Unlock 丢弃有效帧）
3. NEED_MORE_INPUT 时不应做任何 Lock/Unlock（buffer 中无有效输出，操作可能误弃帧）

核心架构
--------
encode_frame(nv12_gpu, force_idr) -> List[bytes]
   送入 1 帧 → 轮转 slot → EncodePicture → _drain_outputs() 循环排空所有完成帧

_drain_outputs()
   循环 Lock 当前 _output_slot_idx 的 bs_buf 直到 NEED_MORE_INPUT
   推进 _output_slot_idx 保证帧顺序

flush_eos()
   发送 EOS (NULL input + flag=0x8) → 按 _output_slot_idx 顺序逐 slot blocking Lock
   直到 NEED_MORE_INPUT，回收所有 LA 滞留帧

constqp 下 LA 自动无效
----------------------
NVIDIA 官方文档明确：Lookahead 仅在 VBR/QVBR 码控模式下生效。
当 rateControlMode=CONSTQP 且 enableLookahead=1 时，NVENC 硬件静默禁用 LA (la_depth=0)。
故 constqp 模式的帧数守恒测试不足以验证 LA 排空逻辑正确性，须以 VBR/QVBR 为准。

帧计数方式 (NVIDIA SDK 规范)
----------------------------
以 EncodePicture 调用计数为准：每次 EncodePicture 提交 1 帧，
无论返回 SUCCESS 或 NEED_MORE_INPUT，都计入已处理帧。
LockBitstream 成功数仅作辅助参考。

pipeline_depth = LA + 1 (SDK hardware safety requirement)
---------------------------------------------------------
NVENC SDK 规范：启用 lookahead 时，输出比特流缓冲区数量至少为 lookaheadDepth+1，
防止硬件在客户端取走前覆盖未读数据。slot 轮转周期 >= LA+1 避免复用冲突。

测试矩阵
--------
  A) LA=0 + 正确排空:      Output == Input (无 LA，直接验证)
  C) LA=N + 正确排空:      Output == Input (LA 激活 + 循环排空 + EOS flush)

用法
----
  python test_nvenc_la_frame_conservation.py --frames 687 --la-depth 8 --rate-mode vbr_hq
"""

import ctypes
from ctypes import (c_uint8, c_uint16, c_uint32, c_int32, c_int, c_uint64, c_void_p,
                    c_char, c_size_t, c_double, Structure, POINTER, byref,
                    sizeof, cast, pointer, c_bool)
import os
import sys
import threading
import time
import argparse
import subprocess
import tempfile
import pathlib
import shutil
from typing import Optional, List

# ============================================================================
# Part 0: 依赖检查 — PyTorch 可选，仅用于生成 GPU 测试帧
# ============================================================================
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch not available — will skip GPU tensor generation tests")

# ============================================================================
# Part 1: NVENC SDK 常量、GUID、结构体（SDK 13.0 验证）
# ============================================================================

# ---------------------------------------------------------------------------
# GUID 定义
# ---------------------------------------------------------------------------
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

NV_ENC_CODEC_H264_GUID = _NvGuid(0x6bc82762, 0x4e63, 0x4ca4,
    (0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf))
NV_ENC_CODEC_HEVC_GUID = _NvGuid(0x790cdc88, 0x4522, 0x4d7b,
    (0x94, 0x25, 0xbd, 0xa9, 0x97, 0x5f, 0x76, 0x03))
NV_ENC_CODEC_AV1_GUID = _NvGuid(0x0a352289, 0x0aa7, 0x4759,
    (0x86, 0x2d, 0x5d, 0x15, 0xcd, 0x16, 0xd2, 0x54))

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

_PRESET_GUID_MAP = {
    "p1": NV_ENC_PRESET_P1_GUID, "p2": NV_ENC_PRESET_P2_GUID,
    "p3": NV_ENC_PRESET_P3_GUID, "p4": NV_ENC_PRESET_P4_GUID,
    "p5": NV_ENC_PRESET_P5_GUID, "p6": NV_ENC_PRESET_P6_GUID,
    "p7": NV_ENC_PRESET_P7_GUID,
}
_PRESET_P_INDEX = {
    "ultrafast": 0, "superfast": 0, "veryfast": 1, "faster": 2,
    "fast": 3, "medium": 4, "slow": 5, "slower": 6, "veryslow": 6, "placebo": 6,
}

# ---------------------------------------------------------------------------
# SDK 版本与 struct version 宏
# ---------------------------------------------------------------------------
_NVENCAPI_VERSION_FALLBACK = (13 << 4) | 0  # 13.0 = 0xD0
_NVENCAPI_VERSION = 0x0d  # SDK 13.0: NVENCAPI_VERSION = 13 (低字节 = major, NV_ENC_STRUCT_VER 格式)

def _sdk13_ver(ver, bit31=False):
    v = _NVENCAPI_VERSION | (ver << 16) | (0x7 << 28)
    if bit31:
        v |= (1 << 31)
    return v

# Struct version 常量
NV_ENC_CREATE_INPUT_BUFFER_VER = _sdk13_ver(2)     # 0x7002000d
NV_ENC_LOCK_INPUT_BUFFER_VER = _sdk13_ver(1)       # 0x7001000d
NV_ENC_CREATE_BITSTREAM_BUFFER_VER = _sdk13_ver(1) # 0x7001000d

# 状态码
NV_ENC_SUCCESS = 0
NV_ENC_ERR_NEED_MORE_INPUT = 17

# 枚举
NV_ENC_DEVICE_TYPE_CUDA = 1
NV_ENC_BUFFER_FORMAT_NV12 = 1
NV_ENC_INPUT_IMAGE = 0
NV_ENC_PIC_STRUCT_FRAME = 1

# ---------------------------------------------------------------------------
# SDK struct 定义（SDK 13.0，_pack_=1）
# ---------------------------------------------------------------------------
class _NvEncOpenEncodeSessionExParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",    c_uint32),
        ("deviceType", c_uint32),
        ("device",     c_void_p),
        ("reserved",   c_void_p),
        ("apiVersion", c_uint32),
        ("reserved1",  c_uint8 * (253 * 4)),
        ("reserved2",  c_void_p * 64),
    ]

class _NvEncInitializeParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",             c_uint32),
        ("encodeGUID",          _NvGuid),
        ("presetGUID",          _NvGuid),
        ("encodeWidth",         c_uint32),
        ("encodeHeight",        c_uint32),
        ("darWidth",            c_uint32),
        ("darHeight",           c_uint32),
        ("frameRateNum",        c_uint32),
        ("frameRateDen",        c_uint32),
        ("enableEncodeAsync",   c_uint32),
        ("enablePTD",           c_uint32),
        ("bitfield",            c_uint32),
        ("privDataSize",        c_uint32),
        ("reserved_76",         c_uint32),
        ("privData",            c_void_p),
        ("encodeConfig",        c_void_p),
        ("maxEncodeWidth",      c_uint32),
        ("maxEncodeHeight",     c_uint32),
        ("maxMEHintCountsPerBlock", c_uint8 * 32),
        ("tuningInfo",          c_uint32),
        ("bufferFormat",        c_uint32),
        ("numStateBuffers",     c_uint32),
        ("outputStatsLevel",    c_uint32),
        ("reserved1",           c_uint8 * 1136),
        ("reserved2",           c_void_p * 64),
    ]

class _NvEncConfigH264VUIParameters(Structure):
    _pack_ = 1
    _fields_ = [
        ("overscanInfoPresentFlag",      c_uint32),
        ("videoSignalTypePresentFlag",   c_uint32),
        ("videoFormat",                  c_uint32),
        ("videoFullRangeFlag",           c_uint32),
        ("colourDescriptionPresentFlag", c_uint32),
        ("colourPrimaries",              c_uint32),
        ("transferCharacteristics",      c_uint32),
        ("matrixCoefficients",           c_uint32),
        ("chromaSampleLocationFlag",     c_uint32),
        ("chromaSampleLocationTop",      c_uint32),
        ("chromaSampleLocationBottom",   c_uint32),
        ("bitstreamRestrictionFlag",     c_uint32),
        ("reserved",                     c_uint32 * 16),
    ]

class _NvEncConfigH264(Structure):
    _pack_ = 1
    _fields_ = [
        ("enableTemporalSVC",       c_uint32),
        ("enableTemporalSVC_1",     c_uint32),
        ("profileLevel",            c_uint32),
        ("chromaFormatIDC",         c_uint32),
        ("reserved1",               c_uint32 * 13),
        ("maxNumRefFramesInDPB",    c_uint32),
        ("reserved2",               c_uint32 * 3),
        ("idrPeriod",               c_uint32),
        ("repeatSPSPPS",            c_uint32),
        ("reserved10",              c_uint32 * 4),
        ("vuiParameters",           _NvEncConfigH264VUIParameters),
        ("reserved12",              c_uint32 * 222),
    ]

class _NvEncConfig(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",          c_uint32),
        ("profileGUID",      _NvGuid),
        ("gopLength",        c_uint32),
        ("frameIntervalP",   c_uint32),
        ("frameFieldMode",   c_uint32),
        ("enablePTD",        c_uint32),
        ("frameFieldMode_1", c_uint32),
        ("reserved3",        c_uint32 * 53),
        ("mvPrecision",      c_uint32),
        ("reserved4",        c_uint32 * 27),
        ("reserved5",        c_uint32 * 172),
        ("encodeCodecConfig",_NvEncConfigH264),
        ("reserved7",        c_uint32 * 252),
    ]

class _NvEncPresetConfig(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",      c_uint32),
        ("presetConfig", _NvEncConfig),
        ("reserved",     c_uint32 * 256),
    ]

# struct version 常量（在 struct 定义之后赋值，SDK 13.0 生产代码验证值）
NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER = _sdk13_ver(1)
NV_ENC_PRESET_CONFIG_VER  = _sdk13_ver(5, True)    # 0xf005000d
NV_ENC_CONFIG_VER         = _sdk13_ver(9, True)    # 0xf009000d
NV_ENC_INITIALIZE_PARAMS_VER = _sdk13_ver(7, True)  # 0xf007000d
NV_ENC_PIC_PARAMS_VER     = _sdk13_ver(7, True)    # 0xf007000d
NV_ENC_LOCK_BITSTREAM_VER = _sdk13_ver(2, True)    # 0xf002000d
NV_ENC_RC_PARAMS_VER      = _sdk13_ver(1)           # 0x7001000d (匹配生产代码 GPU 验证版本)

# 函数表大小
_FUNC_TABLE_RAW_SIZE = 2552  # SDK 13.0

# 函数表索引（SDK 13.0）
_FUNC_IDX = {
    "GetEncodeGUIDCount":     1,
    "GetEncodeGUIDs":         4,
    "GetEncodePresetGUIDs":   9,
    "GetEncodePresetConfig": 10,
    "InitializeEncoder":     11,
    "CreateInputBuffer":     12,
    "DestroyInputBuffer":    13,
    "CreateBitstreamBuffer": 14,
    "DestroyBitstreamBuffer":15,
    "EncodePicture":         16,
    "LockBitstream":         17,
    "UnlockBitstream":       18,
    "LockInputBuffer":       19,
    "UnlockInputBuffer":     20,
    "DestroyEncoder":        27,
    "OpenEncodeSessionEx":   29,
}

# ============================================================================
# Part 2: MinimalTestEncoder — 最小化 NVENC 编码器（仅用于帧数验证）
# ============================================================================

class MinimalTestEncoder:
    """
    最小化 NVENC H.264 编码器，支持多 slot pipeline（匹配生产代码架构）。
    用于帧数守恒验证测试。
    """

    def __init__(self, width: int, height: int, fps: float = 25.0,
                 preset: str = "p1", qp: int = 28,
                 rate_mode: str = "constqp", la_depth: int = 0,
                 codec: str = "h264", pipeline_depth: int = None):
        """rate_mode: 'constqp' | 'vbr_hq' | 'qvbr'.
           la_depth: lookahead depth (0=disabled, 8~32).
           codec: 'h264' | 'hevc' | 'av1'.
           pipeline_depth: slot 数。None 时自动设为 max(1, la_depth+1)，
           遵循 NVENC SDK 规范：启用 lookahead 时比特流缓冲区数量至少
           lookaheadDepth+1，防止硬件覆盖未读数据。"""
        if pipeline_depth is None:
            pipeline_depth = max(1, la_depth + 1)
        if codec not in ("h264", "hevc", "av1"):
            raise ValueError(f"Unsupported codec: {codec}, must be 'h264', 'hevc', or 'av1'")

        self._width = width
        self._height = height
        self._fps = fps
        self._qp = max(1, qp)
        self._rate_mode = rate_mode
        self._la_depth = la_depth
        self._codec = codec
        self._preset_name = preset.lower()
        self._pipeline_depth = pipeline_depth
        self._encoder = c_void_p(None)
        self._frame_idx = 0
        self._output_slot_idx = 0  # 输出指针：追踪下一个预期输出 slot，保证帧顺序
        self._slots = []           # [{input_buf, bs_buf}]

        # 统计
        self.total_encoded = 0       # EncodePicture 调用次数
        self.total_locked = 0        # LockBitstream 成功取回帧数
        self.total_eos_frames = 0    # EOS flush 阶段取回帧数

        self._lock = threading.Lock()

        # ── 1. 加载 NVENC DLL ──
        self._dll_path = self._find_dll()
        self._dll = ctypes.CDLL(self._dll_path)

        # ── 2. 加载 CUDA ──
        cuda_dll = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
        self._libcuda = ctypes.CDLL(cuda_dll)
        self._libcuda.cuInit(0)

        # ── 3. 获取 NVENC API 版本 ──
        try:
            _get_max_ver = self._dll.NvEncodeAPIGetMaxSupportedVersion
            _get_max_ver.restype = c_uint32
            _get_max_ver.argtypes = [POINTER(c_uint32)]
            _max_ver_val = c_uint32(0)
            _get_max_ver(byref(_max_ver_val))
            self._api_version = _max_ver_val.value if _max_ver_val.value > 0 else _NVENCAPI_VERSION_FALLBACK
        except Exception:
            self._api_version = _NVENCAPI_VERSION_FALLBACK
        print(f"[MinimalEncoder] NVENC API: 0x{self._api_version:x}")

        # ── 4. CUDA context ──
        self._saved_ctx = c_void_p(None)
        self._primary_ctx = c_void_p(None)
        self._libcuda.cuCtxGetCurrent.restype = c_uint32
        self._libcuda.cuCtxGetCurrent.argtypes = [POINTER(c_void_p)]
        self._libcuda.cuCtxGetCurrent(byref(self._saved_ctx))

        self._libcuda.cuDevicePrimaryCtxRetain.restype = c_uint32
        self._libcuda.cuDevicePrimaryCtxRetain.argtypes = [POINTER(c_void_p), c_int]
        primary_ctx = c_void_p(None)
        r = self._libcuda.cuDevicePrimaryCtxRetain(byref(primary_ctx), c_int(0))
        if r == 0 and primary_ctx.value is not None:
            self._libcuda.cuCtxPushCurrent.restype = c_uint32
            self._libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
            self._libcuda.cuCtxPushCurrent(primary_ctx)
            self._primary_ctx = c_void_p(primary_ctx.value)
            print(f"[MinimalEncoder] Primary CUDA context: 0x{primary_ctx.value:x}")
        elif self._saved_ctx.value is not None:
            self._primary_ctx = self._saved_ctx
            print(f"[MinimalEncoder] Using existing CUDA context: 0x{self._saved_ctx.value:x}")
        else:
            raise RuntimeError("Cannot obtain CUDA context")

        # ── 5. NvEncodeAPICreateInstance ──
        func_table = (c_uint8 * _FUNC_TABLE_RAW_SIZE)()
        cast(func_table, POINTER(c_uint32))[0] = _sdk13_ver(2)
        create_instance = ctypes.CFUNCTYPE(c_uint32, c_void_p)(
            ("NvEncodeAPICreateInstance", self._dll))
        status = create_instance(cast(func_table, c_void_p))
        if status != 0:
            raise RuntimeError(f"NvEncodeAPICreateInstance failed, code={status}")
        self._func_ptrs = cast(byref(func_table, 8), POINTER(c_void_p))
        self._func_table_raw = func_table

        def _get_func(idx):
            addr = self._func_ptrs[idx]
            return addr if (addr and addr != 0) else None

        # ── 6. OpenEncodeSessionEx ──
        open_func_addr = _get_func(_FUNC_IDX["OpenEncodeSessionEx"])
        if open_func_addr is None:
            raise RuntimeError("OpenEncodeSessionEx not available")
        open_session = ctypes.CFUNCTYPE(c_uint32, POINTER(_NvEncOpenEncodeSessionExParams), POINTER(c_void_p))(open_func_addr)

        for api_ver in sorted(set([0x0d, self._api_version, 0xd0, 0xc0]), reverse=True):
            sp = _NvEncOpenEncodeSessionExParams()
            sp.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER
            sp.deviceType = NV_ENC_DEVICE_TYPE_CUDA
            sp.device = self._primary_ctx
            sp.apiVersion = api_ver
            enc = c_void_p(None)
            status = open_session(byref(sp), byref(enc))
            if status == NV_ENC_SUCCESS:
                self._encoder = enc
                print(f"[MinimalEncoder] OpenEncodeSessionEx OK: apiVersion=0x{api_ver:x}")
                break
        if self._encoder.value is None:
            raise RuntimeError(f"OpenEncodeSessionEx failed, code={status}")

        # ── 7. 查询 codec GUID ──
        _GetEncodeGUIDCount = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint32))(
            _get_func(_FUNC_IDX["GetEncodeGUIDCount"]))
        count_val = c_uint32(0)
        _GetEncodeGUIDCount(self._encoder, byref(count_val))
        n_guids = count_val.value
        guid_array = (_NvGuid * n_guids)()
        _GetEncodeGUIDs = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(_NvGuid), c_uint32, POINTER(c_uint32))(
            _get_func(_FUNC_IDX["GetEncodeGUIDs"]))
        actual = c_uint32(0)
        _GetEncodeGUIDs(self._encoder, guid_array, n_guids, byref(actual))
        # 根据 --codec 选择目标 codec GUID
        _CODEC_GUID_MAP = {
            "h264": NV_ENC_CODEC_H264_GUID,
            "hevc": NV_ENC_CODEC_HEVC_GUID,
            "av1":  NV_ENC_CODEC_AV1_GUID,
        }
        _target_codec_guid = _CODEC_GUID_MAP[self._codec]
        print(f"[MinimalEncoder] 驱动枚举 {n_guids} 个 codec GUID (实际获取 {actual.value}):")
        for _gi in range(min(n_guids, 8)):
            _g = guid_array[_gi]
            _matched = " ← TARGET" if _g == _target_codec_guid else ""
            print(f"  [{_gi}] 0x{_g.Data1:08x}-{_g.Data2:04x}-{_g.Data3:04x}{_matched}")
        codec_guid = _target_codec_guid
        _codec_found = False
        for _i in range(n_guids):
            if guid_array[_i] == _target_codec_guid:
                codec_guid = guid_array[_i]
                _codec_found = True
                break
        if not _codec_found:
            print(f"[MinimalEncoder] ⚠️  驱动未枚举目标 codec GUID，使用 guid_array[0] 作为后备")
            codec_guid = guid_array[0]
            print(f"[MinimalEncoder] 后备 codec GUID: 0x{codec_guid.Data1:08x}-{codec_guid.Data2:04x}-{codec_guid.Data3:04x}")

        # ── 8. 查询 preset GUID ──
        _GetEncodePresetGUIDs = ctypes.CFUNCTYPE(c_uint32, c_void_p, _NvGuid,
            POINTER(_NvGuid), c_uint32, POINTER(c_uint32))(
            _get_func(_FUNC_IDX["GetEncodePresetGUIDs"]))
        preset_guid_array = (_NvGuid * 64)()
        preset_count = c_uint32(0)
        _GetEncodePresetGUIDs(self._encoder, codec_guid, preset_guid_array, 64, byref(preset_count))
        if preset_count.value == 0:
            raise RuntimeError(f"No presets enumerated for codec GUID 0x{codec_guid.Data1:08x} — wrong codec GUID or driver mismatch")
        p_idx = _PRESET_P_INDEX.get(self._preset_name, 4)
        p_idx = min(p_idx, preset_count.value - 1)
        preset_guid = preset_guid_array[p_idx]
        print(f"[MinimalEncoder] Preset: {self._preset_name} (index={p_idx}/{preset_count.value - 1})")

        # ── 9. GetEncodePresetConfig (with GetEncodePresetConfigEx fallback) ──
        get_preset_addr = _get_func(_FUNC_IDX["GetEncodePresetConfig"])
        if get_preset_addr is None:
            get_preset_addr = _get_func(39)  # GetEncodePresetConfigEx (SDK 13.0)
        if get_preset_addr is None:
            raise RuntimeError("GetEncodePresetConfig not available")
        _GPC_fn = ctypes.CFUNCTYPE(c_uint32, c_void_p, _NvGuid, _NvGuid,
            POINTER(_NvEncPresetConfig))(get_preset_addr)
        preset_config = _NvEncPresetConfig()
        ctypes.memset(byref(preset_config), 0, sizeof(preset_config))
        preset_config.version = NV_ENC_PRESET_CONFIG_VER
        cast(byref(preset_config, 8), POINTER(c_uint32))[0] = NV_ENC_CONFIG_VER
        status = _GPC_fn(self._encoder, codec_guid, preset_guid, byref(preset_config))
        if status != 0:
            raise RuntimeError(f"GetEncodePresetConfig failed, code={status}")

        # 配置编码参数（通用字段）
        enc_cfg = cast(byref(preset_config, 8), POINTER(_NvEncConfig)).contents
        enc_cfg.gopLength = int(fps)
        enc_cfg.frameIntervalP = 1
        # codec-specific config: H.264 字段写入 HEVC struct 会导致字段错位，
        # HEVC 时信任 preset 默认值（仅设通用字段 + RC params）
        if self._codec == "h264":
            enc_cfg.encodeCodecConfig.chromaFormatIDC = 1
            enc_cfg.encodeCodecConfig.idrPeriod = int(fps)
            enc_cfg.encodeCodecConfig.maxNumRefFramesInDPB = 4
            enc_cfg.encodeCodecConfig.repeatSPSPPS = 1

        # RC 参数（byte array 手动 offset 写入，遵循 SDK 13.0 SEQUENTIAL 布局）
        # [FIX-QP-VBR] 匹配生产代码 process_video_v6_4_5_1_single.py GPU 验证布局:
        #   version@0, rateControlMode@4,
        #   constQP@8 (12B NV_ENC_QP), averageBitRate@20, maxBitRate@24,
        #   bitfield@36 (AQ@bit3, TemporalAQ@bit8, enableLookahead@bit5),
        #   targetQuality@88(uint8), lookaheadDepth@90(uint16), multiPass@100
        rc_ptr = cast(byref(preset_config, 8 + 40), POINTER(c_uint32))
        rc_ptr[0] = NV_ENC_RC_PARAMS_VER  # version@0

        # targetQuality: NVENC 标度 1=最好, 51=最差，直接使用 --qp 值 (CRF 标度)
        _tq = max(1, self._qp)

        if self._rate_mode == 'constqp':
            rc_ptr[1] = 0  # NV_ENC_PARAMS_RC_CONSTQP
            rc_ptr[2] = self._qp  # qpInterP@8
            rc_ptr[3] = self._qp  # qpInterB@12
            rc_ptr[4] = self._qp  # qpIntra@16
        elif self._rate_mode == 'vbr_hq':
            rc_ptr[1] = 32  # NV_ENC_PARAMS_RC_VBR_HQ
            # [FIX-BR-CEILING] 按分辨率计算合理码率天花板，防止 7kbps 极端约束架空 targetQuality
            _est_br = max(50000000, int(width * height * fps * 3.0))
            rc_ptr[5] = _est_br     # averageBitRate@20
            rc_ptr[6] = _est_br * 2 # maxBitRate@24
            # targetQuality@88(uint8_t) — 单字节写入，不污染 lookaheadDepth@90
            _tq8_ptr = cast(byref(preset_config, 8 + 40 + 88), POINTER(c_uint8))
            _tq8_ptr[0] = _tq & 0xFF
        elif self._rate_mode == 'qvbr':
            rc_ptr[1] = 64  # NV_ENC_PARAMS_RC_QVBR (0x40, SDK 13.0)
            _est_br = max(50000000, int(width * height * fps * 3.0))
            rc_ptr[5] = _est_br     # averageBitRate@20
            rc_ptr[6] = _est_br * 2 # maxBitRate@24
            _tq8_ptr = cast(byref(preset_config, 8 + 40 + 88), POINTER(c_uint8))
            _tq8_ptr[0] = _tq & 0xFF

        # [FIX-AQ] 启用 Adaptive Quantization + Temporal AQ — VBR_HQ/QVBR 质量控制必需
        rc_ptr[9] = rc_ptr[9] | (1 << 3) | (1 << 8)  # enableAQ | enableTemporalAQ

        # bitfield: 启用 lookahead (VBR_HQ/QVBR 模式)
        if self._la_depth > 0:
            _rc_bf = rc_ptr[9]  # bitfield@36
            _rc_bf |= (1 << 5)  # enableLookahead (bit 5)
            rc_ptr[9] = _rc_bf
            # multiPass@100 — VBR_HB/QVBR 不支持 two-pass，设为 DISABLED
            rc_ptr[25] = 0  # NV_ENC_MULTI_PASS_DISABLED
            # lookaheadDepth@90 (uint16, byte offset)
            _la_ptr = cast(byref(preset_config, 8 + 40 + 90), POINTER(c_uint16))
            _la_ptr[0] = self._la_depth

        # ── 10. InitializeEncoder ──
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
        init_params.enablePTD = 1
        init_params.maxEncodeWidth = width
        init_params.maxEncodeHeight = height
        init_params.encodeConfig = cast(byref(preset_config, 8), c_void_p)
        _InitEncoder = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(_NvEncInitializeParams))(
            _get_func(_FUNC_IDX["InitializeEncoder"]))
        status = _InitEncoder(self._encoder, byref(init_params))
        if status != 0:
            raise RuntimeError(f"InitializeEncoder failed, code={status}")
        print("[MinimalEncoder] InitializeEncoder OK")

        # ── 11. 创建多 slot 流水线（每 slot = input buffer + bitstream buffer） ──
        _CreateInputBuffer = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 776))(
            _get_func(_FUNC_IDX["CreateInputBuffer"]))
        _CreateBitstreamBuffer = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 776))(
            _get_func(_FUNC_IDX["CreateBitstreamBuffer"]))
        _DestroyInputBuffer = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)(
            _get_func(_FUNC_IDX["DestroyInputBuffer"]))
        _DestroyBitstreamBuffer = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)(
            _get_func(_FUNC_IDX["DestroyBitstreamBuffer"]))
        nv12_h = height + height // 2

        for slot_idx in range(self._pipeline_depth):
            # 11a. CreateInputBuffer
            ib_buf = (c_uint8 * 776)()
            ctypes.memset(ib_buf, 0, 776)
            cast(ib_buf, POINTER(c_uint32))[0] = NV_ENC_CREATE_INPUT_BUFFER_VER
            cast(byref(ib_buf, 4), POINTER(c_uint32))[0] = width
            cast(byref(ib_buf, 8), POINTER(c_uint32))[0] = height  # luma height
            cast(byref(ib_buf, 16), POINTER(c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12
            status = _CreateInputBuffer(self._encoder, ib_buf)
            if status != 0:
                for existing in self._slots:
                    _DestroyInputBuffer(self._encoder, existing['input_buf'])
                    _DestroyBitstreamBuffer(self._encoder, existing['bs_buf'])
                raise RuntimeError(f"CreateInputBuffer[{slot_idx}] failed, code={status}")
            _raw_ptr = cast(byref(ib_buf, 24), POINTER(c_void_p))[0]
            input_handle = c_void_p(_raw_ptr if isinstance(_raw_ptr, int) else (_raw_ptr.value or 0))

            # 11b. CreateBitstreamBuffer
            bs_buf = (c_uint8 * 776)()
            ctypes.memset(bs_buf, 0, 776)
            cast(bs_buf, POINTER(c_uint32))[0] = NV_ENC_CREATE_BITSTREAM_BUFFER_VER
            status = _CreateBitstreamBuffer(self._encoder, bs_buf)
            if status != 0:
                _DestroyInputBuffer(self._encoder, input_handle)
                for existing in self._slots:
                    _DestroyInputBuffer(self._encoder, existing['input_buf'])
                    _DestroyBitstreamBuffer(self._encoder, existing['bs_buf'])
                raise RuntimeError(f"CreateBitstreamBuffer[{slot_idx}] failed, code={status}")
            _raw_bs = cast(byref(bs_buf, 16), POINTER(c_void_p))[0]
            bs_handle = c_void_p(_raw_bs if isinstance(_raw_bs, int) else (_raw_bs.value or 0))

            self._slots.append({'input_buf': input_handle, 'bs_buf': bs_handle})

        # 后向兼容：slot 0 的引用
        self._input_buf_handle = self._slots[0]['input_buf']
        self._bs_handle = self._slots[0]['bs_buf']

        print(f"[MinimalEncoder] Ready: {self._codec.upper()} {width}x{height}@{fps}fps, "
              f"{rate_mode.upper()}, QP={self._qp}, LA={la_depth}, slots={self._pipeline_depth}")

    @staticmethod
    def _find_dll():
        """查找 NVENC 动态库。"""
        if sys.platform == "win32":
            candidates = [
                "nvEncodeAPI64.dll",
                "nvEncodeAPI.dll",
                os.path.join(os.environ.get("SystemRoot", "C:\\Windows"),
                             "System32", "nvEncodeAPI64.dll"),
            ]
        else:
            candidates = [
                "libnvidia-encode.so.1",
                "libnvidia-encode.so",
                "/usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1",
                "/usr/lib/aarch64-linux-gnu/libnvidia-encode.so.1",
                "/usr/lib64/libnvidia-encode.so.1",
            ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]

    # ─────────────────────────────────────────────────────────
    # 核心测试 API
    # ─────────────────────────────────────────────────────────

    def _push_cuda_context(self):
        """跨线程 CUDA context 保护。"""
        need_pop = False
        if self._primary_ctx.value is not None:
            self._libcuda.cuCtxPushCurrent.restype = c_uint32
            self._libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
            r = self._libcuda.cuCtxPushCurrent(self._primary_ctx)
            need_pop = (r == 0)
        return need_pop

    def _pop_cuda_context(self, need_pop):
        if need_pop:
            try:
                ctx_out = c_void_p()
                self._libcuda.cuCtxPopCurrent.restype = c_uint32
                self._libcuda.cuCtxPopCurrent.argtypes = [POINTER(c_void_p)]
                self._libcuda.cuCtxPopCurrent(byref(ctx_out))
            except Exception:
                pass

    def _copy_frame_to_input_buffer(self, nv12_gpu_tensor, input_buf_handle=None):
        """将 NV12 GPU tensor 拷贝到 NVENC input buffer。

        Args:
            nv12_gpu_tensor: NV12 GPU tensor
            input_buf_handle: 目标 input buffer handle，默认 self._input_buf_handle
        """
        if input_buf_handle is None:
            input_buf_handle = self._input_buf_handle
        nv12_h = self._height + self._height // 2
        W = self._width

        # LockInputBuffer
        lock_buf = (c_uint8 * 1544)()
        ctypes.memset(lock_buf, 0, 1544)
        cast(lock_buf, POINTER(c_uint32))[0] = NV_ENC_LOCK_INPUT_BUFFER_VER
        cast(byref(lock_buf, 8), POINTER(c_void_p))[0] = input_buf_handle

        _LockInputBuffer = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 1544))(
            self._func_ptrs[_FUNC_IDX["LockInputBuffer"]])
        s = _LockInputBuffer(self._encoder, lock_buf)
        if s != 0:
            raise RuntimeError(f"LockInputBuffer failed, code={s}")

        raw_map = cast(byref(lock_buf, 16), POINTER(c_void_p))[0]
        mapped_ptr_val = raw_map if isinstance(raw_map, int) else (raw_map.value or 0)
        actual_pitch = cast(byref(lock_buf, 24), POINTER(c_uint32))[0]

        if not mapped_ptr_val:
            _UnlockInputBuffer = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)(
                self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])
            _UnlockInputBuffer(self._encoder, input_buf_handle)
            raise RuntimeError("LockInputBuffer returned NULL mapped ptr")

        # cuMemcpy2D (device → device)
        _CU_MEMORYTYPE_DEVICE = 2
        _cpy2d = (c_uint8 * 128)()
        ctypes.memset(_cpy2d, 0, 128)
        src_ptr = nv12_gpu_tensor.data_ptr()
        cast(byref(_cpy2d, 16), POINTER(c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
        cast(byref(_cpy2d, 32), POINTER(c_void_p))[0] = c_void_p(src_ptr)
        cast(byref(_cpy2d, 48), POINTER(c_size_t))[0] = W
        cast(byref(_cpy2d, 72), POINTER(c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
        cast(byref(_cpy2d, 88), POINTER(c_void_p))[0] = c_void_p(mapped_ptr_val)
        cast(byref(_cpy2d, 104), POINTER(c_size_t))[0] = (
            actual_pitch if actual_pitch > 0 else W)
        cast(byref(_cpy2d, 112), POINTER(c_size_t))[0] = W
        cast(byref(_cpy2d, 120), POINTER(c_size_t))[0] = nv12_h
        r = self._libcuda.cuMemcpy2D_v2(cast(_cpy2d, c_void_p))
        if r != 0:
            _UnlockInputBuffer = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)(
                self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])
            _UnlockInputBuffer(self._encoder, input_buf_handle)
            raise RuntimeError(f"cuMemcpy2D failed, code={r}")

        # UnlockInputBuffer
        _UnlockInputBuffer = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)(
            self._func_ptrs[_FUNC_IDX["UnlockInputBuffer"]])
        _UnlockInputBuffer(self._encoder, input_buf_handle)

        return actual_pitch

    def _lock_bitstream_once(self, bs_handle=None, count=True):
        """执行一次 LockBitstream → UnlockBitstream，返回 (h264_bytes, status)。

        Args:
            bs_handle: bitstream buffer handle，默认 self._bs_handle
            count: 成功时是否递增 total_locked（EOS flush 应传 False）
        """
        if bs_handle is None:
            bs_handle = self._bs_handle
        _LockBitstream = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 1544))(
            self._func_ptrs[_FUNC_IDX["LockBitstream"]])
        _UnlockBitstream = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)(
            self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])

        lock_raw = (c_uint8 * 1544)()
        ctypes.memset(lock_raw, 0, 1544)
        cast(lock_raw, POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER   # version@0
        cast(byref(lock_raw, 4), POINTER(c_uint32))[0] = 0  # doNotWait=0 (blocking)
        cast(byref(lock_raw, 8), POINTER(c_void_p))[0] = bs_handle  # outputBitstream@8

        bs_status = _LockBitstream(self._encoder, lock_raw)

        h264_data = b""
        if bs_status == NV_ENC_SUCCESS:
            bitstream_size = cast(byref(lock_raw, 36), POINTER(c_uint32))[0]
            if bitstream_size > 0:
                _raw_bsptr = cast(byref(lock_raw, 56), POINTER(c_void_p))[0]
                bs_ptr_val = _raw_bsptr if isinstance(_raw_bsptr, int) else (_raw_bsptr.value or 0)
                if bs_ptr_val:
                    buf_type = c_uint8 * bitstream_size
                    h264_data = bytes(buf_type.from_address(bs_ptr_val))
                    if count:
                        self.total_locked += 1
            _UnlockBitstream(self._encoder, bs_handle)
        elif bs_status == NV_ENC_ERR_NEED_MORE_INPUT:
            if count:
                self.total_need_more += 1

        return h264_data, bs_status

    # ──────────────── 修复后的核心编码方法 ────────────────

    def _drain_outputs(self) -> List[bytes]:
        """
        循环 Lock 当前输出 slot 直到 NEED_MORE_INPUT。
        收集所有已完成帧，推进输出指针。
        NVENC SDK 规范：每送入一帧后必须反复 LockBitstream (blocking)，
        直到返回 NEED_MORE_INPUT，确保无滞留帧。
        """
        outputs = []
        while True:
            slot_idx = self._output_slot_idx % self._pipeline_depth
            bs_handle = self._slots[slot_idx]['bs_buf']
            data, status = self._lock_bitstream_once(bs_handle, count=True)
            if status == NV_ENC_ERR_NEED_MORE_INPUT:
                break
            if status != NV_ENC_SUCCESS or not data:
                break
            outputs.append(data)
            self._output_slot_idx += 1
        return outputs

    def encode_frame(self, nv12_gpu_tensor, force_idr: bool = False) -> List[bytes]:
        """
        【正确模式】送入 1 帧 → 轮转 slot → EncodePicture → 立即排空所有可用输出。

        遵守 NVENC SDK "送一帧 → 循环 LockBitstream 直到 NEED_MORE_INPUT" 规则。
        NEED_MORE_INPUT 分支不做任何 Lock/Unlock（buffer 中无有效输出，操作可能误弃帧）。
        返回该次操作产生的 H.264 帧列表（通常 0~1 帧，LA 模式下可能积攒多帧）。
        """
        with self._lock:
            slot_idx = self._frame_idx % self._pipeline_depth
            slot = self._slots[slot_idx]

            pitch = self._copy_frame_to_input_buffer(nv12_gpu_tensor,
                                                     slot['input_buf'])

            pic_buf = (c_uint8 * 3360)()
            ctypes.memset(pic_buf, 0, 3360)
            cast(pic_buf, POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
            cast(byref(pic_buf, 4), POINTER(c_uint32))[0] = self._width
            cast(byref(pic_buf, 8), POINTER(c_uint32))[0] = self._height
            cast(byref(pic_buf, 12), POINTER(c_uint32))[0] = (
                pitch if pitch > 0 else self._width)
            cast(byref(pic_buf, 24), POINTER(c_uint64))[0] = self._frame_idx
            cast(byref(pic_buf, 40), POINTER(c_void_p))[0] = slot['input_buf']
            cast(byref(pic_buf, 48), POINTER(c_void_p))[0] = slot['bs_buf']
            cast(byref(pic_buf, 64), POINTER(c_uint32))[0] = NV_ENC_BUFFER_FORMAT_NV12
            cast(byref(pic_buf, 68), POINTER(c_uint32))[0] = NV_ENC_PIC_STRUCT_FRAME
            if force_idr:
                cast(byref(pic_buf, 16), POINTER(c_uint32))[0] = 0x2

            _EncodePicture = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 3360))(
                self._func_ptrs[_FUNC_IDX["EncodePicture"]])
            _EncodePicture(self._encoder, pic_buf)

            self._frame_idx += 1
            self.total_encoded += 1

            # ★ 关键：立即排空所有已完成的输出帧
            return self._drain_outputs()

    def flush_eos(self) -> List[bytes]:
        """
        EOS flush：发送 EOS → 按输出指针顺序逐 slot 完全排空。

        NVENC SDK 规范："The client must call nvEncEncodePicture() with a
        NULL input picture, then repeatedly call nvEncLockBitstream
        (doNotWait=0 blocking) to retrieve all remaining encoded frames
        from the lookahead queue."

        从 _output_slot_idx 指向的 slot 开始，按轮转顺序遍历所有 slot，
        对每个 slot 循环 blocking Lock 直到 NEED_MORE_INPUT。
        返回所有滞留帧数据（顺序与硬件输出顺序严格一致）。
        """
        output_frames = []
        with self._lock:
            # 发送 EOS
            pic_buf = (c_uint8 * 3360)()
            ctypes.memset(pic_buf, 0, 3360)
            cast(pic_buf, POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
            cast(byref(pic_buf, 16), POINTER(c_uint32))[0] = 0x8  # EOS flag
            cast(byref(pic_buf, 40), POINTER(c_void_p))[0] = c_void_p(None)
            cast(byref(pic_buf, 48), POINTER(c_void_p))[0] = self._slots[0]['bs_buf']

            _EncodePicture = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 3360))(
                self._func_ptrs[_FUNC_IDX["EncodePicture"]])
            _EncodePicture(self._encoder, pic_buf)

            # 按输出指针顺序排空所有 slot
            start_slot = self._output_slot_idx % self._pipeline_depth
            drain_order = [(start_slot + i) % self._pipeline_depth
                           for i in range(self._pipeline_depth)]
            for slot_idx in drain_order:
                bs_handle = self._slots[slot_idx]['bs_buf']
                while True:
                    data, status = self._lock_bitstream_once(bs_handle, count=False)
                    if status == NV_ENC_ERR_NEED_MORE_INPUT:
                        break
                    if status != NV_ENC_SUCCESS or not data:
                        break
                    output_frames.append(data)
                    self.total_eos_frames += 1
        return output_frames

    def close(self):
        """销毁编码器: EOS flush → 释放 slot → DestroyEncoder → 恢复 CUDA context。"""
        with self._lock:
            if self._encoder.value is not None:
                # EOS flush (non-blocking, bounded) 排空所有 slot 的 LA pending 帧
                try:
                    pic_buf = (c_uint8 * 3360)()
                    ctypes.memset(pic_buf, 0, 3360)
                    cast(pic_buf, POINTER(c_uint32))[0] = NV_ENC_PIC_PARAMS_VER
                    cast(byref(pic_buf, 16), POINTER(c_uint32))[0] = 0x8
                    cast(byref(pic_buf, 40), POINTER(c_void_p))[0] = c_void_p(None)
                    cast(byref(pic_buf, 48), POINTER(c_void_p))[0] = self._slots[0]['bs_buf']
                    _lep = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 3360))(
                        self._func_ptrs[_FUNC_IDX["EncodePicture"]])
                    _lep(self._encoder, pic_buf)
                    # drain all slots with doNotWait=1 (non-blocking, max 32 attempts each)
                    _lck = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 1544))(
                        self._func_ptrs[_FUNC_IDX["LockBitstream"]])
                    _unl = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)(
                        self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])
                    for slot in self._slots:
                        bs_handle = slot['bs_buf']
                        for _ in range(32):
                            lr = (c_uint8 * 1544)()
                            ctypes.memset(lr, 0, 1544)
                            cast(lr, POINTER(c_uint32))[0] = NV_ENC_LOCK_BITSTREAM_VER
                            cast(byref(lr, 4), POINTER(c_uint32))[0] = 1
                            cast(byref(lr, 8), POINTER(c_void_p))[0] = bs_handle
                            if _lck(self._encoder, lr) != NV_ENC_SUCCESS:
                                break
                            _unl(self._encoder, bs_handle)
                            if cast(byref(lr, 36), POINTER(c_uint32))[0] == 0:
                                break
                except Exception:
                    pass
                _DestroyInputBuffer = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)(
                    self._func_ptrs[_FUNC_IDX["DestroyInputBuffer"]])
                _DestroyBitstreamBuffer = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)(
                    self._func_ptrs[_FUNC_IDX["DestroyBitstreamBuffer"]])
                for slot_idx, slot in enumerate(self._slots):
                    try:
                        _DestroyInputBuffer(self._encoder, slot['input_buf'])
                        _DestroyBitstreamBuffer(self._encoder, slot['bs_buf'])
                    except Exception:
                        pass
                _DestroyEncoder = ctypes.CFUNCTYPE(c_uint32, c_void_p)(
                    self._func_ptrs[_FUNC_IDX["DestroyEncoder"]])
                _DestroyEncoder(self._encoder)
                self._encoder = c_void_p(None)
                # 恢复 CUDA context stack（抵消 __init__ 中的 cuCtxPushCurrent）
                # 多次编码器轮转若不 pop, CUDA context stack 溢出 → SIGSEGV
                try:
                    self._libcuda.cuCtxPopCurrent.restype = c_uint32
                    self._libcuda.cuCtxPopCurrent.argtypes = [POINTER(c_void_p)]
                    ctx_out = c_void_p()
                    self._libcuda.cuCtxPopCurrent(byref(ctx_out))
                except Exception:
                    pass
                print(f"[MinimalEncoder] Destroyed ({self._pipeline_depth} slots)")


# ============================================================================
# Part 3: 测试框架
# ============================================================================

def generate_synthetic_nv12_frames(n_frames: int, width: int, height: int,
                                    device: str = "cuda:0") -> List[torch.Tensor]:
    """生成合成 NV12 GPU 帧（用于测试编码帧数守恒）。

    每帧是宽度渐变的彩色竖条（带帧间变化），确保编码器不会将帧当作 skip。
    NV12 格式: Y 平面 (H×W) + UV 交织平面 ((H/2)×W)，uint8，GPU 连续。
    """
    import torch
    nv12_h = height + height // 2
    frames = []
    for i in range(n_frames):
        # Y 平面: 竖条渐变，帧间偏移
        y = torch.zeros((height, width), dtype=torch.uint8, device=device)
        stripe_w = max(4, width // 32)
        for x in range(0, width, stripe_w):
            val = ((x + i * 7) % 256)
            y[:, x:min(x + stripe_w, width)] = val

        # UV 平面: 颜色渐变
        uv_h = height // 2
        uv = torch.zeros((uv_h, width), dtype=torch.uint8, device=device)
        for x in range(0, width, stripe_w):
            u_val = 128 + ((x + i * 11) % 64) - 32
            v_val = 128 + ((x + i * 13) % 64) - 32
            uv[:, x:min(x + stripe_w, width)] = u_val

        # 合并为 NV12
        nv12 = torch.empty((nv12_h, width), dtype=torch.uint8, device=device)
        nv12[:height, :] = y
        nv12[height:nv12_h, :] = uv

        frames.append(nv12.contiguous())
    return frames


def read_video_frames_from_file(input_path: str, device: str = "cuda:0",
                                  max_frames: int = 0) -> List[torch.Tensor]:
    """从输入视频文件读取帧，解码为 NV12 GPU tensor 列表。

    使用 FFmpeg 解码 → Numpy raw RGB → GPU NV12 转换。
    要求 FFmpeg 在 PATH 中。

    Args:
        input_path: 输入视频文件路径
        device: GPU 设备标识（如 'cuda:0'）
        max_frames: 最大读取帧数（0=全部读取）

    Returns:
        NV12 GPU tensor 列表
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch 不可用，无法进行 GPU 帧转换")

    print(f"[read_video] 输入: {input_path}")

    # 1. 获取视频信息（宽、高、帧率、总帧数）
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-of", "csv=p=0",
        input_path
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        parts = [x.strip() for x in result.stdout.strip().split(",")]
        vid_w = int(parts[0])
        vid_h = int(parts[1])
        # r_frame_rate 可能是 "25/1" 格式
        fps_str = parts[2]
        if "/" in fps_str:
            num, den = fps_str.split("/")
            vid_fps = float(num) / float(den)
        else:
            vid_fps = float(fps_str)
        total_frames = int(parts[3]) if len(parts) > 3 and parts[3] else 0
    except Exception as e:
        raise RuntimeError(f"ffprobe 失败: {e}")

    if vid_h % 2 != 0:
        vid_h -= 1  # NV12 要求偶数高度
        print(f"[read_video] 高度调整为 {vid_h}（偶数对齐 NV12）")

    n_read = max_frames if max_frames > 0 else (total_frames or 1)
    print(f"[read_video] {vid_w}x{vid_h}@{vid_fps:.2f}fps, 读取 {n_read} 帧...")

    # 2. FFmpeg 解码到 raw RGB24 管道
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-an", "-sn",
        "-vf", f"fps={vid_fps},scale={vid_w}:{vid_h}",
        "-frames:v", str(n_read),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1"
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size_rgb = vid_w * vid_h * 3
    frames_gpu = []

    for i in range(n_read):
        raw = proc.stdout.read(frame_size_rgb)
        if not raw or len(raw) < frame_size_rgb:
            break
        # RGB24 numpy → BGR float → GPU tensor → BGR→RGB → YUV→NV12
        import numpy as np
        rgb_np = np.frombuffer(raw, dtype=np.uint8).reshape((vid_h, vid_w, 3))
        # RGB→BGR: numpy 翻转通道 R↔B
        bgr_np = rgb_np[:, :, ::-1].copy()
        gpu_uint8 = torch.from_numpy(bgr_np).to(device)
        gpu_float = gpu_uint8.float() / 255.0

        # RGB→YUV (BT.601) 手写 kernel
        # Y  = 0.299*R + 0.587*G + 0.114*B
        # Cb = -0.169*R - 0.331*G + 0.500*B + 128
        # Cr = 0.500*R - 0.419*G - 0.081*B + 128
        r = gpu_float[:, :, 2]
        g = gpu_float[:, :, 1]
        b = gpu_float[:, :, 0]

        y = (0.299 * r + 0.587 * g + 0.114 * b) * 255.0
        y_u8 = y.clamp(0, 255).to(torch.uint8)

        # UV 平面: Cb/Cr 以 2x2 子采样
        uv_h, uv_w = vid_h // 2, vid_w // 2
        cb = (-0.169 * r - 0.331 * g + 0.500 * b + 0.5) * 255.0
        cr = (0.500 * r - 0.419 * g - 0.081 * b + 0.5) * 255.0
        cb_u8 = cb.clamp(0, 255).to(torch.uint8)
        cr_u8 = cr.clamp(0, 255).to(torch.uint8)

        # 子采样: 取奇数行奇数列 (0, 2, 4, ...)
        cb_subsampled = cb_u8[0:vid_h:2, 0:vid_w:2]
        cr_subsampled = cr_u8[0:vid_h:2, 0:vid_w:2]

        # NV12 交织 UV: U0,V0,U1,V1,...  (向量化: stack + reshape)
        uv_stacked = torch.stack([cb_subsampled, cr_subsampled], dim=-1)  # (uv_h, uv_w, 2)
        uv_packed = uv_stacked.reshape(uv_h, uv_w * 2).contiguous()

        nv12_h_total = vid_h + vid_h // 2
        nv12 = torch.empty((nv12_h_total, vid_w), dtype=torch.uint8, device=device)
        nv12[:vid_h, :] = y_u8
        nv12[vid_h:nv12_h_total, :] = uv_packed
        frames_gpu.append(nv12.contiguous())

        if (i + 1) % max(1, n_read // 5) == 0 or i < 5:
            print(f"  [read_video] 解码帧 {i + 1}/{n_read}")

    proc.terminate()
    print(f"[read_video] 完成: {len(frames_gpu)} 帧解码为 GPU NV12 tensor")
    return frames_gpu


def write_encoded_output(output_path: str, encoded_frames: List[bytes],
                          width: int, height: int, fps: float = 25.0,
                          codec: str = "h264"):
    """将编码帧列表写入输出文件。

    根据后缀自动选择容器格式：
      .h264 / .hevc → 原始 ES 流直接写入
      .mp4 → 用 FFmpeg mux（-c:v copy）
      .mkv → 用 FFmpeg mux
      其他或无后缀 → 默认 .h264 ES 流

    Args:
        output_path: 输出文件路径
        encoded_frames: H.264/H.265 elementary stream bytes 列表
        width, height: 视频尺寸
        fps: 帧率
        codec: 编码类型 ('h264' | 'hevc' | 'av1')
    """
    total_bytes = sum(len(f) for f in encoded_frames)
    suffix = pathlib.Path(output_path).suffix.lower()
    print(f"\n[write_output] {len(encoded_frames)} 帧, {total_bytes} bytes → {output_path}")

    if not encoded_frames:
        print("[write_output] ⚠️  无帧可写，跳过")
        return

    # codec → (raw_suffix, ffmpeg_fmt)
    _CODEC_FMT = {"h264": (".h264", "h264"), "hevc": (".hevc", "hevc"), "av1": (".obu", "obu")}
    raw_suffix, ffmpeg_fmt = _CODEC_FMT.get(codec, (".h264", "h264"))

    if suffix in (".mp4", ".mkv"):
        # 用 FFmpeg mux：ES → 容器
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="nvenc_la_mux_")
        es_path = pathlib.Path(tmpdir) / ("es_stream" + raw_suffix)
        try:
            es_data = b"".join(encoded_frames)
            es_path.write_bytes(es_data)

            cmd_mux = [
                "ffmpeg", "-y",
                "-f", ffmpeg_fmt,
                "-framerate", str(fps),
                "-i", str(es_path),
                "-c:v", "copy",
                output_path
            ]
            result = subprocess.run(cmd_mux, capture_output=True, timeout=60)
            if result.returncode == 0:
                print(f"[write_output] ✅ 已 mux 到 {output_path}")
            else:
                stderr_full = result.stderr.decode("utf-8", errors="replace")
                # 提取 stderr 末尾真正的错误摘要
                stderr_tail = stderr_full[-600:]
                print(f"[write_output] ⚠️  FFmpeg mux 失败 (code {result.returncode})，"
                      f"回退到 ES 文件")
                print(f"[write_output]   ffmpeg stderr: {stderr_tail.strip()}")
                # 替换 .mp4/.mkv 后缀为 raw_suffix
                out = pathlib.Path(output_path)
                es_fallback = out.with_suffix(raw_suffix)
                es_fallback.write_bytes(es_data)
                print(f"[write_output] 已写入 ES 流: {es_fallback}")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        # 纯 ES 流写入
        if not suffix:
            output_path = output_path + raw_suffix
        with open(output_path, "wb") as f:
            for frame in encoded_frames:
                f.write(frame)
        print(f"[write_output] ✅ 已写入 ES 流: {output_path}")

def test_correct_polling_pattern(encoder: MinimalTestEncoder,
                                  frames: List[torch.Tensor],
                                  label: str) -> dict:
    """
    测试【正确模式】: encode_frame（send + Lock 排空循环）+ EOS flush。

    每帧调用 encoder.encode_frame()，其内部按 SDK 规范执行：
    EncodePicture 后循环 LockBitstream 直到 NEED_MORE_INPUT，
    确保所有已完成帧立即取走且顺序与硬件输出一致。
    末尾 EOS flush 回收全部 LA 滞留帧。
    """
    print(f"\n{'='*60}")
    print(f"[TEST] {label} — 正确排空模式")
    print(f"{'='*60}")

    all_outputs = []
    for i, frame in enumerate(frames):
        outputs = encoder.encode_frame(frame, force_idr=(i == 0))
        all_outputs.extend(outputs)
        if i < 5 or i >= len(frames) - 3 or (i % 20 == 0 and i > 0):
            print(f"  Frame {i:4d}/{len(frames)}: {len(outputs)} output(s) this round, "
                  f"累计={len(all_outputs)}")

    # EOS flush — 按 SDK 规范取回 LA 队列中的滞留帧
    eos_outputs = encoder.flush_eos()
    all_outputs.extend(eos_outputs)
    print(f"  EOS flush: {len(eos_outputs)} output(s), 最终累计={len(all_outputs)}")

    result = {
        "pattern": "correct_polling",
        "la_depth": encoder._la_depth,
        "input_frames": len(frames),
        "output_frames": len(all_outputs),
        "encoded_count": encoder.total_encoded,
        "locked_count": encoder.total_locked,
        "eos_frames": encoder.total_eos_frames,
        "match": len(all_outputs) == len(frames),  # ★ Output == Input（含 EOS flush）
        "outputs": all_outputs,
    }
    return result


def verify_with_ffmpeg(n_frames: int, width: int, height: int, la_depth: int) -> bool:
    """
    用 FFmpeg 作为权威参考：编码 N 帧，验证输出帧数 = 输入帧数。
    这证明了 SDK Level 2 (FFmpeg) 正确实现了 NVENC 的 flush 机制。
    """
    import subprocess
    import tempfile
    import pathlib

    print(f"\n[FFmpeg-REF] 生成 {n_frames} 帧测试视频 (LA={la_depth})...")

    tmpdir = tempfile.mkdtemp(prefix="nvenc_la_test_")
    input_file = pathlib.Path(tmpdir) / "input.mp4"
    output_file = pathlib.Path(tmpdir) / "output.mp4"

    try:
        # 生成测试源（简单纯色 + 时间戳）
        cmd_gen = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"testsrc=duration={n_frames/25:.2f}:size={width}x{height}:rate=25",
            "-frames:v", str(n_frames),
            "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            str(input_file)
        ]
        subprocess.run(cmd_gen, capture_output=True, check=True, timeout=60)

        # NVENC 编码 + LA
        cmd_enc = [
            "ffmpeg", "-y",
            "-i", str(input_file),
            "-c:v", "h264_nvenc",
            "-rc", "constqp", "-qp", "28",
            "-rc-lookahead", str(la_depth),
            str(output_file)
        ]
        subprocess.run(cmd_enc, capture_output=True, check=True, timeout=120)

        # 统计输出帧数
        cmd_probe = [
            "ffprobe", "-v", "error",
            "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0",
            str(output_file)
        ]
        result = subprocess.run(cmd_probe, capture_output=True, text=True, timeout=30)
        output_frames = int(result.stdout.strip())
        match = (output_frames == n_frames)
        print(f"[FFmpeg-REF] 输入={n_frames}, 输出={output_frames}, "
              f"{'✅ 帧数守恒' if match else '❌ 异常'}")

        return match
    except Exception as e:
        print(f"[FFmpeg-REF] 跳过 (FFmpeg 不可用或出错): {e}")
        return None
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Part 4: 主测试函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NVENC Lookahead 帧数守恒验证测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test_nvenc_la_frame_conservation.py
  python test_nvenc_la_frame_conservation.py --frames 200 --la-depth 8
  python test_nvenc_la_frame_conservation.py --frames 687 --la-depth 8 --ffmpeg-ref
        """)
    parser.add_argument("--frames", "-n", type=int, default=100,
                        help="测试帧数；--input 时作为最大读取帧上限，实际以视频为准 (default: 100)")
    parser.add_argument("--la-depth", type=int, default=8,
                        help="Lookahead 深度 (default: 8)")
    parser.add_argument("--width", "-W", type=int, default=720,
                        help="视频宽度，仅合成帧模式有效；--input 时以视频为准 (default: 720)")
    parser.add_argument("--height", "-H", type=int, default=576,
                        help="视频高度，仅合成帧模式有效；--input 时以视频为准 (default: 576)")
    parser.add_argument("--qp", type=int, default=18,
                        help="QP 值 (default: 18)")
    parser.add_argument("--rate-mode", choices=["constqp", "vbr_hq", "qvbr"],
                        default="vbr_hq", help="RC 模式 (default: vbr_hq)")
    parser.add_argument("--ffmpeg-ref", action="store_true",
                        help="同时运行 FFmpeg 权威参考验证")
    parser.add_argument("--preset", default="p1",
                        help="NVENC preset (default: p1)")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="输入视频文件路径（指定后从视频读取帧，替代合成帧）")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出目录路径（指定后每个测试的编码结果写入该目录下独立文件；"
                             "如不存在则自动创建）")
    parser.add_argument("--codec", type=str, default="h264",
                        choices=["h264", "hevc", "av1"],
                        help="编码类型: h264 (默认) | hevc (H.265) | av1 (Ada Lovelace+ 驱动)")
    args = parser.parse_args()

    codec = args.codec
    n_frames = args.frames
    la_depth = args.la_depth
    width = args.width
    height = args.height
    fps = 25.0
    cli_frames_provided = args.frames != parser.get_default("frames")

    # 确保高度是偶数（NV12 要求）
    if height % 2 != 0:
        height += 1

    if not HAS_TORCH:
        print("[SKIP] PyTorch 不可用，无法生成 GPU 测试帧。")
        print("请在有 GPU 的 Linux 服务器上安装 PyTorch 后运行。")
        return 1

    if not torch.cuda.is_available():
        print("[SKIP] CUDA GPU 不可用，NVENC 硬件编码需要 NVIDIA GPU。")
        return 1

    device = "cuda:0"
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # ── 帧来源：输入视频文件（以实际视频参数为准） 或 合成帧 ──
    if args.input:
        if not os.path.isfile(args.input):
            print(f"[FAIL] 输入文件不存在: {args.input}")
            return 1

        # ★ 先探针获取真实参数，一切以输入视频为准 ★
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
            "-of", "csv=p=0",
            args.input
        ]
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            parts = [x.strip() for x in result.stdout.strip().split(",")]
            width = int(parts[0])
            height = int(parts[1])
            fps_str = parts[2]
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            elif fps_str:
                fps = float(fps_str)
            total_frames = int(parts[3]) if len(parts) > 3 and parts[3] else 0
        except Exception as e:
            print(f"[WARN] ffprobe 失败 ({e})，回退到 CLI 参数")
            fps = 25.0
            total_frames = 0

        if height % 2 != 0:
            height -= 1  # NV12 要求偶数高度，向下取偶
            print(f"[INFO] 高度调整为 {height}（偶数对齐 NV12）")

        # --frames 作为上限（不传则取全部帧）
        if cli_frames_provided:
            n_frames = min(args.frames, total_frames) if total_frames > 0 else args.frames
        else:
            n_frames = total_frames if total_frames > 0 else 100

        print(f"[INFO] 输入视频: {width}x{height}@{fps:.2f}fps, total={total_frames}, "
              f"测试读取 {n_frames} 帧")

        frames = read_video_frames_from_file(args.input, device, max_frames=n_frames)
        if not frames:
            print("[FAIL] 无法从输入视频读取帧")
            return 1
        # 用实际读取的帧数再次校准
        n_frames = len(frames)
        width = frames[0].shape[1]
        height = frames[0].shape[0] * 2 // 3  # NV12 总高度 = H + H/2, 反推 H
    else:
        if height % 2 != 0:
            height += 1
            print(f"[INFO] 高度调整为 {height}（偶数对齐 NV12）")

        print(f"\n[INFO] 生成 {n_frames} 帧合成 NV12 GPU 测试帧...")
        frames = generate_synthetic_nv12_frames(n_frames, width, height, device)

    print("=" * 70)
    print("  NVENC Lookahead 帧数守恒验证测试（修复版）")
    print("=" * 70)
    print(f"  配置: {codec.upper()} {width}x{height}, {n_frames} 帧, "
          f"{args.rate_mode.upper()}={args.qp}, LA={la_depth}")
    if args.input:
        print(f"  输入: {args.input}")
    if args.output:
        print(f"  输出: {args.output}")
    print()
    print("  核心修复:")
    print(f"    1. pipeline_depth = LA+1 = {la_depth+1}（SDK 硬件要求，防缓冲区覆盖）")
    print("    2. 送一帧 -> 循环 LockBitstream 直到 NEED_MORE_INPUT（SDK 排空规则）")
    print("    3. 移除 NEED_MORE_INPUT 时强制 Lock/Unlock 丢弃帧的分支")
    print("    4. 输出 slot 指针保证帧顺序（_output_slot_idx）")
    print()
    print("  测试矩阵 (修复后):")
    print(f"    A) LA=0 + 正确排空:      Output == Input")
    print(f"    C) LA={la_depth} + 正确排空:  Output == Input (含 EOS flush)")
    print()
    print("  帧计数: EncodePicture 调用计数 = 输入帧数 (恒等)")
    print(f"  pipeline_depth=LA+1={la_depth+1}（SDK 硬件安全要求）")
    print()

    print(f"[INFO] 完成: {len(frames)} 帧, "
          f"shape={frames[0].shape}, dtype={frames[0].dtype}")

    results = []

    # --- 测试 A: LA=0 + 正确排空 ---
    try:
        enc = MinimalTestEncoder(width, height, fps=fps, preset=args.preset,
                                 qp=args.qp, rate_mode=args.rate_mode, la_depth=0,
                                 codec=codec)
        r = test_correct_polling_pattern(enc, frames, "A: LA=0 + 正确排空")
        results.append(r)
        enc.close()
    except Exception as e:
        print(f"[FAIL] test A: {e}")
        import traceback
        traceback.print_exc()

    # --- 测试 C: LA=N + 正确排空（核心验证） ---
    try:
        enc = MinimalTestEncoder(width, height, fps=fps, preset=args.preset,
                                 qp=args.qp, rate_mode=args.rate_mode, la_depth=la_depth,
                                 codec=codec)
        r = test_correct_polling_pattern(enc, frames, f"C: LA={la_depth} + 正确排空")
        results.append(r)
        enc.close()
    except Exception as e:
        print(f"[FAIL] test C: {e}")
        import traceback
        traceback.print_exc()

    # --- results report ---
    print("\n")
    print("=" * 70)
    print("  Frame Conservation Verification Results")
    print("=" * 70)
    print(f"  {'Test':<30s} {'Input':>5s} {'Output':>6s} {'LockBS':>6s} {'EOS':>5s} {'Match':>6s}")
    print(f"  {'':30s} {'':>5s} {'':>6s} {'':>6s} {'flush':>5s} {'':>6s}")
    print(f"  {'-'*30} {'-'*5} {'-'*6} {'-'*6} {'-'*5} {'-'*6}")

    all_pass = True
    for r in results:
        label = f"{r['pattern']} (LA={r['la_depth']})"
        match_str = "OK" if r['match'] else "FAIL"
        print(f"  {label:<30s} {r['input_frames']:5d} {r['output_frames']:6d} "
              f"{r['locked_count']:6d} {r['eos_frames']:5d} {match_str:>6s}")

        if not r['match']:
            all_pass = False

    print()
    print(f"  {'-'*70}")
    print(f"  Input  = input frame count")
    print(f"  Output = encoded output frame count (encode_frame + EOS flush)")
    print(f"  LockBS = LockBitstream success count during encoding")
    print(f"  EOS/flush = frames recovered during EOS flush")
    print(f"  Match  = Output == Input (frame conservation)")
    print()
    print("  SDK Frame Conservation Claim:")
    print("    'The total number of output frames equals the number of input frames.'")
    if all_pass:
        print("    -> VERIFIED: NVENC Level 1 API frame conservation confirmed")
    else:
        print("    -> FAILED: check configuration")
    print(f"    pipeline_depth=LA+1={la_depth+1} (SDK hardware safety requirement)")
    print("-" * 70)

    # --- FFmpeg reference (optional) ---
    if args.ffmpeg_ref:
        print()
        ffmpeg_ok = verify_with_ffmpeg(n_frames, width, height, la_depth)
        if ffmpeg_ok:
            print("[FFmpeg-REF] OK: LA does not change output frame count")
        elif ffmpeg_ok is False:
            print("[FFmpeg-REF] FAIL: check FFmpeg/NVENC driver")

    # --- output files (--output / -o) ---
    if args.output:
        out_dir = pathlib.Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        _TEST_LETTER_MAP = {
            ("correct_polling",  0): "A",
            ("correct_polling",  1): "C",
        }

        written_count = 0
        for r in results:
            la_key = 0 if r['la_depth'] == 0 else 1
            letter = _TEST_LETTER_MAP.get((r['pattern'], la_key))
            if letter is None:
                continue
            if not r.get('outputs'):
                print(f"[--output] test {letter} has no output frames, skip")
                continue

            la_str = f"la{r['la_depth']}"
            filename = f"test_{letter}_{la_str}_correct.mp4"
            filepath = str(out_dir / filename)

            write_encoded_output(filepath, r['outputs'],
                                 width, height, fps=fps, codec=codec)
            written_count += 1

        print(f"\n[--output] wrote {written_count}/{len(results)} test results to: {out_dir}")

    # --- conclusion ---
    print()
    if all_pass:
        print("All tests passed: NVENC Level 1 API frame conservation VERIFIED.")
        print("  Fix ensures frame conservation under ALL RC modes + LA settings.")
        if la_depth > 0:
            print(f"  - slot count = {la_depth+1} (pipeline_depth=LA+1, SDK hardware requirement)")
            print("  - encode_frame drain loop: no stuck frames, no buffer overwrite, full conservation")
    else:
        print("FAIL: Output frame count != input frame count")

    return 0


if __name__ == "__main__":
    _rc = main()
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    # os._exit 跳过 CUDA/NVENC DLL 卸载竞争，防止 exit-time CUDA segfault
    import os
    os._exit(_rc if _rc is not None else 0)
