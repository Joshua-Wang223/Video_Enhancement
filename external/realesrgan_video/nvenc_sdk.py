#!/usr/bin/env python3
"""
NVENC SDK Level 1 编码模块 (ctypes CUDA/NVENC SDK 13.0)

从 IFRNet v6.4.5.1 提取，供 Real-ESRGAN 复用。

  [FIX-ASYNC-COPY]  NVENC 输入拷贝原用同步 cuMemcpy2D_v2（无 stream 参数，
                     走 legacy default/null stream）。该 stream 在同一 CUDA
                     context 下具有隐式全局同步语义——每次拷贝都强制等待
                     Real-ESRGAN 管线的 transfer_stream/sr_stream/
                     gfpgan_stream 排空，反之亦然，导致编码与 SR 推理无法
                     真正并行，GPU 利用率周期性骤降。改用 cuMemcpy2DAsync_v2
                     + 专用非默认 CUDA stream(self._stream_encode)，只同步
                     这一个私有 stream，不再对其他 stream 施加隐式屏障。
                     三处 NVENC 输入拷贝调用点统一改为调用新增的
                     _copy_into_input_buffer() 辅助方法；专用 stream 创建
                     失败时自动回退到原同步拷贝，不影响功能正确性。

  [FIX-FLUSH-GRANULARITY]  原 _LA_CHUNK_SIZE 按内存预算动态计算但上限高达
                     500 帧，编码线程会把多达 500 帧的 Lock/Copy/Unlock/
                     Encode/LockBitstream 压缩成一次持续较长的连续同步脉冲。
                     期间编码线程输入队列迅速被 Writer 写满 → Writer 阻塞
                     → 无法消费上游 result_queue → 推理线程被阻塞产帧 →
                     GPU 利用率骤降（队列反压，与 CUDA stream 选择无关，
                     是独立于上一条的第二个根因）。新增 flush_chunk_frames
                     参数（默认 128）把触发编码的粒度调小，摊薄成多次小
                     脉冲；同时移植 LA>0 时队列深度自动提升至 16（LA>0
                     队列中的数据已在 pinned host 内存，不占 VRAM，可安全
                     加深）。

组件:
  - NVENCEncoder: GPU 直接 H.264 硬件编码器 (CE-pipeline, LA 支持)
  - _NVENCEncodeThread: daemon 编码线程 (异步编码 + 空帧防御)
  - _rgb_to_nv12_gpu / _rgb_to_nv12_gpu_batch: GPU RGB→NV12 色彩空间转换
  - FFmpegMuxer: H.264 ES → MP4 纯 muxer (-c:v copy)

跨段复用:
  - NVENCEncoder Python 对象 + 驱动会话跨段持续复用（缓存在
    enhancer['_sdk_nvenc_encoder']），段边界不销毁重建。
    [FIX-SKIP-REOPEN] 跳过 reopen()，对齐 IFRNet v6.4.5.1 生产零崩溃验证。
  - _frame_idx/_output_slot_idx 跨段单调连续递增，不归零。
    _stream_begin(force=True) 每段重置 _slot_pending/_slots_warmed/
    _prev_chunk_outputs（对齐 IFRNet 同名方法）。
  - Phase D 移除 FIX-SEG-START-SYNC：段首 batch 直接走 CE pipeline，
    无需同步 encode_frame 过渡。
    配套诊断/加固：[DIAG-SEGFAULT] main.py 启用 faulthandler（崩溃时
    dump 全线程栈）；[FIX-SEG2-PREFLIGHT] 编码线程启动预检 context/
    会话句柄；[FIX-RC-CHECK] ce_pipeline/flush 的 libcuda 与 EOS 调用
    返回码检查（把"被污染的 context"从不可归因 SIGSEGV 转化为带上下文
    的异常）；[FIX-REOPEN-CTX-ORDER] close() 先 push primary
    ctx 再销毁（原顺序下销毁调用在缺 context 线程上会静默失败 → 驱动
    资源泄漏）；[FIX-SYNC-BEFORE-SUBMIT] 活跃 NVENCWriter 提交前补
    默认 stream 同步（此前只补在未使用的 nvenc_writer.py 副本）；
    [EARLY-FLUSH] 活跃 NVENCWriter 补 begin_flush() 公共包装；
    环境变量 ESRGAN_DISABLE_SDK_NVENC=1 可整体回退 FFmpeg 编码路径。
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
import torch
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

# [DIAG-LEVEL] H.264 profile_idc → 名称映射（日志诊断 NVDEC 软解回退用）
_H264_PROFILE_NAMES = {66: "Baseline", 77: "Main", 88: "Extended",
                       100: "High", 110: "High10", 122: "High422", 244: "High444"}
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
# 当前实际行为（与代码 line 867/880/1046 一致）：targetQuality = max(1, CRF)
# 直接映射（CRF=18 → tq=18；CRF=23 → tq=23），并非 51-CRF 反置。
#   · 值越小质量越高（1=best），越大质量越低（51=worst）
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

# ── _NVENC_QVBR_ENABLE_VBV ──
# QVBR 模式是否启用 VBV 缓冲合规约束（vbvBufferSize / vbvInitialDelay）。
#   True  — 启用：设置 4MB buffer + 2MB initial delay，保证实时流合规，
#           但每帧编码后需更新 VBV 水位 + QP 范围约束，增加 ~5% 延迟。
#   False — 关闭：跳过 VBV 设置，编码器不追踪虚拟解码器缓冲，离线编码
#           无合规需求，质量无明显差异。
_NVENC_QVBR_ENABLE_VBV: bool = False

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
        self._la_depth = la_depth  # 在 constqp/CRF0 调整后重新赋值，确保实例属性与最终值一致
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
        # [FIX-ASYNC-COPY] 专用非默认 CUDA stream，用于 NVENC 输入拷贝
        # (cuMemcpy2DAsync)。提前声明为 None，防止初始化中途失败时
        # close() 访问未定义属性。移植自 IFRNet v6.4.5.1。
        self._stream_encode = c_void_p(None)

        # 多 slot 缓冲池: _slot_count >= _required_buffers (SDK 硬件安全要求)
        self._slot_count = _slot_count
        # [FIX-SLOT-DEQUE] _slot_pending 从固定大小 list 改为 dict[int, deque]，
        # 每个 slot 可持有多个待处理条目（FIFO）。新帧追加而非覆盖旧条目，
        # 避免 drain 延迟时跨 chunk slot 轮转覆盖导致的帧乱序/跳跃/重复。
        # deque 元素格式: (global_fi, bs_buf, force_idr, ep_status) — encode_frames_batch
        #                (ce, fi, ep_status, force_idr, bs_buf) — encode_frames_batch_ce_pipeline
        self._slot_pending: dict = {}
        self._slots_warmed: set = set()
        self._slots: list = []
        # [FIX-AUX-NO-CLEAR] / [FIX-DRAIN-ORDER-DEFENSE] 诊断与兜底状态
        self._diag_aux_block = 0
        self._diag_phase_shift = 0
        self._diag_slot_drain_fallback = 0
        self._prev_stream_h264: Optional[bytes] = None

        # Backward compat: legacy refs (initialized after slot creation)
        self._input_buf_handle = c_void_p(None)
        self._bs_handle = c_void_p(None)

        # [SEGMENT-REUSE] 缓存首段 SPS+PPS NAL 单元，后续段预挂到首帧前
        self._cached_sps_pps: Optional[bytes] = None
        self._sps_pps_injected: bool = False  # [FIX-SPS-PPS-V2] Writer-thread-side 注入已完成标志
        # [FIX-GLOBAL-FI] 跨 chunk 延迟输出收集列表，由 encode_frames_batch() 每次调用后填充，
        # 由 _write_la_output() 读取并优先写入。
        self._prev_chunk_outputs: list = []
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
        self._nvenc_api_version = _nvenc_api_version  # reopen() 复用
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

        # [FIX-CROSS-SEGMENT-SESSION] 记录会话创建所需的上下文/版本，
        # 供 reopen() 在段边界原地重建驱动会话时复用（不重复 retain）。
        self._open_device_ctx = cuda_ctx
        self._needs_reopen = False   # EOS 后由 flush()/flush_and_join() 置位
        self._session_gen = 0        # 驱动会话代数（每次 _open_session 递增）
        self._closed = False         # close() 幂等标志
        self._open_session()

    def _open_session(self):
        """创建 NVENC 驱动会话：CreateInstance → OpenEncodeSessionEx →
        GUID/preset 查询 → InitializeEncoder → 多 slot buffer/event →
        专用编码拷贝 stream。

        __init__ 首次调用；reopen()（段边界会话重启）在销毁旧会话后再次
        调用。因此本方法只依赖实例属性，不依赖 __init__ 的局部变量。
        """
        width, height, fps, qp = self._width, self._height, self._fps, self._qp
        la_depth = self._la_depth
        cuda_ctx = self._open_device_ctx
        _nvenc_api_version = self._nvenc_api_version
        self._session_gen += 1
        self._slots = []

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
            _est_br = max(50000000, int(width * height * fps * 3.0))
            rc_ptr[5] = _est_br                      # averageBitRate @offset 20 (速度天花板)
            rc_ptr[6] = _est_br * 2                  # maxBitRate @offset 24
            _tq = max(1, _qp_val)  # VBR_HQ targetQuality = CRF (QP标度, 1=最好, 51=最差)
            # targetQuality: uint8_t at rcParams+88 (nvEncodeAPI.h SEQUENTIAL, GPU verified)
            _tq8_ptr = cast(byref(preset_config, 8 + 40 + 88), ctypes.POINTER(c_uint8))
            _tq8_ptr[0] = _tq & 0xFF
            # [DIAG-TERM] targetQuality 是 RC 质量参数（CRF 语义 1-51），不是 H.264 Level。
            print(f"[NVENCEncoder] VBR_HQ: crf={_qp_val} targetQuality(CRF)={_tq} avgBitrate={_est_br//1000}kbps", flush=True)
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

        # [FIX-ASYNC-COPY] 移植自 IFRNet v6.4.5.1。cuMemcpy2D_v2（同步、走
        # legacy default/null stream）会隐式地在本 CUDA context 下的**所有**
        # stream 之间做全局同步——每次 NVENC 输入拷贝都会强制等待 SR 推理用
        # 的计算/H2D/D2H stream 排空，反之亦然。在分块累积编码场景下，编码
        # 线程连续、逐帧执行这个同步拷贝，会周期性阻塞推理线程的 GPU kernel，
        # 表现为 GPU 利用率规律性骤降（与 IFRNet 项目诊断出的问题同根同源）。
        #
        # 修复：改用 cuMemcpy2DAsync_v2 + 专用非默认 CUDA stream
        # (self._stream_encode)。拷贝仍然入队后立即 cuStreamSynchronize，
        # 编码线程自身的等待时间不变，但**只同步这一个私有 stream**，
        # 不再对其他 stream 施加隐式全局屏障。
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
            _extra += " tq(CRF)=%d" % _tq
        print("[NVENCEncoder] Ready: %dx%d@%.1ffps H.264 %s QP=%d preset=%s slots=%d%s (GPU direct SDK 13.0)" %
              (width, height, fps, _mode_label, _qp_val, self._preset_name, self._slot_count, _extra), flush=True)

    def _copy_into_input_buffer(self, cpy2d_struct) -> int:
        """[FIX-ASYNC-COPY] 把 NV12 源数据拷贝进 NVENC 输入缓冲区。

        优先走 cuMemcpy2DAsync_v2 + 专用非默认 stream(self._stream_encode)，
        入队后立即 cuStreamSynchronize —— 编码线程自身仍然同步等待拷贝完成
        (语义与之前一致，帧顺序/正确性不变)，但只同步这一个私有 stream，
        不再像 cuMemcpy2D_v2(legacy null stream) 那样对同 context 下所有
        stream(包括 SR 推理用的计算/H2D/D2H stream)施加隐式全局屏障。

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
    def _parse_h264_sps_level(sps_nal: bytes) -> Optional[tuple]:
        """[DIAG-LEVEL] 从 H.264 SPS NAL（含起始码）解析 (profile_idc, level_idc)。

        SPS RBSP 前 4 字节（不受 emulation prevention 影响）:
          byte[0] = NAL header (nal_unit_type=7)
          byte[1] = profile_idc
          byte[2] = constraint_setN_flags + reserved
          byte[3] = level_idc
        HEVC SPS (type=33) 布局不同，本项目仅 H.264，暂不支持。
        """
        try:
            data = bytes(sps_nal)
            if data.startswith(b"\x00\x00\x00\x01"):
                data = data[4:]
            elif data.startswith(b"\x00\x00\x01"):
                data = data[3:]
            if len(data) < 4 or (data[0] & 0x1F) != 7:
                return None
            return data[1], data[3]
        except Exception:
            return None

    def _extract_sps_pps(self, h264_data: bytes) -> Optional[bytes]:
        """从 H.264 Annex B ES 中提取 SPS+PPS NAL 单元（含起始码）。"""
        # [DIAG-LEVEL] 首次提取到 SPS 时解析实际 level（诊断 profileLevel 是否生效）
        _sps_prof_lvl = None
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
        无 VCL 时返回 None。NVENC LA 冷会话预热期会把 SPS/PPS/AUD 作为独立
        辅助块（aux block）drain 出来，不对应任何已提交帧：VCL 块占帧槽，
        辅助块仅缓存参数集、不占 fi、不进 results、不清 pending。
        移植自 IFRNet v6.4.5.1（test11 根因，2026-08-14）。
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

        返回 [(frame_index_estimate, output_timestamp, h264_bytes), ...] 列表。
        output_timestamp 为 NV_ENC_LOCK_BITSTREAM.outputTimeStamp@40 回显的
        提交时 inputTimeStamp（仅用于 [FIX-DRAIN-ORDER-DEFENSE] 相位漂移诊断，
        不作为标签来源 —— 标签一律来自 per-slot FIFO 记账）。

        注意：调用方必须已确保正确的 CUDA context 为 current
        （encode_frames_batch / encode_frame 在调用前已 push primary context）。
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
            slot = self._slots[slot_idx]
            bs_handle = slot['bs_buf']
            _bs_val = bs_handle.value if hasattr(bs_handle, 'value') else bs_handle
            if not _bs_val:
                break  # 无效的 bs_handle，跳过该 slot

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
                est_fi = self._output_slot_idx
                outputs.append((est_fi, out_ts, h264_data))
                unlock_fn(self._encoder, bs_handle)
                self._output_slot_idx += 1
            else:
                # [FIX-DRAIN-COUNTER-DRIFT] size>0 但数据指针为空：未取到码流。
                # 此时推进 _output_slot_idx 会造成"帧计数超前于实际取回"的
                # 相位漂移，必须 break 且不推进。
                unlock_fn(self._encoder, bs_handle)
                break

        return outputs

    def _reset_output_slot_idx(self, start: int = 0):
        """重置输出槽位指针（新批次开始时调用）。"""
        self._output_slot_idx = start

    def _apply_drained_entries(self, drained: list, chunk_start_global: int,
                                n_frames: int, results: list,
                                prev_chunk_outputs: list) -> None:
        """[FIX-SLOT-BACKPRESSURE] 把 _drain_outputs_blocking() 返回的条目
        分发到 results[] 或 prev_chunk_outputs，与原来内联在 encode_frames_batch
        提交循环里的逻辑完全一致，抽成方法是为了让"提交前背压检查"能复用同一套
        分发逻辑，而不是复制粘贴一份容易和原逻辑跑偏的副本。
        """
        for _est_fi, _out_ts, _h264_data in drained:
            # [FIX-AUX-NO-CLEAR] 先判 VCL 再消费：无 VCL 辅助块（独立 SPS/PPS/AUD）
            # 仅缓存参数集，不占 fi、不进 results、不清 pending —— 否则该槽真实
            # 帧的 VCL 数据仍滞留驱动输出队列而 pending 已清空，_ensure_slot_free
            # 误判槽位空闲 → 输入 buffer 未排空即复用 → LA 延迟消费读到被覆盖的
            # 像素 → 数据与标签错位 → 每 9 帧迟一窗（IFRNet test11 根因）。
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
            _entry_deque = self._slot_pending.get(_drain_slot)
            if not _entry_deque:
                continue
            _global_fi, _, _is_idr, _ep_s = _entry_deque[0]  # peek 最旧条目
            _actual_fi = _global_fi - chunk_start_global
            # [FIX-DRAIN-ORDER-DEFENSE] 相位漂移检测：VCL 块 outputTimeStamp@40
            # 回显提交时 inputTimeStamp（tt6/tt7 生产双射验证）。与 FIFO 队首
            # gfi 不一致 = LA 重路由/输入覆盖征兆；标签仍以 FIFO 队首为准。
            if _out_ts is not None and _out_ts != _global_fi:
                self.__dict__.setdefault('_diag_phase_shift', 0)
                self._diag_phase_shift += 1
                if self._diag_phase_shift <= 5 or self._diag_phase_shift % 50 == 0:
                    print(f'[NVENC-Enc] ⚠️ 相位漂移 #{self._diag_phase_shift} '
                          f'(ts={_out_ts} != fifo_gfi={_global_fi} '
                          f'slot={_drain_slot} est={_est_fi})', flush=True)
            if _actual_fi < 0:
                # 属于前一个 chunk（或前一段残留）的延迟输出。不能丢弃，
                # 收集到 prev_chunk_outputs，由调用方在写入本 chunk 输出之前先写入。
                if _h264_data:
                    if _is_idr and self._cached_sps_pps is not None and \
                            not self._has_sps_pps(_h264_data):
                        _h264_data = self._cached_sps_pps + _h264_data
                    elif self._cached_sps_pps is None:
                        self._cached_sps_pps = self._extract_sps_pps(_h264_data)
                        if self._cached_sps_pps:
                            print("\n[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                                  len(self._cached_sps_pps), flush=True)
                    prev_chunk_outputs.append(_h264_data)
                    self._prev_stream_h264 = _h264_data
                _entry_deque.popleft()
                if not _entry_deque:
                    del self._slot_pending[_drain_slot]
                continue
            if _actual_fi >= n_frames:
                continue
            if results[_actual_fi] is not None and results[_actual_fi] != b"":
                continue

            if _h264_data:
                if _is_idr and self._cached_sps_pps is not None and \
                        not self._has_sps_pps(_h264_data):
                    _h264_data = self._cached_sps_pps + _h264_data
                elif self._cached_sps_pps is None:
                    self._cached_sps_pps = self._extract_sps_pps(_h264_data)
                    if self._cached_sps_pps:
                        print("\n[NVENCEncoder] Cached SPS+PPS: %d bytes" %
                              len(self._cached_sps_pps), flush=True)
                        if self._muxer_ref is not None:
                            try:
                                self._muxer_ref.write_sps_pps(self._cached_sps_pps)
                                self._sps_pps_injected = True
                            except Exception:
                                pass
                results[_actual_fi] = _h264_data
                self._prev_stream_h264 = _h264_data
            elif _ep_s == NV_ENC_ERR_NEED_MORE_INPUT:
                results[_actual_fi] = b""
            _entry_deque.popleft()
            if not _entry_deque:
                del self._slot_pending[_drain_slot]

    def _ensure_slot_free(self, slot_idx: int, chunk_start_global: int,
                           n_frames: int, results: list,
                           prev_chunk_outputs: list) -> None:
        """[FIX-SLOT-BACKPRESSURE] 在把 slot_idx 对应的物理 bs_buf 复用给新一帧之前，
        必须先确认这个物理 slot 上一次占用者的码流已经被排空。

        根因：EncodePicture 提交新帧时会复用 self._slots[slot_idx]['bs_buf']
        这块物理缓冲区。如果上一次用这个 slot 提交的帧的码流还没被 LockBitstream
        取走，这次 EncodePicture 会让驱动直接在硬件层面覆盖/丢弃那份还没读出来的
        码流——这个丢失发生在物理层面，一旦覆盖就无法通过"事后多 drain 几遍"补救。

        原来的实现只在"提交之后"做 best-effort 的增量 drain
        （_pending_cnt = _frame_idx - _output_slot_idx，drain up to slot_count），
        指望排空节奏能跟上提交节奏。但 la_depth(8) 和 slot_count(9) 之间余量只有 1
        帧，一旦 chunk/段边界的抖动（比如 send_eos 收尾时新提交帧变少、
        drain 触发次数跟着变少）让 backlog 超过 slot_count，某个物理 slot 就会被
        "带病"复用，对应帧的码流永久丢失——这正是分段视频里出现"某段少几帧、
        紧邻的下一段多出几乎相同数量帧"的根因（丢的帧因为跨 chunk 的
        _actual_fi < 0 兜底逻辑，被记进了下一段的 prev_chunk_outputs）。

        修复：把 backpressure 从"提交后尽量追上"改成"提交前强制等到干净"，
        结构性地保证每个物理 slot 被复用时，它上一次的码流必然已经被排空，
        不会再发生覆盖丢失。
        """
        _guard = 0
        while self._slot_pending.get(slot_idx):
            _drained = self._drain_outputs_blocking(max_slots=1)
            if _drained:
                # [FIX-AUX-NO-CLEAR] 辅助块会被 _apply_drained_entries 跳过
                # （不 pop），但物理轮转指针已推进 —— 循环继续直至目标槽排空。
                _guard = 0
                self._apply_drained_entries(_drained, chunk_start_global, n_frames,
                                             results, prev_chunk_outputs)
                continue
            # 阻塞 LockBitstream 理论上应该等到这帧真正完成
            # （NEED_MORE_INPUT 之外的分支才会 break）。这里给一个
            # 保守的循环次数上限，避免因意外状态导致死循环卡死整个进程。
            _guard += 1
            if _guard > self._slot_count * 4:
                # [FIX-SLOT-DRAIN-TARGET] 轮转探测无法触达目标槽（LA 重路由/
                # 辅助块使目标槽输出晚于其他槽就绪）→ 直接对目标槽自身做
                # blocking LockBitstream（doNotWait=0）。
                _h264_t, _st_t = self._lock_bitstream_blocking(
                    self._slots[slot_idx]['bs_buf'], timeout_ms=2000)
                if _h264_t:
                    self._apply_drained_entries([(slot_idx, None, _h264_t)],
                                                chunk_start_global, n_frames,
                                                results, prev_chunk_outputs)
                    _guard = 0
                    continue
                # [FIX-SLOT-BACKPRESSURE-B] 硬件仍无数据：绝不带 pending 复用，
                # 将目标槽 pending 逐条以空帧占位（prev 填充）消费并推进 FIFO；
                # 宁可写占位也绝不覆盖未取回的码流。
                _dq = self._slot_pending.get(slot_idx)
                if _dq:
                    self.__dict__.setdefault('_diag_slot_drain_fallback', 0)
                    self._diag_slot_drain_fallback += 1
                    if self._diag_slot_drain_fallback <= 5 or \
                            self._diag_slot_drain_fallback % 50 == 0:
                        print(f'[NVENC-Enc] ⚠️ slot={slot_idx} 排空超限 '
                              f'#{self._diag_slot_drain_fallback}：'
                              f'空帧占位兜底（prev 填充）', flush=True)
                    _fill = self._prev_stream_h264 or b""
                    while _dq:
                        _gfi_f, _, _is_idr_f, _ep_s_f = _dq[0]
                        _actual_fi_f = _gfi_f - chunk_start_global
                        if 0 <= _actual_fi_f < n_frames and \
                                (results[_actual_fi_f] is None or results[_actual_fi_f] == b""):
                            results[_actual_fi_f] = _fill
                        _dq.popleft()
                    del self._slot_pending[slot_idx]
                    self._output_slot_idx += 1
                break
            continue

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
        # [FIX-EOS-EMPTY-CHUNK] 旧代码在 nv12_tensors 为空时无条件提前 return []，
        # 完全跳过了下面 send_eos 分支（EOS picture 发送 + per-slot 阻塞 drain）。
        # 当一个 segment 的总帧数恰好被 chunk 大小整除、最终块累积不到任何剩余
        # 原始帧（_acc_nv12 为空）时，_NVENCEncodeThread._loop() 仍会以
        # encode_frames_batch([], False, send_eos=True) 调用本函数——此时 EOS 从未
        # 真正发给驱动，仍滞留在硬件 LA 重排序队列里的最后 LA_depth 帧（本例 8 帧）
        # 永远不会被 drain 出来：既不计入 written 也不计入 empty，是纯粹的静默丢帧
        # （对应 [NVENC-Enc] "帧未被回收就结束段编码" 告警）。又因为 encoder 硬件
        # session 跨 segment 持续复用（不 close），这些帧不会真的消失，而是在下一个
        # segment 提交新帧时被自然 drain 出来，被 _actual_fi_d<0 分支误判为"属于
        # 上一个 chunk"，写进了下一个 segment 自己的 muxer——对应本段丢尾部帧、
        # 下一段开头混入本段尾部帧内容（且下一段引用的解码参考帧在新 GOP 里不存在，
        # 多数因此在下游解码时被判为损坏帧而丢弃）的现象。
        # 修复：仅当确实没有新帧 *且* 不需要发送 EOS 时才提前返回；send_eos=True 时
        # 即使 nv12_tensors 为空也必须继续往下走，让 EOS + per-slot 阻塞 drain 逻辑
        # 执行，把滞留在硬件里的帧真正回收并写回本 segment 自己的输出。
        if n_frames == 0 and not send_eos:
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
                # [FIX-CHUNKED-LA] _slot_pending 和 _slots_warmed 已提升为实例变量
                # (self._slot_pending, self._slots_warmed)，
                # 支持 chunked LA 模式下跨 encode_frames_batch() 调用的 drain 映射正确性。
                # 不再在此处重置 — 跨 chunk drain 依赖持久化。
                # [FIX-GLOBAL-FI] 记录本次调用（chunk）起始的全局帧索引，
                # 用于把 _slot_pending 中存储的全局 fi 换算为本 chunk 的 local fi。
                # 之前版本存储 local fi 导致跨 chunk slot 复用时相互覆盖（见 v3 fix doc）。
                _chunk_start_global = self._frame_idx
                _prev_chunk_outputs: list = []  # 跨块延迟输出（属于前一个 chunk 的帧）
                W = self._width
                nv12_h = self._height + self._height // 2

                _LockInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.POINTER(c_uint8 * 1544))
                _UnlockInputBufferProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, c_void_p)
                _CU_MEMORYTYPE_DEVICE = 2
                _CU_MEMORYTYPE_HOST   = 1  # [FIX-HOST-SOURCE] D2H path support

                for fi in range(n_frames):
                    slot_idx = self._frame_idx % self._slot_count
                    # [FIX-SLOT-BACKPRESSURE] 复用这个物理 slot 的 bs_buf 之前，
                    # 强制确保它上一次占用者的码流已经被排空，避免硬件层面覆盖丢帧。
                    # 详见 _ensure_slot_free() 的说明。
                    self._ensure_slot_free(slot_idx, _chunk_start_global, n_frames,
                                           results, _prev_chunk_outputs)
                    slot = self._slots[slot_idx]
                    # [FIX-PIPE4-LA8] fi==0 only force_idr（单 IDR 替代 per-slot IDR）。
                    # 原 per-slot 判定在 LA 预热期（NEED_MORE_INPUT）slot 永不 warm →
                    # 段首多帧连续 IDR（per-slot IDR 特征）。对齐 ce_pipeline 已验证方案。
                    force_idr = force_idr_first and (fi == 0)

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
                    # [FIX-HOST-SOURCE] D2H 路径下 nv12_tensors 可能已在 pinned host
                    # 内存，此处按来源张量所在设备决定 srcMemoryType。
                    _src_t = nv12_tensors[fi]
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

                    # [FIX-NV12-EARLY-FREE] cuMemcpy2D 完成后源 NV12 tensor 不再需要，
                    # 立即释放引用让 PyTorch CUDA allocator 回收显存。
                    # LA>0 模式下 _acc_nv12 累积全段帧 → 高分辨率超分时
                    # （1536×1152 NV12 ≈ 2.5MB/帧，1000帧 = 2.5GB）必然 OOM。
                    nv12_tensors[fi] = None

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
                    # EncodePicture (同 encode_frame 已验证模式)。无 CE 的同步 EncodePicture
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

                    # Record submission context for drain-to-index mapping
                    # [FIX-GLOBAL-FI] 存储全局 fi（_chunk_start_global + local fi）而非 local fi。
                    # [FIX-SLOT-DEQUE] 新帧追加到 slot 的 deque 而非覆盖旧条目，
                    # 避免 drain 延迟时跨 chunk slot 覆盖导致的帧乱序/跳跃/重复。
                    _global_fi_submitted = _chunk_start_global + fi
                    _entry = (_global_fi_submitted, slot['bs_buf'], force_idr, _ep_status)
                    if slot_idx not in self._slot_pending:
                        self._slot_pending[slot_idx] = deque()
                    self._slot_pending[slot_idx].append(_entry)

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
                        pass  # [FIX-PIPE4-LA8] IDR 由 fi==0 控制，无需 per-slot 标记

                    # ── Global drain: Loop blocking LockBitstream to drain ALL completed slots ──
                    # [FIX-DRAIN-UNSUBMITTED-SLOT] 只 drain 已提交但未排空的 slot。
                    # CONSTQP+LA=0 下 LockBitstream 未提交的 bitstream buffer 会导致
                    # NVENC driver segfault（不返回 NEED_MORE_INPUT）。使用已提交帧数
                    # 减去已排空帧数（_output_slot_idx）作为上限确保不触及未提交 slot。
                    # [FIX-SLOT-BACKPRESSURE] 分发逻辑抽到 _apply_drained_entries()，
                    # 与 _ensure_slot_free() 的提交前背压检查共用同一套实现。
                    _pending_cnt = self._frame_idx - self._output_slot_idx
                    _max_drain = min(_pending_cnt, self._slot_count)
                    _drained = self._drain_outputs_blocking(max_slots=_max_drain) if _max_drain > 0 else []
                    self._apply_drained_entries(_drained, _chunk_start_global, n_frames,
                                                 results, _prev_chunk_outputs)

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
                                # [FIX-AUX-NO-CLEAR] EOS 排空同样先判 VCL：
                                # 无 VCL 辅助块仅缓存参数集，不 pop pending、
                                # 不写 results、不推进帧计数（保留队首供真实帧）。
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
                                # [FIX-SLOT-DEQUE] EOS per-slot drain: 每次 while 迭代取最旧条目
                                _entry_deque = self._slot_pending.get(_ds)
                                if _entry_deque:
                                    _global_fi_d, _, _is_idr_d, _ep_s_d = _entry_deque[0]
                                    # [FIX-GLOBAL-FI] 全局 fi → local fi，负值属于前一 chunk。
                                    _actual_fi_d = _global_fi_d - _chunk_start_global
                                    if _actual_fi_d < 0:
                                        if _is_idr_d and self._cached_sps_pps is not None and \
                                                not self._has_sps_pps(_eos_data):
                                            _eos_data = self._cached_sps_pps + _eos_data
                                        _prev_chunk_outputs.append(_eos_data)
                                        _entry_deque.popleft()
                                        if not _entry_deque:
                                            del self._slot_pending[_ds]
                                        self._output_slot_idx += 1
                                    elif _actual_fi_d < n_frames:
                                        results[_actual_fi_d] = _eos_data
                                        _entry_deque.popleft()
                                        if not _entry_deque:
                                            del self._slot_pending[_ds]
                                        self._output_slot_idx += 1
                            _UnlockBS(self._func_ptrs[_FUNC_IDX["UnlockBitstream"]])(self._encoder, _bs_h)

                else:
                    # Without EOS: final drain attempt (LA frames may not be ready)
                    # [FIX-DRAIN-UNSUBMITTED-SLOT] 限制只 drain 剩余未排空的帧
                    _final_pending = self._frame_idx - self._output_slot_idx
                    _final_max = min(_final_pending, self._slot_count)
                    _drained_final = self._drain_outputs_blocking(
                        max_slots=_final_max) if _final_max > 0 else []
                    for _est_fi, _out_ts_f, _h264_data in _drained_final:
                        # [FIX-AUX-NO-CLEAR] 末尾 drain 同样跳过无 VCL 辅助块
                        if self._nal_first_vcl_type(_h264_data) is None:
                            self.__dict__.setdefault('_diag_aux_block', 0)
                            self._diag_aux_block += 1
                            if self._diag_aux_block <= 5 or \
                                    self._diag_aux_block % 50 == 0:
                                print(f'[NVENC-Enc] ℹ️ 末尾辅助块 '
                                      f'#{self._diag_aux_block} '
                                      f'(est={_est_fi}) 跳过帧记账', flush=True)
                            if self._cached_sps_pps is None:
                                _sps_f = self._extract_sps_pps(_h264_data)
                                if _sps_f:
                                    self._cached_sps_pps = _sps_f
                                    print("[NVENCEncoder] Cached SPS+PPS: %d bytes"
                                          % len(_sps_f), flush=True)
                            continue
                        _drain_slot = _est_fi % self._slot_count
                        # [FIX-SLOT-DEQUE] 获取 slot 的待处理 deque（FIFO）
                        _entry_deque = self._slot_pending.get(_drain_slot)
                        if not _entry_deque:
                            continue
                        _global_fi, _, _is_idr, _ep_s = _entry_deque[0]
                        # [FIX-GLOBAL-FI] 全局 fi → local fi，负值属于前一 chunk。
                        _actual_fi = _global_fi - _chunk_start_global
                        if _actual_fi < 0:
                            if _h264_data:
                                if _is_idr and self._cached_sps_pps is not None and \
                                        not self._has_sps_pps(_h264_data):
                                    _h264_data = self._cached_sps_pps + _h264_data
                                _prev_chunk_outputs.append(_h264_data)
                                self._prev_stream_h264 = _h264_data
                            _entry_deque.popleft()
                            if not _entry_deque:
                                del self._slot_pending[_drain_slot]
                            continue
                        if _actual_fi >= n_frames:
                            continue
                        if results[_actual_fi] is not None:
                            continue
                        if _h264_data:
                            results[_actual_fi] = _h264_data
                            self._prev_stream_h264 = _h264_data
                        elif _ep_s == NV_ENC_ERR_NEED_MORE_INPUT:
                            results[_actual_fi] = b""
                        _entry_deque.popleft()
                        if not _entry_deque:
                            del self._slot_pending[_drain_slot]

                # Fill remaining None entries
                for _fi in range(n_frames):
                    if results[_fi] is None:
                        results[_fi] = b""

                # [FIX-GLOBAL-FI] 暴露本次调用中恢复到的、属于前一个 chunk 的延迟输出，
                # 供调用方 (_write_la_output) 在写入本 chunk 结果之前先行写入，
                # 保证跨 chunk 输出顺序正确。
                self._prev_chunk_outputs = _prev_chunk_outputs

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

                # [FIX-DEADCODE-SLOTPENDING] 实际挂起状态用的是 self._slot_pending
                # (实例变量，5-tuple: ce_handle, frame_index, ep_status, force_idr, bs_buf)。
                # 之前这里还留了一个同名的局部变量 `_slot_pending = [None]*pd`，
                # 从未被读写过（下方全部用 self._slot_pending），纯属重构残留死代码，
                # 容易让后续维护者误以为它是生效状态，故删除。
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
                    # [FIX-SLOT-DEQUE] CE pipeline: peek+popleft 最旧待处理条目
                    _entry_deque = self._slot_pending.get(slot_idx)
                    if _entry_deque:
                        _prev_ce, _prev_fi, _prev_ep_s, _prev_idr, _prev_bs = _entry_deque[0]
                        # Wait for CE → data is now in _prev_bs (the bs_buf this frame was submitted to)
                        if _prev_ce.value is not None:
                            self._libcuda.cuEventSynchronize.restype = c_uint32
                            self._libcuda.cuEventSynchronize.argtypes = [c_void_p]
                            # [FIX-RC-CHECK] 检查 libcuda 返回码：若 CUDA context 已被
                            # 先前的异步故障污染（如 700 illegal address），第一个受检
                            # 调用会在此抛出带上下文的 RuntimeError，而不是让进程在
                            # 后续某个无关位置 SIGSEGV。
                            _r_esync = self._libcuda.cuEventSynchronize(_prev_ce)
                            if _r_esync != 0:
                                raise RuntimeError(
                                    f'[NVENC-Enc] cuEventSynchronize failed code={_r_esync} '
                                    f'(Phase1 slot={slot_idx} fi={fi} '
                                    f'gen={getattr(self, "_session_gen", "?")})')
                            self._libcuda.cuEventDestroy.restype = c_uint32
                            self._libcuda.cuEventDestroy.argtypes = [c_void_p]
                            _r_edes = self._libcuda.cuEventDestroy(_prev_ce)
                            if _r_edes != 0:
                                raise RuntimeError(
                                    f'[NVENC-Enc] cuEventDestroy failed code={_r_edes} '
                                    f'(Phase1 slot={slot_idx} fi={fi} '
                                    f'gen={getattr(self, "_session_gen", "?")})')
                        # harvest from stored bs_buf (not current slot's bs_buf)
                        h264_data, bs_status = self._lock_bitstream_with_retry(_prev_bs)
                        # [FIX-BOUNDS-CE-PHASE1] 防御性边界检查：_prev_fi 理论上总是本次调用
                        # 范围内的 local fi（Phase 3 保证每次调用结束时 self._slot_pending
                        # 已被清空，不会跨调用残留），但仍加上检查以避免任何未预见路径下
                        # 越界写入 results 或产生静默错位。
                        if not (0 <= _prev_fi < n_frames):
                            print(f'[NVENC-Enc] ⚠️ Phase1 harvest _prev_fi={_prev_fi} 超出本批次范围 '
                                  f'[0,{n_frames})，跳过写入（slot={slot_idx}）', flush=True)
                        elif h264_data:
                            # [SEGMENT-REUSE] SPS/PPS caching and pre-pending
                            if _prev_idr and self._cached_sps_pps is not None and \
                                    not self._has_sps_pps(h264_data):
                                h264_data = self._cached_sps_pps + h264_data
                            elif _prev_idr and self._cached_sps_pps is None and h264_data:
                                self._cached_sps_pps = self._extract_sps_pps(h264_data)
                                if self._cached_sps_pps:
                                    print("\n[NVENCEncoder] Cached SPS+PPS: %d bytes" %
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
                                    print("\n[NVENCEncoder] Cached SPS+PPS: %d bytes" %
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
                        # [FIX-SLOT-DEQUE] 处理完成，弹出最旧条目
                        _entry_deque.popleft()
                        if not _entry_deque:
                            del self._slot_pending[slot_idx]

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
                        # [FIX-RC-CHECK] 返回码检查（理由见 Phase1 处注释）
                        _r_ecre = self._libcuda.cuEventCreate(ctypes.byref(_ce), 0)
                        if _r_ecre != 0:
                            raise RuntimeError(
                                f'[NVENC-Enc] cuEventCreate failed code={_r_ecre} '
                                f'(slot={slot_idx} fi={fi} '
                                f'gen={getattr(self, "_session_gen", "?")})')

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
                    # [FIX-SLOT-DEQUE] 追加到 slot 的 deque 而非覆盖旧条目
                    _entry = (_ce, fi, _ep_status, force_idr, slot['bs_buf'])
                    if slot_idx not in self._slot_pending:
                        self._slot_pending[slot_idx] = deque()
                    self._slot_pending[slot_idx].append(_entry)

                    # [FIX-LA-CE-DRAIN] LA>0: inline global drain after each frame.
                    # Without CE, completed frames from other slots must be collected
                    # immediately via synchronous drain.
                    if _la_disable_ce:
                        _drained_inline = self._drain_outputs_blocking()
                        for _est_fi, _out_ts_i, _h264_data in _drained_inline:
                            # [FIX-AUX-NO-CLEAR] CE inline drain 同样先判 VCL
                            if self._nal_first_vcl_type(_h264_data) is None:
                                self.__dict__.setdefault('_diag_aux_block', 0)
                                self._diag_aux_block += 1
                                if self._diag_aux_block <= 5 or \
                                        self._diag_aux_block % 50 == 0:
                                    print(f'[NVENC-Enc] ℹ️ inline 辅助块 '
                                          f'#{self._diag_aux_block} '
                                          f'(est={_est_fi}) 跳过帧记账', flush=True)
                                if self._cached_sps_pps is None:
                                    _sps_i = self._extract_sps_pps(_h264_data)
                                    if _sps_i:
                                        self._cached_sps_pps = _sps_i
                                        print("[NVENCEncoder] Cached SPS+PPS: %d bytes"
                                              % len(_sps_i), flush=True)
                                continue
                            _drain_slot = _est_fi % pd
                            # [FIX-SLOT-DEQUE] CE inline drain: 5-tuple peek+popleft
                            _entry_deque = self._slot_pending.get(_drain_slot)
                            if not _entry_deque:
                                continue
                            _, _actual_fi, _ep_s, _is_idr, _ = _entry_deque[0]
                            if _actual_fi >= n_frames:
                                continue
                            if results[_actual_fi] is not None and results[_actual_fi] != b"":
                                continue
                            if _h264_data:
                                results[_actual_fi] = _h264_data
                                self._prev_stream_h264 = _h264_data
                            elif _ep_s == NV_ENC_ERR_NEED_MORE_INPUT:
                                results[_actual_fi] = b""
                            _entry_deque.popleft()
                            if not _entry_deque:
                                del self._slot_pending[_drain_slot]

                # ═══════════════════════════════════════════════
                # Phase 3: Drain remaining pending slots
                # [FIX-SLOT-DEQUE] 每个 slot 可能有多条待处理条目，while 循环全部排空
                # ═══════════════════════════════════════════════
                for slot_idx in range(pd):
                    while True:
                        _entry_deque = self._slot_pending.get(slot_idx)
                        if not _entry_deque:
                            break
                        _pending_ce, _pending_fi, _pending_ep_s, _pending_idr, _pending_bs = \
                            _entry_deque[0]
                        if _pending_ce.value is not None:
                            self._libcuda.cuEventSynchronize.restype = c_uint32
                            self._libcuda.cuEventSynchronize.argtypes = [c_void_p]
                            # [FIX-RC-CHECK] 返回码检查（理由见 Phase1 处注释）
                            _r_esync3 = self._libcuda.cuEventSynchronize(_pending_ce)
                            if _r_esync3 != 0:
                                raise RuntimeError(
                                    f'[NVENC-Enc] cuEventSynchronize failed code={_r_esync3} '
                                    f'(Phase3 slot={slot_idx} '
                                    f'gen={getattr(self, "_session_gen", "?")})')
                            self._libcuda.cuEventDestroy.restype = c_uint32
                            self._libcuda.cuEventDestroy.argtypes = [c_void_p]
                            _r_edes3 = self._libcuda.cuEventDestroy(_pending_ce)
                            if _r_edes3 != 0:
                                raise RuntimeError(
                                    f'[NVENC-Enc] cuEventDestroy failed code={_r_edes3} '
                                    f'(Phase3 slot={slot_idx} '
                                    f'gen={getattr(self, "_session_gen", "?")})')
                        h264_data, bs_status = self._lock_bitstream_with_retry(_pending_bs)
                        # [FIX-BOUNDS-CE-PHASE3] 防御性边界检查：同 Phase 1，_pending_fi
                        # 理论上总在 [0, n_frames) 内，加检查避免任何未预见路径下越界/错位写入。
                        if not (0 <= _pending_fi < n_frames):
                            print(f'[NVENC-Enc] ⚠️ Phase3 drain _pending_fi={_pending_fi} 超出本批次范围 '
                                  f'[0,{n_frames})，跳过写入（slot={slot_idx}）', flush=True)
                        elif h264_data:
                            # [SEGMENT-REUSE] SPS/PPS caching
                            if _pending_idr and self._cached_sps_pps is not None and \
                                    not self._has_sps_pps(h264_data):
                                h264_data = self._cached_sps_pps + h264_data
                            elif _pending_idr and self._cached_sps_pps is None and h264_data:
                                self._cached_sps_pps = self._extract_sps_pps(h264_data)
                                if self._cached_sps_pps:
                                    print("\n[NVENCEncoder] Cached SPS+PPS: %d bytes" %
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
                                    print("\n[NVENCEncoder] Cached SPS+PPS: %d bytes" %
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
                        _entry_deque.popleft()
                        if not _entry_deque:
                            del self._slot_pending[slot_idx]

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

                # ── Global drain: 排空所有已完成 slot（替代单 slot 0 LockBitstream）──
                _drained = self._drain_outputs_blocking()
                if _drained:
                    # [FIX-AUX-NO-CLEAR] 取首个 VCL 帧；跳过无 VCL 辅助块
                    # （LA 冷会话下 SPS/PPS/AUD 可能先于首帧输出）。
                    for _e_d in _drained:
                        if self._nal_first_vcl_type(_e_d[2]) is not None:
                            h264_data = _e_d[2]
                            self._prev_stream_h264 = h264_data
                            break

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
                if force_idr and self._cached_sps_pps is not None and \
                        not self._has_sps_pps(h264_data):
                    h264_data = self._cached_sps_pps + h264_data
                elif force_idr and self._cached_sps_pps is None and h264_data:
                    # [FIX-SPS-CACHE] 首个 IDR 帧: 从码流提取并缓存 SPS+PPS
                    self._cached_sps_pps = self._extract_sps_pps(h264_data)
                    if self._cached_sps_pps:
                        print("\n[NVENCEncoder] Cached SPS+PPS: %d bytes" % len(self._cached_sps_pps),
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
                        print("\n[NVENCEncoder] Cached SPS+PPS: %d bytes" % len(self._cached_sps_pps),
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
                # [FIX-RC-CHECK] EOS EncodePicture 此前完全忽略返回值：会话异常时
                # 静默失败，后续 per-slot drain 会在带病会话上继续执行。
                _eos_status = encode_picture(self._encoder, cast(pic_buf, ctypes.POINTER(_NvEncPicParams)))
                if _eos_status != NV_ENC_SUCCESS:
                    raise RuntimeError(
                        f'[NVENCEncoder] flush: EOS EncodePicture failed, '
                        f'code={_eos_status} (gen={getattr(self, "_session_gen", "?")})')

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
                # EOS 已发出。跨段复用架构下 _needs_reopen 仅作为簿记标记
                # （记录 EOS 已发出的事实），不再触发会话重建。
                # [FIX-SKIP-REOPEN] 跨段真复用已验证可行。
                self._needs_reopen = True
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

    def _stream_begin(self, force: bool = False):
        """[FIX-STREAM-BEGIN] 开始一个新的编码流（每段一次）。

        重置跨 chunk 持久的 per-segment 状态：slot 未决表、跨块延迟输出缓存。
        对齐 IFRNet v6.4.5.1 的同名方法。
        [FIX-PIPE4-LA8] _slots_warmed per-slot IDR 表已删除（死状态）——
        force_idr 判定改为 fi==0，_slots_warmed = set() 保留仅为历史兼容。

        _sps_pps_injected 的重置由调用方（_NVENCEncodeThread.__init__）负责，
        因为 SPS/PPS 注入涉及 muxer 交互，属于线程层而非编码器层状态。
        """
        if force:
            self._slot_pending.clear()
            self._slots_warmed = set()
            self._prev_chunk_outputs = []
            self._diag_aux_block = 0
            self._diag_phase_shift = 0
            self._diag_slot_drain_fallback = 0
            self._prev_stream_h264 = None

    def _close_driver_session(self):
        """销毁 NVENC 会话、全部 slot(buffer/event) 与专用编码 stream。

        不触碰 DLL/libcuda 句柄与 primary CUDA context（retain 计数在
        close() 中配对释放）。内部方法，调用方须已持有 self._lock。
        所有步骤幂等且防御性执行，允许在 __init__ 中途失败后调用。
        """
        try:
            self._destroy_all_slots()
        except Exception:
            pass
        self._slots = []
        self._input_buf_handle = c_void_p(None)
        self._bs_handle = c_void_p(None)

        # [FIX-ASYNC-COPY] 销毁专用编码拷贝 stream
        if getattr(self, '_stream_encode', None) is not None and \
                self._stream_encode.value is not None:
            try:
                self._libcuda.cuStreamDestroy_v2.restype = c_uint32
                self._libcuda.cuStreamDestroy_v2.argtypes = [c_void_p]
                self._libcuda.cuStreamDestroy_v2(self._stream_encode)
            except Exception:
                pass
            self._stream_encode = c_void_p(None)

        _func_ptrs = getattr(self, '_func_ptrs', None)
        if getattr(self, '_encoder', None) is not None and \
                self._encoder.value is not None and _func_ptrs:
            destroy_addr = _func_ptrs[_FUNC_IDX["DestroyEncoder"]]
            if destroy_addr:
                _NvEncDestroyEncoderProto(destroy_addr)(self._encoder)
        self._encoder = c_void_p(None)

    # [FIX-SKIP-REOPEN] reopen() 已删除。Phase B 验证跨段真复用可行后，
    # 会话重启不再需要。_close_driver_session() 保留供 close() 使用，
    # per-segment 状态重置已移至 _stream_begin()（对齐 IFRNet v6.4.5.1）。

    def close(self):
        with self._lock:
            if getattr(self, '_closed', True):
                return
            self._closed = True

            # [FIX-REOPEN-CTX-ORDER] 销毁前确保当前线程持有 primary context，
            # 理由同 reopen()：_close_driver_session() 内部全是 NVENC/libcuda
            # 调用，缺 context 时静默失败会造成驱动侧资源泄漏。
            _need_pop = False
            if getattr(self, '_primary_ctx', None) is not None and \
                    self._primary_ctx.value is not None:
                try:
                    self._libcuda.cuCtxPushCurrent.restype = c_uint32
                    self._libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
                    _need_pop = (self._libcuda.cuCtxPushCurrent(self._primary_ctx) == 0)
                except Exception:
                    _need_pop = False
            try:
                self._close_driver_session()
            finally:
                if _need_pop:
                    try:
                        _ctx_out = c_void_p()
                        self._libcuda.cuCtxPopCurrent.restype = c_uint32
                        self._libcuda.cuCtxPopCurrent.argtypes = [ctypes.POINTER(c_void_p)]
                        self._libcuda.cuCtxPopCurrent(ctypes.byref(_ctx_out))
                    except Exception:
                        pass
            self._sps_pps_injected = False   # [FIX-SPS-PPS-V2] 跨段重置换，支持 encoder 复用
            print("[NVENCEncoder] Encoder closed", flush=True)

            # [FIX-PRIMARY-CTX-LEAK] 配对 __init__ 的 cuDevicePrimaryCtxRetain。
            # 原实现只 retain 从不 release：单进程生命周期内影响为零，但
            # reopen/重建路径下 retain 计数会持续累积。只释放本对象持有的
            # 那一次 retain，PyTorch 自身的 primary ctx 引用不受影响。
            if getattr(self, '_primary_ctx', None) is not None and \
                    self._primary_ctx.value is not None:
                try:
                    self._libcuda.cuDevicePrimaryCtxRelease.restype = c_uint32
                    self._libcuda.cuDevicePrimaryCtxRelease.argtypes = [c_int]
                    self._libcuda.cuDevicePrimaryCtxRelease(c_int(0))
                except Exception:
                    pass
                self._primary_ctx = c_void_p(None)

            # Restore saved context
            if getattr(self, '_saved_ctx', None) is not None and \
                    self._saved_ctx.value is not None:
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

    def __init__(self, nvenc_encoder, writer, encode_queue_depth: int = 4,
                 flush_chunk_frames: int = 128):  # [FIX-FLUSH-GRANULARITY]
        self._nvenc   = nvenc_encoder
        self._writer  = writer
        # [FIX-LA-QUEUE-DEPTH] 移植自 IFRNet v6.4.5.1：LA>0 时队列深度从 4
        # 提高到 16，降低编码线程 flush 期间 Writer 被阻塞、进而反压 T2
        # 推理的概率。这个前提是：队列里的 NV12 tensor 必须已经 D2H 搬到
        # pinned host 内存（不再占用 VRAM），否则 depth 16 等于把显存缓冲区
        # 放大 4 倍。
        #
        # [FIX-LA-D2H-HOST] 该前提由 NVENCWriter._to_pinned_host_if_la()
        # 落地实现（在 write_frame_batch / _flush_mini_batch 里对齐
        # process_video_v6_4_5_1_single.py 的 [FIX-LA-ACC-HOST-V2]：stack
        # 成一个连续 GPU tensor 后一次性 D2H 拷到预分配 pinned buffer）。
        # 历史教训：此前这里的注释直接声称"已经搬到 pinned host"，但
        # write_frame_batch 实际从未做过 D2H，nv12_list 全程是 GPU tensor，
        # 属于"changelog/注释与代码不符"的静默回归（同类问题见
        # FIX-SLOT-BACKPRESSURE-RESTORE）。submit() 中新增的
        # [FIX-LA-QUEUE-DEPTH-GUARD] 断言用于防止该前提被再次悄悄破坏。
        # LA=0 路径（per-batch CE pipeline，队列中仍是 GPU tensor）保持原
        # depth=4 不变，避免额外占用 VRAM。仅在调用方未显式传入非默认值时生效。
        _la = getattr(nvenc_encoder, '_la_depth', 0)
        if _la > 0 and encode_queue_depth == 4:
            encode_queue_depth = 16
        self._q: queue.Queue = queue.Queue(maxsize=encode_queue_depth)
        # [FIX-FLUSH-GRANULARITY] 移植自 IFRNet v6.4.5.1。原 _LA_CHUNK_SIZE
        # 按内存预算动态计算但上限高达 500 帧，编码线程会把 500 帧的
        # LockInputBuffer/Copy/UnlockInputBuffer/EncodePicture/LockBitstream
        # 压缩成一次持续较长的连续同步脉冲。期间 self._q 很快被 Writer
        # 写满（尤其 LA=0 或未触发上面的队列深度提升时，depth 仅 4），
        # Writer 阻塞在 submit() → 无法消费上游 result_queue → T2 推理
        # 线程被阻塞产帧 → GPU 利用率周期性骤降（与 IFRNet 项目诊断出的
        # 反压机制同根同源，见 _loop() 中的应用位置）。
        # 引入更小的 flush 粒度上限，把大脉冲摊薄成多次小脉冲。
        self._flush_chunk_frames = flush_chunk_frames
        self.error: Optional[Exception] = None
        self._written = 0
        self._empty   = 0
        self._prev_h264: Optional[bytes] = None  # Tier 1-A: 空帧补偿用前一帧 H.264 数据
        # [FIX-STREAM-BEGIN] 对齐 IFRNet v6.4.5.1：per-segment 编码器状态
        # 封装重置（_slot_pending / _slots_warmed / _prev_chunk_outputs）。
        self._nvenc._stream_begin(force=True)
        # SPS/PPS 注入标志属于线程层（涉及 muxer 交互），在此单独重置。
        # 每个新段创建新的 _NVENCEncodeThread + 新 muxer，
        # 确保新 muxer 总能拿到一次 SPS/PPS 预注入。
        if getattr(self._nvenc, '_cached_sps_pps', None) is not None:
            self._nvenc._sps_pps_injected = False
        # [EARLY-FLUSH] 两阶段 flush 的幂等守卫：begin_flush() 置位后
        # flush_and_join() 跳过重复的 q.put(SENTINEL)。每段 _NVENCEncodeThread
        # 新建，标志自然重置为 False。
        self._flush_started = False
        # [FIX-SKIP-REOPEN] 跨段复用驱动会话（已生产验证）。
        # Phase B env 回滚门已移除：_needs_reopen 仅作为 flush()/flush_and_join()
        # 中 EOS 已发出的簿记标记保留（不再被消费），不做会话重建。
        # IFRNet v6.4.5.1 生产零崩溃验证：EOS 后驱动会话可持续使用，
        # Phase D 已移除 FIX-SEG-START-SYNC，段首 batch 直接走 CE pipeline。
        if getattr(self._nvenc, '_needs_reopen', False):
            self._nvenc._needs_reopen = False
            print(f"[FIX-SKIP-REOPEN] 跨段真复用，会话跳过重建 "
                  f"(gen={self._nvenc._session_gen}, frame_idx={self._nvenc._frame_idx})",
                  flush=True)
        self._th = threading.Thread(target=self._loop, daemon=True, name='NVENC-Enc')
        self._th.start()

    def submit(self, nv12_list: list, force_idr_first: bool = False):
        """T3 Writer 线程调用：提交一批 NV12 tensor 给编码线程。

        ⚠️ LA=0 路径：nv12_list 元素为 GPU tensor，调用前必须已完成
        torch.cuda.current_stream().synchronize()，确保 NV12 tensor 写入
        完成，防止编码线程 cuMemcpy2D 读到 GPU 未写完的数据。

        ⚠️ LA>0 路径：nv12_list 元素必须是 pinned host tensor（见
        NVENCWriter._to_pinned_host_if_la() / [FIX-LA-D2H-HOST]），
        因为 __init__ 里的 [FIX-LA-QUEUE-DEPTH] 已经把队列深度从 4 放大
        到 16 —— 这个放大的前提就是队列元素不再占用 VRAM。
        """
        if self.error is not None:
            raise self.error
        # [FIX-LA-QUEUE-DEPTH-GUARD] 防止 FIX-LA-D2H-HOST 被静默破坏：
        # 一旦有人在 LA>0 路径下又把 GPU tensor 直接塞进队列（例如新增了
        # 一条不经过 _to_pinned_host_if_la() 的调用路径），队列深度 16
        # 会让显存占用比预期多出最多 ~4 倍，且不会有任何报错——历史上
        # 这正是 FIX-LA-QUEUE-DEPTH 注释与实现脱节导致高 VRAM 的根因。
        # 用断言在第一时间炸出来，而不是让它以"VRAM 莫名偏高"的形式
        # 在生产环境里悄悄发生。
        if getattr(self._nvenc, '_la_depth', 0) > 0 and nv12_list:
            assert not any(t.is_cuda for t in nv12_list), (
                "[FIX-LA-QUEUE-DEPTH-GUARD] LA>0 时 submit() 收到了仍在 GPU "
                "上的 NV12 tensor —— FIX-LA-QUEUE-DEPTH 把队列深度放大到 16 "
                "的前提（元素已 D2H 到 pinned host）不成立，队列会额外占用 "
                "最多 ~4x VRAM。请检查调用方是否经过 "
                "NVENCWriter._to_pinned_host_if_la()。"
            )
        self._q.put((nv12_list, force_idr_first))

    def _loop(self):
        """编码线程主循环：LA>0 全段累积 + send_eos，LA=0 per-batch ce_pipeline。"""
        # [FIX-ENC-CTX] daemon 线程 CUDA context 绑定
        # [FIX-SEGFAULT-CROSS-SEGMENT] 保留 cuCtxSetCurrent（首段冷启动兼容性好，
        # cuCtxPushCurrent 在 daemon 线程启动时可能因 context 栈状态不可预测而失败）。
        # 关键修复：不再静默忽略 cuCtxSetCurrent 失败——若失败立即 abort，防止后续
        # NVENC API ctypes 调用在无效 context 下访问 GPU 内存 → SIGSEGV。
        # 注意：cuCtxSetCurrent 直接替换当前 context，不 push 到栈，故无需
        # cuCtxPopCurrent 配对（与 encode_frames_batch()/flush() 内部的
        # cuCtxPushCurrent/cuCtxPopCurrent 不会相互干扰——后者自维护栈平衡）。
        _libcuda = getattr(self._nvenc, '_libcuda', None)
        _primary = getattr(self._nvenc, '_primary_ctx', None)
        if _libcuda is not None and _primary is not None and _primary.value is not None:
            try:
                _libcuda.cuCtxSetCurrent.restype  = c_uint32
                _libcuda.cuCtxSetCurrent.argtypes = [c_void_p]
                _r_set = _libcuda.cuCtxSetCurrent(_primary)
                if _r_set != 0:
                    self.error = RuntimeError(
                        f'[NVENC-Enc] cuCtxSetCurrent failed (code={_r_set}), '
                        f'CUDA context may have been invalidated during segment '
                        f'transition; aborting encode thread to prevent SIGSEGV')
                    return
            except Exception as _e:
                self.error = RuntimeError(
                    f'[NVENC-Enc] cuCtxSetCurrent exception: {_e}; '
                    f'aborting encode thread to prevent SIGSEGV')
                return

        # [FIX-SEG2-PREFLIGHT] 段启动预检：确认本线程当前 context 确为 primary
        # ctx，且 encoder 会话句柄有效。段 2+ 首个编码 batch 是 SIGSEGV 高发点，
        # 预检把"无效 context / 无效会话"从不可归因的 SIGSEGV 转化为可诊断的
        # RuntimeError（含 session 代数，便于定位是哪一代会话出问题）。
        _enc_val = getattr(getattr(self._nvenc, '_encoder', None), 'value', None)
        if _enc_val is None:
            self.error = RuntimeError(
                f'[NVENC-Enc] preflight: encoder handle 为空 '
                f'(gen={getattr(self._nvenc, "_session_gen", "?")})，'
                f'会话未建立或已销毁，中止编码线程')
            return
        if _libcuda is not None and _primary is not None and _primary.value is not None:
            try:
                _cur_ctx = c_void_p(None)
                _libcuda.cuCtxGetCurrent.restype = c_uint32
                _libcuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(c_void_p)]
                _r_get = _libcuda.cuCtxGetCurrent(ctypes.byref(_cur_ctx))
                if _r_get != 0 or _cur_ctx.value != _primary.value:
                    self.error = RuntimeError(
                        f'[NVENC-Enc] preflight: 当前 context '
                        f'(0x{(_cur_ctx.value or 0):x}, rc={_r_get}) != primary ctx '
                        f'(0x{_primary.value:x}) (gen={getattr(self._nvenc, "_session_gen", "?")})，'
                        f'中止编码线程以防止 SIGSEGV')
                    return
            except Exception as _e:
                self.error = RuntimeError(
                    f'[NVENC-Enc] preflight cuCtxGetCurrent 异常: {_e}')
                return

        # [FIX-LA-ACCUMULATE] LA>0: cross-batch frame accumulation to preserve LA buffer continuity.
        # Per-batch encoding with local fi%pd slot assignment causes LA buffered frames
        # from one batch to overwrite slots in the next. Fix: accumulate all frames
        # across batches, encode once at SENTINEL with send_eos=True.
        # *** [FIX-CHUNKED-LA] 上述全段累积在超分高分辨率下导致 OOM ***
        # 1536×1152 NV12 ≈ 2.5MB/帧，9000帧 = 22.8GB → 必然 OOM。
        # 改为分块累积 (CHUNK_SIZE=150 ≈ 380MB)，逐块调用 encode_frames_batch(send_eos=False)，
        # 配合持久化 self._slot_pending 保证跨块 drain 映射正确。
        # LA=0: continue per-batch ce_pipeline for async CE performance advantage.
        # [FIX-DYN-CHUNK] 动态 chunk 大小替代硬编码 150:
        # 基于 ~500MB pinned 内存预算，按帧字节数自适应。
        # 低分辨率（如 640×360）自动增大减少调用开销，
        # 高分辨率（如 1536×1152）自动缩小避免 OOM。
        # 初值 150 在收到首帧后被覆盖（同 IFRNet v6.4.5.1 _chunk_frames 逻辑）。
        _LA_CHUNK_SIZE = 150
        # [FIX-PER-CHUNK-DIAG] per-chunk 帧数守恒诊断计数器
        _chunk_submitted = 0  # 本段已提交编码帧数
        _chunk_returned  = 0  # 本段已取回（非空）帧数
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
                # [FIX-CHUNKED-LA] LA>0: 分块累积+逐块编码，避免全段 NV12 tensor 累积 OOM。
                if _first_batch:
                    _acc_force_idr = force_idr
                    # [FIX-DYN-CHUNK] 收到首帧后按帧字节数动态计算 chunk 上限
                    _fb = nv12_list[0].numel() if nv12_list else 0
                    if _fb > 0:
                        # [FIX-FLUSH-GRANULARITY] 内存预算与安全上限(500)之外，
                        # 再与 flush_chunk_frames(默认128) 取最小——防止大脉冲
                        # 反压 T2 推理，同时仍受内存预算保护避免 OOM。
                        _LA_CHUNK_SIZE = min(max(100, min(500, int(5e8 / _fb))),
                                              self._flush_chunk_frames)
                        if not self._nvenc._sps_pps_injected:  # 仅首段打印
                            print(f'\n[NVENC-Enc] LA={self._nvenc._la_depth} 分块编码: '
                                  f'chunk={_LA_CHUNK_SIZE} 帧/块 (动态, ~500MB budget, '
                                  f'flush≤{self._flush_chunk_frames})', flush=True)
                    _first_batch = False
                _acc_nv12.extend(nv12_list)

                if len(_acc_nv12) >= _LA_CHUNK_SIZE:
                    chunk = _acc_nv12  # 传递引用，encode 内逐帧释放
                    _acc_nv12 = []     # 新 list，旧引用仅 chunk 持有 → 编码完成后 GC
                    try:
                        _chunk_submitted += len(chunk)  # [FIX-PER-CHUNK-DIAG]
                        h264_list = self._nvenc.encode_frames_batch(
                            chunk, _acc_force_idr, send_eos=False)
                        _chunk_returned += sum(1 for h in h264_list if h)  # [FIX-PER-CHUNK-DIAG]
                        self._write_la_output(h264_list, is_final=False)
                    except Exception as e:
                        self.error = e
                        return
                    _acc_force_idr = False  # 仅首块发 IDR
            else:
                # LA=0: per-batch ce_pipeline (original path)
                try:
                    # [FIX-SKIP-REOPEN] 跨段真复用已验证可行，段首 batch 直接走 CE pipeline
                    # （对齐 IFRNet v6.4.5.1 LA=0 路径，生产零崩溃）。
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

        # [FIX-CHUNKED-LA] LA>0: 最终块编码 + EOS 排空。
        # 中间块已在上面的 while 循环中通过 send_eos=False 分别编码。
        # 最终块（剩余帧 + EOS）触发全 slot drain，产出所有延迟帧。
        if _la_mode:
            # 插入 stashed f0（ESRGAN 路径下 _pending_f0 总是 None，IFRNet 兼容保留）
            if _pending_f0 is not None:
                _acc_nv12.insert(0, _pending_f0)
                if _f0_idr:
                    _acc_force_idr = True

            try:
                if _acc_nv12:
                    _chunk_submitted += len(_acc_nv12)  # [FIX-PER-CHUNK-DIAG]
                    h264_list = self._nvenc.encode_frames_batch(
                        _acc_nv12, _acc_force_idr, send_eos=True)
                else:
                    # 所有帧已在中间块提交完毕，仅需 EOS drain
                    h264_list = self._nvenc.encode_frames_batch(
                        [], False, send_eos=True)
                _chunk_returned += sum(1 for h in h264_list if h)  # [FIX-PER-CHUNK-DIAG]
                self._write_la_output(h264_list, is_final=True)
                # [FIX-PER-CHUNK-DIAG] 帧数守恒最终校验。
                # _chunk_returned 仅统计 h264_list 内非空条目，漏计了通过
                # _prev_chunk_outputs 跨 chunk 边界的帧。改用 self._written
                # （实际写入文件的帧，包含 _prev_chunk_outputs 帧）+
                # self._pending（LA 缓冲中尚未排空的帧）作为正确输出总数。
                _pending_now = self.__dict__.get('_pending', 0)
                _actual_out = self._written + _pending_now
                if _actual_out != _chunk_submitted:
                    print(f'[NVENC-Enc] ⚠️ 帧数守恒校验失败: 提交 {_chunk_submitted} '
                          f'vs 实际输出 {_actual_out} '
                          f'(written={self._written}, pending={_pending_now}, '
                          f'_chunk_returned={_chunk_returned}) '
                          f'(差 {_chunk_submitted - _actual_out})',
                          flush=True)
            except Exception as e:
                self.error = e
                return

        # EOS flush: LA=0 (ce_pipeline) path residual frame drain.
        # LA>0 path already completed EOS + full-slot drain above.
        if not _la_mode:
            try:
                flush_data = self._nvenc.flush()
                if flush_data:
                    self._writer.write(flush_data)
                    self._written += 1
                    self._prev_h264 = flush_data
            except Exception as _fe:
                # [FIX-RC-CHECK] flush 异常不再静默吞掉：打印出来便于诊断，
                # 但不致命化（段末 flush 失败不应拖垮整段输出）。
                print(f'[NVENC-Enc] ⚠️ EOS flush 异常: {_fe}', flush=True)

    def _write_la_output(self, h264_list, is_final: bool):
        """[FIX-CHUNKED-LA] LA 分块编码输出处理。

        is_final=False: 中间块 — 跳过 None/b"" 条目（LA 缓冲的帧，后续块或 EOS drain 会产出）。
        is_final=True:  最终块 — 空条目用 _prev_h264 fallback（EOS 后应已全部产出，空帧为异常兜底）。
        """
        # [FIX-GLOBAL-FI] 先写入上一个 chunk 延迟产出、本次 encode_frames_batch() 调用中
        # 才被 drain 出来的帧。这些帧在时间顺序上早于 h264_list 中的当前块帧，必须先写。
        _prev = getattr(self._nvenc, '_prev_chunk_outputs', None)
        if _prev:
            for h264_data in _prev:
                if not h264_data:
                    continue
                if not self._nvenc._sps_pps_injected:
                    _sps = getattr(self._nvenc, '_cached_sps_pps', None)
                    if _sps:
                        self._writer.write_sps_pps(_sps)
                        self._nvenc._sps_pps_injected = True
                self._writer.write(h264_data)
                self._written += 1
                self._prev_h264 = h264_data
                # [FIX-AUDIT-PENDING] 这一帧原先在某个非-final chunk 里被计入 _pending
                # （LA 缓冲待回收），现在真正被 drain 出来写入了，配平审计计数。
                if self.__dict__.get('_pending', 0) > 0:
                    self._pending -= 1
            self._nvenc._prev_chunk_outputs = []

        for h264_data in h264_list:
            if h264_data is None or not h264_data:
                if is_final:
                    # 最终块不应有空帧，但有则用前一帧填充
                    self._empty += 1
                    if self._prev_h264 is not None:
                        self._writer.write(self._prev_h264)
                        self._written += 1
                else:
                    # [FIX-AUDIT-PENDING] 非-final chunk 的 b""/None 视为"LA 缓冲中，
                    # 待后续 chunk 或 EOS drain 回收"。之前这里既不计入 written 也不计入
                    # empty，如果因跨段状态错位（如 _frame_idx 时间戳不连续）导致这些帧
                    # 永远无法被回收，会在完全无日志痕迹的情况下静默丢帧。这里显式计入
                    # _pending，最终 flush_and_join() 时校验 written+empty+pending
                    # == submitted，一旦再次出现类似回归能立刻在 NVENC-CLOSED 处报错，
                    # 而不是要靠事后翻 5000 行日志才能定位。
                    self.__dict__.setdefault('_pending', 0)
                    self._pending += 1
                continue
            # Real H.264 data
            if not self._nvenc._sps_pps_injected:
                _sps = getattr(self._nvenc, '_cached_sps_pps', None)
                if _sps:
                    self._writer.write_sps_pps(_sps)
                    self._nvenc._sps_pps_injected = True
            self._writer.write(h264_data)
            self._written += 1
            self._prev_h264 = h264_data

    def begin_flush(self) -> None:
        """[EARLY-FLUSH] 非阻塞启动编码收尾：仅发送 SENTINEL，编码线程收到后
        立即开始最终 chunk 编码 + per-slot EOS drain。主线程立即返回，
        不等待 drain 完成——实现 GPU drain 与 CPU 管线清理真并行。

        幂等：多次调用仅首次生效。编码线程已死亡 / 已出错时 no-op。
        异常/中断路径下未被调用时，flush_and_join() 走原有 put+join 路径兜底。

        安全性依据：_write_frames 返回时所有帧已由主线程同步 q.put 进入
        编码队列，此时发送 SENTINEL 排在队尾，无丢帧风险。
        """
        if self._flush_started:
            return
        if self.error is not None:
            return
        if not self._th.is_alive():
            return  # 编码线程已退出，无需发送

        self._flush_started = True  # 先置位：即使 put 失败也阻止后续重复 put
        try:
            self._q.put(self._SENTINEL, timeout=30)
        except Exception:
            # [EARLY-FLUSH-TOCTOU] put 超时/失败的可能原因：
            # 1. 编码线程在 is_alive() 检查与 put() 之间恰好死亡 → 队列无消费者
            # 2. 编码线程死锁 → 队列满
            # 回退标志，由 flush_and_join() 的阻塞 put 兜底处理。
            self._flush_started = False
            if not self._th.is_alive():
                print(f'[NVENC-Enc] begin_flush: 编码线程在 SENTINEL 发送前退出，'
                      f'由 flush_and_join 兜底', flush=True)
            else:
                print(f'[NVENC-Enc] ⚠️ begin_flush: SENTINEL 发送超时（30s），'
                      f'编码线程可能死锁，由 flush_and_join 兜底重试', flush=True)

    def flush_and_join(self, timeout: float = 120.0):
        """等待编码线程完成所有已提交帧，并返回 (written, empty)。

        编码线程在 _loop() 末尾已执行 NVENC EOS flush。
        [EARLY-FLUSH] 若 begin_flush() 已提前发送 SENTINEL，
        此处跳过 put 仅做 join + 审计；否则走原有 put+join 路径。
        必须在 SENTINEL 处理后、muxer.close() 之前调用。
        """
        if not self._flush_started:
            self._q.put(self._SENTINEL)
        self._th.join(timeout=timeout)
        if self._th.is_alive():
            print(f'[NVENC-Enc] ⚠️ 编码线程未在 {timeout:.0f}s 内退出，可能死锁', flush=True)
        if self.error is not None:
            raise RuntimeError(f'[NVENC-Enc] 编码线程异常: {self.error}') from self.error
        # [FIX-ENC-FLUSH] NVENC EOS flush 已移至编码线程 _loop() 末尾执行，
        # 确保与 encode_frames_batch 在同一 CUDA context。此处不再重复 flush。
        # [FIX-AUDIT-PENDING] segment 结束时 _pending 应已归零（所有非-final chunk 里
        # 缓冲的 LA 帧都应在后续 chunk 或最终 EOS drain 中被回收）。若不为 0，说明有帧
        # 在 _write_la_output 的非-final 分支里被跳过后从未被 _prev_chunk_outputs
        # 机制回收——这正是本次 [FIX-FRAMEIDX-RESET-REGRESSION] 修复前的静默丢帧模式。
        # 这里只做告警不中断流程，避免把审计手段本身变成新的稳定性风险。
        _pending_leftover = self.__dict__.get('_pending', 0)
        if _pending_leftover > 0:
            print(f'[NVENC-Enc] ⚠️⚠️ 检测到 {_pending_leftover} 帧未被回收就结束段编码，'
                  f'很可能存在静默丢帧（跨段状态污染 / LA drain 逻辑异常），'
                  f'请检查 _frame_idx 时间戳连续性与 _slot_pending 状态', flush=True)
        # 段编码全部完成：无论 LA=0(flush) 还是 LA>0(encode_frames_batch
        # send_eos=True)，EOS 均已发出。_needs_reopen 仅作为簿记标记保留，
        # 不再触发会话重建（[FIX-SKIP-REOPEN] 跨段真复用已验证）。
        self._nvenc._needs_reopen = True
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
                 audio_src=None):
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
        pass  # 帧数守恒已确保每段独立排空，无需 is_last_segment

        # Mini-batch accumulator
        self._mini_batch: list = []
        self._first_batch = True
        self._pending_force_idr = False

        # [FIX-LA-D2H-HOST] pin_memory 分配失败时只打印一次告警，避免刷屏。
        self._d2h_pin_warned = False

        if not self._quiet:
            print(f'[NVENCWriter] Ready: {width}x{height}@{fps:.1f}fps, '
                  f'slots={self._enc._slot_count}, la_depth={self._enc._la_depth}', flush=True)

    def _to_pinned_host_if_la(self, nv12_list: list) -> list:
        """[FIX-LA-D2H-HOST] LA>0 时把 NV12 tensor 从 GPU D2H 搬到 pinned
        host 内存，再交给 _NVENCEncodeThread.submit()。

        背景：_NVENCEncodeThread._loop() 在 LA>0 时会做跨批次累积
        （[FIX-CHUNKED-LA]/_acc_nv12）并把编码队列深度放大到 16
        （[FIX-LA-QUEUE-DEPTH]）。这两处放大都是以"队列/累积区里的元素
        不占 VRAM"为前提设计的——如果元素仍是 GPU tensor，几百帧高分辨率
        NV12 数据（单帧 (H+H//2)*W 字节，超分场景下 1536×1152 ≈2.5MB/帧）
        叠加 16 倍队列深度，足以在 SR 推理本身就占用大量显存的情况下把
        T4 16GB 顶满。

        做法对齐 process_video_v6_4_5_1_single.py 的 [FIX-LA-ACC-HOST-V2]：
        先 stack 成一个连续 GPU tensor，再一次性 D2H 拷贝到预分配的
        pinned buffer（而不是逐帧 .cpu()，减少多次小额同步拷贝的开销）。
        pinned 内存分配失败时（例如系统锁页内存额度耗尽）退化为普通
        可分页内存，只牺牲后续 H2D/编码拷贝带宽，不影响正确性。

        LA=0 路径不调用本方法，NV12 tensor 继续留在 GPU 上参与
        encode_frames_batch_ce_pipeline 的零拷贝路径，性能不受影响。
        """
        if getattr(self._enc, '_la_depth', 0) <= 0 or not nv12_list:
            return nv12_list

        _gpu_stack = torch.stack(nv12_list)
        try:
            _pinned = torch.empty(_gpu_stack.shape, dtype=_gpu_stack.dtype,
                                   device='cpu', pin_memory=True)
            _pinned.copy_(_gpu_stack, non_blocking=False)
        except RuntimeError:
            if not self._d2h_pin_warned:
                print('[NVENCWriter] \u26a0\ufe0f pin_memory 分配失败，LA>0 的 '
                      'NV12 D2H 搬运退化为可分页内存（性能略降，功能不受影响）',
                      flush=True)
                self._d2h_pin_warned = True
            _pinned = _gpu_stack.cpu()
        del _gpu_stack
        # 切片视图，逐帧供 encode_frames_batch() 按 [FIX-HOST-SOURCE]
        # 分支（is_cuda=False → srcMemoryType=CU_MEMORYTYPE_HOST）消费。
        return [_pinned[i] for i in range(_pinned.shape[0])]

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
            nv12 = _rgb_to_nv12_gpu(batch_t[i], input_is_bgr=False)
            nv12_list.append(nv12)
        del batch_t

        # [FIX-SYNC-BEFORE-SUBMIT] submit() 契约（_NVENCEncodeThread.submit
        # docstring）要求：提交前必须确保 NV12 GPU tensor 已写完。编码线程在
        # _stream_encode（CU_STREAM_NON_BLOCKING）上做 cuMemcpy2DAsync，与
        # 默认 stream 之间无隐式同步，缺这一步会读到未写完的 NV12 数据（花帧）。
        # 该同步此前只补在从未被 import 的 nvenc_writer.py 副本（死代码）里，
        # 活跃路径（本类）一直缺失。
        torch.cuda.current_stream().synchronize()

        # [FIX-LA-D2H-HOST] LA>0 时在入队前把 NV12 从 GPU 搬到 pinned host，
        # 使 [FIX-LA-QUEUE-DEPTH]（队列深度放大到 16）的前提真正成立，
        # 避免几百帧 GPU 常驻 NV12 数据把显存顶满。LA=0 时原样返回，
        # 零额外开销。
        nv12_list = self._to_pinned_host_if_la(nv12_list)

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
            nv12 = _rgb_to_nv12_gpu(batch_t[i], input_is_bgr=False)
            nv12_list.append(nv12)
        del batch_t

        # [FIX-SYNC-BEFORE-SUBMIT] 同 write_frame_batch()：提交前显式同步
        # 默认 stream，保证编码线程读到的 NV12 数据已写完。
        torch.cuda.current_stream().synchronize()

        # [FIX-LA-D2H-HOST] 同 write_frame_batch()：LA>0 时先落地到 pinned
        # host 再入队，保证 [FIX-LA-QUEUE-DEPTH] 的"队列不占 VRAM"前提成立。
        nv12_list = self._to_pinned_host_if_la(nv12_list)

        force_idr = self._first_batch
        self._enc_thread.submit(nv12_list, force_idr_first=force_idr)
        self._frames_submitted += n
        self._first_batch = False
        self._mini_batch.clear()

    def begin_flush(self):
        """[EARLY-FLUSH] 段末提前触发 NVENC 编码收尾的公共入口。

        pipeline.py 在 _write_frames() 返回后、管线线程清理前调用：
        先把 mini-batch 残留帧提交（单帧 write_frame 路径可能留有
        < _MINI_BATCH 帧），再委托 _NVENCEncodeThread.begin_flush()
        非阻塞发送 SENTINEL，使 GPU EOS drain 与后续 CPU 管线清理并行。

        注意：必须先 flush mini-batch —— SENTINEL 入队后再 submit 的帧
        会排在 SENTINEL 之后，编码线程拿到 SENTINEL 即退出，这些帧会被
        静默丢弃（段尾丢帧）。本方法的语义是"所有帧已提交，可以收尾"。

        幂等：线程端 _flush_started 守卫，重复调用 no-op。
        异常不抛出：close() 中 flush_and_join() 的 put+join 路径兜底。
        PreviewWriter 经 __getattr__ 透明代理到本方法。
        """
        if self._broken:
            return
        if self._enc_thread.error is not None:
            return
        try:
            if self._mini_batch:
                self._flush_mini_batch()
            self._enc_thread.begin_flush()
        except Exception as e:
            print(f'[NVENCWriter] begin_flush 异常（由 close() 兜底）: {e}',
                  flush=True)

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
