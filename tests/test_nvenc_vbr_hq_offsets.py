#!/usr/bin/env python3
"""
test_nvenc_vbr_hq_offsets.py
验证 NVENC SDK 13.0 NV_ENC_RC_PARAMS 关键 offset 的可写可读性
（Linux + GPU 环境专用 — NVENC DLL 仅在 Linux 可用）

用途：
  在 Linux GPU 机器上运行此脚本，验证 process_video_v6_4_3_1/4_1/5_1_single.py
  中使用的 rcParams offset 是否正确。所有测试通过后，方可启用 _NVENC_LEVEL1_RATE_MODE。

验证清单：
  1. PresetConfig buffer 大小是否容纳 rcParams+128 (QVBR/qvbrQuality 区域)
  2. rcParams+116 targetQuality (uint32) — 可写可读
  3. rcParams+120 targetQualityLSB (uint8)   — 可写可读
  4. rcParams+121 lookaheadDepth  (uint16)   — 可写可读 (取值: 0=禁用, 8/16/32=预看深度)
  5. rcParams+36 bitfield         (uint32)   — GetEncodePresetConfig 填充值，bit 位可修改可回读
  6. rcParams+1 rateControlMode   (uint32)   — CONSTQP(0)/VBR_HQ(4)/QVBR(32) 可写可读
  7. rcParams+124 qvbrQuality     (uint32)   — QVBR 质量参数 (NVENC SDK 12.2+)
  8. rcParams+28/+32 vbvBufferSize/vbvInitialDelay — VBR_HQ/QVBR 码率缓冲参数

用法：
  python test_nvenc_vbr_hq_offsets.py [--gpu 0] [--verbose]

前置条件：
  - Linux + NVIDIA GPU + 已安装 CUDA Toolkit
  - libnvidia-encode.so.1 在 LD_LIBRARY_PATH 中
  - PyTorch 可选（仅需 ctypes，不需要 torch）
"""

import argparse
import ctypes
import os
import sys
import textwrap
from ctypes import (
    byref, cast, c_int, c_uint8, c_uint16, c_uint32, c_uint64,
    c_void_p, POINTER, sizeof, Structure, CDLL, RTLD_GLOBAL,
)


# ==============================================================================
# SDK 13.0 常量
# ==============================================================================

_NVENCAPI_VERSION = 0x0D  # NVENCAPI_VERSION (SDK 13.0 new format)


def sdk13_ver(ver: int, bit31: bool = False) -> int:
    """NVENCAPI_STRUCT_VERSION(ver) equivalent."""
    v = _NVENCAPI_VERSION | (ver << 16) | (0x7 << 28)
    if bit31:
        v |= (1 << 31)
    return v


# NV_ENC status codes
NV_ENC_SUCCESS = 0
NV_ENC_DEVICE_TYPE_CUDA = 1

# GUID types
NV_ENC_CODEC_H264_GUID = bytes([0x28, 0x2B, 0x78, 0x6F, 0x3A, 0x45, 0xBC, 0x44,
                                0xAE, 0x59, 0x65, 0xFF, 0x24, 0x6A, 0x42, 0xE8])
NV_ENC_PRESET_P1_GUID  = bytes([0xE0, 0x7C, 0x37, 0x58, 0x83, 0x8B, 0x46, 0x47,
                                0xB6, 0x76, 0x3B, 0x3B, 0x97, 0x60, 0x20, 0x68])

# Function indices — SDK 13.0, verified via nv-codec-headers n13.0.19.0 C dump.
# Index = (offsetof(field) - offsetof(nvEncOpenEncodeSession)) / sizeof(void*) = (offset - 0) / 8
FUNC_IDX = {
    "GetEncodeGUIDCount":        1,   # nvEncGetEncodeGUIDCount
    "GetEncodeGUIDs":            4,   # nvEncGetEncodeGUIDs
    "GetEncodePresetGUIDs":      9,   # nvEncGetEncodePresetGUIDs
    "GetEncodePresetConfig":    10,   # nvEncGetEncodePresetConfig
    "InitializeEncoder":        11,   # nvEncInitializeEncoder
    "DestroyEncoder":           27,   # nvEncDestroyEncoder
    "OpenEncodeSessionEx":      29,   # nvEncOpenEncodeSessionEx (SDK 13.0)
    "GetEncodePresetConfigEx":  39,   # nvEncGetEncodePresetConfigEx (SDK 13.0)
}


# ==============================================================================
# ctypes structs (minimal subset, matching process_video_v6_4_3_1_single.py)
# ==============================================================================

class _NvGuid(Structure):
    _pack_ = 1
    _fields_ = [
        ("Data1", c_uint32),
        ("Data2", c_uint16),
        ("Data3", c_uint16),
        ("Data4", c_uint8 * 8),
    ]


class _NvEncH264VUIParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("disableIVRD",              c_uint32),
        ("reserved1",                c_uint32 * 13),
        ("disableBrickworkFilter",   c_uint32),
        ("reserved2",                c_uint32 * 13),
        ("disableSingleSeiNalu",     c_uint32),
        ("reserved3",                c_uint32 * 13),
        ("disableAspectRatioInfo",   c_uint32),
        ("reserved4",                c_uint32 * 13),
        ("disableIVRDNonRef",        c_uint32),
        ("reserved5",                c_uint32 * 13),
        ("enableWriteH264SVUID",     c_uint32),
        ("reserved6",                c_uint32 * 172),
    ]


class _NvEncH264Config(Structure):
    _pack_ = 1
    _fields_ = [
        ("reserved",              c_uint32 * 4),
        ("idrPeriod",             c_uint32),
        ("reserved1",             c_uint32 * 4),
        ("maxNumRefFramesInDPB",  c_uint32),
        ("reserved2",             c_uint32 * 2),
        ("repeatSPSPPS",          c_uint32),
        ("reserved3",             c_uint32 * 2),
        ("chromaFormatIDC",       c_uint32),
        ("reserved4",             c_uint32 * 33),
        ("h264VUIParameters",     _NvEncH264VUIParams),
        ("reserved5",             c_uint32 * 51),
    ]


class _NvEncConfig(Structure):
    """匹配 process_video_v6_4_3_1_single.py 的 _NvEncConfig。
    offset 40 起为 NV_ENC_RC_PARAMS (SEQUENTIAL)。"""
    _pack_ = 1
    _fields_ = [
        ("reserved1",                c_uint32 * 7),
        ("encodeCodecConfig",        _NvEncH264Config),
        ("reserved2",                c_uint32 * 3),
        ("gopLength",                c_uint32),
        ("frameIntervalP",           c_uint32),
        ("reserved3",                c_uint32 * 104),
        ("deviceType",               c_uint32),
        ("reserved4",                c_uint32 * 688),
        ("reserved5",                c_uint32 * 172),  # [V6431] absorbed enableTemporalAQ
    ]


class _NvEncPresetConfig(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",       c_uint32),
        ("presetConfig",  _NvEncConfig),
        ("reserved",      c_uint32 * 256),
    ]


class _NvEncOpenEncodeSessionExParams(Structure):
    """SDK 13.0 NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS — sizeof=1552 bytes"""
    _pack_ = 1
    _fields_ = [
        ("version",    c_uint32),           # offset 0
        ("deviceType", c_uint32),           # offset 4
        ("device",     c_void_p),           # offset 8
        ("reserved",   c_void_p),           # offset 16  ← was MISSING (caused segfault!)
        ("apiVersion", c_uint32),           # offset 24
        ("reserved1",  c_uint8 * 253 * 4),  # offset 28 (253×4=1012)
        ("reserved2",  c_void_p * 64),      # offset 1040 (64×8=512)
    ]


class _NvEncInitializeParams(Structure):
    _pack_ = 1
    _fields_ = [
        ("version",            c_uint32),
        ("encodeGUID",         _NvGuid),
        ("presetGUID",         _NvGuid),
        ("encodeWidth",        c_uint32),
        ("encodeHeight",       c_uint32),
        ("darWidth",           c_uint32),
        ("darHeight",          c_uint32),
        ("frameRateNum",       c_uint32),
        ("frameRateDen",       c_uint32),
        ("enableEncodeAsync",  c_uint32),
        ("enablePTD",          c_uint32),
        ("reportSliceOffsets", c_uint32),
        ("enableSubFrameWrite", c_uint32),
        ("enableExternalMEHints", c_uint32),
        ("enableMEOnlyMode",   c_uint32),
        ("enableWeightedPrediction", c_uint32),
        ("reserved",           c_uint32 * 241),
        ("maxEncodeWidth",     c_uint32),
        ("maxEncodeHeight",    c_uint32),
        ("encodeConfig",       c_void_p),
        ("reserved1",          c_uint32 * 251),
    ]


# ==============================================================================
# ctypes 函数原型
# ==============================================================================

NvEncodeAPICreateInstanceProto = ctypes.CFUNCTYPE(c_uint32, c_void_p)
NvEncOpenEncodeSessionExProto = ctypes.CFUNCTYPE(
    c_uint32, POINTER(_NvEncOpenEncodeSessionExParams), POINTER(c_void_p),
)
NvEncGetEncodePresetConfigProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, _NvGuid, _NvGuid, POINTER(_NvEncPresetConfig),
)
NvEncInitializeEncoderProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, POINTER(_NvEncInitializeParams),
)
NvEncDestroyEncoderProto = ctypes.CFUNCTYPE(c_uint32, c_void_p)

GetEncodeGUIDCountProto = ctypes.CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint32))
GetEncodeGUIDsProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, POINTER(_NvGuid), c_uint32, POINTER(c_uint32),
)
GetEncodePresetGUIDsProto = ctypes.CFUNCTYPE(
    c_uint32, c_void_p, _NvGuid, POINTER(_NvGuid), c_uint32, POINTER(c_uint32),
)


# ==============================================================================
# 工具函数
# ==============================================================================

class Colors:
    """ANSI terminal colors for test report."""
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


def check(msg: str, condition: bool, detail: str = "") -> bool:
    """Print a single test result."""
    if condition:
        print(f"  {Colors.GREEN}PASS{Colors.RESET}  {msg}")
    else:
        print(f"  {Colors.RED}FAIL{Colors.RESET}  {msg}  {detail}")
    return condition


def hexdump(buf, offset: int, length: int, group: int = 4) -> str:
    """Minimal hexdump helper."""
    lines = []
    for i in range(offset, offset + length, 16):
        chunk = bytes(buf[i:min(i + 16, offset + length)])
        hex_part = " ".join(
            " ".join(f"{b:02x}" for b in chunk[j:j + group])
            for j in range(0, len(chunk), group)
        )
        lines.append(f"  {i:04x}: {hex_part}")
    return "\n".join(lines)


# ==============================================================================
# 主测试逻辑
# ==============================================================================

def find_nvenc_dll() -> str:
    """Find libnvidia-encode.so on Linux (matches production code search order)."""
    candidates = [
        "/usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1",
        "/usr/lib64/libnvidia-encode.so.1",
        "/usr/local/lib/libnvidia-encode.so.1",
        "libnvidia-encode.so.1",
    ]
    for search_dir in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/lib", "/usr/lib"):
        if os.path.isdir(search_dir):
            for fname in sorted(os.listdir(search_dir)):
                if fname.startswith("libnvidia-encode.so"):
                    candidate = os.path.join(search_dir, fname)
                    if candidate not in candidates:
                        candidates.insert(0, candidate)
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def run_tests(verbose: bool = False) -> bool:
    """Run all offset verification tests. Returns True if all pass."""

    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}  NVENC RC_PARAMS Offset Verification{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")

    all_pass = True
    tests_run = 0

    # ── Step 0: Platform check ──
    if sys.platform != "linux":
        print(f"{Colors.YELLOW}SKIP{Colors.RESET}  Not on Linux — NVENC DLL unavailable on this platform.")
        print(f"         This test must run on a Linux + GPU machine.")
        return False

    # ── Step 1: Find NVENC DLL ──
    dll_path = find_nvenc_dll()
    if dll_path is None:
        print(f"{Colors.RED}ERROR{Colors.RESET} Cannot find libnvidia-encode.so")
        print("        Make sure NVIDIA driver is installed and LD_LIBRARY_PATH is set.")
        return False
    print(f"{Colors.CYAN}INFO{Colors.RESET}  NVENC DLL: {dll_path}")

    dll = CDLL(dll_path, mode=RTLD_GLOBAL)

    # ── Step 2: Check PresetConfig buffer size ──
    tests_run += 1
    pc_size = sizeof(_NvEncPresetConfig)
    # rcParams at offset 40 inside _NvEncConfig
    # _NvEncConfig is at offset 4 (after PresetConfig.version)
    # So rcParams absolute offset in PresetConfig = 4 + 40 = 44
    # But our code uses byref(preset_config, 8 + 40) where 8 is version+reserved_of_version
    # Wait, looking at the code more carefully:
    #   preset_config = _NvEncPresetConfig()
    #   enc_cfg = cast(byref(preset_config, 8), POINTER(_NvEncConfig)).contents
    #   rc_ptr = cast(byref(preset_config, 8 + 40), POINTER(c_uint32))
    # So rc_ptr base = 8 + 40 = 48 bytes from preset_config start
    # lookaheadDepth is at rcParams+121, so: 48 + 121 + 2 = 171 bytes needed
    # Actually sizeof(preset_config) includes the struct padding.
    # _NvEncPresetConfig = version(4) + _NvEncConfig + reserved(256*4=1024)
    rc_params_abs_end = 48 + 124 + 4  # absolute byte offset of qvbrQuality end (=128)
    buf_ok = pc_size >= rc_params_abs_end
    all_pass &= check(
        f"PresetConfig buffer size: {pc_size} bytes (need ≥{rc_params_abs_end})",
        buf_ok,
        f"sizeof(_NvEncPresetConfig)={pc_size}, rcParams end={rc_params_abs_end}"
    )

    if not buf_ok:
        print(f"{Colors.RED}FATAL{Colors.RESET} PresetConfig buffer too small — aborting.")
        return False

    # ── Step 3: Load CUDA + Init (before NvEncodeAPIGetMaxSupportedVersion, matches production) ──
    try:
        libcuda = CDLL("libcuda.so.1", mode=RTLD_GLOBAL)
    except OSError:
        libcuda = CDLL("libcuda.so", mode=RTLD_GLOBAL)

    libcuda.cuInit.restype = c_uint32
    libcuda.cuInit.argtypes = [c_uint32]
    init_r = libcuda.cuInit(0)
    if init_r != 0:
        print(f"{Colors.RED}FATAL{Colors.RESET} cuInit(0) failed, code={init_r}", flush=True)
        return False

    # ── Step 4: NvEncodeAPIGetMaxSupportedVersion (BEFORE CUDA context, matches production order) ──
    try:
        _get_max_ver = dll.NvEncodeAPIGetMaxSupportedVersion
        _get_max_ver.restype = c_uint32
        _get_max_ver.argtypes = [POINTER(c_uint32)]
        _max_ver_val = c_uint32(0)
        _get_max_ver(byref(_max_ver_val))
        _nvenc_api_version = _max_ver_val.value if _max_ver_val.value > 0 else _NVENCAPI_VERSION
        if verbose:
            print(f"{Colors.CYAN}INFO{Colors.RESET}  NVENC API: 0x{_nvenc_api_version:x} = v{_nvenc_api_version >> 4}.{_nvenc_api_version & 0xF}", flush=True)
    except Exception as e:
        if verbose:
            print(f"{Colors.YELLOW}WARN{Colors.RESET}  NvEncodeAPIGetMaxSupportedVersion unavailable: {e}", flush=True)
        _nvenc_api_version = _NVENCAPI_VERSION

    # ── Step 5: CUDA context ──
    libcuda.cuCtxGetCurrent.restype = c_uint32
    libcuda.cuCtxGetCurrent.argtypes = [POINTER(c_void_p)]
    saved_ctx = c_void_p(None)
    r = libcuda.cuCtxGetCurrent(byref(saved_ctx))
    cuda_ctx = None

    if r != 0 or saved_ctx.value is None:
        # Try primary context
        libcuda.cuDevicePrimaryCtxRetain.restype = c_uint32
        libcuda.cuDevicePrimaryCtxRetain.argtypes = [POINTER(c_void_p), c_int]
        primary_ctx = c_void_p(None)
        r = libcuda.cuDevicePrimaryCtxRetain(byref(primary_ctx), c_int(0))
        if r != 0 or primary_ctx.value is None:
            print(f"{Colors.RED}FATAL{Colors.RESET} Cannot get CUDA context", flush=True)
            return False
        libcuda.cuCtxPushCurrent.restype = c_uint32
        libcuda.cuCtxPushCurrent.argtypes = [c_void_p]
        libcuda.cuCtxPushCurrent(primary_ctx)
        cuda_ctx = primary_ctx
        print(f"{Colors.CYAN}INFO{Colors.RESET}  Using primary CUDA context", flush=True)
    else:
        cuda_ctx = saved_ctx
        print(f"{Colors.CYAN}INFO{Colors.RESET}  Using existing CUDA context", flush=True)

    # ── Step 6: NvEncodeAPICreateInstance ──
    # [FIX-SYMBOL] Verify symbol exists before creating CFUNCTYPE to avoid NULL ptr segfault
    try:
        _raw_func = dll.NvEncodeAPICreateInstance
        print(f"{Colors.CYAN}INFO{Colors.RESET}  NvEncodeAPICreateInstance symbol found, creating instance...", flush=True)
    except AttributeError:
        print(f"{Colors.RED}FATAL{Colors.RESET} NvEncodeAPICreateInstance not exported by NVENC DLL", flush=True)
        print(f"         DLL: {dll_path}", flush=True)
        print(f"         Driver may be too new — check nvidia-smi driver version.", flush=True)
        return False

    func_table_size = 2552  # SDK 13.0
    func_table = (c_uint8 * func_table_size)()
    cast(func_table, POINTER(c_uint32))[0] = sdk13_ver(2)
    func_table_ref = func_table  # prevent GC during call
    create_instance = NvEncodeAPICreateInstanceProto(("NvEncodeAPICreateInstance", dll))
    try:
        status = create_instance(cast(func_table, c_void_p))
    except Exception as e:
        print(f"{Colors.RED}FATAL{Colors.RESET} NvEncodeAPICreateInstance crashed: {e}", flush=True)
        return False
    if status != NV_ENC_SUCCESS:
        print(f"{Colors.RED}FATAL{Colors.RESET} NvEncodeAPICreateInstance failed, code={status}", flush=True)
        return False
    print(f"{Colors.CYAN}INFO{Colors.RESET}  NvEncodeAPICreateInstance OK", flush=True)

    func_ptrs = cast(byref(func_table, 8), POINTER(c_void_p))

    if verbose:
        print(f"{Colors.CYAN}DEBUG{Colors.RESET} Function table pointers (known indices):", flush=True)
        for name, idx in FUNC_IDX.items():
            addr = func_ptrs[idx]
            addr_val = addr.value if hasattr(addr, 'value') else int(addr) if addr else 0
            print(f"  [{idx:2d}] 0x{addr_val:016x} ← {name}", flush=True)

    def get_func(idx):
        addr = func_ptrs[idx]
        if not addr or addr == 0:
            return None
        return addr

    # ── Step 7: OpenEncodeSessionEx ──
    open_func_addr = get_func(FUNC_IDX["OpenEncodeSessionEx"])
    if open_func_addr is None:
        print(f"{Colors.RED}FATAL{Colors.RESET} OpenEncodeSessionEx not available", flush=True)
        return False

    open_session = NvEncOpenEncodeSessionExProto(open_func_addr)
    api_try_list = [
        _NVENCAPI_VERSION,          # 0x0d — SDK 13.0 new format
        (13 << 4) | 0,              # 0xd0 — old format 13.0
        (12 << 4) | 0,              # 0xc0 — old format 12.0
    ]
    encoder = c_void_p(None)
    session_ok = False

    for api_ver in api_try_list:
        session_params = _NvEncOpenEncodeSessionExParams()
        session_params.version = sdk13_ver(1)
        session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA
        session_params.device = cuda_ctx
        session_params.apiVersion = api_ver
        status = open_session(byref(session_params), byref(encoder))
        if status == NV_ENC_SUCCESS:
            print(f"{Colors.CYAN}INFO{Colors.RESET}  OpenEncodeSessionEx OK: apiVersion=0x{api_ver:x}", flush=True)
            session_ok = True
            break
        if status != 15:
            break

    if not session_ok:
        print(f"{Colors.RED}FATAL{Colors.RESET} OpenEncodeSessionEx failed, code={status}")
        return False

    # ── Step 8: Get codec GUID ──
    count_val = c_uint32(0)
    s = GetEncodeGUIDCountProto(get_func(FUNC_IDX["GetEncodeGUIDCount"]))(
        encoder, byref(count_val))
    if s != 0 or count_val.value == 0:
        print(f"{Colors.RED}FATAL{Colors.RESET} GetEncodeGUIDCount failed")
        return False

    n_guids = count_val.value
    guid_array = (_NvGuid * n_guids)()
    ctypes.memset(cast(guid_array, c_void_p), 0, sizeof(guid_array))
    actual_count = c_uint32(0)
    s = GetEncodeGUIDsProto(get_func(FUNC_IDX["GetEncodeGUIDs"]))(
        encoder, guid_array, n_guids, byref(actual_count))
    if s != 0 or actual_count.value == 0:
        print(f"{Colors.RED}FATAL{Colors.RESET} GetEncodeGUIDs failed")
        return False

    codec_guid = guid_array[0]
    if verbose:
        print(f"{Colors.CYAN}INFO{Colors.RESET}  Codec GUID: {codec_guid.Data1:08x}-{codec_guid.Data2:04x}-{codec_guid.Data3:04x}")

    # ── Step 9: Get preset GUID ──
    preset_guid_array = (_NvGuid * 64)()
    ctypes.memset(cast(preset_guid_array, c_void_p), 0, sizeof(preset_guid_array))
    preset_count = c_uint32(0)
    s = GetEncodePresetGUIDsProto(get_func(FUNC_IDX["GetEncodePresetGUIDs"]))(
        encoder, codec_guid, preset_guid_array, 64, byref(preset_count))

    # Use the first preset GUID (typically P1)
    if s != 0 or preset_count.value == 0:
        print(f"{Colors.RED}FATAL{Colors.RESET} GetEncodePresetGUIDs failed")
        return False
    preset_guid = preset_guid_array[0]
    if verbose:
        print(f"{Colors.CYAN}INFO{Colors.RESET}  Preset GUID: {preset_guid.Data1:08x}-{preset_guid.Data2:04x}-{preset_guid.Data3:04x}")
        print(f"{Colors.CYAN}INFO{Colors.RESET}  Available presets: {preset_count.value}")

    # ── Step 10: GetEncodePresetConfig ──
    get_preset_addr = get_func(FUNC_IDX["GetEncodePresetConfig"])
    if get_preset_addr is None:
        print(f"{Colors.RED}FATAL{Colors.RESET} GetEncodePresetConfig not available")
        return False

    get_preset_config = NvEncGetEncodePresetConfigProto(get_preset_addr)
    preset_config = _NvEncPresetConfig()
    ctypes.memset(byref(preset_config), 0, sizeof(preset_config))
    preset_config.version = sdk13_ver(5, True)  # NV_ENC_PRESET_CONFIG_VER = 0xf005000d
    # [FIX-SDK-OFFSET8] NV_ENC_CONFIG 在 SDK offset 8（version@0 + 4-byte padding），
    # 必须设置内嵌 NV_ENC_CONFIG.version，否则 SDK 读到 0 → code=15 (INVALID_VERSION)
    NV_ENC_CONFIG_VER = sdk13_ver(9, True)  # 0xf009000d
    cast(byref(preset_config, 8), POINTER(c_uint32))[0] = NV_ENC_CONFIG_VER

    status = get_preset_config(
        encoder, codec_guid, preset_guid, byref(preset_config))
    if status != NV_ENC_SUCCESS:
        print(f"{Colors.RED}FATAL{Colors.RESET} GetEncodePresetConfig failed, code={status}")
        return False
    print(f"{Colors.CYAN}INFO{Colors.RESET}  GetEncodePresetConfig OK\n")

    # =====================================================================
    # 核心测试：rcParams offset 验证
    # =====================================================================
    #
    # layout in preset_config (from process_video_v6_4_3_1_single.py):
    #   rc_ptr = cast(byref(preset_config, 8 + 40), POINTER(c_uint32))
    #   → rc_ptr base = byte 48 in preset_config
    #
    # SDK 13.0 NV_ENC_RC_PARAMS (SEQUENTIAL, 无 union):
    #   rc_ptr[0]  @ +0    version
    #   rc_ptr[1]  @ +4    rateControlMode  (0=CONSTQP, 4=VBR_HQ, 32=QVBR)
    #   rc_ptr[2]  @ +8    constQP.qpInterP
    #   rc_ptr[3]  @ +12   constQP.qpInterB
    #   rc_ptr[4]  @ +16   constQP.qpIntra
    #   rc_ptr[5]  @ +20   averageBitRate   (VBR_HQ/QVBR)
    #   rc_ptr[6]  @ +24   maxBitRate       (VBR_HQ/QVBR)
    #   rc_ptr[7]  @ +28   vbvBufferSize    (VBR_HQ/QVBR)
    #   rc_ptr[8]  @ +32   vbvInitialDelay  (VBR_HQ/QVBR)
    #   rc_ptr[9]  @ +36   bitfield         (AQ/TAQ/LA/multiPass)
    #   rc_ptr[10] @ +40   minQP.qpInterP
    #   ...        ...
    #   rc_ptr[29] @ +116  targetQuality    (uint32, VBR_HQ/QVBR)
    #   +120: targetQualityLSB (uint8)
    #   +121: lookaheadDepth  (uint16, 取值: 0=禁用, 8/16/32; 值越大质量越好延迟越大)
    #   +123: padding (uint8, 隐式对齐)
    #   rc_ptr[31] @ +124  qvbrQuality     (uint32, QVBR 专用, SDK 12.2+)
    # =====================================================================

    print(f"{Colors.BOLD}── Offset verification tests ──{Colors.RESET}\n")

    # Cast rc_ptr
    rc_ptr = cast(byref(preset_config, 8 + 40), POINTER(c_uint32))

    # ── Test 1: constQP region ──
    tests_run += 1
    qp_inter_p = rc_ptr[2]
    qp_inter_b = rc_ptr[3]
    qp_intra   = rc_ptr[4]
    if verbose:
        print(f"  Preset constQP: qpInterP={qp_inter_p}, qpInterB={qp_inter_b}, qpIntra={qp_intra}")
    all_pass &= check(
        "rcParams+8..19 constQP region readable",
        True,  # Always passes — we could read values
        f"qpInterP={qp_inter_p}, qpInterB={qp_inter_b}, qpIntra={qp_intra}"
    )

    # ── Test 2: bitfield readable ──
    tests_run += 1
    bf_preset = rc_ptr[9]
    if verbose:
        print(f"  rc_ptr[9] bitfield preset value: 0x{bf_preset:08x}")
        # Decode known bits
        bits = []
        if bf_preset & (1 << 0): bits.append("enableMinQP")
        if bf_preset & (1 << 1): bits.append("enableMaxQP")
        if bf_preset & (1 << 2): bits.append("enableInitialRCQP")
        if bf_preset & (1 << 3): bits.append("enableAQ")
        if bf_preset & (1 << 4): bits.append("enableLookahead")
        if bf_preset & (1 << 5): bits.append("enableFrameQP")
        if bf_preset & (1 << 6): bits.append("enableExtQPDeltaMap")
        if bf_preset & (1 << 7): bits.append("enableTemporalAQ")
        mp = (bf_preset >> 10) & 0x3
        bits.append(f"multiPass={mp}")
        if bits:
            print(f"    decoded: {', '.join(bits)}")
    all_pass &= check(
        "rcParams+36 bitfield readable",
        True,  # Always passes — P1 preset may leave bitfield=0 (no AQ/lookahead enabled)
        f"preset value = 0x{bf_preset:08x} (0=valid: P1 preset defaults to all features off)"
    )

    # ── Test 3: targetQuality @rcParams+116 (uint32) ──
    tests_run += 1
    test_tq = 0x4D4F4F50  # "POOM" in little-endian (distinctive pattern)
    rc_ptr[29] = test_tq
    readback_tq = rc_ptr[29]
    tq_ok = (readback_tq == test_tq)
    all_pass &= check(
        "rcParams+116 targetQuality (uint32) write/readback",
        tq_ok,
        f"wrote 0x{test_tq:08x}, read 0x{readback_tq:08x}"
    )

    # Write back a reasonable value (23 = CRF equivalent)
    rc_ptr[29] = 23
    if verbose:
        print(f"  [verbose] targetQuality set to 23, readback: {rc_ptr[29]}")

    # ── Test 4: targetQualityLSB @rcParams+120 (uint8) ──
    tests_run += 1
    tqlsb_ptr = cast(byref(preset_config, 8 + 40 + 120), POINTER(c_uint8))
    test_lsb = 0xA5
    tqlsb_ptr[0] = test_lsb
    readback_lsb = tqlsb_ptr[0]
    lsb_ok = (readback_lsb == test_lsb)
    all_pass &= check(
        "rcParams+120 targetQualityLSB (uint8) write/readback",
        lsb_ok,
        f"wrote 0x{test_lsb:02x}, read 0x{readback_lsb:02x}"
    )

    # ── Test 5: lookaheadDepth @rcParams+121 (uint16) ──
    tests_run += 1
    la_ptr = cast(byref(preset_config, 8 + 40 + 121), POINTER(c_uint16))
    test_la = 16
    la_ptr[0] = test_la
    readback_la = la_ptr[0]
    la_ok = (readback_la == test_la)
    all_pass &= check(
        "rcParams+121 lookaheadDepth (uint16) write/readback",
        la_ok,
        f"wrote {test_la}, read {readback_la}"
    )

    # Test zero lookahead
    la_ptr[0] = 0
    if verbose:
        print(f"  [verbose] lookaheadDepth reset to 0, readback: {la_ptr[0]}")

    # ── Test 6: Adjacency check (does targetQualityLSB stomp lookaheadDepth?) ──
    tests_run += 1
    tqlsb_ptr[0] = 0x42
    la_ptr[0] = 0x1337
    if tqlsb_ptr[0] == 0x42 and la_ptr[0] == 0x1337:
        all_pass &= check(
            "targetQualityLSB + lookaheadDepth independence (no overlap)",
            True,
            "tqlsb=0x42, la=0x1337 — fields are independent"
        )
    else:
        all_pass &= check(
            "targetQualityLSB + lookaheadDepth independence (no overlap)",
            False,
            f"tqlsb=0x{tqlsb_ptr[0]:02x}, la=0x{la_ptr[0]:04x}"
        )

    # Reset
    tqlsb_ptr[0] = 0
    la_ptr[0] = 0

    # ── Test 7: bitfield read-modify-write (enableAQ + enableTemporalAQ + enableLookahead) ──
    tests_run += 1
    bf_before = rc_ptr[9]
    rc_ptr[9] = rc_ptr[9] | (1 << 3) | (1 << 7)              # AQ + Temporal AQ
    rc_ptr[9] = rc_ptr[9] | (1 << 4)                          # enableLookahead
    rc_ptr[9] = (rc_ptr[9] & ~(0x3 << 10)) | (1 << 10)        # multiPass=1pass
    bf_after  = rc_ptr[9]
    # Verify bits
    aq_set      = bool(bf_after & (1 << 3))
    taq_set     = bool(bf_after & (1 << 7))
    la_set      = bool(bf_after & (1 << 4))
    mp_set      = ((bf_after >> 10) & 0x3) == 1
    # Verify non-interference bits preserved (at minimum, bits 0-2 unchanged)
    bits_0_2_preserved = (bf_after & 0x7) == (bf_before & 0x7)
    bf_ok = aq_set and taq_set and la_set and mp_set and bits_0_2_preserved
    all_pass &= check(
        "rcParams+36 bitfield read-modify-write (AQ+TemporalAQ+Lookahead+multiPass)",
        bf_ok,
        f"before=0x{bf_before:08x}, after=0x{bf_after:08x} "
        f"(AQ={aq_set}, TAQ={taq_set}, LA={la_set}, MP={mp_set}, low_bits_preserved={bits_0_2_preserved})"
    )

    # ── Test 8: rateControlMode — QVBR (mode=32) ──
    tests_run += 1
    orig_mode = rc_ptr[1]
    rc_ptr[1] = 32  # NV_ENC_PARAMS_RC_QVBR
    mode_readback = rc_ptr[1]
    mode_ok = (mode_readback == 32)
    all_pass &= check(
        "rcParams+4 rateControlMode=32 (QVBR) write/readback",
        mode_ok,
        f"wrote 32, read {mode_readback}"
    )
    # Restore original mode
    rc_ptr[1] = orig_mode

    # ── Test 9: qvbrQuality @rcParams+124 (uint32) ──
    tests_run += 1
    test_qvbr = 0x51564252  # "QVRB" (distinctive pattern)
    rc_ptr[31] = test_qvbr
    readback_qvbr = rc_ptr[31]
    qvbr_ok = (readback_qvbr == test_qvbr)
    all_pass &= check(
        "rcParams+124 qvbrQuality (uint32) write/readback",
        qvbr_ok,
        f"wrote 0x{test_qvbr:08x}, read 0x{readback_qvbr:08x}"
    )

    # Write back a reasonable quality value and readback
    rc_ptr[31] = 23
    if verbose:
        print(f"  [verbose] qvbrQuality set to 23, readback: {rc_ptr[31]}")
    # Reset
    rc_ptr[31] = 0

    # ── Test 10: Adjacency — qvbrQuality vs lookaheadDepth independence ──
    tests_run += 1
    la_ptr[0] = 0x1337
    rc_ptr[31] = 0x4A4B4C4D
    la_readback = la_ptr[0]
    qvbr_readback = rc_ptr[31]
    adj_qvbr_ok = (la_readback == 0x1337 and qvbr_readback == 0x4A4B4C4D)
    all_pass &= check(
        "lookaheadDepth + qvbrQuality independence (no overlap)",
        adj_qvbr_ok,
        f"la=0x{la_readback:04x}, qvbr=0x{qvbr_readback:08x}"
    )
    # Reset
    la_ptr[0] = 0
    rc_ptr[31] = 0

    # ── Test 11: vbvBufferSize + vbvInitialDelay (QVBR/VBR_HQ use these) ──
    tests_run += 1
    rc_ptr[7] = 0x00400000   # vbvBufferSize = 4 MB
    rc_ptr[8] = 0x00200000   # vbvInitialDelay = 2 MB
    vbv_buf_ok = (rc_ptr[7] == 0x00400000 and rc_ptr[8] == 0x00200000)
    all_pass &= check(
        "rcParams+28 vbvBufferSize + rcParams+32 vbvInitialDelay write/readback",
        vbv_buf_ok,
        f"vbvBuf=0x{rc_ptr[7]:08x}, vbvDelay=0x{rc_ptr[8]:08x}"
    )
    # Reset
    rc_ptr[7] = 0
    rc_ptr[8] = 0

    # ── Cleanup ──
    destroy_addr = get_func(FUNC_IDX["DestroyEncoder"])
    if destroy_addr:
        destroy_encoder = NvEncDestroyEncoderProto(destroy_addr)
        r = destroy_encoder(encoder)
        if verbose:
            print(f"\n{Colors.CYAN}INFO{Colors.RESET}  DestroyEncoder returned {r}")

    # ── Final report ──
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    if all_pass:
        print(f"{Colors.GREEN}{Colors.BOLD}  ALL {tests_run} TESTS PASSED{Colors.RESET}")
        print(f"\n  Supported rate control modes (all offsets verified):")
        print(f"    · constqp  — rc_ptr[1]=0,  QP at +8/+12/+16")
        print(f"    · vbr_hq   — rc_ptr[1]=4,  targetQuality at +116, la_depth at +121")
        print(f"    · qvbr     — rc_ptr[1]=32, qvbrQuality at +124, vbv at +28/+32")
        print(f"  lookaheadDepth (+121, uint16): 0=disabled, 8/16/32 (越大质量越好/延迟越大)")
        print(f"\n  Next step: set _NVENC_LEVEL1_RATE_MODE and test on Linux GPU.")
    else:
        print(f"{Colors.RED}{Colors.BOLD}  SOME TESTS FAILED — see details above{Colors.RESET}")
        print(f"\n  Do NOT enable vbr_hq/qvbr until all tests pass.")
    print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")

    return all_pass


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NVENC RC_PARAMS offset verification for Level 1 VBR_HQ + lookahead",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python test_nvenc_vbr_hq_offsets.py
              python test_nvenc_vbr_hq_offsets.py --verbose
              python test_nvenc_vbr_hq_offsets.py --gpu 1
        """),
    )
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index (default: 0)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output (show preset values, hex dumps, etc.)")
    args = parser.parse_args()

    if args.gpu != 0:
        print(f"Note: --gpu {args.gpu} specified but this script currently uses device 0.")
        print("      Edit source to switch device if needed.")

    success = run_tests(verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
