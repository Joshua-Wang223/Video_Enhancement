#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 NV_ENC_LOCK_BITSTREAM.outputTimeStamp 回显偏移 - GPU 实测 (2026-08-11)。

背景（生产段1花屏根因链）:
  external/IFRNet/process_video_v6_4_5_1_single.py 的 _drain_outputs_blocking
  返回 est_fi = _output_slot_idx ——「第 N 次 LockBitstream 取回 = 第 N 个提交帧」
  的纯顺序假设。冷会话 LA=8 预热期硬件输出 buffer 重路由使该假设失效 → 数据块
  贴错 fi → 码流呈 9 帧窗口固定错位、frame_num 回退 552 次、段1 仅 19% 帧幸存
  （离线取证: temp/test2/nal_dump_seg000.txt，NAL total=20060，字节完整纯乱序）。

修复方向: 读取 lock struct 中 outputTimeStamp 回显的真实 inputTimeStamp，
fi = ts - ts_base 恢复真实流内帧号，摆脱顺序假设。本脚本在 GPU 上验证
outputTimeStamp 是否可靠回显，并 sweep 确定其在 1544B lock struct 中的偏移。

已验证锚点 (SDK 13.0, memory/nvenc_ctypes_verified_layouts.md):
  version@0, bitfield(doNotWait)@4, outputBitstream@8,
  bitstreamSizeInBytes@36, bitstreamBufferPtr@56

方法: 编码已知 inputTimeStamp = BASE+i 序列 (LA=0 与 LA=8 两种模式)，drain 时
对整个 lock struct 按 u64(步长8) 与 u32(步长4) sweep，寻找能双射回显提交 ts
集合的偏移。三重旁证: (1) 值集合双射命中; (2) 与码流内 frame_num 关联一致
(第 i 帧 frame_num==i，故 ts == BASE + frame_num); (3) 输出乱序下 ts 仍正确。

用法 (Linux 生产 / 任意带 NVENC 的 GPU):
  python tests/diagnose_lockbitstream_timestamp.py            # LA=0 快速验证
  python tests/diagnose_lockbitstream_timestamp.py --la 8     # LA=8 复现生产段1场景
  python tests/diagnose_lockbitstream_timestamp.py --la 8 --frames 24 --sweep-end 256

退出码: 0=命中(找到双射偏移) 1=未命中 2=环境错误。
"""
import argparse
import ctypes
import struct
import sys
import time
from ctypes import (c_uint32, c_uint16, c_uint8, c_int32, c_uint64,
                    c_size_t, c_void_p, POINTER, byref, cast, sizeof,
                    memset, CFUNCTYPE, CDLL)

# ═══════════════════════════════════════════════
# 版本常量（与生产 _sdk13_ver 一致）
# ═══════════════════════════════════════════════
def _v(ver, b31=False):
    return 0x0d | (ver << 16) | (0x7 << 28) | (0x80000000 if b31 else 0)

PRESET_CFG_VER, CONFIG_VER, INIT_VER, PIC_VER, LBS_VER, LIB_VER = _v(5, 1), _v(9, 1), _v(7, 1), _v(7, 1), _v(2, 1), _v(1)
NV12, PIC_FRAME, RC_CONSTQP, RC_VBR_HQ, SUCCESS = 1, 2, 0, 32, 0
NV_ENC_ERR_NEED_MORE_INPUT = 17
BASE = 1000            # inputTimeStamp 基线（避开小整数与指针值）
W, H, FPS = 640, 480, 60
_FI = {"GUIDCnt": 1, "GUIDs": 4, "PGUIDs": 9, "PresetCfg": 10, "InitEnc": 11, "CIn": 12, "DIn": 13,
       "CBs": 14, "DBs": 15, "Encode": 16, "LockBS": 17, "UnlBS": 18, "LockIB": 19, "UnlIB": 20,
       "DestEnc": 27, "Open": 29, "PresetCfgEx": 39}


# ── struct 布局与生产 process_video_v6_4_5_1_single.py 完全一致 ──
class _G(ctypes.Structure):
    _fields_ = [("a", c_uint32), ("b", c_uint16), ("c", c_uint16), ("d", c_uint8 * 8)]


class _H264VUI(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("overscanInfoPresentFlag", c_uint32), ("videoSignalTypePresentFlag", c_uint32),
                ("videoFormat", c_uint32), ("videoFullRangeFlag", c_uint32),
                ("colourDescriptionPresentFlag", c_uint32), ("colourPrimaries", c_uint32),
                ("transferCharacteristics", c_uint32), ("matrixCoefficients", c_uint32),
                ("chromaSampleLocationFlag", c_uint32), ("chromaSampleLocationTop", c_uint32),
                ("chromaSampleLocationBottom", c_uint32), ("bitstreamRestrictionFlag", c_uint32),
                ("reserved", c_uint32 * 16)]


class _H264(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("enableTemporalSVC", c_uint32), ("enableTemporalSVC_1", c_uint32),
                ("profileLevel", c_uint32), ("chromaFormatIDC", c_uint32),
                ("reserved1", c_uint32 * 13), ("maxNumRefFramesInDPB", c_uint32),
                ("reserved2", c_uint32 * 3), ("idrPeriod", c_uint32),
                ("repeatSPSPPS", c_uint32), ("reserved10", c_uint32 * 4),
                ("vuiParameters", _H264VUI), ("reserved12", c_uint32 * 222)]


class _Cfg(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("version", c_uint32), ("profileGUID", _G), ("gopLength", c_uint32),
                ("frameIntervalP", c_uint32), ("frameFieldMode", c_uint32),
                ("enablePTD", c_uint32), ("frameFieldMode_1", c_uint32),
                ("reserved3", c_uint32 * 53), ("mvPrecision", c_uint32),
                ("reserved4", c_uint32 * 27), ("reserved5", c_uint32 * 172),
                ("encodeCodecConfig", _H264), ("reserved7", c_uint32 * 252)]


class _Pcfg(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("version", c_uint32), ("presetConfig", _Cfg),
                ("reserved", c_uint32 * 256)]


class _Init(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("version", c_uint32), ("encodeGUID", _G), ("presetGUID", _G),
                ("encodeWidth", c_uint32), ("encodeHeight", c_uint32),
                ("darWidth", c_uint32), ("darHeight", c_uint32),
                ("frameRateNum", c_uint32), ("frameRateDen", c_uint32),
                ("enableEncodeAsync", c_uint32), ("enablePTD", c_uint32),
                ("bitfield", c_uint32), ("privDataSize", c_uint32), ("reserved_76", c_uint32),
                ("privData", c_void_p), ("encodeConfig", c_void_p),
                ("maxEncodeWidth", c_uint32), ("maxEncodeHeight", c_uint32),
                ("maxMEHintCountsPerBlock", c_uint8 * 32), ("tuningInfo", c_uint32),
                ("bufferFormat", c_uint32), ("numStateBuffers", c_uint32),
                ("outputStatsLevel", c_uint32), ("reserved1", c_uint8 * 1136),
                ("reserved2", c_void_p * 64)]


_NvEncCreate = CFUNCTYPE(c_uint32, c_void_p)
_NvOpen = CFUNCTYPE(c_uint32, POINTER(c_uint8 * 1552), POINTER(c_void_p))
_NvCreate = CFUNCTYPE(c_uint32, c_void_p, POINTER(_Init))
_NvEncode = CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 3360))

_CTX, _CUDA, _DLL, _DONE = c_void_p(None), None, None, False


def _ctx():
    global _CTX, _CUDA, _DLL, _DONE
    if _DONE:
        return
    _DLL = CDLL("nvEncodeAPI64.dll" if sys.platform == "win32" else "libnvidia-encode.so.1")
    _CUDA = CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
    _CUDA.cuInit(0)
    d = c_int32(0)
    _CUDA.cuDeviceGet(byref(d), 0)
    _CUDA.cuDevicePrimaryCtxRetain(byref(_CTX), d)
    _CUDA.cuCtxPushCurrent(_CTX)
    # 验证 CUDA 上下文就绪（不依赖 torch）
    _CUDA.cuMemAlloc_v2.restype = c_uint32
    _CUDA.cuMemAlloc_v2.argtypes = [POINTER(c_void_p), c_size_t]
    _p = c_void_p(None)
    if _CUDA.cuMemAlloc_v2(byref(_p), 64) == 0 and _p.value:
        _CUDA.cuMemFree_v2.restype = c_uint32
        _CUDA.cuMemFree_v2.argtypes = [c_void_p]
        _CUDA.cuMemFree_v2(_p)
    _CUDA.cuMemcpy2D_v2.restype = c_uint32
    _DONE = True
    print("  [GPU context ready]", flush=True)


# ═══════════════════════════════════════════════
# H.264 slice frame_num 解析（从 tests/verify_segment_bitstream_v2.py 移植）
# ═══════════════════════════════════════════════
class _BitReader:
    """RBSP 位读取器（MSB-first），按需去除 emulation prevention bytes。"""

    def __init__(self, data):
        self.raw = data
        self.rbsp = bytearray()
        self.raw_pos = 0
        self.bit_pos = 0
        self._zeros = 0

    def _fill_bytes(self, need_bytes):
        while len(self.rbsp) < need_bytes and self.raw_pos < len(self.raw):
            b = self.raw[self.raw_pos]
            self.raw_pos += 1
            if self._zeros >= 2 and b == 0x03:
                self._zeros = 0
                continue
            if b == 0:
                self._zeros += 1
            else:
                self._zeros = 0
            self.rbsp.append(b)

    def read_bit(self):
        byte_idx = self.bit_pos >> 3
        self._fill_bytes(byte_idx + 1)
        if byte_idx >= len(self.rbsp):
            raise ValueError('RBSP 位越界')
        bit = (self.rbsp[byte_idx] >> (7 - (self.bit_pos & 7))) & 1
        self.bit_pos += 1
        return bit

    def read_bits(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | self.read_bit()
        return v

    def read_ue(self):
        leading = 0
        while self.read_bit() == 0:
            leading += 1
        if leading > 31:
            raise ValueError('ue(v) 溢出')
        return (1 << leading) - 1 + (self.read_bits(leading) if leading else 0)

    def read_se(self):
        """Exp-Golomb se(v)：ue(v) 后按 (v+1)//2 符号展开。"""
        v = self.read_ue()
        if v & 1:
            return (v + 1) // 2
        return -(v // 2)


def _parse_sps(payload):
    br = _BitReader(payload)
    profile_idc = br.read_bits(8)
    br.read_bits(8)
    br.read_bits(8)
    br.read_ue()
    separate_colour_plane = False
    high_profile = profile_idc in (100, 110, 122, 244, 44, 83, 86, 118, 128, 134, 135, 138, 139)
    if high_profile:
        chroma_format_idc = br.read_ue()
        if chroma_format_idc == 3:
            separate_colour_plane = bool(br.read_bit())
        br.read_ue()
        br.read_ue()
        br.read_bit()
        if br.read_bit():
            n_scaling = 8 if chroma_format_idc != 3 else 12
            for _i in range(n_scaling):
                if br.read_bit():
                    _skip_scaling_list(br)
    return br.read_ue() + 4, separate_colour_plane


def _skip_scaling_list(br):
    size = 16 if br.read_ue() == 0 else 64
    last_scale = 8
    next_scale = 8
    for _j in range(size):
        if next_scale != 0:
            delta_scale = br.read_se()
            next_scale = (last_scale + delta_scale + 256) % 256
        last_scale = (next_scale if next_scale != 0 else last_scale)


def _parse_frame_num(es):
    """解析 Annex B ES 首个 VCL slice 的 frame_num。返回 (frame_num, frame_num_bits) 或 None。"""
    frame_num_bits = 8
    separate_colour_plane = False
    offset = 0
    n = len(es)
    while offset < n:
        next_sc3 = es.find(b'\x00\x00\x01', offset)
        if next_sc3 == -1:
            break
        if next_sc3 >= 1 and es[next_sc3 - 1] == 0x00:
            sc_pos = next_sc3 - 1
            sc_len = 4
        else:
            sc_pos = next_sc3
            sc_len = 3
        payload_start = sc_pos + sc_len
        next_sc3 = es.find(b'\x00\x00\x01', payload_start)
        if next_sc3 == -1:
            payload_end = n
        else:
            payload_end = next_sc3 - 1 if (next_sc3 >= 1 and es[next_sc3 - 1] == 0x00) else next_sc3
        payload = es[payload_start:payload_end]
        offset = payload_end
        if len(payload) < 2:
            continue
        nal_type = payload[0] & 0x1F
        if nal_type == 7:
            try:
                frame_num_bits, separate_colour_plane = _parse_sps(payload[1:])
            except Exception:
                pass
        elif nal_type in (1, 5):
            try:
                br = _BitReader(payload[1:])
                br.read_ue()   # first_mb_in_slice
                br.read_ue()   # slice_type
                br.read_ue()   # pic_parameter_set_id
                if separate_colour_plane:
                    br.read_bits(2)
                return br.read_bits(frame_num_bits), frame_num_bits
            except Exception:
                return None, frame_num_bits
    return None, frame_num_bits


# ═══════════════════════════════════════════════
# 核心: 编码已知 ts 序列 + lock struct sweep
# ═══════════════════════════════════════════════
def run(la_depth, n_frames, sweep_end, pipe_depth=4):
    """编码 n_frames 帧 (inputTimeStamp = BASE+i)，drain 时 sweep lock struct。
    返回 (drain_records, submitted_ts, slot_count, err)。
    drain_records: [(drain_order, slot_idx, est_fi, u64_map, u32_map, frame_num)]。
    """
    try:
        _ctx()
    except Exception as e:
        return None, None, None, f"GPU init: {e}"
    libcuda, dll = _CUDA, _DLL
    try:
        mv = c_uint32(0)
        dll.NvEncodeAPIGetMaxSupportedVersion(byref(mv))
        api = mv.value or 0x0d
    except Exception:
        api = 0x0d

    ft = (c_uint8 * 2552)()
    cast(ft, POINTER(c_uint32))[0] = _v(2)
    if _NvEncCreate(("NvEncodeAPICreateInstance", dll))(cast(ft, c_void_p)) != 0:
        return None, None, None, "NvEncodeAPICreateInstance failed"
    f = cast(byref(ft, 8), POINTER(c_void_p))
    gp = lambda i: f[i] if (f[i] and f[i] != 0) else None

    enc = c_void_p(None)
    ok = False
    for av in sorted({0x0d, api, 0xd0, 0xc0}, reverse=True):
        sp = (c_uint8 * 1552)()
        memset(sp, 0, 1552)
        cast(sp, POINTER(c_uint32))[0] = _v(1)
        cast(byref(sp, 4), POINTER(c_uint32))[0] = 1
        cast(byref(sp, 8), POINTER(c_void_p))[0] = _CTX
        cast(byref(sp, 24), POINTER(c_uint32))[0] = av
        if _NvOpen(gp(_FI["Open"]))(sp, byref(enc)) == 0:
            ok = True
            break
    if not ok:
        return None, None, None, "OpenEncodeSessionEx failed"

    _GCN = CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint32))
    _GG = CFUNCTYPE(c_uint32, c_void_p, POINTER(_G), c_uint32, POINTER(c_uint32))
    _PG = CFUNCTYPE(c_uint32, c_void_p, _G, POINTER(_G), c_uint32, POINTER(c_uint32))
    _GPC = CFUNCTYPE(c_uint32, c_void_p, _G, _G, POINTER(_Pcfg))
    cv = c_uint32(0)
    _GCN(gp(_FI["GUIDCnt"]))(enc, byref(cv))
    ga = (_G * cv.value)()
    memset(cast(ga, c_void_p), 0, sizeof(ga))
    ac = c_uint32(0)
    _GG(gp(_FI["GUIDs"]))(enc, ga, cv.value, byref(ac))
    codec = ga[0]
    pga = (_G * 64)()
    memset(cast(pga, c_void_p), 0, sizeof(pga))
    pc = c_uint32(0)
    _PG(gp(_FI["PGUIDs"]))(enc, codec, pga, 64, byref(pc))
    preset = pga[min(4, pc.value - 1)]

    p = _Pcfg()
    memset(byref(p), 0, sizeof(p))
    p.version = PRESET_CFG_VER
    cast(byref(p, 8), POINTER(c_uint32))[0] = CONFIG_VER
    gpc = gp(_FI["PresetCfg"]) or gp(_FI["PresetCfgEx"])
    if _GPC(gpc)(enc, codec, preset, byref(p)) != 0:
        return None, None, None, "GetEncodePresetConfig failed"

    # ── 配置: 与生产对齐（rc 区绝对偏移 presetCfg@8 + rcParams@40）──
    cfg = cast(byref(p, 8), POINTER(_Cfg)).contents
    cfg.gopLength = FPS
    cfg.frameIntervalP = 1
    cfg.encodeCodecConfig.chromaFormatIDC = 1
    cfg.encodeCodecConfig.idrPeriod = FPS
    cfg.encodeCodecConfig.maxNumRefFramesInDPB = 4
    cfg.encodeCodecConfig.repeatSPSPPS = 1
    rc_ptr = cast(byref(p, 8 + 40), POINTER(c_uint32))
    rc_ptr[0] = _v(1)
    if la_depth > 0:
        # VBR_HQ + LA（复现生产段1 场景）
        rc_ptr[1] = RC_VBR_HQ
        rc_ptr[5] = 5000000     # avgBitRate
        rc_ptr[6] = 10000000    # maxBitRate
        rc_ptr[22] = 23         # targetQuality (CRF 23)
        _rc_bf = rc_ptr[9] | (1 << 3) | (1 << 5) | (1 << 8)  # enableAQ + enableLookahead + enableTemporalAQ
        rc_ptr[9] = _rc_bf
        rc_ptr[25] = 0          # multiPass DISABLED
        _la_ptr = cast(byref(p, 8 + 40 + 90), POINTER(c_uint16))
        _la_ptr[0] = la_depth
    else:
        # CONSTQP（生产 constqp 快速路径，零空帧）
        rc_ptr[1] = RC_CONSTQP
        rc_ptr[2] = rc_ptr[3] = rc_ptr[4] = 25
        rc_ptr[9] = rc_ptr[9] | (1 << 3) | (1 << 8)

    slot_count = max(pipe_depth, la_depth + 1)
    print(f"  [cfg] rc={ 'VBR_HQ' if la_depth else 'CONSTQP' } LA={la_depth} slots={slot_count} "
          f"frames={n_frames} ts_base={BASE}", flush=True)

    ip = _Init()
    memset(byref(ip), 0, sizeof(ip))
    ip.version = INIT_VER
    ip.encodeGUID = codec
    ip.presetGUID = preset
    ip.encodeWidth = W
    ip.encodeHeight = H
    ip.darWidth = W
    ip.darHeight = H
    ip.frameRateNum = FPS * 1000
    ip.frameRateDen = 1000
    ip.enablePTD = 1
    ip.encodeConfig = cast(byref(p, 8), c_void_p)
    if _NvCreate(gp(_FI["InitEnc"]))(enc, byref(ip)) != 0:
        return None, None, None, "InitializeEncoder failed"

    def _mkb(idx, ver):
        b = (c_uint8 * 776)()
        memset(b, 0, 776)
        cast(b, POINTER(c_uint32))[0] = ver
        if idx == 12:  # CreateInputBuffer
            cast(byref(b, 4), POINTER(c_uint32))[0] = W
            cast(byref(b, 8), POINTER(c_uint32))[0] = H
            cast(byref(b, 16), POINTER(c_uint32))[0] = NV12
        r = CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 776))(gp(idx))(enc, b)
        if r != 0:
            raise RuntimeError(f"Create buffer fn#{idx} failed, code={r}")
        return cast(byref(b, 24), POINTER(c_void_p))[0] if idx == 12 else cast(byref(b, 16), POINTER(c_void_p))[0]

    slots = []
    for _s in range(slot_count):
        slots.append({'ib': _mkb(12, _v(2)), 'bb': _mkb(14, _v(1))})
    _Li = CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 776))
    _Ui = CFUNCTYPE(c_uint32, c_void_p, c_void_p)
    _Lb = CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8 * 1544))
    _Ub = CFUNCTYPE(c_uint32, c_void_p, c_void_p)
    _En = _NvEncode

    nv12_sz = W * (H + H // 2)
    nv12 = c_void_p(None)
    libcuda.cuMemAlloc_v2(byref(nv12), nv12_sz)
    if not nv12.value:
        return None, None, None, "cuMemAlloc NV12 buffer failed"
    libcuda.cuMemsetD8_v2.restype = c_uint32
    libcuda.cuMemsetD8_v2.argtypes = [c_void_p, c_uint8, c_size_t]
    libcuda.cuMemsetD8_v2(nv12, 0, nv12_sz)

    # ── 提交状态（复现生产 encode_frames_stream 的 slot 轮转 + backpressure）──
    pending = [None] * slot_count          # slot_idx -> fi（该 slot 提交但未 drain 的帧）
    submitted_ts = []                      # 已提交帧的 inputTimeStamp（按提交顺序）
    output_slot_idx = 0
    drain_records = []

    def _submit_ts(i):
        return BASE + i

    def _drain_one():
        """模拟生产 _drain_outputs_blocking（单帧）。返回 (lock_bytes, slot_idx, est_fi, data, status)。"""
        nonlocal output_slot_idx
        slot_idx = output_slot_idx % slot_count
        bs_handle = slots[slot_idx]['bb']
        lr = (c_uint8 * 1544)()
        memset(lr, 0, 1544)
        cast(lr, POINTER(c_uint32))[0] = LBS_VER
        cast(byref(lr, 8), POINTER(c_void_p))[0] = bs_handle
        st = _Lb(gp(_FI["LockBS"]))(enc, lr)
        if st == NV_ENC_ERR_NEED_MORE_INPUT or st != SUCCESS:
            return None
        n = cast(byref(lr, 36), POINTER(c_uint32))[0]
        pv = cast(byref(lr, 56), POINTER(c_void_p))[0]
        pvv = pv if isinstance(pv, int) else (pv.value or 0)
        data = b""
        if n > 0 and pvv:
            data = bytes((c_uint8 * n).from_address(pvv))
        _Ub(gp(_FI["UnlBS"]))(enc, bs_handle)
        est_fi = output_slot_idx
        output_slot_idx += 1
        return (bytes(lr), slot_idx, est_fi, data, st)

    # ── 提交 n_frames 帧 ──
    for i in range(n_frames):
        slot_idx = i % slot_count
        # backpressure: slot 忙则 drain 直到空闲（与生产 _strm_slot_pending 循环一致）
        guard = 0
        while pending[slot_idx] is not None:
            d = _drain_one()
            if d is None:
                guard += 1
                if guard > slot_count * 4:
                    break
                time.sleep(0.001)
                continue
            lr_bytes, s_idx, _est, _data, _st = d
            drain_records.append((lr_bytes, s_idx, _est, _data, _st))
            pending[s_idx] = None
        slot = slots[slot_idx]
        # LockInputBuffer → 拷贝 NV12 → Unlock
        lb = (c_uint8 * 776)()
        memset(lb, 0, 776)
        cast(lb, POINTER(c_uint32))[0] = LIB_VER
        cast(byref(lb, 8), POINTER(c_void_p))[0] = slot['ib']
        if _Li(gp(_FI["LockIB"]))(enc, lb) != 0:
            return None, None, None, f"LockInputBuffer failed (fi={i})"
        mp = cast(byref(lb, 16), POINTER(c_void_p))[0]
        ap = cast(byref(lb, 24), POINTER(c_uint32))[0]
        cp = (c_uint8 * 128)()
        memset(cp, 0, 128)
        cast(byref(cp, 16), POINTER(c_uint32))[0] = 2
        cast(byref(cp, 32), POINTER(c_void_p))[0] = nv12
        cast(byref(cp, 48), POINTER(c_size_t))[0] = W
        cast(byref(cp, 72), POINTER(c_uint32))[0] = 2
        cast(byref(cp, 88), POINTER(c_void_p))[0] = c_void_p(mp)
        cast(byref(cp, 104), POINTER(c_size_t))[0] = ap if ap else W
        cast(byref(cp, 112), POINTER(c_size_t))[0] = W
        cast(byref(cp, 120), POINTER(c_size_t))[0] = H + H // 2
        if libcuda.cuMemcpy2D_v2(cp) != 0:
            return None, None, None, f"cuMemcpy2D failed (fi={i})"
        _Ui(gp(_FI["UnlIB"]))(enc, slot['ib'])

        # EncodePicture: inputTimeStamp = BASE+i（对齐生产 line 2370）
        pic = (c_uint8 * 3360)()
        memset(pic, 0, 3360)
        cast(pic, POINTER(c_uint32))[0] = PIC_VER
        cast(byref(pic, 4), POINTER(c_uint32))[0] = W
        cast(byref(pic, 8), POINTER(c_uint32))[0] = H
        cast(byref(pic, 12), POINTER(c_uint32))[0] = W
        cast(byref(pic, 24), POINTER(c_uint64))[0] = _submit_ts(i)
        cast(byref(pic, 40), POINTER(c_void_p))[0] = slot['ib']
        cast(byref(pic, 48), POINTER(c_void_p))[0] = slot['bb']
        cast(byref(pic, 64), POINTER(c_uint32))[0] = NV12
        cast(byref(pic, 68), POINTER(c_uint32))[0] = PIC_FRAME
        if i == 0:
            cast(byref(pic, 16), POINTER(c_uint32))[0] = 0x2  # NV_ENC_PIC_FLAG_FORCEIDR
        # CE: LA>0 同步（生产 FIX-LA-SYNC），LA=0 带 CE（生产 FIX-CONSTQP-FRAME-CE）
        _ep_ce = c_void_p(None)
        if la_depth == 0:
            libcuda.cuEventCreate.restype = c_uint32
            libcuda.cuEventCreate.argtypes = [POINTER(c_void_p), c_uint32]
            if libcuda.cuEventCreate(byref(_ep_ce), 0) == 0:
                cast(byref(pic, 56), POINTER(c_void_p))[0] = _ep_ce
        r = _En(gp(_FI["Encode"]))(enc, pic)
        # [FIX-DIAG-LA-NEED-MORE-INPUT] LA 模式下前 la_depth 帧 EncodePicture 返回
        # NEED_MORE_INPUT(code=17) 是正常行为（帧已被 LA 缓冲接受入队，仅暂不产出），
        # 不是失败；生产 encode_frames_stream 对此有专门处理（line 2413）。其他错误码
        # 才是真实失败。
        if r != 0 and r != NV_ENC_ERR_NEED_MORE_INPUT:
            return None, None, None, f"EncodePicture failed (fi={i}, code={r})"
        if _ep_ce.value is not None:
            libcuda.cuEventSynchronize.restype = c_uint32
            libcuda.cuEventSynchronize.argtypes = [c_void_p]
            libcuda.cuEventSynchronize(_ep_ce)
            libcuda.cuEventDestroy.restype = c_uint32
            libcuda.cuEventDestroy.argtypes = [c_void_p]
            libcuda.cuEventDestroy(_ep_ce)
        submitted_ts.append(_submit_ts(i))
        pending[slot_idx] = i

    # ── EOS + 全部排空 ──
    eos = (c_uint8 * 3360)()
    memset(eos, 0, 3360)
    cast(eos, POINTER(c_uint32))[0] = PIC_VER
    cast(byref(eos, 16), POINTER(c_uint32))[0] = 0x8
    cast(byref(eos, 40), POINTER(c_void_p))[0] = c_void_p(None)
    cast(byref(eos, 48), POINTER(c_void_p))[0] = slots[0]['bb']
    _En(gp(_FI["Encode"]))(enc, eos)
    _miss = 0
    for _ in range(6 * slot_count + 8):
        d = _drain_one()
        if d is None:
            _miss += 1
            if _miss >= 4:
                break
            time.sleep(0.001)
            continue
        lr_bytes, s_idx, _est, _data, _st = d
        drain_records.append((lr_bytes, s_idx, _est, _data, _st))
        _miss = 0

    # ── 清理 ──
    for s in slots:
        try:
            CFUNCTYPE(c_uint32, c_void_p, c_void_p)(gp(_FI["DBs"]))(enc, s['bb'])
        except Exception:
            pass
        try:
            CFUNCTYPE(c_uint32, c_void_p, c_void_p)(gp(_FI["DIn"]))(enc, s['ib'])
        except Exception:
            pass
    if nv12.value:
        libcuda.cuMemFree_v2.restype = c_uint32
        libcuda.cuMemFree_v2.argtypes = [c_void_p]
        libcuda.cuMemFree_v2(nv12)
    CFUNCTYPE(c_uint32, c_void_p)(gp(_FI["DestEnc"]))(enc)

    # ── 组装记录: 每 drain 帧扫描 u64/u32 + 解析 frame_num ──
    parsed = []
    for order, (lr_bytes, s_idx, est, data, st) in enumerate(drain_records):
        u64_map = {}
        u32_map = {}
        for off in range(0, sweep_end + 1 - 8, 8):
            u64_map[off] = struct.unpack_from('<Q', lr_bytes, off)[0]
        for off in range(0, sweep_end + 1 - 4, 4):
            u32_map[off] = struct.unpack_from('<I', lr_bytes, off)[0]
        fn, _fb = _parse_frame_num(data) if data else (None, 8)
        parsed.append({'order': order, 'slot': s_idx, 'est': est,
                       'u64': u64_map, 'u32': u32_map, 'frame_num': fn})
    return parsed, submitted_ts, slot_count, None


def analyze(records, submitted_ts, sweep_end):
    """分析 sweep 结果，找能双射回显提交 ts 的偏移。"""
    submitted_set = set(submitted_ts)
    n_drain = len(records)
    # LA 模式 drain 含无 VCL 辅助块（独立 SPS/PPS/AUD，ts=0/重复、不占 fi），
    # 判定前先过滤出真实图像帧子集：frame_num 已由 _parse_frame_num 解析，辅助块为 None。
    # 这与生产 _nal_first_vcl_type 的「无 VCL 辅助块不占 fi、不进 seen」策略同源。
    vcl_records = [r for r in records if r['frame_num'] is not None]
    # ── u64 sweep ──
    hits64 = []
    for off in range(0, sweep_end + 1 - 8, 8):
        vals = [r['u64'][off] for r in vcl_records]
        uniq = set(vals)
        hit = uniq & submitted_set
        # 双射 = VCL 帧 ts 唯一集合 == 提交 ts 集合（等集即覆盖全且无多余 VCL ts；
        # 多 slice 同帧 ts 重复经 set 去重兼容）。不要求 n_drain==提交数
        # （LA 预热期辅助块恒使该计数等式失败——tt6 实测 drain 71>18、127>34）。
        bij = (len(uniq) == len(submitted_set) and uniq == submitted_set)
        hits64.append((off, n_drain, len(uniq), len(hit), bij))
    # ── u32 sweep ──
    hits32 = []
    for off in range(0, sweep_end + 1 - 4, 4):
        vals = [r['u32'][off] for r in vcl_records]
        uniq = set(vals)
        hit = uniq & submitted_set
        bij = (len(uniq) == len(submitted_set) and uniq == submitted_set)
        hits32.append((off, n_drain, len(uniq), len(hit), bij))
    return hits64, hits32


def print_report(records, submitted_ts, sweep_end, la_depth, slot_count):
    hits64, hits32 = analyze(records, submitted_ts, sweep_end)
    # drain 总数含 LA 预热期无 VCL 辅助块（独立 SPS/PPS/AUD），实际图像帧 = VCL 数
    n_vcl = sum(1 for r in records if r['frame_num'] is not None)
    n_aux = len(records) - n_vcl
    print("=" * 78)
    print(f"sweep 报告 | LA={la_depth} slots={slot_count} 提交={len(submitted_ts)} 帧 "
          f"drain={len(records)} 帧 (VCL={n_vcl} 辅助块={n_aux}) "
          f"ts 范围={BASE}..{BASE+len(submitted_ts)-1}")
    print("=" * 78)
    bij64 = [h for h in hits64 if h[4]]
    hit64 = [h for h in hits64 if h[3] > 0 and not h[4]]
    print(f"[u64] 完全双射偏移: {[h[0] for h in bij64]}")
    if hit64:
        print(f"[u64] 部分命中（top10）:")
        for off, n, nu, nh, bij in sorted(hit64, key=lambda h: -h[3])[:10]:
            print(f"      @{off:>4}  n={n} 唯一={nu} 命中={nh}/{len(submitted_ts)}")
    else:
        print("[u64] 无部分命中")
    bij32 = [h for h in hits32 if h[4]]
    hit32 = [h for h in hits32 if h[3] > 0 and not h[4]]
    print(f"[u32] 完全双射偏移: {[h[0] for h in bij32]}")
    if hit32:
        print(f"[u32] 部分命中（top10）:")
        for off, n, nu, nh, bij in sorted(hit32, key=lambda h: -h[3])[:10]:
            print(f"      @{off:>4}  n={n} 唯一={nu} 命中={nh}/{len(submitted_ts)}")
    else:
        print("[u32] 无部分命中")

    # ── 命中的详细验证 ──
    cand = []
    if bij64:
        cand.append(('u64', bij64[0][0]))
    if bij32 and (not bij64 or bij32[0][0] != bij64[0][0]):
        cand.append(('u32', bij32[0][0]))
    for kind, off in cand:
        print(f"\n── 验证 [{kind}] @{off}（outputTimeStamp 候选）──")
        print(f"   {'drain#':>6} {'slot':>4} {'est_fi':>6} {'ts_val':>8} {'→fi':>5} {'frame_num':>9}  一致?")
        # 辅助块（frame_num=None）不占 fi、ts=0 是合法回显值，标注 (aux,skip) 不参与一致性比对
        vcl_records = [r for r in records if r['frame_num'] is not None]
        for r in records:
            val = r['u64'][off] if kind == 'u64' else r['u32'][off]
            fn = r['frame_num']
            mapped = val - BASE
            match = (mapped == fn) if fn is not None else None
            fn_disp = 'aux' if fn is None else str(fn)
            mark = '✓' if match else ('(aux,skip)' if fn is None else '✗')
            print(f"   {r['order']:>6} {r['slot']:>4} {r['est']:>6} {val:>8} {mapped:>5} {fn_disp:>9}   "
                  f"{mark}")
        # 顺序统计: est_fi 顺序 vs ts 顺序（仅 VCL 帧，辅助块不占 fi）
        est_seq = [r['est'] for r in vcl_records]
        ts_seq = [(r['u64'][off] if kind == 'u64' else r['u32'][off]) - BASE for r in vcl_records]
        print(f"\n   est_fi 序列(前12, 仅VCL): {est_seq[:12]}")
        print(f"   ts-fi  序列(前12, 仅VCL): {ts_seq[:12]}")
        print(f"   est_fi 乱序?: {est_seq != sorted(est_seq)} | ts-fi 乱序?: {ts_seq != sorted(ts_seq)}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser(description='NV_ENC_LOCK_BITSTREAM.outputTimeStamp 偏移 sweep 诊断')
    ap.add_argument('--la', type=int, default=0, help='lookahead depth (0=CONSTQP, 8=复现生产段1)')
    ap.add_argument('--frames', type=int, default=16, help='提交帧数 (建议 LA 模式 >= slot_count*2)')
    ap.add_argument('--sweep-end', type=int, default=256, help='lock struct sweep 上限偏移（默认 256）')
    args = ap.parse_args()

    print("=" * 78)
    print(f"LOCK_BITSTREAM.outputTimeStamp 回显诊断 | LA={args.la} | {W}x{H}@{FPS}fps | "
          f"ts_base={BASE} | sweep 0..{args.sweep_end}")
    print("=" * 78)
    n_frames = args.frames
    if args.la > 0 and n_frames < (args.la + 1) * 2:
        n_frames = (args.la + 1) * 2
        print(f"[NOTE] LA 模式帧数提升至 {n_frames}（>= slot_count*2）")
    records, submitted_ts, slot_count, err = run(args.la, n_frames, args.sweep_end)
    if err:
        print(f"运行失败: {err}")
        sys.exit(2)
    print_report(records, submitted_ts, args.sweep_end, args.la, slot_count)

    # 判定
    hits64, hits32 = analyze(records, submitted_ts, args.sweep_end)
    if any(h[4] for h in hits64) or any(h[4] for h in hits32):
        print("\n✅ 结论: 找到 outputTimeStamp 双射回显偏移（ts 重关联可行）")
        sys.exit(0)
    print("\n❌ 结论: 未找到双射偏移（outputTimeStamp 回显不可靠 → 需 Plan B: frame_num 解析重关联）")
    sys.exit(1)


if __name__ == '__main__':
    main()
