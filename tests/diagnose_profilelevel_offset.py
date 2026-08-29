#!/usr/bin/env python3
"""诊断 NV_ENC_CONFIG_H264.profileLevel 偏移 - GPU 实测 (2026-08-07)。

背景: 60fps 时 NVENC 自动选 Level 6.1 > T4 NVDEC 上限 5.2，生产修复在
encodeCodecConfig.profileLevel 写 51 (process_video_v6_4_5_1_single.py line 1313)。

偏移推导（SDK 13.0 nvEncodeAPI.h 逐字节验证, 2026-08-07）:
  NV_ENC_PRESET_CONFIG{version@0, presetCfg@8}
  NV_ENC_CONFIG{version@0, profileGUID@4, gopLength@20, frameIntervalP@24,
    monoChromeEncoding@28, frameFieldMode@32, mvPrecision@36,
    rcParams@40(128B) => encodeCodecConfig@168}      ← 非生产 ctypes 假设的 @1052
  NV_ENC_CONFIG_H264{bitfield(enableTemporalSVC..reservedBitFields)@0,
    level@4, idrPeriod@8, separateColourPlaneFlag@12, ...}
  => 理论绝对偏移 = 8 + 168 + 4 = 180。
  [FIX-DIAG-SDK13] 生产 _NvEncConfig.encodeCodecConfig@1052 是错误布局
  (reserved3[53] 覆盖 rcParams 区 + reserved5[172] 推至 1052)，
  经结构体写入 profileLevel=51 实际落在 NV_ENC_CONFIG.reserved[278] 保留区 → 驱动忽略。

验证: 候选偏移写 level=51 编码 8 帧，解析输出码流 SPS level_idc==51 => 命中。
用法: python diagnose_profilelevel_offset.py [--sweep]
"""
import ctypes, sys, time
from ctypes import (c_uint32, c_uint16, c_uint8, c_int32, c_uint64,
                    c_size_t, c_void_p, POINTER, byref, cast, sizeof,
                    memset, CFUNCTYPE, CDLL)

def _v(ver, b31=False):
    return 0x0d | (ver << 16) | (0x7 << 28) | (0x80000000 if b31 else 0)

PRESET_CFG_VER, CONFIG_VER, INIT_VER, PIC_VER, LBS_VER, LIB_VER = _v(5,1),_v(9,1),_v(7,1),_v(7,1),_v(2,1),_v(1)
NV12, PIC_FRAME, RC_VBR_HQ, SUCCESS = 1, 2, 32, 0
WRITE_LEVEL, W, H, FPS, N = 51, 640, 480, 60, 8
_FI = {"GUIDCnt":1,"GUIDs":4,"PGUIDs":9,"PresetCfg":10,"InitEnc":11,"CIn":12,"DIn":13,
       "CBs":14,"DBs":15,"Encode":16,"LockBS":17,"UnlBS":18,"LockIB":19,"UnlIB":20,
       "DestEnc":27,"Open":29,"PresetCfgEx":39}

# ── 以下 struct 布局与 process_video_v6_4_5_1_single.py 完全一致（生产实测验证）──
class _G(ctypes.Structure):
    _fields_ = [("a", c_uint32),("b", c_uint16),("c", c_uint16),("d", c_uint8*8)]

class _H264VUI(ctypes.Structure):
    """生产 _NvEncConfigH264VUIParameters (line 801-817): 12 字段 + reserved[16]"""
    _pack_ = 1
    _fields_ = [("overscanInfoPresentFlag", c_uint32),("videoSignalTypePresentFlag", c_uint32),
                ("videoFormat", c_uint32),("videoFullRangeFlag", c_uint32),
                ("colourDescriptionPresentFlag", c_uint32),("colourPrimaries", c_uint32),
                ("transferCharacteristics", c_uint32),("matrixCoefficients", c_uint32),
                ("chromaSampleLocationFlag", c_uint32),("chromaSampleLocationTop", c_uint32),
                ("chromaSampleLocationBottom", c_uint32),("bitstreamRestrictionFlag", c_uint32),
                ("reserved", c_uint32 * 16)]

class _H264(ctypes.Structure):
    """生产 _NvEncConfigH264: enableTemporalSVC@0, enableTemporalSVC_1@4, profileLevel@8, chroma@12"""
    _pack_ = 1
    _fields_ = [("enableTemporalSVC", c_uint32),("enableTemporalSVC_1", c_uint32),
                ("profileLevel", c_uint32),("chromaFormatIDC", c_uint32),
                ("reserved1", c_uint32 * 13),("maxNumRefFramesInDPB", c_uint32),
                ("reserved2", c_uint32 * 3),("idrPeriod", c_uint32),
                ("repeatSPSPPS", c_uint32),("reserved10", c_uint32 * 4),
                ("vuiParameters", _H264VUI),("reserved12", c_uint32 * 222)]

class _Cfg(ctypes.Structure):
    """生产 _NvEncConfig: rcParams@40(128B), encodeCodecConfig@1052"""
    _pack_ = 1
    _fields_ = [("version", c_uint32),("profileGUID", _G),("gopLength", c_uint32),
                ("frameIntervalP", c_uint32),("frameFieldMode", c_uint32),
                ("enablePTD", c_uint32),("frameFieldMode_1", c_uint32),
                ("reserved3", c_uint32 * 53),("mvPrecision", c_uint32),
                ("reserved4", c_uint32 * 27),("reserved5", c_uint32 * 172),
                ("encodeCodecConfig", _H264),("reserved7", c_uint32 * 252)]

class _Pcfg(ctypes.Structure):
    """生产 _NvEncPresetConfig (line 866-872): version@0, presetCfg@8 (SDK 布局；
    ctypes pack=1 放 @4，生产代码一律 cast(byref(p, 8)) 按 SDK 布局访问)，
    尾部 reserved[256] 必须保留——GetEncodePresetConfig 驱动按完整 SDK 结构写入，
    缺少尾部会堆越界写导致 segfault。sizeof = 4196 与生产一致。"""
    _pack_ = 1
    _fields_ = [("version", c_uint32),("presetConfig", _Cfg),
                ("reserved", c_uint32 * 256)]

class _Init(ctypes.Structure):
    """生产 _NvEncInitializeParams: encodeConfig@88"""
    _pack_ = 1
    _fields_ = [("version", c_uint32),("encodeGUID", _G),("presetGUID", _G),
                ("encodeWidth", c_uint32),("encodeHeight", c_uint32),
                ("darWidth", c_uint32),("darHeight", c_uint32),
                ("frameRateNum", c_uint32),("frameRateDen", c_uint32),
                ("enableEncodeAsync", c_uint32),("enablePTD", c_uint32),
                ("bitfield", c_uint32),("privDataSize", c_uint32),("reserved_76", c_uint32),
                ("privData", c_void_p),("encodeConfig", c_void_p),
                ("maxEncodeWidth", c_uint32),("maxEncodeHeight", c_uint32),
                ("maxMEHintCountsPerBlock", c_uint8 * 32),("tuningInfo", c_uint32),
                ("bufferFormat", c_uint32),("numStateBuffers", c_uint32),
                ("outputStatsLevel", c_uint32),("reserved1", c_uint8 * 1136),
                ("reserved2", c_void_p * 64)]

_NvEncCreate = CFUNCTYPE(c_uint32, c_void_p)
# 生产 _NvEncOpenEncodeSessionExParams sizeof=1552: version@0, deviceType@4, device@8,
# reserved@16, apiVersion@24, reserved1[253*4], reserved2[64*8]
_NvOpen = CFUNCTYPE(c_uint32, POINTER(c_uint8*1552), POINTER(c_void_p))
_NvCreate = CFUNCTYPE(c_uint32, c_void_p, POINTER(_Init))
_NvEncode = CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8*3360))  # pic raw 3360B

_CTX, _CUDA, _DLL, _DONE = c_void_p(None), None, None, False

def _ctx():
    global _CTX, _CUDA, _DLL, _DONE
    if _DONE: return
    _DLL = CDLL("nvEncodeAPI64.dll" if sys.platform == "win32" else "libnvidia-encode.so.1")
    _CUDA = CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
    _CUDA.cuInit(0)
    d = c_int32(0); _CUDA.cuDeviceGet(byref(d), 0)
    _CUDA.cuDevicePrimaryCtxRetain(byref(_CTX), d)
    _CUDA.cuCtxPushCurrent(_CTX)
    # [FIX-DIAG-NOTORCH] 去掉 torch 依赖：分配/释放一个 64B device buffer
    # 验证 CUDA 上下文就绪（替代 torch.randn 触发 context 的方式）。
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

def run(off=None):
    """off: 绝对字节偏移写 WRITE_LEVEL; None=baseline。返回 (SPS level_idc 列表, err)"""
    try: _ctx()
    except Exception as e: return None, f"GPU init: {e}"
    libcuda, dll = _CUDA, _DLL
    try:
        mv = c_uint32(0); dll.NvEncodeAPIGetMaxSupportedVersion(byref(mv)); api = mv.value or 0x0d
    except Exception: api = 0x0d
    ft = (c_uint8 * 2552)()
    cast(ft, POINTER(c_uint32))[0] = _v(2)
    _s = _NvEncCreate(("NvEncodeAPICreateInstance", dll))(cast(ft, c_void_p))
    if _s != 0: return None, f"NvEncodeAPICreateInstance failed, code={_s}"
    f = cast(byref(ft, 8), POINTER(c_void_p))
    def gp(i): return f[i] if (f[i] and f[i] != 0) else None
    enc = c_void_p(None); ok = False
    for av in sorted({0x0d, api, 0xd0, 0xc0}, reverse=True):
        sp = (c_uint8 * 1552)(); memset(sp, 0, 1552)
        cast(sp, POINTER(c_uint32))[0] = _v(1)
        cast(byref(sp, 4), POINTER(c_uint32))[0] = 1     # encoderDeviceType=GPU
        cast(byref(sp, 8), POINTER(c_void_p))[0] = _CTX  # device
        cast(byref(sp, 24), POINTER(c_uint32))[0] = av   # apiVersion@24 (SDK 布局)
        if _NvOpen(gp(_FI["Open"]))(sp, byref(enc)) == 0: ok = True; break
    if not ok: return None, "OpenEncodeSessionEx failed"

    _GCN = CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint32))
    _GG = CFUNCTYPE(c_uint32, c_void_p, POINTER(_G), c_uint32, POINTER(c_uint32))
    _PG = CFUNCTYPE(c_uint32, c_void_p, _G, POINTER(_G), c_uint32, POINTER(c_uint32))
    _GPC = CFUNCTYPE(c_uint32, c_void_p, _G, _G, POINTER(_Pcfg))
    cv = c_uint32(0); _GCN(gp(_FI["GUIDCnt"]))(enc, byref(cv))
    ga = (_G * cv.value)(); memset(cast(ga, c_void_p), 0, sizeof(ga))
    ac = c_uint32(0); _GG(gp(_FI["GUIDs"]))(enc, ga, cv.value, byref(ac))
    codec = ga[0]
    pga = (_G * 64)(); memset(cast(pga, c_void_p), 0, sizeof(pga))
    pc = c_uint32(0); _PG(gp(_FI["PGUIDs"]))(enc, codec, pga, 64, byref(pc))
    preset = pga[min(4, pc.value - 1)]

    p = _Pcfg(); memset(byref(p), 0, sizeof(p))
    p.version = PRESET_CFG_VER
    cast(byref(p, 8), POINTER(c_uint32))[0] = CONFIG_VER
    gpc = gp(_FI["PresetCfg"]) or gp(_FI["PresetCfgEx"])
    if _GPC(gpc)(enc, codec, preset, byref(p)) != 0: return None, "GetEncodePresetConfig failed"

    # ── 配置（与生产 __init__ 对齐；profileLevel 留给 off sweep 写）──
    cfg = cast(byref(p, 8), POINTER(_Cfg)).contents
    cfg.gopLength = FPS
    cfg.frameIntervalP = 1
    cfg.encodeCodecConfig.chromaFormatIDC = 1
    cfg.encodeCodecConfig.idrPeriod = FPS
    cfg.encodeCodecConfig.maxNumRefFramesInDPB = 4
    cfg.encodeCodecConfig.repeatSPSPPS = 1
    rc = cast(byref(p, 48), POINTER(c_uint32))  # rcParams@40 in cfg => 8+40=48
    # [FIX-DIAG-RC] RC 改 CONSTQP：生产 encode_frames_stream 默认 constqp（零空帧，584FPS
    # 实测）。VBR_HQ 在无 LA 下有输出延迟/空帧风险，首轮实测 SPS=0 即其路径未产出 SPS。
    rc[0] = _v(1); rc[1] = 0                    # NV_ENC_PARAMS_RC_CONSTQP
    rc[2] = rc[3] = rc[4] = 25                  # qpInterP / qpInterB / qpIntra
    rc[9] = rc[9] | (1 << 3) | (1 << 8)         # enableAQ(bit3)+enableTemporalAQ(bit8)，对齐生产 L1386-1387

    tag = f"abs@{off}" if off is not None else "baseline"
    if off is not None:
        cast(byref(p, off), POINTER(c_uint32))[0] = WRITE_LEVEL

    ip = _Init(); memset(byref(ip), 0, sizeof(ip))
    ip.version = INIT_VER; ip.encodeGUID = codec; ip.presetGUID = preset
    ip.encodeWidth = W; ip.encodeHeight = H; ip.darWidth = W; ip.darHeight = H
    ip.frameRateNum = FPS * 1000; ip.frameRateDen = 1000
    ip.enablePTD = 1
    ip.encodeConfig = cast(byref(p, 8), c_void_p)
    if _NvCreate(gp(_FI["InitEnc"]))(enc, byref(ip)) != 0:
        return None, f"InitializeEncoder failed ({tag})"

    def _mkb(idx, ver):
        b = (c_uint8 * 776)(); memset(b, 0, 776)
        cast(b, POINTER(c_uint32))[0] = ver
        if idx == 12:  # CreateInputBuffer: width@4, height@8 (NV12 luma only), bufferFmt@16
            cast(byref(b, 4), POINTER(c_uint32))[0] = W
            cast(byref(b, 8), POINTER(c_uint32))[0] = H
            cast(byref(b, 16), POINTER(c_uint32))[0] = NV12
        r = CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8*776))(gp(idx))(enc, b)
        if r != 0:
            raise RuntimeError(f"Create buffer fn#{idx} failed, code={r}")
        return cast(byref(b, 24), POINTER(c_void_p))[0] if idx == 12 else cast(byref(b, 16), POINTER(c_void_p))[0]
    ib = _mkb(12, _v(2)); bb = _mkb(14, _v(1))
    _Li = CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8*1544))
    _Ui = CFUNCTYPE(c_uint32, c_void_p, c_void_p)
    _Lb = CFUNCTYPE(c_uint32, c_void_p, POINTER(c_uint8*1544))
    _Ub = CFUNCTYPE(c_uint32, c_void_p, c_void_p)
    _En = _NvEncode

    # [FIX-DIAG-NOTORCH] NV12 输入用 cuMemAlloc 分配设备显存(替代 torch.zeros)：
    # 诊断脚本不应依赖 torch（环境无关），NVENC 只认 CUDA device 指针。
    nv12_sz = W * (H + H // 2)
    nv12 = c_void_p(None)
    libcuda.cuMemAlloc_v2(byref(nv12), nv12_sz)
    if not nv12.value:
        return None, "cuMemAlloc NV12 buffer failed"
    libcuda.cuMemsetD8_v2.restype = c_uint32
    libcuda.cuMemsetD8_v2.argtypes = [c_void_p, c_uint8, c_size_t]
    libcuda.cuMemsetD8_v2(nv12, 0, nv12_sz)
    out = bytearray()

    lock_stats = []
    def lock_bs(max_try=20):
        # [FIX-DIAG-LOCK] 对齐生产 _lock_bitstream_with_retry：失败/空帧 sleep 1ms 重试
        st = -1
        for t in range(max_try):
            lr = (c_uint8 * 1544)(); memset(lr, 0, 1544)
            cast(lr, POINTER(c_uint32))[0] = LBS_VER
            cast(byref(lr, 8), POINTER(c_void_p))[0] = bb
            st = _Lb(gp(_FI["LockBS"]))(enc, lr)
            if st == 0:
                n = cast(byref(lr, 36), POINTER(c_uint32))[0]
                # [FIX-DIAG-PTR] 铁律: offset@56 存的是 pBitstreamBuffer 指针值(8B)，
                # 必须先解引用再读内容——直接 cast(byref(lr,56)) 会把 lock struct
                # 自身内存(指针值+保留字段)当码流读出 => SPS 解析永远失败(head=指针值)。
                pv = cast(byref(lr, 56), POINTER(c_void_p))[0]
                if n > 0 and pv:
                    q = cast(pv, POINTER(c_uint8))
                    out.extend(bytes(q[:n]))
                _Ub(gp(_FI["UnlBS"]))(enc, bb)
                lock_stats.append((st, n))
                return st, n
            if t < max_try - 1:
                time.sleep(0.001)
        lock_stats.append((st, -1))
        return st, -1

    for i in range(N):
        lb = (c_uint8 * 1544)(); memset(lb, 0, 1544)
        cast(lb, POINTER(c_uint32))[0] = LIB_VER
        cast(byref(lb, 8), POINTER(c_void_p))[0] = ib
        if _Li(gp(_FI["LockIB"]))(enc, lb) != 0: return None, "LockInputBuffer failed"
        mp = cast(byref(lb, 16), POINTER(c_void_p))[0]
        ap = cast(byref(lb, 24), POINTER(c_uint32))[0]
        cp = (c_uint8 * 128)(); memset(cp, 0, 128)
        cast(byref(cp, 16), POINTER(c_uint32))[0] = 2
        cast(byref(cp, 32), POINTER(c_void_p))[0] = nv12
        cast(byref(cp, 48), POINTER(c_size_t))[0] = W
        cast(byref(cp, 72), POINTER(c_uint32))[0] = 2
        cast(byref(cp, 88), POINTER(c_void_p))[0] = c_void_p(mp)
        cast(byref(cp, 104), POINTER(c_size_t))[0] = ap if ap else W
        cast(byref(cp, 112), POINTER(c_size_t))[0] = W
        cast(byref(cp, 120), POINTER(c_size_t))[0] = H + H//2
        _cp_r = libcuda.cuMemcpy2D_v2(cp)
        if _cp_r != 0:
            return None, f"cuMemcpy2D[fi={i}] failed, code={_cp_r}"
        _Ui(gp(_FI["UnlIB"]))(enc, ib)

        # EncodePicture (raw 3360B，对齐生产 encode_frames_stream line 2059-2088)
        pic = (c_uint8 * 3360)(); memset(pic, 0, 3360)
        cast(pic, POINTER(c_uint32))[0] = PIC_VER
        cast(byref(pic, 4), POINTER(c_uint32))[0] = W
        cast(byref(pic, 8), POINTER(c_uint32))[0] = H
        cast(byref(pic, 12), POINTER(c_uint32))[0] = W
        cast(byref(pic, 24), POINTER(c_uint64))[0] = i * 1000000 // FPS  # inputTimeStamp
        cast(byref(pic, 40), POINTER(c_void_p))[0] = ib   # inputBuffer
        cast(byref(pic, 48), POINTER(c_void_p))[0] = bb   # outputBitstream (必设)
        cast(byref(pic, 64), POINTER(c_uint32))[0] = NV12  # bufferFormat (生产必写)
        cast(byref(pic, 68), POINTER(c_uint32))[0] = PIC_FRAME  # pictureStruct
        if i == 0:
            cast(byref(pic, 16), POINTER(c_uint32))[0] = 0x2  # NV_ENC_PIC_FLAG_FORCEIDR
        # [FIX-DIAG-CE] 对齐生产 FIX-CONSTQP-FRAME-CE (L2069-2096): constqp+la=0 同步
        # EncodePicture 必须带 CE。无 CE 在 T4 上 LockBitstream 时 segfault / 无输出。
        _ep_ce = c_void_p(None)
        libcuda.cuEventCreate.restype = c_uint32
        libcuda.cuEventCreate.argtypes = [POINTER(c_void_p), c_uint32]
        if libcuda.cuEventCreate(byref(_ep_ce), 0) == 0:
            cast(byref(pic, 56), POINTER(c_void_p))[0] = _ep_ce  # completionEvent@56
        r = _En(gp(_FI["Encode"]))(enc, pic)
        if r != 0: return None, f"EncodePicture failed (fi={i}, code={r})"
        if _ep_ce.value is not None:
            libcuda.cuEventSynchronize.restype = c_uint32
            libcuda.cuEventSynchronize.argtypes = [c_void_p]
            libcuda.cuEventSynchronize(_ep_ce)
            libcuda.cuEventDestroy.restype = c_uint32
            libcuda.cuEventDestroy.argtypes = [c_void_p]
            libcuda.cuEventDestroy(_ep_ce)
        lock_bs()
    # EOS (对齐生产 line 2160-2167: flags=0x8, inputBuffer=NULL, outputBitstream 必设)
    eos = (c_uint8 * 3360)(); memset(eos, 0, 3360)
    cast(eos, POINTER(c_uint32))[0] = PIC_VER
    cast(byref(eos, 16), POINTER(c_uint32))[0] = 0x8  # NV_ENC_PIC_FLAG_EOS
    cast(byref(eos, 40), POINTER(c_void_p))[0] = c_void_p(None)
    cast(byref(eos, 48), POINTER(c_void_p))[0] = bb
    _En(gp(_FI["Encode"]))(enc, eos)
    # [FIX-DIAG-DRAIN] 排空循环: 连续 4 次无输出(NEED_MORE_INPUT/空)即止，上限 24 次
    _miss = 0
    for _ in range(24):
        _st, _n = lock_bs(max_try=1)
        if _st == 0 and _n > 0: _miss = 0
        else: _miss += 1
        if _miss >= 4: break

    for idx, h in ((15, bb), (13, ib)):  # DestroyBitstreamBuffer / DestroyInputBuffer
        try: CFUNCTYPE(c_uint32, c_void_p, c_void_p)(gp(_FI["DBs"] if idx == 15 else _FI["DIn"]))(enc, h)
        except Exception: pass
    if nv12.value:
        libcuda.cuMemFree_v2.restype = c_uint32
        libcuda.cuMemFree_v2.argtypes = [c_void_p]
        libcuda.cuMemFree_v2(nv12)
    CFUNCTYPE(c_uint32, c_void_p)(gp(_FI["DestEnc"]))(enc)

    # ── 诊断: 输出流统计（定位 SPS=0）──
    print(f"  [{tag}] out_len={len(out)} lock_calls={lock_stats}", flush=True)
    if out:
        print(f"  [{tag}] head={bytes(out[:24]).hex()}", flush=True)

    # 解析 SPS level_idc: 找起始码, NAL header, profile_idc, constraint, level_idc
    lv = []
    pos = 0
    while True:
        i = out.find(b"\x00\x00\x01", pos)
        if i < 0: break
        hdr = i + 3
        if hdr >= len(out): break
        if out[hdr] & 0x1F == 7 and hdr + 3 < len(out):
            lv.append(out[hdr + 3])
        pos = hdr
    return lv, None

# [FIX-DIAG-SDK13] profileLevel 真实绝对偏移（SDK 13 头文件逐字节验证，nvEncodeAPI.h）：
#   presetCfg@8 + NV_ENC_CONFIG.encodeCodecConfig@168(rcParams@40+128B) + NV_ENC_CONFIG_H264.level@4
#   注意: NV_ENC_CONFIG_H264 的 bitfield 区(enableTemporalSVC..reservedBitFields:10)=32bit@0,
#   故 level 在 @4（不是生产 _NvEncConfigH264 假设的 profileLevel@8=idrPeriod 位置!）。
#   生产 _NvEncConfig.encodeCodecConfig@1052 是错误布局(reserved3[53]覆盖 rcParams 区)，
#   经结构体写入 profileLevel=51 实际落在 reserved[278] 保留区 → 驱动忽略（此脚本当初据此全灭）。
_THEORY = 8 + 168 + 4

def main():
    print("=" * 70)
    print(f"profileLevel 偏移诊断 | 写入={WRITE_LEVEL} | {W}x{H}@{FPS}fps | 理论偏移={_THEORY}")
    print("=" * 70)
    b, err = run(None)
    if err: print(f"baseline 失败: {err}"); sys.exit(1)
    bl = b[0] if b else None
    print(f"[baseline] level_idc={bl} (自动)  SPS数={len(b)}")
    cand = [_THEORY + o for o in range(-16, 32, 4)] if "--sweep" in sys.argv else \
           [_THEORY - 8, _THEORY - 4, _THEORY, _THEORY + 4, _THEORY + 8, _THEORY + 12]
    hits = []
    for o in cand:
        lv, err = run(o)
        if err:
            print(f"[abs@{o:>4}]  {err}"); continue
        l = lv[0] if lv else None
        m = "✅ 命中 profileLevel" if l == WRITE_LEVEL else ("⚠️ 写入无效(同baseline)" if l == bl else "？level_idc变化但非51")
        print(f"[abs@{o:>4}]  level_idc={l}  SPS数={len(lv)}  {m}")
        if l == WRITE_LEVEL: hits.append(o)
    print("=" * 70)
    print(f"结论: profileLevel 有效偏移 = {hits}" if hits else
          f"结论: 未命中。用 --sweep 扫 {_THEORY-16}..{_THEORY+28}，或检查 encodeCodecConfig 实际偏移。")
    print("=" * 70)

if __name__ == "__main__":
    main()
