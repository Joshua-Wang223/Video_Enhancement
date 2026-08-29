---
name: nvenc-ctypes-verified-layouts
description: Verified struct layouts for NVENC SDK 13.0 — NV_ENC_RC_PARAMS (128B sequential, NO union), all struct offsets C-dump/GPU verified
metadata: 
  node_type: memory
  type: reference
  originSessionId: e90439f4-b1cf-475c-a858-19d405501d5f
---

## Verified NVENC SDK 13.0 struct layouts (nv-codec-headers n13.0.19.0)

All offsets C-dump verified and confirmed working via `test_nvenc_pre_torch.py`.

### NV_ENC_CREATE_INPUT_BUFFER (776 bytes)
| Field | Offset | Type |
|-------|--------|------|
| version | 0 | uint32 |
| width | 4 | uint32 |
| height | 8 | uint32 |
| memoryHeap | 12 | uint32 (deprecated, set 0) |
| bufferFmt | 16 | uint32 (NV_ENC_BUFFER_FORMAT_NV12=1) |
| reserved | 20 | uint32 |
| inputBuffer | 24 | void* (OUTPUT) |
| pSysMemBuffer | 32 | void* |

### NV_ENC_LOCK_INPUT_BUFFER (1544 bytes)
| Field | Offset | Type |
|-------|--------|------|
| version | 0 | uint32 |
| doNotWait:1 + reservedBitFields:31 | 4 | uint32 bitfield |
| inputBuffer | 8 | void* (INPUT) |
| bufferDataPtr | 16 | void* (OUTPUT) |
| pitch | 24 | uint32 (OUTPUT) |

### NV_ENC_CREATE_BITSTREAM_BUFFER (776 bytes)
| Field | Offset | Type |
|-------|--------|------|
| version | 0 | uint32 |
| bitstreamBuffer | 16 | void* (OUTPUT) |

### NV_ENC_LOCK_BITSTREAM (1544 bytes)
| Field | Offset | Type |
|-------|--------|------|
| version | 0 | uint32 |
| bitfield (doNotWait etc.) | 4 | uint32 |
| outputBitstream | 8 | void* (INPUT) |
| bitstreamSizeInBytes | 36 | uint32 (OUTPUT) |
| bitstreamBufferPtr | 56 | void* (OUTPUT) |

### NV_ENC_PIC_PARAMS (3360 bytes)
| Field | Offset | Type |
|-------|--------|------|
| version | 0 | uint32 |
| inputWidth | 4 | uint32 |
| inputHeight | 8 | uint32 |
| inputPitch | 12 | uint32 |
| encodePicFlags | 16 | uint32 |
| frameIdx | 20 | uint32 |
| inputTimeStamp | 24 | uint64 |
| inputDuration | 32 | uint64 |
| inputBuffer | 40 | void* |
| outputBitstream | 48 | void* |
| completionEvent | 56 | void* |
| bufferFmt | 64 | uint32 |
| pictureStruct | 68 | uint32 |
| pictureType | 72 | uint32 |
| codecPicParams | 76 | union (H264=1048B) |

### NV_ENC_RC_PARAMS (128 bytes) — 2026-06-09 verified via nvEncodeAPI.h master + GPU test

Source: `nvEncodeAPI.h` from nv-codec-headers master (github.com/FFmpeg/nv-codec-headers).
**CRITICAL: NO union at offset 8.** NV_ENC_QP is a 12-byte SEQUENTIAL struct {qpInterP, qpInterB, qpIntra}, not a 4-byte enum.

```
Offset  rc_ptr   Type      Field                    Notes
────────────────────────────────────────────────────────────────────
  0       [0]    uint32    version                  _sdk13_ver(1)
  4       [1]    uint32    rateControlMode          0=CONSTQP, 32=VBR_HQ, 64=QVBR
  8       [2]    uint32    constQP.qpInterP         NV_ENC_QP struct (12B)
 12       [3]    uint32    constQP.qpInterB
 16       [4]    uint32    constQP.qpIntra
 20       [5]    uint32    averageBitRate
 24       [6]    uint32    maxBitRate
 28       [7]    uint32    vbvBufferSize
 32       [8]    uint32    vbvInitialDelay
 36       [9]    uint32    bitfield                 see bit layout below ⬇
 40      [10]    uint32    minQP.qpInterP           NV_ENC_QP struct (12B)
 44      [11]    uint32    minQP.qpInterB
 48      [12]    uint32    minQP.qpIntra
 52      [13]    uint32    maxQP.qpInterP           NV_ENC_QP struct (12B)
 56      [14]    uint32    maxQP.qpInterB
 60      [15]    uint32    maxQP.qpIntra
 64      [16]    uint32    initialRCQP.qpInterP     NV_ENC_QP struct (12B)
 68      [17]    uint32    initialRCQP.qpInterB
 72      [18]    uint32    initialRCQP.qpIntra
 76      [19]    uint32    temporallayerIdxMask
 80       —       uint8    temporalLayerQP[8]
 88       —       uint8    targetQuality             ← KEY! NOT at uint32 boundary
 89       —       uint8    targetQualityLSB
 90       —       uint16   lookaheadDepth
 92       —       uint8    lowDelayKeyFrameScale
 93       —       int8     yDcQPIndexOffset
 94       —       int8     uDcQPIndexOffset
 95       —       int8     vDcQPIndexOffset
 96      [24]    uint32    qpMapMode
100      [25]    uint32    multiPass                 ← SEPARATE field, NOT in bitfield!
104      [26]    uint32    alphaLayerBitrateRatio
108       —       int8     cbQPIndexOffset
109       —       int8     crQPIndexOffset
110       —       uint16   reserved2
112      [28]    uint32    lookaheadLevel
116       —       uint8    viewBitrateRatios[7]
123       —       uint8    reserved3
124      [31]    uint32    reserved1
```

**Bitfield layout at rc_ptr[9] (offset 36):**
| Bit | Field |
|-----|-------|
| 0 | enableMinQP |
| 1 | enableMaxQP |
| 2 | enableInitialRCQP |
| 3 | enableAQ |
| 4 | reservedBitField1 |
| 5 | enableLookahead |
| 6 | disableIadapt |
| 7 | disableBadapt |
| 8 | enableTemporalAQ |
| 9 | zeroReorderDelay |
| 10 | enableNonRefP |
| 11 | strictGOPTarget |
| 12-15 | aqStrength |
| 16 | enableExtLookahead |

**Key codec modes:**
| Mode | rc_ptr[1] | Description |
|------|-----------|-------------|
| CONSTQP | 0 | Direct QP: rc_ptr[2-4] = qpInterP/B/Intra |
| VBR_HQ | 32 (0x20) | CQ via targetQuality@88: `max(1, 51-CRF)` |
| QVBR | 64 (0x40) | Quality VBR via targetQuality@88 |

### NV_ENC_CONFIG 布局（SDK 13.0）— encodeCodecConfig@168（绝对 176）[FIX-DIAG-SDK13]

2026-08-07 sweep 三重旁证（`tests/diagnose_profilelevel_offset.py --sweep`）。

`NV_ENC_PRESET_CONFIG`：version@0 + padding@4 + `presetCfg@8`（即 NV_ENC_CONFIG）。

| Field | 相对 presetCfg | 绝对 | 类型 |
|-------|--------------|------|------|
| version | 0 | 8 | uint32 |
| profileGUID | 4 | 12 | GUID (16B) |
| gopLength | 20 | 28 | uint32 |
| frameIntervalP | 24 | 32 | uint32 |
| monoChrome | 28 | 36 | uint32 |
| frameFieldMode | 32 | 40 | uint32 |
| mvPrecision | 36 | 44 | uint32 |
| rcParams | 40 | 48 | NV_ENC_RC_PARAMS (128B) |
| **encodeCodecConfig** | **168** | **176** | union（H264=NV_ENC_CONFIG_H264） |

**生产 `_NvEncConfig.encodeCodecConfig@1052` 是错误布局**：前 28 字节（version/profileGUID/gopLength/frameIntervalP）恰好正确，从 @28 起错位（monoChrome→frameFieldMode、mvPrecision→frameFieldMode_1、rcParams 区被 reserved3[53] 吞掉），reserved5[172] 把 encodeCodecConfig 推到 1052。codec 字段经结构体写入全部落保留区被驱动忽略——修复统一走硬编码绝对偏移（对齐 rcParams 模式）。

### NV_ENC_CONFIG_H264（SDK 13.0）— 相对 encodeCodecConfig / 绝对

| Field | 相对 | 绝对 | 说明 |
|-------|------|------|------|
| bitfield | 0 | 176 | repeatSPSPPS=bit12 |
| **level** | **4** | **180** | **profileLevel_idc（sweep 命中 51 ✅）** |
| idrPeriod | 8 | 184 | IDR 周期 = fps |
| spsId | 24 | 200 | sweep 证 SPS 内容变化 |
| maxNumRefFrames | 60 | 236 | 4（FIX-GPU-STAY 首次真正生效） |
| VUI | 72 | 240 | 112B |
| chromaFormatIDC | 192 | 368 | 1=4:2:0 |

## Function indices (SDK 13.0)
| Function | Index |
|----------|-------|
| GetEncodeGUIDCount | 1 |
| GetEncodeGUIDs | 4 |
| GetEncodePresetGUIDs | 9 |
| GetEncodePresetConfig | 10 |
| InitializeEncoder | 11 |
| CreateInputBuffer | 12 |
| DestroyInputBuffer | 13 |
| CreateBitstreamBuffer | 14 |
| DestroyBitstreamBuffer | 15 |
| EncodePicture | 16 |
| LockBitstream | 17 |
| UnlockBitstream | 18 |
| LockInputBuffer | 19 |
| UnlockInputBuffer | 20 |
| MapInputResource | 25 |
| UnmapInputResource | 26 |
| DestroyEncoder | 27 |
| OpenEncodeSessionEx | 29 |
| RegisterResource | 30 |
| UnregisterResource | 31 |
| GetEncodePresetConfigEx | 39 |

## Version constants (SDK 13.0)
Formula: `NVENCAPI_STRUCT_VERSION(ver, bit31=False)` = `0x0d | (ver << 16) | (0x7 << 28) | (0x80000000 if bit31 else 0)`

| Constant | Value |
|----------|-------|
| NV_ENC_CREATE_INPUT_BUFFER_VER | 0x7002000d |
| NV_ENC_LOCK_INPUT_BUFFER_VER | 0x7001000d |
| NV_ENC_CREATE_BITSTREAM_BUFFER_VER | 0x7001000d |
| NV_ENC_LOCK_BITSTREAM_VER | 0xf002000d |
| NV_ENC_PIC_PARAMS_VER | 0xf007000d |
| NV_ENC_INITIALIZE_PARAMS_VER | 0xf007000d |
| NV_ENC_PRESET_CONFIG_VER | 0xf005000d |
| NV_ENC_CONFIG_VER | 0xf009000d |

## Critical lessons

1. **GUIDs must be dynamically queried** from driver via GetEncodeGUIDs/GetEncodePresetGUIDs — hardcoded header GUIDs don't match driver
2. **ALL ctypes structs had wrong field offsets** — use byte arrays with manual offset writes for reliability
3. **RegisterResource segfaults** on T4/driver 580 — use CreateInputBuffer instead
4. **ctypes.memmove cannot read GPU memory** — use cuMemcpyDtoD_v2 for GPU→GPU copy
5. **PyTorch context invisible to driver API** — use cuDevicePrimaryCtxRetain + cuCtxPushCurrent for context management
6. **cuCtxCreate may fail with code=2** (OUT_OF_MEMORY) — fallback to primary context works for NVENC
7. **EncodePicture requires outputBitstream** pointing to a CreateBitstreamBuffer handle
8. **NV_ENC_QP is a 12-byte SEQUENTIAL struct** {qpInterP, qpInterB, qpIntra}, NOT a 4-byte union/enum. Auto-parsers that mis-identify it as 4B will cascade all subsequent RC_PARAMS offsets by 8 bytes.
9. **targetQuality is at rcParams byte 88** (uint8, NOT uint32-aligned) — between temporalLayerQP[8] and targetQualityLSB. Writing to byte 116 (viewBitrateRatios) is a silent no-op.
10. **multiPass is a SEPARATE field at offset 100**, NOT embedded in the bitfield at offset 36. Lookahead=bit5 (not bit4), TemporalAQ=bit8 (not bit7)
11. **targetQuality@88 sweep-verified 2026-06-09**: `diagnose_targetquality_offset.py --sweep` 在 Tesla T4 上对所有候选偏移(76-124)逐一测试 tq=1 vs tq=51。只有 offset 88 产生真正的 CQ 效果(99.8% 大小差异)。其他"有效"偏移(104/120/124)均为假阳性 — 破坏其他字段的副作用。92-100/108/112 触发 NV_ENC_ERR_INVALID_PARAM(code=8)。
12. **CUDA context 建立顺序至关重要**: Driver API 先建立 primary context(cuDevicePrimaryCtxRetain + cuCtxPushCurrent 仅一次)，然后 PyTorch Runtime 重用同一 context。禁止在 encode_frame/close 中 push/pop — 二次 push 导致 CUDA_ERROR_LAUNCH_FAILURE(201) → context 栈损坏 → segfault。
