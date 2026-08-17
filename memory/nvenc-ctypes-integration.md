---
name: nvenc-ctypes-integration
description: NVENC SDK 13.0 ctypes 直接编码 — 已验证的 struct 布局、函数索引、版本常量、版本号公式、码率控制、关键 bug 模式及修复方案
metadata:
  node_type: memory
  type: reference
  originSessionId: 9ac38aa0-722c-43f8-aa1e-b6fad7621a9e
---

# NVENC SDK 13.0 ctypes 直接编码完整参考

> 信息来源：nv-codec-headers n13.0.19.0，已在 Tesla T4 / driver 580 上通过 `test_nvenc_pre_torch.py` 和多段视频编码验证。

## 版本号公式

```python
_NVENCAPI_VERSION = 0x0d  # SDK 13.0: MAJOR | (MINOR << 24) = 13 | (0 << 24)

def _sdk13_ver(ver, bit31=False):
    v = _NVENCAPI_VERSION | (ver << 16) | (0x7 << 28)
    if bit31:
        v |= (1 << 31)
    return v
```

源自 `NVENCAPI_STRUCT_VERSION(ver)` 宏：`NVENCAPI_VERSION | (ver << 16) | (0x7 << 28)`。低字节必须是 `NVENCAPI_VERSION`（0x0d），**不能用 `sizeof(struct)`**。

## 版本常量速查

| 常量 | 值 | `_sdk13_ver(x)` |
|------|-----|-----------------|
| NV_ENC_CREATE_INPUT_BUFFER_VER | 0x7002000d | ver=2 |
| NV_ENC_LOCK_INPUT_BUFFER_VER | 0x7001000d | ver=1 |
| NV_ENC_CREATE_BITSTREAM_BUFFER_VER | 0x7001000d | ver=1 |
| NV_ENC_LOCK_BITSTREAM_VER | 0xf002000d | ver=2, bit31=True |
| NV_ENC_PIC_PARAMS_VER | 0xf007000d | ver=7, bit31=True |
| NV_ENC_INITIALIZE_PARAMS_VER | 0xf007000d | ver=7, bit31=True |
| NV_ENC_PRESET_CONFIG_VER | 0xf005000d | ver=5, bit31=True |
| NV_ENC_CONFIG_VER | 0xf009000d | ver=9, bit31=True |
| NV_ENC_RC_PARAMS_VER | ver=1 |

## 关键枚举值

| 常量 | 值 | 说明 |
|------|-----|------|
| NV_ENC_SUCCESS | 0 | 成功 |
| NV_ENC_DEVICE_TYPE_CUDA | 1 | **不是 4！** |
| NV_ENC_PIC_FLAG_EOS | 0x8 | EOS 帧标志，**不是 1！** |
| NV_ENC_PIC_FLAG_FORCEINTRA | 0x1 | 强制 Intra 帧 |
| NV_ENC_PIC_FLAG_FORCEIDR | 0x2 | 强制 IDR 帧 |
| NV_ENC_BUFFER_FORMAT_NV12 | 1 | NV12 颜色格式 |
| NV_ENC_PIC_STRUCT_FRAME | 2 | 逐行帧 |
| NV_ENC_PARAMS_RC_CONSTQP | 0 | 恒定量化参数 |
| NV_ENC_PARAMS_RC_VBR | 1 | 可变码率（**不推荐，值未生效**） |

## 函数索引（SDK 13.0 func_table）

func_table 原始大小 2552 bytes，函数指针从 offset 8 开始排列：

| 函数 | Index |
|------|-------|
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

## 已验证的 struct 布局（byte array + 手动 offset write）

所有 offset 已经过 C-dump 验证并在 Tesla T4 上测试通过。

### NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS (1552 bytes)

| 字段 | Offset | 类型 |
|------|--------|------|
| version | 0 | uint32 |
| deviceType | 4 | uint32 |
| device | 8 | void* |
| reserved | 16 | void* |
| apiVersion | 24 | uint32 |
| reserved1 | 28 | uint8[253×4] |
| reserved2 | 1040 | void*[64] ← **必须包含，否则 sizeof 不对** |

### NV_ENC_CREATE_INPUT_BUFFER (776 bytes)

| 字段 | Offset | 类型 |
|------|--------|------|
| version | 0 | uint32 |
| width | 4 | uint32 |
| height | 8 | uint32 ← **NV12 时必须是 luma height（如 576），不是 NV12 total height（如 864）！** |
| memoryHeap | 12 | uint32 (废弃，设 0) |
| bufferFmt | 16 | uint32 |
| reserved | 20 | uint32 |
| inputBuffer | 24 | void* (OUTPUT) |
| pSysMemBuffer | 32 | void* |

**Why height=luma:** NVENC SDK 文档要求 NV12 的 height 为 luma height（H），驱动用 height × pitch 计算 chroma 平面偏移。如果设为 NV12 total height（H + H/2），chroma 偏移错误 → 编码器从未初始化内存读 chroma → 零值/128 → 灰色输出。

### NV_ENC_LOCK_INPUT_BUFFER (1544 bytes)

| 字段 | Offset | 类型 |
|------|--------|------|
| version | 0 | uint32 |
| doNotWait:1 + reservedBitFields:31 | 4 | uint32 |
| inputBuffer | 8 | void* (INPUT) |
| bufferDataPtr | 16 | void* (OUTPUT) |
| pitch | 24 | uint32 (OUTPUT) |

### NV_ENC_CREATE_BITSTREAM_BUFFER (776 bytes)

| 字段 | Offset | 类型 |
|------|--------|------|
| version | 0 | uint32 |
| bitstreamBuffer | 16 | void* (OUTPUT) |

### NV_ENC_LOCK_BITSTREAM (1544 bytes)

| 字段 | Offset | 类型 |
|------|--------|------|
| version | 0 | uint32 |
| bitfield (doNotWait 等) | 4 | uint32 |
| outputBitstream | 8 | void* (INPUT) |
| bitstreamSizeInBytes | 36 | uint32 (OUTPUT) |
| bitstreamBufferPtr | 56 | void* (OUTPUT) |

### NV_ENC_PIC_PARAMS (3360 bytes)

| 字段 | Offset | 类型 |
|------|--------|------|
| version | 0 | uint32 |
| inputWidth | 4 | uint32 |
| inputHeight | 8 | uint32 |
| inputPitch | 12 | uint32 |
| encodePicFlags | 16 | uint32 — EOS=0x8 |
| frameIdx | 20 | uint32 |
| inputTimeStamp | 24 | uint64 |
| inputDuration | 32 | uint64 |
| inputBuffer | 40 | void* |
| outputBitstream | 48 | void* — **EOS 帧也必须设！** |
| completionEvent | 56 | void* |
| bufferFmt | 64 | uint32 |
| pictureStruct | 68 | uint32 |
| pictureType | 72 | uint32 |
| codecPicParams | 76 | union (H264=1048B) |

## 码率控制（rcParams）

### SDK 13.0 NV_ENC_CONFIG.rcParams offset = 40

`NV_ENC_RC_PARAMS` 是**纯顺序 struct**（nvEncodeAPI.h master 验证，2026-06-09）。**NO union** — 之前的 union 假设是错误的，`verify_rcparams_offset.py` 误将 `NV_ENC_QP`（12B struct）当成 4B 枚举导致整体偏移计算错误。

```
NV_ENC_RC_PARAMS 完整布局 (128 bytes total):
  0: version (uint32)                                  rc_ptr[0]
  4: rateControlMode (NV_ENC_PARAMS_RC_MODE, uint32)   rc_ptr[1]
  8: constQP.qpInterP (uint32, NV_ENC_QP struct 12B)   rc_ptr[2]
 12: constQP.qpInterB (uint32)                          rc_ptr[3]
 16: constQP.qpIntra (uint32)                           rc_ptr[4]
 20: averageBitRate (uint32)                            rc_ptr[5]
 24: maxBitRate (uint32)                                rc_ptr[6]
 28: vbvBufferSize (uint32)                             rc_ptr[7]
 32: vbvInitialDelay (uint32)                           rc_ptr[8]
 36: bitfield (uint32)                                  rc_ptr[9]
       bits: enableMinQP:0 enableMaxQP:1 enableInitialRCQP:2
             enableAQ:3 reservedBitField1:4 enableLookahead:5
             disableIadapt:6 disableBadapt:7 enableTemporalAQ:8
             zeroReorderDelay:9 enableNonRefP:10
             strictGOPTarget:11 aqStrength:12-15
             enableExtLookahead:16 reserved:17-31
 40: minQP {InterP@40, InterB@44, Intra@48}            rc_ptr[10-12]
 52: maxQP {InterP@52, InterB@56, Intra@60}            rc_ptr[13-15]
 64: initialRCQP {InterP@64, InterB@68, Intra@72}      rc_ptr[16-18]
 76: temporallayerIdxMask (uint32)                      rc_ptr[19]
 80: temporalLayerQP[8] (uint8[8])
 88: targetQuality (uint8)                              ← NOT at uint32 boundary!
 89: targetQualityLSB (uint8)
 90: lookaheadDepth (uint16)
 92: lowDelayKeyFrameScale (uint8)
 93: yDcQPIndexOffset (int8)
 94: uDcQPIndexOffset (int8)
 95: vDcQPIndexOffset (int8)
 96: qpMapMode (uint32)                                 rc_ptr[24]
100: multiPass (uint32)                                  rc_ptr[25]  ← 独立字段！
104: alphaLayerBitrateRatio (uint32)                     rc_ptr[26]
108: cbQPIndexOffset (int8)
109: crQPIndexOffset (int8)
110: reserved2 (uint16)
112: lookaheadLevel (uint32)                             rc_ptr[28]
116: viewBitrateRatios[7] (uint8[7])
123: reserved3 (uint8)
124: reserved1 (uint32)                                  rc_ptr[31]
```

### CONSTQP 写入（已验证✅）

```python
rc_ptr[0] = _sdk13_ver(1)    # NV_ENC_RC_PARAMS_VER
rc_ptr[1] = 0                 # NV_ENC_PARAMS_RC_CONSTQP
rc_ptr[2] = _qp_val            # constQP.qpInterP @8
rc_ptr[3] = _qp_val            # constQP.qpInterB @12
rc_ptr[4] = _qp_val            # constQP.qpIntra @16
```

### VBR_HQ + targetQuality（2026-06-09 GPU 验证 ✅）

v6.4.3.1+ 使用 VBR_HQ (mode=32) + targetQuality 替代 CONSTQP 做 CQ 编码。
**GPU 验证通过**（Tesla T4, driver 580, CRF=18 vs 28）：
- CRF=18 → targetQuality=33 → 21.1MB / 6447kbps ✅
- CRF=28 → targetQuality=23 → 6.1MB / 1866kbps ✅
- 差距 3.5x，targetQuality 完全生效

正确 offset（nvEncodeAPI.h master 源码验证）：
- averageBitRate@20(rc_ptr[5]), maxBitRate@24(rc_ptr[6])
- targetQuality@88(uint8): `max(1, 51 - CRF)`
- bitfield@36(rc_ptr[9]): AQ=bit3, TemporalAQ=bit8
- lookaheadDepth@90(uint16), multiPass@100(rc_ptr[25])

**三轮 CRF 不生效根因链**：
1. **原始 bug**（120.4MB 无区分）：`targetQuality` 写到 byte 116（viewBitrateRatios），驱动读到 0 → 仅用 avgBitrate 做 VBR
2. **第一次"修复"**（193kbps/0.6MB 无区分）：`averageBitRate` 从 rc_ptr[5] 移到 rc_ptr[3]（覆盖 qpInterB），驱动从 offset 20 读到 preset default ~200kbps
3. **第二次修复**（GPU 验证通过）：全部 offset 从 nvEncodeAPI.h master 原文验证。avgbr@20, maxbr@24, tq@88, bitfield@36, lookahead@90, multiPass@100

## 已踩过的坑（按错误码）

| 错误码 | 症状 | 根因 | 修复 |
|--------|------|------|------|
| 15 (INVALID_VERSION) | OpenEncodeSessionEx 失败 | 缺少 `reserved2[64]`，sizeof 不匹配 | 补全 struct 字段 |
| 2 (OUT_OF_MEMORY) | OpenEncodeSessionEx 失败 | `deviceType = 4`（应为 1=CUDA） | 改为 1 |
| 8 (INVALID_PARAM) | InitializeEncoder 失败 (VBR_HQ/QVBR) | (A) UNKNOWN UNION 布局: targetQuality@76→temporallayerIdxMask, lookaheadDepth@78→同字段, enableLookahead+lookaheadDepth=0矛盾 (B) multiPass uint8 写入破坏 uint32 字段 (C) VBR_HQ 模式 multiPass=1(TWO_PASS) 不合法 — two-pass 仅用于 CBR/VBR 目标码率模式，非质量驱动模式 | (A) 76→88, 78→90 (SEQUENTIAL 布局) (B) rc_ptr[25]=0 而非 uint8@+100 (C) multiPass=0(disable) |
| 8 (INVALID_PARAM) | InitializeEncoder 失败 (v6.4.3.1) | rcParams offsets 全部正确，但 multiPass=1(NV_ENC_TWO_PASS_QUARTER_RESOLUTION) VBR_HQ 不支持 two-pass | rc_ptr[25]=0 禁用 two-pass |
| 死锁 (self-deadlock) | close() 永久阻塞 | `close()` 持 `_lock` 后调 `self.flush()` 再次获取 `_lock` | close() 中移除 flush() |
| — | outputBitstream=NULL 的 EOS 帧无输出 | EOS pic_params 未设置 outputBitstream | 设为 `self._bs_handle` |
| — | 灰色输出 (R=G=B) | (1) D2H 异步竞态无 event.synchronize() (2) CreateInputBuffer.height=NV12 total height (3) presetConfig SDK offset 错误 | synchronize + luma height + offset 8 |
| 15 (INVALID_VERSION) | GetEncodePresetConfig 失败 | presetConfig offset 4（ctypes packed）而非 SDK offset 8 | 恢复 offset 8 |
| — | CRF 不生效（3 轮迭代） | targetQuality 写入错误 offset → 驱动读不到 | 见上文 VBR_HQ + targetQuality 根因链 |

## NV_ENC_RC_PARAMS 布局验证方法

从 nvEncodeAPI.h 获取准确布局的标准流程：
1. **优先用 curl 下载**：`curl -L -o /tmp/nvEncodeAPI.h "https://raw.githubusercontent.com/FFmpeg/nv-codec-headers/master/include/ffnvcodec/nvEncodeAPI.h"`（已验证可靠，`sdk/13.0` 分支不存在，master 分支是最新的）
2. 手动解析 C struct 定义：注意嵌套 struct（如 NV_ENC_QP 是 12B 而非 4B）和 bitfield 位序
3. 跨校验：`rc_ptr[2]/[3]/[4]` 对应 CONSTQP 的 qpInterP/qpInterB/qpIntra，已有 GPU 验证的正确代码可作锚点
4. **不要信任自动解析脚本**：`verify_rcparams_offset.py` 将 NV_ENC_QP（12B struct）误当 4B enum，导致所有后续字段偏移错误 8 字节

## 其他关键经验

1. **GUID 必须从 driver 动态查询**（GetEncodeGUIDs / GetEncodePresetGUIDs），不能硬编码头文件 GUID
2. **RegisterResource 在 T4/driver 580 上 segfault**，用 CreateInputBuffer 替代
3. **ctypes.memmove 不能读 GPU 显存**，GPU→GPU 拷贝用 `cuMemcpyDtoD_v2`
4. **PyTorch context 对 driver API 不可见**，用 `cuDevicePrimaryCtxRetain` + `cuCtxPushCurrent` 管理
5. **cuCtxCreate 可能报 code=2**（OOM），fallback 到 primary context 即可
6. **EncodePicture 必须有 outputBitstream** 指向 CreateBitstreamBuffer 返回的 handle
7. **NVENCEncoder._lock 是 `threading.Lock()`（不可重入）**，禁止持有锁时调用自身方法
8. **所有 ctypes struct 都用 byte array + 手动 offset write**，不使用 `ctypes.Structure` 子类（field offset 不可靠）
9. **ctypes Structure 可能基于旧 SDK 布局，缺少新字段** — v6.4.3 的 `_NvEncConfig` 完全缺失 `rcParams`，导致 NVENC 以默认码率编码（文件膨胀 3.3x）。必须对照 SDK 头文件验证 struct 字段完整性
10. **`CreateInputBuffer.height` 对 NV12 必须是 luma height**（H），不是 NV12 total height（H + H/2） — 驱动用 height 计算 chroma 偏移
11. **`NV_ENC_PRESET_CONFIG` SDK offset = 8**（version@0 + 4-byte padding + presetCfg@8）。ctypes `_pack_=1` 让 presetCfg 在 ctypes 字段 offset 4，但 SDK 函数按 SDK 布局解析。**所有 presetConfig 访问必须用 SDK offset 8。** 曾尝试改为 offset 4 导致 code=15。

## rcParams 写入（SDK 13.0 byte array 方式）

`_NvEncConfig` ctypes Structure 缺少 rcParams，需用 raw pointer 在 SDK offset 40 写入：

```python
# preset_config 是 GetEncodePresetConfig 返回的 bytearray
# NV_ENC_CONFIG.rcParams 在 SDK 13.0 的 offset 40
# NV_ENC_RC_PARAMS: version@0, rateControlMode@4, constQP{qpInterP}@8, qpInterB@12, qpIntra@16
rc_ptr = cast(byref(preset_config, 8 + 40), ctypes.POINTER(c_uint32))
rc_ptr[0] = _sdk13_ver(1)  # NV_ENC_RC_PARAMS_VER
rc_ptr[1] = 0              # NV_ENC_PARAMS_RC_CONSTQP
rc_ptr[2] = qp              # constQP.qpInterP
rc_ptr[3] = qp              # constQP.qpInterB
rc_ptr[4] = qp              # constQP.qpIntra
```

offset 是 `8 + 40` 而非 `40`，因为 `_NvEncPresetConfig.presetConfig` 在 SDK 中从 offset 8 开始（version + 4byte padding），而 ctypes `_pack_=1` 让它在 ctypes 层面从 offset 4 开始。SDK 函数按 SDK 布局解析，所以以 SDK offset 为准。

## 参考代码

- 实现：`external/IFRNet/process_video_v6_4_3_single.py` NVENCEncoder 类 (line ~915)
- SDK 头文件：`/tmp/nv-codec-headers/include/ffnvcodec/nvEncodeAPI.h`
- Bug 修复记录：[[v642-v643-bug-fixes]]
- SPS/PPS 修复：[[nvenc-sps-pps-debugging]]
