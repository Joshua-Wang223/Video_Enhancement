---
name: nvenc-rc-params-sequential-layout
description: NV_ENC_RC_PARAMS SEQUENTIAL vs UNION 布局澄清，所有版本可执行代码一致，注释标签不同
metadata: 
  node_type: memory
  type: project
  originSessionId: e47eeb99-6d8c-4065-8b2f-1ca21ec2534e
---

# NV_ENC_RC_PARAMS SEQUENTIAL 布局

## 背景

`NV_ENC_RC_PARAMS` 在不同 SDK 版本中有不同的内存布局。旧版 SDK 文档暗示 `constQP` 和 VBR 参数共享内存（union@8），新版 SDK (12.0+) 是 SEQUENTIAL 布局。

## 正确的 SEQUENTIAL 布局 (SDK 13.0, nvEncodeAPI.h)

```
offset 0:  version (uint32)
offset 4:  rateControlMode (uint32)
offset 8:  constQP.qpInterP (uint32)     ← NOT a union!
offset 12: constQP.qpInterB (uint32)
offset 16: constQP.qpIntra (uint32)
offset 20: averageBitRate (uint32)        ← 始终在 +20
offset 24: maxBitRate (uint32)
offset 28: vbvBufferSize (uint32)
offset 32: vbvInitialDelay (uint32)
offset 36: bitfield (uint32)              ← AQ, Lookahead, TemporalAQ bits
offset 40: minQP (NV_ENC_QP, 12B)
offset 52: maxQP (NV_ENC_QP, 12B)
offset 64: initialRCQP (NV_ENC_QP, 12B)
offset 76: temporalLayerIdxMask
offset 80: temporalLayerQP[8]
offset 88: targetQuality (uint8)          ← 关键!
offset 89: targetQualityLSB (uint8)
offset 90: lookaheadDepth (uint16)        ← 关键!
offset 92-99: lowDelayKeyFrameScale, DcQPIndexOffset[3]
offset 100: multiPass (uint32)             ← rc_ptr[25]
...
offset 124: reserved1
```

## 所有版本的实现一致性

v6.4.3.1/4.1/5/5.1 的可执行代码**完全相同**：
```python
rc_ptr = cast(byref(preset_config, 8 + 40), ctypes.POINTER(c_uint32))
# targetQuality 始终在 rcParams+88:
_tq8_ptr = cast(byref(preset_config, 8 + 40 + 88), ctypes.POINTER(c_uint8))
# lookaheadDepth 始终在 rcParams+90:
_la_ptr = cast(byref(preset_config, 8 + 40 + 90), ctypes.POINTER(c_uint16))
```

## 版本间差异：仅注释标签

| 版本 | 标签 | 含义 |
|------|------|------|
| v6.4.3.1 | `[V6431-RC-FIXED]` | 首次 GPU 验证 (116→76→88 三轮) |
| v6.4.4.1 | `[V6441-RC-FIXED]` | 继承验证结果 |
| v6.4.5 | `[V6451-RC-FIXED]` | 回移自 5.1 |
| v6.4.5.1 | `[V6451-RC-FIXED]` | 独立验证 + 最完整注释 (GPU verified 2026-06-09) |

## `targetQuality` offset 验证历史

`targetQuality` 的正确 offset 经历了三轮修正: 116 → 76 → **88**。最终通过 GPU 端 H.264 header 解析确认。

## `_NvEncConfig.reserved5` 吸收 enableTemporalAQ

```python
("reserved5", c_uint32 * 172),  # 172*4 = 688 bytes
```
`enableTemporalAQ` 是 rcParams bitfield bit8，不需要在 `NV_ENC_CONFIG` 中保留独立字段，被 reserved5 正确吸收。

**Why:** 验证了所有版本的 NVENC RC params 字节布局一致性
**How to apply:** 无需消除版本间注释差异；使用 `targetQuality@88(uint8)` 和 `lookaheadDepth@90(uint16)`
