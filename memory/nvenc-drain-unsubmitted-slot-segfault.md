---
name: nvenc-drain-unsubmitted-slot-segfault
description: CONSTQP+LA=0 下 encode_frames_batch LockBitstream segfault 的完整修复
metadata:
  type: feedback
  originSessionId: f064f7c2-01bd-4a11-a842-bbb890a39e9f
---

# Fix: encode_frames_batch CONSTQP+LA=0 LockBitstream segfault

## 问题

`encode_frames_batch()` 在 CONSTQP+LA=0 模式下，EncodePicture 后调用
`_drain_outputs_blocking()` 时 LockBitstream segfault。
但 `encode_frame()` 用相同的 LockBitstream 函数不 crash。

## 根因

`encode_frame` 使用 completionEvent + cuEventSynchronize，GPU pipeline 在
LockBitstream 前已排空。`encode_frames_batch` 无 completionEvent，NVENC driver
内部状态在同步 EncodePicture 返回后未完全排空 → LockBitstream 访问未就绪的
bitstream buffer → segfault。

只发生在 Tesla T4 特定 driver 组合上，IL=0 无 NEED_MORE_INPUT 保护。

## 修复

在 CONSTQP+LA=0 模式下，EncodePicture 前设置 completionEvent，后做
cuEventSynchronize（与 encode_frame 完全一致的模式）：

```python
_ep_ce = c_void_p(None)
if (self._rate_mode, self._la_depth) == ('constqp', 0):
    self._libcuda.cuEventCreate(ctypes.byref(_ep_ce), 0)
    if _ep_ce.value is not None:
        cast(byref(pic_buf, 56), ctypes.POINTER(c_void_p))[0] = _ep_ce

# EncodePicture with CE
encode_picture(self._encoder, cast(pic_buf, ...))

# Sync CE before drain
if _ep_ce.value is not None:
    self._libcuda.cuEventSynchronize(_ep_ce)
    self._libcuda.cuEventDestroy(_ep_ce)
    cast(byref(pic_buf, 56), ...) = c_void_p(None)
```

## 之前的错误尝试

| 尝试 | 结果 | 原因 |
|------|------|------|
| FIX-DRAIN-UNSUBMITTED-SLOT: max_slots 限制 | ❌ 单独不够 | 即使 drain 已提交的 slot 也 crash |
| 移除 _drain_outputs_blocking 中 CUDA ctx push/pop | ❌ 不够 | 冗余 push 不是根因 |
| cuCtxSynchronize 在 drain 前 | ❌ 不够 | 同步了整个 context 但 CE 更精确 |
| 在 EncodePicture 后设置 CE 重新提交 | ❌ 逻辑错误 | CE 必须在 EncodePicture 前设置 |
| completionEvent per-frame | ✅ 最终有效 | 匹配 encode_frame 已验证模式 |

## 验证

`python Accessory/probe/nvenc_sdk_realesrgan_suite.py -v -m gpu` → **28/28 PASSED, exit 0** ✅

关键测试:
- `test_no_empty_frames_constqp_la0` → PASSED（原 segfault）
- `test_frame_conservation_constqp_la0` → PASSED
- `test_frame_conservation_vbr_hq_la8` → PASSED
- `test_frame_conservation_qvbr_la8` → PASSED

## 影响文件

| 文件 | 修复类型 | 状态 |
|------|---------|------|
| `external/realesrgan_video/nvenc_sdk.py` | FIX-CONSTQP-FRAME-CE | ✅ |
| `external/IFRNet/process_video_v6_4_3_single.py` | FIX-CONSTQP-FRAME-CE（inline LockBitstream） | ✅ |
| `external/IFRNet/process_video_v6_4_3_1_single.py` | FIX-CONSTQP-FRAME-CE + FIX-DRAIN-UNSUBMITTED-SLOT | ✅ |
| `external/IFRNet/process_video_v6_4_4_single.py` | FIX-CONSTQP-FRAME-CE（inline LockBitstream） | ✅ |
| `external/IFRNet/process_video_v6_4_4_1_single.py` | FIX-CONSTQP-FRAME-CE + FIX-DRAIN-UNSUBMITTED-SLOT | ✅ |
| `external/IFRNet/process_video_v6_4_5_single.py` | FIX-CONSTQP-FRAME-CE（inline LockBitstream） | ✅ |
| `external/IFRNet/process_video_v6_4_5_1_single.py` | FIX-CONSTQP-FRAME-CE + FIX-DRAIN-UNSUBMITTED-SLOT | ✅ 上次会话 |

### 各版本分类

| 类型 | 版本 | 修复内容 |
|------|------|---------|
| **Inline LockBitstream**（无 `_drain_outputs_blocking`，EncodePicture 后直接 LockBitstream 同一 slot） | v6.4.3, v6.4.4, v6.4.5 | 仅 FIX-CONSTQP-FRAME-CE（completionEvent 同步） |
| **Global Drain**（有 `_drain_outputs_blocking`，per-frame 循环 drain 所有 slot） | v6.4.3.1, v6.4.4.1, v6.4.5.1 | FIX-CONSTQP-FRAME-CE + FIX-DRAIN-UNSUBMITTED-SLOT（max_slots 限制 + CE 同步） |

参考: [[nvenc-ce-pipeline-architecture]]、[[pipeline-depth-slot-rotation-confusion]]
