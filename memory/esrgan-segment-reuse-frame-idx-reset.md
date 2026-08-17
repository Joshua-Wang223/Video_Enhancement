---
name: esrgan-segment-reuse-frame-idx-reset
description: 跨段复用 encoder 时 _frame_idx/_output_slot_idx 被错误清零导致段2+ 85%帧丢失
metadata: 
  node_type: memory
  type: project
  status: fixed
  originSessionId: 279c3f6f-f525-402f-b6ed-7f6152663198
---

# ESRGAN 跨段复用 _frame_idx 错误重置

## 症状

段 1 (全新 session) 正常，段 2-6 每个 chunk `h264_valid≈17~18 h264_empty≈150` (valid 占比 ~1/9 = 1/slot_count)。片段 2-4、6 丢失 85%-88% 帧，片段 5 偶然正常。输出视频乱序、丢帧、花屏。

## 根因

`nvenc_sdk.py` `_NVENCEncodeThread.__init__` 中的 `[FIX-SEGMENT-REUSE]` 补丁在跨段复用同一个 NVENC 硬件 session 时：
- 错误地 `self._nvenc._frame_idx = 0`
- 错误地 `self._nvenc._output_slot_idx = 0`

**关键**: `_frame_idx` 在第 1303 行被直接写入提交给驱动的 `NV_ENC_PIC_PARAMS.inputTimeStamp`，驱动的 LA 重排序状态机依赖此时间戳**连续递增**。段 1 结束后时间戳在 7223，段 2 却重置为 0 → 驱动内部状态机被"时间戳回退"打乱 → `LockBitstream` 长期卡在 `NEED_MORE_INPUT` → 绝大多数帧拿不到码流。

软件层：拿不到码流的帧 (`h264_data == b""`) 在 `_write_la_output()` 非-final 分支被 `continue` 静默丢弃，不计 `written` 也不计 `empty`，日志无告警。

**矛盾**: 此补丁的注释和文件顶部模块 docstring ("_frame_idx 不重置，保持 LA FIFO 连续性") 直接矛盾，是后来引入的回归。

## 修复

**文件**: `nvenc_sdk.py`

删掉 `_frame_idx = 0` 和 `_output_slot_idx = 0` 两行，保持跨段连续递增（匹配模块设计文档原意）。**保留** `_slots_warmed = set()` 重置 — 这个不依赖 frame_idx 是否清零，空集确保新段首帧仍强制 IDR。

额外添加：
- `_pending` 审计计数器追踪未回收帧
- `flush_and_join()` 结束时校验 `_pending == 0`，非零时打印告警

## 验证

修复后段 2-5 `written == submitted` 完全平帧，无丢帧。

## 关联

这是 ESRGAN chunked-LA + 跨段复用独有的 bug。IFRNet (`process_video_v6_4_5_1_single.py`) 的 encoder 不跨段复用（每段重建），不受影响。

## 后续反转（2026-07-28，FIX-REOPEN-SLOT-PHASE）

本文结论仅适用于 **session 跨段持续存在** 的旧架构。[FIX-CROSS-SEGMENT-SESSION]
引入 `reopen()`（段边界销毁并重建驱动会话）后前提消失：新会话时间戳从 0 重启
与 gen=1 同构。此时**不重置 `_frame_idx` 反而致病**——提交 slot
（`_frame_idx % N`）与 drain slot（`_output_slot_idx % N`）相位错位，段 2+
`_ensure_slot_free` 告警刷屏并丢帧。修复：`reopen()` 内 `_frame_idx` 与
`_output_slot_idx` **同步**归零。详见 [[esrgan-reopen-slot-phase-misalignment]]。
