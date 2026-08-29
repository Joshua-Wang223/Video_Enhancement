---
name: esrgan-nvenc-slot-backpressure
description: "物理 slot 复用前无背压检查导致 bitstream buffer 被\"带病\"覆盖"
metadata: 
  node_type: memory
  type: project
  status: fixed
  originSessionId: 279c3f6f-f525-402f-b6ed-7f6152663198
---

# ESRGAN 物理 Slot 复用无背压

## 症状

段 0 少 8 帧，紧邻的段 1 多出 8 帧 (`written > submitted`)。整个视频总共少 6 帧（8 帧流窜到下段后，P/B 帧解码失败丢弃 6 帧，仅 2 帧可解码）。

## 根因

**文件**: `nvenc_sdk.py` `encode_frames_batch()`

`EncodePicture` 提交新帧时复用物理 `self._slots[slot_idx]['bs_buf']` 缓冲区。原始代码只在**提交后**做 best-effort 增量 drain，指望排空节奏追上提交节奏。

但 `la_depth=8` 和 `slot_count=9` 之间只有 1 帧余量。段边界（EOS 收尾阶段新提交帧变少，触发 drain 的机会也变少）让 backlog 超过 `slot_count` → 某个物理 slot 在上一帧的码流还没被 `LockBitstream` 读出时就被新帧的 `EncodePicture` 覆盖 → **物理层面丢失**，事后无论如何加强 drain 循环都补不回来。

日志证据链：段 0 少 8 帧 → 段 1 `written=7209 submitted=7201` 多出 8 帧 → 这 8 帧走了"跨 chunk 延迟输出"的 `_prev_chunk_outputs` 兜底逻辑，被记到了下一段的 muxer。

## 修复

**文件**: `nvenc_sdk.py`

将背压从"提交后尽量追上"改成**"提交前强制等到干净"**：

```python
def _ensure_slot_free(self, slot_idx, chunk_start_global, n_frames, results, prev_chunk_outputs):
    while self._slot_pending.get(slot_idx):
        _drained = self._drain_outputs_blocking(max_slots=1)
        if not _drained:
            # 保守上限兜底，避免死锁
            break
        self._apply_drained_entries(_drained, chunk_start_global, n_frames, results, prev_chunk_outputs)
```

在每帧提交前调用，结构性保证复用某个物理 slot 时其上一次码流必然已被排空。同时把原来内联的 drain 分发逻辑抽成 `_apply_drained_entries()`，避免两份逻辑跑偏。

## 关联

此 bug 与 [[esrgan-segment-reuse-frame-idx-reset]] 的修复配合才能完全消除段间帧泄漏。仅修复 slot 背压不足以解决 `_frame_idx` 重置引发的 85% 丢帧，但二者独立修复后共同保证了帧守恒。

## 2026-08-14 追加：辅助块账记加固（FIX-AUX-NO-CLEAR 移植）

IFRNet test11 证明 NVENC LA 冷会话预热期会把 SPS/PPS/AUD 作为独立无 VCL 辅助块
drain；旧 FIFO 记账未分类，辅助块会 pop 真实帧的 pending 记录 → 背压失效 →
输入覆盖 → 每 9 帧迟一窗（549 次 frame_num 回退）。ESRGAN 与 IFRNet 同驱动同
架构，存在同样的潜在风险（此前生产未触发，可能与 ≥1080p 自动 CONSTQP+LA=0
及时序窗口有关）。

本次将 IFRNet v6.4.5.1 的修复移植到 `nvenc_sdk.py`：
- `_nal_first_vcl_type()` VCL/辅助块分类；辅助块仅缓存 SPS/PPS，不占 fi、
  不进 results、不清 pending（`_apply_drained_entries` / EOS drain /
  末尾 drain / ce_pipeline inline drain / encode_frame 全部生效）。
- `_ensure_slot_free` 轮转排空 + 目标槽直探兜底（FIX-SLOT-DRAIN-TARGET）；
  guard 超限以空帧占位（`_prev_stream_h264` 兜底）消费 pending，绝不带
  pending 复用。
- `_drain_outputs_blocking` 返回三元组（含 outputTimeStamp@40），数据指针
  为空时不推进 `_output_slot_idx`（FIX-DRAIN-COUNTER-DRIFT）；VCL 块 ts 与
  FIFO 队首 gfi 比对产生 `_diag_phase_shift` 诊断。
- 新增 `_diag_aux_block` / `_diag_phase_shift` / `_diag_slot_drain_fallback`，
  `_stream_begin` 每段重置。

已知边界：ce_pipeline Phase 1 harvest（非生产路径，LA>0 已路由到
encode_frames_batch）未加辅助块分类，如需彻底兜底可后续补。
