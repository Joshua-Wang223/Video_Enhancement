# IFRNet test11 首段异常：最根本原因与最直接修复（定论）

> 2026-08-14 定论。完整证据链见 [[ifrnet-la-aux-no-clear-test11-fix]]；
> 历史演化见 [[stream-ts-reassociation-fix]]、[[ifrnet-la-aux-slot-clearing-fix]]、
> [[ifrnet-la-drain-order-fix]]、[[la-window-rotation-fix]]（已废弃）。

## 最根本原因（一句话）

**drain 侧把 NVENC LA 冷会话预热期吐出的"无 VCL 辅助块"（独立 SPS/PPS/AUD）
当成普通帧消费（pop 队首、进 pairs），破坏了 per-slot FIFO 标签与物理码流数据
的双射：一个辅助块 pop 掉真实帧的 pending 记录后，真实帧的数据被贴到后一帧的
标签上，相位偏移 1 个槽位并随 9 槽轮转永久传播（每 9 帧迟一窗）。**

test11 量化表现：首段 4960 包仅 1411 帧可解码，549 次 frame_num 回退，106 条
pts drop（est_missing=3452）；ES 排列精确为 `[0..8]` 正确、`[10..17,9]`、
`[19..26,18]`…（slot=LA+1=9）。

辅助块清 pending 的二级效应：`_ensure_slot_free` 误判槽位空闲 → 输入 buffer
未排空即复用 → LA 延迟消费读到被覆盖的像素 → 数据与标签在内容层面也错位。
seg1/seg2 正常是因为会话已热、驱动不再吐辅助块。

## 最直接修复（一句话）

**消费每个 drain 块之前先做 VCL 分类（`_nal_first_vcl_type`）：无 VCL 辅助块
仅缓存 SPS/PPS，不占 fi、不进 pairs/results、不清 pending（FIX-AUX-NO-CLEAR）；
VCL 块才按 FIFO 队首消费。**

这一处改动同时恢复了两个不变式：
1. 标签↔数据双射（辅助块不再 pop 真实帧条目）；
2. 槽位复用背压（pending 保留 → `_ensure_slot_free` 排空后才复用输入 buffer）。

配套加固（非根因必需，但消除同类路径）：
- `_ensure_slot_free` 目标槽直探（FIX-SLOT-DRAIN-TARGET）+ guard 超限空帧占位，
  绝不带 pending 复用；
- `_drain_outputs_blocking` 数据指针为空时不推进计数器（FIX-DRAIN-COUNTER-DRIFT）；
- 相位漂移诊断 `_diag_phase_shift`（VCL 块 outputTimeStamp@40 vs FIFO 队首 gfi）。

## 验证

- v6.4.5.1 / v6.4.4.1 / v6.4.3.1 三文件 + ESRGAN nvenc_sdk.py 同步修复；
- 生产 Linux GPU test11 同参数（VBR_HQ + LA=8, 2x）复验通过；
- 三 IFRNet 文件归一化后函数体一致，全部 py_compile 通过。
