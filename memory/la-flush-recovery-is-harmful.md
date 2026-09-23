---
name: la-flush-recovery-is-harmful
description: "[已修正] LA flush 恢复的帧在错误的排空逻辑下产生花屏，而非 NVENC 设计行为；正确排空可完整恢复"
metadata: 
  node_type: memory
  type: project
  originSessionId: 28ead61a-c614-4d3a-9dd9-702328f1818a
---

# LA Flush 恢复是花屏残帧 — 旧结论已修正

## 旧结论（基于错误的排空逻辑，现已修正）

旧版生产代码使用的编码循环存在 3 个 SDK 违规：
1. EncodePicture 后只 LockBitstream 一次（而非循环直到 NEED_MORE_INPUT）
2. NEED_MORE_INPUT 时强制 Lock/Unlock 丢弃了有效帧
3. pipeline_depth = LA（而非 LA+1），缓冲区覆盖

在此错误实现下观察到的现象：
- LA=8 时输出 679 帧（输入 687 帧）
- EOS flush 排出的 8 帧是花屏/灰屏残片
- 尝试将这些 flush 帧写入 muxer 导致视频尾部花屏

**此现象被误判为 NVENC LA 的"设计行为"** — 实际是错误排空逻辑导致的缓冲区覆盖与帧丢失。

## 修正后的结论（基于 SDK 合规排空逻辑）

在正确的编码循环下（见 [[nvenc-la-frame-conservation-fix]]）：
- LA flush 排出的帧是完整有效的编码帧
- 输入严格等于输出帧数（帧数守恒）
- 帧内容顺序与原始输入一致（开头不丢、末尾不重复）

**花屏并非 NVENC 固有行为，而是错误客户端代码的症状。**

## 生产代码影响

生产代码 v6.4.x 系列中的 `_NVENCEncodeThread` flush 处理（写入 b'FLUSH' 标记并在 Writer 端丢弃）：
- 当编码循环修复为 SDK 合规后，flush 帧变为有效帧 → 应正常写入 muxer
- 旧版丢弃 flush 帧的逻辑在修复后排空循环后不再适用
- 修复后的代码无需特殊 flush 处理 — 帧直接通过正常 LockBitstream 路径取回

## 需要还原的错误修复（旧结论下所做的修改）

以下基于旧结论的修改在正确排空逻辑下是错误的：
- v6.4.3.1/4.1/5.1 Writer 端 `b'FLUSH'` 标记丢弃逻辑 → 修复后应移除
- `_flush_frame_count` 仅作诊断 → 修复后可作为正常帧计数
- "LA 丢帧是设计行为" 的所有注释 → 需更新

## 参考
- [[nvenc-la-frame-conservation-fix]] — 正确排空逻辑的完整 3 项修复
- [[pipe4-la8-root-cause-fix]] — per-slot IDR 花屏的独立根因（与本问题不同）

**Why:** 对 LA flush 花屏的根因误判导致生产代码持续丢弃有效 flush 帧。实际根因是排空逻辑不符合 SDK 规范，而非 NVENC 硬件行为。
**How to apply:** 1) 实现 SDK 合规的排空逻辑；2) 移除 flush 帧丢弃代码；3) pipeline_depth 设为 LA+1；4) 参见 [[nvenc-la-frame-conservation-fix]] 完整方案。
