---
name: verify-bitstream-idr-slice-fix
description: verify_segment_bitstream_v2.py 同帧 IDR slice 误报修复：NVENC 多 slice/帧聚类统计，消除连IDR=3 假阳性
metadata: 
  node_type: memory
  type: project
  modified: 2026-08-07T04:35:19.317Z
  originSessionId: 422ae849-ffc4-4927-b7f4-8ecbda158a9e
---

## 问题

NVENC 在较大分辨率下每帧编码为多个 slice（1280×720 为 4 slice/帧），同一 IDR 帧的 4 个 IDR slice (NAL type=5, frame_num相同) 在 NAL 流中连续出现。旧逻辑按 NAL/slice 级统计 `idr_within_32_after_first`，正常码流的同帧 IDR slice 被误报为「连IDR=3」。

## 修复方案

仅改 `check_nal_stats()` 函数体，未动 `parse_h264_es`/`frame_num_regress`/调用方：

1. **IDR slice 按帧聚类**：判据 = 与**上一个 IDR slice** NAL 索引相邻（`i == prev_slice_idx + 1`）+ frame_num 相同 → 同一帧，多 slice 只计 1 个 IDR 帧。
   - 关键细节：prev 必须跟踪「上一个 IDR slice」而非「上一帧起始 slice」，否则同帧第3+个 slice 会因与帧起始不相邻被误判为新帧。
2. `idr_within_32_after_first` 从「IDR slice 数」改为「不同 IDR 帧数」，32 NAL 窗口不变。
3. 真异常「连IDR=22」仍能检出：真异常 IDR 帧间有 AUD/SEI/P slice 等分隔 NAL，索引必然不相邻，聚类不会吞掉跨帧 IDR。

## 验证结果

| 用例 | 连IDR | FN回退 | 结论 |
|------|-------|--------|------|
| ESRGAN 正常输出 `esrgan_nal_000.txt` | 3→**0** | 0 | 误报消除 |
| IFRNet try_fix1 `ifrnet_tryfix1_nal_000.txt` | 3→**0** | 294 | 单 IDR 确认 + 真异常仍检出 |
| 构造 22 连 IDR（1 slice/帧） | 16 | 0 | 不失检 |
| 构造 22 连 IDR（4 slice/帧） | 6 | 0 | 不失检 |
| 构造 FN 回退模拟 | 0 | 1 | 检出 |
| 单 slice 正常码流 | 0 | 0 | 全 0 |

## 相关记忆

- [[esrgan-consecutive-idr-false-positive]] — ESRGAN 连IDR=3 误报根因分析
- [[ifrnet-v6451-seg1-fix-stack]] — 段1花屏修复中使用此工具诊断
- [[pipe4-la8-root-cause-fix]] — per-slot IDR 历史根因
