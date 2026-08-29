---
name: ifrnet-v6451-seg1-fix-stack
description: IFRNet v6.4.5.1 段1花屏根因修复栈：FIX-STREAM-BACKPRESSURE（真根因）+ FIX-PIPE4-LA8（加剧因子），Linux 生产验证通过
metadata: 
  node_type: memory
  type: project
  modified: 2026-08-07T04:35:11.898Z
  originSessionId: 422ae849-ffc4-4927-b7f4-8ecbda158a9e
---

## 问题

IFRNet stream 编码模式下，第一段（segment 000）输出花屏/卡顿，段 2/3 正常。

## 根因分析

### 被证伪的假设：per-slot IDR warmup [FIX-PIPE4-LA8]

`encode_frames_stream` 原逻辑：
```python
force_idr = force_idr_first and (slot_idx not in self._strm_slots_warmed)
```
LA 预热期 EncodePicture 返回 NEED_MORE_INPUT 使 slot 永不 warm → 段首 LA+1 帧全强制 IDR（22连IDR）。

**修复**：改为 `force_idr = force_idr_first and (fi == 0)`（单 IDR，与 ce_pipeline 对齐），清理 `_strm_slots_warmed` 死状态。

**tryfix1（仅有此修复）实机验证失败**：段000 frames(828)≠packets(4960)、FN回退=258 依旧 → 证明 per-slot IDR 不是段1根因。

### 真根因（已验证）：[FIX-STREAM-BACKPRESSURE]

`encode_frames_stream` 的 slot 复用背压循环在 drain 出数据后只推进 `_output_slot_idx` 却不消费数据（不写入 pairs、不清 `_strm_slot_pending`）→ slot 相位错开 + pending 残留 → LA 输出 buffer 重路由错位（每9帧迟一窗）→ frame_num 周期性回退 → 解码器静默丢帧（无解码错误日志，只表现为花屏/卡顿）。

修复借鉴 ESRGAN 侧 `_ensure_slot_free`/`_apply_drained_entries` 设计：drain 出的每帧按主 drain 循环同样消费——写入 pairs + 清除 `_strm_slot_pending[slot] = None`。

## 修复位置

`encode_frames_stream()` 方法，slot 复用锁 + LockInputBuffer 之间（约 line 1967-1997），`while self._strm_slot_pending[slot_idx] is not None:` 循环内 drain 后必须完整消费数据。

## 修复文件版本关系

| 文件 | 修复内容 | 验证结果 |
|------|---------|---------|
| `_tryfix1.py` | 仅 [FIX-PIPE4-LA8] 单 IDR | ❌ 段000 FN回退=258 依旧 |
| `_tryfix2.py` = 当前主文件 | [FIX-PIPE4-LA8] + [FIX-STREAM-BACKPRESSURE] | ✅ Linux 生产验证通过 |

## 关键教训

- **per-slot IDR 是加剧因子，不是根因**。单纯消除连 IDR 不解决问题。
- **drain 数据消费是强制义务**。任何从 `_drain_outputs_blocking()` 取出的帧必须写入 pairs + 清 pending，丢弃即相位错位。
- **码流解剖是诊断金标准**：frames≠packets + FN回退>0 比视觉观察更精确。

## 相关记忆

- [[nvenc-stream-drain-backpressure-iron-law]] — drain 消费铁律通用规则
- [[pipe4-la8-root-cause-fix]] — per-slot IDR 历史根因修复
- [[esrgan-consecutive-idr-false-positive]] — 同帧 IDR slice 误报分析
- [[verify-bitstream-idr-slice-fix]] — 码流验证工具修复
