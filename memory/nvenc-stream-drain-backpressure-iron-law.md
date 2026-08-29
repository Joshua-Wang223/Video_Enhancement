---
name: nvenc-stream-drain-backpressure-iron-law
description: NVENC stream 模式 drain 消费铁律：任何 drained 数据必须完整消费（写入 pairs + 清 pending），丢弃即相位错位
metadata: 
  node_type: memory
  type: project
  modified: 2026-08-07T04:35:34.858Z
  originSessionId: 422ae849-ffc4-4927-b7f4-8ecbda158a9e
---

## 设计模式

NVENC stream 编码模式（`encode_frames_stream`）的 drain/backpressure 消费铁律——从 `[[ifrnet-v6451-seg1-fix-stack]]` 根因修复提炼的通用规则。

## 背景

`encode_frames_stream` 与 `ce_pipeline` 是 NVENC SDK 编码的两条并行路径。stream 模式通过 `_strm_slot_pending[slot_idx]=(fi, idr, cumul_count, ...)` 追踪每个 slot 的 in-flight 编码帧，通过 `_drain_outputs_blocking()` 从 NVENC 驱动取出已完成编码的码流。

## 铁律

任何从 `_drain_outputs_blocking()` 取出的数据帧，必须按主 drain 循环的相同方式完整消费，不得因"只需腾 slot"而丢弃数据。具体：

1. 将 H.264 码流写入 pairs 列表（应用 SPS/PPS 注入）
2. 将对应的 `_strm_slot_pending[slot]` 显式设为 None
3. 不消费直接丢弃 → pending 残留 → 后续该 slot 被重用时 `_strm_slot_pending[slot_idx] is not None` 为真 → 再次 drain → 但已无数据产生 → deadlock guard 触发 break → 实际上并未真正消费数据 → slot 相位永久错位

## tryfix1 失败模式

tryfix1 的 drain 循环（line 1970-1980）只调用了 `_drain_outputs_blocking` 但无任何消费代码：

```python
# tryfix1: 只排不消费 ❌
while self._strm_slot_pending[slot_idx] is not None:
    _drained = self._drain_outputs_blocking(max_slots=1)
    # ... guard ...
    # ← 缺少数据消费：无 pairs.append, 无 _strm_slot_pending 清除
```

正确的 tryfix2/main 版本：

```python
# tryfix2/main: 排 + 消费 ✅
while self._strm_slot_pending[slot_idx] is not None:
    _drained = self._drain_outputs_blocking(max_slots=1)
    # ... guard ...
    for _est_fi, _h264_data in _drained:
        _drain_slot = _est_fi % self._slot_count
        _actual_fi, _, _is_idr, _ep_s = self._strm_slot_pending[_drain_slot]
        pairs.append((_actual_fi, self._apply_sps_pps(_h264_data, _is_idr)))
        self._strm_slot_pending[_drain_slot] = None
```

## ESRGAN 侧正确参考实现

ESRGAN `nvenc_sdk.py` 的 `_ensure_slot_free` + `_apply_drained_entries` 设计正确消费 drained 数据（写入 `_codemgr_buffer` + 清除 `_slot_pending`），是此修复的重要参考。

## 涉及文件

- `process_video_v6_4_5_1_single.py` line 1967-1997（`encode_frames_stream` 方法内）
- `external/realesrgan_video/nvenc_sdk.py` line 1278（`_apply_drained_entries`）、line 1335（`_ensure_slot_free`）

## 相关记忆

- [[ifrnet-v6451-seg1-fix-stack]] — 此铁律的来源根因修复
- [[esrgan-consecutive-idr-false-positive]] — ESRGAN 侧为何同样 per-slot IDR 无害
- [[esrgan-nvenc-slot-backpressure]] — ESRGAN slot 背压相关
- [[pipe4-la8-root-cause-fix]] — per-slot IDR 历史根因
