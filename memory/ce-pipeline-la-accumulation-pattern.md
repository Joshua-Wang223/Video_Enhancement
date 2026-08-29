---
name: ce-pipeline-la-accumulation-pattern
description: LA>0 ce-pipeline 帧守恒实现模式 — 累积编码、send_eos、slot 分配、f0 暂存
metadata: 
  node_type: memory
  type: project
  originSessionId: 71a52bbb-72b5-45ca-94eb-834c56f9e215
---

# ce-pipeline LA>0 帧守恒 — 生产级实现模式

## 核心矛盾

ce-pipeline 的 per-batch 编码与 LA（Lookahead）的跨帧依赖是根本矛盾的：

- **LA 需要连续帧流**：前 LA 帧处于缓冲期，编码器需要连续填充前向预看窗口
- **per-batch 编码打断连续流**：每个 batch 独立调用 `encode_frames_batch`，local `fi` 从 0 开始分配 slot，覆盖上一批次 LA 缓冲帧

## 解决模式 1：LA>0 全段累积 + 一次性编码

### v6.4.5.1/4.1（`_NVENCEncodeThread._loop()` 路径）

```python
_la_mode = (self._nvenc._la_depth > 0)
_acc_nv12 = []
_acc_force_idr = False
while True:
    item = self._q.get()
    if item is self._SENTINEL:
        break
    if _la_mode:
        _acc_nv12.extend(nv12_list)      # 累积所有帧
    else:
        self._nvenc.encode_frames_batch_ce_pipeline(...)  # LA=0 保持 per-batch

# 收到 SENTINEL 后一次性编码
if _la_mode and _acc_nv12:
    h264_list = self._nvenc.encode_frames_batch(
        _acc_nv12, _acc_force_idr, send_eos=True)
```

### v6.4.3.1（writer_loop 内联编码路径）

相同模式：writer_loop 累积 NV12 tensor，SENTINEL break 后 `encode_frames_batch(send_eos=True)`。

## 解决模式 2：`send_eos` 参数 — EOS + 全槽排空内置于编码方法

```python
def encode_frames_batch(self, nv12_tensors, force_idr_first=False, send_eos=False):
    # ... normal encoding loop ...
    if send_eos:
        # 发送 EOS
        EncodePicture(NULL input + flag=0x8)
        # 按 _output_slot_idx 起始顺序逐 slot blocking LockBitstream 排空
        for slot in drain_order:
            while True:
                data, status = LockBitstream(blocking)
                if status == NEED_MORE_INPUT: break
                results[actual_fi] = data  # 放入正确帧索引位置
                self._output_slot_idx += 1
    else:
        # 原有尝试性 drain（无 EOS 保证）
```

`send_eos=False` 保留向后兼容（旧 per-batch 路径），`send_eos=True` 用于全段累积路径。

## 解决模式 3：slot 分配用 global `_frame_idx`

```python
# ❌ 旧：批次内 fi 从 0 开始，f0 用 slot 0 后 batch fi=0 也用 slot 0 → 冲突
slot_idx = fi % self._pipeline_depth

# ✅ 新：全局 _frame_idx 保证跨 f0 + batch 的 slot 编号连续
slot_idx = self._frame_idx % self._pipeline_depth
```

`encode_frames_batch` 和 `encode_frames_batch_ce_pipeline` 两处都需要改。

## 解决模式 4：f0 暂存 + 插入累积 batch

参见 [[f0-first-frame-loss-ce-pipeline]]。

## 模式选择流程

```
LA=0  ──→ per-batch ce_pipeline（CE 异步流水线，性能最优）
LA>0  ──→ 全段累积 + encode_frames_batch(send_eos=True)
          + f0 暂存 encoder._pending_f0_nv12 + insert(0, ...)
```

## EOS flush 去重规则

- `send_eos=True` 已内置 EOS + 全槽排空
- encoder.flush() **仅 LA=0 路径调用**（避免双重 EOS）
- LA>0 路径的 `_loop()` 末尾 `flush_data = self._nvenc.flush()` 仅对 `not _la_mode` 生效

**Why:** 生产脚本的 ce-pipeline 架构需要跨批次 LA 连续性。4 个模式共同解决：累积消除跨批次 slot 覆盖、send_eos 保证帧数守恒、global slot 防止冲突、f0 暂存恢复首帧。

**How to apply:** 新版本或变体需要 LA>0 支持时：1) encode thread 使用累积模式；2) 编码方法添加 send_eos 参数；3) slot 分配使用 global _frame_idx；4) f0 按 LA 深度分流处理。

**Related:** [[nvenc-la-frame-conservation-fix]], [[f0-first-frame-loss-ce-pipeline]], [[nvenc-ce-pipeline-architecture]]
