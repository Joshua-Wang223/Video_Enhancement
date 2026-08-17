---
name: f0-first-frame-loss-ce-pipeline
description: ce-pipeline 生产脚本首帧 f0 丢失的根因、LA>0/LA=0 两种修复路径，与 v6.4.4 基准对比验证
metadata: 
  node_type: memory
  type: project
  originSessionId: 71a52bbb-72b5-45ca-94eb-834c56f9e215
---

# ce-pipeline 首帧 f0 丢失 — 根因与修复

## 现象

v6.4.5.1/4.1/3.1 输出 1372 帧，比 v6.4.4 基准（1373 帧）少 1 帧，缺失的是第 1 帧 f0。
LA=0 (constqp) 和 LA=8 (VBR_HB/QVBR) 均丢失。benchmark 中 `文件Δ = -1`。

## 根因

v6.4.5.1 引入的 `[FIX-PIPE4-LA8]` 将 f0 编码逻辑从：

```python
# v6.4.4 正确版本 (line 6104-6115)
if _nvenc_encoder is not None:
    first_nv12 = _rgb_to_nv12_gpu(first_gpu, input_is_bgr=True)
    h264_data = _nvenc_encoder.encode_frame(first_nv12, force_idr=force_idr)
    writer.write(h264_data)
    output_count += 1
```

改为：

```python
# v6.4.5.1 错误版本 (line 6999-7001)
if _nvenc_encoder is not None:
    pass  # 首帧已由 pipeline 内部 CE pipeline 处理 ← 错误假设
```

注释声称"CE pipeline 内部会处理 f0"，但 **GPU_RAW 路径的 encode_order 仅包含插值帧 + img1（右帧）**，f0 从未出现在任何 encode_order 中。Reader loop 中 f0 仅作为 `first_raw` 锚点用于插值计算，不进入输出。

## 修复：按 LA depth 分流

### 为什么不能简单地恢复 `encode_frame`

LA>0 时，NVENC 前 `LA` 帧处于前向预看缓冲期，`EncodePicture` 返回 `NEED_MORE_INPUT`，`encode_frame` 返回 `0 bytes`（诊断 log 已验证）。恢复原代码会写入空帧。

### LA>0 路径：暂存 encoder → 插入累积 batch

```python
# _process_segment() 中：
if _nvenc_encoder._la_depth > 0:
    # encode_frame 返回 0 bytes（LA 缓冲），不能直接写入
    _nvenc_encoder._pending_f0_nv12 = first_nv12
    _nvenc_encoder._pending_f0_force_idr = force_idr
else:
    # LA=0: 无缓冲延迟，直接 encode_frame
    h264_data = _nvenc_encoder.encode_frame(first_nv12, force_idr=force_idr)
    writer.write(h264_data)
    output_count += 1
```

```python
# _NVENCEncodeThread._loop() 累积编码前：
_pending_f0 = getattr(self._nvenc, '_pending_f0_nv12', None)
if _pending_f0 is not None:
    _acc_nv12.insert(0, _pending_f0)  # 插入 batch 开头
    if _f0_idr: _acc_force_idr = True
```

### LA=0 路径：直接 encode_frame（保持原 v6.4.4 行为）

LA=0 时 NVENC 每帧立即产出，`encode_frame` 可正常返回 H.264 数据，直接写入。

## 配套修复：slot 分配从 local fi → global _frame_idx

```python
# 原来：批次内部 fi 从 0 开始，f0 已用 slot 0 后 batch fi=0 也分配到 slot 0 → 冲突
slot_idx = fi % self._pipeline_depth    # ❌

# 修复：使用全局 _frame_idx（f0 encode_frame 已递增过一次）
slot_idx = self._frame_idx % self._pipeline_depth  # ✅
```

同时需要 `encode_frames_batch` 和 `encode_frames_batch_ce_pipeline` 两处都改。

## 配套修复：output_count 计数逻辑

```python
# 原来：替换式赋值，覆盖 f0 的 output_count
output_count = _actual_written   # ❌

# 修复：累加式
output_count += _actual_written  # ✅
```

## 验证

```
687 原始帧 × 2 插帧 → 期望 1373 帧 (= 687×2-1)

版本         ffprobe  文件Δ
v6.4.4       1373     ✓      (基准)
v6.4.3.1     1373     ✓
v6.4.4.1     1373     ✓
v6.4.5.1     1373     ✓
```

**Why:** `[FIX-PIPE4-LA8]` 误以为 CE pipeline 会输出 f0，但 GPU_RAW encode_order 从不包含首帧。LA>0 时 encode_frame 返回空数据进一步掩盖问题。修复需按 LA 深度分流：LA>0 插入累积 batch，LA=0 直接编码。

**How to apply:** 1) 恢复 f0 编码逻辑，按 `_la_depth > 0` 分流；2) LA>0 暂存 `_pending_f0_nv12` 到 encoder，encode thread 插入累积 batch 开头；3) LA=0 直接 `encode_frame` + `writer.write`；4) slot 分配改为 global `_frame_idx % pd`；5) output_count 累加非替换。

**Related:** [[nvenc-la-frame-conservation-fix]], [[nvenc-ce-pipeline-architecture]], [[pipeline-depth-slot-rotation-confusion]]
