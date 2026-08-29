---
name: nvenc-la-frame-conservation-fix
description: NVENC SDK 合规排空逻辑的 3 项修复：循环 LockBitstream、pipeline_depth=LA+1、移除 NEED_MORE_INPUT 误弃帧
metadata: 
  node_type: memory
  type: project
  originSessionId: 71a52bbb-72b5-45ca-94eb-834c56f9e215
---

# NVENC Lookahead 帧数守恒 — 正确实现方案

## SDK 声明（官方文档，已 GPU 验证为真）

> "Lookahead depth adds latency. The total number of output frames equals
>  the number of input frames. No frames are discarded by the encoder."

—— NVIDIA Video Codec SDK Programming Guide, 《Lookahead》章节

## constqp 下 LA 自动无效

NVIDIA 官方文档明确：Lookahead 仅在 VBR/QVBR 码控模式下生效。
当 `rateControlMode=CONSTQP` 且 `enableLookahead=1` 时，NVENC 硬件**静默禁用 LA** (la_depth=0)。
因此 constqp 模式的测试不足以验证 LA 排空逻辑正确性，**必须用 VBR_HQ/QVBR 测试**。

## 旧版代码的 3 个 SDK 违规（导致开头丢帧+末尾重复的根因）

### BUG 1：未循环排空输出帧
- SDK 要求：EncodePicture 后必须反复调用 LockBitstream (blocking) 直到 NEED_MORE_INPUT
- 旧版：只 Lock 一次 → 硬件同时完成多帧时，部分帧滞留在其他 slot 缓冲区中被覆盖

### BUG 2：NEED_MORE_INPUT 时强制 Lock/Unlock 丢弃有效帧
- 旧版在 EncodePicture 返回 NEED_MORE_INPUT 时 Lock/Unlock 对应 slot 的 bs_buf
- 若因 BUG 1 该 slot 残留之前已完成但未取走的帧 → 被静默取走并丢弃

### BUG 3：pipeline_depth 不足（`max(1, LA)` 而非 `max(1, LA+1)`）
- SDK 硬件安全要求：启用 lookahead 时比特流缓冲区数量至少 `lookaheadDepth + 1`
- `max(1, LA)` 导致缓冲区覆盖 → 开头帧数据被破坏

## 正确的编码循环（修复方案）

### encode_frame: 送一帧 + 立即排空
```python
def encode_frame(self, nv12_gpu, force_idr=False) -> List[bytes]:
    slot_idx = self._frame_idx % self._pipeline_depth
    # ... LockInputBuffer, cuMemcpy2D, EncodePicture ...
    self._frame_idx += 1
    return self._drain_outputs()  # 立即排空所有完成帧

def _drain_outputs(self) -> List[bytes]:
    outputs = []
    while True:
        slot_idx = self._output_slot_idx % self._pipeline_depth
        data, status = self._lock_bitstream_once(slot_idx, count=True)
        if status == NV_ENC_ERR_NEED_MORE_INPUT:
            break
        if status != NV_ENC_SUCCESS or not data:
            break
        outputs.append(data)
        self._output_slot_idx += 1
    return outputs
```

### flush_eos: 按输出指针顺序排空
```python
def flush_eos(self) -> List[bytes]:
    # Send EOS (NULL input + flag=0x8)
    ...
    start_slot = self._output_slot_idx % self._pipeline_depth
    for slot_idx in drain_order:  # 按输出指针轮转顺序
        while True:
            data, status = LockBitstream(blocking)
            if status == NEED_MORE_INPUT: break
            output_frames.append(data)
    return output_frames
```

### 硬件配置
```python
pipeline_depth = max(1, la_depth + 1)  # 必须 >= LA+1, 非 LA
```

## 修复后的验证结果

| 测试 | RC 模式 | LA | Input | Output | 结果 |
|------|---------|:--:|:-----:|:------:|:----:|
| A | constqp | 0 | 687 | 687 | ✅ OK |
| C | vbr_hq  | 8 | 687 | 687 | ✅ OK |
| C | qvbr    | 8 | 687 | 687 | ✅ OK |

所有 RC 模式 + LA 组合均 Output == Input，帧顺序严格一致。

## 生产脚本修复（v6.4.3.1 / 4.1 / 5.1 — 2026-07-03）

### 各版本 Bug 清单

| Bug | 严重度 | v6.4.3.1 | v6.4.4.1 | v6.4.5.1 | 描述 |
|-----|--------|:---:|:---:|:---:|------|
| BUG-1 | 🔴 | ✅ | ✅ | ✅ | `encode_frames_batch` 仅 LockBitstream 当前 slot，不排空其他 |
| BUG-2 | 🔴 | ✅ | ✅ | ✅(变体) | NEED_MORE_INPUT 后 skip drain；v6.4.3.1/4.1 还会强制 Lock/Unlock 丢弃帧 |
| BUG-3 | 🔴 | ✅ | ✅ | — | `pipeline_depth = max(1, min(8, 4))` = 4，LA=8 时不足（v6.4.5.1 已修复自动校准） |
| BUG-B | 🟡 | ✅ | ✅ | ✅ | `encode_frame` 仅 LockBitstream slot 0 |
| BUG-C | 🟡 | — | — | ✅ | ce_pipeline `_reset_output_slot_idx(0)` 每批次重置 → 帧序号不同步 |
| BUG-D | 🟡 | ✅ | ✅ | ✅ | ce_pipeline + `_eos_sent` 防止后续批次发送 EOS → LA 帧卡死 |

### 修复方案（所有版本统一模式）

#### 修复 1：`encode_frames_batch()` — 全局 drain

```python
# 在方法开头：pre-allocate results + _slot_pending 映射
results = [None] * n_frames
_slot_pending = [None] * self._pipeline_depth  # (fi, bs_buf, force_idr, ep_status)

# 每帧提交后：
_slot_pending[slot_idx] = (fi, slot['bs_buf'], force_idr, _ep_status)

# NEED_MORE_INPUT 后不 continue，继续执行全局 drain：
_drained = self._drain_outputs_blocking()
for _est_fi, _h264_data in _drained:
    _drain_slot = _est_fi % self._pipeline_depth
    _pending = _slot_pending[_drain_slot]
    # ... 映射到 results[_actual_fi]，SPS/PPS 缓存，清除 _slot_pending

# 最终 drain + None→b"" 填充
```

关键：将单 slot `_lock_bitstream_with_retry(slot['bs_buf'])` 替换为 `self._drain_outputs_blocking()`，通过 `_slot_pending[slot]` 映射 drain 输出的 `_est_fi` 到实际帧索引 `_actual_fi`。

#### 修复 2：LA>0 路由到同步 batch（非 ce_pipeline）

```python
# EncodeThread._loop() 中：
if self._nvenc._la_depth > 0:
    h264_list = self._nvenc.encode_frames_batch(nv12_list, force_idr)
else:
    h264_list = self._nvenc.encode_frames_batch_ce_pipeline(nv12_list, force_idr)
```

理由：ce_pipeline 的 per-batch EOS 打断 LA 连续流 + `_output_slot_idx` 重置导致帧序号不同步。LA=0 时 CE 异步流水线保留性能优势。

#### 修复 3：`encode_frame()` — 全局 drain

```python
# 替换单 slot LockBitstream：
_drained = self._drain_outputs_blocking()
h264_data = _drained[0][1] if _drained else b""
```

#### 修复 4：v6.4.5.1 移除 `_eos_sent` 标志

EOS 仅在 `flush()` 中发送一次，ce_pipeline 不再发送 per-batch EOS。

### 基础设施要求（v6.4.3.1/4.1 已 backport）

- `_drain_outputs_blocking()` — 按 `_output_slot_idx` 顺序循环 blocking LockBitstream
- `_output_slot_idx` — 输出槽位指针
- `_reset_output_slot_idx()` — 批次开始时重置
- `pipeline_depth = max(pipeline_depth, LA+1)` — SDK 自动校准

### ✅ GPU 验证通过（2026-07-04, Tesla T4）

Benchmark 验证结果（687 帧, VBR_HQ+LA=8, 2x 插帧, crf=21）：

| 版本 | 输出帧 | 容器帧 | 文件Δ | 空H264 | Lock降级 | FPS |
|------|--------|--------|-------|--------|---------|-----|
| v6.4.4 (基准) | 1373 | 1373 | ✓ | 0 | 0 | 61.7 |
| v6.4.3.1 | 1373 | 1373 | ✓ | 0 | 0 | 55.6 |
| v6.4.4.1 | 1373 | 1373 | ✓ | 0 | 0 | 55.5 |
| v6.4.5.1 | 1373 | 1373 | ✓ | 0 | 0 | 56.9 |

**零丢帧、零空帧、零 LockBitstream 降级。管內偏移 = +0。文件Δ = ✓。**

### 第二轮修复：首帧 f0 恢复（2026-07-04）

第一轮 LA 累积修复后 benchmark 发现 3 个版本仍比 v6.4.4 基准少 1 帧（1372 vs 1373）。
根因：v6.4.5.1 引入的 `[FIX-PIPE4-LA8]` 将 `encode_frame` 替换为 `pass`，错误假设 CE pipeline 会输出 f0。参见 [[f0-first-frame-loss-ce-pipeline]]。

### 相关文件

- `tests/NVENC Lookahead 帧守恒修复IFRNet生产脚本的方案要点.txt` — 需求文档
- `tests/NVENC Lookahead 帧数守恒验证与实现问题的完整总结.txt` — 根因分析
- `external/IFRNet/process_video_v6_4_5_1_single.py` — 修复完成 ✅
- `external/IFRNet/process_video_v6_4_3_1_single.py` — 修复完成 ✅
- `external/IFRNet/process_video_v6_4_4_1_single.py` — 修复完成 ✅

## 测试脚本
- `tests/test_nvenc_la_frame_conservation.py` — 修复版测试脚本
- `tests/deepseek_python_20260630_636184.py` — 独立修复样例
- `tests/NVENC Lookahead 帧数守恒验证与实现问题的完整总结.txt` — 完整分析文档

**Why:** 旧版排空逻辑的 3 个 SDK 违规导致开头丢帧+末尾重复，被误判为"NVENC LA 设计行为"。正确的 SDK 合规排空逻辑证明帧数守恒声明为真。
**How to apply:** 1) pipeline_depth = max(1, LA+1); 2) 每帧 EncodePicture 后循环 LockBitstream 直到 NEED_MORE_INPUT; 3) 移除 NEED_MORE_INPUT 时的 Lock/Unlock; 4) EOS flush 按输出指针顺序排空; 5) 参见 [[la-flush-recovery-is-harmful]] 修正后的记忆。
