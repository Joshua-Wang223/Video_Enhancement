---
name: esrgan-la-nv12-accumulation-oom
description: ESRGAN VBR_HQ+LA=8 NV12 GPU tensor 全段累积导致 OOM — 分块编码修复
metadata:
  type: project
---

## 问题

ESRGAN 超分 VBR_HQ+LA=8 (9 slots) 下 GPU 显存阶梯式暴涨至 100% → SR 连续 OOM → batch_size 死亡螺旋 (24→12→6→3→1)，FPS 从 17 暴跌至 1.2。

## 根因

`_NVENCEncodeThread._loop()` LA>0 路径将整个 segment 的所有 NV12 GPU tensor 累积在 `_acc_nv12` 中，segment 结束时（收到 SENTINEL）才调用 `encode_frames_batch(_acc_nv12, send_eos=True)`。

每帧 NV12 (1536×1152) ≈ 2.5 MB → 9000 帧 ≈ 22.8 GB → 叠加 SR TRT engine → 16 GB GPU 必然 OOM。

**第一次修复失败原因**: `nv12_tensors[fi] = None` 在 `encode_frames_batch()` 内部逐帧释放，但该函数在 segment 结束时才被调用。峰值在调用前已发生。

## 修复 (v2): 分块编码 (Chunked Encoding)

**核心思路**: 将 LA 模式从"全段累积一次编码"改为"分块累积逐块编码"。

### 关键使能: 持久化 `self._slot_pending`

`encode_frames_batch()` 中 `_slot_pending` 原是局部变量（每次调用重置），负责将 drain 返回的 slot → frame_index 映射。分块后若使用局部变量，chunk 2 的 drain 会因 `_slot_pending` 全 None 而静默丢弃 chunk 1 的延迟 LA 输出。

改为 `self._slot_pending` 实例变量，跨 `encode_frames_batch()` 调用保留未 drain 的 slot 映射。

### 修改清单

| 文件 | 位置 | 修改 |
|------|------|------|
| `nvenc_sdk.py:__init__` | L541 | 添加 `self._slot_pending` + `self._slots_warmed` 持久化变量 |
| `nvenc_sdk.py:encode_frames_batch` | L1222-1224 | 移除局部 `_slot_pending` / `_slots_warmed`，改用 `self.` 引用 |
| `nvenc_sdk.py:_loop()` | L2170-2212 | `_LA_CHUNK_SIZE=150`，while 循环内分块提交 |
| `nvenc_sdk.py:_loop()` | L2238-2256 | 最终块 EOS 排空（处理剩余帧或空 EOS drain） |
| `nvenc_sdk.py` | L2273-2296 | 新增 `_write_la_output()`：中间块跳过 b""，最终块 fallback |

### 内存效果

| 场景 | 修复前峰值 | 修复后峰值 |
|------|----------|----------|
| 9000 帧 @ 1536×1152 | ~22.8 GB | ~380 MB (CHUNK_SIZE=150) |
| 500 帧 @ 1536×1152 | ~1.27 GB | ~380 MB |

### 防御性修复 (v1, 保留)

- `nvenc_sdk.py:encode_frames_batch()`: `nv12_tensors[fi] = None` after cuMemcpy2D（逐帧释放）
- `pipeline.py:825,846`: `torch.cuda.synchronize()` before `empty_cache()`（排空 cudaFreeAsync）

**Why:** cuMemcpy2D 是同步拷贝，完成后数据已在 NVENC input buffer。分块编码确保 `encode_frames_batch()` 被频繁调用（每 150 帧），NV12 tensor 在编码过程中逐个释放，而非 segment 结束时一次性处理。

**How to apply:** LA 模式下的帧累积必须有限制。CHUNK_SIZE 的选择原则: `total_vram - sr_engine_vram - nvenc_slot_vram - safety_margin`。150 帧在 1536×1152 下约 380 MB，保守安全。

## 相关记忆

- [[expandable-segments-synchronize-after-del]]
- [[trt-batch-recovery-anti-patterns]]
- [[realesrgan-nvenc-module-architecture]]
