---
name: pipe4-la8-production-garbled-fix
description: [已废弃] 方案A+方案B旧方案 — 被 [[pipe4-la8-root-cause-fix]] 替代，三文件三处就地修复更简洁有效
metadata: 
  node_type: memory
  type: project
  originSessionId: 2524a2cc-7f5a-402b-82bd-9fad0c6238b6
  status: superseded
  superseded_by: pipe4-la8-root-cause-fix
---

# ⚠️ 已废弃 — 见 [[pipe4-la8-root-cause-fix]]

## 旧方案对比

| 维度 | 旧方案A (此文件) | 新方案 (root-cause-fix) |
|------|----------------|----------------------|
| encode_frame 处理 | 转为 ce_pipeline([first_nv12]) | 直接 **删除** (pass) |
| IDR 策略 | 保留 per-slot | **fi==0 only** |
| 新文件 | 方案B 建议新建 .2 文件 | **无需新建**，就地修复 |
| 复杂度 | 6 文件 × 多步 | 3 文件 × 3 修改 |

## 原方案 (仅供参考)

原方案A 将 `encode_frame()` → `encode_frames_batch_ce_pipeline([first_nv12])`
仍会产生 1 个 IDR on slot0 → 1 × LA_DEPTH = 8 帧 LA 开销。
新方案直接删除 encode_frame 调用 → 首帧由 CE pipeline 自然处理 → 真正零开销。

---

# (以下为原始内容，仅作参考)

# pipe=4+LA=8 生产花屏修复 (v6.4.3.2/4.2/5.2 + .1补丁)

## 根因

`encode_frame()`（同步,slot0）与 `encode_frames_batch_ce_pipeline()`（异步,4-slot轮转）混用：
1. encode_frame 使用 slot 0 同步编码首帧 → LA=8 时 NMI → NVENC 内部缓冲帧数据引用 slot 0 buffer
2. 随后 ce_pipeline 轮转到 slot 0 → cuMemcpy2D 覆盖 input buffer → LA 最终编码时读取覆盖后数据 → **花屏**
3. per-slot 多 IDR（4个）+ LA 交互：每个 IDR 都 NMI → CE 循环创建/销毁 → 数据混叠加剧
4. SPS/PPS 在 ce_pipeline harvest 中预挂 → per-slot IDR 收到双份 SPS+PPS → FFmpegMuxer 上下文损坏

## 方案A（修复 v6.4.3.1/4.1/5.1）

两个就地修改：
1. `_process_segment` 首帧编码：`encode_frame()` → `encode_frames_batch_ce_pipeline([first_nv12], force_idr_first=True)`
2. RC guard 移除：所有 RC 模式 pipe=4+LA=8 均安全

## 方案B（新建 v6.4.3.2/4.2/5.2）

从 .1 复制并额外修改：
1. 方案A 全部修改
2. force_idr: `slot_idx not in _slots_warmed` → `fi == 0`
3. 移除 `_slots_warmed` per-slot DPB 跟踪
4. 移除 ce_pipeline 内部 SPS/PPS 预挂（由 `_NVENCEncodeThread._loop()` 统一注入）
5. 版本号更新

## 修改文件

| 文件 | 方案 |
|------|:----:|
| `process_video_v6_4_3_1_single.py` | A |
| `process_video_v6_4_4_1_single.py` | A |
| `process_video_v6_4_5_1_single.py` | A |
| `process_video_v6_4_3_2_single.py` | B (新建) |
| `process_video_v6_4_4_2_single.py` | B (新建) |
| `process_video_v6_4_5_2_single.py` | B (新建) |

所有 6 文件语法检查通过。

**Why:** encode_frame + ce_pipeline 混用 + per-slot 多 IDR + SPS/PPS 重复注入 → 生产花屏
**How to apply:** 已在 .1 文件中完成就地修复，.2 文件为完全对齐测试脚本的版本
