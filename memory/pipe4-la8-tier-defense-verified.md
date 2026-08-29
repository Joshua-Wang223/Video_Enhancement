---
name: pipe4-la8-tier-defense-verified
description: pipe=4 + LA=8 GPU 验证：Tier 1-B/A 防御 100% 恢复空帧，lost=0，vs pipe=1 +7.3% FPS
metadata: 
  node_type: memory
  type: project
  originSessionId: f2f87520-6146-4edf-840f-d91c8a3434ff
---

# pipe=4 + LA=8: Tier 1-B/A 防御 GPU 验证通过

## 结论

**pipe=4 + LA=8 已完全可行**。VBR_HQ/QVBR 下 ~0.3-0.6% 空帧由 Tier 1-B (`encode_frame` 重试) + Tier 1-A (`prev_h264` 回退) 100% 恢复。吞吐比 pipe=1+LA=8 高 +7.3%。

## GPU 验证数据 (Tesla T4, 720x576@30fps, N=2000, VBR_HQ, QP23)

### Plan A 防御效果 (bs=1000)

| 路径 | FPS | empty_rate | 防御 | lost |
|------|:---:|:---:|:---:|:---:|
| ce-pipeline pipe=4 LA=0 | 379 👑 | 0% | — | 0 |
| ce-pipeline pipe=4 LA=8 | 343 | **0.45%** | Def=9 (1B=6, 1A=3) | **0** ✅ |
| ce-pipeline pipe=1 LA=8 | 320 | 0% | — | 0 |

### 不同 batch_size 防御效果

| bs | pipe=4 LA=8 FPS | 空帧率 | Def | lost |
|:--:|:---:|:---:|:---:|:---:|
| 24 | 334.5 | 0.3% raw | — | 0 |
| 500 | 335.1 | ~0.75% | 15 (1B=10, 1A=5) | 0 |
| 1000 | 343.0 | ~0.45% | 9 (1B=6, 1A=3) | 0 |

### Tier 1-B/A 分担率

- **Tier 1-B** (encode_frame 同步重试): 恢复了 **60-67%** 的空帧
- **Tier 1-A** (prev_h264 回退): 补偿了 **33-40%** 的空帧
- **综合**: 100% 空帧被防御恢复，lost_frames=0

## 实施

v6.4.3.1/4.1 的 `NVENCEncoder.__init__` 中移除 VBR/QVBR 的 pipe 强制降级：

```python
# 改前 (强制 pipe=1):
if la_depth > 0 and pipeline_depth > 1:
    if rate_mode == "constqp": pass
    else: pipeline_depth = 1

# 改后 (保留 pipe=4):
if la_depth > 0 and pipeline_depth > 1:
    # 依赖 Tier 1-B/A 防御处理 ~0.5% 空帧（GPU 验证 100% 可恢复）
    # 不做降级
```

CONSTQP 路径不变（LA=8 + pipe=4 本就 0% 空帧）。

## 背景

早期 `_NVENC_LEVEL1_LOOKAHEAD = 0` 因 pipe=4+LA 的 slot-rotation 空帧问题而全局禁用 LA。空帧机制：

```
帧 0-3 → slot 0-3 → NEED_MORE_INPUT → CE destroy
帧 4-7 → slot 0-3 → EncodePicture SUCCESS
     → harvest slot 0 → 拿到帧 4 的输出
     → 但帧 0 的 LA 输出写入 slot 0 → 与帧 4 数据冲突 → 空帧
```

pipe 强制降级到 1 消除了空帧但也损失了 ~15% 吞吐。GPU 验证证明现有 Tier 防御可完全处理残余空帧。

## v4 更新: 真实视频上 LA=8 零空帧

v4 4D全矩阵测试 (真实视频 word_world_2.mp4) 发现 **LA=8 + pipe=4 在真实视频上 0% 空帧**，无论 RC 模式。
v3 的 0.3-0.6% 空帧是合成帧 (无帧间相关性) 的 artifact。
详见 [[la8-real-vs-synthetic-reversal]]。

## v4 更新: phase4 系列在 constqp 下 Tier 防御失败

constqp 的高速编码 (~500+ FPS) 缩小了 DMA 竞态窗口，Tier 1-B 固定 backoff 不足覆盖 →
phase4-slot/phase4-pfce 在 constqp 下永久丢帧 (LOST=59~119)。
**ce-pipeline 不受影响** — deferred harvest 的时间窗口天然覆盖竞态。
详见 [[phase4-constqp-defense-failure]]。

## v6.4.x 最终结论 (2026-06-17 更新)

Tier 防御可恢复**同步空帧** (DMA 竞态)，但无法修复 per-slot 多 IDR +
encode_frame 混用导致的**结构性空帧** (40×LA 开销帧)。

根因已定位并修复: 删除 encode_frame() + fi==0 only force_idr → 零 IDR → 零空帧。
**pipe forced guard 已注释，pipe=4+LA=8 安全恢复。**

详见 [[pipe4-la8-root-cause-fix]]。

**Why:** [[pipe4-la8-root-cause-fix]] 根因修复后 pipe=4+LA=8 零空帧，不再需要 Tier 防御或 pipe 强制降级
**How to apply:** pipe forced guard 已注释；三文件的 `[FIX-PIPE4-LA8]` 标签定位所有修改点

