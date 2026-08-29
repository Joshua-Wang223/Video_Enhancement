---
name: v4-production-best-config
description: v4 4D全矩阵测试最终结论 — ce-pipeline+pipe=4+LA=8+constqp 为生产最优解，584 FPS 零丢帧
metadata: 
  node_type: memory
  type: project
  originSessionId: 2524a2cc-7f5a-402b-82bd-9fad0c6238b6
---

# v4 4D 全矩阵测试 — 生产最优配置

## 最终推荐

```
ce-pipeline + pipe=4 + LA=8 + constqp
  → 584 FPS (Tesla T4, 720×576@25fps 真实视频)
  → 零丢帧 (lost=0)
  → 零空帧 (0% empty)
  → 输出 6.2 MB (H.264 ES, 687帧)
  → 比 sync-batch pipe=1 基准快 +20%
```

## v3 → v4 关键变化

| 发现 | v3 (合成帧, 2000帧) | v4 (真实视频, 687帧) |
|------|:-------------------:|:---------------------:|
| LA=8 + pipe=4 空帧 | 0.3% (6帧空) | 0% |
| LA=8 vs LA=0 FPS | LA惩罚 -3.7% | LA增益 +1.1% |
| 测试维度 | 2D (tech×pipe×LA) | **4D** (RC×tech×pipe×LA×segments) |
| 输入 | 合成噪声帧 | 真实视频 word_world_2.mp4 |
| 防御 | FLUSH恢复 | Tier 1-B retry + Tier 1-A prev_h264 |

## 三种段数验证

| 段数 | 每段帧数 | ce-pipeline pipe=4 LA=8 constqp FPS |
|:----:|---------|------------------------------------:|
| 4seg | ~172 | 584.2 |
| 3seg | ~229 | 584.7 |
| 2seg | ~344 | 579.8 |

段数对 FPS 影响 <1%，encoder 复用机制验证通过。

## 输出文件验证

TOP-3 输出文件全部 frame-accurate (输入687 = 输出687 = 容器687)，编码器 h264_nvenc：
- constqp: 6.2 MB
- qvbr: 7.2 MB (质量略好，文件大 16%)

## 决策优先级

| 优先级 | 配置 | 适用场景 |
|:------:|------|----------|
| 1 | ce-pipeline pipe=4 LA=8 constqp | 生产默认 |
| 2 | ce-pipeline pipe=4 LA=0 constqp | 保守方案 (580 FPS) |
| 3 | ce-pipeline pipe=1 LA=8 constqp | 单 slot 场景 (495 FPS) |
| 避免 | phase4-slot / phase4-pfce | 任何 pipe=4 配置都丢帧 |

**Why:** 4D 全矩阵 180 组合 GPU 验证，真实视频输入，constqp 在性能和文件大小双赢
**How to apply:** IFRNet v6.4.3/4/5.1 的 `_NVENC_LEVEL1_LOOKAHEAD` 设为 8，`_NVENC_LEVEL1_PIPELINE_DEPTH` 设为 4，`_NVENC_LEVEL1_RATE_MODE` 设为 constqp
