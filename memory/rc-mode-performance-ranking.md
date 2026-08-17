---
name: rc-mode-performance-ranking
description: RC三种模式(constqp/qvbr/vbr_hq)性能排名 — constqp 统一最快，比vbr_hq快29~66%
metadata: 
  node_type: memory
  type: project
  originSessionId: 2524a2cc-7f5a-402b-82bd-9fad0c6238b6
---

# RC 模式性能排名 (v4 4D全矩阵, Tesla T4, 真实视频)

## 统一排名

**constqp > qvbr > vbr_hq** — 所有技术路线、所有 pipe/LA 组合一致。

## 典型 FPS 对比 (ce-pipeline pipe=4)

| LA | constqp | qvbr | vbr_hq | constqp/vbr_hq |
|:--:|--------:|-----:|-------:|:--------------:|
| LA=0 | 578 | 557 | 449 | **1.29×** |
| LA=8 | 584 | 488 | 352 | **1.66×** |

## 文件大小

| RC 模式 | 687帧输出 | 相对大小 |
|---------|:--------:|:-------:|
| constqp | 6.2 MB | 基准 |
| qvbr | 7.2 MB | +16% |

## 根因分析

- **constqp** 最快：无码率计算开销，固定 QP 直接量化，NVENC HW 路径最短
- **qvbr** 居中：有质量因子计算但无两遍分析，码率约束轻量
- **vbr_hq** 最慢：targetQuality 驱动 + NVENC 内部重分析，CPU/GPU 开销最大

## vbr_hq 仅在 LA=8 时加速缩小差距

```
ce-pipeline pipe=4 LA=0: constqp 578 / vbr_hq 449 → +29%
ce-pipeline pipe=4 LA=8: constqp 584 / vbr_hq 352 → +66%
```

vbr_hq 在 LA=8 时不仅没加速反而更慢，因为 LA 的分析开销叠加 targetQuality 的计算开销。

## 生产建议

离线视频增强场景：constqp 统一最优 — 最快 + 文件最小。
需要质量优先的场景：qvbr (文件大 16%，FPS 低 ~5%)。
**不要用 vbr_hq** — 慢 30~66% 且文件更大，无任何优势。

**Why:** GPU 验证 180 组数据，constqp 在所有维度上碾压 vbr_hq/qvbr
**How to apply:** `_NVENC_LEVEL1_RATE_MODE = 'constqp'`；如需质量优先用 qvbr，永不使用 vbr_hq
