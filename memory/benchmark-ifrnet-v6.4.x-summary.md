---
name: benchmark-ifrnet-v6-4-x-summary
description: IFRNet v6.4.3→v6.4.5.1 六版本基准测试综合分析报告，CRF=23 batch + CRF=21 individual 双轮验证
metadata: 
  node_type: memory
  type: project
  originSessionId: b6bcafb5-af2b-4b88-95e4-bcb2e53f3525
---

# IFRNet v6.4.x 全版本基准测试综合报告

## 测试配置

- **GPU**: Tesla T4 (14.6 GiB) | **输入**: word_world_2.mp4 (687帧, 720×576)
- **插帧**: 2x | **Batch-size**: 24 | **模型**: IFRNet_S_Vimeo90K
- **TRT**: 启用 | **CUDA Graph**: 关闭 | **Compile**: 关闭
- **两轮测试**: Batch (CRF=23, `--no-warmup`) + Individual (CRF=21, 详细日志)

## 性能排名 (Batch, CRF=23, pipe=4)

| 排名 | 版本 | 耗时 | FPS | 输出帧 | 文件 | RC模式 | GPU% |
|------|------|------|-----|--------|------|--------|------|
| 🥇 | **v6.4.4** | 20.9s | **65.7** | 1373 ✓ | 8.9MB | CONSTQP | 57.6% |
| 🥈 | v6.4.5 | 21.4s | 64.1 | 1373 ✓ | 12.9MB | VBR_HQ | 57.0% |
| 🥈 | v6.4.5.1 | 21.4s | 64.1 | 1373 ✓ | 13.2MB | QVBR | 55.5% |
| 4 | v6.4.4.1 | 21.5s | 63.5 | 1368 ⚠️ | 14.2MB | VBR_HQ | 56.5% |
| 5 | v6.4.3.1 | 21.4s | 61.4 | 1311 ❌ | 13.0MB | VBR_HQ | 47.3% |
| 6 | v6.4.3 | 29.4s | 46.7 | 1373 ✓ | 8.7MB | CONSTQP | 40.4% |

**性能差距**: 40.7% (最快 vs 最慢)

## 帧完整性: 丢帧根因

```
版本         期望帧   实际帧   丢失   根因
v6.4.3       1373     1373      0   CONSTQP + pipe=4, 稳定
v6.4.3.1     1373     1311    -62   VBR_HQ + LA=8 + pipe=4 → SPS/PPS 损坏 ❌
v6.4.4       1373     1373      0   CONSTQP + pipe=4, 稳定
v6.4.4.1     1373     1368     -5   VBR_HQ + LA=8 + pipe=4 → 轻微空帧 ⚠️
v6.4.5       1373     1373      0   VBR_HQ (无LA), 稳定
v6.4.5.1     1373     1373      0   QVBR (无LA), 稳定
```

**v6.4.3.1 丢62帧 (-4.5%)** 最严重 — 根因为 LA=8 + pipe=4 启动时空帧 #1 触发 SPS/PPS
未正确写入 muxer 头部，导致 FFmpegMuxer 持续报 `non-existing PPS 0 referenced`。

## RC 模式三维对比

| 维度 | CONSTQP | VBR_HQ | QVBR |
|------|---------|--------|------|
| 速度 | **最快** (65.7 FPS) | 中等 (61-64 FPS) | 中等 (64 FPS) |
| 文件大小 | **最小** (8.7-8.9 MB) | 较大 (12.9-14.2 MB) | 中等 (13.2 MB) |
| 稳定性 | ✅ 零空帧 | ⚠️ LA时空帧+SPS损坏 | ⚠️ LA时性能退化 |
| pipe=4 兼容 | ✅ | ⚠️ LA=8时有风险 | ⚠️ LA=8强制pipe→1 |

**结论**: CONSTQP 三维全面领先 — 验证了 [[rc-mode-performance-ranking]] 的发现。

## v6.4.5/.1 Individual Run 性能塌方

Batch (CRF=23) 与 Individual (CRF=21) 运行结果出现严重矛盾：

| 版本 | Batch CRF=23 | Individual CRF=21 | 退步倍数 | 根因 |
|------|-------------|-------------------|---------|------|
| v6.4.5 | 64.1 FPS | 18.9 FPS | **2.2× 慢** | VBR_HQ bitrate=unconstrained |
| v6.4.5.1 | 64.1 FPS | 27.9 FPS | 1.4× 慢 | QVBR bitrate=unconstrained |

**根因**: v6.4.5/.1 的 `averageBitRate=0` + `maxBitRate=0` 导致 NVENC 在没有码率约束
的情况下进入极慢的质量搜索模式，GPU 利用率仅 35%，空闲 65%。v6.4.3.1 使用
`avgBitrate=62208kbps` 则正常运行。详见 [[v6.4.5-bitrate-unconstrained-degradation]]。

## LA=8 + pipe=4 兼容性矩阵再验证

| RC Mode | LA=0 + pipe=4 | LA=8 + pipe=1 | LA=8 + pipe=4 |
|---------|:---:|:---:|:---:|
| CONSTQP | ✅ | — | ✅ (0空帧) |
| VBR_HQ | ✅ | ✅ | ❌ SPS/PPS损坏 + 空帧 |
| QVBR | ⚠️ 慢(无avgBr) | ✅ (auto pipe→1) | ❌ SPS/PPS损坏 |

此矩阵验证了 [[nvenc-ce-pipeline-architecture]] 的兼容性表和
[[pipe4-la8-tier-defense-verified]] 的修复方向。

## 自适应调优 (Auto-Tune) 共性偏差

全部版本报告 1019-1039% 的 T2 估算偏差（估算 30ms vs 实测 335ms），详见 [[t2-static-estimation-undershoot]]。

## 生产推荐

| 场景 | 推荐版本 | RC模式 | 理由 |
|------|---------|--------|------|
| **生产（推荐）** | **v6.4.4** | CONSTQP | 最快+最稳+文件最小 |
| 需要码率控制 | v6.4.5 | VBR_HQ (no LA) | 帧完整，性能好 |
| 最新代码基线 | v6.4.5.1 | QVBR (no LA) | 帧完整，性能好 |

## 避免的组合

- v6.4.3.1 + VBR_HQ + LA=8 + pipe=4 → 大量丢帧
- v6.4.5/.1 + VBR_HQ/QVBR + LA=8 + pipe=4 → SPS/PPS损坏
- v6.4.5/.1 + VBR_HQ + bitrate=unconstrained → GPU闲置65%

**Why:** 六版本横向对比，验证v6.4.4为最优生产版本，确认CONSTQP综合最佳，发现v6.4.5/.1 bitrate=unconstrained退化bug
**How to apply:** 生产环境使用 v6.4.4 + CONSTQP；v6.4.5/.1 需先修复 bitrate=unconstrained 问题
