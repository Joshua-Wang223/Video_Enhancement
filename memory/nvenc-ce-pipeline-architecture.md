---
name: nvenc-ce-pipeline-architecture
description: NVENC ce-pipeline 异步编码架构设计、三阶段流程、性能数据和实施要点
metadata: 
  node_type: memory
  type: project
  originSessionId: e47eeb99-6d8c-4065-8b2f-1ca21ec2534e
---

# NVENC CE-Pipeline 异步编码架构

## 核心设计

`encode_frames_batch_ce_pipeline()` 是 NVENC 编码的最高性能路径。

### 关键洞察

**EncodePicture 同步阻塞是瓶颈** — 传统方式中调用线程被卡住等待 NVENC 硬件完成编码。CE-pipeline 通过 per-frame CUDA completionEvent 消除这一阻塞。

### 三阶段流程

```
Phase 1 (Harvest):  slot 重用时 cuEventSynchronize 等待上一轮 CE → LockBitstream 获取 H.264
Phase 2 (Submit):   LockInputBuffer → cuMemcpy2D → EncodePicture + cuEventCreate(新CE)
Phase 3 (Drain):    批次结束 drain 所有 pending slots
```

**核心机制**: EncodePicture 提交时附带 per-frame CE 立即返回；LockBitstream 延迟到 slot 下次轮转时执行（pipe_depth 帧后），此时 CE 已触发→立即拿到数据。

## GPU 验证性能数据 — v4 4D全矩阵最终版 (Tesla T4, 720×576@25fps 真实视频, 687帧, 2/3/4段)

### 各 RC 模式最优配置

| RC Mode | 最优配置 | FPS | 提升 vs sync-batch pipe=1 | 文件大小 |
|---------|---------|----:|:-------------------------:|:-------:|
| **constqp** | ce-pipeline pipe=4 LA=8 | **584** | +20% | 6.2 MB |
| qvbr | ce-pipeline pipe=4 LA=0 | 557 | +20% | 7.2 MB |
| vbr_hq | ce-pipeline pipe=4 LA=0 | 455 | +38% | (未导出) |

### ce-pipeline pipe=4 LA=8 跨段验证 (constqp)

| 段数 | FPS | 空帧 | 丢帧 |
|:----:|----:|:----:|:----:|
| 2seg | 579.8 | 0% | 0 |
| 3seg | 584.7 | 0% | 0 |
| 4seg | 584.2 | 0% | 0 |

**段数对性能影响 <1%** — encoder 复用机制充分摊销 open/close 开销。

### LA=8 行为逆转

| 输入类型 | LA=8 vs LA=0 | 空帧 |
|----------|:------------:|:----:|
| 合成随机帧 (v3) | **-3.7%** | 0.3% |
| 真实视频 (v4) | **+1.1%** | 0% |

LA 收益取决于帧间相关性 — 真实视频上 LA 预分析可早决策帧类型，合成无相关帧反而增加延迟。详见 [[la8-real-vs-synthetic-reversal]]。

## 实施要求

1. `pipeline_depth` 默认 4 (范围 1-8)
2. 每 slot: `{input_buf, bs_buf, event}` — 初始化时 `cuEventCreate(0)`
3. `_slot_pending[pd]` 跟踪每槽的 `(ce_handle, frame_index, ep_status, force_idr)`
4. `_slots_warmed` set 跟踪已初始化 DPB 的 slot
5. LA 处理: `NEED_MORE_INPUT` 时 CE destroy + 返回 `b""`
6. 多 slot flush: 遍历所有 slot 的 `bs_buf` 排空

## LA 与 pipeline 兼容性 (2026-06-15 更新)

| RC Mode | LA=8 + pipe=4 | empty | defense | 结论 |
|---------|:---:|:---:|:---:|------|
| CONSTQP | 0% | 0 | — | ✅ 安全 |
| VBR_HQ | **0.45%** | 0 | 1-B(67%)+1-A(33%) | ✅ **可行** (Tier 防御 100% 恢复) |
| QVBR | 0.3-0.6% | 0 | 同上 | ✅ |

**旧数据 (2026-06-11) 已过时**：VBR_HQ "0.9% 空帧 FPS 倒退 18% → 强制 pipe=1" 是在 Tier 1-B/A 防御未启用时的结论。
当前 Tier 防御可以 100% 处理残余空帧，pipe=4+LA=8 比 pipe=1 快 +7.3%。

详细验证数据见 [[pipe4-la8-tier-defense-verified]]。

## 🚨 v6.4.5/.1 bitrate=unconstrained 性能塌方警告 (2026-06-16)

v6.4.5/.1 在 VBR_HQ/QVBR 模式下将 `averageBitRate=0` + `maxBitRate=0`，
导致 NVENC 无码率天花板约束 → 质量搜索进入极慢模式 → GPU 闲置 65-70% →
FPS 从 64 暴跌至 18.9 (2.2× 慢)。恢复 avgBitRate 估计值可修复。

详见 [[v6.4.5-bitrate-unconstrained-degradation]]。

## LA=8 + pipe=4 启动 SPS/PPS 损坏 (2026-06-16)

LA buffering 期间的空帧 + pipe=4 slot rotation 可能触发 ctypes `repeatSPSPPS`
无效或 muxer SPS/PPS 上下文未建立，导致 `non-existing PPS 0 referenced` 错误。
v6.4.3.1 在 CRF=23 batch 中因此丢失 62 帧 (-4.5%)。

详见 [[sps-pps-la-pipe4-startup-corruption]]。

**Why:** 性能提升来自消除 EncodePicture 同步阻塞，让 NVENC HW 的帧间流水线化生效
**How to apply:** backport 到任一版本需修改 12-16 处: 构造函数、buffer创建、destroy方法、flush、_writer_loop/_loop
