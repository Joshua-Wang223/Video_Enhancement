---
name: la-harvest-dead-end
description: LA-Aware Harvest (completion-driven) 实验失败：12-slot 开销+cuEventQuery 轮询使其比 pipe=1 ce-pipeline 还慢
metadata: 
  node_type: memory
  type: project
  originSessionId: f2f87520-6146-4edf-840f-d91c8a3434ff
---

# LA-Aware Harvest 实验 — 死胡同

## 方案

用 completion-driven harvest 替代 slot-rotation harvest：
- 槽位扩容: `pipeline_depth + la_depth` (12 slots)
- 空闲池: `deque` 而非 `fi % pd` 轮转
- 收获: `cuEventQuery` 轮询 CE 触发时而非 slot 轮转时
- LA 帧 CE 保留（不 destroy）

## GPU 验证结果 (Tesla T4, N=2000, bs=1000)

| 路径 | FPS | empty | lost | 结论 |
|------|:---:|:---:|:---:|------|
| ce-pipeline pipe=4 LA=0 | 371 👑 | 0% | 0 | 基线 |
| ce-pipeline pipe=4 LA=8 | 354 | 0.3% | 0 | 现有最佳 |
| **la-harvest pipe=4 LA=8** | **319** ❌ | 0.3% | 0 | 失败 |

la-harvest 比 ce-pipeline pipe=4 LA=8 **慢 10%**，比 ce-pipeline pipe=1 LA=8 **慢 0.9%**。

## 失败原因

1. **12-slot 初始化开销**: `CreateInputBuffer × 12 + CreateBitstreamBuffer × 12` 增加启动延迟
2. **cuEventQuery 轮询**: 每帧遍历 12 个 slot 调用 `cuEventQuery` → 12 × 1μs × 2000 帧 = 不可忽略
3. **LA 缓冲前 8 帧**: 全部 NEED_MORE_INPUT，无 slot 可复用，无法利用多槽并行
4. **NVENC 内部流水线**: 12 slot 反而打散了 NVENC 的帧间优化（ME 参考帧更分散）

## 结论

**completion-driven harvest 是反模式**。现有 slot-rotation + Tier 防御（方案 A）优于重新设计收获机制。

**Why:** 12-slot 初始化和 cuEventQuery 轮询开销超过了消除 0.5% 空帧的收益
**How to apply:** 不要实施 la-harvest；使用方案 A（pipe=4+LA+现有 Tier 防御）
