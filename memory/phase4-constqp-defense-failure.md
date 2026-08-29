---
name: phase4-constqp-defense-failure
description: phase4-slot/pfce 在 constqp 下 Tier 1-B/A 防御失败，vbr_hq/qvbr 下可恢复 — constqp 高速编码缩小了竞态窗口
metadata: 
  node_type: memory
  type: project
  originSessionId: 2524a2cc-7f5a-402b-82bd-9fad0c6238b6
---

# phase4 系列在 constqp 下防御失败

## 现象

v4 4D 全矩阵测试揭示 phase4-slot 和 phase4-pfce 在不同 RC 模式下表现分裂：

| 组合 | RC 模式 | 空帧率 | 防御效果 | LOST |
|------|---------|:-----:|:--------:|:----:|
| phase4-pfce pipe=4 LA=0 | **constqp** | 16-17% | ❌ 失败 | **93~119** |
| phase4-pfce pipe=4 LA=0 | vbr_hq | 0% | 🛡️ DEF=114~120 | 0 |
| phase4-pfce pipe=4 LA=0 | qvbr | 0% | 🛡️ DEF=114~120 | 0 |
| phase4-slot pipe=4 LA=0 | **constqp** | 11% | ❌ 失败 | **59~78** |
| phase4-slot pipe=4 LA=0 | vbr_hq | 12-13% | ❌ 失败 | 64~89 |
| phase4-slot pipe=4 LA=0 | qvbr | 10-13% | ❌ 失败 | 51~88 |

## 根因假说

**constqp 的高速编码 (~500 FPS) 缩小了 DMA 竞态窗口**:

```
vbr_hq/qvbr: 编码慢 (~250-350 FPS) → CE 触发 → DMA 写入 → LockBitstream
             ├── 时间窗口大 → Tier 1-B retry backoff 足够覆盖竞态 ──┤

constqp:     编码快 (~500+ FPS) → CE 触发 → DMA 写入窗口极短
             ├── Tier 1-B retry backoff 不够 → LockBitstream 返回空 ──┤
```

Tier 1-B 的固定指数退避 (500μs / 1000μs / 2000μs) 在 constqp 的高速循环中来不及生效。

## phase4 系列的终结

**phase4-slot 和 phase4-pfce 应在所有配置下废弃**:
- constqp 下: 防御失败，帧永久丢失
- vbr_hq 下: 防御可恢复但 FPS 仅为 ce-pipeline 的 50-60% (255 vs 449 FPS)
- qvbr 下: 同上

ce-pipeline 在所有 RC 模式下均零丢帧且性能碾压 phase4。

**Why:** constqp 高速编码缩小 DMA 竞态窗口，固定 backoff retry 不足覆盖
**How to apply:** 生产代码中移除 phase4-slot/phase4-pfce 路径，统一使用 ce-pipeline
