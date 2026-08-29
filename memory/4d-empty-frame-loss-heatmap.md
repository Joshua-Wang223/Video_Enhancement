---
name: 4d-empty-frame-loss-heatmap
description: v4 4D全矩阵空帧/丢帧热力图 — RC×Technique×Pipe×LA，区分 empty_rate / LOST / DEF，覆盖合成帧+真实视频
metadata: 
  node_type: memory
  type: project
  originSessionId: 2524a2cc-7f5a-402b-82bd-9fad0c6238b6
---

# v4 4D 空帧/丢帧热力图

## 图例

| 标记 | 含义 |
|:----:|------|
| 🟢 0% | 零空帧/零丢帧 ✅ |
| 🟡 <1% | 轻量空帧但 Tier 防御可恢复 ✅ |
| 🛡️ DEF=N | 有空帧但 Tier 1-B/1-A 100% 恢复，lost=0 |
| 🔴 E% | 空帧率 >1%，防御失败或不可用 |
| 🔥 LOST=N | 帧永久丢失，无法恢复 |
| — | 未测试 |

---

## 4D 矩阵一: CONSTQP (合成帧 N=2000, bs=24)

```
                    Pipe=4              Pipe=1
              LA=0        LA=8       LA=0      LA=8
              ──────      ──────     ──────    ──────
sync-batch      🟢 0%     🟢 0%       🟢 0%     🟢 0%
single-ce       🟢 0%     🟢 0%       🟢 0%     🟢 0%
batch-ce        🟢 0%     🟢 0%       🟢 0%     🟢 0%
phase4-slot     🔴 11.0%  🔴 10.3%    —          —
phase4-pfce     🔴 16.8%  🔴 16.8%    —          —
ce-pipeline     🟢 0%     🟢 0%       🟢 0%     🟢 0%
```

**constqp 特点**: ce-pipeline/sync/single/batch 全部零空帧；phase4 系列 11-17% 空帧且 Tier 防御失败 → LOST=220~334。

---

## 4D 矩阵二: VBR_HQ (合成帧 N=2000, bs=24) — v3 基线

```
                    Pipe=4              Pipe=1
              LA=0        LA=8       LA=0      LA=8
              ──────      ──────     ──────    ──────
sync-batch      🟢 0%     🟡 0.3%     🟢 0%     🟢 0%
single-ce       🟢 0%     🟢 0%       🟢 0%     🟢 0%
batch-ce        🟢 0%     🟡 0.3%     🟢 0%     🟢 0%
phase4-slot     🔴 12.6%  🔴 16.7%    🟢 0%     🔴 4.2%
phase4-pfce     🔴 16.8%  🔴 16.7%    🔴 4.2%   🔴 4.2%
ce-pipeline     🟢 0%     🟡 0.3%     🟢 0%     🟢 0%
```

**vbr_hq 特点**: pipe=4+LA=8 产生 0.3% 轻量空帧 (slot=1,2,3 的 LA 尾帧)，pipe=1 全零。

---

## 4D 矩阵三: VBR_HQ (真实视频 N=687, bs=24, x4seg)

```
                    Pipe=4              Pipe=1
              LA=0        LA=8       LA=0      LA=8
              ──────      ──────     ──────    ──────
sync-batch      🟢 0%     🟢 0%       🟢 0%     🟢 0%
single-ce       🟢 0%     🟢 0%       🟢 0%     🟢 0%
batch-ce        🟢 0%     🟢 0%       🟢 0%     🟢 0%
phase4-slot     🔴 12.0%  🔴 15.4%    —          —
                🔥LOST=64 🔥LOST=59
phase4-pfce     🛡️ DEF=114 🛡️ DEF=114 —         —
                (1B=114)  (1B=58,1A=56)
ce-pipeline     🟢 0%     🟢 0%       🟢 0%     🟢 0%
```

| 关键差异 vs 合成帧 | 合成帧 | 真实视频 |
|:-------------------|:-----:|:-------:|
| ce-pipeline pipe=4 LA=8 空帧 | 0.3% | **0%** |
| phase4-pfce LA=0 空帧 | 16.8% 🔴 | **🛡️DEF=114 可恢复** |
| phase4-slot LA=0 空帧 | 12.6% 🔴 | 12.0% 🔴 (仍失败) |

**根因**: 真实视频编码更慢 (vbr_hq ~350 FPS vs 合成帧 ~450 FPS)，DMA 竞态窗口更大 → Tier 1-B 有足够时间覆盖 phase4-pfce 的空帧。phase4-slot 的 slot-reuse 竞态与编码速度无关，仍然失败。

---

## 4D 矩阵四: QVBR (真实视频 N=687, bs=24, x4seg)

```
                    Pipe=4              Pipe=1
              LA=0        LA=8       LA=0      LA=8
              ──────      ──────     ──────    ──────
sync-batch      🟢 0%     🟢 0%       🟢 0%     🔥 LOST=16
single-ce       🟢 0%     🟢 0%       🟢 0%     🔥 LOST=16
batch-ce        🟢 0%     🟢 0%       🟢 0%     🔥 LOST=16
phase4-slot     🔴 10.2%  🔴 11.0%    —          —
                🔥LOST=51 🔥LOST=26
phase4-pfce     🛡️ DEF=114 🛡️ DEF=114 —         —
                (1B=114)  (1B=58,1A=56)
ce-pipeline     🟢 0%     🟢 0%       🟢 0%     🔥 LOST=16
```

**qvbr 独有**: pipe=1 + LA=8 所有技术路线一致丢 **16 帧** (LOST = 4 × 4seg)。pipe=4 不受影响。

---

## 4D 矩阵五: CONSTQP (真实视频 N=687, bs=24, x4seg)

```
                    Pipe=4              Pipe=1
              LA=0        LA=8       LA=0      LA=8
              ──────      ──────     ──────    ──────
sync-batch      🟢 0%     🟢 0%       🟢 0%     🟢 0%
single-ce       🟢 0%     🟢 0%       🟢 0%     🟢 0%
batch-ce        🟢 0%     🟢 0%       🟢 0%     🟢 0%
phase4-slot     🔴 11.3%  🔴 12.0%    —          —
                🔥LOST=59 🔥LOST=64
phase4-pfce     🔴 16.1%  🔴 16.1%    —          —
                🔥LOST=93 🔥LOST=93
ce-pipeline     🟢 0%     🟢 0%       🟢 0%     🟢 0%
```

**constqp 关键**: phase4 系列防御完全失败 (Tier 1-B 不足覆盖高速编码的竞态)；ce-pipeline 零丢帧。

---

## LOST 帧模式汇总 (真实视频, N=687)

| 模式 | 触发条件 | 丢失数 | 规律 |
|------|----------|:-----:|------|
| **phase4 constqp** | pipe=4 + phase4-slot/pfce | 59~119 | 编码过程中散布，FLUSH 无法恢复 |
| **qvbr pipe=1 LA=8** | pipe=1 + LA=8 + qvbr | 16 (x4seg) | **LOST = 4 × Nseg**，段间 encoder 状态残留 |
| vbr_hq phase4-slot | pipe=4 + phase4-slot | 59~64 | 同 constqp 模式 |
| phase4-pfce vbr_hq/qvbr | pipe=4 + pfce | 0 (全部 DEF 恢复) | 🛡️ 防御生效 |

---

## 综合安全矩阵 (所有 RC 模式取最坏情况, 真实视频)

```
                    Pipe=4              Pipe=1
              LA=0        LA=8       LA=0      LA=8
              ──────      ──────     ──────    ──────
sync-batch      🟢 OK     🟢 OK       🟢 OK     🟡 qvbr丢16
single-ce       🟢 OK     🟢 OK       🟢 OK     🟡 qvbr丢16
batch-ce        🟢 OK     🟢 OK       🟢 OK     🟡 qvbr丢16
phase4-slot     🔴❌       🔴❌         —          —
phase4-pfce     🟡 constqp❌ 🟡 constqp❌ —         —
ce-pipeline     🟢✅       🟢✅         🟢 OK     🟡 qvbr丢16
```

✅ = 所有 RC 模式零丢帧 | OK = 大多数模式零丢帧 | ❌ = 不可用

**唯一全局安全组合**: `ce-pipeline + pipe=4 + LA=0/8 + constqp`。

**Why:** v4 4D 全矩阵 (3 RC × 6 tech × 2 pipe × 2 LA × 3 seg-counts = 180+ 组合) GPU 验证
**How to apply:** 选择组合时参照此热力图避开 🔴 区域；pipe=1+qvbr+LA=8 需加段末 safety padding 缓解丢帧
