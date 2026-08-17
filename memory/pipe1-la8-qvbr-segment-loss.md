---
name: pipe1-la8-qvbr-segment-loss
description: pipe=1+LA=8+qvbr 固定丢帧模式 — 每个 segment 丢 4 帧，LA 尾部滞留帧 LockBitstream 无法排出
metadata: 
  node_type: memory
  type: project
  originSessionId: 2524a2cc-7f5a-402b-82bd-9fad0c6238b6
---

# pipe=1 + LA=8 + qvbr 固定丢帧模式

## 规律

v4 4D 全矩阵测试在真实视频多段编码下发现新丢帧模式：

| 段数 | LA=8 + pipe=1 + qvbr | LOST | 公式 |
|:----:|-----------------------|:----:|------|
| 4seg | 所有技术路线一致 | 16 | 4×4 |
| 3seg | 所有技术路线一致 | 12 | 4×3 |
| 2seg | 所有技术路线一致 | 8 | 4×2 |

**LOST = 4 × num_segments**，固定规律，非随机。

## 影响范围

- **仅 qvbr 受影响** — constqp 和 vbr_hq 在此配置下丢帧 = 0
- **仅 pipe=1 受影响** — pipe=4 没有此问题
- **仅 LA=8 受影响** — LA=0 没有此问题
- **所有技术路线一致** — sync-batch、batch-ce、single-ce、ce-pipeline 全部复现

## 根因假说

每个 segment 结束时，LA=8 缓冲区有 4 帧被 NVENC 内部"吸收"但从未通过 LockBitstream 暴露：

```
Segment 编码流程 (pipe=1):
  帧 1-8 → LA 缓冲 → NEED_MORE_INPUT
  帧 9-N → 正常编码 → LockBitstream 输出
  帧 N-3 ~ N → 编码完成但 LA 内部持有
  EOS → flush → 成功排空 4 帧
  段间 reset → 新 segment 初始化
  ... 但 qvbr 的段间 encoder 状态残留导致 4 帧未输出
```

qvbr 的段间 encoder 状态残留（可能涉及 VBV 缓冲状态）与 constqp/vbr_hq 不同。

## 缓解方案

1. **生产使用 ce-pipeline pipe=4** — 不受此问题影响
2. 若必须 pipe=1 + qvbr: 每个 segment 末尾额外喂入 4 帧 dummy 数据垫底
3. 或切换为 constqp（完全不丢帧）

**Why:** qvbr 段间 encoder 状态残留导致每 segment 固定丢 4 帧
**How to apply:** 优先用 pipe=4 规避；pipe=1 场景切换为 constqp 或在 segment 末尾加 4 帧 safety padding
