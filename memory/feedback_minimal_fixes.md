---
name: minimal-targeted-fixes
description: 用户要求只修 bug 不改架构 — 保留现有架构（PinnedRingBuffer、NVENC 4级降级），只做针对性最小修复
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 442d2544-5553-4ef0-8e6e-6b0eddbdd4d1
---

只修 bug，不改变架构。

**Why:** 用户明确拒绝了用 v6.4.1 代码整体替换 v6.4.2/v6.4.3 的做法（"恢复本次修复，我要的不是这样的"）。架构选择（PinnedRingBuffer、NVENC 4级降级体系）是有意为之的设计决策，不是 bug。

**How to apply:** 修复 v6.4.2/v6.4.3 的问题时：
- 保留 PinnedRingBuffer、NVENCEncoder、FFmpegMuxer 等现有架构组件
- 只修改具体的错误代码（如错误的 channel flip、缺失的参数配置）
- 不做架构级简化或回退
- 修复后 v6.4.2 应仍是 PinnedRingBuffer 版本，v6.4.3 应仍是 NVENC 4级降级版本
