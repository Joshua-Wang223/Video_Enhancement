---
name: pipeline-depth-slot-rotation-confusion
description: pipeline_depth 与 multi-slot 轮转的三重概念混淆分析，以及缺失的防御代码
metadata:
  type: project
  originSessionId: 5a4d3e05-fe13-4a65-8198-31f96f85c71e
related: [[nvenc-la-frame-conservation-fix]] [[nvenc-empty-frame-defense]]
---

# pipeline_depth 与 multi-slot 轮转的概念混淆

## 三重概念混淆

`pipeline_depth` 变量同时承担三个角色：

| 概念 | 含义 | 当前实现 |
|------|------|----------|
| 硬件流水线深度 | NVENC 芯片内部可同时处理的帧数 | 控制 buffer 对数量 (`self._slots`) |
| 软件轮转模数 | 帧到物理 buffer 的映射 | `slot_idx = fi % pipeline_depth` |
| 轮转周期 | slot 被 harvest 的间隔 | 每 `pipeline_depth` 帧 Phase 1 harvest |

在传统设计中这三个概念应当分离（轮转 slot 数 ≥ 硬件深度），但当前代码强制绑定为同一个值。

## 证据

1. **`NVENCEncoderMode6` 测试类明确承认此冲突**（`tests/test_nvenc_completion_event_v4.py` 行 2704-2713）：
   ```python
   """Solves the pipe=4 + LA>0 slot-rotation conflict by:
   1. Expanding slots: total_slots = pipeline_depth + la_depth
   2. Free-pool allocation: deque of free slot indices instead of fi % pd round-robin"""
   ```
   证明作者知道 `fi % pd` 轮转存在问题。

2. **auto-calibration 误用 SDK 约束**：`_min_pd = LA+1` 是 buffer 数量的下限，被直接当作轮转模数。

## 缺失的防御代码（高风险）

记忆文件 `nvenc-empty-frame-defense.md` 记录的硬件约束：

> `lookahead_depth > 0 AND pipeline_depth > 1 → 空帧风险 (~1.89%)`

根因：NVENC 驱动在 LA 激活时对 bitstream 输出 buffer 内部重路由，`EncodePicture(bs_buf=buf[slot])` 指定的输出 buffer 在 NEED_MORE_INPUT 阶段不保证是实际目标。

推荐的防御代码：
```python
if la_depth > 0 and pipeline_depth > 1:
    pipeline_depth = 1
```

**该防御代码在 v6.4.5.1、v6.4.3.1、v6.4.4.1 中均未实现。**
当前默认配置（VBR_HQ + LA=8）经 auto-calibration 后 `pipeline_depth = max(2, 9) = 9`，直接违反约束。

## 其他影响

- **inline drain 索引脆弱**：`_drain_slot = _est_fi % pd`（行 2137），注释自承认可能导致花屏
- **pd 4→2 变化**：v6.4.5.1 将 `_NVENC_LEVEL1_PIPELINE_DEPTH` 从 4 降为 2，在 CONSTQP (crf=0) 模式下实际生效，harvest 窗口仅 2 帧
- **sync vs CE 路径 IDR 策略不一致**：sync 路径 per-slot IDR，CE 路径单 IDR

## 长期方向

分离 `pipeline_depth`（硬件深度）和 `slot_count`（轮转 slot 数），或采用 `NVENCEncoderMode6` 的 free-pool allocation 方案。

**Why:** `pipeline_depth` 多重语义导致 slot 轮转设计与 NVENC API 约束冲突，缺少防御代码可能导致 LA>0 时输出视频花屏/乱序。
**How to apply:** 短期：实现防御 `if la_depth > 0 and pipeline_depth > 1: pipeline_depth = 1`；长期：分离 pipeline_depth 和 slot_count 为独立变量。参见 [[nvenc-la-frame-conservation-fix]] 中通过全局 drain 绕过此约束的修复方案。
