---
name: ifrnet-v6-4x-version-matrix
description: IFRNet v6.4.x 全版本架构对比、演化路径和最终收敛状态
metadata: 
  node_type: memory
  type: project
  originSessionId: e47eeb99-6d8c-4065-8b2f-1ca21ec2534e
---

# IFRNet v6.4.x 全版本架构矩阵

> ⚠️ 2026-08-13 修正（以代码实读为准）：v6.4.5 与 v6.4.5.1 的角色与原记录相反。
> - **v6.4.5.1（当前活跃）＝全 RC 流式版**：默认 `_NVENC_LEVEL1_RATE_MODE="vbr_hq"` + `_NVENC_LEVEL1_LOOKAHEAD=8`，独有 `encode_frames_stream()` / `_stream_begin()` / per-slot FIFO drain。
> - **v6.4.5 ＝ CONSTQP-only batch 版**：`self._rate_mode='constqp'` 硬编码（`_la_depth` 恒 0），无 `encode_frames_stream`，LA rotation 为死代码。
> **不要用 v6.4.5 验证 VBR_HQ/QVBR/LA 相关行为**，相关验证一律以 v6.4.5.1 为准。

## 版本演化路径

```
v6.4.3    (6659行) — NVENC SDK 起点, CONSTQP-only, _writer_loop 内编码
v6.4.3.1  (7874行) — +VBR_HQ/QVBR/LA/AQ, 仍在 _writer_loop
v6.4.4    (6925行) — CONSTQP-only + _NVENCEncodeThread 独立编码线程 + FIX-ENC-CTX
v6.4.4.1  (8169行) — 全RC + _NVENCEncodeThread, 功能最全
v6.4.5    (7074行) — CONSTQP-only batch 版: ce-pipeline + encode_frame, 无 encode_frames_stream, _la_depth 恒 0
v6.4.5.1  (8655行) — 全RC 流式版(当前活跃): encode_frames_stream + _stream_begin + per-slot FIFO drain + FIX-ASYNC-COPY / FIX-FLUSH-GRANULARITY
```

## 三个架构层

| 层 | 版本 | 编码架构 |
|----|------|---------|
| 1 | v6.4.3, v6.4.3.1 | 编码在 `_writer_loop` 内, 无独立线程 |
| 2 | v6.4.4, v6.4.4.1 | `_NVENCEncodeThread` 独立编码线程, NV12 kernel ↔ 编码重叠 |
| 3 | v6.4.5 (batch/CONSTQP-only), v6.4.5.1 (stream/全RC) | 同层2 + per-frame CE + SPS/PPS 注入 + GPU 验证文档；两版本差异见"仅存不可消除差异" |

## 最终收敛状态 (所有版本均具备)

- ✅ 多 slot 环形缓冲 (pipeline_depth=4)
- ✅ `encode_frames_batch_ce_pipeline()` per-frame CE 异步编码
- ✅ `encode_frames_batch()` 同步 fallback
- ✅ 多 slot flush 排空 + `_flush_frame_count`
- ✅ RC 自适应: CONSTQP+LA 保留 pipe=4; VBR/QVBR+LA pipe=4 + Tier 防御 (2026-06-15 更新)
- ✅ CONSTQP LockBitstream `max_retries=2`
- ✅ CONSTQP Tier 1-B `encode_frame()` 重试跳过
- ⚠️ 以上 RC/LA 相关收敛描述以 v6.4.5.1 为准；v6.4.5 为 CONSTQP-only, LA 恒 0

## 2026-06-15 更新: pipe=4+LA=8 决策

**旧行为**：VBR/QVBR + LA>0 → 强制 pipe=1（避免 slot-rotation 空帧）
**新行为**：保留 pipe=4，依赖 Tier 1-B/A 防御处理 ~0.5% 残余空帧
**GPU 验证**：9/2000 frames empty → Tier 1-B 恢复 6 + Tier 1-A 补偿 3 → lost=0
**收益** vs pipe=1: +7.3% FPS (343 vs 320)

`_NVENC_LEVEL1_LOOKAHEAD`: 0→8 (v6.4.3.1/4.1/5.1)

## 仅存不可消除差异

1. v6.4.3/3.1 无 `_NVENCEncodeThread` (架构级, 有意)
2. v6.4.3/4/**5** CONSTQP-only (功能子集, 有意)
3. **v6.4.5 vs v6.4.5.1**: batch vs stream 编码路径。v6.4.5 无 `encode_frames_stream`/`_stream_begin`/FIFO drain, 不支持 VBR_HQ/QVBR/LA；v6.4.5.1 为唯一支持全 RC + LA 的 6.4.5.x 版本
4. RC params 注释标签 (V6431/V6441/V6451) 不同但可执行代码完全相同

**Why:** 多轮优化后所有版本收敛于相同的核心编码架构，差异仅在功能子集选择
**How to apply:** 根据需求选择: CONSTQP极致→v6.4.3/v6.4.5, 全RC轻量→v6.4.3.1, 全功能→v6.4.4.1, **GPU验证RC/LA→v6.4.5.1（唯一支持 VBR_HQ/QVBR/LA 的 6.4.5.x 版本）**
