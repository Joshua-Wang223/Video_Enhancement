---
name: constqp-fast-path
description: CONSTQP 零空帧特性推导的快速路径优化（LockBitstream 重试 + Tier 1-B 跳过）
metadata: 
  node_type: memory
  type: project
  originSessionId: e47eeb99-6d8c-4065-8b2f-1ca21ec2534e
---

# CONSTQP 快速路径优化

## 测试发现

4 轮完整 3D 矩阵测试 (VBR_HQ, QVBR×2, CONSTQP×2) 在 Tesla T4 上的关键发现：

**CONSTQP 在所有测试轮次中均为 0% 空帧** — VBR_HQ/QVBR 在 LA=8+pipe=4 时产生 ~0.9% 空帧，但 CONSTQP 完全不受影响。

## 两项优化

### 1. LockBitstream 重试次数缩减

```python
# _lock_bitstream_with_retry() 中:
if self._rate_mode == 'constqp':
    max_retries = min(max_retries, 2)  # 默认5 → 2
```

### 2. Tier 1-B encode_frame 重试跳过

```python
# _NVENCEncodeThread._loop() 或 _writer_loop 中:
if getattr(self._nvenc, '_rate_mode', '') != 'constqp':
    # 仅非 CONSTQP 时执行 Tier 1-B 重试
    for _retry in range(2):
        h264_data_retry = self._nvenc.encode_frame(...)
```

## 实施位置

| 版本 | _lock_bitstream | Tier 1-B 跳过 | 触发方式 |
|------|:-:|:-:|------|
| v6.4.3 | ✅ (永久生效) | ✅ (永久生效) | CONSTQP-only, 无条件优化 |
| v6.4.3.1 | ✅ | ✅ | `if rate_mode == 'constqp'` |
| v6.4.4 | ✅ (永久生效) | ✅ (永久生效) | CONSTQP-only, 无条件优化 |
| v6.4.4.1 | ✅ | ✅ | `if rate_mode == 'constqp'` |
| v6.4.5/5.1 | ✅ | ✅ | `if rate_mode == 'constqp'` |

## v4 4D全矩阵 RC 三模式对比 (Tesla T4, 真实视频 687帧)

| RC 模式 | ce-pipeline pipe=4 LA=8 FPS | 相对性能 | 文件大小 | 丢帧 |
|---------|---------------------------:|:-------:|:-------:|:----:|
| **constqp** | **584** | **100%** | 6.2 MB | 0 |
| qvbr | 488 | 84% | 7.2 MB | 0 (pipe=4) |
| vbr_hq | 352 | 60% | 未导出 | 0 |

**constqp 比 vbr_hq 快 1.66×，文件更小！** 在所有技术路线×pipe×LA 组合上 constqp 统一最快。
QVBR 居中，VBR_HQ 在所有维度上都无优势。

## 性能收益

### 宏观 (RC 模式选择)
constqp vs vbr_hq: **+29%~66% FPS**，文件更小 (6.2MB vs 更大)，无 targetQuality 计算开销。

### 微观 (LockBitstream 优化)
~1-3% FPS，减少 CPU 侧开销。主要在 `_lock_bitstream_with_retry` 中避免不必要的 `time.sleep(backoff_us)` 调用。

## v4 新发现: phase4 在 constqp 下防御失败

constqp 的高速编码 (~500 FPS) 缩小 DMA 竞态窗口，导致 phase4-slot/pfce 的 Tier 1-B retry 不足覆盖 → 永久丢帧。
**ce-pipeline 不受影响** — deferred harvest 的时间窗口足够大。
详见 [[phase4-constqp-defense-failure]]。

## v6.4.x 六版本 benchmark 再次确认 (2026-06-16)

CRF=23 batch 测试中 CONSTQP 零空帧、零丢帧，同时文件最小 (8.7 vs 13-14 MB)、
FPS 最高 (65.7)。VBR_HQ+LA=8+pipe=4 的 v6.4.3.1 丢失 62 帧，
v6.4.5 bitrate=unconstrained 导致性能塌方 2.2× 慢。

**CONSTQP 是当前所有维度上的最优 RC 模式。**

详见 [[benchmark-ifrnet-v6.4.x-summary]] [[rc-mode-performance-ranking]]。

**Why:** CONSTQP 无 bitrate 控制复杂度，NVENC HW 编码路径简单，DMA 竞态窗口极小
**How to apply:** 在全 RC 版本中通过 `rate_mode` 条件启用；在 CONSTQP-only 版本中永久启用
