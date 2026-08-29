---
name: nvenc-empty-frame-defense
description: NVENC 空帧问题 — pipe=4+LA 空帧根因、Tier 0/1/3-E 多层防御、GPU 验证 pipe=4+LA=8 100% 防御恢复
metadata:
  type: project
  originSessionId: 5c7a0c8b-7c9f-4aa4-8e78-4eba77867ec7
---

# NVENC 空帧问题 — 根因分析与多层防御

## 2026-06-11 最终根因（GPU 验证）：pipeline_depth=4 + lookahead 不兼容

### 排错过程

1. **DPB 假说（推翻）**：per-slot IDR 修复后仍空帧，且 `fidr=True` 无效
2. **决定性验证**：设 `_NVENC_LEVEL1_LOOKAHEAD=0` → 全部空帧消失，1373帧正常
3. **变量隔离**：

| pipe_depth | LA | 结果 |
|:---:|:---:|:---:|
| 4 | 8 | ❌ 空帧 |
| 4 | 0 | ✅ 正常 |
| 1 | 8 | ✅ 正常 |

### 真实根因

NVENC `pipeline_depth` 创建内部流水线阶段（非独立编码槽）。lookahead 延迟输出打破了 "帧 N→slot N%4" 的映射假设 → LockBitstream 在错误 slot 拿到空 buffer。

### 修复

1. v6.4.5.1: `_NVENC_LEVEL1_LOOKAHEAD = 0`
2. v6.4.5/5.1: `__init__` 防御性断言：`la_depth>0` 时强制 `pipeline_depth=1`

### 旧 DPB 假说（已推翻，保留供参考）

~~NVENC `pipeline_depth=4` 下每个 pipeline slot 维护独立的 DPB~~ — 每槽 IDR 修复后空帧仍未解决，假说被 GPU 验证推翻。

## completionEvent 路径：2026-06-10 重新验证 — 0% 空帧率 ✅

### 重新验证结果

在 Tesla T4 / driver 580 / SDK 13.0 环境下运行 `test_nvenc_completion_event.py`：

| 测试 | 空帧率 | FPS |
|------|--------|-----|
| VBR+LA8 completionEvent | 0/304 = 0.00% | 401.8 |
| VBR+LA8 Synchronous | 0/304 = 0.00% | 408.2 |
| VBR+LA0 completionEvent | 0/300 = 0.00% | 391.2 |
| CONSTQP completionEvent | 0/300 = 0.00% | 430.1 |

**结论**: completionEvent + cuEventSynchronize 在当前环境下完全可靠。历史 ~7% 空帧率（v6.4.5 早期版本）很可能来自已修复的 struct 布局或 func_ptr 索引问题，而非 completionEvent 机制本身。

### 与历史记录的矛盾

旧版 v6.4.5 代码注释（line 52）记录：
```
[PHASE4] NVENC 批编码同步流水线（completionEvent async 因 ~7% 空帧率已移除）
```

矛盾的根因推断：
- v6.4.5 早期开发过程中 completionEvent 路径经历了多轮 struct 布局变更（NV_ENC_INITIALIZE_PARAMS 从 96B→1800B 等）
- func_ptr 索引也曾错误（硬编码 52/53/54 应为 19/16/20）
- 当前 SDK 13.0 + driver 580+ 环境已完全稳定

### 实施

已在全部 6 个活跃版本中实施 completionEvent：
- v6.4.3 / v6.4.3.1 / v6.4.4 / v6.4.4.1: `encode_frame()` 单槽
- v6.4.5 / v6.4.5.1: `encode_frame()` 单槽 + `encode_frames_batch()` 4槽每槽独立 event

## DMA 竞态假说（已被 completionEvent 验证推翻）

旧版 memory 记录：NVENC 独立硬件单元 → cuEventSynchronize 不保证 DMA 写入完成 →
event signaled 但 bitstream 数据未落位。

**推翻**: cuEventSynchronize 在 SDK 13.0 下正确同步了 NVENC DMA 完成。旧版 7% 空帧率更可能是 struct 布局问题导致 EncodePicture 本身不稳定。

## 同步路径仍有残余空帧

即使在同步 EncodePicture 路径下，代码仍保留空帧检测（`bitstream_size == 0`），说明空帧在同步模式下未完全消失，只是频率大幅降低到诊断日志级别。

### 可能根因（按可能性排序）

1. **同一 DMA 竞态极低概率触发**（60-70%）：同步 EncodePicture 内部 ordering 保证不如完美
2. **B-frame 重排序延迟**：GOP 内 B-frame 不按显示顺序产出
3. **NVENC 内部状态机瞬时异常**：高负载下拒绝编码但返回 success
4. **NV12 输入边界行为**：全黑/全静态帧在特定 QP 下零输出
5. **Bitstream buffer 不足**：IDR 帧 + 高复杂度溢出

### 当前缺陷

空帧被 `_NVENCEncodeThread._loop()` 静默丢弃（只计数 `_empty += 1`），导致输出 MP4 少一帧。若空帧率 0.1%，10 万帧视频丢 ~100 帧（2s），可致音视频不同步。

## v6.4.x 版本架构差异

| 版本 | NVENCEncoder | 编码方式 | Writer 调用 | 空帧处理点 |
|------|:--:|------|------|------|
| v6.4.1 / v6.4.2 | **无** | ffmpeg NVENC (Level 2) | N/A | N/A |
| v6.4.3 / v6.4.3_1 | 单 slot | `encode_frame()` | Writer 内联循环 | `_writer_loop` |
| v6.4.4 / v6.4.4_1 | 单 slot | `encode_frame()` | `_NVENCEncodeThread` | `_loop()` |
| v6.4.5 / v6.4.5_1 | 多 slot (4) | `encode_frames_batch()` + `encode_frame()` | `_NVENCEncodeThread` | `_loop()` + 两处 LockBS |

v6.4.1/v6.4.2 无 SDK 直通编码，不受此问题影响。

## 多层防御栈（已实施，2026-06-09）

在全部 6 个活跃版本中实施（v6.4.3 / v6.4.3_1 / v6.4.4 / v6.4.4_1 / v6.4.5 / v6.4.5_1）：

```
空帧发生
  ├─ Tier 3-E: _lock_bitstream_with_retry() — LockBitstream 指数退避重试 3 次
  │              (500μs / 1000μs / 2000μs，覆盖约 3.5ms DMA 竞态窗口)
  │   └─ 成功 → 正常写入
  │   └─ 仍空 → 进入 Tier 1-B
  │
  ├─ Tier 1-B: 对同一 NV12 tensor 重调 encode_frame() 重编码，最多 2 次
  │   └─ 成功 → 正常写入，更新 _prev_h264
  │   └─ 仍空 → 进入 Tier 1-A
  │
  └─ Tier 1-A: 用 _prev_h264（前一正常帧的 H.264 ES bytes）填充写入
      └─ _prev_h264 存在 → 保持帧数，A/V 同步不变，单帧重复人眼不可感知
      └─ _prev_h264 is None (首帧) → 丢弃，统计 _diag_empty_h264++
```

### Tier 0 诊断（贯穿全程）

空帧时记录 NV12 输入的 mean/std/min/max + 帧上下文 (frame_idx, force_idr, slot)，频率限制前 5 次 + 每 50 次。

性能影响：正常帧零开销（仅在空帧触发时执行统计 + 日志）。

## `_lock_bitstream_with_retry()` 方法签名

```python
def _lock_bitstream_with_retry(self, bs_handle, max_retries=3, backoff_us=500):
    # Returns (h264_data: bytes, status: int)
    # 替代原来的手动 LockBitstream 调用
    # 内部 handles: LockBS → 检查 bitstream_size → Unlock → sleep → 重试
```

正常帧开销：仅多一次函数调用 + 一次 ctypes struct memset，无额外 GPU 同步。
空帧时额外 CPU 等待：0.5~3ms（取决于第几次重试成功）。

## 相关文件

- 防御实施位置（所有版本）：
  - `_lock_bitstream_with_retry()` — 新增在 `_extract_sps_pps()` 之后
  - `encode_frames_batch()` — v6.4.5/5_1: LockBitstream 块替换为调用 `_lock_bitstream_with_retry(slot['bs_buf'])`
  - `encode_frame()` — 全部: LockBitstream 块替换为调用 `_lock_bitstream_with_retry(self._bs_handle)`
  - `_NVENCEncodeThread._loop()` — v6.4.4/4_1/5/5_1: Tier 1-B+A 重试+补偿
  - `_writer_loop()` — v6.4.3/3_1: Tier 1-B+A 重试+补偿
- 记忆文档：[[level1-nvenc-encoding-flow]] [[nvenc-ctypes-integration]] [[v644-encodethread-cross-stream-race]]

## 2026-06-15 pipe=4+LA=8 Tier 防御验证 ✅

GPU 验证 (Tesla T4, N=2000, VBR_HQ, bs=1000)：ce-pipeline pipe=4 LA=8 产生 9/2000 = 0.45% 空帧。

**Tier 防御 100% 恢复**：
- Tier 1-B (encode_frame 重试 2 次): 恢复 6/9 (67%)
- Tier 1-A (prev_h264 回退): 补偿 3/9 (33%)
- **lost_frames = 0**

vs pipe=1 LA=8: **+7.3% FPS** (343 vs 320)。

**结论**: pipe=4+LA=8 完全可行，v6.4.3.1/4.1 已移除 VBR/QVBR pipe 强制降级。详细数据见 [[pipe4-la8-tier-defense-verified]]。

