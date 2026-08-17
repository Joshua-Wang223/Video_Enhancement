---
name: v644-encodethread-cross-stream-race
description: v6.4.4 隔帧花屏根因：wait_stream 仅创建 GPU 侧 barrier，Infer 线程与 Writer 线程的 per-thread default stream 缺少 CPU 侧 synchronize 保证，修复：synchronize()
metadata: 
  node_type: memory
  type: project
  originSessionId: 59a25f45-885a-4b11-ba4c-4b62ed652b0a
---

# v6.4.4 隔帧花屏 — 真实根因与修复

## 症状

v6.4.4 单 segment 输出视频产生非确定性隔帧花屏：
- **奇数帧（img1/原始帧）损坏**，偶数帧（interp/插值帧）正常
- batch_size=24 时，每 batch 48 帧内全部 img1 帧可能批量损坏
- Bug 非确定性：同一 batch 在不同运行中可能正常或损坏
- v6.4.3 同条件输出正常

## 真实根因：`wait_stream` 不足以保护跨线程 GPU data hazard

### 关键代码（`_infer_batch` GPU-STAY 路径）

```python
torch.cuda.default_stream(self.device).wait_stream(self.stream_compute)
return ('GPU', interp_gpu, img1_rgb, B, T, orig_H, orig_W)
```

`wait_stream()` 仅创建 **GPU 侧依赖**（Infer 线程的 per-thread default stream 等待 stream_compute），**不阻塞 CPU 线程**，且**不影响 Writer 线程的 per-thread default stream**。

### 竞争链路

```
Infer线程 stream_compute:                    Writer线程 default_stream:
  写 interp_gpu, img1_rgb to VRAM            
  wait_stream(default) ← GPU侧barrier        
  return tensors ───────────────────────────→ torch.cat([interp, img1])
                                               _rgb_to_nv12_gpu_batch()
                                               ← 可能读到未完成的数据！
```

CUDA per-thread default stream 模式下，每个线程有独立的 NULL stream（stream 0 映射到各自的 per-thread stream）。Infer 线程的 stream 与 Writer 线程的 stream 之间**无任何 happens-before 关系**。

### 为何 v6.4.3 不受影响

v6.4.3 Writer 在线程内同步调用 `NVENCEncoder.encode_frame()`，每帧阻塞 ~1ms+，整个 batch 耗时 ~50ms。从 `_infer_batch` return（`wait_stream` 后）到 Writer 构建 `all_frames` 之间有 ~50ms，远大于 stream_compute 的 ~13ms GPU 执行时间。**时序隐式保护**而非逻辑保证。

这是典型的 **Heisenbug**——依赖 SLO 而非同步原语。

### 为何 v6.4.4 受影响

v6.4.4 引入 `_NVENCEncodeThread`（独立编码线程）。Writer 的 GPU_RAW 路径不再做同步编码，改为：

```python
all_nv12 = _rgb_to_nv12_gpu_batch(all_frames)   # GPU kernel launch (async!)
encode_order = [views of all_nv12]
torch.cuda.current_stream().synchronize()        # Writer的stream
_enc_thread.submit(encode_order)                 # 入队，即刻返回
```

Writer 从接收 tensor 到 `_rgb_to_nv12_gpu_batch` 仅 ~1-2ms，但 `stream_compute.synchronize()` 消失 — 之前被同步编码的 ~50ms 阻塞隐式覆盖了。竞争窗口从 <0ms 变为 ~11ms（13ms GPU 执行 - 1-2ms Writer 到达）。

### 诊断确认过程

通过 `IFRNET_DIAG_NV12=1` 环境变量添加三层诊断：
- `[DIAG-INFER]`：Infer 线程 return 前 interp/img1 的 RGB 统计
- `[DIAG-SRC]`：Writer 线程 torch.cat 前 interp/img1 的 RGB 统计  
- `[DIAG-RGB]`：all_frames 每帧 RGB 通道统计
- `[DIAG-NV12]`：NV12 转换后每帧像素统计

结论：三个诊断层**均无 NVENC API 错误、无 h264_size=0**，但 IMG1 帧的 NV12 std 是正常值的 2 倍，data_ptr 跨 batch 复用。排除编码线程、NVENC SDK、RGB→NV12 转换问题，定位到 Infer→Writer 跨线程传输环节。

## 修复

### 主修复（v6.4.4 + v6.4.3 均已应用）

在 `_infer_batch` GPU-STAY 路径 return 前添加：

```python
torch.cuda.default_stream(self.device).wait_stream(self.stream_compute)
self.stream_compute.synchronize()   # ← 关键：CPU 侧阻塞直到 GPU 写入物理完成
```

`synchronize()` 阻塞当前 CPU 线程（Infer 线程）直到 stream_compute 上所有 GPU 操作完成。之后 tensor 才通过 result_queue 传递给 Writer 线程消费。建立跨线程的 happen-before 关系。

性能影响：每 batch ~0.1ms CPU 等待，可忽略。

### 历史修复（v6.4.3 已验证有效，v6.4.4 也保留）

段末尾二次 flush（`nvenc_encoder.flush()` → `writer.write()`）解决 NVENC B-frame 重排导致的 GOP 末尾延迟帧排空问题。与本次根因正交。

## 教训

1. **`wait_stream()` 不保证跨线程正确性**：它只建 intra-device 依赖，不建 CPU 侧 barrier，也不影响其他线程。跨线程共享 GPU tensor 需要 `synchronize()` 或 CUDA event 跨线程同步。
2. **时序隐式保护不可靠**：v6.4.3 正常是因为同步编码耗时 > GPU 计算耗时，不是设计正确。异步优化打破了时序假设才暴露。
3. **Heisenbug 用诊断日志定位**：三层数据流追踪（Infer → Writer → NV12）是定位此 Bug 的关键。

## 相关代码位置

- v6.4.4 `_infer_batch` GPU-STAY return 前：`process_video_v6_4_4_single.py:5106-5107`
- v6.4.3 `_infer_batch` GPU-STAY return 前：`process_video_v6_4_3_single.py:4943-4944`
- v6.4.4 `_NVENCEncodeThread` 类：`process_video_v6_4_4_single.py:1628-1718`
- v6.4.4 Writer GPU_RAW 路径：`process_video_v6_4_4_single.py:3434-3488`
- v6.4.3 Writer GPU_RAW 路径（内联编码）：`process_video_v6_4_3_single.py:3255-3287`
- v6.4.3/v6.4.4 段末尾二次 flush：`process_video_v6_4_3_single.py:5486-5489` / `process_video_v6_4_4_single.py:5650-5653`

## 相关记忆

- [[level1-nvenc-encoding-flow]] — Level 1 数据流架构
- [[project_v642_v643_bugs]] — 历史 Bug 修复记录（二次 flush 的来源）
