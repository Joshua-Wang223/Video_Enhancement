---
name: esrgan-empty-final-chunk-skips-eos
description: encode_frames_batch() 空批次快速路径无视 send_eos 导致 LA 帧滞留在硬件
metadata: 
  node_type: memory
  type: project
  status: fixed
  originSessionId: 279c3f6f-f525-402f-b6ed-7f6152663198
---

# ESRGAN 空 Final Chunk 跳过 EOS

## 症状

段 0 少 8 帧，段 1 多出 8 帧（其中仅 2 帧可解码），整个视频总计少 6 帧。日志证据：

| 段 | NVENC-FINAL-CHUNK n= | submitted | written | 差值 |
|---|---------------------|-----------|---------|------|
| 0 | **0** (EOS 被跳过) | 7223 | 7215 | -8 |
| 1 | 145 | 7201 | **7209** | +8 |
| 2-5 | 非零 | N | N | 0 |

仅段 0 最终块 `n=0`，其余各段都有非零剩余帧 → EOS 正常触发。

## 根因

**文件**: `nvenc_sdk.py` `encode_frames_batch()` 第 1303-1305 行

```python
n_frames = len(nv12_tensors)
if n_frames == 0:
    return []          # ← 无条件返回，send_eos 参数被完全无视
```

chunked-LA 模式下，`_NVENCEncodeThread._loop()` 最终块调用：
```python
h264_list = self._nvenc.encode_frames_batch([], False, send_eos=True)
```

当段总帧数恰好被 chunk size (168) 整除时，最终块 nv12_tensors=[] → `if n_frames == 0` 命中 → EOS picture 连发送都没有 → per-slot 阻塞 drain 整段跳过 → LA=8 重排序队列里的 ~8 帧滞留硬件，被下一段自然 drain 出来，但误判为"上一个 chunk"写进了下一段的 muxer。

这些帧是 P/B 帧，运动补偿参考的是段 0 GOP 的解码图像缓冲区 — 段 1 有自己的独立 IDR，缓冲区状态不存在，所以多数无法解码 (6 帧丢弃)，仅少数 (2 帧) 碰巧可解。

## 修复

**文件**: `nvenc_sdk.py`

```python
# [FIX-EOS-EMPTY-CHUNK]
if n_frames == 0 and not send_eos:
    return []
```

仅改一处。当最终块无剩余原始帧但 `send_eos=True` 时，继续往下走 — 真正发送 EOS + 执行 per-slot 阻塞 drain，在**当前段自己的**写入流程里回收所有滞留帧。

## 触发条件

这是"运气触发"的 bug — 仅当某段总帧数恰好被 chunk size 整除时触发。此 case 里 chunk_size=150（累积 7 batch=168），触发概率较低。

## 关联

与 [[esrgan-nvenc-slot-backpressure]] 配合修复段间帧泄漏。两者独立但互补：slot 背压防止提交侧覆盖，EOS 修复防止排空侧滞留。
