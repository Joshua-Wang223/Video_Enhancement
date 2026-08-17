---
name: level1-nvenc-encoding-flow
description: Level 1 NVENC GPU 直通编码的完整数据流、FFmpegMuxer 架构和与其他 Level 的降级关系
metadata:
  node_type: memory
  type: project
  originSessionId: 9ac38aa0-722c-43f8-aa1e-b6fad7621a9e
---

# Level 1 NVENC GPU 直通编码数据流

## 三级降级架构

```
Level 1 (最优): NVENC SDK GPU 直通 → FFmpegMuxer (GPU→NV12→H.264 ES, -c:v copy)
Level 2:        Pinned Ring Buffer + FFmpeg NVENC (h264_nvenc via ffmpeg)
Level 3:        Pinned Ring Buffer + software encoding (libx264)
Level 4:        标准 PinnedResultPool 路径（最低效）
```

## Level 1 数据流

```
GPU tensor (RGB) → _rgb_to_nv12_gpu() → NV12 GPU tensor
    → NVENCEncoder.encode_frame()
        → LockInputBuffer → cuMemcpyDtoD_v2 → UnlockInputBuffer
        → EncodePicture (async, with outputBitstream)
        → LockBitstream (blocking, no doNotWait) → 获取 H.264 ES bytes
    → result_queue 传递 (h264_frames list, batch)
    → Writer 线程: FFmpegMuxer.write(h264_bytes) → ffmpeg stdin
    → FFmpegMuxer.close() → stdin.close() → ffmpeg 写 moov atom → 完成
```

## 关键差异 vs Level 2/3/4

| 特性 | Level 1 | Level 2/3/4 |
|------|---------|------------|
| 编码位置 | GPU (NVENC SDK) | CPU (ffmpeg 子进程) |
| 码率控制 | CONSTQP (qpInterP/B/Intra) | FFmpeg -cq:v / -crf |
| 传递给 Writer | H.264 ES bytes | raw BGR frames |
| muxer 类型 | FFmpegMuxer (-c:v copy) | FFmpegWriter (重编码) |
| 首帧处理 | 需编码为 H.264 | 直接写 raw frame |
| outputBitstream 设置 | 每帧必须设 bs_handle | N/A |

## 码率控制模式

Level 1 使用 **CONSTQP** (rateControlMode=0)，直接设置量化参数：

```python
rc_ptr[1] = 0              # NV_ENC_PARAMS_RC_CONSTQP
rc_ptr[2] = qp              # constQP.qpInterP
rc_ptr[3] = qp              # constQP.qpInterB
rc_ptr[4] = qp              # constQP.qpIntra
```

**Why CONSTQP 而非 VBR:** VBR 模式（rateControlMode=1）的 averageBitRate/maxBitRate 均未生效，编码器使用默认高码率导致文件膨胀 3.3x（[[v642-v643-bug-fixes]]）。CONSTQP 直接生效，QP 值与 FFmpeg `-cq:v` 语义接近但映射不完全相同（正常现象）。

## FFmpegMuxer 注意事项

- 输入格式声明为 `-f h264`，ffmpeg 期望纯 H.264 ES
- 使用 `-c:v copy` 不做重编码
- `write()` 方法接收 `bytes`（H.264 ES），不是 numpy array
- `close()` 等待 ffmpeg 子进程最多 60s
- 如果 H.264 ES 数据格式不对（如缺 SPS/PPS），ffmpeg 会报错但不会阻塞

**Why:** 曾出现 `ValueError: ambiguous truth value of array` 错误——raw BGR numpy array 被传给 FFmpegMuxer 的 bytes 参数。根因是首帧未编码直接写入 muxer。

## NV12 格式关键约束

- `CreateInputBuffer.height` 必须是 **luma height**（H），不是 NV12 total height（H + H/2）
- 驱动用 height × pitch 计算 chroma 平面偏移，错误高度导致 chroma 偏移异常 → 灰色输出
- D2H 拷贝后必须 `event.synchronize()` 确保 pinned buffer 数据就绪，否则首帧读到零值 → 黑帧频闪

## 相关代码位置

- NVENCEncoder 类: `external/IFRNet/process_video_v6_4_3_single.py:915-1435`
- FFmpegMuxer 类: `external/IFRNet/process_video_v6_4_3_single.py:1484-1561`
- Level 选择逻辑: `external/IFRNet/process_video_v6_4_3_single.py:4889-4943`
- 首帧编码: `external/IFRNet/process_video_v6_4_3_single.py:4963-4971`
- ctypes struct 布局参考: [[nvenc-ctypes-integration]]
- Bug 修复记录: [[v642-v643-bug-fixes]]
