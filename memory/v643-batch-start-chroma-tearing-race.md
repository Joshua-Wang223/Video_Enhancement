# v6.4.3 周期性布纹花屏/帧闪烁/文件体积 2x 根因：批首 NV12 跨流撕裂竞态

**状态**: 根因已确认（2026-08-17 取证），v6.4.3 已实施 FIX-ENC-SYNC（2026-08-17，Fix 1 + Fix 2）。

## 症状

真实视频 benchmark（word_world_2, 720×576@25fps, x2, batch_size=24, CONSTQP QP=23, pipe=4）：

- v6.4.3 输出周期性"布纹状"花屏 + 帧闪烁 + 画面不稳定；v6.4.4/5 完全干净。
- v6.4.3 文件 36.2MB / 码率 11.06 Mbps，v6.4.4/5 仅 18.4MB / 5.64 Mbps（同为 CONSTQP QP=23）。
- 所有码流结构检查全部通过：frames=packets=1373、无连 IDR、frame_num 单调、无 pts_anomaly、无解码错误 → 旧版 verify 脚本判 v6.4.3 PASS（漏检）。

## 取证证据（temp/_forensic_bitrate.py）

| 指标 | v6.4.3 | v6.4.4/5 |
|------|--------|----------|
| P-slice 平均字节 | 6676（max 224594） | 3340（max 22516） |
| 色度 U 平面帧间 MAD（闪烁） | mean 4.8, max 169.4 | mean 1.35, max 15.7 |
| U std 坏帧（>1.6×median） | 36 帧，std 73-90（基线 36） | 0 帧（max 42.9） |
| 坏帧分布 | 簇集于 batch 起点，~96 帧周期（隔 batch） | — |

坏帧索引: [1, 49-52, 145-148, 241-244, 337-340, 435-436, 530-532, 625-628, 817-820, 1249-1252, 1347-1348]。
花屏帧为 NV12 撕裂噪声 → CONSTQP 固定 QP 下噪声极费比特 → P-slice 2x → 文件 2x。IDR 均值同样 1.5x（17892 vs 11753）。

## 根因链

1. 2026-07-24 backport FIX-ASYNC-COPY 后，`NVENCEncoder._copy_into_input_buffer()` 的 D2D 拷贝
   走 `cuMemcpy2DAsync_v2` + 专用 **non-blocking** `_stream_encode`，随后 `cuStreamSynchronize(_stream_encode)`
   只等拷贝完成，**不再等待 PyTorch stream**（backport 前用同步 `cuMemcpy2D_v2`，legacy null stream
   语义 = 上下文全局 barrier，隐式安全）。
2. v6.4.3 `_writer_loop()` GPU_RAW 分支（~line 3819-3830）**内联**调用
   `encode_frames_batch_ce_pipeline()`，且 `_rgb_to_nv12_gpu_batch(all_frames)`（PyTorch kernel，
   writer 当前 stream）与 ce-pipeline 内部的首个 `cuMemcpy2DAsync_v2` 之间**没有任何 synchronize**。
3. 拷贝与 RGB→NV12 kernel 竞态：GPU 繁忙时拷贝先执行，读到 kernel 未写完/未开始的 NV12 数据
   （撕裂 = 行级新旧数据混合 = "布纹"）→ 编码进 bitstream。拷贝落后 kernel 越多的批越易中招，
   队列振荡形成 ~96 帧周期。
4. `torch.cat`/`_rgb_to_nv12_gpu_batch` 均落在 writer 的 legacy default stream（synchronizing），
   会自动等 Infer 线程 stream_compute 的推理 kernel —— 该环节天然有序，无需额外修复。
   唯一竞态点就是 2→3。

## 修复方案（对齐 v6.4.4/5）

v6.4.4/5 的对应代码（process_video_v6_4_4_single.py line 4090-4093）:

```python
# [FIX-ENC-THREAD] CRITICAL: 等待当前 PyTorch stream 完成 NV12 写入，
# 再交给编码线程的 cuMemcpy2D 读取，防止 GPU 数据未就绪导致静默花帧。
torch.cuda.current_stream().synchronize()
_enc_thread.submit(encode_order, force_idr_first=_is_first_submit)
```

v6.4.3 最小修复（不引入 _NVENCEncodeThread，遵守"只修 bug 不改架构"）：
在 `_writer_loop()` GPU_RAW 分支 `h264_list = _nvenc.encode_frames_batch_ce_pipeline(nv12_list)`
之前插入同样的 `torch.cuda.current_stream().synchronize()`。每 batch 一次、开销可忽略
（writer 本来就要阻塞等 `cuStreamSynchronize(_stream_encode)`）。

预期修复后：v6.4.3 无花屏、码率回落 ~5.6 Mbps、文件 ~18MB。
可选加固：D2H fallback 分支（line 4344/4350 `encode_frame` 前）与段首首帧（line 6074）
同属 `_copy_into_input_buffer` 调用方，理论上同类竞态，但路径罕见，可后续补 synchronize。

## 修复实施记录（2026-08-17，FIX-ENC-SYNC）

已修改 `external/IFRNet/process_video_v6_4_3_single.py`（+20 行，版本号不变）：

| # | 位置（修复后行号） | 内容 |
|---|-------------------|------|
| 文件头 | Backport 2026-08-17 块 | FIX-ENC-SYNC 说明（竞态机制 + 修复方式） |
| Fix 1 | line 3838-3841，ce-pipeline 批量入口 | `torch.cuda.current_stream().synchronize()`（对齐 v6.4.4/5 line 4090-4093 写法） |
| Fix 2a | line 4356-4358 / 4365-4366，D2H fallback 每帧 | interp 帧与 img1 帧 encode_frame 前各一次同步 |
| Fix 2b | line 6088-6090，段首首帧 | encode_frame(first_nv12) 前同步 |

已通过 `python -m py_compile` 与 IDE 静态检查（无错误）。
本机（Windows）无 torch/GPU 环境且 WSL 不可用（Hyper-V 未启用），GPU 验证需在
生产 Linux 服务器执行：`python tests/benchmark_ifrnet_versions.py -i temp/fix_input_60s.mp4
-o temp/benchmark_output_fix --versions v6.4.3 --keep-outputs`，随后对输出跑
`tests/verify_segment_bitstream_v3.py`（预期：色度坏帧簇=0、PASS）。

## 验收脚本配套修复（tests/verify_segment_bitstream_v3.py v6）

- 检查 2「段首连 IDR」阈值 `> 0` → `>= 3`：v6.4.4/5 段首双 IDR 是 FIX-SPS-PPS-V2 冗余重注入的
  良性恢复点（首个 IDR → 冗余 SPS/PPS/AUD → 恢复点 IDR）；真 per-slot IDR 异常为 6-16+ 连 IDR。
- 新增检查 4 `check_chroma_corruption()`：yuv420p 流式解码逐帧 U/V std，自校准阈值
  `max(median*1.6, 12.0)`，坏帧分簇（间距 <8 帧同簇）后 ≥3 簇判 FAIL。
- 验证结果：v6.4.4 PASS、v6.4.5 PASS、v6.4.3 FAIL（色度坏帧 12 簇 @ [1,49,145,241,337,435,529,625,817,1249,1297,1347]）。

## 相关记忆

- [[v644-encodethread-cross-stream-race]] — v6.4.4 早期同类竞态（per-thread stream 缺 synchronize）
- [[_rgb_to_nv12_gpu-bgr-optimization]] — RGB→NV12 批量 kernel
- [[v6.4.x-backport-fixes]] — FIX-ASYNC-COPY backport 背景
- [[esrgan-pinned-buffer-pool-race]] — ESRGAN 侧同类"读早于 GPU 完成"竞态
