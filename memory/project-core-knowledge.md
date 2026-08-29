---
name: project-core-knowledge
description: 项目综合认知入口（opencode/Claude 共享），含架构、活跃文件、NVENC 核心规则、关键坑、工作偏好
type: project
---

# Video Enhancement 项目综合认知（opencode/Claude 共享）

本文件为项目认知的浓缩入口，由 AGENTS.md / CLAUDE.md / MEMORY.md 及各专题记忆文件汇总而成，供任意 AI 助手快速建立上下文。

## 1. 项目定位

GPU 加速视频处理流水线 = **IFRNet 插帧**（2x–16x）+ **Real-ESRGAN 超分**（2x/4x），NVIDIA GPU + 可选 TensorRT/FP16/CUDA Graph/torch.compile。

- 开发环境 Windows（编码/调试），生产部署 Linux（真实视频处理），**所有代码必须跨平台兼容**
- 入口命令：`python src/main_video_optimized.py -c config/default_config.json -i input.mp4 -o output.mp4`
- 双模式：`interpolate_then_upscale`（默认）/ `upscale_then_interpolate`；`--skip-interpolate` / `--skip-upscale`

## 2. 活跃文件（其余全部历史/参考）

| 用途 | 当前文件 |
|------|---------|
| 主入口 | `src/main_video_optimized.py`（唯一入口） |
| IFRNet processor | `src/processors/ifrnet_processor_video_optimized.py` |
| IFRNet backend | `external/ifrnet_video/main.py`（包：pipeline / nvenc_sdk / tensorrt_accel / ffmpeg_io / config / ifrnet_utils） |
| ESRGAN processor | `src/processors/realesrgan_processor_video_optimized.py` |
| ESRGAN backend | `external/realesrgan_video/main.py` + `pipeline.py` + `nvenc_sdk.py` |
| Config | `src/utils/config_manager.py` + `config/default_config.json`（JSON 是唯一权威） |
| Video utils | `src/utils/video_utils.py` |

数据流（interpolate_then_upscale）：抽音频 → 分段 → IFRNet 逐段插帧 → 段列表直接传 ESRGAN 逐段超分（跳过中间 merge）→ merge → mux 音频。直接段传递省 ~30% I/O。

## 3. 核心架构

**IFRNet v6.4.5.1（2026-08-17 模块化）**：三线程 T1 Reader（NVDEC 解码）/ T2 GPU（IFRNet 推理）/ T3 Writer（编码）。双 CUDA transfer stream + CudaEventPool，GPU 监控线程 2s 采样，段后自适应调队列。TRT engine 缓存 `.trt_cache/`（文件名编码 model/bs/resolution/FP16/SM arch）。原 8756 行单文件已逐字拆分至 `external/ifrnet_video/`（镜像 realesrgan_video 模块划分），TRT 逻辑抽取为 `tensorrt_accel.TensorRTAccelMixin`；细节见 [[ifrnet-v6451-modular-split]]。

**ESRGAN（2026-07-29 跨段优化）**：GFPGAN 子进程跨段保活 + 删 `reopen()` + `_stream_begin(force=True)` + encoder 参数 key 一致性检查 + 移除 FIX-SEG-START-SYNC。段切换从几十秒降至毫秒级，仅重建 `_NVENCEncodeThread` + `FFmpegMuxer`。

## 4. NVENC 编码子系统（核心难点）

### 四级降级架构
Level 1 NVENC SDK GPU 直通（ctypes + byte array 手动 offset）→ Level 2 Pinned Ring Buffer + h264_nvenc → Level 3 libx264 → Level 4 PinnedResultPool。

### 铁律（每条都踩过坑）
1. **ctypes struct 全部用 byte array + 手动 offset 写**，禁用 `ctypes.Structure`（field offset 不可靠且易缺字段）；offset 必须从 SDK 头文件逐字节验证
2. `CreateInputBuffer.height` 对 NV12 必须用 **luma height（H）**，非 total height（H+H/2）→ 否则灰色输出
3. `outputBitstream` 必须**每帧设置**（含 EOS 帧）→ 否则 LockBitstream 无输出
4. `_lock`（threading.Lock 不可重入）持锁时**严禁调用 `flush()`** → 死锁
5. GUID 必须从 driver 动态查询，不能硬编码；RegisterResource 在 T4/driver 580 上 segfault，用 CreateInputBuffer
6. `avgBitRate=0`+`maxBitRate=0` → NVENC 无码率天花板 → GPU 65% 空闲、FPS 暴跌 2.2×；须设估计值（如 7000）
7. **drain 数据必须完整消费**，丢弃即相位错位（`nvenc-stream-drain-backpressure-iron-law`）

### 关键 offset（SDK 13.0）
- NV_ENC_RC_PARAMS：version@0, mode@4, constQP@8-16, avgBR@20, maxBR@24, targetQuality@88, lookaheadDepth@90, multiPass@100
- NV_ENC_PIC_PARAMS：version@0, inputBuffer@40, outputBitstream@48, completionEvent@56, codecPicParams@76
- NV_ENC_LOCK_BITSTREAM：version@0, bitfield@4, outputBitstream@8, size@36, ptr@56

### CE-Pipeline 异步编码
`encode_frames_batch_ce_pipeline()` 三阶段（Harvest/Submit/Drain），per-frame completionEvent 消除同步阻塞。pipeline_depth=4 默认。**生产最优：ce-pipeline + pipe=4 + LA=8 + CONSTQP = 584 FPS**（Tesla T4 720×576@25fps）。

### 空帧防御栈
异步 NVENC ~7% completionEvent 空帧率（DMA 竞态假说）。Tier 0 首 LockBitstream 重试 → Tier 1-B 零长度 IDR 重编码（~67%）→ Tier 1-A Writer 帧计数插值（~33%）→ Tier 3-E `__del__` 兜底。CONSTQP 零空帧可跳 Tier 1-B。

### RC 模式排名
CONSTQP 🥇（+29~66%，文件最小）> QVBR 🥈 > VBR_HQ 🥉。避免：v6.4.3.1+VBR_HQ+LA=8+pipe=4（-4.5% 丢帧）；VBR_HQ+LA=8（-35%）；VBR/QVBR+avgBitrate=0。

## 5. 工作偏好

- 简体中文交流，代码/技术术语保留英文；**修改代码保留原有注释**
- 只修 Bug 不改架构、外科手术式最小修改、不主动重构无关代码
- 遇到歧义先说明假设/多解，不静默选型
- OOM 自动降级：batch 减半重试至 1，持久化到 config；ESRGAN 还可降 tile_size
- Checkpoint/resume：`temp/{video_name}_ifrnet/esrgan/checkpoint.json`，删文件强制重处理
- 切勿调用 `pycuda.autoinit`（与 PyTorch CUDA context 冲突）

## 6. 相关记忆索引

- 版本矩阵：[[ifrnet-v6-4x-version-matrix]]
- 模块化拆分：[[ifrnet-v6451-modular-split]]
- 生产配置：[[v4-production-best-config]]、[[rc-mode-performance-ranking]]、[[benchmark-ifrnet-v6.4.x-summary]]
- 跨段优化：[[esrgan-cross-segment-optimization-complete]]
- 完整坑清单：[[nvenc-ctypes-integration]]、[[nvenc_ctypes_verified_layouts]]、[[nvenc-ce-pipeline-architecture]]、[[nvenc-empty-frame-defense]]
