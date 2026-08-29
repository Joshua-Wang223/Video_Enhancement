# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言偏好 / Language Preference

思考及回答首选**简体中文**，代码和专业技术术语保留英文。

## Project overview

Video Enhancement is a GPU-accelerated video processing pipeline that integrates **IFRNet frame interpolation** (2x–16x frame rate) and **Real-ESRGAN super-resolution** (2x/4x resolution). It runs on NVIDIA GPUs with optional TensorRT, FP16, CUDA Graph, and torch.compile acceleration.

## Main entry point

```bash
python src/main_video_optimized.py -c config/default_config.json -i input.mp4 -o output.mp4
```

**`src/main_video_optimized.py`** is the only active entry point. All other `src/main_video_*.py` files are historical/legacy.

Two processing modes:
- `interpolate_then_upscale` (default) — interpolate at original resolution first, then upscale
- `upscale_then_interpolate` — upscale first, then interpolate on higher-res frames

To skip a stage: `--skip-interpolate` or `--skip-upscale`. Common flags: `--use-tensorrt-ifrnet`, `--use-tensorrt-esrgan`, `--face-enhance`, `--batch-mode`, `--dry-run`.

## Architecture

```
src/main_video_optimized.py          # CLI entry, orchestration, VideoProcessor
  ├── src/processors/ifrnet_processor_video_optimized.py   # IFRNet processor
  │     └── external/ifrnet_video/main.py            #   IFRNet backend (v6.4.5.1)
  ├── src/processors/realesrgan_processor_video_optimized.py  # Real-ESRGAN processor
  │     └── external/realesrgan_video/main.py            #   Real-ESRGAN backend (v6.4)
  ├── src/utils/config_manager.py   # Config loading, path derivation, CLI override
  └── src/utils/video_utils.py      # Split, merge, audio extraction, VideoInfo
```

**Data flow (interpolate_then_upscale):**
Input → extract audio → split into segments → IFRNet interpolate each segment → pass segment list directly to Real-ESRGAN upscale each segment (no intermediate merge) → merge all segments → mux audio → output

The "direct segment passthrough" optimization skips the intermediate merge+re-split step, saving ~30% I/O.

## Configuration is the single source of truth

**`config/default_config.json`** holds all defaults. When the JSON and README/code comments disagree, the JSON wins. `src/utils/config_manager.py` (`Config` class) loads JSON, derives paths from `base_dir` upward, and applies CLI overrides.

Key config sections: `processing` (mode, factors, segment duration), `paths` (auto-derived from `base_dir`), `models.ifrnet`, `models.realesrgan`, `output`, `temp_files`, `logging`.

The JSON uses `"// key"` convention for documentation comments — these keys are ignored at runtime.

## Which files are current vs. historical

This project has accumulated many versioned scripts. Only these are active:

| Purpose | Current file |
|---------|-------------|
| Main entry | `src/main_video_optimized.py` |
| IFRNet processor | `src/processors/ifrnet_processor_video_optimized.py` |
| IFRNet backend | `external/ifrnet_video/main.py` |
| Real-ESRGAN processor | `src/processors/realesrgan_processor_video_optimized.py` |
| Real-ESRGAN backend | `external/realesrgan_video/main.py` |
| Config manager | `src/utils/config_manager.py` |
| Video utils | `src/utils/video_utils.py` |

Everything else in `external/IFRNet/process_video_v*.py`, `external/Real-ESRGAN/inference_realesrgan_video_v*.py`, `src/main_video_v*.py`, `src/processors/*_v[1-5]*.py` is historical or reference-only. Files with `_bak` or ` - Copy` suffix are development backups safe to delete.

## IFRNet backend (v6.4.5.1, modular in `external/ifrnet_video/`)

`external/ifrnet_video/` is the current single-GPU backend package. It was created by verbatim-preserving decomposition of the v6.4.5.1 monolith `external/IFRNet/process_video_v6_4_5_1_single.py`, mirroring `external/realesrgan_video/` layout: `main.py` (IFRNetVideoProcessor + entry), `pipeline.py` (IFRNetPipelineRunner + GPU monitor + queue tuning), `nvenc_sdk.py` (NVENCEncoder / _NVENCEncodeThread / FFmpegMuxer), `tensorrt_accel.py` (TensorRTAccelMixin), `ffmpeg_io.py` (FFmpegFrameReader / FFmpegWriter / HardwareCapability), `config.py` / `ifrnet_utils.py`. Key internal architecture:

- **Three-stage pipeline**: T1 (Reader: NVDEC decode + frame prep → pair_queue), T2 (GPU: IFRNet model inference), T3 (Writer: FFmpeg H.264 encode from result_queue)
- Uses dual CUDA transfer streams (`stream_h2d` for prefetch, `stream_d2h` for output) with CudaEventPool
- `FFmpegFrameReader` reads via ffmpeg pipe with internal prefetch queue
- `FFmpegWriter` writes via ffmpeg stdin pipe
- GPU monitoring thread samples utilization every 2s
- Adaptive queue tuning post-segment (pair_queue / result_queue sizing)
- TRT engine cached under `.trt_cache/` with GPU SM-architecture in filename

The multi-GPU variant `process_video_v6_3_3.py` (historical) has equivalent code but with multi-GPU distribution.

### 关键记忆文件参考

| 主题 | 记忆文件 | 说明 |
|------|---------|------|
| 版本演化路径 | [ifrnet-v6-4x-version-matrix.md](memory/ifrnet-v6-4x-version-matrix.md) | 6 版本 3 层架构演化，最终收敛状态 |
| 线程协调规则 | [pipeline-thread-coordination.md](memory/pipeline-thread-coordination.md) | 退出时序、NVENCEncoder 锁使用规则、Writer 循环条件 |
| 基准测试报告 | [benchmark-ifrnet-v6.4.x-summary.md](memory/benchmark-ifrnet-v6.4.x-summary.md) | 六版本性能排名、帧完整性分析、生产推荐 |
| 原始数据速查 | [v6.4.x-benchmark-raw-data.md](memory/v6.4.x-benchmark-raw-data.md) | Batch+Individual 全部数据表、帧计数诊断 |
| T2 估算修正 | [t2-static-estimation-undershoot.md](memory/t2-static-estimation-undershoot.md) | _T2_VAR_MS_TRT 25ms→335ms 修复 |
| bitrate 退化修复 | [v6.4.5-bitrate-unconstrained-degradation.md](memory/v6.4.5-bitrate-unconstrained-degradation.md) | avgBitrate=0 导致 GPU 65% 空闲 |

## Real-ESRGAN backend (v6.4, realesrgan_video)

`external/realesrgan_video/` is a modular subproject with:
- `main.py` — entry, `create_video_enhancer()` / `run_pipeline_for_video()` for multi-segment engine reuse
- `pipeline.py` — 4-level parallel pipeline (read → SR → GFPGAN → write)
- `ffmpeg_io.py` — FFmpeg reader/writer with async prefetch
- `tensorrt_accel.py` — TRT acceleration wrapper
- `gfpgan_subprocess.py` — GFPGAN in isolated subprocess (optional TRT)
- `face_utils.py` — face detection and enhancement
- `config.py` — model paths resolved relative to project root

### 跨段复用优化（2026-07-29 完成）

历经四个 Phase 的渐进优化，ESRGAN 跨段切换从几十秒级降至毫秒级，与 IFRNet v6.4.5.1 架构对齐：

- **Phase A**：GFPGAN 子进程跨段保活（pipeline.close() 回注 `args._early_gfpgan_subprocess`）
- **Phase B-D**：NVENC 驱动会话跨段持续复用（`FIX-SKIP-REOPEN`），跳过 reopen，删除 `reopen()` 方法
- **新增**：`NVENCEncoder._stream_begin(force=True)` 封装 per-segment 状态重置，对齐 IFRNet 同名方法
- **新增**：`main.py` encoder 缓存 key 一致性检查（`_sdk_nvenc_key`），对齐 IFRNet `_get_or_create_nvenc_encoder` 防御模式
- **移除**：`FIX-SEG-START-SYNC` 段首 batch 同步过渡，LA=0 首 batch 直接走 CE pipeline
- **结果**：段边界仅重建 `_NVENCEncodeThread` + `FFmpegMuxer`，encoder 对象与驱动会话跨段持续

记忆文件：[[esrgan-cross-segment-optimization-complete]]

## NVENC 编码子系统 (Level 1 GPU 直通编码)

自 v6.4.2 起引入了直接通过 NVIDIA NVENC SDK 13.0 ctypes 进行 GPU 硬件编码的路径，避免 CPU 侧重编码开销。当前活跃版本（v6.4.3.1/4.1/5.1/5）均包含此子系统。

### 四级降级架构

IFRNet Writer 端根据运行时条件自动选择编码路径：

| Level | 编码方式 | 适用条件 |
|-------|---------|---------|
| Level 1 | NVENC SDK GPU 直通 → FFmpegMuxer (`-c:v copy`) | GPU 可用、NVENC 初始化成功 |
| Level 2 | Pinned Ring Buffer + FFmpeg NVENC (`h264_nvenc`) | NVENC SDK 初始化失败 |
| Level 3 | Pinned Ring Buffer + libx264 软编码 | GPU 编码不可用 |
| Level 4 | 标准 PinnedResultPool 路径 | 最低效 fallback |

Level 选择逻辑: `external/ifrnet_video/main.py:~1315-1418`

### Level 1 核心数据流

```
GPU tensor (RGB) → _rgb_to_nv12_gpu() → NV12 GPU tensor
    → NVENCEncoder.encode_frame()
        → LockInputBuffer → cuMemcpyDtoD_v2 → UnlockInputBuffer
        → EncodePicture (async, with outputBitstream)
        → LockBitstream (blocking) → 获取 H.264 ES bytes
    → result_queue → Writer 线程: FFmpegMuxer → ffmpeg stdin (-f h264 -c:v copy)
    → FFmpegMuxer.close() → ffmpeg 写 moov atom
```

### 关键实现文件

| 版本文件 | NVENC 状态 | 备注 |
|---------|-----------|------|
| `process_video_v6_4_5_1_single.py` | ✅ 参考源 | 已逐字拆分至 `external/ifrnet_video/`（当前活跃后端），原文件保留为参考源/回滚参照；含 ce-pipeline（FIX-ASYNC-COPY / FIX-FLUSH-GRANULARITY / VRAM-CLEANUP） |
| `process_video_v6_4_5_single.py` | ✅ | 与 5.1 等价；2026-07-24 回植 FIX-ASYNC-COPY + VRAM-CLEANUP |
| `process_video_v6_4_4_1_single.py` | ✅ | backport 含三项修复；2026-07-24 回植 FIX-ASYNC-COPY / FIX-FLUSH-GRANULARITY / VRAM-CLEANUP，并修复 `_batchself` sed 误替换 |
| `process_video_v6_4_4_single.py` | ✅ | 修正 cross-stream race；2026-07-24 回植 FIX-ASYNC-COPY + VRAM-CLEANUP |
| `process_video_v6_4_3_1_single.py` | ✅ | backport 含三项修复；2026-07-24 回植 FIX-ASYNC-COPY + VRAM-CLEANUP |
| `process_video_v6_4_3_single.py` | ✅ | 首个稳定 NVENC 版本；2026-07-24 回植 FIX-ASYNC-COPY + VRAM-CLEANUP |
| `process_video_v6_4_2_single.py` | ✅ | 首个 NVENC 实现 (PinnedRingBuffer 路径) |
| `process_video_v6_4_1_single.py` | ❌ | 纯 FFmpeg 编码 |
| `process_video_v6_3_*_single.py` | ❌ | 纯 FFmpeg 编码 |

### NVENCEncoder 类 (ctypes 直接编码)

文件位置: `external/ifrnet_video/nvenc_sdk.py`（`NVENCEncoder@495`、`_NVENCEncodeThread@2780`、`FFmpegMuxer@3179`）

- 使用 `ctypes.CDLL` 直接加载 `libnvcuvid.so` / `nvcuda.dll`，通过 `func_table` 动态索引 NVENC API 函数
- 所有 struct 使用 **byte array + 手动 offset 写入**（不用 `ctypes.Structure` 子类，因 field offset 不可靠）
- input buffer: NV12 GPU tensor → `cuMemcpyDtoD_v2` 到 NVENC input buffer
- bitstream buffer: LockBitstream 获取 H.264 ES bytes
- CUDA primary context 管理: `cuDevicePrimaryCtxRetain` + `cuCtxPushCurrent`（不创建新 context）
- **线程锁**: `_lock` 是 `threading.Lock()`（不可重入），严禁持有 `_lock` 时调用 `flush()`

### NVENC SDK struct 布局关键参考 (SDK 13.0, nvEncodeAPI.h)

最新完整布局在 [nvenc_ctypes_verified_layouts.md](memory/nvenc_ctypes_verified_layouts.md)，核心偏移：

| Struct | 关键字段偏移 | 常见错误 |
|--------|-------------|---------|
| NV_ENC_RC_PARAMS | version@0, mode@4, constQP@8-16, avgBR@20, maxBR@24, targetQuality@88, lookaheadDepth@90, multiPass@100 | 误用 union 布局假设 → offset 计算错误 |
| NV_ENC_PIC_PARAMS | version@0, inputBuffer@40, outputBitstream@48, completionEvent@56, codecPicParams@76 | EOS 帧缺 outputBitstream → 无输出 |
| NV_ENC_LOCK_BITSTREAM | version@0, bitfield@4, outputBitstream@8, size@36, ptr@56 | — |
| NV_ENC_CREATE_INPUT_BUFFER | version@0, height@8 (**luma height 非 total height**) | 设 NV12 total height → 灰色输出 |

struct 版本常量使用 `_sdk13_ver(ver)` 宏：`NVENCAPI_VERSION`(0x0d) | (ver << 16) | (0x7 << 28)。

### CE-Pipeline 异步编码

`encode_frames_batch_ce_pipeline()` 是最高性能编码路径，通过 per-frame CUDA completionEvent 消除 EncodePicture 同步阻塞：

```
Phase 1 (Harvest): slot 重用时 cuEventSynchronize 等待 CE → LockBitstream 获取 H.264
Phase 2 (Submit):  LockInputBuffer → EncodePicture + cuEventCreate(新 CE)
Phase 3 (Drain):   批次结束 drain 所有 pending slots
```

- `pipeline_depth` = 4（默认），范围 1-8
- 每个 slot 持有 `{input_buf, bs_buf, event}`，初始化时 `cuEventCreate(0)`
- ce-pipeline + pipe=4 + LA=8 在 Tesla T4 上达到 **584 FPS** (constqp, 720×576@25fps)，比同步 batch pipe=1 快 +20%

**LA (lookahead) 兼容性**：pipe=4 + LA=8 在 VBR_HQ/QVBR 下有 0.3-0.6% 空帧率，由 Tier 防御栈 100% 恢复。CONSTQP 零空帧。

### 空帧多层防御栈

异步 NVENC 存在 ~7% completionEvent 同步空帧率（DMA 竞态假说），分为三层防御：

| Tier | 位置 | 机制 | 恢复率 |
|------|------|------|--------|
| Tier 0 | 首次 LockBitstream 阻塞重试 | 发现空帧立即 retry | 部分 |
| Tier 1-B | 检查 LockBitstream 返回大小 | 零长度 → IDR 重新编码 | ~67% |
| Tier 1-A | Writer 线程帧计数 | count < expected → 插值填充 | ~33% |
| Tier 3-E | NVENCEncoder.__del__ | 最终确保帧完整性 | 兜底 |

CONSTQP 模式下 Tier 1-B/A 可 100% 恢复空帧，验证见 [pipe4-la8-tier-defense-verified](memory/pipe4-la8-tier-defense-verified.md)。

### RC 模式

| RC Mode | 性能排名 | 说明 |
|---------|---------|------|
| **CONSTQP** (mode=0) | 🥇 | 直接设 QP(qpInterP/qpInterB/qpIntra)，统一最快 (+29~66% vs vbr_hq)，文件最小 |
| VBR_HQ (mode=32) + targetQuality | 🥈 | 替代 CONSTQP 做 CQ 编码，targetQuality = max(1, 51-CRF) |
| QVBR (mode=34) | 🥉 | 介于 constqp 和 vbr_hq 之间 |

**CONSTQP 快速路径**：空帧和 LockBitstream 统计表明 CONSTQP 空帧率为零 → 可跳过 LockBitstream 重试和 Tier 1-B。

**重要警告**: v6.4.5/.1 将 `averageBitRate=0` + `maxBitRate=0` 导致 NVENC 无码率天花板 → GPU 65% 空闲，FPS 暴跌 2.2×。修复方法：设置 avgBitRate 估计值（如 7000）。

### SPS/PPS 处理

- ctypes bitfield 布局不匹配导致 `repeatSPSPPS` 无效（驱动不识别 ctypes `_pack_=1` 布局）
- V2 修复方案：在 encode_frame 返回后、frame write 前，手动注入 SPS/PPS NAL 单元缓存到每帧开头
- LA + pipe=4 启动阶段空帧叠加 per-slot IDR 症状，可能导致 `non-existing PPS 0 referenced` 错误
- 根因修复见 [pipe4-la8-root-cause-fix](memory/pipe4-la8-root-cause-fix.md)：`encode_frame` 中处理 per-slot IDR

### 已知 Bug 模式速查

| 错误码 | 症状 | 根因 |
|--------|------|------|
| code=15 (INVALID_VERSION) | OpenEncodeSessionEx/GetEncodePresetConfig 失败 | struct sizeof 不匹配 (缺字段/SDK offset 错误) |
| code=8 (INVALID_PARAM) | InitializeEncoder 失败 | rcParams offset 错误 / multiPass VBR_HQ 不合法 |
| 死锁 | close() 永久阻塞 | `_lock` 持锁调用 `flush()` |
| 灰色输出 | R=G=B | height 设 NV12 total height 或 D2H 缺 synchronize |
| CRF 不生效 | 码率无区分 | targetQuality 写到错误 offset（3 轮迭代才修复） |
| 文件膨胀 3.3x | 码率失控 | `ctypes.Structure` 缺失 rcParams 字段 → 驱动用默认码率 |

### 关键经验总结

1. **struct 布局必须从 SDK 头文件逐字节验证**，自动解析脚本不可靠（如 `verify_rcparams_offset.py` 误将 NV_ENC_QP 12B struct 当 4B enum）
2. **所有 ctypes struct 用 byte array + 手动 offset 写**，不用 `ctypes.Structure`（field offset 不可靠，且可能缺字段）
3. **GUID 必须从 driver 动态查询**，不能硬编码
4. **RegisterResource 在 T4/driver 580 上 segfault**，用 CreateInputBuffer 替代
5. **outputBitstream 必须每帧设置**（含 EOS 帧），否则 LockBitstream 无输出
6. **CreateInputBuffer.height 对 NV12 必须是 luma height**（H），不是 NV12 total height（H+H/2）

完整参考见 [nvenc-ctypes-integration](memory/nvenc-ctypes-integration.md)。

### 生产配置推荐

基于 v4 4D 全矩阵 GPU 验证 (Tesla T4, 720×576@25fps 真实视频, 180 组合)：

| 场景 | 推荐版本 | RC 模式 | Pipe | LA | 预期 FPS | 理由 |
|------|---------|---------|:----:|:--:|:--------:|------|
| **生产默认** | v6.4.4 | CONSTQP | 4 | 8 | 584 | 最快+最稳+文件最小 |
| 需要质量优先 | v6.4.5.1 | QVBR (无LA) | 4 | 0 | 557 | 稍大文件 +5% 质量 |
| 需要码率控制 | v6.4.5 | VBR_HQ (无LA) | 4 | 0 | 449 | 帧完整但有性能代价 |
| 保守方案 | v6.4.4 | CONSTQP | 4 | 0 | 578 | 零空帧保证 |

**避免的组合**：
- v6.4.3.1 + VBR_HQ + LA=8 + pipe=4 → 大量丢帧 (-4.5%)
- v6.4.5/.1 + VBR_HB/QVBR + avgBitrate=0 → GPU 65% 空闲，性能塌方 2.2×
- VBR_HQ + LA=8 → 空帧 + 性能倒退 (-35% vs LA=0)

**RC 模式选择**：CONSTQP 🥇 > QVBR 🥈 > VBR_HQ 🥉 (CONSTQP 快 29-66% 且文件最小)

详细数据见 [v4-production-best-config](memory/v4-production-best-config.md)、[rc-mode-performance-ranking](memory/rc-mode-performance-ranking.md)、[benchmark-ifrnet-v6.4.x-summary](memory/benchmark-ifrnet-v6.4.x-summary.md)。

### Bug 修复历史速查

| Bug | 影响版本 | 修复版本 | 记忆文件 |
|-----|---------|---------|---------|
| 灰色输出 / 文件膨胀 3.3x / code=15 | v6.4.2/v6.4.3 初始版本 | v6.4.3 稳定版 | [project_v642_v643_bugs](memory/project_v642_v643_bugs.md) |
| 隔帧花屏 (cross-stream race) | v6.4.4 | v6.4.4 修正版 | [v644-encodethread-cross-stream-race](memory/v644-encodethread-cross-stream-race.md) |
| pipe=4+LA=8 启动 SPS/PPS 损坏 | v6.4.3.1/4.1/5/5.1 | V2 Writer-thread-side 注入 | [sps-pps-la-pipe4-startup-corruption](memory/sps-pps-la-pipe4-startup-corruption.md) |
| pipe=4+LA=8 per-slot IDR 花屏 | v6.4.3.1/4.1/5/5.1 | 三文件三处就地修复 | [pipe4-la8-root-cause-fix](memory/pipe4-la8-root-cause-fix.md) |
| pipe=1+LA=8+qvbr 段丢帧 (LOST=4×Nseg) | 特定组合 | 约束配置避免 | [pipe1-la8-qvbr-segment-loss](memory/pipe1-la8-qvbr-segment-loss.md) |
| T2 静态估算低估 1039% | 所有版本 | v6.4.5.1 + backport | [t2-static-estimation-undershoot](memory/t2-static-estimation-undershoot.md) |
| avgBitRate=0 性能塌方 2.2× | v6.4.5/.1 | 设置 avbBitRate 估计值 | [v6.4.5-bitrate-unconstrained-degradation](memory/v6.4.5-bitrate-unconstrained-degradation.md) |
| ctypes bitfield SPS/PPS 不匹配 | 所有 NVENC 版本 | 手动缓存 SPS/PPS NAL | [nvenc-sps-pps-debugging](memory/nvenc-sps-pps-debugging.md) |

## TRT engine caching

Both IFRNet and Real-ESRGAN share `trt_cache_dir` (default `base_dir/.trt_cache`). Engine filenames encode model name, batch size, resolution, FP16 mode, and GPU SM architecture — so different configurations produce different caches. Once built, engines are reused across segments and across video runs. Cache rebuilt automatically when GPU SM changes.

## Dependencies

PyTorch must be installed manually first (matching CUDA version), then `pip install -r requirements.txt`. TensorRT components (`tensorrt`, `pycuda`, `onnx`, `onnxruntime-gpu`) are optional. Do NOT call `pycuda.autoinit` — it conflicts with PyTorch's CUDA context.

## OOM handling

Both processors auto-degrade: on CUDA OOM, batch_size is halved and retried, down to 1. The reduced value is persisted to `max_batch_size` in config. For Real-ESRGAN, `tile_size` can also be reduced (e.g., 512 or 256).

## Checkpoint/resume

Each processor maintains `temp/{video_name}_ifrnet/checkpoint.json` and `temp/{video_name}_esrgan/checkpoint.json`. Re-running the same command skips completed segments. Delete the checkpoint file to force re-processing.

## 跨平台开发与部署 / Cross-Platform Development & Deployment

- **开发环境**: Windows 11 (PowerShell 5.1 / Bash via Git)，日常编码、调试、本地测试在 Windows 上进行
- **生产运行环境**: Linux (目标部署服务器)，实际视频处理任务在 Linux 上执行
- **跨平台要求**: 所有代码必须兼容 Windows 和 Linux，禁止使用平台特定 API 或硬编码路径分隔符

### 必须遵守的跨平台规范

| 场景 | 正确做法 | 禁止做法 |
|------|---------|----------|
| 路径操作 | `pathlib.Path` / `os.path.join` | 硬编码 `\` 或 `/`，字符串拼接路径 |
| 子进程调用 | `subprocess.Popen(..., shell=False)` | `shell=True`（平台注入风险 + 行为差异） |
| 文件编码 | 明确指定 `encoding='utf-8'` | 依赖系统默认编码 |
| 换行符 | Python 通用换行模式（默认 `\n`） | 硬编码 `\r\n` 或手动 CRLF 处理 |
| 可执行文件查找 | `shutil.which('ffmpeg')` | 硬编码路径如 `C:\ffmpeg\bin\ffmpeg.exe` |
| 临时文件 | `tempfile.gettempdir()` + `pathlib` | 硬编码 `/tmp` 或 `C:\Temp` |
| 文件权限 | 避免依赖 Unix 权限位（chmod/chown） | `os.chmod` 设置 0o755 等（Windows 无意义） |
| GPU 检测 | `torch.cuda.is_available()` | 依赖 `/dev/nvidia*` 或 `nvidia-smi` 路径 |
| 系统信号 | 避免 `signal.SIGUSR1` 等 Unix 特有信号 | Windows 不支持 SIGUSR/SIGTERM 语义 |

### 环境依赖
- **Python**: 3.9+，Windows 和 Linux 均需安装
- **PyTorch**: CUDA 版本，需手动安装（匹配目标 CUDA 版本）
- **FFmpeg**: 必须在 PATH 中可用，版本 ≥ 4.3
- **TensorRT** (可选): `tensorrt`, `pycuda`, `onnx`, `onnxruntime-gpu` — 切勿调用 `pycuda.autoinit`（与 PyTorch CUDA context 冲突）

## 项目记忆文件索引

所有项目记忆文件存储在 `memory/` 目录（相对于 `.claude/projects/-workspace-Video-Enhancement/memory/`），MEMORY.md 为索引入口。

### NVENC 编码子系统

| 文件 | 内容 |
|------|------|
| [level1-nvenc-encoding-flow](memory/level1-nvenc-encoding-flow.md) | 四级降级架构、数据流、CONSTQP、FFmpegMuxer 注意事项 |
| [nvenc-ctypes-integration](memory/nvenc-ctypes-integration.md) | SDK 13.0 完整参考：struct 布局、函数索引、版本常量、踩坑记录 |
| [nvenc_ctypes_verified_layouts](memory/nvenc_ctypes_verified_layouts.md) | NV_ENC_RC_PARAMS 128B SEQUENTIAL 布局 GPU 验证 |
| [nvenc-ce-pipeline-architecture](memory/nvenc-ce-pipeline-architecture.md) | 三阶段异步编码设计、性能数据(+13~39%)、LA 兼容性矩阵 |
| [nvenc-rc-params-sequential-layout](memory/nvenc-rc-params-sequential-layout.md) | SEQUENTIAL vs UNION 布局澄清 |
| [nvenc-empty-frame-defense](memory/nvenc-empty-frame-defense.md) | 空帧 DMA 竞态假说、Tier 0/1/3-E 多层防御栈 |
| [nvenc-sps-pps-debugging](memory/nvenc-sps-pps-debugging.md) | ctypes bitfield 不匹配导致 repeatSPSPPS 无效 |
| [nvenc-encodethread-architecture-decision](memory/nvenc-encodethread-architecture-decision.md) | 独立编码线程的架构决策理由 |
| [constqp-fast-path](memory/constqp-fast-path.md) | CONSTQP 零空帧推导的 LockBitstream 重试缩减 |

### Bug 修复与调试

| 文件 | 内容 |
|------|------|
| [project_v642_v643_bugs](memory/project_v642_v643_bugs.md) | 灰色输出、文件膨胀、code=15、Writer 崩溃的根因与修复 |
| [v644-encodethread-cross-stream-race](memory/v644-encodethread-cross-stream-race.md) | 隔帧花屏 Heisenbug：per-thread stream 缺 synchronize |
| [sps-pps-la-pipe4-startup-corruption](memory/sps-pps-la-pipe4-startup-corruption.md) | LA+pipe4 启动 SPS/PPS 损坏机制分析 |
| [pipe4-la8-root-cause-fix](memory/pipe4-la8-root-cause-fix.md) | pipe=4+LA=8 终极根因：encode_frame+per-slot IDR |
| [pipe4-la8-production-garbled-fix](memory/pipe4-la8-production-garbled-fix.md) | [已废弃] 旧方案 A/B 记录 |
| [pipe4-la8-tier-defense-verified](memory/pipe4-la8-tier-defense-verified.md) | Tier 1-B/A 防御 100% 恢复空帧 GPU 验证 |
| [pipe4-la8-lessons-learned](memory/pipe4-la8-lessons-learned.md) | 三轮失败复盘：根因过度泛化、调用链盲区 |
| [pipe1-la8-qvbr-segment-loss](memory/pipe1-la8-qvbr-segment-loss.md) | pipe=1+LA=8+qvbr 固定丢帧 4×Nseg 模式 |
| [phase4-constqp-defense-failure](memory/phase4-constqp-defense-failure.md) | CONSTQP 高速编码缩小 DMA 竞态窗口 Tier 防御失效 |
| [v6.4.5-bitrate-unconstrained-degradation](memory/v6.4.5-bitrate-unconstrained-degradation.md) | avgBitRate=0 导致 GPU 65% 空闲 |
| [v6.4.x-backport-fixes](memory/v6.4.x-backport-fixes.md) | T2估算/bitrate天花板/SPS-PPS backport 到四个旧版本 |

### 性能测试与配置

| 文件 | 内容 |
|------|------|
| [benchmark-ifrnet-v6.4.x-summary](memory/benchmark-ifrnet-v6.4.x-summary.md) | 六版本基准测试综合分析、生产推荐 |
| [v6.4.x-benchmark-raw-data](memory/v6.4.x-benchmark-raw-data.md) | Batch+Individual 原始数据速查表 |
| [v4-production-best-config](memory/v4-production-best-config.md) | 4D 全矩阵最终结论：ce-pipeline+pipe=4+LA=8+constqp=584FPS |
| [4d-empty-frame-loss-heatmap](memory/4d-empty-frame-loss-heatmap.md) | RC×Tech×Pipe×LA 完整空帧/丢帧热力图 |
| [rc-mode-performance-ranking](memory/rc-mode-performance-ranking.md) | constqp > qvbr > vbr_hq 性能排名 |
| [ifrnet-v6-4x-version-matrix](memory/ifrnet-v6-4x-version-matrix.md) | 6 版本 3 层架构演化路径和收敛状态 |
| [la8-real-vs-synthetic-reversal](memory/la8-real-vs-synthetic-reversal.md) | LA=8 合成帧有害(-3.7%) 真实视频有益(+1%) |
| [t2-static-estimation-undershoot](memory/t2-static-estimation-undershoot.md) | T2 TRT 估算 25ms→335ms 修复 |
| [la-harvest-dead-end](memory/la-harvest-dead-end.md) | completion-driven LA harvest 12-slot 实验失败 |

### 开发流程与编码技巧

| 文件 | 内容 |
|------|------|
| [curl-header-fetch-method](memory/curl-header-fetch-method.md) | curl 下载 NVENC 头文件可靠方法 |
| [_rgb_to_nv12_gpu-bgr-optimization](memory/_rgb_to_nv12_gpu-bgr-optimization.md) | input_is_bgr=True 跳过 BGR→RGB 翻转 |
| [pipeline-thread-coordination](memory/pipeline-thread-coordination.md) | 三线程同步、退出时序、锁规则、看门狗 |
| [feedback_minimal_fixes](memory/feedback_minimal_fixes.md) | 用户反馈：只修 bug 不改架构 |
| [episodic-memory-mcp-fix](memory/episodic-memory-mcp-fix.md) | WPS Claude Code Node.js 缺失修复 |

### ESRGAN NVENC 跨段修复（2026-07-28）

| 文件 | 内容 |
|------|------|
| [esrgan-cross-segment-sigsegv-fix-stack](memory/esrgan-cross-segment-sigsegv-fix-stack.md) | 段2 SIGSEGV 根因链（EOS 复用→reopen→gen≥2 必崩）与修复栈；死代码 nvenc_writer.py 教训；逃生门 ESRGAN_DISABLE_SDK_NVENC=1 |
| [esrgan-reopen-slot-phase-misalignment](memory/esrgan-reopen-slot-phase-misalignment.md) | reopen() 保留 _frame_idx 致 LA>0 提交/drain slot 相位错位，段2+ _ensure_slot_free 告警刷屏+丢帧；同步归零修复 |
| [esrgan-segment-reuse-frame-idx-reset](memory/esrgan-segment-reuse-frame-idx-reset.md) | [reopen 前时代] _frame_idx 清零破坏驱动 LA 状态机，段2+ 丢帧 85%+；文末附 reopen 时代反转说明 |
| ESRGAN 侧更多记忆 | esrgan-nvenc-slot-backpressure / esrgan-empty-final-chunk-skips-eos / esrgan-pinned-buffer-pool-race / realesrgan-nvenc-module-architecture 等，见 MEMORY.md |
