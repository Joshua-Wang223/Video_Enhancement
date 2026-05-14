# Video Enhancement — 项目技术总结

> **版本**: v2.0.0（优化版）｜IFRNet 后端 **v6.3.5**｜Real-ESRGAN 路径 **`realesrgan_video`（v6.4 深度流水线）**  
> **最后更新**: 2026-05-14

---

## 项目概述

Video Enhancement 是一套 GPU 侧视频增强流水线，整合：

- **IFRNet（v6.3.5 后端）**：光流插帧，帧率提升 2x–16x；双流深度流水线、可选 TensorRT / CUDA Graph / `torch.compile`。
- **Real-ESRGAN（优化处理器 → `external/realesrgan_video`）**：真实场景超分，2x / 4x；深度模块化流水线（读帧→SR→GFPGAN→写帧）、多片段复用模型/TRT。

历史入口 `main_video_v5_single.py`、`external/Real-ESRGAN/inference_realesrgan_video_v5_single.py` 等仍保留作对照；**当前推荐编排**见下文 CLI 真源。

---

## 配置真源

未在命令行显式覆盖时，**默认值以仓库根目录 [`config/default_config.json`](config/default_config.json) 为准**（`processing` / `paths` / `models` / `output` / `temp_files` / `logging`）。  
`paths.base_dir` 留空时由 `config_manager` 根据配置文件位置自动推算项目根；`output_dir`、`temp_dir`、`log_dir`、`trt_cache_dir`、`models_gfpgan_dir` 留空时派生为 `base_dir/output`、`base_dir/temp`、`base_dir/logs`、`base_dir/.trt_cache`、`base_dir/models_GFPGAN`。

---

## 架构设计

### 模块层次

```
┌─────────────────────────────────────────────────────────────┐
│  CLI 入口（推荐）                                           │
│  src/main_video_optimized.py  · VideoProcessor              │
│  [FIX-C 归一化 / --report / --skip-seg-normalize]            │
└────────────────────────────┬────────────────────────────────┘
                             │
              ┌──────────────┴───────────────┐
              │                              │
    ┌─────────▼──────────┐       ┌─────────▼──────────────────────┐
    │ IFRNet 处理器       │       │ Real-ESRGAN 处理器（优化版）    │
    │ ifrnet_processor_   │       │ realesrgan_processor_video_   │
    │ v6_1_single.py      │       │ optimized.py                  │
    └─────────┬───────────┘       └─────────┬──────────────────────┘
              │                              │
    ┌─────────▼───────────┐       ┌──────────▼─────────────────────┐
    │ IFRNet 后端          │       │ realesrgan_video           │
    │ external/IFRNet/    │       │ external/realesrgan_video/ │
    │ process_video_      │       │ main.py（main_optimized）      │
    │ v6_3_5_single.py    │       │   create_video_enhancer()     │
    └─────────────────────┘       │   run_pipeline_for_video()    │
                                  │   _HWProfile / PreviewWriter  │
                                  │ pipeline / ffmpeg_io / TRT 等   │
                                  └────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  工具层                                                      │
│  config_manager · video_utils · video_fixer · output_filter │
└─────────────────────────────────────────────────────────────┘
```

### 核心数据流（`interpolate_then_upscale`）

音频提取 → 按时长分段 → IFRNet 分段插帧（断点）→ 分段列表直连 Real-ESRGAN 超分（断点）→ [FIX-C timescale 归一化] → `merge_videos_by_codec` → 合并音频 → 输出 / 清理临时文件。

---

## 核心模块说明（当前推荐路径）

| 组件 | 路径 | 说明 |
|------|------|------|
| 主入口 | `src/main_video_optimized.py` | 流程编排、`VideoProcessor`、CLI 覆盖配置；FIX-C 归一化、`--report` 流水线报告 |
| IFRNet 处理器 | `src/processors/ifrnet_processor_v6_1_single.py` | 对接 `process_video_v6_3_5_single.py` |
| Real-ESRGAN 处理器 | `src/processors/realesrgan_processor_video_optimized.py` | 动态加载并调用 `realesrgan_video/main.py` |
| IFRNet 后端 | `external/IFRNet/process_video_v6_3_5_single.py` | 当前默认后端版本 |
| Real-ESRGAN 深度子项目 | `external/realesrgan_video/` | v6.4 流水线实现；含 HW Profile 探测、PreviewWriter、多片段复用 |
| 配置 | `src/utils/config_manager.py` | JSON 加载、路径派生、CLI 覆盖 |

### `src/utils/video_utils.py`（摘录）

`VideoInfo`、`smart_extract_audio`、`split_video_by_time`、`merge_videos_by_codec`、`verify_video_integrity`、`get_video_codec`、`encode_video` 等 — 与分段直连、合并阶段共用。

---

## 新增核心特性详解

### FIX-C：分段 timescale 归一化（v2.0.0）

**入口：** `main_video_optimized.py` `_normalize_segs_for_copy()`

**问题：** 多个独立 ffmpeg 子进程编码的分段在 MP4 timescale 上存在微差，导致 `concat` demuxer 触发 `AVEROR_EXIT(254)`。

**解法：** 最终合并前对所有 H.264/H.265 分段执行零损耗 remux，统一 `-video_track_timescale 90000`。使用 `ffprobe` 三重校验（rc=0 + 文件 > 4KB + 视频帧 duration > 0），替代旧版 `-bsf:v dump_extra`（该 BSF 在 MP4/AVCC 格式下产生畸形 packet 导致视频流静默丢失）。

**CLI 控制：** `--skip-seg-normalize`（调试用）。

### 流水线报告（--report）

**入口：** `main_video_optimized.py` `_write_final_report()`

端到端流水线 JSON 报告，成功或失败均写出，包含：
- 运行环境（Python / PyTorch / GPU / FFmpeg）
- 输入输出文件大小、时长、体积比
- GPU 峰值显存
- 关键 CLI 参数快照
- 批量模式可附加 `extra` 字段

### GPU 硬件型号探测（HW Profile）

**入口：** `external/realesrgan_video/main.py` `_HWProfile` / `_GPU_PROFILES_TABLE`

内置 20+ GPU 型号静态表（H100/H800 → GTX 1080），包含算力分级、NVDEC/NVENC 支持、PCIe 带宽。用于：
- 自动选择最优 NVENC 编码器（`HardwareCapability.best_encoder()`）
- 打印硬件 profile 供运维参考

不依赖运行时 NVML 检测，避免驱动兼容问题。

### PreviewWriter + 子进程 GUI 检测

**入口：** `external/realesrgan_video/main.py` `PreviewWriter`

实时预览窗口，支持按 `q` 键退出预览而不中断流水线。GUI 可用性在独立子进程中测试（防止 Qt xcb abort 误杀主进程）。

### 多片段复用 API

**入口：** `external/realesrgan_video/main.py`

- `create_video_enhancer(args)` — 一次性加载 / 构建所有重型组件（模型、TRT Engine、GFPGAN 子进程），返回可复用 enhancer 字典
- `run_pipeline_for_video(enhancer, input_video, output_video)` — 使用预建 enhancer 处理单视频，完成后关闭一次性资源但不销毁 enhancer

多段任务中首个分段调用 `create_video_enhancer`，后续分段直接复用。

---

## 外部后端版本对照（摘录）

### IFRNet

| 文件 | 说明 |
|------|------|
| `process_video_v6_3_5_single.py` | **当前对接**（v6.3.5） |
| `process_video_v6_3_0_single.py` 等 | 历史 / 对照 |

### Real-ESRGAN

| 路径 | 说明 |
|------|------|
| `external/realesrgan_video/` | **优化版处理器默认调用**（深度流水线 + 多片段复用 + HW Profile + PreviewWriter） |
| `external/Real-ESRGAN/inference_realesrgan_video_v6_4_optimized.py` 等 | 独立优化脚本，保留参考 |

---

## 设计原则

**模块化**：处理器与后端通过配置对象解耦；Real-ESRGAN 侧子项目可单独演进。

**配置驱动**：JSON 为默认真源，CLI 为覆盖层。

**渐进降级**：OOM 降批量、硬件编解码不可用时回退软解/软编。

**I/O 最小化**：分段直连减少中间合并；智能音频 copy 减少重编码。

**鲁棒性优先**：FIX-C 三重校验确保合并可靠性；HW Profile 不依赖 NVML；子进程 GUI 检测防止崩溃。

---

## 依赖关系图（推荐入口）

```
main_video_optimized.py
  ├── config_manager.Config
  ├── video_utils
  ├── ifrnet_processor_v6_1_single
  │     └── external/IFRNet/process_video_v6_3_5_single.py
  └── realesrgan_processor_video_optimized
        └── external/realesrgan_video/main.py + pipeline + ffmpeg_io
              ├── basicsr / realesrgan / facexlib / gfpgan（人脸分支）
              ├── 可选 tensorrt / pycuda / onnx（TRT 路径）
              ├── _HWProfile / _GPU_PROFILES_TABLE（硬件型号表）
              └── PreviewWriter（子进程 GUI 检测）
```

---

## 许可证

MIT License — 见根目录 LICENSE；外部子项目遵循各自许可证。
