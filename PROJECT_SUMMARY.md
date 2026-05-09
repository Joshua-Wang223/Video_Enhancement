# Video Enhancement — 项目技术总结

> **版本**: v2.0.0（优化版）｜IFRNet 后端 **v6.3.3**｜Real-ESRGAN 路径 **`realesrgan_video_ds`（v6.4 深度流水线）**  
> **最后更新**: 2026-05-09

---

## 项目概述

Video Enhancement 是一套 GPU 侧视频增强流水线，整合：

- **IFRNet（v6.3.3 后端）**：光流插帧，帧率提升 2x–16x；双流深度流水线、可选 TensorRT / CUDA Graph / `torch.compile`。
- **Real-ESRGAN（优化处理器 → `external/realesrgan_video_ds`）**：真实场景超分，2x / 4x；深度模块化流水线（读帧→SR→GFPGAN→写帧）、多片段复用模型/TRT。

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
    │ IFRNet 后端          │       │ realesrgan_video_ds           │
    │ external/IFRNet/    │       │ external/realesrgan_video_ds/ │
    │ process_video_      │       │ main.py（main_optimized）      │
    │ v6_3_3_single.py    │       │ pipeline / ffmpeg_io / TRT 等   │
    └─────────────────────┘       └────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  工具层                                                      │
│  config_manager · video_utils · video_fixer · output_filter │
└─────────────────────────────────────────────────────────────┘
```

### 核心数据流（`interpolate_then_upscale`）

与 README 一致：音频提取 → 按时长分段 → IFRNet 分段插帧（断点）→ 分段列表直连 Real-ESRGAN 超分（断点）→ `merge_videos_by_codec` → 合并音频 → 输出 / 清理临时文件。

---

## 核心模块说明（当前推荐路径）

| 组件 | 路径 | 说明 |
|------|------|------|
| 主入口 | `src/main_video_optimized.py` | 流程编排、`VideoProcessor`、CLI 覆盖配置 |
| IFRNet 处理器 | `src/processors/ifrnet_processor_v6_1_single.py` | 对接 `process_video_v6_3_3_single.py` |
| Real-ESRGAN 处理器 | `src/processors/realesrgan_processor_video_optimized.py` | 动态加载并调用 `realesrgan_video_ds/main.py` |
| IFRNet 后端 | `external/IFRNet/process_video_v6_3_3_single.py` | 当前默认后端版本 |
| Real-ESRGAN 深度子项目 | `external/realesrgan_video_ds/` | v6.4 流水线实现 |
| 配置 | `src/utils/config_manager.py` | JSON 加载、路径派生、CLI 覆盖 |

### `src/utils/video_utils.py`（摘录）

`VideoInfo`、`smart_extract_audio`、`split_video_by_time`、`merge_videos_by_codec`、`verify_video_integrity` 等 — 与分段直连、合并阶段共用。

---

## 外部后端版本对照（摘录）

### IFRNet

| 文件 | 说明 |
|------|------|
| `process_video_v6_3_3_single.py` | **当前对接**（v6.3.3） |
| `process_video_v6_3_0_single.py` 等 | 历史 / 对照 |

### Real-ESRGAN

| 路径 | 说明 |
|------|------|
| `external/realesrgan_video_ds/` | **优化版处理器默认调用**（深度流水线） |
| `external/Real-ESRGAN/inference_realesrgan_video_v6_4_optimized.py` 等 | 独立优化脚本，保留参考 |

---

## 设计原则

**模块化**：处理器与后端通过配置对象解耦；Real-ESRGAN 侧子项目可单独演进。

**配置驱动**：JSON 为默认真源，CLI 为覆盖层。

**渐进降级**：OOM 降批量、硬件编解码不可用时回退软解/软编。

**I/O 最小化**：分段直连减少中间合并；智能音频 copy 减少重编码。

---

## 依赖关系图（推荐入口）

```
main_video_optimized.py
  ├── config_manager.Config
  ├── video_utils
  ├── ifrnet_processor_v6_1_single
  │     └── external/IFRNet/process_video_v6_3_3_single.py
  └── realesrgan_processor_video_optimized
        └── external/realesrgan_video_ds/main.py + pipeline + ffmpeg_io
              ├── basicsr / realesrgan / facexlib / gfpgan（人脸分支）
              └── 可选 tensorrt / pycuda / onnx（TRT 路径）
```

---

## 许可证

MIT License — 见根目录 LICENSE；外部子项目遵循各自许可证。
