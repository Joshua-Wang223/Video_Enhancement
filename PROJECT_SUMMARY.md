# Video Enhancement — 项目技术总结

> **版本**: v5（单卡优化版）  
> **最后更新**: 2026-03-04

---

## 项目概述

Video Enhancement 是一个面向 GPU 工作站的批量视频增强流水线，整合了两个深度学习模型：

- **IFRNet**：基于双向光流估计的视频帧插值，将视频帧率提升 2x–16x
- **Real-ESRGAN**：基于 GAN 的真实世界超分辨率，将视频分辨率放大 2x 或 4x

项目从 v1 演进至 v5，核心改进集中在推理加速（FP16、torch.compile、CUDA Graph、TensorRT）、I/O 优化（分段直连流水线、NVDEC/NVENC）和稳定性（OOM 自动降级、断点续传）三个维度。

---

## 架构设计

### 模块层次

```
┌──────────────────────────────────────────────────────┐
│                     CLI 入口层                        │
│           src/main_video_v5_single.py                │
│       VideoProcessor（流程编排 + 音频管理）             │
└────────────────────┬─────────────────────────────────┘
                     │
       ┌─────────────┴──────────────┐
       │                            │
┌──────▼──────────┐      ┌──────────▼──────────────┐
│  IFRNet处理器   │      │  Real-ESRGAN处理器       │
│  v5 单卡版      │      │  v5 单卡版               │
│                 │      │                          │
│ · FP16          │      │ · FP16                  │
│ · torch.compile │      │ · torch.compile         │
│ · CUDA Graph    │      │ · 批量推理+预取          │
│ · NVDEC/NVENC   │      │ · NVDEC/NVENC           │
│ · OOM降级       │      │ · OOM降级               │
│ · 断点续传      │      │ · 断点续传              │
└──────┬──────────┘      └──────────┬──────────────┘
       │                            │
       │  process_video_v5_single   │  inference_realesrgan_video_v5_single
       │                            │
┌──────▼──────────────────────────────▼──────────────┐
│                   外部后端层                         │
│   external/IFRNet/    external/Real-ESRGAN/         │
└─────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│                   工具层                             │
│   config_manager  video_utils  video_fixer           │
│   output_filter                                      │
└─────────────────────────────────────────────────────┘
```

### 核心数据流（interpolate_then_upscale 模式）

```
输入视频
  │
  ├─ extract_audio() → audio.aac（或直接 copy）
  │
  ├─ split_video_by_time() → [seg_000.mp4, seg_001.mp4, ...]
  │                                ↓ 每段独立处理（支持断点）
  ├─ IFRNet 批量推理 FP16 → [ifr_000.mp4, ifr_001.mp4, ...]
  │         ↕ checkpoint.json
  │                                ↓ 直接传递（无中间合并）
  ├─ Real-ESRGAN 批量推理 FP16 → [esr_000.mp4, esr_001.mp4, ...]
  │         ↕ checkpoint.json
  │
  ├─ merge_videos_by_codec() → merged.mp4
  │
  ├─ add_audio_to_video() → final.mp4
  │
  └─ 输出 / 清理临时文件
```

---

## 核心模块说明

### `src/main_video_v5_single.py` — 流程编排器

**类**：`VideoProcessor`

| 方法 | 说明 |
|------|------|
| `process_single_video(input)` | 处理单个视频的完整生命周期 |
| `process_batch()` | 批量处理，失败跳过继续 |
| `_process_interpolate_then_upscale()` | 先插帧后超分的具体流程 |
| `_process_upscale_then_interpolate()` | 先超分后插帧的具体流程 |
| `_extract_audio()` | 智能音频提取（copy 优先） |
| `prompt_cleanup()` | 处理后询问或自动清理 |

**临时文件追踪**：使用 `temp_files_tracker` 字典记录所有临时目录、音频文件、断点文件，确保清理完整。

### `src/processors/ifrnet_processor_v5_single.py` — IFRNet 处理器

**类**：`IFRNetProcessor`

| 方法 | 说明 |
|------|------|
| `process_video_segments(input)` | 对完整视频分段并插帧，返回分段列表 |
| `process_segments_directly(segs, name)` | 直接接收上游分段并插帧（直连接口） |
| `_process_segments(segs, checkpoint)` | 带断点的分段处理循环 |
| `_load_checkpoint()` | 读取断点记录 |
| `_save_checkpoint(done)` | 保存已完成分段记录 |

**v5 新增参数**：`use_fp16`、`use_compile`、`use_cuda_graph`、`use_tensorrt`、`use_hwaccel`、`report_json`

### `src/processors/realesrgan_processor_video_v5_single.py` — Real-ESRGAN 处理器

**类**：`RealESRGANVideoProcessor`

| 方法 | 说明 |
|------|------|
| `process_video(input, output)` | 完整处理单个视频（含合并） |
| `process_video_segments(input)` | 分段超分，返回分段列表 |
| `process_segments_directly(segs, name)` | 直接接收上游分段并超分（直连接口） |

**v5 新增参数**：`batch_size`、`prefetch_factor`、`use_compile`、`use_tensorrt`、`use_hwaccel`、`report_json`

### `src/utils/video_utils.py` — 视频工具库

| 类/函数 | 说明 |
|---------|------|
| `VideoInfo` | 视频元数据封装（分辨率、帧率、时长、编码、音频等） |
| `smart_extract_audio()` | 智能音频提取（优先 copy，失败则转码） |
| `extract_audio()` | 标准音频提取 |
| `add_audio_to_video()` | 合并音频到视频 |
| `get_video_duration()` | 获取视频时长 |
| `verify_video_integrity()` | 验证视频文件完整性 |
| `split_video_by_time()` | 按时长分割视频为多段 |
| `merge_videos()` | 合并多个视频文件 |
| `merge_videos_by_codec()` | 智能合并（自动检测编码一致性） |
| `get_video_codec()` | 获取视频编码格式 |
| `encode_video()` | 视频编码（CRF、preset 等参数） |
| `format_time()` | 秒数格式化为 HH:MM:SS |

### `src/utils/config_manager.py` — 配置管理

**类**：`Config`

| 方法 | 说明 |
|------|------|
| `get(*keys, default)` | 嵌套键路径读取（如 `get("models", "ifrnet", "use_fp16")`） |
| `set(*keys, value)` | 嵌套键路径写入（命令行参数覆盖用） |
| `get_temp_dir(subdir)` | 获取临时目录路径 |
| `get_input_videos()` | 解析并返回输入视频列表 |
| `get_output_path(input, suffix)` | 生成输出文件路径 |
| `get_section(key)` | 获取整个配置节 |
| `save(path)` | 保存当前配置为 JSON |

### `src/utils/video_fixer.py` — 视频修复

**类**：`VideoFixer`

处理损坏或不完整的视频文件：支持重新封装（remux）、关键帧修复、时间戳修复等策略。

### `src/utils/output_filter.py` — 输出过滤

**类/函数**：`TileFilter`、`filter_tile_output()`

过滤 Real-ESRGAN 推理时产生的分块（tile）信息输出，避免刷屏。

---

## 版本演进历史

### v1.0（2026-02-04）— 基础版本

- 实现 IFRNet 插帧 + Real-ESRGAN 超分的基础流程
- 支持 `interpolate_then_upscale` 和 `upscale_then_interpolate` 两种模式
- 基础断点续传（仅插帧阶段）
- 批量处理

### v2.0（2026-02-04）— 流水线优化

- **分段直连**：插帧结果分段直接传入超分，省去中间合并与二次分段
- 超分阶段断点续传
- 批量处理后统一询问清理（延迟清理）
- 磁盘 I/O 减少 38%，总时间减少 33%

### v3.0 — 视频直接处理

- Real-ESRGAN 改为直接处理视频文件（而非提取帧→处理→重组）
- 减少中间 PNG 文件暂存，节省大量磁盘空间
- 引入 `merge_videos_by_codec()` 智能编码一致性检测

### v5.0（当前）— 单卡深度优化

- **FP16 推理**：全面支持半精度推理，显存减半
- **torch.compile**：PyTorch 2.0 图编译支持
- **CUDA Graph**：IFRNet 推理图固化，降低 CPU 调度开销
- **TensorRT**：可选 TensorRT 引擎加速
- **NVDEC/NVENC**：硬件视频编解码自动探测
- **OOM 自动降级**：显存不足时自动减小批量大小
- **批量推理 + 预取**：Real-ESRGAN 批量帧推理 + `prefetch_factor` 数据预取
- **智能音频**：`smart_extract_audio()` 优先无损 copy
- **性能报告**：可选 JSON 格式性能指标输出

---

## 外部后端说明

### IFRNet 后端版本

| 文件 | 版本 | 说明 |
|------|------|------|
| `process_video.py` | v1 | 基础版本 |
| `process_video_v3.py` | v3 | 分段处理 |
| `process_video_v4.py` | v4 | 批量推理 |
| `process_video_v5.py` | v5 多卡 | 多 GPU 版本 |
| `process_video_v5_single.py` | **v5 单卡（★ 当前）** | FP16 + CUDA Graph + NVDEC/NVENC |

### Real-ESRGAN 后端版本

| 文件 | 版本 | 说明 |
|------|------|------|
| `inference_realesrgan_video.py` | v1 | 基础版本 |
| `inference_realesrgan_video_v3.py` | v3 | 分段处理 |
| `inference_realesrgan_video_v4.py` | v4 | 批量推理 |
| `inference_realesrgan_video_v5.py` | v5 多卡 | 多 GPU 版本 |
| `inference_realesrgan_video_v5_single.py` | **v5 单卡（★ 当前）** | FP16 + 批量推理 + NVDEC/NVENC |
| `inference_realesrgan_video_v6_single.py` | v6（实验） | 进一步优化，待稳定 |

---

## 设计原则

**模块化**：处理器与后端解耦，通过配置对象传参，便于独立测试和替换后端。

**配置驱动**：所有参数统一在 JSON 配置文件管理，命令行参数作为覆盖层，无需修改源码。

**渐进降级**：OOM 自动降级、硬件编解码不可用时回退软件编解码，确保在各种环境下均能运行。

**断点安全**：每个处理分段完成后立即保存断点，中断损失最小化。

**I/O 最小化**：分段直连消除中间合并，智能音频 copy 避免重新编码，硬件编解码减少 CPU 负担。

---

## 未来计划

**近期**
- [ ] 整合 v6 Real-ESRGAN 后端（当前为实验状态）
- [ ] 添加 Web UI（Gradio 或 Streamlit）
- [ ] 支持 AV1（libsvtav1）输出编码

**中期**
- [ ] 多 GPU 并行批量处理
- [ ] 实时处理预览模式
- [ ] 视频质量自动评估（PSNR/SSIM）

**长期**
- [ ] 集成更多 AI 增强模型（去噪、去模糊等）
- [ ] Docker 化部署
- [ ] 云端处理支持

---

## 依赖关系图

```
main_video_v5_single.py
  ├── config_manager.py
  ├── video_utils.py
  ├── ifrnet_processor_v5_single.py
  │     ├── config_manager.py
  │     ├── video_utils.py
  │     └── external/IFRNet/process_video_v5_single.py
  │           ├── IFRNet 模型（IFRNet_S / IFRNet_L）
  │           └── CUDA / cuDNN / TensorRT（可选）
  └── realesrgan_processor_video_v5_single.py
        ├── config_manager.py
        ├── video_utils.py
        ├── output_filter.py
        └── external/Real-ESRGAN/inference_realesrgan_video_v5_single.py
              ├── basicsr / realesrgan / facexlib / gfpgan
              ├── Real-ESRGAN 模型
              └── CUDA / cuDNN / TensorRT（可选）
```

---

## 许可证

MIT License — 见根目录 LICENSE 文件。  
外部依赖（IFRNet、Real-ESRGAN）遵循各自许可证。
