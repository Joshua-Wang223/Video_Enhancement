# 详细使用指南 — Video Enhancement v5

## 目录

1. [安装与环境配置](#1-安装与环境配置)
2. [配置文件详解](#2-配置文件详解)
3. [命令行用法](#3-命令行用法)
4. [处理流程说明](#4-处理流程说明)
5. [v5 硬件加速参数](#5-v5-硬件加速参数)
6. [模型选择指南](#6-模型选择指南)
7. [断点续传](#7-断点续传)
8. [批量处理](#8-批量处理)
9. [故障排除](#9-故障排除)
10. [性能优化建议](#10-性能优化建议)
11. [附录](#11-附录)

---

## 1. 安装与环境配置

### 1.1 系统依赖

#### Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git python3-pip
```

#### macOS
```bash
brew install ffmpeg git
```

#### Windows
1. 下载并安装 [FFmpeg](https://ffmpeg.org/download.html)，添加到系统 PATH
2. 安装 [Git for Windows](https://git-scm.com/download/win)

验证 FFmpeg 安装：
```bash
ffmpeg -version
ffprobe -version
```

### 1.2 Python 环境

推荐使用 conda 或 venv 隔离环境：

```bash
# 使用 conda
conda create -n video_enhance python=3.10
conda activate video_enhance

# 使用 venv
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 1.3 安装 PyTorch

根据 CUDA 版本选择安装命令（务必先安装 PyTorch，再安装其他依赖）：

```bash
# CUDA 12.1（推荐）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU 版本（不推荐，速度极慢）
pip install torch torchvision
```

验证 CUDA 可用：
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 1.4 安装项目依赖

```bash
cd Video_Enhancement
pip install -r requirements.txt
```

### 1.5 初始化项目目录

```bash
python setup_project.py

# 或指定自定义目录
python setup_project.py --base-dir /path/to/your/location
```

初始化脚本会创建 `models_IFRNet/`、`models_RealESRGAN/`、`output/`、`temp/`、`logs/` 等目录。

### 1.6 下载模型文件

#### IFRNet 模型

| 模型文件 | 大小 | 说明 |
|----------|------|------|
| `IFRNet_S_Vimeo90K.pth` | ~10 MB | Small 版，速度快，**推荐日常使用** |
| `IFRNet_L.pth` | ~25 MB | Large 版，质量更高，显存需求更大 |

```bash
mkdir -p models_IFRNet/checkpoints

# Small 模型（推荐）
wget -O models_IFRNet/checkpoints/IFRNet_S_Vimeo90K.pth \
  https://github.com/ltkong218/IFRNet/releases/download/v1.0/IFRNet_S.pth

# Large 模型（可选）
wget -O models_IFRNet/checkpoints/IFRNet_L.pth \
  https://github.com/ltkong218/IFRNet/releases/download/v1.0/IFRNet_L.pth
```

#### Real-ESRGAN 模型

| 模型文件 | 大小 | 适用场景 |
|----------|------|---------|
| `realesr-general-x4v3.pth` | ~64 MB | 通用 4x 超分，**推荐首选** |
| `RealESRGAN_x4plus.pth` | ~64 MB | 写实照片/视频 4x 超分 |
| `RealESRGAN_x2plus.pth` | ~64 MB | 写实照片/视频 2x 超分 |
| `RealESRGANv2-animevideo-xsx2.pth` | ~64 MB | 动漫视频 2x 超分（专用） |

```bash
mkdir -p models_RealESRGAN

# 通用 4x（推荐）
wget -O models_RealESRGAN/realesr-general-x4v3.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth

# 写实 4x（可选）
wget -O models_RealESRGAN/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# 动漫 2x（可选，动漫视频专用）
wget -O models_RealESRGAN/RealESRGANv2-animevideo-xsx2.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx2.pth
```

---

## 2. 配置文件详解

配置文件位于 `config/default_config.json`，是所有处理参数的统一管理入口。命令行参数可覆盖配置文件中的同名设置。

### 2.1 路径配置（`paths`）

```json
{
  "paths": {
    "base_dir": "/workspace/Video_Enhancement",  // 项目根目录（必须修改为实际路径）
    "input_video": "",          // 单视频输入路径（命令行 -i 参数覆盖）
    "output_dir": "/workspace/Video_Enhancement/output",  // 输出目录
    "temp_dir": "/workspace/Video_Enhancement/temp",      // 临时目录（建议使用 SSD）
    "log_dir": "/workspace/Video_Enhancement/logs"        // 日志目录
  }
}
```

> ⚠️ **重要**：首次使用必须修改 `base_dir`、`output_dir`、`temp_dir` 为你的实际路径。

### 2.2 处理配置（`processing`）

```json
{
  "processing": {
    "mode": "interpolate_then_upscale",  // 处理模式，见下方说明
    "interpolation_factor": 2,            // 插帧倍数：2 / 4 / 8 / 16
    "upscale_factor": 2,                  // 超分倍数：2 / 4
    "segment_duration": 30,               // 视频分段时长（秒），影响断点粒度
    "auto_fix_corrupted": false,          // 是否自动修复损坏视频
    "auto_cleanup_temp": true             // 是否自动清理临时文件（false 则询问）
  }
}
```

**处理模式对比**：

| 模式 | 值 | 适用场景 | 优缺点 |
|------|-----|---------|--------|
| 先插帧后超分 | `interpolate_then_upscale` | 低帧率视频（<30fps） | 插帧在原始分辨率进行，速度快 |
| 先超分后插帧 | `upscale_then_interpolate` | 低分辨率视频（<1080p） | 插帧基于高分辨率帧，质量好 |

**`segment_duration` 说明**：
- 决定视频被切分的粒度，断点续传以分段为最小单位
- 推荐值：30 秒（平衡断点粒度与临时文件数量）
- 显存较小时可调小至 15 秒

### 2.3 IFRNet 模型配置（`models.ifrnet`）

```json
{
  "models": {
    "ifrnet": {
      "model_path": "/workspace/Video_Enhancement/models_IFRNet/checkpoints/IFRNet_S_Vimeo90K.pth",
      "use_gpu": true,
      "batch_size": 4,          // 批量推理大小（OOM 时自动降级）
      "max_batch_size": 8,      // 批量大小上限

      // v5 推理优化
      "use_fp16": true,         // FP16 半精度推理（节省显存，提升速度）
      "use_compile": true,      // torch.compile 图编译（PyTorch 2.0+ 有效）
      "use_cuda_graph": true,   // CUDA Graph 推理图固化（降低 CPU 调度开销）
      "use_tensorrt": false,    // TensorRT 加速（需单独安装 tensorrt）

      // v5 硬件编解码
      "use_hwaccel": true,      // 自动探测并启用 NVDEC/NVENC
      "codec": "libx264",       // 分段编码器：libx264 / libx265
      "crf": 18,                // 编码质量（0–51，越低越好，18 为高质量）
      "keep_audio": true,       // 处理分段时保留音频
      "ffmpeg_bin": "ffmpeg",   // FFmpeg 可执行路径

      "report_json": null       // 性能报告输出路径（null 表示不输出）
    }
  }
}
```

### 2.4 Real-ESRGAN 模型配置（`models.realesrgan`）

```json
{
  "models": {
    "realesrgan": {
      "model_name": "realesr-general-x4v3.pth",  // 模型名称
      "model_path": "/workspace/Video_Enhancement/models_RealESRGAN/realesr-general-x4v3.pth",
      "use_gpu": true,
      "denoise_strength": 0.5,  // 降噪强度（0–1），通用模型有效
      "tile_size": 0,           // 分块大小（0 = 自动；显存不足时设 512 或 256）
      "tile_pad": 10,           // 分块边缘填充，避免拼接缝
      "pre_pad": 0,             // 预填充像素
      "fp32": false,            // 是否强制使用 FP32（更精确但更慢）
      "face_enhance": false,    // 是否启用人脸增强（需 GFPGAN）

      // v5 推理优化
      "batch_size": 4,          // 批量推理大小
      "prefetch_factor": 8,     // 数据预取因子（提升 GPU 利用率）
      "use_compile": true,      // torch.compile
      "use_tensorrt": false,    // TensorRT 加速

      // v5 硬件编解码
      "use_hwaccel": true,
      "video_codec": "libx264",
      "crf": 18,
      "ffmpeg_bin": "ffmpeg",

      "report_json": null       // 性能报告输出路径
    }
  }
}
```

**`tile_size` 参数指导**：

| GPU VRAM | 输入分辨率 | 推荐 tile_size |
|----------|-----------|--------------|
| 4 GB | ≤720p | 256 |
| 6 GB | ≤1080p | 512 |
| 8 GB | ≤1080p | 0（自动） |
| 10 GB+ | ≤4K | 0（自动） |

### 2.5 输出配置（`output`）

```json
{
  "output": {
    "format": "mp4",
    "codec": "libx264",          // 最终输出编码器：libx264 / libx265 / libsvtav1
    "preset": "medium",          // 编码速度预设（影响文件大小）
    "crf": 23,                   // 最终输出质量（18–23 为高质量）
    "pix_fmt": "yuv420p",        // 像素格式（yuv420p 兼容性最好）
    "audio_format": "smart",     // 音频处理：smart（自动）/ aac / copy
    "audio_codec": "copy",       // 音频编码器
    "audio_bitrate": "192k"      // 音频码率（audio_format 非 copy 时生效）
  }
}
```

**编码速度预设**（`preset`）：

| 预设 | 说明 |
|------|------|
| `ultrafast` / `superfast` / `veryfast` | 文件大，速度快 |
| `fast` / `medium` | 平衡（推荐） |
| `slow` / `slower` / `veryslow` | 文件小，速度慢 |

**`audio_format: "smart"` 说明**：
- 自动检测原始音频编码
- 无损或兼容编码（AAC、MP3 等）：直接 copy，避免二次压缩失真
- 不兼容编码：转码为 AAC

---

## 3. 命令行用法

### 3.1 完整参数列表

```
usage: main_video_v5_single.py [-h] [--config CONFIG] [--input INPUT]
                               [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]
                               [--mode {interpolate_then_upscale,upscale_then_interpolate}]
                               [--batch] [--interpolation-factor {2,4,8,16}]
                               [--upscale-factor {2,4}]

参数说明：
  -c, --config CONFIG           配置文件路径（默认：config/default_config.json）
  -i, --input INPUT             输入视频文件路径
  --input-dir INPUT_DIR         批量处理：输入视频目录
  -o, --output-dir OUTPUT_DIR   输出目录（覆盖配置文件）
  -m, --mode MODE               处理模式（覆盖配置文件）
  -b, --batch                   启用批量处理模式
  --interpolation-factor N      插帧倍数：2 / 4 / 8 / 16（覆盖配置文件）
  --upscale-factor N            超分倍数：2 / 4（覆盖配置文件）
```

### 3.2 常用示例

```bash
# 最简单的单视频处理（使用默认配置）
python src/main_video_v5_single.py -i input.mp4

# 指定处理模式
python src/main_video_v5_single.py -i input.mp4 -m upscale_then_interpolate

# 自定义倍数（4x插帧 + 4x超分）
python src/main_video_v5_single.py -i input.mp4 \
  --interpolation-factor 4 --upscale-factor 4

# 指定输出目录
python src/main_video_v5_single.py -i input.mp4 -o /data/output

# 批量处理目录下所有视频
python src/main_video_v5_single.py --input-dir ./videos --batch

# 使用自定义配置文件
python src/main_video_v5_single.py -c my_config.json -i input.mp4

# 批量处理 + 自定义倍数
python src/main_video_v5_single.py --input-dir ./videos --batch \
  --interpolation-factor 2 --upscale-factor 2
```

### 3.3 典型场景

#### 场景 A：老电影修复（720p 24fps → 高清 48fps）
```bash
python src/main_video_v5_single.py -i old_movie.mp4 \
  -m upscale_then_interpolate \
  --upscale-factor 4 \
  --interpolation-factor 2
```

#### 场景 B：游戏录屏增强（1080p 30fps → 流畅高质量）
```bash
python src/main_video_v5_single.py -i gameplay.mp4 \
  -m interpolate_then_upscale \
  --interpolation-factor 2 \
  --upscale-factor 2
```

#### 场景 C：动漫视频超分（专用模型）

修改 `config/default_config.json` 中：
```json
{
  "models": {
    "realesrgan": {
      "model_name": "RealESRGANv2-animevideo-xsx2.pth",
      "model_path": "/path/to/models_RealESRGAN/RealESRGANv2-animevideo-xsx2.pth"
    }
  }
}
```

然后运行：
```bash
python src/main_video_v5_single.py -i anime.mp4 --upscale-factor 2
```

#### 场景 D：批量处理家庭视频（节省磁盘）
```bash
python src/main_video_v5_single.py \
  --input-dir ./family_videos \
  --batch \
  --interpolation-factor 2 \
  --upscale-factor 2
```

处理完成后统一询问是否清理临时文件。

---

## 4. 处理流程说明

### 4.1 完整流程图

```
输入视频
   │
   ├─ 完整性检查（verify_video_integrity）
   │   └─ 如损坏 → 自动修复（可选）
   │
   ├─ 提取音频（smart_extract_audio）
   │   └─ 智能 copy 或转码
   │
   ├─ [模式 1: interpolate_then_upscale]
   │   ├─ 分割为 N 个分段（segment_duration）
   │   ├─ IFRNet 插帧处理每个分段
   │   │   └─ 断点保存 checkpoint.json
   │   ├─ [直接对接，无需合并] ──────────────── v5 优化关键
   │   ├─ Real-ESRGAN 超分处理每个分段
   │   │   └─ 断点保存 checkpoint.json
   │   └─ merge_videos_by_codec 合并所有分段
   │
   ├─ [模式 2: upscale_then_interpolate]
   │   ├─ 分割为 N 个分段
   │   ├─ Real-ESRGAN 超分处理每个分段
   │   ├─ [直接对接]
   │   ├─ IFRNet 插帧处理每个分段
   │   └─ 合并所有分段
   │
   ├─ 添加音频（add_audio_to_video）
   │
   └─ 输出最终视频
```

### 4.2 分段直连流水线（v5 核心优化）

v1 的处理方式：
```
分段 → 插帧每段 → 合并 → 再分段 → 超分每段 → 最终合并
                 ^^^^^^^^^^^^^^^^^^^
                 这两步在 v5 中被完全省去
```

v5 的处理方式：
```
分段 → 插帧每段 → [直接传递分段列表] → 超分每段 → 最终合并
```

v5 通过 `process_segments_directly(segment_list)` 接口实现分段直接对接，完全消除了中间合并和二次分段，节省了约 30–50% 的磁盘 I/O 和总处理时间。

---

## 5. v5 硬件加速参数

### 5.1 FP16 推理（`use_fp16`）

将模型权重和中间张量从 FP32 降为 FP16：
- 显存节省约 50%
- 推理速度提升约 1.5–2x（依 GPU 架构而定）
- 精度损失极小，肉眼不可见

适用范围：IFRNet；Real-ESRGAN 通过 `"fp32": false` 控制（效果相同）。

### 5.2 torch.compile（`use_compile`）

PyTorch 2.0+ 引入的图编译优化：
- 首次运行时编译约需 1–5 分钟（后续运行直接使用缓存）
- 编译后推理速度提升 10–40%（依模型和 GPU 而定）
- 需要 PyTorch 2.0+，否则自动忽略

### 5.3 CUDA Graph（`use_cuda_graph`，仅 IFRNet）

将推理图固化为 CUDA Graph，消除 CPU 调度开销：
- 适合固定输入尺寸的场景（分块处理时效果最佳）
- 可降低 CPU/GPU 同步开销 20–40%

### 5.4 TensorRT 加速（`use_tensorrt`，可选）

通过 NVIDIA TensorRT 对模型进行引擎优化：
- 需要额外安装 `tensorrt` Python 包（`pip install tensorrt`）
- 首次构建引擎约需 5–15 分钟（引擎与 GPU 型号绑定，更换 GPU 需重建）
- 推理速度可提升 2–4x

### 5.5 NVDEC/NVENC 硬件编解码（`use_hwaccel`）

通过 FFmpeg 调用 NVIDIA 硬件编解码：
- NVDEC：硬件解码，释放 CPU 资源
- NVENC：硬件编码，速度比软件编码快 3–10x
- 需要 NVIDIA GPU 和支持 NVDEC/NVENC 的 FFmpeg（`ffmpeg -hwaccels` 验证）

### 5.6 OOM 自动降级

当 `batch_size` 导致 CUDA out of memory 时：
1. 自动将 `batch_size` 减半
2. 清空 GPU 缓存后重试
3. 直到 `batch_size = 1` 仍失败则报错并建议调整 `tile_size`

### 5.7 性能报告（`report_json`）

设置 JSON 路径后，处理完成会输出详细性能报告：
```json
{
  "total_time": 1234.5,
  "segments_processed": 10,
  "avg_fps": 2.5,
  "peak_vram_mb": 8192,
  "batch_size_used": 4
}
```

---

## 6. 模型选择指南

### IFRNet 模型选择

| 模型 | 速度 | 质量 | 显存 | 推荐场景 |
|------|------|------|------|---------|
| IFRNet_S（Small） | 快 | 优秀 | ~2 GB | 日常使用，**首选** |
| IFRNet_L（Large） | 较慢 | 更优 | ~4 GB | 对质量有极高要求 |

### Real-ESRGAN 模型选择

| 模型 | 倍数 | 适用内容 | 推荐场景 |
|------|------|---------|---------|
| `realesr-general-x4v3` | 4x | 通用（写实+动漫） | **首选，通用性最强** |
| `RealESRGAN_x4plus` | 4x | 写实照片/视频 | 写实内容 4x 超分 |
| `RealESRGAN_x2plus` | 2x | 写实照片/视频 | 写实内容 2x 超分，速度更快 |
| `RealESRGANv2-animevideo-xsx2` | 2x | 动漫视频 | 动漫专用，效果最佳 |

---

## 7. 断点续传

### 工作原理

每个处理器（IFRNet、Real-ESRGAN）都会在 `temp/` 目录下维护独立的 `checkpoint.json`，记录已完成的分段列表。

处理中断后，重新运行相同命令：
1. 程序检测到 `checkpoint.json` 存在
2. 跳过已处理的分段（显示 `⏭️ 已处理`）
3. 从第一个未处理的分段继续

### 断点文件位置

```
temp/
├── {video_name}_ifrnet/
│   ├── checkpoint.json          # IFRNet 断点
│   ├── segment_000.mp4
│   └── ...
└── {video_name}_esrgan/
    ├── checkpoint.json          # Real-ESRGAN 断点
    ├── processed_000.mp4
    └── ...
```

### 手动清除断点

如需从头重新处理，删除对应的 `checkpoint.json`：
```bash
rm temp/{video_name}_ifrnet/checkpoint.json
rm temp/{video_name}_esrgan/checkpoint.json
```

---

## 8. 批量处理

批量模式处理 `--input-dir` 目录下所有视频文件（支持 mp4、mkv、avi、mov 等）。

```bash
python src/main_video_v5_single.py \
  --input-dir /data/input_videos \
  --batch \
  --interpolation-factor 2 \
  --upscale-factor 2
```

处理逻辑：
- 视频依次处理（非并行）
- 单个视频失败时记录错误并继续处理下一个
- 所有视频处理完成后统一显示摘要
- 若 `auto_cleanup_temp: false`，完成后统一询问清理

---

## 9. 故障排除

### CUDA out of memory

```
RuntimeError: CUDA out of memory
```

v5 会自动降级批量大小，但若仍失败：

1. 在配置中减小 `tile_size`：
   ```json
   "tile_size": 512   // 或 256
   ```
2. 减小 `batch_size`：
   ```json
   "batch_size": 1
   ```
3. 关闭其他占用 GPU 的程序
4. 禁用 `use_cuda_graph`（会占用固定显存）

### torch.compile 失败

```
torch._dynamo.exc.BackendCompilerFailed
```

降级处理：
```json
"use_compile": false
```
或升级 PyTorch 至最新版。

### NVDEC/NVENC 不可用

```bash
# 验证 FFmpeg NVDEC 支持
ffmpeg -hwaccels 2>&1 | grep cuda

# 若不支持，在配置中禁用
"use_hwaccel": false
```

### 视频分割失败

1. 验证原视频完整性：
   ```bash
   ffprobe input.mp4
   ```
2. 启用自动修复：
   ```json
   "auto_fix_corrupted": true
   ```
3. 或手动修复：
   ```bash
   python src/utils/video_fixer.py input.mp4 -o fixed.mp4
   ```

### 输出视频没有音频

1. 确认源视频有音频轨：
   ```bash
   ffprobe -show_streams input.mp4 | grep codec_type=audio
   ```
2. 检查配置：
   ```json
   "audio_format": "smart"
   ```
3. 确认 FFmpeg 版本支持所需音频编解码器

### 模型加载失败

1. 检查模型路径是否正确：
   ```bash
   ls -lh models_IFRNet/checkpoints/
   ls -lh models_RealESRGAN/
   ```
2. 验证模型文件完整性（文件大小应符合预期）
3. 重新下载模型

---

## 10. 性能优化建议

### 硬件层面

- **GPU**：优先使用 RTX 3080 / 4080 或更高；安培架构（RTX 30xx）FP16 性能提升最明显
- **临时目录**：将 `temp_dir` 设置到 NVMe SSD，减少 I/O 瓶颈
- **内存**：16 GB+ RAM，保证分段缓存充足

### 软件层面

- 启用 `use_fp16: true`（最重要，提升约 1.5–2x）
- 启用 `use_compile: true`（需 PyTorch 2.0+，提升 10–40%）
- 启用 `use_hwaccel: true`（减少 CPU 编解码负担）
- `prefetch_factor: 8`（Real-ESRGAN 数据预取，提升 GPU 利用率）

### 参数层面

| 需求 | 推荐配置 |
|------|---------|
| 最快速度 | IFRNet_S + upscale=2 + FP16 + compile + preset=veryfast |
| 平衡 | IFRNet_S + upscale=4 + FP16 + compile + preset=medium |
| 最高质量 | IFRNet_L + upscale=4 + FP16 + compile + crf=15 + preset=slow |

### 磁盘空间估算

- **临时文件**：原视频大小 × (插帧倍数 + 超分倍数²) × 约 3
- **最终输出**：原视频大小 × 插帧倍数 × 超分倍数² × 约 0.5（取决于 CRF）

---

## 11. 附录

### 支持的视频格式

**输入**：mp4、mkv、avi、mov、flv、webm（及所有 FFmpeg 支持的格式）  
**输出**：mp4（H.264，最兼容）；可配置为 H.265 / AV1

### 处理时间估算（粗略，RTX 3080 参考）

```
插帧时间  ≈  原时长 × (插帧倍数 - 1) × 0.7（分钟）
超分时间  ≈  原时长 × 超分倍数² × 1.0（分钟，FP16）

示例：10 分钟视频，2x 插帧 + 4x 超分
  插帧  = 10 × 1 × 0.7 = 7 分钟
  超分  = 10 × 16 × 1.0 = 160 分钟
  合计  ≈ 167 分钟（含合并约 5 分钟）
```

### 视频信息查询

```bash
# 查看视频基本信息
ffprobe -v quiet -print_format json -show_format -show_streams input.mp4

# 或使用项目工具
python -c "
from src.utils.video_utils import VideoInfo
info = VideoInfo('input.mp4')
print(f'分辨率: {info.width}x{info.height}')
print(f'帧率: {info.fps:.2f} fps')
print(f'时长: {info.duration:.1f} 秒')
print(f'编码: {info.codec}')
print(f'音频: {info.audio_codec if info.has_audio else \"无\"}')
"
```

---

如有问题，请查阅 [README.md](README.md) 或提交 Issue。
