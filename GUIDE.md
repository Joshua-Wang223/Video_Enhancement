# 详细使用指南 — Video Enhancement v2（优化版）

> **主入口**：`src/main_video_optimized.py`  
> **配置真源**：未在命令行覆盖时，默认值以 [`config/default_config.json`](config/default_config.json) 为准（与 README 一致）。
> **最后更新**: 2026-05-14

## 目录

1. [安装与环境配置](#1-安装与环境配置)
2. [配置文件详解](#2-配置文件详解)
3. [命令行用法](#3-命令行用法)
4. [处理流程说明](#4-处理流程说明)
5. [推理与硬件加速参数](#5-推理与硬件加速参数)
6. [模型选择指南](#6-模型选择指南)
7. [断点续传](#7-断点续传)
8. [批量处理](#8-批量处理)
9. [FIX-C 分段归一化](#9-fix-c-分段归一化)
10. [流水线报告 --report](#10-流水线报告---report)
11. [故障排除](#11-故障排除)
12. [性能优化建议](#12-性能优化建议)
13. [附录](#13-附录)

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

务必先安装与 GPU / 驱动匹配的 PyTorch，再安装 `requirements.txt`。详细 wheel 选择与架构对照见 [README.md](README.md)「安装依赖」；常见选项示例：

```bash
# CUDA 12.8（推荐：PyTorch 2.7+，RTX 50xx 等）
pip install "torch>=2.7.0" "torchvision>=0.22.0" \
    --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.1（兼容 RTX 20xx/30xx/40xx）
# pip install "torch>=2.0.0" "torchvision>=0.15.0" \
#     --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8（GTX 10xx Pascal 等）
# pip install "torch>=2.0.0" "torchvision>=0.15.0" \
#     --index-url https://download.pytorch.org/whl/cu118

# CPU（不推荐，仅调试）
# pip install torch torchvision
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

# 或指定自定义根目录（将创建 models_IFRNet/checkpoints、models_RealESRGAN、models_GFPGAN、output、temp、logs、.trt_cache 等）
python setup_project.py --base-dir /path/to/project/root
```

建议在仓库根目录执行；目录布局与 README「项目结构」一致（含 IFRNet / Real-ESRGAN / GFPGAN 模型目录与 `.trt_cache`）。

### 1.6 下载模型文件

#### IFRNet 模型

| 模型文件 | 大小 | 说明 |
|----------|------|------|
| `IFRNet_S_Vimeo90K.pth` | ~10 MB | Small 版，速度快，**推荐日常使用** |
| `IFRNet_L_Vimeo90K.pth` | ~25 MB | Large 版，质量更高，显存需求更大 |

```bash
mkdir -p models_IFRNet/checkpoints

# Small 模型（推荐；保存文件名与默认配置 model_name 一致）
wget -O models_IFRNet/checkpoints/IFRNet_S_Vimeo90K.pth \
  https://github.com/ltkong218/IFRNet/releases/download/v1.0/IFRNet_S.pth

# Large 模型（可选）
wget -O models_IFRNet/checkpoints/IFRNet_L_Vimeo90K.pth \
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

配置文件位于 `config/default_config.json`，是所有处理参数的统一管理入口；**若本文与 JSON 不一致，以 JSON 为准**。命令行参数可覆盖配置中的对应项。

### 2.1 路径配置（`paths`）

```json
{
  "paths": {
    "base_dir": "",
    "input_video": "",
    "input_dir": "",
    "output_dir": "",
    "temp_dir": "",
    "log_dir": "",
    "trt_cache_dir": "",
    "models_gfpgan_dir": ""
  }
}
```

> **说明**：`base_dir` 推荐留空，由 `config_manager` 根据配置文件位置自动推算项目根。`output_dir` / `temp_dir` / `log_dir` / `trt_cache_dir` / `models_gfpgan_dir` 留空时分别派生为 `base_dir/output`、`base_dir/temp`、`base_dir/logs`、`base_dir/.trt_cache`、`base_dir/models_GFPGAN`。TensorRT Engine 缓存在 `trt_cache_dir`（默认 `.trt_cache`），IFRNet 与 Real-ESRGAN 共用该目录。

### 2.2 处理配置（`processing`）

```json
{
  "processing": {
    "mode": "interpolate_then_upscale",
    "interpolation_factor": 2,
    "upscale_factor": 2,
    "segment_duration": 30,
    "auto_fix_corrupted": false,
    "auto_cleanup_temp": true,
    "batch_mode": false
  }
}
```

`batch_mode: true` 时从 `paths.input_dir` 批量读取视频；单文件模式使用 `paths.input_video` 或命令行 `-i`。

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

后端脚本：`external/IFRNet/process_video_v6_3_3_single.py`（IFRNet **v6.3.3**）。默认使用 `model_name` 自动拼接 `base_dir/models_IFRNet/checkpoints/<name>.pth`，也可用 `model_path` 指定绝对路径。

```json
{
  "models": {
    "ifrnet": {
      "model_name": "IFRNet_S_Vimeo90K",
      "model_path": "",
      "use_gpu": true,
      "batch_size": 24,
      "max_batch_size": 36,

      "use_fp16": true,
      "use_compile": false,
      "use_cuda_graph": false,
      "use_tensorrt": true,

      "use_hwaccel": true,
      "codec": "libx264",
      "crf": 23,
      "x264_preset": "medium",
      "keep_audio": true,
      "ffmpeg_bin": "ffmpeg",

      "report_json": null
    }
  }
}
```

### 2.4 Real-ESRGAN 模型配置（`models.realesrgan`）

优化版处理器调用 **`external/realesrgan_video`**（深度流水线 `main_optimized`）。`model_name` 为不含后缀的名称（与底层脚本一致）。

```json
{
  "models": {
    "realesrgan": {
      "model_name": "realesr-general-x4v3",
      "model_path": "",
      "use_gpu": true,
      "denoise_strength": 0.5,
      "tile_size": 0,
      "tile_pad": 10,
      "pre_pad": 0,
      "use_fp16": true,
      "face_enhance": false,

      "batch_size": 24,
      "prefetch_factor": 96,
      "use_compile": true,
      "use_cuda_graph": true,
      "use_tensorrt": false,
      "gfpgan_trt": false,

      "use_hwaccel": true,
      "video_codec": "libx264",
      "codec": "libx264",
      "x264_preset": "medium",
      "crf": 23,
      "ffmpeg_bin": "ffmpeg",

      "preview": false,
      "report_json": null
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
    "codec": "libx264",
    "preset": "medium",
    "crf": 23,
    "pix_fmt": "yuv420p",
    "audio_format": "smart",
    "audio_codec": "copy",
    "audio_bitrate": "192k"
  }
}
```

**copy-by-default 策略**：未在 CLI 指定 `--output-codec/crf/preset` 时，最终合并阶段自动使用 `-c:v copy`（不重新编码），继承第二阶段编码器的输出质量。若任一输出参数在 CLI 中指定，则触发重新编码。

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

入口脚本：**`src/main_video_optimized.py`**。完整参数请执行：

```bash
python src/main_video_optimized.py --help
```

要点：**单文件模式**需同时指定 `-i` 与 `-o`（输出为**带文件名的路径**）；**批量模式**使用 `--batch-mode`，并指定 `--input-dir` 与 `--output-dir`。配置文件默认 `config/default_config.json`。

### 3.1 常用参数（摘录）

| 参数 | 说明 |
|------|------|
| `-c`, `--config` | 配置文件路径 |
| `-i`, `--input` | 输入视频（单文件模式） |
| `-o`, `--output` | 输出视频路径，须含文件名（单文件模式） |
| `--batch-mode` | 启用批量模式 |
| `--input-dir` | 批量输入目录 |
| `--output-dir` | 批量输出目录 |
| `-m`, `--mode` | `interpolate_then_upscale` / `upscale_then_interpolate` |
| `--interpolation-factor` | 2 / 4 / 8 / 16 |
| `--upscale-factor` | 2 / 4 |
| `--trt-cache-dir` | TRT Engine 缓存目录（与配置 `paths.trt_cache_dir` 一致） |
| `--dry-run` | 仅打印配置与环境，不执行处理 |
| `--report PATH` | 输出端到端流水线 JSON 报告 |
| `--skip-seg-normalize` | 跳过合并前分段 timescale 归一化（调试用） |

IFRNet / Real-ESRGAN 细粒度开关（如 `--use-tensorrt-ifrnet`、`--batch-size-esrgan` 等）见 `--help` 分组说明。

### 3.2 常用示例

```bash
# 单视频（指定配置文件与输出文件路径）
python src/main_video_optimized.py -c config/default_config.json -i input.mp4 -o output/enhanced.mp4

# 处理模式
python src/main_video_optimized.py -c config/default_config.json -i input.mp4 -o output/out.mp4 \
  -m upscale_then_interpolate

# 插帧 / 超分倍数
python src/main_video_optimized.py -c config/default_config.json -i input.mp4 -o output/out.mp4 \
  --interpolation-factor 4 --upscale-factor 4

# 流水线报告
python src/main_video_optimized.py -c config/default_config.json -i input.mp4 -o output/out.mp4 \
  --report pipeline_report.json

# 批量处理
python src/main_video_optimized.py -c config/default_config.json \
  --batch-mode --input-dir ./videos --output-dir ./enhanced

# 自定义配置文件
python src/main_video_optimized.py -c my_config.json -i input.mp4 -o output/out.mp4
```

### 3.3 典型场景

#### 场景 A：老电影修复（720p 24fps → 高清 48fps）
```bash
python src/main_video_optimized.py -c config/default_config.json -i old_movie.mp4 -o output/old_movie_out.mp4 \
  -m upscale_then_interpolate \
  --upscale-factor 4 \
  --interpolation-factor 2
```

#### 场景 B：游戏录屏增强（1080p 30fps → 流畅高质量）
```bash
python src/main_video_optimized.py -c config/default_config.json -i gameplay.mp4 -o output/gameplay_out.mp4 \
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
      "model_name": "RealESRGANv2-animevideo-xsx2"
    }
  }
}
```

然后运行：

```bash
python src/main_video_optimized.py -c config/default_config.json -i anime.mp4 -o output/anime_out.mp4 --upscale-factor 2
```

#### 场景 D：批量处理家庭视频
```bash
python src/main_video_optimized.py -c config/default_config.json \
  --batch-mode \
  --input-dir ./family_videos \
  --output-dir ./family_enhanced \
  --interpolation-factor 2 \
  --upscale-factor 2
```

处理完成后按配置 `auto_cleanup_temp` 自动清理或询问清理临时文件。

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
   │   ├─ [直接对接，无需合并] ──────────────── 分段直连优化
   │   ├─ Real-ESRGAN 超分处理每个分段
   │   │   └─ 断点保存 checkpoint.json
   │   ├─ [FIX-C] timescale 归一化（H.264/H.265）
   │   │   └─ --skip-seg-normalize 可跳过
   │   └─ merge_videos_by_codec 合并所有分段
   │
   ├─ [模式 2: upscale_then_interpolate]
   │   ├─ 分割为 N 个分段
   │   ├─ Real-ESRGAN 超分处理每个分段
   │   ├─ [直接对接]
   │   ├─ IFRNet 插帧处理每个分段
   │   ├─ [FIX-C] timescale 归一化
   │   └─ 合并所有分段
   │
   ├─ 添加音频（add_audio_to_video）
   │
   └─ 输出最终视频
```

### 4.2 分段直连流水线（优化版核心）

v1 的处理方式：
```
分段 → 插帧每段 → 合并 → 再分段 → 超分每段 → 最终合并
                 ^^^^^^^^^^^^^^^^^^^
                 这两步在当前优化版中被省去
```

优化版的处理方式：
```
分段 → 插帧每段 → [直接传递分段列表] → 超分每段 → [FIX-C 归一化] → 最终合并
```

流水线通过 `process_segments_directly(segment_list)` 等接口实现分段直接对接，省去中间合并与二次分段，显著降低磁盘 I/O 与总耗时（参见 README 特性说明）。

### 4.3 FIX-C 合并后验证

最终合并输出后增加 ffprobe 主动验证：
- 验证视频流存在且 duration > 0
- 验证音频流存在（当提供外部音频时）
- 及早发现并报告静默损坏（rc=0 但无视频帧的合并产物）

---

## 5. 推理与硬件加速参数

### 5.1 FP16 推理（`use_fp16`）

将模型权重和中间张量从 FP32 降为 FP16：
- 显存节省约 50%
- 推理速度提升约 1.5–2x（依 GPU 架构而定）
- 精度损失极小，肉眼不可见

适用范围：IFRNet 与 Real-ESRGAN 均在配置中以 `use_fp16` 控制（CLI 对应 `--no-fp16-ifrnet` / `--no-fp16-esrgan` 等）。

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
- 优化版内置 GPU 硬件型号表（20+ 型号），自动选择最优编码器

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
python src/main_video_optimized.py -c config/default_config.json \
  --batch-mode \
  --input-dir /data/input_videos \
  --output-dir /data/output_videos \
  --interpolation-factor 2 \
  --upscale-factor 2
```

处理逻辑：
- 视频依次处理（非并行）
- 单个视频失败时记录错误并继续处理下一个
- 所有视频处理完成后统一显示摘要
- 若 `auto_cleanup_temp: false`，完成后统一询问清理

批量模式下也可使用 `--report` 输出流水线报告，每个视频的 `extra` 字段包含批量索引。

---

## 9. FIX-C 分段归一化

### 问题背景

多个独立 ffmpeg 子进程编码的分段（即使编解码参数完全相同）在 MP4 timescale 上存在微差。这些分段通过 `concat` demuxer 合并时，demuxer 遇到 timescale 跳变会触发 `AVEROR_EXIT(254)`，合并失败。

### 解决方案

`_normalize_segs_for_copy()` 函数在最终合并前对分段进行零损耗 remux 归一化：

1. **自动检测编解码器：** 仅对 H.264 / H.265 分段执行（VP9 / AV1 / ProRes 等自动跳过）
2. **统一时间基：** 使用 `-video_track_timescale 90000`（90 kHz，MP4 标准）
3. **三重校验：** 每个分段 remux 后验证（rc=0 + 文件 > 4KB + ffprobe 确认视频帧 duration > 0）
4. **原地替换：** 成功则覆盖原分段文件，失败保留原始文件

### 与旧方案的区别

> **不再使用 `-bsf:v dump_extra`。**  
> 该 BSF 专为 Annex B 格式（MPEG-TS / 裸流）设计。在 MP4/AVCC 格式上应用 `dump_extra` 会产生畸形 packet，导致：
> - ffmpeg rc=0 但视频流被静默丢弃  
> - 合并输出文件"只有音频"  
> - 这是早期版本此类问题的根本原因

### CLI 控制

```bash
# 默认：FIX-C 归一化自动执行
python src/main_video_optimized.py -i input.mp4 -o output.mp4

# 跳过归一化（调试用，或已确认分段 timescale 一致）
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --skip-seg-normalize
```

### 合并后验证

`merge_videos_by_codec()` 在 ffmpeg rc=0 后增加主动 ffprobe 验证：
- 验证输出文件包含有效视频流（codec_type=video 且 duration > 0）
- 验证外部音频流已正确混入
- 及早发现 rc=0 但视频流丢失的情况

---

## 10. 流水线报告（--report）

### 功能

`--report` 参数输出端到端流水线 JSON 报告，成功或失败均会写出。

### 使用方式

```bash
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --report /path/to/report.json
```

### 报告内容

```json
{
  "version":             "2.0.0",
  "timestamp":           "2026-05-14T12:00:00",
  "success":             true,
  "input":               "/path/to/input.mp4",
  "output":              "/path/to/output.mp4",
  "mode":                "interpolate_then_upscale",
  "elapsed_seconds":     1234.56,
  "elapsed_human":       "20分34秒",
  "input_size_mb":       150.5,
  "output_size_mb":      890.2,
  "size_ratio":          5.914,
  "input_duration_s":    600.0,
  "output_duration_s":   600.5,
  "gpu_peak_memory_gb":  7.824,
  "env": {
    "python":            "3.10.12",
    "torch":             "2.7.0+cu128",
    "gpu_name":          "NVIDIA RTX 3080",
    "gpu_memory_gb":     10.0,
    "ffmpeg":            true
  },
  "params": { /* 关键 CLI 参数快照 */ }
}
```

### 批量模式

批量模式下，`--report` 会为每个视频写入独立的 `extra.batch_index` 字段。也可以为每个单独的视频写报告。

### 与处理器级报告的区别

| 报告层级 | CLI 参数 | 内容范围 |
|---------|---------|---------|
| 流水线报告 | `--report` | 端到端（耗时/大小/GPU峰值/参数快照） |
| IFRNet 报告 | `--report-ifrnet` | IFRNet 推理延迟/硬件统计 |
| ESRGan 报告 | `--report-esrgan` | ESRGan 推理延迟/GFPGAN 统计 |

---

## 11. 故障排除

### CUDA out of memory

```
RuntimeError: CUDA out of memory
```

处理器会自动降级批量大小，但若仍失败：

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

### "合并后只有音频无视频"

这是旧版 `-bsf:v dump_extra` 问题的特征。优化版已通过 FIX-C 机制修复：

- 改用 `-video_track_timescale 90000` 统一时间基
- 合并后增加 ffprobe 视频流验证
- 如果仍遇到，使用 `--skip-seg-normalize` 跳过归一化并报告

### 模型加载失败

1. 检查模型路径是否正确：
   ```bash
   ls -lh models_IFRNet/checkpoints/
   ls -lh models_RealESRGAN/
   ```
2. 验证模型文件完整性（文件大小应符合预期）
3. 重新下载模型

---

## 12. 性能优化建议

### 硬件层面

- **GPU**：优先使用 RTX 3080 / 4080 或更高；安培架构（RTX 30xx）FP16 性能提升最明显
- **临时目录**：将 `temp_dir` 设置到 NVMe SSD，减少 I/O 瓶颈
- **内存**：16 GB+ RAM，保证分段缓存充足

### 软件层面

- 启用 `use_fp16: true`（最重要，提升约 1.5–2x）
- 启用 `use_compile: true`（需 PyTorch 2.0+，提升 10–40%）
- 启用 `use_hwaccel: true`（减少 CPU 编解码负担）
- `prefetch_factor` 默认见 `default_config.json`（Real-ESRGAN / `realesrgan_video` 读帧预取深度）

### 参数层面

| 需求 | 推荐配置 |
|------|---------|
| 最快速度 | IFRNet_S + upscale=2 + FP16 + compile + preset=veryfast |
| 平衡 | IFRNet_S + upscale=4 + FP16 + compile + preset=medium |
| 最高质量 | IFRNet_L + upscale=4 + FP16 + compile + crf=15 + preset=slow |

### FIX-C 性能影响

FIX-C 归一化对每个 H.264/H.265 分段执行零损耗 remux（约 1-3 秒/分段），分段数量较多时总体时间可忽略。若确认分段 timescale 已一致，可用 `--skip-seg-normalize` 跳过。

### 磁盘空间估算

- **临时文件**：原视频大小 × (插帧倍数 + 超分倍数²) × 约 3
- **最终输出**：原视频大小 × 插帧倍数 × 超分倍数² × 约 0.5（取决于 CRF）

---

## 13. 附录

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

# 或使用项目工具（需在仓库根目录执行，且保证 src/utils 在 PYTHONPATH；或直接参考 video_utils 中的用法）
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src/utils').resolve()))
from video_utils import VideoInfo
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
