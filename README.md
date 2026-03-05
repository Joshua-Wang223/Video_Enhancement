# Video Enhancement — 视频增强处理系统

> **当前版本**: v5（单卡优化版）｜IFRNet 后端 v5 / Real-ESRGAN 后端 v6  
> **最后更新**: 2026-03-05

一套完整的 AI 视频增强解决方案，整合了**视频插帧（IFRNet）**与**视频超分辨率（Real-ESRGAN）**两大模块，并针对单 GPU 环境进行了深度优化。

---

## 功能特性

| 功能 | 说明 |
|------|------|
| 🎬 **视频插帧** | 基于 IFRNet 光流估计，支持 2x / 4x / 8x / 16x 帧率提升 |
| 🖼️ **视频超分** | 基于 Real-ESRGAN，支持 2x / 4x 分辨率放大 |
| ⚡ **FP16 推理** | 半精度推理，显存减半，速度提升约 1.5–2x |
| 🔧 **torch.compile** | PyTorch 2.0+ 图编译加速 |
| 🚀 **CUDA Graph** | IFRNet 支持 CUDA Graph 固化推理图，降低 CPU 调度开销 |
| 🎥 **硬件编解码** | 自动探测并启用 NVDEC 解码 / NVENC 编码 |
| 🔁 **TensorRT 加速** | 可选 TensorRT 引擎加速（需额外安装） |
| 🛡️ **OOM 自动降级** | 显存不足时自动减小批量大小，无需手动干预 |
| 🔗 **分段直连流水线** | 插帧→超分分段直接对接，省去中间合并步骤，减少 30–50% I/O |
| 💾 **断点续传** | 分段级别断点保存，处理中断后从断点继续 |
| 👤 **人脸增强（GFPGAN）** | 批量推理 + 原始帧检测 + CPU-GPU 流水线，face_enhance 速度显著提升（v6） |
| 🎵 **智能音频处理** | 自动检测音频编码，无损时直接 copy，否则转码 |
| 📦 **批量处理** | 支持目录批量处理，失败自动跳过继续 |
| 🧹 **灵活清理** | 处理完成后询问或自动清理临时文件 |

---

## 项目结构

```
Video_Enhancement/
├── README.md                       # 本文件
├── GUIDE.md                        # 详细使用指南
├── PROJECT_SUMMARY.md              # 架构与技术总结
├── FILE_LIST.md                    # 文件清单与模块说明
├── QUICK_UPGRADE.md                # 版本升级说明
│
├── requirements.txt                # Python 依赖
├── fix_import.py                   # 导入路径修复工具
├── run.py                          # 旧版快速入口（v1，向后兼容）
│
├── config/
│   └── default_config.json         # 主配置文件（v5 参数完整版）
│
├── src/
│   ├── main_video_v5_single.py     # ★ 当前主入口（v5 单卡版）
│   ├── main_video_v3.py            # 历史版本（保留参考）
│   ├── main_v2.py                  # 历史版本（保留参考）
│   ├── main.py                     # 历史版本 v1
│   │
│   ├── processors/
│   │   ├── ifrnet_processor_v5_single.py           # ★ IFRNet 处理器 v5（可独立调用）
│   │   ├── realesrgan_processor_video_v5_single.py # ★ Real-ESRGAN 处理器 v5（可独立调用）
│   │   └── ...                                     # 历史版本处理器
│   │
│   └── utils/
│       ├── config_manager.py       # 配置管理
│       ├── video_utils.py          # 视频工具（分割、合并、音频、编解码）
│       ├── video_fixer.py          # 损坏视频修复
│       └── output_filter.py        # FFmpeg 输出过滤
│
├── external/
│   ├── IFRNet/
│   │   ├── process_video_v5_single.py  # ★ IFRNet v5 后端核心
│   │   └── models/                     # IFRNet 模型定义
│   └── Real-ESRGAN/
│       ├── inference_realesrgan_video_v6_single.py  # ★ ESRGAN v6 后端核心（当前）
│       └── realesrgan/                              # Real-ESRGAN 库
│
├── models_IFRNet/
│   └── checkpoints/
│       ├── IFRNet_S_Vimeo90K.pth   # IFRNet Small（推荐，速度快）
│       └── IFRNet_L_Vimeo90K.pth   # IFRNet Large（质量更高）
│
├── models_RealESRGAN/
│   ├── realesr-general-x4v3.pth        # 通用 4x 超分（推荐）
│   ├── RealESRGAN_x4plus.pth           # 写实 4x 超分
│   ├── RealESRGAN_x2plus.pth           # 写实 2x 超分
│   ├── RealESRGANv2-animevideo-xsx2.pth # 动漫 2x 超分
│   └── GFPGANv1.4.pth                  # 人脸增强模型（face_enhance 时必需）
│
├── output/                         # 处理结果输出目录
├── temp/                           # 临时目录（自动管理）
└── logs/                           # 日志目录
```

---

## 快速开始

### 1. 系统要求

**最低配置**
- Python 3.9+
- NVIDIA GPU（6 GB VRAM）、CUDA 11.8 或 12.x  
  （GTX 10xx / Pascal sm_61 仅支持 CUDA 11.8，不支持 CUDA 12.8）
- FFmpeg 5.0+
- 16 GB RAM、100 GB 可用磁盘空间

**推荐配置**
- NVIDIA RTX 3080 / 4080 或更高（10 GB+ VRAM）
- CUDA 12.8+、PyTorch 2.7+（Blackwell/RTX 50xx 原生支持；启用 torch.compile）
- 驱动 ≥ 560.x（CUDA 12.8 要求），GPU 架构 ≥ Turing（sm_75+）
- 32 GB RAM、200 GB+ SSD

### 2. 安装依赖

```bash
# ── 步骤一：安装 PyTorch（根据 GPU 架构 / 驱动选择 wheel）──────────────────

# ★ 推荐：CUDA 12.8（PyTorch 2.7+，RTX 50xx Blackwell 原生支持）
#   要求：驱动 ≥ 560.x，GPU 架构 ≥ Turing（RTX 20xx / GTX 16xx，sm_75+）
pip install "torch>=2.7.0" "torchvision>=0.22.0" \
    --index-url https://download.pytorch.org/whl/cu128

# 向后兼容：CUDA 12.1（RTX 20xx/30xx/40xx，驱动 ≥ 525.x）
# pip install "torch>=2.0.0" "torchvision>=0.15.0" \
#     --index-url https://download.pytorch.org/whl/cu121

# 向后兼容：CUDA 11.8（GTX 10xx Pascal 及更早，驱动 ≥ 450.x）
# ⚠️  GTX 10xx（sm_61）不支持 CUDA 12.8，必须用此选项
# pip install "torch>=2.0.0" "torchvision>=0.15.0" \
#     --index-url https://download.pytorch.org/whl/cu118

# GPU 架构速查：
#   RTX 50xx (sm_120) → cu128（必须）
#   RTX 40xx (sm_89)  → cu128 或 cu121
#   RTX 30xx (sm_86)  → cu128 或 cu121
#   RTX 20xx / GTX 16xx (sm_75) → cu128 或 cu121
#   GTX 10xx (sm_61)  → cu118（cu128 不支持此架构）

# ── 步骤二：安装项目依赖 ──────────────────────────────────────────────────────
pip install -r requirements.txt

# ── 步骤三：安装系统 FFmpeg（Ubuntu/Debian）──────────────────────────────────
sudo apt-get install -y ffmpeg
```

### 3. 下载模型

```bash
mkdir -p models_IFRNet/checkpoints models_RealESRGAN

# IFRNet Small（推荐，速度快）
wget -O models_IFRNet/checkpoints/IFRNet_S_Vimeo90K.pth \
  https://github.com/ltkong218/IFRNet/releases/download/v1.0/IFRNet_S.pth

# Real-ESRGAN 通用 4x（推荐）
wget -O models_RealESRGAN/realesr-general-x4v3.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth

# GFPGAN 人脸增强模型（仅 face_enhance=true 时需要）
wget -O models_RealESRGAN/GFPGANv1.4.pth \
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
```

其他可选模型见 [FILE_LIST.md](FILE_LIST.md)。

### 4. 配置路径

`config/default_config.json` 中 `base_dir` 留空时，`config_manager` 会自动从配置文件位置向上推算项目根目录（标准部署无需手动填写）。非标准部署或路径不正确时，手动指定：

```json
{
  "paths": {
    "base_dir":   "/your/project/path",
    "output_dir": "/your/project/path/output",
    "temp_dir":   "/your/project/path/temp",
    "log_dir":    "/your/project/path/logs"
  },
  "models": {
    "ifrnet": {
      "model_name": "IFRNet_S_Vimeo90K"
    }
  }
}
```

> **说明：** `models.ifrnet.model_name` 是推荐方式，处理器会自动在 `models_IFRNet/checkpoints/<model_name>.pth` 下查找权重文件。若需指定任意路径，改用 `model_path`（优先级更高）：
> ```json
> "model_path": "/absolute/path/to/IFRNet_S_Vimeo90K.pth"
> ```

### 5. 运行

```bash
# 单视频处理（先插帧再超分，默认）
python src/main_video_v5_single.py -i input.mp4

# 先超分再插帧
python src/main_video_v5_single.py -i input.mp4 -m upscale_then_interpolate

# 自定义倍数（4x 插帧 + 4x 超分）
python src/main_video_v5_single.py -i input.mp4 \
  --interpolation-factor 4 --upscale-factor 4

# 指定 IFRNet 模型（L 版质量更高，速度更慢）
python src/main_video_v5_single.py -i input.mp4 \
  --ifrnet-model IFRNet_L_Vimeo90K

# 直接指定 IFRNet .pth 路径
python src/main_video_v5_single.py -i input.mp4 \
  --ifrnet-model-path /workspace/models/IFRNet_S_Vimeo90K.pth

# 调整批大小（根据显存调优）
python src/main_video_v5_single.py -i input.mp4 \
  --ifrnet-batch-size 8 --esrgan-batch-size 16

# 开启人脸增强（需预先下载 GFPGANv1.4.pth）
python src/main_video_v5_single.py -i input.mp4 --face-enhance

# 人脸增强精细控制（GFPGAN 版本、融合权重、防 OOM 批大小）
python src/main_video_v5_single.py -i input.mp4 --face-enhance \
  --gfpgan-model 1.4 --gfpgan-weight 0.7 --gfpgan-batch-size 4

# 批量处理
python src/main_video_v5_single.py --input-dir ./videos --batch

# 指定输出目录 + 自动清理临时文件
python src/main_video_v5_single.py -i input.mp4 -o ./my_output --auto-cleanup
```

---

## 处理模式

### 模式 1：先插帧再超分（`interpolate_then_upscale`，默认）

```
输入视频 → 分段 → IFRNet 插帧 ──→（直连）──→ Real-ESRGAN 超分 → 合并 → 输出
```

适用于**低帧率**视频（<30 fps），优先提升流畅度。插帧在原始分辨率下进行，运算量更小。

### 模式 2：先超分再插帧（`upscale_then_interpolate`）

```
输入视频 → 分段 → Real-ESRGAN 超分 ──→（直连）──→ IFRNet 插帧 → 合并 → 输出
```

适用于**低分辨率**视频（<1080p），优先提升画质。插帧基于高分辨率帧，光流更精确。

---

## 处理示例

### 示例一：低分辨率真人视频——先超分再插帧 + 人脸增强

**场景**：720p 真人视频，30fps，需要提升分辨率至 1440p 并开启人脸精修。  
**策略**：先超分（`upscale_then_interpolate`），画质优先；GFPGAN 1.4 融合权重 0.7，保留真实感的同时修复人脸细节；分段 10 秒便于显存管理。

```bash
python src/main_video_v5_single.py \
    -c /workspace/Video_Enhancement/config/default_config.json \
    -m upscale_then_interpolate \
    -i /workspace/input_videos/test1.mp4 \
    --interpolation-factor 2 \
    --upscale-factor 2 \
    --esrgan-model realesr-general-x4v3 \
    --segment-duration 10 \
    --face-enhance \
    --gfpgan-model 1.4 \
    --gfpgan-weight 0.7 \
    --gfpgan-batch-size 12
```

**处理流程**：
```
test1.mp4 (1280×720, 30fps)
  → 分段（10 秒/段）
  → Real-ESRGAN 超分 2x  →  2560×1440
  → IFRNet 插帧 2x       →  60fps
  → 合并 + 音频回填
  → test1_processed.mp4 (2560×1440, 60fps)
```

**关键参数说明**：
- `--gfpgan-weight 0.7`：融合权重偏向 GFPGAN 输出，人脸修复效果更明显，同时保留约 30% 原始纹理。
- `--gfpgan-batch-size 12`：单次前向处理 12 张人脸，T4/16G 显存可承受；OOM 时自动降级并持久化。
- `--segment-duration 10`：较短分段，减少单段显存峰值，适合高分辨率或人脸密集场景。

---

### 示例二：低帧率动画/录屏视频——先插帧再超分（轻量模式）

**场景**：720p 低帧率视频（如录屏、动画），需要提升流畅度，无需人脸增强。  
**策略**：先插帧（`interpolate_then_upscale`，默认），帧率优先；使用默认配置，参数最简。

```bash
python src/main_video_v5_single.py \
    -c /workspace/Video_Enhancement/config/default_config.json \
    -i /workspace/input_videos/word_world_2.mp4 \
    -m interpolate_then_upscale \
    --interpolation-factor 2 \
    --upscale-factor 2
```

**处理流程**：
```
word_world_2.mp4 (原始分辨率, 低帧率)
  → 分段（30 秒/段，默认）
  → IFRNet 插帧 2x       →  帧率翻倍
  → Real-ESRGAN 超分 2x  →  分辨率翻倍
  → 合并 + 音频回填
  → word_world_2_processed.mp4
```

**关键参数说明**：
- 未指定 `--ifrnet-model`，使用配置文件中 `model_name: IFRNet_S_Vimeo90K`（轻量快速）。
- 未指定 `--esrgan-model`，使用配置文件中 `model_name: realesr-general-x4v3`（通用超分）。
- 未指定 `--segment-duration`，使用默认 30 秒，I/O 次数更少，适合普通场景。
- 两个批大小均沿用配置文件默认值，无需手动调整。

---

## 命令行参数速查

```
python src/main_video_v5_single.py [选项]

输入 / 输出:
  -i, --input PATH              输入视频路径（单视频）
  --input-dir DIR               输入视频目录（自动启用批量模式）
  -o, --output-dir DIR          输出目录（覆盖配置文件）
  -c, --config PATH             配置文件（默认: config/default_config.json）
  -b, --batch                   启用批量处理模式

处理控制:
  -m, --mode MODE               interpolate_then_upscale（默认）
                                upscale_then_interpolate
  --interpolation-factor N      插帧倍数：2 / 4 / 8 / 16（默认 2）
  --upscale-factor N            超分倍数：2 / 4（默认 2）
  --segment-duration SECS       分段时长（秒，默认 30）

IFRNet 模型:
  --ifrnet-model MODEL_NAME     IFRNet 模型名称（覆盖配置）
                                  IFRNet_S_Vimeo90K（默认，轻量快速）
                                  IFRNet_L_Vimeo90K（高质量，速度更慢）
  --ifrnet-model-path PATH      IFRNet .pth 绝对路径（优先级高于 --ifrnet-model）
  --ifrnet-batch-size N         IFRNet 推理批大小（覆盖配置，默认 4）

Real-ESRGAN 模型:
  --esrgan-model MODEL_NAME     模型名称，如 realesr-general-x4v3（覆盖配置）
  --tile-size N                 tile 切块大小（0=不切块；显存不足时设 512）
  --esrgan-batch-size N         ESRGAN 推理批大小（覆盖配置，T4/16G 建议 8~12）

人脸增强（GFPGAN）:
  --face-enhance                开启人脸增强（需 GFPGANv1.4.pth）
  --no-face-enhance             关闭人脸增强（覆盖配置中 face_enhance=true）
  --gfpgan-model VER            GFPGAN 版本：1.3 / 1.4 / RestoreFormer（默认 1.4）
  --gfpgan-weight W             融合权重 0.0~1.0（0=不增强，1=完全替换，默认 0.5）
  --gfpgan-batch-size N         单次 GFPGAN 最多处理人脸数，防 OOM（默认 8）

硬件加速 / 精度:
  --no-hwaccel                  禁用 NVDEC 硬件解码（IFRNet + ESRGAN 同时生效）
  --no-fp16                     禁用 FP16，改用 FP32（IFRNet）
  --no-compile                  禁用 torch.compile（短视频可跳过预热）
  --no-cuda-graph               禁用 CUDA Graph（IFRNet）

其他:
  --auto-cleanup                处理完成后自动清理临时文件，不再询问
```

---

## 性能参考

> 测试环境：RTX 3080（10 GB VRAM），1080p 输入，v5 单卡版，FP16 开启

| 操作 | 设置 | 10 分钟视频耗时 |
|------|------|--------------|
| 插帧 | 2x IFRNet_S | ~7 分钟 |
| 插帧 | 4x IFRNet_S | ~14 分钟 |
| 超分 | 2x | ~50 分钟 |
| 超分 | 4x | ~125 分钟 |
| **插帧 2x + 超分 4x（直连流水线）** | **推荐配置** | **~132 分钟** |

v5 直连流水线相比 v1 节省约 30% 总时间和 38% 磁盘 I/O。

---

## 配置速查

以下为 `config/default_config.json` 关键参数，可按需修改或通过命令行参数覆盖：

```json
{
  "processing": {
    "mode": "interpolate_then_upscale",
    "interpolation_factor": 2,
    "upscale_factor": 2,
    "segment_duration": 30,
    "auto_cleanup_temp": true
  },
  "models": {
    "ifrnet": {
      "model_name": "IFRNet_S_Vimeo90K",
      "model_path": "",
      "use_fp16": true,
      "use_compile": true,
      "use_cuda_graph": true,
      "use_hwaccel": true,
      "batch_size": 4,
      "crf": 23
    },
    "realesrgan": {
      "model_name": "realesr-general-x4v3",
      "tile_size": 0,
      "batch_size": 8,
      "prefetch_factor": 16,
      "use_compile": false,
      "use_hwaccel": true,
      "crf": 23,
      "face_enhance": false,
      "gfpgan_model": "1.4",
      "gfpgan_weight": 0.5,
      "gfpgan_batch_size": 8
    }
  }
}
```

> **关键参数说明：**
> - `ifrnet.model_name`：推荐方式，处理器自动拼接 `models_IFRNet/checkpoints/<model_name>.pth`；填写 `model_path`（绝对路径）可跳过自动拼接，优先级更高。
> - ESRGAN `batch_size=8`，可充分利用 T4/RTX 3080 等 16 GB 显存；`prefetch_factor` 建议 ≥ `batch_size × 2`。
> - ESRGAN `use_compile` 默认关闭（首次编译需 1–3 分钟），适合长批量任务时开启。
> - `face_enhance` 对 anime 系模型（`realesr-animevideov3` 等）无效，底层脚本会自动禁用。
> - `gfpgan_batch_size` 控制单次前向处理的人脸数；人脸密集场景可调小至 4 防 OOM，降级值自动持久化。

完整配置说明见 [GUIDE.md](GUIDE.md)。

---

## 独立调用处理器

两个处理器均可脱离主流程独立运行，方便单独调试某一模块：

```bash
# ── IFRNet 独立插帧（对接 external/IFRNet/process_video_v5_single.py）──────
# 基础 2x 插帧（使用配置中 model_name 自动拼路径）
python src/processors/ifrnet_processor_v5_single.py -i input.mp4 -o output_2x.mp4

# 指定 IFRNet 模型名称
python src/processors/ifrnet_processor_v5_single.py \
  -i input.mp4 -o output_2x.mp4 --ifrnet-model IFRNet_L_Vimeo90K

# 直接指定 .pth 路径（优先级高于 --ifrnet-model）
python src/processors/ifrnet_processor_v5_single.py \
  -i input.mp4 -o output_2x.mp4 \
  --ifrnet-model-path /workspace/models/IFRNet_S_Vimeo90K.pth

# 4x 插帧，关闭 compile（短视频跳过 1–3 分钟预热）
python src/processors/ifrnet_processor_v5_single.py \
  -i input.mp4 -o output_4x.mp4 \
  --interpolation-factor 4 --no-compile --no-cuda-graph

# ── Real-ESRGAN 独立超分（对接 external/Real-ESRGAN/inference_realesrgan_video_v6_single.py）
# 基础 4x 超分
python src/processors/realesrgan_processor_video_v5_single.py \
  -i input.mp4 -o output_4x.mp4 --upscale-factor 4

# 开启人脸增强，精细控制 GFPGAN
python src/processors/realesrgan_processor_video_v5_single.py \
  -i input.mp4 -o output.mp4 --face-enhance \
  --gfpgan-model 1.4 --gfpgan-weight 0.5 --gfpgan-batch-size 4
```

---

## 常见问题

**Q: CUDA out of memory？**  
A: v5 支持 OOM 自动降级，会自动减小批量大小并持久化。如仍失败，在配置中设置 `"tile_size": 512` 并将 `"batch_size"` 调小至 1–2。

**Q: 启动时报 `FileNotFoundError: No such file or directory: ''`（IFRNet 模型路径为空）？**  
A: 这是模型文件未找到的明确报错。处理器会在加载前校验路径并打印完整提示，请按提示操作：
- 确认权重文件已下载并放置到 `models_IFRNet/checkpoints/IFRNet_S_Vimeo90K.pth`，或
- 命令行传入 `--ifrnet-model IFRNet_S_Vimeo90K`（自动拼接路径），或
- 命令行传入 `--ifrnet-model-path /绝对/路径/model.pth` 直接指定。

**Q: 如何从中断恢复？**  
A: 重新运行完全相同的命令，程序自动读取 `temp/` 下的 `checkpoint.json` 从断点继续。

**Q: 输出视频没有音频？**  
A: v5 使用智能音频提取（`audio_format: "smart"`），无损时直接 copy，否则转码。请确认源视频含音频且 FFmpeg 已正确安装。

**Q: torch.compile 报错？**  
A: 需要 PyTorch 2.0+，首次编译约需 1–3 分钟。如遇兼容性问题，传入 `--no-compile` 或在配置中设置 `"use_compile": false`。

**Q: 临时文件占用空间太大？**  
A: 传入 `--auto-cleanup` 或设置 `"auto_cleanup_temp": true`，处理完成后自动清理。临时文件通常为原视频大小的 10–20 倍。

**Q: 使用 face_enhance 需要额外下载什么？**  
A: 需要 GFPGAN 模型文件，放置到 `models_RealESRGAN/` 目录。默认使用 `GFPGANv1.4.pth`，也可通过 `--gfpgan-model` 切换版本：
- `GFPGANv1.3.pth` — 轻量版
- `GFPGANv1.4.pth` — 默认，推荐
- `RestoreFormer.pth` — 细节更丰富

下载地址：https://github.com/TencentARC/GFPGAN/releases

**Q: face_enhance 开启后很慢或 OOM？**  
A: GFPGAN 每帧需额外做人脸检测 + StyleGAN2 解码。建议先用 `--gfpgan-batch-size 4` 减小批大小；底层 OOM 时会自动降级并持久化。`realesr-animevideov3` 等动漫模型不支持 face_enhance，底层会自动禁用。

**Q: 如何启用 TensorRT 加速（`use_tensorrt=true`）？**  
A: TensorRT 为可选依赖，需与已安装的 TensorRT 库版本严格匹配，建议通过 NVIDIA pip 源统一安装：

```bash
pip install tensorrt pycuda onnx "onnxruntime-gpu>=1.16.0" \
    --extra-index-url https://pypi.ngc.nvidia.com
```

- `tensorrt`：TRT Python 绑定（8.x / 10.x，脚本已做双版本 API 兼容）  
- `pycuda`：TRT Engine 执行所需的 CUDA 驱动绑定  
- `onnx`：PyTorch 模型导出为 ONNX（TRT Engine 构建中间格式）  
- `onnxruntime-gpu`：可选，ONNX 导出验证；跳过则 TRT 构建时不做 ONNX 校验  

> ⚠️ **不要**调用 `pycuda.autoinit`，会与 PyTorch 的 CUDA context 冲突导致推理报错。

---

## 技术栈

- **IFRNet** — 基于双向光流的高质量视频插帧
- **Real-ESRGAN** — 基于 GAN 的真实世界超分辨率
- **GFPGAN** — 基于 StyleGAN2 的人脸盲修复
- **PyTorch 2.0+**（推荐 2.7+）— FP16 / torch.compile / CUDA Graph
- **FFmpeg** — 视频处理（支持 NVDEC/NVENC）
- **OpenCV** — 帧级图像处理

---

## 许可证

本项目源代码遵循 **MIT 许可证**。

集成的外部项目请遵循各自许可证：
- [IFRNet](https://github.com/ltkong218/IFRNet) — MIT License
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — BSD-3-Clause License
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — Apache-2.0 License

## 致谢

- [IFRNet](https://github.com/ltkong218/IFRNet) by ltkong218
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by xinntao
- [GFPGAN](https://github.com/TencentARC/GFPGAN) by TencentARC
- FFmpeg 项目 / PyTorch 团队
