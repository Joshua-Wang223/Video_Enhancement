# Video Enhancement — 视频增强处理系统

> **当前版本**: v6（单卡优化版）｜IFRNet 后端 v6.1 / Real-ESRGAN 后端 v6.1  
> **最后更新**: 2026-03-11

一套完整的 AI 视频增强解决方案，整合了**视频插帧（IFRNet）**与**视频超分辨率（Real-ESRGAN）**两大模块，并针对单 GPU 环境进行了深度优化。

---

## 功能特性

| 功能 | 说明 |
|------|------|
| 🎬 **视频插帧** | 基于 IFRNet 光流估计，支持 2x / 4x / 8x / 16x 帧率提升 |
| 🖼️ **视频超分** | 基于 Real-ESRGAN，支持 2x / 4x 分辨率放大 |
| ⚡ **FP16 推理** | 半精度推理，显存减半，速度提升约 1.5–2x |
| 🔧 **torch.compile** | PyTorch 2.0+ 图编译加速（与 TRT 互斥） |
| 🚀 **CUDA Graph** | IFRNet 支持 CUDA Graph 固化推理图，降低 CPU 调度开销（compile 激活时自动接管） |
| 🎥 **硬件编解码** | 自动探测并启用 NVDEC 解码 / NVENC 编码 |
| 🔁 **TensorRT 加速** | 可选 TRT Engine 加速；Engine 缓存于统一目录 `base_dir/.trt_cache`，首次构建后所有分段共享，无需重复构建 |
| 🛡️ **OOM 自动降级** | 显存不足时自动减小批量大小并持久化上限，无需手动干预 |
| 🔗 **分段直连流水线** | 插帧→超分分段直接对接，省去中间合并步骤，减少 30–50% I/O |
| 💾 **断点续传** | 分段级别断点保存，处理中断后从断点继续 |
| 👤 **人脸增强（GFPGAN）** | 批量推理 + 原始帧检测 + CPU-GPU 流水线 + GFPGAN FP16（v6 深度优化） |
| 🎵 **智能音频处理** | 自动检测音频编码，无损时直接 copy，否则转码 |
| 📦 **批量处理** | 支持目录批量处理，失败自动跳过继续 |
| 🧹 **灵活清理** | 处理完成后询问或自动清理临时文件 |
| 🔍 **帧预览** | IFRNet 处理过程中可选弹出帧预览窗口（调试用） |

---

## 项目结构

```
Video_Enhancement/
├── README.md                       # 本文件
├── GUIDE.md                        # 详细使用指南
├── PROJECT_SUMMARY.md              # 架构与技术总结
├── FILE_LIST.md                    # 文件清单与模块说明
│
├── requirements.txt                # Python 依赖
│
├── config/
│   └── default_config.json         # 主配置文件（v6 参数完整版）
│
├── src/
│   ├── main_video_v6_single.py     # ★ 当前主入口（v6 单卡版）
│   ├── main_video_v5_single.py     # 历史版本（保留参考）
│   │
│   ├── processors/
│   │   ├── ifrnet_processor_v6_single.py           # ★ IFRNet 处理器 v6（可独立调用）
│   │   ├── realesrgan_processor_video_v6_single.py # ★ Real-ESRGAN 处理器 v6（可独立调用）
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
│   │   ├── process_video_v6_1_single.py  # ★ IFRNet v6.1 后端核心（单卡）
│   │   ├── process_video_v6_1.py         # IFRNet v6.1 后端核心（多卡）
│   │   └── models/                       # IFRNet 模型定义
│   └── Real-ESRGAN/
│       ├── inference_realesrgan_video_v6_1_single.py  # ★ ESRGAN v6.1 后端核心（单卡）
│       ├── inference_realesrgan_video_v6_1.py         # ESRGAN v6.1 后端核心（多卡）
│       └── realesrgan/                                # Real-ESRGAN 库
│
├── models_IFRNet/
│   └── checkpoints/
│       ├── IFRNet_S_Vimeo90K.pth   # IFRNet Small（推荐，速度快）
│       ├── IFRNet_L_Vimeo90K.pth   # IFRNet Large（质量更高）
│       └── .trt_cache/             # ★ IFRNet TRT Engine 缓存（自动生成）
│
├── models_RealESRGAN/
│   ├── realesr-general-x4v3.pth        # 通用 4x 超分（推荐）
│   ├── RealESRGAN_x4plus.pth           # 写实 4x 超分
│   ├── RealESRGAN_x2plus.pth           # 写实 2x 超分
│   ├── realesr-animevideov3.pth        # 动漫视频专用
│   ├── RealESRGANv2-animevideo-xsx2.pth # 动漫视频 2x 轻量版
│   └── GFPGANv1.4.pth                  # 人脸增强模型（face_enhance 时必需）
│
├── .trt_cache/                     # ★ TRT Engine 统一缓存目录（默认位置，自动生成）
│                                   #   Engine 文件名含模型/shape 信息，IFRNet 与 ESRGan 共享
│                                   #   可通过 --trt-cache-dir 或配置 paths.trt_cache_dir 指定
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

`config/default_config.json` 中各路径字段留空时，`config_manager` 会自动从配置文件位置向上推算项目根目录并派生所有路径（标准部署无需手动填写）：

| 字段 | 留空时自动派生为 |
|------|----------------|
| `paths.base_dir` | 配置文件位置向上两级（`config/default_config.json` → 项目根） |
| `paths.output_dir` | `base_dir/output` |
| `paths.temp_dir` | `base_dir/temp` |
| `paths.log_dir` | `base_dir/logs` |
| `paths.trt_cache_dir` | `base_dir/.trt_cache`（IFRNet + ESRGan 共享） |

非标准部署或路径不正确时，按需手动填写：

```json
{
  "paths": {
    "base_dir":      "/your/project/path",
    "output_dir":    "/your/project/path/output",
    "temp_dir":      "/your/project/path/temp",
    "log_dir":       "/your/project/path/logs",
    "trt_cache_dir": "/your/project/path/.trt_cache"
  },
  "models": {
    "ifrnet": {
      "model_name": "IFRNet_S_Vimeo90K"
    }
  }
}
```

> **模型路径说明：** `models.ifrnet.model_name` 是推荐方式，处理器会自动在 `models_IFRNet/checkpoints/<model_name>.pth` 下查找权重文件。若需指定任意路径，改用 `model_path`（优先级更高）：
> ```json
> "model_path": "/absolute/path/to/IFRNet_S_Vimeo90K.pth"
> ```

### 5. 运行

```bash
# 单视频处理（先插帧再超分，默认）
python src/main_video_v6_single.py -i input.mp4 -o output.mp4

# 先超分再插帧
python src/main_video_v6_single.py -i input.mp4 -o output.mp4 \
    --mode upscale_then_interpolate

# 自定义倍数（4x 插帧 + 4x 超分）
python src/main_video_v6_single.py -i input.mp4 -o output.mp4 \
    --interpolation-factor 4 --upscale-factor 4

# 开启 TRT 加速（IFRNet + ESRGan 共用同一缓存目录，首次构建后重复使用）
python src/main_video_v6_single.py -i input.mp4 -o output.mp4 \
    --use-tensorrt-ifrnet --use-tensorrt-esrgan

# 指定 TRT Engine 缓存目录（覆盖默认 base_dir/.trt_cache）
python src/main_video_v6_single.py -i input.mp4 -o output.mp4 \
    --use-tensorrt-ifrnet --use-tensorrt-esrgan \
    --trt-cache-dir /data/trt_engines

# 指定 IFRNet 模型（L 版质量更高，速度更慢）
python src/main_video_v6_single.py -i input.mp4 -o output.mp4 \
    --ifrnet-model IFRNet_L_Vimeo90K

# 调整批大小（根据显存调优）
python src/main_video_v6_single.py -i input.mp4 -o output.mp4 \
    --batch-size-ifrnet 8 --max-batch-size-ifrnet 16 \
    --batch-size-esrgan 16 --prefetch-factor-esrgan 32

# 开启人脸增强（需预先下载 GFPGANv1.4.pth）
python src/main_video_v6_single.py -i input.mp4 -o output.mp4 \
    --face-enhance

# 人脸增强精细控制（GFPGAN 版本、融合权重、防 OOM 批大小）
python src/main_video_v6_single.py -i input.mp4 -o output.mp4 \
    --face-enhance \
    --gfpgan-model 1.4 --gfpgan-weight 0.7 --gfpgan-batch-size 8

# 批量处理
python src/main_video_v6_single.py --batch-mode \
    --input-dir ./videos --output-dir ./output

# 仅插帧（跳过超分）
python src/main_video_v6_single.py -i input.mp4 -o output.mp4 \
    --skip-upscale --interpolation-factor 4

# 仅超分（跳过插帧）
python src/main_video_v6_single.py -i input.mp4 -o output.mp4 \
    --skip-interpolate --use-tensorrt-esrgan

# 处理完成后自动清理临时文件
python src/main_video_v6_single.py -i input.mp4 -o output.mp4 \
    --auto-cleanup
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

适用于**低分辨率**视频（<1080p），优先提升画质。插帧基于高分辨率帧，光流估计更精确。

---

## 处理示例

### 示例一：低分辨率真人视频——先超分再插帧 + 人脸增强

**场景**：720p 真人视频，30fps，需提升分辨率至 1440p 并开启人脸精修。  
**策略**：先超分（`upscale_then_interpolate`），画质优先；GFPGAN 1.4 融合权重 0.7；TRT 加速加快推理；缓存目录统一指定，Engine 构建一次全程复用。

```bash
python src/main_video_v6_single.py \
    -c /workspace/Video_Enhancement/config/default_config.json \
    -i /workspace/input_videos/test1.mp4 \
    -o /workspace/output/test1_enhanced.mp4 \
    --mode upscale_then_interpolate \
    --interpolation-factor 2 \
    --upscale-factor 2 \
    --esrgan-model realesr-general-x4v3 \
    --segment-duration 10 \
    --use-tensorrt-ifrnet --use-tensorrt-esrgan \
    --trt-cache-dir /workspace/trt_engines \
    --face-enhance \
    --gfpgan-model 1.4 \
    --gfpgan-weight 0.7 \
    --gfpgan-batch-size 8
```

**处理流程**：
```
test1.mp4 (1280×720, 30fps)
  → 分段（10 秒/段）
  → Real-ESRGAN 超分 2x  →  2560×1440
  → IFRNet 插帧 2x       →  60fps
  → 合并 + 音频回填
  → test1_enhanced.mp4 (2560×1440, 60fps)
```

**关键参数说明**：
- `--trt-cache-dir /workspace/trt_engines`：所有分段共用同一套 Engine，仅首次处理需要构建，后续段及后续视频直接复用。
- `--gfpgan-weight 0.7`：融合权重偏向 GFPGAN 输出，人脸修复效果更明显，同时保留约 30% 原始纹理。
- `--gfpgan-batch-size 8`：单次前向处理 8 张人脸，T4/16G 显存可承受；OOM 时自动降级并持久化。
- `--segment-duration 10`：较短分段，减少单段显存峰值，适合高分辨率或人脸密集场景。

---

### 示例二：低帧率动画/录屏视频——先插帧再超分（轻量模式）

**场景**：720p 低帧率视频（如录屏、动画），需提升流畅度，无需人脸增强。  
**策略**：先插帧（`interpolate_then_upscale`，默认），帧率优先；使用默认配置，参数最简。

```bash
python src/main_video_v6_single.py \
    -c /workspace/Video_Enhancement/config/default_config.json \
    -i /workspace/input_videos/animation.mp4 \
    -o /workspace/output/animation_enhanced.mp4 \
    --mode interpolate_then_upscale \
    --esrgan-model realesr-animevideov3 \
    --interpolation-factor 2 \
    --upscale-factor 2
```

**处理流程**：
```
animation.mp4 (原始分辨率, 低帧率)
  → 分段（30 秒/段，默认）
  → IFRNet 插帧 2x       →  帧率翻倍
  → Real-ESRGAN 超分 2x  →  分辨率翻倍
  → 合并 + 音频回填
  → animation_enhanced.mp4
```

**关键参数说明**：
- 未指定 `--ifrnet-model`，使用配置中 `model_name: IFRNet_S_Vimeo90K`（轻量快速）。
- 超分模型 `realesr-animevideov3` 适合动画处理，速度最快；该模型不支持 `face_enhance`（底层自动禁用）。
- 未指定 `--trt-cache-dir`，TRT 缓存默认使用 `base_dir/.trt_cache`。

---

## 命令行参数速查

```
python src/main_video_v6_single.py [选项]

基础参数:
  -i, --input PATH              输入视频路径（单文件模式）
  -o, --output PATH             输出视频路径（含文件名，单文件模式）
  -c, --config PATH             配置文件（默认: config/default_config.json）

批量模式:
  --batch-mode                  启用批量模式（扫描 --input-dir 下所有视频）
  --input-dir DIR               批量输入视频目录
  --output-dir DIR              批量输出目录

处理控制:
  --mode MODE                   interpolate_then_upscale（默认）
                                upscale_then_interpolate
  --interpolation-factor N      插帧倍数：2 / 4 / 8 / 16（默认 2）
  --upscale-factor N            超分倍数：2 / 4（默认 2）
  --segment-duration SECS       分段时长（秒，默认 30）
  --skip-interpolate            跳过 IFRNet 插帧，仅执行超分
  --skip-upscale                跳过 Real-ESRGAN 超分，仅执行插帧
  --auto-cleanup                全流程结束后自动删除所有临时分段文件

IFRNet 模型:
  --ifrnet-model MODEL_NAME     IFRNet 模型名称（覆盖配置）
                                  IFRNet_S_Vimeo90K（默认，轻量快速）
                                  IFRNet_L_Vimeo90K（高质量，速度更慢）
                                  IFRNet_Vimeo90K（标准版）
  --ifrnet-model-path PATH      IFRNet .pth 绝对路径（优先级高于 --ifrnet-model）
  --batch-size-ifrnet N         IFRNet 推理批大小（默认 24）
  --max-batch-size-ifrnet N     IFRNet 批大小上限，OOM 时的天花板（默认 36）

IFRNet 推理优化:
  --no-fp16-ifrnet              禁用 IFRNet FP16（默认开启）
  --no-compile-ifrnet           禁用 IFRNet torch.compile（短视频可跳过预热）
  --no-cuda-graph-ifrnet        禁用 IFRNet CUDA Graph
  --use-tensorrt-ifrnet         启用 IFRNet TensorRT 加速（与 compile/cuda-graph 互斥）
  --no-hwaccel-ifrnet           禁用 IFRNet NVDEC 硬件解码
  --crf-ifrnet N                IFRNet 分段输出 CRF（0~51，默认 23）
  --codec-ifrnet CODEC          IFRNet 分段输出编码器（默认 libx264；有 NVENC 时自动升级）
  --report-ifrnet PATH          IFRNet JSON 性能报告输出路径
  --preview-ifrnet              IFRNet 处理时弹出帧预览窗口（调试用）
  --preview-interval-ifrnet N   IFRNet 帧预览间隔（每隔 N 帧预览一次，默认 30）

Real-ESRGAN 模型:
  --esrgan-model MODEL_NAME     模型名称（覆盖配置，不存在时自动下载）
                                  realesr-general-x4v3（默认，通用高质量）
                                  RealESRGAN_x4plus / RealESRGAN_x2plus
                                  realesr-animevideov3（动漫视频专用）
                                  RealESRGANv2-animevideo-xsx2（动漫 2x 轻量）
  --denoise-strength F          降噪强度 0~1（仅 realesr-general-x4v3 有效，默认 0.5）
  --tile-size N                 tile 切块大小（0=不切块；VRAM 不足时设 512）
  --tile-pad N                  tile 边缘填充（默认 10）
  --pre-pad N                   预处理填充（默认 0）
  --batch-size-esrgan N         ESRGan SR 批处理大小（默认 12）
  --prefetch-factor-esrgan N    ESRGan 读帧预取深度（建议 ≥ batch_size×2，默认 36）

Real-ESRGAN 推理优化:
  --fp32-esrgan                 ESRGan 使用 FP32 精度（默认 FP16）
  --use-compile-esrgan          启用 ESRGan torch.compile（reduce-overhead 模式）
  --use-tensorrt-esrgan         启用 ESRGan TensorRT 加速
  --no-hwaccel-esrgan           禁用 ESRGan NVDEC 硬件解码
  --crf-esrgan N                ESRGan 分段输出 CRF（0~51，默认 23）
  --codec-esrgan CODEC          ESRGan 分段输出编码器（默认 libx264；有 NVENC 时自动升级）
  --report-esrgan PATH          ESRGan JSON 性能报告输出路径

人脸增强（GFPGAN）:
  --face-enhance                开启人脸增强（需 GFPGANv1.4.pth）
  --no-face-enhance             关闭人脸增强（覆盖配置中 face_enhance=true）
  --gfpgan-model VER            GFPGAN 版本：1.3 / 1.4 / RestoreFormer（默认 1.4）
  --gfpgan-weight W             融合权重 0.0~1.0（0=不增强，1=完全替换，默认 0.5）
  --gfpgan-batch-size N         单次 GFPGAN 最多处理人脸数，防 OOM（默认 12）

TRT Engine 缓存（IFRNet / ESRGan 共用）:
  --trt-cache-dir DIR           TRT Engine 缓存目录
                                  未指定 → 读取配置 paths.trt_cache_dir
                                  配置为空 → 自动使用 base_dir/.trt_cache
                                Engine 文件名含模型名称和输入 shape，
                                两个模型共用目录不会冲突，首次构建后全程复用

最终合并输出:
  --output-codec CODEC          最终合并编码器（默认 libx264）
  --output-crf N                最终合并 CRF 质量（0~51，默认 23）
  --output-preset PRESET        最终合并编码预设（如 medium / slow）
```

---

## 性能参考

> 测试环境：RTX 3080（10 GB VRAM），1080p 输入，v6 单卡版，FP16 开启

| 操作 | 设置 | 10 分钟视频耗时 |
|------|------|--------------|
| 插帧 | 2x IFRNet_S | ~7 分钟 |
| 插帧 | 4x IFRNet_S | ~14 分钟 |
| 超分 | 2x | ~50 分钟 |
| 超分 | 4x | ~125 分钟 |
| **插帧 2x + 超分 4x（直连流水线）** | **推荐配置** | **~132 分钟** |

v6 直连流水线相比 v1 节省约 30% 总时间和 38% 磁盘 I/O。TRT 加速在 Engine 已缓存的情况下，推理吞吐量可额外提升 15–30%。

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
  "paths": {
    "base_dir":      "",
    "trt_cache_dir": ""
  },
  "models": {
    "ifrnet": {
      "model_name":     "IFRNet_S_Vimeo90K",
      "model_path":     "",
      "use_fp16":       true,
      "use_compile":    false,
      "use_cuda_graph": true,
      "use_tensorrt":   true,
      "use_hwaccel":    true,
      "batch_size":     24,
      "max_batch_size": 36,
      "crf":            23
    },
    "realesrgan": {
      "model_name":        "realesr-general-x4v3",
      "tile_size":         0,
      "batch_size":        12,
      "prefetch_factor":   36,
      "use_compile":       false,
      "use_tensorrt":      true,
      "use_hwaccel":       true,
      "fp32":              false,
      "crf":               23,
      "face_enhance":      false,
      "gfpgan_model":      "1.4",
      "gfpgan_weight":     0.5,
      "gfpgan_batch_size": 12
    }
  }
}
```

> **关键参数说明：**
> - `paths.trt_cache_dir`：留空时自动派生为 `base_dir/.trt_cache`；IFRNet 与 ESRGan 共享同一目录，Engine 文件名含模型/shape 信息不会冲突。命令行可用 `--trt-cache-dir` 覆盖。
> - `ifrnet.use_tensorrt`：与 `use_compile` / `use_cuda_graph` 互斥；启用后 config_manager 会自动强制关闭后两者。
> - `ifrnet.max_batch_size`：OOM 自动降级的天花板，首次 OOM 后永久压低至此值，保护显存。
> - `realesrgan.prefetch_factor`：建议 ≥ `batch_size × 2`，充分隐藏磁盘读取延迟。
> - `realesrgan.use_compile` 默认关闭（首次编译需 1–3 分钟），适合长批量任务时开启。
> - `face_enhance` 对 anime 系模型（`realesr-animevideov3` 等）无效，底层脚本自动禁用。
> - `gfpgan_batch_size` 控制单次前向处理的人脸数；人脸密集场景可调小至 4 防 OOM，降级值自动持久化。

完整配置说明见 [GUIDE.md](GUIDE.md)。

---

## 独立调用处理器

两个处理器均可脱离主流程独立运行，方便单独调试：

```bash
# ── IFRNet 独立插帧 ───────────────────────────────────────────────────────────

# 基础 2x 插帧（使用配置中 model_name 自动拼路径）
python src/processors/ifrnet_processor_v6_single.py \
    -i input.mp4 -o output_2x.mp4

# 指定模型名称 + 4x 插帧
python src/processors/ifrnet_processor_v6_single.py \
    -i input.mp4 -o output_4x.mp4 \
    --interpolation-factor 4 --ifrnet-model IFRNet_L_Vimeo90K

# 启用 TRT 加速 + 指定缓存目录
python src/processors/ifrnet_processor_v6_single.py \
    -i input.mp4 -o output_2x.mp4 \
    --use-tensorrt --trt-cache-dir /data/trt_engines

# 关闭 compile（短视频跳过 1–3 分钟预热）
python src/processors/ifrnet_processor_v6_single.py \
    -i input.mp4 -o output_2x.mp4 \
    --no-compile --no-cuda-graph

# ── Real-ESRGAN 独立超分 ──────────────────────────────────────────────────────

# 基础 4x 超分
python src/processors/realesrgan_processor_video_v6_single.py \
    -i input.mp4 -o output_4x.mp4 --upscale-factor 4

# 启用 TRT 加速 + 指定缓存目录
python src/processors/realesrgan_processor_video_v6_single.py \
    -i input.mp4 -o output_4x.mp4 \
    --use-tensorrt --trt-cache-dir /data/trt_engines

# 开启人脸增强，精细控制 GFPGAN
python src/processors/realesrgan_processor_video_v6_single.py \
    -i input.mp4 -o output.mp4 \
    --face-enhance --gfpgan-model 1.4 \
    --gfpgan-weight 0.5 --gfpgan-batch-size 8
```

---

## TRT Engine 缓存机制

v6 引入了统一 TRT 缓存目录，解决了早期版本中每个分段、每个视频各自重新构建 Engine 的问题。

**优先级链：**
```
CLI --trt-cache-dir
      ↓ 写入 config
config paths.trt_cache_dir（JSON 显式值）
      ↓ 若为空
config_manager 自动派生 → base_dir/.trt_cache
      ↓ 若处理器层仍拿到空值（最终兜底）
底层脚本默认：base_dir/.trt_cache（IFRNet 与 ESRGan 统一）
```

**Engine 文件命名规则：** `<模型名称>_B<batch>_H<height>_W<width>_fp16.trt`（示例：`ifrnet_S_B24_H576_W736_fp16.trt`）。IFRNet 与 ESRGan 的 Engine 文件名不同，共用同一目录不会冲突。

**实际效果：**
- 同一视频：所有分段共用同一 Engine，仅第一个分段需要构建（约 3–10 分钟），后续分段直接加载。
- 跨视频：只要分辨率和批大小不变，Engine 在不同视频间复用，无需重新构建。
- TRT 版本升级后需手动删除 `.trt_cache/` 目录以触发重建。

---

## 常见问题

**Q: CUDA out of memory？**  
A: v6 支持 OOM 自动降级，会自动减小批量大小并持久化上限（`max_batch_size`）。如仍失败，在配置中设置 `"tile_size": 512` 并将 `"batch_size"` 调小至 1–2。

**Q: 启动时报 `FileNotFoundError`（IFRNet 模型路径为空）？**  
A: 处理器会在加载前校验路径并打印完整提示，请按提示操作：
- 确认权重文件已下载并放置到 `models_IFRNet/checkpoints/IFRNet_S_Vimeo90K.pth`，或
- 命令行传入 `--ifrnet-model IFRNet_S_Vimeo90K`（自动拼接路径），或
- 命令行传入 `--ifrnet-model-path /绝对/路径/model.pth` 直接指定。

**Q: 如何从中断恢复？**  
A: 重新运行完全相同的命令，程序自动读取 `temp/` 下的 `checkpoint.json` 从断点继续。

**Q: 输出视频没有音频？**  
A: v6 使用智能音频提取（`audio_format: "smart"`），无损时直接 copy，否则转码。请确认源视频含音频且 FFmpeg 已正确安装。

**Q: torch.compile 报错？**  
A: 需要 PyTorch 2.0+，首次编译约需 1–3 分钟。如遇兼容性问题，传入 `--no-compile-ifrnet` 或在配置中设置 `"use_compile": false`。注意 `use_tensorrt=true` 时 compile 和 CUDA Graph 会被自动关闭。

**Q: TRT Engine 每次都重新构建，没有复用？**  
A: 检查 `--trt-cache-dir` 是否指向一个稳定固定的目录，而不是每次变化的临时路径。或在配置文件中显式设置 `"paths.trt_cache_dir": "/stable/path/.trt_cache"`。另外，若输入视频分辨率或 `batch_size` 发生变化，Engine 文件名不同，会触发新建。

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
- **TensorRT 8.x / 10.x** — 可选推理加速，双版本 API 兼容
- **FFmpeg** — 视频处理（NVDEC 解码 / NVENC 编码）
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
