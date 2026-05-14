# Video Enhancement — 视频增强处理系统

> **当前版本**: v2.0.0（优化版）｜IFRNet 后端 **v6.3.3**（双流深度流水线 + 段后调参 / 编码与 NVENC 调优）/ Real-ESRGAN 后端 v6.4（深度模块化架构 + 独立优化版）  
> **最后更新**: 2026-05-14

一套完整的 AI 视频增强解决方案，整合了**视频插帧（IFRNet v6.3.3）**与**视频超分辨率（Real-ESRGAN v6.4）**两大模块，并针对单 GPU 环境进行了深度优化。IFRNet 后端在 v6.3.0 双流架构（H2D 预取 / D2H 独立流 + CudaEventPool）之上，v6.3.3 进一步改进段完成后 RETUNE 统计、队列与 PinnedPool 可观测性、软编并行与 NVENC pipe 场景参数等。Real-ESRGAN 侧通过 `realesrgan_video`（深度模块化、4 级并行 + **多片段复用**：模型/TRT 一次初始化、分段复用）与独立优化版 v6.4 并存。

**配置真源：** 未在命令行显式覆盖时，所有默认值以仓库内 [`config/default_config.json`](config/default_config.json) 为准；若本 README 与 JSON 不一致，以 JSON 为准。

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
| 🔀 **双流深度流水线** | IFRNet v6.3.x 双 transfer stream（H2D 预取 + D2H 输出独立流），计算/传输三流全重叠 |
| 🎯 **CudaEventPool** | IFRNet 预分配 CUDA Event 对象池，消除每批次 Event 创建/销毁开销 |
| 📊 **GPU 监测** | IFRNet 后台 GPU 监测线程（2 秒采样），运行结束后输出利用率/显存统计与 batch_size 推荐 |
| 🚄 **批大小可调** | 默认批大小见 `default_config.json`（IFRNet 当前为 24，上限 36）；显存充足可提高以换吞吐 |
| 🔁 **多片段复用（ESRGAN）** | `create_video_enhancer` + `run_pipeline_for_video`：多段任务只加载一次模型/TRT，降低段间启动开销 |
| 🛡️ **OOM 自动降级** | 显存不足时自动减小批量大小并持久化上限，无需手动干预 |
| 🔗 **分段直连流水线** | 插帧→超分分段直接对接，省去中间合并步骤，减少 30–50% I/O |
| 💾 **断点续传** | 分段级别断点保存，处理中断后从断点继续 |
| 👤 **人脸增强（GFPGAN）** | 批量推理 + 原始帧检测 + CPU-GPU 流水线 + GFPGAN FP16 |
| 🎯 **人脸检测置信度过滤** | 可配置阈值过滤低质量检测，减少无效 GFPGAN 推理 |
| 🔄 **自适应批处理** | 根据人脸密度动态调整读帧批大小，平衡吞吐与显存 |
| 🧠 **GFPGAN TRT 子进程** | GFPGAN 可选 TensorRT 子进程加速（独立进程隔离，防止主进程 OOM） |
| 🏗️ **深度流水线架构** | Real-ESRGAN 端 4 级并行：读帧→SR→GFPGAN→写帧 + SR H2D 预取重叠 |
| 🎵 **智能音频处理** | 自动检测音频编码，无损时直接 copy，否则转码 |
| 📦 **批量处理** | 支持目录批量处理，失败自动跳过继续 |
| 🧹 **灵活清理** | 处理完成后询问或自动清理临时文件 |
| 🔍 **帧预览** | IFRNet / Real-ESRGAN 处理过程中可选弹出帧预览窗口（调试用，按 q 退出） |
| 🧪 **Dry-run 模式** | `--dry-run` 仅打印配置和环境信息，不实际处理 |
| 🧹 **可选预去噪** | 支持 NAFNet / DnCNN / SCUNet 预去噪阶段（在插帧/超分之前执行） |
| 📊 **JSON 性能报告** | IFRNet / Real-ESRGAN 均支持输出 JSON 性能报告（含推理延迟、硬件状态等） |
| 📋 **流水线报告** | `--report` 输出端到端流水线 JSON 报告（含耗时、大小、GPU 峰值显存、参数快照） |
| 🔧 **FIX-C 分段归一化** | copy 合并前自动统一 MP4 时间基（90 kHz），消除 concat demuxer AVERROR_EXIT(254) |
| 🖥️ **GPU 硬件型号探测** | 自动识别 GPU 型号（20+ 型号表），选择最优 NVENC 编码器 |

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
│   └── default_config.json         # ★ 主配置（默认真源；processing/paths/models/output/temp_files/logging）
│
├── src/
│   ├── main_video_optimized.py     # ★ 当前主入口（优化版，对接 realesrgan_video）
│   │                                #   FIX-C: 合并前分段 timescale 归一化
│   │                                #   --report: 端到端流水线 JSON 报告
│   │                                #   --skip-seg-normalize: 跳过归一化
│   ├── main_video_v6_single.py     # v6 主入口（保留参考，对接 external/Real-ESRGAN）
│   ├── main_video_v5_single.py     # 历史版本主入口（保留参考）
│   │
│   ├── processors/
│   │   ├── ifrnet_processor_v6_1_single.py          # ★ IFRNet 处理器 v6.1（对接 v6.3.3 后端）
│   │   ├── ifrnet_processor_v6_single.py            # IFRNet 处理器 v6（历史版本，保留参考）
│   │   ├── realesrgan_processor_video_optimized.py  # ★ Real-ESRGAN 处理器（优化版，对接 realesrgan_video）
│   │   ├── realesrgan_processor_video_v6_single.py  # Real-ESRGAN 处理器 v6（历史版本，保留参考）
│   │   └── ...                                       # 更早历史版本处理器
│   │
│   └── utils/
│       ├── config_manager.py       # 配置管理
│       ├── video_utils.py          # 视频工具（分割、合并、音频、编解码）
│       ├── video_fixer.py          # 损坏视频修复
│       └── output_filter.py        # FFmpeg 输出过滤
│
├── external/
│   ├── IFRNet/
│   │   ├── process_video_v6_3_3_single.py  # ★ IFRNet v6.3.3 后端核心（当前对接）
│   │   ├── process_video_v6_3_0_single.py    # IFRNet v6.3.0（历史，双流流水线基线）
│   │   ├── process_video_v6_1_single.py    # IFRNet v6.1 后端核心（历史版本）
│   │   ├── process_video_v6_1.py           # IFRNet v6.1 后端核心（多卡）
│   │   ├── process_video_v6_2_4_single.py  # IFRNet v6.2.4 后端核心（过渡版本）
│   │   └── models/                         # IFRNet 模型定义
│   │
│   └── realesrgan_video/                # ★ Real-ESRGAN v6.4 深度模块化子项目
│       ├── main.py                       #   主入口（main_optimized），由处理器调用
│       │                                  #   create_video_enhancer() / run_pipeline_for_video()
│       │                                  #   _HWProfile / _GPU_PROFILES_TABLE（硬件型号探测）
│       │                                  #   PreviewWriter（子进程 GUI 检测 + 实时预览）
│       ├── pipeline.py                   #   深度流水线架构（4级并行处理）
│       ├── ffmpeg_io.py                  #   FFmpeg 读写器（NVDEC/NVENC + 异步预取）
│       ├── tensorrt_accel.py             #   SR TensorRT 加速器
│       ├── gfpgan_subprocess.py          #   GFPGAN TRT 子进程
│       ├── face_utils.py                 #   人脸检测 / 增强工具
│       ├── config.py                     #   模型配置与路径
│       ├── realesrgan_utils.py           #   Real-ESRGAN 模型构建工具
│       ├── async_dispatcher.py           #   异步调度器
│       ├── __init__.py                   #   包初始化
│       └── realesrgan/                   #   Real-ESRGAN 库（内嵌）
│
│   └── Real-ESRGAN/                      # ★ Real-ESRGAN v6.4 独立优化版（保留参考）
│       ├── inference_realesrgan_video_v6_4_optimized.py  # v6.4 架构优化版
│       ├── inference_realesrgan_video_v6_3_optimized.py  # v6.3 优化版
│       ├── inference_realesrgan_video_v6_1_single.py     # v6.1 单卡版
│       └── ...                                             # 更早版本
│
├── models_IFRNet/
│   └── checkpoints/
│       ├── IFRNet_S_Vimeo90K.pth   # IFRNet Small（推荐，速度快）
│       ├── IFRNet_L_Vimeo90K.pth   # IFRNet Large（质量更高）
│       └── .trt_cache/             # IFRNet TRT Engine 缓存（自动生成）
│
├── models_RealESRGAN/
│   ├── realesr-general-x4v3.pth        # 通用 4x 超分（推荐）
│   ├── RealESRGAN_x4plus.pth           # 写实 4x 超分
│   ├── RealESRGAN_x2plus.pth           # 写实 2x 超分
│   ├── realesr-animevideov3.pth        # 动漫视频专用
│   ├── RealESRGANv2-animevideo-xsx2.pth # 动漫视频 2x 轻量版
│   └── GFPGANv1.4.pth                  # 人脸增强模型（face_enhance 时必需）
│
├── models_GFPGAN/                  # ★ GFPGAN 模型（人脸增强统一目录）
│   ├── GFPGANv1.4.pth                  # GFPGAN 主生成模型（推荐）
│   ├── GFPGANv1.3.pth                  # GFPGAN 轻量版
│   └── gfpgan/weights/                 # facexlib 辅助模型（自动下载）
│       ├── detection_Resnet50_Final.pth # 人脸检测
│       └── parsing_parsenet.pth        # 人脸解析
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
mkdir -p models_GFPGAN
wget -O models_GFPGAN/GFPGANv1.4.pth \
  https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
```

其他可选模型见 [FILE_LIST.md](FILE_LIST.md)。

### 4. 配置路径

`config/default_config.json` 为**默认真源**。其中各路径字段留空时，`config_manager` 会从配置文件位置向上推算项目根目录并派生路径（标准部署无需手动填写）：

| 字段 | 留空时自动派生为 |
|------|----------------|
| `paths.base_dir` | 配置文件位置向上两级（`config/default_config.json` → 项目根） |
| `paths.output_dir` | `base_dir/output` |
| `paths.temp_dir` | `base_dir/temp` |
| `paths.log_dir` | `base_dir/logs` |
| `paths.trt_cache_dir` | `base_dir/.trt_cache`（IFRNet + ESRGan 共享） |
| `paths.models_gfpgan_dir` | `base_dir/models_GFPGAN`（GFPGAN 主模型与 facexlib 辅助权重） |

**批量模式：** `processing.batch_mode` 为 `true` 时，从 `paths.input_dir` 扫描视频；单文件模式使用 `paths.input_video` 或由命令行 `-i`/`-o` 覆盖。

非标准部署或路径不正确时，按需手动填写（示例）：

```json
{
  "paths": {
    "base_dir":           "/your/project/path",
    "output_dir":         "/your/project/path/output",
    "temp_dir":           "/your/project/path/temp",
    "log_dir":            "/your/project/path/logs",
    "trt_cache_dir":      "/your/project/path/.trt_cache",
    "models_gfpgan_dir":  "/your/project/path/models_GFPGAN"
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
python src/main_video_optimized.py -i input.mp4 -o output.mp4

# 先超分再插帧
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --mode upscale_then_interpolate

# 自定义倍数（4x 插帧 + 4x 超分）
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --interpolation-factor 4 --upscale-factor 4

# 开启 TRT 加速（IFRNet + ESRGan 共用同一缓存目录，首次构建后重复使用）
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --use-tensorrt-ifrnet --use-tensorrt-esrgan

# 指定 TRT Engine 缓存目录（覆盖默认 base_dir/.trt_cache）
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --use-tensorrt-ifrnet --use-tensorrt-esrgan \
    --trt-cache-dir /data/trt_engines

# 指定 IFRNet 模型（L 版质量更高，速度更慢）
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --ifrnet-model IFRNet_L_Vimeo90K

# 调整批大小（根据显存调优）
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --batch-size-ifrnet 8 --max-batch-size-ifrnet 16 \
    --batch-size-esrgan 8 --prefetch-factor-esrgan 96

# 开启人脸增强（需预先下载 GFPGANv1.4.pth）
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --face-enhance

# 人脸增强精细控制（GFPGAN 版本、融合权重、检测阈值、自适应批处理）
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --face-enhance \
    --gfpgan-model 1.4 --gfpgan-weight 0.7 --gfpgan-batch-size 8 \
    --face-det-threshold 0.7

# 全 TensorRT 加速（含 GFPGAN TRT 子进程）
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --use-tensorrt-ifrnet --use-tensorrt-esrgan \
    --gfpgan-trt --face-enhance

# 批量处理
python src/main_video_optimized.py --batch-mode \
    --input-dir ./videos --output-dir ./output

# 仅插帧（跳过超分）
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --skip-upscale --interpolation-factor 4

# 仅超分（跳过插帧）
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --skip-interpolate --use-tensorrt-esrgan

# 输出流水线 JSON 报告
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --report pipeline_report.json

# 处理完成后自动清理临时文件
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --auto-cleanup

# 跳过合并前分段 timescale 归一化（调试用）
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --skip-seg-normalize

# 预去噪 + 插帧 + 超分 全流水线
python src/main_video_optimized.py -i noisy.mp4 -o clean_4x.mp4 \
    --denoise --denoise-model nafnet --denoise-strength-pre 0.5

# 低显存模式（分块 + 小批量 + 禁用高级优化）
python src/main_video_optimized.py -i input.mp4 -o output.mp4 \
    --tile-size 512 --batch-size-esrgan 2 \
    --no-cuda-graph-esrgan --no-fp16-esrgan

# Dry-run（仅打印配置和环境信息，不实际处理）
python src/main_video_optimized.py -i input.mp4 -o output.mp4 --dry-run
```

---

## 处理模式

### 模式 1：先插帧再超分（`interpolate_then_upscale`，默认）

```
输入视频 → [可选去噪] → 分段 → IFRNet 插帧 ──→（直连）──→ Real-ESRGAN 超分 → [FIX-C 归一化] → 合并 → 输出
```

适用于**低帧率**视频（<30 fps），优先提升流畅度。插帧在原始分辨率下进行，运算量更小。

### 模式 2：先超分再插帧（`upscale_then_interpolate`）

```
输入视频 → [可选去噪] → 分段 → Real-ESRGAN 超分 ──→（直连）──→ IFRNet 插帧 → [FIX-C 归一化] → 合并 → 输出
```

适用于**低分辨率**视频（<1080p），优先提升画质；或需要人脸增强（减轻人脸增强所需的高计算量）。插帧基于高分辨率帧，光流估计更精确。

---

## 处理示例

### 示例一：低分辨率真人视频——先超分再插帧 + 人脸增强

**场景**：720p 真人视频，30fps，需提升分辨率至 1440p 并开启人脸精修。  
**策略**：先超分（`upscale_then_interpolate`），画质优先；GFPGAN 1.4 融合权重 0.7 + 检测阈值 0.7 过滤模糊人脸；TRT 加速加快推理；缓存目录统一指定，Engine 构建一次全程复用。

```bash
python src/main_video_optimized.py \
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
    --gfpgan-batch-size 8 \
    --face-det-threshold 0.7
```

**处理流程**：
```
test1.mp4 (1280×720, 30fps)
  → 分段（10 秒/段）
  → Real-ESRGAN 超分 2x  →  2560×1440（深度流水线 4 级并行）
  → IFRNet 插帧 2x       →  60fps
  → [FIX-C] timescale 归一化 → 合并 + 音频回填
  → test1_enhanced.mp4 (2560×1440, 60fps)
```

**关键参数说明**：
- `--trt-cache-dir /workspace/trt_engines`：所有分段共用同一套 Engine，仅首次处理需要构建，后续段及后续视频直接复用。
- `--gfpgan-weight 0.7`：融合权重偏向 GFPGAN 输出，人脸修复效果更明显，同时保留约 30% 原始纹理。
- `--gfpgan-batch-size 8`：单次前向处理 8 张人脸，T4/16G 显存可承受；OOM 时自动降级并持久化。
- `--face-det-threshold 0.7`：过滤模糊远景中的低置信度人脸检测，减少无效 GFPGAN 推理。
- `--segment-duration 10`：较短分段，减少单段显存峰值，适合高分辨率或人脸密集场景。

---

### 示例二：低帧率动画/录屏视频——先插帧再超分（轻量模式）

**场景**：720p 低帧率视频（如录屏、动画），需提升流畅度，无需人脸增强。  
**策略**：先插帧（`interpolate_then_upscale`，默认），帧率优先；使用默认配置，参数最简。

```bash
python src/main_video_optimized.py \
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
  → Real-ESRGAN 超分 2x  →  分辨率翻倍（深度流水线 4 级并行）
  → [FIX-C] timescale 归一化 → 合并 + 音频回填
  → animation_enhanced.mp4
```

**关键参数说明**：
- 未指定 `--ifrnet-model`，使用配置中 `model_name: IFRNet_S_Vimeo90K`（轻量快速）。
- 超分模型 `realesr-animevideov3` 适合动画处理，速度最快；该模型不支持 `face_enhance`（底层自动禁用）。
- 未指定 `--trt-cache-dir`，TRT 缓存默认使用 `base_dir/.trt_cache`。

---

### 示例三：全 TensorRT 加速 + GFPGAN TRT 子进程

**场景**：大批量真人视频处理，需要最大化吞吐。  
**策略**：IFRNet TRT + ESRGan TRT + GFPGAN TRT 三路加速，子进程隔离 GFPGAN 防止主进程 OOM。

```bash
python src/main_video_optimized.py \
    -i input.mp4 -o output.mp4 \
    --use-tensorrt-ifrnet --use-tensorrt-esrgan \
    --gfpgan-trt --face-enhance \
    --batch-size-esrgan 8 --prefetch-factor-esrgan 96 \
    --trt-cache-dir /data/trt_engines \
    --report pipeline_report.json
```

**关键参数说明**：
- `--gfpgan-trt`：启用时自动禁用 FP16 改用 FP32（防止 GFPGAN TRT FP16 溢出噪斑）。
- GFPGAN 在独立子进程中运行，与主进程 SR 推理隔离，子进程 OOM 不影响主进程。
- `--report`：输出端到端流水线 JSON 报告（含耗时、大小、GPU 峰值显存等）。

---

## 命令行参数速查

```
python src/main_video_optimized.py [选项]

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
  --keep-intermediate           保留去噪等中间文件（调试用）
  --dry-run                     仅打印配置和环境信息，不实际处理

去噪参数（可选前处理阶段）:
  --denoise                     启用预去噪阶段（在插帧/超分之前执行）
  --denoise-model MODEL         去噪模型：nafnet / dncnn / scunet（默认 nafnet）
  --denoise-strength-pre F      预去噪强度 0.0~1.0（默认 0.5）

IFRNet 模型:
  --ifrnet-model MODEL_NAME     IFRNet 模型名称（覆盖配置）
                                  IFRNet_S_Vimeo90K（默认，轻量快速）
                                  IFRNet_L_Vimeo90K（高质量，速度更慢）
                                  IFRNet_Vimeo90K（标准版）
  --ifrnet-model-path PATH      IFRNet .pth 绝对路径（优先级高于 --ifrnet-model）
  --batch-size-ifrnet N         IFRNet 推理批大小（覆盖配置；未指定时取自 default_config.json）
  --max-batch-size-ifrnet N     IFRNet 批大小上限 / OOM 天花板（覆盖配置）

IFRNet 推理优化:
  --no-fp16-ifrnet              禁用 IFRNet FP16（默认开启）
  --no-compile-ifrnet           禁用 IFRNet torch.compile（短视频可跳过预热）
  --no-cuda-graph-ifrnet        禁用 IFRNet CUDA Graph
  --use-tensorrt-ifrnet         启用 IFRNet TensorRT 加速（与 compile/cuda-graph 互斥）
  --no-hwaccel-ifrnet           禁用 IFRNet NVDEC 硬件解码
  --no-audio-ifrnet             IFRNet 分段处理时不保留音轨（主流程会统一处理音频）
  --crf-ifrnet N                IFRNet 分段输出 CRF（0~51，默认 23）
  --codec-ifrnet CODEC          IFRNet 分段输出编码器（默认 libx264；有 NVENC 时自动升级）
  --x264-preset-ifrnet PRESET   IFRNet libx264/libx265 预设（ultrafast~veryslow，默认见配置）
  --report-ifrnet PATH          IFRNet JSON 性能报告输出路径
  --preview-ifrnet              IFRNet 处理时弹出帧预览窗口（调试用）
  --preview-interval-ifrnet N   IFRNet 帧预览间隔（每隔 N 帧预览一次，默认 30）
  --quiet-ifrnet                静默 IFRNet 底层冗余输出（默认开启）
  --no-quiet-ifrnet             打印 IFRNet 底层详细日志（如 FFmpeg 命令行）

IFRNet 高优先级覆盖:
  --use-cuda-graph-ifrnet       [覆盖] 强制启用 IFRNet CUDA Graph
  --use-compile-ifrnet          [覆盖] 强制启用 IFRNet torch.compile
  --no-tensorrt-ifrnet          [覆盖] 强制禁用 IFRNet TensorRT

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
  --batch-size-esrgan N         ESRGan SR 批处理大小（覆盖配置；未指定时取自 default_config.json）
  --prefetch-factor-esrgan N    ESRGan 读帧预取深度（建议 ≥ batch_size×2；默认见配置）

Real-ESRGAN 推理优化:
  --no-fp16-esrgan              ESRGan 禁用 FP16（默认开启）
  --no-compile-esrgan           禁用 ESRGan torch.compile（默认开启）
  --no-cuda-graph-esrgan        禁用 ESRGan CUDA Graph（默认开启）
  --use-tensorrt-esrgan         启用 ESRGan TensorRT 加速
  --no-hwaccel-esrgan           禁用 ESRGan NVDEC 硬件解码
  --crf-esrgan N                ESRGan 分段输出 CRF（0~51，默认 23）
  --codec-esrgan CODEC          ESRGan 分段输出编码器（默认 libx264；有 NVENC 时自动升级）
  --x264-preset-esrgan PRESET   ESRGan libx264/libx265 编码预设（ultrafast~veryslow，默认 medium）
  --ffmpeg-bin PATH             ffmpeg 可执行文件路径（默认 ffmpeg）

Real-ESRGAN 高优先级覆盖:
  --no-tensorrt-esrgan          [覆盖] 强制禁用 ESRGan TensorRT
  --use-compile-esrgan          [覆盖] 强制启用 ESRGan torch.compile
  --use-cuda-graph-esrgan       [覆盖] 强制启用 ESRGan CUDA Graph

Real-ESRGAN 预览与报告:
  --report-esrgan PATH          ESRGan JSON 性能报告输出路径
  --preview-esrgan              启用 ESRGan 实时预览窗口（显示最终输出结果，按 q 退出）
  --preview-interval-esrgan N   ESRGan 预览帧间隔（每多少帧刷新一次窗口，默认 30）
  --quiet-esrgan                静默 ESRGan 底层冗余输出（默认开启）
  --no-quiet-esrgan             打印 ESRGan 底层详细日志

人脸增强（GFPGAN）:
  --face-enhance                开启人脸增强（需 GFPGANv1.4.pth）
  --no-face-enhance             关闭人脸增强（覆盖配置中 face_enhance=true）
  --gfpgan-model VER            GFPGAN 版本：1.3 / 1.4 / RestoreFormer（默认 1.4）
  --gfpgan-weight W             融合权重 0.0~1.0（0=不增强，1=完全替换；默认见配置，当前为 0.7）
  --gfpgan-batch-size N         单次 GFPGAN 最多处理人脸数，防 OOM（默认见配置，当前为 4）
  --face-det-threshold F        人脸检测置信度阈值 0.0~1.0（默认见配置，当前为 0.7）
                                  0.5=保留多数人脸，0.7=过滤模糊远景，0.9=仅保留清晰人脸
  --no-adaptive-batch-esrgan    禁用基于人脸密度的自适应批处理（默认开启）
  --gfpgan-trt                  GFPGAN TensorRT 子进程加速（启用时自动禁用 FP16 改用 FP32）

TRT Engine 缓存（IFRNet / ESRGan 共用）:
  --trt-cache-dir DIR           TRT Engine 缓存目录
                                  未指定 → 读取配置 paths.trt_cache_dir
                                  配置为空 → 自动使用 base_dir/.trt_cache
                                Engine 文件名含模型名称和输入 shape，
                                两个模型共用目录不会冲突，首次构建后全程复用

最终合并输出:
  --output-codec CODEC          最终合并编码器（默认 libx264；未指定时自动使用 -c:v copy）
  --output-crf N                最终合并 CRF 质量（0~51，默认 23）
  --output-preset PRESET        最终合并编码预设（如 medium / slow）

流水线报告:
  --report PATH                 输出端到端流水线 JSON 报告（含耗时、大小、GPU 峰值显存等）

杂项:
  --skip-seg-normalize          跳过合并前分段 timescale 归一化（FIX-C；调试用）
  --quiet-ifrnet / --quiet-esrgan  静默底层输出（默认开启）
  --version, -V                 显示版本号
```

---

## 性能参考

> 测试环境：RTX 3080（10 GB VRAM），1080p 输入，优化版，FP16 开启。下表插帧行在 **高显存调优**（例如 `batch_size` 提到 48）下的参考值；仓库默认配置见 `default_config.json`（IFRNet `batch_size` 为 24）。

| 操作 | 设置 | 10 分钟视频耗时 |
|------|------|--------------|
| 插帧 | 2x IFRNet_S（v6.3.3，bs=48 调优） | ~5 分钟 |
| 插帧 | 4x IFRNet_S（v6.3.3，bs=48 调优） | ~10 分钟 |
| 超分 | 2x | ~50 分钟 |
| 超分 | 4x | ~125 分钟 |
| **插帧 2x + 超分 4x（直连流水线）** | **推荐配置** | **~127 分钟** |

IFRNet v6.3.x 双流深度流水线在提高 `batch_size` 与队列深度后，插帧吞吐可较保守默认再提升约 20–30%（视显存与分辨率而定）。优化版直连流水线相比 v1 节省约 30% 总时间和 38% 磁盘 I/O。Real-ESRGAN v6.4 深度流水线 + 多片段复用降低段间启动成本；TRT 在 Engine 已缓存时推理吞吐可额外提升约 15–30%。

---

## 配置速查

以下为 [`config/default_config.json`](config/default_config.json) 的**值快照**（已省略仅作说明的 `"// ..."` 注释键；修改默认行为请直接编辑该文件）。命令行参数可覆盖对应项。

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
  },
  "paths": {
    "base_dir": "",
    "input_video": "",
    "input_dir": "",
    "output_dir": "",
    "temp_dir": "",
    "log_dir": "",
    "trt_cache_dir": "",
    "models_gfpgan_dir": ""
  },
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
    },
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
      "gfpgan_model": "1.4",
      "gfpgan_model_path": "",
      "gfpgan_weight": 0.7,
      "gfpgan_batch_size": 4,
      "face_det_threshold": 0.7,
      "adaptive_batch": true,
      "use_hwaccel": true,
      "video_codec": "libx264",
      "codec": "libx264",
      "x264_preset": "medium",
      "crf": 23,
      "ffmpeg_bin": "ffmpeg",
      "preview": false,
      "preview_interval": 30,
      "report_json": null
    }
  },
  "output": {
    "format": "mp4",
    "codec": "libx264",
    "preset": "medium",
    "crf": 23,
    "pix_fmt": "yuv420p",
    "audio_format": "smart",
    "audio_codec": "copy",
    "audio_bitrate": "192k",
    "bitrate": "192k"
  },
  "temp_files": {
    "segment_prefix": "segment_",
    "processed_prefix": "processed_",
    "temp_video_suffix": "_temp.mp4"
  },
  "logging": {
    "level": "INFO",
    "log_to_file": true,
    "log_to_console": true
  }
}
```

> **关键参数说明：**
> - `processing.auto_fix_corrupted`：是否在处理前尝试修复损坏视频（默认 `false`，按需在 JSON 中开启）。
> - `processing.batch_mode` + `paths.input_dir`：批量扫描输入目录；与命令行 `--batch-mode` / `--input-dir` 一致。
> - `paths.models_gfpgan_dir`：留空则派生为 `base_dir/models_GFPGAN`。
> - `paths.trt_cache_dir`：留空派生 `base_dir/.trt_cache`；IFRNet 与 ESRGan 共用目录，Engine 文件名区分模型与 shape。可用 `--trt-cache-dir` 覆盖。
> - `ifrnet`：当前默认 **关闭** `use_compile` / `use_cuda_graph`、**开启** `use_tensorrt`（与 TRT 互斥时由 `config_manager` 关闭 compile/cuda-graph）。`report_json` 为 `null` 表示不写 IFRNet JSON 报告。
> - `ifrnet.max_batch_size`：OOM 自动降级天花板，持久化压低以保护显存。
> - `realesrgan`：默认 `batch_size` **24**、`prefetch_factor` **96**、`use_tensorrt` **false**；`video_codec` 为底层主键，`codec` 为兼容别名。
> - `realesrgan.gfpgan_weight` / `gfpgan_batch_size` / `face_det_threshold`：与当前 JSON 一致（如权重 0.7、GFPGAN 批 4、检测阈值 0.7）；`preview` / `report_json` 控制实时预览与性能报告路径。
> - `output`：最终合并采用 **copy-by-default** 策略（未指定 `--output-codec/crf/preset` 时自动 `-c:v copy`，不重新编码），`audio_format: smart`、`audio_codec: copy` 等为默认智能/无损回写策略。
> - `temp_files`：分段与中间文件命名前缀；`logging`：日志级别与输出目标。

完整配置说明见 [GUIDE.md](GUIDE.md)。

---

## FIX-C：分段 timescale 归一化

**问题背景：** 多个独立 ffmpeg 子进程编码的分段在 MP4 timescale 上存在微差（即使编解码参数完全相同），导致 `concat` demuxer 触发 `AVEROR_EXIT(254)` 合并失败。

**解决方案：** 最终合并前对所有分段执行零损耗 remux 归一化：
1. 自动检测分段编解码器（仅 H.264/H.265 需要）
2. 统一 MP4 时间基为 90 kHz（`-video_track_timescale 90000`）
3. 三重校验（rc=0 + 文件非空 + 视频帧可解码）
4. 其他编码格式（VP9/AV1/ProRes 等）自动跳过

> **注意：** 不再使用 `-bsf:v dump_extra`。该 BSF 专为 Annex B 格式设计，用于 MP4/AVCC 时会产生畸形 packet 导致视频流静默丢失——这是早期版本"合并后只有音频"的直接原因。

可通过 `--skip-seg-normalize` 跳过此步骤（调试用，或已确认分段 timescale 完全一致时）。

---

## 流水线报告（--report）

优化版支持端到端流水线 JSON 报告，包含：

```json
{
  "version": "2.0.0",
  "timestamp": "2026-05-14T12:00:00",
  "success": true,
  "input": "/path/to/input.mp4",
  "output": "/path/to/output.mp4",
  "mode": "interpolate_then_upscale",
  "elapsed_seconds": 1234.56,
  "elapsed_human": "20分34秒",
  "input_size_mb": 150.5,
  "output_size_mb": 890.2,
  "size_ratio": 5.914,
  "input_duration_s": 600.0,
  "output_duration_s": 600.5,
  "gpu_peak_memory_gb": 7.824,
  "env": {
    "python": "3.10.12",
    "torch": "2.7.0+cu128",
    "gpu_name": "NVIDIA RTX 3080",
    "gpu_memory_gb": 10.0,
    "ffmpeg": true
  },
  "params": {
    "interpolation_factor": 2,
    "upscale_factor": 2,
    "mode": "interpolate_then_upscale",
    "use_tensorrt_ifrnet": true,
    "use_tensorrt_esrgan": true,
    "face_enhance": false,
    "batch_size_esrgan": 24,
    ...
  }
}
```

使用方式：
```bash
python src/main_video_optimized.py -i input.mp4 -o output.mp4 --report report.json
```

成功或失败均会写出；批量模式下可为每个视频添加 `extra` 字段写入批量索引。

---

## 独立调用处理器

两个处理器均可脱离主流程独立运行，方便单独调试：

```bash
# ── IFRNet 独立插帧（v6.1 处理器，对接 v6.3.3 后端）──────────────────────────

# 基础 2x 插帧（使用配置中 model_name 自动拼路径）
python src/processors/ifrnet_processor_v6_1_single.py \
    -i input.mp4 -o output_2x.mp4

# 指定模型名称 + 4x 插帧
python src/processors/ifrnet_processor_v6_1_single.py \
    -i input.mp4 -o output_4x.mp4 \
    --interpolation-factor 4 --ifrnet-model IFRNet_L_Vimeo90K

# 启用 TRT 加速 + 指定缓存目录
python src/processors/ifrnet_processor_v6_1_single.py \
    -i input.mp4 -o output_2x.mp4 \
    --use-tensorrt --trt-cache-dir /data/trt_engines

# 关闭 compile（短视频跳过 1–3 分钟预热）
python src/processors/ifrnet_processor_v6_1_single.py \
    -i input.mp4 -o output_2x.mp4 \
    --no-compile --no-cuda-graph

# ── Real-ESRGAN 独立超分（优化版）─────────────────────────────────────────────

# 基础 4x 超分
python src/processors/realesrgan_processor_video_optimized.py \
    -i input.mp4 -o output_4x.mp4 --upscale-factor 4

# 启用 TRT 加速 + 指定缓存目录
python src/processors/realesrgan_processor_video_optimized.py \
    -i input.mp4 -o output_4x.mp4 \
    --use-tensorrt --trt-cache-dir /data/trt_engines

# 开启人脸增强，精细控制 GFPGAN + 检测阈值
python src/processors/realesrgan_processor_video_optimized.py \
    -i input.mp4 -o output.mp4 \
    --face-enhance --gfpgan-model 1.4 \
    --gfpgan-weight 0.7 --gfpgan-batch-size 4 \
    --face-det-threshold 0.7

# GFPGAN TensorRT 子进程加速 + 性能报告
python src/processors/realesrgan_processor_video_optimized.py \
    -i input.mp4 -o output.mp4 \
    --face-enhance --gfpgan-trt \
    --report report.json

# 实时预览 + 自适应批处理
python src/processors/realesrgan_processor_video_optimized.py \
    -i input.mp4 -o output.mp4 \
    --preview --preview-interval 30

# 禁用所有 GPU 加速（调试模式）
python src/processors/realesrgan_processor_video_optimized.py \
    -i input.mp4 -o output.mp4 \
    --no-compile --no-cuda-graph --no-fp16
```

---

## TRT Engine 缓存机制

优化版继承并增强了统一 TRT 缓存目录，解决了早期版本中每个分段、每个视频各自重新构建 Engine 的问题。

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

**Engine 文件命名规则：** `<模型名称>_B<batch>_H<height>_W<width>_fp16_sm<arch>_<gpu>.trt`（示例：`ifrnet_S_B24_H576_W736_fp16_sm86_rtx3080.trt`）。IFRNet 与 ESRGan 的 Engine 文件名不同，共用同一目录不会冲突。文件名含 GPU SM 架构标记，跨 GPU 迁移时自动重建。

**实际效果：**
- 同一视频：所有分段共用同一 Engine，仅第一个分段需要构建（约 3–10 分钟），后续分段直接加载。
- 跨视频：只要分辨率和批大小不变，Engine 在不同视频间复用，无需重新构建。
- TRT 版本升级后需手动删除 `.trt_cache/` 目录以触发重建。
- GPU 更换后 Engine 自动检测 SM 不匹配并重建（无需手动删除）。

---

## GPU 硬件型号探测

Real-ESRGAN 后端内置 GPU 硬件型号静态表（`_GPU_PROFILES_TABLE`），覆盖 20+ 常见 GPU 型号（H100/H800 → GTX 1080），用于：
- 自动选择最优 NVENC 编码器（通过 `HardwareCapability.best_encoder()`）
- GPU 算力分级（tier 值）
- PCIe 带宽参考

探测结果在初始化时打印（如 `[FIX-NVENC-UNIFIED] HW Profile: ...`），不依赖运行时 NVML 检测。

---

## 常见问题

**Q: CUDA out of memory？**  
A: 优化版支持 OOM 自动降级，会自动减小批量大小并持久化上限（`max_batch_size`）。如仍失败，尝试以下组合：`--tile-size 512 --batch-size-esrgan 2 --no-cuda-graph-esrgan`。IFRNet 侧也可降低 `--batch-size-ifrnet 2`。

**Q: 启动时报 `FileNotFoundError`（IFRNet 模型路径为空）？**  
A: 处理器会在加载前校验路径并打印完整提示，请按提示操作：
- 确认权重文件已下载并放置到 `models_IFRNet/checkpoints/IFRNet_S_Vimeo90K.pth`，或
- 命令行传入 `--ifrnet-model IFRNet_S_Vimeo90K`（自动拼接路径），或
- 命令行传入 `--ifrnet-model-path /绝对/路径/model.pth` 直接指定。

**Q: 启动时报 `FileNotFoundError`（realesrgan_video 目录不存在）？**  
A: 优化版的 Real-ESRGAN 后端已重构为独立子项目 `external/realesrgan_video/`。请确认该目录及其所有文件已正确放置。若仍使用旧版后端，请使用 `main_video_v6_single.py` 入口。

**Q: 如何从中断恢复？**  
A: 重新运行完全相同的命令，程序自动读取 `temp/` 下的 `checkpoint.json` 从断点继续。

**Q: 输出视频没有音频？**  
A: 优化版使用智能音频提取（从原始输入一次性提取，最终合并时无损回写）。请确认源视频含音频且 FFmpeg 已正确安装。**若合并后只有音频无视频**，这通常是分段 timescale 不匹配所致——再次运行即可触发 FIX-C 归一化修复。

**Q: "合并后只有音频无视频"？**  
A: 这是早期版本 `-bsf:v dump_extra` 导致的已知问题。优化版已通过 FIX-C 机制修复：改用 `-video_track_timescale 90000` 统一时间基替代 BSF，并在合并后增加视频流 ffprobe 验证。如果仍遇到，请使用 `--skip-seg-normalize` 跳过归一化并报告问题。

**Q: torch.compile 报错？**  
A: 需要 PyTorch 2.0+，首次编译约需 1–3 分钟。如遇兼容性问题，IFRNet 侧传入 `--no-compile-ifrnet`，ESRGan 侧传入 `--no-compile-esrgan`。注意 `use_tensorrt=true` 时 compile 和 CUDA Graph 会被自动关闭。当前 `default_config.json` 中 IFRNet 默认 **关闭** compile/cuda-graph、**开启** TRT；Real-ESRGAN 默认 **开启** compile/cuda-graph、TRT 默认关闭。

**Q: TRT Engine 每次都重新构建，没有复用？**  
A: 检查 `--trt-cache-dir` 是否指向一个稳定固定的目录，而不是每次变化的临时路径。或在配置文件中显式设置 `"paths.trt_cache_dir": "/stable/path/.trt_cache"`。另外，若输入视频分辨率或 `batch_size` 发生变化，Engine 文件名不同，会触发新建。优化版还增加了 GPU SM 架构检测，跨 GPU 迁移会自动重建。

**Q: 临时文件占用空间太大？**  
A: 传入 `--auto-cleanup` 或设置 `"auto_cleanup_temp": true`，处理完成后自动清理。临时文件通常为原视频大小的 10–20 倍。

**Q: 使用 face_enhance 需要额外下载什么？**  
A: 需要 GFPGAN 模型文件，放置到 `models_GFPGAN/` 目录。默认使用 `GFPGANv1.4.pth`，也可通过 `--gfpgan-model` 切换版本：
- `GFPGANv1.3.pth` — 轻量版
- `GFPGANv1.4.pth` — 默认，推荐
- `RestoreFormer.pth` — 细节更丰富

下载地址：https://github.com/TencentARC/GFPGAN/releases

**Q: face_enhance 开启后很慢或 OOM？**  
A: GFPGAN 每帧需额外做人脸检测 + StyleGAN2 解码。建议：
- 用 `--gfpgan-batch-size 4` 减小批大小；底层 OOM 时会自动降级并持久化。
- 用 `--face-det-threshold 0.7` 过滤低置信度检测，减少无效 GFPGAN 推理。
- 用 `--gfpgan-trt` 启用 GFPGAN TRT 子进程加速（注意会自动禁用 FP16）。
- `realesr-animevideov3` 等动漫模型不支持 face_enhance，底层会自动禁用。

**Q: 什么是自适应批处理？如何禁用？**  
A: 优化版默认根据人脸密度动态调整读帧批大小。人脸密集帧自动缩小批量以控制显存峰值，人脸稀疏帧自动扩大批量以提升吞吐。传入 `--no-adaptive-batch-esrgan` 或在配置中设置 `"adaptive_batch": false` 可禁用，使用固定 `batch_size`。

**Q: `--gfpgan-trt` 启用后输出有噪斑？**  
A: GFPGAN TRT 模式下 FP16 可能产生溢出噪斑。优化版已自动处理——启用 `--gfpgan-trt` 时会自动禁用 FP16 改用 FP32。如仍有问题，检查是否手动覆盖了 FP16 设置。

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

**Q: 如何只打印配置不实际处理？**  
A: 传入 `--dry-run`，将只打印环境信息和完整配置摘要然后退出。

**Q: 旧版 `main_video_v6_single.py` 还能用吗？**  
A: 可以。旧版主入口对接的是 `external/Real-ESRGAN/inference_realesrgan_video_v6_1_single.py` 后端，仍保留在项目中供参考。但推荐迁移到 `main_video_optimized.py`，后者对接性能更优的 `realesrgan_video` 深度模块化架构。

**Q: IFRNet v6.3.3 相比旧版有哪些提升？**  
A: **v6.3.0** 起：双 transfer stream（`stream_h2d` / `stream_d2h`）+ CudaEventPool，实现计算与 H2D/D2H 三流重叠；GPU 监测线程给出 batch 建议。**v6.3.1+** 推理线程与双槽 H2D 预取。**v6.3.2+** 强化编码瓶颈识别、PinnedPool 上限与队列深度，避免错误堆队列。**v6.3.3** 将 RETUNE 统计移到段完成后、增强日志可追溯性，并改进软编并行（Linux 下可探测 CPU/内存；Windows 可能回退为保守默认）与 NVENC pipe 参数。默认 `batch_size` 以 `default_config.json` 为准（当前 24），显存充足可自行调高。

**Q: IFRNet `stream_transfer` 属性报错？**  
A: v6.3.0 起将 `stream_transfer` 拆分为 `stream_h2d`（预取）和 `stream_d2h`（输出）。如有上层代码直接引用 `processor.stream_transfer`，需改为 `stream_h2d` 或 `stream_d2h` 分别使用。

**Q: 启动时报 PyTorch / NVML 相关内部断言？**  
A: `realesrgan_video` 已在入口设置 `PYTORCH_NVML_BASED_CUDA_CHECK=0` 以避免部分驱动与 NVML 不匹配导致的检测失败。若仍遇 NVML 问题，请检查 NVIDIA 驱动与 CUDA 版本是否与 PyTorch wheel 一致。

**Q: `--skip-seg-normalize` 是做什么用的？**  
A: FIX-C 机制默认在合并前对所有分段执行 timescale 归一化。如果确认分段 timescale 已一致（例如手动预处理过），或需要调试合并问题，可用 `--skip-seg-normalize` 跳过此步骤以节省时间。

**Q: `--report` 报告包含哪些字段？**  
A: 见上方"流水线报告"章节。包含：版本、时间戳、成功标志、输入输出路径、处理模式、总耗时、输入输出大小/时长、体积比、GPU 峰值显存、运行环境、关键 CLI 参数快照等。批量模式下还可写入 `extra` 字段。

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
