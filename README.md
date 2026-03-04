# Video Enhancement — 视频增强处理系统

> **当前版本**: v5（单卡优化版）｜IFRNet 后端 v5 / Real-ESRGAN 后端 v6  
> **最后更新**: 2026-03-04

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
| 🎵 **智能音频处理** | 自动检测音频编码，无损时直接 copy，否则转码 AAC |
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
├── CHANGELOG.md                    # 版本历史与升级指南
│
├── requirements.txt                # Python 依赖
├── setup_project.py                # 项目初始化脚本
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
│   │   ├── ifrnet_processor_v5_single.py           # ★ IFRNet 处理器 v5（对接 IFRNet v5 后端）
│   │   ├── realesrgan_processor_video_v5_single.py # ★ Real-ESRGAN 处理器 v5（对接 ESRGAN v6 后端）
│   │   └── ...                                     # 历史版本处理器
│
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
│       ├── inference_realesrgan_video_v5_single.py  # ⚠️  ESRGAN v5++ 后端（历史，非face_enhance场景可用）
│       └── realesrgan/                              # Real-ESRGAN 库
│
├── models_IFRNet/
│   └── checkpoints/
│       ├── IFRNet_S_Vimeo90K.pth   # IFRNet Small（推荐，速度快）
│       └── IFRNet_L.pth            # IFRNet Large（质量更高）
│
├── models_RealESRGAN/
│   ├── realesr-general-x4v3.pth    # 通用 4x 超分（推荐）
│   ├── RealESRGAN_x4plus.pth       # 写实 4x 超分
│   ├── RealESRGAN_x2plus.pth       # 写实 2x 超分
│   └── RealESRGANv2-animevideo-xsx2.pth  # 动漫 2x 超分
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
- FFmpeg 5.0+
- 16 GB RAM、100 GB 可用磁盘空间

**推荐配置**
- NVIDIA RTX 3080 / 4080 或更高（10 GB+ VRAM）
- CUDA 12.1+、PyTorch 2.0+（启用 torch.compile）
- 32 GB RAM、200 GB+ SSD

### 2. 安装依赖

```bash
# 安装 PyTorch（根据 CUDA 版本选择）
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装项目依赖
pip install -r requirements.txt

# 安装系统 FFmpeg（Ubuntu/Debian）
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
```

其他可选模型见 [FILE_LIST.md](FILE_LIST.md)。

### 4. 配置路径

编辑 `config/default_config.json`，修改 `base_dir` 为你的实际项目路径：

```json
{
  "paths": {
    "base_dir": "/your/project/path",
    "output_dir": "/your/project/path/output",
    "temp_dir": "/your/project/path/temp"
  }
}
```

### 5. 运行

```bash
# 单视频处理（先插帧再超分，默认）
python src/main_video_v5_single.py -i input.mp4

# 先超分再插帧
python src/main_video_v5_single.py -i input.mp4 -m upscale_then_interpolate

# 自定义倍数
python src/main_video_v5_single.py -i input.mp4 \
  --interpolation-factor 4 --upscale-factor 4

# 批量处理
python src/main_video_v5_single.py --input-dir ./videos --batch

# 指定输出目录
python src/main_video_v5_single.py -i input.mp4 -o ./my_output
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
      "use_fp16": true,
      "use_compile": true,
      "use_cuda_graph": true,
      "use_hwaccel": true
    },
    "realesrgan": {
      "tile_size": 0,
      "batch_size": 8,
      "prefetch_factor": 16,
      "use_compile": false,
      "use_hwaccel": true,
      "face_enhance": false,
      "gfpgan_model": "1.4",
      "gfpgan_weight": 0.5,
      "gfpgan_batch_size": 8
    }
  }
}
```

> `batch_size` 默认值 v6 起调整为 8（原 v5 为 4），可充分利用 T4/RTX 3080 等 15 GB+ 显存。  
> `gfpgan_*` 参数仅在 `face_enhance: true` 时生效，`gfpgan_batch_size` 可防止人脸密集场景 OOM。

完整配置说明见 [GUIDE.md](GUIDE.md)。

---

## 常见问题

**Q: CUDA out of memory？**  
A: v5 支持 OOM 自动降级，会自动减小批量大小。如仍失败，在配置中设置 `"tile_size": 512` 并将 `"batch_size"` 调小至 1–2。

**Q: 如何从中断恢复？**  
A: 重新运行完全相同的命令，程序自动读取 `temp/` 下的 `checkpoint.json` 从断点继续。

**Q: 输出视频没有音频？**  
A: v5 使用智能音频提取，无损时直接 copy，否则转码 AAC。请确认源视频含音频且 FFmpeg 已正确安装。

**Q: torch.compile 报错？**  
A: 需要 PyTorch 2.0+，首次编译约需 1–3 分钟。如遇兼容性问题，设置 `"use_compile": false` 禁用。

**Q: 临时文件占用空间太大？**  
A: 设置 `"auto_cleanup_temp": true`，或处理完成后在提示时输入 `Y`。临时文件通常为原视频大小的 10–20 倍。

---

## 技术栈

- **IFRNet** — 基于双向光流的高质量视频插帧
- **Real-ESRGAN** — 基于 GAN 的真实世界超分辨率
- **PyTorch 2.0+** — FP16 / torch.compile / CUDA Graph
- **FFmpeg** — 视频处理（支持 NVDEC/NVENC）
- **OpenCV** — 帧级图像处理

---

## 许可证

本项目源代码遵循 **MIT 许可证**。

集成的外部项目请遵循各自许可证：
- [IFRNet](https://github.com/ltkong218/IFRNet) — MIT License
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — BSD-3-Clause License

## 致谢

- [IFRNet](https://github.com/ltkong218/IFRNet) by ltkong218
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by xinntao
- FFmpeg 项目 / PyTorch 团队
