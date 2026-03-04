# 文件清单 — Video Enhancement v5

> **最后更新**: 2026-03-04

本文档列出项目的所有文件，标注各文件的用途、状态和主要接口。

---

## 根目录

| 文件 | 状态 | 说明 |
|------|------|------|
| `README.md` | ✅ 当前 | 项目主文档、快速开始、功能概览 |
| `GUIDE.md` | ✅ 当前 | 详细安装、配置、使用指南 |
| `PROJECT_SUMMARY.md` | ✅ 当前 | 架构设计与技术总结 |
| `FILE_LIST.md` | ✅ 当前 | 本文件，文件清单 |
| `CHANGELOG.md` | ✅ 当前 | 版本历史与升级说明 |
| `requirements.txt` | ✅ 当前 | Python 依赖清单 |
| `setup_project.py` | ✅ 当前 | 项目初始化脚本（创建目录结构） |
| `run.py` | ⚠️ 旧版 | 向后兼容入口，对接 `src/main.py`（v1） |

---

## `config/`

| 文件 | 状态 | 说明 |
|------|------|------|
| `config/default_config.json` | ✅ 当前 | 主配置文件，包含全部 v5 参数 |

### `default_config.json` 结构

```
processing        处理模式、插帧/超分倍数、分段时长、清理设置
paths             项目根目录、输入/输出/临时/日志目录
models.ifrnet     IFRNet 模型路径、推理参数、v5 硬件加速参数
models.realesrgan Real-ESRGAN 模型路径、推理参数、v5 硬件加速参数
output            最终输出编码器、CRF、预设、音频设置
temp_files        临时文件命名前缀/后缀
```

---

## `src/` — 源代码

### 主程序入口

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/main_video_v5_single.py` | ★ **当前主入口** | VideoProcessor v5，分段直连流水线 |
| `src/main_video_v3.py` | ⚠️ 历史 | VideoProcessor v3，保留参考 |
| `src/main_video_v2.py` | ⚠️ 历史 | VideoProcessor v2，保留参考 |
| `src/main_video.py` | ⚠️ 历史 | VideoProcessor v1 video 版，保留参考 |
| `src/main_v2.py` | ⚠️ 历史 | v2 主程序，保留参考 |
| `src/main.py` | ⚠️ 历史 | v1 主程序（被 run.py 引用） |
| `src/__init__.py` | — | 包初始化 |

> ⚠️ **历史文件**：仅保留供代码参考，不建议直接使用。生产请使用 `main_video_v5_single.py`。

### `src/processors/` — 处理器模块

| 文件 | 状态 | 说明 |
|------|------|------|
| `ifrnet_processor_v5_single.py` | ★ **当前** | IFRNet 处理器 v5，FP16 + CUDA Graph + NVDEC/NVENC |
| `realesrgan_processor_video_v5_single.py` | ★ **当前** | Real-ESRGAN 处理器 v5，批量推理 + 预取 + NVDEC/NVENC |
| `ifrnet_processor_v3.py` | ⚠️ 历史 | IFRNet 处理器 v3 |
| `ifrnet_processor_v2.py` | ⚠️ 历史 | IFRNet 处理器 v2 |
| `ifrnet_processor.py` | ⚠️ 历史 | IFRNet 处理器 v1 |
| `realesrgan_processor_video_v3.py` | ⚠️ 历史 | Real-ESRGAN 视频处理器 v3 |
| `realesrgan_processor_video_v2.py` | ⚠️ 历史 | Real-ESRGAN 视频处理器 v2 |
| `realesrgan_processor_video.py` | ⚠️ 历史 | Real-ESRGAN 视频处理器 v1 |
| `realesrgan_processor_v2.py` | ⚠️ 历史 | Real-ESRGAN 图像处理器 v2 |
| `realesrgan_processor.py` | ⚠️ 历史 | Real-ESRGAN 图像处理器 v1 |
| `__init__.py` | — | 包初始化 |

#### `ifrnet_processor_v5_single.py` 主要接口

```python
class IFRNetProcessor:
    def __init__(config: Config)
    def process_video_segments(input_video: str) -> List[str]
        # 完整视频分段插帧，返回插帧后的分段路径列表
    def process_segments_directly(input_segments: List[str], video_name: str) -> List[str]
        # 直接接收上游分段列表，执行插帧，返回处理后分段列表
```

#### `realesrgan_processor_video_v5_single.py` 主要接口

```python
class RealESRGANVideoProcessor:
    def __init__(config: Config)
    def process_video(input_video: str, output_video: str) -> bool
        # 完整单视频超分（含合并）
    def process_video_segments(input_video: str) -> List[str]
        # 完整视频分段超分，返回超分后的分段路径列表
    def process_segments_directly(input_segments: List[str], video_name: str) -> List[str]
        # 直接接收上游分段列表，执行超分，返回处理后分段列表
```

### `src/utils/` — 工具模块

| 文件 | 状态 | 说明 |
|------|------|------|
| `config_manager.py` | ✅ 当前 | 配置文件加载、验证、多级键路径读写 |
| `video_utils.py` | ✅ 当前 | 视频分割、合并、音频提取、编解码工具集 |
| `video_fixer.py` | ✅ 当前 | 损坏视频修复（remux、关键帧、时间戳） |
| `output_filter.py` | ✅ 当前 | FFmpeg/Real-ESRGAN 分块输出过滤 |
| `__init__.py` | — | 包初始化 |

#### `config_manager.py` 主要接口

```python
class Config:
    def __init__(config_path: Optional[str] = None)
    def get(*keys, default=None) -> Any           # 嵌套键路径读取
    def set(*keys, value)                          # 嵌套键路径写入
    def get_temp_dir(subdir: str = "") -> Path     # 获取临时目录
    def get_input_videos() -> List[str]            # 解析输入视频列表
    def get_output_path(input_path: str, suffix="_processed") -> str
    def get_section(section_key: str, default=None) -> Any
    def save(output_path: Optional[str] = None)
```

#### `video_utils.py` 主要接口

```python
class VideoInfo:
    width: int; height: int; fps: float
    duration: float; frame_count: int
    codec: str; has_audio: bool; audio_codec: str

def smart_extract_audio(video_path, temp_dir) -> Optional[str]
def extract_audio(video_path, audio_output, config=None) -> bool
def add_audio_to_video(video_path, audio_path, output_path, ...) -> bool
def get_video_duration(video_path) -> Optional[float]
def verify_video_integrity(video_path) -> bool
def split_video_by_time(input_video, output_dir, segment_duration) -> List[str]
def merge_videos(video_list, output_path, ...) -> bool
def merge_videos_by_codec(video_list, output_path, audio_path=None, config=None) -> bool
def get_video_codec(video_file) -> str
def encode_video(input_path, output_path, ...) -> bool
def format_time(seconds) -> str
```

---

## `external/` — 外部后端

### `external/IFRNet/`

| 文件 | 状态 | 说明 |
|------|------|------|
| `process_video_v5_single.py` | ★ **当前后端** | FP16 + CUDA Graph + TensorRT + NVDEC/NVENC |
| `process_video_v5.py` | ⚠️ 历史（多卡） | 多 GPU 版本 |
| `process_video_v4.py` | ⚠️ 历史 | 批量推理版本 |
| `process_video_v3.py` | ⚠️ 历史 | 分段处理版本 |
| `process_video.py` | ⚠️ 历史 | 基础版本 |
| `process_video_v5_single_bak.py` | 🗑️ 备份 | 开发备份，可删除 |
| `process_video_v5_single_bak2.py` | 🗑️ 备份 | 开发备份，可删除 |
| `models/IFRNet.py` | ✅ 模型定义 | IFRNet Large 架构 |
| `models/IFRNet_L.py` | ✅ 模型定义 | IFRNet Large 变体 |
| `models/IFRNet_S.py` | ✅ 模型定义 | IFRNet Small 架构 |
| `utils.py` | ✅ 工具 | IFRNet 内部工具函数 |
| `loss.py` | — 训练用 | 损失函数（推理时不需要） |
| `datasets.py` | — 训练用 | 数据集加载（推理时不需要） |
| `train_vimeo90k.py` | — 训练用 | Vimeo90K 训练脚本 |
| `train_gopro.py` | — 训练用 | GoPro 训练脚本 |
| `liteflownet/` | — 依赖 | LiteFlowNet 相关组件 |
| `benchmarks/` | — 评测 | 多个基准测试脚本 |

### `external/Real-ESRGAN/`

| 文件 | 状态 | 说明 |
|------|------|------|
| `inference_realesrgan_video_v5_single.py` | ★ **当前后端** | FP16 + 批量推理 + 预取 + NVDEC/NVENC |
| `inference_realesrgan_video_v6_single.py` | 🧪 实验 | v6 进一步优化，待稳定 |
| `inference_realesrgan_video_v5.py` | ⚠️ 历史（多卡） | 多 GPU 版本 |
| `inference_realesrgan_video_v4.py` | ⚠️ 历史 | 批量推理版本 |
| `inference_realesrgan_video_v3.py` | ⚠️ 历史 | 分段处理版本 |
| `inference_realesrgan_video.py` | ⚠️ 历史 | 基础版本 |
| `inference_realesrgan_video_original.py` | ⚠️ 历史 | 原版脚本 |
| `inference_realesrgan.py` | ⚠️ 图像版 | 单图像超分脚本 |
| `inference_realesrgan_video_v5_single_bak*.py` | 🗑️ 备份 | 开发备份（共 13 个），可删除 |
| `realesrgan/` | ✅ 库 | Real-ESRGAN Python 包 |
| `setup.py` | ✅ 安装 | Real-ESRGAN 包安装脚本 |
| `docs/` | ✅ 文档 | 原版文档（训练、模型等） |

---

## 模型文件

### `models_IFRNet/checkpoints/`

| 文件 | 大小 | 说明 | 下载地址 |
|------|------|------|---------|
| `IFRNet_S_Vimeo90K.pth` | ~10 MB | **★ 推荐**，Small 版，速度快 | [GitHub Releases](https://github.com/ltkong218/IFRNet/releases/download/v1.0/IFRNet_S.pth) |
| `IFRNet_L.pth` | ~25 MB | Large 版，质量更高 | [GitHub Releases](https://github.com/ltkong218/IFRNet/releases/download/v1.0/IFRNet_L.pth) |

### `models_RealESRGAN/`

| 文件 | 大小 | 适用场景 | 下载地址 |
|------|------|---------|---------|
| `realesr-general-x4v3.pth` | ~64 MB | **★ 推荐**，通用 4x，写实+动漫 | [GitHub Releases](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth) |
| `RealESRGAN_x4plus.pth` | ~64 MB | 写实 4x | [GitHub Releases](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) |
| `RealESRGAN_x2plus.pth` | ~64 MB | 写实 2x | [GitHub Releases](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) |
| `RealESRGANv2-animevideo-xsx2.pth` | ~64 MB | 动漫视频 2x | [GitHub Releases](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx2.pth) |

---

## 运行时目录

| 目录 | 说明 |
|------|------|
| `output/` | 最终处理结果输出（mp4 文件） |
| `temp/` | 处理临时文件（分段、音频），可随时清理 |
| `logs/` | 运行日志 |

### `temp/` 子目录结构（运行时生成）

```
temp/
├── {video_name}_ifrnet/
│   ├── checkpoint.json       # 断点记录（已完成分段列表）
│   ├── segment_000.mp4       # 插帧后的分段
│   ├── segment_001.mp4
│   └── ...
└── {video_name}_esrgan/
    ├── checkpoint.json       # 断点记录
    ├── processed_000.mp4     # 超分后的分段
    ├── processed_001.mp4
    └── ...
```

---

## 可清理的开发备份文件

以下文件为开发过程中留下的备份，不影响正常使用，可酌情删除以节省空间：

```
external/IFRNet/process_video_v5_single_bak.py
external/IFRNet/process_video_v5_single_bak2.py
external/Real-ESRGAN/inference_realesrgan_video_v5_single_bak.py
external/Real-ESRGAN/inference_realesrgan_video_v5_single_bak2.py
... （共约 15 个 _bak*.py 文件）
```

---

## 文件状态图例

| 图标 | 含义 |
|------|------|
| ★ | 当前活跃使用的主要文件 |
| ✅ | 正常状态，当前版本 |
| ⚠️ | 历史版本，保留供参考 |
| 🧪 | 实验性，尚未稳定 |
| 🗑️ | 开发备份，可安全删除 |
| — | 辅助/训练用，推理时不需要 |
