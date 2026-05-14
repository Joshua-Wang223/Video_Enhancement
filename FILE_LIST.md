# 文件清单 — Video Enhancement（当前流水线）

> **最后更新**: 2026-05-14

本文档列出项目的核心文件，标注各文件的用途、状态和主要接口。生产环境请以 **`src/main_video_optimized.py`** + **`config/default_config.json`** 为准。

---

## 根目录

| 文件 | 状态 | 说明 |
|------|------|------|
| `README.md` | ✅ 当前 | 项目主文档、快速开始、功能概览 |
| `GUIDE.md` | ✅ 当前 | 详细安装、配置、使用指南 |
| `PROJECT_SUMMARY.md` | ✅ 当前 | 架构设计与技术总结 |
| `FILE_LIST.md` | ✅ 当前 | 本文件，文件清单 |
| `QUICK_UPGRADE.md` | ✅ 当前 | 快速升级说明 |
| `V2_UPGRADE_GUIDE.md` | ✅ 当前 | v2 升级指南（历史迁移参考） |
| `requirements.txt` | ✅ 当前 | Python 依赖清单 |
| `setup_project.py` | ✅ 当前 | 项目初始化脚本（创建目录结构） |
| `run.py` | ⚠️ 旧版 | 向后兼容入口，对接 `src/main.py`（v1） |

---

## `config/`

| 文件 | 状态 | 说明 |
|------|------|------|
| `config/default_config.json` | ✅ **默认真相源** | 主配置：处理模式、路径、模型、输出、临时文件、日志等 |

### `default_config.json` 结构

```
processing        处理模式、插帧/超分倍数、分段时长、批量模式、自动修复与临时清理
paths             base_dir、输入/输出/临时/日志、trt_cache_dir、models_gfpgan_dir（GFPGAN）
models.ifrnet     IFRNet 模型与推理参数（含 FP16 / compile / CUDA Graph / TensorRT / NVDEC·NVENC）
models.realesrgan Real-ESRGAN + GFPGAN 推理参数（tile、批处理、预取、人脸增强、预览与报告等）
output            最终合并阶段：编码器、CRF、预设、像素格式、音频策略（含 use_copy copy-by-default）
temp_files        分段/处理后文件命名前缀与临时后缀
logging           日志级别、是否写入文件/控制台
```

---

## `src/` — 源代码

### 主程序入口

| 文件 | 状态 | 说明 |
|------|------|------|
| `src/main_video_optimized.py` | ★ **当前主入口** | 统一调度 IFRNet（`ifrnet_processor_v6_1_single`）+ Real-ESRGAN DS（`realesrgan_processor_video_optimized` → `external/realesrgan_video`）。**新增特性：** FIX-C 分段 timescale 归一化（`_normalize_segs_for_copy`）、`--report` 流水线报告（`_write_final_report`）、`--skip-seg-normalize` 跳过开关 |
| `src/main_video_v6_single.py` | ⚠️ 对照 | v6 单卡主流程：对接 `ifrnet_processor_v6_single` + `realesrgan_processor_video_v6_single`（底层分别为 IFRNet v6.2.x 脚本与 `external/Real-ESRGAN` 推理脚本） |
| `src/main_video_v5_single.py` | ⚠️ 历史 | VideoProcessor v5，分段流水线 |
| `src/main_video_v3.py` | ⚠️ 历史 | VideoProcessor v3 |
| `src/main_video_v2.py` | ⚠️ 历史 | VideoProcessor v2 |
| `src/main_video.py` | ⚠️ 历史 | VideoProcessor v1 video 版 |
| `src/main_v2.py` | ⚠️ 历史 | v2 主程序 |
| `src/main.py` | ⚠️ 历史 | v1 主程序（被 `run.py` 引用） |
| `src/__init__.py` | — | 包初始化 |

> ⚠️ **历史/对照入口**：日常处理请使用 **`main_video_optimized.py`**。

### `src/processors/` — 处理器模块

| 文件 | 状态 | 说明 |
|------|------|------|
| `ifrnet_processor_v6_1_single.py` | ★ **当前** | IFRNet 插帧；底层 **`external/IFRNet/process_video_v6_3_3_single.py`（v6.3.3）** |
| `realesrgan_processor_video_optimized.py` | ★ **当前** | Real-ESRGAN 视频超分；底层 **`external/realesrgan_video/main.py`**，支持 **`create_video_enhancer` / `run_pipeline_for_video`** 多片段复用 |
| `realesrgan_processor_video_v6_single.py` | ⚠️ 历史 | 对接 `external/Real-ESRGAN` 下各版 `inference_realesrgan_video_*.py` |
| `ifrnet_processor_v6_single.py` | ⚠️ 历史 | 对接较早 IFRNet 脚本（如 `process_video_v6_2_2_single.py`），非 v6.3.3 主线 |
| `ifrnet_processor_v5_single.py` | ⚠️ 历史 | IFRNet 处理器 v5 |
| `realesrgan_processor_video_v5_single.py` | ⚠️ 历史 | Real-ESRGAN 处理器 v5 |
| `ifrnet_processor_v3.py` | ⚠️ 历史 | IFRNet 处理器 v3 |
| `ifrnet_processor_v2.py` | ⚠️ 历史 | IFRNet 处理器 v2 |
| `ifrnet_processor.py` | ⚠️ 历史 | IFRNet 处理器 v1 |
| `realesrgan_processor_video_v3.py` | ⚠️ 历史 | Real-ESRGAN 视频处理器 v3 |
| `realesrgan_processor_video_v2.py` | ⚠️ 历史 | Real-ESRGAN 视频处理器 v2 |
| `realesrgan_processor_video.py` | ⚠️ 历史 | Real-ESRGAN 视频处理器 v1 |
| `realesrgan_processor_v2.py` | ⚠️ 历史 | Real-ESRGAN 图像处理器 v2 |
| `realesrgan_processor.py` | ⚠️ 历史 | Real-ESRGAN 图像处理器 v1 |
| `__init__.py` | — | 包初始化 |

#### `ifrnet_processor_v6_1_single.py` 主要接口

```python
class IFRNetProcessor:
    def __init__(config: Config)
    def process_video_segments(input_video: str) -> List[str]
        # 完整视频分段插帧，返回插帧后的分段路径列表
    def process_segments_directly(input_segments: List[str], video_name: str) -> List[str]
        # 直接接收上游分段列表，执行插帧，返回处理后分段列表
```

#### `realesrgan_processor_video_optimized.py` 主要接口

```python
class RealESRGANVideoProcessor:
    def __init__(config: Config)
    def process_video(input_video: str, output_video: str) -> bool
        # 完整单视频超分（含合并）；内部通过底层 main 模块的 create_video_enhancer / run_pipeline_for_video 复用引擎
    def process_video_segments(input_video: str) -> List[str]
        # 完整视频分段超分，返回超分后的分段路径列表
    def process_segments_directly(input_segments: List[str], video_name: str) -> List[str]
        # 直接接收上游分段列表；首段创建 enhancer，后续片段复用（多片段复用优化）
```

### `src/utils/` — 工具模块

| 文件 | 状态 | 说明 |
|------|------|------|
| `config_manager.py` | ✅ 当前 | 配置文件加载、验证、路径派生（含 `models_gfpgan_dir`、`trt_cache_dir` 等）、多级键读写 |
| `video_utils.py` | ✅ 当前 | 视频分割、合并、音频提取、编解码工具集 |
| `video_fixer.py` | ✅ 当前 | 损坏视频修复（remux、关键帧、时间戳） |
| `output_filter.py` | ✅ 当前 | FFmpeg/Real-ESRGAN 分块输出过滤 |
| `PyTorch_NVML_Test.py` | 🧪 辅助 | PyTorch / NVML 环境探测脚本（开发调试用） |
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

def smart_extract_audio(video_path, temp_dir, stream_index=0, overwrite=False, ...) -> Optional[str]
def extract_audio(video_path, audio_output, config=None) -> bool
def add_audio_to_video(video_path, audio_path, output_path, config=None) -> bool
def get_video_duration(video_path) -> Optional[float]
def verify_video_integrity(video_path) -> bool
def split_video_by_time(input_video, output_dir, segment_duration, reuse_existing=True) -> List[str]
def merge_videos(video_files, output_path, audio_path=None, config=None, ...) -> str
def merge_videos_by_codec(file_list, output_path, audio_path=None, *, config=None, ...) -> bool
def get_video_codec(video_file) -> str
def encode_video(input_path, output_path, ...) -> bool
def format_time(seconds) -> str
```

---

## `external/` — 外部后端

### `external/IFRNet/`

| 文件 | 状态 | 说明 |
|------|------|------|
| `process_video_v6_3_3_single.py` | ★ **当前后端** | 单卡 IFRNet 推理主实现（**v6.3.3**；当前处理器默认对接） |
| `process_video_v6_3_2_single.py` | ⚠️ 历史 | v6.3.2 迭代版本 |
| `process_video_v6_3_1_single.py` | ⚠️ 历史 | v6.3.1 迭代版本 |
| `process_video_v6_3_0_single.py` | ⚠️ 历史 | v6.3.0（较早迭代，非当前主线） |
| `process_video_v6_2_*_single.py` 等 | ⚠️ 历史 | v6.2.x 系列脚本 |
| `process_video_v5_single.py` | ⚠️ 历史 | v5 单卡后端 |
| `process_video_v5.py` | ⚠️ 历史 | v5 多卡 |
| `process_video_v4.py` | ⚠️ 历史 | 批量推理版本 |
| `process_video_v3.py` | ⚠️ 历史 | 分段处理版本 |
| `process_video.py` | ⚠️ 历史 | 基础版本 |
| `process_video_v5_single_bak*.py` | 🗑️ 备份 | 开发备份，可删除 |
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

### `external/realesrgan_video/` — Real-ESRGAN 优化流水线（当前超分后端）

| 文件 | 状态 | 说明 |
|------|------|------|
| `main.py` | ★ **当前后端** | CLI、`main_optimized`、`create_video_enhancer()`、`run_pipeline_for_video(enhancer, …)（多片段引擎复用）；`_HWProfile` / `_GPU_PROFILES_TABLE`（硬件型号探测）；`PreviewWriter`（子进程 GUI 检测 + 实时预览） |
| `pipeline.py` | ✅ 当前 | 深度流水线（`DeepPipelineOptimizer` 等） |
| `ffmpeg_io.py` | ✅ 当前 | FFmpeg 读帧/写帧封装 |
| `config.py` | ✅ 当前 | 模型路径与常量（与项目根目录 `models_RealESRGAN` / `models_GFPGAN` 解析一致） |
| `realesrgan_utils.py` | ✅ 当前 | 超分构建与元数据辅助 |
| `tensorrt_accel.py` | ✅ 当前 | TensorRT 加速封装 |
| `gfpgan_subprocess.py` | ✅ 当前 | GFPGAN 子进程 / TRT 子进程相关 |
| `face_utils.py` | ✅ 当前 | 人脸检测与增强辅助 |
| `async_dispatcher.py` | ✅ 当前 | 异步调度组件 |
| `__init__.py` | — | 包初始化 |
| `*_bak*.py`、`main - Copy.py` 等 | 🗑️ 备份 | 开发备份，可酌情删除 |

#### `main.py` 中与处理器对接的核心函数

```python
def create_video_enhancer(args):
    # 一次性加载 SR / GFPGAN / TRT 等重型组件，供多片段复用

def run_pipeline_for_video(enhancer, input_video, output_video):
    # 使用已创建的 enhancer 处理单个视频文件
```

#### `main.py` 新增核心组件

```python
@dataclass
class _HWProfile:
    gpu_name: str; gpu_tier: float; has_nvdec: bool
    has_nvenc: bool; pcie_bw_gbs: float; cpu_cores: int

# _GPU_PROFILES_TABLE: 20+ GPU 型号（H100 → GTX 1080）
# _detect_hw_profile(device) -> _HWProfile

class PreviewWriter:
    # 包装 FFmpegWriter，支持实时预览
    # 子进程 GUI 检测防止 Qt abort 杀死主进程
    # 按 'q' 键退出预览而不中断流水线
```

### `external/Real-ESRGAN/`

| 文件 | 状态 | 说明 |
|------|------|------|
| `inference_realesrgan_video_v6_single.py`、`inference_realesrgan_video_v6_*_single.py` | ⚠️ 历史/对照 | 非 DS 管线的视频推理脚本；**`main_video_v6_single`** 等入口仍可能引用 |
| `inference_realesrgan_video_v6_3_optimized.py`、`inference_realesrgan_video_v6_4_optimized.py` | ⚠️ 历史/实验 | 优化版单文件推理（与 `realesrgan_video` 并行演进） |
| `inference_realesrgan_video_v5_single.py` | ⚠️ 历史 | v5 单卡视频超分 |
| `inference_realesrgan_video_v5.py` | ⚠️ 历史 | v5 多 GPU |
| `inference_realesrgan_video_v4.py` 等 | ⚠️ 历史 | 更旧版本 |
| `inference_realesrgan.py` | ⚠️ 图像版 | 单图像超分 |
| `inference_realesrgan_video_v5_single_bak*.py` 等 | 🗑️ 备份 | 开发备份 |
| `realesrgan/` | ✅ 库 | Real-ESRGAN Python 包 |
| `setup.py` | ✅ 安装 | Real-ESRGAN 包安装脚本 |
| `docs/` | ✅ 文档 | 原版文档（训练、模型等） |

---

## 模型文件

### `models_IFRNet/checkpoints/`

| 文件（示例） | 大小 | 说明 | 下载地址 |
|------|------|------|---------|
| `IFRNet_S_Vimeo90K.pth` | ~10 MB | **默认配置推荐**（`model_name`: `IFRNet_S_Vimeo90K`）；旧发行包中同名可为 `IFRNet_S.pth` | [GitHub Releases](https://github.com/ltkong218/IFRNet/releases/download/v1.0/IFRNet_S.pth) |
| `IFRNet_L_Vimeo90K.pth` / `IFRNet_L.pth` | ~25 MB | Large，质量更高 | [GitHub Releases](https://github.com/ltkong218/IFRNet/releases/download/v1.0/IFRNet_L.pth) |

> 实际文件名以 **`models.ifrnet.model_name`** 与 **`ifrnet_processor_v6_1_single.MODEL_NAME_MAP`** 为准。

### `models_RealESRGAN/`

| 文件 | 大小 | 适用场景 | 下载地址 |
|------|------|---------|---------|
| `realesr-general-x4v3.pth` | ~64 MB | **默认推荐**，通用 4x | [GitHub Releases](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth) |
| `RealESRGAN_x4plus.pth` | ~64 MB | 写实 4x | [GitHub Releases](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) |
| `RealESRGAN_x2plus.pth` | ~64 MB | 写实 2x | [GitHub Releases](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) |
| `RealESRGANv2-animevideo-xsx2.pth` | ~64 MB | 动漫视频 2x | [GitHub Releases](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.3.0/RealESRGANv2-animevideo-xsx2.pth) |

### `models_GFPGAN/`（可选，人脸增强）

| 说明 |
|------|
| 由 **`paths.models_gfpgan_dir`** 指定（留空时默认为 `base_dir/models_GFPGAN`）。存放 GFPGAN 主权重及 facexlib 相关缓存；启用 **`models.realesrgan.face_enhance`** 时需要。 |

---

## 运行时目录

| 目录 | 说明 |
|------|------|
| `output/` | 最终处理结果输出（如 mp4） |
| `temp/` | 处理临时文件（分段、音频），可随时清理 |
| `logs/` | 运行日志 |
| `.trt_cache/`（默认） | TensorRT Engine 缓存（`paths.trt_cache_dir` 留空时由 `config_manager` 派生） |

### `temp/` 子目录结构（运行时生成）

```
temp/
├── main_audio/                   # 从输入视频一次性提取的原始音频
├── {video_name}_ifrnet/
│   ├── checkpoint.json           # 断点记录（已完成分段列表）
│   ├── segment_000.mp4           # 插帧后的分段
│   ├── segment_001.mp4
│   └── ...
└── {video_name}_esrgan/
    ├── checkpoint.json           # 断点记录
    ├── processed_000.mp4         # 超分后的分段
    ├── processed_001.mp4
    └── ...
```

---

## 可清理的开发备份文件

以下文件为开发过程中留下的备份，不影响 **`main_video_optimized`** 正常使用，可酌情删除以节省空间：

```
external/IFRNet/process_video_*_bak*.py
external/Real-ESRGAN/inference_realesrgan_video_*_bak*.py
external/realesrgan_video/*_bak*.py
external/realesrgan_video/main - Copy.py
src/processors/realesrgan_processor_video_optimized_bak(多片段复用优化).py
src/utils/video_utils - Copy.py
```

---

## 文件状态图例

| 图标 | 含义 |
|------|------|
| ★ | 当前活跃使用的主要文件 |
| ✅ | 正常状态，当前版本 |
| ⚠️ | 历史或对照版本，保留供参考 |
| 🧪 | 实验/辅助，非主流程必需 |
| 🗑️ | 开发备份，可安全删除 |
| — | 辅助/训练用，推理时不需要 |
