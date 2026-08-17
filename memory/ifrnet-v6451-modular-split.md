# IFRNet v6.4.5.1 深度模块化拆分（ifrnet_video）

日期：2026-08-17

## 目标

把单文件后端 `external/IFRNet/process_video_v6_4_5_1_single.py`（8756 行）重构为
独立子项目 `external/ifrnet_video/`，目录与模块划分**镜像 `external/realesrgan_video/`**；
同时新增中间适配层 `src/processors/ifrnet_processor_video_optimized.py` 对接新包。

三条硬约束：
1. 结构对齐：模块划分与 realesrgan_video 一致（pipeline / nvenc_sdk /
   tensorrt_accel / ffmpeg_io / config / utils / main）。
2. 特性全保留：v6.4.5.1 四级回退编码、[FIX-AUX-NO-CLEAR] / [FIX-DRAIN-ORDER-DEFENSE]
   LA 辅助块修复、[FIX-ASYNC-COPY] 异步拷贝等全部逻辑逐字迁移，不丢行为。
3. 模块化解耦：按职责拆分，保留高性能执行流。

## 模块映射（与 realesrgan_video 对齐）

| ifrnet_video 模块 | 内容（自原文件行区间逐字迁移） | realesrgan_video 对应 |
|-------------------|-------------------------------|-----------------------|
| `config.py` | 路径常量（base_dir/models_ifrnet）、MODEL_STRIDE、MODEL_MODULE_MAP、MODEL_NAME_MAP | `config.py` |
| `ifrnet_utils.py` | `_cached_warp` / `_load_ifrnet_module`、ThroughputMeter、PinnedResultPool/PinnedRingBuffer/CudaEventPool、PinnedBufferPool、pad/frames_to_tensor/tensor_to_np/TensorPool、_clamp_decode_threads | `realesrgan_utils.py` |
| `ffmpeg_io.py` | HardwareCapability、_X264_PRESET_FACTOR/_X264_TO_NVENC_PRESET、_software_encode_fps、_detect_encode_parallelism、_NVENC_SURFACES_PIPE/_NVENC_LOOKAHEAD_VBR、FFmpegFrameReader/_probe_video/FFmpegWriter | `ffmpeg_io.py` |
| `nvenc_sdk.py` | NVENC ctypes SDK（GUID/struct/常量/原型）、_clamp_bitrate、_RotationBitReader、NVENCEncoder、_NVENCEncodeThread、_rgb_to_nv12_gpu(_batch)、FFmpegMuxer、_NVENC_LEVEL1_* / _NVENC_CRF0_* / _NVENC_QVBR_ENABLE_VBV 常量 | `nvenc_sdk.py` |
| `pipeline.py` | GPUStats/GPUMonitor、_HWProfile/_GPU_PROFILES_TABLE/_detect_hw_profile、t2 cache、_pool_limit_mb_for_profile/_PINNED_POOL_MAX_MB、_compute_max_result_queue/_compute_max_pair_queue、_MODEL_T2_FACTOR/_INFER_BACKEND_FACTORS、_auto_queue_depths、IFRNetPipelineRunner | `pipeline.py` |
| `tensorrt_accel.py` | `TensorRTAccelMixin`：`_build_trt_engine`（原 7028-7208）+ `_infer_batch_trt`（原 _infer_batch 的 TRT 分支 7318-7366，末尾补 `return pred_big`） | `tensorrt_accel.py` |
| `main.py` | 环境/NVML 过滤设置、`Model, _ifrnet_s_mod = _load_ifrnet_module(...)`、IFRNetVideoProcessor（继承 TensorRTAccelMixin）、main() | `main.py` |
| `__init__.py` | `__version__ = "6.4.5.1"` | `__init__.py` |

## 保真性保证

- 拆分采用**逐字机械迁移**：原文件每一行都精确归属唯一模块（行覆盖 100%、无重叠），
  拆后各段与原文件 byte-for-byte 一致（临时工具 `tests/_ifrnet_split.py --verify` 通过）。
- 全部 143 个顶层定义 + IFRNetVideoProcessor 全部 16 个方法无丢失（AST 完整性校验）。
- 仅有 3 处刻意改写：
  1. `class IFRNetVideoProcessor(TensorRTAccelMixin)`（继承 mixin）；
  2. `_infer_batch` 的 TRT elif 分支 → `pred_big = self._infer_batch_trt(img0_exp, img1_exp, timesteps, B)`；
  3. mixin 的 `_infer_batch_trt` 末尾补 `return pred_big`（原分支为同作用域赋值）。

## 跨模块依赖（无环 DAG）

```
config ← ifrnet_utils ← ffmpeg_io ← pipeline ← main
nvenc_sdk（独立）← pipeline / main
tensorrt_accel（独立）← main
```

关键取舍：
- `_detect_encode_parallelism` 放 ffmpeg_io（镜像 realesrgan），GPUMonitor 放 pipeline，
  避免 ifrnet_utils ↔ ffmpeg_io 循环。
- `_NVENC_LEVEL1_*` 常量放 nvenc_sdk（NVENCEncoder 引用 `_NVENC_QVBR_ENABLE_VBV`，
  放 pipeline 会造成 pipeline ↔ nvenc_sdk 循环）。
- 执行顺序保真：main.py 先装 _NVMLFilter + PYTORCH_ALLOC_CONF 再导入 pipeline
  （pipeline 模块级 `_PINNED_POOL_MAX_MB` 会触发 CUDA 初始化），与原文件一致。

## 处理器层

- `src/processors/ifrnet_processor_video_optimized.py` 由
  `ifrnet_processor_v6_4_single.py` 复制而来，仅改对接点：
  `self.ifrnet_dir = base_dir/external/ifrnet_video`；sys.path 同时加入
  `external/`（包导入）与 `external/ifrnet_video/`（兄弟模块平级导入）；
  `from ifrnet_video.main import IFRNetVideoProcessor`。
- `src/main_video_optimized.py` 的 IFRNet 导入已切至新适配层；
  `ifrnet_processor_v6_4_single.py` 保留为历史适配层。

## 验证状态

- ✅ 全部 7 个模块 py_compile 通过；AST 未定义名审计 0 告警。
- ✅ 跨模块 `from X import Y` 名称逐一校验全部解析成功。
- ⚠️ 真实 GPU 运行验证未执行（开发机无 torch/CUDA）：需在 Linux 生产环境以
  真实视频复跑插帧，比对帧数完整性、bitstream、空帧/码率与旧单文件基线。
- 旧单文件 `external/IFRNet/process_video_v6_4_5_1_single.py` 原样保留，
  作为拆分源与回滚参照。

## 文档同步

- README.md：Mermaid 架构图、后端内部架构、目录树、独立调用示例已更新。
- AGENTS.md：活跃文件表、架构树、IFRNet backend 小节、Level 选择逻辑/NVENCEncoder
  行号引用已更新至 ifrnet_video。
