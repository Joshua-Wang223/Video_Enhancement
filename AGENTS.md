# Repository Guidelines

Video Enhancement 是 GPU 加速视频处理管线：IFRNet 帧插帧（2x–16x）+ Real-ESRGAN 超分（2x/4x），支持 TensorRT、FP16、CUDA Graph 与 torch.compile。本文是面向贡献者的快速指南，完整项目文档见文末附录。

## Project Structure & Module Organization

- `src/main_video_optimized.py` — 唯一活跃入口；历史版本已于 2026-08-24 归档至 `archive/src_legacy/`（含旧 main*/processors v2~v6/backup 目录）。
- `src/processors/` — IFRNet 与 Real-ESRGAN 处理器（各仅存 `*_video_optimized.py`）；后端位于 `external/ifrnet_video/`（v6.4.5.1 深度模块化包）与 `external/realesrgan_video/`。
- `src/utils/` — `config_manager.py`（配置加载与 CLI 覆盖）、`video_utils.py`（切片/合并/音轨）、`logger.py`（统一日志，[P2.5]）。
- `external/nvenc_common/nal_utils.py` — 两子系统共享的 NAL 扫描参考实现（[P3.2]，等价性由回归测试锁定）。
- `config/default_config.json` — 全部默认配置，JSON 是单一事实来源。
- `tests/verify_plan_implementation.py` — 最终后验证脚本（v2 三合一：前置/静态核验/行为验证/运行时/冒烟；`python tests/verify_plan_implementation.py` 全量 72 项，CPU 可跑）。`tests/test_regression_min.py` 为其兼容别名（--behavior-only 转发）。其余 tests/ 为临时诊断脚本（verify_post_run.py 已归档至 archive/tests_legacy/）。

## Build, Test, and Development Commands

```bash
python src/main_video_optimized.py -c config/default_config.json -i input.mp4 -o output.mp4
```

常用旗标（经 `src/main_video_optimized.py` 验证）：`--skip-interpolate`、`--skip-upscale`、`--use-tensorrt-ifrnet`、`--use-tensorrt-esrgan`、`--face-enhance`、`--batch-mode`、`--dry-run`、`--report`、`--skip-seg-normalize`、`--quiet-ifrnet`、`--quiet-esrgan`、`--denoise`、`--denoise-model`、`--denoise-strength-pre`。依赖安装：先手动安装与 CUDA 匹配的 PyTorch，再 `pip install -r requirements.txt`；TensorRT 组件可选。本仓库无构建步骤，`tests/` 脚本直接运行即可。

## Coding Style & Naming Conventions

- Python 3.9+，4 空格缩进，UTF-8，snake_case 命名。
- 跨平台规范：路径用 `pathlib.Path` / `os.path.join`；子进程必须 `shell=False`；文件读写显式 `encoding='utf-8'`；可执行文件用 `shutil.which('ffmpeg')` 查找。
- 版本化脚本沿用 `_v6_4_5_1` 式命名；新增逻辑优先落在当前活跃版本文件。
- 代码质量工具体验性使用（无 ruff.toml 配置）；历史告警不强制修复。

## Testing Guidelines

- `tests/` 不是 pytest 套件，不要假设 pytest 可用。
- 以真实 GPU 运行验证：帧数完整性、bitstream 解析、空帧/码率统计。
- 命名沿用现有风格：`test_nvenc_*.py`、`verify_segment_bitstream*.py`。

## Commit & Pull Request Guidelines

- 仓库当前未初始化 git（无 .git），尚无既定提交历史；建议使用 Conventional Commits（`feat:` / `fix:` / `perf:` / `docs:`）。
- PR 需说明动机与影响面、关联 issue 或记忆文件；GPU/NVENC 改动附运行配置（RC 模式、pipe、LA、分辨率）与前后基准数据；视觉缺陷附截图或样张。

## Security & Configuration Tips

- `Video_Enhancement_github_token.txt` 是敏感凭据，除非明确要求不得读取或回显。
- 不要调用 `pycuda.autoinit`（与 PyTorch CUDA context 冲突）。
- 修改行为优先调整 `config/default_config.json`，并注意 OOM 自动降级与 checkpoint/resume 语义。

## Environment & Pitfalls

- `src/main_video_optimized.py` 入口自动设置 `PYTORCH_NVML_BASED_CUDA_CHECK=0`（见文件第 149 行），以避免 NVML/RM 版本不匹配导致的 INTERNAL ASSERT FAILED。外部运行脚本时如遇此类 CUDA 检测错误，可手动 export 该变量。
- FFmpeg 必须在 PATH 中可用（版本 ≥ 4.3）。Windows 上需确认 `ffmpeg` 可通过 `shutil.which('ffmpeg')` 找到。
- 模型权重需手动下载：`models_IFRNet/checkpoints/`, `models_RealESRGAN/`, `models_GFPGAN/` 目录下的 `.pth` 文件不会自动创建。详见 README.md "下载模型" 节。
- TRT Engine 首次构建较慢（尤其 IFRNet），缓存于 `base_dir/.trt_cache/`。更换 GPU SM 架构或模型配置后会自动重建。
- `config/default_config.json` 中 `paths.base_dir` 留空时，`config_manager` 会从配置文件位置向上两级推算项目根目录。非标准部署需手动填写。

---

## 附录：完整项目文档 / Appendix: Full Project Documentation

完整项目文档分散于以下位置 — AGENTS.md 不重复浩繁细节，按需读取：

- **README.md**（仓库根目录）— 功能特性、系统架构（Mermaid）、项目结构树、快速开始、模型下载、配置说明等全量文档。
- **`memory/` 目录**（仓库根）— 项目记忆文件，`MEMORY.md` 为索引入口。关键子系统文档：
  - NVENC 编码子系统：struct 布局、CE-Pipeline、空帧防御、RC 模式、生产配置推荐、bug 模式速查
  - Bug 修复与调试：各版本故障根因与修复记录
  - 性能测试与配置：v6.4.x 基准测试、生产最佳配置、RC 模式排名
  - 开发流程与编码技巧
  - ESRGAN NVENC 跨段修复（2026-07-28）
- **`.qoder/repowiki/`** — 已有知识库（`repowiki/knowledge/en/**/*.yaml`），可读取参考。

另有一份记忆文件镜像副本于 `C:\Users\Administrator\.claude\projects\D--Workspace-Python-Video-Enhancement-Video-Enhancement\memory`，两处互为镜像，编辑任何一侧后必须同步另一侧。

- `tests/` — 临时诊断脚本（test_nvenc_*.py、verify_segment_bitstream*.py 等），**非 pytest 套件**，不要假设 pytest 可用
- `Video_Enhancement_github_token.txt` — 敏感凭据文件，除非用户明确要求，不读取、不回显其内容
- 仓库当前**未初始化 git**（无 .git）
