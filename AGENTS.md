# Repository Guidelines

Video Enhancement 是 GPU 加速视频处理管线：IFRNet 帧插帧（2x–16x）+ Real-ESRGAN 超分（2x/4x），支持 TensorRT、FP16、CUDA Graph 与 torch.compile。本文是面向贡献者的快速指南，完整项目文档见文末附录。

## Project Structure & Module Organization

- `src/main_video_optimized.py` — 唯一活跃入口；历史版本已于 2026-08-24 归档至 `archive/src_legacy/`（含旧 main*/processors v2~v6/backup 目录）。
- `src/processors/` — IFRNet 与 Real-ESRGAN 处理器（各仅存 `*_video_optimized.py`）；后端位于 `external/ifrnet_video/`（v6.4.5.1 深度模块化包）与 `external/realesrgan_video/`。
- `src/utils/` — `config_manager.py`（配置加载与 CLI 覆盖）、`video_utils.py`（切片/合并/音轨）、`logger.py`（统一日志，[P2.5]）。
- `external/nvenc_common/nal_utils.py` — 两子系统共享的 NAL 扫描参考实现（[P3.2]，等价性由回归测试锁定）。
- `config/default_config.json` — 全部默认配置，JSON 是单一事实来源。
- `tests/verify_plan_implementation.py` — 最终后验证脚本（三合一：前置/静态核验/行为验证/运行时/冒烟 + F-修复效果 phase（2026-08-27 水彩综合修复 + 2026-08-28 FIX-HEVC-LA-OPEN）；`python tests/verify_plan_implementation.py` 全量 90 项，CPU 可跑；2026-08-29 Linux 生产基准：88 PASS / 0 FAIL / 0 WARN / 2 SKIP；Windows 开发机无 NVIDIA GPU 时为 86 PASS / 0 FAIL / 2 WARN（R5 CUDA / R7 NVENC 环境探测）/ 2 SKIP（R8 NVML 提示、RT-0 输出视频存在），均属环境差异而非功能失败）。`tests/test_regression_min.py` 为其兼容别名（--behavior-only 转发）。其余 tests/ 为临时诊断脚本（verify_post_run.py 已归档至 archive/tests_legacy/）。

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
- 段级验收用 `tests/verify_segment_bitstream_v5.py`（帧守恒/单 IDR/frame_num 单调/pts/色度簇，含解码级检查与检查间并行）；NVENC 编码器层级行为回归用 `tests/diagnose_hevc_la.py`（11 变体矩阵，`--skip-reproducers` 跑 6 变体回归集）。
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
- HEVC LA 安全性由 `FIX-HEVC-COUNTED/EOS/EOS-FLUSH`（nvenc_sdk 层）+ `P1-FIX-H2D-EVENT-SYNC`（水彩根治）+ EOS 排空硬化保障；`hevc_la_disable` 已软退役为应急开关（默认 `false`，命中仅 WARN 不降级，两侧仓库一致）。异常回滚：显式置 true 或 `NVENC_HEVC_ALLOW_LA=0`（详见 `memory/hevc-la-open-production.md` / `memory/hevc-la-soft-retired.md`）。
- 段级验收含解码级门禁（`video_utils.validate_decodable_video` / `count_decoded_video_frames`，verify_plan RT-4/RT-5），包级 frames==packets 对"包在但解不出"是盲区。
- `verify_segment_bitstream_v5.py` 的色度检查（检查 4）可靠性有限，验证硬指标（帧守恒/IDR/frame_num/pts）时建议加 `--skip-chroma`。
- `--mode upscale_then_interpolate` 会被引擎自动保护改写：超分后像素数超过 `processing.max_upscale_then_interpolate_pixels`（默认 3670016 ≈ 2560×1440，`0` 禁用自动切换）时自动切 `interpolate_then_upscale` 并打印警告（`main_video_optimized.py::_select_optimal_mode`，配置摘要 + `_process_single` 双重调用）。原因：高分辨率插帧在 T4 级 GPU 上会早期 EOF/丢帧（详见 `memory/mode-auto-protect-upscale-then-interpolate.md`）。
- `--ifrnet-model` 生效依赖 `config_manager` 按 `model_name` 派生 `model_path`（CLI 覆盖后由 `config._derive_model_paths()` 重算）；`--ifrnet-model-path` 优先级更高。历史 bug 曾硬编码 S 模型导致所有模型输出相同（详见 `memory/ifrnet-model-selection-bug-fix.md`）。

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
  - HEVC LA 修复栈（2026-08）：排空诊断（hevc-la-drain-diagnosis）、水彩花屏综合修复（ifrnet-watercolor-tail-defect-investigation）、LA>0 生产就绪/软退役（hevc-la-open-production、hevc-la-soft-retired）
  - 入口行为与配置派生（2026-08-29）：upscale 模式自动保护（mode-auto-protect-upscale-then-interpolate）、IFRNet 模型选择修复（ifrnet-model-selection-bug-fix）
- **`.qoder/repowiki/`** — 已有知识库（`repowiki/knowledge/en/**/*.yaml`），可读取参考。

#### ⚠️ 记忆文件镜像同步（强制规则，2026-09-08 用户确认）

`memory/` 有两处，互为镜像，**编辑任何一侧后必须立即同步另一侧，不得只改一侧**：

| | 路径 | 说明 |
|---|---|---|
| A（canonical） | `/workspace/Video_Enhancement/memory` | 项目内，仓库根 `memory/` |
| B（会话侧） | `/root/.codebuddy/projects/workspace-Video_Enhancement/memory` | CodeBuddy 会话自动加载侧 |

```bash
A=/workspace/Video_Enhancement/memory
B=/root/.codebuddy/projects/workspace-Video_Enhancement/memory
cp -a "$A/." "$B/"     # A → B（同名覆盖；反向同理，按"最后改动的一侧"为准）
diff -r "$A" "$B" && echo "✅ 两处一致"   # 同步后必须复核
```

要点：
- **每次**新建/修改 `memory/` 下任何文件（含 `MEMORY.md` 索引）后都要同步，索引条目也要一并加上。
- 同步用 `cp -a` 同名覆盖；同步后必须用 `diff -r` 复核两侧文件集合与内容一致。
- ⚠️ **`cp -a` 不传导删除**：在一侧删除/改名文件后，对侧的同名残留不会被清掉。
  凡有删除或改名，必须用 `rsync -a --delete "$A/" "$B/"`（或两侧分别删除）后再次复核。
  - `--delete` 是破坏性操作：**先用 `rsync -a --delete --dry-run "$A/" "$B/"` 预览**，
    确认待删清单无误后再去掉 `--dry-run` 执行。
  - ⚠️ **本容器实测未安装 `rsync`**（2026-09-08 核实）。改用无依赖方式列出差异文件，
    人工确认后再删（只列不删，安全）：
    `comm -13 <(cd "$A" && ls -1|sort) <(cd "$B" && ls -1|sort)`  # 仅 B 有
    `comm -23 <(cd "$A" && ls -1|sort) <(cd "$B" && ls -1|sort)`  # 仅 A 有
- 写中文必须用支持 UTF-8 的直写工具（Write/Edit），**禁止**经 shell 管道/heredoc 写中文
  （历史事故：控制台代码页把中文转码成字面 `?`，不可逆）。写后抽检：中文字符相邻的 `0x3F` 应为 0。
- 历史镜像（Windows 开发机，路径已失效，仅存档）：
  `D:\Workspace_Python\Video_Enhancement\Video_Enhancement\memory`、
  `C:\Users\Administrator\.claude\projects\D--Workspace-Python-Video-Enhancement-Video-Enhancement\memory`

- `tests/` — 临时诊断脚本（test_nvenc_*.py、verify_segment_bitstream*.py 等），**非 pytest 套件**，不要假设 pytest 可用
- `Video_Enhancement_github_token.txt` — 敏感凭据文件，除非用户明确要求，不读取、不回显其内容
- 仓库当前**未初始化 git**（无 .git）
