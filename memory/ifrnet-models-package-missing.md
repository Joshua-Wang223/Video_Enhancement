---
name: IFRNet 报 "No module named 'models'" ⇒ external/IFRNet/models/ 架构源码目录缺失
description: 工作区重建/恢复后 external/IFRNet/models/（IFRNet.py/IFRNet_S.py/IFRNet_L.py 上游架构源码）可能整体丢失，导致 import ifrnet_video.main 直接失败；含恢复来源、报错形态判据与排查口诀
type: project
---

**事实**：IFRNet 的网络结构**源码**位于 `<repo>/external/IFRNet/models/`（`IFRNet.py` / `IFRNet_S.py` / `IFRNet_L.py`，来自上游 `ltkong218/IFRNet` 仓库）。它属于工作区里的**普通源码目录**，不是 pip 包、不是权重；工作区重建/恢复时容易被整体漏掉。

**Why**：`external/ifrnet_video/config.py` 把 `external/IFRNet` 插进 `sys.path`，正是为了让 `MODEL_MODULE_MAP` 里的 `'models.IFRNet_S'` 等能解析；`ifrnet_utils._load_ifrnet_module()` 用 `importlib.import_module()` 加载它。目录一丢，整条导入链就断。

**How to apply（排查口诀）**：见 `❌ 无法导入 ifrnet_video.main: No module named 'models'`（或 `'models.IFRNet_S'`）**先 `ls external/IFRNet/models/`**，不要先去查 pip 依赖 —— 与第三方包无关。两种报错形态可反推根因：

- 报 `No module named 'models'`：**任何** sys.path 条目上都没有 `models` 目录。`python src/main_video_optimized.py` 启动时 `sys.path[0]` 是 `…/src`、**项目根不在 sys.path**，所以连顶层 `models` 都找不到（2026-09-23 生产实例）。
- 报 `No module named 'models.IFRNet_S'`：`models` 顶层包在（例如从项目根目录 cwd 运行时命中了 `<repo>/models/`），但里面没有 IFRNet 架构模块。

**恢复来源**：
- 本地快照 `/workspace/Video_Enhancement_retest_full_20260904.tar.gz` → `retest_package_20260904/code/current/external/IFRNet/models/`（含 3 个 .py，已实测可让导入恢复）。
- 或重新 `git clone https://github.com/ltkong218/IFRNet.git`，只取 `models/` 拷到 `external/IFRNet/models/`（`setup_project.py` 的仓库 URL 即此）。
- 依赖齐全性：`models/IFRNet_S.py` 还需 `external/IFRNet/utils.py`、`loss.py`（`from utils import warp, get_robust_weight` / `from loss import *`）。当前工作区这两个文件都在。

**已落地的代码侧加固（2026-09-23）**：
- `[FIX-MODEL-ARCH-LAZY]`（`external/ifrnet_video/main.py`）—— 原模块级硬编码 `Model, _ifrnet_s_mod = _load_ifrnet_module('IFRNet_S_Vimeo90K')` 是**导入期**硬依赖（且模块级无从得知运行期 `model_name`），已改为 PEP 562 模块 `__getattr__` 惰性解析；运行期架构解析仍由 `_load_model()` 按 `self.model_name` 完成（`[P0-FIX-MODEL-ARCH]`）。
- `[FIX-IMPORT-DIAG]`（两个 processor 的 `except ImportError`）—— 补 `traceback.print_exc()`，不再把"后端源码/依赖缺失"笼统成一句"无法导入 xxx.main"。
- 门禁新增静态断言 `[FIX-MODEL-ARCH-LAZY]`（`tests/verify_plan_implementation.py` 的 F-修复效果）：用 **AST** 判定"导入期是否存在 `_load_ifrnet_module(` 调用"——只跳过 `def/async def` 体，`class` 体与顶层 `if` 仍算导入期；注释/文档串不参与判定（符合本文件"禁止只匹配注释文案"的约定）。2026-09-23 实测 `--skip-behavior` **49 项 / 47 通过 / 0 失败 / 2 跳过**（此前静态 48 项）。

**易混淆的相邻物（勿混为一谈）**：
- `<repo>/models/`（2026-09-23 02:25 新出现的**权重**目录）：`models/IFRNet/IFRNet_{S,L}.pth` 是 **9 字节、内容为字面量 `Not Found`** 的下载失败残骸；`models/RealESRGAN/*.pth` 是真实权重。配置实际用的是 `models_IFRNet/checkpoints/`，与架构包无关。但它带 `IFRNet`/`RealESRGAN` 子目录，若项目根进入 `sys.path` 会变成 `models` 的 namespace portion，改变报错形态（见上）。
- 历史单文件 `external/IFRNet/process_video_v*.py`（25 个）含同名硬编码行，均**非生产代码**，未随本次修复改动。
