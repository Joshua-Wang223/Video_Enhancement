---
name: 失败路径必须暴露真实根因（勿笼统化异常 / 勿导入期硬编码运行期配置）
description: 用户明确判定为缺陷的两类写法：except 只 print(str(e)) 丢掉 traceback；模块级（导入期）硬编码运行期才确定的配置。含 Why（2026-09-23 IFRNet models 缺失误诊）与套用场景
type: feedback
---

两条由用户直接判定为 **bug** 的写法，都属"把真实根因藏起来"，本仓库内一律按缺陷处理、要求修掉。

## 规则 1：错误处理路径不得用一句笼统消息吞掉真实失败点

**规则**：`except ImportError` / `except Exception as e` 内必须保留完整现场 —— 补 `traceback.print_exc()`（或等价记录异常链），且消息里要带上能区分根因的上下文（缺哪个**目录**/模块/文件），不要只 `print(f"... {e}")`。

**Why**：2026-09-23 生产实例 —— IFRNet 报 `❌ 无法导入 ifrnet_video.main: No module named 'models'`，而 processor 的 `except ImportError` 只打印 `e`、丢掉 traceback，把"上游网络结构源码目录 `external/IFRNet/models/` 整体缺失"误诊成"缺第三方包"。用户原话是"补 traceback.print_exc()，**避免以后再被这条笼统消息误导**"，即他要求的是消除"笼统化"本身，而不只是这一个报错。

**How to apply**：改/审 `src/processors/*.py` 中包裹后端导入的 try/except，以及任何 `except ... as e: print(e)` 的位置，一律补 traceback + 根因上下文。已落地标记 `[FIX-IMPORT-DIAG]`（ifrnet / realesrgan 两个 processor 的 `except ImportError` 均补 `traceback.print_exc()`）。

## 规则 2：运行期才确定的配置，不得在模块导入期硬编码

**规则**：模块级代码处在"导入时刻"而非"运行时刻"；凡依赖 CLI/配置才能确定的东西（模型名、路径、编解码器、码率…），一律在实例化/调用时按参数解析，不在模块级写死。需要保留的向后兼容名改用 PEP 562 模块 `__getattr__` 惰性解析。

**Why**：同一事故中，`external/ifrnet_video/main.py` 模块级硬编码 `Model, _ifrnet_s_mod = _load_ifrnet_module('IFRNet_S_Vimeo90K')` 造成双重伤害 —— ① 模块级无从得知运行期 `model_name`，只能钉死 S 架构；② 使 `models.IFRNet_S` 成为**整包导入**的硬依赖（即使只跑 IFRNet_L，S 源码缺失也整体导入失败），正是它把一次"目录缺失"放大成"包导不进来"。用户直接称其"**是硬编码的bug**"。

**How to apply**：见到模块级（类/函数定义之外的顶层语句）读配置、拼路径、加载模型，先问"这里知道运行期参数吗"；不确定就挪到实例化路径。已落地标记 `[FIX-MODEL-ARCH-LAZY]`（改为模块 `__getattr__` 惰性解析；运行期架构解析仍由 `_load_model()` 按 `self.model_name` 完成）。同类写法在历史单文件 `external/IFRNet/process_video_v*.py`（25 个）中仍大量存在，但**非生产代码**，勿据此推翻修复状态。
