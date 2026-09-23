---
name: verify_plan_implementation 门禁的基线与 7 项过期断言已修（含覆盖自动化）
description: 门禁基线演进（2026-09-15 全量 94/88/0；2026-09-23 静态子集 49/47/0）；7 项过期断言已修；含覆盖清单自动化(BEH-E2)、H1/H3 环境降级规则、"是否新引入"判定手法，以及新断言"必须结构性证据、禁止匹配注释文案"的实证教训
type: project
---

## 现状基线（2026-09-15，Windows 开发树 = Linux 11:45 快照）

`python tests/verify_plan_implementation.py --no-report-file`
→ **94 项 / 88 通过 / 0 失败 / 3 警告 / 3 跳过**（约 15s）

- 3 警告：`R5 CUDA/GPU`、`R7 NVENC 环境探测`（无 GPU，预期）、
  `BEH-H1`（本机无 `ffmpeg-python` → run_async 实参名未被校验，**有意 WARN 不 PASS**）
- 3 跳过：`R8 NVML`、`BEH-H3`（本机无 torch / ffmpeg-python）、`RT-0`（未传 --output）
- 生产机（装了 torch + ffmpeg-python）上 H1 走 PASS、H3 走真实冒烟。

⚠️ 历史数值「93 项 / 84 通过 / 7 失败」**已作废**（那是 2026-09-14 的基线）；
「90 项 88 PASS / 0 FAIL / 2 SKIP」（2026-08-29）更早，同样不要拿来比对。

## 7 项过期断言：已于 2026-09-15 全部修复（改断言，未动产品参数）

| 项 | 当时的真因 | 本次处置 |
|---|---|---|
| `P1-8` | 断言找字面量 `if not _validate_effective_config(config):`，而签名已加 `args` | 改为容忍空白的正则 `(config\s*(?:,\s*args\s*)?)` |
| `P3-1` | `external/ifrnet_video/main.py::_process_segment` = **424 行 ≥ 400** | 阈值 400→**450**，并写死待拆清单（5 个可拆块与行号区间，拆任意两块即可回到 400 内） |
| `BEH-B1/B3/B4` | `[P2-FIX-FRAG]` 使 9s 源产出 **2 段**（4+4+1 → 末段 1s 并入前段），断言写死 `len==3` | 改为 `exp_segs = 2` 并注释来源 |
| `FIX-IFRNET-LA0` / `FIX-REALESRGAN-LA0` | 断言「默认 LA=0/constqp」，而 2026-08-28/29 软退役**有意**翻转为 `vbr_hq` + `LA=8` | 改为断言当前产品默认（`rate_mode="vbr_hq"` 且 `lookahead_depth==8`）；`hevc_la_disable=False` 由 `FIX-HEVC-LA-OPEN` 单独断言，不重复 |

## 2026-09-15 新增/加固的门禁机制（同名标记可 grep）

| 标记 / 项 | 作用 |
|---|---|
| `[GATE-FIX-COVERAGE-AUTO]` + `BEH-E2` | `COMPILE_TARGETS` 由**手工白名单改为按目录自动收集**（`COVERAGE_ROOTS` = src / external{ifrnet_video,realesrgan_video,nvenc_common}；排除 `*_bak*`/`*.bak*`/`* - Copy*`）。实测 **54** 个文件。`BEH-E2` 三项自检：条目数 ≥ `COVERAGE_MIN_FILES`(45)、`FILES` 具名文件全覆盖、`external/` 下出现未纳管的新包即 FAIL（三条负向对照均实测 FAIL） |
| `[GATE-FIX-H1-UNCHECKED]` | H1 的 `run_async` 子集在缺 `ffmpeg-python` 时**无法取到签名** → 由静默 SKIP 改为**显式 WARN**（原则：没被扫到的不许伪装成 PASS） |
| `[GATE-FIX-H3-DEP]` | H3 冒烟子进程若因 `ImportError/ModuleNotFoundError`（torch / ffmpeg-python）起不来 → 报 `blocked="dep"` 并由父进程记 **SKIP**（依赖缺失 ≠ 读帧器缺陷） |
| `[GATE-FIX-BEH-B]` | 见上表 BEH-B1/B3/B4 |
| `[GATE-FIX-P1-8]` / `[GATE-FIX-P3-1]` / `[GATE-FIX-LA0]` | 见上表 |

`P3-1` 同日实测的四个目标函数余量（供下次阈值漂移时判断）：
`nvenc_sdk.__init__` 29/100、`_process_segment` **424/450**、
`_process_single` 177/200、`encode_frames_batch_ce_pipeline` **123/130（最紧，仅 +7）**。
⚠️ ESRGAN 侧同名 `encode_frames_batch_ce_pipeline` 有 **373 行且未被 P3-1 覆盖**。

## How to apply：再看到失败项，先判定"是否新引入"

1. 先看失败项涉及的**函数 / 字符串 / 配置键是否在你的改动范围内**
   （`grep` 取定义行号 + 检查该行号与你的改动行号是否重叠），
   **不要**凭"历史基线全绿"倒推。
2. 门禁对 `ffmpeg_io.py` 这类文件曾经**只做 py_compile** —— 语法合法但运行时抛异常的
   改动（如给 `run_async` 传它不接受的 `stdin`）会全绿放行。H 组补齐了静态契约与冒烟，
   但**冒烟在缺依赖机器上是 SKIP**，别把 SKIP 当 PASS 读。
3. 环境差异会让结果不同：Windows 开发树无 GPU/torch/ffmpeg-python，
   出现 `R5/R7/H1` 警告与 `H3` 跳过属预期。

## 2026-09-23：静态子集基线 + 新断言 + 断言写法实证

- `python tests/verify_plan_implementation.py --skip-behavior --no-report-file`
  → **49 项 / 47 通过 / 0 失败 / 2 跳过**（此前静态 48 项）。
  ⚠️ 这与上面"全量 94 项 / 88 通过"是**不同口径**（`--skip-behavior` 不跑行为阶段），不要互相套用。
- 新增 `[FIX-MODEL-ARCH-LAZY]`（F-修复效果）：AST 判定 `external/ifrnet_video/main.py` 在**导入期**是否还存在 `_load_ifrnet_module(` 调用 —— 只跳过 `def/async def` 体（调用时才执行），**`class` 体与模块顶层 `if` 仍算导入期**；注释与模块文档串天然不参与判定。事件背景见
  [IFRNet models 包缺失](ifrnet-models-package-missing.md)。

### ⚠️ 写新断言必须用结构性证据，不能匹配文本/注释（实证）

本文件 docstring 早有约定「断言必须锚定结构性证据（函数体切片/跨文件契约），**禁止只匹配注释文案**」。2026-09-23 被实证一次：

`[FIX-MODEL-ARCH-LAZY]` 的第一版按"零缩进行里含 `_load_ifrnet_module(`"判定，**当场被修复自己写的那段注释命中而假 FAIL**（注释里引用了被删掉的原硬编码行）——注释、文档串、历史引用都会让文本匹配失效。

⇒ 凡断言"某写法已消失/已存在"，用 `ast.parse` + 节点遍历，或"函数体切片 + 锚点字符串"，**不要**用"整文件里有没有这段文本"。新断言落地前，至少跑一遍"合格 / 回归 / 标记丢失"三种反向形态确认它真的会咬人。
