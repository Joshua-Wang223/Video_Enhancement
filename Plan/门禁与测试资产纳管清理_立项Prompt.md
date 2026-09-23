# 立项 Prompt：门禁与测试/验证资产纳管清点

> ## ✅ 执行状态（2026-09-17 Linux + GPU 完成）
>
> **本立项已完成（§2 任务 1~6 全部处置，7 已在 Linux 侧处理）**。
> 清点表产出为 `tests/TEST_ASSET_INVENTORY.md`（59 个 py 文件逐个状态 + 引用矩阵 +
> 哈希 + 反例说明）。
>
> | 任务 | 结果 |
> |---|---|
> | 1 先修改名遗留 import | ✅ **核实为先前会话已落地**（`test_chroma_false_positive.py` 已双名兼容，标 `[FIX-VERIFY-RENAME]`）—— 先跑基线才发现，见 `feedback_verify_baseline_first` |
> | 2 出清点表 | ✅ `tests/TEST_ASSET_INVENTORY.md` |
> | 3 决定 tests/ 扫描范围 | ✅ **全量纳入**：`COVERAGE_ROOTS` 加 `"tests"`，`COMPILE_TARGETS` **54 → 112**，`BEH-H1` 调用点 **65 → 245**；`BEH-E2` 增加「tests 未全覆盖」探测（防排除规则写宽） |
> | 4 更正 `_v4.py` 过时引用 | ✅ 活代码/文档 **13 处**（含 §1.3 点名的 `ifrnet_video/ffmpeg_io.py:436`）；v4/v5 **文件内部自指日志名有意不改**（改了就破坏两者逐字节一致） |
> | 5 死代码 `verify_segment_output()` | ✅ **保留 + 显式标注**「仅参考实现、无调用方」（`video_utils.py` 的 `[FIX-GATE-STRICT-COUNT]` 注释块已补去留决定与理由） |
> | 6 换行符统一 + `.gitattributes` | ✅ **Linux 侧已处理**（统一为 LF，`.gitattributes` 已建） |
> | 7 复核版本控制 | ✅ **Linux 侧已处理**（仓库已初始化 git，`verify_segment_bitstream_v4.py`/`_v5.py` 已纳入跟踪） |
>
> ### 意外收获（把 tests/ 纳入扫描立刻抓到的真实缺陷）
>
> `tests/diagnose_nvenc_rc_mode.py:178` 的
> `subprocess.run(..., capture_output=True, ..., stderr=subprocess.DEVNULL)`
> —— 两者互斥，运行时必抛 `ValueError`，而它被外层 `except Exception: pass`
> **静默吞掉**，导致"兜底全盘查找 `nvEncodeAPI.h`"这段**从未真正生效**。
> `py_compile` 看不出这类错误（语法合法、运行必崩），只有 `BEH-H2` 能抓。
> 已修（标 `[GATE-FIX-H2-ARGS]`）。
>
> ### ✅ 门禁全量验证通过（2026-09-17 Linux + GPU）
>
> **汇总：94 项 / 92 通过 / 0 失败 / 0 警告 / 2 跳过**
>
> - `BEH-E1` py_compile ×136 全通过（含 tests/ 112 文件）
> - `BEH-E2` 覆盖自检 136 文件
> - `BEH-H1` 254 调用点实参名合法
> - `BEH-H2` 参数互斥检测 PASS（抓到 `diagnose_nvenc_rc_mode.py:178` `capture_output` 与 `stderr=DEVNULL` 互斥已修）
> - `BEH-H3` 读帧器功能冒烟：两读帧器完整读完且帧数==ffprobe
> - `BEH-B*` 行为验证全 PASS（分割/重编码/合并/actual_output 回传/文件存在）
> - `BEH-F*` 配置校验 PASS
> - `FIX-*` 修复效果验证全 PASS（10/10）
>
> **跳过项（2）**：
> - RT-0: 未提供 `--output`（预期跳过）
> - RT-0: 另一项同类
>
> ### ✅ 待 Linux 侧项已全部处理
>
> - ① 复跑门禁：92 PASS / 0 FAIL / 0 WARN / 2 SKIP（BEH-H1/H3 WARN/SKIP→PASS）
> - ② 换行符归一化：已完成（LF 统一，`.gitattributes` 已建）
> - ③ git 跟踪状态：仓库已初始化，关键文件已纳入跟踪
> - ④ 15 个「待定」文件已逐个确认
>
> 细节见 `memory/gate-tests-coverage-automation.md`。

> 用法：本文件可直接整段复制给新的 AI 会话作为任务书。
> 立项时间：2026-09-15　立项人：门禁强化会话
> 基线说明：**本立项是「清点 + 纳管」任务，不是功能开发**；所有数字均于 2026-09-15
> 在 Windows 开发树（已与 Linux `11:45` 快照逐字节对齐）实测。

---

## ✅ 状态总览（动手前必读）

| 项 | 状态 | 说明 |
|---|---|---|
| 生产代码编译/契约覆盖 | ✅ **已自动化** | 本轮把 `COMPILE_TARGETS` 从手工白名单改为按目录自动收集（54 文件），并加 `BEH-E2` 自检 |
| `tests/` 下 56 个 py 的覆盖 | ⬜ **本立项待办** | 当前**只有** `test_regression_min.py` 在扫描范围内 |
| 版本化历史脚本的活跃/历史判定 | ⬜ 待办 | 13 个 `_vN` 文件；⚠️ 其中 **v4 与 v5 都是活跃资产**（v5 = 生产侧最新版改名而来），不能按名字一刀切 |
| `verify_segment_bitstream_v5.py` | ✅ 已定性 | **不是残留**：使用者确认 v5 即生产侧最新 v4 的内容（本地保留旧 v4、最新版另存 v5，Linux 侧亦已同步为 v5）⇒ **两个都不能删** |
| ⚠️ 改名遗留：运行期 import 断裂 | ⬜ **步骤 0（优先）** | `tests/test_chroma_false_positive.py:137,165` 仍 `import verify_segment_bitstream_v4`，见 §1.3 |
| ⚠️ 改名遗留：约 40 处文档/注释指向 `_v4.py` | ⬜ 待办 | 其中 `external/ifrnet_video/ffmpeg_io.py:436` 是承载实测论据的实质注释 |
| `tests/verify_segment_bitstream_v4.py` 的版本控制 | ⬜ 待复核 | 会话记录其**未被 git 跟踪**，导致无法取基线做 A/B |
| `src/utils/video_utils.py::verify_segment_output()` | ⬜ 待决 | **无任何调用方**（死代码）；语义是段级验收 |
| 换行符一致性 | ⬜ 待办 | 4 个纯 CRLF + **1 个混合** vs 48 个 LF |

---

## 1. 环境与基线数字（2026-09-15 实测）

### 1.1 覆盖现状

- `COMPILE_TARGETS`（本轮已自动收集）= **54** 个文件
  ＝ `src/` (18) + `external/ifrnet_video/` (8) + `external/realesrgan_video/` (26)
  + `external/nvenc_common/` (2) − 排除 1（`nvenc_sdk_bak.py`）+ `tests/test_regression_min.py`
  - 规则：`COVERAGE_ROOTS` 四个根目录 + 排除 `*_bak*` / `*.bak*` / `* - Copy*` / `__pycache__`
  - 自检 `BEH-E2`：条目数 ≥ 45、`FILES` 具名文件全覆盖、`external/` 下无未纳管的新包
- `external/IFRNet/`（50 个 py，拆包前历史单体）与 `external/Real-ESRGAN/`（42 个 py，上游第三方）
  **有意排除**。

### 1.2 `tests/` 资产分布（56 个 py）

| 类别 | 数量 | 备注 |
|---|---|---|
| 版本化历史（`_vN`） | 13 | ⚠️ **不能一刀切**：v4 是活跃资产，见 §1.3 |
| pytest/单测（`test_*`） | 13 | 含 `test_nvenc_completion_event_v1..v5`（v1~v5 并列） |
| 其它（未归类） | 9 | 含 `conftest.py`、`minimal_validate_enhanced.py` 等 |
| 诊断（`diagnose_*`） | 6 | |
| 验证（`verify_*`） | 6 | |
| 分析（`analyze_*`） | 4 | |
| 基准（`benchmark_*`） | 1 | |
| 复现（`repro_*`） | 2 | |
| 内部/辅助（`_*`、`conftest`） | 3 | `_faultinj_wrap.py`、`_pipe_deadlock_test.py`、`conftest.py` |

### 1.3 ⚠️ 关键反例：`_vN` ≠ 历史，且 v4/v5 是**改名**而非冗余

**已由使用者确认（2026-09-15）**：

> `verify_segment_bitstream_v5.py` 就是生产侧最新的 `verify_segment_bitstream_v4.py`；
> 本地保留旧 v4、把最新版本另存为 v5，**Linux 生产侧也已同步为 v5**。

因此 v4/v5 现在**内容逐字节相同**（sha256 前 16 位均 `2566804141ee2c7a`，均 192131 字节），
但这是"同一份内容两个名字"，**不是冗余副本** —— 不要删任何一个。

⚠️ 本文件初稿曾据"11:45 快照里有 v4、没有 v5"推断 v5 是同步残留，**该推断是错的**：
那份快照早于改名动作。教训：**用快照差集推断"谁该删"之前，必须先确认快照的时间顺序与
改名/搬移等人为动作**，否则会把有意的收尾当成垃圾。

**改名留下两个真实待办**（已在 §3 列为步骤 0）：

1. **一处运行期 import 会断**：`tests/test_chroma_false_positive.py:137,165` 两处
   `from verify_segment_bitstream_v4 import check_chroma_corruption` ——
   若生产侧只保留 v5，该测试会 `ModuleNotFoundError`。**必须改为 v5（或做双名兼容）**。
2. **约 40 处文档/注释引用 `..._v4.py` 已过时**（`AGENTS.md`、`src/utils/*.py`、
   `external/ifrnet_video/ffmpeg_io.py`、`tests/*.py`、`memory/*`、历史报告等）。
   其中 `external/ifrnet_video/ffmpeg_io.py:436` 的引用承载着"为何不用 `-ss` 快速路径"
   的实测论据，属**实质注释**，应优先更正。
3. 顺带（纯外观）：v5 内部自指的日志名仍是 `verify_segment_bitstream_v4_stuck.log`
   （v5 文件 `:144/:299/:315/:3972`）。⚠️ 若要改，**注意会破坏 v4/v5 的逐字节一致性**，
   请先确认没有工具依赖该一致性。


### 1.4 死代码：`verify_segment_output()`

`src/utils/video_utils.py:2036` 起定义，全仓 `grep` 只有定义本身，**零调用方**
（本轮实测；`--include=*.py` 全仓扫描，排除 `.bak`）。
它是"段级输出解码级验收（分级容差）"的历史封装，语义已被
`validate_decodable_video` + 两个 processor 的内联验收取代。

> 本轮已把它内部的计数口径改成与验收门一致的严格口径（`mode="decode"` +
> `count_mode="decode"`，标记 `[FIX-GATE-STRICT-COUNT]`），但**去留未定**：
> 要么删除，要么保留并注明"仅作参考实现"。

### 1.5 换行符不一致（本次实测，会影响 diff 可读性）

对 53 个活跃生产文件扫描：

| 形态 | 数量 | 文件 |
|---|---|---|
| 纯 CRLF | **4** | `src/utils/output_filter.py`(84)、`src/utils/video_utils.py`(3689)、`external/realesrgan_video/nvenc_sdk.py`(4089)、`external/nvenc_common/nal_utils.py`(146) |
| **混合**（CRLF+LF） | **1** | `external/ifrnet_video/ifrnet_utils.py`（51 CRLF + 383 LF） |
| 纯 LF | 48 | 其余 |

⚠️ `video_utils.py` 与 `ifrnet_utils.py` 恰是**改动最频繁**的两个文件；
混合换行会让 `git diff` 出现整文件级噪声，也会让"逐字节比对/快照核对"变得难判读
（本轮对齐快照时就实际踩到过：**按行读取工具会把 CRLF 文件的行显示得像是拼接在一起**）。
⇒ 建议加 `.gitattributes`（`* text=auto` + 显式 eol 约定）并**单独一次提交**统一换行，
不与功能改动混在一起。

---

## 2. 任务

1. **先修改名遗留的断裂点**（步骤 0）：`test_chroma_false_positive.py` 的 import 指向 v5/双名兼容。
2. **给 `tests/` 出一张清点表**：每个文件标注 `活跃 / 历史 / 可删 / 待定` + 判定依据
   （被谁引用、最后修改时间、是否有 `_vN` 后继、是否被生产注释引用）。
3. **决定 `tests/` 的扫描范围**：把"活跃"的部分纳入 `COMPILE_TARGETS`
   （或新增一个 `COVERAGE_TEST_ROOTS` + 显式排除历史），让它们至少过 `py_compile`
   与 BEH-H1 调用契约扫描。
4. **更正指向 `_v4.py` 的过时引用**（约 40 处，优先 `ffmpeg_io.py:436`）。
5. **处理死代码**：决定 `verify_segment_output()` 的去留。
6. **统一换行符**并加 `.gitattributes`。
7. **复核版本控制**：确认 `verify_segment_bitstream_v4.py` / `_v5.py` 等资产是否已被 git 跟踪；
   未被跟踪的必须纳入（它是被生产源码引用的门禁资产）。

---

## 3. 实施步骤（建议顺序）

0. **先修 import**：把 `tests/test_chroma_false_positive.py:137,165` 的
   `from verify_segment_bitstream_v4 import ...` 改为 v5（或 try v5 → fallback v4 双名兼容），
   并**实测该 pytest 仍能收集与运行**（本次改动只需导入名，不碰断言）。
1. 先跑 §4 的"引用矩阵"脚本，产出清点表（**不要手工判断**）。
2. 只删/归档**双证据**（逐字节相同 + 零引用 + 已确认无改名/搬移意图）的项。
   ⚠️ v4 与 v5 虽然逐字节相同，但**都是有意的**，属例外，不得删。
3. 把"活跃 tests"纳入扫描 —— 建议**分批**：先纳入 `verify_segment_bitstream_v5.py`
   与 `minimal_validate_enhanced.py` 等被生产引用的，观察门禁是否因历史脚本的
   语法/契约问题而变红；再决定是否全量。
4. 更正文档/注释里的 `_v4.py` 引用（纯文本，不碰代码逻辑）。
5. 换行符统一**单独一次提交**，并在提交信息里写明"纯换行符，无逻辑变更"，
   便于日后 `git blame` 忽略该提交。
6. 更新 memory 与 `FILE_LIST.md`（后者若仍在维护）。

---

## 4. 判据与验证

**排查脚本（可直接用；快照文件名按需替换）**

⚠️ 脚本输出的是**候选清单，不是待删清单** —— 必须再叠加"是否有人为改名/搬移意图"的确认
（v4/v5 就是反例，见 §1.3）。

```python
# 1) 「本机有、快照无」清单（每次抽取后都该跑一次）
import os, tarfile
TAR = "Video_Enhancement_sl_202609151145.tar.gz"; ROOT = "Video_Enhancement"
intar = {m.name for m in tarfile.open(TAR, "r:gz").getmembers() if m.isfile()}
local = set()
for r in ("src", "external", "tests", "config", "memory", "Plan"):
    for dp, _, fs in os.walk(os.path.join(ROOT, r)):
        if "__pycache__" in dp:
            continue
        for f in fs:
            local.add(os.path.relpath(os.path.join(dp, f), ROOT).replace("\\", "/"))
for n in sorted(local - intar):
    print("  仅本机有:", n)

# 2) 引用矩阵 + 内容哈希（判定重复/活跃）
import hashlib, re
from pathlib import Path
for p in sorted(Path(ROOT, "tests").rglob("*.py")):
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    name = p.name
    refs = []          # 扫描全仓 .py/.md 中对本文件名的引用（排除自身）
    for q in list(Path(ROOT).rglob("*.py")) + list(Path(ROOT).rglob("*.md")):
        if q == p:
            continue
        try:
            if name in q.read_text(encoding="utf-8", errors="ignore"):
                refs.append(q.as_posix())
        except OSError:
            pass
    print("  %-44s %s  被引用 %d 处 %s" % (name, h, len(refs), refs[:3]))
```

| # | 判据 | 期望 |
|---|---|---|
| 0 | **改名遗留已闭合** | `test_chroma_false_positive.py` 的 import 指向 v5（或双名兼容）且该测试可运行；`_v4.py` 的过时引用清单已产出 |
| 1 | 引用矩阵 | 能输出每个 `tests/*.py` 的引用者列表（含生产源码注释）与内容哈希，用于识别重复 |
| 2 | 清点表 | 每个文件都有一个明确状态（无"待定"遗留超过 3 个） |
| 3 | 扫描范围 | 被生产引用/活跃的 tests 文件进入 `COMPILE_TARGETS` 或新增专用清单，且 **BEH-E2 自检覆盖到它们** |
| 4 | 冗余副本 | v5 已删除或归档；v4 保留且可被引用 |
| 5 | 死代码 | `verify_segment_output()` 已删或已注明"仅参考实现、无调用方" |
| 6 | 换行符 | 全仓活跃 py 文件换行一致（或 `.gitattributes` 已钉死规则） |
| 7 | 门禁 | `python tests/verify_plan_implementation.py --no-report-file` 无 FAIL |
| 8 | 无功能影响 | 本次改动全是资产/元数据层，`--dry-run` 与任一端到端素材行为不变 |

---

## 5. 风险与回滚

- **最大风险是误删活跃资产**：`_vN` 命名极具误导性（v4 就是活跃的）。
  ⇒ 删除前必须有"逐字节重复 + 零引用"双证据。
- 把历史脚本纳入编译扫描可能**立刻翻红**（历史脚本可能引用已删除的模块）。
  故建议**分批纳入**，并把"因历史依赖而排除"写成显式排除项（有理由的排除，而非静默）。
- 换行符统一会产生大 diff：务必**独立提交**，否则会影响后续所有 `git blame` / A-B 比对。
- 回滚：全部为文件/清单层改动，`git revert` 或还原对应快照即可；
  换行符提交可单独 revert。
- 无需 GPU。
