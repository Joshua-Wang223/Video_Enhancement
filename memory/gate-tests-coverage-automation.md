---
name: gate-tests-coverage-automation
description: 门禁把 tests/ 全量纳入 py_compile + H1 扫描（54→112 文件），并据此抓到一处被 except 掩盖多年的真实参数互斥缺陷
type: project
---

# 门禁覆盖扩到 `tests/`（2026-09-15）

立项：`Plan/门禁与测试资产纳管清理_立项Prompt.md`
清点表：`tests/TEST_ASSET_INVENTORY.md`（59 个 py 文件的逐个状态 + 引用矩阵 + 哈希）

## 落地内容

| 项 | 变化 |
|---|---|
| `COVERAGE_ROOTS` | 新增 `"tests"`（此前 tests/ **完全不在扫描范围**，只有 `test_regression_min.py` 经 `COVERAGE_EXTRA` 单点纳入） |
| `COMPILE_TARGETS` | 54 → **112** 个文件 |
| `BEH-H1` 扫描调用点 | 65 → **245** |
| `BEH-E2` | 新增「tests/ 未被覆盖」探测（防止排除规则写宽后整批静默剔掉）；`COVERAGE_MIN_FILES` 45 → 95 |
| 删除 | `COVERAGE_EXTRA`（tests 成为根目录后成死配置） |

**为什么敢全量**：实测 59 个 `tests/**/*.py` **全部** `py_compile` 通过、
`BEH-H1` 调用契约也无违规 —— 立项文档担心的"历史脚本会立刻翻红"在本树不成立，
所以不需要维护"活跃 tests 白名单"。

## 立刻抓到的真实缺陷（值得记住的一类）

`tests/diagnose_nvenc_rc_mode.py:178`：

```python
subprocess.run([...], capture_output=True, text=True, timeout=30,
               stderr=subprocess.DEVNULL)      # ← 互斥，运行时必抛 ValueError
```

`capture_output=True` 已含 `stdout/stderr=PIPE`，再传 `stderr=DEVNULL` 会抛
`ValueError`；而它被外层 `except Exception: pass` **静默吞掉**，
于是这段"兜底全盘查找 `nvEncodeAPI.h`"**从未真正生效**。

⇒ **`py_compile` 看不出这类错误**（语法合法、运行必崩），只有 `BEH-H2` 的
参数组合扫描能抓到；"语法过了就没事"是错觉。

## 实测纠正：v4/v5 与 `_v4.py` 引用

* `tests/verify_segment_bitstream_v4.py` 与 `_v5.py` **逐字节相同**
  （sha256 前 16 位均 `2566804141ee2c7a`，均 192131 字节），但**都是有意保留**的
  "同一份内容两个名字"，**两个都不能删**。
* `test_chroma_false_positive.py` 的 `v5 → v4` 双名兼容（`[FIX-VERIFY-RENAME]`）
  在本轮开工前**已经落地**（先跑基线才发现，见 `feedback_verify_baseline_first`）。
* 活代码/文档里指向 `_v4.py` 的过时引用已更正 13 处。
  **有意不改**的是 v4/v5 **文件内部自指的日志名** `verify_segment_bitstream_v4_stuck.log`
  —— 一改就破坏两者逐字节一致，而那正是核对"两个名字同源"的手段。

## 换行符：本机不动（以 Linux 侧为准）

本机 53 个活跃生产 py：4 个纯 CRLF + 1 个混合（`ifrnet_utils.py`）+ 48 个 LF。
使用者裁定「**以 Linux 侧为准，本机只是开发环境**」，故本机**不新增
`.gitattributes`、不改这 5 个文件的字节**（本树非 git 仓库，"独立提交隔离 diff"
的前提不成立，归一化只会产生不可复核的大 diff 并整体传导）。
Linux 侧待办：确认 5 个文件在 git 索引里的形态 → 加 `* text=auto` + `eol=lf` →
**单独一次提交**归一化。

## 版本控制

本开发树**不是 git 仓库**（`git status` → `fatal: not a git repository`），
故立项里「复核 `verify_segment_bitstream_v4.py`/`_v5.py` 是否被跟踪」无法在本机做。
`memory/env-ffmpeg-ffprobe-gotchas.md` 记录 v4 **未被跟踪**，而它是被生产源码引用的
门禁资产 ⇒ 必须在 Linux 侧纳入跟踪。

## How to apply

* 新增 `tests/*.py` 会自动进 `COMPILE_TARGETS`，**无需**手工登记；
  但把**非 py 资产**（`.sh`/`.md`）加进去仍需显式处理。
* `BEH-H2` 的"参数互斥"扫描是 `py_compile` 的盲区补充，遇到"某段兜底逻辑从未生效"
  类问题可以先怀疑它。
* 新增 `tests/` 文件后，`tests/TEST_ASSET_INVENTORY.md` 的行需要手工补
  （或用文档 §2 的引用矩阵脚本重生成）。

**Related**：[[gate-verify-plan-known-failures]]、[[feedback_verify_baseline_first]]、
[[gate-segment-validation-metadata-shortcut]]
