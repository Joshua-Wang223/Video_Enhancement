---
name: gate-stub-session-gen-audit-2026-10-10
description: 审核其他会话遗留的 10 份 verification_report 产物，查出 plan_gate 的 BEH-ERR 组级异常长期吞掉 12 项检查（89 实为 101 项）；根因是 d215fc2 加了 _session_gen 却没同步门禁桩，已修复并复核 97 PASS / 0 FAIL
metadata:
  type: project
---

2026-10-10 审核工作区里 10 份未跟踪的 `verification_report/` 产物（其他会话 05:42~06:23 留下），
查出一处**长期存在、无人察觉的门禁盲区**。

## 1. 这 10 份是什么

| 文件 | 轮次 | 命令 / 环境 |
|------|------|-------------|
| `CRF_CQ统一验证{报告,结果}_20261009_054553 / 055613 / 062348` | 3 次 | `crf_cq_unification_verify.py --quick --no-gpu`，**NVENC 不可用**，PASS=103 / FAIL=0 / WARN=0 / SKIP=12 |
| `verification_report_20261009_054234 / 054516` | 2 次 | `plan_implementation_gate.py`，**无 GPU**（R5/R7 WARN，`Cannot load libcuda.so.1`），89 项 / 84 PASS / 3 WARN / 2 SKIP |

**审核结论：重复品。**
- 3 份 CRF_CQ 报告归一化时间戳后 **diff = 0 行**，是同一轮的三次重跑；
- 2 份 plan_gate 报告仅差**指针地址**（`0x564dbc…` vs `0x56346f…`）与**临时目录名**（`/tmp/tmpkpd39e18` vs `/tmp/tmpbe0s3nqn`），实质同一轮。
⇒ 10 份文件实际只代表 **2 次独立运行**。无敏感信息（token/密钥扫描 0 命中），合计 240K。

价值：这是**无 GPU 轮**的可审计留痕，可部分补上 [[l40-verification-scope-2026-10]] 记的
「CPU 侧只有 commit message 无报告留痕」缺口（限于 `crf_cq_cpu` 与 `plan_gate` 两项）。

## 2. ⛔ 主发现：BEH-ERR 组级异常长期吞掉 12 项检查

审核时注意到 05:42/05:45 的报告（**晚于** `d215fc2` 修复提交 04:39 整整 1 小时）仍报：

```
| WARN | BEH-ERR | beh_group_g | 行为执行 |
  组级异常: AttributeError: 'NVENCEncoder' object has no attribute '_session_gen' |
```

而 `Plan/PROMPT_L40_全流程验证.md` 明写「该缺陷已于提交 `d215fc2`（04:39）修复，该报告早于修复 12 分钟」。
⇒ **该结论不成立**：修复一小时后仍在复现。

### 根因：门禁桩没跟上生产改动（**生产代码是对的**）

1. `d215fc2` 给生产侧加了会话代数 `_session_gen` / `_cached_sps_pps_gen`
   （`external/ifrnet_video/nvenc_sdk.py:760`、`:2360-2362`），逻辑本身**正确**。
2. 但 `plan_implementation_gate.py` 的桩 `make_enc()` 走 `NVENCEncoder.__new__(NVENCEncoder)`
   —— **绕过 `__init__`**，只手工设了 5 个属性，**没有 `_session_gen`**。
3. `make_enc_for_begin()` 随后调 `e._stream_begin(force=True)`，撞上 `:2360` 的
   **非防御式** `self._session_gen`（同一行的 `getattr(self,'_cached_sps_pps_gen',-1)` 却是防御式，
   这个不对称就是线索）⇒ 抛 `AttributeError`。
4. 异常冒泡到组级 ⇒ 记成一条 `BEH-ERR` WARN，**整组检查结果全部丢失**：
   **门禁长期只报 89 项，实际应为 101 项**——`BEH-G9/G10/G11` 等 12 项**从未真正执行过**。

⇒ 后果：`d215fc2` 自称「Resolves BEH-G9 … plan_gate: 86 PASS / 0 FAIL」的**回归测试本身没跑过**，
其修复处于**未验证**状态（正是 [[feedback_artifacts_not_execution_evidence]] 的典型形态）。

## 3. 修复与复核

`make_enc()` 桩补两个属性，**取值必须与真实序列一致**：
`__init__` 给 `_session_gen=0` / `_cached_sps_pps_gen=-1`，而 `_stream_begin` 每段开头会把
`_cached_sps_pps_gen` 同步成 `_session_gen`。故「同一会话、缓存已填充」应取**二者相等（0/0）**。

⚠️ **踩过的坑（已排除误报）**：第一次取 `-1/0`，`_stream_begin` 判定「缓存属旧会话」而清空，
跑出 `BEH-G9 FAIL`。**这不是生产缺陷**，是桩取值不符真实序列，改成 `0/0` 后即 PASS。
（`_cache_param_sets` 只写 `_cached_sps_pps` 不写 `_cached_sps_pps_gen`，但因 `_stream_begin`
每段开头同步，二者始终一致 ⇒ 生产逻辑自洽。）

复核（同环境、同命令 `python3 Accessory/verify/plan_implementation_gate.py`）：

| 状态 | 项数 | 通过 | 失败 | 警告 | 跳过 |
|------|------|------|------|------|------|
| 修复前 | 89 | 84 | 0 | 3（含 BEH-ERR） | 2 |
| 修复后 | **101** | **97** | **0** | 2 | 2 |

`BEH-G9/G10/G11` 三项恢复执行且**全部 PASS** ⇒ 确认生产侧 `d215fc2` 修复有效，
缺陷**仅在门禁桩**。

⚠️ 修复后剩余 2 条 WARN 均为**环境差异**（R5 CUDA 不可用、R7 NVENC 探测失败），非功能失败。

## 4. 教训

- **组级异常会掩盖整组检查**：`BEH-ERR` 这类「一条 WARN 概括整组」的设计，使 12 项静默消失长达数小时，
  且总数从 89 变 101 时**没有任何告警**。新增组级兜底时，应在异常分支里补一条
  「本组 N 项未执行」的显式计数，避免总数被无声改小。
- 「某缺陷已由 commit X 修复」必须有**晚于该 commit 的实跑报告**佐证；本例中修复提交自带
  「plan_gate 86 PASS / 0 FAIL」的描述，但因同一门禁自己坏了，该数字不可采信。
- 审核他人在制品时，先做**归一化 diff** 判重复（本例 10 份实为 2 轮），再动结论。

## 相关

- [[l40-verification-scope-2026-10]] — 无 GPU 轮证据缺口的原始记录（本条部分补上）
- [[feedback_artifacts_not_execution_evidence]] — 产物存在 ≠ 测试执行过（BEH-G9 的形态）
- [[l40-full-verification-run-2026-10-09]] — 同日 L40 实跑现场（落表双轴 PASS / G7-6 未达成）