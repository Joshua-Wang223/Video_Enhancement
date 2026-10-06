---
name: esf-fill-advance-pointer-bug
description: ESRGAN _ensure_slot_free 排空超限兜底分支把该槽 FIFO 全部条目消费却只推进指针 1（应按消费条目数），属潜伏指针漂移缺陷；含两侧记账语义差异的准确表述与 code=8 机制仍未定论
type: project
---

# ESRGAN 排空超限兜底的指针推进量缺陷（2026-10-05 修复）

## 缺陷

`external/realesrgan_video/nvenc_sdk.py::_ensure_slot_free` 的兜底分支
（`[FIX-SLOT-BACKPRESSURE-B]`）把目标槽 per-slot FIFO 的**全部**条目以 prev 帧占位消费
（`while _dq: _dq.popleft()` + `del self._slot_pending[slot_idx]`），
但指针原先硬编码 `self._output_slot_idx += 1`。

per-slot FIFO 存 N>1 条是**常态**（跨 chunk 累积，`[FIX-SLOT-DEQUE]` 明确 append 永不覆盖），
于是 `+= 1` 只推进 1 ⇒ **指针落后 N−1** ⇒ 后续 drain 从错误物理槽起轮转
⇒ 相位漂移 / 帧错位。

IFRNet 同分支是 `self._output_slot_idx += _n_fill`（`_n_fill = len(_dq)`，**消费前**取），
本次修复即对齐该语义 ⇒ 两侧该处逻辑恢复同构。

**修复**：`[FIX-ESF-FILL-ADVANCE]`，把 `+= 1` 改为 `+= _n_fill`，
`_n_fill = len(_dq)` 取在 popleft 循环**之前**（popleft 后 deque 已空，len 恒 0）。

**⚠ 生产可达性**：该分支在 **8 份存档生产日志中命中数为 0**（`排空超限` /
`非法 size 已强制消费` 全未出现）⇒ **属潜伏缺陷，不是当前可见故障**。
修复是「消除潜在错误」，不是「修正在发生的错误」。故障注入开关：
`ESRGAN_NVENC_MAX_BS_BYTES=1024`（`:750`，仓内注释记载它能让 1 次钳制就打满 9 槽
并让 `_ensure_slot_free` 无限自旋）。

**测试**：`Accessory/test/test_esf_fill_advance.py`（11 项，纯 CPU，无需 GPU）——
用打桩 `_lock_bitstream_blocking` 返回空 + `_drain_outputs_blocking` 返回 `[]`
构造「guard 耗尽 ⇒ 必然进兜底」，参数化 N=1/2/3/5/9 验证指针推进量== 消费条目数；
另含 AST 断言（推进量必须来自 `_n_fill` 的 Name 节点，不得是字面量 `1`）
与「`_n_fill` 须在 popleft 之前取」的**剥注释**位置断言。
变异测试：还原成 `+= 1` → 7 failed；把 `_n_fill` 挪到 popleft 之后 → 8 failed。

## ⚠ 纠正：两侧记账差异的准确表述

先前 memory 写「IFRNet 严格单点 / ESRGAN 多点推进」是**错的**（只对
`_drain_outputs_blocking` 成立）。AST 全量统计：

| 推进点 | IFRNet | ESRGAN |
|---|---|---|
| `_drain_outputs_blocking` | 1 | **2** |
| `_sizecap_force_consume` | 1 | 0（内联在 drain 里） |
| `_ensure_slot_free` | **2** | 1（本次修复后语义与 IFRNet 对齐） |
| `encode_frames_stream` / `encode_frames_batch` | 1 | **2** |
| `_ce_harvest_slot` | 2 | 0（ESRGAN 无此方法） |
| `_ce_final_drain` | 2 | 2 |

准确说法：**两侧骨架同构**（正常取回 1 处 + 异常补偿若干），差异在**补偿点的分布与形态**
（IFRNet 把 sizecap 抽成独立方法；ESRGAN 内联且补偿更靠后）。

## code=8：**早已有定论**（2026-08-26），我一度误记为「基线缺口/未定论」

⚠ **纠正本条早先的错误表述**。`[[ifrnet-watercolor-tail-defect-investigation]]`
（2026-08-26）**已记载**：

> 「逐帧 drain 全程 code=8（INVALID_PARAM）、帧全靠段末 EOS 排空是**既有隐性模式**
> （追加 V 定性），[P0-FIX-RC-TOLERANT] 仅加了遥测。」

⇒ code=8 **不是新现象、也不是「基线缺口」**，是长期已定性的隐性模式。
本条早先写的「⚠ memory 与存档 README 都没记 code=8 的历史频次 ⇒ 基线缺口」**是错的**
（该记载在水彩调查 memory 里，不在 S8 存档 README 里；我只 grep 了后者）。
**教训**：下「某物无记载」的结论前，必须 grep **整个 memory 目录**，
不要只查自己刚写的那几篇。

## ⚠⚠ 更要紧的发现：同一条兜底分支，两侧「放弃槽位后」语义不同

| | IFRNet | ESRGAN |
|---|---|---|
| `_strict_eos` 字段 | **有**（`:643`，默认 `"1"`） | **完全没有** |
| 放弃目标槽时 | `raise RuntimeError("strict drain abandoned target slot")`（`:2096-2099`）⇒ **终止本段**，由 checkpoint/resume 重试 | 直接占位兜底 + `break`，**继续提交后续帧** |

而 [[ifrnet-watercolor-tail-defect-investigation]] 的**症状 A（尾帧参考链断裂 / CRA 重启组）**
主嫌疑原文正是：

> 「主嫌疑：`_ensure_slot_free` 的『排空超限→空帧占位兜底』路径**放弃槽位后继续提交**，
> 驱动 LA 链断裂自行重启 GOP（新 IDR/CRA），之后提交的帧落入重启组」

⇒ **ESRGAN 侧正是这条被点名的链**（IFRNet 已用 strict raise 堵上，ESRGAN 没堵）。
本次修的指针推进量只修了该分支的**记账**子问题，**没动「放弃后继续提交」这个语义**。

⚠ 当前 8 份存档生产日志中 `排空超限` 命中 **0** ⇒ 该链**目前未复现**
（与 2026-08-28「生产回归未再复现」一致）。但它是**已记录过的真实故障链**，
不是假想风险 ⇒ 若要动 ESRGAN 侧，**优先级高于**指针记账。

**Why:** 用户在评估「统一两侧排空记账」时发现了这个指针推进量缺陷。
⚠ 但复查 memory 后发现：**同一条兜底分支**上还有一个更严重的问题 ——
ESRGAN 缺 IFRNet 的 `_strict_eos` fail-fast（放弃槽位后继续提交），
而这正是 2026-08-26「症状A 尾帧参考链断裂 / CRA 重启组」记录的主嫌疑链。
⇒ 该分支的修复优先级：strict raise（语义）> 指针记账（本条）。
**How to apply:** 下「某现象无历史记载」前grep 整个 memory 目录（本条曾因只 grep
S8 存档 README 而误判 code=8 无记载）。
**How to apply:** 见到「某处把整个 FIFO 消费掉、指针却只 +1」一律按本条处理
（两侧都查一遍）；排查 NVENC 指针漂移类问题时，先确认 per-slot FIFO 深度是否 >1
（常态，不是异常）。判据类改动**必须** GPU 实测——本轮 CPU 单测全绿的移植
在 GPU 上直接丢帧。