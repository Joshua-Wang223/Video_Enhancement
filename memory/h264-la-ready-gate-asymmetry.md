---
name: h264-la-ready-gate-asymmetry
description: ESRGAN 侧 h264+LA>0 缺 IFRNet 那套 h264 就绪门（实测 code=8 的不对称来源），但「照抄 IFRNet 就绪门」在 ESRGAN 的 _ensure_slot_free 里会引入 prev 帧占位回归；含槽数 la+1 vs la+3 的两侧差异取证
type: project
---

# h264+LA 就绪门的不对称（2026-10-05，T4 取证）

## 事实：两侧不对称，且方向与直觉相反

| | IFRNet 侧 | ESRGAN 侧 |
|---|---|---|
| 槽数公式 | `_required_buffers = max(1, la_depth + 3)`，**无 codec 分支** | `if codec in ("hevc","av1"): la+3`，**h264 走 `la+1`** |
| h264 + LA=8 实际槽数 | **11** | **9** |
| `_hevc_ready_count` 定义 | 无门控，但**所有调用点**都在 `if codec in ("hevc","av1")` 内 ⇒ 实际等价门控 | 定义内建门控 `if codec not in (hevc","av1") or la<=0: return limit` |
| **h264 专用就绪门** | **有**（`_ensure_slot_free` 内 `if self._frame_idx - _oldest_gfi <= self._la_depth + 1` ⇒ not ready） | **没有**（只有 codec 门控的 hevc 检查） |

⇒ **该改的是 ESRGAN 侧**（它是 h264+LA 唯一没有就绪门的一侧），不是「IFRNet 已对齐、ESRGAN 没跟上」。

## 实测证据：code=8 只出现在缺门的一侧

T4 上h264_nvenc + vbr_hq + LA=8 跑批（100 s / 5 段 / 4999 帧）：
- **超分段（ESRGAN，slots=9=la+1）**：LA warmup 期出现 **3 次**
  `drain LockBitstream code=8 (slot=0, fi=1..3)`，诊断计数器停在 `#3` 再未增长
- **插帧段（IFRNet，slots=11=la+3）**：全程**零** code=8
- 帧守恒无损：5/5 段 `decoded==expected`，产物 4999 = 分段合计 4999，**零空帧补偿**

`code=8` 本身是已定性的既有容忍模式（`[P0-FIX-RC-TOLERANT]`，
`realesrgan_video/nvenc_sdk.py` 与 `ifrnet_video/nvenc_sdk.py` 逐字同构）：
LA 流式首排空期 `LockBitstream` 稳定返回 `INVALID_PARAM(8)` 而非 `NEED_MORE_INPUT(17)`，
帧由**段末 EOS 全量排空回收**。
⚠ **memory 与 `verification_report/s8_20261004_raw/README.md` 都没记code=8 的历史频次**
⇒ 「3 次」暂无历史基线可比（基线缺口，已记入本条）。

## ⚠ 关键：**不能**照抄 IFRNet 的就绪门（已实施并回滚的教训）

在 ESRGAN 的 `_ensure_slot_free` 里加 h264 就绪门，有两种写法，**两种都会引入比 code=8
严重得多的回归**，务必不要：

1. **落兜底分支（错）**：兜底分支（`[FIX-SLOT-BACKPRESSURE-B]`）会把该pending 以
   **prev 帧占位** 写进 `results`（`results[fi] = _prev_stream_h264`）⇒ **永久丢弃该帧真实
   码流**，用重复帧顶替。⇒ 直接违背刚修好的帧守恒（B1）。
2. **有界等待（错）**：`_ensure_slot_free` 在 `self._lock` 内、**提交新帧之前**被调用，
   而 `_frame_idx += 1` 发生在其**之后**（`encode_frames_batch` 内，`nvenc_sdk.py:2250`，
   调用点 `:2103`）⇒ **等待期间 `_frame_idx` 不会前进**，LA 延迟永远不满足 ⇒ 纯自旋。

⇒ **在 ESRGAN 现有结构下，h264 就绪门无法在 `_ensure_slot_free` 内实现。**
可行落点只能在「**提交之后**」的inline drain / per-frame drain 站点
（那里 `_frame_idx` 已前进），或改用 [FIX-SLOT-DEQUE] 式 free-pool 替代 `fi % slot_count` 轮转。

## 数据结构坑：ESRGAN `_slot_pending` 有**两种 entry 形状**

- `encode_frames_batch`（LA>0 分块流式，`nvenc_sdk.py:2032`）→
  4 元组 `(global_fi, bs_buf, force_idr, ep_status)`，`[0]` **是全局 fi**
- `encode_frames_batch_ce_pipeline`（LA=0，`:2463`）→
  5 元组 `(_ce, fi, ep_status, force_idr, bs_buf)`，`[0]` **是 ce_handle**

⇒ 任何读 `_slot_pending[...][0]` 当帧号的代码**必须先按路径/长度区分**，
否则在 LA=0 路径上会把 ce_handle 当帧号。
（IFRNet 侧 `_strm_slot_pending` 只有一种形状：`(fi_global, bs_buf, force_idr, ep_status)`。）

## 槽数改成 la+3 的可行性（若仍要走这条路）

- **显存代价可忽略**：每 slot = NV12 `W*H*1.5` + bs buffer + 1 CUDA event。
  1440x1152 下 NV12 = 2.37 MiB/slot ⇒ 9→11 每 session 多 **7~21 MiB**，close 即释放。
  ⇒ 「显存压力」不构成反对理由。
- **有测试钉死当前值**：`Accessory/probe/nvenc_sdk_realesrgan_suite.py:145,151`
  断言 LA=8 时 `_slot_count == 9` 且 `len(_slots) == 9`（**未传 codec ⇒ 默认 h264**）
  ⇒ 改公式必须同步改这两处断言，且该测试是 `@gpu` 标记（T4 可跑）。
- **⚠ 测得code=8 只在 slots=9 出现，但这是**相关不是因果**：LA warmup 期「什么都没就绪」
  与「槽位不足导致提交/排空循环依赖」两种成因都能产生 code=8，
  单凭 3 次观测**无法区分**。要区分需做 A/B：同素材、同 LA，只改槽数 9 vs 11，
  比code=8 次数 + 帧守恒 + 吞吐。

## ✅ A/B 已跑完（2026-10-05）：**槽数不是 code=8 的根因**（结论：保持 la+1，不改）

`Accessory/probe/ab_h264_la_slots.py`，100 s 素材 / 5 段 / h264_nvenc / vbr_hq / bs=8，
两臂**各 2 轮交替**，杠杆经`Ready` 行与 `[AB-SLOT-LEVER]` 双重确认生效：

| 臂 | 超分段 slots | code=8 | 段级守恒 | 验收 | 耗时 | 峰值 RSS |
|---|---|---|---|---|---|---|
| 9（la+1） | 9 | **3, 3** | 10/10, 10/10 不守恒 0 | 8/0/0 | 550.1 / 528.4 s | 5978 / 5285 MB |
| 11（la+3） | 11 | **3, 3** | 10/10, 8/8 不守恒 0 | 8/0/0 | 566.4 s | 7220 MB |

⇒ **预注册 D2：code=8 未减少（3→3）⇒ NO**。且 11 槽零改善却 **+2.9% 墙钟 / +1242 MB**。
**保持 ESRGAN h264 = `la+1`，不对齐 IFRNet。**

### 三条被A/B 澄清/新增的事实

1. **同run 对照最干净**：11 臂里**两个编码器都是 11 槽**，但只有 ESRGAN 那个报 code=8
   ⇒ 槽数相同下gate 有无才是差异变量（比 2026-09-04 那份日志更强：那份早于 gate 提交
   `1e57c0b`(2026-09-18)）。
2. **`_diag_lock_err_*` 是终身计数器，`_stream_begin` 从不清它**（只有
   `_diag_aux_block` / `_diag_phase_shift` / `_diag_slot_drain_fallback` 被重置）
   ⇒ **`#3` 不是"本段 3 次"而是"全程累计 3 次"**。
3. **3 次全部落在第 1 段 warmup**（两臂皆然，段 2~5 零命中）
   ⇒ 指向**首个编码器/会话的warmup 状态**，而非每段 LA 状态。
   ⚠ 「为什么是 fi=1/2/3 而不是 fi=4..8」仍**机制未明**（驱动内部 LA 填充状态，
   仓库内无任何探针可区分 `drain 返8` vs `返17`）——
   落门前应先加 `NVENC_LA_WARMUP_TRACE=1` 计数器，否则「告警消失」可能修错了东西。

### 已落地（零运行时影响）

- 修正 `external/realesrgan_video/nvenc_sdk.py` 槽数分支处的**错误注释**：
  旧注释写「h264 保持原 la+1（空槽返回 SUCCESS+size=0，**无此问题**）」——
  实测是**零余量**配置（复用时刻 `_frame_idx-_oldest_gfi == la+1`，恰好压在
  IFRNet就绪线上），h264 不出事只因驱动在边界龄帧上恰好返SUCCESS（巧合非保证）。
  现注释记录 A/B 结论 + 「别把就绪门加到 `_ensure_slot_free`」的两个理由。
- `[AB-SLOT-LEVER]` 注释改为「用途已了结，仅供回归对照，生产禁止设置」。
- **`_required_buffers` 公式逐字未动**（git diff 确认只改注释 + 杠杆）。

### 真正的修复仍待做 —— ⚠️ **已尝试并被 GPU 实测证伪（2026-10-05），本仓不加这道门**

曾按IFRNet 移植 `[FIX-H264-LA-DRAIN-READY]` 到 ESRGAN 的 `_drain_outputs_blocking`
（判据抽成 helper `_h264_target_slot_ready()`，未就绪即 `break`）。**CPU 单测全绿，
GPU 实跑立刻回归**：

| 指标 | 移植前（A/B 4 轮） | 移植后 |
|---|---|---|
| code=8 | 3（仅 warmup） | 0 |
| 门命中 | — | **196**（`frame_idx` 1→1024 **全程**，非仅 warmup） |
| 段 1 帧守恒 | 4999 == 4999 | **decoded=1024 expected=1025** |
| 结果 | rc=0，8/0/0 | **rc=1，片段 1 失败终止** |

**机制根因（⚠️ 2026-10-05 评估阶段修正了此处早先的不准确表述）**：

我先前写「IFRNet 仅 1 处推进 / ESRGAN 2 处」，**只对 `_drain_outputs_blocking` 成立**。
AST 全量统计（`test_h264_la_drain_ready_gate.py` 锁住）显示**两侧都是多路径推进**：

| 推进点 | IFRNet | ESRGAN |
|---|---|---|
| `_drain_outputs_blocking` | 1 | **2** |
| `_sizecap_force_consume` | 1 | 0（**内联**在 drain 里，行 1852） |
| `_ensure_slot_free` | **2** | 1 |
| `encode_frames_stream` / `encode_frames_batch` | 1 | **2** |
| `_ce_harvest_slot` | 2 | 0（无此方法） |
| `_ce_final_drain` | 2 | 2 |

⇒ 准确表述：**两侧语义骨架同构**（成功取回 1 处 + 异常补偿若干处），
差异在**补偿点的分布与形态**（IFRNet 把 sizecap 抽成独立方法 + `_ensure_slot_free` 有 2 处；
ESRGAN 内联 sizecap 且补偿更靠后）。
**不是「一个严格一个松散」。**

**关键补充（决定性的生产事实）**：ESRGAN 那条额外的 sizecap 强制消费分支
（`self._slot_pending.pop(slot_idx, None)` 后推进）
**在所有存档生产跑批中从未触发**（查`/tmp/ab_slots/rep*/out*`、`/tmp/s8b/out`、
`/tmp/gate_b2/out` 共 8 份日志，`非法 size 已强制消费` / `sizecap force-drop` 命中数**全为 0**）
⇒ **它目前是死代码**。
⚠ 因此**不能把code=8 或移植失败归因到「sizecap 破坏了指针同步」** ——
那是我早先未取证就下的结论。真实机制（门为何恒判未就绪）**至今未定论**，
缺的是「指针 vs 队首 gfi」的运行时对照 instrumentation（见下）。

⇒ 现阶段结论：**code=8 无害**（帧守恒、零占位、S1~S8 全绿），不需要修；
「统一两侧记账」**尚无证据支撑为必要**，其前置是先补 instrumentation把机制定论。

⚠ **三条硬约束（都实证过，别再重犯）**：① 不能放 `_ensure_slot_free`
（未就绪落 `[FIX-SLOT-BACKPRESSURE-B]` 用 prev 帧顶替真实码流；且该函数在锁内、
`_frame_idx += 1` 之前 ⇒ 等待纯自旋）；② 不能放 drain 循环（本次证伪）；
③ 判据须按 entry 长度分派（LA>0 是 4 元组 `[0]`=gfi，LA=0 是 5 元组 `[0]`=ce_handle）。

### 两条硬约束（已实证过，别再重犯）

① **不能放 `_ensure_slot_free`**：其未就绪分支会落进 `[FIX-SLOT-BACKPRESSURE-B]`
用 prev 帧顶替真实码流；且该函数在锁内、`_frame_idx += 1` **之前**调用
⇒ 等待永不满足（纯自旋）。
② **判据必须按 entry 长度分派**：LA>0 是 4 元组 `[0]`=gfi，LA=0 是 5 元组 `[0]`=ce_handle。

`Accessory/probe/nvenc_sdk_realesrgan_suite.py` 的导入路径在 `1565908`（测试资产迁入
`Accessory/`）时**漏改一层 `..`** ⇒ 恒 `No module named 'nvenc_sdk'`
⇒7 failed + **22 个 @gpu 测试被静默 skip**（`_HAS_GPU` 依赖该 import）。
修正后 **30 passed / 1 skipped**（不需外部 PYTHONPATH 即可跑）。
⚠ **教训**：「@gpu 标记 + skip」会让人以为「本机无 GPU 才skip」，
实际是 import 失败 ⇒ 静默跳过了整份硬件测试；查 skip 原因时要看 `_HAS_*` 变量而非标记。

**Why:** 用户看到超分段日志里的 code=8 提出「h264 也该按hevc 用 la+2/3」，
但真正的不对称是**就绪门的有无**，不是槽数；而照抄就绪门会因 ESRGAN 的调用位置
（锁内、提交前）与兜底语义（prev 占位）引入更严重的帧损坏。
**已实测定案（2026-10-05）**：改槽数**无效**（9/11 槽 code=8 均为 3 次）且有代价
（+2.9% 墙钟 / +1242 MB）⇒ **保持 la+1**。
**How to apply:** 再遇 LA>0 h264 排空异常，先分清「槽位不足」还是「缺就绪门」——
**后者才是真因**（已A/B 排除前者）。修就绪门只能加在**提交后**站点，
且判据须按 entry 长度分派。**不要**在 `_ensure_slot_free` 内加等待或兜底。
⚠ **`#3` 是终身计数不是分段计数**：`_diag_lock_err_*` 从不被 `_stream_begin` 重置，
读日志时别把它当「本段 3 次」。