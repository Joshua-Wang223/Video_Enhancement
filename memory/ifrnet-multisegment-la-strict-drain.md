---
name: ifrnet-multisegment-la-strict-drain
description: 已修复：IFRNet 多段运行第 2 段 strict drain abandoned target slot（根因=段内 fi 当全局槽号取模）＋ 全 hevc 第 1 段 strict EOS 漏取 9 帧（根因=SUCCESS+size==0 被当成排空结束）
type: project
---

# IFRNet 多段运行第 2 段 `strict drain abandoned target slot`（已修复）

**状态**：2026-09-18 定位 + 修复 + GPU 验证。**既有缺陷，非 commit `765eeb8` 引入**
（该 commit 不含任何 `external/ifrnet_video/` 文件）。

## 症状

任何 “分段数 ≥ 2” 的运行，在 IFRNet 第 2 段抛错并中止：

```
[IFRNet-Writer] 写线程异常: RuntimeError: [NVENC-Enc] 编码线程异常:
  [NVENC-Enc] 编码线程异常(LA chunk encode) frame_idx=1660
  pending_slots=10 pending_count=10:
  [NVENCEncoder] strict drain abandoned target slot: slot=10, pending_fis=[1649]
→ [FFmpegMuxer ERR] Error opening output file ... Invalid argument
```

单段运行正常；与 codec（h264/hevc）和分段时长（10s/25s）无关。

## 根因（代码级，已用插桩复现确认）

一句话：**排空条目回传的是「段内 fi」，下游却按「全局槽号」取模反查物理槽。**

- `_drain_outputs_blocking()`（`external/ifrnet_video/nvenc_sdk.py:1736-1745`）从该槽
  pending 队首取全局 gfi 后，回传的是 **段内 fi**：
  ```python
  _actual_fi = _actual_gfi - self._strm_ts_base     # ← 段内 fi
  outputs.append((_actual_fi, out_ts, h264_data))
  ```
- `_apply_drained_entries()`（同文件 1876）用它反查物理槽：
  ```python
  _drain_slot = _est_fi % self._slot_count          # ← 当成全局 fi 用
  ```
- 而物理槽分配用的是**全局**序号：`slot = self._frame_idx % self._slot_count`（2376）。

段 1 时 `_strm_ts_base == 0` → 段内 fi == 全局 gfi → 巧合正确（这也是为什么单段一直没事）。
段 2+ 时 `_strm_ts_base = 段1帧数`，若 `_strm_ts_base % _slot_count != 0`，算出的槽号就**错位**：
FIFO 查不到 → `_diag_slot_mismatch` → 该帧 pending **永不消费**，而 `_output_slot_idx` 已推进
→ 槽位被"带病"复用 → `_ensure_slot_free` 耗尽 guard 后抛 strict。

插桩实测（seg1=200 帧、seg2 起 `_strm_ts_base=200`、`_slot_count=11`）：
```
DRAIN fi=210 ... got=[0]  -> out=201          # 回传段内 fi=0，但该帧物理槽 = 200%11 = 2
ESF-enter fi=211 out=202 ... slot=2 pend={2:[200], ...}
DRAIN fi=211 ... max=1 got=[]  (×N)           # 槽 2 的 FIFO 永不清空
FAILED: strict drain abandoned target slot: slot=2, pending_fis=[200]
```

## 修复

`_drain_outputs_blocking()` 改为回传**全局 gfi**（`_entry_deque[0][0]`），兜底分支仍回传
`_output_slot_idx`（其 `% _slot_count` 同样是刚锁定的物理槽）。标记 `[FIX-SEGMENT-SLOT-KEY]`。

**影响面（已核对）**：`_strm_slot_pending` 仅在 `encode_frames_stream()` 内写入（2501-2502），
故该分支只在流式 LA 路径命中；ce_pipeline（`_slot_pending` 列表）走兜底分支，行为**逐字不变**。
下游 `_apply_drained_entries` 只用条目首元素算槽号（真实帧号取自 FIFO 队首），不把它当帧索引。

## 验证（T4 / CUDA 13.0）

- 最小复现：单会话 2 段 ×400 帧（chunk=128，段间 `_stream_begin(force=True)`）
  → 修复前 seg2 抛错；修复后 **seg1 400/400、seg2 400/400，empty=0 → PASS**。
- 端到端 3 段（75s 输入、`--segment-duration 25`、IFRNet h264 + ESRGAN hevc）：RC=0，
  3/3 段解码级验收通过（1649/1149/649），`verify_segment_bitstream_v4` 最终输出 **PASS**
  （frames=packets=3447、frame_num 无回退、无 pts_anomaly、色度正常）。
- 端到端 3 段全 hevc（`--segment-duration 10`）：3/3 段通过。
- 回归：单段端到端 1149 帧与修复前一致；`verify_plan` 94 项 / 92 PASS / 0 FAIL；
  `pytest -k frame_conservation` 6 passed；隔离套件 0 FAIL / 0 CRASH。

## 第二个根因（同轮一并修复）：HEVC EOS 把 `SUCCESS+size==0` 当成"排空结束"

同一场景还有**独立**的一处：全 hevc 时第 1 段 `(EOS final chunk) strict EOS left
undecoded AU(s): [490..498] (count=9)`（`_strm_ts_base==0`，与上面的槽号 bug 无关）。

插桩实测（`IFRNET_EOS_DEBUG=1`）：

```
[EOS-DBG] drain_slots=[5,6,7,8,9,10,0,1,2,3] pending={5:1,6:1,...,3:1} out=489 fi=499 donotwait=1
[EOS-DBG] slot=6 break size==0 after 1 locks, pending=1     ← 首锁即 break
[EOS-DBG] slot=7 break size==0 after 1 locks ...
（slot 5 取到数据；其余 9 槽全部首锁 size==0 → 遗留 9 帧 = LA+1）
```

根因：**HEVC/AV1 + `doNotWait=1` 时，"尚未产出"的槽返回的是 `SUCCESS + size==0`
（不是 `NEED_MORE_INPUT`）**。原代码 `if _bs_size == 0: break` 把"还没好"误判成
"排空结束" → 整条 LA 尾（9 帧）被遗留 → strict 抛错。EOS 后这些帧**必然**产出，
只是需要几 ms。

**为什么纯编码压测复现不到**：空闲 GPU 下这些帧已就绪（返回真实 size）；
端到端有 T2 推理并发占 GPU 时才稳定复现（实测 E2E 2/3→3/3 失败，压测 0/3）。

修复（`[FIX-HEVC-EOS-ZEROSIZE]`）：`size==0` 在 HEVC 下改为**在停滞窗口内重试**
（与既有"非法 size 重试到 deadline"同口径），窗口耗尽才 break。

同时修正停滞窗口的两处实现（`[FIX-EOS-STALL-DEADLINE]`）：
- 由"整轮 EOS 共用一个 5s 绝对截止"改为**每槽独立**，可用 `NVENC_HEVC_EOS_STALL`（默认 10s）覆盖；
- 窗口刷新只在**真正取到一帧**后发生（此前误写成"LockBitstream 成功即刷新"，
  会让一直 `size==0` 的槽无限续期）。

## 验证（T4 / CUDA 13.0，两处修复合并后）

- 最小复现：单会话 2 段 ×400 帧 → seg1/seg2 各 400/400，empty=0 → PASS。
- 全 hevc 多段端到端 ×3：**3/3 全部 RC=0**，IFRNet 3/3 段，逐段解码级 499/499/149。
- h264 多段端到端（75s/3 段）：RC=0，IFRNet 3/3、ESRGAN 3/3。
- `verify_segment_bitstream_v4`：hevc 多段最终输出 PASS（frames=packets=1147、
  无 frame_num 回退、无 pts_anomaly、色度正常）；h264 3 段输出 PASS（3447）。
- 回归：单段端到端 1149 帧（与修复前一致）；`verify_plan` 94 项 / 92 PASS / 0 FAIL；
  `pytest -k frame_conservation` 6 passed；隔离套件 0 FAIL / 0 CRASH。

## 调试开关
- `IFRNET_EOS_DEBUG=1`：打印 EOS 排空的 drain_slots / pending / 每槽 break 原因。
- `NVENC_HEVC_EOS_STALL=<sec>`：HEVC EOS 每槽停滞窗口（默认 10s）。
