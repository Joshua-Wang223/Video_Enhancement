---
name: ifrnet-hevc-la-slot-headroom-deadlock
description: "HEVC+LA=8+VBR_HQ 段卡死：LA 输出延迟实测=LA+1，而物理槽仅 LA+1，余量为 0 导致提交↔排空循环依赖，LockBitstream 在驱动内永久阻塞"
metadata:
  node_type: memory
  type: project
  status: fixed
---

# IFRNet HEVC + LA>0 物理槽余量不足导致编码线程死锁

## 症状

`hevc_nvenc` + `vbr_hq` + `--lookahead-depth-ifrnet 8` 时，段 1 处理到第 ~130 帧永久卡死：

```
[GPU0] 插帧:  92%| 321/348 [00:45<...]      ← 进度停在此处
[AUTO-TUNE-RETUNE] 实测 T2=445.4ms（全段 19 batches）  ← T2 已完成
[IFRNet-Writer] ❌ 写线程在 140s 内未退出（疑似死锁），段处理失败
[FFmpegMuxer ERR] [out#0/mp4] Output file does not contain any stream
```

**注意"写线程未退出"是第二手症状**：真正卡死的是 `NVENC-Enc` 线程，写线程只是被 `submit()` 的 `queue.put()` 堵住。且此时**编码线程异常路径不会触发**——它卡在 `LockBitstream` 内部而非抛异常，所以 `self.error` 永远为 None，`flush_and_join()` 那句"编码线程未在 Ns 内退出"**不会打印**。不要据此误判为写线程自身的问题。

## 定位方法（本次关键手段）

**py-spy 抓栈 + `--locals`**（而非靠日志推测）：

```bash
py-spy dump --pid <PID> --locals
```

现场：

```
IFRNet-Writer (idle)   → put (queue.py:140)
                       → submit (nvenc_sdk.py:3199)        # 编码队列已满
NVENC-Enc     (active) → _drain_outputs_blocking (nvenc_sdk.py:1444) → lock_bs_fn()
```

`active`（非 idle）是判定"卡在驱动调用内部、而非 Python 层轮询"的决定性证据。

再临时加诊断（环境变量开关 `IFRNET_NVENC_TRACE=1`，段首 20 帧打印），拿到计数轨迹：

```
pf.drain i=0..7   max=0                              # LA 预热，正确不 drain
pf.drain i=8  frame_idx=9  out_idx=0  max=1
drain.enter max=1 out_idx=0 slot=0    drain.exit n=0 out_idx=0   ← gfi 0 仍未就绪
esf.enter slot=0 pending=1 frame_idx=9 guard=0,1,2,...N         ← 无限重试 → 挂死
```

## 根因

`_required_buffers = max(1, la_depth + 1)` → LA=8 分配 9 个物理槽。

但**实测 LA 输出延迟是 `la_depth + 1`，不是 `la_depth`**：提交 gfi 8（frame_idx=9）时锁 slot 0 仍返回 `NEED_MORE_INPUT`（drain 返回 n=0）。"buffers ≥ LA+1" 只是驱动不报错的下限，**不是物理槽可安全复用的下限**。

于是余量为 0，slot 0 恰在 gfi 0 就绪**前一刻**被要求复用，形成循环依赖：

```
_ensure_slot_free(0) 要求"先排空 gfi 0 才能提交 gfi 9"
        ↑                                    ↓
排空 gfi 0 需要提交更多帧  ←────────────────┘
```

`_ensure_slot_free` 是唯一在**提交前**无条件发起 Lock 的站点，于是它反复锁一个未就绪的槽，最终在驱动内永久阻塞。

### 关键反直觉点：`doNotWait=1` 救不了

既有的 `P0-FIX-HEVC-DRAIN-HANG` 已让 HEVC/AV1 走 `_drain_poll_mode`（`doNotWait=1` + 2s deadline）。现场显示 `poll=True **同样挂死**——**驱动侧的阻塞发生在 `lock_bs_fn()` 调用内部，`doNotWait` 参数并不能解除。

结论：**HEVC/AV1 下唯一可靠的保护是在软件层保证"绝不锁未就绪的槽"**，不能依赖 Lock 的调用参数。

### 一个被证伪的假设（避免重复踩坑）

最初怀疑是"AUX 辅助块（VPS/SPS/PPS）推进了 `_output_slot_idx` 但未消费 FIFO，导致指针跑到 pending 记账之前"。诊断后**证伪**：日志未打印 `ℹ️ 辅助块 #1`，`Cached SPS+PPS: 101 bytes` 来自 `_cache_param_sets()` 的**正常 IDR 帧路径**（活跃版是 `external/ifrnet_video/nvenc_sdk.py`，HEVC 的 SPS/PPS 随 IDR 一起返回，不是独立辅助块）。基于该假设的改动已全部回滚。

## 修复

文件：`external/ifrnet_video/nvenc_sdk.py`（同步到历史单文件版 `external/IFRNet/process_video_v6_4_5_1_single.py`）

1. **`_required_buffers = max(1, la_depth + 2)`** —— 根治。为 LA 输出延迟预留 1 帧，从结构上消除循环依赖（LA=8 → 10 槽）。LA=0 路径不受影响（此时仍由 `pipeline_depth` 决定）。
2. **`_ensure_slot_free` 就绪上界保护** —— 防御。`ready = frame_idx - la_depth - output_slot_idx`；`ready ≤ 0` 时**不发起任何 Lock**（含 target_probe），直接走空帧占位兜底。注：pending 非空 ⟹ `frame_idx > output_slot_idx` ⟹ LA=0 时 ready 恒 > 0，故 H.264/LA=0 行为完全不变。
3. **target_probe 取帧后补偿推进 `_output_slot_idx`** —— 记账一致性。该路径绕过 `_drain_outputs_blocking`，不补偿会让指针落后于 FIFO 实际消费量，导致后续 drain 重复锁同一个已取空的槽。

## 验证

| 项 | 修复前 | 修复后 |
|---|---|---|
| 结果 | ❌ 段失败 | ✅ 3 段全部成功，2分48秒 |
| 物理槽 | 9 | 10 |
| 告警 | — | 0（无空帧占位/排空超限/帧数守恒失败） |
| 帧数 | — | 1591 = 2×797−3 ✓（每段 2n−1，含 f0） |
| 解码 | — | 1591 帧零错误 |

## 关联

- [[esrgan-nvenc-slot-backpressure]] —— **同一根因的另一种表现**。该记忆已指出"`la_depth=8` 与 `slot_count=9` 之间只有 1 帧余量"，但其表现是帧被覆盖丢失、通过"提交前强制排空"(`_ensure_slot_free`) 解决；本次证明余量实际是 **0 而非 1**，且当 `_ensure_slot_free` 无法排空时（HEVC）会退化为死锁。**两者应合并理解：LA 输出延迟 = LA+1，物理槽下限应为 LA+2。**
- [[hevc-la-drain-diagnosis]] —— HEVC 阻塞死锁的另一处（分块 lookahead）；共同点是 HEVC 下空槽 Lock 不可靠。
- [[nvenc-la-frame-conservation-fix]] —— 记录了 "pipeline_depth = LA+1" 的修复，本次将其**精化为 LA+2**，属结论演进而非回退。

## 排查教训：先确认活跃代码路径

`CODEBUDDY.md` 的"Current vs. Historical Files"表**已过时**（列出的 `ifrnet_processor_v6_1_single.py`、`process_video_v6_3_5_single.py` 均不存在）。实际链路：

```
src/main_video_optimized.py
  → src/processors/ifrnet_processor_video_optimized.py   (打印 "v6.4.5.1")
  → external/ifrnet_video/                                (包：nvenc_sdk.py / pipeline.py / main.py)
```

`external/IFRNet/process_video_v6_4_5_1_single.py` 是**拆分母本/历史文件**，日志里的 "v6.4.5.1" 来自 processor 层而非该文件。本次一开始误改了历史文件，靠 subagent 核对 import 链才纠正。**排查前务必先确认真实调用链。**

## 排查铁律

多线程/多进程卡死，**先用 py-spy dump 定位卡死线程与栈帧，再分析代码**。不要靠日志顺序推测——本例中所有"写线程超时"的日志都指向错误的方向（真正的卡点在另一个线程，且它不抛异常、不打日志）。项目注释里"py-spy 一步定位 GIL 竞争"的铁律同样适用于 NVENC 驱动阻塞。
