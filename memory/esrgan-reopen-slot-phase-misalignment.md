---
name: esrgan-reopen-slot-phase-misalignment
description: "[历史] reopen() 时代的 slot 相位错位 — 已随 reopen() 删除而消失"
metadata: 
  node_type: memory
  type: project
  status: historical-reference
  superseded_by: esrgan-cross-segment-optimization-complete
  tags: 
    - realesrgan
    - nvenc
    - cross-segment
    - lookahead
    - historical
  originSessionId: 0567f4ad-2f7a-4f67-a136-73bc484b316b
  modified: 2026-07-29T08:09:30.595Z
---

> **⚠️ 问题已随架构演进消失（2026-07-29）**：
> - `reopen()` 方法已删除，`_frame_idx`/`_output_slot_idx` 不再段间归零
> - 当前架构：驱动会话跨段持续，frame_idx 单调递增，相位自然一致
> - 全貌见 [[esrgan-cross-segment-optimization-complete]]
    - reopen
---

# ESRGAN reopen() slot 相位错位（FIX-REOPEN-SLOT-PHASE）

## 症状

注释屏蔽 main.py `[FIX-HIGHRES-RC]`（≥1080p 输出回退 vbr_hq + LA=8）后：
段 1 正常完成，段 2+ 报大量
`[NVENC-Enc] ⚠️⚠️ _ensure_slot_free 等待 slot=x 排空超过预期轮次仍未完成，
可能存在编码器状态异常，放弃等待以避免死锁（可能导致该 slot 数据被覆盖丢失）`。
告警刷屏的同时段 2+ 实际在丢帧，产物不可用。

## 根因

LA>0 chunked 路径（`encode_frames_batch`）的一对耦合计数器：

- 提交 slot = `_frame_idx % slot_count`（LA=8 时 slot_count = 9）
- drain 顺序 = `_output_slot_idx % slot_count` 轮转
  （`_drain_outputs_blocking` / `_ensure_slot_free`）

**两者是同一相位**：est_fi % N 必须等于该帧的提交 slot。

段 1 结束 EOS drain 后 `_frame_idx` = F1（段 1 总帧数）。段间 `reopen()`
重建驱动会话时归零了 `_output_slot_idx = 0`，但**刻意保留了 `_frame_idx`**——
那是 [[esrgan-segment-reuse-frame-idx-reset]] 时代（session 跨段持续存在）
的约束遗留："时间戳回退会破坏驱动 LA 状态机"。

段 2：提交从 slot F1%9 起、drain 从 slot 0 起 → blocking LockBitstream
永远命中输出未就绪（或本会话从未提交）的 slot → 立即 NEED_MORE_INPUT →
drain 永久空转 → `_ensure_slot_free` 守卫 36 轮（slot_count×4）后放弃 →
告警 + 带病覆盖 slot → 丢帧。

段 1 正常是因为 gen=1 两计数器同从 0 开始，天然同相位。
FIX-HIGHRES-RC 开启时掩盖此 bug：CONSTQP+LA=0 走 ce_pipeline，每批开头
`_reset_output_slot_idx(0)` 自对齐且每帧立即产出，与相位无关。

## 修复

**文件**: `external/realesrgan_video/nvenc_sdk.py` `reopen()`

```python
self._output_slot_idx = 0
self._frame_idx = 0   # [FIX-REOPEN-SLOT-PHASE] 必须同步归零
```

reopen() 后驱动会话全新，inputTimeStamp 从 0 重启与 gen=1 完全同构——
旧约束的前提（session 持续存在）已随 [FIX-CROSS-SEGMENT-SESSION] 消失。

同步更新三处注释：reopen() docstring、_NVENCEncodeThread.__init__ 的
FIX-FRAMEIDX-RESET-REGRESSION 块、模块 docstring 跨段复用段。

已核实 `_frame_idx` 全部读取方（slot 分配、inputTimeStamp、_pending_cnt、
chunk 簿记）归零后与 gen=1 同构；`frame_count` 属性无活跃读取方。
py_compile 通过；**待 GPU 多分段实测**（段 2+ 无告警、各段帧数守恒）。

## 教训

1. **成对耦合计数器必须同点重置**——slot 分配模数与 drain 模数是同一相位。
2. **旧禁忌的前提消失后要重新评估**——"禁止重置 _frame_idx"在 reopen
   架构下从保护变成病因。
3. **屏蔽 workaround 是有效探针**——FIX-HIGHRES-RC 的开关实验暴露了被
   LA=0 路径掩盖的 LA>0 独占 bug（此 bug 与分辨率无关，任何 LA>0 配置的
   段 2+ 都会中招）。

## 关联

- [[esrgan-segment-reuse-frame-idx-reset]] — 前时代的反向 bug（重置致病）；
  时代分界线 = [FIX-CROSS-SEGMENT-SESSION] reopen() 的引入
- [[esrgan-nvenc-slot-backpressure]] — _ensure_slot_free 机制本身
- [[esrgan-cross-segment-sigsegv-fix-stack]] — reopen() 的引入背景
