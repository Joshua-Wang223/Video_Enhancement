# IFRNet LA 排空顺序防御（test8 复现，2026-08-13）

## 结论
v6.4.5.1 / v6.4.4.1 / v6.4.3.1 三文件定向修复 NVENC LA 编码三处缺陷 + 一个独立崩溃：
- FIX-EOS-LEFTOVER-G-NAMEERROR（缺陷 A）
- FIX-SLOT-DRAIN-TARGET（缺陷 B）
- FIX-DRAIN-ORDER-DEFENSE（缺陷 C(a)）
- FIX-EMPTY-PREV-FILL（缺陷 C(b)）
- FIX-ENC-EXC-CONTEXT（缺陷 4）

核心假设失效：**"驱动输出顺序 == 提交顺序" 在 LA=8 + 9 slots 下不成立**
（test8: 1440×960@30fps, VBR_HQ+LA=8, 2x）。

## 背景数据（test8 旧代码）
- 段0/1: 输出包数≈预期（10196/9752 包）但仅 ~26% 帧可解码（2641/2590）；
  frame_num 回退 1130 次、pts drop、illegal short term buffer state。
- 段2: 自 ~61 帧起 _ensure_slot_free 告警刷屏，EOS 后编码线程 NameError: '_g' is not defined
  → 流水线中止，缺失 323.69s。

## 根因
### A. EOS 残留占位 _g NameError（确定性崩溃）
`_leftover_fis = sorted(_g - self._strm_ts_base for _dq in ... for _ent in _dq)` 生成器体引用
未绑定变量 `_g`（应为 4 元组 `_ent` 的 `[0]`）。sorted() 消费生成器必抛 NameError → 段2 崩溃。

### B. _ensure_slot_free 单槽探测局限
`_ensure_slot_free(slot_idx, pairs)` 调用 `_drain_outputs_blocking(max_slots=1)`，后者只探测
`_output_slot_idx % slot_count` 指向的单一物理槽，而非目标 slot_idx。LA 重路由/辅助块使目标槽
输出晚于其他槽就绪 → 探测空 → guard 超限放弃 → 目标槽带 pending 复用 → 覆盖丢帧 + EOS 残留。

### C. drain 顺序 == 提交顺序不变式失效
`_apply_drained_entries` 假设驱动按提交顺序输出、`_drain_slot = est_fi % slot_count` 命中且队首
gfi 单调；test8 规模下 LA 重路由使该假设失效 → 帧字节与标签错配 → frame_num 非单调 → DPB 损坏。

## 修复
- A: 生成器表达式 `_g` → `_ent[0]`（三文件各一行）。
- B(a): _ensure_slot_free 改为对目标槽自身 `self._slots[slot_idx]['bs_buf']` 做 per-slot blocking
  LockBitstream（doNotWait=0，while-True 直到 NEED_MORE_INPUT）；成功 → `[(slot_idx, h264)]` 经
  `_apply_drained_entries` 消费（est_fi=slot_idx 命中目标槽队首）+ `_output_slot_idx += 1`。
- B(b): NEED_MORE_INPUT 持续且 guard 超限 → 仅告警不 abort，但把目标槽 deque 未消费条目逐个
  popleft → 空帧占位（prev 填充）→ `_output_slot_idx += 1` → 删槽 deque；宁可写占位也绝不覆盖
  未取回码流（B(c)，消除"放弃等待→槽位覆盖"路径）。
- C(a): `_apply_drained_entries` 一致性防御：deque 空 → `_diag_slot_mismatch` 计数 + 诊断；
  队首 gfi ≤ `_last_drained_gfi` → `_diag_gfi_regress` 计数 + 诊断；仍 popleft 消费推进（不丢数据）。
- C(b): 空帧占位 `b""` → `self._prev_stream_h264 or b""`；非空帧写入同步更新 `_prev_stream_h264`；
  `__init__` 初始化、`_stream_begin` 每段重置。边界：prev 填充只用于"真缺失"帧（EOS 残留 +
  B(b) 超限占位），绝不用于 LA 正常缓冲帧（`_drain_outputs_blocking` 只产出非空 h264，中间
  chunk 不会产生 b""，已隐式满足；编码器级 prev 比 writer 侧 _prev_h264 "最后写出帧"更准确）。
- 4: 5.1/4.1 编码线程 `_loop` 三处 `self.error = e` → `_wrap_exc(e, where)` RuntimeError 包装
  (frame_idx, pending_slots, pending_count)；flush_and_join 原样 re-raise 透传上下文。
  3.1 无独立编码线程，在 encode_frames_stream 两个调用点（chunk / EOS）附加同样诊断上下文。

## 新增状态（三文件命名一致，_stream_begin 每段重置）
- `_last_drained_gfi`（C(a) gfi 单调基准）
- `_prev_stream_h264`（C(b) prev 填充缓存）
- `_diag_slot_mismatch` / `_diag_gfi_regress`（C(a) 诊断计数，沿用 _diag_empty 命名风格）

## 变量名差异
- v6.4.5.1: `_strm_slot_pending`；v6.4.4.1 / v6.4.3.1: `_batch_slot_pending`。
  修复逻辑一致（归一化变量名后函数体 diff 为空）。

## 验证
1. `python -m py_compile` 三文件全部通过。
2. 跨文件一致性：`_apply_drained_entries` / `_ensure_slot_free` / EOS leftover 归一化后完全一致。
3. test8 全流程重跑（VBR_HQ+LA=8, 2x）需在生产 Linux 机执行（本机缺模型权重/TRT 缓存）：
   `tests/verify_segment_bitstream_v3.py` 断言 frames==packets、frame_num 无回退、pts 无 drop。

## 范围边界
- 仅改三后端文件；processor `ifrnet_processor_v6_4_single.py` 的 break 语义（段失败即终止+保存进度）不改。
- 不重构 FIFO 记账架构；不复活 ts-trust / outputTimeStamp 重关联 / 重排缓冲。
- 审核观察（本次未改动）：辅助块消费分支未实现（C(a) 仅检测不补逻辑）；`_drain_outputs_blocking`
  在 bitstream_ptr_val 为假时推进 `_output_slot_idx` 的潜在漂移未处理。