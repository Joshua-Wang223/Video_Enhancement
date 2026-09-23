# IFRNet LA 辅助块清空 slot pending → 周期性错位花屏/丢帧（tt6 根因与修复）

> 文件: `external/IFRNet/process_video_v6_4_5_1_single.py`（FIX-AUX-NO-CLEAR / FIX-REORDER-NEXT / FIX-EOS-EXPLICIT-SLOT）
> 生产故障: `112 Max Bed Time.avi` 经 upscale→interpolate 管线，插帧段输出
> `frames(9802)≠packets(10197)`、frame_num 回退 77/74 次、pts 异常 151 条、段1 守恒失败 9825 vs 9824。

## 一、现象（test6 离线取证）

| 段 | AUs | 解码帧 | 回退 | 周期 | 守恒 |
|----|-----|--------|------|------|------|
| seg0 | 10197 | 9802 | **77** | 位置 mod 132 ∈ {124,125}，间隔 132 | 5098→5098 OK |
| seg1 | 9824 | 9644 | **74** | 位置 mod 132 ∈ {124,125}，间隔 132 | 9825→9824 **差 1** |
| seg2 | 9573 | 9573 | **0** | 无 | 9573→9573 OK |
| 终片 | 29594 | 29019 | **151** = 77+74 | — | — |

- 码流 AU 级解剖（analyze_au_order.py，按 parse_h264_es 聚类多 slice 帧）：
  每次损坏窗口 = 8 帧（= LA 深度）：**1 帧早写**（fn 跳 +52/+55/+58，如 AU 124 的
  fn=58 即「本应在 51 帧之后位置的数据」）+ **1-7 帧晚写**（fn 7→13 = 真 fi 124→130
  的数据贴到标签 125→131）。之后回同步。
- ffmpeg verbose pts 异常 est 分布 {1:92, 4:25, 7:23, 51:2, 54:1, 57:3} =
  解码器因引用链断裂（引用的帧被 DPB 逐出/从未解码）跳过不可解码 AU。
- 生产日志：**零** ts 重复回显/ts 无效 fallback 告警 → 标签链（_resolve_drained_fi）
  从未 fallback，错位不是「标签错」，而是**数据与标签不匹配**。
- 生产日志：224 次「重排缓冲过大」强制排空，三段均每 ~66 输入帧（≈128 输出帧/chunk）
  一次，段末 seg0 残余 41 帧、seg1 残余 64 帧、seg2 无残余。

## 二、根因链（三层缺陷叠加）

### 缺陷 1（触发源）: 辅助块清空 `_strm_slot_pending` — `_consume_one_drained`

NVENC 在 LA 下把 SPS/PPS/AUD 作为**独立输出块**（无 VCL）drain 出来，其
outputTimeStamp@40 回显的是该物理 slot「最后一次提交帧」的 ts（tt6 诊断脚本
diagnose_lockbitstream_timestamp.py 实测：LA=16 时辅助块 ts 按 1017..1033 轮转回显
= 各 slot 最后提交帧的 ts）。

旧代码对辅助块执行 `self._strm_slot_pending[_drain_slot] = None` —— 但该槽对应
**真实帧的 VCL 数据仍在驱动输出队列中**。清空后：

1. `encode_frames_stream` 的 slot 复用反压检查 `while self._strm_slot_pending[slot_idx]
   is not None`（[FIX-SLOT-BACKPRESSURE]）**误判槽位空闲** → 未排空即复用输入 buffer；
2. LA 模式下驱动**延迟消费**输入像素（EncodePicture 返回 NEED_MORE_INPUT 时只入队），
   复用时 cuMemcpy2D 覆盖的像素在驱动真正编码时被读到 → **输出的数据块携带旧 ts
   标签、但内容是后续帧的像素、fn 按驱动实际编码位置计算** → 数据与标签错位；
3. 错位对 (fi, data) 进入 writer 重排缓冲，按 fi 排序写出 → P 帧引用链断裂 →
   不可解码 AU → 解码器跳过 → frames < packets + pts 异常 + fn 回退。

seg2 同样有 ~73 次强制排空却零损坏：**排序写出本身无错**（标签正确时 sorted 写出
= 正确序），错位只发生在「辅助块清 pending → 反压旁路 → 输入覆盖」被时序触发的
seg0/seg1。

### 缺陷 2（放大器）: 强制排空后 `_reorder_next` 永不前进 — `_drain_write`

```python
_reorder_next = max(_reorder_pending.keys()) + 1 if _reorder_pending else _reorder_next
```

在 pop 全部之后求 max（恒为空）→ 指针停留在旧值（已写过的 fi）→ 后续到达的帧
永远「不 ready」→ 每个 chunk（128 帧）缓冲重新累积到阈值 → **每 ~132 帧一次周期
强制排空**（生产 224 次、回退位置 mod 132 完美对账）。修正：pop 前先
`_max_written = max(_reorder_pending)`，排空后 `_reorder_next = _max_written + 1`。

### 缺陷 3（次要）: EOS 排空同槽多帧的 slot 归属 — encode_frames_stream 末尾

`while True` 可对同一物理槽连续 LockBitstream 取回多帧，旧实现传
`self._output_slot_idx` 当 est_fi，`_consume_one_drained` 内 `est_fi % slot_count`
不再是真实物理槽 → pending 清除/回退命中错误槽位。修正：`_consume_one_drained`
新增 `explicit_drain_slot=None` 参数，EOS 循环显式传 `_ds`。

## 三、修复（process_video_v6_4_5_1_single.py）

1. `_consume_one_drained` 辅助块分支：**保留 pending 记录**，待该槽真实帧数据
   drain 时再清除（FIX-AUX-NO-CLEAR）→ 反压检查恢复正确，槽位必在输入覆盖前
   排空，标签-数据错位源头消除。
2. `_drain_write` 强制排空：先记 `_max_written` 再 pop（FIX-REORDER-NEXT）→
   指针越过已写出最大 fi，缓冲不再周期性膨胀，强制排空退化为罕见一次性事件。
3. EOS 排空循环：`explicit_drain_slot=_ds` 显式传物理槽（FIX-EOS-EXPLICIT-SLOT）。

不涉及 _resolve_drained_fi（ts 重关联 V2 本身验证正确：零 fallback、双射回显
18/18、34/34）；不涉及 FFmpegMuxer。

## 四、验证

- `python -m py_compile external/IFRNet/process_video_v6_4_5_1_single.py` 通过；
- 段级解剖交叉确认：seg0/seg1 回退位置 mod 132=124/125（71/70 次）、间隔 132
  （68/63 次）与生产强制排空节奏精确对账；seg2 零回退零间隔 → 根因链唯一自洽；
- 结论可证伪点：若缺陷 2 单独存在（标签正确）→ 周期排空但不损坏（seg2 即此态）；
  若缺陷 1 单独存在 → 错位一次而非 77 次。生产同时呈现两者。

## 五、适用范围

- 仅影响 v6.4.5.1（含 FIX-STREAM-REORDER-BUFFER / FIX-STREAM-TS-REASSOC-V2 的
  LA>0 分块流式路径）。v6.4.5 及更早为旧架构（无 _strm_slot_pending /
  encode_frames_stream / 重排缓冲），不适用。
- ESRGAN 侧 realesrgan_video 的 `_ensure_slot_free` 为独立实现（esrgan-nvenc-slot-
  backpressure），本修复不涉及。
