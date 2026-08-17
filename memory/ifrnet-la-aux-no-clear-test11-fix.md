# IFRNet LA 辅助块账记修复（FIX-AUX-NO-CLEAR 回归，test11 复现 2026-08-14）

## 现象（temp/test11）

- 仅 `interpolated_segment_000.mp4` 损坏：packets=4960、解码 frames=1411（期望 4961，差 -3550）；seg1/seg2 完全正常。
- ffmpeg showinfo：106 条 pts drop、est_missing=3452，每 8 个可解码帧一跳变（diff≈9500）。
- NAL 解剖（nal_tt11_seg0.csv）：IDR=108、frame_num 回退 **549** 次；前 9 帧 `[0..8]` 正确，第 9 帧起每个 9 帧块旋转为 `[10..17,9]`、`[19..26,18]`…（slot 相位永久偏移 1）。
- 日志零 `_diag_slot_mismatch` / `_diag_gfi_regress` 告警 → 标签链看似一致，实际数据与标签错位。

## 根因（与 tt6 根因链一致，8/13-8/14 FIFO 重构后回归）

NVENC LA 冷会话预热期把 SPS/PPS/AUD 作为独立**无 VCL 辅助块** drain（outputTimeStamp@40 回显该物理 slot 上次提交帧 ts）。当前 `_apply_drained_entries` 未实现 VCL 分类（`_nal_first_vcl_type` 已删除，文档声称已处理但代码没有）：

1. 辅助块被当普通帧 pop 该槽 pending 队首 → 该槽真实帧 VCL 数据仍滞留驱动队列，pending 却被清空；
2. `_ensure_slot_free` 背压检查误判槽位空闲 → 输入 buffer 未排空即复用；
3. LA 延迟消费读到被覆盖的像素 → 数据与标签错位 → 9 槽轮转传播 → 每 9 帧迟一窗 → frame_num 周期性回退 → 解码器静默丢弃。

seg1/seg2 干净：会话已热，无辅助块，顺序假设碰巧成立。

## 修复（v6.4.5.1 / v6.4.4.1 / v6.4.3.1 三文件同步）

- **FIX-AUX-NO-CLEAR**：恢复 `_nal_first_vcl_type()`；无 VCL 辅助块仅缓存 SPS/PPS，不 pop pending、不进 pairs、不推进帧计数（物理轮转指针照常推进）。主 drain / EOS drain / `_ensure_slot_free` 三处统一生效。
- **FIX-SLOT-DRAIN-TARGET**：`_ensure_slot_free` 轮转探测无法触达目标槽时直探目标槽 blocking LockBitstream；guard 超限以空帧占位（prev 填充）消费 pending，**绝不带 pending 复用**。
- **FIX-DRAIN-COUNTER-DRIFT**：`_drain_outputs_blocking` 在 bitstream 数据指针为空时不再推进 `_output_slot_idx`。
- **FIX-DRAIN-ORDER-DEFENSE**：drain 返回三元组 `(est_fi, out_ts, h264_data)`；VCL 块 out_ts（tt6/tt7 双射已验证）与 FIFO 队首 gfi 比对，不一致时 `_diag_phase_shift` 告警；新增 `_diag_aux_block` / `_diag_slot_drain_fallback`，与既有 `_diag_slot_mismatch` / `_diag_gfi_regress` 一起在 `_stream_begin` 每段重置。
- 消费点适配：ce_pipeline inline/redrain 与 encode_frame 解包三元组。

变量名：5.1 用 `_strm_slot_pending`，4.1/3.1 用 `_batch_slot_pending`；归一化后函数体 diff 为空（已机器校验）。

## 验证状态

- 三文件 `python -m py_compile` 通过。
- 归一化一致性校验：`_nal_first_vcl_type` / `_apply_drained_entries` / `_ensure_slot_free` 三文件完全一致；`_drain_outputs_blocking` 去注释后完全一致。
- 待生产 GPU 复验（Linux，VBR_HQ+LA=8, 2x，test11 同参数）：`verify_segment_bitstream_v3.py` 应 frames==packets、frame_num 回退=0、pts drop=0；日志允许出现少量 `辅助块` 信息行（预热期正常），不应出现 `相位漂移` / `slot 记账错配` / `gfi 回退` 告警。

## 镜像同步

按 AGENTS.md 约定，本文件需镜像到 `C:\Users\Administrator\.claude\projects\D--Workspace-Python-Video-Enhancement-Video-Enhancement\memory\`（需手动复制或在下一次可写会话同步）。

## 2026-08-14 追加：ESRGAN 移植

已将同款修复移植到 `external/realesrgan_video/nvenc_sdk.py`（FIX-AUX-NO-CLEAR
完整语义：VCL 分类、辅助块不占帧槽、目标槽直探、计数器防漂移、三组诊断计数；
`_prev_stream_h264` 占位兜底）。生产路径全部覆盖；ce_pipeline Phase 1 harvest
为非生产路径（LA>0 已路由 encode_frames_batch），未覆盖，详见
[[esrgan-nvenc-slot-backpressure]] 追加章节。
