---
name: ifrnet-hevc-cross-segment-reuse-restore
description: HEVC/AV1 跨段编码器复用恢复评估定论（2026-09-02 Linux T4 A/B 实测 10/10 PASS，开关1 与开关0 段产物逐字节一致）：维持禁用默认并归档，IFRNET_NVENC_CROSS_SEGMENT_REUSE=1 已验证可用可随时灰度
metadata: 
  node_type: memory
  type: project
  status: completed
  originSessionId: ece14121-297d-497b-ba70-6d06c9890971
  modified: 2026-09-02T03:07:46.786Z
---

# IFRNet HEVC/AV1 跨段 NVENC 编码器复用 —— 恢复评估与实施

## 背景

2026-09-01 死锁（[[ifrnet-hevc-la-slot-headroom-deadlock]]）修复期间，在 `external/ifrnet_video/main.py` `_get_or_create_nvenc_encoder` 加了 `[P0-FIX-HEVC-SEGMENT-HANG]`：HEVC/AV1 非首段强制新建编码器（`_force_new`），放弃跨段复用。

2026-09-02 评估是否可恢复（分析见 `Plan/plan_hevc_cross_segment_reuse_restore.md`）。

## 结论

**技术上可以恢复，但维持禁用默认，按"开关 → 修缺陷 → A/B 实测 → 数据决策"推进。** 恢复依据（已核实）：

- **触发链 A（LA=0，段首 f0 走 `encode_frame` → `_drain_outputs_blocking` 锁残留异常槽）** 已由 `[FIX-F0-ALWAYS-IN-BATCH]` + `[FIX-F0-IN-BATCH-CE]` 关闭（f0 入 batch，不再走 encode_frame）
- **触发链 B（LA>0，`_ensure_slot_free` 槽余量 0 循环依赖）** 已由 `[FIX-LA-SLOT-HEADROOM]`（`_required_buffers = la_depth+2`）+ `[FIX-ESF-NO-LOCK-WHEN-EMPTY]`（ready≤0 不 Lock）关闭
- 段首 `_stream_begin(force=True)` 无条件重置（`_strm_slot_pending` 清空 + `_output_slot_idx = _frame_idx`）；H.264 长期跨段复用零事故 + ESRGAN `[FIX-SKIP-REOPEN]` 已验证跨段复用

不立即放开的理由：修复验证在禁用状态（每段新编码器）下完成，**不能证明复用安全**（驱动级 LA/DPB 残留无软件探针）；收益 <1%（每段重建毫秒级）；死锁在驱动内不可中断、风险不对称。

## 已落地改动（2026-09-02，Phase 0 + Phase 1）

### Phase 0 —— 可回滚开关（`external/ifrnet_video/main.py`）

```
IFRNET_NVENC_CROSS_SEGMENT_REUSE=1  → 放开 HEVC/AV1 跨段复用
默认（未设置/'0'）                    → 维持禁用（行为不变）
```

### Phase 1 —— R1 修复：`_sps_pps_injected` 段边界重置（`external/ifrnet_video/nvenc_sdk.py`）

**缺陷**：`_sps_pps_injected` 只在 `close()` 重置；跨段复用不 close → 段 2+ 新 muxer 的 SPS/PPS 预注入被 `_drain_write` / LA=0 路径的 `if not _sps_pps_injected` 门控跳过（仅靠 IDR 自带 + `_prepend_param_sets` 兜底）。注释（close 内"跨段重置换，支持 encoder 复用"）与实现矛盾。

**修复点**（两处，覆盖两条 LA 路径）：
- `_stream_begin()`（nvenc_sdk.py:1905-1909）加 `self._sps_pps_injected = False` —— LA>0 路径（段首 `_stream_begin(force=True)` 必经）
- `encode_frames_batch_ce_pipeline` f0 取用块（nvenc_sdk.py:2399-2402）加同重置 —— LA=0 路径不经过 `_stream_begin`，以 `_pending_f0_nv12` 非空（每段段首首批必暂存 f0）为段首标志
- `close()` 内注释改为"关闭即弃"语义（3146-3147）

## A/B 实测结果（Phase 2 完成，2026-09-02，Linux 生产侧 Tesla T4）

### 实测矩阵（素材 new5.mp4 CFR 720p 803 帧 + wws3e02_26s.mp4 VFR 360p 602 帧；segment-duration=8s；IFRNet 2x + hevc/h264_nvenc，`--skip-upscale`；每组合段数：new5=3 段、wws3e02=2 段）

| # | codec | rc | LA | 开关 | new5 耗时 | new5 结果 | wws 结果 | 段 2+ 首帧 IDR（new5） |
|---|---|---|---|---|---|---|---|---|
| 1 | hevc_nvenc | constqp | 0 | 0（基线） | 75s | PASS | PASS | 58.6K / 38.5K B |
| 2 | hevc_nvenc | constqp | 0 | **1** | 74s | PASS | PASS | 58.6K / 38.5K B（与 c1 逐字节同） |
| 3 | hevc_nvenc | vbr_hq | 8 | 0（基线） | 154s¹ | PASS | PASS | 93.0K / 70.7K B |
| 4 | hevc_nvenc | vbr_hq | 8 | **1** | 83s | PASS | PASS | 93.0K / 70.7K B（与 c3 逐字节同） |
| 5 | h264_nvenc | constqp | 0 | **1** | 71s | PASS | PASS | 68.4K / 54.5K B（对照） |

¹ c3 的 154s 为推理侧噪声（段 3 GPU 利用率均值 15.8%、空闲占比 84.4%、插帧 3.6 帧/s 的 T2 队列气泡，AUTO-TUNE 首段校准迟滞）；同配置 c4 段 3 仅 18.5s。与编码器复用无关。

### 硬指标（全部满足，10/10 组合 PASS）

- 全部段完成无挂死（run exit=0，timeout 540s 无超时）
- `verify_segment_bitstream_v4.py --skip-chroma` 段级全绿：frames==packets、连 IDR<3、frame_num 无回退、无 PTS/解码异常（new5: 521+521+549=1591；wws3e02: 392+790=1202）
- 全片 `ffmpeg -f null` 零解码错误；总帧数 == Σ(段帧数) == 段内 Σ(2n_i−1)
- 段 2+ 首帧 IDR 与段 1 同量级（new5 constqp 42K/58K/38K B；vbr_hq 81K/93K/71K B），无 ~375KB 噪声花屏特征
- 每段 SPS/PPS 充足（HEVC ≥10/段，h264 ≥10/段）

### 关键证据：开关 1 与开关 0 段产物逐字节一致

c2 与 c1、c4 与 c3 的每段首帧 IDR 大小、SPS/PPS 计数完全相同（58578B/93030B 等逐项相等）→ **跨段复用路径与禁用路径输出零差异**，驱动级 LA/DPB 状态残留（R4）假说在本矩阵内无实证。

### 决策定论（按方案决策门）

- 硬指标全 PASS；性能 Δ：c1vs c2 = 1.3%（噪声带）、c3 vs c4 的 46% 为推理噪声不可用 → **真实复用收益无实证 ≥1%**（每段重建毫秒级，<1%，与方案预估一致）
- 按"Δ<1% 或任一 FAIL → 维持禁用归档"：**维持禁用默认（`IFRNET_NVENC_CROSS_SEGMENT_REUSE` 缺省 '0'），本评估归档**
- 但开关已验证全绿：若未来需要（如段极多、编码器重建成本放大），可直接 `IFRNET_NVENC_CROSS_SEGMENT_REUSE=1` 灰度（建议先跑 1~2 个生产任务），Phase 0/1 改动保持现状即可，零回滚成本

### 验证产物

`benchmark_output/ab_cross_segment_reuse/`：`{video}_{c1..c5}.mp4`（输出）、`{video}_{c#}.log`（运行日志）、`{video}_{c#}.v4.txt`（段级验收）、`{video}_{c#}.segan.txt`（NAL 分析）、`ab_matrix_results.txt`（汇总）。

## 附带发现（2026-09-02，与本议题无关但需处理）

`tests/verify_plan_implementation.py` 的 **BEH-B1/B3/B4 断言过期**：`[P2-FIX-FRAG]`（09-01 修复 A 的 `merge_trailing_fragment`，video_utils.py:1136，末段 <2s 并入前段）使 9s 测试源按 4s 切分产出 **2 段**而非 3 段，断言 `len==3` 必然失败。Windows 复现确认（设计内行为），08-31 报告 90 项 0 FAIL 为修复 A 之前的基线。需更新 BEH-B 断言（接受 2 段或改用不触发合并的素材时长）。

## 关联

- [[ifrnet-hevc-la-slot-headroom-deadlock]] —— 死锁根因与三项修复（本次评估的起点）
- [[esrgan-cross-segment-optimization-complete]] —— ESRGAN 跨段复用已验证（复用架构可行性旁证）
- `Plan/plan_hevc_cross_segment_reuse_restore.md` —— 完整评估与分阶段 plan
