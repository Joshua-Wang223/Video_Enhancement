# REDRAIN 缺口独立立项：ESRGAN 二次排空安全网（高风险未闭环）

> 立项日期：2026-09-18  
> 关联审计：`AUDIT_REPORT_2026-09-18.md`（补遗修复记录）  
> 关联计划：`Plan/Conversation-持续跟进6方案计划（GPU部分）-2026-09-17.txt` §5  
> 风险等级：**高**（可能静默丢帧，段级帧数审计仅为事后发现，无法事前防御）

---

## 1. 缺口描述

ESRGAN 侧的 `encode_frames_batch_ce_pipeline`（`external/realesrgan_video/nvenc_sdk.py`）与 IFRNet 参考实现（`external/ifrnet_video/nvenc_sdk.py`）在 **二次排空安全网（[FIX-LA-REDRAIN]）** 上存在实质差异：

| 对比项 | IFRNet 参考实现（v6.4.5.1） | ESRGAN 当前实现 | 差异说明 |
|---|---|---|---|
| 二次排空入口 | `_ce_final_drain()`（`line 3059-3145`） | 部分实现于 `line 2688-2712`（`[FIX-LA-REDRAIN]` 打印与回收），但 **无独立的 `_ce_final_drain` 方法** | ESRGAN 没有与 IFRNet 完全等价的独立二次排空阶段 |
| 代码自认差异 | — | `line 1625` 明确标注：**“ESRGAN 侧没有 `[FIX-LA-REDRAIN]` 二次排空安全网”** | 代码内已确认缺失，非遗漏描述 |
| `_output_slot_idx` 推进语义 | Phase 3（Drain）在成功取回时推进（`line 3102`）；二次排空由 `_ce_final_drain` 统一管理指针与槽位回收 | Phase 3（`line 2629-2676`）**不推进** `_output_slot_idx`；批末二次排空（`line 2688-2712`）存在，但与 IFRNet 的 `_ce_final_drain` 语义不同 | 若 BLKRETRY（阻塞重试）在批末失败，ESRGAN 无第二道安全网回收已提交但未取回的帧，可能导致静默丢帧 |
| 静默丢帧风险 | 由 `_ce_final_drain` + 段级帧数守恒审计双重保障 | 仅由段级帧数守恒审计（事后发现）保障，**无事前防御** | 高风险：帧数不守恒时只能在段收尾发现，无法在编码阶段阻止 |

---

## 2. 证据链（可追溯）

- **方案文件声称**：`Plan/Conversation-持续跟进6方案计划（GPU部分）-2026-09-17.txt` §5 声称“REDRAIN 已移植到 `encode_frames_batch_ce_pipeline`，Phase 1/3/末均推进指针”。
- **审计修正**：`AUDIT_REPORT_2026-09-18.md` 已将 §5 修正为“⚠️ 部分执行，缺二次排空安全网”。
- **代码自证**：`external/realesrgan_video/nvenc_sdk.py:1625`（“与 IFRNet 的一处实质差异：ESRGAN 侧没有 `[FIX-LA-REDRAIN]` 二次排空安全网”）；`line 2688-2712` 为部分二次排空实现（打印与回收），但无独立 `_ce_final_drain` 结构。
- **参考实现**：`external/ifrnet_video/nvenc_sdk.py:3059-3145`（`_ce_final_drain` 完整实现，含 `_output_slot_idx` 管理、槽位循环 LockBitstream、失败重试与指针推进）。
- **记忆文档**：`memory/realesrgan-missing-la-redrain.md`（如存在）；`memory/hevc-la-soft-retired.md`（软退役策略参考）。

---

## 3. 风险影响

- **直接影响**：在 `LA=0` 或 `LA>0` 且批末 `BLKRETRY` 阻塞重试失败时，已提交到编码器的帧可能无法被取回，导致输出帧数少于输入帧数（帧数不守恒）。
- **发现难度**：仅能通过段级帧数比对（`tests/verify_plan_implementation.py` 的行为验证阶段、`tests/verify_segment_bitstream_v5.py` 的解码级检查）在编码完成后发现，无法在编码阶段阻止或自动修复。
- **生产影响**：在长视频批量处理（`--batch-mode`）或分段处理（`segment_duration` 配置）场景中，若某一段发生静默丢帧，后续段的时间戳与帧号将出现不连续，影响下游合并与同步。

---

## 4. 后续修复目标（建议立项范围）

### 4.1 必须完成项（高优先级）
- [ ] **独立 `_ce_final_drain` 方法**：在 `external/realesrgan_video/nvenc_sdk.py` 中增加与 IFRNet `line 3059-3145` 等价的 `_ce_final_drain` 方法，包含：
  - 按 `_output_slot_idx` 顺序循环 `LockBitstream`
  - 成功取回时推进 `_output_slot_idx`
  - 失败时不推进（避免相位漂移）
  - 二次排空完成后清理 `pending` 与缓存 SPS/PPS
- [ ] **批末调用点**：在 `encode_frames_batch_ce_pipeline` 的批末（`line 2688` 附近）明确调用 `_ce_final_drain()`，并确保与现有 `REDRAIN` 打印（`line 2712`）不重复、不冲突。
- [ ] **指针一致性验证**：确保 Phase 1（Harvest）、Phase 3（Drain）、批末二次排空（`_ce_final_drain`）三处的 `_output_slot_idx` 推进语义与 IFRNet 完全一致，避免“双重推进”或“漏推进”。

### 4.2 建议完成项（中优先级）
- [ ] **回归测试锁定**：新增或扩展 `tests/test_nvenc_la_frame_conservation.py`（或在 `tests/test_regression_min.py` 中新增行为断言），验证 `ESRGAN` 在 `LA=0` 与 `LA=8` 下的帧数守恒，包含真实 GPU 路径（若环境支持）。
- [ ] **遥测与诊断**：在 `_ce_final_drain` 中增加与 IFRNet 等价的遥测计数（`_diag_lock_err_*`、`_diag_illegal_size`、`_sizecap_force_dropped`），确保静默丢帧可被诊断而非仅靠段级审计发现。
- [ ] **文档同步**：更新 `AGENTS.md`、`README.md` 及 `memory/` 索引（`MEMORY.md`），记录 REDRAIN 移植状态（已部分实现，但二次排空安全网未完全等价于 IFRNet 参考）。

### 4.3 可选优化项（低优先级）
- [ ] **代码清理**：将已归档遗留文件（`external/IFRNet/process_video_v6_3_4.py`、`_single.py`）的预存语法错误文件移至 `archive/` 目录，避免后续编译扫描噪声。
- [ ] **镜像同步自动化**：在 `memory/` 修改后增加 `diff -rq` 复核脚本（或 `rsync --dry-run` 预览），确保 A/B 两侧内容一致，而非仅文件数一致。

---

## 5. 验收标准

- [ ] `external/realesrgan_video/nvenc_sdk.py` 中存在独立的 `_ce_final_drain` 方法，且与 IFRNet 参考实现（`line 3059-3145`）逐行对照通过。
- [ ] 批末调用点明确且无重复推进 `/` 漏推进 `_output_slot_idx`。
- [ ] 回归测试（行为验证阶段 + 真实 GPU 路径，如环境支持）在修复后通过，无新增 FAIL。
- [ ] 审计报告（`AUDIT_REPORT_2026-09-18.md` 或后续版本）中 §5 状态由“⚠️ 部分执行”修正为“✅ 完整执行”。
- [ ] 相关 `memory/` 文档已同步更新，并反映在两侧镜像（A: `/workspace/Video_Enhancement/memory`；B: `/root/.codebuddy/projects/workspace-Video_Enhancement/memory`）。

---

## 6. 关联资源

- 代码：`external/realesrgan_video/nvenc_sdk.py`、`external/ifrnet_video/nvenc_sdk.py`
- 参考实现：`memory/nvenc-ce-pipeline-architecture.md`、`memory/realesrgan-missing-la-redrain.md`（如存在）
- 测试：`tests/test_regression_min.py`、`tests/verify_plan_implementation.py`、`tests/test_nvenc_la_frame_conservation.py`
- 审计：`AUDIT_REPORT_2026-09-18.md`、`Plan/Conversation-持续跟进6方案计划（GPU部分）-2026-09-17.txt`
