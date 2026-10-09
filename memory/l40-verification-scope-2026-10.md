---
name: L40 验收范围与 CPU/T4 已完成项（2026-10-09 核查）
description: L40 侧全流程验证只占 4 项（AV1 冒烟 S1~S8 / crf_cq G7-6 / AV1 矩阵 AC1 / AV1 落表 LOO），其余 9 项已在 CPU/T4 侧完成并逐项标注证据文件与强度；含 T4 侧 86/0/1/2、crf_cq --gpu 114/0/2 等实测数字，以及「CPU 7/7 只有 commit message 无报告留痕」「L40 标定已降级为仓内不可审计」两条证据强度警告
type: project
---

# L40 验收范围（2026-10-09 核查收敛）

CPU 侧与 T4 侧已收口，**唯一能力缺口是 AV1 NVENC 硬编全链路**。L40 只需 4 项：

| # | 测试项 | 脚本 | 关键判据 |
|---|--------|------|----------|
| L40-1 | AV1 长视频端到端冒烟 S1~S8 | `Accessory/verify/av1_pipeline_smoke.py --codec av1_nvenc` | 帧守恒/解码级/编码器确认/S8 斜率 ≤50 MB/min |
| L40-2 | crf_cq GPU 判据的 G7-6 | `Accessory/verify/crf_cq_unification_verify.py --gpu` | G7-6 av1_nvenc `-cq` PASS；G7/G8 FAIL=0 |
| L40-3 | AV1 质量矩阵 B 组 AC1 | `Accessory/probe/av1_vp9_quality_matrix.py --only av1_nvenc` | 表值 QP 落带内（AC2 即 L40-2，AC4 跨仓 B 组） |
| L40-4 | AV1 标定落表 LOO | `Accessory/probe/eqq_pool_fit_table.py` | `--axis cq --sides gpu_l40_cq` / `--axis qp --sides gpu_l40_qp`，两轴 LOO 均 ≤5.9 |

**AV1 NVENC 可用性必须实跑一帧判定**（不能用 `-h encoder=av1_nvenc`，Turing 上也会打印完整选项表而误报可用）。

方案全文见 `Plan/PROMPT_L40_全流程验证.md`。

---

# 已在 T4 侧完成（Tesla T4, SM75, 14.6 GiB）

| 项 | 结论 | 证据 |
|----|------|------|
| plan_gate 完整 | 86 PASS / 0 FAIL / 1 WARN / 2 SKIP（共 89 项） | `verification_report/verification_report_20261009_042708.json`。WARN=`BEH-ERR`(`_session_gen`)，该缺陷由提交 `d215fc2`（04:39）修复，**报告早于修复 12 分钟**；SKIP=R8(RVML 提示)、RT-0(未给 `-o`) |
| crf_cq --gpu | 114 PASS / 0 FAIL / 2 SKIP | `verification_report/CRF_CQ统一验证报告_20261009_035735.md`。SKIP: `G7-6`「No capable devices found」+ `G8-4*` |
| segment_bitstream_verify_v5 | hevc+LA=8 / 720p：frames=packets=603 | 提交 `6ed9ceb` |
| nvenc_vbr_hq_verify（V8~V15） | 裁定方案 A 并落地：驱动接受 `rc_ptr[1]=32`，三档字节互异（1076383/2159969/1018148）；ΔVMAF −0.048/−0.130 | `t4-vbrhq-verification-plan.md` |
| nvenc_rc_diagnose | V11：VBR_HQ 未在 SDK 头文件、FFmpeg9 已移除、`-rc` 仅 constqp/vbr/cbr | `Plan/T4_NVENC_vbr_hq移除_验证专项.md:180` |
| h264/hevc 冒烟（S1~S8 代理） | hevc 两臂 14/2；h264 constqp 7/2、vbr_hq 0/1（B1 缺陷首现，已修） | `verification_report/s8_t4_*.md` + `s8_20261004_raw/*.mem.tsv` |
| NVENC 标定落表 | T4-1~T4-9 全绿：CQ h264 LOO 3.98/hevc 5.81；QP h264 3.47/hevc 3.72 | `equal-quality-t4-nvenc-calibration.md`；点数据 `points/gpu_t4_{cq,qp}/` |

**T4 跑不了 AV1**（已实测确认，4 处独立证据）：`av1_vp9_matrix_T4_20260929.md:16`、`CRF_CQ统一验证报告_20261009_035735.md:158`、`Plan/PROMPT_T4_NVENC等质量标定专项执行方案.md:34`、`Plan/PROMPT_T4_全流程验证.md:34`。

同日 4 次门禁时间线（全部 T4）：03:24 与 03:35 各 106 项含 1 FAIL(BEH-G9)；03:54 101 项含同 1 FAIL；**04:27 = 89 项 86/0/1/2**（最新）。

---

# 已在 CPU 侧完成（7 项）—— ⚠ 证据薄弱

| 项 | 结论 | 证据 |
|----|------|------|
| verify_equal_quality | 537s，ΔVMAF ≤1.0，5/5 rc=0 | ⚠ 仅 commit `9d15dc6` message + `MEMORY.md:182` 一行 |
| crf_cq_cpu | 5s，G1~G6/G10 | ⚠ 同上 |
| av1_vp9_matrix_cpu | 582s，libvpx-vp9/libsvtav1/libaom-av1 PASS | ⚠ 同上 |
| segment_bitstream_verify | 2s，帧守恒/解码级 | ⚠ 同上 |
| eqq_pool_fit_selftest | 1.7s | ⚠ 同上 + 提交 `6ed9ceb` |
| calibrate_eq_selftest | 0.3s，39 项 | ⚠ 同上 |
| nvenc_tuning_verify | 0.1s | ⚠ 同上 |
| plan_gate（CPU 轮） | **预期跳过**（BEH-G9 需 GPU） | — |

⚠️ 该轮**无任何报告文件/日志留痕**（`comprehensive_verify.py` 不写报告，`logs/` 无对应条目），环境记为 WSL Ubuntu。索引指向的 `memory/cpu-verification-pass.md` **两侧镜像均不存在且 git 全历史从未有过**（悬空链接，违反 [索引双向完整](memory-index-integrity.md)）。准确口径是「7 项 PASS + plan_gate 预期跳过」，不是「7/7」。

---

# 跨仓 ⑨ 组一致性

- 最近一次**实跑**记录在案：2026-10-03，`memory/eqq-batch-measure-parallel-constraints.md:467`。
- 2026-10-09 的「⑨ 组 14/14 一致」（`implementation-verification-report-2026-10-09.md`）是**静态代码核对**而非实跑。
- **2026-10-09 实测：13 PASS + 1 FAIL** —— ⑦ `[7] --threads 显式值两边不一致`（VE 得 `'3'` / VU 得 `'2'`），该组退出码非 0。属非 AV1 差异，需单独立项修 VU 侧 `vidcrop_hwaccel.py` 的 `--threads` 钳位。
- 两仓 `verification_report/` 与 `*.log` 内**均无 ⑨ 组运行原始输出**。
- 脚本路径已迁：`/workspace/VidUtils/Accessory/verify/verify_quality_mapping.py`（VU 提交 `ebdad41`，2026-10-06），旧路径 `/workspace/VidUtils/verify/` 已空。

---

# 两条证据强度警告

1. **L40 侧标定已降级为「仓内不可审计」**：`av1-nvenc-l40-calibration.md` 末节列出审计链 6 处断点（报告声称的 workdir `temp/eqq_gpu_l40/` 不存在、`logs/` 无 GPU 标定日志、`MANIFEST.md` 未收录 `gpu_l40_*`、无 `/tmp` 痕迹等）。历史 L40 报告数字只能作参考，**不可反向引用为「已验收」**。
2. **S8 峰值基线是 T4 标定值**：`--mem-peak-mb` 默认 16000 MB 按 T4 / 720×576 / bs=24 实测标定（`t4-s8-findings-and-blockers.md`）。L40 + `interpolate_then_upscale` 上分辨率下须按实际 batch-size 重标定，否则假 FAIL。

相关：[验证脚本三处路径/口径陷阱](verification-script-path-and-axis-traps.md)、[L40 AV1 标定](av1-nvenc-l40-calibration.md)、[T4 NVENC 标定](equal-quality-t4-nvenc-calibration.md)、[自述不是执行证据](feedback_artifacts_not_execution_evidence.md)