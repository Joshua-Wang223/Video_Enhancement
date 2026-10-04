---
name: T4 NVENC 等质量标定（CQ+QP 双轴）完成
description: 2026-10-04 T4 上 h264/hevc NVENC 两轴标定落表；含 screen_ui_code 同名冲突陷阱与「判据钉 size 口径」关键点
type: project
---

2026-10-04 在 Tesla T4 执行 `Plan/PROMPT_T4_NVENC等质量标定专项执行方案.md` 的 T4-1~T4-9，全部完成、门禁全绿。

**表值**（17 素材，锚点 18/21/24/27/30，`n_subsample=1`，720p prep，`--src-is-prep`）
- **CQ 轴** → `QUALITY_MAP`（已并入提交 `8f0a605`，与 VidUtils 逐字相等，⑨ 组 14/14）：
  `h264_nvenc (0.9295, 6.2523)` LOO 3.98；`hevc_nvenc (1.1116, 2.1606)` LOO 5.81
- **QP 轴** → `QUALITY_MAP_QP`（**仅 VE**，D2b）：`h264_nvenc (0.9704, 1.4767)` LOO 3.47；
  `hevc_nvenc (1.1083, -2.9183)` LOO 3.72

**Why**：把等质量口径从「仅软编 6 档」扩到 T4 可跑的 `h264_nvenc`/`hevc_nvenc`（CQ 轴落 `QUALITY_MAP`、QP 轴落 `QUALITY_MAP_QP`）。

**How to apply**
- ⚠ **同名不同内容陷阱**：`screen_ui_code_src1280x720.mp4` 同时是 6s 与 10s 切片（内容不同、文件名相同）⇒
  直接按文件名池化会把两者并成 **16** 素材，给出**另一组** a/b（h264 `0.9256/6.3447`，而非 17 素材的 `0.9295/6.2523`）。
  必须去重为 17 个不同素材名。复现落表值前先确认素材数 = 17。
- ⚠ **判据钉 size 口径**：`crf_cq_unification_verify.load_quality_map()` 内 `set_quality_mode("size")`
  ⇒ G1/G3/G6/G7 期望值**不受** `QUALITY_MAP_QP` 落表影响（计划 §4.4 担心的 G3-1/G3-2/REF21 同步**并不需要**，
  实测 `--no-gpu --quick` 仍 PASS=94/FAIL=0）。落表后 **quality 口径**（生产默认）的 `to_constqp_qp`
  变为 h264 CQ26→QP22、hevc CQ28→QP23、h264 QP0→1；size 口径不变。
- 数据落点：`Accessory/data/eqq_calibration/points/gpu_t4_cq`（3 文件 442 点）、`gpu_t4_qp`（17 文件 459 点）；
  素材在**仓库外** `/workspace/input_videos/eqq_calib/`，用 symlink 按「素材名」接入后 `--src <link>`。
- 顺带修复 `Accessory/probe/eqq_calibrate_batch.py` 两个 latent bug：`load()` 是生成器却 `len(items)`
  （无 `--only` 必崩）、`--sweep` 为 `None` 时被迭代。
- 验收：`crf_cq --gpu` PASS=101/FAIL=0/WARN=4；`verify_equal_quality` 5/5；`plan_implementation_gate` 96/94/0/2；
  `pytest` 29 passed/2 failed（既有 chroma 环境项）；生产冒烟 hevc/h264 + LA=8 帧守恒（frames==packets=199）。
- 17 素材 QP 轴跑批约 27 min（`--jobs 4`；T4 有 2 个 NVENC 引擎，用户确认资源充足时并行安全）。
