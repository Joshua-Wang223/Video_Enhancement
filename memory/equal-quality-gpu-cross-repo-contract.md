---
name: 等质量 GPU 标定的跨仓契约（VE↔VU）
description: M4 GPU 等质量标定的跨仓契约（VE↔VU）：preset/RC 口径分歧会让两仓共享 QUALITY_MAP 的 NVENC 行不逐字相等（⑨ 组红）。CR-1 preset 已统一 p4；CR-2 因 FFmpeg 9.0 移除 vbr_hq/qvbr 二次修订为 CLI h264/hevc `vbr -tune hq -multipass fullres`（SDK 仍 RC_VBR_HQ=32）、av1 保持 vbr；含 VU 待同步 handoff、SDK 不支持 vbr 的陷阱与 p5 残留甄别
type: project
---

# 等质量 GPU 标定（M4）的跨仓契约 —— VE ↔ VidUtils

**核心事实**：M4（NVENC 等质量标定）的产物 `h264_nvenc` / `hevc_nvenc` / `av1_nvenc` 行要写入
**两仓共享**的 `QUALITY_MAP`，而 VidUtils 的 `verify/verify_quality_mapping.py` ⑨ 组断言
**两仓该表逐条相等**。⇒ 两侧若用**不同的编码口径**标定，表必然不等，⑨ 组红。

**Why**：两侧产品默认曾不同 —— VE 的 `medium→p4`（方案 E5 对齐 ffmpeg 官方枚举），
VU 原 `DEFAULT_PRESET_GPU="p5"`。preset 与 rate control 会整体平移率失真曲线 ⇒ 同 `-cq`
在不同 preset/rc 下达不到同一等效点。这是**口径分歧**，不是谁对谁错。
⇒ 2026-10-04 裁定**统一到 p4**，VU 侧已对齐（CR-1 收口）。

**How to apply**：上机（T4/L40）**之前**先落定 CR-1/CR-2；否则白跑一轮标定。
- **CR-1 已收口**（2026-10-04，两仓均 p4）：VE 本已 p4；VU 已把生产默认 + harness + 探针
  一并改 p4（残留 p5 为兼容显式 p5 的有意保留，详见 VE 方案 §12.5 的甄别表）。
- **CR-2 已裁定「路线 B」并因 FFmpeg 9.0 **二次修订**（2026-10-04）**：
  * **原裁定**（同日早些）：h264/hevc 统一 `vbr_hq`、av1 保持 `vbr`（理由：VE 生产 SDK Level 1 = RC_VBR_HQ）。
  * **二次修订**：FFmpeg 9.0 CLI **移除 `vbr_hq`/`qvbr`**（`-rc` 只剩 constqp/vbr/cbr；传 vbr_hq 报
    `Unable to parse "rc" option value`，rc=234）⇒ **CLI/harness/探针层** h264/hevc 改为
    **`-rc:v vbr -tune hq -multipass fullres`**；av1 保持 plain `vbr`。
  * **VE 侧已落地**：`ffmpeg_io` 的 `_rc_v_map`/`_NVENC_RC_MAP`（[FIX-FFMPEG9-VBRHQ]）+ `calibrate_equal_quality.BASE_LOCK`
    + `av1_vp9_quality_matrix._PROD_RC` + `crf_cq_unification_verify` 的 `enc_nvenc`/G6 期望。**VE 生产 SDK 路径不动**
    （`rc_ptr[1]=32`，T4 实测驱动 13.0 仍接受且行为非静默钳制）⇒ 快路径逐字节不变。
  * **⚠ VU 必须重新同步（handoff，本仓无法代改）**：VU 生产/harness/探针/`t4_acceptance` A4 的
    `-rc:v vbr_hq` 同样会被 FFmpeg 9.0 拒绝 ⇒ 改 `vbr -tune hq -multipass fullres`，否则 ⑨ 组变红。
  * **陷阱不变**：VE `nvenc_sdk` **不支持内部名 `vbr`**（`else` 静默落 CONSTQP + LA 门控只认
    `vbr_hq/qvbr`）⇒ 只改 CLI token；**不要**把内部 rate_mode 改成 vbr（除非同时上方案 B）。

## 契约清单

| 编号 | 内容 | 现状 | 处置 |
|---|---|---|---|
| **CR-1** | NVENC preset：VE `p4` vs VU `p5` | ✅ **已收口（两仓均 p4）** | **VE 无需改**（本就 p4）。**VU 已改**：生产 `DEFAULT_PRESET_GPU` p5→p4（`vidcrop_cpu_v2.py` + `vidcrop_hwaccel.py` 孪生）+ harness `BASE_LOCK` p4 + 探针 `NVENC_PRESET='p4'` + README/`test/baseline` 同步。残留 `p5` 均**有意保留**（反向降级表 `p5→medium`、官方枚举 `slow→p5`、`_NVENC_PRESET_RETRY` 兼容显式 p5、QSV 断言） |
| **CR-2** | NVENC rate control 口径：h264/hevc 与 av1 的 rc 两仓统一 | ⚠ **FFmpeg 9.0 二次修订后：VE 已落地，VU 待同步** | CLI/harness 层 h264/hevc 改 `vbr -tune hq -multipass fullres`、av1 `vbr`（均显式 `-rc`）；**VE SDK 仍 RC_VBR_HQ=32 不动**。**VE 已落地**（生产 writer `_rc_v_map` + harness `BASE_LOCK` + 探针 `_PROD_RC` + 判据 `enc_nvenc`/G6）。**VU 待同步**：其 `_NVENC_DEFAULT_RC`/harness/探针/`t4_acceptance` A4 的 `vbr_hq` 需改 `vbr -tune hq -multipass fullres`，否则 ⑨ 组红（handoff）。原「路线 B（vbr_hq）」口径被 FFmpeg 9.0 移除所推翻 |
| **CR-3** | harness 是否同版 | VE 有 `--axis`（QP 轴 D2b），VU 无 ⇒ md5 不同 | **有意差异**；共享块改动两侧同步，`--axis` 各自演进 |
| **CR-4** | QP 轴归属 | `QUALITY_MAP_QP` 仅 VE；VU 只有 `_QP_SCALE` | VU 改 `_QP_SCALE`（av1 ×3）须通知 VE 同步 |
| **CR-5** | 素材池 | 17 条切片在 VE `input_videos/eqq_calib/`（**仓库外、不入 git**） | 上机机需先就位；两仓共用同一池保证口径一致 |

## 态势快照（2026-10-04）

- **VU harness 早已实现 G0/T0/A0**（NVENC + `--require-codecs`/`--expect-av1` fail-fast +
  `nvidia-smi` 指纹 + `_table_range` 回退 + **跨仓态势 `cross_repo_status`**）；
  VU 专项方案 `Plan/VidUtils_等质量标定_{T4,L40}_专项执行方案.md`（任务 `G0~G7`/`T0~T6`/`A0~A5`）。
- **VE harness 已移植 VU 实现 + 叠加 VE 独有 `--axis {cq,qp}`**（2026-10-04）：
  selftest 39 项、CPU 干跑通过、跨仓态势打印正确、无 GPU 时探测 exit 2。
- **跨仓态势已双向**：此前只有 VU 能感知 VE，现 VE 也打印「两表是否相等 / 对侧 harness 是否同版 /
  对侧方案文档」（VE harness `cross_repo_status()` / `_print_cross_repo()`）。
- **`_qp_model` 已改「模式感知」**：quality 口径优先 `QUALITY_MAP_QP`、size 口径保持 override→活动表。
  ⚠ 2026-10-04 **B1**：`crf_cq` 判据口径由 size **改为 quality**（== 生产默认），故 G3/G6 期望值
  已更新为 quality 值（不再是"零侵入"）。
- **CR-1 preset 已收口**（2026-10-04 复检）：VU 生产 `DEFAULT_PRESET_GPU` 两文件均 **p4**、harness/探针 p4；
  残留 `p5` 全为有意保留（反向降级表 / 官方枚举 / `_NVENC_PRESET_RETRY` / QSV 断言）——
  甄别表见 VE 方案 §12.5。
- **CR-2：FFmpeg 9.0 引发二次修订（2026-10-04）**：FFmpeg 9.0 CLI 移除 `vbr_hq`/`qvbr` ⇒
  CLI/harness/探针层 h264/hevc 从 `vbr_hq` 改为 `vbr -tune hq -multipass fullres`（av1 仍 plain `vbr`）。
  **VE 已落地**；**VE SDK 路径不动**（`rc_ptr[1]=32`，T4 实测驱动 13.0 仍接受且行为非静默钳制）。
  **⚠ VU 待同步**（生产/harness/探针/A4 的 `vbr_hq`），否则 ⑨ 组变红。**关键陷阱**：VE `nvenc_sdk`
  不支持内部名 `vbr`（静默 CONSTQP + LA 失效）⇒ 只改 CLI token。详见 [[t4-vbrhq-verification-plan]]。

## 关联

- 方案：`Plan/PROMPT_T4_NVENC等质量标定专项执行方案.md` §12、`Plan/PROMPT_L40_AV1等质量标定专项执行方案.md` §12
- 立项目标与 B 组待办：`Plan/PROMPT_等质量换算立项.md` §0.0 / §7.1
- VU 侧：`/workspace/VidUtils/Plan/VidUtils_等质量标定_{T4,L40}_专项执行方案.md`
- ⑨ 组脚本：`/workspace/VidUtils/verify/verify_quality_mapping.py`
