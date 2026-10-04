---
name: VE 质量换算口径（size/quality）与 --quality-mode CLI
description: VE 生产新增 --quality-mode（对齐 VU）；口径含义/默认 quality/影响面/与门禁口径的关系；含 G3-9
type: project
---

VE 生产（`src/main_video_optimized.py`）2026-10-04 新增 `--quality-mode {size,quality}`
（对齐 VidUtils 的 `--quality-mode`），并加配置键 `processing.quality_mode`
（**CLI 覆盖 config，默认 `quality`**）。

**口径含义**（`src/utils/convert_crf.py`，全局态 `_QUALITY_MODE`）
- `size`（等体积）：用 `SIZE_MAP`，固定线性 `value=a·x264_crf+b`，目标**同文件大小/码率**。
- `quality`（等质量，**默认**）：用 `QUALITY_MAP`（CQ 轴）+ `QUALITY_MAP_QP`（QP 轴），实拍标定，目标**同 VMAF**；未覆盖编码器回退 `SIZE_MAP`。
- 消费方：`resolve_quality` / `to_x264_crf` / `from_x264_crf` / `to_constqp_qp`（经 `get_quality_map()`）。

**Why**：此前 VE 无开关、静默用默认 `quality`；VU 有 `--quality-mode`。补开关对齐两仓，并让用户可显式选等体积/等质量。

**How to apply**
- 应用点：`main()` 内 `_apply_cli_overrides` 之后、校验/处理之前
  `set_quality_mode(config.get("processing","quality_mode", default="quality"))`。
  默认不变 ⇒ **生产默认命令逐字不变**。
- ⚠ **门禁口径 ≠ 生产口径**：`crf_cq_unification_verify` 内部**硬钉 `size`**（`load_quality_map`）、
  `verify_equal_quality` **硬钉 `quality`**。故 G3/G6/G7 的 constqp 期望是 **size 值**
  （h264 CQ26→QP**21**），而**生产 quality 口径是 QP 22**（hevc CQ28→23；av1 27→63 两口径一致）。
- **覆盖闭环**：G6-1x 锁命令**形状**（口径无关）+ 新增 **G3-9** 锁 quality 口径**数值**
  （h264 CQ26→22 / hevc CQ28→23 / av1 CQ27→63；临时 `set_quality_mode('quality')` 后断言、`finally` 复位 size）
  ⇒ 生产 constqp 全覆盖。G3-9 已反向校验（扰动 `QUALITY_MAP_QP['h264_nvenc']` → 断言 FAIL）。
- ⚠ `to_constqp_qp(codec, 0)` 在 quality 口径返回 **1**（仿射模型不过原点）；生产**无损**走独立分支
  硬编码 `-qp 0`，不受影响（`crf_cq` 的 G3-4 "QP=0 保持 0" 是 size 口径断言）。
- 顺带修掉 `quality_map.py` 里"默认 `size`"的过期注释（真源 `convert_crf._QUALITY_MODE='quality'`）。
