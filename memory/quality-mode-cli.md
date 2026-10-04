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
- ✅ **门禁口径 == 生产默认口径（B1，2026-10-04 迁移）**：`crf_cq_unification_verify` 的口径钉
  已由 `size` **改为 `quality`**（`load_quality_map` 内 `set_quality_mode("quality")`）⇒ 其
  G1-2/G2/G3/G6 期望值随 `QUALITY_MAP` 更新（hevc 28→26、svtav1 24→29、vp9 28→26、rav1e 66→64；
  G3-1 21→22、G3-2 20→23、G6-2/5/17 `-qp 21`→22）。`verify_equal_quality` 本就用 quality。
- **覆盖闭环**：G6-1x 锁命令**形状**（口径无关）+ G3-1/G3-2 锁 quality 数值；
  **G3-9** 反向锁 **size 对照**（h264 CQ26→21 / hevc CQ28→20 / av1→63，临时切 size 后复位）。
- ⚠ `to_constqp_qp(codec, 0)` 在 quality 口径返回 **1**（仿射模型不过原点）⇒ 原 G3-4「QP=0 保持 0」
  已改为「quality 下 QP=0→1」，**生产无损契约改由新断言 G6-18/19 守**（writer 的 `crf==0` 分支
  硬编码 `-rc constqp -qp 0`，不经该函数）。G6-18/19 已反向校验（扰动无损分支 → FAIL）。
- 顺带修掉 `quality_map.py` 里"默认 `size`"的过期注释（真源 `convert_crf._QUALITY_MODE='quality'`）；
  `--quality-mode` 可随时在两口径间切换（生产默认仍 quality）。
