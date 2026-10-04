---
name: NVENC -tune/-multipass 改为显式 opt-in（裸 CQ 默认 + 目标码率）
description: 2026-10-04 VE 校正：CQ 默认改回裸 `-rc:v vbr`；新增 --nvenc-tune-*/--nvenc-multipass-*/--bitrate-*；唯一真源 src/utils/nvenc_tuning.py；仅 ffmpeg_io 层
type: project
---

2026-10-04 VE 按 VU 侧 A/B 结论（`Plan/ffmpeg_nvenc_knowledge.md` §5.1/§5.2）做**二次校正**：
FFmpeg 9.0 `vbr_hq` 迁移时给 CLI 加的 `-tune hq -multipass fullres` **被推翻**——
`-tune hq` 是 ffmpeg 默认值（写了等于没写），固定 CQ 下 `-multipass` **不升 VMAF**
（fullres ΔVMAF −0.006~−0.108 / qres −0.067~−0.335，码率 ×0.98~0.997）。

**新口径**
- 生产 CQ 默认 = **裸 `-rc:v vbr -cq:v N -b:v 0 -preset p4`**（去掉两对 token）。
- 显式 opt-in：`--nvenc-tune-ifrnet|-esrgan`（默认不发；`uhq` 仅 hevc/av1，显式落 `h264_nvenc` → **CLI 退出 2**）、
  `--nvenc-multipass-ifrnet|-esrgan`（默认不发；`rc_mode=='cbr'` 或给了目标码率时**自动补 fullres**，
  显式值优先含 `disabled`；`constqp` 忽略并告知）；`-tune` 与 RC 正交，constqp 下仍可下发。
- 目标码率 `--bitrate-ifrnet|-esrgan`、`--output-bitrate`、`--split-bitrate`（改发 `-b:v X` 去掉 `-cq:v`；
  **软编也支持**；NVENC 侧自动 fullres；output/split 侧同样自动 fullres）。
- **唯一真源 `src/utils/nvenc_tuning.py`**（`resolve_nvenc_tokens`/`resolve_tune_token`/`note_suppressed`/
  `note_non_nvenc`/`is_bitrate`），被两个 `external/*/ffmpeg_io.py` 共用。
- ⚠ **仅 ffmpeg_io / CLI 层**：SDK ctypes 主路径（`tuningInfo`/`multiPass`）**不变**。

**Why**：见上；CQ 归档无 multipass 收益，值在有码率目标的场景（CBR/受限码率）。

**How to apply**
- **默认命令必须保持裸**：判据 `crf_cq_unification_verify` 的 G6-1/G6-4 已改为 forbid `-tune`/`-multipass`；
  新增 G6-11~17 覆盖 opt-in/自动 fullres/`disabled`/目标码率/uhq。**这些断言做过反向校验**
  （故意注入 `-tune hq` 时 G6-1/G6-4/G6-11/G6-16 均 FAIL）。
- **标定溯源**：`QUALITY_MAP` 的 h264/hevc CQ 行是用**带 fullres** 的命令标的；Δ≤0.11 VMAF（LOO 噪声内）
  ⇒ **不重标**；`calibrate_equal_quality.BASE_LOCK` 已改为裸命令供后续复现。
- 生产 config 键：`models.ifrnet/realesrgan.nvenc_tune|nvenc_multipass|bitrate`、
  `output.video_bitrate`、`split.video_bitrate`（⚠ `output.bitrate` 是**音频**，别混）。
- 验证：`crf_cq --gpu` 108/FAIL=0/WARN=4、`--no-gpu --quick` 101/0/11、`plan_gate` 96/94/0/2、
  selftest 39/39、`--dry-run` rc=0、`uhq+h264` rc=2；pytest 29/2（既有 chroma 环境项）。
