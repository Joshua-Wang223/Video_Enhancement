---
name: T4 vbr_hq 移除验证专项
description: FFmpeg 9.0 移除 vbr_hq 的 T4 GPU 验证——2026-10-04 已执行完成，裁定方案 A（仅 CLI 映射）
type: project
---

FFmpeg 9.0.2（编译用 libffmpeg-nvenc-dev 12.1.14.0）移除 `-rc:v vbr_hq`，但 SDK 12.x 头文件仍定义 `NV_ENC_PARAMS_RC_VBR_HQ=32`。核心判定：FFmpeg CLI 层必炸；ctypes SDK 层（`rc_ptr[1]=32`）需 T4 实测确认。

验证方案见 `Plan/T4_NVENC_vbr_hq移除_验证专项.md`。

## ✅ 实测结果（2026-10-04，Tesla T4 / 驱动 580.65.06 / FFmpeg 9.0.2 / NVENCAPI 13.0）

**裁定 P0 = 方案 A（仅改 FFmpeg CLI 映射；`nvenc_sdk.py` 不动）。**

- V8：`-rc:v vbr_hq` 被拒 rc=234（`Unable to parse "rc" option value`）
- V9：`-rc:v vbr -tune hq -multipass fullres` rc=0（h264/hevc 均通）
- V10：nv-codec-headers **13.0/13.1 头文件已删除** `VBR_HQ/QVBR/VBR_MINQP/CBR_HQ`（enum 仅 CONSTQP/VBR/CBR）——按原判定表本应落方案 B
- V12（决定性）：**驱动运行时仍接受 `rc_ptr[1]=32`** → `NVENCEncoder(rate_mode="vbr_hq")` 初始化成功（apiVersion=0xd0=13.0）
  - **行为反证**：同 30 帧噪声 `vbr_hq`=1,076,383B / `constqp`=2,159,969B / `qvbr`=1,018,148B，三者互异 ⇒ mode 32 真实生效，**非静默钳制为 CONSTQP**
  - 结论：头文件删除是开源 nv-codec-headers 裁剪，≠ 驱动移除；驱动向后兼容 32
- V14 画质（真实素材 640×360，cq=23）：h264 ΔVMAF=−0.048 / hevc ΔVMAF=−0.130（≤0.3 PASS）。⚠ 旧基线用备份 **FFmpeg 6.1.1**（`/var/backups/ffmpeg-v9/20261001-015049/usr_bin/ffmpeg`，接受 `vbr_hq`），存在版本混淆
- V15 LA：`-rc-lookahead 8` 输出 ≠ LA=0（生效）；SDK 路径 vbr_hq+LA8 E2E hevc 150→299 帧守恒

## 方案 A 改动范围（**已落地 2026-10-04**）

| 文件 | 改动 |
|---|---|
| `external/ifrnet_video/ffmpeg_io.py` | `_rc_v_map` vbr_hq/qvbr → `'vbr'`，默认 `.get(rc_mode,'vbr')`；vbr_hq/qvbr 追加 `-tune hq -multipass fullres`（h264/hevc） |
| `external/realesrgan_video/ffmpeg_io.py` | 同上（`_NVENC_RC_MAP`），与 ifrnet 侧逐字同源（G5-10） |
| `external/*/nvenc_sdk.py` | 仅加 `[PLAN-B-CANDIDATE]` 注释，行为不变 |
| `Accessory/probe/calibrate_equal_quality.py` | BASE_LOCK h264/hevc → `vbr -tune hq -multipass fullres`；selftest 同步 |
| `Accessory/verify/crf_cq_unification_verify.py` | `enc_nvenc`/`_probe_raw_driver_illegal`/G6-1/G6-4/G5-3 正则同步 |
| `Accessory/probe/av1_vp9_quality_matrix.py` | `_PROD_RC` h264/hevc 同步 |
| `Plan/` T4 母版 + FFmpeg9 说明 + L40 | 文档同步 |
| **不动** | `config/default_config.json` / `config_manager` / `main.py` / `_NVENC_LEVEL1_RATE_MODE` / 缓存 key（内部名 `vbr_hq` 保留） |

**验证**：`calibrate_equal_quality --selftest` ✅；`crf_cq_unification_verify --no-gpu --quick` 94/0/0；`plan_implementation_gate --skip-behavior` 50/48/0/2（同基线）；真实 FFmpegWriter 下发 `-rc:v vbr -tune hq -multipass fullres -cq:v 26 -b:v 0 -bf 0 -rc-lookahead 8` 出 30 帧有效 h264；生产 E2E hevc LA8 150→299 守恒。
**证据/数据落点**：`Accessory/data/vbrhq_migration_2026-10-04/`（README + metrics.json + `quality_compare/` 的 §4 新旧产物 + `smoke/` E2E 输入输出 + `writer_e2e/` + `reports/`，含 md5）。⚠ 旧产物由本机备份 FFmpeg 6.1.1 生成，9.0 无法重造 ⇒ 已入库。
**跨仓 handoff**：VU 侧 h264/hevc harness/生产/探针的 `-rc:v vbr_hq` 需同步改，否则 ⑨ 组红（本仓无法代改）。

## 关键陷阱（实测确认）

- `--rate-mode-ifrnet vbr` 走 SDK 路径会**静默落 CONSTQP 且 LA 失效**（`nvenc_sdk.py:1044` else→`rc_ptr[1]=0`；`:1056` LA 门控 `in ('vbr_hq','qvbr')`）。实测 Ready 行打印 `HEVC CONSTQP ... la=8`（la 只是回显，未真正启用）⇒ 方案 A 下**不要**给生产传 `vbr`。
- `Accessory/probe/nvenc_vbr_hq_verify.py` 原有 2 个 bug（已修）：① `sys.path` 少一层 `dirname`→`No module named 'external'`；② `subprocess.run(...).values()`（`CompletedProcess` 无 `.values()`）。

Why: FFmpeg 9.0 升级后生产必现；GPU 实测把改动范围钉死在 CLI 层，避免对 SDK ctypes 路径的无谓改动。
How to apply: 落地方案 A 时只动两个 `ffmpeg_io.py`；生产/harness 内部 rate_mode 名与 config 默认值均保持 `vbr_hq`。
