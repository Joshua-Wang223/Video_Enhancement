# NVENC `-tune hq` / `-multipass fullres` 论断评审（2026-10-04）

评审对象：`Plan/ffmpeg_nvenc_knowledge.md` 第 5 节"重要"提示与
`Plan/FFmpeg9.0-移除-vbr_hq-速率控制模式.md` 的迁移口径。

## 结论：CQ 下禁用 `-tune hq -multipass fullres` 的论断不成立

| 论断 | 判定 | 依据 |
|---|---|---|
| `-tune hq` 会改变画质 | 错 | 它是 h264/hevc_nvenc 的默认值，`-h encoder` 显示 `(default hq)`，显式写是 no-op |
| "CQ 模式"不是 VBR，`-multipass` 在 CQ 下无效 | 错 | NVENC API 层 CQ 就是 `NV_ENC_PARAMS_RC_VBR` + targetQuality；nvenc.c:1046-1048 明确 `quality>=0 → rc=VBR` |
| 叠加后降 2.7–3.3 VMAF | 无据 | 引用源（arXiv 2605.01187）用纯 CBR、未测 multipass、报 BD-Rate % 不是 VMAF 分 |
| 官方推荐该组合 | 反证 | NVIDIA SDK 13.1 FFmpeg 指南 §6.2.8 官方配方 = `-rc vbr -cq 19 ... -multipass fullres` |

真正不适用 `-multipass` 的是 `-rc constqp`（NVEncC 文档：仅 --vbr / --cbr）。

## 关键源码事实（FFmpeg master, libavcodec/nvenc.c）

- L1036 `rcParams.multiPass = ctx->multipass;` —— 无 rc 模式门控，无条件下发给驱动。
- L1038-1041 仅 legacy preset 别名覆盖：`slow`→P7+TWO_PASSES，`medium`/`fast`→单遍。
- L1139-1151 CQ 分支：强制 `averageBitRate=0`、`vbvBufferSize=0`，只认 `maxBitRate`。
  即 CQ 下 `-b:v` 被丢弃，必须给 `-maxrate` 才能约束峰值。

## 对本项目的影响

1. FFmpeg 9.0 已移除 `vbr_hq` 等废弃 RC 选项（Changelog 确认）。本仓库 Level 2 FFmpeg 回退路径
   仍大量下发 `-rc:v vbr_hq`（ifrnet/realesrgan 的 `ffmpeg_io.py`、`config_manager.py` 默认
   `rate_mode="vbr_hq"`），升级 FFmpeg 9.0 会整条命令失败。迁移口径：`-rc vbr -cq N -maxrate M`。
2. Level 1 SDK 直通用的是 `NV_ENC_PARAMS_RC_VBR_HQ`（mode=32），不受 FFmpeg 选项移除影响，
   但同为已废弃 SDK 模式，需一并规划迁移。
3. 未实测：本机无 NVIDIA GPU。建议在 GPU 环境用 `Accessory/verify/crf_cq_unification_verify.py`
   的 VMAF 资产做固定 CQ 的 ±multipass A/B。
