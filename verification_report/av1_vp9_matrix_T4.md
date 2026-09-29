# AV1 / VP9 画质族矩阵

- 生成时间：2026-09-28T06:54:34
- 素材：`/workspace/input_videos/word_world_2.mp4`（687 帧）
- 软编基准：libx264 crf21 = 1427 kbps / PSNR 46.580407 dB
- 容忍带：RATE_PASS=(0.65, 1.5)、RATE_WARN=(0.55, 1.65)、TOL_PSNR_DB=1.5、TOL_PSNR_WARN=3.0

**汇总：PASS=0 / WARN=1 / FAIL=0 / SKIP=6**

## A 组 · 质量族矩阵

| 编码器 | 结论 | 下发 | 码率比 | 朴素比 | ΔPSNR | 说明 |
|---|:--:|---|---:|---:|---:|---|
| `av1_nvenc` | ⏭️ SKIP | — | — | — | — | 实跑一帧失败（需 Ada 及以上 GPU（L40/A10/RTX40））: [av1_nvenc @ 0x55adcc377480] No capable devices found |
| `av1_qsv` | ⏭️ SKIP | — | — | — | — | 本机 ffmpeg 构建不含该编码器（需 Intel QSV（Arc / 新 iGPU）） |
| `av1_amf` | ⏭️ SKIP | — | — | — | — | 本机 ffmpeg 构建不含该编码器（需 AMD AMF（RDNA3+）） |
| `libsvtav1` | ⏭️ SKIP | — | — | — | — | 本机 ffmpeg 构建不含该编码器（需 ffmpeg 构建含 libsvtav1） |
| `libaom-av1` | ⏭️ SKIP | — | — | — | — | 本机 ffmpeg 构建不含该编码器（需 ffmpeg 构建含 libaom-av1） |
| `librav1e` | ⏭️ SKIP | — | — | — | — | 本机 ffmpeg 构建不含该编码器（需 ffmpeg 构建含 librav1e） |
| `libvpx-vp9` | WARN | `-crf 28` | 0.95× | 1.30× | -1.67 dB | 默认基准 CRF 21 |

## B 组 · AV1 CONSTQP QP 尺度（方案 §7 AC1）

未执行（`av1_nvenc` 不可用或已跳过）——AC1 需 Ada/L40。

> 度量口径与 `Accessory/verify/crf_cq_unification_verify.py` 同源：
> 码率 = `ffprobe format=bit_rate`；PSNR = `-v info` + 显式 `[0:v][1:v]psnr`。
> ⚠ 裸 `-lavfi psnr` 或 `-v error` 都会给出错值。
