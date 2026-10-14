# AV1 / VP9 画质族矩阵

- 生成时间：2026-10-14T13:23:38
- 素材：`../input_videos/word_world_2.mp4`（687 帧）
- 软编基准：libx264 crf21 = 1426 kbps / PSNR 46.581025 dB
- 容忍带：RATE_PASS=(0.65, 1.5)、RATE_WARN=(0.55, 1.65)、TOL_PSNR_DB=1.5、TOL_PSNR_WARN=3.0

**汇总：PASS=3 / WARN=0 / FAIL=0 / SKIP=0**

## A 组 · 质量族矩阵

| 编码器 | 结论 | 下发 | 码率比 | 朴素比 | ΔPSNR | 说明 |
|---|:--:|---|---:|---:|---:|---|
| `libsvtav1` | PASS | `-crf 29` | 0.84× | 1.22× | -1.33 dB | 默认基准 CRF 21 |
| `libaom-av1` | PASS | `-crf 26` | 0.75× | 0.94× | -0.76 dB | 默认基准 CRF 21 |
| `libvpx-vp9` | PASS | `-crf 26` | 1.04× | 1.30× | -1.04 dB | 默认基准 CRF 21 |

## B 组 · AV1 CONSTQP QP 尺度（方案 §7 AC1）

未执行 —— SKIP：av1_nvenc 不可用（AC1 需 Ada/L40）。

> 度量口径与 `Accessory/verify/crf_cq_unification_verify.py` 同源：
> 码率 = `ffprobe format=bit_rate`；PSNR = `-v info` + 显式 `[0:v][1:v]psnr`。
> ⚠ 裸 `-lavfi psnr` 或 `-v error` 都会给出错值。
