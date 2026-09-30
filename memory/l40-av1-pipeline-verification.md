---
name: L40 上 AV1 全链路验证（AC1 复核 + 长视频冒烟 P3 + 三处缺陷修复）
description: 2026-09-30 在 Tesla L40（Ada sm89）上复核质量控制参数方案的 AC1/AC2/AC4/AC7 与长视频冒烟 P3；AC1 ×3 独立复现；发现并修复「cv2 拒收 AV1」「ESRGAN writer 漏 av1_nvenc 静默改 libx264」「AV1 未降级 -rc:v vbr_hq 导致命令失败」三处缺陷；AC5 在 ffmpeg 7.1 上无法从选项表取 QSV/AMF 的 -cq 量程
type: project
---

2026-09-30 在 **Tesla L40（Ada，sm89，46 GB，48 核）** 上执行
`Plan/Video_Enhancement_质量控制参数修复方案.md` 的 §7 剩余项与 P3 长视频冒烟。
环境：torch 2.10.0+cu128、ffmpeg 7.1-\+av1（重编含 libsvtav1/libaom-av1/librav1e）、
OpenCV 4.13.0。权威记录见方案 **§8（本次新增）**。

## 一、AC 复核（一条命令入口 `Accessory/probe/av1_vp9_quality_matrix.py`）

| 项 | 结果 |
|---|---|
| **AC1** AV1 CONSTQP QP 尺度 | **独立复现 ×3**：`-qp 63` = 1.16× / −0.59 dB **PASS**；`21` = 2.64× / +3.86 dB FAIL（近无损暴涨，证实旧实现错）；`84` = 0.93× / −2.00 dB WARN；`105` = 0.74× / −3.54 dB FAIL |
| **AC2** G7-6（`crf_cq_unification_verify.py --gpu`） | `av1_nvenc -cq:v 27` ΔPSNR **+0.16 dB** / 1.28×（朴素 2.05×）**PASS**；全表 **PASS 101 / FAIL 0 / WARN 2**（WARN 仅 G7-1/G7-2 已知内容相关 −1.96/−2.66 dB） |
| **AC4** A 组 av1 格 | `-cq:v 27` 1.14× / −0.29 dB **PASS**（朴素 1.84×） |
| **AC7** | `libsvtav1 -crf 24` 1.09×/+0.25 PASS、`libvpx-vp9 -crf 28` 0.95×/−1.67 WARN（同 T4 逐位一致）；`libaom-av1`/`librav1e` 本轮**未跑**（CPU 上 1.9 MB 产物需 ~2 min/条，与"快速执行"冲突；两者已在 §6.10/§6.11 闭环） |
| **AC5** | ⏭️ 仍无法关闭，但**方法本身被证伪**（见下） |

### 探针的一处缺陷（已修）
`av1_vp9_quality_matrix.py` 的 AC1 判读把 **`84`（×4 假设）写死**在脚本里。表在
2026-09-29 已按 L40 实测改成 ×3（QP 63）后，该脚本在 L40 上会打印
「84 未落带内 ⇒ 改 a = 4.0」这种**与表自相矛盾**的结论。
已改为 `av1_expected_qp()` 现场走 `resolve_quality → to_constqp_qp` 推导锚点，
扫描点 `{21, 表值, 84, 105}`，并在 JSON/MD 里落 `av1_qp_expected` + `ac1_verdict`。
⇒ **教训：判据/探针里凡是硬编码的期望值，都要检查它是否已被上游改动作废。**

### AC5 的方法学纠正（重要）
方案 §7 AC5 写「探测 `ffmpeg -h encoder=av1_qsv | grep -A2 -- '-cq'` 取实际量程」——
**在 ffmpeg 7.1 上这条路走不通**：
- `av1_qsv` / `h264_qsv` / `hevc_qsv` 的编码器选项表里**根本没有 `-cq` / `-global_quality`**；
- `-cq` 只以**通用 AVCodecContext 选项**形式存在（`ffmpeg -h full` 里有 3 份副本：
  NVENC AV1 0~63、NVENC H.264/HEVC 0~51），通用条目**不按编码器区分量程**；
- `av1_amf` 连编码器名都不在本 build 里（需 `--enable-amf` 重编）。
⇒ 量程只能靠**实跑**取（像 `av1_nvenc` 那样），Ada 卡覆盖不了 QSV/AMF。
`QUALITY_MAP` 里 `av1_qsv`/`av1_amf` 的 `hi=51` 维持现状并标注"未核实"。

## 二、P3 长视频冒烟（本次主项）

素材：`input_videos/WordWorld_S2/2-01 …avi` 截取 **330 s / 640×360 / 7912 帧 @23.98 fps / 带 AAC 音轨**，
`interpolate_then_upscale`、2× 插帧 + 2× 超分（→1280×720）、`--segment-duration 30`（11 段）、
IFRNet 与 ESRGan **两侧都** `--codec-* av1_nvenc`。

| 项 | constqp | vbr |
|---|---|---|
| 管线退出码 | **0** | **0** |
| 段级解码级守恒 | 11/11 `decoded == expected` | 11/11 |
| 最终产物 | AV1 / 1280×720 / **15803 帧**（= 各段 2n−1 之和）/ AAC 立体声 / 330.000 s / 2.52 Mbps | 同帧数 |
| `validate_decodable_video(count_mode='decode')` | **ok=True**，15803 帧，0 解码错误，NVDEC 直解 | ok=True |
| `segment_bitstream_verify_v5 --skip-chroma` | frames==packets ✅、无 pts 异常 ✅ | ✅ |
| QA sidecar | 生成（`encoding_generation` / `codec_hint` / `rate_mode` / `fixes_applied` 齐全） | 生成 |
| 耗时 | 7 分 20 秒（**≈0.75× 实时**，含 TRT 引擎命中缓存） | 7 分 45 秒 |
| 宿主 RSS | 起步 0.6 GB → 平台期 ~5.0 GB；**后半段斜率 −326 MB/min（无泄漏）** | 同形态，峰值 6.0 GB |
| GPU 显存 | 峰值 6.6 GB / 46 GB（14%） | 5.6 GB |

⚠ **色度检查（检查 4）在 AV1 长片上是内容相关假阳性**：5 min AV1 产物报 113 个"坏帧簇"
（索引呈**固定步长 8**：716, 724, 732 …），而**未经管线的源片段自己就报 40 个**（358, 366, 374 …），
同一 30 s 干净片段上 AV1 / x264 crf21 / x264 lossless 三者均为 0 簇
⇒ 是内容触发、不是编码缺陷。**验收硬指标一律加 `--skip-chroma`**（与既有结论一致）。

## 三、本次发现并修复的三处缺陷（都在 AV1 路径上）

1. **`verify_video_integrity()` 用 cv2 读首帧 → 把完好的 AV1 产物判成损坏**
   （`src/utils/video_utils.py`）。OpenCV 4.13 自带 FFmpeg **无 AV1 解码**，
   `cap.read()` 返回 False，而系统 ffmpeg 解同一文件 1437/1437 帧、rc=0。
   上层随即 `unlink` 掉正确产物并终止整条流水线 ⇒ **AV1 在本机完全跑不通**。
   修法：cv2 失败时回退 `ffmpeg -v error -frames:v 1 -f null -` 探针
   （新增 `_ffmpeg_first_frame_ok`）。严格验收仍由其后的
   `validate_decodable_video(count_mode="decode")` 承担，未被放宽。
2. **ESRGan 侧 `FFmpegWriter` 把 `av1_nvenc` 漏出 NVENC 分支**
   （`external/realesrgan_video/ffmpeg_io.py`）。判定用的是精确元组
   `video_codec in ('h264_nvenc','hevc_nvenc')` ⇒ `av1_nvenc` 落到 `else` 的 libx264 分支，
   **静默**把 NVENC 的 CQ 值（27）当 CRF 下发（实测：请求 av1_nvenc，产物是 `libx264 -crf 27`），
   而且末尾 `[FIX-NVENC-PIPE]` 摘要还打印 "NVENC constqp(cq=27)"，**日志与实际命令自相矛盾**。
   修法：两处判定改为 `'nvenc' in video_codec`（与 `ifrnet_video/ffmpeg_io.py` 同口径），
   preset 映射也随之覆盖 av1（`medium`→`p4`，对应 av1_nvenc 的 default 档）。
3. **两侧 CLI writer 都没把 AV1 的 `vbr_hq/qvbr` 降级成 `vbr`**
   （`external/{ifrnet,realesrgan}_video/ffmpeg_io.py`）。av1_nvenc 的 `-rc` 只接受
   `constqp/vbr/cbr`，实测 `-rc:v vbr_hq` → `Undefined constant or missing '(' in 'vbr_hq'`
   → `Unable to parse option value` → **整条编码命令失败**。
   而配置默认 `rate_mode=vbr_hq`，且 AV1 的 SDK 直通路径在 L40 上必然失败（见下）⇒
   **"AV1 + 默认配置"在这台机器上开箱即挂**。修法：两侧与
   `nvenc_sdk.NVENCEncoder` 的既有降级同口径（`av1` in codec 且 rc ∈ {vbr_hq,qvbr} → `vbr` + 警告）。

### 附带观察：AV1 在 L40 上永远走不到 SDK 直通
`[NVENCEncoder] Level 1 失败: GetEncodePresetConfig failed, code=12`（`NVENC_ERR_INVALID_PARAM`）
—— 两侧都命中，随后按设计降级到 Level 2/3（ffmpeg CLI 管道 + NVENC）。
⇒ **AV1 段实际是"ffmpeg CLI 调 av1_nvenc"**，功能正确、只是少一层直通优化。
这也正是缺陷 2/3 会在生产暴露的原因：**AV1 从来只走 CLI writer 路径**。

## 四、回归（改动后）

`crf_cq_unification_verify.py --quick` = PASS 91 / FAIL 0 / SKIP 11；
`plan_implementation_gate.py` = 96 项 / 94 通过 / 0 失败 / 0 警告 / 2 跳过；
`pytest Accessory/test -q` = 24 passed。

**Why:** 用户要求"快速执行方案中最后剩余的需要 L40/Ada 的测试"。P3 长视频冒烟在
**开箱状态下直接失败**，根因是三处 AV1 路径缺陷（其中 1 处是验收层把好文件判坏）。
不修就无法验证 AV1 的 constqp/vbr 全链路。

**How to apply:** 新增/修改任何"按编码器名分支"的代码时，**用子串判定
（`'nvenc' in codec`）而不是精确元组**；新增硬编编码器时，务必核对它的
`-rc` / `-preset` / 质量参数名是否被 CLI writer 覆盖到。
验收层不要用 OpenCV 的解码能力当"文件是否完好"的判据。
相关：[[quality-params-t4-verification]]、[[nvenc-preset-and-encoder-availability]]、
[[ffmpeg-metric-measurement-traps]]。
