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

## 五、第二轮（同日续做）：回归保护 + V9 复核

### 5.1 判据：G6-8 / G6-9 / G6-10（`av1_nvenc` 命令形状）

§三 的 ② ③ 两处缺陷本质是"下发了错的命令"，最该由命令形状断言守住
（G6 组本就是干这个的，但只覆盖 constqp）。三格都**不需要 AV1 硬件**：

| ID | 侧 | rate_mode | 必须出现 | 必须不出现 |
|---|---|---|---|---|
| G6-8 | ESRGAN | constqp | `-vcodec av1_nvenc`、`-preset p4`、`-rc:v constqp`、`-qp 63` | `-crf`、`-cq:v` |
| G6-9 | IFRNet | **vbr_hq** | `-rc:v vbr`、`-cq:v 27`、`-b:v 0`、`-rc-lookahead 8` | `-rc:v vbr_hq` |
| G6-10 | ESRGAN | **vbr_hq** | `-vcodec av1_nvenc`、`-rc:v vbr`、`-cq:v 27`、`-b:v 0` | `-rc:v vbr_hq`、`-crf` |

**反向验证**：临时回退 ② ③ 两处修复 → 三格**全 FAIL**（且诊断精确指出缺哪些/多哪些），
恢复后 PASS=93。**新断言必须先证明"回退修复后它会 FAIL"，否则可能一辈子抓不到东西。**

### 5.2 pytest `Accessory/test/test_verify_video_integrity_fallback.py`（3 例）

猴补丁把 `cv2.VideoCapture` 换成"读不出帧"：① 好文件仍判完好；② **4 KB 垃圾文件仍判坏**
（证明回退没把判定放宽）；③ 缺文件 / <1KB 老闸门不变。反向验证同样做过（旧行为 ⇒ ① FAIL）。
**不需要 AV1 硬件**（用 libx264 样本 + 猴补丁即可）。

### 5.3 可重复脚本 `Accessory/verify/av1_pipeline_smoke.py`

把 §二 那次手搓的冒烟固化成一条命令；8 项验收（退出码 / 段级门禁 / 帧数等式 /
`validate_decodable_video` / 码流硬指标 / QA sidecar / **产物编码器确为 av1** / 泄漏斜率），
支持 `--checks-only` 复验既有产物（**不需要 GPU**）。退出码 0/1/**2（环境前置不成立）**。
⚠ 该脚本的 `S7`（产物编码器 = av1）正是能抓住 5.1 表里 ② 那类"静默换编码器"的断言。

### 5.4 P2 · V9 三行复核（无缓存脚本 + `--dense`，3 素材）

| 素材 | `libx265` | `libvpx-vp9` | `libsvtav1` |
|---|---|---|---|
| **m1 · 720p30（原标定基准）** | Δ **+0.00** | Δ **+0.14** | Δ **−0.02** |
| m2 · 1080p30 | −0.52 | +0.32 | −1.32 |
| m3 · 360p 低复杂度 | −0.89 | **−2.27** | −1.07 |

（Δ = 实测 crf21 落点 − `QUALITY_MAP` 现表落点。）

⇒ **现表正确，不改表**。残差是**内容/分辨率相关**，最大 −2.27 落在 `libvpx-vp9`
（= AC7 里 ΔPSNR −1.67 dB 那个已知 WARN 的编码器）。单一线性常量压不掉内容相关误差
⇒ 与 E9 判 `CQ_OFFSET=0`、E10 判"合成过配/真实欠配"同源。证据
`verification_report/v9_calib_nocache_m{1,2,3}_*.json`。
⚠ 4K / 50fps+ 类型仍未覆盖（CPU 标定耗时过长）。

### 5.5 ⚠ 会话中途 GPU 断联（环境教训）

06:07 起 `/usr/lib/x86_64-linux-gnu/libcuda.so.1` 与 `libnvidia-ml.so.1` 被指向 **0 字节**的
`libcuda.so.580.65.06`，`/dev/nvidia*` 消失 ⇒ `nvidia-smi` / `av1_nvenc` / `torch.cuda` 全废。
后果：**P3″（AV1 Level 1 `GetEncodePresetConfig code=12` 根因）与 AC5 一并按"环境不支持"跳过**。

* ✅ 门禁的探测按设计降级：`R5 CUDA`、`R7 NVENC` 两项 **WARN，FAIL 仍为 0**
  ⇒ "环境差异 ≠ 功能失败"的设计意图得到实证。
* ⚠ 教训：长会话里 GPU 可能被回收，**GPU 结论必须当场实测**，历史报告不能替代复跑。
* 复现 NVENC SDK 问题时**必须先 `import torch; torch.cuda.init()`** ——
  否则会撞上 0 字节的 `libcuda.so.1` 桩，报 `Cannot load CUDA library: ... file too short`。
* P3″ 的假设与可直接执行的复现片段见方案 **§9.6**（`GetEncodePresetConfigEx` 回退，索引 39）。

## 六、回归（第二轮后）

`--quick` = PASS 93 / FAIL 0 / SKIP 12（+3 = G6-8/9/10）；
门禁 = 96 项 / 92 通过 / **0 失败** / 2 警告（GPU 断联）/ 2 跳过；`pytest` = **27 passed**。

**Why:** 用户要求"快速执行方案中最后剩余的需要 L40/Ada 的测试"。P3 长视频冒烟在
**开箱状态下直接失败**，根因是三处 AV1 路径缺陷（其中 1 处是验收层把好文件判坏）。
不修就无法验证 AV1 的 constqp/vbr 全链路。

**How to apply:** 新增/修改任何"按编码器名分支"的代码时，**用子串判定
（`'nvenc' in codec`）而不是精确元组**；新增硬编编码器时，务必核对它的
`-rc` / `-preset` / 质量参数名是否被 CLI writer 覆盖到。
验收层不要用 OpenCV 的解码能力当"文件是否完好"的判据。
相关：[[quality-params-t4-verification]]、[[nvenc-preset-and-encoder-availability]]、
[[ffmpeg-metric-measurement-traps]]。

---

## 2026-10-06 核对：L40 专项实质已完成，方案 §0.4/§0.5 过期自相矛盾

用户要求核对 `Plan/PROMPT_L40_AV1等质量标定专项执行方案.md` 是否只剩 L40 侧未完成。
逐条与代码 + 头部 + 本记忆对账后结论：**实质 L40 工作早已落表完成**（提交 `b64c1d2`/`20b9e81`/`95a3f75`）：

- **L40-1 CQ**：`src/utils/convert_crf.py:212` = `(1.4566, 1.2165, 0, 63)`（crf21→`-cq:v 32`）。
- **L40-2 QP**：`src/utils/quality_map.py:202` = `(7.9338, -97.5136, 0, 255)`（仿射，取代 ×3；`_QP_MAP_OVERRIDE:365` 仅余 size 口径/回退）。
- **L40-3 AC1**：`verification_report/av1_vp9_matrix_L40_cqlanded_2026-10-04.md` 表值 `-qp 70` 落带内 PASS。
- **L40-4 AC2/G7-6**：`verification_report/crfcq_gpu_L40_cqlanded_2026-10-04.md` PASS=113 / FAIL=0。
- **L40-8 跨仓**：`/workspace/VidUtils/convert_crf.py:207` 已是 `(1.4566, 1.2165)`；`verify_nvenc_quality_gpu.py:116` 有仿射 `(7.9338, -97.5136)`。

⚠ **方案自身自相矛盾**：§0.4（:346-358）与 §0.5 步 6/7（:362-372）在**最新提交 `1bfea98`** 里把
L40-1/2 标定与 `QUALITY_MAP` / `QUALITY_MAP_QP` 落表重新标成「⏸ 阻塞：需 L40」，
而同文件**头部**（提交 `ea65e19`）写「2026-10-04 执行完毕并落表」⇒ 与代码 / 本记忆直接冲突。
定性：把 **T4 母版的状态块**搬入时未与 L40 落表结果对账留下的**过期块**（不是新回归）。

**真正未完成项（不全是 L40）**：
- L40-5 / S8：见文末「### S8 裁定」——**判「工程闭环」**（T4 代理路径不复现），
  但**不等于根因已解释**；且原「窗口伪影」论证经复核**不成立**（跨窗口错误对比）。
- L40-6（可选）：AV1 Level 1 `GetEncodePresetConfig code=12` 根因，未做。
- **非 L40**：B2 **方案 B（真 vbr 分支）未做**（方案 §0.1.5 步 4 自认）；§9.5 遗留 E①
  （`av1_vp9_quality_matrix` 退出码 1 vs 0）未闭环；文档卫生——§5.4:547 / §10:654:689 仍是过期
  `--rate-modes constqp,vbr`（§0.2.2 已改 `constqp,vbr_hq`）、§5.3:515:522 / §10:681 用相对路径
  `input_videos/eqq_calib/`（素材池在**仓库外** `/workspace/input_videos/eqq_calib/`，会失败）。

**环境**：本会话无 GPU（`nvidia-smi` 找不到 `libnvidia-ml.so`、torch `is_available()=False`）；
`git status` 干净，`HEAD == origin/main == 1bfea98`。

**How to apply:** 引用该方案的 §0.4/§0.5 前先与代码落表值 + 头部 + 本记忆对账；
不要把已落表的 L40-1/2 当待办；文档更新时优先修 §0.4/§0.5 与 §5.4/§10 命令口径。

---

### S8 裁定（2026-10-06，无 GPU，离线复算）

用户要求裁定 S8 是否算闭环。逐条用**归档原始数据**（`verification_report/s8_20261004_raw/*.mem.tsv`）
独立复算后结论：**可判「工程闭环」，但属「不可复现 / 风险接受」型，不等于「根因已解释」。**

- **判据口径**：S8 = `av1_pipeline_smoke.py` 的**后半程（后 50%）RSS OLS 斜率**；
  L40 当次所用脚本（`b64c1d2`）同为 `half = rows[len//2:]` ⇒ L40 的 **+149.5 是后 50%** 口径。
- **支持闭环**：匹配条件（同素材 / bs=24 / 后 50%）T4 constqp 臂无泄漏 ——
  `h264_constqp −55.6±11.0`、`hevc_constqp +3.1±14.9` MB/min（95%CI 均不含 +50）；
  bs=8 复跑 hevc +47.4。判据窗口极不稳定（同数据 −786~+1434）。
- **必须更正的论证**：文档/记忆「L40 +149.5 ≈ T4 **全程** +149.4 ⇒ 窗口伪影」是
  **跨窗口错误对比**（L40 是后 50%、T4 是全程）；且 L40 **原始采样从未落盘**
  （未用 `--mem-dump-dir`）⇒ 闭环属**类比推断**，非直接复测。
- **噪声口径纠正**：后 50% 斜率 **SE 实算仅 ±8~15 MB/min（95%CI ±20~30）**；
  记忆里的「±66 MB/min」是**残差 sd(≈1356 MB)** 量级、不是斜率不确定度
  ⇒ 在该口径下 **+149.5 不是噪声**（"噪声"解释不成立）。
- **同臂两跑不稳**：L40 `av1_nvenc` constqp 两次 = **+112.1 / +149.5**（耗时 1052s vs 661s）。
- **裁定**：记为「**闭环（不可复现；判据不稳定）**」并从方案 §0.4/§0.5 移出；
  若要「硬闭（根因级）」须一次 L40 复跑（`--batch-size 8` + `--mem-dump-dir`）。

**How to apply（补充）：** 判"某 FAIL 是伪影/噪声"必须能在**同一口径**复现该数字；只能跨口径得到
"相似数字"时，结论降级为「不可复现/风险接受」，不得写成「根因已解释」。原则见
[[feedback_closure_evidence_same_scope]]。
