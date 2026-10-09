---
name: l40-full-verification-run-2026-10-09
description: 2026-10-09 在 L40 上实跑 Plan/PROMPT_L40_全流程验证.md 的 4 项结果（落表双轴 PASS / crf_cq --gpu 113-0-3 且 G7-6 仍 WARN / AV1 矩阵 AC1 PASS / 冒烟 S8 假 FAIL 机理），并记录长视频素材被源结构门禁硬拒的阻塞与待决项；测试已暂停
metadata:
  type: project
---

2026-10-09 按 `Plan/PROMPT_L40_全流程验证.md` 在 L40（sm89，驱动/CUDA 正常）实跑。
**用户中途要求「保存现场并暂停测试」**，本条为暂停时的现场快照，恢复时从 §5 待决项接手。

## 0. 前置门禁（AV1 可用性必须实跑一帧）

`nvidia-smi -L` = NVIDIA L40；`torch.cuda` = L40 / True。
三编码器各实跑一帧（**不用 `-h encoder=`**）：`h264_nvenc` rc=0、`hevc_nvenc` rc=0、`av1_nvenc` rc=0。
⇒ AV1 门禁**通过**，4 项均可执行。

## 1. L40-4 落表门禁 —— ✅ 双轴 PASS（本轮唯一完全达标项）

点数据实际在 `Accessory/data/eqq_calibration/points/`（**不是** 仓库根的 `points/`）。

| 轴 | 命令 | av1_nvenc 拟合 | LOO[0,27] | 门限 | 判定 |
|----|------|----------------|-----------|------|------|
| cq | `--sides gpu_l40_cq --axis cq` | (1.4566, +1.2165, 0, 63) | **3.13** | ≤5.9 | ✅ |
| qp | `--sides gpu_l40_qp --axis qp` | (7.9338, −97.5136, 0, 255) | **2.61** | ≤5.9 | ✅ |

- 顺序无关性 3 seed × 6 档逐位一致 ✅（两轴均是）。
- 落表候选与现值**逐位一致**：`src/utils/convert_crf.py:212` = `(1.4566, 1.2165, 0, 63)`（**CQ 表在 `convert_crf.py`，不在 `quality_map.py`** —— 后者只有 `QUALITY_MAP_QP`）、`src/utils/quality_map.py:202` = `(7.9338, -97.5136, 0, 255)`。⇒ 无需改表。
- 其余 8 档打印「拟合失败（样本 0 < 4）」属正常：该 L40 池只采了 av1_nvenc 的点。**判据是「输出中确有 av1_nvenc 的 LOO 数值行」**，本轮已满足 ⇒ 不属 [[verification-script-path-and-axis-traps]] ② 的静默假通过。
- ⚠ 同目录下存在 `gpu_l40_cq_dense/`；`find_points()` 用 `Path(points_dir)/side` 精确 join + `*.json` glob，**不会**被 `--sides gpu_l40_cq` 误纳，已核实。

## 2. L40-2 `crf_cq_unification_verify --gpu` —— ⚠ 方案判据未达成

`verification_report/crfcq_gpu_L40_2026-10-09.md` / `CRF_CQ统一验证结果_20261009_072836.json`
**PASS=113 / FAIL=0 / WARN=3 / SKIP=0**

- 「G7/G8 FAIL=0」✅ 达成。
- **「G7-6 av1_nvenc `-cq` PASS」❌ 未达成**：G7-6 仍为 **WARN**，且数值与方案记录的历史基线**完全相同**：

  | | ΔPSNR(有符号) | 码率比 | 换算值读数 | 朴素对照 |
  |---|---|---|---|---|
  | 本轮（`-cq:v 32`） | **−2.14 dB** | **0.87×** | PSNR 44.44 / SSIM 0.9845 / 1240 kbps | `-cq:v 21`：49.33 / 0.9934 / 2923 kbps（2.05×） |
  | 方案历史基线 | −2.14 dB | 0.87× | — | — |

  ⇒ **`-cq 27 → -cq 32` 并未修正 G7-6**（质量比 libx264 crf21 松 2.14 dB，但码率只用 0.87×，属「偏松」而非码率超标）。
- 另两条 WARN：G7-7（libsvtav1 −1.64 dB / 0.84×）、G8-4H（>1080p 档位天花板，`avgBitRate` 非硬上限、真正硬上限是 `maxBitRate=2×cap`）。
- 口径对齐：T4 侧记为 114 PASS/0 FAIL/2 SKIP，L40 侧 113+3 WARN=116，与 T4 的 114+2 SKIP=116 守恒 —— 差别只是 G7-6 在 T4 上 SKIP、在 L40 上真跑出 WARN。
- ⚠ 注意 `memory/l40-av1-pipeline-verification.md:17` 记的是更早的 `-cq:v 27` 读数「ΔPSNR **+0.16 dB** / 1.28× PASS」。**同一 G7-6 在 cq27 时 PASS、改 cq32 后变 −2.14 dB WARN** ⇒ 那条历史行已过期，勿再引用为「已验收」。

## 3. L40-3 AV1 质量矩阵 AC1 —— ✅ PASS

`verification_report/av1_vp9_matrix_L40_2026-10-09.md`（`--only av1_nvenc`，A/B 组共 5 行）

- **AC1 PASS**：表值 `-qp 70` 落带内（码率比 **1.07×** / ΔPSNR **−1.13 dB**），与历史基线（1.07× / −1.13 dB）一致。
- B 组其余 3 点为**有意带外**的对照值（`-qp 21/84/105`），出带属预期，脚本明示「不计退出码」。
- A 组 `av1_nvenc -cq:v 32` 码率比 0.77× / ΔPSNR −2.60 dB（与 §2 的 G7-6 同向）。

## 4. L40-1 AV1 冒烟 —— ✅ 功能全通；S8 的 FAIL 已定位为**测量假象**

### 4.1 `new5_raw.mp4`（1920×1080 → 输出 **3840×2160** AV1），15 PASS / 1 FAIL

`verification_report/av1_smoke_L40_new5raw_2026-10-09.md`；明细 `/tmp/s8_mem_new5/{constqp,vbr}.mem.tsv`

| 臂 | S1~S7 | S8 | rc | 耗时 |
|----|-------|----|----|------|
| constqp | ✅ 全通 | ❌ 斜率 **+712.0 MB/min** | 0 | 844.9s |
| vbr | ✅ 全通 | ✅ 斜率 **−928.5 MB/min** | 0 | 240.4s |

**S8 FAIL 是假象，已用逐点采样证实。** constqp 臂 RSS 是**阶跃**而非线性累积：

| t(s) | 0 | 55 | 334 | 390 | 614 | 670 | end(835) |
|---|---|---|---|---|---|---|---|
| RSS MB | 8 | 2180 | 2423 | 5248 | 5495 | 9310 | 7073 |

每一段平台期内 RSS 平坦到 ±5 MB；台阶分别对应**一次性 TRT engine 构建**（IFRNet @55s、ESRGAN @390s）与编码阶段切换。后半程 OLS 把这些台阶读成了线性增长。
**对照证据**：vbr 臂引擎已缓存 ⇒ 斜率 **−928.5 MB/min**（单调下降）。
⇒ 判定**无内存泄漏**。这正是 [[av1-nvenc-l40-calibration]] 里「⚠ 只跑 constqp 不足以判读，须 constqp,vbr 同素材对照」的设计意图，本轮以对照臂闭环。

### 4.2 复现要点（下次直接照做）

- `--mem-peak-mb 0`：默认 16000 MB 是 **T4 / 720×576 / bs=24** 标定值，L40 + 高分辨率下会假 FAIL，本轮全程用 0（只判斜率）。印证 [[l40-verification-scope-2026-10]] 的告警。
- **先跑一次短片预热 engine 缓存**，正式跑时 constqp 臂即可拿到干净 S8（本轮对 `112 Max Bed Time` 做了预热，`H480_W736`/`H480_W720` 两个 sm89 engine 已缓存）。
- engine 按 shape 缓存，换分辨率要重建：IFRNet ~5-10 min、ESRGAN ~8-12 min。
- `[FIX-HIGHRES-RC]` 实测生效：输出 ≥2160p 时自动 `lookahead 8→0`（消除高分辨率 NVENC drain 背压），本轮 4K 输出臂命中。
- S8 的 `显存峰值` 恒为 0 MiB（watcher 未采到 NVML），但管线自身报的 GPU 峰值可信（constqp 臂 4.55 GB）。

## 5. 长视频素材选型 —— 已定：`大红狗 Clifford…`，但**未跑**（掉卡）

### 5.0 先前两个 AVI 候选：均不可用（已排除）

用户先后指定 `112 Max Bed Time.avi` 与 `402 The Blue Tarantula .avi`，**两者音轨 mp3 头都损坏**：

| | 112 Max Bed Time | 402 The Blue Tarantula |
|---|---|---|
| 分辨率/编码 | 720×480 / mpeg4 | 720×480 / mpeg4 |
| fps → 2x 输出 | 29.97 → **59.94** | 23.976 → 47.95 |
| 帧数/时长 | 14799 / 493.8s（17 段） | 10703 / 446.4s（15 段） |
| 音轨 | mp3 44.1k，**头损坏** | mp3，**同样头损坏** |

**根因**：`[mp3float] Header missing` ⇒ 被 `video_utils.validate_source_video_structurally()`
（`[P5-FIX-SOURCE-STRUCT-GATE]`，`video_utils.py:1285`）硬拒，**两臂均 rc=1、5.1s 速退**：
`❌ 源视频结构校验失败: source_decode_errors: [mp3float] Header missing`。
⇒ 这是门禁**正确工作**（属 QA sidecar 的 `DECODABLE_GATE` 修复项），不是管线缺陷。

**已试过的绕行（均未彻底解决，勿重复踩）**：
1. `-map 0:v -map 0:a -c:v copy -c:a aac` 重封 → 视频 MD5 **bit-identical**（`3276140bb5eb401d85bfd3c31d471619`）、14799 帧/493.79s 完整，但暴露**第二个问题**：源 AVI 末帧**非单调 DTS**（`14798 >= 14797`），`-c:v copy` 把它带进了 MP4。
2. `-fflags +genpts` / `-avoid_negative_ts make_zero` / `-fflags +igndts` 三种组合**均无效**，dts 照旧。
3. 换 MKV 容器 → 视频流损坏（`Cannot determine format of input 0:0 after EOF`）。
4. 若将来仍要用这两个 AVI，只能视频也重编码（`-c:v libx264 -crf 18`）造干净源 —— 会使素材不再是原始码流。

⚠ **不建议**去改 `validate_source_video_structurally` 的判定口径：门禁「有任何 stderr 就拒」偏严，
但放行 mpeg4 **warning** 级（`Discarding excessive bitstream`）会削弱对真损坏的拦截力，属独立立项。

### 5.1 最终选型：`大红狗 Clifford the Big Red Dog DVDR.58.mp4` ✅ 前置全通

| 项 | 值 |
|----|-----|
| 容器/编码 | MP4 / **h264**（非 mpeg4，无 xvid 码流告警） |
| 分辨率 / fps | 720×576 / 25 → 2x 输出 **1440×960 @ 50fps** |
| 帧数 / 时长 | **18011 / 720.52s**（24 段 @30s），码率 900 kbps |
| 音轨 | **aac 48kHz 立体声，解码零错** |

**已核验的前置条件（均无需 GPU 即可复核）**：
- ✅ **源结构门禁通过**：`ffmpeg -v error -i <src> -map 0:v:0 -f null -` ⇒ **rc=0、stderr 空**（两个 AVI 正是卡在这）。
- ✅ **TRT engine 零构建**：`.trt_cache/` 已有该形状的 sm89 引擎 ——
  `IFRNet_S_Vimeo90K_B8_H576_W736_fp16_sm89_nvidial40.onnx` 与
  `realesr-general-x4v3_B8_C3_H576_W720_fp16_sm89_nvidial40.onnx`。
  ⇒ 直接开跑即可，**省掉 IFRNet 5-10min + ESRGAN 8-12min**，且 constqp 臂能直接拿到干净 S8。
- 预估单臂 25~40 min（720×576 每帧比 new5_raw 的 1080p 便宜约 5x，但帧数多 22x），两臂约 50~80 min。

### 5.2 ⛔ 未执行原因：掉卡

开跑时 GPU 已不可用：`nvidia-smi` 报 `couldn't find libnvidia-ml.so`，torch 侧 `Cannot load libcuda.so.1`。
`av1_pipeline_smoke.py` 的**前置门禁正确拒绝**：
`前置: av1_nvenc ⏭️ 不可用 — Cannot load libcuda.so.1` → **exit 2**。
⇒ **未产出任何数据**（无 output dir、无 mem.tsv、无报告），现场无误导性半成品。恢复时直接重跑即可。

恢复命令（GPU 回来后）：
```bash
cd /workspace/Video_Enhancement
python3 Accessory/verify/av1_pipeline_smoke.py \
  --src "/workspace/input_videos/大红狗 Clifford the Big Red Dog DVDR.58.mp4" \
  --rate-modes constqp,vbr --segment-duration 30 --batch-size 8 \
  --mem-interval 5 --mem-peak-mb 0 \
  --mem-dump-dir Accessory/data/s8_mem_L40_2026-10-09/clifford \
  --report verification_report/av1_smoke_L40_clifford_$(date +%F).md < /dev/null
```

## 6. 本轮产出物

- `verification_report/av1_smoke_L40_new5raw_2026-10-09.md`（15P/1F）
- `verification_report/av1_smoke_L40_maxbed_2026-10-09.md`（0P/2F，门禁速退）
- `verification_report/av1_vp9_matrix_L40_2026-10-09.md`（AC1 PASS）
- `verification_report/crfcq_gpu_L40_2026-10-09.md` + `CRF_CQ统一验证结果_20261009_072836.json`（113/0/3/0）
- 内存采样已**从 `/tmp` 迁入仓库**：`Accessory/data/s8_mem_L40_2026-10-09/{new5,warm,maxbed}/`
  + `README.md`（含格式说明与 S8 假 FAIL 的逐点证据表）。⚠ 原 `/tmp/s8_mem_*` 随容器重建会丢，已规避。
- 长视频（Clifford）：**无产出**（掉卡，前置门禁 exit 2）
- 未做：Step 5 跨仓 `verify_quality_mapping`（VidUtils 侧 ⑨ 组）

## 7. 与方案基线的净差异

| 项 | 方案基线 | 本轮实测 | 结论 |
|----|---------|---------|------|
| AV1 落表 CQ/QP | LOO 3.13 / 2.61 | 3.13 / 2.61 | 一致 ✅ |
| G7-6 | WARN −2.14 dB / 0.87×（待确认是否已修） | WARN −2.14 dB / 0.87× | **未修正，判据未达成** ❌ |
| AC1 `-qp 70` | 1.07× / −1.13 dB | 1.07× / −1.13 dB | 一致 ✅ |
| AV1 冒烟 S1~S8 | constqp S3 double-count；vbr 8/8 | 两臂 S1~S7 全通，帧数精确守恒（产物 1605 = 分段 1605，源 803） | 无 S3 差异 ✅ |

## 相关

- [[l40-verification-scope-2026-10]] — L40 只占 4 项的范围界定与两条证据强度警告
- [[verification-script-path-and-axis-traps]] — 素材池在仓库外 / 落表器轴后缀 / 静默假通过
- [[av1-nvenc-l40-calibration]] — 标定落表史与「须 constqp,vbr 对照」铁律
- [[l40-av1-pipeline-verification]] — ⚠ 其中 `:17` 的 `-cq 27` PASS 读数已过期（见 §2）
- [[feedback_closure_evidence_same_scope]] — 判 FAIL 闭环须同口径证据（本轮 S8 即照此执行）