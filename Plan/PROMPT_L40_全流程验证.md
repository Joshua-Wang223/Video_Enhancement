# L40 侧全流程验证立项

## 背景

CPU 侧与 T4 侧验证已收口，**唯一未覆盖的能力缺口是 AV1 NVENC 硬编全链路**（Turing 无 AV1 编码器，实跑报 `No capable devices found`）。因此本方案**只列必须在 L40/Ada(sm89) 上跑的项**；其余项在 CPU/T4 侧已完成，状态见 §「已在 CPU/T4 侧完成（不在本方案范围）」。

> **2026-10-14 修订**：逐项核对脚本代码后确认，除 §验收范围表中的 3 项外，**其余全部不依赖 L40/Ada**，
> 可在任意带 ffmpeg≥9.0 + 素材池的主机上执行。原「已在 CPU/T4 侧完成」表中 ❌/⚠ 标记的缺口项，
> 已整理为 §「CPU 侧待补跑清单」（含一键命令与逐项手工命令），切换主机后按清单落盘即可。
> 该章同时记录了 `comprehensive_verify.py` 两处过度门控（`nvenc_rc_diagnose` 被 `envs=[T4,L40]`
> + `required_gpu=True` 挡住，实际是纯主机侧工具）。

## 环境要求
- L40 / Ada (sm89) / 驱动 ≥535 / CUDA 12.x
- FFmpeg 9.0+ with av1_nvenc, h264_nvenc, hevc_nvenc
- PyTorch CUDA 可用
- 长视频素材（L40-1，**2026-10-09 修正**）：`/workspace/input_videos/大红狗 Clifford the Big Red Dog DVDR.58.mp4`
  （MP4/h264 720×576 25fps，18011 帧 / 720.5s，aac 48kHz；源结构门禁 rc=0 通过）
  - 原候选 `../input_videos/01 the race to mystery island fixed.avi` 仍可用（mpeg4 容器，885s），但需另建 `H576_W736` engine
  - ⛔ `112 Max Bed Time.avi` / `402 The Blue Tarantula .avi` **不可用**：音轨 mp3 头损坏，被 `[P5-FIX-SOURCE-STRUCT-GATE]` 硬拒
  - ⚠ 本项需 GPU 在场；掉卡时脚本前置门禁会以 `Cannot load libcuda.so.1` → exit 2 正确拒绝
- 模型权重就位

### 素材路径约定（2026-10-09 修正）

素材池在**仓库外**：`<项目父目录>/input_videos`（生产 Linux 与 WSL 同为 `/workspace/input_videos`）。
- 仓库内**没有** `input_videos/` 目录，写 `input_videos/xxx.mp4` 会因 cwd 是仓库根而找不到文件。
- 命令行一律写 **`../input_videos/xxx`**；脚本内默认素材目录由 `comprehensive_verify.py` 的 `INPUT_VIDEOS_CANDIDATES` 按 `PROJECT_ROOT.parent / "input_videos"` 优先探测，Windows 固定路径 `/mnt/d/Workspace_Python/input_videos` 仅作兜底。

---

## 验收范围（仅 L40 独占）

**判据共同前提**：AV1 NVENC 可用性必须用**实跑一帧**判定，不能用 `ffmpeg -h encoder=av1_nvenc`（Turing 上也会打印选项表而误报可用）。

| # | 测试项 | 脚本 | 为什么必须 L40 | 关键判据 |
|---|--------|------|----------------|----------|
| L40-1 | **AV1 长视频端到端冒烟 S1~S8** | `av1_pipeline_smoke.py --codec av1_nvenc` | 全链路唯一实跑 AV1 硬编的项；编排器 `envs=[L40]`，无其它环境可替代 | 帧守恒 / 解码级门禁 / 编码器确认 / S8 内存斜率 ≤50 MB/min |
| L40-2 | **crf_cq --gpu · G7-6** | `crf_cq_unification_verify.py --gpu` | G7-6 是 av1_nvenc `-cq` 表值判定，T4 侧恒 SKIP | G7-6 av1_nvenc `-cq` PASS；G7/G8 FAIL=0 |
| L40-3 | **AV1 质量矩阵 B 组 AC1** | `av1_vp9_quality_matrix.py --only av1_nvenc` | AC1 是 av1_nvenc `-qp` 落带扫描，T4 侧 B 组整体不执行 | 表值 QP 落带内；AC2 判据即 L40-2 的 G7-6；AC4 跨仓 B 组 |
| L40-4 | **NVENC 标定落表门禁** | `eqq_pool_fit_table.py` | 复核 L40 上采的 av1_nvenc 点数据落表 LOO（非 GPU 需求，但**只能有 L40 数据才能判**） | `--axis cq --sides gpu_l40_cq`：av1_nvenc LOO ≤5.9；`--axis qp --sides gpu_l40_qp`：LOO ≤5.9 |

### CPU / 主机侧可执行项（不占 L40，可切任意主机跑）

> 2026-10-14 逐项核对脚本代码确认：除上表 3 项（L40-1/2/3）外，**其余全部不依赖 L40/Ada**。
> 下表「证据」列为当前留痕状态；❌ = 仍缺报告文件，切换主机后按下 §「CPU 侧待补跑清单」落盘。

| 项 | 脚本 | GPU 依赖的代码判据 | 证据 |
|----|------|--------------------|------|
| `plan_gate` | `Accessory/verify/plan_implementation_gate.py` | 整体 CPU 可跑；无 GPU 时 R5/R7 降 WARN、R8/RT-0 降 SKIP，**无 FAIL** | ✅ 已有（89/101 项两份） |
| `nvenc_rc_diagnose` | `Accessory/probe/nvenc_rc_mode_diagnose.py` | **纯主机侧**：只 `ffmpeg -h encoder=h264_nvenc` + 磁盘搜 `nvEncodeAPI.h`，不建 NVENC session、不编码一帧；零 AV1 内容 | ✅ 有 V11 结论 |
| `verify_equal_quality` | `Accessory/verify/verify_equal_quality.py` | `envs=[cpu]`，纯 CPU 逻辑（ΔVMAF ≤1.0） | ❌ 仅 commit message |
| `crf_cq_cpu` | `Accessory/verify/crf_cq_unification_verify.py --quick --no-gpu` | 纯静态/纯函数断言 | ✅ 103 PASS / 12 SKIP |
| `av1_vp9_matrix_cpu` | `Accessory/probe/av1_vp9_quality_matrix.py --only <软编>` | A 组软编族；B 组 AC1 无 GPU 时 **SKIP**（非失败） | ❌ 缺 |
| `segment_bitstream_verify` | `Accessory/verify/segment_bitstream_verify_v5.py` | ffmpeg 侧通用，不绑定 codec；`--hwaccel` 可选默认关 | ❌ 缺 |
| `eqq_pool_fit_selftest` | `Accessory/probe/eqq_pool_fit_table.py --sides 6s,10s,legacy10s --axis cq` | 纯 JSON 拟合（软编 6 档） | ❌ 缺 |
| `calibrate_eq_selftest` | `Accessory/probe/calibrate_equal_quality.py --selftest` | 纯函数自检（39 项，秒级） | ❌ 缺 |
| `nvenc_tuning_verify` | `src/utils/nvenc_tuning.py` 参数表打印 | 纯函数自检（参数表唯一真源） | ❌ 缺 |

⚠ **编排器门控过宽（已知，非本方案引入）**：`comprehensive_verify.py` 对 `nvenc_rc_diagnose` 设了
`envs=[T4,L40]` + `required_gpu=True`，导致**无法在 CPU 队列里跑到它**；但它实测是纯主机侧工具，
切换任意主机后可直接手工执行（见下 §清单第 9 项）。L40-4 的 QP 轴子测试同理被钉在 `envs=[T4,L40]`。

---

## CPU 侧待补跑清单（切换主机后执行）

> 上表 ❌ 行 + 已知缺口的报告落盘。**全部不依赖 L40**；在任意有 ffmpeg≥9.0 + 素材池的主机上执行。
> 统一约定：所有脚本 `< /dev/null` 运行（否则后台进程组可能整组 SIGTTOU 挂起）。
> 素材池在仓库外 `<项目父目录>/input_videos`；命令行写 `../input_videos/xxx`。

### 一键跑 CPU 队列（推荐）

```bash
cd /workspace/Video_Enhancement
python3 Accessory/verify/comprehensive_verify.py --env cpu < /dev/null
# --only 可过滤；例：--only plan_gate,crf_cq_cpu,calibrate_eq_selftest
# ⚠ CPU 队列**不含** nvenc_rc_diagnose（被 envs=[T4,L40] 挡住），需手工补跑见下
```

### 逐项手工命令（含编排器未覆盖者）

| # | 项 | 命令 | 预计耗时 |
|---|----|------|---------|
| 1 | `plan_gate` | `python3 Accessory/verify/plan_implementation_gate.py` | ~1 min |
| 2 | `verify_equal_quality` | `python3 Accessory/verify/verify_equal_quality.py --src ../input_videos/word_world_2.mp4 --codecs libx265,libvpx-vp9,libsvtav1,libaom-av1,librav1e --duration 6` | ~9 min |
| 3 | `crf_cq_cpu` | `python3 Accessory/verify/crf_cq_unification_verify.py --quick --no-gpu` | ~5 s |
| 4 | `av1_vp9_matrix_cpu` | `python3 Accessory/probe/av1_vp9_quality_matrix.py --quality-mode quality --src ../input_videos/word_world_2.mp4 --only libvpx-vp9,libsvtav1,libaom-av1` | ~10 min |
| 5 | `segment_bitstream_verify` | `python3 Accessory/verify/segment_bitstream_verify_v5.py <产出视频> --skip-chroma` | ~秒级（**需先有一段已产出视频**） |
| 6 | `eqq_pool_fit_selftest` | `python3 Accessory/probe/eqq_pool_fit_table.py --sides 6s,10s,legacy10s --axis cq` | ~2 s |
| 7 | `calibrate_eq_selftest` | `python3 Accessory/probe/calibrate_equal_quality.py --selftest` | ~0.3 s |
| 8 | `nvenc_tuning_verify` | `python3 -c "import sys;sys.path.insert(0,'src/utils');from nvenc_tuning import NVENC_TUNE_VALUES,NVENC_MULTIPASS_VALUES,is_bitrate;print(NVENC_TUNE_VALUES);print(NVENC_MULTIPASS_VALUES);print(is_bitrate('vbr'),is_bitrate('vbr_hq'))"` | ~0.1 s |
| 9 | `nvenc_rc_diagnose`（编排器未覆盖） | `python3 Accessory/probe/nvenc_rc_mode_diagnose.py` | ~秒级 |
| 10 | **跨仓 ⑨ 组真源一致** | `cd /workspace/VidUtils && python3 Accessory/verify/verify_quality_mapping.py` | ~秒级 |

**第 5 项的素材来源**：本仓无预置产出视频。可用 `/workspace/test_materials_4k/` 下既有片段（如
`seg_hevc_la8_720p.mp4`，2560×1440 / 603 帧 / 10.05s，其 `.qa.json` 记录 `interpolate_then_upscale` +
IFRNet 2x + ESRGAN 4x 生成链），或先用短素材跑一次管线取得产出。

**第 10 项已知差异**（2026-10-09 实测，非本轮引入）：⑦ `[7] --threads` 显式值两边不一致
（VE 得 `'3'` / VU 得 `'2'`）⇒ 该组退出码非 0，需单独立项修 VU 侧 `vidcrop_hwaccel.py` 的 `--threads` 钳位。

**跑完后**：报告默认落在 `verification_report/`；把结果回填到本文件 §「已在 CPU/T4 侧完成」的
CPU 侧表格（把 ❌ 改为 ✅ + 报告路径 + 实测数字），并按 AGENTS.md 规则同步 `memory/` 双侧镜像。

---

## 执行命令

```bash
cd /workspace/Video_Enhancement

# 0. 环境体检
nvidia-smi -L
python3 -c "import torch;print(torch.cuda.get_device_name(0),torch.cuda.is_available())"
for C in h264_nvenc hevc_nvenc av1_nvenc; do
  ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 -c:v $C -f null - < /dev/null; echo "$C rc=$?"
done
# 期望: h264=0, hevc=0, av1=0   ← av1 rc≠0 即整批 L40 项无法执行，先停

# 1. [L40-1] AV1 长视频端到端冒烟（S1~S8）
#    ⚠ 2026-10-09 实测后修正：素材改用「大红狗 Clifford」，且需 GPU 在场。
#    前置已核验（无需 GPU）：源结构门禁 rc=0 通过；.trt_cache 已有
#    IFRNet_S_Vimeo90K_B8_H576_W736_sm89 与 realesr-general-x4v3_B8_C3_H576_W720_sm89
#    ⇒ 零 engine 构建，constqp 臂可直接拿到干净 S8。
#    两个旧候选 112 Max Bed Time.avi / 402 The Blue Tarantula.avi 音轨 mp3 头损坏，
#    会被 [P5-FIX-SOURCE-STRUCT-GATE] 硬拒（rc=1），勿再选用。
python3 Accessory/verify/av1_pipeline_smoke.py \
    --src "/workspace/input_videos/大红狗 Clifford the Big Red Dog DVDR.58.mp4" \
    --rate-modes constqp,vbr \
    --segment-duration 30 --batch-size 8 --mem-interval 5 --mem-peak-mb 0 \
    --mem-dump-dir Accessory/data/s8_mem_L40_2026-10-09/clifford \
    --report verification_report/av1_smoke_L40_clifford_$(date +%F).md < /dev/null
# 或走编排器（仅此一项；--only 过滤掉其余 11 个非 L40 子测试）
python3 Accessory/verify/comprehensive_verify.py --env l40 \
    --only av1_pipeline_smoke \
    --long-source "/workspace/input_videos/大红狗 Clifford the Big Red Dog DVDR.58.mp4" \
    --segment-duration 30 --mem-interval 5 --mem-peak-mb 0 \
    --mem-dump-dir Accessory/data/s8_mem_L40_2026-10-09/clifford < /dev/null

# 2. [L40-3] AV1 质量矩阵（仅 av1_nvenc）
python3 Accessory/probe/av1_vp9_quality_matrix.py --quality-mode quality \
    --src ../input_videos/word_world_2.mp4 --only av1_nvenc \
    --report verification_report/av1_vp9_matrix_L40_$(date +%F).md < /dev/null

# 3. [L40-2] GPU 画质判据
python3 Accessory/verify/crf_cq_unification_verify.py --gpu \
    --source ../input_videos/word_world_2.mp4 \
    --bitrate-source ../input_videos/new4_raw.mp4 \
    --report verification_report/crfcq_gpu_L40_$(date +%F).md < /dev/null

# 4. [L40-4] 落表门禁（CQ / QP 两轴点数据不可混池，必须分两次调用）
#    ⚠ 目录名带轴后缀：gpu_l40_cq / gpu_l40_qp。旧写法 `--sides gpu_l40` 会
#    匹配不到任何目录 → 三个 NVENC 档「拟合失败（样本 0 < 4）」但仍 exit 0（静默假通过）
python3 Accessory/probe/eqq_pool_fit_table.py --sides gpu_l40_cq --axis cq < /dev/null
python3 Accessory/probe/eqq_pool_fit_table.py --sides gpu_l40_qp --axis qp < /dev/null
# 期望: av1_nvenc CQ LOO ≤5.9 / QP LOO ≤5.9（注意 QP 档门限同为 5.9，
#       7.5 只放宽给 librav1e 族），顺序无关性 3 seed 逐位一致

# 5. 跨仓真源一致（非 L40 独占，但每次改动质量表后需复跑）
cd /workspace/VidUtils && python3 Accessory/verify/verify_quality_mapping.py < /dev/null
# ⚠ 脚本已于 2026-10-06 从 verify/ 迁至 Accessory/verify/（VU 提交 ebdad41），
#   旧路径 /workspace/VidUtils/verify/ 已空
# 期望: ⑨ 组 14/14 一致
# ⚠ 已知非 AV1 差异（2026-10-09 实测，不在本方案范围）:
#   ⑦ [7] --threads 显式值两边不一致（VE 得 '3' / VU 得 '2'）⇒ 该组退出码非 0，
#     需单独立项修 VU 侧 vidcrop_hwaccel.py 的 --threads 钳位
```

---

## 门禁基线（2026-10-06 L40 历史实测，⚠ 见下方证据强度说明）

### 2026-10-09 L40 实跑复核（本次唯一有可审计执行证据的一轮）

| # | 项 | 判据 | 实测 | 裁定 |
|---|---|------|------|------|
| L40-4 | 落表门禁 cq | av1_nvenc LOO ≤5.9 | **3.13** ✅ | 落表候选 `(1.4566,1.2165,0,63)` 与 `convert_crf.py:212` 逐位一致 |
| L40-4 | 落表门禁 qp | av1_nvenc LOO ≤5.9 | **2.61** ✅ | 与 `quality_map.py:202` `(7.9338,-97.5136,0,255)` 逐位一致 |
| L40-3 | AC1 | 表值 QP 落带内 | `-qp 70` **1.07× / −1.13 dB** ✅ | 与历史基线一致 |
| L40-2 | G7/G8 | FAIL=0 | **113 PASS / 0 FAIL / 3 WARN / 0 SKIP** ✅ | G7/G8 FAIL=0 达成 |
| L40-2 | G7-6 | av1_nvenc `-cq` **PASS** | **WARN −2.14 dB / 0.87×** ❌ | **判据未达成**：`-cq 27→32` 未修正，数值与历史基线完全相同 |
| L40-1 | S1~S8 | 全通 + S8 斜率 ≤50 MB/min | 两臂 S1~S7 全通；**S8 constqp +712 MB/min 已被证伪为测量假象** | 无内存泄漏（缓存引擎的 vbr 对照臂 **−928.5 MB/min**） |

**本轮证据文件**：`verification_report/{crfcq_gpu,av1_vp9_matrix,av1_smoke_L40_new5raw}_*_2026-10-09.md`；
内存采样明细 `Accessory/data/s8_mem_L40_2026-10-09/`（含 README 与逐点证据表）。

⚠ **两处历史读数已过期，勿再引用**：
- 本方案 §门禁基线 的「G7-6 历史记为 WARN（−2.14 dB / 0.87×）」——本轮复现同值，说明 `-cq 27→32` 未修正它；
- `memory/l40-av1-pipeline-verification.md:17` 记的 G7-6「`-cq:v 27` ΔPSNR **+0.16 dB** / 1.28× **PASS**」
  —— 那是 cq27 的读数，改 cq32 后变为 −2.14 dB WARN，**不可作为「已验收」依据**。

⚠ **L40-1 长视频（Clifford 720.5s）尚未执行**：选型与前置已核验（门禁 rc=0、engine 已缓存），
但开跑时掉卡，前置门禁 exit 2，**无产出**。恢复命令见本文件 §执行命令 1。

### 以下为 2026-10-06 历史基线（⚠ 见证据强度说明）

- `av1_vp9_quality_matrix`: AC1 `-qp 70` 落带内（1.07× / −1.13 dB）/ AC2 +0.16 dB / AC4 1.14× / −0.29 dB
- `av1_pipeline_smoke`: S1~S8 退出码 0；constqp 臂 S3 计数差异为脚本 double-count，非管线缺陷；vbr 臂 8/8 全 PASS，RSS 斜率 −2.8 MB/min
- `crf_cq --gpu`: 历史记为 111 PASS / 0 FAIL / 5 SKIP，其中 **G7-6 为 WARN（−2.14 dB / 0.87×）**——与 §「执行命令」要求的 PASS 不符，需在本轮实跑中确认是否已随 `-cq 27 → -cq 32` 修正
- 落表：`QUALITY_MAP['av1_nvenc']=(1.4566,1.2165,0,63)`（LOO[0,27] 3.13）、`QUALITY_MAP_QP['av1_nvenc']=(7.9338,-97.5136,0,255)`（LOO 2.61）

**证据强度警告**：`memory/av1-nvenc-l40-calibration.md` 末节已把「L40 标定测试已执行」降级为**仓内不可审计**（审计链 6 处断点：无 GPU 标定日志、MANIFEST 未收录 `gpu_l40_*`、workdir 不存在等）。上述基线数字只能作**历史参考**，本轮须以实跑报告为准，不得反向引用为「已验收」。

**S8 峰值基线注意事项**：`--mem-peak-mb` 默认 16000 MB 是按 **T4 / 720×576 / bs=24** 标定的（`memory/t4-s8-findings-and-blockers.md`）。L40 + `interpolate_then_upscale` 上分辨率下须按实际 batch-size 重标定，否则会假 FAIL。

---

## 已在 CPU/T4 侧完成（不在本方案范围）

> 核查日期 2026-10-09。证据强度已逐项标注，**⚠ 标记表示只有自述/commit message、无报告文件留痕**。

### plan_gate 记录更正（2026-10-10）

⚠ 本文件原写「`plan_gate` 86 PASS / 0 FAIL / 1 WARN / 2 SKIP（共 89 项），WARN=`BEH-ERR`(`_session_gen`)
已于 `d215fc2`（04:39）修复，该报告早于修复 12 分钟」——**该结论不成立，已更正**。

审核他人在制品时发现：**晚于 `d215fc2` 整整 1 小时的 05:42/05:45 报告仍复现同一 `BEH-ERR`**。
根因不是生产代码：`d215fc2` 的生产侧修复（`external/ifrnet_video/nvenc_sdk.py:760/2360-2362`）是对的，
但门禁桩 `plan_implementation_gate.py::make_enc()` 走 `NVENCEncoder.__new__()` **绕过 `__init__`**，
没补 `_session_gen` ⇒ `_stream_begin()` 抛 `AttributeError` ⇒ 被**组级** `BEH-ERR` 吞掉整组结果。

**后果**：`BEH-G9/G10/G11` 等 **12 项从未真正执行**，门禁长期只报 89 项（实为 101 项），
且总数变化无任何告警。`d215fc2` 自述的「plan_gate 86 PASS / 0 FAIL」因同一门禁自身失效而**不可采信**。

已修桩并复核（同命令）：**101 项 / 97 PASS / 0 FAIL / 2 WARN / 2 SKIP**，
`BEH-G9/G10/G11` 恢复执行且全部 PASS ⇒ 确认生产修复有效，缺陷仅在门禁桩。
剩余 2 WARN 为环境差异（R5 CUDA、R7 NVENC）。

### T4 侧（Tesla T4, SM75, 14.6 GiB）

| 项 | 结论 | 证据 |
|----|------|------|
| `plan_gate` 完整 | ⚠ **原记录有误，已更正（2026-10-10）** | 见下方「plan_gate 记录更正」 | 详见 `memory/gate-stub-session-gen-audit-2026-10-10.md` |
| `crf_cq_unification_verify --gpu` | ✅ 114 PASS / 0 FAIL / 2 SKIP | `verification_report/CRF_CQ统一验证报告_20261009_035735.md`。SKIP: `G7-6` av1_nvenc「No capable devices found」+ `G8-4*`。**注：AV1 相关项仍需 L40-2 复跑** |
| `segment_bitstream_verify_v5` | ✅ hevc+LA=8 / 720p：frames=packets=603，帧守恒 OK | 提交 `6ed9ceb` message |
| `nvenc_vbr_hq_verify`（V8~V15） | ✅ 裁定方案 A 并落地：驱动接受 `rc_ptr[1]=32`，三档字节互异（1076383 / 2159969 / 1018148）；ΔVMAF −0.048 / −0.130 | `memory/t4-vbrhq-verification-plan.md` |
| `nvenc_rc_diagnose` | ✅ V11：VBR_HQ 未在 SDK 头文件、FFmpeg9 已移除、`-rc` 仅 constqp/vbr/cbr | `Plan/T4_NVENC_vbr_hq移除_验证专项.md:180`；`memory/nvenc-rc-enum-illegal-vbr-hq.md` |
| h264/hevc 端到端冒烟（S1~S8 代理） | ✅ hevc 两臂 14 PASS / 2 FAIL；h264 constqp 7/2、vbr_hq **0/1**（B1 缺陷首现，已由 `d215fc2` 修） | `verification_report/s8_t4_*.md` + 原始采样 `s8_20261004_raw/*.mem.tsv` |
| NVENC 标定落表 | ✅ T4-1~T4-9 全绿：CQ h264 LOO 3.98 / hevc 5.81；QP h264 3.47 / hevc 3.72 | `memory/equal-quality-t4-nvenc-calibration.md`；点数据 `points/gpu_t4_{cq,qp}/` |
| **T4 跑不了 AV1（已实测确认）** | ⛔ 4 处独立证据：`av1_vp9_matrix_T4_20260929.md:16`（`av1_nvenc ⏭️ SKIP`「Cannot load libcuda.so.1」，B 组整体未执行）、`CRF_CQ统一验证报告_20261009_035735.md:158`、`Plan/PROMPT_T4_NVENC等质量标定专项执行方案.md:34`、`Plan/PROMPT_T4_全流程验证.md:34` | — |

### CPU 侧（7 项）

> **2026-10-14 状态**：下表 ❌/⚠ 项已全部整理进 §「CPU 侧待补跑清单」，**切换任意主机即可执行**
> （均不依赖 L40/Ada）。本轮 L40 容器无 GPU 可用（`libnvidia-ml.so` 缺失），仅先完成「能否不靠 L40 跑」
> 的代码级判定与命令整备，**未实跑**。`plan_gate` 一项例外——已在本容器修桩并实跑（101 项 / 97 PASS）。

| 项 | 结论 | 证据强度 |
|----|------|----------|
| `verify_equal_quality` | 537s，ΔVMAF ≤1.0，5/5 rc=0 | ❌ 仅 commit `9d15dc6` message + `memory/MEMORY.md:182` 一行摘要，**无报告文件/日志留痕**。索引指向的 `memory/cpu-verification-pass.md` **两侧镜像均不存在且 git 全历史从未有过**（悬空链接）。**待补跑**（清单第 2 项） |
| `crf_cq_cpu` | 5s，G1~G6/G10 PASS | ✅ 已有 3 份重复报告：`verification_report/CRF_CQ统一验证报告_20261009_0{54553,55613,62348}.md`，均为 103 PASS / 0 FAIL / 0 WARN / 12 SKIP（`--quick --no-gpu`） |
| `av1_vp9_matrix_cpu` | 582s，libvpx-vp9 / libsvtav1 / libaom-av1 三软编 PASS | ❌ 同上（无留痕）。**待补跑**（清单第 4 项） |
| `segment_bitstream_verify` | 2s，帧守恒/解码级 PASS | ❌ 同上（无留痕）。**待补跑**（清单第 5 项）；⚠ T4 侧另有一份 frames=packets=603 的 hevc 证据（提交 `6ed9ceb` message） |
| `eqq_pool_fit_selftest` | 1.7s，LOO/顺序无关性 PASS | ❌ 同上 + 提交 `6ed9ceb`。**待补跑**（清单第 6 项） |
| `calibrate_eq_selftest` | 0.3s，39 项 PASS | ❌ 同上。**待补跑**（清单第 7 项） |
| `nvenc_tuning_verify` | 0.1s，参数表 PASS | ❌ 同上。**待补跑**（清单第 8 项） |
| `plan_gate`（CPU 轮） | ✅ **101 项 / 97 PASS / 0 FAIL / 2 WARN / 2 SKIP** | `verification_report/verification_report_20261010_012342.md`（修桩后）；2 WARN 为环境差异 R5/R7 |
| `segment_bitstream`（CPU 轮） | 需 `-o` 已存在的输出视频 | 素材来源见清单第 5 项说明 |

**口径纠正**：原始表述为「CPU 侧 7/7 通过」，准确说法是 **7 项 PASS + plan_gate 预期跳过**（`MEMORY.md:182` 明写）。且该轮环境记为 WSL Ubuntu，与当前 Linux 容器不同。**「7/7」至今无报告留痕**，只有 commit message，不能作为已验收依据（同 [[feedback_artifacts_not_execution_evidence]]）。

### 跨仓一致性

| 项 | 结论 | 证据 |
|----|------|------|
| ⑨ 组 14/14 一致 | ⚠ 最近一次**实跑**记录在案为 2026-10-03（`memory/eqq-batch-measure-parallel-constraints.md:467`）；2026-10-09 的「14/14 一致」（`memory/implementation-verification-report-2026-10-09.md`）是**静态代码核对**而非实跑 | 两仓 `verification_report/` 与 `*.log` 内**均无 ⑨ 组运行原始输出** |
| ⑨ 组当前实跑状态 | ⚠ 2026-10-09 实测：**13 PASS + 1 FAIL**（⑦ `[7] --threads 显式值两边不一致`，VE 得 `'3'` / VU 得 `'2'`），该组退出码非 0 | 本轮实跑（脚本已迁至 `Accessory/verify/`） |

---

## 产出物

**L40 独占项（切回 L40 容器后）**
- `verification_report/av1_smoke_L40_clifford_YYYYMMDD.md` + `Accessory/data/s8_mem_L40_2026-10-09/clifford/*.mem.tsv`
- `verification_report/av1_vp9_matrix_L40_YYYYMMDD.md/.json`
- `verification_report/crfcq_gpu_L40_YYYYMMDD.md`
- 落表复核输出：`QUALITY_MAP['av1_nvenc']` / `QUALITY_MAP_QP['av1_nvenc']` 应与 `src/utils/convert_crf.py:212` / `src/utils/quality_map.py:202` 现值一致

**CPU 侧补跑项（切任意主机后）** —— 见 §「CPU 侧待补跑清单」第 1~10 项，报告落 `verification_report/`，
回填 §「已在 CPU/T4 侧完成」CPU 侧表格。

**本方案已收口的既有产出**
- ✅ L40-4 落表双轴：`Accessory/data/eqq_calibration/points/gpu_l40_{cq,qp}/` 17 素材，LOO 3.13 / 2.61
- ✅ L40-2 `crf_cq --gpu`：`verification_report/crfcq_gpu_L40_2026-10-09.md`（113 PASS / 0 FAIL / 3 WARN）
- ✅ L40-3 AC1：`verification_report/av1_vp9_matrix_L40_2026-10-09.md`（`-qp 70` 落带 1.07× / −1.13 dB）
- ✅ L40-1 部分（new5_raw 1080p→4K）：`verification_report/av1_smoke_L40_new5raw_2026-10-09.md`（15 PASS / 1 FAIL，S8 已证伪为 TRT 构建假 FAIL）
- ✅ S8 内存采样原始数据：`Accessory/data/s8_mem_L40_2026-10-09/`（含 README 与逐点证据表）
- ✅ 门禁桩 BEH-ERR 审计与修复：`memory/gate-stub-session-gen-audit-2026-10-10.md`

## 当前总体状态（2026-10-14）

| 类别 | 状态 |
|---|---|
| L40-1 长视频冒烟（Clifford 720.5s） | ⛔ **未执行** —— 掉卡，前置门禁 exit 2；素材/engine/命令已就绪，恢复命令见 §执行命令 第 1 项 |
| L40-1 短素材冒烟（new5_raw 1080p→4K） | ✅ S1~S7 全通；S8 假 FAIL 已证伪（无泄漏） |
| L40-2 crf_cq --gpu | ⚠ G7/G8 FAIL=0 达成；**G7-6 仍 WARN，`-cq 27→32` 未修正 ⇒ 判据未达成** |
| L40-3 AC1 | ✅ PASS |
| L40-4 落表双轴 | ✅ PASS（LOO 3.13 / 2.61） |
| CPU 侧 8 项 | 📋 **待补跑**（命令已整备，均不依赖 L40） |
| 跨仓 ⑨ 组 | ⚠ 13 PASS + 1 FAIL（`--threads` 差异，需单独立项修 VU 侧） |
