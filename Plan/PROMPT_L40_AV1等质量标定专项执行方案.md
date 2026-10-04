# Video_Enhancement L40 专项执行方案 —— av1_nvenc 等质量标定（M4·L40 侧，**只针对 AV1**）

> **本方案只做 `av1_nvenc`**。共享方法论 / harness 改动 / 优秀做法 / 验收门禁的
> **权威定义在 T4 母版**：`Plan/PROMPT_T4_NVENC等质量标定专项执行方案.md`（下称「T4 方案」）。
> 本文件只写 **AV1 差异 + 执行步骤**，共享部分引用，避免两份实现漂移。
> ⚠ L40 虽是 Ada 卡、h264/hevc NVENC 也可用，但**本仓 h264/hevc 基线在 T4 标定**
> ⇒ 本专项**不重复标 h264/hevc**（重复标会引入第二套值）。
>
> **上位文档**：`Plan/PROMPT_等质量换算立项.md`（§0.0 / §7.1 B 组）
> **姊妹方案**：`Plan/Video_Enhancement_质量控制参数修复方案.md`（§7 AC1~AC7 / §8 L40 收口 / §9.6）
> **总览指南**：`Accessory/docs/EQQ_CALIBRATION_OVERVIEW.md`
>
> 状态（2026-10-04 **执行完毕并落表；S8 遗留项的无 GPU 准备已完成**）：本容器已具备 L40（Ada）硬件，
> §1~§7 全部走完。**唯一遗留 = S8（constqp 内存斜率）**，其定位工具已就绪，待下次上机读结论。
>
> **落表结果（已提交 `b64c1d2` / `20b9e81`）**
> - `QUALITY_MAP['av1_nvenc'] = (1.4566, 1.2165, 0, 63)`（VE 规范化池 `points/gpu_l40_cq`，17 素材；
>   crf21 → `-cq:v 32`）。⚠ 原并行会话/VU 值 `(1.4573, 1.1022)` 无法由任何 VE 池复现，已改；
>   **VE↔VU 该行分叉** ⇒ 已发 CR-4 handoff 请 VU 同步（见 §12 / `VidUtils/Plan/CR-4_av1_QP轴_handoff_VE_to_VU_20261004.md`）。
> - `QUALITY_MAP_QP['av1_nvenc'] = (7.9338, -97.5136, 0, 255)`（LOO[0,27] 2.61）。
>   ⚠ 实测为**仿射**，**取代**早期只在 ref21 验证的 `_QP_MAP_OVERRIDE` ×3；QP 期望 ref21 → **70**（非 63/71）。
> - 门禁同步：G3-7 CQ32→QP70 / G6-7,8 `-qp 70` / G6-9,10 `-cq 32` / G1-2 av1 `-cq:v 32`（G3-9 size 仍 63）。
>
> **验收（已提交 `95a3f75`）**
> - `crf_cq --quick --no-gpu` **104/0/0/11**；`--gpu` **113/0/3**；`plan_implementation_gate` **94/0/2**；
> - AC1 探针：表值 **`-qp 70` 落带内 PASS**（1.07× / −1.13 dB）——「×3(63)」结论已被仿射表取代；
> - ⚠ 工具口径：`av1_vp9_quality_matrix` 退出码仍为 1（扫描点 21/84/105 属**有意带外**被计入 `n_fail`），
>   与 §5.4「退出码 0」不一致，属文档/工具口径问题，未改判据。
>
> **门禁口径修订（本次新增，影响全部编码器）**
> - `eqq_pool_fit_table` 新增 `GATE_ANCHORS = [0,27]`（生产工作区间）：判据只看 `LOO[0,27]`，
>   全锚点 worst 降级为**监控列**（不计 FAIL）。
> - 依据：全 8 档实测 worst **对每个编码器都来自最高锚点 crf30**（逐锚点跨素材最差 0.94/1.93/2.91/3.25/**6.41**），
>   属门禁边界锚点等权的系统偏差，非换算缺陷。av1 CQ 判据 LOO 由 6.21 → **3.13 ✅**。
> - 前提：生产 `crf_ref` 不用 >27（仓主确认）；该前提改变须恢复全区间或改稳健统计量。
>
> **L40-5 长视频冒烟（358.8s 真实素材）**
> - `[FIX-S3-STAGE-DEDUP]`：两阶段管线各校验一次同批分段致 S3 假 FAIL（脚本 double-count）；
>   修复后复跑 **S3 ✅ 17926 = 产物帧数**；S1/S2/S4/S5/S6/S7 全过。
> - ❌ **S8（constqp）复现 FAIL**：空闲机斜率 **+149.5 MB/min**（vbr −21.4 通过）⇒ 非并发污染，
>   疑 constqp 路径真实增长。`MemWatcher` 为进程树 RSS 求和且未落盘进程数 ⇒ 待增强采样定位。
>   **与本专项换算正确性无关**（未改管线代码），作为**遗留项**单列。
>
> **S8 的无 GPU 准备项（2026-10-04 已完成，静态审阅 + 采样增强）**
> - **静态审阅（排除性结论）**：① AV1 走 **SDK 直通**，ffmpeg 只做 muxer ⇒ CLI 的
>   `-rc-lookahead`/`-b:v 0` 差异**排除**为累积源；② 两次跑的真正差异是
>   **两条不同代码路径**（constqp→LA=0→`encode_frames_batch_ce_pipeline` per-batch；
>   vbr→LA=8→`encode_frames_stream` 分块累积），**不是同一 encoder 的两种 RC 配置**；
>   ③ `_strm_slot_pending`/`_slot_pending`/`_cached_sps_pps`/`results`/`_slots`/`_la_pinned_pool`
>   **均确认有界**；④ AV1/HEVC **每段强制新建编码器** ⇒ 跨段累积结构性排除；
>   ⑤ 头号候选（per-frame CUDA event 在 `cuEventSynchronize` 失败 raise 前跳过销毁）
>   **被 rc=0 证伪**（该 raise 会置编码线程 error ⇒ rc≠0，而实测 S1 rc=0 ⇒ 从未触发）。
>   ⚠ **没有任何一条能正面解释 +150 MB/min** —— 下次上机的价值是用增强采样**直接定位归属**，
>   不是继续静态猜。详见 `memory/av1-nvenc-l40-calibration.md`。
> - **采样增强已落地**（`Accessory/verify/av1_pipeline_smoke.py`）：
>   `[FIX-S8-ATTRIB]` 保留并落盘**进程数 `n`** + **主进程/子进程分组 RSS 与 PSS** +
>   逐进程明细（`--mem-dump-dir` → `<rate_mode>.mem.tsv`，可离线重分析）；
>   显存改 `--query-compute-apps=pid` **按 pid 归属**（整卡值在共享主机上会污染）。
>   `[FIX-S8-CRITERIA]` 判据 = **斜率（口径不变，与 +149.5/−21.4 可比）+ 峰值上界
>   （`--mem-peak-mb` 默认 12000）+ 样本充分性（`--mem-min-samples` 默认 8，不足报 SKIP）**；
>   斜率超阈值时 detail 直接附主/子分组斜率。
> - **CPU 自测已过**：合成泄漏子进程被检出 +1887.9 MB/min 且分组归因精确
>   （`main +1888.0 / child −0.1`）；静止进程组 0.36 MB/min（噪声量级）⇒ 无误报；
>   `plan_implementation_gate` 84 项 **0 失败**。
>   ⚠ 唯一未被 CPU 覆盖的点：`--query-compute-apps` 在本容器无 GPU 时返回空（显存记 0），
>   **显存归属判定须在 L40 复验**。
> - **下次上机的一条命令**（读 detail 即得归属，无需二次跑）：
>   ```bash
>   python3 Accessory/verify/av1_pipeline_smoke.py --src <330s+真实素材> \
>       --rate-modes constqp --mem-interval 5 --mem-dump-dir /tmp/s8_mem < /dev/null
>   ```
>
> **未做（本方案范围外/条件未触发）**：AC5（av1_qsv/amf，需 Intel/AMD）；L40-6（AV1 Level 1 `code=12`，可选）；
> AV1 `-tune/-multipass` 同 T4 走显式 opt-in（AV1 本就 plain `vbr`，无附加项）。

---

## 0. 一句话范围

在 L40（Ada）上补齐 **`av1_nvenc` 的等质量标定**两条轴，并复验 AV1 端到端能力：

- **CQ 轴**（`-cq:v`，量程 **0~63**，rate control 用 **`vbr`**）→ 落 `QUALITY_MAP['av1_nvenc']`；
- **QP 轴**（`-qp`，量程 **0~255**，与 CQ 非同刻度；L40 实测为**仿射** `(7.9338, −97.5136)`，**非** ×3）→ 落 `QUALITY_MAP_QP['av1_nvenc']`（D2b）；
- 复验 AC1（QP 尺度）/ AC2（G7-6）/ AC7（软编族，已闭环）与 **AV1 长视频冒烟 S1/S2/S3/S8**；
- 可选：查 AV1 Level 1 直通失败的 `GetEncodePresetConfig code=12`（§9.6）。

---

## 1. 环境体检（Gate 0，最先做）

```bash
cd /workspace/Video_Enhancement
nvidia-smi -L                                                       # 期望 NVIDIA L40（Ada / sm89）
python3 -c "import torch;print(torch.cuda.get_device_name(0),torch.version.cuda,torch.cuda.is_available())"
ffmpeg -hide_banner -encoders | grep -E 'av1_nvenc|av1_qsv|av1_amf'
ffmpeg -hide_banner -version | head -1

# ① 构建里有没有 av1_nvenc（构建层）
ffmpeg -hide_banner -encoders | grep av1_nvenc
# ② 硬件编不编得动：唯一可靠判据是【实跑一帧】（-h encoder=av1_nvenc 在 Turing 上照样打印选项表）
ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 \
       -c:v av1_nvenc -f null - < /dev/null ; echo "rc=$?"
# rc=0 ⇒ 可执行本方案全部项；rc≠0（No capable devices found）⇒ 环境不成立，停止
```

> ⚠ `av1_qsv` 常在构建里但实跑失败（`Error creating a MFX session: -9`）；
> `av1_amf` 通常不在构建。两者**均不在本专项范围**（需 Intel/AMD 硬件，见方案 §8.4 / AC5）。

---

## 2. 待办任务清单（L40 侧，AV1 专项）

| ID | 任务 | 轴 | 产出 | 依据 |
|---|---|---|---|---|
| **L40-1** | `av1_nvenc` CQ 等质量标定（`-rc:v vbr -cq:v`） | `-cq:v` | `QUALITY_MAP['av1_nvenc']` | 立项 §7.1 B |
| **L40-2** | `av1_nvenc` QP 等质量标定（`-rc:v constqp -qp`） | `-qp` | `QUALITY_MAP_QP['av1_nvenc']` | D2b |
| **L40-3** | AC1 复验：QP 尺度 ×3（扫 21 / 63 / 84 / 105） | `-qp` | 报告（已闭环，复验） | 方案 §7 AC1 |
| **L40-4** | AC2：G7-6 `av1_nvenc -cq` 等质量 | `-cq:v` | 报告 | 方案 §7 AC2 / §8.3 |
| **L40-5** | AV1 长视频冒烟 S1/S2/S3/S8（`av1_pipeline_smoke.py` 完整跑批） | — | 报告 | 方案 §9.4 / §8.5 |
| **L40-6** | （可选）AV1 Level 1 `GetEncodePresetConfig code=12` 根因 | — | 结论 | 方案 §9.6 / P3″ |
| **L40-7** | harness / `_qp_model` 共享改动复用 | — | 代码 | T4 方案 §4 |
| **L40-8** | 跨仓 `QUALITY_MAP` 同步 + ⑨ 组 | — | 门禁 | T4 方案 §6 |

> T4 侧清单（h264/hevc）见 T4 方案 §2。**两卡不互替**：T4 无 AV1 NVENC。

---

## 3. 优秀做法吸收（AV1 专属，通用 17 条见 T4 方案 §3）

| # | 做法 | 出处 | 落地 |
|---|---|---|---|
| A1 | 硬件能力只认「实跑一帧」，不认 `-h encoder` | 方案 §7 AC0 / §8.1 | §1 |
| A2 | `av1_nvenc` 的 `-cq` 量程是 **0~63**（不是 51），`-qp` 是 **0~255** | 方案 E0 / §6.11.2 | §4.1 |
| A3 | **AV1 的 `-rc` 只接受 `constqp/vbr/cbr`** ⇒ `vbr_hq/qvbr` 必须降级为 `vbr` | 方案 §8.6-③ | §4.1 / harness 锁定 |
| A4 | AV1 的 `-qp` 与 `-cq` 是两条刻度；L40 实测 QP 轴为**仿射**（×3 仅 ref21 近似，crf≥24 起偏离） | 方案 §7 AC1 | §4.2；已落 `QUALITY_MAP_QP` |
| A5 | AV1 **Level 1 SDK 直通恒失败**（`GetEncodePresetConfig code=12`）⇒ 实际走 ffmpeg CLI | 方案 §8.6 | 标定的 CLI 口径即生产实际口径 |
| A6 | AV1 长视频色度检查（检查 4）**内容相关假阳性** ⇒ 验收加 `--skip-chroma` | 方案 §8.5 | §5.4 |
| A7 | `verify_video_integrity` 的 cv2→ffmpeg 回退（OpenCV 无 AV1 解码） | 方案 §8.6-① / §9.3 | 若冒烟失败先查是否回退未生效 |
| A8 | 判据/探针里**硬编码的期望值**要跟随上游表改动（A15：写死 84） | 方案 §8.2 / A15 | 探针已改 `av1_expected_qp()` 现场推导 |
| A9 | 命令形状断言（G6-8/9/10）+ 反向验证 | 方案 §9.2 | 改 AV1 下发前先跑 |
| A10 | AC7 软编族口径（`libsvtav1` 用 `-preset 8`、`libaom-av1` 用 `-cpu-used 6`） | 报告 §1.4 | 复跑 AC7 时锁定 |

---

## 4. AV1 差异（相对于 T4 方案的共享改动）

> 共享改动（`SWEEP`/`BASE_LOCK`/`QUALITY_FLAG`/`--axis`/`_qp_model` 模式感知/落表器
> `TIERS`+`GATE`+`tag`）**以 T4 方案 §4 为准**。以下是 AV1 的差异点。

### 4.1 AV1 的档位定义（T4 方案 §4.1 中已含，此处强调）

```python
SWEEP['av1_nvenc']        = [12, 18, 23, 27, 31, 36, 41, 47, 54, 63]   # -cq 量程 0~63
BASE_LOCK['av1_nvenc']    = ['-rc:v', 'vbr', '-b:v', '0', '-preset', 'p4']  # CR-2：显式 vbr
QUALITY_FLAG['av1_nvenc'] = '-cq:v'
HW_CODECS 含 av1_nvenc
```

> ℹ **CR-2（rate control）口径**：av1 统一**显式** `-rc:v vbr`（不加 HQ 附加项）
> （VE 生产 writer 的 av1 降级路径本就是 `vbr`；VU 已把 av1 也改为显式 `-rc vbr`）。
> ⚠ 2026-10-04 起 FFmpeg 9.0 **移除 `vbr_hq`/`qvbr`**（`-rc` 只剩 constqp/vbr/cbr）⇒
>   h264/hevc 的 CLI/harness 口径改为**裸 `vbr`**（`-tune`/`-multipass` 改显式 opt-in；VE SDK 侧仍走
>   `RC_VBR_HQ(32)`，实测驱动仍接受）。**AV1（本专项唯一目标）不受影响**——它本就是 plain `vbr`。
> 探针侧 VE 已修（`av1_vp9_quality_matrix.py` 的 `_PROD_RC`）；VU 侧 h264/hevc 待重新同步（T4 方案 §12.3）。

QP 轴由 `--axis qp` 切到 `-rc:v constqp -qp`，量程 `(0, 255)`（T4 方案 §4.2 的量程分支已含 AV1）。

### 4.2 AV1 的 QP 尺度与既有 override（**2026-10-04 结果已出**）

标定前的回退值（`src/utils/quality_map.py`）：

```python
_QP_MAP_OVERRIDE['av1_nvenc'] = (3.0, 0.0, 0, 255)   # ×3：仅 ref21 附近验证过的近似
```

- **等质量（quality）口径**：标定后由 **`QUALITY_MAP_QP['av1_nvenc'] = (7.9338, −97.5136, 0, 255)`
  优先命中**（仿射，**非 ×3**）。实测 ×3 在 crf24/27/30 残差 −20/−37/−51 ⇒ 已取代。
  ref21 → QP **70**（`to_constqp_qp(32)`）；G3-7/G6-7/8 期望随之更新。
- **等体积（size）口径**：保持 `_QP_MAP_OVERRIDE` 的 ×3（G3-9 期望 av1 CQ27→QP63 不变）。
- **两条轴不是同一刻度**：CQ 0~63 / QP 0~255；且 **CQ 行的 `b` 会经 `to_x264_crf` 往返影响 QP 期望**
  （b=1.1022→71、b=1.2165→70）。

### 4.3 AV1 的锚点/轴映射（标定的物理含义）

| 轴 | ffmpeg 参数 | rate control | 生产对应路径 |
|---|---|---|---|
| CQ | `-cq:v <0~63>` | `-rc:v vbr` | ffmpeg CLI writer（Level 2/3）+ SDK vbr 分支 |
| QP | `-qp <0~255>` | `-rc:v constqp` | ffmpeg CLI constqp + SDK Level 1（若未来恢复直通） |

⇒ **两条轴都要标**，`QUALITY_MAP` 与 `QUALITY_MAP_QP` 各一行。

---

## 5. 执行步骤（L40）

### 5.1 基线（改动前后对照）

```bash
python3 Accessory/verify/crf_cq_unification_verify.py --quick < /dev/null   # PASS=104 / FAIL=0 / SKIP=11（quality 口径，2026-10-04 B1）
python3 Accessory/verify/plan_implementation_gate.py < /dev/null            # FAIL=0
python3 Accessory/probe/calibrate_equal_quality.py --selftest               # 39 项（含 NVENC/axis/跨仓）
python3 Accessory/probe/av1_vp9_quality_matrix.py --selftest 2>/dev/null || true
```

### 5.2 素材（真实切片，覆盖 6 类，≤7 条，AV1 编码慢需控量）

同 T4 方案 §5.2 的 7 条（`--src-is-prep`）。⚠ AV1 **软件**编码慢的是 `libaom-av1`/`librav1e`，
本专项只用 `av1_nvenc`（硬件），编码快；成本主要在 VMAF（`n_subsample=1`，20~45 s/点）。

### 5.3 标定（AV1 两条轴）

```bash
# ── CQ 轴 ──────────────────────────────────────────────────────
for M in live_kids_play tv_bbc_s01e01 cganim_edu_wordworld anim2d_subs_tobot \
         doc_dark_earth screen_ui_code live_texture_frog; do
  python3 Accessory/probe/eqq_calibrate_clip.py \
      --src input_videos/eqq_calib/<side>/${M}_src*.mp4 \
      --out temp/eqq_gpu_l40/cq_${M} --tiers av1_nvenc \
      --duration <6|10> --src-is-prep < /dev/null
done
# ── QP 轴 ──────────────────────────────────────────────────────
for M in <同上 7 条>; do
  python3 Accessory/probe/eqq_calibrate_clip.py \
      --src input_videos/eqq_calib/<side>/${M}_src*.mp4 \
      --out temp/eqq_gpu_l40/qp_${M} --tiers av1_nvenc --axis qp \
      --duration <6|10> --src-is-prep < /dev/null
done
```

### 5.4 AV1 端到端复验（AC1/AC2/AC4 + 冒烟）

```bash
# ── AC1/AC2/AC4 一条命令（自动判读，口径与判据同源）──────────
python3 Accessory/probe/av1_vp9_quality_matrix.py \
    --src /workspace/input_videos/word_world_2.mp4 \
    --only av1_nvenc \
    --report verification_report/av1_vp9_matrix_L40_$(date +%F).md \
    --json   verification_report/av1_vp9_matrix_L40_$(date +%F).json < /dev/null
# 判据：AC1 表值 -qp 落 RATE_PASS=(0.65,1.50) 且 ΔPSNR ≥ -1.5 dB；退出码 0

# ── G7/G8 GPU 判据（AC2 = G7-6 转正）─────────────────────────
python3 Accessory/verify/crf_cq_unification_verify.py --gpu \
    --source /workspace/input_videos/word_world_2.mp4 \
    --bitrate-source /workspace/input_videos/new4_raw.mp4 \
    --report verification_report/crfcq_gpu_L40_$(date +%F).md < /dev/null

# ── AV1 长视频冒烟（P3 完整跑批，S1/S2/S3/S8 需 GPU）─────────
python3 Accessory/verify/av1_pipeline_smoke.py --src <330s真实素材> \
    --rate-modes constqp,vbr --segment-duration 30 \
    --mem-interval 5 --mem-dump-dir /tmp/s8_mem \
    --report verification_report/av1_smoke_L40_$(date +%F).md < /dev/null
# 退出码：0 无 FAIL / 1 有 FAIL / 2 环境前置不成立
# 采样判据：S1 退出码 / S2 段级 decoded==expected / S3 产物帧数=各段之和
#          / S4 解码级验证 / S5 segment_bitstream_verify_v5 --skip-chroma
#          / S6 QA sidecar / S7 产物编码器确为 av1
#          / S8 泄漏：后半程 RSS 斜率 ≤ +50 MB/min【口径未变】+ 峰值 ≤ --mem-peak-mb
#             （默认 12000）+ 采样点 ≥ --mem-min-samples（默认 8，不足报 SKIP）
#          S8 判读：detail 里的「主 x / 子 y MB/min」直接给出泄漏归属
#                 （main ⇒ 主进程；ffmpeg ⇒ 读帧器/分段 muxer）；PSS 同步涨=真泄漏
#          逐进程明细：/tmp/s8_mem/<rate_mode>.mem.tsv（可离线重算，不必二次上机）
```

### 5.5 入池 → LOO → 落表候选

```bash
python3 Accessory/probe/eqq_pool_fit_table.py \
    --sides 6s,10s,legacy10s,gpu_l40 --out /tmp/table_l40.txt < /dev/null
# 判据：av1_nvenc 行 LOO ≤5.9；顺序无关断言通过
```

### 5.6 （可选）L40-6 · AV1 Level 1 `code=12`

按方案 §9.6 的复现片段（`GetEncodePresetConfigEx` 回退），
⚠ 必须先 `import torch; torch.cuda.init()` 再加载 CUDA 库。
判读：`GPCEx -> 0` 且能 `InitializeEncoder` ⇒ 加回退（两侧 `nvenc_sdk.py`）；
否则记为驱动侧限制并注释「AV1 恒走 Level 2/3」。

---

## 6. 落表与跨仓同步

| 表 | 键 | L40 新增 | 同步 |
|---|---|---|---|
| `QUALITY_MAP`（等质量） | `av1_nvenc` | CQ 轴标定值 `(a, b, 0, 63)` | **两仓逐条相等**（⑨ 组）；VidUtils 侧须同步 |
| `QUALITY_MAP_QP`（QP 轴，仅本仓） | `av1_nvenc` | QP 轴标定值 `(a, b, 0, 255)` | 仅 VE |

⚠ 写入后 `QUALITY_MAP['av1_nvenc']` 的 hi **必须是 63**（不是 51）——`SIZE_MAP` 已如此，
等质量表须保持一致，否则 `crf_ref≥45` 被挤到 51（方案 E0 的老 bug）。

> ℹ **无损语义（2026-10-04 B1/`[FIX-QP-LOSSLESS]`）**：`to_constqp_qp(codec, 0)` 在 **size/quality
> 两口径均返回 0**（对 `value==0` 短路）。故 L40 落 `QUALITY_MAP_QP['av1_nvenc']` 后，AV1 的
> `-qp 0` 仍是 0（无损/最高质档），不被标定表的 `a/b` 外推；G3-4 已锁双口径、G3-9 锁 size 对照。
> 生产无损另由 writer `crf==0` 分支硬编码（G6-18/19）。
> ⚠ 该口径分流**不影响 AV1 的 AC1 结论**（两口径 `-qp` 均 63）。

---

## 7. 验收门禁

| 门 | 命令 | 判据 |
|---|---|---|
| 落表器 | `eqq_pool_fit_table.py --sides …,gpu_l40` | av1 行 LOO ≤5.9 + 顺序无关 ✅ |
| AC1/AC2/AC4 探针 | `av1_vp9_quality_matrix.py --only av1_nvenc` | 退出码 0；AC1 表值落带内 |
| GPU 判据 | `crf_cq_unification_verify.py --gpu …` | G7-6 PASS、FAIL=0 |
| AV1 冒烟 | `av1_pipeline_smoke.py` | 退出码 0；S1~S8 无 FAIL |
| 本仓判据/门禁/pytest | 同 T4 方案 §7 | FAIL=0 / 全绿 |
| 跨仓真源 | `VidUtils/verify/verify_quality_mapping.py` | ⑨ 组 14/14 |

---

## 8. 回滚

| 触发 | 动作 |
|---|---|
| av1 表 LOO 超门禁 | 不落表；保留 `gpu_l40` points 与报告，标注根因 |
| `QUALITY_MAP_QP['av1_nvenc']` 与 ×3 冲突 | 以实测为准；若证据不足则维持 `_QP_MAP_OVERRIDE` 的 ×3 并回滚新行 |
| 冒烟 S 项 FAIL | 先查 §8.6 三处 AV1 修复是否在位（②③ 命令形状 / ① cv2 回退）；再查编码线程 |
| Level 1 回退无收益 | 保持现状（Level 2/3 功能正确），仅注释说明 |

---

## 9. 风险与坑

| 风险 | 对策 |
|---|---|
| **`-rc:v vbr_hq/qvbr` 对 av1 非法** | 锁定 `vbr`（A3 / §4.1） |
| **AV1 Level 1 恒降级** | 标定走 CLI 口径即生产实际口径（A5）；勿假设 SDK 直通 |
| **色度检查假阳性** | 验收加 `--skip-chroma`（A6） |
| **cv2 无 AV1 解码** | 若产物被判损坏，查 `[FIX-AV1-CV2]` 回退是否生效（A7） |
| **探针硬编码期望值过期** | 已改现场推导 `av1_expected_qp()`（A8），勿再写死 |
| **L40 会话中途 GPU 被回收** | 每步当场复跑；同 T4 方案 P17 |
| **同机并发 NVENC** | `--jobs 1`；先查 `nvidia-smi` 与他人任务 |
| **误把 av1_qsv/amf 当 AV1 NVENC** | §1 已区分；AC5 需 Intel/AMD 硬件，本专项不含 |

---

## 9.5 下次上机待办（T4 / L40 通用，按优先级）

> 背景：§1~§7 已收口，**唯一遗留是 S8**（constqp 后半程 RSS 斜率 +149.5 MB/min，
> vbr −21.4 通过，两次复现）。定位工具已就绪（见 §0「S8 的无 GPU 准备项」），下机只需跑一条命令读结论。

| # | 待办 | 卡在哪 | 上机怎么做 | 完成判据 |
|---|---|---|---|---|
| **A** | **S8 定位（最高优先）** | 需真实 NVENC 负载 | 见下方「A 的命令」 | S8 detail 给出 `主 x / 子 y MB/min` 归属，读数即结论；**无需二次跑** |
| **B** | **显存归属口径复验** | 本容器无 GPU，`--query-compute-apps` 返回空（显存恒 0） | 随 A 的同一次跑批自动覆盖 | `/tmp/s8_mem/*.mem.tsv` 的 `gpu_tree_mib` 列**非 0** 且与 `nvidia-smi` 整卡值可对账 |
| **C** | **S8 判据峰值上界校准** | 默认 12000 MB 是**估值**，非实测推导 | 读 A 的 `RSS 峰值`，若 vbr 与 constqp 峰值差 >2× 则据实调 `--mem-peak-mb` | 阈值有实测依据，或明确记为「宽裕上界·不敏感」 |
| **D** | **constqp 路径可在 T4 复现（降本验证）** | 未验证 | T4 上 `h264_nvenc`/`hevc_nvenc` + `--rate-mode-* constqp` 跑同一冒烟 | 若 T4 也出正斜率 ⇒ 与 AV1 无关，**修复可在 T4 开发验证**（成本远低于等 L40）；若不复现 ⇒ 回 L40 查 AV1 特有因素 |
| **E** | 顺带复核项（非阻塞） | — | ①`av1_vp9_quality_matrix` 退出码口径（§0 遗留的 1 vs 0）；②L40-6（AV1 Level 1 `code=12`）若仍想做 | 有结论或明确记为不做 |

**A 的命令**（一条跑完，读 `S8` 的 detail 即可）：

```bash
cd /workspace/Video_Enhancement
python3 Accessory/verify/av1_pipeline_smoke.py --src <≥330s 真实素材> \
    --rate-modes constqp,vbr --segment-duration 30 \
    --mem-interval 5 --mem-dump-dir /tmp/s8_mem \
    --report verification_report/av1_smoke_$(date +%F).md < /dev/null
```

**读数判读表**（直接对应 S8 detail 的「主 x / 子 y MB/min」）：

| 观测 | 结论 | 下一步 |
|---|---|---|
| `子` 为正、`主` ≈ 0 | **ffmpeg 子进程累积**（读帧器 / 分段 muxer），主进程无辜 | 查 reader 帧队列与分段 muxer 的 `_stderr_lines` |
| `主` 与 `子` 同为正、PSS 同步涨 | **主进程真泄漏**（不是共享页虚高） | 按 `/tmp/s8_mem/*.mem.tsv` 里 top RSS 的 pid 定位到具体对象 |
| `主` 正、**PSS 不涨** | 多为 CUDA 上下文 / 共享页虚高，非真泄漏 | 降级判据（考虑改用 PSS 作主判据），别急着改管线 |
| 两者都 ≈ 0 但**峰值超上界** | 是峰值台阶而非单调泄漏（如某段累积后释放） | 查段切换的清理路径（`main.py:1155-1157` 每段新建编码器） |

⚠ **只跑 `constqp` 不足以判读**：必须 `constqp,vbr` 同素材对照，才能区分「constqp 特有」
与「长跑本身的时间相关项」。

---

## 10. 复现命令汇总

```bash
cd /workspace/Video_Enhancement
# 0 体检（AV1 实跑一帧）
ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 -c:v av1_nvenc -f null - < /dev/null; echo rc=$?
# 1 标定（CQ 轴，单素材示例）
python3 Accessory/probe/eqq_calibrate_clip.py \
    --src input_videos/eqq_calib/6s/live_kids_play_src1280x720.mp4 \
    --out temp/eqq_gpu_l40/cq_live_kids_play --tiers av1_nvenc \
    --duration 6 --src-is-prep < /dev/null
# 2 入池落表
python3 Accessory/probe/eqq_pool_fit_table.py --sides 6s,10s,legacy10s,gpu_l40 < /dev/null
# 3 AV1 端到端复验
python3 Accessory/probe/av1_vp9_quality_matrix.py --src /workspace/input_videos/word_world_2.mp4 --only av1_nvenc < /dev/null
python3 Accessory/verify/crf_cq_unification_verify.py --gpu --source /workspace/input_videos/word_world_2.mp4 --bitrate-source /workspace/input_videos/new4_raw.mp4 < /dev/null
python3 Accessory/verify/av1_pipeline_smoke.py --src <330s素材> --rate-modes constqp,vbr \
    --mem-interval 5 --mem-dump-dir /tmp/s8_mem < /dev/null
# 4 门禁
python3 Accessory/verify/plan_implementation_gate.py < /dev/null
cd /workspace/VidUtils && python3 verify/verify_quality_mapping.py < /dev/null
```

---

## 11. AC 覆盖现状（进入本专项前请核对）

| AC | 内容 | 现状（方案 §8 收口） | 本专项动作 |
|---|---|---|---|
| AC1 | AV1 constqp QP 尺度 | ✅ 已闭环（L40：旧 ×3/63 已在 ref21 验证；**现为仿射表**，表值 `-qp 70` → 1.07× / −1.13 dB 落带内 PASS） | 已复核 |
| AC2 | AV1 `-cq` 等质量（G7-6） | ✅ PASS（+0.16 dB / 1.28×） | 复验（L40-4） |
| AC3 | AV1 constqp 命令形状（G6-7） | ✅ PASS（无需硬件） | — |
| AC4 | AV1 `-cq` B 组 | ✅ PASS（1.14× / −0.29 dB） | 复验 |
| AC5 | av1_qsv / av1_amf 量程 | ⏭️ SKIP（方法已证伪，需 Intel/AMD） | 不在本专项 |
| AC6 | 跨项目 C 组交叉印证 | ✅ 由 AC1 同源覆盖 | VidUtils 侧另跑 |
| AC7 | AV1/VP9 软编族 | ✅ 完成（T4 重编构建） | 不在本专项（无硬件依赖） |

> **但「换算正确 ≠ 管线能跑」**：AV1 端到端能力由 P3 冒烟（§8.5）+ 三处修复（§8.6）保障，
> 本专项的 **L40-5** 是它首次在 GPU 上的完整回归。**S8 是唯一未闭环项**，定位工具已就绪，
> 待办与读数判读表见 **§9.5**。
>
> **2026-10-04 口径变更对 AC 的影响**：门禁口径已迁 **quality**（B1）且 `to_constqp_qp(0)=0` 双口径，
> 但 **AV1 的 `-cq`/`-qp` 数值在两口径相同**（`-qp 63`、`-cq` 表值不变）⇒ **AC1~AC4 的判据与期望值不变**，
> 仅"门禁口径 == 生产默认"这一形式更强。L40 上机前先按 §0「前置已就绪」核对。

---

## 12. VidUtils（VU）对等方案态势（**协同必备**）

> 完整契约见 **T4 方案 §12**；本节只列 AV1 相关的态势与协同点。

- **VU 侧有同构的 L40 方案**：`/workspace/VidUtils/Plan/VidUtils_等质量标定_L40_AV1专项执行方案.md`
  （其任务编号 `A0~A5` ↔ 本专项 `L40-1~L40-5`；VU `G3` ↔ 本专项 `L40-1`）。
- **VU harness 已实现 AV1/NVENC + `--expect-av1`**；VE harness 已移植（+ `--axis` 扩展）。
  两侧都用「实编探测 + fail-fast」，判据同为「rc==0 且产物非空」（T4 的 av1 会「列表里有、实编 -22」）。
- **AV1 的 `-cq` 等质量行写入两仓共享 `QUALITY_MAP`** ⇒ 受 **CR-1（preset 口径）** 与
  **CR-2（rate control 口径）** 约束。**CR-1 已收口：两仓统一 `p4`**（2026-10-04）——
  VE 本就 p4；**VU 已把生产 `DEFAULT_PRESET_GPU` / harness / 探针一并改 p4**（残留 `p5` 均为
  兼容显式 p5 的有意保留，见 T4 方案 §12.5）。
- **CR-2（rate-control 口径）**：av1 `vbr`，**均显式下发 `-rc`**。⚠ 2026-10-04 FFmpeg 9.0 移除
  `vbr_hq`/`qvbr` ⇒ h264/hevc 的 CLI/harness 口径从 `vbr_hq` 改为**裸 `vbr`**（2026-10-04 二次校正）
  （VE 已改；VE SDK 侧仍 `RC_VBR_HQ(32)`，实测驱动仍接受）。**本专项（AV1）不受影响**（本就 plain `vbr`）。
  VU 侧 h264/hevc 需重新同步，否则共享 `QUALITY_MAP` 的 ⑨ 组变红（handoff）。
- **QP 轴（`QUALITY_MAP_QP` / `_QP_SCALE`）**：VE 侧是独立表；VU 侧无表，只有 `_QP_SCALE`。
  本专项落 `QUALITY_MAP_QP['av1_nvenc']` 后，若与 VU 的 `_QP_SCALE=3` 冲突，**通知 VU 同步**（CR-4）。
- **无损契约 handoff（2026-10-04 实测）**：VU 自有 `to_constqp_qp(codec, 0)` **不保证 0**——
  NVENC 靠 `_QP_LIMITS` 夹回 0，但 **`librav1e`→48/52、`libsvtav1`→10/9**（表 b<0 使 ref>0）。
  VU 生产因 `_resolve_quality_params` 的 `[LOSSLESS]` 短路而未触发，但**函数级与「无损=0」不符**。
  VE 已加 `[FIX-QP-LOSSLESS]` 短路（两口径均 0）；**建议 VU 同步加 `if value==0: return 0`**。
- **跨仓态势已双向对称**：VE harness 现与 VU 一样会打印「两表是否相等 / 对侧 harness 是否同版 /
  对侧方案文档」。上机前先看这一行，再决定要不要协调对侧。
- **素材池共用**：17 条切片在 VE `input_videos/eqq_calib/`（仓库外）——L40 机上需先就位（CR-5）。
