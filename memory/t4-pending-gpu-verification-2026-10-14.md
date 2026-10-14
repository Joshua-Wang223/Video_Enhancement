# T4 GPU 待测项目盘点 + 补测结果（2026-10-14）

> ## ✅ 补测已执行（2026-10-14 本机 T4，同日两批）
>
> **P0+P1 全部完成**（B1 / B2 / B4 / plan_gate 101 项 / hevc-la-drain baseline /
> esrgan reopen / LA=16）。**结果表见 §2.0**，其中：
> - **B1 判据全过**，但过程中**发现并修复 1 个 P0 生产缺陷**（ESRGan 侧 `_qp_val` 缺失 ⇒
>   vbr_hq 默认档静默降级 Level 2，见 [[esrgan-nvenc-qpval-l1-fallback-2026-10-14]]）；
> - **B4 裁定为「驱动接受 32/64」⇒ 不触发全套等质量标定重跑**；
> - **S8 内存斜率仍 FAIL**，根因已定位为 pinned result pool 自适应爬坡（**非无界泄漏**），留作 P1 遗留。
>
> 未做：P2 开发型（B2 方案 B、pytest SIGSEGV 隔离）、P3 疑似陈旧项核销（需用户裁定是否仍要）。

**盘点结论（一句话）**：T4 等质量标定专项（T4-1~T4-9）已收口 ✅，但**所有 T4 GPU 证据都停在 2026-10-04**，
早于最近两次 NVENC 改动 —— `ecd60f8`(2026-10-05，B2 分级拒绝 + RC 枚举探针) 与
`d215fc2`(2026-10-09，B1 跨段 SPS/PPS 缓存保留) ⇒ **当前代码在 T4 上零回归证据**。

## 0. 证据时间线（决定优先级的事实）

| 日期 | 事件 | 之后有 T4 实跑？ |
|---|---|---|
| 2026-10-04 | 最后一批 T4 GPU 实跑（`s8_t4_{h264_constqp,h264_vbrhq,hevc_ab}_20261004.*`） | — |
| 2026-10-05 | `ecd60f8` B2 分级拒绝 + `_log_ready` 回显 + RC 枚举探针转正 | ❌ 无 |
| 2026-10-09 | `d215fc2` B1 修复（跨段复用保留 `_cached_sps_pps`） | ❌ 无 |
| 2026-10-10~14 | plan_gate 修桩后 BEH-G9/G10/G11 恢复执行并 PASS（**桩级，非真机**） | ❌ 无 |

## 1. 已收口（无需 T4）

| 项 | 结论 | 证据 |
|---|------|------|
| 等质量标定专项 T4-1~T4-9 | ✅ 全绿（含 T4-8 h264/hevc 生产管线长视频冒烟） | `Plan/PROMPT_T4_NVENC等质量标定专项执行方案.md:12,66-74` |
| T4 落表双轴 | ✅ CQ h264 3.98/hevc 5.81、QP h264 3.47/hevc 3.72 | `memory/equal-quality-t4-nvenc-calibration.md` |
| `crf_cq --gpu` | ✅ 114 PASS / 0 FAIL / 2 SKIP（SKIP=AV1 硬件缺失，属预期） | `verification_report/CRF_CQ统一验证报告_20261009_035735.md` |
| `nvenc_vbr_hq_verify` V8~V15 | ✅ 方案 A（驱动接受 rc_ptr[1]=32，三档字节互异，ΔVMAF −0.048/−0.130） | `memory/t4-vbrhq-verification-plan.md` |
| `nvenc_rc_diagnose` | ✅ V11 | `Plan/T4_NVENC_vbr_hq移除_验证专项.md:180` |
| `segment_bitstream_verify_v5` hevc+LA=8 | ✅ frames=packets=603 | 提交 `6ed9ceb` message |
| B3 显存四态口径 | ✅ 落地（无 GPU 需求） | `t4-s8-findings-and-blockers.md` |
| B4 静态断言 | ✅ 已撤回（「枚举中不存在 32/64」不再成立） | `memory/nvenc-rc-enum-illegal-vbr-hq.md` |

## 2. 待测清单（按优先级）

### 2.0 补测结果总表（2026-10-14 实跑，T4 / driver 580.65.06 / SDK apiVersion 13.0）

| ID | 判据 | 结果 | 证据 |
|---|---|---|---|
| **B4** | `rc_ptr[1]=32/64` 是否被驱动接受 | ✅ **接受**（InitializeEncoder rc=0，0/1/2/32/64 全通过）⇒ **不需重跑全套标定** | `Accessory/probe/nvenc_rc_enum_truth.py`（**修桩后**，见下方「探针自身 bug」） |
| **B1** | h264+LA=8 ≥4 段跨段复用不崩 | ✅ **PASS**（S1~S7 全绿；S2 段级门禁 4/4、S5 帧守恒/单 IDR/frame_num 单调/pts 全过） | `verification_report/b1_h264_la8_t4_postfix_2026-10-14.md`（80s 素材 / 20s 分段 / 4 段 / 4800 帧） |
| **B1（判据逐条核销）** | 段 2 起崩溃 / `non-existing PPS` / pipe broken / rc=1 | ✅ 四项**全部未出现**：4 段全部 `处理完成`，`Cached SPS+PPS` 每段重新注入，rc=0 | 同上 + `B1post_clean.log` |
| **B2 ①** | 三档分级拒绝行为 | ✅ 5/5：h264_nvenc+vbr→rc=2、hevc_nvenc+cbr→rc=2、libx264+vbr→rc=0 且告警、av1_nvenc+vbr→rc=0、vbr_hq 基线→rc=0 | `--dry-run` 矩阵（本文件 §4 命令） |
| **B2 ②** | 方案 B 基线（`vbr` 在 Level 1 的落点） | ✅ 记录：`Ready: … H.264 CONSTQP QP=26 … la=8->0(未使能:CONSTQP) [未实现的rate_mode='vbr'→实际CONSTQP]`（`cbr` 同）——**QP 未过 `to_constqp_qp` 换算** | `B2_fallback_baseline.log`（两侧同构复现） |
| **plan_gate** | 101 项口径真机复跑 | ✅ **101 项 / 99 PASS / 0 FAIL / 0 WARN / 2 SKIP**（SKIP=R8 NVML 提示、RT-0 输出视频存在）；`BEH-G9/G10/G11` **确认真执行且 PASS** | `verification_report/verification_report_20261014_074251.md` |
| **hevc-la-drain baseline** | `conserved=True` + decode OK + `pending_before` 恒 8 | ✅ `vcl_total=700 leftover=0 mismatch=0`、`pending_before=[692..699]`（**恒 8 条**）、`ffmpeg decode OK` | `hevc_lookahead_diagnose.py --variant baseline --codec h264 --la-depth 8 --decode-check` |
| **esrgan reopen 槽位相位** | 段 2+ 无告警 + 各段帧数守恒 | ✅ 3 次 `[FIX-SKIP-REOPEN] …(gen=1, frame_idx=1281/2640/3639)`，无告警；S2 4/4 + S5 帧数守恒 | 同 B1 报告 |
| **LA=16 → W=17 复验** | 首段窗口无整体循环移位 | ✅ `segment_bitstream_verify_v5` **帧号单调**（无循环移位）；⚠ `slots` 实测 **19**（非 17）—— 见下方「W 口径修正」 | `verification_report/la16_w17_ifrnet8seg_t4_2026-10-14.md`（8 段 / 4796 帧 / S1~S5 全绿） |
| **S8 内存斜率** | 后半程 RSS 斜率 ≤ +50 MB/min | ❌ **FAIL**（+91.4 MB/min，峰值 11175 MB）——根因见下方「S8 根因」 | B1 报告；明细 `memdump/vbr_hq.mem.tsv` |

#### ⚠ 探针自身 bug（B4 差点给出反向裁定）

`Accessory/probe/nvenc_rc_enum_truth.py` 初版有两个缺陷，**若照单全收会误触发「全套标定重跑」**：

1. **`query_caps()` 用裸 `byref()/sizeof()`** —— 本模块只 `import ctypes`，必然 `NameError`，
   被 `except` 吞成「caps 查询不可用」并静默退回弱判据的 `--try` 路径。
   修复后 caps 查询真实执行，返回 **驱动 code=8（该 caps 不可用）** ⇒ T4/driver 580 上 caps 路径不可用，
   只能靠实编 A/B 判定（此为**驱动观测**，非探针失效）。
2. **`try_rc_values()` 「先记录后覆写」** —— `written.append(rc_ptr[1])` 记的是**覆写前**生产自己写的值
   （`constqp` ⇒ 0），却拿它跟请求值 `v` 比对 ⇒ `v=32/64` 恒判「写入回读不符」= **假 FAIL**，
   并以 `exit 3` 输出「驱动拒绝、生产代码会崩」。真实判据是覆写**后** cfg 里的值，
   且该 cfg 随后送进 `InitializeEncoder`（构造成功即驱动接受）。

⇒ 两处已修（`ctypes.byref/sizeof` + 写入后回读，并同时打印覆写前值便于对照）。

#### ⚠ W 口径修正（`la-window-rotation-fix.md` 的「W=LA+1」不普适）

实测 `_slot_count` **随分辨率变化**：720p 下 `LA=8 → slots=11`、`LA=16 → slots=19`（均 = LA+3），
2560x1440（ESRGan 侧）下 `LA=8 → slots=9`（= LA+1）。
memory 原文的 `W = max(pipeline_depth, LA+1)` 只在 2560x1440 成立（该行由 `LA=8→W=9` 反推）。
**LA=16 复验的真正判据是「首段窗口无整体循环移位」（frame_num 单调），该项已 PASS**；
「W=17」应改述为「W 随分辨率/管线配置而定，720p/LA=16 实测 19」。

#### ⚠ S8 根因（FAIL 但**非无界泄漏**）

`_get_or_create_result_pool` 的复用条件是 `cached_pool_size >= pool_size`，而 `pool_size` 是**自适应**的
⇒ 每次变大都重新分配一个新池：

| 运行 | 分段 | 池大小演进 | 累计分配 | 实测 RSS 斜率 |
|---|---|---|---|---|
| B1（20s 分段） | 4 段 | 14→20→24→25 | **1877 MB** | +91.4 MB/min（262 样本 / ~22 min） |
| LA=16（10s 分段） | 8 段 | 14→20→24→39→50 | **3324 MB** | 全程 +3210 / 后半程 +1808 MB/min（28 样本 / 144 s） |

序列在收尾期**回落**（`8860→8752→8573→7388→6164 MB`，段末释放生效）⇒ 不是单调无界增长；
但**分段越短、段数越多，池爬坡越大**，T4 16 GB 上属真实内存压力风险。
⇒ 遗留建议（P1，未做）：池**原地扩容**而非「新分配+free 旧池」；或 S8 判据排除爬坡段后拟合。

### 2.1 原待测清单（已执行，判据见 §2.0）

### P0 · 已改代码但真机零证据

| ID | 内容 | 代码状态 | 待测判据 | 估时 |
|---|---|---|---|---|
| **B1** | h264 + LA>0 + 跨段复用 ⇒ 段 2 起必崩（**生产默认路径**） | ✅ 已修：`nvenc_sdk.py:2342` `[FIX-B1-SPS-PPS-SESSION-REUSE]`（只重置 `_sps_pps_injected`，`_cached_sps_pps` 按 `_cached_sps_pps_gen != _session_gen` 条件清） | h264+LA=8 **≥4 段**产出：可解码 / frames==packets / 段 2+ 无 `non-existing PPS` / **S8 斜率可算**（补齐缺失的 A/B 对照臂——2026-10-04 因 B1 崩溃只剩 12 点启动爬坡段，得出的是 +4585/+1872 启动伪影） | **15~25 min** |
| **B2** | `--rate-mode vbr/cbr` 在 Level 1 静默落 CONSTQP | ✅ 分级拒绝 + `_log_ready` 按硬件真值回显（`ecd60f8`）；⚠ **方案 B（真 vbr 分支）未实现** | ① 三档分级行为符合预期（硬编非 AV1 硬拒 / 软编仅告警 / av1 放行）；② 记录 `--rate-mode vbr` 在 Level 1 的当前落点作为方案 B 基线 | **10~15 min**（方案 B 开发另计 1~3 h） |

### P1 · 裁决型探测（分钟级，但决定是否重跑全套标定）

| ID | 内容 | 状态 | 判据 / 分支 | 估时 |
|---|---|---|---|---|
| **B4** | `rc_ptr[1]=32/64` 是否真被驱动接受 | 探针**已就绪**：`Accessory/probe/nvenc_rc_enum_truth.py`（驱动 runtime 裁决 + caps 序号运行时自检） | 探针出裁定。若**不支持** ⇒ `vbr_hq` 分支改写 `rc_ptr[1]=1`+保留 `targetQuality`+`averageBitRate`，且**必须重跑全套等质量标定**（T4 双轴 + L40 AV1 双轴） | **1~5 min** |

### P1 · 多分段 / 长跑类待复跑（可与 P0 同批共用素材与 session）

| 来源 | 内容 | 判据 | 估时 |
|---|---|---|---|
| `hevc-la-drain-diagnosis.md:363` | 待 GPU 复跑 baseline(h264) | `conserved=True` + decode OK + 日志 `pending_before` 恒 8 条 | 5~15 min |
| `esrgan-reopen-slot-phase-misalignment.md:77` | reopen 槽位相位修复待实测 | 段 2+ 无告警、各段帧数守恒 | 10~20 min |
| `la-window-rotation-fix.md:32` | LA=16 → W=17 待复验（LA=8/W=9 已验） | 首段窗口无整体循环移位 | 10~15 min |

### P2 · 开发型（需 GPU 才能推进，且改运行时语义 ⇒ 按 `feedback_no_gpu_work_mode` 只给方案不落地）

| 项 | 内容 | 估时 |
|---|---|---|
| B2 方案 B | Level 1 真 vbr 分支 | 开发 1~3 h + 验证 30 min |
| `nvenc-hardware-test-isolation` 方案 B/C | 全量 pytest SIGSEGV 隔离根治（`conftest.py` autouse 清理 fixture 等） | 1~2 h + 验证 |

### P3 · 疑似陈旧，建议核销（历史 memory 的「需 GPU 验证」可能已被后续覆盖）

| 来源 | 内容 |
|---|---|
| `v6.4.x-backport-fixes.md:75` | 2026-08 的 backport 需 GPU 验证 |
| `project_v642_v643_bugs.md:18` | 帧数异常 +10~18 帧「未定位」，疑竞态 |
| `optimization-execution-2026-08.md:51,67` | logger 迁移分批回归 / CE-pipeline 结构重构需 GPU 基准 |
| `av1-nvenc-l40-calibration.md:80` | `_stderr_lines` 需上机打点（L40 侧；h264/hevc 同构，也可在 T4 打点） |

## 3. 建议批次（一轮 T4，40~70 min，覆盖 P0+P1）

```bash
# 0. 环境体检（av1 rc≠0 属预期）
nvidia-smi -L; for C in h264_nvenc hevc_nvenc av1_nvenc; do
  ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 -c:v $C -f null - < /dev/null
  echo "$C rc=$?"; done

# 1. B4 裁决（先跑：决定后面要不要重跑全套标定）—— 分钟级
python3 Accessory/probe/nvenc_rc_enum_truth.py < /dev/null

# 2. B1 真机复验（≥4 段 h264+LA=8 冒烟 + 段级验收 + S8 采样）
python3 Accessory/verify/av1_pipeline_smoke.py --codec h264_nvenc \
    --src <90s 素材> --rate-modes vbr_hq --segment-duration 20 --batch-size 8 \
    --mem-interval 5 --mem-peak-mb 16000 --keep < /dev/null
python3 Accessory/verify/segment_bitstream_verify_v5.py <产物> --skip-chroma < /dev/null

# 3. B2 分级拒绝 + esrgan reopen 多分段 + LA=16 复验（同批素材可复用）
```

### 4. 本批实际使用的命令（可复现，2026-10-14）

```bash
cd /workspace/Video_Enhancement
B=/tmp/opencode/t4batch            # 长素材与日志落点（仓库外，勿提交）

# 0. 素材：从 4K 长片裁 100s 720p（含音轨），再 stream-copy 出 80s 版
ffmpeg -y -ss 420 -i "../input_videos/Earth.at.Night.in.Color.S02E01.mp4" -t 100 \
  -vf "scale=1280:720,fps=30" -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p \
  -c:a aac -b:a 128k $B/src100s_720p.mp4
ffmpeg -y -i $B/src100s_720p.mp4 -t 80 -c copy $B/src80s_720p.mp4

# 1. B4 裁决（先跑：决定要不要重跑全套标定）
python3 Accessory/probe/nvenc_rc_enum_truth.py --try-values 0,1,2,32,64 --frames 120 < /dev/null

# 2. plan_gate 101 项真机复跑（改完 nvenc_sdk.py 后再跑一次做回归）
python3 Accessory/verify/plan_implementation_gate.py < /dev/null

# 3. B1：h264+LA=8、80s 素材、20s 分段 ⇒ 4 段跨段复用
python3 Accessory/verify/av1_pipeline_smoke.py --codec h264_nvenc \
  --src $B/src80s_720p.mp4 --out-dir $B/B1post_h264_la8 \
  --rate-modes vbr_hq --segment-duration 20 --batch-size 8 \
  --mem-interval 5 --mem-peak-mb 16000 --keep \
  --report verification_report/b1_h264_la8_t4_postfix_2026-10-14.md \
  --json   verification_report/b1_h264_la8_t4_postfix_2026-10-14.json < /dev/null

# 4. B2 ① 分级拒绝矩阵（--dry-run 不碰 GPU，秒级）
SRC=$B/src80s_720p.mp4
python3 src/main_video_optimized.py -c config/default_config.json -i $SRC -o /tmp/b2.mp4 \
  --dry-run --codec-ifrnet h264_nvenc --rate-mode-ifrnet vbr          # 期望 rc=2
python3 src/main_video_optimized.py -c config/default_config.json -i $SRC -o /tmp/b2.mp4 \
  --dry-run --codec-ifrnet hevc_nvenc --rate-mode-ifrnet cbr          # 期望 rc=2
python3 src/main_video_optimized.py -c config/default_config.json -i $SRC -o /tmp/b2.mp4 \
  --dry-run --codec-ifrnet libx264    --rate-mode-ifrnet vbr          # 期望 rc=0 + 告警
python3 src/main_video_optimized.py -c config/default_config.json -i $SRC -o /tmp/b2.mp4 \
  --dry-run --codec-ifrnet av1_nvenc  --rate-mode-ifrnet vbr          # 期望 rc=0（放行）

# 5. hevc-la-drain baseline（h264 + LA=8）
python3 Accessory/probe/hevc_lookahead_diagnose.py --variant baseline --codec h264 \
  --la-depth 8 --decode-check --frames 700 --out $B/drain_baseline_h264.log < /dev/null

# 6. LA=16 → W 复验（8 段 × 10s；--skip-upscale 只跑 IFRNet，~2.5 min）
python3 Accessory/verify/av1_pipeline_smoke.py --codec h264_nvenc \
  --src $B/src80s_720p.mp4 --out-dir $B/LA16 --rate-modes vbr_hq \
  --segment-duration 10 --batch-size 8 --mem-interval 5 --mem-peak-mb 16000 \
  --mem-dump-dir $B/memdump --keep \
  --report verification_report/la16_w17_ifrnet8seg_t4_2026-10-14.md \
  --json   verification_report/la16_w17_ifrnet8seg_t4_2026-10-14.json \
  -- --lookahead-depth-ifrnet 16 --lookahead-depth-esrgan 16 --skip-upscale < /dev/null
```

⚠ **两个踩坑点**（照抄可省时间）：
1. `av1_pipeline_smoke.py` 的透传参数是**位置参数**（`extra`），`--lookahead-depth-*` 会被 argparse
   当未知选项报错 ⇒ 必须用 `--` 分隔（`... -- --lookahead-depth-ifrnet 16 --skip-upscale`）。
2. 该脚本的 S6（QA sidecar）在 `--skip-upscale` 下必然 FAIL（sidecar 不生成）⇒ 属**用例局限**非缺陷，
   引用该报告时须剔除 S6。

⚠ **前置纪律**：B1 属生产默认路径缺陷，验收判据必须**逐条对 `d215fc2` 的缺陷描述**核销
（段 2 起崩溃 / non-existing PPS / pipe broken / rc=1），不得只看总 PASS 数
（参见 [[feedback_artifacts_not_execution_evidence]]、[[feedback_verify_cross_session_claims]]）。

## 相关

- [[esrgan-nvenc-qpval-l1-fallback-2026-10-14]] — 本批发现并修复的 P0 缺陷（ESRGan 侧 Level 1 静默降级）
- [[t4-s8-findings-and-blockers]] — B1/B2/B3/B4 权威原文与 E1~E5 勘误表
- [[nvenc-rc-enum-illegal-vbr-hq]] — B4 静态断言撤回与 caps 裁决入口
- [[l40-full-verification-run-2026-10-09]] — L40 侧同批盘点（含 L40-1b 折中方案）
- [[gate-stub-session-gen-audit-2026-10-10]] — 为何「plan_gate PASS」不能替代真机证据