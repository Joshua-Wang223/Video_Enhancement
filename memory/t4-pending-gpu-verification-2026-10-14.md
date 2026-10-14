# T4 GPU 待测项目盘点（2026-10-14）

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

⚠ **前置纪律**：B1 属生产默认路径缺陷，验收判据必须**逐条对 `d215fc2` 的缺陷描述**核销
（段 2 起崩溃 / non-existing PPS / pipe broken / rc=1），不得只看总 PASS 数
（参见 [[feedback_artifacts_not_execution_evidence]]、[[feedback_verify_cross_session_claims]]）。

## 相关

- [[t4-s8-findings-and-blockers]] — B1/B2/B3/B4 权威原文与 E1~E5 勘误表
- [[nvenc-rc-enum-illegal-vbr-hq]] — B4 静态断言撤回与 caps 裁决入口
- [[l40-full-verification-run-2026-10-09]] — L40 侧同批盘点（含 L40-1b 折中方案）
- [[gate-stub-session-gen-audit-2026-10-10]] — 为何「plan_gate PASS」不能替代真机证据