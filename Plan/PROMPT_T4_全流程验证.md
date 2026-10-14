# T4 侧全流程验证立项

## 背景
CPU 侧 7/7 测试已通过，需在 Tesla T4 (Turing, sm75) 上完成 GPU 相关验收。

## 环境要求
- Tesla T4 (sm75) / 驱动 ≥535 / CUDA 12.x
- FFmpeg 9.0+ with h264_nvenc, hevc_nvenc
- PyTorch CUDA 可用
- 模型权重就位：`models_IFRNet/checkpoints/`, `models_RealESRGAN/`, `models_GFPGAN/`

## 验收范围（T4 独有）

| 测试项 | 脚本 | 关键判据 |
|--------|------|----------|
| plan_gate (完整) | `plan_implementation_gate.py` | **101 项 PASS/0 FAIL**（⚠ 2026-10-14 更正：旧写「89 项」是门禁桩失效期的读数——`BEH-G9/G10/G11` 等 12 项曾被组级 `BEH-ERR` 吞掉从未执行，2026-10-10 修桩后才恢复；详见下方门禁基线与 `memory/gate-stub-session-gen-audit-2026-10-10.md`） |
| crf_cq --gpu | `crf_cq_unification_verify.py` | G7/G8 画质/码率天花板 FAIL=0（质量对齐以 **G7-5 VMAF 口径**为准，2026-10-14 口径修订；T4 侧 `G7-6` 因无 AV1 恒 SKIP，不受影响） |
| nvenc_vbr_hq_verify | `nvenc_vbr_hq_verify.py` | CLI 拒绝 vbr_hq / SDK 接受 rc_ptr[1]=32 / ΔVMAF≤0.3 |
| nvenc_rc_diagnose | `nvenc_rc_mode_diagnose.py` | RC 枚举/驱动能力/选项表确认 |
| segment_bitstream_verify (hevc+LA=8) | `segment_bitstream_verify_v5.py` | 帧守恒/单 IDR/frame_num 单调/无色度簇 |
| av1_vp9_matrix_cpu | 复用 CPU 结果 | 软编 3 编码器 PASS |
| eqq_pool_fit (含 gpu_t4) | `eqq_pool_fit_table.py` | h264/hevc CQ/QP LOO ≤5.9 |

## 执行命令
```bash
cd /workspace/Video_Enhancement

# 0. 环境体检
nvidia-smi -L
python3 -c "import torch;print(torch.cuda.get_device_name(0),torch.cuda.is_available())"
for C in h264_nvenc hevc_nvenc av1_nvenc; do
  ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 -c:v $C -f null - < /dev/null; echo "$C rc=$?"
done
# 期望: h264=0, hevc=0, av1≠0

# 1. 完整验证（含冒烟）
python3 Accessory/verify/comprehensive_verify.py --env t4 \
    -i ../input_videos/word_world_2.mp4 -o output.mp4 \
    --source ../input_videos/word_world_2.mp4 \
    --bitrate-source ../input_videos/new4_raw.mp4 \
    --smoke --smoke-mode interpolate_then_upscale

# 2. 单独跑 NVENC 专项（如需）
python3 Accessory/probe/nvenc_vbr_hq_verify.py
python3 Accessory/probe/nvenc_rc_mode_diagnose.py $(find / -name nvEncodeAPI.h 2>/dev/null | head -1)

# 3. hevc+LA=8 帧守恒回归
python3 run.py -i /tmp/seg_src_5s.mp4 -o /tmp/seg_hevc_la8.mp4 \
    --mode interpolate_then_upscale \
    --codec-ifrnet hevc_nvenc --codec-esrgan hevc_nvenc \
    --rate-mode-ifrnet vbr_hq --lookahead-depth-ifrnet 8 < /dev/null
python3 Accessory/verify/segment_bitstream_verify_v5.py /tmp/seg_hevc_la8.mp4 < /dev/null

# 4. 落表器含 GPU 侧
python3 Accessory/probe/eqq_pool_fit_table.py --sides 6s,10s,legacy10s,gpu_t4_cq --axis cq < /dev/null
python3 Accessory/probe/eqq_pool_fit_table.py --sides gpu_t4_qp --axis qp < /dev/null
```

## 门禁基线（2026-10-04 T4 实测）
- `plan_implementation_gate`: ~~96/94/0/2~~ → **已过期，见下方 plan_gate 记录更正**
- `crf_cq --gpu`: PASS=101 / FAIL=0 / WARN=4 / SKIP=1
- `nvenc_vbr_hq_verify`: V12 接受 rc_ptr[1]=32 → 方案 A
- hevc+LA=8: frames=packets=199, 无色度簇

### plan_gate 记录更正（2026-10-10，2026-10-14 同步到本方案）

2026-10-04 的「96/94/0/2（共 89 项）」与「89 项 PASS/0 FAIL」**两个数字都不可采信**：

- 门禁桩 `plan_implementation_gate.py::make_enc()` 走 `NVENCEncoder.__new__()` **绕过 `__init__`**，
  未补 `_session_gen` ⇒ `_stream_begin()` 抛 `AttributeError` ⇒ 被**组级 `BEH-ERR` 吞掉整组**。
  ⇒ **`BEH-G9`（NVENC 段边界）/G10/G11 等 12 项从未真正执行**，门禁长期只报 89 项（实为 101 项）
  且总数变化无任何告警。
- 根因**不在生产代码**：`d215fc2` 的 `_session_gen` 会话代数逻辑是对的；缺陷仅在桩。
- 已修桩并复核：**101 项 / 97 PASS / 0 FAIL / 2 WARN / 2 SKIP**，`BEH-G9/G10/G11` 恢复执行且全部 PASS
  ⇒ 生产修复有效。
- ⚠ **对 T4 的含义**：T4 上从未跑过「真·101 项 + BEH 组真执行」的门禁。上机时须按 101 项口径复跑（见待测清单 P1）。

## 待测清单（2026-10-14 盘点新增）

> 原「验收范围」7 项（T4-1~T4-9 等质量标定 / crf_cq --gpu / vbr_hq_verify / rc_diagnose /
> segment_bitstream / 落表）**均已收口 ✅**。但盘点发现：**全部 T4 GPU 证据停在 2026-10-04**，
> 早于 `ecd60f8`(10-05 B2 分级拒绝) 与 `d215fc2`(10-09 B1 修复) ⇒ **当前代码在 T4 上零回归证据**。
> 完整清单与批次命令见 `memory/t4-pending-gpu-verification-2026-10-14.md`。摘要：

| 优先级 | ID | 待测内容 | 估时 |
|---|---|---|---|
| **P0** | B1 | h264+LA>0+跨段复用段 2 起必崩（**生产默认路径**）——修复在 `nvenc_sdk.py:2342`，**零真机证据**；跑 ≥4 段 h264+LA=8 冒烟 + 段级验收 + S8 采样 | 15~25 min |
| **P0** | B2 | Level 1 `--rate-mode vbr/cbr` 静默落 CONSTQP：验证三档分级拒绝行为 + 记录当前落点作为方案 B 基线 | 10~15 min |
| **P1** | B4 | `rc_ptr[1]=32/64` 是否被驱动接受——探针 `Accessory/probe/nvenc_rc_enum_truth.py` 已就绪；**裁定为「不支持」则须重跑全套等质量标定** | 1~5 min |
| **P1** | plan_gate | 按 **101 项**口径在 T4 真机复跑（含修桩后才真执行的 `BEH-G9/G10/G11`）——T4 从未跑过真·101 项 | ~2 min |
| **P1** | — | hevc-la-drain baseline 复跑（`conserved=True` + `pending_before` 恒 8） | 5~15 min |
| **P1** | — | esrgan reopen 槽位相位：段 2+ 无告警、各段帧数守恒 | 10~20 min |
| **P1** | — | LA=16 → W=17 复验（LA=8/W=9 已验） | 10~15 min |
| **P2** | — | 开发型：B2 方案 B 真 vbr 分支（1~3 h）、pytest SIGSEGV 隔离方案 B/C（1~2 h） | 开发另计 |
| **P3** | — | 疑似陈旧待核销：v6.4.x backport / v642 帧数异常未定位 / 2026-08 重构基准 / `_stderr_lines` 打点 | — |

**建议一轮 T4 批次 40~70 min 覆盖 P0+P1**（B4 先跑，分钟级且决定后续是否重跑标定）。
⚠ B1 属生产默认路径缺陷，判据须**逐条对 `d215fc2` 的缺陷描述核销**
（段 2 起崩溃 / `non-existing PPS` / pipe broken / rc=1），不得只看总 PASS 数。

## 产出物
- `verification_report/verification_report_YYYYMMDD_HHMMSS.json/.md`
- `verification_report/CRF_CQ统一验证报告_YYYYMMDD_HHMMSS.md`
- `verification_report/av1_vp9_matrix_T4_YYYYMMDD.md`
- `verification_report/nvenc_vbrhq_migration_YYYYMMDD/`