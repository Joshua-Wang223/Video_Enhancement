# L40-1b 降级版冒烟实跑（2026-10-15，L40 在场）

**执行环境**：标准 Linux 容器，`/workspace/Video_Enhancement` 布局，GPU = **NVIDIA L40**（UUID GPU-bf031bb6…），
`torch.cuda.is_available()=True`，ffmpeg 9.0.2（h264/hevc/**av1** NVENC 实跑一帧均 rc=0），
素材池 `/workspace/input_videos`，Python 3.11.1（pyenv）。
动机：执行 `Plan/PROMPT_L40_全流程验证.md` §「折中执行方案」的 **L40-1b**（该方案最后剩余的唯一零证据项），
并按证据链核验结论**跳过** L40-2/3/4 的补跑（详见下）。

## 逐项结果

| 步骤 | 项 | 结果 | 实测 |
|---|---|---|---|
| 4a | 裁剪 120s 子片 | ✅ rc=0 | `-ss 60 -t 120`，libx264 veryfast crf18 + aac；120.000s / 720×576 / 25fps；源先 `cp` 到 ASCII 路径 `/tmp/clifford_src.mp4` 规避中文路径传递 |
| 4b | **L40-1b 冒烟**（av1_pipeline_smoke，constqp 单臂，S3 复用 S4 已启用） | ✅ **PASS=8 / FAIL=0 / SKIP=0** | 管线 rc=0 仅 **182.9s**；S2 段门禁 4/4；S3+S4 帧数 **5996=5996**（源 3000，120s×25fps×2 插值）；S5 帧守恒/单 IDR/frame_num 单调/无 pts_anomaly（3.18s）；S6 sidecar 完整；S7 codec=**av1** 确认；**S8 斜率 −109.3 MB/min**（RSS 峰值仅 4874 MB，36 样本，主 −93.6 / 子 −15.7 / PSS −113.5），远优于 ≤+50 门限 |
| 1 | L40-4 落表双轴复跑 | ✅ 双轴 PASS | CQ：`av1_nvenc (1.4566,+1.2165,0,63)` LOO **3.13** ≤5.9，17 样本，3 seed 逐位一致，最差留出素材 live_kids_play；QP：`av1_nvenc (7.9338,−97.5136,0,255)` LOO **2.61** ≤5.9；与 `convert_crf.py:212` / `quality_map.py:202` 现值逐位一致 |

**证据文件**：`verification_report/av1_smoke_L40_clifford120s_2026-10-15.md`；
内存采样 `Accessory/data/s8_mem_L40_2026-10-15/clifford_120s/constqp.mem.tsv`。

## 补测判定（L40-2 / L40-3 / L40-4 均不补跑）

- 三者 2026-10-09 实跑证据在库（`crfcq_gpu_L40_2026-10-09.md` 113 PASS/0 FAIL/3 WARN、
  `av1_vp9_matrix_L40_2026-10-09.md` `-qp 70` 落带 1.07×/−1.13 dB、落表 LOO 3.13/2.61）。
- 判定依据：相关表值/脚本/points 最后变更全部早于 10-09 实跑——
  `quality_map.py`/`eqq_pool_fit_table.py`/gpu_l40 points = `b64c1d2`（10-04）、
  `convert_crf.py`/`crf_cq_unification_verify.py` = `20b9e81`（10-04）、
  `av1_vp9_quality_matrix.py` = `ecd60f8`（10-05）；工作树 clean，远程无领先提交。
- L40-2 的 2026-10-14 判据修订是**文档级方案 A**（G7-6 降备案观测、G7-5 VMAF 对齐转正），
  10-09 轮 G7-5 已 ΔVMAF 0.89 ≤2.5 PASS ⇒ 新判据在旧数据上即达成，重跑不期望改变 G7-6
  （AV1 族 plain-PSNR 结构性偏置，见 [[g7-6-metric-mismatch-2026-10-14]]）。
- `6a51633`（10-14，ESRGAN NVENC `_qp_val` L1 静默降级修复）只动管线侧 nvenc_sdk，
  不影响 crf_cq/矩阵/落表工具；L40-1b 即该修复在 L40 的首次全链路实跑。

## 耗时口径修正（重要）

- 本轮 L40 在场总耗时 ≈ **6 min**（裁剪 ~0.5 min + 冒烟 182.9s + 校验 ~2 min + 落表 ~15 s），
  远低于方案估的 15~25 min / 30~50 min。原因：L40@2880×2304 实测吞吐约 32 输出帧/s，
  显著高于 new5_raw 单臂外推基准（845s@4K/1605 帧，含 TRT 冷启动）。
- 方案 §「折中执行方案 30~50 min」上限只在触发 **S8 补臂（+15~25 min）** 或补跑 L40-2/3 时才逼近。

## 已知未覆盖（留档声明，不得据此宣称长视频已验收）

- **24→4 段拼接**：段边界压力下降（本版仅 4 段）。
- **S8 长窗斜率**：采样窗 ~3 min，缓泄漏判别力弱（本轮 −109.3 MB/min 仅证本轮无泄漏）。
- **长跑尾帧**（水彩/尾损历史缺陷）：120s 内未完整覆盖。
- 完整版 Clifford 720.5s 单臂 ~1.3~2.3 h / 两臂 2.5~4.5 h，未排期；
  sm89 双 engine 已缓存，恢复命令见 `Plan/PROMPT_L40_全流程验证.md` §执行命令 第 1 项。

## 附：本轮踩到并排除的读表陷阱

`eqq_pool_fit_table.py` 输出行序为 6 软编档 → h264_nvenc → hevc_nvenc → av1_nvenc，
**用 `grep av1_nvenc | head -8` 会把 av1 行截掉**，误判「静默假通过」。核对该脚本输出时
必须看全表或直接 `tail`；rc=0 本身不构成通过证据（AGENTS.md 已记录该陷阱）。
