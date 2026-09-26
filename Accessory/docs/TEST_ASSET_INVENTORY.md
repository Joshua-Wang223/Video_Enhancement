# `Accessory/` 资产清点表（2026-09-15，Windows 开发树实测）

> 立项来源：`Plan/门禁与测试资产纳管清理_立项Prompt.md`
> 数据来源：本机实测（`sha256` 前 16 位 + 全仓引用矩阵 + `mtime`）。
> ⚠️ 本表是**候选状态**，不是待删清单。删除任何一项都必须先有
> **「逐字节重复 + 零引用 + 已确认无改名/搬移意图」** 双证据（v4/v5 就是反例）。

## 0. 结论摘要

| 项 | 结果 |
|---|---|
| `Accessory/` 下 py 文件总数 | **59**（`Accessory/*.py` 55 + `Accessory/video_check/*.py` 4） |
| 纳入门禁扫描 | **59 全部**（`COVERAGE_ROOTS` 新增 `"tests"`） |
| `COMPILE_TARGETS` | **54 → 112**（生产 53 + tests 59） |
| `BEH-H1` 扫描调用点 | **65 → 245** |
| 扫描后新发现的真实缺陷 | **1 处**（`nvenc_rc_mode_diagnose.py:178` 参数互斥，已修） |
| 状态分布 | 活跃 22 / 历史 22 / 待定 15 |
| 待删项 | **0**（无一项同时满足双证据） |
| 换行符统一 | **本机不执行**（见 §3） |

> 本轮新增 3 个 `Accessory/` 资产（均为本机可跑的纯 CPU 件，另见 §3）：
> `test_reader_unbound_watchdog.py`（读帧器看门狗回归）、
> `test_esrgan_apply_sps_pps_equivalence.py`（SPS/PPS 统一入口等价性）、
> `reader_rgb_path_diagnose.py`（NVDEC vs 软解 RGB 诊断）。

## 1. 判定规则（先定规则再看表，避免逐文件拍脑袋）

| 状态 | 判定依据 |
|---|---|
| `活跃·生产引用` | 被 `src/` 或 `external/`（**排除** `external/IFRNet/`、`external/Real-ESRGAN/` 两个历史/第三方目录）按文件名引用 |
| `活跃·门禁/验收` | 门禁主体、其兼容别名、或被 `AGENTS.md` 等现行文档点名的验收资产 |
| `活跃·近期维护` | `mtime ≥ 2026-09-01` 且属当前链路（切镜/CRF-CQ/读帧器/帧计数） |
| `历史·前代版本` | 存在更高的 `_vN` 后继，且无生产引用 |
| `历史·已收口诊断` | 针对已闭环缺陷的一次性诊断/复现脚本 |
| `待定` | 其余（**一律保留**；标注"未确认可删"，留待用户在 Linux 侧按引用矩阵复核） |

> `external/IFRNet/`、`external/Real-ESRGAN/` 里的同名引用**不计入**"生产引用"
> —— 它们是拆包前的历史单体与上游第三方，不属于生产调用链。

## 2. 清点表

`refs` = 引用该文件名的**其它**源文件数（`prod` / `tests` / `memory+Plan`）。
`sha16` = `sha256` 前 16 位，用于识别逐字节重复。

| 文件 | size | mtime | sha16 | refs p/t/m | 状态 | 依据 |
|---|---:|---|---|---:|---|---|
| `fault_injection_wrapper.py` | 622 | 2026-09-09 | `867d0f094a41f7e9` | 0/0/2 | 活跃·工具 | `ifrnet_lookahead_repro.py` 的注入包装 |
| `pipe_deadlock_repro.py` | 2678 | 2026-08-29 | `9e52505b94bac356` | 0/0/1 | 待定 | 管道死锁复现，未确认可删 |
| `interp_ghost_analyzer.py` | 5324 | 2026-09-04 | `c53739ddb9fc67dc` | 0/0/1 | 待定 | 单列诊断 |
| `video_ifrnet_analyzer.py` | 41434 | 2026-07-17 | `6d9cef776b80272b` | 0/1/3 | 历史·前代 | 被 `video_pipeline_analyzer.py` 引用，无后继依赖 |
| `video_pipeline_analyzer.py` | 47387 | 2026-07-29 | `629ef791173d4b45` | 0/2/4 | 历史·前代 | 被 v2/v3 引用 |
| `video_pipeline_analyzer_v2.py` | 82481 | 2026-08-03 | `e86b67531d7b7545` | 0/0/3 | 历史·前代 | 有 v3 后继 |
| `video_pipeline_analyzer_v3.py` | 118355 | 2026-08-14 | `772932dd14b95089` | **3**/6/8 | **活跃·生产引用** | `src/utils/{video_utils,system_resources,parallel_executor}.py` |
| `video_realesrgan_analyzer.py` | 50339 | 2026-07-24 | `aaac4a8bd2764708` | 0/1/3 | 历史·前代 | 被 `video_pipeline_analyzer.py` 引用 |
| `ifrnet_versions_benchmark.py` | 38671 | 2026-06-30 | `d0c18b6e2c2e7c01` | 0/2/5 | 历史·前代 | 有 v2/v3 后继 |
| `ifrnet_versions_benchmark_v2.py` | 117358 | 2026-08-05 | `9972d0987d7dd0f6` | 0/1/3 | 历史·前代 | 有 v3 后继 |
| `ifrnet_versions_benchmark_v3.py` | 136911 | 2026-08-20 | `63403a7990160e48` | **3**/0/3 | **活跃·生产引用** | 同上三处 `src/utils` |
| `scene_cut_threshold_calibrator.py` | 3619 | 2026-09-04 | `c199635437de78e6` | 0/0/1 | 活跃·近期维护 | 切镜阈值标定（现行链路） |
| `conftest.py` | 119 | 2026-08-29 | `3b4fc4265ce20d0c` | 0/0/2 | 活跃·基建 | pytest 收集排除规则 |
| `hevc_lookahead_diagnose.py` | 75395 | 2026-08-18 | `8451f641ee6c1344` | 0/0/18 | 活跃·验收 | `AGENTS.md` 点名的 NVENC 层级回归资产 |
| `lockbitstream_timestamp_diagnose.py` | 32248 | 2026-08-12 | `815f1c536103d6f5` | 0/0/5 | 历史·已收口诊断 | 时间戳重关联已闭环 |
| `nvenc_rc_mode_diagnose.py` | 16701 | 2026-09-15 | `1a463ff5c41c1973` | 0/0/3 | 历史·已收口诊断 | 本轮修 H2 违规（§4） |
| `nvenc_profilelevel_offset_diagnose.py` | 20481 | 2026-08-07 | `b0189164553fdc9e` | 0/0/5 | 历史·已收口诊断 | profileLevel=51 已修 |
| `scene_cut_ghost_analyzer.py` | 3666 | 2026-09-04 | `29f7526fc70cacb9` | 0/0/1 | 活跃·近期维护 | 切镜残影诊断 |
| `nvenc_targetquality_offset_diagnose.py` | 45173 | 2026-06-09 | `66b521786495fedd` | 0/0/4 | 历史·已收口诊断 | RC params 布局已定标 |
| `minimal_enhanced_validation.py` | 4332 | 2026-09-15 | `092578014b21ae44` | 0/0/1 | 活跃·验收 | 立项文档点名纳入 |
| `cuda_context_probe.py` | 19615 | 2026-09-04 | `10006f19699ff7f1` | **1**/0/1 | **活跃·生产引用** | `external/realesrgan_video/nvenc_sdk.py` |
| `pytorch_nvml_probe.py` | 1760 | 2026-05-03 | `3e7a81581c189543` | 0/0/3 | 待定 | 最早期探针，未确认可删 |
| `ifrnet_lookahead_repro.py` | 7689 | 2026-09-09 | `982d187dedc37553` | 0/2/2 | 活跃·复现 | LA 缺陷复现器 |
| `real_frames_encode_repro.py` | 3709 | 2026-09-09 | `5faf85dd166878aa` | 0/0/1 | 活跃·复现 | 真实帧复现器 |
| `test_chroma_false_positive.py` | 8271 | 2026-09-15 | `53307e9448903081` | 0/0/3 | 活跃·回归 | v5 双名兼容（本轮已核） |
| `test_frame_count_probe.py` | 6030 | 2026-09-15 | `656ee5477c8056da` | 0/0/2 | 活跃·回归 | 帧计数严格口径探针 |
| `nvenc_completion_event_matrix.py` | 150121 | 2026-06-12 | `c13e751585725c59` | 0/4/5 | 历史·前代 | 有 v1~v5 后继 |
| `nvenc_completion_event_matrix_v1.py` | 138464 | 2026-06-12 | `2f84029ded946719` | 0/0/3 | 历史·前代 | — |
| `nvenc_completion_event_matrix_v2.py` | 159079 | 2026-06-12 | `5b612f89143ced99` | 0/0/3 | 历史·前代 | — |
| `nvenc_completion_event_matrix_v3.py` | 182127 | 2026-06-15 | `2564420277895b60` | 0/0/3 | 历史·前代 | — |
| `nvenc_completion_event_matrix_v4.py` | 224640 | 2026-06-25 | `4779c9b5b6fcc5cf` | 1/1/7 | 历史·前代 | — |
| `nvenc_completion_event_matrix_v5.py` | 247116 | 2026-07-03 | `5ada2ce1ff908629` | 0/1/3 | 历史·前代（最新世代） | — |
| `nvenc_comprehensive_matrix.py` | 63056 | 2026-06-29 | `ff9c46f1e262c32f` | 0/0/4 | 待定 | — |
| `nvenc_ipc_worker_probe.py` | 39471 | 2026-05-20 | `b3442d3630448f39` | 0/0/4 | 待定 | 疑为 spawn 子进程脚本 |
| `nvenc_la_frame_conservation_suite.py` | 71046 | 2026-07-03 | `c1d018d335b84ab3` | 3/3/14 | **活跃·回归资产** | 立项文档点名的 `_drain_outputs()` 模板 |
| `nvenc_session_pre_torch_probe.py` | 38719 | 2026-05-20 | `1f5c181a094f76b8` | 3/6/6 | 待定 | 引用来自历史 `external/IFRNet/` |
| `nvenc_sdk_realesrgan_suite.py` | 22158 | 2026-07-06 | `ae7e0062b5c3d9e6` | 0/0/4 | 待定 | NVENC 隔离立项点名的崩溃族 |
| `nvenc_vbr_hq_offsets_probe.py` | 32818 | 2026-06-08 | `568eae8a72b02022` | 0/0/4 | 待定 | — |
| `parallel_validation.py` | 3453 | 2026-09-15 | `852c94b1756b004b` | 0/0/0 | 活跃·近期维护 | 并行验证方案样板 |
| `pipe4_la8_corruption_diff.py` | 10380 | 2026-06-17 | `152be5933ce95b7d` | 0/0/5 | 历史·已收口诊断 | pipe4+LA8 已根治 |
| `test_reader_unbound_watchdog.py` | 15883 | 2026-09-15 | `f006aeb22a1144db` | 0/1/0 | **活跃·本轮新增** | 读帧器看门狗 CPU 回归（10/10 PASS） |
| `test_esrgan_apply_sps_pps_equivalence.py` | 8030 | 2026-09-15 | `f5995a97471b52bf` | 0/0/0 | **活跃·本轮新增** | SPS/PPS 统一入口等价性（120 组合 PASS） |
| `reader_rgb_path_diagnose.py` | 32084 | 2026-09-15 | `790eb8292e4e5cba` | 0/0/0 | **活跃·本轮新增** | NVDEC vs 软解 RGB 一致性诊断（GPU 侧执行） |
| `test_regression_min.py` | 865 | 2026-08-24 | `e037edfeeebdb45d` | 1/1/17 | 活跃·门禁别名 | 门禁兼容入口 |
| `sps_pps_startup_repro.py` | 35820 | 2026-06-16 | `a7af504502560988` | 0/0/5 | 历史·已收口诊断 | 已由 root-cause-fix 覆盖 |
| `crf_cq_unification_verify.py` | 124599 | 2026-09-11 | `c5c960ec9eef6c97` | 0/0/1 | 活跃·验收 | CRF/CQ 统一优化验收 |
| `plan_implementation_gate.py` | 120418 | 2026-09-15 | `282a9dcf687e98c3` | 1/1/30 | **活跃·门禁主体** | 94 项门禁 |
| `nvenc_rcparams_offset_verify.py` | 16590 | 2026-06-09 | `ed14904b5a9f4989` | 0/0/4 | 历史·已收口诊断 | RC 布局已定标 |
| `stream_ts_reassoc_backport_verify.py` | 11128 | 2026-08-11 | `946c30f386d7efac` | 0/0/4 | 历史·已收口诊断 | 旋转已改由合并阶段处理 |
| `scene_cut_fix_verify.py` | 3320 | 2026-09-04 | `ae8ce5551747a962` | 0/0/1 | 活跃·验收 | 切镜修复验收 |
| `segment_bitstream_verify.py` | 50087 | 2026-08-07 | `e04f74d2712ac0c8` | 0/0/3 | 历史·前代 | 有 v2~v5 后继 |
| `segment_bitstream_verify_v2.py` | 73083 | 2026-08-11 | `5f39a759f7725538` | 0/1/7 | 历史·前代 | — |
| `segment_bitstream_verify_v3.py` | 104672 | 2026-08-17 | `cf86de94e70eeab0` | 0/2/12 | 历史·前代 | — |
| `segment_bitstream_verify_v4.py` | 192131 | 2026-09-15 | `2566804141ee2c7a` | 0/0/24 | **活跃（不得删）** | 生产侧**旧名**，与 v5 逐字节相同 |
| `segment_bitstream_verify_v5.py` | 192131 | 2026-09-14 | `2566804141ee2c7a` | 4/1/2 | **活跃（不得删）** | 生产侧**新名**，Linux 侧已同步 |
| `video_check/gpu_video_check.py` | 9274 | 2026-07-02 | `27da64bf78d7edff` | 0/0/0 | 待定 | 独立小工具集 |
| `video_check/gpu_video_check1.py` | 7268 | 2026-07-02 | `b367e73c530ef24c` | 0/0/0 | 待定 | — |
| `video_check/video_quality_check.py` | 8500 | 2026-07-02 | `b428d0348090086d` | 0/0/1 | 待定 | — |
| `video_check/video_quality_check1.py` | 7456 | 2026-07-02 | `0a9afdcb57c3f84e` | 0/0/1 | 待定 | — |

> `video_check/` 整体归为**待定**：它是 4 个独立小工具（`gpu_video_check.py`、
> `gpu_video_check1.py`、`video_quality_check.py`、`video_quality_check1.py`）
> 外加 5 个 JSON 预设，无任何 src/external 引用，也未被现行文档点名。
> 该目录与本项目主链路无连接，删除与否由使用者决定。

### 2.1 关于 `segment_bitstream_verify_v4.py` / `_v5.py`（反例，不得删）

两者 `sha256` 前 16 位**完全相同**（`2566804141ee2c7a`）、大小同为 192131 字节，
但这是**「同一份内容两个名字」而非冗余副本**：使用者已确认 v5 即生产侧最新版本
另存，Linux 侧也已同步为 v5。**两个都不能删。**

两个真实待办的现状：

1. **运行期 import 断裂 —— 已闭合**。
   `Accessory/test/test_chroma_false_positive.py` 已改为 `v5 → v4` 双名兼容
   （标记 `[FIX-VERIFY-RENAME]`，`:22-37`），不再依赖单一名字。本轮已核验该 pytest
   可正常收集（本机缺 torch 时于门禁内 SKIP）。
2. **v5 内部自指的日志名仍是 `verify_segment_bitstream_v4_stuck.log`**
   （v5 文件 `:144`/`:299`/`:315`/`:3972`）。**有意不改** —— 一旦改就会破坏
   v4/v5 的逐字节一致性，而该一致性正是使用者当前核对"两个名字同源"的手段。
   仅在需要时再一并改两名。

## 3. 本轮已执行 / 已决定不做

| 项 | 处置 | 说明 |
|---|---|---|
| 步骤 0：修 `_v4` import 断裂 | ✅ 已闭合（核实为**先前会话已落地**） | `test_chroma_false_positive.py` 双名兼容 |
| 引用矩阵 + 内容哈希 | ✅ 已产出 | 见 §2 表 |
| `Accessory/` 扫描范围 | ✅ **全量纳入** | `COVERAGE_ROOTS` 增 `"tests"`；`COVERAGE_MIN_FILES` 45 → 95 |
| `BEH-E2` 自检扩展到 tests | ✅ 已加 | 新增「Accessory/ 未被覆盖」探测，防止排除规则写宽后静默剔除 |
| `_v4.py` 过时引用（活代码/文档） | ✅ 已更正 **13 处** | `ffmpeg_io.py:436`、`video_utils.py`×3、`system_resources.py`×2、`parallel_executor.py:5`、`minimal_enhanced_validation.py`×2、`parallel_validation.py`×2、`AGENTS.md`×2 |
| `verify_segment_output()` 死代码 | ✅ **保留 + 显式标注** | 见 `video_utils.py:2073` 起的去留决定注释；理由：删除收益 < 风险（本树非 git 仓库） |
| 冗余副本清理 | ✅ 复核完成，**待删 0 项** | v4/v5 为有意保留；无其它逐字节重复 |
| 换行符统一 + `.gitattributes` | ⏸ **本机不执行** | 使用者裁定：**以 Linux 侧为准，本机只是开发环境**。见 §3.1 |
| 版本控制复核 | ⚠ **本树非 git 仓库** | 见 §3.2 |

### 3.1 换行符：本机不执行（转 Linux 侧）

本机实测（53 个活跃生产 py）：

| 形态 | 数量 | 文件 |
|---|---:|---|
| 纯 CRLF | 4 | `src/utils/output_filter.py`(84)、`src/utils/video_utils.py`(3689)、`external/realesrgan_video/nvenc_sdk.py`(4089)、`external/nvenc_common/nal_utils.py`(146) |
| 混合 | 1 | `external/ifrnet_video/ifrnet_utils.py`（51 CRLF + 383 LF） |
| 纯 LF | 48 | 其余 |

**决定**：本机**不新增 `.gitattributes`、不改这 5 个文件的字节**。

理由：
- 本开发树不是 git 仓库，"独立一次提交以隔离 diff"这一前提在本机不成立，
  归一化只会产生**不可复核的大 diff** 并整体传导到 Linux；
- 使用者裁定以 Linux 侧为准，本机仅为开发副本 —— 归一化应在 Linux 仓库侧
  用一次独立提交完成。

**Linux 侧待办**（建议规则，勿盲改）：
1. 先确认 Linux 仓库中这 5 个文件在 git 索引里的形态；
2. 新增 `.gitattributes`：`* text=auto` + 显式 `eol=lf`（`*.bat`/`*.cmd` 除外）；
3. 归一化**单独一次提交**，提交信息写明"纯换行符，无逻辑变更"；
4. ⚠️ 加 `eol=lf` 会让 git 在下次 `add` 时对既有 CRLF 文件做**批量重归一化** ——
   必须作为独立提交并单独审查，不要与功能改动混在一起。

### 3.2 版本控制：本树不是 git 仓库

```
$ git status
fatal: not a git repository (or any of the parent directories): .git
```

因此立项文档中的两项无法在本机完成，转 Linux 侧：

1. **确认 `Accessory/verify/segment_bitstream_verify_v4.py` / `_v5.py` 是否已被跟踪** ——
   记忆 `env-ffmpeg-ffprobe-gotchas.md` 记录 v4 **未被跟踪**（`git ls-files` 无记录），
   导致无法用 `git show HEAD:<path>` 取基线做 A/B。**它是被生产源码引用的门禁资产，
   必须纳入跟踪**（含 v5）。
2. `.gitignore` 复核：现有规则含 `*_bak*` / `*.bak*` / `*.tar.gz` / `*.core`，
   与门禁的排除口径一致，无需改动。

## 4. 扫描范围扩大到 `Accessory/` 后新发现的真实缺陷

| 文件:行 | 缺陷 | 处置 |
|---|---|---|
| `Accessory/probe/nvenc_rc_mode_diagnose.py:178` | `subprocess.run(capture_output=True, ..., stderr=subprocess.DEVNULL)` —— 两者互斥，运行时必抛 `ValueError`；而它被外层 `except Exception: pass` **静默吞掉**，导致"兜底全盘查找 nvEncodeAPI.h"这段从未真正生效 | ✅ 改为 `stdout=subprocess.PIPE, stderr=subprocess.DEVNULL`（标记 `[GATE-FIX-H2-ARGS]`），保留原意 |

这正是把 `Accessory/` 纳入 `BEH-H2` 扫描的价值：**只做 `py_compile` 看不出这类错误**
（语法合法、运行必崩），而它恰好又被 `except` 掩盖了多年。

## 5. 门禁复跑结果（本机，2026-09-15）

```
BEH-E1  py_compile ×112 全通过                    PASS
BEH-E2  覆盖清单自检（112 文件，下限 95）          PASS
       自动收集 ✓ / FILES 具名文件全覆盖 ✓ /
       external 无未纳管包 ✓ / Accessory/ 全覆盖（59 个）✓
BEH-H1  实参名被目标签名接受（112 文件 / 245 调用点） WARN（缺 ffmpeg-python，见下）
BEH-H2  参数组合无互斥冲突                        PASS
汇总: 共 94 项 | 通过 88 | 失败 0 | 警告 3 | 跳过 3
```

`BEH-H1` 的 WARN 与基线一致（本机缺 `ffmpeg-python` → 1 个 `run_async` 调用点
的实参名无法校验），非本轮引入；Linux 侧装上 `ffmpeg-python` 后该组应转 PASS。

## 6. 仍需在 Linux（GPU）侧补跑

- [ ] `Accessory/` 全量纳入后，在装有 `torch` + `ffmpeg-python` 的机器上复跑门禁：
      期望 `BEH-H1` 由 WARN → PASS、`BEH-H3` 由 SKIP → PASS。
- [ ] 复核 `segment_bitstream_verify_v4.py` / `_v5.py` 的 git 跟踪状态（§3.2）。
- [ ] 按 §3.1 在 Linux 仓库侧执行换行符归一化（独立提交）。
- [ ] 待定项（15 个）按引用矩阵逐个确认后再决定归档/删除，**不做批量清理**。
