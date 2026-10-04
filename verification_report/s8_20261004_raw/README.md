# S8 内存采样原始数据（2026-10-04，T4）

本目录是 §9.5-D 项（constqp 泄漏能否在 T4 复现）跑批的**原始内存采样**，
是 `Plan/PROMPT_L40_AV1等质量标定专项执行方案.md` **§0.1** 全部斜率结论的证据来源。
**保留这些文件是为了让后续修复不必再花一次 GPU 上机时间**（`--mem-dump-dir` 的设计目的）。

## 采样器口径

`Accessory/verify/av1_pipeline_smoke.py` 的 `MemWatcher`，`--mem-interval 5`：

| 列 | 含义 |
|---|---|
| `t_s` | 距采样开始的秒数 |
| `n_proc` | 进程树进程数（判断子进程是否随时间增加） |
| `rss_mb` | 进程树 RSS **求和**（⚠ 共享页在每个进程各算一份，进程数越多越虚高） |
| `pss_mb` | 进程树 PSS 求和（`/proc/<pid>/smaps_rollup`，**跨进程可比，口径优先用这个**） |
| `rss_main_mb` / `rss_child_mb` | 主进程 / 子进程分组 RSS（泄漏归属） |
| `gpu_tree_mib` | ⚠ **恒为 0，勿用** —— 见下方「已知失效字段」 |
| `procs_json` | 逐进程明细 `[{"pid","rss_mb","pss_mb","tag"}]`，`tag ∈ {main,ffmpeg,ffprobe,other}` |

## 各文件对应的跑批

| 文件 | 臂 | rc | 时长 | 采样点 | 后半程 RSS 斜率 |
|---|---|---|---|---|---|
| `h264_constqp.mem.tsv` | h264_nvenc + constqp（LA=0 / `ce_pipeline`） | 0 | 2968s | 572 | **−55.6 MB/min** |
| `h264_vbr_FAILED.mem.tsv` | h264_nvenc + vbr | **1** | ~56s | **12** | ❌ **无效，见下** |
| `h264_vbrhq_FAILED.mem.tsv` | h264_nvenc + vbr_hq（LA=8 / `encode_frames_stream`） | **1** | ~57s | **12** | ❌ **无效，见下** |
| `hevc_constqp.mem.tsv` | hevc_nvenc + constqp（LA=0） | 0 | 2452s | 471 | **+3.1 MB/min** |
| `hevc_vbrhq.mem.tsv` | hevc_nvenc + vbr_hq（LA=8 / `encode_frames_stream`） | 0 | 3177s | 610 | **+35.8 MB/min** |

素材：`01 the race to mystery island.fixed.mp4`（358.76s / 720×576 / 12 段 / 8969 源帧），
`--segment-duration 30`，`batch_size=24`（⚠ 见下「可比性」）。

## ⚠ 三条使用纪律

**1. 两个 `*_FAILED.mem.tsv` 不是斜率证据。**
它们在 **61s / 62s 即 rc=1 崩溃**，有效样本仅 **12 点**（文件 13 行 = 1 表头 + 12 数据），
全部落在启动爬坡段，**零有效稳态样本**。复算它们的「后半程斜率」会得到
**+4585.8 / +1872.4 MB/min** 这种纯启动爬坡伪影（两者 rc 不同、数值也不同 ⇒ 不可解释为真实趋势）。
若只拿它们画图会得到假象。崩溃根因见 §0.1.3 的 **B1**
（h264 + LA>0 + 跨段复用 ⇒ 片段 2 起 muxer 收不到参数集），**与 S8 主题无关**。

**2. 换 `batch_size` 的读数不可比。** 本目录全部为 **bs=24**；冒烟脚本现已默认 bs=8
（实测 bs=8 单批 90ms vs bs=24 的 343~363ms、pinned 池 102MB vs 305MB）。
L40 那次 `+149.5 MB/min` 也是 bs=24 下测的 ⇒ **只有同为 bs=24 才能与它比**。

**3. 同一份数据换 OLS 窗口，斜率可差 3 个数量级。**
`hevc_constqp.mem.tsv` 实测：前 10% **+1434.2** / 后 50%（S8 判据口径）**+3.1** /
**全程 +149.4** / 稳态（剔前 40%）+36.1 / 后 25% −240.5 / 末 10% **−786.7** MB/min。
⚠ **「全程 +149.4」与 L40 记录的「+149.5」几乎相同** ⇒ L40 那个数**疑为窗口伪影**。
且后半程 sd ≈ 1356 MB（不确定度约 ±66 MB/min）⇒ `|slope| < 5` 的阈值都是随意的。
**用本目录数据复算时必须同时报窗口与噪声底噪。**

## 已知失效字段：`gpu_tree_mib`

**恒为 0，不代表「没占显存」。** 跑批期间整卡实测 7698 MiB / 100% 利用率。
真因：`nvidia-smi --query-compute-apps=pid` 返回的是**宿主机命名空间 pid**
（实测 725115 / 1071006），在容器 `/proc` 下**不存在**；容器内 `NSpid` 只有一层
⇒ 驱动侧与容器 PID namespace 不通，pid 查表永远 miss。
**这是"测不到"，不是"没有"** —— 报告里不得填 0 冒充无占用。
仅影响显存维度；RSS/PSS 走 `/proc`，口径正确。
