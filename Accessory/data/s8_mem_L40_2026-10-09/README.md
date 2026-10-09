# S8 内存采样原始数据 · L40 · 2026-10-09

`av1_pipeline_smoke.py --mem-dump-dir` 的逐进程 RSS 采样明细（每臂一个 `<rate_mode>.mem.tsv`）。
本目录是**执行证据**，配套报告在 `verification_report/`；叙述与判读见
[`memory/l40-full-verification-run-2026-10-09.md`](../../../memory/l40-full-verification-run-2026-10-09.md)。

采集环境：NVIDIA L40（sm89，UUID GPU-6a0d2f4a-…），FFmpeg 9.x，av1_nvenc。
统一参数：`--batch-size 8`、`--segment-duration 30`、`--mem-interval 5`、**`--mem-peak-mb 0`**
（默认 16000 MB 是 T4 / 720×576 / bs=24 标定值，L40 + 高分辨率下会假 FAIL，故本轮只判斜率）。

## 目录内容

| 子目录 | 素材 | 分辨率 | 结果 | 说明 |
|--------|------|--------|------|------|
| `new5/` | `input_videos/new5_raw.mp4` | 1920×1080 → 输出 **3840×2160** | 15 PASS / 1 FAIL | **S8 constqp +712.0 MB/min 已被证伪为测量假象**；见下「关键结论」 |
| `warm/` | 由 720×480 短片预热 | 720×480 | 7 PASS / 1 FAIL | 用途是**预热 TRT engine 缓存**（`H480_W736` / `H480_W720`），非验收项；其 S8 同为构建期台阶 |
| `maxbed/` | `input_videos/112 Max Bed Time.avi` | 720×480 | 0 PASS / 2 FAIL | 两臂 5.1s 速退：源 mp3 头损坏被 `[P5-FIX-SOURCE-STRUCT-GATE]` 硬拒 |

⚠ **本目录不含 `clifford/`**：`大红狗 Clifford the Big Red Dog DVDR.58.mp4`（720×576 / 18011 帧 / 720.5s）
已通过全部前置检查并选为长视频素材，但**开跑时 GPU 已掉卡**（`libnvidia-ml.so` 缺失），
脚本前置门禁以 `Cannot load libcuda.so.1` → **exit 2** 拒绝执行，**未产出任何数据**。
该素材的选型依据与恢复要点见 memory 条目 §5.1。

## 关键结论：S8 斜率不可直接采信 constqp 单臂

`new5/constqp.mem.tsv` 逐点 RSS 呈**阶跃**而非线性累积：

| t(s) | 0 | 55 | 334 | 390 | 614 | 670 | 835(end) |
|---|---|---|---|---|---|---|---|
| RSS MB | 8 | 2180 | 2423 | 5248 | 5495 | 9310 | 7073 |

台阶对应**一次性 TRT engine 构建**（IFRNet @55s、ESRGAN @390s）与编码阶段切换；每段平台期内
RSS 平坦到 ±5 MB。后半程 OLS 把台阶读成线性增长 ⇒ 假 FAIL。

**对照臂**（引擎已缓存，无构建开销）`new5/vbr.mem.tsv` 斜率 **−928.5 MB/min**（单调下降）⇒ 判定**无内存泄漏**。
这正是 `memory/av1-nvenc-l40-calibration.md`「⚠ 只跑 constqp 不足以判读，须 constqp,vbr 同素材对照」的设计意图。

复现要点：正式跑前先用同分辨率短片**预热 engine 缓存**，可让 constqp 臂直接拿到干净 S8。

## 文件格式

TSV，首行以 `#` 开头为表头：

```
# t_s  n_proc  rss_mb  pss_mb  rss_main_mb  rss_child_mb  gpu_tree_mib  procs_json
```

- `t_s`：秒（相对采样起点）
- `rss_mb`：主进程 + 子进程 RSS 之和（本目录各臂峰值 4.1–9.8 GB）
- `rss_main_mb` / `rss_child_mb`：主 / 子进程拆分
- `pss_mb`：按比例共享内存分摊
- `procs_json`：逐进程明细 `[{"pid":…, "rss_mb":…, "pss_mb":…, "tag":"main"|…}]`
- ⚠ `gpu_tree_mib` 恒为 0：watcher 未采到 NVML，**显存读数不可用**；管线自身日志报的 GPU 峰值可信