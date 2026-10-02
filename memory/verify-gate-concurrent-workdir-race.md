---
name: 验证门禁的 workdir 也是固定名 —— 并发会话会互相覆盖，「rc=0 但产物缺失」是signature
description: 2026-10-02 第八版落表后 verify_equal_quality.py 崩在 rav1e 段（moov atom not found / out.stat() FileNotFoundError），真因是另一会话并发跑同一条命令、同一 workdir；含证伪表值的取证顺序与「改表值≠门禁通过」纪律
type: project
---

## 现象与真因

`Accessory/verify/verify_equal_quality.py`（约 12 min）在 rav1e 段崩溃：

    moov atom not found / Error opening input file .../librav1e_63.mp4
    File ".../calibrate_equal_quality.py", line 209, in encode
        return out.stat().st_size, time.time() - t0
    FileNotFoundError: ... temp/verify_equal_quality/librav1e_63.mp4

**`ffmpeg` 返回 0 却没有产物** —— `run()` 只在 `returncode != 0` 时抛，
所以异常从 `out.stat()`（下一行）而非编码调用处冒出。
**真因＝另一会话并发跑同一条命令、同一 workdir** `temp/verify_equal_quality/`
（`ps` 见另一进程正写同目录的 `libaom-av1_26.mp4`），
文件被互相 unlink/覆盖 ⇒ 这是 `eqq-batch-measure-parallel-constraints` 那条
「同 workdir 绝对不能并行」的**验证侧变种**：标定 harness 的固定名是 `prep.mp4`，
验证脚本的固定名是 `temp/verify_equal_quality/<codec>_<q>.mp4` + `prep.mp4` + `vmaf.json`。

**Why this matters**：它发生在「刚改完等质量表、准备收尾」的时刻，
若不取证就改表值回退，会把一次环境竞争误判成标定缺陷并回退已验证的成果。

## 取证顺序（表值有变化时必须走完，否则等于用环境问题掩盖真缺陷）

第八版把 crf21 的 `librav1e` native 从 qp **66 改到 63**，这个改动是唯一的嫌疑点。
排除它的最短路径（不猜、直接复现）：

1. **用同一素材 + 同一 prep 单点复现** `encode(prep,'librav1e',63,out)`
   ⇒ **成功**（470 s，2.73 MB，rc=0）。
2. **再跑第七版的 qp=66** ⇒ 同样成功。
   ⇒ 两个版本的值都能正常编码 ⇒ **表值不是原因**。
3. 用 `ps -eo pcpu,cmd --sort=-pcpu` / `ps -eo cmd | grep '[v]erify_equal_quality'`
   查同名并发 ⇒ 命中另一会话 ⇒ 环境竞争定性。

⚠ 第2 步很关键：只跑新值成功还不足以排除，必须**新旧两个值都成功**，
否则「能编码」可能只是碰巧。

**How to apply**：
- **「rc=0 但输出文件不存在」= 并发覆盖的signature，不是编码失败的signature。**
  这类失败**不会**在编码调用处报错，只会延迟到下一次 `stat()` 才炸 ⇒ 容易误判方向。
- 改任何表值/参数后，**「单点能编码」不等于「门禁通过」** ——
  必须**重跑完整门禁**才算闭合。本例因 workdir 被占，门禁 1 留作待办并写清「不涉及表值变更」。
- 收尾期撞上并发会话：**先取证定性、再决定是否回滚**，不要用回退掩盖未诊断的失败。
- 监控 cron / 门禁重跑前，先确认目标 workdir 没有另一进程（`ps` 核真实 PID，
  不用 `pgrep -c`——它会匹配检查命令自身，见 [[eqq-batch-measure-parallel-constraints]]）。
- 共享主机的性能污染已有记录（[[shared-gpu-host-concurrent-jobs]]），
  本条是它对**验证/落表阶段**的补充：那类记录讲「测出幽灵回归」，本条讲「门禁崩在产物缺失」。

## ✅ 已闭合：独占重跑后 **5/5 达标**（2026-10-03 00:42）

结果（锚点 x264 crf21，VMAF 99.135）：

| 编码器 | q | ΔVMAF |
|---|---|---|
| libx265 | 21 | +0.183 |
| libvpx-vp9 | 26 | +0.636 |
| libaom-av1 | 26 | +0.185 |
| libsvtav1 | 29 | +0.287 |
| **librav1e** | **64** | **+0.482** |

⇒ **主门禁 5/5 达标，rc=0**（5 项 ΔPSNR/ΔPSNR-HVS 超界属参考项，不判红）。
**表值无需任何改动。**

⚠️ **两次崩溃的根因都不是表值/代码缺陷**，且**互相完全不同**：
1. 第一次＝**并发抢 CPU**（本容器 8 核/7 GB，`librav1e -qp 64` 180 帧 720p
   单点需 **7m44s**；同时跑其它门禁 ⇒ ffmpeg 非正常退出）；
2. 第二次＝**我自己补跑前 `rm -f temp/verify_equal_quality/*.mp4`**，
   把脚本**自建的 `prep.mp4`** 删了 ⇒ 第一步锚点编码就失败
   ⇒ **补跑前不要预删 workdir**。

⚠️ 排查中踩到的两个**自伤式判据**（都会把失败误判成成功）：
- **`$?` 取错进程的退出码**：`( ffmpeg ... | tail -3 ); echo $?` 拿到的是 `tail` 的 rc
  ⇒ **把 ffmpeg 失败误判成 rc=0**。正确写法：`cmd > log 2>&1; echo $?`。
  我据此一度错误排除了「编码失败」，白跑一轮。
- **`pgrep -cf <pat>` / `ps aux | grep -c '[m]easure_uni'` 会匹配到检查命令自身**
  ⇒ 报出**并不存在的进程**（实测报 2 个，实际 `ps -C python3` 为空）。
  可靠判据＝**`ps -C <进程名>`**（按名字匹配，不含自身）＋ 逐个读 `/proc/<pid>/cmdline`。
- **Python stdout 重定向到文件时行缓冲关闭** ⇒ 跑完前日志可长期 0 行，
  **「0 行」≠ 卡死**。判定顺序：查进程 → 查 ffmpeg CPU → 才看日志。

**How to apply（补强本文件主条目）**：并发/占用类失败的排查，
**每一步的判据本身要先自证**——「rc=0」「有进程在跑」「日志有输出」这三条都曾
在本次任务里给出过错误答案。判据被证伪时先怀疑判据，再怀疑被测对象。