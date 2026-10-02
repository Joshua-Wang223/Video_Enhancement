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