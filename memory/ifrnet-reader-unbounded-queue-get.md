---
name: IFRNet 读帧器 read() 无界阻塞 —— 已修复（[FIX-READER-UNBOUND]）
description: 2026-09-15 落地：FFmpegFrameReader.read() 由裸 queue.get() 改为有界等待 + 存活判定，死亡路径 ≤1×T 抛、静默存活路径 ≤2×T 抛；含 IFRNET_READER_TIMEOUT 回滚开关与 CPU 回归测试
type: project
---

## 结论（2026-09-15 落地）

`external/ifrnet_video/ffmpeg_io.py::FFmpegFrameReader.read()` 不再是裸
`self._queue.get()`。现签名 `read(timeout: Optional[float] = None)`，行为：

| `timeout` | 语义 |
|---|---|
| `None`（默认） | 取环境变量 `IFRNET_READER_TIMEOUT`，缺省/非法 → **120s** |
| `<= 0` | **关闭看门狗**，退回原无界阻塞（现场对照 / 回滚开关） |
| `> 0` | 有界等待（见下） |

有界等待的两级判据（`_producer_state()`，刻意保守）：

* 先等 `timeout`；队列仍空 → 查存活。
* **已死**（`not thread.is_alive()` 或 `proc.poll() is not None`）→ 立即抛
  `RuntimeError`，消息含 `thread_alive / child_poll / queue=n/max / frame=N`。
* **仍活**（线程活 + 子进程在跑）→ 再给一个 `timeout` 观察窗（慢 ≠ 错，保住反压）；
  第二个窗口也空才抛，状态标 `producer_alive_but_silent_2xT`。

**为什么必须同时给"活着"留观察窗**：T1(读帧) 比 T2(推理) 快 25~30×，
`queue.get()` 的阻塞**就是**反压机制；把"暂时无帧"误判成失败会制造假失败。

## Why（2026-09-14 实测代价）

给门禁写 H3「读帧器冒烟」时，循环里的 deadline 只在两次 `read()` **之间**判断，
一旦卡进 `read()` 就再也出不来 —— **把 `verify_plan_implementation.py` 挂了 10 分钟**，
最终只能靠人工中断，并被迫改成「有界子进程 + 线程 join(60s)」双重设界才绕过。
那个复杂度本应属于读帧器自身。

## How to apply

* 新代码消费 `read()` 时**不必**再自己套 deadline/子进程 —— 它现在自己会抛。
  但仍要捕获 `RuntimeError`（它会冒泡到 `main.py` 的读帧循环）。
* 现场遇到"看起来卡住"时，异常消息里的 `thread_alive/child_poll/queue/frame`
  可直接区分三种成因：读线程死了 / 子 ffmpeg 退出没送哨兵 / 子进程活着但不出数据。
* **不要**把 ESRGAN 侧 `get_frame()` 的 `FRAME_TIMEOUT` 哨兵契约照搬过来，
  也不要把本侧改成返回哨兵：本侧消费方是
  `while True: pair = reader.read()`，返回 None 会被当成"正常 EOF"而**静默少帧**。
* 镜像提醒：两个后端的 `ffmpeg_io.py` 结构对称，但**这一处 API 语义有意不同**。

**相关**：[[env-ffmpeg-ffprobe-gotchas]]（SIGTTOU 停住子 ffmpeg 正是"活着但静默"的
典型成因）、[[gate-verify-plan-known-failures]]（BEH-H3 的隔离复杂度出处）。

## 验证状态

* **本机（Windows，无 GPU，无 torch/ffmpeg-python）**：
  `python tests/test_reader_unbound_watchdog.py` → **10/10 PASS**，
  含真实 ffmpeg 的 24 帧素材逐字节比对、反压慢消费帧守恒、以及两种注入
  （`_read_loop` 直接 return / 线程活着但永不产出）。
* **待 Linux（GPU）补跑**：真 `SIGSTOP` 子 ffmpeg 的变体；门禁 `BEH-H3`；
  一段真实素材端到端帧守恒（`--dry-run` + 解码级验收）。
