# 立项 Prompt：IFRNet 读帧器 `read()` 无界阻塞（缺看门狗与可诊断性）

> ## ✅ 执行状态（2026-09-17 Linux + GPU 完成）
>
> **本立项已完成**，标记 `[FIX-READER-UNBOUND]`。
>
> | 项 | 结果 |
> |---|---|
> | `read()` 有界等待 + 存活判定 | ✅ 已落地（`external/ifrnet_video/ffmpeg_io.py`） |
> | `IFRNET_READER_TIMEOUT`（缺省 120s；`0`＝关闭看门狗，退回无界阻塞） | ✅ 已落地 |
> | 遥测 `_frames_read` + 归因消息（thread/child/queue/frame） | ✅ 已落地 |
> | 反压语义 | ✅ 未破坏（判据保守：线程活+子进程在跑 → 再给一个观察窗，≤2×T 才抛） |
> | 验收判据 1/2/4 | ✅ **Linux 已跑通**（`tests/test_reader_unbound_watchdog.py` 11/11 PASS，含真实 ffmpeg 24 帧逐字节比对 + 慢消费反压帧守恒） |
> | 验收判据 3（注入） | ✅ **Linux 真变体已跑通**（`SIGSTOP` 真子进程变体 `test_real_sigstop_child_ffmpeg_raises`，1.00s ≈ 2×T） |
> | 验收判据 5/6（门禁 BEH-H3 / 生产回归） | ✅ **门禁 BEH-H3 PASS**，生产回归端到端已验证 |
> | 完整测试脚本 | ✅ `tests/test_reader_unbound_watchdog.py` (11/11 PASS，含真实 SIGSTOP 变体 C3) |
>
> **落地形态与本文档 §3 的差异**：判别「已死」后**立即抛**，判别「仍活」时
> **再给一个 `timeout` 观察窗**（本文档 §3 只写了"否则继续等"）——
> 否则 SIGTTOU 停住场景（线程活 + 子进程在跑 + 永无输出）仍会**无界挂死**，
> 与判据 3 冲突。故总上界为：死亡路径 ≈ `1×T`，静默存活路径 ≈ `2×T`，二者都有界。
>
> **Linux 全判据已通过（2026-09-17，Tesla T4 / CUDA 13.0）**：
> - `test_reader_unbound_watchdog.py` 11/11 PASS（含真实 SIGSTOP 变体 C3：1.00s ≈ 2×T）
> - 门禁 BEH-H3 PASS
> - 生产端到端帧守恒验证通过
>
> 用法：本文件可直接整段复制给新的 AI 会话作为任务书。行号/代码片段均于
> **2026-09-17 在 Linux 生产树逐行核对**（与 Windows 开发树逐字节对齐）。
> 立项时间：2026-09-15　立项人：门禁强化会话
> **完成时间：2026-09-17**　立项人：门禁强化会话
> 关联记忆：`memory/ifrnet-reader-unbounded-queue-get.md`
> 关联已落地项：`tests/verify_plan_implementation.py` 的 **BEH-H3**（读帧器冒烟）
> 之所以必须跑在「有界子进程 + 线程 join(60s)」里，就是被本缺陷逼出来的。

---

## ✅ 状态总览（动手前必读）

| 项 | 状态 | 说明 |
|---|---|---|
| IFRNet `read()` 加超时 + 生产者存活判定 | ⬜ **本立项核心待办** | 缺失，见 §2.1 |
| `_read_loop` 内 `stdout.read()` 可被卡死 | ⬜ 待办（与上面同批处理） | 见 §2.2 |
| ESRGAN 侧同类保护 | ✅ 已有 | `FRAME_TIMEOUT` 哨兵 + 消费端看门狗，见 §2.3 |
| 门禁冒烟侧规避 | ✅ 已落地 | BEH-H3 有界子进程 + 双 60s join |
| 诊断手段 | ⬜ 待办 | 卡住时无 slot/queue/进程状态可查，见 §2.4 |

---

## 0. 任务

让 `external/ifrnet_video/ffmpeg_io.py::FFmpegFrameReader.read()` 在「生产者已死或
已停止产出」时**有界地失败并给出可归因的错误**，而不是永久静默挂起；
同时不得破坏 T1(读帧)→T2(推理) 之间赖以工作的**反压语义**。

验收目标一句话：**正常路径逐帧行为与帧数完全不变；异常路径在可配置时限内抛出
带上下文的异常而非挂死。**

---

## 1. 环境与代码基线

- 本工程：`D:\Workspace_Python\Video_Enhancement\Video_Enhancement`（Windows 开发树，
  **非 git 仓库**）；生产在 Linux `/workspace/Video_Enhancement`。
- 相关文件与关键行（2026-09-15 核对）：

  | 位置 | 内容 |
  |---|---|
  | `external/ifrnet_video/ffmpeg_io.py:546` | `self._queue = queue.Queue(maxsize=max(prefetch, 4))` ← 有界队列，反压来源 |
  | 同文件 `:547` | `self._thread = threading.Thread(target=self._read_loop, daemon=True)` |
  | 同文件 `:562-587` | `_read_loop`：循环 `self._proc.stdout.read(fb - len(buf))`，正常/异常出口分别投递 `_SENTINEL` / Exception，**均在 `self._queue.put()`** |
  | 同文件 `:589-595` | `read()`：`item = self._queue.get()` ← **无 timeout** |
  | 同文件 `:597+` | `close()`：先 `self._proc.terminate()` 再收尾（这个顺序是对的，勿改） |
  | `external/realesrgan_video/ffmpeg_io.py:331` | `FRAME_TIMEOUT = object()` |
  | 同文件 `:661-665` | `get_frame()` 超时返回 `FRAME_TIMEOUT`（≠ None，消费端据此重试/看门狗） |

---

## 2. 已确认的事实（实测确立，无需重做，不得违反）

### 2.1 `read()` 是无界阻塞，且完全没有存活判定

```python
589    def read(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
590        item = self._queue.get()          # ← 无 timeout，无 is_alive 检查
591        if item is self._SENTINEL:
592            return None
593        if isinstance(item, Exception):
594            raise item
595        return item
```

后果：只要 `_read_loop` 不投递任何东西，调用方就**永久静默挂起**，无超时、无日志、
无法从现象区分「还在解码」与「已经死了」。

### 2.2 卡死的**主要**入口是 `_read_loop` 里的 `stdout.read()`

`_read_loop` 的异常/收尾出口都会投递哨兵（`:584-587`），所以「线程自己抛异常」
是安全的。真正危险的是**卡在 `self._proc.stdout.read()` 里不返回**：

- 子 ffmpeg 被 `SIGTTOU` 停住（本容器既有环境坑，见
  `memory/env-ffmpeg-ffprobe-gotchas.md`）→ 既不产出也不退出 → `stdout.read()` 永久阻塞。
- 子 ffmpeg 因码流/驱动问题死锁而不退出 → 同上。

此时线程**是活的**（`is_alive()==True`）、队列**是空的**，`read()` 就无限等下去。
⇒ 只加 `queue.get(timeout=)` 不足以定位，必须同时能识别「子进程/线程已停止推进」。

### 2.3 ESRGAN 侧已有同类保护（可对照移植，但**不可照抄 API**）

`FFmpegReader.get_frame()` 在队列暂空时返回 `FRAME_TIMEOUT` 哨兵（`:661-665`），
上游 `pipeline.py` 的正确消费姿势是「识别哨兵 → 重试」，并有连续错误上限
（`_max_consecutive_err = 3` 等，见 `ffmpeg_io.py:1243/1255/1345`）。
IFRNet 侧**没有任何对应物**。

### 2.4 现场证据（为什么这条立项不是纸上风险）

1. **2026-09-14 写门禁冒烟时挂了 10 分钟**：H3 最初直接调用 `read()`，
   整个门禁无输出卡死，最终只能靠人工中断（记忆 `ifrnet-reader-unbounded-queue-get.md`）。
2. 上述事件之后 H3 被改成「有界子进程 `run_cmd(timeout=240)` + 子进程内线程
   `join(60s)` **双重**设界」—— **门禁为绕开本缺陷付出了额外复杂度**，
   而这个复杂度本应属于读帧器自身。

---

## 3. 实施步骤（建议）

1. **加常量与存活判定**（`FFmpegFrameReader`）：
   - `self._read_timeout`（默认取环境变量 `IFRNET_READER_TIMEOUT`，缺省建议 **120s**：
     取值原则 = 「远大于单帧正常间隔（毫秒级），但远小于人工发现挂死的时间」）。
   - `read(timeout=None)`：`timeout` 显式传入时优先，便于门禁传小值。
2. **`read()` 改为有界等待 + 归因**（示意，实现时保留原返回契约）：

   ```python
   def read(self, timeout=None):
       t = self._read_timeout if timeout is None else timeout
       try:
           item = self._queue.get(timeout=t)
       except queue.Empty:
           # 队列空且超时：区分「仍在推进」与「已死」
           raise RuntimeError(
               '[FIX-READER-UNBOUND] 读帧器 %.0fs 无产出: '
               'thread_alive=%s, child_poll=%s, queue=%d/%d, frame=%d'
               % (t, self._thread.is_alive(),
                  self._proc.poll(), self._queue.qsize(),
                  self._queue.maxsize, self._frames_read))
       ...
   ```
   - 「仍在推进」的判据要保守：**只有** `not self._thread.is_alive()` 或
     `self._proc.poll() is not None`（子进程已退出却无哨兵）才应判定为死亡；
     否则继续等（保住反压，不引入假失败）。
   - 若确实需要阻塞等待又不想抛异常，可提供 `strict=False` 模式退回原行为，
     但**默认必须是抛**（默认安全）。
3. **计数与遥测**：维护 `self._frames_read`，让异常信息能回答
   「卡在第几帧、队列积压多少」。
4. **子进程侧兜底**（可选但推荐）：给 `_read_loop` 加「静默时长」看门狗——若
   `stdout.read()` 超过 `t` 无数据且 `self._proc.poll() is not None`，主动
   `put(SENTINEL)` 收尾；或直接在此条件下 `self._proc.kill()` 并投递异常。
   ⚠️ 不要用「定期轮询 stdout 非阻塞读」改写整条读流水（会破坏 `STDIN/STDOUT`
   拷帧的原子性假设，且性能敏感）。
5. **两者镜像**：本项目 IFRNet / Real-ESRGAN 双后端结构对称，改完务必确认
   ESRGAN 侧语义未被破坏（`get_frame()` 的 `FRAME_TIMEOUT` 契约**不要动**）。

---

## 4. 验收判据

| # | 判据 | 期望 |
|---|---|---|
| 1 | 正常路径 | 合成 24 帧素材，`read()` 逐帧返回，帧数 == `ffprobe -count_frames`，逐帧字节不变 |
| 2 | 死亡路径（注入） | 人为让 `_read_loop` 直接 `return`（不投递哨兵）→ `read()` 在 ≤ 限额内抛 `RuntimeError`，消息含 `thread_alive/child_poll/queue` |
| 3 | 子进程停住路径（注入） | `SIGSTOP` 子 ffmpeg（Linux）/ 等价手段 → `read()` 在 ≤ 限额内抛出而非挂死 |
| 4 | 反压未被破坏 | 用小于 `prefetch` 的消费速率跑长片，内存不增长、无丢帧、帧守恒 |
| 5 | 门禁 | `python tests/verify_plan_implementation.py --no-report-file` 无 FAIL；BEH-H3 仍 PASS（Linux，装有 torch/ffmpeg-python 时） |
| 6 | 生产回归 | `--dry-run` 正常；一段真实素材端到端帧守恒（解码级验收通过） |

---

## 5. 风险与回滚

- **最大风险 = 破坏反压**：T1 比 T2 快 25–30×，`queue.get()` 的阻塞**就是**背压机制。
  超时期间若误判为「死亡」会制造假失败。⇒ 死亡判据必须**同时**要求
  「线程不活」或「子进程已退出」，且默认超时取值足够大（≥60s）。
- 次要风险：`read()` 签名变化 —— 全仓检索调用点，保持 `timeout=None` 兼容。
- 回滚：本改动是纯 Python 分支，回滚 = 还原 `read()` 与新增常量；
  建议保留环境变量开关（如 `IFRNET_READER_TIMEOUT=0` 表示退回无界阻塞）便于现场对照。
- **无需 GPU** 即可完成 1~4 项（合成素材 + 注入）；第 5~6 项需在 Linux（GPU）上跑。
