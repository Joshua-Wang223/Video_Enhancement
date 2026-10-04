---
name: pipeline-thread-coordination
description: IFRNet 三线程流水线的线程同步、退出时序、队列管理的关键规则和已知竞态；⚠ 2026-10-04 复核：退出时序一节所述 b'FLUSH' 机制在活跃代码已不存在、其前提已被推翻，本文件仅线程协调与锁规则部分仍有效
metadata: 
  node_type: memory
  type: project
  originSessionId: 9ac38aa0-722c-43f8-aa1e-b6fad7621a9e
---

> ⚠ **2026-10-04 复核：下方「退出时序」一节已失效，勿照此改代码。**
> 该节要求「put FLUSH tuple → `(b'FLUSH', leftover)` 到 result_queue」，其前提是
> **LA flush 帧是花屏残片、需在 Writer 端丢弃** —— 该前提已被推翻：
> `[[la-flush-recovery-is-harmful]]` 证明正确排空逻辑下 flush 帧是完整有效帧，
> `[[nvenc-la-frame-conservation-fix]]` 已用 SDK 合规排空实现帧数守恒，
> 因此**不需要也不应**在 Writer 端丢弃 flush 帧。
> **代码证据**：活跃包 `external/ifrnet_video/` 与 `external/realesrgan_video/` 全库
> `grep -rn "b'FLUSH'"` = **0 命中**（仅历史单文件 `external/IFRNet/process_video_v6_4_*`
> 仍有，属 CODEBUDDY.md 所述无生产 import 的历史代码）。该节所有 `file:line` 亦全部指向历史文件。
> **仍有效的部分**：下方「NVENCEncoder 锁使用规则」（不可重入锁 + 严禁持锁调 flush）
> 与「Writer 循环静默退出条件」两段与 LA flush 结论无关，可继续参考。

# IFRNet 三线程流水线线程协调

## 线程架构

```
Reader 线程: NVDEC 解码 → pair_queue
Infer  线程: GPU 推理 → result_queue
Writer 线程: 写入/编码 → FFmpegMuxer / FFmpegWriter
```

## 退出时序（关键！）

Infer 线程 finally 块中的操作顺序非常重要：

1. **flush NVENC** → 获取 EOS 后残留的 H.264 数据
2. **put FLUSH tuple** → `(b'FLUSH', leftover)` 到 result_queue
3. **put SENTINEL** → 通知 Writer 退出
4. **最后才设 `self.running = False`** → Writer 循环条件 `while self.running or not self.result_queue.empty()`

**Why:** 如果 `self.running = False` 在 FLUSH/SENTINEL 之前设置，Writer 线程发现 `running=False AND result_queue.empty()` 会立即退出，导致 FLUSH bytes 永远无法被写入 muxer，输出文件损坏。

**验证:** `external/IFRNet/process_video_v6_4_3_single.py:3480-3500`

## NVENCEncoder 锁使用规则

- `NVENCEncoder._lock` 是 `threading.Lock()`（不可重入）
- `encode_frame()` 和 `flush()` 都通过 `with self._lock` 获取锁
- **严禁**在持有 `_lock` 的情况下调用 `self.flush()`（包括在 `close()` 中）
- `close()` 不应调用 `self.flush()`，flush 应在调用 close 之前显式执行

**Why:** 曾出现 self-deadlock：主线程调用 `close()` → 获取 `_lock` → 调用 `self.flush()` → flush 尝试获取 `_lock` → 永久阻塞。

## Writer 循环静默退出条件

```python
while self.running or not self.result_queue.empty():
```

只有当 **BOTH** `self.running=False` AND `result_queue.empty()` 两个条件同时满足时循环才退出。单独一个条件为 True 就继续循环。

## 看门狗机制

IDLE_DEADLOCK_TIMEOUT = 120s。Writer 线程在队列全空且未收到 SENTINEL 时启动计时，超时后强制退出并打印线程 dump。

**相关代码:** `external/IFRNet/process_video_v6_4_3_single.py:2764-2765, 2968-2992`
