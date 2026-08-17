---
name: esrgan-pinned-buffer-pool-race
description: PinnedBufferPool 无同步保护导致 D2H 读取早于异步拷贝完成，输出帧内容被旧批次顶替
metadata: 
  node_type: memory
  type: project
  status: fixed
  originSessionId: 279c3f6f-f525-402f-b6ed-7f6152663198
---

# ESRGAN Pinned Buffer Pool 竞态条件

## 症状

输出视频局部帧内容被更早批次顶替（帧重复/乱序），尤其在第一批 SR 推理后（GFPGAN TRT 验证阻塞 ~180s 放大了竞态窗口）。

`[DIAG] ⚠️ DUPLICATE-BATCH-DETECTED` 命中（`written_count=24` 的指纹与 `prev_seen_at_written=0` 一致），像素级比对确认输出第 24/25 帧与第 0/1 帧逐像素几乎相同。

## 根因

**文件**: `realesrgan_utils.py` + `pipeline.py`

`PinnedBufferPool` 输入/输出各只有**一块共享 buffer，完全没有任何同步保护**。调用方也用错了：

1. **输出端 (最直接的 bug)** — `pipeline.py` `_sr_infer_batch`:
```python
out_pinned.copy_(out_perm, non_blocking=True)   # 异步 D2H，刚发起
out_np = out_pinned.numpy()                     # 立刻读 — GPU 可能还没执行完！
sr_results = [out_np[i].copy() for i in range(B)]
```
`non_blocking=True` 的 D2H 拷贝发起后**没有任何等待**就转 numpy 读取 → 读到的是上一批遗留的旧数据。

2. **输入端** — `get_for_frames()` CPU 写入和上一次异步 H2D 读取竞争同一块内存。原有代码用了一个笨重的 `self.transfer_stream.synchronize()` 部分兜住，但不可靠。

3. 第一批后的 GFPGAN TRT post-SR 验证 (`wait_validate(timeout=180.0)`) 长时间阻塞，Reader/Detect 继续堆积帧到队列，SR 线程恢复后预取更激进，让竞态窗口必现。

## 修复

**文件 1: `realesrgan_utils.py`**
- 输入/输出各改为**双缓冲环形池**，每个 slot 关联一个 `torch.cuda.Event`
- 复用某个 slot 前，先 `event.synchronize()` 等待上一次关联的拷贝完成
- 调用方忘记配合调用 `mark_issued`/`mark_output_issued` 时，自动退化为整卡 `torch.cuda.synchronize()` 兜底

**文件 2: `pipeline.py`** — 三处配合修改：
1. **输入端**: H2D 发起后调用 `pool.mark_issued(transfer_stream)` (记录 event)
2. **输出端** (根因修复): D2H 发起后调用 `pool.mark_output_issued()` + `pool.wait_output_ready()`，确保读 numpy 前拷贝已落地
3. **预取**: 去掉笨重的 `self.transfer_stream.synchronize()`，改用池内精确 per-slot event 等待 → 恢复了预取本该有的重叠性能收益

## 关联

这是 ESRGAN pipeline 层独有的 bug，IFRNet pipeline 不使用 PinnedBufferPool。
