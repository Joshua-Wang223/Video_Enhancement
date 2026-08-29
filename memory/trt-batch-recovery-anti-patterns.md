---
name: trt-batch-recovery-anti-patterns
description: TRT OOM 降级后的 batch 恢复机制中的反模式和正确做法
metadata: 
  node_type: memory
  type: project
  tags: 
    - tensorrt
    - oom
    - batch-size
    - recovery
    - expandable-segments
  originSessionId: b9b77fa5-2bfd-4abb-a9a7-c262321e2a94
---

# TRT Batch 恢复反模式与正确设计

Real-ESRGAN TRT 推理中，engine B（编译时固定）≠ 运行时 B 时触发 padding，OOM 降级后 batch
恢复机制经历了多次迭代，发现并修复了以下反模式。

## 反模式 1：`_trt_working_bs = 0` 清零

```python
# ❌ 死亡螺旋
if _next_bs >= _trt_engine_B:
    self._trt_working_bs = 0  # 语义："恢复正常"

# 后果：下批 L726 `_is_trt and self._trt_working_bs > 0` 为 False
# → 从 optimal_batch_size=8 开始 → 立即 OOM → 降级 → 5 批 → 恢复 → 清零 → 循环
```

**修复**：`_trt_working_bs` 到达 engine_B 后保持非零值，继续走 tracking 路径，永不绕过。

## 反模式 2：恢复阈值过低

旧阈值 = 5 批。bs=4 时 ~0.3s（15fps）即触发恢复，显存压力远未缓解。生产日志中 4848→4880→4920 帧每 32 帧一次 OOM 循环。

**修复**：阈值提升到 50 批，确保真正稳定后才扩大。

## 反模式 3：恢复前无显存守卫

12.72/14.6 GB（87% full）时仍然尝试 bs=4→8 恢复，直接 OOM。

**修复**：恢复前检查 `torch.cuda.mem_get_info()`，`free < total * 0.20` 时跳过恢复。

## 反模式 4：expandable_segments 下每批 empty_cache()

```python
# ❌ 反效
if retry_bs < _trt_engine_B:
    torch.cuda.empty_cache()  # 释放整个池回 OS
# → 下次 alloc 走 cudaMallocAsync → 重新分配 → 增加碎片+延迟
```

`empty_cache()` 在 expandable_segments 模式下释放整个内存池回操作系统，下一次
分配需重新走 `cudaMallocAsync` 向驱动申请，适得其反。

**修复**：移除每批 `empty_cache()`，低频兜底由 `_batch_cnt % 100` 处理。

## 反模式 5：打印变量时序错误

```python
# ❌ 先赋值后打印，新旧值相同
self._trt_working_bs = _next_bs
print(f'bs={self._trt_working_bs}→{_next_bs}')  # 始终 X→X
```

**修复**：先保留旧值再赋值：`_old_bs = self._trt_working_bs`。

## 正确的恢复架构

```
OOM → 降级 bs ÷ 2
  → 稳定 50 批
  → 检查 mem_get_info(): free > 20% total?
      YES → bs × 2（渐进），cap at engine_B
      NO  → 重置 streak，维持当前 bs
  → _trt_working_bs 永不归零（到 engine_B 后保持 = engine_B）
  → 不在成功路径调用 empty_cache()
```

## 相关记忆

- [[expandable-segments-synchronize-after-del]] — synchronize 必须在 del 之后
- [[realesrgan-gpu-memory-accumulation-fixes]] — Real-ESRGAN 管线修复汇总
