---
name: expandable-segments-synchronize-after-del
description: expandable_segments 模式下 del 后的 synchronize 必须放在 del 之后而非之前
metadata: 
  node_type: memory
  type: reference
  tags: 
    - pytorch
    - cuda
    - memory
    - expandable-segments
    - synchronize
    - cudaFreeAsync
  originSessionId: b9b77fa5-2bfd-4abb-a9a7-c262321e2a94
---

# expandable_segments 模式下 synchronize 必须在 del 之后

## 规则

```python
# ❌ 错误：synchronize 在 del 之前
torch.cuda.synchronize(device)       # 只排空 compute/transfer 流
del batch_t, output_t, out_u8, ...
return sr_results                    # cudaFreeAsync 仍在排队，内存未归还

# ✅ 正确：synchronize 在 del 之后
del batch_t, output_t, out_u8, ...   # 触发 cudaFreeAsync 排队到 CUDA 流
torch.cuda.synchronize(device)       # 排空所有 pending cudaFreeAsync
return sr_results                    # 内存已归还，下一批 alloc 可复用
```

## 原理

`PYTORCH_ALLOC_CONF=expandable_segments:True` 启用 `cudaMallocAsync` / `cudaFreeAsync` 分配器。

- `torch.Tensor.__del__` 在 expandable_segments 模式下调用 **`cudaFreeAsync`**（非同步 `cudaFree`）
- `cudaFreeAsync` 只是**排队**到当前 CUDA 流，不立即归还内存
- `torch.cuda.synchronize()` 排空流上所有 pending 操作，**包括异步释放**
- 若 synchronize 在 del 之前，只排空计算/传输，不排空后续 del 触发的 free

## 后果

- 下一批 `alloc` 时 CUDA pool 检测到池内无可用块 → 向驱动请求新内存 → pool 膨胀
- 旧 `cudaFreeAsync` 最终完成 → allocated 短暂下降（锯齿下降沿）
- 但 pool 已膨胀，新基线高于旧基线 → **整体趋势单调上升**

## 适用范围

- 所有使用 `expandable_segments:True` 的 PyTorch 代码
- IFRNet 和 Real-ESRGAN 均受影响（同环境均出现阶梯状累积）
- 不是特定模块的 bug，是 expandable_segments 的系统性行为

## 相关记忆

- [[realesrgan-gpu-memory-accumulation-fixes]] — Real-ESRGAN 管线修复汇总
- [[trt-batch-recovery-anti-patterns]] — TRT batch 恢复的反模式
