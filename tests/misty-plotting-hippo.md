# IFRNet v6.4.3+ 修复计划：综合 Claude Web 方案与 ESRGAN 经验

## 一、方案对比：Claude Web vs 本分析

### Claude Web 修复方案（`process_video_v6_4_5_1_single_fix.py`）

Claude Web 针对 **单一症状（GPU OOM）** 做了精确定位修复：

| 修复内容 | 涉及代码 | 覆盖版本 |
|---------|---------|---------|
| `_writer_loop` D2H 搬迁 | NV12 GPU tensor → pinned host memory | 仅 v6.4.5.1 |
| `encode_frames_batch` cuMemcpy2D 源类型自适应 | HOST/DEVICE 动态选择 | 仅 v6.4.5.1 |
| 提交后立即释放 tensor 引用 | `nv12_tensors[fi] = None` | 仅 v6.4.5.1 |
| 空批次EOS修复 | **未包含** | — |
| Slot背压修复 | **未包含** | — |

### 本分析方案

基于 ESRGAN 4 个 bug 的修复经验 + 6 个 IFRNet 版本的架构分析，扩展为：

| 修复内容 | 涉及代码 | 覆盖版本 |
|---------|---------|---------|
| `_writer_loop` D2H 搬迁 | 同 Claude Web | v6.4.3.1, v6.4.4.1, v6.4.5.1 |
| cuMemcpy2D 源类型自适应 | 同 Claude Web | v6.4.3.1, v6.4.4.1, v6.4.5.1 |
| 提交后释放 tensor 引用 | 同 Claude Web | v6.4.3.1, v6.4.4.1, v6.4.5.1 |
| 空批次EOS修复 | ESRGAN Bug 4 经验移植 | v6.4.3.1, v6.4.4.1, v6.4.5.1（有 `send_eos` 的版本） |
| Slot背压修复 | ESRGAN Bug 3 经验 | **本轮不实施**（架构差异大，需单独设计） |

### 关键差异总结

1. **版本覆盖**：Claude Web 仅修 v6.4.5.1；本方案覆盖 3 个 LA 累积版本
2. **Bug 覆盖**：Claude Web 仅修 OOM（Bug 5）；本方案同时修空批次EOS（Bug 4）
3. **修复来源**：Claude Web 基于 OOM 日志直接定位；本方案额外参考 ESRGAN 4 个 bug 的修复模式
4. **v6.4.3.1 特殊性**：该版本没有 `_NVENCEncodeThread`，LA 累积直接在 `_writer_loop` 中进行（变量名 `_la_acc_nv12`），需要**手动适配**而非机械移植

## 二、版本架构总览（来自实际代码分析）

```
v6.4.3    → 无 _NVENCEncodeThread，无 LA 累积，无 send_eos → 仅需防御性 Fix 2
v6.4.3.1  → 无 _NVENCEncodeThread，有 LA 累积(_la_acc_nv12)，有 send_eos → 需 Fix 1+2（手动适配）
v6.4.4    → 有 _NVENCEncodeThread，无 LA 累积，无 send_eos → 仅需防御性 Fix 2
v6.4.4.1  → 有 _NVENCEncodeThread，有 LA 累积(_acc_nv12)，有 send_eos → 需 Fix 1+2（机械移植）
v6.4.5    → 有 _NVENCEncodeThread，无 LA 累积，无 send_eos → 仅需防御性 Fix 2
v6.4.5.1  → 有 _NVENCEncodeThread，有 LA 累积(_acc_nv12)，有 send_eos → 需 Fix 1+2（参考实现）
```

## 三、最终修复方案

### Fix 1: `[FIX-LA-ACC-HOST]` — LA 累积 GPU 显存 OOM 修复

**适用版本**：v6.4.3.1, v6.4.4.1, v6.4.5.1（有 LA 累积的版本）

**根因**：`[FIX-LA-ACCUMULATE]` 为了保证 LA FIFO 连续性，将整个 segment 的所有 NV12 帧作为 GPU tensor 累积在列表中，7000+ 帧 × (H+H/2)×W 字节 = 8-10GB+ 显存，与 T2 推理争抢显存，随 segment 推进阶梯状上升直至 OOM。

**修复包含 3 个子修改**（参考 `process_video_v6_4_5_1_single_fix.py`）：

#### 子修改 A：`_writer_loop` 中 NV12 D2H 搬迁

在 `_rgb_to_nv12_gpu_batch(all_frames)` 和 `torch.cuda.current_stream().synchronize()` 之后、编码线程提交之前，插入：

```python
# [FIX-LA-ACC-HOST] LA>0 累积模式：将 NV12 帧搬迁到 pinned host 内存
if _nvenc._la_depth > 0:
    # [FIX-LA-ACC-HOST-V2] 直接分配 pinned 缓冲区，一次 D2H 拷贝落地
    try:
        _pinned = torch.empty(all_nv12.shape, dtype=all_nv12.dtype,
                               device='cpu', pin_memory=True)
        _pinned.copy_(all_nv12, non_blocking=False)
        all_nv12 = _pinned
    except RuntimeError:
        # pinned 分配失败 → 退化为普通 pageable 内存
        all_nv12 = all_nv12.cpu()
```

并在 `_enc_thread.submit()` 之后释放 GPU 源引用：
```python
if _nvenc._la_depth > 0:
    all_frames = None
```

**v6.4.3.1 特别注意**：该版本没有 `_NVENCEncodeThread`，LA 累积直接在 `_writer_loop` 中进行（使用 `_la_acc_nv12` 而非 `_acc_nv12`）。D2H 搬迁后应 `_la_acc_nv12.extend(all_nv12)`（host 版本），而非原来的 GPU tensor。

#### 子修改 B：`encode_frames_batch()` cuMemcpy2D 源类型自适应

将硬编码的 DEVICE-only cuMemcpy2D 改为根据 tensor 位置动态选择：

```python
# 原代码（所有版本）:
src_ptr = nv12_tensors[fi].data_ptr()
cast(byref(_cpy2d, 16), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
cast(byref(_cpy2d, 32), ctypes.POINTER(c_void_p))[0] = c_void_p(src_ptr)

# 修复后:
_src_t = nv12_tensors[fi]
_src_is_cuda = _src_t.is_cuda
src_ptr = _src_t.data_ptr()
if _src_is_cuda:
    cast(byref(_cpy2d, 16), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_DEVICE
    cast(byref(_cpy2d, 32), ctypes.POINTER(c_void_p))[0] = c_void_p(src_ptr)  # srcDevice
else:
    cast(byref(_cpy2d, 16), ctypes.POINTER(c_uint32))[0] = _CU_MEMORYTYPE_HOST
    cast(byref(_cpy2d, 24), ctypes.POINTER(c_void_p))[0] = c_void_p(src_ptr)  # srcHost
```

需确保 `_CU_MEMORYTYPE_HOST = 1` 常量已定义（靠近已有的 `_CU_MEMORYTYPE_DEVICE = 2`）。

#### 子修改 C：提交后立即释放 tensor 引用

cuMemcpy2D 成功后、UnlockInputBuffer 之后，添加：
```python
nv12_tensors[fi] = None  # 立即释放引用，降低峰值内存
```

这使已提交帧的 host 内存随该批次 storage 引用归零而立即释放，而非等整个 segment 编码完成。

### Fix 2: `[FIX-EOS-EMPTY-CHUNK]` — 空批次不跳过 send_eos

**适用版本**：v6.4.3.1, v6.4.4.1, v6.4.5.1（有 `send_eos` 参数的版本）

**根因**（来自 ESRGAN Bug 4 经验）：`encode_frames_batch()` 中 `if n_frames == 0: return []` 无条件返回，当 segment 总帧数恰好被 chunk size 整除导致最终块为空时，`send_eos=True` 被完全无视，EOS picture 从未发给驱动，LA 缓冲帧滞留在硬件中。

**注意**：IFRNet 的全集累积模式意味着 `_acc_nv12` 始终有帧（`n_frames > 0`），所以这个 bug 在 IFRNet 中**概率极低**。但作为防御性修复，参考 ESRGAN 的经验教训，仍应修正此代码模式防止未来架构变更引入此问题。

**修复**（一行修改）：
```python
# Before:
if n_frames == 0:
    return []

# After:
if n_frames == 0 and not send_eos:
    return []
```

## 四、实施计划

### Phase 1：v6.4.5.1（参考实现，优先级最高）

**文件**：`external/IFRNet/process_video_v6_4_5_1_single.py`

1. Fix 2：修改 `encode_frames_batch()` 第 1717 行空批次守卫（1 行）
2. Fix 1-A：在 `_writer_loop` 中插入 D2H 搬迁块（~15 行新代码）
3. Fix 1-A：在 `_writer_loop` 中插入 GPU 引用释放（3 行）
4. Fix 1-B：修改 `encode_frames_batch()` cuMemcpy2D 块（~10 行替换）
5. Fix 1-C：在 cuMemcpy2D 后插入 tensor 释放（1 行）
6. Fix 1-B：同步修改 `encode_frames_batch_ce_pipeline()` cuMemcpy2D（同样模式）
7. 语法检查

**验证**：修复后的 diff 与 `process_video_v6_4_5_1_single_fix.py` 对比，关键修改区域应一致。

### Phase 2：v6.4.4.1（机械移植）

**文件**：`external/IFRNet/process_video_v6_4_4_1_single.py`

架构与 v6.4.5.1 完全相同（`_NVENCEncodeThread` + `_acc_nv12`），行号不同但代码结构一致。使用 grep 定位对应位置后逐处移植 Phase 1 的修改。语法检查。

### Phase 3：v6.4.3.1（手动适配）

**文件**：`external/IFRNet/process_video_v6_4_3_1_single.py`

**关键差异**：该版本**没有** `_NVENCEncodeThread`。LA 累积直接在 `_writer_loop` 中进行：
- 变量名是 `_la_acc_nv12`（非 `_acc_nv12`）
- 编码调用是 `self._nvenc.encode_frames_batch(_la_acc_nv12, ..., send_eos=True)` 直接在 writer 线程中

**适配要点**：
1. Fix 2：`encode_frames_batch()` 空批次守卫（与 Phase 1 相同）
2. Fix 1-A：D2H 搬迁插入点不同——在 `_rgb_to_nv12_gpu_batch` 之后、`_la_acc_nv12.extend()` 之前
3. Fix 1-B、1-C：`encode_frames_batch()` 内的修改与 Phase 1 相同
4. 语法检查

**风险**：v6.4.3.1 的 writer loop 结构与其他版本不同，需仔细定位插入点。

### Phase 4：v6.4.3, v6.4.4, v6.4.5（仅防御性 Fix 2）

这三个版本**没有 LA 累积**（无 `_acc_nv12`/`_la_acc_nv12`），且 `encode_frames_batch()` **没有** `send_eos` 参数。应用 Fix 2 需要：
1. 在函数签名添加 `send_eos: bool = False` 参数
2. 修改空批次守卫为 `if n_frames == 0 and not send_eos:`

由于没有任何代码路径传入 `send_eos=True`，此修改是**纯防御性**的，确保未来如果这些版本被升级支持 LA 累积时不会引入此 bug。

## 五、不纳入本轮的内容

### Slot 背压修复（ESRGAN Bug 3）

**不纳入原因**：
1. IFRNet 的 LA 累积模式将所有帧一次性提交到 `encode_frames_batch()`，不存在 ESRGAN 那种跨 chunk 的 slot 复用竞态
2. 实现需要将 `_slot_pending` 从局部变量提升为类级持久化结构 + 新增 `_ensure_slot_free()` 方法 + 与现有 Phase 1/2 提交流程集成——每个版本约 100+ 行改动
3. 在 Bug 5（OOM）和 Bug 4（EOS）修复后，IFRNet 的帧守恒性已得到保证
4. 建议作为独立的后续任务，单独设计、实施和测试

## 六、验证方法

1. **每个文件语法检查**：`python -m py_compile <file>`
2. **v6.4.5.1 diff 验证**：与 `process_video_v6_4_5_1_single_fix.py` 对比关键修改
3. **生产验证**（需用户在 Linux GPU 环境执行）：
   - 跑一个 2 分段的短样本验证无语法/导入错误
   - 用 `nvidia-smi -l 1` 监控 GPU 显存，验证不再阶梯状上升
   - 验证分段输出帧数守恒（submitted == written）
