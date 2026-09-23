---
name: ifrnet-f0-la0-double-consume
description: LA=0 路径每段少 1 帧——_NVENCEncodeThread._loop() 提前取走 _pending_f0_nv12，ce_pipeline 的 [FIX-F0-IN-BATCH-CE] 读到 None，首帧永久丢失，GPU 验证修复 ✅
metadata:
  node_type: memory
  type: project
---

# LA=0 每段少 1 帧 — f0 被双重消费

## 现象

h264_nvenc + constqp + LA=0，分段处理时每段输出比期望少 1 帧：

```
[GPU0] 完成 | 原始帧=349 → 输出帧=696 (期望≈697.0, 差1.0 ⚠️)
   ❌ 解码级验收失败: decoded=696 expected=697 reason=decoded_frame_mismatch
⚠️  片段 1 处理失败，终止后续处理
```

与 [[f0-first-frame-loss-ce-pipeline]] 的历史症状一致 —— 丢的正是首帧 f0。

## 根因：f0 交接点被两处争抢

`external/ifrnet_video/nvenc_sdk.py` `_NVENCEncodeThread._loop()`：

```python
# 循环之前（错误版本）
_pending_f0 = getattr(self._nvenc, '_pending_f0_nv12', None)
if _pending_f0 is not None:
    self._nvenc._pending_f0_nv12 = None      # ← 提前清空
    _f0_idr = getattr(self._nvenc, '_pending_f0_force_idr', False)
    self._nvenc._pending_f0_force_idr = False
```

而 `while True` 循环体内两条分支的 f0 交接点**并不相同**：

```python
if _la_mode:                       # LA>0：分块累积，f0 在首块 append 进 _acc_nv12
    if _first_batch:
        ...
        if _pending_f0 is not None:
            _acc_nv12.append(_pending_f0)
else:                              # LA=0：per-batch，f0 由下面这个方法自己读
    h264_list = self._nvenc.encode_frames_batch_ce_pipeline(nv12_list, force_idr)
```

`encode_frames_batch_ce_pipeline()` 内部有 `[FIX-F0-IN-BATCH-CE]`：

```python
_pending_f0 = getattr(self, '_pending_f0_nv12', None)   # ← 已被 _loop 清成 None
if _pending_f0 is not None:
    nv12_tensors = [_pending_f0] + list(nv12_tensors)
```

**LA=0 走到这里时 `_pending_f0_nv12` 已被 `_loop()` 取走并置 None
→ `[FIX-F0-IN-BATCH-CE]` 读到 None → f0 从未进入任何 batch → 首帧永久丢失。**

即：LA>0 分支需要 `_loop()` 代取，LA=0 分支需要 `_loop()` **不要碰**，
早期版本在循环前无条件代取，两条路径只能保住一条。

## 修复：把取用下放到 LA>0 分支内

```python
# 循环之前：仅声明占位，绝不清空
_pending_f0 = None
_f0_idr = False
```

```python
# LA>0 的 if _first_batch 分支内，真正取用
if _pending_f0 is None:
    _pending_f0 = getattr(self._nvenc, '_pending_f0_nv12', None)
    if _pending_f0 is not None:
        self._nvenc._pending_f0_nv12 = None
        _f0_idr = bool(getattr(self._nvenc, '_pending_f0_force_idr', False))
        self._nvenc._pending_f0_force_idr = False
if _pending_f0 is not None:
    _acc_nv12.append(_pending_f0)
```

LA=0 分支保持不动，`_pending_f0_nv12` 原样留给
`encode_frames_batch_ce_pipeline()` 的 `[FIX-F0-IN-BATCH-CE]`。

## 验证

修复后同一命令 3 段全通过，且不再有 ⚠️ 偏差：

```
原始帧=349 → 输出帧=697    ✅ 解码级验收通过: decoded=697 expected=697
原始帧=261 → 输出帧=521    ✅ 解码级验收通过: decoded=521 expected=521
原始帧=187 → 输出帧=373    ✅ 解码级验收通过: decoded=373 expected=373
```

segment_002 首帧包结构同时恢复正常：修复前是 `K`(374751B) + `K`(75301B) 两个
IDR，修复后是单个 `K`(78330B) + P 帧序列。

## 排查要点

- `realesrgan_video/nvenc_sdk.py` 有同名的 `_pending_f0_nv12` 读取代码，但
  该侧**从未给 `_pending_f0_nv12` 赋值**（全仓无赋值点），是死代码，不受影响。
- 帧数守恒诊断看两个数：`输出帧`（内部计数）与 `decoded`（ffprobe 实测）。
  两者同时少 1 → 帧真的没写出去（不是计数问题）；`输出帧` 多而 `decoded` 对
  → 写出去了空包（另一个 bug 形态）。

**Why:** 同一份"段首帧暂存"被两个不同 LA 路径以两套机制消费，交接点必须按
LA 分叉下放，不能在公共入口统一代取。

**How to apply:** 任何新增的"暂存到 encoder 上、由下游取用"的字段，都要确认
每条分支的取用点唯一且互斥；在公共入口统一清空前，先枚举所有消费方。

**Related:** [[f0-first-frame-loss-ce-pipeline]], [[ifrnet-f0-nv12-async-copy-race]], [[ce-pipeline-la-accumulation-pattern]], [[nvenc-la-frame-conservation-fix]]
