---
name: t2-static-estimation-undershoot
description: T2 静态估算 _T2_VAR_MS_TRT=25.0 严重低估 TRT 实际 ~340ms (偏差1039%)，导致首段队列饥饿；修复 _T2_VAR_MS_TRT≈335ms
metadata: 
  node_type: memory
  type: project
  originSessionId: b6bcafb5-af2b-4b88-95e4-bcb2e53f3525
---

# T2 静态估算严重低估 (30ms → 实测 340ms, 偏差 1039%)

## 现象

全部 6 个版本在 benchmark 首段 (Segment 1) 均报告:

```
[AUTO-TUNE-RETUNE] 实测 T2=335.9ms (全段 26 batches 中位数)
| 静态估算=30.0ms | 偏差=1020% | 当前 result_queue=8
| 校准建议 pair=8 result=8 (下次生效)
```

## 根因

T2 双分量模型 (`process_video_v6_4_5_1_single.py:3469-3473`):

```python
_T2_BASELINE_HWB  = float(576 * 736 * 24)   # 基准工作集
_T2_FIXED_MS      = 240.0   # eager/compile 路径固定 overhead（含 JIT）
_T2_FIXED_MS_TRT  = 5.0     # TRT 固定 overhead
_T2_VAR_MS        = 25.0    # ← 低估！对 TRT 路径实际 ~335ms
```

`_T2_VAR_MS = 25.0` 是为 eager/compile 路径校准的斜率参数。
**TRT 路径的实际 _T2_VAR_MS 约为 335ms** — 两者相差 13.4×。

计算过程 (L4463-4466):
```python
_fixed_ms = _T2_FIXED_MS_TRT if infer_be == 'trt' else _T2_FIXED_MS  # = 5.0
_t2b = (_fixed_ms + _T2_VAR_MS * _HWB / _T2_BASELINE_HWB) \
       / max(self._hw_profile.gpu_tier, 0.05)
# = (5.0 + 25.0 * 1.0) / 1.0 = 30.0ms  ← 严重低估！
_t2_estimated_ms = max(_t2b * _ifactor * _mfactor, 1.0)  # = 30.0
```

**而实测**: `_fixed_ms + _T2_VAR_MS * _HWB / _T2_BASELINE_HWB` 应等于 5.0 + 335.0 ≈ 340ms。

## 后果

初始 `pair_queue=5, result_queue=8` 基于 T2=30ms 计算 → 队列深严重不足 →
Segment 1 运行时 GPU 饥饿 (空闲占比 45-52%) → 首次运行性能远不如后续段。

RETUNE 在 Segment 1 完成后用实测值更新 `t2_measured_ms` 并写入 T2-CACHE →
后续段 (Segment 2+) 使用实测值 → 队列正确 → 但首次运行的开销已无法挽回。

## 修复方案

新增 TRT 专用 `_T2_VAR_MS_TRT` 常量，基于实测校准:

```python
# process_video_v6_4_5_1_single.py:3472-3473
_T2_FIXED_MS_TRT  = 5.0
_T2_VAR_MS        = 25.0     # eager/compile 路径 (保留不变)
_T2_VAR_MS_TRT    = 335.0    # [FIX-T2-VAR-TRT] TRT 路径基于 T4 实测 (Tesla T4, bs=24, 576×736)

# process_video_v6_4_5_1_single.py:4463-4464 (pipeline.run)
_fixed_ms = _T2_FIXED_MS_TRT if infer_be == 'trt' else _T2_FIXED_MS
_var_ms   = _T2_VAR_MS_TRT if infer_be == 'trt' else _T2_VAR_MS    # ← 新增
_t2b = (_fixed_ms + _var_ms * _HWB / _T2_BASELINE_HWB) \
       / max(self._hw_profile.gpu_tier, 0.05)

# process_video_v6_4_5_1_single.py:3852-3853 (_auto_queue_depths 同样需修复)
_fixed_ms = _T2_FIXED_MS_TRT if infer_backend == 'trt' else _T2_FIXED_MS
_var_ms   = _T2_VAR_MS_TRT if infer_backend == 'trt' else _T2_VAR_MS    # ← 新增
t2_base = (_fixed_ms + _var_ms * HWB / _T2_BASELINE_HWB) / max(profile.gpu_tier, 0.05)
```

## 影响范围

| 文件 | 行号 | 修复位置 |
|------|------|---------|
| `process_video_v6_4_5_1_single.py` | 3469-3473 | 新增 `_T2_VAR_MS_TRT` 常量 |
| `process_video_v6_4_5_1_single.py` | 4463-4464 | `Pipeline.run()` 中估算 |
| `process_video_v6_4_5_1_single.py` | 3852-3853 | `_auto_queue_depths()` 中估算 |
| `process_video_v6_4_5_single.py` | 对应位置 | 同样存在 |

**预期效果**: Segment 1 初始 T2 估算从 30.0ms → ~340ms，消除 1039% 偏差，
pair_queue 初始值从 5→8，result_queue 从 8→12，首段 GPU 利用率提升 ~20pp。

**Why:** TRT 路径的 T2 推理延迟远大于静态估算表中的 25ms 斜率参数，需 335ms 才能匹配实测
**How to apply:** 新增 `_T2_VAR_MS_TRT=335.0`，在 `_auto_queue_depths()` 和 `Pipeline.run()` 两处添加 `_var_ms` 分支选择
