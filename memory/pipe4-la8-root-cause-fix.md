---
name: pipe4-la8-root-cause-fix
description: pipe=4+LA=8 终极根因与修复 — encode_frame+per-slot IDR 是唯一根因，handle共享不是根因，三文件三处修改已GPU验证
metadata: 
  node_type: memory
  type: project
  originSessionId: b6bcafb5-af2b-4b88-95e4-bcb2e53f3525
---

# pipe=4+LA=8: 终极根因与修复 (GPU 验证通过 ✅)

## 结论

**pipe=4+LA=8 完美支持，零花屏、零丢帧、零卡顿。** 所有 RC 模式 (CONSTQP/VBR_HQ/QVBR) 均安全。

## 根因链

```
encode_frame() 单独调用
  ↓ 使用 slot[0] legacy ref (_input_buf_handle = slots[0].input_buf)
  ↓ LA=8 buffering → NEED_MORE_INPUT → slot0 帧被 NVENC 内部持有
  
+ per-slot _slots_warmed force_idr ×4
  ↓ slot0 enc_frame IDR + slot1/slot2/slot3 ce_pipeline IDR → 5 IDRs/段
  ↓ 每个 IDR 触发 LA 帧类型重决策 → 5 × LA_depth = 40 帧 LA 开销
  
= 大量空帧 + slot0 buffer 混叠 + SPS 跨槽注入 → 丢帧 + 花屏 + 卡顿
```

## 三个根因与证据矩阵

| 根因 | T0 BUG 表现 | T4 pipe=1 对照 | T3 GOAL |
|------|-----------|---------------|---------|
| **encode_frame 混用** | 1 enc + 174 ce (slot0) | 1 enc + 696 ce (slot0) | 0 enc + 174 ce |
| **per-slot IDR ×4** | 5 IDRs: {0:2, 1:1, 2:1, 3:1} | 2 IDRs: {0:2} | 0 IDRs |
| **SPS 跨槽注入** | cache@slot0 + prepend@slot1 | cache+prepend@slot0 | 无 |
| **LA 开销** | 40 帧 (5×8) | 16 帧 (2×8) | 0 帧 |
| **结果** | 丢 31-57 帧 ❌ 花屏 | 丢 5-7 帧 ✅ 视频正常 | 零丢帧 ✅ |

## 关键洞察

**T4 (pipe=1) 证明 handle 共享不是根因**: slot0 被 encode_frame + ce_pipeline 共享 697 次，但 pipe=1 下视频正常。将 encode_frame+per-slot IDR 同时修复后 pipe=4 也正常。

## 修复方案 (T3 GOAL)

三个修改均已在 `v6.4.3.1/4.1/5.1` 中完成:

### Fix 1: 注释 pipe forced 4→1 guard
```python
# [FIX-PIPE4-LA8] 保留注释以备回退:
# if la_depth > 0 and pipeline_depth > 1:
#     if rate_mode == "constqp": ...
#     else: pipeline_depth = 1
```

### Fix 2: per-slot IDR → fi==0 only
```python
# 改前: force_idr = force_idr_first and (slot_idx not in _slots_warmed)
# 改后: force_idr = force_idr_first and (fi == 0)
# 删除: _slots_warmed = set() 及 _slots_warmed.add(slot_idx)
```

### Fix 3: Remove encode_frame() in _process_segment
```python
# 改前: encode_frame(first_nv12, force_idr=...) → writer.write(h264_data)
# 改后: pass  # 首帧已由 CE pipeline 统一处理
```

## 模拟验证方法论

`tests/test_pipe4_la8_corruption.py` — 差分分析模拟:
- 不模拟 NVENC 内部行为 (先前 4 个版本错误地尝试这样做)
- 只追踪 **可观测的语义差异**: 调用者、slot、IDR、SPS、handle 共享
- 6 个场景 (T0-T5) 的决策矩阵直接显示 T3=T5 REFERENCE → 确定修复方案

## 已应用文件

| 文件 | Fix 1 | Fix 2 | Fix 3 | 标签数 |
|------|:-----:|:-----:|:-----:|:-----:|
| `process_video_v6_4_5_1_single.py` | ✅ | ✅ | ✅ | 4× `[FIX-PIPE4-LA8]` |
| `process_video_v6_4_5_single.py` | ✅ | ✅ | ✅ | 3× `[FIX-PIPE4-LA8]` |
| `process_video_v6_4_4_1_single.py` | ✅ | ✅ | ✅ | 4× `[FIX-PIPE4-LA8]` |
| `process_video_v6_4_3_1_single.py` | ✅ | ✅ | ✅ | 4× `[FIX-PIPE4-LA8]` |

**Why:** encode_frame + per-slot IDR 是 pipe=4+LA=8 花屏/丢帧的唯一根因。handle 共享被 T4 pipe=1 证伪。差分模拟确定 T3 GOAL 方案，GPU 验证通过。
**How to apply:** 四文件已就地修复。搜索 `[FIX-PIPE4-LA8]` 标签定位所有修改点。旧 `encode_frames_batch` (死代码) 中的 `_slots_warmed` 残留安全忽略。
