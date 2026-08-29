---
name: esrgan-cross-segment-optimization-complete
description: ESRGAN 跨段切换全栈优化：GFPGAN 保活 + 跳过 reopen + 移除 FIX-SEG-START-SYNC，从几十秒级降至毫秒级
metadata: 
  node_type: memory
  type: project
  originSessionId: 0567f4ad-2f7a-4f67-a136-73bc484b316b
  modified: 2026-07-29T08:09:01.584Z
---

## 概述

2026-07-29 完成的 ESRGAN 跨段切换四级优化，将段切换耗时从"NVENC 会话全量重建 + GFPGAN 子进程重建 + 流水线对象重建"缩减为"仅流水线对象重建"，与 IFRNet 插帧侧的段切换体验对齐。

**优化前段切换耗时构成**：
- NVENC 每段 reopen（DestroyEncoder + 全量 session/slot/event/stream 重建）：~百毫秒级
- GFPGAN 子进程每段 spawn + TRT warmup：典型几十秒/段
- 段首 batch 同步 encode_frame 过渡：~百毫秒级

**优化后**：仅 pipeline 对象重建（毫秒级），其余全部跨段持续复用。

## 五个 Phase 总览

| Phase | 内容 | 文件 | 状态 |
|-------|------|------|------|
| A | GFPGAN 子进程跨段保活 | `pipeline.py`, `processor.py` | ✅ 生产验证 |
| B | 跳过 NVENC 每段 reopen（env var 回滚门） | `nvenc_sdk.py` | ✅ 生产验证 |
| C1 | 清理 env var 开关 + 删除 `reopen()` 死代码 | `nvenc_sdk.py` | ✅ 生产验证 |
| C2 | `_stream_begin` 封装（对齐 IFRNet Item 3） | `nvenc_sdk.py` | ✅ 生产验证 |
| C3 | Encoder 缓存 key 模式（对齐 IFRNet Item 5） | `main.py` | ✅ 生产验证 |
| D | 取消 FIX-SEG-START-SYNC（对齐 IFRNet Item 4） | `nvenc_sdk.py` | ✅ 生产验证 |

## Phase A：GFPGAN 子进程跨段保活

### 机制

- `pipeline.py` `DeepPipelineOptimizer.close()`：子进程存活时回注 `self.args._early_gfpgan_subprocess`，断开 pipeline 引用
- 下一段 `__init__` 已有的预启动消费路径（`is_alive() + 死亡回退`）天然接住回注的存活子进程
- `processor.py` `_process_segments` finally 块 + 单段路径 finally 块显式调用 `close_enhancer()` 统一关闭

### 关键代码

```
# pipeline.py close()
if self.gfpgan_subprocess.process.is_alive():
    self.args._early_gfpgan_subprocess = self.gfpgan_subprocess
    self.gfpgan_subprocess = None  # 断开引用防止重复处置
else:
    self.gfpgan_subprocess.close()  # 死亡回退

# processor.py _process_segments finally 块
self.close_enhancer()  # 统一关闭保活子进程
```

### 不改动
- `__init__` 预启动消费路径 — 已含 `is_alive()` + 死亡回退
- `_async_dispatcher.close()` — 每段新建 dispatcher，挂到同一子进程安全
- `GFPGANSubprocess.close()` — 已有 `is_alive()` 守卫，幂等安全

## Phase B-D：NVENC 跨段真复用 + 架构对齐

### 关键决策链

1. **Phase B**：`ESRGAN_NVENC_REOPEN_PER_SEGMENT=1` 环境变量作为回滚门，默认跳过 reopen
2. **Phase C1**：验证成功后清理回滚门，删除 `reopen()` 方法（~78 行死代码），简化 FIX-SKIP-REOPEN 为无条件跳过
3. **Phase C2**：新增 `NVENCEncoder._stream_begin(force=True)` 封装 per-segment 状态重置（`_slot_pending.clear()`, `_slots_warmed=set()`, `_prev_chunk_outputs=[]`），对齐 IFRNet 同名方法
4. **Phase C3**：`main.py` 加入 `(out_w, out_h, fps, preset, qp, rate_mode, la_depth)` 参数 key 一致性检查，对齐 IFRNet `_get_or_create_nvenc_encoder` 防御模式
5. **Phase D**：移除 FIX-SEG-START-SYNC（`_first_batch_sync` 变量 + `encode_frame` 逐帧同步路径），LA=0 段首 batch 直接走 CE pipeline，对齐 IFRNet 生产零崩溃验证

### 最终跨段架构

```
段边界处理：
  段末: EOS flush → _needs_reopen = True（仅簿记标记，不触发动作）
  段间: _NVENCEncodeThread 新建 → _stream_begin(force=True) 重置状态
       → _needs_reopen 清除 → CE pipeline 直接启动编码
```

### IFRNet v6.4.5.1 对比差异项处置

| Item | 描述 | 决策 | 理由 |
|------|------|------|------|
| 1 | Slot 未决表数据结构 (list vs dict) | 跳过 | 设计差异，无功能问题 |
| 2 | 双层帧号 (`_strm_next_fi` + `_frame_idx`) | 跳过 | ESRGAN 单计数器 + `% slot_count` 等效 |
| 3 | `_stream_begin` 封装 | ✅ 实施 | 封装分散重置，提升代码整洁度 |
| 4 | LA=0 首 batch 直接 CE pipeline | ✅ 实施 | 移除 FIX-SEG-START-SYNC |
| 5 | Encoder 缓存 key | ✅ 实施 | 参数 key 比较，防御性更强 |
| 6 | PinnedResultPool 跨段复用 | N/A | ESRGAN CE pipeline 不走 Pool 路径 |
| 7 | PinnedRingBuffer 跨段复用 | N/A | ESRGAN Level 1 不使用 RingBuffer |
| 8 | 段首 f0 强制 IDR (`force_idr_f0`) | 跳过 | `_slots_warmed=set()` 重置 + CE pipeline `force_idr_first` 等效且更保守 |
| 9 | Muxer life per-segment | 等效 | 两端都是每段新建 FFmpegMuxer |
| 10 | Encoder 生命周期 | 等效 | 两端都在全部段结束后统一 close |
| 11 | Writer 生命周期 | 等效 | 两端每段新建编码线程 |
| 12 | `_la_chunk_safe` 硬上限 | 跳过 | ESRGAN 硬上限 500 已比 IFRNet 768 更保守 |

## 修改的文件

| 文件 | Phase | 净变化 |
|------|-------|--------|
| `external/realesrgan_video/pipeline.py` | A | +8 行 |
| `src/processors/realesrgan_processor_video_optimized.py` | A | +20 行 |
| `external/realesrgan_video/nvenc_sdk.py` | B + C1 + C2 + D | -124 行 net（删 reopen + FIX-SEG-START-SYNC，增 _stream_begin） |
| `external/realesrgan_video/main.py` | C3 | +19 行 |

## 备份

| 日期 | 目录 | 用途 |
|------|------|------|
| 2026-07-29 09:43 | `backup202607290943/` | Phase A 改前 |
| 2026-07-29 11:18 | `backup202607291118/` | Phase B 改前 |
| 2026-07-29 14:35 | `backup202607291435/` | Phase C 改前 |
| 2026-07-29 15:29 | `backup202607291529/` | Phase D 改前 |

## 关键日志标识

```
[FIX-GFPGAN-KEEPALIVE] GFPGAN 子进程保活        # Phase A：段间子进程复用
[FIX-SKIP-REOPEN] 跨段真复用，会话跳过重建        # Phase B-D：驱动会话跨段持续
[NVENCEncoder] 复用编码器                        # Phase C3：encoder 参数一致复用
```

## 相关记忆

- [[esrgan-cross-segment-sigsegv-fix-stack]] — SIGSEGV 修复历史（已被本优化取代）
- [[esrgan-reopen-slot-phase-misalignment]] — reopen 时代相位错位问题（已随 reopen 删除而消失）
- [[esrgan-segment-reuse-frame-idx-reset]] — 旧架构 frame_idx 错误重置（已随 reopen 删除而消失）
- [[realesrgan-nvenc-module-architecture]] — ESRGAN NVENC 模块架构参考
- [[nvenc-ce-pipeline-architecture]] — CE pipeline 底层机制
- [[ifrnet-v6-4x-version-matrix]] — IFRNet 参考架构
