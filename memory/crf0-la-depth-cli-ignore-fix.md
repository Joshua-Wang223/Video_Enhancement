---
name: crf0-la-depth-cli-ignore-fix
description: CRF=0 + FORCE_CONSTQP=False 分支硬编码 _NVENC_CRF0_LOOKAHEAD 无视 CLI --lookahead-depth-ifrnet；constqp 下 self._la_depth 未同步导致 Ready 日志误显 la=8
metadata: 
  node_type: memory
  type: project
  originSessionId: 4346bc88-8136-4059-8479-502e13b13fe7
---

# CRF=0 时 LA depth 无视 CLI 参数的三层 BUG (2026-07-17~18 修复)

## Bug A — 行为 BUG: CRF=0 分支硬编码 LA

`_process_segment` 中 Level 1 参数决策三分支，`elif self.crf == 0:` 分支曾写死
`_level1_la = _NVENC_CRF0_LOOKAHEAD` (=8)，无视 CLI `--lookahead-depth-ifrnet 0`
（CRF>0 的 else 分支一直是正确的 `getattr(self, '_la_depth', ...)`）。

**影响**: CRF=0 + `_NVENC_CRF0_FORCE_CONSTQP=False`（临时调试常改为 False）+ vbr_hq/qvbr 时，
实际编码用 LA=8 而非用户指定值 — 不只是显示错误。

**修复**: `_level1_la = getattr(self, '_la_depth', _NVENC_CRF0_LOOKAHEAD)`

## Bug B — 显示 BUG: constqp 清零只改局部变量

`NVENCEncoder.__init__` 中 `if rate_mode == "constqp": la_depth = 0` 只重置局部变量，
`self._la_depth` 仍保留传入值 (如 8) → Ready 日志行 `if self._la_depth > 0` 误显 `la=8`。
slot 计算用局部 `la_depth`，故 slots=4 正确，仅日志误导。

**修复**: 清零后追加 `self._la_depth = 0` 同步实例变量。

## Bug C — 决策层显示 BUG: constqp 时 _level1_la 未收敛 (2026-07-18)

Bug A 修复后 `_level1_la = getattr(self, '_la_depth', ...)` 能正确读取实例值，
但当 CLI 设 `--rate-mode-ifrnet constqp` 且未同时指定 `--lookahead-depth-ifrnet 0` 时，
config 默认值 8 仍会流入 `_level1_la` → `[NVENC]` 日志仍显示 `LA=8`。
虽然 NVENCEncoder 内部会清零 (Bug B 已修)，但决策层日志和 cache key 仍携带无意义的 LA 值。

**修复**: 三分支结束后、CRF0 日志前插入：
```python
if _level1_rate == "constqp":
    _level1_la = 0
```

## 三层修复协同验证 (constqp + CRF=0 + FORCE_CONSTQP=False)

| 层 | 变量/输出 | 修复前 | 修复后 |
|----|----------|--------|--------|
| 决策 | `[NVENC] ...LA=N` | `LA=8` | `LA=0` |
| 编码器 Init | `CONSTQP + LA=N: X slots` | `LA=0: 4 slots` ✅ | 不变 ✅ |
| 编码器 Ready | `la=N` | `la=8` | 无 la= 标签 ✅ |
| 硬件 | 实际 LA | 0 (slots=4 证明) | 0 ✅ |

## 修复范围

| 文件 | Bug A | Bug B |
|------|:---:|:---:|
| `process_video_v6_4_5_1_single.py` | ✅ | ✅ |
| `process_video_v6_4_4_1_single.py` | ✅ | ✅ |
| `process_video_v6_4_3_1_single.py` | ✅ | ✅ |

- 非 .1 版本 (v6_4_3/4/5) 是 CONSTQP-only 架构，NVENCEncoder 无 rate_mode/la_depth 参数，无此代码路径
- `external/realesrgan_video/nvenc_sdk.py` 逻辑正确：`_NVENC_CRF0_LOOKAHEAD=0` 且 `self._la_depth` 在所有调整后统一赋值 (line 525)
- backup* 快照目录有意不修改

**诊断入口**: 日志两处 LA 标记 — `[NVENC] CRF=0 ... LA=N`（决策层，Bug A 可见）和
`[NVENCEncoder] Ready: ... la=N`（编码器层，Bug B 可见）；`slots (HW pipeline buffers>=N)`
行反映的才是硬件真实 LA。

相关: [[v6.4.x-backport-fixes]], [[constqp-fast-path]], [[nvenc-ce-pipeline-architecture]]
