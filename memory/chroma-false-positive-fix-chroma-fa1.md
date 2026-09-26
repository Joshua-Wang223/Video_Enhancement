# Chroma 检查色度污染检测误报修复 (FIX-CHROMA-FA1)

**状态**: 已修复 (2026-08-20)，GPU 验证 ✅

## 症状

`segment_bitstream_verify_v4.py` v8 新增的色度检查 4 (`check_chroma_corruption`)
存在严重误报：正常视频（色度 std 稳定在 25-35 范围）因正常像素噪声波动
被判定为花屏坏帧，导致 `bad_count >= 3` → FAIL。

## 根因

`_chroma_postprocess` (line ~2562) 将**时间域跳变掩码** (`trans`) 直接并入
`bad_any` 作为坏帧判据：

```python
# BUG: 跳变掩码被并入坏帧判据
trans = np.zeros((n_frames, n_rois), dtype=bool)
trans[1:] = diff_u | diff_v       # diff_u: per-ROI 跳变掩码
bad_any |= trans                  # ← 任何跳变都标记为坏帧
```

跳变阈值 `max(median_diff * 8, 1.5)` 在 `median_diff` 极小时 (平画面/低噪声):
- `floor=1.5` 主导阈值
- 正常像素噪声引起的 std 波动 (1-3 点) 超过 1.5 → 触发跳变
- 跳变掩码 → bad_any → 误判为坏帧

**实测复现:** 随机游走噪声 (std=1.8) 视频: bad_count=10 (误报)

## 修复 (FIX-CHROMA-FA1)

**原则:** 时间域跳变仅用于**场景切割检测** (cut exemption)，不作为坏帧信号。
坏帧判据仅基于**绝对水平** (std > max(median * 1.6, floor))。

```python
# 修复后: 跳变不并入 bad_any，仅用于剪辑豁免
bad_any = bad_u | bad_v       # 仅绝对阈值
bad_any[cut] = False         # 剪辑豁免
```

同时修正 `cluster_regions` 攻击区域报告: 仅报告绝对阈值命中的 ROI，
不再包含跳变命中。

## 验证

| 场景 | 修复前 | 修复后 |
|------|--------|--------|
| 随机游走噪声 (std=1.8, 120帧) | bad_count=10 ❌ | bad_count=0 ✅ |
| 单帧 pop (+2 std) | bad_count=1 ❌ | bad_count=0 ✅ |
| 实际色度污染 (3x baseline, 3 簇) | bad_count=3 ✅ | bad_count=3 ✅ |
| 自然噪声视频 (end-to-end) | bad_count>=3 ❌ | bad_count=0 ✅ |
| 含确实花屏视频 | bad_count>=3 ✅ | bad_count>=3 ✅ |

## 影响

- 消除正常视频的色度检查误报
- 保持对真实色度污染 (U/V std 飙升 2-3 倍) 的检测能力
- 剪辑豁免逻辑增强: 跳变检测仅用于场景切割识别，更符合实际语义
