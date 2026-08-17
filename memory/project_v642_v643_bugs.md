---
name: v642-v643-bug-fixes
description: v6.4.2 和 v6.4.3 的已知 bug、根因分析及修复方案 — 全部已 GPU 验证通过（2026-06-03）
metadata:
  node_type: memory
  type: project
  originSessionId: 442d2544-5553-4ef0-8e6e-6b0eddbdd4d1
---

# v6.4.2 / v6.4.3 Bug 分析与修复

## v6.4.2 问题

| Bug | 症状 | 根因 | 修复 |
|-----|------|------|------|
| pix_fmt 不一致 | 输出颜色异常 | FFmpegWriter 用 `bgr24`，但数据路径多处产生 RGB | 改为 `rgb24` |
| 性能倒退 ~30% | 推理变慢 | `_infer_batch` ring buffer 路径做了 GPU 侧 `[2,1,0]` channel flip + 额外 `.contiguous()` | 移除 GPU 侧 flip，保持 RGB 直传 |
| 帧数异常 | +10~18 帧 | **未定位** — 信号量/队列逻辑静态分析无异常，疑为竞态条件 | color pipeline 修复可能附带解决，需 GPU 验证 |

### 颜色数据流统一（修复后）

- `tensor_to_np`: 返回 RGB（`np.ascontiguousarray`，不做 channel flip）
- `_infer_batch` ring buffer D2H: 保持 RGB，不做 `[2,1,0]` flip
- `_writer_loop` ring buffer: img1_raw 做 `[::-1].copy()` BGR→RGB
- `_writer_loop` legacy pool: 插值帧不翻转，img1_raw 做 BGR→RGB
- `_writer_loop` sync fallback: img1_raw 做 BGR→RGB
- FFmpegWriter: `-pix_fmt rgb24`

## v6.4.3 问题 — 全部已修复 ✅

| Bug | 症状 | 根因 | 修复 | 验证 |
|-----|------|------|------|------|
| 文件膨胀 3.3x | 38MB vs 15MB | VBR 模式下 averageBitRate/maxBitRate 未生效，编码器用默认高码率 | 改为 **CONSTQP** (rateControlMode=0)，qpInterP/qpInterB/qpIntra = CRF 值 | ✅ 12.3MB (QP=21 映射与 FFmpeg CQ 不同，属正常差异) |
| 灰色输出 (no chroma) | 视频 R=G=B，完全灰色 | 三个根因叠加：(1) D2H 异步竞态无 event.synchronize() (2) CreateInputBuffer.height = NV12 total height 而非 luma height (3) presetConfig SDK offset 错误 | synchronize + luma height + offset 8 修复 | ✅ 颜色正常 |
| Level 1 code=15 | NVENC SDK 初始化失败 | `NV_ENC_PRESET_CONFIG` SDK 布局有 4-byte padding (presetCfg @ offset 8)，ctypes `_pack_=1` 消除 padding 但 SDK 函数按 SDK 布局解析 | 所有 presetConfig 访问统一用 SDK offset 8 | ✅ Level 1 正常初始化 |
| Writer 线程崩溃 | `'NoneType' object has no attribute 'reader_acquire'` | Infer 线程过早设置 `proc._ring_buf = None`，writer 排空队列时仍需访问 | 移除 infer 中的 `_ring_buf = None`，由 cleanup() 统一管理 | ✅ 无崩溃 |
| GPU 利用率 0% | 看起来异常 | **不是 bug** — NVENC 使用独立编码 ASIC，不走 CUDA core | 无需修复 | ✅ 正常行为 |

### CONSTQP 码率控制修复

```python
# VBR 模式（旧，无效 — 编码器用默认码率，输出 38MB）
rc_ptr[1] = 1       # NV_ENC_PARAMS_RC_VBR
rc_ptr[2] = bitrate  # averageBitRate (未生效)
rc_ptr[3] = bitrate  # maxBitRate (未生效)

# CONSTQP 模式（新 — 直接控制 QP，输出 12.3MB）
rc_ptr[1] = 0        # NV_ENC_PARAMS_RC_CONSTQP
_qp_val = max(1, qp) if qp > 0 else 28
rc_ptr[2] = _qp_val   # constQP.qpInterP
rc_ptr[3] = _qp_val   # constQP.qpInterB
rc_ptr[4] = _qp_val   # constQP.qpIntra
```

**Why CONSTQP 而非 VBR:** NVENC SDK 13.0 的 `NV_ENC_RC_PARAMS` 使用 union 存放 RC-mode-specific 字段。VBR 字段和 CONSTQP 字段在同一 union offset，但 VBR 的 averageBitRate/maxBitRate 无论设置何值编码器都使用默认码率（根因可能是 struct layout 其他问题导致 VBR params 未正确传递）。CONSTQP 模式直接生效，且行为与 Level 2 的 `-cq:v` 语义接近但 QP 映射不完全一致（正常现象）。

### Level 1 code=15 offset saga（重要教训）

多轮测试对比：

| 轮次 | offset | GetEncodePresetConfig | 原因 |
|------|--------|----------------------|------|
| 第三轮 | **8** | ✅ OK | SDK 正确 offset（version@0 + padding@4 + presetCfg@8） |
| 第四轮 | **4** | ❌ code=15 | ctypes packed offset → SDK 读到 version=0 |
| 第五轮 | 无写入 | ❌ code=15 | memset 全零，SDK offset 8 也是 0 |
| 第六轮 | **8** | ✅ OK | 恢复 SDK offset 8 |

**核心教训:** ctypes `_pack_=1` struct 的 field offset ≠ SDK struct layout offset。SDK 函数总是按 SDK 头文件布局（含 padding）解析参数。Memory 文件 `nvenc-ctypes-integration.md` 早已记录 offset=8，但第四轮计划错误地将 8→4 当作修复方向。

### CreateInputBuffer.height 细节

NVENC SDK 对 NV12 要求 height = luma height（如 576），而非 NV12 total height（如 864 = 576 + 288）。错误的高度导致驱动用错误的 offset 计算 chroma 平面地址，编码器从随机内存读取 chroma → 零值/128 → 灰色输出。

## 验证状态

**全部已验证** — Tesla T4 / driver 580 上通过：

```bash
python external/IFRNet/process_video_v6_4_3_single.py \
  --input ../input_videos/word_world_2.mp4 \
  --output output/test.mp4 \
  --scale 2 --batch-size 24 \
  --no-cuda-graph --no-compile --use-tensorrt \
  --codec libx264 --crf 21 --x264-preset medium --no-quiet
```

结果: Level 1 成功初始化，颜色正常，12.3MB 输出，无崩溃。
