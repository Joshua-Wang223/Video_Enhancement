---
name: _rgb_to_nv12_gpu-bgr-optimization
description: _rgb_to_nv12_gpu 的 input_is_bgr 参数可跳过 BGR→RGB 翻转，避免不必要的色彩空间转换
metadata: 
  node_type: memory
  type: project
  originSessionId: b8e08761-2f4c-4d02-8181-03638348a934
---

`external/IFRNet/process_video_v6_4_3_single.py` 中的 `_rgb_to_nv12_gpu(rgb_tensor, input_is_bgr: bool = False)` 函数自带 BGR 支持。当 `input_is_bgr=True` 时，函数内部直接从 BGR 通道顺序计算 Y/Cb/Cr（`r=ch2, g=ch1, b=ch0`），无需调用者先做 `[:, :, ::-1].copy()` 翻转。

**Why:** reader 输出的 `img1_raw` 是 BGR 格式（OpenCV 默认），直接传入 `input_is_bgr=True` 可以省掉 numpy BGR→RGB 翻转（`img1_raw[bi][:, :, ::-1].copy()`）这一步 CPU 拷贝。

**How to apply:** 所有调用 `_rgb_to_nv12_gpu` 的地方，如果输入 tensor 来自 reader（BGR），传 `input_is_bgr=True`；如果来自 IFRNet 模型输出（RGB）或 H2D prefetch `to_rgb=True`（RGB），传 `input_is_bgr=False`（默认值）。文件中共有三处调用点：
1. GPU-STAY 路径 `interp` — 模型输出 RGB → `input_is_bgr=False` ✓
2. GPU-STAY 路径 `img1` — prefetch `to_rgb=True` RGB → `input_is_bgr=False` ✓
3. 旧 NVENC 路径 `img1_raw` — reader BGR → `input_is_bgr=True` ✓

Related: [[level1-nvenc-encoding-flow]] [[nvenc-ctypes-integration]]
