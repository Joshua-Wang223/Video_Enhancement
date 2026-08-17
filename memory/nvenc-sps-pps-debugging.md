---
name: nvenc-sps-pps-debugging
description: v6.4.3 跨段 NVENC SPS/PPS 缺失 — 根因已确认，修复已实施并 GPU 验证通过
metadata:
  node_type: memory
  type: project
  originSessionId: 22340d91-22c8-4473-ab56-1c8823f229a3
---

# v6.4.3 跨段 NVENC SPS/PPS 缺失 — 已修复并验证

**状态：** 根因已确认，修复已实施并 GPU 验证通过（2026-06-03）。

## 根因

Python ctypes struct `_NvEncConfigH264` 布局与 C 头文件 `nvEncodeAPI.h` 不匹配：

- C 结构体前 4 字节为 22 个 1-bit 位域（含 `repeatSPSPPS` @ bit 12）
- Python 结构体将每个位域当作独立的 `c_uint32`（各 4 字节）
- `enc_cfg.encodeCodecConfig.repeatSPSPPS = 1` 写入完全错误的偏移，NVENC 从未收到此配置
- 第一次修复（`repeatSPSPPS=1` + `FORCEIDR`）因此无效

C 头文件参考：`/tmp/nv-codec-headers/include/ffnvcodec/nvEncodeAPI.h` line 1805-1851

## 修复方案（v2 — 已验证）

不依赖 NVENC `repeatSPSPPS`，改为在 `NVENCEncoder` 层手动缓存和预挂 SPS/PPS：

1. **`__init__`** — 添加 `self._cached_sps_pps: Optional[bytes] = None`
2. **`_extract_sps_pps()`** — 静态方法，从 H.264 Annex B ES 中提取 SPS (nal_type=7) 和 PPS (nal_type=8) NAL 单元
3. **`encode_frame()`** — 首段首次编码时缓存 SPS/PPS；后续段 `force_idr` 帧预挂缓存的 SPS/PPS

## 修改文件

`external/IFRNet/process_video_v6_4_3_single.py`:
- ~line 950: `_cached_sps_pps` 字段
- ~line 1253: `_extract_sps_pps()` 静态方法
- ~line 1409-1415: 缓存/预挂逻辑

## 验证结果

- 多段视频编码通过，段 2+ 不再报 `non-existing PPS 0 referenced`
- 输出日志出现 `[NVENCEncoder] Cached SPS+PPS: XXX bytes`
- SPS 解析确认 `chroma_format_idc=1`（非 monochrome），视频颜色正常
