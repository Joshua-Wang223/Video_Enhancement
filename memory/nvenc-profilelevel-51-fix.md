---
name: nvenc-profilelevel-51-fix
description: NVENC profileLevel=51 (H.264 L5.1) 不生效完整修复演进 — ctypes 结构体布局系统性错位 884B 根因链 + FIX-SDK13-CODEC 硬编码偏移修复 + sweep 三重旁证
metadata:
  node_type: memory
  type: reference
---

# NVENC profileLevel=51 修复完整演进（2026-08-07）

## 背景

生产 60fps 时 NVENC 自动选 Level 6.1，超出 Tesla T4 NVDEC 上限 5.2 → NVDEC 硬解回退（`cuvidCreateDecoder` 返回 `CUDA_ERROR_INVALID_VALUE`）。

## 第一轮失败（结构体写入，驱动从未收到）

7 个规范文件（IFRNet v6.4.3/3.1/4/4.1/5/5.1 + ESRGAN nvenc_sdk.py）添加：

```python
enc_cfg.encodeCodecConfig.profileLevel = 51
```

**但经生产 ctypes 结构体写入实际落在 `NV_ENC_CONFIG.reserved[278]` 保留区（绝对 1068），驱动从未收到** → 60fps 仍自动选 Level 6.1。

## 真根因：`_NvEncConfig` 布局系统性错位 884B

`_NvEncConfig.encodeCodecConfig@1052` 是错误布局（SDK 13.0 真实位置 @168 = 绝对 176）：

- 前 28 字节（version/profileGUID/gopLength/frameIntervalP）恰好正确
- 从 @28 开始错位：monoChrome→frameFieldMode、mvPrecision→frameFieldMode_1、**rcParams 区被 reserved3[53] 吞掉**
- reserved5[172] 把 encodeCodecConfig 推到 1052

**教训**：rcParams 一直用硬编码偏移访问（正确）而 codec 字段走结构体（错误），同一结构体两套访问方式并存导致此 bug 潜伏。

## FIX-SDK13-CODEC：硬编码绝对偏移写入

对齐 rcParams 既有模式 `cast(byref(preset_config, 偏移), ctypes.POINTER(c_uint32))[0]`：

```python
_h264_cfg_off = 8 + 168
# chromaFormatIDC@192: 1=4:2:0 [FIX-CHROMA]
cast(byref(preset_config, _h264_cfg_off + 192), ctypes.POINTER(c_uint32))[0] = 1
# idrPeriod@8: IDR 周期 = fps
cast(byref(preset_config, _h264_cfg_off + 8), ctypes.POINTER(c_uint32))[0] = int(fps)
# maxNumRefFrames@60: 4 [FIX-GPU-STAY] 首次真正生效
cast(byref(preset_config, _h264_cfg_off + 60), ctypes.POINTER(c_uint32))[0] = 4
# profileLevel@4: 51 (L5.1)
cast(byref(preset_config, _h264_cfg_off + 4), ctypes.POINTER(c_uint32))[0] = 51
# repeatSPSPPS 不写（_apply_sps_pps 手动注入保持现状）
```

## sweep 三重旁证（tests/diagnose_profilelevel_offset.py --sweep）

| 写入偏移 | 结果 |
|---------|------|
| 绝对 176（config 起始） | InitializeEncoder 失败（写坏） |
| **绝对 180** | **level_idc=51 命中 ✅** |
| 绝对 200（spsId） | SPS 内容变化但 level_idc=31 |
| 绝对 208 | SPS 内容变化但 level_idc=31 |
| 其他候选（164/168/172/184/188/192/196/204/208） | 写入无效，同 baseline |

## 影响文件（7 个规范文件 × 2 处修改）

- codec 写入段（原 5 行结构体赋值 → 注释 `[FIX-SDK13-CODEC]` + 硬编码写入）
- DIAG-LEVEL 日志：`_req_lvl` 由 `enc_cfg.encodeCodecConfig.profileLevel` 改为固定 `51` 回显

覆盖文件：IFRNet `process_video_v6_4_3/_3_1/_4/_4_1/_5/_5_1_single.py` + ESRGAN `external/realesrgan_video/nvenc_sdk.py`

## 验证清单（生产实机）

1. 首帧日志 `[NVENCEncoder] H.264 SPS actual` 显示 `level_idc=51`（不再 6.1）
2. NVDEC 硬解无回退（`cuvidCreateDecoder` 不再返回 `CUDA_ERROR_INVALID_VALUE`）
3. 段 2+ 无花屏/丢帧回归（注意 `maxNumRefFrames=4` 首次真正生效）
4. 可选：`repeatSPSPPS` 原生能力是否可恢复（当前刻意保持手动注入不变）

## 关联

- 布局参考：[[nvenc_ctypes_verified_layouts]]（NV_ENC_CONFIG_H264 完整偏移表）
- 诊断脚本：`tests/diagnose_profilelevel_offset.py`
