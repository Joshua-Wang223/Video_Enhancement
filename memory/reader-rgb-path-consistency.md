---
name: reader-rgb-path-consistency
description: NVDEC 与软解的 RGB 差异诊断 —— 实测纠正两个假设：-hwaccel auto 在无 GPU 机器上会假阳性退到 dxva2/d3d11va，且钉住矩阵/范围不需要 libzimg（输入侧 -color_range/-colorspace 就够）
type: project
---

# 硬件解码 vs 软解：`nv12→rgb24` 的 RGB 差异

立项：`Plan/NVDEC与软解RGB一致性_立项Prompt.md`
诊断脚本：`Accessory/probe/reader_rgb_path_diagnose.py`（2026-09-15 新增）

## 两条实测纠正（都会让立项文档给的复测命令得出错误结论）

### 1. `-hwaccel auto` 的返回码**不能**用来判定"NVDEC 可用"

2026-09-15 在**没有 NVIDIA GPU** 的 Windows 机器上实测：

```
-hwaccel cuda    → rc!=0  "Device creation failed: -1"   ✅ 正确判为不可用
-hwaccel auto    → rc=0   "Using auto hwaccel type dxva2" ❌ 假阳性
-hwaccel d3d11va → rc=0   "（WARP 支撑）"                  ❌ 同样不是 NVIDIA
```

立项 §4 的复测命令用的是 `-hwaccel auto`。在没有 NVIDIA 卡的机器上它会静默退到
别的后端并**返回成功**，于是"两条路径有差异"会被误当成 NVDEC 现象，
而"NVDEC 可用"会被误报。

⇒ 判活必须**点名逐个试**（`cuda` 优先）+ 检查 stderr 失败标记
（`Device creation failed` / `No device available` / `Hardware device setup failed` …），
并且**只有可用后端确实是 `cuda` 时才叫 NVDEC**。脚本里的
`_detect_hwaccel()` 就是这么做的。

### 2. 钉住矩阵/范围**不需要** libzimg

立项 §2.4 的两个判断都对：本容器的 `scale` 滤镜 `in_range`/`in_color_matrix` 是
no-op；该 build 也没有 `zscale`。但由此推出的"必须换带 libzimg 的 ffmpeg build"
**是多余的前提** —— 还有一条不需要任何滤镜的路：

**把 `-color_range` / `-colorspace` 放在 `-i` 之前**（输入侧选项），
直接覆盖解码器输出的颜色元数据，随后 ffmpeg 自动插入的 scaler 就会按这组参数做
`yuv→rgb`。实测它**真的生效**（Windows / ffmpeg N-122480，同一素材）：

```
默认(无元数据)                 sha=23cb17e921f8f3e2
-color_range tv                sha=23cb17e921f8f3e2   ← 与默认相同（默认就是 tv）
-color_range pc                sha=211007eee3bb31b1   ← 变了
-colorspace bt709              sha=c2b681994c1e5255   ← 变了
-colorspace bt601              → 直接报错：**没有 bt601 这个取值**
```

⚠️ 取值坑：601 系要用 `smpte170m`（NTSC）或 `bt470bg`（PAL），**没有 `bt601`**。
⚠️ 顺序坑：输入侧选项必须在 `-i` **之前**（与 `-fflags +genpts` 同一类坑）。

另外本 build 的 `zscale` 只暴露整数型 `range/primaries/transfer`，
**没有经典的 `matrixin=`/`matrixout=` 字符串选项** —— 照抄立项 §3 步骤 2 的
zscale 写法会直接 `Option not found`。

## 本机（Windows，无 NVIDIA GPU）实测到的定性一致性

用 `d3d11va` 作为"硬件路径"代跑（**不是 NVDEC，结论不可直接迁移**）：

| 判据 | 结果 |
|---|---|
| 1a 平面级（Y/UV） | 24 帧 **全部 0 差异** ⇒ 差异 100% 在 `nv12→rgb24`，与立项 §2.1 定性一致 |
| 1b RGB 级 | 最大逐像素差 **87**，不同像素 **43.03%** |
| 2 reader 忠实性 | IFRNet `FFmpegFrameReader` vs 同参数参考解码：**24 帧 0 不一致** ⇒ 忠实 |
| 3 参数穷举 | 8 个候选（4 矩阵 × 2 范围）**无一对 d3d11va 逐字节相等**（最接近的是 `smpte170m/tv`，43.14%） |

⇒ 定性现象（平面相同、RGB 不同、reader 忠实）在**非 NVDEC 的硬件后端**上也成立，
说明根因确实是"硬件路径与软解对 `nv12→rgb24` 的矩阵/范围选择不同"这一类问题。

## 仍需在 Linux + NVDEC 上做

1. `python Accessory/probe/reader_rgb_path_diagnose.py /tmp/clip_sd_25s.mp4 --sweep --json r.json --report r.md`
2. 从穷举里找与 NVDEC **逐字节相等**的那组 ⇒ 钉住 NVDEC 实际用的 range/matrix；
3. 判据 4（「对齐」/「声明不对齐」）二选一，写进验收说明并回填脚本生成的报告模板。

## 做 A/B 对照前必须先冻结路径

自适应策略（P2-2）会让低码率素材自动切到软解，改前改后跑同一素材**不是逐字节可比**：
`READER_HWACCEL=on` 强制 NVDEC（与旧行为一致）／`READER_HWACCEL=off` 强制软解。
本会话已发生过一次"把路径切换误判成回归"。

**Related**：[[p2-1-nv12-consumer-conversion-rejected]]、[[env-ffmpeg-ffprobe-gotchas]]、
[[feedback_no_gpu_work_mode]]
