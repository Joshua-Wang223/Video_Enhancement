---
name: P2-1「reader 输出 nv12 + 消费端转换」已实测否决
description: numpy 在 4K 上做 nv12→rgb24 约 833ms/帧（14x 回归），读帧器降本只能靠 GPU 侧转换，且端到端收益≈0
type: project
---

结论：**不要**再提「把 `FFmpegFrameReader` 改成输出 nv12、在消费端做 nv12→rgb24」这条优化路线。已用真实消费者实测否决。

**Why（2026-09-14 实测，Tesla T4 + 8 vCPU，4K H.264 8.5Mbps，20s/480 帧，按帧读满管道）：**

| 方案 | 墙钟 |
|---|---|
| 当前 `-hwaccel_output_format nv12` + `-pix_fmt rgb24`（读+reshape） | 27.60s |
| cuda 输出 + `hwdownload,format=nv12` + **numpy 转换** | **400s（超时被杀，≈833ms/帧）** |
| 同上但不转换（`-pix_fmt nv12`） | 6.99s |
| 纯 CPU 软解 + rgb24（读+reshape） | 13.66s |

numpy 的色度 2× 上采样（`np.repeat`）+ 矩阵变换在 4K 上完全无法与 SIMD 的 swscale 竞争（我试过 LUT 查表版，同样慢）。另外 `scale_cuda=format=rgb24` 在本 build 直接 rc=218 失败，GPU 侧 nv12→rgb24 走 ffmpeg 滤波器也不通。

**How to apply:**
- 读帧路径要真降本，只有「帧留在 GPU（`-hwaccel_output_format cuda`）、用 torch/NPP 在 GPU 上转 RGB 并直接喂模型」这一条，属推理输入链路重构（读帧器 + 配对/填充 + 模型输入三处），不是加个开关能做的。
- 即便做成，**端到端收益≈0**：代码自述 T1(读取) 比 T2(推理) 快 25–30×（`external/ifrnet_video/pipeline.py` 的 `_compute_max_pair_queue` 注释），读帧根本不是瓶颈。
- 用户最初基于我给出的（合成脚本测出的）"2.0x"选择了「P2-1 也一起做、默认启用」。改用**真实 reader + 真实消费者**复测后校正为：4K 低码率纯 CPU 软解比 NVDEC 路径快 **1.58x**（26.35s → 16.73s，3 轮中位数）。教训：reader 类优化必须带真实消费端测量，合成脚本（灌 /dev/null）测的只是"生产者能力"，会严重高估。
- 替代方案已落地：P2-2 读帧器 hwaccel 自适应（`src/utils/reader_hwaccel.py`）。
