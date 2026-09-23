---
name: 容器 GPU 时有时无的判据
description: Video_Enhancement 容器可能被重建而未挂载 GPU；给出一眼判定的命令组合与已观测到的特征
type: project
---

`/workspace/Video_Enhancement` 所在容器**可能在重建后未挂载 GPU**，表现为"昨天还能跑 NVENC，今天全部失败"。

已观测特征（2026-09-08）：
- `ls /dev/nvidia*` → 无匹配；`/dev` 下只有 19 个基础节点，无 `nvidiactl`/`nvidia-uvm`
- `env` 中 `NVIDIA_VISIBLE_DEVICES=GPU-xxxx` **存在**（Pod 声明了 T4），但 `CUDA_VISIBLE_DEVICES=` 为空
- `nvidia-smi` → `couldn't find libnvidia-ml.so`，因为 `libnvidia-ml.so.580.65.06` 与 `libcuda.so.580.65.06` 都是 **0 字节桩**（真身是 580.126.20）
- `torch` → `CUDA initialization ... Error 304: OS call failed or operation not supported`
- `/proc/driver/nvidia/version` 仍显示宿主内核模块 580.65.06 —— **这个不能作为 GPU 可用的证据**
- 可能伴随 `/workspace/output_videos/...` 变成 `Read-only file system`

**Why:** 这属于基础设施层问题，容器内无法通过改 symlink 修复（缺设备节点是硬阻塞）。曾据此误判为"代码回归"。

**How to apply:** 跑测前先做环境体检；若 `/dev/nvidia*` 不存在，直接按 `feedback_no_gpu_work_mode.md` 切到纯修复模式，不要改系统库、不要反复重启跑测。
