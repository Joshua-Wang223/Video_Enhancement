---
name: 本机是多会话共享的 GPU 主机，性能测量可能被他方任务污染
description: 同容器/同 GPU 上有其他会话在跑 4K nvinterpolate 任务；测量出现孤立异常值或 NVDEC 利用率偏低时先查有无并发任务
type: project
---

本机（`/workspace` 容器，单卡 Tesla T4）**不止本会话在用**。测量视频性能前必须先确认没有别的任务在占 GPU/CPU，否则会追出"幽灵回归"。

**Why:** 2026-09-14 本会话排查读帧器时，同一条命令反复测出 **~16s**（正常 ~0.95s），
三次高度一致（15.98/16.07/16.01s）。当时误以为是刚做的 stdin 加固引入了回归。
实际用 `ps` 查到另一个会话的活跃任务：

```
bash /workspace/interp_2x_safe.sh /workspace/input_videos/Earth.at.Night.in.Color.S02E01.mp4
  └─ ffmpeg -nostdin -hwaccel cuda -hwaccel_output_format cuda -ss 0 -i <同一 4K 素材> -t 300 \
       -filter_complex nvinterpolate=fps=source_fps*2,trim=start_frame=3,setpts=PTS-STARTPTS \
       -c:v hevc_nvenc -preset p5 -cq 25 -f mpegts /workspace/interp_2x/.../parts/p00000.ts.part
```
（启动于 09:07:20，`nvidia-smi` 显示 encoder 86%。）
外部任务结束后同命令稳定 0.93~0.95s × 5 轮 —— 16s 纯属争用，与代码无关。

**How to apply:**
1. 任何"性能异常"结论之前，先看 `nvidia-smi` 与 `ps -eo pid,etime,pcpu,comm --sort=-pcpu`，
   确认没有 `<别的工作目录>/*.sh`、`nvinterpolate`、`hevc_nvenc` 之类的并发任务。
2. 计时类测量要做**多轮重复取中位数**；本会话正是靠 3 轮重复才发现 16s 是稳定异常值
   （稳定异常 ⇒ 先怀疑环境；随机抖动 ⇒ 才怀疑代码）。
3. 共享主机上的 NVDEC/NVENC 利用率读数同样会被他方任务抬高或压低，**不要**在并发任务
   运行期间采信绝对利用率数字，也不要据此推翻既有结论。
4. 已在跑的他方任务**不要 kill**（可能是在跑生产或另一个实验）；只避让、等空闲再测。
