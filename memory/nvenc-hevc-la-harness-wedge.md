---
name: nvenc-hevc-la-harness-wedge
description: test_nvenc_la_frame_conservation.py 的 harness HEVC 支持：曾因「无就绪门控 + 同步排空」在驱动内死锁；2026-09-18 已移植生产架构（FIFO+就绪门控+轮转序EOS）修复，h264 逐字节不变；并纠正「NVENC 引擎被卡死」实为 ffmpeg SIGTTOU 假象
type: project
---

# HEVC 测试 harness（`test_nvenc_la_frame_conservation.py`）：死锁 → 修复

**结论（2026-09-18，T4 / 驱动 580.65.06 / CUDA 13.0）**：
`MinimalTestEncoder` 现已支持 **HEVC LA=0 与 LA=8**，帧数守恒（编码级 + 解码级）；
h264 输出与修复前**逐字节相同**（md5 一致）。

## 曾经的根因（修复前）

harness 是「1 bs_buf/slot + 与提交同步排空 + 无就绪门控 + 无提交前背压」：
- `_lock_bitstream_once()` 以 `doNotWait=0` 锁当前输出槽，无 HEVC 就绪判断；
- HEVC 驱动对**空/未就绪槽**的 `LockBitstream` 在 `doNotWait=0/1` 下**均永不返回**；
- HEVC 输出延迟（LA=0 约 4 帧，LA>0 为 la+2）> 槽数时，槽内 bs_buf 在旧帧取走前被复用覆盖 → 丢帧。

实测栈：主线程 R 态 100% CPU，gdb 落在 `libnvidia-encode → libnvcuvid`；
或阻塞在 `_lock_bitstream_once`（`_drain_outputs` / `flush_eos` / `close`）。

## 修复（移植生产 IFRNet 架构，`[FIX-HEVC-READY]`）

对照 `external/ifrnet_video/nvenc_sdk.py`：
1. **槽数**：hevc/av1 取 `max(la+3, 6)`（la=0 下限 6 覆盖 ~4 帧延迟）；h264 仍 `max(1, la+1)`。
2. **per-slot FIFO**（`_slot_pending`）+ **提交前背压** `_ensure_slot_free()`
   —— 带 pending 绝不复用物理槽；就绪不足直接 **fail loud**（不静默丢帧/不挂死）。
3. **就绪门控** `_hevc_ready_count()`：`ready = submitted - la - margin - drained`，
   margin＝LA>0 时 2、LA=0 时 4；只在上界内 Lock（h264 路径原样不变）。
4. **EOS**：只排空有 pending 的槽，**保持 `_output_slot_idx` 轮转序**（不是槽号升序），
   非阻塞锁 + 墙钟截止 + 队首出队。
5. **size 钳制**（铁律 3）：`size@36 > max(64K, W*H*4)` 视为非法 → 不 `from_address`（防 SIGSEGV）。
6. **close()**：hevc/av1 只锁有 pending 的槽（原先遍历全部槽会锁空槽 → 永久阻塞）。
7. 修 `total_need_more` 未初始化（h264 因空槽返回 SUCCESS+size=0 从不触发，HEVC 会 AttributeError）。

验证：h264 LA=8/100 帧 md5 与修复前一致；hevc LA=0、LA=8 各 300 帧 → 300==300，
`ffprobe -count_frames=300`、`ffmpeg -v error` 零输出（无 POC 报错）。

## ⚠️ 纠正：当初「宿主 NVENC 引擎被卡死」是**误判**

调查中一度以为 harness 死锁把整个宿主 NVENC 卡死（连 h264、甚至 ffmpeg CPU 编码都挂），
并据此走了「杀进程 → GPU reset」流程。**该结论已被证伪**：

- 真元凶是容器 ffmpeg 的 **SIGTTOU 假象**（见 [[env-ffmpeg-ffprobe-gotchas]]）：
  后台进程组下 ffmpeg 对 fd0 调 `ioctl(TCSETS)` 被 `SIGTTOU` 停住 → 秒卡、0% CPU、无输出。
- 证据：`ffmpeg ... < /dev/null` 后 **h264_nvenc / hevc_nvenc 均 rc=0**；纯 CPU 的
  `libx264` 不 detach stdin 时同样挂 → 与 NVENC 无关。
- 容器内 `nvidia-smi --gpu-reset` 确实被拒（persistence mode 卡 Enabled，`-pm 0` 报 Unknown Error），
  但这不代表引擎坏了。

**教训**：凡「ffmpeg 挂起」先按 [[env-ffmpeg-ffprobe-gotchas]] 加 `< /dev/null` 复测。

## How to apply

- harness 现在可直接跑 hevc：`python tests/test_nvenc_la_frame_conservation.py --codec hevc --la-depth 8`。
- 修改 harness 排空逻辑时，务必保持三条铁律：**不带 pending 复用物理槽**、
  **不 Lock 未就绪/空槽**、**EOS 按轮转序排空**。
- 生产侧同名加固见 [[realesrgan-missing-la-redrain]]（`[FIX-HEVC-READY]` /
  `[FIX-LA-SLOT-HEADROOM]`）与 [[nvenc-eos-drain-slot-order-tail-corruption]]。

## 关联
- [[realesrgan-missing-la-redrain]]、[[ifrnet-hevc-la-slot-headroom-deadlock]]、
  [[env-ffmpeg-ffprobe-gotchas]]、[[shared-gpu-host-concurrent-jobs]]、
  [[nvenc-eos-drain-slot-order-tail-corruption]]
