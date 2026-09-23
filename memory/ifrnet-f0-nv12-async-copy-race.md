---
name: ifrnet-f0-nv12-async-copy-race
description: 段首帧麻布状花屏根因——f0 的 _rgb_to_nv12_gpu 异步产出与 NVENC 私有拷贝流 _stream_encode 之间无依赖边，GPU 验证修复 ✅
metadata:
  node_type: memory
  type: project
---

# IFRNet 段首帧麻布状花屏 — NV12 异步产出 vs NVENC 私有拷贝流竞态

## 现象

`new5.mp4` + h264_nvenc + constqp + LA=0（分段 9s，3 段），两轮复现：
**concat 输出第 1219 帧（1-indexed）= 0-indexed 1218 = segment_002 的第 0 帧**
上半幅出现规则麻布/网格状噪声，下半幅基本正常。

## 定位手法（可复用）

高频梯度能量扫描：逐帧算 `mean(|diff(gray,axis=1)|) + mean(|diff(gray,axis=0)|)`，
分别统计上/下半幅，以 `max(median*3, p99*1.5)` 为阈值。

| 指标 | SEG1（干净） | SEG2（花屏 frame 0） |
|---|---|---|
| 首帧 IDR 大小 | 71,737 B | **374,751 B**（5.2×） |
| 上半幅梯度能量 | 8.2 | **150.9**（全片中位 7.6，20×） |
| SPS | `67 64 00 33 ac 2b 28…` | **逐字节相同** |

**判读要点**：CONSTQP QP=21 下真实画面不可能有 5 倍体积差 —— 只有把**噪声**喂进
编码器才会这样（噪声不可压缩）。SPS 逐字节相同则排除参数集问题。
→ 结论：NVENC 输入面上的数据本身就是垃圾。

## 根因

`external/ifrnet_video/main.py` `_process_segment()` 首帧分支：

```python
first_gpu  = torch.from_numpy(first).cuda()
first_nv12 = _rgb_to_nv12_gpu(first_gpu, input_is_bgr=False)   # 异步！十余个 kernel
_nvenc_encoder._pending_f0_nv12 = first_nv12                    # 直接交给编码器
```

- `_rgb_to_nv12_gpu()` 在 **PyTorch 当前流**上异步排入十余个 kernel
  （Y/Cb/Cr 计算、2×2 降采样、UV 交织、`torch.cat`）；
- NVENC 的输入拷贝走 `self._stream_encode` —— 一条
  `cuStreamCreate(NON_BLOCKING)` **私有流**（`[FIX-ASYNC-COPY]`）；
- **两条流之间没有任何依赖边**：`cuStreamSynchronize(self._stream_encode)`
  只等私有流自己的拷贝，**不等** PyTorch 流上的 NV12 kernel。

段首窗口最危险：f0 的 NV12 kernel 刚入队，紧接着就是 PinnedResultPool 重建
+ 流水线冷启动，等到首个 batch 把 f0 取走编码时 kernel 可能仍未落盘
→ NVENC 读到未初始化/半写完的输入面 → 段首帧编码成噪声。

### 为什么其它路径没这个问题（三处对照）

| 路径 | 是否有同步 |
|---|---|
| 批量路径 `pipeline.py` `[FIX-ENC-THREAD]`（`_rgb_to_nv12_gpu_batch` 后） | ✅ `torch.cuda.current_stream().synchronize()`，注释即"防止 GPU 数据未就绪导致静默花帧" |
| ESRGAN `nvenc_sdk.py` 的 `_rgb_to_nv12_gpu` 调用点 | ✅ 同样有（两处） |
| **IFRNet 的 f0 路径** | ❌ **漏了** |

`[FIX-ASYNC-COPY]` 本身是正确优化（消除 legacy null stream 的全局隐式同步），
但它**移除了**原本由 null stream 免费提供的跨流顺序保证，却没有用事件补上。

## 修复

```python
first_nv12 = _rgb_to_nv12_gpu(first_gpu, input_is_bgr=False)
# [FIX-F0-NV12-STREAM-SYNC]
if os.environ.get('IFRNET_F0_NV12_SYNC', '1') != '0':
    torch.cuda.current_stream().synchronize()
```

每段仅一次，开销可忽略。`IFRNET_F0_NV12_SYNC=0` 可关闭做 A/B。

## 验证

修复后重跑同一命令（3 段全通过）：

- segment_002 首帧 IDR：**374,751 B → 78,330 B**（正常量级）
- 全片扫描（SEG0/SEG1/SEG2/FINAL 共 3182 帧）：**✅ 未发现花屏帧**
- 三段帧数 697 / 521 / 373 全部 `解码级验收通过`

## 同类隐患的收尾：Level 2/3 降级路径用 CUDA event 修复

`pipeline.py` Level 2/3（Ring Buffer）降级路径同样在 `_rgb_to_nv12_gpu()` 后
直接 `encode_frame()` 而无同步。该路径**逐帧**调用，补 `synchronize()` 会
阻塞 CPU、破坏 T2/T3 的 GPU 并行重叠，因此改用 **CUDA event 依赖**：

```python
# nvenc_sdk.py：_setup_copy_and_stream() 接入原型
self._libcuda.cuStreamWaitEvent.restype  = c_uint32
self._libcuda.cuStreamWaitEvent.argtypes = [c_void_p, c_void_p, c_uint32]

# nvenc_sdk.py：新增公开方法
def wait_on_event(self, cuda_event: int) -> bool:
    """让 _stream_encode 等待外部 CUevent；False = 无法建立依赖，调用方回退 synchronize()。"""
    if self._stream_encode.value is None or not cuda_event:
        return False
    rc = self._libcuda.cuStreamWaitEvent(self._stream_encode,
                                         c_void_p(int(cuda_event)), 0)
    return rc == 0

# pipeline.py：Level 2/3 逐帧站点
_nv12_ev = torch.cuda.Event()
...
nv12_gpu = _rgb_to_nv12_gpu(rgb_gpu)
_nv12_ev.record()
if not nvenc_encoder.wait_on_event(_nv12_ev.cuda_event):
    torch.cuda.current_stream().synchronize()   # 私有流不可用时的正确性兜底
h264_data = nvenc_encoder.encode_frame(nv12_gpu)
```

要点：
- `torch.cuda.Event` 通过 `.cuda_event` 暴露原始 CUevent 句柄（int），可直接喂给
  ctypes 调用（PyTorch 2.10 验证可用）。
- **同一 event 反复 `record()` 是安全的**：当前流上工作累积有序，后一次 record
  捕获的进度必然包含前一次，等待只会更强，不会变弱。
- 事件只让拷贝流等待，**不阻塞 CPU**，保住 T2/T3 重叠；一次性场景（段首 f0）
  仍用 `synchronize()` 更简单直观。

验证（`scripts` 级单测，直接打 encode_frame 链路，90 帧 720p QP21）：
全部 `wait_on_event=True`、全部帧非空、IDR 14,184 B / P 帧均值 6,261 B /
最大 15,043 B，解码 90 帧 —— 无噪声帧（阈值 200KB）。

## 为什么 realesrgan_video（超分侧）没有这一系列问题

排查时常被拿来对照，三层原因，逐层不同：

### 1. NV12 竞态：ESRGAN **早就修过**，且注释写的就是同一个 bug

`realesrgan_video/nvenc_sdk.py` 的 `write_frame_batch()` / `_flush_mini_batch()`
在 `_rgb_to_nv12_gpu()` 之后、`submit()` 之前有 `[FIX-SYNC-BEFORE-SUBMIT]`：

> 编码线程在 `_stream_encode`（CU_STREAM_NON_BLOCKING）上做 cuMemcpy2DAsync，
> 与默认 stream 之间无隐式同步，**缺这一步会读到未写完的 NV12 数据（花帧）**。
> 该同步此前只补在从未被 import 的 `nvenc_writer.py` 副本（死代码）里，活跃路径一直缺失。

即 ESRGAN 侧踩过同一个坑并已修；IFRNet 的批量路径也补了（`[FIX-ENC-THREAD]`），
**唯独 f0 路径漏了** —— 这就是本文修的东西。

### 2. f0 丢帧：帧映射模型不同，ESRGAN **结构上不存在**

| | 帧映射 | 首帧是否需要特殊处理 |
|---|---|---|
| Real-ESRGAN（超分） | **1:1**，N 帧输入 → N 帧输出，逐帧独立 | 否。第一帧就是第一批的第一个元素，`write_frame()` 只是 append 进 `_mini_batch` |
| IFRNet（插帧） | **N : 2N-1**，每对相邻源帧产出 1 插值帧 + 右帧 | **是**。`encode_order` 只含插值帧 + img1（右帧），f0 是插值锚点、不是产物，**永远不在任何 encode_order 里** |

所以 IFRNet 必须给 f0 开特殊通道，而这条通道的历史就是一部 bug 史：
v6.4.4 用 `encode_frame()` 单独编码（OK）→ `[FIX-PIPE4-LA8]` 误删（丢 f0）
→ `[FIX-F0-ALWAYS-IN-BATCH]` 暂存进 batch → `_loop()` 又提前取走（LA=0 再丢，
见 [[ifrnet-f0-la0-double-consume]]）。ESRGAN 没有这条通道，就没有这块 bug 面。

### 3. code=8 告警：ESRGAN **不是没有，是不报**

`realesrgan_video/nvenc_sdk.py` 的 `_drain_outputs_blocking()`：

```python
if bs_status == NV_ENC_ERR_NEED_MORE_INPUT:
    break
if bs_status != NV_ENC_SUCCESS:
    break          # 静默 break：无计数、无打印
```

IFRNet 同位置经历三轮演进（注释"静默→fail-fast→容忍+遥测"），现在是**容忍 + 打印**；
ESRGAN 原停在第一版。因此 ESRGAN 大概率同样会撞 code=8，只是静默吞掉看不见。
**代价：ESRGAN 侧若真在 LA 预热期丢帧，没有任何遥测能发现，只剩帧数守恒审计兜底。**

> **已补齐（本次）**：`realesrgan_video/nvenc_sdk.py` 的 `_drain_outputs_blocking()`
> 已移植同款 `[P0-FIX-RC-TOLERANT]`（计数 + `_n<=3 or %200==0` 节流 + LA 分支文案）。
> 实测 vbr_hq + LA=8 720p 60 帧：打印 #1/#2/#3（`slot=0, fi=1/2/3`，与
> `Forward lookahead buffering (1/8)(2/8)(3/8)…(8/8)` 同步出现，确为 LA 预热期），
> 帧守恒 60/60 valid、0 empty、0 None、无噪声帧；LA=0 同样 60/60。
>
> **与 IFRNet 的实质差异（保留未改）**：ESRGAN 侧**没有** IFRNet 的
> `[FIX-LA-REDRAIN]` 二次排空安全网。BLKRETRY 的触发条件是
> `_pending_ep_s == NEED_MORE_INPUT`（**EncodePicture** 返回码，非 LockBitstream
> 的 bs_status），与 IFRNet `[FIX-LA-BLKRETRY-ALWAYS]` 语义等价；但若 BLKRETRY
> 也失败，IFRNet 还有 REDRAIN 兜底，ESRGAN 就是真丢帧。故本遥测对超分侧
> **比插帧侧更关键** —— 它是唯一能关联"少帧"与"LockBitstream 返回码"的证据。

### 4. 旁证

`grep -rn "encode_frame(" external/realesrgan_video/*.py` → 只有 **1 处定义
（nvenc_sdk.py:2439）和 0 处调用**。ESRGAN 确实从不走单帧编码路径，与
[[f0-first-frame-loss-ce-pipeline]] 的记述一致。

**Related:** [[ifrnet-f0-la0-double-consume]], [[f0-first-frame-loss-ce-pipeline]], [[realesrgan-video-nvenc-sdk-audit]]

## 附带发现：T4 上偶发的"整段码流长度前缀变随机垃圾"

回归时 segment_001 偶发 `❌ 输出文件验证失败`，伴随数百条：

```
[h264 @ ...] Invalid NAL unit size (-370330720 > 20765).
[h264 @ ...] missing picture in access unit with size 20769
[h264 @ ...] Error splitting the input into NAL units.
```

特征：每帧报 4 条，长度前缀是**随机垃圾值**（正/负、量级 1e9），而剩余字节数
（20765/23224/21361…）恰是本段 P 帧的正常大小区间。→ 帧体积正常，但**码流缓冲
内容被污染**，与本文"输入面未就绪"是同一类跨流竞态，只是发生在**输出侧**
（LockBitstream 取到的 bitstream 缓冲），不是输入侧。

**该失效是偶发的**：同一命令、同一环境下重跑即通过（3/3 段，NAL 错误 0）。
判定是否回归的正确做法是先证伪"改动是否可达"，而不是直接归因：
本次给 Level 2/3 分支加了 `IFRNET_DIAG=1` 计数打印，跑完整流程显示**命中 0 次**
（Level 1 走 `results[0]=='GPU'` 分支，根本到不了 1935 行那个分支）→ 改动惰性。

排查同类问题时优先用这条路径：加计数诊断证明分支可达性 → 再决定是否归因。

**Why:** `[FIX-ASYNC-COPY]` 用 non-blocking 私有流替代 legacy null stream 后，
跨流顺序保证从"隐式全局"退化为"无"，所有向 NVENC 交 GPU tensor 的站点都必须
显式建立依赖边。ESRGAN 侧与批量路径都做了，**只有 IFRNet 的 f0 路径漏做**。

**How to apply:** 凡是 `_rgb_to_nv12_gpu()/_rgb_to_nv12_gpu_batch()` 之后把 GPU
tensor 交给 NVENC 编码器的位置，都必须先同步当前流（或记录 CUDA event 让
`_stream_encode` 等待）。诊断同类花屏时，先比对该段首帧 IDR 与干净段的字节数，
5 倍量级差异即可判定为"编码了噪声"而非编码参数问题。

**Related:** [[f0-first-frame-loss-ce-pipeline]], [[ifrnet-f0-la0-double-consume]], [[v644-encodethread-cross-stream-race]], [[esrgan-pinned-buffer-pool-race]]
