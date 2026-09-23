# 立项 Prompt：IFRNet 侧尺寸钳制「强制消费」仅覆盖 1/5 站点，故障注入下不收敛

> 用法：本文件可直接整段复制给新的 AI 会话作为任务书。所有路径、命令、行号均已实测核对。
> 立项时间：2026-09-08　立项人：Video_Enhancement 复测会话（G3 遗留问题）
> 前序文档：`Plan/H264_LA排空放弃缺陷_立项Prompt.md`（任务 A / A′ 的立项，本次是其遗留问题）

---

## 0. 状态总览（动手前必读）

| 子任务 | 状态 | 说明 |
|---|---|---|
| **A′-1** 属性名 typo（`_slot_pending` → `_strm_slot_pending`） | ✅ **已修并验证** | 2026-09-08 修复，见 §4.1。**勿重复实施** |
| **A′-2** 其余 **4 个钳制站点**仍是「放弃不推进」语义 | ❌ **待办（本次核心）** | 与 ESRGAN 侧已证明会死锁的语义同型，见 §5.2 |
| **A′-3** 故障注入下**不收敛**（420s 跑不完） | ❌ **待办** | 先定位耗时（§6），再决定是修是熔断 |

> 本次**不是**从零开始：typo 已修，崩溃已消失。本次要解决的是
> 「为什么仍然跑不完」以及「语义不一致」。

---

## 1. 环境与代码基线

- 项目：`/workspace/Video_Enhancement`（IFRNet 插帧 + Real-ESRGAN 超分，NVENC SDK 直通编码）
- GPU：Tesla T4 / 15360 MiB / **driver 580.65.06** / CUDA 12.8 / torch 2.10.0+cu128 / NVENC API v13.0
- 主要改动文件：`external/ifrnet_video/nvenc_sdk.py`

**基线校验（改动前先确认，防止环境/仓库被重置）：**

```bash
cd /workspace/Video_Enhancement
md5sum external/ifrnet_video/nvenc_sdk.py
# 期望 6a9cd7064e51b88abc8f3720f775f4e1

# 13 处 HEVC 修复必须仍在（期望 = 13；为 0 说明修复丢失）
grep -c "FIX-CTX-ALREADY-CURRENT\|FIX-HEVC-COUNTED-MARGIN\|_HEVC_DRAIN_MARGIN\|FIX-HEVC-EOS-NONBLOCKING\|FIX-HEVC-FLUSH-NONBLOCKING\|FIX-HEVC-DRAIN-POLL" external/ifrnet_video/nvenc_sdk.py
# 若丢失：cp temp/retest/handoff/nvenc_sdk_ifrnet_CURRENT.py external/ifrnet_video/nvenc_sdk.py
# 但该备份不含 A′-1 修复，恢复后需重做 §4.1 的一行改动

# GPU 体检（⚠️ 容器可能被重建而未挂载 GPU，见
#   /root/.codebuddy/projects/workspace-Video_Enhancement/memory/project_gpu_container_flaky.md）
ls /dev/nvidia* && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch;print(torch.cuda.is_available())"
```

---

## 2. 三条驱动铁律（实测确立，不得违反）

1. **`cuCtxSetCurrent(p)` 之后再 `cuCtxPushCurrent(p)` 必返 201**；只有当前线程无 current context 时 push 才返 0。
2. **HEVC/AV1 对「未就绪 buffer」的 `LockBitstream`，`doNotWait=0/1` 都可能永久阻塞**；
   H.264 则是**空槽返回 SUCCESS + size=0**（不阻塞）。⇒ 绝不能靠 Lock 返回值试探就绪状态。
3. **`doNotWait=1` 存在「SUCCESS + 垃圾 size」竞态**（实测 1.66 / 3.57 / 7.42 MB，远超合法上限），
   按该长度 `from_address()` 越界读会 SIGSEGV，且此时**重锁同一槽会永久阻塞**。

**第 3 条的推论（本次核心）**：`LockBitstream` 既然已返回 **SUCCESS**，说明该槽输出**已被取出**，
只是 size 不可信。因此正确处理是 **unlock → 强制消费该槽（清 pending + 推进指针）→ 丢弃该帧**。
「放弃不推进」会让该槽 pending 记账永久滞留，槽位占满后 `_ensure_slot_free` 无限自旋
（ESRGAN 侧实测：仅需 1 次钳制即挂死，GPU 0% + 显存 10.2 GiB 不退）。

---

## 3. 复现

```bash
cd /workspace/Video_Enhancement

# 【主用例】900 帧 × 3 段，cap=1024（每帧都会被判非法）——420s 内跑不完
timeout 420 env IFRNET_NVENC_MAX_BS_BYTES=1024 python tests/repro_ifrnet_lookahead.py \
  --frames 900 --codec hevc --la 8 --rc vbr_hq --qp 21 --chunk 128 --segments 3 \
  --out temp/retest/g3_faultinj2.mp4 > temp/retest/g3_faultinj2.txt 2>&1
echo "exit=$?"   # 期望 124（超时）；若 139 = SIGSEGV，说明 typo 修复被回退

# 【缩小用例】100 帧 × 1 段，同样 420s 跑不完
timeout 420 env IFRNET_NVENC_MAX_BS_BYTES=1024 python tests/repro_ifrnet_lookahead.py \
  --frames 100 --codec hevc --la 8 --rc vbr_hq --qp 21 --chunk 32 --segments 1 \
  --out temp/retest/g3_faultinj_small.mp4 > temp/retest/g3_faultinj_small.txt 2>&1

# 【对照】正常上界（不注入）——必须 ALL PASS
python tests/repro_ifrnet_lookahead.py --frames 900 --codec hevc --la 8 \
  --rc vbr_hq --qp 21 --chunk 128 --segments 3 --out temp/retest/ok_hevc_la8.mp4
```

已有基线日志（可直接看，不必重跑）：

| 文件 | 内容 |
|---|---|
| `temp/retest/g3_faultinj.txt` | **修复前**：`AttributeError` + `dumped core`（exit=139） |
| `temp/retest/g3_faultinj2.txt` | **修复后** 900×3：445 次强制消费，exit=124 |
| `temp/retest/g3_faultinj_small.txt` | **修复后** 100×1：45 次强制消费，exit=124 |

---

## 4. 已确认的事实（无需重做）

### 4.1 A′-1 属性名 typo（✅ 已修）

`external/ifrnet_video/nvenc_sdk.py` 的钳制分支原照搬 ESRGAN 实现，写成了
`self._slot_pending.pop(...)`，而 **IFRNet 侧的 pending 表叫 `_strm_slot_pending`（全文件 25 处）**。
2026-09-08 已改为 `self._strm_slot_pending.pop(slot_idx, None)`（现约 1592 行，带警示注释）。

修复前后实测：

| | exit | 现象 |
|---|---|---|
| 修复前 | **139 (SIGSEGV)** | `AttributeError: 'NVENCEncoder' object has no attribute '_slot_pending'` → teardown 崩溃 |
| 修复后 | 124（超时） | 强制消费正常打印 445 / 45 次，**零** `AttributeError` / `Traceback` / `dumped core` |

### 4.2 修复后仍然不收敛，但**不是死锁**

`temp/retest/g3_faultinj2.txt` 实测：

- 提交进度持续推进：`submitted 128 → 256 → 384 → 640 → 768 → 896 /900`
- 诊断信号计数**全为 0**：`排空超限` / `疑似停滞` / `提前EOF` / `Traceback` / `SIGSEGV`
- 速率：900×3 用例约 **1 次强制消费/秒**（445 次 / 420s）；
  100×1 用例约 **0.1 次/秒**（45 次 / 420s）→ **非线性，越到尾部越慢**

### 4.3 ESRGAN 侧对照（已达标，可作参照系）

同款故障注入（`ESRGAN_NVENC_MAX_BS_BYTES=1024`，900×3）：
**859 次强制消费 / 86 秒 / EXIT=1 干净退出 / GPU 释放**，明确报
`decoded=33 expected=900 reason=decoded_frame_mismatch`。

⇒ IFRNet 侧慢了约一个数量级，且不能自行收敛退出。

---

## 5. 代码锚点（行号为 2026-09-08 工作区状态）

### 5.1 五个钳制站点现状

| # | 行号 | 所在函数 | 钳制失败后的语义 | 是否强制消费 |
|---|---|---|---|---|
| 1 | 1575 | `_drain_outputs_blocking` | unlock → pop `_strm_slot_pending` → `_output_slot_idx += 1` → 丢帧 | ✅ **是**（A′-1 已修） |
| 2 | 1960 | `_lock_bitstream_blocking` | `unlock_fn(...)` → `return b"", 8` | ❌ **否** |
| 3 | 2010 | `_lock_bitstream_with_retry` | `_unlock_fn(...)` → `return b"", 8` | ❌ **否** |
| 4 | 2486 | `encode_frames_stream`（EOS 排空） | unlock → 重试到 `_eos_deadline` → 打印「EOS 垃圾 size 已丢弃」 | ❌ 否（有 deadline 重试） |
| 5 | 3303 | `flush` | unlock → 重试到 `_first_deadline` → **raise RuntimeError** | ❌ 否（有 deadline 重试） |

> 站点 2/3 返回 `(b"", 8)`：本文件只定义了 `NV_ENC_SUCCESS = 0`（156 行）与
> `NV_ENC_ERR_NEED_MORE_INPUT = 17`（157 行），**`8` 是硬编码字面量、未定义常量**。
> 按 nvEncodeAPI.h 枚举序（1=NO_ENCODE_DEVICE, 2=UNSUPPORTED_DEVICE, 3=INVALID_PTR,
> 4=INVALID_EVENT, 5=INVALID_PARAM, 6=INVALID_CALL, 7=OUT_OF_MEMORY,
> **8=ENCODER_NOT_INITIALIZED**）推测为 `NV_ENC_ERR_ENCODER_NOT_INITIALIZED`，
> 但**承接方应在 NVENC 头文件核对后再引用此结论**。
> 调用方（`_ensure_slot_free` 约 1853 行的探测分支）拿到空数据后不会推进 `_output_slot_idx`。

### 5.2 关键不一致（本次首要嫌疑）

A′-1 只修了站点 1，**站点 2/3/4/5 仍是「放弃不推进」**。而按铁律 3 的推论，
这个语义正是 ESRGAN 侧实测会挂死的那一种。本次没有挂死，很可能只是因为
站点 1 的强制消费持续把槽位让出来，掩盖了其余站点的记账滞留——代价就是**极慢**。

### 5.3 相关开关与常量

| 位置 | 内容 |
|---|---|
| 548–554 | `_max_valid_bitstream_bytes`；`IFRNET_NVENC_MAX_BS_BYTES` 覆盖（**仅供故障注入**，生产不得设） |
| 1526–1527 | `_drain_poll_mode = (codec in hevc/av1) and NVENC_HEVC_DRAIN_POLL == "1"`，**默认关** |
| 1544 / 1551 | `_lock_deadline`（仅 poll 模式用）；`doNotWait = 1 if _drain_poll_mode else 0` |
| 1575–1596 | 钳制分支主体（A′-1 修复处） |
| 1689 / 1694 | `_HEVC_DRAIN_MARGIN`（默认 2）、`_hevc_ready_count` |
| ~1853 | `_ensure_slot_free` 的 `_lock_bitstream_blocking` 探测分支（`_allow_probe` 门控） |

⚠️ **HEVC 默认 `doNotWait=0`（阻塞锁）**：一次失败的 `LockBitstream` 可能就在驱动内耗掉 ~1s，
这与实测「~1 次强制消费/秒」高度吻合——这是待验证的首要假设之一（见 §6）。

---

## 6. 诊断步骤（**先测量，再动手改**）

目前**不知道** 407ms/批 之外的时间花在哪。按顺序做，每步都应留下数据：

1. **加临时计时探针**（`_drain_outputs_blocking` 循环内）：
   `lock_bs_fn` 耗时、`unlock_fn` 耗时、单次循环总耗时、进入循环的 `slot_idx` 与
   `_strm_slot_pending` 长度。每 50 次打印一次均值，形如 `[PROF-FAULT]`。
2. **统计站点命中分布**：在 5 个钳制站点各加一个计数器，看故障注入下
   到底是哪几个站点在触发（判断站点 2/3 是否也在大量触发）。
3. **确认 `_ensure_slot_free` 是否在自旋**：打印 `_guard` 达到 `_slot_count * 4` 的次数。
4. **对比实验**：`NVENC_HEVC_DRAIN_POLL=1` 下重跑同一注入，看速率是否变化
   （验证「阻塞锁耗时」假设）。
5. **对比实验**：`--codec h264`（H.264 空槽返回 SUCCESS+size=0，不阻塞）下重跑，
   看是否同样不收敛。

**假设清单（按优先级，均需实测确认/否证）**

- **H1**：时间主要耗在 HEVC 的阻塞 `LockBitstream`（每次 ~1s）。→ 由步骤 1、4 判定。
- **H2**：站点 2/3 的「放弃不推进」导致槽位记账滞留，`_ensure_slot_free` 反复空转。
  → 由步骤 2、3 判定。
- **H3**：`_output_slot_idx` 与 FIFO 实际消费量不同步，造成相位漂移后反复锁同一批槽。
  → 由步骤 1 的 `slot_idx` 分布判定。

---

## 7. 修复方向（供参考，最终方案由承接方定并**先给方案再动手**）

- **首选**：把 5 个站点的钳制失败语义**统一为强制消费**（清 pending + 推进指针 + 丢帧），
  与站点 1 及 ESRGAN 侧一致。站点 4/5 已有 deadline 重试，可在 deadline 到期后转为强制消费。
- **次选**：在 `_drain_outputs_blocking` 内加**熔断**——连续 N 次（如 32 次）强制消费后，
  直接判定本段编码不可恢复，走 `_strict_eos` 的 fail-fast 路径，避免长时间空转。
- **不推荐**：把 `_strict_eos` 默认改成 0，或调大 `_max_valid_bitstream_bytes` 来"绕过"——
  那只会把致命错误降级成静默丢帧。

**必须保持的硬约束**

1. 帧数守恒语义不变（正常工况下钳制**必须 0 次触发**）。
2. 不得回退 13 处 HEVC 修复中的任何一处（§1 的 grep 校验 = 13）。
3. 不得违反 §2 三条铁律；**严禁在钳制失败分支里重试 Lock**。
4. 正常回归矩阵（H.264/HEVC × LA=0/8）必须保持 4/4 ALL PASS。

---

## 8. 验收标准

| # | 项 | 通过条件 |
|---|---|---|
| 1 | 故障注入收敛性 | `IFRNET_NVENC_MAX_BS_BYTES=1024` + 900×3：在 **≤ 420s** 内自行结束（不再 exit=124），且 exit **≠ 139**（无 SIGSEGV） |
| 2 | 故障注入可诊断 | 明确打印强制消费总次数，并以 `decoded_frame_mismatch`（或等价的段失败）**显式报失败**，而非静默挂起 |
| 3 | 正常回归 G2 矩阵 | H.264/HEVC × LA=0/8 四组仍 `ALL PASS`；每组 `pairs=900 expected=900`、`empty=0`、`重复fi=0`、`fi序列正确=True`、`decode_errors=0` |
| 4 | 正常工况钳制 0 触发 | 跑一次不注入的真实素材（如 `word_world_2.mp4`），日志中 `非法 size` / `强制消费` 计数 = **0** |
| 5 | 无死锁 | 全程无「编码线程未在 120s 内退出」；`nvidia-smi` 显存能回落 |
| 6 | 站点语义一致 | `grep -c "强制消费\|force_dropped"` 与 5 个站点的强制消费路径一一对应（或明确记录哪些站点刻意保留重试语义及理由） |
| 7 | 13 处 HEVC 修复未丢 | §1 的 grep = 13 |

---

## 9. 交付物

1. **诊断结论**：`[PROF-FAULT]` 数据 + H1/H2/H3 的确认或否证（§6）
2. 补丁（先给方案，批准后再改；改动集中在 `external/ifrnet_video/nvenc_sdk.py`）
3. §8 七项验收的完整日志
4. 若新增环境变量，需给出默认值标定依据
5. 更新 `temp/retest/HANDOFF_RESUME.md` §9.6.1（把「部分通过」改为最终结论）

---

## 10. 相关文档与工具

- 前序立项：`Plan/H264_LA排空放弃缺陷_立项Prompt.md`（任务 A / A′ 立项；§10.6 是 A′ 原始描述）
- 现场交接：`temp/retest/HANDOFF_RESUME.md`（**§9.6.1 = 本次问题的实测记录**，§9.7 = typo 根因）
- G3 基线日志：`temp/retest/g3_faultinj.txt` / `g3_faultinj2.txt` / `g3_faultinj_small.txt`
- 编码层回归：`tests/repro_ifrnet_lookahead.py`
- 记忆文件（**在会话目录，不在项目内**）：
  `/root/.codebuddy/projects/workspace-Video_Enhancement/memory/project_gpu_container_flaky.md`
  （容器 GPU 时有时无的判据）、`feedback_no_gpu_work_mode.md`（无 GPU 时先做纯修复 + 列 GPU 待验清单）
- 完整备份（含 9/7 全部产物）：`temp/dc8d1beef8344674853af3ba59829e8f/temp/temp.tar.gz`（12 GB）

---

## 11. 已踩过的坑（务必避开）

1. **照搬 ESRGAN 代码时属性名不同**：IFRNet 侧是 `_strm_slot_pending`，ESRGAN 侧是 `_slot_pending`。
   这是 A′-1 的直接根因。
2. **`ldconfig` 会按 SONAME 把 `libcuda.so.1` / `libnvidia-encode.so.1` 的 symlink 改回最高版本**；
   手工 `ln -sfn` 之后**不要再跑 `ldconfig`**。
3. **后台跑测必须加 `< /dev/null`**，否则进程阻塞在交互式输入上，表现为「日志停启动行、GPU 0%、CPU 0%」的假死。
4. **`utilization.gpu` 只统计 SM/3D，不含 NVENC 也不含 NVDEC**；纯编解码阶段显示 0% 属正常，
   排查编码瓶颈要加 `utilization.encoder`，排查解码要另看进程推进。
5. **进度条用 `\r` 分隔**，grep 前必须 `tr '\r' '\n'`。
6. **修改断点前先想清楚**：断点失效会让已完成的插帧产物被重新编码覆盖
   （2026-09-08 曾因此毁掉 786 MB 的段 0）。指纹要取**运行日志打印的「当前值」**，不要手算。
