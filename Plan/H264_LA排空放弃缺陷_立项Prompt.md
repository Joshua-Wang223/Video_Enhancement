# 立项 Prompt：NVENC LockBitstream 尺寸钳制缺失 + H.264 + LA>0 排空缺陷

> 用法：本文件可直接整段复制给新的 AI 会话作为任务书。所有路径、命令、行号均已实测核对。
> 立项时间：2026-09-04　立项人：Video_Enhancement 复测会话
> **最后更新：2026-09-09（全三项子任务收口，本文转为「已完成记录 + 回归依据」）**

## ✅ 状态总览（2026-09-09 更新，动手前必读）

| 子任务 | 状态 | 说明 |
|---|---|---|
| **任务 B**（ESRGAN 侧尺寸钳制） | ✅ **已完成并验证** | 2026-09-05 落地，B1–B6 全部通过。见 §10，**勿重复实施** |
| **任务 A**（H.264 + LA>0 排空缺陷） | ✅ **已完成并验证** | 2026-09-08 落地（本表 09-05 版误标为「未开始」），2026-09-09 复测确认。见 §13，**勿重复实施** |
| **任务 A′**（IFRNet 侧钳制失败语义的死锁隐患） | ✅ **已完成并验证** | A′-1（属性名 typo）2026-09-08；A′-2（5 站点统一强制消费）+ A′-3（故障注入收敛）2026-09-09。见 §14，**勿重复实施** |

> ⚠️ **本立项已全部收口，无待办子任务。** 承接方请勿重复实施任何一项；
> 若需回归，直接执行 §8 更新后的验收表即可。
>
> ⚠️ 状态表曾落后于代码（09-05 版把任务 A 标为「未开始」时代码已修好）。
> **动手前务必先跑 §3 复现命令与 §8 验收矩阵，用实测结果反推真实待办。**
>
> A′-2/A′-3 的详细过程另见 `Plan/IFRNet_尺寸钳制强制消费遗留_立项Prompt.md`。

---

## 0. 任务

本立项包含两个子任务（分别对应铁律 2 与铁律 3）：

- **任务 A（主／待办）**：修复 `external/ifrnet_video/nvenc_sdk.py` 中
  **H.264 + lookahead>0** 路径下的
  `RuntimeError: [NVENCEncoder] strict drain abandoned target slot: slot=0, pending_fis=[11]`，
  并使 4 组编解码矩阵（H.264/HEVC × LA=0/8）全部通过。
- **任务 B（并案／✅ 已完成）**：给 `external/realesrgan_video/nvenc_sdk.py` 补上
  **`_is_legal_bitstream_size` 尺寸钳制**。原状态：该文件完全没有该保护，
  5 处 `from_address()` 越界读隐患全部裸奔——**现已全部修复并验证**（见 §10）。

另有 **任务 A′**（IFRNet 侧钳制失败语义的死锁隐患），在任务 B 实施过程中实测发现，
见 §10.6。它与任务 A 同在 `external/ifrnet_video/nvenc_sdk.py`，建议一并处理。

⚠️ 已确认 `RESUME.md` 第 6 节对 ESRGAN 侧的描述有一处**已过时**（见 §11.3），不要照抄旧结论。

---

## 1. 环境与代码基线

- 项目：`/workspace/Video_Enhancement`（IFRNet 插帧 + Real-ESRGAN 超分，NVENC SDK 直通编码）
- GPU：Tesla T4 / 15360 MiB，driver 580.65.06，CUDA 12.4，torch 2.10.0+cu128，NVENC API v13.0
- **当前工作区有 13 处未提交 git 的 NVENC 修复**（HEVC+LA 链路，2026-09-03 会话产物）。
  ⚠️ 若新环境从仓库重新拉取代码，修复会全部丢失。恢复方式：
  ```bash
  cd /workspace/Video_Enhancement
  grep -c "FIX-CTX-ALREADY-CURRENT\|FIX-HEVC-COUNTED-MARGIN\|_HEVC_DRAIN_MARGIN\|FIX-HEVC-EOS-NONBLOCKING\|FIX-HEVC-FLUSH-NONBLOCKING\|FIX-HEVC-DRAIN-POLL" external/ifrnet_video/nvenc_sdk.py
  # 期望 = 13；若为 0：
  cp temp/retest/handoff/nvenc_sdk_ifrnet_CURRENT.py external/ifrnet_video/nvenc_sdk.py
  ```

## 2. 三条驱动铁律（实测确立，不得违反）

1. **`cuCtxSetCurrent(p)` 之后再 `cuCtxPushCurrent(p)` 必然返回 201**
   （同一 context 已是 current 时不可再入栈；只有当前线程无 current context 时 push 才返回 0）。
   证据：`tests/probe_cuda_context.py` 的 `depth` / `seq` / `sim` 模式。
2. **HEVC 驱动对「未就绪 buffer」的 `LockBitstream`，`doNotWait=0/1` 都可能永久阻塞**；
   H.264 则是**空槽返回 SUCCESS + size=0**（不阻塞）。
   ⇒ 绝不能靠 Lock 返回值试探就绪状态。
3. **`doNotWait=1` 存在「SUCCESS + 垃圾 size」竞态**（实测 1.66/3.57/7.42 MB，远超合法上限），
   按该长度 `from_address()` 越界读会 SIGSEGV，且此时重锁同一槽会永久阻塞。

**第 2 条是本缺陷的关键**：HEVC 与 H.264 的未就绪槽语义**根本不同**，
而现有排空守卫是按 HEVC 语义写的，却被**无条件**应用到 H.264。

---

## 3. 复现（约 30 秒）

```bash
cd /workspace/Video_Enhancement

# 失败用例（本缺陷）
python tests/repro_ifrnet_lookahead.py --frames 900 --codec h264 --la 8 \
    --rc vbr_hq --qp 21 --chunk 128 --segments 3 --out temp/retest/issue_h264_la8.mp4

# 对照：以下 3 组均通过
python tests/repro_ifrnet_lookahead.py --frames 900 --codec h264 --la 0  --rc vbr_hq --qp 21 --chunk 128 --segments 3 --out temp/retest/ok_h264_la0.mp4
python tests/repro_ifrnet_lookahead.py --frames 900 --codec hevc --la 8 --rc vbr_hq --qp 21 --chunk 128 --segments 3 --out temp/retest/ok_hevc_la8.mp4
python tests/repro_ifrnet_lookahead.py --frames 900 --codec hevc --la 0 --rc vbr_hq --qp 21 --chunk 128 --segments 3 --out temp/retest/ok_hevc_la0.mp4
```

失败输出：
```
RuntimeError: [NVENCEncoder] strict drain abandoned target slot: slot=0, pending_fis=[11]
```

## 4. 已确认的事实（立项前已完成，无需重做）

- ✅ **非本次改动引入**：用 `temp/retest/handoff/nvenc_sdk_ifrnet_CURRENT.py`（打补丁前版本）
  替换后重跑，失败完全一致 → **既有缺陷**。
- ✅ 与「201 警告补丁」无关，与「切镜鬼影修复」无关（两者均已单独 A/B 验证通过）。
- ✅ `NVENC_STRICT_EOS` 默认 `1`，是它把「排空放弃」从警告升级成 `RuntimeError` 终止整段。
- ✅ 历史同类记录（`temp/retest/handoff/RESUME.md` 第 105–107 行）：
  - margin=1 + slot=10 → 挂死（SIGKILL）
  - **margin=2 + slot=10 → 抛 `strict drain abandoned target slot: slot=0, pending_fis=[0]`（循环依赖）**
  - margin=2 + slot=11 → HEVC ALL PASS
  ⇒ **当前的报错形态与「槽数不足导致循环依赖」完全同型**，只是本例 slot 已是 11、
  `pending_fis=[11]`（而非 `[0]`），说明不是简单的槽数不够。

## 5. 关键代码锚点（行号为当前工作区状态）

| 位置 | 内容 |
|---|---|
| `nvenc_sdk.py:582` | `_required_buffers = max(1, la_depth + 3)` → LA=8 时槽数 = 11 |
| `nvenc_sdk.py:1653` | `_HEVC_DRAIN_MARGIN = int(os.environ.get("NVENC_HEVC_DRAIN_MARGIN", "2"))` |
| `nvenc_sdk.py:1655` | `def _hevc_ready_count(self, limit)` —— 就绪上界：<br>`已提交 − la_depth − margin − 已取回` |
| `nvenc_sdk.py:1463` | `def _drain_outputs_blocking(...)` |
| `nvenc_sdk.py:1762` | `def _ensure_slot_free(self, slot_idx, pairs)` |
| `nvenc_sdk.py:1783` | `_ensure_slot_free` 内：`_ready = self._hevc_ready_count(self._slot_count)`；<br>`_ready <= 0` 时**禁止任何 Lock**，直接进入空帧占位兜底 |
| `nvenc_sdk.py:1799` | `_guard > self._slot_count * 4` → 对目标槽自身做 blocking Lock 探测（`_allow_probe` 受 `_ready<=0` 限制） |
| `nvenc_sdk.py:1828-1832` | 兜底失败 + `_strict_eos` → **raise `strict drain abandoned target slot`**（本缺陷抛错点） |
| `nvenc_sdk.py:2113` | `def encode_frames_stream(...)` |
| `nvenc_sdk.py:2330` / `2527` | `encode_frames_stream` 内两处 `_hevc_ready_count(_max_drain)` / `_hevc_ready_count(_final_max)` |
| `nvenc_sdk.py:550` | `self._strict_eos = os.environ.get("NVENC_STRICT_EOS","1") not in (...)` |
| `nvenc_sdk.py:1849` | `def _lock_bitstream_blocking(bs_handle, timeout_ms=500)` |

**注意**：`_hevc_ready_count` 名字带 HEVC，但三处调用（1783 / 2330 / 2527）**均无 codec 判定**，
即 H.264 也被同一套「HEVC 保守口径」约束。

## 6. 首要假设（✅ 2026-09-09 已验证成立，结论见 §13.1）

> **H.264 的未就绪槽返回 SUCCESS+size=0（不阻塞），根本不需要 HEVC 那套 counted-drain 守卫。**
> 现在把 `_hevc_ready_count` 无条件套用到 H.264，导致：
> `_ready<=0` → 禁止 Lock（`_allow_probe=False`）→ 永远摸不到目标槽 →
> `_guard` 超限 → `_strict_eos=1` 下直接 raise 终止整段。
> 换言之：**这是「HEVC 防挂死守卫」在 H.264 上的过度约束（假阳性），不是真的排不出来。**

建议的验证顺序（每步都很快）：

1. **`NVENC_STRICT_EOS=0` 跑 H.264+LA=8**：若能通过（仅打印「⚠️ slot=0 排空超限…空帧占位兜底」），
   则确认为「放弃排空」而非「真的死锁」，假设成立。
2. **`NVENC_HEVC_DRAIN_MARGIN=0` / `=1` 跑 H.264+LA=8**：若 margin 降低后通过，
   进一步坐实「就绪上界过于保守」。
3. 在 `_ensure_slot_free` 的 1783 附近加临时日志，打印
   `self._codec / _la_depth / _frame_idx / _output_slot_idx / _ready / slot_idx / len(_dq)`，
   确认 `_ready` 是否恒为 0。

## 7. 修复方向（供参考，最终方案由承接方定并先给方案再动手）

- **首选**：把 counted-drain 守卫按 codec 分流——
  HEVC/AV1 沿用 `_hevc_ready_count`；H.264 恢复「靠 Lock 返回值判定」（空槽 SUCCESS+size=0，
  正是铁律 2 所述 H.264 的安全语义），不套用 HEVC 余量。
- **次选**：给 H.264 单独标定一个 margin（如新增 `NVENC_H264_DRAIN_MARGIN`，默认 0），
  并把 `_required_buffers` 的 `la_depth+3` 对 H.264 重新标定。
- **不推荐**：直接把 `_strict_eos` 默认改成 0 —— 那只是把致命错误降级成静默丢帧/占位帧，
  会掩盖真实的帧丢失。

**必须保持的硬约束**：
1. **帧数守恒 `2N−1` 不变**（整条验收链、断点续跑、timescale 归一化都依赖它）。
2. 不得回退 13 处 HEVC 修复中的任何一处；4 组矩阵必须全绿。
3. 不得违反第 2 节三条铁律。

## 8. 验收标准（✅ 2026-09-09 全项实测通过）

复现命令（4 组矩阵，各 900 帧 × 3 段）：

```bash
cd /workspace/Video_Enhancement
for spec in "h264:8" "h264:0" "hevc:8" "hevc:0"; do
  c=${spec%%:*}; l=${spec##*:}
  python tests/repro_ifrnet_lookahead.py --frames 900 --codec $c --la $l \
      --rc vbr_hq --qp 21 --chunk 128 --segments 3 \
      --out temp/retest/A1/fin_${c}_la${l}.mp4 < /dev/null
done
```

| # | 项 | 通过条件 | 2026-09-09 实测 |
|---|---|---|---|
| 1 | H.264 + LA=8 | `=== 总结: ALL PASS ===`，exit=0 | ✅ ALL PASS，exit=0 |
| 2 | H.264 + LA=0 | ALL PASS | ✅ ALL PASS，exit=0 |
| 3 | HEVC + LA=8 | ALL PASS（回归，不得破坏）| ✅ ALL PASS，exit=0 |
| 4 | HEVC + LA=0 | ALL PASS（回归）| ✅ ALL PASS，exit=0 |
| 5 | 帧数守恒 | 每组 3 段各 `pairs=900 expected=900`，`empty=0`、`重复fi=0`、`相邻逐字节重复=0`、`fi序列正确=True`、`decode_errors=0` | ✅ 12/12 段全项达标 |
| 6 | 无挂死 | 全程无「编码线程未在 120s 内退出」，无 `NVENC_HEVC_DRAIN_POLL` 相关死锁 | ✅ 单段 5–7s；结束后无残留进程、显存回落 0 MiB |
| 7 | 真实素材抽验 | `--codec h264 --lookahead-depth-ifrnet 8`，解码级验收通过、exit=0 | ✅ `tests/repro_real_frames.py -i benchmark_output/word_world_2_v6.4.5_x2.mp4 --codec h264 --la 8 --frames 300` → `pairs=300 expected=300`，钳制 0 次触发（HEVC 同档亦通过）|
| 8 | ~~任务 B：ESRGAN 尺寸钳制~~ | 回归时确认 `grep -c _is_legal_bitstream_size external/realesrgan_video/nvenc_sdk.py` 仍 ≥ 6 | ✅ = 6（2026-09-09 复核）|
| 9 | **任务 A′：IFRNet 侧强制消费** | 故障注入下**不挂死、无 SIGSEGV**，正常工况下钳制 0 次触发 | ✅ 见下方 A′ 实测 |
| 10 | 13 处 HEVC 修复未丢 | §1 的 `grep -c` = 13 | ✅ = 13（2026-09-09 复核）|

**第 9 项 A′ 故障注入实测**（`IFRNET_NVENC_MAX_BS_BYTES=1024`）：

| 用例 | 修复前 | 修复后 |
|---|---|---|
| 100×1 / HEVC / LA=8 | exit=**139**（SIGSEGV，teardown DestroyEncoder）或 124（永久挂死） | **8s 内 exit=1**，显式 `sizecap force-drop budget exhausted: dropped=32 >= 32`，`Encoder closed` 正常打印 |
| 900×3 / HEVC / LA=8 | exit=124（420s 跑不完，45 次强制消费） | **10s 内 exit=1**，同上 |
| 900×3 / H.264 / LA=8 | — | **exit=1**，同上 |

补充稳定性：H.264 + LA=8 另跑 chunk=64/5 段、chunk=256/3 段、chunk=128/6 段共 **14 段全过**，
无 `排空超限` / `强制消费` / `非法 size` / `Traceback`。

## 9. 交付物（✅ 均已交付）

1. 根因结论 —— **已确认「counted-drain 守卫在 H.264 上的过度约束（假阳性）」**，见 §13.1
2. 补丁 —— 集中在 `external/ifrnet_video/nvenc_sdk.py`，见 §13.2 / §14.2
3. 第 8 节 10 项验收的完整日志 —— 见 §8 表内「2026-09-09 实测」列；
   原始日志在 `temp/retest/A1/`（`fin_*.txt` / `fi900.txt` / `fi_h264.txt` / `prof_small*.txt`）
4. 环境变量标定依据 —— 未新增 `NVENC_H264_DRAIN_MARGIN`（首选方案按 codec 分流，无需新 margin）；
   新增 `IFRNET_NVENC_SIZECAP_ABORT`（默认 32）与 `IFRNET_NVENC_PROF_FAULT`（默认关），标定依据见 §14.3
5. ~~任务 B：ESRGAN 侧 5 处钳制站点的改造说明与故障注入验证记录~~ → **已交付（2026-09-05），见 §10 开头落地摘要**
6. **任务 A′**：IFRNet 侧钳制失败语义改为强制消费 + 故障注入验证记录 → **已交付（2026-09-09），见 §14**

---

## 10. 【任务 B · ✅ 已完成 2026-09-05】ESRGAN 侧 `_is_legal_bitstream_size` 尺寸钳制

> ### ✅ 本节为**已完成记录**，不是待办。承接方请勿重复实施，仅作参考与回归依据。

**落地摘要**（文件 `external/realesrgan_video/nvenc_sdk.py`，md5 `e757295a232c25ca4a8b36ba4681d5cb`）：

| 项目 | 落地位置（当前行号） |
|---|---|
| 上界 `_max_valid_bitstream_bytes` | 第 609–613 行（`__init__` 内，按**输出**分辨率 `W*H*4`，支持 `ESRGAN_NVENC_MAX_BS_BYTES` 覆盖用于故障注入） |
| `_sizecap_force_dropped` 计数器 | 第 619 行 |
| `_is_legal_bitstream_size()` 定义 | 第 1421 行 |
| 5 个钳制站点 | 第 1397（退避重试 LockBitstream）、1542（`_drain_outputs_blocking`）、1791（`_ensure_slot_free` 探测）、2086（EOS 排空）、2887（`flush()`） |
| **强制消费**防死锁 | 第 1553–1570 行（见 §10.6） |

**验证结果（全部通过）**：

| # | 项 | 结果 |
|---|---|---|
| B1 | 站点覆盖 | `grep -c _is_legal_bitstream_size` = **6**（1 定义 + 5 使用）✅ |
| B2 | 故障注入（cap=1KB） | 修复前**永久卡死**；修复后 **EXIT=1、1分26秒正常退出、GPU 释放**，859 次强制消费，明确报 `decoded=33 expected=900 reason=decoded_frame_mismatch` ✅ |
| B3 | 正常回归 | 真实素材 3 段，3/3 通过，EXIT=0 ✅ |
| B4 | 双开关矩阵 | `ESRGAN_NVENC_CROSS_SEGMENT_REUSE` 取 0 与 1 **两档均通过** ✅ |
| B5 | 帧数守恒 | 各段 `decoded == expected`（900/409/493）✅ |
| B6 | 无回归 | IFRNet 侧修复未被破坏 ✅ |

备份：`temp/retest/patched_backup/realesrgan_nvenc_sdk.py`

---

### 10.1 风险等级与危害（立项时的原始分析，保留存档）

`external/realesrgan_video/nvenc_sdk.py` 中 **`_is_legal_bitstream_size` 出现次数 = 0**，
也没有任何等价的 `_max_valid_bitstream_bytes` 上界。全部 5 处 `from_address()` 都只判
`bitstream_size == 0`，**没有上界**。按铁律 3，驱动在 `doNotWait=1` 下会返回
`SUCCESS + 垃圾 size`（实测 1.66 / 3.57 / 7.42 MB，远超合法上限），
按该长度 `(c_uint8 * size).from_address(ptr)` 即**越界读 → SIGSEGV**。

这不是理论风险：IFRNet 侧在补上该钳制前，`tests/../evidence/fault.txt` 的 faulthandler
已把 SIGSEGV 精确定位到 `_drain_outputs_blocking` 的 `from_address`。
**ESRGAN 侧目前处于 IFRNet 侧修复之前的同等裸奔状态。**

### 10.2 IFRNet 侧参考实现（直接照搬即可）

```python
# external/ifrnet_video/nvenc_sdk.py:544-548  （__init__ 内）
# [P3-FIX-LockBitstream-SizeCap] 驱动异常时曾见 SUCCESS+数十 MB 垃圾 size。
# 用“约4倍原始 RGBA 上限”拒绝明显非法输出；正常压缩码流远低于该值。
self._max_valid_bitstream_bytes = max(
    64 * 1024, int(width) * int(height) * 4)

# external/ifrnet_video/nvenc_sdk.py:1642-1644
def _is_legal_bitstream_size(self, size: int) -> bool:
    """[P3-FIX-LockBitstream-SizeCap] 拒绝零或超出原始像素上界的垃圾返回。"""
    return 0 < int(size) <= int(self._max_valid_bitstream_bytes)
```

IFRNet 侧在 **5 个** Lock 站点全部启用（行号 1557 / 1903 / 1953 / 2429 / 3246）。

### 10.3 ESRGAN 侧需补钳制的 5 个站点（行号为当前工作区状态）

| # | 行号 | size 读取 | 数组构造 / from_address | 所在函数 |
|---|---|---|---|---|
| 1 | 1360 / 1365 | `cast(byref(lock_raw, 36), POINTER(c_uint32))[0]` | `buf_type = c_uint8 * bitstream_size` | `[Tier 3-E] 带指数退避重试的 LockBitstream`（1335） |
| 2 | 1481 / 1490 | 同上 | `buf_type = c_uint8 * bitstream_size` | `_drain_outputs_blocking`（1374） |
| 3 | 1699 / 1707 | 同上 | `buf_type = c_uint8 * bitstream_size` | `_ensure_slot_free`（1602）内探测 |
| 4 | 1984 / 1991 | `cast(byref(_lr, 36), POINTER(c_uint32))[0]` | `_eos_data = bytes((c_uint8 * _bs_size).from_address(_bs_ptr))` | EOS 排空路径 |
| 5 | 2773 / 2782 | `cast(byref(lock_raw, 36), POINTER(c_uint32))[0]` | `buf_type = c_uint8 * bitstream_size` | `flush()`（2692） |

改造方式：在 `if bitstream_size == 0: ...` 判定处追加
`or not self._is_legal_bitstream_size(bitstream_size)`，
并**复用 IFRNet 侧的失败语义**——
`unlock` 后**放弃本轮、不推进** `_output_slot_idx`（**严禁重试**：重锁同一槽在 HEVC 下会永久阻塞）。

### 10.4 实施步骤（✅ 已执行完毕，保留作参考）

1. 在 `NVENCEncoder.__init__` 增加 `self._max_valid_bitstream_bytes = max(64*1024, W*H*4)`。
   ⚠️ ESRGAN 的 `W/H` 是**超分后**尺寸（如 1536×1152），务必用输出尺寸而非输入尺寸。
2. 新增 `_is_legal_bitstream_size()`（与 IFRNet 完全一致）。
3. 在 §10.3 的 5 个站点逐一加判，并打印含 `cap=` 的诊断日志（对齐 IFRNet 1564 / 2440 行）。
4. **故障注入验证**：临时把上界改成极小值（如 1 KB）跑一次，确认——
   钳制生效（打印告警）、**不崩溃**、帧数守恒或按设计失败，**绝不出现 SIGSEGV**。
5. 恢复正常上界后跑真实素材回归。

### 10.5 验收标准（✅ 全部通过，详见本节开头的落地摘要）

| # | 项 | 通过条件 |
|---|---|---|
| B1 | 站点覆盖 | `grep -c "_is_legal_bitstream_size" external/realesrgan_video/nvenc_sdk.py` ≥ **6**（1 定义 + 5 使用） |
| B2 | 故障注入 | 上界改为 1 KB 时：打印钳制告警、**无 SIGSEGV**、进程正常退出 |
| B3 | 正常回归 | `--mode upscale_then_interpolate --skip-interpolate`，真实素材 3 段，3/3 段解码级验收通过、exit=0 |
| B4 | 双开关矩阵 | `ESRGAN_NVENC_CROSS_SEGMENT_REUSE` 取 `0` 与 `1` **两档都要过**（本项与已完成的方案 1/2 联动） |
| B5 | 帧数守恒 | 超分不改帧数，每段 `decoded == expected == 源帧数` |
| B6 | 无回归 | 不得破坏 IFRNet 侧任何一处修复 |

### 10.6 ✅【任务 A′ · 已于 2026-09-09 完成】钳制失败语义的死锁隐患（IFRNet 侧）

> **⚠️ 本节是立项时的原始分析，任务已于 2026-09-09 完成。**
> **最终结论与落地改动见 §14**（根因与此处推测不同：真正主因是站点 1 的 fall-through
> 而非单纯的 pending 滞留）。以下保留存档。

IFRNet 侧现有的钳制失败语义是「**放弃本轮、不推进 `_output_slot_idx`、交由后续 drain /
段末 EOS 回收**」。**这个语义在 ESRGAN 侧被实测证明会死锁**，IFRNet 侧存在同类隐患。

**待修位置**：`external/ifrnet_video/nvenc_sdk.py:1554-1567`
（注释仍写着「放弃本轮，不推进 `_output_slot_idx`」）。它属于经过大量标定的 HEVC 排空
热路径，改动风险较高，**必须配故障注入验证**：用 `NVENC_HEVC_DRAIN_MARGIN` 等现有开关
无法触发钳制，需临时把 `_max_valid_bitstream_bytes` 改小（可参照 ESRGAN 侧新增的
`ESRGAN_NVENC_MAX_BS_BYTES` 覆盖方式，为 IFRNet 侧加一个同款环境变量）。

**故障注入复现**（`ESRGAN_NVENC_MAX_BS_BYTES=1024`，60s/3 段素材）：
```
[NVENC-Enc] ⚠️ 非法 bitstream size #1: size=1734, cap=1024 (slot=6, fi=42) — 放弃本轮，不推进指针
[优化流水线]: 59%|█████▊ | 528/900 [09:51<06:56, 1.12s/frame, ... S:0/O:16]   ← 永久卡死
```
- **只需 1 次**钳制即死锁。原因是该槽的 pending 记账永久滞留，
  9 个槽全部占满后 `_ensure_slot_free` 无限自旋。
- 症状：**GPU 利用率 0% + 显存 10.2 GiB 持续不退 + 进程 CPU 100% 自旋 + 日志停止刷新**，
  无任何报错，极难诊断（本次是靠用户肉眼发现）。
- 注意：`nvidia-smi` 在容器里报告的是**宿主机 PID**，与本容器 PID 对不上，排查时别被误导。

**正确做法（已在本仓库 ESRGAN 侧落地）**：
`LockBitstream` 既然已返回 **SUCCESS**，就说明该槽输出**已被取出**，只是 size 不可信；
因此必须 **unlock → 强制消费该槽（`_slot_pending.pop` + `_output_slot_idx += 1`）→ 丢弃该帧**。

```
修复前（错）：放弃不推进 → 1 次钳制即永久挂死
修复后（对）：强制消费 + 丢帧 → EXIT=1、1分26秒正常退出、GPU 释放、
            859 次强制消费、明确报
            ❌ 解码级验收失败: decoded=33 expected=900 reason=decoded_frame_mismatch
```
**设计原则：宁可丢帧被验收明确判失败，也绝不允许静默挂死。**

### 10.7 其它注意事项

- **不要**在钳制失败分支里重试 Lock（铁律 2/3：重锁同一槽在 HEVC 下永久阻塞）。
- 上界用**输出**分辨率计算，用错成输入分辨率会让正常码流被误判为非法。
- 已在 ESRGAN 侧 `__init__` 提供 `ESRGAN_NVENC_MAX_BS_BYTES` 覆盖开关，专门用于故障注入回归；
  生产环境**不得**设置该变量。IFRNet 侧同款开关 `IFRNET_NVENC_MAX_BS_BYTES`
  **已于 2026-09-08 补齐**（§5 锚点表 548–554 行），任务 A′ 的故障注入即使用它。
- 本任务与「已完成的 ESRGAN 方案 1（HEVC/AV1 跨段不复用）+ 方案 2（按会话代数保留 SPS/PPS）」
  是**互补**关系：方案 1/2 解决参数集缺失，本任务解决越界读崩溃，两者不可互相替代。

---

## 11. 附录：ESRGAN 侧对照检查结论（2026-09-04 实测）

立项时已对 ESRGAN 侧做过一次完整对照，结论如下，**承接方无需重做**。

### 11.1 结论：ESRGAN 侧**不存在** H.264 + LA>0 排空缺陷

任务 A 的 `strict drain abandoned` **不会**在 ESRGAN 侧出现——该侧根本没有
counted-drain 守卫与 `_strict_eos`，结构上不具备产生该错误的条件。

### 11.2 两侧结构对比（行号均指各自 `nvenc_sdk.py`）

| 检查项 | IFRNet 侧 | ESRGAN 侧 |
|---|---|---|
| `_hevc_ready_count`（counted-drain 守卫） | 有（1655） | **无** |
| `_HEVC_DRAIN_MARGIN` | 有（1653，默认 2） | **无** |
| `strict drain abandoned` 抛错 | 有（1831） | **无** |
| `_strict_eos` | 有（550，默认 1） | **无** |
| `_ensure_slot_free` 就绪门控 | `_ready<=0` 时**禁止** Lock | 无门控；`_guard` 超限即直接 blocking Lock 探测（更宽松） |
| `_required_buffers` | `max(1, la_depth+3)`（582） | `max(1, la_depth+1)`（592） |
| `_drain_order` | 有 | **有**（1957 / 2737） |
| `_is_legal_bitstream_size` | 有（5 处） | **✅ 已有（5 处，2026-09-05 补齐）**（原为 0 处） |
| 钳制失败是否强制消费 | ❌ 未修（**任务 A′**） | ✅ 已强制消费，不会死锁 |

⚠️ 推论：若将来要给 ESRGAN 移植 counted drain，
`_required_buffers` 必须**同步**从 `la+1` 改为 `la+3`，否则会重演
RESUME.md 第 106 行记录的「循环依赖」。

### 11.3 ⚠️ `RESUME.md` 第 6 节有一处描述已过时（不要照抄）

原文称 ESRGAN 侧的 EOS/flush 排空「**仍是** `sorted(self._slot_pending.keys())` 顺序，
未做 IFRNet 侧已有的 `_drain_order` 顺序保持修复」。

**实测：该说法已过时**——ESRGAN 侧 `_drain_order` 已存在于 1957 / 2737 两处。
第 6 节的另一处短板（未装 `_is_legal_bitstream_size` 钳制）**依然成立**，即本文件的任务 B。

### 11.4 与本次 R1 故障的关系（重要，避免误判方向）

2026-09-04 的 R1 中 ESRGAN 片段 2 曾 exit=139（SIGSEGV），
崩溃栈落在 `realesrgan_video/nvenc_sdk.py:2884 _close_driver_session`（DestroyEncoder）。
**但根因不是任务 B，也不是任务 A**，而是：
「跨段复用 HEVC 编码器 + `[FIX-SKIP-REOPEN]` 会话跳过重建 → 新段缺 SPS/PPS」
→ `PPS id out of range: 0` ×224 → muxer 拒绝写头 → 管道断裂 → 清理阶段 DestroyEncoder 触发 SIGSEGV。

该根因已由**方案 1（HEVC/AV1 跨段不复用，`ESRGAN_NVENC_CROSS_SEGMENT_REUSE` 默认 0）**
与**方案 2（`_stream_begin` 按会话代数条件保留 `_cached_sps_pps`）**修复并验证
（60s / 3 段素材：默认档 3/3、`REUSE=1` 档 3/3，均 EXIT=0）。
**承接方请勿把这三者混为一谈。**

---

## 12. 相关文档与工具

- 交接文档：`temp/retest/handoff/RESUME.md`（第 3 节修复清单、第 2 节三条铁律；
  ⚠️ 第 6 节关于 ESRGAN 排空顺序的说法已过时，见 §11.3）、
  `temp/retest/handoff/CHANGES.md`、`temp/retest/handoff/commands.md`
- 本次全部改动备份：`temp/retest/patched_backup/`
  （`ifrnet_nvenc_sdk_patched.py` / `ifrnet_pipeline.py` / `realesrgan_nvenc_sdk.py` /
   `realesrgan_main.py` / `realesrgan_processor_video_optimized.py` / `src_video_utils.py`）
- 编码层回归：`tests/repro_ifrnet_lookahead.py`（⚠️ 已修复其 `mux()` 硬编码 `-f hevc` 的 bug，
  否则 H.264 用例会假失败）
- CUDA context 语义探针：`tests/probe_cuda_context.py`（7 种模式）
- 鬼影/切镜分析：`tests/analyze_interp_ghost.py`、`tests/diagnose_scene_cut_ghost.py`、
  `tests/calibrate_scene_cut_threshold.py`、`tests/verify_scene_cut_fix.py`
- libcuda 调用日志钩子：`tests/sitecustomize.py`
  （`NVENC_CTXLOG=/abs/log PYTHONPATH=/workspace/Video_Enhancement/tests python ...`）
- 帧数探测优化校验：`tests/test_frame_count_probe.py`（2026-09-05 新增，验证 P0–P4）
- **R1 现场交接 / 断点续跑手册**：`temp/retest/HANDOFF_RESUME.md`
  （含本次会话全部 10 项修复清单、9 个文件 md5、续跑步骤、未决问题）

### 12.1 可复用的性能探针（2026-09-05 已埋点，排查同类问题直接用）

排查「GPU 饿死 / 速率腰斩」时，以下探针已在 `external/ifrnet_video/pipeline.py` 中埋好，
每 50 批打印一次均值：

| 探针 | 位置 | 输出 |
|---|---|---|
| `[PROF-Infer]` | 推理线程 `_safe_infer` | `safe_infer=XXX ms` |
| `[PROF-Writer]` | writer 线程 GPU_RAW 分支 | `cat+nv12 / sync / d2h / order / submit` 分项 |
| `[PROF-Loop]` | 推理线程整轮 | `整轮 / safe_infer / infer后处理 / infer前等待` |
| `[PROF-PINNED-FALLBACK]` | pinned 分配失败兜底 | 降级次数（**正常必须为 0**） |

```bash
# 读取方式（进度条用 \r 分隔，必须 tr 后才能正确切分）
tr '\r' '\n' < temp/retest/R1_log1_full_end2end.txt | grep -o "\[PROF-[A-Za-z-]*\][^|]*"
```

⚠️ 采样 GPU 时注意：`nvidia-smi --query-gpu=utilization.gpu` **只统计 SM/3D**，
**不含 NVENC 编码单元**。要查编码瓶颈必须加 `utilization.encoder`：
```bash
nvidia-smi --query-gpu=utilization.gpu,utilization.encoder,memory.used,power.draw --format=csv,noheader
```

---

## 13. 【任务 A · ✅ 已完成】H.264 + LA>0 排空缺陷

> ### ✅ 本节为**已完成记录**，不是待办。承接方请勿重复实施，仅作回归依据。

### 13.1 根因结论：确认是 counted-drain 守卫在 H.264 上的**过度约束（假阳性）**

§6 的首要假设成立。`_hevc_ready_count` 是按「HEVC 对未就绪槽 Lock 会永久阻塞」标定的保守口径，
却被无条件套用到 H.264；H.264 空槽本就返回 `SUCCESS + size=0`（不阻塞，铁律 2 后半句），
于是 `_ready<=0` → 禁止任何 Lock → 永远摸不到目标槽 → `_guard` 超限 →
`_strict_eos=1` 下直接 `raise strict drain abandoned target slot`。**不是真的排不出来。**

### 13.2 落地改动（`external/ifrnet_video/nvenc_sdk.py`，2026-09-08 会话）

按 §7 **首选方案**（按 codec 分流）实施，未新增 `NVENC_H264_DRAIN_MARGIN`：

| 锚点 | 改动 |
|---|---|
| `_drain_outputs_blocking`（`[FIX-H264-LA-DRAIN-READY]`） | H.264 + LA>0 在 Lock 前按目标槽最旧 pending 的 gfi 与 `la_depth+1` 判就绪，不再走 `_hevc_ready_count` |
| `_ensure_slot_free`（codec 分流分支） | HEVC/AV1 才用 `_hevc_ready_count`；H.264 走独立 `la_depth+1` 就绪检查，绕过「`_ready<=0` → 禁止 Lock → raise」假阳性链路 |

`encode_frames_stream` 的 per-frame drain 与 EOS drain 早已是
`if self._codec in ("hevc", "av1")` 条件包住 `_hevc_ready_count`，未改动。

### 13.3 复测结果（2026-09-09）

- 4 组矩阵（H.264/HEVC × LA=0/8，各 900 帧 × 3 段）**4/4 ALL PASS**，详见 §8。
- H.264 + LA=8 稳定性：chunk=64/5 段、chunk=256/3 段、chunk=128/6 段共 **14 段全过**。

---

## 14. 【任务 A′ · ✅ 已完成】IFRNet 侧钳制失败语义「强制消费」+ 故障注入收敛

> ### ✅ 本节为**已完成记录**，不是待办。承接方请勿重复实施。
> A′-2/A′-3 的完整诊断过程另见 `Plan/IFRNet_尺寸钳制强制消费遗留_立项Prompt.md`。

### 14.1 根因（2026-09-09 诊断，与 §10.6 的推测不同）

§10.6 推测的「pending 滞留导致槽位占满」只是次生问题。**真正主因是站点 1 的 fall-through**：

`_drain_outputs_blocking` 的钳制分支设 `_got_valid = True` 后 break，控制流会继续落到下方
`bitstream_size` 读取处 —— 按**垃圾 size 再次 `from_address()`**（越界读 → SIGSEGV 隐患），
并**二次推进 `_output_slot_idx`**。每帧净推进 2 → 相位漂移 → 段末 EOS 排空据此锁到未就绪槽
→ HEVC 下 `doNotWait=0/1` 均**永久阻塞**。

faulthandler 定位（修复前）：`encode_frames_stream` EOS `LockBitstream`（HEVC）。
`[PROF-FAULT]` 实测 `lock=0.49ms`，**否证**了「时间耗在阻塞 Lock」的假设 H1。

### 14.2 落地改动（`external/ifrnet_video/nvenc_sdk.py`）

| # | 改动 | 标记 |
|---|---|---|
| 1 | 新增 `_sizecap_force_consume(slot_idx, size, site, pending_map=None)` 统一语义：unlock → 清 pending → 推进指针 → 丢帧 → 计数 → 熔断检查 | `[FIX-SIZECAP-FORCE-CONSUME]` |
| 2 | 新增 `_slot_of_bs_handle()`：由 bs_handle 反查物理槽号（供站点 2/3） | — |
| 3 | 站点 1 修 fall-through：置 `_forced_drop` 跳过数据读取与二次推进 | `[FIX-SIZECAP-NO-FALLTHROUGH]` |
| 4 | 站点 2（`_lock_bitstream_blocking`）/ 3（`_lock_bitstream_with_retry`）：命中即强制消费 | 同 1 |
| 5 | 站点 4（EOS 排空）/ 5（`flush`）：保留 deadline 重试，到期后转强制消费并 `continue`（原为 break/raise，留下悬空记账） | 同 1 |
| 6 | **修 `close()` 销毁顺序**：改为 NVIDIA 官方顺序（先 `DestroyEncoder` 再 `_destroy_all_slots()`） | `[FIX-CLOSE-ORDER]` |
| 7 | `tests/repro_ifrnet_lookahead.py` 加 try/finally 保证异常路径也 close | `[FIX-REPRO-CLOSE]` |

**第 6 项是故障注入下 exit=139 的直接根因**：原实现先 `_destroy_all_slots()` 后
`DestroyEncoder`，段中途异常退出时驱动 LA 管线仍持有这些 buffer 的引用，
先释放 buffer 会让 `DestroyEncoder` 踩到已回收内存 → SIGSEGV（faulthandler 定位 `close`），
且**掩盖了真正的失败原因**。正常（已排空）路径两种顺序等价，故无回归风险。

### 14.3 新增环境变量与标定依据

| 变量 | 默认值 | 依据 |
|---|---|---|
| `IFRNET_NVENC_SIZECAP_ABORT` | `32` | 单段内强制消费预算。正常工况钳制恒为 0 次（§8 第 7 项实测）；32 帧丢失已必然触发段级帧数不守恒，故以此为 fail-fast 阈值。`0` = 禁用熔断（**仅用于诊断**：实测禁用后 EOS 排空仍会永久阻塞，exit=124） |
| `IFRNET_NVENC_PROF_FAULT` | 关（需显式 `=1`） | `[PROF-FAULT]` 计时探针。`_drain_outputs_blocking` 是每帧热路径，无条件每 50 轮打印会淹没生产日志；排查同类问题时才开 |

⚠️ **熔断是防挂死的实际兜底**：ctypes 无法给 FFI 调用加超时，一旦 LockBitstream 在驱动内
永久阻塞，Python 层的熔断也救不回来 —— 所以必须在走到 EOS 排空之前就熔断。

### 14.4 配套工具（2026-09-09 新增）

- `tests/_faultinj_wrap.py` —— 故障注入包装器：`faulthandler.enable()`（SIGSEGV 也 dump Python 栈）
  + `dump_traceback_later(N)` 定位挂死点。
  ```bash
  env IFRNET_NVENC_MAX_BS_BYTES=1024 python tests/_faultinj_wrap.py 120 \
      --frames 100 --codec hevc --la 8 --rc vbr_hq --qp 21 --chunk 32 --segments 1 \
      --out temp/retest/x.mp4
  ```
- `tests/repro_real_frames.py` —— 真实素材编码回归（验证正常工况钳制 0 次触发）。
  ```bash
  python tests/repro_real_frames.py -i benchmark_output/word_world_2_v6.4.5_x2.mp4 \
      --codec h264 --la 8 --frames 300
  ```

改动前备份：`temp/retest/A1/nvenc_sdk_ifrnet_BEFORE_A2.py`（md5 `4a61ec75d0a5dbc97aeb3d07fa722069`）。
