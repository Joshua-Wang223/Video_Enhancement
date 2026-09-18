---
name: nvenc-eos-drain-slot-order-tail-corruption
description: HEVC/AV1 + LA>0 段尾 5~8 帧参考链断裂（"Could not find ref with POC X"）根因 = EOS 排空按物理槽号升序而非帧输出顺序；ESRGAN 侧漏了 IFRNet 侧已有的 [P2-FIX-EOS-OUTPUT-ORDER]
metadata:
  node_type: memory
  type: project
---

# NVENC EOS 排空顺序错误 → 段尾帧乱序（HEVC/AV1 + LA>0）

## 症状

段级解码验收失败（帧数守恒、但尾部不可解码）：

```
❌ 解码级验收失败: decoded=723 expected=723 reason=decode_errors
[hevc @ ...] Could not find ref with POC 54 / 53 / 52 / 51 ...
```

- **帧数守恒**（`written == submitted`），`empty=0` → 包级/帧数守恒校验完全盲区，只有
  `ffmpeg -v error` 全解码才能发现。
- 只发生在**跨段复用会话的第 2 段起**；同一进程内有的段坏、有的段好。
- 与源内容有关，但**与源是否损坏无关**：`--normalize-source` 重跑仍然复现。

## 根因（代码级，已实测确认）

`NVENCEncoder.encode_frames_batch()` 的 `send_eos=True` 分支（EOS 后逐槽排空）：

```python
_drain_slots = sorted(self._slot_pending.keys())   # ❌ 按物理槽号升序
```

`slot = gfi % slot_count`（LA=8 → slot_count=9）**在段尾回绕**，升序槽号 ≠ 帧输出顺序。
被排空的"上一 chunk 延迟帧"（local fi<0）按取出顺序 append 进 `_prev_chunk_outputs`，
而 `_write_la_output()` 原样按该顺序把它们写在当前 chunk 帧之前 → 段尾乱序。

实测（3 段 × 723 帧，同内容，NVENC_EOS_DEBUG=1）：

| 段 | 起始 gfi | pending 槽（升序） | 实际写出顺序 | 结果 |
|----|---------|------------------|-------------|------|
| seg0 | 0    | 4,5,6,7,8,0,1,2 | 715..719  720,721,722 | 恰好正确 |
| seg1 | 723  | 7,8,0,1,2,3,4,5 | **717,718,719,715,716**,720,721,722 | ❌ 尾帧乱序 |
| seg2 | 1446 | 1,2,3,4,5,6,7,8 | 715..722 | 正确 |

触发条件：段尾 LA 滞留帧的槽号窗口**跨过 slot_count 回绕点**且回绕点落在
"属于上一 chunk 的帧"中间 → 由 `段起始 gfi % slot_count`、`总帧数`、`chunk(128)`、
`la_depth` 共同决定 → **表现为输入相关**、`new5.mp4` 不触发纯属对齐运气。

## 修复

过滤空槽时保持 `_output_slot_idx` 起始的全局轮转顺序（对齐
`external/ifrnet_video/nvenc_sdk.py` 已有的 `[P2-FIX-EOS-OUTPUT-ORDER]`，
ESRGAN 侧当初迁移时漏了这两处）：

```python
_drain_order = [(_start_slot + i) % self._slot_count for i in range(self._slot_count)]
_pending_slots = set(k for k, dq in self._slot_pending.items() if dq)
_drain_slots = ([_s for _s in _drain_order if _s in _pending_slots]
                if _hevc_eos else _drain_order)
```

**⚠️ 状态纠正（2026-09-18）**：本条原写「已修」与**实际不符** —— 核查
`git show HEAD:external/realesrgan_video/nvenc_sdk.py` 与工作区，两处 EOS 站点都仍是
`sorted(self._slot_pending.keys())`，且全仓库搜不到 `NVENC_EOS_DEBUG`。即该修复
**当时并未落地**（疑似被后续大重构覆盖，或从未合入）。

**已于 2026-09-18 重新落地并 GPU 验证**：`encode_frames_batch()` EOS 分支与 `flush()`
分支改为「轮转序 + pending 过滤」，并加入 `NVENC_EOS_DEBUG=1` 诊断。实测
hevc LA=8 / 300 帧：`drain_slots=[4,5,6,7,8,9,10,0,1,2]`、`gfi_seq=[290..299]`、
`顺序正确=True`、帧守恒 300==300（解码级 300）。

## 残留（未改，均非生产路径 / 无实际触发）

- `external/IFRNet/process_video_v6_4_5_1_single.py:2588` — 历史单文件副本，同样写法。
- `external/ifrnet_video/nvenc_sdk.py` `flush()` 分支 — LA=0 才走该路径，此时
  `_strm_slot_pending` 恒为空，无影响。

## 验证

- 独立复现（绕过 SR 推理，纯编码压测）：`temp/repro_nvenc_la.py`
  `python temp/repro_nvenc_la.py --frames 723 --codec hevc --la 8 --rate vbr_hq
   --segments 3 --seed 0 --out temp/x.mp4`
  修复前 seg1 13 行解码错误；修复后 3 段全部 0 错误、`顺序正确=True`。
- 生产回归：`wws3e02_26s.mp4` 原命令 → ESRGAN 2/2 段验收通过，最终输出 1202 帧
  `ffmpeg -v error` 零错误（修复前第二段被判败丢弃，输出仅 9.4MB；修复后 23.0MB）。

## 附带发现（独立问题，2026-08-31 验证 new5.mp4 时暴露）

`external/realesrgan_video/main.py` 的 `[FIX-HIGHRES-RC]` 被工作区未提交改动把阈值
**`out_h >= 1080` 放宽为 `>= 2160`** → 2560x1440 输出不再降级为 constqp+LA=0：

- 实测 Tesla T4：1440p + VBR_HQ + LA=8 → SDK `InitializeEncoder failed, code=8`，
  降级 FFmpeg 管道后仍 `CreateInputBuffer failed: out of memory (10)` → 整段编码失败。
- 放宽对 720p 输出（如 wws3e02）**零收益**（720 < 1080 本来就走 LA=8），
  只让 1080p~1440p 失去保护 → 净负收益，已恢复为 1080 并在代码注释里写明实测依据。
- 与段尾乱序根因无关：该场景 SDK 会话都未创建成功，根本没进入 EOS 排空代码。

## 相关记忆

- [[ifrnet-watercolor-tail-defect-investigation]] — 症状 A 的历史记录，当时未定位到本根因
- [[nvenc-stream-drain-backpressure-iron-law]] — 同一模块 drain 消费铁律
- [[hevc-la-open-production]] / [[hevc-la-drain-diagnosis]] — HEVC LA 排空诊断史
