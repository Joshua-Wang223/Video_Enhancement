---
name: sps-pps-la-pipe4-startup-corruption
description: LA=8 + pipe=4 启动时 SPS/PPS 损坏机制 — 首帧空帧后 muxer 上下文未建立，_cached_sps_pps+repeatSPSPPS 均失效
metadata: 
  node_type: memory
  type: project
  originSessionId: b6bcafb5-af2b-4b88-95e4-bcb2e53f3525
---

# SPS/PPS 启动损坏: LA=8 + pipe=4 + VBR_HB/QVBR

## 现象

v6.4.3.1, v6.4.4.1, v6.4.5.1 (LA=8 forced pipe=4) 都出现相同错误模式:

```
[NVENC-Enc] ⚠️ 空帧 #1 (ce-pipe fi=7 slot=3 ep_s=0 bs_s=8 la_buf=8)
[NVENCEncoder] Cached SPS+PPS: 49 bytes    ← 缓存成功
[FFmpegMuxer ERR] non-existing PPS 0 referenced  ← 但 muxer 不认！
[FFmpegMuxer ERR] decode_slice_header error
[FFmpegMuxer ERR] no frame!
```

## 完整时序链

### Step 1: LA buffering (帧 0-7)

LA=8 启动时，前 8 帧送入 NVENC 作预测分析:
```
fi=0→slot[0]: EncodePicture → NEED_MORE_INPUT → b""
fi=1→slot[1]: EncodePicture → NEED_MORE_INPUT → b""
...
fi=7→slot[3]: EncodePicture → NEED_MORE_INPUT → b""   (la_buf=8)
```

### Step 2: 首个实帧 IDR (帧 8→slot[0])

帧 8 是首个实帧。EncodePicture SUCCESS。但在 `ce-pipeline` 路径中，
**harvest 发生在下次 slot rotation 时**（帧 12 时 harvest slot[0]）。

### Step 3: Harvest 帧 8 (帧 12→slot[0])
```
Phase 1 (Harvest): cuEventSynchronize(slot[0].ce) → LockBitstream → h264_data
  → _cached_sps_pps = _extract_sps_pps(h264_data)  # 提取 SPS+PPS (48-49 bytes)
  → h264_data = _cached_sps_pps + h264_data         # prepend SPS+PPS
  → writer.write(h264_data)                         # 写入 muxer
```

### Step 4: muxer 解析失败

问题在于：**帧 8 的 IDR 数据是首个写入 muxer 的字节流**。
FFmpeg `-f h264` parser 在收到第一个字节时就开始解析。
但 `_cached_sps_pps + h264_data` 的 prepend 虽然正确将 SPS+PPS 放在
数据流前面，**parser 在收到第一个 NAL 单元 (SPS) 后需要 PPS 来建立
解码上下文**。如果 SPS+PPS 与后面的 slice 数据之间的边界处理有误
（如缺少 `access_unit_delimiter`），parser 可能将相邻的 slice NAL 
错误地解析为 PPS-less 的独立 slice。

另一个可能：`repeatSPSPPS=1` 的 ctypes 设置因 `_NvEncConfigH264`
struct 偏移量不匹配而实际无效 → NVENC 编码的 IDR 帧本身就缺少
SPS/PPS NAL → `_extract_sps_pps` 提取为空 → `_cached_sps_pps = None`
→ 后续所有帧都无法写入正确的 SPS/PPS 头部。

## 三层防御分析

| 防御层 | 机制 | 为什么失败 |
|--------|------|-----------|
| **NVENC repeatSPSPPS=1** | 每个 IDR 前 NVENC 自动重发 SPS+PPS NAL | ctypes struct 偏移不匹配 → 可能无效 |
| **_cached_sps_pps prepend** | 手工提取+预挂 SPS+PPS 到每个 IDR 帧前 | 首帧 prepend 在 muxer 解析器初始化完成前写入 |
| **FFmpegMuxer -f h264** | FFmpeg H.264 parser 自动从 ES 提取 codecpar | parser 需要有效的首个 AU 建立解码器上下文 |

## 根因假说 (待GPU验证)

**假说 1**: `repeatSPSPPS=1` ctypes 无效 — 最可能根因
- `_NvEncConfigH264` 有 ctypes `_pack_=1` 且大量 `reserved` 填充
- 需要与 nvEncodeAPI.h master 逐字段对比偏移量

**假说 2**: LA buffering 空帧干扰 muxer 初始化
- `b""` 被写入 muxer → muxer 尝试解析失败 → parser 状态机进入错误状态
- 后续即使收到正确的 SPS+PPS+IDR，parser 也无法恢复

**假说 3**: prepend SPS+PPS 被延迟写入
- `_cached_sps_pps` 在 harvest 时设置，但此时已在 muxer pipeline 中
- Writer 线程可能在设置 `_cached_sps_pps` 之前已读取该帧结果

## 最终结论 (2026-06-17 更新)

**SPS/PPS 启动损坏是 pipe=4+LA=8 根因的症状，不是独立问题。**

根因修复见 [[pipe4-la8-root-cause-fix]]: 删除 encode_frame() + fi==0 only force_idr
后，零 IDR → 零 LA 开销 → 零空帧 → SPS/PPS 缓存和 muxer 初始化正常进行，
不再有 "non-existing PPS 0 referenced" 错误。

per-slot IDR ×4 导致的 SPS 跨槽注入 (slot0 cache → slot1 prepend) 也一并消除。

---

### 方案B (ctypes 验证) — ✅ 已通过

`tests/test_sps_pps_startup.py` 验证结果:
- `_NvEncConfigH264.repeatSPSPPS` @ offset=88 — 与 nvEncodeAPI.h SDK 13.0 **完全一致** ✅
- 全部 11 个关键字段偏移量均匹配 ✅
- `repeatSPSPPS=1` 可正常设置 ✅

**结论: ctypes 结构体布局正确，`repeatSPSPPS=1` 设置确实传递给了 NVENC SDK。
问题不在 ctypes 层面。**

### 根因重新分析

既然 ctypes 正确，`repeatSPSPPS=1` 确实被设置，那为什么首帧 IDR 仍然缺少 SPS/PPS？

**新假说**: NVENC 硬件在 LA buffering (NEED_MORE_INPUT) 后处理的首个实帧，
即使 `repeatSPSPPS=1`，也可能不输出 SPS/PPS — 这是因为 LA 内部状态机可能认为
"此 IDR 不是 GOP 起始" 或 encoder DPB 尚未完全初始化。

### 推荐方案: Scheme A (muxer 预注入) + Scheme C (首帧 prepend)

既然 ctypes 正确但 NVENC 硬件仍可能不输出 SPS/PPS，
最可靠的修复是在软件层面确保 muxer 收到 SPS/PPS：

**方案A (主防御)**: FFmpegMuxer.write_sps_pps() — 在首帧数据前显式预注入
**方案C (兜底)**: 修复 `_cached_sps_pps` 首帧 prepend 逻辑 (简单一行修复)

具体代码修改见测试脚本输出。方案B不需要任何修改。

**Why:** LA=8+pipe=4 组合在 v6.4.3.1/4.1/5.1 中触发 SPS/PPS 损坏导致丢帧，ctypes 验证通过证明根因在 NVENC 硬件行为，需软件层防御
**How to apply:** 方案A: FFmpegMuxer 新增 write_sps_pps() + NVENCEncoder 设置缓存后回调; 方案C: 修复首帧 prepend 逻辑的一行代码
