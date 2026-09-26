---
name: 段级验收曾被「容器元数据捷径」静默降级（FIX-GATE-STRICT-COUNT）
description: validate_decodable_video 不传 count_mode 时走 auto，NVDEC 不可用会退回容器 nb_frames 元数据；且帧数缓存 key 不含 mode，预热会把低可信值喂给严格验收门
type: project
---

**结论（2026-09-15 定位并修复）**：验收门的「解码级」语义曾经在**无 NVDEC 的机器上被
静默降级成容器元数据比对**，而且是**缓存导致的跨调用污染**。两条独立成因，缺一不可。

## 成因 1：`count_mode` 缺省 = auto = 可能只读元数据

`validate_decodable_video(..., count_mode=None)` → `count_decoded_video_frames(mode=None)`
→ 取环境变量 `NVENC_COUNT_FRAMES`，默认 **`auto`**。而 auto 的分支是：

1. 先试 NVDEC 硬解计数（真解码）；
2. 若不可用/失败 → `_read_nb_frames_metadata()`：读容器 `nb_frames`，
   与 `duration×fps` 偏差 ≤5% 即**采信并直接返回，完全不解码**；
   只有该函数返回 None 时才回退全解码。

这正是 `[P4-FIX-COUNT]` 注释自己点名的盲区（容器帧数对
"包存在但解码失败/参考链断裂"完全盲区）。

修复前**未传** `count_mode` 的调用点（即会走 auto 的）：

- `src/processors/ifrnet_processor_video_optimized.py`（段内联验收 + 期望值）
- `src/processors/realesrgan_processor_video_optimized.py`（两处：段内联 + 独立段）
- `src/utils/video_utils.py::verify_segment_output`（**该函数当前无任何调用方**，属死代码）
- 两个 processor 计算 `expected_frames` 的 `count_decoded_video_frames(...)`

已固定为 `count_mode="decode"` 的（本来就严格，可作对照）：
`_validate_single_task`（批量段级验收）、`main_video_optimized.py` 的最终门未降级分支。

## 成因 2（更隐蔽）：帧数缓存 key 不含 mode

`_PROBE_FRAME_CACHE` 的 key 只是「绝对路径 + size + mtime_ns」。于是：

> 上游 `count_frames_parallel(segment_files, max_workers=4)` **预热**（默认 `mode=None` → auto）
> 先把**元数据值**写进缓存 → 之后任何要求 `mode='decode'` 的严格调用都会**命中该缓存
> 并直接复用** → decode 语义被彻底架空。

也就是说：只给验收点加 `count_mode="decode"` **改不动任何东西**，必须同时处理缓存。

## 修复（`[FIX-GATE-STRICT-COUNT]`，可 grep）

1. `_PROBE_FRAME_CACHE` 的值由裸帧数改为 **`(帧数, 来源)`**，来源 ∈ `{'metadata','decode'}`；
   `mode='decode'` 的调用**拒绝复用 `metadata` 来源的值**，并在随后用真解码结果覆盖
   （顺带升级 auto 调用方）。
2. 所有门禁调用点显式 `count_mode="decode"`；期望值侧的 `count_decoded_video_frames`
   也一并 `mode="decode"` —— **期望值与实测值必须同源严格**，否则会变成
   "元数据期望 vs 全解码实测"，在无 NVDEC 机器上系统性不等 → **假失败并 unlink 正确产物**。
3. 两个 processor 的预热改为 `count_frames_parallel(..., mode="decode")`，与验收同口径
   （既严格又不重复付费；否则缓存要么低可信、要么白预热）。
4. 回归检查落在 `Accessory/test/test_frame_count_probe.py::check_strict_cache_provenance()`：
   用桩把元数据设为 100、真解码设为 98，断言
   「auto 首次=100/metadata → decode=98 → 缓存升级为 decode → 其后 auto=98」。
   **已做负向对照**：还原成无条件复用缓存时该断言失败（decode 拿到 100）。

## How to apply

- 任何**新增**验收/计数调用，必须显式写明 `count_mode`；不要依赖缺省。
- 看到"加了 count_mode='decode' 却没效果"时，第一嫌疑就是**缓存 key 不带 mode** ——
  本仓有多处 auto 预热（`count_frames_parallel`），它一定先跑。
- 该缺陷的**实际影响面**：只在 NVDEC 不可用/失败且容器 `nb_frames` 可信时显现；
  且多半表现为**假失败**（噪声）而非假通过。生产 GPU 机上 NVDEC 通常可用，故长期未被发现。
