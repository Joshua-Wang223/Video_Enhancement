---
name: P3-2 切镜预扫描 / PROBE-OPT 帧数预热的断点恢复分析
description: 切镜预扫描与帧数预热都是进程内缓存、断点重启必重算；2026-09-23 已按路线 B 落盘 sidecar 实现，含三个落盘陷阱的规避
type: project
---

**事实（2026-09-23 用户提问「[P3-2]/[PROBE-OPT] 能否也断点恢复，不必每次重算」）**：
`[P3-2] 切镜预扫描`（`_SCENE_CUT_CACHE`，`external/ifrnet_video/pipeline.py`）与
`[PROBE-OPT] 帧数预热`（`_PROBE_FRAME_CACHE`，`src/utils/video_utils.py`）都是**纯进程内 dict**，
进程退出即归零；且两者都在 `_process_segments()` **之前**对**全部** `segment_files` 执行，
不理会 checkpoint 的 `processed_segments`（断点跳过在更晚的 `_process_segments` 里才生效）。
实测量级：24 分钟 / 5 段视频，每次 resume 白付 ~28.3s（切镜）+ ~9.4s（帧数）。

**Why**：分段文件在 resume 时走 `split_video_by_time(reuse_existing=True)` 复用分支，
字节与 mtime 不变 → 两个缓存的 key（`path+size+mtime_ns`）跨进程稳定 →
缓存值本质是「文件内容的纯函数」，落盘后跨进程复用是安全的。

**关键非平凡结论（路线 A 不能套用到 PROBE-OPT）**：
- 只扫「未完成段」（按 checkpoint 过滤）对 `[P3-2]` 是净赚：跳过段不进 `_process_segment`，切镜结果无人消费。
- 但对 `[PROBE-OPT]` 收益打折甚至倒退：`_process_segments` 的批量验收对**全部**
  `segment_files`（含已跳过段）计算 expected 帧数，滤掉后这些段会退化成**串行**补算
  （每段 ~6.9s），把并行省下的又吐回去。

**推荐路线 B（真正断点恢复）**：把两个缓存落盘为 checkpoint **同目录的 sidecar JSON**
（如 `probe_cache.json` / `scene_cuts.json`），启动加载、命中即跳过；只持久化**成功项**，
用 tmp + `os.replace` 原子替换（沿用 `_save_checkpoint` 的 `[P1-FIX-ATOMIC]` 模式）。
副作用：PROBE 落盘后验收阶段也全部命中，顺带补上路线 A 的短板。数据量极小（5 段→几 KB）。

**三个必须处理的落盘陷阱（不处理会比不落盘更危险）**：
1. **不能持久化「失败」**：`prescan_scene_cuts._one()` 异常时也写 `set()`，与「真无切镜」不可区分；
   帧数失败会缓存 `(None,'decode')`。原样落盘 = 一次瞬时 ffmpeg 失败被永久固化 →
   静默跳过插值 / 验收门失效。须带成功标记，失败项不写。
2. **切镜 key 缺 `IFRNET_SCENE_CUT_THRESHOLD`**：现 key 只有 `path|size|mtime_ns`，
   进程内生命周期短看不出问题，落盘后改阈值会读到旧结果，须把阈值 + 判据版本号并入 key。
3. **`_PROBE_DETAIL_CACHE` 与 `_PROBE_FRAME_CACHE` 分离**：只落帧数不落
   `error_lines/rc/hw_failed`，恢复后 `validate_decodable_video` 仍会为错误检查再解码一遍。

**How to apply**：截至 2026-09-23 用户**尚未确认是否落地**，只要求先给分析+方案。
若要实现，建议路线 B 单独做（叠加 A 已冗余），用现有 `IFRNET_SCENE_CUT_PRESCAN` /
`NVENC_PREWARM_PROBE` 作回滚开关，另加独立持久化开关便于对照。
验证方式：同输入连跑两次 + 中途 kill 再恢复，断言第二次切镜集 / 帧数与第一次一致
（两者均为纯函数，可逐值对拍）；无 GPU 环境亦可覆盖大部分。

---

**✅ 已落地（2026-09-23，用户确认「按方案 B 落地」）**

实现要点（路线 B 单独做，未叠加路线 A）：
- `video_utils.py`：新增 `set_probe_cache_file()` / `_load_probe_cache()` /
  `_persist_probe_cache()` / `_probe_detail_persistable()`；sidecar 形如
  `{version, frames:{key:{value,kind}}, detail:{key:...}}`，tmp+`os.replace` 原子写。
  key 用 `json.dumps(list(k))` 编码（避免路径含分隔符歧义）。
- `external/ifrnet_video/pipeline.py`：`_SCENE_CUT_CACHE_OK` 集合区分「扫描成功」
  与「失败退化」；`_resolve_scene_cut_threshold()` 成阈值唯一入口，
  `_scene_cut_cache_key(path, threshold)` 并入阈值 + `_SCENE_CUT_CRITERIA_VERSION`；
  `_scan_scene_cuts()` 不可用时返回 `None`（不再与真空集混淆）。
- processor 侧：IFRNet `_configure_prescan_caches()`、ESRGAN `_configure_probe_cache()`，
  在 `_setup_temp_dirs`/`_load_checkpoint` 之后调用，sidecar 取 checkpoint 同目录
  （`probe_cache.json` / `scene_cuts.json`），随 `temp/{stage}/{prefix}_{video}/` 清理。
- 回滚开关：`NVENC_PROBE_CACHE_PERSIST=0` / `IFRNET_SCENE_CUT_CACHE_PERSIST=0`
  （另保留原 `NVENC_PREWARM_PROBE` / `IFRNET_SCENE_CUT_PRESCAN`）。

验证（全部通过）：
- 新增 `tests/test_prescan_cache_persistence.py`（14/14，纯桩不调 ffmpeg）：成功落盘、
  重启命中不重算、失败（None/异常）不落盘、metadata 不落盘、detail 只落成功证据、
  阈值/判据版本入 key、真空集落盘且命中。
- 真实文件双进程 E2E（`ffmpeg testsrc` 生成 3s clip）：进程 A 冷跑落盘，
  进程 B 重启后 `decode` 0.3ms + `cache_hits 0→1`、`prescan` 0.0ms 且 sidecar 载入 1 段。
- 门禁静态子集 `verify_plan_implementation.py --skip-behavior`：49/47/0/2，与基线一致。

**遗留边界（未改，非本次范围）**：`validate_decodable_video_batch` 若走
`parallel_mode="process"`，子进程不继承 `_PROBE_CACHE_FILE`，其帧数/证据仍会重算
（默认 `thread` 模式不受影响）。
