---
name: P3-2 切镜预扫描 / PROBE-OPT 帧数预热的断点恢复分析
description: 切镜预扫描与帧数预热都是进程内缓存、断点重启必重算；2026-09-23 已按路线 B 落盘 sidecar 实现，含三个落盘陷阱的规避；同日补齐收段入口 process_segments_directly（[P3-2-RECEIVE]/[PROBE-OPT-RECEIVE]）
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

**process 模式（2026-09-23 补充 E2E 时发现并修复，`[PROBE-CACHE-PERSIST-PROC]`）**：
- 先前判断「子进程不继承 `_PROBE_CACHE_FILE`、会重算」**是错的**。容器 Python 3.11 +
  Linux 默认 `fork`，子进程**继承**模块全局（含 `_PROBE_CACHE_FILE` 与父进程预热好的
  `_PROBE_FRAME_CACHE`），所以命中本来就有。
- 真正的问题是**反向的**：若父进程未预热（缓存为空），每个子进程各自解码后
  `_persist_probe_cache()` 用「自己那份快照」**覆盖写**同一 sidecar → **最后写者胜**，
  实测 8 段只留 1 条（`validate` 结果仍 8/8 正确，但落盘等于白做）；并发写同一个
  `.tmp` 路径还有交错损坏风险。
- 修复：`_write_probe_cache_merged()` 用 `fcntl.flock` 串行化「读旧 → 合并 → 原子写」，
  临时文件名带 `os.getpid()`；进程内仍由 `_PROBE_CACHE_LOCK` 串行。非 POSIX 无 `fcntl`
  时退化为不加锁（Windows 走 spawn，子进程本就无 `_PROBE_CACHE_FILE`，无此竞态）。
  切镜 sidecar 无多进程写入路径，仅把 tmp 名改成带 pid 作预防。
- 修复后实测：8 段 process 模式 → `entries=8/8`，5 轮 sidecar 均有效。

**验证（补充后）**：`tests/test_prescan_cache_persistence.py` **18/18**（新增 process 模式
E2E：真 ffmpeg 4 段 → 验收 4/4、父进程内存缓存为 0（证明确在独立进程）、落盘 4/4 累积、
重启后 4/4 零解码命中；并用「还原覆盖写」做负向校验确认该断言确实会失败）。

---

**✅ 收段入口补齐（2026-09-23 晚，`[P3-2-RECEIVE]` / `[PROBE-OPT-RECEIVE]`）**

**触发**：用户拿三份日志问「切镜预扫描那段信息为什么不见了」，前提是"同一命令的前后对比"。
拆解后否掉了两个看似合理的解释（都不是根因）：
- ❌「缓存命中所以跳过」——只能解释 `[P3-2] 切镜预扫描: N 段` + 逐段 `[切镜]` 行消失，
  **解释不了** processor 那句**无条件**打印的 `✂️ [P3-2] 切镜预扫描完成: N/N 段` 一起消失；
- ❌「预扫描块被优化删掉」——`git blame` 证明该块自 `0c3badc` 起未被改动。
- ✅ 真根因：那两段日志**根本不是插帧阶段打的**。`🔪 分割视频...` / `✅ 共 N 个片段` 在
  IFRNet（`ifrnet_processor_video_optimized.py:269/284`）与 Real-ESRGAN
  （`realesrgan_processor_video_optimized.py:349/362`）两侧**文案一字不差**；用户跑的是
  `--skip-interpolate`（只有 ESRGAN），所以看到的是超分阶段的分割日志 —— 后面直接接
  `[probe]/[PROBE-OPT]`、没有 `[P3-2]` 就是判别特征。

**顺带查出的真缺口**：IFRNet 的 `process_segments_directly()`（`upscale_then_interpolate`
的 Step 2 = 收上游 ESRGAN 的分段）里**压根没有预扫描块** —— 只有 `_configure_prescan_caches()`
把 `scene_cuts.json` / `probe_cache.json` 两个 sidecar 配好了却**无人消费**（帧数侧靠惰性
`count_decoded_video_frames` 落了盘，慢；切镜侧连并行扫描都没有）。Real-ESRGAN 的
`process_segments_directly()` 同样缺 `[PROBE-OPT]`。

**修法**：两条入口补齐同款两块，位置都是 `_configure_*_cache()` 之后、`_process_segments()`
之前（变量换成 `input_segments`，异常语义与开关完全沿用）：
- IFRNet：`[P3-2]` 切镜预扫描 + `[PROBE-OPT]` 帧数预热；
- Real-ESRGAN：仅 `[PROBE-OPT]` 帧数预热（无切镜逻辑）。

**How to apply（非平凡教训）**：判断「某优化是否覆盖某入口」时**别只看 `process_video_segments`**。
每个处理器有**两条入口**：源片（`process_video_segments`，自己切分段）与收段
（`process_segments_directly`，对接上游产出）。两条各自独立接预扫描/预热，漏一条就会出现
`upscale_then_interpolate` 只有 Step1 享受并行加速、Step2 退回串行的不对称。排查"优化没生效"
类问题时，**先确认日志到底出自哪个阶段**（同文案不同来源是常态），再谈缓存命中。

**验证**：本机为 Windows 开发机（无 torch/pytest/GPU），只做了 `py_compile` +
AST 断言。**完整回归需在 Linux+GPU 侧跑** `tests/test_prescan_cache_persistence.py`。

**门禁新增静态断言 `[FIX-PRESCAN-RECEIVE]`（2026-09-24）**：`tests/verify_plan_implementation.py`
的 `F-修复效果` 段新增一项，用 **AST 取方法函数体切片**（非注释匹配，遵守本文件
"禁止只匹配注释文案"的约定）判定四条组合都接好了线：
IFRNet 两条入口须同时含 `prescan_scene_cuts` + `count_frames_parallel`；Real-ESRGAN
两条入口须含 `count_frames_parallel`（无切镜逻辑，不要求 `prescan_scene_cuts`）；
另要求两个 processor 文件都带 `[FIX-PRESCAN-RECEIVE]` 锚点（标签=代码↔脚本契约）。
- **负向校验已做**（防止"断言恒真"）：把收段入口的 `prescan_scene_cuts(input_segments)`
  调用抹掉 → 该断言转 FAIL；再抹掉 ESRGAN 收段的 `count_frames_parallel` → 仍 FAIL。
  负向脚本一次性执行后即删（在 gitignore 的 `tmp/` 下），未入库。
- 基线：`--skip-behavior` 由 **49/47/0/2 → 50/48/0/2**（多的一项即本断言）。
