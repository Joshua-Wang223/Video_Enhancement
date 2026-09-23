# 最终综合方案：IFRNet 水彩花屏 + HEVC 尾帧损坏修复

## Context

三个方案会话（fd31/fd31_updated/fcd1，2026-08-24）执行后，IFRNet 插帧输出出现"水彩画色彩润开"花屏。两轮独立调查：

- **Claude Code 调查（本会话，2026-08-26）**：已完成双症状定性、码流解剖（段尾 CRA 重启组）、逐帧二分、备份对照，结论存于 `memory/ifrnet-watercolor-tail-defect-investigation.md`。
- **OpenCode 调查（Plan/session-ses_fc2f.md，9236 行）**：独立得出类似双分类（T1 内容级 / T2 码流级），其独有证据为**源视频 `input_videos/word_world_2.mp4` frame 556（0-based）本身就是撕裂帧**（已逐帧复核 554-558），闭合了 word_world 的损伤放大链。

本方案对比两方案后形成最终综合方案。**总判定：OpenCode 方案方向正确但存在两处关键遗漏（S01E12 水彩的管线内机制、doNotWait=1 回退），一处未验证归因（S01E12 源损伤）**；修复面取两方案并集，按证据强度排序。

## 一、两方案对比评估

| 维度 | OpenCode (fc2f) | Claude (本会话) | 裁决 |
|---|---|---|---|
| 双分类 | T1 内容级撕裂（源损伤放大）+ T2 段尾 8 帧参考链断裂 | 症状 B 中段水彩单帧 + 症状 A 段尾不可解码 | 分类一致；**但 T1 归因分歧见下** |
| T2 既有性 | ✓ 备份 la16 复现（1373→1360） | ✓ 备份**同配置 la8** 复现（1373→1368）+ 码流解剖 | 一致：v6.4.5.1 既有缺陷，三方案未引入未修复 |
| T2 机制深度 | EOS 排空缺陷（未解剖码流） | 段尾 VPS/SPS/PPS+CRA 重启组、组内尾帧**编码时即断链**（单独解码验证 16OK+8 坏）、组内容=本段尾帧（frame_0262≈源131）、CRA 触发点疑 `_ensure_slot_free` 兜底 | **Claude 更深**；CRA 触发点仍待生产日志确认 |
| **T1 归因** | word_world：源 frame 556 撕裂实锤（✓ 有效）；**S01E12：推断"源也有 soup"，未验证**（其 565.4s 提取的是合并输出非源） | word_world：frame_0024 继承自 upscaled_12（✓ 与 OpenCode 同链）；**S01E12：输入二分验证 upscaled 7786/7787/8310/8315 干净 → 水彩为 IFRNet 阶段产生**；形态=平滑混叠（放大验证无撕裂带）≠源撕裂 | **Claude 的 S01E12 证据更强**：输入干净+输出混叠 → 管线内产生，非源损伤 |
| 水彩机制 | 未识别管线内机制（其 A1/A3 只治源损伤） | **H2D 预取 pinned 槽无同步竞态**（代码级：`ifrnet_utils.py:295 dst.copy_` 无事件保护；稳态靠 238ms 推理 ≫ H2D 时序巧合，批变小/停顿后窗口打开 → 两对时间不同的帧混入同一 GPU 张量） | **Claude 独有发现**，OpenCode 漏检；其"管线无辜（除放大）"结论对 S01E12 不成立 |
| P2.3 池深缺陷 | ✓ 指出（理论在途 19.7>18）+ free-list 根治 | ✓ 同判（潜伏缺陷，当前节奏打不到） | 一致，纳入 |
| 参数集可疑 | PPS 12B≠11B 注入不一致 | 缓存 33B vs 实际 78B、头部双参数集块 | 一致：需审计 HEVC 参数集识别（P3.2 nal_utils 可切换） |
| doNotWait=1 冲突 | 未提 | **与 diagnose_hevc_la test3/test10 GPU 实测冲突**（挂起/垃圾 size 7.2-21MB），`_ensure_slot_free` 兜底在用 | **Claude 独有发现**，OpenCode 漏检 |
| 验证盲区 | ✓ 指认 verify_plan_implementation.py:1430/1455/1543 用容器 nb_frames | ✓ 同判（包级守恒对"包在解不出"盲） | 一致 |
| 新增建议 | 源预检门禁/代际防护/插值抑制/段级解码闭环/失败语义诊断包 | 验收门禁（decode-error==0+色度簇）/LA=0 规避/判败语义 | 并集互补 |

**两方案分歧裁决（形成本方案的关键）**：
1. **S01E12 水彩 = 管线内产生**（T1b），不是源损伤（T1a）。依据：输入三分位干净（实测）、输出平滑混叠（放大图）、机制代码级成立。OpenCode 的"565.4s soup 同源"是把合并输出的水彩当成源证据的循环论证，且其源位置未验证。
2. **T1a（源损伤放大链）对 word_world 成立**：源 556 撕裂（OpenCode 实锤）→ upscaled_12（我验证）→ frame_0024（我验证）→ 最终输出 22.2s。两方案证据互补闭合。
3. 推论：**两条水彩路径并存**——源撕裂放大（word_world，管线无辜）+ 预取竞态混叠（S01E12，管线产生）。修复必须覆盖两者；OpenCode 的 A1/A3 只治前者。

## 二、最终综合方案（按优先级）

### P0 立即可执行（零代码，先出干净片）
1. 两个任务以 **HEVC+LA=0**（`--lookahead-depth-ifrnet 0`，ce_pipeline 已 GPU 验证）或 **H.264 NVENC** 重跑。
2. 重跑前源预检：`ffmpeg -v error -i <src> -map 0:v -f null -`（结构）+ 抽样帧检（撕裂/混叠）。
3. 输出验收双命令：`ffprobe -count_frames` 比对 nb_read_frames==nb_frames；`ffmpeg -v error -i <out> -f null -` 无输出。

### P1 水彩管线内机制根治（T1b，小改，先做）
**预取槽事件同步** — `external/ifrnet_video/pipeline.py` `_try_prefetch_next`（L1135-1197）+ `external/ifrnet_video/ifrnet_utils.py` `PinnedBufferPool`（L266-296）：
- 池增加每槽最近 H2D 事件（`torch.cuda.Event`，在 `img0_pin.to(device, non_blocking=True)` 后于 stream_h2d 上 `record_event`）；
- `get_for_frames` 覆写该槽前 `event.synchronize()`（正常路径事件早已完成，零开销；同步路径 `frames_to_tensor` 同享保护）；
- 根治"CPU 覆写 pinned 与在途异步 H2D 竞态"——稳态时序巧合不再作为正确性前提。

### P2 段尾损坏（T2）定位与修复
1. **定位**（先于改码）：grep 生产运行日志中"排空超限/空帧占位兜底/帧数守恒校验失败"出现位置，与段尾 CRA 重启点比对；扩展 `tests/diagnose_hevc_la.py` 增加"槽位放弃后继续提交"变体复现重启组。
2. **修复**（`external/ifrnet_video/nvenc_sdk.py`）：
   - EOS 排空硬化：EOS 后循环 LockBitstream 至全槽排空；code=8 有界重试（≥3）；仍有 pending → **段判败**（`encode_frames_stream` EOS 段 L2040-2162 + `flush()`）；
   - **禁止 prev 占位写入正式输出**（L1523-1525、L1559-1578、L2144-2156、编码线程 L3106-3114）：占位保帧数但产出不可解码 AU；改为判败 → 既有 checkpoint/resume 重处理该段；
   - `_ensure_slot_free` 兜底语义：放弃槽位（L1559-1578）应判段失败而非继续提交（当前嫌疑为 CRA 重启组触发点）。
3. **规避路由**：`main.py`/config 增加 HEVC+LA 组合默认降 LA=0 的开关（`models.ifrnet.hevc_la_disable`，默认 true + 日志明示），根治前生效。

### P3 结构性隐患排雷
1. **P2.3 pinned 池不变量**（`pipeline.py` L1319-1355）：池深 ≥ `qdepth + ceil(chunk_frames/batch_frames) + 2`；推荐根治：编码线程 H2D 完成后归还显式 free-list（`queue.Queue`），生产者取用/空则新分配。
2. **回退 `[P0-FIX-HEVC-HANG]` 的 doNotWait=1**（`_lock_bitstream_blocking` L1582-1646）：HEVC/AV1 恢复 blocking（doNotWait=0），死锁防护改为外部 watchdog 线程；若保留轮询，必须对返回 size 钳制 ≤ W×H×4（防 test10 垃圾 size 竞态写入流）。
3. **HEVC 参数集识别审计**（`_extract_sps_pps`/`_has_sps_pps`/muxer `_es_has_param_sets`）：33B 缓存 vs 78B 实际、头部双参数集块——切换至 `external/nvenc_common/nal_utils.py`（P3.2 参考实现，回归测试 D 组已锁行为等价）；注入前与流内当前参数集字节比对，不一致用流内新版并告警（OpenCode B4）。

### P4 帧校验与错误处理机制（用户点名项）
1. `tests/verify_plan_implementation.py` RT 阶段（L1430/1455/1543）容器 `nb_frames` 改为 `-count_frames` 的 `nb_read_frames`，并新增"输出 decode-error==0"断言（`ffmpeg -v error -i out -f null -` 空输出）。
2. 段级解码闭环（`src/main_video_optimized.py` 或 processor 层）：每段 mux 完成即解码验证，不等 → 判败重处理（与 P2 判败共用实现）。
3. 最终合并输出再验证一次（concat 也可能引入问题）。
4. 失败语义：段败 → checkpoint/resume 自动重处理；连续 N 次失败 → 输出 `_diag_*` 诊断包并终止。
5. 源预检门禁（OpenCode A1）：处理前结构校验 + 抽样损伤检测（撕裂行相关/帧差尖峰），命中列出坏帧号，默认拒绝、`--allow-damaged-source` 放行。

### P5 源损伤处置（T1a，可选项）
1. 代际防护（OpenCode A2）：输出写 `encoding_generation` metadata，检测输入为本管线产物且源有损伤 → 告警（损伤代际累积）。
2. 损伤帧插值抑制（OpenCode A3，可选）：reader 层标记可疑帧，插帧时对标记帧跳过插值（复制相邻帧），阻断 1→2.5 帧扩散。

## 三、实施顺序与验证

1. **先验证后修复**（P2 定位先行）：拿两份生产运行日志（word_world/S01E12 两次测试）比对"排空超限/占位兜底"位置 → 确认/否决 `_ensure_slot_free` 为 CRA 触发点；同时确认 S01E12 输入视频路径，对 S01E12 源做一次二分（若源干净 → P1 为唯一水彩机制；若源有损伤 → 按 T1a 补充标记）。
2. **代码改动顺序**：P1（独立小改）→ P3.2/P3.3（回退与审计）→ P2（判败语义，依赖定位结论）→ P4（验收接线，随 P2 同步）→ P5（可选）。
3. **GPU 生产验证矩阵**（Linux/T4）：
   - 修复后同一命令重跑两个任务：`ffprobe -count_frames` 帧数守恒 + `ffmpeg -v error` 零错误 + 色度坏帧簇=0（`tests/verify_segment_bitstream_v4.py` 检查4）；
   - HEVC+LA=8 与 HEVC+LA=0、H.264 三组合对照（P2 修复后 HEVC+LA=8 应无段尾断链）；
   - 备份对照复测：备份 benchmark 同文件应仍报同样 POC 错误（证明修复前后差异定位准确）。
4. **回归**：`tests/verify_plan_implementation.py` 全量（含 D1 改造后断言）+ `tests/test_regression_min.py` 34/34。

## 四、待用户确认事项

1. 两份测试的**完整运行日志**（含"排空超限/空帧占位兜底/守恒校验失败"行）——P2 定位输入。
2. **S01E12 输入视频的确切路径**——T1a/T1b 裁决的最终闭环。
3. P3.2 doNotWait=1 回退是否接受（会改变 HEVC 死锁防护策略，需 GPU 复测挂起场景）。

## 关键文件清单

- `external/ifrnet_video/pipeline.py`（P1 预取同步；P3.1 池深；P2 无）
- `external/ifrnet_video/ifrnet_utils.py`（P1 池事件）
- `external/ifrnet_video/nvenc_sdk.py`（P2 EOS 硬化/占位禁止/兜底语义；P3.2 doNotWait 回退；P3.3 参数集）
- `external/ifrnet_video/main.py`（P2 规避路由开关）
- `external/realesrgan_video/*`（镜像同步同类修复；word_world T1a 放大链在 ESRGAN 侧）
- `src/main_video_optimized.py`（P4 段级解码闭环/失败语义；P5 代际防护）
- `tests/verify_plan_implementation.py`（P4 D1）
- `external/nvenc_common/nal_utils.py`（P3.3 复用）
