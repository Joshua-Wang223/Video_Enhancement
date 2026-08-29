# 优化方案全量执行记录（2026-08-24）

> 执行环境：Windows/CPU 开发机（无 GPU）。全部改动通过 `python -m py_compile` 与
> `tests/test_regression_min.py`（30/30 通过，含真实 ffmpeg 切片/复用/换源重切/
> 重编码合并端到端）。**GPU 相关路径尚未在生产（Linux/GPU）验证**，验证清单见文末。

## Phase 0 止血修复（8 项）

| 标签 | 缺陷 | 修复位置 |
|---|---|---|
| [P0-FIX-MODEL-ARCH] | A1 致命：库内路径 Model 硬编码 S 架构 + processor 不透传 model_name → L/V 模型必崩 | ifrnet main.py `_load_model` 按实例 model_name 动态解析；processor `_get_or_create_video_processor` 透传 `model_name` |
| [P0-FIX-RC] | H1/H2 EOS×2 返回码未查→LA 滞留帧被冻结占位且静默"成功"；drain 非 SUCCESS 错误码无声 break 真实丢帧；DestroyEncoder/event create/sync 返回码漏检 | ifrnet nvenc_sdk：encode_frames_stream send_eos、flush() EOS、_drain_outputs_blocking、flush 内层 drain、close DestroyEncoder、ce_pipeline/encode_frame 的 cuEventCreate/Synchronize 全部 fail-fast 或显式告警 |
| [P0-FIX-HEVC-HANG] | H3 blocking LockBitstream 对 HEVC/AV1 空/未就绪槽永久挂起（deadline 失效） | `_lock_bitstream_blocking` 按 codec 分流：H.264 保持 doNotWait=0（已验证安全），HEVC/AV1 改非阻塞轮询+deadline，绕开 T4 doNotWait=1 segfault 场景 |
| [P0-FIX-MUX-WRITE-TIMEOUT] / [P0-FIX-CLOSE-RC] / [P0-FIX-JOIN-TIMEOUT] | H4 muxer stdin 无超时写（持锁连锁死锁）；close rc≠0 静默；编码线程 flush_and_join 超时仅打印 | FFmpegMuxer 常驻写线程+NVENC_MUX_WRITE_TIMEOUT 超时；有序关闭+_mux_failed 置位供段判败；main.py 段收尾检查 _mux_failed 判败；flush_and_join 超时 raise；FFmpegWriter(ifrnet) close rc!=0 上抛（无在途异常时）|
| [P0-FIX-CODEC-GUID] | H5 AV1 请求在不支持 GPU 上 encoder 静默回退 H264 而 muxer 按 OBU 封装 → 必产废文件 | GUID 未命中改 raise，让四级编码回退接管 |
| [P0-FIX-BATCH-ISOLATION] / [P0-FIX-CONFIG-RESTORE] / [P0-FIX-EXT-PROP] | H6 批量循环无逐文件隔离；keep_audio 仅两路径恢复；merge 强改扩展名致假阴性链 | main_video_optimized 批量 try/except；skip_upscale keep_audio finally；merge_videos_by_codec 新增 actual_output 出参并在扩展名改写时告警+回传，主调用点采用实际路径 |
| [P0-FIX-AUDIO-SRC] / [P0-FAIL-FAST] | H8/H10 ESRGAN 流水线异常被吞仍 return True；audio_src 属性不存在→SDK 路径无声 | esrgan main.py：异常置 _pipeline_error 返回 False、成功横幅条件化；NVENCWriter audio_src 传源视频路径（分段无音轨时 -map 1:a? 安全跳过）|
| [P0-FIX-FPS-EVAL] / [P0-FIX-BARE-EXCEPT] | H12 eval(r_frame_rate) 异常穿透；三处裸 except 吞 Ctrl+C | video_utils Fraction 解析（'N/A'/除零安全）；except Exception 收敛 |

## Phase 1 稳定性加固

| 标签 | 内容 | 位置 |
|---|---|---|
| [P1-FIX-FINGERPRINT] | checkpoint 上游指纹从"排序路径名"升级为 path:size:mtime_ns，内容变更必然失效断点 | 两个 processor process_segments_directly |
| [P1-FIX-ATOMIC] | checkpoint tmp+os.replace 原子写；读加 encoding='utf-8' | 两个 processor _save/_load_checkpoint |
| [P1-FIX-BS1-CIRCUIT] | bs=1 连续 3 轮深度清理仍 OOM → fail-fast（含 mem_get_info 诊断），消除活锁 | ifrnet main.py _safe_infer |
| [P1-FIX-CACHE-ROLLBACK] | get_or_create 先失效缓存再构造，防"已 free 旧对象+旧 key"悬空命中 | ifrnet main.py result_pool/ring_buf |
| [P1-FIX-RING-RESET] | 每段编码级别决策前复位 self._ring_buf，防跨 Level 抢占旧容量 ring | ifrnet main.py _process_segment |
| [P1-FIX-F0-PROTECT] | 首帧读取+f0 编码区 try 保护（原异常泄漏 reader/writer 双 ffmpeg 进程）；GPUMonitor stop 移入 finally | ifrnet main.py |
| [P1-FIX-WATCHDOG] | T2 OOM 恢复期设置 5min 宽限时间戳，writer 空转看门狗宽限期内不计时（防误杀健康段） | ifrnet main._safe_infer + pipeline._writer_loop |
| [P1-FIX-WIN-SELECT] | ESRGAN FFmpegWriter `_write_with_timeout` Windows 分流：常驻写线程+Event.wait 超时（select 不支持管道 fd 致软编链整体不可用） | esrgan ffmpeg_io.py |
| [P1-FIX-SPLIT-FINGERPRINT] / [P1-FIX-SEG-COUNT] | split 复用判定加 (source,size,mtime_ns,segment_duration) 指纹侧车 `.segments_fingerprint.json`；浮点容差取整；实产段数不符显式告警 | video_utils.split_video_by_time + 新增 `_fingerprint_matches` |
| [P1-FIX-AUDIO-FRESHNESS] | smart_extract_audio 同名复用前校验源指纹侧车 `<out>.src.json`，不符重提取 | video_utils |
| [P1-FIX-INPLACE-GUARD] | 批量 input==output 就地覆盖守卫 | main_video_optimized 批量循环 |
| [P1-FIX-AUDIO-CLEAN] | 音频临时文件清理矩阵：修剪后删原音频/失败分支回收/skip 两分支统一走新 helper `_rewrite_audio_and_cleanup`（finally 清 .with_audio.mp4）+ `_cleanup_audio_temp` | main_video_optimized |
| [P1-FIX-LA-ACTUAL] / [P1-FIX-LA-FPS] | LA 补偿判定改为查询两处理器实际 NVENCEncoder `_la_depth`（自动涵盖 CRF=0 清零与 ≥1080p 自动 constqp），阈值换算用真实输出帧率替代硬编码 60fps；移除死条件与 import 探测 | main_video_optimized |
| [P1-FIX-VALIDATE] | CLI falsy 判断（segment_duration/batch_size×3/gfpgan_batch）改 is not None；新增 `_validate_effective_config` 在 CLI 覆盖后校验 segment_duration/factor/crf/batch/LA/rate_mode/gfpgan_weight/tile_size 范围，非法拒绝启动(exit 2) | main_video_optimized |

## Phase 2 性能与一致性

| 标签 | 内容 | 位置 |
|---|---|---|
| [P2.1-TILE-RESTORE] | **tile 真实接入**：新增批级平铺前向 `_sr_tile_forward`（完整复刻官方 pre_pad/mod_pad/tile_pad 反射补边+切块回贴算法，批维保持），eager/compile 路径 tile_size>0 时启用；TRT 静态形状不适用并启动日志明示。说明：未逐帧调用官方 enhance()——那是单帧 numpy API（每帧同步 D2H/H2D 回环+B=1），会摧毁批量推理/异步 D2H/defer_resolve 架构；本实现使 tile 配置真实生效且零架构损伤。顺带修复官方 tile_process 吞错继续跑的隐患 | esrgan pipeline.py + main.py 启动明示 |
| [P2.2-COMPILE-IMPL] | **ESRGAN compile/CUDA Graph 实现**：use_compile(无TRT)→torch.compile(default,dynamic=True)；use_cuda_graph(无TRT/compile)→torch.compile(mode='reduce-overhead')（inductor cudagraphs 即 CUDA Graph）；编译缓存 .torch_compile_cache_sr；摘除 pipeline 中恒假的 cuda_graph_accel 死分支 | esrgan main.py + pipeline.py |
| [P2.2-MUTEX-PARAM] | **IFRNet 互斥裁定前移参数层**：TRT > compile > CUDA Graph 在 processor 库内构造路径与独立 CLI 仲裁处先行裁定并回写 config（后端防御检查保留兜底），"强制启用"旗标语义不再被静默推翻 | src/processors/ifrnet_processor_video_optimized.py 两处 |
| [P2.3-ZERO-COPY] | writer 逐帧冗余 .copy() 消除（reader 帧缓冲不复用，安全）；RING bytearray.extend → b"".join 单次拼接 | ifrnet pipeline.py |
| [P2.3-LA-PINNED-REUSE] | LA D2H pinned 缓冲改为形状键控旋转池（池深=编码队列深+2，FIFO 在途界保证严格无数据竞争） | ifrnet pipeline.py GPU_RAW 路径 |
| [P2.4a] | 热路径 CFUNCTYPE 原型模块级预构造（LockBitstream/Unlock/LockInputBuffer 共 12 处内联替换） | ifrnet nvenc_sdk |
| [P2.4b] | ctx push/pop 收敛为 `_ctx_push/_ctx_pop` 单一实现（3 prologue + 4 epilogue 替换） | ifrnet nvenc_sdk |
| [P2.4d-EMPTY-FRAME] | ce_pipeline 空字节帧与 None 同口径：prev 占位保帧数守恒（原 pass→MP4 帧数缩水） | ifrnet nvenc_sdk 编码线程 |
| [P2.5] | logging 基础设施 `src/utils/logger.py`（幂等 init、UTF-8 文件按 config/logging 落盘）+ 主入口接入（阶段横幅写日志，运行日志路径打印）。external/* print 遥测保留，迁移待 GPU 回归分批进行 | src/utils/logger.py + main_video_optimized |

## Phase 3 架构与可维护性

| 项 | 内容 |
|---|---|
| P3.2 | 新建共享包 `external/nvenc_common/nal_utils.py`（NAL 扫描参考实现：iter_nals/first_vcl_type/has_param_sets/extract_param_sets，h264/hevc/av1）；ifrnet 侧暂不改写（混合编码大文件高风险），由回归测试 D 组以未绑定方法调用锁定两侧行为等价（30 断言全过），后续可安全切换 |
| P3.3 | 删除 `external/realesrgan_video/nvenc_writer.py` 整文件（grep 证实零代码导入，07-28 曾把修复误打其上的地雷）；删除 ifrnet nvenc_sdk 死代码 `_FUNC_TABLE_SIZE`、`_NVENC_VBR_QUALITY_OFFSET`(定义)、`_RotationBitReader` 类；src 历史版本 21 个文件/目录归档至 `archive/src_legacy/`（src 下现存唯一入口+两处理器+utils） |
| P3.4 | 新增 `tests/test_regression_min.py`（无 pytest 依赖，直接运行）：A 指纹侧车 5 断言；B 真 ffmpeg e2e 7 断言（lavfi 合成 -g 40 强制关键帧→切片3段/侧车写入/同源复用/换源重切/mpeg4 重编码 actual_output=.mp4 传播）；C nvenc_sdk 导入契约 5 断言；D NAL 等价性 13 断言。**当前 30/30 通过**。测试过程实测确认了 segment muxer 关键帧对齐特性（单 keyint 源合法产单段，段数核对告警正确触发） |

## 明确延后项（需生产 GPU 环境先行回归）

1. **[P2.4c] SPS/PPS 七处阶梯统一走 `_apply_sps_pps`**：各 ladder 与统一入口存在
   `_sps_pps_injected` muxer 标志等细微语义差，盲改风险高于收益；建议在生产环境
   以 tt7 同款 VCL 过滤诊断脚本建立基线后逐处切换。
2. **[P3.1] 上帝函数拆解**（_open_session ~549 行 / _process_segment ~532 行 /
   ce_pipeline ~411 行 / _process_single ~450 行）：纯结构重构，需 GPU 基准护航；
   本轮已完成其中最高价值的低风险收敛（ctx 模板/原型/NAL 参考实现）。
3. **logging 全量迁移 external/**：print 遥测含运维依赖的 FIX 标注与解析格式，
   分批迁移。

## 生产（Linux/GPU）验证清单

按风险从高到低，每项须记录前后基准（FPS/P50 利用率/峰值显存/帧数守恒/bitstream 解析/空帧统计）：

1. **model_name 全矩阵**：IFRNet_S/L/Vimeo90K 各跑一个视频（P0-1 直接影响加载成败）
2. **NVENC RC×codec×LA**：h264 {constqp,vbr_hq,qvbr}×{LA=0,8,16}；hevc 至少 {vbr_hq,16}（P0-2/P0-3 触及排空与锁路径；hevc 轮询模式为新增行为需重点观察是否仍有挂起/segfault 迹象）
3. **EOS/收尾链路**：短段（<LA 窗口）、帧数恰被 chunk 整除的段、段尾时长核对（P0-2 EOS fail-fast 改变了静默路径）
4. **muxer 写超时**：模拟慢盘（如 NFS/限速盘）确认 NVENC_MUX_WRITE_TIMEOUT 生效且段判败而非挂死（P0-4）
5. **AV1 请求 on 非 Ada GPU**：确认 Level1 显式失败并降级 FFmpegWriter 成功出片（P0-5）
6. **ESRGAN compile/cudagraphs**：--use-compile 与 --use-cuda-graph 各一次冷/热启动；adaptive batch 变长下 reduce-overhead 是否频繁重捕（预期首启慢、稳态收益）（P2.2a）
7. **tile**：tile_size=512 于 4K 输入 eager 路径，检查拼缝/边缘与吞吐对比 batch-only（P2.1）
8. **OOM 阶梯+熔断**：受控占用显存触发 bs=1 三轮熔断报错（P1-BS1）
9. **watchdog 宽限**：OOM 恢复期间 writer 不再误杀（P1-WATCHDOG）
10. **checkpoint 指纹**：上游换参重跑确认断点整体作废而非混流（P1-FINGERPRINT）
11. **音轨**：SDK Level1 输出有声（有音轨源直跑场景）；≥1080p 自动 constqp 后不再误剪音频头（P0-7/P1-LA-ACTUAL）
12. **ctx 泄漏观察**：多段运行 nvidia-smi 句柄/驱动内存平稳（P0-2 DestroyEncoder rc 日志）

## 改动文件清单

- src/main_video_optimized.py（P0-6/P1-7cde/P1-8/P2.5 接入）
- src/utils/video_utils.py（P0-8/P1-7ab）
- src/utils/logger.py（新增）
- src/processors/ifrnet_processor_video_optimized.py（P0-1/P1-1/P2-2b）
- src/processors/realesrgan_processor_video_optimized.py（P1-1）
- external/ifrnet_video/main.py（P0-1/P1-2/3/4/5 部分/P1-watchdog 发送端/P1-f0/P1-monitor）
- external/ifrnet_video/pipeline.py（P1-5 watchdog 接收端/P2-3）
- external/ifrnet_video/nvenc_sdk.py（P0-2/3/4/5/P2-4ad/删除死代码）
- external/ifrnet_video/ffmpeg_io.py（P0-4 close rc）
- external/realesrgan_video/main.py（P0-7/P2-1 明示/P2-2a）
- external/realesrgan_video/pipeline.py（P2-1/P2-2a 摘除死分支）
- external/realesrgan_video/ffmpeg_io.py（P1-6）
- external/nvenc_common/{__init__,nal_utils}.py（新增）
- tests/test_regression_min.py（新增）
- 已删除：external/realesrgan_video/nvenc_writer.py；已归档 archive/src_legacy/（21 项）

## 后验证脚本三合一合并（2026-08-24 追加）

对 `test_regression_min.py` / `verify_plan_implementation.py` / `verify_post_run.py`
三个后验证脚本做覆盖对比后，整合为单一最终脚本 **`tests/verify_plan_implementation.py`（v2）**：

| 来源 | 处置 |
|---|---|
| verify_plan_implementation.py | 载体重写：修正 9 处按"假想实现"校准的失准断言（改锚定实际 `[P*-FIX-*]` 标签+结构特征），吸收 regression_min 全部行为断言为 C 阶段 |
| test_regression_min.py | 全量并入 C-行为阶段（BEH-A~F 组）；原文件降级为兼容别名（--behavior-only 转发） |
| verify_post_run.py | 仅吸收 3 个有效点（NVENC 无 torch 探测=R7、NVML 环境提示=R8、GPU 详情入报告）；其余因 ffprobe 参数非法（`-count_entries/nb_entries` 不存在）、配置键路径错位（读顶层而 schema 在 models.* 下）、tile/model 断言与修复后状态矛盾而废弃；文件归档至 `archive/tests_legacy/` |

**对比过程中发现并修复的真实问题**：
1. video_utils `get_frame_rate` 内第 4 处裸 except（此前只修了 3 处）→ 已收敛；
2. realesrgan_utils `get_video_meta_info` 内裸 except → 已收敛；
3. 合并脚本自身两处断言缺陷（同名 close 方法误匹配、方法行数统计漏缩进）→ 修正断言。

**最终运行基线**：全量 72 项 = 通过 66 / 失败 0 / 警告 3（R5 无 torch、R7 无 NVIDIA 卡、R8 NVML 提示——开发机预期）/ 跳过 3（P2-4c 与 P3-1 计划内延后、RT-0 未提供 --output），EXIT=0。
兼容入口 `python tests/test_regression_min.py`（--behavior-only）：34/34 通过。

## 生产首轮验证反馈与修复（2026-08-24 追加 II）

生产环境（Linux/T4）运行 v2 验证脚本暴露两个仅在真实环境显现的问题：

1. **P0-9 误报（裸 except ×13）**：根因是生产仓库尚未同步 `archive/src_legacy/`
   归档，历史版本 `src/main_v2.py` 等仍在 src/ 下被 rglob 命中。修复：扫描改为
   **活跃文件白名单**（BARE_EXCEPT_ACTIVE_SRC 8 文件 + external 三目录），
   判定与归档同步进度解耦（verify_plan_implementation.py）。

2. **SMOKE rc=-11（SIGSEGV，重要）**：全流程冒烟在 IFRNet LA 流式排空早期
   （frame_idx=1）LockBitstream 返回瞬态 INVALID_PARAM(code=8)——旧代码静默
   break 吸收了它；本方案首版 fail-fast 直接上抛 → 段中止 + 异常拆卸期 SIGSEGV。
   修复（[P0-FIX-RC-RETRY]，nvenc_sdk._drain_outputs_blocking）：非 SUCCESS
   先有界重试（≤3 次、1ms、不消耗槽轮转预算），持续失败才上抛判败。
   ⚠️ 最终语义后续升级为容忍+遥测，见追加 V。

3. 附带修正：P3-1 目标 `_open_session` 已内联于 NVENCEncoder.__init__（生产报告
   显示 0 行），锚点改为 __init__。

## 验证输出卫生与统计修正（2026-08-24 追加 III）

针对生产报告 verification_1.txt 中其余报警的核查结论与处理：

1. `⚠️[merge] out.mkv→out.mp4` 与两行 `❌ 生效配置未通过范围校验` —— 定性为
   **负路径行为测试的预期证据**（BEH-B5 故意触发重编码改写验证 P0-7 出参契约；
   BEH-F2/F3 故意喂非法值验证 P1-8 校验器拒绝）。非缺陷，但直通控制台会污染
   人工扫查 → 已改为 **capture 式执行**（_capture_stdout）：证据写入断言
   名称/详情字段，控制台零 ❌/⚠️ 残留。
2. `P3-1 _process_single=992行` —— 统计缺陷而非真实体积：func_body 边界前瞻
   缺顶层 `\ndef/@` 分支，对文件尾前无类定义的顶层函数会吞到 EOF（生产与本地
   同为 992 印证）。修正边界后真实值 **448 行**；_process_segment=565 不变。

## verification_2 逐项定性与处置（2026-08-24 追加 IV）

| 现象 | 定性 | 处置 |
|---|---|---|
| R8 NVML WARN | 脚本噪声（纯提示） | 降级 SKIP：入口运行时自动 setdefault，仅外部直调需手动 |
| "为何每次分割3次" | **测试设计使然，非重复执行**：BEH-B 三次调用 = B1 首切(reuse=False)/B3 同源复用命中(♻️)/B4 换源指纹不符重切；中间一次并未重割 | split 调用全部 capture 式执行，动作摘要写入断言详情，控制台零泄漏 |
| SMOKE-2 rc=-11 code=8 | 初判"漏同步"；带全现场复现推翻，最终定性见追加 V | dev 版加重试并增强 persistent 现场诊断 |
| SMOKE-2 upscale_then_interpolate rc=-124 超时3600s | **证据不足待隔离复现**：无阶段日志可判定卡点 | 下轮先单独跑 --smoke-mode upscale_only 并保留完整 stdout/stderr 尾部 |
| P0-9 裸 except×13 | 脚本问题已修复但生产仍跑旧版脚本 + src 归档未同步 | 同步最新 tests/verify_plan_implementation.py 即消；扫描已改活跃白名单 |

**本轮新增同步文件清单（生产必更新）**：
1. external/ifrnet_video/nvenc_sdk.py
2. tests/verify_plan_implementation.py

**本地门禁**：alias 34/34 EXIT=0、全量 EXIT=0、控制台 ❌/⚠️/split 动作打印=0。

## code=8 根因定性与最终语义（2026-08-24 追加 V · 重要）

生产带全现场的复现（persistent after 3 retries, slot=0, frame_idx=1,
out_slot_idx=0, ts_base=0, pending_head_fi=0, h264/vbr_hq/la=8）确认：
这不是漏同步，而是该环境的确定性常态。推理链：

1. EncodePicture 已成功（否则更早 raise）→ 帧已进 LA 管线；
2. la=8 下首次 drain 应返回 NEED_MORE_INPUT(17)，实际稳定返回 INVALID_PARAM(8)
   —— T4 此驱动在流式首排空期的固有行为；
3. 历史 LA=8 生产验证全部通过：旧实现静默 break 逐帧吸收，帧由段末 EOS 全量
   排空回收 + 守恒校验兜底——"逐帧 drain 失败但段末守恒"是既有隐性模式；
4. v1 的 fail-fast 打破此平衡：段中止 + 异常拆卸期 SIGSEGV。

**最终语义（三轮演进：静默 → fail-fast → 容忍+遥测）**
`[P0-FIX-RC-TOLERANT]`（nvenc_sdk._drain_outputs_blocking）：
- 非 SUCCESS：记 `_diag_lock_err_<code>` 计数 + ≤5/%50 节流打印 + break；
- EOS×2 与 flush 排空保持 fail-fast（那里丢的才是真尾帧）；
- 段级守恒审计（main._process_segment 期望帧数比对）兜底，短差判段失败。
验证脚本 P0-2 断言同步改为分级处置口径。

**生产可选 bisect（刻画边界）**：
- `--rate-mode-ifrnet constqp`：历史验证最充分组合；
- `--lookahead-depth 0`：绕开 stream-drain 路径走 ce_pipeline。
二者通过而默认组合守恒短差 → 升级驱动级专项（回传守恒数字与 _diag_lock_err_8 总数）。

## TRT 引擎 tag 的 B/H/W 控制链路（2026-08-24 追加 VI）

现象答疑：为何引擎名 `IFRNet_S_Vimeo90K_B24_H1440_W2560_fp16_sm75_teslat4.trt`
而非脚本 1253 行的 12？且分辨率远高于 ESRGAN 的 H360_W640？

1. **B 维度链路**：config `models.ifrnet.batch_size`（默认 24）→ processor 透传 →
   backend `self.batch_size` → main.py `sh=(self.batch_size,3,H,W)` → tag `B{B}`。
   verify 脚本 1253 行是 BEH-F 组 FakeConfig 内存夹具，仅喂校验器逻辑断言，
   与 TRT 构建无关。CLI 可用 `--batch-size-ifrnet` 覆盖。
2. **H/W 取自"该阶段首个输入"**（`_probe_video` → MODEL_STRIDE=32 向上对齐）：
   - interpolate 先行：IFRNet 吃原始 360×640（插帧不改分辨率）；
   - upscale 先行：IFRNet 吃 ESRGAN ×4 输出 = 1440×2560（恰被 32 整除）。
   观察到的 {IFRNet:1440p, ESRGAN:360p} 组合证明该对引擎铸造于
   upscale_then_interpolate 那次超时运行。ESRGAN 无论顺序输入均为 360×640
   （它自己才是改分辨率的阶段），故其 tag 恒为 H360_W640。
3. **冒烟可控性**：已给 verify 脚本加 `--ifrnet-batch/--esrgan-batch` 直通
   （缺省不传沿用 config 默认），显式固定 B 维度保证缓存命中可预测。
4. ⚠️ TRT 每进程只建一次（`_trt_built` 门控）、跨进程按 tag 缓存；同进程混分辨率
   分段无静态形状保护（已知限制），冒烟请固定顺序+batch。

## 维护警告：memory 写入编码规范（2026-08-24 事故）

**禁止通过 PowerShell 管道/heredoc（`@'...'@ | python -` 等）向 memory 写中文**：
PS 向本地管道编码走控制台代码页，中文被有损替换为字面 `?`(0x3F) 后落盘，
不可逆（本次第 108–208 行共 1549 处损毁的根因）。必须使用支持 UTF-8 的直写
工具（opencode write/edit 等），写后以字节扫描抽检 0x3F 异常。

> 本文件于 2026-08-24 由会话 ses_fcd19c… 诊断为追加段落乱码后，依据原始上下文
> 全文重建（第 108 行起的追加 I–V 为忠实复原，VI 与本警告为新增）。

## 全库乱码审计与修复（2026-08-24 追加 VII）

对本次改动涉及的全部代码做字节级扫描（ASCII 0x3F × CJK 混排 / 连续 `??` 运行）：

| 位置 | 定性 | 处置 |
|---|---|---|
| ifrnet nvenc_sdk.py:373-374 | **真损坏**：P2.4a 原型块注释经管道写入被 ? 替换 | 按上下文重写为原意中文 |
| AGENTS.md:12 | **真损坏**：行级替换脚本经管道写入 | 重写完整条目 |
| nvenc_sdk.py `_ctx_push/_ctx_pop` docstring | 曾写入 `\uff0c` 转义字面量 | 此前已修，复扫确认无残留 |
| nvenc_sdk.py:3128 `"?"` / main:1273 `-map 0:a?` 等 | 合法（代码字面量/ffmpeg 可选流语法） | 保留 |
| 历史"疑似乱码"（av1 注释、═ 分隔线等） | 字节层合法 UTF-8，仅 GBK 控制台显示问题 | 无需处理 |

**根因补充**：PS5.1 向本地管道（`| python -`）输出默认 ASCII 编码——凡中文字符串
经此通道必损；PowerShell 原生执行与 opencode Edit/Write 工具不受影响。
**门禁**：nvenc_sdk 编译通过；alias/full 均 EXIT=0。
