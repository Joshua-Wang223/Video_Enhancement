"""
IFRNet 视频插帧处理脚本 —— 四级回退统一编码版 v6.4.5.1（单卡版）
============================================================
基于 IFRNet（Intermediate Flow-based Recursive Network）的视频帧插值脚本，
面向单 GPU 生产环境的极致性能实现。
【2026-08-14 新增修复 — LA 辅助块账记（test11 复现 tt6 根因）】
  [FIX-AUX-NO-CLEAR] 恢复 _nal_first_vcl_type() VCL/辅助块分类：NVENC LA 冷
                    会话预热期会把 SPS/PPS/AUD 作为独立无 VCL 辅助块 drain，
                    旧 FIFO 记账将其当普通帧 pop 队首 → pending 被清空而真实
                    VCL 数据仍滞留 → 背压失效 → 输入覆盖 → 每 9 帧迟一窗的
                    frame_num 回退（test11: 549 次、解码 1411/4960）。
                    修复：辅助块仅缓存参数集，不占 fi、不进 pairs、不清
                    pending；主 drain / EOS drain / _ensure_slot_free 统一生效。
  [FIX-SLOT-DRAIN-TARGET] _ensure_slot_free 轮转探测无法触达目标槽时直探
                    目标槽；guard 超限以空帧占位（prev 填充）消费 pending，
                    绝不带 pending 复用（宁可写占位也不覆盖未取回码流）。
  [FIX-DRAIN-COUNTER-DRIFT] _drain_outputs_blocking 在 bitstream 指针为空时
                    不再推进 _output_slot_idx（消除计数器超前漂移）。
  [FIX-DRAIN-ORDER-DEFENSE] drain 三元组携带 outputTimeStamp@40（VCL 块
                    双射已验证），与 FIFO 队首 gfi 比对产生 _diag_phase_shift
                    诊断；新增 _diag_aux_block / _diag_slot_drain_fallback。
  同修复已回植 v6.4.4.1 / v6.4.3.1（变量名 _batch_slot_pending）。
【2026-08-13 新增修复 — LA 排空顺序防御（test8 复现）】
  [FIX-EOS-LEFTOVER-G-NAMEERROR] 修复 EOS 残留占位路径 _g NameError（确定性崩溃）：
                    生成器体引用未绑定变量 _g → 改为 _ent[0] - _strm_ts_base。
  [FIX-SLOT-DRAIN-TARGET] _ensure_slot_free 改为探测目标槽自身（per-slot blocking
                    LockBitstream），消除单槽探测局限导致的带 pending 槽位复用
                    覆盖丢帧；guard 超限时以空帧占位（prev 填充）推进 FIFO，
                    绝不带 pending 复用。
  [FIX-DRAIN-ORDER-DEFENSE] _apply_drained_entries 增加 deque 空（_diag_slot_mismatch）
                    与队首 gfi 回退（_diag_gfi_regress）检测计数，错配不再静默；
                    _last_drained_gfi 段内单调基准，_stream_begin 每段重置。
  [FIX-EMPTY-PREV-FILL] 空帧占位 b"" → _prev_stream_h264 前一帧码流兜底（编码器级
                    prev 跟踪，仅用于 EOS 残留/槽位排空超限的真缺失帧，绝不用于
                    LA 正常缓冲帧）。
  [FIX-ENC-EXC-CONTEXT] 编码线程异常包装 (frame_idx, pending_slots, pending_count)
                    诊断上下文后 re-raise，便于定位 LA 路径崩溃。

【新增修复（基于 FIX-ASYNC-COPY）】
  [FIX-FLUSH-GRANULARITY]  修复 FIX-ASYNC-COPY 之后仍残留的周期性 GPU 利用率骤降：
                     · 根因（与 FIX-ASYNC-COPY 是两个独立问题）：
                       _NVENCEncodeThread 原设计"攒够 chunk_frames(默认
                       768=encode_queue_depth×batch_frames)才一次性调用
                       encode_frames_stream()编码一整块"。即使 FIX-ASYNC-COPY
                       消除了 cuMemcpy2D 走 legacy stream 造成的跨 stream
                       隐式全局同步，这个"攒 768 帧再一次性编码"的脉冲本身
                       仍然是编码线程独占、持续 1-2s+ 的一段连续同步 CPU/
                       NVENC 调用（LockInputBuffer/Copy/UnlockInputBuffer/
                       EncodePicture/LockBitstream 逐帧循环）。期间编码线程
                       输入队列 self._q(maxsize=encode_queue_depth)迅速被
                       Writer 写满 → Writer 阻塞在 submit() → Writer 无法
                       消费 result_queue → result_queue 触顶后 T2 推理线程
                       被阻塞产帧 → GPU 计算利用率骤降。这是**队列反压**，
                       和 CUDA stream 选择无关，是残留周期性掉底的主因。
                     · 这也解释了"分段开头平稳、之后才出现周期性凹陷"：
                       段首 _acc_nv12 从空开始累积，尚未触发第一次 flush，
                       队列有充足余量吸收 Writer 产出；一旦跨过第一个
                       chunk 边界、发生第一次同步脉冲，才开始出现反压。
                     · 修复：新增 flush_chunk_frames 参数（默认 128），把
                       "触发编码"的粒度从安全上限 768 帧大幅调小，让 NVENC
                       工作摊薄成许多次小脉冲而非少数几次大脉冲；
                       _la_chunk_safe(768) 仍保留作为硬上限（三者取最小），
                       正确性不变（encode_frames_stream 本就为跨 chunk
                       连续调用设计）。
                     · 分段之间空闲时间较长（10-20s/段）为另一独立现象：
                       tqdm 插帧进度 100% 后，仍需同步完成"末块编码 + EOS +
                       LA(8)窗口全排空 + muxer/ffmpeg 子进程收尾"，且当前
                       架构下这段收尾与下一分段的 FFmpeg 读取启动完全串行、
                       无重叠。本次 flush_chunk_frames 调小后，末块残留帧数
                       上限从 768 降到 128，可缩短收尾尾部，但 LA 窗口排空
                       与下一段读取启动的串行结构本身未变——若要进一步压缩，
                       需要"分段间预取重叠"这类更大的架构改动，风险与收益
                       需要单独评估，本次未纳入。

  [FIX-ASYNC-COPY]  修复 NVENC 编码线程周期性阻塞插帧推理导致的 GPU 利用率
                     规律性骤降：
                     · 根因：NVENC 输入拷贝使用同步 cuMemcpy2D_v2（无 stream
                       参数，走 legacy default/null stream）。该 stream 在
                       同一 CUDA context 下具有隐式全局同步语义——每次拷贝
                       都会强制等待推理用的 stream_h2d/stream_compute/
                       stream_d2h 排空，反之亦然。
                     · LA>0（VBR_HQ/QVBR + lookahead）模式下 _NVENCEncodeThread
                       攒够 chunk_frames(=encode_queue_depth×batch_frames，
                       默认 768 帧)才一次性调用 encode_frames_stream()，
                       其内部对整块帧连续、逐帧执行该同步拷贝，产生持续
                       1-2s+ 的连续隐式全局同步脉冲，表现为 nvtop/nvitop
                       中周期性(约 chunk_frames/当前fps 一个周期)出现的
                       GPU 利用率骤降，插帧推理与 NVENC 编码并非真正并行。
                     · 修复：新增专用非默认 CUDA stream(self._stream_encode,
                       CU_STREAM_NON_BLOCKING) + cuMemcpy2DAsync_v2，拷贝
                       入队后仅 cuStreamSynchronize 这一个私有 stream，
                       不再对其他 stream 施加隐式全局屏障。三处 NVENC 输入
                       拷贝调用点（encode_frames_stream / 
                       encode_frames_batch_ce_pipeline / encode_frame）统一
                       改为调用新增的 _copy_into_input_buffer() 辅助方法；
                       若专用 stream 创建失败自动回退到原同步拷贝，不影响
                       功能正确性。

【v6.4.5 修复（基于 v6.4.4）】
  [PHASE4-v645] 恢复多 slot 异步流水线 encode_frames_batch()：
               · 根因修复后（v6.4.4 隔帧花屏 → Infer 线程缺少 synchronize），
                 恢复 4-slot 轮转 NVENC 编码，Lock/Copy 与 Encode 跨 slot 并行。
               · pipeline_depth 自动校准为 >= LA+1 (SDK 硬件安全要求)
               · 与 _NVENCEncodeThread 异步编码线程配合，进一步提升吞吐。

  [PHASE4-CE-PIPELINE] encode_frames_batch_ce_pipeline() — per-frame CE 异步流水线：
               · EncodePicture 附带 per-frame CUDA completion event 异步提交，
                 LockBitstream 延迟到 slot 下次轮转时 harvest（pipe_depth 帧后），
                 CE 已触发→立即拿到数据，消除同步 EncodePicture 阻塞。
               · Phase 1 (Harvest) / Phase 2 (Submit) / Phase 3 (Drain) 三阶段。
               · GPU 验证 (T4, 720×576, VBR_HQ, pipe=4, LA=0):
                 523 FPS vs sync-batch 375 FPS (+39.5%)，0% 空帧。
               · _NVENCEncodeThread._loop() 已切换默认使用此路径。
               · LA=0 保证与 pipe=4 兼容，消除 lookahead 路由错位导致的空帧。

【v6.4.4 新增修复（基于 v6.4.3）】
  [FIX-ENC-CTX]  _NVENCEncodeThread CUDA context 缺失修复（cuMemcpy2D code=201 崩溃）：
               · 根因：_NVENCEncodeThread._loop() 是新建 daemon 线程，启动时
                 CUDA context stack 为空。encode_frames_batch 内 cuCtxPushCurrent
                 若返回非零 CUDA 错误码（如 CUDA_ERROR_INVALID_CONTEXT=201）不会
                 触发 Python 异常，被 except Exception: pass 静默吞掉；但
                 _need_pop = True 仍被错误设置，实际 context 从未激活，导致后续
                 cuMemcpy2D_v2 以同一错误码 201 失败，写线程在第二次 submit() 时
                 重新抛出该异常（frame≈25 处崩溃）。
               · 修复 A：在 _loop() 线程启动阶段调用 cuCtxSetCurrent(_primary)，
                 一次性将 primary context 绑定到编码线程；后续 encode_frames_batch
                 的 push/pop 循环在已激活的 context 上正常工作。
               · 修复 B：encode_frames_batch / encode_frame 两处将 _need_pop = True
                 改为 _need_pop = (r_push == 0)，确保 cuCtxPushCurrent 失败时不会
                 错误触发 cuCtxPopCurrent，避免意外弹出 SetCurrent 设置的 context。

【v6.4.3 新增修复（基于 v6.3.5）】
  [FIX-T3-V643] 四级回退统一编码体系：
               · NVENCEncoder 跨平台改造: CDLL + 运行时 API 版本探测 + Linux CUDA ctx
               · 新增 PinnedRingBuffer: N 槽预分配 pinned memory 环形缓冲（Level 2/3）
               · FFmpegWriter.write_direct(): memoryview 零拷贝写 pipe
               · Level 1 (最优):  NVENC SDK GPU 直通 → FFmpegMuxer (GPU→NV12→H.264 ES)
               · Level 2 (次优):  Pinned Ring Buffer + FFmpeg NVENC 硬件编码
               · Level 3 (兜底):  Pinned Ring Buffer + FFmpeg 软编码
               · Level 4 (最后):  标准 PinnedResultPool + FFmpegWriter (v6.3.5 原路径)
               · _infer_loop + _writer_loop: 三级数据协议统一 (5-tuple / RING / PinnedResultItem)
               · 新增 _rgb_to_nv12_gpu: GPU 侧 RGB→YUV420 颜色空间转换
               · FFmpegWriter → FFmpegMuxer: 仅做比特流封装（-c:v copy），不再编码

【v6.4.3 Phase 1-3 修复（GPU-STAY 性能退化）】
  [FIX-GPU-STAY] Level 1 NVENC SDK 直通性能修复：
               · Phase 1: maxNumRefFramesInDPB 16→4（减少 ME 开销 ~50%）
               · Phase 2: 新增 _rgb_to_nv12_gpu_batch()，N 帧批量转换消除 kernel launch 开销
               · Phase 3: NVENC 编码从 Infer 线程移到 Writer 线程
                  — 旧: T2(推理 + 串行NVENC编码) → result_queue → T3(写H.264)
                  — 新: T2(推理) → result_queue(GPU tensor) → T3(NVENC编码 + 写H.264)
                  — 推理与编码并行，GPU 不再被同步 EncodePicture 阻塞

  [PHASE4] NVENC 同步批编码流水线（completionEvent 不可靠，保持同步）：
               · 多 slot (4) × (input buffer + bitstream buffer) — 分散 buffer 争用
               · encode_frames_batch(): 逐帧同步编码（blocking EncodePicture + LockBitstream retry）
               · encode_frame(): 单槽 completionEvent + cuEventSynchronize（CE 对 pipeline=1 可靠）
               · PHASE4 异步 (_submit_to_slot/_harvest_slot) 经 GPU 验证: event 复用导致花屏 → 已移除
               · flush 排空全部 4 slot（修复 -7 帧丢失）

【v6.3.5 新增修复（基于 v6.3.4）】
  [FIX-POOL-LEAK]     PinnedResultPool 跨段泄漏修复（后 3 段速度减半问题）：
                   · 根因：段结束后 PinnedResultPool 持有的 pinned 内存未显式释放，
                     Python GC 不会立即回收 cudaHostAlloc 分配的锁页内存，导致各
                     段 Pool 叠加累积（944MB → 1557MB → 1935MB → 1982MB+），
                     DMA 带宽被竞争，D2H 传输变慢，result_queue 长期满载，
                     GPU P50 利用率从 88% 跌至 8%，吞吐量减半（83fps → 41fps）。
                   · 修复：新增 PinnedResultPool.free() 方法，显式 del 全部 pinned
                     buffer；在 _infer_loop finally 块中调用，确保每段结束后立即
                     解除锁页内存占用；同时触发 gc.collect() 加速 Python 对象回收。

  [FIX-MAXRQ-DYNAMIC] result_queue 上限改为三轴动态计算（替代静态 _PINNED_POOL_MAX_MB）：
                   · 根因：_PINNED_POOL_MAX_MB 为 GPU-tier 静态常量（T4=2048MB），
                     无法反映实际 RAM 余量、分辨率变化、T3/T2 速度比等动态因素；
                     换机（RAM ≠ 32GB）或换分辨率时上限或过保守或过激进。
                   · 新增 _compute_max_result_queue(slot_mb, mem_avail_gb, T2_ms, T3_ms)
                     三轴联合约束：
                     轴1 RAM 上限  = mem_avail × 6% / slot_size（主要约束）
                     轴2 T3/T2 下限= T3_ms/T2_ms × 0.22（最小解耦需求，libx264 流式系数）
                     轴3 绝对上限  = 48（防估算失控保险）
                     结果 = max(floor_by_t3, min(cap_by_ram, 48))
                   · 替换 _auto_queue_depths / get_queue_suggestions /
                     ADAPTIVE-QUEUE 三处 _PINNED_POOL_MAX_MB 硬编码引用。
                   · _PINNED_POOL_MAX_MB 保留为模块级常量（PinnedPool 构建阶段
                     的参考值），不再用于运行时队列约束。

【v6.3.4 新增修复（基于 v6.3.3）】
  [FIX-BATCHCAP]     跨段 batch_size 误降修复（Segment 2+ bs 32→7 问题）：
                   · 根因：free_bytes = torch.cuda.mem_get_info()[0] 仅返回 OS 层面
                     真正空闲 VRAM，忽略 PyTorch allocator 已 reserved 但未 allocated
                     的可复用缓存（段结束后 TRT engine 保留 ~14 GB reserved pool）。
                   · 修复：effective_free = free_bytes + (reserved − allocated)，
                     使跨段 batch_size 估算与段内实际可分配量一致。
                   · 同时修复 _estimate_safe_batch_size() 中同一问题（OOM 恢复路径）。

  [FIX-NVENC-UNIFIED] NVENC 双检测路径统一：
                   · 根因：AUTO-TUNE 用静态 GPU 型号表（_HWProfile.has_nvenc=True），
                     HardwareCapability.best_encoder() 用 ffmpeg 实际 probe，两者结
                     果在 Docker 环境下不一致（probe 失败 → has_nvenc=False → 回退
                     libx264），导致 T3 实测 8fps、GPU 空闲 86% 的锯齿形瓶颈。
                   · 修复：best_encoder() 新增可选参数 hw_profile；当 hw_profile 提
                     供时优先信任静态表（GPU 型号已知）；仅在无 profile 时回退 probe。
                   · _run_segment() 在分辨率限制检查后缓存 hw_profile 并传给
                     best_encoder()，保证 NVENC 检测与 AUTO-TUNE 一致。

  [FIX-T2-TRT-CALIB] TRT 路径 T2 冷启动估算修正（8× 高估→精确）：
                   · 根因：_T2_FIXED_MS = 240ms 为 torch.compile/eager JIT overhead，
                     TRT 路径实测固定 overhead 仅 2-5ms，高估导致 Segment 1 初始
                     result_queue 过深（1150MB pinned），浪费锁页内存。
                   · 新增常量 _T2_FIXED_MS_TRT = 5.0ms（TRT 专用）。
                   · _auto_queue_depths() 和 pipeline.run() 中估算均依 infer_backend
                     分支选择对应固定 overhead 常量。
                   · 修复后 Segment 1 初始 pool 估算从 ~1150MB 降至 ~150MB。

  [FIX-POOL-AUTOSCALE] PinnedPool 上限依 GPU 型号自动缩放（替代硬编码 1024MB）：
                   · 根因：1024MB 对 T4（bs=32 时单 slot ~110MB × 10=1100MB 即超限）
                     过于保守，对 A100/H100（余量充裕）过于宽松。
                   · 新增 _pool_limit_mb_for_profile(profile) 函数，按 gpu_tier 分 6
                     档：GTX 1080=1024  T4/RTX2080=2048  RTX3090/4070=3072
                     A10/L40S/RTX4080+=4096  A100/A800=6144  H100/H800=8192（MiB）。
                   · 兼顾系统可用 RAM：上限不超过 MemAvailable × 12%（最低 1024MB）。
                   · get_queue_suggestions() 新增 hw_profile 参数接收动态上限；
                     _auto_queue_depths() 直接用 profile 计算上限。

【v6.3.3 新增修复（基于 v6.3.2）】
  [FIX-RETUNE-POSTRUN]  AUTO-TUNE-RETUNE 计算时机改为段完成后：
                   · 原实现在 _infer_loop 内 timing[3:8] 共 5 个 batch 做中位数，
                     约占全段 1% 的数据，受流水线启动状态影响较大。
                   · 新实现在 run() 中 _infer_th.join() 后，用 timing[3:] 全段
                     稳定 batch 的中位数（通常 100+ 样本），精度显著提升。
                   · T2-CACHE 文件更新策略不变（早期写入保留，段完成后再精校）。
                   · 与 GPU-MONITOR 统一在段完成后计算，日志顺序更直观。

  [FIX-MEMCAP-LOG]      PinnedPool 内存上限截断时输出显式 log：
                   · 原实现在 ADAPTIVE-QUEUE 合并建议时若 PinnedPool 动态上限
                     将建议值静默截断，用户无法得知"GPU-MONITOR 建议 23 但实际用 15"
                     的原因。
                   · 新增截断时打印：
                     [ADAPTIVE-QUEUE] PinnedPool 内存上限截断: result_queue 19 → 15
                      (slot=58.9 MB × 17 ≈ 动态上限 MB)

  [FIX-RETUNE-DISPLAY]  ADAPTIVE-QUEUE 综合建议打印推导链路：
                   · 原实现只打印最终值，无法追溯 GPU-MONITOR / RETUNE 各自贡献。
                   · 新增来源注释：
                     [ADAPTIVE-QUEUE] 下次将使用 pair_queue=8 result_queue=15
                       (GPU-MONITOR=23 RETUNE=15 → avg=19)

  [FIX-SLICE-THREAD] FFmpegWriter 软编码并行升级：自动探测 CPU 逻辑核心数、物理核心
                   数和系统可用内存，为 libx264/libx265 自动注入最优线程和分片参数。
                   · libx264: -threads N + -x264-params threads=N:slices=S，启用
                     intra-frame slice-based threading（单帧分 S 片并行编码），在
                     pipe 流式输入场景中比 frame-parallel 延迟更低、吞吐更高。
                   · libx265: -x265-params pools=N:frame-threads=F，替换旧
                     pools=none（完全禁用线程池）为正确的多线程配置。
                   · 新增 _detect_encode_parallelism() 函数：读取 /proc/cpuinfo
                     获取物理核心数，读取 /proc/meminfo 获取可用内存，综合计算
                     encode_threads / slices / ffmpeg_threads 三项参数。
                   · FFmpegWriter 新增 n_threads 参数（None=自动探测），兼容旧接口。

  [FIX-NVENC-PIPE]   FFmpegWriter NVENC pipe 模式优化：针对 pipe 流式输入场景补全四项
                   NVENC 硬件编码参数，显著改善编码吞吐与码率控制质量。
                   背景：NVENC 是 GPU 固定功能硬件单元，完全不受 FFmpeg CPU 线程数
                   控制（-threads 仅作用于 demux/filter graph），但以下 NVENC 自身
                   参数在 pipe 场景下对吞吐和质量有显著影响：
                   · -bf 0         禁用 B 帧，消除 NVENC 流水线多帧缓冲延迟
                                   （B 帧需双向参考，持有前后帧缓冲后再输出），
                                   两种模式（无损/VBR）均启用。
                   · -surfaces 32  扩大 NVENC 内部帧缓冲（默认 8 → 32），防止 pipe
                                   输入速率不均时编码器因缓冲不足频繁暂停（饥饿停顿）。
                                   两种模式均启用，值由 _NVENC_SURFACES_PIPE 常量控制。
                   · -delay 0      零延迟输出模式（仅 crf=0 无损/QP=0 路径）：与 -bf 0
                                   协同，使每帧编完即输出，最小化 pipe 端到端延迟。
                                   与 -rc-lookahead 互斥，故仅在无需前瞻的 QP 模式启用。
                   · -rc-lookahead 16  前向帧预看（仅 crf>0 VBR 模式）：编码器向前看
                                   16 帧进行码率分配，在场景切换和高运动区域有效降低
                                   码率浪费、提升 PSNR/SSIM。因需要前瞻缓冲与 -delay 0
                                   互斥，值由 _NVENC_LOOKAHEAD_VBR 常量控制。
                   新增 _NVENC_SURFACES_PIPE / _NVENC_LOOKAHEAD_VBR 模块级常量，
                   便于调优时统一修改。同步补全 NVENC 路径的 [FFmpegWriter] 参数摘要日志。

【v6.3.2 新增修复（基于 v6.3.1）】
  [FIX-CRF0-CALIB]  _software_encode_fps 无损编码校准因子修复：crf=0（lossless）时，
                   x264 实际吞吐远低于理论模型（实测约为估算的 1/18）。新增常量
                   _CRF0_X264_CALIB_FACTOR = 0.055，使 T3 静态估算更贴近实测，
                   从而改善初始 result_queue 深度。

  [FIX-T3-FPS]     T3 写入线程实测 fps 采样：_writer_loop 新增起止时间戳，
                   段结束后计算 _t3_fps_measured；通过 _next_t3_fps_measured 跨段
                   传递，供下一段 _auto_queue_depths 用实测 T3 速度代替静态估算。

  [FIX-T3-REPORT]  T3-bottleneck 诊断报告增强：[ADAPTIVE-QUEUE] T3-bottleneck 分支
                   新增实测 T3 fps、理论估算 fps 及偏差倍数显示；NVENC 可用时显示
                   预期加速比和 Docker 设备映射提示，否则建议降低 preset/crf 参数。

  [FIX-T3-DETECT]  GPU-MONITOR 误判修复：新增 _is_t3_bottleneck() 静态检测器；
                   当 GPU 空闲占比 > 60%、P95 > 85%、稳定均值 < 30% 时，判定编码
                   器（T3）是真正瓶颈。此时不再增大 result_queue（只增 PinnedPool
                   内存压力，对提速毫无帮助），并对超大值主动缩小以回收内存。

  [FIX-T3-MEMCAP]  PinnedPool 雪球效应修复：_auto_queue_depths() 和
                   get_queue_suggestions() 均新增 PinnedPool 内存上限约束
                   （当前已由 FIX-POOL-AUTOSCALE 按 GPU 型号自动缩放）；result_queue 不再无限制增大，
                   防止锁页内存随段数累积至 2 GiB+ 导致 DMA 带宽压力恶化。

  [FIX-RETUNE-SKIP] T2 RETUNE 稳定性修复：引入 _CALIB_SKIP=3 跳过段初热身 batch，
                   避免流水线未稳定时的突发性快速采样污染 T2 测量值和 T2-CACHE，
                   同时过滤 < 1ms 的明显异常值（enqueue burst 假像）。

  [FIX-CALIB-KEY]  修复 _last_calib_config 缺少 model_name 的 bug：与 run() 中
                   _current_cfg 的构造不一致，导致跨模型切换时 t2_measured_ms
                   未正确清零，复用了上一模型的 T2 缓存值。

  [FIX-LOSSLESS]   crf=0 无损参数正确映射：
                   · libx264       → -qp 0（严格逐像素无损）
                   · libx265       → -x265-params lossless=1（crf=0 在 x265 中
                                     不是无损！仅为极高质量有损）
                   · h264/hevc_nvenc → -qp 0 -b:v 0（切换至常量 QP 无损模式，
                                     去掉 -rc:v vbr / -cq:v）

【v6.3.1 新增修复（基于 v6.3.0）】
  [FIX-INFER-THREAD]  T2 推理从 run() 主线程提取为独立线程 _infer_loop()，
                       仿 ESRGAN _sr_thread 架构，消除 GIL 竞争，波形趋于平顶。

  [FIX-DOUBLEBUF-H2D] 单槽预取（_prefetch_item）→ deque 双槽；
                       _try_prefetch_next() 以 while 循环填满至 2 个 in-flight，
                       大 bs 下 H2D 等待气泡消除，GPU 利用率更平滑。

【v6.3.0 核心升级（基于 v6.2.5）】
  [STREAM-DUAL]    双 transfer stream 架构：
                   · stream_h2d  专用 H2D 预取（原 stream_transfer 职责一拆二）
                   · stream_d2h  专用 D2H 输出
                   彻底消除旧版中同一条流上 D2H 阻塞 H2D 预取的根因；
                   stream_compute.wait_stream(stream_h2d) 只等 H2D，不再被 D2H 污染。
                   float() 类型转换从 default stream 移入 stream_d2h，主线程可立即
                   提交下一批推理，消除每批约 20-50ms 的 default stream 空档。
                   实现三流全重叠：compute(N) ‖ h2d(N+1) ‖ d2h(N-1)。

  [EVENT-POOL]     CudaEventPool 预分配 CUDA Event 对象池（默认 8 个），避免
                   每批次 cudaEventCreate/Destroy 带来的约 0.5-1ms 开销；
                   T3-Writer 写完后将 Event 归还池，形成完整的复用闭环。

  [BATCH-UP]       默认 batch_size 24 → 48，充分利用 T4 30% 空闲显存，
                   理论吞吐提升约 20-30%（TRT Engine 首次运行需重建缓存）。

  [GPU-MONITOR]    后台 GPU 监测线程（1 秒采样），运行结束后打印：
                   · 完整运行利用率：均值 / P50 / P95 / 峰值 / σ / 空闲占比
                   · 稳定段（去掉前 15% 预热）同上四项
                   · 最近 30s 滑动窗口：均值 / P95
                   · 显存：均值 / P95 / 峰值
                   · 三项调优建议：batch_size / pair_queue / result_queue。

【v6.2.5 完整特性（全部继承）】
  推理加速：FP16 / torch.compile / CUDA Graph / TensorRT / OOM 自动降级
  I/O 加速：NVDEC / NVENC / 异步预取 / 批量写帧
  三级深度流水线：T1-Reader / T2-Infer / T3-Writer
  AUTO-TUNE 队列深度 / T2 持久化缓存 / RETUNE 偏差报告
  PINNED-D2H 结果零拷贝 / 死锁看门狗 / JSON 性能报告

【命令行使用示例】
  # 基础用法（FP16 + torch.compile + NVDEC/NVENC 自动启用）
  python process_video_v6_3_0_single.py \\
      --input input.mp4 --output output_2x.mp4 --scale 2

  # TensorRT 加速（bs=48，首次构建 Engine）
  python process_video_v6_3_0_single.py \\
      --input input.mp4 --output output.mp4 --scale 2 --use-tensorrt

  # 输出性能报告
  python process_video_v6_3_0_single.py \\
      --input input.mp4 --output output.mp4 --scale 2 --report report.json

【注意事项】
  · v6.3.0 升级 batch_size 默认值为 48；TRT 用户若沿用旧 .trt 缓存（bs=24），
    首次运行会因 shape 不匹配而自动删除旧缓存并重建 Engine（约需 20-30 分钟）。
  · stream_transfer 属性已拆分为 stream_h2d / stream_d2h，上层调用方如有直接
    引用 processor.stream_transfer 需改为 stream_h2d（预取）或 stream_d2h（输出）。
"""

# ════════════════════════════════════════════════════════════════════
# 模块化 ifrnet_video 包的主入口。v6.4.5.1 单文件被拆分为本包，目录与
# 模块划分镜像 external/realesrgan_video（main / pipeline / nvenc_sdk /
# tensorrt_accel / ffmpeg_io / config / ifrnet_utils）。
# 下方实现均逐字保留自原文件，未做行为改动。
# ════════════════════════════════════════════════════════════════════


from __future__ import annotations

import argparse
import dataclasses
import json
import os
import queue
import subprocess
import sys
import threading
import time
import warnings
from collections import deque
from contextlib import nullcontext
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


warnings.filterwarnings('ignore')

# ── [FIX-NML] stderr 过滤器 ──────────────────────────────────────────────────
import re as _re, sys as _sys

class _NVMLFilter:
    _pat = _re.compile(r'NVML_SUCCESS|INTERNAL ASSERT FAILED.*CUDACachingAllocator')
    def __init__(self, s): self._s = s
    def write(self, m):
        if not self._pat.search(m): self._s.write(m)
    def flush(self): self._s.flush()
    def __getattr__(self, a): return getattr(self._s, a)

_sys.stderr = _NVMLFilter(_sys.stderr)

os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import logging as _logging
_logging.getLogger('torch._inductor.utils').setLevel(_logging.ERROR)
_logging.getLogger('torch.utils._sympy.interp').setLevel(_logging.ERROR)
_logging.getLogger('torch.utils._sympy').setLevel(_logging.ERROR)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── 路径配置（原单文件路径常量已迁移至 config.py） ─────────────────────────────

# ── ifrnet_video 模块化分解：跨模块导入 ───────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from ifrnet_video.config import MODEL_NAME_MAP, MODEL_STRIDE, base_dir, models_ifrnet
from ifrnet_video.ifrnet_utils import (
    PinnedResultPool,
    PinnedRingBuffer,
    TensorPool,
    ThroughputMeter,
    _PinnedResultItem,
    _load_ifrnet_module,
    frames_to_tensor,
    tensor_to_np,
)
from ifrnet_video.ffmpeg_io import (
    FFmpegFrameReader,
    FFmpegWriter,
    HardwareCapability,
    _detect_encode_parallelism,
    _probe_video,
    _software_encode_fps,
)
from ifrnet_video.nvenc_sdk import (
    FFmpegMuxer,
    NVENCEncoder,
    _NVENC_CRF0_FORCE_CONSTQP,
    _NVENC_CRF0_LOOKAHEAD,
    _NVENC_CRF0_QUALITY,
    _NVENC_LEVEL1_DEFAULT_SLOTS,
    _NVENC_LEVEL1_LOOKAHEAD,
    _NVENC_LEVEL1_RATE_MODE,
    _PRESET_P_INDEX,
    _rgb_to_nv12_gpu,
)
from ifrnet_video.pipeline import (
    GPUMonitor,
    IFRNetPipelineRunner,
    _PAIR_Q_ABS_CAP,
    _T2_CACHE_DIR_DEFAULT,
    _compute_max_pair_queue,
    _compute_max_result_queue,
    _detect_hw_profile,
)
from ifrnet_video.tensorrt_accel import TensorRTAccelMixin


Model, _ifrnet_s_mod = _load_ifrnet_module('IFRNet_S_Vimeo90K')



# ─────────────────────────────────────────────────────────────────────────────
# 核心推理类
# ─────────────────────────────────────────────────────────────────────────────

class IFRNetVideoProcessor(TensorRTAccelMixin):

    def __init__(
        self,
        model_path:       str,
        device:           str  = 'cuda',
        batch_size:       int  = 48,     # [BATCH-UP] 默认 48（原 24）
        max_batch_size:   int  = 64,
        use_fp16:         bool = True,
        use_compile:      bool = True,
        use_cuda_graph:   bool = True,
        use_tensorrt:     bool = False,
        use_hwaccel:      bool = True,
        codec:            str  = 'libx264',
        crf:              int  = 23,
        x264_preset:      str  = 'medium',
        rate_mode:        str  = _NVENC_LEVEL1_RATE_MODE,
        lookahead_depth:  int  = _NVENC_LEVEL1_LOOKAHEAD,
        keep_audio:       bool = True,
        ffmpeg_bin:       str  = 'ffmpeg',
        report_json:      Optional[str] = None,
        trt_cache_dir:    Optional[str] = None,
        t2_cache_dir:     Optional[str] = None,
        model_name: str = 'IFRNet_S_Vimeo90K',   # 新增
        quiet:            bool = True,
    ):
        self.model_path      = model_path
        self.device_str      = device
        self.batch_size      = batch_size
        self._max_batch_size = max(batch_size, max_batch_size)
        self._oom_cooldown   = 0
        self.use_fp16        = use_fp16 and torch.cuda.is_available()
        self.use_cuda_graph  = use_cuda_graph and torch.cuda.is_available()
        self.use_tensorrt    = use_tensorrt
        self.use_hwaccel     = use_hwaccel
        self.codec           = codec
        self.crf             = crf
        self.x264_preset     = x264_preset
        self._rate_mode      = rate_mode          # 实例级 rate_mode（可被 processor 层覆盖模块常量）
        self._la_depth       = lookahead_depth    # 实例级 lookahead 深度
        self.keep_audio      = keep_audio
        self.ffmpeg_bin      = ffmpeg_bin
        self.report_json     = report_json
        self.dtype           = torch.float16 if self.use_fp16 else torch.float32
        self.trt_cache_dir   = trt_cache_dir
        self.t2_cache_dir    = t2_cache_dir or _T2_CACHE_DIR_DEFAULT
        self.model_name = model_name      # 保存模型名称
        self.quiet           = quiet
        self._pipeline_runner: Optional[IFRNetPipelineRunner] = None
        self._result_pool:     Optional[PinnedResultPool]     = None
        self._ring_buf:        Optional[PinnedRingBuffer]    = None  # [FIX-T3-V643] Level 2/3
        self._pool          = TensorPool()
        self._graph:        dict = {}
        self._graph_inputs: dict = {}
        self._timing:       List[float] = []

        # 跨段自适应队列（由上一次运行的综合建议决定）
        self._next_pair_queue = None      # int or None
        self._next_result_queue = None    # int or None
        self._next_t3_fps_measured = 0.0  # [FIX-T3-FPS] 跨段实测 T3 fps（0 表示无实测）

        # [SEGMENT-REUSE] 跨段复用追踪
        self._segment_index    = 0        # 当前分段序号（1-based，process_video 入口递增）
        self._total_segments   = 0        # 总分段数（由外部传入）
        self._last_seg_resolution: Optional[Tuple[int, int]] = None  # 上段分辨率缓存
        self._last_effective_bs = 0       # 上段有效 batch_size

        # [SEGMENT-REUSE] NVENC 编码器缓存
        self._cached_nvenc_encoder: Optional['NVENCEncoder'] = None
        self._cached_nvenc_key: Optional[Tuple[int, int, float, str, int]] = None

        # [SEGMENT-REUSE] PinnedResultPool / PinnedRingBuffer 缓存
        self._cached_result_pool: Optional['PinnedResultPool'] = None
        self._cached_pool_key: Optional[Tuple[int, int, int]] = None     # (max_BT, H_pad, W_pad)
        self._cached_pool_size: int = 0
        self._cached_ring_buf: Optional['PinnedRingBuffer'] = None
        self._cached_ring_key: Optional[Tuple[int, int, int, int]] = None  # (num_slots, max_frames, H, W)

        # [FIX-TRT-MUTEX]
        if self.use_tensorrt:
            if self.use_cuda_graph:
                self.use_cuda_graph = False
                print('  [FIX-TRT-MUTEX] use_tensorrt=True → 已禁用手动 CUDA Graph（互斥）')
            if use_compile:
                use_compile = False
                print('  [FIX-TRT-MUTEX] use_tensorrt=True → 已跳过 torch.compile（互斥）')

        self.use_compile = use_compile
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self._load_model(self.device, use_compile)

        # [GPU-MONITOR] 监测器（process_video 中启动）
        self._gpu_monitor = GPUMonitor(self.device, interval=1.0, window_seconds=30.0)

        self._trt_built   = False  # 标记 TRT Engine 是否已构建

    def _load_model(self, device: torch.device, use_compile: bool = True):
        print(f'  加载模型: {self.model_path} → {device}')
        model = Model()
        ckpt  = torch.load(self.model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt)
        model = model.to(device).eval()

        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled   = True
            print('  [FIX-CU] cudnn.benchmark = True 已启用')

        if self.use_fp16:
            model = model.half()
            print('  FP16 推理已启用')

        if use_compile and hasattr(torch, 'compile'):
            try:
                torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
                cache_dir = os.path.join(
                    os.path.dirname(os.path.abspath(self.model_path)),
                    '.torch_compile_cache',
                )
                os.makedirs(cache_dir, exist_ok=True)
                os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', cache_dir)
                model = torch.compile(model, mode='default', dynamic=True)
                if self.use_tensorrt:
                    print('  torch.compile 已加载（TRT 激活时推理走 TRT 分支，compile 不执行）')
                else:
                    print(f'  torch.compile 加速已启用 (mode=default, dynamic=True)')
                    print(f'  编译缓存目录: {cache_dir}')
                    print('  首次运行将触发编译（约1-3分钟），后续运行秒启动')
                if self.use_cuda_graph:
                    self.use_cuda_graph = False
                    if not self.use_tensorrt:
                        print('  手动 CUDA Graph 已禁用（由 torch.compile 接管）')
            except Exception as e:
                print(f'  torch.compile 不可用: {e}')
                if self.use_tensorrt and self.use_cuda_graph:
                    self.use_cuda_graph = False
                    print('  [FIX-TRT-MUTEX] compile 异常 + use_tensorrt=True → 补充禁用手动 CUDA Graph')

        self.model = model

        if device.type == 'cuda':
            self.stream_compute  = torch.cuda.Stream(device=device)
            # [STREAM-DUAL] H2D 预取专用流 / D2H 输出专用流
            self.stream_h2d = torch.cuda.Stream(device=device)
            self.stream_d2h = torch.cuda.Stream(device=device)
        else:
            self.stream_compute = self.stream_h2d = self.stream_d2h = None


    # ── CUDA Graph ────────────────────────────────────────────────────────────

    def _get_cuda_graph(self, shape_key, img0, img1, embt, imgt_approx):
        if shape_key in self._graph:
            s = self._graph_inputs[shape_key]
            s['img0'].copy_(img0)
            s['img1'].copy_(img1)
            s['embt'].copy_(embt)
            self._graph[shape_key].replay()
            return s['output']

        print(f'  [CUDA Graph] 捕获 shape={shape_key} ...')
        static_img0 = img0.clone()
        static_img1 = img1.clone()
        static_embt = embt.clone()

        for _ in range(5):
            with torch.cuda.stream(self.stream_compute):
                _ = self.model.inference(static_img0, static_img1, static_embt)
        torch.cuda.synchronize(self.device)

        g = torch.cuda.CUDAGraph()
        _saved_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False
        try:
            with torch.cuda.graph(g, stream=self.stream_compute):
                static_output = self.model.inference(static_img0, static_img1, static_embt)
        except Exception as e:
            torch.backends.cudnn.benchmark = _saved_benchmark
            try: torch.cuda.synchronize(self.device)
            except Exception: pass
            torch.cuda.empty_cache()
            self.use_cuda_graph = False
            print(f'  [CUDA Graph] 捕获失败（{type(e).__name__}: {str(e)[:120]}），'
                  f'已禁用，后续走普通推理路径。')
            with torch.cuda.stream(self.stream_compute):
                return self.model.inference(img0, img1, embt)
        finally:
            torch.backends.cudnn.benchmark = _saved_benchmark

        with torch.cuda.stream(self.stream_compute):
            g.replay()

        self._graph[shape_key] = g
        self._graph_inputs[shape_key] = {
            'img0': static_img0, 'img1': static_img1,
            'embt': static_embt, 'output': static_output,
        }
        return static_output

    # ── 核心批推理 ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer_batch(
        self,
        img0_list: List[np.ndarray],
        img1_list: List[np.ndarray],
        timesteps: List[float],
        orig_H:    int,
        orig_W:    int,
        prefetched_img0_t: Optional[torch.Tensor] = None,
        prefetched_img1_t: Optional[torch.Tensor] = None,
        return_gpu: bool = False,
    ):
        B  = len(img0_list)
        T  = len(timesteps)
        t0 = time.perf_counter()

        # ── H2D：优先使用预取 tensor ─────────────────────────────────────────
        _use_prefetch = (
            prefetched_img0_t is not None and
            prefetched_img1_t is not None and
            prefetched_img0_t.shape[0] == B and
            prefetched_img1_t.shape[0] == B
        )
        if _use_prefetch:
            img0 = prefetched_img0_t
            img1 = prefetched_img1_t
            # [STREAM-DUAL] compute 只需等待 H2D（stream_h2d），不等 D2H
            if self.stream_compute is not None and self.stream_h2d is not None:
                self.stream_compute.wait_stream(self.stream_h2d)
            elif self.stream_h2d is not None:
                self.stream_h2d.synchronize()
        else:
            # [STREAM-DUAL] H2D 在 stream_h2d 上执行
            img0 = frames_to_tensor(img0_list, self.device, self.stream_h2d, self.dtype, slot=0)
            img1 = frames_to_tensor(img1_list, self.device, self.stream_h2d, self.dtype, slot=1)
            if self.stream_compute is not None:
                self.stream_compute.wait_stream(self.stream_h2d)

        img0_exp  = img0.unsqueeze(1).expand(B, T, *img0.shape[1:]).reshape(B * T, *img0.shape[1:])
        img1_exp  = img1.unsqueeze(1).expand(B, T, *img1.shape[1:]).reshape(B * T, *img1.shape[1:])
        shape_key = (B * T, 3, img0.shape[2], img0.shape[3], T)

        # ── 推理分支 ──────────────────────────────────────────────────────────
        if self.use_cuda_graph:
            with torch.cuda.stream(self.stream_compute):
                t_vals      = timesteps * B
                embt        = torch.tensor(t_vals, dtype=self.dtype,
                                           device=self.device).view(-1, 1, 1, 1)
                img0_big    = img0_exp.contiguous()
                img1_big    = img1_exp.contiguous()
                imgt_approx = img0_big * (1 - embt) + img1_big * embt
                pred_big    = self._get_cuda_graph(shape_key, img0_big, img1_big,
                                                   embt, imgt_approx)
            if self._pipeline_runner is not None:
                self._pipeline_runner._try_prefetch_next()

        elif getattr(self, '_trt_ok', False):
            # [TRT-EXTRACT] TRT 推理分支已抽取至 tensorrt_accel.TensorRTAccelMixin。
            pred_big = self._infer_batch_trt(img0_exp, img1_exp, timesteps, B)

        else:
            autocast_ctx = (
                torch.amp.autocast(device_type='cuda', dtype=torch.float16)
                if self.use_fp16 else nullcontext()
            )
            stream_ctx = (
                torch.cuda.stream(self.stream_compute)
                if self.stream_compute else nullcontext()
            )
            with stream_ctx, autocast_ctx:
                t_vals   = timesteps * B
                embt     = torch.tensor(t_vals, dtype=self.dtype,
                                        device=self.device).view(-1, 1, 1, 1)
                pred_big = self.model.inference(img0_exp, img1_exp, embt)
            if self._pipeline_runner is not None:
                self._pipeline_runner._try_prefetch_next()

        # ── [FIX-T3-V643-GPU] GPU-STAY 路径（Level 1 NVENC 直通） ───────────
        # 推理结果保持在 GPU 上，仅做 float/clamp/byte/HWC 处理，
        # 跳过 D2H → 避免 GPU↔CPU round-trip。编码由 _infer_loop 完成。
        if return_gpu and pred_big.device.type == 'cuda':
            BT = pred_big.shape[0]
            with torch.cuda.stream(self.stream_compute):
                if self.use_fp16:
                    pred_f = pred_big.float()
                else:
                    pred_f = pred_big
                # GPU 侧: 量化 + CHW→HWC + 裁剪到 orig_H×orig_W（保持 RGB）
                pred_u8 = (
                    pred_f.clamp_(0.0, 1.0).mul_(255.0).byte()
                    .permute(0, 2, 3, 1)                              # CHW → HWC (BT,H,W,3) RGB
                    .contiguous()
                )
                interp_gpu = pred_u8[:, :orig_H, :orig_W, :].contiguous()
                # img1 (原始帧): 同样在 GPU 上做量化 + 裁剪
                # img1 来自 prefetch (数据已是 RGB，无需翻转)，形状 (B, C, H_pad, W_pad)
                if self.use_fp16:
                    img1_gpu = img1.float()
                else:
                    img1_gpu = img1
                img1_u8 = (
                    img1_gpu.clamp_(0.0, 1.0).mul_(255.0).byte()
                    .permute(0, 2, 3, 1)                              # CHW → HWC (B,H,W,3) RGB
                    .contiguous()
                )
                img1_rgb = img1_u8[:, :orig_H, :orig_W, :].contiguous()
            # [FIX-STREAM-SYNC] 确保 default stream 等待 stream_compute 完成。
            # [FIX-STREAM-SYNC-v2] wait_stream 仅创建 GPU 侧依赖（Infer 线程 default stream），
            # 不阻塞 CPU 线程。Writer 线程使用 per-thread default stream，与 Infer 线程
            # 的 default stream 无同步关系。必须加 CPU 侧 synchronize() 确保 stream_compute
            # 上所有 GPU 写入（interp_gpu 和 img1_rgb 的 clamp/permute/contiguous）完全完成，
            # 再通过 result_queue 将 tensor 交给 Writer 线程，消除非确定性的隔帧花屏。
            # v6.4.3 不受影响因其同步编码耗时足够长 (>10ms) 掩盖了竞争窗口；
            # v6.4.4 异步 _NVENCEncodeThread 提交后 Writer 即刻进入下一批 → 窗口缩短。
            torch.cuda.default_stream(self.device).wait_stream(self.stream_compute)
            self.stream_compute.synchronize()
            self._timing.append(time.perf_counter() - t0)
            # 返回 GPU tensor: interp_gpu (BT,orig_H,orig_W,3) uint8 RGB
            #                  img1_rgb  (B, orig_H,orig_W,3) uint8 RGB
            # 元数据: B, T, orig_H, orig_W
            return ('GPU', interp_gpu, img1_rgb, B, T, orig_H, orig_W)

        # ── [FIX-T3-V643] RING-BUFFER D2H 路径（Level 2/3） ──────────────
        # float() + 量化 + DMA 全在 stream_d2h 上执行；GPU 侧保持 RGB + 裁剪；
        # 写入 PinnedRingBuffer slot（无 padding，每帧连续），返回轻量句柄。
        if (self._ring_buf is not None
                and self.stream_d2h is not None
                and pred_big.device.type == 'cuda'):
            BT = pred_big.shape[0]
            slot_id, slot_tensor = self._ring_buf.writer_acquire()
            ev = None
            try:
                with torch.cuda.stream(self.stream_d2h):
                    self.stream_d2h.wait_stream(self.stream_compute)
                    if self.use_fp16:
                        pred_f = pred_big.float()
                    else:
                        pred_f = pred_big
                    # [FIX-T3-V643] GPU 侧: 量化 + CHW→HWC + 裁剪到 orig_H×orig_W（保持 RGB）
                    pred_u8 = (
                        pred_f.clamp_(0.0, 1.0).mul_(255.0).byte()
                        .permute(0, 2, 3, 1)                              # CHW → HWC (BT,H,W,3) RGB
                        .contiguous()
                    )
                    # 裁剪到原始分辨率，保持 RGB（pix_fmt 已是 rgb24）
                    pred_rgb = pred_u8[:, :orig_H, :orig_W, :].contiguous()
                    slot_tensor[:BT].copy_(pred_rgb, non_blocking=True)
                ev = (self._pipeline_runner._event_pool.acquire()
                      if self._pipeline_runner is not None
                      else torch.cuda.Event())
                ev.record(self.stream_d2h)
            except Exception:
                # 归还已获取的 slot（通过 release write_sem）
                self._ring_buf._write_sem.release()
                if ev is not None and self._pipeline_runner is not None:
                    self._pipeline_runner._event_pool.release(ev)
                raise
            self._timing.append(time.perf_counter() - t0)
            # 返回轻量句柄：("RING", slot_id, event, B, T, orig_H, orig_W, BT)
            return ('RING', slot_id, ev, B, T, orig_H, orig_W, BT)

        # ── [STREAM-DUAL] PINNED-D2H 路径（Level 4） ──────────────────────
        # float() + 量化 + DMA 全在 stream_d2h 上执行，彻底绕开 default stream，
        # 主线程可立即提交下一批 TRT/compute kernel，消除每批空档。
        if (self._result_pool is not None
                and self.stream_d2h is not None
                and pred_big.device.type == 'cuda'):
            BT     = pred_big.shape[0]
            pinned = self._result_pool.acquire()
            ev = None   # ✅ [EVENT-POOL] 提前初始化，确保异常路径可安全归还
            try:
                with torch.cuda.stream(self.stream_d2h):
                    # 直接等 compute stream，不经 default stream
                    self.stream_d2h.wait_stream(self.stream_compute)
                    # float() 转换也在 stream_d2h 上排队，不阻塞主线程
                    if self.use_fp16:
                        pred_f = pred_big.float()
                    else:
                        pred_f = pred_big
                    pred_u8 = (
                        pred_f.clamp_(0.0, 1.0).mul_(255.0).byte()
                        .permute(0, 2, 3, 1).contiguous()   # (BT, H, W, 3) RGB uint8
                    )
                    pinned[:BT].copy_(pred_u8, non_blocking=True)
                # [EVENT-POOL] 从池中取 Event，记录在 stream_d2h
                ev = (self._pipeline_runner._event_pool.acquire()
                      if self._pipeline_runner is not None
                      else torch.cuda.Event())
                ev.record(self.stream_d2h)
            except Exception:
                self._result_pool.release(pinned)
                if ev is not None and self._pipeline_runner is not None:   # ✅ ev 已取出则归还，防止池耗尽
                    self._pipeline_runner._event_pool.release(ev)
                raise
            self._timing.append(time.perf_counter() - t0)
            return _PinnedResultItem(
                buf=pinned, event=ev,
                B=B, T=T, orig_H=orig_H, orig_W=orig_W,
                pool=self._result_pool,
            )

        # ── 同步回退路径（CPU / pool 不可用）─────────────────────────────────
        # 此处保留显式 wait + float()，保证同步路径的正确性。
        if self.stream_compute is not None:
            torch.cuda.default_stream(self.device).wait_stream(self.stream_compute)
        if self.use_fp16:
            pred_big = pred_big.float()
        all_np = tensor_to_np(pred_big, orig_H, orig_W, sync_stream=self.stream_compute)
        result = [[all_np[i * T + j] for j in range(T)] for i in range(B)]
        self._timing.append(time.perf_counter() - t0)
        return result

    # ── OOM 自动降级 ──────────────────────────────────────────────────────────

    def _estimate_safe_batch_size(self, H: int, W: int) -> int:
        if not torch.cuda.is_available():
            return 1
        try:
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            # [FIX-BATCHCAP] mem_get_info 仅返回 OS 层面真正空闲 VRAM，
            # 不含 PyTorch allocator 已 reserved 但未 allocated 的可复用缓存。
            # 跨段时 TRT engine 保留大量 reserved 池，导致 free_bytes 严重低估，
            # 必须叠加 cached_free（reserved - allocated）才能得到真实可用量。
            cached_free    = (torch.cuda.memory_reserved(self.device)
                              - torch.cuda.memory_allocated(self.device))
            effective_free = free_bytes + cached_free
            bytes_per_frame = H * W * 3 * 2 * 6
            estimated = max(1, int(effective_free * 0.7 / bytes_per_frame))
            return min(estimated, self._max_batch_size)
        except Exception:
            return 1

    def _safe_infer(self, img0_list, img1_list, timesteps, orig_H, orig_W,
                    prefetched_img0_t=None, prefetched_img1_t=None,
                    return_gpu=False):
        in_oom_cascade = False
        _first_attempt = True

        while True:
            try:
                _p0 = prefetched_img0_t if _first_attempt else None
                _p1 = prefetched_img1_t if _first_attempt else None
                result = self._infer_batch(img0_list, img1_list, timesteps, orig_H, orig_W,
                                           prefetched_img0_t=_p0, prefetched_img1_t=_p1,
                                           return_gpu=return_gpu)
                in_oom_cascade = False
                if self._oom_cooldown > 0:
                    self._oom_cooldown -= 1
                elif (self.batch_size < self._max_batch_size
                      and not getattr(self, '_trt_ok', False)):
                    # [FIX-HIGHRES-OOM] 高分辨率 (>2MP) 不自动恢复 batch_size，
                    # 避免 batch_size 震荡导致反复 OOM。
                    _megapixels = orig_H * orig_W / 1_000_000
                    if _megapixels < 2.0:
                        new_bs = min(self.batch_size + 1, self._max_batch_size)
                        print(f'[恢复] 显存充裕，batch_size {self.batch_size} → {new_bs}')
                        self.batch_size = new_bs
                return result

            except torch.cuda.OutOfMemoryError:
                _first_attempt = False
                prefetched_img0_t = prefetched_img1_t = None
                torch.cuda.empty_cache()
                self._pool.clear()
                self._graph.clear()
                self._graph_inputs.clear()

                # [STREAM-DUAL] OOM 后重建全部三条流
                if self.stream_compute is not None:
                    try: torch.cuda.synchronize(self.device)
                    except Exception: pass
                    self.stream_compute = torch.cuda.Stream(device=self.device)
                    self.stream_h2d     = torch.cuda.Stream(device=self.device)
                    self.stream_d2h     = torch.cuda.Stream(device=self.device)

                if not in_oom_cascade:
                    safe_ceiling = max(1, self.batch_size - 1)
                    if self._max_batch_size > safe_ceiling:
                        print(f'[OOM] 永久降低 max_batch_size: {self._max_batch_size} → {safe_ceiling}')
                        self._max_batch_size = safe_ceiling
                    in_oom_cascade = True

                if self.batch_size <= 1:
                    _megapixels_oom = orig_H * orig_W / 1_000_000
                    print(f'\n[OOM] batch_size=1 仍 OOM ({_megapixels_oom:.1f}MP)，深度清理后按剩余显存估算恢复...')
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        try: torch._dynamo.reset()
                        except Exception: pass
                        torch.cuda.empty_cache()
                    recovered_bs = self._estimate_safe_batch_size(orig_H, orig_W)
                    if recovered_bs < self._max_batch_size:
                        print(f'[OOM] 深度清理后估算安全 batch_size={recovered_bs}，'
                              f'更新 max_batch_size: {self._max_batch_size} → {recovered_bs}')
                        self._max_batch_size = recovered_bs
                    self.batch_size    = recovered_bs
                    self._oom_cooldown = 20
                    in_oom_cascade     = False
                    # [FIX-HIGHRES-OOM] 高分辨率 OOM: 额外减小 pair_queue 覆盖值,
                    # 减少流水线中缓存的帧对数据量。
                    if _megapixels_oom > 1.0:
                        _cur_pq = getattr(self, '_next_pair_queue', None)
                        _prev_pq = _cur_pq if _cur_pq is not None else 4
                        _new_pq = max(1, _prev_pq // 2)
                        self._next_pair_queue = _new_pq
                        print(f'[OOM] 高分辨率 OOM: pair_queue {_prev_pq} → {_new_pq}（减少帧缓冲）')
                    print(f'[OOM] 恢复 batch_size={self.batch_size}，继续处理...')
                    continue

                self.batch_size    = max(1, self.batch_size // 2)
                self._oom_cooldown = 10
                print(f'\n[OOM] 自动降低 batch_size → {self.batch_size}')

            except (RuntimeError, Exception) as _cg_err:
                _first_attempt = False
                prefetched_img0_t = prefetched_img1_t = None
                _err_s = str(_cg_err)
                _is_cg = (
                    'FIND was unable to find an engine' in _err_s
                    or 'cudaErrorStreamCaptureInvalidated' in _err_s
                    or 'operation failed due to a previous error during capture' in _err_s
                    or 'cudaErrorIllegalState' in _err_s
                    or ('AcceleratorError' in type(_cg_err).__name__ and 'capture' in _err_s)
                )
                if self.use_cuda_graph and _is_cg:
                    print(f'[CUDA Graph 错误] {type(_cg_err).__name__}: {_err_s[:200]}')
                    print('  → 禁用 CUDA Graph，重建 CUDA 流，后续走普通推理路径...')
                    self.use_cuda_graph = False
                    self._graph.clear()
                    self._graph_inputs.clear()
                    self._pool.clear()
                    torch.cuda.empty_cache()
                    if self.stream_compute is not None:
                        try: torch.cuda.synchronize(self.device)
                        except Exception: pass
                        self.stream_compute = torch.cuda.Stream(device=self.device)
                        self.stream_h2d     = torch.cuda.Stream(device=self.device)
                        self.stream_d2h     = torch.cuda.Stream(device=self.device)
                    continue
                raise

    # ── 单段处理核心 ──────────────────────────────────────────────────────────

    # ── [SEGMENT-REUSE] 跨段资源复用方法 ────────────────────────────────────────

    def _is_first_segment(self) -> bool:
        return self._segment_index <= 1

    # 帧数守恒确保每段独立排空，无需 _is_last_segment

    def _get_or_create_nvenc_encoder(self, W: int, H: int, fps: float,
                                      preset: str, qp: int,
                                      rate_mode: str = "vbr_hq", la_depth: int = 0,
                                      pipeline_depth: int = _NVENC_LEVEL1_DEFAULT_SLOTS) -> 'NVENCEncoder':
        """跨段复用 NVENC 编码器，参数不变时跳过 11 行初始化日志和 DLL 加载。"""
        key = (W, H, fps, preset, qp, rate_mode, la_depth, pipeline_depth)
        if self._cached_nvenc_encoder is not None and self._cached_nvenc_key == key:
            if not self.quiet:
                print(f'   [NVENC] 复用已激活编码器 ({W}x{H}@{fps:.1f}fps rate={rate_mode})', flush=True)
            return self._cached_nvenc_encoder
        if self._cached_nvenc_encoder is not None:
            self._cached_nvenc_encoder.close()
        encoder = NVENCEncoder(W, H, fps, preset=preset, qp=qp,
                               rate_mode=rate_mode, la_depth=la_depth,
                               pipeline_depth=pipeline_depth)
        self._cached_nvenc_encoder = encoder
        self._cached_nvenc_key = key
        return encoder

    def _get_or_create_result_pool(self, pool_size: int, max_BT: int,
                                    H_pad: int, W_pad: int) -> 'Tuple[PinnedResultPool, bool]':
        """跨段复用 PinnedResultPool，参数匹配时跳过锁页内存分配。

        保留 FIX-POOL-LEAK 语义：全局只有一个池实例，不会跨段累积。"""
        key = (max_BT, H_pad, W_pad)
        if (self._cached_result_pool is not None and self._cached_pool_key == key
                and self._cached_pool_size >= pool_size):
            return self._cached_result_pool, False
        if self._cached_result_pool is not None:
            self._cached_result_pool.free()
        pool = PinnedResultPool(pool_size, max_BT, H_pad, W_pad)
        self._cached_result_pool = pool
        self._cached_pool_key = key
        self._cached_pool_size = pool_size
        return pool, True

    def _get_or_create_ring_buf(self, num_slots: int, max_frames_per_slot: int,
                                 H: int, W: int) -> 'Tuple[PinnedRingBuffer, bool]':
        """跨段复用 PinnedRingBuffer。"""
        key = (num_slots, max_frames_per_slot, H, W)
        if self._cached_ring_buf is not None and self._cached_ring_key == key:
            return self._cached_ring_buf, False
        if self._cached_ring_buf is not None:
            self._cached_ring_buf.free()
        ring = PinnedRingBuffer(num_slots=num_slots, max_frames_per_slot=max_frames_per_slot,
                                H=H, W=W)
        self._cached_ring_buf = ring
        self._cached_ring_key = key
        return ring, True

    def cleanup(self, release_model: bool = False):
        """统一释放所有跨段缓存资源。由处理器层在全部流程结束后调用。

        Args:
            release_model: False（默认）仅释放跨段缓存的 NVENC/Pool/RingBuffer，
                完整保留模型/TRT Engine/CUDA Graph —— 跨段复用与历史行为不变；
                True 追加销毁模型/TRT/CUDA Graph，仅供处理器层在全部阶段结束、
                实例即将销毁时调用（_cleanup_video_processor）。
        """
        if self._cached_nvenc_encoder is not None:
            self._cached_nvenc_encoder.close()
            self._cached_nvenc_encoder = None
            self._cached_nvenc_key = None
        if self._cached_result_pool is not None:
            self._cached_result_pool.free()
            self._cached_result_pool = None
            self._cached_pool_key = None
            self._cached_pool_size = 0
        if self._cached_ring_buf is not None:
            self._cached_ring_buf.free()
            self._cached_ring_buf = None
            self._cached_ring_key = None
        if not release_model:
            return
        # [VRAM-CLEANUP] 仅 release_model=True 时执行：进一步释放重量级 GPU 资源。
        # 释放顺序：CUDA Graph 静态缓冲 → TRT context（依赖 engine，先销毁）→ TRT engine → torch 模型。
        # 仅在全部阶段处理完成后调用，不影响分段间的 TRT Engine 复用；
        # 处理器层 del 本实例后不会复用（_video_processor=None 时按需重建，TRT 引擎从 .trt_cache 反序列化）。
        # empty_cache 由处理器层在 del 实例后统一调用，此处不重复。
        try:
            self._graph.clear()
            self._graph_inputs.clear()
        except Exception:
            pass
        if getattr(self, '_trt_context', None) is not None:
            del self._trt_context
            self._trt_context = None
        if getattr(self, '_trt_engine', None) is not None:
            del self._trt_engine
            self._trt_engine = None
        self._trt_ok = False
        if getattr(self, 'model', None) is not None:
            del self.model
            self.model = None

    def _process_segment(
        self,
        input_path:         str,
        output_path:        str,
        scale:              float,
        frame_start:        int  = 0,
        frame_end:          int  = -1,
        skip_first_output:  bool = False,
        audio_src:          Optional[str] = None,
        codec_override:     Optional[str] = None,
        extra_codec_args:   Optional[List[str]] = None,
        worker_label:       str  = '',
        preview:            bool = False,
        preview_interval:   int  = 30,
        # 跨段自适应队列建议
        pair_queue_override:   Optional[int] = None,
        result_queue_override: Optional[int] = None,
        t3_fps_measured:       float = 0.0,   # [FIX-T3-FPS] 跨段实测 T3 fps
    ) -> Tuple[bool, int, int]:
        reader = FFmpegFrameReader(
            input_path,
            frame_start  = frame_start,
            frame_end    = frame_end,
            prefetch     = self.batch_size * 3,
            use_hwaccel  = self.use_hwaccel,
            ffmpeg_bin   = self.ffmpeg_bin,
            pad_stride   = MODEL_STRIDE,
        )
        W, H      = reader.width, reader.height
        fps       = reader.fps
        n_seg_est = reader._segment_frames

        bytes_per_frame = W * H * 3 * 2 * 6
        # [SEGMENT-REUSE] 分辨率未变时跳过 VRAM 重算，复用上段 effective_bs
        if self._last_seg_resolution == (W, H) and self._last_effective_bs > 0:
            effective_bs = self._last_effective_bs
        else:
            # [FIX-BATCHCAP] mem_get_info()[0] 仅返回 OS 层面空闲 VRAM，跨段后 PyTorch
            # allocator 仍持有大量 reserved 缓存（TRT engine 等），导致估算严重偏低。
            # 修复：effective_free = OS空闲 + PyTorch可复用缓存（reserved - allocated）
            _seg_free = 0
            if torch.cuda.is_available():
                _raw_free, _  = torch.cuda.mem_get_info(self.device)
                _cached_free  = (torch.cuda.memory_reserved(self.device)
                                 - torch.cuda.memory_allocated(self.device))
                _seg_free = _raw_free + _cached_free
            effective_bs = self.batch_size
            if _seg_free > 0:
                # [FIX-HIGHRES-OOM] 综合 VRAM 预估：batch 数据 + CUDA Graph + 模型开销
                # 高分辨率下 CUDA Graph 额外消耗约 1-2× batch 数据量，需使用更保守的安全因子。
                _megapixels = W * H / 1_000_000
                _cugraph_safety = 0.45 if _megapixels > 1.0 else 0.60  # >1MP 时更保守
                _vram_est_bytes = bytes_per_frame * effective_bs * 2.5  # batch × (1 + CUDA Graph ~1.5x)
                if _vram_est_bytes > _seg_free * 0.80:
                    # 高分辨率或 VRAM 紧张：切换到保守因子
                    _cugraph_safety = 0.35
                res_max_bs = max(1, int(_seg_free * _cugraph_safety / bytes_per_frame))
                if effective_bs > res_max_bs:
                    if self._segment_index <= 1:
                        print(f'[分辨率限制] {W}×{H} ({_megapixels:.1f}MP) '
                              f'batch_size {effective_bs} → {res_max_bs} '
                              f'(VRAM free={_seg_free/1024**3:.1f}GiB safety={_cugraph_safety})')
                    effective_bs = res_max_bs
                if self._max_batch_size > res_max_bs:
                    self._max_batch_size = max(effective_bs, res_max_bs)
            self._last_seg_resolution = (W, H)
            self._last_effective_bs = effective_bs

        # [FIX-NVENC-UNIFIED] 在分辨率检查后缓存 hw_profile，
        # 作为 best_encoder() 的主判断依据，确保与 AUTO-TUNE 的 nvenc 检测一致
        if not hasattr(self, '_hw_profile_cache'):
            self._hw_profile_cache = _detect_hw_profile(self.device)

        pad_h    = reader._pad_h
        pad_w    = reader._pad_w

        scale_frac = Fraction(scale).limit_denominator(64)
        n_interp   = int(scale_frac) - 1
        if n_interp < 1:
            print(f'[{worker_label}] 错误: scale 必须 ≥ 2，当前={scale}')
            reader.close()
            return False, 0, 0
        if n_interp > 32:
            scale_frac = Fraction(33)
            n_interp   = 32
        timesteps = [float(Fraction(i, int(scale_frac))) for i in range(1, int(scale_frac))]
        new_fps   = fps * float(scale_frac)

        # [FIX-NVENC-UNIFIED] 传入 hw_profile，统一两套 NVENC 检测路径
        _lossless_extra = None
        if self.crf == 0 and not codec_override:
            use_codec, _lossless_extra = HardwareCapability.lossless_encoder()
        else:
            use_codec = codec_override or HardwareCapability.best_encoder(
                self.codec, hw_profile=self._hw_profile_cache)
        # crf=0 时使用 lossless_encoder 的 extra_args（除非调用方显式传入了 extra_codec_args）
        use_extra = extra_codec_args if extra_codec_args is not None else _lossless_extra
        if 'nvenc' in use_codec:
            if self._segment_index <= 1:
                print(f'\n[{worker_label}] NVENC 编码已激活: {use_codec}')

        # [FIX-TSTART] 含 warmup 的端到端计时
        t_start = time.time()

        # ── torch.compile 预热 ────────────────────────────────────────────────
        if (self.use_compile
                and not getattr(self, '_warmup_done', False)
                and not getattr(self, '_trt_ok', False)):
            _WARM_H, _WARM_W = 32, 32
            _bs_warm = 1
            print(f'  [预热] torch.compile 编译中 (小形状预热 {_bs_warm}×3×{_WARM_H}×{_WARM_W})...',
                  flush=True)
            _t_warm = time.perf_counter()
            try:
                with torch.no_grad():
                    _d0   = torch.zeros(_bs_warm, 3, _WARM_H, _WARM_W,
                                        dtype=self.dtype, device=self.device)
                    _d1   = torch.zeros_like(_d0)
                    _embt = torch.tensor([0.5] * _bs_warm,
                                         dtype=self.dtype, device=self.device).view(-1, 1, 1, 1)
                    _out  = self.model.inference(_d0, _d1, _embt)
                    del _out, _d0, _d1, _embt
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                print(f'  [预热] 编译完成，耗时 {time.perf_counter()-_t_warm:.1f}s', flush=True)
            except Exception as _we:
                print(f'  [预热] 编译失败，回退至 eager 模式: {_we}', flush=True)
                if hasattr(self.model, '_orig_mod'):
                    self.model = self.model._orig_mod
                else:
                    try: torch._dynamo.reset()
                    except Exception: pass
            self._warmup_done = True

        # [FIX-T3-V643] 四级自动回退编码路径探测
        _nvenc_encoder = None
        _active_level = 4  # 默认最低级别
        _use_nvenc_direct = False

        # [V6451-CRF0-DECOUPLE] 解耦 crf=0 与 rate_mode/lookahead：
        # crf=0 历史上强制 CONSTQP qp=0 + la_depth=0（三元表达式），
        # 现在通过 _NVENC_CRF0_FORCE_CONSTQP 常量控制该行为。
        # True (默认) → 行为与当前 100% 一致。
        # False → crf=0 不覆盖 rate_mode/lookahead，使用独立 quality 常量。
        _level1_pd = getattr(self, '_slot_count', _NVENC_LEVEL1_DEFAULT_SLOTS)
        if self.crf == 0 and _NVENC_CRF0_FORCE_CONSTQP:
            _level1_qp = 0
            _level1_rate = "constqp"
            _level1_la = 0
        elif self.crf == 0:
            _level1_qp = _NVENC_CRF0_QUALITY
            _level1_rate = getattr(self, '_rate_mode', _NVENC_LEVEL1_RATE_MODE)
            _level1_la = getattr(self, '_la_depth', _NVENC_CRF0_LOOKAHEAD)
        else:
            _level1_qp = self.crf
            _level1_rate = getattr(self, '_rate_mode', _NVENC_LEVEL1_RATE_MODE)
            _level1_la = getattr(self, '_la_depth', _NVENC_LEVEL1_LOOKAHEAD)

        # CONSTQP 硬件静默禁用 LA；决策层清零避免 [NVENC] 日志和 cache key 携
        # 带无意义的 LA 值（NVENCEncoder 内部也会清零，但决策层需保证自身一致性）。
        if _level1_rate == "constqp":
            _level1_la = 0

        # [V6451-CRF0-LOG] 打印 CRF=0 时的 rate_mode 决策日志
        if self.crf == 0:
            _cfg_rate = getattr(self, '_rate_mode', _NVENC_LEVEL1_RATE_MODE)
            if _NVENC_CRF0_FORCE_CONSTQP:
                print(f'   [NVENC] CRF=0 + _NVENC_CRF0_FORCE_CONSTQP=True → '
                      f'强制 CONSTQP qp=0 LA=0（配置 rate_mode={_cfg_rate} 被覆盖）', flush=True)
            else:
                print(f'   [NVENC] CRF=0 + _NVENC_CRF0_FORCE_CONSTQP=False → '
                      f'rate_mode={_level1_rate} qp={_level1_qp} LA={_level1_la}', flush=True)

        # ── Level 1: NVENC SDK GPU 直通编码 ──
        if 'nvenc' in use_codec:
            _level1_preset = self.x264_preset if self.x264_preset else 'p1'
            try:
                _nvenc_encoder = self._get_or_create_nvenc_encoder(
                    W, H, new_fps,
                    preset=_level1_preset,
                    qp=_level1_qp,
                    rate_mode=_level1_rate,
                    la_depth=_level1_la,
                    pipeline_depth=_level1_pd,
                )
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                writer = FFmpegMuxer(
                    output_path, new_fps,
                    audio_src=audio_src,
                    ffmpeg_bin=self.ffmpeg_bin,
                    quiet=self.quiet,
                )
                _nvenc_encoder.set_muxer_ref(writer)  # [FIX-SPS-PPS] 建立 muxer 回引用
                _use_nvenc_direct = True
                _active_level = 1
                if self._is_first_segment():
                    print(f'[NVENCEncoder] Level 1: NVENC GPU 直通编码 ({W}x{H}@{new_fps:.1f}fps)', flush=True)
            except Exception as _nv_err:
                _err_str = str(_nv_err)
                # [V6451-CODEC8-FALLBACK] 严格 preset（slow/slower/veryslow=p6/p7）组合
                # 高帧率/高分辨率时 InitializeEncoder code=8 拒绝。保持 CRF/RC/LA 不变，
                # 仅降级到更宽松的 p4/medium 重试一次，仍留在 SDK Level 1。
                _p_idx = _PRESET_P_INDEX.get(_level1_preset, 4)
                if ('InitializeEncoder failed, code=8' in _err_str
                        and _p_idx >= 5
                        and _active_level == 4):  # 尚未成功
                    _fallback_preset = 'p4'
                    try:
                        if self._is_first_segment():
                            print(f'[NVENCEncoder] 严格 preset p{_p_idx+1} 初始化失败 (code=8)，'
                                  f'降级到 {_fallback_preset} 重试 (保持 RC={_level1_rate} LA={_level1_la})',
                                  flush=True)
                        _nvenc_encoder = self._get_or_create_nvenc_encoder(
                            W, H, new_fps,
                            preset=_fallback_preset,
                            qp=_level1_qp,
                            rate_mode=_level1_rate,
                            la_depth=_level1_la,
                            pipeline_depth=_level1_pd,
                        )
                        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                        writer = FFmpegMuxer(
                            output_path, new_fps,
                            audio_src=audio_src,
                            ffmpeg_bin=self.ffmpeg_bin,
                            quiet=self.quiet,
                        )
                        _nvenc_encoder.set_muxer_ref(writer)
                        _use_nvenc_direct = True
                        _active_level = 1
                        if self._is_first_segment():
                            print(f'[NVENCEncoder] Level 1: NVENC GPU 直通编码 ({W}x{H}@{new_fps:.1f}fps) '
                                  f'preset={_fallback_preset}', flush=True)
                    except Exception as _retry_err:
                        if self._is_first_segment():
                            print(f'[NVENCEncoder] Level 1 降级重试仍失败: {_retry_err}', flush=True)
                        # [FIX-PRESET-DEGRADE-CACHE] 降级也失败：清空 cache，防止
                        # 后续段同参数命中返回半初始化失败 encoder（隐患 B）。
                        self._cached_nvenc_encoder = None
                        self._cached_nvenc_key = None
                else:
                    if self._is_first_segment():
                        print(f'[NVENCEncoder] Level 1 失败: {_nv_err}', flush=True)
                    # [FIX-PRESET-DEGRADE-CACHE] 非降级路径（p1-p4 失败或非 code=8）：
                    # 同样清空 cache，防止后续段复用失败 encoder。
                    self._cached_nvenc_encoder = None
                    self._cached_nvenc_key = None

        # [DIAG] 记录最终激活的编码级别，供 benchmark 诊断使用
        self._diag_active_level = _active_level

        # ── Level 2/3: Pinned Ring Buffer + FFmpegWriter ──
        if _active_level == 4:
            _ring_max_frames = effective_bs * (n_interp + 1)  # B × (T+original)
            _ring_slot_bytes = _ring_max_frames * H * W * 3
            _ring_num_slots = min(8, max(2,
                (4 * 1024 * 1024 * 1024) // _ring_slot_bytes if _ring_slot_bytes > 0 else 4
            ))
            try:
                self._ring_buf, _ring_created = self._get_or_create_ring_buf(
                    num_slots=_ring_num_slots,
                    max_frames_per_slot=_ring_max_frames,
                    H=H, W=W,
                )
                _lv = 2 if 'nvenc' in use_codec else 3
                _active_level = _lv
                if _ring_created and not self.quiet:
                    print(f'[NVENCEncoder] Level {_lv}: Ring Buffer + '
                          f'{"NVENC" if _lv == 2 else "软编码"} ({_ring_num_slots} slots × {_ring_max_frames} frames)', flush=True)
            except Exception as _rb_err:
                if self._is_first_segment():
                    print(f'[NVENCEncoder] Level 2/3 (Ring Buffer) 初始化失败: {_rb_err}', flush=True)

        # ── Level 4: 标准路径 ──
        if _active_level == 4:
            if self._is_first_segment():
                print(f'[NVENCEncoder] Level 4: 标准 PinnedResultPool 路径', flush=True)

        if not _use_nvenc_direct:
            writer = FFmpegWriter(
                output_path, W, H, new_fps,
                codec            = use_codec,
                extra_codec_args = use_extra,
                crf              = self.crf,
                preset           = self.x264_preset,
                audio_src        = audio_src,
                ffmpeg_bin       = self.ffmpeg_bin,
                quiet            = self.quiet,
                rc_mode          = _level1_rate,
            )
        # [FIX-NVENC-AWARE] 保存实际使用的编码器，供段后诊断代码使用
        self._last_used_codec = use_codec

        frame_count  = 0
        output_count = 0
        meter        = ThroughputMeter(window=20)
        desc         = f'[{worker_label}] 插帧'
        pbar = tqdm(total=n_seg_est, unit='帧', desc=desc,
                    dynamic_ncols=True) if HAS_TQDM else None

        # ── 读取第一帧 ────────────────────────────────────────────────────────
        pair = reader.read()
        if pair is None:
            print(f'[{worker_label}] 无法读取首帧')
            reader.close(); writer.close()
            if pbar: pbar.close()
            return False, 0, 0
        first, first_padded = pair

        # [FIX-F0-MISSING] 首帧 f0 必须显式编码并写入。CE pipeline 的 encode_order
        # 仅包含插值帧+img1（右帧），f0 从未被列入 — 不在此处写入则永久丢失。
        # encode_frames_batch / ce_pipeline 已改为全局 _frame_idx slot 分配，
        # 与 encode_frame 的 slot 0 使用不冲突（f0→slot0, 后续→slot1+）。
        if not skip_first_output:
            if _nvenc_encoder is not None:
                first_gpu = torch.from_numpy(first).cuda()
                first_nv12 = _rgb_to_nv12_gpu(first_gpu, input_is_bgr=False)
                force_idr_f0 = not self._is_first_segment()
                if _nvenc_encoder._la_depth > 0:
                    # [FIX-F0-IN-BATCH] LA>0 时 encode_frame 返回 0 bytes（LA 缓冲），
                    # 将 f0 NV12 tensor 暂存 encoder，由编码线程插入累积 batch 开头。
                    _nvenc_encoder._pending_f0_nv12 = first_nv12
                    _nvenc_encoder._pending_f0_force_idr = force_idr_f0
                else:
                    # LA=0: 无缓冲延迟，直接 encode_frame 即可返回数据
                    h264_data = _nvenc_encoder.encode_frame(first_nv12, force_idr=force_idr_f0)
                    writer.write(h264_data)
                    output_count += 1
            else:
                writer.write(first)
                output_count += 1

        frame_count = 1
        if pbar:
            pbar.update(1)

        # ── 主处理 ────────────────────────────────────────────────────────────
        preview_interrupted = False
        if self.device.type == 'cuda':
            pipeline = IFRNetPipelineRunner(
                self,
                auto_tune    = True,
                codec        = use_codec,
                x264_preset  = self.x264_preset,
                crf          = self.crf,
                t2_cache_dir = self.t2_cache_dir,
                pair_queue_override   = pair_queue_override,
                result_queue_override = result_queue_override,
                t3_fps_measured       = t3_fps_measured,   # [FIX-T3-FPS]
            )
            try:
                fc_extra, oc_extra = pipeline.run(
                    reader            = reader,
                    writer            = writer,
                    timesteps         = timesteps,
                    H                 = H,
                    W                 = W,
                    effective_bs      = effective_bs,
                    first_raw         = first,
                    first_padded      = first_padded,
                    skip_first_output = skip_first_output,
                    pbar              = pbar,
                    n_seg_est         = n_seg_est,
                    meter             = meter,
                    H_pad             = H + pad_h,
                    W_pad             = W + pad_w,
                    nvenc_encoder     = _nvenc_encoder,  # [NVENCEncoder]
                )
            except Exception as e:
                print(f'[{worker_label}] 流水线异常: {e}', flush=True)
                reader.close()
                writer.close()
                if pbar: pbar.close()
                return False, 0, 0
            if _nvenc_encoder:
                # GPU_RAW 路径 (Level 1) 下 _NVENCEncodeThread 已在 flush_and_join()
                # 中执行过 NVENC EOS flush，此处无需重复 flush。
                # 非 GPU_RAW 路径 (Level 2/3/4, _enc_thread=None) 才需在此 flush。
                if _active_level != 1:
                    _leftover = _nvenc_encoder.flush()
                    if _leftover:
                        writer.write(_leftover)
                # [SEGMENT-REUSE] 编码器跨段复用，仅 flush 不 close；由 cleanup() 统一销毁
            # [GPU-MONITOR-v2] 保存实际队列深度，供 print_report() 调优建议使用
            self._last_pair_q_size   = pipeline.pair_queue.maxsize
            self._last_result_q_size = pipeline.result_queue.maxsize
            # [FIX-T3-MEMCAP] 记录每个 result slot 的 MiB，供 get_queue_suggestions 约束
            _max_BT = effective_bs * len(timesteps)
            self._last_pool_slot_mb = (
                _max_BT * (H + pad_h) * (W + pad_w) * 3 / 1e6
            )
            # [FIX-MAXPQ-DYNAMIC] 记录每个 pair slot 的 MiB（raw+2×pad 共 3 帧），
            # 供 ADAPTIVE-QUEUE 决策块的 _compute_max_pair_queue 内存约束使用
            self._last_pair_slot_mb = (
                effective_bs * 3 * (H + pad_h) * (W + pad_w) * 3 / 1e6
            )
            # [FIX-T3-FPS] 保存实测 T3 fps + 编码分辨率，供 process_video 报告使用
            self._last_t3_fps_measured = getattr(pipeline, '_t3_fps_measured', 0.0)
            self._last_encode_hw = (H, W)
            # 诊断计数器：reader/infer/writer 三级帧数追踪（供 benchmark 交叉验证）
            self._diag_reader_pairs  = getattr(pipeline, '_diag_reader_pairs', 0)
            self._diag_infer_batches = getattr(pipeline, '_diag_infer_batches', 0)
            self._diag_infer_pairs   = getattr(pipeline, '_diag_infer_pairs', 0)
            self._diag_writer_frames = getattr(pipeline, '_written', 0)
            self._diag_gpu_stay_batches = getattr(pipeline, '_diag_gpu_stay_batches', 0)
            self._diag_nvenc_frames   = getattr(pipeline, '_diag_nvenc_frames', 0)
            self._diag_empty_h264     = getattr(pipeline, '_diag_empty_h264', 0)
            self._diag_external_first_frame = 1  # f0 不在 pipeline._written 计数内（CE pipeline 首帧由 first_raw 传入但未计入 _written）
            frame_count  += fc_extra
            # [FIX-LA-COUNT] 使用 Writer 线程的实际写入帧数，而非推理线程的理论 oc_extra。
            # NVENC LA 缓冲消耗的帧不会写入 muxer（h264_data==b"" 不计入 written），
            # Writer 线程已正确排除。pipeline._written 在 writer thread join 后可用。
            if _nvenc_encoder is not None:
                _actual_written = getattr(pipeline, '_written', 0)
                # [FIX-MUXER-COUNT] Level 1 GPU_RAW 路径以 muxer 实际写入帧数为
                # 最高优先级：writer/编码线程异常时 pipeline._written 可能为 0 或
                # 只反映提交数，FFmpegMuxer._write_count 记录真正写入 stdin 的帧数，
                # 是输出文件实际帧数的直接度量。
                _muxer_written = getattr(writer, '_write_count', 0) if _active_level == 1 else 0
                if _muxer_written > 0:
                    output_count += _muxer_written
                elif _actual_written > 0:
                    output_count += _actual_written  # [FIX-F0] 累加(含f0)，不再替换
                else:
                    output_count += oc_extra
            else:
                output_count += oc_extra
            if n_seg_est > 0:
                _shortfall = n_seg_est - frame_count
                if _shortfall > 1:
                    print(
                        f'[{worker_label}] ⚠️ 提前EOF！预期 {n_seg_est} 帧，实际读取 {frame_count} 帧 '
                        f'（缺失 {_shortfall} 帧，{_shortfall/n_seg_est*100:.1f}%）',
                        flush=True,
                    )
        else:
            # 同步回退路径
            preview_interrupted = False
            padded_buf = [first_padded]
            raw_buf    = [first]

            def flush_buf():
                nonlocal output_count
                if len(raw_buf) < 2:
                    return
                n_pairs = len(raw_buf) - 1
                results = self._safe_infer(padded_buf[:-1], padded_buf[1:], timesteps, H, W)
                for i, interps in enumerate(results):
                    for interp_frame in interps:
                        writer.write(interp_frame)
                        output_count += 1
                    writer.write(raw_buf[i + 1])
                    output_count += 1
                meter.update(n_pairs)

            while True:
                pair = reader.read()
                if pair is None:
                    break
                frame, frame_padded = pair
                frame_count += 1
                raw_buf.append(frame)
                padded_buf.append(frame_padded)
                if len(raw_buf) == effective_bs + 1:
                    flush_buf()
                    raw_buf    = [raw_buf[-1]]
                    padded_buf = [padded_buf[-1]]
                if pbar:
                    avg_t = np.mean(self._timing[-20:]) * 1000 if self._timing else 0
                    pbar.set_postfix(
                        fps=f'{meter.fps():.1f}',
                        eta=f'{meter.eta(n_seg_est):.0f}s',
                        ms=f'{avg_t:.0f}',
                        refresh=False,
                    )
                    pbar.update(1)
                if preview and frame_count % preview_interval == 0:
                    import cv2
                    cv2.imshow(f'IFRNet Preview [{worker_label}]', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        preview_interrupted = True
                        break
            if len(raw_buf) > 1:
                flush_buf()

        # ── 收尾 ──────────────────────────────────────────────────────────────
        if pbar:
            pbar.close()
        writer.close()
        reader.close()

        if n_seg_est > 0:
            _shortfall = n_seg_est - frame_count
            if _shortfall > 1:
                print(
                    f'[{worker_label}] ⚠️ 提前EOF！预期 {n_seg_est} 帧，实际读取 {frame_count} 帧 '
                    f'（缺失 {_shortfall} 帧，{_shortfall/n_seg_est*100:.1f}%）',
                    flush=True,
                )

        elapsed = time.time() - t_start
        # 期望帧数 = 插值理论值: (fc-1) 对 插值产生 scale 倍, 加首帧 = (fc-1)*scale + 1
        _nvenc = getattr(self, '_cached_nvenc_encoder', None)
        _la_depth = getattr(_nvenc, '_la_depth', 0) if _nvenc is not None else 0
        _expected_out = (frame_count - 1) * scale + 1  # 帧数守恒: 输出=输入*Scale-(Scale-1)
        _out_loss = _expected_out - output_count
        if _out_loss == 0:
            _out_disp = f'{output_count}'
        else:
            _out_disp = f'{output_count} (期望≈{_expected_out}, 差{_out_loss} ⚠️)'
        print(f'[{worker_label}] 完成 | 原始帧={frame_count} → 输出帧={_out_disp} | '
              f'{frame_count/elapsed:.1f} 原始帧/s（含 warmup/初始化）')
        if preview_interrupted:
            print(f'[{worker_label}] ⚠️  用户按 q 提前退出预览，输出不完整')
            return False, 0, 0
        # [FIX-FRAME-CONSERVATION] 帧数守恒硬校验（Level 1 GPU_RAW 路径）：
        # 实际写入与期望差异 >2 帧说明发生了截断/丢帧（如 muxer 被提前关闭、
        # 编码线程异常），必须判段失败，防止截断文件带着 checkpoint 流入
        # 后续阶段。容差 2 帧豁免既有 f0 计数口径差。
        if _active_level == 1 and _out_loss > 2:
            print(f'[{worker_label}] ❌ 输出帧数严重不足: 实际 {output_count} vs '
                  f'期望 {_expected_out} (缺 {_out_loss} 帧)，段处理失败', flush=True)
            return False, frame_count, output_count
        return True, frame_count, output_count

    # ── 对外公开接口 ──────────────────────────────────────────────────────────

    def process_video(
        self,
        input_path:       str,
        output_path:      str,
        scale:            float = 2.0,
        preview:          bool  = False,
        preview_interval: int   = 30,
        total_segments:   int   = 1,       # [SEGMENT-REUSE] 总分段数
        segment_index:    int   = 1,       # [SEGMENT-REUSE] 当前分段序号（1-based）
    ) -> bool:
        if not os.path.exists(input_path):
            print(f'错误: 输入不存在 - {input_path}')
            return False
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        self._segment_index  = segment_index
        self._total_segments = total_segments
        _is_first = (segment_index == 1)
        _is_last  = (segment_index >= total_segments)

        audio_src = input_path if self.keep_audio else None

        # if self.use_tensorrt:
        if self.use_tensorrt and not self._trt_built:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            meta = _probe_video(input_path)
            _trt_ceil = lambda x, s: x if x % s == 0 else x + (s - x % s)
            _trt_H    = _trt_ceil(meta['height'], MODEL_STRIDE)
            _trt_W    = _trt_ceil(meta['width'],  MODEL_STRIDE)
            sh        = (self.batch_size, 3, _trt_H, _trt_W)
            trt_dir   = self.trt_cache_dir or os.path.join(base_dir, '.trt_cache')
            self._build_trt_engine(sh, trt_dir)
            # 无论成功或失败，标记已尝试，避免重复构建
            self._trt_built = True

        # [GPU-MONITOR] 启动后台监测
        self._gpu_monitor.start()

        ok, fc, oc = self._process_segment(
            input_path, output_path, scale,
            frame_start=0, frame_end=-1,
            skip_first_output=False,
            audio_src=audio_src,
            worker_label='GPU0',
            preview=preview,
            preview_interval=preview_interval,
            # 传入跨段自适应建议
            pair_queue_override=self._next_pair_queue,
            result_queue_override=self._next_result_queue,
            t3_fps_measured=self._next_t3_fps_measured,   # [FIX-T3-FPS]
        )

        # [GPU-MONITOR-v2] 停止采样，输出精细统计 + 三项调优建议
        # [SEGMENT-REUSE] 仅最后一段输出完整报告，中间段静默
        self._gpu_monitor.stop()
        _gpu_stats = self._gpu_monitor.get_stats()
        _verbose_report = _is_last or total_segments <= 1
        if _gpu_stats.sample_count > 0:
            _cur_pair_q   = getattr(self, '_last_pair_q_size',   4)
            _cur_result_q = getattr(self, '_last_result_q_size', 16)
            _slot_mb = getattr(self, '_last_pool_slot_mb', 0.0)
            if _verbose_report:
                print()
                self._gpu_monitor.print_report(
                    _gpu_stats,
                    current_bs       = self.batch_size,
                    current_pair_q   = _cur_pair_q,
                    current_result_q = _cur_result_q,
                    codec            = getattr(self, '_last_used_codec', self.codec),
                    slot_mb          = _slot_mb,   # [FIX-MAXRQ-DYNAMIC] 传入以应用内存上限约束
                )
            # [FIX-T3-DETECT] 获取 GPU-MONITOR 的队列建议（含 T3-bottleneck 检测）
            pair_gpu_sug, result_gpu_sug = self._gpu_monitor.get_queue_suggestions(
                _gpu_stats, _cur_pair_q, _cur_result_q,
                slot_mb=_slot_mb,           # 传入每 slot 大小，用于 PinnedPool 内存约束
                codec=getattr(self, '_last_used_codec', self.codec),           # [FIX-NVENC-AWARE] 使用实际编码器
            )
            # 获取 AUTO-TUNE-RETUNE 的建议（如果存在）
            retune_pair_q   = getattr(self, '_retune_pair_q',   None)
            retune_result_q = getattr(self, '_retune_result_q', None)

            # [FIX-T3-DETECT] 先检测是否 T3-bottleneck，再决定综合策略
            _is_t3 = GPUMonitor._is_t3_bottleneck(_gpu_stats, codec=getattr(self, '_last_used_codec', self.codec))
            if _is_t3:
                # T3 是真正瓶颈：不增大队列，result_queue 可适当缩小以回收 pinned 内存
                final_pair_q   = _cur_pair_q
                final_result_q = max(16, _cur_result_q - 8) if _cur_result_q > 16 else _cur_result_q
                if _verbose_report:
                    print(
                        f'[ADAPTIVE-QUEUE] ⚠️  T3-bottleneck 确认（编码器是瓶颈）：'
                        f'pair_queue={final_pair_q}（不变）'
                        f' result_queue={_cur_result_q}->{final_result_q}（适当缩小，回收锁页内存）'
                    )
                # [FIX-T3-REPORT] 增强诊断：实测 vs 理论 T3 fps + 具体编码建议
                _t3_fps_meas = getattr(self, '_last_t3_fps_measured', 0.0)
                _H_enc, _W_enc = getattr(self, '_last_encode_hw', (0, 0))
                _nvenc_already_active = 'nvenc' in getattr(self, '_last_used_codec', self.codec).lower()
                _t3_fps_est = 0.0
                # [FIX-NVENC-AWARE] _software_encode_fps 估算的是 x264 软编码速度，
                # 对 NVENC 硬编码毫无意义；当 NVENC 已激活时跳过此估算。
                if (not _nvenc_already_active
                        and _H_enc > 0 and _W_enc > 0):
                    _t3_fps_est = _software_encode_fps(
                        os.cpu_count() or 4, _H_enc, _W_enc,
                        self.codec, self.x264_preset, self.crf,
                    )
                if _verbose_report:
                    _diag_parts = []
                    if _t3_fps_meas > 0:
                        _diag_parts.append(f'实测 T3={_t3_fps_meas:.0f} fps')
                    if _t3_fps_est > 0:
                        _diag_parts.append(f'理论估算={_t3_fps_est:.0f} fps')
                    if _t3_fps_meas > 0 and _t3_fps_est > 0:
                        _degrade = _t3_fps_est / max(_t3_fps_meas, 1.0)
                        _diag_parts.append(f'偏差={_degrade:.1f}×（含热节流因素）')
                    _diag_str = '  [' + '  '.join(_diag_parts) + ']' if _diag_parts else ''
                    _has_nvenc_h264 = HardwareCapability.has_nvenc('h264_nvenc')
                    if _nvenc_already_active:
                        _encoder_tip = (
                            f'NVENC 已激活但 T3 仍为瓶颈（实测 {_t3_fps_meas:.0f} fps）。'
                            f'建议：1) 尝试 --x264-preset p1（最快 NVENC preset） '
                            f'2) 尝试 --crf 0 无损模式（跳过 VBR 前向预看） '
                            f'3) rgb24→yuv420p CPU 格式转换 / pipe 写入带宽 / 非标准分辨率'
                        )
                    elif _has_nvenc_h264 and _t3_fps_meas > 0:
                        _nvenc_fps = 3000.0
                        if _H_enc > 0 and _W_enc > 0:
                            _nvenc_fps = min(3000.0, 3000.0 * 1920 * 1080 / (_H_enc * _W_enc))
                        _speedup = _nvenc_fps / max(_t3_fps_meas, 1.0)
                        _encoder_tip = (
                            f'建议切换 --codec h264_nvenc（理论 ~{_nvenc_fps:.0f} fps，'
                            f'约 {_speedup:.0f}× 加速）'
                            f'；注: Docker 环境需确认 NVENC 设备映射（--gpus）'
                        )
                    elif _has_nvenc_h264:
                        _encoder_tip = (
                            '建议切换 --codec h264_nvenc（NVENC 约 10-20× 加速）'
                        )
                    else:
                        _encoder_tip = (
                            '考虑降低编码参数：--x264-preset veryfast --crf 18'
                            '（实测约 5-10× 加速，画质略降但通常可接受）'
                        )
                    print(f'[ADAPTIVE-QUEUE] 提示：真正瓶颈在编码器{_diag_str}  {_encoder_tip}')
                # [FIX-T3-FPS] 保存实测 T3 fps 供下段使用
                self._next_t3_fps_measured = _t3_fps_meas
            else:
                # 正常路径：综合 GPU-MONITOR 和 RETUNE 两方建议（双向调优版）
                # [FIX-ADAPTIVE-BIDIR] 三项改造（根治单向只增不减 + 均值半吊子调优）：
                #   1) 校准信号去冲突：按 GPU 稳定段 σ 加权合并 result 建议——
                #      σ>25%（不稳定）取大（信任 GPU-MONITOR 扩容建议）；
                #      σ≤15%（稳定）取小（信任 RETUNE 收缩建议）；中间区取均值。
                #   2) R 带迟滞收缩：合并建议连续 2 段低于当前值才允许收缩，
                #      步长 ≤25%/段，下限 8，避免单段抖动引发队列震荡。
                #   3) P 跟随 R：P = clamp(max(R//2, P建议), 8, 动态上限)，
                #      保持输入/输出缓冲对称；P 不再 max(..., current)，可随 R 下降。
                _sigma = getattr(_gpu_stats, 'stable_std', 0.0) or 0.0
                _rq_retune = retune_result_q if retune_result_q is not None else None
                if _rq_retune is None:
                    _rq_merged    = result_gpu_sug
                    _merge_reason = 'GPU-MONITOR 唯一来源'
                elif _sigma > 25.0:
                    _rq_merged    = max(result_gpu_sug, _rq_retune)
                    _merge_reason = f'σ={_sigma:.0f}%>25%，取大（信任 GPU-MONITOR）'
                elif _sigma <= 15.0:
                    _rq_merged    = min(result_gpu_sug, _rq_retune)
                    _merge_reason = f'σ={_sigma:.0f}%≤15%，取小（信任 RETUNE）'
                else:
                    _rq_merged    = (result_gpu_sug + _rq_retune) // 2
                    _merge_reason = f'σ={_sigma:.0f}% 中间区，取均值'

                # R 迟滞收缩：连续 2 段建议低于当前才收缩，步长 ≤25%，下限 8
                _shrink_streak = getattr(self, '_rq_shrink_streak', 0)
                if _rq_merged < _cur_result_q:
                    _shrink_streak += 1
                    if _shrink_streak >= 2:
                        _rq_floor      = max(8, int(_cur_result_q * 0.75))
                        final_result_q = max(_rq_merged, _rq_floor)
                        _shrink_note   = f'连续{_shrink_streak}段收缩建议，R 下降（步长≤25%）'
                    else:
                        final_result_q = _cur_result_q
                        _shrink_note   = f'收缩建议第{_shrink_streak}段，迟滞保持'
                else:
                    _shrink_streak = 0
                    final_result_q = _rq_merged
                    _shrink_note   = 'R 维持/增大'
                self._rq_shrink_streak = _shrink_streak
                _rq_combined_raw = _rq_merged   # 供下方推导链路打印复用

                # P 跟随 R：对称缓冲；叠加 P 侧建议（σ 大/空闲高时 GPU-MONITOR 建议加深）
                _pq_sug      = max(pair_gpu_sug,
                                   retune_pair_q if retune_pair_q is not None else 0)
                final_pair_q = max(final_result_q // 2, _pq_sug)

                # 硬上限（[FIX-MAXPQ-DYNAMIC] pair 上限改为动态两轴函数，替换原硬编码 8）
                _pair_slot_mb_aq = getattr(self, '_last_pair_slot_mb', 0.0)
                if _pair_slot_mb_aq > 0.0:
                    _pair_cap_aq = _compute_max_pair_queue(
                        slot_mb      = _pair_slot_mb_aq,
                        mem_avail_gb = _detect_encode_parallelism()['mem_avail_gb'],
                    )
                else:
                    _pair_cap_aq = _PAIR_Q_ABS_CAP
                final_pair_q   = max(8, min(final_pair_q, _pair_cap_aq))
                final_result_q = min(final_result_q, 64)
                # [FIX-T3-MEMCAP / FIX-POOL-AUTOSCALE / FIX-MAXRQ-DYNAMIC]
                # PinnedPool 内存上限约束：改用三轴动态函数，并显式 log 截断原因。
                if _slot_mb > 0.0:
                    _mem_avail_gb_aq = _detect_encode_parallelism()['mem_avail_gb']
                    _max_rq_mem = _compute_max_result_queue(
                        slot_mb      = _slot_mb,
                        mem_avail_gb = _mem_avail_gb_aq,
                    )
                    if final_result_q > _max_rq_mem:
                        if _verbose_report:
                            _ram_budget_mb = _mem_avail_gb_aq * 1024.0 * 0.06
                            print(
                                f'[ADAPTIVE-QUEUE] PinnedPool 动态上限截断: '
                                f'result_queue {final_result_q} → {_max_rq_mem}'
                                f'  (slot={_slot_mb:.1f} MB × {_max_rq_mem}'
                                f' ≈ {_slot_mb * _max_rq_mem:.0f} MB'
                                f'  ≤ RAM预算 {_ram_budget_mb:.0f} MB'
                                f'  [mem_avail={_mem_avail_gb_aq:.1f} GB × 6%])'
                            )
                    final_result_q = min(final_result_q, _max_rq_mem)

                # [FIX-RETUNE-DISPLAY] 打印推导链路，让建议来源透明可追溯
                _retune_str = (f'RETUNE={_rq_retune}' if retune_result_q is not None
                               else 'RETUNE=N/A')
                if _verbose_report:
                    print(
                        f'[ADAPTIVE-QUEUE] 下次将使用 pair_queue={final_pair_q} '
                        f'result_queue={final_result_q}'
                        f'  (GPU-MONITOR={result_gpu_sug} {_retune_str} → 合并={_rq_combined_raw}'
                        f'  [{_merge_reason}；{_shrink_note}])'
                    )
                # [FIX-T3-FPS] 非 T3-bottleneck 时也更新实测 T3 fps（更可靠）
                self._next_t3_fps_measured = getattr(self, '_last_t3_fps_measured', 0.0)

            self._next_pair_queue   = final_pair_q
            self._next_result_queue = final_result_q
            # [SEGMENT-REUSE] 非末段输出简短队列调整摘要
            if not _verbose_report:
                print(f'[ADAPTIVE-QUEUE] 段{segment_index}→段{segment_index+1}: '
                      f'pair_queue={_cur_pair_q}→{final_pair_q} '
                      f'result_queue={_cur_result_q}→{final_result_q}')
        else:
            print('[GPU-MONITOR] 警告：未能获取任何 GPU 采样数据，'
                  '请检查 nvidia-ml-py 安装或驱动状态。')

        if ok:
            self._print_summary(input_path, output_path, fc, oc, scale)
            self._dump_report(input_path, output_path, fc, oc, scale)
        return ok

    def _print_summary(self, input_path, output_path, fc, oc, scale):
        print(f'\n✅ 插帧完成！')
        if oc > 0:
            # 期望帧数 = 插值理论值: (fc-1) 对 插值产生 scale 倍, 加首帧 = (fc-1)*scale + 1
            _nvenc = getattr(self, '_cached_nvenc_encoder', None)
            _la_depth = getattr(_nvenc, '_la_depth', 0) if _nvenc is not None else 0
            _expected = (fc - 1) * scale + 1  # 帧数守恒: 输出=输入*Scale-(Scale-1)
            _loss = _expected - oc
            if _loss == 0:
                print(f'   原始帧: {fc} → 输出帧: {oc} (×{scale:.1f}, LA={_la_depth})')
            else:
                print(f'   原始帧: {fc} → 输出帧: {oc} '
                      f'(期望≈{_expected}, 差 {_loss} 帧 ⚠️)')
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f'   输出: {output_path} ({size_mb:.1f} MB)')

    def _dump_report(self, input_path, output_path, fc, oc, scale):
        if not self.report_json or not self._timing:
            return
        report = {
            'input':      input_path,
            'output':     output_path,
            'scale':      scale,
            'batch_size': self.batch_size,
            'fp16':       self.use_fp16,
            'cuda_graph': self.use_cuda_graph,
            'tensorrt':   getattr(self, '_trt_ok', False),
            'nvdec':      HardwareCapability.has_nvdec(),
            'nvenc':      HardwareCapability.best_encoder(self.codec).endswith('nvenc'),
            'n_workers':  1,
            'frame_count':  fc,
            'output_count': oc,
            'infer_latency_ms': {
                'mean': round(float(np.mean(self._timing)) * 1000, 2),
                'p95':  round(float(np.percentile(self._timing, 95)) * 1000, 2),
                'max':  round(float(np.max(self._timing)) * 1000, 2),
            },
        }
        with open(self.report_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f'   性能报告: {self.report_json}')


# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='IFRNet 视频插帧 —— 终极优化版 v6.4.5（单卡版）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 基础参数
    parser.add_argument('--input',      required=True,  help='输入视频路径')
    parser.add_argument('--output',     required=True,  help='输出视频路径')
    parser.add_argument('--scale',      type=float, default=2.0, help='插帧倍数（≥2 整数）')
    parser.add_argument('--model',      default='IFRNet_S_Vimeo90K', help='模型名称或 .pth 路径')
    parser.add_argument('--device',     default='cuda', choices=['cuda', 'cpu'])
    # [BATCH-UP] 默认 48
    parser.add_argument('--batch-size', type=int, default=48,
                        help='每批帧对数（默认 48，TRT 用户首次运行需重建 Engine）')
    # 推理优化
    parser.add_argument('--no-fp16',       action='store_true', help='禁用 FP16')
    parser.add_argument('--no-compile',    action='store_true', help='禁用 torch.compile')
    parser.add_argument('--no-cuda-graph', action='store_true', help='禁用 CUDA Graph')
    parser.add_argument('--use-tensorrt',  action='store_true',
                        help='启用 TensorRT 加速（首次需构建 Engine）')
    # 高优先级覆盖参数
    parser.add_argument('--use-cuda-graph', dest='use_cuda_graph_force',
                        action='store_true', default=False,
                        help='[覆盖] 强制启用 CUDA Graph，覆盖 --no-cuda-graph')
    parser.add_argument('--use-compile', dest='use_compile_force',
                        action='store_true', default=False,
                        help='[覆盖] 强制启用 torch.compile，覆盖 --no-compile')
    parser.add_argument('--no-tensorrt', dest='no_tensorrt',
                        action='store_true', default=False,
                        help='[覆盖] 强制禁用 TensorRT，覆盖 --use-tensorrt')
    # 硬件加速
    parser.add_argument('--no-hwaccel', action='store_true', help='强制禁用 NVDEC')
    # 编码参数
    parser.add_argument('--codec',       default='libx264')
    parser.add_argument('--crf',         type=int, default=23)
    parser.add_argument('--x264-preset', type=str, default='medium',
                        choices=['ultrafast','superfast','veryfast','faster','fast',
                                 'medium','slow','slower','veryslow'])
    parser.add_argument('--no-audio',    action='store_true')
    parser.add_argument('--ffmpeg-bin',  type=str, default='ffmpeg')
    # 调试
    parser.add_argument('--preview',           action='store_true')
    parser.add_argument('--preview-interval',  type=int, default=30)
    parser.add_argument('--report',            default=None, help='JSON 性能报告路径')
    parser.add_argument('--quiet', action=argparse.BooleanOptionalAction, default=True,
                        help='静默模式（默认开启），仅显示关键信息；--no-quiet 开启详细日志')
    parser.add_argument('--trt-cache-dir',   default=None)
    parser.add_argument('--t2-cache-dir',    default=None)

    args = parser.parse_args()

    # ── 高优先级覆盖参数解析 ──────────────────────────────────────────────────
    _cli_overrides: list = []

    if args.no_tensorrt and args.use_tensorrt:
        args.use_tensorrt = False
        _cli_overrides.append('--no-tensorrt  覆盖了  --use-tensorrt  → TensorRT 已禁用')

    if args.use_compile_force and args.no_compile:
        args.no_compile = False
        _cli_overrides.append('--use-compile  覆盖了  --no-compile  → torch.compile 已启用')

    if args.use_cuda_graph_force and args.no_cuda_graph:
        args.no_cuda_graph = False
        _cli_overrides.append('--use-cuda-graph  覆盖了  --no-cuda-graph  → CUDA Graph 已启用')

    _effective_trt     = args.use_tensorrt
    _effective_compile = not args.no_compile
    _effective_cugraph = not args.no_cuda_graph

    if args.use_cuda_graph_force and _effective_compile and not _effective_trt:
        print('[CLI警告] --use-cuda-graph 与 torch.compile 互斥：compile 成功后 CUDA Graph 将被自动禁用。')
    if args.use_cuda_graph_force and _effective_trt:
        print('[CLI警告] --use-cuda-graph 与 --use-tensorrt 互斥：TensorRT 优先。')
    if args.use_compile_force and _effective_trt:
        print('[CLI警告] --use-compile 与 --use-tensorrt 互斥：TensorRT 优先。')

    if _cli_overrides:
        print('[CLI覆盖] 以下设置已被高优先级参数覆盖：')
        for msg in _cli_overrides:
            print(f'          · {msg}')
        print()

    # 模型路径解析
    if args.model in MODEL_NAME_MAP:
        model_path = f'{models_ifrnet}/{MODEL_NAME_MAP[args.model]}'
        model_name = args.model
    else:
        model_path = args.model
        model_name = os.path.splitext(os.path.basename(args.model))[0]   # ✅ 自定义路径取 basename，防止斜杠污染 TRT/T2 缓存文件名
    if not os.path.exists(model_path):
        print(f'错误: 模型不存在 - {model_path}')
        sys.exit(1)

    global Model
    Model, _ = _load_ifrnet_module(args.model)

    print('=' * 65)
    print('  IFRNet 视频插帧 —— 终极优化版 v6.4.5（单卡版）')
    print('=' * 65)
    print(f'  模型:   {args.model}')
    print(f'  设备:   {args.device} | GPU: '
          f'{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print(f'  FP16:   {not args.no_fp16} | '
          f'Compile: {not args.no_compile} | '
          f'CUDA Graph: {not args.no_cuda_graph} | '
          f'TensorRT: {args.use_tensorrt}')
    print(f'  NVDEC:  {HardwareCapability.has_nvdec() and not args.no_hwaccel} | '
          f'NVENC(h264): {HardwareCapability.has_nvenc("h264_nvenc")} | '
          f'NVENC(hevc): {HardwareCapability.has_nvenc("hevc_nvenc")}')
    _codec_actual = HardwareCapability.best_encoder(args.codec)
    if args.crf == 0:
        # [FIX-LOSSLESS] 提示用户实际使用的无损参数
        if 'nvenc' in _codec_actual:
            _lossless_note = '(-qp 0 无损，常量 QP 模式)'
        elif args.codec == 'libx265':
            _lossless_note = '(-x265-params lossless=1，注意：crf=0 在 x265 中不是无损！)'
        else:
            _lossless_note = '(-qp 0 严格逐像素无损)'
        print(f'  编码器: {args.codec} → 实际: {_codec_actual} | '
              f'CRF: 0 → 无损模式 {_lossless_note} | batch_size: {args.batch_size}')
    else:
        print(f'  编码器: {args.codec} → 实际: {_codec_actual} | '
              f'CRF: {args.crf} | batch_size: {args.batch_size}')
    if args.use_tensorrt:
        _tcd = args.trt_cache_dir or f'(自动: {base_dir}/.trt_cache)'
        print(f'  TRT 缓存: {_tcd}')
    print()

    t_total   = time.time()
    processor = IFRNetVideoProcessor(
        model_path     = model_path,
        device         = args.device,
        batch_size     = args.batch_size,
        max_batch_size = args.batch_size * 4,
        use_fp16       = not args.no_fp16,
        use_compile    = not args.no_compile,
        use_cuda_graph = not args.no_cuda_graph,
        use_tensorrt   = args.use_tensorrt,
        use_hwaccel    = not args.no_hwaccel,
        codec          = args.codec,
        crf            = args.crf,
        x264_preset    = args.x264_preset,
        keep_audio     = not args.no_audio,
        ffmpeg_bin     = args.ffmpeg_bin,
        report_json    = args.report,
        trt_cache_dir  = args.trt_cache_dir,
        t2_cache_dir   = getattr(args, 't2_cache_dir', None),
        model_name     = model_name,   # ✅ 使用规范化后的 model_name（自定义路径时为 basename）
        quiet          = getattr(args, 'quiet', True),
    )

    ok = processor.process_video(
        args.input, args.output,
        scale            = args.scale,
        preview          = args.preview,
        preview_interval = args.preview_interval,
    )

    m, s = divmod(int(time.time() - t_total), 60)
    print(f'\n⏱️  总耗时（含模型加载）: {m}分{s}秒')
    if ok and os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / 1024 / 1024
        print(f'✅ 输出: {args.output} ({size_mb:.1f} MB)')
    else:
        print('❌ 处理失败')
        sys.exit(1)


if __name__ == '__main__':
    main()
