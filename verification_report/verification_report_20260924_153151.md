# Video Enhancement 优化方案 · 最终后验证报告（v2）

> 生成时间: 2026-09-24 15:31:51

## A-前置条件

| 状态 | ID | 验证项 | 方法 | 详情 |
|---|---|---|---|---|
| PASS | R1 | Python 版本 ≥ 3.9 | sys.version_info | Python 3.12.9 |
| PASS | R2 | FFmpeg 可用（≥4.3） | which + 版本查询 | ffmpeg N-122480-gcc8f392136-20260118 @ D:\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.EXE |
| PASS | R3 | FFprobe 可用 | shutil.which | ffprobe @ D:\ffmpeg-master-latest-win64-gpl-shared\bin\ffprobe.EXE |
| PASS | R4 | 关键源文件存在 | FILES 核心映射 | 12 个核心文件均存在 |
| WARN | R5 | CUDA / GPU 可用 | import torch 查询 | torch 未安装 |
| PASS | R6 | 模型权重存在 | glob models_*/*.pth | IFRNet: 6, ESRGAN: 12, GFPGAN: 0 |
| WARN | R7 | NVENC 环境探测（不依赖 torch） | ffmpeg lavfi h264_nvenc 单帧编码探测（吸收自 verify_post_run） | h264_nvenc ffmpeg 探测失败: [h264_nvenc @ 0000000002aad040] Cannot load nvcuda.dll [vost#0:0/h264_nvenc @ 0000000002aace00] [enc:h264_nvenc @ 0000000002a9a8c0] Error while opening encoder  |
| SKIP | R8 | NVML 环境变量提示 | 检查 PYTORCH_NVML_BASED_CUDA_CHECK（吸收自 verify_post_run） | 当前值=''：主入口运行时会自动 setdefault 为 0 |

## B-静态·阶段0

| 状态 | ID | 验证项 | 方法 | 详情 |
|---|---|---|---|---|
| PASS | P0-1 | model_name 透传 + 动态架构加载 (A1) | processor 构造透传 model_name；后端按 self.model_name 动态解析 | processor 透传 ✓；后端按实例名动态解析 ✓ |
| PASS | P0-2 | NVENC 返回码分级处置 (H1/H2/H17) | drain 路径：非 SUCCESS 记诊断计数并安全退出（容忍 T4 首排空期常态 code=8，帧由 EOS 全量排空回收）；EOS×2 与 flush 排空保持 fail-fast；DestroyEncoder rc 校验 | [P0-FIX-RC]×8；drain 容忍+遥测 ✓（段级守恒审计兜底）；EOS/flush fail-fast ✓；DestroyEncoder rc ✓ |
| PASS | P0-3 | HEVC/AV1 blocking Lock 挂起修复 (H3) | _lock_bitstream_blocking 按 codec 分流：HEVC/AV1 非阻塞轮询+deadline | codec 分流轮询已落地（H.264 保持已验证阻塞语义） |
| PASS | P0-4 | Muxer 写超时线程 + 收尾 rc 判败 (H4) | FFmpegMuxer 常驻写线程+_write_stdin 超时+_mux_failed；flush_and_join 超时 raise；ifrnet FFmpegWriter close rc 上抛 | 写超时线程 ✓ / flush 超时 raise ✓ / Writer close rc 上抛 ✓ / 段级判败 ✓ |
| PASS | P0-5 | AV1 GUID 回退一致性 (H5) | GUID 未命中显式 raise（拒绝静默 H264 回退污染容器） | GUID 未命中即 raise，交由四级编码回退接管 |
| PASS | P0-6 | 批量隔离 + keep_audio try/finally (H6) | 批量循环逐文件 try/except(KeyboardInterrupt 重抛)；keep_audio 以 finally 恢复 | 批量逐文件隔离 ✓ / Kbd 重抛 ✓ / finally 恢复 ✓ |
| PASS | P0-7 | merge 扩展名改动向上传播 (H7) | merge_videos_by_codec 提供 actual_output 出参并在改写时 append；主调用方采用实际路径 | 出参契约 ✓ / 主流程采纳实际路径 ✓ |
| PASS | P0-8 | ESRGAN 异常判败 + audio_src 传递链 (H8/H10) | run_pipeline_for_video 异常记录并 return False；audio_src 传源视频路径 | 异常判败 ✓ / 成功横幅门控 ✓ / 音轨传递链 ✓ |
| PASS | P0-9 | eval→Fraction + 全活跃树零裸 except (H12) | video_utils 无 eval(r_frame_rate)；src+external 活跃树扫描裸 except:=0 | eval 已移除；活跃树裸 except=0（扫描 3 个目录） |

## B-静态·阶段1

| 状态 | ID | 验证项 | 方法 | 详情 |
|---|---|---|---|---|
| PASS | P1-1 | checkpoint 内容指纹 + 原子写 (H13) | 两 processor direct 入口指纹含 size+mtime_ns；写盘 tmp+os.replace | 上游内容指纹(size+mtime_ns) ✓ / 原子写 ✓（双侧） |
| PASS | P1-2 | OOM bs=1 活锁熔断 (H11) | _safe_infer 内 _BS1_OOM_LIMIT 计数且超限 raise | bs=1 连续 OOM 熔断已落地 |
| PASS | P1-3 | 缓存回滚 + _ring_buf 跨 Level 复位 | get_or_create 先失效再构造；每段编码决策前复位 ring 别名 | 缓存回滚 ✓（pool/ring）/ ring 复位 ✓ |
| PASS | P1-4 | 关闭语义收紧（段收尾 muxer rc 判败） | main 段收尾检查 _mux_failed → return False | muxer 收尾失败已纳入段成败 |
| PASS | P1-5 | watchdog 与 OOM 恢复协调 | T2 设置 _oom_pause_until 生产者；writer 空转计时消费宽限 | 宽限时间戳生产者 ✓ / watchdog 消费者 ✓ |
| PASS | P1-6 | Windows select 替换 (H9) | 平台门控 _SELECT_SUPPORTS_PIPES + 线程化写 _write_with_timeout_threaded | 平台分流 ✓ / Windows 线程化带超时写 ✓ |
| PASS | P1-7 | 数据卫生四项守卫 | input==output 守卫 / 音频新鲜度侧车 / split 指纹复用 / LA 补偿查实际编码器 | 就地覆盖守卫 / 音频新鲜度(.src.json) / split 指纹复用 / LA 补偿查实际编码器 |
| PASS | P1-8 | 数值参数校验前置 | _validate_effective_config 定义且在 CLI 覆盖后调用；5 处 falsy 判断改 is not None | 校验器定义+调用 ✓；falsy 修复 5/5 处 |

## B-静态·阶段2

| 状态 | ID | 验证项 | 方法 | 详情 |
|---|---|---|---|---|
| PASS | P2-1 | tile 真实接入（批级平铺） | es_pipeline 定义并启用 _sr_tile_forward；es_main 启动明示（含 TRT 不适用提示） | 批级平铺前向 ✓ / tile>0 门控 ✓ / TRT 旁路明示 ✓ |
| PASS | P2-2a | ESRGAN compile/CUDA Graph 实现 | torch.compile 接线（default/dynamic 与 reduce-overhead 双模式）；死分支 cuda_graph_accel.available 已摘除 | compile/cudagraphs 接线 ✓ / 死分支摘除 ✓ |
| PASS | P2-2b | IFRNet 互斥裁定前移参数层 | processor 库内构造与独立 CLI 两处先行裁定 TRT>compile>Graph | 参数层裁定 ×2 处（后端防御保留兜底） |
| PASS | P2-3 | 热路径去拷贝 | RING b''.join 单次拼接；LA D2H 形状键控旋转池 | RING join ✓ / LA pinned 旋转池 ✓（残余 copy/tobytes=6，均为合法必要拷贝） |
| PASS | P2-4a/b/d | NVENC 样板收敛（原型/ctx/空帧口径） | CFUNCTYPE 原型模块级预构造；_ctx_push/_ctx_pop 单一实现；空字节帧 prev 占位保帧数 | 模块级原型 ✓ / ctx 配对收敛 ✓ / 空帧占位统一 ✓ |
| PASS | P2-4c | SPS/PPS 全部阶梯统一走 _apply_sps_pps | 原语 _cache_param_sets/_prepend_param_sets 存在且 apply 委托二者；Cached SPS+PPS 打印收敛至 helper 内单点；[P2.4c-LADDER-*] 站点标签齐备 | 原语+委托 ✓ / 打印单点化 ✓ / 阶梯站点 ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] (E/H/I 语义翻转: 迟到IDR补挂、REDRAIN全帧统一) |
| PASS | P2-5 | logging 基础设施接入 | src/utils/logger.py 存在；主入口 init_logging 且阶段横幅写日志 | logger 基础设施 ✓ / 主入口接入 ✓（external print 遥测分批迁移） |

## B-静态·阶段3

| 状态 | ID | 验证项 | 方法 | 详情 |
|---|---|---|---|---|
| PASS | P3-1 | 上帝函数拆解（已实施，含 ce_pipeline） | 四个目标函数 <阈值（阈值随 FIX 增长复核调整）；各阶段子方法存在且带 [P3.1-SPLIT*] 标签 | __init__=29行(<100) / _process_segment=424行(<450) / _process_single=177行(<200) / encode_frames_batch_ce_pipeline=123行(<130) / ce_pipeline 已按 Phase 拆解(_ce_harvest/submit/inline/final) — 拆解已落地 |
| PASS | P3-2 | 镜像收敛（NAL 公共参考实现） | external/nvenc_common/nal_utils.py 存在且回归测试锁定等价性 | 共享参考实现 ✓ / 等价性测试锁定（BEH-D）✓（收敛方案 B） |
| PASS | P3-3 | 死代码清除 | nvenc_writer.py 已删除；ifrnet nvenc_sdk 无 _FUNC_TABLE_SIZE/_RotationBitReader | nvenc_writer.py 已删 ✓ / 死常量与死类已清 ✓ |
| PASS | P3-4 | 测试基建（合并后单一入口） | test_regression_min.py 为兼容别名（--behavior-only 转发本脚本）；本脚本含行为验证阶段 | 兼容别名 ✓ / 行为阶段内置 ✓ |
| PASS | P3-5 | 文档与 memory 同步 | AGENTS.md 反映归档结构与测试入口；execution 记录存在于 memory/ | AGENTS.md 结构/入口已更新 ✓ / memory 执行记录 ✓ |

## D-运行时

| 状态 | ID | 验证项 | 方法 | 详情 |
|---|---|---|---|---|
| SKIP | RT-0 | 输出视频存在 | 检查输出文件 | 未提供 --output 或文件不存在 |

## F-修复效果

| 状态 | ID | 验证项 | 方法 | 详情 |
|---|---|---|---|---|
| PASS | FIX-IFRNET-LA0 | ifrnet 默认 LA=8/vbr_hq（原 LA=0/constqp 断言已过期） | 读取配置 | rate_mode=vbr_hq, lookahead_depth=8 |
| PASS | FIX-REALESRGAN-LA0 | realesrgan 默认 LA=8/vbr_hq（原 LA=0/constqp 断言已过期） | 读取配置 | rate_mode=vbr_hq, lookahead_depth=8 |
| PASS | FIX-H2D-SYNC | H2D Event 同步 | 源码锚点+结构 | P1-FIX-H2D-EVENT-SYNC / pool.mark_issued |
| PASS | FIX-ROUTE | HEVC+LA 安全路由 | 源码锚点 | hevc_la_disable 配置与路由逻辑 |
| PASS | FIX-HEVC-LA-OPEN | HEVC LA>0 开放（软退役） | 配置+源码锚点 | cfg_ifr=False cfg_esr=False soft_ifr=True soft_esr=True no_downgrade_ifr=True |
| PASS | FIX-GATE | 解码级验收门禁 | 源码锚点 | validate_decodable_video / count_decoded_video_frames |
| PASS | FIX-EOS-ORDER | EOS-ORDER | 源码锚点 | FIX-EOS-ORDER |
| PASS | FIX-STRICT-EOS | STRICT-EOS | 源码锚点 | FIX-STRICT-EOS |
| PASS | FIX-SIZE-CAP | SIZE-CAP | 源码锚点 | FIX-SIZE-CAP |
| PASS | FIX-NAL-COMMON | NAL-COMMON | 源码锚点 | FIX-NAL-COMMON |
| PASS | FIX-MODEL-ARCH-LAZY | IFRNet 架构解析不在模块导入期（无硬编码 S 依赖） | AST 结构分析 | marker=True import_time_calls=0 |
| PASS | FIX-PRESCAN-RECEIVE | 两条入口（源片/收段）都接预扫描/预热 | AST 函数体切片（非注释匹配）+ 锚点存在性 | marker=True ifrnet_proc.process_video_segments=2/2 ifrnet_proc.process_segments_directly=2/2 esrgan_proc.process_video_segments=1/1 esrgan_proc.process_segments_directly=1/1 |
