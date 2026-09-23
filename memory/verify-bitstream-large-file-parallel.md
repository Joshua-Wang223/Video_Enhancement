# 大文件验收效率提升：分片并行可行性分析与 v4 落地方案

> 主题：`tests/verify_segment_bitstream_v3.py` 单文件大视频串行验收耗时 400+ 秒的优化分析
> 结论落地：新建 `tests/verify_segment_bitstream_v4.py`（v3 保持不变）
> 日期：2026-08-19

---

## 1. 背景与痛点

### 1.1 实测性能数据（`Plan/verify_segment_bitstream_test1.log`）

| 场景 | 编码 | 帧数 | 硬件 | 耗时 |
|------|------|------|------|------|
| S01E11_Berry Hunt.mp4 | hevc | 86887 | CPUx8 / 33.3GB（无 GPU） | 449.94s |
| S03E12_Boots' Special Day.mp4 | hevc | 86831 | Tesla T4（GPU） | 398.33s |

两个场景均用 `--skip-chroma` 跳过检查 4，且检查 1/2/3 全部 PASS（frames==packets、IDR=5 连IDR=0、无 pts 异常）。即便如此单文件仍需 400 秒级。

### 1.2 瓶颈定位：检查级依赖分析

`_verify_one_video()` 对单个文件串行执行 4 项检查，按耗时降序：

1. **检查 1+3**（`check_decode_integrity`，line 193）：`ffmpeg -v verbose -f null -vf showinfo` **一次全量解码**，从 stderr 提取 frames + pts 序列。对 hevc 8.6 万帧这是绝对主瓶颈（估算占 400s 中的 ~85%+）。
2. **检查 2**（`extract_annexb_es` + `parse_h264_es`/`parse_mpeg4_es`/`parse_hevc_es` + 统计）：`-c copy` 提取完整 Annex B ES 到内存（`capture_output=True`），再**单线程纯 Python** 逐 NAL/VOP 解析。第二瓶颈。
3. **检查 4**（`check_chroma_corruption`，line 1452）：第二次全量解码（`-f rawvideo`），已分块流式（`_CHROMA_CHUNK_FRAMES=128`）。日志均跳过。

关键事实：现有 `run_verify_parallel`（Thread/ProcessPoolExecutor + GPU 信号量）**只对多个文件并行，对单个大文件无效**——`main()` 在 `single_file` 分支（line 2118）直接串行调用 `_verify_one_video`。

---

## 2. 分片策略对比（用户维度一）

用户提出的「按大小 / 按行 / 按记录」三种通用分片策略，映射到视频码流领域后对应三种切分方式：

### 2.1 按大小分片 → 按字节切 Annex B ES

将提取出的 ES 字节流按固定字节数（如 64MB）切块，各块并行解析。

- **适用场景**：检查 2 的 NAL/VOP 扫描（`bytes.find` 本身是 C 级线性扫描，可分段）。
- **优点**：切分 O(1)，无预处理，块大小可控（内存有界）。
- **缺点**：
  1. **NAL/VOP 跨块边界**——start code 或 NAL payload 可能横跨两个块，需实现「块边界缝合」（保留上一块尾部若干字节）。
  2. `frame_num` 单调性、连 IDR、vop_time 回退都是**全局/跨段性质**，各块独立统计后还需跨块合并，合并逻辑复杂且易错。
  3. 纯 Python 解析本身已用 `_BitReader` 懒加载 + `bytes.find` 优化到 O(ES_size)，单核解析速度已不慢，分片收益被「扫描本身快」稀释。

### 2.2 按行分片 → 按 NAL/VOP 记录切分

先做一次 O(n) 预扫描标出所有 NAL/VOP 边界，再按记录（每个 NAL/VOP）分片并行解析。

- **适用场景**：检查 2 中「每个 NAL/VOP 独立解析」的场景（slice header、VOP header）。
- **优点**：记录边界天然对齐，无跨块问题；负载可按记录数均衡。
- **缺点**：
  1. 预扫描本身是单线程 O(ES_size)，且 `bytes.find` 已是 C 级，预扫描就是当前解析的主体，分片只省「slice header 位解析」部分（`_BitReader` 懒加载已将其降到每 slice 5~10 字节）。
  2. SPS/VOL 与后续 slice 存在**解析依赖**（`frame_num_bits`、`vop_time_increment_bits` 来自 SPS/VOL），记录间不独立，需先串行解析头部再并行解析体。
  3. 统计合并仍受全局性质约束（连 IDR / 回退）。

### 2.3 按记录分片 → 按帧/按时间切分（ffmpeg segment muxer）

用 `split_video_by_time`（`-c copy` + segment muxer，不重编码）把视频切成 N 段，各段并行验收。

- **适用场景**：检查 4（chroma 像素域，逐帧独立、局部性质）。
- **优点**：复用现成工具 `split_video_by_time`；段级独立，天然并行。
- **缺点（致命）**：
  1. **检查 1 语义破坏**：`-c copy` 在关键帧处切分，段边界帧可能重复/丢失，各段 frames 求和 ≠ 原文件 frames，帧数守恒检查失真。
  2. **检查 2 语义破坏**：每个段切点都是新 GOP/IDR，「段首连 IDR」「frame_num 全局单调」在每个段都重新起算，跨段连续性信息丢失。
  3. **检查 3 语义破坏**：`-reset_timestamps 1` 使每段 pts 归零，pts 连续性/回退/跳变检测失效。
  4. 切段本身有 I/O 开销（写 N 个临时 mp4）＋关键帧对齐的帧偏移误差。

### 2.4 分片策略对比小结

| 分片方式 | 视频映射 | 适合检查项 | 核心障碍 | 结论 |
|----------|----------|-----------|----------|------|
| 按大小 | 按字节切 ES | 检查 2（部分） | NAL 跨边界 + 全局统计合并 | 收益有限，不建议 |
| 按行 | 按 NAL/VOP 记录 | 检查 2（部分） | SPS/VOL 头部依赖 + 预扫描即瓶颈 | 收益有限，不建议 |
| 按记录 | 按帧/时间切段 | 检查 4（chroma） | 破坏检查 1/2/3 全局语义 | 仅检查 4 可行 |

**核心结论：对主瓶颈（检查 1+3 全量解码）三种分片方式均不可行**，因为帧数守恒、pts 连续性、连 IDR、frame_num 单调都是全局/跨段性质，任何切分都会破坏语义。

---

## 3. 并行处理框架选择（用户维度二）

### 3.1 多线程（threading / ThreadPoolExecutor）

- **适用**：I/O 密集 + 释放 GIL 的场景。本脚本的「重活」是 ffmpeg 子进程（`subprocess` 阻塞等待），Python 线程在 `subprocess` 调用期间**释放 GIL**，多线程可真正重叠多个 ffmpeg 进程。
- **已用**：`run_verify_parallel` 的 ThreadPoolExecutor（多文件并行）+ GPU 信号量。
- **v4 新增**：检查间并行用 `threading.Thread`（`_check2_worker`）——检查 2 的 ES 提取（`-c copy`，I/O 密集）与检查 1+3 的 ffmpeg 解码（子进程）重叠，二者都是「等待子进程」模式，线程天然高效。
- **局限**：纯 CPU 的 Python 解析（`parse_*_es`）受 GIL 限制，多线程无法加速单文件的解析本身。

### 3.2 多进程（ProcessPoolExecutor）

- **适用**：CPU 密集（纯 Python NAL 解析、NumPy std 归约），绕过 GIL。
- **已用**：`--parallel process` 分支。
- **局限**：
  1. 进程 spawn 开销大（每进程重载解释器 + 传参序列化），小任务得不偿失。
  2. **GPU 信号量不可跨进程共享**（`_GPU_SEMAPHORE` 是线程级），process 模式下 GPU 闸门失效，只能靠 `--gpu-workers` 上限约束——这也是 `main()` 里「GPU 启用时强制切回 thread」的原因（line 2070）。
- **适用项**：检查 4 chroma 分片（每个分片独立解码 + NumPy std），可用 ProcessPoolExecutor 真正并行。

### 3.3 分布式（多机）

- **适用**：海量文件、跨机器吞吐扩展。
- **方案**：任务分片 + 结果聚合（Ray / Dask / Celery，或自建「主节点分发文件列表 + 工作节点跑 `_verify_one_video` + 汇总 JSON」）。
- **评估结论**：本项目当前瓶颈是「单文件大视频」而非「海量文件」，分布式解决的是横向扩展（吞吐），不解决单文件纵向加速（时延）。**仅当批处理文件数达千级、单机 CPU/GPU 饱和时才有价值**，本轮不实现，列为进阶方案。

### 3.4 选择依据总结

| 维度 | 多线程 | 多进程 | 分布式 |
|------|--------|--------|--------|
| 加速对象 | I/O + 子进程等待 | CPU 密集 | 吞吐（多机） |
| GIL 影响 | 有（纯 Python） | 无 | 无 |
| GPU 信号量 | 可共享 | 不可共享 | 不可共享 |
| 启动开销 | 低 | 高（spawn） | 高（调度） |
| 本项目结论 | ✅ 检查间并行首选 | ✅ chroma 分片可选 | ⚠️ 进阶、非本轮 |

---

## 4. 内存管理与 I/O 优化（用户维度三）

### 4.1 现状问题

1. **`extract_annexb_es` 全量驻留**：`subprocess.run(..., capture_output=True)` 把整个 Annex B ES 读入内存。hevc 8.6 万帧的 Annex B ES 可能达 1~2GB，峰值内存高。
2. **检查 2 与检查 1+3 串行**：两次完整读文件（ES 提取 + 全量解码），I/O 不重叠。
3. **检查 4 已流式**：`check_chroma_corruption` 用 `Popen` + `_CHROMA_CHUNK_FRAMES=128` 分块读 rawvideo，内存 O(chunk*frame_size) 有界，已是最优。

### 4.2 优化方向

| 优化 | 方案 | 收益 | 风险 | 结论 |
|------|------|------|------|------|
| ES 流式提取 | `Popen` + 分块 `stdout.read()` | 早释放子进程、避免 stderr 全量缓冲 | 低 | 可选（收益小） |
| ES 流式解析 | 边读边解析、逐 NAL 丢弃 | 峰值内存从 GB 降到 KB 级 | **高**（NAL 跨块边界 + SPS/VOL 前向依赖） | 进阶，不建议 |
| I/O 重叠 | 检查 2 与检查 1+3 并行 | 两次读文件重叠，缩短墙钟 | 极低（线程间无共享状态） | ✅ v4 已做 |
| `-f null` 落盘 | 输出到 `pipe:` / 空设备 | 解码不落盘，避免临时文件 I/O | 无（现已是 `-f null -`） | 已最优 |

### 4.3 I/O 特征说明

`-f null` 解码是「读输入文件 → 解码 → 丢弃输出」的流式过程，I/O 主体是**顺序读输入 mp4**。检查 2 的 ES 提取（`-c copy`）也是顺序读同一输入文件。二者串行时输入文件被读两遍；并行后两遍读重叠，利用 OS page cache 可省一次磁盘读，这也是检查间并行的额外收益。

---

## 5. 分片粒度与任务调度开销平衡（用户维度四）

- **粒度过细**：每个分片的任务调度开销（线程/进程 spawn、结果聚合、序列化）超过计算本身。检查 2 单 NAL 解析仅 5~10 字节位读取，逐 NAL 分片纯属负优化。
- **粒度过粗**：分片数 ≤ CPU 核数时负载不均衡（长尾），并行度不足。
- **本项目结论**：
  - 检查 2 的解析单核已快（`bytes.find` C 级 + `_BitReader` 懒加载），**分片粒度再细也换不来收益，调度开销反而吃掉收益**。
  - 检查间并行是**粗粒度（2 个任务）**，调度开销 = 1 次线程创建，远小于省下的检查 2 串行时间。
  - chroma 分片若实现，建议 `n_shards = min(cpu_count, 8)`、按**帧区间**（而非字节）切分，每分片解码一段，负载按帧数均衡。

---

## 6. 逐检查项可行性结论矩阵

| 检查项 | 性质 | 分片并行可行？ | 检查间并行？ | 落地动作 |
|--------|------|---------------|-------------|----------|
| 检查 1 frames==packets | 全局（帧数守恒） | ❌ 破坏语义 | ✅ 与检查 2 并行 | v4 已做 |
| 检查 2 NAL/VOP 统计 | 局部解析 + 全局统计 | ⚠️ 边界复杂收益低 | ✅ 与检查 1+3 并行 | v4 已做（抽线程） |
| 检查 3 pts 连续性 | 全局（时序） | ❌ 破坏语义 | 随检查 1（已合并） | v4 已做 |
| 检查 4 chroma | **局部（逐帧独立）** | ✅ 唯一安全分片项 | ✅ 已在 GPU 闸门外 | 进阶可选 |

**核心可行性结论**：
1. 主瓶颈（检查 1+3 全量解码）**无法分片并行**，唯一安全优化是「检查间并行」——让检查 2 的 I/O/CPU 与检查 1+3 的解码重叠，把串行时间 `T1+T2` 压缩为 `max(T1,T2)+ε`。
2. 检查 4 是 embarrassingly parallel 的局部性质，是唯一适合分片的检查项，但日志场景均 `--skip-chroma`，收益有限，列为进阶方案。
3. 分布式解决吞吐不解决单文件时延，当前不适用。

---

## 7. 落地改造方案（v4 已实施）

新建 `tests/verify_segment_bitstream_v4.py`（由 v3 逐字复制后定向修改，v3 保持不变），三处改动：

### 7.1 文件头 docstring 新增 v4 说明

记录 v4 变更、分析结论引用本文档。

### 7.2 新增 `_check2_worker(video_path, codec, dump_nal, out)`

将原 `_verify_one_video` 中「检查 2 按编码分支」的整段逻辑（mpeg4/h264/hevc 三分支 + 异常捕获）**逐字迁移**为独立线程函数，结果写入可变 dict `out`（`out['stats']` / `out['err']`）。关键点：

- `out['err']` 非 None 的语义与原 v3 逐分支 `fails.append('nal')` **完全等价**（N/A 时 err=None 不判失败；h264 提取为空、连 IDR、frame_num 回退、mpeg4 连 I、vop_time 回退、解析异常均判失败）。
- `out['stats']` 存 `check_nal_stats` / `check_vop_stats` / `check_hevc_stats` 返回 dict。

### 7.3 改造 `_verify_one_video()`

1. **probe codec 提前**（原在检查 2 开头，现提前到函数开头）：O(1) ffprobe，决定是否启动检查 2 线程，codec 立即回填 `result['codec']`。
2. **启动检查 2 线程**：`codec in ('h264','hevc') or codec in _MPEG4_FAMILY` 时 `threading.Thread(target=_check2_worker, ..., daemon=True).start()`。
3. **检查 2 移出 GPU 信号量**：原 v3 检查 2 位于 try 信号量块内，但 ES 提取为 `-c copy` **不占 NVDEC 解码会话**，持有信号量反而让 CPU 任务被 GPU 队列串行化（与 `FIX-CHROMA-GPU-GATE` 同理）。v4 将 `gpu_sem.acquire()` 移到「启动检查 2 线程之后、检查 1+3 之前」，信号量仅保护检查 1+3。
4. **join 聚合**：`finally` 释放信号量后，`nal_thread.join()` 并从 `nal_result` 回填 `nal_stats`/`nal_err`，err 非 None 时 `fails.append('nal')`。

### 7.4 正确性保证

- 检查 2 与检查 1+3 **无数据依赖**：前者读 `-c copy` ES，后者全量解码，各自独立 spawn ffmpeg 子进程，无共享可变状态（`dump_nal` 仅检查 2 线程使用，且多文件模式下为 None）。
- 线程通过 `join()` 同步，`nal_result` dict 无需锁。
- 语义等价性已逐分支核对（见 7.2），不改变任何检查的判定结果。

---

## 8. 进阶方案（本轮未实施，按需启用）

1. **chroma 分片并行**：`check_chroma_corruption` 增加 `n_shards`，按帧区间切分，ProcessPoolExecutor 并行解码各区间 + NumPy std，聚合 bad_frames（加帧偏移）。注意进程 spawn 开销与帧定位精度（`-ss` 关键帧对齐误差）。
2. **ES 流式解析**：`parse_*_es` 改流式（边读边解析、逐 NAL 丢弃），峰值内存 GB→KB 级，但需处理 NAL 跨块边界 + SPS/VOL 前向依赖，风险高。
3. **分布式批处理**：主节点分发文件列表 + 工作节点跑 `_verify_one_video` + JSON 汇总，适用于千级文件吞吐扩展，非单文件时延优化。

---

## 9. 总结

| 问题 | 答案 |
|------|------|
| 主瓶颈能分片吗？ | ❌ 不能（检查 1+3 全局语义） |
| 最优落地是什么？ | ✅ 检查间并行（v4 已实施） |
| 唯一安全分片项？ | 检查 4 chroma（进阶可选） |
| 分布式适用吗？ | ⚠️ 仅海量文件吞吐，当前不适用 |

---

## 10. v5 落地（2026-08-19，原地升级 `tests/verify_segment_bitstream_v4.py`）

v4「检查间并行」之后原地升级 v5，实现资源自适应最大化。要点：

### 10.1 策略化解码（`--decode-strategy auto|showinfo|gpu_dual|framecrc`）

- GPU 路径默认 `gpu_dual`：`-f null` 无 filter 解码（hwaccel 帧不下载像素）取
  frames/packets/decode errors，并发 `ffprobe -show_frames` 取逐帧 pts；
  消除 showinfo 强制 D2H 的 GPU 路径瓶颈（T4 87k 帧 398s ≈ 218fps 的主因）。
- CPU 路径默认 `showinfo`（与 v4 语义一致）；`framecrc` 为 A/B 备用
  （无 filtergraph，stdout 逐帧含 pts）。
- packets/decode errors 改从同一次解码 stderr 汇总行解析
  （`N packets read; N frames decoded; N decode errors`），
  `ffprobe -count_packets` 降为 fallback——该命令本身是一次全文件 demux
  扫描，并非 O(1) 读包头（v4 分析的事实错误在此修正）。

### 10.2 检查 1+3 分块并行解码（`--decode-chunks auto/N`，仅 CPU 软解）

- `ffprobe -show_packets`（不解码）构造关键帧对齐分块计划；
  各块 `ffmpeg -copyts -ss <关键帧pts> -i in -frames:v <块帧数+8余量>` 并行解码。
- 按**显示 pts 窗口** `[start, end)` 过滤边界泄漏帧；失败块以 `-threads 1`
  单线程确定性重试；Σ帧数 != 容器帧数或重试仍失败 → 自动回退整文件单次解码。
- 实测踩坑记录（本机 ffmpeg N-122480 2026-01 构建）：
  1. 输出型 `-ss`/`-to`（`-i` 之后）不生效，showinfo 仍看到全部帧；
  2. 输入型 `-ss` + `-copyts` 保留原始 pts，但 `-t` 时序语义异常，须用
     `-frames:v` 帧数截止；
  3. B 帧重排/线程级解码使泄漏帧位置不稳定（有时第 N+1 帧、有时顶替第 N 帧），
     不能按数量修剪，必须按 pts 窗口过滤；
  4. 包索引区间计数 ≠ 显示 pts 区间计数（HEVC b-pyramid 尾部块 363 vs 360），
     计划计数必须按 pts 值而非索引。
- 验证：1200 帧 h264/hevc 样本分块 4-8 路，稳定性 20/20 + 8/8 零回退；
  色度分片 8/8；中段比特损坏文件分块路径正确检出解码错误并 FAIL。

### 10.3 检查 4 色度分块（`--chroma-shards auto/N`，仅 CPU 软解）

复用分块计划；每块 rawvideo 解码 + showinfo 逐帧 pts 过滤像素帧
（解码器 EOF flush 时 showinfo 可能比像素多 1 行，取最小长度前缀），
坏帧索引全局聚合；任一分块失败自动回退单流。

### 10.4 检查 2 ES 内存有界 + 资源自动探测

- `extract_annexb_es` 由 `capture_output=True` 全量驻留改为流式写临时文件 +
  `mmap` 只读映射（页缓存背衬，峰值堆内存 O(1)），批处理 ×workers 不再放大。
- 新增 `detect_vram_gb()`（pynvml/nvidia-smi），打印完整资源计划
  CPU/RAM/VRAM/workers/gpu-workers/decode-chunks/chroma-shards；
  容器感知（cgroup v1/v2 + cpuset）沿用并参与分块数计算。
- 新增 `--timing` 逐检查分项计时（probe/decode/check2/chroma）。

### 10.5 后续 A/B 建议

在真实 8.6 万帧 hevc 文件上对比 v3/v4/v5（单次 vs 分块），记录 decode
分项耗时、回退率与 pts 边界缝合正确性；GPU 路径对比 gpu_dual 与 showinfo
确认 D2H 假设；生产部署前在目标 ffmpeg 版本上跑一次等价性自检
（已知好文件断言 Σframes==packets 且无回退）。

---

## 文件命名（2026-09-15 使用者确认，勿再推断为冗余/残留）

`tests/verify_segment_bitstream_v5.py` 就是**生产侧最新版** `..._v4.py` 的内容：
使用者保留了本地旧 v4，把最新版本另存为 v5，**Linux 生产侧亦已同步为 v5**。
因此当前 v4 与 v5 **逐字节相同**（sha256 前 16 位均 `2566804141ee2c7a`，各 192131 字节）
—— 这是「同一份内容两个名字」，**不是冗余副本，两个都不要删**。

⚠️ 教训：本仓的 `_vN` 后缀**既可能表示历史版本，也可能表示「最新版另存」**。
用快照差集或「逐字节相同」推断「谁该删」之前，必须先确认是否存在人为改名/搬移。
（曾据此误判 v5 为同步残留 —— 原因是所依据的 `11:45` 快照早于改名动作。）

改名遗留的两个真实待办：
1. `tests/test_chroma_false_positive.py` 原以**运行期 import** 方式
   `from verify_segment_bitstream_v4 import check_chroma_corruption`（2 处）——
   已改为 `_load_chroma_check()` **双名兼容**（v5 优先，回退 v4）。
2. 约 40 处文档/注释仍写 `..._v4.py`；其中 `external/ifrnet_video/ffmpeg_io.py:436`
   承载「为何不用 `-ss` 快速路径」的实测论据，属实质注释，应优先更正。
3. v5 内部自指的日志名仍是 `verify_segment_bitstream_v4_stuck.log`（纯外观；
   若要改会破坏 v4/v5 逐字节一致性，先确认无工具依赖该一致性）。

（更完整的清点任务见 `Plan/门禁与测试资产纳管清理_立项Prompt.md`。）
