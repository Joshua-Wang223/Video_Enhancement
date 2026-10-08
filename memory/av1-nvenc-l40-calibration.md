---
name: av1_nvenc L40 等质量标定（CQ 行 + QP 仿射行落地）
description: 2026-10-04 L40 落 QUALITY_MAP_QP['av1_nvenc']=(7.9338,-97.5136)（替代 ×3）+ 门禁同步 + CQ 行 LOO 结构性失败；含标定执行坑（manifest 路径/symlink/同名）；⚠ 2026-10-06 复核：L40 标定的**执行证据不可审计**（待仓主澄清，状态降级为「未验证」）
type: project
---

# av1_nvenc L40 等质量标定（2026-10-04）

**事实（已落地）**
- **CQ 轴**：`QUALITY_MAP['av1_nvenc'] = (1.4566, 1.2165, 0, 63)`（**本会话按 VE 规范化池
  `points/gpu_l40_cq` 落表**；此前为并行会话/VU 的 `(1.4573, 1.1022)`，无法由任何 VE 池复现）
  → crf21 = `-cq:v 32`（与旧值相同，功能无差异）。✅ VU 已同步（VU `ee3bfd9`），⑨ 恢复。
- **QP 轴**：本会话落 `QUALITY_MAP_QP['av1_nvenc'] = (7.9338, -97.5136, 0, 255)`（LOO 3.64）→ crf21→69、crf30→141。**取代** `_QP_MAP_OVERRIDE` 的 ×3。
- **门禁同步**（`Accessory/verify/crf_cq_unification_verify.py`）：G1-2 av1 `-cq:v 32`；G3-7 → CQ32→QP70；G6-7/G6-8 `-qp 70`；G6-9/G6-10 `-cq:v 32`；G3-9（size 口径）仍 63 不变。探针 AC1 结论文案改指 `QUALITY_MAP_QP`。
  ⚠ **CQ 行的 b 会经 `to_x264_crf` 往返影响 QP 期望**：旧 b(1.1022)→71、新 b(1.2165)→70。
- **验收**：`crf_cq --quick --no-gpu` 104/0/0/11；`--gpu` 113/0/3；`plan_implementation_gate` 无失败。

**CQ 行 LOO 结构性失败（不新增，保留既有行）**
- 17 素材 LOO = 6.21（稀疏）/ 6.08（加密），超门禁 5.9；**单素材离群** `anim2d_forest`（6.08 独占，次高 4.01，其余 ≤2.43）。
- 已排除：噪声（重测逐位相同）、采样稀疏（统一加密仅 6.21→6.08）、拐点陡（其斜率比排第 9）。
- 真根因 = **模型形态**：全局仿射 `crf→cq` 给 crf30 → cq45.3，该素材真等质点 cq41.8 → ΔVMAF 6.08（内容相关编码效率偏移）。
- "增加素材"（bootstrap n=4..16，中位 6.30→6.21 **平台**）与"改模型形态"（仿射 6.41 / 二次 **6.80 更差** / 幂 6.18）**均无效**。
- 属 memory `equal-quality-loo-model-form-failure.md` 同类结构性上限。

**跨仓（CR-4）· 已闭环（2026-10-04）**
- VU 已全套对齐：CQ 行 → `(1.4566,1.2165)`（`ee3bfd9`）；quality 口径 QP 改仿射 `_QP_AFFINE_QUALITY['av1_nvenc']=(7.9338,-97.5136)`（`e32d71c`）；LOO 门禁锚点钉 `[0,27]`（`2e9f50d`）。
- 两仓共享行**逐条相等**（含 SIZE_MAP / QUALITY_MAP native / h264,hevc,av1 CQ / h264,hevc,av1 QP 仿射）；VU 侧有 CR-4 相等性断言。
- 高 ref 差异：×3 残差 crf24 −20 / crf27 −37 / crf30 −51。已写 handoff：`VidUtils/Plan/CR-4_av1_QP轴_handoff_VE_to_VU_20261004.md`。

**执行坑（本轮新踩，复用时必看）**
1. **manifest `slice_path` 是 WSL 绝对路径**（`/mnt/d/...`），batch loader 对绝对路径不回退 ⇒ 全部 `[skip]`。须生成指向 `/workspace/input_videos/eqq_calib/` 的修正 manifest。⚠ 修正时**不要用 `.resolve()`**——会把 symlink 解析成目标名（见 3）。
2. **素材池在仓库外** `/workspace/input_videos/eqq_calib/`（12×6s + 5×10s = 17），不在 repo `input_videos/`（该目录不存在）。方案文档 §5.3/§10 写的相对路径会失败。
3. **`screen_ui_code` 6s/10s 同名不同内容**（6.0s vs 10.0s）⇒ 必须按库内约定给 10s 加 `_10s` 后缀（symlink 即可，勿成改原文件）；否则 batch workdir 撞车 + points key 合并成 16 素材（LOO 虚降 5.69~5.93）。
4. **6s/10s/legacy10s 侧 points 的素材名是原始源名**（`new1.mp4`/`word_world_2.mp4`），而 gpu_l40 侧是**切片名**（`live_kids_play_src1280x720.mp4`）⇒ 两套名字不重合、不互相 merge；落表器按 points key 首字段识别素材（不走 `clip_name_mapping`）。
5. **multi-session**：本机是共享 GPU 主机，曾有并行会话在会话期间改 `convert_crf.py`（落 CQ 行）⇒ 动手前先 `git status`/mtime 核对，勿假设树静止。
6. 落表器 `--sides gpu_l40_{cq,qp}` 单独即可复现 NVENC 行（每条文件自带 libx264 锚点）；加 `6s,10s,legacy10s` 不改变 NVENC 行。

**已知边界（非表错）**：标定切片是 **720p prep**，生产/G7 在全分辨率量测 ⇒ 映射非分辨率不变量（word_world_2 全分辨率 cq32 ΔPSNR −2.14 WARN，同值在 720p 切片 −0.34 PASS）。

**CQ 结构性上限的解法（2026-10-04 收口）**：不是表的问题，是**门禁加权**问题 —— 逐锚点最差
0.94/1.93/2.91/3.25/**6.41(crf30)**，8/8 编码器 worst 都来自 crf30。已按仓主裁定改
`GATE_ANCHORS=[0,27]`（+ crf30 降为监控列）⇒ av1 CQ 判据 LOO **3.13 ✅**。详见
`equal-quality-loo-gate-anchor-range.md`。

**L40-5 冒烟（358.8s 真实素材）**：首轮 PASS=13/FAIL=3。
- S3（两模式）为**脚本 double-count bug**（`Step 1/2`+`Step 2/2` 各校验一次分段）⇒ 已修
  `[FIX-S3-STAGE-DEDUP]`（只取 `Step 2/2` 区域）；用已有日志回放：24→12 条、35852→**17926 = 产物帧数** ✅，管线本身正确。
- S8（constqp）**空闲机复跑仍 FAIL**（斜率 **+149.5 MB/min**，峰值 8506 MB；vbr −21.4 通过）
  ⇒ 两次（并发/空闲）都复现、非污染，疑**真实增长**；`MemWatcher` 是进程树 RSS 求和且未落进程数
  ⇒ 暂**无法区分「主进程泄漏」与「子进程累积」**，需增强采样后复跑定位。与标定无关（未改管线）。
  复跑同时确认 S3 修复有效：S3 **17926=17926 PASS**。
  **挂账（2026-10-04 仓主裁定：留待下次上机继续）**：
  · **硬件依赖**：S8 的*精确冒烟项*用 `av1_nvenc` ⇒ **只能 Ada/L40**；但 S8 所测*机制*是
    **rc 模式（constqp vs vbr）**，而 `external/ifrnet_video/ffmpeg_io.py:934-955` 的 constqp 分支
    对**整个 NVENC 族**一致（只差「不发 `-rc-lookahead`（LA 硬件禁用）+ 省 `-b:v 0`」），**非 AV1 专属**
    ⇒ **T4 上用 `h264_nvenc`/`hevc_nvenc` + constqp 很可能复现，且可承担修复的开发/验证**（成本远低于等 L40）。
  · **纯 CPU 不可行**：CPU 无 NVENC，且 `constqp` 是 NVENC 概念；软编无同义路径（仅可作「已证机制与编码器无关」后的旁证）。
  · **不依赖 GPU、可提前做**：增强 `MemWatcher`（记录进程数 `n` + 每采样 top-N RSS 并落盘）+ 静态审阅
    constqp/vbr 路径的缓冲差异，把下次上机压缩成「一条命令 + 读结论」。

**S8 静态审阅结论（2026-10-04，纯 CPU 完成）——排除性结论比候选更重要**
审阅对象是**生产链路** `external/ifrnet_video/{pipeline,nvenc_sdk,ffmpeg_io}.py` +
`external/realesrgan_video/` 同构件（历史 `external/IFRNet/process_video_v6_4_*` 仅供对照）。
1. **「constqp vs vbr 的差异只有 LA 与 `-b:v`」只在 CLI writer 成立，在本次冒烟里根本不适用**：
   AV1 走 **SDK 直通**（Level 1/2），`ffmpeg_io` 的 `quality_args` 只在 fallback 路径生效，
   ffmpeg 只做 muxer ⇒ **ffmpeg 命令形状（`-rc-lookahead` / `-b:v 0`）排除为累积源**。
2. **两次跑的真正差异是「两条不同代码路径」，不是同一个 encoder 的两种 RC 配置**：
   `main.py:1842-1843` 对 constqp 强制 `_level1_la=0`，vbr 保留配置 LA（8）
   ⇒ constqp 走 `encode_frames_batch_ce_pipeline`（per-batch），vbr 走 `_encode_chunk`
   → `encode_frames_stream`（`_acc_nv12` 分块累积）。**S8 斜率差异应按「路径差异」而非「RC 参数差异」解读。**
3. **已确认有界（排除）**：`_strm_slot_pending` / `_slot_pending`（constqp 的 ce_pipeline 从不填，
   append 点只在 `encode_frames_stream`）、`_cached_sps_pps`（单值覆盖）、每批 `results`、
   `_slots`（段级 `_destroy_all_slots`）、`_la_pinned_pool`（形状键 + 取模，vbr 侧）。
4. **跨段累积被结构性排除**：AV1/HEVC **每段强制新建编码器**（`main.py:1155-1157` `_force_new`）
   ⇒ 任何挂在 encoder 实例上的容器都活不过一段。
5. **头号候选被证伪**：per-frame CUDA event（LA=0 专属，`nvenc_sdk.py:3079-3081` 建、
   `:2973/:3161` 销）在 `cuEventSynchronize` 失败时 `raise` 前**跳过销毁**看似泄漏；
   但该 `raise` 会让编码线程置 `error` ⇒ **rc≠0**，而实测 S1 rc=0 ⇒ **这条路径从未触发**。
   ⚠ 这类「只在异常路径泄漏」的机制，**用 rc=0 就能排除**，不必上机。
6. **剩余存疑（需上机打点）**：`_stderr_lines`（ifrnet `nvenc_sdk.py:4290` / esrgan `:3911`
   纯 append **无裁剪**，而同仓 `esrgan ffmpeg_io.py:425` 有 `>400 删前 200` ⇒ 两处不一致）；
   `self.timing`（esrgan `pipeline.py`）。二者量级估都远小于 150 MB/min，**仅作排除用**。
7. **未解释的部分要如实说**：以上都是「排除」，**没有任何一条能正面解释 +150 MB/min**。
   ⇒ 下次上机的价值在于**用增强后的采样直接定位归属**（主进程 vs ffmpeg 子进程 vs GPU 侧），
   而不是继续静态猜。若分组斜率显示增长在 `ffmpeg` 标签 ⇒ 是子进程（读帧器/分段 muxer）；
   若在 `main` ⇒ 看 `pss` 是否同步涨（真泄漏）还是只有 RSS 涨（CUDA 上下文/共享页虚高）。

**S8 准备项已落地（2026-10-04，`Accessory/verify/av1_pipeline_smoke.py`）**
- `[FIX-S8-ATTRIB]` MemWatcher 不再只存 `(t, rss, gpu)`：**保留并落盘进程数 `n`**，新增
  主进程 / 子进程**分组 RSS 与 PSS**（PSS 取 `/proc/<pid>/smaps_rollup`，避免共享页在多进程里重复计数），
  逐进程明细（pid/rss/pss/角色）经 `--mem-dump-dir` 落 `<rate_mode>.mem.tsv`（可离线重分析，不必再上机）。
  显存改为 **`--query-compute-apps=pid` 按 pid 归属**统计（`_gpu_mem_by_pid`），
  整卡 `memory.used` 只作参考 —— 共享 GPU 主机上整卡值会把他人的任务算进来（本仓已有实证）。
- `[FIX-S8-CRITERIA]` 判据扩为三项：**斜率（口径与历史完全一致，未改，保证与 +149.5/−21.4 可比）
  + 峰值上界（`--mem-peak-mb` 默认 12000，短跑里 OLS 斜率对进段时机/段内台阶敏感，峰值是独立第二道判据）
  + 样本充分性（`--mem-min-samples` 默认 8，不足报 SKIP 而非给假 PASS/FAIL）**。
  斜率超阈值时 detail 直接附**主/子分组斜率**，报告里即可读出归属，不必再跑一轮。
- **自测（无 GPU，已做）**：合成泄漏子进程（每 0.5 s 追加 20 MB）被正确检出
  斜率 +1887.9 MB/min，且分组斜率 `main +1888.0 / child −0.1` 精确归因；
  静止进程组 slope 0.36 MB/min（噪声量级）⇒ 无误报。`plan_implementation_gate` 无失败。
- ⚠ ~~本容器 `nvidia-smi --query-compute-apps` 返回空（无 GPU 挂载）⇒ 显存字段为 0~~
  **该归因已于 2026-10-04 被 T4 实测推翻**，见下方「§9.5-D 项已在 T4 执行完毕」的 B3。

**下次上机待办（2026-10-04 定稿，权威版在方案 §9.5）**
- **A｜S8 定位（最高优先）**：一条命令跑完读 S8 detail 的 `主 x / 子 y MB/min` 即可得归属，**无需二次跑**。
- **B｜显存归属口径复验**：随 A 同一次跑批覆盖；判据 = `.mem.tsv` 的 `gpu_tree_mib` 列非 0 且能与整卡对账。
- **C｜S8 峰值上界校准**：`--mem-peak-mb` 默认 12000 是估值；据 A 的实测峰值调，或记为「宽裕上界·不敏感」。
- **D｜constqp 能否在 T4 复现（降本）**：T4 用 `h264_nvenc`+constqp 跑同一冒烟。复现 ⇒ 与 AV1 无关，
  **修复可在 T4 开发验证**（远低于等 L40）；不复现 ⇒ 回 L40 查 AV1 特有因素。
- **E｜顺带**：① `av1_vp9_quality_matrix` 退出码口径（1 vs 0）② L40-6（AV1 Level 1 `code=12`）。
- ⚠ **只跑 constqp 不足以判读**，必须 `constqp,vbr` 同素材对照，才能区分「constqp 特有」与「长跑时间相关项」。
- 读数判读表（方案 §9.5 有完整版）：子正主零=ffmpeg 子进程累积；主+子同正且 PSS 同步涨=主进程真泄漏；
  主正但 PSS 不涨=CUDA 上下文/共享页虚高（**降级判据，别急着改管线**）；两者≈0 但峰值超上界=段切换清理问题。
- **方案文档已定稿待办章节**：`Plan/PROMPT_L40_AV1等质量标定专项执行方案.md` **§9.5**
  「下次上机待办（T4/L40 通用，按优先级）」含A~E 清单 + 一条命令 + 上述判读表；
  §0 状态已改为「执行完毕并落表；S8 遗留的无 GPU 准备已完成」，§5.4/§10 命令已带
  `--mem-interval 5 --mem-dump-dir /tmp/s8_mem`。
- **本专项的验证基线（别当回归）**：本容器（无 GPU 无 cv2）实测
  `plan_implementation_gate` **84 项/ 0 失败**（4 WARN / 5 SKIP 为环境性）、
  `crf_cq_unification_verify --quick --no-gpu` **PASS=51 / FAIL=3**，
  那 3 个 FAIL 全部是 `ModuleNotFoundError: No module named 'cv2'` ⇒ **缺 cv2 的既有环境问题**，
  与改动无关；已用 `git stash` 对比确认与改动前逐项一致。

---

**§9.5-D 项已在 T4 执行完毕（2026-10-04）—— S8 收口 + 三个新阻塞项**

执行环境 = **Tesla T4（sm75）**，非 L40 ⇒ 走的正是 D 项（降本验证）。素材
`01 the race to mystery island.fixed.mp4`（358.76s / 720×576 / 12 段 / 8969 源帧）。
脚本改动：`av1_pipeline_smoke.py` 增 `--codec`（默认 av1_nvenc 不变）+ `--batch-size`（默认 8）；
门禁 **96 项 / 94 通过 / 0 失败**。完整版见 `Plan/PROMPT_L40_AV1等质量标定专项执行方案.md` **§0.1**。

**S8 结论（措辞勿越界）**：T4 上 constqp 两臂（h264 −55.6、hevc +3.1 MB/min）**均无泄漏**
⇒ 否证「constqp 路径泄漏」。但 **h264 的 vbr 对照臂崩了**（见 B1），A/B 只在 hevc 成立；
**S8 原始问题（L40 的 +149.5 是否 AV1 特有）仍未回答**。

**⚠ S8 判据本身不可靠（本轮最重要的方法论产出）**：同一份 HEVC constqp 数据只换 OLS 窗口
⇒ 前 10% **+1434.2** / 后 50%（判据口径）+3.1 / **全程 +149.4** / 稳态 +36.1 /
后 25% −240.5 / 末 10% **−786.7**。⇒ **「全程 +149.4」与 L40 的「+149.5」几乎相同**
⇒ L40 那个数**疑为窗口伪影**而非真泄漏。锯齿幅度来源之一 = **pinned result pool 随
batch_size 线性增长**（bs=24 实测逐段 305→549→794→1038→1190→**1251 MB**）⇒ 冒烟默认
`--batch-size` 改 8（**实测**：T4 40s 段插帧阶段 2 轮交替 A/B，bs=24 墙钟 33.45s / 单批
343~363ms / pinned 305MB vs bs=8 墙钟 **29.14s（−12.9%）** / 单批 **90ms** / pinned **102MB**，
两轮各差 <0.5%；仅插帧阶段，超分阶段收益未实测）。**跨 bs 的 S8 读数不可比。**
后半程 sd 1356 MB ⇒ 不确定度约 ±66 MB/min，
**|slope| < 5 的阈值都是随意的**，须先定噪声底噪。
**⚠ `--mem-peak-mb` 默认 12000 已被实测否决**：T4 两臂峰值 13578~13877 MB **超上界但斜率正常**
⇒ 峰值判据比斜率更易误报。

**B1（生产级缺陷）h264 + LA>0 + 跨段复用 ⇒ 片段 2 起必崩**
`non-existing PPS 0 referenced` → muxer pipe broken → rc=1（61s/62s，vbr 与 vbr_hq 两臂同现）。
根因四段闭环：`ffmpeg_io.py:163` `nvenc_map={'libx264':'h264_nvenc'}` ⇒ **config 默认 libx264
在 T4 上就升级为 h264_nvenc（生产默认路径）**；`main.py:1155` `_force_new` 只含 `("hevc","av1")`
⇒ h264 复用编码器；`nvenc_sdk.py:2282/2285` `_stream_begin` 每段清 `_cached_sps_pps=None`；
`nvenc_sdk.py:971` `repeatSPSPPS bit12 不写` ⇒ 驱动不重吐参数集 ⇒
`_prepend_param_sets`(`:2297`) 与 `_drain_write`(`:3927`) 两条通道同时失效。
**为何 constqp 臂同素材同复用却不崩**：constqp 清 LA → `ce_pipeline` 路径 `:2858-2859`
**只重置 `_sps_pps_injected`、不清 `_cached_sps_pps`** ⇒ 段 2 由 `_prepend_param_sets` 补挂。
**这个 LA=0 / LA>0 的不对称就是缺陷面。** 回归点 `1e57c0b`(2026-09-18)。
ESRGAN 侧 `realesrgan_video/nvenc_sdk.py` 逐字同构。
修复方向（待做）：`_stream_begin` **保留** `_cached_sps_pps`、只重置 `_sps_pps_injected`
（与 LA=0 路径 `:2858` 对齐）；hevc 臂作负向回归（现 0 错，不能被改坏）。

**B2 `--rate-mode vbr/cbr` 在 Level 1 静默落 CONSTQP 且 LA 未启用**
`nvenc_sdk.py:1055` `else` 兜底写 `rc_ptr[1]=0`；`:1069` LA 门控 `in ('vbr_hq','qvbr')` 排除 vbr；
`:634` 清 LA 判据是 `== 'constqp'` ⇒ **Python 认为 LA=8 而硬件是 CONSTQP+LA=0**。
既有 memory 只记「静默落 CONSTQP + LA 失效 + 别传 vbr」，**本轮新增三点**：
① `[FIX-CONSTQP-FRAME-CE]` 守卫是 `('constqp',0)`（`:2500`），vbr 走 `('vbr',8)` 不命中
⇒ **绕过 per-frame completionEvent**，正落在 [[nvenc-drain-unsubmitted-slot-segfault]] 记载的
T4 崩溃组合上（**代码路径确认、崩溃未复现**，本轮 rc=1 是 muxer pipe broken 非 segfault）；
② QP 未过 `to_constqp_qp` 换算 ⇒ 画质偏松（实测 h264 constqp Ready `QP=22` vs vbr `QP=26`）；
③ `nvenc_sdk.py:568`（IFRNet）/:636（ESRGAN）**AV1 自动降级产生 `'vbr'`**，是**无需用户传参**的
第三入口（当前因 AV1 Level 1 code=12 降级而潜伏）。Level 2/3 CLI 路径**正确**
（`ffmpeg_io.py:926` 真下发 `-rc:v vbr`）⇒ **两条路径口径分裂**。
⚠ 已**排除**：B2 不是 vbr 臂崩溃的原因 —— vbr_hq 臂是真 VBR_HQ 仍报同样错误。

**B3 显存 `gpu_tree_mib` 恒 0（静默假零）** —— **推翻了旧归因**
`nvidia-smi --query-compute-apps` 返回**宿主 pid**（实测 725115 / 1071006），在容器 `/proc`
下**不存在**；`/proc/<pid>/status` 的 `NSpid` 只有一层 ⇒ **容器 PID namespace 与驱动侧不通**，
pid 查表永远 miss。**与 GPU 是否挂载无关**（跑批时 GPU 实际满载 7698 MiB/100%）。
仅污染 S8 的显存维度，RSS/PSS 口径正确。`0.0` 会被读成「确实没用显存」，
**须改报 `None` 才表达「测不到」**。⇒ [[memory-leak-attribution-measurement]] 里
「按 pid 归属」的规矩在**容器内不可实现**，缺的是容器场景落地。

**记录勘误**：E1 [[nvenc-sps-pps-debugging]] 与 E2 [[sps-pps-la-pipe4-startup-corruption]]
「多段段 2+ 不再报 non-existing PPS」在 **h264+LA>0+跨段复用**下**已不成立**（成立范围仅
LA=0 或 hevc/av1 每段新建编码器）；E3 [[ifrnet-multisegment-la-strict-drain]] 未记同提交
`1e57c0b` 引入的 SPS/PPS 清空缺陷（B1，机制不同但同源）；E4 [[t4-vbrhq-verification-plan]]:43
行号 `1044/1056` 已漂移为 **`1055/1069`**（`be8e1db` 插入 11 行注释）；E5 同条缺上述三点。

**原始采样已存档（2026-10-04）**：`verification_report/s8_20261004_raw/` —— 五份 `.mem.tsv`
（h264_constqp / h264_vbr_FAILED / h264_vbrhq_FAILED / hevc_constqp / hevc_vbrhq）
+ `README.md`（采样器列口径、每份对应哪条臂、**三条使用纪律**、失效字段说明）。
⇒ **后续修复与复算不必再花一次 GPU 上机**。⚠ 两条纪律已写进 README：
失败臂仅 12 点全在启动爬坡段（复算会得 +4585.8/+1872.4 的**伪影**，不是趋势）；
全部为 bs=24，与 L40 的 +149.5 才可比（脚本现已默认 bs=8）。
已核验 1677/1677 行 `gpu_tree_mib` 全为 0（佐证 B3）。

**下一步顺序**（有依赖，勿打乱）：修 B3（无需 GPU）→ 修 B1（需 GPU + hevc 负向回归）→
重跑 h264 vbr 臂（第一次真正测到 h264 的 LA>0 路径）→ B2 方案 A（脚本级禁 vbr/cbr + 修
`_log_ready` 自相矛盾回显；方案 B 需 GPU A/B）→ 标定 `--mem-peak-mb` → L40 三臂同批。

---

## ✅ S8 原始问题已回答（2026-10-05，T4，358.76s 标准素材双臂同批）

**结论：`+149.5 MB/min` 是窗口伪影，S8 无泄漏。**
素材 `01 the race to mystery island.fixed.mp4`（358.76s / 720x576 / 12 段 / 8969 源帧），
`--segment-duration 30`、`batch_size=8`、`h264_nvenc`，两臂**同批跑**：

|臂 | rate_mode | rc | 耗时 | 后半程 RSS 斜率（主/子/PSS） | 峰值 RSS | code=8 | S1~S8 |
|---|---|---|---|---|---|---|---|
| 1 | constqp（LA=0 / ce_pipeline） | 0 | 1821 s | **−76.8**（−76.4 / −0.4 / −76.3） | 6730 MB | **0** | 全 PASS |
| 2 | vbr_hq（LA=8 / encode_frames_stream） | 0 | 2081 s | **+12.6**（+13.2 / −0.6 / +13.0） | 7694 MB | **3** | 全 PASS |

两臂均 **16 PASS / 0 FAIL**、12/12 段 `decoded==expected`、产物 17926 = 分段合计 17926。

⇒ **L40 的 +149.5 不复现**：T4 上同素材两臂斜率分别 −76.8 / +12.6，
与 §0.1.2 的「同一份 hevc constqp 数据换 OLS 窗口可得 −786~+1434」一致
⇒ L40 那个数是**窗口伪影**，非真实泄漏。**S8 至此闭环。**

⚠ **但「code=8 只在 vbr_hq 臂出现」现在是同批对照（0 vs 3），比之前的跨run 比较强**：
它确认了 code=8 与 **LA>0 路径**绑定（constqp 臂零命中），
但**仍不能区分**「槽位不足」vs「LA warmup 期未就绪」——这正是
[[h264-la-ready-gate-asymmetry]] 的 A/B（9vs 11 槽）在做的事。

### `--mem-peak-mb` 标定完成（方案 §0.1.5 步5）
旧默认 12000 是 **bs=24 下的估值**，实测被否决：bs=24 三臂峰值
12393/13578/13877 **全超上界而斜率正常** ⇒ 必然误报。
实测峰值随 bs 近似线性：**bs=8 → 6730(constqp)/7694(vbr_hq)**；
**bs=24 → 12393/13877**。
⇒ 已改为 `[FIX-S8-PEAK-BS-AWARE]` **按 bs 自动标定**：
`_auto_peak_mb(bs) = 7694 × bs/8 × 1.25`（bs=8→9618、bs=24→28852，
两档均高于实测值 ⇒ 不误报）。`--mem-peak-mb` 显式给值则关闭自适应。
⚠ **仍不能跨素材通用**（分辨率/时长影响工作集）；换素材后若报「峰值超上界」
而斜率正常，先怀疑需重标定，别直接判失控。

---

## ⚠ 2026-10-06 复核修正：「窗口伪影」论证不成立，S8 改判「工程闭环（不可复现）」

上节「结论：`+149.5 MB/min` 是窗口伪影 … S8 至此闭环」**论据有误**，逐条离线复算
（`verification_report/s8_20261004_raw/*.mem.tsv`）后更正：

1. **跨窗口错误对比**：L40 的 +149.5 是**后 50%** 口径（当次脚本 `b64c1d2` 的
   `half = rows[len//2:]`），而被类比的 T4 **+149.4 是全程**口径 ⇒ "几乎相同"不构成证据。
2. **噪声口径纠错**：后 50% 斜率 **SE 仅 ±8~15 MB/min（95%CI ±20~30）**，不是本节记的
   「±66 MB/min」（那是残差 sd≈1356 MB 的量级）⇒ **+149.5 不是噪声**。
3. **不可直接复测**：L40 原始采样**从未落盘**（未用 `--mem-dump-dir`），无法复算其其它窗口。
4. **同臂两跑不稳**：L40 av1 constqp 两次 = +112.1 / +149.5。

⇒ 改判：**「工程闭环（不可复现 / 风险接受）」**，**不得**写成「根因已解释」；
硬闭须一次 L40 复跑（`--batch-size 8` + `--mem-dump-dir`）。判据与证据见
[[l40-av1-pipeline-verification]] 的「### S8 裁定」；原则见
[[feedback_closure_evidence_same_scope]]。

---

## ⚠ 2026-10-06 复核：L40 av1 标定的**执行证据不可审计**（仓主质疑，状态降级）

**触发**：仓主直接质疑「`av1_nvenc` + L40/Ada 侧**始终没有真正跑过标定测试**，怎么能说已经完成」。
复核后**收回**「L40 侧实质工作已完成并落表」的说法。

**支持"跑过"的产物（仅仓内可见）**：
- 池 `Accessory/data/eqq_calibration/points/gpu_l40_{cq,qp}/`（各 17 素材，av1_nvenc 实测点：CQ 255 / QP 272）；
- 自述主机 = L40/Ada sm89 / driver 580.65.06 的报告 `reports/av1_nvenc_L40_calibration_report_20261004.md:4`；
- fitter `eqq_pool_fit_table.py --sides gpu_l40_{cq,qp}` 能复现表值（**只证内部自洽**）；VU 侧镜像同结论。

**审计链断点（⇒ 无法证实执行）**：
1. 报告 `:275` 声称原始 workdir 在 `temp/eqq_gpu_l40/` —— **不存在**；
2. `logs/` 只有软编素材生成日志（`gen6s_*` / `vu_rav1e.log`），**无 GPU 标定日志**；
3. 归档权威 `MANIFEST.md:108-121` 目录树**未收录** `gpu_l40_*`（只有 `gpu_t4_*`）；
4. 全机 `/tmp` **无 2026-10-03~05 痕迹**（容器已重建，仓库文件 mtime 全为 Oct 6）；
5. 唯一存档的 L40 会话痕迹 `/tmp/opencode/`（**2026-09-30** 那次）只有**验证类**（`av1_matrix_L40.log` /
   `crfcq_gpu_L40.log` / `smoke_long_*`），**没有任何 `eqq_calibrate`**；
6. 报告 `:140-160`(§4.6) 自述一次**并行会话冲突**，且 CQ 行 LOO 6.21 超门禁、靠把门禁锚点改 `[0,27]` 才变绿。

**硬约束**：池里是 `av1_nvenc` 实测点，而 `av1_nvenc` **在 T4 上跑不了** ⇒ 若为真实测量必来自某台 Ada；
但**仓内无法证明是本项目哪次会话、更无法证明是 L40**。若非实测（导入/合成），"L40 标定"不成立。

**更正状态**：**不是**「已完成并落表」；准确表述 = 「表值已落下、存在自述为 L40 的池与报告，但**标定测试的实测执行
在仓内不可审计**」⇒ 在仓主澄清前按 **未完成/未验证** 对待。
方案 `Plan/PROMPT_L40_AV1等质量标定专项执行方案.md` 的 §0.4/§0.5（HEAD `1bfea98` 加入）把 L40-1/2 标为「阻塞：需 L40」，
与文档头部（`ea65e19`「执行完毕并落表」）**自相矛盾** —— 从"可审计性"看 §0.4 更稳。

待仓主澄清：**(a)** 从未在任何 Ada 上跑过（则池数据来源需交代、落表值应回滚）；**(b)** 跑过但非 L40 / 未按 §5.3 规格 / 未过门禁。
原则见 [[feedback_artifacts_not_execution_evidence]]。

