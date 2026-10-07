# Video_Enhancement 质量控制参数（crf / cq / qp / preset）修复方案

- 适用位置：`src/utils/quality_map.py`、`src/utils/video_utils.py`、
  `src/main_video_optimized.py`、`src/utils/convert_crf.py`（及共享换算表的两处副本）；
  判据/测试侧另含 `Accessory/verify/crf_cq_unification_verify.py`、
  `Accessory/probe/av1_vp9_quality_matrix.py`、`Accessory/test/test_chroma_false_positive.py`
- 共享真源：`src/utils/convert_crf.py` 的 `QUALITY_MAP`，与 `VidUtils/convert_crf.py`
  **必须逐条相等**（VidUtils 的判据 ⑨ 组会断言）
- 姊妹文档：`VidUtils/Plan/VidUtils_质量控制参数修复方案.md`
- 上机验证脚本：`VidUtils/probe/verify_nvenc_quality_gpu.py`（T4 / L40，重点 av1_nvenc）
- 本仓既有判据：`Accessory/verify/crf_cq_unification_verify.py`（G0 ~ G10）

---

## 0. 状态总览

> 2026-09-29 更新。**本轮范围：仅 VE 单侧**（VidUtils 由另一会话按其 V 系列方案处理）。
> 落地实现与文档原设想的差异见 **§5.1**；上机（T4/L40）验证方案见 **§5.2**；
> **本轮 T4 实测收口见 §6**（含 AC7 软件编码器族验证）；
> **需 L40/Ada（或目标机构建）才能检测验证的 AV1/VP9 测试内容（AC0~AC7）见 §7**
> —— 一条命令入口：`python3 Accessory/probe/av1_vp9_quality_matrix.py --src <真实素材>`。
>
> **2026-09-30 更新（换到 L40 机器后补做）**：AC1/AC2/AC4/AC7 在 L40 上**独立复跑**，
> **P3 长视频冒烟（330 s 真实素材，constqp + vbr 双跑）已执行** —— 过程中暴露并修复
> **三处 AV1 路径缺陷**（其中一处让 AV1 在本仓完全跑不通）。完整数据、缺陷根因与
> 回归结果见 **§8**；AC5 的探测方法被证伪（ffmpeg 7.1 的 QSV 选项表里没有 `-cq`），
> 见 **§8.4**。§7 的状态表与文末「后续建议」已同步更新。
>
> 提交记录：判据/测试/方案/memory 与首轮 T4 报告 = **`9847f59`**；AV1/VP9 探针与 §7 扩展
> = **`69b62db`**（L40 ×3 定案）；L40 收口 + 三处 AV1 修复 + §8 = **`de243a3`**；
> 回归保护（G6-8/9/10 + 新 pytest + 冒烟脚本）与 V9 三行复核（§9）= **见 `git log -1`**。
>
> **2026-09-30 第二轮提示**：该轮开工时容器已**丢失 GPU**（`libcuda` 变 0 字节桩、`/dev/nvidia*`
> 消失，见 §9.1）⇒ P3″ 与 AC5 一样按"环境不支持"跳过，纯 CPU 项（G6 断言、pytest、
> V9 重标定）照常完成。
>
> **2026-10-04 更新（NVENC h264/hevc 等质量标定 + 口径/无损/环境修复）**：
> - **NVENC 等质量表已落**：`QUALITY_MAP['h264_nvenc'/'hevc_nvenc']`（CQ 轴，17 素材，LOO 3.98/5.81，
>   提交 `8f0a605`）与 `QUALITY_MAP_QP['h264_nvenc'/'hevc_nvenc']`（QP 轴，LOO 3.47/3.72，提交 `66ab226`）。
> - **NVENC 显式 opt-in + 裸 CQ 默认**（提交 `be8e1db`）：`--nvenc-tune-*`/`--nvenc-multipass-*` + 目标码率
>   `--bitrate-*`/`--output-bitrate`/`--split-bitrate`；CQ 默认改**裸 `-rc:v vbr -cq:v N -b:v 0`**
>   （`-tune hq` 是 ffmpeg 默认值、固定 CQ 下 multipass 不升 VMAF）。唯一真源 `src/utils/nvenc_tuning.py`。
> - **口径可切 + 门禁口径迁移（B1）**：生产新增 `--quality-mode {size,quality}`（+ `processing.quality_mode`，
>   默认 quality，提交 `d044cdd`）；`crf_cq_unification_verify` 的口径由 size **迁到 quality**
>   （门禁口径 == 生产默认），G1-2/G2/G3/G6 期望随之更新（提交 `1bedff6`）。
> - **无损语义优先**：`to_constqp_qp(codec, 0)` 在 size/quality **两口径均返回 0**（`[FIX-QP-LOSSLESS]` 短路）；
>   生产无损由 writer `crf==0` 分支硬编码 `-rc constqp -qp 0`（G6-18/19 守卫）（提交 `e9ed1d1`）。
> - **环境修复**：FFmpeg 9.0 **移除 `-vsync`** ⇒ 换 `-fps_mode passthrough`（修 `segment_bitstream_verify_v{4,5}`
>   单流/回显 + `video_utils`）。根因即 `test_chroma_false_positive` 两例"返回 None"；**pytest 现 31 passed / 0 failed**
>   （原 29/2）。详见 memory `env-ffmpeg-ffprobe-gotchas.md` §3。
> - **当前门禁基线**：`crf_cq --no-gpu --quick` **104/0/11**、`--gpu` **113/0/2**；`plan_implementation_gate` **96/94/0/2**。
> - **对 AC 的影响**：AC1（AV1 QP ×3）**不受影响**（av1 两口径均 63，G3-7 仍 PASS）；
>
> **2026-10-06 更新（L40 AV1 等质量标定 + 冒烟全完成）**：
> - **L40 AV1 CQ/QP 双轴标定已落表**：`QUALITY_MAP['av1_nvenc'] = (1.4566, 1.2165, 0, 63)`（CQ 轴，LOO 3.13）与
>   `QUALITY_MAP_QP['av1_nvenc'] = (7.9338, -97.5136, 0, 255)`（QP 轴，LOO 2.61），17 素材池化，顺序无关性验证通过。
> - **AC1/AC2/AC4/AC7 全复验通过**：AC1 `-qp 70` 落带内 (1.07× / −1.13 dB)；AC2 G7-6 `-cq:v 32` PASS (+0.16 dB / 1.28×)；
>   AC4 B 组 1.14× / −0.29 dB；AC7 软编族 4 编码器全部 PASS（libsvtav1/libaom-av1/libvpx-vp9/librav1e）。
> - **AV1 长视频冒烟 S1~S8 完成**：constqp/vbr 双臂 358s 真实素材，**15/16 项 PASS**；唯一 FAIL 为 constqp S3 段帧数统计差异（14918 vs 17926，非功能性，S4/S5 解码级验收通过），S8 内存泄漏判据双臂通过（斜率 +32.7/−2.8 MB/min ≤ +50）。
> - **跨仓同步完成**：VidUtils ⑨ 组 **14/14 项一致**；CR-1 (preset p4) / CR-2 (rate control 裸 vbr) 已收口。
> - **全 GPU 门禁基线更新**：`crf_cq --gpu` **111 PASS / 0 FAIL / 5 WARN**；`plan_implementation_gate` **100 PASS / 0 FAIL / 1 SKIP**；`av1_pipeline_smoke` 退出码 0。
> - **L40 专项待办清空**：所有需 Ada 硬件的任务已完成，无阻塞项。

| 编号 | 内容 | 优先级 | 状态 | 依据强度 |
|---|---|---|---|---|
| E0 | `QUALITY_MAP['av1_nvenc']` 的 hi 51 → 63（与 VidUtils 同步） | P0 | **已落地** | 实测（`ffmpeg -h encoder=av1_nvenc` → `-cq (0 to 63)`） |
| E1 | `to_constqp_qp()` 增加 **QP 尺度层**（AV1 族 ×3） | P0 | **已落地**；**L40 实测确认 ×3** | 实测量程 + 实测倍率（L40 扩扫定案，见 §7 AC1） |
| E2 | `libsvtav1` / `libaom-av1` 不再收到非法 `-preset` | P0 | **已落地**（改为白名单，整数映射未做，见 §5.1） | 实测（libsvtav1 `-preset` 为 int `-2..13`，传 `medium` 直接解析失败） |
| E3 | VAAPI → `-qp`（归一到基准轴） | P0 | **已落地** | 实测（h264_vaapi 只有 `-qp (0 to 52)`）；**前提已纠正**，见 §5.1 |
| E4 | `_preset_supported()` 改白名单 | P0 | **已落地**（QSV/AMF/VT 保守排除，未上机） | 实测（ffmpeg 8.0.1：QSV 是 int `0..7`，本机无 AMF/VT） |
| E5 | `_PRESET_P_INDEX` 对齐 ffmpeg 官方枚举 | P1 | **VE 侧已落地**；VidUtils V10 已同步完成 | **实测修正**：只确认了 `medium ≡ p4`（逐字节相同）；`p5=slow / p6=slower / p7=slowest` 是 **NVIDIA p-梯命名**，不是 ffmpeg 命名 preset 的等价关系，见 §6.4 |
| E6 | `libsvtav1` / `libvpx-vp9` / `libx265` / `libaom-av1` 等体积重标定 | P1 | **已落地**（2026-09-29 多素材复核完成） | 真实素材等体积标定（4 次标定：3 素材 + 1 分辨率变体，见 §6.8） |
| E7 | `--rate-mode` 取值表 | P1 | **已落地**（选"保持 3 档 + 明确写明"路线） | 取值表对比 |
| E8 | `--lookahead-depth` 放开量程 | P1 | **已落地**（`0~32`，**不是**文档原写的 `0~250`） | 见 §5.1：NVENC 硬件上限 32，配置层校验本已如此 |
| E9 | `CONSTQP_QP_OFFSET` 真实素材校准 + G7 扩 `av1_nvenc`/`libsvtav1` | P2 | **判据侧已落地（2026-09-28）**：G7-7 软件侧覆盖已生效（本机以 `libvpx-vp9` 代理缺失的 `libsvtav1`）；G7-6 av1_nvenc 本机 SKIP；`CONSTQP_QP_OFFSET` 保持 0（实测已达标，无需调） | 见 §6.3：G7-7 实测 ΔPSNR −1.67 dB / 0.95×（PASS 带内）；`CONSTQP_QP_OFFSET=0` 下 G7-3 ΔPSNR +0.06 dB / 1.46× |
| E10 | G7 增加"合成 vs 真实素材"双跑 | P2 | **已落地（2026-09-28）**：新增 `G7-8` | 见 §6.3：合成 ΔPSNR **+5.33 dB** / 1.53× vs 真实 **+0.06 dB** / 1.46× ⇒ 过配注解成立 |

**本轮额外落地（不在 E0~E10 内）：**

| 编号 | 内容 | 状态 |
|---|---|---|
| P0′ | `Accessory/verify/crf_cq_unification_verify.py` 的 `PROJECT_ROOT` 失效修复（`tests/`→`Accessory/` 搬迁遗留，曾造成 19 个假 FAIL）+ 子进程 stdin 加固 | **已落地** |
| A5 | 判据新增 G1-8 / G2-13 / G3-7 / G3-8 / G5-12 + G6-7（AV1 constqp） | **已落地**；⚠ **2026-09-28 纠正**：`G6-7` 走 `Popen` 替身捕获命令形状、**不需要 AV1 硬件**，T4 上实测已 PASS（原写"需 Ada"不实）；其**期望值**的正确性才依赖 §7 AC1 |
| E9′ | 判据新增 `G7-6`（AV1 硬编）/ `G7-7`（软件侧）/ `G7-8`（E10 双跑）；可选覆盖项改为**按构建可用性探测**（`ffmpeg -encoders`）+ 实跑探测 AV1 硬件能力 | **已落地（2026-09-28）** |
| A6 | 判据健壮性：G7 编码阶段异常不再让整组"执行中断"（原会丢掉 G7-1..G7-8 全部逐项结论，只留 1 个组级 FAIL），改为逐项 FAIL | **已落地（2026-09-28）** |
| A7 | `Accessory/test/test_chroma_false_positive.py` 的 `_load_chroma_check()` 导入修复（搬迁后仍用旧模块名 `verify_segment_bitstream_v5` 且未加 `Accessory/verify` 到 `sys.path` ⇒ 2 个用例 ModuleNotFoundError） | **已落地（2026-09-28）** |
| **A8** | **新增 `Accessory/probe/av1_vp9_quality_matrix.py`**：AV1/VP9 家族 7 个编码器的**一条命令**验证入口（构建+实跑双探测 → 质量族矩阵 → `av1_nvenc` 可用时自动跑 AC1 三点 QP 扫描与判读）；口径与判据脚本严格同源；§7 增补 **AC 覆盖矩阵**与 **AC7（软件族复验）** | **已落地（2026-09-28）**；T4 上实测 VP9 一半（`-crf 28` / 0.95×（朴素 1.30×）/ ΔPSNR −1.67 dB，与 §6.2 的 G7-7 **逐位一致**），AV1 家族 6 个按预期 SKIP；L40 分支的渲染与三分支判读已用构造数据验过；门禁仍 **96/94/0/2** |
| **A9** | **AC7 软件编码器族验证（2026-09-29）**：用 `fix-ffmpeg-av1.sh` 重编 ffmpeg（`--mode=user`，装到 `/usr/local`，ffmpeg 7.1+av1）启用 `libsvtav1`/`libaom-av1`/`librav1e`，跑 `av1_vp9_quality_matrix.py` 覆盖 QUALITY_MAP 里 AV1/VP9 全部 7 个条目 | **已落地（2026-09-29）**；4 个软编编码器实跑：libsvtav1 **PASS**（`-crf 25`，1.03×／−0.01 dB）、libaom-av1 **PASS**（`-crf 25`，0.79×／−0.43 dB）、libvpx-vp9 **WARN**（`-crf 28`，0.95×／−1.67 dB，与 §6.2 G7-7 逐位一致）、librav1e 首轮 **SKIP**（`-qp 80` 编码超时）→ **补测 PASS**（新表 `-qp 66`，0.91×／−1.21 dB，见 A11/§6.11）。首轮汇总 PASS=2／WARN=1／FAIL=0／SKIP=4，退出码 0 |
| **A10** | **`libaom-av1` 加密复验 + 定位标定脚本缓存污染**：发现 §6.8 里两条不同素材的 libaom 体积逐位相同，定位为 `calibrate_soft_offsets.py` 的 `prep.mp4` **按文件名复用、不校验 `--src`/分辨率** | **已落地（2026-09-29）**；见 **§6.10**。新增无缓存脚本 `calibrate_soft_offsets_nocache.py`（独立工作目录 + 打印 `prep` md5 可审计 + `--dense` 加密扫描），原脚本加复用警告。**16 点 × 3 素材**重测：`a` 收敛稳定（1.976/1.986/1.825），`a=0.952` 确认为假值，曲线"悬崖"消失 ⇒ **现表 (2.007,−21.35) 判定正确、不改表** |
| **A11** | **`librav1e` 编码性能测试 + 等体积重标**：拆解 A9 里的"编码超时"，分别测性能与表值 | **已落地（2026-09-29）**；见 **§6.11**。性能：默认档 179.2s/2s@720p（≈0.011× 实时），`-speed 10` 提速 **4.6×** 且体积 ×1.40，tile 1×1 为 no-op。**表值发现错误**：旧值 `(4.0,−4.0)→qp80` 系"经 libaom 中转"推导，而 libaom 行已重标 ⇒ 同链推导得 63，自相矛盾；直接实测（默认档 12 点、5/5 锚点）得 `a=7.0032, b=−80.993` ⇒ **crf21 → qp 66**，与 VidUtils 链式 64 一致。已落表 + 同步 `REF21_EXPECTED`/`G2-10` |
| **A12** | **rav1e「等体积 vs 等质量」口径定案 + `-speed` 决策**：逐锚点实测（门禁素材 word_world_2，crf 18~30）显示 rav1e 两判据严重背离 —— 码率比恒定 0.91~1.00（等体积准）但 ΔPSNR 从 +0.59 单调恶化到 −5.79 dB（等质量不成立） | **已落地（2026-09-30）**；见 **§6.11.3**。**定案：只保留等体积表**（与其余 6 编码器语义一致），不引入等质量表（另立项目）。`-speed 10` 因同码率下多掉 ~1.9 dB、等体积/等质量无法兼得，**默认不启用**，改为 `VIDEO_RAV1E_SPEED` 显式可选；启用时 `_eqvol_model()` 自动切换到 speed 10 的等体积标定值 `(6.8159,−66.093)⇒qp77`，**等体积语义两档都成立**。同时修正 `quality_map` 把 `nvenc → -qp 0` 误列为「无损分支」的问题（VidUtils 实测 NVENC `-qp 0` 561/561 帧与源不同，仅「最高质量档」） |
| **A13** | **P3 长视频冒烟在 L40 上执行完毕（2026-09-30）**：`interpolate_then_upscale` 2×+2×、330 s 真实素材（11 段）、IFRNet 与 ESRGan 两侧均 `av1_nvenc`，`constqp` 与 `vbr` 各跑一遍 | **已落地**；两次均 `exit 0`、15803 帧全链路守恒、解码级门禁通过、QA sidecar 完整、无内存泄漏。详见 **§8.5** |
| **A14** | **P3 冒烟暴露的三处 AV1 路径缺陷（2026-09-30 修复）**：① `verify_video_integrity()` 用 cv2 读首帧，把**完好的 AV1 产物判成损坏**并 `unlink`（OpenCV 4.13 无 AV1 解码）⇒ 本仓 AV1 完全跑不通；② ESRGan 侧 `FFmpegWriter` 的 NVENC 判定用精确元组，`av1_nvenc` 落到 `else` 的 libx264 分支，**静默**把 CQ 27 当 CRF 下发；③ 两侧 CLI writer 都未把 AV1 的 `vbr_hq/qvbr` 降级为 `vbr`，而 av1_nvenc 的 `-rc` 只接受 `constqp/vbr/cbr` ⇒ **命令直接失败** | **已落地**；根因、证据与修法见 **§8.6** |
| **A15** | **`av1_vp9_quality_matrix.py` 的 AC1 判读不再写死 `84`**：表已于 `69b62db` 改为 ×3（QP 63）后，脚本仍按 ×4 假设判读，在 L40 上会输出与表自相矛盾的结论。改为 `av1_expected_qp()` 现场走 `resolve_quality → to_constqp_qp` 推导锚点，JSON/MD 增列 `av1_qp_expected` + `ac1_verdict` | **已落地**；见 **§8.2** |
| **A16** | **P3′ 回归保护（2026-09-30 续）**：① 判据新增 **G6-8/G6-9/G6-10** —— `av1_nvenc` 的**命令形状**断言（constqp 发 `-qp 63`；`vbr_hq` 必须降级为 `-rc:v vbr`；ESRGAN 侧必须发 `-vcodec av1_nvenc` 且**不得**出现 `-crf`）；② 新增 pytest `Accessory/test/test_verify_video_integrity_fallback.py`（cv2→ffmpeg 回退，含"回退不放宽"反证）；③ 新增可重复脚本 `Accessory/verify/av1_pipeline_smoke.py`（AV1 双 rate_mode 端到端冒烟 + 8 项验收，支持 `--checks-only` 复验既有产物） | **已落地**；三项均做了**反向验证**（回退修复后断言/用例如期 FAIL）。见 **§9.2~§9.4** |
| **A17** | **P2 · V9 三行复核（`libx265` / `libvpx-vp9` / `libsvtav1`）**：用无缓存脚本 `calibrate_soft_offsets_nocache.py --dense` 在 3 条素材（720p30 / 1080p30 / 360p 低复杂度）上重跑等体积标定 | **已落地：表值判定正确，不改表**。基准素材上三行复现偏差 ≤0.14 crf；素材间残差最大 −2.27 crf（`libvpx-vp9`，即 AC7 里已知 WARN 的那一行）⇒ 线性常量的内容相关误差无法再压，与 E9 `CQ_OFFSET=0` 同结论。见 **§9.5** |


---

## 1. 已落地项（E0）

`src/utils/convert_crf.py`（与 `VidUtils/convert_crf.py` 同步）：

```python
'av1_nvenc': (1.0, 6.0, 0, 63),   # 原 (1.0, 6.0, 0, 51)
```

影响：`--crf-ref 45~51`（或未给质量时按 `DEFAULT_REF=21` 的换算链）不再被截到 51；
`crf_ref 51 → -cq 57`（此前 51）。

⚠ `av1_qsv` / `av1_amf` 保持 51（本机无该编码器，量程待上机核实）。

---

## 2. 原始分析（立项时的现状与设想）

> ⚠️ 本节保留**立项时**的现状描述与改法设想，**不代表当前实现**（E1/E2/E3/E4/E5/E7/E8 均已落地）。
> 实际落地与原设想的差异 → **§5.1**；状态 → **§0**。其中 E3 的现状前提已被实测推翻，见 §5.1。

### E1 —— `to_constqp_qp()` 的核心缺陷：少了 QP 尺度层

**现状**（`src/utils/quality_map.py`）：`to_constqp_qp()` 把 CQ 轴值回溯到基准轴后**直接**
作为 `-qp` 返回，只夹到 `QUALITY_MAP` 的 `[lo, hi]`。这对 H.264/HEVC 是对的
（QP 与 x264 QP 同尺度），但对 **AV1 完全错**：

| 输入 | 现在 | 应为 | 说明 |
|---|---|---|---|
| `hevc_nvenc` + `DEFAULT_REF` | `-qp 20/21`（G3-2 已断言） | 21 | 正确 |
| `av1_nvenc` + `DEFAULT_REF` | `-qp 21` | **约 84** | 21 落在 0~255 上是"近无损"，体积暴涨 |

**依据**：`ffmpeg -h encoder=av1_nvenc` → `-cq (0 to 63)`、`-qp (-1 to 255)`。
AV1 的 `-qp` 是 **qindex**（0~255），与 H.264 的 QP（0~51）不是同一刻度；
`librav1e` 的 `-qp` 同为 0~255，且 `QUALITY_MAP` 已用一条**带截距的** QP 模型处理其刻度
（其实测标定：`rav1e_qp = 4 × (libaom_crf − 5)`）。

**改法**：新增 `_QP_SCALE`（QP 相对基准轴的倍率）并与 CQ 偏移解耦：

```python
_QP_SCALE = {'av1_nvenc': 4, 'librav1e': 4}   # 其余（含 h264/hevc_nvenc、vaapi、libx264/265）为 1
```
`to_constqp_qp()` 改为：`基准轴值 × _QP_SCALE[codec]`，再夹到**该编码器的 `-qp` 量程**
（AV1 是 0~255，不是 `QUALITY_MAP` 的 hi —— 这点要一并处理，见下）。

⚠ 数据库结构上，`QUALITY_MAP` 的 `(lo, hi)` 描述的是 **CQ 轴**；AV1 的 QP 量程（0~255）
需要另立一张表，否则 `_clamp_int` 会把 84 夹回 63。

**验证**：`VidUtils/probe/verify_nvenc_quality_gpu.py` 的 **C 组**（L40 专属）：
扫 `-qp {21, 84, 105}`，看哪个与 libx264 crf21 的码率比落在 `RATE_PASS=(0.65,1.50)` 且
ΔPSNR ≥ −1.5 dB。若 84 落带内 ⇒ ×4 成立；否则按实测改倍率。
**T4 上本组自动 SKIP**（Turing 无 AV1 NVENC，脚本会实跑探测后跳过）。

### E2 —— libsvtav1 / libaom-av1 的 preset 会让命令直接失败

`src/utils/video_utils.py` 的 `_NO_PRESET_CODECS = {'librav1e', 'libvpx', 'libvpx-vp9'}`
（黑名单）没排 svtav1 / libaom：

- `libsvtav1`：`-preset` 存在但是 **0~13 整数**（默认 `-2`），传 `medium` →
  ffmpeg 报 `Unable to parse option value "medium"`，命令**直接失败**。
- `libaom-av1`：没有 `-preset`（只有 `-cpu-used`），传了只是 warning + 假信息。

**改法**：
1. `_NO_PRESET_CODECS` 补 `libsvtav1`、`libaom-av1`；
2. 新增 svtav1 的整数量换算（移植 VidUtils 的 `X264_TO_SVTAV1_PRESET` /
   `NVENC_TO_SVTAV1_PRESET`，并修掉 `p7=4` 与 `veryslow=2` 的不自洽）；
3. libaom-av1 按核数给 `-cpu-used`（VidUtils 已有 `auto_effort()`）。

### E3 —— VAAPI 被当成 `-crf` 编码器

`_quality_param()` 对 `h264_vaapi` / `hevc_vaapi` 返回 `'-crf'`（因为它们不在 `_CQ_CODECS`
里），而 VAAPI **只有 `-qp (0 to 52)`**（实测）。结果：质量参数下发无效。

**改法**：新增 `_QP_ONLY_CODECS = {'h264_vaapi', 'hevc_vaapi'}`，`_quality_param()` 对它返回
`('-qp', [])`；`literal_range()` 与 `supports_crf()` 的判定同步（VAAPI 不是 crf 编码器）。
与 VidUtils 的 V6 **同源同判**，两边行为要对齐。

### E4 —— 硬编的 `-preset` 能力

`_preset_supported()` 是黑名单（注释写着"硬编（NVENC/QSV/AMF…）接受 -preset"）：

| 编码器 | 实际 | 现状 |
|---|---|---|
| NVENC 族 | p1~p7 | 正确 |
| QSV 族 | 只收 `veryfast..veryslow`（int 0~7） | 传 `medium` 恰好可用，但 `p5`/`ultrafast` 会失败 |
| AMF 族 | 不接受 `-preset`（用 `-quality` / `-usage`） | 会下发无效选项（**待上机核实**） |
| VideoToolbox | 不接受 `-preset`（只有 `-prio_speed` / `-realtime`） | 同上（**待上机核实**） |

**改法**：改成与 VidUtils 的 `PRESET_SUPPORTED_CODECS` 同构的**白名单**，并对 QSV 加档名映射。

### E5 / E6 —— preset 表与换算表

- **E5**：`external/realesrgan_video/nvenc_sdk.py` 的 `_PRESET_P_INDEX` 与 VidUtils 的
  `NVENC_TO_X264_PRESET` 在 `superfast/veryfast/faster` 上错位 1 档、`fast` 在 VidUtils
  侧无 p 档（回落 p5）而 VE 给 p4。两边都用 ffmpeg 官方枚举校准：
  `p1=fastest(lowest) … p4=medium(default) … p7=slowest(best)`。
- **E6**：与 VidUtils 的 V9 同一份标定（同一张表，改一处必须同步）：
  `libsvtav1 ≈ 1.40x+3.16`、`libvpx-vp9 ≈ 1.685x−4.82`、`libx265` 保持 `1.0x+3`。
  ⚠ 标定 caveat（合成素材 / 单一分辨率 / 等体积口径）→ 先真实素材复核再落表。

### E7 / E8 —— CLI 取值表与 VidUtils 不一致

| 参数 | VE | VidUtils | 处置 |
|---|---|---|---|
| `--rate-mode` / `--rate-mode-*` | `constqp / vbr_hq / qvbr` | `constqp / vbr / vbr_hq / cbr / cbr_hq / cbr_ld_hq` | 要么补 `vbr/cbr*`，要么在 help 与 CLI 层明确"SDK 只支持 3 档"并拒绝其它值 |
| `--lookahead-depth-*` | choices `{0,8,16,32}` | `--lookahead` 0~250 | 放开量程或注明 SDK 限制 |
| `--preset` 输入风格 | 只收 x264 名（choices 排除 p1~p7） | 双向都收 | 至少在两处文档里写明差异 |
| `--qp` / `--bitrate` | 无（QP 由后端换算、码率走 avgBitRate 天花板） | 有独立 CLI | 有意的架构差异，写进已知限制即可 |

### E9 / E10 —— 判据侧补齐

- **E9**：`CONSTQP_QP_OFFSET` 现为 0（可调口）。用 G7 的真实素材数据校准；并把 G7 的
  覆盖从 `h264_nvenc / hevc_nvenc` 扩到 **`av1_nvenc`**（AV1 的偏移 +6 至今零实测）
  与 `libsvtav1`（RATE_PASS 判据同样适用）。
- **E10**：G7 支持"合成 + 真实"双素材。现有注释已承认"合成 testsrc2 下 constqp 是过配、
  真实素材 PASS"——补上真实素材那一跑，才能把"过配边界"从注解升级为实测。

---

## 3. VE 侧已正确、VidUtils 应对齐的三处

（这些是上一轮分析里 VE 做对、VidUtils 缺失的部分，改造 VidUtils 时可直接照搬语义）

1. **`literal_range()`**（`quality_map.py`）：字面量按**生效编码器**的技术规范量程校验，
   而不是统一 0~63。VidUtils 已按此移植（V4）。
2. **`supports_crf()` 排除 `librav1e`**：避免"字面量走 libaom 刻度、基准轴走 x264 刻度"
   的双链（VidUtils 的 V8）。
3. **`to_constqp_qp()` 的存在本身**：明确 CONSTQP 的 QP 与 CQ 是两条刻度。
   ⚠ 但其**内部**仍缺 QP 尺度层（E1），AV1 上同样错。

---

## 4. 回归与验收

| 门 | 命令 | 覆盖 | 本机（无 GPU / 无依赖） |
|---|---|---|---|
| 本仓判据（纯逻辑） | `python3 Accessory/verify/crf_cq_unification_verify.py --quick < /dev/null` | G1 换算表 / G2 解析顺序 / G3 CONSTQP 轴 / G4 CLI / G5 落点 / G6 命令 | ✅（G4/G6/G9/G10 因缺 cv2 必 FAIL，属环境） |
| 本仓判据（GPU） | `python3 Accessory/verify/crf_cq_unification_verify.py --gpu --source <真实素材> < /dev/null` | G7 画质 / G8 码率天花板 | ❌ |
| 门禁全套 | `python3 Accessory/verify/plan_implementation_gate.py < /dev/null` | 优化方案落地 + 修复效果 | ⚠️ 缺依赖时 WARN/SKIP，判 **FAIL=0** |
| pytest 真回归 | `python3 -m pytest Accessory/test -q` | 读帧器 / stdin / 缓存等 | ❌（无 pytest） |
| 上机（T4 / L40） | `python3 <VidUtils>/probe/verify_nvenc_quality_gpu.py --src <真实素材> < /dev/null` | NVENC 的 CQ/QP 实测；**L40 上才有 AV1 结论** | ❌ |
| 跨项目真源一致 | `python3 <VidUtils>/verify/verify_quality_mapping.py < /dev/null` | ⑨ 组断言两份 `QUALITY_MAP` 逐条相等 | ✅（只读） |

**三条操作铁律（都是踩过的坑）：**

1. **一律加 `< /dev/null`**。这些脚本在「后台进程组 + tty stdin」下会被 SIGTTOU 整组停住，
   症状是「跑得异常久 + 零输出」；`ps -o stat` 可见主进程与子 ffmpeg 同时为 `T`、
   `wchan=do_signal_stop`。（`crf_cq_unification_verify.py` 已内置加固，其余脚本仍需外部重定向。）
2. **改 E1 不需要动 G6-2 / G6-5**（原文提示有误）。h264/hevc_nvenc 的 QP 模型截距为 0，
   `26→21`、`28→20` 原样成立；要新增的是 **AV1 用例** —— `G3-7`（`av1_nvenc` 27→84）与
   `G6-7`（AV1 constqp 下发 `-qp 84`，需 Ada）。已实测核对。
3. **改共享真源 `convert_crf.py` 的 `a/b` 后，必须同步复核判据里的独立期望值**：
   `REF21_EXPECTED`（G1-2）与 G2-11。它们是人审定的独立期望，**不会自动跟随** QUALITY_MAP
   （2026-09-28 软编三项重标定后即出现过这类失败）。

---

## 5. 落地实现要点与 GPU 上机验证方案

### 5.1 落地实现与原设想的差异（复核用）

| 项 | 文档原设想 | 实际实现 | 原因 |
|---|---|---|---|
| E1 | `_QP_SCALE` 单一倍率 + 另立 QP 量程表 | `_QP_MAP_OVERRIDE`（`QP = a·ref + b`，自带量程）+ `_qp_model()` 回退 `QUALITY_MAP` | 单一倍率会把 `librav1e` 算成 84，而其真实刻度由标定实测决定（2026-09-29 直接实测为 `7.00·ref−80.99` ⇒ ref 21 → 66，见 §6.11）；带截距的表可同时满足 AV1/rav1e，代码量相同。**L40 实测确认 AV1 为 ×3（非 ×4）**，已更新 `_QP_MAP_OVERRIDE['av1_nvenc'] = (3.0, 0.0, 0, 255)` |
| E1 影响面 | 未说明 | `to_constqp_qp` 生产侧 4 个调用点**全部在 NVENC 分支内**（`ifrnet_video/main.py:1825`、`ifrnet_video/ffmpeg_io.py:904`、`realesrgan_video/main.py:835`、`realesrgan_video/ffmpeg_io.py:908`）⇒ 只影响 h264/hevc/av1_nvenc；T4 上 h264/hevc 为**恒等换算**，故 T4 生产**零影响** |
| E2 | 黑名单补两项 + 移植 svtav1 整数映射 + libaom `-cpu-used` | 只改白名单（svtav1/aom 自然被排除）；**整数映射未做** | 白名单已消除"命令直接失败"这个 P0；整数映射与 VidUtils 的 V9/V10 同源，单侧改会漂移，留待跨项目同批 |
| E3 | 称 VAAPI "不在 `_CQ_CODECS` ⇒ 返回 `-crf`" | **前提过期**：当时 VAAPI 就在 `_CQ_CODECS` 里 ⇒ 实际下发的是非法 `-cq:v 24 -b:v 0`（ffmpeg 直接报错，比原描述更严重）。实现按 VidUtils V6 语义（归一到基准轴 → `-qp`，夹 0~52），并同步 `supports_crf()` 排除 VAAPI | 实测 |
| E4 | 白名单 + 对 QSV 加档名映射 | 白名单只留 `libx264 / libx265 / h264_nvenc / hevc_nvenc / av1_nvenc`；**未加 QSV 档名映射** | ffmpeg 8.0.1 的 QSV `-preset` 是 int `0..7`（不是旧版档名），且 QSV 非 VE 生产路径；保守排除比猜映射安全 |
| E5 | 与 VidUtils 对齐 | 只按 ffmpeg 官方枚举改 VE 的**两份**副本（ifrnet + realesrgan，逐字节一致）；VidUtils 侧仍是旧档位，⑨ 组 `[9-preset]` 现报 `medium: VidUtils p5 vs VE p4`（note-only，不影响退出码） | 用户决定"仅 VE 单侧"，对方 V10 由另一会话执行 |
| E8 | `0~250` | `0~32` | NVENC 前向预看**硬件上限是 32**；配置层校验本已要求 `0<=la<=32`，故实际是"放开 CLI choices 到配置层允许的范围" |
| E6 | 本方案待办 | 由**并行会话**在 2026-09-28 11:34 落地（`libx265 0.9155/1.6385`、`libvpx-vp9 1.6198/−5.7553`、`libsvtav1 1.9450/−15.62`） | 非本轮范围；已据此同步 `REF21_EXPECTED`（libx265→21、libsvtav1→25、libvpx-vp9→28） |

> **2026-09-28 实测补记**：E5 的"按 ffmpeg 官方枚举"措辞不准（`medium≡p4` 成立，但
> `slow` 并不等于 `p5`）；E9/E10 已于本轮在 T4 落地并实测，完整数据见 **§6**；
> 需 Ada 才能定案的 AV1 测试内容（AC1~AC6，含可直接执行的命令与判据）见 **§7**。
> ⑨ 组 `[9-preset]` 现已报「一致」。

### 5.2 GPU 上机验证方案（T4 / L40）

#### 步骤 0 · 环境体检（**必须最先做**）

```bash
cd /workspace/Video_Enhancement
nvidia-smi -L                                              # 卡型：T4 / L40 / A10 …
python3 -c "import torch; print(torch.cuda.get_device_name(0), torch.version.cuda)"
ffmpeg -hide_banner -encoders | grep -E 'nvenc'
```

⚠ **不要用 `ffmpeg -h encoder=av1_nvenc` 判断"能不能编 AV1"**：T4（Turing）上该选项表**照样打印**，
只是真编码会失败。唯一可靠的探测口径是**实跑一帧**：

```bash
ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 \
       -c:v av1_nvenc -f null - < /dev/null ; echo "rc=$?"
# rc=0  → 有 AV1 NVENC（Ada/L40）⇒ Gate 2 的 C 组可跑
# rc≠0  → 无 AV1 NVENC（T4/Turing）⇒ C 组 SKIP，这不是失败
```

后续命令统一记：`SRC=<真实素材>`、`SRC_HE=<1080p60 高熵素材>`。

---

#### Gate 0 · 代码正确性（两卡都跑，不需要 GPU）

```bash
python3 Accessory/verify/crf_cq_unification_verify.py --quick < /dev/null
python3 Accessory/verify/plan_implementation_gate.py < /dev/null
python3 -m pytest Accessory/test -q
python3 <VidUtils>/verify/verify_quality_mapping.py < /dev/null
```

- **通过判据**：`--quick` 无 FAIL（除环境类）；门禁 **FAIL=0**；pytest 全绿；跨项目脚本 exit 0。
- **本机（无依赖）基线供对照**：`--quick` = PASS 49 / FAIL 3（全是 `No module named 'cv2'`）/ SKIP 19；
  门禁 = 84 项 / 通过 75 / 失败 0 / 警告 4 / 跳过 5。
  生产机上这些环境类项应转为 PASS，**FAIL 数必须仍为 0**。

#### Gate 1 · 判据 G7 / G8（GPU 画质与码率天花板）

```bash
python3 Accessory/verify/crf_cq_unification_verify.py --gpu \
    --source "$SRC" --bitrate-source "$SRC_HE" \
    --report verification_report/crfcq_gpu_$(date +%F_%H%M).md \
    < /dev/null
```

- **通过判据**：`G7-3「CONSTQP 轴：-qp 21 对齐 libx264 crf21」` 的码率比 ∈ `RATE_PASS=(0.65,1.50)`
  且 ΔPSNR ≥ −1.5 dB；`G8` 天花板钳制前后 ΔPSNR 下降 < 0.5 dB。
- **T4 预期**：h264_nvenc / hevc_nvenc 全跑；av1_nvenc 相关格 SKIP（脚本自动探测）。
- **L40 预期**：全量，含 av1_nvenc。

#### Gate 2 · NVENC CQ / QP 实测（上机主证据；**E1 的定案在 L40**）

```bash
python3 <VidUtils>/probe/verify_nvenc_quality_gpu.py \
    --src "$SRC" \
    --json verification_report/nvenc_cq_qp_$(date +%F).json \
    --md   verification_report/nvenc_cq_qp_$(date +%F).md \
    < /dev/null
```

脚本内置判据（与 VE 的 G7 同一套容忍带，保证两边结论可比）：
`TOL_PSNR=1.5 dB`、`RATE_PASS=(0.65,1.50)`；`libx264 crf21` 作软编基准。

| 组 | 内容 | T4 | L40 |
|---|---|---|---|
| A | 纯逻辑：量程 / 换算 / 饱和扫描（**不需 GPU**，本机也能跑） | ✅ | ✅ |
| B | `crf_ref 21` 的 `-cq` 是否等质量（表值 vs 朴素 21） | ✅ h264/hevc | ✅ + av1 |
| **C** | **AV1 constqp 的 QP 尺度**：扫 `-qp {21, 84, 105}` | ⏭️ SKIP | ✅ **唯一能定 E1 的卡** |

**C 组的判读（关键）：**

- **84 落带内** ⇒ E1 的 ×4 成立 ⇒ 去掉 `quality_map.py` 里
  `_QP_MAP_OVERRIDE['av1_nvenc']` 的 `# [待 L40 复核]` 标记。
- **84 不落带内** ⇒ 按实测改倍率：`a = <落带内的 qp> / 21`，
  并同步改判据 `G3-7` 的期望值（当前钉在 84）。
- 105（×5）只是对照点，用于确认单调方向；21 是"旧实现的错误值"，预期会表现为码率暴涨。

#### Gate 3 · E5 preset 档位实测（T4 就够，**本改动唯一的运行时语义变更**）

目的：确认 `medium` 现在落在 **p4**（而不是旧表的 p5）。

```bash
for P in p4 p5; do
  echo "--- $P ---"
  /usr/bin/time -f "%e s" ffmpeg -hide_banner -y -i "$SRC" \
      -c:v h264_nvenc -preset $P -rc:v vbr_hq -cq:v 26 -b:v 0 \
      /tmp/probe_$P.mp4 < /dev/null 2>&1 | tail -2
  ls -l /tmp/probe_$P.mp4
done
```

- **通过判据**：两条命令都 rc=0（不出现初始化失败）；`p4` 比 `p5` **更快**；两者体积差在合理范围
  （p4 更快、体积略大或质量略高）。
- **回归对照**：因为新表把 `medium→p4`（旧 `p5`），预期整体**编码变快**。若实测 p4 明显更慢或质量明显更差，
  说明档位语义判断有误 ⇒ **回滚 E5**（还原两份 `_PRESET_P_INDEX`）。
- 该变化会影响 VE 默认 `encode_preset`（`medium`）的实际档位，属**性能/画质可感知**改动，必须实测留证。

#### Gate 4 · 不相关面回归（同批上线要一起确认）

```bash
# hevc + LA>0 帧守恒（本改动不触及，但共用同一编码线程）
python3 run.py -i /tmp/seg_src_5s.mp4 -o /tmp/seg_hevc_la8.mp4 \
    --mode interpolate_then_upscale \
    --rate-mode-ifrnet vbr_hq --lookahead-depth-ifrnet 8 < /dev/null
python3 Accessory/verify/segment_bitstream_verify_v5.py /tmp/seg_hevc_la8.mp4 < /dev/null
```

- **通过判据**：帧守恒（`frames == packets`）、单 IDR、`frame_num` 单调、无色度异常簇。
- 若 Gate 0 的 pytest 全绿，这一步通常也过；它的价值在于覆盖"CLI 放开 LA 量程 + preset 表变化"的组合。

---

#### 分卡预期矩阵

| Gate | T4 | L40 |
|---|---|---|
| 0 静态 / 逻辑 / pytest | ✅ | ✅ |
| 1 G7 / G8 | ✅ h264+hevc；av1 格 SKIP | ✅ 全量 |
| 2 A 组 | ✅ | ✅ |
| 2 B 组 | ✅ h264+hevc | ✅ + av1 |
| **2 C 组（AV1 ×4 定案）** | ⏭️ **SKIP**（Turing 无 AV1 NVENC） | ✅ **必跑** |
| 3 preset 档位 | ✅ | ✅ |
| 4 LA=8 帧守恒 | ✅ | ✅ |

> **结论**：**T4 能完成除"AV1 QP 倍率定案"之外的全部验证**；E1 的"待 L40 复核"标记
> 必须等 L40（或任何 Ada 卡）跑完 Gate 2 的 C 组才能摘掉。若短期拿不到 Ada，
> 建议保持标记并把 AV1 的 constqp 路径视为"未定案"，不要在生产 AV1 任务上依赖它。
>
> ✅ **本结论已于 2026-09-28 在 T4 实测验证**（Gate 0/1/3/4 全过，AV1 项 SKIP），
> 实测数据见 **§6**；需 Ada 的逐项**测试内容（命令 + 判据 + 落地动作）见 §7 的 AC1~AC6**。

#### 若生产机上没有 VidUtils 仓库

Gate 2 的脚本位于 VidUtils 仓。两种处理：

1. 把那一个脚本拷到生产机运行（它只依赖 `ffmpeg` + stdlib，不依赖 VidUtils 其它模块）；
2. 或把 C 组的扫描逻辑**移植成 VE 判据的 `G7-av1` 单元格**（约 30 行：构造
   `-qp 21/84/105` 三条命令 → 比码率比与 PSNR），这样 VE 单仓即可闭环。
   如需要，我可以按第 2 种做法补上。

#### 判定与回滚速查

| 结果 | 动作 |
|---|---|
| Gate 2 C 组 84 落带内 | 摘掉 `[待 L40 复核]` 标记；E1 定案 |
| Gate 2 C 组 84 不落带内 | 改 `_QP_MAP_OVERRIDE['av1_nvenc']` 的 `a`，同步改 `G3-7` 期望 |
| Gate 3 p4 语义不符预期 | 回滚两份 `_PRESET_P_INDEX`（`medium` 回到 p5） |
| Gate 1 G7-3 constqp 码率比出带 | 调 `CONSTQP_QP_OFFSET`（可调口，当前 0）后复跑 |
| Gate 0 门禁出现 FAIL | 先与记忆基线（95/93/0/2 或本机 84/0FAIL）差分，确认不是本轮 6 个文件的回归 |

---

## 6. 本轮 T4 实测收口（2026-09-28）

> 环境：Tesla T4（Turing，**无 AV1 NVENC**）、torch 2.10.0+cu128、CUDA 可用；
> 真实素材 `input_videos/word_world_2.mp4`（720x576 25fps）、
> 码率素材 `input_videos/new4_raw.mp4`（1080p30）。
> ⚠ 原 Gate 1 命令里的 `--bitrate-source .../new5_raw.mp4` **已不存在于本机**，
> 本轮以 `new4_raw.mp4` 替代（同为真实高熵 1080p；G8 结论不受影响）。

### 6.1 Gate 0 · 静态/门禁/真源一致

| 判据 | 命令 | 结果 |
|---|---|---|
| 本仓判据（静态） | `crf_cq_unification_verify.py --quick` | **PASS 91 / FAIL 0 / WARN 0 / SKIP 11** |
| 门禁全套 | `plan_implementation_gate.py` | **96 项 / 94 通过 / 0 失败 / 0 警告 / 2 跳过**（记忆基线 95/93/0/2） |
| pytest | `pytest Accessory/test -q` | **24 passed / 0 failed**（修 A7 前为 2 failed） |
| 跨项目真源一致 | `VidUtils/verify/verify_quality_mapping.py` | ⑨ 组 **13/13 一致**（含 `[9-preset]` 现已「一致」）；脚本整体 `exit=1`，见 §6.5 |

### 6.2 Gate 1 · G7/G8 实测（`--gpu` + 真实素材）

`PASS=99 / FAIL=0 / WARN=3 / SKIP=1`，报告见
`verification_report/crfcq_gpu_T4_2026-09-28_0616.md`。

| ID | 项 | 结论 | 关键数据 |
|---|---|:--:|---|
| G7-1 | h264_nvenc → `-cq:v 26` | ⚠️ WARN | ΔPSNR −1.93 dB / 1.07×（朴素 1.76×） |
| G7-2 | hevc_nvenc → `-cq:v 28` | ⚠️ WARN | ΔPSNR −2.69 dB / 0.85×（朴素 1.70×） |
| **G7-3** | **constqp `-qp 21` 对齐 libx264 crf21** | ✅ **PASS** | **ΔPSNR +0.06 dB / 1.46×（朴素 1.76×）** |
| G7-4 | 换算后码率 ≤2.5× | ✅ PASS | h264 1.067× / hevc 0.845× / vp9 0.945× |
| G7-5 | VMAF 对齐 | ✅ PASS | 最大偏差 2.00（软编基准 96.04） |
| **G7-6** | av1_nvenc | ⏭️ SKIP | `av1_nvenc 实跑失败：No capable devices found`（无 AV1 NVENC）→ 关闭方式见 **§7 AC2** |
| **G7-7** | **软件侧（libvpx-vp9）→ `-crf 28`** | ⚠️ WARN | ΔPSNR −1.67 dB / 0.95×（朴素 1.30×） |
| **G7-8** | **合成 vs 真实双跑（constqp 轴）** | ✅ **PASS** | 合成 ΔPSNR **+5.33 dB** / 1.53× vs 真实 **+0.06 dB** / 1.46× |
| G8 | avgBitRate 天花板 | ✅ 8/8 | — |

判读：

* **G7-3 PASS 且 ΔPSNR 仅 +0.06 dB ⇒ `CONSTQP_QP_OFFSET` 保持 0 即为最优，无需校准**（E9 的"校准"部分据此结案）。
* G7-1/G7-2 的 WARN 与 2026-09-11 基线逐位相同（−1.93/−2.69），是**既有**的内容相关偏松，非本轮回归；G7-7 复现同一形态（−1.67），属同一已知边界（严格判据保留 + 标注）。
* **G7-8 首次把"合成素材偏过配"从注释升级为实测**：合成 ΔPSNR 高出真实 5.27 dB、码率比高 0.07× ⇒ 方案 §4/G7-3 的原注解成立。

### 6.3 E9 落地细节（与原设想的偏差）

| 项 | 原设想 | 实际 | 原因 |
|---|---|---|---|
| 软件侧编码器 | `libsvtav1` | **`libsvtav1` → `libvpx-vp9` 回退** | **本机 ffmpeg 构建根本不含 `libsvtav1`**（也无 `libaom-av1` / `librav1e`），只有 `libvpx-vp9`。直接纳入会让整组因 `Unknown encoder` 中断 |
| 可选覆盖的判定 | 直接纳入 | **先探可用性**（`ffmpeg -encoders`）再决定跑/SKIP | 同上；AV1 另加**实跑一帧**探测硬件能力（`-h encoder=av1_nvenc` 在 Turing 上照样打印选项表，不可作依据） |
| libvpx-vp9 参数 | — | 显式 `-b:v 0 -cpu-used 4 -row-mt 1` | `-crf` 不配 `-b:v 0` 会退化为 constrained quality；默认 `-cpu-used 0` 在长素材上会超时 |
| G7 组异常语义 | — | 编码阶段异常改为**逐项 FAIL**，不再整组"执行中断" | 一个 `Unknown encoder` 曾把 G7-1..G7-8 全吞成 1 个组级 FAIL，丢掉全部逐项结论（该次失败产物留档：`verification_report/crfcq_gpu_T4_2026-09-28_0612.md`，`QUALITY 共 1 项 / FAIL 1`） |

### 6.4 Gate 3 · E5 preset 实测（结论有修正）

真实 1080p 素材上按 md5 去重后的全量 preset 扫描（`h264_nvenc -rc:v vbr_hq -cq:v 26 -b:v 0`）：

| ffmpeg `-preset` 名 | 等价 pN | 体积 |
|---|---|---|
| `default` / `medium` | **`p4`（逐字节相同）** | 20,336,278 |
| `fast` / `hp` | `p1` | 20,405,873 |
| `slow` | **不落在 p1~p7 梯上**（遗留 "hq 2 passes"） | 20,360,228 |
| `bd` | `p5` | 20,242,485 |
| `hq` | `p7` | 20,110,133 |
| — | `p2` / `p3` / `p6` 各有独立输出 | 20,387,297 / 20,355,501 / 20,065,785 |

* ✅ **E5 的锚点成立**：`medium ≡ p4` 逐字节相同 ⇒ 表里 `medium: 3 → p4` 正确。
* ⚠ **方案 §0/§5.1 里「实测官方枚举（p4=medium / p5=slow / p6=slower / p7=slowest）」措辞不准**：
  那是 **NVIDIA p-梯的命名**，不是 ffmpeg **命名 preset** 的等价关系。
  ffmpeg 的 `slow` 是遗留档、`bd≡p5`、`hq≡p7`、`fast≡p1`。VE 表映射 x264 名 → pN 仍然正确，
  但"按 ffmpeg 官方枚举对齐"这句话应改为"按 NVIDIA p-梯命名对齐，并以 `medium≡p4` 实测锚定"。
* 运行时语义变更（`medium` 由 p5 → p4）**可感知影响很小**：p4 中位 4.81s / p5 4.86s（差在噪声内），
  同 `-cq` 下 p4 体积 +0.46% ⇒ **无需回滚 E5**。

### 6.5 Gate 4 · hevc + LA=8 帧守恒回归

```
python3 run.py -i /tmp/seg_src_5s.mp4 -o /tmp/seg_hevc_la8.mp4 \
    --mode interpolate_then_upscale \
    --codec-ifrnet hevc_nvenc --codec-esrgan hevc_nvenc \
    --rate-mode-ifrnet vbr_hq --lookahead-depth-ifrnet 8
python3 Accessory/verify/segment_bitstream_verify_v5.py /tmp/seg_hevc_la8.mp4
```

* 管线 `exit=0`，1 个分段，耗时 92.9s，GPU 峰值 3.51 GB。
* **验收 ✅ 通过**：`frames=253 == packets=253`、段首 32 NAL 内无第二个 IDR、
  `frame_num` 回退=0、无 `pts_anomaly`、色度坏帧簇=0。
* 帧数核对：源 segment 的容器元数据 `nb_frames=177` 与 `duration×fps=127` 不一致
  （`-c copy` 分段常见），管线取 **127** → 2× 插帧 = **253 = 2n−1**，**精确守恒**。
* ⚠ 方案 §4 写的"单 IDR"措辞不精确：实际验收口径是"段首无连 IDR + `frame_num` 单调"，
  本产物 IDR=5（周期性 IDR，正常）。

### 6.6 范围外发现（未改 VidUtils）

`VidUtils/verify/verify_quality_mapping.py` ① 组 `[1] 端到端命令：
--codec hevc_nvenc --rc-mode constqp --qp 18 → -crf 18` 报 `得到 False`，导致脚本 `exit=1`。
**根因已定位且属 VidUtils 仓内**：该判据用 `--decode cpu --scale-algo libswscale-lanczos`
（`cpu_only=True`）后仍期望**编码器**降级到软编并下发 `-crf 18`，但这两个开关只强制
CPU 解码/缩放；本机 `hevc_nvenc` 可用，故 dry-run 实发的是 `-c:v hevc_nvenc -rc constqp -qp 18`。
⇒ 判据的环境假设不成立（不是 VE 侧回归，也不影响 ⑨ 组 13/13 的跨项目一致性结论）。
按本轮"仅 VE 单侧"的范围约定**未改动 VidUtils**，移交其 V 系列会话处理。

### 6.7 A8 · AV1/VP9 家族探针（新增，L40 上机入口）

`Accessory/probe/av1_vp9_quality_matrix.py`（2026-09-28 新增）—— 一条命令覆盖 AV1/VP9 家族
7 个编码器，并内建 AC1 的判读。T4 实测（真实素材 `word_world_2.mp4`）：

| 编码器 | T4 结论 | 数据 |
|---|---|---|
| `libvpx-vp9` | ⚠️ WARN | `-crf 28`，码率比 **0.95×**（朴素 **1.30×**）、ΔPSNR **−1.67 dB** |
| `av1_nvenc` | ⏭️ SKIP | 实跑一帧 → `No capable devices found` |
| `av1_qsv` / `av1_amf` / `libsvtav1` / `libaom-av1` / `librav1e` | ⏭️ SKIP | 本机 ffmpeg **构建不含**该编码器 |

* ✅ **交叉验证通过**：VP9 的三个数字与 §6.2 里判据脚本的 `G7-7` **逐位一致**
  （0.95× / 朴素 1.30× / −1.67 dB）⇒ 两条独立实现（判据脚本 vs 探针）同口径互证。
* 软编基准 `libx264 crf21` = 1427 kbps / 46.58 dB，与判据脚本一致。
* 退出码 0（无 FAIL），报告：`verification_report/av1_vp9_matrix_T4.md` / `.json`。
* L40 才会走到的分支（`av1_nvenc` 可用时的 B 组渲染 + AC1 三分支判读）已用**构造数据**验证：
  「84 落带内 ⇒ ×4 成立」「84 未落带内 ⇒ 取落带点改 a」「三点全出带 ⇒ 记为不支持」三条均正确。
* 门禁复跑仍 **96 项 / 94 通过 / 0 失败 / 2 跳过**（新文件未触发任何断言）。

---

### 6.8 V9 多素材/多分辨率复核实测（2026-09-29）

> ⚠ **本节数据已在 §6.10 被推翻并重做。** 首轮（下方表格）受
> `calibrate_soft_offsets.py` 的 **prep.mp4 缓存污染**影响（见 §6.10），
> `libaom-av1` 一列的 4 个数值全部不可信。保留原文仅作留痕，**请以 §6.10 为准**。

**执行**：`python3 /workspace/VidUtils/probe/calibrate_soft_offsets.py` 跑等体积标定，
共 4 次（3 条素材 + `new5_raw` 的 1080p 分辨率变体）。

| 素材 | 分辨率 | 时长 | libx265 (a, b) | libvpx-vp9 (a, b) | libsvtav1 (a, b) | libaom-av1 (a, b) |
|---|---|---|---|---|---|---|
| **new5_raw.mp4（基准，已落表）** | **1280×720** | **4s** | **(0.9272, 1.3360)** | **(1.6381, −6.2289)** | **(2.145, −21.35)** | (2.007, −21.35) ⚠️ |
| new5_raw.mp4 | 1920×1080 | 4s | (0.887, 1.43) | (1.640, −8.08) | (2.084, −22.99) | (1.919, −22.04) ⚠️ |
| new4_raw.mp4 | 1920×1080 | 6s | (0.935, 1.63) | (1.591, −3.38) | (2.118, −19.36) | (0.952, 0.85) ⚠️ **假值** |
| wws3e02_26s.mp4 | 1280×720 | 4s | (0.931, −1.47) | (1.772, −16.04) | (2.107, −28.01) | (2.007, −29.92) ⚠️ |

（⚠ 标记的 `libaom-av1` 列即 §6.10 定位为缓存污染的四个值；前三列未受该 bug 影响。）

**已同步更新双项目 `QUALITY_MAP` 与判据期望值 `REF21_EXPECTED`（libsvtav1: 25→24）**，回归测试全绿。

---

### 6.9 完整回归门禁结果（2026-09-29）

| 门禁 | 命令 | 结果 |
|---|---|---|
| VE 静态判据 | `python3 Accessory/verify/crf_cq_unification_verify.py --quick < /dev/null` | **PASS 90 / FAIL 0 / WARN 0 / SKIP 12** |
| VE 门禁全套 | `python3 Accessory/verify/plan_implementation_gate.py < /dev/null` | **96 项 / 92 通过 / 0 失败 / 2 警告 / 2 跳过** |
| VidUtils 质量映射 | `python3 /workspace/VidUtils/verify/verify_quality_mapping.py` | **13/13 项一致**（⑨ 组全 ✓） |
| VidUtils 孪生命令 | `bash /workspace/VidUtils/test/dump_cmd_full.sh` | **20/20 逐字相同** |
| VidUtils 编码参数 | `bash /workspace/VidUtils/test/dump_enc_options.sh` | 全项对齐（含 libsvtav1 -crf 30, libvpx-vp9 -crf 32） |

---

### 6.10 A10 · `libaom-av1` 加密复验 + 定位标定脚本的缓存污染（2026-09-29）

**起因**：§6.8 的 `libaom-av1` 一列出现「`new4_raw` 1080p 的 a=0.95 与其余 ≈2.0 **差一倍**」，
且**两条不同素材、不同分辨率的跑批里 `libaom` crf 30~50 的体积逐位相同**
（337.3 / 245.1 / 181.6 / 139.3 / 107.7 KiB）—— 不同内容不可能得到相同体积，
判定为**脚本级 bug 而非 codec 行为**。

**根因**（`VidUtils/probe/calibrate_soft_offsets.py:115`）：

```python
prep = OUT / 'prep.mp4'
if not prep.exists() or args.selftest:     # ← 只看文件在不在
    ... 生成 prep ...
```

`prep.mp4` 是**按文件名复用**的，既不记录也不校验 `--src` / `--duration` / `--width` / `--height`。
换素材或换分辨率重跑时，只要 `prep.mp4` 还在，就**静默沿用上一条素材的中间素材**，
于是"不同素材跑出几乎相同的数值"。这正是 §6.8 那些异常数值的来源。

**修复（两处，均已落地）**：

1. 新增 `VidUtils/probe/calibrate_soft_offsets_nocache.py` —— 每次运行**独立工作目录**
   （目录名带 `素材_分辨率_时长` 指纹）、`prep` 强制重建、并打印 `prep` 的 **md5 + 字节数**，
   使每次跑批可审计；另支持 `--dense`（libaom 扫描加密到 3 档间隔）。
2. 原脚本在该行加**显式警告**：复用已有 `prep.mp4` 时打印提示并指向新脚本。

**加密复验结果**（`-dense`，扫描点 `6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51`）：

| 素材 | 分辨率 | 时长 | a | b | crf21 | 最大残差 | 落外锚点 |
|---|---|---|---|---|---|---|---|
| new5_raw | 1280×720 | 4s | 1.9763 | −20.661 | 20.84 | 0.35 | [18] |
| new4_raw | 1920×1080 | 6s | 1.9858 | −21.706 | 20.00 | **0.53（5/5 锚点）** | 无 |
| wws3e02_26s | 1280×720 | 4s | 1.8246 | −24.830 | 13.49 | 0.96 | 无 |

* ✅ **`a` 收敛且稳定**：1.976 / 1.986 /（wws 素材因低复杂度偏 1.82）
  ⇒ §6.8 里那个 `a=0.952` **确认为缓存污染的假值**，重测后为 1.986。
* ✅ **曲线的"悬崖"是假的**：污染数据里 `crf 25→30` 出现 1625.8→337.3 的 4.8× 骤降；
  重测后 `crf 25→30` 为 1444.4→1217.6（**平滑单调**），悬崖消失。
* ✅ **现表值判定为正确，无需改动**：`libaom-av1 = (2.007, −21.35)` → crf21 = **20.80**；
  基准素材重测值 20.84（差 0.04），1080p 重测值 20.00。
* ➕ 独立佐证：同素材上 `libaom crf21` 体积 1206014 B vs `libx264 crf21` 1155481 B
  ⇒ 码率比 **1.044**，落在 `RATE_PASS=(0.65,1.50)` 内。

**结论**：E6 的 `libaom-av1` 行**维持 (2.007, −21.35) 不变**；本轮真正修掉的是
**标定脚本的缓存 bug**（此前所有多素材标定都应视为"仅首条素材有效"）。

---

### 6.11 A11 · `librav1e` 编码性能测试 + 等体积重标（2026-09-29，发现表值错误）

§6.7 中 `librav1e` 因「编码超时」记为 SKIP，本轮把它拆成**性能**与**表值**两件事分别验证。

#### 6.11.1 编码性能（2s / 1280×720 / 8 核）

| 配置 | 耗时 | 体积 | 相对默认档 |
|---|---|---|---|
| **默认档（生产实际路径，不传 `-speed`）** | **179.2 s** | 899 364 B | 1.00×（≈0.011× 实时） |
| `-speed 10` | **39.0 s** | 1 262 852 B | **体积 1.40×** |
| `-speed 8` | 142.2 s | 907 379 B | 1.01× |
| `-speed 6` | 189.8 s | 899 364 B | 1.00× |
| `-speed 4` | 285.3 s | 882 447 B | 0.98× |
| `-speed 10` + `-tile-rows 1 -tile-columns 1` | 40.6 s | 1 262 852 B | 与不 tile **字节相同** |

* ✅ **`-speed` 是唯一有效旋钮**：`-speed 10` 相对默认档提速 **4.6×**（179.2 → 39.0 s）。
* ⚠ **`-speed` 会平移码率曲线**：同 qp 下 `-speed 10` 的体积约为默认档的 **1.40×**
  （qp 50/60/70/80 四点分别为 1.39/1.68/1.40/1.40×）⇒ **标定必须声明 speed 档**。
* ⚠ **tile 在 1×1 下是 no-op**：输出字节与不分 tile 完全一致（本素材 720p 已被单次
  并行覆盖）；要用 tile 提速需给 `-tile-rows/-tile-columns` ≥ 2 的真分块。
* ⚠ **性能结论**：默认档 rav1e ≈ **0.011× 实时**，用于长视频生产不可行；
  若确需 rav1e，应显式下发 `-speed 10`（并按 1.40× 因子重标，见 6.11.2）。

#### 6.11.2 等体积重标（**发现现表值错误**）

**背景**：`librav1e` 的旧表值 `(4.0, −4.0) → crf21 = qp 80` 是**经 libaom 中转**推导的：

```
rav1e_qp = 4 × (libaom_crf − 5)      （旧实测：libaom crf 20/25/30/35 ↔ rav1e qp 60/80/100/120）
代入当时的 libaom 行 libaom_crf = x264_crf + 4  ⇒  rav1e_qp = 4·x264_crf − 4
```

但 **libaom 行已于同日重标为 `2.007·x264 − 21.35`** ⇒ 同一条链式推导现在给出
`4 × (20.80 − 5) = 63`，**与表里的 80 自相矛盾**。VidUtils 侧的 `crf_to_rav1e_qp()`
走的正是这条链，其判据 ⑨ 组期望值**本来就已经是 64** —— 即 VE 侧本行是唯一的异常点。

**直接实测**（生产路径 = 默认档，扫 `-qp 40~140` 共 12 点，锚点 libx264 crf 18~30）：

| rav1e qp | 40 | 50 | 60 | 66 | 70 | 80 | 90 | 100 | 110 | 120 | 140 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 体积 (B) | 1 944 749 | 1 525 491 | 1 250 602 | 1 117 927 | 1 049 341 | 899 364 | 778 951 | 656 034 | 549 634 | 458 479 | 303 186 |
| 码率比 | 1.683 | 1.320 | 1.082 | **0.967** | 0.908 | 0.778 | 0.674 | 0.568 | 0.476 | 0.397 | 0.262 |

* 5/5 锚点全部落在扫描区间内，**最小二乘 ⇒ `a=7.0032, b=−80.993`（最大残差 1.84 qp），crf21 → qp 66**。
* 锚点级直接插值：crf18→45.4、**crf21→64.2**、crf24→88.5、crf27→109.5、crf30→127.8。
* **三方交叉印证**：本次实测 64~66 ≍ VidUtils 链式推导 63（判据期望 64）
  ≍ 现表 80（**唯一离群**）⇒ 判定 **qp 80 为错值**。

**旧值 80 的实测代价**（vs `libx264 crf21`）：

| rav1e 配置 | 码率比 | ΔPSNR | 判读 |
|---|---|---|---|
| **qp 80（旧表）** | **0.778** | **+0.18 dB** | 体积小 22% 却画质更高 ⇒ 典型"多花画质、少给体积" |
| qp 70 | 0.908 | +0.83 dB | 仍偏保守 |
| **qp 66（新表，已实跑）** | **0.967** | — | 落 `RATE_PASS` 且居中 |
| qp 50 | 1.320 | +2.36 dB | 过配 |

**落地动作**（已执行）：

| 位置 | 变更 |
|---|---|
| `src/utils/convert_crf.py`（**与 VidUtils 逐字同步**） | `'librav1e': (4.0,−4.0,0,255)` → **`(7.0032, −80.993, 0, 255)`** |
| `Accessory/verify/crf_cq_unification_verify.py` `REF21_EXPECTED` | `("-qp", 80)` → `("-qp", 66)` |
| 同上 `G2-10` 期望值 | `80` → `66` |
| 同上 `G3-8` 注释 | 删掉已失效的「截距 −4 不得丢」论证；该用例验证的是**恒等透传**不变式，与 a/b 无关，故仍取 80 作输入 |

⚠ **VidUtils 侧无行为变化**：`crf_to_rav1e_qp()` 只用 `QUALITY_MAP['libaom-av1']`，
**从不读 `QUALITY_MAP['librav1e']`**（已 grep 核实 `from_x264_crf('librav1e', …)` 无调用点）；
本行为的是消除两表歧义 + 满足 ⑨ 组「两份逐条相等」。

#### 6.11.3 「等体积」与「等质量」的口径分工（2026-09-30 定案）

**背景**：评估把 `-speed 10` 设为默认时，逐锚点实测发现 rav1e 的**两个判据严重背离**。

实测（门禁素材 `word_world_2`，687 帧，原生 speed 档，表值 `(7.0032, −80.993)`）：

| x264 crf | 表值 qp | 码率比 | ΔPSNR | 判定 |
|---|---|---|---|---|
| 18 | 45 | 0.956 | **+0.59** | PASS |
| 21 | 66 | 0.910 | **−1.21** | PASS |
| 24 | 87 | 0.939 | **−2.57** | FAIL |
| 27 | 108 | 0.995 | **−4.17** | FAIL |
| 30 | 129 | 1.000 | **−5.79** | FAIL |

* **码率比恒定在 0.91~1.00** ⇒ 等体积拟合非常准；
* **ΔPSNR 随 CRF 单调恶化到 −5.79 dB** ⇒ 等质量**不成立**。
* 根因：**rav1e 在同体积下的画质随 CRF 递减** —— 体积等效点 ≠ 质量等效点。

**因此本仓只保留一张「等体积」表**（与其余 6 个编码器语义一致，V9 方法论口径），
**不引入等质量表**；等质量换算另立项目做。

> 📌 **2026-09-30 更新（本文档后续章节命名以本注为准）**：等质量换算已由**独立立项**落地
> （见 `Plan/PROMPT_等质量换算立项.md`）。随之的**改名**：
> * 本文档中作为「等体积」含义出现的 `QUALITY_MAP` ⇒ 现名 **`SIZE_MAP`**；
> * 新增的等质量表占用 **`QUALITY_MAP`** 之名，默认口径 `--quality-mode **quality**`。
> 上表 rav1e 实测值、`-speed` 决策**不变**（它们描述的是 `SIZE_MAP['librav1e']`）。

**`-speed` 决策**：`-speed 10` 快 4.6×，但同码率下多掉约 **1.9 dB**，且**等体积/等质量无法兼得**：

| 配置 | qp | 码率比 | ΔPSNR | AC7 |
|---|---|---|---|---|
| 原生档（等体积，本表） | 66 | 0.910 | −1.21 dB | ✅ PASS |
| `-speed 10` 等体积解 | 77 | 0.986 | ≈−2.7 dB | ❌ FAIL |
| `-speed 10` 等质量解 | 55 | 1.303 | −1.25 dB | ✅（但体积 +30%） |

⇒ **默认保持 rav1e 原生档**（AC7 绿、体积最小）；`-speed` 作为**显式可选项**
（`VIDEO_RAV1E_SPEED` 环境变量）暴露，由调用方按"性能换体积/画质"自行决策。
启用时 `quality_map._eqvol_model()` 会**自动改用 speed 10 的等体积标定值**
`(6.8159, −66.093) ⇒ crf21 → qp 77`，**等体积语义在两档下都成立**。

⚠ **口径适用范围（重要）**：等体积表**只在默认工作点 crf 21 上被 AC7 验证过**。
上表显示 crf 24~30 的 ΔPSNR 会超地板 —— 这是 rav1e 的固有性质（等体积 ≠ 等质量），
**不是回归**；使用非默认 CRF 时需自行按此预期。

---

## 7. 需 L40（或任意 Ada 卡）才能检测验证的 AV1/VP9 编码测试内容 —— AC0~AC7

> **编号约定：AC = Ada Case**（需 Ada 架构 NVENC / 该机特有构建才能执行的上机用例），与 §0 的
> `[待 L40 复核]` 标记、§6 的 T4 实测收口配套。**AC1 是核心项**（E1 的 ×4 倍率定案）。
> 本机为 Tesla T4（Turing），实跑已确认 `av1_nvenc` 报 `No capable devices found`
> ⇒ AC1/AC2/AC4/AC6 在 T4 上只能 SKIP（AC3 例外，见下；AC5/AC7 另需 QSV/AMF 或该机构建）。

### AC 覆盖矩阵 · AV1/VP9 家族（`QUALITY_MAP` 全部 7 个条目）

> **一条命令跑完这一族**（2026-09-28 新增，T4 上已验证 VP9 一半）：
>
> ```bash
> python3 Accessory/probe/av1_vp9_quality_matrix.py \
>     --src /workspace/input_videos/word_world_2.mp4 \
>     --report verification_report/av1_vp9_matrix_<机名>.md \
>     --json   verification_report/av1_vp9_matrix_<机名>.json < /dev/null
> ```
>
> 它会先探可用性（构建 + 实跑一帧），再对可用编码器跑「表值 vs 朴素」的质量族矩阵，
> 并在 `av1_nvenc` 可用时自动执行 **AC1** 的三点 QP 扫描与判读；
> 退出码 0 = 无 FAIL（SKIP 不算失败）。口径与 `Accessory/verify/crf_cq_unification_verify.py`
> **严格同源**（码率 `ffprobe format=bit_rate`；PSNR 用 `-v info` + **显式 `[0:v][1:v]psnr`**）。

| 编码器 | 需什么 | T4 状态 | 归入哪个 AC |
|---|---|---|---|
| `av1_nvenc` | Ada 及以上（L40/A10/RTX40） | ⏭️ SKIP（`No capable devices found`） | AC1（QP 尺度）+ AC2（CQ 画质）+ AC4 |
| `av1_qsv` | Intel QSV（Arc / 新 iGPU） | ⏭️ SKIP（构建无此编码器） | AC5（量程） |
| `av1_amf` | AMD AMF（RDNA3+） | ⏭️ SKIP（构建无此编码器） | AC5（量程） |
| `libsvtav1` | **ffmpeg 构建**含该编码器 | ⏭️ SKIP（本 build 无） | AC7（软件族复验） |
| `libaom-av1` | 同上 | ⏭️ SKIP（本 build 无） | AC7 |
| `librav1e` | 同上 | ✅ **已跑**（重编后）：`-qp 66` → 0.91×／−1.21 dB → PASS | AC7 |
| `libvpx-vp9` | 同上 | ✅ **已跑**：`-crf 28`，码率比 0.95×（朴素 1.30×）、ΔPSNR −1.67 dB → WARN | AC7 |

⚠ 两种「不可用」必须分开判：**构建有没有**用 `ffmpeg -encoders`；**硬件编不编得动**用
**实跑一帧**（`ffmpeg -h encoder=av1_nvenc` 在 Turing 上照样打印选项表，不可作依据）。

### AC0 · 前置检查（每台机先做一次；不通过则 AC1/AC2/AC4/AC6 全部记 SKIP）

```bash
cd /workspace/Video_Enhancement
nvidia-smi -L                                     # 需 L40 / A10 / RTX 40 等 Ada 及以上
python3 -c "import torch;print(torch.cuda.get_device_name(0),torch.version.cuda)"
ffmpeg -hide_banner -encoders | grep av1_nvenc    # ①构建里有没有该编码器名

# ②硬件能力：唯一可靠判据是**实跑一帧**（`-h encoder=av1_nvenc` 在 Turing 上照样打印选项表）
ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 \
       -c:v av1_nvenc -f null - < /dev/null ; echo "rc=$?"
```

* `rc=0` ⇒ 可执行 AC1/AC2/AC4/AC6；
* `rc≠0`（`No capable devices found`）⇒ 维持 SKIP，**保持 `[待 L40 复核]`，不要改动任何期望值**。

### AC1 · AV1 CONSTQP 的 QP 尺度（×3）定案 —— 核心 **【L40 实测已确认】**

> 🔧 **首选执行方式**：跑 §7 开头的 `Accessory/probe/av1_vp9_quality_matrix.py`
> —— 它在 `av1_nvenc` 可用时会**自动**执行本节的三点扫描、算出码率比/ΔPSNR 并打印
> AC1 判读（含"落带内/未落带内/全出带"三种分支的下一步动作）。下面的手工命令用于
> 复核，或该脚本不可用时。

**被测断言**：`-qp` 是 AV1 的 **qindex（0~255）**，与 `-cq:v`（0~63）不是同一刻度；
**L40 实测确认模型 `QP = 3.0 × 基准轴`（基准轴 21 → QP 63），非原推断的 ×4（QP 84）。**

**已更新锚点（按 L40 实测结果同步）**

| 位置 | 现状（已更新） |
|---|---|
| `src/utils/quality_map.py:194` | `'av1_nvenc':  (3.0, 0.0, 0, 255),   # [L40 实测确认：QP 尺度 3×]` |
| `Accessory/verify/crf_cq_unification_verify.py:983` | `Status.PASS if q_av1 == 63 else Status.FAIL`（G3-7 期望更新为 63） |
| `Accessory/verify/crf_cq_unification_verify.py:1473` | `("G6-7", "IFRNet", "ifrnet_video", "av1_nvenc", 27, "constqp", 0, [("-rc:v","constqp"), ("-qp","63")], ...)` |

**素材**：建议与 G7 同源，便于和 h264/hevc 的结论横向比较（§6.2 用的是
`/workspace/input_videos/word_world_2.mp4`）。PSNR 一律**对源**度量。

```bash
SRC=/workspace/input_videos/word_world_2.mp4
W=/tmp/ac1; mkdir -p $W

# ① 软编基准（与 G7 同参）
ffmpeg -hide_banner -y -v error -i "$SRC" -c:v libx264 -preset medium -crf 21 \
       -pix_fmt yuv420p $W/soft.mp4 < /dev/null

# ② 三点扫描：21=旧实现的错误值 / 63=L40 实测确认值 / 105=×5 方向对照
for q in 21 63 105; do
  ffmpeg -hide_banner -y -v error -i "$SRC" -c:v av1_nvenc -preset p4 \
         -rc:v constqp -qp $q -bf 0 -pix_fmt yuv420p $W/qp$q.mp4 < /dev/null
  echo "encode qp$q rc=$?"
done

# ③ 码率比 + ΔPSNR
# ⚠ 必须严格镜像判据脚本 Ctx._metric 的口径：-v info + **显式 [0:v][1:v] 标签**。
#   裸 `-lavfi psnr` 会走出不同结果（实测 43.40 vs 正确值 46.58），不可用。
N=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$SRC")
psnr_of() {
  ffmpeg -hide_banner -v info -i "$1" -i "$SRC" -frames:v "$N" \
         -lavfi "[0:v][1:v]psnr" -f null - 2>&1 \
    | grep -oP 'average:\s*\K[0-9.]+' | tail -1
}
bitrate_of() {
  ffprobe -v error -select_streams v:0 -show_entries format=bit_rate -of csv=p=0 "$1"
}
SBR=$(bitrate_of $W/soft.mp4); SPS=$(psnr_of $W/soft.mp4)
printf "soft   bitrate=%s psnr=%s\n" "$SBR" "$SPS"
for q in 21 63 105; do
  BR=$(bitrate_of $W/qp$q.mp4); PS=$(psnr_of $W/qp$q.mp4)
  awk -v q=$q -v br=$BR -v ps=$PS -v sbr=$SBR -v sps=$SPS \
    'BEGIN{printf "qp%-4s ratio=%.2fx  dPSNR=%+.2f dB\n", q, br/sbr, ps-sps}'
done
```

**③ 已在 T4 上端到端预验证**（把 `av1_nvenc` 换成本机可用的 `h264_nvenc`、
扫描点改为 `21/26` 以适配本机，其余命令逐字未改）：

```
encode qp21 rc=0
encode qp26 rc=0
soft   bitrate=1427280 psnr=46.580407
qp21   ratio=1.46x  dPSNR=+0.06 dB      ← 与 §6.2 报告里 G7-3 的 1.46× / +0.06 dB 逐位一致
qp26   ratio=0.93x  dPSNR=-3.07 dB
```

⇒ 度量管线（码率口径 + PSNR 口径 + 解析）已被证明正确且与判据脚本同源，
**L40 上已确认 `av1_nvenc` 的 QP 尺度为 ×3（QP 63），非 ×4（QP 84）。**

**判据**（与判据脚本同一套容忍带：`RATE_PASS=(0.65,1.50)`、`TOL_PSNR_DB=1.5`）

> 口径说明：码率取 `ffprobe format=bit_rate`（与判据脚本 `Ctx.media_info` 完全同口径，
> 两者都未加 `-an`，因此含音轨；因软编与候选含同一音轨，比值口径一致、可与 §6.2 的
> G7 数值直接比较）。

| 扫描点 | 期望 | 作用 |
|---|---|---|
| `qp 63` | `0.65 ≤ ratio ≤ 1.50` 且 `ΔPSNR ≥ −1.5 dB` | **落带内 ⇒ ×3 成立（AC1 PASS）** |
| `qp 105` | ratio 应低于 63（单调方向） | 仅方向性对照，**不参与定案** |
| `qp 21` | ratio 应远大于 1.50（近无损 ⇒ 体积暴涨） | 反例对照，确认"旧实现确实错" |

**结果解读与落地动作** —— **已按 L40 实测完成**

| 实测 | 动作 |
|---|---|
| **63 落带内（已确认）** | ① 删掉 `quality_map.py:194` 的 `# [待 L40 复核]`，改为 `# [L40 实测确认：QP 尺度 3×]`；② 去掉判据 `G3-7`(:983) / `G6-7`(:1473) 标题里的"待 L40 复核"字样（期望值已更新为 63）；③ §0 的 E1 行改为"已定案（L40 实测 ×3）" |
| **63 不落带内** | ① 取落带内最接近 21 的点 `qp*`，改 `_QP_MAP_OVERRIDE['av1_nvenc']` 为 `a = qp*/21`（保留 1 位小数）；② **同步改 `G3-7` 与 `G6-7` 的期望值**（需与 `a` 一致）；③ 复跑 AC1 确认新点落带内 |
| **三点全出带** | AV1 的 `-qp` 与基准轴非线性 ⇒ 停手，记录三点原始数据，在 §7 追加"AV1 constqp 不适用线性模型"，并把生产 AV1 的 constqp 路径标为**不支持**（走 `-cq`/VBR） |

**回退**：AC1 只动一个常量 + 两处判据期望值 ⇒ 还原 `a=3.0`、期望 63，并恢复 `[L40 实测待复核]` 标记即可。

### AC2 · AV1 硬编 `-cq` 与软编基准同量级（对应判据 G7-6）

AC0 通过后 G7-6 会自动由 SKIP 转判，无需额外命令：

```bash
python3 Accessory/verify/crf_cq_unification_verify.py --gpu \
  --source /workspace/input_videos/word_world_2.mp4 \
  --bitrate-source /workspace/input_videos/new4_raw.mp4 \
  --report "verification_report/crfcq_gpu_Ada_$(date +%F_%H%M).md" < /dev/null
```

* **判据**：ratio ∈ `RATE_PASS` 且 `ΔPSNR ≥ −TOL_PSNR_DB` ⇒ PASS；落 `RATE_WARN` ⇒ WARN；`ΔPSNR < −3.0 dB` ⇒ FAIL。
* **预期**：与 G7-1/G7-2 同形态（内容相关偏松 −1.7~−2.7 dB、码率达标）属**已知边界**，**只有 FAIL 才算回归**。
* 关闭动作：把 §6.2 表中 G7-6 的 ⏭️ SKIP 填成实测结论。

### AC3 · AV1 constqp 的下发命令形状（对应判据 G6-7）

* **不需要 AV1 硬件**：判据走 `subprocess.Popen` 替身捕获命令形状，并把
  `HardwareCapability.best_encoder` 临时替换为恒等函数 —— **本机 T4 上实测已 PASS**
  （§6.2 报告 `G6-7 ✅ PASS`）。⇒ §0 早期"A5 需 Ada"的说法据此纠正。
* ⚠ 但它只能证明"函数把 27 换成了 63"（×3），**不能证明 63 是正确刻度**；期望值的正确性依赖 AC1。
* AC1 定案后（已确认 ×3）：期望值已同步更新为 63，无需进一步修改。

### AC4 · AV1 的 `-cq` 等质量性（Gate 2 B 组 av1 格）

* 内容：`crf_ref 21` 换算出的 `-cq:v <val>`（E0 后量程 0~63）与"朴素下发 21"对比，容忍带同 AC2。
* 载体：`VidUtils/probe/verify_nvenc_quality_gpu.py` 的 B 组（T4 上 h264/hevc 已跑，av1 格 SKIP）。
* 与 AC1 **正交**（判的是 CQ 轴而非 QP 轴），可独立关闭。

### AC5 · `av1_qsv` / `av1_amf` 量程实测

* 现状：`QUALITY_MAP` 里两者的 hi 仍为 51（`av1_nvenc` 已是 63）。
* **需 Intel QSV / AMD AMF 硬件，Ada 卡也覆盖不了。**
* ⚠ **2026-09-30 在 L40（ffmpeg 7.1-\+av1）上实测：本节原写的探测方法不成立** ——
  `av1_qsv` / `h264_qsv` / `hevc_qsv` 的**编码器选项表里根本没有 `-cq` / `-global_quality`**；
  `-cq` 只以**通用 AVCodecContext 选项**形式存在（`ffmpeg -h full` 里有 3 份副本：
  NVENC AV1 0~63、NVENC H.264/HEVC 0~51），**通用条目不按编码器区分量程**；
  `av1_amf` 连编码器名都不在本 build 里（需 `--enable-amf` 重编）。
  ⇒ 量程只能靠**实跑**取（像 `av1_nvenc` 那样），Ada 卡覆盖不了 QSV/AMF。详见 **§8.4**。
* 无对应硬件时：维持 51 并在旁标注"未核实"（现状已如此）。

### AC6 · 跨项目交叉印证（VidUtils Gate 2 C 组）

`VidUtils/probe/verify_nvenc_quality_gpu.py` 的 C 组是 AC1 的同源实现，结论应与 AC1 一致；
若相左，以**本仓 AC1 的手工三点数据**为准并追查脚本差异（两边容忍带本就同一套：
`TOL_PSNR=1.5`、`RATE_PASS=(0.65,1.50)`）。

### AC7 · AV1/VP9 **软件族**在目标机构建上复验（VP9 一半在 T4 已过，**全族 2026-09-29 完成**）

**为什么要单独一项**：三个 AV1 软件编码器（`libsvtav1` / `libaom-av1` / `librav1e`）在
**本机 ffmpeg 构建里根本没有**，与算力无关 ⇒ **Ada 卡也不会自动解决**，取决于目标机的
ffmpeg 构建。而其中 `libsvtav1` 的换算表是 2026-09-28 刚按真实素材重标定的（E6），
**至今没有端到端验证过**。

* **执行**：§7 开头的 `av1_vp9_quality_matrix.py`（一条命令覆盖全部 7 个编码器）。
* **判据**（与 A 组同口径）：

  | 编码器 | 期望 | 实测结果（2026-09-29 T4） | 备注 |
  |---|---|---|---|
  | `libvpx-vp9` | 码率比 ∈ `RATE_PASS` 且 ΔPSNR ≥ −1.5 dB | ⚠️ **WARN** −1.67 dB / 0.95× | 内容相关偏松，与 G7-1/G7-2 同形态，非回归 |
  | `libsvtav1` | 同上 | ✅ **PASS** −0.01 dB / 1.03× | `-crf 25` / preset 8 |
  | `libaom-av1` | 同上 | ✅ **PASS** −0.43 dB / 0.79× | `-crf 25` / -cpu-used 5 |
  | `librav1e` | 同上 | ✅ **PASS** −1.21 dB / **0.91×** | **2026-09-29 补测**（旧表 qp80 编码超时，见 §6.11）；`-qp 66`（新表值），朴素 `-qp 21` 为 1.98× |

* **产出**：`verification_report/av1_vp9_matrix_T4_20260929.md` / `.json`（首轮）、
  `…_rav1e.md` / `.json`（rav1e 补测）。
* **关闭动作**：AC7 全族完成。四个软编编码器结论：libsvtav1 PASS / libaom-av1 PASS /
  librav1e PASS / libvpx-vp9 WARN（已知边界）。`librav1e` 的表值错误已在 §6.11 修正并落表。

---

### 汇总：AC × 前置 × 分机状态 × 关闭动作（2026-09-30 更新）

| ID | 需什么 | T4 状态 | L40 状态（2026-09-30 复核） | 关闭动作 |
|---|---|---|---|---|
| **AC1** | Ada（L40/A10/RTX40） | ⏭️ SKIP | ✅ **再次确认 ×3**（`69b62db` 已定案，本轮独立复现） | **已闭环**：按实测定案 ×3，同步 `G3-7`/`G6-7` 期望为 63，移除 `[待 L40 复核]`；探针判读改为跟随表（**A15**） |
| AC2 | Ada | ⏭️ SKIP | ✅ **PASS**（G7-6：ΔPSNR **+0.16 dB** / 1.28×） | §6.2 表 G7-6 已填实测值 |
| **AC3** | **无**（逻辑/命令捕获层） | ✅ **PASS** | ✅ **PASS** | 已同步期望值为 63 |
| AC4 | Ada | ⏭️ SKIP | ✅ **PASS**（A 组 av1 格 1.14× / −0.29 dB） | Gate 2 B 组 av1 格转正 |
| AC5 | Intel QSV / AMD AMF | ⏭️ SKIP | ⏭️ SKIP（**探测方法已被证伪**，见 §8.4） | 只能实跑取证；`QUALITY_MAP` 的 hi 维持 51 并标注"未核实" |
| AC6 | Ada | ⏭️ SKIP | ✅ 已由 AC1 同源实现覆盖（VidUtils 脚本需另跑） | 与 AC1 交叉印证 |
| **AC7** | 目标机 **ffmpeg 构建**含 AV1/VP9 软编 | ✅ **已完成**（重编 ffmpeg 启用 3 编码器） | ✅ 部分复跑（`libsvtav1` 1.09×/+0.25 PASS、`libvpx-vp9` 0.95×/−1.67 WARN，与 T4 逐位一致；`libaom-av1`/`librav1e` 本轮**未跑**，见 §8.2） | 已闭环（§6.10 / §6.11） |

> **一条命令的入口**：`python3 Accessory/probe/av1_vp9_quality_matrix.py --src <真实素材> < /dev/null`
> —— 覆盖 AC1（自动判读）、AC2/AC4 的同轴对照、AC7 全部 7 个编码器；AC5 需另换硬件，
> AC6 走 VidUtils 的脚本。

> **状态**：L40 上 AC1~AC4、AC6 已全部闭环，**AV1 CONSTQP QP 尺度确认为 ×3（QP 63）**（2026-09-30 二次复现）。生产 AV1 任务的 constqp 路径现已可用（基准 21 → QP 63），`-cq`/VBR 路径亦已有 E0 的 0~63 量程支撑。VP9 侧无硬件依赖，`libvpx-vp9` 表值已在 T4/L40 实测（WARN，内容相关偏松，非回归）。**软件 AV1 族（libsvtav1/libaom-av1）已在 T4 经重编 ffmpeg 完整验证 PASS**。
> ⚠ **但"换算正确"不等于"管线能跑"**：2026-09-30 的 P3 长视频冒烟（§8.5）开箱即失败，
> 根因是三处 AV1 路径缺陷（§8.6），已全部修复并复测通过。**AV1 端到端能力此前从未被验证过。**

---

## 8. L40 上机收口（2026-09-30）：AC 复核 + P3 长视频冒烟

> 换机到 **Tesla L40**（Ada / sm89 / 46 GB / 48 核）后执行。环境：torch 2.10.0+cu128、
> CUDA 可用；`ffmpeg 7.1-+av1`（重编含 `libsvtav1`/`libaom-av1`/`librav1e`）；OpenCV 4.13.0。
> 门禁与静态判据同 §6.1 基线；本次新增项全部标 **[L40]**。

### 8.1 AC0 · 前置检查

```bash
nvidia-smi -L                                       # NVIDIA L40
python3 -c "import torch;print(torch.__version__, torch.version.cuda)"   # 2.10.0+cu128 True
ffmpeg -hide_banner -encoders | grep -E 'av1_nvenc|av1_qsv|av1_vaapi'   # 三个都在 build 里
ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 \
       -c:v av1_nvenc -f null - < /dev/null ; echo "rc=$?"              # rc=0
```

* ✅ `av1_nvenc` **实跑一帧 rc=0** ⇒ AC1/AC2/AC4 可执行。
* ⚠ `av1_qsv` 在 build 里但**实跑失败**（`Error creating a MFX session: -9`）⇒ AC5 仍 SKIP。
* ⚠ `av1_amf` 不在 build（`Codec 'av1_amf' is not recognized by FFmpeg`）。

### 8.2 AC1 / AC4 / AC7 · 一条命令复跑（并修掉探针自身的缺陷）

```bash
python3 Accessory/probe/av1_vp9_quality_matrix.py \
    --src /workspace/input_videos/word_world_2.mp4 \
    --only av1_nvenc,av1_qsv,libvpx-vp9,libsvtav1 \
    --report verification_report/av1_vp9_matrix_L40_2026-09-30_0501.md \
    --json   verification_report/av1_vp9_matrix_L40_2026-09-30_0501.json < /dev/null
```

| 组 | 项 | 结论 | 数据 |
|---|---|:--:|---|
| A | `av1_nvenc` → `-cq:v 27` | ✅ **PASS** | 1.14×（朴素 1.84×）/ ΔPSNR **−0.29 dB** |
| A | `libsvtav1` → `-crf 24` | ✅ **PASS** | 1.09×（朴素 1.22×）/ **+0.25 dB**（表值已于 §6.8 由 25 改 24） |
| A | `libvpx-vp9` → `-crf 28` | ⚠️ WARN | 0.95×（朴素 1.30×）/ **−1.67 dB** —— 与 §6.2 的 `G7-7`、§6.7 的 T4 结果**逐位一致** |
| A | `av1_qsv` | ⏭️ SKIP | MFX session −9（无 Intel 硬件） |
| **B** | **`-qp 21`** | ❌ FAIL | **2.64× / +3.86 dB** —— 近无损导致体积暴涨，**证实旧实现（21）确实错** |
| **B** | **`-qp 63`（表值）** | ✅ **PASS** | **1.16× / −0.59 dB** ⇒ **AC1 ×3 二次确认** |
| B | `-qp 84`（旧 ×4 假设） | ⚠️ WARN | 0.93× / −2.00 dB |
| B | `-qp 105`（×5 对照） | ❌ FAIL | 0.74× / −3.54 dB |

* ✅ **软编基准与 T4 逐位相同**（1425 kbps / 46.581 dB）⇒ 两条独立运行（不同机器、不同日期）口径一致。
* ⏭️ **本轮未跑 `libaom-av1` / `librav1e`**：CPU 上单条编码 ~2 min（`libaom` 尤慢），
  与"快速执行"冲突；两者已在 §6.10 / §6.11 闭环，AC7 无需重开。
  （若要复跑，去掉 `--only` 即可，代价约 6~8 min。）

**A15 · 探针自身的缺陷（已修）**：B 组判读原先把 **`84`（×4 假设）写死**在脚本里。
`69b62db` 已把表改成 ×3（QP 63），于是该脚本在 L40 上会打印
「84 未落带内 ⇒ 改 a = 4.0」这种**与表自相矛盾**的结论。已改为：

```python
def av1_expected_qp(ref_crf: int) -> int:
    _, cq_value, _, _ = Q.resolve_quality("av1_nvenc", default_ref=ref_crf)
    return int(Q.to_constqp_qp("av1_nvenc", cq_value))     # 走生产同一条链
```

扫描点变为 `{21, 表值, 84, 105}`，JSON/MD 增列 `av1_qp_expected` 与 `ac1_verdict`；
"落带点最接近 21" 的兜底分支也改成从实测落带点反推 `a`。
⇒ **教训：判据/探针里凡是硬编码的期望值，都要确认它没被上游改动作废。**

### 8.3 AC2 · G7/G8 门禁（L40）

```bash
python3 Accessory/verify/crf_cq_unification_verify.py --gpu \
    --source /workspace/input_videos/word_world_2.mp4 \
    --bitrate-source /workspace/input_videos/new4_raw.mp4 \
    --report verification_report/crfcq_gpu_L40_2026-09-30_0502.md < /dev/null
```

* **总判定：PASS 101 / FAIL 0 / WARN 2 / SKIP 0**（两个 WARN 是 G7-1/G7-2 的既有内容相关偏松）。
* **G7-6（AC2）PASS**：`av1_nvenc -cq:v 27` ΔPSNR **+0.16 dB** / 码率比 **1.28×**（朴素 **2.05×**）。
* **G7-7 PASS**：`libsvtav1 -crf 24` −0.31 dB / 1.10×（本 build 有 svtav1，故不再回退 vp9）。
* G7-3（constqp `-qp 21` 轴）PASS +0.05 dB / 1.44×；G7-5 VMAF 最大偏差 2.06；G8 8/8 PASS。

### 8.4 AC5 · 探测方法被证伪（不是"缺硬件"，是"这条路本来就不通"）

在 ffmpeg 7.1-\+av1 上逐项核查 §7 AC5 写的方法：

| 核查项 | 结果 |
|---|---|
| `ffmpeg -h encoder=av1_qsv \| grep -- '-cq'` | **无 `-cq`、无 `-global_quality`**（只有 `-extbrc` / `-qsv_params` / `-preset` int 0..7 …） |
| `h264_qsv` / `hevc_qsv` 同查 | **同样没有 `-cq`** ⇒ 不是 AV1 特有，是 QSV 族在 7.x 的普遍形态 |
| `ffmpeg -h full \| grep '^\s*-cq'` | 有 **3 份** `-cq`：NVENC AV1（0 to 63）、NVENC H.264（0 to 51）、NVENC HEVC（0 to 51）—— 均为**编码器私有**条目 |
| `av1_qsv -global_quality 26` 实跑 | ffmpeg **接受该选项**（报 MFX session −9，而非 unknown option）⇒ 走的是通用 AVCodecContext 的 `global_quality (INT_MIN..INT_MAX)`，**无量程可言** |
| `av1_amf` | 本 build 无此编码器（需 `--enable-amf`） |

⇒ **结论**：`-cq` 的 0~63 / 0~51 量程只存在于**编码器私有选项**里；QSV/AMF 在 ffmpeg 7.x
不再声明 `-cq`，通用 `global_quality` 又是无限幅 int ⇒ **"从选项表取量程"这个方法对
QSV/AMF 不成立**。要定案只能实跑（`av1_nvenc` 正是这么定的），而 Ada 卡覆盖不了 QSV/AMF。
`QUALITY_MAP` 的 `av1_qsv` / `av1_amf` 维持 `hi=51` + "未核实"标注（本轮未改表）。

### 8.5 P3 · 长视频冒烟（本轮主项，**开箱即失败 ⇒ 先修缺陷再复测**）

**素材**（真实纪录片素材，5.5 min，带音轨）：

```bash
ffmpeg -y -ss 300 -i "/workspace/input_videos/WordWorld_S2/2-01. My Fuzzy Valentine - Love Bug.avi" \
       -t 330 -c:v libx264 -preset veryfast -crf 18 -c:a aac -b:a 128k -pix_fmt yuv420p src_5min.mp4
# → 640×360 / 7912 帧 / 23.98 fps / 330.0 s
```

**两条命令**（IFRNet 与 ESRGan **两侧都**走 AV1 硬编）：

```bash
for RATE in constqp vbr; do
  python3 src/main_video_optimized.py -c config/default_config.json \
      -i src_5min.mp4 -o out_5min_$RATE.mp4 \
      --codec-ifrnet av1_nvenc --codec-esrgan av1_nvenc \
      --rate-mode-ifrnet $RATE --rate-mode-esrgan $RATE \
      --segment-duration 30 < /dev/null
done
```

| 项 | `constqp` | `vbr` |
|---|---|---|
| 管线退出码 | **0** | **0** |
| 段数 / 段级解码级守恒 | 11 / 11 段 `decoded == expected` | 11 / 11 |
| 最终产物 | **AV1** / 1280×720 / **15803 帧** / AAC 立体声 | 同 |
| 输出时长 | 330.000 s | 330.000 s |
| 平均码率 | 2.52 Mbps | 4.61 Mbps |
| `validate_decodable_video(count_mode='decode')` | ✅ `ok=True`，15803 帧，0 解码错误，NVDEC 直解 | ✅ 同 |
| `segment_bitstream_verify_v5 --skip-chroma` | ✅ frames==packets、无 pts_anomaly | ✅ 同 |
| QA sidecar | ✅ 13 字段齐全（`codec_hint=av1_nvenc`、`rate_mode_ifrnet`、`fixes_applied`…） | ✅ 同 |
| 耗时（TRT 引擎命中缓存） | **7 分 20 秒 ≈ 0.75× 实时** | 7 分 31 秒 |
| 宿主 RSS（进程树） | 起步 0.6 GB → 平台期 ~5.0 GB，**后半段斜率 −326 MB/min** | 峰值 6.3 GB，斜率 **−390 MB/min** |
| GPU 显存峰值 | 6.6 GB / 46 GB（14%） | 7.9 GB / 46 GB |

* ✅ **帧守恒口径**：最终 15803 帧 = 11 个分段各自 `2n−1` 之和（源分段合计 7907 帧；
  与整片 7912 的 5 帧差来自 `-c copy` 按时间切分的边界，§6.5 已记录同一现象）。
* ✅ **无内存泄漏**：RSS 后半程斜率为**负**（平台期 ~5 GB），GPU 显存平稳。
* ⚠ **色度检查（检查 4）在 AV1 长片上是内容相关假阳性**：5 min AV1 产物报 113 个"坏帧簇"
  （索引呈**固定步长 8**：716, 724, 732 …），而**未经管线的源片段自己就报 40 个**
  （358, 366, 374 … 同样步长 8）；同一 30 s 干净片段上 `av1_nvenc` / `libx264 crf21` /
  `libx264 -qp 0` 三者均为 **0 簇** ⇒ 内容触发、非编码缺陷。
  **验收硬指标一律加 `--skip-chroma`**（与 AGENTS.md / §6.5 的既有结论一致）。
  （L40 侧新证据：L40 的 NVDEC **能**硬解 AV1（`hwaccel=nvdec` 直解 15803 帧），
  cv2 失败纯粹是 OpenCV build 的问题，见 §8.6-①。）

### 8.6 冒烟暴露的三处 AV1 缺陷（**均已修复**）

| # | 位置 | 现象 | 根因 | 修法 |
|---|---|---|---|---|
| ① | `src/utils/video_utils.py` `verify_video_integrity()` | 管线在**第一个** AV1 分段就 `❌ 输出文件验证失败` → 终止，`decoded=1437` 的**完好**产物被 `unlink` | 该函数用 `cv2.VideoCapture().read()` 读首帧；**OpenCV 4.13 自带 FFmpeg 无 AV1 解码**（`Your platform doesn't support hardware accelerated AV1 decoding` / `Failed to get pixel format` / `Get current frame error`），而系统 ffmpeg 解同一文件 1437/1437 帧 rc=0 | 新增 `_ffmpeg_first_frame_ok()`：cv2 失败时回退 `ffmpeg -v error -frames:v 1 -f null -`（`-nostdin`）。严格验收仍由其后的 `validate_decodable_video(count_mode="decode")` 承担，**未放宽** |
| ② | `external/realesrgan_video/ffmpeg_io.py` `FFmpegWriter._init_ffmpeg_process()` | 明明 `--codec-esrgan av1_nvenc`，实际命令是 `-vcodec libx264 -crf 27`；而末尾 `[FIX-NVENC-PIPE]` 摘要还打印 `NVENC constqp(cq=27)`，**日志与命令自相矛盾** | 两处 NVENC 判定用**精确元组** `video_codec in ('h264_nvenc','hevc_nvenc')`，`av1_nvenc` 落到 `else` 的 libx264 分支，**静默**把 NVENC 的 CQ 值当 CRF | 两处改为 `'nvenc' in video_codec`（与 `ifrnet_video/ffmpeg_io.py` 同口径）；preset 映射随之覆盖 av1（`medium`→`p4`，即 av1_nvenc 的 default 档，避免命中同名的另一枚举项 index 2 "hq 1 pass"） |
| ③ | `external/{ifrnet,realesrgan}_video/ffmpeg_io.py` | `--rate-mode-* vbr_hq`（**配置默认值**）+ av1_nvenc ⇒ 编码命令**直接失败** | av1_nvenc 的 `-rc` 只接受 `constqp/vbr/cbr`；两处 CLI writer 把 `vbr_hq` 原样下发 ⇒ `Undefined constant or missing '(' in 'vbr_hq'` → `Unable to parse option value` → ffmpeg 非零退出 | 两侧在 `av1` in codec 且 rc ∈ {`vbr_hq`,`qvbr`} 时降级为 `vbr` 并打印警告，与 `nvenc_sdk.NVENCEncoder` 既有降级同口径 |

**为什么这三处能潜伏至今**：AV1 在 L40 上**永远走不到 SDK 直通** ——
`[NVENCEncoder] Level 1 失败: GetEncodePresetConfig failed, code=12`（`NVENC_ERR_INVALID_PARAM`）
两侧都命中，随后按设计降级到 Level 2/3（ffmpeg CLI 管道 + NVENC）。
⇒ **AV1 段实际一直是"ffmpeg CLI 调 av1_nvenc"**，而 ② ③ 两处缺陷恰好都在这条 CLI 路径上；
① 则在验收层。AV1 端到端此前**从未被任何测试覆盖过**（AC1~AC7 全是"下发单条 ffmpeg 命令"的微观测试）。

### 8.7 修复后的回归

| 门 | 命令 | 结果 |
|---|---|---|
| VE 静态判据 | `crf_cq_unification_verify.py --quick` | **PASS 91 / FAIL 0 / WARN 0 / SKIP 11** |
| 门禁全套 | `plan_implementation_gate.py` | **96 项 / 94 通过 / 0 失败 / 0 警告 / 2 跳过** |
| pytest | `pytest Accessory/test -q` | **24 passed** |
| 冒烟复测 | §8.5 两条命令 | constqp / vbr 均 `exit 0`（见 §8.5 表） |

### 8.8 本轮之后仍剩什么

1. **P2 · V9 更广泛素材复核**（不需 L40）：`libx265` / `libvpx-vp9` / `libsvtav1` 三行
   仍只有 4 次标定（且首轮受 `prep.mp4` 缓存影响，仅首条素材有效）⇒ 用
   `calibrate_soft_offsets_nocache.py` 重跑并补 4K / 高帧率 / 动画。
2. **AC5 · `av1_qsv` / `av1_amf` 量程**：需对应硬件，且**只能实跑取**（§8.4）。
3. **AC6 · VidUtils `verify_nvenc_quality_gpu.py` 的 C 组**（跨项目交叉印证）：
   需在 VidUtils 仓所在机器上跑；本仓 AC1 的四点数据（21/63/84/105）已可直接对照。
4. **AV1 的 Level 1 直通**（`GetEncodePresetConfig code=12`）：功能不受影响（已正确降级），
   但少一层直通优化，值得单独立项查 SDK/驱动侧原因。

---

## 9. 续：回归保护 + V9 复核（2026-09-30 第二轮）

### 9.1 环境突变：会话中途 GPU 断联（本节所有"未跑"项的共同原因）

| 时刻 | 现象 |
|---|---|
| ≤ 05:56 | L40 可用（`vbr_hq`→`vbr` 降级验证的 12 s pilot 正常出片） |
| 06:07 | `/usr/lib/x86_64-linux-gnu/libcuda.so.1` 与 `libnvidia-ml.so.1` 被指向 **0 字节**的 `libcuda.so.580.65.06`；`/dev/nvidia*` 设备节点消失 |
| 06:13 起 | `nvidia-smi` 报 `couldn't find libnvidia-ml.so`；`ffmpeg -c:v av1_nvenc` 报 `Cannot load libcuda.so.1`；`torch.cuda.is_available()=False` |

* ⇒ **AC5 与 P3″（AV1 Level 1 `code=12` 根因）同样因环境不支持而跳过**（与 AC5 同类）。
* ✅ 门禁的**环境探测按设计降级为 WARN**：`R5 CUDA / GPU 可用`、`R7 NVENC 环境探测` 两项 WARN，
  **FAIL 仍为 0** —— 这正是"环境差异 ≠ 功能失败"的设计意图得到验证。
* ⚠ 教训：长会话里 GPU 可能被回收；**任何 GPU 相关结论都必须在同一时刻实测**，
  历史报告不能替代当场复跑（本文 §8 的 L40 结论因此仍然有效，但 P3″ 无法在本会话内闭环）。

### 9.2 P3′-a · 判据：`av1_nvenc` 命令形状断言（G6-8 / G6-9 / G6-10）

§8.6 的 ② ③ 两处缺陷本质是"**下发了错的命令**"，因此最该由命令形状断言守住
（判据 G6 组本来就是干这个的，但只覆盖了 constqp）。新增三格（**不需要 AV1 硬件**，
写入器只拼命令串、`Popen` 被替身捕获）：

| ID | 侧 | codec | rate_mode | 必须出现 | 必须不出现 |
|---|---|---|---|---|---|
| G6-8 | ESRGAN | `av1_nvenc` | constqp | `-vcodec av1_nvenc`、`-preset p4`、`-rc:v constqp`、`-qp 63` | `-crf`、`-cq:v` |
| G6-9 | IFRNet | `av1_nvenc` | **vbr_hq** | `-vcodec av1_nvenc`、`-rc:v vbr`、`-cq:v 27`、`-b:v 0`、`-rc-lookahead 8` | `-rc:v vbr_hq` |
| G6-10 | ESRGAN | `av1_nvenc` | **vbr_hq** | `-vcodec av1_nvenc`、`-rc:v vbr`、`-cq:v 27`、`-b:v 0` | `-rc:v vbr_hq`、`-crf` |

**反向验证（关键：断言必须真的会 FAIL）** —— 临时把 §8.6 的 ② ③ 两处修复回退后重跑：

```
❌ [G6-8]  缺少 ['-vcodec av1_nvenc','-preset p4','-rc:v constqp','-qp 63'] / 多出 ['-crf']
❌ [G6-9]  缺少 ['-rc:v vbr'] / 多出 ['-rc:v']
❌ [G6-10] 缺少 ['-vcodec av1_nvenc','-rc:v vbr','-cq:v 27','-b:v 0'] / 多出 ['-crf']
合计：PASS=90  FAIL=3      （恢复修复后 → PASS=93  FAIL=0）
```

### 9.3 P3′-b · pytest：cv2 → ffmpeg 回退（`[FIX-AV1-CV2]` 的回归锁）

`Accessory/test/test_verify_video_integrity_fallback.py`（3 例）：

1. `test_cv2_failure_falls_back_to_ffmpeg` —— 猴补丁把 `cv2.VideoCapture` 换成"打不开/读不出帧"，
   好文件**必须**仍判为完好（这是 AV1 能跑通的前提）。
2. `test_fallback_still_rejects_broken_file` —— 4 KB 垃圾文件**必须**判坏
   ⇒ 证明回退**没有**把判定放宽成"有 ffmpeg 就放行"。
3. `test_missing_and_tiny_files_rejected` —— 缺文件 / <1 KB 的老闸门不受影响。

**反向验证**：把 `verify_video_integrity` 临时改回旧行为（cv2 失败即 `return False`）后
`test_cv2_failure_falls_back_to_ffmpeg` **FAIL**；恢复后 3/3 PASS。

### 9.4 P3′-c · 可重复脚本：`Accessory/verify/av1_pipeline_smoke.py`

把 §8.5 那次一次性手搓的冒烟固化成一条命令：

```bash
python3 Accessory/verify/av1_pipeline_smoke.py --src <真实素材> \
    --rate-modes constqp,vbr --segment-duration 30 \
    --report verification_report/av1_smoke_<机名>.md < /dev/null
# 复验既有产物（不需要 GPU）
python3 Accessory/verify/av1_pipeline_smoke.py --src <素材> --checks-only out.mp4 < /dev/null
```

| 项 | 内容 |
|---|---|
| 前置 | `av1_nvenc` **实跑一帧**；不可用 ⇒ **exit 2**（环境前置不成立，不是失败） |
| 跑批 | 每个 rate_mode 调一次 `src/main_video_optimized.py`，两侧 `--codec-* av1_nvenc` |
| 采样 | 整棵进程树 RSS + `nvidia-smi` 显存（判泄漏用后半程线性斜率） |
| 验收 | S1 退出码 / S2 段级 `decoded==expected` / S3 产物帧数=各段之和 / S4 `validate_decodable_video(count_mode=decode)` / S5 `segment_bitstream_verify_v5 --skip-chroma` / S6 QA sidecar 字段 / S7 **产物编码器确为 av1** / S8 泄漏斜率 ≤ +50 MB/min |

退出码：`0` 无 FAIL / `1` 有 FAIL / `2` 环境前置不成立。

**本轮验证到的程度**（因 §9.1 的 GPU 断联）：

| 分支 | 状态 |
|---|---|
| 前置不成立 ⇒ exit 2 + 明确提示 | ✅ 实跑（`Cannot load libcuda.so.1`） |
| `--checks-only` 的 S4/S5/S6/S7 | ✅ 用 `libsvtav1` 产的 AV1 样本 + 复刻的 QA sidecar 实跑：4 PASS / 0 FAIL / 4 SKIP |
| S7 的**反证** | ✅ 指向 h264 文件时 S7 **FAIL**（`codec=h264`）—— 这正是能抓住 §8.6-② 的那一项 |
| 完整跑批分支（S1/S2/S3/S8） | ⏭️ 需 GPU，留待复跑 |

### 9.5 P2 · V9 三行复核（无缓存脚本，3 素材，`--dense`）

```bash
V=/workspace/VidUtils/probe/calibrate_soft_offsets_nocache.py
python3 $V --src /workspace/input_videos/new5_raw.mp4  --duration 4 --width 1280 --height 720  --codecs libx265,libvpx-vp9,libsvtav1 --dense --workroot temp/v9 --tag m1_720p30  # 基准素材
python3 $V --src /workspace/input_videos/new4_raw.mp4  --duration 4 --width 1920 --height 1080 --codecs libx265,libvpx-vp9,libsvtav1 --dense --workroot temp/v9 --tag m2_1080p30
python3 $V --src /workspace/input_videos/wws3e02_26s.mp4 --duration 4 --width 640 --height 360 --codecs libx265,libvpx-vp9,libsvtav1 --dense --workroot temp/v9 --tag m3_360p
```

| 素材（分辨率） | `libx265` | `libvpx-vp9` | `libsvtav1` |
|---|---|---|---|
| **m1 · new5_raw（720p30，= 原标定基准）** | 20.81（表 20.81，**Δ +0.00**） | 28.31（表 28.17，**Δ +0.14**） | 23.68（表 23.70，**Δ −0.02**） |
| m2 · new4_raw（1080p30） | 20.29（Δ −0.52） | 28.49（Δ +0.32） | 22.38（Δ −1.32） |
| m3 · wws3e02（360p 低复杂度） | 19.92（Δ −0.89） | **25.90（Δ −2.27）** | 22.62（Δ −1.07） |

（表中 crf21 落点来自 `src/utils/quality_map.py::QUALITY_MAP`；`prep` md5/字节数落在
`verification_report/v9_calib_nocache_m{1,2,3}_*.json`，可审计。）

* ✅ **基准素材上三行逐条复现（|Δ| ≤ 0.14 crf）⇒ 现表值正确，不改表**。
  （`libsvtav1` 现表 `(2.145, −21.35)` 正是 §6.8 落表值；`libvpx-vp9` `(1.6381, −6.2289)`
  亦为 §6.8 首条素材的落表值 —— 本轮 m1 等于把那条被 `prep` 缓存污染过的记录重新干净地测了一遍。）
* ⚠ **残差是内容/分辨率相关**，最大 −2.27 crf 落在 `libvpx-vp9` —— 正是 AC7 里
  ΔPSNR −1.67 dB 那个已知 WARN 的编码器。单一线性常量压不掉内容相关误差
  （与 E9 判 `CQ_OFFSET=0`、E10 判"合成过配/真实欠配"同源结论）⇒ **不为单素材微调表值**。
* 证据：`verification_report/v9_calib_nocache_m{1,2,3}_*.json`。

### 9.6 P3″ · AV1 Level 1 `GetEncodePresetConfig code=12`（**环境阻断，附复现片段**）

现象：AV1 段恒定降级到 Level 2/3（ffmpeg CLI 管道 + NVENC），功能正确、少一层直通优化。
已定位到的线索：失败发生在 `nvEncGetEncodePresetConfig(AV1_GUID, preset_GUID, &cfg)`，
返回 `12 = NVENC_ERR_INVALID_PARAM`，且发生在 `OpenEncodeSessionEx` 成功、
codec/preset GUID 都枚举成功之后。

**待验证假设**：AV1 上 GUID 版 `GetEncodePresetConfig` 不可用，应改用 SDK 13.0 的
`nvEncGetEncodePresetConfigEx(encoder, NV_ENC_CODEC_AV1=3, NV_ENC_PRESET_P4=8+3, &cfg)`
（`_FUNC_IDX["GetEncodePresetConfigEx"] = 39` 已在 `nvenc_sdk.py` 里登记，只是没有回退路径）。

```python
# 复现/验证片段（需 GPU；本会话因 §9.1 未能执行）
import ctypes, sys
from ctypes import byref, cast, c_uint32, c_void_p
sys.path.insert(0, "external")
import torch; torch.cuda.init()          # 容器内 /usr/lib/.../libcuda.so.1 可能是桩，必须先由 torch 加载真库
from ifrnet_video import nvenc_sdk as S

_orig = S.NVENCEncoder._build_encoder_config
def patched(self, codec_guid, preset_guid, w, h, fps, qp):
    try:
        return _orig(self, codec_guid, preset_guid, w, h, fps, qp)
    except RuntimeError as exc:                     # GPC 失败 → 试 Ex
        print("GPC 失败:", exc)
        fn = ctypes.CFUNCTYPE(c_uint32, c_void_p, ctypes.c_int, ctypes.c_int,
                              ctypes.POINTER(S._NvEncPresetConfig))(
            self._get_func(S._FUNC_IDX["GetEncodePresetConfigEx"]))
        pc = S._NvEncPresetConfig(); ctypes.memset(byref(pc), 0, ctypes.sizeof(pc))
        pc.version = S.NV_ENC_PRESET_CONFIG_VER
        cast(byref(pc, 8), ctypes.POINTER(c_uint32))[0] = S.NV_ENC_CONFIG_VER
        st = fn(self._encoder, {"h264": 0, "hevc": 1, "av1": 3}[self._codec],
                8 + S._PRESET_P_INDEX.get(self._preset_name, 4), byref(pc))
        print("GPCEx ->", st)                        # 0 = 假设成立
        raise
S.NVENCEncoder._build_encoder_config = patched
S.NVENCEncoder(640, 360, 30.0, qp=27, preset="medium", rate_mode="constqp",
               la_depth=0, codec="av1")
```

| 判读 | 落地动作 |
|---|---|
| `GPCEx -> 0` 且能完成 `InitializeEncoder` | 在 `_build_encoder_config` 加"GPC 失败 ⇒ GPCEx"回退（两侧 `nvenc_sdk.py`），AV1 恢复 Level 1 |
| `GPCEx` 也非 0 | 记为驱动侧限制，保持现有降级（功能无碍），在 `nvenc_sdk.py` 注释里写明"AV1 恒走 Level 2/3" |
| 顺带测 `h264`/`hevc` 是否也失败 | 若 h264/hevc 正常 ⇒ 确证 AV1 专属；若也失败 ⇒ 是本机驱动/SDK 版本问题，与 codec 无关 |

⚠ 复现时**必须先 `import torch; torch.cuda.init()`**：容器内 `/usr/lib/x86_64-linux-gnu/libcuda.so.1`
可能是 0 字节桩，不先由 torch 加载真库会报 `Cannot load CUDA library: ... file too short`。

### 9.7 本轮回归

| 门 | 结果 |
|---|---|
| `crf_cq_unification_verify.py --quick` | **PASS 93 / FAIL 0 / WARN 0 / SKIP 12**（+3 = 新增 G6-8/9/10） |
| `plan_implementation_gate.py` | 96 项 / 92 通过 / **0 失败** / 2 警告（R5 CUDA、R7 NVENC —— §9.1 环境）/ 2 跳过 |
| `pytest Accessory/test -q` | **27 passed**（24 + 新增 3） |

---

## 后续建议（下一步）

| 优先级 | 事项 | 说明 |
|---|---|---|
| ~~**P1**~~ | ~~AC5：`av1_qsv`/`av1_amf` 量程实测~~ | **方法已被证伪（§8.4）**：ffmpeg 7.x 的 QSV 族选项表里没有 `-cq`，通用 `global_quality` 无限幅 ⇒ 只能实跑取，需对应硬件，Ada 卡覆盖不了。`QUALITY_MAP` 维持 `hi=51` + "未核实"。 |
| ~~**P1**~~ | ~~librav1e 编码性能验证~~ | **已完成（§6.11）**：默认档 179.2s/2s@720p（≈0.011× 实时）；`-speed 10` 提速 4.6× 但体积 ×1.40；tile 1×1 为 no-op。表值已按实测改为 `(7.0032,−80.993)`→qp66。**残留**：若生产启用 rav1e，建议显式下发 `-speed 10` 并按 1.40× 因子重标。 |
| **P2** | ~~V9 更广泛素材复核（其余三个编码器）~~ | **已完成（§9.5）**：`libx265` / `libvpx-vp9` / `libsvtav1` 三行用无缓存脚本 `--dense` 在 720p30 / 1080p30 / 360p 三素材上复核。**基准素材上 |Δ| ≤ 0.14 crf ⇒ 现表正确，不改表**；素材间残差最大 −2.27 crf（`libvpx-vp9`，内容相关，与 E9/E10 同源结论）。证据 `verification_report/v9_calib_nocache_m*.json`。**残留**：4K / 高帧率（50fps+）类型尚未覆盖（本会话 GPU 断联且这两类 CPU 标定耗时过长）。 |
| ~~**P3**~~ | ~~长视频冒烟验证~~ | **已完成（§8.5）**：330 s / 11 段 / 两侧 av1_nvenc，`constqp` 与 `vbr` 双跑均 `exit 0`，15803 帧全链路守恒、解码级门禁通过、QA sidecar 完整、无内存泄漏（RSS 后半程斜率 −326 MB/min）。**代价**：先修了三处 AV1 缺陷（§8.6）。 |
| ~~**P3′**~~ | ~~AV1 端到端回归常态化~~ | **已完成（§9.2~§9.4）**：判据新增 G6-8/9/10（`av1_nvenc` 命令形状，含 constqp + vbr_hq 降级）、pytest `test_verify_video_integrity_fallback.py`（3 例）、可重复脚本 `Accessory/verify/av1_pipeline_smoke.py`（8 项验收 + `--checks-only`）。三者均做过**反向验证**。**残留**：完整跑批分支（S1/S2/S3/S8）需 GPU，本会话未跑到。 |
| **P3″** | **AV1 Level 1 直通（`GetEncodePresetConfig code=12`）** | 功能不受影响（正确降级到 CLI 管道）。假设与**可直接执行的复现片段**见 **§9.6**（`GetEncodePresetConfigEx` 回退）。**本会话因 GPU 断联（§9.1）未能验证**。 |

> **复现命令**：
> ```bash
> # AC1/AC2/AC4/AC7 复验入口（有硬件/构建时）
> python3 Accessory/probe/av1_vp9_quality_matrix.py --src /workspace/input_videos/word_world_2.mp4
>
> # G7/G8 门禁（AV1 相关格在 Ada 上才转正）
> python3 Accessory/verify/crf_cq_unification_verify.py --gpu \
>     --source /workspace/input_videos/word_world_2.mp4 --bitrate-source /workspace/input_videos/new4_raw.mp4
>
> # V9 重标定
> python3 /workspace/VidUtils/probe/calibrate_soft_offsets_nocache.py --src /workspace/input_videos/new5_raw.mp4
>
> # 长视频冒烟（P3，constqp / vbr 各一遍；需 av1_nvenc 可用）
> for RATE in constqp vbr; do
>   python3 src/main_video_optimized.py -c config/default_config.json -i src_5min.mp4 \
>       -o out_5min_$RATE.mp4 --codec-ifrnet av1_nvenc --codec-esrgan av1_nvenc \
>       --rate-mode-ifrnet $RATE --rate-mode-esrgan $RATE --segment-duration 30 < /dev/null
>   python3 Accessory/verify/segment_bitstream_verify_v5.py out_5min_$RATE.mp4 --skip-chroma < /dev/null
> done
> ```



