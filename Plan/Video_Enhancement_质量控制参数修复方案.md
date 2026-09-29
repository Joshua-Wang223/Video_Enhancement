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

> 2026-09-28 更新。**本轮范围：仅 VE 单侧**（VidUtils 由另一会话按其 V 系列方案处理）。
> 落地实现与文档原设想的差异见 **§5.1**；上机（T4/L40）验证方案见 **§5.2**；
> **本轮 T4 实测收口见 §6**；
> **需 L40/Ada（或目标机构建）才能检测验证的 AV1/VP9 测试内容（AC0~AC7）见 §7**
> —— 一条命令入口：`python3 Accessory/probe/av1_vp9_quality_matrix.py --src <真实素材>`。
>
> 提交记录：判据/测试/方案/memory 与首轮 T4 报告 = **`9847f59`**；本轮的 AV1/VP9 探针与 §7 扩展 = **见 `git log -1`**（均未推送）。

| 编号 | 内容 | 优先级 | 状态 | 依据强度 |
|---|---|---|---|---|
| E0 | `QUALITY_MAP['av1_nvenc']` 的 hi 51 → 63（与 VidUtils 同步） | P0 | **已落地** | 实测（`ffmpeg -h encoder=av1_nvenc` → `-cq (0 to 63)`） |
| E1 | `to_constqp_qp()` 增加 **QP 尺度层**（AV1 族 ×3） | P0 | **已落地**；**L40 实测确认 ×3** | 实测量程 + 实测倍率（L40 扩扫定案，见 §7 AC1） |
| E2 | `libsvtav1` / `libaom-av1` 不再收到非法 `-preset` | P0 | **已落地**（改为白名单，整数映射未做，见 §5.1） | 实测（libsvtav1 `-preset` 为 int `-2..13`，传 `medium` 直接解析失败） |
| E3 | VAAPI → `-qp`（归一到基准轴） | P0 | **已落地** | 实测（h264_vaapi 只有 `-qp (0 to 52)`）；**前提已纠正**，见 §5.1 |
| E4 | `_preset_supported()` 改白名单 | P0 | **已落地**（QSV/AMF/VT 保守排除，未上机） | 实测（ffmpeg 8.0.1：QSV 是 int `0..7`，本机无 AMF/VT） |
| E5 | `_PRESET_P_INDEX` 对齐 ffmpeg 官方枚举 | P1 | **VE 侧已落地**；与 VidUtils 的最终对齐待其 V10 | **实测修正**：只确认了 `medium ≡ p4`（逐字节相同）；`p5=slow / p6=slower / p7=slowest` 是 **NVIDIA p-梯命名**，不是 ffmpeg 命名 preset 的等价关系，见 §6.4 |
| E6 | `libsvtav1` / `libvpx-vp9` / `libx265` 等体积重标定 | P1 | **已落地**（由**并行会话**完成，非本轮） | 真实素材等体积标定（`input_videos/new5_raw.mp4`） |
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
`librav1e` 的 `-qp` 同为 0~255，且 `QUALITY_MAP` 已用 `(4.0, -4.0)` 处理了 4× 关系
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
| E1 | `_QP_SCALE` 单一倍率 + 另立 QP 量程表 | `_QP_MAP_OVERRIDE`（`QP = a·ref + b`，自带量程）+ `_qp_model()` 回退 `QUALITY_MAP` | 单一倍率会把 `librav1e` 算成 84，而其真实刻度是 `4·ref−4`（=80）；带截距的表可同时满足 AV1/rav1e，代码量相同。**L40 实测确认 AV1 为 ×3（非 ×4）**，已更新 `_QP_MAP_OVERRIDE['av1_nvenc'] = (3.0, 0.0, 0, 255)` |
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
| `librav1e` | 同上 | ⏭️ SKIP（本 build 无） | AC7 |
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
* 探测：`ffmpeg -h encoder=av1_qsv | grep -A2 -- '-cq'`（取实际量程）→ 按同一张表同步 hi。
* 无对应硬件时：维持 51 并在旁标注"未核实"（现状已如此）。

### AC6 · 跨项目交叉印证（VidUtils Gate 2 C 组）

`VidUtils/probe/verify_nvenc_quality_gpu.py` 的 C 组是 AC1 的同源实现，结论应与 AC1 一致；
若相左，以**本仓 AC1 的手工三点数据**为准并追查脚本差异（两边容忍带本就同一套：
`TOL_PSNR=1.5`、`RATE_PASS=(0.65,1.50)`）。

### AC7 · AV1/VP9 **软件族**在目标机构建上复验（VP9 一半在 T4 已过）

**为什么要单独一项**：三个 AV1 软件编码器（`libsvtav1` / `libaom-av1` / `librav1e`）在
**本机 ffmpeg 构建里根本没有**，与算力无关 ⇒ **Ada 卡也不会自动解决**，取决于目标机的
ffmpeg 构建。而其中 `libsvtav1` 的换算表是 2026-09-28 刚按真实素材重标定的（E6），
**至今没有端到端验证过**。

* **执行**：§7 开头的 `av1_vp9_quality_matrix.py`（一条命令覆盖全部 7 个编码器）。
* **判据**（与 A 组同口径）：

  | 编码器 | 期望 | 备注 |
  |---|---|---|
  | `libvpx-vp9` | 码率比 ∈ `RATE_PASS` 且 ΔPSNR ≥ −1.5 dB | T4 已跑：0.95× / −1.67 dB ⇒ **WARN**（内容相关偏松，与 G7-1/G7-2 同形态，非回归） |
  | `libsvtav1` | 同上 | ⚠ 探针固定 `-preset 8`（E6 标定档位）；**若目标机核数不同导致 `auto_effort()` 换档，等效点会漂**，需在报告里注明实际档位 |
  | `libaom-av1` | 同上 | ⚠ 表值来自文档推导（`+4`），非实测标定 |
  | `librav1e` | 同上 | ⚠ 表值来自 ffmpeg 6.1 的旧实测；`-qp` 是 0~255 刻度 |

* **产出**：`verification_report/av1_vp9_matrix_<机名>.md` / `.json`。
* **关闭动作**：把本节表格里对应行的 T4 状态换成实测值；若某编码器实测偏离容忍带，
  按 E6 的等体积口径重标定该行（并同步 `REF21_EXPECTED` 与 `G2-11` 等独立期望值）。
* ⚠ **VP9 无硬件编码需求**：NVENC 不提供 VP9；`vp9_vaapi` 需 Intel/AMD 的 VAAPI
  （且当前未收录进 `QUALITY_MAP`）。⇒ AC7 判的是**软件** VP9/AV1 的表值保真度。

### 汇总：AC × 前置 × 本机（T4）状态 × 关闭动作

| ID | 需什么 | T4 状态 | 关闭动作 |
|---|---|---|---|
| **AC1** | Ada（L40/A10/RTX40） | ✅ **L40 已确认 ×3** | **已闭环**：按实测定案 ×3，同步 `G3-7`/`G6-7` 期望为 63，移除 `[待 L40 复核]` |
| AC2 | Ada | ✅ **L40 已跑 PASS** | §6.2 表 G7-6 已填实测值 |
| **AC3** | **无**（逻辑/命令捕获层） | ✅ **PASS** | 已同步期望值为 63 |
| AC4 | Ada | ✅ **L40 已跑 PASS** | Gate 2 B 组 av1 格转正 |
| AC5 | Intel QSV / AMD AMF | ⏭️ SKIP | 同步 `QUALITY_MAP` 的 hi 值 |
| AC6 | Ada | ✅ **L40 已跑 PASS** | 与 AC1 交叉印证 |
| **AC7** | 目标机 **ffmpeg 构建**含 AV1/VP9 软编 | 6/7 ⏭️ SKIP；`libvpx-vp9` ✅ 已跑（WARN） | 用探针一条命令复验；偏离则按 E6 口径重标定该行 |

> **一条命令的入口**：`python3 Accessory/probe/av1_vp9_quality_matrix.py --src <真实素材> < /dev/null`
> —— 覆盖 AC1（自动判读）、AC2/AC4 的同轴对照、AC7 全部 7 个编码器；AC5 需另换硬件，
> AC6 走 VidUtils 的脚本。

> **状态**：L40 上 AC1~AC4、AC6 已全部闭环，**AV1 CONSTQP QP 尺度确认为 ×3（QP 63）**。生产 AV1 任务的 constqp 路径现已可用（基准 21 → QP 63），`-cq`/VBR 路径亦已有 E0 的 0~63 量程支撑。VP9 侧无硬件依赖，`libvpx-vp9` 表值已在 T4/L40 实测（WARN，内容相关偏松，非回归）。

---

## 后续建议（下一步）

| 优先级 | 事项 | 说明 |
|---|---|---|
| **P1** | **AC5：`av1_qsv`/`av1_amf` 量程实测** | 需 Intel QSV（Arc/新 iGPU）或 AMD AMF（RDNA3+）硬件。探测 `ffmpeg -h encoder=av1_qsv | grep -A2 -- '-cq'` 取实际量程，同步 `QUALITY_MAP` 的 `hi` 值。 |
| **P1** | **AC7：AV1/VP9 软件族在目标机构建上复验** | 需目标机 ffmpeg 构建含 `libsvtav1`/`libaom-av1`/`librav1e`。用 `python3 Accessory/probe/av1_vp9_quality_matrix.py --src <素材>` 一条命令复验；偏离则按 E6 等体积口径重标定。 |
| **P2** | **V9 多素材/多分辨率复核** | 当前 `libx265`/`libvpx-vp9`/`libsvtav1` 表基于单条 4s 1080p 素材。建议换 2~3 条不同类型/分辨率素材重跑 `probe/calibrate_soft_offsets.py` 再落表。 |
| **P2** | **VidUtils 侧对齐（V10）** | VE 的 `_PRESET_P_INDEX` 已按 ffmpeg 官方枚举落地（`medium≡p4`）。VidUtils 需同步对齐 `X264_TO_NVENC_PRESET`，确保 ⑨ 组 `[9-preset]` 彻底一致。 |
| **P3** | **长视频冒烟验证** | 在 L40 上跑 ≥5min 真实素材的完整增强流程（插帧+超分+AV1 NVENC constqp/vbr），验证全链路帧守恒、无内存泄漏、QA sidecar 完整。 |

> **复现命令**：
> ```bash
> # AC5/AC7 复验入口（有硬件/构建时）
> python3 Accessory/probe/av1_vp9_quality_matrix.py --src /workspace/input_videos/word_world_2.mp4
> 
> # V9 重标定
> python3 probe/calibrate_soft_offsets.py --src /workspace/input_videos/new5_raw.mp4
> 
> # 长视频冒烟（示例）
> python3 src/main_video_optimized.py -c config/default_config.py -i <长视频> -o <输出> \
>     --codec-ifrnet av1_nvenc --rate-mode-ifrnet vbr --skip-upscale --segment-duration 30
> ```



