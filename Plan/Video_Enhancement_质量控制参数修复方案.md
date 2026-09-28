# Video_Enhancement 质量控制参数（crf / cq / qp / preset）修复方案

- 适用位置：`src/utils/quality_map.py`、`src/utils/video_utils.py`、
  `src/main_video_optimized.py`、`src/utils/convert_crf.py`（及共享换算表的两处副本）
- 共享真源：`src/utils/convert_crf.py` 的 `QUALITY_MAP`，与 `VidUtils/convert_crf.py`
  **必须逐条相等**（VidUtils 的判据 ⑨ 组会断言）
- 姊妹文档：`VidUtils/Plan/VidUtils_质量控制参数修复方案.md`
- 上机验证脚本：`VidUtils/probe/verify_nvenc_quality_gpu.py`（T4 / L40，重点 av1_nvenc）
- 本仓既有判据：`Accessory/verify/crf_cq_unification_verify.py`（G0 ~ G10）

---

## 0. 状态总览

> 2026-09-28 更新。**本轮范围：仅 VE 单侧**（VidUtils 由另一会话按其 V 系列方案处理）。
> 落地实现与文档原设想的差异见 **§5.1**；上机（T4/L40）验证方案见 **§5.2**。

| 编号 | 内容 | 优先级 | 状态 | 依据强度 |
|---|---|---|---|---|
| E0 | `QUALITY_MAP['av1_nvenc']` 的 hi 51 → 63（与 VidUtils 同步） | P0 | **已落地** | 实测（`ffmpeg -h encoder=av1_nvenc` → `-cq (0 to 63)`） |
| E1 | `to_constqp_qp()` 增加 **QP 尺度层**（AV1 族 ×4） | P0 | **已落地**；AV1 倍率标 `[待 L40 复核]` | 实测量程 + 推断倍率（需 L40 定案，见 §5.2 Gate 2） |
| E2 | `libsvtav1` / `libaom-av1` 不再收到非法 `-preset` | P0 | **已落地**（改为白名单，整数映射未做，见 §5.1） | 实测（libsvtav1 `-preset` 为 int `-2..13`，传 `medium` 直接解析失败） |
| E3 | VAAPI → `-qp`（归一到基准轴） | P0 | **已落地** | 实测（h264_vaapi 只有 `-qp (0 to 52)`）；**前提已纠正**，见 §5.1 |
| E4 | `_preset_supported()` 改白名单 | P0 | **已落地**（QSV/AMF/VT 保守排除，未上机） | 实测（ffmpeg 8.0.1：QSV 是 int `0..7`，本机无 AMF/VT） |
| E5 | `_PRESET_P_INDEX` 对齐 ffmpeg 官方枚举 | P1 | **VE 侧已落地**；与 VidUtils 的最终对齐待其 V10 | 实测官方枚举（p4=medium / p5=slow / p6=slower / p7=slowest） |
| E6 | `libsvtav1` / `libvpx-vp9` / `libx265` 等体积重标定 | P1 | **已落地**（由**并行会话**完成，非本轮） | 真实素材等体积标定（`input_videos/new5_raw.mp4`） |
| E7 | `--rate-mode` 取值表 | P1 | **已落地**（选"保持 3 档 + 明确写明"路线） | 取值表对比 |
| E8 | `--lookahead-depth` 放开量程 | P1 | **已落地**（`0~32`，**不是**文档原写的 `0~250`） | 见 §5.1：NVENC 硬件上限 32，配置层校验本已如此 |
| E9 | `CONSTQP_QP_OFFSET` 真实素材校准 + G7 扩 `av1_nvenc`/`libsvtav1` | P2 | 待办（需 GPU） | — |
| E10 | G7 增加"合成 vs 真实素材"双跑 | P2 | 待办（需 GPU） | — |

**本轮额外落地（不在 E0~E10 内）：**

| 编号 | 内容 | 状态 |
|---|---|---|
| P0′ | `Accessory/verify/crf_cq_unification_verify.py` 的 `PROJECT_ROOT` 失效修复（`tests/`→`Accessory/` 搬迁遗留，曾造成 19 个假 FAIL）+ 子进程 stdin 加固 | **已落地** |
| A5 | 判据新增 G1-8 / G2-13 / G3-7 / G3-8 / G5-12 + G6-7（AV1 constqp，需 Ada） | **已落地** |


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
| E1 | `_QP_SCALE` 单一倍率 + 另立 QP 量程表 | `_QP_MAP_OVERRIDE`（`QP = a·ref + b`，自带量程）+ `_qp_model()` 回退 `QUALITY_MAP` | 单一倍率会把 `librav1e` 算成 84，而其真实刻度是 `4·ref−4`（=80）；带截距的表可同时满足 AV1/rav1e，代码量相同 |
| E1 影响面 | 未说明 | `to_constqp_qp` 生产侧 4 个调用点**全部在 NVENC 分支内**（`ifrnet_video/main.py:1825`、`ifrnet_video/ffmpeg_io.py:904`、`realesrgan_video/main.py:835`、`realesrgan_video/ffmpeg_io.py:908`）⇒ 只影响 h264/hevc/av1_nvenc；T4 上 h264/hevc 为**恒等换算**，故 T4 生产**零影响** |
| E2 | 黑名单补两项 + 移植 svtav1 整数映射 + libaom `-cpu-used` | 只改白名单（svtav1/aom 自然被排除）；**整数映射未做** | 白名单已消除"命令直接失败"这个 P0；整数映射与 VidUtils 的 V9/V10 同源，单侧改会漂移，留待跨项目同批 |
| E3 | 称 VAAPI "不在 `_CQ_CODECS` ⇒ 返回 `-crf`" | **前提过期**：当时 VAAPI 就在 `_CQ_CODECS` 里 ⇒ 实际下发的是非法 `-cq:v 24 -b:v 0`（ffmpeg 直接报错，比原描述更严重）。实现按 VidUtils V6 语义（归一到基准轴 → `-qp`，夹 0~52），并同步 `supports_crf()` 排除 VAAPI | 实测 |
| E4 | 白名单 + 对 QSV 加档名映射 | 白名单只留 `libx264 / libx265 / h264_nvenc / hevc_nvenc / av1_nvenc`；**未加 QSV 档名映射** | ffmpeg 8.0.1 的 QSV `-preset` 是 int `0..7`（不是旧版档名），且 QSV 非 VE 生产路径；保守排除比猜映射安全 |
| E5 | 与 VidUtils 对齐 | 只按 ffmpeg 官方枚举改 VE 的**两份**副本（ifrnet + realesrgan，逐字节一致）；VidUtils 侧仍是旧档位，⑨ 组 `[9-preset]` 现报 `medium: VidUtils p5 vs VE p4`（note-only，不影响退出码） | 用户决定"仅 VE 单侧"，对方 V10 由另一会话执行 |
| E8 | `0~250` | `0~32` | NVENC 前向预看**硬件上限是 32**；配置层校验本已要求 `0<=la<=32`，故实际是"放开 CLI choices 到配置层允许的范围" |
| E6 | 本方案待办 | 由**并行会话**在 2026-09-28 11:34 落地（`libx265 0.9155/1.6385`、`libvpx-vp9 1.6198/−5.7553`、`libsvtav1 1.9450/−15.62`） | 非本轮范围；已据此同步 `REF21_EXPECTED`（libx265→21、libsvtav1→25、libvpx-vp9→28） |

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

