# Video_Enhancement T4 专项执行方案 —— h264_nvenc / hevc_nvenc 等质量标定（M4·T4 侧）

> **本方案是 2 份 GPU 专项方案中的「T4 母版」**。共享方法论、harness 改动、优秀做法、
> 验收门禁的**权威定义在本文件**；L40 方案（`Plan/PROMPT_L40_AV1等质量标定专项执行方案.md`）
> 只写 AV1 差异，共享部分引用本文件，避免两份实现漂移。
>
> **上位文档**：`Plan/PROMPT_等质量换算立项.md`（§0.0 状态 / §7.1 B 组待办 / §11.3 GPU 侧）
> **姊妹方案**：`Plan/Video_Enhancement_质量控制参数修复方案.md`（E0~E10 / AC1~AC7 / §8.6 三处 AV1 修复）
> **标定报告**：`Plan/等质量换算表_实现与标定报告.md`（第八版全池表值 / LOO / 门禁分档）
> **总览指南**：`Accessory/docs/EQQ_CALIBRATION_OVERVIEW.md`
>
> 状态（2026-10-06 更新）：**T4-1~T4-9 已全部执行完毕，门禁全绿**（执行记录见 §13）。
> 本容器有 Tesla T4（`nvidia-smi` 可用），`h264_nvenc`/`hevc_nvenc` 实跑 rc=0。
> - **CQ 轴（T4-1/2）**：17 素材 GPU 标定 → `QUALITY_MAP['h264_nvenc']=(0.9295, 6.2523)`
>   LOO 3.98、`['hevc_nvenc']=(1.1116, 2.1606)` LOO 5.81。已并入 `8f0a605`，与 VidUtils 逐字相等（⑨ 组 14/14）。
> - **QP 轴（T4-3/4）**：17 素材 `--axis qp` 标定 → `QUALITY_MAP_QP['h264_nvenc']=(0.9704, 1.4767)`
>   LOO 3.47、`['hevc_nvenc']=(1.1083, -2.9183)` LOO 3.72（仅 VE，D2b）。
> **CR-1/CR-2 均已收口**；CR-2 因 FFmpeg 9.0 移除 `vbr_hq`/`qvbr` **二次修订**为 CLI/harness
> **裸 `vbr`**（`-tune hq` 是默认值、固定 CQ 下 multipass 不升 VMAF ⇒ 二者改为显式 opt-in；
> SDK 路径不动）——见 §4.1/§12.5 与
> `Plan/T4_NVENC_vbr_hq移除_验证专项.md`（§11 二次校正）。VU 侧已重新同步。
> - **L40 协同完成**：L40 侧 `av1_nvenc` CQ/QP 双轴标定已完成（§13.7），跨仓 ⑨ 组 14/14 一致，**全 GPU 专项收口**。

---

## 0. 一句话范围

把**等质量口径**从「仅软编 6 档」扩到 **T4 可跑的 `h264_nvenc` / `hevc_nvenc`**，
补齐两条轴：

- **CQ 轴**（`-cq:v`，VBR：生产 vbr_hq 路径 / SDK vbr）→ 落 `QUALITY_MAP`；
- **QP 轴**（`-qp`，CONSTQP：SDK Level 1 直通 + ffmpeg CLI constqp）→ 落 `QUALITY_MAP_QP`（D2b）。

AV1（`av1_nvenc`）**不在 T4 范围**（Turing 无 AV1 NVENC，实跑报 `No capable devices found`），
见 L40 专项。

---

## 1. 环境体检（Gate 0，最先做，否则后续判读全不可信）

```bash
cd /workspace/Video_Enhancement
nvidia-smi -L                                                        # 期望 Tesla T4
python3 -c "import torch;print(torch.cuda.get_device_name(0),torch.version.cuda,torch.cuda.is_available())"
ffmpeg -hide_banner -encoders | grep -E 'h264_nvenc|hevc_nvenc|av1_nvenc'
ffmpeg -hide_banner -version | head -1

# NVENC 可用性：唯一可靠判据是【实跑一帧】，不是 -h encoder（Turing 上选项表照样打印）
for C in h264_nvenc hevc_nvenc av1_nvenc; do
  ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 \
         -c:v $C -f null - < /dev/null >/dev/null 2>&1
  echo "$C rc=$?"
done
# 预期：h264_nvenc rc=0 / hevc_nvenc rc=0 / av1_nvenc rc≠0（⇒ AV1 归 L40）
```

⚠ 判据：`rc≠0` 只对 `av1_nvenc` 成立才算「符合 T4 预期」；若 `h264_nvenc`/`hevc_nvenc`
也 rc≠0，先修 NVENC 环境（驱动/SDK），**不要往下走**。

---

## 2. 待办任务清单（T4 / L40 分列）

| ID | 任务 | 轴 | 产出 | T4 | L40 | 依据 |
|---|---|---|---|:--:|:--:|---|
| **T4-1** | `h264_nvenc` CQ 等质量标定 | `-cq:v` | `QUALITY_MAP['h264_nvenc']` | ✅ | — | 立项 §7.1 B |
| **T4-2** | `hevc_nvenc` CQ 等质量标定 | `-cq:v` | `QUALITY_MAP['hevc_nvenc']` | ✅ | — | 立项 §7.1 B |
| **T4-3** | `h264_nvenc` QP 等质量标定 | `-qp` | `QUALITY_MAP_QP['h264_nvenc']` | ✅ | — | D2b |
| **T4-4** | `hevc_nvenc` QP 等质量标定 | `-qp` | `QUALITY_MAP_QP['hevc_nvenc']` | ✅ | — | D2b |
| **T4-5** | harness 扩展（SWEEP/LOCK/FLAG/`--axis`/落表） | — | 代码 | ✅ | 复用 | 本方案 §4 |
| **T4-6** | `_qp_model` 改「模式感知」 | — | 代码 | ✅ | ✅ | 本方案 §4.4 |
| **T4-7** | G7/G8 GPU 画质与码率门禁 | — | 报告 | ✅ | ✅ | 方案 §5.2 Gate1 |
| **T4-8** | h264/hevc 生产管线长视频冒烟 | — | 报告 | ✅ | — | 方案 §5.2 Gate4 |
| **T4-9** | 跨仓 `QUALITY_MAP` 同步 + ⑨ 组 | — | 门禁 | ✅ | ✅ | 立项 §8 |
| — | **L40-1** `av1_nvenc` CQ 等质量 | `-cq:v` | `QUALITY_MAP['av1_nvenc']` | — | ✅ | L40 方案 |
| — | **L40-2** `av1_nvenc` QP 等质量 | `-qp` | `QUALITY_MAP_QP['av1_nvenc']` | — | ✅ | L40 方案 |
| — | **L40-3** AC1 复验（QP ×3） | `-qp` | 报告 | ⏭️ SKIP | ✅ | 方案 §7 AC1（已闭环，复验） |
| — | **L40-4** AC2（G7-6 `-cq`） | `-cq:v` | 报告 | ⏭️ SKIP | ✅ | 方案 §7 AC2 |
| — | **L40-5** AV1 长视频冒烟 S1/S2/S3/S8 | — | 报告 | ⏭️ SKIP | ✅ | 方案 §9.4 |
| — | **L40-6** AV1 Level 1 `code=12`（可选） | — | 结论 | — | ✅ | 方案 §9.6 |

> **分工铁律**：T4 与 L40 **不可互相替代**。T4 无 AV1 NVENC；L40 虽也能跑 h264/hevc，
> 但本仓基线（`SIZE_MAP` 的 h264/hevc 偏移 +5/+7.5）在 T4 上标定，L40 上重复标会引入
> 第二套值 ⇒ 本方案明确规定 **h264/hevc 只在 T4 标**，L40 只标 AV1。

---

## 3. 优秀做法吸收清单（来自既有方案，逐条落地）

| # | 做法 | 出处 | 在本专项的落地 |
|---|---|---|---|
| P1 | **先环境体检、再动手**；硬件能力只认「实跑一帧」 | 方案 §5.2 步骤0 / §7 AC0 | §1 的 Gate 0 |
| P2 | **常量化测量用真实素材**，不用合成（合成会误判过配） | 方案 E10 / G7-8 | 用 `input_videos/eqq_calib/` 的真实切片 |
| P3 | **无缓存 prep**：每次独立 workdir + prep 重建 + md5 审计 | 立项 §6.2 / A10 | harness 已有（`make_prep` 先 unlink） |
| P4 | **VMAF `n_subsample=1` 铁律**（>1 偏置随编码器而异） | 报告 §1.5 | harness 默认 1，GPU 必须沿用 |
| P5 | **PSNR 口径**：`-v info` + 显式 `[0:v][1:v]`；`-v error` 会让 ΔPSNR 恒 0 | 立项 §6.1 K3 | harness `_filters_pass` 已合规 |
| P6 | **锚点统一 `18/21/24/27/30`**，两仓一致 | 报告 §3.1.1 | harness `ANCHOR_CRFS` |
| P7 | **断点续跑**：逐点落 `points.json` | harness `--resume` | GPU 长跑必备 |
| P8 | **留一交叉验证（LOO）为落表前置门禁**，单素材判据不能为表值背书 | 报告 §5.1 / §3.2.1 | 落表前跑 `loo_equal_quality.py` |
| P9 | **池化按素材去重**；同 key 取均值；池化前打印 4 个数核对完整性 | 报告 §3.1.0 / §3.1.5 | GPU 点入池后复核 |
| P10 | **双仓两张表逐条相等**（⑨ 组断言） | 立项 §1.3 / §8 | §6 跨仓同步 |
| P11 | **所有脚本加 `< /dev/null`**（否则后台进程组被 SIGTTOU 整组停住） | 方案 §4 铁律1 | 全部命令 |
| P12 | **判据里硬编码的期望值要跟随上游改动**（A15 教训：写死 84） | 方案 §8.2 / A15 | §4.4 期望值同步 |
| P13 | **命令形状断言 + 反向验证**（断言必须真的会 FAIL） | 方案 §9.2 | 新增/改判据必须反向验证 |
| P14 | **不为消警放宽判据**，保留严格判据并标注「合成已知边界/真实 PASS」 | memory `feedback_keep_strict_criteria_annotate` | Gate1 的 WARN 处理 |
| P15 | **动手前先跑基线实测反推真实待办**，不照抄状态表 | memory `feedback_verify_baseline_first` | §5 每步先跑基线 |
| P16 | **无 GPU 时只给方案不落地**（会改 GPU 运行时语义的改动不预先合并） | memory `feedback_no_gpu_work_mode` | 本方案定位 |
| P17 | **长会话 GPU 可能被回收**：任何 GPU 结论都须当场复跑 | 方案 §9.1 | 报告写「实测时刻」 |

---

## 4. harness 扩展（**已落地，2026-10-04**）

> 目标文件：`Accessory/probe/calibrate_equal_quality.py`（VE）。
> **实施方式：移植 VidUtils 已实现的接口**（VU 侧 harness 早已完成 NVENC 支持 + 跨仓态势，
> 见 §12 态势），VE 侧**只在其上叠加 `--axis` 扩展**，避免两份实现分叉。
>
> **落地状态**：`--selftest` **39 项全过**；CPU 干跑（`--quick --codecs libx265`）端到端通过并
> 打印跨仓态势；无 GPU 时硬件探测优雅降级（exit 2 明确报错）。
> **2026-10-04 方案 A 落地后复跑（T4）**：`--selftest` 全绿；`crf_cq_unification_verify.py
> --no-gpu --quick` = **PASS=94 / FAIL=0 / SKIP=11**；`plan_implementation_gate.py` 静态
> **50/48/0/2** + 行为 **46/46**（合并 **96/94/0/2**）（详见 §12.4 验证记录）。

### 4.1 新增 NVENC 档位与配套参数（已落地）

`SWEEP` / `HW_CODECS` / `BASE_LOCK` / `QUALITY_FLAG` 均已加 NVENC（**与 VU 同值**，便于横向可比）：

```python
SWEEP = {
    ...
    'h264_nvenc': [10, 15, 19, 23, 26, 29, 33, 37, 42, 48, 51],   # -cq 0~51
    'hevc_nvenc': [12, 17, 21, 25, 28, 32, 36, 40, 45, 51],
    'av1_nvenc':  [12, 18, 23, 27, 31, 36, 41, 47, 54, 63],       # -cq 0~63
}
HW_CODECS = {'h264_nvenc', 'hevc_nvenc', 'av1_nvenc'}   # 需「实编探测」的档位
BASE_LOCK = {
    ...
    # ✅ CR-1（preset）已收口：统一 **p4**（以 VE 的 E5 为准）。
    # ✅ CR-2（rate control）已裁定（路线 B；2026-10-04 因 FFmpeg 9.0 更新）：
    #   · h264/hevc → **裸 `-rc:v vbr`**（FFmpeg 9.0 移除 vbr_hq/qvbr；`-tune hq` 是默认值
    #     写了等于没写、固定 CQ 下 multipass 不升 VMAF ⇒ 均**不下发**，见 knowledge §5.1）。
    #     ⚠ 2026-10-04 二次校正（原为 `vbr -tune hq -multipass fullres`）；
    #       `-tune`/`-multipass` 改为显式 opt-in（`--nvenc-tune-*`/`--nvenc-multipass-*`）。
    #       VE SDK 侧内部仍走 RC_VBR_HQ(32)——T4 实测驱动 13.0 仍接受，故仅 CLI/harness 迁移。
    #   · av1_nvenc 的 `-rc` 只接受 constqp/vbr/cbr ⇒ **plain `vbr`**（与 VE 生产降级口径一致）。
    #   ⚠ VU 侧同步（handoff）：VU harness/探针应保持裸 `-rc:v vbr`，否则 ⑨ 组变红。
    'h264_nvenc': ['-rc:v', 'vbr', '-b:v', '0', '-preset', 'p4'],
    'hevc_nvenc': ['-rc:v', 'vbr', '-b:v', '0', '-preset', 'p4'],
    'av1_nvenc':  ['-rc:v', 'vbr',    '-b:v', '0', '-preset', 'p4'],
}
QUALITY_FLAG = {..., 'h264_nvenc': '-cq:v', 'hevc_nvenc': '-cq:v', 'av1_nvenc': '-cq:v'}
```

配套的 GPU 能力探测（移植自 VU）：`run_try` / `gpu_info` / `make_probe_src` /
`probe_hw_codec`，CLI `--require-codecs` / `--expect-av1`（fail-fast，exit 2）。

### 4.2 VE 特有 `--axis {cq,qp}`（CQ 轴 / CONSTQP 轴分离，已落地）

`-cq:v` 与 `-qp` 是两条刻度、两套 rate control，不能一次扫完。新增开关，同一次运行
全部档位共用一条轴：

```python
AXES = ('cq', 'qp')
QP_LOCK = {                                          # qp 轴：只锁 rc + preset
    'h264_nvenc': ['-rc:v', 'constqp', '-preset', 'p4'],
    'hevc_nvenc': ['-rc:v', 'constqp', '-preset', 'p4'],
    'av1_nvenc':  ['-rc:v', 'constqp', '-preset', 'p4'],
}
QP_LIMITS = {'av1_nvenc': 255}                       # QP 轴量程（其余 0~51）
QP_SWEEP = {                                         # AV1 qp ≈ 3×基准轴 ⇒ 右移
    'h264_nvenc': [10, 15, 19, 23, 26, 29, 33, 37, 42, 48, 51],
    'hevc_nvenc': [10, 15, 19, 23, 26, 29, 33, 37, 42, 48, 51],
    'av1_nvenc':  [30, 45, 60, 75, 90, 105, 120, 150, 180, 210, 255],
}

def _qflag(key, axis):    return '-qp' if axis == 'qp' else QUALITY_FLAG[_ffcodec(key)]
def _axis_lock(key, axis): ...
def _axis_range(key, axis): ...                      # cq→_table_range；qp→QP_LIMITS
def _sweep_for(key, axis): ...                       # cq→SWEEP；qp→QP_SWEEP
def encode_cmd(ref, key, value, out, axis='cq'): ... # 纯函数，供 selftest 断言命令形状
```

- `encode()` 增 `axis` 形参；锚点恒走 `cq`（libx264 `-crf`）。
- 聚合段量程改用 `_axis_range()`（替换原 `CRF.QUALITY_MAP[...]` 直索引，**消除 KeyError**）。
- `--axis qp` 对软编**直接 exit 2**（软编 QP 轴 = CRF 轴，无需单独标定）。
- ⚠ **两条轴的点必须分目录**（points 键不含轴；同目录混跑会互相污染）。

### 4.3 单素材/批量/落表脚本接入（已落地）

- `eqq_calibrate_clip.py`：新增 `--axis`，默认扫描点按轴取（`CH._sweep_for`）；
  打印带 `_qflag`；**两轴须用不同 `--out` workdir**。
- `eqq_calibrate_batch.py`：新增 `--axis` 透传到 clip；GPU 跑批用 `--jobs 1`
  （**NVENC 会话不能并发多进程抢同一张卡**，见 §9 风险）。
- 新增 side 目录：GPU 点入 `Accessory/data/eqq_calibration/points/gpu_t4_{cq,qp}/`
  （L40 用 `gpu_l40_{cq,qp}/`）。
- `eqq_pool_fit_table.py`：`TIERS` 加三档 NVENC；`GATE` 沿用软编 ≤5.9；
  落表候选 `tag` 改为**从目标轴量程取**（`CH._axis_range`），不再硬编码 `255/51/63`；
  新增 `--axis`（cq→候选 `QUALITY_MAP`；qp→候选 `QUALITY_MAP_QP`）。
  **回归**：现有软编 6 档仍**逐位复现库内表值**（rc=0），NVENC 无数据时打印「拟合失败」并跳过。

### 4.4 `quality_map.py` 改「模式感知」（**已落地 + 已验证**）

原 `_qp_model()` **无条件优先** `_QP_MAP_OVERRIDE`，会让新落表的 `QUALITY_MAP_QP['h264_nvenc']`
**永远被遮蔽**。已改为**按口径分流**：

```python
def _qp_model(codec, table=None):
    c = str(codec).lower()
    if get_quality_mode() == 'quality':
        if c in QUALITY_MAP_QP:          # D2b 标定表优先（NVENC 标定后由此生效）
            return QUALITY_MAP_QP[c]
        if c in _QP_MAP_OVERRIDE:        # 未标定的硬编/VAAPI 回退
            return _QP_MAP_OVERRIDE[c]
        return _active_table(table).get(c)
    # size 口径：完全保持改造前行为（override → 活动表），不受 QUALITY_MAP_QP 影响
    if c in _QP_MAP_OVERRIDE:
        return _QP_MAP_OVERRIDE[c]
    return _active_table(table).get(c)
```

> ⚠ **勘误**：本方案早期草稿曾写成「quality 分支判断 + 末尾再无条件判断 `QUALITY_MAP_QP`」
> 的三分支形式 —— 那会让 **size 口径的软编也读 `QUALITY_MAP_QP`**，破坏判据 G3/G6 的
> 「零侵入」。**以本节为准**。

**为什么安全（已实测验证）**：`_qp_model()` 按口径分流后，size 口径仍走 `_QP_MAP_OVERRIDE`
（不受 `QUALITY_MAP_QP` 影响），quality 口径走标定表。⚠ 判据口径已于 2026-10-04 由 size
**迁移到 quality**（见 §13.3 后续注），故 G3 期望值随之更新为 quality 值。
（历史：迁移前该改造是**恒等变换**，已用「旧模块(HEAD) vs 新模块」全矩阵对比 `2 模式 × 20
编码器 × 17 值 = 680 组`逐位一致验证。）回归测试：`Accessory/test/test_quality_map_qp_mode.py`。

⚠ **同步复核独立期望值**：若标定后 `QUALITY_MAP_QP['h264_nvenc']` 在基准轴 21 处
不等于 21（即 a≠1.0 或 b≠0），须同步：
- `crf_cq_unification_verify.py` 的 G3-1（h264 26→21）/ G3-2（hevc 28→20）；
- `REF21_EXPECTED`（G1-2）与 G2-11（**不自动跟随表，必须人审**，见 P12）。
若标定确认仍为恒等，则无需改动，仅更新注释来源。

### 4.5 跨仓态势（**已落地**，VU 对等方案的对称能力）

移植 VU 的 `cross_repo_status()` / `_cmp_tables()` / `_repo_role()` / `_default_sibling()`，
并新增 `_print_cross_repo()`。开跑前打印 + 写入 `report.json['cross_repo']`：

```
跨仓态势（本仓 VE；对侧 /workspace/VidUtils）：
  SIZE_MAP: ✅ 两仓逐条相等
  QUALITY_MAP: ✅ 两仓逐条相等
  对侧 harness: /workspace/VidUtils/probe/calibrate_equal_quality.py  ⚠ 与本仓不同版（VE 有 --axis 扩展，属预期差异）
  对侧方案文档: 3 份
```

CLI `--sibling-root` 可显式指定对侧；不给则自动找同级 `VidUtils`。详见 **§12**。

### 4.6 回归保护（同批上线）

- 新增/改造判据时**必须做反向验证**（临时回退修复 → 断言应 FAIL）。
- 本轮的 CPU 回归：`--selftest` 39 项、新 pytest 4 例、判据/门禁/pytest 与基线一致。
- 新增 manifest/文案更新：`Accessory/data/eqq_calibration/MANIFEST.md`、
  `Accessory/docs/EQQ_CALIBRATION_OVERVIEW.md`（GPU 侧命令行与素材清单）。

---

## 5. 执行步骤（T4）

### 5.1 基线（改动前后都跑，作对照）

```bash
# 纯 CPU 基线（本容器即可跑，用于确认「改动前」门禁数）
python3 Accessory/verify/crf_cq_unification_verify.py --quick < /dev/null   # FAIL=0（除 cv2 环境项）
python3 Accessory/verify/plan_implementation_gate.py < /dev/null            # FAIL=0
python3 Accessory/probe/calibrate_equal_quality.py --selftest               # 39 项全过
python3 Accessory/probe/eqq_pool_fit_table.py < /dev/null                   # 逐位复现库内表值 rc=0
```

### 5.2 素材（真实切片，覆盖 6 类内容，≤7 条控制成本）

`input_videos/eqq_calib/` 下的 720p 切片，**必须加 `--src-is-prep`**（切片已是参考片，
再 prep 一次会引入 crf10 重编码，锚点 VMAF 偏移 0.02~0.29，与库内不可比）。推荐 7 条：

```
6s/live_kids_play_src1280x720.mp4      实拍
6s/tv_bbc_s01e01_src1920x1080.mp4      实拍剧集
6s/cganim_edu_wordworld_src720x576.mp4 动画（门禁同源）
6s/anim2d_subs_tobot_src*.mp4          二维动画+字幕
6s/doc_dark_earth_src*.mp4             暗场/纪录片
6s/screen_ui_code_src*.mp4             屏幕 UI
10s/live_texture_frog_src*.mp4         高细节纹理
```

（全部存在于 `input_videos/eqq_calib/manifest_6s.json` / `manifest_10s.json`，用 `--src-is-prep`
与 `--duration 6` 或 `10` 与切片实际时长对齐，**开跑前先 `ffprobe` 核实未被静默截断**。）

### 5.3 标定（两条轴各一轮）

```bash
# ── CQ 轴（VBR 路径）────────────────────────────────────────────
for M in live_kids_play tv_bbc_s01e01 cganim_edu_wordworld anim2d_subs_tobot \
         doc_dark_earth screen_ui_code live_texture_frog; do
  python3 Accessory/probe/eqq_calibrate_clip.py \
      --src input_videos/eqq_calib/<side>/${M}_src*.mp4 \
      --out temp/eqq_gpu_t4/cq_${M} \
      --tiers h264_nvenc,hevc_nvenc \
      --duration <6|10> --src-is-prep < /dev/null
done

# ── QP 轴（CONSTQP 路径）────────────────────────────────────────
#   同 harness，新增 --axis qp（见 §4.2）；档位同上
for M in <同上 7 条>; do
  python3 Accessory/probe/eqq_calibrate_clip.py \
      --src input_videos/eqq_calib/<side>/${M}_src*.mp4 \
      --out temp/eqq_gpu_t4/qp_${M} \
      --tiers h264_nvenc,hevc_nvenc --axis qp \
      --duration <6|10> --src-is-prep < /dev/null
done
```

> ⚠ 单点成本：VMAF（`n_subsample=1`）约 20~45 s，编码在 GPU 上为秒级 ⇒ 每条素材每轴
> 约 15 点 ≈ 8~12 min。7 素材 × 2 档 × 2 轴 ≈ **4~6 h**（含锚点）。可把 7 条素材拆到
> 多台机/多个时段，但**同机禁止并发 NVENC 会话**。

### 5.4 入池 → LOO → 落表候选

```bash
python3 Accessory/probe/eqq_pool_fit_table.py \
    --sides 6s,10s,legacy10s,gpu_t4_cq --axis cq --out /tmp/table_t4.txt < /dev/null
# 判据：新增 h264_nvenc / hevc_nvenc 两行 LOO ≤5.9；顺序无关断言通过；0 评估点记 inf
```

### 5.5 写入表 + 同步（§6）后跑全套门禁（§7）

### 5.6 生产管线冒烟（h264/hevc）

```bash
# 与 方案 §5.2 Gate4 同口径：hevc + LA=8 帧守恒；额外补一条 h264
python3 run.py -i /tmp/seg_src_5s.mp4 -o /tmp/seg_hevc_la8.mp4 \
    --mode interpolate_then_upscale \
    --codec-ifrnet hevc_nvenc --codec-esrgan hevc_nvenc \
    --rate-mode-ifrnet vbr_hq --lookahead-depth-ifrnet 8 < /dev/null
python3 Accessory/verify/segment_bitstream_verify_v5.py /tmp/seg_hevc_la8.mp4 < /dev/null
# 通过判据：frames==packets、段首无连 IDR、frame_num 单调、无色度异常簇
```

---

## 6. 落表与跨仓同步

| 表 | 键 | T4 新增 | 同步要求 |
|---|---|---|---|
| `QUALITY_MAP`（等质量） | `h264_nvenc` / `hevc_nvenc` | CQ 轴标定值 `(a, b, 0, 51)` | **两仓逐条相等**（⑨ 组）；VidUtils 侧须同步 |
| `QUALITY_MAP_QP`（QP 轴，仅本仓） | `h264_nvenc` / `hevc_nvenc` | QP 轴标定值 `(a, b, 0, 51)` | 仅 VE（VU 无 ctypes 直连层） |

```bash
A=/workspace/Video_Enhancement/memory; B=/root/.codebuddy/projects/workspace-Video_Enhancement/memory
python3 /workspace/VidUtils/verify/verify_quality_mapping.py < /dev/null   # ⑨ 组须 14/14
```

⚠ 若 VidUtils 侧不在同一台机/同一会话，**必须在方案里显式登记「VU 侧待同步」**，
不得只改 VE 单侧后宣称完成（⑨ 组会红）。

---

## 7. 验收门禁（全绿才算完成）

| 门 | 命令 | 判据 |
|---|---|---|
| harness 自测 | `python3 Accessory/probe/calibrate_equal_quality.py --selftest` | 39 项全过 |
| 落表器 | `python3 Accessory/probe/eqq_pool_fit_table.py --sides …,gpu_t4_cq --axis cq` | 2 行 LOO ≤5.9 + 顺序无关 ✅ |
| 等质量专用判据 | `python3 Accessory/verify/verify_equal_quality.py < /dev/null` | 主门禁 ΔVMAF 达标；退出 0 |
| 本仓静态判据 | `python3 Accessory/verify/crf_cq_unification_verify.py --quick < /dev/null` | **FAIL=0**（G3 期望值同步后） |
| 本仓门禁 | `python3 Accessory/verify/plan_implementation_gate.py < /dev/null` | **FAIL=0** |
| pytest | `python3 -m pytest Accessory/test -q` | 全绿 |
| GPU 判据 | `python3 Accessory/verify/crf_cq_unification_verify.py --gpu --source <真实素材> --bitrate-source <高熵素材> < /dev/null` | G7/G8：**FAIL=0**（WARN 属已知内容相关偏松） |
| 跨仓真源 | `cd /workspace/VidUtils && python3 verify/verify_quality_mapping.py` | ⑨ 组 14/14 |
| 生产冒烟 | §5.6 | 帧守恒 ✅ |

---

## 8. 回滚

| 触发 | 动作 |
|---|---|
| LOO 超门禁（>5.9） | 不落表；保留 `gpu_t4_{cq,qp}` 原始 points 与报告，标注失败项与根因（**不自行放水**） |
| `QUALITY_MAP_QP` 标定后 G3 期望值变动导致判据红 | 复核是「表值正确、判据过期」还是「表值错」；前者改判据期望值（P12），后者回滚表值 |
| `_qp_model` 模式感知改造引发意外 | 单点回退：恢复「override 优先」并移除 `QUALITY_MAP_QP` 的 NVENC 行 |
| 生产冒烟帧守恒失败 | 与 `QUALITY_MAP` 无关（等质量只改质量参数数值）⇒ 查编码线程/LA，不在此专项范围 |

---

## 9. 风险与坑

| 风险 | 对策 |
|---|---|
| **T4/L40 混用** | h264/hevc 只在 T4 标；AV1 只在 L40 标；本方案明令 |
| **同机并发 NVENC 会话** | `--jobs 1`；并发会抢 encoder session 并污染耗时/结果（共享 GPU 主机，先查 `nvidia-smi`） |
| **`-rc:v vbr_hq` 对 av1 非法** | 已在 §4.1 锁定 av1 用 `vbr`；L40 方案复用 |
| **`_QP_MAP_OVERRIDE` 遮蔽新表** | §4.4 模式感知改造是**前置**，否则落表不生效 |
| **判据硬编码期望值过期** | §4.4 的同步复核清单（P12） |
| **素材短于口径被静默截断** | 开跑前 `ffprobe` 核实（P3） |
| **VMAF 子采样偏置** | 恒 `n_subsample=1`（P4） |
| **会话中途 GPU 被回收** | 每步当场复跑；报告写实测时刻（P17） |
| **`< /dev/null` 缺失致 SIGTTOU 假挂起** | 全部命令加（P11） |

---

## 10. 复现命令汇总（一条条可复制）

```bash
cd /workspace/Video_Enhancement
# 0 体检
bash -c 'nvidia-smi -L; ffmpeg -hide_banner -f lavfi -i testsrc2=size=320x240:rate=30:duration=1 -c:v h264_nvenc -f null - < /dev/null; echo rc=$?'
# 1 基线
python3 Accessory/probe/calibrate_equal_quality.py --selftest
python3 Accessory/probe/eqq_pool_fit_table.py < /dev/null
# 2 标定（CQ 轴，单素材示例）
python3 Accessory/probe/eqq_calibrate_clip.py \
    --src ../input_videos/eqq_calib/6s/live_kids_play_src1280x720.mp4 \
    --out temp/eqq_gpu_t4/cq_live_kids_play --tiers h264_nvenc,hevc_nvenc \
    --duration 6 --src-is-prep < /dev/null
# 3 入池落表
python3 Accessory/probe/eqq_pool_fit_table.py --sides 6s,10s,legacy10s,gpu_t4_cq --axis cq < /dev/null
# 4 门禁
python3 Accessory/verify/verify_equal_quality.py < /dev/null
python3 Accessory/verify/crf_cq_unification_verify.py --quick < /dev/null
python3 Accessory/verify/plan_implementation_gate.py < /dev/null
cd /workspace/VidUtils && python3 verify/verify_quality_mapping.py < /dev/null
```

---

## 11. 与 L40 方案的边界

- 本方案定义**共享改动**（§4 harness / `_qp_model`）、**共享方法论**（§3）、**共享门禁**（§7）。
- L40 方案只写 AV1 差异（AV1 CQ 量程 0~63、QP 轴 0~255 且实测 ×3、AC1~AC2 复验、
  AV1 冒烟、Level 1 `code=12`）。harness 对 AV1 的 `-cq:v` 走**默认 rc + `-b:v 0`**
  （与 VU 同款；不显式下发 vbr_hq，故不触发 §8.6-③ 的非法 rc）。
- **改动共享代码前，先在 T4 落地并跑通本方案门禁**，L40 直接复用同一份 harness（避免两份实现漂移）。

---

## 12. VidUtils（VU）对等方案态势与协同契约

> **这是本专项「协同处理」的核心**：VU 侧已有**同构的两份专项方案**与**已实现的 harness**，
> 且其 harness 已内建跨仓态势。两侧必须按同一契约行动，否则共享表（⑨ 组断言逐字相等）会红。
> **VU 侧路径**：`/workspace/VidUtils`。

### 12.1 VU 对等方案现状（2026-10-03 快照）

| 项 | VU 侧 | VE 侧（本仓） |
|---|---|---|
| 专项方案 | `Plan/VidUtils_等质量标定_T4专项执行方案.md`、`..._L40_AV1专项执行方案.md` | 本文件、`PROMPT_L40_AV1…` |
| 任务编号 | 全局 `G0~G7`；T4 `T0~T6`；L40 `A0~A5` | 本文件 `T4-1~T4-9`／L40 `L40-1~L40-8` |
| harness | `probe/calibrate_equal_quality.py`（**已支持 NVENC + 跨仓态势**） | `Accessory/probe/…`（已移植 VU 实现 + `--axis` 扩展） |
| 落表真源 | `convert_crf.py`（根目录） | `src/utils/convert_crf.py` |
| QP 轴表 | **无**（`to_constqp_qp` = 基准轴 × `_QP_SCALE`） | `QUALITY_MAP_QP`（D2b，**仅 VE**） |
| 门禁 | `verify/verify_equal_quality.py`（`SOFT` 只列软编）、`verify/verify_quality_mapping.py` ⑨ 组 | 同构 |
| 上机探针 | `probe/verify_nvenc_quality_gpu.py --expect-av1`、`probe/t4_acceptance.py` | `Accessory/probe/av1_vp9_quality_matrix.py` 等 |
| 生产 preset 默认 | ✅ **已统一 `p4`**（生产 `DEFAULT_PRESET_GPU="p4"` + harness/探针 p4） | **`medium→p4`**（跨仓标准） |

**任务编号对照**（避免两侧会话说不到一起）：

| VU | VE | 内容 |
|---|---|---|
| G0 / T0 / A0 | T4-5 + 本方案 §4 | harness 扩展（VU 已实现；VE 已移植） |
| G1 | T4-1 | `h264_nvenc` `-cq` 等质量 |
| G2 | T4-2 | `hevc_nvenc` `-cq` 等质量 |
| G3 | **L40-1** | `av1_nvenc` `-cq` 等质量 |
| G4 | T4-3/4 + **L40-2** | constqp/QP 轴等质量（**VE 主**，VU 只复核 `_QP_SCALE`） |
| G5 | T4-7 门禁解锁 | 硬编条目去 SKIP |
| G6 | T4-8 / L40-5 | 真机长视频 |
| G7 | T4-9 / L40-3 | 落点/运行期回归复核 |

### 12.2 已实现的对称能力（本次落地）

- **VE harness 已能感知 VU**：`cross_repo_status()` 报告对侧仓库/两表是否相等/对侧 harness 是否同版/
  对侧方案文档（§4.5）。此前该能力**只有 VU 有**（单向），现已**双向对称**。
- 实测：VE `--quick` 已打印 `SIZE_MAP ✅ 两仓逐条相等 / QUALITY_MAP ✅ 两仓逐条相等`。

### 12.3 协同契约（CR，**上机前必须逐条裁定**）

| 编号 | 冲突/约束 | 现状 | 处置建议 |
|---|---|---|---|
| **CR-1** | **NVENC preset 口径**：VE `p4` vs VU `p5` | ✅ **已收口（两仓均 p4）** | VE 无需改。VU 已改：生产 `DEFAULT_PRESET_GPU` **p5→p4**（`vidcrop_cpu_v2.py` + `vidcrop_hwaccel.py` 孪生）+ harness `BASE_LOCK` p4 + 探针 `NVENC_PRESET='p4'` + README/基线同步。残留 `p5` 均为**有意保留**（见 §12.5） |
| **CR-2** | **NVENC rate control 口径**：CQ 轴在不同 rc（vbr / vbr_hq）下同 `-cq` 的等效点可能微异 | ✅ **已裁定路线 B，生产 + harness + VE 探针两仓/各层均已统一**：h264/hevc `vbr_hq`、av1 `vbr`，**均显式 `-rc`** | **VE**：生产不动；harness `-rc:v vbr_hq`/`-rc:v vbr`；**探针已修**（`av1_vp9_quality_matrix.py` 加 `_PROD_RC`）；判据 `crf_cq_unification_verify.py` 本就带 `-rc:v`。**VU 已同步（含 av1）**：`_NVENC_DEFAULT_RC={h264_nvenc,hevc_nvenc:"vbr_hq", av1_nvenc:"vbr"}`（两脚本孪生）+ harness `-rc vbr_hq`/`-rc vbr` + `t4_acceptance` A4。✅ **全链路对齐（无残留）**：VU 探针 `verify_nvenc_quality_gpu.py` 已加 `_cq_rc`（cq 分支 `-rc vbr_hq`/`vbr` + selftest 断言）⇒ 生产 / harness / 探针 / 判据两仓各层**均显式 `-rc`**。**路线 B 的理由**：VE h264/hevc 走 ctypes 直连 SDK，`nvenc_sdk` **不支持 `vbr`**（静默 CONSTQP + LA 失效）⇒ 改 VE 风险高 |
| **CR-3** | **harness 同版性**：VE 有 `--axis`（VU 无） | md5 不同（`harness_in_sync=False`） | 这是**有意差异**（VE 独有 QP 轴 D2b）。协同点：改共享块（SWEEP/LOCK/CQ 轴/探测/跨态势）时**两侧同步**；`--axis` 块可各自演进 |
| **CR-4** | **QP 轴归属**：D2b `QUALITY_MAP_QP` 仅 VE | VU 无此表 | VU **不需要**改动；但 VU 若改了 `_QP_SCALE`（AV1 ×3），**必须通知 VE** 同步 `_QP_MAP_OVERRIDE` 注释与 G3-7 期望来源 |
| **CR-5** | **素材池共用**：17 条切片在 VE `input_videos/eqq_calib/`（仓库外、不入 git） | — | T4/L40 机上需先把切片就位；VU 可复用同一池以保证两仓口径一致 |

### 12.4 本次 CPU 侧验证记录（无 GPU，仅纯逻辑可验证部分）

| 门 | 结果 |
|---|---|
| `calibrate_equal_quality.py --selftest` | **39 项全过**（含 NVENC/axis/跨仓纯函数） |
| CPU 干跑 `--quick --codecs libx265` | 端到端通过；**跨仓态势打印正确**（两表相等） |
| 无 GPU 探测 / `--expect-av1` | 均 **exit 2** 且报错明确（优雅降级） |
| `test_quality_map_qp_mode.py`（新） | **4/4 通过** |
| `crf_cq_unification_verify.py --no-gpu --quick` | **PASS=94 / FAIL=0 / SKIP=11**（2026-10-04 方案 A 落地后复跑；旧记录 `PASS=93` 为不同调用口径，勿混用） |
| `plan_implementation_gate.py`（静态 / 行为） | 静态 **50/48/0/2** + 行为 **46/46/0/0** = 合并 **96/94/0/2**（与基线一致） |
| `pytest Accessory/test` | 33 采集：**31 passed**（含新增 4）、2 failed —— 2 例为 `test_chroma_false_positive.py` 的**既有环境问题**（`git stash` 隔离后同样失败，非本次引入） |

> **仍需 GPU 才能验证**：所有实际 NVENC 探测/标定/落表、`--axis qp` 的真实编码结果、
> ⑨ 组在 NVENC 行加入后的相等性。见 §5（T4）/ L40 方案 §5。

### 12.5 CR-1 收口复检：VU 侧残留 `p5` 的甄别（2026-10-04 复测）

复检命令（全仓）：

```bash
cd /workspace/VidUtils && grep -rn '\bp5\b' --include='*.py' --include='*.md' --include='*.sh' . | grep -v __pycache__
```

**结论：VU 侧 NVENC 默认口径已全为 `p4`**（生产 `DEFAULT_PRESET_GPU="p4"` 两文件、
harness `BASE_LOCK`、探针 `NVENC_PRESET='p4'`、README/基线同步）。残留 `p5` **均为有意保留**：

| 位置 | 保留 p5 的原因 | 是否需改 |
|---|---|---|
| `NVENC_TO_X264_PRESET['p5']='medium'`（两脚本孪生） | **反向降级表**，兼容"用户显式 `--preset p5`"的既有行为（p4/p5 同效） | ❌ 保留 |
| `X264_TO_NVENC_PRESET['slow']='p5'` | **ffmpeg 官方枚举**（slow↔p5），非默认档 | ❌ 保留 |
| `_NVENC_PRESET_RETRY={'p5':'p4',...}`（hwaccel） | 用户**显式**给 p5/p6/p7 失败时的降级重试；注释已注明默认档是 p4 | ❌ 保留（`verify_borrow_enhancement.py:177` 有对应断言） |
| `X264_TO_SVTAV1_PRESET` 的 `'p5':8` | svtav1 映射，兼容显式 p5 | ❌ 保留 |
| `verify_quality_mapping.py` [11] 断言 QSV 默认=medium（非 p5） | 断言"不误下发 p5" | ❌ 保留（正是守卫） |
| `verify_color_tagging.py:53` / `verify_chroma_hook.py:104` 的 `preset='p5'` | 测试**显式**传 p5（走反向表→medium），非默认路径 | ⚠ 可留（覆盖显式 p5 场景） |
| README / memory / 历史 Plan 中解释性文字 | 记录"由 p5 改为 p4"的变更 | ❌ 保留 |

⇒ **CR-1 在 VE/VU 两侧均已收口**；VE 侧本就全 p4。

**CR-2 复检（2026-10-04，两轮）**：先确认 **VU 生产默认 rc = `--rc-mode auto` = 不发 `-rc`**；
**裁定「路线 B」：h264/hevc 统一 `vbr_hq`、av1 保持 `vbr`**。两仓各层落地情况：
- **VE**：生产**不动**；harness h264/hevc `-rc:v vbr_hq`、av1 `-rc:v vbr`；**探针已修**
  （`av1_vp9_quality_matrix.py` 加 `_PROD_RC`，命令形状已 CPU 验证：av1→`vbr`/h264→`vbr_hq`/软编无 `-rc`）；
  判据 `crf_cq_unification_verify.py` 的 NVENC 编码**本就带 `-rc:v`**（`:609-626`）⇒ VE 侧无缺口。
- **VU 已同步（含 av1）**：`_NVENC_DEFAULT_RC={h264_nvenc,hevc_nvenc:"vbr_hq", av1_nvenc:"vbr"}`
  （两脚本孪生）+ harness `-rc vbr_hq`/`-rc vbr`（selftest 过）+ `t4_acceptance` A4 `-rc vbr_hq`。
- **VU 探针已修**（2026-10-04）：`probe/verify_nvenc_quality_gpu.py::encode_nvenc(mode='cq')` 改走
  `_cq_rc(codec)`（av1→`vbr`、h264/hevc→`vbr_hq`）+ selftest 断言。
- ✅ **CR-2 收尾：生产 / harness / 探针 / 判据 两仓各层全部显式对齐，无残留。**
- **为何不选「统一到 vbr」**：VE 的 h264/hevc 生产走 **ctypes 直连 SDK Level 1**，而
  `nvenc_sdk` **不支持 `vbr`**（`else` 分支会静默落到 **CONSTQP**，且 SDK 的 LA 门控只认
  `vbr_hq/qvbr`）⇒ 改 VE 会动生产快路径且易静默出错；改 VU（CLI 层）风险低得多。

**CR-2 二次更新（2026-10-04，FFmpeg 9.0 实测后）**：FFmpeg 9.0.2 **CLI 移除 `vbr_hq` 与
`qvbr`**（`-rc` 只剩 constqp/vbr/cbr；传 vbr_hq 报 `Unable to parse "rc" option value`，rc=234）
⇒ 上面「harness `-rc:v vbr_hq`」口径在 FFmpeg 9.0 上不可运行，必须迁移：
- **VE CLI/harness/探针**：h264/hevc 改 **裸 `-rc:v vbr`**（2026-10-04 二次校正；
  `-tune`/`-multipass` 改为显式 opt-in，见 `Plan/ffmpeg_nvenc_knowledge.md` §5.2）
  （`calibrate_equal_quality.BASE_LOCK`、`av1_vp9_quality_matrix._PROD_RC`、
  `crf_cq_unification_verify` 的 `enc_nvenc`/G6 期望）；av1 保持 plain `vbr`。
  生产 writer `ffmpeg_io` 的 `_rc_v_map`/`_NVENC_RC_MAP` 同步（[FIX-FFMPEG9-VBRHQ]）。
- **VE SDK 侧不动**：`nvenc_sdk` 仍写 `rc_ptr[1]=32`——T4 实测驱动 13.0 仍接受且行为非静默
  钳制（vbr_hq/constqp/qvbr 输出互异）⇒ 生产快路径逐字节不变（见
  `Plan/T4_NVENC_vbr_hq移除_验证专项.md`）。
- **VU 必须重新同步**：VU 生产/harness/探针/`t4_acceptance` A4 的 `-rc:v vbr_hq` 同样会被
  FFmpeg 9.0 拒绝 ⇒ 需同步改**裸 `vbr`**，否则共享 `QUALITY_MAP` 的
  ⑨ 组跨仓一致性变红（handoff，本仓无法代改）。

---

## 13. 执行记录（T4，2026-10-04）

### 13.1 环境指纹

Tesla T4（UUID `GPU-1f4c2ae9-…`）/ 驱动 580.65.06 / CUDA 13.0 / torch 2.10.0+cu128 /
FFmpeg 9.0.2。Gate 0：`h264_nvenc` rc=0、`hevc_nvenc` rc=0、`av1_nvenc` rc=187（T4 无 AV1 NVENC，符合预期）。

### 13.2 任务完成矩阵

| ID | 任务 | 结果 |
|---|---|---|
| T4-1/2 | CQ 轴 h264/hevc 等质量 | ✅ 17 素材，已落 `QUALITY_MAP`（并入 `8f0a605`） |
| T4-3/4 | QP 轴 h264/hevc 等质量 | ✅ 17 素材，已落 `QUALITY_MAP_QP` |
| T4-5/6 | harness 扩展 / `_qp_model` 模式感知 | ✅ 既有（本次复核 selftest 39/39） |
| T4-7 | G7/G8 GPU 画质与码率门禁 | ✅ `crf_cq --gpu` PASS=101 / FAIL=0 / WARN=4 / SKIP=1 |
| T4-8 | h264/hevc 生产管线冒烟 | ✅ hevc+h264 双跑，帧守恒（见 13.4） |
| T4-9 | 跨仓 ⑨ 组 | ✅ 14/14 |

### 13.3 标定结果（17 素材，锚点 18/21/24/27/30，`n_subsample=1`，720p prep）

**CQ 轴**（`-rc:v vbr -cq:v`，落 `QUALITY_MAP`；标定时命令带 `-tune hq -multipass fullres`，
⚠ 2026-10-04 二次校正后默认改为裸命令，Δ≤0.11 VMAF 在 LOO 噪声内 ⇒ **不重标**）：

| 档位 | a | b | 样本 | LOO | 门禁 |
|---|---|---|---|---|---|
| `h264_nvenc` | 0.9295 | 6.2523 | 17 | 3.98 | ≤5.9 ✅ |
| `hevc_nvenc` | 1.1116 | 2.1606 | 17 | 5.81 | ≤5.9 ✅ |

**QP 轴**（`-rc:v constqp -qp`，落 `QUALITY_MAP_QP`，仅 VE）：

| 档位 | a | b | 样本 | LOO | 门禁 |
|---|---|---|---|---|---|
| `h264_nvenc` | 0.9704 | 1.4767 | 17 | 3.47 | ≤5.9 ✅ |
| `hevc_nvenc` | 1.1083 | -2.9183 | 17 | 3.72 | ≤5.9 ✅ |

> 两轴**非同一刻度**（QP 轴 a≠1、b≠0）⇒ 落表后 `to_constqp_qp` 的 **quality 口径**输出改变
> （h264 CQ26→QP22、hevc CQ28→QP23、h264 QP0→**0**〔无损短路，两口径一致〕）。
> ⚠ **2026-10-04 B1 迁移**：判据 `crf_cq_unification_verify` 的口径由 size **改为 quality**
> （`load_quality_map` 内 `set_quality_mode("quality")`），使**门禁口径 == 生产默认口径**；
> 相应更新 G1-2/G2/G3/G6 期望值（并新增 G6-18/19 锁"生产无损硬编码 `-qp 0`"、G3-9 反向锁 size 对照）。
> 迁移前是「判据钉 size ⇒ 期望值不变」；迁移后条数 104（CPU）/113（GPU）全绿。

### 13.4 门禁与验证（全绿）

| 门 | 结果 |
|---|---|
| harness `--selftest` | 39/39 ✅ |
| `eqq_pool_fit_table` 软编复现 | 6 档逐位复现库内表值 ✅ |
| `verify_equal_quality.py` | 5/5 达标（主门禁 ΔVMAF），exit 0 ✅ |
| `crf_cq_unification_verify --no-gpu --quick` | PASS=94 / FAIL=0 / SKIP=11 ✅ |
| `crf_cq_unification_verify --gpu`（真实素材） | PASS=101 / **FAIL=0** / WARN=4 / SKIP=1 ✅ |
| `plan_implementation_gate.py` | 96/94/0/2 ✅ |
| `pytest Accessory/test` | 29 passed / 2 failed（既有 chroma 环境项，与表无关） |
| `VidUtils verify_quality_mapping.py`（⑨） | 14/14 ✅ |
| 生产冒烟 hevc+LA=8 / h264+LA=8 | `segment_bitstream_verify_v5`：frames==packets=199、段首无连 IDR、frame_num 单调、无色度坏帧簇 ✅ |

> `crf_cq --gpu` 的 4 条 WARN（G7-2 hevc ΔPSNR −2.57dB / G7-3 / G7-5 / G7-8）为**已知内容相关诊断项**，
> 主判据（ΔVMAF）达标，按 P14 **不放宽判据**。

### 13.5 数据与代码落点

- 数据：`Accessory/data/eqq_calibration/points/gpu_t4_cq/`（3 文件 442 点）、
  `points/gpu_t4_qp/`（17 文件 459 点）。
- 代码：`src/utils/quality_map.py` 新增 `QUALITY_MAP_QP` 的 h264/hevc 两行。
- 顺带修复 `Accessory/probe/eqq_calibrate_batch.py` 两个 latent bug：
  ① `load()` 是生成器却 `len(items)`（无 `--only` 必崩）；② `--sweep` 为 `None` 时被迭代。
- `Accessory/test/test_quality_map_qp_mode.py` 更新为「标定后」语义（原第 2 例假设未标定；
  第 3 例的 `pop` 会误删真实标定行，改为保存/还原）。

### 13.7 L40 协同完成（2026-10-06）

L40 侧 `av1_nvenc` 专项已全部完成，与本方案共享的 harness/门禁/方法论复用验证通过：

| 项 | 结果 | 指标 |
|---|---|---|
| **L40-1** `av1_nvenc` CQ 等质量 | ✅ | `QUALITY_MAP['av1_nvenc']=(1.4566, 1.2165, 0, 63)` LOO 3.13 |
| **L40-2** `av1_nvenc` QP 等质量 | ✅ | `QUALITY_MAP_QP['av1_nvenc']=(7.9338, -97.5136, 0, 255)` LOO 2.61 |
| **L40-3** AC1 复验 | ✅ | `-qp 70` 落带内 1.07× / −1.13 dB |
| **L40-4** AC2 (G7-6) | ✅ | `-cq:v 32` PASS +0.16 dB / 1.28× |
| **L40-5** AV1 长视频冒烟 S1~S8 | ✅ | 15/16 PASS（constqp S3 计数差异，非功能性），S8 斜率通过 |
| **L40-8** 跨仓 ⑨ 组同步 | ✅ | 14/14 项一致 |

- 共享 harness（`calibrate_equal_quality.py` + `--axis qp`）、共享落表器（`eqq_pool_fit_table.py`）、共享门禁（`crf_cq_unification_verify.py` / `plan_implementation_gate`）均在 L40 上复用通过，无改动。
- **全 GPU 专项（T4 h264/hevc + L40 av1）收口，无阻塞项**。

### 13.6 两个易踩坑（复用本专项时务必遵守）

1. **`screen_ui_code_src1280x720.mp4` 在同一批里既是 6s 又是 10s 切片**（内容不同、同名）⇒
   直接按文件名池化会把两者并成 1 个素材（17→16），给出**另一组** a/b
   （h264 `0.9256/6.3447` 而非 `0.9295/6.2523`）。**必须去重为 17 个不同素材名**。
2. **素材均以 symlink 按「素材名」接入**（`/tmp/eqq_gpu_src*`），切片真实路径在仓库外
   `/workspace/input_videos/eqq_calib/`；`--src-is-prep` + 每素材独立 workdir 是复现前提。
