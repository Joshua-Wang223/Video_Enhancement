# eqq_calibration —— 等质量（CRF→CQ）标定数据归档

对应素材库：`input_videos/eqq_calib/`（见其 `MANIFEST.md`）。
落表产物：`src/utils/quality_map.py` 的 `QUALITY_MAP` + `_EQQUAL_SPEED_OVERRIDE`。

📖 **总览与复用指南**：`Accessory/docs/EQQ_CALIBRATION_OVERVIEW.md`
（含素材表、5 个脚本的用法、7 条踩坑教训）。本文件只讲数据本身。

## 权威数据源（18 个文件 / 1400 原始点 / ACC 1153）

```
文件数 = 18
逐文件点数求和 = 1400
ACC 总点数 = 1153      ← 合并重复观测后的唯一 (素材,档位,参数) 数
合并重复点 = 195
素材池 = 17 条
```

⚠ **这四个数必须在每次池化前打印并与预期比对**（见下方教训③）。

| 目录 | 文件 | 点数 | 说明 |
|---|---|---|---|
| `points/6s/` | `vu_m2_7src_x264_soft.json` | 203 | VU 7 素材：x264 锚点 + 4 软编 + rav1e native |
| | `vu_m2_7src_rav1e_native.json` | 105 | 同 7 素材，rav1e native QP 扫描 |
| | `vu_m2_7src_rav1e_speed10.json` | 105 | 同 7 素材，rav1e `-speed 10` QP 扫描 |
| | `gen_live_kids_play.json` | 65 | 幼儿园彩色积木 |
| | `gen_cganim_edu_wordworld.json` | 65 | WordWorld 三维动画 |
| | `gen_tv_bbc_molly_s0{1,3,5}e01.json` | 65×3 | BBC 实拍剧集 3集 |
| `points/10s/` | `anim2d_forest.json` | 65 | 二维动画森林 |
| | `anim2d_subs_tobot.json` | 65 | 二维动画带字幕 |
| | `live_night_wolf.json` | 65 | 夜间灰狼（暗场） |
| | `live_texture_frog.json` | 65 | 毛绒玩具（高细节纹理） |
| | `screen_ui_code.json` | 65 | 代码编辑器 UI |
| `points/legacy10s/` | `eqq2_10s_n4.json` | 220 |旧 10s 口径 4 素材 × 55 点 |
| | `eqq2_10s_n2.json` | 75 | 旧 10s 口径 5 素材 × 15 点 |
| | `eqq2_10s_anchorB.json` | 12 |锚点集探测（B 套 18/21/24/27/30） |
| | `eqq2_10s_bbc_anchorB.json` | 9 | 同上，仅 BBC 3 条 |
| | `vu_m2_anchorA.json` | 21 | 锚点集探测（A 套） |

### GPU 标定数据（T4，硬编 h264/hevc，2026-10-04）

与软编库**分开存放**（不并入上面的 18 文件/1400 点口径计数），`--sides` 用时显式加上：

| 目录 | 文件数 | 点数 | 说明 |
|---|---|---|---|
| `points/gpu_t4_cq/` | 3 | 442 | CQ 轴（`-cq:v`，VBR）标定：17 素材 × {h264,hevc} + libx264 锚点。`--sides …,gpu_t4_cq --axis cq` |
| `points/gpu_t4_qp/` | 17 | 459 | QP 轴（`-rc:v constqp -qp`）标定：17 素材 × {h264,hevc} + 锚点。`--sides …,gpu_t4_qp --axis qp` |

- 素材 = `eqq_calib` 的 17 条切片（12×6s + 5×10s），锚点 18/21/24/27/30，`n_subsample=1`，720p prep，`--src-is-prep`。
- ⚠ **`screen_ui_code_src1280x720.mp4` 6s/10s 同名但内容不同** ⇒ `gpu_t4_*` 内 10s 侧已改名
  `screen_ui_code_src1280x720_10s.mp4`；否则池化会并成 16 素材、给出另一组 a/b。
- 落表值：CQ → `QUALITY_MAP['h264_nvenc']=(0.9295,6.2523)` LOO 3.98 / `['hevc_nvenc']=(1.1116,2.1606)` LOO 5.81；
  QP → `QUALITY_MAP_QP['h264_nvenc']=(0.9704,1.4767)` LOO 3.47 / `['hevc_nvenc']=(1.1083,-2.9183)` LOO 3.72。

### ⚠ `legacy10s/` 是有效数据，不可剔除

这 5 个文件（337 点）是 **12 条素材的 10s 侧观测**。历史上曾被误判为「跨时长脏数据」
而舍弃 —— 见下方教训②与 git `eda957d`。

素材名相同**掩盖了数据缺口**：脚本按素材名去重后以为「6s 侧已全覆盖」，
实际漏读了 10s 侧观测，舍弃它们换来的更低 LOO 是**样本覆盖变窄导致的虚假改善**。

12 条跨口径同名素材：`new1.mp4`、`new4_raw.mp4`、`new5_raw.mp4`、`word_world_2.mp4`、
BBC×3、`cc_anim_300s.mkv`、`cc_subs_105s.mp4`、`earth_dark_80s.mp4`、
`natgeo_grass_40s.mp4`、`ui_screen_10s.mp4`。
合并规则：**同 key 取均值**（该规则对文件顺序对称 ⇒ 顺序无关性成立，已实测）。

命名对照：文件用**切片语义名**，`points/*.json` 内的 key 仍是**原始文件名**。
对照关系见 `points/clip_name_mapping.json`。

## 复现表值

```bash
python3 Accessory/probe/eqq_pool_fit_table.py
```

已验证**逐位复现**库内 `QUALITY_MAP`（6 档 a/b/LOO 全等，rc=0）：

| 档位 | a | b | 样本 | LOO | 门禁 |
|---|---|---|---|---|---|
| `libx265` | 1.0979 | −2.3119 | 17 | 4.24 | ≤5.9 ✅ |
| `libvpx-vp9` | 1.9716 | −15.0929 | 17 | 3.82 | ≤5.9 ✅ |
| `libsvtav1` | 2.3961 | −21.3615 | 17 | 4.88 | ≤5.9 ✅ |
| `libaom-av1` | 2.3219 | −22.3927 | 17 | 5.35 | ≤5.9 ✅ |
| `librav1e` | 7.9326 | −102.2078 | 17 | 5.76 | ≤7.5 ✅ |
| `librav1e@10` | 7.9173 | −106.6317 | 17 | 5.99 | ≤7.5 ✅（→ `_EQQUAL_SPEED_OVERRIDE`） |

`QUALITY_MAP_QP` 的软编 4 行同步镜像。

## 三条数据完整性教训（复用本库时务必遵守）

1. **素材名去重 ≠ 数据完整**。曾按素材名判定「6s 侧已覆盖」，漏读 10s 侧 316 点并据此落表。
2. **LOO 明显变好时先查数据完整性**，不要当成精度提升 —— 那个「改善」来自样本变窄。
3. **池化前打印四个数**：文件数 / 逐文件点数 / ACC 总点数 / 合并重复点数，与预期比对。
   `eqq_pool_fit_table.py` 已把这四个数做成**强制首段输出**。

（另有一条已不适用：「同 key 取均值只适合真噪声」在本库不成立 ——
`legacy10s` 的重复观测来自**系统性时长差异**，但它是**有效观测、必须保留**，
与「脏数据应舍弃」是两件事。真正的风险是「假装它不存在」。）

## 已作废数据（`superseded/`，**不要用于落表**）

| 文件 | 内容 | 作废原因 |
|---|---|---|
| `n3_subsample8_INVALID.json` | 204 点 / 3 素材 | **`subsample=8`**，实测偏置 VMAF 1.9~3.0。现行数据全部 `subsample=1` |
| `n1_3s_quick_smoke.json` | 9 点 | 3s 口径冒烟，仅验证 harness 可跑通 |

## 目录

```
eqq_calibration/
├── points/
│   ├── 6s/           8 个points（738 点）
│   ├── 10s/          5 个 points（325 点）
│   ├── legacy10s/    5 个 points（337 点，有效观测）
│   ├── gpu_t4_cq/    3 个 points（442 点，硬编 CQ 轴）
│   ├── gpu_t4_qp/    17 个 points（459 点，硬编 QP 轴）
│   └── clip_name_mapping.json
├── superseded/仅 2 个真正作废的数据集
├── reports/          门禁与验证日志、最终看板
└── logs/             逐素材采集日志、看护进度日志
```