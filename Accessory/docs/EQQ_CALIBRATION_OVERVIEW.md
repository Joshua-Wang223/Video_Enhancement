# 等质量标定（CRF→CQ）总览 —— 素材 / 脚本 / 数据 / 复现指南

一篇文档回答：这套东西**含什么**、**为什么这么切**、**怎么用**、**怎么验证没坏**。

标定结论落在 `src/utils/quality_map.py` 的 `QUALITY_MAP`（6 档编码器的
CRF→CQ 线性换算表）。本文档只讲**怎么产出来、怎么复用**。

---

## 1. 这是干什么的

不同编码器在**同一 CRF 下画质不同**，跨编码器复用 CRF 参数会偏。
等质量标定就是为每个编码器单独拟合一条 `crf → cq` 直线，
使同一条素材在 x264 / x265 / vp9 / aom / svtav1 / rav1e 上**落在同一 VMAF 水平**。

**换锚点就得重标**。锚点是 `libx264` 的 5 个 CRF（`ANCHOR_CRFS = [18,21,24,27,30]`，
两仓已统一），x264 的 VMAF 是所有档位的公共基准，改它等于换坐标系。

落表值（第八版，当前在库）：

| 档位 | a | b | LOO[0,27] | 监控[全] | 门禁 |
|---|---|---|---|---|---|
| `libx265` | 1.0979 | −2.3119 | 3.04 | 4.24 | ≤5.9 ✅ |
| `libvpx-vp9` | 1.9716 | −15.0929 | 2.83 | 3.82 | ≤5.9 ✅ |
| `libsvtav1` | 2.3961 | −21.3615 | 3.59 | 4.88 | ≤5.9 ✅ |
| `libaom-av1` | 2.3219 | −22.3927 | 4.19 | 5.35 | ≤5.9 ✅ |
| `librav1e`（native） | 7.9326 | −102.2078 | 4.47 | 5.76 | ≤7.5 ✅ |
| `librav1e@10`（speed10） | 7.9173 | −106.6317 | 4.52 | 5.99 | ≤7.5 ✅ |

LOO = 留一素材交叉验证的 worst ΔVMAF。门限分档是因为 rav1e 是 0~255 QP 刻度、
软编是 0~63，残差天然不可比。

⚠ **判据锚点口径 = 生产工作区间 [0,27]**（2026-10-04 起，仓主确认生产 `crf_ref` 不用 >27）。
全锚点（含 crf30）的 worst 只作为**监控列**打印、**不计 FAIL** —— 依据：全 8 档实测显示
现状 worst **对每个编码器都来自最高锚点 crf30**（8/8 满足 `≤24 < ≤27 < 全区间`），
crf30 处各编码器曲线进入陡降段把小的参数偏差放大成 5~6 dB，属**门禁给边界锚点同等权重**
的系统偏差，而非换算缺陷。监控列保证高 ref 段不被隐藏、只是不阻断。详见
`Accessory/probe/eqq_pool_fit_table.py` 的 `GATE_ANCHORS` 注释。

> 等质量 **NVENC** 行另见 `QUALITY_MAP`（h264/hevc = T4；av1 = L40）与
> `QUALITY_MAP_QP`（QP 轴，含 av1 仿射行 `(7.9338, −97.5136)`，见
> `Accessory/data/eqq_calibration/reports/av1_nvenc_L40_calibration_report_20261004.md`）。

⚠ rav1e 两档的**代码落点不同**：`QUALITY_MAP['librav1e']` 是 native，
speed10 的数值在 `_EQQUAL_SPEED_OVERRIDE`（键仍是 `'librav1e'`，按编码器名索引，
**仅当 `RAV1E_SPEED > 0` 时生效**）。`QUALITY_MAP_QP` 不登记 rav1e，避免与档位语义冲突。

---

## 2. 素材：`input_videos/eqq_calib/`（17 条 / 114MB）

⚠ 该目录**在仓库外**（`/mnt/d/Workspace_Python/input_videos/`），不入 git。

### 存的是切片，不是原片

标定的参考片不是原片，而是「**720p /固定时长 / lanczos / crf10**」切片
（harness 的 `make_prep()`）。决定 VMAF 曲线的是切片，不是原片 ——
所以库里存切片，原片只作为溯源信息记在清单里。

BBC 3 条原片各~536MB / ~850s，存于网络盘 `/mnt/f`，不入库。

### 分类与命名

两级目录：`<口径>/`，其中 `6s/` 12 条、`10s/` 5 条。
文件名 `<类别>_<题材>_src<原片宽>x<高>.mp4`，其中 `src1920x1080` 指的是**原片**分辨率
（文件本身一律 720p）。

| 口径 | 切片 | 类别 | 题材 | 帧数 |
|---|---|---|---|---|
| 6s | `live_kids_play_src1280x720.mp4` | 实拍 | 幼儿园彩色积木游戏 | 179 |
| 6s | `live_kids_seated_src1920x1080.mp4` | 实拍 | 幼儿园室内坐姿 | 180 |
| 6s | `live_kids_table_src1920x1080.mp4` | 实拍 | 幼儿园围桌 | 180 |
| 6s | `cganim_edu_wordworld_src720x576.mp4` | 三维动画 | 教育情景 WordWorld | 150 |
| 6s | `cganim_talking_tom_src1920x1080.mp4` | 三维动画 | 会说话汤姆猫 | 144 |
| 6s | `cganim_subs_src1920x1080.mp4` | 三维动画 | 三维动画带字幕 | 144 |
| 6s | `tv_bbc_molly_s01e01_src1920x1080.mp4` | 实拍剧集 | BBC《Molly and Mack》S01E01 | 180 |
| 6s | `tv_bbc_molly_s03e01_src1920x1080.mp4` | 实拍剧集 | 同上 S03E01 | 180 |
| 6s | `tv_bbc_molly_s05e01_src1920x1080.mp4` | 实拍剧集 | 同上 S05E01 | 180 |
| 6s | `doc_dark_earth_src3840x2160.mp4` | 实拍纪录片 | 暗场·地球夜景 | 284 |
| 6s | `doc_grassland_src1280x720.mp4` | 实拍纪录片 | 高细节草原风光 | 148 |
| 6s | `screen_ui_code_src1280x720.mp4` | 屏幕录制 | 代码编辑器 UI | 180 |
| 10s | `anim2d_forest_src1280x720.mp4` | 二维动画 | 森林植被 | 239 |
| 10s | `anim2d_subs_tobot_src1280x720.mp4` | 二维动画 | 二维动画带字幕 | 239 |
| 10s | `live_night_wolf_src1280x720.mp4` | 实拍 | 暗场夜间灰狼 | 479 |
| 10s | `live_texture_frog_src1280x720.mp4` | 实拍 | 高细节毛绒玩具纹理 | 300 |
| 10s | `screen_ui_code_src1280x720.mp4` | 屏幕录制 | 代码编辑器 UI | 300 |

**为什么必须有 6 类内容**：只用单一类别会让该类内容的失真特征主导全表。
库里有实测 —— 只用 10s 侧素材 ⇒ svtav1 LOO 7.33 超门禁；只用 6s 侧 ⇒ vp9 LOO 6.85 超门禁。
暗场 / 高细节 / 带字幕是三条**属性**变体，用来打破内容类别的系统性偏置。

⚠ **内容类别是目视关键帧逐条确认的**，不靠文件名猜 —— `word_world_2`
看似实拍实测是 3D 动画，`new1/4/5` 是幼儿园实拍。

### 清单文件

| 文件 | 内容 |
|---|---|
| `MANIFEST.md` | 人类可读：素材表 + 切片口径 + 复现命令 + 5 条注意事项 |
| `manifest_6s.json` / `manifest_10s.json` | 机读：原片绝对路径、原始分辨率/fps/时长/transfer、切片帧数/时长/md5、是否 tonemap |

---

## 3. 脚本：`Accessory/probe/eqq_*.py`（5 个）

工作流是三步走（切片 → 测量 → 落表），另加单素材调试与批量看护。

```
原片 ──eqq_slice_prep.py──▶ 720p 切片 + manifest
                                        │
切片 ──eqq_calibrate_batch.py──────────▶ points.json × N
      └─eqq_calibrate_clip.py（单条）        │
                          └─eqq_watch_batch.py（看护/ ETA）
                                   │
points ──eqq_pool_fit_table.py──────▶ QUALITY_MAP 候选 + 门禁
```

### `eqq_slice_prep.py` —— 原片 → 切片 + manifest

```bash
python3 Accessory/probe/eqq_slice_prep.py \
    --spec clips.json --outdir input_videos/eqq_calib
```

`--spec` 是素材规格 JSON 数组，每条 `{src, name, side, category, topic, origin}`。
复用 harness 的 `make_prep()`，保证切片口径与已落表数据**逐字一致**。
**切片实际时长短于口径即 `exit`**（ffmpeg 只会静默截断，见 §6.3）。
其它选项：`--dry-run`（只报告）、`--duration`（覆盖口径）、`--workroot`。

### `eqq_calibrate_clip.py` —— 单素材测量器

```bash
# 首次测量
python3 Accessory/probe/eqq_calibrate_clip.py \
    --src input_videos/eqq_calib/6s/live_kids_play_src1280x720.mp4 \
    --out /tmp/eqq_run/live_kids_play --duration 6

# 复核库内数据（切片已是参考片 ⇒ 必须加 --src-is-prep）
python3 Accessory/probe/eqq_calibrate_clip.py \
    --src input_videos/eqq_calib/6s/live_kids_play_src1280x720.mp4 \
    --out /tmp/eqq_recheck --duration 6 --src-is-prep
```

对一条素材扫完所有档位，产出 `points.json`。**断点续跑**：已有 key 直接跳过。
调试可减档位/ 减扫描点：`--tiers libx265,libvpx-vp9` `--sweep libx265=30`。

⚠ **每素材必须独立 `--out`**。harness 的 `make_prep()` 固定写 `work/prep.mp4`
（先 unlink 再建）、`_save_points()` 是无锁 `write_text()` 覆盖
⇒ 同目录并行会**互删 prep / 丢点**。

### `eqq_calibrate_batch.py` —— 批量并行

```bash
python3 Accessory/probe/eqq_calibrate_batch.py \
    --manifest input_videos/eqq_calib/manifest_6s.json \
    --outroot /tmp/eqq_run --jobs 4 --src-is-prep
```

替代了原先硬编码素材清单的 `run_gen.sh` / `run_ji.sh`（换批次要改脚本）。
新增素材只需更新 manifest。每素材自动建独立 workdir，并把 `src.txt` /
`duration.txt` 写进去供看护脚本自动重启。其它选项：`--only tag1,tag2`（单跑几条）、
`--tiers`、`--sweep`、`--keep-prep`。

### `eqq_pool_fit_table.py` —— 落表器（含门禁断言）

```bash
python3 Accessory/probe/eqq_pool_fit_table.py
python3 Accessory/probe/eqq_pool_fit_table.py --out /tmp/table.txt   # 写出候选行
```

读 `Accessory/data/eqq_calibration/points/`，跑两项硬断言，任一不过 `exit 1`：

1. **顺序无关性回归** —— 3 个随机种子打乱文件顺序复算，6 档表值须逐位一致（1e-12）。
   合并规则「同 key 取均值」对文件顺序对称，所以顺序无关是**实测结论**。
2. **LOO 门禁** —— 软编 ≤5.9 / rav1e ≤7.5，**不自行放水**。
   判据锚点 = **生产工作区间 [0,27]**（全锚点 worst 仅作监控打印、不计 FAIL；口径依据见上文表下注）。
   ⚠ 0 评估点记`inf`（模型失效）而非 PASS —— 曾出现「预测越界⇒回查 None⇒被当 PASS」。

另有**只告警不失败**的跨口径同名素材统计（原因见 §6.1）。

### `eqq_watch_batch.py` —— 批量看护

```bash
python3 Accessory/probe/eqq_watch_batch.py --outroot /tmp/eqq_run --once
python3 Accessory/probe/eqq_watch_batch.py --outroot /tmp/eqq_run \
    --interval 600 --restart-on-abnormal --log /tmp/watch.log
```

进度快照 + **按 tier 加权** 的 ETA + 异常诊断 + 按 resume 语义重启。
⚠ 重启前会先检查同 workdir 是否已有进程在跑。满档判据 = 65 点/素材
（`--expect`可改）。

⚠ **ETA 必须按 tier 分档**：rav1e 单点 ~184s（native）/ ~54s（@10），软编仅 ~35s
⇒ 剩余点数少不代表剩余时间少。按整体均值估会严重低估。

---

## 4. 数据：`Accessory/data/eqq_calibration/`

```
eqq_calibration/
├── points/
│   ├── 6s/           8 个 points（738 点）
│   ├── 10s/          5 个 points（325 点）
│   ├── legacy10s/    5 个 points（337 点）★ 有效数据，不可剔除
│   └── clip_name_mapping.json
├── superseded/       仅 2 个真正作废的数据集
├── reports/          门禁与验证日志、最终看板
└── logs/             逐素材采集日志、看护进度日志
```

**18 个 points 文件 / 1400 原始点 / ACC 1153 / 素材池 17 条。**

### 落表复算

```bash
python3 Accessory/probe/eqq_pool_fit_table.py
```

已验证**逐位复现**库内 `QUALITY_MAP`（6 档 a/b/LOO 全等，rc=0）。

### `points/*.json` 的 key 用的是**原始文件名**

`new1.mp4`、`S01E01._The_New_Stall.mp4`… 与切片语义名不一致，
对照关系见 `points/clip_name_mapping.json`（17 条全覆盖）。

### `superseded/` 只剩 2 个

| 文件 | 作废原因 |
|---|---|
| `n3_subsample8_INVALID.json` | **`subsample=8`**，实测偏置 VMAF 1.9~3.0。现行数据全部 `subsample=1` |
| `n1_3s_quick_smoke.json` | 3s 口径冒烟，仅验证 harness 可跑通 |

---

## 5. 一次性诊断：`Accessory/archive/eqq_diag/`（45 个）

标定过程中的排查脚本（锚点取证、LOO 诊断、表值落表演进、cron 辅助…）。
**不是复用工具**，保留只为让当时结论可追溯。该目录有 `README.md`
说明分组与每个旧脚本的现代替代物。

---

## 6. 踩过的坑（复用前必读）

### 6.1 素材名去重 ≠ 数据完整（代价最大的一次）

曾按素材名判定「6s 侧已覆盖」，漏读 10s 侧 **316 点**并据此落表
（x265 1.0908 / LOO 3.37）。那批数据后来定性为**有效观测**，
舍弃它们换来的更低 LOO 是**样本覆盖变窄导致的虚假改善**，不是精度提升。

⇒ **判据**：池化前打印「文件数 / 逐文件点数 / ACC 总点数 / 合并重复点数」四个数
并与预期比对。`eqq_pool_fit_table.py` 已把这四个数做成**强制首段输出**。
⇒ **LOO 明显变好时先查数据完整性**，别当成精度提升。

### 6.2 切片需要 `--src-is-prep`

切片**已经是**参考片，再 `make_prep()` 会多一轮 crf10 重编码，
锚点 VMAF 偏移 **0.02~0.29**。
实测：加 `--src-is-prep` ⇒ 与库内 `points.json` **|Δ| = 0.0000（6/6）**；不加 ⇒ 偏移 0.02~0.29。

### 6.3 素材短于口径时 ffmpeg 静默截断且不报错

会污染整个口径且**无任何告警**。历史上5 条 10s 素材实测只有 9.958~10.010s。
⇒ 新增素材先 `ffprobe` 核实实际时长 ≥ 口径，不足则**重新采集**（不能靠放宽阈值）。
`eqq_slice_prep.py` / `eqq_calibrate_clip.py` 都会在这种情况下报错退出。

### 6.4 VMAF 必须 `subsample=1`

`subsample > 1` 偏置 VMAF **1.9~3.0**。库内已把 `subsample=8` 的数据标作废。

### 6.5 同 workdir 并行会丢点

`make_prep()` 先 unlink 再建 + `_save_points()` 无锁覆盖。

### 6.6 rav1e 的 `-speed` 要单独成档

`speed10` 会整体平移码率曲线 ⇒ native 与 `-speed10` 必须各出一行。
标定数据里键为 `librav1e` / `librav1e@10`，但**落表时都以编码器名为键**：
native 进 `QUALITY_MAP`，speed10 进 `_EQQUAL_SPEED_OVERRIDE`（见 §1）。

### 6.7 落表值别自己改

`QUALITY_MAP` 是 `quality_map.py` 的唯一真源，改动前先跑一次
`eqq_pool_fit_table.py` 拿到基线，改完再跑比对。

---

## 7. 相关文件速查

| 用途 | 路径 |
|---|---|
| 表值真源 | `src/utils/quality_map.py`（`QUALITY_MAP` / `_EQQUAL_SPEED_OVERRIDE`） |
| CRF→CQ 换算 | `src/utils/convert_crf.py`（B 仓同名文件需与之逐条相等） |
| 核心 harness | `Accessory/probe/calibrate_equal_quality.py`（所有工具都调它） |
| 独立 LOO / 缓存转换 | `Accessory/probe/loo_equal_quality.py`、`convert_points_cache.py` |
| 等质量门禁 | `Accessory/verify/verify_equal_quality.py` |
| 换算表校验（VidUtils 侧） | `VidUtils/verify/verify_quality_mapping.py` |
| 素材清单 | `../input_videos/eqq_calib/MANIFEST.md`（**仓库外**，不入 git） |
| 数据清单 | `Accessory/data/eqq_calibration/MANIFEST.md` |
| 标定过程记录 | `memory/equal-quality-anchor-unification.md` |
| 资产归整记录 | `memory/equal-quality-asset-consolidation.md` |

## 8. 环境前提

- `ffmpeg` / `ffprobe` 在 PATH；各编码器可用（`libsvtav1` / `libaom-av1` 等
  **本 build 可能没有** ⇒ 先 `-encoders` 探）
- **不需要 GPU**，纯 CPU 度量
- rav1e 单点 ~184s（native）是全流程最慢项，估 ETA 别按同族最慢档一刀切
