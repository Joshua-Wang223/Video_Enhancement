---
name: 等质量标定资产归整（素材/数据/脚本三处落点）
description: 2026-10-03 收尾：17 条素材切片化入 input_videos/eqq_calib/、18 文件 1400 点归入 Accessory/data/eqq_calibration/、5 个 eqq_* 泛化工具入 probe/ + 54 个一次性脚本入 archive/eqq_diag/；含两条实证（切片可逐位复现、legacy10s 不可剔除）
type: project
---

标定已完成（表值见 [[equal-quality-anchor-unification]]），本条记录**资产归整的落点**，
便于后续同类任务直接复用，不需重新考古 `/tmp`。

## 三处落点

| 内容 | 位置 | 规模 |
|---|---|---|
| 素材切片 + 清单 | `input_videos/eqq_calib/{6s,10s}/` | 17 条 / 114MB + `manifest_*.json` + `MANIFEST.md` |
| 标定数据 | `Accessory/data/eqq_calibration/` | 18 points 文件 / 1400 原始点 + reports + logs + superseded（仅 2 个） |
| 可复用工具 | `Accessory/probe/eqq_*.py` | 5 个（slice_prep / calibrate_clip / calibrate_batch / pool_fit_table / watch_batch） |
| 一次性诊断 | `Accessory/archive/eqq_diag/` | 54 个 + 分组 README |

⚠ `input_videos` 在**仓库外的上层目录**（`/mnt/d/Workspace_Python/input_videos`），
不在 git 跟踪内 —— 切片丢了只能从原片重生成，BBC 3 条原片在 `/mnt/f` 网络盘。

## 素材命名规则

`<类别>_<题材>_src<原片宽>x<高>.mp4`，分`6s/`（12 条）与 `10s/`（5 条）两个口径目录。
内容类别：实拍 / 实拍剧集（BBC）/ 三维动画 / 二维动画 / 实拍纪录片 / 屏幕录制。
**`src1920x1080` 指原片分辨率**（文件本身都720p）。全部内容类别经**目视关键帧确认**，
不靠文件名猜（`word_world_2` 实为 3D 动画而非预期类别、`new1/4/5` 是幼儿园实拍）。

## 两条实证（都做过实测，别再踩）

**① 切片 + `--src-is-prep` 可逐位复现库内数据（|Δ| = 0.0000，6/6 点）**
用 `eqq_slice_prep.py` 从原片重生成切片，再跑 `eqq_calibrate_clip.py --src-is-prep`
（切片**已是参考片**，跳过 `make_prep`），实测锚点 VMAF 与库内 `points.json` 逐位相同。
⚠ 不加 `--src-is-prep` 会**多一轮 crf10 重编码**，锚点 VMAF 偏移 **0.02~0.29**，
与库内数据不可比 —— 这两个数就是发现该缺陷的依据。

**② `legacy10s/` 是有效数据，不可剔除**
第四个数据目录（5 文件 337 点：4 个 `eqq2_10s_*` + `vu_m2_anchorA`），是 12 条素材的
10s 侧观测。**我一度把它归档进 `superseded/` 并标为作废**—— 错。
`git eda957d` 已定性：「不是该舍弃的脏数据，而是有效观测；舍弃它们换来的更低 LOO是
**样本覆盖变窄导致的虚假改善**」。
⇒ `eqq_pool_fit_table.py` 里「跨口径同名素材必须为 0」这条断言已从**失败条件降为告警**，
真正的保证来自 3 seed 顺序无关性实测。
⚠ **该脚本用 `parent.name` 判定口径目录名**，故目录名必须是 `6s` / `10s` / `legacy10s`。

## 落表器已验证逐位复现

`eqq_pool_fit_table.py` 对库内 18 文件跑出 6 档 (a, b, LOO) **全部逐位等于**
`src/utils/quality_map.py` 的 `QUALITY_MAP`，rc=0。首段强制打印四个完整性计数
（文件数 18 / 逐文件求和 1400 / ACC 1153 / 合并重复 195），落实[[equal-quality-anchor-unification]] 教训③。

## ⚠ `/tmp` 与 `VidUtils/temp/` 下的原片是临时的

`VidUtils/temp/m2_srcs/*`、`/tmp/eqq_uni_10s/src/*`、`/tmp/eqq_native_srcs/*`（软链到 `/mnt/f`）
重启即丢。长期保存的只有 `input_videos/eqq_calib/` 下的切片。
用户 2026-10-03 明确要求：`/tmp` 遗留产物与日志**原地保留**，不清理。

**How to apply**：下次做同类标定，直接 `eqq_slice_prep.py` → `eqq_calibrate_batch.py`
→ `eqq_pool_fit_table.py` 三步走；素材不够就扩`input_videos/eqq_calib/` 并更新 manifest，
不要另起目录。池化前先跑落表器看四个计数对不对。