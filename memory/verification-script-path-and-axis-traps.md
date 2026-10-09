---
name: 验证脚本三处路径/口径陷阱（素材目录、落表器轴后缀、L40 范围收敛）
description: 2026-10-09 修 Plan/PROMPT_L40_全流程验证.md 时核出的三类错误——① 素材池在仓库外 `<项目父目录>/input_videos`，写 `input_videos/xxx` 必然找不到（含 comprehensive_verify.py 硬编码 Windows 路径 + nvenc_tuning_verify 内联硬编码 src 路径两处代码缺陷）；② eqq_pool_fit_table.py --sides 的 GPU 目录带轴后缀 gpu_{t4,l40}_{cq,qp}，旧名 `gpu_t4`/`gpu_l40` 匹配不到目录却 exit 0（静默假通过）；③ L40 验收只占 4 项，其余项已在 CPU/T4 侧完成；并记录 nvenc_rc_diagnose 实际是纯主机侧工具（required_gpu 属过度门控）、⑨ 组脚本已迁到 Accessory/verify/ 且当前 13/14（--threads 差异）
type: feedback
---

# 三类陷阱的成因与判据

## ① 素材路径：素材池在仓库外

**现象**：文档/脚本里写 `input_videos/xxx.mp4`，从仓库根跑必然 `No such file`；而素材实际存在。

**根因**：素材池**在仓库外**，路径是 `<项目父目录>/input_videos`（生产 Linux 与 WSL 同为 `/workspace/input_videos`）。仓库内没有 `input_videos/` 目录，`cwd` 又是仓库根 ⇒ 相对路径解析到不存在的位置。

**How to apply**：命令行一律写 `../input_videos/xxx`。脚本内用候选列表探测，不要单点硬编码。

已修 `Accessory/verify/comprehensive_verify.py`：
- `INPUT_VIDEOS_CANDIDATES = (PROJECT_ROOT.parent / "input_videos", Path("/mnt/d/Workspace_Python/input_videos"))`，`_input_videos_base()` 取首个存在者，`_default_source()` 逐候选探测。
- 顺带修了 `nvenc_tuning_verify` 子测试内联的 `sys.path.insert(0, '/mnt/d/Workspace_Python/Video_Enhancement/src/utils')` —— 该子测试在 Linux 上**必然 FAIL**（只是被 `segment_bitstream_verify` 因未给 `-o` 先计入 FAIL 而掩盖了）。已改为 `{str(PROJECT_ROOT / 'src' / 'utils')!r}`。

**教训**：仓内文档间引用路径不一致（`/workspace/input_videos/` vs `/mnt/d/...` vs `input_videos/`）本身就是信号，说明存在平台硬编码。

## ② 落表器 `--sides` 的 GPU 目录带轴后缀

**现象**：`--sides 6s,10s,legacy10s,gpu_l40` 跑完打印三个 NVENC 档「拟合失败（样本 0 < 4）」，但**退出码 0**，汇总还打「✅ 全部档位 LOO 达标」。

**根因**：`Accessory/data/eqq_calibration/points/` 下的真实目录名带轴后缀：`gpu_t4_cq` / `gpu_t4_qp` / `gpu_l40_cq` / `gpu_l40_qp`（另有 `gpu_l40_cq_dense`）。`find_points()` 在目录不存在时返回 `[]` 而不报错 ⇒ 不带后缀的 `gpu_l40` 静默贡献 0 文件。

**Why 会踩**：CQ 与 QP 是两条独立标定轴（`QUALITY_MAP` / `QUALITY_MAP_QP` 两张表），点数据必须分目录；改名成带后缀时**没有同步改全部调用方**——`Plan/` 下 4 份方案文档 + `comprehensive_verify.py` 的 `build_test_matrix()` 都还是旧名。

**How to apply**：
```bash
python3 Accessory/probe/eqq_pool_fit_table.py --sides gpu_l40_cq --axis cq < /dev/null
python3 Accessory/probe/eqq_pool_fit_table.py --sides gpu_l40_qp --axis qp < /dev/null
```
判据：**必须核对输出中确有 LOO 数值行**（如 `av1_nvenc 1.4566 +1.2165 17 3.13`）。门限：软编 5.9、`librav1e` 族 7.5、NVENC 硬编 **CQ 与 QP 轴同为 5.9**（7.5 只给 rav1e）。

已修：4 份 Plan 文档（`PROMPT_T4_全流程验证.md`、`PROMPT_T4_NVENC等质量标定专项执行方案.md`、`PROMPT_L40_AV1等质量标定专项执行方案.md`、`PROMPT_等质量换算立项.md`）+ README + AGENTS.md + `comprehensive_verify.py`（拆成 `eqq_pool_fit_selftest`(CQ) 与 `eqq_pool_fit_selftest_qp`(QP) 两个 TestCase）。

顺带修 `comprehensive_verify.py` 的 `args.env` bug：`args.env` 在 argparse 下是 `str`，原 `args.env == Env.T4` 靠 `Env(str, Enum)` 子类才成立；新代码用 `args.env.value` 会 `KeyError` ⇒ 加 `_env_val()` 归一化。

## ③ L40 验收范围收敛到 4 项

**结论**：CPU 侧与 T4 侧已收口，**唯一能力缺口是 AV1 NVENC 硬编**（Turing 无 AV1 编码器，实跑报 `No capable devices found`）。L40 只需 4 项：

| 项 | 为什么必须 L40 |
|----|----------------|
| AV1 长视频冒烟 S1~S8 | 全链路唯一实跑 AV1 硬编；`envs=[L40]` |
| crf_cq --gpu 的 G7-6 | av1_nvenc `-cq` 表值判定，T4 恒 SKIP |
| AV1 矩阵 B 组 AC1 | av1_nvenc `-qp` 落带扫描，T4 B 组整体不执行 |
| AV1 标定落表 LOO | 非 GPU 需求，但只有 L40 数据才能判 |

**移出 L40 范围的项及理由**：
- `plan_gate` —— 整体 CPU 可跑，无 GPU 时 R5/R7→WARN、R8/RT-0/SMOKE-0→SKIP，**无 FAIL**。
- `nvenc_rc_diagnose` —— **纯主机侧工具**：只读 ffmpeg 选项表 + 磁盘搜 `nvEncodeAPI.h` 做文本解析，**不建 NVENC session、不编码任何一帧**；且全文零 AV1 内容（只看 `h264_nvenc`，退化到 `hevc_nvenc`）。`comprehensive_verify.py` 对它设 `required_gpu=True` 属**过度门控**。真正需要 GPU+SDK 的是 `nvenc_vbr_hq_verify.py`。
- `verify_equal_quality`（`envs=[cpu]`）、`nvenc_vbr_hq_verify`（`envs=[t4]`）—— L40 队列本就不含。

**Why 重要**：把「AV1 硬编」这一个缺口收敛成 4 个可执行动作，避免 L40 上重复跑 12 个子测试里 8 个与 AV1 无关的项。

## ④ 顺带核出的三处文档失效

1. **跨仓 ⑨ 组脚本已迁路径**：VU 提交 `ebdad41`（2026-10-06）把 `verify/verify_quality_mapping.py` 移到 `Accessory/verify/`，旧路径 `/workspace/VidUtils/verify/` 现为空目录。
2. **⑨ 组当前不是 14/14**：实跑 13 PASS + 1 FAIL —— ⑦ `[7] --threads 显式值两边不一致`（VE 得 `'3'` / VU 得 `'2'`），该组退出码非 0。属非 AV1 差异，需单独立项。
3. **CPU 侧 7/7 证据薄弱**：只有 commit `9d15dc6` message + `MEMORY.md:182` 一行，**无任何报告/日志留痕**；索引指向的 `memory/cpu-verification-pass.md` 两侧镜像都不存在且 git 全历史从未有过（悬空链接，违反 [索引双向完整](memory-index-integrity.md)）。准确口径是「7 项 PASS + plan_gate 预期跳过」，不是 7/7。

## 教训：脚本报「✅ 全部达标」不等于验证覆盖

`eqq_pool_fit_table.py` 在 0 个点时打印「拟合失败（样本 0 < 4）」并继续，最后仍打印「✅ 全部档位 LOO 达标」+ `exit 0`。这类**部分失败降级为提示**的设计让编排器无法据此判 FAIL（编排器只认退出码）。

⇒ 凡是判据脚本跑完一律要**看输出正文**，核对关键档位那几行是否真有数值，不能只看退出码和汇总符号。这与 [自述不是执行证据](feedback_artifacts_not_execution_evidence.md)、[先算完再落盘](feedback_atomic_file_write.md) 同源。

相关：[CPU/T4 已完成项清单](l40-verification-scope-2026-10.md)、[L40 AV1 标定（已降级为不可审计）](av1-nvenc-l40-calibration.md)、[T4 NVENC 标定](equal-quality-t4-nvenc-calibration.md)