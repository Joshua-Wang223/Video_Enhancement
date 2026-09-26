# Accessory —— 测试 / 验收 / 诊断资产集中区

原 `tests/` 下 87 项资产按**用途**分类搬入本目录（2026-09-26）。`tests/` 已清空移除。

## 目录语义

| 目录 | 用途 | 典型场景 |
|---|---|---|
| `analyze/` | 事后分析：对产物视频/日志做质量分析、阈值标定 | 花屏/鬼影定位、码流质量统计、切镜阈值标定 |
| `benchmark/` | 性能基准：版本/参数对比跑分 | IFRNet 各版本 FPS 与帧完整性横评 |
| `probe/` | 探测·诊断·复现：GPU/CUDA/NVENC 语义探针、故障定位、最小复现、硬件实测矩阵 | CUDA context 语义、NVENC 字段偏移、段错误复现 |
| `test/` | pytest 真回归（**保留 `test_` 前缀**，pytest 依赖此前缀收集） | 读帧器超时、stdin 加固、缓存持久化等 CPU 可跑断言 |
| `verify/` | 验收·门禁：码流完整性、优化方案落地、修复效果校验 | `plan_implementation_gate.py` 门禁全套 |
| `infra/` | 测试基建：注入/包装工具（不改生产代码） | `sitecustomize.py` libcuda 调用日志、故障注入包装器 |
| `docs/` | 结论与记录（非脚本）：md / txt / log | 实测结论、诊断日志、资产清单 |
| `archive/` | 历史备份件（`*.bak*`） | 门禁脚本旧版、码流验收旧版 |
| `video_check/` | 视频质量检查工具与配置（整体搬迁，内部结构未动） | GPU 视频质检 |

根级保留两项**必须**在测试树根的资产：
- `conftest.py` —— pytest 对本目录及所有子目录生效；
- `run_all_isolated.sh` —— 逐文件隔离跑入口（NVENC SIGSEGV 隔离方案 A）。

## 命名约定（本次改名原则）

1. **去掉误导性的 `test_` 前缀**：原 `test_nvenc_*` 多数并非 pytest 模块（或显式 `__test__ = False`），
   却会被 `pytest` 收集、被 `run_all_isolated.sh` 当作测试文件，且是 SIGSEGV 的主要来源。
   改名后它们落到 `probe/`，由 `run_all_isolated.sh` 以 `python <file>` 方式执行。
2. **真 pytest 回归保留 `test_` 前缀**（`Accessory/test/`），否则 pytest 收不到。
3. 名称按「对象 + 动作/性质」命名：`nvenc_completion_event_matrix.py`、
   `segment_bitstream_verify_v5.py`、`batch_size_optimizer_probe.py`。
4. 版本后缀（`_v1`~`_v5`）保持不变，避免与历史结论对不上号。

## 旧名 → 新名全表

### analyze/（11）
| 旧名 | 新名 |
|---|---|
| `analyze_interp_ghost.py` | `interp_ghost_analyzer.py` |
| `analyze_video_ifrnet.py` / `.com` | `video_ifrnet_analyzer.py` / `.cmd` |
| `analyze_video_pipeline.py` / `_v2` / `_v3` | `video_pipeline_analyzer.py` / `_v2` / `_v3` |
| `analyze_video_realesrgan.py` / `.com` | `video_realesrgan_analyzer.py` / `.cmd` |
| `calibrate_scene_cut_threshold.py` | `scene_cut_threshold_calibrator.py` |
| `diagnose_scene_cut_ghost.py` | `scene_cut_ghost_analyzer.py` |

### benchmark/（3）
| 旧名 | 新名 |
|---|---|
| `benchmark_ifrnet_versions.py` / `_v2` / `_v3` | `ifrnet_versions_benchmark.py` / `_v2` / `_v3` |

### probe/（31）
| 旧名 | 新名 |
|---|---|
| `probe_cuda_context.py` | `cuda_context_probe.py` |
| `Probe_Optimisation_batch_size.py` | `batch_size_optimizer_probe.py` |
| `PyTorch_NVML_Test.py` | `pytorch_nvml_probe.py` |
| `diagnose_hevc_la.py` | `hevc_lookahead_diagnose.py` |
| `diagnose_lockbitstream_timestamp.py` | `lockbitstream_timestamp_diagnose.py` |
| `diagnose_nvenc_qp0_segv.py` | `nvenc_qp0_segv_repro.py` |
| `diagnose_nvenc_rc_mode.py` | `nvenc_rc_mode_diagnose.py` |
| `diagnose_profilelevel_offset.py` | `nvenc_profilelevel_offset_diagnose.py` |
| `diagnose_reader_rgb_path_consistency.py` | `reader_rgb_path_diagnose.py` |
| `diagnose_targetquality_offset.py` | `nvenc_targetquality_offset_diagnose.py` |
| `repro_ifrnet_lookahead.py` | `ifrnet_lookahead_repro.py` |
| `repro_real_frames.py` | `real_frames_encode_repro.py` |
| `test_nvenc_completion_event.py` / `_v1`..`_v5` | `nvenc_completion_event_matrix.py` / `_v1`..`_v5` |
| `test_nvenc_comprehensive.py` | `nvenc_comprehensive_matrix.py` |
| `test_nvenc_ipc_worker.py` | `nvenc_ipc_worker_probe.py` |
| `test_nvenc_la_frame_conservation.py` | `nvenc_la_frame_conservation_suite.py` |
| `test_nvenc_pre_torch.py` | `nvenc_session_pre_torch_probe.py` |
| `test_nvenc_sdk_realesrgan.py` | `nvenc_sdk_realesrgan_suite.py` |
| `test_nvenc_vbr_hq_offsets.py` | `nvenc_vbr_hq_offsets_probe.py` |
| `test_nvenc_struct_dump` / `.c` | `nvenc_struct_dump_bin` / `nvenc_struct_dump.c` |
| `test_pipe4_la8_corruption.py` | `pipe4_la8_corruption_diff.py` |
| `test_sps_pps_startup.py` | `sps_pps_startup_repro.py` |
| `test_ltrace_ffmpeg_nvenc.sh` | `ltrace_ffmpeg_nvenc.sh` |
| `test_rate_mode_upgrade.sh` | `nvenc_rate_mode_upgrade_check.sh` |
| `test_segmentation_pipeline.sh` | `segmentation_pipeline_e2e.sh` |

### test/（7，名称不变）
`test_chroma_false_positive.py`、`test_esrgan_apply_sps_pps_equivalence.py`、
`test_frame_count_probe.py`、`test_prescan_cache_persistence.py`、
`test_probe_batch_size_search_logic.py`、`test_reader_unbound_watchdog.py`、
`test_stdin_hardening_linux.py`

### verify/（13）
| 旧名 | 新名 |
|---|---|
| `verify_plan_implementation.py` | `plan_implementation_gate.py` |
| `test_regression_min.py` | `verify/test_regression_min.py`（门禁别名，**保留 `test_` 前缀**） |
| `verify_segment_bitstream.py` / `_v2`..`_v5` | `segment_bitstream_verify.py` / `_v2`..`_v5` |
| `verify_crf_cq_unification.py` | `crf_cq_unification_verify.py` |
| `verify_rcparams_offset.py` | `nvenc_rcparams_offset_verify.py` |
| `verify_rotation_backport.py` | `stream_ts_reassoc_backport_verify.py` |
| `verify_scene_cut_fix.py` | `scene_cut_fix_verify.py` |
| `minimal_validate_enhanced.py` | `minimal_enhanced_validation.py` |
| `test_parallel_validate.py` | `parallel_validation.py` |

### infra/（3）
| 旧名 | 新名 |
|---|---|
| `sitecustomize.py` | `sitecustomize.py`（Python 自动导入，名称不可改） |
| `_faultinj_wrap.py` | `fault_injection_wrapper.py` |
| `_pipe_deadlock_test.py` | `pipe_deadlock_repro.py` |

### docs/（13）/ archive/（3）/ video_check/（整体搬迁）
均为原名搬入，未改名。

## 迁移时同步修正的引用

- `pytest.ini`：`testpaths = Accessory`
- `Accessory/run_all_isolated.sh`：`find Accessory/test -name 'test_*.py'`（pytest）+ `find Accessory/probe -name '*.py'`（`python <file>` 直跑）
- `Accessory/verify/plan_implementation_gate.py`：`COVERAGE_ROOTS` 加 `Accessory`、`regression_min` 路径、`tests_dir` 自检路径
- 14 个脚本的「项目根推导」：`.parent.parent` → `.parent.parent.parent`（原在 `tests/` 一层下，现两层）
- 同目录 import 改名：`hevc_lookahead_diagnose.py` → `nvenc_la_frame_conservation_suite`；
  `test_regression_min.py` → `plan_implementation_gate`；`test_probe_batch_size_search_logic.py` → `batch_size_optimizer_probe`
- `.gitignore`：`tests/validation_report_final.txt` → `Accessory/validation_report_final.txt`
- 全仓 147 个文档/脚本中的 `tests/<旧名>` 引用已替换为 `Accessory/<分类>/<新名>`

## ⚠️ 待 GPU 侧复核

1. `python Accessory/verify/plan_implementation_gate.py` —— 门禁基线可能因路径/文件数变化而漂移，需重新取基线。
2. `bash Accessory/run_all_isolated.sh` —— `probe/` 下 harness 由 pytest 改为直跑，四态分布会变化。
3. `pytest -m "not hw"` —— 改名后 NVENC harness 不再被 pytest 收集，收集数量会下降（预期内）。
4. `bash Accessory/probe/batch_size_optimizer_probe.py`（原 `Probe_Optimisation_batch_size.py`）仍需 GPU 实跑。
