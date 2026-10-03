# eqq_diag —— 等质量标定的一次性诊断脚本（归档）

**这些不是复用工具**，是 2026-10-02/03 第七、八版标定过程中的**一次性排查脚本**，
保留仅为让当时的结论可追溯、可复现。日常复用请看 `Accessory/probe/eqq_*.py`。

可复用的工具已泛化重构到 `Accessory/probe/`：

| 本目录旧脚本 | 现行复用工具 |
|---|---|
| `measure_uni.py` / `run_gen.sh` / `run_ji.sh` | `probe/eqq_calibrate_clip.py` + `probe/eqq_calibrate_batch.py` |
| `final_table_v8_commit.py` / `harvest_*.py` | `probe/eqq_pool_fit_table.py` |
| `/tmp/_slices/gen.py`（一次性，未归档） | `probe/eqq_slice_prep.py` |
| `eqq_watch.sh` / `eqq_status.sh` / `watch_ji.sh` | `probe/eqq_watch_batch.py` |

## 文件分组

**锚点/ 素材集合取证**（用于定位「素材名相同掩盖数据缺口」）
`diag_*.py`、`cmp_anchor*.py`、`cmp_ab_anchor_7src.py`、`collect_10s.py`、
`measure_anchorA.py`、`measure_anchorB_ve.py`、`measure_bbc_anchorB.py`、`recalib_anchorB.py`

**LOO / 拟合行为诊断**
`diag_loo.py`、`diag_loo2.py`、`diag_disp.py`、`diag_pw.py`、`diag_model.py`、
`diag_split.py`、`diag_quad_check.py`、`diag_peranchor.py`、`diag_nanchor.py`、
`diag_newanchors.py`、`diag_weighted.py`、`diag_pooled.py`、`diag_why_grow.py`、
`diag_vp9.py`、`diag_w2.py`、`diag_ceiling.py`、`diag_ab.py`、`diag_cause.py`

**表值落表演进**（第四~八版）
`final_table_b.py`、`final_table_v5.py`、`final_table_v6.py`、`final_table_v7.py`、
`final_table_v8.py`、`final_table_v8_strict.py`、`apply_v8.py`、`root_cause_v8.py`、
`rename_tables.py`

**会话/ cron 辅助**
`read_cron.py`、`fix_cron_dup.py`、`check_rename.py`

**编码器冒烟**
`repro_enc.py`