---
name: 测试资产已迁入 Accessory/ 并按用途分类
description: 2026-09-26：原 tests/ 整体搬迁到 Accessory/<分类>/（analyze|benchmark|probe|test|verify|infra|docs|archive|video_check），多数脚本同时改名
type: project
---

`tests/` 目录**已不存在**（2026-09-26 清空移除）。原 87 项资产按用途迁入仓库根 `Accessory/`：

`analyze/`（事后分析）· `benchmark/`（性能基准）· `probe/`（探测·诊断·复现·硬件实测矩阵）·
`test/`（pytest 真回归，**保留 `test_` 前缀**）· `verify/`（验收门禁）· `infra/`（测试基建）·
`docs/`（结论记录）· `archive/`（*.bak* 备份件）· `video_check/`（整体搬迁）。
根级保留 `conftest.py`（pytest 对整棵树生效）与 `run_all_isolated.sh`（隔离跑入口）。

**Why:** 用户要求「按用途分类 + 名称更符合实际用意 + 集中到 Accessory 便于管理」。核心是消除误导性的 `test_` 前缀——原 `test_nvenc_*` 多为 `__test__ = False` 的独立 harness，却被 pytest 收集、被 `run_all_isolated.sh` 当测试跑，且是 NVENC SIGSEGV 的主要来源。

**How to apply:**
- 找测试/验收脚本先去 `Accessory/` 对应分类目录，**不要再 grep `tests/`**；完整旧名→新名映射见 `Accessory/README.md`。
- 改名后 NVENC harness 不再被 pytest 收集（预期内），改由 `run_all_isolated.sh` 用 `python <file>` 直跑；`pytest -m "not hw"` 与门禁基线（`plan_implementation_gate.py`）的**计数会漂移，需在生产侧重取基线**。
- 新增脚本请落进对应分类目录；真 pytest 回归必须放 `test/` 且保留 `test_` 前缀，否则收集不到。
- 本次同步修正：`pytest.ini` 的 `testpaths`、门禁 `COVERAGE_ROOTS`、14 个脚本的项目根推导（多了一层目录）、3 处同目录 import 改名、全仓 147 个文档的路径引用。
