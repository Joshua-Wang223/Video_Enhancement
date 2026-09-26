#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_regression_min.py — 兼容别名（v2 合并后）
================================================

原独立行为回归脚本已并入 Accessory/verify/plan_implementation_gate.py 的「C-行为验证」阶段
（BEH-A~F 组，含原 30 断言并新增编译扫描与配置校验器动态行为）。

本文件保留仅为兼容 AGENTS.md 记载的既有命令入口，等价于：

    python Accessory/verify/plan_implementation_gate.py --behavior-only --no-report-file

新增验证项请直接修改 plan_implementation_gate.py，勿在此文件堆叠逻辑。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plan_implementation_gate as _vpi  # noqa: E402

if __name__ == "__main__":
    sys.exit(_vpi.main(argv=["--behavior-only", "--no-report-file"]))
