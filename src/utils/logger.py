#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# [P2.5] 统一日志基础设施。
#
# 设计目标：
#   · 为编排层/工具层提供 logging 入口（external/* 子系统暂保留 print 遥测，
#     其输出含大量运维依赖的 FIX 标注，迁移需 GPU 环境回归后分批进行）。
#   · 控制台输出保持与原 print 行为兼容；文件日志按 config/logging 配置落盘，
#     UTF-8 编码，供事后排障（此前运行记录只存在于终端缓冲）。
#
# 使用：
#   from logger import init_logging, get_logger
#   init_logging(level="INFO", log_dir=..., log_to_file=True)   # 进程内一次
#   log = get_logger("main")                                    # 任意处获取

import logging
import os
from datetime import datetime
from pathlib import Path

_INITIALIZED = False
_LOG_FILE = None

_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def init_logging(level: str = "INFO",
                 log_dir: str = "",
                 log_to_file: bool = True,
                 log_to_console: bool = True) -> None:
    """初始化根日志配置（进程内幂等，重复调用仅更新级别）。"""
    global _INITIALIZED, _LOG_FILE

    root = logging.getLogger()
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(lvl)
    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    # 幂等：清掉本模块此前挂的 handler，避免重复行
    for h in list(root.handlers):
        if getattr(h, "_ve_marker", False):
            root.removeHandler(h)

    if log_to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch._ve_marker = True
        root.addHandler(ch)

    if log_to_file:
        try:
            out_dir = Path(log_dir) if log_dir else Path("logs")
            out_dir.mkdir(parents=True, exist_ok=True)
            _LOG_FILE = out_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            fh = logging.FileHandler(str(_LOG_FILE), encoding="utf-8")
            fh.setFormatter(formatter)
            fh._ve_marker = True
            root.addHandler(fh)
        except OSError as e:
            # 文件不可写时降级为仅控制台，不影响主流程
            logging.getLogger(__name__).warning(f"日志文件创建失败({e})，仅控制台输出")

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """获取命名 logger（未 init 时也可用，退化为默认 stderr WARNING+）。"""
    return logging.getLogger(name)


def log_file_path() -> str:
    """返回当前运行日志文件路径（未启用文件日志返回空串）。"""
    return str(_LOG_FILE) if _LOG_FILE else ""
