#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# IFRNet Video Enhancement - 配置模块（路径与模型常量）。
# 镜像 external/realesrgan_video/config.py 的职责。

from __future__ import annotations

import os
import sys
from typing import Dict

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)


_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
base_dir      = str(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
models_ifrnet = os.path.join(base_dir, 'models_IFRNet', 'checkpoints')
sys.path.insert(0, os.path.join(base_dir, 'external', 'IFRNet'))
sys.path.insert(0, models_ifrnet)

# ── [FIX-CUDA-GRAPH-WARP] CUDA-Graph 安全 warp ───────────────────────────────

MODEL_MODULE_MAP: Dict[str, str] = {
    'IFRNet_Vimeo90K':   'models.IFRNet',
    'IFRNet_S_Vimeo90K': 'models.IFRNet_S',
    'IFRNet_L_Vimeo90K': 'models.IFRNet_L',
}

MODEL_STRIDE = 32
MODEL_NAME_MAP: Dict[str, str] = {
    'IFRNet_Vimeo90K':   'IFRNet_Vimeo90K.pth',
    'IFRNet_S_Vimeo90K': 'IFRNet_S_Vimeo90K.pth',
    'IFRNet_L_Vimeo90K': 'IFRNet_L_Vimeo90K.pth',
}
