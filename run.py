#!/usr/bin/env python3
"""
快速运行脚本
"""

import sys
import os

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入主程序
from main import main

if __name__ == "__main__":
    sys.exit(main())
