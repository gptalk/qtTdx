# -*- coding: utf-8 -*-
"""把 backtrace/ 插进 sys.path,使测试能 `from common import ...`(与脚本同一套导入约定)。"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTRACE_DIR = os.path.join(REPO_ROOT, 'backtrace')
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)