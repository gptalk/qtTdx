# -*- coding: utf-8 -*-
"""tsfresh pipeline 共享配置。改这里等于改所有脚本。"""
import os

# -------- TQ 路径 --------
TQ_PLUGINS_DIR = 'C:/new_tdx_mock/PYPlugins/user'

# -------- 数据回退 --------
LOCAL_FALLBACK_CODES = ['000001.SH', '002475.SZ']
SECTOR_NAME = '通达信88'
LOOKBACK_YEARS = 5

# -------- 滑窗 / 标签 --------
WINDOW = 30       # 每个样本的 bar 数
HORIZON = 5       # 标签 = 窗口结束后 N 日

# -------- tsfresh --------
FDR_LEVEL = 0.05          # 多重检验校正阈值
TSFRESH_N_JOBS = 0        # 0=单进程,Windows 下 multiprocessing 容易卡死

# -------- 模型 --------
LR_C = 0.5
LR_MAX_ITER = 2000
LR_CLASS_WEIGHT = 'balanced'
LR_RANDOM_STATE = 42

# -------- 路径 --------
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # backtrace/
OUTPUTS_DIR = os.path.join(BACKTRACE_DIR, 'outputs')                            # backtrace/outputs/
