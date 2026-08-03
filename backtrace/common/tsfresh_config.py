# -*- coding: utf-8 -*-
"""
tsfresh pipeline 共享配置。改这里等于改所有脚本。

约定/做法:
  - 所有脚本 `from common import tsfresh_config as C`,统一用 C.WINDOW 等引用
  - 输出 CSV/HTML 默认落 backtrace/outputs/(gitignored)
  - tsfresh 强制单进程(Windows 下 multiprocessing 卡死)

输入/输出:
  - 输入:无(纯常量模块)
  - 输出:14 个公开常量

依赖:无

用法:
  from common import tsfresh_config as C
  print(C.WINDOW, C.HORIZON)
"""
import os

# -------- TQ 路径 --------
TQ_PLUGINS_DIR = 'C:/new_tdx_mock/PYPlugins/user'   # tqcenter.py 所在目录,sys.path.insert 用

# -------- 数据回退 --------
LOCAL_FALLBACK_CODES = ['000001.SH', '002475.SZ']    # TQ 拉取失败时回退的本地 CSV 代码(必须有对应 backtrace/<code>_daily.csv)
SECTOR_NAME = '通达信88'                              # 默认板块(88 个通达信行业)
LOOKBACK_YEARS = 5                                    # 默认回看年数,实际取 LOOKBACK_YEARS*365+30 天

# -------- 滑窗 / 标签 --------
WINDOW = 30       # 每个样本的 bar 数(tsfresh 滚动窗口大小)
HORIZON = 5       # 标签 = 窗口结束后 N 日的收益(正负/与大盘对比)

# -------- tsfresh --------
FDR_LEVEL = 0.05          # 多重检验校正阈值(Benjamini-Hochberg);0 特征时 select_relevant 会自动放宽到 0.20
TSFRESH_N_JOBS = 0        # 0=单进程,Windows 下 multiprocessing 容易卡死,生产环境必须保持 0

# -------- 模型 --------
LR_C = 0.5                # LogisticRegression 正则强度倒数
LR_MAX_ITER = 2000        # LR 最大迭代次数
LR_CLASS_WEIGHT = 'balanced'   # 自动平衡正负样本
LR_RANDOM_STATE = 42      # 复现实验结果

# -------- 路径 --------
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # backtrace/ 绝对路径
OUTPUTS_DIR = os.path.join(BACKTRACE_DIR, 'outputs')                            # backtrace/outputs/(gitignored)
