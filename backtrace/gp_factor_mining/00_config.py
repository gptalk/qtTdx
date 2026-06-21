# -*- coding: utf-8 -*-
"""
GP 因子挖掘 — 统一配置
所有模块共享:训练/测试时间窗、股票池、种群规模、残差轮数、路径等。
修改下面这块就够了,不要散落在各模块里写死。
"""
from pathlib import Path

# ========================= 路径 =========================
# 项目根(本文件在 backtrace/gp_factor_mining/ 下)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GP_DIR       = Path(__file__).resolve().parent
DATA_DIR     = GP_DIR / "data"          # 中间数据(parquet/csv 落盘位置)
FACTOR_DIR   = GP_DIR / "factors"       # 挖出的因子公式 + 值
LOG_DIR      = GP_DIR / "logs"
for d in (DATA_DIR, FACTOR_DIR, LOG_DIR):
    d.mkdir(exist_ok=True)

# ========================= 数据源 =========================
USE_TQ       = True                     # True 优先 TQ,失败自动回退本地 CSV
TQ_SECTOR    = "沪深A股"                 # TQ 板块名;若报错可改 "全部A股" / "中证流通"
TQ_INIT_PATH = "C:/new_tdx_mock/PYPlugins/user"

# ========================= 训练 / 测试时间窗 =========================
# 训练期做因子挖掘;测试期做样本外验证 + 回测
TRAIN_START  = "2015-01-01"
TRAIN_END    = "2020-12-31"
TEST_START   = "2021-01-01"
TEST_END     = "2025-12-31"

# 数据下载窗口:要比训练/测试各往前推 60 天,留出指标 warm-up
DATA_FETCH_START = "2014-10-01"

# ========================= 股票池过滤 =========================
# 中证全指 ≈ 全部 A 股,这里只做硬过滤
MIN_LIST_DAYS   = 60                    # 上市不满 60 天的次新股剔除
MIN_PRICE       = 2.0                   # 股价 < 2 元的壳股剔除(可调)
EXCLUDE_ST      = True                  # 剔 ST/*ST
EXCLUDE_SUSPEND = True                  # 停牌日剔除(成交量=0 或 close 缺失)

# ========================= 标签 =========================
# 预测未来 N 日收益,GP 用它当 y
HOLD_PERIOD = 20                        # 持有期 20 个交易日 ≈ 1 个月
LABEL_NAME  = f"fwd_ret_{HOLD_PERIOD}d"

# ========================= GP 参数 =========================
POP_SIZE            = 2000              # 种群规模(用户指定)
N_GENERATIONS       = 30                # 进化代数
TOURNAMENT_SIZE     = 20
P_CROSSOVER         = 0.80
P_SUBTREE_MUTATION  = 0.10
P_HOIST_MUTATION    = 0.05
P_POINT_MUTATION    = 0.05
P_POINT_REPLACE     = 0.05
PARSIMONY_COEFFICIENT = 0.001           # 防膨胀(bloat),越大越偏好小树
MAX_DEPTH_INIT      = 6                 # 初始最大深度
MAX_DEPTH           = 8                 # 进化最大深度

# ========================= 多轮残差挖掘 =========================
N_RESIDUAL_ROUNDS   = 5                 # 残差轮数(用户指定开启多轮残差)
MIN_IMPROVE_IC      = 0.005             # 继续挖的边际 IC 阈值;低于则停
RESIDUAL_TOP_K      = 5                 # 每轮保留 top K 因子

# ========================= 入库标准 =========================
# 因子进入正式因子库的硬门槛
IN_SAMPLE_ICIR_MIN  = 1.5               # 样本内 RankICIR >= 1.5
OUT_SAMPLE_IC_MIN   = 0.04              # 样本外 RankIC >= 0.04
OUT_SAMPLE_ICIR_MIN = 1.0               # 样本外 RankICIR >= 1.0
MAX_TURNOVER        = 1.0               # 月换手率上限
MAX_CORR_WITH_POOL  = 0.70              # 与已有因子库最大相关性

# ========================= 回测参数 =========================
BACKTEST_TOP_N      = 50                # 每月选 top N 只
BACKTEST_REBAL_FREQ = "M"               # 月频调仓
INIT_CASH           = 1_000_000
FEE_RATE            = 0.0003            # 单边手续费 0.03%
SLIPPAGE            = 0.001             # 滑点 0.1%
SIZE_GRANULARITY    = 100               # A 股最小 100 股

# ========================= 随机性 =========================
RANDOM_STATE = 42

# ========================= 截面标准化 =========================
# 所有特征和标签都做截面 rank → zscore,去掉市值/价格水平影响
CS_STD_METHOD = "rank_zscore"           # rank_zscore | zscore | rank_pct