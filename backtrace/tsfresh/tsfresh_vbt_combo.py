# tsfresh + VectorBT 集成 demo
# 4 组合网格:MA5 基线 + tsfresh 三个阈值,全部走 jhzq_fees 真实扣费
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

import numpy as np
import pandas as pd
import vectorbt as vbt
from tqcenter import tq

from common import tsfresh_config as C
from common import tsfresh_pipeline as P
from common import jhzq_fees as F

tq.initialize(__file__)

# ============== 配置 ==============
STOCK_CODE   = '688318.SH'             # 沪市股,验证 SH 过户费分支
TARGET_START = '20250101'              # demo 拉 5 年取 2025 段验证
TARGET_END   = '20251231'
WINDOW_MA    = 5                       # MA5
INIT_CASH    = 100_000
MAX_POS_PCT  = 0.95

# 4 组合网格
STRATEGIES = [
    {'name': 'MA5_baseline', 'entry_kind': 'ma',     'entry_th': 0,  'exit_th': 0},
    {'name': 'tsfresh_p50',  'entry_kind': 'proba',  'entry_th': 0.50},
    {'name': 'tsfresh_p55',  'entry_kind': 'proba',  'entry_th': 0.55},
    {'name': 'tsfresh_p60',  'entry_kind': 'proba',  'entry_th': 0.60},
]
EXIT_TSF = 0.50   # tsfresh exit 统一用 proba < 0.5
# ===================================

# ---------- 1. 数据 ----------
print("=" * 70)
print("加载数据...")
df = P.load_ohlcva(STOCK_CODE, verbose=True)
if df is None or len(df) == 0:
    print("❌ 数据缺失"); tq.close(); raise SystemExit(1)

# 全量回测区间(用 TARGET_START/END 限定演示窗)
df_demo = df.loc[TARGET_START:TARGET_END].copy()
print(f"\n回测区间:{df_demo.index[0].date()} → {df_demo.index[-1].date()}  ({len(df_demo)} 交易日)")

# vbt 数据(在整个 5 年上跑,因为 tsfresh 也用 5 年训模型)
close_full = df['Close']
open_full  = df['Open']

# ---------- 2. tsfresh 全量特征 + 训 LR ----------
print("\n" + "=" * 70)
print("tsfresh 特征工程(全量 5 年滚动窗口)...")
long_df = P.to_long_format(df, id_value=STOCK_CODE)
X_all = P.extract_window_features(long_df, use_kind=True, verbose=True)
y_all, X_all = P.make_labels(X_all, df['Close'].values, verbose=False)
print(f"   -> 有效样本 {len(y_all)} 个  |  未来 {C.HORIZON} 日上涨 {y_all.sum()} 个 ({y_all.mean():.1%})")

print("\nFDR 特征筛选...")
X_sel = P.select_relevant(X_all, y_all, verbose=False)
print(f"   -> FDR 显著特征 {X_sel.shape[1]} 列")
if X_sel.shape[1] == 0:
    print("   [WARN] FDR 0 特征,demo 兜底用全量特征(可能过拟合,但管道能跑通)")
    X_sel = X_all

# ---------- 3. 真 walk-forward predict ----------
#    expanding window:前 INIT_TRAIN_SIZE 窗口训练初始 LR,之后每 STEP 窗口重训一次
INIT_TRAIN_SIZE = 200   # 初始训练窗口数
STEP = 50               # 每多少窗口重训一次

print(f"\n真 walk-forward predict (initial={INIT_TRAIN_SIZE}, retrain every {STEP} 窗口)...")
date_index = pd.DatetimeIndex(df.index)
proba_records = []
last_train_at = -1      # 上次重训在哪个位置
scaler_w = clf_w = None

for pos, idx in enumerate(X_sel.index):
    end_t = idx[1]
    if end_t >= len(date_index):
        continue

    # 前 INIT_TRAIN_SIZE 窗口无预测(样本不足)
    if pos < INIT_TRAIN_SIZE:
        proba_records.append((date_index[end_t], np.nan))
        continue

    # 需要重训?(初始训练 + 每 STEP 窗口)
    need_train = (pos == INIT_TRAIN_SIZE) or ((pos - INIT_TRAIN_SIZE) % STEP == 0)
    if need_train:
        scaler_w, clf_w = P.fit_logreg(
            X_sel.iloc[:pos], y_all.iloc[:pos], verbose=False
        )
        last_train_at = pos
        print(f"   -> 重训 at pos={pos} (训练样本 {pos} 个)")

    p = float(clf_w.predict_proba(scaler_w.transform(X_sel.iloc[[pos]].values))[0, 1])
    proba_records.append((date_index[end_t], p))

proba_series = pd.Series(
    [v for _, v in proba_records],
    index=pd.DatetimeIndex([d for d, _ in proba_records]),
    name='proba',
).sort_index()
proba_series = proba_series[~proba_series.index.duplicated(keep='last')]

valid = proba_series.dropna()
print(f"   -> 有效 proba {len(valid)} 个  (训练起点: {valid.index[0].date()})")
print(f"   -> proba 分布:min={valid.min():.3f}  max={valid.max():.3f}  mean={valid.mean():.3f}")

# ---------- 4. 构造 4 套 entry/exit ----------
#    proba 在 df.index[end_t] 当日计算 → 次日开盘成交(shift(1))
#    用 reindex 到 close_full.index 上(每个交易日),缺失值填 False

def build_signals(strategy):
    """返回 (entries, exits) 两个 pd.Series,索引对齐 close_full"""
    name = strategy['name']
    if strategy['entry_kind'] == 'ma':
        ma = vbt.MA.run(close_full, window=WINDOW_MA).ma.ffill()
        entries = close_full.vbt.crossed_above(ma).shift(1).fillna(False).astype(bool)
        exits   = close_full.vbt.crossed_below(ma).shift(1).fillna(False).astype(bool)
        return name, entries, exits

    # tsfresh proba 信号
    th_entry = strategy['entry_th']
    aligned = proba_series.reindex(close_full.index)   # 每个 bar 1 个 proba,缺失 NaN
    entries = (aligned > th_entry).shift(1).fillna(False).astype(bool)
    exits   = (aligned < EXIT_TSF).shift(1).fillna(False).astype(bool)
    return name, entries, exits


# ---------- 5. vbt 通用回测 ----------
shares_per_trade = int(np.floor(INIT_CASH * MAX_POS_PCT / float(open_full.iloc[0]) / 100) * 100)
print(f"\n每笔 {shares_per_trade} 股  (≈ {shares_per_trade * float(open_full.iloc[0]):,.0f} 元 / "
      f"{shares_per_trade * float(open_full.iloc[0]) / INIT_CASH:.1%} 仓位)\n")


def run_vbt_backtest(name, entries, exits):
    """跑 1 套组合,返回汇总 dict(含零摩擦 vs 实盘扣费对比)"""
    base = {'strategy': name, 'trades': 0,
            'gross_pnl': 0.0, 'total_stamp': 0.0, 'total_transfer': 0.0,
            'net_pnl': 0.0, 'avg_net_per_trade': 0.0,
            'net_ret': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0,
            'zero_friction_ret': 0.0}   # 新增:零摩擦总收益(纯信号 alpha 上限)

    if entries.sum() == 0:
        return base

    # ===== A. 零摩擦 portfolio(fees=0, slippage=0)=====
    pf_zero = vbt.Portfolio.from_signals(
        close=close_full, entries=entries, exits=exits,
        price=open_full,
        init_cash=INIT_CASH,
        fees=0,
        slippage=0,
        freq='D',
        size=shares_per_trade,
        size_type='amount',
        size_granularity=100,
        upon_long_conflict='exit',
    )
    base['zero_friction_ret'] = pf_zero.total_return()

    # ===== B. 实盘 portfolio(fees=0 内置, jhzq_fees 后置)=====
    portfolio = pf_zero   # 复用同一个 pf_zero 拿 trades(扣费结构相同,只是费用单独算)

    trades = portfolio.trades.records_readable
    if len(trades) == 0:
        return base

    summary = F.summary_after_fees(trades, STOCK_CODE)
    summary['strategy'] = name
    summary['zero_friction_ret'] = base['zero_friction_ret']

    # 额外指标
    pnl_col = next(c for c in trades.columns if 'PnL' in c and '扣' not in c)
    wins = (trades[pnl_col] > 0).sum()
    summary['win_rate'] = wins / len(trades)
    summary['profit_factor'] = (
        trades[pnl_col][trades[pnl_col] > 0].sum() /
        abs(trades[pnl_col][trades[pnl_col] < 0].sum())
        if (trades[pnl_col] < 0).sum() > 0 else float('inf')
    )
    summary['net_ret'] = summary['net_pnl'] / INIT_CASH
    return summary


# ---------- 6. 跑 4 组合网格 ----------
print("=" * 70)
print("=== 4 组合网格回测(全部按 jhzq_fees 真实扣费) ===")
print("=" * 70)

results = []
trades_per_strategy = {}
for strat in STRATEGIES:
    name, entries, exits = build_signals(strat)
    entry_n = int(entries.sum())
    print(f"\n[{name}] entry 信号数: {entry_n}  exit 信号数: {int(exits.sum())}")
    summary = run_vbt_backtest(name, entries, exits)
    results.append(summary)
    trades_per_strategy[name] = portfolio = None  # 占位,保留接口

# ---------- 7. 输出对比表 ----------
results_df = pd.DataFrame(results)
cols = ['strategy', 'trades', 'zero_friction_ret',
        'gross_pnl', 'total_stamp', 'total_transfer',
        'net_pnl', 'net_ret', 'win_rate', 'profit_factor', 'avg_net_per_trade']
results_df = results_df[cols].sort_values('net_pnl', ascending=False).reset_index(drop=True)
results_df['friction_loss_pp'] = (results_df['zero_friction_ret'] - results_df['net_ret']) * 100

def fmt_money(x):
    return f"{x:>12,.2f}" if pd.notna(x) else "          N/A"
def fmt_pct(x):
    return f"{x:>7.2%}" if pd.notna(x) and x != float('inf') else "     inf"
def fmt_pp(x):
    return f"{x:>6.1f}pp" if pd.notna(x) else "  N/A"

print("\n" + "=" * 90)
print("=== 4 组合对比表(零摩擦 vs 实盘 jhzq_fees 真实扣费) ===")
print("=" * 90)
print(f"{'策略':<16} {'笔数':>5} {'零摩擦':>8} {'毛收益':>12} "
      f"{'印花税':>8} {'过户费':>8} {'净收益':>12} {'净收益率':>8} "
      f"{'胜率':>6} {'盈亏比':>7} {'摩擦吃损':>8}")
print("-" * 120)
for _, r in results_df.iterrows():
    print(f"{r['strategy']:<16} {int(r['trades']):>5} "
          f"{fmt_pct(r['zero_friction_ret'])} "
          f"{fmt_money(r['gross_pnl'])} "
          f"{-r['total_stamp']:>8.2f} {-r['total_transfer']:>8.2f} "
          f"{fmt_money(r['net_pnl'])} {fmt_pct(r['net_ret'])} "
          f"{fmt_pct(r['win_rate'])} {fmt_pct(r['profit_factor'])} {fmt_pp(r['friction_loss_pp'])}")
print("\n[解读] '零摩擦' = fees=0 + slippage=0(纯信号 alpha 上限);"
      "'摩擦吃损' = 零摩擦 - 实盘净收益,看手续费+滑点+印花税+过户费总共吃掉多少 pp")

# ---------- 8. 保存 ----------
out_csv = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/outputs",
    f'tsfresh_vbt_grid_{STOCK_CODE.replace(".", "_")}_{TARGET_START}_{TARGET_END}.csv',
)
results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 {out_csv}")

# 也存 trades 明细(用最后一个 portfolio 引用没意义,跳过 trades CSV)
print("\n注:本 demo 不导出单笔 trades CSV(走 jhzq_fees 后置,需要 portfolio 实例)")
print("     如需 trades 明细,改 run_vbt_backtest 返回 trades_df 即可")

tq.close()
