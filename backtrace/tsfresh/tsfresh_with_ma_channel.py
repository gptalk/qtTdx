# tsfresh + MA5 作独立通道 → vbt + jhzq_fees 真实扣费
# 核心创新:把 ma5 / ma10 / ma20 + close/ma5 偏离度 也作为通道喂给 tsfresh
# 对比 3 个 tsfresh 通道方案 + 1 个 MA5 基线
# 输出:tsfresh_with_ma_<code>_<start>_<end>.csv
# 用法:验证把"均线 / 偏离度"当额外 tsfresh 通道,能否挑出 MA5 看不出的反转点
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
STOCK_CODE   = '688318.SH'
TARGET_START = '20250101'
TARGET_END   = '20251231'
INIT_CASH    = 100_000
MAX_POS_PCT  = 0.95

INIT_TRAIN_SIZE = 200   # walk-forward 初始训练窗口
STEP = 50               # 重训步长
EXIT_TSF = 0.50
ENTRY_TSF = 0.60        # 当前最佳阈值
# ===================================

# ---------- 1. 数据 ----------
print("=" * 70)
df = P.load_ohlcva(STOCK_CODE, verbose=True)
if df is None or len(df) == 0:
    print("❌ 数据缺失"); tq.close(); raise SystemExit(1)
df_demo = df.loc[TARGET_START:TARGET_END].copy()
print(f"\n回测区间:{df_demo.index[0].date()} → {df_demo.index[-1].date()}  ({len(df_demo)} 交易日)")

# 计算 MA 通道
for w in [5, 10, 20]:
    df[f'ma{w}'] = vbt.MA.run(df['Close'], window=w).ma.ffill()
# close 相对 ma5 的偏离度(标准化)
df['rel_ma5'] = (df['Close'] - df['ma5']) / df['ma5'].replace(0, np.nan)

close_full = df['Close']
open_full  = df['Open']

# ---------- 2. 跑两组 tsfresh 通道方案 ----------
PROBA_CACHE = {}   # key=channels -> pd.Series(proba, index=date)


def build_tsfresh_proba(channels, label):
    """
    给定通道列表,跑全量 tsfresh + walk-forward。
    返回 proba Series(每个窗口结束日 1 个值)。
    """
    print(f"\n--- {label} ({len(channels)} 通道: {channels}) ---")
    df_fill = df[channels].copy().fillna(0.0)   # tsfresh 不允许 NaN(ma5 在头几天为 0/NaN)
    long_df = P.to_long_format(df_fill, channels=channels, id_value=STOCK_CODE)
    X_all = P.extract_window_features(long_df, use_kind=True, verbose=False)
    y_all, X_all = P.make_labels(X_all, df['Close'].values, verbose=False)
    print(f"   样本 {len(y_all)} 个  |  正样本 {y_all.mean():.1%}  |  特征 {X_all.shape[1]} 列")

    X_sel = P.select_relevant(X_all, y_all, verbose=False)
    if X_sel.shape[1] == 0:
        print(f"   [WARN] FDR=0,用全量 {X_all.shape[1]} 特征")
        X_sel = X_all

    # walk-forward
    print(f"   walk-forward (initial={INIT_TRAIN_SIZE}, retrain every {STEP})...")
    date_index = pd.DatetimeIndex(df.index)
    proba_records = []
    scaler_w = clf_w = None
    for pos, idx in enumerate(X_sel.index):
        end_t = idx[1]
        if end_t >= len(date_index):
            continue
        if pos < INIT_TRAIN_SIZE:
            proba_records.append((date_index[end_t], np.nan))
            continue
        if pos == INIT_TRAIN_SIZE or (pos - INIT_TRAIN_SIZE) % STEP == 0:
            scaler_w, clf_w = P.fit_logreg(X_sel.iloc[:pos], y_all.iloc[:pos], verbose=False)
        p = float(clf_w.predict_proba(scaler_w.transform(X_sel.iloc[[pos]].values))[0, 1])
        proba_records.append((date_index[end_t], p))

    proba = pd.Series([v for _, v in proba_records],
                      index=pd.DatetimeIndex([d for d, _ in proba_records]),
                      name='proba').sort_index()
    proba = proba[~proba.index.duplicated(keep='last')].dropna()
    print(f"   proba {len(proba)} 个  |  mean={proba.mean():.3f}")
    return proba


# 5 通道原始
PROBA_CACHE['basic'] = build_tsfresh_proba(
    channels=['Open', 'High', 'Low', 'Close', 'Volume'],
    label='basic(5 通道 OHLCV)',
)

# 8 通道(原始 + MA5/10/20 + 偏离度)
PROBA_CACHE['with_ma'] = build_tsfresh_proba(
    channels=['Open', 'High', 'Low', 'Close', 'Volume',
              'ma5', 'ma10', 'ma20', 'rel_ma5'],
    label='with_ma(8 通道 = 5 OHLCV + MA5/10/20 + rel_ma5)',
)


# ---------- 3. 4 组对比策略 ----------
shares_per_trade = int(np.floor(INIT_CASH * MAX_POS_PCT / float(open_full.iloc[0]) / 100) * 100)
print(f"\n每笔 {shares_per_trade} 股  (≈ {shares_per_trade * float(open_full.iloc[0]):,.0f} 元)\n")


def build_signals(strategy):
    name = strategy['name']
    if strategy['kind'] == 'ma':
        ma = vbt.MA.run(close_full, window=5).ma.ffill()
        entries = close_full.vbt.crossed_above(ma).shift(1).fillna(False).astype(bool)
        exits   = close_full.vbt.crossed_below(ma).shift(1).fillna(False).astype(bool)
        return name, entries, exits

    if strategy['kind'] == 'tsfresh':
        proba = PROBA_CACHE[strategy['proba_key']]
        aligned = proba.reindex(close_full.index)
        if strategy['mode'] == 'pure':
            entries = (aligned > ENTRY_TSF).shift(1).fillna(False).astype(bool)
            exits   = (aligned < EXIT_TSF).shift(1).fillna(False).astype(bool)
        elif strategy['mode'] == 'confirmed':
            # MA5 触发 + tsfresh 确认(双条件 AND)
            ma = vbt.MA.run(close_full, window=5).ma.ffill()
            ma_entry = close_full.vbt.crossed_above(ma).shift(1).fillna(False).astype(bool)
            tsf_entry = (aligned > ENTRY_TSF).shift(1).fillna(False).astype(bool)
            entries = ma_entry & tsf_entry
            exits   = close_full.vbt.crossed_below(ma).shift(1).fillna(False).astype(bool)
        return name, entries, exits
    raise ValueError(strategy['kind'])


def run_one(strategy):
    name, entries, exits = build_signals(strategy)
    if entries.sum() == 0:
        return {'strategy': name, 'trades': 0,
                'gross_pnl': 0.0, 'total_stamp': 0.0, 'total_transfer': 0.0,
                'net_pnl': 0.0, 'net_ret': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'zero_friction_ret': 0.0}

    pf_zero = vbt.Portfolio.from_signals(
        close=close_full, entries=entries, exits=exits, price=open_full,
        init_cash=INIT_CASH, fees=0, slippage=0, freq='D',
        size=shares_per_trade, size_type='amount', size_granularity=100,
        upon_long_conflict='exit',
    )
    trades = pf_zero.trades.records_readable
    zero_ret = pf_zero.total_return()
    if len(trades) == 0:
        return {'strategy': name, 'trades': 0,
                'gross_pnl': 0.0, 'total_stamp': 0.0, 'total_transfer': 0.0,
                'net_pnl': 0.0, 'net_ret': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'zero_friction_ret': zero_ret}

    summary = F.summary_after_fees(trades, STOCK_CODE)
    summary['strategy'] = name
    summary['zero_friction_ret'] = zero_ret
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


STRATEGIES = [
    {'name': 'MA5_baseline',          'kind': 'ma'},
    {'name': 'tsfresh_basic_p60',     'kind': 'tsfresh', 'proba_key': 'basic',   'mode': 'pure'},
    {'name': 'tsfresh_with_ma_p60',   'kind': 'tsfresh', 'proba_key': 'with_ma', 'mode': 'pure'},
    {'name': 'MA5_AND_with_ma_p60',   'kind': 'tsfresh', 'proba_key': 'with_ma', 'mode': 'confirmed'},
]

print("=" * 70)
print("=== 4 策略对比 ===")
print("=" * 70)
for strat in STRATEGIES:
    name, entries, _ = build_signals(strat)
    print(f"  [{strat['name']}] entry 信号数: {int(entries.sum())}")

results = [run_one(s) for s in STRATEGIES]
results_df = pd.DataFrame(results)
cols = ['strategy', 'trades', 'zero_friction_ret',
        'gross_pnl', 'total_stamp', 'total_transfer',
        'net_pnl', 'net_ret', 'win_rate', 'profit_factor']
results_df = results_df[cols].sort_values('net_pnl', ascending=False).reset_index(drop=True)
results_df['friction_loss_pp'] = (results_df['zero_friction_ret'] - results_df['net_ret']) * 100


def fmt_pct(x):
    return f"{x:>7.2%}" if pd.notna(x) and x != float('inf') else "     inf"
def fmt_money(x):
    return f"{x:>12,.2f}" if pd.notna(x) else "          N/A"
def fmt_pp(x):
    return f"{x:>6.1f}pp" if pd.notna(x) else "  N/A"

print("\n" + "=" * 110)
print("=== 4 策略对比表(basic vs with_ma vs MA5 复合) ===")
print("=" * 110)
print(f"{'策略':<22} {'笔数':>5} {'零摩擦':>8} "
      f"{'净收益':>12} {'净收益率':>8} {'胜率':>6} {'盈亏比':>8} {'摩擦吃损':>10}")
print("-" * 110)
for _, r in results_df.iterrows():
    print(f"{r['strategy']:<22} {int(r['trades']):>5} "
          f"{fmt_pct(r['zero_friction_ret'])} "
          f"{fmt_money(r['net_pnl'])} {fmt_pct(r['net_ret'])} "
          f"{fmt_pct(r['win_rate'])} {fmt_pct(r['profit_factor'])} "
          f"{fmt_pp(r['friction_loss_pp'])}")

# 保存
out_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/outputs",
                       f'tsfresh_with_ma_grid_{STOCK_CODE.replace(".", "_")}_{TARGET_START}_{TARGET_END}.csv')
results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 {out_csv}")

tq.close()
