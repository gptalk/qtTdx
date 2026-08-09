# -*- coding: utf-8 -*-
"""tsfresh + VectorBT 集成 demo

4 组合网格:MA5 基线 + tsfresh 三个阈值,全部走 jhzq_fees 真实扣费
输出:tsfresh_vbt_grid_<code>_<start>_<end>.csv(每策略 trades + 净 PnL)
用法:验证 tsfresh 信号当 vbt entry/exit 是否比纯 MA5 基线多赚(扣费后)

瘦身后:walk-forward proba、vbt 回测、80% 拒单 warning 都走 common 模块。
"""
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
from common import tsfresh_walkforward as W
from common import vbt_jhzq_backtest as B
from common import jhzq_fees as F

tq.initialize(__file__)

# ============== 配置 ==============
STOCK_CODE   = '688318.SH'             # 沪市股,验证 SH 过户费分支
TARGET_START = '20250101'
TARGET_END   = '20251231'
WINDOW_MA    = 5
INIT_CASH    = 100_000
MAX_POS_PCT  = 0.95
INIT_TRAIN_SIZE = 200
STEP            = 50

# 4 组合网格
STRATEGIES = [
    {'name': 'MA5_baseline', 'kind': 'ma',     'entry_th': 0,  'exit_th': 0},
    {'name': 'tsfresh_p50',  'kind': 'proba',  'entry_th': 0.50, 'exit_th': 0.50},
    {'name': 'tsfresh_p55',  'kind': 'proba',  'entry_th': 0.55, 'exit_th': 0.50},
    {'name': 'tsfresh_p60',  'kind': 'proba',  'entry_th': 0.60, 'exit_th': 0.50},
]
# ===================================

# ---------- 1. 数据 ----------
print("=" * 70)
df = P.load_ohlcva(STOCK_CODE, verbose=True)
if df is None or len(df) == 0:
    print("❌ 数据缺失"); tq.close(); raise SystemExit(1)

df_demo = df.loc[TARGET_START:TARGET_END].copy()
print(f"\n回测区间:{df_demo.index[0].date()} → {df_demo.index[-1].date()}  ({len(df_demo)} 交易日)")

# ---------- 2. walk-forward proba(bfill 默认)----------
print("\n" + "=" * 70)
print("tsfresh + walk-forward proba...")
proba, X_sel = W.tsfresh_walkforward_proba(
    df, channels=['Open', 'High', 'Low', 'Close', 'Volume'],
    init_train_size=INIT_TRAIN_SIZE, step=STEP, verbose=True,
)
W.report_channel_composition(X_sel, label='OHLCV')

# ---------- 3. 4 套 entry/exit ----------
def build_signals(strategy):
    name = strategy['name']
    if strategy['kind'] == 'ma':
        ma = vbt.MA.run(df['Close'], window=WINDOW_MA).ma.ffill()
        entries = df['Close'].vbt.crossed_above(ma).shift(1).fillna(False).astype(bool)
        exits = df['Close'].vbt.crossed_below(ma).shift(1).fillna(False).astype(bool)
        return name, entries, exits
    entries, exits = B.build_proba_signals(
        proba, df.index,
        entry_th=strategy['entry_th'], exit_th=strategy['exit_th'],
    )
    return name, entries, exits


# ---------- 4. 跑 4 组合网格 ----------
print("=" * 70)
print("=== 4 组合网格回测 ===")
print("=" * 70)

results = []
for strat in STRATEGIES:
    name, entries, exits = build_signals(strat)
    summary = B.run_vbt_backtest(
        df, entries, exits, name,
        init_cash=INIT_CASH, max_pos_pct=MAX_POS_PCT,
    )
    results.append(summary)

# ---------- 5. 输出对比表 ----------
results_df = pd.DataFrame(results)
cols = ['strategy', 'trades', 'zero_friction_ret',
        'gross_pnl', 'total_stamp', 'total_transfer',
        'net_pnl', 'net_ret', 'win_rate', 'profit_factor', 'avg_net_per_trade']
results_df = results_df[cols].sort_values('net_pnl', ascending=False).reset_index(drop=True)
results_df['friction_loss_pp'] = (results_df['zero_friction_ret'] - results_df['net_ret']) * 100

print("\n" + "=" * 90)
print("=== 4 组合对比表 ===")
print("=" * 90)
print(f"{'策略':<16} {'笔数':>5} {'零摩擦':>8} {'毛收益':>12} "
      f"{'印花税':>8} {'过户费':>8} {'净收益':>12} {'净收益率':>8} "
      f"{'胜率':>6} {'盈亏比':>7} {'摩擦吃损':>8}")
print("-" * 120)
for _, r in results_df.iterrows():
    print(f"{r['strategy']:<16} {int(r['trades']):>5} "
          f"{B.fmt_pct(r['zero_friction_ret'])} "
          f"{B.fmt_money(r['gross_pnl'])} "
          f"{-r['total_stamp']:>8.2f} {-r['total_transfer']:>8.2f} "
          f"{B.fmt_money(r['net_pnl'])} {B.fmt_pct(r['net_ret'])} "
          f"{B.fmt_pct(r['win_rate'])} {B.fmt_pct(r['profit_factor'])} {B.fmt_pp(r['friction_loss_pp'])}")

# ---------- 6. 保存 ----------
out_csv = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/outputs",
    f'tsfresh_vbt_grid_{STOCK_CODE.replace(".", "_")}_{TARGET_START}_{TARGET_END}.csv',
)
results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 {out_csv}")

tq.close()