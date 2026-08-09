# -*- coding: utf-8 -*-
"""tsfresh + MA5 作独立通道 → vbt + jhzq_fees 真实扣费

核心创新:把 ma5 / ma10 / ma20 + close/ma5 偏离度 也作为通道喂给 tsfresh
对比 3 个 tsfresh 通道方案 + 1 个 MA5 基线
输出:tsfresh_with_ma_<code>_<start>_<end>.csv

⚠️ 已知陷阱:
  1. MA 通道与 Close 结构性共线 — 详见 W.report_channel_composition 自动诊断
  2. 阈值选择偏差 — 0.60 是从同一段 5 年回测网格搜索出的「最优」,二次拟合风险大。
     默认改用 0.55(非拟合中点)
  3. bfill 而非 fillna(0.0) — 由 W.tsfresh_walkforward_proba 默认保证

瘦身后:MA 通道、walk-forward proba、通道构成报告、vbt 回测都走 common 模块。
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
STOCK_CODE   = '688318.SH'
TARGET_START = '20250101'
TARGET_END   = '20251231'
INIT_CASH    = 100_000
MAX_POS_PCT  = 0.95
INIT_TRAIN_SIZE = 200
STEP = 50
EXIT_TSF = 0.50
ENTRY_TSF = 0.55
# ===================================

# ---------- 1. 数据 ----------
print("=" * 70)
df = P.load_ohlcva(STOCK_CODE, verbose=True)
if df is None or len(df) == 0:
    print("❌ 数据缺失"); tq.close(); raise SystemExit(1)
df_demo = df.loc[TARGET_START:TARGET_END].copy()
print(f"\n回测区间:{df_demo.index[0].date()} → {df_demo.index[-1].date()}  ({len(df_demo)} 交易日)")

# ---------- 2. 计算 MA 通道(走 common)----------
df = W.add_ma_channels(df)

# ---------- 3. 2 套通道 proba(bfill 默认)----------
PROBA_CACHE = {}
for key, chs in [
    ('basic',   ['Open', 'High', 'Low', 'Close', 'Volume']),
    ('with_ma', ['Open', 'High', 'Low', 'Close', 'Volume',
                 'ma5', 'ma10', 'ma20', 'rel_ma5']),
]:
    proba, X_sel = W.tsfresh_walkforward_proba(
        df, channels=chs, init_train_size=INIT_TRAIN_SIZE, step=STEP, verbose=True,
    )
    W.report_channel_composition(X_sel, label=f'{key}({len(chs)} 通道)')
    PROBA_CACHE[key] = proba


# ---------- 4. 4 策略 ----------
def build_signals(strategy):
    """返回 (name, entries, exits)"""
    name = strategy['name']
    if strategy['kind'] == 'ma':
        ma = vbt.MA.run(df['Close'], window=5).ma.ffill()
        entries = df['Close'].vbt.crossed_above(ma).shift(1).fillna(False).astype(bool)
        exits = df['Close'].vbt.crossed_below(ma).shift(1).fillna(False).astype(bool)
        return name, entries, exits

    proba = PROBA_CACHE[strategy['proba_key']]
    if strategy['mode'] == 'pure':
        entries, exits = B.build_proba_signals(
            proba, df.index, entry_th=ENTRY_TSF, exit_th=EXIT_TSF,
        )
    elif strategy['mode'] == 'confirmed':
        ma = vbt.MA.run(df['Close'], window=5).ma.ffill()
        ma_entry = df['Close'].vbt.crossed_above(ma).shift(1).fillna(False).astype(bool)
        tsf_entry, _ = B.build_proba_signals(
            proba, df.index, entry_th=ENTRY_TSF, exit_th=EXIT_TSF,
        )
        entries = ma_entry & tsf_entry
        exits = df['Close'].vbt.crossed_below(ma).shift(1).fillna(False).astype(bool)
    return name, entries, exits


STRATEGIES = [
    {'name': 'MA5_baseline',          'kind': 'ma'},
    {'name': 'tsfresh_basic_p60',     'kind': 'tsfresh', 'proba_key': 'basic',   'mode': 'pure'},
    {'name': 'tsfresh_with_ma_p60',   'kind': 'tsfresh', 'proba_key': 'with_ma', 'mode': 'pure'},
    {'name': 'MA5_AND_with_ma_p60',   'kind': 'tsfresh', 'proba_key': 'with_ma', 'mode': 'confirmed'},
]

# ---------- 5. 跑 4 策略 ----------
print("=" * 70)
print("=== 4 策略对比 ===")
print("=" * 70)
results = []
for strat in STRATEGIES:
    name, entries, exits = build_signals(strat)
    summary = B.run_vbt_backtest(
        df, entries, exits, name,
        init_cash=INIT_CASH, max_pos_pct=MAX_POS_PCT,
    )
    results.append(summary)

# ---------- 6. 输出对比表 ----------
results_df = pd.DataFrame(results)
cols = ['strategy', 'trades', 'zero_friction_ret',
        'gross_pnl', 'total_stamp', 'total_transfer',
        'net_pnl', 'net_ret', 'win_rate', 'profit_factor']
results_df = results_df[cols].sort_values('net_pnl', ascending=False).reset_index(drop=True)
results_df['friction_loss_pp'] = (results_df['zero_friction_ret'] - results_df['net_ret']) * 100

print("\n" + "=" * 110)
print("=== 4 策略对比表(basic vs with_ma vs MA5 复合) ===")
print("=" * 110)
print(f"{'策略':<22} {'笔数':>5} {'零摩擦':>8} "
      f"{'净收益':>12} {'净收益率':>8} {'胜率':>6} {'盈亏比':>8} {'摩擦吃损':>10}")
print("-" * 110)
for _, r in results_df.iterrows():
    print(f"{r['strategy']:<22} {int(r['trades']):>5} "
          f"{B.fmt_pct(r['zero_friction_ret'])} "
          f"{B.fmt_money(r['net_pnl'])} {B.fmt_pct(r['net_ret'])} "
          f"{B.fmt_pct(r['win_rate'])} {B.fmt_pct(r['profit_factor'])} "
          f"{B.fmt_pp(r['friction_loss_pp'])}")

# ---------- 7. 保存 ----------
out_csv = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/outputs",
    f'tsfresh_with_ma_grid_{STOCK_CODE.replace(".", "_")}_{TARGET_START}_{TARGET_END}.csv',
)
results_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 {out_csv}")

tq.close()
