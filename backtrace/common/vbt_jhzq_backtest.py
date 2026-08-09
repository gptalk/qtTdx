# -*- coding: utf-8 -*-
"""vbt + jhzq_fees 单次回测 + 格式化工具。

约定(写在本模块 docstring 顶部):
  1. **pf_zero reuse** — 复用 fees=0+slippage=0 的 pf_zero 拿 trades,
     jhzq_fees 后置单独算扣费。前提:策略的 entry/exit 判定逻辑与交易费用无关
     (signal 按价格穿越触发)。若未来新增"预期收益需覆盖手续费才 entry"类策略,
     需单独跑一次有费率 portfolio(原 vbt_combo.py:174-176 caveat 注释)。
  2. **80% 拒单 warning** — 当实际成交笔数 < 信号数 * 0.8 时打印 [WARN],
     原因:MAX_POS_PCT=0.95 + 固定 shares,股价上涨后资金不足导致 vbt 静默拒单
     (原 vbt_combo.py:212-222)。
  3. **friction_loss_pp 符号检查** — zero_friction_ret - net_ret 应恒 ≥ 0,
     负值说明 zero/net_ret 口径不一致或费率 bug(原 vbt_combo.py:234-235)。
"""
import warnings
import numpy as np
import pandas as pd
import vectorbt as vbt

from common import jhzq_fees as F


def compute_shares_per_trade(init_cash, max_pos_pct, init_open):
    """每笔固定股数 = floor(init_cash * max_pos_pct / open0 / 100) * 100。
    返回 0 表示价格/仓位下没有 100 股整手(调用方应跳过该票)。"""
    if not np.isfinite(init_open) or init_open <= 0:
        return 0
    raw = init_cash * max_pos_pct / init_open
    if not np.isfinite(raw) or raw < 100:
        return 0
    return int(np.floor(raw / 100) * 100)


def build_proba_signals(proba, bar_index, *, entry_th, exit_th,
                        shift_for_next_open=True):
    """proba reindex 到 bar_index 上 → 生成 (entries, exits) 布尔 Series,
    shift(1) 视作次日开盘成交(默认开启)。

    边界:aligned 全 NaN → 返回 (全 False, 全 False),不会崩。
    """
    aligned = proba.reindex(bar_index)
    entries = (aligned > entry_th).fillna(False).astype(bool)
    exits = (aligned < exit_th).fillna(False).astype(bool)
    if shift_for_next_open:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            entries = entries.shift(1).fillna(False).astype(bool)
            exits = exits.shift(1).fillna(False).astype(bool)
    return entries, exits


def fmt_money(x):
    """'   1,234.50' 格式;NaN → '          N/A'"""
    return f'{x:>12,.2f}' if pd.notna(x) else '          N/A'


def fmt_pct(x):
    """'   12.30%' 格式;inf → '     inf'"""
    if pd.notna(x) and x != float('inf'):
        return f'{x:>7.2%}'
    return '     inf'


def fmt_pp(x):
    """'   2.5pp' 格式;NaN → '  N/A'"""
    return f'{x:>6.1f}pp' if pd.notna(x) else '  N/A'


def run_vbt_backtest(ohlcv_df, entries, exits, stock_code, *,
                     init_cash=100_000, max_pos_pct=0.95,
                     upon_long_conflict='exit',
                     print_rejection_warning=True):
    """跑 vbt + jhzq_fees 真实扣费的单次回测。

    返回 summary dict(11 列):
      strategy, trades, gross_pnl, total_stamp, total_transfer,
      net_pnl, avg_net_per_trade, net_ret, win_rate, profit_factor,
      zero_friction_ret

    副作用:
      - 当实际成交笔数 < 信号数 * 0.8 时打印 [WARN] 拒单警告
      - 当 friction_loss_pp < 0 时打印 [WARN] 口径不一致警告
    """
    base = {'strategy': stock_code, 'trades': 0,
            'gross_pnl': 0.0, 'total_stamp': 0.0, 'total_transfer': 0.0,
            'net_pnl': 0.0, 'avg_net_per_trade': 0.0,
            'net_ret': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0,
            'zero_friction_ret': 0.0}

    entry_signals = int(entries.sum())
    if entry_signals == 0:
        return base

    init_open = float(ohlcv_df['Open'].iloc[0])
    shares = compute_shares_per_trade(init_cash, max_pos_pct, init_open)
    if shares == 0:
        return base

    close = ohlcv_df['Close']
    open_ = ohlcv_df['Open']

    # ===== A. 零摩擦 portfolio(fees=0, slippage=0)=====
    pf_zero = vbt.Portfolio.from_signals(
        close=close, entries=entries, exits=exits, price=open_,
        init_cash=init_cash, fees=0, slippage=0, freq='D',
        size=shares, size_type='amount', size_granularity=100,
        upon_long_conflict=upon_long_conflict,
    )
    base['zero_friction_ret'] = pf_zero.total_return()

    # ===== B. 复用 pf_zero 的 trades 算 jhzq_fees =====
    # 前提:策略的 entry/exit 判定逻辑与交易费用无关
    # 若未来新增"预期收益需覆盖手续费才 entry"类策略,需单独跑有费率 portfolio
    trades = pf_zero.trades.records_readable
    if len(trades) == 0:
        return base

    summary = F.summary_after_fees(trades, stock_code)
    summary['strategy'] = stock_code
    summary['zero_friction_ret'] = base['zero_friction_ret']

    pnl_col = next(c for c in trades.columns if 'PnL' in c and '扣' not in c)
    wins = (trades[pnl_col] > 0).sum()
    summary['win_rate'] = wins / len(trades) if len(trades) > 0 else 0.0
    summary['profit_factor'] = (
        trades[pnl_col][trades[pnl_col] > 0].sum() /
        abs(trades[pnl_col][trades[pnl_col] < 0].sum())
        if (trades[pnl_col] < 0).sum() > 0 else float('inf')
    )
    summary['net_ret'] = summary['net_pnl'] / init_cash

    # 80% 拒单 warning
    actual = int(summary.get('trades', 0))
    if print_rejection_warning and actual < entry_signals * 0.8:
        print(f'   [WARN] 信号 {entry_signals} 个 → 实际成交 {actual} 笔 '
              f'({(1 - actual / entry_signals):.0%} 被拒)')
        print(f'          可能因 MAX_POS_PCT={max_pos_pct} 时股价上涨后资金不足;')
        print(f'          收益对比会失真,降 MAX_POS_PCT 或加现金补充')

    # friction_loss_pp 符号检查
    friction_loss_pp = (base['zero_friction_ret'] - summary['net_ret']) * 100
    if friction_loss_pp < 0:
        print(f'   [WARN] friction_loss_pp={friction_loss_pp:.1f} 负值,'
              f'检查 zero_friction_ret 与 net_ret 口径是否一致')

    return summary