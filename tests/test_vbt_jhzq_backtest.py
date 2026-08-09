# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pytest

from common.vbt_jhzq_backtest import (
    compute_shares_per_trade,
    build_proba_signals,
    fmt_money, fmt_pct, fmt_pp,
)


def test_compute_shares_basic():
    # 100000 * 0.95 / 30 = 3166.67 → 31 手 → 3100 股
    assert compute_shares_per_trade(100_000, 0.95, 30.0) == 3100


def test_compute_shares_returns_zero_below_one_lot():
    assert compute_shares_per_trade(100_000, 0.95, 100_000.0) == 0


def test_compute_shares_handles_invalid_open():
    assert compute_shares_per_trade(100_000, 0.95, 0.0) == 0
    assert compute_shares_per_trade(100_000, 0.95, float('nan')) == 0


def test_build_proba_signals_basic():
    idx = pd.date_range('2026-01-01', periods=10, freq='D')
    proba = pd.Series([0.4, 0.6, 0.7, 0.3, 0.8, 0.55, 0.2, 0.65, 0.45, 0.75], index=idx)
    bar_index = idx
    entries, exits = build_proba_signals(proba, bar_index, entry_th=0.55, exit_th=0.50)
    # shift(1) → 第 1 天对应 proba 第 0 天 = 0.4 → 不 entry
    # 第 2 天对应 proba 第 1 天 = 0.6 > 0.55 → entry
    assert entries.iloc[2] == True
    assert entries.iloc[0] == False
    # 第 7 天对应 proba 第 6 天 = 0.2 < 0.50 → exit
    assert exits.iloc[7] == True


def test_build_proba_signals_all_nan_returns_false():
    idx = pd.date_range('2026-01-01', periods=5, freq='D')
    proba = pd.Series([np.nan] * 5, index=idx)
    bar_index = idx
    entries, exits = build_proba_signals(proba, bar_index, entry_th=0.55, exit_th=0.50)
    assert not entries.any()
    assert not exits.any()


def test_fmt_money_basic():
    assert fmt_money(1234.5) == '    1,234.50'


def test_fmt_money_handles_nan():
    assert 'N/A' in fmt_money(float('nan'))


def test_fmt_pct_basic():
    out = fmt_pct(0.123)
    assert '%' in out


def test_fmt_pct_handles_inf():
    assert fmt_pct(float('inf')) == '     inf'


def test_fmt_pp_basic():
    out = fmt_pp(2.5)
    assert 'pp' in out


def test_run_vbt_backtest_returns_dict_with_expected_keys():
    """小数据集 — 验证返回 dict 的 schema"""
    from common.vbt_jhzq_backtest import run_vbt_backtest
    idx = pd.date_range('2026-01-01', periods=100, freq='D')
    np.random.seed(42)
    df = pd.DataFrame({
        'Open':   10 + np.cumsum(np.random.randn(100) * 0.1),
        'High':   10 + np.cumsum(np.random.randn(100) * 0.1) + 0.5,
        'Low':    10 + np.cumsum(np.random.randn(100) * 0.1) - 0.5,
        'Close':  10 + np.cumsum(np.random.randn(100) * 0.1),
        'Volume': np.random.uniform(1e6, 2e6, 100),
    }, index=idx)
    entries = pd.Series(False, index=idx)
    exits = pd.Series(False, index=idx)
    entries.iloc[10] = True
    exits.iloc[20] = True
    entries.iloc[40] = True
    exits.iloc[50] = True
    entries.iloc[70] = True
    exits.iloc[80] = True
    summary = run_vbt_backtest(df, entries, exits, 'TEST.SH',
                               init_cash=100_000, max_pos_pct=0.5)
    expected_keys = {'strategy', 'trades', 'gross_pnl', 'total_stamp',
                     'total_transfer', 'net_pnl', 'avg_net_per_trade',
                     'net_ret', 'win_rate', 'profit_factor', 'zero_friction_ret'}
    assert expected_keys.issubset(summary.keys())


def test_run_vbt_backtest_no_signals_returns_zero_summary():
    from common.vbt_jhzq_backtest import run_vbt_backtest
    idx = pd.date_range('2026-01-01', periods=30, freq='D')
    df = pd.DataFrame({'Open': 10.0, 'High': 10.0, 'Low': 10.0,
                       'Close': 10.0, 'Volume': 1e6}, index=idx)
    entries = pd.Series(False, index=idx)
    exits = pd.Series(False, index=idx)
    summary = run_vbt_backtest(df, entries, exits, 'TEST.SH')
    assert summary['trades'] == 0
    assert summary['zero_friction_ret'] == 0.0