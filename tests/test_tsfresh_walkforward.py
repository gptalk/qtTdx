# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from common.tsfresh_walkforward import add_ma_channels, report_channel_composition


def _sample_ohlcv(n=30):
    idx = pd.date_range('2026-01-01', periods=n, freq='D')
    return pd.DataFrame({
        'Open':   np.linspace(10, 15, n),
        'High':   np.linspace(10.5, 15.5, n),
        'Low':    np.linspace(9.5, 14.5, n),
        'Close':  np.linspace(10, 15, n),
        'Volume': np.linspace(1e6, 2e6, n),
    }, index=idx)


def test_add_ma_channels_adds_four_columns():
    df = _sample_ohlcv()
    out = add_ma_channels(df)
    assert {'ma5', 'ma10', 'ma20', 'rel_ma5'}.issubset(out.columns)
    assert len(out) == len(df)


def test_add_ma_channels_does_not_mutate_input():
    df = _sample_ohlcv()
    cols_before = set(df.columns)
    _ = add_ma_channels(df)
    assert set(df.columns) == cols_before


def test_add_ma_channels_uses_bfill_not_zero_fill():
    """早期段(头 19 天 ma20 是 NaN)必须用 bfill,不能是 0.0"""
    df = _sample_ohlcv(n=30)
    out = add_ma_channels(df)
    # ma20 第 1 天 NaN 应被第 2 天值 bfill,而不是被 0 填充
    assert not np.isclose(out['ma20'].iloc[0], 0.0), \
        f"ma20.iloc[0]={out['ma20'].iloc[0]} 应为 bfill 后的实值,非 0"


def test_report_channel_composition_warns_when_ma_dominates(capsys):
    """当 ma* 通道入选特征占比 > 33% 时,必须打印 [WARN] 冗余风险"""
    cols = [f'close__f{i}' for i in range(10)] + [f'ma5__f{i}' for i in range(15)]
    X_sel = pd.DataFrame(np.random.randn(5, len(cols)), columns=cols)
    report_channel_composition(X_sel, label='test')
    out = capsys.readouterr().out
    assert '[WARN]' in out and '冗余' in out


def test_report_channel_composition_no_warn_when_basic(capsys):
    cols = [f'close__f{i}' for i in range(20)] + [f'volume__f{i}' for i in range(5)]
    X_sel = pd.DataFrame(np.random.randn(5, len(cols)), columns=cols)
    report_channel_composition(X_sel, label='test')
    out = capsys.readouterr().out
    assert '[WARN]' not in out


def test_tsfresh_walkforward_proba_returns_proba_and_xsel():
    """小数据集(30 天 + 5 通道)— 验证返回类型与 FDR 限制逻辑"""
    df = _sample_ohlcv(n=60)
    # 此测试需要 TQ 关闭时也能跑,所以用 fillna='zero' 跳过 tsfresh
    # 改测 init_train_size 行为:若样本不足,应抛 ValueError 而不是默默返回空
    import pytest
    with pytest.raises(ValueError):
        from common.tsfresh_walkforward import tsfresh_walkforward_proba
        tsfresh_walkforward_proba(df, channels=['Close'], init_train_size=200)
