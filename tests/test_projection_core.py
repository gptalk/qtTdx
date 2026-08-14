# -*- coding: utf-8 -*-
"""backtrace/projection/_projection_core.py 单元测试 — 覆盖 lag=0 / lag=1 双路径。"""
import sys, os
import numpy as np
import pandas as pd
import pytest

# 与脚本同一套导入约定:backtrace/ 进 sys.path
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTRACE = os.path.join(REPO, 'backtrace')
PROJECTION = os.path.join(BACKTRACE, 'projection')
if BACKTRACE not in sys.path:
    sys.path.insert(0, BACKTRACE)

from projection._projection_core import compute_vectors, build_result_df, load_pair


def _make_pair(n=10):
    """造一对 (stock_df, index_df),index=stock 索引,数据有差异。"""
    idx = pd.date_range('2026-07-01', periods=n, freq='D')
    index_df = pd.DataFrame({
        'Volume': np.linspace(1e7, 1.5e7, n),
        'Amount': np.linspace(1e11, 1.5e11, n),
        'Close':  np.linspace(3000, 3200, n),
    }, index=idx)
    stock_df = pd.DataFrame({
        'Volume': np.linspace(2e6, 3e6, n),
        'Amount': np.linspace(2e10, 3e10, n),
        'Close':  np.linspace(20, 25, n),
    }, index=idx)
    return stock_df, index_df


def _add_prev(df, vol_prev, amt_prev):
    out = df.copy()
    out['Volume_prev'] = vol_prev
    out['Amount_prev'] = amt_prev
    return out


def test_compute_vectors_lag0_returns_2_columns():
    """lag=0 默认: 输出向量 shape=(T, 2)。"""
    stock_df, index_df = _make_pair(10)
    v_ix, v_st, v_ix_n, v_st_n, _ = compute_vectors(stock_df, index_df, '000001', '002475')
    assert v_ix.shape == (10, 2)
    assert v_st.shape == (10, 2)
    assert v_ix_n.shape == (10, 2)
    assert v_st_n.shape == (10, 2)


def test_compute_vectors_lag1_returns_4_columns():
    """lag=1: 输出向量 shape=(T, 4),顺序 Vol_t, Amt_t, Vol_prev, Amt_prev。"""
    stock_df, index_df = _make_pair(10)
    stock_df = _add_prev(stock_df, np.linspace(1.9e6, 2.8e6, 10), np.linspace(1.9e10, 2.8e10, 10))
    index_df = _add_prev(index_df, np.linspace(0.95e7, 1.45e7, 10), np.linspace(0.95e11, 1.45e11, 10))
    v_ix, v_st, v_ix_n, v_st_n, _ = compute_vectors(stock_df, index_df, '000001', '002475', lag=1)
    assert v_ix.shape == (10, 4)
    assert v_st.shape == (10, 4)
    # 前 2 列 = 今日 (与 lag=0 一致)
    np.testing.assert_array_equal(v_ix[:, :2], index_df[['Volume', 'Amount']].values)
    # 后 2 列 = 昨日
    np.testing.assert_array_equal(v_ix[:, 2:], index_df[['Volume_prev', 'Amount_prev']].values)


def test_compute_vectors_norm_range_in_unit_interval():
    """归一化后每列在 [0, 1]。"""
    stock_df, index_df = _make_pair(10)
    stock_df = _add_prev(stock_df, np.linspace(1.9e6, 2.8e6, 10), np.linspace(1.9e10, 2.8e10, 10))
    index_df = _add_prev(index_df, np.linspace(0.95e7, 1.45e7, 10), np.linspace(0.95e11, 1.45e11, 10))
    _, _, v_ix_n, v_st_n, _ = compute_vectors(stock_df, index_df, '000001', '002475', lag=1)
    assert (0.0 <= v_ix_n).all() and (v_ix_n <= 1.0).all()
    assert (0.0 <= v_st_n).all() and (v_st_n <= 1.0).all()


def test_compute_vectors_norm_params_lists_four_ranges_at_lag1():
    """lag=1 时 norm_params 字符串包含 4 个范围(每个 tag × 2 个列 + 2 个 prev 列)。"""
    stock_df, index_df = _make_pair(10)
    stock_df = _add_prev(stock_df, np.linspace(1.9e6, 2.8e6, 10), np.linspace(1.9e10, 2.8e10, 10))
    index_df = _add_prev(index_df, np.linspace(0.95e7, 1.45e7, 10), np.linspace(0.95e11, 1.45e11, 10))
    _, _, _, _, params = compute_vectors(stock_df, index_df, '000001', '002475', lag=1)
    # 期望含 4 个 vol_/amt_ 范围 + 4 个 vol_prev/amt_prev 范围 → 8 个 "["
    assert params.count('[') == 8


def test_build_result_df_lag0_returns_19_columns():
    """回归: lag=0 保持 19 列。"""
    n = 10
    common_idx = pd.date_range('2026-07-01', periods=n, freq='D')
    v_ix = np.random.rand(n, 2)
    v_st = np.random.rand(n, 2)
    proj = {'projections': np.zeros((n, 2)), 'residuals': np.zeros((n, 2)),
            'dot_after': np.zeros(n), 'proj_coeffs': np.zeros(n),
            'proj_mags': np.zeros(n), 'proj_prices': np.zeros(n), 'resi_prices': np.zeros(n)}
    df = build_result_df(
        common_idx, v_ix, v_st, v_ix, v_st,
        proj['projections'], proj['residuals'], proj['dot_after'],
        proj['proj_coeffs'], proj['proj_mags'], proj['proj_prices'], proj['resi_prices'],
        "vol_000001:[1,2] amt_000001:[1,2] vol_002475:[1,2] amt_002475:[1,2]",
        '000001', '002475',
    )
    assert df.shape == (n, 19)


def test_build_result_df_lag1_returns_27_columns():
    """lag=1: 增加 8 个 prev 列 = 27 列。"""
    n = 10
    common_idx = pd.date_range('2026-07-01', periods=n, freq='D')
    v_ix = np.random.rand(n, 4)
    v_st = np.random.rand(n, 4)
    proj = {'projections': np.zeros((n, 4)), 'residuals': np.zeros((n, 4)),
            'dot_after': np.zeros(n), 'proj_coeffs': np.zeros(n),
            'proj_mags': np.zeros(n), 'proj_prices': np.zeros(n), 'resi_prices': np.zeros(n)}
    df = build_result_df(
        common_idx, v_ix, v_st, v_ix, v_st,
        proj['projections'], proj['residuals'], proj['dot_after'],
        proj['proj_coeffs'], proj['proj_mags'], proj['proj_prices'], proj['resi_prices'],
        "vol_000001:[1,2] amt_000001:[1,2] vol_002475:[1,2] amt_002475:[1,2] "
        "vol_prev_000001:[1,2] amt_prev_000001:[1,2] vol_prev_002475:[1,2] amt_prev_002475:[1,2]",
        '000001', '002475', lag=1,
    )
    assert df.shape == (n, 27)
    # 检查 prev_raw + prev_norm 4 对列都在
    for tag in ('000001', '002475'):
        for kind in ('Vol', 'Amt'):
            assert f'{kind}_{tag}_prev_raw' in df.columns
            assert f'{kind}_{tag}_prev_norm' in df.columns


def test_build_result_df_lag1_preserves_projection_columns_after_prev_block():
    """lag=1 时,Proj_Vol 仍是第 18 列(prev 块在 10-17)。"""
    n = 5
    common_idx = pd.date_range('2026-07-01', periods=n, freq='D')
    v_ix = np.random.rand(n, 4)
    v_st = np.random.rand(n, 4)
    proj = {'projections': np.zeros((n, 4)), 'residuals': np.zeros((n, 4)),
            'dot_after': np.zeros(n), 'proj_coeffs': np.zeros(n),
            'proj_mags': np.zeros(n), 'proj_prices': np.zeros(n), 'resi_prices': np.zeros(n)}
    df = build_result_df(
        common_idx, v_ix, v_st, v_ix, v_st,
        proj['projections'], proj['residuals'], proj['dot_after'],
        proj['proj_coeffs'], proj['proj_mags'], proj['proj_prices'], proj['resi_prices'],
        "x", '000001', '002475', lag=1,
    )
    cols = list(df.columns)
    assert cols[0] == 'Date'
    assert cols[17] == 'Proj_Vol'
    assert cols[18] == 'Proj_Amt'


def test_build_result_df_lag1_resi_price_present():
    """find_resi_positive.py 依赖 Resi_Price 列,lag=1 必须保留。"""
    n = 5
    common_idx = pd.date_range('2026-07-01', periods=n, freq='D')
    v_ix = np.random.rand(n, 4)
    v_st = np.random.rand(n, 4)
    proj = {'projections': np.zeros((n, 4)), 'residuals': np.zeros((n, 4)),
            'dot_after': np.zeros(n), 'proj_coeffs': np.zeros(n),
            'proj_mags': np.zeros(n), 'proj_prices': np.zeros(n), 'resi_prices': np.zeros(n)}
    df = build_result_df(
        common_idx, v_ix, v_st, v_ix, v_st,
        proj['projections'], proj['residuals'], proj['dot_after'],
        proj['proj_coeffs'], proj['proj_mags'], proj['proj_prices'], proj['resi_prices'],
        "x", '000001', '002475', lag=1,
    )
    assert 'Resi_Price' in df.columns


class _FakePipeline:
    """最小化的 tsfresh_pipeline 替身:返回内存中的 DataFrame,不读 data/。"""

    def __init__(self, df_by_code):
        self._df = df_by_code

    def load_ohlcva(self, code, use_tq=False, verbose=False):
        return self._df.get(code)


def _make_ohlcv(n, base_vol, base_amt):
    idx = pd.date_range('2026-07-01', periods=n, freq='D')
    return pd.DataFrame({
        'Volume': np.linspace(base_vol, base_vol * 1.5, n),
        'Amount': np.linspace(base_amt, base_amt * 1.5, n),
        'Close':  np.linspace(100, 110, n),
    }, index=idx)


def test_load_pair_lag0_does_not_add_prev_columns():
    """lag=0: 返回的 stock_df / index_df 不含 Volume_prev。"""
    df = _make_ohlcv(10, 1e6, 1e10)
    pipe = _FakePipeline({'000001.SH': df, '002475.SZ': df})
    out = load_pair('002475.SZ', days=10, pipeline=pipe, index_code='000001.SH')
    assert 'Volume_prev' not in out['stock_df'].columns
    assert 'Volume_prev' not in out['index_df'].columns
    assert len(out['common_idx']) == 10


def test_load_pair_lag1_adds_prev_columns_and_drops_first_row():
    """lag=1: 返回的 df 含 prev 列,common_idx 比原始少 1 行(首行 prev=NaN 被 dropna)。"""
    df = _make_ohlcv(10, 1e6, 1e10)
    pipe = _FakePipeline({'000001.SH': df, '002475.SZ': df})
    out = load_pair('002475.SZ', days=10, pipeline=pipe, index_code='000001.SH', lag=1)
    assert 'Volume_prev' in out['stock_df'].columns
    assert 'Amount_prev' in out['stock_df'].columns
    assert 'Volume_prev' in out['index_df'].columns
    assert 'Amount_prev' in out['index_df'].columns
    assert len(out['common_idx']) == 9, "首行 prev=NaN 应被 dropna 丢弃"
    # dropna 后第 0 行(now index 2026-07-02)的 Volume_prev = 原始第 0 行 Volume(2026-07-01)
    assert out['index_df']['Volume_prev'].iloc[0] == df['Volume'].iloc[0]


def test_load_pair_lag1_raises_when_data_too_short():
    """数据 < 2 行时 lag=1 必须 ValueError。"""
    df = _make_ohlcv(1, 1e6, 1e10)
    pipe = _FakePipeline({'000001.SH': df, '002475.SZ': df})
    with pytest.raises(ValueError, match="≥2"):
        load_pair('002475.SZ', days=10, pipeline=pipe, index_code='000001.SH', lag=1)

