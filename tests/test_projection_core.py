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

from projection._projection_core import (
    compute_vectors,
    compute_projections,
    build_result_df,
    load_pair,
    compute_movement_projection,
    build_movement_result_df,
    build_movement_intermediate_df,
)


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


def _make_proj_dict(n, k):
    """造一个 compute_projections 风格的返回 dict,k=2 或 4 表示向量维度。"""
    return {
        'projections': np.zeros((n, k)), 'residuals': np.zeros((n, k)),
        'dot_after': np.zeros(n), 'proj_coeffs': np.zeros(n),
        'proj_mags': np.zeros(n), 'proj_prices': np.zeros(n),
        # 8 维度框架新增(state_)
        'state_stock_mag': np.zeros(n), 'state_index_mag': np.zeros(n),
        'state_relative_move': np.zeros(n),
    }


def test_build_result_df_lag0_returns_21_columns():
    """回归: lag=0 → 21 列(State_ 前缀 + 8 维度幅度量,2026-08-16 删除
    State_Resi_Price 后从 22 列降为 21 列)。"""
    n = 10
    common_idx = pd.date_range('2026-07-01', periods=n, freq='D')
    v_ix = np.random.rand(n, 2)
    v_st = np.random.rand(n, 2)
    proj = _make_proj_dict(n, 2)
    df = build_result_df(
        common_idx, v_ix, v_st, v_ix, v_st,
        proj['projections'], proj['residuals'], proj['dot_after'],
        proj['proj_coeffs'], proj['proj_mags'], proj['proj_prices'],
        proj['state_stock_mag'], proj['state_index_mag'], proj['state_relative_move'],
        "vol_000001:[1,2] amt_000001:[1,2] vol_002475:[1,2] amt_002475:[1,2]",
        '000001', '002475',
    )
    assert df.shape == (n, 21)


def test_build_result_df_lag1_returns_29_columns():
    """lag=1: 21 + 8 个 prev 列 = 29 列。"""
    n = 10
    common_idx = pd.date_range('2026-07-01', periods=n, freq='D')
    v_ix = np.random.rand(n, 4)
    v_st = np.random.rand(n, 4)
    proj = _make_proj_dict(n, 4)
    df = build_result_df(
        common_idx, v_ix, v_st, v_ix, v_st,
        proj['projections'], proj['residuals'], proj['dot_after'],
        proj['proj_coeffs'], proj['proj_mags'], proj['proj_prices'],
        proj['state_stock_mag'], proj['state_index_mag'], proj['state_relative_move'],
        "vol_000001:[1,2] amt_000001:[1,2] vol_002475:[1,2] amt_002475:[1,2] "
        "vol_prev_000001:[1,2] amt_prev_000001:[1,2] vol_prev_002475:[1,2] amt_prev_002475:[1,2]",
        '000001', '002475', lag=1,
    )
    assert df.shape == (n, 29)
    # 检查 prev_raw + prev_norm 4 对列都在
    for tag in ('000001', '002475'):
        for kind in ('Vol', 'Amt'):
            assert f'{kind}_{tag}_prev_raw' in df.columns
            assert f'{kind}_{tag}_prev_norm' in df.columns


def test_build_result_df_lag1_preserves_projection_columns_after_prev_block():
    """lag=1 时,3 个 State_*_Magnitude/Relative 落在 idx 17/18/19,
    State_Proj_Vol/Amt 在 idx 20/21(prev 块在 9-16)。
    2026-08-16 删除 State_Resi_Price(原 idx 22)后,投影 5 列和 resi 2 列
    自然前移到 idx 17-21 / 22-23。
    """
    n = 5
    common_idx = pd.date_range('2026-07-01', periods=n, freq='D')
    v_ix = np.random.rand(n, 4)
    v_st = np.random.rand(n, 4)
    proj = _make_proj_dict(n, 4)
    df = build_result_df(
        common_idx, v_ix, v_st, v_ix, v_st,
        proj['projections'], proj['residuals'], proj['dot_after'],
        proj['proj_coeffs'], proj['proj_mags'], proj['proj_prices'],
        proj['state_stock_mag'], proj['state_index_mag'], proj['state_relative_move'],
        "x", '000001', '002475', lag=1,
    )
    cols = list(df.columns)
    assert cols[0] == 'Date'
    # 8 维度幅度量在 prev 块后(9-16)
    assert cols[17] == 'State_Stock_Magnitude'
    assert cols[18] == 'State_Index_Magnitude'
    assert cols[19] == 'State_Relative_Move'
    # 然后是 State_Proj_Vol/Amt
    assert cols[20] == 'State_Proj_Vol'
    assert cols[21] == 'State_Proj_Amt'
    # 残差侧只剩 Vol/Amt(无 State_Resi_Price)
    assert cols[22] == 'State_Resi_Vol'
    assert cols[23] == 'State_Resi_Amt'
    # State_Resi_Price 必须不存在
    assert 'State_Resi_Price' not in df.columns


class _FakePipeline:
    """最小化的 tsfresh_pipeline 替身:返回内存中的 DataFrame,不读 data/。"""

    def __init__(self, df_by_code):
        self._df = df_by_code

    def load_ohlcva(self, code, use_tq=False, verbose=False, period='daily'):
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


# ========================== compute_movement_projection ==========================

def test_movement_basic_projection_along_index_direction():
    """claude 建议里的算例验证: stock 运动 (30,60) 完全落在 index 运动 (50,100) 上 → β=0.6, proj=(30,60), residual=(0,0)"""
    idx = pd.date_range('2026-07-01', periods=3, freq='D')
    stock_df = pd.DataFrame({
        'Volume': [100.0, 130.0, 150.0],   # Δ = [30, 20]
        'Amount': [200.0, 260.0, 300.0],   # Δ = [60, 40]
    }, index=idx)
    index_df = pd.DataFrame({
        'Volume': [1000.0, 1050.0, 1100.0],  # Δ = [50, 50]
        'Amount': [2000.0, 2100.0, 2200.0],  # Δ = [100, 100]
    }, index=idx)
    mv = compute_movement_projection(stock_df, index_df)
    # 丢首行后剩 2 行
    assert mv['stock_move'].shape == (2, 2)
    assert mv['index_move'].shape == (2, 2)
    # 第一日: stock Δ=(30,60), index Δ=(50,100)
    # β = (30·50 + 60·100) / (50² + 100²) = 7500 / 12500 = 0.6
    np.testing.assert_allclose(mv['proj_coeff'][0], 0.6, rtol=1e-9)
    np.testing.assert_allclose(mv['proj'][0], [30.0, 60.0], rtol=1e-9)
    np.testing.assert_allclose(mv['residual'][0], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(mv['dot_after'][0], 0.0, atol=1e-9)


def test_movement_residual_orthogonal_to_index():
    """claude 算例 6: stock Δ=(40,50), index Δ=(50,100) → β=0.56, proj=(28,56), residual=(12,-6), residual·index=0"""
    idx = pd.date_range('2026-07-01', periods=2, freq='D')
    stock_df = pd.DataFrame({
        'Volume': [100.0, 140.0],   # Δ = 40
        'Amount': [200.0, 250.0],   # Δ = 50
    }, index=idx)
    index_df = pd.DataFrame({
        'Volume': [1000.0, 1050.0],  # Δ = 50
        'Amount': [2000.0, 2100.0],  # Δ = 100
    }, index=idx)
    mv = compute_movement_projection(stock_df, index_df)
    # β = (40·50 + 50·100) / (50² + 100²) = 7000/12500 = 0.56
    np.testing.assert_allclose(mv['proj_coeff'][0], 0.56, rtol=1e-9)
    np.testing.assert_allclose(mv['proj'][0], [28.0, 56.0], rtol=1e-9)
    np.testing.assert_allclose(mv['residual'][0], [12.0, -6.0], rtol=1e-9)
    # residual · index Δ = 12·50 + (-6)·100 = 600 - 600 = 0(正交)
    np.testing.assert_allclose(mv['dot_after'][0], 0.0, atol=1e-9)


def test_movement_zero_index_movement_safe():
    """index ΔV=ΔA=0 时,β 应为 0(分母保护),不报 /0。"""
    idx = pd.date_range('2026-07-01', periods=3, freq='D')
    stock_df = pd.DataFrame({
        'Volume': [100.0, 110.0, 120.0],
        'Amount': [200.0, 210.0, 220.0],
    }, index=idx)
    index_df = pd.DataFrame({
        'Volume': [1000.0, 1000.0, 1000.0],   # 完全不变
        'Amount': [2000.0, 2000.0, 2000.0],
    }, index=idx)
    mv = compute_movement_projection(stock_df, index_df)
    assert mv['proj_coeff'].tolist() == [0.0, 0.0]
    assert mv['proj'].shape == (2, 2)
    assert (mv['proj'] == 0).all()


def test_movement_raises_when_missing_column():
    """缺 Volume / Amount 列时 KeyError。"""
    idx = pd.date_range('2026-07-01', periods=2, freq='D')
    stock_df = pd.DataFrame({'Volume': [1.0, 2.0], 'Close': [3.0, 4.0]}, index=idx)
    index_df = pd.DataFrame({'Volume': [10.0, 20.0], 'Amount': [30.0, 40.0]}, index=idx)
    with pytest.raises(KeyError, match="Amount"):
        compute_movement_projection(stock_df, index_df)


def test_build_movement_result_df_columns():
    """build_movement_result_df 产 18 列(含 Date),行数 = common_idx[1:] 长度。

    8 维度框架:所有运动列加 Move_ 前缀,新增 |u|/|v|/R=|u|/|v| 3 列幅度量。
    """
    idx = pd.date_range('2026-07-01', periods=5, freq='D')
    common_idx = idx  # 5 日
    stock_move = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    index_move = np.array([[10, 20], [30, 40], [50, 60], [70, 80]], dtype=float)
    proj = stock_move.copy()                                  # 假 β=1
    residual = np.zeros_like(stock_move)
    mv = {
        'stock_move': stock_move,
        'index_move': index_move,
        # 8 维度框架新增
        'stock_move_mag': np.linalg.norm(stock_move, axis=1),
        'index_move_mag': np.linalg.norm(index_move, axis=1),
        'relative_move': np.array([0.1, 0.1, 0.1, 0.1]),
        'proj_coeff': np.array([1.0, 1.0, 1.0, 1.0]),
        'proj': proj,
        'residual': residual,
        'proj_mag': np.linalg.norm(proj, axis=1),
        'resi_mag': np.zeros(4),
        'dot_after': np.zeros(4),
        'proj_prices': np.array([2.0, 1.333, 1.2, 1.143]),
        'resi_prices': np.array([0.0, 0.0, 0.0, 0.0]),
    }
    df = build_movement_result_df(common_idx[1:], mv, 'IX_TEST', 'ST_TEST')
    assert len(df) == 4
    expected_cols = [
        'Date',
        'Move_Delta_Vol_IX_TEST', 'Move_Delta_Amt_IX_TEST',
        'Move_Delta_Vol_ST_TEST', 'Move_Delta_Amt_ST_TEST',
        'Move_Stock_Magnitude', 'Move_Index_Magnitude', 'Move_Relative_Move',
        'Move_Proj_Coeff', 'Move_Proj_Vol', 'Move_Proj_Amt',
        'Move_Proj_Magnitude', 'Move_Proj_Price',
        'Move_Resi_Vol', 'Move_Resi_Amt', 'Move_Resi_Magnitude', 'Move_Resi_Price',
        'Move_Dot_After',
    ]
    assert list(df.columns) == expected_cols
    np.testing.assert_array_equal(df['Move_Delta_Vol_ST_TEST'].to_numpy(), stock_move[:, 0])
    np.testing.assert_array_equal(df['Move_Delta_Vol_IX_TEST'].to_numpy(), index_move[:, 0])
    # Move_ 前缀的 price 列数据透传
    np.testing.assert_allclose(df['Move_Proj_Price'].to_numpy(), [2.0, 1.333, 1.2, 1.143], rtol=1e-3)
    np.testing.assert_array_equal(df['Move_Resi_Price'].to_numpy(), [0.0, 0.0, 0.0, 0.0])


def test_movement_returns_proj_price_and_resi_price():
    """compute_movement_projection 同时返回 proj_prices / resi_prices,
    以及 8 维度幅度量 stock_move_mag / index_move_mag / relative_move。"""
    idx = pd.date_range('2026-07-01', periods=4, freq='D')
    df = pd.DataFrame({
        'Volume': [10.0, 12.0, 15.0, 18.0],
        'Amount': [100.0, 130.0, 160.0, 200.0],
    }, index=idx)
    mv = compute_movement_projection(df, df)
    # 价格(slope)
    assert 'proj_prices' in mv
    assert 'resi_prices' in mv
    # 幅度量(8 维度框架新增)
    assert 'stock_move_mag' in mv
    assert 'index_move_mag' in mv
    assert 'relative_move' in mv
    # 长度对齐
    assert len(mv['proj_prices']) == 3
    assert len(mv['resi_prices']) == 3
    assert len(mv['stock_move_mag']) == 3
    assert len(mv['index_move_mag']) == 3
    assert len(mv['relative_move']) == 3
    # 这里 stock == index,β=1,residual=0 → proj_price = ΔA/ΔV 大盘,
    # resi_price 应该是 0(residual 是 0 向量,被 safe_ratios 算 0/0 → 0)
    # 注意:numpy 0/0 用 where 分支保护返回 0,不是 nan
    assert all(np.isfinite(mv['proj_prices'])) or all(mv['proj_prices'] == 0)
    # relative_move = stock_move_mag / index_move_mag → stock==index 时 = 1
    np.testing.assert_allclose(mv['relative_move'], [1.0, 1.0, 1.0])


def test_movement_proj_price_matches_index_movement_direction():
    """proj_price = β·ΔA_i / β·ΔV_i = ΔA_i / ΔV_i(β 抵消)。

    所以 proj_price 应等于大盘运动方向的边际成交均价,不含个股信息。
    """
    idx = pd.date_range('2026-07-01', periods=4, freq='D')
    # 大盘 Amount/Volume 比恒为 2.5(ΔA/ΔV = 5/2)
    index_df = pd.DataFrame({
        'Volume': [100.0, 102.0, 104.0, 106.0],
        'Amount': [250.0, 255.0, 260.0, 265.0],
    }, index=idx)
    # 个股运动完全不同方向
    stock_df = pd.DataFrame({
        'Volume': [10.0, 13.0, 11.0, 14.0],
        'Amount': [40.0, 60.0, 50.0, 70.0],
    }, index=idx)
    mv = compute_movement_projection(stock_df, index_df)
    # proj_price 应当 = ΔA_i / ΔV_i = 5/2 = 2.5,不受 stock 影响
    np.testing.assert_allclose(mv['proj_prices'], [2.5, 2.5, 2.5], rtol=1e-6)


def test_movement_resi_price_caps_outliers():
    """resi_price 在 residual_ΔV≈0 时应被限幅(沿用 compute_projections 的 cap 逻辑)。"""
    idx = pd.date_range('2026-07-01', periods=5, freq='D')
    # 大盘 ΔV 大、个股 ΔV 极小(接近 0):残差近似 = -大盘方向,residual_ΔV 也很小
    index_df = pd.DataFrame({
        'Volume': [100.0, 200.0, 300.0, 400.0, 500.0],
        'Amount': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
    }, index=idx)
    stock_df = pd.DataFrame({
        'Volume': [10.0, 10.001, 10.002, 10.003, 10.004],   # ΔV ≈ 0
        'Amount': [100.0, 100.1, 100.2, 100.3, 100.4],
    }, index=idx)
    mv = compute_movement_projection(stock_df, index_df)
    # resi_prices 应被 cap 到已算值的最大绝对值,所有值都不超 3
    assert np.all(np.abs(mv['resi_prices']) <= 3.0 + 1e-9), (
        f"resi_prices 应被 cap 到 ±3,实际 {mv['resi_prices']}"
    )


# ========================== build_movement_intermediate_df (复核 CSV) ==========================

def test_build_movement_intermediate_df_columns_and_shape():
    """复核 DataFrame:25 列,行数 = common_idx[1:] 长度。

    设计目的:`projection_2d.py --movement` 顺手落一份 CSV 到 data/projection/intermediate/,
    每行覆盖原始 Vol/Ama、Δ、β 分子分母、|u|/|v|/R、proj/resi、点积、|x|>3 异常 —
    人工逐日核对公式。
    """
    idx = pd.date_range('2026-07-01', periods=5, freq='D')
    stock_df = pd.DataFrame({
        'Volume': [1e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6],
        'Amount': [1e7, 1.15e7, 1.3e7, 1.45e7, 1.6e7],
    }, index=idx)
    index_df = pd.DataFrame({
        'Volume': [1e8, 1.05e8, 1.1e8, 1.15e8, 1.2e8],
        'Amount': [1e9, 1.08e9, 1.16e9, 1.24e9, 1.32e9],
    }, index=idx)
    mv = compute_movement_projection(stock_df, index_df)

    df = build_movement_intermediate_df(idx[1:], mv, stock_df, index_df, 'IX_TEST', 'ST_TEST')
    assert len(df) == 4

    expected_cols = [
        'Date',
        'Move_Vol_IX_TEST', 'Move_Amt_IX_TEST', 'Move_Vol_ST_TEST', 'Move_Amt_ST_TEST',
        'Move_Delta_Vol_IX_TEST', 'Move_Delta_Amt_IX_TEST',
        'Move_Delta_Vol_ST_TEST', 'Move_Delta_Amt_ST_TEST',
        'Move_V_dot_V', 'Move_U_dot_V',
        'Move_Stock_Magnitude', 'Move_Index_Magnitude', 'Move_Relative_Move',
        'Move_Proj_Coeff', 'Move_Proj_Vol', 'Move_Proj_Amt',
        'Move_Proj_Magnitude', 'Move_Proj_Price',
        'Move_Resi_Vol', 'Move_Resi_Amt', 'Move_Resi_Magnitude',
        'Move_Resi_Price_Raw', 'Move_Resi_Price',
        'Move_Dot_After',
    ]
    assert list(df.columns) == expected_cols
    assert len(expected_cols) == 25


def test_build_movement_intermediate_df_recomputes_every_step():
    """复核 CSV 中每一步数值可由前几列独立算回,人工核对友好。

    校验链:
      ΔV/ΔA = 当前 Vol/Ama - 上一行 Vol/Ama
      V_dot_V = ΔV_idx² + ΔA_idx²
      U_dot_V = ΔV_stk·ΔV_idx + ΔA_stk·ΔA_idx
      |u| = √(ΔV_stk² + ΔA_stk²),|v| = √(ΔV_idx² + ΔA_idx²)
      R = |u| / |v|
      Proj_Coeff = U_dot_V / V_dot_V
      Proj_Vol/Amt = Proj_Coeff × ΔV/ΔA_idx
      Resi_Vol/Amt = ΔV/ΔA_stk - Proj_Vol/Amt
      Dot_After = Resi_Vol·ΔV_idx + Resi_Amt·ΔA_idx (理想 = 0)
      Proj_Price = ΔA_idx / ΔV_idx(β 抵消)
      Resi_Price_Raw = Resi_Amt / Resi_Vol

    用小整数数据(与 test_movement_residual_orthogonal_to_index 同款)避免
    float64 roundoff 让 Dot_After 看上去不严格为 0。
    """
    idx = pd.date_range('2026-07-01', periods=5, freq='D')
    stock_df = pd.DataFrame({
        'Volume': [100, 140, 180, 220, 260],      # ΔV = [40, 40, 40, 40]
        'Amount': [200, 250, 310, 360, 410],      # ΔA = [50, 60, 50, 50]
    }, index=idx)
    index_df = pd.DataFrame({
        'Volume': [1000, 1050, 1100, 1150, 1200],   # ΔV = [50, 50, 50, 50]
        'Amount': [2000, 2100, 2200, 2300, 2400],   # ΔA = [100, 100, 100, 100]
    }, index=idx)
    mv = compute_movement_projection(stock_df, index_df)
    df = build_movement_intermediate_df(idx[1:], mv, stock_df, index_df, 'IX_TEST', 'ST_TEST')

    # ΔV/ΔA = diff
    expected_dv_idx = np.diff(index_df['Volume'].to_numpy())
    expected_da_idx = np.diff(index_df['Amount'].to_numpy())
    np.testing.assert_allclose(df['Move_Delta_Vol_IX_TEST'].to_numpy(), expected_dv_idx)
    np.testing.assert_allclose(df['Move_Delta_Amt_IX_TEST'].to_numpy(), expected_da_idx)

    # V_dot_V / U_dot_V
    np.testing.assert_allclose(
        df['Move_V_dot_V'].to_numpy(),
        expected_dv_idx**2 + expected_da_idx**2,
    )
    dv_stk = np.diff(stock_df['Volume'].to_numpy())
    da_stk = np.diff(stock_df['Amount'].to_numpy())
    np.testing.assert_allclose(
        df['Move_U_dot_V'].to_numpy(),
        dv_stk * expected_dv_idx + da_stk * expected_da_idx,
    )

    # |u|/|v|/R
    np.testing.assert_allclose(
        df['Move_Stock_Magnitude'].to_numpy(),
        np.sqrt(dv_stk**2 + da_stk**2),
    )
    np.testing.assert_allclose(
        df['Move_Index_Magnitude'].to_numpy(),
        np.sqrt(expected_dv_idx**2 + expected_da_idx**2),
    )
    np.testing.assert_allclose(
        df['Move_Relative_Move'].to_numpy(),
        np.sqrt(dv_stk**2 + da_stk**2) / np.sqrt(expected_dv_idx**2 + expected_da_idx**2),
        rtol=1e-9,
    )

    # Proj_Coeff = U/V(分母非零时)
    expected_beta = df['Move_U_dot_V'].to_numpy() / df['Move_V_dot_V'].to_numpy()
    np.testing.assert_allclose(df['Move_Proj_Coeff'].to_numpy(), expected_beta, rtol=1e-9)

    # Proj_Vol/Amt = β × ΔV/ΔA_idx
    np.testing.assert_allclose(
        df['Move_Proj_Vol'].to_numpy(),
        df['Move_Proj_Coeff'].to_numpy() * expected_dv_idx,
        rtol=1e-9,
    )

    # Resi_Vol/Amt = ΔV/ΔA_stk - Proj_Vol/Amt
    np.testing.assert_allclose(
        df['Move_Resi_Vol'].to_numpy(),
        dv_stk - df['Move_Proj_Vol'].to_numpy(),
        rtol=1e-9,
    )

    # Dot_After(数值上 = resi · v,公式上为 0;小整数数据下 roundoff 可忽略)
    resi_dv = df['Move_Resi_Vol'].to_numpy()
    resi_da = df['Move_Resi_Amt'].to_numpy()
    expected_dot = resi_dv * expected_dv_idx + resi_da * expected_da_idx
    np.testing.assert_allclose(df['Move_Dot_After'].to_numpy(), expected_dot, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(df['Move_Dot_After'].to_numpy(), 0.0, atol=1e-9)

    # Proj_Price = ΔA_idx / ΔV_idx(β 抵消)
    np.testing.assert_allclose(
        df['Move_Proj_Price'].to_numpy(),
        expected_da_idx / expected_dv_idx,
        rtol=1e-9,
    )

    # Resi_Price_Raw = Resi_Amt / Resi_Vol(可能 Inf/NaN)
    raw = df['Move_Resi_Price_Raw'].to_numpy()
    expected_raw = resi_da / resi_dv
    fin_mask = np.isfinite(raw) & np.isfinite(expected_raw)
    np.testing.assert_allclose(raw[fin_mask], expected_raw[fin_mask], rtol=1e-9)
    # Resi_Price(限幅后)应 ≤ 3
    assert np.all(np.abs(df['Move_Resi_Price'].to_numpy()) <= 3.0 + 1e-9)


def test_build_movement_intermediate_df_resi_price_raw_shows_div_by_zero_as_nan():
    """residual_Vol = 0 时,Move_Resi_Price_Raw 应当是 NaN(不是被替换为 0)。

    设计目的:复核 CSV 里 0/0 用 NaN 暴露,人工一眼能看出"这天残差为 0 向量,price 无意义"。
    Move_Resi_Price 走 _movement_safe_ratios 时会被替换为 0(防 /0),两者并列保留。

    触发条件:stock 运动完全沿 index 方向 → u == proj → residual == (0, 0)。
    与 test_movement_basic_projection_along_index_direction 同算例(ΔV/ΔA = 30/60 沿 50/100)。
    """
    idx = pd.date_range('2026-07-01', periods=3, freq='D')
    stock_df = pd.DataFrame({
        'Volume': [100.0, 130.0, 150.0],
        'Amount': [200.0, 260.0, 300.0],
    }, index=idx)
    index_df = pd.DataFrame({
        'Volume': [1000.0, 1050.0, 1100.0],
        'Amount': [2000.0, 2100.0, 2200.0],
    }, index=idx)
    mv = compute_movement_projection(stock_df, index_df)
    df = build_movement_intermediate_df(idx[1:], mv, stock_df, index_df, 'IX', 'ST')

    # 残差向量 = 0 → residual_Vol = 0 → Move_Resi_Price_Raw 应 = NaN(0/0)
    assert np.all(np.isnan(df['Move_Resi_Price_Raw'].to_numpy())), (
        f"residual = 0,Move_Resi_Price_Raw 应 NaN,实际 {df['Move_Resi_Price_Raw'].tolist()}"
    )
    # Move_Resi_Price(走 _movement_safe_ratios 0/0 保护)应 = 0
    assert np.all(df['Move_Resi_Price'].to_numpy() == 0.0)
    # Move_Resi_Vol / Move_Resi_Amt 也应为 0 — 双重确认残差为 0 向量
    np.testing.assert_array_equal(df['Move_Resi_Vol'].to_numpy(), [0.0, 0.0])
    np.testing.assert_array_equal(df['Move_Resi_Amt'].to_numpy(), [0.0, 0.0])


def test_build_movement_intermediate_df_raw_vol_amt_columns_match_input():
    """Move_Vol/Move_Amt 原始列(非 Δ)= caller 传 stock_df/index_df 对应行的
    Volume/Amount,丢首行对齐。"""
    idx = pd.date_range('2026-07-01', periods=5, freq='D')
    stock_df = pd.DataFrame({
        'Volume': [10.0, 20.0, 30.0, 40.0, 50.0],
        'Amount': [100.0, 200.0, 300.0, 400.0, 500.0],
    }, index=idx)
    index_df = pd.DataFrame({
        'Volume': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        'Amount': [1e4, 2e4, 3e4, 4e4, 5e4],
    }, index=idx)
    mv = compute_movement_projection(stock_df, index_df)
    df = build_movement_intermediate_df(idx[1:], mv, stock_df, index_df, 'IX', 'ST')

    # 丢首行后,Move_Vol_ST 应 = stock_df[1:] 的 Volume
    np.testing.assert_array_equal(
        df['Move_Vol_ST'].to_numpy(), stock_df['Volume'].to_numpy()[1:]
    )
    np.testing.assert_array_equal(
        df['Move_Amt_ST'].to_numpy(), stock_df['Amount'].to_numpy()[1:]
    )
    np.testing.assert_array_equal(
        df['Move_Vol_IX'].to_numpy(), index_df['Volume'].to_numpy()[1:]
    )
    np.testing.assert_array_equal(
        df['Move_Amt_IX'].to_numpy(), index_df['Amount'].to_numpy()[1:]
    )


# ========================== 8 维度幅度量(magnitude / R)单测 ==========================

def test_movement_returns_magnitudes_and_relative_move():
    """compute_movement_projection 的 stock_move_mag / index_move_mag / relative_move
    应等于手算的 ‖u‖ / ‖v‖ / ‖u‖/‖v‖,大盘运动太小(v·v < 1e-12)时 R → 0。

    设计要点:这是 8 维度框架里「幅度量」与原 Price(方向斜率)的区分 —
    Price 是 β·ΔA/β·ΔV(方向),Magnitude 是 √(ΔV² + ΔA²)(大小),识别
    「大盘没动 / 个股暴动」靠 Magnitude 而不是 Price。
    """
    idx = pd.date_range('2026-07-01', periods=3, freq='D')
    stock_df = pd.DataFrame({
        'Volume': [100.0, 130.0, 150.0],   # ΔV = [30, 20]
        'Amount': [200.0, 260.0, 300.0],   # ΔA = [60, 40]
    }, index=idx)
    index_df = pd.DataFrame({
        'Volume': [1000.0, 1050.0, 1100.0],  # ΔV = [50, 50]
        'Amount': [2000.0, 2100.0, 2200.0],  # ΔA = [100, 100]
    }, index=idx)
    mv = compute_movement_projection(stock_df, index_df)

    # 丢首行剩 2 行
    assert mv['stock_move_mag'].shape == (2,)
    assert mv['index_move_mag'].shape == (2,)
    assert mv['relative_move'].shape == (2,)

    # |u| = √(ΔV_s² + ΔA_s²)
    # 第一行:√(30² + 60²) = √(900+3600) = √4500 ≈ 67.082
    # 第二行:√(20² + 40²) = √(400+1600) = √2000 ≈ 44.721
    np.testing.assert_allclose(mv['stock_move_mag'], [np.sqrt(4500), np.sqrt(2000)], rtol=1e-9)

    # |v| = √(ΔV_i² + ΔA_i²) = √(50² + 100²) = √12500 ≈ 111.803(两行一样)
    np.testing.assert_allclose(mv['index_move_mag'], [np.sqrt(12500), np.sqrt(12500)], rtol=1e-9)

    # R = |u|/|v|
    np.testing.assert_allclose(
        mv['relative_move'],
        [np.sqrt(4500) / np.sqrt(12500), np.sqrt(2000) / np.sqrt(12500)],
        rtol=1e-9,
    )


def test_movement_relative_move_handles_zero_index_movement():
    """大盘运动 ‖v‖ → 0 时,relative_move 应为 0(沿用 β 阈值同款保护),
    而不是 NaN/Inf。这样下游图表不会因少量 Inf 行被刷坏。
    """
    idx = pd.date_range('2026-07-01', periods=3, freq='D')
    stock_df = pd.DataFrame({
        'Volume': [100.0, 110.0, 120.0],
        'Amount': [200.0, 210.0, 220.0],
    }, index=idx)
    index_df = pd.DataFrame({
        'Volume': [1000.0, 1000.0, 1000.0],   # 完全不变 → ΔV = 0
        'Amount': [2000.0, 2000.0, 2000.0],   # ΔA = 0
    }, index=idx)
    mv = compute_movement_projection(stock_df, index_df)

    # ‖v‖ = 0,但 ‖u‖ = √(10² + 10²) ≠ 0。R 应被保护为 0,不爆 NaN/Inf
    assert np.all(np.isfinite(mv['relative_move'])), (
        f"相对运动应全 finite(被 where 分支保护),实际 {mv['relative_move']}"
    )
    np.testing.assert_array_equal(mv['relative_move'], [0.0, 0.0])
    # 自身幅度量仍正常
    np.testing.assert_allclose(mv['index_move_mag'], [0.0, 0.0])
    np.testing.assert_allclose(mv['stock_move_mag'], [np.sqrt(200), np.sqrt(200)], rtol=1e-9)


def test_load_pair_period_default_is_daily():
    from projection import _projection_core as P
    import inspect
    sig = inspect.signature(P.load_pair)
    assert 'period' in sig.parameters
    assert sig.parameters['period'].default == 'daily'

def test_load_pair_invalid_period_raises():
    from projection import _projection_core as P
    import pytest
    with pytest.raises(ValueError, match="period"):
        P.load_pair('000001.SZ', 5, None, period='3m')  # pipeline=None triggers later check


def test_state_returns_magnitudes_and_relative_move():
    """compute_projections 的 state_stock_mag / state_index_mag / state_relative_move
    应等于手算的 ‖u‖ / ‖v‖ / ‖u‖/‖v‖。切原始量纲后输入是原始向量
    (这里用 0/1 数值,既是 valid 原始值也是 valid 归一化值,数字断言不变)。

    2026-08-16:compute_projections 不再返回 resi_prices(2-D 退化),
    此测试只验证 magnitude 三件套。
    """
    # 简单造 4 行:u/v 都是单位向量不同比例
    # u1=[1, 0],v1=[1, 0]  → |u|=1, |v|=1, R=1
    # u2=[0, 1],v2=[0, 1]  → |u|=1, |v|=1, R=1
    # u3=[1, 1],v3=[1, 0]  → |u|=√2, |v|=1, R=√2
    # u4=[0, 0],v4=[0, 0]  → |u|=0, |v|=0, R=0(0/0 保护)
    vec_stock = np.array([
        [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0],
    ])
    vec_index = np.array([
        [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0],
    ])
    proj = compute_projections(vec_stock, vec_index)
    assert 'state_stock_mag' in proj
    assert 'state_index_mag' in proj
    assert 'state_relative_move' in proj
    # 2026-08-16:resi_prices 已从 compute_projections 删除
    assert 'resi_prices' not in proj
    assert proj['state_stock_mag'].shape == (4,)
    assert proj['state_index_mag'].shape == (4,)
    assert proj['state_relative_move'].shape == (4,)

    # |v|=1 在第 1/2/3 行,|v|=0 在第 4 行
    np.testing.assert_allclose(
        proj['state_index_mag'], [1.0, 1.0, 1.0, 0.0], atol=1e-12,
    )
    # |u| = 1, 1, √2, 0
    np.testing.assert_allclose(
        proj['state_stock_mag'], [1.0, 1.0, np.sqrt(2), 0.0], atol=1e-12,
    )
    # R = 1, 1, √2, 0(0/0 保护)
    np.testing.assert_allclose(
        proj['state_relative_move'], [1.0, 1.0, np.sqrt(2), 0.0], atol=1e-12,
    )
    # 全 finite
    assert np.all(np.isfinite(proj['state_relative_move']))

