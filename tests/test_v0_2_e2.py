# -*- coding: utf-8 -*-
"""Tests for V0.2-E1 E2 helper + cross-sectional script.

Locks down three contracts:
1. `_e2_features.extract_features_one('002475.SZ')` returns dict with required
   keys (beta_market / stock_volatility / liquidity) when real data exists.
2. `extract_features_one('NONEXISTENT.XY')` returns None (skips insufficient).
3. Running `v0_2_e2_cross_sectional_q.main()` on synthetic 100-stock paired +
   kc + 10 daily files produces `cross_sectional_correlations.csv` with the
   expected shape (7 features × 4 columns).

The helper bug fixed in commit 6162488 is exercised in tests 1 + 3 because
both call `extract_features_one` on real/synthetic aligned stock+market data.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECTION_DIR = os.path.join(REPO, 'backtrace', 'projection')
for p in (PROJECTION_DIR, REPO, os.path.join(REPO, 'backtrace')):
    if p not in sys.path:
        sys.path.insert(0, p)


# ----- Test 1: helper on real data, returns dict with required keys -----

def test_e2_features_helper_one():
    """E2: extract_features_one('002475.SZ') returns dict with required keys.

    Skips if `data/stocks/002475_SZ_daily.csv` or `data/indices/399001_SZ_daily.csv`
    are missing from the local cache.
    """
    stock_path = os.path.join(REPO, 'data', 'stocks', '002475_SZ_daily.csv')
    index_path = os.path.join(REPO, 'data', 'indices', '399001_SZ_daily.csv')
    if not (os.path.exists(stock_path) and os.path.exists(index_path)):
        pytest.skip(f'real data missing: {stock_path} or {index_path}')

    from _e2_features import extract_features_one

    feats = extract_features_one('002475.SZ')
    assert feats is not None, 'extract_features_one returned None for 002475.SZ'
    required_keys = {'code', 'beta_market', 'stock_volatility', 'liquidity'}
    missing = required_keys - set(feats)
    assert not missing, f'missing keys in features dict: {missing}'
    # code must round-trip the input
    assert feats['code'] == '002475.SZ'
    # numeric features should be finite floats
    for key in ('beta_market', 'stock_volatility', 'liquidity'):
        v = feats[key]
        assert isinstance(v, float), f'{key} is {type(v).__name__}, expected float'
        assert np.isfinite(v), f'{key} is not finite: {v}'


# ----- Test 2: helper skips insufficient / missing data -----

def test_e2_features_helper_skips_insufficient(tmp_path, monkeypatch):
    """E2: extract_features_one('NONEXISTENT.XY') returns None.

    Runs from a fresh tmp cwd so the helper sees no `data/stocks/` /
    `data/indices/` directories and short-circuits before hitting pandas.
    """
    monkeypatch.chdir(tmp_path)
    # Sanity: confirm the helper can't see real data from this cwd
    assert not os.path.exists('data/stocks'), 'tmp cwd unexpectedly contains data/stocks/'

    from _e2_features import extract_features_one

    feats = extract_features_one('NONEXISTENT.XY')
    assert feats is None, f'expected None, got {feats}'


# ----- Test 3: end-to-end pipeline → correlations CSV -----

def _make_synthetic_paired(n: int, seed: int) -> pd.DataFrame:
    """Build a synthetic V0.2-C1 paired CSV with 100 stocks."""
    rng = np.random.default_rng(seed)
    half = n // 2
    codes = (
        [f'{600000 + i:06d}.SH' for i in range(half)]
        + [f'{100000 + i:06d}.SZ' for i in range(n - half)]
    )
    df = pd.DataFrame({
        'code': codes,
        'name': [f'stock_{i}' for i in range(n)],
        'ic_real_C0': rng.normal(0, 0.5, n),
        'ic_real_C1': rng.normal(0, 0.5, n),
        'delta_oos_ic': rng.normal(0, 0.18, n),
        'q_drift_C0': rng.normal(-0.1, 0.2, n),
        'q_drift_C1': rng.normal(-0.05, 0.15, n),
        'delta_q_drift': rng.normal(0.05, 0.1, n),
        'q_hat_C0': rng.uniform(0, 1, n),
        'q_hat_C1': rng.uniform(0, 0.6, n),
        'delta_q_hat': rng.normal(-0.2, 0.1, n),
        'test_fit_r2_C0': rng.uniform(0, 1, n),
        'test_fit_r2_C1': rng.uniform(0, 1, n),
        'delta_test_fit_r2': rng.normal(0, 0.2, n),
        'oos_r2_C0': rng.normal(-1e6, 1e6, n),
        'oos_r2_C1': rng.normal(-1e6, 1e6, n),
        'delta_oos_r2': rng.normal(0, 1e6, n),
        'condition_number_C0': rng.uniform(1, 50, n),
        'condition_number_C1': rng.uniform(1, 30, n),
        'delta_cond': rng.normal(-5, 5, n),
        'sign_flipped': rng.choice([True, False], n),
        'q_drift_attenuated': rng.choice([True, False], n),
        'q_drift_amplified': rng.choice([True, False], n),
        'ic_improved': rng.choice([True, False], n),
        'ic_worsened': rng.choice([True, False], n),
    })
    return df


def _make_synthetic_kc(codes: list[str], seed: int) -> pd.DataFrame:
    """Build a synthetic kc_estimates_model2_diag CSV matching the 7 cols
    the E2 script actually reads (code, q_hat, r2, condition_number, ic_real)."""
    rng = np.random.default_rng(seed + 1)
    n = len(codes)
    return pd.DataFrame({
        'code': codes,
        'q_hat': rng.uniform(0, 1, n),
        'r2': rng.uniform(0, 1, n),
        'condition_number': rng.uniform(1, 50, n),
        'ic_real': rng.normal(-0.5, 0.3, n),
    })


def _make_synthetic_daily(
    code: str,
    seed: int,
    n_days: int = 300,
    start: str = '2024-01-01',
) -> pd.DataFrame:
    """Build a synthetic daily OHLCVA CSV matching `data_store` schema."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq='D')
    close = 10.0 + rng.normal(0, 0.5, n_days).cumsum()
    open_ = close + rng.normal(0, 0.1, n_days)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, n_days))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, n_days))
    return pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': rng.uniform(1e6, 5e6, n_days),
        'Amount': rng.uniform(1e9, 5e9, n_days),
    }, index=dates)


def _write_csv(df: pd.DataFrame, path) -> None:
    """Write DataFrame to CSV in the `data_store` convention:
    unnamed datetime index, OHLCVA columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, encoding='utf-8')


def test_e2_correlation_matrix_shape(tmp_path):
    """E2: synthetic 100-stock data (10 with daily files) →
    cross_sectional_correlations.csv exists with shape (7 features, 4 cols).

    Only 10 of 100 paired codes have daily CSVs, mirroring the real-world
    sparsity. extract_features_cached should skip codes without daily files.
    """
    import v0_2_e2_cross_sectional_q as e2

    # Build synthetic 100-stock universe
    n = 100
    paired = _make_synthetic_paired(n, seed=11)
    codes = paired['code'].tolist()

    # Build synthetic kc (covers all 100 codes)
    kc = _make_synthetic_kc(codes, seed=11)

    # Write inputs at the script's expected relative paths
    paired_dir = tmp_path / 'data' / 'projection_v01_c1'
    stocks_dir = tmp_path / 'data' / 'stocks'
    indices_dir = tmp_path / 'data' / 'indices'
    _write_csv(paired, paired_dir / 'c0_c1_paired_compare.csv')
    _write_csv(kc, paired_dir / 'kc_estimates_model2_diag.csv')

    # 10 daily files — first 5 SH + first 5 SZ
    daily_codes = [c for c in codes if c.endswith('.SH')][:5] \
                + [c for c in codes if c.endswith('.SZ')][:5]
    daily_dates = pd.date_range('2024-01-01', periods=300, freq='D')
    for i, code in enumerate(daily_codes):
        df = _make_synthetic_daily(code, seed=i + 100)
        # Align all daily files to a common date range so the helper's
        # intersection with the index files yields >= 100 common dates.
        fname = f'{code.replace(".", "_")}_daily.csv'
        _write_csv(df, stocks_dir / fname)

    # 2 index files aligned to the same date range as stocks
    for idx_code in ('000001.SH', '399001.SZ'):
        rng = np.random.default_rng(hash(idx_code) % (2**32))
        close = 3000.0 + rng.normal(0, 5, 300).cumsum()
        open_ = close + rng.normal(0, 1, 300)
        high = np.maximum(open_, close) + np.abs(rng.normal(0, 3, 300))
        low = np.minimum(open_, close) - np.abs(rng.normal(0, 3, 300))
        idx_df = pd.DataFrame({
            'Open': open_,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': rng.uniform(1e9, 5e9, 300),
            'Amount': rng.uniform(1e12, 5e12, 300),
        }, index=daily_dates)
        fname = f'{idx_code.replace(".", "_")}_daily.csv'
        _write_csv(idx_df, indices_dir / fname)

    # Run E2 main() under chdir(tmp_path) so all relative paths resolve there
    for mod in ('v0_2_e2_cross_sectional_q', '_e2_features'):
        if mod in sys.modules:
            del sys.modules[mod]
    import v0_2_e2_cross_sectional_q as e2_mod
    # Pass a default namespace so main() never calls parse_args() (which would
    # read pytest's sys.argv and exit on --period / -v).
    default_args = e2_mod.parse_args(['--period=daily'])
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        e2_mod.main(default_args)
    finally:
        os.chdir(old_cwd)

    # Verify correlations CSV exists with expected shape
    corr_path = (
        tmp_path / 'data' / 'projection_v01_e2' / 'cross_sectional_correlations.csv'
    )
    assert corr_path.exists(), f'correlations CSV missing: {corr_path}'

    corr = pd.read_csv(corr_path)
    # 7 features (FEATURES list in v0_2_e2_cross_sectional_q.py)
    assert len(corr) == len(e2_mod.FEATURES), (
        f'expected {len(e2_mod.FEATURES)} rows (one per feature), got {len(corr)}'
    )
    # 4 columns: feature / spearman_rho / p_value / n
    expected_cols = {'feature', 'spearman_rho', 'p_value', 'n'}
    assert set(corr.columns) == expected_cols, (
        f'column mismatch.\n'
        f'  expected: {expected_cols}\n'
        f'  got:      {set(corr.columns)}'
    )
    # Every row should have a finite spearman_rho (n >= 3 for the 10 matched stocks)
    assert (corr['spearman_rho'].notna() & np.isfinite(corr['spearman_rho'])).all(), (
        f'some spearman_rho values are NaN/inf:\n{corr}'
    )