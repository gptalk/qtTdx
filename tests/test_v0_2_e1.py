# -*- coding: utf-8 -*-
"""Tests for V0.2-E1 E1 diagnostic script — ΔIC distribution analysis.

Locks down `v0_2_e1_delta_ic_distribution.main()` contract:
- Reads `data/projection_v01_c1/c0_c1_paired_compare.csv` (relative to cwd)
- Writes `data/projection_v01_e1/{delta_ic_summary.csv, delta_ic_buckets.csv, ...}`
- Summary CSV: 17 metrics including n, mean, sign_test_p_gt_0
- Buckets CSV: 6 fixed bucket labels, counts sum to N

We use synthetic paired CSVs (5208 + 1000 rows) so the test runs without the
real V0.2-C1 paired output on disk. The script's main() reads only `delta_oos_ic`
plus a few other columns for plotting — the synthetic paired frame supplies all
required columns.
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


# V0.2-C1 paired CSV has 25 columns; populate all to keep main() happy even
# though it only consumes `delta_oos_ic` (the rest are passed through).
_PAIRED_COLS = [
    'code', 'name',
    'ic_real_C0', 'ic_real_C1', 'delta_oos_ic',
    'q_drift_C0', 'q_drift_C1', 'delta_q_drift',
    'q_hat_C0', 'q_hat_C1', 'delta_q_hat',
    'test_fit_r2_C0', 'test_fit_r2_C1', 'delta_test_fit_r2',
    'oos_r2_C0', 'oos_r2_C1', 'delta_oos_r2',
    'condition_number_C0', 'condition_number_C1', 'delta_cond',
    'sign_flipped', 'q_drift_attenuated', 'q_drift_amplified',
    'ic_improved', 'ic_worsened',
]


def _make_synthetic_paired(n_stocks: int, seed: int) -> pd.DataFrame:
    """Build a synthetic paired CSV matching V0.2-C1 25-col schema.

    `delta_oos_ic` drawn from N(0, 0.18) so the bucket distribution is wide
    enough to exercise all 6 buckets but not degenerate.
    """
    rng = np.random.default_rng(seed)
    half = n_stocks // 2
    codes = (
        [f'{600000 + i:06d}.SH' for i in range(half)]
        + [f'{100000 + i:06d}.SZ' for i in range(n_stocks - half)]
    )
    return pd.DataFrame({
        'code': codes,
        'name': [f'stock_{i}' for i in range(n_stocks)],
        'ic_real_C0': rng.normal(0, 0.5, n_stocks),
        'ic_real_C1': rng.normal(0, 0.5, n_stocks),
        'delta_oos_ic': rng.normal(0, 0.18, n_stocks),
        'q_drift_C0': rng.normal(-0.1, 0.2, n_stocks),
        'q_drift_C1': rng.normal(-0.05, 0.15, n_stocks),
        'delta_q_drift': rng.normal(0.05, 0.1, n_stocks),
        'q_hat_C0': rng.uniform(0, 1, n_stocks),
        'q_hat_C1': rng.uniform(0, 0.6, n_stocks),
        'delta_q_hat': rng.normal(-0.2, 0.1, n_stocks),
        'test_fit_r2_C0': rng.uniform(0, 1, n_stocks),
        'test_fit_r2_C1': rng.uniform(0, 1, n_stocks),
        'delta_test_fit_r2': rng.normal(0, 0.2, n_stocks),
        'oos_r2_C0': rng.normal(-1e6, 1e6, n_stocks),
        'oos_r2_C1': rng.normal(-1e6, 1e6, n_stocks),
        'delta_oos_r2': rng.normal(0, 1e6, n_stocks),
        'condition_number_C0': rng.uniform(1, 50, n_stocks),
        'condition_number_C1': rng.uniform(1, 30, n_stocks),
        'delta_cond': rng.normal(-5, 5, n_stocks),
        'sign_flipped': rng.choice([True, False], n_stocks),
        'q_drift_attenuated': rng.choice([True, False], n_stocks),
        'q_drift_amplified': rng.choice([True, False], n_stocks),
        'ic_improved': rng.choice([True, False], n_stocks),
        'ic_worsened': rng.choice([True, False], n_stocks),
    }, columns=_PAIRED_COLS)


def _setup_paired_in_tmp(tmp_path, n_stocks: int, seed: int) -> None:
    """Write synthetic paired CSV to tmp_path/data/projection_v01_c1/."""
    paired_dir = tmp_path / 'data' / 'projection_v01_c1'
    paired_dir.mkdir(parents=True, exist_ok=True)
    paired = _make_synthetic_paired(n_stocks, seed)
    paired.to_csv(paired_dir / 'c0_c1_paired_compare.csv', index=False)


def _run_e1_main(tmp_path):
    """Import E1 main() fresh and run it under chdir(tmp_path)."""
    # Force re-import in case a prior test polluted sys.modules.
    for mod in ('v0_2_e1_delta_ic_distribution',):
        if mod in sys.modules:
            del sys.modules[mod]
    import v0_2_e1_delta_ic_distribution as e1
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        e1.main()
    finally:
        os.chdir(old_cwd)
    return e1


# ----- Test 1: 5208-row summary stats -----

def test_e1_summary_stats_compute(tmp_path):
    """E1: 5208-row synthetic paired CSV → summary CSV has 17 metrics,
    n == 5208, mean ≈ 0 (random normal), sign_test in (0, 1)."""
    n = 5208
    _setup_paired_in_tmp(tmp_path, n_stocks=n, seed=42)

    _run_e1_main(tmp_path)

    summary_path = tmp_path / 'data' / 'projection_v01_e1' / 'delta_ic_summary.csv'
    assert summary_path.exists(), f'summary CSV missing: {summary_path}'
    summary = pd.read_csv(summary_path, index_col='metric')

    # 17 metrics as defined in E1 main() summary dict
    expected_metrics = {
        'n', 'mean', 'median', 'std',
        'p5', 'p10', 'p25', 'p75', 'p90', 'p95', 'min', 'max',
        'sign_test_p_gt_0', 'sign_test_p_gt_0.05',
        'large_movers_pct', 'very_negative_pct', 'very_positive_pct',
    }
    assert set(summary.index) == expected_metrics, (
        f'metric set mismatch.\n'
        f'  expected: {expected_metrics}\n'
        f'  got:      {set(summary.index)}'
    )
    # n must equal synthetic row count
    assert summary.loc['n', 'value'] == n, (
        f"summary['n'] = {summary.loc['n', 'value']}, expected {n}"
    )
    # mean should be near 0 since delta_oos_ic ~ N(0, 0.18); tolerance 0.01
    assert abs(summary.loc['mean', 'value']) < 0.01, (
        f"mean={summary.loc['mean', 'value']:+.4f}, expected near 0"
    )
    # sign_test_p_gt_0 must lie strictly in (0, 1)
    p_pos = float(summary.loc['sign_test_p_gt_0', 'value'])
    assert 0.0 < p_pos < 1.0, f'sign_test_p_gt_0 = {p_pos}, expected in (0, 1)'


# ----- Test 2: bucket counts sum to N -----

def test_e1_buckets_sum_to_n(tmp_path):
    """E1: 1000-row synthetic → bucket counts sum to N (=1000)."""
    n = 1000
    _setup_paired_in_tmp(tmp_path, n_stocks=n, seed=7)

    _run_e1_main(tmp_path)

    buckets_path = tmp_path / 'data' / 'projection_v01_e1' / 'delta_ic_buckets.csv'
    assert buckets_path.exists(), f'buckets CSV missing: {buckets_path}'
    buckets = pd.read_csv(buckets_path)

    # 6 buckets
    assert len(buckets) == 6, f'expected 6 buckets, got {len(buckets)}'
    # bucket counts sum to N (synthetic data has no NaN in delta_oos_ic)
    assert buckets['count'].sum() == n, (
        f'bucket counts sum to {buckets["count"].sum()}, expected {n}'
    )
    # All 6 expected labels present
    expected_labels = {
        '(-∞,-0.1]', '(-0.1,-0.05]', '(-0.05,0]',
        '(0,0.05]', '(0.05,0.1]', '(0.1,∞)',
    }
    assert set(buckets['bucket']) == expected_labels, (
        f'bucket label mismatch.\n'
        f'  expected: {expected_labels}\n'
        f'  got:      {set(buckets["bucket"])}'
    )
    # pct column sums to ~100
    assert 99.0 < buckets['pct'].sum() < 101.0, (
        f'bucket pct sum = {buckets["pct"].sum():.2f}, expected ~100'
    )