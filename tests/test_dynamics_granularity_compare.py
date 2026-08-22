# -*- coding: utf-8 -*-
"""Tests for dynamics_granularity_compare (Task C4)."""

def test_daily_vs_5min_summary_columns():
    """build_daily_vs_5min_report emits expected columns."""
    from dynamics import dynamics_granularity_compare as G
    cols = G.REPORT_TABLE_COLS
    expected_subset = ['factor', 'horizon', 'ic_mean_daily', 'ic_mean_5min',
                       'delta_ic', 'delta_ic_ir']
    for c in expected_subset:
        assert c in cols


def test_decision_thresholds_constants():
    """Decision thresholds from spec §7.1 are exposed as module constants."""
    from dynamics import dynamics_granularity_compare as G
    assert G.DELTA_IC_MIN == 0.02
    assert G.DELTA_IC_PVALUE_MAX == 0.05
    assert G.DELTA_IC_IR_MIN == 0.1
    assert G.DELTA_OOS_RMSE_MAX == -0.05
    assert G.DELTA_SI_LAGGED_IC_MIN == 0.02
    assert G.DELTA_HIT_RATE_MIN == 0.03
