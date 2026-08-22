def test_decision_adopt_when_all_thresholds_met():
    """All hard thresholds met → outcome='adopt'."""
    from dynamics import dynamics_granularity_compare as G

    def verdict(delta_ic, delta_ic_ir, delta_oos_rmse, delta_hit_rate):
        # Mirror the spec §7.1 logic
        return ('adopt' if delta_ic >= G.DELTA_IC_MIN and delta_ic_ir >= G.DELTA_IC_IR_MIN
                and delta_oos_rmse <= G.DELTA_OOS_RMSE_MAX and delta_hit_rate >= G.DELTA_HIT_RATE_MIN
                else 'archive-or-kill')

    assert verdict(0.025, 0.15, -0.10, 0.04) == 'adopt'


def test_decision_kill_when_no_signal():
    """ΔIC ≈ 0, no other signals → outcome != 'adopt'."""
    from dynamics import dynamics_granularity_compare as G

    def verdict(delta_ic, delta_ic_ir, delta_oos_rmse, delta_hit_rate):
        return ('adopt' if delta_ic >= G.DELTA_IC_MIN and delta_ic_ir >= G.DELTA_IC_IR_MIN
                and delta_oos_rmse <= G.DELTA_OOS_RMSE_MAX and delta_hit_rate >= G.DELTA_HIT_RATE_MIN
                else 'archive-or-kill')

    assert verdict(0.001, 0.001, 0.001, 0.001) != 'adopt'


def test_decision_threshold_constants_match_spec():
    """Spec §7.1 values must match module constants exactly."""
    from dynamics import dynamics_granularity_compare as G
    assert G.DELTA_IC_MIN == 0.02
    assert G.DELTA_IC_PVALUE_MAX == 0.05
    assert G.DELTA_IC_IR_MIN == 0.1
    assert G.DELTA_OOS_RMSE_MAX == -0.05
    assert G.DELTA_SI_LAGGED_IC_MIN == 0.02
    assert G.DELTA_HIT_RATE_MIN == 0.03
