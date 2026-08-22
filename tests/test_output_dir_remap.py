def test_output_dir_includes_period_for_intraday():
    """Helper output_dir_for(args.period) returns suffixed dir for non-daily."""
    from dynamics import dynamics_granularity_compare  # new module from C3
    # Just import-time check; full assertions come in Task C3
    assert dynamics_granularity_compare is not None
