# -*- coding: utf-8 -*-
"""Test that all dynamics CLIs expose --period flag (except math-only dynamics_forced_response)."""

import subprocess
import sys


def test_dynamics_cli_help_exposes_period():
    """All dynamics CLIs except forced_response should expose --period."""
    scripts = [
        'backtrace/dynamics/dynamics_system.py',
        'backtrace/dynamics/dynamics_batch.py',
        'backtrace/dynamics/dynamics_1step_oos.py',
        'backtrace/dynamics/dynamics_state_backtest.py',
        'backtrace/dynamics/dynamics_eigen_analysis.py',
        'backtrace/dynamics/dynamics_oos_viz.py',
        'backtrace/dynamics/dynamics_oos_batch.py',
        'backtrace/dynamics/dynamics_state_timeline.py',
        'backtrace/dynamics/dynamics_si_freq_response.py',
        'backtrace/dynamics/dynamics_si_ic.py',
        'backtrace/dynamics/dynamics_si_timeseries.py',
        'backtrace/dynamics/dynamics_si_lagged_ic.py',
        'backtrace/dynamics/dynamics_factor_validation.py',
    ]
    for s in scripts:
        out = subprocess.run(
            [sys.executable, s, '--help'],
            capture_output=True, text=True, encoding='utf-8',
        )
        assert '--period' in out.stdout, f"{s} missing --period"


def test_forced_response_no_period():
    """dynamics_forced_response.py is math-only and should NOT have --period."""
    out = subprocess.run(
        [sys.executable, 'backtrace/dynamics/dynamics_forced_response.py', '--help'],
        capture_output=True, text=True, encoding='utf-8',
    )
    assert '--period' not in out.stdout
