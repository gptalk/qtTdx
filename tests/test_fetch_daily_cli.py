# -*- coding: utf-8 -*-
"""Smoke tests for fetch_daily.py CLI flags."""
import subprocess


def test_fetch_daily_help_shows_period():
    """`fetch_daily.py --help` should expose --period and --lookback-days."""
    out = subprocess.run(
        ['python', 'backtrace/data_fetch/fetch_daily.py', '--help'],
        capture_output=True, text=True, encoding='utf-8',
    )
    assert '--period' in out.stdout, f"--period missing from help:\n{out.stdout}"
    assert '--lookback-days' in out.stdout, f"--lookback-days missing from help:\n{out.stdout}"
    assert '5m' in out.stdout, f"'5m' choice missing from help:\n{out.stdout}"


def test_fetch_daily_help_exits_zero():
    """`fetch_daily.py --help` should exit with code 0."""
    out = subprocess.run(
        ['python', 'backtrace/data_fetch/fetch_daily.py', '--help'],
        capture_output=True, text=True, encoding='utf-8',
    )
    assert out.returncode == 0, f"--help exited with {out.returncode}"
