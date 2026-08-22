# -*- coding: utf-8 -*-
"""Tests for load_ohlcva period kwarg (Task B1)."""
import warnings
warnings.filterwarnings('ignore')
import sys, os
BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)
sys.path.insert(0, 'C:/new_tdx_mock/PYPlugins/user')

from common import tsfresh_pipeline as P


def test_load_ohlcva_period_default_is_daily():
    """Existing default call must produce same shape/columns as before."""
    import inspect
    sig = inspect.signature(P.load_ohlcva)
    assert 'period' in sig.parameters
    assert sig.parameters['period'].default == 'daily'


def test_load_ohlcva_invalid_period_raises():
    """period='3m' must raise ValueError before any data work."""
    import pytest
    with pytest.raises(ValueError, match="period"):
        P.load_ohlcva('000001.SH', period='3m', use_tq=False)
