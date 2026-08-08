# -*- coding: utf-8 -*-
"""验证 TQ 不可用时,load_ohlcva 真能从 data/ 回退拿到数据。

修复前 _try_local_csv 读 backtrace/{code}_daily.csv,而 CSV 实际在别处,
所以这些测试在修复前必然失败(返回 None)。
"""
import pandas as pd
import pytest

from common import data_store
from common import tsfresh_pipeline as P


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(data_store, 'DATA_DIR', str(tmp_path))
    return tmp_path


@pytest.fixture
def sample_df():
    idx = pd.to_datetime(['2026-08-05', '2026-08-06', '2026-08-07'])
    return pd.DataFrame({
        'Open':   [10.0, 10.5, 10.2],
        'High':   [10.8, 10.9, 10.6],
        'Low':    [9.9, 10.1, 10.0],
        'Close':  [10.5, 10.2, 10.4],
        'Volume': [1000000.0, 1200000.0, 900000.0],
        'Amount': [10000000.0, 12000000.0, 9000000.0],
    }, index=idx)


def test_load_ohlcva_falls_back_to_data_dir(tmp_store, sample_df):
    data_store.save_daily('000001.SH', sample_df, 'indices')
    got = P.load_ohlcva('000001.SH', use_tq=False)
    assert got is not None, "回退失效:data/indices/ 下有 CSV 却拿到 None"
    assert list(got.columns) == ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
    assert len(got) == 3


def test_load_ohlcva_returns_none_when_absent(tmp_store):
    assert P.load_ohlcva('999999.SZ', use_tq=False) is None


def test_fallback_finds_stock_kind_too(tmp_store, sample_df):
    data_store.save_daily('002475.SZ', sample_df, 'stocks')
    assert P.load_ohlcva('002475.SZ', use_tq=False) is not None