# -*- coding: utf-8 -*-
import os
import sys

import pandas as pd

# data_fetch/ 不是 common/,需单独加进 path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'backtrace', 'data_fetch'))

import fetch_daily as FD


def test_filter_st_drops_st_and_delisted():
    items = [
        {'Code': '000001.SZ', 'Name': '平安银行'},
        {'Code': '000002.SZ', 'Name': 'ST康美'},
        {'Code': '000003.SZ', 'Name': '*ST夏利'},
        {'Code': '000004.SZ', 'Name': '乐视退'},
        {'Code': '000005.SZ', 'Name': '万科A'},
    ]
    assert FD.filter_st(items) == ['000001.SZ', '000005.SZ']


def test_filter_st_skips_malformed_entries():
    items = [None, {}, {'Name': '无代码'}, {'Code': '600000.SH', 'Name': '浦发银行'}]
    assert FD.filter_st(items) == ['600000.SH']


def test_chunked_splits_evenly():
    assert list(FD.chunked([1, 2, 3, 4, 5], size=2)) == [[1, 2], [3, 4], [5]]


def test_chunked_empty():
    assert list(FD.chunked([], size=250)) == []


def test_calendar_days_covers_500_trading_days():
    days = FD.calendar_days_for(500)
    # 实测交易日占比 0.670 -> 500/0.670 ≈ 746;必须留余量但别夸张
    assert 746 <= days <= 850, f"天数 {days} 不合理"


def test_trim_tail_keeps_last_n_sorted():
    idx = pd.to_datetime(['2026-08-07', '2026-08-05', '2026-08-06'])
    df = pd.DataFrame({'Close': [3.0, 1.0, 2.0]}, index=idx)
    got = FD.trim_tail(df, n=2)
    assert list(got['Close']) == [2.0, 3.0]      # 先排序再取尾


def test_trim_tail_keeps_short_series_intact():
    idx = pd.to_datetime(['2026-08-05', '2026-08-06'])
    df = pd.DataFrame({'Close': [1.0, 2.0]}, index=idx)
    assert len(FD.trim_tail(df, n=500)) == 2     # 不补齐、不丢弃


def test_index_codes_are_sse_and_szse():
    assert FD.INDEX_CODES == ['000001.SH', '399001.SZ']