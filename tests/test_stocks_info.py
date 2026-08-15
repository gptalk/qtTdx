# -*- coding: utf-8 -*-
"""backtrace/common/stocks_info.py 单测。

策略:
  - monkeypatch `stocks_info.BASIC_CSV` 指向 tmp_path,避免污染真实 data/stock_basic.csv
  - 每个用例前显式 cache_clear(),避免 lru_cache 跨用例复用导致 fixture 失效
"""
import os
import warnings

import pandas as pd
import pytest

from common import stocks_info


@pytest.fixture(autouse=True)
def tmp_basic(monkeypatch, tmp_path):
    """把 stock_basic.csv 重定向到 tmp_path,清理 lru_cache。"""
    monkeypatch.setattr(stocks_info, 'BASIC_CSV', str(tmp_path / 'stock_basic.csv'))
    stocks_info.load_basic_df.cache_clear()
    return tmp_path


def _write_basic(path, rows):
    """写一份 4 列基本表:code, market, name, status。"""
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8')


def test_load_basic_df_returns_indexed_df(tmp_basic):
    """load_basic_df 返回 index=code、含 market/name/status 三列。"""
    p = tmp_basic / 'stock_basic.csv'
    _write_basic(p, [
        {'code': '002475.SZ', 'market': 'SZ', 'name': '立讯精密', 'status': 'active'},
        {'code': '601609.SH', 'market': 'SH', 'name': '国联证券', 'status': 'active'},
        {'code': '000xxx.BJ', 'market': 'BJ', 'name': '北证某',  'status': 'bj'},
    ])
    df = stocks_info.load_basic_df()
    assert df.index.tolist() == ['002475.SZ', '601609.SH', '000xxx.BJ']
    assert df.columns.tolist() == ['market', 'name', 'status']
    assert df.at['601609.SH', 'name'] == '国联证券'


def test_lookup_name_existing_returns_name(tmp_basic):
    p = tmp_basic / 'stock_basic.csv'
    _write_basic(p, [{'code': '601609.SH', 'market': 'SH', 'name': '国联证券', 'status': 'active'}])
    assert stocks_info.lookup_name('601609.SH') == '国联证券'


def test_lookup_market_existing_returns_market(tmp_basic):
    p = tmp_basic / 'stock_basic.csv'
    _write_basic(p, [
        {'code': '002475.SZ', 'market': 'SZ', 'name': '立讯精密', 'status': 'active'},
        {'code': '601609.SH', 'market': 'SH', 'name': '国联证券', 'status': 'active'},
    ])
    assert stocks_info.lookup_market('002475.SZ') == 'SZ'
    assert stocks_info.lookup_market('601609.SH') == 'SH'


def test_lookup_missing_returns_default(tmp_basic):
    """找不到的代码走默认值,不抛。"""
    p = tmp_basic / 'stock_basic.csv'
    _write_basic(p, [{'code': '601609.SH', 'market': 'SH', 'name': '国联证券', 'status': 'active'}])
    assert stocks_info.lookup_name('999999.SH') == ''
    assert stocks_info.lookup_market('999999.SH') == ''
    assert stocks_info.lookup_name('999999.SH', default='未知') == '未知'


def test_lookup_empty_name_returns_default(tmp_basic):
    """name 字段为空字符串(CSV 里真实存在但 name 是 ''):返回 default,不返回 ''。"""
    p = tmp_basic / 'stock_basic.csv'
    _write_basic(p, [{'code': '600000.SH', 'market': 'SH', 'name': '', 'status': 'unknown'}])
    assert stocks_info.lookup_name('600000.SH', default='?') == '?'
    assert stocks_info.lookup_market('600000.SH') == 'SH'   # market 不受 name 影响


def test_missing_csv_returns_empty_and_warns(tmp_basic):
    """stock_basic.csv 缺失 → 返回空 df + warn,不抛。"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        df = stocks_info.load_basic_df()
    assert df.empty
    assert any('stock_basic.csv 缺失' in str(wi.message) for wi in w)
    # 查询函数走默认值,不抛
    assert stocks_info.lookup_name('601609.SH') == ''


def test_lru_cache_reused_across_calls(tmp_basic):
    """同一进程多次查询只读一次 CSV(性能保证)。"""
    p = tmp_basic / 'stock_basic.csv'
    _write_basic(p, [{'code': '601609.SH', 'market': 'SH', 'name': '国联证券', 'status': 'active'}])

    # 第一次读 → 缓存
    stocks_info.lookup_name('601609.SH')
    # 修改磁盘文件,第二次查询应仍返回旧值(证明是缓存,不是重读)
    _write_basic(p, [{'code': '601609.SH', 'market': 'SH', 'name': '改名了', 'status': 'active'}])
    assert stocks_info.lookup_name('601609.SH') == '国联证券'   # 缓存命中

    # cache_clear 后才读新值
    stocks_info.load_basic_df.cache_clear()
    assert stocks_info.lookup_name('601609.SH') == '改名了'