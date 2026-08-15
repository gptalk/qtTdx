# -*- coding: utf-8 -*-
"""fetch_daily.build_stock_universe 单测。

策略:
  - 用 fake tq + monkeypatch 把 DATA_DIR 切到 tmp
  - 直接调 build_stock_universe,验证它:
    1) 写 union.csv(去重、含全部代码)
    2) 写 stock_basic.csv(4 列)
    3) kept 仅含 status='active' 的代码
    4) 北证代码被源头剔除(.BJ 在 sector members 阶段已剔除)
"""
import os
import sys

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTRACE = os.path.join(REPO, 'backtrace')
DATA_FETCH = os.path.join(BACKTRACE, 'data_fetch')
if BACKTRACE not in sys.path:
    sys.path.insert(0, BACKTRACE)
if DATA_FETCH not in sys.path:
    sys.path.insert(0, DATA_FETCH)


class _FakeTq:
    """最小化的 TQ 替身,只覆盖 build_stock_universe 调到的两个方法。

    TQ 实际返回字段名是 'Name'(大写 N),实测于 2026-08-15。
    测试桩也用 'Name' 反映真实行为;容错 'name' 由 build_basic 内部保证。
    """

    def __init__(self, members_by_sector, info_by_code):
        self._members = members_by_sector   # {sector_code: [member_code, ...]}
        self._info = info_by_code           # {code: {'Name': '...'}, ...}

    def get_stock_list_in_sector(self, sector_code, block_type=0):
        return self._members.get(sector_code, [])

    def get_stock_info(self, code):
        return self._info.get(code, {})


@pytest.fixture
def tmp_data(monkeypatch, tmp_path):
    """切 DATA_DIR 到 tmp,清掉 lru_cache 防止 stocks_info 残留。"""
    import fetch_daily as fd_mod
    monkeypatch.setattr(fd_mod.C, 'DATA_DIR', str(tmp_path))
    # stocks_info 的 BASIC_CSV 是模块级常量(不是 C.DATA_DIR 派生),需手动重定位
    from common import stocks_info
    monkeypatch.setattr(stocks_info, 'BASIC_CSV', str(tmp_path / 'stock_basic.csv'))
    stocks_info.load_basic_df.cache_clear()
    return tmp_path


def _run_build(fd_mod, tq, sectors, sector_names):
    """直接调 build_stock_universe,过滤 union.csv 的 side effect。"""
    return fd_mod.build_stock_universe(tq, sectors, sector_names)


def test_build_stock_universe_writes_union_and_basic_and_filters(tmp_data):
    """完整链路:5 只 active + 1 ST + 1 退市,应保留 5 只。"""
    import fetch_daily as fd_mod

    # 2 个行业,sector 阶段已剔除北证
    sectors = ['881001.SH', '881002.SH']
    sector_names = {'881001.SH': '行业A', '881002.SH': '行业B'}
    members_by_sector = {
        '881001.SH': ['600000.SH', '600001.SH', 'ST.SH'],
        '881002.SH': ['000001.SZ', '000002.SZ', '退.SZ'],
    }
    # TQ 实际返回 'Name'(大写 N),实测于 2026-08-15
    info_by_code = {
        '600000.SH': {'Name': '浦发银行'},
        '600001.SH': {'Name': '邯郸钢铁'},
        'ST.SH':     {'Name': 'ST华联'},          # 应被 status='st' 过滤
        '000001.SZ': {'Name': '平安银行'},
        '000002.SZ': {'Name': '万科A'},
        '退.SZ':     {'Name': '退市某'},          # 应被 status='delisted' 过滤
    }
    tq = _FakeTq(members_by_sector, info_by_code)

    kept = _run_build(fd_mod, tq, sectors, sector_names)

    # 1) 保留 5 只 active
    assert set(kept) == {'600000.SH', '600001.SH', '000001.SZ', '000002.SZ'}, (
        f"ST.SH / 退.SZ 应被 status 过滤,实际 kept={kept}"
    )

    # 2) union.csv 写出来了
    union_path = tmp_data / 'sw2' / 'union.csv'
    assert union_path.exists(), f"union.csv 未生成: {union_path}"
    union_df = pd.read_csv(union_path, dtype={'code': str})
    # union 含全部 6 只(ST 过滤前)
    assert len(union_df) == 6
    assert set(union_df['code']) == {'600000.SH', '600001.SH', 'ST.SH',
                                     '000001.SZ', '000002.SZ', '退.SZ'}

    # 3) stock_basic.csv 写出来了,且 status 正确
    basic_path = tmp_data / 'stock_basic.csv'
    assert basic_path.exists(), f"stock_basic.csv 未生成: {basic_path}"
    basic_df = pd.read_csv(basic_path, dtype={'code': str})
    assert set(basic_df.columns) == {'code', 'market', 'name', 'status'}
    assert set(basic_df['code']) == set(union_df['code'])
    status_by_code = dict(zip(basic_df['code'], basic_df['status']))
    assert status_by_code['600000.SH'] == 'active'
    assert status_by_code['ST.SH'] == 'st'
    assert status_by_code['退.SZ'] == 'delisted'
    # market 派生自 code
    assert status_by_code['600000.SH'] == 'active'  # warm
    assert basic_df.set_index('code').at['600000.SH', 'market'] == 'SH'
    assert basic_df.set_index('code').at['000001.SZ', 'market'] == 'SZ'


def test_build_stock_universe_strips_bj_at_sector_stage(tmp_data):
    """.BJ 代码在 sector members 阶段就被 is_bj 剔除(源头),不进 union/stock_basic。

    这是 fetch_daily.build_stock_universe:175 设计的源头过滤,
    避免 union.csv / stock_basic.csv 出现 .BJ,北证交易规则不同策略不覆盖。
    """
    import fetch_daily as fd_mod

    sectors = ['881001.SH']
    sector_names = {'881001.SH': '行业A'}
    members_by_sector = {'881001.SH': ['400001.BJ', '600000.SH']}
    info_by_code = {
        '400001.BJ': {'Name': '北证某'},
        '600000.SH': {'Name': '上证某'},
    }
    tq = _FakeTq(members_by_sector, info_by_code)

    kept = _run_build(fd_mod, tq, sectors, sector_names)

    # .BJ 不进 kept
    assert kept == ['600000.SH'], f".BJ 不应进 kept,实际 {kept}"
    # .BJ 也不进 union.csv(源头剔除)
    union_df = pd.read_csv(tmp_data / 'sw2' / 'union.csv', dtype={'code': str})
    assert '400001.BJ' not in set(union_df['code'])
    assert set(union_df['code']) == {'600000.SH'}


def test_build_stock_universe_handles_empty_name(tmp_data):
    """get_stock_info 返回空 name 时,status='unknown',不归 active。"""
    import fetch_daily as fd_mod

    sectors = ['881001.SH']
    sector_names = {'881001.SH': '行业A'}
    members_by_sector = {'881001.SH': ['600000.SH', '600001.SH']}
    info_by_code = {
        '600000.SH': {'Name': '正常股'},
        '600001.SH': {'Name': ''},            # 空名 → unknown
    }
    tq = _FakeTq(members_by_sector, info_by_code)

    kept = _run_build(fd_mod, tq, sectors, sector_names)

    assert kept == ['600000.SH']
    basic_df = pd.read_csv(tmp_data / 'stock_basic.csv', dtype={'code': str})
    status_by_code = dict(zip(basic_df['code'], basic_df['status']))
    assert status_by_code['600001.SH'] == 'unknown'


def test_filter_st_legacy_still_works(tmp_data):
    """filter_st 是文档/测试参考,逻辑保留,直接调用验证仍然正确。"""
    import fetch_daily as fd_mod
    items = [
        {'Code': '600000.SH', 'Name': '浦发'},
        {'Code': '600001.SH', 'Name': '*ST华联'},
        {'Code': '600002.SH', 'Name': '退市某'},
        {'Code': '600003.SH', 'Name': '正常'},
    ]
    kept = fd_mod.filter_st(items)
    assert kept == ['600000.SH', '600003.SH']


def test_build_stock_universe_accepts_lowercase_name_fallback(tmp_data):
    """TQ 真机返回 'Name'(大写),但代码容错 'name' 小写,防未来 TQ 改大小写。"""
    import fetch_daily as fd_mod

    sectors = ['881001.SH']
    sector_names = {'881001.SH': '行业A'}
    members_by_sector = {'881001.SH': ['600000.SH']}
    info_by_code = {
        '600000.SH': {'name': '小写name也OK'},   # 不是大写 Name
    }
    tq = _FakeTq(members_by_sector, info_by_code)

    kept = _run_build(fd_mod, tq, sectors, sector_names)
    assert kept == ['600000.SH']
    basic_df = pd.read_csv(tmp_data / 'stock_basic.csv', dtype={'code': str})
    assert basic_df.set_index('code').at['600000.SH', 'name'] == '小写name也OK'
    assert basic_df.set_index('code').at['600000.SH', 'status'] == 'active'