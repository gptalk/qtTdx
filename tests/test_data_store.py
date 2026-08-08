# -*- coding: utf-8 -*-
import os

import pandas as pd
import pytest

from common import data_store


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    """把缓存根指向临时目录,避免测试污染真实 data/"""
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


def test_csv_path_uses_underscore_and_kind_subdir(tmp_store):
    p = data_store.csv_path('000001.SH', 'indices')
    assert p.endswith(os.path.join('indices', '000001_SH_daily.csv'))


def test_csv_path_rejects_unknown_kind(tmp_store):
    with pytest.raises(ValueError):
        data_store.csv_path('000001.SH', 'bogus')


def test_save_then_load_roundtrip(tmp_store, sample_df):
    data_store.save_daily('000001.SZ', sample_df, 'stocks')
    got = data_store.load_daily('000001.SZ')
    pd.testing.assert_frame_equal(got, sample_df)


def test_load_searches_all_kinds(tmp_store, sample_df):
    data_store.save_daily('399001.SZ', sample_df, 'indices')
    assert data_store.load_daily('399001.SZ') is not None


def test_load_missing_returns_none(tmp_store):
    assert data_store.load_daily('999999.SZ') is None


def test_has_daily(tmp_store, sample_df):
    assert data_store.has_daily('600000.SH') is False
    data_store.save_daily('600000.SH', sample_df, 'stocks')
    assert data_store.has_daily('600000.SH') is True


def test_read_write_share_one_path(tmp_store, sample_df):
    """防回归:写入路径必须正是 load 查找的路径 —— 这正是当前 bug 的根因"""
    written = data_store.save_daily('600000.SH', sample_df, 'stocks')
    assert os.path.exists(written)
    assert data_store.load_daily('600000.SH') is not None


def test_save_leaves_no_tmp_file(tmp_store, sample_df):
    data_store.save_daily('600000.SH', sample_df, 'stocks')
    assert list(tmp_store.rglob('*.tmp')) == []


def test_manifest_roundtrip(tmp_store):
    man = data_store.load_manifest()
    assert man['entries'] == {}
    man['entries']['000001.SZ'] = {'kind': 'stocks', 'rows': 500, 'status': 'ok'}
    data_store.save_manifest(man)
    assert data_store.load_manifest()['entries']['000001.SZ']['rows'] == 500


def test_manifest_leaves_no_tmp_file(tmp_store):
    data_store.save_manifest(data_store.load_manifest())
    assert list(tmp_store.rglob('*.tmp')) == []