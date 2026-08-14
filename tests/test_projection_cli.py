# -*- coding: utf-8 -*-
"""projection_batch.py / projection_2d.py 的 CLI / process_one 单测。

策略: 不调 subprocess(DATA_DIR 在子进程里没法 monkeypatch,脆弱)。
改为:
- 用 argparse.Namespace 模拟 parse_args 输出,直接调 process_one
- monkeypatch tsfresh_pipeline.load_ohlcva 返回内存 DataFrame
- 临时切到 tmp cwd 避免污染 data/projection/
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTRACE = os.path.join(REPO, 'backtrace')
PROJECTION = os.path.join(BACKTRACE, 'projection')
if BACKTRACE not in sys.path:
    sys.path.insert(0, BACKTRACE)
if PROJECTION not in sys.path:
    sys.path.insert(0, PROJECTION)


class _FakePipeline:
    """最小化的 tsfresh_pipeline 替身:返回内存中的 DataFrame,不读 data/。"""

    def __init__(self, df_by_code):
        self._df = df_by_code

    def load_ohlcva(self, code, use_tq=False, verbose=False):
        return self._df.get(code)


def _fake_pair(n=5):
    idx = pd.date_range('2026-07-01', periods=n, freq='D')
    base = pd.DataFrame({
        'Volume': np.linspace(1e6, 1.5e6, n),
        'Amount': np.linspace(1e7, 1.5e7, n),
        'Close':  np.linspace(100, 110, n),
    }, index=idx)
    return base


def test_batch_process_one_lag0_writes_19_col_csv(tmp_path, monkeypatch):
    """process_one 默认 lag=0 写出 19 列 CSV。"""
    # 切 cwd 到 tmp,避免污染 data/projection/
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('projection_batch.CSV_OUT_DIR', str(tmp_path))

    df = _fake_pair(5)
    fake_pipe = _FakePipeline({'000001.SH': df.copy(), '002475.SZ': df.copy()})
    # patch P 引用本身(在 projection_batch 模块里是 tsfresh_pipeline 别名)
    import projection_batch as pb_mod
    monkeypatch.setattr(pb_mod, 'P', fake_pipe)

    row = pb_mod.process_one(
        stock_code='002475.SZ', stock_name='立讯精密',
        days=5, prefer_industry=False, index_code='000001.SH', lag=0,
    )
    assert row['status'] == 'ok', row
    assert row['rows'] == 5
    assert os.path.exists(row['csv_path'])
    csv_df = pd.read_csv(row['csv_path'])
    assert csv_df.shape[1] == 19
    assert 'Resi_Price' in csv_df.columns


def test_batch_process_one_lag1_writes_27_col_csv(tmp_path, monkeypatch):
    """process_one lag=1 写出 27 列 CSV,首行降。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('projection_batch.CSV_OUT_DIR', str(tmp_path))

    df = _fake_pair(5)
    fake_pipe = _FakePipeline({'000001.SH': df.copy(), '002475.SZ': df.copy()})
    import projection_batch as pb_mod
    monkeypatch.setattr(pb_mod, 'P', fake_pipe)

    row = pb_mod.process_one(
        stock_code='002475.SZ', stock_name='立讯精密',
        days=5, prefer_industry=False, index_code='000001.SH', lag=1,
    )
    assert row['status'] == 'ok', row
    assert row['rows'] == 4, f"5 行 - 首行 = 4 行,实际 {row['rows']}"
    csv_df = pd.read_csv(row['csv_path'])
    assert csv_df.shape[1] == 27
    assert 'Resi_Price' in csv_df.columns
    assert 'Vol_000001_prev_raw' in csv_df.columns
    assert 'Vol_000001_prev_norm' in csv_df.columns


def test_single_2d_does_not_set_two_day_vec_by_default(monkeypatch):
    """parse_args 默认解析后 two_day_vec=False。"""
    import importlib
    # 必须在 import projection_2d 前 patch argv,否则 import 时 parse_args 就已 SystemExit
    monkeypatch.setattr(sys, 'argv', ['projection_2d.py'])
    import projection_2d as p2d_mod
    importlib.reload(p2d_mod)  # re-run parse_args with patched argv
    assert not p2d_mod.TWO_DAY_VEC
    assert p2d_mod.FILE_PREFIX == 'proj2d_'
    assert p2d_mod.LAG == 0


def test_single_two_day_vec_sets_4d_prefix(tmp_path, monkeypatch):
    """parse_args 收到 --two-day-vec 后 FILE_PREFIX='proj2d_4d_' 且 LAG=1。"""
    monkeypatch.setattr(sys, 'argv', [
        'projection_2d.py', '--code', '002475.SZ',
        '--name', '立讯精密', '--days', '5', '--index', '000001.SH',
        '--two-day-vec',
    ])
    import importlib
    import projection_2d as p2d_mod
    importlib.reload(p2d_mod)
    assert p2d_mod.TWO_DAY_VEC is True
    assert p2d_mod.FILE_PREFIX == 'proj2d_4d_'
    assert p2d_mod.LAG == 1