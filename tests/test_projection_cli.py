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


def test_batch_process_one_lag0_writes_21_col_csv(tmp_path, monkeypatch):
    """process_one 默认 lag=0 写出 21 列 CSV(State_ 前缀 + 8 维度幅度量,
    2026-08-16 删除 State_Resi_Price 后从 22 列降为 21 列)。"""
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
    assert csv_df.shape[1] == 21
    # 2026-08-16:State_Resi_Price 已删除
    assert 'State_Resi_Price' not in csv_df.columns
    # 8 维度幅度量
    assert 'State_Stock_Magnitude' in csv_df.columns
    assert 'State_Index_Magnitude' in csv_df.columns
    assert 'State_Relative_Move' in csv_df.columns


def test_batch_process_one_lag1_writes_29_col_csv(tmp_path, monkeypatch):
    """process_one lag=1 写出 29 列 CSV(2026-08-16 后从 30 列降为 29 列),首行降;State_ 前缀。"""
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
    assert csv_df.shape[1] == 29
    assert 'State_Resi_Price' not in csv_df.columns
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


# ========================== --movement flag ==========================

def test_batch_process_one_with_movement_writes_extra_csv(tmp_path, monkeypatch):
    """--movement 开启时,process_one 除常规 CSV 外,额外产 movement_*.csv(行数 = 共同交易日 - 1,18 列,Move_ 前缀)。"""
    import projection_batch as pb_mod

    n = 6
    idx = pd.date_range('2026-07-01', periods=n, freq='D')
    df = pd.DataFrame({
        'Volume': np.linspace(1e6, 2e6, n),
        'Amount': np.linspace(1e10, 2e10, n),
        'Close':  np.linspace(20.0, 25.0, n),
    }, index=idx)
    pipe = _FakePipeline({'000001.SH': df, '002475.SZ': df})
    monkeypatch.setattr(pb_mod, 'P', pipe)
    monkeypatch.chdir(tmp_path)                # 隔离 data/projection/ 写入路径

    row = pb_mod.process_one(
        '002475.SZ', '立讯精密', days=10,
        prefer_industry=True, index_code='000001.SH',
        lag=0, movement=True,
    )
    assert row['status'] == 'ok', row['status']
    # 必有 movement_*.csv
    mv_csv = tmp_path / 'data' / 'projection' / 'movement_000001_002475.csv'
    assert mv_csv.exists(), f"movement CSV 未生成: {mv_csv}"
    mv_df = pd.read_csv(mv_csv)
    # 18 列:13 基础列 + Proj_Price + Resi_Price + 3 个幅度量(Move_ 前缀)
    assert mv_df.shape == (n - 1, 18), f"期望 {n-1} 行 × 18 列, 实际 {mv_df.shape}"
    assert 'Move_Delta_Vol_000001' in mv_df.columns
    assert 'Move_Delta_Vol_002475' in mv_df.columns
    assert 'Move_Proj_Coeff' in mv_df.columns
    assert 'Move_Dot_After' in mv_df.columns
    assert 'Move_Proj_Price' in mv_df.columns
    assert 'Move_Resi_Price' in mv_df.columns
    # 8 维度幅度量
    assert 'Move_Stock_Magnitude' in mv_df.columns
    assert 'Move_Index_Magnitude' in mv_df.columns
    assert 'Move_Relative_Move' in mv_df.columns


def test_batch_process_one_without_movement_does_not_write_movement_csv(tmp_path, monkeypatch):
    """--movement 默认关闭:不应产 movement_*.csv。"""
    import projection_batch as pb_mod

    n = 4
    idx = pd.date_range('2026-07-01', periods=n, freq='D')
    df = pd.DataFrame({
        'Volume': np.linspace(1e6, 2e6, n),
        'Amount': np.linspace(1e10, 2e10, n),
        'Close':  np.linspace(20.0, 25.0, n),
    }, index=idx)
    pipe = _FakePipeline({'000001.SH': df, '002475.SZ': df})
    monkeypatch.setattr(pb_mod, 'P', pipe)
    monkeypatch.chdir(tmp_path)

    row = pb_mod.process_one(
        '002475.SZ', '立讯精密', days=10,
        prefer_industry=True, index_code='000001.SH',
        lag=0, movement=False,
    )
    assert row['status'] == 'ok'
    assert not (tmp_path / 'data' / 'projection' / 'movement_000001_002475.csv').exists()


def test_single_movement_flag_recognized(monkeypatch):
    """projection_2d.py --movement 被 parse_args 正确接收。"""
    monkeypatch.setattr(sys, 'argv', [
        'projection_2d.py', '--code', '002475.SZ', '--days', '5',
        '--index', '000001.SH', '--movement',
    ])
    import importlib
    import projection_2d as p2d_mod
    importlib.reload(p2d_mod)
    # args.movement 为 True,MOVEMENT_PREFIX 也已定义
    assert p2d_mod.args.movement is True
    assert p2d_mod.MOVEMENT_PREFIX == 'projmv_'