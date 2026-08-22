# -*- coding: utf-8 -*-
"""
本地日线缓存的唯一真相源:定义 CSV 放哪、叫什么、怎么读写。

约定/做法:
  - 读写共用 csv_path(),杜绝"写在 A、读在 B"的路径分家 bug
    (历史教训:_try_local_csv 读 backtrace/,而 CSV 实际写在 backtrace/outputs/,回退长期失效)
  - 纯文件 IO,不 import TQ —— 离线读取无需 TQ 客户端在场,也使本模块可独立测试
  - 原子写(.tmp + os.replace),中途 Ctrl-C 不留半截文件被下次当成有效缓存

磁盘布局:
  data/stocks/000001_SZ_daily.csv     沪深 A 股(去 ST/退市)
  data/sectors/880xxx_SH_daily.csv    申万二级 128 行业指数
  data/indices/000001_SH_daily.csv    上证综指 / 深证成指
  data/manifest.json                  每只票的行数/首末日期/拉取时间/失败原因

输入/输出:
  - 输入:股票代码 code、DataFrame
  - 输出:7 个公开函数

依赖:pandas, common.tsfresh_config

用法:
  from common import data_store
  data_store.save_daily('000001.SZ', df, 'stocks')
  df = data_store.load_daily('000001.SZ')      # 跨 kind 查找
"""
import json
import os

import pandas as pd

from common import tsfresh_config as C

# 模块级变量而非常量:测试用 monkeypatch.setattr(data_store, 'DATA_DIR', tmp) 重定向
DATA_DIR = C.DATA_DIR

KINDS = ('stocks', 'sectors', 'indices')

PERIODS = ('daily', '15m', '5m', '1m')

# CSV schema —— 与 backtrace/outputs/*_daily.csv 既有格式一致,不要改
COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']


def _filename(code, period='daily'):
    """000001.SH + 'daily' -> '000001_SH_daily.csv'  (legacy)
       000001.SH + '5m'    -> '000001_SH_5m.csv'
    """
    if period == 'daily':
        return f"{code.replace('.', '_')}_daily.csv"
    if period not in PERIODS:
        raise ValueError(f"period 必须是 {PERIODS} 之一,收到 {period!r}")
    return f"{code.replace('.', '_')}_{period}.csv"


def csv_path(code, period='daily', kind='stocks'):
    """路径的唯一真相。读和写都必须经过这里。"""
    if period not in PERIODS:
        raise ValueError(f"period 必须是 {PERIODS} 之一,收到 {period!r}")
    if kind not in KINDS:
        raise ValueError(f"kind 必须是 {KINDS} 之一,收到 {kind!r}")
    return os.path.join(DATA_DIR, kind, _filename(code, period))


def save_daily(code, df, kind='stocks'):
    """原子写(先 .tmp 再 os.replace),返回落盘路径。"""
    path = csv_path(code, kind=kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    df.to_csv(tmp, encoding='utf-8')
    os.replace(tmp, path)
    return path


def save_df(code, df, period='daily', kind='stocks'):
    """通用 period-aware 写盘。daily 时等价 save_daily。"""
    return save_daily(code, df, kind) if period == 'daily' else _save_with_period(code, df, period, kind)


def _save_with_period(code, df, period, kind):
    path = csv_path(code, period, kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    df.to_csv(tmp, encoding='utf-8')
    os.replace(tmp, path)
    return path


def load_df(code, period='daily'):
    """跨 kind 查找(stocks → sectors → indices);period 与 kind 都参与路径。"""
    for kind in KINDS:
        p = csv_path(code, period, kind)
        if os.path.exists(p):
            return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    return None


def load_daily(code):
    """按 stocks -> sectors -> indices 顺序查找;都没有返回 None。

    跨目录查找是必要的:调用方(如 _try_local_csv)只有 code,不知道它是个股还是指数。
    """
    for kind in KINDS:
        p = csv_path(code, kind=kind)
        if os.path.exists(p):
            return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    return None


def has_daily(code):
    """任一 kind 目录下存在该 code 的 CSV(供断点续传判断)"""
    return any(os.path.exists(csv_path(code, kind=k)) for k in KINDS)


def manifest_path():
    return os.path.join(DATA_DIR, 'manifest.json')


def load_manifest():
    """不存在时返回空骨架,调用方无需处理 None。"""
    p = manifest_path()
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'generated_at': None, 'trading_days': None, 'entries': {}}


def save_manifest(man):
    """原子写,返回落盘路径。"""
    p = manifest_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    return p