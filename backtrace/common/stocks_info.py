# -*- coding: utf-8 -*-
"""股票基础信息反查表(code → name / market / status)。

数据源:data/stock_basic.csv,由 backtrace/data_fetch/fetch_stock_basic.py 生成。
设计目标:让所有脚本在不依赖 TQ 客户端、不重复硬编码默认值的前提下,
        通过代码拿中文名 / 交易所缩写。

输入/输出:
  - 输入:股票代码 code(如 '002475.SZ')
  - 输出:三个公开函数
      load_basic_df()           → pd.DataFrame (index=code, columns=market/name/status)
      lookup_name(code, ...)    → str
      lookup_market(code, ...)  → str

缓存策略:
  - load_basic_df 用 functools.lru_cache(maxsize=1)
  - 整个进程生命周期内 CSV 只读一次;更新 stock_basic.csv 后需重启脚本
  - 文件缺失:返回空 DataFrame + warn,不抛(让查询函数走默认值路径)

用法:
  from common import stocks_info
  stocks_info.lookup_name('601609.SH')     # '国联证券'
  stocks_info.lookup_market('002475.SZ')   # 'SZ'
"""
import functools
import os
import warnings

import pandas as pd

from common import tsfresh_config as C

BASIC_CSV = os.path.join(C.DATA_DIR, 'stock_basic.csv')

_EMPTY = pd.DataFrame(columns=['market', 'name', 'status']).astype({
    'market': 'object',
    'name': 'object',
    'status': 'object',
})


@functools.lru_cache(maxsize=1)
def load_basic_df() -> pd.DataFrame:
    """读 stock_basic.csv,index=code,缺失返回空表。"""
    if not os.path.exists(BASIC_CSV):
        warnings.warn(
            f"{BASIC_CSV} 缺失 — 查询函数将返回默认值。"
            f"运行: PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_stock_basic.py",
            stacklevel=2,
        )
        return _EMPTY.copy()
    df = pd.read_csv(
        BASIC_CSV,
        dtype={'code': str, 'name': str, 'market': str, 'status': str},
        keep_default_na=False,   # 空名保持 '' 不变成 NaN
    )
    df.set_index('code', inplace=True)
    return df


def lookup_name(code: str, default: str = '') -> str:
    """code → 中文名。找不到返回 default(默认空串)。"""
    df = load_basic_df()
    if code in df.index:
        return df.at[code, 'name'] or default
    return default


def lookup_market(code: str, default: str = '') -> str:
    """code → 交易所缩写(SH / SZ / BJ)。找不到返回 default。"""
    df = load_basic_df()
    if code in df.index:
        return df.at[code, 'market'] or default
    return default