# -*- coding: utf-8 -*-
"""
拉取沪深全市场 + 申万二级行业指数 + 两大盘指数的日线,落盘到仓库根 data/。

职责边界:本模块只做「编排」—— universe、分批、重试、进度。
落盘一律经由 common.data_store,自己不拼任何路径。

用法:
  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py            # 全量
  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py --limit 20 # 冒烟
  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py --force    # 忽略 manifest 重拉
  PYTHONIOENCODING=utf-8 python backtrace/data_fetch/fetch_daily.py --probe    # 只探测 TQ 列表接口
"""
import os
import sys

import pandas as pd

BACKTRACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKTRACE_DIR not in sys.path:
    sys.path.insert(0, BACKTRACE_DIR)

# ========================= 配置 =========================
TRADING_DAYS = 500        # 每只票保留的交易日数
BATCH_SIZE = 250          # 每批喂给 get_market_data 的代码数
                          # 依据:CLAUDE.md 记录 6000 只 timeout、~600 只可行,250 留足余量
TRADING_DAY_RATIO = 0.670 # 实测交易日/自然日占比(000001_SH_daily.csv 181 行 / 270 天)
CALENDAR_MARGIN = 1.05    # 自然日请求余量
INDEX_CODES = ['000001.SH', '399001.SZ']   # 上证综指 / 深证成分指数
SW2_LIST_ARG = '11'       # get_stock_list('11', list_type=1) -> 128 申万二级行业
FIELDS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount']
# ======================================================


def filter_st(items):
    """[{'Code','Name'}, ...] -> [code],剔除 ST/*ST/SST 与退市标的。

    条目可能是 None 或缺 Code(TQ 返回偶有脏数据),一律跳过。
    """
    out = []
    for it in items or []:
        if not it or not it.get('Code'):
            continue
        name = it.get('Name') or ''
        if 'ST' in name.upper() or '退' in name:
            continue
        out.append(it['Code'])
    return out


def chunked(seq, size=BATCH_SIZE):
    """把列表切成每块 size 个,末块可短。"""
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def calendar_days_for(trading_days=TRADING_DAYS):
    """交易日数 -> 需向 TQ 请求的自然日数。

    多请求的成本几乎为零(TQ 按区间返回),少拉却要整轮重来,所以宁可多留余量。
    """
    return int(trading_days / TRADING_DAY_RATIO * CALENDAR_MARGIN)


def trim_tail(df, n=TRADING_DAYS):
    """排序后取尾部 n 行。不足 n 行的原样返回 —— 次新股照收,不补齐、不丢弃。"""
    return df.sort_index().tail(n)