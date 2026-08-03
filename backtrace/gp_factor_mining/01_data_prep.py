# -*- coding: utf-8 -*-
"""
01_data_prep.py — TDX 取数 + 清洗 + 截面标准化 + 落盘

数据源:优先 TQ(沪深A股板块),失败回退到本地 *_daily.csv。
输出:gp_factor_mining/data/panel.parquet(长表,每行 = (date, code))
    列:date, code, Open, High, Low, Close, Volume, Amount,
        fwd_ret_20d(标签), cs_rank_close, cs_rank_volume, ...

用法:
    cd backtrace/gp_factor_mining
    python 01_data_prep.py
"""
import warnings
warnings.filterwarnings('ignore')

import sys
import os
from datetime import datetime, timedelta
import traceback

import numpy as np
import pandas as pd

from sklearn.preprocessing import QuantileTransformer

# ========================= 配置 =========================
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from importlib import import_module
cfg = import_module('00_config')

# ========================================================


# ---------- TQ 取数 ----------
def _init_path():
    """与 common.tsfresh_pipeline.init_tq_path 行为一致;优先 __file__,回退 sys.argv[0]/cwd"""
    if '__file__' in globals() and __file__:
        return os.path.abspath(__file__)
    return os.path.abspath(sys.argv[0]) if sys.argv else os.getcwd()


def fetch_from_tq(codes, start, end):
    """
    TQ 批量拉日线;TQ 失败不会抛(逐票 try/except),调用方拿 dict 自己处理
    参数:
      codes : list[str]   股票代码列表(含 .SH/.SZ)
      start : datetime    起始日
      end   : datetime    截止日
    返回:dict[code] -> DataFrame(OHLCV+A,DatetimeIndex,Close 已剔 <=0 与 NaN)
    """
    sys.path.insert(0, cfg.TQ_INIT_PATH)
    from tqcenter import tq
    tq.initialize(_init_path())

    df_real = tq.get_market_data(
        field_list=['Open', 'High', 'Low', 'Close', 'Volume', 'Amount'],
        stock_list=codes,
        start_time=start.strftime("%Y%m%d"),
        end_time=end.strftime("%Y%m%d"),
        dividend_type='front', period='1d', fill_data=True,
    )

    out = {}
    for c in codes:
        try:
            sub = pd.DataFrame({
                'Open':   pd.to_numeric(df_real['Open'][c],   errors='coerce'),
                'High':   pd.to_numeric(df_real['High'][c],   errors='coerce'),
                'Low':    pd.to_numeric(df_real['Low'][c],    errors='coerce'),
                'Close':  pd.to_numeric(df_real['Close'][c],  errors='coerce'),
                'Volume': pd.to_numeric(df_real['Volume'][c], errors='coerce'),
                'Amount': pd.to_numeric(df_real['Amount'][c], errors='coerce'),
            })
            sub = sub.replace([0, np.inf, -np.inf], np.nan).sort_index()
            sub = sub[sub['Close'].notna() & (sub['Close'] > 0)]
            if len(sub) > 0:
                out[c] = sub
        except Exception as e:
            print(f"[TQ] {c} 解析失败:{type(e).__name__}: {e}")
    tq.close()
    return out


def fetch_panel():
    """
    主入口:取数 + 清洗 + 构造标签 + 截面标准化 + 落盘。
    流程(按代码顺序):
      1. 拉数(TQ 优先,失败回退本地 *_daily.csv)
      2. 拼接长表 (date, code, OHLCVA)
      3. 股票池过滤(上市天数、壳股、ST、停牌)
      4. 构造未来 N 日收益标签
      5. 截面标准化(每日横截面 rank→zscore)
      6. 落盘到 DATA_DIR/panel.parquet
    返回:panel DataFrame(已在内存;同时落盘)
    """
    print("=" * 70)
    print("[01] 数据准备")
    print("=" * 70)

    end_dt   = datetime.strptime(cfg.TEST_END, "%Y-%m-%d")
    start_dt = datetime.strptime(cfg.DATA_FETCH_START, "%Y-%m-%d")
    codes    = None

    stock_data = {}

    # ---- TQ 优先 ----
    if cfg.USE_TQ:
        try:
            sys.path.insert(0, cfg.TQ_INIT_PATH)
            from tqcenter import tq
            tq.initialize(_init_path())
            codes = tq.get_stock_list_in_sector(cfg.TQ_SECTOR) or []
            print(f"[TQ] 板块「{cfg.TQ_SECTOR}」成分股 {len(codes)} 只")
            if not codes:
                raise RuntimeError(f"板块 {cfg.TQ_SECTOR} 拉不到")
            tq.close()

            # 分批拉(避免单次太多)
            BATCH = 500
            for i in range(0, len(codes), BATCH):
                batch = codes[i:i+BATCH]
                print(f"[TQ] 拉第 {i//BATCH+1}/{(len(codes)-1)//BATCH+1} 批 ({len(batch)} 只)...")
                sub = fetch_from_tq(batch, start_dt, end_dt)
                stock_data.update(sub)
            print(f"[TQ] 成功拉到 {len(stock_data)}/{len(codes)} 只股票")
        except Exception as e:
            print(f"[TQ] 拉取失败 ({type(e).__name__}: {e})")
            print("[TQ] 完整 traceback ↓")
            traceback.print_exc()
            print("[TQ] 自动回退到本地 CSV\n")
            stock_data = {}

    # ---- 本地 CSV 回退(若 TQ 失败,只用到 *_daily.csv 同目录的那些)----
    if not stock_data:
        base = cfg.GP_DIR
        for f in base.glob('*_daily.csv'):
            code = f.stem.replace('_daily', '').replace('_', '.')
            try:
                df = pd.read_csv(f, index_col=0, parse_dates=True).sort_index()
                if 'Amount' not in df.columns:
                    df['Amount'] = df['Volume'] * df['Close']
                df = df.replace([0, np.inf, -np.inf], np.nan)
                df = df[df['Close'].notna() & (df['Close'] > 0)]
                stock_data[code] = df
            except Exception as e:
                print(f"[CSV] {code} 加载失败:{e}")
        codes = list(stock_data.keys())
        print(f"[CSV] 本地回退拿到 {len(stock_data)} 只")

    if not stock_data:
        sys.exit("[FATAL] 无任何股票数据,请检查 TQ / CSV 路径")

    # ---- 拼长表 ----
    print(f"\n[02] 拼接长表...")
    rows = []
    for code, df in stock_data.items():
        tmp = df.copy()
        tmp['code'] = code
        tmp = tmp.reset_index().rename(columns={'index': 'date'})
        rows.append(tmp)
    panel = pd.concat(rows, ignore_index=True)
    panel['date'] = pd.to_datetime(panel['date'])
    print(f"  原始长表: {panel.shape[0]:,} 行, "
          f"{panel['code'].nunique()} 只, "
          f"{panel['date'].min().date()} → {panel['date'].max().date()}")

    # ---- 股票池过滤 ----
    print(f"\n[03] 股票池过滤...")
    n0 = panel['code'].nunique()

    # 上市满 60 天(用数据可用长度近似上市天数)
    listed_days = panel.groupby('code').size()
    keep_codes = listed_days[listed_days >= cfg.MIN_LIST_DAYS].index
    panel = panel[panel['code'].isin(keep_codes)]

    # 收盘价 < 2 元剔除(壳股,可选)
    panel = panel[panel['Close'] >= cfg.MIN_PRICE]

    # ST 过滤:TQ 名字里带 ST / *ST(本地 CSV 无名字信息,跳过)
    if cfg.EXCLUDE_ST and cfg.USE_TQ:
        try:
            sys.path.insert(0, cfg.TQ_INIT_PATH)
            from tqcenter import tq
            tq.initialize(_init_path())
            st_codes = set()
            for c in panel['code'].unique():
                try:
                    info = tq.get_stock_info(c) or {}
                    name = str(info.get('name', ''))
                    if 'ST' in name or '退' in name:
                        st_codes.add(c)
                except Exception:
                    pass
            tq.close()
            panel = panel[~panel['code'].isin(st_codes)]
            print(f"  剔 ST/*ST {len(st_codes)} 只")
        except Exception as e:
            print(f"  [ST 过滤跳过] {e}")

    # 停牌剔除:当日 Volume==0 或 Close 缺失 → 整个交易日作废
    if cfg.EXCLUDE_SUSPEND:
        before = len(panel)
        panel = panel[panel['Volume'].notna() & (panel['Volume'] > 0)]
        print(f"  剔停牌/异常行 {before - len(panel):,}")

    n1 = panel['code'].nunique()
    print(f"  过滤后:{n1}/{n0} 只股票, 长表 {len(panel):,} 行")

    # ---- 构造标签:未来 20 日收益 ----
    print(f"\n[04] 构造标签(fwd_ret_{cfg.HOLD_PERIOD}d)...")
    panel = panel.sort_values(['code', 'date']).reset_index(drop=True)
    panel[cfg.LABEL_NAME] = (
        panel.groupby('code')['Close']
             .shift(-cfg.HOLD_PERIOD) / panel['Close'] - 1
    )

    # ---- 截面标准化(每日横截面 rank → zscore)----
    print(f"\n[05] 截面标准化({cfg.CS_STD_METHOD})...")
    num_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', cfg.LABEL_NAME]

    if cfg.CS_STD_METHOD == "rank_zscore":
        # 截面排序后映射到 N(0,1),抗极端值 + 抗量纲
        def _rank_z(s):
            r = s.rank(method='first', na_option='keep')
            return (r - r.mean()) / r.std()

        for c in num_cols:
            panel[f'cs_{c}'] = panel.groupby('date')[c].transform(_rank_z)

    elif cfg.CS_STD_METHOD == "rank_pct":
        for c in num_cols:
            panel[f'cs_{c}'] = panel.groupby('date')[c].rank(method='first', pct=True)

    elif cfg.CS_STD_METHOD == "zscore":
        for c in num_cols:
            mu = panel.groupby('date')[c].transform('mean')
            sd = panel.groupby('date')[c].transform('std')
            panel[f'cs_{c}'] = (panel[c] - mu) / sd

    print(f"  已生成 cs_* 列:{[c for c in panel.columns if c.startswith('cs_')]}")

    # ---- 落盘 ----
    out_path = cfg.DATA_DIR / "panel.parquet"
    panel.to_parquet(out_path, index=False)
    print(f"\n[OK] 长表已落盘 → {out_path}")
    print(f"     shape={panel.shape}, "
          f"日期={panel['date'].min().date()} ~ {panel['date'].max().date()}, "
          f"股票={panel['code'].nunique()}")

    return panel


if __name__ == "__main__":
    panel = fetch_panel()
    print("\n=== head ===")
    print(panel.head())
    print("\n=== describe(cs_Close) ===")
    print(panel['cs_Close'].describe())