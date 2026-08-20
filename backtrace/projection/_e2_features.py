# -*- coding: utf-8 -*-
"""V0.2-E1 E2 helper: per-stock β / vol / liquidity extraction from daily data.

Loads `data/stocks/{code}_daily.csv` + market index `data/indices/{market_code}_daily.csv`,
computes features over the last `train_days` of common dates.

Used by `v0_2_e2_cross_sectional_q.py` for cross-sectional analysis.
"""
import os

import numpy as np
import pandas as pd


# Core dependencies only — helper is intentionally lightweight (no common.tsfresh_pipeline).
MARKET_INDEX = {
    'SH': '000001.SH',
    'SZ': '399001.SZ',
}


def _stock_filename(code: str) -> str:
    return f'{code.replace(".", "_")}_daily.csv'


def _index_filename(index_code: str) -> str:
    return f'{index_code.replace(".", "_")}_daily.csv'


def _load_daily(path: str) -> pd.DataFrame | None:
    """Load a daily CSV from the data/ cache. Returns None if missing.

    CSV format (uniform across stocks/ + indices/): index is the datetime column
    (no name), then Open/High/Low/Close/Volume/Amount. Use index_col=0 + parse_dates
    to match the convention used by backtrace/common/data_store.py.
    """
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()


def extract_features_one(
    code: str,
    stocks_dir: str = 'data/stocks',
    indices_dir: str = 'data/indices',
    train_days: int = 168,
) -> dict | None:
    """Extract β_market / stock_volatility / liquidity for a single stock.

    Returns None if insufficient data (<100 common dates) or missing files.
    """
    market = 'SH' if code.endswith('.SH') else 'SZ'
    market_code = MARKET_INDEX[market]

    stock_path = os.path.join(stocks_dir, _stock_filename(code))
    index_path = os.path.join(indices_dir, _index_filename(market_code))

    stock_df = _load_daily(stock_path)
    index_df = _load_daily(index_path)
    if stock_df is None or index_df is None:
        return None

    common = stock_df.index.intersection(index_df.index)
    if len(common) < 100:
        return None

    train_dates = common[-train_days:]
    # Align stock + market returns on the SAME dates, then drop NaN jointly.
    # Independent dropna() misaligns β when stock has NaN closes that market doesn't
    # (or vice versa): the two arrays become different lengths and np.cov raises;
    # if they happen to match in length, position i of stock_ret pairs with position
    # i of market_ret but the underlying dates are different — silent β corruption.
    aligned = pd.DataFrame({
        'stock_ret': stock_df.loc[train_dates, 'Close'].pct_change(),
        'market_ret': index_df.loc[train_dates, 'Close'].pct_change(),
    }).dropna()
    stock_ret = aligned['stock_ret'].values
    market_ret = aligned['market_ret'].values

    if len(stock_ret) < 50:
        return None  # too few paired observations for stable β

    cov = np.cov(stock_ret, market_ret)
    beta_market = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else float('nan')
    stock_vol = float(stock_ret.std(ddof=1))
    liquidity = float(stock_df.loc[train_dates, 'Volume'].median())

    return {
        'code': code,
        'beta_market': beta_market,
        'stock_volatility': stock_vol,
        'liquidity': liquidity,
    }


def extract_features_all(
    codes: list[str],
    stocks_dir: str = 'data/stocks',
    indices_dir: str = 'data/indices',
    train_days: int = 168,
    cache_path: str = 'data/projection_v01_e2/_features_cache.csv',
) -> pd.DataFrame:
    """Extract features for a list of codes; cache result to `cache_path`.

    Skips codes that fail (insufficient data, missing files). Returns DataFrame.
    """
    # Check cache first
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        cached_codes = set(cached['code'].astype(str).tolist())
        missing = [c for c in codes if c not in cached_codes]
        if not missing:
            return cached[cached['code'].astype(str).isin(codes)].reset_index(drop=True)

    rows = []
    for code in codes:
        feats = extract_features_one(code, stocks_dir, indices_dir, train_days)
        if feats is not None:
            rows.append(feats)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Save cache (only newly extracted rows, append if cache exists)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        existing = pd.read_csv(cache_path)
        combined = pd.concat(
            [existing, df[~df['code'].isin(existing['code'])]],
            ignore_index=True,
        )
    else:
        combined = df
    combined.to_csv(cache_path, index=False, encoding='utf-8')
    return df
