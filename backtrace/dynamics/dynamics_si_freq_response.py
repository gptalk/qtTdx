"""v5.3 — Real SI Frequency Response 时序动画 overlay.

读 parameter_fit --rolling-time 输出 (kc_estimates_time.csv),按 asof_date 切片
+ 行业聚合 + top-N 选取,通过 plotly animation_frame 联动多帧 Bode overlay。
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

REQUIRED_COLUMNS = ('code', 'index_code', 'asof_date', 'k_hat', 'c_hat', 'status', 'n_valid_days')
RAMP_UP_DAYS = 192  # 沿用 v4.9


def load_kc_time_series(csv_path: str) -> pd.DataFrame:
    """读 parameter_fit --rolling-time 输出 kc_estimates_time.csv。

    必需列:code, index_code, asof_date, k_hat, c_hat, status, n_valid_days
    过滤:status='ok' AND n_valid_days >= 192 (ramp-up)

    Raises:
        FileNotFoundError: csv_path 不存在
        ValueError: 缺必需列(错误信息列出缺失列名)
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'kc_estimates_time.csv 缺必需列: {missing}')
    return df[(df['status'] == 'ok') & (df['n_valid_days'] >= RAMP_UP_DAYS)].copy()


def aggregate_by_industry_per_date(
    df: pd.DataFrame,
    dates: list,
    group_col: str = 'index_code',
    agg: str = 'median',
) -> dict:
    """按 (asof_date, group_col) 聚合 (k̂, ĉ),每片一个 DataFrame。

    Args:
        df: load_kc_time_series 输出
        dates: asof_date 列表 (YYYY-MM-DD str)
        group_col: 分组列(默认 'index_code')
        agg: 聚合方法(目前仅 'median')

    Returns:
        {asof_date: DataFrame [group_col, n_stocks, k_hat, c_hat]},每片按 group_col 排序
    """
    if agg != 'median':
        raise ValueError(f'agg={agg!r} 不支持,目前仅 median')

    out = {}
    for date in dates:
        slice_df = df[df['asof_date'] == date]
        if slice_df.empty:
            continue
        grouped = slice_df.groupby(group_col).agg(
            n_stocks=('code', 'count'),
            k_hat=('k_hat', 'median'),
            c_hat=('c_hat', 'median'),
        ).reset_index().sort_values(group_col).reset_index(drop=True)
        out[date] = grouped
    return out


import numpy as np


def select_top_n_per_date(
    per_date_dfs: dict,
    criterion: str = 'by_n_stocks',
    n: int = 5,
    group_col: str = 'index_code',
) -> list:
    """每个 asof_date 选 top-N industries,转动画 overlay 格式。

    Args:
        per_date_dfs: aggregate_by_industry_per_date 输出 {date: DataFrame}
        criterion: 'by_n_stocks' / 'by_c_over_k' / 'by_k_over_c'
        n: top N(每个 date 最多选 n 个行业)

    Returns:
        [(asof_date, k̂, ĉ, "Industry {group_col}"), ...],按 date 排序
    """
    if criterion not in ('by_n_stocks', 'by_c_over_k', 'by_k_over_c'):
        raise ValueError(f'criterion={criterion!r} 不支持')

    pairs = []
    for date in sorted(per_date_dfs.keys()):
        df = per_date_dfs[date]
        if criterion == 'by_n_stocks':
            sorted_df = df.sort_values('n_stocks', ascending=False).head(n)
        elif criterion == 'by_c_over_k':
            df_copy = df.copy()
            df_copy['ratio'] = df_copy['c_hat'] / df_copy['k_hat'].replace(0, np.nan)
            sorted_df = df_copy.sort_values('ratio', ascending=False, na_position='last').head(n)
        else:  # by_k_over_c
            df_copy = df.copy()
            df_copy['ratio'] = df_copy['k_hat'] / df_copy['c_hat'].replace(0, np.nan)
            sorted_df = df_copy.sort_values('ratio', ascending=False, na_position='last').head(n)
        for _, row in sorted_df.iterrows():
            pairs.append((
                date,
                float(row['k_hat']),
                float(row['c_hat']),
                f'Industry {row[group_col]}',
            ))
    return pairs