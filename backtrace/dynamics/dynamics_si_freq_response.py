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