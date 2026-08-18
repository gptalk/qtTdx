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