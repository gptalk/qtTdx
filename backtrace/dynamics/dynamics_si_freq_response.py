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


import os
import sys
import argparse
import numpy as np
import plotly.graph_objects as go

# 让 from backtrace.dynamics... import 能找到包
BACKTRACE_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKTRACE_PARENT not in sys.path:
    sys.path.insert(0, BACKTRACE_PARENT)


DEFAULT_OMEGA_GRID = np.linspace(0.001, np.pi, 200)
DEFAULT_TOP_N = 5
DEFAULT_MAX_DATES = 12
HTML_OUT = 'backtrace/outputs/dynsys_si_freq_response.html'
SUMMARY_OUT = 'backtrace/outputs/dynsys_si_freq_response_summary.txt'
PAIRS_OUT = 'data/dynamics/si_freq_response_pairs.csv'

# Reuse v5.1 / v5.2 zero-modification helpers
from backtrace.dynamics.dynamics_forced_response import natural_frequency, magnitude_phase


def build_animated_overlay_html(
    pairs_per_date: list,
    omega_grid: np.ndarray,
    output_path: str,
    title: str = 'Industry G(ω) Frequency Response — Time Series',
) -> None:
    """构建 plotly 动画 slider:每帧一个 asof_date,每帧 N 条 industry Bode 曲线。

    Args:
        pairs_per_date: [(asof_date, k̂, ĉ, label), ...] from select_top_n_per_date
        omega_grid: 共享 ω 网格(np.ndarray,默认 linspace(0.001, π, 200))
        output_path: HTML 输出路径
        title: 图表标题

    Raises:
        ValueError: pairs_per_date 为空
    """
    if not pairs_per_date:
        raise ValueError('pairs_per_date 为空,无法构建动画')

    # 按 date 分组,每帧 N 条 trace
    dates = sorted(set(p[0] for p in pairs_per_date))
    initial_date = dates[0]
    initial_traces = [
        go.Scatter(
            x=omega_grid.tolist(),
            y=magnitude_phase(omega_grid * 1j, p[1], p[2])[0].tolist(),
            mode='lines',
            name=p[3],
        )
        for p in pairs_per_date if p[0] == initial_date
    ]

    fig = go.Figure(data=initial_traces)

    frames = []
    for date in dates:
        frame_traces = [
            go.Scatter(
                x=omega_grid.tolist(),
                y=magnitude_phase(omega_grid * 1j, p[1], p[2])[0].tolist(),
                mode='lines',
                name=p[3],
            )
            for p in pairs_per_date if p[0] == date
        ]
        frames.append(go.Frame(data=frame_traces, name=date))

    fig.frames = frames

    # Slider
    slider_steps = [
        dict(
            method='animate',
            args=[[date], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': True}}],
            label=date,
        )
        for date in dates
    ]

    # Play/Pause button
    play_button = dict(
        label='Play',
        method='animate',
        args=[None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
    )
    pause_button = dict(
        label='Pause',
        method='animate',
        args=[[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
    )

    fig.update_layout(
        title=title,
        xaxis_title='ω (rad/day)',
        yaxis_title='|H(jω)| dB',
        updatemenus=[dict(
            type='buttons', showactive=False, y=1.15, x=0.5, xanchor='center',
            buttons=[play_button, pause_button],
        )],
        sliders=[dict(active=0, steps=slider_steps, x=0.1, len=0.9, xanchor='left',
                      y=0, yanchor='top', currentvalue=dict(prefix='asof_date: ', visible=True))],
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')


def write_animated_summary_txt(
    pairs_per_date: list,
    dates: list,
    output_path: str,
) -> None:
    """写 UTF-8 中文业务解读:每个 asof_date 一段(top-N industries + 业务解读)。"""
    from collections import defaultdict
    by_date = defaultdict(list)
    for date, k, c, label in pairs_per_date:
        by_date[date].append((k, c, label))

    lines = ['# Industry G(ω) 时序动画 — 业务解读', '']
    for date in dates:
        if date not in by_date:
            continue
        lines.append(f'## {date}')
        for k, c, label in by_date[date]:
            omega_n = natural_frequency(k, c)
            regime = '过阻尼 (低通过滤器)' if c * c > 4 * k else ('临界阻尼' if abs(c * c - 4 * k) < 1e-6 else '欠阻尼 (有共振)')
            lines.append(f'  - {label}: k̂={k:.4f}, ĉ={c:.4f}, ω_n={omega_n:.4f}, {regime}')
        lines.append('')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def write_animated_pairs_csv(pairs_per_date: list, output_path: str) -> None:
    """写 UTF-8-sig 审计 CSV:每个 (asof_date, industry) 一行 + (k̂, ĉ)。"""
    rows = []
    for d, k, c, label in pairs_per_date:
        # label 形如 'Industry 801010',拆出 index_code
        idx_code = label.split(' ', 1)[1] if ' ' in label else label
        rows.append({
            'asof_date': d,
            'index_code': idx_code,
            'k_hat': k,
            'c_hat': c,
            'industry_label': label,
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='v5.3 — Real SI Frequency Response 时序动画 overlay',
    )
    p.add_argument('--kc-time-csv', default='data/projection/kc_estimates_time.csv',
                   help='parameter_fit --rolling-time 输出 CSV')
    p.add_argument('--top-n-industries', type=int, default=DEFAULT_TOP_N,
                   help='每个 asof_date 选 top-N industries')
    p.add_argument('--industry-selection', default='by_n_stocks',
                   choices=['by_n_stocks', 'by_c_over_k', 'by_k_over_c'],
                   help='排序标准')
    p.add_argument('--max-dates', type=int, default=DEFAULT_MAX_DATES,
                   help='最多取最近 N 个 asof_date(默认 12,避免动画过慢)')
    p.add_argument('--html-output', default=HTML_OUT)
    p.add_argument('--summary-output', default=SUMMARY_OUT)
    p.add_argument('--pairs-csv-output', default=PAIRS_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load
    kc_df = load_kc_time_series(args.kc_time_csv)
    if kc_df.empty:
        raise RuntimeError(f'{args.kc_time_csv} 过滤后为空 — 检查 status / n_valid_days')

    # 2. 取所有 unique asof_date,排序,截断到最近 max_dates
    all_dates = sorted(kc_df['asof_date'].unique().tolist())
    if len(all_dates) > args.max_dates:
        print(f'[v5.3] asof_date 共 {len(all_dates)} 个,截断到最近 {args.max_dates} 个')
        all_dates = all_dates[-args.max_dates:]

    # 3. 聚合
    per_date_dfs = aggregate_by_industry_per_date(kc_df, dates=all_dates, group_col='index_code')
    if not per_date_dfs:
        raise RuntimeError('聚合后为空')

    # 4. 选 top-N
    pairs = select_top_n_per_date(per_date_dfs, criterion=args.industry_selection, n=args.top_n_industries)
    if not pairs:
        raise RuntimeError('选不到任何 industry pair')

    # 5. omega_grid
    omega_grid = DEFAULT_OMEGA_GRID

    # 6. 写 3 输出
    build_animated_overlay_html(pairs, omega_grid, args.html_output)
    write_animated_summary_txt(pairs, all_dates, args.summary_output)
    write_animated_pairs_csv(pairs, args.pairs_csv_output)

    print(f'[v5.3] {len(pairs)} 个 (date, industry) 对已写入:')
    print(f'  - {args.html_output}')
    print(f'  - {args.summary_output}')
    print(f'  - {args.pairs_csv_output}')


if __name__ == '__main__':
    main()