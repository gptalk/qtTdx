"""v5.3 — Real SI Frequency Response 时序动画 overlay.

读 parameter_fit --rolling-time 输出 (kc_estimates_time.csv),按 asof_date 切片
+ 行业聚合 + top-N 选取,通过 plotly animation_frame 联动多帧 Bode overlay。
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

# v5.6 — matplotlib 静态 PNG 导出(必须在 import pyplot 前 use('Agg'))
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# v5.6 — regime 颜色字典(镜像 v5.5 _regime_color 闭包 dict)
REGIME_COLORS = {
    'overdamped':  '#2ca02c',   # 绿, Schur 内稳定
    'critical':    '#ff7f0e',   # 橙, Schur 边界
    'underdamped': '#d62728',   # 红, Schur 外共振
    'anti_damped': '#9467bd',   # 紫, 病态
}

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
from plotly.subplots import make_subplots

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
from backtrace.dynamics.dynamics_forced_response import classify_response_type


def build_animated_overlay_html(
    pairs_per_date: list,
    omega_grid: np.ndarray,
    output_path: str,
    title: str = 'Industry G(ω) Frequency Response — Time Series',
) -> None:
    """构建 plotly 动画 slider:每帧一个 asof_date,每帧 N × 2 条 industry Bode 曲线。

    v5.4 双子图:
        - 上子图 |H(jω)| dB vs ω
        - 下子图 ∠H(jω) degrees vs ω(共享 x 轴)

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

    dates = sorted(set(p[0] for p in pairs_per_date))
    initial_date = dates[0]

    # Phase 1: build initial-figure traces using go.Figure (one row = one subplot)
    # We use make_subplots(2, 1, shared_xaxes=True) for the dual-pane layout.
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=('|H(jω)| dB', '∠H(jω) deg'),
        vertical_spacing=0.10,
    )

    def _magnitude_db(p):
        return magnitude_phase(omega_grid * 1j, p[1], p[2])[0].tolist()

    def _phase_deg(p):
        return np.degrees(magnitude_phase(omega_grid * 1j, p[1], p[2])[1]).tolist()

    def _regime_color(k, c):
        """Map (k, c) to regime color hex per classify_response_type."""
        regime = classify_response_type(k, c)
        return {
            'overdamped':  '#2ca02c',   # 绿, Schur 内稳定
            'critical':    '#ff7f0e',   # 橙, Schur 边界
            'underdamped': '#d62728',   # 红, Schur 外共振
            'anti_damped': '#9467bd',   # 紫, 病态
        }.get(regime, '#7f7f7f')        # 灰 fallback (理论不应触发)

    # Initial-state traces (first date)
    for p in (p_ for p_ in pairs_per_date if p_[0] == initial_date):
        color = _regime_color(p[1], p[2])
        fig.add_trace(go.Scatter(x=omega_grid.tolist(), y=_magnitude_db(p),
                                 mode='lines', name=p[3], legendgroup=p[3],
                                 line=dict(color=color)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=omega_grid.tolist(), y=_phase_deg(p),
                                 mode='lines', name=p[3], legendgroup=p[3],
                                 showlegend=False,
                                 line=dict(color=color)),
                      row=2, col=1)

    # Phase 2: build frames — one frame per date, each frame has 2 × N traces
    frames = []
    for date in dates:
        frame_traces = []
        for p in (p_ for p_ in pairs_per_date if p_[0] == date):
            color = _regime_color(p[1], p[2])
            frame_traces.append(go.Scatter(x=omega_grid.tolist(), y=_magnitude_db(p),
                                           mode='lines', name=p[3], legendgroup=p[3],
                                           line=dict(color=color)))
            frame_traces.append(go.Scatter(x=omega_grid.tolist(), y=_phase_deg(p),
                                           mode='lines', name=p[3], legendgroup=p[3],
                                           showlegend=False,
                                           line=dict(color=color)))
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

    # Play/Pause buttons
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
        # Only the bottom subplot shows the x-axis title (shared axes)
        xaxis2_title='ω (rad/day)',
        yaxis_title='|H(jω)| dB',
        yaxis2_title='∠H(jω) deg',
        updatemenus=[dict(
            type='buttons', showactive=False, y=1.15, x=0.5, xanchor='center',
            buttons=[play_button, pause_button],
        )],
        sliders=[dict(active=0, steps=slider_steps, x=0.1, len=0.9, xanchor='left',
                      y=0, yanchor='top', currentvalue=dict(prefix='asof_date: ', visible=True))],
        height=700,  # taller to accommodate 2 subplots
    )

    # v5.5 color legend annotation (top-right)
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.99, y=1.08, xanchor='right', yanchor='top',
        showarrow=False,
        text=('颜色 = 阻尼 regime: '
              '<span style="color:#2ca02c">●</span> 过阻尼 (stable)  '
              '<span style="color:#ff7f0e">●</span> 临界 (critical)  '
              '<span style="color:#d62728">●</span> 欠阻尼 (resonance)  '
              '<span style="color:#9467bd">●</span> anti-damped'),
        align='left',
        font=dict(size=11),
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.write_html(output_path, include_plotlyjs='cdn')


def build_static_bode_grid(
    pairs_per_date: list,
    omega_grid: np.ndarray,
    output_path: str,
    title: str = 'Industry G(ω) Frequency Response — Static Grid',
    dpi: int = 100,
) -> None:
    """Render all dates' Bode curves as a 2D matplotlib grid (rows = dates, cols = |H| + ∠H).

    Args:
        pairs_per_date: [(asof_date, k̂, ĉ, label), ...] from select_top_n_per_date
        omega_grid: 共享 ω 网格
        output_path: PNG 输出路径
        title: figure 标题
        dpi: PNG 分辨率(默认 100)

    Raises:
        ValueError: pairs_per_date 为空
    """
    if not pairs_per_date:
        raise ValueError('pairs_per_date 为空,无法构建 static grid')

    dates = sorted(set(p[0] for p in pairs_per_date))
    n_rows = len(dates)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows), sharex=True, sharey='col')
    if n_rows == 1:
        axes = np.array([axes])  # 2D array for indexing

    def _mag_db(p):
        return magnitude_phase(omega_grid * 1j, p[1], p[2])[0].tolist()

    def _phase_deg(p):
        return np.degrees(magnitude_phase(omega_grid * 1j, p[1], p[2])[1]).tolist()

    def _color(k, c):
        return REGIME_COLORS.get(classify_response_type(k, c), '#7f7f7f')

    for i, date in enumerate(dates):
        ax_mag = axes[i, 0]
        ax_phase = axes[i, 1]
        for p in (p_ for p_ in pairs_per_date if p_[0] == date):
            color = _color(p[1], p[2])
            ax_mag.plot(omega_grid, _mag_db(p), color=color, label=p[3], linewidth=1.5)
            ax_phase.plot(omega_grid, _phase_deg(p), color=color, label=p[3], linewidth=1.5)
        ax_mag.set_ylabel('|H(jω)| dB' if i == 0 else '')
        ax_phase.set_ylabel('∠H(jω) deg' if i == 0 else '')
        ax_mag.set_title(f'{date}')
        ax_mag.grid(True, alpha=0.3)
        ax_phase.grid(True, alpha=0.3)
        if i == 0:
            ax_mag.legend(loc='upper right', fontsize=8)

    # Bottom row xlabel
    axes[-1, 0].set_xlabel('ω (rad/day)')
    axes[-1, 1].set_xlabel('ω (rad/day)')

    # I-1 fix: regime color legend (matplotlib equivalent of v5.5 plotly annotation)
    regime_patches = [
        mpatches.Patch(color='#2ca02c', label='overdamped (stable)'),
        mpatches.Patch(color='#ff7f0e', label='critical'),
        mpatches.Patch(color='#d62728', label='underdamped (resonance)'),
        mpatches.Patch(color='#9467bd', label='anti_damped'),
    ]
    fig.legend(handles=regime_patches, loc='upper center',
               bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False, fontsize=9)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


REGIME_ABBREV = {
    'overdamped':  'over',
    'critical':    'crit',
    'underdamped': 'under',
    'anti_damped': 'anti',
}


def build_regime_heatmap(
    pairs_per_date: list,
    output_path: str,
    title: str = 'Industry Regime Stability — Heatmap',
    dpi: int = 100,
) -> None:
    """Render regime for each (date, industry) as a 2D heatmap.

    Args:
        pairs_per_date: [(asof_date, k̂, �, label), ...] from select_top_n_per_date
        output_path: PNG 输出路径
        title: figure 标题
        dpi: PNG 分辨率(默认 100)

    Raises:
        ValueError: pairs_per_date 为空
    """
    if not pairs_per_date:
        raise ValueError('pairs_per_date 为空,无法构建 heatmap')

    dates = sorted(set(p[0] for p in pairs_per_date))
    industries = sorted(set(p[3] for p in pairs_per_date))
    n_rows = len(dates)
    n_cols = len(industries)

    # 索引 (date, industry) → (k̂, ĉ)
    pair_lookup = {}
    for p in pairs_per_date:
        pair_lookup[(p[0], p[3])] = (p[1], p[2])

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * n_cols), max(4, 0.6 * n_rows)))

    # 画 cell
    for i, date in enumerate(dates):
        for j, industry in enumerate(industries):
            k, c = pair_lookup.get((date, industry), (None, None))
            if k is None:
                color = '#7f7f7f'  # 灰 (无数据)
                text = '?'
            else:
                regime = classify_response_type(k, c)
                color = REGIME_COLORS.get(regime, '#7f7f7f')
                text = REGIME_ABBREV.get(regime, '?')

            rect = mpatches.Rectangle(
                (j, i), 1, 1,
                facecolor=color, edgecolor='white', linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, text,
                    ha='center', va='center',
                    fontsize=10, color='black')

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(industries, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(dates, fontsize=9)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()  # 日期早的在顶部
    ax.set_aspect('equal')

    # 顶部 4 色 legend (v5.6 I-1 模式)
    regime_patches = [
        mpatches.Patch(color='#2ca02c', label='overdamped (stable)'),
        mpatches.Patch(color='#ff7f0e', label='critical'),
        mpatches.Patch(color='#d62728', label='underdamped (resonance)'),
        mpatches.Patch(color='#9467bd', label='anti_damped'),
    ]
    fig.legend(handles=regime_patches, loc='upper center',
               bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False, fontsize=9)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


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
    p.add_argument(
        '--static-output',
        type=str,
        default='backtrace/outputs/dynsys_si_freq_response_static.png',
        help='PNG 静态网格输出路径',
    )
    p.add_argument(
        '--heatmap-output',
        type=str,
        default='backtrace/outputs/dynsys_regime_heatmap.png',
        help='Regime heatmap PNG 输出路径',
    )
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
    build_static_bode_grid(pairs, omega_grid, args.static_output)
    build_regime_heatmap(pairs, args.heatmap_output)
    print(f'[v5.7] regime heatmap 已写入 {args.heatmap_output}')

    print(f'[v5.3] {len(pairs)} 个 (date, industry) 对已写入:')
    print(f'  - {args.html_output}')
    print(f'  - {args.summary_output}')
    print(f'  - {args.pairs_csv_output}')
    print(f'[v5.6] 静态 PNG 已写入 {args.static_output}')


if __name__ == '__main__':
    main()